use std::fs;
use std::path::{Path, PathBuf};

use crate::manifest::{
    ensure_within, is_git_url, safe_package_name_from_git, safe_to_remove_dir, Dependency,
    Lockfile, Manifest,
};
use crate::resolver;
use crate::version::Version;

/// 保存 manifest 并依据解析结果写入锁文件（M4.1）。
///
/// 先做依赖解析（传递依赖 + 冲突检测）；失败时**响亮报错且不落盘**
/// （保持原子性，避免留下半成品 manifest/锁文件状态）。
fn save_and_lock(
    manifest: &Manifest,
    manifest_path: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    let resolution = resolver::resolve(manifest, Path::new("."))
        .map_err(|e| -> Box<dyn std::error::Error> { e.into() })?;
    manifest.save_to_file(manifest_path)?;
    let lock_path = Path::new("Tenth.lock");
    let lockfile = Lockfile::from_resolution(&resolution);
    lockfile.save_to_file(lock_path)?;
    Ok(())
}

/// Install a package from a git URL, local path, `.tenthpkg` file, or a local
/// registry directory into deps/ and add it to the project's Tenth.toml.
///
/// This is similar to `add` but is the primary way to fetch a package
/// that you want to use as a dependency. The difference from `add`:
/// - `add` modifies Tenth.toml (and clones git deps)
/// - `install` does the same but is the "fetch" mental model
///
/// In practice both do the same thing for now. `install` also supports
/// installing without a project (global install to ~/.tenth/packages/).
///
/// M4.1 新增：
/// - `install <file>.tenthpkg` —— 从本地归档安装（发布→安装闭环）
/// - `install <name> --registry <dir>` —— 从本地 registry 目录安装
pub fn install(package: &str, registry: Option<&str>) -> Result<(), Box<dyn std::error::Error>> {
    // 1. .tenthpkg 文件安装（发布→安装闭环）
    if package.ends_with(".tenthpkg") {
        let p = Path::new(package);
        if p.exists() {
            return install_from_pkg(p);
        }
        return Err(format!("找不到 .tenthpkg 文件: {}", package).into());
    }

    // 2. 从本地 registry 目录安装
    if let Some(reg) = registry {
        return install_from_registry(package, reg);
    }

    let manifest_path = Path::new("Tenth.toml");

    if manifest_path.exists() {
        // Project-local install
        install_local(package)?;
    } else {
        // Global install (no project context)
        install_global(package)?;
    }

    Ok(())
}

/// 从本地 `.tenthpkg` 归档安装：解包到 deps/<name>/ 并登记为 path 依赖。
fn install_from_pkg(pkg_path: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let archive = crate::pkg::read_archive(pkg_path)
        .map_err(|e| -> Box<dyn std::error::Error> { e.into() })?;
    // read_archive 已校验包名合法
    let package_name = archive.manifest.package.name.clone();
    let package_version = archive.manifest.package.version.clone();

    let manifest_path = Path::new("Tenth.toml");
    if !manifest_path.exists() {
        return Err("当前目录未找到 Tenth.toml — 请在项目目录内安装包".into());
    }
    let mut manifest = Manifest::load_from_file(manifest_path)?;

    // 解包到 deps/<name>/
    let deps_dir = Path::new("deps");
    if !deps_dir.exists() {
        fs::create_dir(deps_dir)?;
    }
    let target_dir = deps_dir.join(&package_name);
    ensure_within(deps_dir, &target_dir).map_err(|e| -> Box<dyn std::error::Error> {
        e.into()
    })?;
    if target_dir.exists() {
        safe_to_remove_dir(&target_dir)?;
        fs::remove_dir_all(&target_dir)?;
    }
    for (rel, content) in &archive.files {
        let dest = target_dir.join(rel);
        if let Some(parent) = dest.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(dest, content)?;
    }

    let dependency = Dependency {
        version: package_version.clone(),
        path: Some(format!("deps/{}", package_name)),
        git: None,
    };
    manifest.dependencies.insert(package_name.clone(), dependency);
    save_and_lock(&manifest, manifest_path)?;

    println!(
        "Installed `{}` v{} from {}!",
        package_name,
        package_version,
        pkg_path.display()
    );
    Ok(())
}

/// 从本地 registry 目录安装：扫描 `*.tenthpkg`，取匹配包名的最高版本。
fn install_from_registry(name: &str, registry: &str) -> Result<(), Box<dyn std::error::Error>> {
    // 安全：包名必须合法
    crate::manifest::validate_package_name(name)?;

    let reg_dir = Path::new(registry);
    if !reg_dir.is_dir() {
        return Err(format!("registry 目录不存在: {}", reg_dir.display()).into());
    }

    let mut best: Option<(Version, PathBuf)> = None;
    for entry in fs::read_dir(reg_dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_file() && path.extension().is_some_and(|e| e == "tenthpkg") {
            let archive = match crate::pkg::read_archive(&path) {
                Ok(a) => a,
                Err(_) => continue, // 损坏的归档跳过（registry 目录里可能有无关文件）
            };
            if archive.manifest.package.name != name {
                continue;
            }
            if let Some(v) = Version::parse(&archive.manifest.package.version) {
                let is_better = match &best {
                    Some((bv, _)) => v > *bv,
                    None => true,
                };
                if is_better {
                    best = Some((v, path));
                }
            }
        }
    }

    let (_, pkg_path) = best.ok_or_else(|| {
        format!(
            "本地 registry `{}` 中未找到包 `{}`（缺失依赖必须响亮报错）",
            reg_dir.display(),
            name
        )
    })?;

    install_from_pkg(&pkg_path)
}

fn install_local(package: &str) -> Result<(), Box<dyn std::error::Error>> {
    let manifest_path = Path::new("Tenth.toml");
    let mut manifest = Manifest::load_from_file(manifest_path)?;

    if is_git_url(package) {
        // 安全：拒绝 `https://attacker.invalid/..` 之类路径穿越输入
        let package_name = safe_package_name_from_git(package)?;

        let deps_dir = Path::new("deps");
        if !deps_dir.exists() {
            fs::create_dir(deps_dir)?;
        }

        let target_dir = deps_dir.join(&package_name);
        // 安全闸门：确保 target_dir 始终位于 deps/ 之内
        ensure_within(deps_dir, &target_dir).map_err(|e| -> Box<dyn std::error::Error> {
            e.into()
        })?;
        if target_dir.exists() {
            println!("Updating `{}` in deps/{}...", package, package_name);
            let _ = std::process::Command::new("git")
                .args(["-C", &format!("deps/{}", package_name), "pull"])
                .status();
        } else {
            println!("Installing `{}` into deps/{}...", package, package_name);
            // L-5: 克隆时禁用 file:// 和 git:// 协议（防 submodule 钩子注入），
            // 克隆后立即将 core.hooksPath 指向空设备，让所有 hook 失效。
            let target = format!("deps/{}", package_name);
            let status = std::process::Command::new("git")
                .args([
                    "clone",
                    "--config", "protocol.file.allow=deny",
                    "--config", "protocol.git.allow=deny",
                    package,
                    &target,
                ])
                .status()
                .map_err(|e| {
                    if e.kind() == std::io::ErrorKind::NotFound {
                        "git 未安装或不在 PATH 中".to_string()
                    } else {
                        format!("运行 git 失败: {}", e)
                    }
                })?;

            if !status.success() {
                return Err(format!(
                    "git clone 失败 (退出码: {:?})",
                    status.code()
                )
                .into());
            }
            // 纵深防御：禁用 cloned 仓库的 hooks 路径
            crate::manifest::disable_hooks(&target);
        }

        let dependency = Dependency {
            version: "*".to_string(),
            path: None,
            git: Some(package.to_string()),
        };

        manifest.dependencies.insert(package_name.clone(), dependency);
        save_and_lock(&manifest, manifest_path)?;

        println!("Installed `{}` successfully!", package_name);
    } else if Path::new(package).exists() {
        // Local path install — copy into deps/
        let dep_path = Path::new(package);
        let package_name = dep_path
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .ok_or("无法从路径提取包名")?;
        // 安全：本地路径依赖也要校验包名，避免 `deps/../` 之类
        crate::manifest::validate_package_name(&package_name)?;

        let deps_dir = Path::new("deps");
        if !deps_dir.exists() {
            fs::create_dir(deps_dir)?;
        }

        let target_dir = deps_dir.join(&package_name);
        ensure_within(deps_dir, &target_dir).map_err(|e| -> Box<dyn std::error::Error> {
            e.into()
        })?;
        if target_dir.exists() {
            safe_to_remove_dir(&target_dir)?;
            fs::remove_dir_all(&target_dir)?;
        }
        copy_dir(dep_path, &target_dir)?;

        let dependency = Dependency {
            version: "*".to_string(),
            path: Some(format!("deps/{}", package_name)),
            git: None,
        };

        manifest.dependencies.insert(package_name.clone(), dependency);
        save_and_lock(&manifest, manifest_path)?;

        println!("Installed `{}` from local path!", package_name);
    } else {
        return Err(format!(
            "包 `{}` 不是 git URL、本地路径，且远程注册中心尚未实现",
            package
        )
        .into());
    }

    Ok(())
}

fn install_global(package: &str) -> Result<(), Box<dyn std::error::Error>> {
    let home = std::env::var("HOME")
        .or_else(|_| std::env::var("USERPROFILE"))
        .map_err(|_| "无法确定用户主目录 (HOME/USERPROFILE)")?;

    let global_dir = Path::new(&home).join(".tenth").join("packages");
    fs::create_dir_all(&global_dir)?;

    if is_git_url(package) {
        // 安全：拒绝路径穿越输入
        let package_name = safe_package_name_from_git(package)?;

        let target_dir = global_dir.join(&package_name);
        ensure_within(&global_dir, &target_dir).map_err(|e| -> Box<dyn std::error::Error> {
            e.into()
        })?;
        if target_dir.exists() {
            // 安全闸门：删除前必须通过校验（防止 ~/.tenth/packages/.. 之类）
            safe_to_remove_dir(&target_dir)?;
            fs::remove_dir_all(&target_dir)?;
        }

        println!("Installing `{}` globally...", package);
        // L-5: 克隆时禁用 file:// 和 git:// 协议，克隆后禁用 hooks 路径。
        let target_str = target_dir.to_str().unwrap();
        let status = std::process::Command::new("git")
            .args([
                "clone",
                "--config", "protocol.file.allow=deny",
                "--config", "protocol.git.allow=deny",
                package,
                target_str,
            ])
            .status()
            .map_err(|e| format!("运行 git 失败: {}", e))?;

        if !status.success() {
            return Err(format!("git clone 失败 (退出码: {:?})", status.code()).into());
        }
        // 纵深防御：禁用 cloned 仓库的 hooks 路径
        crate::manifest::disable_hooks(target_str);

        println!("Installed `{}` to {}", package_name, target_dir.display());
    } else {
        return Err(format!(
            "全局安装仅支持 git URL。包 `{}` 无法识别。",
            package
        )
        .into());
    }

    Ok(())
}

/// Recursively copy a directory.
fn copy_dir(src: &Path, dst: &Path) -> Result<(), Box<dyn std::error::Error>> {
    fs::create_dir_all(dst)?;
    for entry in fs::read_dir(src)? {
        let entry = entry?;
        let path = entry.path();
        let target = dst.join(entry.file_name());
        if path.is_dir() {
            // Skip hidden directories and target/
            let name = path.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default();
            if name.starts_with('.') || name == "target" {
                continue;
            }
            copy_dir(&path, &target)?;
        } else if path.is_file() {
            // Skip hidden files and lock files
            let name = path.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default();
            if name.starts_with('.') || name == "Tenth.lock" {
                continue;
            }
            fs::copy(&path, &target)?;
        }
    }
    Ok(())
}
