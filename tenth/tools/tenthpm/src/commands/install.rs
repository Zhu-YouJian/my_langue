use std::fs;
use std::path::Path;
use std::process::Command;

use crate::manifest::{Dependency, Lockfile, Manifest};

/// Install a package from a git URL or local path into deps/ and add it
/// to the project's Tenth.toml.
///
/// This is similar to `add` but is the primary way to fetch a package
/// that you want to use as a dependency. The difference from `add`:
/// - `add` modifies Tenth.toml (and clones git deps)
/// - `install` does the same but is the "fetch" mental model
///
/// In practice both do the same thing for now. `install` also supports
/// installing without a project (global install to ~/.tenth/packages/).
pub fn install(package: &str) -> Result<(), Box<dyn std::error::Error>> {
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

fn install_local(package: &str) -> Result<(), Box<dyn std::error::Error>> {
    let manifest_path = Path::new("Tenth.toml");
    let mut manifest = Manifest::load_from_file(manifest_path)?;

    if is_git_url(package) {
        let package_name = extract_package_name(package)
            .ok_or("无法从 git URL 提取包名")?;

        let deps_dir = Path::new("deps");
        if !deps_dir.exists() {
            fs::create_dir(deps_dir)?;
        }

        let target_dir = deps_dir.join(&package_name);
        if target_dir.exists() {
            println!("Updating `{}` in deps/{}...", package, package_name);
            let _ = Command::new("git")
                .args(["-C", &format!("deps/{}", package_name), "pull"])
                .status();
        } else {
            println!("Installing `{}` into deps/{}...", package, package_name);
            let status = Command::new("git")
                .args(["clone", package, &format!("deps/{}", package_name)])
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
        }

        let dependency = Dependency {
            version: "*".to_string(),
            path: None,
            git: Some(package.to_string()),
        };

        manifest.dependencies.insert(package_name.clone(), dependency);
        manifest.save_to_file(manifest_path)?;

        let lock_path = Path::new("Tenth.lock");
        let lockfile = Lockfile::from_manifest(&manifest);
        let _ = lockfile.save_to_file(lock_path);

        println!("Installed `{}` successfully!", package_name);
    } else if Path::new(package).exists() {
        // Local path install — copy into deps/
        let dep_path = Path::new(package);
        let package_name = dep_path
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .ok_or("无法从路径提取包名")?;

        let deps_dir = Path::new("deps");
        if !deps_dir.exists() {
            fs::create_dir(deps_dir)?;
        }

        let target_dir = deps_dir.join(&package_name);
        if target_dir.exists() {
            fs::remove_dir_all(&target_dir)?;
        }
        copy_dir(dep_path, &target_dir)?;

        let dependency = Dependency {
            version: "*".to_string(),
            path: Some(format!("deps/{}", package_name)),
            git: None,
        };

        manifest.dependencies.insert(package_name.clone(), dependency);
        manifest.save_to_file(manifest_path)?;

        let lock_path = Path::new("Tenth.lock");
        let lockfile = Lockfile::from_manifest(&manifest);
        let _ = lockfile.save_to_file(lock_path);

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
        let package_name = extract_package_name(package)
            .ok_or("无法从 git URL 提取包名")?;

        let target_dir = global_dir.join(&package_name);
        if target_dir.exists() {
            fs::remove_dir_all(&target_dir)?;
        }

        println!("Installing `{}` globally...", package);
        let status = Command::new("git")
            .args(["clone", package, target_dir.to_str().unwrap()])
            .status()
            .map_err(|e| format!("运行 git 失败: {}", e))?;

        if !status.success() {
            return Err(format!("git clone 失败 (退出码: {:?})", status.code()).into());
        }

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

fn is_git_url(package: &str) -> bool {
    package.starts_with("http://")
        || package.starts_with("https://")
        || package.starts_with("git://")
        || package.starts_with("ssh://")
        || package.ends_with(".git")
}

fn extract_package_name(url: &str) -> Option<String> {
    let url = url.strip_suffix(".git").unwrap_or(url);
    let name = url.rsplit('/').next()?;
    if name.is_empty() {
        None
    } else {
        Some(name.to_string())
    }
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
