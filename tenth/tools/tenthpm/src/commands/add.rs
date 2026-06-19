use std::fs;
use std::path::Path;
use std::process::Command;

use crate::manifest::{Dependency, Lockfile, Manifest};

/// Check if a string looks like a git URL.
fn is_git_url(package: &str) -> bool {
    package.starts_with("http://")
        || package.starts_with("https://")
        || package.starts_with("git://")
        || package.starts_with("ssh://")
        || package.ends_with(".git")
}

/// Extract the package name from a git URL (last path component, minus .git).
fn extract_package_name(url: &str) -> Option<String> {
    let url = url.strip_suffix(".git").unwrap_or(url);
    let name = url.rsplit('/').next()?;
    if name.is_empty() {
        None
    } else {
        Some(name.to_string())
    }
}

/// Check if a string looks like a local path.
fn is_local_path(package: &str) -> bool {
    Path::new(package).exists()
}

/// Add a dependency to the project.
///
/// Supports three forms:
/// 1. `tenthpm add <name>` — registry dependency (version optional, defaults to "*")
/// 2. `tenthpm add <path>` — local path dependency (path must exist)
/// 3. `tenthpm add <git-url>` — git dependency (cloned into deps/)
pub fn add(package: &str, version: Option<&str>) -> Result<(), Box<dyn std::error::Error>> {
    let manifest_path = Path::new("Tenth.toml");
    if !manifest_path.exists() {
        return Err("当前目录未找到 Tenth.toml".into());
    }

    let mut manifest = Manifest::load_from_file(manifest_path)?;

    if is_git_url(package) {
        add_git_dependency(&mut manifest, manifest_path, package, version)?;
    } else if is_local_path(package) {
        add_path_dependency(&mut manifest, manifest_path, package, version)?;
    } else {
        add_registry_dependency(&mut manifest, manifest_path, package, version)?;
    }

    Ok(())
}

fn add_git_dependency(
    manifest: &mut Manifest,
    manifest_path: &Path,
    url: &str,
    version: Option<&str>,
) -> Result<(), Box<dyn std::error::Error>> {
    let package_name = extract_package_name(url)
        .ok_or("无法从 git URL 提取包名")?;

    let deps_dir = Path::new("deps");
    if !deps_dir.exists() {
        fs::create_dir(deps_dir)?;
    }

    let target_dir = deps_dir.join(&package_name);
    if target_dir.exists() {
        // If already exists, pull updates instead of erroring
        println!("Updating `{}` in deps/{}...", url, package_name);
        let status = Command::new("git")
            .args(["-C", &format!("deps/{}", package_name), "pull"])
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
                "git pull 失败 (退出码: {:?})",
                status.code()
            )
            .into());
        }
    } else {
        println!("Cloning `{}` into deps/{}...", url, package_name);
        let status = Command::new("git")
            .args(["clone", url, &format!("deps/{}", package_name)])
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
        version: version.unwrap_or("*").to_string(),
        path: None,
        git: Some(url.to_string()),
    };

    manifest
        .dependencies
        .insert(package_name.clone(), dependency);
    manifest.save_to_file(manifest_path)?;

    // Update lock file
    let lock_path = Path::new("Tenth.lock");
    let lockfile = Lockfile::from_manifest(manifest);
    let _ = lockfile.save_to_file(lock_path);

    println!("Added dependency `{}` (git: {})", package_name, url);
    Ok(())
}

fn add_path_dependency(
    manifest: &mut Manifest,
    manifest_path: &Path,
    path: &str,
    version: Option<&str>,
) -> Result<(), Box<dyn std::error::Error>> {
    let dep_path = Path::new(path);
    let dep_manifest_path = dep_path.join("Tenth.toml");

    // Try to read the dependency's name from its manifest
    let package_name = if dep_manifest_path.exists() {
        let dep_manifest = Manifest::load_from_file(&dep_manifest_path)?;
        dep_manifest.package.name
    } else {
        // Fall back to directory name
        dep_path
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .ok_or("无法从路径提取包名")?
    };

    let dependency = Dependency {
        version: version.unwrap_or("*").to_string(),
        path: Some(path.to_string()),
        git: None,
    };

    manifest.dependencies.insert(package_name.clone(), dependency);
    manifest.save_to_file(manifest_path)?;

    // Update lock file
    let lock_path = Path::new("Tenth.lock");
    let lockfile = Lockfile::from_manifest(manifest);
    let _ = lockfile.save_to_file(lock_path);

    println!("Added dependency `{}` (path: {})", package_name, path);
    Ok(())
}

fn add_registry_dependency(
    manifest: &mut Manifest,
    manifest_path: &Path,
    name: &str,
    version: Option<&str>,
) -> Result<(), Box<dyn std::error::Error>> {
    let dep_version = version.unwrap_or("*").to_string();
    let dependency = Dependency {
        version: dep_version,
        path: None,
        git: None,
    };

    manifest.dependencies.insert(name.to_string(), dependency);
    manifest.save_to_file(manifest_path)?;

    // Update lock file
    let lock_path = Path::new("Tenth.lock");
    let lockfile = Lockfile::from_manifest(manifest);
    let _ = lockfile.save_to_file(lock_path);

    println!("Added dependency `{}`", name);
    Ok(())
}
