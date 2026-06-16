use std::fs;
use std::path::Path;
use std::process::Command;

use crate::manifest::{Dependency, Manifest};

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

pub fn add(package: &str, version: Option<&str>) -> Result<(), Box<dyn std::error::Error>> {
    let manifest_path = Path::new("Tenth.toml");
    if !manifest_path.exists() {
        return Err("Tenth.toml not found in current directory".into());
    }

    let mut manifest = Manifest::load_from_file(manifest_path)?;

    if is_git_url(package) {
        let package_name = extract_package_name(package)
            .ok_or("Could not extract package name from git URL")?;

        let deps_dir = Path::new("deps");
        if !deps_dir.exists() {
            fs::create_dir(deps_dir)?;
        }

        let target_dir = deps_dir.join(&package_name);
        if target_dir.exists() {
            return Err(format!(
                "Directory `deps/{}` already exists. Remove it first if you want to re-clone.",
                package_name
            )
            .into());
        }

        println!("Cloning `{}` into `deps/{}`...", package, package_name);

        let status = Command::new("git")
            .args(["clone", package, &format!("deps/{}", package_name)])
            .status()
            .map_err(|e| {
                if e.kind() == std::io::ErrorKind::NotFound {
                    "git is not installed or not found in PATH".to_string()
                } else {
                    format!("Failed to run git: {}", e)
                }
            })?;

        if !status.success() {
            return Err(format!(
                "git clone failed for `{}` (exit code: {:?})",
                package,
                status.code()
            )
            .into());
        }

        let dependency = Dependency {
            version: version.unwrap_or("*").to_string(),
            path: Some(format!("deps/{}", package_name)),
        };

        manifest
            .dependencies
            .insert(package_name.clone(), dependency);
        manifest.save_to_file(manifest_path)?;

        println!(
            "Added dependency `{}` from git URL `{}`",
            package_name, package
        );
    } else {
        let dep_version = version.unwrap_or("*").to_string();
        let dependency = Dependency {
            version: dep_version,
            path: None,
        };

        manifest.dependencies.insert(package.to_string(), dependency);
        manifest.save_to_file(manifest_path)?;

        println!("Added dependency `{}`", package);
    }

    Ok(())
}
