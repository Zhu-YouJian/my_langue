use std::fs;
use std::path::Path;

use crate::manifest::{Lockfile, Manifest};
use crate::resolver;

/// Remove a dependency from the project.
///
/// - Removes the entry from Tenth.toml
/// - Updates Tenth.lock (re-resolves the remaining graph, M4.1)
/// - For git deps: removes the cloned directory under deps/
/// - For path deps: does NOT delete the source (it's user's own code)
pub fn remove(package: &str) -> Result<(), Box<dyn std::error::Error>> {
    let manifest_path = Path::new("Tenth.toml");
    if !manifest_path.exists() {
        return Err("当前目录未找到 Tenth.toml".into());
    }

    let mut manifest = Manifest::load_from_file(manifest_path)?;

    let dep = manifest
        .dependencies
        .remove(package)
        .ok_or_else(|| format!("依赖 `{}` 不在 Tenth.toml 中", package))?;

    // M4.1：重新解析剩余依赖图；冲突 / 缺失响亮报错（不落盘，保持原子性）
    let resolution = resolver::resolve(&manifest, Path::new("."))
        .map_err(|e| -> Box<dyn std::error::Error> { e.into() })?;
    manifest.save_to_file(manifest_path)?;

    // Update lock file with the resolved graph
    let lock_path = Path::new("Tenth.lock");
    let lockfile = Lockfile::from_resolution(&resolution);
    let _ = lockfile.save_to_file(lock_path);

    // For git deps, remove the cloned directory
    if dep.is_git() {
        let deps_dir = Path::new("deps").join(package);
        if deps_dir.exists() {
            fs::remove_dir_all(&deps_dir)?;
            println!("Removed `deps/{}`", package);
        }
    }

    println!("Removed dependency `{}`", package);
    Ok(())
}
