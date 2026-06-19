use std::fs;
use std::path::Path;

use crate::manifest::{Lockfile, Manifest};

/// Remove a dependency from the project.
///
/// - Removes the entry from Tenth.toml
/// - Updates Tenth.lock
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

    manifest.save_to_file(manifest_path)?;

    // Update lock file
    let lock_path = Path::new("Tenth.lock");
    let lockfile = Lockfile::from_manifest(&manifest);
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
