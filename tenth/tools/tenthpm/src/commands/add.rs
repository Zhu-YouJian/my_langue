use std::path::Path;

use crate::manifest::{Dependency, Manifest};

pub fn add(package: &str, version: Option<&str>) -> Result<(), Box<dyn std::error::Error>> {
    let manifest_path = Path::new("Tenth.toml");
    if !manifest_path.exists() {
        return Err("Tenth.toml not found in current directory".into());
    }

    let mut manifest = Manifest::load_from_file(manifest_path)?;

    let dep_version = version.unwrap_or("*").to_string();
    let dependency = Dependency {
        version: dep_version,
        path: None,
    };

    manifest.dependencies.insert(package.to_string(), dependency);
    manifest.save_to_file(manifest_path)?;

    println!("Added dependency `{}`", package);

    Ok(())
}
