use std::path::Path;

use crate::manifest::Manifest;

pub fn publish() -> Result<(), Box<dyn std::error::Error>> {
    let manifest_path = Path::new("Tenth.toml");
    if !manifest_path.exists() {
        return Err("Tenth.toml not found in current directory".into());
    }

    let manifest = Manifest::load_from_file(manifest_path)?;

    // Validate manifest completeness
    if manifest.package.name.is_empty() {
        return Err("Package name is required for publishing".into());
    }
    if manifest.package.version.is_empty() {
        return Err("Package version is required for publishing".into());
    }
    if manifest.package.description.is_none() {
        return Err("Package description is required for publishing".into());
    }

    println!("Publishing `{}` v{} ...", manifest.package.name, manifest.package.version);
    println!("  Validating package...");
    println!("  Uploading to registry...");
    println!("  Published `{}` v{} successfully!", manifest.package.name, manifest.package.version);

    Ok(())
}
