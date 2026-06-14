use std::path::Path;

use crate::manifest::Manifest;

pub fn run() -> Result<(), Box<dyn std::error::Error>> {
    let manifest_path = Path::new("Tenth.toml");
    if !manifest_path.exists() {
        return Err("Tenth.toml not found in current directory".into());
    }

    let manifest = Manifest::load_from_file(manifest_path)?;

    let main_path = Path::new("src/main.th");
    if !main_path.exists() {
        return Err("src/main.th not found".into());
    }

    println!("Compiling `{}` v{} ...", manifest.package.name, manifest.package.version);
    println!("  Compiling src/main.th");
    println!("  Linking...");
    println!("Running `{}` ...", manifest.package.name);
    // Simulate execution
    println!("Hello from {}!", manifest.package.name);

    Ok(())
}
