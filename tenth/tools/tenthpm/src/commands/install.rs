use std::fs;
use std::path::Path;

pub fn install(package: &str) -> Result<(), Box<dyn std::error::Error>> {
    let install_dir = Path::new(".tenth/packages").join(package);

    println!("Installing `{}` ...", package);
    println!("  Downloading from registry...");

    // Simulate download and install
    fs::create_dir_all(&install_dir)?;

    println!("  Installing to {} ...", install_dir.display());
    println!("  Installed `{}` successfully!", package);

    Ok(())
}
