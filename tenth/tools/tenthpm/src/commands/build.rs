use std::fs;
use std::path::Path;

use crate::manifest::Manifest;

pub fn build() -> Result<(), Box<dyn std::error::Error>> {
    let manifest_path = Path::new("Tenth.toml");
    if !manifest_path.exists() {
        return Err("Tenth.toml not found in current directory".into());
    }

    let manifest = Manifest::load_from_file(manifest_path)?;
    println!("Building `{}` v{} ...", manifest.package.name, manifest.package.version);

    // Find all .th files in src/
    let src_dir = Path::new("src");
    if !src_dir.exists() {
        return Err("src/ directory not found".into());
    }

    let mut source_files = Vec::new();
    collect_th_files(src_dir, &mut source_files)?;

    if source_files.is_empty() {
        return Err("No .th source files found in src/".into());
    }

    for file in &source_files {
        println!("  Compiling {}", file.display());
    }

    // Create target directory
    fs::create_dir_all("target")?;

    // Simulate compilation
    println!("  Linking...");
    println!("Build finished successfully.");
    println!("Output: target/{}", manifest.package.name);

    Ok(())
}

fn collect_th_files(dir: &Path, files: &mut Vec<std::path::PathBuf>) -> Result<(), Box<dyn std::error::Error>> {
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir() {
            collect_th_files(&path, files)?;
        } else if path.extension().is_some_and(|ext| ext == "th") {
            files.push(path);
        }
    }
    Ok(())
}
