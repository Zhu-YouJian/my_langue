use std::fs;
use std::io::Write;
use std::path::Path;

use crate::engine;
use crate::manifest::Manifest;

/// Publish the project: validate, compile-check, and package into a
/// .tenthpkg archive (a simple tar-like format).
///
/// The archive format is:
/// ```
/// TENTHPKG\0          (8-byte magic)
/// <manifest_len:u32>  (little-endian)
/// <manifest_json>     (Tenth.toml content)
/// <file_count:u32>    (little-endian)
/// For each file:
///   <path_len:u32>    (little-endian)
///   <path_bytes>      (UTF-8 path, relative to project root)
///   <data_len:u32>    (little-endian)
///   <data_bytes>      (file content)
/// ```
pub fn publish() -> Result<(), Box<dyn std::error::Error>> {
    let manifest_path = Path::new("Tenth.toml");
    if !manifest_path.exists() {
        return Err("当前目录未找到 Tenth.toml".into());
    }

    let manifest = Manifest::load_from_file(manifest_path)?;

    // Validate for publishing
    manifest.validate(true).map_err(|e| -> Box<dyn std::error::Error> { e.into() })?;

    println!(
        "Publishing `{}` v{} ...",
        manifest.package.name, manifest.package.version
    );

    // Step 1: Validate — compile-check all source files
    println!("  Validating package...");
    let src_dir = Path::new("src");
    if !src_dir.exists() {
        return Err("src/ 目录不存在".into());
    }

    let th_files = engine::collect_th_files(src_dir)?;
    if th_files.is_empty() {
        return Err("src/ 下没有 .th 文件".into());
    }

    for file in &th_files {
        engine::check_file(file).map_err(|e| -> Box<dyn std::error::Error> { e.into() })?;
    }
    println!("  {} source file(s) validated.", th_files.len());

    // Step 2: Collect files to package
    let mut files_to_pack: Vec<(String, Vec<u8>)> = Vec::new();

    // Add Tenth.toml
    let manifest_content = fs::read(manifest_path)?;
    files_to_pack.push(("Tenth.toml".to_string(), manifest_content));

    // Add all .th files under src/
    for file in &th_files {
        let rel = file.strip_prefix(".").unwrap_or(file);
        let rel_str = rel.to_string_lossy().replace('\\', "/");
        let content = fs::read(file)?;
        files_to_pack.push((rel_str, content));
    }

    // Add README.md if exists
    let readme = Path::new("README.md");
    if readme.exists() {
        let content = fs::read(readme)?;
        files_to_pack.push(("README.md".to_string(), content));
    }

    // Step 3: Write the package archive
    let archive_name = format!(
        "{}-{}.tenthpkg",
        manifest.package.name, manifest.package.version
    );
    let archive_path = Path::new(&archive_name);

    let mut buf = Vec::new();

    // Magic
    buf.extend_from_slice(b"TENTHPKG\0");

    // Manifest content
    let manifest_bytes = fs::read_to_string(manifest_path)?;
    let manifest_bytes = manifest_bytes.as_bytes();
    buf.extend_from_slice(&(manifest_bytes.len() as u32).to_le_bytes());
    buf.extend_from_slice(manifest_bytes);

    // File count
    buf.extend_from_slice(&(files_to_pack.len() as u32).to_le_bytes());

    // Each file
    for (path, data) in &files_to_pack {
        let path_bytes = path.as_bytes();
        buf.extend_from_slice(&(path_bytes.len() as u32).to_le_bytes());
        buf.extend_from_slice(path_bytes);
        buf.extend_from_slice(&(data.len() as u32).to_le_bytes());
        buf.extend_from_slice(data);
    }

    let mut file = fs::File::create(archive_path)?;
    file.write_all(&buf)?;

    println!(
        "  Packaged {} file(s) into {} ({} bytes)",
        files_to_pack.len(),
        archive_name,
        buf.len()
    );
    println!(
        "  Published `{}` v{} successfully!",
        manifest.package.name, manifest.package.version
    );
    println!();
    println!("Note: Remote registry is not yet implemented.");
    println!("      The package archive has been created locally.");
    println!("      To distribute, share the .tenthpkg file or push to a git repository.");

    Ok(())
}
