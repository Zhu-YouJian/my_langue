use std::fs;
use std::path::Path;

use crate::engine;
use crate::manifest::Manifest;
use crate::pkg;
use crate::resolver;

/// Publish the project: validate (incl. dependency resolution), compile-check,
/// and package into a `.tenthpkg` archive. Optionally publish to a local
/// registry directory (`--registry <dir>`).
///
/// The archive format is:
/// ```
/// TENTHPKG\0          (8-byte magic)
/// <manifest_len:u32>  (little-endian)
/// <manifest_toml>     (Tenth.toml content)
/// <file_count:u32>    (little-endian)
/// For each file:
///   <path_len:u32>    (little-endian)
///   <path_bytes>      (UTF-8 path, relative to project root)
///   <data_len:u32>    (little-endian)
///   <data_bytes>      (file content)
/// ```
pub fn publish(registry: Option<&str>) -> Result<(), Box<dyn std::error::Error>> {
    let manifest_path = Path::new("Tenth.toml");
    if !manifest_path.exists() {
        return Err("当前目录未找到 Tenth.toml".into());
    }

    let manifest = Manifest::load_from_file(manifest_path)?;

    // Validate for publishing
    manifest
        .validate(true)
        .map_err(|e| -> Box<dyn std::error::Error> { e.into() })?;

    println!(
        "Publishing `{}` v{} ...",
        manifest.package.name, manifest.package.version
    );

    // Step 0: 依赖解析（M4.1）——冲突/缺失依赖在发布前响亮报错
    let resolution = resolver::resolve(&manifest, Path::new("."))
        .map_err(|e| -> Box<dyn std::error::Error> { e.into() })?;
    if !resolution.packages.is_empty() {
        println!(
            "  Resolved {} dependency package(s).",
            resolution.packages.len()
        );
    }

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

    pkg::write_archive(&manifest, &files_to_pack, archive_path)?;
    let size = fs::metadata(archive_path)
        .map(|m| m.len())
        .unwrap_or(0);

    println!(
        "  Packaged {} file(s) into {} ({} bytes)",
        files_to_pack.len(),
        archive_name,
        size
    );

    // Step 4: 可选——发布到本地 registry 目录
    if let Some(reg) = registry {
        let reg_dir = Path::new(reg);
        fs::create_dir_all(reg_dir).map_err(|e| -> Box<dyn std::error::Error> {
            format!("创建 registry 目录 {} 失败: {}", reg_dir.display(), e).into()
        })?;
        let target = reg_dir.join(&archive_name);
        fs::copy(archive_path, &target).map_err(|e| -> Box<dyn std::error::Error> {
            format!("复制归档到 registry 失败: {}", e).into()
        })?;
        println!("  Published to local registry: {}", target.display());
    }

    println!(
        "  Published `{}` v{} successfully!",
        manifest.package.name, manifest.package.version
    );
    println!();
    println!("Note: Remote registry is not yet implemented.");
    println!("      The package archive has been created locally.");
    println!("      To distribute, share the .tenthpkg file, publish to a");
    println!("      local registry dir (`publish --registry <dir>`), or push to a git repository.");

    Ok(())
}
