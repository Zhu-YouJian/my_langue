use std::path::Path;

use crate::engine;
use crate::manifest::{Lockfile, Manifest};
use crate::resolver;

/// Build the project: resolve dependencies, then compile-check all .th files
/// under src/.
///
/// This runs the full lex → parse → lower pipeline in-process with proper
/// search paths (including deps/ for dependencies). It does not execute
/// the code — use `tenthpm run` for that.
///
/// M4.1：构建前先做依赖解析（传递依赖 + 版本冲突检测）。冲突 / 缺失依赖
/// 会**响亮报错**（绝不静默使用错误版本），并把解析结果（直接 + 传递）
/// 锁定进 Tenth.lock。
pub fn build() -> Result<(), Box<dyn std::error::Error>> {
    let manifest_path = Path::new("Tenth.toml");
    if !manifest_path.exists() {
        return Err("当前目录未找到 Tenth.toml".into());
    }

    let manifest = Manifest::load_from_file(manifest_path)?;
    manifest.validate(false).map_err(|e| -> Box<dyn std::error::Error> { e.into() })?;

    println!(
        "Building `{}` v{} ...",
        manifest.package.name, manifest.package.version
    );

    // M4.1：依赖解析（传递依赖 + 冲突检测），失败响亮报错
    let resolution = resolver::resolve(&manifest, Path::new("."))
        .map_err(|e| -> Box<dyn std::error::Error> { e.into() })?;
    if !resolution.packages.is_empty() {
        println!(
            "  Resolved {} package(s): {}",
            resolution.packages.len(),
            resolution.packages.iter().map(|p| p.name.as_str()).collect::<Vec<_>>().join(", ")
        );
    }

    let src_dir = Path::new("src");
    if !src_dir.exists() {
        return Err("src/ 目录不存在".into());
    }

    // Collect all .th files under src/
    let th_files = engine::collect_th_files(src_dir)?;

    if th_files.is_empty() {
        return Err("src/ 下没有 .th 文件".into());
    }

    let mut had_error = false;
    for file in &th_files {
        let rel = file.strip_prefix(".").unwrap_or(file);
        print!("  Compiling {} ... ", rel.display());
        match engine::check_file(file) {
            Ok(()) => println!("ok"),
            Err(e) => {
                println!("FAILED");
                eprintln!("    {}", e);
                had_error = true;
            }
        }
    }

    if had_error {
        return Err("编译失败".into());
    }

    // Update lock file with the resolved graph (direct + transitive)
    let lock_path = Path::new("Tenth.lock");
    let lockfile = Lockfile::from_resolution(&resolution);
    lockfile.save_to_file(lock_path)?;

    println!("Build finished successfully.");
    Ok(())
}
