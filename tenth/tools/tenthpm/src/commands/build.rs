use std::path::Path;

use crate::engine;
use crate::manifest::{Lockfile, Manifest};

/// Build the project: compile-check all .th files under src/.
///
/// This runs the full lex → parse → lower pipeline in-process with proper
/// search paths (including deps/ for dependencies). It does not execute
/// the code — use `tenthpm run` for that.
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

    // Update lock file
    let lock_path = Path::new("Tenth.lock");
    let lockfile = Lockfile::from_manifest(&manifest);
    lockfile.save_to_file(lock_path)?;

    println!("Build finished successfully.");
    Ok(())
}
