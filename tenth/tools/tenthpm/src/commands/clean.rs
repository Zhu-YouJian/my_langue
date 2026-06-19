use std::fs;
use std::path::Path;

/// Clean build artifacts: Tenth.lock, deps/ cache, and .wasm files.
///
/// Options:
///   --deps   Also remove deps/ directory (git dependencies)
///   --all    Remove everything including deps/ and .wasm files
pub fn clean(deps_too: bool, all: bool) -> Result<(), Box<dyn std::error::Error>> {
    let mut removed = 0;

    // Remove lock file
    let lock_path = Path::new("Tenth.lock");
    if lock_path.exists() {
        fs::remove_file(lock_path)?;
        println!("Removed Tenth.lock");
        removed += 1;
    }

    // Remove .wasm build artifacts
    let wasm_files = find_wasm_files(Path::new("."))?;
    for wasm in &wasm_files {
        fs::remove_file(wasm)?;
        println!("Removed {}", wasm.display());
        removed += 1;
    }

    // Remove deps/ if requested
    if deps_too || all {
        let deps_dir = Path::new("deps");
        if deps_dir.exists() {
            fs::remove_dir_all(deps_dir)?;
            println!("Removed deps/");
            removed += 1;
        }
    }

    // Remove target/ if exists (from cargo build)
    if all {
        let target_dir = Path::new("target");
        if target_dir.exists() {
            fs::remove_dir_all(target_dir)?;
            println!("Removed target/");
            removed += 1;
        }
    }

    if removed == 0 {
        println!("Nothing to clean.");
    } else {
        println!("Cleaned {} item(s).", removed);
    }

    Ok(())
}

fn find_wasm_files(dir: &Path) -> Result<Vec<std::path::PathBuf>, Box<dyn std::error::Error>> {
    let mut files = Vec::new();
    find_wasm_files_inner(dir, &mut files)?;
    Ok(files)
}

fn find_wasm_files_inner(
    dir: &Path,
    files: &mut Vec<std::path::PathBuf>,
) -> Result<(), Box<dyn std::error::Error>> {
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir() {
            // Skip hidden directories and target/
            let name = path.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default();
            if name.starts_with('.') || name == "target" {
                continue;
            }
            find_wasm_files_inner(&path, files)?;
        } else if path.extension().is_some_and(|ext| ext == "wasm") {
            files.push(path);
        }
    }
    Ok(())
}
