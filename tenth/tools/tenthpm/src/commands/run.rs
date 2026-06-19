use std::path::Path;

use crate::engine;
use crate::manifest::Manifest;

/// Build and run the project: execute src/main.th in-process.
///
/// Uses the shared engine which sets up proper search paths for `use`
/// imports, including deps/ for dependencies.
pub fn run() -> Result<(), Box<dyn std::error::Error>> {
    let manifest_path = Path::new("Tenth.toml");
    if !manifest_path.exists() {
        return Err("当前目录未找到 Tenth.toml".into());
    }

    let manifest = Manifest::load_from_file(manifest_path)?;

    let main_path = Path::new("src/main.th");
    if !main_path.exists() {
        return Err("src/main.th 不存在".into());
    }

    println!(
        "Running `{}` v{} ...",
        manifest.package.name, manifest.package.version
    );

    engine::run_file(main_path).map_err(|e| -> Box<dyn std::error::Error> { e.into() })
}
