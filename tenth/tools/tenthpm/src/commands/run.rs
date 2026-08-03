use std::path::Path;

use crate::engine;
use crate::manifest::Manifest;
use crate::resolver;

/// Build and run the project: execute src/main.th in-process.
///
/// Uses the shared engine which sets up proper search paths for `use`
/// imports, including deps/ for dependencies.
///
/// M4.1：运行前先做依赖解析（传递依赖 + 冲突检测），冲突 / 缺失依赖
/// 响亮报错，绝不静默运行在错误版本上。
pub fn run() -> Result<(), Box<dyn std::error::Error>> {
    let manifest_path = Path::new("Tenth.toml");
    if !manifest_path.exists() {
        return Err("当前目录未找到 Tenth.toml".into());
    }

    let manifest = Manifest::load_from_file(manifest_path)?;

    // M4.1：依赖解析检查
    resolver::resolve(&manifest, Path::new("."))
        .map_err(|e| -> Box<dyn std::error::Error> { e.into() })?;

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
