use std::path::Path;

use crate::manifest::Manifest;

/// List all dependencies in the project.
pub fn list() -> Result<(), Box<dyn std::error::Error>> {
    let manifest_path = Path::new("Tenth.toml");
    if !manifest_path.exists() {
        return Err("当前目录未找到 Tenth.toml".into());
    }

    let manifest = Manifest::load_from_file(manifest_path)?;

    if manifest.dependencies.is_empty() {
        println!("`{}` v{} has no dependencies.",
            manifest.package.name, manifest.package.version);
        return Ok(());
    }

    println!(
        "`{}` v{} dependencies:",
        manifest.package.name, manifest.package.version
    );
    println!();

    // Sort for stable output
    let mut deps: Vec<_> = manifest.dependencies.iter().collect();
    deps.sort_by(|a, b| a.0.cmp(b.0));

    // Compute column widths
    let name_width = deps.iter().map(|(n, _)| n.len()).max().unwrap_or(4).max(4);
    let ver_width = deps
        .iter()
        .map(|(_, d)| d.version.len())
        .max()
        .unwrap_or(7)
        .max(7);

    println!(
        "  {:<width_n$}  {:<width_v$}  Source",
        "Name",
        "Version",
        width_n = name_width,
        width_v = ver_width
    );
    println!(
        "  {}  {}  ------",
        "-".repeat(name_width),
        "-".repeat(ver_width)
    );

    for (name, dep) in &deps {
        println!(
            "  {:<width_n$}  {:<width_v$}  {}",
            name,
            dep.version,
            dep.source_display(),
            width_n = name_width,
            width_v = ver_width
        );
    }

    println!();
    println!("Total: {} dependencies", deps.len());
    Ok(())
}
