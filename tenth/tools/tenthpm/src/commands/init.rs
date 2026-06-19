use std::fs;
use std::path::Path;

use crate::manifest::Manifest;

pub fn init(name: Option<&str>) -> Result<(), Box<dyn std::error::Error>> {
    let project_name = match name {
        Some(n) => n.to_string(),
        None => std::env::current_dir()?
            .file_name()
            .unwrap_or_default()
            .to_string_lossy()
            .to_string(),
    };

    let project_dir = Path::new(&project_name);

    // Create directory structure
    fs::create_dir_all(project_dir.join("src"))?;
    fs::create_dir_all(project_dir.join("tests"))?;

    // Generate Tenth.toml
    let manifest = Manifest::new(&project_name);
    manifest.save_to_file(&project_dir.join("Tenth.toml"))?;

    // Generate src/main.th
    let main_th = format!("fn main() {{ println(\"Hello from {}!\"); }}", project_name);
    fs::write(project_dir.join("src/main.th"), main_th)?;

    println!("Created Tenth project `{}`", project_name);
    println!("  {}/", project_name);
    println!("  ├── Tenth.toml");
    println!("  ├── src/");
    println!("  │   └── main.th");
    println!("  └── tests/");

    Ok(())
}
