use std::collections::HashMap;
use std::fs;
use std::path::Path;

use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
pub struct Manifest {
    pub package: PackageInfo,
    #[serde(default)]
    pub dependencies: HashMap<String, Dependency>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct PackageInfo {
    pub name: String,
    pub version: String,
    #[serde(default = "default_edition")]
    pub edition: String,
    #[serde(default)]
    pub authors: Vec<String>,
    #[serde(default)]
    pub description: Option<String>,
}

fn default_edition() -> String {
    "2024".to_string()
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Dependency {
    pub version: String,
    #[serde(default)]
    pub path: Option<String>,
}

impl Manifest {
    pub fn load_from_file(path: &Path) -> Result<Manifest, Box<dyn std::error::Error>> {
        let content = fs::read_to_string(path)?;
        let manifest: Manifest = toml::from_str(&content)?;
        Ok(manifest)
    }

    pub fn save_to_file(&self, path: &Path) -> Result<(), Box<dyn std::error::Error>> {
        let content = toml::to_string_pretty(self)?;
        fs::write(path, content)?;
        Ok(())
    }

    pub fn new(name: &str) -> Manifest {
        Manifest {
            package: PackageInfo {
                name: name.to_string(),
                version: "0.1.0".to_string(),
                edition: default_edition(),
                authors: Vec::new(),
                description: None,
            },
            dependencies: HashMap::new(),
        }
    }
}
