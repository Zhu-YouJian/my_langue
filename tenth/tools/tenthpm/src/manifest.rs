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

#[derive(Debug, Serialize, Deserialize)]
pub struct Lockfile {
    pub version: u32,
    pub packages: Vec<LockPackage>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct LockPackage {
    pub name: String,
    pub version: String,
    pub source: Option<String>,
    pub checksum: Option<String>,
}

impl Lockfile {
    pub fn new() -> Self {
        Lockfile {
            version: 1,
            packages: Vec::new(),
        }
    }

    pub fn load_from_file(path: &Path) -> Result<Lockfile, Box<dyn std::error::Error>> {
        if !path.exists() {
            return Ok(Lockfile::new());
        }
        let content = fs::read_to_string(path)?;
        let lockfile: Lockfile = toml::from_str(&content)?;
        Ok(lockfile)
    }

    pub fn save_to_file(&self, path: &Path) -> Result<(), Box<dyn std::error::Error>> {
        let content = toml::to_string_pretty(self)?;
        fs::write(path, content)?;
        Ok(())
    }

    pub fn from_manifest(manifest: &Manifest) -> Lockfile {
        let packages: Vec<LockPackage> = manifest.dependencies.iter().map(|(name, dep)| {
            LockPackage {
                name: name.clone(),
                version: dep.version.clone(),
                source: dep.path.as_ref().map(|p| format!("path:{}", p)),
                checksum: None,
            }
        }).collect();
        Lockfile { version: 1, packages }
    }
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
