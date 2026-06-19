use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

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
    #[serde(default)]
    pub license: Option<String>,
}

fn default_edition() -> String {
    "2024".to_string()
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Dependency {
    pub version: String,
    #[serde(default)]
    pub path: Option<String>,
    #[serde(default)]
    pub git: Option<String>,
}

impl Dependency {
    /// Returns the source location as a human-readable string.
    pub fn source_display(&self) -> String {
        if let Some(g) = &self.git {
            format!("git:{}", g)
        } else if let Some(p) = &self.path {
            format!("path:{}", p)
        } else {
            format!("registry:{}", self.version)
        }
    }

    /// Returns true if this is a local path dependency.
    pub fn is_path(&self) -> bool {
        self.path.is_some()
    }

    /// Returns true if this is a git dependency.
    pub fn is_git(&self) -> bool {
        self.git.is_some()
    }
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

#[allow(dead_code)]
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

    /// Build a lockfile from the manifest, computing checksums for path deps.
    pub fn from_manifest(manifest: &Manifest) -> Lockfile {
        let packages: Vec<LockPackage> = manifest
            .dependencies
            .iter()
            .map(|(name, dep)| {
                let source = if dep.is_git() {
                    dep.git.clone().map(|g| format!("git:{}", g))
                } else if dep.is_path() {
                    dep.path.clone().map(|p| format!("path:{}", p))
                } else {
                    Some(format!("registry:{}", dep.version))
                };

                let checksum = dep.path.as_ref().and_then(|p| {
                    let dep_path = Path::new(p);
                    let dep_manifest = dep_path.join("Tenth.toml");
                    if dep_manifest.exists() {
                        fs::read_to_string(&dep_manifest)
                            .ok()
                            .map(|content| fnv1a_64(content.as_bytes()))
                    } else {
                        None
                    }
                });

                LockPackage {
                    name: name.clone(),
                    version: dep.version.clone(),
                    source,
                    checksum,
                }
            })
            .collect();
        Lockfile {
            version: 1,
            packages,
        }
    }

    /// Check if the lockfile is up to date with the manifest.
    pub fn is_up_to_date(&self, manifest: &Manifest) -> bool {
        if self.packages.len() != manifest.dependencies.len() {
            return false;
        }
        for lock_pkg in &self.packages {
            match manifest.dependencies.get(&lock_pkg.name) {
                Some(dep) => {
                    if dep.version != lock_pkg.version {
                        return false;
                    }
                }
                None => return false,
            }
        }
        true
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
                license: None,
            },
            dependencies: HashMap::new(),
        }
    }

    /// Find the project root by searching for Tenth.toml in the current
    /// directory and its ancestors.
    #[allow(dead_code)]
    pub fn find_project_root() -> Result<PathBuf, String> {
        let cwd = std::env::current_dir().map_err(|e| format!("无法获取当前目录: {}", e))?;
        let mut current = cwd.as_path();
        loop {
            let manifest_path = current.join("Tenth.toml");
            if manifest_path.exists() {
                return Ok(current.to_path_buf());
            }
            match current.parent() {
                Some(parent) => current = parent,
                None => return Err("未找到 Tenth.toml — 不在 Tenth 项目中".into()),
            }
        }
    }

    /// Validate the manifest for completeness.
    pub fn validate(&self, for_publish: bool) -> Result<(), String> {
        if self.package.name.is_empty() {
            return Err("包名不能为空".into());
        }
        if self.package.version.is_empty() {
            return Err("版本号不能为空".into());
        }
        // Validate version format (semver-ish: X.Y.Z)
        if !is_valid_version(&self.package.version) {
            return Err(format!(
                "版本号 '{}' 格式无效，应为 X.Y.Z 格式",
                self.package.version
            ));
        }
        if for_publish {
            if self.package.description.is_none() {
                return Err("发布包需要 description 字段".into());
            }
            if self.package.license.is_none() {
                return Err("发布包需要 license 字段".into());
            }
        }
        Ok(())
    }
}

/// Check if a version string matches X.Y.Z format.
fn is_valid_version(v: &str) -> bool {
    let parts: Vec<&str> = v.split('.').collect();
    if parts.len() != 3 {
        return false;
    }
    parts.iter().all(|p| p.parse::<u32>().is_ok())
}

/// FNV-1a 64-bit hash, returned as hex string.
fn fnv1a_64(data: &[u8]) -> String {
    let mut hash: u64 = 0xcbf29ce484222325;
    for &byte in data {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    format!("{:016x}", hash)
}
