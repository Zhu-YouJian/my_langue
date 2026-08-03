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
    /// 该包的直接依赖名列表（M4.1 传递依赖锁定）。
    #[serde(default)]
    pub dependencies: Vec<String>,
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
                    checksum_of_file(&dep_path.join("Tenth.toml"))
                });

                LockPackage {
                    name: name.clone(),
                    version: dep.version.clone(),
                    source,
                    checksum,
                    dependencies: Vec::new(),
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
pub(crate) fn fnv1a_64(data: &[u8]) -> String {
    let mut hash: u64 = 0xcbf29ce484222325;
    for &byte in data {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    format!("{:016x}", hash)
}

/// 计算文件的 FNV-1a 64 位 checksum（hex 字符串）；文件不存在/不可读返回 `None`。
pub(crate) fn checksum_of_file(path: &Path) -> Option<String> {
    if !path.exists() {
        return None;
    }
    fs::read_to_string(path)
        .ok()
        .map(|content| fnv1a_64(content.as_bytes()))
}

// ── 包名 / git URL 安全校验（避免路径穿越）──────────────────────────────────
//
// 安全背景：早期 `extract_package_name` 直接取 URL 末段作为目录名，对
// `https://attacker.invalid/..` 这类输入会得到 `..`，进而在 install_global
// 中触发 `fs::remove_dir_all("~/.tenth/packages/..")` 递归删除用户主目录。
// 这一组函数集中校验包名合法性，所有调用方必须经过 `validate_package_name`
// 才能将包名用于文件系统操作。

/// 判断字符串是否为合法的 git URL。
///
/// L-4: 默认仅放行 `https://`。`http://`/`git://`/`ssh://` 与裸 `.git` 后缀的 URL
/// 在启用 ssh agent forwarding 的环境中存在被恶意服务器利用的风险
/// （参见 CVE-2017-1000117 等）。如需使用这些协议，调用方/用户必须显式设置
/// 环境变量 `TENTH_ALLOW_INSECURE_GIT=1` 表示自担风险。
///
/// 注意：即使放行 `.git` 后缀，[`safe_package_name_from_git`] 仍会调用
/// [`validate_package_name`] 拒绝 `..`、含路径分隔符等危险包名。
pub fn is_git_url(package: &str) -> bool {
    let allow_insecure = std::env::var("TENTH_ALLOW_INSECURE_GIT")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false);
    if allow_insecure {
        package.starts_with("https://")
            || package.starts_with("http://")
            || package.starts_with("git://")
            || package.starts_with("ssh://")
            || package.ends_with(".git")
    } else {
        // 安全默认：仅 https://。其余协议需用户显式 opt-in。
        package.starts_with("https://")
    }
}

/// 从 git URL 提取末段作为候选包名（不去除前导分隔符后的空段）。
/// **调用方必须再用 [`validate_package_name`] 校验返回值**。
pub fn extract_package_name(url: &str) -> Option<String> {
    let url = url.strip_suffix(".git").unwrap_or(url);
    // 去掉 query/fragment，避免 `https://x/y?..` 之类被当作包名
    let url = url.split(['?', '#']).next().unwrap_or(url);
    let name = url.rsplit('/').next()?;
    if name.is_empty() {
        None
    } else {
        Some(name.to_string())
    }
}

/// 校验包名是否可作为安全的目录名。拒绝一切可能导致路径穿越或目录覆盖的输入。
///
/// 规则：
/// - 非空，长度 ≤ 128
/// - 不含路径分隔符（`/`、`\`）、空字节、控制字符
/// - 不等于 `.` 或 `..`
/// - 不以 `.` 开头（隐藏目录）
/// - 不含 Windows 保留名（CON/PRN/AUX/NUL/COM*/LPT*，大小写不敏感）
/// - 仅允许字母、数字、`-`、`_`、`.`（中间的 `.`）
pub fn validate_package_name(name: &str) -> Result<(), String> {
    if name.is_empty() {
        return Err("包名不能为空".into());
    }
    if name.len() > 128 {
        return Err("包名过长（>128 字符）".into());
    }
    if name == "." || name == ".." {
        return Err(format!("包名 `{}` 保留，拒绝使用", name));
    }
    if name.starts_with('.') {
        return Err("包名不能以 `.` 开头".into());
    }
    if name.contains('\0') {
        return Err("包名不能含空字节".into());
    }
    if name.contains('/') || name.contains('\\') {
        return Err("包名不能含路径分隔符".into());
    }
    if name.chars().any(|c| c.is_control() || c.is_ascii_whitespace()) {
        return Err("包名不能含控制字符或空白".into());
    }
    // Windows 保留名（大小写不敏感）
    let upper = name.to_uppercase();
    let reserved = ["CON", "PRN", "AUX", "NUL"];
    if reserved.contains(&upper.as_str())
        || upper.starts_with("COM")
        || upper.starts_with("LPT")
    {
        return Err(format!("包名 `{}` 为 Windows 保留名", name));
    }
    // 仅允许 [A-Za-z0-9_\-.]，且 `.` 不在首尾
    if !name
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-' || c == '.')
    {
        return Err("包名仅允许字母、数字、`_`、`-`、`.`".into());
    }
    if name.ends_with('.') {
        return Err("包名不能以 `.` 结尾".into());
    }
    Ok(())
}

/// 提取并校验包名，一步到位。任何非法输入返回 `Err`。
pub fn safe_package_name_from_git(url: &str) -> Result<String, String> {
    if !is_git_url(url) {
        return Err(format!("`{}` 不是合法的 git URL", url));
    }
    let name = extract_package_name(url).ok_or("无法从 git URL 提取包名")?;
    validate_package_name(&name)?;
    Ok(name)
}

/// L-5: 禁用指定 git 仓库的 hooks 路径。clone 后立即调用，作为纵深防御。
///
/// 将 `core.hooksPath` 指向一个不会包含任何 hook 的"空"路径：
/// - Unix: `/dev/null`（设备文件，git 无法在其中查找 hook → 跳过）
/// - Windows: `nul`（同上，Windows 的空设备）
///
/// 失败时静默忽略（best-effort），因为：
/// 1. 现代 git（≥2.20）默认不克隆 origin 的 hooks；
/// 2. 调用方已通过 `--config protocol.file.allow=deny` 阻止 submodule 钩子注入；
/// 3. 此处仅作为最后一道防线，不构成安全边界。
pub fn disable_hooks(repo_dir: &str) {
    let dev_null = if cfg!(windows) { "nul" } else { "/dev/null" };
    let _ = std::process::Command::new("git")
        .args(["-C", repo_dir, "config", "--local", "core.hooksPath", dev_null])
        .status();
}

/// 校验 `target` 目录确实位于 `root` 目录之内，且二者都存在或可创建。
/// 用于在执行 `remove_dir_all` / `copy_dir` 等危险操作前做最后防线。
///
/// 实现使用 `canonicalize` 解析符号链接，避免 `deps/symlink-to-home` 之类逃逸。
/// `root` 不存在时返回 `Ok(())` 由调用方自行处理。
pub fn ensure_within(root: &Path, target: &Path) -> Result<(), String> {
    let root_canon = match fs::canonicalize(root) {
        Ok(p) => p,
        Err(_) => return Ok(()), // root 不存在，无边界可比
    };
    // target 可能尚不存在（首次 clone），先 canonicalize 其父目录
    let target_canon = if target.exists() {
        fs::canonicalize(target).map_err(|e| format!("无法解析目标路径: {}", e))?
    } else {
        let parent = target.parent().ok_or_else(|| "目标路径无父目录".to_string())?;
        let parent_canon = fs::canonicalize(parent)
            .map_err(|e| format!("无法解析目标父目录: {}", e))?;
        parent_canon.join(target.file_name().ok_or("目标路径无文件名")?)
    };
    if !target_canon.starts_with(&root_canon) {
        return Err(format!(
            "目标路径 `{}` 不在根目录 `{}` 之内（路径穿越被拒绝）",
            target_canon.display(),
            root_canon.display()
        ));
    }
    Ok(())
}

/// 在执行 `remove_dir_all` 之前的最终安全闸门。拒绝删除系统关键目录、
/// 空字符串、用户主目录等。即便前置校验失误，这层也能兜底。
pub fn safe_to_remove_dir(path: &Path) -> Result<(), String> {
    let path_str = path.to_string_lossy();
    if path_str.is_empty() {
        return Err("拒绝删除空路径".into());
    }
    // 拒绝删除根、用户主目录、常见系统目录
    let home = std::env::var("HOME")
        .or_else(|_| std::env::var("USERPROFILE"))
        .unwrap_or_default();
    let dangerous: [&str; 4] = ["/", "\\", "C:\\", "C:/"];
    if dangerous.iter().any(|d| path_str.as_ref() == *d) || (!home.is_empty() && path_str == home) {
        return Err(format!("拒绝删除关键目录 `{}`", path_str));
    }
    // 必须经过 canonicalize，确保 `~/foo/..` 之类被解析
    let canon = fs::canonicalize(path).map_err(|e| format!("无法解析路径: {}", e))?;
    let canon_str = canon.to_string_lossy();
    if canon_str == "/" || canon_str == "\\" || canon_str.ends_with(":\\") {
        return Err(format!("拒绝删除根目录 `{}`", canon_str));
    }
    if !home.is_empty() && canon_str == home {
        return Err("拒绝删除用户主目录".into());
    }
    // 拒绝删除 .ssh / .aws 等敏感目录
    if canon_str.ends_with("/.ssh") || canon_str.ends_with("\\.ssh")
        || canon_str.ends_with("/.aws") || canon_str.ends_with("\\.aws")
        || canon_str.ends_with("/.config") || canon_str.ends_with("\\.config")
    {
        return Err(format!("拒绝删除敏感目录 `{}`", canon_str));
    }
    Ok(())
}
