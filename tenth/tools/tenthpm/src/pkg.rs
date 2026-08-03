//! `.tenthpkg` 归档读写（M4.1 发布流程）。
//!
//! 归档格式（与 publish.rs 历史格式一致）：
//! ```text
//! TENTHPKG\0          (9-byte magic: "TENTHPKG" + NUL)
//! <manifest_len:u32>  (little-endian)
//! <manifest_bytes>    (Tenth.toml 内容，TOML)
//! <file_count:u32>    (little-endian)
//! 对每个文件:
//!   <path_len:u32>    (little-endian)
//!   <path_bytes>      (UTF-8 相对路径)
//!   <data_len:u32>    (little-endian)
//!   <data_bytes>      (文件内容)
//! ```
//!
//! 安全（护城河红线）：读取时校验 magic、长度边界（拒绝损坏归档导致的 panic/越界）、
//! 包名合法性与归档内路径合法性（拒绝 `..`、绝对路径、盘符——防路径穿越）。

use std::fs;
use std::path::Path;

use crate::manifest::{validate_package_name, Manifest};

/// 归档魔数（9 字节：`TENTHPKG` + NUL，与历史 publish.rs 格式一致）。
pub const MAGIC: &[u8; 9] = b"TENTHPKG\0";

/// 一个已解析的 `.tenthpkg` 归档。
#[derive(Debug)]
pub struct PkgArchive {
    pub manifest: Manifest,
    /// (相对路径, 内容)
    pub files: Vec<(String, Vec<u8>)>,
}

/// 写入 `.tenthpkg` 归档到 `path`。
pub fn write_archive(
    manifest: &Manifest,
    files: &[(String, Vec<u8>)],
    path: &Path,
) -> Result<(), String> {
    let mut buf: Vec<u8> = Vec::new();
    buf.extend_from_slice(MAGIC);

    let manifest_str = toml::to_string(manifest)
        .map_err(|e| format!("序列化 manifest 失败: {}", e))?;
    let manifest_bytes = manifest_str.as_bytes();
    buf.extend_from_slice(&(manifest_bytes.len() as u32).to_le_bytes());
    buf.extend_from_slice(manifest_bytes);

    buf.extend_from_slice(&(files.len() as u32).to_le_bytes());
    for (p, data) in files {
        let pb = p.as_bytes();
        buf.extend_from_slice(&(pb.len() as u32).to_le_bytes());
        buf.extend_from_slice(pb);
        buf.extend_from_slice(&(data.len() as u32).to_le_bytes());
        buf.extend_from_slice(data);
    }

    fs::write(path, buf).map_err(|e| format!("写入归档 {} 失败: {}", path.display(), e))
}

/// 从文件读取 `.tenthpkg` 归档并校验。
pub fn read_archive(path: &Path) -> Result<PkgArchive, String> {
    let data = fs::read(path).map_err(|e| format!("读取归档 {} 失败: {}", path.display(), e))?;
    parse_archive(&data)
}

/// 解析归档字节流（带完整边界与安全校验）。
pub fn parse_archive(data: &[u8]) -> Result<PkgArchive, String> {
    if data.len() < 9 || &data[..9] != MAGIC {
        return Err("不是有效的 .tenthpkg 文件（magic 不匹配）".to_string());
    }
    let mut pos = 9usize;

    let mlen = read_u32(data, &mut pos)? as usize;
    if pos + mlen > data.len() {
        return Err("归档损坏：manifest 长度越界".to_string());
    }
    let manifest_bytes = &data[pos..pos + mlen];
    pos += mlen;
    let manifest_str = std::str::from_utf8(manifest_bytes)
        .map_err(|_| "归档 manifest 不是 UTF-8".to_string())?;
    let manifest: Manifest = toml::from_str(manifest_str)
        .map_err(|e| format!("归档 manifest 解析失败: {}", e))?;
    // 安全：包名必须合法，否则拒绝解包
    validate_package_name(&manifest.package.name)?;

    let count = read_u32(data, &mut pos)? as usize;
    let mut files: Vec<(String, Vec<u8>)> = Vec::new();
    for _ in 0..count {
        let plen = read_u32(data, &mut pos)? as usize;
        if pos + plen > data.len() {
            return Err("归档损坏：路径长度越界".to_string());
        }
        let path_bytes = &data[pos..pos + plen];
        pos += plen;
        let path_str = std::str::from_utf8(path_bytes)
            .map_err(|_| "归档路径不是 UTF-8".to_string())?;
        // 安全：归档内路径必须相对且无 `..`，防解包路径穿越
        validate_archive_path(path_str)?;

        let dlen = read_u32(data, &mut pos)? as usize;
        if pos + dlen > data.len() {
            return Err("归档损坏：内容长度越界".to_string());
        }
        let content = data[pos..pos + dlen].to_vec();
        pos += dlen;

        files.push((path_str.to_string(), content));
    }

    Ok(PkgArchive { manifest, files })
}

fn read_u32(data: &[u8], pos: &mut usize) -> Result<u32, String> {
    if *pos + 4 > data.len() {
        return Err("归档损坏：长度字段越界".to_string());
    }
    let v = u32::from_le_bytes([
        data[*pos],
        data[*pos + 1],
        data[*pos + 2],
        data[*pos + 3],
    ]);
    *pos += 4;
    Ok(v)
}

/// 校验归档内路径：必须相对、不含 `..`/`.` 段、非盘符路径。
fn validate_archive_path(p: &str) -> Result<(), String> {
    if p.is_empty() {
        return Err("归档内空路径".to_string());
    }
    if Path::new(p).is_absolute() {
        return Err(format!("归档路径必须相对: {}", p));
    }
    let norm = p.replace('\\', "/");
    if norm.starts_with('/') {
        return Err(format!("归档路径不能以 / 开头: {}", p));
    }
    // 盘符路径（C:/ 或 C:\）
    if norm.len() >= 2 && norm.as_bytes()[1] == b':' {
        return Err(format!("归档路径不能含盘符: {}", p));
    }
    for seg in norm.split('/') {
        if seg == ".." || seg == "." {
            return Err(format!("归档路径含非法段 `{}`，拒绝解包: {}", seg, p));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::manifest::Manifest;

    fn sample_manifest() -> Manifest {
        let mut m = Manifest::new("sample");
        m.package.description = Some("desc".to_string());
        m.package.license = Some("MIT".to_string());
        m
    }

    #[test]
    fn test_archive_roundtrip() {
        let dir = std::env::temp_dir().join("tenthpm_pkg_roundtrip");
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();

        let m = sample_manifest();
        let files: Vec<(String, Vec<u8>)> = vec![
            ("src/main.th".to_string(), b"fn main() {}".to_vec()),
            ("Tenth.toml".to_string(), toml::to_string(&m).unwrap().into_bytes()),
        ];
        let p = dir.join("sample-0.1.0.tenthpkg");
        write_archive(&m, &files, &p).unwrap();

        let a = read_archive(&p).unwrap();
        assert_eq!(a.manifest.package.name, "sample");
        assert_eq!(a.files.len(), 2);
        assert_eq!(a.files[0].0, "src/main.th");
        assert_eq!(a.files[0].1, b"fn main() {}");
    }

    #[test]
    fn test_parse_archive_rejects_bad_magic() {
        let err = parse_archive(b"NOTAPKG\0....").unwrap_err();
        assert!(err.contains("magic"));
    }

    #[test]
    fn test_parse_archive_rejects_truncated() {
        // 构造一个 magic 正确但长度越界的归档
        let mut data: Vec<u8> = Vec::new();
        data.extend_from_slice(MAGIC);
        data.extend_from_slice(&1000u32.to_le_bytes()); // manifest_len=1000，但后面没数据
        let err = parse_archive(&data).unwrap_err();
        assert!(err.contains("越界"));
    }

    #[test]
    fn test_validate_archive_path_safety() {
        assert!(validate_archive_path("src/main.th").is_ok());
        assert!(validate_archive_path("a/b/c.th").is_ok());
        assert!(validate_archive_path("README.md").is_ok());
        // 路径穿越
        assert!(validate_archive_path("../evil.th").is_err());
        assert!(validate_archive_path("a/../../evil.th").is_err());
        assert!(validate_archive_path("./evil.th").is_err());
        assert!(validate_archive_path("a/./evil.th").is_err());
        // 绝对路径 / 盘符
        assert!(validate_archive_path("/etc/passwd").is_err());
        assert!(validate_archive_path("C:/Windows/evil").is_err());
        assert!(validate_archive_path("C:\\Windows\\evil").is_err());
        // 空
        assert!(validate_archive_path("").is_err());
    }

    #[test]
    fn test_parse_archive_rejects_traversal_path() {
        // 构造含 `../evil.th` 的归档，应拒绝解包
        let mut data: Vec<u8> = Vec::new();
        data.extend_from_slice(MAGIC);
        let m = sample_manifest();
        let ms = toml::to_string(&m).unwrap();
        data.extend_from_slice(&(ms.len() as u32).to_le_bytes());
        data.extend_from_slice(ms.as_bytes());
        data.extend_from_slice(&1u32.to_le_bytes());
        let p = "../evil.th";
        data.extend_from_slice(&(p.len() as u32).to_le_bytes());
        data.extend_from_slice(p.as_bytes());
        let c = b"x";
        data.extend_from_slice(&(c.len() as u32).to_le_bytes());
        data.extend_from_slice(c);

        let err = parse_archive(&data).unwrap_err();
        assert!(err.contains("非法段"), "应拒绝路径穿越，实际: {}", err);
    }

    #[test]
    fn test_parse_archive_rejects_bad_pkg_name() {
        // 包名含路径分隔符 → 拒绝
        let mut data: Vec<u8> = Vec::new();
        data.extend_from_slice(MAGIC);
        let ms = "[package]\nname = \"a/b\"\nversion = \"0.1.0\"\nedition = \"2024\"\nauthors = []\n[dependencies]\n";
        data.extend_from_slice(&(ms.len() as u32).to_le_bytes());
        data.extend_from_slice(ms.as_bytes());
        data.extend_from_slice(&0u32.to_le_bytes());

        let err = parse_archive(&data).unwrap_err();
        assert!(err.contains("包名"), "应拒绝非法包名，实际: {}", err);
    }
}
