//! 依赖解析器（M4.1）：传递依赖解析 + 版本冲突检测 + 循环检测。
//!
//! 设计原则（护城河红线）：
//! - 依赖冲突 / 依赖缺失必须**响亮报错**（返回 `Err`），绝不静默选择错误版本；
//! - registry 依赖（无本地副本）无法核对具体版本 → 用约束区间可满足性检测冲突；
//! - path / git 依赖有本地副本 → 以实际版本核对全部约束。
//!
//! 解析结果（`Resolution`）供 `Tenth.lock` 锁定：记录**全部**已解析包
//! （直接 + 传递），每个包含版本、来源、checksum 与依赖列表。

use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};

use crate::manifest::{validate_package_name, Dependency, Lockfile, LockPackage, Manifest};
use crate::version::{reqs_conflict, Version, VersionReq};

/// 单个已解析的依赖包（含传递依赖）。
#[derive(Debug, Clone)]
pub struct ResolvedPackage {
    pub name: String,
    pub version: String,
    pub source: Option<String>,
    pub checksum: Option<String>,
    pub dependencies: Vec<String>,
}

/// 一次依赖解析的完整结果。
#[derive(Debug)]
pub struct Resolution {
    pub packages: Vec<ResolvedPackage>,
}

impl Resolution {
    /// 包名列表（稳定排序）。
    #[allow(dead_code)] // 供集成测试（tests/resolver_tests.rs）使用
    pub fn names(&self) -> Vec<&str> {
        self.packages.iter().map(|p| p.name.as_str()).collect()
    }
}

/// 一条依赖边：`from` 声明了对 `name` 的依赖，约束为 `req`。
#[derive(Debug, Clone)]
struct Edge {
    from: String,
    name: String,
    req: VersionReq,
    req_str: String,
    source: Option<String>,
}

/// 解析项目依赖（含传递依赖）。
///
/// `project_root` 用于解析相对路径依赖与 `deps/`（git 依赖位置）。
pub fn resolve(manifest: &Manifest, project_root: &Path) -> Result<Resolution, String> {
    let mut edges: Vec<Edge> = Vec::new();
    // name -> [(具体版本, source, checksum)]（来自本地副本的可用版本）
    let mut available: HashMap<String, Vec<(Version, String, Option<String>)>> = HashMap::new();
    // 已展开的本地 manifest 路径（规范化），避免重复读取同一文件；
    // 注意：按路径而非包名去重——同名不同目录（如 libv1/libv2）都要贡献版本，
    // 否则可能静默漏掉唯一满足约束的版本（护城河红线）。
    let mut visited: HashSet<PathBuf> = HashSet::new();
    let mut stack: Vec<String> = Vec::new();

    for (name, dep) in &manifest.dependencies {
        visit(
            "root",
            name,
            dep,
            project_root,
            &mut edges,
            &mut available,
            &mut visited,
            &mut stack,
        )?;
    }

    // 按包名聚合，稳定输出
    let mut names: Vec<String> = edges
        .iter()
        .map(|e| e.name.clone())
        .collect::<HashSet<_>>()
        .into_iter()
        .collect();
    names.sort();

    let mut packages: Vec<ResolvedPackage> = Vec::new();
    for name in names {
        let name_edges: Vec<&Edge> = edges.iter().filter(|e| e.name == name).collect();
        let reqs: Vec<VersionReq> = name_edges.iter().map(|e| e.req.clone()).collect();
        let avail = available.get(&name).cloned().unwrap_or_default();

        let (chosen_version, source, checksum) = if avail.is_empty() {
            // registry-only：无本地副本，只能核对约束可满足性
            if reqs_conflict(&reqs) {
                return Err(conflict_message(&name, &name_edges, &avail));
            }
            // 记录约束串（多个不同约束时以 " || " 连接），锁定解析意图
            let mut req_strs: Vec<String> = name_edges.iter().map(|e| e.req_str.clone()).collect();
            req_strs.sort();
            req_strs.dedup();
            let version = req_strs.join(" || ");
            let source = Some(format!("registry:{version}"));
            (version, source, None)
        } else {
            // 有本地副本：以具体版本核对全部约束
            let mut candidates: Vec<(Version, String, Option<String>)> = avail
                .iter()
                .filter(|(v, _, _)| reqs.iter().all(|r| r.matches(v)))
                .cloned()
                .collect();
            if candidates.is_empty() {
                return Err(conflict_message(&name, &name_edges, &avail));
            }
            // 取最高兼容版本
            candidates.sort_by(|a, b| b.0.cmp(&a.0));
            let (v, s, c) = candidates.remove(0);
            (v.to_string(), Some(s), c)
        };

        // 该包的依赖 = 以它为 from 的边的目标集
        let mut deps: Vec<String> = edges
            .iter()
            .filter(|e| e.from == name)
            .map(|e| e.name.clone())
            .collect::<HashSet<_>>()
            .into_iter()
            .collect();
        deps.sort();

        packages.push(ResolvedPackage {
            name: name.clone(),
            version: chosen_version,
            source,
            checksum,
            dependencies: deps,
        });
    }

    packages.sort_by(|a, b| a.name.cmp(&b.name));
    Ok(Resolution { packages })
}

/// 深度优先遍历单个依赖边。
#[allow(clippy::too_many_arguments)]
fn visit(
    from: &str,
    name: &str,
    dep: &Dependency,
    root: &Path,
    edges: &mut Vec<Edge>,
    available: &mut HashMap<String, Vec<(Version, String, Option<String>)>>,
    visited: &mut HashSet<PathBuf>,
    stack: &mut Vec<String>,
) -> Result<(), String> {
    let req = VersionReq::parse(&dep.version)
        .map_err(|e| format!("依赖 `{}` 的版本约束无效: {}", name, e))?;

    // 确定本地副本位置（path 依赖 / git 依赖）；registry 依赖无本地副本
    let local_dir: Option<PathBuf> = if let Some(p) = &dep.path {
        let dir = root.join(p);
        if !dir.exists() {
            return Err(format!(
                "依赖 `{}` 的路径不存在: {}（缺失依赖必须响亮报错）",
                name,
                dir.display()
            ));
        }
        Some(dir)
    } else if dep.is_git() {
        validate_package_name(name)
            .map_err(|e| format!("依赖 `{}` 包名非法: {}", name, e))?;
        let dir = root.join("deps").join(name);
        if !dir.exists() {
            return Err(format!(
                "git 依赖 `{}` 未获取（deps/{} 不存在），请先运行 `tenthpm install`",
                name, name
            ));
        }
        Some(dir)
    } else {
        None
    };

    // 循环检测（当前 DFS 路径上再次出现同名包）
    if stack.iter().any(|s| s == name) {
        let mut chain = stack.clone();
        chain.push(name.to_string());
        return Err(format!("检测到循环依赖: {}", chain.join(" → ")));
    }

    edges.push(Edge {
        from: from.to_string(),
        name: name.to_string(),
        req,
        req_str: dep.version.clone(),
        source: Some(dep.source_display()),
    });

    // 读取本地副本（按 manifest 路径去重；同一包名的多条边/多个目录各自记录）
    if let Some(dir) = &local_dir {
        let mpath = dir.join("Tenth.toml");
        if mpath.exists() {
            let key = fs::canonicalize(&mpath).unwrap_or_else(|_| mpath.clone());
            if !visited.contains(&key) {
                let sub = Manifest::load_from_file(&mpath)
                    .map_err(|e| format!("依赖 `{}` 的 Tenth.toml 无法解析: {}", name, e))?;
                let ver = Version::parse(&sub.package.version).ok_or_else(|| {
                    format!(
                        "依赖 `{}` 的版本号 `{}` 无效（应为 X.Y.Z）",
                        name, sub.package.version
                    )
                })?;
                let checksum = crate::manifest::checksum_of_file(&mpath);
                available
                    .entry(name.to_string())
                    .or_default()
                    .push((ver, dep.source_display(), checksum));
                visited.insert(key);
                stack.push(name.to_string());
                for (n2, d2) in &sub.dependencies {
                    visit(
                        name, n2, d2, root, edges, available, visited, stack,
                    )?;
                }
                stack.pop();
            }
            // 无 Tenth.toml 的 path 依赖：视为叶子（无版本信息，无法核对约束）
        }
    }

    Ok(())
}

/// 构造冲突错误消息（响亮、带依赖链）。
fn conflict_message(
    name: &str,
    edges: &[&Edge],
    available: &[(Version, String, Option<String>)],
) -> String {
    let mut msg = format!("版本冲突：包 `{}` 的依赖约束无法同时满足：\n", name);
    for e in edges {
        msg.push_str(&format!(
            "  {} → {} 需要 {} (来源: {})\n",
            e.from, e.name, e.req_str, e.source.as_deref().unwrap_or("?")
        ));
    }
    if !available.is_empty() {
        msg.push_str("  本地可用版本: ");
        let vers: Vec<String> = available
            .iter()
            .map(|(v, s, _)| format!("{} ({})", v, s))
            .collect();
        msg.push_str(&vers.join(", "));
        msg.push('\n');
    }
    msg.push_str("请调整依赖版本约束，使所有约束可同时满足。");
    msg
}

impl Lockfile {
    /// 由解析结果构建锁文件（锁定直接 + 传递依赖）。
    pub fn from_resolution(res: &Resolution) -> Lockfile {
        let packages: Vec<LockPackage> = res
            .packages
            .iter()
            .map(|p| LockPackage {
                name: p.name.clone(),
                version: p.version.clone(),
                source: p.source.clone(),
                checksum: p.checksum.clone(),
                dependencies: p.dependencies.clone(),
            })
            .collect();
        Lockfile {
            version: 1,
            packages,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn write(path: &std::path::Path, content: &str) {
        if let Some(p) = path.parent() {
            fs::create_dir_all(p).unwrap();
        }
        fs::write(path, content).unwrap();
    }

    /// 生成一个标准 manifest TOML。deps 元素为 (名字, 版本约束, path 可选项)。
    /// 注意：Dependency 是结构体，registry 依赖也必须写内联表 `{ version }`。
    fn pkg_toml(name: &str, version: &str, deps: &[(&str, &str, Option<&str>)]) -> String {
        let mut s = format!(
            "[package]\nname = \"{}\"\nversion = \"{}\"\nedition = \"2024\"\nauthors = []\n",
            name, version
        );
        s.push_str("[dependencies]\n");
        for (n, v, p) in deps {
            match p {
                Some(path) => s.push_str(&format!(
                    "{} = {{ version = \"{}\", path = \"{}\" }}\n",
                    n, v, path
                )),
                None => s.push_str(&format!("{} = {{ version = \"{}\" }}\n", n, v)),
            }
        }
        s
    }

    #[test]
    fn test_resolve_no_deps() {
        let dir = std::env::temp_dir().join("tenthpm_resolve_empty");
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        write(&dir.join("Tenth.toml"), &pkg_toml("root", "0.1.0", &[]));

        let m = Manifest::load_from_file(&dir.join("Tenth.toml")).unwrap();
        let res = resolve(&m, &dir).unwrap();
        assert!(res.packages.is_empty());
    }

    #[test]
    fn test_resolve_transitive_path() {
        let dir = std::env::temp_dir().join("tenthpm_resolve_trans");
        let _ = fs::remove_dir_all(&dir);
        // lib_c ← lib_b ← root（依赖目录均在项目根内，用 ./ 路径）
        write(
            &dir.join("lib_c").join("Tenth.toml"),
            &pkg_toml("lib_c", "0.3.0", &[]),
        );
        write(
            &dir.join("lib_b").join("Tenth.toml"),
            &pkg_toml("lib_b", "0.2.0", &[("lib_c", "*", Some("./lib_c"))]),
        );
        write(
            &dir.join("Tenth.toml"),
            &pkg_toml("root", "0.1.0", &[("lib_b", "*", Some("./lib_b"))]),
        );

        let m = Manifest::load_from_file(&dir.join("Tenth.toml")).unwrap();
        let res = resolve(&m, &dir).unwrap();
        let names = res.names();
        assert!(names.contains(&"lib_b"), "应包含直接依赖 lib_b: {:?}", names);
        assert!(names.contains(&"lib_c"), "应包含传递依赖 lib_c: {:?}", names);

        let lib_c = res.packages.iter().find(|p| p.name == "lib_c").unwrap();
        assert_eq!(lib_c.version, "0.3.0");
        assert!(lib_c.dependencies.is_empty());

        let lib_b = res.packages.iter().find(|p| p.name == "lib_b").unwrap();
        assert_eq!(lib_b.version, "0.2.0");
        assert_eq!(lib_b.dependencies, vec!["lib_c".to_string()]);
    }

    #[test]
    fn test_resolve_conflict_loud() {
        let dir = std::env::temp_dir().join("tenthpm_resolve_conflict");
        let _ = fs::remove_dir_all(&dir);
        // app_a → lib@1.0.0(^1.0)，app_b → lib@2.0.0(^2.0)，root → app_a + app_b
        write(
            &dir.join("libv1").join("Tenth.toml"),
            &pkg_toml("lib", "1.0.0", &[]),
        );
        write(
            &dir.join("libv2").join("Tenth.toml"),
            &pkg_toml("lib", "2.0.0", &[]),
        );
        write(
            &dir.join("app_a").join("Tenth.toml"),
            &pkg_toml("app_a", "0.1.0", &[("lib", "^1.0.0", Some("./libv1"))]),
        );
        write(
            &dir.join("app_b").join("Tenth.toml"),
            &pkg_toml("app_b", "0.1.0", &[("lib", "^2.0.0", Some("./libv2"))]),
        );
        write(
            &dir.join("Tenth.toml"),
            &pkg_toml(
                "root",
                "0.1.0",
                &[("app_a", "*", Some("./app_a")), ("app_b", "*", Some("./app_b"))],
            ),
        );

        let m = Manifest::load_from_file(&dir.join("Tenth.toml")).unwrap();
        let err = resolve(&m, &dir).unwrap_err();
        assert!(
            err.contains("版本冲突") && err.contains("lib"),
            "冲突应响亮报错，实际: {}",
            err
        );
    }

    #[test]
    fn test_resolve_picks_highest_compatible() {
        let dir = std::env::temp_dir().join("tenthpm_resolve_highest");
        let _ = fs::remove_dir_all(&dir);
        // lib 只有一个副本 1.5.0，两个约束 ^1.0.0 与 >=1.3.0 都满足 → 取 1.5.0
        write(
            &dir.join("lib").join("Tenth.toml"),
            &pkg_toml("lib", "1.5.0", &[]),
        );
        write(
            &dir.join("app_a").join("Tenth.toml"),
            &pkg_toml("app_a", "0.1.0", &[("lib", "^1.0.0", Some("./lib"))]),
        );
        write(
            &dir.join("Tenth.toml"),
            &pkg_toml(
                "root",
                "0.1.0",
                &[("app_a", "*", Some("./app_a")), ("lib", "*", Some("./lib"))],
            ),
        );

        let m = Manifest::load_from_file(&dir.join("Tenth.toml")).unwrap();
        let res = resolve(&m, &dir).unwrap();
        let lib = res.packages.iter().find(|p| p.name == "lib").unwrap();
        assert_eq!(lib.version, "1.5.0");
    }

    #[test]
    fn test_resolve_cycle_detected() {
        let dir = std::env::temp_dir().join("tenthpm_resolve_cycle");
        let _ = fs::remove_dir_all(&dir);
        // a → b，b → a
        write(
            &dir.join("a").join("Tenth.toml"),
            &pkg_toml("a", "0.1.0", &[("b", "*", Some("./b"))]),
        );
        write(
            &dir.join("b").join("Tenth.toml"),
            &pkg_toml("b", "0.1.0", &[("a", "*", Some("./a"))]),
        );
        write(
            &dir.join("Tenth.toml"),
            &pkg_toml("root", "0.1.0", &[("a", "*", Some("./a"))]),
        );

        let m = Manifest::load_from_file(&dir.join("Tenth.toml")).unwrap();
        let err = resolve(&m, &dir).unwrap_err();
        assert!(
            err.contains("循环依赖"),
            "循环依赖应响亮报错，实际: {}",
            err
        );
    }

    #[test]
    fn test_resolve_missing_path_loud() {
        let dir = std::env::temp_dir().join("tenthpm_resolve_missing");
        let _ = fs::remove_dir_all(&dir);
        write(
            &dir.join("Tenth.toml"),
            &pkg_toml("root", "0.1.0", &[("ghost", "*", Some("./does_not_exist"))]),
        );

        let m = Manifest::load_from_file(&dir.join("Tenth.toml")).unwrap();
        let err = resolve(&m, &dir).unwrap_err();
        assert!(
            err.contains("不存在"),
            "缺失依赖应响亮报错，实际: {}",
            err
        );
    }

    #[test]
    fn test_resolve_registry_only_conflict() {
        let dir = std::env::temp_dir().join("tenthpm_resolve_reg_conflict");
        let _ = fs::remove_dir_all(&dir);
        // 两个 registry 依赖对同名包 lib 提出互斥约束（无本地副本）
        write(
            &dir.join("Tenth.toml"),
            &pkg_toml("root", "0.1.0", &[("lib", "^1.0.0", None)]),
        );
        let m = Manifest::load_from_file(&dir.join("Tenth.toml")).unwrap();
        // 单约束不冲突
        assert!(resolve(&m, &dir).is_ok());

        // 直接在同名 key 上写互斥约束做不到（HashMap 同 key），用两个约束谓词：
        // ">=2.0.0,<1.0.0" 本身不可满足 → 应报错
        write(
            &dir.join("Tenth.toml"),
            &pkg_toml("root", "0.1.0", &[("lib", ">=2.0.0,<1.0.0", None)]),
        );
        let m2 = Manifest::load_from_file(&dir.join("Tenth.toml")).unwrap();
        let err = resolve(&m2, &dir).unwrap_err();
        assert!(err.contains("版本冲突"), "实际: {}", err);
    }

    #[test]
    fn test_lockfile_from_resolution() {
        let dir = std::env::temp_dir().join("tenthpm_lock_res");
        let _ = fs::remove_dir_all(&dir);
        write(
            &dir.join("lib_c").join("Tenth.toml"),
            &pkg_toml("lib_c", "0.3.0", &[]),
        );
        write(
            &dir.join("lib_b").join("Tenth.toml"),
            &pkg_toml("lib_b", "0.2.0", &[("lib_c", "*", Some("./lib_c"))]),
        );
        write(
            &dir.join("Tenth.toml"),
            &pkg_toml("root", "0.1.0", &[("lib_b", "*", Some("./lib_b"))]),
        );

        let m = Manifest::load_from_file(&dir.join("Tenth.toml")).unwrap();
        let res = resolve(&m, &dir).unwrap();
        let lock = Lockfile::from_resolution(&res);
        assert_eq!(lock.packages.len(), 2);
        let lib_b = lock.packages.iter().find(|p| p.name == "lib_b").unwrap();
        assert_eq!(lib_b.dependencies, vec!["lib_c".to_string()]);
        let lib_c = lock.packages.iter().find(|p| p.name == "lib_c").unwrap();
        assert!(lib_c.checksum.is_some(), "path 依赖应有 checksum");
    }

    #[test]
    fn test_resolve_same_name_two_dirs_picks_highest() {
        // 同名 lib 的两个目录（1.0.0 / 2.0.0），两条依赖边约束都是 `*`：
        // 两个目录都必须被计入可用版本，取最高 2.0.0——绝不静默选 1.0.0。
        let dir = std::env::temp_dir().join("tenthpm_resolve_two_dirs");
        let _ = fs::remove_dir_all(&dir);
        write(
            &dir.join("libv1").join("Tenth.toml"),
            &pkg_toml("lib", "1.0.0", &[]),
        );
        write(
            &dir.join("libv2").join("Tenth.toml"),
            &pkg_toml("lib", "2.0.0", &[]),
        );
        write(
            &dir.join("app_a").join("Tenth.toml"),
            &pkg_toml("app_a", "0.1.0", &[("lib", "*", Some("./libv1"))]),
        );
        write(
            &dir.join("app_b").join("Tenth.toml"),
            &pkg_toml("app_b", "0.1.0", &[("lib", "*", Some("./libv2"))]),
        );
        write(
            &dir.join("Tenth.toml"),
            &pkg_toml(
                "root",
                "0.1.0",
                &[("app_a", "*", Some("./app_a")), ("app_b", "*", Some("./app_b"))],
            ),
        );

        let m = Manifest::load_from_file(&dir.join("Tenth.toml")).unwrap();
        let res = resolve(&m, &dir).unwrap();
        let lib = res.packages.iter().find(|p| p.name == "lib").unwrap();
        assert_eq!(lib.version, "2.0.0", "同名两目录都满足约束时应取最高版本");
    }
}
