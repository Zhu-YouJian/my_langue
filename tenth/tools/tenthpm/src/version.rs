//! 语义化版本子集：版本解析、约束匹配、区间可满足性。
//!
//! 保守实现（M4.1）：仅支持数字 `X.Y.Z`（无预发布/构建元数据）。约束语法：
//! - `*`                任意版本
//! - `1.2.3`            精确版本
//! - `^1.2.3`           caret：`>=1.2.3 <2.0.0`（`^0.2.3` → `>=0.2.3 <0.3.0`，`^0.0.3` → 精确 0.0.3）
//! - `>=1.2.3` / `>1.2.3` / `<=2.0.0` / `<2.0.0`
//! - 多个谓词以逗号分隔（AND），如 `>=1.2.0,<2.0.0`
//!
//! 依赖解析的护城河红线：约束冲突必须**响亮报错**，绝不静默选择错误版本。
//! 因此本模块同时提供约束集合的可满足性判定（`reqs_conflict`），用于
//! registry 依赖（无本地副本、无法核对具体版本）时的冲突检测。

use std::cmp::Ordering;
use std::fmt;

/// 一个具体版本号 `X.Y.Z`（无预发布/构建元数据）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Version {
    pub major: u32,
    pub minor: u32,
    pub patch: u32,
}

impl Version {
    pub fn new(major: u32, minor: u32, patch: u32) -> Self {
        Version {
            major,
            minor,
            patch,
        }
    }

    /// 解析 `X.Y.Z` 格式，三段均为十进制无符号整数。
    pub fn parse(s: &str) -> Option<Version> {
        let s = s.trim();
        let parts: Vec<&str> = s.split('.').collect();
        if parts.len() != 3 {
            return None;
        }
        Some(Version {
            major: parts[0].parse().ok()?,
            minor: parts[1].parse().ok()?,
            patch: parts[2].parse().ok()?,
        })
    }
}

impl fmt::Display for Version {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}.{}.{}", self.major, self.minor, self.patch)
    }
}

impl Ord for Version {
    fn cmp(&self, other: &Self) -> Ordering {
        (self.major, self.minor, self.patch).cmp(&(other.major, other.minor, other.patch))
    }
}

impl PartialOrd for Version {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// 版本约束的运算符。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Op {
    /// `*` — 任意
    Any,
    /// 精确版本 `1.2.3`
    Exact,
    /// caret `^1.2.3`
    Caret,
    /// `>1.2.3`
    Gt,
    /// `>=1.2.3`
    Ge,
    /// `<1.2.3`
    Lt,
    /// `<=1.2.3`
    Le,
}

/// 单个版本谓词：`<Op> <Version>`。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Predicate {
    pub op: Op,
    pub ver: Version,
}

/// 一个版本约束 = 若干谓词的合取（逗号分隔 AND）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VersionReq {
    pub predicates: Vec<Predicate>,
}

impl VersionReq {
    /// 解析版本约束字符串。失败时返回带定位信息的错误（响亮，不静默回退）。
    pub fn parse(s: &str) -> Result<VersionReq, String> {
        let s = s.trim();
        if s.is_empty() {
            return Err("版本约束不能为空".to_string());
        }
        if s == "*" {
            return Ok(VersionReq {
                predicates: vec![Predicate {
                    op: Op::Any,
                    ver: Version::new(0, 0, 0),
                }],
            });
        }
        let mut predicates = Vec::new();
        for part in s.split(',') {
            let part = part.trim();
            if part.is_empty() {
                return Err(format!("版本约束 `{}` 含空谓词", s));
            }
            let (op, rest) = if let Some(r) = part.strip_prefix(">=") {
                (Op::Ge, r)
            } else if let Some(r) = part.strip_prefix("<=") {
                (Op::Le, r)
            } else if let Some(r) = part.strip_prefix('>') {
                (Op::Gt, r)
            } else if let Some(r) = part.strip_prefix('<') {
                (Op::Lt, r)
            } else if let Some(r) = part.strip_prefix('^') {
                (Op::Caret, r)
            } else {
                (Op::Exact, part)
            };
            let rest = rest.trim();
            let ver = Version::parse(rest)
                .ok_or_else(|| format!("版本约束 `{}` 中 `{}` 不是有效的 X.Y.Z 版本", s, part))?;
            predicates.push(Predicate { op, ver });
        }
        Ok(VersionReq { predicates })
    }

    /// 判断给定版本是否满足本约束。
    pub fn matches(&self, v: &Version) -> bool {
        self.predicates.iter().all(|p| match p.op {
            Op::Any => true,
            Op::Exact => *v == p.ver,
            Op::Caret => {
                if p.ver.major > 0 {
                    v.major == p.ver.major && *v >= p.ver
                } else if p.ver.minor > 0 {
                    v.major == 0 && v.minor == p.ver.minor && *v >= p.ver
                } else {
                    *v == p.ver
                }
            }
            Op::Gt => *v > p.ver,
            Op::Ge => *v >= p.ver,
            Op::Lt => *v < p.ver,
            Op::Le => *v <= p.ver,
        })
    }

    /// 将单个约束映射为区间/精确点表示（用于可满足性判定）。
    fn interval(&self) -> Interval {
        let mut lo: Option<(Version, bool)> = None; // (版本, 是否含等)
        let mut hi: Option<(Version, bool)> = None;
        let mut exact: Option<Version> = None;
        for p in &self.predicates {
            match p.op {
                Op::Any => {}
                Op::Exact => exact = Some(p.ver),
                Op::Caret => {
                    let lower = p.ver;
                    let upper = if p.ver.major > 0 {
                        Version::new(p.ver.major + 1, 0, 0)
                    } else if p.ver.minor > 0 {
                        Version::new(0, p.ver.minor + 1, 0)
                    } else {
                        p.ver
                    };
                    lo = max_lo(lo, (lower, true));
                    if upper != lower {
                        hi = min_hi(hi, (upper, false));
                    }
                }
                Op::Ge => lo = max_lo(lo, (p.ver, true)),
                Op::Gt => lo = max_lo(lo, (p.ver, false)),
                Op::Le => hi = min_hi(hi, (p.ver, true)),
                Op::Lt => hi = min_hi(hi, (p.ver, false)),
            }
        }
        Interval { lo, hi, exact }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Interval {
    lo: Option<(Version, bool)>,
    hi: Option<(Version, bool)>,
    exact: Option<Version>,
}

fn max_lo(a: Option<(Version, bool)>, b: (Version, bool)) -> Option<(Version, bool)> {
    match a {
        None => Some(b),
        Some((av, ai)) => {
            if av > b.0 {
                Some((av, ai))
            } else if av < b.0 {
                Some(b)
            } else {
                // 同一下界：任一含等即可（取更宽的）
                Some((av, ai || b.1))
            }
        }
    }
}

fn min_hi(a: Option<(Version, bool)>, b: (Version, bool)) -> Option<(Version, bool)> {
    match a {
        None => Some(b),
        Some((av, ai)) => {
            if av < b.0 {
                Some((av, ai))
            } else if av > b.0 {
                Some(b)
            } else {
                // 同一上界：须两侧都含等才含等（取更窄的）
                Some((av, ai && b.1))
            }
        }
    }
}

/// 判断一组约束是否**不可满足**（不存在任何版本能同时满足全部约束）。
///
/// 返回 `true` 表示冲突（护城河红线：必须响亮报错）。
/// 用于 registry 依赖（无本地副本、没有可核对的具体版本）时的冲突检测。
pub fn reqs_conflict(reqs: &[VersionReq]) -> bool {
    if reqs.is_empty() {
        return false;
    }
    let mut lo: Option<(Version, bool)> = None;
    let mut hi: Option<(Version, bool)> = None;
    let mut exacts: Vec<Version> = Vec::new();
    for r in reqs {
        let iv = r.interval();
        lo = match (lo, iv.lo) {
            (Some(a), Some(b)) => Some(max_lo(Some(a), b).unwrap()),
            (a, b) => a.or(b),
        };
        hi = match (hi, iv.hi) {
            (Some(a), Some(b)) => Some(min_hi(Some(a), b).unwrap()),
            (a, b) => a.or(b),
        };
        if let Some(e) = iv.exact {
            exacts.push(e);
        }
    }
    // 若存在精确版本候选：只要任一精确版本满足全部约束即不冲突
    if !exacts.is_empty() {
        for e in &exacts {
            if reqs.iter().all(|r| r.matches(e)) {
                return false;
            }
        }
        return true;
    }
    // 纯区间：检查 [lo, hi] 非空
    match (lo, hi) {
        (None, None) => false,
        (Some(_), None) | (None, Some(_)) => false,
        (Some((l, li)), Some((h, hii))) => {
            if l < h {
                false
            } else if l > h {
                true
            } else {
                // 相等：需要两侧都含等
                !(li && hii)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version_parse_ok() {
        assert_eq!(Version::parse("1.2.3"), Some(Version::new(1, 2, 3)));
        assert_eq!(Version::parse("0.0.0"), Some(Version::new(0, 0, 0)));
        assert_eq!(Version::parse("10.20.30"), Some(Version::new(10, 20, 30)));
        assert_eq!(Version::parse(" 1.2.3 "), Some(Version::new(1, 2, 3)));
    }

    #[test]
    fn test_version_parse_bad() {
        assert_eq!(Version::parse("1.2"), None);
        assert_eq!(Version::parse("1"), None);
        assert_eq!(Version::parse("1.2.3.4"), None);
        assert_eq!(Version::parse("a.b.c"), None);
        assert_eq!(Version::parse("1.2.3-alpha"), None);
        assert_eq!(Version::parse(""), None);
        assert_eq!(Version::parse("1.2.x"), None);
    }

    #[test]
    fn test_version_ordering() {
        assert!(Version::new(1, 0, 0) < Version::new(1, 0, 1));
        assert!(Version::new(1, 0, 1) < Version::new(1, 1, 0));
        assert!(Version::new(1, 9, 9) < Version::new(2, 0, 0));
    }

    #[test]
    fn test_req_parse() {
        assert!(VersionReq::parse("*").is_ok());
        assert!(VersionReq::parse("1.2.3").is_ok());
        assert!(VersionReq::parse("^1.2.3").is_ok());
        assert!(VersionReq::parse(">=1.2.3").is_ok());
        assert!(VersionReq::parse(">1.2.3,<2.0.0").is_ok());
        assert!(VersionReq::parse("").is_err());
        assert!(VersionReq::parse("abc").is_err());
        assert!(VersionReq::parse("^1.2").is_err());
        assert!(VersionReq::parse(">=1.2.3,<2.0").is_err());
    }

    #[test]
    fn test_req_matches_exact() {
        let r = VersionReq::parse("1.2.3").unwrap();
        assert!(r.matches(&Version::new(1, 2, 3)));
        assert!(!r.matches(&Version::new(1, 2, 4)));
        assert!(!r.matches(&Version::new(1, 3, 0)));
    }

    #[test]
    fn test_req_matches_any() {
        let r = VersionReq::parse("*").unwrap();
        assert!(r.matches(&Version::new(0, 0, 1)));
        assert!(r.matches(&Version::new(99, 99, 99)));
    }

    #[test]
    fn test_req_matches_caret() {
        let r = VersionReq::parse("^1.2.3").unwrap();
        assert!(r.matches(&Version::new(1, 2, 3)));
        assert!(r.matches(&Version::new(1, 5, 0)));
        assert!(r.matches(&Version::new(1, 99, 99)));
        assert!(!r.matches(&Version::new(1, 2, 2)));
        assert!(!r.matches(&Version::new(2, 0, 0)));
        assert!(!r.matches(&Version::new(0, 9, 0)));

        let r0 = VersionReq::parse("^0.2.3").unwrap();
        assert!(r0.matches(&Version::new(0, 2, 3)));
        assert!(r0.matches(&Version::new(0, 2, 9)));
        assert!(!r0.matches(&Version::new(0, 3, 0)));
        assert!(!r0.matches(&Version::new(0, 1, 9)));

        let r00 = VersionReq::parse("^0.0.3").unwrap();
        assert!(r00.matches(&Version::new(0, 0, 3)));
        assert!(!r00.matches(&Version::new(0, 0, 4)));
    }

    #[test]
    fn test_req_matches_range() {
        let r = VersionReq::parse(">=1.0.0,<2.0.0").unwrap();
        assert!(r.matches(&Version::new(1, 0, 0)));
        assert!(r.matches(&Version::new(1, 9, 9)));
        assert!(!r.matches(&Version::new(2, 0, 0)));
        assert!(!r.matches(&Version::new(0, 9, 0)));

        let r2 = VersionReq::parse(">1.0.0,<=1.5.0").unwrap();
        assert!(!r2.matches(&Version::new(1, 0, 0)));
        assert!(r2.matches(&Version::new(1, 0, 1)));
        assert!(r2.matches(&Version::new(1, 5, 0)));
        assert!(!r2.matches(&Version::new(1, 5, 1)));
    }

    #[test]
    fn test_reqs_conflict_detects() {
        // 精确版本互斥
        assert!(reqs_conflict(&[
            VersionReq::parse("1.0.0").unwrap(),
            VersionReq::parse("2.0.0").unwrap(),
        ]));
        // 精确不在区间内
        assert!(reqs_conflict(&[
            VersionReq::parse("1.0.0").unwrap(),
            VersionReq::parse("^2.0.0").unwrap(),
        ]));
        // caret 区间互斥（^1 vs ^2）
        assert!(reqs_conflict(&[
            VersionReq::parse("^1.0.0").unwrap(),
            VersionReq::parse("^2.0.0").unwrap(),
        ]));
        // 区间完全错开
        assert!(reqs_conflict(&[
            VersionReq::parse(">=2.0.0").unwrap(),
            VersionReq::parse("<1.0.0").unwrap(),
        ]));
        // 半开区间仅端点相接（>1.0.0 与 <=1.0.0 无交集）
        assert!(reqs_conflict(&[
            VersionReq::parse(">1.0.0").unwrap(),
            VersionReq::parse("<=1.0.0").unwrap(),
        ]));
    }

    #[test]
    fn test_reqs_conflict_satisfiable() {
        // 相同精确版本
        assert!(!reqs_conflict(&[
            VersionReq::parse("1.0.0").unwrap(),
            VersionReq::parse("1.0.0").unwrap(),
        ]));
        // 精确落在区间内
        assert!(!reqs_conflict(&[
            VersionReq::parse("1.5.0").unwrap(),
            VersionReq::parse("^1.0.0").unwrap(),
        ]));
        // caret 区间相交
        assert!(!reqs_conflict(&[
            VersionReq::parse("^1.0.0").unwrap(),
            VersionReq::parse(">=1.5.0").unwrap(),
        ]));
        // 开闭区间相接（>=1.0.0 与 <2.0.0 有交集）
        assert!(!reqs_conflict(&[
            VersionReq::parse(">=1.0.0").unwrap(),
            VersionReq::parse("<2.0.0").unwrap(),
        ]));
        // 空约束
        assert!(!reqs_conflict(&[]));
        // 单个任意约束
        assert!(!reqs_conflict(&[VersionReq::parse("*").unwrap()]));
    }
}
