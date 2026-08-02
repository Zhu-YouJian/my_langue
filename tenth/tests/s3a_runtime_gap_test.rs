//! M1-S3a 守护：两个运行时小缺口（VM = 解释器）
//!
//! 子任务 A：解释器 `broadcast_to` 方法分派（手册 §11.3，位置参数形式）
//!   - 修复前：VM 可用、解释器报「未知的张量方法: broadcast_to」
//!   - 修复：interpreter/methods.rs 补 broadcast_to（语义与 VM natives.rs 对齐）
//!
//! 子任务 B：内联 `mod { }` 块限定调用（手册 §9.1/9.2）
//!   - 修复前：VM 报「未定义的函数 'mod::fn'」、解释器可用
//!   - 修复：lower_expr.rs::try_resolve_module_qualified 支持内联 mod 键，
//!     编译期把 `mod::fn(...)` 解析为底层函数名并补入 functions
//!
//! 全部走真实二进制子进程，VM（默认）与解释器（TENTH_NO_VM=1）双路径断言。

use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};

/// 被测二进制（同 crate 的 bin target，cargo test 会自动构建）。
const TENTH_EXE: &str = env!("CARGO_BIN_EXE_tenth");
/// 包根目录（= tenth/），作为子进程 cwd。
const TENTH_DIR: &str = env!("CARGO_MANIFEST_DIR");

static COUNTER: AtomicU64 = AtomicU64::new(0);

/// 运行一个 .th 程序，返回 (exit_code, stdout, stderr)。
fn run_th(prog: &str, use_vm: bool) -> (i32, String, String) {
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!("tenth_s3a_{}_{}", std::process::id(), n));
    std::fs::create_dir_all(&dir).unwrap();
    let file = dir.join("s3a.th");
    std::fs::write(&file, prog).unwrap();

    let mut cmd = Command::new(TENTH_EXE);
    cmd.arg("run").arg(&file).current_dir(TENTH_DIR);
    if !use_vm {
        cmd.env("TENTH_NO_VM", "1");
    }
    let out = cmd.output().expect("运行 tenth.exe 失败");
    let stdout = String::from_utf8_lossy(&out.stdout).to_string();
    let stderr = String::from_utf8_lossy(&out.stderr).to_string();
    let code = out.status.code().unwrap_or(-1);
    let _ = std::fs::remove_dir_all(&dir);
    (code, stdout, stderr)
}

/// 断言 exit 0 且 stdout 含 `= true`（末尾布尔表达式求值为真）。
fn assert_smoke(name: &str, prog: &str, use_vm: bool) {
    let (code, stdout, stderr) = run_th(prog, use_vm);
    assert!(
        code == 0 && stdout.contains("= true"),
        "[{}] 失败: exit={}\n--- stdout ---\n{}\n--- stderr ---\n{}",
        name,
        code,
        stdout,
        stderr
    );
}

/// 断言 exit 0 且 stdout 含 marker（用于非布尔输出验证）。
fn assert_out(name: &str, prog: &str, use_vm: bool, marker: &str) {
    let (code, stdout, stderr) = run_th(prog, use_vm);
    assert!(
        code == 0 && stdout.contains(marker),
        "[{}] 失败: exit={}\n--- stdout ---\n{}\n--- stderr ---\n{}",
        name,
        code,
        stdout,
        stderr
    );
}

// ══════════════════════════════════════════════════════════════════
// 子任务 B：内联 `mod { }` 限定调用（手册 §9.1/9.2）
// ══════════════════════════════════════════════════════════════════

/// §9.1 直呼：`mod math { }` + `math::add(1, 2)`（无 use）。
const IM_DIRECT: &str = r#"
mod math {
    fn add(a: i32, b: i32) -> i32 { a + b }
}
math::add(1, 2)
"#;

/// §9.1/9.2 手册模式：pub 函数 + 私有 helper + `use math::add;` + 限定调用。
const IM_USE_QUALIFIED: &str = r#"
mod math {
    pub fn add(a: i64, b: i64) -> i64 { a + b }
    fn helper() -> i64 { 42 }   // 私有
}
use math::add;
let r = math::add(1, 2);
r == 3
"#;

/// 限定调用出现在表达式中（非仅语句）。
const IM_EXPR_POSITION: &str = r#"
mod math {
    fn add(a: i32, b: i32) -> i32 { a + b }
}
math::add(1, 2) + 10
"#;

/// 回归：既有裸名导入（`use math::double;` + `double(21)`）不受影响。
const IM_REG_BARE: &str = r#"
mod math {
    fn double(x: i32) -> i32 { x * 2 }
}
use math::double;
double(21)
"#;

/// 回归：内联 mod + glob 导入（`use math::*` + 裸名调用）不受影响。
const IM_REG_GLOB: &str = r#"
mod math {
    fn add(a: i32, b: i32) -> i32 { a + b }
    fn mul(a: i32, b: i32) -> i32 { a * b }
}
use math::*;
add(3, 4) + mul(2, 5)
"#;

// ══════════════════════════════════════════════════════════════════
// 子任务 A：解释器 `broadcast_to` 分派（手册 §11.3）
// ══════════════════════════════════════════════════════════════════

/// 手册 §11.3 位置参数形式：t（shape [1,4]）→ broadcast_to(3, 4)。
const BT_POSITIONAL: &str = r#"
let t = tensor[[1.0, 2.0, 3.0, 4.0]];
let b = t.broadcast_to(3, 4);
b[2][3] == 4.0 && b.numel() == 12
"#;

/// 2D 广播语义：shape [1,2] → [3,2]，每行复制源数据。
const BT_2D_BROADCAST: &str = r#"
let t = tensor[[1.0, 2.0]];
let b = t.broadcast_to(3, 2);
b[0][1] == 2.0 && b[2][1] == 2.0 && b.numel() == 6
"#;

// ══════════════════════════════════════════════════════════════════
// 测试
// ══════════════════════════════════════════════════════════════════

// ── 子任务 B ──

#[test]
fn inline_mod_direct_vm() {
    assert_out("内联 mod 直呼 VM", IM_DIRECT, true, "= 3");
}

#[test]
fn inline_mod_direct_interp() {
    assert_out("内联 mod 直呼 解释器", IM_DIRECT, false, "= 3");
}

#[test]
fn inline_mod_use_qualified_vm() {
    assert_smoke("内联 mod use 限定调用 VM", IM_USE_QUALIFIED, true);
}

#[test]
fn inline_mod_use_qualified_interp() {
    assert_smoke("内联 mod use 限定调用 解释器", IM_USE_QUALIFIED, false);
}

#[test]
fn inline_mod_expr_position_vm() {
    assert_out("内联 mod 表达式位置 VM", IM_EXPR_POSITION, true, "= 13");
}

#[test]
fn inline_mod_expr_position_interp() {
    assert_out("内联 mod 表达式位置 解释器", IM_EXPR_POSITION, false, "= 13");
}

#[test]
fn inline_mod_reg_bare_vm() {
    assert_out("内联 mod 裸名回归 VM", IM_REG_BARE, true, "= 42");
}

#[test]
fn inline_mod_reg_bare_interp() {
    assert_out("内联 mod 裸名回归 解释器", IM_REG_BARE, false, "= 42");
}

#[test]
fn inline_mod_reg_glob_vm() {
    assert_out("内联 mod glob 回归 VM", IM_REG_GLOB, true, "= 17");
}

#[test]
fn inline_mod_reg_glob_interp() {
    assert_out("内联 mod glob 回归 解释器", IM_REG_GLOB, false, "= 17");
}

// ── 子任务 A ──

#[test]
fn broadcast_to_positional_vm() {
    assert_smoke("broadcast_to 位置参数 VM", BT_POSITIONAL, true);
}

#[test]
fn broadcast_to_positional_interp() {
    assert_smoke("broadcast_to 位置参数 解释器", BT_POSITIONAL, false);
}

#[test]
fn broadcast_to_2d_vm() {
    assert_smoke("broadcast_to 2D 广播 VM", BT_2D_BROADCAST, true);
}

#[test]
fn broadcast_to_2d_interp() {
    assert_smoke("broadcast_to 2D 广播 解释器", BT_2D_BROADCAST, false);
}
