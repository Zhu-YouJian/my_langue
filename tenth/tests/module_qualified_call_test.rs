//! AUDIT-11.4.23 守护：文件级模块限定调用 `mod::fn(...)` 可用（VM = 解释器）。
//!
//! 覆盖：
//! - `use std::env;` + `env::get_or_empty(...)`（文件模块，手册 §12.14.2 模式）
//! - `use std::collections;` + `collections::flat_map(...)`（目录模块，
//!   经 `std/collections/collections.th` 解析，AUDIT-11.4.28 同源路径）
//! - `use nn::activations;` + `activations::leaky_relu_select_default(...)`
//!   （文件模块 + 非泛型函数，手册 §11.6/§12.13 同款 use 写法）
//! - 回归：既有直接导入（`use std::env::get_or_empty;` 裸名调用）不受影响
//! - 回归：glob 导入（`use std::env::*`）不受影响
//!
//! 说明：泛型函数普通调用（无显式类型参数）为既有语言限制（同文件同样失败，
//! 见 `.trae/tmp/manual_audit/m_generic_add.th` 系诊断），非本 AUDIT 范围；
//! 泛型模块限定调用需显式类型参数 `mod::fn<T>(...)`。

use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};

/// 被测二进制（同 crate 的 bin target，cargo test 会自动构建）。
const TENTH_EXE: &str = env!("CARGO_BIN_EXE_tenth");
/// 包根目录（= tenth/），作为子进程 cwd 使 `use std::...` 解析到 tenth/std/。
const TENTH_DIR: &str = env!("CARGO_MANIFEST_DIR");

static COUNTER: AtomicU64 = AtomicU64::new(0);

/// 运行一个 .th 程序（走真实 use 路径），返回 (exit_code, stdout, stderr)。
fn run_th(prog: &str, use_vm: bool) -> (i32, String, String) {
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!("tenth_mq_{}_{}", std::process::id(), n));
    std::fs::create_dir_all(&dir).unwrap();
    let file = dir.join("mq.th");
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
// AUDIT-11.4.23 主场景：`use <mod>;` + `<mod>::fn(...)`
// ══════════════════════════════════════════════════════════════════

/// 手册 §12.14.2 模式：`use std::env;` + `env::get_or_empty("HOME")`。
const MQ_ENV_QUALIFIED: &str = r#"
use std::env;

fn main() {
    let home = env::get_or_empty("TENTH_MQ_MISSING");
    home == ""
}
"#;

/// 目录模块：`use std::collections;` + `collections::flat_map(...)`。
const MQ_COLLECTIONS_QUALIFIED: &str = r#"
use std::collections;

fn main() {
    let v = [1, 2, 3, 4];
    let doubled = collections::flat_map(v, |x| [x * 2]);
    doubled.len() == 4 && doubled[0] == 2 && doubled[3] == 8
}
"#;

/// 文件模块 + 非泛型函数：`use nn::activations;` + `activations::leaky_relu_select_default(...)`。
const MQ_ACTIVATIONS_QUALIFIED: &str = r#"
use nn::activations;

fn main() {
    let x = tensor[[-1.0, 0.0, 2.0]];
    let y = activations::leaky_relu_select_default(x);
    y[0][0] == -0.01 && y[0][1] == 0.0 && y[0][2] == 2.0
}
"#;

/// 手册 §12.14.2 完整模式：`env::get_or_empty` 限定调用 + 裸 `get_result` 同文件混用。
const MQ_ENV_MIXED: &str = r#"
use std::env;

fn main() {
    let home = env::get_or_empty("TENTH_MQ_MISSING");
    let r = get_result("TENTH_MQ_MISSING");
    // 限定调用返回空串 + 裸调用返回 Err（缺失变量）
    home == "" && match r {
        Result::Err(_) => true,
        Result::Ok(_) => false,
    }
}
"#;

// ══════════════════════════════════════════════════════════════════
// 回归：既有 use 模式不破坏
// ══════════════════════════════════════════════════════════════════

/// 直接导入函数名 + 裸名调用（AUDIT 所述"正确写法"）不回归。
const MQ_REG_DIRECT_IMPORT: &str = r#"
use std::env::get_or_empty;

fn main() {
    get_or_empty("TENTH_MQ_MISSING") == ""
}
"#;

/// glob 导入不回归。
const MQ_REG_GLOB: &str = r#"
use std::env::*

fn main() {
    set("TENTH_MQ_VAR", "abc");
    get("TENTH_MQ_VAR", "") == "abc"
}
"#;

// ══════════════════════════════════════════════════════════════════
// 测试
// ══════════════════════════════════════════════════════════════════

#[test]
fn module_qualified_env_vm() {
    assert_smoke("env 限定调用 VM", MQ_ENV_QUALIFIED, true);
}

#[test]
fn module_qualified_env_interp() {
    assert_smoke("env 限定调用 解释器", MQ_ENV_QUALIFIED, false);
}

#[test]
fn module_qualified_collections_vm() {
    assert_smoke("collections 目录模块限定调用 VM", MQ_COLLECTIONS_QUALIFIED, true);
}

#[test]
fn module_qualified_collections_interp() {
    assert_smoke("collections 目录模块限定调用 解释器", MQ_COLLECTIONS_QUALIFIED, false);
}

#[test]
fn module_qualified_activations_vm() {
    assert_smoke("activations 限定调用 VM", MQ_ACTIVATIONS_QUALIFIED, true);
}

#[test]
fn module_qualified_activations_interp() {
    assert_smoke("activations 限定调用 解释器", MQ_ACTIVATIONS_QUALIFIED, false);
}

#[test]
fn module_qualified_env_mixed_vm() {
    assert_smoke("env 限定+裸调用混用 VM", MQ_ENV_MIXED, true);
}

#[test]
fn module_qualified_env_mixed_interp() {
    assert_smoke("env 限定+裸调用混用 解释器", MQ_ENV_MIXED, false);
}

#[test]
fn module_qualified_reg_direct_import_vm() {
    assert_smoke("直接导入回归 VM", MQ_REG_DIRECT_IMPORT, true);
}

#[test]
fn module_qualified_reg_direct_import_interp() {
    assert_smoke("直接导入回归 解释器", MQ_REG_DIRECT_IMPORT, false);
}

#[test]
fn module_qualified_reg_glob_vm() {
    assert_smoke("glob 导入回归 VM", MQ_REG_GLOB, true);
}

#[test]
fn module_qualified_reg_glob_interp() {
    assert_smoke("glob 导入回归 解释器", MQ_REG_GLOB, false);
}

/// 额外输出形态守护：限定调用返回值打印为字符串（VM）。
#[test]
fn module_qualified_env_output_vm() {
    assert_out("env 限定调用输出 VM", MQ_ENV_QUALIFIED, true, "true");
}
