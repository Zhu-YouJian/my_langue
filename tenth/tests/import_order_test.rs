//! AUDIT-11.4.41 守护：文件模块导入的**顺序无关性**与「不再静默」。
//!
//! 真根因（上一轮预备审计更正，**不是**「符号覆盖」）：
//! 1. `hir/lower/import.rs` 把「已导入（去重/循环守卫短路）」复用为 `Ok(None)`
//!    ——与「文件不存在」同一返回值；
//! 2. `load_and_compile_file` 只把 `imported_files` 复制下去、合并回来，
//!    **从不回流 / 下传 `sub_lowerer.modules`**；
//! ⇒ 随后 lowered 的**兄弟模块**其内部 `use` 命中去重守卫 → 被打回「找不到」→
//! 缓存又取不到 → 落到 inline-mod fallback **静默空转**。
//! ⇒ 表现为**顺序相关**：**后导入者失效**（可能报「未定义」，也可能静默 Unit）。
//!
//! 本文件钉住两件事：
//! - **顺序无关**：两个内部 use 同一模块的兄弟模块，谁先谁后都必须正常解析，
//!   且 VM 与解释器（`TENTH_NO_VM=1`）行为一致；
//! - **不再静默**：`use` 整条解析不到任何模块时，必须给出编译期警告
//!   （此前完全静默产出空）。
//!
//! 说明：子进程 cwd = 测试自建的临时目录，模块文件也在其中，故不依赖仓库内
//! 任何新增文件（`use p::m1::*` 经 cwd 搜索路径解析）。

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};

/// 被测二进制（同 crate 的 bin target，cargo test 会自动构建）。
const TENTH_EXE: &str = env!("CARGO_BIN_EXE_tenth");

static COUNTER: AtomicU64 = AtomicU64::new(0);

fn new_temp_dir() -> PathBuf {
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!(
        "tenth_import_order_{}_{}",
        std::process::id(),
        n
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn write_file(dir: &Path, rel: &str, content: &str) {
    let p = dir.join(rel);
    if let Some(parent) = p.parent() {
        std::fs::create_dir_all(parent).unwrap();
    }
    std::fs::write(&p, content).unwrap();
}

/// 在临时目录里落盘 `files`，跑 `main.th`，返回 (exit_code, stdout, stderr)。
fn run_prog(files: &[(&str, &str)], use_vm: bool) -> (i32, String, String) {
    let dir = new_temp_dir();
    for (rel, content) in files {
        write_file(&dir, rel, content);
    }

    let mut cmd = Command::new(TENTH_EXE);
    cmd.arg("run").arg(dir.join("main.th")).current_dir(&dir);
    if !use_vm {
        cmd.env("TENTH_NO_VM", "1");
    }
    let out = cmd.output().expect("运行 tenth.exe 失败");
    let code = out.status.code().unwrap_or(-1);
    let stdout = String::from_utf8_lossy(&out.stdout).to_string();
    let stderr = String::from_utf8_lossy(&out.stderr).to_string();
    let _ = std::fs::remove_dir_all(&dir);
    (code, stdout, stderr)
}

/// 「后导入者失效」最小场景的模块文件集（`order` 决定两条 use 的先后）。
fn later_importer_files(order: [&'static str; 2]) -> Vec<(&'static str, &'static str)> {
    let main_src: &'static str = match order {
        ["p::m1", "p::m2"] => "use p::m1::*\nuse p::m2::*\nfn main() { println(m1() + m2()) }\n",
        _ => "use p::m2::*\nuse p::m1::*\nfn main() { println(m1() + m2()) }\n",
    };
    vec![
        ("main.th", main_src),
        // 两个兄弟模块各自 `use` **同一个**模块的同一个函数——
        // 旧实现下先 lowered 的那个把 "q::shared" 写进 imported_files，
        // 后 lowered 的那个内部 use 即被打回「文件不存在」而静默空转。
        ("p/m1.th", "use q::shared::helper\nfn m1() -> i64 { helper() }\n"),
        ("p/m2.th", "use q::shared::helper\nfn m2() -> i64 { helper() * 2 }\n"),
        ("q/shared.th", "fn helper() -> i64 { 7 }\n"),
    ]
}

fn assert_ok(name: &str, files: &[(&str, &str)], expected_stdout: &str, use_vm: bool) {
    let (code, stdout, stderr) = run_prog(files, use_vm);
    assert!(
        code == 0 && stdout.contains(expected_stdout),
        "[{}] 路径失败: exit={} 期望 stdout 含 '{}'\n--- stdout ---\n{}\n--- stderr ---\n{}",
        name,
        code,
        expected_stdout,
        stdout,
        stderr
    );
}

// ══════════════════════════════════════════════════════════════════
// 1. 顺序无关：后导入者的内部 use 必须生效，两路径一致
// ══════════════════════════════════════════════════════════════════

#[test]
fn later_importer_resolves_m1_then_m2_vm() {
    let files = later_importer_files(["p::m1", "p::m2"]);
    assert_ok("m1 先 / VM", &files, "21", true);
}

#[test]
fn later_importer_resolves_m1_then_m2_interp() {
    let files = later_importer_files(["p::m1", "p::m2"]);
    assert_ok("m1 先 / 解释器", &files, "21", false);
}

#[test]
fn later_importer_resolves_m2_then_m1_vm() {
    // 顺序对调：失败方必须随之对调（旧实现是「后导入者失效」）→ 修后两者都通过
    let files = later_importer_files(["p::m2", "p::m1"]);
    assert_ok("m2 先 / VM", &files, "21", true);
}

#[test]
fn later_importer_resolves_m2_then_m1_interp() {
    let files = later_importer_files(["p::m2", "p::m1"]);
    assert_ok("m2 先 / 解释器", &files, "21", false);
}

#[test]
fn import_order_two_paths_agree() {
    // 验收要求：两路径（VM / TENTH_NO_VM=1）行为一致
    for order in [["p::m1", "p::m2"], ["p::m2", "p::m1"]] {
        let files = later_importer_files(order);
        let (c_vm, o_vm, e_vm) = run_prog(&files, true);
        let (c_it, o_it, e_it) = run_prog(&files, false);
        assert_eq!(c_vm, 0, "VM exit={c_vm} stderr={e_vm}");
        assert_eq!(c_it, 0, "解释器 exit={c_it} stderr={e_it}");
        assert_eq!(
            o_vm, o_it,
            "两路径 stdout 不一致：VM={:?} 解释器={:?}",
            o_vm, o_it
        );
        assert!(o_vm.contains("21"), "stdout 应为 21，实际 {:?}", o_vm);
    }
}

// ══════════════════════════════════════════════════════════════════
// 2. 不再静默：解析不到的 use 必须响亮（编译期警告）
// ══════════════════════════════════════════════════════════════════

const UNRESOLVED_MAIN: &str = "use no_such_module::nope\nfn main() { println(1) }\n";

#[test]
fn unresolved_use_is_loud_vm() {
    let files: Vec<(&str, &str)> = vec![("main.th", UNRESOLVED_MAIN)];
    let (code, stdout, stderr) = run_prog(&files, true);
    assert_eq!(code, 0, "应仍可运行（警告非致命），stderr={stderr}");
    assert!(stdout.contains("1"), "stdout={stdout}");
    assert!(
        stderr.contains("未能解析") && stderr.contains("no_such_module"),
        "解析不到的 use 必须响亮（stderr 应含警告）\n--- stderr ---\n{}",
        stderr
    );
}

#[test]
fn unresolved_use_is_loud_interp() {
    let files: Vec<(&str, &str)> = vec![("main.th", UNRESOLVED_MAIN)];
    let (code, stdout, stderr) = run_prog(&files, false);
    assert_eq!(code, 0, "应仍可运行（警告非致命），stderr={stderr}");
    assert!(stdout.contains("1"), "stdout={stdout}");
    assert!(
        stderr.contains("未能解析") && stderr.contains("no_such_module"),
        "解析不到的 use 必须响亮（解释器路径 stderr 应含警告）\n--- stderr ---\n{}",
        stderr
    );
}

#[test]
fn resolvable_use_produces_no_warning() {
    // 零误报：能正常解析的 use 不得产生「未能解析」警告
    let files: Vec<(&str, &str)> = vec![
        ("main.th", "use q::shared::helper\nfn main() { println(helper()) }\n"),
        ("q/shared.th", "fn helper() -> i64 { 7 }\n"),
    ];
    for use_vm in [true, false] {
        let (code, stdout, stderr) = run_prog(&files, use_vm);
        assert_eq!(code, 0, "exit={code} stderr={stderr}");
        assert!(stdout.contains("7"), "stdout={stdout}");
        assert!(
            !stderr.contains("未能解析"),
            "正常 use 不应产生解析警告（use_vm={}）\n--- stderr ---\n{}",
            use_vm,
            stderr
        );
    }
}

/// 深路径 use（4 段）在解释器路径可用（`interpreter/core.rs` 的多段回填修复守护）。
#[test]
fn deep_segment_use_works_on_both_paths() {
    let files: Vec<(&str, &str)> = vec![
        ("main.th", "use q::sub::shared::helper\nfn main() { println(helper()) }\n"),
        ("q/sub/shared.th", "fn helper() -> i64 { 5 }\n"),
    ];
    for use_vm in [true, false] {
        let (code, stdout, stderr) = run_prog(&files, use_vm);
        assert!(
            code == 0 && stdout.contains("5"),
            "深路径 use 失败（use_vm={}）: exit={} stdout={:?}\n--- stderr ---\n{}",
            use_vm,
            code,
            stdout,
            stderr
        );
        assert!(!stderr.contains("未能解析"), "深路径 use 不应报警告: {stderr}");
    }
}
