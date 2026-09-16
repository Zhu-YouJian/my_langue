//! AUDIT-11.4.60 守护：`use` 搜索路径含**脚本自身目录**（脚本目录优先）+ 多处命中响亮警告。
//!
//! 真缺口（登记册）：
//! - `main.rs::source_to_hir` 建的搜索路径只有 cwd / exe 同级 `std/` / `tenth/` / `tenth/std`，
//!   **不含脚本自身所在目录** ⇒ 脚本无法导入自己旁边的模块（多文件项目只能挤成单文件）；
//! - 子 lowerer 的搜索路径与顶层相同 ⇒ 模块 A 无法导入**与自己同目录**的模块 B。
//!
//! 本文件钉住四件事（不动 `import_order_test.rs` 的既有断言）：
//! 1. **脚本目录优先**：脚本在子目录、模块在旁边 ⇒ `use <模块>` 可解析并运行（VM 与解释器一致）；
//! 2. **子目录模块可导入同级模块**（`sub/b.th` 里 `use helper::greet`）；
//! 3. **多处命中必须响亮**：同一模块名在两个搜索目录都存在 ⇒ stderr 列出**全部候选 + 最终采用者**；
//! 4. **零回归**：脚本目录入表后，cwd 相对路径的 `use` 行为不变（`import_order_test` 的形态）且不产生假歧义警告。
//!
//! 全部用例的 cwd 与模块文件都在测试自建的临时目录内，不依赖仓库内任何新增文件；
//! 用例中出现的路径只进断言（不落盘），符合「日志不带工作区绝对路径」的收尾要求。

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};

/// 被测二进制（同 crate 的 bin target，cargo test 会自动构建）。
const TENTH_EXE: &str = env!("CARGO_BIN_EXE_tenth");

static COUNTER: AtomicU64 = AtomicU64::new(0);

fn new_temp_dir() -> PathBuf {
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!(
        "tenth_module_search_{}_{}",
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

/// 在临时目录里落盘 `files`，以 `cwd_rel` 作为 cwd 跑 `script_rel`，
/// 返回 (exit_code, stdout, stderr)。
fn run_prog_in(
    files: &[(&str, &str)],
    cwd_rel: &str,
    script_rel: &str,
    use_vm: bool,
) -> (i32, String, String) {
    let dir = new_temp_dir();
    for (rel, content) in files {
        write_file(&dir, rel, content);
    }
    let cwd = if cwd_rel.is_empty() { dir.clone() } else { dir.join(cwd_rel) };

    let mut cmd = Command::new(TENTH_EXE);
    cmd.arg("run").arg(dir.join(script_rel)).current_dir(&cwd);
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

/// 同 `run_prog_in`，但 cwd = 临时目录根。
fn run_prog(files: &[(&str, &str)], script_rel: &str, use_vm: bool) -> (i32, String, String) {
    run_prog_in(files, "", script_rel, use_vm)
}

// ══════════════════════════════════════════════════════════════════
// 1. 脚本自身目录入搜索路径（AUDIT-11.4.60(a)）
// ══════════════════════════════════════════════════════════════════

/// 正例：脚本在子目录、模块在旁边。**cwd = 临时目录根**（脚本目录之外）⇒
/// 修复前必然 NotFound（只按 cwd / std 搜），修复后经「脚本目录优先」解析成功。
#[test]
fn script_dir_module_is_searchable_both_paths() {
    let files: Vec<(&str, &str)> = vec![
        ("sub/main.th", "use helper::greet\nfn main() { println(greet()) }\n"),
        ("sub/helper.th", "fn greet() -> i64 { 42 }\n"),
    ];
    for use_vm in [true, false] {
        let (code, stdout, stderr) = run_prog(&files, "sub/main.th", use_vm);
        assert_eq!(
            code, 0,
            "[脚本目录搜索] 应可运行（use_vm={use_vm}）\n--- stdout ---\n{stdout}\n--- stderr ---\n{stderr}"
        );
        assert!(
            stdout.contains("42"),
            "[脚本目录搜索] stdout 应含 42（use_vm={use_vm}）\n--- stdout ---\n{stdout}\n--- stderr ---\n{stderr}"
        );
        assert!(
            !stderr.contains("未能解析"),
            "[脚本目录搜索] 不应产生「未能解析」警告（use_vm={use_vm}）\n--- stderr ---\n{stderr}"
        );
    }
}

/// 目录型模块形态 3（`<dir>/<path>/<末段>.th`）同样走脚本目录：
/// `use p::thing` → `sub/p/thing.th`。
#[test]
fn script_dir_dirnamed_form_is_searchable() {
    let files: Vec<(&str, &str)> = vec![
        ("sub/main.th", "use p::thing::value\nfn main() { println(value()) }\n"),
        ("sub/p/thing.th", "fn value() -> i64 { 7 }\n"),
    ];
    let (code, stdout, stderr) = run_prog(&files, "sub/main.th", true);
    assert!(
        code == 0 && stdout.contains("7"),
        "目录型模块应经脚本目录解析: exit={code} stdout={stdout:?}\n--- stderr ---\n{stderr}"
    );
}

// ══════════════════════════════════════════════════════════════════
// 2. 子 lowerer 前置「被加载模块文件所在目录」（AUDIT-11.4.60(a′)）
// ══════════════════════════════════════════════════════════════════

/// `sub/b.th`（被 `use b::…` 加载）内部再 `use helper::greet`——
/// `helper.th` 与 `b.th` 同目录。修复前子 lowerer 的搜索路径里没有 `sub/`，
/// 该内部 use 只能落空转警告。
#[test]
fn module_can_import_sibling_in_same_dir_both_paths() {
    let files: Vec<(&str, &str)> = vec![
        ("sub/main.th", "use b::twice\nfn main() { println(twice(21)) }\n"),
        ("sub/b.th", "use helper::greet\nfn twice(x: i64) -> i64 { greet() + x }\n"),
        ("sub/helper.th", "fn greet() -> i64 { 21 }\n"),
    ];
    for use_vm in [true, false] {
        let (code, stdout, stderr) = run_prog(&files, "sub/main.th", use_vm);
        assert_eq!(
            code, 0,
            "[同级模块导入] 应可运行（use_vm={use_vm}）\n--- stdout ---\n{stdout}\n--- stderr ---\n{stderr}"
        );
        assert!(
            stdout.contains("42"),
            "[同级模块导入] stdout 应含 42（21+21，use_vm={use_vm}）\n--- stdout ---\n{stdout}\n--- stderr ---\n{stderr}"
        );
        assert!(
            !stderr.contains("未能解析"),
            "[同级模块导入] 不应产生「未能解析」警告（use_vm={use_vm}）\n--- stderr ---\n{stderr}"
        );
    }
}

// ══════════════════════════════════════════════════════════════════
// 3. 优先级：脚本目录优先（遮蔽）+ 多处命中响亮警告
// ══════════════════════════════════════════════════════════════════

/// 脚本目录里的 `helper` 与 cwd 里的 `helper` 同名：
/// - **采用脚本目录的那个**（打印 10 而非 99）；
/// - 且必须**响亮**：stderr 列出全部候选与最终采用者（不是静默取第一个）。
#[test]
fn script_dir_wins_and_ambiguity_is_loud() {
    let files: Vec<(&str, &str)> = vec![
        ("sub/main.th", "use helper::greet\nfn main() { println(greet()) }\n"),
        ("sub/helper.th", "fn greet() -> i64 { 10 }\n"),
        ("helper.th", "fn greet() -> i64 { 99 }\n"),
    ];
    let (code, stdout, stderr) = run_prog(&files, "sub/main.th", true);
    assert_eq!(code, 0, "exit={code}\n--- stderr ---\n{stderr}");
    assert!(
        stdout.contains("10") && !stdout.contains("99"),
        "[脚本目录优先] 应采用脚本目录的 helper（10），实际 stdout={stdout:?}"
    );
    // 响亮警告：候选列出 + 采用者标明 + 计数
    assert!(
        stderr.contains("多个搜索目录"),
        "[多处命中] stderr 应含响亮警告标题\n--- stderr ---\n{stderr}"
    );
    assert!(
        stderr.contains("候选"),
        "[多处命中] stderr 应列出候选数\n--- stderr ---\n{stderr}"
    );
    assert!(
        stderr.contains("最终采用"),
        "[多处命中] stderr 应标明最终采用者\n--- stderr ---\n{stderr}"
    );
    assert!(
        stderr.contains("sub") && stderr.contains("helper.th"),
        "[多处命中] 候选列表应包含两个候选文件路径\n--- stderr ---\n{stderr}"
    );
}

/// 零误报：**只命中一处**时不得出现歧义警告，也不得出现「未能解析」警告。
#[test]
fn single_hit_is_not_loud() {
    let files: Vec<(&str, &str)> = vec![
        ("sub/main.th", "use helper::greet\nfn main() { println(greet()) }\n"),
        ("sub/helper.th", "fn greet() -> i64 { 5 }\n"),
    ];
    for use_vm in [true, false] {
        let (code, stdout, stderr) = run_prog(&files, "sub/main.th", use_vm);
        assert_eq!(code, 0, "exit={code}\n--- stderr ---\n{stderr}");
        assert!(stdout.contains('5'), "stdout={stdout}");
        assert!(
            !stderr.contains("多个搜索目录") && !stderr.contains("未能解析"),
            "[零误报] 单一命中不得报歧义/未解析（use_vm={use_vm}）\n--- stderr ---\n{stderr}"
        );
    }
}

/// 脚本目录 == cwd（最常见的 `tenth.exe run ./x/main.th`）时，
/// 同一文件被列两次不得算作「多处命中」（假歧义会淹没真警告）。
#[test]
fn script_dir_equal_cwd_is_not_false_ambiguity() {
    let files: Vec<(&str, &str)> = vec![
        ("main.th", "use helper::greet\nfn main() { println(greet()) }\n"),
        ("helper.th", "fn greet() -> i64 { 3 }\n"),
    ];
    let (code, stdout, stderr) = run_prog(&files, "main.th", true);
    assert_eq!(code, 0, "exit={code}\n--- stderr ---\n{stderr}");
    assert!(stdout.contains('3'), "stdout={stdout}");
    assert!(
        !stderr.contains("多个搜索目录"),
        "[假歧义] 脚本目录与 cwd 相同不得报警\n--- stderr ---\n{stderr}"
    );
}

// ══════════════════════════════════════════════════════════════════
// 4. 零回归：脚本目录入表不改变 cwd 相对路径的既有解析
// ══════════════════════════════════════════════════════════════════

/// `import_order_test` 的形态（cwd 相对路径 `p::m1` / `q::shared`）在 cwd 与
/// 脚本目录**不同目录**时仍必须解析到 cwd 下的文件（脚本目录优先只影响
/// 「脚本目录里确实有同名文件」的情形）。
#[test]
fn cwd_relative_use_still_resolves_when_cwd_differs() {
    let files: Vec<(&str, &str)> = vec![
        ("sub/main.th", "use q::shared::helper\nfn main() { println(helper()) }\n"),
        ("q/shared.th", "fn helper() -> i64 { 11 }\n"),
    ];
    for use_vm in [true, false] {
        let (code, stdout, stderr) = run_prog(&files, "sub/main.th", use_vm);
        assert_eq!(
            code, 0,
            "[cwd 零回归] exit={code}（use_vm={use_vm}）\n--- stdout ---\n{stdout}\n--- stderr ---\n{stderr}"
        );
        assert!(
            stdout.contains("11"),
            "[cwd 零回归] stdout 应含 11（use_vm={use_vm}）\n--- stdout ---\n{stdout}\n--- stderr ---\n{stderr}"
        );
        assert!(!stderr.contains("多个搜索目录"), "不得报假歧义: {stderr}");
    }
}
