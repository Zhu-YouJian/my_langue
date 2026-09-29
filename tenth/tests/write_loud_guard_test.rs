//! C4 写失败响亮化守护 —— 「写失败被当成功」（假绿）的常驻防线。
//!
//! 覆盖 AUDIT 条目：
//! - `11.4.101` `save_weights` 写失败静默 ⇒ checkpoint 静默丢失（VM + 解释器 + WASM host）
//! - `11.4.102` `compile_host`/`compile_program` 写失败仍返回 `0` ⇒ 自举 `[OK]` 不是落盘证据
//! - `11.4.103` WASM host `write_file` 丢错（此处只守护「写失败必须响亮」；沙箱缺口见报告）
//! - `11.4.104` `write_bytes` 跨后端实参顺序与元素解包
//!
//! ## 设计要点
//! 1. **失败注入跨平台必然失败**：一律用「**父目录不存在**」或「**目标是目录**」——
//!    **不依赖本机沙箱**（否则 Linux/macOS CI 上会假绿，见 brief §三）。
//! 2. **两路径都测**：默认路径（VM/JIT）与 `TENTH_NO_VM=1`（解释器）各 spawn 一次，
//!    断言 stdout/stderr/退出码三轴（`env!("CARGO_BIN_EXE_tenth")`，不硬编码 target 路径）。
//! 3. **路径全部相对 cwd**（cwd 钉在 `CARGO_MANIFEST_DIR`），临时文件只落
//!    `tenth/target/**`（与 cwd 无关、不进 Git、不写本机绝对路径）。
//! 4. **反向守护**：成功路径（写合法路径）必须仍然成功且**内容正确**——写失败响亮化
//!    不许把正常落盘一起改坏（`write_bytes` 曾静默写 0 字节文件的旧缺陷同处守护）。
//! 5. **不让本机时区/编码/沙箱影响判据**：只断言「非零退出 + stderr 关键字」与
//!    「产物字节内容」，不比对报错全文（VM 与解释器的前缀本就不同）。

use std::path::{Path, PathBuf};
use std::process::Command;

/// 被测二进制（同 crate 的 bin target，`cargo test` 会自动构建）。
const TENTH_EXE: &str = env!("CARGO_BIN_EXE_tenth");
/// 包根目录（= `tenth/`）：子进程 cwd 与临时目录推导基准。
const TENTH_DIR: &str = env!("CARGO_MANIFEST_DIR");
/// 临时语料目录（相对 cwd = `tenth/`，同时用于子进程内的相对路径）。
const REL_DIR: &str = "target/tmp_write_loud_guard";

// ════════════════════════════════════════════════════════════════════
// 子进程运行三轴
// ════════════════════════════════════════════════════════════════════

struct Run {
    exit: Option<i32>,
    stdout: String,
    stderr: String,
}

impl Run {
    fn dump(&self) -> String {
        format!(
            "exit={:?}\n  --- stdout ---\n{}\n  --- stderr ---\n{}",
            self.exit, self.stdout, self.stderr
        )
    }
}

fn scratch() -> PathBuf {
    let dir = Path::new(TENTH_DIR).join(REL_DIR);
    std::fs::create_dir_all(&dir)
        .unwrap_or_else(|e| panic!("创建临时目录失败 {}：{}", dir.display(), e));
    dir
}

/// 写一份 `.th` 语料（写在 `tenth/target/**`，源码内只用相对路径）。
fn write_case(slug: &str, src: &str) -> PathBuf {
    let file = scratch().join(format!("{slug}.th"));
    std::fs::write(&file, src).unwrap_or_else(|e| panic!("写入语料失败 {}：{}", file.display(), e));
    file
}

/// 以 `cwd = tenth/`（或指定 cwd）运行一份 `.th`。`interpreter=true` 走 `TENTH_NO_VM=1`。
fn run_in(file: &Path, interpreter: bool, cwd: &Path) -> Run {
    let mut cmd = Command::new(TENTH_EXE);
    cmd.arg("run").arg(file).current_dir(cwd);
    if interpreter {
        cmd.env("TENTH_NO_VM", "1");
    } else {
        // main.rs 用 `var("TENTH_NO_VM").is_ok()` 判定——设成 "0" 也算已设置，
        // 必须显式移除，否则「默认路径」被静默跑成解释器（假绿）。
        cmd.env_remove("TENTH_NO_VM");
    }
    let out = cmd
        .output()
        .unwrap_or_else(|e| panic!("spawn 被测二进制失败：{}（{}）", TENTH_EXE, e));
    Run {
        exit: out.status.code(),
        stdout: String::from_utf8_lossy(&out.stdout).to_string(),
        stderr: String::from_utf8_lossy(&out.stderr).to_string(),
    }
}

/// 两条路径都跑一遍（同一份语料文件），返回 `(默认路径, 解释器路径)`。
fn run_both(file: &Path) -> (Run, Run) {
    (
        run_in(file, false, Path::new(TENTH_DIR)),
        run_in(file, true, Path::new(TENTH_DIR)),
    )
}

/// 「响亮失败」判据：非零退出 + stderr 含关键字；stdout 为空（不许先静默输出）。
fn assert_loud(label: &str, run: &Run, needles: &[&str]) {
    assert!(
        matches!(run.exit, Some(code) if code != 0),
        "{label}：应非零退出（写失败不许当成功）{}",
        run.dump()
    );
    assert!(
        run.stdout.is_empty(),
        "{label}：stdout 应为空（失败前不许有静默输出）{}",
        run.dump()
    );
    for needle in needles {
        assert!(
            run.stderr.contains(needle),
            "{label}：stderr 应含 {:?}（写失败必须说明原因）{}",
            needle,
            run.dump()
        );
    }
}

/// 两路径同源对拍：stdout 与退出码必须一致（错误文案前缀不同，故只比对 stderr 的**非空标志**）。
fn assert_axes_equal(d: &Run, i: &Run) {
    assert_eq!(
        d.stdout, i.stdout,
        "跨路径 stdout 不一致\n默认路径：\n{}\n解释器路径：\n{}",
        d.dump(),
        i.dump()
    );
    assert_eq!(d.exit, i.exit, "跨路径退出码不一致\n默认：{}\n解释器：{}", d.dump(), i.dump());
    assert_eq!(
        !d.stderr.is_empty(),
        !i.stderr.is_empty(),
        "跨路径 stderr 非空标志不一致（一侧响亮、一侧静默）\n默认：{}\n解释器：{}",
        d.dump(),
        i.dump()
    );
}

fn artifact(name: &str) -> PathBuf {
    scratch().join(name)
}

fn remove_artifact(name: &str) {
    let _ = std::fs::remove_file(artifact(name));
}

// ════════════════════════════════════════════════════════════════════
// AUDIT-11.4.101：save_weights 写失败必须响亮（checkpoint 不许静默丢）
// ════════════════════════════════════════════════════════════════════

#[test]
fn save_weights_missing_parent_is_loud_on_both_paths() {
    // 失败注入：「父目录不存在」——跨平台必然失败，不依赖本机沙箱。
    let file = write_case(
        "save_fail",
        r#"
fn main() {
    let mut v = Vec::new();
    v.push(zeros(2, 2));
    save_weights("target/tmp_write_loud_guard/no_such_dir/a.thw", v);
}
"#,
    );
    remove_artifact("no_such_dir/a.thw");
    let (d, i) = run_both(&file);
    assert_axes_equal(&d, &i);
    assert_loud("save_weights/默认路径", &d, &["save_weights", "失败"]);
    assert_loud("save_weights/解释器路径", &i, &["save_weights", "失败"]);
    assert!(
        !artifact("no_such_dir/a.thw").exists(),
        "写失败时不该留下产物：{}",
        artifact("no_such_dir/a.thw").display()
    );
}

#[test]
fn save_weights_success_path_still_writes_artifact() {
    // 反向守护：合法路径必须仍然成功，且产物真的是 .thw（magic "THW1"）。
    let file = write_case(
        "save_ok",
        r#"
fn main() {
    let mut v = Vec::new();
    v.push(zeros(2, 2));
    save_weights("target/tmp_write_loud_guard/ok.thw", v);
}
"#,
    );
    for interpreter in [false, true] {
        remove_artifact("ok.thw");
        let run = run_in(&file, interpreter, Path::new(TENTH_DIR));
        let label = if interpreter { "解释器路径" } else { "默认路径" };
        assert_eq!(run.exit, Some(0), "{label}：成功路径应退出 0{}", run.dump());
        assert!(run.stdout.is_empty(), "{label}：成功路径不该有输出{}", run.dump());
        let bytes = std::fs::read(artifact("ok.thw"))
            .unwrap_or_else(|e| panic!("{label}：产物应已落盘（{}）{}", e, run.dump()));
        assert!(bytes.len() > 12, "{label}：产物不应为空{}", run.dump());
        assert_eq!(&bytes[..4], b"THW1", "{label}：产物 magic 应为 THW1{}", run.dump());
    }
}

// ════════════════════════════════════════════════════════════════════
// AUDIT-11.4.102：compile_host 写失败必须返回 1（0 ＝ 编译成功且写出成功）
// ════════════════════════════════════════════════════════════════════

#[test]
fn compile_host_write_failure_returns_one_not_zero() {
    let file = write_case(
        "ch_fail",
        r#"
fn main() {
    let ok = compile_host("fn main() { }", "target/tmp_write_loud_guard/no_such_dir/out.wasm");
    println("OK=" + format("{}", ok));
}
"#,
    );
    let (d, i) = run_both(&file);
    assert_axes_equal(&d, &i);
    assert_eq!(d.exit, Some(0), "compile_host 失败是返回值语义，不该中止程序{}", d.dump());
    assert_eq!(d.stdout.trim(), "OK=1", "写失败必须返回 1（0 是假绿）{}", d.dump());
    assert_eq!(i.stdout.trim(), "OK=1", "解释器路径同判据{}", i.dump());
    // 仅失败路径打一行 stderr 诊断（不得无条件 eprintln，见 AUDIT-11.4.72 教训）
    assert!(d.stderr.contains("compile_host"), "默认路径应有失败诊断{}", d.dump());
    assert!(i.stderr.contains("compile_host"), "解释器路径应有失败诊断{}", i.dump());
    assert!(!artifact("no_such_dir/out.wasm").exists(), "写失败不该留下产物");
}

#[test]
fn compile_host_success_returns_zero_and_writes_artifact() {
    // 与上例**同一份源码**：此处 0 证明上面那条 1 确实来自写失败而非编译失败。
    let file = write_case(
        "ch_ok",
        r#"
fn main() {
    let ok = compile_host("fn main() { }", "target/tmp_write_loud_guard/ok.wasm");
    println("OK=" + format("{}", ok));
}
"#,
    );
    for interpreter in [false, true] {
        remove_artifact("ok.wasm");
        let run = run_in(&file, interpreter, Path::new(TENTH_DIR));
        let label = if interpreter { "解释器路径" } else { "默认路径" };
        assert_eq!(run.exit, Some(0), "{label}{}", run.dump());
        assert_eq!(run.stdout.trim(), "OK=0", "{label}：写出成功才允许返回 0{}", run.dump());
        assert!(run.stderr.is_empty(), "{label}：成功路径不该有 stderr 诊断{}", run.dump());
        let bytes = std::fs::read(artifact("ok.wasm"))
            .unwrap_or_else(|e| panic!("{label}：产物应已落盘（{}）{}", e, run.dump()));
        assert!(!bytes.is_empty(), "{label}：产物不应为空{}", run.dump());
        assert_eq!(&bytes[..4], b"\0asm", "{label}：产物应为 wasm{}", run.dump());
    }
}

// ════════════════════════════════════════════════════════════════════
// AUDIT-11.4.104：write_bytes 三侧一致（实参顺序 + 元素解包 + 错误语义）
// ════════════════════════════════════════════════════════════════════

fn assert_write_bytes_effect(slug: &str, src: &str, out_name: &str) {
    let file = write_case(slug, src);
    let (d, i) = run_both(&file);
    assert_axes_equal(&d, &i);
    assert_eq!(d.exit, Some(0), "写合法路径应成功{}", d.dump());
    let bytes = std::fs::read(artifact(out_name))
        .unwrap_or_else(|e| panic!("产物应已落盘（{}）\n默认：{}\n解释器：{}", e, d.dump(), i.dump()));
    assert_eq!(
        bytes,
        vec![104u8, 105],
        "两路径都必须写出真实字节（旧缺陷：解释器静默写 0 字节）\n默认：{}\n解释器：{}",
        d.dump(),
        i.dump()
    );
}

/// `(路径, 字节数组)` 是权威顺序（与 `std::fs::fs::write_bytes` 声明一致）。
#[test]
fn write_bytes_path_first_literal_array_writes_real_bytes() {
    remove_artifact("lit.bin");
    assert_write_bytes_effect(
        "wb_lit",
        r#"
fn main() {
    write_bytes("target/tmp_write_loud_guard/lit.bin", [104, 105]);
}
"#,
        "lit.bin",
    );
}

#[test]
fn write_bytes_path_first_pushed_vec_writes_real_bytes() {
    remove_artifact("push.bin");
    assert_write_bytes_effect(
        "wb_push",
        r#"
fn main() {
    let v = Vec::new();
    v.push(104);
    v.push(105);
    write_bytes("target/tmp_write_loud_guard/push.bin", v);
}
"#,
        "push.bin",
    );
}

#[test]
fn stdlib_fs_write_bytes_wrapper_writes_real_bytes() {
    // 用户可见拼法 `std::fs::fs::write_bytes(path, bytes)` 必须两路径都写出真实字节。
    // 实测注记（C4）：`fs.th:69-71` 的 wrapper **体当前不可达**——`write_bytes` 与
    // 同名 native 冲突时 native 胜出（对照实验：wrapper 内插 `println` 不打印；
    // 而 `fs::write_text` 这类无同名 native 的 wrapper 正常执行，`fs::exists`/`fs::join`
    // 亦正常）⇒ 本用例钉的是**对外拼法的可观测契约**，服务它的是 native。
    // wrapper 行仍一并改为 `(path, bytes)`：与声明一致，防「同名遮蔽」将来被修好时带病上线。
    remove_artifact("wrap.bin");
    assert_write_bytes_effect(
        "wb_wrap",
        r#"
use std::fs::fs
fn main() {
    fs::write_bytes("target/tmp_write_loud_guard/wrap.bin", [104, 105]);
}
"#,
        "wrap.bin",
    );
}

#[test]
fn write_bytes_missing_parent_is_loud_on_both_paths() {
    let file = write_case(
        "wb_fail",
        r#"
fn main() {
    write_bytes("target/tmp_write_loud_guard/no_such_dir/b.bin", [1, 2]);
}
"#,
    );
    let (d, i) = run_both(&file);
    assert_axes_equal(&d, &i);
    assert_loud("write_bytes/默认路径", &d, &["写入字节失败"]);
    assert_loud("write_bytes/解释器路径", &i, &["写入字节失败"]);
    assert!(!artifact("no_such_dir/b.bin").exists(), "写失败不该留下产物");
}

#[test]
fn write_bytes_legacy_bytes_first_order_is_rejected_on_both_paths() {
    // 旧顺序 `(bytes, path)` 只在 VM 侧可用（跨后端分歧的根源）——现在**两侧同判据拒绝**，
    // 且是响亮报错而非静默返回 0/1。
    let file = write_case(
        "wb_legacy",
        r#"
fn main() {
    write_bytes([104, 105], "target/tmp_write_loud_guard/legacy.bin");
}
"#,
    );
    remove_artifact("legacy.bin");
    let (d, i) = run_both(&file);
    assert_axes_equal(&d, &i);
    assert_loud("write_bytes 旧顺序/默认路径", &d, &["write_bytes"]);
    assert_loud("write_bytes 旧顺序/解释器路径", &i, &["write_bytes"]);
    assert!(!artifact("legacy.bin").exists(), "旧顺序不该落盘");
}

// ════════════════════════════════════════════════════════════════════
// 自举直接验收：写失败必须 [FAIL]，写成功才 [OK] 且产物非空
// ════════════════════════════════════════════════════════════════════

/// 递归拷贝目录（自举需要 cwd 下有 `tenthc/` 源码树）。
fn copy_tree(from: &Path, to: &Path) {
    std::fs::create_dir_all(to).unwrap_or_else(|e| panic!("创建 {} 失败：{}", to.display(), e));
    for entry in std::fs::read_dir(from).unwrap_or_else(|e| panic!("读 {} 失败：{}", from.display(), e)) {
        let entry = entry.expect("dir entry");
        let src = entry.path();
        let dst = to.join(entry.file_name());
        if src.is_dir() {
            copy_tree(&src, &dst);
        } else {
            std::fs::copy(&src, &dst)
                .unwrap_or_else(|e| panic!("拷贝 {} 失败：{}", src.display(), e));
        }
    }
}

/// 自举源码树的仓库根（= `tenth/` 的父目录；`tenthc/` 与产物同在工作区根）。
fn repo_root() -> PathBuf {
    Path::new(TENTH_DIR)
        .parent()
        .expect("tenth/ 应有父目录（仓库根）")
        .to_path_buf()
}

#[test]
fn bootstrap_reports_fail_when_product_cannot_be_written() {
    // 注入：把产物路径 `tenthc_full.wasm` 占成**目录** ⇒ 写入跨平台必然失败。
    // 修复前：`compile_host` 丢错返回 0 ⇒ 自举打印 [OK]（假绿，产物并不存在）。
    // 修复后：返回 1 ⇒ 必须打印 [FAIL]。
    let boot = scratch().join("boot");
    let _ = std::fs::remove_dir_all(&boot);
    copy_tree(&repo_root().join("tenthc"), &boot.join("tenthc"));
    let product = boot.join("tenthc_full.wasm");
    std::fs::create_dir_all(&product).expect("把产物路径占成目录");

    let run = run_in(Path::new("tenthc/main.th"), false, &boot);
    assert!(
        !run.stdout.contains("[OK]"),
        "产物写不进去时不许报 [OK]（自举假绿）{}",
        run.dump()
    );
    assert!(
        run.stdout.contains("[FAIL]"),
        "产物写不进去时必须报 [FAIL]{}",
        run.dump()
    );
    assert!(run.stderr.contains("compile_host"), "应有一行写失败诊断{}", run.dump());

    // 还原（移除占位目录）⇒ 必须 [OK] 且产物非空（把「产物已落盘」从推断变断言）。
    std::fs::remove_dir_all(&product).expect("移除占位目录");
    let run = run_in(Path::new("tenthc/main.th"), false, &boot);
    assert!(run.stdout.contains("[OK]"), "写成功后应报 [OK]{}", run.dump());
    let bytes = std::fs::read(&product)
        .unwrap_or_else(|e| panic!("自举产物应已落盘（{}）{}", e, run.dump()));
    assert!(!bytes.is_empty(), "自举产物不应为空{}", run.dump());
    assert_eq!(&bytes[..4], b"\0asm", "自举产物应为 wasm{}", run.dump());
}

// ════════════════════════════════════════════════════════════════════
// AUDIT-11.4.103：WASM host write_file 丢错（两份手写副本：wasmi / wasmtime）
// ════════════════════════════════════════════════════════════════════

/// 把 .th 源码编译成 wasm（Rust 全栈路径：lexer → parser → HIR → wasm）。
fn compile_th_to_wasm(src: &str) -> Vec<u8> {
    use tenth::hir::lower::Lowerer;
    use tenth::lexer::lexer::Lexer;
    use tenth::parser::parser::Parser;
    let tokens = Lexer::new(src).tokenize().expect("tokenize");
    let prog = Parser::new(tokens).parse_program().expect("parse");
    let hir = Lowerer::new().lower_program(&prog).expect("lower");
    tenth::compile::compile_to_wasm(&hir).expect("compile_to_wasm")
}

const WASM_WRITE_FAIL_SRC: &str = r#"
fn main() {
    write_file("target/tmp_write_loud_guard/no_such_dir/w.txt", "hi");
}
"#;

#[test]
fn wasm_host_write_file_failure_is_loud_wasmi() {
    // wasmi 是 `tenth wasm` 的宿主；写失败必须以 trap 形式传播到宿主（旧行为：静默 no-op）。
    use tenth::compile::wasm::run_wasm_module;
    let wasm = compile_th_to_wasm(WASM_WRITE_FAIL_SRC);
    let err = run_wasm_module(&wasm).expect_err("写失败必须报错（不许静默 no-op）");
    let msg = format!("{err}");
    assert!(msg.contains("write_file"), "错误应指明 write_file：{msg}");
    assert!(!artifact("no_such_dir/w.txt").exists(), "写失败不该留下产物");
}

#[test]
fn wasm_host_write_file_failure_is_loud_wasmtime() {
    // wasmtime 是第二份手写副本（AUDIT-11.4.95 指出重复实现）——同判据。
    use tenth::compile::wasm::instantiate_wasmtime;
    let wasm = compile_th_to_wasm(WASM_WRITE_FAIL_SRC);
    let (mut store, instance) = instantiate_wasmtime(&wasm).expect("instantiate");
    let main_fn = instance
        .get_typed_func::<(), i32>(&mut store, "main")
        .expect("main");
    let err = main_fn.call(&mut store, ()).expect_err("写失败必须报错（不许静默 no-op）");
    // wasmtime 的 trap 文本把宿主错误放在错误链里，Display 只给 wasm backtrace
    // ⇒ 三种格式一起查（含 Debug 与 alternate）。
    let msg = format!("{err}|{err:?}|{err:#}");
    assert!(msg.contains("write_file"), "错误应指明 write_file：{msg}");
    assert!(!artifact("no_such_dir/w.txt").exists(), "写失败不该留下产物");
}

#[test]
fn wasm_host_write_file_success_path_still_writes() {
    // 反向守护：合法路径必须仍然写入（响亮化不许把正常路径改坏）。
    use tenth::compile::wasm::run_wasm_module;
    let wasm = compile_th_to_wasm(
        r#"
fn main() {
    write_file("target/tmp_write_loud_guard/wasm_ok.txt", "hi");
}
"#,
    );
    remove_artifact("wasm_ok.txt");
    run_wasm_module(&wasm).expect("写合法路径应成功");
    let text = std::fs::read_to_string(artifact("wasm_ok.txt")).expect("产物应已落盘");
    assert_eq!(text, "hi");
}
