//! M5.2 大规模回归套件：Tenth实例/ 全部实例批量守护（固化到自动化测试）
//!
//! 背景：`Tenth实例/` 62 目录 68 个 .th 实例此前仅靠手动扫描验证，
//! 本次（M5.2）固化为自动化测试：每次 cargo test 自动重跑全部实例，
//! 任何实例"崩溃 / 回归 / 从预期拦截变成静默通过"都会立即暴露。
//!
//! 分类（基于 2026-08-04 M5.2 基线实测）：
//!   - 默认（正常）：默认路径（VM/JIT）exit==0 且 stderr 无 panic
//!   - EXPECT_COMPILE_FAIL：编译期预期拦截（typestate 非法状态），exit!=0 且 stderr 无 panic
//!   - VM_GAP：VM 后端缺口（bintree/queue），VM 路径运行时错误（非 panic），
//!     解释器路径（TENTH_NO_VM=1）必须 exit==0 —— 守护"解释器兜底可运行"
//!   - KNOWN_JIT_PANIC_STDERR：已知 JIT 低化 panic（AUDIT-11.4.43），功能靠
//!     catch_unwind fallback 正确（exit==0 + 输出正确），stderr 有 panic 噪音容忍
//!
//! 约定：新增实例默认按「正常」守护；新增"预期拦截"实例必须加入对应清单。

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::mpsc;
use std::time::Duration;

/// 被测二进制（cargo test 自动构建）。
const TENTH_EXE: &str = env!("CARGO_BIN_EXE_tenth");
/// tenth/ 包根。
const MANIFEST_DIR: &str = env!("CARGO_MANIFEST_DIR");

/// 仓库根（= tenth/ 的上一级，Tenth实例 所在处）。
fn repo_root() -> PathBuf {
    Path::new(MANIFEST_DIR)
        .parent()
        .expect("CARGO_MANIFEST_DIR 的 parent 应为仓库根")
        .to_path_buf()
}

/// 实例根目录。
fn instances_dir() -> PathBuf {
    repo_root().join("Tenth实例")
}

// ── 分类清单（文件名 = 实例 .th 的 last component）─────────────

/// 编译期预期拦截（非法程序示例，exit!=0 且无 panic 为预期）。
const EXPECT_COMPILE_FAIL: &[&str] = &[
    "typestate_illegal.th",
    "typestate_arg_illegal.th",
];

/// VM 后端缺口（VM 路径运行时错误，非 panic；解释器路径必须可运行）。
const VM_GAP: &[&str] = &[
    "bintree.th",
    "queue.th",
];

/// 已知 JIT 低化 panic（AUDIT-11.4.43）：stderr 有 panic 噪音，功能靠 fallback 正确。
const KNOWN_JIT_PANIC_STDERR: &[&str] = &[
    "union_demo.th",
];

fn classify(file_name: &str) -> &'static str {
    if EXPECT_COMPILE_FAIL.contains(&file_name) {
        "expect-compile-fail"
    } else if VM_GAP.contains(&file_name) {
        "vm-gap"
    } else if KNOWN_JIT_PANIC_STDERR.contains(&file_name) {
        "known-jit-panic"
    } else {
        "normal"
    }
}

/// 运行 `tenth.exe run <file>`，带超时，返回 (exit_code, stdout, stderr)。
fn run_with_timeout(bin: &Path, file: &Path, cwd: &Path, no_vm: bool, timeout: Duration) -> (i32, String, String) {
    let mut cmd = Command::new(bin);
    cmd.arg("run").arg(file).current_dir(cwd);
    if no_vm {
        cmd.env("TENTH_NO_VM", "1");
    }
    let mut child = cmd
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .expect("spawn tenth.exe 失败");

    // 超时 kill：主线程等待 + 计时线程竞争
    let (tx, rx) = mpsc::channel();
    let pid = child.id();
    std::thread::spawn(move || {
        let _ = tx.send(());
        // 不在这里 sleep——超时由主流程用 recv_timeout 处理
        let _ = pid;
    });
    let _ = rx.recv_timeout(timeout).unwrap_or(());

    // 简单等待：轮询 child.try_wait，超时 kill
    let deadline = std::time::Instant::now() + timeout;
    loop {
        if let Some(status) = child.try_wait().expect("wait 失败") {
            let code = status.code().unwrap_or(-1);
            let out = child.wait_with_output();
            match out {
                Ok(o) => {
                    return (code,
                        String::from_utf8_lossy(&o.stdout).to_string(),
                        String::from_utf8_lossy(&o.stderr).to_string());
                }
                Err(_) => return (code, String::new(), String::new()),
            }
        }
        if std::time::Instant::now() >= deadline {
            let _ = child.kill();
            let _ = child.wait();
            return (-1, String::new(), "TIMEOUT_KILLED".to_string());
        }
        std::thread::sleep(Duration::from_millis(20));
    }
}

fn has_panic(stderr: &str) -> bool {
    stderr.contains("panicked") || stderr.contains("stack overflow")
}

/// 收集全部实例 .th 文件（相对仓库根的路径）。
fn collect_instances() -> Vec<PathBuf> {
    let dir = instances_dir();
    assert!(dir.exists(), "实例目录不存在: {}", dir.display());
    let mut files: Vec<PathBuf> = Vec::new();
    fn walk(d: &Path, out: &mut Vec<PathBuf>) {
        if let Ok(entries) = std::fs::read_dir(d) {
            for e in entries.flatten() {
                let p = e.path();
                if p.is_dir() {
                    walk(&p, out);
                } else if p.extension().map(|x| x == "th").unwrap_or(false) {
                    out.push(p);
                }
            }
        }
    }
    walk(&dir, &mut files);
    files.sort();
    files
}

/// 单一实例守护：按分类断言。
fn guard_instance(rel: &str, file_name: &str) {
    let root = repo_root();
    let file = root.join(rel);
    let timeout = Duration::from_secs(120);
    let kind = classify(file_name);

    // 默认路径（VM/JIT）
    let (code, _stdout, stderr) = run_with_timeout(
        Path::new(TENTH_EXE), &file, &root, false, timeout);

    match kind {
        "normal" => {
            assert_eq!(code, 0,
                "[{rel}] 正常实例应 exit 0，实际 {code}\nstderr: {stderr}");
            assert!(!has_panic(&stderr),
                "[{rel}] 正常实例不应 panic\nstderr: {stderr}");
        }
        "expect-compile-fail" => {
            assert_ne!(code, 0,
                "[{rel}] 预期编译拦截实例应 exit!=0，实际 0（回归：拦截失效？）");
            assert!(!has_panic(&stderr),
                "[{rel}] 编译拦截应为正常编译错误而非 panic\nstderr: {stderr}");
        }
        "vm-gap" => {
            // VM 缺口：VM 路径可 exit!=0（运行时错误，非 panic）
            assert!(!has_panic(&stderr),
                "[{rel}] VM 缺口不应 panic\nstderr: {stderr}");
            // 解释器路径必须可运行
            let (icode, _o, istd) = run_with_timeout(
                Path::new(TENTH_EXE), &file, &root, true, timeout);
            assert_eq!(icode, 0,
                "[{rel}] VM 缺口实例解释器路径应 exit 0，实际 {icode}\nstderr: {istd}");
            assert!(!has_panic(&istd),
                "[{rel}] 解释器路径不应 panic\nstderr: {istd}");
        }
        "known-jit-panic" => {
            // 功能正确（fallback），容忍 stderr panic 噪音（AUDIT-11.4.43）
            assert_eq!(code, 0,
                "[{rel}] 已知 JIT panic 实例功能应 exit 0（fallback 兜底），实际 {code}");
            let (_, stdout, _) = run_with_timeout(
                Path::new(TENTH_EXE), &file, &root, false, timeout);
            assert!(stdout.contains("100"),
                "[{rel}] 已知 JIT panic 实例输出应含 '100'（功能正确），实际: {stdout}");
        }
        other => panic!("未知分类: {other}"),
    }
}

#[test]
fn instance_batch_all_guard() {
    let files = collect_instances();
    assert!(
        files.len() >= 68,
        "实例数量异常：期望 >=68，实际 {}（实例目录被移动/删除？）", files.len()
    );

    let mut failures: Vec<String> = Vec::new();
    for f in &files {
        let rel = f.strip_prefix(&repo_root()).unwrap_or(f).to_string_lossy().to_string();
        let file_name = f.file_name().unwrap_or_default().to_string_lossy().to_string();
        // 用子测试捕获：单实例失败不中断整体
        let res = std::panic::catch_unwind(|| guard_instance(&rel, &file_name));
        if let Err(p) = res {
            let msg = if let Some(s) = p.downcast_ref::<String>() {
                s.clone()
            } else if let Some(s) = p.downcast_ref::<&str>() {
                s.to_string()
            } else {
                "未知断言失败".to_string()
            };
            failures.push(format!("[{rel}] {msg}"));
        }
    }

    assert!(
        failures.is_empty(),
        "{} 个实例守护失败：\n{}",
        failures.len(),
        failures.join("\n")
    );
}

/// 分类清单自检：清单中的文件名必须真实存在于实例目录（防拼写漂移）。
#[test]
fn instance_classify_lists_are_valid() {
    let files = collect_instances();
    let names: Vec<String> = files
        .iter()
        .map(|f| f.file_name().unwrap_or_default().to_string_lossy().to_string())
        .collect();
    for name in EXPECT_COMPILE_FAIL.iter().chain(VM_GAP).chain(KNOWN_JIT_PANIC_STDERR) {
        assert!(names.iter().any(|n| n == name),
            "分类清单引用了不存在的实例文件: {name}");
    }
}
