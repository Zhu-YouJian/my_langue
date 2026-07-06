//! Stage 1+2 I/O 原语集成测试。
//!
//! 覆盖 6 个 I/O native（`eprint`/`eprintln`/`read_line`/`env_get`/`env_set`/`exit`）：
//! - `eprint(value)` → Unit（stderr 输出，无换行）
//! - `eprintln(value)` → Unit（stderr 输出，带换行）
//! - `read_line()` → Result<String>（EOF/错误返回 Err）
//! - `env_get(name: String)` → Result<String>（未找到返回 Err）
//! - `env_set(name: String, value: String)` → Unit
//! - `exit(code: i64)` → 不返回（无法在 #[test] 中测试，见 test_exit_skipped 注释）
//!
//! 使用解释器（Interpreter）执行——其 `call_named_fn`（natives.rs）已内置全部 I/O native。

use tenth::lexer::lexer::Lexer;
use tenth::parser::parser::Parser;
use tenth::hir::lower::Lowerer;
use tenth::runtime::interpreter::Interpreter;
use tenth::runtime::value::Value;

/// Run source through lexer → parser → HIR → interpreter.
fn run_code(src: &str) -> Result<Option<Value>, String> {
    let mut lexer = Lexer::new(src);
    let tokens = lexer.tokenize().map_err(|e| e.to_string())?;
    let mut parser = Parser::new(tokens);
    let program = parser.parse_program().map_err(|e| e.to_string())?;
    let mut lowerer = Lowerer::new();
    let hir = lowerer.lower_program(&program).map_err(|e| e.to_string())?;
    let mut interpreter = Interpreter::new(&hir);
    interpreter.execute_program(&hir).map_err(|e| e.to_string())
}

// ─── Test 1: eprint/eprintln 不 panic ───────────────────────────────────────

#[test]
fn test_eprint_eprintln_no_panic() {
    // 测试 eprint/eprintln 调用不 panic，输出到 stderr（不验证内容）
    let src = r#"
        eprint("hello");
        eprintln("world");
        eprint(42);
        eprintln(true);
    "#;
    let result = run_code(src);
    assert!(result.is_ok(), "eprint/eprintln 不应 panic: {:?}", result.err());
    // 所有语句以 `;` 结尾；解释器返回 Some(Unit) 或 None 均可——关键是 不 panic
}

// ─── Test 2: env_get 未找到返回 Err ─────────────────────────────────────────

#[test]
fn test_env_get_not_found_returns_err() {
    // 使用一个肯定不存在的环境变量名
    let src = r#"
        let r = env_get("TENTH_TEST_NONEXISTENT_VAR_12345");
        match r {
            Result::Err(_) => 1,
            Result::Ok(_) => 0,
        }
    "#;
    let result = run_code(src).unwrap();
    match result {
        Some(Value::Int(1)) => {}
        v => panic!("期望 Some(Int(1)) 表示 Err，got {:?}", v),
    }
}

// ─── Test 3: env_set + env_get 往返 ─────────────────────────────────────────

#[test]
fn test_env_set_then_get() {
    let src = r#"
        env_set("TENTH_TEST_IO_VAR", "hello");
        let r = env_get("TENTH_TEST_IO_VAR");
        match r {
            Result::Ok(v) => v,
            Result::Err(_) => "FAIL",
        }
    "#;
    let result = run_code(src).unwrap();
    match result {
        Some(Value::String(s)) => assert_eq!(s, "hello"),
        v => panic!("期望 Some(String(\"hello\"))，got {:?}", v),
    }
}

// ─── Test 4: read_line 返回 Result 类型 ─────────────────────────────────────

#[test]
fn test_read_line_returns_result_type() {
    // read_line 会阻塞等待 stdin 输入；仅当 stdin 不是 TTY 时才实际运行
    // （CI 环境 / 管道输入时 stdin 通常为 EOF，返回 Err("EOF")）
    use std::io::IsTerminal;
    if std::io::stdin().is_terminal() {
        eprintln!("跳过：stdin 是终端，read_line 会阻塞");
        return;
    }
    let src = r#"
        let r = read_line();
        match r {
            Result::Ok(_) => 1,
            Result::Err(_) => 0,
        }
    "#;
    let result = run_code(src);
    assert!(result.is_ok(), "read_line 不应 panic: {:?}", result.err());
    match result.unwrap() {
        Some(Value::Int(n)) => assert!(n == 0 || n == 1, "期望 0 或 1，got {}", n),
        v => panic!("期望 Some(Int(0|1))，got {:?}", v),
    }
}

// ─── Test 5: exit 不测试（会终止进程）────────────────────────────────────────
//
// exit 会调用 std::process::exit，无法在 #[test] 中测试——它会终止整个测试进程，
// 导致 cargo test 无法收集后续测试结果。因此此处不编写测试用例。
// 如需验证 exit 行为，应通过子进程方式（如 std::process::Command 启动独立进程，
// 检查退出码），这超出了本测试文件的范围。
