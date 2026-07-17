//! 子进程原语集成测试。
//!
//! 覆盖 4 个 subprocess native：
//! - `command_new(program: String)` → `Result<i64>`（返回 1-based handle）
//! - `command_arg(cmd_handle: i64, arg: String)` → `Unit`
//! - `command_run(cmd_handle: i64)` → `Result<i64>`（返回 exit code）
//! - `command_output(cmd_handle: i64)` → `Result<String>`（消费 handle，再次调用返回 Err）
//!
//! 跨平台适配：Windows 上 `echo` 是 cmd 内建，需用 `cmd /c echo hello`。
//! Unix 上直接用 `echo hello`。

use tenth::hir::lower::Lowerer;
use tenth::lexer::lexer::Lexer;
use tenth::parser::parser::Parser;
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

// ─── Test 1: command_new 成功返回句柄 ─────────────────────────────────────

#[test]
fn test_command_new_success() {
    let prog = if cfg!(windows) { "cmd" } else { "echo" };
    let src = format!(
        r#"
        let r = command_new("{prog}");
        match r {{
            Result::Ok(h) => h,
            Result::Err(_) => -1,
        }}
        "#
    );
    let result = run_code(&src).unwrap();
    match result {
        Some(Value::Int(h, _)) if h >= 1 => {}
        v => panic!("期望 Some(Int(>=1)) 表示 command_new 成功，got {:?}", v),
    }
}

// ─── Test 2: command_output 捕获 stdout ───────────────────────────────────

#[test]
fn test_command_output_echo() {
    // Windows: cmd /c echo hello → stdout 含 "hello"
    // Unix: echo hello → stdout 含 "hello"
    let src = if cfg!(windows) {
        r#"
        let r = command_new("cmd");
        match r {
            Result::Ok(h) => {
                command_arg(h, "/c");
                command_arg(h, "echo hello");
                let out = command_output(h);
                match out {
                    Result::Ok(s) => s,
                    Result::Err(_) => "FAIL",
                }
            },
            Result::Err(_) => "FAIL",
        }
        "#
    } else {
        r#"
        let r = command_new("echo");
        match r {
            Result::Ok(h) => {
                command_arg(h, "hello");
                let out = command_output(h);
                match out {
                    Result::Ok(s) => s,
                    Result::Err(_) => "FAIL",
                }
            },
            Result::Err(_) => "FAIL",
        }
        "#
    };
    let result = run_code(src).unwrap();
    match result {
        Some(Value::String(s)) if s.contains("hello") => {}
        v => panic!("期望 Some(String(含'hello'))，got {:?}", v),
    }
}

// ─── Test 3: command_run 返回退出码 0 ─────────────────────────────────────

#[test]
fn test_command_run_exit_zero() {
    let src = if cfg!(windows) {
        r#"
        let r = command_new("cmd");
        match r {
            Result::Ok(h) => {
                command_arg(h, "/c");
                command_arg(h, "echo hello");
                let run_r = command_run(h);
                match run_r {
                    Result::Ok(code) => code,
                    Result::Err(_) => -99,
                }
            },
            Result::Err(_) => -1,
        }
        "#
    } else {
        r#"
        let r = command_new("echo");
        match r {
            Result::Ok(h) => {
                command_arg(h, "hello");
                let run_r = command_run(h);
                match run_r {
                    Result::Ok(code) => code,
                    Result::Err(_) => -99,
                }
            },
            Result::Err(_) => -1,
        }
        "#
    };
    let result = run_code(src).unwrap();
    match result {
        Some(Value::Int(0, _)) => {}
        v => panic!("期望 Some(Int(0)) 表示退出码 0，got {:?}", v),
    }
}

// ─── Test 4: command_output 消费句柄，二次调用返回 Err ────────────────────

#[test]
fn test_command_output_consumes_handle() {
    let src = if cfg!(windows) {
        r#"
        let r = command_new("cmd");
        match r {
            Result::Ok(h) => {
                command_arg(h, "/c");
                command_arg(h, "echo hello");
                let _ = command_output(h);
                let out2 = command_output(h);
                match out2 {
                    Result::Ok(_) => 0,
                    Result::Err(_) => 1,
                }
            },
            Result::Err(_) => -1,
        }
        "#
    } else {
        r#"
        let r = command_new("echo");
        match r {
            Result::Ok(h) => {
                command_arg(h, "hello");
                let _ = command_output(h);
                let out2 = command_output(h);
                match out2 {
                    Result::Ok(_) => 0,
                    Result::Err(_) => 1,
                }
            },
            Result::Err(_) => -1,
        }
        "#
    };
    let result = run_code(src).unwrap();
    match result {
        Some(Value::Int(1, _)) => {}
        v => panic!("期望 Some(Int(1)) 表示二次 output 返回 Err，got {:?}", v),
    }
}

// ─── Test 5: 无效句柄返回 Err ─────────────────────────────────────────────

#[test]
fn test_command_invalid_handle() {
    let src = r#"
        let out = command_output(999);
        match out {
            Result::Ok(_) => 0,
            Result::Err(_) => 1,
        }
    "#;
    let result = run_code(src).unwrap();
    match result {
        Some(Value::Int(1, _)) => {}
        v => panic!("期望 Some(Int(1)) 表示无效句柄返回 Err，got {:?}", v),
    }
}
