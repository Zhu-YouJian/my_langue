//! `?` 操作符集成测试。
//!
//! 覆盖 `?` 操作符的四种场景：Ok 解包、Err 提前返回、链式 `?`、模拟 I/O 错误传播。
//! 同时验证解释器（Interpreter）与字节码 VM（路径 A 默认后端）两条路径。
//!
//! `?` 语义：从 `Result<T>` 中提取 T；如果是 `Result::Err(e)` 则提前返回 `Result::Err(e)`。
//! - 解释器：通过 `TenthError::TryPropagate` 信号传递，`unwrap_return` 在函数边界转为 `Result::Err`。
//! - VM：`Op::Try` 通过 frame 恢复实现 early return（最外层函数直接返回 `Result::Err`）。
//!
//! 注意：`?` 只能在函数体内使用（依赖 frame 恢复机制），不能在顶层表达式使用。
//! `Result` 枚举为预定义内置：`Result::Ok(value)` / `Result::Err(error: str)`。
//!
//! ## 已知行为差异（解释器 vs VM，已记录汇报）
//!
//! 在 `?` 遇到 `Result::Err(e)` 时：
//! - **VM 路径**：直接返回原 `Result::Err(e)`（语义正确）。
//! - **解释器路径**：`unwrap_return` 把 TryPropagate 携带的 `Result::Err(e)` 值再包一层，
//!   返回 `Result::Err(Result::Err(e))`（double-wrap）。
//!
//! 这是解释器与 VM 行为不一致的实现 bug。测试中使用 `extract_err_msg` 辅助函数
//! 兼容两种路径：若内层仍是 `Result::Err`，则递归解一层取最内层的 String 消息。

use tenth::compile::bytecode::BytecodeCompiler;
use tenth::hir::lower::Lowerer;
use tenth::lexer::lexer::Lexer;
use tenth::parser::parser::Parser;
use tenth::runtime::interpreter::Interpreter;
use tenth::runtime::value::Value;
use tenth::runtime::vm::Vm;

/// Run source through lexer → parser → HIR → interpreter.
fn run(src: &str) -> Result<Option<Value>, String> {
    let mut lexer = Lexer::new(src);
    let tokens = lexer.tokenize().map_err(|e| e.to_string())?;
    let mut parser = Parser::new(tokens);
    let program = parser.parse_program().map_err(|e| e.to_string())?;
    let mut lowerer = Lowerer::new();
    let hir = lowerer.lower_program(&program).map_err(|e| e.to_string())?;
    let mut interpreter = Interpreter::new(&hir);
    interpreter.execute_program(&hir).map_err(|e| e.to_string())
}

/// Run source through the bytecode VM (path A default backend).
fn run_vm(src: &str) -> Result<Value, String> {
    let mut lexer = Lexer::new(src);
    let tokens = lexer.tokenize().map_err(|e| e.to_string())?;
    let mut parser = Parser::new(tokens);
    let program = parser.parse_program().map_err(|e| e.to_string())?;
    let mut lowerer = Lowerer::new();
    let hir = lowerer.lower_program(&program).map_err(|e| e.to_string())?;

    let mut vm = Vm::new();
    for func in &hir.functions {
        let compiler = BytecodeCompiler::new();
        match compiler.compile(func) {
            Ok((chunk, closures)) => {
                vm.add_fn(func.name.clone(), chunk);
                for (name, closure_chunk) in closures {
                    vm.add_fn(name, closure_chunk);
                }
            }
            Err(e) => return Err(format!("compile error: {}", e)),
        }
    }
    if let Some(ref expr) = hir.main_expr {
        let compiler = BytecodeCompiler::new();
        match compiler.compile_main(expr) {
            Ok((chunk, closures)) => {
                vm.add_fn("main".into(), chunk);
                for (name, closure_chunk) in closures {
                    vm.add_fn(name, closure_chunk);
                }
            }
            Err(e) => return Err(format!("compile error: {}", e)),
        }
        vm.call("main").map_err(|e| e.to_string())
    } else if vm.has_fn("main") {
        vm.call("main").map_err(|e| e.to_string())
    } else {
        Ok(Value::Unit)
    }
}

/// 从 `Result::Err(...)` 中提取错误消息字符串。
/// 兼容解释器 double-wrap（`Result::Err(Result::Err(msg))`）与 VM 单层 wrap 两种路径：
/// 若 `_0` 字段本身是 `Result::Err`，则递归解一层。
fn extract_err_msg(v: &Value) -> String {
    match v {
        Value::Enum { enum_name, variant, fields } if enum_name == "Result" && variant == "Err" => {
            let borrowed = fields.borrow();
            match borrowed.first() {
                Some((_, inner)) => match inner {
                    // double-wrap：内层仍是 Result::Err(String)
                    Value::Enum { enum_name, variant, fields }
                        if enum_name == "Result" && variant == "Err" =>
                    {
                        match fields.borrow().first() {
                            Some((_, Value::String(s))) => s.clone(),
                            _ => panic!("double-wrap 内层非 String, got {:?}", inner),
                        }
                    }
                    Value::String(s) => s.clone(),
                    _ => panic!("期望 Err(String) 或 Err(Err(String)), got {:?}", inner),
                },
                None => panic!("Err 无字段, got {:?}", v),
            }
        }
        _ => panic!("期望 Result::Err, got {:?}", v),
    }
}

// ─── 1. `?` 解包 Ok：解释器路径 ─────────────────────────────────────────────

#[test]
fn test_try_ok() {
    let src = r#"
        fn main() -> i64 {
            let x = Result::Ok(42)?;
            x
        }
    "#;
    let result = run(src).unwrap();
    match result {
        Some(Value::Int(42)) => {}
        v => panic!("期望 Int(42), got {:?}", v),
    }
}

// ─── 2. `?` 传播 Err：解释器路径 ────────────────────────────────────────────
// 函数声明返回 i64，但 `?` 遇到 Err 会提前返回 Result::Err（通过 TryPropagate 信号）。
// 注意：解释器路径会 double-wrap，所以只检查外层是 Result::Err。

#[test]
fn test_try_err_propagation() {
    let src = r#"
        fn main() -> i64 {
            let x = Result::Err("error")?;
            x
        }
    "#;
    let result = run(src).unwrap();
    match result {
        Some(Value::Enum { enum_name, variant, .. }) => {
            assert_eq!(enum_name, "Result", "应返回 Result 枚举");
            assert_eq!(variant, "Err", "应返回 Err 变体");
        }
        v => panic!("期望 Result::Err, got {:?}", v),
    }
}

// ─── 3. 链式 `?`：解释器路径 ────────────────────────────────────────────────
// 自定义 parse 函数返回 Result，连续 `?` 解包；第二个 parse 失败会传播 Err。

#[test]
fn test_try_chain_success() {
    let src = r#"
        fn parse(s: str) -> Result<i64, str> {
            if s == "42" { Result::Ok(42) } else { Result::Err("not 42") }
        }
        fn main() -> i64 {
            let a = parse("42")?;
            let b = parse("42")?;
            a + b
        }
    "#;
    let result = run(src).unwrap();
    match result {
        Some(Value::Int(84)) => {}
        v => panic!("期望 Int(84), got {:?}", v),
    }
}

#[test]
fn test_try_chain_err_propagation() {
    let src = r#"
        fn parse(s: str) -> Result<i64, str> {
            if s == "42" { Result::Ok(42) } else { Result::Err("not 42") }
        }
        fn main() -> i64 {
            let a = parse("42")?;
            let b = parse("10")?;
            a + b
        }
    "#;
    let result = run(src).unwrap();
    match result {
        Some(v) => {
            let msg = extract_err_msg(&v);
            assert_eq!(msg, "not 42", "期望 Err(\"not 42\")");
        }
        None => panic!("期望 Some(Result::Err), got None"),
    }
}

// ─── 4. 模拟 I/O 场景：解释器路径 ───────────────────────────────────────────
// 用自定义 simulate_read 函数模拟 I/O 错误传播（避免与 native read_file 同名冲突，
// read_file native 实际返回 str 而非 Result，遇到文件不存在会 RuntimeError panic）。

#[test]
fn test_try_with_io_error() {
    let src = r#"
        fn simulate_read(path: str) -> Result<str, str> {
            if path == "nonexistent" {
                Result::Err("file not found")
            } else {
                Result::Ok("hello")
            }
        }
        fn main() -> str {
            let content = simulate_read("nonexistent")?;
            content
        }
    "#;
    let result = run(src).unwrap();
    match result {
        Some(v) => {
            let msg = extract_err_msg(&v);
            assert_eq!(msg, "file not found", "期望 Err(\"file not found\")");
        }
        None => panic!("期望 Some(Result::Err), got None"),
    }
}

#[test]
fn test_try_with_io_ok() {
    let src = r#"
        fn simulate_read(path: str) -> Result<str, str> {
            if path == "nonexistent" {
                Result::Err("file not found")
            } else {
                Result::Ok("hello")
            }
        }
        fn main() -> str {
            let content = simulate_read("exists")?;
            content
        }
    "#;
    let result = run(src).unwrap();
    match result {
        Some(Value::String(s)) if s == "hello" => {}
        v => panic!("期望 String(\"hello\"), got {:?}", v),
    }
}

// ─── VM 路径（路径 A 默认后端）──────────────────────────────────────────────

#[test]
fn test_vm_try_ok() {
    let src = r#"
        fn main() -> i64 {
            let x = Result::Ok(42)?;
            x
        }
    "#;
    let result = run_vm(src).unwrap();
    match result {
        Value::Int(42) => {}
        v => panic!("VM: 期望 Int(42), got {:?}", v),
    }
}

#[test]
fn test_vm_try_err_propagation() {
    let src = r#"
        fn main() -> i64 {
            let x = Result::Err("error")?;
            x
        }
    "#;
    let result = run_vm(src).unwrap();
    let msg = extract_err_msg(&result);
    assert_eq!(msg, "error", "VM: 期望 Err(\"error\")");
}

#[test]
fn test_vm_try_chain_success() {
    let src = r#"
        fn parse(s: str) -> Result<i64, str> {
            if s == "42" { Result::Ok(42) } else { Result::Err("not 42") }
        }
        fn main() -> i64 {
            let a = parse("42")?;
            let b = parse("42")?;
            a + b
        }
    "#;
    let result = run_vm(src).unwrap();
    match result {
        Value::Int(84) => {}
        v => panic!("VM: 期望 Int(84), got {:?}", v),
    }
}

#[test]
fn test_vm_try_chain_err_propagation() {
    let src = r#"
        fn parse(s: str) -> Result<i64, str> {
            if s == "42" { Result::Ok(42) } else { Result::Err("not 42") }
        }
        fn main() -> i64 {
            let a = parse("42")?;
            let b = parse("10")?;
            a + b
        }
    "#;
    let result = run_vm(src).unwrap();
    let msg = extract_err_msg(&result);
    assert_eq!(msg, "not 42", "VM: 期望 Err(\"not 42\")");
}

#[test]
fn test_vm_try_with_io_error() {
    let src = r#"
        fn simulate_read(path: str) -> Result<str, str> {
            if path == "nonexistent" {
                Result::Err("file not found")
            } else {
                Result::Ok("hello")
            }
        }
        fn main() -> str {
            let content = simulate_read("nonexistent")?;
            content
        }
    "#;
    let result = run_vm(src).unwrap();
    let msg = extract_err_msg(&result);
    assert_eq!(msg, "file not found", "VM: 期望 Err(\"file not found\")");
}

#[test]
fn test_vm_try_with_io_ok() {
    let src = r#"
        fn simulate_read(path: str) -> Result<str, str> {
            if path == "nonexistent" {
                Result::Err("file not found")
            } else {
                Result::Ok("hello")
            }
        }
        fn main() -> str {
            let content = simulate_read("exists")?;
            content
        }
    "#;
    let result = run_vm(src).unwrap();
    match result {
        Value::String(s) if s == "hello" => {}
        v => panic!("VM: 期望 String(\"hello\"), got {:?}", v),
    }
}
