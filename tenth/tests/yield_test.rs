//! `yield` 关键字全链路测试（基本功核查第 93 项）。
//!
//! 验证 `yield` / `yield expr` 经过 parser → HIR → bytecode → VM 全链路可用，
//! 且解释器路径正确报错（yield 是 VM 调度器独有能力）。
//!
//! 设计说明：
//! - VM 测试只验证"编译通过 + VM 执行不 panic"，不验证 yield 的实际让出控制权行为
//!   （那是 VM 调度器测试范围，见 async_basic_test.rs）。
//! - yield 让出后调度器把 task_id 推回 ready_queue 尾部，单任务场景下会立即恢复，
//!   不会卡死。

use tenth::lexer::lexer::Lexer;
use tenth::parser::parser::Parser;
use tenth::hir::lower::Lowerer;
use tenth::runtime::vm::Vm;
use tenth::runtime::value::Value;
use tenth::compile::bytecode::BytecodeCompiler;
use tenth::runtime::interpreter::Interpreter;
use tenth::error::TenthError;

/// 通过 VM 执行源码：lexer → parser → HIR → bytecode → VM。
/// 不注册任何 native（测试源码不依赖 print 等）。
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

/// 通过解释器执行源码：lexer → parser → HIR → Interpreter。
fn run_interpreter(src: &str) -> Result<Value, TenthError> {
    let mut lexer = Lexer::new(src);
    let tokens = lexer.tokenize()?;
    let mut parser = Parser::new(tokens);
    let program = parser.parse_program()?;
    let mut lowerer = Lowerer::new();
    let hir = lowerer.lower_program(&program)?;
    let mut interp = Interpreter::new(&hir);
    interp.deadline_ms = None;
    match interp.execute_program(&hir)? {
        Some(v) => Ok(v),
        None => Ok(Value::Unit),
    }
}

// ─── 1. `yield;`（无值）能解析+编译+VM 执行 ─────────────────────────────

#[test]
fn test_yield_parses_and_compiles() {
    let src = r#"
        fn gen() {
            yield;
        }
        fn main() {
            gen();
        }
    "#;
    let result = run_vm(src);
    assert!(result.is_ok(), "yield; should parse+compile+run via VM: {:?}", result.err());
    // main 返回 Unit
    assert!(matches!(result.unwrap(), Value::Unit));
}

// ─── 2. `yield expr;`（带值）能解析+编译+VM 执行 ─────────────────────────

#[test]
fn test_yield_with_value() {
    let src = r#"
        fn gen() {
            yield 42;
        }
        fn main() {
            gen();
        }
    "#;
    let result = run_vm(src);
    assert!(result.is_ok(), "yield 42; should parse+compile+run via VM: {:?}", result.err());
    assert!(matches!(result.unwrap(), Value::Unit));
}

// ─── 3. yield 在 main 中直接执行（单任务场景，让出后立即恢复）─────────────

#[test]
fn test_yield_in_main_vm() {
    let src = r#"
        fn main() {
            yield;
        }
    "#;
    let result = run_vm(src);
    assert!(result.is_ok(), "yield in main should not hang: {:?}", result.err());
    assert!(matches!(result.unwrap(), Value::Unit));
}

// ─── 4. yield 在 main 中带值执行（验证 inner 求值后丢弃，栈平衡）──────────
//
// 注意：`yield expr` 中 expr 由 parse_unary 解析（与 await 一致），
// 所以 `yield 1 + 2` 会被解析为 `(yield 1) + 2`（Unit + Int 类型不匹配）。
// 用括号显式分组 `yield (1 + 2 * 3)` 让整个算术表达式作为 yield 的 inner。

#[test]
fn test_yield_with_value_in_main_vm() {
    let src = r#"
        fn main() {
            yield (1 + 2 * 3);
        }
    "#;
    let result = run_vm(src);
    assert!(result.is_ok(), "yield expr in main should not hang: {:?}", result.err());
    assert!(matches!(result.unwrap(), Value::Unit));
}

// ─── 5. 解释器路径调用 yield 必须报错 ─────────────────────────────────────

#[test]
fn test_yield_in_interpreter_errors() {
    let src = r#"
        fn main() {
            yield;
        }
    "#;
    let result = run_interpreter(src);
    assert!(result.is_err(), "interpreter should reject yield");
    let err = result.unwrap_err();
    match err {
        TenthError::RuntimeError { message, .. } => {
            assert!(
                message.contains("yield") && message.contains("解释器"),
                "expected yield interpreter error, got: {}",
                message
            );
        }
        other => panic!("expected RuntimeError, got {:?}", other),
    }
}

// ─── 6. 解释器路径调用带值 yield 也必须报错 ───────────────────────────────

#[test]
fn test_yield_with_value_in_interpreter_errors() {
    let src = r#"
        fn main() {
            yield 42;
        }
    "#;
    let result = run_interpreter(src);
    assert!(result.is_err(), "interpreter should reject yield with value");
    let err = result.unwrap_err();
    match err {
        TenthError::RuntimeError { message, .. } => {
            assert!(
                message.contains("yield") && message.contains("解释器"),
                "expected yield interpreter error, got: {}",
                message
            );
        }
        other => panic!("expected RuntimeError, got {:?}", other),
    }
}
