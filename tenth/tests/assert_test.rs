//! assert / assert_eq 内置 native 行为测试。
//!
//! 验证：
//! - 成功路径：assert(true) / assert_eq(等值) 返回 Unit，不报错
//! - 失败路径：assert(false) / assert_eq(不等值) 返回 RuntimeError（Err），而非 Rust panic
//!
//! assert/assert_eq 在 VM 和解释器两侧都已注册：
//! - tenth/src/runtime/natives.rs:register_all_natives (VM 侧，行 2634/2657)
//! - tenth/src/runtime/interpreter/natives.rs:call_named_fn (解释器侧，行 2421/2442)
//!
//! 本测试通过 VM 路径验证（调用 register_all_natives 注册全部 native，
//! 确保 assert/assert_eq 可用）。解释器路径下 Var 解析阶段（eval.rs）
//! 未将 assert/assert_eq 列入 native 名单，导致解释器路径下报"未定义变量"——
//! 这是已知的解释器覆盖缺口，不影响 VM 路径的正确性。

use tenth::compile::bytecode::BytecodeCompiler;
use tenth::hir::lower::Lowerer;
use tenth::lexer::lexer::Lexer;
use tenth::parser::parser::Parser;
use tenth::runtime::natives::register_all_natives;
use tenth::runtime::value::Value;
use tenth::runtime::vm::Vm;

/// 通过 VM 执行 .th 源码，返回结果。
/// 注册全部 native（含 assert/assert_eq），参考 native_parity_test.rs 的 run_vm 模式。
fn run_vm(src: &str) -> Result<Value, String> {
    let mut lexer = Lexer::new(src);
    let tokens = lexer.tokenize().map_err(|e| e.to_string())?;
    let mut parser = Parser::new(tokens);
    let program = parser.parse_program().map_err(|e| e.to_string())?;
    let mut lowerer = Lowerer::new();
    let hir = lowerer.lower_program(&program).map_err(|e| e.to_string())?;

    let mut vm = Vm::new();
    register_all_natives(&mut vm);

    for func in &hir.functions {
        let compiler = BytecodeCompiler::new();
        match compiler.compile(func) {
            Ok((chunk, closures)) => {
                vm.add_fn(func.name.clone(), chunk);
                for (name, closure_chunk) in closures {
                    vm.add_fn(name, closure_chunk);
                }
                vm.set_global(func.name.clone(), Value::FnRef {
                    name: func.name.clone(),
                    params: func.params.clone(),
                    return_type: func.return_type.clone(),
                });
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

// ─── 成功路径：assert / assert_eq 不报错 ────────────────────────────────

#[test]
fn test_assert_true_literal() {
    // assert(true) 不 panic，返回 Unit
    let src = "assert(true)";
    let result = run_vm(src);
    assert!(result.is_ok(), "assert(true) should succeed, got {:?}", result);
    match result.unwrap() {
        Value::Unit => {}
        v => panic!("expected Unit, got {:?}", v),
    }
}

#[test]
fn test_assert_true_expression() {
    // assert(1 == 1) 不 panic
    let src = "assert(1 == 1)";
    let result = run_vm(src);
    assert!(result.is_ok(), "assert(1 == 1) should succeed, got {:?}", result);
}

#[test]
fn test_assert_eq_int_equal() {
    // assert_eq(1, 1) 不 panic
    let src = "assert_eq(1, 1)";
    let result = run_vm(src);
    assert!(result.is_ok(), "assert_eq(1, 1) should succeed, got {:?}", result);
}

#[test]
fn test_assert_eq_string_equal() {
    // assert_eq("hello", "hello") 不 panic
    let src = r#"assert_eq("hello", "hello")"#;
    let result = run_vm(src);
    assert!(result.is_ok(), r#"assert_eq("hello", "hello") should succeed, got {:?}"#, result);
}

#[test]
fn test_assert_eq_float_equal() {
    // assert_eq(1.5, 1.5) 不 panic
    let src = "assert_eq(1.5, 1.5)";
    let result = run_vm(src);
    assert!(result.is_ok(), "assert_eq(1.5, 1.5) should succeed, got {:?}", result);
}

// ─── 失败路径：assert / assert_eq 返回 Err（RuntimeError） ──────────────

#[test]
fn test_assert_false_returns_err() {
    // assert(false) 应返回 Err（RuntimeError），而非 Rust panic
    let src = "assert(false)";
    let result = run_vm(src);
    assert!(result.is_err(), "assert(false) should return Err, got {:?}", result);
    let err_msg = result.unwrap_err();
    assert!(err_msg.contains("assertion failed"), "error message should mention assertion, got: {}", err_msg);
}

#[test]
fn test_assert_eq_int_not_equal_returns_err() {
    // assert_eq(1, 2) 应返回 Err
    let src = "assert_eq(1, 2)";
    let result = run_vm(src);
    assert!(result.is_err(), "assert_eq(1, 2) should return Err, got {:?}", result);
    let err_msg = result.unwrap_err();
    assert!(err_msg.contains("1") && err_msg.contains("2"), "error should mention both values, got: {}", err_msg);
    assert!(err_msg.contains("!="), "error should mention inequality, got: {}", err_msg);
}

#[test]
fn test_assert_eq_string_not_equal_returns_err() {
    // assert_eq("a", "b") 应返回 Err
    let src = r#"assert_eq("a", "b")"#;
    let result = run_vm(src);
    assert!(result.is_err(), r#"assert_eq("a", "b") should return Err, got {:?}"#, result);
    let err_msg = result.unwrap_err();
    assert!(err_msg.contains("a") && err_msg.contains("b"), "error should mention both values, got: {}", err_msg);
}
