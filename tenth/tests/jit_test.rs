//! JIT compiler integration tests.
//!
//! These tests run Tenth source through the full pipeline (lex → parse →
//! lower → bytecode-compile) and then execute via the Cranelift JIT path
//! (`compile::jit::run_jit`) instead of the interpreter. They verify that
//! JIT-compiled functions produce identical results to the interpreter for
//! scalar arithmetic, control flow, and function calls.
//!
//! Autodiff recording mode is NOT tested here — `run_jit` falls back to the
//! interpreter when `vm.recording` is true, so those paths are covered by
//! the existing `vm_autodiff_test.rs` suite.

use tenth::lexer::lexer::Lexer;
use tenth::parser::parser::Parser;
use tenth::hir::lower::Lowerer;
use tenth::runtime::vm::Vm;
use tenth::runtime::value::Value;
use tenth::compile::bytecode::BytecodeCompiler;
use tenth::compile::jit;
use std::rc::Rc;
use std::cell::RefCell;

/// Run source code through the VM via the JIT path.
/// Mirrors `run_vm` from autodiff_test.rs but calls `jit::run_jit` instead
/// of `vm.call`. `run_jit` internally falls back to `vm.call` when JIT
/// compilation is not possible, so this always produces a result.
fn run_jit(src: &str) -> Result<Value, String> {
    let mut lexer = Lexer::new(src);
    let tokens = lexer.tokenize().map_err(|e| e.to_string())?;
    let mut parser = Parser::new(tokens);
    let program = parser.parse_program().map_err(|e| e.to_string())?;
    let mut lowerer = Lowerer::new();
    let hir = lowerer.lower_program(&program).map_err(|e| e.to_string())?;

    let mut vm = Vm::new();
    vm.add_native("println".into(), |_vm, args| {
        for a in args { print!("{a}"); }
        println!();
        Ok(Value::Unit)
    });
    vm.add_native("Vec::new".into(), |_vm, _args| {
        Ok(Value::Vec(Rc::new(RefCell::new(Vec::new()))))
    });

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
        jit::run_jit(&mut vm, "main").map_err(|e| e.to_string())
    } else if vm.has_fn("main") {
        jit::run_jit(&mut vm, "main").map_err(|e| e.to_string())
    } else {
        Ok(Value::Unit)
    }
}

// ── Basic arithmetic ──────────────────────────────────────────────────────

#[test]
fn test_jit_int_addition() {
    let src = "fn main() -> Int { 3 + 4 }";
    let result = run_jit(src).unwrap();
    match result {
        Value::Int(n) => assert_eq!(n, 7),
        v => panic!("expected Int(7), got {:?}", v),
    }
}

#[test]
fn test_jit_int_subtraction() {
    let src = "fn main() -> Int { 10 - 3 }";
    let result = run_jit(src).unwrap();
    match result {
        Value::Int(n) => assert_eq!(n, 7),
        v => panic!("expected Int(7), got {:?}", v),
    }
}

#[test]
fn test_jit_int_multiplication() {
    let src = "fn main() -> Int { 6 * 7 }";
    let result = run_jit(src).unwrap();
    match result {
        Value::Int(n) => assert_eq!(n, 42),
        v => panic!("expected Int(42), got {:?}", v),
    }
}

#[test]
fn test_jit_float_arithmetic() {
    let src = "fn main() -> Float { 3.5 * 2.0 }";
    let result = run_jit(src).unwrap();
    match result {
        Value::Float(f) => assert!((f - 7.0).abs() < 1e-9, "expected 7.0, got {}", f),
        v => panic!("expected Float(7.0), got {:?}", v),
    }
}

// ── Variables and locals ──────────────────────────────────────────────────

#[test]
fn test_jit_local_variables() {
    let src = r#"
        fn main() -> Int {
            let x = 10
            let y = 20
            x + y
        }
    "#;
    let result = run_jit(src).unwrap();
    match result {
        Value::Int(n) => assert_eq!(n, 30),
        v => panic!("expected Int(30), got {:?}", v),
    }
}

// ── Control flow ──────────────────────────────────────────────────────────

#[test]
fn test_jit_if_else_true() {
    let src = r#"
        fn main() -> Int {
            if 1 < 2 { 100 } else { 200 }
        }
    "#;
    let result = run_jit(src).unwrap();
    match result {
        Value::Int(n) => assert_eq!(n, 100),
        v => panic!("expected Int(100), got {:?}", v),
    }
}

#[test]
fn test_jit_if_else_false() {
    let src = r#"
        fn main() -> Int {
            if 1 > 2 { 100 } else { 200 }
        }
    "#;
    let result = run_jit(src).unwrap();
    match result {
        Value::Int(n) => assert_eq!(n, 200),
        v => panic!("expected Int(200), got {:?}", v),
    }
}

#[test]
fn test_jit_if_bool_literal_true() {
    let src = r#"
        fn main() -> Int {
            if true { 100 } else { 200 }
        }
    "#;
    let result = run_jit(src).unwrap();
    match result {
        Value::Int(n) => assert_eq!(n, 100, "if true should take then-branch"),
        v => panic!("expected Int(100), got {:?}", v),
    }
}

// ── Function calls ────────────────────────────────────────────────────────

#[test]
fn test_jit_function_call() {
    let src = r#"
        fn add(a: Int, b: Int) -> Int { a + b }
        fn main() -> Int { add(3, 4) }
    "#;
    let result = run_jit(src).unwrap();
    match result {
        Value::Int(n) => assert_eq!(n, 7),
        v => panic!("expected Int(7), got {:?}", v),
    }
}

#[test]
fn test_jit_nested_calls() {
    let src = r#"
        fn double(x: Int) -> Int { x * 2 }
        fn add(a: Int, b: Int) -> Int { a + b }
        fn main() -> Int { double(add(3, 4)) }
    "#;
    let result = run_jit(src).unwrap();
    match result {
        Value::Int(n) => assert_eq!(n, 14),
        v => panic!("expected Int(14), got {:?}", v),
    }
}
