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
        Value::Int(n, _) => assert_eq!(n, 7),
        v => panic!("expected Int(7), got {:?}", v),
    }
}

#[test]
fn test_jit_int_subtraction() {
    let src = "fn main() -> Int { 10 - 3 }";
    let result = run_jit(src).unwrap();
    match result {
        Value::Int(n, _) => assert_eq!(n, 7),
        v => panic!("expected Int(7), got {:?}", v),
    }
}

#[test]
fn test_jit_int_multiplication() {
    let src = "fn main() -> Int { 6 * 7 }";
    let result = run_jit(src).unwrap();
    match result {
        Value::Int(n, _) => assert_eq!(n, 42),
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
        Value::Int(n, _) => assert_eq!(n, 30),
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
        Value::Int(n, _) => assert_eq!(n, 100),
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
        Value::Int(n, _) => assert_eq!(n, 200),
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
        Value::Int(n, _) => assert_eq!(n, 100, "if true should take then-branch"),
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
        Value::Int(n, _) => assert_eq!(n, 7),
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
        Value::Int(n, _) => assert_eq!(n, 14),
        v => panic!("expected Int(14), got {:?}", v),
    }
}

// ── Loop fallback to VM ───────────────────────────────────────────────────
//
// 盲点修复：此前 jit_test.rs 只有 if/else 和函数调用测试，完全没有循环测试。
// Cranelift JIT translator 对循环回边（while/for/do-while/loop）会触发
// is_sealed panic（leader block 密封策略不支持回边）。context.rs:43-63 用
// catch_unwind 捕获该 panic，转为 Err，mod.rs:62-65 降级到 vm.call 执行。
// 这些测试验证：含循环的函数经 run_jit 仍能返回正确结果（不 crash）。

#[test]
fn test_jit_while_loop_fallback_to_vm() {
    // sum_loop(10) = 0+1+...+9 = 45
    // JIT 编译 sum_loop 时，while 回边触发 is_sealed panic → catch_unwind 捕获
    // → fallback 到 VM 解释执行 → 返回正确结果 45。
    let src = r#"
        fn sum_loop(n: Int) -> Int {
            let s = 0;
            let i = 0;
            while i < n {
                s = s + i;
                i = i + 1;
            };
            s
        }
        fn main() -> Int { sum_loop(10) }
    "#;
    let result = run_jit(src).expect("while loop should fall back to VM and produce a result");
    match result {
        Value::Int(n, _) => assert_eq!(n, 45, "sum(0..10) = 45, got {}", n),
        v => panic!("expected Int(45), got {:?}", v),
    }
}

#[test]
fn test_jit_for_loop_fallback_to_vm() {
    // for_range_sum(5) = 0+1+2+3+4 = 10
    // JIT 编译 for 循环时同样触发回边 panic → fallback → 正确结果。
    let src = r#"
        fn for_range_sum(n: Int) -> Int {
            let s = 0;
            for i in 0..n {
                s = s + i;
            };
            s
        }
        fn main() -> Int { for_range_sum(5) }
    "#;
    let result = run_jit(src).expect("for loop should fall back to VM and produce a result");
    match result {
        Value::Int(n, _) => assert_eq!(n, 10, "sum(0..5) = 10, got {}", n),
        v => panic!("expected Int(10), got {:?}", v),
    }
}

#[test]
fn test_jit_loop_fallback_no_panic() {
    // 关键断言：run_jit 不应 panic（即使 JIT 编译期间 Cranelift 内部断言失败）。
    // catch_unwind 应捕获 is_sealed panic 并转为 Err → fallback。
    // 若 catch_unwind 缺失或失效，此测试会因 panic 中止而非正常断言失败。
    let src = r#"
        fn count_loop(n: Int) -> Int {
            let count = 0;
            let i = 0;
            while i < n {
                count = count + 1;
                i = i + 1;
            };
            count
        }
        fn main() -> Int { count_loop(7) }
    "#;
    let result = run_jit(&src);
    assert!(result.is_ok(), "run_jit should not panic/error on loop: {:?}", result);
    if let Ok(Value::Int(v, _)) = result {
        assert_eq!(v, 7, "count_loop(7) = 7, got {}", v);
    } else {
        panic!("expected Ok(Int(7))");
    }
}
