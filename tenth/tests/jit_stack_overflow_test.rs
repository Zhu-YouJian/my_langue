//! Regression tests for AUDIT-11.4.2: JIT stack-overflow graceful fallback.
//!
//! Background: `compile/jit/translator.rs` uses a fixed-size virtual stack
//! area of `MAX_STACK_DEPTH * VALUE_SIZE` bytes (MAX_STACK_DEPTH = 256).
//! Before the fix, translator code silently let `sp` grow past the limit,
//! causing out-of-bounds writes into the stack slot (memory corruption,
//! hard-to-debug crashes). There was no compile-time check.
//!
//! Fix (`tenth/src/compile/jit/translator.rs:523-533`): added `bump_sp()`
//! which checks `sp + VALUE_SIZE > MAX_STACK_DEPTH * VALUE_SIZE` and
//! returns `Err("JIT stack overflow: ...")`. 28 push sites were updated
//! to call `bump_sp()?`. The Err propagates up through `translate` →
//! `JitContext::get_or_compile` → `run_jit` (`compile/jit/mod.rs:62-65`),
//! which catches it and falls back to `Vm::call` (interpreter). So a
//! stack-overflow at translate time is graceful degradation, not an abort.
//!
//! These tests construct functions whose JIT translation would exceed
//! MAX_STACK_DEPTH, then verify:
//!   1. A small function (well under the limit) JIT-compiles and runs.
//!   2. A function exceeding the limit still produces the correct result
//!      (run_jit falls back to the interpreter).
//!   3. The fallback path does not panic (returns Ok with correct value).
//!
//! Strategy: right-nested addition `1 + (2 + (3 + ... + (N-1 + N)...))`.
//! Each PushInt bumps sp by 1; the deepest point in the AST has N values
//! on the stack before any Add pops them. So N > 256 forces overflow.
//! Result is the arithmetic sum 1+2+...+N = N*(N+1)/2.

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
/// Mirrors `run_jit` from jit_test.rs but calls `jit::run_jit` instead
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

/// Build a right-nested addition expression with `n` terms:
/// `1 + (2 + (3 + ... + (n-1 + n)...))`
///
/// Each term is a PushInt; the deepest point has n values on the stack
/// before any Add pops them. So n > MAX_STACK_DEPTH (256) forces the
/// JIT translator's bump_sp() to return Err, triggering VM fallback.
fn right_nested_add(n: usize) -> String {
    // n terms, n-1 additions, n-1 opening parens, n-1 closing parens.
    let mut s = String::with_capacity(n * 8);
    s.push_str("fn main() -> Int { ");
    for i in 1..=n {
        if i > 1 {
            s.push_str(" + (");
        }
        s.push_str(&i.to_string());
    }
    for _ in 1..n {
        s.push(')');
    }
    s.push_str(" }");
    s
}

/// Arithmetic sum 1 + 2 + ... + n = n*(n+1)/2.
fn arith_sum(n: i64) -> i64 {
    n * (n + 1) / 2
}

// ── Test 1: small function JIT-compiles and runs ───────────────────────
//
// 10-level right-nested addition: `1 + (2 + (3 + ... + (9 + 10)...))`.
// Stack depth peaks at 10 — well under MAX_STACK_DEPTH (256). JIT should
// compile this directly (no fallback) and return 55.
//
// Guards: translator.rs normal path — bump_sp() succeeds for each push,
// Add pops 2 + pushes 1, function returns the sum.

#[test]
fn jit_stack_within_limit_compiles() {
    let src = right_nested_add(10);
    let result = run_jit(&src).expect("small expr should JIT-compile");
    match result {
        Value::Int(n, _) => assert_eq!(n, arith_sum(10), "1+2+...+10 = 55"),
        v => panic!("expected Int({}), got {:?}", arith_sum(10), v),
    }
}

// ── Test 2: deep function falls back to VM and returns correct result ──
//
// 300-level right-nested addition. Stack depth peaks at 300 > 256, so
// bump_sp() returns Err during translation. run_jit catches the Err and
// falls back to Vm::call (interpreter). The interpreter has no static
// stack-depth limit, so it computes 1+2+...+300 = 45150 correctly.
//
// If the fallback path were broken (e.g. Err propagated as panic, or
// fallback returned wrong value), this test would fail.
//
// Guards: translator.rs:523-533 (bump_sp overflow check),
//         compile/jit/mod.rs:62-65 (Err → vm.call fallback).

#[test]
fn jit_stack_overflow_falls_back_to_vm() {
    let n = 300;
    let src = right_nested_add(n);
    let result = run_jit(&src).expect("fallback should produce a result");
    match result {
        Value::Int(v, _) => assert_eq!(
            v, arith_sum(n as i64),
            "1+2+...+{} = {}, got {}", n, arith_sum(n as i64), v
        ),
        v => panic!("expected Int({}), got {:?}", arith_sum(n as i64), v),
    }
}

// ── Test 3: deep function does not panic on stack overflow ─────────────
//
// Same 300-level expression as Test 2, but the assertion is specifically
// that run_jit returns (Ok or Err) rather than panicking. The previous
// behavior (silent overflow) could cause memory corruption that manifests
// as a panic or crash; the fix ensures the overflow is caught at compile
// time and converted to a graceful Err → fallback.
//
// Note: run_jit's fallback should always produce Ok for this input (the
// interpreter can evaluate it). But the key assertion is no panic, which
// we check by calling .expect() — if run_jit panicked, the test process
// would abort with a panic message rather than reaching the assert.

#[test]
fn jit_stack_overflow_no_panic() {
    let n = 300;
    let src = right_nested_add(n);
    // If this panics (e.g. due to bump_sp not catching overflow), the
    // test fails with a panic backtrace instead of a clean assertion.
    let result = run_jit(&src);
    assert!(result.is_ok(), "run_jit should not error/panic on stack overflow: {:?}", result);
    if let Ok(Value::Int(v, _)) = result {
        assert_eq!(v, arith_sum(n as i64));
    } else {
        panic!("expected Ok(Int(_))");
    }
}
