//! Regression tests for AUDIT-11.4.2 / AUDIT-11.4.51: JIT stack-overflow graceful fallback.
//!
//! Background: `compile/jit/translator.rs` uses a fixed-size virtual stack
//! area of `MAX_STACK_DEPTH * VALUE_SIZE` bytes. The live value is
//! **`MAX_STACK_DEPTH = 64`** (`tenth/src/compile/jit/translator.rs`
//! `const MAX_STACK_DEPTH: u32 = 64;` — 本文件此前多处把该值写成 256，
//! 属 AUDIT-11.4.51 记录的「注释失实」，已按源码更正）。
//! Before the fix, translator code silently let `sp` grow past the limit,
//! causing out-of-bounds writes into the stack slot (memory corruption,
//! hard-to-debug crashes). There was no compile-time check.
//!
//! Fix (`tenth/src/compile/jit/translator.rs` `fn bump_sp`, 现落于 2820-2830):
//! it checks `sp + VALUE_SIZE > MAX_STACK_DEPTH * VALUE_SIZE` and returns
//! `Err("JIT stack overflow: ...")`. All push sites call `bump_sp()?`.
//! The Err propagates up through `translate` → `JitContext::get_or_compile`
//! → `run_jit` (`compile/jit/mod.rs`: `Err(_) => return vm.call(name)`，现落于
//! 84-87 行), which catches it and falls back to `Vm::call` (interpreter).
//! So a stack-overflow at translate time is graceful degradation, not an abort.
//!
//! These tests construct functions whose JIT translation would exceed
//! MAX_STACK_DEPTH, then verify:
//!   1. A small function (well under the limit) JIT-compiles and runs
//!      **without any fallback** (`JitContext::is_failed(chunk) == false`).
//!   2. A function exceeding the limit still produces the correct result
//!      (run_jit falls back to the interpreter) — **and the fallback is
//!      proven, not assumed** (AUDIT-11.4.51):
//!        (0) `JitContext::get_or_compile` on that chunk returns
//!            `Err(.."JIT stack overflow"..)` — the trigger condition;
//!        (1) after `run_jit`, `JitContext::is_failed(chunk) == true` —
//!            i.e. `mod.rs` 的 `Err(_) => return vm.call(name)` 分支确实被走到。
//!      此前两条用例只断言结果值，而该值在 JIT 与 fallback 两条路径下都正确
//!      ⇒ 用例并未断言 fallback 真的发生（测试静默失效）。
//!   3. The fallback path does not panic (returns Ok with correct value).
//!
//! Strategy: right-nested addition `1 + (2 + (3 + ... + (N-1 + N)...))`.
//! Each PushInt bumps sp by 1; the deepest point in the AST has N values
//! on the stack before any Add pops them. So N > 64 forces overflow.
//! Result is the arithmetic sum 1+2+...+N = N*(N+1)/2.

use tenth::lexer::lexer::Lexer;
use tenth::parser::parser::Parser;
use tenth::hir::lower::Lowerer;
use tenth::runtime::vm::{Chunk, Vm};
use tenth::runtime::value::Value;
use tenth::compile::bytecode::BytecodeCompiler;
use tenth::compile::jit;
use tenth::compile::jit::context::{ChunkSig, JitContext};
use std::rc::Rc;
use std::cell::RefCell;

/// 源码 → HIR → 字节码 → `Vm`（`main` 已注册为 chunk）。
/// 与 `run_jit` 的前半段一致，抽出以便同时做「直接驱动 JIT 编译入口」的探针。
fn build_vm(src: &str) -> Result<Vm, String> {
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
    }
    Ok(vm)
}

/// 一次 JIT 运行的**可观测**结果（AUDIT-11.4.51：把「确实走了 fallback」
/// 变成可断言的事实，而不是默认假设）。
struct JitRun {
    /// `run_jit` 的返回值（错误已转字符串）。
    result: Result<Value, String>,
    /// `Some(true)` = 该 chunk 被 `JitContext` 标记为**编译失败**，
    /// 即 `compile/jit/mod.rs:86` 的 `Err(_) => return vm.call(name)`
    /// 回退分支被走到（run_jit 内部）；`Some(false)` = JIT 编译成功，未回退。
    jit_compile_failed: Option<bool>,
    /// `main` 的 chunk 索引（未注册时为 None）。
    chunk_idx: Option<usize>,
}

/// 与 `run_jit` 同路径，但额外回报「回退是否真的发生」。
fn run_jit_observed(src: &str) -> JitRun {
    let mut vm = match build_vm(src) {
        Ok(v) => v,
        Err(e) => return JitRun { result: Err(e), jit_compile_failed: None, chunk_idx: None },
    };
    let chunk_idx = vm.chunk_index_of("main");
    if chunk_idx.is_none() {
        return JitRun { result: Ok(Value::Unit), jit_compile_failed: None, chunk_idx: None };
    }
    let result = jit::run_jit(&mut vm, "main").map_err(|e| e.to_string());
    // run_jit 内部已建立 jit_ctx（JIT 路径必经）；`is_failed` 是 JIT 编译失败的台账。
    let failed = chunk_idx.map(|i| {
        vm.jit_ctx.as_ref().map(|ctx| ctx.is_failed(i)).unwrap_or(false)
    });
    JitRun { result, jit_compile_failed: failed, chunk_idx }
}

/// 运行源码（JIT 路径，失败时 `run_jit` 内部回退解释器）。
fn run_jit(src: &str) -> Result<Value, String> {
    run_jit_observed(src).result
}

/// **直接**驱动 `JitContext::get_or_compile`（`run_jit` 内部 fallback 判据的
/// 同源入口），返回编译器给出的原始结果：
/// - `Ok(())` — JIT 翻译成功 ⇒ `run_jit` 走 JIT，**不会**回退；
/// - `Err(msg)` — JIT 翻译失败（本条用例要求 msg 含 `JIT stack overflow`）
///   ⇒ `run_jit` 走 `Err(_) => vm.call(name)` 回退解释器。
///
/// 配置顺序与 `compile/jit/mod.rs::run_jit` 保持一致（name→chunk 表、
/// 全部 chunk、特化签名表、skip_chunk_ctx 重算、表定容）。
fn jit_translate_main(src: &str) -> Result<(), String> {
    let vm = build_vm(src)?;
    let idx = vm
        .chunk_index_of("main")
        .ok_or_else(|| "main 未注册为 chunk，无法驱动 JIT 编译".to_string())?;
    let chunk: Chunk = vm.chunk_at(idx).clone();

    let mut ctx = JitContext::new();
    ctx.set_name_to_chunk(vm.functions.clone());
    let all: Vec<Chunk> = (0..vm.chunk_count()).map(|i| vm.chunk_at(i).clone()).collect();
    ctx.set_all_chunks(all);
    let sigs: Vec<Option<ChunkSig>> =
        (0..vm.chunk_count()).map(|i| vm.chunk_at(i).scalar_sig.clone()).collect();
    ctx.set_chunk_sigs(sigs);
    ctx.recompute_skip_chunk_ctx();
    ctx.ensure_table(vm.chunk_count());
    ctx.ensure_spec_table(vm.chunk_count());

    match ctx.get_or_compile(idx, &chunk) {
        Ok(_) => Ok(()),
        Err(e) => {
            assert!(
                ctx.is_failed(idx),
                "get_or_compile 返回 Err 但未把 chunk {idx} 记入 failed 台账（{}）\
                 ——run_jit 的 fail→缓存放大会变，回退判定会失准",
                e
            );
            Err(e)
        }
    }
}

/// Build a right-nested addition expression with `n` terms:
/// `1 + (2 + (3 + ... + (n-1 + n)...))`
///
/// Each term is a PushInt; the deepest point has n values on the stack
/// before any Add pops them. So n > MAX_STACK_DEPTH (64) forces the
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

// ── Test 1: small function JIT-compiles and runs (no fallback) ─────────
//
// 10-level right-nested addition: `1 + (2 + (3 + ... + (9 + 10)...))`.
// Stack depth peaks at 10 — well under MAX_STACK_DEPTH (64). JIT compiles
// this directly; the fallback must NOT be triggered (断言成对：Test 2 断言
// 回退发生，本用例断言**未**发生 —— 否则「回退断言」可能退化为恒真）。
//
// Guards: translator.rs normal path — bump_sp() succeeds for each push,
// Add pops 2 + pushes 1, function returns the sum.

#[test]
fn jit_stack_within_limit_compiles() {
    let n = 10;
    let src = right_nested_add(n);
    let run = run_jit_observed(&src);
    let result = run.result.clone().expect("small expr should JIT-compile");
    match result {
        Value::Int(v, _) => assert_eq!(v, arith_sum(10), "1+2+...+10 = 55"),
        v => panic!("expected Int({}), got {:?}", arith_sum(10), v),
    }
    assert_eq!(
        run.jit_compile_failed,
        Some(false),
        "n={} 远低于 MAX_STACK_DEPTH=64，JIT 应编译成功（不得回退解释器）；\
         实测 jit_compile_failed={:?}（chunk_idx={:?}）",
        n, run.jit_compile_failed, run.chunk_idx
    );
    assert_eq!(
        jit_translate_main(&src),
        Ok(()),
        "n={} 时 get_or_compile 应返回 Ok（bump_sp 溢出护栏未被触发）；\
         若此处失败，说明阈值/构造已变，Test 2 的 fallback 断言需要重新校准",
        n
    );
}

// ── Test 2: deep function falls back to VM and returns correct result ──
//
// 300-level right-nested addition. Stack depth peaks at 300 > 64, so
// bump_sp() returns Err during translation. run_jit catches the Err and
// falls back to Vm::call (interpreter). The interpreter has no static
// stack-depth limit, so it computes 1+2+...+300 = 45150 correctly.
//
// AUDIT-11.4.51：本用例此前**只**断言结果值，而该值在 JIT 与 fallback
// 下都正确 ⇒ 未断言 fallback 真的发生（测试静默失效）。现补两条守护断言：
//   (0) JIT 编译入口（get_or_compile）必须报 `JIT stack overflow`；
//   (1) run_jit 之后该 chunk 必须被记为编译失败（= 走了 vm.call 回退分支）。
// 任一条不成立 ⇒ 本用例报红，而不是继续「绿着但不再覆盖 fallback」。
//
// Guards: translator.rs bump_sp 溢出检查,
//         compile/jit/mod.rs 的 Err → vm.call fallback 分支。

#[test]
fn jit_stack_overflow_falls_back_to_vm() {
    let n = 300;
    let src = right_nested_add(n);

    // (0) 触发条件：JIT 翻译确实因栈溢出而失败（否则本用例没覆盖 fallback）。
    let err = jit_translate_main(&src).expect_err(&format!(
        "n={} 应超过 MAX_STACK_DEPTH=64 使 JIT 翻译因 bump_sp 溢出而失败；\
         实测 get_or_compile 返回 Ok（JIT 编译成功 ⇒ 本用例并未覆盖 fallback 路径）",
        n
    ));
    assert!(
        err.contains("JIT stack overflow"),
        "n={} 的 JIT 编译错误应含 `JIT stack overflow`（translator.rs bump_sp），实测: {}",
        n, err
    );

    // (1) 端到端：run_jit 走了回退分支（chunk 被记为编译失败），且结果值正确。
    let run = run_jit_observed(&src);
    assert_eq!(
        run.jit_compile_failed,
        Some(true),
        "n={} 应触发 fallback：JitContext::is_failed(main) 必须为 true\
         （compile/jit/mod.rs:86 `Err(_) => return vm.call(name)`）；\
         实测 jit_compile_failed={:?}（chunk_idx={:?}）",
        n, run.jit_compile_failed, run.chunk_idx
    );
    let result = run.result.clone().expect("fallback should produce a result");
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
