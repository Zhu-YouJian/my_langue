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
use tenth::runtime::natives::register_all_natives;
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
                // 与 main.rs 对齐：函数注册为全局 FnRef（函数作值传递时按名解析，
                // 如 `partial(add, 5)` 的实参 `add`）。
                vm.set_global(func.name.clone(), Value::FnRef {
                    name: func.name.clone(),
                    params: func.params.clone(),
                    return_type: func.return_type.clone(),
                    captures: vec![],
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

// ── a1 P1/P2：闭包值调用对拍（VM vs JIT）───────────────────────────────────
//
// 背景：a1 引入 Op::CallClosure（opcode 57，间接调用栈上闭包/函数值）后，
// VM 与 JIT 对「闭包值调用」必须产出一致结果。此前闭包值调用在两条路径
// 都失败（VM 无调用指令；JIT call_with_args 不解析 FnRef）——见
// `.trae/tmp/a1_closure_call_plan.md`。以下用例逐个对拍，覆盖：
// - 单闭包无捕获（let f = |x| x+1; f(5)）
// - 多捕获（|x| x*scale+base）
// - 闭包作参数传入函数、函数体内调用闭包值（HOF）
// - 多函数各含闭包（闭包 chunk 名跨函数唯一性——防静默错值回归）
// - 递归闭包（fact 通过全局名自引用）

/// 纯 VM 路径（vm.call，不经 JIT）——与 run_jit 对照。
/// 与 dyn_trait_test.rs 的 run_vm 同构，注册全部 natives。
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
                // 与 main.rs 对齐：函数注册为全局 FnRef（函数作值传递时按名解析）。
                vm.set_global(func.name.clone(), Value::FnRef {
                    name: func.name.clone(),
                    params: func.params.clone(),
                    return_type: func.return_type.clone(),
                    captures: vec![],
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

/// 断言 VM 与 JIT 对同一源码产出一致且等于预期的 Int 结果。
fn assert_vm_jit_int(src: &str, expected: i64, label: &str) {
    let vm_res = run_vm(src).unwrap_or_else(|e| panic!("[{}] VM 执行失败: {}", label, e));
    let jit_res = run_jit(src).unwrap_or_else(|e| panic!("[{}] JIT 执行失败: {}", label, e));
    let vm_int = match vm_res { Value::Int(n, _) => n, v => panic!("[{}] VM 期望 Int，实际 {:?}", label, v) };
    let jit_int = match jit_res { Value::Int(n, _) => n, v => panic!("[{}] JIT 期望 Int，实际 {:?}", label, v) };
    assert_eq!(vm_int, expected, "[{}] VM 结果错误: {} != {}", label, vm_int, expected);
    assert_eq!(jit_int, expected, "[{}] JIT 结果错误: {} != {}", label, jit_int, expected);
    assert_eq!(vm_int, jit_int, "[{}] VM/JIT 不一致: {} != {}", label, vm_int, jit_int);
}

#[test]
fn test_closure_call_single() {
    // 单闭包无捕获：let f = |x| x+1; f(5) → 6
    assert_vm_jit_int(r#"
        fn main() -> Int {
            let f = |x: Int| x + 1;
            f(5)
        }
    "#, 6, "single");
}

#[test]
fn test_closure_call_capture() {
    // 多捕获：|x| x*scale+base → 3*2+10 = 16
    assert_vm_jit_int(r#"
        fn main() -> Int {
            let base = 10;
            let scale = 2;
            let compute = |x: Int| x * scale + base;
            compute(3)
        }
    "#, 16, "capture");
}

#[test]
fn test_closure_call_param_hof() {
    // 闭包作参数传入函数，函数体内调用闭包值（HOF）：apply(f,x){ f(f(x)) } → 7
    assert_vm_jit_int(r#"
        fn apply_twice(f: fn(Int) -> Int, x: Int) -> Int {
            f(f(x))
        }
        fn main() -> Int {
            let inc = |x: Int| x + 1;
            apply_twice(inc, 5)
        }
    "#, 7, "hof");
}

#[test]
fn test_closure_call_multiple_factories() {
    // 多函数各含闭包：闭包 chunk 名必须跨函数唯一，否则后注册者覆盖先注册者
    // （FnRef 解析到错误 chunk → 静默错值）。inc(5)*100 + dbl(5) = 6*100+10 = 610；
    // 若名冲突回归（inc→10, dbl→10）则得 1010，可辨识。
    assert_vm_jit_int(r#"
        fn make_inc() -> fn(Int) -> Int {
            |x: Int| x + 1
        }
        fn make_double() -> fn(Int) -> Int {
            |x: Int| x * 2
        }
        fn main() -> Int {
            let inc = make_inc();
            let dbl = make_double();
            inc(5) * 100 + dbl(5)
        }
    "#, 610, "multi-factory");
}

#[test]
fn test_closure_call_curry() {
    // 闭包返回闭包（curry/partial 风格）：partial(add,5) 返回 |x| f(arg,x)，
    // 内层闭包捕获外层参数 f/arg（走全局捕获 hack）。add5(3) → 8。
    // 注：递归闭包（`let fact = |n| ... fact(n-1)`）在 lowering 阶段被拒
    // （"未定义变量 'fact'"——闭包初始化器内自引用未绑定），属前端限制，
    // 非 a1 VM/JIT 范畴，故对拍用例不含递归闭包。
    assert_vm_jit_int(r#"
        fn partial(f: fn(Int, Int) -> Int, arg: Int) -> fn(Int) -> Int {
            |x: Int| f(arg, x)
        }
        fn add(a: Int, b: Int) -> Int { a + b }
        fn main() -> Int {
            let add5 = partial(add, 5);
            add5(3)
        }
    "#, 8, "curry");
}

#[test]
fn test_capture_inline_adder_independent() {
    // a1 P3 核心：捕获值内联后多实例捕获独立。P3 前 make_adder(5)/make_adder(10)
    // 捕获同名 n 走全局名 hack 互相覆盖，add5(3) 静默返回 13（应为 8）。
    // 8*100+13 = 813；若捕获串扰（add5 看到 n=10）则 13*100+13=1313，可辨识。
    assert_vm_jit_int(r#"
        fn make_adder(n: Int) -> fn(Int) -> Int {
            |x: Int| x + n
        }
        fn main() -> Int {
            let add5 = make_adder(5);
            let add10 = make_adder(10);
            add5(3) * 100 + add10(3)
        }
    "#, 813, "adder-independent");
}

#[test]
fn test_capture_inline_nested_closure() {
    // a1 P3：嵌套闭包 `|n| |x| x+n`（闭包返回闭包）。内层闭包捕获外层参数 n（值内联），
    // 且内层 chunk 必须并入父级注册表——否则 FnRef 名查不到 → 「未定义的函数」。
    assert_vm_jit_int(r#"
        fn main() -> Int {
            let make_adder = |n: Int| |x: Int| x + n;
            let add5 = make_adder(5);
            let add10 = make_adder(10);
            add5(3) * 100 + add10(3)
        }
    "#, 813, "nested-closure");
}

#[test]
fn test_capture_inline_hof_param() {
    // a1 P3：带捕获的闭包作参数传入 HOF，函数体内经参数槽间接调用（CallClosure 捕获注入）。
    // apply_twice(addn, 3) = f(f(3)) = (3+5)+5 = 13。
    assert_vm_jit_int(r#"
        fn apply_twice(f: fn(Int) -> Int, x: Int) -> Int {
            f(f(x))
        }
        fn main() -> Int {
            let n = 5;
            let addn = |x: Int| x + n;
            apply_twice(addn, 3)
        }
    "#, 13, "hof-capturing");
}
