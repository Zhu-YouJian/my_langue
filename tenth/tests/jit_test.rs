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
    // 注：递归闭包（`let fact = |n| ... fact(n-1)`）自引用解析见任务 9 子任务 9b
    // （test_recursive_closure_self_reference 对拍），此处 curry 用例专注捕获参数。
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

// ── 任务 9：a1 遗留三项补齐 ─────────────────────────────────────────────

#[test]
fn test_native_alias_binding() {
    // 9a：`let p = println` native 别名——bytecode/VM 侧 `let p = <native名>` 能把
    // native 作为可调用值绑定（FnRef 指向 native 名；LoadGlobal 未命中时查 natives）。
    // 修复前 JIT/VM 报「期望可调用值，得到 Unit」。println 返回 Unit，故断言两路径
    // 均成功（不报错）即可辨识回归。
    let src = r#"
        fn main() {
            let p = println;
            p("hello alias");
        }
    "#;
    let vm_res = run_vm(src);
    let jit_res = run_jit(src);
    assert!(vm_res.is_ok(), "[native-alias] VM 应成功，实际 {:?}", vm_res);
    assert!(jit_res.is_ok(), "[native-alias] JIT 应成功，实际 {:?}", jit_res);
}

#[test]
fn test_native_alias_passed_as_value() {
    // 9a：native 别名作函数实参传递——apply(p, 5) 内经参数槽间接调用
    // （CallClosure → call_value → natives 查询）。修复前实参 LoadGlobal 得 Unit。
    let src = r#"
        fn apply(f: fn(i64) -> i64, x: i64) -> i64 {
            f(x)
        }
        fn main() -> Int {
            let p = println;
            let r = apply(p, 5);
            r
        }
    "#;
    // println 返回 Unit；apply 返回 Unit。两路径都不应报错（可辨识回归）。
    let vm_res = run_vm(src);
    let jit_res = run_jit(src);
    assert!(vm_res.is_ok(), "[native-alias-arg] VM 应成功，实际 {:?}", vm_res);
    assert!(jit_res.is_ok(), "[native-alias-arg] JIT 应成功，实际 {:?}", jit_res);
}

#[test]
fn test_recursive_closure_self_reference() {
    // 9b：递归闭包 `let fact = |n| ... fact(n-1)`——前端自引用解析（lowering 不再报
    // 「未定义变量 'fact'」），运行时按名解析（VM globals / 解释器作用域链）。
    // fact(5) = 120，VM=JIT 对拍一致。
    assert_vm_jit_int(r#"
        fn main() -> Int {
            let fact = |n: Int| if n <= 1 { 1 } else { n * fact(n - 1) };
            fact(5)
        }
    "#, 120, "recursive-closure");
}

#[test]
fn test_recursive_closure_with_capture() {
    // 9b：递归闭包 + 捕获混用——自引用名排除出 captures（运行时按名解析），
    // 同时 scale/base 正常捕获（值内联）。f(n)=n<=1?1:n*f(n-1)*scale+base：
    // f(1)=1, f(2)=14, f(3)=94, f(4)=762。VM=JIT 对拍一致。
    assert_vm_jit_int(r#"
        fn main() -> Int {
            let scale = 2;
            let base = 10;
            let f = |n: Int| if n <= 1 { 1 } else { n * f(n - 1) * scale + base };
            f(4)
        }
    "#, 762, "recursive-closure-capture");
}

// ── M2-A1：JIT-to-JIT 直接调用对拍 ────────────────────────────────────────
//
// 背景：A1 前任何 Call/CallN/MethodCall 都经 host_call 逃逸回 VM/解释器，
// 递归零收益（fib(28) ≈ 180ms）。A1 后 translator 对已注册用户函数生成
// **直接调用**（快路径 call_indirect 已编译机器码；首次遇未编译函数走
// host_jit_call trampoline 编译注册）。以下用例对拍 VM=JIT，覆盖：
// - 递归 fib（自引用，快路径递归展开）
// - 同一函数二次调用（先慢后快：验证 trampoline 编译注册后走快路径）
// - 非递归多层嵌套直接调用链（跨函数快路径）
// - 递归 + 调用后运算（曾触发原生栈溢出的形态，见 MAX_STACK_DEPTH 注释）
// - 中等深度线性递归（栈回归守护：JIT 帧 7KB，深度 100 ≈ 700KB < 测试线程栈）

#[test]
fn test_jit_recursive_fib_direct() {
    // 递归 fib 经 JIT 直接调用：fib(20) = 6765。VM=JIT 对拍一致。
    // A1 前此用例 JIT 路径每次调用都逃逸解释器（仍正确但慢）；
    // A1 后走快路径 call_indirect，fib(28) 实测 ~28ms（<50ms 验收线）。
    assert_vm_jit_int(r#"
        fn fib(n: Int) -> Int {
            if n < 2 { n } else { fib(n - 1) + fib(n - 2) }
        }
        fn main() -> Int { fib(20) }
    "#, 6765, "recursive-fib-direct");
}

#[test]
fn test_jit_direct_call_fast_path() {
    // 同一函数二次调用：第一次 host_jit_call trampoline 编译注册，
    // 第二次直接 call_indirect（快路径）。两路径结果一致 + VM 对拍。
    assert_vm_jit_int(r#"
        fn add(a: Int, b: Int) -> Int { a + b }
        fn main() -> Int {
            let x = add(3, 4);
            let y = add(x, 5);
            y * 10 + x
        }
    "#, 127, "direct-call-fast-path");
}

#[test]
fn test_jit_nested_multi_level_direct() {
    // 非递归 6 层嵌套直接调用链：a→b→c→d→e→f，每层 +1。结果 6。
    // 覆盖跨函数快路径链（非自引用）。
    assert_vm_jit_int(r#"
        fn f1() -> Int { 1 }
        fn f2() -> Int { f1() + 1 }
        fn f3() -> Int { f2() + 1 }
        fn f4() -> Int { f3() + 1 }
        fn f5() -> Int { f4() + 1 }
        fn f6() -> Int { f5() + 1 }
        fn main() -> Int { f6() }
    "#, 6, "nested-multi-level-direct");
}

#[test]
fn test_jit_recursive_accumulate_after_call() {
    // 递归 + 调用后运算（A1 调试期触发原生栈溢出的形态）：
    // f(n) = n<=1 ? 1 : f(n-1)*2 + n。f(15) 精确值。
    // 覆盖「快路径调用返回后继续 hostcall 运算」的正确性。
    let expected: i64 = {
        let mut v = 1i64;
        for n in 2..=15 { v = v * 2 + n; }
        v
    };
    assert_vm_jit_int(r#"
        fn f(n: Int) -> Int {
            if n <= 1 { 1 } else { f(n - 1) * 2 + n }
        }
        fn main() -> Int { f(15) }
    "#, expected, "recursive-accumulate");
}

#[test]
fn test_jit_linear_recursion_depth_guard() {
    // 深度 100 线性递归（每层一帧）：守护 MAX_STACK_DEPTH 回归——
    // 若帧恢复到 256（28KB），深度 100 ≈ 2.8MB 会压垮测试线程栈。
    // sum(n) = n<=0 ? 0 : sum(n-1) + n，sum(100) = 5050。
    assert_vm_jit_int(r#"
        fn sum(n: Int) -> Int {
            if n <= 0 { 0 } else { sum(n - 1) + n }
        }
        fn main() -> Int { sum(100) }
    "#, 5050, "linear-recursion-depth");
}

#[test]
fn test_jit_direct_call_method_fallback() {
    // MethodCall 目标为 native（VM 不把用户方法编译为 chunk）→ 保持
    // host_method_call fallback，语义与 VM 一致。String::len 对拍。
    let src = r#"
        fn main() -> Int {
            let s = "hello";
            s.len()
        }
    "#;
    let vm_res = run_vm(src).unwrap();
    let jit_res = run_jit(src).unwrap();
    let vm_int = match vm_res { Value::Int(n, _) => n, v => panic!("VM 期望 Int，实际 {:?}", v) };
    let jit_int = match jit_res { Value::Int(n, _) => n, v => panic!("JIT 期望 Int，实际 {:?}", v) };
    assert_eq!(vm_int, 5, "VM s.len() = 5, got {}", vm_int);
    assert_eq!(jit_int, 5, "JIT s.len() = 5, got {}", jit_int);
}

// ── M2-A2：小函数内联 + 标量专用化对拍 ─────────────────────────────────────
//
// A2 在 A1 基础上新增：
// 1. **调用点内联**：被调 chunk 满足内联条件（≤16 指令、纯标量/控制流、无递归/
//    调用/字符串/复杂类型）时，把 body 直插调用方机器码（不 emit call）；不满足
//    则静默回退 A1 直接调用。
// 2. **标量专用化**：跨块 must 分析证明局部恒为 Int/F64/Bool 时，Load/Store/
//    算术/比较走原生路径（伴随标量槽），Value 槽延迟物化/双写保持正确。
// 以下用例对拍 VM=JIT，覆盖：
// - 小函数内联正确性（多参数、带 if/else、连续多次调用）
// - 内联失败回退（大函数/含调用/字符串 → 静默走 A1 直接调用）
// - 内联与直接调用混用（同一函数部分调用点内联、部分不内联）
// - 内联函数返回被调用方后续消费（Store/Load/算术链）
// - 标量 loop（Int 局部：sum/i 跨回边——must 分析 + 原生算术）
// - 浮点标量 loop（F64 局部）
// - 溢出/除零在标量路径下报错且与 VM 一致

#[test]
fn test_jit_inline_small_function() {
    // 小纯函数 add（4 指令）被内联：add(3,4) 与 add(x,5) 均内联。结果 127。
    assert_vm_jit_int(r#"
        fn add(a: Int, b: Int) -> Int { a + b }
        fn main() -> Int {
            let x = add(3, 4);
            let y = add(x, 5);
            y * 10 + x
        }
    "#, 127, "inline-small-function");
}

#[test]
fn test_jit_inline_with_control_flow() {
    // 含 if/else 的小函数被内联：max(3,5)=5, max(7,2)=7。覆盖内联体控制流。
    assert_vm_jit_int(r#"
        fn max(a: Int, b: Int) -> Int { if a > b { a } else { b } }
        fn main() -> Int {
            let x = max(3, 5);
            let y = max(7, 2);
            x * 10 + y
        }
    "#, 57, "inline-with-control-flow");
}

#[test]
fn test_jit_inline_fallback_large() {
    // 大函数（>16 指令）不满足内联条件 → 静默回退 A1 直接调用（正确性优先）。
    // 多分支求和，结果 55。
    assert_vm_jit_int(r#"
        fn big_sum(n: Int) -> Int {
            let s = 0;
            let i = 0;
            while i < n {
                s = s + i;
                i = i + 1;
            };
            s
        }
        fn main() -> Int { big_sum(11) }
    "#, 55, "inline-fallback-large");
}

#[test]
fn test_jit_inline_fallback_string() {
    // 含字符串的小函数不内联（PushStr 依赖 current_chunk_idx 字符串表）→ 回退。
    assert_vm_jit_int(r#"
        fn greet() -> Int { "hi".len() }
        fn main() -> Int { greet() }
    "#, 2, "inline-fallback-string");
}

#[test]
fn test_jit_inline_mixed_with_direct() {
    // 同一函数 double：main 内联调用点 + 非内联路径（其他函数直接调用）。
    // 内联与直接调用混用结果一致。
    assert_vm_jit_int(r#"
        fn double(x: Int) -> Int { x * 2 }
        fn use_direct(x: Int) -> Int { double(x) + 1 }
        fn main() -> Int {
            let a = double(5);       // 内联（main 内）
            let b = use_direct(5);   // use_direct 内调用 double——可能内联或直接
            a * 100 + b
        }
    "#, 1011, "inline-mixed-direct");
}

#[test]
fn test_jit_inline_result_consumed() {
    // 内联结果被调用方后续 Store/Load/算术链消费（回归：A2 曾因物化保留旧标量
    // 跟踪导致内联结果读错——静默错值红线）。
    assert_vm_jit_int(r#"
        fn combo(a: Int, b: Int) -> Int { a * 100 + b }
        fn main() -> Int {
            let r = combo(3, 4);
            let s = combo(r, 5);
            s
        }
    "#, 30405, "inline-result-consumed");
}

#[test]
fn test_jit_scalar_loop_int() {
    // 标量 loop：Int 局部 sum/i 跨回边（must 分析证明恒为 Int），原生算术。
    // 1..=100 求和 = 5050；再加 i%7 的部分。
    // sum(0..100)=4950；sum(i%7, 0..100)=14*21+1=295；合计 5245（VM 实测一致）。
    assert_vm_jit_int(r#"
        fn main() -> Int {
            let mut sum = 0;
            let mut i = 0;
            while i < 100 {
                sum = sum + i + (i % 7);
                i = i + 1;
            };
            sum
        }
    "#, 5245, "scalar-loop-int");
}

#[test]
fn test_jit_scalar_loop_float() {
    // 标量 loop：Float 局部走 F64 原生路径。sum += 0.5 × 10 = 5.0。
    let src = r#"
        fn main() -> Float {
            let mut sum = 0.0;
            let mut i = 0;
            while i < 10 {
                sum = sum + 0.5;
                i = i + 1;
            };
            sum
        }
    "#;
    let vm_res = run_vm(src).unwrap();
    let jit_res = run_jit(src).unwrap();
    let vm_f = match vm_res { Value::Float(f) => f, v => panic!("VM 期望 Float，实际 {:?}", v) };
    let jit_f = match jit_res { Value::Float(f) => f, v => panic!("JIT 期望 Float，实际 {:?}", v) };
    assert!((vm_f - 5.0).abs() < 1e-9, "VM = {}, want 5.0", vm_f);
    assert!((jit_f - 5.0).abs() < 1e-9, "JIT = {}, want 5.0", jit_f);
    assert!((vm_f - jit_f).abs() < 1e-9, "VM/JIT 不一致: {} != {}", vm_f, jit_f);
}

#[test]
fn test_jit_scalar_loop_overflow_consistent() {
    // 标量路径溢出检查与 VM 一致：I32 范围溢出（loop 累加超过 i32::MAX）。
    let src = r#"
        fn main() -> Int {
            let mut sum = 2147483640;
            let mut i = 0;
            while i < 10 {
                sum = sum + i;
                i = i + 1;
            };
            sum
        }
    "#;
    let vm_res = run_vm(src);
    let jit_res = run_jit(src);
    assert_eq!(vm_res.is_err(), jit_res.is_err(), "VM/JIT 溢出成功性不一致: {:?} vs {:?}", vm_res, jit_res);
}

#[test]
fn test_jit_inline_div_zero_error() {
    // 内联函数内除零：报错与 VM 一致（错误消息「整数除零」）。
    let src = r#"
        fn div(a: Int, b: Int) -> Int { a / b }
        fn main() -> Int {
            let x = div(10, 0);
            x
        }
    "#;
    let vm_res = run_vm(src);
    let jit_res = run_jit(src);
    assert!(vm_res.is_err(), "VM 应报除零错误");
    assert!(jit_res.is_err(), "JIT 应报除零错误");
    let vm_msg = vm_res.unwrap_err();
    let jit_msg = jit_res.unwrap_err();
    assert!(vm_msg.contains("整数除零"), "VM 消息: {}", vm_msg);
    assert!(jit_msg.contains("整数除零"), "JIT 消息: {}", jit_msg);
}
