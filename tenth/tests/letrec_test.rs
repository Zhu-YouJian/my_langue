//! M1-S2：true letrec（递归闭包自引用 cell）守护测试。
//!
//! 背景（AUDIT-11.4.30 遗留）：任务 9 递归闭包按名解析——逃逸定义作用域 + 多实例
//! 共存时 VM 经全局名别名互相覆盖（静默错值：f1 递归误调 f2 → 522769 应为 762769）、
//! 解释器报「未定义变量」（定义作用域已弹出）——VM=解释器不一致。
//!
//! 本套件守护 true letrec 语义：递归闭包自引用绑定随闭包实例走（可变 cell 捕获），
//! 每实例独立，VM=JIT=解释器三方对拍一致，无静默错值。

use tenth::lexer::lexer::Lexer;
use tenth::parser::parser::Parser;
use tenth::hir::lower::Lowerer;
use tenth::runtime::vm::Vm;
use tenth::runtime::value::Value;
use tenth::runtime::natives::register_all_natives;
use tenth::compile::bytecode::BytecodeCompiler;
use tenth::compile::jit;
use tenth::runtime::interpreter::Interpreter;

/// 纯 VM 路径（vm.call，不经 JIT）。
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

/// JIT 路径（jit::run_jit，失败内部 fallback VM）。
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
        Ok(Value::Vec(std::rc::Rc::new(std::cell::RefCell::new(Vec::new()))))
    });

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

/// 解释器路径（tree-walk）。
fn run_interp(src: &str) -> Result<Value, String> {
    let mut lexer = Lexer::new(src);
    let tokens = lexer.tokenize().map_err(|e| e.to_string())?;
    let mut parser = Parser::new(tokens);
    let program = parser.parse_program().map_err(|e| e.to_string())?;
    let mut lowerer = Lowerer::new();
    let hir = lowerer.lower_program(&program).map_err(|e| e.to_string())?;
    let mut interp = Interpreter::new(&hir);
    interp.execute_program(&hir)
        .map(|v| v.unwrap_or(Value::Unit))
        .map_err(|e| e.to_string())
}

fn as_int(v: Value, label: &str) -> i64 {
    match v { Value::Int(n, _) => n, v => panic!("[{}] 期望 Int，实际 {:?}", label, v) }
}

/// 三方对拍：VM=JIT=解释器 且都等于预期 Int。
fn assert_three_way_int(src: &str, expected: i64, label: &str) {
    let vm = run_vm(src).unwrap_or_else(|e| panic!("[{}] VM 执行失败: {}", label, e));
    let jit = run_jit(src).unwrap_or_else(|e| panic!("[{}] JIT 执行失败: {}", label, e));
    let interp = run_interp(src).unwrap_or_else(|e| panic!("[{}] 解释器执行失败: {}", label, e));
    let (vi, ji, ii) = (as_int(vm, label), as_int(jit, label), as_int(interp, label));
    assert_eq!(vi, expected, "[{}] VM 结果错误: {} != {}", label, vi, expected);
    assert_eq!(ji, expected, "[{}] JIT 结果错误: {} != {}", label, ji, expected);
    assert_eq!(ii, expected, "[{}] 解释器结果错误: {} != {}", label, ii, expected);
    assert_eq!(vi, ji, "[{}] VM/JIT 不一致: {} != {}", label, vi, ji);
    assert_eq!(vi, ii, "[{}] VM/解释器不一致: {} != {}", label, vi, ii);
}

// ── M1-S2：true letrec 守护 ─────────────────────────────────────────────

/// 多实例独立：make_fact(2,10)/make_fact(3,1) 两实例并存，各自递归正确。
/// 修复前 VM 全局名别名互相覆盖 → 522769（f1 递归误调 f2）；解释器报未定义变量。
/// 修复后每实例独立：f1(4)=762, f2(4)=769 → 762769，三方一致。
#[test]
fn test_letrec_multi_instance_independent() {
    assert_three_way_int(r#"
        fn make_fact(scale: Int, base: Int) -> fn(Int) -> Int {
            let fact = |n: Int| if n <= 1 { 1 } else { n * fact(n - 1) * scale + base };
            fact
        }
        fn main() -> Int {
            let f1 = make_fact(2, 10);
            let f2 = make_fact(3, 1);
            f1(4) * 1000 + f2(4)
        }
    "#, 762769, "letrec-multi-instance");
}

/// 逃逸作用域：函数返回递归闭包后调用（定义作用域已弹出，cell 随闭包走）。
/// 修复前解释器报「未定义变量」（作用域链找不到 fact）。
#[test]
fn test_letrec_escape_scope() {
    assert_three_way_int(r#"
        fn make_fact() -> fn(Int) -> Int {
            let fact = |n: Int| if n <= 1 { 1 } else { n * fact(n - 1) };
            fact
        }
        fn main() -> Int {
            let f = make_fact();
            f(6)
        }
    "#, 720, "letrec-escape");
}

/// 回归：main 级递归闭包 fact(5)=120（任务 9 已修场景，true letrec 不回归）。
#[test]
fn test_letrec_main_level_fact() {
    assert_three_way_int(r#"
        fn main() -> Int {
            let fact = |n: Int| if n <= 1 { 1 } else { n * fact(n - 1) };
            fact(5)
        }
    "#, 120, "letrec-main-fact");
}

/// 回归：递归 + 捕获混用（f(4)=762，scale/base 值内联 + 自引用 cell）。
#[test]
fn test_letrec_recursive_with_capture() {
    assert_three_way_int(r#"
        fn main() -> Int {
            let scale = 2;
            let base = 10;
            let f = |n: Int| if n <= 1 { 1 } else { n * f(n - 1) * scale + base };
            f(4)
        }
    "#, 762, "letrec-capture");
}

/// 三个不同递归闭包实例各算各的（互不干扰，防跨实例别名静默错值）。
#[test]
fn test_letrec_three_instances() {
    assert_three_way_int(r#"
        fn make_mul_fact(mult: Int) -> fn(Int) -> Int {
            let fact = |n: Int| if n <= 1 { 1 } else { fact(n - 1) * n * mult };
            fact
        }
        fn main() -> Int {
            let a = make_mul_fact(1);   // n!
            let b = make_mul_fact(2);   // fact(n-1)*n*2
            let c = make_mul_fact(3);   // fact(n-1)*n*3
            a(5) * 1000000 + b(4) * 1000 + c(3)
        }
    "#, 120 * 1000000 + 192 * 1000 + 54, "letrec-three-inst");
    // a(n)=n!：a(5)=120；b(1)=1,b(2)=4,b(3)=24,b(4)=192；c(1)=1,c(2)=6,c(3)=54
}

/// HOF：JIT 可编译的函数（无 MakeCell）体内经 CallClosure/host_call_indirect
/// 调用 letrec 递归闭包（Vm::call_value 的 Shared cell 解包路径）。
#[test]
fn test_letrec_hof_jit_path() {
    assert_three_way_int(r#"
        fn apply(f: fn(Int) -> Int, x: Int) -> Int {
            f(x)
        }
        fn make(scale: Int) -> fn(Int) -> Int {
            let fact = |n: Int| if n <= 1 { 1 } else { n * fact(n - 1) * scale };
            fact
        }
        fn main() -> Int {
            let f = make(2);
            apply(f, 4)
        }
    "#, 192, "letrec-hof");
    // f(n)=n<=1?1:n*f(n-1)*2：f(1)=1,f(2)=4,f(3)=24,f(4)=192
}

/// 纯尾递归闭包（TCO 路径安全：self_ref 槽位随 TailCallClosure 追加捕获传递）。
#[test]
fn test_letrec_tail_recursive() {
    assert_three_way_int(r#"
        fn main() -> Int {
            // acc 尾递归：f(n, acc) = n<=1 ? acc : f(n-1, acc*n)
            let f = |n: Int, acc: Int| if n <= 1 { acc } else { f(n - 1, acc * n) };
            f(5, 1)
        }
    "#, 120, "letrec-tail");
}

/// 嵌套：外层递归闭包体内再定义递归闭包。
/// inner 自引用（自己的 cell）+ 捕获外层 letrec 名 outer（按值捕获其 cell）。
/// inner(y)=y<=1?1:y*inner(y-1)+outer(0)，outer(0)=1：
///   inner(0)=1, inner(1)=1, inner(2)=3, inner(3)=10
/// outer(x)=x<=0?1:inner(x)+outer(x-1)：
///   outer(0)=1, outer(1)=2, outer(2)=5, outer(3)=15
#[test]
fn test_letrec_nested() {
    assert_three_way_int(r#"
        fn main() -> Int {
            let outer = |x: Int| {
                if x <= 0 { 1 }
                else {
                    let inner = |y: Int| if y <= 1 { 1 } else { y * inner(y - 1) + outer(0) };
                    inner(x) + outer(x - 1)
                }
            };
            outer(3)
        }
    "#, 15, "letrec-nested");
}

/// Debug 有界性：自引用 cell（Value::Shared → FnRef → captures → cell）在
/// `{:?}` 上不得无限递归（M1-S2 把 Value 的 Debug 转发 Display，FnRef 只打印
/// `<fn {name}>`）。返回递归闭包的程序值直接 format Debug 验证无栈溢出。
#[test]
fn test_letrec_debug_bounded() {
    let src = r#"
        fn make_fact(scale: Int) -> fn(Int) -> Int {
            let fact = |n: Int| if n <= 1 { 1 } else { n * fact(n - 1) * scale };
            fact
        }
        fn main() -> fn(Int) -> Int {
            make_fact(2)
        }
    "#;
    let vm = run_vm(src).unwrap_or_else(|e| panic!("[letrec-debug] VM 执行失败: {}", e));
    let debug = format!("{:?}", vm);
    assert!(!debug.is_empty(), "[letrec-debug] Debug 输出为空");
    // 有界输出（Display 转发）：FnRef 打印 <fn ...>，不递归 captures
    assert!(debug.contains("<fn"), "[letrec-debug] 应含 <fn，实际: {}", debug);
}

/// 逃逸 + 多实例：同一工厂多次调用返回的递归闭包并存（最高危场景）。
#[test]
fn test_letrec_factory_multiple_escaped() {
    assert_three_way_int(r#"
        fn make(base: Int) -> fn(Int) -> Int {
            let f = |n: Int| if n <= 1 { base } else { n * f(n - 1) + base };
            f
        }
        fn main() -> Int {
            let f1 = make(10);
            let f2 = make(100);
            // f1(n) = n<=1 ? 10 : n*f1(n-1)+10
            // f1(3) = 3*(2*(1*10)+10)+10 = 3*(30)+10 = 100
            // f2(n) = n<=1 ? 100 : n*f2(n-1)+100
            // f2(3) = 3*(2*100+100)+100 = 900+100 = 1000
            f1(3) * 10000 + f2(3)
        }
    "#, 100 * 10000 + 1000, "letrec-factory-escaped");
}
