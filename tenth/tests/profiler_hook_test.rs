//! M4.4 性能分析：VM 剖析钩子测试。
//!
//! 守护 `Vm::profile_hook` 机制——函数级计时/调用计数的基础：
//! - 钩子在函数入口/出口（chunk 切换）发出 Enter/Exit 事件
//! - 调用计数正确（递归函数 fib(10) = 177 次调用）
//! - Enter/Exit 平衡（顶层入口函数无 Exit，Enter = Exit + 1）
//! - 区间累积（inclusive 时间）：父函数总时间 >= 子函数总时间（嵌套正确）
//! - 计时正确性：事件序列为 Enter→(Enter/Exit)*→程序结束
//! - 无钩子时行为完全不变（回归守护）

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;
use std::time::Instant;

use tenth::compile::bytecode::BytecodeCompiler;
use tenth::error::{TenthError, TenthResult};
use tenth::hir::lower::Lowerer;
use tenth::lexer::lexer::Lexer;
use tenth::parser::parser::Parser;
use tenth::runtime::natives::register_all_natives;
use tenth::runtime::value::Value;
use tenth::runtime::vm::{Vm, VmProfileEvent, VmProfileKind};

/// 剖析事件：函数名 + 种类 + 时间戳（验证区间累积用）。
type ProfEvent = (String, VmProfileKind, Instant);

/// 解析源码并 lower 成 HIR。
fn source_to_hir(src: &str) -> TenthResult<tenth::hir::hir::HirProgram> {
    let mut lexer = Lexer::new(src);
    let tokens = lexer.tokenize()?;
    let mut parser = Parser::new(tokens);
    let program = parser.parse_program()?;
    let mut lowerer = Lowerer::new();
    lowerer.lower_program(&program)
}

/// 通过 VM（非 JIT）执行源码；剖析钩子把事件（含时间戳）记录进 `events`。
fn run_vm_prof(
    src: &str,
    events: Rc<RefCell<Vec<ProfEvent>>>,
) -> Result<Value, TenthError> {
    let hir = source_to_hir(src)?;

    let mut vm = Vm::new();
    register_all_natives(&mut vm);

    for func in &hir.functions {
        let compiler = BytecodeCompiler::new();
        let (chunk, closures) = compiler.compile(func)?;
        vm.add_fn(func.name.clone(), chunk);
        for (name, closure_chunk) in closures {
            vm.add_fn(name, closure_chunk);
        }
        vm.set_global(
            func.name.clone(),
            Value::FnRef {
                name: func.name.clone(),
                params: func.params.clone(),
                return_type: func.return_type.clone(),
                captures: vec![],
            },
        );
    }

    let events_hook = Rc::clone(&events);
    vm.set_profile_hook(Some(Box::new(
        move |vm: &mut Vm, event: VmProfileEvent| -> TenthResult<()> {
            let name = vm
                .chunk_name_at(event.chunk_idx)
                .unwrap_or_else(|| format!("__chunk_{}", event.chunk_idx));
            events_hook.borrow_mut().push((name, event.kind, Instant::now()));
            Ok(())
        },
    )));

    if let Some(ref expr) = hir.main_expr {
        let compiler = BytecodeCompiler::new();
        let (chunk, closures) = compiler.compile_main(expr)?;
        vm.add_fn("main".into(), chunk);
        for (name, closure_chunk) in closures {
            vm.add_fn(name, closure_chunk);
        }
        vm.call("main")
    } else if vm.has_fn("main") {
        vm.call("main")
    } else {
        Ok(Value::Unit)
    }
}

// ─── 调用计数 / 事件序列 ─────────────────────────────────────────────

#[test]
fn test_profiler_hook_counts_recursive_calls() {
    // fib(10) 共 177 次调用：Enter(fib) 应恰好 177 次。
    let src = r#"fn fib(n: i64) -> i64 {
    if n < 2 { n } else { fib(n - 1) + fib(n - 2) }
}

fn main() {
    let r = fib(10);
    r
}
"#;
    let events: Rc<RefCell<Vec<ProfEvent>>> = Rc::new(RefCell::new(Vec::new()));
    let result = run_vm_prof(src, Rc::clone(&events)).unwrap();
    match result {
        Value::Int(55, _) => {}
        v => panic!("fib(10) 应返回 55，实际 {:?}", v),
    }
    let ev = events.borrow();
    let fib_enter = ev.iter().filter(|(n, k, _)| n == "fib" && *k == VmProfileKind::Enter).count();
    let fib_exit = ev.iter().filter(|(n, k, _)| n == "fib" && *k == VmProfileKind::Exit).count();
    assert_eq!(fib_enter, 177, "fib 应被调用 177 次");
    assert_eq!(fib_exit, 177, "fib 的 Enter/Exit 应平衡");

    // 主函数只进入一次。
    let main_enter = ev.iter().filter(|(n, k, _)| n == "main" && *k == VmProfileKind::Enter).count();
    assert_eq!(main_enter, 1, "main 应只进入一次");
}

#[test]
fn test_profiler_enter_exit_balance() {
    // 全局：Enter 总数 = Exit 总数 + 1（顶层入口函数经隐式 Ret 完成，无 Exit）。
    let src = r#"fn helper(x: i64) -> i64 {
    x * 2
}

fn main() {
    let mut s = 0;
    let mut i = 0;
    while i < 100 {
        s = s + helper(i);
        i = i + 1;
    }
    s
}
"#;
    let events: Rc<RefCell<Vec<ProfEvent>>> = Rc::new(RefCell::new(Vec::new()));
    let result = run_vm_prof(src, Rc::clone(&events)).unwrap();
    match result {
        Value::Int(9900, _) => {} // 2*(0+...+99) = 9900
        v => panic!("期望 Int(9900)，实际 {:?}", v),
    }
    let ev = events.borrow();
    let enters = ev.iter().filter(|(_, k, _)| *k == VmProfileKind::Enter).count();
    let exits = ev.iter().filter(|(_, k, _)| *k == VmProfileKind::Exit).count();
    assert_eq!(enters, exits + 1, "Enter 应比 Exit 多 1（入口函数未显式 Exit）");

    // helper 被调用 100 次。
    let helper_enter = ev.iter().filter(|(n, k, _)| n == "helper" && *k == VmProfileKind::Enter).count();
    assert_eq!(helper_enter, 100, "helper 应被调用 100 次");
}

// ─── 计时正确性（区间累积 / inclusive 时间）──────────────────────────

#[test]
fn test_profiler_timing_nested_inclusive() {
    // 栈式 inclusive 计时：helper 只被调用一次 → helper 的区间是 main 区间的
    // 子集（main 全程活跃），故 main_total > helper_total，且两者均非零。
    // （注意：多嵌套下各函数 inclusive 区间相互重叠，总和可超墙钟——
    //   因此不能用「父 >= 所有子之和」断言。）
    let src = r#"fn helper(x: i64) -> i64 {
    let mut s = 0;
    let mut i = 0;
    while i < 20000 {
        s = s + x;
        i = i + 1;
    }
    s
}

fn main() {
    let r = helper(1);
    r
}
"#;
    let events: Rc<RefCell<Vec<ProfEvent>>> = Rc::new(RefCell::new(Vec::new()));
    let result = run_vm_prof(src, Rc::clone(&events)).unwrap();
    match result {
        Value::Int(20000, _) => {}
        v => panic!("helper(1) 应返回 20000，实际 {:?}", v),
    }

    // 用记录的真实时间戳做栈式 inclusive 累积（与剖析器工具相同逻辑），
    // 验证嵌套计时正确归属：Enter 压栈、Exit 弹栈、结束冲刷剩余栈。
    let ev = events.borrow();
    let mut totals: HashMap<String, u128> = HashMap::new();
    let mut stack: Vec<(String, Instant)> = Vec::new();
    for (name, kind, ts) in ev.iter() {
        match kind {
            VmProfileKind::Enter => stack.push((name.clone(), *ts)),
            VmProfileKind::Exit => {
                if let Some((n, t0)) = stack.pop() {
                    let dt = ts.duration_since(t0).as_nanos();
                    *totals.entry(n).or_insert(0) += dt;
                }
            }
        }
    }
    // 冲刷剩余栈（顶层入口函数 main 从未显式 Exit）。
    let end = Instant::now();
    for (n, t0) in stack.drain(..) {
        let dt = end.duration_since(t0).as_nanos();
        *totals.entry(n).or_insert(0) += dt;
    }

    let helper_total = totals.get("helper").copied().unwrap_or(0);
    let main_total = totals.get("main").copied().unwrap_or(0);
    assert!(helper_total > 0, "helper 应有非零耗时");
    assert!(main_total > 0, "main 应有非零耗时");
    assert!(
        main_total > helper_total,
        "单次调用的 helper 区间应严格小于 main 全程区间"
    );
}

// ─── 钩子可中止（错误响亮）──────────────────────────────────────────

#[test]
fn test_profiler_hook_error_aborts() {
    // 钩子出错 → 立即中止 VM 执行。
    let src = r#"fn main() {
    let x = 1;
    x + 1
}
"#;
    let events: Rc<RefCell<Vec<ProfEvent>>> = Rc::new(RefCell::new(Vec::new()));
    let err_events = Rc::clone(&events);
    let hir = source_to_hir(src).unwrap();
    let mut vm = Vm::new();
    register_all_natives(&mut vm);
    for func in &hir.functions {
        let compiler = BytecodeCompiler::new();
        let (chunk, closures) = compiler.compile(func).unwrap();
        vm.add_fn(func.name.clone(), chunk);
        for (name, closure_chunk) in closures {
            vm.add_fn(name, closure_chunk);
        }
    }
    vm.set_profile_hook(Some(Box::new(
        move |vm: &mut Vm, event: VmProfileEvent| -> TenthResult<()> {
            let name = vm
                .chunk_name_at(event.chunk_idx)
                .unwrap_or_else(|| "?".to_string());
            err_events.borrow_mut().push((name, event.kind, Instant::now()));
            // 首个事件（Enter main）即主动报错——验证错误响亮中止。
            Err(TenthError::RuntimeError {
                line: None,
                col: None,
                message: "剖析器主动中止".into(),
            })
        },
    )));
    let result = if let Some(ref expr) = hir.main_expr {
        let compiler = BytecodeCompiler::new();
        let (chunk, closures) = compiler.compile_main(expr).unwrap();
        vm.add_fn("main".into(), chunk);
        for (name, closure_chunk) in closures {
            vm.add_fn(name, closure_chunk);
        }
        vm.call("main")
    } else {
        vm.call("main")
    };
    match result {
        Err(TenthError::RuntimeError { message, .. }) => {
            assert!(message.contains("剖析器主动中止"), "实际: {}", message);
        }
        other => panic!("期望剖析器中止错误，实际: {:?}", other),
    }
}

// ─── 无钩子回归 ──────────────────────────────────────────────────────

#[test]
fn test_no_profiler_hook_unaffected() {
    // 无剖析钩子：VM 行为完全不变（回归守护）。
    let src = r#"fn main() {
    let mut total = 0;
    let mut i = 0;
    while i < 100000 {
        total = total + 1;
        i = i + 1;
    }
    total
}
"#;
    let hir = source_to_hir(src).unwrap();
    let mut vm = Vm::new();
    register_all_natives(&mut vm);
    for func in &hir.functions {
        let compiler = BytecodeCompiler::new();
        let (chunk, closures) = compiler.compile(func).unwrap();
        vm.add_fn(func.name.clone(), chunk);
        for (name, closure_chunk) in closures {
            vm.add_fn(name, closure_chunk);
        }
    }
    let result = if let Some(ref expr) = hir.main_expr {
        let compiler = BytecodeCompiler::new();
        let (chunk, closures) = compiler.compile_main(expr).unwrap();
        vm.add_fn("main".into(), chunk);
        for (name, closure_chunk) in closures {
            vm.add_fn(name, closure_chunk);
        }
        vm.call("main")
    } else {
        vm.call("main")
    };
    match result {
        Ok(Value::Int(100000, _)) => {}
        other => panic!("期望 Int(100000)，实际 {:?}", other),
    }
}
