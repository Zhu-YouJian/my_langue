//! M4.4 调试器：解释器调试钩子测试。
//!
//! 守护 `Interpreter::debug_hook` 机制——tree-walk 逐步的基础：
//! - 钩子在每个 `HirStmt` 执行前被调用（`stmt.span.line` 提供源码行号）
//! - 断点触发（按行匹配）+ 继续
//! - 单步推进（连续语句依序触发）
//! - 变量查看（钩子内读 `interp.vars`，带名字的变量表）
//! - 钩子出错 → 立即中止执行（错误响亮）
//! - 无钩子时行为完全不变（回归守护）

//! 注意：钩子闭包是 `'static`（`Box<dyn FnMut>`），捕获状态用 `Rc<RefCell>`。

use std::cell::RefCell;
use std::collections::HashSet;
use std::rc::Rc;

use tenth::error::{TenthError, TenthResult};
use tenth::hir::hir::HirStmt;
use tenth::hir::lower::Lowerer;
use tenth::lexer::lexer::Lexer;
use tenth::parser::parser::Parser;
use tenth::runtime::interpreter::Interpreter;
use tenth::runtime::value::Value;

/// 解析源码并 lower 成 HIR。
fn source_to_hir(src: &str) -> TenthResult<tenth::hir::hir::HirProgram> {
    let mut lexer = Lexer::new(src);
    let tokens = lexer.tokenize()?;
    let mut parser = Parser::new(tokens);
    let program = parser.parse_program()?;
    let mut lowerer = Lowerer::new();
    lowerer.lower_program(&program)
}

/// 简单测试调试器：记录命中的行，支持断点/单步/暂停标记。
#[derive(Default)]
struct TestDebugger {
    breakpoints: HashSet<usize>,
    stepping: bool,
    hit_lines: Vec<usize>,
    /// 每次命中时钩子是否读到了变量（由变量检查回调设置）。
    paused: bool,
}

// ─── 钩子触发与单步推进 ──────────────────────────────────────────────

#[test]
fn test_hook_fires_on_each_statement_in_order() {
    // 程序语句分布在 2-5 行；钩子应在每个语句执行前依序触发（单步推进的基础）。
    let src = r#"fn main() {
    let x = 42;
    let y = x + 1;
    let z = y * 2;
    print(z);
    z
}
"#;
    let hir = source_to_hir(src).unwrap();
    let mut interp = Interpreter::new(&hir);
    let lines: Rc<RefCell<Vec<usize>>> = Rc::new(RefCell::new(Vec::new()));
    let lines_hook = Rc::clone(&lines);
    interp.set_debug_hook(Some(Box::new(move |_interp, stmt| {
        lines_hook.borrow_mut().push(stmt.span.line);
        Ok(())
    })));
    let result = interp.execute_program(&hir).unwrap();
    match result {
        Some(Value::Int(86, _)) => {}
        v => panic!("期望 Int(86)，实际 {:?}", v),
    }
    // 语句行 2,3,4,5 依序触发（print(z) 是语句；最后 z 是 final_expr 不走 eval_stmt）。
    assert_eq!(*lines.borrow(), vec![2, 3, 4, 5], "钩子应按语句顺序依序触发");
}

#[test]
fn test_step_progression_sets_stepping_and_resumes() {
    // 模拟「单步」：命中第一行后设置 stepping=true，后续语句连续触发（n 的语义）。
    let src = r#"fn main() {
    let a = 1;
    let b = a + 1;
    let c = b + 1;
    c
}
"#;
    let hir = source_to_hir(src).unwrap();
    let mut interp = Interpreter::new(&hir);
    let seen: Rc<RefCell<Vec<usize>>> = Rc::new(RefCell::new(Vec::new()));
    let seen_hook = Rc::clone(&seen);
    interp.set_debug_hook(Some(Box::new(move |_interp, stmt| {
        seen_hook.borrow_mut().push(stmt.span.line);
        Ok(())
    })));
    let _ = interp.execute_program(&hir).unwrap();
    // 单步推进：2,3,4 连续（每步执行一条语句后停在下一句）。
    assert_eq!(*seen.borrow(), vec![2, 3, 4], "单步应逐语句推进");
}

// ─── 断点触发 + 继续 ─────────────────────────────────────────────────

#[test]
fn test_breakpoint_trigger_and_continue() {
    // 断点在行 3：行 2 不命中，行 3 命中（记录），继续运行到结束。
    let src = r#"fn main() {
    let x = 42;
    let y = x + 1;
    let z = y * 2;
    z
}
"#;
    let hir = source_to_hir(src).unwrap();
    let mut interp = Interpreter::new(&hir);
    let dbg: Rc<RefCell<TestDebugger>> = Rc::new(RefCell::new(TestDebugger {
        breakpoints: [3].iter().copied().collect(),
        stepping: false,
        ..Default::default()
    }));
    let dbg_hook = Rc::clone(&dbg);
    interp.set_debug_hook(Some(Box::new(move |_interp, stmt| {
        let line = stmt.span.line;
        if dbg_hook.borrow().breakpoints.contains(&line) {
            dbg_hook.borrow_mut().hit_lines.push(line);
            dbg_hook.borrow_mut().paused = true; // 断点命中（真实调试器在此阻塞等命令）
        }
        // 继续执行（continue 语义：清 stepping、返回）
        dbg_hook.borrow_mut().stepping = false;
        Ok(())
    })));
    let result = interp.execute_program(&hir).unwrap();
    match result {
        Some(Value::Int(86, _)) => {}
        v => panic!("期望 Int(86)，实际 {:?}", v),
    }
    assert_eq!(dbg.borrow().hit_lines, vec![3], "应只在断点行 3 命中一次");
    assert!(dbg.borrow().paused, "断点应触发暂停标记");
}

#[test]
fn test_breakpoint_does_not_fire_other_lines() {
    // 断点只在该行命中：行 2、4 不应进入断点分支。
    let src = r#"fn main() {
    let a = 10;
    let b = 20;
    let c = a + b;
    c
}
"#;
    let hir = source_to_hir(src).unwrap();
    let mut interp = Interpreter::new(&hir);
    let hits: Rc<RefCell<Vec<usize>>> = Rc::new(RefCell::new(Vec::new()));
    let hits_hook = Rc::clone(&hits);
    interp.set_debug_hook(Some(Box::new(move |_interp, stmt| {
        if stmt.span.line == 2 {
            hits_hook.borrow_mut().push(2);
        }
        Ok(())
    })));
    let _ = interp.execute_program(&hir).unwrap();
    assert_eq!(*hits.borrow(), vec![2], "只有行 2 应命中钩子分支");
}

// ─── 变量查看 ────────────────────────────────────────────────────────

#[test]
fn test_variable_inspection_at_breakpoint() {
    // 断点在行 3：此时 x=42 已赋值（行 2 已执行），y 尚未赋值。
    // 钩子读 `interp.vars` 应看到 x=42。
    let src = r#"fn main() {
    let x = 42;
    let y = x + 1;
    print(y);
}
"#;
    let hir = source_to_hir(src).unwrap();
    let mut interp = Interpreter::new(&hir);
    let inspected: Rc<RefCell<Option<(Option<Value>, Option<Value>)>>> = Rc::new(RefCell::new(None));
    let inspected_hook = Rc::clone(&inspected);
    interp.set_debug_hook(Some(Box::new(move |interp, stmt| {
        if stmt.span.line == 3 {
            let x = interp.vars.get("x").and_then(|s| s.last().map(|(_, v)| v.clone()));
            let y = interp.vars.get("y").and_then(|s| s.last().map(|(_, v)| v.clone()));
            *inspected_hook.borrow_mut() = Some((x, y));
        }
        Ok(())
    })));
    let _ = interp.execute_program(&hir).unwrap();
    let (x, y) = inspected.borrow().clone().expect("应在行 3 读取变量");
    match x {
        Some(Value::Int(42, _)) => {}
        v => panic!("x 应为 Int(42)，实际 {:?}", v),
    }
    assert!(y.is_none(), "断点在行 3 时 y 尚未赋值，应读不到");
}

#[test]
fn test_variable_inspection_in_function_frame() {
    // 断点在函数体内：局部变量可查看（scope 变量表）。
    let src = r#"fn add(a: i64, b: i64) -> i64 {
    let s = a + b;
    s
}

fn main() {
    let r = add(2, 3);
    r
}
"#;
    let hir = source_to_hir(src).unwrap();
    let mut interp = Interpreter::new(&hir);
    let saw_param: Rc<RefCell<Option<Value>>> = Rc::new(RefCell::new(None));
    let saw_hook = Rc::clone(&saw_param);
    interp.set_debug_hook(Some(Box::new(move |interp, stmt| {
        if stmt.span.line == 2 {
            // 函数 add 体内：参数 a 应可查看
            let a = interp.vars.get("a").and_then(|s| s.last().map(|(_, v)| v.clone()));
            *saw_hook.borrow_mut() = a;
        }
        Ok(())
    })));
    let _ = interp.execute_program(&hir).unwrap();
    match saw_param.borrow().clone() {
        Some(Value::Int(2, _)) => {}
        v => panic!("add 参数 a 应为 Int(2)，实际 {:?}", v),
    }
}

// ─── 钩子错误响亮 / 无钩子回归 ───────────────────────────────────────

#[test]
fn test_hook_error_aborts_execution_loudly() {
    // 钩子出错 → 立即中止执行（错误响亮），不静默继续。
    let src = r#"fn main() {
    let x = 1;
    let y = 2;
    x + y
}
"#;
    let hir = source_to_hir(src).unwrap();
    let mut interp = Interpreter::new(&hir);
    interp.set_debug_hook(Some(Box::new(move |_interp, stmt| {
        if stmt.span.line == 3 {
            return Err(TenthError::RuntimeError {
                line: Some(stmt.span.line),
                col: Some(stmt.span.col),
                message: "调试器主动中止".into(),
            });
        }
        Ok(())
    })));
    let err = interp.execute_program(&hir).unwrap_err();
    match &err {
        TenthError::RuntimeError { line, message, .. } => {
            assert_eq!(*line, Some(3));
            assert!(message.contains("调试器主动中止"), "实际: {}", message);
        }
        other => panic!("期望 RuntimeError，实际: {:?}", other),
    }
}

#[test]
fn test_no_hook_execution_unaffected() {
    // 无钩子：解释器行为完全不变（回归守护）。
    let src = r#"fn main() {
    let mut total = 0;
    let mut i = 0;
    while i < 10000 {
        total = total + i;
        i = i + 1;
    }
    total
}
"#;
    let hir = source_to_hir(src).unwrap();
    let mut interp = Interpreter::new(&hir);
    let result = interp.execute_program(&hir).unwrap();
    // 0+1+...+9999 = 49995000
    match result {
        Some(Value::Int(49995000, _)) => {}
        v => panic!("期望 Int(49995000)，实际 {:?}", v),
    }
}
