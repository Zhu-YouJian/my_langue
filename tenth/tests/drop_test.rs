//! M2.5：Drop trait / RAII 测试。
//!
//! 覆盖（两条路径）：
//! - **结构性 HIR 断言（解释器路径）**：lower 后验证 `__dyn_Drop_{Type}_drop`
//!   调用被正确生成——块退出时、嵌套块只生成一次、已 move 变量不生成、
//!   非 Drop 类型不生成、多变量生成多条、函数作用域。
//! - **VM 运行时（全链路）**：自定义 println native 记录副作用，验证
//!   drop 实际执行时序——块退出、嵌套块时序（回归：不再二次 drop）、
//!   函数作用域、move 后单次 drop。
//! - impl Drop 缺少 drop 方法 → 编译期错误。

use tenth::lexer::lexer::Lexer;
use tenth::parser::parser::Parser;
use tenth::hir::lower::Lowerer;
use tenth::hir::hir::{HirExpr, HirExprKind, HirStmt, HirStmtKind};
use tenth::runtime::vm::Vm;
use tenth::runtime::value::Value;
use tenth::compile::bytecode::BytecodeCompiler;
use tenth::error::TenthError;
use std::rc::Rc;
use std::cell::RefCell;
use std::sync::Mutex;

/// VM 运行时记录：`add_native` 要求无捕获的 fn 指针，因此用全局静态日志。
/// 运行时测试在同一 #[test] 内顺序执行、每次清空，避免并行竞争。
static PRINT_LOG: Mutex<Vec<String>> = Mutex::new(Vec::new());

fn recording_println(_vm: &mut Vm, args: &[Value]) -> Result<Value, TenthError> {
    let s: String = args.iter().map(|a| format!("{}", a)).collect();
    PRINT_LOG.lock().unwrap().push(s);
    Ok(Value::Unit)
}

fn to_string_native(_vm: &mut Vm, args: &[Value]) -> Result<Value, TenthError> {
    match args.first() {
        Some(Value::Int(n, _)) => Ok(Value::String(n.to_string())),
        Some(Value::Float(f)) => Ok(Value::String(f.to_string())),
        Some(Value::String(s)) => Ok(Value::String(s.clone())),
        _ => Ok(Value::String("?".into())),
    }
}

fn vec_new_native(_vm: &mut Vm, _args: &[Value]) -> Result<Value, TenthError> {
    Ok(Value::Vec(Rc::new(RefCell::new(Vec::new()))))
}

// ── 结构性 HIR 遍历 ──────────────────────────────────────────────

/// lower 后收集 main（`<expr>` 或 `fn main` 函数体）中所有 `__dyn_Drop_*_drop` 调用。
fn collect_drop_calls(src: &str) -> Result<Vec<String>, String> {
    let mut lexer = Lexer::new(src);
    let tokens = lexer.tokenize().map_err(|e| e.to_string())?;
    let mut parser = Parser::new(tokens);
    let program = parser.parse_program().map_err(|e| e.to_string())?;
    let mut lowerer = Lowerer::new();
    let hir = lowerer.lower_program(&program).map_err(|e| e.to_string())?;
    let mut out = Vec::new();
    if let Some(main) = &hir.main_expr {
        walk_expr(main, &mut out);
    }
    // `fn main() {}` 形式的程序：main 体在 functions 表（HirProgram.main_expr 为 None）
    for f in &hir.functions {
        if f.name == "main" {
            walk_expr(&f.body, &mut out);
        }
    }
    Ok(out)
}

fn walk_expr(e: &HirExpr, out: &mut Vec<String>) {
    match &e.kind {
        HirExprKind::Call { func, args, .. } => {
            if let HirExprKind::Var(name) = &func.kind {
                if name.starts_with("__dyn_Drop_") {
                    out.push(name.clone());
                }
            }
            for a in args { walk_expr(a, out); }
        }
        HirExprKind::Block { stmts, final_expr } => {
            for s in stmts { walk_stmt(s, out); }
            if let Some(fe) = final_expr { walk_expr(fe, out); }
        }
        HirExprKind::If { then_branch, else_branch, .. } => {
            walk_expr(then_branch, out);
            if let Some(eb) = else_branch { walk_expr(eb, out); }
        }
        HirExprKind::Match { arms, .. } => {
            for arm in arms { walk_expr(&arm.body, out); }
        }
        HirExprKind::MethodCall { receiver, args, .. } => {
            walk_expr(receiver, out);
            for a in args { walk_expr(a, out); }
        }
        _ => {}
    }
}

fn walk_stmt(s: &HirStmt, out: &mut Vec<String>) {
    match &s.kind {
        HirStmtKind::Let { init, .. } => {
            if let Some(i) = init { walk_expr(i, out); }
        }
        HirStmtKind::Expr(e) => walk_expr(e, out),
        HirStmtKind::Return(e) => {
            if let Some(e) = e { walk_expr(e, out); }
        }
        HirStmtKind::While { body, .. } => walk_stmt(body, out),
        HirStmtKind::For { body, .. } => walk_stmt(body, out),
        HirStmtKind::Loop { body, .. } => {
            for s in body { walk_stmt(s, out); }
        }
        HirStmtKind::DoWhile { body, .. } => walk_stmt(body, out),
        _ => {}
    }
}

// ── VM 运行时（记录 println）──────────────────────────────────────

/// 用记录型 println/to_string 运行程序，返回 main 执行期间的打印序列快照。
fn run_vm_recorded(src: &str) -> Result<(Value, Vec<String>), String> {
    let mut lexer = Lexer::new(src);
    let tokens = lexer.tokenize().map_err(|e| e.to_string())?;
    let mut parser = Parser::new(tokens);
    let program = parser.parse_program().map_err(|e| e.to_string())?;
    let mut lowerer = Lowerer::new();
    let hir = lowerer.lower_program(&program).map_err(|e| e.to_string())?;

    let mut vm = Vm::new();
    vm.add_native("println".into(), recording_println);
    vm.add_native("to_string".into(), to_string_native);
    vm.add_native("Vec::new".into(), vec_new_native);

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
    let main_val = if let Some(ref expr) = hir.main_expr {
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
        vm.call("main").map_err(|e| e.to_string())?
    } else if vm.has_fn("main") {
        vm.call("main").map_err(|e| e.to_string())?
    } else {
        Value::Unit
    };
    Ok((main_val, PRINT_LOG.lock().unwrap().clone()))
}

const RES_DEF: &str = r#"
struct Res { id: i64 }
impl Drop for Res {
    fn drop(self) { println("DROP:" + to_string(self.id)) }
}
"#;

// ── 结构性：drop 调用生成 ─────────────────────────────────────────

#[test]
fn drop_hir_generates_drop_call_in_main() {
    let src = format!(r#"{RES_DEF}
        fn main() {{
            let a = Res {{ id: 1 }}
        }}
    "#);
    let calls = collect_drop_calls(&src).unwrap();
    assert_eq!(calls.len(), 1, "main 中一个 Drop 变量应生成一次 drop 调用");
    assert!(calls[0].starts_with("__dyn_Drop_Res_drop"), "调用目标应为 __dyn_Drop_Res_drop，实际 {}", calls[0]);
}

#[test]
fn drop_hir_nested_block_single_call() {
    // 嵌套块：x 只在内层块退出时 drop 一次（外层不得再 drop 同一变量）
    let src = format!(r#"{RES_DEF}
        fn main() {{
            {{
                let x = Res {{ id: 10 }}
            }}
            let y = Res {{ id: 20 }}
        }}
    "#);
    let calls = collect_drop_calls(&src).unwrap();
    assert_eq!(calls.len(), 2, "内层 x + 外层 y = 2 次 drop 调用（x 不应被外层重复 drop），实际 {calls:?}");
}

#[test]
fn drop_hir_moved_var_not_dropped() {
    // `let b = move a` 后 a 不再拥有值，只 drop b
    let src = format!(r#"{RES_DEF}
        fn main() {{
            let a = Res {{ id: 1 }}
            let b = move a
        }}
    "#);
    let calls = collect_drop_calls(&src).unwrap();
    assert_eq!(calls.len(), 1, "move 后只 drop 接收者（1 次），实际 {calls:?}");
}

#[test]
fn drop_hir_non_drop_type_no_call() {
    let src = r#"
        struct Plain { id: i64 }
        fn main() {
            let a = Plain { id: 1 }
        }
    "#;
    let calls = collect_drop_calls(&src).unwrap();
    assert!(calls.is_empty(), "非 Drop 类型不应生成 drop 调用，实际 {calls:?}");
}

#[test]
fn drop_hir_multiple_vars_multiple_calls() {
    let src = format!(r#"{RES_DEF}
        fn main() {{
            let a = Res {{ id: 1 }}
            let b = Res {{ id: 2 }}
            let c = Res {{ id: 3 }}
        }}
    "#);
    let calls = collect_drop_calls(&src).unwrap();
    assert_eq!(calls.len(), 3, "3 个 Drop 变量应生成 3 次 drop 调用，实际 {calls:?}");
}

#[test]
fn drop_hir_function_scope() {
    // 函数体内的 Drop 变量在函数体块退出时 drop
    let src = format!(r#"{RES_DEF}
        fn make() {{
            let r = Res {{ id: 7 }}
        }}
        fn main() {{
            make()
        }}
    "#);
    // main_expr 中无直接 Drop 变量（make 体内的 drop 在 make 函数块中）
    let calls = collect_drop_calls(&src).unwrap();
    assert!(calls.is_empty(), "main 表达式自身无 Drop 变量（make 的 drop 在其函数体内），实际 {calls:?}");
}

#[test]
fn drop_impl_missing_method_errors() {
    let src = r#"
        struct Res { id: i64 }
        impl Drop for Res {
        }
        fn main() { let a = Res { id: 1 } }
    "#;
    let result = collect_drop_calls(src);
    assert!(result.is_err(), "impl Drop 缺少 drop 方法应报编译期错误");
}

// ── VM 运行时：drop 实际执行与时序 ─────────────────────────────────
// 运行时测试共用全局 PRINT_LOG（add_native 要求无捕获 fn 指针），
// 故合并为单个 #[test] 顺序执行各场景，每次清空日志避免并行竞争。

#[test]
fn drop_vm_runtime_scenarios() {
    // 场景 1：块退出时 drop（END 之后）
    {
        let src = format!(r#"{RES_DEF}
            fn main() {{
                let a = Res {{ id: 1 }}
                println("END")
            }}
        "#);
        PRINT_LOG.lock().unwrap().clear();
        let (_, log) = run_vm_recorded(&src).unwrap();
        assert_eq!(log, vec!["END".to_string(), "DROP:1".to_string()],
            "场景1：drop 应在 main 作用域退出时执行（END 之后）");
    }

    // 场景 2：嵌套块时序——内层 x 在内层块退出时 drop 一次（回归：不得二次 drop），
    // 外层 y 在 main 退出时 drop。
    {
        let src = format!(r#"{RES_DEF}
            fn main() {{
                {{
                    let x = Res {{ id: 10 }}
                    println("IN")
                }}
                println("OUT")
                let y = Res {{ id: 20 }}
                println("END")
            }}
        "#);
        PRINT_LOG.lock().unwrap().clear();
        let (_, log) = run_vm_recorded(&src).unwrap();
        assert_eq!(log,
            vec!["IN".to_string(), "DROP:10".to_string(), "OUT".to_string(), "END".to_string(), "DROP:20".to_string()],
            "场景2：内层 x 在 IN 后立即 drop；y 在 main 退出时 drop；x 不得被二次 drop");
    }

    // 场景 3：函数作用域——make 体内的 r 在 make 返回时 drop（MAKE 之后、MAIN 之前）
    {
        let src = format!(r#"{RES_DEF}
            fn make() {{
                let r = Res {{ id: 7 }}
                println("MAKE")
            }}
            fn main() {{
                make()
                println("MAIN")
            }}
        "#);
        PRINT_LOG.lock().unwrap().clear();
        let (_, log) = run_vm_recorded(&src).unwrap();
        assert_eq!(log, vec!["MAKE".to_string(), "DROP:7".to_string(), "MAIN".to_string()],
            "场景3：make 体内的 r 应在 make 返回时 drop（MAKE 之后、MAIN 之前）");
    }

    // 场景 4：move 后只 drop 接收者（a 已转移所有权）——单一资源只释放一次
    {
        let src = format!(r#"{RES_DEF}
            fn main() {{
                let a = Res {{ id: 1 }}
                let b = move a
                println("M")
            }}
        "#);
        PRINT_LOG.lock().unwrap().clear();
        let (_, log) = run_vm_recorded(&src).unwrap();
        assert_eq!(log, vec!["M".to_string(), "DROP:1".to_string()],
            "场景4：move 后应只 drop 一次，实际 {log:?}");
    }
}
