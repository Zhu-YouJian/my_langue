//! S4b-2（M1）：无 init 的 `let` 语义守护测试。
//!
//! 决策（调研后修正）：Tenth 存在既定用法——tenthc 自举编译器用
//! `let char_val: str;`（带类型注解、先声明后赋值，自举路径不可破坏）。
//! 因此语义定为：
//!   - `let x: T;`（带类型注解、无 init）→ 合法，运行时取类型零值默认
//!     （str→""、i64/i32→0、f64→0.0、bool→false），VM=解释器一致；
//!   - `let x;`（无注解、无 init）→ 编译期 TypeError（无法推断类型）；
//!   - 顶层无 init（`let x;` / `let x: T;`）→ 编译期 TypeError（全局必须带初始值）。
//!
//! 覆盖：函数内无注解报错 / 顶层（被引用提升路径）报错 / 顶层（未引用留在 main）报错 /
//! 顶层带注解报错 / mut 无注解报错 / 带注解零值默认（VM+解释器）/ tenthc 式先声明后赋值。

use tenth::lexer::lexer::Lexer;
use tenth::parser::parser::Parser;
use tenth::hir::lower::Lowerer;
use tenth::runtime::interpreter::Interpreter;
use tenth::runtime::vm::Vm;
use tenth::runtime::value::Value;
use tenth::runtime::natives::register_all_natives;
use tenth::compile::bytecode::BytecodeCompiler;

/// 只做 lower（用于断言编译期错误）。
fn lower_error(src: &str) -> Result<(), String> {
    let mut lexer = Lexer::new(src);
    let tokens = lexer.tokenize().map_err(|e| e.to_string())?;
    let mut parser = Parser::new(tokens);
    let program = parser.parse_program().map_err(|e| e.to_string())?;
    let mut lowerer = Lowerer::new();
    lowerer.lower_program(&program).map(|_| ()).map_err(|e| e.to_string())
}

/// 解释器路径。
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

/// 纯 VM 路径。
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
        // `fn main() {...}` 在 hir.functions 中已注册，直接调用
        vm.call("main").map_err(|e| e.to_string())
    } else {
        Ok(Value::Unit)
    }
}

#[test]
fn let_no_init_function_local_errors() {
    let src = r#"
    fn main() {
        let x;
        x
    }
    "#;
    let err = lower_error(src).expect_err("函数内 let x; 应编译期报错");
    assert!(err.contains("必须初始化"), "错误信息应提示必须初始化，实际: {}", err);
}

#[test]
fn let_no_init_toplevel_referenced_errors() {
    // 顶层 let x; 被 main 引用 → 被提升为全局（collect_toplevel_globals 路径）→ 报错
    let src = r#"
    let x;
    fn main() { x }
    "#;
    let err = lower_error(src).expect_err("被引用的顶层 let x; 应编译期报错");
    assert!(err.contains("必须初始化"), "错误信息应提示必须初始化，实际: {}", err);
}

#[test]
fn let_no_init_toplevel_unreferenced_errors() {
    // 未被函数引用的顶层 let x; 留在 main_expr → 走 lower_stmt 同样报错
    let src = r#"
    let x;
    fn main() { 1 }
    "#;
    let err = lower_error(src).expect_err("未被引用的顶层 let x; 也应编译期报错");
    assert!(err.contains("必须初始化"), "错误信息应提示必须初始化，实际: {}", err);
}

#[test]
fn let_no_init_toplevel_annotated_errors() {
    // 顶层带注解无 init 也报错（全局必须带初始值；无既有用法）
    let src = r#"
    let x: i64;
    fn main() { x }
    "#;
    let err = lower_error(src).expect_err("顶层 let x: i64; 应编译期报错");
    assert!(err.contains("必须初始化"), "错误信息应提示必须初始化，实际: {}", err);
}

#[test]
fn let_no_init_mut_errors() {
    let src = r#"
    fn main() {
        let mut x;
        x = 1;
        x
    }
    "#;
    let err = lower_error(src).expect_err("let mut x; 也应编译期报错");
    assert!(err.contains("必须初始化或标注类型"), "错误信息应提示必须初始化，实际: {}", err);
}

// ── 带注解无 init：类型零值默认（VM=解释器一致）────────────────────────────

#[test]
fn let_annotated_no_init_int_default() {
    // `let x: i64;` → 零值 0
    let src = r#"
    fn main() -> i64 {
        let x: i64;
        x
    }
    "#;
    let vm = run_vm(src).unwrap();
    let interp = run_interp(src).unwrap();
    match (vm, interp) {
        (Value::Int(a, _), Value::Int(b, _)) => {
            assert_eq!(a, 0, "VM let x: i64; 默认应为 0");
            assert_eq!(b, 0, "解释器 let x: i64; 默认应为 0");
        }
        (vm, interp) => panic!("两路径都应为 Int(0)，VM={:?} 解释器={:?}", vm, interp),
    }
}

#[test]
fn let_annotated_no_init_str_default() {
    // `let x: str;` → 零值 ""（tenthc 自举编译器先声明后赋值的类型）
    let src = r#"
    fn main() -> i64 {
        let x: str;
        if x.len() == 0 { 1 } else { 0 }
    }
    "#;
    let vm = run_vm(src).unwrap();
    let interp = run_interp(src).unwrap();
    match (vm, interp) {
        (Value::Int(a, _), Value::Int(b, _)) => {
            assert_eq!(a, 1, "VM let x: str; 默认应为空串（len==0）");
            assert_eq!(b, 1, "解释器 let x: str; 默认应为空串（len==0）");
        }
        (vm, interp) => panic!("两路径都应为 Int(1)，VM={:?} 解释器={:?}", vm, interp),
    }
}

#[test]
fn let_annotated_no_init_declare_then_assign() {
    // tenthc 式先声明后赋值（`let char_val: str;` … `char_val = c;`）双路径一致
    let src = r#"
    fn pick(c: i64) -> str {
        let r: str;
        if c == 1 { r = "one"; } else { r = "other"; };
        r
    }
    fn main() -> i64 {
        let a = pick(1);
        let b = pick(2);
        if a == "one" && b == "other" { 1 } else { 0 }
    }
    "#;
    let vm = run_vm(src).unwrap();
    let interp = run_interp(src).unwrap();
    match (vm, interp) {
        (Value::Int(a, _), Value::Int(b, _)) => {
            assert_eq!(a, 1, "VM 先声明后赋值应正确");
            assert_eq!(b, 1, "解释器先声明后赋值应正确");
        }
        (vm, interp) => panic!("两路径都应为 Int(1)，VM={:?} 解释器={:?}", vm, interp),
    }
}

#[test]
fn let_with_init_still_ok() {
    // 有 init 的 let（含 mut / 类型注解）不回归
    let src = r#"
    fn main() -> i64 {
        let x = 42;
        let mut y: i64 = 1;
        y = x + y;
        y
    }
    "#;
    lower_error(src).expect("有 init 的 let 应正常编译");
}
