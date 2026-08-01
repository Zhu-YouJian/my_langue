//! M3.3：声明式宏（最小可行版本）测试。
//!
//! 设计（最小可行版本）：
//! - 声明：`macro name(param1, param2) { body_expr }`（body 是表达式模板）
//! - 调用：`name(args)`（与函数调用同形），编译期展开为 body（参数按名代入）
//! - 展开时机：`parse_program` 末尾的 AST pass（parse 后、lower 前）
//! - 嵌套：宏体内可调用其他宏（含自身），递归展开 + 深度上限 64 防死循环
//! - 边界：参数个数不匹配 / 重复定义 / 递归超深 → 编译期 ParseError；
//!   未定义名调用照常走函数调用路径（lower 报未定义函数）；
//!   展开后类型错误正常报（与手写等价）
//! - 不做：hygiene（标识符捕获）、模式匹配宏、过程宏、`name!(args)` 语法、
//!   tenthc 语法层（自举编译器不解析宏，标注遗留）
//! - 宏与函数同名时：宏在调用点优先展开（编译期构造，文档标注）

use tenth::lexer::lexer::Lexer;
use tenth::parser::parser::Parser;
use tenth::hir::lower::Lowerer;
use tenth::runtime::interpreter::Interpreter;
use tenth::runtime::vm::Vm;
use tenth::runtime::value::Value;
use tenth::compile::bytecode::BytecodeCompiler;

/// 解释器路径。
fn run(src: &str) -> Result<Option<Value>, String> {
    let mut lexer = Lexer::new(src);
    let tokens = lexer.tokenize().map_err(|e| e.to_string())?;
    let mut parser = Parser::new(tokens);
    let program = parser.parse_program().map_err(|e| e.to_string())?;
    let mut lowerer = Lowerer::new();
    let hir = lowerer.lower_program(&program).map_err(|e| e.to_string())?;
    let mut interpreter = Interpreter::new(&hir);
    interpreter.execute_program(&hir).map_err(|e| e.to_string())
}

/// VM 路径。
fn run_vm(src: &str) -> Result<Value, String> {
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
    vm.add_native("assert".into(), |_vm, args| {
        if let Some(Value::Bool(b)) = args.first() {
            assert!(*b, "assertion failed");
        }
        Ok(Value::Unit)
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
        vm.call("main").map_err(|e| e.to_string())
    } else {
        Ok(Value::Unit)
    }
}

fn expect_int(v: &Value, expected: i64, tag: &str) {
    match v {
        Value::Int(n, _) => assert_eq!(*n, expected, "{}: 期望 {}，实际 {}", tag, expected, n),
        other => panic!("{}: 期望 Int({})，实际 {:?}", tag, expected, other),
    }
}

// ── 基本展开（任务示例）──────────────────────────────────────────────────

#[test]
fn test_basic_expand() {
    // 任务示例：`macro twice(x) { x + x }` + `{ twice(21) }` → 42
    let src = r#"
    macro twice(x) { x + x }
    { twice(21) }
    "#;
    let r = run(src).unwrap();
    expect_int(&r.unwrap(), 42, "解释器");

    let r = run_vm(src).unwrap();
    expect_int(&r, 42, "VM");
}

#[test]
fn test_multi_param() {
    let src = r#"
    macro add3(a, b, c) { a + b + c }
    { add3(1, 2, 3) }
    "#;
    expect_int(&run(src).unwrap().unwrap(), 6, "解释器");
    expect_int(&run_vm(src).unwrap(), 6, "VM");
}

#[test]
fn test_zero_param() {
    // 0 参宏：带括号调用
    let src = r#"
    macro forty_two() { 42 }
    { forty_two() }
    "#;
    expect_int(&run(src).unwrap().unwrap(), 42, "解释器(带括号)");
    expect_int(&run_vm(src).unwrap(), 42, "VM(带括号)");

    // 0 参宏：声明可省略括号
    let src2 = r#"
    macro seven { 7 }
    { seven() }
    "#;
    expect_int(&run(src2).unwrap().unwrap(), 7, "解释器(声明无括号)");
    expect_int(&run_vm(src2).unwrap(), 7, "VM(声明无括号)");
}

// ── 嵌套与组合 ───────────────────────────────────────────────────────────

#[test]
fn test_nested_macro_call_in_args() {
    // 实参本身是宏调用
    let src = r#"
    macro twice(x) { x + x }
    { twice(twice(5)) }
    "#;
    // 内层 twice(5) → 5+5=10，外层 twice(10) → 20
    expect_int(&run(src).unwrap().unwrap(), 20, "解释器");
    expect_int(&run_vm(src).unwrap(), 20, "VM");
}

#[test]
fn test_macro_calling_macro() {
    // 宏体内调用其他宏（一层嵌套展开）
    let src = r#"
    macro sq(x) { x * x }
    macro sum_sq(a, b) { sq(a) + sq(b) }
    { sum_sq(3, 4) }
    "#;
    expect_int(&run(src).unwrap().unwrap(), 25, "解释器");
    expect_int(&run_vm(src).unwrap(), 25, "VM");
}

#[test]
fn test_arg_is_expression() {
    // 实参是任意表达式（非简单标识符）
    let src = r#"
    macro dbl(x) { x + x }
    { dbl(3 * 7) }
    "#;
    expect_int(&run(src).unwrap().unwrap(), 42, "解释器");
    expect_int(&run_vm(src).unwrap(), 42, "VM");
}

#[test]
fn test_macro_forward_reference() {
    // 宏先使用后定义（定义收集与使用顺序无关）
    let src = r#"
    { twice(10) }
    macro twice(x) { x + x }
    "#;
    expect_int(&run(src).unwrap().unwrap(), 20, "解释器");
    expect_int(&run_vm(src).unwrap(), 20, "VM");
}

#[test]
fn test_expansion_equivalent_to_handwritten() {
    // 展开后与手写等价
    let src = r#"
    macro twice(x) { x + x }
    { twice(21) }
    "#;
    let handwritten = r#"
    { 21 + 21 }
    "#;
    let a = run(src).unwrap().unwrap();
    let b = run(handwritten).unwrap().unwrap();
    expect_int(&a, 42, "宏展开");
    expect_int(&b, 42, "手写");
}

// ── 宏在各类代码结构中使用（walker 覆盖面）──────────────────────────────

#[test]
fn test_macro_in_function_body() {
    let src = r#"
    macro twice(x) { x + x }
    fn f(n: i64) -> i64 { twice(n) }
    { f(21) }
    "#;
    expect_int(&run(src).unwrap().unwrap(), 42, "解释器");
    expect_int(&run_vm(src).unwrap(), 42, "VM");
}

#[test]
fn test_macro_in_if() {
    let src = r#"
    macro dbl(x) { x + x }
    { if true { dbl(10) } else { 0 } }
    "#;
    expect_int(&run(src).unwrap().unwrap(), 20, "解释器");
    expect_int(&run_vm(src).unwrap(), 20, "VM");
}

#[test]
fn test_macro_in_match() {
    let src = r#"
    macro dbl(x) { x + x }
    { match 3 { 3 => dbl(5), _ => 0 } }
    "#;
    expect_int(&run(src).unwrap().unwrap(), 10, "解释器");
    expect_int(&run_vm(src).unwrap(), 10, "VM");
}

#[test]
fn test_macro_in_block_and_struct() {
    // 块内 + 结构体字段值
    let src = r#"
    struct P { x: i64, y: i64 }
    macro dbl(x) { x + x }
    { let a = { dbl(2) }; let p = P { x: dbl(3), y: 4 }; a + p.x + p.y }
    "#;
    expect_int(&run(src).unwrap().unwrap(), 14, "解释器");
    expect_int(&run_vm(src).unwrap(), 14, "VM");
}

// ── 错误边界 ─────────────────────────────────────────────────────────────

#[test]
fn test_undefined_macro_call_errors() {
    // 未定义名调用：不是宏也不是函数 → lower 报错
    let src = r#"
    { undefined_macro(1) }
    "#;
    assert!(run(src).is_err(), "未定义调用应报错（解释器）");
    assert!(run_vm(src).is_err(), "未定义调用应报错（VM）");
}

#[test]
fn test_arg_count_mismatch() {
    let src = r#"
    macro f(a, b) { a + b }
    { f(1) }
    "#;
    let err = run(src).unwrap_err();
    assert!(
        err.contains("参数个数"),
        "参数个数不匹配应报清晰错误，实际: {}",
        err
    );
    assert!(run_vm(src).is_err(), "VM 路径也应报错");
}

#[test]
fn test_duplicate_definition() {
    let src = r#"
    macro f(x) { x }
    macro f(y) { y }
    { f(1) }
    "#;
    let err = run(src).unwrap_err();
    assert!(err.contains("重复定义"), "重复定义应报错，实际: {}", err);
}

#[test]
fn test_recursive_macro_depth() {
    // 递归宏：展开后无限增长 → 深度上限报错（防死循环）
    let src = r#"
    macro f(x) { f(x) }
    { f(1) }
    "#;
    let err = run(src).unwrap_err();
    assert!(
        err.contains("最大深度"),
        "递归宏应报深度上限错误，实际: {}",
        err
    );
}

#[test]
fn test_type_check_after_expansion() {
    // 展开后类型错误正常报（与手写等价：true + true 是类型错误）
    let src = r#"
    macro twice(x) { x + x }
    { twice(true) }
    "#;
    let err = run(src).unwrap_err();
    assert!(
        err.contains("类型") || err.contains("bool") || err.contains("不匹配"),
        "展开后类型检查应报错，实际: {}",
        err
    );
    assert!(run_vm(src).is_err(), "VM 路径也应报错");
}

// ── 语义取舍记录 ─────────────────────────────────────────────────────────

#[test]
fn test_macro_shadows_function_call() {
    // 宏与函数同名时，调用点宏优先展开（编译期构造，文档标注）。
    // 函数 m 保留但不会被调用。
    let src = r#"
    macro m(x) { x + 100 }
    fn m(a: i64) -> i64 { a }
    { m(1) }
    "#;
    expect_int(&run(src).unwrap().unwrap(), 101, "解释器");
    expect_int(&run_vm(src).unwrap(), 101, "VM");
}
