//! M1.4 生命周期语义化决策落地测试
//!
//! 决策（总师 M1.4，2026-08-01）：Tenth 的借用检查是**语句粒度保守近似**
//! （T19 论文），不消费显式 lifetime 标注。`&'a T` 生命周期语法**可解析并透传**
//! （Type::Ref/MutRef 的第二参数 Option<String>），但**不参与借用检查语义**。
//! 这是设计取舍（与 GC/goto 同列），不是缺陷。
//!
//! 本测试守护：
//! 1. `&'a T` 生命周期注解语法可解析（函数参数 / let 绑定 / 返回值）
//! 2. lifetime 透传不破坏既有借用检查行为（引用释放、移动语义不变）
//! 3. 无 lifetime 标注的 `&T` 行为与带标注的 `&'a T` 一致

use tenth::lexer::lexer::Lexer;
use tenth::parser::parser::Parser;
use tenth::hir::lower::Lowerer;
use tenth::runtime::interpreter::Interpreter;
use tenth::runtime::value::Value;

fn lower_code(src: &str) -> Result<tenth::hir::hir::HirProgram, String> {
    let mut lexer = Lexer::new(src);
    let tokens = lexer.tokenize().map_err(|e| e.to_string())?;
    let mut parser = Parser::new(tokens);
    let program = parser.parse_program().map_err(|e| e.to_string())?;
    let mut lowerer = Lowerer::new();
    lowerer.lower_program(&program).map_err(|e| e.to_string())
}

fn run_code(src: &str) -> Result<Option<Value>, String> {
    let mut lexer = Lexer::new(src);
    let tokens = lexer.tokenize().map_err(|e| e.to_string())?;
    let mut parser = Parser::new(tokens);
    let program = parser.parse_program().map_err(|e| e.to_string())?;
    let mut lowerer = Lowerer::new();
    let hir = lowerer.lower_program(&program).map_err(|e| e.to_string())?;
    let mut interpreter = Interpreter::new(&hir);
    interpreter.execute_program(&hir).map_err(|e| e.to_string())
}

/// 语法透传 1：函数参数可带 `&'a T` 生命周期注解（函数级 `<'a>` 泛型参数不支持，见文件头决策说明）
#[test]
fn test_fn_param_with_lifetime_annotation() {
    let src = "fn deref(x: &'a i64) -> i64 { *x } { deref(&42) }";
    let result = run_code(src).unwrap();
    match result {
        Some(Value::Int(42, _)) => {}
        v => panic!("expected Some(Int(42)), got {:?}", v),
    }
}

/// 语法透传 2：let 绑定可带 `&'a T` 生命周期注解
#[test]
fn test_let_binding_with_lifetime_annotation() {
    let src = "{ let x = 7; let r: &'a i64 = &x; *r }";
    let result = run_code(src).unwrap();
    match result {
        Some(Value::Int(7, _)) => {}
        v => panic!("expected Some(Int(7)), got {:?}", v),
    }
}

/// 语法透传 3：带生命周期注解的引用与无标注引用行为一致（借用可读）
#[test]
fn test_lifetime_annotated_ref_still_readable() {
    let src = "{ let x = 100; let r: &'a i64 = &x; let r2 = &x; *r + *r2 }";
    let result = run_code(src).unwrap();
    match result {
        Some(Value::Int(200, _)) => {}
        v => panic!("expected Some(Int(200)), got {:?}", v),
    }
}

/// 借用检查不变 1：语句粒度释放——引用在语句结束即释放，后续可用原变量
#[test]
fn test_lifetime_annotated_borrow_released() {
    // 与 ownership_test::test_ref_and_deref 同模式，确认 lifetime 透传不改变释放语义
    let src = "{ let x = 42; let r: &'a i64 = &x; *r; x }";
    let result = run_code(src).unwrap();
    match result {
        Some(Value::Int(42, _)) => {}
        v => panic!("expected Some(Int(42)), got {:?}", v),
    }
}

/// 借用检查不变 2：可变引用 `&'a mut T` 注解可解析，写回生效
#[test]
fn test_lifetime_annotated_mut_ref_modify() {
    let src = "{ let mut x = 10; let m: &'a mut i64 = &mut x; *m = 20; x }";
    let result = run_code(src).unwrap();
    match result {
        Some(Value::Int(20, _)) => {}
        v => panic!("expected Some(Int(20)), got {:?}", v),
    }
}

/// 借用检查不变 3：带 lifetime 的引用结构体字段访问仍可用
#[test]
fn test_lifetime_annotated_struct_field() {
    let src = "struct Point { x: f64, y: f64 }; { let p = Point { x: 1.0, y: 2.0 }; let r: &'a Point = &p; (*r).x }";
    let result = run_code(src).unwrap();
    match result {
        Some(Value::Float(1.0)) => {}
        v => panic!("expected Some(Float(1.0)), got {:?}", v),
    }
}

/// 借用检查不变 4：引用传递后原值仍可读（共享引用不移动）
#[test]
fn test_lifetime_annotated_ref_does_not_move() {
    let src = "{ let x = 5; let r: &'a i64 = &x; *r; x }";
    let result = run_code(src).unwrap();
    match result {
        Some(Value::Int(5, _)) => {}
        v => panic!("expected Some(Int(5)), got {:?}", v),
    }
}

/// 编译通过性：带生命周期注解的函数（参数+返回值）必须能正常 lower（不报错）
#[test]
fn test_lifetime_annotation_lowers_ok() {
    let src = "fn f(x: &'a i64) -> &'a i64 { x }";
    assert!(lower_code(src).is_ok(), "带生命周期注解的函数应能 lower");
}
