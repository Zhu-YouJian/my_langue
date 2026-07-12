use tenth::lexer::lexer::Lexer;
use tenth::parser::parser::Parser;
use tenth::hir::lower::Lowerer;
use tenth::hir::types::{Type, BaseType};
use tenth::runtime::value::Value;

fn lower_code(src: &str) -> tenth::hir::hir::HirProgram {
    let mut lexer = Lexer::new(src);
    let tokens = lexer.tokenize().unwrap();
    let mut parser = Parser::new(tokens);
    let program = parser.parse_program().unwrap();
    let mut lowerer = Lowerer::new();
    lowerer.lower_program(&program).unwrap()
}

fn main_expr_type(hir: &tenth::hir::hir::HirProgram) -> Option<Type> {
    hir.main_expr.as_ref().map(|e| e.ty.clone())
}

#[test]
fn test_int_default_is_i32() {
    let hir = lower_code("42");
    let ty = main_expr_type(&hir).unwrap();
    assert_eq!(ty, Type::Base(BaseType::I32));
}

#[test]
fn test_int_u8_suffix() {
    let hir = lower_code("42u8");
    let ty = main_expr_type(&hir).unwrap();
    assert_eq!(ty, Type::Base(BaseType::U8));
}

#[test]
fn test_int_i64_suffix() {
    let hir = lower_code("42i64");
    let ty = main_expr_type(&hir).unwrap();
    assert_eq!(ty, Type::Base(BaseType::I64));
}

#[test]
fn test_int_u32_suffix() {
    let hir = lower_code("42u32");
    let ty = main_expr_type(&hir).unwrap();
    assert_eq!(ty, Type::Base(BaseType::U32));
}

#[test]
fn test_int_i16_suffix() {
    let hir = lower_code("42i16");
    let ty = main_expr_type(&hir).unwrap();
    assert_eq!(ty, Type::Base(BaseType::I16));
}

#[test]
fn test_u8_max_ok() {
    let hir = lower_code("255u8");
    let ty = main_expr_type(&hir).unwrap();
    assert_eq!(ty, Type::Base(BaseType::U8));
}

#[test]
fn test_u8_overflow_fails() {
    let mut lexer = Lexer::new("256u8");
    let result = lexer.tokenize();
    assert!(result.is_err(), "256u8 应超出 u8 范围报错");
}

#[test]
fn test_i8_overflow_fails() {
    let mut lexer = Lexer::new("128i8");
    let result = lexer.tokenize();
    assert!(result.is_err(), "128i8 应超出 i8 范围报错");
}

#[test]
fn test_i8_max_ok() {
    let hir = lower_code("127i8");
    let ty = main_expr_type(&hir).unwrap();
    assert_eq!(ty, Type::Base(BaseType::I8));
}

#[test]
fn test_u16_max_ok() {
    let hir = lower_code("65535u16");
    let ty = main_expr_type(&hir).unwrap();
    assert_eq!(ty, Type::Base(BaseType::U16));
}

#[test]
fn test_u16_overflow_fails() {
    let mut lexer = Lexer::new("65536u16");
    let result = lexer.tokenize();
    assert!(result.is_err(), "65536u16 应超出 u16 范围报错");
}

#[test]
fn test_value_int_dtype_preserved() {
    let v = Value::Int(42, BaseType::U8);
    assert_eq!(v.type_of(), Type::Base(BaseType::U8));
}

#[test]
fn test_value_int_i64_dtype() {
    let v = Value::Int(1000000, BaseType::I64);
    assert_eq!(v.type_of(), Type::Base(BaseType::I64));
}

#[test]
fn test_value_int_default_i32() {
    let v = Value::Int(42, BaseType::I32);
    assert_eq!(v.type_of(), Type::Base(BaseType::I32));
}
