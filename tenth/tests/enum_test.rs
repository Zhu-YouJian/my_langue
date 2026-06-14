use tenth::lexer::lexer::Lexer;
use tenth::parser::parser::Parser;
use tenth::hir::lower::Lowerer;
use tenth::runtime::interpreter::Interpreter;
use tenth::runtime::value::Value;

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

#[test]
fn test_enum_simple() {
    let src = "enum Color { Red, Green, Blue }; Color::Red";
    let result = run(src).unwrap();
    assert!(result.is_some(), "result should be Some");
}

#[test]
fn test_enum_with_fields() {
    let src = "enum Option { Some(value: i32), None }; Option::Some(value: 42)";
    let result = run(src).unwrap();
    assert!(result.is_some(), "result should be Some");
}

#[test]
fn test_match_enum_some() {
    let src = "enum Option { Some(value: i32), None }; let x = Option::Some(value: 42); match x { Option::Some(value: v) => v * 2, Option::None => 0, }";
    let result = run(src).unwrap();
    match result {
        Some(Value::Int(84)) => {}
        v => panic!("expected Int(84), got {:?}", v),
    }
}

#[test]
fn test_match_enum_none() {
    let src = "enum Option { Some(value: i32), None }; let x = Option::None; match x { Option::Some(value: v) => v, Option::None => -1, }";
    let result = run(src).unwrap();
    match result {
        Some(Value::Int(-1)) => {}
        v => panic!("expected Int(-1), got {:?}", v),
    }
}

#[test]
fn test_match_wildcard() {
    let src = "enum Option { Some(value: i32), None }; let x = Option::None; match x { Option::Some(value: v) => v, _ => -1, }";
    let result = run(src).unwrap();
    match result {
        Some(Value::Int(-1)) => {}
        v => panic!("expected Int(-1), got {:?}", v),
    }
}

#[test]
fn test_enum_tuple_variant() {
    let src = "enum TokenKind { IntLiteral(i64), Identifier(str), Plus, Eof }; let tok = TokenKind::IntLiteral(42); tok";
    let result = run(src).unwrap();
    match result {
        Some(Value::Enum { enum_name, variant, .. }) => {
            assert_eq!(enum_name, "TokenKind");
            assert_eq!(variant, "IntLiteral");
        }
        v => panic!("expected Enum, got {:?}", v),
    }
}

#[test]
fn test_enum_tuple_variant_match() {
    let src = "enum TokenKind { IntLiteral(i64), Identifier(str), Plus, Eof }; let tok = TokenKind::IntLiteral(42); match tok { TokenKind::IntLiteral(n) => n, TokenKind::Identifier(s) => 0, TokenKind::Plus => -1, TokenKind::Eof => -2, }";
    let result = run(src).unwrap();
    match result {
        Some(Value::Int(42)) => {}
        v => panic!("expected Int(42), got {:?}", v),
    }
}

#[test]
fn test_enum_tuple_variant_multi_field() {
    let src = "enum Expr { Add(i64, i64), Val(i64), None }; let e = Expr::Add(3, 5); match e { Expr::Add(a, b) => a + b, Expr::Val(v) => v, Expr::None => 0, }";
    let result = run(src).unwrap();
    match result {
        Some(Value::Int(8)) => {}
        v => panic!("expected Int(8), got {:?}", v),
    }
}

#[test]
fn test_enum_tuple_variant_string_field() {
    let src = "enum Result { Ok(i64), Err(str) }; let r = Result::Err(\"not found\"); match r { Result::Ok(v) => v, Result::Err(msg) => -1, }";
    let result = run(src).unwrap();
    match result {
        Some(Value::Int(-1)) => {}
        v => panic!("expected Int(-1), got {:?}", v),
    }
}