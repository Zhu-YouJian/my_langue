use tenth::lexer::lexer::Lexer;
use tenth::parser::ast::*;
use tenth::parser::parser::Parser;

fn parse_expr(src: &str) -> Result<Expr, String> {
    let mut lexer = Lexer::new(src);
    let tokens = lexer.tokenize().map_err(|e| e.to_string())?;
    let mut parser = Parser::new(tokens);
    parser.parse_expr().map_err(|e| e.to_string())
}

fn parse_program(src: &str) -> Result<Program, String> {
    let mut lexer = Lexer::new(src);
    let tokens = lexer.tokenize().map_err(|e| e.to_string())?;
    let mut parser = Parser::new(tokens);
    parser.parse_program().map_err(|e| e.to_string())
}

#[test]
fn test_simple_int() {
    let expr = parse_expr("42").unwrap();
    match expr.kind {
        ExprKind::Literal(Literal::Int(42, _)) => {},
        _ => panic!("expected int literal"),
    }
}

#[test]
fn test_binary_expr() {
    let expr = parse_expr("1 + 2").unwrap();
    match expr.kind {
        ExprKind::Binary { op: BinOp::Add, .. } => {},
        _ => panic!("expected Add binary"),
    }
}

#[test]
fn test_function_def() {
    let prog = parse_program("fn add(a: i32, b: i32) -> i32 { a + b }").unwrap();
    assert_eq!(prog.items.len(), 1);
    match &prog.items[0].kind {
        ItemKind::Function { name, .. } => assert_eq!(name.name, "add"),
        _ => panic!("expected function"),
    }
}

#[test]
fn test_tensor_literal() {
    let expr = parse_expr("tensor[[1.0, 2.0], [3.0, 4.0]]").unwrap();
    match expr.kind {
        ExprKind::Call { .. } => {},
        _ => panic!("expected tensor call"),
    }
}

#[test]
fn test_if_expr() {
    let expr = parse_expr("if true { 1 } else { 2 }").unwrap();
    match expr.kind {
        ExprKind::If { .. } => {},
        _ => panic!("expected if expression"),
    }
}