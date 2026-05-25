use tenth::lexer::lexer::Lexer;
use tenth::parser::parser::Parser;
use tenth::hir::lower::Lowerer;
use tenth::compile;

fn compile_to_c(src: &str) -> String {
    let mut lexer = Lexer::new(src);
    let tokens = lexer.tokenize().unwrap();
    let mut parser = Parser::new(tokens);
    let program = parser.parse_program().unwrap();
    let mut lowerer = Lowerer::new();
    let hir = lowerer.lower_program(&program).unwrap();
    compile::compile_to_c(&hir).unwrap()
}

#[test]
fn test_compile_simple_arithmetic() {
    let c = compile_to_c("1 + 2");
    assert!(c.contains("int main"), "should contain main");
}

#[test]
fn test_compile_variable() {
    let c = compile_to_c("{ let x = 42; x + 10 }");
    assert!(!c.is_empty(), "should produce C output");
}

#[test]
fn test_compile_if_else() {
    let c = compile_to_c("if true { 1 } else { 2 }");
    assert!(!c.is_empty(), "should produce C output");
}

#[test]
fn test_compile_function() {
    let c = compile_to_c("fn add(a: i32, b: i32) -> i32 { a + b }");
    assert!(c.contains("add"), "should contain function name");
}

#[test]
fn test_compile_struct() {
    let c = compile_to_c("struct Point { x: f64, y: f64 }");
    assert!(!c.is_empty(), "should produce C output");
}
