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
    compile::compile_to_c(&hir, true).unwrap()
}

fn compile_to_c_noopt(src: &str) -> String {
    let mut lexer = Lexer::new(src);
    let tokens = lexer.tokenize().unwrap();
    let mut parser = Parser::new(tokens);
    let program = parser.parse_program().unwrap();
    let mut lowerer = Lowerer::new();
    let hir = lowerer.lower_program(&program).unwrap();
    compile::compile_to_c(&hir, false).unwrap()
}

#[test]
fn test_const_fold_optimization() {
    let opt = compile_to_c("2 + 3 * 4");
    let _noopt = compile_to_c_noopt("2 + 3 * 4");
    // Optimized version should contain 14 directly, not the full expression
    assert!(opt.contains("14"), "optimized output should contain constant-folded 14: {}", opt);
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
