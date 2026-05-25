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
fn test_module_function_direct() {
    let src = r#"
    mod math {
        fn add(a: i32, b: i32) -> i32 { a + b }
    }
    math::add(1, 2)
    "#;
    let result = run(src).unwrap();
    match result {
        Some(Value::Int(3)) => {}
        v => panic!("expected Int(3), got {:?}", v),
    }
}

#[test]
fn test_use_import() {
    let src = r#"
    mod math {
        fn double(x: i32) -> i32 { x * 2 }
    }
    use math::double;
    double(21)
    "#;
    let result = run(src).unwrap();
    match result {
        Some(Value::Int(42)) => {}
        v => panic!("expected Int(42), got {:?}", v),
    }
}