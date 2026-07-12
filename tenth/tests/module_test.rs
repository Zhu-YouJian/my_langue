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
        Some(Value::Int(3, _)) => {}
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
        Some(Value::Int(42, _)) => {}
        v => panic!("expected Int(42), got {:?}", v),
    }
}

// --- Glob import: use path::* ---

#[test]
fn test_use_glob_import() {
    let src = r#"
    mod math {
        fn add(a: i32, b: i32) -> i32 { a + b }
        fn mul(a: i32, b: i32) -> i32 { a * b }
    }
    use math::*;
    add(3, 4) + mul(2, 5)
    "#;
    let result = run(src).unwrap();
    match result {
        Some(Value::Int(17, _)) => {}
        v => panic!("expected Int(17), got {:?}", v),
    }
}

// --- Nested modules ---

#[test]
fn test_nested_module() {
    let src = r#"
    mod outer {
        mod inner {
            fn value() -> i32 { 42 }
        }
        use inner::value;
    }
    use outer::value;
    value()
    "#;
    let result = run(src).unwrap();
    match result {
        Some(Value::Int(42, _)) => {}
        v => panic!("expected Int(42), got {:?}", v),
    }
}

// --- pub function visibility ---

#[test]
fn test_pub_function() {
    let src = r#"
    mod utils {
        pub fn helper() -> i32 { 99 }
    }
    use utils::helper;
    helper()
    "#;
    let result = run(src).unwrap();
    match result {
        Some(Value::Int(99, _)) => {}
        v => panic!("expected Int(99), got {:?}", v),
    }
}

// --- Module with multiple functions ---

#[test]
fn test_module_multiple_functions() {
    let src = r#"
    mod math {
        fn square(x: i32) -> i32 { x * x }
        fn cube(x: i32) -> i32 { x * x * x }
    }
    use math::*;
    square(3) + cube(2)
    "#;
    let result = run(src).unwrap();
    match result {
        Some(Value::Int(17, _)) => {}
        v => panic!("expected Int(17), got {:?}", v),
    }
}