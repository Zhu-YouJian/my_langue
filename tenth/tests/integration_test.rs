use tenth::lexer::lexer::Lexer;
use tenth::parser::parser::Parser;
use tenth::hir::lower::Lowerer;
use tenth::runtime::interpreter::Interpreter;
use tenth::runtime::value::Value;

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

#[test]
fn test_simple_arithmetic() {
    let result = run_code("1 + 2").unwrap();
    match result {
        Some(Value::Int(3)) => {}
        v => panic!("expected Some(Int(3)), got {}", v.unwrap()),
    }
}

#[test]
fn test_variable_and_use() {
    let src = "let x = 42; x + 10";
    let result = run_code(src).unwrap();
    match result {
        Some(Value::Int(52)) => {}
        v => panic!("expected Some(Int(52)), got {}", v.unwrap()),
    }
}

#[test]
fn test_float_arithmetic() {
    let result = run_code("3.14 * 2.0").unwrap();
    match result {
        Some(Value::Float(n)) => assert!((n - 6.28).abs() < 0.01),
        v => panic!("expected Some(Float(6.28)), got {}", v.unwrap()),
    }
}

#[test]
fn test_boolean_ops() {
    let result = run_code("true && false").unwrap();
    match result {
        Some(Value::Bool(false)) => {}
        v => panic!("expected Some(Bool(false)), got {}", v.unwrap()),
    }
}

#[test]
fn test_comparison() {
    let result = run_code("5 > 3").unwrap();
    match result {
        Some(Value::Bool(true)) => {}
        v => panic!("expected Some(Bool(true)), got {}", v.unwrap()),
    }
}

#[test]
fn test_if_expression() {
    let src = "if true { 1 } else { 2 }";
    let result = run_code(src).unwrap();
    match result {
        Some(Value::Int(1)) => {}
        v => panic!("expected Some(Int(1)), got {}", v.unwrap()),
    }
}

#[test]
fn test_tensor_creation_and_sum() {
    let src = "tensor[[1.0, 2.0], [3.0, 4.0]].sum()";
    let result = run_code(src).unwrap();
    match result {
        Some(Value::Float(n)) => assert!((n - 10.0).abs() < 0.01),
        v => panic!("expected Some(Float(10.0)), got {}", v.unwrap()),
    }
}

#[test]
fn test_tensor_methods_relu() {
    let src = "tensor[[-1.0, 0.0, 1.0]].relu()";
    let result = run_code(src).unwrap();
    match result {
        Some(Value::Tensor(_)) => {}
        v => panic!("expected Some(Tensor), got {}", v.unwrap()),
    }
}

#[test]
fn test_while_loop() {
    let src = "{ let mut x = 0; while x < 3 { x = x + 1 }; x }";
    let result = run_code(src).unwrap();
    match result {
        Some(Value::Int(3)) => {}
        v => panic!("expected Some(Int(3)), got {}", v.unwrap()),
    }
}

#[test]
fn test_string_literal() {
    let result = run_code("\"hello\"").unwrap();
    match result {
        Some(Value::String(s)) => assert_eq!(s, "hello"),
        v => panic!("expected Some(String(\"hello\")), got {}", v.unwrap()),
    }
}

#[test]
fn test_function_definition_and_call() {
    let src = "fn add(a: f64, b: f64) -> f64 { a + b }";
    let result = run_code(src);
    assert!(result.is_ok());
}

#[test]
fn test_closure_simple() {
    let src = "{ let f = |x, y| x + y; f(10, 20) }";
    let result = run_code(src).unwrap();
    match result {
        Some(Value::Int(30)) => {}
        v => panic!("expected Some(Int(30)), got {:?}", v),
    }
}

#[test]
fn test_tensor_rand() {
    let src = "rand(2, 3).sum()";
    let result = run_code(src).unwrap();
    match result {
        Some(Value::Float(v)) => assert!(v >= 0.0, "got {}", v),
        v => panic!("expected Float >= 0, got {:?}", v),
    }
}

#[test]
fn test_tensor_softmax_sum_to_one() {
    let src = "tensor[[1.0, 2.0, 3.0]].softmax().sum()";
    let result = run_code(src).unwrap();
    match result {
        Some(Value::Float(v)) => assert!((v - 1.0).abs() < 1e-6, "got {}", v),
        v => panic!("expected Float(1.0), got {:?}", v),
    }
}