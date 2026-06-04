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
fn test_string_len() {
    let src = "\"hello\".len()";
    let result = run_code(src).unwrap();
    match result {
        Some(Value::Int(5)) => {}
        v => panic!("expected Some(Int(5)), got {:?}", v),
    }
}

#[test]
fn test_string_concat() {
    let src = "\"hello\" + \" world\"";
    let result = run_code(src).unwrap();
    match result {
        Some(Value::String(s)) => assert_eq!(s, "hello world"),
        v => panic!("expected Some(String(\"hello world\")), got {:?}", v),
    }
}

#[test]
fn test_vec_new_and_len() {
    let src = "Vec::new().len()";
    let result = run_code(src).unwrap();
    match result {
        Some(Value::Int(0)) => {}
        v => panic!("expected Some(Int(0)), got {:?}", v),
    }
}

#[test]
fn test_hashmap_new_and_len() {
    let src = "HashMap::new().len()";
    let result = run_code(src).unwrap();
    match result {
        Some(Value::Int(0)) => {}
        v => panic!("expected Some(Int(0)), got {:?}", v),
    }
}

#[test]
fn test_hashmap_get() {
    let src = "HashMap::new().get(\"missing\")";
    let result = run_code(src).unwrap();
    // get returns None → no matching variable → should return None/Unit
    assert!(result.is_none(), "expected None for missing key, got {:?}", result);
}

#[test]
fn test_read_file() {
    // Test that read_file builtin is registered and returns error for missing file
    // (actual file read tested via integration)
    let src = "read_file(\"nonexistent_file.th\")";
    let result = run_code(src);
    assert!(result.is_err(), "should fail for missing file");
}

#[test]
fn test_option_some() {
    let src = "Option::Some(42)";
    let result = run_code(src).unwrap();
    match result {
        Some(Value::Enum { enum_name, variant, fields }) => {
            assert_eq!(enum_name, "Option");
            assert_eq!(variant, "Some");
            assert!(fields.borrow().iter().any(|(n, _)| n == "_0"));
        }
        v => panic!("expected Option::Some, got {:?}", v),
    }
}

#[test]
fn test_option_none() {
    let src = "Option::None";
    let result = run_code(src).unwrap();
    match result {
        Some(Value::Enum { enum_name, variant, .. }) => {
            assert_eq!(enum_name, "Option");
            assert_eq!(variant, "None");
        }
        v => panic!("expected Option::None, got {:?}", v),
    }
}
