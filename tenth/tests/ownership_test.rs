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

#[test]
fn test_ref_and_deref() {
    let src = "{ let x = 42; let r = &x; *r }";
    let result = run_code(src).unwrap();
    match result {
        Some(Value::Int(42)) => {}
        v => panic!("expected Some(Int(42)), got {:?}", v),
    }
}

#[test]
fn test_mut_ref_modify() {
    let src = "{ let mut x = 10; let m = &mut x; *m = 20; x }";
    let result = run_code(src).unwrap();
    match result {
        Some(Value::Int(20)) => {}
        v => panic!("expected Some(Int(20)), got {:?}", v),
    }
}

#[test]
fn test_shared_ref_still_readable() {
    let src = "{ let x = 100; let r1 = &x; let r2 = &x; *r1 + *r2 }";
    let result = run_code(src).unwrap();
    match result {
        Some(Value::Int(200)) => {}
        v => panic!("expected Some(Int(200)), got {:?}", v),
    }
}

#[test]
fn test_ref_struct_field() {
    let src = "struct Point { x: f64, y: f64 }; { let p = Point { x: 1.0, y: 2.0 }; let r = &p; (*r).x }";
    let result = run_code(src).unwrap();
    match result {
        Some(Value::Float(n)) => assert!((n - 1.0).abs() < 0.01),
        v => panic!("expected Some(Float(1.0)), got {:?}", v),
    }
}

#[test]
fn test_deref_non_ref_should_fail() {
    let src = "*42";
    let result = run_code(src);
    assert!(result.is_err(), "expected error when dereferencing non-reference");
}

#[test]
fn test_move_semantics() {
    let src = "{ let x = 42; let y = move x; y }";
    let result = run_code(src).unwrap();
    match result {
        Some(Value::Int(42)) => {}
        v => panic!("expected Some(Int(42)), got {:?}", v),
    }
}

#[test]
fn test_use_after_move_should_fail() {
    let src = "{ let x = 42; let y = move x; x }";
    let result = run_code(src);
    assert!(result.is_err(), "expected error when using moved value");
}

#[test]
fn test_move_struct() {
    let src = "struct Point { x: f64, y: f64 }; { let p = Point { x: 1.0, y: 2.0 }; let q = move p; q.x }";
    let result = run_code(src).unwrap();
    match result {
        Some(Value::Float(n)) => assert!((n - 1.0).abs() < 0.01),
        v => panic!("expected Some(Float(1.0)), got {:?}", v),
    }
}

// --- Borrow checker compilation tests ---

#[test]
fn test_borrow_check_use_after_move() {
    let src = "{ let x = 42; let y = move x; x }";
    let result = lower_code(src);
    assert!(result.is_err(), "expected compile error: use of moved value");
}

#[test]
fn test_borrow_check_mut_while_shared() {
    // Strict borrow checking: cannot take &mut while & is active
    let src = "{ let x = 42; let r = &x; let m = &mut x; *r }";
    let result = lower_code(src);
    assert!(result.is_err(), "expected compile error: cannot borrow as mutable while shared");
}

#[test]
fn test_borrow_check_shared_while_mut() {
    // Strict borrow checking: cannot take & while &mut is active
    let src = "{ let x = 42; let m = &mut x; let r = &x; *m }";
    let result = lower_code(src);
    assert!(result.is_err(), "expected compile error: cannot borrow as shared while mutable borrow is active");
}
