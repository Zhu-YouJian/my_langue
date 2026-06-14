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
fn test_match_binding() {
    let src = r#"
        fn main() -> i32 {
            let x = 42;
            match x {
                n => n + 1,
            }
        }
    "#;
    let result = run(src).unwrap();
    match result {
        Some(Value::Int(43)) => {}
        v => panic!("expected Int(43), got {:?}", v),
    }
}

#[test]
fn test_match_range_exclusive() {
    let src = r#"
        fn main() -> i32 {
            let x = 5;
            match x {
                1..5 => 10,
                5..10 => 20,
                _ => 30,
            }
        }
    "#;
    let result = run(src).unwrap();
    match result {
        Some(Value::Int(20)) => {}
        v => panic!("expected Int(20), got {:?}", v),
    }
}

#[test]
fn test_match_range_inclusive() {
    let src = r#"
        fn main() -> i32 {
            let x = 5;
            match x {
                1..=5 => 10,
                _ => 20,
            }
        }
    "#;
    let result = run(src).unwrap();
    match result {
        Some(Value::Int(10)) => {}
        v => panic!("expected Int(10), got {:?}", v),
    }
}

#[test]
fn test_match_range_no_match() {
    let src = r#"
        fn main() -> i32 {
            let x = 15;
            match x {
                1..5 => 10,
                5..=10 => 20,
                _ => 30,
            }
        }
    "#;
    let result = run(src).unwrap();
    match result {
        Some(Value::Int(30)) => {}
        v => panic!("expected Int(30), got {:?}", v),
    }
}

#[test]
fn test_match_tuple_destructuring() {
    let src = r#"
        fn main() -> i32 {
            let t = (10, 20);
            match t {
                (a, b) => a + b,
            }
        }
    "#;
    let result = run(src).unwrap();
    match result {
        Some(Value::Int(30)) => {}
        v => panic!("expected Int(30), got {:?}", v),
    }
}

#[test]
fn test_match_tuple_with_literal() {
    let src = r#"
        fn main() -> i32 {
            let t = (1, 2);
            match t {
                (1, b) => b * 10,
                (a, 2) => a * 100,
                _ => 0,
            }
        }
    "#;
    let result = run(src).unwrap();
    match result {
        Some(Value::Int(20)) => {}
        v => panic!("expected Int(20), got {:?}", v),
    }
}

#[test]
fn test_match_guard() {
    let src = r#"
        fn main() -> i32 {
            let x = 15;
            match x {
                n if n > 10 => 100,
                n if n > 5 => 50,
                _ => 0,
            }
        }
    "#;
    let result = run(src).unwrap();
    match result {
        Some(Value::Int(100)) => {}
        v => panic!("expected Int(100), got {:?}", v),
    }
}

#[test]
fn test_match_guard_fallback() {
    let src = r#"
        fn main() -> i32 {
            let x = 7;
            match x {
                n if n > 10 => 100,
                n if n > 5 => 50,
                _ => 0,
            }
        }
    "#;
    let result = run(src).unwrap();
    match result {
        Some(Value::Int(50)) => {}
        v => panic!("expected Int(50), got {:?}", v),
    }
}

#[test]
fn test_match_guard_with_enum() {
    let src = r#"
        enum Option { Some(i32), None };
        fn main() -> i32 {
            let x = Option::Some(42);
            match x {
                Option::Some(n) if n > 10 => n * 2,
                Option::Some(n) => n,
                Option::None => 0,
            }
        }
    "#;
    let result = run(src).unwrap();
    match result {
        Some(Value::Int(84)) => {}
        v => panic!("expected Int(84), got {:?}", v),
    }
}

#[test]
fn test_match_tuple_three_elements() {
    let src = r#"
        fn main() -> i32 {
            let t = (1, 2, 3);
            match t {
                (a, b, c) => a + b + c,
            }
        }
    "#;
    let result = run(src).unwrap();
    match result {
        Some(Value::Int(6)) => {}
        v => panic!("expected Int(6), got {:?}", v),
    }
}

#[test]
fn test_match_binding_with_range_fallback() {
    let src = r#"
        fn main() -> i32 {
            let x = 100;
            match x {
                1..=10 => 1,
                11..=50 => 2,
                n => n,
            }
        }
    "#;
    let result = run(src).unwrap();
    match result {
        Some(Value::Int(100)) => {}
        v => panic!("expected Int(100), got {:?}", v),
    }
}
