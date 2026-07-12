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

// --- Basic lazy iterator: .iter().collect() ---

#[test]
fn test_iter_collect() {
    let src = r#"
        let v = [1, 2, 3]
        v.iter().collect()
    "#;
    let result = run(src).unwrap();
    match result {
        Some(Value::Vec(v)) => assert_eq!(v.borrow().len(), 3),
        v => panic!("expected Vec, got {:?}", v),
    }
}

// --- Lazy map ---

#[test]
fn test_iter_map() {
    let src = r#"
        let v = [1, 2, 3]
        v.iter().map(|x| x * 2).collect()
    "#;
    let result = run(src).unwrap();
    match result {
        Some(Value::Vec(v)) => {
            let items = v.borrow();
            assert_eq!(items.len(), 3);
            // Items are wrapped in Shared
            for item in items.iter() {
                let val = match item {
                    Value::Shared(rc) => rc.borrow().clone(),
                    other => other.clone(),
                };
                match val {
                    Value::Int(n, _) => assert!(n == 2 || n == 4 || n == 6, "got {}", n),
                    v => panic!("expected Int, got {:?}", v),
                }
            }
        }
        v => panic!("expected Vec, got {:?}", v),
    }
}

// --- Lazy filter ---

#[test]
fn test_iter_filter() {
    let src = r#"
        let v = [1, 2, 3, 4, 5]
        v.iter().filter(|x| x > 3).collect()
    "#;
    let result = run(src).unwrap();
    match result {
        Some(Value::Vec(v)) => {
            assert_eq!(v.borrow().len(), 2);
        }
        v => panic!("expected Vec, got {:?}", v),
    }
}

// --- Chained map + filter ---

#[test]
fn test_iter_map_filter() {
    let src = r#"
        let v = [1, 2, 3, 4, 5]
        v.iter().map(|x| x * 10).filter(|x| x > 20).collect()
    "#;
    let result = run(src).unwrap();
    match result {
        Some(Value::Vec(v)) => {
            assert_eq!(v.borrow().len(), 3); // 30, 40, 50
        }
        v => panic!("expected Vec, got {:?}", v),
    }
}

// --- Range iterator ---

#[test]
fn test_range_iter() {
    let src = r#"
        (0..5).iter().map(|x| x + 1).collect()
    "#;
    let result = run(src).unwrap();
    match result {
        Some(Value::Vec(v)) => {
            assert_eq!(v.borrow().len(), 5);
        }
        v => panic!("expected Vec, got {:?}", v),
    }
}

// --- Take ---

#[test]
fn test_iter_take() {
    let src = r#"
        let v = [1, 2, 3, 4, 5]
        v.iter().take(3).collect()
    "#;
    let result = run(src).unwrap();
    match result {
        Some(Value::Vec(v)) => {
            assert_eq!(v.borrow().len(), 3);
        }
        v => panic!("expected Vec, got {:?}", v),
    }
}

// --- Skip ---

#[test]
fn test_iter_skip() {
    let src = r#"
        let v = [1, 2, 3, 4, 5]
        v.iter().skip(2).collect()
    "#;
    let result = run(src).unwrap();
    match result {
        Some(Value::Vec(v)) => {
            assert_eq!(v.borrow().len(), 3);
        }
        v => panic!("expected Vec, got {:?}", v),
    }
}

// --- For loop over iterator ---

#[test]
fn test_for_iterator() {
    let src = r#"
        fn main() -> i32 {
            let sum = 0
            for x in [10, 20, 30].iter().map(|x| x / 10) {
                sum = sum + x
            }
            sum
        }
        main()
    "#;
    let result = run(src).unwrap();
    match result {
        Some(Value::Int(6, _)) => {},
        v => panic!("expected Int(6), got {:?}", v),
    }
}

// --- Vec.map() shorthand ---

#[test]
fn test_vec_map_shorthand() {
    let src = r#"
        let v = [1, 2, 3]
        v.map(|x| x * x).collect()
    "#;
    let result = run(src).unwrap();
    match result {
        Some(Value::Vec(v)) => {
            assert_eq!(v.borrow().len(), 3);
        }
        v => panic!("expected Vec, got {:?}", v),
    }
}

// --- Empty iterator ---

#[test]
fn test_empty_iter() {
    let src = r#"
        let v: Vec<i32> = []
        v.iter().map(|x| x + 1).collect()
    "#;
    let result = run(src).unwrap();
    match result {
        Some(Value::Vec(v)) => {
            assert_eq!(v.borrow().len(), 0);
        }
        v => panic!("expected empty Vec, got {:?}", v),
    }
}
