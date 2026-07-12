use tenth::lexer::lexer::Lexer;
use tenth::parser::parser::Parser;
use tenth::parser::ast::*;
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

fn parse_program(src: &str) -> Result<Program, String> {
    let mut lexer = Lexer::new(src);
    let tokens = lexer.tokenize().map_err(|e| e.to_string())?;
    let mut parser = Parser::new(tokens);
    parser.parse_program().map_err(|e| e.to_string())
}

#[test]
fn test_generic_identity_i32() {
    let src = "fn identity<T>(x: T) -> T { x }; identity<i32>(42)";
    let result = run(src).unwrap();
    match result {
        Some(Value::Int(42, _)) => {}
        v => panic!("expected Int(42), got {}", v.unwrap()),
    }
}

#[test]
fn test_generic_identity_f64() {
    let src = "fn identity<T>(x: T) -> T { x }; identity<f64>(3.14)";
    let result = run(src).unwrap();
    match result {
        Some(Value::Float(v)) => assert!((v - 3.14).abs() < 1e-10),
        v => panic!("expected Float(3.14), got {}", v.unwrap()),
    }
}

#[test]
fn test_generic_struct_pair() {
    let src = r#"
        struct Pair<T, U> { first: T, second: U }
        let p = Pair<i32, f64> { first: 42, second: 3.14 };
        p.first + p.second
    "#;
    let result = run(src).unwrap();
    match result {
        Some(Value::Float(v)) => assert!((v - 45.14).abs() < 1e-10),
        v => panic!("expected Float(45.14), got {}", v.unwrap()),
    }
}

#[test]
fn test_generic_struct_single_param() {
    let src = r#"
        struct Box<T> { value: T }
        let b = Box<i32> { value: 100 };
        b.value
    "#;
    let result = run(src).unwrap();
    match result {
        Some(Value::Int(100, _)) => {}
        v => panic!("expected Int(100), got {}", v.unwrap()),
    }
}

#[test]
fn test_generic_with_bound() {
    let src = r#"
        struct Point { x: f64, y: f64 }
        trait Add {
            fn add(self, other: Self) -> Self;
        }
        impl Add for Point {
            fn add(self, other: Point) -> Point {
                Point { x: self.x + other.x, y: self.y + other.y }
            }
        }
        fn sum<T: Add>(a: T, b: T) -> T { a.add(b) }
        let p = Point { x: 1.0, y: 2.0 };
        let q = Point { x: 3.0, y: 4.0 };
        sum<Point>(p, q)
    "#;
    let result = run(src).unwrap();
    match result {
        Some(Value::Struct { name, fields }) => {
            assert_eq!(name, "Point");
            let fields = fields.borrow();
            let x = fields.iter().find(|(n,_)| n=="x").map(|(_,v)| match v { Value::Float(f) => *f, _ => panic!() }).unwrap();
            let y = fields.iter().find(|(n,_)| n=="y").map(|(_,v)| match v { Value::Float(f) => *f, _ => panic!() }).unwrap();
            assert!((x - 4.0).abs() < 0.001);
            assert!((y - 6.0).abs() < 0.001);
        }
        v => panic!("expected Struct, got {:?}", v),
    }
}

#[test]
fn test_generic_return_type_vec() {
    // Test that fn f() -> Vec<i64> parses correctly
    let src = r#"
        fn make_vec() -> Vec<i64> {
            let v = Vec::new();
            v
        }
        make_vec()
    "#;
    let result = run(src);
    assert!(result.is_ok(), "parse/lower/run should succeed: {:?}", result);
    match result.unwrap() {
        Some(Value::Vec(_)) => {}
        other => panic!("expected Vec, got {:?}", other),
    }
}

#[test]
fn test_generic_return_type_vec_token() {
    // Test that fn tokenize() -> Vec<Token> parses correctly (Token is a custom type)
    let src = r#"
        fn tokenize() -> Vec<Token> {
            let v = Vec::new();
            v
        }
        tokenize()
    "#;
    let result = run(src);
    assert!(result.is_ok(), "parse/lower/run should succeed: {:?}", result);
}

#[test]
fn test_generic_param_hashmap() {
    // Test that fn f(m: HashMap<str, i64>) parses correctly
    let src = r#"
        fn get_count(m: HashMap<str, i64>) -> i64 {
            0
        }
        let h = HashMap::new();
        get_count(h)
    "#;
    let result = run(src);
    assert!(result.is_ok(), "parse/lower/run should succeed: {:?}", result);
}

#[test]
fn test_nested_generic_type() {
    // Test that HashMap<str, Vec<i64>> parses correctly (nested generics)
    let src = r#"
        fn nested() -> HashMap<str, Vec<i64>> {
            let h = HashMap::new();
            h
        }
        nested()
    "#;
    let result = run(src);
    assert!(result.is_ok(), "parse/lower/run should succeed: {:?}", result);
}

#[test]
fn test_generic_return_type_ast() {
    // Verify the AST structure for fn f() -> Vec<Token>
    let src = "fn tokenize() -> Vec<Token> { Vec::new() }";
    let prog = parse_program(src).unwrap();
    assert_eq!(prog.items.len(), 1);
    match &prog.items[0].kind {
        ItemKind::Function { name, return_type, .. } => {
            assert_eq!(name.name, "tokenize");
            let rt = return_type.as_ref().expect("should have return type");
            match rt {
                TypeAnnotation::Generic { base, args } => {
                    assert_eq!(base.name, "Vec");
                    assert_eq!(args.len(), 1);
                    match &args[0] {
                        TypeAnnotation::Named(id) => assert_eq!(id.name, "Token"),
                        other => panic!("expected Named type for arg, got {:?}", other),
                    }
                }
                other => panic!("expected Generic type annotation, got {:?}", other),
            }
        }
        other => panic!("expected Function item, got {:?}", other),
    }
}

#[test]
fn test_nested_generic_type_ast() {
    // Verify the AST structure for HashMap<str, Vec<i64>>
    let src = "fn f() -> HashMap<str, Vec<i64>> { HashMap::new() }";
    let prog = parse_program(src).unwrap();
    match &prog.items[0].kind {
        ItemKind::Function { return_type, .. } => {
            let rt = return_type.as_ref().expect("should have return type");
            match rt {
                TypeAnnotation::Generic { base, args } => {
                    assert_eq!(base.name, "HashMap");
                    assert_eq!(args.len(), 2);
                    match &args[0] {
                        TypeAnnotation::Named(id) => assert_eq!(id.name, "str"),
                        _ => panic!("expected Named for first arg"),
                    }
                    match &args[1] {
                        TypeAnnotation::Generic { base: inner_base, args: inner_args } => {
                            assert_eq!(inner_base.name, "Vec");
                            assert_eq!(inner_args.len(), 1);
                            match &inner_args[0] {
                                TypeAnnotation::Named(id) => assert_eq!(id.name, "i64"),
                                other => panic!("expected Named for inner arg, got {:?}", other),
                            }
                        }
                        other => panic!("expected Generic for second arg, got {:?}", other),
                    }
                }
                other => panic!("expected Generic type annotation, got {:?}", other),
            }
        }
        other => panic!("expected Function item, got {:?}", other),
    }
}