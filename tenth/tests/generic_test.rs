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
fn test_generic_identity_i32() {
    let src = "fn identity<T>(x: T) -> T { x }; identity<i32>(42)";
    let result = run(src).unwrap();
    match result {
        Some(Value::Int(42)) => {}
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
        Some(Value::Int(100)) => {}
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