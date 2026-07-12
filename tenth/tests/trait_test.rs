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
fn test_trait_add_for_point() {
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
        let p = Point { x: 1.0, y: 2.0 };
        let q = Point { x: 3.0, y: 4.0 };
        p.add(q)
    "#;
    let result = run_code(src).unwrap();
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
fn test_inherent_impl_still_works() {
    let src = r#"
        struct Point { x: f64, y: f64 }
        impl Point {
            fn get_x(self) -> f64 { self.x }
        }
        let p = Point { x: 5.0, y: 3.0 };
        p.get_x()
    "#;
    let result = run_code(src).unwrap();
    match result {
        Some(Value::Float(v)) => assert!((v - 5.0).abs() < 0.001),
        v => panic!("expected Float(5.0), got {:?}", v),
    }
}

#[test]
fn test_trait_without_self() {
    let src = r#"
        struct Point { x: f64, y: f64 }
        trait Describe {
            fn describe(self) -> string;
        }
        impl Describe for Point {
            fn describe(self) -> string { "a point" }
        }
        let p = Point { x: 1.0, y: 2.0 };
        p.describe()
    "#;
    let result = run_code(src).unwrap();
    match result {
        Some(Value::String(s)) => assert_eq!(s, "a point"),
        v => panic!("expected String, got {:?}", v),
    }
}

#[test]
fn test_builtin_trait_in_bound() {
    let src = r#"
        fn show<T: Display>(x: T) -> T { x }
        show<i32>(42)
    "#;
    let result = run_code(src).unwrap();
    match result {
        Some(Value::Int(42, _)) => {},
        v => panic!("expected Int(42), got {:?}", v),
    }
}

// --- Default method implementation ---

#[test]
fn test_trait_default_method() {
    let src = r#"
        struct Point { x: f64, y: f64 }
        trait Describe {
            fn describe(self) -> string;
            fn debug_str(self) -> string { "debug: " + self.describe() }
        }
        impl Describe for Point {
            fn describe(self) -> string { "point" }
        }
        let p = Point { x: 1.0, y: 2.0 };
        p.describe()
    "#;
    let result = run_code(src).unwrap();
    match result {
        Some(Value::String(s)) => assert_eq!(s, "point"),
        v => panic!("expected String, got {:?}", v),
    }
}

// --- Associated types ---

#[test]
fn test_trait_associated_type() {
    let src = r#"
        trait Container {
            type Item;
            fn get(self) -> Item;
        }
        struct Box { value: i32 }
        impl Container for Box {
            fn get(self) -> i32 { self.value }
        }
        let b = Box { value: 42 };
        b.get()
    "#;
    let result = run_code(src).unwrap();
    match result {
        Some(Value::Int(42, _)) => {},
        v => panic!("expected Int(42), got {:?}", v),
    }
}

// --- Trait bound check: missing required method ---

#[test]
fn test_trait_bound_check_missing_method() {
    let src = r#"
        trait Required {
            fn must_impl(self) -> i32;
        }
        struct S { x: i32 }
        impl Required for S { }
    "#;
    let result = run_code(src);
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(err.contains("缺少方法") || err.contains("missing implementation"), "expected missing impl error, got: {}", err);
}

// --- Trait with default method only (no required methods) ---

#[test]
fn test_trait_all_default_methods() {
    let src = r#"
        struct S { x: i32 }
        trait Greet {
            fn hello() -> string { "hello" }
        }
        impl Greet for S { }
        42
    "#;
    let result = run_code(src).unwrap();
    match result {
        Some(Value::Int(42, _)) => {},
        v => panic!("expected Int(42), got {:?}", v),
    }
}

// --- Multiple trait bounds ---

#[test]
fn test_multiple_trait_bounds() {
    let src = r#"
        fn both<T: Display + Clone>(x: T) -> T { x }
        both<i32>(7)
    "#;
    let result = run_code(src).unwrap();
    match result {
        Some(Value::Int(7, _)) => {},
        v => panic!("expected Int(7), got {:?}", v),
    }
}