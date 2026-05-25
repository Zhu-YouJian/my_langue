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
fn test_struct_definition_and_field_access() {
    let src = "struct Point { x: f64, y: f64 }; let p = Point { x: 3.0, y: 4.0 }; p.x + p.y";
    let result = run(src).unwrap();
    match result {
        Some(Value::Float(v)) => assert!((v - 7.0).abs() < 1e-10, "got {}", v),
        v => panic!("expected Float(7.0), got {:?}", v),
    }
}

#[test]
fn test_nested_struct() {
    let src = "struct Point { x: f64, y: f64 }; struct Rect { top_left: Point, bottom_right: Point }; let r = Rect { top_left: Point { x: 0.0, y: 10.0 }, bottom_right: Point { x: 10.0, y: 0.0 } }; r.top_left.y + r.bottom_right.x";
    let result = run(src).unwrap();
    match result {
        Some(Value::Float(v)) => assert!((v - 20.0).abs() < 1e-10, "got {}", v),
        v => panic!("expected Float(20.0), got {:?}", v),
    }
}

#[test]
fn test_impl_method_call() {
    let src = r#"
    struct Point { x: f64, y: f64 }
    impl Point {
        fn dist(self) -> f64 {
            (self.x * self.x + self.y * self.y).sqrt()
        }
    }
    let p = Point { x: 3.0, y: 4.0 };
    p.dist()
    "#;
    let result = run(src).unwrap();
    match result {
        Some(Value::Float(v)) => assert!((v - 5.0).abs() < 1e-10, "got {}", v),
        v => panic!("expected Float(5.0), got {:?}", v),
    }
}

#[test]
fn test_impl_method_with_args() {
    let src = r#"
    struct Vec2 { x: f64, y: f64 }
    impl Vec2 {
        fn add(self, other: Vec2) -> Vec2 {
            Vec2 { x: self.x + other.x, y: self.y + other.y }
        }
    }
    let a = Vec2 { x: 1.0, y: 2.0 };
    let b = Vec2 { x: 3.0, y: 4.0 };
    let c = a.add(b);
    c.x + c.y
    "#;
    let result = run(src).unwrap();
    match result {
        Some(Value::Float(v)) => assert!((v - 10.0).abs() < 1e-10, "got {}", v),
        v => panic!("expected Float(10.0), got {:?}", v),
    }
}