use tenth::lexer::lexer::Lexer;
use tenth::parser::parser::Parser;
use tenth::hir::lower::Lowerer;
use tenth::runtime::interpreter::Interpreter;
use tenth::runtime::vm::Vm;
use tenth::compile::bytecode::BytecodeCompiler;
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

/// Run source through the bytecode VM (path A default backend).
fn run_vm(src: &str) -> Result<Value, String> {
    let mut lexer = Lexer::new(src);
    let tokens = lexer.tokenize().map_err(|e| e.to_string())?;
    let mut parser = Parser::new(tokens);
    let program = parser.parse_program().map_err(|e| e.to_string())?;
    let mut lowerer = Lowerer::new();
    let hir = lowerer.lower_program(&program).map_err(|e| e.to_string())?;

    let mut vm = Vm::new();
    for func in &hir.functions {
        let compiler = BytecodeCompiler::new();
        match compiler.compile(func) {
            Ok((chunk, closures)) => {
                vm.add_fn(func.name.clone(), chunk);
                for (name, closure_chunk) in closures {
                    vm.add_fn(name, closure_chunk);
                }
            }
            Err(e) => return Err(format!("compile error: {}", e)),
        }
    }
    if let Some(ref expr) = hir.main_expr {
        let compiler = BytecodeCompiler::new();
        match compiler.compile_main(expr) {
            Ok((chunk, closures)) => {
                vm.add_fn("main".into(), chunk);
                for (name, closure_chunk) in closures {
                    vm.add_fn(name, closure_chunk);
                }
            }
            Err(e) => return Err(format!("compile error: {}", e)),
        }
        vm.call("main").map_err(|e| e.to_string())
    } else if vm.has_fn("main") {
        vm.call("main").map_err(|e| e.to_string())
    } else {
        Ok(Value::Unit)
    }
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
        Some(Value::Int(43, _)) => {}
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
        Some(Value::Int(20, _)) => {}
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
        Some(Value::Int(10, _)) => {}
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
        Some(Value::Int(30, _)) => {}
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
        Some(Value::Int(30, _)) => {}
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
        Some(Value::Int(20, _)) => {}
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
        Some(Value::Int(100, _)) => {}
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
        Some(Value::Int(50, _)) => {}
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
        Some(Value::Int(84, _)) => {}
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
        Some(Value::Int(6, _)) => {}
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
        Some(Value::Int(100, _)) => {}
        v => panic!("expected Int(100), got {:?}", v),
    }
}

#[test]
fn test_match_struct_destructuring_shorthand() {
    let src = r#"
        struct Point { x: f64, y: f64 };
        fn main() -> f64 {
            let p = Point { x: 3.0, y: 4.0 };
            match p {
                Point { x, y } => x + y,
            }
        }
    "#;
    let result = run(src).unwrap();
    match result {
        Some(Value::Float(v)) if (v - 7.0).abs() < 1e-9 => {}
        v => panic!("expected Float(7.0), got {:?}", v),
    }
}

#[test]
fn test_match_struct_destructuring_named_bind() {
    let src = r#"
        struct Point { x: f64, y: f64 };
        fn main() -> f64 {
            let p = Point { x: 3.0, y: 4.0 };
            match p {
                Point { x: a, y: b } => a * a + b * b,
            }
        }
    "#;
    let result = run(src).unwrap();
    match result {
        Some(Value::Float(v)) if (v - 25.0).abs() < 1e-9 => {}
        v => panic!("expected Float(25.0), got {:?}", v),
    }
}

#[test]
fn test_match_struct_with_wildcard_fallback() {
    let src = r#"
        struct Point { x: i32, y: i32 };
        struct Rect { w: i32, h: i32 };
        fn main() -> i32 {
            let r = Rect { w: 10, h: 20 };
            match r {
                Point { x, y } => x + y,
                Rect { w, h } => w * h,
                _ => 0,
            }
        }
    "#;
    let result = run(src).unwrap();
    match result {
        Some(Value::Int(200, _)) => {}
        v => panic!("expected Int(200), got {:?}", v),
    }
}

#[test]
fn test_match_struct_partial_bind() {
    let src = r#"
        struct Point { x: i32, y: i32 };
        fn main() -> i32 {
            let p = Point { x: 7, y: 99 };
            match p {
                Point { x, y: _ } => x,
            }
        }
    "#;
    let result = run(src).unwrap();
    match result {
        Some(Value::Int(7, _)) => {}
        v => panic!("expected Int(7), got {:?}", v),
    }
}

// ── VM path (path A default backend) ──────────────────────────────────────

#[test]
fn test_vm_match_struct_destructuring_shorthand() {
    let src = r#"
        struct Point { x: i32, y: i32 };
        fn main() -> i32 {
            let p = Point { x: 3, y: 4 };
            match p {
                Point { x, y } => x + y,
            }
        }
    "#;
    let result = run_vm(src).unwrap();
    match result {
        Value::Int(7, _) => {}
        v => panic!("expected Int(7), got {:?}", v),
    }
}

#[test]
fn test_vm_match_struct_with_fallback() {
    let src = r#"
        struct Point { x: i32, y: i32 };
        struct Rect { w: i32, h: i32 };
        fn main() -> i32 {
            let r = Rect { w: 10, h: 20 };
            match r {
                Point { x, y } => x + y,
                Rect { w, h } => w * h,
                _ => 0,
            }
        }
    "#;
    let result = run_vm(src).unwrap();
    match result {
        Value::Int(200, _) => {}
        v => panic!("expected Int(200), got {:?}", v),
    }
}
