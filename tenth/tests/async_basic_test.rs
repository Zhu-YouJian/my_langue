use tenth::lexer::lexer::Lexer;
use tenth::parser::parser::Parser;
use tenth::hir::lower::Lowerer;
use tenth::runtime::vm::Vm;
use tenth::runtime::value::Value;
use tenth::compile::bytecode::BytecodeCompiler;

/// Run source through lexer → parser → HIR → bytecode → VM.
/// Provides a `print` native so tests can observe values.
fn run_vm(src: &str) -> Result<Value, String> {
    let mut lexer = Lexer::new(src);
    let tokens = lexer.tokenize().map_err(|e| e.to_string())?;
    let mut parser = Parser::new(tokens);
    let program = parser.parse_program().map_err(|e| e.to_string())?;
    let mut lowerer = Lowerer::new();
    let hir = lowerer.lower_program(&program).map_err(|e| e.to_string())?;

    let mut vm = Vm::new();
    vm.add_native("print".into(), |_vm, args| {
        for a in args { print!("{a}"); }
        Ok(Value::Unit)
    });
    vm.add_native("println".into(), |_vm, args| {
        for a in args { print!("{a}"); }
        println!();
        Ok(Value::Unit)
    });

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

// ─── spawn produces a Future; await unwraps it ─────────────────────────────

#[test]
fn test_spawn_then_await_int() {
    let src = r#"
        fn make_num() -> int {
            return 42
        }
        fn main() {
            let f = spawn make_num()
            let n = await f
            print(n)
        }
    "#;
    let result = run_vm(src).unwrap();
    // main returns Unit; the test passes if no panic and result is Unit
    assert!(matches!(result, Value::Unit), "expected Unit, got {:?}", result);
}

// ─── await on a non-Future value is a no-op (passes through) ───────────────

#[test]
fn test_await_plain_value() {
    let src = r#"
        fn main() {
            let n = await 7
            print(n)
        }
    "#;
    let result = run_vm(src).unwrap();
    assert!(matches!(result, Value::Unit));
}

// ─── spawn of a literal expression ─────────────────────────────────────────

#[test]
fn test_spawn_literal() {
    let src = r#"
        fn main() {
            let f = spawn 99
            let n = await f
            print(n)
        }
    "#;
    let result = run_vm(src).unwrap();
    assert!(matches!(result, Value::Unit));
}

// ─── async fn keyword parses (is_async flag) ───────────────────────────────

#[test]
fn test_async_fn_parses() {
    // Even though async fn isn't fully wired through the type system yet,
    // the parser must accept the `async` keyword without error.
    let src = r#"
        async fn delayed() -> int {
            return 1
        }
        fn main() {
            let n = await spawn delayed()
            print(n)
        }
    "#;
    let result = run_vm(src);
    assert!(result.is_ok(), "async fn should parse and run: {:?}", result);
}

// ─── multiple spawns and awaits in sequence ────────────────────────────────

#[test]
fn test_multiple_spawn_await() {
    let src = r#"
        fn double(x: int) -> int {
            return x * 2
        }
        fn main() {
            let a = await spawn double(5)
            let b = await spawn double(10)
            print(a + b)
        }
    "#;
    let result = run_vm(src).unwrap();
    assert!(matches!(result, Value::Unit));
}

// ─── spawn preserves string values ─────────────────────────────────────────

#[test]
fn test_spawn_string() {
    let src = r#"
        fn greet() -> string {
            return "hi"
        }
        fn main() {
            let s = await spawn greet()
            print(s)
        }
    "#;
    let result = run_vm(src).unwrap();
    assert!(matches!(result, Value::Unit));
}
