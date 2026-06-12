use std::rc::Rc;
use std::cell::RefCell;
use tenth::error::TenthResult;
use tenth::repl;
use tenth::runtime::limits::MemoryConfig;
use tenth::lexer::lexer::Lexer;
use tenth::parser::parser::Parser;
use tenth::hir::lower::Lowerer;
use tenth::runtime::interpreter::Interpreter;
use tenth::runtime::vm::Vm;
use tenth::runtime::value::Value;
use tenth::compile;
use tenth::compile::bytecode::BytecodeCompiler;

fn main() -> TenthResult<()> {
    let args: Vec<String> = std::env::args().collect();

    if args.len() >= 2 {
        match args[1].as_str() {
            "build" if args.len() >= 3 => {
                return build_wasm(&args[2]);
            }
            "run" if args.len() >= 3 => {
                return run_file(&args[2]);
            }
            "wasm" if args.len() >= 3 => {
                return run_wasm(&args[2]);
            }
            _ => {}
        }
    }

    // Default: REPL mode
    let max_memory_mb: usize = args.iter()
        .position(|a| a == "--max-memory")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);

    if max_memory_mb > 0 {
        let config = MemoryConfig {
            max_arena_bytes: max_memory_mb * 1024 * 1024,
            max_variables: 10_000,
            max_accumulated_defs: 5_000,
            max_tensor_elements: max_memory_mb * 1024 * 128,
            track_allocations: true,
        };
        println!("[limits] Max memory: {} MiB", max_memory_mb);
        repl::run_repl_with_limits(config)?;
    } else {
        repl::run_repl()?;
    }

    Ok(())
}

/// lex → parse → lower → HIR (shared pipeline)
fn source_to_hir(source: &str) -> TenthResult<tenth::hir::hir::HirProgram> {
    let mut lexer = Lexer::new(source);
    let tokens = lexer.tokenize()?;
    let mut parser = Parser::new(tokens);
    let program = parser.parse_program()?;
    let mut lowerer = Lowerer::new();
    lowerer.lower_program(&program)
}

/// Run a .th source file — try VM first, fall back to tree-walk interpreter.
fn run_file(path: &str) -> TenthResult<()> {
    let source = std::fs::read_to_string(path)
        .map_err(|e| tenth::error::TenthError::RuntimeError {
            message: format!("cannot read {}: {}", path, e),
        })?;
    let hir = source_to_hir(&source)?;

    match vm_execute(&hir) {
        Ok(val) => {
            if !matches!(val, Value::Unit) { println!("= {}", val); }
            return Ok(());
        }
        Err(_) => {}
    }
    let mut interpreter = Interpreter::new(&hir);
    match interpreter.execute_program(&hir)? {
        Some(val) => println!("= {}", val),
        None => {}
    }
    Ok(())
}

/// Execute a HirProgram via the VM. Returns the result or an error.
fn vm_execute(hir: &tenth::hir::hir::HirProgram) -> TenthResult<Value> {
    let mut vm = Vm::new();
    register_natives(&mut vm);

    for func in &hir.functions {
        let compiler = BytecodeCompiler::new();
        if let Ok(chunk) = compiler.compile(func) {
            vm.add_fn(func.name.clone(), chunk);
        }
    }

    if vm.has_fn("main") {
        vm.call("main")
    } else if let Some(ref expr) = hir.main_expr {
        let compiler = BytecodeCompiler::new();
        let chunk = compiler.compile_main(expr)
            .map_err(|_| tenth::error::TenthError::RuntimeError { message: "VM compile failed".into() })?;
        vm.add_fn("main".into(), chunk);
        vm.call("main")
    } else if hir.functions.is_empty() {
        Ok(Value::Unit)
    } else {
        // Functions exist but none could be compiled for VM → signal fallback
        Err(tenth::error::TenthError::RuntimeError {
            message: "VM: main not compiled (unsupported constructs, falling back to interpreter)".into(),
        })
    }
}

fn register_natives(vm: &mut Vm) {
    vm.add_native("println".into(), |_vm, args| {
        for a in args { print!("{a}"); }
        println!();
        Ok(Value::Unit)
    });
    vm.add_native("read_file".into(), |_vm, args| {
        if let Some(Value::String(path)) = args.first() {
            match std::fs::read_to_string(path) {
                Ok(s) => Ok(Value::String(s)),
                Err(e) => Err(tenth::error::TenthError::RuntimeError { message: format!("read_file: {e}") }),
            }
        } else {
            Ok(Value::String(String::new()))
        }
    });
    vm.add_native("Vec::new".into(), |_vm, _args| {
        Ok(Value::Vec(Rc::new(RefCell::new(Vec::new()))))
    });
    vm.add_native("write_bytes".into(), |_vm, args| {
        if args.len() >= 2 {
            if let Value::String(path) = &args[1] {
                if let Value::Vec(items) = &args[0] {
                    let bytes: Vec<u8> = items.borrow().iter().map(|v| v.as_int().unwrap_or(0) as u8).collect();
                    let _ = std::fs::write(path, &bytes);
                    return Ok(Value::Int(0));
                }
            }
        }
        Ok(Value::Int(1))
    });
    vm.add_native("compile_host".into(), |_vm, args| {
        if args.len() >= 2 {
            if let (Value::String(src), Value::String(out)) = (&args[0], &args[1]) {
                match tenth::lexer::lexer::Lexer::new(src).tokenize()
                    .and_then(|tokens| tenth::parser::parser::Parser::new(tokens).parse_program())
                    .and_then(|prog| tenth::hir::lower::Lowerer::new().lower_program(&prog))
                    .and_then(|hir| tenth::compile::compile_to_wasm(&hir))
                {
                    Ok(bytes) => { let _ = std::fs::write(out, &bytes); return Ok(Value::Int(0)); }
                    Err(_) => return Ok(Value::Int(1)),
                }
            }
        }
        Ok(Value::Int(1))
    });
    vm.add_native("compile_program".into(), |_vm, args| {
        if args.len() >= 2 {
            if let Value::String(out) = &args[1] {
                match tenth::compile::compile_program_to_wasm(&args[0]) {
                    Ok(bytes) => { let _ = std::fs::write(out, &bytes); return Ok(Value::Int(0)); }
                    Err(_) => return Ok(Value::Int(1)),
                }
            }
        }
        Ok(Value::Int(1))
    });
}

/// Compile a .th file to a .wasm binary.
fn build_wasm(path: &str) -> TenthResult<()> {
    let source = std::fs::read_to_string(path)
        .map_err(|e| tenth::error::TenthError::RuntimeError {
            message: format!("cannot read {}: {}", path, e),
        })?;
    let hir = source_to_hir(&source)?;
    let wasm_bytes = compile::compile_to_wasm(&hir)?;

    let out_path = path.replace(".th", ".wasm");
    std::fs::write(&out_path, &wasm_bytes)
        .map_err(|e| tenth::error::TenthError::RuntimeError {
            message: format!("cannot write {}: {}", out_path, e),
        })?;
    println!("Compiled to {}", out_path);
    Ok(())
}

/// Compile a .th file to WASM and execute it via wasmi.
fn run_wasm(path: &str) -> TenthResult<()> {
    let source = std::fs::read_to_string(path)
        .map_err(|e| tenth::error::TenthError::RuntimeError {
            message: format!("cannot read {}: {}", path, e),
        })?;
    let hir = source_to_hir(&source)?;
    compile::run_wasm(&hir)
}

/// Compile a .th file to bytecode and execute via the stack VM.
#[allow(dead_code)]
fn vm_run(path: &str) -> TenthResult<()> {
    let source = std::fs::read_to_string(path)
        .map_err(|e| tenth::error::TenthError::RuntimeError {
            message: format!("cannot read {}: {}", path, e),
        })?;
    let hir = source_to_hir(&source)?;
    let mut vm = Vm::new();

    // Register native functions
    vm.add_native("println".into(), |_vm, args| {
        for a in args { print!("{a}"); }
        println!();
        Ok(Value::Unit)
    });
    vm.add_native("read_file".into(), |_vm, args| {
        if let Some(Value::String(path)) = args.first() {
            match std::fs::read_to_string(path) {
                Ok(s) => Ok(Value::String(s)),
                Err(e) => Err(tenth::error::TenthError::RuntimeError { message: format!("read_file: {e}") }),
            }
        } else {
            Ok(Value::String(String::new()))
        }
    });
    vm.add_native("Vec::new".into(), |_vm, _args| {
        Ok(Value::Vec(Rc::new(RefCell::new(Vec::new()))))
    });
    vm.add_native("compile_host".into(), |_vm, args| {
        if args.len() >= 2 {
            if let (Value::String(src), Value::String(out)) = (&args[0], &args[1]) {
                match tenth::lexer::lexer::Lexer::new(src).tokenize()
                    .and_then(|tokens| tenth::parser::parser::Parser::new(tokens).parse_program())
                    .and_then(|prog| tenth::hir::lower::Lowerer::new().lower_program(&prog))
                    .and_then(|hir| tenth::compile::compile_to_wasm(&hir))
                {
                    Ok(bytes) => { let _ = std::fs::write(out, &bytes); return Ok(Value::Int(0)); }
                    Err(_) => return Ok(Value::Int(1)),
                }
            }
        }
        Ok(Value::Int(1))
    });
    vm.add_native("compile_program".into(), |_vm, args| {
        if args.len() >= 2 {
            if let Value::String(out) = &args[1] {
                match tenth::compile::compile_program_to_wasm(&args[0]) {
                    Ok(bytes) => {
                        let _ = std::fs::write(out, &bytes);
                        return Ok(Value::Int(0));
                    }
                    Err(_) => return Ok(Value::Int(1)),
                }
            }
        }
        Ok(Value::Int(1))
    });

    // Compile each function to bytecode
    for func in &hir.functions {
        let compiler = BytecodeCompiler::new();
        match compiler.compile(func) {
            Ok(chunk) => {
                vm.add_fn(func.name.clone(), chunk);
                // Also set as global so it can be called
                vm.set_global(func.name.clone(), Value::FnRef {
                    name: func.name.clone(),
                    params: func.params.clone(),
                    return_type: func.return_type.clone(),
                });
            }
            Err(_) => {
                // Fallback to tree-walk if compilation fails
            }
        }
    }

    // Execute main
    if vm.functions.contains_key("main") {
        match vm.call("main") {
            Ok(val) => {
                if !matches!(val, Value::Unit) {
                    println!("= {}", val);
                }
            }
            Err(e) => {
                // Fallback to tree-walk interpreter
                eprintln!("[vm] error: {} — falling back to interpreter", e);
                let mut interpreter = Interpreter::new(&hir);
                match interpreter.execute_program(&hir)? {
                    Some(val) => println!("= {}", val),
                    None => {}
                }
            }
        }
    } else if let Some(ref expr) = hir.main_expr {
        let compiler = BytecodeCompiler::new();
        match compiler.compile_main(expr) {
            Ok(chunk) => {
                vm.add_fn("main".into(), chunk);
                let val = vm.call("main")?;
                if !matches!(val, Value::Unit) {
                    println!("= {}", val);
                }
            }
            Err(_) => {
                let mut interpreter = Interpreter::new(&hir);
                match interpreter.execute_program(&hir)? {
                    Some(val) => println!("= {}", val),
                    None => {}
                }
            }
        }
    }

    Ok(())
}
