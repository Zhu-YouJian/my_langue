use tenth::error::TenthResult;
use tenth::repl;
use tenth::runtime::limits::MemoryConfig;
use tenth::lexer::lexer::Lexer;
use tenth::parser::parser::Parser;
use tenth::hir::lower::Lowerer;
use tenth::runtime::interpreter::Interpreter;
use tenth::compile;

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

/// Interpret a .th source file via the tree-walk interpreter.
fn run_file(path: &str) -> TenthResult<()> {
    let source = std::fs::read_to_string(path)
        .map_err(|e| tenth::error::TenthError::RuntimeError {
            message: format!("cannot read {}: {}", path, e),
        })?;
    let hir = source_to_hir(&source)?;
    let mut interpreter = Interpreter::new(&hir);
    match interpreter.execute_program(&hir)? {
        Some(val) => println!("= {}", val),
        None => {}
    }
    Ok(())
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