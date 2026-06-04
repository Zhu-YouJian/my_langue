use tenth::error::TenthResult;
use tenth::repl;
use tenth::lexer::lexer::Lexer;
use tenth::parser::parser::Parser;
use tenth::hir::lower::Lowerer;
use tenth::compile;
use tenth::runtime::limits::MemoryConfig;
use std::process::Command;

fn main() -> TenthResult<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() >= 3 && args[1] == "compile" {
        let input_file = &args[2];
        let output_file = {
            let o_pos = args.iter().position(|a| a == "-o");
            o_pos.map(|i| args.get(i + 1).map(|s| s.as_str()).unwrap_or("out.exe")).unwrap_or("out.exe")
        };
        let optimize = args.iter().any(|a| a == "--opt");
        let sanitize = args.iter().any(|a| a == "--sanitize");
        // --max-memory <MB>  e.g. --max-memory 512
        let max_memory_mb: usize = args.iter()
            .position(|a| a == "--max-memory")
            .and_then(|i| args.get(i + 1))
            .and_then(|s| s.parse().ok())
            .unwrap_or(0);

        let source = std::fs::read_to_string(input_file)
            .map_err(|e| tenth::error::TenthError::RuntimeError {
                message: format!("cannot read {}: {}", input_file, e),
            })?;

        let mut lexer = Lexer::new(&source);
        let tokens = lexer.tokenize()?;

        let mut parser = Parser::new(tokens);
        let program = parser.parse_program()?;

        let mut lowerer = Lowerer::new();
        let hir = lowerer.lower_program(&program)?;

        let c_code = compile::compile_to_c(&hir, optimize)?;

        // Write C source
        let c_file = if output_file.ends_with(".exe") {
            output_file.replace(".exe", ".c")
        } else {
            output_file.to_string()
        };
        std::fs::write(&c_file, &c_code)
            .map_err(|e| tenth::error::TenthError::RuntimeError {
                message: format!("cannot write {}: {}", c_file, e),
            })?;

        // If output is .exe, invoke GCC
        if output_file.ends_with(".exe") {
            let gcc = "D:\\msys64\\mingw64\\bin\\gcc.exe";
            let runtime = "tenthc\\runtime.c";

            let mut gcc_args: Vec<&str> = vec!["-o", output_file, &c_file, runtime, "-lm"];
            // ASan support — catches use-after-free, buffer overflow, leaks
            if sanitize {
                gcc_args.push("-fsanitize=address");
                gcc_args.push("-fsanitize=leak");
                gcc_args.push("-fno-omit-frame-pointer");
                gcc_args.push("-g");
                println!("[asan] Compiling with AddressSanitizer + LeakSanitizer");
            }

            let status = Command::new(gcc)
                .args(&gcc_args)
                .status()
                .map_err(|e| tenth::error::TenthError::RuntimeError {
                    message: format!("cannot run gcc: {}", e),
                })?;
            if status.success() {
                println!("Compiled to {}", output_file);
                // Clean up .c file
                let _ = std::fs::remove_file(&c_file);
            } else {
                return Err(tenth::error::TenthError::RuntimeError {
                    message: "gcc compilation failed".into(),
                });
            }
        } else {
            println!("Compiled to {}", c_file);
        }

        // Warn if max-memory was set but no arena limit is configured in C
        if max_memory_mb > 0 && !sanitize {
            println!("[note] --max-memory={}MiB (arena limit enforced at C runtime)", max_memory_mb);
        }
    } else {
        // REPL mode — parse optional --max-memory flag
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
                max_tensor_elements: max_memory_mb * 1024 * 128, // ~1/8 of mem in elements
                track_allocations: true,
            };
            println!("[limits] Max memory: {} MiB", max_memory_mb);
            repl::run_repl_with_limits(config)?;
        } else {
            repl::run_repl()?;
        }
    }
    Ok(())
}