use tenth::error::TenthResult;
use tenth::repl;
use tenth::lexer::lexer::Lexer;
use tenth::parser::parser::Parser;
use tenth::hir::lower::Lowerer;
use tenth::compile;
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
            let status = Command::new(gcc)
                .args(["-o", output_file, &c_file, runtime, "-lm"])
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
    } else {
        repl::run_repl()?
    }
    Ok(())
}