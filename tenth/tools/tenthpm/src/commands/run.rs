use std::path::Path;
use std::process::Command;

use crate::manifest::Manifest;

pub fn run() -> Result<(), Box<dyn std::error::Error>> {
    let manifest_path = Path::new("Tenth.toml");
    if !manifest_path.exists() {
        return Err("Tenth.toml not found in current directory".into());
    }

    let manifest = Manifest::load_from_file(manifest_path)?;

    let main_path = Path::new("src/main.th");
    if !main_path.exists() {
        return Err("src/main.th not found".into());
    }

    println!("Compiling `{}` v{} ...", manifest.package.name, manifest.package.version);

    // Try to run using the `tenth` binary
    let tenth_bin = find_tenth_binary();

    let status = Command::new(&tenth_bin)
        .arg("run")
        .arg(main_path)
        .status();

    match status {
        Ok(s) => {
            if !s.success() {
                return Err("Execution failed".into());
            }
        }
        Err(_) => {
            // Fallback: compile and run in-process
            println!("  (running in-process)");
            run_in_process(main_path)?;
        }
    }

    Ok(())
}

/// Run a .th file in-process using the Tenth compiler and interpreter.
fn run_in_process(path: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let source = std::fs::read_to_string(path)
        .map_err(|e| format!("cannot read {}: {}", path.display(), e))?;

    let mut lexer = tenth::lexer::lexer::Lexer::new(&source);
    let tokens = lexer.tokenize()
        .map_err(|e| format!("lexer error: {}", e))?;

    let mut parser = tenth::parser::parser::Parser::new(tokens);
    let program = parser.parse_program()
        .map_err(|e| format!("parse error: {}", e))?;

    let mut lowerer = tenth::hir::lower::Lowerer::new();
    let hir = lowerer.lower_program(&program)
        .map_err(|e| format!("type error: {}", e))?;

    let mut interpreter = tenth::runtime::interpreter::Interpreter::new(&hir);
    interpreter.execute_program(&hir)
        .map_err(|e| format!("runtime error: {}", e))?;

    Ok(())
}

fn find_tenth_binary() -> String {
    if let Ok(bin) = std::env::var("TENTH_BIN") {
        return bin;
    }
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            let candidate = dir.join("tenth");
            if candidate.exists() {
                return candidate.to_string_lossy().to_string();
            }
        }
    }
    "tenth".to_string()
}
