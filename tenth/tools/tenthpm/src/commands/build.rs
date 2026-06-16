use std::path::Path;
use std::process::Command;

use crate::manifest::Manifest;

pub fn build() -> Result<(), Box<dyn std::error::Error>> {
    let manifest_path = Path::new("Tenth.toml");
    if !manifest_path.exists() {
        return Err("Tenth.toml not found in current directory".into());
    }

    let manifest = Manifest::load_from_file(manifest_path)?;
    println!("Building `{}` v{} ...", manifest.package.name, manifest.package.version);

    let src_dir = Path::new("src");
    if !src_dir.exists() {
        return Err("src/ directory not found".into());
    }

    let main_path = src_dir.join("main.th");
    if !main_path.exists() {
        return Err("src/main.th not found".into());
    }

    // Compile by running the Tenth compiler on the main file
    // This validates syntax and type-checks without executing
    let tenth_bin = find_tenth_binary();

    let output = Command::new(&tenth_bin)
        .arg("check")
        .arg(&main_path)
        .output();

    match output {
        Ok(out) => {
            if out.status.success() {
                println!("  Compiling src/main.th");
                println!("  Checking syntax and types...");
                println!("Build finished successfully.");
            } else {
                let stderr = String::from_utf8_lossy(&out.stderr);
                let stdout = String::from_utf8_lossy(&out.stdout);
                if !stdout.is_empty() {
                    eprintln!("{}", stdout);
                }
                if !stderr.is_empty() {
                    eprintln!("{}", stderr);
                }
                return Err("Build failed".into());
            }
        }
        Err(_) => {
            // Fallback: just verify the file can be parsed
            println!("  Compiling src/main.th");
            match verify_syntax(&main_path) {
                Ok(()) => {
                    println!("  Syntax check passed.");
                    println!("Build finished successfully.");
                }
                Err(e) => {
                    return Err(format!("Build failed: {}", e).into());
                }
            }
        }
    }

    Ok(())
}

/// Verify that a .th file can be lexed and parsed.
fn verify_syntax(path: &Path) -> Result<(), String> {
    let source = std::fs::read_to_string(path)
        .map_err(|e| format!("cannot read {}: {}", path.display(), e))?;

    let mut lexer = tenth::lexer::lexer::Lexer::new(&source);
    let tokens = lexer.tokenize()
        .map_err(|e| format!("lexer error: {}", e))?;

    let mut parser = tenth::parser::parser::Parser::new(tokens);
    let _program = parser.parse_program()
        .map_err(|e| format!("parse error: {}", e))?;

    Ok(())
}

/// Find the `tenth` binary, checking CARGO_BIN first, then PATH.
fn find_tenth_binary() -> String {
    // When running via cargo, the binary may be in target/debug/
    if let Ok(bin) = std::env::var("TENTH_BIN") {
        return bin;
    }

    // Check relative to current exe
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            let candidate = dir.join("tenth");
            if candidate.exists() {
                return candidate.to_string_lossy().to_string();
            }
        }
    }

    // Fallback: assume `tenth` is on PATH
    "tenth".to_string()
}
