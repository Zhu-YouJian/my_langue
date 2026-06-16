use std::fs;
use std::path::Path;
use std::process::Command;

use crate::manifest::Manifest;

pub fn test() -> Result<(), Box<dyn std::error::Error>> {
    let manifest_path = Path::new("Tenth.toml");
    if !manifest_path.exists() {
        return Err("Tenth.toml not found in current directory".into());
    }

    let manifest = Manifest::load_from_file(manifest_path)?;

    let tests_dir = Path::new("tests");
    if !tests_dir.exists() {
        println!("No tests/ directory found. Nothing to test.");
        return Ok(());
    }

    let mut test_files = Vec::new();
    collect_th_files(tests_dir, &mut test_files)?;

    if test_files.is_empty() {
        println!("No test files found in tests/");
        return Ok(());
    }

    println!("Running tests for `{}` v{}", manifest.package.name, manifest.package.version);

    let _total = test_files.len();
    let mut passed = 0;
    let mut failed = 0;

    for file in &test_files {
        let file_name = file.display().to_string();
        print!("  test {} ... ", file_name);

        match run_test(file) {
            Ok(()) => {
                println!("ok");
                passed += 1;
            }
            Err(e) => {
                println!("FAILED");
                println!("    {}", e);
                failed += 1;
            }
        }
    }

    println!();
    if failed > 0 {
        println!("Test result: FAILED. {} passed; {} failed.", passed, failed);
        Err(format!("{} test(s) failed", failed).into())
    } else {
        println!("Test result: ok. {} passed; {} failed.", passed, failed);
        Ok(())
    }
}

fn run_test(path: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let tenth_bin = find_tenth_binary();

    // Try running with the tenth binary
    let output = Command::new(&tenth_bin)
        .arg("run")
        .arg(path)
        .output();

    match output {
        Ok(out) => {
            if out.status.success() {
                Ok(())
            } else {
                let stderr = String::from_utf8_lossy(&out.stderr);
                let stdout = String::from_utf8_lossy(&out.stdout);
                let mut msg = String::new();
                if !stdout.is_empty() { msg.push_str(&stdout); }
                if !stderr.is_empty() { msg.push_str(&stderr); }
                Err(msg.into())
            }
        }
        Err(_) => {
            // Fallback: run in-process
            run_test_in_process(path)
        }
    }
}

fn run_test_in_process(path: &Path) -> Result<(), Box<dyn std::error::Error>> {
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

fn collect_th_files(dir: &Path, files: &mut Vec<std::path::PathBuf>) -> Result<(), Box<dyn std::error::Error>> {
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir() {
            collect_th_files(&path, files)?;
        } else if path.extension().is_some_and(|ext| ext == "th") {
            files.push(path);
        }
    }
    Ok(())
}
