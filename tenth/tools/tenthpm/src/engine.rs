use std::path::{Path, PathBuf};

use tenth::hir::hir::HirProgram;
use tenth::hir::lower::Lowerer;
use tenth::lexer::lexer::Lexer;
use tenth::parser::parser::Parser;
use tenth::runtime::interpreter::Interpreter;
use tenth::runtime::value::Value;
use tenth::runtime::vm::Vm;
use tenth::compile::bytecode::BytecodeCompiler;
use tenth::compile::jit;

/// Build the list of search paths for `use` imports.
///
/// Order:
/// 1. The directory containing the source file (so relative imports work)
/// 2. `deps/` in the current project root (path / git dependencies)
/// 3. `std/` relative to the `tenth` executable (installed standard library)
/// 4. `tenth/std/` relative to CWD (development mode)
pub fn build_search_paths(source_path: &Path) -> Vec<String> {
    let mut paths: Vec<String> = Vec::new();

    // 1. Directory of the source file
    if let Some(parent) = source_path.parent() {
        paths.push(parent.to_string_lossy().to_string());
    }

    // 2. Project-level deps/ directory (path & git dependencies live here)
    let deps_dir = Path::new("deps");
    if deps_dir.exists() {
        paths.push(deps_dir.to_string_lossy().to_string());
        // Also add each subdirectory of deps/ so `use pkg::module` works
        if let Ok(entries) = std::fs::read_dir(deps_dir) {
            for entry in entries.flatten() {
                let p = entry.path();
                if p.is_dir() {
                    let src = p.join("src");
                    if src.exists() {
                        paths.push(src.to_string_lossy().to_string());
                    }
                    paths.push(p.to_string_lossy().to_string());
                }
            }
        }
    }

    // 3. std/ relative to the tenth executable (installed layout)
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            let std_near_exe = dir.join("std");
            if std_near_exe.exists() {
                paths.push(std_near_exe.to_string_lossy().to_string());
            }
            // Also check one level up (tenthpm lives in tenth/tools/tenthpm/)
            if let Some(grandparent) = dir.parent() {
                let std_up = grandparent.join("std");
                if std_up.exists() {
                    paths.push(std_up.to_string_lossy().to_string());
                }
            }
        }
    }

    // 4. tenth/std/ relative to CWD (development mode)
    let std_dev = Path::new("tenth/std");
    if std_dev.exists() {
        paths.push(std_dev.to_string_lossy().to_string());
    }

    paths
}

/// Lex → Parse → Lower → HIR with proper search paths.
pub fn source_to_hir(source: &str, source_path: &Path) -> Result<HirProgram, String> {
    let mut lexer = Lexer::new(source);
    let tokens = lexer.tokenize().map_err(|e| format!("词法错误: {}", e))?;

    let mut parser = Parser::new(tokens);
    let program = parser.parse_program().map_err(|e| format!("语法错误: {}", e))?;

    let search_paths = build_search_paths(source_path);
    let mut lowerer = Lowerer::with_search_paths(search_paths);
    lowerer
        .lower_program(&program)
        .map_err(|e| format!("{}", e.display_with_source(Some(source))))
}

/// Run a .th file in-process: VM first, interpreter fallback.
pub fn run_file(path: &Path) -> Result<(), String> {
    let source = std::fs::read_to_string(path)
        .map_err(|e| format!("无法读取 {}: {}", path.display(), e))?;

    let hir = source_to_hir(&source, path)?;

    // Try VM first
    match vm_execute(&hir) {
        Ok(_) => return Ok(()),
        Err(_) => {}
    }

    // Fallback to tree-walk interpreter
    let mut interpreter = Interpreter::new(&hir);
    interpreter
        .execute_program(&hir)
        .map_err(|e| format!("运行时错误: {}", e))?;
    Ok(())
}

/// Check (compile-only) a .th file: lex + parse + lower, no execution.
pub fn check_file(path: &Path) -> Result<(), String> {
    let source = std::fs::read_to_string(path)
        .map_err(|e| format!("无法读取 {}: {}", path.display(), e))?;
    source_to_hir(&source, path)?;
    Ok(())
}

/// Execute a HirProgram via the VM.
fn vm_execute(hir: &HirProgram) -> Result<Value, String> {
    let mut vm = Vm::new();
    register_natives(&mut vm);

    for func in &hir.functions {
        let compiler = BytecodeCompiler::new();
        if let Ok((chunk, closures)) = compiler.compile(func) {
            vm.add_fn(func.name.clone(), chunk);
            for (name, closure_chunk) in closures {
                vm.add_fn(name, closure_chunk);
            }
        }
    }

    if vm.has_fn("main") {
        jit::run_jit(&mut vm, "main").map_err(|e| format!("{}", e))
    } else if let Some(ref expr) = hir.main_expr {
        let compiler = BytecodeCompiler::new();
        if let Ok((chunk, closures)) = compiler.compile_main(expr) {
            vm.add_fn("main".into(), chunk);
            for (name, closure_chunk) in closures {
                vm.add_fn(name, closure_chunk);
            }
            jit::run_jit(&mut vm, "main").map_err(|e| format!("{}", e))
        } else {
            Err("VM 编译失败".to_string())
        }
    } else if hir.functions.is_empty() {
        Ok(Value::Unit)
    } else {
        Err("VM: main 未编译".to_string())
    }
}

/// Register essential native functions for VM execution.
fn register_natives(vm: &mut Vm) {
    use std::rc::Rc;
    use std::cell::RefCell;

    vm.add_native("println".into(), |_vm, args| {
        for a in args { print!("{a}"); }
        println!();
        Ok(Value::Unit)
    });
    vm.add_native("print".into(), |_vm, args| {
        for a in args { print!("{a}"); }
        Ok(Value::Unit)
    });
    vm.add_native("read_file".into(), |_vm, args| {
        if let Some(Value::String(path)) = args.first() {
            match std::fs::read_to_string(path) {
                Ok(s) => Ok(Value::String(s)),
                Err(e) => Err(tenth::error::TenthError::RuntimeError { line: None, col: None,
                    message: format!("读取文件: {e}"),
                }),
            }
        } else {
            Ok(Value::String(String::new()))
        }
    });
    vm.add_native("write_file".into(), |_vm, args| {
        if args.len() >= 2 {
            if let (Value::String(path), Value::String(content)) = (&args[0], &args[1]) {
                match std::fs::write(path, content) {
                    Ok(()) => Ok(Value::Unit),
                    Err(e) => Err(tenth::error::TenthError::RuntimeError { line: None, col: None,
                        message: format!("写入文件失败: {}", e),
                    }),
                }
            } else {
                Err(tenth::error::TenthError::RuntimeError { line: None, col: None,
                    message: "write_file(路径, 内容) 期望两个字符串参数".into(),
                })
            }
        } else {
            Err(tenth::error::TenthError::RuntimeError { line: None, col: None,
                message: "write_file(路径, 内容) 期望两个字符串参数".into(),
            })
        }
    });
    vm.add_native("Vec::new".into(), |_vm, _args| {
        Ok(Value::Vec(Rc::new(RefCell::new(Vec::new()))))
    });
    vm.add_native("HashMap::new".into(), |_vm, _args| {
        Ok(Value::Map(Rc::new(RefCell::new(
            std::collections::HashMap::new(),
        ))))
    });
    vm.add_native("path_exists".into(), |_vm, args| {
        if let Some(Value::String(path)) = args.first() {
            Ok(Value::Bool(std::path::Path::new(path).exists()))
        } else {
            Err(tenth::error::TenthError::RuntimeError { line: None, col: None,
                message: "path_exists(路径) 期望一个字符串路径".into(),
            })
        }
    });
    vm.add_native("path_is_file".into(), |_vm, args| {
        if let Some(Value::String(path)) = args.first() {
            Ok(Value::Bool(std::path::Path::new(path).is_file()))
        } else {
            Err(tenth::error::TenthError::RuntimeError { line: None, col: None,
                message: "path_is_file(路径) 期望一个字符串路径".into(),
            })
        }
    });
    vm.add_native("path_is_dir".into(), |_vm, args| {
        if let Some(Value::String(path)) = args.first() {
            Ok(Value::Bool(std::path::Path::new(path).is_dir()))
        } else {
            Err(tenth::error::TenthError::RuntimeError { line: None, col: None,
                message: "path_is_dir(路径) 期望一个字符串路径".into(),
            })
        }
    });
    vm.add_native("list_dir".into(), |_vm, args| {
        if let Some(Value::String(path)) = args.first() {
            match std::fs::read_dir(path) {
                Ok(entries) => {
                    let items: Vec<Value> = entries
                        .filter_map(|e| e.ok())
                        .map(|e| Value::String(e.file_name().to_string_lossy().to_string()))
                        .collect();
                    Ok(Value::Vec(Rc::new(RefCell::new(items))))
                }
                Err(e) => Err(tenth::error::TenthError::RuntimeError { line: None, col: None,
                    message: format!("列出目录失败: {}", e),
                }),
            }
        } else {
            Err(tenth::error::TenthError::RuntimeError { line: None, col: None,
                message: "list_dir(路径) 期望一个字符串路径".into(),
            })
        }
    });
    vm.add_native("mkdir".into(), |_vm, args| {
        if let Some(Value::String(path)) = args.first() {
            match std::fs::create_dir_all(path) {
                Ok(()) => Ok(Value::Unit),
                Err(e) => Err(tenth::error::TenthError::RuntimeError { line: None, col: None,
                    message: format!("创建目录失败: {}", e),
                }),
            }
        } else {
            Err(tenth::error::TenthError::RuntimeError { line: None, col: None,
                message: "mkdir(路径) 期望一个字符串路径".into(),
            })
        }
    });
    vm.add_native("abs".into(), |_vm, args| match args.first() {
        Some(Value::Int(n)) => Ok(Value::Int(n.abs())),
        Some(Value::Float(f)) => Ok(Value::Float(f.abs())),
        _ => Err(tenth::error::TenthError::RuntimeError { line: None, col: None,
            message: "abs() 需要一个数值参数".into(),
        }),
    });
    vm.add_native("sqrt".into(), |_vm, args| match args.first() {
        Some(Value::Float(f)) => Ok(Value::Float(f.sqrt())),
        Some(Value::Int(n)) => Ok(Value::Float((*n as f64).sqrt())),
        _ => Err(tenth::error::TenthError::RuntimeError { line: None, col: None,
            message: "sqrt() 需要一个数值参数".into(),
        }),
    });
    vm.add_native("to_float".into(), |_vm, args| match args.first() {
        Some(Value::Int(n)) => Ok(Value::Float(*n as f64)),
        Some(Value::Float(f)) => Ok(Value::Float(*f)),
        _ => Err(tenth::error::TenthError::RuntimeError { line: None, col: None,
            message: "to_float() 需要一个数值参数".into(),
        }),
    });
}

/// Find all .th files under a directory, recursively.
pub fn collect_th_files(dir: &Path) -> Result<Vec<PathBuf>, String> {
    let mut files = Vec::new();
    collect_th_files_inner(dir, &mut files)?;
    files.sort();
    Ok(files)
}

fn collect_th_files_inner(
    dir: &Path,
    files: &mut Vec<PathBuf>,
) -> Result<(), String> {
    let entries = std::fs::read_dir(dir).map_err(|e| format!("读取目录 {}: {}", dir.display(), e))?;
    for entry in entries {
        let entry = entry.map_err(|e| format!("读取目录项: {}", e))?;
        let path = entry.path();
        if path.is_dir() {
            collect_th_files_inner(&path, files)?;
        } else if path.extension().is_some_and(|ext| ext == "th") {
            files.push(path);
        }
    }
    Ok(())
}


