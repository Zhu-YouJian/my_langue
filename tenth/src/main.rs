use std::rc::Rc;
use tenth::hir::types::BaseType;
use std::cell::RefCell;
use tenth::error::TenthResult;
use tenth::repl;
use tenth::runtime::limits::{MemoryConfig, FsSandbox};
use tenth::lexer::lexer::Lexer;
use tenth::parser::parser::Parser;
use tenth::runtime::interpreter::Interpreter;
use tenth::runtime::vm::Vm;
use tenth::runtime::value::Value;
use tenth::compile;
use tenth::compile::bytecode::BytecodeCompiler;
use tenth::compile::jit;
use tenth::cli;
use tenth::runtime::natives;

fn main() {
    // Windows 中文终端默认代码页 936（GBK），UTF-8 输出会乱码。
    // 在任何输出之前把控制台输出代码页设为 UTF-8 (65001)。
    // 只影响输出代码页，不影响 rustyline 输入处理，REPL 安全。
    #[cfg(windows)]
    unsafe {
        unsafe extern "C" {
            fn SetConsoleOutputCP(code_page: u32) -> i32;
        }
        SetConsoleOutputCP(65001);
    }
    if let Err(e) = run_main() {
        // Errors from lexer/parser/runtime are already formatted by run_file.
        // For other errors (e.g. file-not-found), print a plain message.
        eprintln!("Error: {}", e);
        std::process::exit(1);
    }
}

fn run_main() -> TenthResult<()> {
    let args: Vec<String> = std::env::args().collect();

    if args.len() >= 2 {
        match args[1].as_str() {
            "build" if args.len() >= 3 => {
                return build_wasm(&args[2]);
            }
            "run" if args.len() >= 3 => {
                // 安全：run_file 默认应用 MemoryConfig::default()，避免恶意 .th
                // 程序在 fallback 到解释器路径时无任何内存护栏。
                // 用户可用 `--no-limits` 显式关闭，或 `--max-memory N` 自定义。
                let config = cli::parse_memory_config(&args[3..]);
                // H-2: 解析文件系统沙箱。默认无沙箱（向后兼容），
                // `--fs-root <dir>` 启用沙箱，`--read-only` 进一步限制为只读。
                let sandbox = cli::parse_fs_sandbox(&args[3..])?;
                // H-4: 解析墙钟超时（秒）。`--timeout <secs>` 启用。
                let timeout_ms = cli::parse_timeout_ms(&args[3..]);
                return run_file(&args[2], config, sandbox, timeout_ms);
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

    // Build search paths for file imports:
    //   1. The directory of the source file (if running from a file)
    //   2. The `std/` directory relative to the executable
    let mut search_paths = Vec::new();

    // Add current directory
    if let Ok(cwd) = std::env::current_dir() {
        search_paths.push(cwd.to_string_lossy().to_string());
    }

    // Add std/ directory relative to Cargo.toml / executable
    if let Ok(exe_dir) = std::env::current_exe()
        .map(|p| p.parent().map(|d| d.to_path_buf()).unwrap_or_default())
    {
        let std_near_exe = exe_dir.join("std");
        if std_near_exe.exists() {
            search_paths.push(std_near_exe.to_string_lossy().to_string());
        }
    }

    // Add tenth/std/ relative to working directory (for development)
    let std_dev = std::path::Path::new("tenth/std");
    if std_dev.exists() {
        // Add the parent of std/ (i.e., tenth/) so that `use std::json::json::parse`
        // resolves to tenth/std/json/json.th
        if let Some(parent) = std_dev.parent() {
            search_paths.push(parent.to_string_lossy().to_string());
        }
        // Also add tenth/std/ itself for use statements without the std:: prefix
        search_paths.push(std_dev.to_string_lossy().to_string());
    }

    // Also handle the case where cwd is already inside tenth/ (e.g., cwd = tenth/)
    let std_local = std::path::Path::new("std");
    if std_local.exists() {
        if let Some(parent) = std_local.parent() {
            let parent_str = parent.to_string_lossy().to_string();
            if !search_paths.iter().any(|p| *p == parent_str) {
                search_paths.push(parent_str);
            }
        }
        let std_str = std_local.to_string_lossy().to_string();
        if !search_paths.iter().any(|p| *p == std_str) {
            search_paths.push(std_str);
        }
    }

    let mut lowerer = tenth::hir::lower::Lowerer::with_search_paths(search_paths);
    lowerer.lower_program(&program)
}

/// Run a .th source file — try VM first, fall back to tree-walk interpreter.
fn run_file(path: &str, config: MemoryConfig, sandbox: Option<FsSandbox>, timeout_ms: Option<u128>) -> TenthResult<()> {
    let source = tenth::error::read_source(path)?;
    let hir = match source_to_hir(&source) {
        Ok(hir) => hir,
        Err(e) => {
            eprintln!("{}", e.display_with_source(Some(&source)));
            std::process::exit(1);
        }
    };

    // 输出编译期警告（内存/算力预估等，非致命）
    for w in &hir.warnings {
        eprintln!("{}", w.display_with_source(Some(&source)));
    }

    // 检测是否有测试函数：收集所有 is_test=true 的函数
    let test_fns: Vec<&str> = hir.functions.iter()
        .filter(|f| f.is_test)
        .map(|f| f.name.as_str())
        .collect();

    if !test_fns.is_empty() {
        // 测试模式：收集并执行所有测试函数，输出 PASS/FAIL 报告
        return run_tests(&hir, test_fns, &source);
    }

    // Skip VM if TENTH_NO_VM env var is set (for debugging interpreter)
    let skip_vm = std::env::var("TENTH_NO_VM").is_ok();
    if !skip_vm {
        match vm_execute(&hir, sandbox.clone(), timeout_ms) {
            Ok(val) => {
                if !matches!(val, Value::Unit) { println!("= {}", val); }
                return Ok(());
            }
            Err(tenth::error::TenthError::VmCompileFailed { message }) => {
                // 编译期失败：VM 不支持某些结构，无副作用，静默回退到解释器。
                // 不打印 warning——这是正常的能力差异，不是错误。
                eprintln!("[info] VM 不支持此程序结构（{}），使用解释器执行。", message);
            }
            Err(e) => {
                // 问题15修复：运行时失败（VM 已部分执行并可能产生副作用），
                // 不再静默回退到解释器（否则会导致副作用双重执行：println 两遍、
                // 文件写两次、网络请求两次）。改为硬失败，避免静默的双重副作用。
                eprintln!("[error] VM 运行时失败：{}", e);
                eprintln!("[error] VM 可能已部分执行并产生副作用，不会回退到解释器以避免副作用重复。");
                eprintln!("[error] 如需强制使用解释器（跳过 VM），请设置环境变量 TENTH_NO_VM=1 后重试。");
                return Err(e);
            }
        }
    }
    // 安全：fallback 到解释器时应用 MemoryConfig，避免 .th 程序触发 OOM。
    let limits = tenth::runtime::limits::RuntimeLimits::new(config);
    let mut interpreter = Interpreter::with_limits(&hir, limits);
    // H-2/H-4: 解释器也应用沙箱和超时
    interpreter.fs_sandbox = sandbox;
    interpreter.deadline_ms = timeout_ms;
    match interpreter.execute_program(&hir)? {
        // 问题16修复：与 VM 路径一致，过滤 Unit 值避免输出 "= ()"
        Some(val) if !matches!(val, Value::Unit) => println!("= {}", val),
        _ => {}
    }
    Ok(())
}

/// 执行测试模式：收集所有 #[test] 函数，逐一执行并输出 PASS/FAIL 报告。
fn run_tests(hir: &tenth::hir::hir::HirProgram, test_fns: Vec<&str>, _source: &str) -> TenthResult<()> {
    let mut passed = 0usize;
    let mut failed = 0usize;
    let mut failures: Vec<String> = Vec::new();

    println!("running {} tests", test_fns.len());

    for fn_name in &test_fns {
        print!("test {} ... ", fn_name);

        // Use the full HirProgram (all functions/modules available)
        // Execute via interpreter
        let mut interpreter = Interpreter::new(&hir);

        // Try calling the test function directly
        match interpreter.execute_fn_test(fn_name) {
            Ok(_) => {
                println!("ok");
                passed += 1;
            }
            Err(e) => {
                println!("FAILED");
                failed += 1;
                failures.push(format!("{}: {}", fn_name, e));
            }
        }
    }

    println!();
    if !failures.is_empty() {
        for fail in &failures {
            eprintln!("{}", fail);
        }
    }
    println!("test result: {} passed; {} failed", passed, failed);

    if failed > 0 {
        std::process::exit(1);
    }
    Ok(())
}

/// Execute a HirProgram via the VM. Returns the result or an error.
fn vm_execute(hir: &tenth::hir::hir::HirProgram, sandbox: Option<FsSandbox>, timeout_ms: Option<u128>) -> TenthResult<Value> {
    let mut vm = Vm::new();
    // H-2/H-4: 应用文件系统沙箱和墙钟超时
    vm.fs_sandbox = sandbox;
    vm.deadline_ms = timeout_ms;
    natives::register_all_natives(&mut vm);

    for func in &hir.functions {
        let compiler = BytecodeCompiler::new();
        if let Ok((chunk, closures)) = compiler.compile(func) {
            vm.add_fn(func.name.clone(), chunk);
            for (name, closure_chunk) in closures {
                vm.add_fn(name, closure_chunk);
            }
            // Also register the function as a global FnRef so it can be passed
            // as a value (e.g. compose(double, inc)) and called by name from
            // closures. Without this, LoadGlobal for a function name returns
            // Unit, breaking higher-order function scenarios.
            vm.set_global(func.name.clone(), Value::FnRef {
                name: func.name.clone(),
                params: func.params.clone(),
                return_type: func.return_type.clone(),
            });
        }
    }

    if vm.has_fn("main") {
        jit::run_jit(&mut vm, "main")
    } else if let Some(ref expr) = hir.main_expr {
        let compiler = BytecodeCompiler::new();
        if let Ok((chunk, closures)) = compiler.compile_main(expr) {
            vm.add_fn("main".into(), chunk);
            for (name, closure_chunk) in closures {
                vm.add_fn(name, closure_chunk);
            }
        } else {
            // 编译期失败：VM 不支持 main_expr 中的结构，无副作用，可安全回退
            return Err(tenth::error::TenthError::VmCompileFailed {
                message: "VM 编译失败".into(),
            });
        }
        jit::run_jit(&mut vm, "main")
    } else if hir.functions.is_empty() {
        Ok(Value::Unit)
    } else {
        // 编译期失败：函数存在但 VM 无法编译 main，无副作用，可安全回退
        Err(tenth::error::TenthError::VmCompileFailed {
            message: "VM: main 未编译（包含不支持的结构，回退到解释器）".into(),
        })
    }
}

/// Compile a .th file to a .wasm binary.
fn build_wasm(path: &str) -> TenthResult<()> {
    let source = tenth::error::read_source(path)?;
    let hir = source_to_hir(&source)?;
    let wasm_bytes = compile::compile_to_wasm(&hir)?;

    let out_path = path.replace(".th", ".wasm");
    std::fs::write(&out_path, &wasm_bytes)
        .map_err(|e| tenth::error::TenthError::RuntimeError { line: None, col: None,
            message: format!("无法写入 {}：{}", out_path, e),
        })?;
    println!("Compiled to {}", out_path);
    Ok(())
}

/// Compile a .th file to WASM and execute it via wasmi.
fn run_wasm(path: &str) -> TenthResult<()> {
    let source = tenth::error::read_source(path)?;
    let hir = source_to_hir(&source)?;
    compile::run_wasm(&hir)
}

/// Compile a .th file to bytecode and execute via the stack VM.
#[allow(dead_code)]
fn vm_run(path: &str) -> TenthResult<()> {
    let source = tenth::error::read_source(path)?;
    let hir = source_to_hir(&source)?;
    let mut vm = Vm::new();

    // Register native functions (delegate to central register_all_natives)
    natives::register_all_natives(&mut vm);
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
                    Ok(bytes) => { let _ = std::fs::write(out, &bytes); return Ok(Value::Int(0, BaseType::I32)); }
                    Err(_) => return Ok(Value::Int(1, BaseType::I32)),
                }
            }
        }
        Ok(Value::Int(1, BaseType::I32))
    });
    vm.add_native("compile_program".into(), |_vm, args| {
        if args.len() >= 2 {
            if let Value::String(out) = &args[1] {
                match tenth::compile::compile_program_to_wasm(&args[0]) {
                    Ok(bytes) => {
                        let _ = std::fs::write(out, &bytes);
                        return Ok(Value::Int(0, BaseType::I32));
                    }
                    Err(_) => return Ok(Value::Int(1, BaseType::I32)),
                }
            }
        }
        Ok(Value::Int(1, BaseType::I32))
    });

    // Compile each function to bytecode
    for func in &hir.functions {
        let compiler = BytecodeCompiler::new();
        match compiler.compile(func) {
            Ok((chunk, closures)) => {
                vm.add_fn(func.name.clone(), chunk);
                for (name, closure_chunk) in closures {
                    vm.add_fn(name, closure_chunk);
                }
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
                // 问题15修复：VM 运行时失败不回退到解释器，避免双重副作用。
                // 用户显式使用 --vm 子命令，应报告失败而非静默回退。
                eprintln!("[vm] error: {}", e);
                eprintln!("[vm] VM 运行时失败，不回退到解释器以避免副作用重复。");
                return Err(e);
            }
        }
    } else if let Some(ref expr) = hir.main_expr {
        let compiler = BytecodeCompiler::new();
        match compiler.compile_main(expr) {
            Ok((chunk, closures)) => {
                vm.add_fn("main".into(), chunk);
                for (name, closure_chunk) in closures {
                    vm.add_fn(name, closure_chunk);
                }
                let val = vm.call("main")?;
                if !matches!(val, Value::Unit) {
                    println!("= {}", val);
                }
            }
            Err(_) => {
                let mut interpreter = Interpreter::new(&hir);
                match interpreter.execute_program(&hir)? {
                    Some(val) if !matches!(val, Value::Unit) => println!("= {}", val),
                    _ => {}
                }
            }
        }
    }

    Ok(())
}
