use std::rc::Rc;
use std::cell::RefCell;
use std::path::Path;
use tenth::error::TenthResult;
use tenth::repl;
use tenth::runtime::limits::{MemoryConfig, FsSandbox};
use tenth::lexer::lexer::Lexer;
use tenth::parser::parser::Parser;
use tenth::runtime::interpreter::Interpreter;
use tenth::runtime::interpreter::{json, datetime};
use tenth::runtime::vm::Vm;
use tenth::runtime::value::Value;
use tenth::runtime::autodiff::Tape;
use tenth::runtime::tensor::Tensor;
use tenth::compile;
use tenth::compile::bytecode::BytecodeCompiler;
use tenth::compile::jit;

fn main() {
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
                let config = parse_memory_config(&args[3..]);
                // H-2: 解析文件系统沙箱。默认无沙箱（向后兼容），
                // `--fs-root <dir>` 启用沙箱，`--read-only` 进一步限制为只读。
                let sandbox = parse_fs_sandbox(&args[3..])?;
                // H-4: 解析墙钟超时（秒）。`--timeout <secs>` 启用。
                let timeout_ms = parse_timeout_ms(&args[3..]);
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

/// 从命令行参数解析内存配置。
/// - `--no-limits` → 无限制（用户自担风险）
/// - `--max-memory N` → 自定义 N MiB
/// - 默认 → `MemoryConfig::default()`（256 MiB arena / 2 GiB 张量元素上限）
fn parse_memory_config(args: &[String]) -> MemoryConfig {
    if args.iter().any(|a| a == "--no-limits") {
        return MemoryConfig::unbounded();
    }
    if let Some(mb) = args.iter()
        .position(|a| a == "--max-memory")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse::<usize>().ok())
    {
        if mb == 0 {
            return MemoryConfig::unbounded();
        }
        return MemoryConfig {
            max_arena_bytes: mb * 1024 * 1024,
            max_variables: 10_000,
            max_accumulated_defs: 5_000,
            max_tensor_elements: mb * 1024 * 128,
            track_allocations: true,
        };
    }
    MemoryConfig::default()
}

/// H-2: 解析文件系统沙箱选项。
/// - `--fs-root <dir>` → 启用沙箱，根目录为 `dir`
/// - `--read-only` → 沙箱只读（必须配合 `--fs-root`）
/// - `--fs-cwd` → 以当前工作目录为沙箱根（等价 `--fs-root .`）
/// - 默认 → `None`（无沙箱，向后兼容）
///
/// 沙箱启用后，所有 `.th` 程序的文件 I/O 原生函数（read_file/write_file/
/// remove_file/mkdir/copy_file/rename_file/compile_host 等）必须经过
/// FsSandbox::check_read/check_write 校验，防止读写沙箱外的文件
/// （如 ~/.ssh/id_rsa、/etc/passwd）。
fn parse_fs_sandbox(args: &[String]) -> TenthResult<Option<FsSandbox>> {
    let read_only = args.iter().any(|a| a == "--read-only");
    if let Some(root) = args.iter()
        .position(|a| a == "--fs-root")
        .and_then(|i| args.get(i + 1))
    {
        let sb = FsSandbox::new(Path::new(root), read_only)
            .map_err(|e| tenth::error::TenthError::RuntimeError { message: e })?;
        return Ok(Some(sb));
    }
    if args.iter().any(|a| a == "--fs-cwd") {
        let sb = FsSandbox::cwd(read_only)
            .map_err(|e| tenth::error::TenthError::RuntimeError { message: e })?;
        return Ok(Some(sb));
    }
    if read_only {
        return Err(tenth::error::TenthError::RuntimeError {
            message: "--read-only 必须配合 --fs-root <dir> 或 --fs-cwd 使用".into(),
        });
    }
    Ok(None)
}

/// H-4: 解析墙钟超时（秒）。
/// - `--timeout <secs>` → 设置超时，返回 `Some(now_ms + secs * 1000)`
/// - 默认 → `None`（无超时，向后兼容）
///
/// 防止 `while true {}` 永久挂起宿主进程。VM 和 Interpreter 在主循环中
/// 周期性检查 `now >= deadline`，超时返回 `TenthError::Timeout`。
fn parse_timeout_ms(args: &[String]) -> Option<u128> {
    let secs = args.iter()
        .position(|a| a == "--timeout")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse::<u64>().ok())?;
    let now_ms = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0);
    // 用 checked 防止 secs * 1000 溢出（u64::MAX / 1000 ≈ 10^16 秒，远超实际）
    let timeout_ms = secs.checked_mul(1000)? as u128;
    Some(now_ms.checked_add(timeout_ms)?)
}

/// Run a .th source file — try VM first, fall back to tree-walk interpreter.
fn run_file(path: &str, config: MemoryConfig, sandbox: Option<FsSandbox>, timeout_ms: Option<u128>) -> TenthResult<()> {
    let source = std::fs::read_to_string(path)
        .map_err(|e| tenth::error::TenthError::RuntimeError {
            message: format!("无法读取 {}：{}", path, e),
        })?;
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

    // Skip VM if TENTH_NO_VM env var is set (for debugging interpreter)
    let skip_vm = std::env::var("TENTH_NO_VM").is_ok();
    if !skip_vm {
        match vm_execute(&hir, sandbox.clone(), timeout_ms) {
            Ok(val) => {
                if !matches!(val, Value::Unit) { println!("= {}", val); }
                return Ok(());
            }
            Err(e) => {
                // Print a warning so the user knows why output may be duplicated or
                // why the interpreter is being used. VM may have partially executed
                // statements (e.g. println) before failing, so the interpreter
                // re-running from the start can produce duplicate side effects.
                eprintln!("[warning] VM 执行失败，回退到解释器重新执行整个程序。");
                eprintln!("[warning] 回退原因: {}", e);
                eprintln!("[warning] 注意: VM 可能已部分执行并产生副作用（如 println 输出），");
                eprintln!("[warning]       解释器将从头重新执行，可能导致副作用重复。");
                eprintln!("[warning] 如需禁用 VM，请设置环境变量 TENTH_NO_VM=1。");
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
        Some(val) => println!("= {}", val),
        None => {}
    }
    Ok(())
}

/// Execute a HirProgram via the VM. Returns the result or an error.
fn vm_execute(hir: &tenth::hir::hir::HirProgram, sandbox: Option<FsSandbox>, timeout_ms: Option<u128>) -> TenthResult<Value> {
    let mut vm = Vm::new();
    // H-2/H-4: 应用文件系统沙箱和墙钟超时
    vm.fs_sandbox = sandbox;
    vm.deadline_ms = timeout_ms;
    register_natives(&mut vm);

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
            return Err(tenth::error::TenthError::RuntimeError {
                message: "VM 编译失败".into(),
            });
        }
        jit::run_jit(&mut vm, "main")
    } else if hir.functions.is_empty() {
        Ok(Value::Unit)
    } else {
        // Functions exist but none could be compiled for VM → signal fallback
        Err(tenth::error::TenthError::RuntimeError {
            message: "VM: main 未编译（包含不支持的结构，回退到解释器）".into(),
        })
    }
}

fn register_natives(vm: &mut Vm) {
    vm.add_native("println".into(), |_vm, args| {
        for a in args { print!("{a}"); }
        println!();
        Ok(Value::Unit)
    });
    vm.add_native("read_file".into(), |vm, args| {
        if let Some(Value::String(path)) = args.first() {
            // H-2: 沙箱校验
            let resolved = if let Some(ref sb) = vm.fs_sandbox {
                match sb.check_read(path) {
                    Ok(p) => p,
                    Err(e) => return Err(tenth::error::TenthError::RuntimeError { message: e }),
                }
            } else {
                std::path::PathBuf::from(path)
            };
            match std::fs::read_to_string(&resolved) {
                Ok(s) => Ok(Value::String(s)),
                Err(e) => Err(tenth::error::TenthError::RuntimeError { message: format!("读取文件: {e}") }),
            }
        } else {
            Ok(Value::String(String::new()))
        }
    });
    vm.add_native("Vec::new".into(), |_vm, _args| {
        Ok(Value::Vec(Rc::new(RefCell::new(Vec::new()))))
    });
    vm.add_native("tensor".into(), |_vm, args| {
        // tensor() constructor: when called as tensor[[...]], the bytecode
        // compiler handles TensorLiteral via Op::MakeTensor directly.
        // This native handles the rare case where tensor() is called as a function.
        if args.len() == 1 {
            Ok(args[0].clone())
        } else {
            Err(tenth::error::TenthError::RuntimeError { message: "tensor() 参数异常".into() })
        }
    });
    vm.add_native("write_bytes".into(), |vm, args| {
        if args.len() >= 2 {
            if let Value::String(path) = &args[1] {
                if let Value::Vec(items) = &args[0] {
                    // H-2: 沙箱校验
                    let resolved = if let Some(ref sb) = vm.fs_sandbox {
                        match sb.check_write(path) {
                            Ok(p) => p,
                            Err(e) => return Err(tenth::error::TenthError::RuntimeError { message: e }),
                        }
                    } else {
                        std::path::PathBuf::from(path)
                    };
                    let bytes: Vec<u8> = items.borrow().iter().map(|v| v.as_int().unwrap_or(0) as u8).collect();
                    let _ = std::fs::write(&resolved, &bytes);
                    return Ok(Value::Int(0));
                }
            }
        }
        Ok(Value::Int(1))
    });
    vm.add_native("read_bytes".into(), |vm, args| {
        if let Some(Value::String(path)) = args.first() {
            // H-2: 沙箱校验
            let resolved = if let Some(ref sb) = vm.fs_sandbox {
                match sb.check_read(path) {
                    Ok(p) => p,
                    Err(e) => return Err(tenth::error::TenthError::RuntimeError { message: e }),
                }
            } else {
                std::path::PathBuf::from(path)
            };
            match std::fs::read(&resolved) {
                Ok(data) => {
                    let bytes: Vec<Value> = data.iter()
                        .map(|b| Value::Int(*b as i64))
                        .collect();
                    Ok(Value::Vec(Rc::new(RefCell::new(bytes))))
                }
                Err(e) => Err(tenth::error::TenthError::RuntimeError {
                    message: format!("读取字节失败: {}", e),
                }),
            }
        } else {
            Err(tenth::error::TenthError::RuntimeError {
                message: "read_bytes(路径) 期望一个字符串路径".into(),
            })
        }
    });
    // Time functions
    vm.add_native("time_now".into(), |_vm, _args| {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs_f64();
        Ok(Value::Float(now))
    });
    vm.add_native("time_now_ms".into(), |_vm, _args| {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as f64;
        Ok(Value::Float(now))
    });
    vm.add_native("time_date".into(), |_vm, _args| {
        let secs = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        let days_since_epoch = secs / 86400;
        let (year, month, day) = datetime::days_to_date(days_since_epoch);
        Ok(Value::String(format!("{}-{:02}-{:02}", year, month, day)))
    });
    vm.add_native("time_time".into(), |_vm, _args| {
        let secs = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs() % 86400;
        let h = secs / 3600;
        let m = (secs % 3600) / 60;
        let s = secs % 60;
        Ok(Value::String(format!("{}:{:02}:{:02}", h, m, s)))
    });
    vm.add_native("time_datetime".into(), |_vm, _args| {
        let secs = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        let days_since_epoch = secs / 86400;
        let (year, month, day) = datetime::days_to_date(days_since_epoch);
        let day_secs = secs % 86400;
        let h = day_secs / 3600;
        let m = (day_secs % 3600) / 60;
        let s = day_secs % 60;
        Ok(Value::String(format!("{}-{:02}-{:02} {}:{:02}:{:02}", year, month, day, h, m, s)))
    });
    vm.add_native("time_sleep_ms".into(), |_vm, args| {
        if let Some(Value::Int(ms)) = args.first() {
            // 安全：拒绝负数（`as u64` 会符号扩展为巨大值，导致近乎永久的 DoS）
            // 上限 24 小时，防止 `.th` 程序意外将进程睡眠数年
            const MAX_SLEEP_MS: i64 = 24 * 60 * 60 * 1000;
            if *ms < 0 {
                return Err(tenth::error::TenthError::RuntimeError {
                    message: format!("time_sleep_ms: 不接受负数（{}）", ms),
                });
            }
            if *ms > MAX_SLEEP_MS {
                return Err(tenth::error::TenthError::RuntimeError {
                    message: format!("time_sleep_ms: 超过 24 小时上限（{}ms）", ms),
                });
            }
            std::thread::sleep(std::time::Duration::from_millis(*ms as u64));
            Ok(Value::Unit)
        } else {
            Err(tenth::error::TenthError::RuntimeError { message: "time_sleep_ms(ms) 期望一个整数".into() })
        }
    });
    // Random functions — 使用 rand crate 的 CSPRNG（thread_rng），避免可预测种子。
    // 历史 `DefaultHasher` + SystemTime 方案可被攻击者枚举纳秒时刻预测输出。
    vm.add_native("random_int".into(), |_vm, args| {
        let lo = match args.first() {
            Some(Value::Int(n)) => *n,
            _ => 0,
        };
        let hi = match args.get(1) {
            Some(Value::Int(n)) => *n,
            _ => lo,
        };
        use rand::Rng;
        // 处理 lo > hi 的边界：交换而不是 (hi - lo + 1) 为负时回绕
        let (low, high) = if lo <= hi { (lo, hi) } else { (hi, lo) };
        // 用 u64 全域取模，避免 i64 范围回绕到负数
        let range = (high as u64).saturating_sub(low as u64).saturating_add(1).max(1);
        let r: u64 = rand::thread_rng().r#gen();
        Ok(Value::Int(low + ((r % range) as i64)))
    });
    vm.add_native("random_float".into(), |_vm, _args| {
        use rand::Rng;
        // [0, 1) 半开区间，标准做法
        let r: f64 = rand::thread_rng().r#gen();
        Ok(Value::Float(r))
    });
    // Math functions（输入为 Float32 时返回 Float32，否则 Float）
    vm.add_native("math_tan".into(), |_vm, args| {
        match args.first() {
            Some(Value::Float(x)) => Ok(Value::Float(x.tan())),
            Some(Value::Float32(x)) => Ok(Value::Float32(x.tan())),
            _ => Ok(Value::Float(0.0))
        }
    });
    vm.add_native("math_asin".into(), |_vm, args| {
        match args.first() {
            Some(Value::Float(x)) => Ok(Value::Float(x.asin())),
            Some(Value::Float32(x)) => Ok(Value::Float32(x.asin())),
            _ => Ok(Value::Float(0.0))
        }
    });
    vm.add_native("math_acos".into(), |_vm, args| {
        match args.first() {
            Some(Value::Float(x)) => Ok(Value::Float(x.acos())),
            Some(Value::Float32(x)) => Ok(Value::Float32(x.acos())),
            _ => Ok(Value::Float(0.0))
        }
    });
    vm.add_native("math_atan".into(), |_vm, args| {
        match args.first() {
            Some(Value::Float(x)) => Ok(Value::Float(x.atan())),
            Some(Value::Float32(x)) => Ok(Value::Float32(x.atan())),
            _ => Ok(Value::Float(0.0))
        }
    });
    vm.add_native("math_atan2".into(), |_vm, args| {
        match (args.first(), args.get(1)) {
            (Some(Value::Float(y)), Some(Value::Float(x))) => Ok(Value::Float(y.atan2(*x))),
            (Some(Value::Float32(y)), Some(Value::Float32(x))) => Ok(Value::Float32(y.atan2(*x))),
            _ => Ok(Value::Float(0.0))
        }
    });
    vm.add_native("math_sinh".into(), |_vm, args| {
        match args.first() {
            Some(Value::Float(x)) => Ok(Value::Float(x.sinh())),
            Some(Value::Float32(x)) => Ok(Value::Float32(x.sinh())),
            _ => Ok(Value::Float(0.0))
        }
    });
    vm.add_native("math_cosh".into(), |_vm, args| {
        match args.first() {
            Some(Value::Float(x)) => Ok(Value::Float(x.cosh())),
            Some(Value::Float32(x)) => Ok(Value::Float32(x.cosh())),
            _ => Ok(Value::Float(0.0))
        }
    });
    vm.add_native("math_tanh".into(), |_vm, args| {
        match args.first() {
            Some(Value::Float(x)) => Ok(Value::Float(x.tanh())),
            Some(Value::Float32(x)) => Ok(Value::Float32(x.tanh())),
            _ => Ok(Value::Float(0.0))
        }
    });
    vm.add_native("math_log10".into(), |_vm, args| {
        match args.first() {
            Some(Value::Float(x)) => Ok(Value::Float(x.log10())),
            Some(Value::Float32(x)) => Ok(Value::Float32(x.log10())),
            _ => Ok(Value::Float(0.0))
        }
    });
    vm.add_native("math_log2".into(), |_vm, args| {
        match args.first() {
            Some(Value::Float(x)) => Ok(Value::Float(x.log2())),
            Some(Value::Float32(x)) => Ok(Value::Float32(x.log2())),
            _ => Ok(Value::Float(0.0))
        }
    });
    vm.add_native("math_exp".into(), |_vm, args| {
        match args.first() {
            Some(Value::Float(x)) => Ok(Value::Float(x.exp())),
            Some(Value::Float32(x)) => Ok(Value::Float32(x.exp())),
            _ => Ok(Value::Float(0.0))
        }
    });
    vm.add_native("math_pow".into(), |_vm, args| {
        match (args.first(), args.get(1)) {
            (Some(Value::Float(base)), Some(Value::Float(exp))) => Ok(Value::Float(base.powf(*exp))),
            (Some(Value::Float32(base)), Some(Value::Float32(exp))) => Ok(Value::Float32(base.powf(*exp))),
            _ => Ok(Value::Float(0.0))
        }
    });
    vm.add_native("math_floor".into(), |_vm, args| {
        match args.first() {
            Some(Value::Float(x)) => Ok(Value::Float(x.floor())),
            Some(Value::Float32(x)) => Ok(Value::Float32(x.floor())),
            _ => Ok(Value::Float(0.0))
        }
    });
    vm.add_native("math_ceil".into(), |_vm, args| {
        match args.first() {
            Some(Value::Float(x)) => Ok(Value::Float(x.ceil())),
            Some(Value::Float32(x)) => Ok(Value::Float32(x.ceil())),
            _ => Ok(Value::Float(0.0))
        }
    });
    vm.add_native("math_round".into(), |_vm, args| {
        match args.first() {
            Some(Value::Float(x)) => Ok(Value::Float(x.round())),
            Some(Value::Float32(x)) => Ok(Value::Float32(x.round())),
            _ => Ok(Value::Float(0.0))
        }
    });
    // CLI functions
    vm.add_native("cli_args_count".into(), |_vm, _args| {
        Ok(Value::Int(1))
    });
    vm.add_native("cli_arg".into(), |_vm, _args| {
        Ok(Value::String(String::new()))
    });
    // JSON functions
    vm.add_native("json_encode".into(), |_vm, args| {
        if let Some(val) = args.first() {
            Ok(Value::String(json::json_encode_value(val)))
        } else {
            Ok(Value::String("null".into()))
        }
    });
    vm.add_native("json_encode_pretty".into(), |_vm, args| {
        if let Some(val) = args.first() {
            Ok(Value::String(json::json_encode_value_pretty(val, 0)))
        } else {
            Ok(Value::String("null".into()))
        }
    });
    vm.add_native("json_decode".into(), |_vm, args| {
        if let Some(Value::String(s)) = args.first() {
            Ok(json::json_decode_string(s))
        } else {
            Ok(Value::Unit)
        }
    });
    vm.add_native("compile_host".into(), |vm, args| {
        if args.len() >= 2 {
            if let (Value::String(src), Value::String(out)) = (&args[0], &args[1]) {
                // H-2/L-7: 沙箱校验写路径
                let out_resolved = if let Some(ref sb) = vm.fs_sandbox {
                    match sb.check_write(out) {
                        Ok(p) => p,
                        Err(e) => return Err(tenth::error::TenthError::RuntimeError { message: e }),
                    }
                } else {
                    std::path::PathBuf::from(out)
                };
                match tenth::lexer::lexer::Lexer::new(src).tokenize()
                    .and_then(|tokens| tenth::parser::parser::Parser::new(tokens).parse_program())
                    .and_then(|prog| tenth::hir::lower::Lowerer::new().lower_program(&prog))
                    .and_then(|hir| tenth::compile::compile_to_wasm(&hir))
                {
                    Ok(bytes) => { let _ = std::fs::write(&out_resolved, &bytes); return Ok(Value::Int(0)); }
                    Err(_) => return Ok(Value::Int(1)),
                }
            }
        }
        Ok(Value::Int(1))
    });
    vm.add_native("compile_program".into(), |vm, args| {
        if args.len() >= 2 {
            if let Value::String(out) = &args[1] {
                // H-2/L-7: 沙箱校验写路径
                let out_resolved = if let Some(ref sb) = vm.fs_sandbox {
                    match sb.check_write(out) {
                        Ok(p) => p,
                        Err(e) => return Err(tenth::error::TenthError::RuntimeError { message: e }),
                    }
                } else {
                    std::path::PathBuf::from(out)
                };
                match tenth::compile::compile_program_to_wasm(&args[0]) {
                    Ok(bytes) => { let _ = std::fs::write(&out_resolved, &bytes); return Ok(Value::Int(0)); }
                    Err(_) => return Ok(Value::Int(1)),
                }
            }
        }
        Ok(Value::Int(1))
    });

    // ── Autodiff native functions ──
    vm.add_native("new_grad".into(), |vm, _args| {
        vm.tape = Some(Tape::new());
        vm.recording = true;
        Ok(Value::Unit)
    });
    vm.add_native("param".into(), |vm, args| {
        if let Some(Value::Tensor(t)) = args.first() {
            if let Some(ref mut tape) = vm.tape {
                let node_id = tape.input(t.clone());
                t.borrow_mut().tape_id = Some(node_id);
            }
            Ok(Value::Tensor(t.clone()))
        } else {
            Err(tenth::error::TenthError::RuntimeError { message: "param() 需要一个张量参数".into() })
        }
    });
    vm.add_native("backward".into(), |vm, args| {
        if let Some(Value::Tensor(t)) = args.first() {
            if let Some(ref tape) = vm.tape {
                let loss_id = t.borrow().tape_id
                    .ok_or_else(|| tenth::error::TenthError::RuntimeError { message: "backward(): 张量没有 tape_id".into() })?;
                // 护城河 F：包裹 backward 错误，附加 formal_explain 根因分析
                match tape.backward(loss_id) {
                    Ok(()) => Ok(Value::Unit),
                    Err(e) => {
                        // 计算 formal_explain 根因候选
                        let causes = tape.formal_explain(loss_id, &[], &[]);
                        let explanations: Vec<String> = causes.iter().map(|c| c.explanation.clone()).collect();
                        // 存到 vm.last_explanation，供 explain_error() native 读取
                        vm.last_explanation = explanations.clone();
                        // 构造 ShapeMismatch 错误（携带 tape 上下文 + 根因消息）
                        let context = tenth::error::TapeErrorContext {
                            tape_node_id: loss_id,
                            op: "backward".to_string(),
                            expected_shape: Vec::new(),
                            actual_shape: Vec::new(),
                        };
                        let root_cause_msg = if explanations.is_empty() {
                            format!("{}", e)
                        } else {
                            format!(
                                "{}\n根因分析（formal_explain）：\n{}",
                                e,
                                explanations.iter()
                                    .map(|s| format!("  - {}", s))
                                    .collect::<Vec<_>>()
                                    .join("\n")
                            )
                        };
                        Err(tenth::error::TenthError::ShapeMismatch {
                            context,
                            message: root_cause_msg,
                        })
                    }
                }
            } else {
                Err(tenth::error::TenthError::RuntimeError { message: "未调用 new_grad()".into() })
            }
        } else {
            Err(tenth::error::TenthError::RuntimeError { message: "backward() 需要一个张量参数".into() })
        }
    });
    // 护城河 F：explain_error() — 返回上一次 backward 失败的根因说明列表
    // 用户在 try-catch backward 错误后调用此 native 获取详细分析。
    vm.add_native("explain_error".into(), |vm, _args| {
        let explanations = std::mem::take(&mut vm.last_explanation);
        let values: Vec<Value> = explanations.into_iter().map(Value::String).collect();
        Ok(Value::Vec(Rc::new(RefCell::new(values))))
    });
    vm.add_native("grad".into(), |_vm, args| {
        if let Some(Value::Tensor(t)) = args.first() {
            let p = t.borrow();
            if let Some(ref grad) = p.grad {
                let grad_tensor = Tensor::from_tensor_data(grad.clone());
                Ok(Value::Tensor(Rc::new(RefCell::new(grad_tensor))))
            } else {
                // 按参数 dtype 返回零张量
                let zeros = if p.is_f32() {
                    Tensor::zeros_f32(&p.shape())
                } else {
                    Tensor::zeros(&p.shape())
                };
                Ok(Value::Tensor(Rc::new(RefCell::new(zeros))))
            }
        } else {
            Err(tenth::error::TenthError::RuntimeError { message: "grad() 需要一个张量参数".into() })
        }
    });
    vm.add_native("stop_grad".into(), |vm, args| {
        if let Some(Value::Tensor(t)) = args.first() {
            let mut detached = t.borrow().clone();
            detached.tape_id = None;
            Ok(Value::Tensor(Rc::new(RefCell::new(detached))))
        } else {
            // No-arg form: stop gradient recording
            vm.recording = false;
            Ok(Value::Unit)
        }
    });
    vm.add_native("zero_grad".into(), |vm, _args| {
        if let Some(ref tape) = vm.tape {
            tape.zero_grad();
        }
        Ok(Value::Unit)
    });
    vm.add_native("select".into(), |vm, args| {
        // select(cond, then, else) — 逐元素条件选择原语（论文 T47/T48/T50）
        // 支持广播；cond 非 0 视为 true。可微：d_then = grad*mask, d_else = grad*(1-mask)
        if args.len() < 3 {
            return Err(tenth::error::TenthError::RuntimeError {
                message: "select(cond, then, else) 期望三个参数".into(),
            });
        }
        let (cond, then, else_) = match (&args[0], &args[1], &args[2]) {
            (Value::Tensor(c), Value::Tensor(t), Value::Tensor(e)) => (c.clone(), t.clone(), e.clone()),
            _ => return Err(tenth::error::TenthError::RuntimeError {
                message: "select(cond, then, else) 期望三个张量参数".into(),
            }),
        };
        let result_tensor = Tensor::select(&cond.borrow(), &then.borrow(), &else_.borrow())
            .map_err(|msg| tenth::error::TenthError::RuntimeError { message: msg })?;
        let result = Rc::new(RefCell::new(result_tensor));
        if vm.recording {
            if let Some(ref mut tape) = vm.tape {
                let then_id = then.borrow().tape_id;
                let else_id = else_.borrow().tape_id;
                let node_id = tape.select(then_id, else_id, cond.clone(), then.clone(), else_.clone(), result.clone());
                result.borrow_mut().tape_id = Some(node_id);
            }
        }
        Ok(Value::Tensor(result))
    });
    vm.add_native("cross_entropy".into(), |vm, args| {
        if args.len() < 2 {
            return Err(tenth::error::TenthError::RuntimeError { message: "cross_entropy(logits, target) 期望两个张量".into() });
        }
        if let (Value::Tensor(logits), Value::Tensor(target)) = (&args[0], &args[1]) {
            let logits_data = logits.borrow();
            let target_data = target.borrow();
            let is_f32 = logits_data.is_f32();
            let sm = logits_data.softmax().ok_or_else(|| {
                tenth::error::TenthError::RuntimeError { message: "cross_entropy 中 softmax 失败".into() }
            })?;
            let eps = 1e-10;
            let sm_data = sm.data.as_standard_layout().to_owned();
            let tgt_flat = target_data.data.as_standard_layout().to_owned();
            let sm_slice = sm_data.as_slice().unwrap_or(&[]);
            let tgt_slice = tgt_flat.as_slice().unwrap_or(&[]);
            let n = sm_slice.len() as f64;
            let mut loss_val = 0.0f64;
            for i in 0..sm_slice.len().min(tgt_slice.len()) {
                let p = sm_slice[i].max(eps);
                loss_val -= tgt_slice[i] * p.ln();
            }
            loss_val /= n.max(1.0);
            // 按 logits dtype 构造对应 loss tensor
            let loss_tensor = if is_f32 {
                Tensor::from_vec_f32(vec![loss_val as f32], vec![1])
            } else {
                Tensor::from_vec(vec![loss_val], vec![1])
            };
            let result = Rc::new(RefCell::new(loss_tensor));
            if vm.recording {
                let sm_rc = Rc::new(RefCell::new(sm));
                if let Some(ref mut tape) = vm.tape {
                    let logits_id = logits.borrow().tape_id
                        .unwrap_or_else(|| tape.input(logits.clone()));
                    let _sm_id = tape.input(sm_rc.clone());
                    let node_id = tape.cross_entropy(
                        logits_id, logits.clone(),
                        sm_rc,
                        target.clone(),
                        result.clone(),
                    );
                    result.borrow_mut().tape_id = Some(node_id);
                }
            }
            Ok(Value::Tensor(result))
        } else {
            Err(tenth::error::TenthError::RuntimeError { message: "cross_entropy(logits, target) 期望两个张量".into() })
        }
    });
    // File system functions
    vm.add_native("write_file".into(), |vm, args| {
        if args.len() >= 2 {
            if let (Value::String(path), Value::String(content)) = (&args[0], &args[1]) {
                // H-2: 沙箱校验
                let resolved = if let Some(ref sb) = vm.fs_sandbox {
                    match sb.check_write(path) {
                        Ok(p) => p,
                        Err(e) => return Err(tenth::error::TenthError::RuntimeError { message: e }),
                    }
                } else {
                    std::path::PathBuf::from(path)
                };
                match std::fs::write(&resolved, content) {
                    Ok(()) => Ok(Value::Unit),
                    Err(e) => Err(tenth::error::TenthError::RuntimeError { message: format!("写入文件失败: {}", e) }),
                }
            } else {
                Err(tenth::error::TenthError::RuntimeError { message: "write_file(路径, 内容) 期望两个字符串参数".into() })
            }
        } else {
            Err(tenth::error::TenthError::RuntimeError { message: "write_file(路径, 内容) 期望两个字符串参数".into() })
        }
    });
    vm.add_native("path_join".into(), |_vm, args| {
        if args.len() >= 2 {
            if let (Value::String(base), Value::String(rest)) = (&args[0], &args[1]) {
                let joined = std::path::Path::new(base).join(rest);
                Ok(Value::String(joined.to_string_lossy().to_string()))
            } else {
                Err(tenth::error::TenthError::RuntimeError { message: "path_join(基础路径, 子路径) 期望两个字符串参数".into() })
            }
        } else {
            Err(tenth::error::TenthError::RuntimeError { message: "path_join(基础路径, 子路径) 期望两个字符串参数".into() })
        }
    });
    vm.add_native("path_exists".into(), |vm, args| {
        if let Some(Value::String(path)) = args.first() {
            // H-2: 沙箱校验
            if let Some(ref sb) = vm.fs_sandbox {
                if let Err(e) = sb.check_read(path) {
                    return Err(tenth::error::TenthError::RuntimeError { message: e });
                }
            }
            Ok(Value::Bool(std::path::Path::new(path).exists()))
        } else {
            Err(tenth::error::TenthError::RuntimeError { message: "path_exists(路径) 期望一个字符串路径".into() })
        }
    });
    vm.add_native("path_is_file".into(), |vm, args| {
        if let Some(Value::String(path)) = args.first() {
            if let Some(ref sb) = vm.fs_sandbox {
                if let Err(e) = sb.check_read(path) {
                    return Err(tenth::error::TenthError::RuntimeError { message: e });
                }
            }
            Ok(Value::Bool(std::path::Path::new(path).is_file()))
        } else {
            Err(tenth::error::TenthError::RuntimeError { message: "path_is_file(路径) 期望一个字符串路径".into() })
        }
    });
    vm.add_native("path_is_dir".into(), |vm, args| {
        if let Some(Value::String(path)) = args.first() {
            if let Some(ref sb) = vm.fs_sandbox {
                if let Err(e) = sb.check_read(path) {
                    return Err(tenth::error::TenthError::RuntimeError { message: e });
                }
            }
            Ok(Value::Bool(std::path::Path::new(path).is_dir()))
        } else {
            Err(tenth::error::TenthError::RuntimeError { message: "path_is_dir(路径) 期望一个字符串路径".into() })
        }
    });
    vm.add_native("mkdir".into(), |vm, args| {
        if let Some(Value::String(path)) = args.first() {
            // H-2: 沙箱校验
            let resolved = if let Some(ref sb) = vm.fs_sandbox {
                match sb.check_write(path) {
                    Ok(p) => p,
                    Err(e) => return Err(tenth::error::TenthError::RuntimeError { message: e }),
                }
            } else {
                std::path::PathBuf::from(path)
            };
            match std::fs::create_dir_all(&resolved) {
                Ok(()) => Ok(Value::Unit),
                Err(e) => Err(tenth::error::TenthError::RuntimeError { message: format!("创建目录失败: {}", e) }),
            }
        } else {
            Err(tenth::error::TenthError::RuntimeError { message: "mkdir(路径) 期望一个字符串路径".into() })
        }
    });
    vm.add_native("list_dir".into(), |vm, args| {
        if let Some(Value::String(path)) = args.first() {
            // H-2: 沙箱校验
            let resolved = if let Some(ref sb) = vm.fs_sandbox {
                match sb.check_read(path) {
                    Ok(p) => p,
                    Err(e) => return Err(tenth::error::TenthError::RuntimeError { message: e }),
                }
            } else {
                std::path::PathBuf::from(path)
            };
            match std::fs::read_dir(&resolved) {
                Ok(entries) => {
                    let items: Vec<Value> = entries
                        .filter_map(|e| e.ok())
                        .map(|e| Value::String(e.file_name().to_string_lossy().to_string()))
                        .collect();
                    Ok(Value::Vec(Rc::new(RefCell::new(items))))
                }
                Err(e) => Err(tenth::error::TenthError::RuntimeError { message: format!("列出目录失败: {}", e) }),
            }
        } else {
            Err(tenth::error::TenthError::RuntimeError { message: "list_dir(路径) 期望一个字符串路径".into() })
        }
    });
    vm.add_native("file_size".into(), |vm, args| {
        if let Some(Value::String(path)) = args.first() {
            // H-2: 沙箱校验
            let resolved = if let Some(ref sb) = vm.fs_sandbox {
                match sb.check_read(path) {
                    Ok(p) => p,
                    Err(e) => return Err(tenth::error::TenthError::RuntimeError { message: e }),
                }
            } else {
                std::path::PathBuf::from(path)
            };
            match std::fs::metadata(&resolved) {
                Ok(meta) => Ok(Value::Int(meta.len() as i64)),
                Err(e) => Err(tenth::error::TenthError::RuntimeError { message: format!("获取文件大小失败: {}", e) }),
            }
        } else {
            Err(tenth::error::TenthError::RuntimeError { message: "file_size(路径) 期望一个字符串路径".into() })
        }
    });
    vm.add_native("remove_file".into(), |vm, args| {
        if let Some(Value::String(path)) = args.first() {
            // H-2: 沙箱校验
            let resolved = if let Some(ref sb) = vm.fs_sandbox {
                match sb.check_write(path) {
                    Ok(p) => p,
                    Err(e) => return Err(tenth::error::TenthError::RuntimeError { message: e }),
                }
            } else {
                std::path::PathBuf::from(path)
            };
            match std::fs::remove_file(&resolved) {
                Ok(()) => Ok(Value::Unit),
                Err(e) => Err(tenth::error::TenthError::RuntimeError { message: format!("删除文件失败: {}", e) }),
            }
        } else {
            Err(tenth::error::TenthError::RuntimeError { message: "remove_file(路径) 期望一个字符串路径".into() })
        }
    });
    vm.add_native("copy_file".into(), |vm, args| {
        if args.len() >= 2 {
            if let (Value::String(src), Value::String(dst)) = (&args[0], &args[1]) {
                // H-2: 沙箱校验
                let src_resolved = if let Some(ref sb) = vm.fs_sandbox {
                    match sb.check_read(src) {
                        Ok(p) => p,
                        Err(e) => return Err(tenth::error::TenthError::RuntimeError { message: e }),
                    }
                } else {
                    std::path::PathBuf::from(src)
                };
                let dst_resolved = if let Some(ref sb) = vm.fs_sandbox {
                    match sb.check_write(dst) {
                        Ok(p) => p,
                        Err(e) => return Err(tenth::error::TenthError::RuntimeError { message: e }),
                    }
                } else {
                    std::path::PathBuf::from(dst)
                };
                match std::fs::copy(&src_resolved, &dst_resolved) {
                    Ok(_) => Ok(Value::Unit),
                    Err(e) => Err(tenth::error::TenthError::RuntimeError { message: format!("复制文件失败: {}", e) }),
                }
            } else {
                Err(tenth::error::TenthError::RuntimeError { message: "copy_file(源路径, 目标路径) 期望两个字符串参数".into() })
            }
        } else {
            Err(tenth::error::TenthError::RuntimeError { message: "copy_file(源路径, 目标路径) 期望两个字符串参数".into() })
        }
    });
    vm.add_native("rename_file".into(), |vm, args| {
        if args.len() >= 2 {
            if let (Value::String(src), Value::String(dst)) = (&args[0], &args[1]) {
                // H-2: 沙箱校验
                let src_resolved = if let Some(ref sb) = vm.fs_sandbox {
                    match sb.check_read(src) {
                        Ok(p) => p,
                        Err(e) => return Err(tenth::error::TenthError::RuntimeError { message: e }),
                    }
                } else {
                    std::path::PathBuf::from(src)
                };
                let dst_resolved = if let Some(ref sb) = vm.fs_sandbox {
                    match sb.check_write(dst) {
                        Ok(p) => p,
                        Err(e) => return Err(tenth::error::TenthError::RuntimeError { message: e }),
                    }
                } else {
                    std::path::PathBuf::from(dst)
                };
                match std::fs::rename(&src_resolved, &dst_resolved) {
                    Ok(()) => Ok(Value::Unit),
                    Err(e) => Err(tenth::error::TenthError::RuntimeError { message: format!("重命名文件失败: {}", e) }),
                }
            } else {
                Err(tenth::error::TenthError::RuntimeError { message: "rename_file(源路径, 目标路径) 期望两个字符串参数".into() })
            }
        } else {
            Err(tenth::error::TenthError::RuntimeError { message: "rename_file(源路径, 目标路径) 期望两个字符串参数".into() })
        }
    });
    vm.add_native("randn".into(), |_vm, args| {
        let rows = match args.first() { Some(Value::Int(n)) => *n as usize, _ => 1 };
        let cols = match args.get(1) { Some(Value::Int(n)) => *n as usize, _ => 1 };
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let data: Vec<f64> = (0..rows * cols).map(|_| {
            // Box-Muller transform for normal distribution
            let u1: f64 = rng.r#gen::<f64>().max(1e-10);
            let u2: f64 = rng.r#gen::<f64>();
            (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
        }).collect();
        Ok(Value::Tensor(Rc::new(RefCell::new(Tensor::from_vec(data, vec![rows, cols])))))
    });
    vm.add_native("randn_f32".into(), |_vm, args| {
        let rows = match args.first() { Some(Value::Int(n)) => *n as usize, _ => 1 };
        let cols = match args.get(1) { Some(Value::Int(n)) => *n as usize, _ => 1 };
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let data: Vec<f32> = (0..rows * cols).map(|_| {
            // Box-Muller transform for normal distribution (f32 版本)
            let u1: f32 = rng.r#gen::<f32>().max(1e-10);
            let u2: f32 = rng.r#gen::<f32>();
            (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
        }).collect();
        Ok(Value::Tensor(Rc::new(RefCell::new(Tensor::from_vec_f32(data, vec![rows, cols])))))
    });
    // ── Tensor 构造函数（与 interpreter::natives 对齐，支持任意 shape）──
    // 历史：这些函数仅在 interpreter 实现，JIT/VM 路径下返回 Unit。
    // 补齐后 zeros(256,256,256).numel() 等才能在默认 tenth run 路径下正常工作。
    vm.add_native("zeros".into(), |_vm, args| {
        let shape: Vec<usize> = args.iter()
            .map(|a| a.as_int().unwrap_or(1) as usize)
            .collect();
        Ok(Value::Tensor(Rc::new(RefCell::new(Tensor::zeros(&shape)))))
    });
    vm.add_native("ones".into(), |_vm, args| {
        let shape: Vec<usize> = args.iter()
            .map(|a| a.as_int().unwrap_or(1) as usize)
            .collect();
        Ok(Value::Tensor(Rc::new(RefCell::new(Tensor::ones(&shape)))))
    });
    vm.add_native("rand".into(), |_vm, args| {
        let shape: Vec<usize> = args.iter()
            .map(|a| a.as_int().unwrap_or(1) as usize)
            .collect();
        Ok(Value::Tensor(Rc::new(RefCell::new(Tensor::rand(&shape)))))
    });
    vm.add_native("zeros_f32".into(), |_vm, args| {
        let shape: Vec<usize> = args.iter()
            .map(|a| a.as_int().unwrap_or(1) as usize)
            .collect();
        Ok(Value::Tensor(Rc::new(RefCell::new(Tensor::zeros_f32(&shape)))))
    });
    vm.add_native("ones_f32".into(), |_vm, args| {
        let shape: Vec<usize> = args.iter()
            .map(|a| a.as_int().unwrap_or(1) as usize)
            .collect();
        Ok(Value::Tensor(Rc::new(RefCell::new(Tensor::ones_f32(&shape)))))
    });
    vm.add_native("rand_f32".into(), |_vm, args| {
        let shape: Vec<usize> = args.iter()
            .map(|a| a.as_int().unwrap_or(1) as usize)
            .collect();
        Ok(Value::Tensor(Rc::new(RefCell::new(Tensor::rand_f32(&shape)))))
    });
    vm.add_native("HashMap::new".into(), |_vm, _args| {
        Ok(Value::Map(Rc::new(RefCell::new(std::collections::HashMap::new()))))
    });
    vm.add_native("print".into(), |_vm, args| {
        for a in args { print!("{a}"); }
        Ok(Value::Unit)
    });
    vm.add_native("abs".into(), |_vm, args| {
        match args.first() {
            Some(Value::Int(n)) => Ok(Value::Int(n.abs())),
            Some(Value::Float(f)) => Ok(Value::Float(f.abs())),
            Some(Value::Float32(f)) => Ok(Value::Float32(f.abs())),
            _ => Err(tenth::error::TenthError::RuntimeError { message: "abs() 需要一个数值参数".into() }),
        }
    });
    vm.add_native("sqrt".into(), |_vm, args| {
        match args.first() {
            Some(Value::Float(f)) => Ok(Value::Float(f.sqrt())),
            Some(Value::Float32(f)) => Ok(Value::Float32(f.sqrt())),
            Some(Value::Int(n)) => Ok(Value::Float((*n as f64).sqrt())),
            _ => Err(tenth::error::TenthError::RuntimeError { message: "sqrt() 需要一个数值参数".into() }),
        }
    });
    vm.add_native("to_float".into(), |_vm, args| {
        match args.first() {
            Some(Value::Int(n)) => Ok(Value::Float(*n as f64)),
            Some(Value::Float(f)) => Ok(Value::Float(*f)),
            Some(Value::Float32(f)) => Ok(Value::Float(*f as f64)),
            _ => Err(tenth::error::TenthError::RuntimeError { message: "to_float() 需要一个数值参数".into() }),
        }
    });
    vm.add_native("to_f64".into(), |_vm, args| {
        match args.first() {
            Some(Value::Int(n)) => Ok(Value::Float(*n as f64)),
            Some(Value::Float(f)) => Ok(Value::Float(*f)),
            Some(Value::Float32(f)) => Ok(Value::Float(*f as f64)),
            _ => Err(tenth::error::TenthError::RuntimeError { message: "to_f64() 需要一个数值参数".into() }),
        }
    });
    vm.add_native("to_f32".into(), |_vm, args| {
        match args.first() {
            Some(Value::Int(n)) => Ok(Value::Float32(*n as f32)),
            Some(Value::Float(f)) => Ok(Value::Float32(*f as f32)),
            Some(Value::Float32(f)) => Ok(Value::Float32(*f)),
            _ => Err(tenth::error::TenthError::RuntimeError { message: "to_f32() 需要一个数值参数".into() }),
        }
    });
    vm.add_native("tensor_from_vec".into(), |_vm, args| {
        if args.len() >= 3 {
            if let (Value::Vec(items), Value::Int(rows), Value::Int(cols)) = (&args[0], &args[1], &args[2]) {
                // 按 Vec 内元素 dtype 判断：含 Float32 → f32 Tensor
                let has_f32 = items.borrow().iter().any(|v| matches!(v, Value::Float32(_)));
                if has_f32 {
                    let data: Vec<f32> = items.borrow().iter().map(|v| v.as_f32().unwrap_or(0.0)).collect();
                    let tensor = Tensor::from_vec_f32(data, vec![*rows as usize, *cols as usize]);
                    Ok(Value::Tensor(Rc::new(RefCell::new(tensor))))
                } else {
                    let data: Vec<f64> = items.borrow().iter().map(|v| v.as_float().unwrap_or(0.0)).collect();
                    let tensor = Tensor::from_vec(data, vec![*rows as usize, *cols as usize]);
                    Ok(Value::Tensor(Rc::new(RefCell::new(tensor))))
                }
            } else {
                Err(tenth::error::TenthError::RuntimeError { message: "tensor_from_vec(vec, rows, cols) 期望一个 Vec 和两个整数".into() })
            }
        } else {
            Err(tenth::error::TenthError::RuntimeError { message: "tensor_from_vec(vec, rows, cols) 期望 3 个参数".into() })
        }
    });

    // ── 论文 T37 修复第二批：补齐 VM 缺失的 17 项 native（与 interpreter::natives 对齐）──
    // 历史：这些 native 仅在解释器实现，VM 路径下返回 Unit（DX/ML 训练关键路径断裂）。

    // 1. to_string — 值转字符串（与解释器 value_to_string 对齐到 Display 实现）
    vm.add_native("to_string".into(), |_vm, args| {
        if let Some(arg) = args.first() {
            Ok(Value::String(format!("{}", arg)))
        } else {
            Ok(Value::String(String::new()))
        }
    });
    // 2. type_name — 值类型名
    vm.add_native("type_name".into(), |_vm, args| {
        if let Some(arg) = args.first() {
            let tn = match arg {
                Value::Int(_) => "int",
                Value::Float(_) => "float",
                Value::Float32(_) => "float",
                Value::Bool(_) => "bool",
                Value::String(_) => "string",
                Value::Unit => "unit",
                Value::Vec(_) => "vec",
                Value::Array(_) => "array",
                Value::Map(_) => "map",
                Value::Tuple(_) => "tuple",
                Value::Closure { .. } => "closure",
                Value::FnRef { .. } => "fn",
                _ => "unknown",
            };
            Ok(Value::String(tn.to_string()))
        } else {
            Ok(Value::String("unknown".to_string()))
        }
    });
    // 3. with_step_limit(limit, fn) — 步数预算内执行闭包
    //    VM 中闭包以 Value::FnRef 表示（Op::MakeClosure 创建），可通过 call_with_args 调用。
    vm.add_native("with_step_limit".into(), |vm, args| {
        if args.len() < 2 {
            return Err(tenth::error::TenthError::RuntimeError {
                message: "with_step_limit(limit, fn) 需要 2 个参数".into(),
            });
        }
        let limit = args[0].as_int().ok_or_else(|| tenth::error::TenthError::RuntimeError {
            message: "with_step_limit 的第一个参数必须是整数步数".into(),
        })?;
        let saved_budget = vm.step_budget;
        let saved_deadline = vm.deadline_ms;
        vm.step_budget = Some(limit.max(0) as u64);
        vm.deadline_ms = None;
        let result = match &args[1] {
            Value::FnRef { name, .. } => vm.call_with_args(name, &[]),
            // VM 无法在 native 内执行 tree-walk 闭包；与解释器 Timeout 语义一致返回 Unit。
            _ => {
                vm.step_budget = saved_budget;
                vm.deadline_ms = saved_deadline;
                return Ok(Value::Unit);
            }
        };
        vm.step_budget = saved_budget;
        vm.deadline_ms = saved_deadline;
        match result {
            Ok(v) => Ok(v),
            Err(tenth::error::TenthError::Timeout { .. }) => Ok(Value::Unit),
            Err(e) => Err(e),
        }
    });
    // 4. with_timeout_ms(ms, fn) — 毫秒截止期内执行闭包
    vm.add_native("with_timeout_ms".into(), |vm, args| {
        if args.len() < 2 {
            return Err(tenth::error::TenthError::RuntimeError {
                message: "with_timeout_ms(ms, fn) 需要 2 个参数".into(),
            });
        }
        let ms = args[0].as_int().ok_or_else(|| tenth::error::TenthError::RuntimeError {
            message: "with_timeout_ms 的第一个参数必须是整数毫秒".into(),
        })?;
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis())
            .unwrap_or(0);
        let saved_budget = vm.step_budget;
        let saved_deadline = vm.deadline_ms;
        // 与解释器一致：用大步数预算作为 tick 载体，deadline 做实际时间比较。
        vm.step_budget = Some(u64::MAX);
        vm.deadline_ms = Some(now + (ms.max(0) as u128));
        let result = match &args[1] {
            Value::FnRef { name, .. } => vm.call_with_args(name, &[]),
            _ => {
                vm.step_budget = saved_budget;
                vm.deadline_ms = saved_deadline;
                return Ok(Value::Unit);
            }
        };
        vm.step_budget = saved_budget;
        vm.deadline_ms = saved_deadline;
        match result {
            Ok(v) => Ok(v),
            Err(tenth::error::TenthError::Timeout { .. }) => Ok(Value::Unit),
            Err(e) => Err(e),
        }
    });
    // 5. is_timeout(result) — 判断是否超时哨兵（Unit）
    vm.add_native("is_timeout".into(), |_vm, args| {
        if let Some(arg) = args.first() {
            Ok(Value::Bool(matches!(arg, Value::Unit)))
        } else {
            Ok(Value::Bool(false))
        }
    });
    // 6. start_grad — 新建 Tape（与 new_grad 同义）
    vm.add_native("start_grad".into(), |vm, _args| {
        vm.tape = Some(Tape::new());
        vm.recording = true;
        Ok(Value::Unit)
    });
    // 7. f64_bits — f64 → i64 位表示
    vm.add_native("f64_bits".into(), |_vm, args| {
        if let Some(arg) = args.first() {
            let f = arg.as_float().ok_or_else(|| tenth::error::TenthError::RuntimeError {
                message: "f64_bits() 期望一个 f64 参数".into(),
            })?;
            Ok(Value::Int(f.to_bits() as i64))
        } else {
            Err(tenth::error::TenthError::RuntimeError { message: "f64_bits() 期望 1 个参数".into() })
        }
    });
    // 8. f64_from_bits — i64 → f64
    vm.add_native("f64_from_bits".into(), |_vm, args| {
        if let Some(arg) = args.first() {
            let n = arg.as_int().ok_or_else(|| tenth::error::TenthError::RuntimeError {
                message: "f64_from_bits() 期望一个 i64 参数".into(),
            })?;
            Ok(Value::Float(f64::from_bits(n as u64)))
        } else {
            Err(tenth::error::TenthError::RuntimeError { message: "f64_from_bits() 期望 1 个参数".into() })
        }
    });
    // 9-12. 标量数学（sin/cos/ln/pow）— 与解释器一致，仅操作 Float（as_float 自动提升 Int/Float32）
    vm.add_native("sin".into(), |_vm, args| {
        if let Some(arg) = args.first() {
            let n = arg.as_float().ok_or_else(|| tenth::error::TenthError::RuntimeError {
                message: "sin() 期望一个数值参数".into(),
            })?;
            Ok(Value::Float(n.sin()))
        } else {
            Err(tenth::error::TenthError::RuntimeError { message: "sin() 期望 1 个参数".into() })
        }
    });
    vm.add_native("cos".into(), |_vm, args| {
        if let Some(arg) = args.first() {
            let n = arg.as_float().ok_or_else(|| tenth::error::TenthError::RuntimeError {
                message: "cos() 期望一个数值参数".into(),
            })?;
            Ok(Value::Float(n.cos()))
        } else {
            Err(tenth::error::TenthError::RuntimeError { message: "cos() 期望 1 个参数".into() })
        }
    });
    vm.add_native("ln".into(), |_vm, args| {
        if let Some(arg) = args.first() {
            let n = arg.as_float().ok_or_else(|| tenth::error::TenthError::RuntimeError {
                message: "ln() 期望一个数值参数".into(),
            })?;
            if n <= 0.0 {
                return Err(tenth::error::TenthError::RuntimeError {
                    message: "ln() 参数必须 > 0".into(),
                });
            }
            Ok(Value::Float(n.ln()))
        } else {
            Err(tenth::error::TenthError::RuntimeError { message: "ln() 期望 1 个参数".into() })
        }
    });
    vm.add_native("pow".into(), |_vm, args| {
        if args.len() >= 2 {
            let base = args[0].as_float().ok_or_else(|| tenth::error::TenthError::RuntimeError {
                message: "pow() 期望数值参数".into(),
            })?;
            let exp = args[1].as_float().ok_or_else(|| tenth::error::TenthError::RuntimeError {
                message: "pow() 期望数值参数".into(),
            })?;
            Ok(Value::Float(base.powf(exp)))
        } else {
            Err(tenth::error::TenthError::RuntimeError { message: "pow() 期望 2 个参数".into() })
        }
    });
    // 13. save_weights(path, tensors) — 张量列表序列化到二进制文件（ML 训练关键路径）
    //     二进制格式与解释器完全一致：i32 num_tensors, [i32 ndim, i32×ndim shape, f64×nel data]（LE）
    vm.add_native("save_weights".into(), |vm, args| {
        if args.len() >= 2 {
            if let Value::String(path) = &args[0] {
                // H-2: 沙箱校验
                let resolved = if let Some(ref sb) = vm.fs_sandbox {
                    match sb.check_write(path) {
                        Ok(p) => p,
                        Err(e) => return Err(tenth::error::TenthError::RuntimeError { message: e }),
                    }
                } else {
                    std::path::PathBuf::from(path)
                };
                let tensors: &Rc<RefCell<Vec<Value>>> = match &args[1] {
                    Value::Vec(v) => v,
                    Value::Array(a) => a,
                    _ => return Err(tenth::error::TenthError::RuntimeError {
                        message: "save_weights 期望一个张量列表".into(),
                    }),
                };
                let tensors_ref = tensors.borrow();
                let mut bytes: Vec<u8> = Vec::new();
                bytes.extend(&(tensors_ref.len() as i32).to_le_bytes());
                for val in tensors_ref.iter() {
                    // 解包 Shared 包装（Vec::push 会将元素包装在 Shared 中）
                    let tensor_rc = match val {
                        Value::Tensor(t) => Some(t.clone()),
                        Value::Shared(rc) => {
                            if let Value::Tensor(t) = &*rc.borrow() {
                                Some(t.clone())
                            } else { None }
                        }
                        _ => None,
                    };
                    if let Some(t) = tensor_rc {
                        let t_ref = t.borrow();
                        let shape = t_ref.shape();
                        let ndim = shape.len() as i32;
                        bytes.extend(&ndim.to_le_bytes());
                        for &d in &shape {
                            bytes.extend(&(d as i32).to_le_bytes());
                        }
                        let flat = t_ref.data.as_standard_layout().to_owned();
                        if let Some(slice) = flat.as_slice() {
                            for &x in slice {
                                bytes.extend(&x.to_le_bytes());
                            }
                        }
                    }
                }
                let _ = std::fs::write(&resolved, &bytes);
                return Ok(Value::Unit);
            }
        }
        Err(tenth::error::TenthError::RuntimeError {
            message: "save_weights(路径, 张量列表)".into(),
        })
    });
    // 14. load_weights(path) — 从二进制文件反序列化张量列表（ML 训练关键路径）
    vm.add_native("load_weights".into(), |vm, args| {
        if let Some(Value::String(path)) = args.first() {
            // H-2: 沙箱校验
            let resolved = if let Some(ref sb) = vm.fs_sandbox {
                match sb.check_read(path) {
                    Ok(p) => p,
                    Err(e) => return Err(tenth::error::TenthError::RuntimeError { message: e }),
                }
            } else {
                std::path::PathBuf::from(path)
            };
            match std::fs::read(&resolved) {
                Ok(bytes) => {
                    if bytes.len() < 4 {
                        return Err(tenth::error::TenthError::RuntimeError {
                            message: "load_weights: 文件过短".into(),
                        });
                    }
                    let num = i32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as usize;
                    let mut offset: usize = 4;
                    let mut result: Vec<Value> = Vec::new();
                    for _ in 0..num {
                        if offset + 4 > bytes.len() { break; }
                        let ndim = i32::from_le_bytes([
                            bytes[offset], bytes[offset+1], bytes[offset+2], bytes[offset+3]
                        ]) as usize;
                        offset += 4;
                        let mut shape = Vec::new();
                        for _ in 0..ndim {
                            if offset + 4 > bytes.len() { break; }
                            let d = i32::from_le_bytes([
                                bytes[offset], bytes[offset+1], bytes[offset+2], bytes[offset+3]
                            ]) as usize;
                            shape.push(d);
                            offset += 4;
                        }
                        let nel: usize = shape.iter().product();
                        let data_len = nel * 8; // f64 = 8 bytes
                        if offset + data_len > bytes.len() { break; }
                        let mut data = Vec::with_capacity(nel);
                        for i in 0..nel {
                            let start = offset + i * 8;
                            let val = f64::from_le_bytes([
                                bytes[start], bytes[start+1], bytes[start+2], bytes[start+3],
                                bytes[start+4], bytes[start+5], bytes[start+6], bytes[start+7],
                            ]);
                            data.push(val);
                        }
                        offset += data_len;
                        result.push(Value::Tensor(Rc::new(RefCell::new(
                            Tensor::from_vec(data, shape)
                        ))));
                    }
                    Ok(Value::Vec(Rc::new(RefCell::new(result))))
                }
                Err(e) => Err(tenth::error::TenthError::RuntimeError {
                    message: format!("load_weights: {}", e),
                }),
            }
        } else {
            Err(tenth::error::TenthError::RuntimeError {
                message: "load_weights(路径)".into(),
            })
        }
    });
    // 15. format(template, args...) — 模板字符串格式化（{}/{{/}}）
    vm.add_native("format".into(), |_vm, args| {
        if args.is_empty() {
            return Err(tenth::error::TenthError::RuntimeError {
                message: "format() 至少需要一个模板字符串".into(),
            });
        }
        if let Value::String(template) = &args[0] {
            let mut result = String::new();
            let mut arg_idx = 1;
            let mut chars = template.chars().peekable();
            while let Some(c) = chars.next() {
                if c == '{' {
                    if chars.peek() == Some(&'{') {
                        chars.next();
                        result.push('{');
                    } else {
                        let mut placeholder = String::new();
                        while let Some(pc) = chars.next() {
                            if pc == '}' {
                                break;
                            }
                            placeholder.push(pc);
                        }
                        if arg_idx < args.len() {
                            result.push_str(&format!("{}", args[arg_idx]));
                            arg_idx += 1;
                        } else {
                            result.push('{');
                            result.push_str(&placeholder);
                            result.push('}');
                        }
                    }
                } else if c == '}' {
                    if chars.peek() == Some(&'}') {
                        chars.next();
                        result.push('}');
                    } else {
                        result.push('}');
                    }
                } else {
                    result.push(c);
                }
            }
            Ok(Value::String(result))
        } else {
            Err(tenth::error::TenthError::RuntimeError {
                message: "format() 第一个参数必须是字符串模板".into(),
            })
        }
    });
    // 16. parse_int(s) — 字符串→整数（解析失败返回 0，与解释器一致）
    vm.add_native("parse_int".into(), |_vm, args| {
        if let Some(Value::String(s)) = args.first() {
            Ok(Value::Int(s.trim().parse::<i64>().unwrap_or(0)))
        } else {
            Err(tenth::error::TenthError::RuntimeError {
                message: "parse_int() 期望一个字符串参数".into(),
            })
        }
    });
    // 17. parse_float(s) — 字符串→浮点（解析失败返回 0.0，与解释器一致）
    vm.add_native("parse_float".into(), |_vm, args| {
        if let Some(Value::String(s)) = args.first() {
            Ok(Value::Float(s.trim().parse::<f64>().unwrap_or(0.0)))
        } else {
            Err(tenth::error::TenthError::RuntimeError {
                message: "parse_float() 期望一个字符串参数".into(),
            })
        }
    });
}

/// Compile a .th file to a .wasm binary.
fn build_wasm(path: &str) -> TenthResult<()> {
    let source = std::fs::read_to_string(path)
        .map_err(|e| tenth::error::TenthError::RuntimeError {
            message: format!("无法读取 {}：{}", path, e),
        })?;
    let hir = source_to_hir(&source)?;
    let wasm_bytes = compile::compile_to_wasm(&hir)?;

    let out_path = path.replace(".th", ".wasm");
    std::fs::write(&out_path, &wasm_bytes)
        .map_err(|e| tenth::error::TenthError::RuntimeError {
            message: format!("无法写入 {}：{}", out_path, e),
        })?;
    println!("Compiled to {}", out_path);
    Ok(())
}

/// Compile a .th file to WASM and execute it via wasmi.
fn run_wasm(path: &str) -> TenthResult<()> {
    let source = std::fs::read_to_string(path)
        .map_err(|e| tenth::error::TenthError::RuntimeError {
            message: format!("无法读取 {}：{}", path, e),
        })?;
    let hir = source_to_hir(&source)?;
    compile::run_wasm(&hir)
}

/// Compile a .th file to bytecode and execute via the stack VM.
#[allow(dead_code)]
fn vm_run(path: &str) -> TenthResult<()> {
    let source = std::fs::read_to_string(path)
        .map_err(|e| tenth::error::TenthError::RuntimeError {
            message: format!("无法读取 {}：{}", path, e),
        })?;
    let hir = source_to_hir(&source)?;
    let mut vm = Vm::new();

    // Register native functions
    vm.add_native("println".into(), |_vm, args| {
        for a in args { print!("{a}"); }
        println!();
        Ok(Value::Unit)
    });
    vm.add_native("read_file".into(), |vm, args| {
        if let Some(Value::String(path)) = args.first() {
            // H-2: 沙箱校验
            let resolved = if let Some(ref sb) = vm.fs_sandbox {
                match sb.check_read(path) {
                    Ok(p) => p,
                    Err(e) => return Err(tenth::error::TenthError::RuntimeError { message: e }),
                }
            } else {
                std::path::PathBuf::from(path)
            };
            match std::fs::read_to_string(&resolved) {
                Ok(s) => Ok(Value::String(s)),
                Err(e) => Err(tenth::error::TenthError::RuntimeError { message: format!("读取文件: {e}") }),
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
                    Some(val) => println!("= {}", val),
                    None => {}
                }
            }
        }
    }

    Ok(())
}
