use std::rc::Rc;
use std::cell::RefCell;
use tenth::error::TenthResult;
use tenth::repl;
use tenth::runtime::limits::MemoryConfig;
use tenth::lexer::lexer::Lexer;
use tenth::parser::parser::Parser;
use tenth::runtime::interpreter::Interpreter;
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
fn run_file(path: &str) -> TenthResult<()> {
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

    // Skip VM if TENTH_NO_VM env var is set (for debugging interpreter)
    let skip_vm = std::env::var("TENTH_NO_VM").is_ok();
    if !skip_vm {
        match vm_execute(&hir) {
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

fn days_to_date(days: u64) -> (u64, u64, u64) {
    let z = days + 719468;
    let era = z / 146097;
    let doe = z - era * 146097;
    let yoe = (doe - doe/1460 + doe/36524 - doe/146096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365*yoe + yoe/4 - yoe/100);
    let mp = (5*doy + 2) / 153;
    let d = doy - (153*mp+2)/5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    (y, m, d)
}

fn json_encode_value(val: &tenth::runtime::value::Value) -> String {
    use tenth::runtime::value::Value;
    match val {
        Value::Int(n) => format!("{}", n),
        Value::Float(f) => format!("{}", f),
        Value::Bool(b) => if *b { "true".into() } else { "false".into() },
        Value::String(s) => format!("\"{}\"", s.replace('\\', "\\\\").replace('"', "\\\"").replace('\n', "\\n").replace('\t', "\\t")),
        Value::Unit => "null".into(),
        Value::Vec(v) => {
            let items: Vec<String> = v.borrow().iter().map(|v| json_encode_value(v)).collect();
            format!("[{}]", items.join(", "))
        }
        Value::Array(a) => {
            let items: Vec<String> = a.borrow().iter().map(|v| json_encode_value(v)).collect();
            format!("[{}]", items.join(", "))
        }
        Value::Map(map) => {
            let entries: Vec<String> = map.borrow().iter().map(|(k, v)| {
                format!("\"{}\": {}", k, json_encode_value(v))
            }).collect();
            format!("{{{}}}", entries.join(", "))
        }
        _ => "null".into(),
    }
}

fn json_encode_value_pretty(val: &tenth::runtime::value::Value, indent: usize) -> String {
    use tenth::runtime::value::Value;
    let prefix = "  ".repeat(indent);
    let inner_prefix = "  ".repeat(indent + 1);
    match val {
        Value::Int(n) => format!("{}", n),
        Value::Float(f) => format!("{}", f),
        Value::Bool(b) => if *b { "true".into() } else { "false".into() },
        Value::String(s) => format!("\"{}\"", s.replace('\\', "\\\\").replace('"', "\\\"").replace('\n', "\\n").replace('\t', "\\t")),
        Value::Unit => "null".into(),
        Value::Vec(v) => {
            if v.borrow().is_empty() { return "[]".into(); }
            let items: Vec<String> = v.borrow().iter().map(|v| format!("{}{}", inner_prefix, json_encode_value_pretty(v, indent + 1))).collect();
            format!("[\n{}\n{}]", items.join(",\n"), prefix)
        }
        Value::Array(a) => {
            if a.borrow().is_empty() { return "[]".into(); }
            let items: Vec<String> = a.borrow().iter().map(|v| format!("{}{}", inner_prefix, json_encode_value_pretty(v, indent + 1))).collect();
            format!("[\n{}\n{}]", items.join(",\n"), prefix)
        }
        Value::Map(map) => {
            if map.borrow().is_empty() { return "{}".into(); }
            let entries: Vec<String> = map.borrow().iter().map(|(k, v)| {
                format!("{}\"{}\": {}", inner_prefix, k, json_encode_value_pretty(v, indent + 1))
            }).collect();
            format!("{{\n{}\n{}}}", entries.join(",\n"), prefix)
        }
        _ => "null".into(),
    }
}

fn json_decode_string(s: &str) -> tenth::runtime::value::Value {
    use tenth::runtime::value::Value;
    use std::rc::Rc;
    use std::cell::RefCell;
    let s = s.trim();
    if s == "null" { return Value::Unit; }
    if s == "true" { return Value::Bool(true); }
    if s == "false" { return Value::Bool(false); }
    if s.starts_with('"') && s.ends_with('"') {
        let inner = &s[1..s.len()-1];
        return Value::String(inner.replace("\\\"", "\"").replace("\\\\", "\\").replace("\\n", "\n").replace("\\t", "\t"));
    }
    if let Ok(n) = s.parse::<i64>() { return Value::Int(n); }
    if let Ok(f) = s.parse::<f64>() { return Value::Float(f); }
    if s.starts_with('[') && s.ends_with(']') {
        let inner = &s[1..s.len()-1];
        if inner.trim().is_empty() { return Value::Vec(Rc::new(RefCell::new(Vec::new()))); }
        let items: Vec<Value> = simple_json_split(inner, ',')
            .iter()
            .map(|s| json_decode_string(s))
            .collect();
        return Value::Vec(Rc::new(RefCell::new(items)));
    }
    if s.starts_with('{') && s.ends_with('}') {
        let inner = &s[1..s.len()-1];
        if inner.trim().is_empty() {
            return Value::Map(Rc::new(RefCell::new(std::collections::HashMap::new())));
        }
        let mut map = std::collections::HashMap::new();
        let entries = simple_json_split(inner, ',');
        for entry in &entries {
            let parts = simple_json_split(entry, ':');
            if parts.len() >= 2 {
                let key = json_decode_string(parts[0].trim());
                if let Value::String(k) = key {
                    let val = json_decode_string(parts[1].trim());
                    map.insert(k, val);
                }
            }
        }
        return Value::Map(Rc::new(RefCell::new(map)));
    }
    Value::Unit
}

fn simple_json_split(s: &str, delimiter: char) -> Vec<String> {
    let mut result = Vec::new();
    let mut current = String::new();
    let mut depth = 0i32;
    let mut in_string = false;
    for c in s.chars() {
        if c == '"' && !in_string { in_string = true; current.push(c); continue; }
        if c == '"' && in_string { in_string = false; current.push(c); continue; }
        if in_string { current.push(c); continue; }
        match c {
            '[' | '{' => { depth += 1; current.push(c); }
            ']' | '}' => { depth -= 1; current.push(c); }
            d if d == delimiter && depth == 0 => {
                result.push(current.trim().to_string());
                current = String::new();
            }
            _ => { current.push(c); }
        }
    }
    let trimmed = current.trim().to_string();
    if !trimmed.is_empty() { result.push(trimmed); }
    result
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
    vm.add_native("read_bytes".into(), |_vm, args| {
        if let Some(Value::String(path)) = args.first() {
            match std::fs::read(path) {
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
        let (year, month, day) = days_to_date(days_since_epoch);
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
        let (year, month, day) = days_to_date(days_since_epoch);
        let day_secs = secs % 86400;
        let h = day_secs / 3600;
        let m = (day_secs % 3600) / 60;
        let s = day_secs % 60;
        Ok(Value::String(format!("{}-{:02}-{:02} {}:{:02}:{:02}", year, month, day, h, m, s)))
    });
    vm.add_native("time_sleep_ms".into(), |_vm, args| {
        if let Some(Value::Int(ms)) = args.first() {
            std::thread::sleep(std::time::Duration::from_millis(*ms as u64));
            Ok(Value::Unit)
        } else {
            Err(tenth::error::TenthError::RuntimeError { message: "time_sleep_ms(ms) 期望一个整数".into() })
        }
    });
    // Random functions
    vm.add_native("random_int".into(), |_vm, args| {
        let lo = match args.first() {
            Some(Value::Int(n)) => *n,
            _ => 0,
        };
        let hi = match args.get(1) {
            Some(Value::Int(n)) => *n,
            _ => lo,
        };
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        let mut hasher = DefaultHasher::new();
        now.hash(&mut hasher);
        let rand_val = hasher.finish();
        let range = (hi - lo + 1).max(1);
        Ok(Value::Int(lo + ((rand_val % (range as u64)) as i64)))
    });
    vm.add_native("random_float".into(), |_vm, _args| {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        let mut hasher = DefaultHasher::new();
        now.hash(&mut hasher);
        let rand_val = hasher.finish();
        Ok(Value::Float((rand_val as f64) / (u64::MAX as f64)))
    });
    // Math functions
    vm.add_native("math_tan".into(), |_vm, args| {
        match args.first() { Some(Value::Float(x)) => Ok(Value::Float(x.tan())), _ => Ok(Value::Float(0.0)) }
    });
    vm.add_native("math_asin".into(), |_vm, args| {
        match args.first() { Some(Value::Float(x)) => Ok(Value::Float(x.asin())), _ => Ok(Value::Float(0.0)) }
    });
    vm.add_native("math_acos".into(), |_vm, args| {
        match args.first() { Some(Value::Float(x)) => Ok(Value::Float(x.acos())), _ => Ok(Value::Float(0.0)) }
    });
    vm.add_native("math_atan".into(), |_vm, args| {
        match args.first() { Some(Value::Float(x)) => Ok(Value::Float(x.atan())), _ => Ok(Value::Float(0.0)) }
    });
    vm.add_native("math_atan2".into(), |_vm, args| {
        match (args.first(), args.get(1)) {
            (Some(Value::Float(y)), Some(Value::Float(x))) => Ok(Value::Float(y.atan2(*x))),
            _ => Ok(Value::Float(0.0))
        }
    });
    vm.add_native("math_sinh".into(), |_vm, args| {
        match args.first() { Some(Value::Float(x)) => Ok(Value::Float(x.sinh())), _ => Ok(Value::Float(0.0)) }
    });
    vm.add_native("math_cosh".into(), |_vm, args| {
        match args.first() { Some(Value::Float(x)) => Ok(Value::Float(x.cosh())), _ => Ok(Value::Float(0.0)) }
    });
    vm.add_native("math_tanh".into(), |_vm, args| {
        match args.first() { Some(Value::Float(x)) => Ok(Value::Float(x.tanh())), _ => Ok(Value::Float(0.0)) }
    });
    vm.add_native("math_log10".into(), |_vm, args| {
        match args.first() { Some(Value::Float(x)) => Ok(Value::Float(x.log10())), _ => Ok(Value::Float(0.0)) }
    });
    vm.add_native("math_log2".into(), |_vm, args| {
        match args.first() { Some(Value::Float(x)) => Ok(Value::Float(x.log2())), _ => Ok(Value::Float(0.0)) }
    });
    vm.add_native("math_exp".into(), |_vm, args| {
        match args.first() { Some(Value::Float(x)) => Ok(Value::Float(x.exp())), _ => Ok(Value::Float(0.0)) }
    });
    vm.add_native("math_pow".into(), |_vm, args| {
        match (args.first(), args.get(1)) {
            (Some(Value::Float(base)), Some(Value::Float(exp))) => Ok(Value::Float(base.powf(*exp))),
            _ => Ok(Value::Float(0.0))
        }
    });
    vm.add_native("math_floor".into(), |_vm, args| {
        match args.first() { Some(Value::Float(x)) => Ok(Value::Float(x.floor())), _ => Ok(Value::Float(0.0)) }
    });
    vm.add_native("math_ceil".into(), |_vm, args| {
        match args.first() { Some(Value::Float(x)) => Ok(Value::Float(x.ceil())), _ => Ok(Value::Float(0.0)) }
    });
    vm.add_native("math_round".into(), |_vm, args| {
        match args.first() { Some(Value::Float(x)) => Ok(Value::Float(x.round())), _ => Ok(Value::Float(0.0)) }
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
            Ok(Value::String(json_encode_value(val)))
        } else {
            Ok(Value::String("null".into()))
        }
    });
    vm.add_native("json_encode_pretty".into(), |_vm, args| {
        if let Some(val) = args.first() {
            Ok(Value::String(json_encode_value_pretty(val, 0)))
        } else {
            Ok(Value::String("null".into()))
        }
    });
    vm.add_native("json_decode".into(), |_vm, args| {
        if let Some(Value::String(s)) = args.first() {
            Ok(json_decode_string(s))
        } else {
            Ok(Value::Unit)
        }
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
                tape.backward(loss_id);
                Ok(Value::Unit)
            } else {
                Err(tenth::error::TenthError::RuntimeError { message: "未调用 new_grad()".into() })
            }
        } else {
            Err(tenth::error::TenthError::RuntimeError { message: "backward() 需要一个张量参数".into() })
        }
    });
    vm.add_native("grad".into(), |_vm, args| {
        if let Some(Value::Tensor(t)) = args.first() {
            let p = t.borrow();
            if let Some(ref grad) = p.grad {
                let grad_tensor = Tensor::from_vec(grad.clone().into_raw_vec(), p.shape());
                Ok(Value::Tensor(Rc::new(RefCell::new(grad_tensor))))
            } else {
                let zeros = Tensor::zeros(&p.shape());
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
    vm.add_native("cross_entropy".into(), |vm, args| {
        if args.len() < 2 {
            return Err(tenth::error::TenthError::RuntimeError { message: "cross_entropy(logits, target) 期望两个张量".into() });
        }
        if let (Value::Tensor(logits), Value::Tensor(target)) = (&args[0], &args[1]) {
            let logits_data = logits.borrow();
            let target_data = target.borrow();
            let sm = logits_data.softmax().ok_or_else(|| {
                tenth::error::TenthError::RuntimeError { message: "cross_entropy 中 softmax 失败".into() }
            })?;
            let eps = 1e-10;
            let sm_data = sm.data.as_standard_layout().to_owned();
            let tgt_flat = target_data.data.as_standard_layout().to_owned();
            let sm_slice = sm_data.as_slice().unwrap_or(&[]);
            let tgt_slice = tgt_flat.as_slice().unwrap_or(&[]);
            let mut loss_val = 0.0f64;
            let n = sm_slice.len() as f64;
            for i in 0..sm_slice.len().min(tgt_slice.len()) {
                let p = sm_slice[i].max(eps);
                loss_val -= tgt_slice[i] * p.ln();
            }
            loss_val /= n.max(1.0);
            let loss_tensor = Tensor::from_vec(vec![loss_val], vec![1]);
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
    vm.add_native("write_file".into(), |_vm, args| {
        if args.len() >= 2 {
            if let (Value::String(path), Value::String(content)) = (&args[0], &args[1]) {
                match std::fs::write(path, content) {
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
    vm.add_native("path_exists".into(), |_vm, args| {
        if let Some(Value::String(path)) = args.first() {
            Ok(Value::Bool(std::path::Path::new(path).exists()))
        } else {
            Err(tenth::error::TenthError::RuntimeError { message: "path_exists(路径) 期望一个字符串路径".into() })
        }
    });
    vm.add_native("path_is_file".into(), |_vm, args| {
        if let Some(Value::String(path)) = args.first() {
            Ok(Value::Bool(std::path::Path::new(path).is_file()))
        } else {
            Err(tenth::error::TenthError::RuntimeError { message: "path_is_file(路径) 期望一个字符串路径".into() })
        }
    });
    vm.add_native("path_is_dir".into(), |_vm, args| {
        if let Some(Value::String(path)) = args.first() {
            Ok(Value::Bool(std::path::Path::new(path).is_dir()))
        } else {
            Err(tenth::error::TenthError::RuntimeError { message: "path_is_dir(路径) 期望一个字符串路径".into() })
        }
    });
    vm.add_native("mkdir".into(), |_vm, args| {
        if let Some(Value::String(path)) = args.first() {
            match std::fs::create_dir_all(path) {
                Ok(()) => Ok(Value::Unit),
                Err(e) => Err(tenth::error::TenthError::RuntimeError { message: format!("创建目录失败: {}", e) }),
            }
        } else {
            Err(tenth::error::TenthError::RuntimeError { message: "mkdir(路径) 期望一个字符串路径".into() })
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
                Err(e) => Err(tenth::error::TenthError::RuntimeError { message: format!("列出目录失败: {}", e) }),
            }
        } else {
            Err(tenth::error::TenthError::RuntimeError { message: "list_dir(路径) 期望一个字符串路径".into() })
        }
    });
    vm.add_native("file_size".into(), |_vm, args| {
        if let Some(Value::String(path)) = args.first() {
            match std::fs::metadata(path) {
                Ok(meta) => Ok(Value::Int(meta.len() as i64)),
                Err(e) => Err(tenth::error::TenthError::RuntimeError { message: format!("获取文件大小失败: {}", e) }),
            }
        } else {
            Err(tenth::error::TenthError::RuntimeError { message: "file_size(路径) 期望一个字符串路径".into() })
        }
    });
    vm.add_native("remove_file".into(), |_vm, args| {
        if let Some(Value::String(path)) = args.first() {
            match std::fs::remove_file(path) {
                Ok(()) => Ok(Value::Unit),
                Err(e) => Err(tenth::error::TenthError::RuntimeError { message: format!("删除文件失败: {}", e) }),
            }
        } else {
            Err(tenth::error::TenthError::RuntimeError { message: "remove_file(路径) 期望一个字符串路径".into() })
        }
    });
    vm.add_native("copy_file".into(), |_vm, args| {
        if args.len() >= 2 {
            if let (Value::String(src), Value::String(dst)) = (&args[0], &args[1]) {
                match std::fs::copy(src, dst) {
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
    vm.add_native("rename_file".into(), |_vm, args| {
        if args.len() >= 2 {
            if let (Value::String(src), Value::String(dst)) = (&args[0], &args[1]) {
                match std::fs::rename(src, dst) {
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
            _ => Err(tenth::error::TenthError::RuntimeError { message: "abs() 需要一个数值参数".into() }),
        }
    });
    vm.add_native("sqrt".into(), |_vm, args| {
        match args.first() {
            Some(Value::Float(f)) => Ok(Value::Float(f.sqrt())),
            Some(Value::Int(n)) => Ok(Value::Float((*n as f64).sqrt())),
            _ => Err(tenth::error::TenthError::RuntimeError { message: "sqrt() 需要一个数值参数".into() }),
        }
    });
    vm.add_native("to_float".into(), |_vm, args| {
        match args.first() {
            Some(Value::Int(n)) => Ok(Value::Float(*n as f64)),
            Some(Value::Float(f)) => Ok(Value::Float(*f)),
            _ => Err(tenth::error::TenthError::RuntimeError { message: "to_float() 需要一个数值参数".into() }),
        }
    });
    vm.add_native("tensor_from_vec".into(), |_vm, args| {
        if args.len() >= 3 {
            if let (Value::Vec(items), Value::Int(rows), Value::Int(cols)) = (&args[0], &args[1], &args[2]) {
                let data: Vec<f64> = items.borrow().iter().map(|v| v.as_float().unwrap_or(0.0)).collect();
                let tensor = Tensor::from_vec(data, vec![*rows as usize, *cols as usize]);
                Ok(Value::Tensor(Rc::new(RefCell::new(tensor))))
            } else {
                Err(tenth::error::TenthError::RuntimeError { message: "tensor_from_vec(vec, rows, cols) 期望一个 Vec 和两个整数".into() })
            }
        } else {
            Err(tenth::error::TenthError::RuntimeError { message: "tensor_from_vec(vec, rows, cols) 期望 3 个参数".into() })
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
    vm.add_native("read_file".into(), |_vm, args| {
        if let Some(Value::String(path)) = args.first() {
            match std::fs::read_to_string(path) {
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
