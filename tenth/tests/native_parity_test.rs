//! 论文 T37 修复第二批：VM/解释器 native 注册对齐测试。
//!
//! 覆盖补齐的 20 项 native（VM 17 项 + 解释器 3 项），验证 VM 路径与
//! 解释器路径对同一 .th 源码产生一致结果。
//!
//! 注意：`register_test_natives` 复制自 `main.rs::register_natives` 中
//! 与本测试相关的 native 子集（17 项新 native + 必要辅助 native）。
//! 修改 main.rs 时需同步此函数。这是项目现有惯例（参见 vm_autodiff_test.rs）。

use tenth::lexer::lexer::Lexer;
use tenth::parser::parser::Parser;
use tenth::hir::lower::Lowerer;
use tenth::runtime::vm::Vm;
use tenth::runtime::interpreter::Interpreter;
use tenth::runtime::value::Value;
use tenth::runtime::autodiff::Tape;
use tenth::runtime::tensor::{Tensor, TensorData};
use tenth::compile::bytecode::BytecodeCompiler;
use std::rc::Rc;
use std::cell::RefCell;
use tenth::hir::types::BaseType;
use tenth::error::TenthWarning;

/// 注册测试所需的 native（复制自 main.rs::register_natives 的相关子集）。
/// 17 项新 native 必须与 main.rs 完全一致；辅助 native（println/Vec::new 等）
/// 仅需满足测试源码的运行需求。
fn register_test_natives(vm: &mut Vm) {
    // ── 辅助 native（main.rs 已有，测试需要）──
    vm.add_native("println".into(), |_vm, args| {
        for a in args { print!("{a}"); }
        println!();
        Ok(Value::Unit)
    });
    vm.add_native("Vec::new".into(), |_vm, _args| {
        Ok(Value::Vec(Rc::new(RefCell::new(Vec::new()))))
    });
    vm.add_native("tensor".into(), |_vm, args| {
        if args.len() == 1 { Ok(args[0].clone()) }
        else { Err(tenth::error::TenthError::RuntimeError { line: None, col: None, message: "tensor() 参数异常".into() }) }
    });
    vm.add_native("zeros".into(), |_vm, args| {
        let shape: Vec<usize> = args.iter().map(|a| a.as_int().unwrap_or(1) as usize).collect();
        Ok(Value::Tensor(Rc::new(RefCell::new(Tensor::zeros(&shape)))))
    });
    vm.add_native("ones".into(), |_vm, args| {
        let shape: Vec<usize> = args.iter().map(|a| a.as_int().unwrap_or(1) as usize).collect();
        Ok(Value::Tensor(Rc::new(RefCell::new(Tensor::ones(&shape)))))
    });
    // 解释器补齐的 3 项（VM 侧 main.rs 已有，这里注册以便 VM 路径测试）
    vm.add_native("print".into(), |_vm, args| {
        for a in args { print!("{a}"); }
        Ok(Value::Unit)
    });
    vm.add_native("to_f64".into(), |_vm, args| {
        match args.first() {
            Some(Value::Int(n, _)) => Ok(Value::Float(*n as f64)),
            Some(Value::Float(f)) => Ok(Value::Float(*f)),
            Some(Value::Float32(f)) => Ok(Value::Float(*f as f64)),
            _ => Err(tenth::error::TenthError::RuntimeError { line: None, col: None, message: "to_f64() 需要一个数值参数".into() }),
        }
    });
    vm.add_native("to_f32".into(), |_vm, args| {
        match args.first() {
            Some(Value::Int(n, _)) => Ok(Value::Float32(*n as f32)),
            Some(Value::Float(f)) => Ok(Value::Float32(*f as f32)),
            Some(Value::Float32(f)) => Ok(Value::Float32(*f)),
            _ => Err(tenth::error::TenthError::RuntimeError { line: None, col: None, message: "to_f32() 需要一个数值参数".into() }),
        }
    });

    // ── 17 项新 native（复制自 main.rs，须保持同步）──
    vm.add_native("to_string".into(), |_vm, args| {
        if let Some(arg) = args.first() { Ok(Value::String(format!("{}", arg))) }
        else { Ok(Value::String(String::new())) }
    });
    vm.add_native("type_name".into(), |_vm, args| {
        if let Some(arg) = args.first() {
            let tn = match arg {
                Value::Int(_, _) => "int",
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
        } else { Ok(Value::String("unknown".to_string())) }
    });
    vm.add_native("with_step_limit".into(), |vm, args| {
        if args.len() < 2 {
            return Err(tenth::error::TenthError::RuntimeError { line: None, col: None, message: "with_step_limit(limit, fn) 需要 2 个参数".into() });
        }
        let limit = args[0].as_int().ok_or_else(|| tenth::error::TenthError::RuntimeError { line: None, col: None,
            message: "with_step_limit 的第一个参数必须是整数步数".into() })?;
        let saved_budget = vm.step_budget;
        let saved_deadline = vm.deadline_ms;
        vm.step_budget = Some(limit.max(0) as u64);
        vm.deadline_ms = None;
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
    vm.add_native("with_timeout_ms".into(), |vm, args| {
        if args.len() < 2 {
            return Err(tenth::error::TenthError::RuntimeError { line: None, col: None, message: "with_timeout_ms(ms, fn) 需要 2 个参数".into() });
        }
        let ms = args[0].as_int().ok_or_else(|| tenth::error::TenthError::RuntimeError { line: None, col: None,
            message: "with_timeout_ms 的第一个参数必须是整数毫秒".into() })?;
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis()).unwrap_or(0);
        let saved_budget = vm.step_budget;
        let saved_deadline = vm.deadline_ms;
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
    vm.add_native("is_timeout".into(), |_vm, args| {
        if let Some(arg) = args.first() { Ok(Value::Bool(matches!(arg, Value::Unit))) }
        else { Ok(Value::Bool(false)) }
    });
    vm.add_native("start_grad".into(), |vm, _args| {
        vm.tape = Some(Tape::new());
        vm.recording = true;
        Ok(Value::Unit)
    });
    vm.add_native("f64_bits".into(), |_vm, args| {
        if let Some(arg) = args.first() {
            let f = arg.as_float().ok_or_else(|| tenth::error::TenthError::RuntimeError { line: None, col: None,
                message: "f64_bits() 期望一个 f64 参数".into() })?;
            Ok(Value::Int(f.to_bits() as i64, BaseType::I32))
        } else { Err(tenth::error::TenthError::RuntimeError { line: None, col: None, message: "f64_bits() 期望 1 个参数".into() }) }
    });
    vm.add_native("f64_from_bits".into(), |_vm, args| {
        if let Some(arg) = args.first() {
            let n = arg.as_int().ok_or_else(|| tenth::error::TenthError::RuntimeError { line: None, col: None,
                message: "f64_from_bits() 期望一个 i64 参数".into() })?;
            Ok(Value::Float(f64::from_bits(n as u64)))
        } else { Err(tenth::error::TenthError::RuntimeError { line: None, col: None, message: "f64_from_bits() 期望 1 个参数".into() }) }
    });
    vm.add_native("sin".into(), |_vm, args| {
        if let Some(arg) = args.first() {
            let n = arg.as_float().ok_or_else(|| tenth::error::TenthError::RuntimeError { line: None, col: None,
                message: "sin() 期望一个数值参数".into() })?;
            Ok(Value::Float(n.sin()))
        } else { Err(tenth::error::TenthError::RuntimeError { line: None, col: None, message: "sin() 期望 1 个参数".into() }) }
    });
    vm.add_native("cos".into(), |_vm, args| {
        if let Some(arg) = args.first() {
            let n = arg.as_float().ok_or_else(|| tenth::error::TenthError::RuntimeError { line: None, col: None,
                message: "cos() 期望一个数值参数".into() })?;
            Ok(Value::Float(n.cos()))
        } else { Err(tenth::error::TenthError::RuntimeError { line: None, col: None, message: "cos() 期望 1 个参数".into() }) }
    });
    vm.add_native("ln".into(), |_vm, args| {
        if let Some(arg) = args.first() {
            let n = arg.as_float().ok_or_else(|| tenth::error::TenthError::RuntimeError { line: None, col: None,
                message: "ln() 期望一个数值参数".into() })?;
            if n <= 0.0 {
                return Err(tenth::error::TenthError::RuntimeError { line: None, col: None, message: "ln() 参数必须 > 0".into() });
            }
            Ok(Value::Float(n.ln()))
        } else { Err(tenth::error::TenthError::RuntimeError { line: None, col: None, message: "ln() 期望 1 个参数".into() }) }
    });
    vm.add_native("pow".into(), |_vm, args| {
        if args.len() >= 2 {
            let base = args[0].as_float().ok_or_else(|| tenth::error::TenthError::RuntimeError { line: None, col: None,
                message: "pow() 期望数值参数".into() })?;
            let exp = args[1].as_float().ok_or_else(|| tenth::error::TenthError::RuntimeError { line: None, col: None,
                message: "pow() 期望数值参数".into() })?;
            Ok(Value::Float(base.powf(exp)))
        } else { Err(tenth::error::TenthError::RuntimeError { line: None, col: None, message: "pow() 期望 2 个参数".into() }) }
    });
    vm.add_native("save_weights".into(), |vm, args| {
        if args.len() >= 2 {
            if let Value::String(path) = &args[0] {
                let resolved = if let Some(ref sb) = vm.fs_sandbox {
                    match sb.check_write(path) {
                        Ok(p) => p,
                        Err(e) => return Err(tenth::error::TenthError::RuntimeError { line: None, col: None, message: e }),
                    }
                } else { std::path::PathBuf::from(path) };
                let tensors: &Rc<RefCell<Vec<Value>>> = match &args[1] {
                    Value::Vec(v) => v,
                    Value::Array(a) => a,
                    _ => return Err(tenth::error::TenthError::RuntimeError { line: None, col: None, message: "save_weights 期望一个张量列表".into() }),
                };
                let tensors_ref = tensors.borrow();
                let mut bytes: Vec<u8> = Vec::new();
                // v2 格式: magic "THW1" + version=2 + num_tensors
                bytes.extend(b"THW1");
                bytes.extend(&2i32.to_le_bytes());
                bytes.extend(&(tensors_ref.len() as i32).to_le_bytes());
                for val in tensors_ref.iter() {
                    let tensor_rc = match val {
                        Value::Tensor(t) => Some(t.clone()),
                        Value::Shared(rc) => {
                            if let Value::Tensor(t) = &*rc.borrow() { Some(t.clone()) } else { None }
                        }
                        _ => None,
                    };
                    if let Some(t) = tensor_rc {
                        let t_ref = t.borrow();
                        let shape = t_ref.shape();
                        let ndim = shape.len() as i32;
                        bytes.extend(&ndim.to_le_bytes());
                        for &d in &shape { bytes.extend(&(d as i32).to_le_bytes()); }
                        // dtype 字段: F32=0, F64=1, F16=2, BF16=3 (Wave 2)
                        let dtype_val: i32 = match &t_ref.data {
                            TensorData::F32(_) => 0,
                            TensorData::F64(_) => 1,
                            TensorData::F16(_) => 2,
                            TensorData::BF16(_) => 3,
                        };
                        bytes.extend(&dtype_val.to_le_bytes());
                        // 按 dtype 分发写数据
                        match &t_ref.data {
                            TensorData::F64(arr) => {
                                let flat = arr.as_standard_layout();
                                if let Some(slice) = flat.as_slice() {
                                    for &x in slice { bytes.extend(&x.to_le_bytes()); }
                                } else {
                                    for &x in flat.iter() { bytes.extend(&x.to_le_bytes()); }
                                }
                            }
                            TensorData::F32(arr) => {
                                let flat = arr.as_standard_layout();
                                if let Some(slice) = flat.as_slice() {
                                    for &x in slice { bytes.extend(&x.to_le_bytes()); }
                                } else {
                                    for &x in flat.iter() { bytes.extend(&x.to_le_bytes()); }
                                }
                            }
                            // Wave 2: F16/BF16 各 2 字节/元素
                            TensorData::F16(arr) => {
                                let flat = arr.as_standard_layout();
                                if let Some(slice) = flat.as_slice() {
                                    for &x in slice { bytes.extend(&x.to_le_bytes()); }
                                } else {
                                    for &x in flat.iter() { bytes.extend(&x.to_le_bytes()); }
                                }
                            }
                            TensorData::BF16(arr) => {
                                let flat = arr.as_standard_layout();
                                if let Some(slice) = flat.as_slice() {
                                    for &x in slice { bytes.extend(&x.to_le_bytes()); }
                                } else {
                                    for &x in flat.iter() { bytes.extend(&x.to_le_bytes()); }
                                }
                            }
                        }
                    }
                }
                let _ = std::fs::write(&resolved, &bytes);
                return Ok(Value::Unit);
            }
        }
        Err(tenth::error::TenthError::RuntimeError { line: None, col: None, message: "save_weights(路径, 张量列表)".into() })
    });
    vm.add_native("load_weights".into(), |vm, args| {
        if let Some(Value::String(path)) = args.first() {
            let resolved = if let Some(ref sb) = vm.fs_sandbox {
                match sb.check_read(path) {
                    Ok(p) => p,
                    Err(e) => return Err(tenth::error::TenthError::RuntimeError { line: None, col: None, message: e }),
                }
            } else { std::path::PathBuf::from(path) };
            match std::fs::read(&resolved) {
                Ok(bytes) => {
                    if bytes.len() < 4 {
                        return Err(tenth::error::TenthError::RuntimeError { line: None, col: None, message: "load_weights: 文件过短".into() });
                    }
                    let mut result: Vec<Value> = Vec::new();
                    // 检测 v2 格式: magic "THW1"
                    let is_v2 = bytes.len() >= 12 && &bytes[0..4] == b"THW1";
                    let num: usize;
                    let mut offset: usize;
                    if is_v2 {
                        let _version = i32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
                        num = i32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize;
                        offset = 12;
                    } else {
                        num = i32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as usize;
                        offset = 4;
                    }
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
                        if is_v2 {
                            if offset + 4 > bytes.len() { break; }
                            let dtype = i32::from_le_bytes([
                                bytes[offset], bytes[offset+1], bytes[offset+2], bytes[offset+3]
                            ]);
                            offset += 4;
                            match dtype {
                                0 => {
                                    let data_len = nel * 4;
                                    if offset + data_len > bytes.len() { break; }
                                    let mut data = Vec::with_capacity(nel);
                                    for i in 0..nel {
                                        let start = offset + i * 4;
                                        let val = f32::from_le_bytes([
                                            bytes[start], bytes[start+1], bytes[start+2], bytes[start+3],
                                        ]);
                                        data.push(val);
                                    }
                                    offset += data_len;
                                    result.push(Value::Tensor(Rc::new(RefCell::new(
                                        Tensor::from_vec_f32(data, shape)
                                    ))));
                                }
                                1 => {
                                    let data_len = nel * 8;
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
                                2 => {
                                    // F16: 2 字节/元素
                                    let data_len = nel * 2;
                                    if offset + data_len > bytes.len() { break; }
                                    let mut data = Vec::with_capacity(nel);
                                    for i in 0..nel {
                                        let start = offset + i * 2;
                                        let val = half::f16::from_le_bytes([
                                            bytes[start], bytes[start+1],
                                        ]);
                                        data.push(val);
                                    }
                                    offset += data_len;
                                    result.push(Value::Tensor(Rc::new(RefCell::new(
                                        Tensor::from_vec_f16(data, shape)
                                    ))));
                                }
                                3 => {
                                    // BF16: 2 字节/元素
                                    let data_len = nel * 2;
                                    if offset + data_len > bytes.len() { break; }
                                    let mut data = Vec::with_capacity(nel);
                                    for i in 0..nel {
                                        let start = offset + i * 2;
                                        let val = half::bf16::from_le_bytes([
                                            bytes[start], bytes[start+1],
                                        ]);
                                        data.push(val);
                                    }
                                    offset += data_len;
                                    result.push(Value::Tensor(Rc::new(RefCell::new(
                                        Tensor::from_vec_bf16(data, shape)
                                    ))));
                                }
                                other => {
                                    return Err(tenth::error::TenthError::RuntimeError { line: None, col: None,
                                        message: format!("load_weights: 未知 dtype={}", other),
                                    });
                                }
                            }
                        } else {
                            let data_len = nel * 8;
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
                    }
                    Ok(Value::Vec(Rc::new(RefCell::new(result))))
                }
                Err(e) => Err(tenth::error::TenthError::RuntimeError { line: None, col: None, message: format!("load_weights: {}", e) }),
            }
        } else {
            Err(tenth::error::TenthError::RuntimeError { line: None, col: None, message: "load_weights(路径)".into() })
        }
    });
    vm.add_native("format".into(), |_vm, args| {
        if args.is_empty() {
            return Err(tenth::error::TenthError::RuntimeError { line: None, col: None, message: "format() 至少需要一个模板字符串".into() });
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
                            if pc == '}' { break; }
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
                    } else { result.push('}'); }
                } else { result.push(c); }
            }
            Ok(Value::String(result))
        } else {
            Err(tenth::error::TenthError::RuntimeError { line: None, col: None, message: "format() 第一个参数必须是字符串模板".into() })
        }
    });
    vm.add_native("parse_int".into(), |_vm, args| {
        if let Some(Value::String(s)) = args.first() {
            Ok(Value::Int(s.trim().parse::<i64>().unwrap_or(0), BaseType::I32))
        } else {
            Err(tenth::error::TenthError::RuntimeError { line: None, col: None, message: "parse_int() 期望一个字符串参数".into() })
        }
    });
    vm.add_native("parse_float".into(), |_vm, args| {
        if let Some(Value::String(s)) = args.first() {
            Ok(Value::Float(s.trim().parse::<f64>().unwrap_or(0.0)))
        } else {
            Err(tenth::error::TenthError::RuntimeError { line: None, col: None, message: "parse_float() 期望一个字符串参数".into() })
        }
    });
}

/// 通过 VM 执行 .th 源码，返回结果。
fn run_vm(src: &str) -> Result<Value, String> {
    let mut lexer = Lexer::new(src);
    let tokens = lexer.tokenize().map_err(|e| e.to_string())?;
    let mut parser = Parser::new(tokens);
    let program = parser.parse_program().map_err(|e| e.to_string())?;
    let mut lowerer = Lowerer::new();
    let hir = lowerer.lower_program(&program).map_err(|e| e.to_string())?;

    let mut vm = Vm::new();
    register_test_natives(&mut vm);

    for func in &hir.functions {
        let compiler = BytecodeCompiler::new();
        match compiler.compile(func) {
            Ok((chunk, closures)) => {
                vm.add_fn(func.name.clone(), chunk);
                for (name, closure_chunk) in closures {
                    vm.add_fn(name, closure_chunk);
                }
                vm.set_global(func.name.clone(), Value::FnRef {
                    name: func.name.clone(),
                    params: func.params.clone(),
                    return_type: func.return_type.clone(),
                    captures: vec![],
                });
            }
            Err(e) => return Err(format!("compile error: {}", e)),
        }
    }

    if let Some(ref expr) = hir.main_expr {
        let compiler = BytecodeCompiler::new();
        match compiler.compile_main(expr) {
            Ok((chunk, closures)) => {
                vm.add_fn("main".into(), chunk);
                for (name, closure_chunk) in closures {
                    vm.add_fn(name, closure_chunk);
                }
            }
            Err(e) => return Err(format!("compile error: {}", e)),
        }
        vm.call("main").map_err(|e| e.to_string())
    } else if vm.has_fn("main") {
        vm.call("main").map_err(|e| e.to_string())
    } else {
        Ok(Value::Unit)
    }
}

/// 通过解释器执行 .th 源码，返回结果（自动注册所有 native）。
fn run_interp(src: &str) -> Result<Value, String> {
    let mut lexer = Lexer::new(src);
    let tokens = lexer.tokenize().map_err(|e| e.to_string())?;
    let mut parser = Parser::new(tokens);
    let program = parser.parse_program().map_err(|e| e.to_string())?;
    let mut lowerer = Lowerer::new();
    let hir = lowerer.lower_program(&program).map_err(|e| e.to_string())?;

    let mut interp = Interpreter::new(&hir);
    match interp.execute_program(&hir) {
        Ok(Some(v)) => Ok(v),
        Ok(None) => Ok(Value::Unit),
        Err(e) => Err(e.to_string()),
    }
}

/// 比较两个 Value 是否语义相等（Float 用近似比较）。
fn values_eq(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::Int(x, _), Value::Int(y, _)) => x == y,
        (Value::Float(x), Value::Float(y)) => (x - y).abs() < 1e-9,
        (Value::Float32(x), Value::Float32(y)) => (x - y).abs() < 1e-6,
        (Value::Bool(x), Value::Bool(y)) => x == y,
        (Value::String(x), Value::String(y)) => x == y,
        (Value::Unit, Value::Unit) => true,
        _ => false,
    }
}

/// 对同一源码断言 VM 与解释器结果一致。
fn assert_parity(src: &str) -> Value {
    let vm_res = run_vm(src).unwrap_or_else(|e| panic!("VM 执行失败: {}\n源码: {}", e, src));
    let interp_res = run_interp(src).unwrap_or_else(|e| panic!("解释器执行失败: {}\n源码: {}", e, src));
    assert!(
        values_eq(&vm_res, &interp_res),
        "VM 与解释器结果不一致\n源码: {}\nVM 结果: {:?}\n解释器结果: {:?}",
        src, vm_res, interp_res
    );
    vm_res
}

/// Lower 源码并返回 HirProgram 的 warnings 列表（用于编译期预估测试）。
fn lower_warnings(src: &str) -> Vec<TenthWarning> {
    let mut lexer = Lexer::new(src);
    let tokens = lexer.tokenize().expect("lex");
    let mut parser = Parser::new(tokens);
    let program = parser.parse_program().expect("parse");
    let mut lowerer = Lowerer::new();
    let hir = lowerer.lower_program(&program).expect("lower");
    hir.warnings
}

// ══════════════════════════════════════════════════════════════════════
// VM 缺失 17 项的 parity 测试
// ══════════════════════════════════════════════════════════════════════

#[test]
fn test_to_string_parity() {
    let v = assert_parity("to_string(42)");
    assert!(matches!(v, Value::String(ref s) if s == "42"));
}

#[test]
fn test_to_string_float_parity() {
    let v = assert_parity("to_string(3.5)");
    assert!(matches!(v, Value::String(ref s) if s == "3.5"));
}

#[test]
fn test_type_name_int_parity() {
    let v = assert_parity("type_name(42)");
    assert!(matches!(v, Value::String(ref s) if s == "int"));
}

#[test]
fn test_type_name_float_parity() {
    let v = assert_parity("type_name(3.14)");
    assert!(matches!(v, Value::String(ref s) if s == "float"));
}

#[test]
fn test_type_name_string_parity() {
    let v = assert_parity("type_name(\"hi\")");
    assert!(matches!(v, Value::String(ref s) if s == "string"));
}

#[test]
fn test_type_name_bool_parity() {
    let v = assert_parity("type_name(true)");
    assert!(matches!(v, Value::String(ref s) if s == "bool"));
}

#[test]
fn test_with_step_limit_parity() {
    // 命名函数作为闭包传递；步数预算充足时应返回函数结果
    let src = "fn make() -> Int { with_step_limit(1000000, get_42) }\nfn get_42() -> Int { 42 }\nmake()";
    let v = assert_parity(src);
    assert!(matches!(v, Value::Int(42, _)));
}

#[test]
fn test_with_timeout_ms_parity() {
    // 毫秒预算充足时应返回函数结果
    let src = "fn make() -> Int { with_timeout_ms(5000, get_42) }\nfn get_42() -> Int { 42 }\nmake()";
    let v = assert_parity(src);
    assert!(matches!(v, Value::Int(42, _)));
}

#[test]
fn test_is_timeout_unit_parity() {
    let v = assert_parity("is_timeout(())");
    assert!(matches!(v, Value::Bool(true)));
}

#[test]
fn test_is_timeout_int_parity() {
    let v = assert_parity("is_timeout(42)");
    assert!(matches!(v, Value::Bool(false)));
}

#[test]
fn test_start_grad_parity() {
    // start_grad 应与 new_grad 同义：设置 tape 后返回 Unit
    let src = "fn run() -> Int { let _ = start_grad(); 42 }\nrun()";
    let v = assert_parity(src);
    assert!(matches!(v, Value::Int(42, _)));
}

#[test]
fn test_f64_bits_parity() {
    // 1.0f64 的位模式是 0x3FF0000000000000 = 4607182418800017408
    let v = assert_parity("f64_bits(1.0)");
    assert!(matches!(v, Value::Int(n, _) if n == 4607182418800017408));
}

#[test]
fn test_f64_from_bits_parity() {
    let v = assert_parity("f64_from_bits(4607182418800017408)");
    assert!(matches!(v, Value::Float(f) if (f - 1.0).abs() < 1e-12));
}

#[test]
fn test_f64_bits_roundtrip_parity() {
    // 位往返：f64_from_bits(f64_bits(x)) == x
    let v = assert_parity("f64_from_bits(f64_bits(3.14159))");
    assert!(matches!(v, Value::Float(f) if (f - 3.14159).abs() < 1e-12));
}

#[test]
fn test_sin_parity() {
    let v = assert_parity("sin(0.0)");
    assert!(matches!(v, Value::Float(f) if f.abs() < 1e-12));
}

#[test]
fn test_cos_parity() {
    let v = assert_parity("cos(0.0)");
    assert!(matches!(v, Value::Float(f) if (f - 1.0).abs() < 1e-12));
}

#[test]
fn test_ln_parity() {
    let v = assert_parity("ln(1.0)");
    assert!(matches!(v, Value::Float(f) if f.abs() < 1e-12));
}

#[test]
fn test_ln_value_parity() {
    let v = assert_parity("ln(2.718281828459045)");
    // ln(e) ≈ 1.0
    assert!(matches!(v, Value::Float(f) if (f - 1.0).abs() < 1e-9));
}

#[test]
fn test_pow_parity() {
    let v = assert_parity("pow(2.0, 3.0)");
    assert!(matches!(v, Value::Float(f) if (f - 8.0).abs() < 1e-9));
}

#[test]
fn test_pow_fraction_parity() {
    let v = assert_parity("pow(9.0, 0.5)");
    assert!(matches!(v, Value::Float(f) if (f - 3.0).abs() < 1e-9));
}

#[test]
fn test_format_basic_parity() {
    let v = assert_parity("format(\"{} + {} = {}\", 1, 2, 3)");
    assert!(matches!(v, Value::String(ref s) if s == "1 + 2 = 3"));
}

#[test]
fn test_format_escape_parity() {
    // {{ 和 }} 应转义为 { 和 }
    let v = assert_parity("format(\"{{}}\", 1)");
    assert!(matches!(v, Value::String(ref s) if s == "{}"));
}

#[test]
fn test_format_string_arg_parity() {
    let v = assert_parity("format(\"hello, {}\", \"world\")");
    assert!(matches!(v, Value::String(ref s) if s == "hello, world"));
}

#[test]
fn test_parse_int_parity() {
    let v = assert_parity("parse_int(\"42\")");
    assert!(matches!(v, Value::Int(42, _)));
}

#[test]
fn test_parse_int_negative_parity() {
    let v = assert_parity("parse_int(\"-17\")");
    assert!(matches!(v, Value::Int(-17, _)));
}

#[test]
fn test_parse_int_invalid_parity() {
    // 解析失败返回 0（与解释器一致）
    let v = assert_parity("parse_int(\"abc\")");
    assert!(matches!(v, Value::Int(0, _)));
}

#[test]
fn test_parse_float_parity() {
    let v = assert_parity("parse_float(\"3.14\")");
    assert!(matches!(v, Value::Float(f) if (f - 3.14).abs() < 1e-9));
}

#[test]
fn test_parse_float_invalid_parity() {
    let v = assert_parity("parse_float(\"xyz\")");
    assert!(matches!(v, Value::Float(f) if f == 0.0));
}

#[test]
fn test_save_load_weights_parity() {
    // 张量列表序列化往返：save 后 load，用 numel 验证 Tensor 形状一致。
    // 注意：VM 和解释器分别用各自路径执行，写到不同文件，再各自 load 验证。
    let vm_path = std::env::temp_dir().join(format!("tenth_parity_vm_{}.bin", std::process::id()));
    let interp_path = std::env::temp_dir().join(format!("tenth_parity_interp_{}.bin", std::process::id()));

    // VM 路径：构造张量、保存、加载、用 numel 验证形状
    let vm_src = format!(
        "fn run() -> Int {{\n\
         let mut v = Vec::new();\n\
         v.push(zeros(2, 2));\n\
         let _ = save_weights(\"{}\", v);\n\
         let loaded = load_weights(\"{}\");\n\
         let t = loaded[0];\n\
         t.numel()\n\
         }}\nrun()",
        vm_path.to_string_lossy().replace('\\', "/"),
        vm_path.to_string_lossy().replace('\\', "/")
    );
    let vm_res = run_vm(&vm_src).expect("VM save/load 应成功");

    // 解释器路径：相同逻辑
    let interp_src = format!(
        "fn run() -> Int {{\n\
         let mut v = Vec::new();\n\
         v.push(zeros(2, 2));\n\
         let _ = save_weights(\"{}\", v);\n\
         let loaded = load_weights(\"{}\");\n\
         let t = loaded[0];\n\
         t.numel()\n\
         }}\nrun()",
        interp_path.to_string_lossy().replace('\\', "/"),
        interp_path.to_string_lossy().replace('\\', "/")
    );
    let interp_res = run_interp(&interp_src).expect("解释器 save/load 应成功");

    assert!(
        values_eq(&vm_res, &interp_res),
        "save_weights/load_weights VM 与解释器结果不一致\nVM: {:?}\n解释器: {:?}",
        vm_res,
        interp_res
    );
    // zeros(2,2).numel() 应为 4
    assert!(matches!(&vm_res, Value::Int(n, _) if *n == 4));

    // 清理临时文件
    let _ = std::fs::remove_file(&vm_path);
    let _ = std::fs::remove_file(&interp_path);
}

#[test]
fn test_save_load_weights_nonzero_parity() {
    // 用 ones 构造非零张量，验证 save/load 保持形状
    let vm_path = std::env::temp_dir().join(format!("tenth_parity_ones_vm_{}.bin", std::process::id()));
    let interp_path = std::env::temp_dir().join(format!("tenth_parity_ones_interp_{}.bin", std::process::id()));

    let vm_src = format!(
        "fn run() -> Int {{\n\
         let mut v = Vec::new();\n\
         v.push(ones(3, 1));\n\
         let _ = save_weights(\"{}\", v);\n\
         let loaded = load_weights(\"{}\");\n\
         let t = loaded[0];\n\
         t.numel()\n\
         }}\nrun()",
        vm_path.to_string_lossy().replace('\\', "/"),
        vm_path.to_string_lossy().replace('\\', "/")
    );
    let vm_res = run_vm(&vm_src).expect("VM save/load ones 应成功");

    let interp_src = format!(
        "fn run() -> Int {{\n\
         let mut v = Vec::new();\n\
         v.push(ones(3, 1));\n\
         let _ = save_weights(\"{}\", v);\n\
         let loaded = load_weights(\"{}\");\n\
         let t = loaded[0];\n\
         t.numel()\n\
         }}\nrun()",
        interp_path.to_string_lossy().replace('\\', "/"),
        interp_path.to_string_lossy().replace('\\', "/")
    );
    let interp_res = run_interp(&interp_src).expect("解释器 save/load ones 应成功");

    assert!(values_eq(&vm_res, &interp_res));
    // ones(3,1).numel() 应为 3
    assert!(matches!(&vm_res, Value::Int(n, _) if *n == 3));

    let _ = std::fs::remove_file(&vm_path);
    let _ = std::fs::remove_file(&interp_path);
}

// ══════════════════════════════════════════════════════════════════════
// 解释器缺失 3 项的 parity 测试
// ══════════════════════════════════════════════════════════════════════

#[test]
fn test_print_parity() {
    // print 返回 Unit；用后续表达式验证不报错
    let src = "fn run() -> Int { let _ = print(\"x\"); 42 }\nrun()";
    let v = assert_parity(src);
    assert!(matches!(v, Value::Int(42, _)));
}

#[test]
fn test_to_f64_from_int_parity() {
    let v = assert_parity("to_f64(42)");
    assert!(matches!(v, Value::Float(f) if (f - 42.0).abs() < 1e-9));
}

#[test]
fn test_to_f64_from_float_parity() {
    let v = assert_parity("to_f64(3.14)");
    assert!(matches!(v, Value::Float(f) if (f - 3.14).abs() < 1e-9));
}

#[test]
fn test_to_f32_from_int_parity() {
    let v = assert_parity("to_f32(42)");
    assert!(matches!(v, Value::Float32(f) if (f - 42.0).abs() < 1e-6));
}

#[test]
fn test_to_f32_from_float_parity() {
    let v = assert_parity("to_f32(3.14)");
    assert!(matches!(v, Value::Float32(f) if (f - 3.14).abs() < 1e-6));
}

// ══════════════════════════════════════════════════════════════════════
// AUDIT #12: bmm FLOPs 预估双侧一致性验证
// ══════════════════════════════════════════════════════════════════════
//
// Rust 母编译器（tenth/src/hir/lower/types.rs::emit_bmm_flop_estimate）
// 与 tenthc 自举编译器（tenthc/hir/lower.th::emit_bmm_flop_estimate）
// 语义对齐：两者均在 bmm 两侧 3D Known 且 B/K 匹配时，对 ≥1 GFLOP 的
// bmm 输出编译期 warning。tenthc 侧原为 no-op（HirType 缺 dim2 字段），
// 已通过新增 dim2 字段修复。
//
// 本测试验证：
// 1. Rust 母编译器对大 bmm 触发 FLOPs warning（消息格式 + 数值）
// 2. 小 bmm 不触发 warning（阈值正确）
// 3. bmm 执行 VM/解释器 parity（结果一致）
// 4. tenthc 侧源码包含非 no-op 的 emit_bmm_flop_estimate 实现

#[test]
fn test_bmm_flop_estimate_parity() {
    // ── 1. 大 bmm 触发 FLOPs warning ──
    // (4, 1024, 1024) @ (4, 1024, 1024) → B*M*K*N*2 = 4*1024^3*2 ≈ 8.59 GFLOPs
    let big_src = r#"
fn big_bmm() -> Tensor[f64, ..] {
    let a = zeros(4, 1024, 1024);
    let b = zeros(4, 1024, 1024);
    a.bmm(b)
}
"#;
    let warnings = lower_warnings(big_src);
    let bmm_warn = warnings.iter()
        .find(|w| w.message.contains("bmm") && w.message.contains("GFLOPs"))
        .unwrap_or_else(|| panic!(
            "期望 bmm GFLOPs warning，实际 warnings: {:?}",
            warnings.iter().map(|w| &w.message).collect::<Vec<_>>()
        ));
    // Rust 侧格式：{:.2} → "8.59"；tenthc 侧整数除 → "8"。两者均含 "8"。
    assert!(bmm_warn.message.contains("8"), "GFLOPs 数值应约 8，实际: {}", bmm_warn.message);
    assert!(bmm_warn.message.contains("1024"), "应包含 shape 1024，实际: {}", bmm_warn.message);
    assert!(bmm_warn.message.contains("编译期预估"), "应包含'编译期预估'，实际: {}", bmm_warn.message);

    // ── 2. 小 bmm 不触发 warning ──
    // (2, 3, 4) @ (2, 4, 5) → 2*3*4*5*2 = 240 FLOPs ≪ 1 GFLOP
    let small_src = r#"
fn small_bmm() -> Tensor[f64, ..] {
    let a = zeros(2, 3, 4);
    let b = zeros(2, 4, 5);
    a.bmm(b)
}
"#;
    let warnings = lower_warnings(small_src);
    let has_bmm_flop = warnings.iter()
        .any(|w| w.message.contains("bmm") && w.message.contains("GFLOPs"));
    assert!(!has_bmm_flop, "小 bmm (240 FLOPs) 不应触发 GFLOPs warning");

    // ── 3. bmm 执行 parity：VM 与解释器结果一致 ──
    // (2,3,4)@(2,4,5) → (2,3,5)，sum = 2*3*5*4 = 120
    let exec_src = r#"
let a = tensor[[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
               [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]].reshape(2, 3, 4);
let b = tensor[[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
               [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
               [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
               [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]].reshape(2, 4, 5);
let c = a.bmm(b);
c.sum()
"#;
    let v = assert_parity(exec_src);
    assert!(matches!(v, Value::Float(f) if (f - 120.0).abs() < 1e-6),
        "bmm sum 应为 120.0，实际: {:?}", v);

    // ── 4. tenthc 侧源码验证：emit_bmm_flop_estimate 不再是 no-op ──
    // 通过检查 tenthc 源码确认 dim2 字段和完整实现已就位。
    // 运行时 parity 由 selfhost_frontend.rs（源码可 lower）+
    // three_stage.rs（tenthc 可编译执行）覆盖。
    let tenthc_lower_src = include_str!("../../tenthc/hir/lower.th");
    assert!(
        tenthc_lower_src.contains("fn emit_bmm_flop_estimate"),
        "tenthc 应包含 emit_bmm_flop_estimate 函数定义"
    );
    assert!(
        tenthc_lower_src.contains("rt.dim_count != 3 || at.dim_count != 3"),
        "tenthc emit_bmm_flop_estimate 应检查 3D shape（不再是 no-op）"
    );
    assert!(
        tenthc_lower_src.contains("let r_d2 = get_tensor_dim2"),
        "tenthc bmm 分支应读取 receiver 的 dim2"
    );
    assert!(
        tenthc_lower_src.contains("let a_d2 = get_tensor_dim2"),
        "tenthc bmm 分支应读取 argument 的 dim2"
    );

    let tenthc_hir_src = include_str!("../../tenthc/hir/hir.th");
    assert!(
        tenthc_hir_src.contains("dim2: i64"),
        "tenthc HirType 应包含 dim2 字段（AUDIT #12 根因修复）"
    );
}
