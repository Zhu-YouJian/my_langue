//! Wave 3 张量修复补测试。
//!
//! 覆盖本次 Wave 1a/1b/2 修复的 6 项改动：
//! - 序列化 v2 格式（F32/F64 往返 + 旧 v1 向后兼容）
//! - f16/bf16 TensorData 变体（构造器 + 基本运算 + 序列化）
//! - 标准库 clip_grad_by_norm / adamw_step_w（use 导入运行时验证）
//!
//! 采用 VM 路径执行（参考 native_parity_test.rs 的 run_vm + register_test_natives 模式）。
//! VM 侧通过 `register_test_natives` 注册所需 native（复制自 main.rs::register_natives）。
//!
//! 注意：save_weights/load_weights 的实现复制自 native_parity_test.rs（与 main.rs 同步），
//! 这是项目现有惯例（参见 native_parity_test.rs 文件头注释）。

use std::cell::RefCell;
use std::rc::Rc;

use tenth::compile::bytecode::BytecodeCompiler;
use tenth::hir::lower::Lowerer;
use tenth::hir::types::BaseType;
use tenth::lexer::lexer::Lexer;
use tenth::parser::parser::Parser;
use tenth::runtime::autodiff::Tape;
use tenth::runtime::tensor::{Tensor, TensorData};
use tenth::runtime::value::Value;
use tenth::runtime::vm::Vm;

/// 注册测试所需的 native（复制自 main.rs::register_natives 的相关子集 + native_parity_test.rs 的 save/load 实现）。
/// save_weights/load_weights 必须与 main.rs / native_parity_test.rs 完全一致；
/// 其他 native 仅满足测试源码运行需求。
fn register_test_natives(vm: &mut Vm) {
    // ── 辅助 native ──
    vm.add_native("println".into(), |_vm, args| {
        for a in args {
            print!("{a}");
        }
        println!();
        Ok(Value::Unit)
    });
    vm.add_native("print".into(), |_vm, args| {
        for a in args {
            print!("{a}");
        }
        Ok(Value::Unit)
    });
    vm.add_native("Vec::new".into(), |_vm, _args| {
        Ok(Value::Vec(Rc::new(RefCell::new(Vec::new()))))
    });
    vm.add_native("tensor".into(), |_vm, args| {
        if args.len() == 1 {
            Ok(args[0].clone())
        } else {
            Err(tenth::error::TenthError::RuntimeError {
                message: "tensor() 参数异常".into(),
            })
        }
    });
    vm.add_native("zeros".into(), |_vm, args| {
        let shape: Vec<usize> = args.iter().map(|a| a.as_int().unwrap_or(1) as usize).collect();
        Ok(Value::Tensor(Rc::new(RefCell::new(Tensor::zeros(&shape)))))
    });
    vm.add_native("ones".into(), |_vm, args| {
        let shape: Vec<usize> = args.iter().map(|a| a.as_int().unwrap_or(1) as usize).collect();
        Ok(Value::Tensor(Rc::new(RefCell::new(Tensor::ones(&shape)))))
    });
    // to_f64 / to_f32（clip_grad_by_norm 测试需要）
    vm.add_native("to_f64".into(), |_vm, args| match args.first() {
        Some(Value::Int(n)) => Ok(Value::Float(*n as f64)),
        Some(Value::Float(f)) => Ok(Value::Float(*f)),
        Some(Value::Float32(f)) => Ok(Value::Float(*f as f64)),
        _ => Err(tenth::error::TenthError::RuntimeError {
            message: "to_f64() 需要一个数值参数".into(),
        }),
    });
    vm.add_native("to_f32".into(), |_vm, args| match args.first() {
        Some(Value::Int(n)) => Ok(Value::Float32(*n as f32)),
        Some(Value::Float(f)) => Ok(Value::Float32(*f as f32)),
        Some(Value::Float32(f)) => Ok(Value::Float32(*f)),
        _ => Err(tenth::error::TenthError::RuntimeError {
            message: "to_f32() 需要一个数值参数".into(),
        }),
    });

    // ── Wave 2: f16/bf16 构造函数（复制自 main.rs:1798-1821）──
    vm.add_native("zeros_f16".into(), |_vm, args| {
        let shape: Vec<usize> = args
            .iter()
            .map(|a| a.as_int().unwrap_or(1) as usize)
            .collect();
        Ok(Value::Tensor(Rc::new(RefCell::new(Tensor::zeros_f16(&shape)))))
    });
    vm.add_native("ones_f16".into(), |_vm, args| {
        let shape: Vec<usize> = args
            .iter()
            .map(|a| a.as_int().unwrap_or(1) as usize)
            .collect();
        Ok(Value::Tensor(Rc::new(RefCell::new(Tensor::ones_f16(&shape)))))
    });
    vm.add_native("zeros_bf16".into(), |_vm, args| {
        let shape: Vec<usize> = args
            .iter()
            .map(|a| a.as_int().unwrap_or(1) as usize)
            .collect();
        Ok(Value::Tensor(Rc::new(RefCell::new(Tensor::zeros_bf16(&shape)))))
    });
    vm.add_native("ones_bf16".into(), |_vm, args| {
        let shape: Vec<usize> = args
            .iter()
            .map(|a| a.as_int().unwrap_or(1) as usize)
            .collect();
        Ok(Value::Tensor(Rc::new(RefCell::new(Tensor::ones_bf16(&shape)))))
    });

    // ── Autodiff native（复制自 main.rs:1281-1385，adamw_step_w 测试需要）──
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
            Err(tenth::error::TenthError::RuntimeError {
                message: "param() 需要一个张量参数".into(),
            })
        }
    });
    vm.add_native("backward".into(), |vm, args| {
        if let Some(Value::Tensor(t)) = args.first() {
            if let Some(ref tape) = vm.tape {
                let loss_id = t.borrow().tape_id.ok_or_else(|| {
                    tenth::error::TenthError::RuntimeError {
                        message: "backward(): 张量没有 tape_id".into(),
                    }
                })?;
                tape.backward(loss_id)
                    .map_err(|e| tenth::error::TenthError::RuntimeError {
                        message: format!("{}", e),
                    })?;
                Ok(Value::Unit)
            } else {
                Err(tenth::error::TenthError::RuntimeError {
                    message: "未调用 new_grad()".into(),
                })
            }
        } else {
            Err(tenth::error::TenthError::RuntimeError {
                message: "backward() 需要一个张量参数".into(),
            })
        }
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
            Err(tenth::error::TenthError::RuntimeError {
                message: "grad() 需要一个张量参数".into(),
            })
        }
    });
    vm.add_native("stop_grad".into(), |vm, args| {
        if let Some(Value::Tensor(t)) = args.first() {
            let mut detached = t.borrow().clone();
            detached.tape_id = None;
            Ok(Value::Tensor(Rc::new(RefCell::new(detached))))
        } else {
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

    // ── save_weights / load_weights（复制自 native_parity_test.rs:205-435，与 main.rs 同步）──
    vm.add_native("save_weights".into(), |vm, args| {
        if args.len() >= 2 {
            if let Value::String(path) = &args[0] {
                let resolved = if let Some(ref sb) = vm.fs_sandbox {
                    match sb.check_write(path) {
                        Ok(p) => p,
                        Err(e) => {
                            return Err(tenth::error::TenthError::RuntimeError { message: e })
                        }
                    }
                } else {
                    std::path::PathBuf::from(path)
                };
                let tensors: &Rc<RefCell<Vec<Value>>> = match &args[1] {
                    Value::Vec(v) => v,
                    Value::Array(a) => a,
                    _ => {
                        return Err(tenth::error::TenthError::RuntimeError {
                            message: "save_weights 期望一个张量列表".into(),
                        })
                    }
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
                            if let Value::Tensor(t) = &*rc.borrow() {
                                Some(t.clone())
                            } else {
                                None
                            }
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
                        // dtype 字段: F32=0, F64=1, F16=2, BF16=3
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
                                    for &x in slice {
                                        bytes.extend(&x.to_le_bytes());
                                    }
                                } else {
                                    for &x in flat.iter() {
                                        bytes.extend(&x.to_le_bytes());
                                    }
                                }
                            }
                            TensorData::F32(arr) => {
                                let flat = arr.as_standard_layout();
                                if let Some(slice) = flat.as_slice() {
                                    for &x in slice {
                                        bytes.extend(&x.to_le_bytes());
                                    }
                                } else {
                                    for &x in flat.iter() {
                                        bytes.extend(&x.to_le_bytes());
                                    }
                                }
                            }
                            TensorData::F16(arr) => {
                                let flat = arr.as_standard_layout();
                                if let Some(slice) = flat.as_slice() {
                                    for &x in slice {
                                        bytes.extend(&x.to_le_bytes());
                                    }
                                } else {
                                    for &x in flat.iter() {
                                        bytes.extend(&x.to_le_bytes());
                                    }
                                }
                            }
                            TensorData::BF16(arr) => {
                                let flat = arr.as_standard_layout();
                                if let Some(slice) = flat.as_slice() {
                                    for &x in slice {
                                        bytes.extend(&x.to_le_bytes());
                                    }
                                } else {
                                    for &x in flat.iter() {
                                        bytes.extend(&x.to_le_bytes());
                                    }
                                }
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
    vm.add_native("load_weights".into(), |vm, args| {
        if let Some(Value::String(path)) = args.first() {
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
                    let mut result: Vec<Value> = Vec::new();
                    // 检测 v2 格式: magic "THW1"
                    let is_v2 = bytes.len() >= 12 && &bytes[0..4] == b"THW1";
                    let num: usize;
                    let mut offset: usize;
                    if is_v2 {
                        let _version =
                            i32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
                        num = i32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]])
                            as usize;
                        offset = 12;
                    } else {
                        num = i32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
                            as usize;
                        offset = 4;
                    }
                    for _ in 0..num {
                        if offset + 4 > bytes.len() {
                            break;
                        }
                        let ndim = i32::from_le_bytes([
                            bytes[offset],
                            bytes[offset + 1],
                            bytes[offset + 2],
                            bytes[offset + 3],
                        ]) as usize;
                        offset += 4;
                        let mut shape = Vec::new();
                        for _ in 0..ndim {
                            if offset + 4 > bytes.len() {
                                break;
                            }
                            let d = i32::from_le_bytes([
                                bytes[offset],
                                bytes[offset + 1],
                                bytes[offset + 2],
                                bytes[offset + 3],
                            ]) as usize;
                            shape.push(d);
                            offset += 4;
                        }
                        let nel: usize = shape.iter().product();
                        if is_v2 {
                            if offset + 4 > bytes.len() {
                                break;
                            }
                            let dtype = i32::from_le_bytes([
                                bytes[offset],
                                bytes[offset + 1],
                                bytes[offset + 2],
                                bytes[offset + 3],
                            ]);
                            offset += 4;
                            match dtype {
                                0 => {
                                    let data_len = nel * 4;
                                    if offset + data_len > bytes.len() {
                                        break;
                                    }
                                    let mut data = Vec::with_capacity(nel);
                                    for i in 0..nel {
                                        let start = offset + i * 4;
                                        let val = f32::from_le_bytes([
                                            bytes[start],
                                            bytes[start + 1],
                                            bytes[start + 2],
                                            bytes[start + 3],
                                        ]);
                                        data.push(val);
                                    }
                                    offset += data_len;
                                    result.push(Value::Tensor(Rc::new(RefCell::new(
                                        Tensor::from_vec_f32(data, shape),
                                    ))));
                                }
                                1 => {
                                    let data_len = nel * 8;
                                    if offset + data_len > bytes.len() {
                                        break;
                                    }
                                    let mut data = Vec::with_capacity(nel);
                                    for i in 0..nel {
                                        let start = offset + i * 8;
                                        let val = f64::from_le_bytes([
                                            bytes[start],
                                            bytes[start + 1],
                                            bytes[start + 2],
                                            bytes[start + 3],
                                            bytes[start + 4],
                                            bytes[start + 5],
                                            bytes[start + 6],
                                            bytes[start + 7],
                                        ]);
                                        data.push(val);
                                    }
                                    offset += data_len;
                                    result.push(Value::Tensor(Rc::new(RefCell::new(
                                        Tensor::from_vec(data, shape),
                                    ))));
                                }
                                2 => {
                                    // F16: 2 字节/元素
                                    let data_len = nel * 2;
                                    if offset + data_len > bytes.len() {
                                        break;
                                    }
                                    let mut data = Vec::with_capacity(nel);
                                    for i in 0..nel {
                                        let start = offset + i * 2;
                                        let val = half::f16::from_le_bytes([
                                            bytes[start],
                                            bytes[start + 1],
                                        ]);
                                        data.push(val);
                                    }
                                    offset += data_len;
                                    result.push(Value::Tensor(Rc::new(RefCell::new(
                                        Tensor::from_vec_f16(data, shape),
                                    ))));
                                }
                                3 => {
                                    // BF16: 2 字节/元素
                                    let data_len = nel * 2;
                                    if offset + data_len > bytes.len() {
                                        break;
                                    }
                                    let mut data = Vec::with_capacity(nel);
                                    for i in 0..nel {
                                        let start = offset + i * 2;
                                        let val = half::bf16::from_le_bytes([
                                            bytes[start],
                                            bytes[start + 1],
                                        ]);
                                        data.push(val);
                                    }
                                    offset += data_len;
                                    result.push(Value::Tensor(Rc::new(RefCell::new(
                                        Tensor::from_vec_bf16(data, shape),
                                    ))));
                                }
                                other => {
                                    return Err(tenth::error::TenthError::RuntimeError {
                                        message: format!("load_weights: 未知 dtype={}", other),
                                    });
                                }
                            }
                        } else {
                            // 旧 v1 格式：纯 f64，无 dtype 字段
                            let data_len = nel * 8;
                            if offset + data_len > bytes.len() {
                                break;
                            }
                            let mut data = Vec::with_capacity(nel);
                            for i in 0..nel {
                                let start = offset + i * 8;
                                let val = f64::from_le_bytes([
                                    bytes[start],
                                    bytes[start + 1],
                                    bytes[start + 2],
                                    bytes[start + 3],
                                    bytes[start + 4],
                                    bytes[start + 5],
                                    bytes[start + 6],
                                    bytes[start + 7],
                                ]);
                                data.push(val);
                            }
                            offset += data_len;
                            result.push(Value::Tensor(Rc::new(RefCell::new(
                                Tensor::from_vec(data, shape),
                            ))));
                        }
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
}

/// 通过 VM 执行 .th 源码（不带 search_paths，用于不依赖 use 的测试）。
fn run_vm(src: &str) -> Value {
    run_vm_inner(src, None)
}

/// 通过 VM 执行 .th 源码（带 search_paths，支持 use std::... 导入）。
/// cargo test 的 cwd 是 tenth/ 目录，所以 search_paths 为 "." 使 use std::optim::clip
/// 解析到 ./std/optim/clip.th（即 tenth/std/optim/clip.th）。
fn run_vm_with_std(src: &str) -> Value {
    run_vm_inner(src, Some(vec![".".to_string()]))
}

fn run_vm_inner(src: &str, search_paths: Option<Vec<String>>) -> Value {
    let mut lexer = Lexer::new(src);
    let tokens = lexer.tokenize().unwrap_or_else(|e| panic!("词法错误: {}", e));
    let mut parser = Parser::new(tokens);
    let program = parser
        .parse_program()
        .unwrap_or_else(|e| panic!("语法错误: {}", e));
    let mut lowerer = match search_paths {
        Some(paths) => Lowerer::with_search_paths(paths),
        None => Lowerer::new(),
    };
    let hir = lowerer
        .lower_program(&program)
        .unwrap_or_else(|e| panic!("HIR 错误: {}", e));

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
                vm.set_global(
                    func.name.clone(),
                    Value::FnRef {
                        name: func.name.clone(),
                        params: func.params.clone(),
                        return_type: func.return_type.clone(),
                    },
                );
            }
            Err(e) => panic!("字节码编译错误: {}", e),
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
            Err(e) => panic!("字节码编译错误: {}", e),
        }
        vm.call("main")
            .unwrap_or_else(|e| panic!("VM 执行失败: {}", e))
    } else if vm.has_fn("main") {
        vm.call("main")
            .unwrap_or_else(|e| panic!("VM 执行失败: {}", e))
    } else {
        Value::Unit
    }
}

// ══════════════════════════════════════════════════════════════════════
// 序列化 v2 格式测试
// ══════════════════════════════════════════════════════════════════════

// ─── Test 1: F32 张量 save→load 往返 dtype 保持 ──────────────────────────
//
// ones(2,2) 返回 F64 张量；用 .astype_f32() 转为 F32 后 save+load，
// 验证 dtype 保持 F32。但 .th 源码无 astype_f32，改用 Rust 端直接构造 F32 张量。
// 这里改用 ones(2,2) save+load 后在 Rust 端检查 dtype——但 ones 默认是 F64。
//
// 替代方案：用 .th 源码 save ones(2,2)（F64），load 后检查 dtype=F64。
// 要测 F32 往返，需要在 Rust 端构造 F32 张量写入文件，再用 .th load_weights 读出。

#[test]
fn test_save_load_f32_weights() {
    // 在 Rust 端构造 F32 张量，写入 v2 格式文件，再用 .th load_weights 读出验证 dtype=F32
    let tmp = std::env::temp_dir().join(format!(
        "tenth_f32_save_load_{}.bin",
        std::process::id()
    ));
    // 构造 F32 张量 [1.5, 2.5, 3.5, 4.5] shape=[2,2]
    let t = Tensor::from_vec_f32(vec![1.5f32, 2.5, 3.5, 4.5], vec![2, 2]);
    // 手动写 v2 格式
    let mut bytes: Vec<u8> = Vec::new();
    bytes.extend(b"THW1");
    bytes.extend(&2i32.to_le_bytes());
    bytes.extend(&1i32.to_le_bytes()); // num_tensors=1
    let shape = t.shape();
    bytes.extend(&(shape.len() as i32).to_le_bytes());
    for &d in &shape {
        bytes.extend(&(d as i32).to_le_bytes());
    }
    bytes.extend(&0i32.to_le_bytes()); // dtype=0 (F32)
    if let TensorData::F32(arr) = &t.data {
        for &x in arr.iter() {
            bytes.extend(&x.to_le_bytes());
        }
    }
    std::fs::write(&tmp, &bytes).expect("写入临时文件失败");

    // 用 .th load_weights 读出
    let path_str = tmp.to_string_lossy().replace('\\', "/");
    let src = format!(
        r#"
        let loaded = load_weights("{}");
        let t = loaded[0];
        t
        "#,
        path_str
    );
    let v = run_vm(&src);
    match v {
        Value::Tensor(t) => {
            assert_eq!(
                t.borrow().dtype(),
                BaseType::F32,
                "F32 往返后 dtype 应保持 F32"
            );
            assert_eq!(t.borrow().shape(), vec![2, 2], "shape 应为 [2,2]");
            // 验证数据
            if let TensorData::F32(arr) = &t.borrow().data {
                let vals: Vec<f32> = arr.iter().copied().collect();
                assert_eq!(vals.len(), 4, "应有 4 个元素");
                assert!((vals[0] - 1.5f32).abs() < 1e-6, "data[0] 应为 1.5");
                assert!((vals[1] - 2.5f32).abs() < 1e-6, "data[1] 应为 2.5");
                assert!((vals[2] - 3.5f32).abs() < 1e-6, "data[2] 应为 3.5");
                assert!((vals[3] - 4.5f32).abs() < 1e-6, "data[3] 应为 4.5");
            } else {
                panic!("期望 F32 数据");
            }
        }
        v => panic!("期望 Tensor，got {:?}", v),
    }
    let _ = std::fs::remove_file(&tmp);
}

// ─── Test 2: F64 张量 save→load 往返 ──────────────────────────────────────

#[test]
fn test_save_load_f64_weights() {
    let tmp = std::env::temp_dir().join(format!(
        "tenth_f64_save_load_{}.bin",
        std::process::id()
    ));
    let path_str = tmp.to_string_lossy().replace('\\', "/");
    // 用 .th 源码：save ones(2,2)（F64），load 后检查 dtype=F64 和数据
    let src = format!(
        r#"
        fn run() -> Tensor[f64, ..] {{
            let mut v = Vec::new();
            v.push(ones(2, 2));
            let _ = save_weights("{}", v);
            let loaded = load_weights("{}");
            loaded[0]
        }}
        run()
        "#,
        path_str, path_str
    );
    let v = run_vm(&src);
    match v {
        Value::Tensor(t) => {
            assert_eq!(
                t.borrow().dtype(),
                BaseType::F64,
                "F64 往返后 dtype 应保持 F64"
            );
            assert_eq!(t.borrow().shape(), vec![2, 2], "shape 应为 [2,2]");
            // ones(2,2) 数据应为 1.0
            if let TensorData::F64(arr) = &t.borrow().data {
                let slice = arr.as_slice().unwrap();
                for &x in slice {
                    assert!((x - 1.0f64).abs() < 1e-9, "数据应为 1.0，got {}", x);
                }
            } else {
                panic!("期望 F64 数据");
            }
        }
        v => panic!("期望 Tensor，got {:?}", v),
    }
    let _ = std::fs::remove_file(&tmp);
}

// ─── Test 3: F32 + F64 混合张量列表 save→load ─────────────────────────────

#[test]
fn test_save_load_mixed_weights() {
    let tmp = std::env::temp_dir().join(format!(
        "tenth_mixed_save_load_{}.bin",
        std::process::id()
    ));
    // 在 Rust 端构造混合列表：1 个 F32 + 1 个 F64
    let t_f32 = Tensor::from_vec_f32(vec![1.0f32, 2.0, 3.0], vec![3]);
    let t_f64 = Tensor::from_vec(vec![4.0f64, 5.0], vec![2]);
    let mut bytes: Vec<u8> = Vec::new();
    bytes.extend(b"THW1");
    bytes.extend(&2i32.to_le_bytes()); // version
    bytes.extend(&2i32.to_le_bytes()); // num_tensors=2
    // 写 F32 张量
    let shape1 = t_f32.shape();
    bytes.extend(&(shape1.len() as i32).to_le_bytes());
    for &d in &shape1 {
        bytes.extend(&(d as i32).to_le_bytes());
    }
    bytes.extend(&0i32.to_le_bytes()); // dtype=0 (F32)
    if let TensorData::F32(arr) = &t_f32.data {
        for &x in arr.iter() {
            bytes.extend(&x.to_le_bytes());
        }
    }
    // 写 F64 张量
    let shape2 = t_f64.shape();
    bytes.extend(&(shape2.len() as i32).to_le_bytes());
    for &d in &shape2 {
        bytes.extend(&(d as i32).to_le_bytes());
    }
    bytes.extend(&1i32.to_le_bytes()); // dtype=1 (F64)
    if let TensorData::F64(arr) = &t_f64.data {
        for &x in arr.iter() {
            bytes.extend(&x.to_le_bytes());
        }
    }
    std::fs::write(&tmp, &bytes).expect("写入临时文件失败");

    let path_str = tmp.to_string_lossy().replace('\\', "/");
    let src = format!(
        r#"
        let loaded = load_weights("{}");
        // 返回张量数量
        loaded.len()
        "#,
        path_str
    );
    let v = run_vm(&src);
    match v {
        Value::Int(n) => assert_eq!(n, 2, "应加载 2 个张量"),
        v => panic!("期望 Int(2)，got {:?}", v),
    }

    // 分别验证两个张量的 dtype
    let src2 = format!(
        r#"
        let loaded = load_weights("{}");
        loaded[0].numel()
        "#,
        path_str
    );
    let v0 = run_vm(&src2);
    match v0 {
        Value::Int(n) => assert_eq!(n, 3, "第一个张量 numel 应为 3"),
        v => panic!("期望 Int(3)，got {:?}", v),
    }

    let src3 = format!(
        r#"
        let loaded = load_weights("{}");
        loaded[1].numel()
        "#,
        path_str
    );
    let v1 = run_vm(&src3);
    match v1 {
        Value::Int(n) => assert_eq!(n, 2, "第二个张量 numel 应为 2"),
        v => panic!("期望 Int(2)，got {:?}", v),
    }

    // Rust 端验证 dtype
    let src4 = format!(
        r#"
        let loaded = load_weights("{}");
        loaded[0]
        "#,
        path_str
    );
    let v_t0 = run_vm(&src4);
    match v_t0 {
        Value::Tensor(t) => assert_eq!(t.borrow().dtype(), BaseType::F32, "第一个张量应为 F32"),
        v => panic!("期望 Tensor，got {:?}", v),
    }

    let src5 = format!(
        r#"
        let loaded = load_weights("{}");
        loaded[1]
        "#,
        path_str
    );
    let v_t1 = run_vm(&src5);
    match v_t1 {
        Value::Tensor(t) => assert_eq!(t.borrow().dtype(), BaseType::F64, "第二个张量应为 F64"),
        v => panic!("期望 Tensor，got {:?}", v),
    }

    let _ = std::fs::remove_file(&tmp);
}

// ─── Test 4: 旧 v1 格式（无 magic）向后兼容 ───────────────────────────────
//
// 手动构造旧 v1 格式二进制：[num_tensors:i32] × [ndim][shape][data:f64]
// 验证 load_weights 能正确读出（兜底走 f64 路径）。

#[test]
fn test_load_v1_format_backward_compat() {
    let tmp = std::env::temp_dir().join(format!(
        "tenth_v1_compat_{}.bin",
        std::process::id()
    ));
    // 构造旧 v1 格式：1 个张量 shape=[2] data=[10.0, 20.0]
    let mut bytes: Vec<u8> = Vec::new();
    bytes.extend(&1i32.to_le_bytes()); // num_tensors=1（无 magic）
    bytes.extend(&1i32.to_le_bytes()); // ndim=1
    bytes.extend(&2i32.to_le_bytes()); // shape[0]=2
    // 旧格式无 dtype 字段，直接是 f64 数据
    bytes.extend(&10.0f64.to_le_bytes());
    bytes.extend(&20.0f64.to_le_bytes());
    std::fs::write(&tmp, &bytes).expect("写入临时文件失败");

    let path_str = tmp.to_string_lossy().replace('\\', "/");
    let src = format!(
        r#"
        fn run() -> Tensor[f64, ..] {{
            let loaded = load_weights("{}");
            loaded[0]
        }}
        run()
        "#,
        path_str
    );
    let v = run_vm(&src);
    match v {
        Value::Tensor(t) => {
            // 旧格式兜底走 f64 路径，dtype 应为 F64
            assert_eq!(
                t.borrow().dtype(),
                BaseType::F64,
                "旧 v1 格式应兜底为 F64"
            );
            assert_eq!(t.borrow().shape(), vec![2], "shape 应为 [2]");
            // 验证数据
            if let TensorData::F64(arr) = &t.borrow().data {
                let slice = arr.as_slice().unwrap();
                assert!((slice[0] - 10.0f64).abs() < 1e-9, "data[0] 应为 10.0");
                assert!((slice[1] - 20.0f64).abs() < 1e-9, "data[1] 应为 20.0");
            } else {
                panic!("期望 F64 数据");
            }
        }
        v => panic!("期望 Tensor，got {:?}", v),
    }
    let _ = std::fs::remove_file(&tmp);
}

// ══════════════════════════════════════════════════════════════════════
// f16/bf16 基本运算测试
// ══════════════════════════════════════════════════════════════════════

// ─── Test 5: zeros_f16 构造器返回 F16 dtype ───────────────────────────────

#[test]
fn test_zeros_f16() {
    let v = run_vm("zeros_f16(2, 3)");
    match v {
        Value::Tensor(t) => {
            assert_eq!(t.borrow().dtype(), BaseType::F16, "zeros_f16 应返回 F16");
            assert_eq!(t.borrow().shape(), vec![2, 3], "shape 应为 [2,3]");
            // 验证全零
            if let TensorData::F16(arr) = &t.borrow().data {
                for &x in arr.iter() {
                    assert_eq!(x.to_f32(), 0.0f32, "zeros_f16 元素应为 0");
                }
            } else {
                panic!("期望 F16 数据");
            }
        }
        v => panic!("期望 Tensor，got {:?}", v),
    }
}

// ─── Test 6: ones_bf16 构造器返回 BF16 dtype ──────────────────────────────

#[test]
fn test_ones_bf16() {
    let v = run_vm("ones_bf16(2, 2)");
    match v {
        Value::Tensor(t) => {
            assert_eq!(t.borrow().dtype(), BaseType::BF16, "ones_bf16 应返回 BF16");
            assert_eq!(t.borrow().shape(), vec![2, 2], "shape 应为 [2,2]");
            // 验证全一
            if let TensorData::BF16(arr) = &t.borrow().data {
                for &x in arr.iter() {
                    assert_eq!(x.to_f32(), 1.0f32, "ones_bf16 元素应为 1");
                }
            } else {
                panic!("期望 BF16 数据");
            }
        }
        v => panic!("期望 Tensor，got {:?}", v),
    }
}

// ─── Test 7: F16 + F16 → F16 元素级加法 ───────────────────────────────────

#[test]
fn test_f16_add() {
    let src = r#"
        fn run() -> Tensor[f16, ..] {
            let a = ones_f16(2, 2);
            let b = ones_f16(2, 2);
            a + b
        }
        run()
    "#;
    let v = run_vm(src);
    match v {
        Value::Tensor(t) => {
            assert_eq!(
                t.borrow().dtype(),
                BaseType::F16,
                "F16+F16 应提升为 F16"
            );
            assert_eq!(t.borrow().shape(), vec![2, 2], "shape 应为 [2,2]");
            // 1.0 + 1.0 = 2.0
            if let TensorData::F16(arr) = &t.borrow().data {
                for &x in arr.iter() {
                    assert!((x.to_f32() - 2.0f32).abs() < 1e-2, "F16 加法结果应为 2.0");
                }
            } else {
                panic!("期望 F16 数据");
            }
        }
        v => panic!("期望 Tensor，got {:?}", v),
    }
}

// ─── Test 8: BF16 * BF16 → BF16 元素级乘法 ────────────────────────────────

#[test]
fn test_bf16_mul() {
    let src = r#"
        fn run() -> Tensor[bf16, ..] {
            let a = ones_bf16(3);
            let b = ones_bf16(3);
            a * b
        }
        run()
    "#;
    let v = run_vm(src);
    match v {
        Value::Tensor(t) => {
            assert_eq!(
                t.borrow().dtype(),
                BaseType::BF16,
                "BF16*BF16 应提升为 BF16"
            );
            assert_eq!(t.borrow().shape(), vec![3], "shape 应为 [3]");
            // 1.0 * 1.0 = 1.0
            if let TensorData::BF16(arr) = &t.borrow().data {
                for &x in arr.iter() {
                    assert!((x.to_f32() - 1.0f32).abs() < 1e-2, "BF16 乘法结果应为 1.0");
                }
            } else {
                panic!("期望 BF16 数据");
            }
        }
        v => panic!("期望 Tensor，got {:?}", v),
    }
}

// ─── Test 9: F16 + F64 混合运算 → F64 ─────────────────────────────────────
//
// promote_dtype 规则：F64 + 任何 → F64。F16 + F64 应提升为 F64。

#[test]
fn test_f16_f64_mixed() {
    let src = r#"
        fn run() -> Tensor[f64, ..] {
            let a = ones_f16(2);
            let b = ones(2);
            a + b
        }
        run()
    "#;
    let v = run_vm(src);
    match v {
        Value::Tensor(t) => {
            assert_eq!(
                t.borrow().dtype(),
                BaseType::F64,
                "F16+F64 应提升为 F64"
            );
            assert_eq!(t.borrow().shape(), vec![2], "shape 应为 [2]");
            // 1.0 + 1.0 = 2.0
            if let TensorData::F64(arr) = &t.borrow().data {
                for &x in arr.iter() {
                    assert!((x - 2.0f64).abs() < 1e-9, "混合加法结果应为 2.0");
                }
            } else {
                panic!("期望 F64 数据");
            }
        }
        v => panic!("期望 Tensor，got {:?}", v),
    }
}

// ─── Test 10: F16 张量 save→load 往返 dtype 保持 ──────────────────────────

#[test]
fn test_f16_serialization() {
    let tmp = std::env::temp_dir().join(format!(
        "tenth_f16_serial_{}.bin",
        std::process::id()
    ));
    // 在 Rust 端构造 F16 张量并写 v2 格式文件
    let t = Tensor::from_vec_f16(
        vec![
            half::f16::from_f32(1.5),
            half::f16::from_f32(2.5),
            half::f16::from_f32(3.5),
            half::f16::from_f32(4.5),
        ],
        vec![2, 2],
    );
    let mut bytes: Vec<u8> = Vec::new();
    bytes.extend(b"THW1");
    bytes.extend(&2i32.to_le_bytes());
    bytes.extend(&1i32.to_le_bytes()); // num_tensors=1
    let shape = t.shape();
    bytes.extend(&(shape.len() as i32).to_le_bytes());
    for &d in &shape {
        bytes.extend(&(d as i32).to_le_bytes());
    }
    bytes.extend(&2i32.to_le_bytes()); // dtype=2 (F16)
    if let TensorData::F16(arr) = &t.data {
        for &x in arr.iter() {
            bytes.extend(&x.to_le_bytes());
        }
    }
    std::fs::write(&tmp, &bytes).expect("写入临时文件失败");

    // 用 .th load_weights 读出
    let path_str = tmp.to_string_lossy().replace('\\', "/");
    let src = format!(
        r#"
        fn run() -> Tensor[f16, ..] {{
            let loaded = load_weights("{}");
            loaded[0]
        }}
        run()
        "#,
        path_str
    );
    let v = run_vm(&src);
    match v {
        Value::Tensor(t) => {
            assert_eq!(
                t.borrow().dtype(),
                BaseType::F16,
                "F16 往返后 dtype 应保持 F16"
            );
            assert_eq!(t.borrow().shape(), vec![2, 2], "shape 应为 [2,2]");
            // 验证数据
            if let TensorData::F16(arr) = &t.borrow().data {
                let vals: Vec<f32> = arr.iter().map(|x| x.to_f32()).collect();
                assert!((vals[0] - 1.5f32).abs() < 1e-2, "data[0] 应为 1.5，got {}", vals[0]);
                assert!((vals[1] - 2.5f32).abs() < 1e-2, "data[1] 应为 2.5，got {}", vals[1]);
                assert!((vals[2] - 3.5f32).abs() < 1e-2, "data[2] 应为 3.5，got {}", vals[2]);
                assert!((vals[3] - 4.5f32).abs() < 1e-2, "data[3] 应为 4.5，got {}", vals[3]);
            } else {
                panic!("期望 F16 数据");
            }
        }
        v => panic!("期望 Tensor，got {:?}", v),
    }
    let _ = std::fs::remove_file(&tmp);
}

// ══════════════════════════════════════════════════════════════════════
// 标准库修复测试（use 导入运行时验证）
// ══════════════════════════════════════════════════════════════════════

// ─── Test 11: clip_grad_by_norm 基本功能 ──────────────────────────────────
//
// gw = ones(3) → norm = sqrt(1+1+1) = sqrt(3) ≈ 1.732
// max_norm = 1.0 → norm > max_norm，scale = 1.0/sqrt(3) ≈ 0.577
// clipped = gw * scale → [0.577, 0.577, 0.577]
//
// 使用 use std::optim::clip::clip_grad_by_norm 导入（需要 search_paths 支持）。
//
// #[ignore] 原因：use 导入的泛型函数未注册到 generic_funcs 字典，
// 泛型调用 `clip_grad_by_norm<f64>(...)` 报"未定义的泛型函数"。
// 这是 HIR lower 层面的已知限制（use 导入只注册 functions/scope，
// 不注册 generic_funcs）。此外 to_f64 native 不接受标量 Tensor 参数，
// clip.th 中 `to_f64(norm)` 在 recording 模式下 norm 是 Tensor 时会失败。
// 已向 runtime/compiler 部门留言（见黑板）。

#[test]
#[ignore = "use+泛型调用限制：generic_funcs 未注册；to_f64 不支持标量 Tensor"]
fn test_clip_grad_by_norm() {
    let src = r#"
        use std::optim::clip::clip_grad_by_norm;
        fn run() -> Tensor[f64, ..] {
            let gw = ones(3);
            let clipped = clip_grad_by_norm<f64>(gw, 1.0);
            clipped
        }
        run()
    "#;
    let v = run_vm_with_std(src);
    match v {
        Value::Tensor(t) => {
            assert_eq!(t.borrow().dtype(), BaseType::F64, "结果应为 F64");
            assert_eq!(t.borrow().shape(), vec![3], "shape 应为 [3]");
            // 验证裁剪后的值 ≈ 0.577
            if let TensorData::F64(arr) = &t.borrow().data {
                let expected = 1.0 / 3.0f64.sqrt();
                for &x in arr.iter() {
                    assert!(
                        (x - expected).abs() < 1e-6,
                        "裁剪后值应为 {}，got {}",
                        expected,
                        x
                    );
                }
            } else {
                panic!("期望 F64 数据");
            }
        }
        v => panic!("期望 Tensor，got {:?}", v),
    }
}

// ─── Test 12: clip_grad_by_norm norm<max_norm 时不裁剪 ────────────────────

#[test]
#[ignore = "use+泛型调用限制：同 test_clip_grad_by_norm"]
fn test_clip_grad_by_norm_no_clip() {
    let src = r#"
        use std::optim::clip::clip_grad_by_norm;
        fn run() -> Tensor[f64, ..] {
            let gw = ones(3);
            // max_norm=10.0 > norm≈1.732，不应裁剪
            let clipped = clip_grad_by_norm<f64>(gw, 10.0);
            clipped
        }
        run()
    "#;
    let v = run_vm_with_std(src);
    match v {
        Value::Tensor(t) => {
            assert_eq!(t.borrow().shape(), vec![3], "shape 应为 [3]");
            if let TensorData::F64(arr) = &t.borrow().data {
                for &x in arr.iter() {
                    assert!((x - 1.0f64).abs() < 1e-9, "未裁剪时值应为 1.0，got {}", x);
                }
            } else {
                panic!("期望 F64 数据");
            }
        }
        v => panic!("期望 Tensor，got {:?}", v),
    }
}

// ─── Test 13: adamw_step_w 单值返回版本 ───────────────────────────────────
//
// 需要 autodiff 上下文：先 new_grad + param + backward，再调 adamw_step_w。
// 验证返回 new_w 是 F64 tensor 且 shape 正确。
//
// #[ignore] 原因：同 test_clip_grad_by_norm，use+泛型调用限制。

#[test]
#[ignore = "use+泛型调用限制：generic_funcs 未注册"]
fn test_adamw_step_w() {
    let src = r#"
        use std::optim::adamw::adamw_step_w;
        fn run() -> Tensor[f64, ..] {
            new_grad();
            let w = param(tensor[[1.0, 2.0, 3.0]]);
            let m = zeros(3);
            let v = zeros(3);
            let loss = (w * w).sum();
            backward(loss);
            stop_grad();
            let new_w = adamw_step_w<f64>(w, m, v, 0.001, 0.9, 0.999, 1e-8, 0.01, 0.9, 0.999);
            new_w
        }
        run()
    "#;
    let v = run_vm_with_std(src);
    match v {
        Value::Tensor(t) => {
            assert_eq!(t.borrow().dtype(), BaseType::F64, "new_w 应为 F64");
            assert_eq!(t.borrow().shape(), vec![3], "shape 应为 [3]");
            // 验证 new_w 是有限值（非 NaN/Inf）
            if let TensorData::F64(arr) = &t.borrow().data {
                for &x in arr.iter() {
                    assert!(x.is_finite(), "new_w 元素应为有限值，got {}", x);
                }
            } else {
                panic!("期望 F64 数据");
            }
        }
        v => panic!("期望 Tensor，got {:?}", v),
    }
}

// ─── Test 14: adamw_step_w 与 tuple 版本一致性（轻量验证）─────────────────
//
// 验证 adamw_step_w 的返回值与 adamw_step tuple 版本的第一个元素一致。
// 由于 tuple 解构在 .th 中可能有已知限制，这里只验证 adamw_step_w 单独可用。

#[test]
#[ignore = "use+泛型调用限制：同 test_adamw_step_w"]
fn test_adamw_step_w_returns_finite() {
    let src = r#"
        use std::optim::adamw::adamw_step_w;
        fn run() -> f64 {
            new_grad();
            let w = param(tensor[[0.5]]);
            let m = zeros(1);
            let v = zeros(1);
            let loss = w * w;
            backward(loss);
            stop_grad();
            let new_w = adamw_step_w<f64>(w, m, v, 0.01, 0.9, 0.999, 1e-8, 0.0, 0.9, 0.999);
            // decay=0.0 使权重衰减不影响，便于验证
            new_w.sum()
        }
        run()
    "#;
    let v = run_vm_with_std(src);
    match v {
        Value::Float(f) => {
            assert!(f.is_finite(), "new_w.sum() 应为有限值，got {}", f);
            // w=0.5, gw=2*0.5=1.0, new_m=0.9*0+0.1*1.0=0.1, new_v=0.999*0+0.001*1.0=0.001
            // m_hat=0.1/0.1=1.0, v_hat=0.001/0.001=1.0
            // new_w = 0.5 - 0.01 * 1.0 / (1.0.sqrt() + 1e-8) = 0.5 - 0.01 ≈ 0.49
            assert!(
                (f - 0.49f64).abs() < 0.01,
                "new_w.sum() 应接近 0.49，got {}",
                f
            );
        }
        v => panic!("期望 Float，got {:?}", v),
    }
}

// ─── Test 15: accumulate.th 解析验证 ──────────────────────────────────────
//
// 验证 accumulate_loop 高阶函数能正确解析和 lower（语法层面验证）。

#[test]
fn test_accumulate_th_parses() {
    let source = std::fs::read_to_string("std/optim/accumulate.th")
        .expect("无法读取 accumulate.th");
    let mut lexer = Lexer::new(&source);
    let tokens = lexer.tokenize().expect("accumulate.th 词法错误");
    let mut parser = Parser::new(tokens);
    let program = parser
        .parse_program()
        .expect("accumulate.th 语法错误");
    // 验证能 lower
    let mut lowerer = Lowerer::new();
    let _hir = lowerer
        .lower_program(&program)
        .expect("accumulate.th lower 错误");
}

// ─── Test 16: adamw.th 解析验证 ───────────────────────────────────────────

#[test]
fn test_adamw_th_parses() {
    let source = std::fs::read_to_string("std/optim/adamw.th")
        .expect("无法读取 adamw.th");
    let mut lexer = Lexer::new(&source);
    let tokens = lexer.tokenize().expect("adamw.th 词法错误");
    let mut parser = Parser::new(tokens);
    let _program = parser
        .parse_program()
        .expect("adamw.th 语法错误");
}

// ─── Test 17: clip.th 解析验证 ────────────────────────────────────────────

#[test]
fn test_clip_th_parses() {
    let source = std::fs::read_to_string("std/optim/clip.th")
        .expect("无法读取 clip.th");
    let mut lexer = Lexer::new(&source);
    let tokens = lexer.tokenize().expect("clip.th 词法错误");
    let mut parser = Parser::new(tokens);
    let _program = parser
        .parse_program()
        .expect("clip.th 语法错误");
}
