// f32 自动微分专项测试 — Phase 4
// 验证方案 B（前向 f32 + 反向 f64 + 梯度按参数 dtype 写回）正确性。
//
// 核心路径：
//   param(f32_tensor) → Tape 记录引用 → 前向运算（f32）
//   → backward（node_grads: ArrayD<f64>，反向用 f64 计算）
//   → acc_grad(&ArrayD<f64>) → 按 tensor.dtype 转换存储（f32 参数→F32 grad）
//   → grad native → from_tensor_data(grad) 保留 dtype → 返回 f32 Tensor
//
// 由于反向计算在 f64 下进行，数值稳定性应与 f64 路径一致（误差 < 1e-10）。

use tenth::hir::types::BaseType;
use tenth::lexer::lexer::Lexer;
use tenth::parser::parser::Parser;
use tenth::hir::lower::Lowerer;
use tenth::runtime::vm::Vm;
use tenth::runtime::value::Value;
use tenth::runtime::autodiff::Tape;
use tenth::runtime::tensor::Tensor;
use tenth::compile::bytecode::BytecodeCompiler;
use std::rc::Rc;
use std::cell::RefCell;

/// Run source code through the VM with full autodiff native functions registered.
fn run_vm_autodiff(src: &str) -> Result<Value, String> {
    let mut lexer = Lexer::new(src);
    let tokens = lexer.tokenize().map_err(|e| e.to_string())?;
    let mut parser = Parser::new(tokens);
    let program = parser.parse_program().map_err(|e| e.to_string())?;
    let mut lowerer = Lowerer::new();
    let hir = lowerer.lower_program(&program).map_err(|e| e.to_string())?;

    let mut vm = Vm::new();

    // ── Standard natives ──
    vm.add_native("println".into(), |_vm, args| {
        for a in args { print!("{a}"); }
        println!();
        Ok(Value::Unit)
    });
    vm.add_native("Vec::new".into(), |_vm, _args| {
        Ok(Value::Vec(Rc::new(RefCell::new(Vec::new()))))
    });
    vm.add_native("tensor".into(), |_vm, args| {
        if args.len() == 1 {
            Ok(args[0].clone())
        } else {
            Err(tenth::error::TenthError::RuntimeError { line: None, col: None, message: "tensor() unexpected args".into() })
        }
    });

    // ── Autodiff natives ──
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
            Err(tenth::error::TenthError::RuntimeError { line: None, col: None, message: "param() requires a tensor argument".into() })
        }
    });
    vm.add_native("backward".into(), |vm, args| {
        if let Some(Value::Tensor(t)) = args.first() {
            if let Some(ref tape) = vm.tape {
                let loss_id = t.borrow().tape_id
                    .ok_or_else(|| tenth::error::TenthError::RuntimeError { line: None, col: None, message: "backward(): tensor has no tape_id".into() })?;
                tape.backward(loss_id);
                Ok(Value::Unit)
            } else {
                Err(tenth::error::TenthError::RuntimeError { line: None, col: None, message: "new_grad() not called".into() })
            }
        } else {
            Err(tenth::error::TenthError::RuntimeError { line: None, col: None, message: "backward() requires a tensor argument".into() })
        }
    });
    vm.add_native("grad".into(), |_vm, args| {
        if let Some(Value::Tensor(t)) = args.first() {
            let p = t.borrow();
            if let Some(ref grad) = p.grad {
                let grad_tensor = Tensor::from_tensor_data(grad.clone());
                Ok(Value::Tensor(Rc::new(RefCell::new(grad_tensor))))
            } else {
                // 按参数 dtype 返回零张量（f32 → zeros_f32）
                let zeros = if p.is_f32() {
                    Tensor::zeros_f32(&p.shape())
                } else {
                    Tensor::zeros(&p.shape())
                };
                Ok(Value::Tensor(Rc::new(RefCell::new(zeros))))
            }
        } else {
            Err(tenth::error::TenthError::RuntimeError { line: None, col: None, message: "grad() requires a tensor argument".into() })
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
    vm.add_native("cross_entropy".into(), |vm, args| {
        if args.len() < 2 {
            return Err(tenth::error::TenthError::RuntimeError { line: None, col: None, message: "cross_entropy(logits, target) expects two tensors".into() });
        }
        if let (Value::Tensor(logits), Value::Tensor(target)) = (&args[0], &args[1]) {
            let logits_data = logits.borrow();
            let target_data = target.borrow();
            let sm = logits_data.softmax().ok_or_else(|| {
                tenth::error::TenthError::RuntimeError { line: None, col: None, message: "softmax failed in cross_entropy".into() }
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
            // 按 logits dtype 构造 loss tensor
            let is_f32 = logits_data.is_f32();
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
            Err(tenth::error::TenthError::RuntimeError { line: None, col: None, message: "cross_entropy(logits, target) expects two tensors".into() })
        }
    });

    for func in &hir.functions {
        let compiler = BytecodeCompiler::new();
        match compiler.compile(func) {
            Ok((chunk, closures)) => {
                vm.add_fn(func.name.clone(), chunk);
                for (name, closure_chunk) in closures {
                    vm.add_fn(name, closure_chunk);
                }
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

/// Extract a scalar f64 from a Value (Float/Float32 or single-element Tensor).
fn as_f64(val: &Value) -> Option<f64> {
    match val {
        Value::Float(f) => Some(*f),
        Value::Float32(f) => Some(*f as f64),
        Value::Tensor(t) => {
            let data = &t.borrow().data;
            if data.len() == 1 { Some(data[0]) } else { None }
        }
        _ => None,
    }
}

// ══════════════════════════════════════════════════════════════════════
// Phase 4: f32 Autodiff End-to-End Tests
// ══════════════════════════════════════════════════════════════════════

#[test]
fn test_f32_new_grad_param() {
    // new_grad + param 注册 f32 Tensor，tape_id 应被设置
    let src = r#"
        new_grad();
        let x = param(tensor[[3.0f32, 4.0f32]]);
        stop_grad();
        x.sum()
    "#;
    let result = run_vm_autodiff(src).unwrap();
    let g = as_f64(&result).expect("expected numeric value");
    assert!((g - 7.0).abs() < 1e-6, "expected 7.0, got {}", g);
}

#[test]
fn test_f32_backward_simple_gradient() {
    // y = x^2, dy/dx = 2x, at x=3, grad = 6 (f32 输入)
    let src = r#"
        new_grad();
        let x = param(tensor[[3.0f32]]);
        let y = x * x;
        backward(y);
        stop_grad();
        grad(x).sum()
    "#;
    let result = run_vm_autodiff(src).unwrap();
    let g = as_f64(&result).expect("expected numeric value");
    // 反向用 f64 计算，误差应极小
    assert!((g - 6.0).abs() < 1e-10, "expected 6.0, got {}", g);
}

#[test]
fn test_f32_backward_relu_gradient() {
    // ReLU(x): if x > 0, grad = 1 (f32 输入)
    let src = r#"
        new_grad();
        let x = param(tensor[[3.0f32]]);
        let y = x.relu();
        backward(y);
        stop_grad();
        grad(x).sum()
    "#;
    let result = run_vm_autodiff(src).unwrap();
    let g = as_f64(&result).expect("expected numeric value");
    assert!((g - 1.0).abs() < 1e-10, "expected 1.0, got {}", g);
}

#[test]
fn test_f32_backward_relu_gradient_negative() {
    // ReLU(x): if x < 0, grad = 0 (f32 输入)
    let src = r#"
        new_grad();
        let x = param(tensor[[-3.0f32]]);
        let y = x.relu();
        backward(y);
        stop_grad();
        grad(x).sum()
    "#;
    let result = run_vm_autodiff(src).unwrap();
    let g = as_f64(&result).expect("expected numeric value");
    assert!(g.abs() < 1e-10, "expected 0.0, got {}", g);
}

#[test]
fn test_f32_backward_matmul() {
    // y = x @ w, f32 输入
    let src = r#"
        new_grad();
        let x = param(tensor[[1.0f32, 2.0f32, 3.0f32]]);
        let w = param(tensor[[1.0f32, 0.0f32], [0.0f32, 1.0f32], [1.0f32, 1.0f32]]);
        let y = x.matmul(w);
        backward(y);
        stop_grad();
        grad(x).sum()
    "#;
    let result = run_vm_autodiff(src).unwrap();
    let g = as_f64(&result).expect("expected numeric value");
    // grad(x) = upstream @ w^T = ones(1,2) @ [[1,0,1],[0,1,1]] = [1,1,2], sum = 4.0
    assert!((g - 4.0).abs() < 1e-10, "expected 4.0, got {}", g);
}

#[test]
fn test_f32_backward_softmax() {
    // softmax(x), grad of sum(softmax) w.r.t x = 0 (因为 sum(softmax) 恒为 1)
    // f32 前向 softmax 有精度损失（exp/sum 在 f32 下计算），误差 ~1e-7 量级
    let src = r#"
        new_grad();
        let x = param(tensor[[1.0f32, 2.0f32, 3.0f32]]);
        let y = x.softmax();
        backward(y);
        stop_grad();
        grad(x).sum()
    "#;
    let result = run_vm_autodiff(src).unwrap();
    let g = as_f64(&result).expect("expected numeric value");
    assert!(g.abs() < 1e-5, "expected ~0.0, got {}", g);
}

#[test]
fn test_f32_backward_sigmoid() {
    // sigmoid(0) = 0.5, grad = 0.5 * 0.5 = 0.25 (f32 输入)
    let src = r#"
        new_grad();
        let x = param(tensor[[0.0f32]]);
        let y = x.sigmoid();
        backward(y);
        stop_grad();
        grad(x).sum()
    "#;
    let result = run_vm_autodiff(src).unwrap();
    let g = as_f64(&result).expect("expected numeric value");
    assert!((g - 0.25).abs() < 1e-10, "expected 0.25, got {}", g);
}

#[test]
fn test_f32_backward_cross_entropy() {
    // f32 cross_entropy: 反向用 f64，grad 应与 f64 路径一致
    let src = r#"
        new_grad();
        let logits = param(tensor[[2.0f32, 1.0f32, 0.5f32]]);
        let target = tensor[[1.0, 0.0, 0.0]];
        let loss = cross_entropy(logits, target);
        backward(loss);
        stop_grad();
        grad(logits).sum()
    "#;
    let result = run_vm_autodiff(src).unwrap();
    let g = as_f64(&result).expect("expected numeric value");
    // softmax(logits) - target 的和 = sum(softmax) - sum(target) = 1 - 1 = 0
    // f32 前向 softmax 有精度损失，误差 ~1e-7 量级
    assert!(g.abs() < 1e-5, "expected ~0.0, got {}", g);
}

#[test]
fn test_f32_grad_dtype() {
    // f32 参数的 grad 应为 F32 dtype（方案 B：反向 f64，写回按参数 dtype 转换）
    let src = r#"
        new_grad();
        let x = param(tensor[[3.0f32]]);
        let y = x * x;
        backward(y);
        stop_grad();
        grad(x)
    "#;
    let result = run_vm_autodiff(src).unwrap();
    match result {
        Value::Tensor(t) => {
            let t = t.borrow();
            assert!(t.is_f32(), "f32 参数的 grad 应为 F32 dtype（方案 B 写回转换）");
            assert_eq!(t.dtype(), BaseType::F32);
            assert_eq!(t.shape(), vec![1, 1]);
            // grad 值应为 2*x = 6.0
            assert!((t.get(&[0, 0]).unwrap() - 6.0).abs() < 1e-5, "expected 6.0");
        }
        other => panic!("期望 Tensor，得到 {:?}", other),
    }
}

#[test]
fn test_f32_extreme_input_stability() {
    // 极端输入下不应出现 NaN/Inf（方案 B 反向用 f64 天然稳定）
    let src = r#"
        new_grad();
        let x = param(tensor[[1e30f32, 1e-30f32, 0.0f32]]);
        let y = x.softmax();
        backward(y);
        stop_grad();
        grad(x).sum()
    "#;
    let result = run_vm_autodiff(src).unwrap();
    let g = as_f64(&result).expect("expected numeric value");
    // softmax 的 sum 恒为 1，grad 之和应为 0
    assert!(g.is_finite(), "grad 应为有限值，得到 {}", g);
    assert!(g.abs() < 1e-5, "expected ~0.0, got {}", g);
}

#[test]
fn test_f32_exp_log_gradient() {
    // y = exp(log(x)), dy/dx = 1 at x=2 (f32 输入)
    let src = r#"
        new_grad();
        let x = param(tensor[[2.0f32]]);
        let y = x.log().exp();
        backward(y);
        stop_grad();
        grad(x).sum()
    "#;
    let result = run_vm_autodiff(src).unwrap();
    let g = as_f64(&result).expect("expected numeric value");
    assert!((g - 1.0).abs() < 1e-5, "expected ~1.0, got {}", g);
}

#[test]
fn test_f32_chain_rule() {
    // y = relu(2*x), dy/dx = 2 (since 2*3=6 > 0, f32 输入)
    let src = r#"
        new_grad();
        let x = param(tensor[[3.0f32]]);
        let y = (2.0f32 * x).relu();
        backward(y);
        stop_grad();
        grad(x).sum()
    "#;
    let result = run_vm_autodiff(src).unwrap();
    let g = as_f64(&result).expect("expected numeric value");
    assert!((g - 2.0).abs() < 1e-10, "expected 2.0, got {}", g);
}
