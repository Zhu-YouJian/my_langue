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
        // tensor() creates a tensor from nested array arguments
        // In Tenth, tensor[[1,2],[3,4]] is parsed as Call("tensor", [ArrayLiteral...])
        // But the bytecode compiler handles TensorLiteral directly via Op::MakeTensor,
        // so this native is only needed if tensor() is called as a function.
        // For safety, handle the case where a single tensor arg is passed through.
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
                let zeros = Tensor::zeros(&p.shape());
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

/// Extract a scalar f64 from a Value (Float or single-element Tensor)
fn as_f64(val: &Value) -> Option<f64> {
    match val {
        Value::Float(f) => Some(*f),
        Value::Tensor(t) => {
            let data = &t.borrow().data;
            if data.len() == 1 { Some(data[0]) } else { None }
        }
        _ => None,
    }
}

// ══════════════════════════════════════════════════════════════════════
// VM Autodiff End-to-End Tests
// ══════════════════════════════════════════════════════════════════════

#[test]
fn test_vm_autodiff_simple_gradient() {
    // y = x^2, dy/dx = 2x, at x=3, grad = 6
    let src = r#"
        new_grad();
        let x = param(tensor[[3.0]]);
        let y = x * x;
        backward(y);
        stop_grad();
        grad(x).sum()
    "#;
    let result = run_vm_autodiff(src).unwrap();
    let g = as_f64(&result).expect("expected numeric value");
    assert!((g - 6.0).abs() < 0.01, "expected 6.0, got {}", g);
}

#[test]
fn test_vm_autodiff_linear_gradient() {
    // y = 2*x + 1, dy/dx = 2
    let src = r#"
        new_grad();
        let x = param(tensor[[5.0]]);
        let y = 2.0 * x + 1.0;
        backward(y);
        stop_grad();
        grad(x).sum()
    "#;
    let result = run_vm_autodiff(src).unwrap();
    let g = as_f64(&result).expect("expected numeric value");
    assert!((g - 2.0).abs() < 0.01, "expected 2.0, got {}", g);
}

#[test]
fn test_vm_autodiff_relu_gradient() {
    // ReLU(x): if x > 0, grad = 1
    let src = r#"
        new_grad();
        let x = param(tensor[[3.0]]);
        let y = x.relu();
        backward(y);
        stop_grad();
        grad(x).sum()
    "#;
    let result = run_vm_autodiff(src).unwrap();
    let g = as_f64(&result).expect("expected numeric value");
    assert!((g - 1.0).abs() < 0.01, "expected 1.0, got {}", g);
}

#[test]
fn test_vm_autodiff_relu_gradient_negative() {
    // ReLU(x): if x < 0, grad = 0
    let src = r#"
        new_grad();
        let x = param(tensor[[-3.0]]);
        let y = x.relu();
        backward(y);
        stop_grad();
        grad(x).sum()
    "#;
    let result = run_vm_autodiff(src).unwrap();
    let g = as_f64(&result).expect("expected numeric value");
    assert!(g.abs() < 0.01, "expected 0.0, got {}", g);
}

#[test]
fn test_vm_autodiff_chain_rule() {
    // y = relu(2*x), dy/dx = 2 (since 2*3=6 > 0)
    let src = r#"
        new_grad();
        let x = param(tensor[[3.0]]);
        let y = (2.0 * x).relu();
        backward(y);
        stop_grad();
        grad(x).sum()
    "#;
    let result = run_vm_autodiff(src).unwrap();
    let g = as_f64(&result).expect("expected numeric value");
    assert!((g - 2.0).abs() < 0.01, "expected 2.0, got {}", g);
}

#[test]
fn test_vm_autodiff_exp_log_gradient() {
    // y = exp(log(x)), dy/dx = 1 at x=2
    let src = r#"
        new_grad();
        let x = param(tensor[[2.0]]);
        let y = x.log().exp();
        backward(y);
        stop_grad();
        grad(x).sum()
    "#;
    let result = run_vm_autodiff(src).unwrap();
    let g = as_f64(&result).expect("expected numeric value");
    assert!((g - 1.0).abs() < 0.05, "expected ~1.0, got {}", g);
}

#[test]
fn test_vm_autodiff_matmul_gradient() {
    // y = x @ w, check that grad(x) has correct shape
    let src = r#"
        new_grad();
        let x = param(tensor[[1.0, 2.0, 3.0]]);
        let w = param(tensor[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]);
        let y = x.matmul(w);
        backward(y);
        stop_grad();
        grad(x).sum()
    "#;
    let result = run_vm_autodiff(src).unwrap();
    let g = as_f64(&result).expect("expected numeric value");
    // grad(x) = upstream @ w^T = ones(1,2) @ [[1,0,1],[0,1,1]] = [1,1,2]
    // sum = 4.0
    assert!((g - 4.0).abs() < 0.01, "expected 4.0, got {}", g);
}

#[test]
fn test_vm_autodiff_softmax_gradient() {
    // softmax(x), gradient of sum(softmax(x)) w.r.t. x is 0
    // because sum(softmax(x)) = 1 for all x
    // Instead, test gradient of a single element: d(softmax(x)_0)/dx
    // which should be y_0 * (1 - y_0) for diagonal, -y_0 * y_j for off-diagonal
    let src = r#"
        new_grad();
        let x = param(tensor[[1.0, 2.0, 3.0]]);
        let y = x.softmax();
        backward(y);
        stop_grad();
        grad(x).sum()
    "#;
    let result = run_vm_autodiff(src).unwrap();
    let g = as_f64(&result).expect("expected numeric value");
    // sum of grad(sum(softmax), x) = 0 because sum(softmax) is constant
    assert!(g.abs() < 0.01, "expected ~0.0, got {}", g);
}

#[test]
fn test_vm_autodiff_sigmoid_gradient() {
    // sigmoid(x), grad = sigmoid(x) * (1 - sigmoid(x))
    let src = r#"
        new_grad();
        let x = param(tensor[[0.0]]);
        let y = x.sigmoid();
        backward(y);
        stop_grad();
        grad(x).sum()
    "#;
    let result = run_vm_autodiff(src).unwrap();
    let g = as_f64(&result).expect("expected numeric value");
    // sigmoid(0) = 0.5, grad = 0.5 * 0.5 = 0.25
    assert!((g - 0.25).abs() < 0.01, "expected 0.25, got {}", g);
}

#[test]
fn test_vm_autodiff_gelu_gradient() {
    // gelu(x) at x=0, grad ≈ 0.5
    let src = r#"
        new_grad();
        let x = param(tensor[[0.0]]);
        let y = x.gelu();
        backward(y);
        stop_grad();
        grad(x).sum()
    "#;
    let result = run_vm_autodiff(src).unwrap();
    let g = as_f64(&result).expect("expected numeric value");
    assert!((g - 0.5).abs() < 0.01, "expected ~0.5, got {}", g);
}

#[test]
fn test_vm_autodiff_zero_grad() {
    // After zero_grad(), gradients should be zero
    let src = r#"
        new_grad();
        let x = param(tensor[[3.0]]);
        let y = x * x;
        backward(y);
        zero_grad();
        grad(x).sum()
    "#;
    let result = run_vm_autodiff(src).unwrap();
    let g = as_f64(&result).expect("expected numeric value");
    assert!(g.abs() < 0.01, "expected 0.0 after zero_grad, got {}", g);
}

#[test]
fn test_vm_gradient_descent_step() {
    // Full training step: x = x - lr * grad(x)
    // y = x^2, at x=3, grad=6, lr=0.1
    // x_new = 3 - 0.1*6 = 2.4
    let src = r#"
        new_grad();
        let x = param(tensor[[3.0]]);
        let y = x * x;
        backward(y);
        let lr = 0.1;
        let g = grad(x);
        let x_new = x - lr * g;
        x_new.sum()
    "#;
    let result = run_vm_autodiff(src).unwrap();
    let x_new = as_f64(&result).expect("expected numeric value");
    assert!((x_new - 2.4).abs() < 0.01, "expected 2.4, got {}", x_new);
}

#[test]
fn test_vm_gradient_descent_convergence() {
    // Test that repeated gradient computation + manual update converges.
    // We can't use a while loop with param reassignment (the new x won't be
    // a tape leaf), so we unroll a few steps manually.
    let src = r#"
        new_grad();
        let x0 = param(tensor[[5.0]]);
        let y0 = x0 * x0;
        backward(y0);
        let lr = 0.1;
        let g0 = grad(x0);
        let x1 = x0 - lr * g0;

        new_grad();
        let x1p = param(x1);
        let y1 = x1p * x1p;
        backward(y1);
        let g1 = grad(x1p);
        let x2 = x1p - lr * g1;

        new_grad();
        let x2p = param(x2);
        let y2 = x2p * x2p;
        backward(y2);
        let g2 = grad(x2p);
        let x3 = x2p - lr * g2;

        x3.sum()
    "#;
    let result = run_vm_autodiff(src).unwrap();
    let x_final = as_f64(&result).expect("expected numeric value");
    // After 3 steps of GD on x^2 starting at 5 with lr=0.1:
    // x1 = 5 - 0.1*10 = 4, x2 = 4 - 0.1*8 = 3.2, x3 = 3.2 - 0.1*6.4 = 2.56
    assert!((x_final - 2.56).abs() < 0.05, "expected ~2.56 after 3 steps, got {}", x_final);
}

#[test]
fn test_vm_autodiff_matmul_training_step() {
    // Simple linear model: y = x @ w, loss = y.sum()
    // grad(w) should be x^T
    let src = r#"
        new_grad();
        let x = param(tensor[[1.0, 2.0]]);
        let w = param(tensor[[0.5], [0.5]]);
        let y = x.matmul(w);
        backward(y);
        let g = grad(w);
        g.sum()
    "#;
    let result = run_vm_autodiff(src).unwrap();
    let g_sum = as_f64(&result).expect("expected numeric value");
    // grad(w) = x^T @ upstream = [[1],[2]] @ [[1]] = [[1],[2]]
    // sum = 3.0
    assert!((g_sum - 3.0).abs() < 0.01, "expected 3.0, got {}", g_sum);
}

#[test]
fn test_vm_autodiff_two_param_model() {
    // y = (x @ w1) @ w2, check both gradients exist
    let src = r#"
        new_grad();
        let x = tensor[[1.0, 2.0]];
        let w1 = param(tensor[[1.0, 0.0], [0.0, 1.0]]);
        let w2 = param(tensor[[1.0], [0.0]]);
        let h = x.matmul(w1);
        let y = h.matmul(w2);
        backward(y);
        let g1 = grad(w1).sum();
        let g2 = grad(w2).sum();
        g1 + g2
    "#;
    let result = run_vm_autodiff(src).unwrap();
    let total_grad = as_f64(&result).expect("expected numeric value");
    // Both gradients should be non-zero
    assert!(total_grad.abs() > 0.01, "expected non-zero total gradient, got {}", total_grad);
}
