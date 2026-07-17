//! MaxPool2D / AvgPool2D 算子测试。
//!
//! 覆盖：
//! - 前向：简单 4x4 输入 + 2x2 kernel → 2x2 输出，值正确
//! - stride=1 → 输出尺寸正确
//! - padding=1 → 输出尺寸正确
//! - backward 梯度正确（max_pool 路由到 argmax，avg_pool 均分）
//!
//! 通过 VM 路径执行（参考 vm_autodiff_test.rs 的 run_vm_autodiff 模式）。

use std::rc::Rc;
use std::cell::RefCell;

use tenth::compile::bytecode::BytecodeCompiler;
use tenth::hir::lower::Lowerer;
use tenth::lexer::lexer::Lexer;
use tenth::parser::parser::Parser;
use tenth::runtime::autodiff::Tape;
use tenth::runtime::tensor::Tensor;
use tenth::runtime::value::Value;
use tenth::runtime::vm::Vm;

/// 通过 VM 执行 .th 源码（带 autodiff natives，参考 vm_autodiff_test.rs）。
fn run_vm_pool(src: &str) -> Result<Value, String> {
    let mut lexer = Lexer::new(src);
    let tokens = lexer.tokenize().map_err(|e| e.to_string())?;
    let mut parser = Parser::new(tokens);
    let program = parser.parse_program().map_err(|e| e.to_string())?;
    let mut lowerer = Lowerer::new();
    let hir = lowerer.lower_program(&program).map_err(|e| e.to_string())?;

    let mut vm = Vm::new();

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
        else {
            Err(tenth::error::TenthError::RuntimeError { line: None, col: None,
                message: "tensor() unexpected args".into() })
        }
    });

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
            Err(tenth::error::TenthError::RuntimeError { line: None, col: None,
                message: "param() requires a tensor argument".into() })
        }
    });
    vm.add_native("backward".into(), |vm, args| {
        if let Some(Value::Tensor(t)) = args.first() {
            if let Some(ref tape) = vm.tape {
                let loss_id = t.borrow().tape_id
                    .ok_or_else(|| tenth::error::TenthError::RuntimeError { line: None, col: None,
                        message: "backward(): tensor has no tape_id".into() })?;
                tape.backward(loss_id);
                Ok(Value::Unit)
            } else {
                Err(tenth::error::TenthError::RuntimeError { line: None, col: None,
                    message: "new_grad() not called".into() })
            }
        } else {
            Err(tenth::error::TenthError::RuntimeError { line: None, col: None,
                message: "backward() requires a tensor argument".into() })
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
            Err(tenth::error::TenthError::RuntimeError { line: None, col: None,
                message: "grad() requires a tensor argument".into() })
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
        if let Some(ref tape) = vm.tape { tape.zero_grad(); }
        Ok(Value::Unit)
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

fn tensor_to_vec(val: &Value) -> Option<Vec<f64>> {
    if let Value::Tensor(t) = val {
        let data = &t.borrow().data;
        Some(data.as_f64_view().iter().cloned().collect())
    } else {
        None
    }
}

fn tensor_shape(val: &Value) -> Option<Vec<usize>> {
    if let Value::Tensor(t) = val {
        Some(t.borrow().shape())
    } else {
        None
    }
}

#[test]
fn test_max_pool2d_forward_basic() {
    let src = r#"
        let x = tensor[[1.0,2.0,3.0,4.0],[5.0,6.0,7.0,8.0],[9.0,10.0,11.0,12.0],[13.0,14.0,15.0,16.0]].reshape(1, 1, 4, 4);
        x.max_pool2d(2, 2, 2, 2, 0, 0)
    "#;
    let result = run_vm_pool(src).unwrap();
    let shape = tensor_shape(&result).expect("expected tensor");
    assert_eq!(shape, vec![1, 1, 2, 2], "max_pool2d output shape");
    let data = tensor_to_vec(&result).expect("expected tensor");
    assert_eq!(data, vec![6.0, 8.0, 14.0, 16.0], "max_pool2d values");
}

#[test]
fn test_max_pool2d_forward_stride1() {
    let src = r#"
        let x = tensor[[1.0,2.0,3.0,4.0],[5.0,6.0,7.0,8.0],[9.0,10.0,11.0,12.0],[13.0,14.0,15.0,16.0]].reshape(1, 1, 4, 4);
        x.max_pool2d(2, 2, 1, 1, 0, 0)
    "#;
    let result = run_vm_pool(src).unwrap();
    let shape = tensor_shape(&result).expect("expected tensor");
    assert_eq!(shape, vec![1, 1, 3, 3], "stride=1 → 3x3 output");
    let data = tensor_to_vec(&result).expect("expected tensor");
    assert_eq!(data, vec![6.0, 7.0, 8.0, 10.0, 11.0, 12.0, 14.0, 15.0, 16.0]);
}

#[test]
fn test_max_pool2d_forward_padding() {
    let src = r#"
        let x = tensor[[1.0,2.0,3.0],[4.0,5.0,6.0],[7.0,8.0,9.0]].reshape(1, 1, 3, 3);
        x.max_pool2d(2, 2, 2, 2, 1, 1)
    "#;
    let result = run_vm_pool(src).unwrap();
    let shape = tensor_shape(&result).expect("expected tensor");
    assert_eq!(shape, vec![1, 1, 2, 2], "padding=1 → 2x2 output");
    let data = tensor_to_vec(&result).expect("expected tensor");
    assert_eq!(data, vec![1.0, 3.0, 7.0, 9.0], "padding=1 values");
}

#[test]
fn test_avg_pool2d_forward_basic() {
    let src = r#"
        let x = tensor[[1.0,2.0,3.0,4.0],[5.0,6.0,7.0,8.0],[9.0,10.0,11.0,12.0],[13.0,14.0,15.0,16.0]].reshape(1, 1, 4, 4);
        x.avg_pool2d(2, 2, 2, 2, 0, 0)
    "#;
    let result = run_vm_pool(src).unwrap();
    let shape = tensor_shape(&result).expect("expected tensor");
    assert_eq!(shape, vec![1, 1, 2, 2], "avg_pool2d output shape");
    let data = tensor_to_vec(&result).expect("expected tensor");
    assert_eq!(data, vec![3.5, 5.5, 11.5, 13.5], "avg_pool2d values");
}

#[test]
fn test_avg_pool2d_forward_padding() {
    let src = r#"
        let x = tensor[[1.0,2.0,3.0],[4.0,5.0,6.0],[7.0,8.0,9.0]].reshape(1, 1, 3, 3);
        x.avg_pool2d(2, 2, 2, 2, 1, 1)
    "#;
    let result = run_vm_pool(src).unwrap();
    let shape = tensor_shape(&result).expect("expected tensor");
    assert_eq!(shape, vec![1, 1, 2, 2], "avg padding=1 → 2x2 output");
    let data = tensor_to_vec(&result).expect("expected tensor");
    assert_eq!(data, vec![1.0, 2.5, 5.5, 7.0], "avg padding=1 values");
}

#[test]
fn test_max_pool2d_backward_argmax_routing() {
    let src = r#"
        new_grad();
        let x = param(tensor[[1.0,2.0,3.0,4.0],[5.0,6.0,7.0,8.0],[9.0,10.0,11.0,12.0],[13.0,14.0,15.0,16.0]].reshape(1, 1, 4, 4));
        let y = x.max_pool2d(2, 2, 2, 2, 0, 0);
        let loss = y.sum();
        backward(loss);
        stop_grad();
        grad(x)
    "#;
    let result = run_vm_pool(src).unwrap();
    let data = tensor_to_vec(&result).expect("expected grad tensor");
    let expected = vec![
        0.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 1.0,
        0.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 1.0,
    ];
    for (i, (got, exp)) in data.iter().zip(expected.iter()).enumerate() {
        assert!((got - exp).abs() < 1e-6, "grad[{}] = {}, expected {}", i, got, exp);
    }
}

#[test]
fn test_avg_pool2d_backward_even_split() {
    let src = r#"
        new_grad();
        let x = param(tensor[[1.0,2.0,3.0,4.0],[5.0,6.0,7.0,8.0],[9.0,10.0,11.0,12.0],[13.0,14.0,15.0,16.0]].reshape(1, 1, 4, 4));
        let y = x.avg_pool2d(2, 2, 2, 2, 0, 0);
        let loss = y.sum();
        backward(loss);
        stop_grad();
        grad(x)
    "#;
    let result = run_vm_pool(src).unwrap();
    let data = tensor_to_vec(&result).expect("expected grad tensor");
    for (i, v) in data.iter().enumerate() {
        assert!((v - 0.25).abs() < 1e-6, "grad[{}] = {}, expected 0.25", i, v);
    }
}

#[test]
fn test_avg_pool2d_backward_padding() {
    let src = r#"
        new_grad();
        let x = param(tensor[[1.0,2.0,3.0],[4.0,5.0,6.0],[7.0,8.0,9.0]].reshape(1, 1, 3, 3));
        let y = x.avg_pool2d(2, 2, 2, 2, 1, 1);
        let loss = y.sum();
        backward(loss);
        stop_grad();
        grad(x)
    "#;
    let result = run_vm_pool(src).unwrap();
    let data = tensor_to_vec(&result).expect("expected grad tensor");
    assert_eq!(data.len(), 9, "grad should be 9 elements (3x3)");
    let expected = vec![
        1.0,  0.5,  0.5,
        0.5,  0.25, 0.25,
        0.5,  0.25, 0.25,
    ];
    for (i, (got, exp)) in data.iter().zip(expected.iter()).enumerate() {
        assert!((got - exp).abs() < 1e-6, "grad[{}] = {}, expected {}", i, got, exp);
    }
}
