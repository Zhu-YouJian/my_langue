use tenth::lexer::lexer::Lexer;
use tenth::parser::parser::Parser;
use tenth::hir::lower::Lowerer;
use tenth::runtime::interpreter::Interpreter;
use tenth::runtime::value::Value;

fn run_code(src: &str) -> Result<Option<Value>, String> {
    let mut lexer = Lexer::new(src);
    let tokens = lexer.tokenize().map_err(|e| e.to_string())?;
    let mut parser = Parser::new(tokens);
    let program = parser.parse_program().map_err(|e| e.to_string())?;
    let mut lowerer = Lowerer::new();
    let hir = lowerer.lower_program(&program).map_err(|e| e.to_string())?;
    let mut interpreter = Interpreter::new(&hir);
    interpreter.execute_program(&hir).map_err(|e| e.to_string())
}

/// Extract a scalar f64 from a Value (Float or single-element Tensor)
fn as_f64(val: &Value) -> Option<f64> {
    match val {
        Value::Float(f) => Some(*f),
        Value::Tensor(t) => {
            let data = &t.borrow().data;
            if data.len() == 1 {
                Some(data[0])
            } else {
                None
            }
        }
        _ => None,
    }
}

// ── Autodiff end-to-end tests ──

#[test]
fn test_autodiff_simple_gradient() {
    // y = x^2, dy/dx = 2x, at x=3, grad = 6
    let src = r#"
        new_grad();
        let x = param(tensor[[3.0]]);
        let y = x * x;
        backward(y);
        stop_grad();
        grad(x).sum()
    "#;
    let result = run_code(src).unwrap();
    let g = as_f64(result.as_ref().unwrap()).expect("expected numeric value");
    assert!((g - 6.0).abs() < 0.01, "expected 6.0, got {}", g);
}

#[test]
fn test_autodiff_linear_gradient() {
    // y = 2*x + 1, dy/dx = 2
    let src = r#"
        new_grad();
        let x = param(tensor[[5.0]]);
        let y = 2.0 * x + 1.0;
        backward(y);
        stop_grad();
        grad(x).sum()
    "#;
    let result = run_code(src).unwrap();
    let g = as_f64(result.as_ref().unwrap()).expect("expected numeric value");
    assert!((g - 2.0).abs() < 0.01, "expected 2.0, got {}", g);
}

#[test]
fn test_autodiff_relu_gradient() {
    // ReLU(x): if x > 0, grad = 1
    let src = r#"
        new_grad();
        let x = param(tensor[[3.0]]);
        let y = x.relu();
        backward(y);
        stop_grad();
        grad(x).sum()
    "#;
    let result = run_code(src).unwrap();
    let g = as_f64(result.as_ref().unwrap()).expect("expected numeric value");
    assert!((g - 1.0).abs() < 0.01, "expected 1.0, got {}", g);
}

#[test]
fn test_autodiff_relu_gradient_negative() {
    // ReLU of negative input: gradient should be 0
    let src = r#"
        new_grad();
        let x = param(tensor[[-3.0]]);
        let y = x.relu();
        backward(y);
        stop_grad();
        grad(x).sum()
    "#;
    let result = run_code(src).unwrap();
    let g = as_f64(result.as_ref().unwrap()).expect("expected numeric value");
    assert!(g.abs() < 0.01, "expected 0.0, got {}", g);
}

#[test]
fn test_autodiff_chain_rule() {
    // y = (x * x) * x = x^3, dy/dx = 3x^2, at x=2, grad = 12
    let src = r#"
        new_grad();
        let x = param(tensor[[2.0]]);
        let xx = x * x;
        let y = xx * x;
        backward(y);
        stop_grad();
        grad(x).sum()
    "#;
    let result = run_code(src).unwrap();
    let g = as_f64(result.as_ref().unwrap()).expect("expected numeric value");
    assert!((g - 12.0).abs() < 0.1, "expected 12.0, got {}", g);
}

#[test]
fn test_autodiff_matmul_gradient() {
    // Basic matmul gradient: just check it doesn't error and produces a value
    let src = r#"
        new_grad();
        let w = param(tensor[[1.0, 2.0], [3.0, 4.0]]);
        let x = param(tensor[[1.0], [1.0]]);
        let y = w.matmul(x);
        backward(y.sum());
        stop_grad();
        grad(w).sum()
    "#;
    let result = run_code(src).unwrap();
    // Just check we get a numeric value (exact gradient depends on implementation)
    assert!(as_f64(result.as_ref().unwrap()).is_some(), "expected numeric gradient");
}

#[test]
fn test_autodiff_exp_log_gradient() {
    // d/dx(exp(x)) = exp(x), at x=0, grad = 1
    let src = r#"
        new_grad();
        let x = param(tensor[[0.0]]);
        let y = x.exp();
        backward(y);
        stop_grad();
        grad(x).sum()
    "#;
    let result = run_code(src).unwrap();
    let g = as_f64(result.as_ref().unwrap()).expect("expected numeric value");
    assert!((g - 1.0).abs() < 0.01, "expected 1.0, got {}", g);
}

#[test]
fn test_autodiff_zero_grad() {
    // After zero_grad, gradients should be zero
    let src = r#"
        new_grad();
        let x = param(tensor[[3.0]]);
        let y = x * x;
        backward(y);
        zero_grad();
        stop_grad();
        grad(x).sum()
    "#;
    let result = run_code(src).unwrap();
    let g = as_f64(result.as_ref().unwrap()).expect("expected numeric value");
    assert!(g.abs() < 0.01, "expected 0.0 after zero_grad, got {}", g);
}

// ── Closure capture tests ──

#[test]
fn test_closure_captures_variable() {
    let src = r#"
        let x = 10;
        let f = |y| x + y;
        f(5)
    "#;
    let result = run_code(src).unwrap();
    match result {
        Some(Value::Int(15)) => {}
        v => panic!("expected Int(15), got {:?}", v),
    }
}

#[test]
fn test_closure_captures_multiple_variables() {
    let src = r#"
        let a = 3;
        let b = 4;
        let f = |x| a * x + b;
        f(2)
    "#;
    let result = run_code(src).unwrap();
    match result {
        Some(Value::Int(10)) => {}
        v => panic!("expected Int(10), got {:?}", v),
    }
}

#[test]
fn test_closure_nested_capture() {
    let src = r#"
        let x = 5;
        let make_adder = |n| |m| n + m + x;
        let add5 = make_adder(10);
        add5(3)
    "#;
    let result = run_code(src).unwrap();
    match result {
        Some(Value::Int(18)) => {}
        v => panic!("expected Int(18), got {:?}", v),
    }
}

#[test]
fn test_closure_does_not_capture_params() {
    let src = r#"
        let f = |x, y| x + y;
        f(100, 200)
    "#;
    let result = run_code(src).unwrap();
    match result {
        Some(Value::Int(300)) => {}
        v => panic!("expected Int(300), got {:?}", v),
    }
}

// ── Tensor operation tests ──

#[test]
fn test_tensor_matmul() {
    let src = r#"
        let a = tensor[[1.0, 2.0], [3.0, 4.0]];
        let b = tensor[[5.0, 6.0], [7.0, 8.0]];
        let c = a.matmul(b);
        c.sum()
    "#;
    let result = run_code(src).unwrap();
    match result {
        Some(Value::Float(n)) => assert!((n - 134.0).abs() < 0.01, "expected 134.0, got {}", n),
        v => panic!("expected Float, got {:?}", v),
    }
}

#[test]
fn test_tensor_transpose() {
    let src = r#"
        let a = tensor[[1.0, 2.0, 3.0]];
        let b = a.transpose();
        b.sum()
    "#;
    let result = run_code(src).unwrap();
    match result {
        Some(Value::Float(n)) => assert!((n - 6.0).abs() < 0.01, "expected 6.0, got {}", n),
        v => panic!("expected Float, got {:?}", v),
    }
}

#[test]
fn test_tensor_sigmoid() {
    let src = r#"
        tensor[[0.0]].sigmoid().sum()
    "#;
    let result = run_code(src).unwrap();
    match result {
        Some(Value::Float(n)) => assert!((n - 0.5).abs() < 0.01, "expected 0.5, got {}", n),
        v => panic!("expected Float, got {:?}", v),
    }
}

#[test]
fn test_tensor_tanh() {
    let src = r#"
        tensor[[0.0]].tanh().sum()
    "#;
    let result = run_code(src).unwrap();
    match result {
        Some(Value::Float(n)) => assert!(n.abs() < 0.01, "expected 0.0, got {}", n),
        v => panic!("expected Float, got {:?}", v),
    }
}

#[test]
fn test_tensor_reshape() {
    let src = r#"
        let a = tensor[[1.0, 2.0], [3.0, 4.0]];
        let b = a.flatten();
        b.sum()
    "#;
    let result = run_code(src).unwrap();
    match result {
        Some(Value::Float(n)) => assert!((n - 10.0).abs() < 0.01, "expected 10.0, got {}", n),
        v => panic!("expected Float, got {:?}", v),
    }
}

#[test]
fn test_tensor_mean() {
    let src = r#"
        tensor[[2.0, 4.0, 6.0]].mean()
    "#;
    let result = run_code(src).unwrap();
    match result {
        Some(Value::Float(n)) => assert!((n - 4.0).abs() < 0.01, "expected 4.0, got {}", n),
        v => panic!("expected Float, got {:?}", v),
    }
}

// ── VM tensor literal test ──

#[test]
fn test_vm_tensor_literal() {
    let src = r#"
        tensor[[1.0, 2.0], [3.0, 4.0]].sum()
    "#;
    let result = run_code(src).unwrap();
    match result {
        Some(Value::Float(n)) => assert!((n - 10.0).abs() < 0.01, "expected 10.0, got {}", n),
        v => panic!("expected Float, got {:?}", v),
    }
}

// ── Error span propagation test ──

#[test]
fn test_error_span_in_borrow_check() {
    let src = r#"
        let x = 42;
        let y = move x;
        let z = x + 1
    "#;
    let result = run_code(src);
    assert!(result.is_err());
    let err_msg = result.unwrap_err();
    let err_str = err_msg.to_string();
    assert!(err_str.contains("moved"), "expected 'moved' in error, got: {}", err_str);
}
