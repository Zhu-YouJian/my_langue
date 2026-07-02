// abs 接入 autodiff 测试套件
// 验证 |x| 的反向梯度：d|x|/dx = sign(x)，x=0 处取 0
// 覆盖：前向、反向基本、零处次梯度、嵌套 abs、l1_loss、huber_loss_train

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

fn as_f64_vec(val: &Value) -> Option<Vec<f64>> {
    match val {
        Value::Tensor(t) => {
            let data = t.borrow().data.as_f64_view();
            Some(data.iter().cloned().collect())
        }
        Value::Float(f) => Some(vec![*f]),
        _ => None,
    }
}

#[test]
fn test_abs_forward() {
    let src = r#"
        let x = tensor[[3.0, -5.0, 0.0]];
        x.abs()
    "#;
    let result = run_code(src).unwrap();
    let v = as_f64_vec(result.as_ref().unwrap()).expect("expected tensor");
    assert!((v[0] - 3.0).abs() < 1e-6, "expected 3.0, got {}", v[0]);
    assert!((v[1] - 5.0).abs() < 1e-6, "expected 5.0, got {}", v[1]);
    assert!((v[2] - 0.0).abs() < 1e-6, "expected 0.0, got {}", v[2]);
}

#[test]
fn test_abs_backward_positive() {
    // x > 0：d|x|/dx = 1
    let src = r#"
        new_grad();
        let x = param(tensor[[3.0, 5.0]]);
        let y = x.abs();
        backward(y);
        stop_grad();
        grad(x)
    "#;
    let result = run_code(src).unwrap();
    let v = as_f64_vec(result.as_ref().unwrap()).expect("expected tensor");
    assert!((v[0] - 1.0).abs() < 1e-6, "expected 1.0, got {}", v[0]);
    assert!((v[1] - 1.0).abs() < 1e-6, "expected 1.0, got {}", v[1]);
}

#[test]
fn test_abs_backward_negative() {
    // x < 0：d|x|/dx = -1
    let src = r#"
        new_grad();
        let x = param(tensor[[-3.0, -5.0]]);
        let y = x.abs();
        backward(y);
        stop_grad();
        grad(x)
    "#;
    let result = run_code(src).unwrap();
    let v = as_f64_vec(result.as_ref().unwrap()).expect("expected tensor");
    assert!((v[0] - (-1.0)).abs() < 1e-6, "expected -1.0, got {}", v[0]);
    assert!((v[1] - (-1.0)).abs() < 1e-6, "expected -1.0, got {}", v[1]);
}

#[test]
fn test_abs_backward_zero() {
    // x = 0：次梯度中点取 0
    let src = r#"
        new_grad();
        let x = param(tensor[[0.0]]);
        let y = x.abs();
        backward(y);
        stop_grad();
        grad(x)
    "#;
    let result = run_code(src).unwrap();
    let v = as_f64_vec(result.as_ref().unwrap()).expect("expected tensor");
    assert!((v[0] - 0.0).abs() < 1e-6, "expected 0.0, got {}", v[0]);
}

#[test]
fn test_abs_backward_mixed() {
    // 混合正负零：梯度应为 [1, -1, 0]
    let src = r#"
        new_grad();
        let x = param(tensor[[2.0, -4.0, 0.0]]);
        let y = x.abs();
        backward(y);
        stop_grad();
        grad(x)
    "#;
    let result = run_code(src).unwrap();
    let v = as_f64_vec(result.as_ref().unwrap()).expect("expected tensor");
    assert!((v[0] - 1.0).abs() < 1e-6, "expected 1.0, got {}", v[0]);
    assert!((v[1] - (-1.0)).abs() < 1e-6, "expected -1.0, got {}", v[1]);
    assert!((v[2] - 0.0).abs() < 1e-6, "expected 0.0, got {}", v[2]);
}

#[test]
fn test_abs_nested() {
    // abs(abs(x)) = abs(x)，梯度 sign(sign(x)) = sign(x)
    let src = r#"
        new_grad();
        let x = param(tensor[[-3.0, 5.0]]);
        let y = x.abs().abs();
        backward(y);
        stop_grad();
        grad(x)
    "#;
    let result = run_code(src).unwrap();
    let v = as_f64_vec(result.as_ref().unwrap()).expect("expected tensor");
    // 外层 abs 的输入 = |x|（已非负），sign(|x|) = 1（x≠0）
    // 内层 abs 的梯度 = 外层 grad * sign(|x|) = 1 * 1 = 1
    // 传到 x 的梯度 = 内层 grad * sign(x) = 1 * sign(x)
    assert!((v[0] - (-1.0)).abs() < 1e-6, "expected -1.0, got {}", v[0]);
    assert!((v[1] - 1.0).abs() < 1e-6, "expected 1.0, got {}", v[1]);
}

#[test]
fn test_l1_loss_backward() {
    // l1_loss = |pred - target|
    // d/d_pred = sign(pred - target)
    let src = r#"
        new_grad();
        let p = param(tensor[[1.0, 3.0, 5.0]]);
        let t = tensor[[2.0, 3.0, 4.0]];
        let diff = p - t;
        let loss = diff.abs();
        backward(loss);
        stop_grad();
        grad(p)
    "#;
    let result = run_code(src).unwrap();
    let v = as_f64_vec(result.as_ref().unwrap()).expect("expected tensor");
    // diff = [-1, 0, 1]，sign = [-1, 0, 1]
    assert!((v[0] - (-1.0)).abs() < 1e-6, "expected -1.0, got {}", v[0]);
    assert!((v[1] - 0.0).abs() < 1e-6, "expected 0.0, got {}", v[1]);
    assert!((v[2] - 1.0).abs() < 1e-6, "expected 1.0, got {}", v[2]);
}

#[test]
fn test_huber_loss_train_backward() {
    // huber_loss_train 内联展开（Interpreter 路径不加载 prelude）
    // abs 接入 autodiff 后，huber_loss 的 abs 部分梯度应正确
    // 线性区：L = δ*(|a|-0.5δ)，dL/da = δ*sign(a)
    // pred=5.0, target=0.0, delta=2.0 → a=5.0 > δ，dL/dpred = δ*sign(5) = 2.0
    let src = r#"
        new_grad();
        let pred = param(tensor[[5.0]]);
        let target = tensor[[0.0]];
        let a = pred - target;
        let abs_a = a.abs();
        let eps = 1e-12;
        let diff = abs_a - 2.0;
        let sign = diff / (diff.abs() + eps);
        let cond = (sign + 1.0) * 0.5;
        let quad = 0.5 * a * a;
        let linear = 2.0 * (abs_a - 1.0);
        let loss = select(cond, linear, quad);
        backward(loss);
        stop_grad();
        grad(pred)
    "#;
    let result = run_code(src).unwrap();
    let v = as_f64_vec(result.as_ref().unwrap()).expect("expected tensor");
    // 线性区 dL/dpred = δ*sign(a) = 2.0*1.0 = 2.0
    // abs 接入后梯度链完整：d(abs_a)/da = sign(a) = 1，d(linear)/d(abs_a) = δ = 2.0
    // select 反向 cond≈1 → d_linear = grad*cond_mask = 1*1 = 1
    // d(abs_a) = 1 * 2.0 = 2.0，d(a) = 2.0 * sign(5) = 2.0
    assert!((v[0] - 2.0).abs() < 0.3, "expected ~2.0 (linear branch with abs grad), got {}", v[0]);
}
