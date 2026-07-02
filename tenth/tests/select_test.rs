// select 原语测试套件（论文 T47/T48/T50 验证）
// 覆盖：前向基本用例、广播用例、反向梯度用例、leaky_relu_select 对比、huber_loss

use tenth::lexer::lexer::Lexer;
use tenth::parser::parser::Parser;
use tenth::hir::lower::Lowerer;
use tenth::runtime::interpreter::Interpreter;
use tenth::runtime::value::Value;
use std::rc::Rc;
use std::cell::RefCell;

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

/// 从 Value 提取标量 f64
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

/// 从 Value 提取 tensor 的 f64 切片
fn as_f64_vec(val: &Value) -> Option<Vec<f64>> {
    match val {
        Value::Tensor(t) => {
            let data = t.borrow().data.as_f64_view();
            Some(data.iter().cloned().collect())
        }
        _ => None,
    }
}

// ── 前向基本用例 ──

#[test]
fn test_select_forward_basic() {
    // cond = [1, 0, 1, 0], then = [10, 20, 30, 40], else = [100, 200, 300, 400]
    // result = [10, 200, 30, 400]
    let src = r#"
        let cond = tensor[[1.0, 0.0, 1.0, 0.0]];
        let then = tensor[[10.0, 20.0, 30.0, 40.0]];
        let else_ = tensor[[100.0, 200.0, 300.0, 400.0]];
        select(cond, then, else_)
    "#;
    let result = run_code(src).unwrap();
    let v = as_f64_vec(result.as_ref().unwrap()).expect("expected tensor");
    assert_eq!(v, vec![10.0, 200.0, 30.0, 400.0]);
}

#[test]
fn test_select_forward_truthy_threshold() {
    // cond > 0.5 视为 true：0.6 → then, 0.4 → else
    let src = r#"
        let cond = tensor[[0.6, 0.4, 0.5]];
        let then = tensor[[1.0, 2.0, 3.0]];
        let else_ = tensor[[10.0, 20.0, 30.0]];
        select(cond, then, else_)
    "#;
    let result = run_code(src).unwrap();
    let v = as_f64_vec(result.as_ref().unwrap()).expect("expected tensor");
    // 0.6 > 0.5 → then=1.0; 0.4 < 0.5 → else=20.0; 0.5 不 > 0.5 → else=30.0
    assert_eq!(v, vec![1.0, 20.0, 30.0]);
}

// ── 广播用例 ──

#[test]
fn test_select_forward_broadcast_cond_scalar() {
    // cond 标量 1.0（true），then/else 张量 → 选 then
    let src = r#"
        let cond = tensor[[1.0]];
        let then = tensor[[10.0, 20.0, 30.0]];
        let else_ = tensor[[100.0, 200.0, 300.0]];
        select(cond, then, else_)
    "#;
    let result = run_code(src).unwrap();
    let v = as_f64_vec(result.as_ref().unwrap()).expect("expected tensor");
    assert_eq!(v, vec![10.0, 20.0, 30.0]);
}

#[test]
fn test_select_forward_broadcast_cond_scalar_false() {
    // cond 标量 0.0（false），then/else 张量 → 选 else
    let src = r#"
        let cond = tensor[[0.0]];
        let then = tensor[[10.0, 20.0, 30.0]];
        let else_ = tensor[[100.0, 200.0, 300.0]];
        select(cond, then, else_)
    "#;
    let result = run_code(src).unwrap();
    let v = as_f64_vec(result.as_ref().unwrap()).expect("expected tensor");
    assert_eq!(v, vec![100.0, 200.0, 300.0]);
}

#[test]
fn test_select_forward_broadcast_mixed() {
    // cond [2,1]，then [2,1]，else 标量 [1]
    // 第 0 行 cond=1 → then; 第 1 行 cond=0 → else
    let src = r#"
        let cond = tensor[[1.0], [0.0]];
        let then = tensor[[10.0], [20.0]];
        let else_ = tensor[[99.0]];
        select(cond, then, else_)
    "#;
    let result = run_code(src).unwrap();
    let v = as_f64_vec(result.as_ref().unwrap()).expect("expected tensor");
    assert_eq!(v, vec![10.0, 99.0]);
}

// ── 反向梯度用例 ──

#[test]
fn test_select_backward_then_grad() {
    // cond = [1, 1], then = x, else = 0
    // result = x, d_result/d_then = 1, d_then = grad * 1
    // backward(sum(result)) → grad = [1, 1], d_then = [1, 1]
    let src = r#"
        new_grad();
        let cond = tensor[[1.0, 1.0]];
        let x = param(tensor[[3.0, 5.0]]);
        let else_ = tensor[[0.0, 0.0]];
        let y = select(cond, x, else_);
        backward(y);
        stop_grad();
        grad(x)
    "#;
    let result = run_code(src).unwrap();
    let v = as_f64_vec(result.as_ref().unwrap()).expect("expected tensor");
    // d_then = grad * cond_mask = [1,1] * [1,1] = [1,1]
    assert!((v[0] - 1.0).abs() < 1e-6, "expected 1.0, got {}", v[0]);
    assert!((v[1] - 1.0).abs() < 1e-6, "expected 1.0, got {}", v[1]);
}

#[test]
fn test_select_backward_else_grad() {
    // cond = [0, 0], then = 0, else = x
    // result = x, d_result/d_else = 1, d_else = grad * 1
    let src = r#"
        new_grad();
        let cond = tensor[[0.0, 0.0]];
        let then = tensor[[0.0, 0.0]];
        let x = param(tensor[[3.0, 5.0]]);
        let y = select(cond, then, x);
        backward(y);
        stop_grad();
        grad(x)
    "#;
    let result = run_code(src).unwrap();
    let v = as_f64_vec(result.as_ref().unwrap()).expect("expected tensor");
    // d_else = grad * (1 - cond_mask) = [1,1] * [1,1] = [1,1]
    assert!((v[0] - 1.0).abs() < 1e-6, "expected 1.0, got {}", v[0]);
    assert!((v[1] - 1.0).abs() < 1e-6, "expected 1.0, got {}", v[1]);
}

#[test]
fn test_select_backward_mixed() {
    // cond = [1, 0], then = x, else = x（同一个 x）
    // result[i] = x[i]（无论 cond），d_result/d_x = 1
    // 但 select 阻断 cond，d_then = grad*mask, d_else = grad*(1-mask)
    // x 同时作为 then 和 else，梯度累加：d_x = d_then + d_else = grad*mask + grad*(1-mask) = grad
    let src = r#"
        new_grad();
        let cond = tensor[[1.0, 0.0]];
        let x = param(tensor[[3.0, 5.0]]);
        let y = select(cond, x, x);
        backward(y);
        stop_grad();
        grad(x)
    "#;
    let result = run_code(src).unwrap();
    let v = as_f64_vec(result.as_ref().unwrap()).expect("expected tensor");
    // d_x = grad*mask + grad*(1-mask) = [1,1]（全 1，因 mask+(1-mask)=1）
    assert!((v[0] - 1.0).abs() < 1e-6, "expected 1.0, got {}", v[0]);
    assert!((v[1] - 1.0).abs() < 1e-6, "expected 1.0, got {}", v[1]);
}

#[test]
fn test_select_backward_partial() {
    // cond = [1, 0, 1], then = x = [a, b, c], else = 0
    // result = [a, 0, c], sum = a + c
    // d_then = grad * mask = [1, 0, 1] * [1, 0, 1] = [1, 0, 1]
    let src = r#"
        new_grad();
        let cond = tensor[[1.0, 0.0, 1.0]];
        let x = param(tensor[[3.0, 5.0, 7.0]]);
        let else_ = tensor[[0.0, 0.0, 0.0]];
        let y = select(cond, x, else_);
        backward(y);
        stop_grad();
        grad(x)
    "#;
    let result = run_code(src).unwrap();
    let v = as_f64_vec(result.as_ref().unwrap()).expect("expected tensor");
    // d_then = grad * cond_mask = [1,1,1] * [1,0,1] = [1, 0, 1]
    assert!((v[0] - 1.0).abs() < 1e-6, "expected 1.0, got {}", v[0]);
    assert!((v[1] - 0.0).abs() < 1e-6, "expected 0.0, got {}", v[1]);
    assert!((v[2] - 1.0).abs() < 1e-6, "expected 1.0, got {}", v[2]);
}

// ── leaky_relu_select 与算术编码版本对比 ──

#[test]
fn test_leaky_relu_select_vs_arithmetic_positive() {
    // x > 0：两版本应一致
    // leaky_relu(x) = x（x>0 时）
    // leaky_relu_select(x) = select(cond≈1, x, 0.01*x) = x
    let src = r#"
        let x = tensor[[2.0, 5.0, 10.0]];
        let eps = 1e-12;
        let sign = x / (x.abs() + eps);
        let cond = (sign + 1.0) * 0.5;
        let sel = select(cond, x, 0.01 * x);
        let arith = x.relu() + 0.01 * (-x).relu();
        sel - arith
    "#;
    let result = run_code(src).unwrap();
    let v = as_f64_vec(result.as_ref().unwrap()).expect("expected tensor");
    for diff in &v {
        assert!(diff.abs() < 1e-6, "leaky_relu_select 与算术版差异过大：{}", diff);
    }
}

#[test]
fn test_leaky_relu_select_vs_arithmetic_negative() {
    // x < 0：两版本应一致
    // leaky_relu(x) = 0.01*x（x<0 时，结果为负）
    // 算术等价编码：x.relu() + 0.01 * (x - x.relu())
    //   x>0: x + 0.01*0 = x ✓   x<0: 0 + 0.01*x = 0.01*x ✓
    // leaky_relu_select(x) = select(cond≈0, x, 0.01*x) = 0.01*x
    let src = r#"
        let x = tensor[[-2.0, -5.0, -10.0]];
        let eps = 1e-12;
        let sign = x / (x.abs() + eps);
        let cond = (sign + 1.0) * 0.5;
        let sel = select(cond, x, 0.01 * x);
        let arith = x.relu() + 0.01 * (x - x.relu());
        sel - arith
    "#;
    let result = run_code(src).unwrap();
    let v = as_f64_vec(result.as_ref().unwrap()).expect("expected tensor");
    for diff in &v {
        assert!(diff.abs() < 1e-6, "leaky_relu_select 与算术版差异过大：{}", diff);
    }
}

// ── huber_loss 前向 + 反向用例 ──

#[test]
fn test_huber_loss_forward_quadratic_region() {
    // |a| < δ：二次区，L = 0.5 * a²
    // pred=1.0, target=0.0, delta=2.0 → a=1.0, |a|=1 < 2 → L = 0.5*1 = 0.5
    let src = r#"
        let pred = tensor[[1.0]];
        let target = tensor[[0.0]];
        let a = pred - target;
        let abs_a = a.abs();
        let eps = 1e-12;
        let diff = abs_a - 2.0;
        let sign = diff / (diff.abs() + eps);
        let cond = (sign + 1.0) * 0.5;
        let quad = 0.5 * a * a;
        let linear = 2.0 * (abs_a - 0.5 * 2.0);
        let loss = select(cond, linear, quad);
        loss.mean()
    "#;
    let result = run_code(src).unwrap();
    let g = as_f64(result.as_ref().unwrap()).expect("expected numeric value");
    // |a|=1 < δ=2 → cond≈0 → select quad = 0.5*1² = 0.5
    assert!((g - 0.5).abs() < 1e-3, "expected 0.5, got {}", g);
}

#[test]
fn test_huber_loss_forward_linear_region() {
    // |a| > δ：线性区，L = δ*(|a| - 0.5*δ)
    // pred=5.0, target=0.0, delta=2.0 → a=5.0, |a|=5 > 2 → L = 2*(5-1) = 8
    let src = r#"
        let pred = tensor[[5.0]];
        let target = tensor[[0.0]];
        let a = pred - target;
        let abs_a = a.abs();
        let eps = 1e-12;
        let diff = abs_a - 2.0;
        let sign = diff / (diff.abs() + eps);
        let cond = (sign + 1.0) * 0.5;
        let quad = 0.5 * a * a;
        let linear = 2.0 * (abs_a - 0.5 * 2.0);
        let loss = select(cond, linear, quad);
        loss.mean()
    "#;
    let result = run_code(src).unwrap();
    let g = as_f64(result.as_ref().unwrap()).expect("expected numeric value");
    // |a|=5 > δ=2 → cond≈1 → select linear = 2*(5-1) = 8
    assert!((g - 8.0).abs() < 1e-3, "expected 8.0, got {}", g);
}

#[test]
fn test_huber_loss_backward_quadratic() {
    // 二次区：L = 0.5*a², dL/da = a, dL/dpred = a
    // pred=1.0, target=0.0, delta=2.0 → a=1.0, dL/dpred = 1.0
    let src = r#"
        new_grad();
        let pred = param(tensor[[1.0]]);
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
    // 二次区 dL/dpred = a = 1.0
    assert!((v[0] - 1.0).abs() < 1e-3, "expected 1.0, got {}", v[0]);
}

#[test]
fn test_huber_loss_backward_linear() {
    // 线性区：L = δ*(|a| - 0.5*δ), dL/da = δ*sign(a), dL/dpred = δ*sign(a)
    // pred=5.0, target=0.0, delta=2.0 → a=5.0 > δ > 0，故 |a|=a，dL/dpred = δ = 2.0
    // 注：abs 尚未接入 autodiff（pre-existing 限制），故线性分支用 a 直接表达（a>0 时 |a|=a），
    // cond 仍用 abs_a 计算（cond 不可微，不影响梯度路径）。
    let src = r#"
        new_grad();
        let pred = param(tensor[[5.0]]);
        let target = tensor[[0.0]];
        let a = pred - target;
        let abs_a = a.abs();
        let eps = 1e-12;
        let diff = abs_a - 2.0;
        let sign_diff = diff / (diff.abs() + eps);
        let cond = (sign_diff + 1.0) * 0.5;
        let quad = 0.5 * a * a;
        let linear = 2.0 * (a - 1.0);
        let loss = select(cond, linear, quad);
        backward(loss);
        stop_grad();
        grad(pred)
    "#;
    let result = run_code(src).unwrap();
    let v = as_f64_vec(result.as_ref().unwrap()).expect("expected tensor");
    // 线性区 dL/dpred = δ*sign(a) = 2.0*1.0 = 2.0
    assert!((v[0] - 2.0).abs() < 1e-3, "expected 2.0, got {}", v[0]);
}

// ── VM 路径验证（select native 在 VM 下可用）──

#[test]
fn test_select_vm_path() {
    // 验证 select 在 VM 路径下前向正确（使用顶层表达式以走 MakeTensor 字面量路径）
    use tenth::runtime::vm::Vm;
    use tenth::compile::bytecode::BytecodeCompiler;

    let src = r#"
        select(tensor[[1.0, 0.0, 1.0]], tensor[[10.0, 20.0, 30.0]], tensor[[100.0, 200.0, 300.0]])
    "#;
    let mut lexer = Lexer::new(src);
    let tokens = lexer.tokenize().unwrap();
    let mut parser = Parser::new(tokens);
    let program = parser.parse_program().unwrap();
    let mut lowerer = Lowerer::new();
    let hir = lowerer.lower_program(&program).unwrap();

    let mut vm = Vm::new();
    vm.add_native("println".into(), |_vm, args| {
        for a in args { print!("{a}"); }
        println!();
        Ok(Value::Unit)
    });
    vm.add_native("Vec::new".into(), |_vm, _args| {
        Ok(Value::Vec(Rc::new(RefCell::new(Vec::new()))))
    });
    // 注册 tensor native：parser 将 tensor[[...]] 解析为 Call{func:"tensor", args:[TensorLiteral]}，
    // 字面量经 Op::MakeTensor 构造为 Tensor 后，此 native 仅作恒等返回（与 main.rs::register_natives 一致）。
    vm.add_native("tensor".into(), |_vm, args| {
        if args.len() == 1 {
            Ok(args[0].clone())
        } else {
            Err(tenth::error::TenthError::RuntimeError { message: "tensor() 参数异常".into() })
        }
    });
    // 注册 select native（与 main.rs::register_natives 一致）
    vm.add_native("select".into(), |vm, args| {
        if args.len() < 3 {
            return Err(tenth::error::TenthError::RuntimeError {
                message: "select 期望三个参数".into(),
            });
        }
        let (cond, then, else_) = match (&args[0], &args[1], &args[2]) {
            (Value::Tensor(c), Value::Tensor(t), Value::Tensor(e)) => (c.clone(), t.clone(), e.clone()),
            _ => return Err(tenth::error::TenthError::RuntimeError {
                message: "select 期望三个张量参数".into(),
            }),
        };
        let result_tensor = tenth::runtime::tensor::Tensor::select(&cond.borrow(), &then.borrow(), &else_.borrow())
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

    for func in &hir.functions {
        let compiler = BytecodeCompiler::new();
        match compiler.compile(func) {
            Ok((chunk, closures)) => {
                vm.add_fn(func.name.clone(), chunk);
                for (name, closure_chunk) in closures {
                    vm.add_fn(name, closure_chunk);
                }
            }
            Err(e) => panic!("compile error: {}", e),
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
            Err(e) => panic!("compile error: {}", e),
        }
    }
    let result = vm.call("main").unwrap();
    let v = as_f64_vec(&result).expect("expected tensor");
    assert_eq!(v, vec![10.0, 200.0, 30.0]);
}
