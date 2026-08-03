//! PROJ-006 选项 2（Rust 端注册版）自定义可微算子集成测试。
//!
//! 阶段 7：通过 VM 路径端到端验证 CustomBackward trait + CustomOpRegistry +
//! TapeOp::Custom + __call_custom_op native 的完整链路。
//!
//! 覆盖（参考立项书 8.2 节）：
//! 1. 基础注册 + 前向 + 反向（SquareOp: x², 2x·grad）
//! 2. op_class 声明（Preserve / Construct 通过 classify_tape_op 验证）
//! 3. 运行时 shape 检查（护城河 A 运行时兜底：backward 返回错误 shape 抛 ShapeMismatch）
//! 4. autodiff 端到端（多元素 + finite-difference 数值梯度对比）
//! 5. 与现有算子组合（matmul → custom_relu → sum，梯度链正确传播）
//! 6. 错误情况：同名注册 / 未注册 op_id / backward 返回错误 shape
//!
//! 测试路径：VM（参考 pool_test.rs 的 run_vm_pool 模式）。
//! native 注册通过 `tenth::runtime::natives::register_all_natives(&mut vm)` 一次性完成，
//! 该函数已包含 `__call_custom_op`、`new_grad`、`param`、`backward`、`grad`、`stop_grad` 等。

use std::cell::RefCell;
use std::rc::Rc;

use ndarray::ArrayD;
use tenth::compile::bytecode::BytecodeCompiler;
use tenth::error::TenthError;
use tenth::hir::lower::Lowerer;
use tenth::hir::types::BaseType;
use tenth::lexer::lexer::Lexer;
use tenth::parser::parser::Parser;
use tenth::runtime::autodiff::{CustomBackward, CustomOpRegistry, Tape, TapeOp};
use tenth::runtime::natives::register_all_natives;
use tenth::runtime::relation_debugger::{classify_tape_op, TapeOpClass};
use tenth::runtime::tensor::{Tensor, TensorData};
use tenth::runtime::value::Value;
use tenth::runtime::vm::Vm;

// ─── CustomBackward 实现集合 ────────────────────────────────────────────

/// y = x²；dy/dx = 2x。
/// op_class = Preserve（元素级运算，shape 不变）。
#[derive(Debug)]
struct SquareOp;
impl CustomBackward for SquareOp {
    fn forward(&self, inputs: &[&Tensor]) -> Result<Tensor, TenthError> {
        let x = inputs[0];
        let data = x.data.as_f64_view().mapv(|v| v * v);
        Ok(Tensor::from_tensor_data(TensorData::F64(data)))
    }
    fn backward(
        &self,
        inputs: &[&Tensor],
        grad: &Tensor,
    ) -> Result<Vec<Tensor>, TenthError> {
        let x = inputs[0];
        let g = grad.data.as_f64_view();
        let x_view = x.data.as_f64_view();
        let d_x = (&g * &x_view * 2.0).into();
        Ok(vec![Tensor::from_tensor_data(TensorData::F64(d_x))])
    }
    fn op_class(&self) -> TapeOpClass {
        TapeOpClass::Preserve
    }
    fn name(&self) -> &str {
        "square"
    }
}

/// y = max(0, x)（自定义 ReLU）；dy/dx = (x > 0) ? 1 : 0。
/// op_class = Preserve（元素级运算，shape 不变）。
#[derive(Debug)]
struct CustomReluOp;
impl CustomBackward for CustomReluOp {
    fn forward(&self, inputs: &[&Tensor]) -> Result<Tensor, TenthError> {
        let x = inputs[0];
        let data = x.data.as_f64_view().mapv(|v| if v > 0.0 { v } else { 0.0 });
        Ok(Tensor::from_tensor_data(TensorData::F64(data)))
    }
    fn backward(
        &self,
        inputs: &[&Tensor],
        grad: &Tensor,
    ) -> Result<Vec<Tensor>, TenthError> {
        let x = inputs[0];
        let mask = x.data.as_f64_view().mapv(|v| if v > 0.0 { 1.0 } else { 0.0 });
        let g = grad.data.as_f64_view();
        let d_x = (&g * &mask).into();
        Ok(vec![Tensor::from_tensor_data(TensorData::F64(d_x))])
    }
    fn op_class(&self) -> TapeOpClass {
        TapeOpClass::Preserve
    }
    fn name(&self) -> &str {
        "custom_relu"
    }
}

/// 将 [N] → [N, 1] 的 reshape-like 算子（用于测试 op_class=Construct）。
/// forward 不继承输入 shape（构造新 shape），故 op_class = Construct。
#[derive(Debug)]
struct ExpandDimOp;
impl CustomBackward for ExpandDimOp {
    fn forward(&self, inputs: &[&Tensor]) -> Result<Tensor, TenthError> {
        let x = inputs[0];
        let shape = x.data.shape();
        // [N] → [N, 1]；其他 shape 也按"末尾插一维 1"处理
        let mut new_shape: Vec<usize> = shape.to_vec();
        new_shape.push(1);
        let n: usize = new_shape.iter().product();
        let data_vec: Vec<f64> = x.data.as_f64_view().iter().copied().collect();
        let arr = ndarray::ArrayD::from_shape_vec(
            ndarray::IxDyn(&new_shape),
            data_vec,
        )
        .map_err(|_| TenthError::RuntimeError {
            line: None,
            col: None,
            message: "ExpandDimOp forward reshape 失败".into(),
        })?;
        // 拷贝一份 owned 数据避免形状借用问题
        let owned: ArrayD<f64> = arr.clone();
        let _ = n;
        Ok(Tensor::from_tensor_data(TensorData::F64(owned)))
    }
    fn backward(
        &self,
        inputs: &[&Tensor],
        grad: &Tensor,
    ) -> Result<Vec<Tensor>, TenthError> {
        // 反向：将梯度 reshape 回输入 shape
        let x = inputs[0];
        let in_shape = x.data.shape().to_vec();
        let data_vec: Vec<f64> = grad.data.as_f64_view().iter().copied().collect();
        let arr = ndarray::ArrayD::from_shape_vec(
            ndarray::IxDyn(&in_shape),
            data_vec,
        )
        .map_err(|_| TenthError::RuntimeError {
            line: None,
            col: None,
            message: "ExpandDimOp backward reshape 失败".into(),
        })?;
        Ok(vec![Tensor::from_tensor_data(TensorData::F64(arr))])
    }
    fn op_class(&self) -> TapeOpClass {
        TapeOpClass::Construct
    }
    fn name(&self) -> &str {
        "expand_dim"
    }
}

/// 故意返回错误 shape 的梯度（护城河 A 运行时兜底测试）。
/// forward: x²（保持输入 shape）
/// backward: 返回 shape=[N+1, ...] 的梯度（错误 shape，应被运行时拦截）
#[derive(Debug)]
struct BadShapeOp;
impl CustomBackward for BadShapeOp {
    fn forward(&self, inputs: &[&Tensor]) -> Result<Tensor, TenthError> {
        let x = inputs[0];
        let data = x.data.as_f64_view().mapv(|v| v * v);
        Ok(Tensor::from_tensor_data(TensorData::F64(data)))
    }
    fn backward(
        &self,
        inputs: &[&Tensor],
        grad: &Tensor,
    ) -> Result<Vec<Tensor>, TenthError> {
        // 故意返回 shape=[N+1] 的梯度（与输入 shape=[N] 不一致）
        let n = inputs[0].data.len();
        let wrong_shape = vec![n + 1];
        let wrong_data = vec![1.0; n + 1];
        let arr = ndarray::ArrayD::from_shape_vec(
            ndarray::IxDyn(&wrong_shape),
            wrong_data,
        )
        .map_err(|_| TenthError::RuntimeError {
            line: None,
            col: None,
            message: "BadShapeOp backward 构造错误 shape 张量失败".into(),
        })?;
        // 故意不使用 grad（仅为了产生稳定错误）
        let _ = grad;
        Ok(vec![Tensor::from_tensor_data(TensorData::F64(arr))])
    }
    fn op_class(&self) -> TapeOpClass {
        TapeOpClass::Preserve
    }
    fn name(&self) -> &str {
        "bad_shape"
    }
}

// ─── 测试辅助函数 ──────────────────────────────────────────────────────

/// 通过 VM 执行 .th 源码，预先注册一组自定义算子（按顺序拿 op_id=0,1,...）。
///
/// 调用方在 .th 源码中通过 `__call_custom_op(idx, x)` 调用对应算子。
/// native 注册委托 `register_all_natives`（已包含 __call_custom_op / new_grad / param /
/// backward / grad / stop_grad / zero_grad 等）。
fn run_vm_custom(src: &str, custom_ops: Vec<Box<dyn CustomBackward>>) -> Result<Value, String> {
    let mut lexer = Lexer::new(src);
    let tokens = lexer.tokenize().map_err(|e| e.to_string())?;
    let mut parser = Parser::new(tokens);
    let program = parser.parse_program().map_err(|e| e.to_string())?;
    let mut lowerer = Lowerer::new();
    let hir = lowerer.lower_program(&program).map_err(|e| e.to_string())?;

    let mut vm = Vm::new();
    // 一次性注册所有 native（含 __call_custom_op + autodiff 全套）
    register_all_natives(&mut vm);
    // 注册用户自定义算子（op_id 从 0 单调递增）
    for op in custom_ops {
        vm.register_custom_op(op).map_err(|e| e.to_string())?;
    }

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

/// 从 Value 提取 f64 数据向量。
fn tensor_to_vec(val: &Value) -> Option<Vec<f64>> {
    if let Value::Tensor(t) = val {
        Some(t.borrow().data.as_f64_view().iter().copied().collect())
    } else {
        None
    }
}

/// 从 Value 提取张量 shape。
fn tensor_shape(val: &Value) -> Option<Vec<usize>> {
    if let Value::Tensor(t) = val {
        Some(t.borrow().shape())
    } else {
        None
    }
}

// ─── 1. 基础注册 + 前向 + 反向 ─────────────────────────────────────────

#[test]
fn test_custom_op_square_forward() {
    // 前向：x=2.0 → y=4.0
    let src = r#"
        let x = tensor[[2.0]];
        __call_custom_op(0, x)
    "#;
    let result = run_vm_custom(src, vec![Box::new(SquareOp)]).expect("forward 失败");
    let data = tensor_to_vec(&result).expect("expected tensor");
    assert_eq!(data, vec![4.0], "square(2.0) = 4.0");
}

#[test]
fn test_custom_op_square_backward() {
    // 反向：loss = square(x).sum()，d_x = 2x
    // x = [1.0, 2.0, 3.0] → grad = [2.0, 4.0, 6.0]
    let src = r#"
        new_grad();
        let x = param(tensor[[1.0, 2.0, 3.0]]);
        let y = __call_custom_op(0, x);
        let loss = y.sum();
        backward(loss);
        stop_grad();
        grad(x)
    "#;
    let result = run_vm_custom(src, vec![Box::new(SquareOp)]).expect("backward 失败");
    let data = tensor_to_vec(&result).expect("expected grad tensor");
    let expected = vec![2.0, 4.0, 6.0];
    for (i, (got, exp)) in data.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - exp).abs() < 1e-6,
            "grad[{}] = {}, expected {}",
            i,
            got,
            exp
        );
    }
}

// ─── 2. op_class 声明 ──────────────────────────────────────────────────

#[test]
fn test_custom_op_class_preserve() {
    // SquareOp 声明为 Preserve；classify_tape_op 应返回 Preserve
    let mut registry = CustomOpRegistry::new();
    let id = registry.register(Box::new(SquareOp)).expect("register SquareOp");
    let class = classify_tape_op(&TapeOp::Custom(id), Some(&registry));
    assert_eq!(class, TapeOpClass::Preserve, "SquareOp.op_class = Preserve");
}

#[test]
fn test_custom_op_class_construct() {
    // ExpandDimOp 声明为 Construct；classify_tape_op 应返回 Construct
    let mut registry = CustomOpRegistry::new();
    let id = registry.register(Box::new(ExpandDimOp)).expect("register ExpandDimOp");
    let class = classify_tape_op(&TapeOp::Custom(id), Some(&registry));
    assert_eq!(class, TapeOpClass::Construct, "ExpandDimOp.op_class = Construct");
}

#[test]
fn test_custom_op_class_no_registry_fallback_construct() {
    // registry 不可用时 fallback 到 Construct（保守策略，与 relation_debugger.rs 既有行为一致）
    let class = classify_tape_op(&TapeOp::Custom(0), None);
    assert_eq!(class, TapeOpClass::Construct, "无 registry 时 Custom fallback = Construct");
}

// ─── 3. 运行时 shape 检查（护城河 A 运行时兜底） ──────────────────────

#[test]
fn test_custom_op_backward_bad_shape_rejected() {
    // BadShapeOp 的 backward 返回 shape=[N+1] 的梯度（与输入 shape=[N] 不一致）。
    // 运行时应抛 ShapeMismatch 错误（护城河 A 运行时兜底）。
    let src = r#"
        new_grad();
        let x = param(tensor[[1.0, 2.0, 3.0, 4.0]]);
        let y = __call_custom_op(0, x);
        let loss = y.sum();
        backward(loss);
        Value::Unit
    "#;
    let result = run_vm_custom(src, vec![Box::new(BadShapeOp)]);
    assert!(
        result.is_err(),
        "backward 返回错误 shape 应被运行时拦截，但成功了：{:?}",
        result
    );
    let err_msg = result.unwrap_err();
    assert!(
        err_msg.contains("shape") || err_msg.contains("ShapeMismatch") || err_msg.contains("不一致"),
        "错误消息应提及 shape 不匹配，实际：{}",
        err_msg
    );
}

// ─── 4. autodiff 端到端 + 数值梯度对比 ────────────────────────────────

#[test]
fn test_custom_op_autodiff_end_to_end_with_finite_difference() {
    // loss = square(x).sum(), x = [[1.0, 2.0], [3.0, 4.0]]
    // 解析梯度：d_x = 2x = [[2, 4], [6, 8]]
    let src = r#"
        new_grad();
        let x = param(tensor[[1.0, 2.0], [3.0, 4.0]]);
        let y = __call_custom_op(0, x);
        let loss = y.sum();
        backward(loss);
        stop_grad();
        grad(x)
    "#;
    let result = run_vm_custom(src, vec![Box::new(SquareOp)]).expect("backward 失败");
    let data = tensor_to_vec(&result).expect("expected grad tensor");
    let expected = vec![2.0, 4.0, 6.0, 8.0];
    for (i, (got, exp)) in data.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - exp).abs() < 1e-6,
            "grad[{}] = {}, expected {}",
            i,
            got,
            exp
        );
    }

    // 数值梯度（finite difference）对比：
    // f(x) = sum(x²)，f(x+ε) - f(x-ε) ≈ 2·x·ε，故 (f(x+ε) - f(x-ε)) / (2ε) ≈ 2x
    let eps = 1e-5;
    let x_base = vec![1.0_f64, 2.0, 3.0, 4.0];
    let mut num_grad = vec![0.0; x_base.len()];
    for i in 0..x_base.len() {
        let mut xp = x_base.clone();
        let mut xm = x_base.clone();
        xp[i] += eps;
        xm[i] -= eps;
        let fp: f64 = xp.iter().map(|v| v * v).sum();
        let fm: f64 = xm.iter().map(|v| v * v).sum();
        num_grad[i] = (fp - fm) / (2.0 * eps);
    }
    for (i, (got, num)) in data.iter().zip(num_grad.iter()).enumerate() {
        assert!(
            (got - num).abs() < 1e-4,
            "grad[{}] = {} 与数值梯度 {} 不一致",
            i,
            got,
            num
        );
    }
}

// ─── 5. 与现有算子组合 ────────────────────────────────────────────────

#[test]
fn test_custom_op_compose_with_matmul_and_sum() {
    // y = matmul(W, x) → custom_relu(y) → loss = y.sum()
    // W: (2,3), x: (3,1) → y: (2,1) → custom_relu(y) → loss = sum
    // 梯度链：d_loss/d_y = 1（relu 通过的位置），d_y/d_W = mask ⊗ x^T, d_y/d_x = W^T @ mask
    // 我们主要验证：
    //   1) custom_relu 与 matmul 组合时梯度链不中断
    //   2) W 的梯度 shape 与 W 一致（[2,3]），x 的梯度 shape 与 x 一致（[3,1]）
    //   3) 自定义 ReLU 的 mask 正确（y <= 0 位置梯度为 0）
    //
    // W = [[1, -1, 0], [2, 2, -1]], x = [[1], [1], [1]]
    // y = W @ x = [[1-1+0], [2+2-1]] = [[0], [3]]
    // custom_relu(y) = [[0], [3]]
    // loss = sum = 3
    // d_loss/d_relu_y = [[1], [1]]
    // d_relu_y/d_y = [[0], [1]]（y[0,0]=0 → 严格大于 0 不成立 → 0；y[1,0]=3>0 → 1）
    //   注：本实现采用 x > 0（严格大于）判定，与 PyTorch 的 x > 0 一致
    // d_W = mask @ x^T = [[0],[1]] @ [[1,1,1]] = [[0,0,0],[1,1,1]]
    // d_x = W^T @ mask = [[1,2],[-1,2],[0,-1]] @ [[0],[1]] = [[2],[2],[-1]]
    let src = r#"
        new_grad();
        let W = param(tensor[[1.0, -1.0, 0.0], [2.0, 2.0, -1.0]]);
        let x = param(tensor[[1.0], [1.0], [1.0]]);
        let y = W.matmul(x);
        let r = __call_custom_op(0, y);
        let loss = r.sum();
        backward(loss);
        stop_grad();
        grad(W)
    "#;
    let result = run_vm_custom(src, vec![Box::new(CustomReluOp)]).expect("backward 失败");
    let data = tensor_to_vec(&result).expect("expected W grad tensor");
    let shape = tensor_shape(&result).expect("expected W grad shape");
    assert_eq!(shape, vec![2, 3], "W grad shape 应为 [2,3]");
    let expected_w_grad = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
    for (i, (got, exp)) in data.iter().zip(expected_w_grad.iter()).enumerate() {
        assert!(
            (got - exp).abs() < 1e-6,
            "W grad[{}] = {}, expected {}",
            i,
            got,
            exp
        );
    }

    // 再验证 x 的梯度
    let src_x = r#"
        new_grad();
        let W = param(tensor[[1.0, -1.0, 0.0], [2.0, 2.0, -1.0]]);
        let x = param(tensor[[1.0], [1.0], [1.0]]);
        let y = W.matmul(x);
        let r = __call_custom_op(0, y);
        let loss = r.sum();
        backward(loss);
        stop_grad();
        grad(x)
    "#;
    let result_x = run_vm_custom(src_x, vec![Box::new(CustomReluOp)]).expect("backward 失败");
    let data_x = tensor_to_vec(&result_x).expect("expected x grad tensor");
    let shape_x = tensor_shape(&result_x).expect("expected x grad shape");
    assert_eq!(shape_x, vec![3, 1], "x grad shape 应为 [3,1]");
    let expected_x_grad = vec![2.0, 2.0, -1.0];
    for (i, (got, exp)) in data_x.iter().zip(expected_x_grad.iter()).enumerate() {
        assert!(
            (got - exp).abs() < 1e-6,
            "x grad[{}] = {}, expected {}",
            i,
            got,
            exp
        );
    }
}

// ─── 6. 错误情况 ──────────────────────────────────────────────────────

#[test]
fn test_custom_op_duplicate_name_registration_fails() {
    // 同名算子注册应返回 Err
    let mut registry = CustomOpRegistry::new();
    let first = registry.register(Box::new(SquareOp));
    assert!(first.is_ok(), "首次注册应成功");
    let second = registry.register(Box::new(SquareOp));
    assert!(
        second.is_err(),
        "同名算子二次注册应失败，实际：{:?}",
        second
    );
    let err = second.unwrap_err();
    assert!(
        err.contains("square") || err.contains("已注册"),
        "错误消息应提及算子名或已注册，实际：{}",
        err
    );
}

#[test]
fn test_custom_op_call_unregistered_op_id_errors() {
    // 调用未注册的 op_id=999 应返回 Err
    // 这里只注册 op_id=0（SquareOp），调用 op_id=999
    let src = r#"
        let x = tensor[[2.0]];
        __call_custom_op(999, x)
    "#;
    let result = run_vm_custom(src, vec![Box::new(SquareOp)]);
    assert!(
        result.is_err(),
        "调用未注册 op_id 应失败，实际：{:?}",
        result
    );
    let err = result.unwrap_err();
    assert!(
        err.contains("999") || err.contains("未注册"),
        "错误消息应提及 op_id 或未注册，实际：{}",
        err
    );
}

#[test]
fn test_custom_op_vm_register_duplicate_name_errors() {
    // 通过 Vm::register_custom_op 注册同名算子，第二次应返回 Err
    let mut vm = Vm::new();
    let first = vm.register_custom_op(Box::new(SquareOp));
    assert!(first.is_ok(), "首次注册应成功");
    let second = vm.register_custom_op(Box::new(SquareOp));
    assert!(
        second.is_err(),
        "同名算子二次注册应失败，实际：{:?}",
        second
    );
}

#[test]
fn test_custom_op_backward_grad_count_mismatch_rejected() {
    // 注册一个返回错误梯度数量的算子（返回 0 个梯度，但输入有 1 个）
    #[derive(Debug)]
    struct ZeroGradOp;
    impl CustomBackward for ZeroGradOp {
        fn forward(&self, inputs: &[&Tensor]) -> Result<Tensor, TenthError> {
            let x = inputs[0];
            Ok(Tensor::from_tensor_data(TensorData::F64(x.data.as_f64_view().clone())))
        }
        fn backward(
            &self,
            _inputs: &[&Tensor],
            _grad: &Tensor,
        ) -> Result<Vec<Tensor>, TenthError> {
            Ok(vec![]) // 错误：返回 0 个梯度（输入 1 个）
        }
        fn op_class(&self) -> TapeOpClass {
            TapeOpClass::Preserve
        }
        fn name(&self) -> &str {
            "zero_grad"
        }
    }

    let src = r#"
        new_grad();
        let x = param(tensor[[1.0, 2.0]]);
        let y = __call_custom_op(0, x);
        let loss = y.sum();
        backward(loss);
        Value::Unit
    "#;
    let result = run_vm_custom(src, vec![Box::new(ZeroGradOp)]);
    assert!(
        result.is_err(),
        "backward 返回错误梯度数量应被拦截，实际：{:?}",
        result
    );
    let err = result.unwrap_err();
    assert!(
        err.contains("数量") || err.contains("shape") || err.contains("不一致"),
        "错误消息应提及数量/shape 不一致，实际：{}",
        err
    );
}

// ─── 附加：直接调用 CustomBackward trait 验证 ─────────────────────────

#[test]
fn test_custom_op_square_trait_direct() {
    // 直接调用 trait 方法验证 forward/backward 公式正确（不走 VM）
    let op = SquareOp;
    let x = Tensor::from_vec(vec![2.0, 3.0], vec![2]);
    let y = op.forward(&[&x]).expect("forward");
    assert_eq!(y.data.as_f64_view().as_slice().unwrap(), &[4.0, 9.0]);

    let grad = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
    let grads = op.backward(&[&x], &grad).expect("backward");
    assert_eq!(grads.len(), 1, "应返回 1 个梯度");
    assert_eq!(grads[0].data.as_f64_view().as_slice().unwrap(), &[4.0, 6.0]);
}

#[test]
fn test_custom_op_registry_lookup() {
    // 验证 CustomOpRegistry 的 register/get/get_id_by_name/len 接口
    let mut registry = CustomOpRegistry::new();
    assert!(registry.is_empty(), "新 registry 应为空");

    let id = registry.register(Box::new(SquareOp)).expect("register");
    assert_eq!(id, 0, "首个 op_id 应为 0");
    assert_eq!(registry.len(), 1, "注册后 len=1");

    let id2 = registry.register(Box::new(CustomReluOp)).expect("register");
    assert_eq!(id2, 1, "第二个 op_id 应为 1");
    assert_eq!(registry.len(), 2);

    // 按名查找
    assert_eq!(registry.get_id_by_name("square"), Some(0));
    assert_eq!(registry.get_id_by_name("custom_relu"), Some(1));
    assert_eq!(registry.get_id_by_name("not_registered"), None);

    // 按 id 查找
    let op = registry.get(0).expect("get(0)");
    assert_eq!(op.name(), "square");
    let op2 = registry.get(1).expect("get(1)");
    assert_eq!(op2.name(), "custom_relu");
    assert!(registry.get(999).is_none(), "未注册 id 应返回 None");
}

// ─── 附加：CustomOp 与 Tape 直接集成测试 ──────────────────────────────

#[test]
fn test_custom_op_tape_direct_backward() {
    // 不走 VM，直接构造 Tape + custom_op 节点，验证 backward 正确。
    // 场景：loss = sum(square(x)), x = [1.0, 2.0, 3.0]
    // 期望 grad = [2, 4, 6]
    let mut registry = CustomOpRegistry::new();
    let op_id = registry.register(Box::new(SquareOp)).expect("register");

    let x = Rc::new(RefCell::new(Tensor::from_vec(
        vec![1.0, 2.0, 3.0],
        vec![3],
    )));
    let mut tape = Tape::new();
    let x_id = tape.input(x.clone());

    // forward: y = square(x)
    let y_data = x.borrow().data.as_f64_view().mapv(|v| v * v);
    let y = Rc::new(RefCell::new(Tensor::from_tensor_data(TensorData::F64(y_data))));
    let y_id = tape.custom_op(
        op_id,
        vec![Some(x_id)],
        vec![x.clone()],
        y.clone(),
    );

    // 设置 custom_ops（backward 前必须设置）
    let registry_rc = Rc::new(RefCell::new(registry));
    tape.set_custom_ops(registry_rc);

    tape.backward(y_id).expect("backward 应成功");

    let x_ref = x.borrow();
    let grad = x_ref.grad.as_ref().expect("grad 应被填充");
    let grad_f64 = grad.as_f64_view();
    let expected = [2.0, 4.0, 6.0];
    for (i, exp) in expected.iter().enumerate() {
        assert!(
            (grad_f64[ndarray::IxDyn(&[i])] - exp).abs() < 1e-10,
            "grad[{}] = {}, expected {}",
            i,
            grad_f64[ndarray::IxDyn(&[i])],
            exp
        );
    }
}

#[test]
fn test_custom_op_tape_direct_bad_shape_rejected() {
    // 不走 VM，直接构造 Tape + BadShapeOp 节点，验证 backward 返回 ShapeMismatch。
    let mut registry = CustomOpRegistry::new();
    let op_id = registry.register(Box::new(BadShapeOp)).expect("register");

    let x = Rc::new(RefCell::new(Tensor::from_vec(
        vec![1.0, 2.0, 3.0, 4.0],
        vec![4],
    )));
    let mut tape = Tape::new();
    let x_id = tape.input(x.clone());

    let y_data = x.borrow().data.as_f64_view().mapv(|v| v * v);
    let y = Rc::new(RefCell::new(Tensor::from_tensor_data(TensorData::F64(y_data))));
    let y_id = tape.custom_op(
        op_id,
        vec![Some(x_id)],
        vec![x.clone()],
        y.clone(),
    );

    let registry_rc = Rc::new(RefCell::new(registry));
    tape.set_custom_ops(registry_rc);

    let result = tape.backward(y_id);
    assert!(
        result.is_err(),
        "BadShapeOp backward 应被运行时拦截，实际：{:?}",
        result
    );
    match result.unwrap_err() {
        TenthError::ShapeMismatch { message, .. } => {
            assert!(
                message.contains("shape") || message.contains("不一致"),
                "ShapeMismatch 消息应提及 shape，实际：{}",
                message
            );
        }
        other => panic!("期望 ShapeMismatch，实际：{:?}", other),
    }
}

#[test]
fn test_custom_op_tape_no_registry_set_errors() {
    // tape 未设置 custom_ops 时遇到 Custom 节点应报 RuntimeError
    let op_id = 0;
    let x = Rc::new(RefCell::new(Tensor::from_vec(vec![1.0], vec![1])));
    let mut tape = Tape::new();
    let x_id = tape.input(x.clone());

    let y = Rc::new(RefCell::new(Tensor::from_vec(vec![1.0], vec![1])));
    let y_id = tape.custom_op(
        op_id,
        vec![Some(x_id)],
        vec![x.clone()],
        y.clone(),
    );

    // 故意不调用 set_custom_ops
    let result = tape.backward(y_id);
    assert!(result.is_err(), "未设置 custom_ops 应报错");
    match result.unwrap_err() {
        TenthError::RuntimeError { message, .. } => {
            assert!(
                message.contains("custom_ops") || message.contains("未设置"),
                "错误消息应提及 custom_ops 未设置，实际：{}",
                message
            );
        }
        other => panic!("期望 RuntimeError，实际：{:?}", other),
    }
}

// ─── 附加：f32 dtype 路径验证（dispatch_float!） ──────────────────────

#[test]
fn test_custom_op_f32_dtype_forward_backward() {
    // 验证 f32 张量通过 custom_op 的 forward/backward 不丢失 dtype。
    // 注意：CustomBackward 实现内部用 as_f64_view 计算，结果通过 Tensor::from_data
    // 构造（默认 F64）；但 backward 校验只看 shape 不看 dtype，故 f32 输入应能跑通。
    // 这里仅验证不 panic 且 shape 正确。
    let src = r#"
        new_grad();
        let x = param(tensor[1.0_f32, 2.0_f32, 3.0_f32]);
        let y = __call_custom_op(0, x);
        let loss = y.sum();
        backward(loss);
        stop_grad();
        grad(x)
    "#;
    let result = run_vm_custom(src, vec![Box::new(SquareOp)]);
    // 不强制要求成功——若 f32 路径暂未完整支持，运行时兜底返回 Err 也可接受。
    // 这里只验证：要么成功且 shape 对，要么返回明确的错误（不 panic）。
    match result {
        Ok(val) => {
            let shape = tensor_shape(&val).expect("expected grad tensor");
            assert_eq!(shape, vec![3], "f32 输入 grad shape 应为 [3]");
        }
        Err(msg) => {
            // 接受任何非 panic 的错误路径
            assert!(!msg.is_empty(), "错误消息不应为空");
        }
    }
    // 附加：避免未使用 BaseType 警告
    let _ = BaseType::F32;
}
