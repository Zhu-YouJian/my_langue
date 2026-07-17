//! 广播规则完善测试（标准库扩展第二波 §五第 5 项）。
//!
//! 验证 Tensor 运算的 NumPy 风格广播语义：
//! - 一元广播：`(1,) → (3,)`、`(1,1) → (2,3)`
//! - 二元广播：`(3,) + (2,3) → (2,3)`、`(2,1) + (1,3) → (2,3)`、`(1,3) + (2,1) → (2,3)`
//! - 标量广播：`tensor + 1.0`、`1.0 * tensor`（通过 add_scalar/mul_scalar 等显式标量运算）
//! - 跨 dtype 广播：f32 + f64 → f64（按 promote_dtype 规则提升）
//! - 反向广播：`x: (3,) + y: (2,3)` 的 backward，`x.grad` 应为 `(3,)` 且每个元素是 y 对应列的 sum
//! - 错误情况：不可广播的 shape 应返回 Err（`(2,) + (3,)` 应失败）
//!
//! 审计结论：当前广播基础设施完整（`broadcast_shape` NumPy 风格 + `elementwise_binary`
//! + `unbroadcast` 反向广播 + `promote_dtype` 跨 dtype 提升）。本测试文件补齐运行时
//! 数值正确性的覆盖缺口，不修改运行时代码。
//!
//! 参考实现：
//! - `Tensor::broadcast_shape` (methods.rs:633)
//! - `Tensor::elementwise_binary` (methods.rs:671)
//! - `unbroadcast` (autodiff/backward.rs:1172)

use std::cell::RefCell;
use std::rc::Rc;

use tenth::hir::types::BaseType;
use tenth::runtime::autodiff::{Tape, TapeOp};
use tenth::runtime::tensor::Tensor;

// ── 辅助函数 ──────────────────────────────────────────────────────────

/// 构造 f64 张量。
fn make_tensor(data: Vec<f64>, shape: Vec<usize>) -> Tensor {
    Tensor::from_vec(data, shape)
}

/// 构造 f32 张量。
fn make_tensor_f32(data: Vec<f32>, shape: Vec<usize>) -> Tensor {
    Tensor::from_vec_f32(data, shape)
}

/// 构造 Rc<RefCell<Tensor>>（autodiff 测试用）。
fn make_tensor_rc(data: Vec<f64>, shape: Vec<usize>) -> Rc<RefCell<Tensor>> {
    Rc::new(RefCell::new(Tensor::from_vec(data, shape)))
}

/// 比较两个 f64 是否在 1e-10 范围内相等。
fn approx_eq(a: f64, b: f64) -> bool {
    (a - b).abs() < 1e-10
}

/// 断言两个 Vec<f64> 在 1e-10 范围内逐元素相等。
fn assert_vec_eq(actual: &[f64], expected: &[f64]) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "长度不匹配：actual={} expected={}",
        actual.len(),
        expected.len()
    );
    for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            approx_eq(*a, *e),
            "元素 {} 不匹配：actual={} expected={}",
            i,
            a,
            e
        );
    }
}

/// 把 Tensor 的数据按 row-major 顺序取为 Vec<f64>。
fn tensor_to_vec(t: &Tensor) -> Vec<f64> {
    let view = t.data.as_f64_view();
    view.iter().copied().collect()
}

// ════════════════════════════════════════════════════════════════════════════
// 1. 一元广播：标量/单元素张量广播到更大 shape
//
// 通过与同 shape 的另一个张量做二元运算来验证广播路径。
// NumPy 规则：右侧对齐，维度为 1 的轴可广播到任意大小。
// ════════════════════════════════════════════════════════════════════════════

#[test]
fn test_broadcast_scalar_to_1d() {
    // (1,) + (3,) → (3,)，单元素张量广播到 (3,)
    let a = make_tensor(vec![10.0], vec![1]);
    let b = make_tensor(vec![1.0, 2.0, 3.0], vec![3]);
    let c = a.add_tensor(&b).unwrap();
    assert_eq!(c.shape(), vec![3]);
    assert_vec_eq(&tensor_to_vec(&c), &[11.0, 12.0, 13.0]);
}

#[test]
fn test_broadcast_scalar_to_2d() {
    // (1,1) + (2,3) → (2,3)
    let a = make_tensor(vec![100.0], vec![1, 1]);
    let b = make_tensor(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let c = a.add_tensor(&b).unwrap();
    assert_eq!(c.shape(), vec![2, 3]);
    assert_vec_eq(&tensor_to_vec(&c), &[101.0, 102.0, 103.0, 104.0, 105.0, 106.0]);
}

#[test]
fn test_broadcast_1d_to_2d_row() {
    // (3,) + (1,3) → (1,3) → 实际结果 (1,3)
    // 验证 1D 与 2D 行向量广播
    let a = make_tensor(vec![10.0, 20.0, 30.0], vec![3]);
    let b = make_tensor(vec![1.0, 2.0, 3.0], vec![1, 3]);
    let c = a.add_tensor(&b).unwrap();
    assert_eq!(c.shape(), vec![1, 3]);
    assert_vec_eq(&tensor_to_vec(&c), &[11.0, 22.0, 33.0]);
}

// ════════════════════════════════════════════════════════════════════════════
// 2. 二元广播：常见 NumPy 广播场景
// ════════════════════════════════════════════════════════════════════════════

#[test]
fn test_broadcast_1d_plus_2d() {
    // (3,) + (2,3) → (2,3)
    // x = [10, 20, 30] (shape (3,))
    // y = [[1,2,3],[4,5,6]] (shape (2,3))
    // result = [[11,22,33],[14,25,36]]
    let x = make_tensor(vec![10.0, 20.0, 30.0], vec![3]);
    let y = make_tensor(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let r = x.add_tensor(&y).unwrap();
    assert_eq!(r.shape(), vec![2, 3]);
    assert_vec_eq(&tensor_to_vec(&r), &[11.0, 22.0, 33.0, 14.0, 25.0, 36.0]);
}

#[test]
fn test_broadcast_col_plus_row() {
    // (2,1) + (1,3) → (2,3)
    // col = [[10],[20]] (shape (2,1))
    // row = [[1,2,3]] (shape (1,3))
    // result = [[11,12,13],[21,22,23]]
    let col = make_tensor(vec![10.0, 20.0], vec![2, 1]);
    let row = make_tensor(vec![1.0, 2.0, 3.0], vec![1, 3]);
    let r = col.add_tensor(&row).unwrap();
    assert_eq!(r.shape(), vec![2, 3]);
    assert_vec_eq(&tensor_to_vec(&r), &[11.0, 12.0, 13.0, 21.0, 22.0, 23.0]);
}

#[test]
fn test_broadcast_row_plus_col() {
    // (1,3) + (2,1) → (2,3)（反向：行向量在前）
    let row = make_tensor(vec![1.0, 2.0, 3.0], vec![1, 3]);
    let col = make_tensor(vec![10.0, 20.0], vec![2, 1]);
    let r = row.add_tensor(&col).unwrap();
    assert_eq!(r.shape(), vec![2, 3]);
    assert_vec_eq(&tensor_to_vec(&r), &[11.0, 12.0, 13.0, 21.0, 22.0, 23.0]);
}

#[test]
fn test_broadcast_mul_col_row() {
    // (2,1) * (1,3) → (2,3)，乘法广播
    // col = [[10],[20]]，row = [[1,2,3]]
    // result = [[10,20,30],[20,40,60]]
    let col = make_tensor(vec![10.0, 20.0], vec![2, 1]);
    let row = make_tensor(vec![1.0, 2.0, 3.0], vec![1, 3]);
    let r = col.mul_tensor(&row).unwrap();
    assert_eq!(r.shape(), vec![2, 3]);
    assert_vec_eq(&tensor_to_vec(&r), &[10.0, 20.0, 30.0, 20.0, 40.0, 60.0]);
}

#[test]
fn test_broadcast_sub_2d_minus_1d() {
    // (2,3) - (3,) → (2,3)，减法广播
    let m = make_tensor(vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0], vec![2, 3]);
    let v = make_tensor(vec![1.0, 2.0, 3.0], vec![3]);
    let r = m.sub_tensor(&v).unwrap();
    assert_eq!(r.shape(), vec![2, 3]);
    assert_vec_eq(&tensor_to_vec(&r), &[9.0, 18.0, 27.0, 39.0, 48.0, 57.0]);
}

#[test]
fn test_broadcast_div_2d_by_col() {
    // (2,3) / (2,1) → (2,3)，除法广播（每行除以对应行的标量）
    let m = make_tensor(vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0], vec![2, 3]);
    let col = make_tensor(vec![10.0, 20.0], vec![2, 1]);
    let r = m.div_tensor(&col).unwrap();
    assert_eq!(r.shape(), vec![2, 3]);
    assert_vec_eq(&tensor_to_vec(&r), &[1.0, 2.0, 3.0, 2.0, 2.5, 3.0]);
}

#[test]
fn test_broadcast_same_shape_no_broadcast() {
    // 同 shape 加法：不应触发广播，结果直接逐元素相加
    let a = make_tensor(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let b = make_tensor(vec![10.0, 20.0, 30.0, 40.0], vec![2, 2]);
    let r = a.add_tensor(&b).unwrap();
    assert_eq!(r.shape(), vec![2, 2]);
    assert_vec_eq(&tensor_to_vec(&r), &[11.0, 22.0, 33.0, 44.0]);
}

#[test]
fn test_broadcast_3d_with_2d() {
    // (2,2,3) + (2,3) → (2,2,3)，3D 与 2D 广播（右侧对齐）
    let a = make_tensor(
        vec![1.0; 12].iter().enumerate().map(|(i, _)| (i as f64) + 1.0).collect(),
        vec![2, 2, 3],
    );
    let b = make_tensor(vec![100.0, 200.0, 300.0], vec![3]);
    let r = a.add_tensor(&b).unwrap();
    assert_eq!(r.shape(), vec![2, 2, 3]);
    // 第一个元素：1 + 100 = 101；第二个：2 + 200 = 202；第三个：3 + 300 = 303
    assert_vec_eq(
        &tensor_to_vec(&r)[..3],
        &[101.0, 202.0, 303.0],
    );
}

// ════════════════════════════════════════════════════════════════════════════
// 3. 标量广播：tensor + scalar / scalar * tensor
//
// Tenth 通过显式 add_scalar/mul_scalar 等方法实现标量运算（语义等价于标量广播）。
// ════════════════════════════════════════════════════════════════════════════

#[test]
fn test_scalar_add_tensor() {
    // tensor + 1.0
    let t = make_tensor(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let r = t.add_scalar(1.0);
    assert_eq!(r.shape(), vec![2, 2]);
    assert_vec_eq(&tensor_to_vec(&r), &[2.0, 3.0, 4.0, 5.0]);
}

#[test]
fn test_scalar_mul_tensor() {
    // 2.0 * tensor 等价于 tensor.mul_scalar(2.0)
    let t = make_tensor(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let r = t.mul_scalar(2.0);
    assert_eq!(r.shape(), vec![2, 2]);
    assert_vec_eq(&tensor_to_vec(&r), &[2.0, 4.0, 6.0, 8.0]);
}

#[test]
fn test_scalar_sub_tensor() {
    // tensor - 5.0
    let t = make_tensor(vec![10.0, 20.0, 30.0], vec![3]);
    let r = t.sub_scalar(5.0);
    assert_eq!(r.shape(), vec![3]);
    assert_vec_eq(&tensor_to_vec(&r), &[5.0, 15.0, 25.0]);
}

#[test]
fn test_scalar_div_tensor() {
    // tensor / 2.0
    let t = make_tensor(vec![10.0, 20.0, 30.0], vec![3]);
    let r = t.div_scalar(2.0);
    assert_eq!(r.shape(), vec![3]);
    assert_vec_eq(&tensor_to_vec(&r), &[5.0, 10.0, 15.0]);
}

#[test]
fn test_scalar_div_inv_tensor() {
    // 100.0 / tensor （标量除以张量）
    let t = make_tensor(vec![2.0, 5.0, 10.0], vec![3]);
    let r = t.div_scalar_inv(100.0);
    assert_eq!(r.shape(), vec![3]);
    assert_vec_eq(&tensor_to_vec(&r), &[50.0, 20.0, 10.0]);
}

#[test]
fn test_scalar_broadcast_combined() {
    // 组合：(tensor * 2.0 + 1.0) 与标量广播结合
    let t = make_tensor(vec![1.0, 2.0, 3.0], vec![3]);
    let r = t.mul_scalar(2.0).add_scalar(1.0);
    assert_vec_eq(&tensor_to_vec(&r), &[3.0, 5.0, 7.0]);
}

// ════════════════════════════════════════════════════════════════════════════
// 4. 跨 dtype 广播
//
// promote_dtype 规则（methods.rs:654）：
// - F64 + 任何 → F64
// - F32 + F32 → F32；F32 + F16/BF16 → F32
// - F16 + F16 → F16；BF16 + BF16 → BF16；F16 + BF16 → F32
// ════════════════════════════════════════════════════════════════════════════

#[test]
fn test_broadcast_f32_plus_f64_promotes_to_f64() {
    // f32 (3,) + f64 (1,) → f64 (3,)
    let a = make_tensor_f32(vec![1.0, 2.0, 3.0], vec![3]);
    let b = make_tensor(vec![10.0], vec![1]); // f64
    let r = a.add_tensor(&b).unwrap();
    assert_eq!(r.dtype(), BaseType::F64, "f32 + f64 应提升为 f64");
    assert_eq!(r.shape(), vec![3]);
    assert_vec_eq(&tensor_to_vec(&r), &[11.0, 12.0, 13.0]);
}

#[test]
fn test_broadcast_f64_plus_f32_promotes_to_f64() {
    // f64 (1,) + f32 (2,3) → f64 (2,3)
    let a = make_tensor(vec![100.0], vec![1]); // f64
    let b = make_tensor_f32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]); // f32
    let r = a.add_tensor(&b).unwrap();
    assert_eq!(r.dtype(), BaseType::F64);
    assert_eq!(r.shape(), vec![2, 3]);
    assert_vec_eq(&tensor_to_vec(&r), &[101.0, 102.0, 103.0, 104.0, 105.0, 106.0]);
}

#[test]
fn test_broadcast_f32_plus_f32_stays_f32() {
    // f32 (1,3) + f32 (2,1) → f32 (2,3)，dtype 保持
    let a = make_tensor_f32(vec![1.0, 2.0, 3.0], vec![1, 3]);
    let b = make_tensor_f32(vec![10.0, 20.0], vec![2, 1]);
    let r = a.add_tensor(&b).unwrap();
    assert_eq!(r.dtype(), BaseType::F32, "f32 + f32 应保持 f32");
    assert_eq!(r.shape(), vec![2, 3]);
    assert_vec_eq(&tensor_to_vec(&r), &[11.0, 12.0, 13.0, 21.0, 22.0, 23.0]);
}

#[test]
fn test_broadcast_f32_mul_f64_promotes_to_f64() {
    // f32 (2,1) * f64 (1,3) → f64 (2,3)，乘法跨 dtype
    let a = make_tensor_f32(vec![10.0, 20.0], vec![2, 1]);
    let b = make_tensor(vec![1.0, 2.0, 3.0], vec![1, 3]);
    let r = a.mul_tensor(&b).unwrap();
    assert_eq!(r.dtype(), BaseType::F64);
    assert_eq!(r.shape(), vec![2, 3]);
    assert_vec_eq(&tensor_to_vec(&r), &[10.0, 20.0, 30.0, 20.0, 40.0, 60.0]);
}

// ════════════════════════════════════════════════════════════════════════════
// 5. 反向广播（autodiff）：梯度 unbroadcast 数值正确性
//
// 反向广播规则：grad 通过 sum_axis 沿广播维度累加回原 shape。
// unbroadcast 实现 (backward.rs:1172)：右侧对齐 + 对 target_shape==1 且 grad>1 的轴求和。
// ════════════════════════════════════════════════════════════════════════════

#[test]
fn test_backward_add_broadcast_1d_to_2d() {
    // f(x, y) = x + y，x shape (3,)，y shape (2,3) → result (2,3)
    // backward(result).sum() 后：
    //   x.grad shape (3,)，每个元素 = 沿 axis 0 (size 2) 的 sum = 1+1 = 2
    //   y.grad shape (2,3)，每个元素 = 1
    let x = make_tensor_rc(vec![10.0, 20.0, 30.0], vec![3]);
    let y = make_tensor_rc(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);

    let mut tape = Tape::new();
    let _x_id = tape.input(x.clone());
    let _y_id = tape.input(y.clone());

    // 前向：用 add_tensor 计算结果（带广播）
    let result_tensor = {
        let x_ref = x.borrow();
        let y_ref = y.borrow();
        x_ref.add_tensor(&y_ref).unwrap()
    };
    let result = Rc::new(RefCell::new(result_tensor));
    let r_id = tape.binary_direct(TapeOp::Add, x.clone(), y.clone(), result.clone());

    tape.backward(r_id).unwrap();

    // x.grad shape 应为 (3,)，每个元素 = 2（沿广播轴 size=2 求和）
    let x_grad = x.borrow().grad.clone().expect("x.grad 应存在");
    let x_grad_view = x_grad.as_f64_view();
    assert_eq!(x_grad_view.shape(), &[3], "x.grad shape 应为 (3,)");
    for i in 0..3 {
        assert!(
            approx_eq(x_grad_view[[i]], 2.0),
            "x.grad[{}] = {}，期望 2.0",
            i,
            x_grad_view[[i]]
        );
    }

    // y.grad shape 应为 (2,3)，每个元素 = 1
    let y_grad = y.borrow().grad.clone().expect("y.grad 应存在");
    let y_grad_view = y_grad.as_f64_view();
    assert_eq!(y_grad_view.shape(), &[2, 3], "y.grad shape 应为 (2,3)");
    for i in 0..6 {
        let v = y_grad_view.iter().nth(i).copied().unwrap();
        assert!(approx_eq(v, 1.0), "y.grad[{}] = {}，期望 1.0", i, v);
    }
}

#[test]
fn test_backward_mul_broadcast_col_row() {
    // f(col, row) = col * row，col shape (2,1)，row shape (1,3) → result (2,3)
    //   d(col)/d(result) = row（广播回 (2,1) 时沿 axis 1 求和）
    //   d(row)/d(result) = col（广播回 (1,3) 时沿 axis 0 求和）
    // 假设 result.grad 全为 1（sum 后向）：
    //   col.grad[i,0] = sum_j(row[0,j] * 1) = row 各元素之和
    //   row.grad[0,j] = sum_i(col[i,0] * 1) = col 各元素之和
    let col = make_tensor_rc(vec![2.0, 3.0], vec![2, 1]);
    let row = make_tensor_rc(vec![4.0, 5.0, 6.0], vec![1, 3]);

    let mut tape = Tape::new();
    let _col_id = tape.input(col.clone());
    let _row_id = tape.input(row.clone());

    let result_tensor = {
        let col_ref = col.borrow();
        let row_ref = row.borrow();
        col_ref.mul_tensor(&row_ref).unwrap()
    };
    let result = Rc::new(RefCell::new(result_tensor));
    let r_id = tape.binary_direct(TapeOp::Mul, col.clone(), row.clone(), result.clone());

    tape.backward(r_id).unwrap();

    // col.grad shape (2,1)
    // d(f)/d(col[i,0]) = sum_j(row[0,j] * grad[i,j]) = sum_j(row[0,j]) = 4+5+6 = 15
    let col_grad = col.borrow().grad.clone().expect("col.grad 应存在");
    let col_grad_view = col_grad.as_f64_view();
    assert_eq!(col_grad_view.shape(), &[2, 1]);
    assert!(approx_eq(col_grad_view[[0, 0]], 15.0), "col.grad[0,0] = {}", col_grad_view[[0, 0]]);
    assert!(approx_eq(col_grad_view[[1, 0]], 15.0), "col.grad[1,0] = {}", col_grad_view[[1, 0]]);

    // row.grad shape (1,3)
    // d(f)/d(row[0,j]) = sum_i(col[i,0] * grad[i,j]) = sum_i(col[i,0]) = 2+3 = 5
    let row_grad = row.borrow().grad.clone().expect("row.grad 应存在");
    let row_grad_view = row_grad.as_f64_view();
    assert_eq!(row_grad_view.shape(), &[1, 3]);
    for j in 0..3 {
        assert!(
            approx_eq(row_grad_view[[0, j]], 5.0),
            "row.grad[0,{}] = {}，期望 5.0",
            j,
            row_grad_view[[0, j]]
        );
    }
}

#[test]
fn test_backward_sub_broadcast_2d_minus_1d() {
    // f(m, v) = m - v，m shape (2,3)，v shape (3,) → result (2,3)
    //   d(m)/d(result) = 1（保持 (2,3)）
    //   d(v)/d(result) = -1，沿 axis 0 求和后每个 v.grad[j] = -2
    let m = make_tensor_rc(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let v = make_tensor_rc(vec![1.0, 1.0, 1.0], vec![3]);

    let mut tape = Tape::new();
    let _m_id = tape.input(m.clone());
    let _v_id = tape.input(v.clone());

    let result_tensor = {
        let m_ref = m.borrow();
        let v_ref = v.borrow();
        m_ref.sub_tensor(&v_ref).unwrap()
    };
    let result = Rc::new(RefCell::new(result_tensor));
    let r_id = tape.binary_direct(TapeOp::Sub, m.clone(), v.clone(), result.clone());

    tape.backward(r_id).unwrap();

    // m.grad shape (2,3)，每个元素 = 1
    let m_grad = m.borrow().grad.clone().expect("m.grad 应存在");
    let m_grad_view = m_grad.as_f64_view();
    assert_eq!(m_grad_view.shape(), &[2, 3]);
    for v in m_grad_view.iter() {
        assert!(approx_eq(*v, 1.0), "m.grad 元素 = {}，期望 1.0", v);
    }

    // v.grad shape (3,)，每个元素 = -1 + -1 = -2（两个 m 行的梯度累加）
    let v_grad = v.borrow().grad.clone().expect("v.grad 应存在");
    let v_grad_view = v_grad.as_f64_view();
    assert_eq!(v_grad_view.shape(), &[3]);
    for j in 0..3 {
        assert!(
            approx_eq(v_grad_view[[j]], -2.0),
            "v.grad[{}] = {}，期望 -2.0",
            j,
            v_grad_view[[j]]
        );
    }
}

#[test]
fn test_backward_div_broadcast_2d_by_col() {
    // f(m, c) = m / c，m shape (2,3)，c shape (2,1) → result (2,3)
    //   d(m)/d(result) = 1/c（保持 (2,3)）
    //   d(c)/d(result) = -m/c^2，沿 axis 1 求和后 c.grad[i,0] = sum_j(-m[i,j]/c[i,0]^2)
    let m_data = vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0];
    let m = make_tensor_rc(m_data.clone(), vec![2, 3]);
    let c = make_tensor_rc(vec![10.0, 20.0], vec![2, 1]);

    let mut tape = Tape::new();
    let _m_id = tape.input(m.clone());
    let _c_id = tape.input(c.clone());

    let result_tensor = {
        let m_ref = m.borrow();
        let c_ref = c.borrow();
        m_ref.div_tensor(&c_ref).unwrap()
    };
    let result = Rc::new(RefCell::new(result_tensor));
    let r_id = tape.binary_direct(TapeOp::Div, m.clone(), c.clone(), result.clone());

    tape.backward(r_id).unwrap();

    // m.grad[i,j] = 1/c[i,0]
    let m_grad = m.borrow().grad.clone().expect("m.grad 应存在");
    let m_grad_view = m_grad.as_f64_view();
    assert_eq!(m_grad_view.shape(), &[2, 3]);
    // 第一行 c=10：m.grad = [0.1, 0.1, 0.1]
    assert!(approx_eq(m_grad_view[[0, 0]], 0.1));
    assert!(approx_eq(m_grad_view[[0, 1]], 0.1));
    assert!(approx_eq(m_grad_view[[0, 2]], 0.1));
    // 第二行 c=20：m.grad = [0.05, 0.05, 0.05]
    assert!(approx_eq(m_grad_view[[1, 0]], 0.05));
    assert!(approx_eq(m_grad_view[[1, 1]], 0.05));
    assert!(approx_eq(m_grad_view[[1, 2]], 0.05));

    // c.grad[i,0] = sum_j(-m[i,j]/c[i,0]^2)
    // 第一行：-(10+20+30)/100 = -60/100 = -0.6
    // 第二行：-(40+50+60)/400 = -150/400 = -0.375
    let c_grad = c.borrow().grad.clone().expect("c.grad 应存在");
    let c_grad_view = c_grad.as_f64_view();
    assert_eq!(c_grad_view.shape(), &[2, 1]);
    assert!(
        approx_eq(c_grad_view[[0, 0]], -0.6),
        "c.grad[0,0] = {}，期望 -0.6",
        c_grad_view[[0, 0]]
    );
    assert!(
        approx_eq(c_grad_view[[1, 0]], -0.375),
        "c.grad[1,0] = {}，期望 -0.375",
        c_grad_view[[1, 0]]
    );
}

#[test]
fn test_backward_add_broadcast_scalar_to_2d() {
    // f(m, s) = m + s，m shape (2,2)，s shape (1,1) → result (2,2)
    //   m.grad shape (2,2)，每个元素 = 1
    //   s.grad shape (1,1)，元素 = sum of all = 4
    let m = make_tensor_rc(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let s = make_tensor_rc(vec![100.0], vec![1, 1]);

    let mut tape = Tape::new();
    let _m_id = tape.input(m.clone());
    let _s_id = tape.input(s.clone());

    let result_tensor = {
        let m_ref = m.borrow();
        let s_ref = s.borrow();
        m_ref.add_tensor(&s_ref).unwrap()
    };
    let result = Rc::new(RefCell::new(result_tensor));
    let r_id = tape.binary_direct(TapeOp::Add, m.clone(), s.clone(), result.clone());

    tape.backward(r_id).unwrap();

    // m.grad：每个元素 = 1
    let m_grad = m.borrow().grad.clone().expect("m.grad 应存在");
    let m_grad_view = m_grad.as_f64_view();
    assert_eq!(m_grad_view.shape(), &[2, 2]);
    for v in m_grad_view.iter() {
        assert!(approx_eq(*v, 1.0));
    }

    // s.grad：所有 4 个 1 求和 = 4
    let s_grad = s.borrow().grad.clone().expect("s.grad 应存在");
    let s_grad_view = s_grad.as_f64_view();
    assert_eq!(s_grad_view.shape(), &[1, 1]);
    assert!(
        approx_eq(s_grad_view[[0, 0]], 4.0),
        "s.grad = {}，期望 4.0",
        s_grad_view[[0, 0]]
    );
}

// ════════════════════════════════════════════════════════════════════════════
// 6. 错误情况：不可广播的 shape 应返回 Err
// ════════════════════════════════════════════════════════════════════════════

#[test]
fn test_broadcast_incompatible_1d_1d_returns_err() {
    // (2,) + (3,) → 不可广播，返回 Err
    let a = make_tensor(vec![1.0, 2.0], vec![2]);
    let b = make_tensor(vec![1.0, 2.0, 3.0], vec![3]);
    let result = a.add_tensor(&b);
    assert!(result.is_err(), "(2,) + (3,) 应返回 Err");
    let err_msg = result.unwrap_err();
    assert!(
        err_msg.contains("broadcast") || err_msg.contains("广播"),
        "错误消息应提及 broadcast/广播，实际：{}",
        err_msg
    );
}

#[test]
fn test_broadcast_incompatible_2d_2d_returns_err() {
    // (2,3) + (3,2) → 不可广播（右侧对齐后 3≠2 且 2≠3），返回 Err
    let a = make_tensor(vec![1.0; 6], vec![2, 3]);
    let b = make_tensor(vec![1.0; 6], vec![3, 2]);
    let result = a.add_tensor(&b);
    assert!(result.is_err(), "(2,3) + (3,2) 应返回 Err");
}

#[test]
fn test_broadcast_incompatible_2d_1d_returns_err() {
    // (2,3) + (4,) → 不可广播（右侧对齐：3≠4 且 2≠1? 不，2 不等于 1 也不等于 4）
    // 实际：右侧对齐：(2,3) vs (4,)
    //   axis -1: 3 vs 4 → 不兼容
    let a = make_tensor(vec![1.0; 6], vec![2, 3]);
    let b = make_tensor(vec![1.0; 4], vec![4]);
    let result = a.add_tensor(&b);
    assert!(result.is_err(), "(2,3) + (4,) 应返回 Err");
}

#[test]
fn test_broadcast_incompatible_3d_2d_returns_err() {
    // (2,3,4) + (5,4) → 不可广播
    // 右侧对齐：axis -1: 4 vs 4 OK；axis -2: 3 vs 5 → 不兼容
    let a = make_tensor(vec![1.0; 24], vec![2, 3, 4]);
    let b = make_tensor(vec![1.0; 20], vec![5, 4]);
    let result = a.add_tensor(&b);
    assert!(result.is_err(), "(2,3,4) + (5,4) 应返回 Err");
}

#[test]
fn test_broadcast_incompatible_mul_returns_err() {
    // (2,) * (3,) → 不可广播，乘法也应返回 Err
    let a = make_tensor(vec![1.0, 2.0], vec![2]);
    let b = make_tensor(vec![1.0, 2.0, 3.0], vec![3]);
    let result = a.mul_tensor(&b);
    assert!(result.is_err(), "(2,) * (3,) 应返回 Err");
}

#[test]
fn test_broadcast_compatible_dim_one_always_broadcasts() {
    // 验证维度为 1 的轴可广播到任意大小（边界情况）
    // (1,5) + (3,5) → (3,5)
    let a = make_tensor(vec![10.0; 5], vec![1, 5]);
    let b = make_tensor(vec![1.0; 15], vec![3, 5]);
    let r = a.add_tensor(&b).unwrap();
    assert_eq!(r.shape(), vec![3, 5]);
    // 第一行：10 + 1 = 11
    assert_vec_eq(&tensor_to_vec(&r)[..5], &[11.0; 5]);
}

#[test]
fn test_broadcast_higher_dim_left_padding() {
    // (3,) + (2,3) → (2,3)，验证 1D 在左侧被 pad 成 (1,3) 再广播
    // 这是 NumPy 风格的左侧 padding 行为
    let a = make_tensor(vec![100.0, 200.0, 300.0], vec![3]);
    let b = make_tensor(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let r = a.add_tensor(&b).unwrap();
    assert_eq!(r.shape(), vec![2, 3]);
    // 第一行：100+1=101, 200+2=202, 300+3=303
    // 第二行：100+4=104, 200+5=205, 300+6=306
    assert_vec_eq(
        &tensor_to_vec(&r),
        &[101.0, 202.0, 303.0, 104.0, 205.0, 306.0],
    );
}
