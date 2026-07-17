//! F16/BF16 算子覆盖度测试 — Wave 2 选项 3 实施阶段。
//!
//! 覆盖审计报告 P0/P1/P2 修复项：
//! - P0：layer_norm F16/BF16 panic 修复（不 panic + 数值正确）
//! - P1：matmul/bmm/im2col/max_pool2d/avg_pool2d F16/BF16 原生路径（dtype 保持 + 数值正确）
//! - P2：sum_axis/cat/select F16/BF16 分支（dtype 保持）
//! - autodiff 端到端：F16/BF16 张量的 matmul + relu + sum loss 的 backward
//!
//! 测试策略：
//! - 直接调用 Tensor API（不经 VM/解释器），减少环境耦合
//! - 数值正确性：与 F32/F64 结果对比，容差按精度选择
//! - dtype 保持：assert dtype 与输入一致
//! - autodiff：用 Tape API 直接构造计算图（参考 broadcast_test.rs 模式）

use std::cell::RefCell;
use std::rc::Rc;

use half::{bf16, f16};
use tenth::hir::types::BaseType;
use tenth::runtime::autodiff::{Tape, TapeOp};
use tenth::runtime::tensor::Tensor;

// ── 辅助函数 ──────────────────────────────────────────────────────────

/// 构造 f64 张量
fn make_tensor(data: Vec<f64>, shape: Vec<usize>) -> Tensor {
    Tensor::from_vec(data, shape)
}

/// 构造 f32 张量
fn make_tensor_f32(data: Vec<f32>, shape: Vec<usize>) -> Tensor {
    Tensor::from_vec_f32(data, shape)
}

/// 构造 f16 张量
fn make_tensor_f16(data: Vec<f16>, shape: Vec<usize>) -> Tensor {
    Tensor::from_vec_f16(data, shape)
}

/// 构造 bf16 张量
fn make_tensor_bf16(data: Vec<bf16>, shape: Vec<usize>) -> Tensor {
    Tensor::from_vec_bf16(data, shape)
}

/// 把 Vec<f64> 转为 Vec<f16>
fn to_f16_vec(data: &[f64]) -> Vec<f16> {
    data.iter().map(|v| f16::from_f64(*v)).collect()
}

/// 把 Vec<f64> 转为 Vec<bf16>
fn to_bf16_vec(data: &[f64]) -> Vec<bf16> {
    data.iter().map(|v| bf16::from_f64(*v)).collect()
}

/// 把 Tensor 数据按 row-major 取为 Vec<f64>（任意 dtype 都 cast 到 f64）
fn tensor_to_vec(t: &Tensor) -> Vec<f64> {
    let view = t.data.as_f64_view();
    view.iter().copied().collect()
}

/// 比较两个 f64 是否在容差范围内相等
fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() < tol
}

/// 断言两个 Vec<f64> 在容差范围内逐元素相等
fn assert_vec_approx(actual: &[f64], expected: &[f64], tol: f64, ctx: &str) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "{}：长度不匹配 actual={} expected={}",
        ctx,
        actual.len(),
        expected.len()
    );
    for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            approx_eq(*a, *e, tol),
            "{}：元素 {} 不匹配 actual={} expected={}",
            ctx,
            i,
            a,
            e
        );
    }
}

// ════════════════════════════════════════════════════════════════════════════
// 1. P0：layer_norm F16/BF16 修复验证
//
// 关键：F16/BF16 输入不应 panic，且数值与 F32 路径接近（容差 1e-2，F16 精度有限）。
// ════════════════════════════════════════════════════════════════════════════

#[test]
fn test_layer_norm_f16_no_panic_and_dtype() {
    // F16 输入：layer_norm 不应 panic，输出 dtype 应为 F16
    let x_f16 = make_tensor_f16(
        to_f16_vec(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
        vec![2, 3],
    );
    let gamma = make_tensor(vec![1.0, 1.0, 1.0], vec![3]);
    let beta = make_tensor(vec![0.0, 0.0, 0.0], vec![3]);

    let result = x_f16.layer_norm(&gamma, &beta, 1e-5).expect("layer_norm F16 不应 panic");
    assert_eq!(result.dtype(), BaseType::F16, "F16 输入应保持 F16 dtype");
    assert_eq!(result.shape(), vec![2, 3], "shape 应保持");
}

#[test]
fn test_layer_norm_bf16_no_panic_and_dtype() {
    let x_bf16 = make_tensor_bf16(
        to_bf16_vec(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
        vec![2, 3],
    );
    let gamma = make_tensor(vec![1.0, 1.0, 1.0], vec![3]);
    let beta = make_tensor(vec![0.0, 0.0, 0.0], vec![3]);

    let result = x_bf16.layer_norm(&gamma, &beta, 1e-5).expect("layer_norm BF16 不应 panic");
    assert_eq!(result.dtype(), BaseType::BF16, "BF16 输入应保持 BF16 dtype");
    assert_eq!(result.shape(), vec![2, 3]);
}

#[test]
fn test_layer_norm_f16_numerical_correctness() {
    // F16 输入的 layer_norm 应与 F32 路径数值接近（容差 1e-2，F16 精度有限）
    let data_f64 = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let x_f16 = make_tensor_f16(to_f16_vec(&data_f64), vec![2, 3]);
    let x_f32 = make_tensor_f32(data_f64.iter().map(|v| *v as f32).collect(), vec![2, 3]);
    let gamma = make_tensor(vec![1.0, 1.0, 1.0], vec![3]);
    let beta = make_tensor(vec![0.0, 0.0, 0.0], vec![3]);

    let r_f16 = x_f16.layer_norm(&gamma, &beta, 1e-5).unwrap();
    let r_f32 = x_f32.layer_norm(&gamma, &beta, 1e-5).unwrap();

    assert_vec_approx(
        &tensor_to_vec(&r_f16),
        &tensor_to_vec(&r_f32),
        1e-2,
        "layer_norm F16 vs F32",
    );
}

#[test]
fn test_layer_norm_bf16_numerical_correctness() {
    let data_f64 = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let x_bf16 = make_tensor_bf16(to_bf16_vec(&data_f64), vec![2, 3]);
    let x_f32 = make_tensor_f32(data_f64.iter().map(|v| *v as f32).collect(), vec![2, 3]);
    let gamma = make_tensor(vec![1.0, 1.0, 1.0], vec![3]);
    let beta = make_tensor(vec![0.0, 0.0, 0.0], vec![3]);

    let r_bf16 = x_bf16.layer_norm(&gamma, &beta, 1e-5).unwrap();
    let r_f32 = x_f32.layer_norm(&gamma, &beta, 1e-5).unwrap();

    assert_vec_approx(
        &tensor_to_vec(&r_bf16),
        &tensor_to_vec(&r_f32),
        1e-2,
        "layer_norm BF16 vs F32",
    );
}

#[test]
fn test_layer_norm_f16_with_gamma_beta() {
    // 非平凡 gamma/beta：验证 F16 路径与 F32 路径一致
    let data_f64 = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let x_f16 = make_tensor_f16(to_f16_vec(&data_f64), vec![2, 3]);
    let x_f32 = make_tensor_f32(data_f64.iter().map(|v| *v as f32).collect(), vec![2, 3]);
    let gamma = make_tensor(vec![2.0, 1.5, 0.5], vec![3]);
    let beta = make_tensor(vec![0.1, 0.2, 0.3], vec![3]);

    let r_f16 = x_f16.layer_norm(&gamma, &beta, 1e-5).unwrap();
    let r_f32 = x_f32.layer_norm(&gamma, &beta, 1e-5).unwrap();

    assert_vec_approx(
        &tensor_to_vec(&r_f16),
        &tensor_to_vec(&r_f32),
        1e-2,
        "layer_norm F16 (non-trivial gamma/beta) vs F32",
    );
}

// ════════════════════════════════════════════════════════════════════════════
// 2. P1：matmul F16/BF16 路径
//
// F16×F16 → F16；BF16×BF16 → BF16；数值与 F64 对比（容差 1e-1，F16/BF16 精度有限）
// ════════════════════════════════════════════════════════════════════════════

#[test]
fn test_matmul_f16_dtype_preserved() {
    let a_data = vec![1.0, 2.0, 3.0, 4.0]; // (1,2)? 实际是 (2,2)
    let b_data = vec![1.0, 0.0, 0.0, 1.0]; // (2,2) 单位阵
    let a = make_tensor_f16(to_f16_vec(&a_data), vec![2, 2]);
    let b = make_tensor_f16(to_f16_vec(&b_data), vec![2, 2]);

    let r = a.matmul(&b).expect("matmul F16 不应失败");
    assert_eq!(r.dtype(), BaseType::F16, "F16×F16 应保持 F16 dtype");
    assert_eq!(r.shape(), vec![2, 2]);
    // a @ I = a
    assert_vec_approx(&tensor_to_vec(&r), &a_data, 1e-2, "matmul F16 @ I");
}

#[test]
fn test_matmul_bf16_dtype_preserved() {
    let a_data = vec![1.0, 2.0, 3.0, 4.0];
    let b_data = vec![1.0, 0.0, 0.0, 1.0];
    let a = make_tensor_bf16(to_bf16_vec(&a_data), vec![2, 2]);
    let b = make_tensor_bf16(to_bf16_vec(&b_data), vec![2, 2]);

    let r = a.matmul(&b).expect("matmul BF16 不应失败");
    assert_eq!(r.dtype(), BaseType::BF16, "BF16×BF16 应保持 BF16 dtype");
    assert_eq!(r.shape(), vec![2, 2]);
    assert_vec_approx(&tensor_to_vec(&r), &a_data, 1e-2, "matmul BF16 @ I");
}

#[test]
fn test_matmul_f16_numerical_correctness() {
    // 比较 F16×F16 与 F64×F64 的结果（容差 1e-1）
    let a_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // (2,3)
    let b_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // (3,2)
    // c = a @ b, c[0,0] = 1*1+2*3+3*5 = 22, c[0,1] = 1*2+2*4+3*6 = 28
    // c[1,0] = 4*1+5*3+6*5 = 49, c[1,1] = 4*2+5*4+6*6 = 64
    let expected = vec![22.0, 28.0, 49.0, 64.0];

    let a_f16 = make_tensor_f16(to_f16_vec(&a_data), vec![2, 3]);
    let b_f16 = make_tensor_f16(to_f16_vec(&b_data), vec![3, 2]);
    let r_f16 = a_f16.matmul(&b_f16).unwrap();

    let a_f64 = make_tensor(a_data, vec![2, 3]);
    let b_f64 = make_tensor(b_data, vec![3, 2]);
    let r_f64 = a_f64.matmul(&b_f64).unwrap();

    assert_eq!(r_f16.dtype(), BaseType::F16);
    assert_vec_approx(&tensor_to_vec(&r_f16), &tensor_to_vec(&r_f64), 1e-1, "matmul F16 vs F64");
    assert_vec_approx(&tensor_to_vec(&r_f16), &expected, 1e-1, "matmul F16 expected");
}

#[test]
fn test_matmul_bf16_numerical_correctness() {
    let a_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let b_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let expected = vec![22.0, 28.0, 49.0, 64.0];

    let a_bf16 = make_tensor_bf16(to_bf16_vec(&a_data), vec![2, 3]);
    let b_bf16 = make_tensor_bf16(to_bf16_vec(&b_data), vec![3, 2]);
    let r_bf16 = a_bf16.matmul(&b_bf16).unwrap();

    let a_f64 = make_tensor(a_data, vec![2, 3]);
    let b_f64 = make_tensor(b_data, vec![3, 2]);
    let r_f64 = a_f64.matmul(&b_f64).unwrap();

    assert_eq!(r_bf16.dtype(), BaseType::BF16);
    assert_vec_approx(&tensor_to_vec(&r_bf16), &tensor_to_vec(&r_f64), 1e-1, "matmul BF16 vs F64");
    assert_vec_approx(&tensor_to_vec(&r_bf16), &expected, 1e-1, "matmul BF16 expected");
}

#[test]
fn test_matmul_f16_1d_2d_paths() {
    // 1D @ 2D 和 2D @ 1D 路径
    let a_data = vec![1.0, 2.0, 3.0]; // (3,)
    let b_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // (3,2)

    let a_f16 = make_tensor_f16(to_f16_vec(&a_data), vec![3]);
    let b_f16 = make_tensor_f16(to_f16_vec(&b_data), vec![3, 2]);
    let r = a_f16.matmul(&b_f16).expect("matmul 1D@2D F16 不应失败");
    assert_eq!(r.dtype(), BaseType::F16);
    assert_eq!(r.shape(), vec![2]);

    let a2_f16 = make_tensor_f16(to_f16_vec(&b_data), vec![3, 2]);
    let b2_f16 = make_tensor_f16(to_f16_vec(&[1.0, 2.0]), vec![2]);
    let r2 = a2_f16.matmul(&b2_f16).expect("matmul 2D@1D F16 不应失败");
    assert_eq!(r2.dtype(), BaseType::F16);
    assert_eq!(r2.shape(), vec![3]);
}

// ════════════════════════════════════════════════════════════════════════════
// 3. P1：bmm F16/BF16 路径
// ════════════════════════════════════════════════════════════════════════════

#[test]
fn test_bmm_f16_dtype_preserved() {
    // (2,3,4) @ (2,4,5) → (2,3,5)，全部元素 1 → 每个元素 = 4
    let a_data = vec![1.0; 24]; // 2*3*4
    let b_data = vec![1.0; 40]; // 2*4*5
    let a = make_tensor_f16(to_f16_vec(&a_data), vec![2, 3, 4]);
    let b = make_tensor_f16(to_f16_vec(&b_data), vec![2, 4, 5]);

    let r = a.bmm(&b).expect("bmm F16 不应失败");
    assert_eq!(r.dtype(), BaseType::F16, "F16 bmm 应保持 F16 dtype");
    assert_eq!(r.shape(), vec![2, 3, 5]);
    // 每个元素 = sum_k(1*1) for k in 0..4 = 4
    let r_vec = tensor_to_vec(&r);
    for (i, v) in r_vec.iter().enumerate() {
        assert!((*v - 4.0).abs() < 1e-2, "bmm F16[{}] = {}，期望 4.0", i, v);
    }
}

#[test]
fn test_bmm_bf16_dtype_preserved() {
    let a_data = vec![1.0; 24];
    let b_data = vec![1.0; 40];
    let a = make_tensor_bf16(to_bf16_vec(&a_data), vec![2, 3, 4]);
    let b = make_tensor_bf16(to_bf16_vec(&b_data), vec![2, 4, 5]);

    let r = a.bmm(&b).expect("bmm BF16 不应失败");
    assert_eq!(r.dtype(), BaseType::BF16, "BF16 bmm 应保持 BF16 dtype");
    assert_eq!(r.shape(), vec![2, 3, 5]);
    let r_vec = tensor_to_vec(&r);
    for (i, v) in r_vec.iter().enumerate() {
        assert!((*v - 4.0).abs() < 1e-2, "bmm BF16[{}] = {}，期望 4.0", i, v);
    }
}

#[test]
fn test_bmm_f16_vs_f64() {
    // 用非平凡数据，验证 F16 bmm 与 F64 bmm 数值接近
    let a_data: Vec<f64> = (0..24).map(|i| (i + 1) as f64 * 0.1).collect();
    let b_data: Vec<f64> = (0..40).map(|i| (i + 1) as f64 * 0.1).collect();
    let a_f16 = make_tensor_f16(to_f16_vec(&a_data), vec![2, 3, 4]);
    let b_f16 = make_tensor_f16(to_f16_vec(&b_data), vec![2, 4, 5]);
    let r_f16 = a_f16.bmm(&b_f16).unwrap();

    let a_f64 = make_tensor(a_data, vec![2, 3, 4]);
    let b_f64 = make_tensor(b_data, vec![2, 4, 5]);
    let r_f64 = a_f64.bmm(&b_f64).unwrap();

    assert_eq!(r_f16.dtype(), BaseType::F16);
    // BF16 容差较大；F16 精度约 3 位十进制，容差 1.0 应该足够
    assert_vec_approx(&tensor_to_vec(&r_f16), &tensor_to_vec(&r_f64), 1.0, "bmm F16 vs F64");
}

// ════════════════════════════════════════════════════════════════════════════
// 4. P1：im2col F16/BF16 路径（影响 Conv2D）
// ════════════════════════════════════════════════════════════════════════════

#[test]
fn test_im2col_f16_dtype_preserved() {
    // 4D NCHW 输入 (1,1,3,3)，kernel 2x2, stride 1, pad 0
    // h_out = (3-2)/1+1 = 2, w_out = 2
    let data: Vec<f64> = (1..=9).map(|i| i as f64).collect();
    let x_f16 = make_tensor_f16(to_f16_vec(&data), vec![1, 1, 3, 3]);
    let (cols, h_out, w_out) = x_f16.im2col(2, 2, 1, 0).expect("im2col F16 失败");
    assert_eq!(cols.dtype(), BaseType::F16, "im2col F16 应保持 F16 dtype");
    assert_eq!(cols.shape(), vec![4, 4]); // n*h_out*w_out = 1*2*2 = 4, c*k_h*k_w = 1*2*2 = 4
    assert_eq!(h_out, 2);
    assert_eq!(w_out, 2);
}

#[test]
fn test_im2col_bf16_dtype_preserved() {
    let data: Vec<f64> = (1..=9).map(|i| i as f64).collect();
    let x_bf16 = make_tensor_bf16(to_bf16_vec(&data), vec![1, 1, 3, 3]);
    let (cols, h_out, w_out) = x_bf16.im2col(2, 2, 1, 0).expect("im2col BF16 失败");
    assert_eq!(cols.dtype(), BaseType::BF16);
    assert_eq!(cols.shape(), vec![4, 4]);
    assert_eq!(h_out, 2);
    assert_eq!(w_out, 2);
}

#[test]
fn test_im2col_f16_values_match_f64() {
    // 验证 F16 im2col 数值与 F64 im2col 一致（值都是 1..9 整数，F16 可精确表示）
    let data: Vec<f64> = (1..=9).map(|i| i as f64).collect();
    let x_f16 = make_tensor_f16(to_f16_vec(&data), vec![1, 1, 3, 3]);
    let x_f64 = make_tensor(data, vec![1, 1, 3, 3]);

    let (cols_f16, _, _) = x_f16.im2col(2, 2, 1, 0).unwrap();
    let (cols_f64, _, _) = x_f64.im2col(2, 2, 1, 0).unwrap();

    assert_vec_approx(
        &tensor_to_vec(&cols_f16),
        &tensor_to_vec(&cols_f64),
        1e-2,
        "im2col F16 vs F64",
    );
}

#[test]
fn test_conv2d_f16_via_im2col_and_matmul() {
    // 模拟 Conv2D 前向：input(F16) → im2col → matmul(weight^T) → reshape
    // 验证 dtype 一路保持 F16
    let input_data: Vec<f64> = (1..=9).map(|i| i as f64).collect();
    let x_f16 = make_tensor_f16(to_f16_vec(&input_data), vec![1, 1, 3, 3]);
    // weight: (c_out=1, c_in=1, kH=2, kW=2) → reshape (1, 4)
    let w_data: Vec<f64> = vec![1.0, 0.0, 0.0, 1.0]; // 对角取值
    let w_f16 = make_tensor_f16(to_f16_vec(&w_data), vec![1, 1, 2, 2]);
    // w_flat: (c_out, c_in*k_h*k_w) = (1, 4)
    let w_flat = w_f16.reshape(&[1, 4]).expect("reshape 失败");
    let w_flat_t = w_flat.transpose().expect("transpose 失败"); // (4, 1)

    let (cols, _h_out, _w_out) = x_f16.im2col(2, 2, 1, 0).expect("im2col 失败");
    assert_eq!(cols.dtype(), BaseType::F16);

    // cols.matmul(w_flat_t) → (4, 1)
    let out_2d = cols.matmul(&w_flat_t).expect("matmul 失败");
    assert_eq!(out_2d.dtype(), BaseType::F16, "Conv2D F16 端到端 dtype 应保持");
    assert_eq!(out_2d.shape(), vec![4, 1]);
}

// ════════════════════════════════════════════════════════════════════════════
// 5. P1：max_pool2d / avg_pool2d F16/BF16 路径
// ════════════════════════════════════════════════════════════════════════════

#[test]
fn test_max_pool2d_f16_dtype_preserved() {
    let data: Vec<f64> = (1..=9).map(|i| i as f64).collect();
    let x_f16 = make_tensor_f16(to_f16_vec(&data), vec![1, 1, 3, 3]);
    let out = x_f16
        .max_pool2d(2, 2, 1, 1, 0, 0)
        .expect("max_pool2d F16 失败");
    assert_eq!(out.dtype(), BaseType::F16, "max_pool2d F16 应保持 F16 dtype");
    assert_eq!(out.shape(), vec![1, 1, 2, 2]);
}

#[test]
fn test_max_pool2d_bf16_dtype_preserved() {
    let data: Vec<f64> = (1..=9).map(|i| i as f64).collect();
    let x_bf16 = make_tensor_bf16(to_bf16_vec(&data), vec![1, 1, 3, 3]);
    let out = x_bf16
        .max_pool2d(2, 2, 1, 1, 0, 0)
        .expect("max_pool2d BF16 失败");
    assert_eq!(out.dtype(), BaseType::BF16);
    assert_eq!(out.shape(), vec![1, 1, 2, 2]);
}

#[test]
fn test_max_pool2d_f16_values() {
    // 3x3 输入，2x2 pool stride 1 → 2x2 输出
    // [[1,2,3],[4,5,6],[7,8,9]] → max(2x2 windows) → [[5,6],[8,9]]
    let data: Vec<f64> = (1..=9).map(|i| i as f64).collect();
    let x_f16 = make_tensor_f16(to_f16_vec(&data), vec![1, 1, 3, 3]);
    let x_f64 = make_tensor(data, vec![1, 1, 3, 3]);

    let r_f16 = x_f16.max_pool2d(2, 2, 1, 1, 0, 0).unwrap();
    let r_f64 = x_f64.max_pool2d(2, 2, 1, 1, 0, 0).unwrap();

    assert_eq!(r_f16.dtype(), BaseType::F16);
    assert_vec_approx(&tensor_to_vec(&r_f16), &tensor_to_vec(&r_f64), 1e-2, "max_pool2d F16 vs F64");
    // 期望值：[5,6,8,9]
    assert_vec_approx(&tensor_to_vec(&r_f16), &[5.0, 6.0, 8.0, 9.0], 1e-2, "max_pool2d F16 expected");
}

#[test]
fn test_avg_pool2d_f16_dtype_and_values() {
    let data: Vec<f64> = (1..=9).map(|i| i as f64).collect();
    let x_f16 = make_tensor_f16(to_f16_vec(&data), vec![1, 1, 3, 3]);
    let x_f64 = make_tensor(data, vec![1, 1, 3, 3]);

    let r_f16 = x_f16.avg_pool2d(2, 2, 1, 1, 0, 0).unwrap();
    let r_f64 = x_f64.avg_pool2d(2, 2, 1, 1, 0, 0).unwrap();

    assert_eq!(r_f16.dtype(), BaseType::F16);
    assert_vec_approx(&tensor_to_vec(&r_f16), &tensor_to_vec(&r_f64), 1e-2, "avg_pool2d F16 vs F64");
    // 期望值：(1+2+4+5)/4=3, (2+3+5+6)/4=4, (4+5+7+8)/4=6, (5+6+8+9)/4=7
    assert_vec_approx(&tensor_to_vec(&r_f16), &[3.0, 4.0, 6.0, 7.0], 1e-2, "avg_pool2d F16 expected");
}

#[test]
fn test_avg_pool2d_bf16_dtype_and_values() {
    let data: Vec<f64> = (1..=9).map(|i| i as f64).collect();
    let x_bf16 = make_tensor_bf16(to_bf16_vec(&data), vec![1, 1, 3, 3]);
    let r_bf16 = x_bf16.avg_pool2d(2, 2, 1, 1, 0, 0).unwrap();
    assert_eq!(r_bf16.dtype(), BaseType::BF16);
    assert_vec_approx(&tensor_to_vec(&r_bf16), &[3.0, 4.0, 6.0, 7.0], 1e-2, "avg_pool2d BF16 expected");
}

// ════════════════════════════════════════════════════════════════════════════
// 6. P2：sum_axis / cat / select F16/BF16 分支
// ════════════════════════════════════════════════════════════════════════════

#[test]
fn test_sum_axis_f16_dtype_preserved() {
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let x_f16 = make_tensor_f16(to_f16_vec(&data), vec![2, 3]);
    let r = x_f16.sum_axis(1).expect("sum_axis F16 失败");
    assert_eq!(r.dtype(), BaseType::F16, "sum_axis F16 应保持 F16 dtype");
    assert_eq!(r.shape(), vec![2]);
    // 每行 sum: 1+2+3=6, 4+5+6=15
    assert_vec_approx(&tensor_to_vec(&r), &[6.0, 15.0], 1e-2, "sum_axis F16 expected");
}

#[test]
fn test_sum_axis_bf16_dtype_preserved() {
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let x_bf16 = make_tensor_bf16(to_bf16_vec(&data), vec![2, 3]);
    let r = x_bf16.sum_axis(0).expect("sum_axis BF16 失败");
    assert_eq!(r.dtype(), BaseType::BF16);
    assert_eq!(r.shape(), vec![3]);
    // 每列 sum: 1+4=5, 2+5=7, 3+6=9
    assert_vec_approx(&tensor_to_vec(&r), &[5.0, 7.0, 9.0], 1e-2, "sum_axis BF16 expected");
}

#[test]
fn test_cat_f16_same_dtype_preserved() {
    // F16 + F16 → F16
    let a = make_tensor_f16(to_f16_vec(&[1.0, 2.0, 3.0, 4.0]), vec![2, 2]);
    let b = make_tensor_f16(to_f16_vec(&[5.0, 6.0, 7.0, 8.0]), vec![2, 2]);

    let r0 = a.cat(&b, 0).expect("cat dim=0 失败");
    assert_eq!(r0.dtype(), BaseType::F16, "cat F16+F16 dim=0 应保持 F16");
    assert_eq!(r0.shape(), vec![4, 2]);

    let r1 = a.cat(&b, 1).expect("cat dim=1 失败");
    assert_eq!(r1.dtype(), BaseType::F16, "cat F16+F16 dim=1 应保持 F16");
    assert_eq!(r1.shape(), vec![2, 4]);
    assert_vec_approx(
        &tensor_to_vec(&r1),
        &[1.0, 2.0, 5.0, 6.0, 3.0, 4.0, 7.0, 8.0],
        1e-2,
        "cat F16 dim=1 values",
    );
}

#[test]
fn test_cat_bf16_same_dtype_preserved() {
    let a = make_tensor_bf16(to_bf16_vec(&[1.0, 2.0, 3.0, 4.0]), vec![2, 2]);
    let b = make_tensor_bf16(to_bf16_vec(&[5.0, 6.0, 7.0, 8.0]), vec![2, 2]);

    let r = a.cat(&b, 0).expect("cat BF16 dim=0 失败");
    assert_eq!(r.dtype(), BaseType::BF16);
    assert_eq!(r.shape(), vec![4, 2]);
}

#[test]
fn test_cat_mixed_dtype_promotes_to_f64() {
    // F16 + F32 → F64（按审计要求：混合提升到 F64）
    let a = make_tensor_f16(to_f16_vec(&[1.0, 2.0, 3.0, 4.0]), vec![2, 2]);
    let b = make_tensor_f32(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]);

    let r = a.cat(&b, 0).expect("cat 混合 dtype 失败");
    assert_eq!(r.dtype(), BaseType::F64, "F16 + F32 混合应提升到 F64");
}

#[test]
fn test_select_f16_then_else_same_dtype() {
    // cond truthy 选 then，否则选 else；then/else 同为 F16 → F16
    let cond = make_tensor(vec![1.0, 0.0, 1.0, 0.0], vec![2, 2]);
    let then_data = vec![10.0, 20.0, 30.0, 40.0];
    let else_data = vec![100.0, 200.0, 300.0, 400.0];
    let then_t = make_tensor_f16(to_f16_vec(&then_data), vec![2, 2]);
    let else_t = make_tensor_f16(to_f16_vec(&else_data), vec![2, 2]);

    let r = Tensor::select(&cond, &then_t, &else_t).expect("select F16 失败");
    assert_eq!(r.dtype(), BaseType::F16, "select F16+F16 应保持 F16");
    // 期望：cond>0.5 时取 then，否则取 else
    let expected = vec![10.0, 200.0, 30.0, 400.0];
    assert_vec_approx(&tensor_to_vec(&r), &expected, 1e-2, "select F16 values");
}

#[test]
fn test_select_bf16_then_else_same_dtype() {
    let cond = make_tensor(vec![0.0, 1.0, 0.0, 1.0], vec![2, 2]);
    let then_data = vec![10.0, 20.0, 30.0, 40.0];
    let else_data = vec![100.0, 200.0, 300.0, 400.0];
    let then_t = make_tensor_bf16(to_bf16_vec(&then_data), vec![2, 2]);
    let else_t = make_tensor_bf16(to_bf16_vec(&else_data), vec![2, 2]);

    let r = Tensor::select(&cond, &then_t, &else_t).expect("select BF16 失败");
    assert_eq!(r.dtype(), BaseType::BF16);
    let expected = vec![100.0, 20.0, 300.0, 40.0];
    assert_vec_approx(&tensor_to_vec(&r), &expected, 1e-2, "select BF16 values");
}

#[test]
fn test_select_mixed_dtype_promotes_to_f64() {
    // then=F16, else=F32 → F64（按审计要求：混合提升到 F64）
    let cond = make_tensor(vec![1.0, 0.0], vec![2]);
    let then_t = make_tensor_f16(to_f16_vec(&[10.0, 20.0]), vec![2]);
    let else_t = make_tensor_f32(vec![100.0, 200.0], vec![2]);

    let r = Tensor::select(&cond, &then_t, &else_t).expect("select 混合 dtype 失败");
    assert_eq!(r.dtype(), BaseType::F64, "F16 + F32 混合 select 应提升到 F64");
}

// ════════════════════════════════════════════════════════════════════════════
// 7. autodiff 端到端：F16 张量的 matmul + relu + sum loss 的 backward
//
// 按 acc_grad 策略（Phase 2 AMP）：F16/BF16 param 的 grad buffer 存储为 F32 中间表示。
// 反向通过 dispatch_float! 走 f32 路径。
// ════════════════════════════════════════════════════════════════════════════

#[test]
fn test_autodiff_f16_matmul_relu_sum_backward() {
    // x: F16 (2,2), w: F16 (2,2)
    // loss = sum(relu(x @ w))
    // 反向后 grad(x) 应存在（dtype 为 F32，按 acc_grad AMP 策略）
    let x_data = vec![1.0, 2.0, 3.0, 4.0];
    let w_data = vec![1.0, 0.0, 0.0, 1.0]; // 单位阵

    let x_tensor = Tensor::from_vec_f16(to_f16_vec(&x_data), vec![2, 2]);
    let w_tensor = Tensor::from_vec_f16(to_f16_vec(&w_data), vec![2, 2]);
    let x = Rc::new(RefCell::new(x_tensor));
    let w = Rc::new(RefCell::new(w_tensor));

    let mut tape = Tape::new();
    let x_id = tape.input(x.clone());
    let w_id = tape.input(w.clone());

    // 前向：mm = x @ w
    let mm_result = {
        let x_ref = x.borrow();
        let w_ref = w.borrow();
        x_ref.matmul(&w_ref).expect("matmul 失败")
    };
    assert_eq!(mm_result.dtype(), BaseType::F16, "前向 matmul 应保持 F16");
    let mm = Rc::new(RefCell::new(mm_result));
    let mm_id = tape.binary(TapeOp::MatMul, x_id, w_id, x.clone(), w.clone(), mm.clone());

    // relu(mm)
    let relu_result = {
        let mm_ref = mm.borrow();
        mm_ref.relu()
    };
    assert_eq!(relu_result.dtype(), BaseType::F16);
    let relu = Rc::new(RefCell::new(relu_result));
    let relu_id = tape.unary(TapeOp::ReLU, mm_id, mm.clone(), relu.clone());

    // sum(relu) → loss
    let sum_val = {
        let relu_ref = relu.borrow();
        relu_ref.sum()
    };
    let loss_tensor = Tensor::from_vec_f16(vec![f16::from_f64(sum_val)], vec![1]);
    let loss = Rc::new(RefCell::new(loss_tensor));
    let loss_id = tape.unary(TapeOp::Sum, relu_id, relu.clone(), loss.clone());

    // backward
    tape.backward(loss_id).expect("backward 不应失败");

    // 验证 x.grad 存在，且按 acc_grad 策略 dtype 为 F32
    let x_guard = x.borrow();
    let x_grad = x_guard.grad.as_ref().expect("x.grad 应存在");
    match x_grad {
        tenth::runtime::tensor::TensorData::F32(_) => {
            // 期望路径：F16 param 的 grad 存储为 F32 中间表示（AMP 策略）
        }
        other => panic!(
            "F16 param 的 grad 应为 F32（AMP 策略），实际 {:?}",
            other.dtype()
        ),
    }
    // 验证 x.grad shape 正确
    let x_grad_shape = x_grad.shape();
    assert_eq!(x_grad_shape, &[2, 2], "x.grad shape 应为 (2,2)");
}

#[test]
fn test_autodiff_bf16_matmul_relu_sum_backward() {
    let x_data = vec![1.0, 2.0, 3.0, 4.0];
    let w_data = vec![1.0, 0.0, 0.0, 1.0];

    let x_tensor = Tensor::from_vec_bf16(to_bf16_vec(&x_data), vec![2, 2]);
    let w_tensor = Tensor::from_vec_bf16(to_bf16_vec(&w_data), vec![2, 2]);
    let x = Rc::new(RefCell::new(x_tensor));
    let w = Rc::new(RefCell::new(w_tensor));

    let mut tape = Tape::new();
    let x_id = tape.input(x.clone());
    let w_id = tape.input(w.clone());

    let mm_result = {
        let x_ref = x.borrow();
        let w_ref = w.borrow();
        x_ref.matmul(&w_ref).expect("matmul 失败")
    };
    assert_eq!(mm_result.dtype(), BaseType::BF16, "前向 matmul 应保持 BF16");
    let mm = Rc::new(RefCell::new(mm_result));
    let mm_id = tape.binary(TapeOp::MatMul, x_id, w_id, x.clone(), w.clone(), mm.clone());

    let relu_result = {
        let mm_ref = mm.borrow();
        mm_ref.relu()
    };
    let relu = Rc::new(RefCell::new(relu_result));
    let relu_id = tape.unary(TapeOp::ReLU, mm_id, mm.clone(), relu.clone());

    let sum_val = {
        let relu_ref = relu.borrow();
        relu_ref.sum()
    };
    let loss_tensor = Tensor::from_vec_bf16(vec![bf16::from_f64(sum_val)], vec![1]);
    let loss = Rc::new(RefCell::new(loss_tensor));
    let loss_id = tape.unary(TapeOp::Sum, relu_id, relu.clone(), loss.clone());

    tape.backward(loss_id).expect("backward 不应失败");

    let x_guard = x.borrow();
    let x_grad = x_guard.grad.as_ref().expect("x.grad 应存在");
    match x_grad {
        tenth::runtime::tensor::TensorData::F32(_) => {
            // 期望路径
        }
        other => panic!(
            "BF16 param 的 grad 应为 F32（AMP 策略），实际 {:?}",
            other.dtype()
        ),
    }
    assert_eq!(x_grad.shape(), &[2, 2]);
}

#[test]
fn test_autodiff_f16_gradient_numerical_correctness() {
    // 验证 F16 路径梯度数值正确性：
    // y = x^2 (elementwise), backward(sum(y)) → grad(x) = 2*x
    // 用单位阵 w 使 matmul 不改变结果，简化验证
    let x_data = vec![1.0, 2.0, 3.0, 4.0];
    let x_tensor = Tensor::from_vec_f16(to_f16_vec(&x_data), vec![2, 2]);
    let x = Rc::new(RefCell::new(x_tensor));

    let mut tape = Tape::new();
    let _x_id = tape.input(x.clone());

    // y = x * x (elementwise mul)
    let y_result = {
        let x_ref = x.borrow();
        x_ref.mul_tensor(&x_ref).expect("mul_tensor 失败")
    };
    assert_eq!(y_result.dtype(), BaseType::F16);
    let y = Rc::new(RefCell::new(y_result));
    let y_id = tape.binary_direct(TapeOp::Mul, x.clone(), x.clone(), y.clone());

    // loss = sum(y)
    let sum_val = {
        let y_ref = y.borrow();
        y_ref.sum()
    };
    let loss_tensor = Tensor::from_vec_f16(vec![f16::from_f64(sum_val)], vec![1]);
    let loss = Rc::new(RefCell::new(loss_tensor));
    let loss_id = tape.unary(TapeOp::Sum, y_id, y.clone(), loss.clone());

    tape.backward(loss_id).expect("backward 失败");

    // grad(x) 应为 2*x = [2, 4, 6, 8]
    let x_guard = x.borrow();
    let x_grad = x_guard.grad.as_ref().expect("x.grad 应存在");
    let x_grad_view = x_grad.as_f64_view();
    let expected = vec![2.0, 4.0, 6.0, 8.0];
    for (i, v) in x_grad_view.iter().enumerate() {
        assert!(
            (v - expected[i]).abs() < 1e-3,
            "grad[{}] = {}，期望 {}",
            i,
            v,
            expected[i]
        );
    }
}
