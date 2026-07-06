// f32 vs f64 系统性 Parity 测试 — f32/f64 双精度对等路线图 阶段 2
//
// 目标：建立 f32 与 f64 路径在全后端下的结果一致性基线，为后续阶段（3-7）
// 提供回归守护。
//
// 测试范围：
//   1. 算术运算（add/sub/mul/div）— f32 路径 vs f64 路径，元素级误差 < 1e-6
//   2. 构造器（zeros/ones）— shape 与元素值一致
//   3. 矩阵乘法（matmul）— 相同输入下结果误差 < 1e-6
//   4. 激活函数（relu/sigmoid）— 元素级误差 < 1e-6
//   5. 规约（sum/mean）— 误差 < 1e-5（reduction 累积误差稍大）
//   6. 混合 dtype 提升 — f32 + f64 → f64 验证
//
// 注：本测试直接调用 Rust Tensor API，绕过解释器/VM，专注张量层 parity。

use tenth::hir::types::BaseType;
use tenth::runtime::tensor::Tensor;

/// 比较两个 Tensor 元素级绝对误差，返回最大误差。
/// f32 与 f64 张量都先归一化为 f64 视图比较。
fn max_abs_diff(a: &Tensor, b: &Tensor) -> f64 {
    let av = a.data.as_f64_view();
    let bv = b.data.as_f64_view();
    assert_eq!(av.shape(), bv.shape(),
        "shape mismatch: {:?} vs {:?}", av.shape(), bv.shape());
    let mut max_diff = 0.0f64;
    for (x, y) in av.iter().zip(bv.iter()) {
        let d = (x - y).abs();
        if d > max_diff {
            max_diff = d;
        }
    }
    max_diff
}

/// 比较两个 Tensor 元素级相对误差，返回最大相对误差。
/// 对零元素使用绝对误差兜底。
fn max_rel_diff(a: &Tensor, b: &Tensor) -> f64 {
    let av = a.data.as_f64_view();
    let bv = b.data.as_f64_view();
    assert_eq!(av.shape(), bv.shape());
    let mut max_rel = 0.0f64;
    for (x, y) in av.iter().zip(bv.iter()) {
        let denom = x.abs().max(y.abs());
        let rel = if denom > 1e-12 {
            (x - y).abs() / denom
        } else {
            (x - y).abs()
        };
        if rel > max_rel {
            max_rel = rel;
        }
    }
    max_rel
}

// ══════════════════════════════════════════════════════════════════════
// 1. 算术运算 parity
// ══════════════════════════════════════════════════════════════════════

#[test]
fn f32_f64_arithmetic_add_parity() {
    // f32 加法 vs f64 加法，相同输入下结果误差 < 1e-6
    let data = vec![1.5, 2.7, 3.14, -0.5, 1e-3, 99.9];
    let a32 = Tensor::from_vec_f32(data.iter().map(|&v| v as f32).collect(), vec![6]);
    let b32 = Tensor::from_vec_f32(vec![0.1f32; 6], vec![6]);
    let a64 = Tensor::from_vec(data, vec![6]);
    let b64 = Tensor::from_vec(vec![0.1; 6], vec![6]);

    let c32 = a32.add_tensor(&b32).unwrap();
    let c64 = a64.add_tensor(&b64).unwrap();

    assert!(c32.is_f32(), "f32 + f32 应保持 F32");
    assert!(c64.is_f64(), "f64 + f64 应保持 F64");
    let diff = max_abs_diff(&c32, &c64);
    assert!(diff < 1e-6, "f32 vs f64 add 误差 {} 超过 1e-6", diff);
}

#[test]
fn f32_f64_arithmetic_sub_parity() {
    // 注：99.9 等数值在 f32 下无法精确表示，故使用相对误差而非绝对误差。
    // f32 路径与 f64 路径在同一数学运算下应一致至 f32 表示精度。
    let data = vec![10.5, 20.7, 3.14, -0.5, 1e-3, 99.9];
    let a32 = Tensor::from_vec_f32(data.iter().map(|&v| v as f32).collect(), vec![6]);
    let b32 = Tensor::from_vec_f32(vec![0.5f32; 6], vec![6]);
    let a64 = Tensor::from_vec(data, vec![6]);
    let b64 = Tensor::from_vec(vec![0.5; 6], vec![6]);

    let c32 = a32.sub_tensor(&b32).unwrap();
    let c64 = a64.sub_tensor(&b64).unwrap();

    let rel = max_rel_diff(&c32, &c64);
    assert!(rel < 1e-5, "f32 vs f64 sub 相对误差 {} 超过 1e-5", rel);
    // 同时保证绝对误差在 f32 表示精度内（< 1e-5）
    let abs = max_abs_diff(&c32, &c64);
    assert!(abs < 1e-5, "f32 vs f64 sub 绝对误差 {} 超过 1e-5", abs);
}

#[test]
fn f32_f64_arithmetic_mul_parity() {
    // 99.9 * 2 = 199.8，f32 表示误差 ~3e-6，使用相对误差
    let data = vec![1.5, 2.7, 3.14, -0.5, 1e-3, 99.9];
    let a32 = Tensor::from_vec_f32(data.iter().map(|&v| v as f32).collect(), vec![6]);
    let b32 = Tensor::from_vec_f32(vec![2.0f32; 6], vec![6]);
    let a64 = Tensor::from_vec(data, vec![6]);
    let b64 = Tensor::from_vec(vec![2.0; 6], vec![6]);

    let c32 = a32.mul_tensor(&b32).unwrap();
    let c64 = a64.mul_tensor(&b64).unwrap();

    let rel = max_rel_diff(&c32, &c64);
    assert!(rel < 1e-5, "f32 vs f64 mul 相对误差 {} 超过 1e-5", rel);
    let abs = max_abs_diff(&c32, &c64);
    assert!(abs < 1e-5, "f32 vs f64 mul 绝对误差 {} 超过 1e-5", abs);
}

#[test]
fn f32_f64_arithmetic_div_parity() {
    let data = vec![1.5, 2.7, 3.14, -0.5, 1e-3, 99.9];
    let a32 = Tensor::from_vec_f32(data.iter().map(|&v| v as f32).collect(), vec![6]);
    let b32 = Tensor::from_vec_f32(vec![3.0f32; 6], vec![6]);
    let a64 = Tensor::from_vec(data, vec![6]);
    let b64 = Tensor::from_vec(vec![3.0; 6], vec![6]);

    let c32 = a32.div_tensor(&b32).unwrap();
    let c64 = a64.div_tensor(&b64).unwrap();

    let diff = max_abs_diff(&c32, &c64);
    assert!(diff < 1e-6, "f32 vs f64 div 误差 {} 超过 1e-6", diff);
}

// ══════════════════════════════════════════════════════════════════════
// 2. 构造器 parity
// ══════════════════════════════════════════════════════════════════════

#[test]
fn f32_f64_zeros_parity() {
    // zeros_f32 vs zeros：元素值与 shape 一致
    let shape = [2, 3, 4];
    let z32 = Tensor::zeros_f32(&shape);
    let z64 = Tensor::zeros(&shape);

    assert_eq!(z32.shape(), z64.shape(), "shape 不一致");
    assert!(z32.is_f32(), "zeros_f32 应为 F32");
    assert!(z64.is_f64(), "zeros 应为 F64");
    let diff = max_abs_diff(&z32, &z64);
    assert!(diff < 1e-12, "zeros 误差 {} 应严格为 0", diff);
    assert_eq!(z32.get(&[1, 2, 3]), Some(0.0));
    assert_eq!(z64.get(&[1, 2, 3]), Some(0.0));
}

#[test]
fn f32_f64_ones_parity() {
    // ones_f32 vs ones：元素值与 shape 一致
    let shape = [3, 5];
    let o32 = Tensor::ones_f32(&shape);
    let o64 = Tensor::ones(&shape);

    assert_eq!(o32.shape(), o64.shape());
    assert!(o32.is_f32());
    assert!(o64.is_f64());
    let diff = max_abs_diff(&o32, &o64);
    assert!(diff < 1e-12, "ones 误差 {} 应严格为 0", diff);
    assert_eq!(o32.get(&[2, 4]), Some(1.0));
    assert_eq!(o64.get(&[2, 4]), Some(1.0));
}

// ══════════════════════════════════════════════════════════════════════
// 3. matmul parity
// ══════════════════════════════════════════════════════════════════════

#[test]
fn f32_f64_matmul_parity() {
    // matmul_f32 vs matmul，相同输入下结果误差 < 1e-6
    // 用小数值避免 f32 累积误差过大
    let a_data = vec![1.0, 2.0, 3.0, 4.0, 0.5, -1.0, 2.0, 1.5];
    let b_data = vec![0.5, 1.0, 1.5, 2.0, -0.5, 0.5, 1.0, -1.0];

    let a32 = Tensor::from_vec_f32(a_data.iter().map(|&v| v as f32).collect(), vec![2, 4]);
    let b32 = Tensor::from_vec_f32(b_data.iter().map(|&v| v as f32).collect(), vec![4, 2]);
    let a64 = Tensor::from_vec(a_data, vec![2, 4]);
    let b64 = Tensor::from_vec(b_data, vec![4, 2]);

    let c32 = a32.matmul(&b32).unwrap();
    let c64 = a64.matmul(&b64).unwrap();

    assert!(c32.is_f32(), "f32@f32 应保持 F32");
    assert!(c64.is_f64(), "f64@f64 应保持 F64");
    assert_eq!(c32.shape(), c64.shape());
    assert_eq!(c32.shape(), vec![2, 2]);

    let diff = max_abs_diff(&c32, &c64);
    assert!(diff < 1e-6, "f32 vs f64 matmul 误差 {} 超过 1e-6", diff);
}

// ══════════════════════════════════════════════════════════════════════
// 4. 激活函数 parity
// ══════════════════════════════════════════════════════════════════════

#[test]
fn f32_f64_relu_parity() {
    // f32 relu vs f64 relu，相同输入下结果误差 < 1e-6
    let data = vec![-2.5, -0.1, 0.0, 0.5, 1.5, 100.0, -1e-3, 1e-3];
    let t32 = Tensor::from_vec_f32(data.iter().map(|&v| v as f32).collect(), vec![8]);
    let t64 = Tensor::from_vec(data, vec![8]);

    let r32 = t32.relu();
    let r64 = t64.relu();

    assert!(r32.is_f32());
    assert!(r64.is_f64());
    let diff = max_abs_diff(&r32, &r64);
    assert!(diff < 1e-6, "f32 vs f64 relu 误差 {} 超过 1e-6", diff);
}

#[test]
fn f32_f64_sigmoid_parity() {
    // f32 sigmoid vs f64 sigmoid
    // sigmoid 用 exp，f32 exp 精度稍差，但小数值范围内应 < 1e-6
    let data = vec![-2.0, -0.5, 0.0, 0.5, 1.0, 2.0, -1.5, 1.5];
    let t32 = Tensor::from_vec_f32(data.iter().map(|&v| v as f32).collect(), vec![8]);
    let t64 = Tensor::from_vec(data, vec![8]);

    let s32 = t32.sigmoid();
    let s64 = t64.sigmoid();

    assert!(s32.is_f32());
    assert!(s64.is_f64());
    let diff = max_abs_diff(&s32, &s64);
    assert!(diff < 1e-6, "f32 vs f64 sigmoid 误差 {} 超过 1e-6", diff);
}

// ══════════════════════════════════════════════════════════════════════
// 5. 规约 parity
// ══════════════════════════════════════════════════════════════════════

#[test]
fn f32_f64_sum_parity() {
    // f32 sum vs f64 sum，误差 < 1e-5（reduction 累积误差）
    let data: Vec<f64> = (1..=10).map(|i| i as f64 * 0.1).collect();
    let t32 = Tensor::from_vec_f32(data.iter().map(|&v| v as f32).collect(), vec![10]);
    let t64 = Tensor::from_vec(data, vec![10]);

    let s32 = t32.sum();
    let s64 = t64.sum();

    let diff = (s32 - s64).abs();
    assert!(diff < 1e-5, "f32 vs f64 sum 误差 {} 超过 1e-5", diff);
}

#[test]
fn f32_f64_mean_parity() {
    // f32 mean vs f64 mean，误差 < 1e-5
    let data: Vec<f64> = (1..=8).map(|i| i as f64 * 0.3).collect();
    let t32 = Tensor::from_vec_f32(data.iter().map(|&v| v as f32).collect(), vec![8]);
    let t64 = Tensor::from_vec(data, vec![8]);

    let m32 = t32.mean();
    let m64 = t64.mean();

    let diff = (m32 - m64).abs();
    assert!(diff < 1e-5, "f32 vs f64 mean 误差 {} 超过 1e-5", diff);
}

// ══════════════════════════════════════════════════════════════════════
// 6. 混合 dtype 提升 parity
// ══════════════════════════════════════════════════════════════════════

#[test]
fn f32_f64_mixed_dtype_promotion() {
    // f32 + f64 → f64 隐式提升验证
    // 注：3.14 在 f32 下不能精确表示，故用 0.5/1.5 等可精确表示的数值。
    let data = vec![1.5, 2.5, 3.5, -0.5];
    let a32 = Tensor::from_vec_f32(data.iter().map(|&v| v as f32).collect(), vec![4]);
    let b64 = Tensor::from_vec(vec![0.5; 4], vec![4]);

    let c = a32.add_tensor(&b64).unwrap();
    assert!(c.is_f64(), "f32 + f64 应提升为 F64");

    // 与纯 f64 路径对比，应严格一致（f32 → f64 cast 对可精确表示的数值无损）
    let a64 = Tensor::from_vec(data, vec![4]);
    let c_pure_f64 = a64.add_tensor(&b64).unwrap();
    let diff = max_abs_diff(&c, &c_pure_f64);
    assert!(diff < 1e-12, "f32+f64 vs 纯 f64 路径误差 {} 应近似 0", diff);

    // 验证元素值：c[i] = a[i] + b[i]
    // a = [1.5, 2.5, 3.5, -0.5], b = [0.5; 4] → c = [2.0, 3.0, 4.0, 0.0]
    assert!((c.get(&[0]).unwrap() - 2.0).abs() < 1e-12);
    assert!((c.get(&[1]).unwrap() - 3.0).abs() < 1e-12);
    assert!((c.get(&[2]).unwrap() - 4.0).abs() < 1e-12);
    assert!((c.get(&[3]).unwrap() - 0.0).abs() < 1e-12, "c[3] should be 0.0, got {}", c.get(&[3]).unwrap());
}

// ══════════════════════════════════════════════════════════════════════
// 附加：dtype 标记 parity（确保 f32 路径不退化为 f64）
// ══════════════════════════════════════════════════════════════════════

#[test]
fn f32_f64_dtype_preservation_parity() {
    // 关键守护：f32 路径的算术运算必须保持 F32 dtype（不能默默退化为 f64）
    let a = Tensor::from_vec_f32(vec![1.0f32, 2.0, 3.0], vec![3]);
    let b = Tensor::from_vec_f32(vec![4.0f32, 5.0, 6.0], vec![3]);

    assert_eq!(a.add_tensor(&b).unwrap().dtype(), BaseType::F32);
    assert_eq!(a.sub_tensor(&b).unwrap().dtype(), BaseType::F32);
    assert_eq!(a.mul_tensor(&b).unwrap().dtype(), BaseType::F32);
    assert_eq!(a.div_tensor(&b).unwrap().dtype(), BaseType::F32);

    // 标量运算也保持 F32
    assert_eq!(a.add_scalar(1.0).dtype(), BaseType::F32);
    assert_eq!(a.mul_scalar(2.0).dtype(), BaseType::F32);

    // matmul 保持 F32
    let m1 = Tensor::from_vec_f32(vec![1.0f32, 2.0, 3.0, 4.0], vec![2, 2]);
    let m2 = Tensor::from_vec_f32(vec![1.0f32, 0.0, 0.0, 1.0], vec![2, 2]);
    assert_eq!(m1.matmul(&m2).unwrap().dtype(), BaseType::F32);

    // 激活函数保持 F32
    assert_eq!(a.relu().dtype(), BaseType::F32);
    assert_eq!(a.sigmoid().dtype(), BaseType::F32);
}

#[test]
fn f32_f64_sum_axis_parity() {
    // f32 sum_axis vs f64 sum_axis，结果应一致
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let t32 = Tensor::from_vec_f32(data.iter().map(|&v| v as f32).collect(), vec![2, 3]);
    let t64 = Tensor::from_vec(data, vec![2, 3]);

    let s32 = t32.sum_axis(1).unwrap();
    let s64 = t64.sum_axis(1).unwrap();

    assert!(s32.is_f32(), "f32 sum_axis 应保持 F32");
    assert!(s64.is_f64(), "f64 sum_axis 应保持 F64");
    assert_eq!(s32.shape(), s64.shape());
    assert_eq!(s32.shape(), vec![2]);

    let diff = max_abs_diff(&s32, &s64);
    assert!(diff < 1e-5, "f32 vs f64 sum_axis 误差 {} 超过 1e-5", diff);
}
