// f32 Tensor 专项测试 — Phase 1 Task 1.4
// 验证 f32 构造器、dtype 标记、运算 dtype 保持/提升、标量运算、reduction、Display。

use tenth::hir::types::BaseType;
use tenth::runtime::tensor::Tensor;

// ── 1. f32 构造器 ──────────────────────────────────────────────────

#[test]
fn test_zeros_f32() {
    let t = Tensor::zeros_f32(&[2, 3]);
    assert!(t.is_f32());
    assert_eq!(t.dtype(), BaseType::F32);
    assert_eq!(t.shape(), vec![2, 3]);
    assert_eq!(t.get(&[0, 0]), Some(0.0));
}

#[test]
fn test_ones_f32() {
    let t = Tensor::ones_f32(&[3]);
    assert!(t.is_f32());
    assert_eq!(t.shape(), vec![3]);
    assert_eq!(t.get(&[0]), Some(1.0));
    assert_eq!(t.get(&[2]), Some(1.0));
}

#[test]
fn test_full_f32() {
    let t = Tensor::full_f32(&[2, 2], 3.5);
    assert!(t.is_f32());
    assert_eq!(t.get(&[0, 0]), Some(3.5));
    assert_eq!(t.get(&[1, 1]), Some(3.5));
}

#[test]
fn test_from_vec_f32() {
    let t = Tensor::from_vec_f32(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    assert!(t.is_f32());
    assert_eq!(t.get(&[0, 0]), Some(1.0));
    assert_eq!(t.get(&[1, 1]), Some(4.0));
}

#[test]
fn test_arange_f32() {
    let t = Tensor::arange_f32(0.0, 5.0, 1.0);
    assert!(t.is_f32());
    assert_eq!(t.shape(), vec![5]);
    assert_eq!(t.get(&[0]), Some(0.0));
    assert_eq!(t.get(&[4]), Some(4.0));
}

#[test]
fn test_eye_f32() {
    let t = Tensor::eye_f32(3);
    assert!(t.is_f32());
    assert_eq!(t.get(&[0, 0]), Some(1.0));
    assert_eq!(t.get(&[1, 0]), Some(0.0));
    assert_eq!(t.get(&[2, 2]), Some(1.0));
}

// ── 2. dtype 标记 ─────────────────────────────────────────────────

#[test]
fn test_f64_constructors_still_f64() {
    let t = Tensor::zeros(&[2]);
    assert!(t.is_f64());
    assert_eq!(t.dtype(), BaseType::F64);
}

#[test]
fn test_zeros_with_dtype_f32() {
    let t = Tensor::zeros_with_dtype(&[2, 2], BaseType::F32);
    assert!(t.is_f32());
}

#[test]
fn test_zeros_with_dtype_f64() {
    let t = Tensor::zeros_with_dtype(&[2, 2], BaseType::F64);
    assert!(t.is_f64());
}

// ── 3. f32 + f32 → f32（dtype 保持）──────────────────────────────

#[test]
fn test_f32_add_f32_stays_f32() {
    let a = Tensor::from_vec_f32(vec![1.0, 2.0, 3.0], vec![3]);
    let b = Tensor::from_vec_f32(vec![4.0, 5.0, 6.0], vec![3]);
    let c = a.add_tensor(&b).unwrap();
    assert!(c.is_f32(), "f32 + f32 should stay f32");
    assert_eq!(c.get(&[0]), Some(5.0));
    assert_eq!(c.get(&[1]), Some(7.0));
    assert_eq!(c.get(&[2]), Some(9.0));
}

#[test]
fn test_f32_sub_f32_stays_f32() {
    let a = Tensor::from_vec_f32(vec![10.0, 20.0], vec![2]);
    let b = Tensor::from_vec_f32(vec![3.0, 7.0], vec![2]);
    let c = a.sub_tensor(&b).unwrap();
    assert!(c.is_f32());
    assert_eq!(c.get(&[0]), Some(7.0));
    assert_eq!(c.get(&[1]), Some(13.0));
}

#[test]
fn test_f32_mul_f32_stays_f32() {
    let a = Tensor::from_vec_f32(vec![2.0, 3.0], vec![2]);
    let b = Tensor::from_vec_f32(vec![4.0, 5.0], vec![2]);
    let c = a.mul_tensor(&b).unwrap();
    assert!(c.is_f32());
    assert_eq!(c.get(&[0]), Some(8.0));
    assert_eq!(c.get(&[1]), Some(15.0));
}

// ── 4. f32 + f64 → f64（dtype 提升）──────────────────────────────

#[test]
fn test_f32_add_f64_promotes_to_f64() {
    let a = Tensor::from_vec_f32(vec![1.0, 2.0], vec![2]);
    let b = Tensor::from_vec(vec![3.0, 4.0], vec![2]);
    let c = a.add_tensor(&b).unwrap();
    assert!(c.is_f64(), "f32 + f64 should promote to f64");
    assert_eq!(c.get(&[0]), Some(4.0));
    assert_eq!(c.get(&[1]), Some(6.0));
}

// ── 5. f32 标量运算保持 f32 ──────────────────────────────────────

#[test]
fn test_f32_add_scalar_stays_f32() {
    let a = Tensor::from_vec_f32(vec![1.0, 2.0, 3.0], vec![3]);
    let c = a.add_scalar(10.0);
    assert!(c.is_f32());
    assert_eq!(c.get(&[0]), Some(11.0));
}

#[test]
fn test_f32_mul_scalar_stays_f32() {
    let a = Tensor::from_vec_f32(vec![1.0, 2.0, 3.0], vec![3]);
    let c = a.mul_scalar(2.0);
    assert!(c.is_f32());
    assert_eq!(c.get(&[2]), Some(6.0));
}

#[test]
fn test_f32_div_scalar_stays_f32() {
    let a = Tensor::from_vec_f32(vec![10.0, 20.0], vec![2]);
    let c = a.div_scalar(2.0);
    assert!(c.is_f32());
    assert_eq!(c.get(&[0]), Some(5.0));
    assert_eq!(c.get(&[1]), Some(10.0));
}

// ── 6. f32 reduction 正确性 ──────────────────────────────────────

#[test]
fn test_f32_sum() {
    let a = Tensor::from_vec_f32(vec![1.0, 2.0, 3.0, 4.0], vec![4]);
    assert_eq!(a.sum(), 10.0);
}

#[test]
fn test_f32_mean() {
    let a = Tensor::from_vec_f32(vec![2.0, 4.0, 6.0], vec![3]);
    assert_eq!(a.mean(), 4.0);
}

#[test]
fn test_f32_max_val() {
    let a = Tensor::from_vec_f32(vec![3.0, 7.0, 2.0, 9.0, 1.0], vec![5]);
    assert_eq!(a.max_val(), 9.0);
}

#[test]
fn test_f32_sum_axis() {
    let a = Tensor::from_vec_f32(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let s = a.sum_axis(1).unwrap();
    assert!(s.is_f32());
    assert_eq!(s.shape(), vec![2]);
    assert_eq!(s.get(&[0]), Some(3.0));  // 1+2
    assert_eq!(s.get(&[1]), Some(7.0));  // 3+4
}

// ── 7. f32 Display 格式化 ────────────────────────────────────────

#[test]
fn test_f32_display() {
    let a = Tensor::from_vec_f32(vec![1.0, 2.0, 3.0], vec![3]);
    let s = format!("{}", a.data);
    assert!(s.starts_with("f32"), "Display should start with dtype: {}", s);
}

#[test]
fn test_f64_display() {
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
    let s = format!("{}", a.data);
    assert!(s.starts_with("f64"), "Display should start with dtype: {}", s);
}

// ── 8. f32 broadcast 运算 ────────────────────────────────────────

#[test]
fn test_f32_broadcast_add() {
    let a = Tensor::from_vec_f32(vec![1.0, 2.0, 3.0], vec![3]);
    let b = Tensor::from_vec_f32(vec![10.0], vec![1]);
    let c = a.add_tensor(&b).unwrap();
    assert!(c.is_f32());
    assert_eq!(c.get(&[0]), Some(11.0));
    assert_eq!(c.get(&[1]), Some(12.0));
    assert_eq!(c.get(&[2]), Some(13.0));
}
