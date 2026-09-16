//! AUDIT-11.4.47 回归守护：`elementwise_binary` 同 shape 无拷贝快路径。
//!
//! 背景：`Tensor::elementwise_binary`（`tenth/src/runtime/tensor/methods.rs`）此前
//! **无条件**走 `broadcast().to_owned()` 物化全量副本 + `ArrayD::zeros` + 两次
//! `zip_mut_with`，即使两操作数 shape 完全相同也一样（1M f64 mul ≈7.1ms vs
//! 归约 sum ≈0.13ms，约 53× 差距，见 `docs/性能基线.md` §四 G2 / AUDIT-11.4.47）。
//!
//! 本次改动：两操作数 shape **完全相同**时走无广播、无操作数副本的直接逐元素
//! 路径（三路 `Zip`，dtype 已匹配的操作数零拷贝借用）；shape 不同时广播路径
//! **完全不变**。本文件是该改动的窄口径守护，断言四件事：
//!   ① 同 shape 结果 == 期望值（f64 / f32 / f16 / bf16，+ - * /）；
//!   ② 同 shape 快路径结果与广播路径结果**逐位一致**（`to_bits()` 精确比较，
//!      含 NaN / ±Inf / -0.0 等位模式敏感输入）；
//!   ③ 广播（shape 不同）路径行为不变（含不能广播时仍返回 Err）；
//!   ④ dtype 提升规则与返回 dtype 推导不变；且**输入张量未被就地修改**
//!      （快路径非 in-place，见 `docs/决策记录/2026-09-09-架构议题台账.md` ARCH-2 红线）。
//!
//! 验证命令（窄口径，走 cargo 槽位锁）：
//!   powershell -NoProfile -File .agents/tmp/cargo_slot.ps1 -Cmd "cargo test --release --manifest-path tenth/Cargo.toml --test elementwise_fastpath_test"

use half::{bf16, f16};

use tenth::hir::types::BaseType;
use tenth::runtime::tensor::Tensor;

// ── 辅助 ──────────────────────────────────────────────────────────────

fn t64(data: Vec<f64>, shape: Vec<usize>) -> Tensor {
    Tensor::from_vec(data, shape)
}

fn t32(data: Vec<f32>, shape: Vec<usize>) -> Tensor {
    Tensor::from_vec_f32(data, shape)
}

fn t16(data: Vec<f64>, shape: Vec<usize>) -> Tensor {
    Tensor::from_vec_f16(data.iter().map(|v| f16::from_f64(*v)).collect(), shape)
}

fn tbf16(data: Vec<f64>, shape: Vec<usize>) -> Tensor {
    Tensor::from_vec_bf16(data.iter().map(|v| bf16::from_f64(*v)).collect(), shape)
}

/// f64 张量的逐元素位模式（含 NaN 载荷）。
fn bits_f64(t: &Tensor) -> Vec<u64> {
    t.data.as_f64().expect("期望 F64 存储").iter().map(|v| v.to_bits()).collect()
}

/// f32 张量的逐元素位模式。
fn bits_f32(t: &Tensor) -> Vec<u32> {
    t.data.as_f32().expect("期望 F32 存储").iter().map(|v| v.to_bits()).collect()
}

/// f16 张量的逐元素位模式。
fn bits_f16(t: &Tensor) -> Vec<u16> {
    t.data.as_f16().expect("期望 F16 存储").iter().map(|v| v.to_bits()).collect()
}

/// bf16 张量的逐元素位模式。
fn bits_bf16(t: &Tensor) -> Vec<u16> {
    t.data.as_bf16().expect("期望 BF16 存储").iter().map(|v| v.to_bits()).collect()
}

// 位模式敏感的输入：0 / -0.0 / NaN / ±Inf / 极值，用来钉死「不因路径变化而重排、
// 不因任何优化而丢失特殊值语义」。
const WEIRD_A: [f64; 8] = [
    0.0,
    -0.0,
    f64::NAN,
    f64::INFINITY,
    f64::NEG_INFINITY,
    1.0,
    -2.5,
    f64::MIN_POSITIVE,
];
const WEIRD_B: [f64; 8] = [
    3.0,
    -7.5,
    2.0,
    0.5,
    -0.25,
    f64::NAN,
    -0.0,
    2.0,
];

/// 同 shape 快路径 vs 广播路径的逐位一致性核心断言（f64）。
///
/// 手法：同一批数值，一次以 (4,2) op (4,2) 走**同 shape 快路径**，一次把左操作数
/// 视作 (1,4,2) op (4,2) 走**广播路径**，两者结果元素序列必须逐位相同。
fn assert_same_shape_eq_broadcast_f64(a: &[f64], b: &[f64]) {
    let lhs_fast = t64(a.to_vec(), vec![4, 2]);
    let rhs_fast = t64(b.to_vec(), vec![4, 2]);
    let lhs_br = t64(a.to_vec(), vec![4, 2]).reshape(&[1, 4, 2]).unwrap();
    let rhs_br = t64(b.to_vec(), vec![4, 2]);

    for (name, fast, broadcast) in [
        ("add", lhs_fast.add_tensor(&rhs_fast).unwrap(), lhs_br.add_tensor(&rhs_br).unwrap()),
        ("sub", lhs_fast.sub_tensor(&rhs_fast).unwrap(), lhs_br.sub_tensor(&rhs_br).unwrap()),
        ("mul", lhs_fast.mul_tensor(&rhs_fast).unwrap(), lhs_br.mul_tensor(&rhs_br).unwrap()),
        ("div", lhs_fast.div_tensor(&rhs_fast).unwrap(), lhs_br.div_tensor(&rhs_br).unwrap()),
    ] {
        assert_eq!(fast.shape(), vec![4, 2], "{name}: 同 shape 结果 shape 应为 [4,2]");
        assert_eq!(broadcast.shape(), vec![1, 4, 2], "{name}: 广播路径结果 shape 应为 [1,4,2]");
        assert_eq!(
            bits_f64(&fast),
            bits_f64(&broadcast),
            "{name}: 同 shape 快路径与广播路径必须逐位一致\nfast={:?}\nbroadcast={:?}",
            bits_f64(&fast),
            bits_f64(&broadcast)
        );
    }
}

// ════════════════════════════════════════════════════════════════════════
// ① 同 shape 结果 == 期望值
// ════════════════════════════════════════════════════════════════════════

#[test]
fn f64_same_shape_matches_expected_values() {
    let a = t64(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let b = t64(vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0], vec![2, 3]);

    let sum = a.add_tensor(&b).unwrap();
    assert_eq!(bits_f64(&sum), bits_f64(&t64(vec![11.0, 22.0, 33.0, 44.0, 55.0, 66.0], vec![2, 3])));
    assert_eq!(sum.shape(), vec![2, 3]);

    let diff = a.sub_tensor(&b).unwrap();
    assert_eq!(bits_f64(&diff), bits_f64(&t64(vec![-9.0, -18.0, -27.0, -36.0, -45.0, -54.0], vec![2, 3])));

    let prod = a.mul_tensor(&b).unwrap();
    assert_eq!(bits_f64(&prod), bits_f64(&t64(vec![10.0, 40.0, 90.0, 160.0, 250.0, 360.0], vec![2, 3])));

    let quot = b.div_tensor(&a).unwrap();
    assert_eq!(bits_f64(&quot), bits_f64(&t64(vec![10.0, 10.0, 10.0, 10.0, 10.0, 10.0], vec![2, 3])));

    // 3D 同 shape
    let c = t64((0..24).map(|i| i as f64).collect(), vec![2, 3, 4]);
    let d = t64(vec![2.0; 24], vec![2, 3, 4]);
    let e = c.mul_tensor(&d).unwrap();
    assert_eq!(e.shape(), vec![2, 3, 4]);
    assert_eq!(
        bits_f64(&e),
        bits_f64(&t64((0..24).map(|i| (i as f64) * 2.0).collect(), vec![2, 3, 4]))
    );

    // 1D 同 shape（含长度 1 的退化情形）
    let f = t64(vec![1.5, -2.5, 3.5], vec![3]);
    let g = t64(vec![2.0, 4.0, -1.0], vec![3]);
    assert_eq!(bits_f64(&f.mul_tensor(&g).unwrap()), bits_f64(&t64(vec![3.0, -10.0, -3.5], vec![3])));

    let h = t64(vec![7.0], vec![1]);
    let i = t64(vec![3.0], vec![1]);
    assert_eq!(
        bits_f64(&h.add_tensor(&i).unwrap()),
        bits_f64(&t64(vec![10.0], vec![1])),
        "长度 1 的同 shape 也应走快路径且正确"
    );
}

#[test]
fn f32_same_shape_matches_expected_values() {
    let a = t32(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let b = t32(vec![0.5, -1.5, 2.0, 4.0], vec![2, 2]);

    let sum = a.add_tensor(&b).unwrap();
    assert_eq!(sum.dtype(), BaseType::F32, "F32+F32 结果必须仍是 F32");
    assert_eq!(bits_f32(&sum), bits_f32(&t32(vec![1.5, 0.5, 5.0, 8.0], vec![2, 2])));

    let prod = a.mul_tensor(&b).unwrap();
    assert_eq!(prod.dtype(), BaseType::F32);
    assert_eq!(bits_f32(&prod), bits_f32(&t32(vec![0.5, -3.0, 6.0, 16.0], vec![2, 2])));

    let diff = a.sub_tensor(&b).unwrap();
    assert_eq!(bits_f32(&diff), bits_f32(&t32(vec![0.5, 3.5, 1.0, 0.0], vec![2, 2])));

    let quot = a.div_tensor(&b).unwrap();
    assert_eq!(bits_f32(&quot), bits_f32(&t32(vec![2.0, -4.0 / 3.0, 1.5, 1.0], vec![2, 2])));
}

#[test]
fn f16_bf16_same_shape_matches_expected_values() {
    let a16 = t16(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let b16 = t16(vec![4.0, 5.0, 6.0, 7.0], vec![2, 2]);
    let sum16 = a16.add_tensor(&b16).unwrap();
    assert_eq!(sum16.dtype(), BaseType::F16, "F16+F16 结果必须仍是 F16");
    assert_eq!(bits_f16(&sum16), bits_f16(&t16(vec![5.0, 7.0, 9.0, 11.0], vec![2, 2])));
    assert_eq!(bits_f16(&a16.mul_tensor(&b16).unwrap()), bits_f16(&t16(vec![4.0, 10.0, 18.0, 28.0], vec![2, 2])));

    let abf = tbf16(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let bbf = tbf16(vec![4.0, 5.0, 6.0, 7.0], vec![2, 2]);
    let sumbf = abf.add_tensor(&bbf).unwrap();
    assert_eq!(sumbf.dtype(), BaseType::BF16, "BF16+BF16 结果必须仍是 BF16");
    assert_eq!(bits_bf16(&sumbf), bits_bf16(&tbf16(vec![5.0, 7.0, 9.0, 11.0], vec![2, 2])));
    assert_eq!(bits_bf16(&abf.sub_tensor(&bbf).unwrap()), bits_bf16(&tbf16(vec![-3.0, -3.0, -3.0, -3.0], vec![2, 2])));
}

// ════════════════════════════════════════════════════════════════════════
// ② 同 shape 快路径 == 广播路径（逐位一致）
// ════════════════════════════════════════════════════════════════════════

#[test]
fn f64_fastpath_matches_broadcast_path_bitwise() {
    let a: Vec<f64> = (0..8).map(|i| (i as f64) * 1.25 - 3.5).collect();
    let b: Vec<f64> = (0..8).map(|i| 7.5 - (i as f64) * 0.75).collect();
    assert_same_shape_eq_broadcast_f64(&a, &b);

    // 位模式敏感输入：NaN / ±Inf / -0.0 / MIN_POSITIVE 也必须逐位一致
    assert_same_shape_eq_broadcast_f64(&WEIRD_A, &WEIRD_B);
    // 反向（避免 op 顺序造成的假一致）
    assert_same_shape_eq_broadcast_f64(&WEIRD_B, &WEIRD_A);
}

#[test]
fn f32_fastpath_matches_broadcast_path_bitwise() {
    let a: Vec<f32> = WEIRD_A.iter().map(|v| *v as f32).collect();
    let b: Vec<f32> = WEIRD_B.iter().map(|v| *v as f32).collect();

    let lhs_fast = t32(a.clone(), vec![4, 2]);
    let rhs_fast = t32(b.clone(), vec![4, 2]);
    let lhs_br = t32(a, vec![4, 2]).reshape(&[1, 4, 2]).unwrap();
    let rhs_br = t32(b, vec![4, 2]);

    for (name, fast, broadcast) in [
        ("add", lhs_fast.add_tensor(&rhs_fast).unwrap(), lhs_br.add_tensor(&rhs_br).unwrap()),
        ("sub", lhs_fast.sub_tensor(&rhs_fast).unwrap(), lhs_br.sub_tensor(&rhs_br).unwrap()),
        ("mul", lhs_fast.mul_tensor(&rhs_fast).unwrap(), lhs_br.mul_tensor(&rhs_br).unwrap()),
        ("div", lhs_fast.div_tensor(&rhs_fast).unwrap(), lhs_br.div_tensor(&rhs_br).unwrap()),
    ] {
        assert_eq!(fast.dtype(), BaseType::F32, "{name}: f32 快路径 dtype 必须仍是 F32");
        assert_eq!(broadcast.dtype(), BaseType::F32, "{name}: f32 广播路径 dtype 必须仍是 F32");
        assert_eq!(
            bits_f32(&fast),
            bits_f32(&broadcast),
            "{name}: f32 同 shape 快路径与广播路径必须逐位一致"
        );
    }
}

#[test]
fn f16_bf16_fastpath_matches_broadcast_path_bitwise() {
    let a: Vec<f64> = vec![1.0, -2.0, 0.5, 4.0, 0.0, -0.0, 8.0, 0.25];
    let b: Vec<f64> = vec![3.0, 0.5, -1.5, 2.0, 7.0, -0.25, 1.0, 16.0];

    // F16：同 shape (4,2) vs 广播 (1,4,2)
    let a16f = t16(a.clone(), vec![4, 2]);
    let b16f = t16(b.clone(), vec![4, 2]);
    let a16b = t16(a.clone(), vec![4, 2]).reshape(&[1, 4, 2]).unwrap();
    let b16b = t16(b.clone(), vec![4, 2]);
    for (name, fast, broadcast) in [
        ("add", a16f.add_tensor(&b16f).unwrap(), a16b.add_tensor(&b16b).unwrap()),
        ("sub", a16f.sub_tensor(&b16f).unwrap(), a16b.sub_tensor(&b16b).unwrap()),
        ("mul", a16f.mul_tensor(&b16f).unwrap(), a16b.mul_tensor(&b16b).unwrap()),
        ("div", a16f.div_tensor(&b16f).unwrap(), a16b.div_tensor(&b16b).unwrap()),
    ] {
        assert_eq!(fast.dtype(), BaseType::F16, "{name}: f16 快路径 dtype 必须仍是 F16");
        assert_eq!(bits_f16(&fast), bits_f16(&broadcast), "{name}: f16 两路径必须逐位一致");
    }

    // BF16：同 shape (4,2) vs 广播 (1,4,2)——位模式敏感输入（NaN / ±Inf / -0.0）
    let wa = WEIRD_A.to_vec();
    let wb = WEIRD_B.to_vec();
    let abf_f = tbf16(wa.clone(), vec![4, 2]);
    let bbf_f = tbf16(wb.clone(), vec![4, 2]);
    let abf_b = tbf16(wa, vec![4, 2]).reshape(&[1, 4, 2]).unwrap();
    let bbf_b = tbf16(wb, vec![4, 2]);
    for (name, fast, broadcast) in [
        ("add", abf_f.add_tensor(&bbf_f).unwrap(), abf_b.add_tensor(&bbf_b).unwrap()),
        ("sub", abf_f.sub_tensor(&bbf_f).unwrap(), abf_b.sub_tensor(&bbf_b).unwrap()),
        ("mul", abf_f.mul_tensor(&bbf_f).unwrap(), abf_b.mul_tensor(&bbf_b).unwrap()),
        ("div", abf_f.div_tensor(&bbf_f).unwrap(), abf_b.div_tensor(&bbf_b).unwrap()),
    ] {
        assert_eq!(fast.dtype(), BaseType::BF16, "{name}: bf16 快路径 dtype 必须仍是 BF16");
        assert_eq!(bits_bf16(&fast), bits_bf16(&broadcast), "{name}: bf16 两路径必须逐位一致");
    }
}

// ════════════════════════════════════════════════════════════════════════
// ③ 广播（shape 不同）路径行为不变
// ════════════════════════════════════════════════════════════════════════

#[test]
fn broadcast_path_unchanged() {
    // (2,1) + (1,3) → (2,3)
    let a = t64(vec![1.0, 2.0], vec![2, 1]);
    let b = t64(vec![10.0, 20.0, 30.0], vec![1, 3]);
    let r = a.add_tensor(&b).unwrap();
    assert_eq!(r.shape(), vec![2, 3]);
    assert_eq!(
        bits_f64(&r),
        bits_f64(&t64(vec![11.0, 21.0, 31.0, 12.0, 22.0, 32.0], vec![2, 3]))
    );

    // (3,) * (2,3) → (2,3)（左侧低维广播）
    let c = t64(vec![1.0, 2.0, 3.0], vec![3]);
    let d = t64(vec![1.0, 1.0, 1.0, 2.0, 2.0, 2.0], vec![2, 3]);
    let r2 = c.mul_tensor(&d).unwrap();
    assert_eq!(r2.shape(), vec![2, 3]);
    assert_eq!(bits_f64(&r2), bits_f64(&t64(vec![1.0, 2.0, 3.0, 2.0, 4.0, 6.0], vec![2, 3])));

    // (1,) 标量式广播：(1,) + (3,) → (3,)（shape 不同，仍走广播路径）
    let e = t64(vec![5.0], vec![1]);
    let f = t64(vec![1.0, 2.0, 3.0], vec![3]);
    let r3 = e.add_tensor(&f).unwrap();
    assert_eq!(r3.shape(), vec![3]);
    assert_eq!(bits_f64(&r3), bits_f64(&t64(vec![6.0, 7.0, 8.0], vec![3])));

    // 4D 广播：(2,1,3,1) + (1,4,1,2) → (2,4,3,2)
    let g = t64((0..6).map(|i| i as f64).collect(), vec![2, 1, 3, 1]);
    let h = t64(vec![1.0; 8], vec![1, 4, 1, 2]);
    let r4 = g.add_tensor(&h).unwrap();
    assert_eq!(r4.shape(), vec![2, 4, 3, 2]);

    // 不可广播仍返回 Err（错误语义不变）
    let bad_a = t64(vec![1.0, 2.0], vec![2]);
    let bad_b = t64(vec![1.0, 2.0, 3.0], vec![3]);
    assert!(bad_a.add_tensor(&bad_b).is_err(), "(2,) + (3,) 必须仍返回 Err");
    assert!(bad_a.mul_tensor(&bad_b).is_err(), "(2,) * (3,) 必须仍返回 Err");
}

#[test]
fn broadcast_path_unchanged_f32_and_half() {
    // f32 广播 (2,1) * (1,3) → (2,3)
    let a = t32(vec![2.0, 3.0], vec![2, 1]);
    let b = t32(vec![1.0, 10.0, 100.0], vec![1, 3]);
    let r = a.mul_tensor(&b).unwrap();
    assert_eq!(r.shape(), vec![2, 3]);
    assert_eq!(r.dtype(), BaseType::F32);
    assert_eq!(
        bits_f32(&r),
        bits_f32(&t32(vec![2.0, 20.0, 200.0, 3.0, 30.0, 300.0], vec![2, 3]))
    );

    // f16 广播 (1,2) + (2,2) → (2,2)，dtype 保持 F16
    let c = t16(vec![1.0, 2.0], vec![1, 2]);
    let d = t16(vec![10.0, 20.0, 30.0, 40.0], vec![2, 2]);
    let r2 = c.add_tensor(&d).unwrap();
    assert_eq!(r2.shape(), vec![2, 2]);
    assert_eq!(r2.dtype(), BaseType::F16);
    assert_eq!(bits_f16(&r2), bits_f16(&t16(vec![11.0, 22.0, 31.0, 42.0], vec![2, 2])));
}

// ════════════════════════════════════════════════════════════════════════
// ④ dtype 提升规则不变 + 快路径不是 in-place
// ════════════════════════════════════════════════════════════════════════

#[test]
fn dtype_promotion_unchanged_on_fastpath() {
    // F32 + F64 → F64（同 shape 走快路径）
    let a = t32(vec![1.0, 2.0], vec![2]);
    let b = t64(vec![0.5, 0.25], vec![2]);
    let r = a.add_tensor(&b).unwrap();
    assert_eq!(r.dtype(), BaseType::F64, "F32+F64 必须提升为 F64");
    assert_eq!(bits_f64(&r), bits_f64(&t64(vec![1.5, 2.25], vec![2])));

    let r2 = b.add_tensor(&a).unwrap();
    assert_eq!(r2.dtype(), BaseType::F64, "F64+F32 必须提升为 F64");
    assert_eq!(bits_f64(&r2), bits_f64(&t64(vec![1.5, 2.25], vec![2])));

    // F16 + BF16 → F32（同 shape 走快路径）
    let c = t16(vec![1.0, 2.0], vec![2]);
    let d = tbf16(vec![0.5, 0.25], vec![2]);
    let r3 = c.mul_tensor(&d).unwrap();
    assert_eq!(r3.dtype(), BaseType::F32, "F16+BF16 必须提升为 F32");
    assert_eq!(bits_f32(&r3), bits_f32(&t32(vec![0.5, 0.5], vec![2])));

    // F16 + F32 → F32
    let e = t16(vec![2.0, 4.0], vec![2]);
    let f = t32(vec![3.0, 0.5], vec![2]);
    let r4 = e.add_tensor(&f).unwrap();
    assert_eq!(r4.dtype(), BaseType::F32, "F16+F32 必须提升为 F32");
    assert_eq!(bits_f32(&r4), bits_f32(&t32(vec![5.0, 4.5], vec![2])));
}

#[test]
fn fastpath_is_not_in_place() {
    // 快路径必须只读操作数（ARCH-2 红线：in-place 会写坏 autodiff tape 梯度）。
    let a = t64(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let b = t64(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]);
    let a_before = bits_f64(&a);
    let b_before = bits_f64(&b);

    let _ = a.add_tensor(&b).unwrap();
    let _ = a.mul_tensor(&b).unwrap();
    let _ = a.div_tensor(&b).unwrap();

    assert_eq!(bits_f64(&a), a_before, "左操作数不得被就地修改");
    assert_eq!(bits_f64(&b), b_before, "右操作数不得被就地修改");

    // f32 同理
    let c = t32(vec![1.0, 2.0], vec![2]);
    let d = t32(vec![3.0, 4.0], vec![2]);
    let c_before = bits_f32(&c);
    let _ = c.sub_tensor(&d).unwrap();
    assert_eq!(bits_f32(&c), c_before, "f32 左操作数不得被就地修改");
}
