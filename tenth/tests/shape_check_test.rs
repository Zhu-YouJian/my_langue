//! Tensor shape validation tests.
//!
//! Verifies that shape mismatches and invalid inputs produce proper errors
//! instead of panics, silent no-ops, or incorrect results.

use tenth::runtime::tensor::Tensor;
use tenth::runtime::autodiff::{Tape, TapeOp};
use std::rc::Rc;
use std::cell::RefCell;

fn make_tensor(data: Vec<f64>, shape: Vec<usize>) -> Rc<RefCell<Tensor>> {
    Rc::new(RefCell::new(Tensor::from_vec(data, shape)))
}

// ── Broadcasting ──────────────────────────────────────────────────────────

#[test]
fn test_broadcast_bidirectional_2d() {
    // (2,1) + (1,3) → (2,3) — the case the old single-direction logic failed on
    let a = Tensor::from_vec(vec![1.0, 2.0], vec![2, 1]);
    let b = Tensor::from_vec(vec![10.0, 20.0, 30.0], vec![1, 3]);
    let result = a.add_tensor(&b).expect("bidirectional broadcast should succeed");
    assert_eq!(result.shape(), vec![2, 3]);
    // Row 0: 1+10, 1+20, 1+30
    // Row 1: 2+10, 2+20, 2+30
    let data = result.data;
    assert_eq!(data[[0, 0]], 11.0);
    assert_eq!(data[[0, 1]], 21.0);
    assert_eq!(data[[0, 2]], 31.0);
    assert_eq!(data[[1, 0]], 12.0);
    assert_eq!(data[[1, 1]], 22.0);
    assert_eq!(data[[1, 2]], 32.0);
}

#[test]
fn test_broadcast_incompatible_returns_err() {
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
    let b = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
    assert!(a.add_tensor(&b).is_err());
    assert!(a.sub_tensor(&b).is_err());
    assert!(a.mul_tensor(&b).is_err());
    assert!(a.div_tensor(&b).is_err());
}

#[test]
fn test_broadcast_scalar_like() {
    // (2,2) + (1,) → (2,2)
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let b = Tensor::from_vec(vec![10.0], vec![1]);
    let result = a.add_tensor(&b).unwrap();
    assert_eq!(result.shape(), vec![2, 2]);
    assert_eq!(result.data[[0, 0]], 11.0);
    assert_eq!(result.data[[1, 1]], 14.0);
}

// ── assign_ ───────────────────────────────────────────────────────────────

#[test]
fn test_assign_shape_mismatch_returns_err() {
    let mut a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    // (3,) cannot broadcast into (2,2) — last dim 3 != 2
    let b = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
    // (3) cannot broadcast into (2,2) — should error, not silently no-op
    assert!(a.assign_(&b).is_err());
    // Original data should be unchanged
    assert_eq!(a.data[[0, 0]], 1.0);
    assert_eq!(a.data[[1, 1]], 4.0);
}

#[test]
fn test_assign_compatible_broadcast() {
    let mut a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let b = Tensor::from_vec(vec![10.0], vec![1]);
    a.assign_(&b).unwrap();
    assert_eq!(a.data[[0, 0]], 10.0);
    assert_eq!(a.data[[1, 1]], 10.0);
}

// ── layer_norm shape validation ───────────────────────────────────────────

#[test]
fn test_layer_norm_gamma_shape_mismatch() {
    let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    // gamma should be (2,) but we pass (3,)
    let gamma = Tensor::from_vec(vec![1.0, 1.0, 1.0], vec![3]);
    let beta = Tensor::from_vec(vec![0.0, 0.0], vec![2]);
    let result = x.layer_norm(&gamma, &beta, 1e-5);
    assert!(result.is_err(), "layer_norm should reject mismatched gamma shape");
}

#[test]
fn test_layer_norm_beta_shape_mismatch() {
    let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let gamma = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
    // beta should be (2,) but we pass (1,)
    let beta = Tensor::from_vec(vec![0.0], vec![1]);
    let result = x.layer_norm(&gamma, &beta, 1e-5);
    assert!(result.is_err(), "layer_norm should reject mismatched beta shape");
}

#[test]
fn test_layer_norm_correct_shapes() {
    let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let gamma = Tensor::from_vec(vec![1.0, 1.0], vec![2]);
    let beta = Tensor::from_vec(vec![0.0, 0.0], vec![2]);
    let result = x.layer_norm(&gamma, &beta, 1e-5);
    assert!(result.is_ok());
    assert_eq!(result.unwrap().shape(), vec![2, 2]);
}

// ── sum_axis validation ───────────────────────────────────────────────────

#[test]
fn test_sum_axis_out_of_bounds() {
    let t = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    // axis 2 is out of bounds for 2-D tensor
    assert!(t.sum_axis(2).is_err());
    assert!(t.sum_axis(5).is_err());
}

#[test]
fn test_sum_axis_valid() {
    let t = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let result = t.sum_axis(0).unwrap();
    assert_eq!(result.shape(), vec![2]);
    assert_eq!(result.data[[0]], 4.0); // 1+3
    assert_eq!(result.data[[1]], 6.0); // 2+4
}

// ── max_val / argmax on empty tensors ─────────────────────────────────────

#[test]
fn test_max_val_empty_tensor() {
    let t = Tensor::zeros(&[0]);
    // Should not panic; returns -inf for empty
    let m = t.max_val();
    assert!(m.is_infinite() && m.is_sign_negative());
}

#[test]
fn test_argmax_empty_tensor() {
    let t = Tensor::zeros(&[0]);
    // Should not panic; returns -1
    assert_eq!(t.argmax(), -1);
}

// ── matmul shape mismatch ─────────────────────────────────────────────────

#[test]
fn test_matmul_shape_mismatch() {
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
    let b = Tensor::from_vec(vec![1.0, 2.0], vec![2, 1]);
    // a is (1,3), b is (2,1) — inner dims 3 != 2
    assert!(a.matmul(&b).is_err());
}

#[test]
fn test_matmul_valid_2d() {
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let b = Tensor::from_vec(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]);
    let result = a.matmul(&b).unwrap();
    assert_eq!(result.shape(), vec![2, 2]);
    // [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]] = [[19,22],[43,50]]
    assert_eq!(result.data[[0, 0]], 19.0);
    assert_eq!(result.data[[1, 1]], 50.0);
}

// ── autodiff MatMul backward for 1D inputs ────────────────────────────────

#[test]
fn test_backward_matmul_1d_2d() {
    // a: (3,), b: (3, 2) → result: (2,)
    // d_a = grad @ b^T  → (3,)
    // d_b = a^T @ grad  → (3, 2)
    let a = make_tensor(vec![1.0, 2.0, 3.0], vec![3]);
    let b = make_tensor(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![3, 2]);

    // Forward: a @ b = [1*1+2*3+3*5, 1*2+2*4+3*6] = [22, 28]
    let a_data = a.borrow().data.clone();
    let b_data = b.borrow().data.clone();
    // Use 1D@2D path
    let a_1d = a_data.view().into_dimensionality::<ndarray::Ix1>().unwrap();
    let b_2d = b_data.view().into_dimensionality::<ndarray::Ix2>().unwrap();
    let result_data = a_1d.dot(&b_2d).into_dyn();
    let result_vec: Vec<f64> = result_data.iter().cloned().collect();
    let result = Rc::new(RefCell::new(Tensor::from_vec(result_vec, result_data.shape().to_vec())));
    assert_eq!(result.borrow().shape(), vec![2]);

    let mut tape = Tape::new();
    tape.input(a.clone());
    tape.input(b.clone());
    let r_id = tape.binary_direct(TapeOp::MatMul, a.clone(), b.clone(), result.clone());

    tape.backward(r_id);

    // Gradients should be propagated (not silently dropped)
    let a_grad = a.borrow().grad.clone();
    assert!(a_grad.is_some(), "a should receive gradient");
    let a_grad = a_grad.unwrap();
    assert_eq!(a_grad.shape(), vec![3], "a grad shape should match a (1D)");

    let b_grad = b.borrow().grad.clone();
    assert!(b_grad.is_some(), "b should receive gradient");
    let b_grad = b_grad.unwrap();
    assert_eq!(b_grad.shape(), vec![3, 2], "b grad shape should match b (2D)");
}

#[test]
fn test_backward_matmul_2d_1d() {
    // a: (2, 3), b: (3,) → result: (2,)
    // d_a = grad @ b^T  → (2, 3)
    // d_b = a^T @ grad  → (3,)
    let a = make_tensor(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let b = make_tensor(vec![1.0, 2.0, 3.0], vec![3]);

    let a_data = a.borrow().data.clone();
    let b_data = b.borrow().data.clone();
    let a_2d = a_data.view().into_dimensionality::<ndarray::Ix2>().unwrap();
    let b_1d = b_data.view().into_dimensionality::<ndarray::Ix1>().unwrap();
    let result_data = a_2d.dot(&b_1d).into_dyn();
    let result_vec: Vec<f64> = result_data.iter().cloned().collect();
    let result = Rc::new(RefCell::new(Tensor::from_vec(result_vec, result_data.shape().to_vec())));
    assert_eq!(result.borrow().shape(), vec![2]);

    let mut tape = Tape::new();
    tape.input(a.clone());
    tape.input(b.clone());
    let r_id = tape.binary_direct(TapeOp::MatMul, a.clone(), b.clone(), result.clone());

    tape.backward(r_id);

    let a_grad = a.borrow().grad.clone().expect("a should receive gradient");
    assert_eq!(a_grad.shape(), vec![2, 3]);

    let b_grad = b.borrow().grad.clone().expect("b should receive gradient");
    assert_eq!(b_grad.shape(), vec![3]);
}
