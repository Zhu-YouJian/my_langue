// Reshape / MaskedFill 接入 autodiff 的专项测试
// 验证 TapeOp::Reshape 与 TapeOp::MaskedFill 的前向记录 + 反向传播正确性。
//
// 覆盖：
//   1. reshape 前向后向：shape (2,3) → (6,) → grad reshape 回 (2,3)
//   2. reshape 链式：reshape → matmul → backward 梯度正确
//   3. masked_fill 前向：mask 指定位置被 value 覆盖
//   4. masked_fill backward：grad 中 mask 位置为 0
//   5. masked_fill 链式：matmul → masked_fill → sum → backward
//   6. reshape + bmm 组合：3D reshape → bmm → backward
//   7. f32 路径正确（Rust API 直接验证）
//   8. 完整 MHA 微型示例：reshape → bmm → masked_fill → softmax → bmm → reshape → backward

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

// ── 1. reshape 前向后向：shape (2,3) → (6,) → grad reshape 回 (2,3) ──

#[test]
fn test_reshape_forward_backward_shape() {
    // x shape (2,3) → reshape (6,) → sum → backward
    // grad(x) 应为 shape (2,3) 全 1
    let src = r#"
        new_grad();
        let x = param(tensor[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
        let y = x.reshape(6);
        backward(y.sum());
        stop_grad();
        grad(x)
    "#;
    let result = run_code(src).unwrap();
    let v = as_f64_vec(result.as_ref().unwrap()).expect("expected tensor grad");
    // grad(x) shape (2,3)，全 1（因为 y.sum() 对 y 的梯度全 1，reshape 不改变元素数）
    assert_eq!(v.len(), 6, "grad should have 6 elements, got {}", v.len());
    for (i, x) in v.iter().enumerate() {
        assert!((x - 1.0).abs() < 1e-6, "grad[{}] expected 1.0, got {}", i, x);
    }
}

// ── 2. reshape 链式：reshape → matmul → backward 梯度正确 ──

#[test]
fn test_reshape_chain_with_matmul() {
    // x shape (6,) → reshape (2,3) → matmul w(3,2) → y(2,2) → sum → backward
    // x = [1,2,3,4,5,6], x_2d = [[1,2,3],[4,5,6]]
    // w = [[1,0],[0,1],[1,1]]
    // y = x_2d @ w = [[1+0+3, 0+2+3], [4+0+6, 0+5+6]] = [[4,5],[10,11]]
    // y.sum() = 30
    // grad_y = ones(2,2) = [[1,1],[1,1]]
    // d_x_2d = grad_y @ w^T, w^T = [[1,0,1],[0,1,1]]
    //   d_x_2d[0] = [1*1+1*0, 1*0+1*1, 1*1+1*1] = [1,1,2]
    //   d_x_2d[1] = [1,1,2]
    // d_x = d_x_2d.reshape(6) = [1,1,2,1,1,2], sum = 8
    let src = r#"
        new_grad();
        let x = param(tensor[[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]]);
        let w = param(tensor[[1.0, 0.0, 0.0, 1.0, 1.0, 1.0]].reshape(3, 2));
        let x_2d = x.reshape(2, 3);
        let y = x_2d.matmul(w);
        backward(y.sum());
        stop_grad();
        grad(x)
    "#;
    let result = run_code(src).unwrap();
    let v = as_f64_vec(result.as_ref().unwrap()).expect("expected tensor grad");
    assert_eq!(v.len(), 6, "grad(x) should have 6 elements, got {}", v.len());
    let expected = [1.0, 1.0, 2.0, 1.0, 1.0, 2.0];
    for (i, x) in v.iter().enumerate() {
        assert!((x - expected[i]).abs() < 1e-6, "grad(x)[{}] expected {}, got {}", i, expected[i], x);
    }
}

// ── 3. masked_fill 前向：mask 指定位置被 value 覆盖 ──

#[test]
fn test_masked_fill_forward() {
    // x = [[1,2],[3,4]], mask = [[1,0],[0,1]], value = -100
    // y = [[-100, 2],[3, -100]]
    // y.sum() = -100 + 2 + 3 + -100 = -195
    let src = r#"
        let x = tensor[[1.0, 2.0], [3.0, 4.0]];
        let mask = tensor[[1.0, 0.0], [0.0, 1.0]];
        let y = x.masked_fill(mask, -100.0);
        y.sum()
    "#;
    let result = run_code(src).unwrap();
    let g = as_f64(result.as_ref().unwrap()).expect("expected scalar");
    assert!((g - (-195.0)).abs() < 1e-6, "expected -195.0, got {}", g);
}

// ── 4. masked_fill backward：grad 中 mask 位置为 0 ──

#[test]
fn test_masked_fill_backward() {
    // x = [[1,2],[3,4]], mask = [[1,0],[0,1]], value = -100
    // y = x.masked_fill(mask, -100)
    // backward(y.sum())
    // grad(y) = ones(2,2)
    // grad(x) = grad(y) * (1 - mask) = [[0,1],[1,0]]
    let src = r#"
        new_grad();
        let x = param(tensor[[1.0, 2.0], [3.0, 4.0]]);
        let mask = tensor[[1.0, 0.0], [0.0, 1.0]];
        let y = x.masked_fill(mask, -100.0);
        backward(y.sum());
        stop_grad();
        grad(x)
    "#;
    let result = run_code(src).unwrap();
    let v = as_f64_vec(result.as_ref().unwrap()).expect("expected tensor grad");
    assert_eq!(v.len(), 4, "grad(x) should have 4 elements, got {}", v.len());
    // grad(x) = [[0,1],[1,0]] → flatten [0,1,1,0]
    let expected = [0.0, 1.0, 1.0, 0.0];
    for (i, x) in v.iter().enumerate() {
        assert!((x - expected[i]).abs() < 1e-6, "grad(x)[{}] expected {}, got {}", i, expected[i], x);
    }
}

// ── 5. masked_fill 链式：matmul → masked_fill → sum → backward ──

#[test]
fn test_masked_fill_chain_matmul() {
    // x = [[1,2],[3,4]], w = [[1,0],[0,1]] (单位阵)
    // y = x @ w = [[1,2],[3,4]]
    // mask = [[1,0],[0,0]], value = 0
    // y2 = y.masked_fill(mask, 0) = [[0,2],[3,4]]
    // backward(y2.sum())
    // grad(y2) = ones(2,2)
    // grad(y) = grad(y2) * (1 - mask) = [[0,1],[1,1]]
    // grad(x) = grad(y) @ w^T = [[0,1],[1,1]] @ [[1,0],[0,1]] = [[0,1],[1,1]]
    // grad(x) flatten = [0,1,1,1], sum = 3
    let src = r#"
        new_grad();
        let x = param(tensor[[1.0, 2.0], [3.0, 4.0]]);
        let w = param(tensor[[1.0, 0.0], [0.0, 1.0]]);
        let mask = tensor[[1.0, 0.0], [0.0, 0.0]];
        let y = x.matmul(w);
        let y2 = y.masked_fill(mask, 0.0);
        backward(y2.sum());
        stop_grad();
        grad(x)
    "#;
    let result = run_code(src).unwrap();
    let v = as_f64_vec(result.as_ref().unwrap()).expect("expected tensor grad");
    assert_eq!(v.len(), 4, "grad(x) should have 4 elements, got {}", v.len());
    // grad(x) = [[0,1],[1,1]] → flatten [0,1,1,1]
    let expected = [0.0, 1.0, 1.0, 1.0];
    for (i, x) in v.iter().enumerate() {
        assert!((x - expected[i]).abs() < 1e-6, "grad(x)[{}] expected {}, got {}", i, expected[i], x);
    }
}

// ── 6. reshape + bmm 组合：3D reshape → bmm → backward ──

#[test]
fn test_reshape_bmm_chain() {
    // a shape (12,) → reshape (2, 2, 3)
    // b shape (2, 3, 2) 全 1
    // c = a_3d.bmm(b)  shape (2, 2, 2)
    //   c[i,m,n] = sum_k a_3d[i,m,k] * b[i,k,n] = sum_k a_3d[i,m,k] * 1 = sum_k a_3d[i,m,k]
    // c.sum() backward
    // grad(c) = ones(2,2,2)
    // d_a_3d = grad @ b^T, b^T shape (2, 2, 3)
    //   d_a_3d[i,m,k] = sum_n grad[i,m,n] * b^T[i,k,n] = sum_n 1 * 1 = 2
    // d_a = d_a_3d.reshape(12) = [2,2,...,2] (12 个 2), sum = 24
    let src = r#"
        new_grad();
        let a = param(tensor[[1.0, 2.0, 3.0, 4.0, 5.0, 6.0,
                              1.0, 2.0, 3.0, 4.0, 5.0, 6.0]]);
        let b = param(tensor[[1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                              1.0, 1.0, 1.0, 1.0, 1.0, 1.0]].reshape(2, 3, 2));
        let a_3d = a.reshape(2, 2, 3);
        let c = a_3d.bmm(b);
        backward(c.sum());
        stop_grad();
        grad(a)
    "#;
    let result = run_code(src).unwrap();
    let v = as_f64_vec(result.as_ref().unwrap()).expect("expected tensor grad");
    assert_eq!(v.len(), 12, "grad(a) should have 12 elements, got {}", v.len());
    for (i, x) in v.iter().enumerate() {
        assert!((x - 2.0).abs() < 1e-6, "grad(a)[{}] expected 2.0, got {}", i, x);
    }
}

// ── 7. f32 路径正确（Rust API 直接验证） ──

#[test]
fn test_reshape_f32_path() {
    use tenth::runtime::autodiff::{Tape, TapeOp};
    use tenth::runtime::tensor::Tensor;
    use std::rc::Rc;
    use std::cell::RefCell;

    // f32 tensor reshape + backward
    // a shape (2,3) f32 → reshape (6,) → backward（seed=ones(6)）
    // grad(a) 应为 shape (2,3) 全 1，dtype f32
    let a = Rc::new(RefCell::new(Tensor::from_vec_f32(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        vec![2, 3],
    )));
    let mut tape = Tape::new();
    let a_id = tape.input(a.clone());

    // reshape (2,3) → (6,)
    let result_tensor = a.borrow().reshape(&[6]).expect("reshape failed");
    let result = Rc::new(RefCell::new(result_tensor));
    let r_id = tape.unary(TapeOp::Reshape, a_id, a.clone(), result.clone());

    // backward from reshape result（seed = ones(6)）
    tape.backward(r_id).expect("backward failed");

    // 检查梯度
    let a_ref = a.borrow();
    let grad = a_ref.grad.as_ref().expect("grad should be set");
    // grad 应为 f32 dtype
    match grad {
        tenth::runtime::tensor::TensorData::F32(arr) => {
            assert_eq!(arr.shape(), &[2, 3], "grad shape should be (2,3), got {:?}", arr.shape());
            for (i, v) in arr.iter().enumerate() {
                assert!((v - 1.0).abs() < 1e-6, "grad[{}] expected 1.0, got {}", i, v);
            }
        }
        other => panic!("expected F32 grad, got {:?}", other.dtype()),
    }
}

// ── 8. 完整 MHA 微型示例：reshape → bmm → masked_fill → softmax → bmm → reshape → backward ──

#[test]
fn test_mha_mini_example_backward() {
    // 微型 MHA：n_heads=1, seq_len=2, d_model=2, d_k=2
    // 数据流：
    //   q = x @ w_q               (2, 2)
    //   q_3d = q.reshape(1, 2, 2) (1, 2, 2)
    //   k_3d = q_3d, v_3d = q_3d  (简化)
    //   scores = q_3d.bmm(k_3d.transpose())  (1, 2, 2)
    //   mask_3d shape (1, 2, 2)
    //   masked = scores.masked_fill(mask_3d, -1e9)
    //   weights = masked.softmax()
    //   out_3d = weights.bmm(v_3d)  (1, 2, 2)
    //   out = out_3d.reshape(2, 2)
    //   backward(out.sum())
    // 主要验证：不报错 + grad(x) 和 grad(w_q) 非零
    let src = r#"
        new_grad();
        let x = param(tensor[[1.0, 0.0], [0.0, 1.0]]);
        let w_q = param(tensor[[1.0, 0.0], [0.0, 1.0]]);
        let q = x.matmul(w_q);
        let q_3d = q.reshape(1, 2, 2);
        let k_3d = q_3d;
        let v_3d = q_3d;
        let k_t = k_3d.transpose();
        let scores = q_3d.bmm(k_t);
        let mask = tensor[[0.0, 1.0, 0.0, 0.0]].reshape(1, 2, 2);
        let masked = scores.masked_fill(mask, -1e9);
        let weights = masked.softmax();
        let out_3d = weights.bmm(v_3d);
        let out = out_3d.reshape(2, 2);
        backward(out.sum());
        stop_grad();
        grad(x)
    "#;
    let result = run_code(src);
    assert!(result.is_ok(), "MHA mini backward failed: {:?}", result.err());
    let v = as_f64_vec(result.unwrap().as_ref().unwrap()).expect("expected tensor grad");
    // grad(x) shape (2,2)，4 个元素，验证非零（至少有一个非零）
    assert_eq!(v.len(), 4, "grad(x) should have 4 elements");
    let has_nonzero = v.iter().any(|x| x.abs() > 1e-10);
    assert!(has_nonzero, "grad(x) should have at least one nonzero element, got {:?}", v);
}

// ── 9. 补充：masked_fill + softmax 链式（验证 softmax 对 masked_fill 梯度传播） ──

#[test]
fn test_masked_fill_softmax_chain() {
    // x = [[1,2],[3,4]], mask = [[0,1],[0,0]], value = -1e9
    // masked = [[1, -1e9],[3, 4]]
    // weights = masked.softmax()  (沿最后一维)
    //   row 0: softmax([1, -1e9]) ≈ [1, 0]
    //   row 1: softmax([3, 4]) ≈ [~0.269, ~0.731]
    // loss = (weights * weights).sum()  （非对称 loss，避免 softmax 每行和为常数导致梯度为 0）
    // backward(loss)
    // 主要验证：不报错 + grad(x) 在 mask=true 位置为 0 + 非 mask 位置有梯度传播
    let src = r#"
        new_grad();
        let x = param(tensor[[1.0, 2.0], [3.0, 4.0]]);
        let mask = tensor[[0.0, 1.0], [0.0, 0.0]];
        let masked = x.masked_fill(mask, -1e9);
        let weights = masked.softmax();
        let loss = (weights * weights).sum();
        backward(loss);
        stop_grad();
        grad(x)
    "#;
    let result = run_code(src).unwrap();
    let v = as_f64_vec(result.as_ref().unwrap()).expect("expected tensor grad");
    assert_eq!(v.len(), 4, "grad(x) should have 4 elements, got {}", v.len());
    // grad(x) shape (2,2)，flatten [x00, x01, x10, x11]
    // mask=true 位置 (0,1) 的梯度应为 0（被 -1e9 覆盖，不传梯度回 x[0,1]）
    assert!(v[1].abs() < 1e-6, "grad(x)[0,1] (masked) expected ~0, got {}", v[1]);
    // 非 mask 位置应有梯度传播（loss = sum(weights^2) 对 x 的梯度在 row 1 非零）
    assert!(v[2].abs() > 1e-6 || v[3].abs() > 1e-6,
        "expected nonzero grad in non-masked row 1 positions, got {:?}", v);
}
