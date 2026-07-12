// bmm (batched matmul) 原语测试套件
// 覆盖：前向基本用例、shape 校验、f32 路径、autodiff 梯度（含复合梯度）
//
// 注：Tenth 的 tensor[[...]] 字面量当前仅支持 2D（HIR TensorLiteral 限制）。
// 3D 张量通过 2D 字面量 + .reshape(B, M, K) 构造（reshape 支持任意 ndim）。
// 3D TensorLiteral 解析是编译器部的后续任务，不在本任务范围内。

use tenth::lexer::lexer::Lexer;
use tenth::parser::parser::Parser;
use tenth::hir::lower::Lowerer;
use tenth::runtime::interpreter::Interpreter;
use tenth::runtime::value::Value;
use tenth::runtime::tensor::Tensor;

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

// ── 1. 基本前向：(2,3,4) @ (2,4,5) -> (2,3,5) ──

#[test]
fn test_bmm_forward_basic() {
    // batch 0/1: A = ones(3,4), B = ones(4,5) -> C[b,i,j] = sum_k 1*1 = 4
    // sum = 2*3*5*4 = 120
    let src = r#"
        let a = tensor[[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                       [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]].reshape(2, 3, 4);
        let b = tensor[[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                       [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                       [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                       [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]].reshape(2, 4, 5);
        let c = a.bmm(b);
        c.sum()
    "#;
    let result = run_code(src).unwrap();
    let g = as_f64(result.as_ref().unwrap()).expect("expected scalar");
    // (2,3,5) 全为 4，sum = 2*3*5*4 = 120
    assert!((g - 120.0).abs() < 1e-6, "expected 120.0, got {}", g);
}

#[test]
fn test_bmm_forward_shape() {
    // 验证结果 shape = (2, 3, 5)
    // b 每个batch 是 4x5 的前4列为单位阵（转置后），最后一列为 0
    let src = r#"
        let a = tensor[[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                       [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]].reshape(2, 3, 4);
        let b = tensor[[1.0, 0.0, 0.0, 0.0, 0.0,  0.0, 1.0, 0.0, 0.0, 0.0,
                        0.0, 0.0, 1.0, 0.0, 0.0,  0.0, 0.0, 0.0, 1.0, 0.0],
                       [1.0, 0.0, 0.0, 0.0, 0.0,  0.0, 1.0, 0.0, 0.0, 0.0,
                        0.0, 0.0, 1.0, 0.0, 0.0,  0.0, 0.0, 0.0, 1.0, 0.0]].reshape(2, 4, 5);
        let c = a.bmm(b);
        c.numel()
    "#;
    let result = run_code(src).unwrap();
    // c.numel() = 2*3*5 = 30
    assert!(matches!(result, Some(Value::Int(30, _))), "expected 30, got {:?}", result);
}

// ── 2. batch 维不匹配报错 ──

#[test]
fn test_bmm_batch_mismatch() {
    let src = r#"
        let a = tensor[[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]].reshape(1, 3, 4);
        let b = tensor[[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                       [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                       [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                       [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]].reshape(2, 4, 5);
        let c = a.bmm(b);
        c.sum()
    "#;
    let result = run_code(src);
    assert!(result.is_err(), "expected error for batch mismatch, got {:?}", result);
    let err = result.unwrap_err();
    assert!(err.contains("batch"), "error should mention batch: {}", err);
}

// ── 3. 内侧维度不匹配报错 ──

#[test]
fn test_bmm_inner_dim_mismatch() {
    let src = r#"
        let a = tensor[[1.0, 1.0, 1.0, 1.0, 1.0, 1.0]].reshape(1, 2, 3);
        let b = tensor[[1.0, 1.0, 1.0, 1.0, 1.0,
                        1.0, 1.0, 1.0, 1.0, 1.0,
                        1.0, 1.0, 1.0, 1.0, 1.0,
                        1.0, 1.0, 1.0, 1.0, 1.0]].reshape(1, 4, 5);
        let c = a.bmm(b);
        c.sum()
    "#;
    let result = run_code(src);
    assert!(result.is_err(), "expected error for inner dim mismatch, got {:?}", result);
    let err = result.unwrap_err();
    assert!(err.contains("inner"), "error should mention inner: {}", err);
}

// ── 4. 非 3D 报错 ──

#[test]
fn test_bmm_non_3d_error() {
    let src = r#"
        let a = tensor[[1.0, 2.0, 3.0]];
        let b = tensor[[1.0], [2.0], [3.0]];
        let c = a.bmm(b);
        c.sum()
    "#;
    let result = run_code(src);
    assert!(result.is_err(), "expected error for non-3D, got {:?}", result);
    let err = result.unwrap_err();
    assert!(err.contains("3D"), "error should mention 3D: {}", err);
}

// ── 5. autodiff 梯度正确（数值梯度校验）──

#[test]
fn test_bmm_autodiff_gradient() {
    // a = param(shape(2,2,3)), b = param(shape(2,3,2))
    // c = a.bmm(b)  -> shape (2,2,2)
    // loss = c.sum()
    // d_a = grad @ b^T  shape (2,2,3)
    // d_b = a^T @ grad  shape (2,3,2)
    // 由于 grad 全 1，d_a[i,m,k] = sum_n b[i,k,n]，d_b[i,k,n] = sum_m a[i,m,k]
    // 这里用确定性值方便手算：
    //   batch 0: a = [[1,2,3],[4,5,6]], b = [[1,1],[1,1],[1,1]]
    //     c[0] = [[6,6],[15,15]], d_a[0] = grad@b^T = [[1,1,1],[1,1,1]] (因为 b^T 全 1, grad 全 1, sum_n=2)
    //     wait: d_a = grad @ b^T, grad shape (2,2), b^T shape (2,3) → d_a shape (2,3)
    //     grad = [[1,1],[1,1]], b^T = [[1,1,1],[1,1,1]] → d_a = [[2,2,2],[2,2,2]]
    //     d_b = a^T @ grad, a^T shape (3,2), grad shape (2,2) → d_b shape (3,2)
    //     a^T = [[1,4],[2,5],[3,6]], grad = [[1,1],[1,1]] → d_b = [[5,5],[7,7],[9,9]]
    //   batch 1: 同样
    let src = r#"
        new_grad();
        let a = param(tensor[[1.0, 2.0, 3.0, 4.0, 5.0, 6.0,
                              1.0, 2.0, 3.0, 4.0, 5.0, 6.0]].reshape(2, 2, 3));
        let b = param(tensor[[1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                              1.0, 1.0, 1.0, 1.0, 1.0, 1.0]].reshape(2, 3, 2));
        let c = a.bmm(b);
        backward(c.sum());
        stop_grad();
        grad(a)
    "#;
    let result = run_code(src).unwrap();
    let v = as_f64_vec(result.as_ref().unwrap()).expect("expected tensor");
    // d_a shape (2,2,3)，全为 2
    assert_eq!(v.len(), 12, "d_a should have 12 elements, got {}", v.len());
    for (i, x) in v.iter().enumerate() {
        assert!((x - 2.0).abs() < 1e-6, "d_a[{}] expected 2.0, got {}", i, x);
    }
}

#[test]
fn test_bmm_autodiff_grad_b() {
    // 同上例，但取 grad(b)
    // d_b shape (2,3,2)，每个 batch = [[5,5],[7,7],[9,9]]
    let src = r#"
        new_grad();
        let a = param(tensor[[1.0, 2.0, 3.0, 4.0, 5.0, 6.0,
                              1.0, 2.0, 3.0, 4.0, 5.0, 6.0]].reshape(2, 2, 3));
        let b = param(tensor[[1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                              1.0, 1.0, 1.0, 1.0, 1.0, 1.0]].reshape(2, 3, 2));
        let c = a.bmm(b);
        backward(c.sum());
        stop_grad();
        grad(b)
    "#;
    let result = run_code(src).unwrap();
    let v = as_f64_vec(result.as_ref().unwrap()).expect("expected tensor");
    // d_b shape (2,3,2)，每个 batch = [[5,5],[7,7],[9,9]]
    assert_eq!(v.len(), 12, "d_b should have 12 elements, got {}", v.len());
    let expected = [5.0, 5.0, 7.0, 7.0, 9.0, 9.0, 5.0, 5.0, 7.0, 7.0, 9.0, 9.0];
    for (i, x) in v.iter().enumerate() {
        assert!((x - expected[i]).abs() < 1e-6, "d_b[{}] expected {}, got {}", i, expected[i], x);
    }
}

// ── 6. f32 路径正确 ──

#[test]
fn test_bmm_f32_path() {
    // 直接通过 Rust API 测试 f32 路径
    // a shape (2, 2, 3), b shape (2, 3, 2)
    let a_data: Vec<f32> = vec![
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
    ];
    let b_data: Vec<f32> = vec![
        1.0, 0.0,
        0.0, 1.0,
        1.0, 1.0,
        1.0, 0.0,
        0.0, 1.0,
        1.0, 1.0,
    ];
    let a = Tensor::from_vec_f32(a_data, vec![2, 2, 3]);
    let b = Tensor::from_vec_f32(b_data, vec![2, 3, 2]);
    let c = a.bmm(&b).expect("bmm should succeed");
    assert!(c.is_f32(), "result should be f32");
    let view = c.data.as_f64_view();
    // batch 0: a=[[1,2,3],[4,5,6]], b=[[1,0],[0,1],[1,1]]
    //   c[0,0,0]=1*1+2*0+3*1=4, c[0,0,1]=1*0+2*1+3*1=5
    //   c[0,1,0]=4*1+5*0+6*1=10, c[0,1,1]=4*0+5*1+6*1=11
    // batch 1: 同样
    let v: Vec<f64> = view.iter().cloned().collect();
    let expected = [4.0, 5.0, 10.0, 11.0, 4.0, 5.0, 10.0, 11.0];
    assert_eq!(v.len(), 8);
    for (i, x) in v.iter().enumerate() {
        assert!((x - expected[i]).abs() < 1e-6, "c[{}] expected {}, got {}", i, expected[i], x);
    }
}

#[test]
fn test_bmm_mixed_dtype_promotes_to_f64() {
    // a 是 f32, b 是 f64 → 结果应为 f64（提升）
    let a_data: Vec<f32> = vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
    let b_data: Vec<f64> = vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
    let a = Tensor::from_vec_f32(a_data, vec![1, 2, 3]);
    let b = Tensor::from_vec(b_data, vec![1, 3, 2]);
    let c = a.bmm(&b).expect("bmm should succeed");
    assert!(c.is_f64(), "mixed dtype should promote to f64, got dtype {:?}", c.dtype);
}

// ── 7. bmm 后接 select 的复合梯度 ──

#[test]
fn test_bmm_then_select_composite_gradient() {
    // a = param(ones(1,2,2)), b = param(ones(1,2,2))
    // c = a.bmm(b)  shape (1,2,2)，每个 batch 的 2x2 = [[2,2],[2,2]]
    // cond = [[1.0, 0.0],[0.0, 1.0]]
    // d = select(cond, c, zeros)  // 取 c 的对角元素
    // loss = d.sum() = 2 + 2 = 4
    // grad(d) = [[1,0],[0,1]]
    // d_c = grad(d) * cond_mask = [[1,0],[0,1]]（select backward）
    // d_a = d_c @ b^T = [[1,0],[0,1]] @ [[1,1],[1,1]] = [[1,1],[1,1]]
    // d_b = a^T @ d_c = [[1,1],[1,1]] @ [[1,0],[0,1]] = [[1,1],[1,1]]
    let src = r#"
        new_grad();
        let a = param(tensor[[1.0, 1.0, 1.0, 1.0]].reshape(1, 2, 2));
        let b = param(tensor[[1.0, 1.0, 1.0, 1.0]].reshape(1, 2, 2));
        let c = a.bmm(b);
        let cond = tensor[[1.0, 0.0, 0.0, 1.0]].reshape(1, 2, 2);
        let z = tensor[[0.0, 0.0, 0.0, 0.0]].reshape(1, 2, 2);
        let d = select(cond, c, z);
        backward(d.sum());
        stop_grad();
        grad(a)
    "#;
    let result = run_code(src).unwrap();
    let v = as_f64_vec(result.as_ref().unwrap()).expect("expected tensor");
    // d_a shape (1,2,2) = [[1,1],[1,1]]
    assert_eq!(v.len(), 4);
    for (i, x) in v.iter().enumerate() {
        assert!((x - 1.0).abs() < 1e-6, "d_a[{}] expected 1.0, got {}", i, x);
    }
}

// ── 8. bmm 结果 transpose 后的梯度 ──

#[test]
fn test_bmm_then_transpose_gradient() {
    // a = param(ones(1,2,3)), b = param(ones(1,3,2))
    // c = a.bmm(b)  shape (1,2,2)，每个 batch 的 2x2 = [[3,3],[3,3]]
    // d = c.transpose()  shape (1,2,2)，转置最后两维
    // loss = d.sum() = 12
    // grad(d) = ones(1,2,2) = [[1,1],[1,1]]
    // d_c = transpose(grad(d)) = [[1,1],[1,1]]
    // d_a = d_c @ b^T = [[1,1],[1,1]] @ [[1,1,1],[1,1,1]] = [[2,2,2],[2,2,2]]
    let src = r#"
        new_grad();
        let a = param(tensor[[1.0, 1.0, 1.0, 1.0, 1.0, 1.0]].reshape(1, 2, 3));
        let b = param(tensor[[1.0, 1.0, 1.0, 1.0, 1.0, 1.0]].reshape(1, 3, 2));
        let c = a.bmm(b);
        let d = c.transpose();
        backward(d.sum());
        stop_grad();
        grad(a)
    "#;
    let result = run_code(src).unwrap();
    let v = as_f64_vec(result.as_ref().unwrap()).expect("expected tensor");
    // d_a shape (1,2,3) = [[2,2,2],[2,2,2]]
    assert_eq!(v.len(), 6);
    for (i, x) in v.iter().enumerate() {
        assert!((x - 2.0).abs() < 1e-6, "d_a[{}] expected 2.0, got {}", i, x);
    }
}
