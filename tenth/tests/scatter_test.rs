// scatter 原语测试套件
// 覆盖：前向基本用例、shape 校验、autodiff 梯度（grad_src / grad_base）、复合梯度

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

// ── 1. 基本前向：base=[0,0,0,0], index=[1,3], src=[10,20] -> [0,10,0,20] ──
// 注：Tenth 的 tensor[[...]] 字面量总是构造 2D（shape [1,N]），scatter 要求 1D，
// 因此用 .flatten() 把 [1,N] 降为 [N]。

#[test]
fn test_scatter_forward_basic() {
    let src = r#"
        let base = tensor[[0.0, 0.0, 0.0, 0.0]].flatten();
        let index = tensor[[1.0, 3.0]].flatten();
        let src = tensor[[10.0, 20.0]].flatten();
        scatter(base, 0, index, src)
    "#;
    let result = run_code(src).unwrap();
    let v = as_f64_vec(result.as_ref().unwrap()).expect("expected tensor");
    assert_eq!(v, vec![0.0, 10.0, 0.0, 20.0]);
}

#[test]
fn test_scatter_forward_preserves_unscattered_values() {
    // base 有非零值，scatter 只覆盖 index 位置
    let src = r#"
        let base = tensor[[1.0, 2.0, 3.0, 4.0, 5.0]].flatten();
        let index = tensor[[0.0, 2.0]].flatten();
        let src = tensor[[100.0, 300.0]].flatten();
        scatter(base, 0, index, src)
    "#;
    let result = run_code(src).unwrap();
    let v = as_f64_vec(result.as_ref().unwrap()).expect("expected tensor");
    // out[0]=100, out[1]=2, out[2]=300, out[3]=4, out[4]=5
    assert_eq!(v, vec![100.0, 2.0, 300.0, 4.0, 5.0]);
}

// ── 2. index 越界报错 ──

#[test]
fn test_scatter_index_out_of_bounds() {
    let src = r#"
        let base = tensor[[0.0, 0.0, 0.0]].flatten();
        let index = tensor[[0.0, 5.0]].flatten();
        let src = tensor[[10.0, 20.0]].flatten();
        scatter(base, 0, index, src)
    "#;
    let result = run_code(src);
    assert!(result.is_err(), "expected error for index out of bounds, got {:?}", result);
    let err = result.unwrap_err();
    assert!(err.contains("越界") || err.contains("out"), "error should mention 越界: {}", err);
}

// ── 3. index/src 形状不匹配报错 ──

#[test]
fn test_scatter_index_src_shape_mismatch() {
    let src = r#"
        let base = tensor[[0.0, 0.0, 0.0, 0.0]].flatten();
        let index = tensor[[1.0, 3.0]].flatten();
        let src = tensor[[10.0, 20.0, 30.0]].flatten();
        scatter(base, 0, index, src)
    "#;
    let result = run_code(src);
    assert!(result.is_err(), "expected error for shape mismatch, got {:?}", result);
    let err = result.unwrap_err();
    assert!(err.contains("不匹配") || err.contains("shape"), "error should mention shape mismatch: {}", err);
}

// ── 4. autodiff: grad_src 正确（gather 语义）──

#[test]
fn test_scatter_autodiff_grad_src() {
    // base = [0, 0, 0, 0]（无梯度需要）
    // index = [1, 3]
    // src = param([10, 20])
    // out = scatter(base, 0, index, src) = [0, 10, 0, 20]
    // loss = out.sum() = 30
    // grad(out) = [1, 1, 1, 1]
    // grad_src[i] = grad[index[i]] = [grad[1], grad[3]] = [1, 1]
    let src = r#"
        new_grad();
        let base = tensor[[0.0, 0.0, 0.0, 0.0]].flatten();
        let index = tensor[[1.0, 3.0]].flatten();
        let s = param(tensor[[10.0, 20.0]].flatten());
        let out = scatter(base, 0, index, s);
        backward(out.sum());
        stop_grad();
        grad(s)
    "#;
    let result = run_code(src).unwrap();
    let v = as_f64_vec(result.as_ref().unwrap()).expect("expected tensor");
    // grad_src = [1, 1]
    assert_eq!(v.len(), 2, "grad_src should have 2 elements, got {}", v.len());
    assert!((v[0] - 1.0).abs() < 1e-6, "grad_src[0] expected 1.0, got {}", v[0]);
    assert!((v[1] - 1.0).abs() < 1e-6, "grad_src[1] expected 1.0, got {}", v[1]);
}

// ── 5. autodiff: grad_base 正确（index 位置为 0）──

#[test]
fn test_scatter_autodiff_grad_base() {
    // base = param([5, 5, 5, 5])
    // index = [1, 3]
    // src = [10, 20]
    // out = scatter(base, 0, index, src) = [5, 10, 5, 20]
    // loss = out.sum() = 40
    // grad(out) = [1, 1, 1, 1]
    // grad_base = [1, 0, 1, 0]（index 位置置 0）
    let src = r#"
        new_grad();
        let b = param(tensor[[5.0, 5.0, 5.0, 5.0]].flatten());
        let index = tensor[[1.0, 3.0]].flatten();
        let s = tensor[[10.0, 20.0]].flatten();
        let out = scatter(b, 0, index, s);
        backward(out.sum());
        stop_grad();
        grad(b)
    "#;
    let result = run_code(src).unwrap();
    let v = as_f64_vec(result.as_ref().unwrap()).expect("expected tensor");
    // grad_base = [1, 0, 1, 0]
    assert_eq!(v.len(), 4, "grad_base should have 4 elements, got {}", v.len());
    let expected = [1.0, 0.0, 1.0, 0.0];
    for (i, x) in v.iter().enumerate() {
        assert!((x - expected[i]).abs() < 1e-6, "grad_base[{}] expected {}, got {}", i, expected[i], x);
    }
}

// ── 6. scatter 后接 matmul 的复合梯度 ──

#[test]
fn test_scatter_then_matmul_composite_gradient() {
    // base = param([1, 1, 1, 1])  shape (4,)
    // src = param([10, 20])
    // index = [1, 3]
    // v = scatter(base, 0, index, src) = [1, 10, 1, 20]
    // w = [[1], [2], [3], [4]]  shape (4, 1)
    // y = v.matmul(w)  // 1D @ 2D -> 1D (1,)
    //   y[0] = 1*1 + 10*2 + 1*3 + 20*4 = 1 + 20 + 3 + 80 = 104
    // loss = y.sum() = 104
    // grad(y) = [1]
    // d_v = w * grad = [1, 2, 3, 4]
    // d_src[i] = d_v[index[i]] = [d_v[1], d_v[3]] = [2, 4]
    // d_base = d_v 但 index 位置置 0 = [1, 0, 3, 0]
    let src = r#"
        new_grad();
        let b = param(tensor[[1.0, 1.0, 1.0, 1.0]].flatten());
        let s = param(tensor[[10.0, 20.0]].flatten());
        let index = tensor[[1.0, 3.0]].flatten();
        let v = scatter(b, 0, index, s);
        let w = tensor[[1.0], [2.0], [3.0], [4.0]];
        let y = v.matmul(w);
        backward(y.sum());
        stop_grad();
        grad(s)
    "#;
    let result = run_code(src).unwrap();
    let v = as_f64_vec(result.as_ref().unwrap()).expect("expected tensor");
    // grad_src = [2, 4]
    assert_eq!(v.len(), 2, "grad_src should have 2 elements, got {}", v.len());
    assert!((v[0] - 2.0).abs() < 1e-6, "grad_src[0] expected 2.0, got {}", v[0]);
    assert!((v[1] - 4.0).abs() < 1e-6, "grad_src[1] expected 4.0, got {}", v[1]);
}

#[test]
fn test_scatter_then_matmul_composite_grad_base() {
    // 同上例，但取 grad(base)
    // d_base = d_v 但 index 位置置 0 = [1, 0, 3, 0]
    let src = r#"
        new_grad();
        let b = param(tensor[[1.0, 1.0, 1.0, 1.0]].flatten());
        let s = param(tensor[[10.0, 20.0]].flatten());
        let index = tensor[[1.0, 3.0]].flatten();
        let v = scatter(b, 0, index, s);
        let w = tensor[[1.0], [2.0], [3.0], [4.0]];
        let y = v.matmul(w);
        backward(y.sum());
        stop_grad();
        grad(b)
    "#;
    let result = run_code(src).unwrap();
    let v = as_f64_vec(result.as_ref().unwrap()).expect("expected tensor");
    // grad_base = [1, 0, 3, 0]
    assert_eq!(v.len(), 4, "grad_base should have 4 elements, got {}", v.len());
    let expected = [1.0, 0.0, 3.0, 0.0];
    for (i, x) in v.iter().enumerate() {
        assert!((x - expected[i]).abs() < 1e-6, "grad_base[{}] expected {}, got {}", i, expected[i], x);
    }
}

// ══════════════════════════════════════════════════════════════════════
// 7. 多维扩展测试（dim>0 + 多维 index/src，PyTorch 对齐）
// ══════════════════════════════════════════════════════════════════════

#[test]
fn scatter_dim1_basic() {
    // base = [[1,2,3],[4,5,6]]  shape (2,3)
    // dim = 1, index = [[0,2],[1,0]]  shape (2,2), src = [[10,20],[30,40]]  shape (2,2)
    // out = base.clone(); out[i, index[i,j]] = src[i,j]
    //   out[0,0]=10, out[0,2]=20, out[1,1]=30, out[1,0]=40
    // out = [[10,2,20],[40,30,6]]  row-major [10,2,20,40,30,6]
    let src = r#"
        let base = tensor[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let index = tensor[[0.0, 2.0], [1.0, 0.0]];
        let src = tensor[[10.0, 20.0], [30.0, 40.0]];
        scatter(base, 1, index, src)
    "#;
    let result = run_code(src).unwrap();
    let v = as_f64_vec(result.as_ref().unwrap()).expect("expected tensor");
    assert_eq!(v, vec![10.0, 2.0, 20.0, 40.0, 30.0, 6.0]);
}

#[test]
fn scatter_dim0_2d() {
    // base = [[0,0],[0,0],[0,0]]  shape (3,2)
    // dim = 0, index = [[0,1],[2,0]]  shape (2,2), src = [[10,20],[30,40]]  shape (2,2)
    // out = base.clone(); out[index[i,j], j] = src[i,j]
    //   out[0,0]=10, out[1,1]=20, out[2,0]=30, out[0,1]=40
    // out = [[10,40],[0,20],[30,0]]  row-major [10,40,0,20,30,0]
    let src = r#"
        let base = tensor[[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]];
        let index = tensor[[0.0, 1.0], [2.0, 0.0]];
        let src = tensor[[10.0, 20.0], [30.0, 40.0]];
        scatter(base, 0, index, src)
    "#;
    let result = run_code(src).unwrap();
    let v = as_f64_vec(result.as_ref().unwrap()).expect("expected tensor");
    assert_eq!(v, vec![10.0, 40.0, 0.0, 20.0, 30.0, 0.0]);
}

#[test]
fn scatter_multidim_index_src() {
    // 多维 index/src（非 1D）：验证 2D index/src 在 dim=1 下正确散布。
    // base = [[0,0,0],[0,0,0]]  shape (2,3)
    // dim = 1, index = [[0,1],[2,0]]  shape (2,2), src = [[7,8],[9,10]]  shape (2,2)
    // out[0,0]=7, out[0,1]=8, out[1,2]=9, out[1,0]=10
    // out = [[7,8,0],[10,0,9]]  row-major [7,8,0,10,0,9]
    let src = r#"
        let base = tensor[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]];
        let index = tensor[[0.0, 1.0], [2.0, 0.0]];
        let src = tensor[[7.0, 8.0], [9.0, 10.0]];
        scatter(base, 1, index, src)
    "#;
    let result = run_code(src).unwrap();
    let v = as_f64_vec(result.as_ref().unwrap()).expect("expected tensor");
    assert_eq!(v, vec![7.0, 8.0, 0.0, 10.0, 0.0, 9.0]);
}

#[test]
fn scatter_backward_dim1() {
    // 验证 scatter 在 dim=1 时 backward 正确（d_src = gather 语义，d_base = grad 但 actual 位置置 0）。
    // base = param([[1,1,1],[1,1,1]])  shape (2,3)
    // index = [[0,2],[1,0]]  shape (2,2), src = param([[10,20],[30,40]])  shape (2,2)
    // out = scatter(base, 1, index, src) = [[10,1,20],[40,30,1]]
    // loss = out.sum() = 10+1+20+40+30+1 = 102
    // grad(out) = ones (2,3)
    // d_src[idx] = grad[actual], actual[1]=index[idx]
    //   d_src[0,0]=grad[0,0]=1, d_src[0,1]=grad[0,2]=1
    //   d_src[1,0]=grad[1,1]=1, d_src[1,1]=grad[1,0]=1
    // d_src = [[1,1],[1,1]]  row-major [1,1,1,1]
    let src = r#"
        new_grad();
        let b = param(tensor[[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]);
        let index = tensor[[0.0, 2.0], [1.0, 0.0]];
        let s = param(tensor[[10.0, 20.0], [30.0, 40.0]]);
        let out = scatter(b, 1, index, s);
        backward(out.sum());
        stop_grad();
        grad(s)
    "#;
    let result = run_code(src).unwrap();
    let v = as_f64_vec(result.as_ref().unwrap()).expect("expected tensor");
    assert_eq!(v.len(), 4, "d_src should have 4 elements, got {}", v.len());
    for (i, x) in v.iter().enumerate() {
        assert!((x - 1.0).abs() < 1e-6, "d_src[{}] expected 1.0, got {}", i, x);
    }
}

#[test]
fn scatter_backward_dim1_grad_base() {
    // 同上例，但取 grad(base)。d_base = grad 但 actual 位置置 0。
    // actual = [0,0],[0,2],[1,1],[1,0] → 这些位置在 d_base 中置 0
    // d_base = [[0,1,0],[0,0,1]]  row-major [0,1,0,0,0,1]
    let src = r#"
        new_grad();
        let b = param(tensor[[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]);
        let index = tensor[[0.0, 2.0], [1.0, 0.0]];
        let s = param(tensor[[10.0, 20.0], [30.0, 40.0]]);
        let out = scatter(b, 1, index, s);
        backward(out.sum());
        stop_grad();
        grad(b)
    "#;
    let result = run_code(src).unwrap();
    let v = as_f64_vec(result.as_ref().unwrap()).expect("expected tensor");
    assert_eq!(v.len(), 6, "d_base should have 6 elements, got {}", v.len());
    let expected = [0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
    for (i, x) in v.iter().enumerate() {
        assert!((x - expected[i]).abs() < 1e-6, "d_base[{}] expected {}, got {}", i, expected[i], x);
    }
}

#[test]
fn scatter_backward_multidim() {
    // 多维场景 backward（dim=0）：
    // base = param([[1,1],[1,1],[1,1]])  shape (3,2)
    // index = [[0,1],[2,0]]  shape (2,2), src = param([[10,20],[30,40]])  shape (2,2)
    // out = scatter(base, 0, index, src)
    //   out[0,0]=10, out[1,1]=20, out[2,0]=30, out[0,1]=40
    //   out = [[10,40],[1,20],[30,1]]
    // loss = out.sum() = 10+40+1+20+30+1 = 102
    // grad(out) = ones (3,2)
    // d_src[idx] = grad[actual], actual[0]=index[idx]
    //   d_src[0,0]=grad[0,0]=1, d_src[0,1]=grad[1,1]=1
    //   d_src[1,0]=grad[2,0]=1, d_src[1,1]=grad[0,1]=1
    // d_src = [[1,1],[1,1]]  row-major [1,1,1,1]
    let src = r#"
        new_grad();
        let b = param(tensor[[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]]);
        let index = tensor[[0.0, 1.0], [2.0, 0.0]];
        let s = param(tensor[[10.0, 20.0], [30.0, 40.0]]);
        let out = scatter(b, 0, index, s);
        backward(out.sum());
        stop_grad();
        grad(s)
    "#;
    let result = run_code(src).unwrap();
    let v = as_f64_vec(result.as_ref().unwrap()).expect("expected tensor");
    assert_eq!(v.len(), 4, "d_src should have 4 elements, got {}", v.len());
    for (i, x) in v.iter().enumerate() {
        assert!((x - 1.0).abs() < 1e-6, "d_src[{}] expected 1.0, got {}", i, x);
    }
}

#[test]
fn scatter_dim_out_of_range_errors() {
    // base 2D（ndim=2），dim=2 >= 2 → 报错
    let src = r#"
        let base = tensor[[0.0, 0.0], [0.0, 0.0]];
        let index = tensor[[0.0, 1.0]];
        let s = tensor[[10.0, 20.0]];
        scatter(base, 2, index, s)
    "#;
    let result = run_code(src);
    assert!(result.is_err(), "expected error for dim out of range, got {:?}", result);
    let err = result.unwrap_err();
    assert!(err.contains("越界") || err.contains("dim"), "error should mention dim 越界: {}", err);
}

#[test]
fn scatter_shape_mismatch_errors() {
    // index shape (2,2) vs src shape (2,3) → 不匹配报错
    // 注：除 dim 维外 index.shape 必须与 base.shape 一致，所以构造合法 base (2,3)，
    // index (2,2) 在 dim=1 上 index.shape[1]=2 != base.shape[1]=3 会先报"维度不一致"。
    // 为精确测试 index.shape != src.shape，构造 base (2,2)，index (2,2)，src (2,3)：
    //   - dim=1 时 index.shape[1]=2 == base.shape[1]=2 ✓
    //   - 但 src.shape (2,3) != index.shape (2,2) → 报"不匹配"
    let src = r#"
        let base = tensor[[0.0, 0.0], [0.0, 0.0]];
        let index = tensor[[0.0, 1.0], [1.0, 0.0]];
        let s = tensor[[10.0, 20.0, 99.0], [30.0, 40.0, 88.0]];
        scatter(base, 1, index, s)
    "#;
    let result = run_code(src);
    assert!(result.is_err(), "expected error for shape mismatch, got {:?}", result);
    let err = result.unwrap_err();
    assert!(err.contains("不匹配") || err.contains("shape"), "error should mention shape mismatch: {}", err);
}
