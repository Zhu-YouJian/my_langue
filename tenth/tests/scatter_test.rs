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
