// f32 标准库端到端测试 — Phase 5.4 Task 5.6
// 验证 zeros_f32/ones_f32/rand_f32 native + 内联 _f32 副本函数的 dtype 保持。
// 用 Interpreter（自带 native），无需手动注册。

use tenth::lexer::lexer::Lexer;
use tenth::parser::parser::Parser;
use tenth::hir::lower::Lowerer;
use tenth::runtime::interpreter::Interpreter;
use tenth::runtime::value::Value;
use tenth::hir::types::BaseType;

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

// ── 1. f32 native 构造函数 ─────────────────────────────────────

#[test]
fn test_zeros_f32_native() {
    let val = run_code("zeros_f32(2, 3)").unwrap().unwrap();
    match val {
        Value::Tensor(t) => {
            let t = t.borrow();
            assert!(t.is_f32(), "zeros_f32 应返回 F32 dtype");
            assert_eq!(t.dtype(), BaseType::F32);
            assert_eq!(t.shape(), vec![2, 3]);
            assert_eq!(t.get(&[0, 0]), Some(0.0));
        }
        other => panic!("期望 Tensor，得到 {:?}", other),
    }
}

#[test]
fn test_ones_f32_native() {
    let val = run_code("ones_f32(2, 3)").unwrap().unwrap();
    match val {
        Value::Tensor(t) => {
            let t = t.borrow();
            assert!(t.is_f32(), "ones_f32 应返回 F32 dtype");
            assert_eq!(t.dtype(), BaseType::F32);
            assert_eq!(t.shape(), vec![2, 3]);
            assert_eq!(t.get(&[0, 0]), Some(1.0));
        }
        other => panic!("期望 Tensor，得到 {:?}", other),
    }
}

#[test]
fn test_rand_f32_native() {
    let val = run_code("rand_f32(2, 3)").unwrap().unwrap();
    match val {
        Value::Tensor(t) => {
            let t = t.borrow();
            assert!(t.is_f32(), "rand_f32 应返回 F32 dtype");
            assert_eq!(t.dtype(), BaseType::F32);
            assert_eq!(t.shape(), vec![2, 3]);
        }
        other => panic!("期望 Tensor，得到 {:?}", other),
    }
}

// ── 2. 内联 _f32 副本函数端到端 ────────────────────────────────

#[test]
fn test_zeros_init_f32_inline() {
    // 内联 zeros_init_f32（来自 std/init/initializers.th）
    let src = r#"
        fn zeros_init_f32(rows: i64, cols: i64) -> Tensor[f64, ..] {
            zeros_f32(rows, cols)
        }
        zeros_init_f32(2, 3)
    "#;
    let val = run_code(src).unwrap().unwrap();
    match val {
        Value::Tensor(t) => {
            let t = t.borrow();
            assert!(t.is_f32(), "zeros_init_f32 应返回 F32 dtype");
            assert_eq!(t.dtype(), BaseType::F32);
            assert_eq!(t.shape(), vec![2, 3]);
        }
        other => panic!("期望 Tensor，得到 {:?}", other),
    }
}

#[test]
fn test_constant_init_f32_inline() {
    // 内联 constant_init_f32（来自 std/init/initializers.th）
    let src = r#"
        fn constant_init_f32(rows: i64, cols: i64, val: f32) -> Tensor[f64, ..] {
            zeros_f32(rows, cols) + val
        }
        constant_init_f32(2, 3, 0.5f32)
    "#;
    let val = run_code(src).unwrap().unwrap();
    match val {
        Value::Tensor(t) => {
            let t = t.borrow();
            assert!(t.is_f32(), "constant_init_f32 应保持 F32 dtype（f32 tensor + f32 标量）");
            assert_eq!(t.dtype(), BaseType::F32);
            assert_eq!(t.get(&[0, 0]), Some(0.5));
        }
        other => panic!("期望 Tensor，得到 {:?}", other),
    }
}

#[test]
fn test_xavier_uniform_f32_inline() {
    // 内联 xavier_uniform_f32（来自 std/init/initializers.th）
    let src = r#"
        fn xavier_uniform_f32(fan_in: i64, fan_out: i64) -> Tensor[f64, ..] {
            let limit = sqrt(6.0f32 / (fan_in + fan_out));
            randn_f32(fan_in, fan_out) * limit
        }
        xavier_uniform_f32(3, 4)
    "#;
    let val = run_code(src).unwrap().unwrap();
    match val {
        Value::Tensor(t) => {
            let t = t.borrow();
            assert!(t.is_f32(), "xavier_uniform_f32 应保持 F32 dtype（randn_f32 × f32 scalar）");
            assert_eq!(t.dtype(), BaseType::F32);
            assert_eq!(t.shape(), vec![3, 4]);
        }
        other => panic!("期望 Tensor，得到 {:?}", other),
    }
}

#[test]
fn test_dropout_f32_inline() {
    // 内联 dropout_f32（来自 std/nn/dropout.th）
    // 注意：dropout 默认 rate=0 时不丢弃，验证 dtype 保持
    let src = r#"
        fn dropout_f32(x: Tensor[f64, ..], rate: f32) -> Tensor[f64, ..] {
            x.dropout(rate)
        }
        dropout_f32(ones_f32(2, 3), 0.0f32)
    "#;
    let val = run_code(src).unwrap().unwrap();
    match val {
        Value::Tensor(t) => {
            let t = t.borrow();
            assert!(t.is_f32(), "dropout_f32 应保持 F32 dtype");
            assert_eq!(t.dtype(), BaseType::F32);
            assert_eq!(t.shape(), vec![2, 3]);
        }
        other => panic!("期望 Tensor，得到 {:?}", other),
    }
}

#[test]
fn test_sgd_step_f32_inline() {
    // 内联 sgd_step_f32（来自 std/optim/sgd.th）
    // sgd_step_f32(w, lr) = w - lr * grad(w)
    // 验证标量 lr: f32 与 f32 tensor 运算保持 F32
    let src = r#"
        fn sgd_step_f32(w: Tensor[f64, ..], lr: f32) -> Tensor[f64, ..] {
            let gw = grad(w);
            w - lr * gw
        }
        let w = ones_f32(2, 2);
        sgd_step_f32(w, 0.1f32)
    "#;
    let val = run_code(src).unwrap().unwrap();
    match val {
        Value::Tensor(t) => {
            let t = t.borrow();
            assert!(t.is_f32(), "sgd_step_f32 应保持 F32 dtype（lr: f32 × f32 grad）");
            assert_eq!(t.dtype(), BaseType::F32);
            assert_eq!(t.shape(), vec![2, 2]);
        }
        other => panic!("期望 Tensor，得到 {:?}", other),
    }
}

#[test]
fn test_f32_scalar_not_promoted_to_f64() {
    // 关键验证：f32 标量 + f32 tensor 不应提升为 f64
    // 对比：如果用 f64 标量（原 sgd_step），结果会提升为 f64
    let src_f32 = r#"
        fn step_f32(w: Tensor[f64, ..], lr: f32) -> Tensor[f64, ..] {
            w - lr * w
        }
        step_f32(ones_f32(2, 2), 0.1f32)
    "#;
    let val = run_code(src_f32).unwrap().unwrap();
    match val {
        Value::Tensor(t) => {
            let t = t.borrow();
            assert!(t.is_f32(), "f32 标量 × f32 tensor 应保持 F32（不提升为 f64）");
            assert_eq!(t.dtype(), BaseType::F32);
        }
        other => panic!("期望 Tensor，得到 {:?}", other),
    }
}

#[test]
fn test_f64_scalar_preserves_f32_tensor_dtype() {
    // 验证 scalar 方法按 tensor dtype 分支：f64 标量 × f32 tensor 仍保持 F32
    // （mul_scalar 内部 match tensor.data { F64 => f64运算, F32 => f32运算 }）
    // 这意味着 _f32 副本的主要价值在于使用 f32 字面量保持标量运算精度，
    // 而非防止 tensor dtype 提升（tensor dtype 始终由 tensor 自身决定）。
    let src = r#"
        fn step_f64(w: Tensor[f64, ..], lr: f64) -> Tensor[f64, ..] {
            w - lr * w
        }
        step_f64(ones_f32(2, 2), 0.1)
    "#;
    let val = run_code(src).unwrap().unwrap();
    match val {
        Value::Tensor(t) => {
            let t = t.borrow();
            assert!(t.is_f32(), "f64 标量 × f32 tensor 应保持 F32（scalar 方法按 tensor dtype 分支）");
            assert_eq!(t.dtype(), BaseType::F32);
        }
        other => panic!("期望 Tensor，得到 {:?}", other),
    }
}
