//! 编译期 Tensor shape 检查测试。
//!
//! 验证 shape 不匹配在编译期（lower 阶段）报错，而非运行时崩溃。
//! 这是 Tenth 的核心卖点之一（路线图阶段 3 目标）。
//!
//! Phase 1 覆盖：
//! - matmul (M,K) @ (K,N)：K 不匹配编译期报错
//! - 二元运算 broadcast：不兼容 shape 编译期报错
//! - 字面量构造函数 shape 推断：zeros(3,4) → Tensor[_, 3, 4]
//! - transpose shape 反转：(M,N).transpose() → (N,M)

use tenth::lexer::lexer::Lexer;
use tenth::parser::parser::Parser;
use tenth::hir::lower::Lowerer;
use tenth::error::TenthError;

fn lower(src: &str) -> Result<(), TenthError> {
    let mut lexer = Lexer::new(src);
    let tokens = lexer.tokenize().expect("lex");
    let mut parser = Parser::new(tokens);
    let program = parser.parse_program().expect("parse");
    let mut lowerer = Lowerer::new();
    lowerer.lower_program(&program).map(|_| ())
}

/// 辅助：断言 lower 失败且错误信息包含指定子串。
fn assert_compile_error(src: &str, expected_msg_part: &str) {
    match lower(src) {
        Err(TenthError::TypeError { message, .. }) => {
            assert!(
                message.contains(expected_msg_part),
                "错误信息不包含预期子串 '{}'\n实际: {}",
                expected_msg_part, message
            );
        }
        Err(other) => panic!("期望 TypeError，实际: {:?}", other),
        Ok(_) => panic!("期望编译失败但成功了；期望错误包含 '{}'", expected_msg_part),
    }
}

// ── matmul shape 检查 ──────────────────────────────────────────────────────

#[test]
fn matmul_k_mismatch_reports_compile_error() {
    // (3, 4) @ (5, 6) — 内侧 4 ≠ 5，应编译期报错
    let src = r#"
fn bad() -> Tensor[f64, ..] {
    let a = zeros(3, 4);
    let b = zeros(5, 6);
    a.matmul(b)
}
"#;
    assert_compile_error(src, "matmul shape 不兼容");
}

#[test]
fn matmul_correct_shape_compiles() {
    // (3, 4) @ (4, 5) → (3, 5)，应编译通过
    let src = r#"
fn good() -> Tensor[f64, ..] {
    let a = zeros(3, 4);
    let b = zeros(4, 5);
    a.matmul(b)
}
"#;
    lower(src).expect("正确的 matmul shape 应编译通过");
}

#[test]
fn matmul_with_transpose_compiles() {
    // attention 模式：q (3,8) @ kT (8,3) → (3,3)
    let src = r#"
fn attention() -> Tensor[f64, ..] {
    let q = randn(3, 8);
    let k = randn(3, 8);
    let kT = k.transpose();
    q.matmul(kT)
}
"#;
    lower(src).expect("q @ k.transpose() 应编译通过");
}

#[test]
fn matmul_transpose_mismatch_reports_error() {
    // 忘记 transpose：q (3,8) @ k (3,8)，内侧 8 ≠ 3
    let src = r#"
fn bad_attention() -> Tensor[f64, ..] {
    let q = randn(3, 8);
    let k = randn(3, 8);
    q.matmul(k)
}
"#;
    assert_compile_error(src, "matmul shape 不兼容");
}

// ── 二元运算 broadcast shape 检查 ──────────────────────────────────────────

#[test]
fn binary_add_incompatible_shapes_reports_error() {
    // (3, 4) + (5, 6) — 无法广播
    let src = r#"
fn bad() -> Tensor[f64, ..] {
    let a = zeros(3, 4);
    let b = zeros(5, 6);
    a + b
}
"#;
    assert_compile_error(src, "shape 不兼容");
}

#[test]
fn binary_add_same_shape_compiles() {
    let src = r#"
fn good() -> Tensor[f64, ..] {
    let a = zeros(3, 4);
    let b = zeros(3, 4);
    a + b
}
"#;
    lower(src).expect("相同 shape 的加法应编译通过");
}

#[test]
fn binary_broadcast_scalar_compiles() {
    // (3, 4) + (1,) — 标量广播，应通过
    let src = r#"
fn good() -> Tensor[f64, ..] {
    let a = zeros(3, 4);
    let b = zeros(1);
    a + b
}
"#;
    lower(src).expect("标量广播应编译通过");
}

#[test]
fn binary_broadcast_row_compiles() {
    // (2, 3) + (1, 3) — 行广播，应通过
    let src = r#"
fn good() -> Tensor[f64, ..] {
    let a = zeros(2, 3);
    let b = zeros(1, 3);
    a + b
}
"#;
    lower(src).expect("行广播应编译通过");
}

// ── 字面量构造函数 shape 推断 ──────────────────────────────────────────────

#[test]
fn zeros_literal_shape_inferred() {
    // zeros(3, 4) 应被推断为 Tensor[f64, 3, 4]
    let src = r#"
fn make() -> Tensor[f64, 3, 4] {
    zeros(3, 4)
}
"#;
    let mut lexer = Lexer::new(src);
    let tokens = lexer.tokenize().unwrap();
    let mut parser = Parser::new(tokens);
    let program = parser.parse_program().unwrap();
    let mut lowerer = Lowerer::new();
    let hir = lowerer.lower_program(&program).expect("lower");
    let make = hir.functions.iter().find(|f| f.name == "make").unwrap();
    // 返回类型应为 Tensor[f64, 3, 4]
    let ret_str = format!("{}", make.return_type);
    assert!(
        ret_str.contains("3") && ret_str.contains("4"),
        "zeros(3, 4) 返回类型应含 3 和 4，实际: {}", ret_str
    );
}

#[test]
fn randn_literal_shape_inferred() {
    // randn(2, 5) 应被推断为 Tensor[f64, 2, 5]
    let src = r#"
fn make() -> Tensor[f64, ..] {
    randn(2, 5)
}
"#;
    let mut lexer = Lexer::new(src);
    let tokens = lexer.tokenize().unwrap();
    let mut parser = Parser::new(tokens);
    let program = parser.parse_program().unwrap();
    let mut lowerer = Lowerer::new();
    let hir = lowerer.lower_program(&program).expect("lower");
    let make = hir.functions.iter().find(|f| f.name == "make").unwrap();
    let body_str = format!("{:?}", make.body.kind);
    // randn(2, 5) 的返回类型应含 Known(2) 和 Known(5)
    assert!(
        body_str.contains("Known(2)") && body_str.contains("Known(5)"),
        "randn(2, 5) 应推断 shape [2, 5]，body: {}", body_str
    );
}

#[test]
fn dynamic_shape_arg_returns_any() {
    // zeros(n) — 参数非字面量，shape 应为 [Any]
    let src = r#"
fn make(n: i64) -> Tensor[f64, ..] {
    zeros(n)
}
"#;
    let mut lexer = Lexer::new(src);
    let tokens = lexer.tokenize().unwrap();
    let mut parser = Parser::new(tokens);
    let program = parser.parse_program().unwrap();
    let mut lowerer = Lowerer::new();
    let hir = lowerer.lower_program(&program).expect("lower");
    let make = hir.functions.iter().find(|f| f.name == "make").unwrap();
    let body_str = format!("{:?}", make.body.kind);
    // 应包含 Any，不应包含 Known(n)
    assert!(
        body_str.contains("Any"),
        "zeros(n) 应返回 [Any]（动态 shape），body: {}", body_str
    );
}

// ── transpose shape 反转推断 ───────────────────────────────────────────────

#[test]
fn transpose_2d_shape_reversed() {
    // (3, 8).transpose() → (8, 3)
    let src = r#"
fn xpose() -> Tensor[f64, ..] {
    let a = zeros(3, 8);
    a.transpose()
}
"#;
    let mut lexer = Lexer::new(src);
    let tokens = lexer.tokenize().unwrap();
    let mut parser = Parser::new(tokens);
    let program = parser.parse_program().unwrap();
    let mut lowerer = Lowerer::new();
    let hir = lowerer.lower_program(&program).expect("lower");
    let xpose = hir.functions.iter().find(|f| f.name == "xpose").unwrap();
    let body_str = format!("{:?}", xpose.body.kind);
    // transpose 结果应含 Known(8) 和 Known(3)（顺序反转）
    assert!(
        body_str.contains("Known(8)") && body_str.contains("Known(3)"),
        "transpose 应反转 shape 为 [8, 3]，body: {}", body_str
    );
}
