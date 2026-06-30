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

// ── Phase 2: let 传播 ──────────────────────────────────────────────────────

#[test]
fn let_propagates_shape_to_next_expr() {
    // let a = zeros(3, 4); a 的 shape [3,4] 应传播到后续使用
    let src = r#"
fn good() -> Tensor[f64, ..] {
    let a = zeros(3, 4);
    let b = zeros(4, 5);
    a.matmul(b)
}
"#;
    lower(src).expect("let 传播的 shape 应让 matmul 编译通过");
}

#[test]
fn let_propagates_shape_mismatch_reports_error() {
    // let a = zeros(3, 4); let b = zeros(5, 6); a.matmul(b) 应报错
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
fn let_alias_preserves_shape() {
    // let a = zeros(3, 4); let c = a; c 应继承 [3, 4]
    let src = r#"
fn good() -> Tensor[f64, ..] {
    let a = zeros(3, 4);
    let c = a;
    let b = zeros(4, 5);
    c.matmul(b)
}
"#;
    lower(src).expect("let c = a 应继承 shape [3,4]，matmul 应编译通过");
}

// ── Phase 2: 函数参数 shape 约束 ───────────────────────────────────────────

#[test]
fn fn_param_shape_constraint_checked() {
    // 函数签名声明 x: Tensor[f64, 3, 4]，函数体内 x.matmul(zeros(5,6)) 应报错
    let src = r#"
fn bad(x: Tensor[f64, 3, 4]) -> Tensor[f64, ..] {
    let b = zeros(5, 6);
    x.matmul(b)
}
"#;
    assert_compile_error(src, "matmul shape 不兼容");
}

#[test]
fn fn_param_shape_constraint_passes() {
    // 函数签名声明 x: Tensor[f64, 3, 4]，函数体内 x.matmul(zeros(4,5)) 应通过
    let src = r#"
fn good(x: Tensor[f64, 3, 4]) -> Tensor[f64, ..] {
    let b = zeros(4, 5);
    x.matmul(b)
}
"#;
    lower(src).expect("函数参数 shape 约束应让 matmul 编译通过");
}

// ── Phase 2: 符号维度（同名 Symbol 等价） ──────────────────────────────────

#[test]
fn matmul_symbol_dims_same_name_compiles() {
    // M, K, N 为符号维度；a: [M, K] @ b: [K, N] → [M, N]，K 同名应通过
    let src = r#"
fn matmul_fn(a: Tensor[f64, M, K], b: Tensor[f64, K, N]) -> Tensor[f64, ..] {
    a.matmul(b)
}
"#;
    lower(src).expect("符号维度同名 K 应编译通过");
}

#[test]
fn matmul_symbol_dims_different_names_reports_error() {
    // a: [M, K] @ b: [P, N]，K ≠ P 应报错（符号维度不同名视为不匹配）
    let src = r#"
fn bad(a: Tensor[f64, M, K], b: Tensor[f64, P, N]) -> Tensor[f64, ..] {
    a.matmul(b)
}
"#;
    assert_compile_error(src, "matmul shape 不兼容");
}

// ── Phase 2: 归约算子 axis 降维 ────────────────────────────────────────────

#[test]
fn sum_with_literal_axis_removes_dim() {
    // (3, 4).sum(0) → [4]
    let src = r#"
fn reduce() -> Tensor[f64, ..] {
    let a = zeros(3, 4);
    a.sum(0)
}
"#;
    let mut lexer = Lexer::new(src);
    let tokens = lexer.tokenize().unwrap();
    let mut parser = Parser::new(tokens);
    let program = parser.parse_program().unwrap();
    let mut lowerer = Lowerer::new();
    let hir = lowerer.lower_program(&program).expect("lower");
    let reduce = hir.functions.iter().find(|f| f.name == "reduce").unwrap();
    let ty_str = format!("{:?}", reduce.body.ty);
    assert!(
        ty_str.contains("Known(4)") && !ty_str.contains("Known(3)"),
        "sum(0) 应移除第 0 维，结果 shape 应含 Known(4) 且不含 Known(3)，ty: {}",
        ty_str
    );
}

#[test]
fn sum_with_literal_axis_1_removes_second_dim() {
    // (3, 4).sum(1) → [3]
    let src = r#"
fn reduce() -> Tensor[f64, ..] {
    let a = zeros(3, 4);
    a.sum(1)
}
"#;
    let mut lexer = Lexer::new(src);
    let tokens = lexer.tokenize().unwrap();
    let mut parser = Parser::new(tokens);
    let program = parser.parse_program().unwrap();
    let mut lowerer = Lowerer::new();
    let hir = lowerer.lower_program(&program).expect("lower");
    let reduce = hir.functions.iter().find(|f| f.name == "reduce").unwrap();
    let ty_str = format!("{:?}", reduce.body.ty);
    assert!(
        ty_str.contains("Known(3)") && !ty_str.contains("Known(4)"),
        "sum(1) 应移除第 1 维，结果 shape 应含 Known(3) 且不含 Known(4)，ty: {}",
        ty_str
    );
}

#[test]
fn sum_no_args_returns_scalar() {
    // 无参数 sum() 全部降维到标量
    let src = r#"
fn reduce() -> f64 {
    let a = zeros(3, 4);
    a.sum()
}
"#;
    let mut lexer = Lexer::new(src);
    let tokens = lexer.tokenize().unwrap();
    let mut parser = Parser::new(tokens);
    let program = parser.parse_program().unwrap();
    let mut lowerer = Lowerer::new();
    let hir = lowerer.lower_program(&program).expect("lower");
    let reduce = hir.functions.iter().find(|f| f.name == "reduce").unwrap();
    let ty_str = format!("{:?}", reduce.body.ty);
    // 应返回 Base(F64)，不应含 Known
    assert!(
        !ty_str.contains("Known(3)") && !ty_str.contains("Known(4)"),
        "sum() 无参数应返回标量，不应含 Known(3)/Known(4)，ty: {}", ty_str
    );
}

#[test]
fn mean_with_literal_axis_removes_dim() {
    // (3, 4).mean(0) → [4]，验证 mean 同 sum 的 axis 逻辑
    let src = r#"
fn reduce() -> Tensor[f64, ..] {
    let a = zeros(3, 4);
    a.mean(0)
}
"#;
    let mut lexer = Lexer::new(src);
    let tokens = lexer.tokenize().unwrap();
    let mut parser = Parser::new(tokens);
    let program = parser.parse_program().unwrap();
    let mut lowerer = Lowerer::new();
    let hir = lowerer.lower_program(&program).expect("lower");
    let reduce = hir.functions.iter().find(|f| f.name == "reduce").unwrap();
    let ty_str = format!("{:?}", reduce.body.ty);
    assert!(
        ty_str.contains("Known(4)"),
        "mean(0) 应移除第 0 维，结果 shape 应含 Known(4)，ty: {}", ty_str
    );
}

// ── Phase 2: reshape 字面量参数推断 ────────────────────────────────────────

#[test]
fn reshape_literal_args_inferred() {
    // x.reshape(3, 4) 应推断为新 shape [3, 4]
    let src = r#"
fn reshape_fn() -> Tensor[f64, ..] {
    let a = zeros(2, 6);
    a.reshape(3, 4)
}
"#;
    let mut lexer = Lexer::new(src);
    let tokens = lexer.tokenize().unwrap();
    let mut parser = Parser::new(tokens);
    let program = parser.parse_program().unwrap();
    let mut lowerer = Lowerer::new();
    let hir = lowerer.lower_program(&program).expect("lower");
    let reshape_fn = hir.functions.iter().find(|f| f.name == "reshape_fn").unwrap();
    let ty_str = format!("{:?}", reshape_fn.body.ty);
    assert!(
        ty_str.contains("Known(3)") && ty_str.contains("Known(4)"),
        "reshape(3, 4) 应推断为新 shape [3, 4]，ty: {}", ty_str
    );
}

#[test]
fn reshape_dynamic_arg_returns_any() {
    // x.reshape(n, m) — 参数非字面量，shape 应为 [Any]
    let src = r#"
fn reshape_fn(n: i64, m: i64) -> Tensor[f64, ..] {
    let a = zeros(2, 6);
    a.reshape(n, m)
}
"#;
    let mut lexer = Lexer::new(src);
    let tokens = lexer.tokenize().unwrap();
    let mut parser = Parser::new(tokens);
    let program = parser.parse_program().unwrap();
    let mut lowerer = Lowerer::new();
    let hir = lowerer.lower_program(&program).expect("lower");
    let reshape_fn = hir.functions.iter().find(|f| f.name == "reshape_fn").unwrap();
    let ty_str = format!("{:?}", reshape_fn.body.ty);
    assert!(
        ty_str.contains("Any"),
        "reshape(n, m) 应返回 [Any]（动态 shape），ty: {}", ty_str
    );
}

// ── Phase 3: 算子覆盖扩展 ──────────────────────────────────────────────────

#[test]
fn permute_literal_dims_reorders_shape() {
    // (3, 8, 5).permute(2, 0, 1) → [5, 3, 8]
    let src = r#"
fn permute_fn() -> Tensor[f64, ..] {
    let a = zeros(3, 8, 5);
    a.permute(2, 0, 1)
}
"#;
    let mut lexer = Lexer::new(src);
    let tokens = lexer.tokenize().unwrap();
    let mut parser = Parser::new(tokens);
    let program = parser.parse_program().unwrap();
    let mut lowerer = Lowerer::new();
    let hir = lowerer.lower_program(&program).expect("lower");
    let f = hir.functions.iter().find(|f| f.name == "permute_fn").unwrap();
    let ty_str = format!("{:?}", f.body.ty);
    // 应按 [2, 0, 1] 索引重排 [3, 8, 5] → [5, 3, 8]
    assert!(
        ty_str.contains("Known(5)") && ty_str.contains("Known(3)") && ty_str.contains("Known(8)"),
        "permute(2,0,1) 应重排 [3,8,5]→[5,3,8]，ty: {}", ty_str
    );
}

#[test]
fn broadcast_to_literal_args_inferred() {
    // (1, 3).broadcast_to(4, 3) → [4, 3]
    let src = r#"
fn bcast_fn() -> Tensor[f64, ..] {
    let a = zeros(1, 3);
    a.broadcast_to(4, 3)
}
"#;
    let mut lexer = Lexer::new(src);
    let tokens = lexer.tokenize().unwrap();
    let mut parser = Parser::new(tokens);
    let program = parser.parse_program().unwrap();
    let mut lowerer = Lowerer::new();
    let hir = lowerer.lower_program(&program).expect("lower");
    let f = hir.functions.iter().find(|f| f.name == "bcast_fn").unwrap();
    let ty_str = format!("{:?}", f.body.ty);
    assert!(
        ty_str.contains("Known(4)") && ty_str.contains("Known(3)"),
        "broadcast_to(4, 3) 应推断为 [4, 3]，ty: {}", ty_str
    );
}

#[test]
fn cat_with_literal_dim_sums_dim() {
    // (2, 3).cat((3, 3), dim=0) → [5, 3]
    let src = r#"
fn cat_fn() -> Tensor[f64, ..] {
    let a = zeros(2, 3);
    let b = zeros(3, 3);
    a.cat(b, 0)
}
"#;
    let mut lexer = Lexer::new(src);
    let tokens = lexer.tokenize().unwrap();
    let mut parser = Parser::new(tokens);
    let program = parser.parse_program().unwrap();
    let mut lowerer = Lowerer::new();
    let hir = lowerer.lower_program(&program).expect("lower");
    let f = hir.functions.iter().find(|f| f.name == "cat_fn").unwrap();
    let ty_str = format!("{:?}", f.body.ty);
    // dim 0 应相加 2+3=5，dim 1 保持 3
    assert!(
        ty_str.contains("Known(5)") && ty_str.contains("Known(3)") && !ty_str.contains("Known(2)"),
        "cat(dim=0) 应相加 dim 0：[2,3]+[3,3]→[5,3]，ty: {}", ty_str
    );
}

#[test]
fn cat_dim_1_sums_second_dim() {
    // (2, 3).cat((2, 4), dim=1) → [2, 7]
    let src = r#"
fn cat_fn() -> Tensor[f64, ..] {
    let a = zeros(2, 3);
    let b = zeros(2, 4);
    a.cat(b, 1)
}
"#;
    let mut lexer = Lexer::new(src);
    let tokens = lexer.tokenize().unwrap();
    let mut parser = Parser::new(tokens);
    let program = parser.parse_program().unwrap();
    let mut lowerer = Lowerer::new();
    let hir = lowerer.lower_program(&program).expect("lower");
    let f = hir.functions.iter().find(|f| f.name == "cat_fn").unwrap();
    let ty_str = format!("{:?}", f.body.ty);
    // dim 1 应相加 3+4=7，dim 0 保持 2
    assert!(
        ty_str.contains("Known(2)") && ty_str.contains("Known(7)"),
        "cat(dim=1) 应相加 dim 1：[2,3]+[2,4]→[2,7]，ty: {}", ty_str
    );
}

#[test]
fn argmax_returns_i64_scalar() {
    // argmax() 返回 i64 标量
    let src = r#"
fn argmax_fn() -> i64 {
    let a = zeros(3, 4);
    a.argmax()
}
"#;
    let mut lexer = Lexer::new(src);
    let tokens = lexer.tokenize().unwrap();
    let mut parser = Parser::new(tokens);
    let program = parser.parse_program().unwrap();
    let mut lowerer = Lowerer::new();
    let hir = lowerer.lower_program(&program).expect("lower");
    let f = hir.functions.iter().find(|f| f.name == "argmax_fn").unwrap();
    let ty_str = format!("{:?}", f.body.ty);
    assert!(
        ty_str.contains("I64") && !ty_str.contains("Tensor"),
        "argmax() 应返回 i64 标量，ty: {}", ty_str
    );
}

#[test]
fn gelu_preserves_shape() {
    // gelu() 保持原 shape
    let src = r#"
fn gelu_fn() -> Tensor[f64, ..] {
    let a = zeros(3, 4);
    a.gelu()
}
"#;
    let mut lexer = Lexer::new(src);
    let tokens = lexer.tokenize().unwrap();
    let mut parser = Parser::new(tokens);
    let program = parser.parse_program().unwrap();
    let mut lowerer = Lowerer::new();
    let hir = lowerer.lower_program(&program).expect("lower");
    let f = hir.functions.iter().find(|f| f.name == "gelu_fn").unwrap();
    let ty_str = format!("{:?}", f.body.ty);
    assert!(
        ty_str.contains("Known(3)") && ty_str.contains("Known(4)"),
        "gelu() 应保持原 shape [3, 4]，ty: {}", ty_str
    );
}

#[test]
fn masked_fill_preserves_shape() {
    // masked_fill(mask, value) 保持原 shape
    let src = r#"
fn mfill_fn() -> Tensor[f64, ..] {
    let a = zeros(3, 4);
    let mask = zeros(3, 4);
    a.masked_fill(mask, 0.0)
}
"#;
    let mut lexer = Lexer::new(src);
    let tokens = lexer.tokenize().unwrap();
    let mut parser = Parser::new(tokens);
    let program = parser.parse_program().unwrap();
    let mut lowerer = Lowerer::new();
    let hir = lowerer.lower_program(&program).expect("lower");
    let f = hir.functions.iter().find(|f| f.name == "mfill_fn").unwrap();
    let ty_str = format!("{:?}", f.body.ty);
    assert!(
        ty_str.contains("Known(3)") && ty_str.contains("Known(4)"),
        "masked_fill() 应保持原 shape [3, 4]，ty: {}", ty_str
    );
}

#[test]
fn flatten_returns_1d() {
    // flatten() 返回 1D
    let src = r#"
fn flat_fn() -> Tensor[f64, ..] {
    let a = zeros(3, 4, 5);
    a.flatten()
}
"#;
    let mut lexer = Lexer::new(src);
    let tokens = lexer.tokenize().unwrap();
    let mut parser = Parser::new(tokens);
    let program = parser.parse_program().unwrap();
    let mut lowerer = Lowerer::new();
    let hir = lowerer.lower_program(&program).expect("lower");
    let f = hir.functions.iter().find(|f| f.name == "flat_fn").unwrap();
    let ty_str = format!("{:?}", f.body.ty);
    // 应为 1D（dims 长度为 1，含 Any）
    assert!(
        ty_str.contains("Any") && !ty_str.contains("Known(3)"),
        "flatten() 应返回 1D [Any]，不含 Known(3)，ty: {}", ty_str
    );
}

