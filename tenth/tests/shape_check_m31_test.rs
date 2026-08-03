//! M3.1 shape 参数一致化（战略方向 B）测试。
//!
//! 护城河：编译期抓住 shape 错误 = 静默错值防线。
//!
//! 覆盖：
//! 1. **param() 类型直通**：param(x) 恒等（注册 tape input 后原样返回），
//!    param(zeros(3,4)) 保持 Tensor[f64, 3, 4]，matmul/广播编译期检查
//!    对 param 包裹的张量恢复生效（此前 types.rs 注册返回 Tensor[..] = [Any]，
//!    shape 被清零漏到运行时）。
//! 2. **tensor[[...]] 字面量编译期 shape 追踪**：字面量结构静态可知，
//!    matmul/广播检查对字面量张量生效（此前 lower 未直通，shape 漏到运行时）。
//! 3. **不规则字面量**（行长度不一致）编译期报错；**嵌套 3D 字面量**保守退化
//!    [Any]（不误报 2D，运行时仍可用）。
//! 4. **matmul FLOPs ×2 因子修正**（每乘加 = 1 mul + 1 add = 2 FLOP，与 bmm 口径一致）。

use tenth::lexer::lexer::Lexer;
use tenth::parser::parser::Parser;
use tenth::hir::lower::Lowerer;
use tenth::error::{TenthError, TenthWarning};

/// 辅助：lower 源码，返回 warnings（成功）或错误。
fn lower(src: &str) -> Result<Vec<TenthWarning>, TenthError> {
    let mut lexer = Lexer::new(src);
    let tokens = lexer.tokenize().expect("lex");
    let mut parser = Parser::new(tokens);
    let program = parser.parse_program().expect("parse");
    let mut lowerer = Lowerer::new();
    let hir = lowerer.lower_program(&program)?;
    Ok(hir.warnings)
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

// ── 1. param() 类型直通 ─────────────────────────────────────────────────

#[test]
fn param_wrapped_matmul_mismatch_reports_compile_error() {
    // param(zeros(3,4)) 应保持 [3,4]，@ zeros(5,4) 内侧 K=4≠5 → 编译期拦截
    let src = r#"
fn bad() -> Tensor[f64, ..] {
    let w = param(zeros(3, 4));
    let x = zeros(5, 4);
    w.matmul(x)
}
"#;
    assert_compile_error(src, "matmul shape 不兼容");
}

#[test]
fn param_wrapped_matmul_correct_compiles() {
    // param(zeros(3,4)) @ zeros(4,5) → 合法，编译通过
    let src = r#"
fn good() -> Tensor[f64, ..] {
    let w = param(zeros(3, 4));
    let x = zeros(4, 5);
    w.matmul(x)
}
"#;
    lower(src).expect("param 直通后的合法 matmul 应编译通过");
}

#[test]
fn param_wrapped_broadcast_mismatch_reports_compile_error() {
    // param(zeros(2,2)) + zeros(1,3) → 无法广播 → 编译期拦截
    let src = r#"
fn bad() -> Tensor[f64, ..] {
    let w = param(zeros(2, 2));
    let b = zeros(1, 3);
    w + b
}
"#;
    assert_compile_error(src, "无法广播");
}

#[test]
fn param_passthrough_keeps_shape() {
    // param(zeros(3,4)) 返回类型应为 Tensor[f64, 3, 4]（直通，非 [Any]）
    let src = r#"
fn make() -> Tensor[f64, ..] {
    param(zeros(3, 4))
}
"#;
    let mut lexer = Lexer::new(src);
    let tokens = lexer.tokenize().unwrap();
    let mut parser = Parser::new(tokens);
    let program = parser.parse_program().unwrap();
    let mut lowerer = Lowerer::new();
    let hir = lowerer.lower_program(&program).expect("lower");
    let make = hir.functions.iter().find(|f| f.name == "make").unwrap();
    let ret_str = format!("{}", make.return_type);
    assert!(
        ret_str.contains("3") && ret_str.contains("4") && !ret_str.contains(".."),
        "param(zeros(3,4)) 返回类型应直通为 [3,4]，实际: {}", ret_str
    );
}

// ── 2. tensor 字面量 shape 追踪 ─────────────────────────────────────────

#[test]
fn tensor_literal_matmul_mismatch_reports_compile_error() {
    // 2×3 @ 2×2 → 内侧 K=3≠2 → 编译期拦截
    let src = r#"
fn bad() -> Tensor[f64, ..] {
    let a = tensor[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
    let b = tensor[[1.0, 2.0], [3.0, 4.0]];
    a.matmul(b)
}
"#;
    assert_compile_error(src, "matmul shape 不兼容");
}

#[test]
fn tensor_literal_matmul_correct_compiles() {
    // 2×3 @ 3×2 → 内侧 K=3=3 → 编译通过
    let src = r#"
fn good() -> Tensor[f64, ..] {
    let a = tensor[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
    let b = tensor[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
    a.matmul(b)
}
"#;
    lower(src).expect("字面量合法 matmul 应编译通过");
}

#[test]
fn tensor_literal_broadcast_mismatch_reports_compile_error() {
    // 2×2 + 1×3 → 无法广播 → 编译期拦截
    let src = r#"
fn bad() -> Tensor[f64, ..] {
    let a = tensor[[1.0, 2.0], [3.0, 4.0]];
    let b = tensor[[1.0, 2.0, 3.0]];
    a + b
}
"#;
    assert_compile_error(src, "无法广播");
}

#[test]
fn tensor_literal_broadcast_correct_compiles() {
    // 2×2 + 1×2 → 行广播 → 编译通过
    let src = r#"
fn good() -> Tensor[f64, ..] {
    let a = tensor[[1.0, 2.0], [3.0, 4.0]];
    let b = tensor[[10.0, 20.0]];
    a + b
}
"#;
    lower(src).expect("字面量行广播应编译通过");
}

#[test]
fn tensor_literal_irregular_rows_reports_compile_error() {
    // 行长度不一致（第 1 行 2 个、第 2 行 3 个）→ 编译期拦截（结构静态可知）
    let src = r#"
fn bad() -> Tensor[f64, ..] {
    let a = tensor[[1.0, 2.0], [3.0, 4.0, 5.0]];
    a
}
"#;
    assert_compile_error(src, "行长度不一致");
}

#[test]
fn tensor_literal_nested_3d_compiles() {
    // 嵌套 3D 字面量：保守退化 [Any]（不误报 2D），编译通过（运行时可用）
    let src = r#"
fn nested() -> Tensor[f64, ..] {
    let a = tensor[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]];
    a
}
"#;
    lower(src).expect("嵌套 3D 字面量应编译通过（保守退化 [Any]）");
}

// ── 3. matmul FLOPs ×2 因子修正 ─────────────────────────────────────────

#[test]
fn matmul_flops_has_x2_factor() {
    // 4000×4000 @ 4000×4000：乘加 = 6.4e10，×2 FLOP = 1.28e11 = 128.00 GFLOP
    // 若缺 ×2 会报 64.00 GFLOP（盲区：此前报 8 GFLOP 实为 16）
    let src = r#"
fn big() -> Tensor[f64, ..] {
    let a = zeros(4000, 4000);
    let b = zeros(4000, 4000);
    a.matmul(b)
}
"#;
    let warnings = lower(src).expect("FLOPs 预估测试源码应 lower 成功");
    let found = warnings.iter().any(|w| w.message.contains("128.00 GFLOPs"));
    assert!(
        found,
        "matmul FLOPs 应含 ×2 因子（128.00 GFLOPs）\n实际 warnings: {:?}",
        warnings.iter().map(|w| &w.message).collect::<Vec<_>>()
    );
}
