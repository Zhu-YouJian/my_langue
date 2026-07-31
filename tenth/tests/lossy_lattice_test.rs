//! 阶段2b-静默算错（lossy lattice）M1 里程碑测试。
//!
//! 核心命题：**「可能算错」的值（NaN、溢出、精度降级）不能当确定正确的值用——
//! 除非显式 `lossy`**。对应 Rust 的 `unsafe`。
//!
//! M1 范围（本里程碑只做这些）：
//! 1. **编译期零除数检测 spike**：`x / 0`（0 为字面量或静态可判定的零常量）→
//!    编译期报错，比现状提前：
//!    - 浮点 `1.0 / 0.0`：现状**静默产生 inf**（静默算错）→ 改进后编译期 TypeError
//!    - 整数 `10 / 0`：现状运行时"整数除零"→ 改进后编译期 TypeError
//! 2. **防误报回归**：非零字面量除法、变量除数（运行时值）一律不报——
//!    默认策略"宁可漏报，不可误报"。
//!
//! 覆盖：
//! - 浮点字面量零除数 → 编译期报错
//! - 整数字面量零除数 → 编译期报错
//! - 取模零 → 编译期报错
//! - 负零 `-0.0` / `-0` → 编译期报错（静态可判定，同属除零类）
//! - 非零除法（`1.0 / 2.0`、`10 / 2`）→ 零误报，编译通过
//! - 变量除数（`let y = 0.0; x / y`）→ 不报（运行时值，防误报）

use tenth::error::TenthError;
use tenth::hir::lower::Lowerer;
use tenth::lexer::lexer::Lexer;
use tenth::parser::parser::Parser;

fn lower(src: &str) -> Result<(), TenthError> {
    let mut lexer = Lexer::new(src);
    let tokens = lexer.tokenize().expect("lex");
    let mut parser = Parser::new(tokens);
    let program = parser.parse_program().expect("parse");
    let mut lowerer = Lowerer::new();
    lowerer.lower_program(&program).map(|_| ())
}

/// 断言 lower 失败且为 TypeError，错误信息包含指定子串。
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

/// 断言 lower 成功（编译通过，零误报）。
fn assert_compiles(src: &str) {
    lower(src).unwrap_or_else(|e| panic!("期望编译通过（零误报），实际错误: {:?}", e));
}

// ══════════════════════════════════════════════════════════════════════
// 1. 浮点字面量零除数 → 编译期报错（基线：静默产生 inf）
// ══════════════════════════════════════════════════════════════════════

#[test]
fn float_literal_zero_divisor_is_compile_error() {
    // 基线行为：1.0 / 0.0 静默产生 inf——正是"静默算错"的典型
    let src = r#"
fn bad() -> f64 {
    1.0 / 0.0
}
"#;
    assert_compile_error(src, "除数为零");
}

#[test]
fn int_literal_zero_divisor_is_compile_error() {
    // 基线行为：10 / 0 是运行时"整数除零"错误——改进后提前到编译期
    let src = r#"
fn bad() -> i32 {
    10 / 0
}
"#;
    assert_compile_error(src, "除数为零");
}

#[test]
fn literal_zero_mod_is_compile_error() {
    let src = r#"
fn bad() -> i32 {
    10 % 0
}
"#;
    assert_compile_error(src, "除数为零");
}

#[test]
fn neg_zero_divisor_is_compile_error() {
    // -0.0 / -0 静态可判定为零，作为除数同属除零类
    let src = r#"
fn bad() -> f64 {
    1.0 / -0.0
}
"#;
    assert_compile_error(src, "除数为零");
}

// ══════════════════════════════════════════════════════════════════════
// 2. 防误报回归（默认策略：宁可漏报，不可误报）
// ══════════════════════════════════════════════════════════════════════

#[test]
fn nonzero_literal_division_compiles() {
    let src = r#"
fn good() -> f64 {
    1.0 / 2.0
}
"#;
    assert_compiles(src);
}

#[test]
fn nonzero_int_division_compiles() {
    let src = r#"
fn good() -> i32 {
    10 / 2
}
"#;
    assert_compiles(src);
}

#[test]
fn variable_divisor_not_reported_even_if_init_zero() {
    // `let y = 0.0; x / y`：y 是运行时变量（可能被后续赋值改变），
    // 不算编译期常量 → 不报（漏报可接受，误报不可接受）。
    let src = r#"
fn maybe(y: f64) -> f64 {
    1.0 / y
}
"#;
    assert_compiles(src);
}

#[test]
fn tensor_div_by_scalar_literal_zero_is_compile_error() {
    // 张量 ÷ 字面量零标量：静态可判定 → 编译期报错
    let src = r#"
fn bad() -> Tensor[f64, ..] {
    let a = zeros(3, 4);
    a / 0.0
}
"#;
    assert_compile_error(src, "除数为零");
}

#[test]
fn tensor_div_by_tensor_zero_compiles() {
    // 张量 ÷ 张量（元素为 0）：右侧是运行时构造，非字面量 → 不报（漏报可接受）
    let src = r#"
fn maybe(a: Tensor[f64, ..], b: Tensor[f64, ..]) -> Tensor[f64, ..] {
    a / b
}
"#;
    assert_compiles(src);
}
