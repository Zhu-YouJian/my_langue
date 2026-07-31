//! 阶段2b-静默算错（lossy lattice）M3 里程碑测试：污点传播与 shape/静态维度/常量信息协同。
//!
//! M3 范围（`hir/lower/types.rs` + `hir/lower/taint.rs`）：
//! 1. **除零精确化（正向豁免）**：除数**静态非零**（非零字面量 `2.0`/`1e3`、张量字面量
//!    全非零、`ones*` 构造）→ 不标 PossibleNaN，零误报。
//! 2. **静态零拦截扩展（M1 升级到张量级）**：除数静态可判定为零扩展到张量——
//!    `zeros(...)`/`zeros_f16(...)` 构造、张量字面量全零 → 编译期硬错误（原来漏报）。
//! 3. **shape 协同核验**：shape/维度静态已知（`zeros(3,4)` Known dims、`Tensor[f64,M,K]`
//!    符号维度）的运算，污点只依赖 dtype——shape 已知**不改变 dtype 结论**；
//!    shape 已知但值未知的除数 → 不 speculate（防误报）。
//! 4. **回归守护**：M1 零除数硬错误、M2 Lossy 使用点检查与 `lossy(...)` 放行不回归。
//!
//! 防误报底线：任何改动不得让现有科学计算/标准库代码大面积新报错。
//! 部分零张量（`tensor[[0.0, 1.0]]`）不拦截（张量级粒度，逐元素粒度过度，漏报接受）。

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
// 1. 除零精确化（正向豁免）：静态非零除数 → 不标 PossibleNaN，零误报
// ══════════════════════════════════════════════════════════════════════

#[test]
fn static_nonzero_literal_divisor_no_false_positive() {
    // 字面量非零除数（含 1e3 科学计数法）：结果 Exact
    let src = r#"
fn good() -> f64 {
    1.0 / 2.0
}
fn good2() -> f64 {
    1e3 / 1e3
}
"#;
    assert_compiles(src);
}

#[test]
fn constant_nonzero_divisor_var_dividend_no_false_positive() {
    // 分子是变量（值未知）、除数是常量非零 2.0：除数静态非零 → 不标 PossibleNaN
    let src = r#"
fn f(x: f64) -> f64 {
    x / 2.0
}
"#;
    assert_compiles(src);
}

#[test]
fn negative_constant_divisor_no_false_positive() {
    // 负的常量非零除数（一元负号包裹）
    let src = r#"
fn f(x: f64) -> f64 {
    x / -3.0
}
"#;
    assert_compiles(src);
}

#[test]
fn tensor_literal_all_nonzero_divisor_no_false_positive() {
    // 张量字面量除数全非零（shape 已知 + 值静态非零）→ 不标 PossibleNaN
    let src = r#"
fn f(a: Tensor[f64, ..]) -> Tensor[f64, ..] {
    a / tensor[[1.0, 2.0], [3.0, 4.0]]
}
"#;
    assert_compiles(src);
}

#[test]
fn ones_ctor_divisor_no_false_positive() {
    // ones 构造（内置全一张量）→ 全非零 → 不标 PossibleNaN
    let src = r#"
fn f(a: Tensor[f64, ..]) -> Tensor[f64, ..] {
    a / ones(3, 4)
}
"#;
    assert_compiles(src);
}

#[test]
fn constant_divisor_sink_no_false_positive() {
    // 常量非零除数的结果送到使用点（to_string）：污点 Exact → 不报
    let src = r#"
fn f(x: f64) -> str {
    to_string(x / 2.0)
}
"#;
    assert_compiles(src);
}

// ══════════════════════════════════════════════════════════════════════
// 2. 静态零拦截扩展（M1 升级到张量级）：zeros 构造 / 张量字面量全零
// ══════════════════════════════════════════════════════════════════════

#[test]
fn zeros_ctor_divisor_is_compile_error() {
    // M3 新行为：除数 `zeros(3,4)` 是内置全零构造（shape/值均静态已知）→ 张量级判零。
    // 基线（M2 及以前）：`a / zeros(3,4)` 静默产生 inf 元素（静默算错）。
    let src = r#"
fn bad() -> Tensor[f64, ..] {
    let a = zeros(3, 4);
    a / zeros(3, 4)
}
"#;
    assert_compile_error(src, "除数为零");
}

#[test]
fn zeros_f16_ctor_divisor_is_compile_error() {
    // f16 全零构造除数（dtype 不同但同样全零）→ 张量级判零
    let src = r#"
fn bad() -> Tensor[f64, ..] {
    let a = zeros(3, 4);
    a / zeros_f16(2, 2)
}
"#;
    assert_compile_error(src, "除数为零");
}

#[test]
fn tensor_literal_all_zero_divisor_is_compile_error() {
    // 张量字面量全零（shape/值均静态已知）→ 张量级判零
    let src = r#"
fn bad() -> Tensor[f64, ..] {
    let a = zeros(2, 2);
    a / tensor[[0.0, 0.0], [0.0, 0.0]]
}
"#;
    assert_compile_error(src, "除数为零");
}

#[test]
fn scalar_literal_zero_divisor_still_compile_error() {
    // M1 回归：标量字面量零除数仍是硬错误
    let src = r#"
fn bad() -> f64 {
    1.0 / 0.0
}
"#;
    assert_compile_error(src, "除数为零");
}

// ══════════════════════════════════════════════════════════════════════
// 3. shape 协同：shape 已知不改变 dtype 结论；值未知的除数不 speculate
// ══════════════════════════════════════════════════════════════════════

#[test]
fn partial_zero_tensor_divisor_not_reported() {
    // 部分零张量（`tensor[[0.0, 1.0]]`）：张量级粒度下不判定（逐元素粒度过度，
    // 漏报接受——宁可漏报，不可误报）
    let src = r#"
fn f(a: Tensor[f64, ..]) -> Tensor[f64, ..] {
    a / tensor[[0.0, 1.0]]
}
"#;
    assert_compiles(src);
}

#[test]
fn symbol_dim_unknown_value_divisor_not_speculated() {
    // shape 已知（符号维度 M,K）但值未知的张量除数 → 不 speculate（防误报）
    let src = r#"
fn f(a: Tensor[f64, M, K], b: Tensor[f64, M, K]) -> Tensor[f64, M, K] {
    a / b
}
"#;
    assert_compiles(src);
}

#[test]
fn known_dim_unknown_value_divisor_not_speculated() {
    // shape 已知（Known dims 注解）但值未知的张量除数 → 不 speculate
    let src = r#"
fn f(a: Tensor[f64, 3, 4], b: Tensor[f64, 3, 4]) -> Tensor[f64, 3, 4] {
    a / b
}
"#;
    assert_compiles(src);
}

#[test]
fn shape_known_does_not_change_dtype_conclusion() {
    // shape 已知（zeros(3,4) Known dims）：f64 张量 × f64 标量无收缩 → Exact
    // （shape 已知不改变 dtype 结论——核实现状，避免重复标记）
    let src = r#"
fn ok() -> str {
    to_string(zeros(3, 4) * 1.5)
}
"#;
    assert_compiles(src);
}

#[test]
fn shape_known_still_lossy_when_dtype_contracts() {
    // shape 已知（zeros_f16(2,2)）但 dtype 收缩仍在（f16 张量 × f64 标量）→ Lossy。
    // 证明 shape 已知**不豁免** Lossy（污点只依赖 dtype），shape 协同不丢既有检查
    let src = r#"
fn bad() -> str {
    to_string(zeros_f16(2, 2) * 1.5)
}
"#;
    assert_compile_error(src, "lossy 污点");
}

#[test]
fn shape_known_lossy_accept_with_lossy() {
    // 同上但 lossy(...) 放行
    let src = r#"
fn ok() -> str {
    to_string(lossy(zeros_f16(2, 2) * 1.5))
}
"#;
    assert_compiles(src);
}

// ══════════════════════════════════════════════════════════════════════
// 4. M2 Lossy 使用点检查回归（不受 M3 影响）
// ══════════════════════════════════════════════════════════════════════

#[test]
fn m2_lossy_sink_check_not_regressed() {
    // M2 回归：隐式标量→张量 dtype 收缩仍在使用点报错
    let src = r#"
fn bad() -> str {
    let t = zeros_f16(2, 2);
    let x = t * 1.23456789012345;
    to_string(x)
}
"#;
    assert_compile_error(src, "lossy 污点");
}

#[test]
fn m2_lossy_accept_not_regressed() {
    // M2 回归：lossy(...) 显式接受放行
    let src = r#"
fn ok() -> str {
    let t = zeros_f16(2, 2);
    let x = t * 1.23456789012345;
    to_string(lossy(x))
}
"#;
    assert_compiles(src);
}

#[test]
fn m2_cross_function_not_regressed() {
    // M2 回归：跨函数污点传播仍生效
    let src = r#"
fn make_lossy() -> Tensor[f16, ..] {
    let t = zeros_f16(2, 2);
    t * 1.23456789012345
}
fn bad() -> str {
    to_string(make_lossy())
}
"#;
    assert_compile_error(src, "lossy 污点");
}
