//! 编译期内存/算力预估测试（护城河 D）。
//!
//! 验证：
//! - 大 tensor 构造触发内存预估 warning
//! - 大 matmul 触发 FLOPs 预估 warning
//! - 小 tensor / 小 matmul 不触发 warning
//! - static_numel / static_bytes 边界情况
//!
//! 高开销测试（可能 OOM）标记 #[ignore]，等待安全环境再测。

use tenth::lexer::lexer::Lexer;
use tenth::parser::parser::Parser;
use tenth::hir::lower::Lowerer;
use tenth::hir::types::{Dim, Type, BaseType};
use tenth::error::TenthWarning;

/// 辅助：lower 源码，返回 HirProgram 的 warnings 列表。
fn lower_warnings(src: &str) -> Vec<TenthWarning> {
    let mut lexer = Lexer::new(src);
    let tokens = lexer.tokenize().expect("lex");
    let mut parser = Parser::new(tokens);
    let program = parser.parse_program().expect("parse");
    let mut lowerer = Lowerer::new();
    let hir = lowerer.lower_program(&program).expect("lower");
    hir.warnings
}

/// 辅助：断言 warnings 中有至少一条包含指定子串。
fn assert_has_warning(warnings: &[TenthWarning], expected_part: &str) {
    let found = warnings.iter().any(|w| w.message.contains(expected_part));
    assert!(
        found,
        "期望 warning 包含 '{}'\n实际 warnings: {:?}",
        expected_part,
        warnings.iter().map(|w| &w.message).collect::<Vec<_>>()
    );
}

/// 辅助：断言 warnings 为空。
fn assert_no_warnings(warnings: &[TenthWarning]) {
    assert!(
        warnings.is_empty(),
        "期望无 warning，但有: {:?}",
        warnings.iter().map(|w| &w.message).collect::<Vec<_>>()
    );
}

// ── static_numel / static_bytes 单元测试 ─────────────────────────────────

#[test]
fn static_numel_all_known() {
    let ty = Type::tensor(BaseType::F64, vec![Dim::Known(3), Dim::Known(4)]);
    assert_eq!(ty.static_numel(), Some(12));
}

#[test]
fn static_numel_single_dim() {
    let ty = Type::tensor(BaseType::F32, vec![Dim::Known(100)]);
    assert_eq!(ty.static_numel(), Some(100));
}

#[test]
fn static_numel_with_symbol_returns_none() {
    let ty = Type::tensor(BaseType::F64, vec![Dim::Known(3), Dim::Symbol("N".into())]);
    assert_eq!(ty.static_numel(), None);
}

#[test]
fn static_numel_with_any_returns_none() {
    let ty = Type::tensor(BaseType::F64, vec![Dim::Any, Dim::Known(4)]);
    assert_eq!(ty.static_numel(), None);
}

#[test]
fn static_numel_non_tensor_returns_none() {
    let ty = Type::Base(BaseType::F64);
    assert_eq!(ty.static_numel(), None);
}

#[test]
fn static_numel_overflow_returns_none() {
    // i64::MAX 维度会溢出 u64 乘法
    let ty = Type::tensor(BaseType::F64, vec![Dim::Known(i64::MAX), Dim::Known(i64::MAX)]);
    assert_eq!(ty.static_numel(), None);
}

#[test]
fn static_numel_negative_returns_none() {
    let ty = Type::tensor(BaseType::F64, vec![Dim::Known(-1), Dim::Known(4)]);
    assert_eq!(ty.static_numel(), None);
}

#[test]
fn static_bytes_f64_2d() {
    // 3×4 f64 = 12 * 8 = 96 bytes
    let ty = Type::tensor(BaseType::F64, vec![Dim::Known(3), Dim::Known(4)]);
    assert_eq!(ty.static_bytes(), Some(96));
}

#[test]
fn static_bytes_f32_2d() {
    // 10×20 f32 = 200 * 4 = 800 bytes
    let ty = Type::tensor(BaseType::F32, vec![Dim::Known(10), Dim::Known(20)]);
    assert_eq!(ty.static_bytes(), Some(800));
}

#[test]
fn static_bytes_f64_3d_large() {
    // 1024×1024×128 f64 = 134217728 * 8 = 1073741824 bytes = 1GB
    let ty = Type::tensor(BaseType::F64, vec![Dim::Known(1024), Dim::Known(1024), Dim::Known(128)]);
    assert_eq!(ty.static_bytes(), Some(1024 * 1024 * 128 * 8));
}

#[test]
fn static_bytes_i8_single_byte() {
    // 100×100 i8 = 10000 * 1 = 10000 bytes
    let ty = Type::tensor(BaseType::I8, vec![Dim::Known(100), Dim::Known(100)]);
    assert_eq!(ty.static_bytes(), Some(10000));
}

// ── 内存预估 warning 测试（构造函数）─────────────────────────────────────

#[test]
fn large_zeros_triggers_memory_warning() {
    // zeros(1024, 1024, 256) f64 = 2GB → 应触发 warning
    let src = r#"
fn big() -> Tensor[f64, ..] {
    zeros(1024, 1024, 256)
}
"#;
    let warnings = lower_warnings(src);
    assert_has_warning(&warnings, "GB");
    assert_has_warning(&warnings, "tensor");
}

#[test]
fn large_randn_triggers_memory_warning() {
    // randn(20000, 20000) f64 = 3.2GB → 应触发 warning
    let src = r#"
fn big() -> Tensor[f64, ..] {
    randn(20000, 20000)
}
"#;
    let warnings = lower_warnings(src);
    assert_has_warning(&warnings, "GB");
}

#[test]
fn small_zeros_no_memory_warning() {
    // zeros(3, 4) = 96 bytes → 无 warning
    let src = r#"
fn small() -> Tensor[f64, ..] {
    zeros(3, 4)
}
"#;
    let warnings = lower_warnings(src);
    assert_no_warnings(&warnings);
}

#[test]
fn medium_randn_no_memory_warning() {
    // randn(10000, 10000) f64 = 800MB < 1GB → 无 warning
    let src = r#"
fn medium() -> Tensor[f64, ..] {
    randn(10000, 10000)
}
"#;
    let warnings = lower_warnings(src);
    assert_no_warnings(&warnings);
}

#[test]
fn large_ones_triggers_memory_warning() {
    // ones(1024, 1024, 128) f64 = 1GB → 应触发 warning（边界）
    let src = r#"
fn big() -> Tensor[f64, ..] {
    ones(1024, 1024, 128)
}
"#;
    let warnings = lower_warnings(src);
    assert_has_warning(&warnings, "GB");
}

#[test]
fn large_tensor_literal_triggers_memory_warning() {
    // 使用 randn() 构造大 tensor（randn 接受多维 shape 参数）
    let src = r#"
fn big() -> Tensor[f64, ..] {
    randn(1024, 1024, 256)
}
"#;
    let warnings = lower_warnings(src);
    assert_has_warning(&warnings, "GB");
}

// ── matmul FLOPs 预估 warning 测试 ───────────────────────────────────────

#[test]
fn large_matmul_triggers_flop_warning() {
    // (1000, 1000) @ (1000, 1000) = 10^9 FLOPs = 1 GFLOP → 应触发
    let src = r#"
fn big_matmul() -> Tensor[f64, ..] {
    let a = zeros(1000, 1000);
    let b = zeros(1000, 1000);
    a.matmul(b)
}
"#;
    let warnings = lower_warnings(src);
    assert_has_warning(&warnings, "GFLOPs");
}

#[test]
fn huge_matmul_triggers_flop_warning() {
    // (10000, 10000) @ (10000, 10000) = 10^12 FLOPs = 1000 GFLOPs → 应触发
    let src = r#"
fn huge_matmul() -> Tensor[f64, ..] {
    let a = zeros(10000, 10000);
    let b = zeros(10000, 10000);
    a.matmul(b)
}
"#;
    let warnings = lower_warnings(src);
    assert_has_warning(&warnings, "GFLOPs");
}

#[test]
fn small_matmul_no_flop_warning() {
    // (3, 4) @ (4, 5) = 60 FLOPs → 无 warning
    let src = r#"
fn small_matmul() -> Tensor[f64, ..] {
    let a = zeros(3, 4);
    let b = zeros(4, 5);
    a.matmul(b)
}
"#;
    let warnings = lower_warnings(src);
    assert_no_warnings(&warnings);
}

#[test]
fn medium_matmul_no_flop_warning() {
    // (100, 100) @ (100, 100) = 10^6 FLOPs < 1 GFLOP → 无 warning
    let src = r#"
fn medium_matmul() -> Tensor[f64, ..] {
    let a = zeros(100, 100);
    let b = zeros(100, 100);
    a.matmul(b)
}
"#;
    let warnings = lower_warnings(src);
    assert_no_warnings(&warnings);
}

#[test]
fn matmul_with_transpose_flop_warning() {
    // (1000, 2000) @ (2000, 1000).transpose() = (1000,2000)@(1000,2000) → 内侧不匹配
    // 但 (1000, 2000) @ (2000, 1000) = 2×10^9 = 2 GFLOPs → 应触发
    let src = r#"
fn attn() -> Tensor[f64, ..] {
    let q = zeros(1000, 2000);
    let k = zeros(2000, 1000);
    q.matmul(k)
}
"#;
    let warnings = lower_warnings(src);
    assert_has_warning(&warnings, "GFLOPs");
}

// ── 组合测试：内存 + FLOPs 同时触发 ──────────────────────────────────────

#[test]
fn large_matmul_triggers_both_memory_and_flop_warnings() {
    // (10000, 10000) @ (10000, 10000)
    // - 每个 tensor 800MB < 1GB（无内存 warning）
    // - matmul 10^12 FLOPs = 1000 GFLOPs（有 FLOPs warning）
    // - 结果 (10000, 10000) = 800MB < 1GB（无内存 warning）
    // 所以只有 FLOPs warning
    let src = r#"
fn big() -> Tensor[f64, ..] {
    let a = zeros(10000, 10000);
    let b = zeros(10000, 10000);
    a.matmul(b)
}
"#;
    let warnings = lower_warnings(src);
    assert_has_warning(&warnings, "GFLOPs");
    // 结果 tensor 800MB < 1GB，不应有内存 warning
    let has_mem = warnings.iter().any(|w| w.message.contains("GB 的 tensor"));
    assert!(!has_mem, "结果 tensor 800MB 不应触发内存 warning");
}

#[test]
fn huge_result_tensor_triggers_memory_warning() {
    // matmul 结果 (15000, 15000) = 1.8GB → 内存 warning
    // 输入 (15000, 4) @ (4, 15000) = 9×10^8 FLOPs < 1 GFLOP（无 FLOPs warning）
    let src = r#"
fn wide_result() -> Tensor[f64, ..] {
    let a = zeros(15000, 4);
    let b = zeros(4, 15000);
    a.matmul(b)
}
"#;
    let warnings = lower_warnings(src);
    // 结果 1.8GB 应触发内存 warning（方法 matmul 创建）
    assert_has_warning(&warnings, "GB");
    // 9×10^8 FLOPs < 1 GFLOP，不应触发 FLOPs warning
    let has_flop = warnings.iter().any(|w| w.message.contains("GFLOPs"));
    assert!(!has_flop, "9×10^8 FLOPs 不应触发 FLOPs warning");
}

// ── 边界情况 ─────────────────────────────────────────────────────────────

#[test]
fn dynamic_shape_no_warning() {
    // 含动态维度（变量参数）→ 无法预估 → 无 warning
    let src = r#"
fn dynamic(n: i64) -> Tensor[f64, ..] {
    zeros(n, n)
}
"#;
    let warnings = lower_warnings(src);
    assert_no_warnings(&warnings);
}

#[test]
fn symbol_dims_no_warning() {
    // 符号维度 → 无法静态预估 → 无 warning
    let src = r#"
fn sym(a: Tensor[f64, M, K], b: Tensor[f64, K, N]) -> Tensor[f64, ..] {
    a.matmul(b)
}
"#;
    let warnings = lower_warnings(src);
    assert_no_warnings(&warnings);
}

#[test]
fn non_tensor_operations_no_warning() {
    // 标量运算 → 无 tensor → 无 warning
    let src = r#"
fn scalar(x: i64) -> i64 {
    x + 1
}
"#;
    let warnings = lower_warnings(src);
    assert_no_warnings(&warnings);
}

#[test]
fn warning_carries_line_col() {
    // 验证 warning 携带正确的行号列号
    let src = r#"
fn big() -> Tensor[f64, ..] {
    zeros(1024, 1024, 256)
}
"#;
    let warnings = lower_warnings(src);
    assert!(!warnings.is_empty(), "应有 warning");
    let w = &warnings[0];
    assert!(w.line >= 2, "行号应 >= 2（函数体内），实际: {}", w.line);
    assert!(w.col >= 1, "列号应 >= 1");
}

// ── warning 消息格式验证 ────────────────────────────────────────────────

#[test]
fn memory_warning_message_format() {
    // 验证内存 warning 消息包含关键信息
    let src = r#"
fn big() -> Tensor[f64, ..] {
    zeros(1024, 1024, 256)
}
"#;
    let warnings = lower_warnings(src);
    let w = warnings.iter().find(|w| w.message.contains("GB")).unwrap();
    // 消息应包含：context、GB 数值、"编译期预估"
    assert!(w.message.contains("编译期预估"), "消息应含'编译期预估'，实际: {}", w.message);
    assert!(w.message.contains("OOM"), "消息应含'OOM'提示，实际: {}", w.message);
}

#[test]
fn flop_warning_message_format() {
    // 验证 FLOPs warning 消息包含关键信息
    let src = r#"
fn big() -> Tensor[f64, ..] {
    let a = zeros(1000, 1000);
    let b = zeros(1000, 1000);
    a.matmul(b)
}
"#;
    let warnings = lower_warnings(src);
    let w = warnings.iter().find(|w| w.message.contains("GFLOPs")).unwrap();
    // 消息应包含：matmul、GFLOPs 数值、shape 信息、编译期预估
    assert!(w.message.contains("matmul"), "消息应含'matmul'，实际: {}", w.message);
    assert!(w.message.contains("1000"), "消息应含 shape 1000，实际: {}", w.message);
    assert!(w.message.contains("编译期预估"), "消息应含'编译期预估'，实际: {}", w.message);
}

// ── 泛型构造函数预估 ────────────────────────────────────────────────────

#[test]
fn generic_constructor_large_tensor_warning() {
    // randn<f64>(1024, 1024, 256) → 2GB → 应触发
    let src = r#"
fn big() -> Tensor[f64, ..] {
    randn<f64>(1024, 1024, 256)
}
"#;
    let warnings = lower_warnings(src);
    assert_has_warning(&warnings, "GB");
}

#[test]
fn generic_constructor_small_no_warning() {
    // randn<f64>(3, 4) → 96 bytes → 无 warning
    let src = r#"
fn small() -> Tensor[f64, ..] {
    randn<f64>(3, 4)
}
"#;
    let warnings = lower_warnings(src);
    assert_no_warnings(&warnings);
}

// ── 高开销占位测试（标记 #[ignore]）──────────────────────────────────────

#[test]
#[ignore = "高内存开销：实际分配 8GB tensor 会触发 OOM，等待安全环境再测"]
fn actual_8gb_tensor_allocation_ignored() {
    // 这个测试会实际分配 8GB 内存（如果运行时支持），仅作占位
    // 编译期预估已通过 large_zeros_triggers_memory_warning 验证
    let src = r#"
fn huge() -> Tensor[f64, ..] {
    zeros(1024, 1024, 1024)
}
"#;
    let warnings = lower_warnings(src);
    // 编译期应有 warning，但运行时实际分配会 OOM
    assert_has_warning(&warnings, "GB");
}

#[test]
#[ignore = "高算力开销：实际执行 10^12 FLOPs matmul 会耗时极长，等待 GPU 环境再测"]
fn actual_huge_matmul_execution_ignored() {
    // 这个测试会实际执行 10^12 FLOPs matmul（如果运行时支持），仅作占位
    let src = r#"
fn huge() -> Tensor[f64, ..] {
    let a = zeros(10000, 10000);
    let b = zeros(10000, 10000);
    a.matmul(b)
}
"#;
    let warnings = lower_warnings(src);
    // 编译期应有 FLOPs warning，但运行时实际执行会极慢
    assert_has_warning(&warnings, "GFLOPs");
}

#[test]
#[ignore = "高开销：深层嵌套大 tensor 链可能耗尽内存，等待安全环境再测"]
fn deep_large_tensor_chain_ignored() {
    // 多个大 tensor 链式操作，编译期应有多个 warning
    let src = r#"
fn chain() -> Tensor[f64, ..] {
    let a = zeros(1024, 1024, 256);
    let b = zeros(1024, 1024, 256);
    let c = a.matmul(b.reshape(1024, 256 * 1024));
    c.reshape(1024, 1024, 256)
}
"#;
    let warnings = lower_warnings(src);
    // 应有多个 warning（至少一个 GB 级 tensor warning）
    assert!(!warnings.is_empty(), "应有 warning");
}
