//! 跨函数 shape 求解测试（短期规划方向 2）。
//!
//! 验证函数体的精确 shape 能传播到调用方：
//! - `fn make() -> Tensor[f64, ..] { zeros(3, 4) }` 调用方拿到 [3, 4]
//! - 多 return 路径 join（相同保留、不同降 Any、维度数不同报错）
//! - 标准库函数 shape 跨调用传播
//! - 递归函数跳过（返回签名 shape，不报错）
//! - 调用方拿到精确 shape 后能触发下游 shape 检查

use tenth::lexer::lexer::Lexer;
use tenth::parser::parser::Parser;
use tenth::hir::lower::Lowerer;
use tenth::hir::types::{BaseType, Dim, Type};
use tenth::error::TenthError;

/// lower 源码，返回 () 或编译错误。
fn lower(src: &str) -> Result<(), TenthError> {
    let mut lexer = Lexer::new(src);
    let tokens = lexer.tokenize().expect("lex");
    let mut parser = Parser::new(tokens);
    let program = parser.parse_program().expect("parse");
    let mut lowerer = Lowerer::new();
    lowerer.lower_program(&program).map(|_| ())
}

/// lower 源码，返回 HirProgram（用于检查 fn_def.return_type）。
fn lower_to_hir(src: &str) -> tenth::hir::hir::HirProgram {
    let mut lexer = Lexer::new(src);
    let tokens = lexer.tokenize().expect("lex");
    let mut parser = Parser::new(tokens);
    let program = parser.parse_program().expect("parse");
    let mut lowerer = Lowerer::new();
    lowerer.lower_program(&program).expect("lower")
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

/// 从 HirProgram 中查找指定函数，返回其 return_type 的 dims（若是 Tensor）。
fn fn_return_dims(hir: &tenth::hir::hir::HirProgram, name: &str) -> Option<Vec<Dim>> {
    hir.functions.iter()
        .find(|f| f.name == name)
        .and_then(|f| match &f.return_type {
            Type::Tensor { dims, .. } => Some(dims.clone()),
            _ => None,
        })
}

// ── 1. 单 return 路径 shape 传播 ───────────────────────────────────────────

#[test]
fn test_single_return_shape_propagation() {
    // fn make() -> Tensor[f64, ..] { zeros(3, 4) }
    // make 的 return_type 应被合并为 Tensor[f64, 3, 4]
    let src = r#"
fn make() -> Tensor[f64, ..] {
    zeros(3, 4)
}
"#;
    let hir = lower_to_hir(src);
    let dims = fn_return_dims(&hir, "make").expect("make 应返回 Tensor");
    assert_eq!(dims, vec![Dim::Known(3), Dim::Known(4)],
        "单 return 路径：make 的 return_type 应为 [3, 4]");
}

#[test]
fn test_single_return_via_return_stmt() {
    // fn make() -> Tensor[f64, ..] { return zeros(3, 4); }
    // 显式 return 语句的 shape 也应被收集
    let src = r#"
fn make() -> Tensor[f64, ..] {
    return zeros(3, 4);
}
"#;
    let hir = lower_to_hir(src);
    let dims = fn_return_dims(&hir, "make").expect("make 应返回 Tensor");
    assert_eq!(dims, vec![Dim::Known(3), Dim::Known(4)],
        "显式 return 语句：make 的 return_type 应为 [3, 4]");
}

// ── 2. 多 return 路径 join ────────────────────────────────────────────────

#[test]
fn test_multi_return_same_shape_preserved() {
    // 两个 return 相同 shape → 保留 [3, 4]
    let src = r#"
fn make() -> Tensor[f64, ..] {
    if true {
        return zeros(3, 4);
    }
    return zeros(3, 4);
}
"#;
    let hir = lower_to_hir(src);
    let dims = fn_return_dims(&hir, "make").expect("make 应返回 Tensor");
    assert_eq!(dims, vec![Dim::Known(3), Dim::Known(4)],
        "两个 return 相同 shape：应保留 [3, 4]");
}

#[test]
fn test_multi_return_different_known_degrades_to_any() {
    // 两个 return 不同 Known → 降为 [Any, 4]（第 0 维 3≠5 降为 Any，第 1 维 4==4 保留）
    let src = r#"
fn make() -> Tensor[f64, ..] {
    if true {
        return zeros(3, 4);
    }
    return zeros(5, 4);
}
"#;
    let hir = lower_to_hir(src);
    let dims = fn_return_dims(&hir, "make").expect("make 应返回 Tensor");
    assert_eq!(dims.len(), 2, "维度数应为 2");
    assert_eq!(dims[0], Dim::Any, "第 0 维 3≠5 应降为 Any");
    assert_eq!(dims[1], Dim::Known(4), "第 1 维 4==4 应保留");
}

#[test]
fn test_multi_return_different_rank_reports_error() {
    // 两个 return 维度数不同 → 报错
    let src = r#"
fn make() -> Tensor[f64, ..] {
    if true {
        return zeros(3, 4);
    }
    return zeros(5, 6, 7);
}
"#;
    assert_compile_error(src, "维度数不匹配");
}

#[test]
fn test_multi_return_symbol_same_preserved() {
    // 两个 return 同名 Symbol → 保留 Symbol
    let src = r#"
fn make() -> Tensor[f64, ..] {
    if true {
        return zeros(3, 4);
    }
    return zeros(3, 4);
}
"#;
    let hir = lower_to_hir(src);
    let dims = fn_return_dims(&hir, "make").expect("make 应返回 Tensor");
    assert_eq!(dims, vec![Dim::Known(3), Dim::Known(4)]);
}

// ── 3. 标准库函数跨调用传播 ───────────────────────────────────────────────

#[test]
fn test_stdlib_randn_shape_propagation() {
    // fn weight() -> Tensor[f64, ..] { randn(3, 4) }
    // 调用方应拿到 [3, 4]
    let src = r#"
fn weight() -> Tensor[f64, ..] {
    randn(3, 4)
}
"#;
    let hir = lower_to_hir(src);
    let dims = fn_return_dims(&hir, "weight").expect("weight 应返回 Tensor");
    assert_eq!(dims, vec![Dim::Known(3), Dim::Known(4)],
        "randn(3,4) 的 shape 应传播到 weight 的 return_type");
}

#[test]
fn test_stdlib_zeros_shape_propagation() {
    // fn bias() -> Tensor[f64, ..] { zeros(8) }
    // 调用方应拿到 [8]
    let src = r#"
fn bias() -> Tensor[f64, ..] {
    zeros(8)
}
"#;
    let hir = lower_to_hir(src);
    let dims = fn_return_dims(&hir, "bias").expect("bias 应返回 Tensor");
    assert_eq!(dims, vec![Dim::Known(8)],
        "zeros(8) 的 shape 应传播到 bias 的 return_type");
}

// ── 4. 递归函数跳过 ──────────────────────────────────────────────────────

#[test]
fn test_recursive_function_skipped() {
    // 递归函数：fib 调用 fib，此时 self.functions 中还没有 fib
    // 应跳过 shape 传播（返回签名 shape），不报错
    let src = r#"
fn fib(n: i64) -> i64 {
    if n < 2 {
        return n;
    }
    return fib(n - 1) + fib(n - 2);
}
"#;
    lower(src).expect("递归函数应编译通过（跳过 shape 传播，返回签名 shape）");
}

#[test]
fn test_recursive_tensor_function_skipped() {
    // 递归函数返回 Tensor：不应因 shape 传播无限循环或报错
    let src = r#"
fn recur(n: i64) -> Tensor[f64, ..] {
    if n <= 0 {
        return zeros(3, 4);
    }
    return recur(n - 1);
}
"#;
    lower(src).expect("递归 Tensor 函数应编译通过（跳过 shape 传播）");
}

// ── 5. 调用方拿到精确 shape 后触发下游 shape 检查 ──────────────────────────

#[test]
fn test_caller_uses_precise_shape_matmul_pass() {
    // make() 返回 [3, 4]，make().matmul(zeros(4, 5)) → [3, 5]，应编译通过
    let src = r#"
fn make() -> Tensor[f64, ..] {
    zeros(3, 4)
}
fn caller() -> Tensor[f64, ..] {
    make().matmul(zeros(4, 5))
}
"#;
    lower(src).expect("make() 返回 [3,4]，matmul(zeros(4,5)) 应编译通过");
}

#[test]
fn test_caller_uses_precise_shape_matmul_fail() {
    // make() 返回 [3, 4]，make().matmul(zeros(5, 6)) → 内侧 4≠5，应编译失败
    // 这验证了调用方拿到了精确 shape [3, 4]（若拿到签名 shape [..] 则会保守通过）
    let src = r#"
fn make() -> Tensor[f64, ..] {
    zeros(3, 4)
}
fn caller() -> Tensor[f64, ..] {
    make().matmul(zeros(5, 6))
}
"#;
    assert_compile_error(src, "matmul shape 不兼容");
}

#[test]
fn test_caller_let_binding_propagates_shape() {
    // let x = make(); x.matmul(zeros(5, 6)) → 内侧 4≠5，应编译失败
    // 验证 let 绑定也能拿到精确 shape
    let src = r#"
fn make() -> Tensor[f64, ..] {
    zeros(3, 4)
}
fn caller() -> Tensor[f64, ..] {
    let x = make();
    x.matmul(zeros(5, 6))
}
"#;
    assert_compile_error(src, "matmul shape 不兼容");
}

#[test]
fn test_caller_let_binding_correct_shape_compiles() {
    // let x = make(); x.matmul(zeros(4, 5)) → [3,4]@[4,5]=[3,5]，应编译通过
    let src = r#"
fn make() -> Tensor[f64, ..] {
    zeros(3, 4)
}
fn caller() -> Tensor[f64, ..] {
    let x = make();
    x.matmul(zeros(4, 5))
}
"#;
    lower(src).expect("let x = make(); x.matmul(zeros(4,5)) 应编译通过");
}

// ── 6. 标准库函数跨调用触发下游检查 ──────────────────────────────────────

#[test]
fn test_stdlib_weight_cross_call_matmul_fail() {
    // weight() 返回 [3, 4]，weight().matmul(zeros(5, 6)) → 内侧 4≠5，应编译失败
    let src = r#"
fn weight() -> Tensor[f64, ..] {
    randn(3, 4)
}
fn caller() -> Tensor[f64, ..] {
    weight().matmul(zeros(5, 6))
}
"#;
    assert_compile_error(src, "matmul shape 不兼容");
}

#[test]
fn test_stdlib_weight_cross_call_matmul_pass() {
    // weight() 返回 [3, 4]，weight().matmul(zeros(4, 5)) → [3, 5]，应编译通过
    let src = r#"
fn weight() -> Tensor[f64, ..] {
    randn(3, 4)
}
fn caller() -> Tensor[f64, ..] {
    weight().matmul(zeros(4, 5))
}
"#;
    lower(src).expect("weight() 返回 [3,4]，matmul(zeros(4,5)) 应编译通过");
}

// ── 7. 边界场景 ──────────────────────────────────────────────────────────

#[test]
fn test_no_return_statement_uses_body_expr_shape() {
    // 无 return 语句，body 末尾是表达式：用表达式 shape
    let src = r#"
fn make() -> Tensor[f64, ..] {
    zeros(2, 3, 4)
}
"#;
    let hir = lower_to_hir(src);
    let dims = fn_return_dims(&hir, "make").expect("make 应返回 Tensor");
    assert_eq!(dims, vec![Dim::Known(2), Dim::Known(3), Dim::Known(4)],
        "无 return 语句：用 body 末尾表达式的 shape [2, 3, 4]");
}

#[test]
fn test_non_tensor_return_not_inferred() {
    // 返回非 Tensor 类型（i64）：不推断 shape，return_type 保持签名
    let src = r#"
fn answer() -> i64 {
    return 42;
}
"#;
    let hir = lower_to_hir(src);
    let f = hir.functions.iter().find(|f| f.name == "answer").expect("find answer");
    assert_eq!(f.return_type, Type::Base(BaseType::I64),
        "非 Tensor 返回：return_type 保持签名 i64");
}

#[test]
fn test_main_function_no_return_type_not_inferred() {
    // 无 Return 语句的函数（如 fn main() { ... }）：不推断
    let src = r#"
fn main() {
    println("hello");
}
"#;
    lower(src).expect("无返回值函数应编译通过");
}

#[test]
fn test_annotated_shape_conflict_reports_error() {
    // 注解 [3, 4] 但 body 推断 [2, 3] → 报错（注解强制化，方向 1 的职责）
    // 此测试验证跨函数 shape 求解不会破坏注解强制化
    let src = r#"
fn make() -> Tensor[f64, 3, 4] {
    zeros(2, 3)
}
"#;
    assert_compile_error(src, "shape 不匹配");
}

#[test]
fn test_annotated_shape_wildcard_uses_inferred() {
    // 注解 [..]（wildcard）+ body 推断 [3, 4] → 用推断 shape
    let src = r#"
fn make() -> Tensor[f64, ..] {
    zeros(3, 4)
}
"#;
    let hir = lower_to_hir(src);
    let dims = fn_return_dims(&hir, "make").expect("make 应返回 Tensor");
    assert_eq!(dims, vec![Dim::Known(3), Dim::Known(4)],
        "注解 [..] + body [3,4] → 用推断 shape [3, 4]");
}

// ── 8. 断点 4.1：符号维度 unify（调用点实参代换）──────────────────────────
// 见 docs/程序代数架构设计.md §4.1。
// `zeros(a,b)`（变量参数构造器）跨函数后是 [Symbol(a), Symbol(b)]；
// 调用点 `make(3,4)` 时实参为字面量，应把 Symbol(a)/Symbol(b) 代换为
// Known(3)/Known(4)，使 matmul/广播检查在编译期生效（不再漏到运行时）。

#[test]
fn test_variable_arg_ctor_cross_fn_literal_mismatch_reports_error() {
    // 验收场景（对应 05_cross_fn_shape.th）：
    // make(3,4) → [3,4]，@ zeros(5,6) 内侧 4≠5 应编译期报错
    let src = r#"
fn make(a: i64, b: i64) -> Tensor[f64, ..] {
    zeros(a, b)
}
fn caller() -> Tensor[f64, ..] {
    let m = make(3, 4);
    let n = zeros(5, 6);
    m.matmul(n)
}
"#;
    assert_compile_error(src, "matmul shape 不兼容");
}

#[test]
fn test_variable_arg_ctor_cross_fn_literal_correct_compiles() {
    // make(3,4) → [3,4]，@ zeros(4,5) 内侧 4==4 应编译通过
    let src = r#"
fn make(a: i64, b: i64) -> Tensor[f64, ..] {
    zeros(a, b)
}
fn caller() -> Tensor[f64, ..] {
    let m = make(3, 4);
    let n = zeros(4, 5);
    m.matmul(n)
}
"#;
    lower(src).expect("make(3,4) @ zeros(4,5) 应编译通过（unify 后 4==4）");
}

#[test]
fn test_variable_arg_ctor_cross_fn_binary_op_mismatch_reports_error() {
    // 二元广播路径同样受益：make(3,4) + zeros(5,6) 无法广播 → 编译期报错
    let src = r#"
fn make(a: i64, b: i64) -> Tensor[f64, ..] {
    zeros(a, b)
}
fn caller() -> Tensor[f64, ..] {
    make(3, 4) + zeros(5, 6)
}
"#;
    assert_compile_error(src, "shape 不兼容");
}

#[test]
fn test_variable_arg_ctor_cross_fn_variable_args_no_false_positive() {
    // 保守场景：实参也是变量时保持 Symbol（代换为 Symbol 重命名），不误报。
    // rows/cols 与 zeros(4,5) 的内侧比较走 Symbol vs Known 保守放行。
    let src = r#"
fn make(a: i64, b: i64) -> Tensor[f64, ..] {
    zeros(a, b)
}
fn caller() -> Tensor[f64, ..] {
    let rows = 3;
    let cols = 4;
    let m = make(rows, cols);
    let n = zeros(4, 5);
    m.matmul(n)
}
"#;
    lower(src).expect("实参为变量：保持 Symbol 保守放行，不应误报");
}

#[test]
fn test_variable_arg_ctor_call_site_result_shape_known_in_hir() {
    // 验证调用点代换真实生效：caller 中 make(3,4) 的结果类型应为 [Known(3), Known(4)]，
    // 而非 [Symbol(a), Symbol(b)]（若 body 中看不到 Known 说明代换未生效）。
    let src = r#"
fn make(a: i64, b: i64) -> Tensor[f64, ..] {
    zeros(a, b)
}
fn caller() -> Tensor[f64, ..] {
    let m = make(3, 4);
    m
}
"#;
    let hir = lower_to_hir(src);
    let caller = hir.functions.iter().find(|f| f.name == "caller").expect("find caller");
    let body_str = format!("{:?}", caller.body.kind);
    assert!(
        body_str.contains("Known(3)") && body_str.contains("Known(4)"),
        "make(3,4) 调用点结果应为 [Known(3), Known(4)]（实参代换生效），body: {}",
        body_str
    );
}

#[test]
fn test_symbol_dim_same_name_cross_fn_compiles() {
    // 07_symbol_dims 场景保持：同名符号维度跨函数仍兼容（不破坏）
    let src = r#"
fn scale(a: Tensor[f64, M, K], b: Tensor[f64, K, N]) -> Tensor[f64, M, N] {
    a.matmul(b)
}
fn caller() -> Tensor[f64, ..] {
    let a = zeros(3, 4);
    let b = zeros(4, 5);
    let c = scale(a, b);
    c.matmul(zeros(4, 5))
}
"#;
    lower(src).expect("同名符号维度跨函数应兼容（07 场景不破坏）");
}

#[test]
fn test_symbol_dim_different_name_cross_fn_reports_error() {
    // 不同名符号维度跨函数仍报错（与 matmul_symbol_dims_different_names_reports_error 对齐）
    let src = r#"
fn scale(a: Tensor[f64, M, K], b: Tensor[f64, P, N]) -> Tensor[f64, ..] {
    a.matmul(b)
}
fn caller() -> Tensor[f64, ..] {
    let a = zeros(3, 4);
    let b = zeros(4, 5);
    let c = scale(a, b);
    c
}
"#;
    assert_compile_error(src, "matmul shape 不兼容");
}
