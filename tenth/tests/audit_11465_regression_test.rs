//! AUDIT-11.4.65 回归测试：`let (a, b, c, d) = f()` 元组解构必须**按位分解元素类型**。
//!
//! 缺陷：`lower_stmt.rs` 的 `let` 解构把 init 的**整体类型**逐个赋给每个绑定名
//! （只有 `match` 模式按位分解）⇒ `let (a, b, c, d) = native()` 的四个变量静态类型
//! 全是那个 `Type::Tuple` = **静默错型**（不硬失败，靠 `infer_binary_type` 兜底 +
//! `check_binary_shape_compat` 只查 Tensor 才没被抓住）。
//!
//! 本测试断言的是**静态类型**，不是"能跑"。

use tenth::hir::hir::HirProgram;
use tenth::hir::lower::Lowerer;
use tenth::hir::types::{BaseType, Type};
use tenth::lexer::lexer::Lexer;
use tenth::parser::parser::Parser;

fn lower(src: &str) -> HirProgram {
    let mut lexer = Lexer::new(src);
    let tokens = lexer.tokenize().expect("lex 失败");
    let mut parser = Parser::new(tokens);
    let program = parser.parse_program().expect("parse 失败");
    let mut lowerer = Lowerer::new();
    lowerer.lower_program(&program).expect("lower 失败")
}

/// 取函数体的静态类型（= 函数最后表达式 / 块的类型）。
fn fn_body_ty(hir: &HirProgram, name: &str) -> Type {
    hir.functions
        .iter()
        .find(|f| f.name == name)
        .unwrap_or_else(|| panic!("找不到函数 {name}"))
        .body
        .ty
        .clone()
}

fn i64ty() -> Type {
    Type::Base(BaseType::I64)
}

// ── 活样本：`date_from_unix_days` 三元素同质元组 ──────────────────────────

const DATE_SRC: &str = r#"
    fn get_y() -> i64 { let (y, m, d) = date_from_unix_days(0); y }
    fn get_m() -> i64 { let (y, m, d) = date_from_unix_days(0); m }
    fn get_d() -> i64 { let (y, m, d) = date_from_unix_days(0); d }
    fn get_triple() -> (i64, i64, i64) { let (y, m, d) = date_from_unix_days(0); (y, m, d) }
"#;

/// 逐个变量：解构出的每个名字的静态类型 = 对应元素类型（i64），不是那个 Tuple。
#[test]
fn date_destructured_each_var_is_i64() {
    let hir = lower(DATE_SRC);
    for f in ["get_y", "get_m", "get_d"] {
        assert_eq!(
            fn_body_ty(&hir, f),
            i64ty(),
            "{f} 的解构变量静态类型应为 I64（整体 Type::Tuple 即 AUDIT-11.4.65 的静默错型）"
        );
    }
}

/// 一次性证据：`(y, m, d)` 的类型是 `(i64, i64, i64)`；
/// 若未按位分解则会得到 `((i64,i64,i64), (i64,i64,i64), (i64,i64,i64))`。
#[test]
fn date_destructured_triple_is_flat_tuple() {
    let hir = lower(DATE_SRC);
    assert_eq!(
        fn_body_ty(&hir, "get_triple"),
        Type::Tuple(vec![i64ty(), i64ty(), i64ty()]),
        "解构后的三元组必须保持扁平（元素类型未被整体 Tuple 污染）"
    );
}

// ── 异质 4 元组：证明是**逐位**赋型，而非"首元素类型套全局" ─────────────────

const HET_SRC: &str = r#"
    fn tup() -> (i64, f64, str, bool) { (7i64, 2.5, "x", true) }
    fn get_a() -> i64 { let (a, b, c, d) = tup(); a }
    fn get_b() -> f64 { let (a, b, c, d) = tup(); b }
    fn get_c() -> str { let (a, b, c, d) = tup(); c }
    fn get_d() -> bool { let (a, b, c, d) = tup(); d }
    fn get_quad() -> (i64, f64, str, bool) { let (a, b, c, d) = tup(); (a, b, c, d) }
"#;

#[test]
fn heterogeneous_destructured_each_var_has_own_type() {
    let hir = lower(HET_SRC);
    assert_eq!(fn_body_ty(&hir, "get_a"), i64ty(), "第 1 位应为 I64");
    assert_eq!(fn_body_ty(&hir, "get_b"), Type::Base(BaseType::F64), "第 2 位应为 F64");
    assert_eq!(fn_body_ty(&hir, "get_c"), Type::Base(BaseType::Str), "第 3 位应为 str");
    assert_eq!(fn_body_ty(&hir, "get_d"), Type::Base(BaseType::Bool), "第 4 位应为 bool");
    assert_eq!(
        fn_body_ty(&hir, "get_quad"),
        Type::Tuple(vec![
            i64ty(),
            Type::Base(BaseType::F64),
            Type::Base(BaseType::Str),
            Type::Base(BaseType::Bool),
        ])
    );
}

// ── 反向守护：单名 `let t = tup();` **不得**被当成解构（仍取整体元组类型） ──

#[test]
fn single_name_let_keeps_whole_tuple_type() {
    let hir = lower(
        r#"
        fn tup() -> (i64, f64, str, bool) { (7i64, 2.5, "x", true) }
        fn get_t() -> (i64, f64, str, bool) { let t = tup(); t }
        "#,
    );
    assert_eq!(
        fn_body_ty(&hir, "get_t"),
        Type::Tuple(vec![
            i64ty(),
            Type::Base(BaseType::F64),
            Type::Base(BaseType::Str),
            Type::Base(BaseType::Bool),
        ]),
        "`let t = tuple` 是整体绑定，不能被误当成解构"
    );
}
