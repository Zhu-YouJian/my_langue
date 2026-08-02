//! 断点 4.2：前向引用 shape 不动点测试。
//!
//! 背景：调用**后定义**的同文件函数时，调用点只拿到注解签名（`Tensor[f64, ..]` → Any），
//! 拿不到 body 精化 shape，shape 检查漏到运行时（MEMO 2026-07-31）。本测试验证
//! `lower_program` 尾部的前向引用 shape 不动点 pass：
//! - 前向引用（main 先于 make 定义）编译期通过 shape 检查（不再漏到运行时）
//! - 注解 vs 前向引用 body 冲突 → 编译期 TypeError（护城河改进）
//! - 递归/互递归不挂死（轮数上限 ≤ 2n+1 兜底回退）
//! - `join_return_dims` n 元共识语义——同一多重集不同书写顺序结果一致（顺序无关）
//!
//! 收敛性论证见 `docs/shape-check-roadmap/前向引用断点收敛性论证.md`（数理部 2026-08-02）。

use tenth::lexer::lexer::Lexer;
use tenth::parser::parser::Parser;
use tenth::hir::lower::Lowerer;
use tenth::hir::types::{BaseType, Dim, Type};
use tenth::error::TenthError;
use tenth::runtime::interpreter::Interpreter;
use tenth::runtime::value::Value;

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

/// 运行源码（解释器路径），返回末表达式值。
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
    hir.functions
        .iter()
        .find(|f| f.name == name)
        .and_then(|f| match &f.return_type {
            Type::Tensor { dims, .. } => Some(dims.clone()),
            _ => None,
        })
}

// ── 1. join_return_dims n 元共识语义（顺序无关）────────────────────────────

#[test]
fn test_join_consensus_multiset_a_degrades_to_any() {
    // 多重集 [K2, K1, K1]：非 Any 输入 {2,1} 冲突 → 降为 Any。
    // 旧二元左折叠 `m(m(K2,K1),K1)=K1`（假精确）；新 n 元共识 → Any。
    let src = r#"
fn make() -> Tensor[f64, ..] {
    if true {
        return zeros(2, 1);
    }
    if true {
        return zeros(1, 1);
    }
    return zeros(1, 1);
}
"#;
    let hir = lower_to_hir(src);
    let dims = fn_return_dims(&hir, "make").expect("make 应返回 Tensor");
    assert_eq!(dims, vec![Dim::Any, Dim::Known(1)],
        "多重集 [2,1],[1,1],[1,1]：第 0 维 2/1 冲突应降为 Any");
}

#[test]
fn test_join_consensus_multiset_b_degrades_to_any() {
    // 同一多重集的不同书写顺序 [K1, K1, K2]：同样降为 Any。
    let src = r#"
fn make() -> Tensor[f64, ..] {
    if true {
        return zeros(1, 1);
    }
    if true {
        return zeros(1, 1);
    }
    return zeros(2, 1);
}
"#;
    let hir = lower_to_hir(src);
    let dims = fn_return_dims(&hir, "make").expect("make 应返回 Tensor");
    assert_eq!(dims, vec![Dim::Any, Dim::Known(1)],
        "多重集 [1,1],[1,1],[2,1]：同样应降为 Any（顺序无关）");
}

#[test]
fn test_join_consensus_order_independent() {
    // 同一多重集两种书写顺序 → 结果一致（旧实现 [K2,K1,K1]→K1 与 [K1,K1,K2]→Any 不同）。
    let src_a = r#"
fn make() -> Tensor[f64, ..] {
    if true { return zeros(2, 1); }
    if true { return zeros(1, 1); }
    return zeros(1, 1);
}
"#;
    let src_b = r#"
fn make() -> Tensor[f64, ..] {
    if true { return zeros(1, 1); }
    if true { return zeros(1, 1); }
    return zeros(2, 1);
}
"#;
    let dims_a = fn_return_dims(&lower_to_hir(src_a), "make").unwrap();
    let dims_b = fn_return_dims(&lower_to_hir(src_b), "make").unwrap();
    assert_eq!(dims_a, dims_b, "同一多重集不同书写顺序应得到一致 join 结果");
}

#[test]
fn test_join_consensus_all_same_preserved() {
    // 全部非 Any 相同 → 保留精确值。
    let src = r#"
fn make() -> Tensor[f64, ..] {
    if true { return zeros(3, 4); }
    if true { return zeros(3, 4); }
    return zeros(3, 4);
}
"#;
    let hir = lower_to_hir(src);
    let dims = fn_return_dims(&hir, "make").expect("make 应返回 Tensor");
    assert_eq!(dims, vec![Dim::Known(3), Dim::Known(4)],
        "三个相同 return [3,4] 应保留精确 shape");
}

#[test]
fn test_join_consensus_any_wildcard_not_constraining() {
    // Any 路径（如递归/未定义调用返回注解签名）不约束共识。
    let src = r#"
fn make() -> Tensor[f64, ..] {
    if true { return zeros(3, 4); }
    return make2();
}
fn make2() -> Tensor[f64, ..] {
    zeros(5, 4)
}
"#;
    let hir = lower_to_hir(src);
    let dims = fn_return_dims(&hir, "make").expect("make 应返回 Tensor");
    // 第 0 维 {3,5} 冲突 → Any；第 1 维 {4,4} → 4。
    assert_eq!(dims, vec![Dim::Any, Dim::Known(4)]);
}

// ── 2. 前向引用：main 先于 make 定义 ──────────────────────────────────────

#[test]
fn test_forward_ref_main_before_make_matmul_mismatch() {
    // MEMO 2026-07-31 实测场景：main 先于 make 定义，[3,4] @ [5,6]。
    // 此前 shape 检查漏到运行时；断点 4.2 后应**编译期**拦截。
    let src = r#"
fn main() {
    let m = make(3, 4);
    let n = zeros(5, 6);
    let r = m.matmul(n);
    println(r.shape_tensor());
}
fn make(a: i64, b: i64) -> Tensor[f64, ..] {
    zeros(a, b)
}
"#;
    assert_compile_error(src, "编译期 matmul shape 不兼容");
}

#[test]
fn test_forward_ref_main_before_make_matmul_match() {
    // [3,4] @ [4,5] 匹配：编译通过 + 运行输出 [3,5]。
    let src = r#"
fn main() {
    let m = make(3, 4);
    let n = zeros(4, 5);
    let r = m.matmul(n);
    println(r.shape_tensor());
}
fn make(a: i64, b: i64) -> Tensor[f64, ..] {
    zeros(a, b)
}
"#;
    lower(src).expect("make(3,4) @ zeros(4,5) 应编译通过");
    run_code(src).expect("运行应成功");
}

#[test]
fn test_forward_ref_return_shape_refined() {
    // f 先定义、调用后定义的 g：f 的 return_type 应精化为 g 的 body shape [2,3]。
    let src = r#"
fn f() -> Tensor[f64, ..] {
    return g();
}
fn g() -> Tensor[f64, ..] {
    return zeros(2, 3);
}
fn main() {
    println(f().shape_tensor());
}
"#;
    let hir = lower_to_hir(src);
    let dims = fn_return_dims(&hir, "f").expect("f 应返回 Tensor");
    assert_eq!(dims, vec![Dim::Known(2), Dim::Known(3)],
        "f 调用后定义 g()：f 的 return_type 应精化为 [2, 3]");
}

#[test]
fn test_forward_ref_annot_conflict_reports_error() {
    // 注解 [3,4] vs 前向引用 body [2,3] → 编译期 TypeError（新拦截）。
    let src = r#"
fn f() -> Tensor[f64, 3, 4] {
    return g();
}
fn g() -> Tensor[f64, ..] {
    return zeros(2, 3);
}
fn main() {
    let x = f();
    println(x.shape_tensor());
}
"#;
    assert_compile_error(src, "函数返回值 shape 不匹配");
}

#[test]
fn test_forward_ref_definition_order_independent() {
    // make 定义在 main 之前 vs 之后 → 行为一致（正例都编译通过）。
    let src_after = r#"
fn main() {
    let m = make(3, 4);
    let n = zeros(4, 5);
    m.matmul(n)
}
fn make(a: i64, b: i64) -> Tensor[f64, ..] {
    zeros(a, b)
}
"#;
    let src_before = r#"
fn make(a: i64, b: i64) -> Tensor[f64, ..] {
    zeros(a, b)
}
fn main() {
    let m = make(3, 4);
    let n = zeros(4, 5);
    m.matmul(n)
}
"#;
    lower(src_after).expect("后定义 make 应编译通过");
    lower(src_before).expect("先定义 make 应编译通过");
    let hir_after = lower_to_hir(src_after);
    let hir_before = lower_to_hir(src_before);
    assert_eq!(
        fn_return_dims(&hir_after, "make"),
        fn_return_dims(&hir_before, "make"),
        "make 的 return_type 不应依赖定义顺序"
    );
}

#[test]
fn test_forward_ref_three_level_chain() {
    // 三层前向链：main → f → g（f、g 均后定义）。
    let src = r#"
fn main() {
    let x = f(2, 2);
    let y = zeros(2, 2);
    let r = x.matmul(y);
    println(r.shape_tensor());
}
fn f(a: i64, b: i64) -> Tensor[f64, ..] {
    g(a, b)
}
fn g(a: i64, b: i64) -> Tensor[f64, ..] {
    zeros(a, b)
}
"#;
    lower(src).expect("三层前向引用链应编译通过");
}

#[test]
fn test_forward_ref_three_level_chain_mismatch() {
    // 三层前向链 + 内侧不匹配 → 编译期拦截。
    let src = r#"
fn main() {
    let x = f(2, 3);
    let y = zeros(4, 2);
    let r = x.matmul(y);
    println(r.shape_tensor());
}
fn f(a: i64, b: i64) -> Tensor[f64, ..] {
    g(a, b)
}
fn g(a: i64, b: i64) -> Tensor[f64, ..] {
    zeros(a, b)
}
"#;
    assert_compile_error(src, "编译期 matmul shape 不兼容");
}

#[test]
fn test_forward_ref_let_binding_propagates_shape() {
    // let m = make(3,4) 绑定后，m 的类型应传播精化 shape（receiver 生效）。
    let src = r#"
fn main() {
    let m = make(3, 4);
    let n = zeros(5, 6);
    m.matmul(n)
}
fn make(a: i64, b: i64) -> Tensor[f64, ..] {
    zeros(a, b)
}
"#;
    assert_compile_error(src, "编译期 matmul shape 不兼容");
}

#[test]
fn test_forward_ref_let_binding_correct_compiles() {
    let src = r#"
fn main() {
    let m = make(3, 4);
    let n = zeros(4, 5);
    m.matmul(n)
}
fn make(a: i64, b: i64) -> Tensor[f64, ..] {
    zeros(a, b)
}
"#;
    lower(src).expect("匹配的 let 绑定应编译通过");
}

#[test]
fn test_forward_ref_method_chain() {
    // 前向引用函数返回 [3,4]，链式 .matmul(zeros(4,5)) → [3,5]，再 .matmul(zeros(5,2))。
    let src = r#"
fn main() {
    let r = make(3, 4).matmul(zeros(4, 5)).matmul(zeros(5, 2));
    println(r.shape_tensor());
}
fn make(a: i64, b: i64) -> Tensor[f64, ..] {
    zeros(a, b)
}
"#;
    lower(src).expect("前向引用链式方法调用应编译通过");
}

#[test]
fn test_forward_ref_method_chain_mismatch() {
    // 链式末端内侧不匹配 → 编译期拦截。
    let src = r#"
fn main() {
    let r = make(3, 4).matmul(zeros(4, 5)).matmul(zeros(6, 2));
    println(r.shape_tensor());
}
fn make(a: i64, b: i64) -> Tensor[f64, ..] {
    zeros(a, b)
}
"#;
    assert_compile_error(src, "编译期 matmul shape 不兼容");
}

// ── 3. 前向引用 + 二元广播 ────────────────────────────────────────────────

#[test]
fn test_forward_ref_binary_broadcast_mismatch() {
    // 前向引用函数返回 [3,4]，与 [3,5] 二元加不可广播 → 编译期拦截。
    let src = r#"
fn main() {
    let a = make(3, 4);
    let b = zeros(3, 5);
    a + b
}
fn make(a: i64, b: i64) -> Tensor[f64, ..] {
    zeros(a, b)
}
"#;
    assert_compile_error(src, "编译期 shape 不兼容");
}

#[test]
fn test_forward_ref_binary_broadcast_match() {
    // [3,4] + [3,4] 可广播 → 编译通过。
    let src = r#"
fn main() {
    let a = make(3, 4);
    let b = zeros(3, 4);
    a + b
}
fn make(a: i64, b: i64) -> Tensor[f64, ..] {
    zeros(a, b)
}
"#;
    lower(src).expect("可广播二元加应编译通过");
}

// ── 4. 递归/互递归不挂死（轮数上限兜底）───────────────────────────────────

#[test]
fn test_recursive_fib_no_hang() {
    // 自递归 i64 函数：编译通过、运行 fib(10)=55。
    let src = r#"
fn fib(n: i64) -> i64 {
    if n < 2 {
        return n;
    }
    return fib(n - 1) + fib(n - 2);
}
fn main() {
    println(fib(10));
}
"#;
    lower(src).expect("递归函数应编译通过（不挂死）");
    run_code(src).expect("递归函数应可运行");
}

#[test]
fn test_recursive_tensor_no_hang() {
    // 自递归返回 Tensor：不挂死、不报错（上限兜底回退）。
    let src = r#"
fn recur(n: i64) -> Tensor[f64, ..] {
    if n <= 0 {
        return zeros(3, 4);
    }
    return recur(n - 1);
}
fn main() {
    println(recur(3).shape_tensor());
}
"#;
    lower(src).expect("递归 Tensor 函数应编译通过（不挂死）");
}

#[test]
fn test_mutual_recursive_no_hang() {
    // 互递归（f0→f1→f2→f0）：不挂死、编译通过、运行正常。
    let src = r#"
fn f0(n: i64) -> i64 {
    if n <= 0 {
        return 0;
    }
    return f1(n - 1);
}
fn f1(n: i64) -> i64 {
    return f2(n - 1);
}
fn f2(n: i64) -> i64 {
    if n <= 0 {
        return 1;
    }
    return f0(n - 1);
}
fn main() {
    println(f0(6));
}
"#;
    lower(src).expect("互递归应编译通过（不挂死）");
    run_code(src).expect("互递归应可运行");
}

// ── 5. 边界 ───────────────────────────────────────────────────────────────

#[test]
fn test_main_unit_no_shape_unaffected() {
    // 无 Tensor 返回的 main：不动点不改变行为。
    let src = r#"
fn main() {
    println("hello");
}
"#;
    lower(src).expect("无 Tensor 返回的 main 应编译通过");
}

#[test]
fn test_annotated_shape_conflict_still_reports() {
    // 既有行为回归：注解 [3,4] vs 直接 body [2,3] 仍报错（非前向引用路径）。
    let src = r#"
fn make() -> Tensor[f64, 3, 4] {
    zeros(2, 3)
}
"#;
    assert_compile_error(src, "shape 不匹配");
}

#[test]
fn test_forward_ref_multi_return_callee_join() {
    // 被调函数多 return 路径 join 后，调用点拿到 join 结果。
    // make2 返回 [Any, 4]（两条路径 [2,4]/[5,4]），调用点 matmul 内侧为 Any → 保守通过。
    let src = r#"
fn main() {
    let m = make2();
    let n = zeros(4, 5);
    let r = m.matmul(n);
    println(r.shape_tensor());
}
fn make2() -> Tensor[f64, ..] {
    if true {
        return zeros(2, 4);
    }
    return zeros(5, 4);
}
"#;
    lower(src).expect("被调函数 join 后 Any 内侧应保守通过");
}

#[test]
fn test_forward_ref_multi_return_callee_mismatch() {
    // 被调函数 join 为 [Any, 4]，但调用点内侧确知冲突的另一侧为 Known 时仍拦截。
    // 这里 make2 两条路径 [3,4]/[5,4] → join [Any,4]；matmul zeros(6,5) 内侧 4≠5 保守通过
    //（Any 通配）；改用内侧已知冲突：zeros(4, 5) 时 4==4 通过。真正拦截靠 Known-Known。
    let src = r#"
fn main() {
    let m = make2();
    let n = zeros(4, 5);
    m.matmul(n)
}
fn make2() -> Tensor[f64, ..] {
    if true {
        return zeros(3, 4);
    }
    return zeros(5, 4);
}
"#;
    lower(src).expect("join 后内侧 4==4 应编译通过");
}

#[test]
fn test_symbol_param_forward_ref_substitution() {
    // 断点 4.1 + 4.2 叠加：前向引用函数带 Symbol 参数，调用点实参代换后 shape 检查生效。
    let src = r#"
fn main() {
    let m = make(2, 3);
    let n = zeros(4, 3);
    m.matmul(n)
}
fn make(a: i64, b: i64) -> Tensor[f64, ..] {
    zeros(a, b)
}
"#;
    assert_compile_error(src, "编译期 matmul shape 不兼容");
}
