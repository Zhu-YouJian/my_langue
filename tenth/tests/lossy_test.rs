//! 阶段2b-静默算错（lossy lattice）M2 里程碑测试：`lossy` 关键字全链路 + 污点传播。
//!
//! 核心命题：「可能算错」的值不能当确定正确的值用——除非显式 `lossy`
//! （对应 Rust 的 `unsafe`）。格：`Exact ≺ PossibleOverflow ≺ PossibleNaN ≺ Lossy`。
//!
//! M2 范围（方案 C 旁路分析，`hir/lower/taint.rs`）：
//! 1. **lossy 语法全链路**：lexer/parser/HIR；运行时 no-op（只求值 inner）。
//! 2. **静态 Lossy 来源**：隐式「标量 → 张量 dtype 收缩」（如 f16 张量 × f64 标量，
//!    标量被静默 cast 到张量 dtype）→ 标 Lossy 污点。
//! 3. **使用点检查**：只对静态确定的 Lossy 在使用点（println/to_string/format/
//!    write_file/save_weights）报错，提示用 `lossy(...)` 显式接受；
//!    PossibleOverflow/PossibleNaN 只传播不做使用点报错（防误报）。
//! 4. **跨函数传播（函子组合性）**：函数返回污点从 body 推导，调用点结果 =
//!    被调函数返回污点 ⊔ 实参污点。
//! 5. **防误报**：f64 张量、f32 张量×f32 标量、泛型/未知类型一律不报。

use tenth::error::TenthError;
use tenth::hir::lower::Lowerer;
use tenth::lexer::lexer::Lexer;
use tenth::parser::parser::Parser;
use tenth::runtime::interpreter::Interpreter;
use tenth::runtime::value::Value;

fn lower(src: &str) -> Result<(), TenthError> {
    let mut lexer = Lexer::new(src);
    let tokens = lexer.tokenize().expect("lex");
    let mut parser = Parser::new(tokens);
    let program = parser.parse_program().expect("parse");
    let mut lowerer = Lowerer::new();
    lowerer.lower_program(&program).map(|_| ())
}

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
// 1. lossy 语法全链路 + 运行时 no-op
// ══════════════════════════════════════════════════════════════════════

#[test]
fn lossy_is_a_keyword_and_prefix_expr() {
    // lossy expr 编译通过（前缀一元，参照 move/await/spawn）
    let src = r#"
fn ok() -> f64 {
    lossy 1.0
}
"#;
    assert_compiles(src);
}

#[test]
fn lossy_with_parens_compiles() {
    // lossy(expr) 同样成立（括号是普通分组）
    let src = r#"
fn ok() -> f64 {
    lossy(1.0)
}
"#;
    assert_compiles(src);
}

#[test]
fn lossy_is_runtime_noop() {
    // 运行时 no-op：lossy(expr) 求值 inner 并返回其值
    let src = r#"
lossy(41) + 1
"#;
    match run_code(src).unwrap() {
        Some(Value::Int(42, _)) => {}
        v => panic!("期望 42，实际 {:?}", v),
    }
}

#[test]
fn lossy_wraps_expression_result() {
    let src = r#"
fn f() -> f64 {
    lossy(3.0 * 2.0)
}
f()
"#;
    match run_code(src).unwrap() {
        Some(Value::Float(6.0)) => {}
        v => panic!("期望 6.0，实际 {:?}", v),
    }
}

// ══════════════════════════════════════════════════════════════════════
// 2. 静态 Lossy 来源：隐式标量 → 张量 dtype 收缩
// ══════════════════════════════════════════════════════════════════════

#[test]
fn f16_tensor_times_f64_scalar_is_lossy_sink_error() {
    // f16 张量 × f64 标量：标量被静默 cast 到 f16 → 精度降级 → Lossy
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
fn f32_tensor_times_f64_scalar_is_lossy_sink_error() {
    // f32 张量 × f64 标量：f64 标量被静默 cast 到 f32 → Lossy
    let src = r#"
fn bad() -> str {
    let t = zeros_f32(2, 2);
    let x = t * 1.23456789012345;
    to_string(x)
}
"#;
    assert_compile_error(src, "lossy 污点");
}

#[test]
fn f16_tensor_plus_f64_scalar_is_lossy_sink_error() {
    let src = r#"
fn bad() -> str {
    let t = ones_f16(2, 2);
    let x = t + 1.23456789012345;
    println(to_string(x));
    ""
}
"#;
    assert_compile_error(src, "lossy 污点");
}

// ══════════════════════════════════════════════════════════════════════
// 3. lossy(...) 显式接受：污点归零，放行
// ══════════════════════════════════════════════════════════════════════

#[test]
fn lossy_accepts_tainted_value() {
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
fn lossy_accepts_tainted_value_inline() {
    let src = r#"
fn ok() -> str {
    let t = zeros_f32(2, 2);
    to_string(lossy(t * 1.23456789012345))
}
"#;
    assert_compiles(src);
}

// ══════════════════════════════════════════════════════════════════════
// 4. 跨函数传播（函子组合性）：返回污点 + 实参污点
// ══════════════════════════════════════════════════════════════════════

#[test]
fn cross_function_return_taint_reaches_sink() {
    // make_lossy 的返回污点 = body 推导（Lossy）；调用点合并 → sink 报错
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

#[test]
fn cross_function_lossy_at_callsite_accepts() {
    // 调用点 lossy 显式接受 → 放行
    let src = r#"
fn make_lossy() -> Tensor[f16, ..] {
    let t = zeros_f16(2, 2);
    t * 1.23456789012345
}
fn ok() -> str {
    to_string(lossy(make_lossy()))
}
"#;
    assert_compiles(src);
}

#[test]
fn call_arg_taint_propagates_to_result() {
    // 被调函数返回 Exact，但实参 Lossy → 调用点结果 = Exact ⊔ Lossy = Lossy
    let src = r#"
fn pass_through(x: Tensor[f16, ..]) -> Tensor[f16, ..] {
    x
}
fn bad() -> str {
    let t = zeros_f16(2, 2);
    let x = t * 1.23456789012345;
    to_string(pass_through(x))
}
"#;
    assert_compile_error(src, "lossy 污点");
}

#[test]
fn cross_function_chain_propagates() {
    // 三层嵌套 f(g(h()))：每一层都传播 Lossy
    let src = r#"
fn h() -> Tensor[f16, ..] {
    let t = zeros_f16(2, 2);
    t * 1.23456789012345
}
fn g(x: Tensor[f16, ..]) -> Tensor[f16, ..] {
    x + x
}
fn f(x: Tensor[f16, ..]) -> str {
    to_string(x)
}
fn bad() -> str {
    f(g(h()))
}
"#;
    assert_compile_error(src, "lossy 污点");
}

// ══════════════════════════════════════════════════════════════════════
// 5. PossibleOverflow：传播但不做使用点报错（默认严格度）
// ══════════════════════════════════════════════════════════════════════

#[test]
fn float_overflow_taint_does_not_error_at_sink() {
    // 1e308 + 1e308 → inf（PossibleOverflow），但只对 Lossy 报错 → 不报
    let src = r#"
fn ok() -> str {
    let x = 1e308 + 1e308;
    to_string(x)
}
"#;
    assert_compiles(src);
}

#[test]
fn overflow_and_lossy_join_is_lossy() {
    // PossibleOverflow ⊔ Lossy = Lossy → sink 报错
    let src = r#"
fn bad() -> str {
    let t = zeros_f16(2, 2);
    let x = t * 1.23456789012345;
    let y = x + (1e308 + 1e308);
    to_string(y)
}
"#;
    assert_compile_error(src, "lossy 污点");
}

// ══════════════════════════════════════════════════════════════════════
// 6. 防误报回归（宁可漏报，不可误报）
// ══════════════════════════════════════════════════════════════════════

#[test]
fn f64_tensor_with_f64_scalar_no_false_positive() {
    // f64 张量 × f64 标量：无降级 → Exact
    let src = r#"
fn ok() -> str {
    let t = zeros(2, 2);
    let x = t * 1.23456789012345;
    to_string(x)
}
"#;
    assert_compiles(src);
}

#[test]
fn f32_tensor_with_f32_scalar_no_false_positive() {
    // f32 张量 × f32 标量：同精度 → Exact
    let src = r#"
fn ok() -> str {
    let t = zeros_f32(2, 2);
    let x = t * 1.23456789012345f32;
    to_string(x)
}
"#;
    assert_compiles(src);
}

#[test]
fn f16_tensor_tensor_no_false_positive() {
    // f16 张量 + f16 张量：同 dtype，无降级 → Exact
    let src = r#"
fn ok() -> str {
    let a = zeros_f16(2, 2);
    let b = ones_f16(2, 2);
    to_string(a + b)
}
"#;
    assert_compiles(src);
}

#[test]
fn generic_function_no_false_positive() {
    // 泛型模板 body 中 T 未知（TypeParam）→ 不判收缩 → 不报
    let src = r#"
fn scale<T>(x: Tensor[T, ..], s: f64) -> Tensor[T, ..] {
    x * s
}
fn ok() -> str {
    let t = zeros_f16(2, 2);
    to_string(scale<f16>(t, 1.5))
}
"#;
    assert_compiles(src);
}

#[test]
fn scalar_scalar_no_false_positive() {
    // 标量-标量混合 dtype 是提升（无降级）→ Exact
    let src = r#"
fn ok() -> str {
    let x = 1.5f32 + 1.23456789012345;
    to_string(x)
}
"#;
    assert_compiles(src);
}

#[test]
fn tensor_tensor_no_false_positive() {
    // 张量-张量混合 dtype 是提升 → Exact
    let src = r#"
fn ok() -> str {
    let a = zeros_f16(2, 2);
    let b = zeros(2, 2);
    to_string(a + b)
}
"#;
    assert_compiles(src);
}

#[test]
fn variable_scalar_no_false_positive() {
    // 变量标量（类型 F64）× f16 张量 → 仍属类型已知的收缩 → Lossy（不是误报）。
    // 此处验证的是：lossy 接受后可放行。
    let src = r#"
fn ok(s: f64) -> str {
    let t = zeros_f16(2, 2);
    to_string(lossy(t * s))
}
"#;
    assert_compiles(src);
}

// ══════════════════════════════════════════════════════════════════════
// 7. 分支合并：分支内赋值合并污点；分支内 let 不外泄
// ══════════════════════════════════════════════════════════════════════

#[test]
fn branch_assignment_merges_taint() {
    // 分支内赋值外部变量 → 合并后 Lossy → sink 报错（x 可能 lossy）
    let src = r#"
fn bad(cond: bool) -> str {
    let t = zeros_f16(2, 2);
    let mut x = 1.0;
    if cond {
        x = t * 1.23456789012345;
    }
    to_string(x)
}
"#;
    assert_compile_error(src, "lossy 污点");
}

#[test]
fn block_scoped_let_does_not_leak() {
    // 分支内 let 是块作用域，不外泄 → 后续使用不误报
    let src = r#"
fn ok(cond: bool) -> str {
    let x = 1.0;
    if cond {
        let t = zeros_f16(2, 2);
        let _y = t * 1.23456789012345;
    }
    to_string(x)
}
"#;
    assert_compiles(src);
}

#[test]
fn branch_lossy_accept_only_one_path_still_tainted() {
    // x 初始 Lossy；if 分支内 lossy(x) 归零，但另一路径仍 Lossy →
    // 合并后仍 Lossy（x 可能算错）→ sink 报错（格语义：可能即报）
    let src = r#"
fn bad(cond: bool) -> str {
    let t = zeros_f16(2, 2);
    let mut x = t * 1.23456789012345;
    if cond {
        x = lossy(x);
    }
    to_string(x)
}
"#;
    assert_compile_error(src, "lossy 污点");
}

// ══════════════════════════════════════════════════════════════════════
// 8. sink 名单补全（G5）与普通串插值（G4）——AUDIT-11.4.48 回归守护
// ══════════════════════════════════════════════════════════════════════
//
// 背景：`is_exact_sink` 原缺 `print`/`json_encode`/`json_encode_pretty`；
// `HirExprKind::InterpolatedString` 原先不遍历 parts（整条静默漏报）。
// 两处均已修，本节把修复钉住，防日后静默回退。
//
// 注意两条路径不同：f-string `f"{x}"` 在 lowering 期脱糖成 `format(...)` 调用
// （lower_expr.rs:1677-1721），而 `format` 原本就在 sink 名单，故 f-string 从来
// 不是缺口；真正的缺口是普通双引号串插值 `"{x}"`（lower_expr.rs:1666-1672
// 只把 `InterpPart::Expr(String)` 透传）。

// ── 8.1 正向：必须报错 ────────────────────────────────────────────────

#[test]
fn print_of_lossy_is_sink_error() {
    // G5：`print` 原先不在 sink 名单 → 静默通过（实测漏报）
    let src = r#"
fn bad() -> str {
    let t = zeros_f16(2, 2);
    let x = t * 1.23456789012345;
    print(x);
    ""
}
"#;
    assert_compile_error(src, "lossy 污点");
}

#[test]
fn json_encode_of_lossy_is_sink_error_even_if_result_unused() {
    // G5 纯漏报口径：编码结果**不再传入任何 sink**。
    // （若写成 println(json_encode(x))，污点会穿过 json_encode 传到 println，
    //   改前就已报错，那样测不出 sink 名单缺失。）
    let src = r#"
fn bad() -> str {
    let t = zeros_f16(2, 2);
    let x = t * 1.23456789012345;
    let s = json_encode(x);
    ""
}
"#;
    assert_compile_error(src, "lossy 污点");
}

#[test]
fn json_encode_pretty_of_lossy_is_sink_error_even_if_result_unused() {
    // 同 json_encode：纯漏报口径
    let src = r#"
fn bad() -> str {
    let t = zeros_f16(2, 2);
    let x = t * 1.23456789012345;
    let s = json_encode_pretty(x);
    ""
}
"#;
    assert_compile_error(src, "lossy 污点");
}

#[test]
fn plain_interp_let_binding_of_lossy_is_sink_error() {
    // G4：普通串插值 `"{x}"` 原先不遍历 parts → 静默通过。
    // 语义上等同于 to_string(x)：VM 为每个 Expr 部件 emit 一次 to_string
    // （compile/bytecode.rs 的 InterpolatedString 分支），解释器调 value_to_string。
    let src = r#"
fn bad() -> str {
    let t = zeros_f16(2, 2);
    let x = t * 1.23456789012345;
    let s = "{x}";
    ""
}
"#;
    assert_compile_error(src, "lossy 污点");
}

#[test]
fn plain_interp_inside_println_of_lossy_is_sink_error() {
    // G4 + 外层 sink 组合：插值体先报错（逃逸点是插值本身，不是外层 println）
    let src = r#"
fn bad() -> str {
    let t = zeros_f16(2, 2);
    let x = t * 1.23456789012345;
    println("{x}");
    ""
}
"#;
    assert_compile_error(src, "lossy 污点");
}

#[test]
fn plain_interp_bare_return_of_lossy_is_sink_error() {
    // 直接返回插值串（无任何外层 sink）也必须报错
    let src = r#"
fn bad() -> str {
    let t = zeros_f16(2, 2);
    let x = t * 1.23456789012345;
    "{x}"
}
"#;
    assert_compile_error(src, "lossy 污点");
}

// ── 8.2 零误报：必须编译通过 ──────────────────────────────────────────

#[test]
fn plain_interp_of_exact_int_no_false_positive() {
    // 既有 std/测试大量使用 `"{val}"`（Int）→ 不得误报
    let src = r#"
fn ok() -> str {
    let val = 42;
    "{val}"
}
"#;
    assert_compiles(src);
}

#[test]
fn plain_interp_of_exact_f32_tensor_no_false_positive() {
    // f32 张量 × f32 标量 = 同精度（无降级）→ Exact → 不报
    let src = r#"
fn ok() -> str {
    let x = zeros_f32(2, 2) * 1.5f32;
    "{x}"
}
"#;
    assert_compiles(src);
}

#[test]
fn plain_interp_after_lossy_accept_no_false_positive() {
    // lossy(...) 归零后插值放行。
    // 注意：普通串插值只接受标识符（lexer.rs 的 is_valid_ident），**无法内联写
    // `"{lossy(x)}"`**——含括号会被当字面文本，故必须先 `let y = lossy(x);` 再插值。
    let src = r#"
fn ok() -> str {
    let t = zeros_f16(2, 2);
    let x = t * 1.23456789012345;
    let y = lossy(x);
    "{y}"
}
"#;
    assert_compiles(src);
}

#[test]
fn fstring_after_lossy_accept_no_false_positive() {
    // f-string 路径（format 脱糖）在显式接受后同样放行
    let src = r#"
fn ok() -> str {
    let t = zeros_f16(2, 2);
    let x = t * 1.23456789012345;
    let y = lossy(x);
    f"{y}"
}
"#;
    assert_compiles(src);
}

// ── 8.3 已知残留漏报：钉住现状（不是期望行为）────────────────────────

#[test]
fn known_residual_gap_dotted_interp_path_still_silent() {
    // ⚠ 已知残留漏报（AUDIT-11.4.48 遗留 1）——**本断言记录的是缺口现状，
    //   不是期望行为**，保留它只为让缺口显形、防被顺手删掉。
    //
    // 成因：普通串插值的部件是 `InterpPart::Expr("x.shape")`（词法层允许 '.'，
    // 见 lexer.rs 的 is_valid_ident 校验），而污点表的键是简单变量名 →
    // `vt.get("x.shape")` 取不到 → 返回 Exact → 静默通过。
    //
    // 为何本轮不修：真修复需 HIR 存表达式而非变量名字符串（跨模块）；
    // 若退化成「按根变量 x 取污点」，`"{t.shape}"` 这类精确字段访问会被误伤
    // （shape 是精确的整数元组）→ 引入误报，更不可接受。
    //
    // 本断言的作用是**钉住**：谁修好了这条路径，本测试即失败 →
    // 请同步更新本断言与 AUDIT-11.4.48 遗留条目，不要直接删断言了事。
    let src = r#"
fn gap() -> str {
    let t = zeros_f16(2, 2);
    let x = t * 1.23456789012345;
    let s = "{x.shape}";
    ""
}
"#;
    match lower(src) {
        Ok(()) => {}
        Err(e) => panic!(
            "已知残留漏报已被修复（这是进步）：`\"{{x.shape}}\"` 现在会报错：{:?}\n请同步更新本断言与 AUDIT-11.4.48 遗留条目，而不是删除断言。",
            e
        ),
    }
}

#[test]
fn fstring_of_lossy_is_caught_via_format_desugar() {
    // 绊线（总师补，2026-09-16）：**f-string 不是缺口这件事，完全依赖一个实现细节**——
    // `ExprKind::FString` 在 lower 期被脱糖为 `Call{ func: Var("format"), .. }`
    // （`hir/lower/lower_expr.rs:1677-1721`），而 `format` 恰好早就在 sink 名单里。
    // 一旦该脱糖目标改名（或不再走 Call），f-string 里的 lossy 值会**静默不再被拦**，
    // 而本节其他用例都不会变红。故此处显式钉住「lossy 值经 f-string 必须报错」。
    let src = r#"
fn bad() -> str {
    let t = zeros_f16(2, 2);
    let x = t * 1.23456789012345;
    let s = f"{x}";
    s
}
"#;
    assert_compile_error(src, "lossy 污点");
}
