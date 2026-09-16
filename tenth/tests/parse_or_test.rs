//! AUDIT-11.4.62：`parse_int_or` / `parse_float_or` —— **带失败信道**的解析原语。
//!
//! 缺陷：`parse_int("n/a")` 静默返回 0、`parse_float("x")` 静默返回 0.0
//! （0 是合法值 ⇒ 调用方无法区分"解析失败"与"结果就是 0"）。
//!
//! 本批新增返回**真 `Result<T, str>`** 的变体（不改旧 API 行为）：
//!   - 静态类型必须是 `Type::Generic{base: Enum("Result"), args:[T, str]}`，
//!     **不是**裸 `Type::Enum("Result")`（后者会让 `or_die` 取不到内型、
//!     并被 M3.4 误用告警刻意跳过 —— 正是 AUDIT-11.4.40 的老路）；
//!   - 运行时形态照真 Result 活样本（`Value::Enum{Result, Ok/Err}`，
//!     与 weak_upgrade 的 Option 构造同构）。
//!
//! 覆盖：合法 → Ok（两路径）、非法 → Err（两路径）、`or_die` 可用、
//!       类型层是 Generic、旧 `parse_int`/`parse_float` 行为不变。

use tenth::hir::lower::Lowerer;
use tenth::hir::types::{BaseType, Type};
use tenth::lexer::lexer::Lexer;
use tenth::parser::parser::Parser;
use tenth::runtime::interpreter::Interpreter;
use tenth::runtime::natives::register_all_natives;
use tenth::runtime::value::Value;
use tenth::runtime::vm::Vm;
use tenth::compile::bytecode::BytecodeCompiler;

fn lower(src: &str) -> Result<tenth::hir::hir::HirProgram, String> {
    let mut lexer = Lexer::new(src);
    let tokens = lexer.tokenize().map_err(|e| e.to_string())?;
    let mut parser = Parser::new(tokens);
    let program = parser.parse_program().map_err(|e| e.to_string())?;
    let mut lowerer = Lowerer::new();
    lowerer.lower_program(&program).map_err(|e| e.to_string())
}

/// VM 路径（`register_all_natives` = 真注册表，非测试内手抄副本）。
fn run_vm(src: &str) -> Result<Value, String> {
    let hir = lower(src)?;
    let mut vm = Vm::new();
    register_all_natives(&mut vm);
    for func in &hir.functions {
        let compiler = BytecodeCompiler::new();
        match compiler.compile(func) {
            Ok((chunk, closures)) => {
                vm.add_fn(func.name.clone(), chunk);
                for (name, closure_chunk) in closures {
                    vm.add_fn(name, closure_chunk);
                }
                vm.set_global(func.name.clone(), Value::FnRef {
                    name: func.name.clone(),
                    params: func.params.clone(),
                    return_type: func.return_type.clone(),
                    captures: vec![],
                });
            }
            Err(e) => return Err(format!("compile error: {e}")),
        }
    }
    if let Some(ref expr) = hir.main_expr {
        let compiler = BytecodeCompiler::new();
        let (chunk, closures) = compiler.compile_main(expr).map_err(|e| format!("compile error: {e}"))?;
        vm.add_fn("main".into(), chunk);
        for (name, closure_chunk) in closures {
            vm.add_fn(name, closure_chunk);
        }
        vm.call("main").map_err(|e| e.to_string())
    } else {
        Ok(Value::Unit)
    }
}

/// 解释器路径（= `TENTH_NO_VM=1`）。
fn run_interp(src: &str) -> Result<Value, String> {
    let hir = lower(src)?;
    let mut interp = Interpreter::new(&hir);
    interp.execute_program(&hir)
        .map(|v| v.unwrap_or(Value::Unit))
        .map_err(|e| e.to_string())
}

fn both(src: &str) -> (Value, Value) {
    let vm = run_vm(src).unwrap_or_else(|e| panic!("VM 执行失败: {e}\n{src}"));
    let ip = run_interp(src).unwrap_or_else(|e| panic!("解释器执行失败: {e}\n{src}"));
    (vm, ip)
}

/// 结果文本化（测试统一返回 String，避免依赖 Value 的 Display 细节）。
fn text(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        Value::Int(n, _) => n.to_string(),
        Value::Float(f) => format!("{}", f),
        Value::Bool(b) => b.to_string(),
        other => format!("{other:?}"),
    }
}

/// 把结果渲染成可比的字符串（match 消费 Result，不用 `?`）。
fn render(src: &str) -> (String, String) {
    let (vm, ip) = both(src);
    (text(&vm), text(&ip))
}

// ── ① 合法输入 → Ok（两路径一致） ───────────────────────────────────────

#[test]
fn parse_int_or_ok_both_paths() {
    let src = r#"
        match parse_int_or("42") {
            Result::Ok(v) => "OK:" + to_string(v),
            Result::Err(e) => "ERR:" + e,
        }
    "#;
    let (vm, ip) = render(src);
    assert_eq!(vm, "OK:42", "VM：合法输入必须走 Ok");
    assert_eq!(ip, "OK:42", "解释器：合法输入必须走 Ok");
}

#[test]
fn parse_float_or_ok_both_paths() {
    let src = r#"
        match parse_float_or("2.5") {
            Result::Ok(v) => "OK:" + to_string(v),
            Result::Err(e) => "ERR:" + e,
        }
    "#;
    let (vm, ip) = render(src);
    assert_eq!(vm, "OK:2.5", "VM：合法输入必须走 Ok");
    assert_eq!(ip, "OK:2.5", "解释器：合法输入必须走 Ok");
}

// ── ② 非法输入 → Err（**失败信道**，两路径一致） ─────────────────────────

#[test]
fn parse_int_or_invalid_is_err_both_paths() {
    let src = r#"
        match parse_int_or("n/a") {
            Result::Ok(v) => "OK:" + to_string(v),
            Result::Err(e) => "ERR:" + e,
        }
    "#;
    let (vm, ip) = render(src);
    // 关键：不再静默返回 0，而是响亮的 Err（含原文与原因）
    assert!(vm.starts_with("ERR:"), "VM：'n/a' 必须是 Err，实际 {vm}");
    assert!(vm.contains("parse_int_or") && vm.contains("n/a"),
        "VM：Err 消息应含原语名与原文，实际 {vm}");
    assert!(ip.starts_with("ERR:"), "解释器：'n/a' 必须是 Err，实际 {ip}");
    assert!(ip.contains("n/a"), "解释器：Err 消息应含原文，实际 {ip}");
}

#[test]
fn parse_float_or_invalid_is_err_both_paths() {
    let src = r#"
        match parse_float_or("abc") {
            Result::Ok(v) => "OK:" + to_string(v),
            Result::Err(e) => "ERR:" + e,
        }
    "#;
    let (vm, ip) = render(src);
    assert!(vm.starts_with("ERR:"), "VM：'abc' 必须是 Err，实际 {vm}");
    assert!(ip.starts_with("ERR:"), "解释器：'abc' 必须是 Err，实际 {ip}");
}

/// 空串同样是解析失败（旧 API 返回 0 —— 这正是无法区分的那种情况）。
#[test]
fn parse_int_or_empty_is_err_both_paths() {
    let src = r#"
        match parse_int_or("") {
            Result::Ok(v) => "OK:" + to_string(v),
            Result::Err(e) => "ERR",
        }
    "#;
    let (vm, ip) = render(src);
    assert_eq!(vm, "ERR", "VM：空串必须 Err");
    assert_eq!(ip, "ERR", "解释器：空串必须 Err");
}

// ── ③ `or_die` 消费路径（两路径一致） ────────────────────────────────────

#[test]
fn parse_or_die_consumption_both_paths() {
    let src = r#"
        let a = or_die(parse_int_or("7"));
        let b = or_die(parse_float_or("1.5"));
        to_string(a) + "|" + to_string(b)
    "#;
    let (vm, ip) = both(src);
    assert_eq!(text(&vm), "7|1.5", "VM：or_die(Ok) 应取回内型");
    assert_eq!(text(&ip), "7|1.5", "解释器：or_die(Ok) 应取回内型");
}

#[test]
fn parse_or_die_err_panics_both_paths() {
    // Err 经 or_die 必须响亮失败（不许静默返回 0）
    let src = r#"or_die(parse_int_or("n/a"))"#;
    let vm = run_vm(src).unwrap_err();
    let ip = run_interp(src).unwrap_err();
    assert!(vm.contains("or_die"), "VM：or_die(Err) 应响亮，实际 {vm}");
    assert!(ip.contains("or_die"), "解释器：or_die(Err) 应响亮，实际 {ip}");

    // 带消息形式：自定义消息必须原样出现在错误里
    let src = r#"or_die(parse_float_or("abc"), "解析失败：不是数字")"#;
    let vm = run_vm(src).unwrap_err();
    let ip = run_interp(src).unwrap_err();
    assert!(vm.contains("解析失败：不是数字"), "VM：or_die msg 应透传，实际 {vm}");
    assert!(ip.contains("解析失败：不是数字"), "解释器：or_die msg 应透传，实际 {ip}");
}

// ── ④ 类型层必须是 Generic{Result, [T, str]}（不是裸 Enum） ───────────────

fn main_expr_ty(src: &str) -> Type {
    lower(src).expect("lower 失败").main_expr.expect("无 main_expr").ty
}

#[test]
fn parse_int_or_static_type_is_generic_result() {
    let ty = main_expr_ty(r#"parse_int_or("42")"#);
    assert_eq!(
        ty,
        Type::Generic {
            base: Box::new(Type::Enum("Result".to_string())),
            args: vec![Type::Base(BaseType::I64), Type::str_()],
        },
        "必须是 Generic{{Result,[i64,str]}}（裸 Enum(\"Result\") 会断掉 or_die 与 M3.4 告警），实际 {ty:?}"
    );
}

#[test]
fn parse_float_or_static_type_is_generic_result() {
    let ty = main_expr_ty(r#"parse_float_or("2.5")"#);
    assert_eq!(
        ty,
        Type::Generic {
            base: Box::new(Type::Enum("Result".to_string())),
            args: vec![Type::f64(), Type::str_()],
        },
        "必须是 Generic{{Result,[f64,str]}}，实际 {ty:?}"
    );
    // 反向守护：绝不是裸 Enum("Result")
    assert!(!matches!(ty, Type::Enum(_)), "不得是裸 Enum（否则重蹈 11.4.40）");
}

/// `or_die` 能从 Generic 形态抽回内型（这正是裸 Enum 做不到的）。
#[test]
fn or_die_extracts_inner_type_from_generic() {
    let ty = main_expr_ty(r#"or_die(parse_int_or("42"))"#);
    assert_eq!(ty, Type::Base(BaseType::I64), "or_die 应抽回 i64 内型，实际 {ty:?}");
}

// ── ⑤ 旧 API 行为不变（逐字回归门） ──────────────────────────────────────

#[test]
fn legacy_parse_int_still_silently_returns_zero() {
    assert_eq!(text(&run_vm(r#"parse_int("n/a")"#).unwrap()), "0");
    assert_eq!(text(&run_interp(r#"parse_int("n/a")"#).unwrap()), "0");
    assert_eq!(text(&run_vm(r#"parse_int("42")"#).unwrap()), "42");
    assert_eq!(text(&run_interp(r#"parse_int("42")"#).unwrap()), "42");
}

#[test]
fn legacy_parse_float_still_silently_returns_zero() {
    assert_eq!(text(&run_vm(r#"parse_float("abc")"#).unwrap()), "0");
    assert_eq!(text(&run_interp(r#"parse_float("abc")"#).unwrap()), "0");
    assert_eq!(text(&run_vm(r#"parse_float("2.5")"#).unwrap()), "2.5");
    assert_eq!(text(&run_interp(r#"parse_float("2.5")"#).unwrap()), "2.5");
}

/// 旧 API 的静态标注**保持**裸 `Enum("Option")`（本批有意不改：改它会放开
/// `tenth/std/**` 下游解析，风险远超收益；见 AUDIT-11.4.40 的取舍）。
#[test]
fn legacy_parse_static_annotation_unchanged() {
    assert_eq!(main_expr_ty(r#"parse_int("42")"#), Type::Enum("Option".to_string()));
    assert_eq!(main_expr_ty(r#"parse_float("2.5")"#), Type::Enum("Option".to_string()));
}
