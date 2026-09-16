//! AUDIT-11.4.40（本轮"部分缓解"路线②）：`Vec.get_opt` / `Vec.try_get` —— **真 Option**。
//!
//! 红线与设计（三处齐改，且**只新增**）：
//!   - 它是**方法**不是 native：落点是 ① VM 方法分派（`vm/natives.rs` 的 `Value::Vec` 臂）
//!     ② 解释器 `interpreter/methods.rs::eval_vec_method` ③ 类型层标注
//!     （`hir/lower/types.rs` 的 `Type::Array` 方法表）。当 native 加成会变**死条目**。
//!   - 值形态照真 Option 活样本 `weak_upgrade`：`Value::Enum{Option, Some/None}`。
//!   - **不动** `get`/`pop` 的运行时语义，也**不动**其类型标注
//!     （`tenth/std/**` 约 178 处 `.get(` 依赖裸元素行为）。
//!   - 类型标注必须是 `Generic{base: Enum("Option"), args:[T]}`，不是裸 `Enum("Option")`。

use tenth::hir::lower::Lowerer;
use tenth::hir::types::Type;
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

/// 结果文本化（避免依赖 Value 的 Display 细节）。
fn text(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        Value::Int(n, _) => n.to_string(),
        Value::Float(f) => format!("{}", f),
        Value::Bool(b) => b.to_string(),
        other => format!("{other:?}"),
    }
}

fn render(src: &str) -> (String, String) {
    let (vm, ip) = both(src);
    (text(&vm), text(&ip))
}

// ── ① 命中 → Some（两路径，且与 `get` 等价） ─────────────────────────────

#[test]
fn get_opt_in_range_is_some_both_paths() {
    let src = r#"
        let v = Vec::new();
        v.push(10);
        v.push(20);
        match v.get_opt(1) {
            Option::Some(x) => "Some:" + to_string(x),
            Option::None => "None",
        }
    "#;
    let (vm, ip) = render(src);
    assert_eq!(vm, "Some:20", "VM：命中必须 Some");
    assert_eq!(ip, "Some:20", "解释器：命中必须 Some");
}

#[test]
fn try_get_alias_is_some_both_paths() {
    let src = r#"
        let v = Vec::new();
        v.push("alpha");
        match v.try_get(0) {
            Option::Some(x) => "Some:" + x,
            Option::None => "None",
        }
    "#;
    let (vm, ip) = render(src);
    assert_eq!(vm, "Some:alpha", "VM：try_get 是 get_opt 的别名");
    assert_eq!(ip, "Some:alpha", "解释器：try_get 是 get_opt 的别名");
}

#[test]
fn get_opt_matches_get_on_hit_both_paths() {
    // 命中时 `get_opt` 与 `get` 的取值必须等价（不得引入新的包装/解包）
    let src = r#"
        let v = Vec::new();
        v.push(3);
        v.push(4);
        let a = or_die(v.get_opt(0));
        let b = v.get(0);
        to_string(a) + "|" + to_string(b)
    "#;
    let (vm, ip) = render(src);
    assert_eq!(vm, "3|3", "VM：get_opt 命中应与 get 等价");
    assert_eq!(ip, "3|3", "解释器：get_opt 命中应与 get 等价");
}

// ── ② 越界 / 空 Vec → None（**响亮面**：不是报错，两路径一致） ───────────

#[test]
fn get_opt_out_of_range_is_none_both_paths() {
    let src = r#"
        let v = Vec::new();
        v.push(1);
        match v.get_opt(5) {
            Option::Some(x) => "Some",
            Option::None => "None",
        }
    "#;
    let (vm, ip) = render(src);
    assert_eq!(vm, "None", "VM：越界必须返回 None（不报错）");
    assert_eq!(ip, "None", "解释器：越界必须返回 None（不报错）");
}

#[test]
fn get_opt_empty_vec_is_none_both_paths() {
    let src = r#"
        let v = Vec::new();
        match v.get_opt(0) {
            Option::Some(x) => "Some",
            Option::None => "None",
        }
    "#;
    let (vm, ip) = render(src);
    assert_eq!(vm, "None", "VM：空 Vec 必须返回 None");
    assert_eq!(ip, "None", "解释器：空 Vec 必须返回 None");
}

#[test]
fn get_opt_none_survives_or_die_with_message() {
    // None 经 `or_die(x, msg)` 必须响亮 panic（并带自定义消息）
    let src = r#"
        let v = Vec::new();
        or_die(v.get_opt(0), "自定义错误消息")
    "#;
    let vm = run_vm(src).unwrap_err();
    let ip = run_interp(src).unwrap_err();
    assert!(vm.contains("自定义错误消息"), "VM：or_die(None, msg) 应带消息，实际 {vm}");
    assert!(ip.contains("自定义错误消息"), "解释器：or_die(None, msg) 应带消息，实际 {ip}");
}

// ── ③ 红线：`get` / `pop` 的语义**逐字未变** ─────────────────────────────

#[test]
fn get_out_of_range_still_loud_both_paths() {
    let src = r#"
        let v = Vec::new();
        v.push(1);
        v.get(5)
    "#;
    let vm = run_vm(src).unwrap_err();
    let ip = run_interp(src).unwrap_err();
    assert!(vm.contains("越界"), "VM：get 越界仍须响亮，实际 {vm}");
    assert!(ip.contains("越界"), "解释器：get 越界仍须响亮，实际 {ip}");
}

#[test]
fn pop_on_empty_still_loud_both_paths() {
    let src = r#"
        let v = Vec::new();
        v.pop()
    "#;
    let vm = run_vm(src).unwrap_err();
    let ip = run_interp(src).unwrap_err();
    assert!(vm.contains("pop"), "VM：空 Vec pop 仍须响亮，实际 {vm}");
    assert!(ip.contains("pop"), "解释器：空 Vec pop 仍须响亮，实际 {ip}");
}

#[test]
fn get_and_pop_still_return_bare_element_both_paths() {
    // `get` 必须仍返**裸元素**（不是 Option）——`toml.th` 的 `parts.get(i).trim()` 依赖它
    let src = r#"
        let v = Vec::new();
        v.push(11);
        v.push(22);
        let a = v.get(1);
        let b = v.pop();
        to_string(a) + "|" + to_string(b)
    "#;
    let (vm, ip) = render(src);
    assert_eq!(vm, "22|22", "VM：get/pop 仍应返裸元素");
    assert_eq!(ip, "22|22", "解释器：get/pop 仍应返裸元素");
}

// ── ④ 类型层：Generic{Option,[T]}（不是裸 Enum） ─────────────────────────

fn main_expr_ty(src: &str) -> Type {
    lower(src).expect("lower 失败").main_expr.expect("无 main_expr").ty
}

#[test]
fn get_opt_static_type_is_generic_option() {
    let ty = main_expr_ty(r#"let v = Vec::new(); v.get_opt(0)"#);
    match &ty {
        Type::Generic { base, args } => {
            assert_eq!(**base, Type::Enum("Option".to_string()), "base 必须是 Enum(\"Option\")");
            assert_eq!(args.len(), 1, "必须带 1 个泛型实参（内型），实际 {ty:?}");
        }
        other => panic!("必须是 Generic{{Option,[T]}}，实际 {other:?}"),
    }
}

#[test]
fn try_get_static_type_is_generic_option() {
    let ty = main_expr_ty(r#"let v = Vec::new(); v.try_get(0)"#);
    assert!(
        matches!(&ty, Type::Generic { base, .. } if **base == Type::Enum("Option".to_string())),
        "必须是 Generic{{Option,[T]}}，实际 {ty:?}"
    );
}

#[test]
fn get_opt_inner_type_follows_element_type() {
    // 元素类型可静态追踪时（`split` 返回 `Vec<str>`），内型必须跟随元素类型
    // 注：`let v: Vec<i64> = Vec::new()` 目前**不会**把 inner 收窄为 i64
    // （Vec::new 的标注是 Array{inner: Unknown}，let 注解不细化容器内型——独立缺口，
    // 不在本批范围），故此处用 `split` 这类内型已知的接收者作证据。
    let ty = main_expr_ty(r#""a|b".split("|").get_opt(0)"#);
    assert_eq!(
        ty,
        Type::Generic {
            base: Box::new(Type::Enum("Option".to_string())),
            args: vec![Type::str_()],
        },
        "内型应跟随元素类型 str，实际 {ty:?}"
    );
}

#[test]
fn or_die_extracts_inner_type_from_get_opt() {
    let ty = main_expr_ty(r#"or_die("a|b".split("|").get_opt(0))"#);
    assert_eq!(ty, Type::str_(), "or_die 应抽回内型 str，实际 {ty:?}");
}

/// 反向守护：`get`/`pop` 的标注仍是裸 `Enum("Option")`（本批有意不动）。
#[test]
fn get_and_pop_annotations_unchanged() {
    assert_eq!(
        main_expr_ty(r#"let v = Vec::new(); v.get(0)"#),
        Type::Enum("Option".to_string()),
        "get 的标注必须逐字未变（裸 Enum(\"Option\")）"
    );
    assert_eq!(
        main_expr_ty(r#"let v = Vec::new(); v.pop()"#),
        Type::Enum("Option".to_string()),
        "pop 的标注必须逐字未变（裸 Enum(\"Option\")）"
    );
}
