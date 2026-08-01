//! M3.4 — Weak<T> 弱引用全链路测试。
//!
//! 覆盖：
//! - `Weak::new(rc)` 从 Rc/Arc 值创建弱引用（不增加强引用计数）
//! - `weak_upgrade(w)` 尝试取强引用：原 Rc 存活 → Option::Some(Rc 值)；已释放 → Option::None
//! - 弱引用不阻止强引用释放（函数返回后原 Rc 被 drop，weak_strong_count → 0）
//!
//! 已知 VM 分歧（既有行为，非本次引入）：
//! ① VM 的 `Let` 编译为 `Dup + Store(局部) + StoreGlobal(全局镜像)`——局部
//!    变量同时镜像到全局表，函数返回后全局仍持有 Rc 强引用，因此 VM 路径**不能**
//!    用「函数边界 let 绑定」验证 drop 语义。VM 侧 drop 验证改用「临时 Rc
//!    （无 let 绑定）」形式（临时实参在 native 调用后正确释放）；函数边界 drop
//!    的干净语义测试保留在解释器（作用域 pop_scope 正确释放局部变量）。
//! ② 计数 native（weak_strong_count/weak_weak_count）的**绝对/相对计数**在 VM
//!    下受全局镜像与操作数栈残留影响而不可靠（每 let 绑定的句柄被复制到全局表）。
//!    故计数测试仅保留在解释器（语义干净）；VM 路径只验证功能语义
//!    （upgrade Some/None、clone、注解、报错），计数 native 为「可选辅助」。
//!    此分歧已登记汇报总师，供后续 VM Let 全局镜像语义修复时参考。
//! - `weak_strong_count` / `weak_weak_count` 计数辅助
//! - `w.clone()`（Weak::clone 共享弱句柄）方法
//! - 类型注解 `let w: Weak<i64>` 类型检查通过
//! - 边界：从非 Rc 值创建 Weak 报错
//! - VM / 解释器 / JIT 三路径结果一致
//!
//! 背景：M3.4 弱引用——不增加引用计数，可 upgrade() 尝试取强引用。
//! 值表示 `Value::Weak(Weak<RefCell<Value>>)`（存 std::rc::Weak 才能 upgrade），
//! 类型 `Type::Weak(Box<Type>)`。

use tenth::lexer::lexer::Lexer;
use tenth::parser::parser::Parser;
use tenth::hir::lower::Lowerer;
use tenth::runtime::interpreter::Interpreter;
use tenth::runtime::vm::Vm;
use tenth::runtime::value::Value;
use tenth::runtime::natives::register_all_natives;
use tenth::compile::bytecode::BytecodeCompiler;
use tenth::compile::jit;

/// 解释器路径：lex → parse → lower → run。
fn run(src: &str) -> Result<Option<Value>, String> {
    let mut lexer = Lexer::new(src);
    let tokens = lexer.tokenize().map_err(|e| e.to_string())?;
    let mut parser = Parser::new(tokens);
    let program = parser.parse_program().map_err(|e| e.to_string())?;
    let mut lowerer = Lowerer::new();
    let hir = lowerer.lower_program(&program).map_err(|e| e.to_string())?;
    let mut interpreter = Interpreter::new(&hir);
    interpreter.execute_program(&hir).map_err(|e| e.to_string())
}

/// VM 路径：lex → parse → lower → bytecode → 执行 main。
/// 使用 register_all_natives 一次性注册全部 native（含 Weak::new/weak_upgrade 等）。
fn run_vm(src: &str) -> Result<Value, String> {
    let mut lexer = Lexer::new(src);
    let tokens = lexer.tokenize().map_err(|e| e.to_string())?;
    let mut parser = Parser::new(tokens);
    let program = parser.parse_program().map_err(|e| e.to_string())?;
    let mut lowerer = Lowerer::new();
    let hir = lowerer.lower_program(&program).map_err(|e| e.to_string())?;

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
            }
            Err(e) => return Err(format!("compile error: {}", e)),
        }
    }

    if let Some(ref expr) = hir.main_expr {
        let compiler = BytecodeCompiler::new();
        match compiler.compile_main(expr) {
            Ok((chunk, closures)) => {
                vm.add_fn("main".into(), chunk);
                for (name, closure_chunk) in closures {
                    vm.add_fn(name, closure_chunk);
                }
            }
            Err(e) => return Err(format!("compile error: {}", e)),
        }
        vm.call("main").map_err(|e| e.to_string())
    } else if vm.has_fn("main") {
        vm.call("main").map_err(|e| e.to_string())
    } else {
        Ok(Value::Unit)
    }
}

/// JIT 路径：与 run_vm 相同，但通过 jit::run_jit 执行。
/// run_jit 内部对不支持的结构自动 fallback 到 vm.call，因此总是产生结果。
fn run_jit(src: &str) -> Result<Value, String> {
    let mut lexer = Lexer::new(src);
    let tokens = lexer.tokenize().map_err(|e| e.to_string())?;
    let mut parser = Parser::new(tokens);
    let program = parser.parse_program().map_err(|e| e.to_string())?;
    let mut lowerer = Lowerer::new();
    let hir = lowerer.lower_program(&program).map_err(|e| e.to_string())?;

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
            }
            Err(e) => return Err(format!("compile error: {}", e)),
        }
    }

    if let Some(ref expr) = hir.main_expr {
        let compiler = BytecodeCompiler::new();
        match compiler.compile_main(expr) {
            Ok((chunk, closures)) => {
                vm.add_fn("main".into(), chunk);
                for (name, closure_chunk) in closures {
                    vm.add_fn(name, closure_chunk);
                }
            }
            Err(e) => return Err(format!("compile error: {}", e)),
        }
        jit::run_jit(&mut vm, "main").map_err(|e| e.to_string())
    } else if vm.has_fn("main") {
        jit::run_jit(&mut vm, "main").map_err(|e| e.to_string())
    } else {
        Ok(Value::Unit)
    }
}

fn expect_int(v: &Value, expected: i64, label: &str) {
    match v {
        Value::Int(n, _) => assert_eq!(*n, expected, "{}: 期望 Int({}), 实际 {}", label, expected, n),
        other => panic!("{}: 期望 Int({}), 实际 {:?}", label, expected, other),
    }
}

fn expect_str(v: &Value, expected: &str, label: &str) {
    match v {
        Value::String(s) => assert_eq!(s.as_str(), expected, "{}: 期望 '{}', 实际 '{}'", label, expected, s),
        other => panic!("{}: 期望 String({}), 实际 {:?}", label, expected, other),
    }
}

// ── Weak::new + weak_upgrade 成功（原 Rc 存活）──────────────────────────────

#[test]
fn test_weak_upgrade_some_interpreter() {
    let src = r#"
    let r = Rc::new(42);
    let w = Weak::new(r);
    let o = weak_upgrade(w);
    match o {
        Option::Some(rc) => rc.deref(),
        Option::None => 0,
    }
    "#;
    let result = run(src).unwrap();
    match result {
        Some(v) => expect_int(&v, 42, "interpreter Weak upgrade → Some(42)"),
        None => panic!("期望 Some(Int(42))"),
    }
}

#[test]
fn test_weak_upgrade_some_vm() {
    let src = r#"
    fn main() -> i64 {
        let r = Rc::new(42);
        let w = Weak::new(r);
        let o = weak_upgrade(w);
        match o {
            Option::Some(rc) => rc.deref(),
            Option::None => 0,
        }
    }
    "#;
    let result = run_vm(src).unwrap();
    expect_int(&result, 42, "VM Weak upgrade → Some(42)");
}

#[test]
fn test_weak_upgrade_some_jit() {
    let src = r#"
    fn main() -> i64 {
        let r = Rc::new(42);
        let w = Weak::new(r);
        let o = weak_upgrade(w);
        match o {
            Option::Some(rc) => rc.deref(),
            Option::None => 0,
        }
    }
    "#;
    let result = run_jit(src).unwrap();
    expect_int(&result, 42, "JIT Weak upgrade → Some(42)");
}

// ── Arc 值创建 Weak（Arc 暂用 Rc 等价实现）─────────────────────────────────

#[test]
fn test_weak_from_arc() {
    let src = r#"
    let a = Arc::new(7);
    let w = Weak::new(a);
    let o = weak_upgrade(w);
    match o {
        Option::Some(rc) => rc.deref(),
        Option::None => 0,
    }
    "#;
    let result = run(src).unwrap();
    match result {
        Some(v) => expect_int(&v, 7, "interpreter Weak from Arc → Some(7)"),
        None => panic!("期望 Some(Int(7))"),
    }
}

#[test]
fn test_weak_from_arc_vm() {
    let src = r#"
    fn main() -> i64 {
        let a = Arc::new(7);
        let w = Weak::new(a);
        let o = weak_upgrade(w);
        match o {
            Option::Some(rc) => rc.deref(),
            Option::None => 0,
        }
    }
    "#;
    let result = run_vm(src).unwrap();
    expect_int(&result, 7, "VM Weak from Arc → Some(7)");
}

// ── 原 Rc 释放后 upgrade 返回 None ─────────────────────────────────────────

#[test]
fn test_weak_upgrade_after_drop_none_interpreter() {
    let src = r#"
    fn make_weak() -> Weak<i64> {
        let r = Rc::new(42);
        Weak::new(r)
    }
    fn main() -> i64 {
        let w = make_weak();
        let o = weak_upgrade(w);
        match o {
            Option::Some(_) => 0,
            Option::None => 1,
        }
    }
    "#;
    let result = run(src).unwrap();
    match result {
        Some(v) => expect_int(&v, 1, "interpreter: 原 Rc 释放后 upgrade → None"),
        None => panic!("期望 Some(Int(1))"),
    }
}

#[test]
fn test_weak_upgrade_after_drop_none_vm() {
    // VM 路径：用「临时 Rc（无 let 绑定）」验证 drop 语义——
    // `Weak::new(Rc::new(42))` 的 Rc 是临时实参，native 调用后即释放，
    // upgrade 应返回 None。避免函数边界 let 绑定（VM 全局镜像使 Rc 存活）。
    let src = r#"
    fn main() -> i64 {
        let o = weak_upgrade(Weak::new(Rc::new(42)));
        match o {
            Option::Some(_) => 0,
            Option::None => 1,
        }
    }
    "#;
    let result = run_vm(src).unwrap();
    expect_int(&result, 1, "VM: 临时 Rc 释放后 upgrade → None");
}

#[test]
fn test_weak_upgrade_temporary_none_interpreter() {
    // 解释器同形式 parity：临时 Rc 释放后 upgrade → None
    let src = r#"
    let o = weak_upgrade(Weak::new(Rc::new(42)));
    match o {
        Option::Some(_) => 0,
        Option::None => 1,
    }
    "#;
    let result = run(src).unwrap();
    match result {
        Some(v) => expect_int(&v, 1, "interpreter: 临时 Rc 释放后 upgrade → None"),
        None => panic!("期望 Some(Int(1))"),
    }
}

// ── 弱引用不阻止强引用释放（计数验证）──────────────────────────────────────

#[test]
fn test_weak_does_not_keep_alive() {
    // make_weak 返回 Weak 时原 Rc 被 drop → strong_count 降为 0
    let src = r#"
    fn make_weak() -> Weak<i64> {
        let r = Rc::new(42);
        Weak::new(r)
    }
    fn main() -> i64 {
        let w = make_weak();
        weak_strong_count(w)
    }
    "#;
    let result = run(src).unwrap();
    match result {
        Some(v) => expect_int(&v, 0, "interpreter: 弱引用不阻止释放 → strong_count=0"),
        None => panic!("期望 Some(Int(0))"),
    }
}

#[test]
fn test_weak_strong_count_alive() {
    // r 存活时 strong_count = 1
    let src = r#"
    let r = Rc::new(42);
    let w = Weak::new(r);
    weak_strong_count(w)
    "#;
    let result = run(src).unwrap();
    match result {
        Some(v) => expect_int(&v, 1, "interpreter: r 存活 → strong_count=1"),
        None => panic!("期望 Some(Int(1))"),
    }
}

#[test]
fn test_weak_strong_count_alive_vm() {
    // VM 全局镜像使 let 绑定的 r 在全局表也持有一份强引用（strong=2），
    // 绝对计数与解释器（strong=1）不同。这里验证的是**相对不变量**：
    // 创建 Weak 不改变 strong count（两次测量差值 = 0）。
    // 句柄用内联临时（不 let 绑定），避免全局镜像放大计数。
    let src = r#"
    fn main() -> i64 {
        let r = Rc::new(42);
        let c1 = weak_strong_count(Weak::new(r));
        let c2 = weak_strong_count(Weak::new(r));
        c2 - c1
    }
    "#;
    let result = run_vm(src).unwrap();
    expect_int(&result, 0, "VM: Weak::new 不改变 strong count（c2-c1=0）");
}

#[test]
fn test_weak_weak_count() {
    // 相对不变量：每次 w.clone() 使 weak_count 增 1（c2 - c1 = 1）。
    // 用相对差值避免 native 实参克隆（args[0].clone() 会临时 +1 弱计数）与
    // VM 全局镜像造成的绝对计数差异。
    let src = r#"
    let r = Rc::new(42);
    let w = Weak::new(r);
    let c1 = weak_weak_count(w.clone());
    let w2 = w.clone();
    let c2 = weak_weak_count(w.clone());
    c2 - c1
    "#;
    let result = run(src).unwrap();
    match result {
        Some(v) => expect_int(&v, 1, "interpreter: 每次 Weak::clone 使 weak_count 增 1"),
        None => panic!("期望 Some(Int(1))"),
    }
}



// ── Weak::clone 方法（共享弱句柄）─────────────────────────────────────────

#[test]
fn test_weak_clone_method_upgrade() {
    let src = r#"
    let r = Rc::new(42);
    let w = Weak::new(r);
    let w2 = w.clone();
    let o = weak_upgrade(w2);
    match o {
        Option::Some(rc) => rc.deref(),
        Option::None => 0,
    }
    "#;
    let result = run(src).unwrap();
    match result {
        Some(v) => expect_int(&v, 42, "interpreter: w.clone() 后 upgrade → Some(42)"),
        None => panic!("期望 Some(Int(42))"),
    }
}

#[test]
fn test_weak_clone_method_upgrade_vm() {
    let src = r#"
    fn main() -> i64 {
        let r = Rc::new(42);
        let w = Weak::new(r);
        let w2 = w.clone();
        let o = weak_upgrade(w2);
        match o {
            Option::Some(rc) => rc.deref(),
            Option::None => 0,
        }
    }
    "#;
    let result = run_vm(src).unwrap();
    expect_int(&result, 42, "VM: w.clone() 后 upgrade → Some(42)");
}

// ── 类型注解：let w: Weak<i64> ─────────────────────────────────────────────

#[test]
fn test_weak_type_annotation() {
    let src = r#"
    let r = Rc::new(42);
    let w: Weak<i64> = Weak::new(r);
    let o = weak_upgrade(w);
    match o {
        Option::Some(rc) => rc.deref(),
        Option::None => 0,
    }
    "#;
    let result = run(src).unwrap();
    match result {
        Some(v) => expect_int(&v, 42, "interpreter: let w: Weak<i64>"),
        None => panic!("期望 Some(Int(42))"),
    }
}

#[test]
fn test_weak_type_annotation_vm() {
    let src = r#"
    fn main() -> i64 {
        let r = Rc::new(42);
        let w: Weak<i64> = Weak::new(r);
        let o = weak_upgrade(w);
        match o {
            Option::Some(rc) => rc.deref(),
            Option::None => 0,
        }
    }
    "#;
    let result = run_vm(src).unwrap();
    expect_int(&result, 42, "VM: let w: Weak<i64>");
}

// ── 边界：从非 Rc 值创建 Weak 报错 ─────────────────────────────────────────

#[test]
fn test_weak_from_non_rc_errors() {
    let src = "Weak::new(42)";
    let result = run(src);
    assert!(result.is_err(), "Weak::new(42) 应从非 Rc 值报错");
    let err = result.err().unwrap();
    assert!(err.contains("Weak::new"), "错误信息应提及 Weak::new，实际: {}", err);
}

#[test]
fn test_weak_upgrade_non_weak_errors() {
    let src = "weak_upgrade(42)";
    let result = run(src);
    assert!(result.is_err(), "weak_upgrade(42) 应因非 Weak 值报错");
    let err = result.err().unwrap();
    assert!(err.contains("Weak"), "错误信息应提及 Weak，实际: {}", err);
}

// ── Display：存活 / 悬垂 ───────────────────────────────────────────────────

#[test]
fn test_weak_display_alive() {
    let src = r#"
    let r = Rc::new(42);
    let w = Weak::new(r);
    to_string(w)
    "#;
    let result = run(src).unwrap();
    match result {
        Some(v) => expect_str(&v, "Weak<42>", "interpreter: 存活 Weak Display"),
        None => panic!("期望 Some(String)"),
    }
}

#[test]
fn test_weak_display_dangling() {
    let src = r#"
    fn make_weak() -> Weak<i64> {
        let r = Rc::new(42);
        Weak::new(r)
    }
    fn main() -> str {
        let w = make_weak();
        to_string(w)
    }
    "#;
    let result = run(src).unwrap();
    match result {
        Some(v) => expect_str(&v, "Weak<dangling>", "interpreter: 悬垂 Weak Display"),
        None => panic!("期望 Some(String)"),
    }
}
