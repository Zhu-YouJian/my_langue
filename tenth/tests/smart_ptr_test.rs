//! M1.1 — Box/Rc/Arc/Pin 智能指针全链路接通测试。
//!
//! 覆盖：
//! - `Box::new(x).deref()` 取值（解释器 + VM 双路径 + JIT fallback）
//! - `Rc::new` / `Arc::new` 共享指针 + clone（Rc::clone 共享语义）
//! - `Pin::new` 固定包装 + deref
//! - 类型注解 `let b: Box<i64> = Box::new(42)` 的类型检查通过
//! - has_struct 防护：用户自定义 `struct Box<T>` 时注解回退为 Generic 不误报
//! - VM/解释器/JIT 三路径结果一致
//!
//! 背景：问题29 系列遗留（缺"类型注解映射 + 方法 + 测试"三块），本文件补齐。

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
/// 使用 register_all_natives 一次性注册全部 native（含 Box::new/Rc::new/Arc::new/Pin::new）。
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

// ── Box：deref 取值 ─────────────────────────────────────────────────────────

#[test]
fn test_box_deref_interpreter() {
    let src = "Box::new(42).deref()";
    let result = run(src).unwrap();
    match result {
        Some(v) => expect_int(&v, 42, "interpreter Box::new(42).deref()"),
        None => panic!("interpreter: 期望 Some(Int(42))"),
    }
}

#[test]
fn test_box_deref_vm() {
    let src = "fn main() -> i64 { Box::new(42).deref() }";
    let result = run_vm(src).unwrap();
    expect_int(&result, 42, "VM Box::new(42).deref()");
}

#[test]
fn test_box_deref_jit() {
    let src = "fn main() -> i64 { Box::new(42).deref() }";
    let result = run_jit(src).unwrap();
    expect_int(&result, 42, "JIT Box::new(42).deref()");
}

#[test]
fn test_box_deref_mut() {
    let src = "Box::new(42).deref_mut()";
    let result = run(src).unwrap();
    match result {
        Some(v) => expect_int(&v, 42, "interpreter Box::new(42).deref_mut()"),
        None => panic!("期望 Some(Int(42))"),
    }
}

// ── Rc / Arc：共享指针 deref ────────────────────────────────────────────────

#[test]
fn test_rc_deref() {
    let src = "Rc::new(42).deref()";
    let result = run(src).unwrap();
    match result {
        Some(v) => expect_int(&v, 42, "interpreter Rc::new(42).deref()"),
        None => panic!("期望 Some(Int(42))"),
    }
}

#[test]
fn test_arc_deref() {
    // Arc 暂用 Rc 等价实现（value.rs 注释）
    let src = "Arc::new(42).deref()";
    let result = run(src).unwrap();
    match result {
        Some(v) => expect_int(&v, 42, "interpreter Arc::new(42).deref()"),
        None => panic!("期望 Some(Int(42))"),
    }
}

#[test]
fn test_rc_deref_vm() {
    let src = "fn main() -> i64 { Rc::new(42).deref() }";
    let result = run_vm(src).unwrap();
    expect_int(&result, 42, "VM Rc::new(42).deref()");
}

#[test]
fn test_arc_deref_vm() {
    let src = "fn main() -> i64 { Arc::new(42).deref() }";
    let result = run_vm(src).unwrap();
    expect_int(&result, 42, "VM Arc::new(42).deref()");
}

// ── Pin：固定包装 deref ────────────────────────────────────────────────────

#[test]
fn test_pin_deref() {
    let src = "Pin::new(42).deref()";
    let result = run(src).unwrap();
    match result {
        Some(v) => expect_int(&v, 42, "interpreter Pin::new(42).deref()"),
        None => panic!("期望 Some(Int(42))"),
    }
}

#[test]
fn test_pin_deref_vm() {
    let src = "fn main() -> i64 { Pin::new(42).deref() }";
    let result = run_vm(src).unwrap();
    expect_int(&result, 42, "VM Pin::new(42).deref()");
}

#[test]
fn test_pin_deref_jit() {
    let src = "fn main() -> i64 { Pin::new(42).deref() }";
    let result = run_jit(src).unwrap();
    expect_int(&result, 42, "JIT Pin::new(42).deref()");
}

// ── clone：深拷贝（Box/Pin）与 Rc::clone 共享 ──────────────────────────────

#[test]
fn test_box_clone_deref() {
    let src = r#"
    let b = Box::new(42);
    let c = b.clone();
    c.deref()
    "#;
    let result = run(src).unwrap();
    match result {
        Some(v) => expect_int(&v, 42, "interpreter Box clone → deref"),
        None => panic!("期望 Some(Int(42))"),
    }
}

#[test]
fn test_box_clone_deref_vm() {
    let src = r#"
    fn main() -> i64 {
        let b = Box::new(42);
        let c = b.clone();
        c.deref()
    }
    "#;
    let result = run_vm(src).unwrap();
    expect_int(&result, 42, "VM Box clone → deref");
}

#[test]
fn test_rc_clone_shares() {
    // Rc::clone 共享同一内部值：r2.deref() 应为 42
    let src = r#"
    let r = Rc::new(42);
    let r2 = r.clone();
    r2.deref()
    "#;
    let result = run(src).unwrap();
    match result {
        Some(v) => expect_int(&v, 42, "interpreter Rc clone → deref"),
        None => panic!("期望 Some(Int(42))"),
    }
}

#[test]
fn test_rc_clone_shares_vm() {
    let src = r#"
    fn main() -> i64 {
        let r = Rc::new(42);
        let r2 = r.clone();
        r2.deref()
    }
    "#;
    let result = run_vm(src).unwrap();
    expect_int(&result, 42, "VM Rc clone → deref");
}

// ── 类型注解：let b: Box<i64> = Box::new(42) 类型检查通过 ──────────────────

#[test]
fn test_type_annotation_box() {
    let src = r#"
    let b: Box<i64> = Box::new(42);
    b.deref()
    "#;
    let result = run(src).unwrap();
    match result {
        Some(v) => expect_int(&v, 42, "interpreter let b: Box<i64>"),
        None => panic!("期望 Some(Int(42))"),
    }
}

#[test]
fn test_type_annotation_box_vm() {
    let src = r#"
    fn main() -> i64 {
        let b: Box<i64> = Box::new(42);
        b.deref()
    }
    "#;
    let result = run_vm(src).unwrap();
    expect_int(&result, 42, "VM let b: Box<i64>");
}

#[test]
fn test_type_annotation_rc() {
    let src = r#"
    let r: Rc<i64> = Rc::new(42);
    r.deref()
    "#;
    let result = run(src).unwrap();
    match result {
        Some(v) => expect_int(&v, 42, "interpreter let r: Rc<i64>"),
        None => panic!("期望 Some(Int(42))"),
    }
}

#[test]
fn test_type_annotation_arc() {
    let src = r#"
    let a: Arc<i64> = Arc::new(42);
    a.deref()
    "#;
    let result = run(src).unwrap();
    match result {
        Some(v) => expect_int(&v, 42, "interpreter let a: Arc<i64>"),
        None => panic!("期望 Some(Int(42))"),
    }
}

#[test]
fn test_type_annotation_pin() {
    let src = r#"
    let p: Pin<i64> = Pin::new(42);
    p.deref()
    "#;
    let result = run(src).unwrap();
    match result {
        Some(v) => expect_int(&v, 42, "interpreter let p: Pin<i64>"),
        None => panic!("期望 Some(Int(42))"),
    }
}

#[test]
fn test_type_annotation_pin_vm() {
    let src = r#"
    fn main() -> i64 {
        let p: Pin<i64> = Pin::new(42);
        p.deref()
    }
    "#;
    let result = run_vm(src).unwrap();
    expect_int(&result, 42, "VM let p: Pin<i64>");
}

// ── 嵌套：Box<Rc<i64>> 类型注解 ────────────────────────────────────────────

#[test]
fn test_nested_box_rc_annotation() {
    let src = r#"
    let br: Box<Rc<i64>> = Box::new(Rc::new(42));
    br.deref().deref()
    "#;
    let result = run(src).unwrap();
    match result {
        Some(v) => expect_int(&v, 42, "interpreter Box<Rc<i64>> 嵌套 deref"),
        None => panic!("期望 Some(Int(42))"),
    }
}

// ── has_struct 防护：用户自定义 struct Box<T> 不被误映射 ────────────────────

#[test]
fn test_user_struct_box_struct_literal_not_mapped() {
    // 回归：generic_test.rs 模式——用户自定义 struct Box<T> + 结构体字面量
    // 必须仍作为用户类型（Generic），不被映射为内置 HeapBox。
    let src = r#"
    struct Box<T> { value: T }
    let b = Box<i32> { value: 100 };
    b.value
    "#;
    let result = run(src).unwrap();
    match result {
        Some(v) => expect_int(&v, 100, "用户自定义 struct Box<T> 字面量"),
        None => panic!("期望 Some(Int(100))"),
    }
}

#[test]
fn test_user_struct_box_annotation_guard() {
    // 用户声明 struct Box<T> 后，`let b: Box<i64>` 注解应回退为 Generic
    // （has_struct 防护）：let 语句不产生 HeapBox/Generic 类型冲突，
    // 编译通过（运行时值仍为 Box::new 构造的 HeapBox）。
    let src = r#"
    struct Box<T> { value: T }
    let b: Box<i64> = Box::new(42);
    b
    "#;
    let result = run(src);
    assert!(result.is_ok(), "has_struct 防护应使注解通过，实际: {:?}", result.err());
}

#[test]
fn test_user_struct_box_deref_errors() {
    // 用户声明 struct Box<T> 后，`Box<i64>` 是用户 Generic 类型而非内置，
    // 内置 deref 不可用 → 编译期报错（typestate 拦截），证明未误映射为内置。
    let src = r#"
    struct Box<T> { value: T }
    let b: Box<i64> = Box::new(42);
    b.deref()
    "#;
    let result = run(src);
    assert!(result.is_err(), "用户自定义 Box 上调用 deref 应报错，实际 {:?}", result);
}

#[test]
fn test_user_struct_rc_literal_not_mapped() {
    let src = r#"
    struct Rc { value: i32 }
    let r = Rc { value: 7 };
    r.value
    "#;
    let result = run(src).unwrap();
    match result {
        Some(v) => expect_int(&v, 7, "用户自定义 struct Rc 字面量"),
        None => panic!("期望 Some(Int(7))"),
    }
}

// ── 容器内嵌套普通值：Box<f64> deref ───────────────────────────────────────

#[test]
fn test_box_deref_float() {
    let src = "Box::new(3.5).deref()";
    let result = run(src).unwrap();
    match result {
        Some(Value::Float(f)) => assert!((f - 3.5).abs() < 1e-10, "期望 3.5, 实际 {}", f),
        other => panic!("期望 Some(Float(3.5)), 实际 {:?}", other),
    }
}

// ── 方法不存在时报错（不静默）──────────────────────────────────────────────

#[test]
fn test_box_unknown_method_errors() {
    let src = "Box::new(42).nonexistent()";
    let result = run(src);
    assert!(result.is_err(), "期望报错，实际 {:?}", result);
    let err = result.unwrap_err();
    assert!(err.contains("没有方法") || err.contains("Box"), "期望 '没有方法' 错误，实际: {}", err);
}
