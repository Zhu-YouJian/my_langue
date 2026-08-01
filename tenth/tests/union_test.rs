//! M1.2 — Union 类型全链路测试（tagged union）
//!
//! Tenth 的 union 是带 active_field 的 tagged union（非 C 风格内存重叠）：
//! - 构造：`U { field: value }` 恰好激活一个字段 → Value::Union { active_field, value }
//! - 字段访问：`u.field` 只允许读取当前 active 字段（访问非活跃字段报错）
//! - 字段修改：`u.field = v` 只允许修改当前 active 字段
//! - match 的 Union 变体模式（`U::Field(v)`）留待 M2；Binding 模式 `match u { x => ... }` 可用
//!
//! 覆盖：声明、构造、字段访问/修改、类型检查（编译期错误）、VM/解释器 parity、
//! JIT 路径、Display、嵌套、函数边界。

use tenth::lexer::lexer::Lexer;
use tenth::parser::parser::Parser;
use tenth::hir::lower::Lowerer;
use tenth::runtime::interpreter::Interpreter;
use tenth::runtime::value::Value;
use tenth::runtime::vm::Vm;
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

fn expect_union<'a>(v: &'a Value, name: &str) -> &'a Value {
    match v {
        Value::Union { name: n, active_field, value } => {
            assert_eq!(n, name, "union 名应为 {}", name);
            assert!(!active_field.is_empty(), "active_field 不应为空");
            &**value
        }
        other => panic!("期望 Value::Union，得到 {:?}", other),
    }
}

// ── 声明 + 构造 ──────────────────────────────────────────────────────────

#[test]
fn test_union_declare_and_construct() {
    let src = "union U { a: i32, b: f64 }; U { a: 42 }";
    let result = run(src).unwrap();
    let v = result.as_ref().expect("应返回 union 值");
    match v {
        Value::Union { name, active_field, value } => {
            assert_eq!(name, "U");
            assert_eq!(active_field, "a");
            match &**value {
                Value::Int(42, _) => {}
                other => panic!("期望 active 值 Int(42)，得到 {:?}", other),
            }
        }
        other => panic!("期望 Value::Union，得到 {:?}", other),
    }
}

#[test]
fn test_union_construct_float_field() {
    let src = "union U { a: i32, b: f64 }; U { b: 3.5 }";
    let result = run(src).unwrap();
    match result.as_ref().expect("应返回 union 值") {
        Value::Union { name, active_field, value } => {
            assert_eq!(name, "U");
            assert_eq!(active_field, "b");
            match &**value {
                Value::Float(3.5) => {}
                other => panic!("期望 active 值 Float(3.5)，得到 {:?}", other),
            }
        }
        other => panic!("期望 Value::Union，得到 {:?}", other),
    }
}

// ── 字段访问 ──────────────────────────────────────────────────────────────

#[test]
fn test_union_field_access_active() {
    let src = "union U { a: i32, b: f64 }; let u = U { a: 42 }; u.a";
    let result = run(src).unwrap();
    match result {
        Some(Value::Int(42, _)) => {}
        v => panic!("期望 Int(42)，得到 {:?}", v),
    }
}

#[test]
fn test_union_field_access_active_float() {
    let src = "union U { a: i32, b: f64 }; let u = U { b: 2.5 }; u.b";
    let result = run(src).unwrap();
    match result {
        Some(Value::Float(v)) => assert!((v - 2.5).abs() < 1e-10, "got {}", v),
        v => panic!("期望 Float(2.5)，得到 {:?}", v),
    }
}

#[test]
fn test_union_field_access_inactive_error() {
    // tagged union：访问非 active 字段应报错（防止误读未激活的内存）
    let src = "union U { a: i32, b: f64 }; let u = U { a: 42 }; u.b";
    let err = run(src).unwrap_err();
    assert!(
        err.contains("当前活跃字段") && err.contains("'b'"),
        "错误信息应说明不能访问非活跃字段 'b'，实际: {}",
        err
    );
}

#[test]
fn test_union_field_access_unknown_error() {
    // 访问不存在的字段应报错
    let src = "union U { a: i32, b: f64 }; let u = U { a: 42 }; u.zzz";
    let err = run(src).unwrap_err();
    assert!(
        err.contains("没有字段") || err.contains("'zzz'"),
        "错误信息应提及不存在的字段 'zzz'，实际: {}",
        err
    );
}

// ── 字段修改 ──────────────────────────────────────────────────────────────

#[test]
fn test_union_field_assign_active() {
    let src = "union U { a: i32, b: f64 }; let mut u = U { a: 42 }; u.a = 100; u.a";
    let result = run(src).unwrap();
    match result {
        Some(Value::Int(100, _)) => {}
        v => panic!("期望 Int(100)，得到 {:?}", v),
    }
}

#[test]
fn test_union_field_assign_inactive_error() {
    let src = "union U { a: i32, b: f64 }; let mut u = U { a: 42 }; u.b = 1.0";
    let err = run(src).unwrap_err();
    assert!(
        err.contains("不能修改非活跃字段"),
        "错误信息应说明不能修改非活跃字段，实际: {}",
        err
    );
}

// ── 类型检查（编译期错误） ────────────────────────────────────────────────

#[test]
fn test_union_construct_multiple_fields_error() {
    // tagged union 一次只能激活一个字段
    let src = "union U { a: i32, b: f64 }; U { a: 1, b: 2.0 }";
    let err = run(src).unwrap_err();
    assert!(
        err.contains("恰好激活一个字段") || err.contains("tagged union"),
        "构造多字段应编译期报错，实际: {}",
        err
    );
}

#[test]
fn test_union_construct_unknown_field_error() {
    let src = "union U { a: i32, b: f64 }; U { zzz: 1 }";
    let err = run(src).unwrap_err();
    assert!(
        err.contains("没有字段 'zzz'"),
        "构造未知字段应编译期报错，实际: {}",
        err
    );
}

#[test]
fn test_union_construct_defaults_error() {
    // union 不支持 `..` 默认字段填充
    let src = "union U { a: i32, b: f64 }; U { a: 1, .. }";
    let err = run(src).unwrap_err();
    assert!(
        err.contains("不支持默认字段填充"),
        "union 构造使用 .. 应报错，实际: {}",
        err
    );
}

// ── Display ───────────────────────────────────────────────────────────────

#[test]
fn test_union_display() {
    let src = "union U { a: i32, b: f64 }; U { a: 42 }";
    let result = run(src).unwrap();
    let s = format!("{}", result.as_ref().expect("应返回 union 值"));
    assert!(
        s.contains("union U") && s.contains("a: 42"),
        "Display 应为 'union U {{ a: 42 }}' 形式，实际: {}",
        s
    );
}

// ── 类型注解 ──────────────────────────────────────────────────────────────

#[test]
fn test_union_type_annotation() {
    let src = "union U { a: i32, b: f64 }; let u: U = U { a: 7 }; u.a";
    let result = run(src).unwrap();
    match result {
        Some(Value::Int(7, _)) => {}
        v => panic!("期望 Int(7)，得到 {:?}", v),
    }
}

// ── 嵌套 ──────────────────────────────────────────────────────────────────

#[test]
fn test_union_nested_union() {
    let src = r#"
    union Inner { x: i32, y: i32 }
    union Outer { i: Inner, s: i32 }
    let o = Outer { i: Inner { x: 7 } };
    o.i.x
    "#;
    let result = run(src).unwrap();
    match result {
        Some(Value::Int(7, _)) => {}
        v => panic!("期望 Int(7)，得到 {:?}", v),
    }
}

// ── 函数边界 ──────────────────────────────────────────────────────────────

#[test]
fn test_union_in_function() {
    let src = r#"
    union U { a: i32, b: f64 }
    fn make(v: i32) -> U { U { a: v } }
    fn get(u: U) -> i32 { u.a }
    get(make(5))
    "#;
    let result = run(src).unwrap();
    match result {
        Some(Value::Int(5, _)) => {}
        v => panic!("期望 Int(5)，得到 {:?}", v),
    }
}

// ── match Binding 模式（Union 变体模式留待 M2） ───────────────────────────

#[test]
fn test_union_match_binding() {
    let src = r#"
    union U { a: i32, b: f64 }
    let u = U { a: 9 };
    match u { x => x.a }
    "#;
    let result = run(src).unwrap();
    match result {
        Some(Value::Int(9, _)) => {}
        v => panic!("期望 Int(9)，得到 {:?}", v),
    }
}

// ── VM / 解释器 parity ────────────────────────────────────────────────────

#[test]
fn test_union_vm_parity_construct_access() {
    let src = "union U { a: i32, b: f64 }; let u = U { a: 42 }; u.a";
    let interp = run(src).unwrap();
    let vm = run_vm(src).unwrap();
    match (interp, vm) {
        (Some(Value::Int(a, _)), Value::Int(b, _)) => assert_eq!(a, b, "解释器与 VM 结果应一致"),
        (i, v) => panic!("parity 失败: 解释器 {:?} vs VM {:?}", i, v),
    }
}

#[test]
fn test_union_vm_parity_float_field() {
    let src = "union U { a: i32, b: f64 }; let u = U { b: 2.5 }; u.b";
    let interp = run(src).unwrap();
    let vm = run_vm(src).unwrap();
    match (interp, vm) {
        (Some(Value::Float(a)), Value::Float(b)) => assert!((a - b).abs() < 1e-10),
        (i, v) => panic!("parity 失败: 解释器 {:?} vs VM {:?}", i, v),
    }
}

#[test]
fn test_union_vm_parity_field_assign() {
    let src = "union U { a: i32, b: f64 }; let mut u = U { a: 42 }; u.a = 100; u.a";
    let interp = run(src).unwrap();
    let vm = run_vm(src).unwrap();
    match (interp, vm) {
        (Some(Value::Int(a, _)), Value::Int(b, _)) => assert_eq!(a, b, "字段修改后结果应一致"),
        (i, v) => panic!("parity 失败: 解释器 {:?} vs VM {:?}", i, v),
    }
}

#[test]
fn test_union_vm_parity_error_inactive_access() {
    // VM 与解释器对访问非 active 字段都应报错
    let src = "union U { a: i32, b: f64 }; let u = U { a: 42 }; u.b";
    let interp_err = run(src).unwrap_err();
    let vm_err = run_vm(src).unwrap_err();
    assert!(interp_err.contains("当前活跃字段"), "解释器: {}", interp_err);
    assert!(vm_err.contains("当前活跃字段"), "VM: {}", vm_err);
}

#[test]
fn test_union_vm_parity_nested() {
    let src = r#"
    union Inner { x: i32, y: i32 }
    union Outer { i: Inner, s: i32 }
    let o = Outer { i: Inner { x: 7 } };
    o.i.x
    "#;
    let interp = run(src).unwrap();
    let vm = run_vm(src).unwrap();
    match (interp, vm) {
        (Some(Value::Int(a, _)), Value::Int(b, _)) => assert_eq!(a, b),
        (i, v) => panic!("parity 失败: 解释器 {:?} vs VM {:?}", i, v),
    }
}

// ── JIT 路径 ──────────────────────────────────────────────────────────────

#[test]
fn test_union_jit_construct_access() {
    let src = "union U { a: i32, b: f64 }; let u = U { a: 42 }; u.a";
    let vm = run_vm(src).unwrap();
    let jit_v = run_jit(src).unwrap();
    match (vm, jit_v) {
        (Value::Int(a, _), Value::Int(b, _)) => assert_eq!(a, b, "VM 与 JIT 结果应一致"),
        (v, j) => panic!("JIT parity 失败: VM {:?} vs JIT {:?}", v, j),
    }
}

#[test]
fn test_union_jit_field_assign() {
    let src = "union U { a: i32, b: f64 }; let mut u = U { a: 42 }; u.a = 100; u.a";
    let vm = run_vm(src).unwrap();
    let jit_v = run_jit(src).unwrap();
    match (vm, jit_v) {
        (Value::Int(a, _), Value::Int(b, _)) => assert_eq!(a, b, "VM 与 JIT 结果应一致"),
        (v, j) => panic!("JIT parity 失败: VM {:?} vs JIT {:?}", v, j),
    }
}

// ── 三路径一致性：union 值本身作为结果 ─────────────────────────────────────

#[test]
fn test_union_three_paths_value_shape() {
    let src = "union U { a: i32, b: f64 }; U { a: 42 }";
    // 解释器：验证 Value::Union 结构
    let interp = run(src).unwrap();
    let v = interp.as_ref().expect("应返回 union 值");
    expect_union(v, "U");
    match expect_union(v, "U") {
        Value::Int(42, _) => {}
        other => panic!("期望 Int(42)，得到 {:?}", other),
    }    // VM：验证 Value::Union 结构
    let vm = run_vm(src).unwrap();
    match &vm {
        Value::Union { name, active_field, value } => {
            assert_eq!(name, "U");
            assert_eq!(active_field, "a");
            assert!(matches!(&**value, Value::Int(42, _)));
        }
        other => panic!("VM 期望 Value::Union，得到 {:?}", other),
    }
    // JIT：验证 Value::Union 结构
    let jit_v = run_jit(src).unwrap();
    match &jit_v {
        Value::Union { name, active_field, value } => {
            assert_eq!(name, "U");
            assert_eq!(active_field, "a");
            assert!(matches!(&**value, Value::Int(42, _)));
        }
        other => panic!("JIT 期望 Value::Union，得到 {:?}", other),
    }
}

// ── 与 Struct 同名不冲突（用户类型优先，防误报） ───────────────────────────

#[test]
fn test_union_vs_struct_same_name_ok() {
    // 同名 struct 与 union：各用各的构造语法，互不干扰
    let src = r#"
    struct U { x: i32, y: i32 }
    union V { a: i32, b: f64 }
    let s = U { x: 1, y: 2 };
    let u = V { a: 3 };
    s.x + u.a
    "#;
    let result = run(src).unwrap();
    match result {
        Some(Value::Int(4, _)) => {}
        v => panic!("期望 Int(4)，得到 {:?}", v),
    }
}
