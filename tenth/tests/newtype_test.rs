//! M2.2：Newtype（tuple struct）模式测试。
//!
//! 覆盖：
//! - `struct Meters(f64)` 声明 + `Meters(3.5)` 构造（Call → StructLiteral 改写）
//! - 字段访问 `m._0`（tuple struct 字段注册为 `_0, _1, ...`）
//! - 多字段 tuple struct（`struct Pair(i64, str)`）
//! - 函数参数 / 返回值（新类型作为名义类型传递）
//! - 与 named struct 并存
//! - 嵌套 Newtype（`struct Outer(Meters)`）
//! - 类型检查（把裸 f64 传给 Meters 形参 → 编译期错误）
//! - 字段访问语法：`.0` 不支持（应解析失败），须用 `._0`
//! - VM / 解释器 parity

use tenth::lexer::lexer::Lexer;
use tenth::parser::parser::Parser;
use tenth::hir::lower::Lowerer;
use tenth::runtime::interpreter::Interpreter;
use tenth::runtime::vm::Vm;
use tenth::runtime::value::Value;
use tenth::compile::bytecode::BytecodeCompiler;
use std::rc::Rc;
use std::cell::RefCell;

/// 解释器路径。
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

/// VM 路径。
fn run_vm(src: &str) -> Result<Value, String> {
    let mut lexer = Lexer::new(src);
    let tokens = lexer.tokenize().map_err(|e| e.to_string())?;
    let mut parser = Parser::new(tokens);
    let program = parser.parse_program().map_err(|e| e.to_string())?;
    let mut lowerer = Lowerer::new();
    let hir = lowerer.lower_program(&program).map_err(|e| e.to_string())?;

    let mut vm = Vm::new();
    vm.add_native("println".into(), |_vm, args| {
        for a in args { print!("{a}"); }
        println!();
        Ok(Value::Unit)
    });
    vm.add_native("Vec::new".into(), |_vm, _args| {
        Ok(Value::Vec(Rc::new(RefCell::new(Vec::new()))))
    });
    vm.add_native("assert".into(), |_vm, args| {
        if let Some(Value::Bool(b)) = args.first() {
            assert!(*b, "assertion failed");
        }
        Ok(Value::Unit)
    });
    vm.add_native("assert_eq".into(), |_vm, args| {
        if args.len() >= 2 {
            assert_eq!(format!("{}", args[0]), format!("{}", args[1]));
        }
        Ok(Value::Unit)
    });

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

fn expect_float(v: Option<Value>, expected: f64, label: &str) {
    match v {
        Some(Value::Float(n)) => assert!((n - expected).abs() < 1e-10, "{}: 期望 Float({}), 实际 {}", label, expected, n),
        other => panic!("{}: 期望 Float({}), 实际 {:?}", label, expected, other),
    }
}

fn expect_int(v: Option<Value>, expected: i64, label: &str) {
    match v {
        Some(Value::Int(n, _)) => assert_eq!(n, expected, "{}: 期望 Int({}), 实际 {}", label, expected, n),
        other => panic!("{}: 期望 Int({}), 实际 {:?}", label, expected, other),
    }
}

// ── 1. 声明 + 构造 + 字段访问 ──

#[test]
fn newtype_declare_construct_access() {
    let src = "struct Meters(f64); let m = Meters(3.5); m._0";
    expect_float(run(src).unwrap(), 3.5, "构造后访问 ._0");
}

#[test]
fn newtype_vm_parity() {
    let src = "struct Meters(f64); let m = Meters(3.5); m._0";
    match run_vm(src).unwrap() {
        Value::Float(n) => assert!((n - 3.5).abs() < 1e-10, "VM: got {}", n),
        other => panic!("VM: 期望 Float(3.5), 实际 {:?}", other),
    }
}

// ── 2. 多字段 tuple struct ──

#[test]
fn newtype_multi_field() {
    let src = r#"
        struct Pair(i64, str)
        let p = Pair(7, "ab")
        p._0 + p._1.len()
    "#;
    expect_int(run(src).unwrap(), 9, "多字段 ._0 + ._1");
}

// ── 3. 函数参数 / 返回值 ──

#[test]
fn newtype_as_fn_param() {
    let src = r#"
        struct Meters(f64)
        fn dbl(m: Meters) -> f64 { m._0 * 2.0 }
        let m = Meters(3.0)
        dbl(m)
    "#;
    expect_float(run(src).unwrap(), 6.0, "Newtype 作为函数参数");
}

#[test]
fn newtype_as_fn_return() {
    let src = r#"
        struct Meters(f64)
        fn make(x: f64) -> Meters { Meters(x + 1.0) }
        make(2.0)._0
    "#;
    expect_float(run(src).unwrap(), 3.0, "Newtype 作为函数返回值");
}

#[test]
fn newtype_vm_param_return() {
    let src = r#"
        struct Meters(f64)
        fn make(x: f64) -> Meters { Meters(x + 1.0) }
        fn dbl(m: Meters) -> f64 { m._0 * 2.0 }
        dbl(make(2.0))
    "#;
    match run_vm(src).unwrap() {
        Value::Float(n) => assert!((n - 6.0).abs() < 1e-10, "VM: got {}", n),
        other => panic!("VM: 期望 Float(6.0), 实际 {:?}", other),
    }
}

// ── 4. 与 named struct 并存 ──

#[test]
fn newtype_coexists_with_named() {
    let src = r#"
        struct Point { x: f64, y: f64 }
        struct Meters(f64)
        let p = Point { x: 1.0, y: 2.0 }
        let m = Meters(5.0)
        p.x + p.y + m._0
    "#;
    expect_float(run(src).unwrap(), 8.0, "named 与 tuple struct 并存");
}

// ── 5. 嵌套 Newtype ──

#[test]
fn newtype_nested() {
    let src = r#"
        struct Meters(f64)
        struct Length(Meters)
        let l = Length(Meters(4.0))
        l._0._0
    "#;
    expect_float(run(src).unwrap(), 4.0, "嵌套 Newtype 访问");
}

// ── 6. 类型检查：裸 f64 不能传给 Meters 形参 ──

#[test]
fn newtype_type_check_rejects_raw_value() {
    let src = r#"
        struct Meters(f64)
        fn need(m: Meters) -> f64 { m._0 }
        need(5.0)
    "#;
    let result = run(src);
    assert!(result.is_err(), "期望编译期错误：f64 不能传给 Meters 形参");
}

#[test]
fn newtype_type_check_accepts_wrapped() {
    let src = r#"
        struct Meters(f64)
        fn need(m: Meters) -> f64 { m._0 }
        need(Meters(5.0))
    "#;
    expect_float(run(src).unwrap(), 5.0, "Meters 值可传 Meters 形参");
}

// ── 7. 字段访问语法：`.0` 不支持（须用 `._0`）──

#[test]
fn newtype_dot_number_syntax_unsupported() {
    let src = "struct Meters(f64); let m = Meters(3.5); m.0";
    // `.` 后跟数字不是合法字段访问（`.0` 语法未实现，须写 `._0`）
    let result = run(src);
    assert!(result.is_err(), "期望错误：`.0` 语法不支持");
}

// ── 8. Newtype 值参与运算（通过 ._0 解包）──

#[test]
fn newtype_arithmetic_via_unwrap() {
    let src = "struct Meters(f64); let m = Meters(2.5); m._0 + 1.0";
    expect_float(run(src).unwrap(), 3.5, "解包后运算");
}
