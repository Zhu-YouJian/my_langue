//! M1.3 — dyn Trait 动态分发全链路测试
//!
//! 方案 A：运行时按类型分派（复用 trait_impls，无真 vtable）
//! - 值表示：`Value::Dyn { trait_name, type_name, value }`
//! - 升级：类型注解驱动——`let d: dyn Shape = circle;` 自动包装（编译期改写为
//!   `into_dyn(circle, "Shape")`，并做 trait impl 检查——未实现 → 编译期 TypeError）
//! - 方法调用：`d.name()` 按 trait_name + type_name 在 trait_impls 查实现
//!   （解释器：trait_impls 直接查表 + eval HIR body；
//!    VM：通过 lowerer 注册的 `__dyn_{trait}_{type}_{method}` 字节码函数调用）
//! - 边界：dyn 值不可 Copy、不自动 drop（编译期既有处理）
//!
//! 覆盖：声明、升级、动态调用（两个类型行为不同）、字段访问型方法、类型检查
//! （未 impl 报编译错误）、无该方法报运行错误、Display、VM/解释器 parity、JIT 路径。

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

// ── 基础：声明 + 升级 + 动态调用（解释器） ────────────────────────────────

#[test]
fn test_dyn_upgrade_and_dispatch_interp() {
    let src = r#"
        struct Circle { radius: f64 }
        struct Square { side: f64 }
        trait Shape {
            fn name(self) -> string;
            fn area(self) -> f64;
        }
        impl Shape for Circle {
            fn name(self) -> string { "circle" }
            fn area(self) -> f64 { 3.14 * self.radius * self.radius }
        }
        impl Shape for Square {
            fn name(self) -> string { "square" }
            fn area(self) -> f64 { self.side * self.side }
        }
        let c = Circle { radius: 2.0 };
        let s = Square { side: 3.0 };
        let dc: dyn Shape = c;
        let ds: dyn Shape = s;
        dc.name() + ":" + ds.name()
    "#;
    let result = run(src).unwrap();
    match result {
        Some(Value::String(s)) => assert_eq!(s, "circle:square"),
        v => panic!("expected String, got {:?}", v),
    }
}

#[test]
fn test_dyn_field_access_method_interp() {
    let src = r#"
        struct Circle { radius: f64 }
        struct Square { side: f64 }
        trait Shape {
            fn area(self) -> f64;
        }
        impl Shape for Circle {
            fn area(self) -> f64 { 3.14 * self.radius * self.radius }
        }
        impl Shape for Square {
            fn area(self) -> f64 { self.side * self.side }
        }
        let c = Circle { radius: 2.0 };
        let s = Square { side: 3.0 };
        let dc: dyn Shape = c;
        let ds: dyn Shape = s;
        dc.area() + ds.area()
    "#;
    let result = run(src).unwrap();
    match result {
        Some(Value::Float(f)) => assert!((f - (12.56 + 9.0)).abs() < 1e-6, "got {}", f),
        v => panic!("expected Float, got {:?}", v),
    }
}

// ── VM/解释器 parity ──────────────────────────────────────────────────────

#[test]
fn test_dyn_vm_interp_parity() {
    let src = r#"
        struct Circle { radius: f64 }
        struct Square { side: f64 }
        trait Shape {
            fn name(self) -> string;
        }
        impl Shape for Circle {
            fn name(self) -> string { "circle" }
        }
        impl Shape for Square {
            fn name(self) -> string { "square" }
        }
        let c = Circle { radius: 2.0 };
        let s = Square { side: 3.0 };
        let dc: dyn Shape = c;
        let ds: dyn Shape = s;
        dc.name() + ":" + ds.name()
    "#;
    let interp = run(src).unwrap();
    let vm = run_vm(src).unwrap();
    let jit = run_jit(src).unwrap();
    match (interp, &vm, &jit) {
        (Some(Value::String(i)), Value::String(v), Value::String(j)) => {
            assert_eq!(i, "circle:square");
            assert_eq!(i, *v, "interpreter/VM 结果不一致");
            assert_eq!(i, *j, "interpreter/JIT 结果不一致");
        }
        (i, v, j) => panic!("unexpected: interp={:?} vm={:?} jit={:?}", i, v, j),
    }
}

// ── 通过普通函数传 dyn（预先升级好的 dyn 值） ────────────────────────────

#[test]
fn test_dyn_passed_to_function() {
    let src = r#"
        struct Circle { radius: f64 }
        struct Square { side: f64 }
        trait Shape {
            fn name(self) -> string;
        }
        impl Shape for Circle {
            fn name(self) -> string { "circle" }
        }
        impl Shape for Square {
            fn name(self) -> string { "square" }
        }
        fn which(s: dyn Shape) -> string { s.name() }
        let c = Circle { radius: 2.0 };
        let s = Square { side: 3.0 };
        let dc: dyn Shape = c;
        let ds: dyn Shape = s;
        which(dc) + ":" + which(ds)
    "#;
    let result = run(src).unwrap();
    match result {
        Some(Value::String(s)) => assert_eq!(s, "circle:square"),
        v => panic!("expected String, got {:?}", v),
    }
}

// ── 类型检查：未实现 trait 的类型不能升级为 dyn（编译期报错） ─────────────

#[test]
fn test_dyn_upgrade_rejects_missing_impl() {
    let src = r#"
        struct Circle { radius: f64 }
        struct Triangle { base: f64 }
        trait Shape {
            fn name(self) -> string;
        }
        impl Shape for Circle {
            fn name(self) -> string { "circle" }
        }
        let t = Triangle { base: 1.0 };
        let dt: dyn Shape = t;
        dt.name()
    "#;
    let err = run(src).unwrap_err();
    assert!(
        err.contains("未实现 trait") || err.contains("无法升级为 dyn"),
        "expected upgrade type error, got: {}",
        err
    );
}

#[test]
fn test_dyn_upgrade_rejects_missing_impl_vm() {
    let src = r#"
        struct Circle { radius: f64 }
        struct Triangle { base: f64 }
        trait Shape {
            fn name(self) -> string;
        }
        impl Shape for Circle {
            fn name(self) -> string { "circle" }
        }
        let t = Triangle { base: 1.0 };
        let dt: dyn Shape = t;
        dt.name()
    "#;
    // 编译期错误在 lower 阶段产生，VM 路径同样应报错
    let err = run_vm(src).unwrap_err();
    assert!(
        err.contains("未实现 trait") || err.contains("无法升级为 dyn"),
        "expected upgrade type error, got: {}",
        err
    );
}

// ── 运行时错误：dyn 值调用 trait 中不存在的方法 → 报错 ────────────────────

#[test]
fn test_dyn_missing_method_runtime_error() {
    let src = r#"
        struct Circle { radius: f64 }
        trait Shape {
            fn name(self) -> string;
        }
        impl Shape for Circle {
            fn name(self) -> string { "circle" }
        }
        let c = Circle { radius: 2.0 };
        let dc: dyn Shape = c;
        dc.nonexistent()
    "#;
    let err = run(src).unwrap_err();
    assert!(
        err.contains("没有方法 'nonexistent'") || err.contains("没有方法"),
        "expected missing method error, got: {}",
        err
    );
}

#[test]
fn test_dyn_missing_method_runtime_error_vm() {
    let src = r#"
        struct Circle { radius: f64 }
        trait Shape {
            fn name(self) -> string;
        }
        impl Shape for Circle {
            fn name(self) -> string { "circle" }
        }
        let c = Circle { radius: 2.0 };
        let dc: dyn Shape = c;
        dc.nonexistent()
    "#;
    let err = run_vm(src).unwrap_err();
    assert!(
        err.contains("没有方法 'nonexistent'") || err.contains("没有方法"),
        "expected missing method error, got: {}",
        err
    );
}

// ── Display：to_string(dyn 值) ────────────────────────────────────────────

#[test]
fn test_dyn_display() {
    let src = r#"
        struct Circle { radius: f64 }
        trait Shape {
            fn name(self) -> string;
        }
        impl Shape for Circle {
            fn name(self) -> string { "circle" }
        }
        let c = Circle { radius: 2.0 };
        let dc: dyn Shape = c;
        to_string(dc)
    "#;
    let result = run(src).unwrap();
    match result {
        Some(Value::String(s)) => assert!(s.contains("dyn Shape") && s.contains("Circle"), "got {}", s),
        v => panic!("expected String, got {:?}", v),
    }
}

// ── 多个 trait、多个类型（动态分派覆盖矩阵） ──────────────────────────────

#[test]
fn test_dyn_multiple_types_multiple_traits() {
    let src = r#"
        struct Dog { age: i64 }
        struct Cat { age: i64 }
        trait Speak {
            fn speak(self) -> string;
        }
        trait Age {
            fn age_str(self) -> string;
        }
        impl Speak for Dog {
            fn speak(self) -> string { "woof" }
        }
        impl Speak for Cat {
            fn speak(self) -> string { "meow" }
        }
        impl Age for Dog {
            fn age_str(self) -> string { "dog:" + to_string(self.age) }
        }
        impl Age for Cat {
            fn age_str(self) -> string { "cat:" + to_string(self.age) }
        }
        let d = Dog { age: 3 };
        let c = Cat { age: 5 };
        let sd: dyn Speak = d;
        let sc: dyn Speak = c;
        let ad: dyn Age = Dog { age: 3 };
        let ac: dyn Age = Cat { age: 5 };
        sd.speak() + ":" + sc.speak() + ":" + ad.age_str() + ":" + ac.age_str()
    "#;
    let interp = run(src).unwrap();
    let vm = run_vm(src).unwrap();
    match (interp, &vm) {
        (Some(Value::String(i)), Value::String(v)) => {
            assert_eq!(i, "woof:meow:dog:3:cat:5");
            assert_eq!(i, *v, "interpreter/VM 结果不一致");
        }
        (i, v) => panic!("unexpected: interp={:?} vm={:?}", i, v),
    }
}
