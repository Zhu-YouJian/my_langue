//! 批次2 C P3：泛型 `<T>` 内 trait/inherent 方法运行时兜底 —— 三路径对拍测试。
//!
//! 机制（编译器侧编译期改写，`lower_expr.rs` GenericCall 分支 + `rewrite_inst_method_calls`）：
//! - 泛型模板 body 在定义时以 `TypeParam("T")` 类型 lower——MethodCall 分支查
//!   `trait_impls`（键是具体类型名）不命中，保持 plain MethodCall；
//! - 实例化时 `substitute_expr` 已把 TypeParam 替换为具体类型，在实例化点对
//!   body 递归应用与普通 MethodCall 分支**同源**的改写规则：
//!   ① inherent 优先（`__{Type}_{method}` mangled 函数存在 → 普通 Call）；
//!   ② trait 其次（`try_rewrite_trait_method` 恰一 trait 命中 → `__dyn_*` Call）；
//!   ③ 都不命中 → 保持 MethodCall fall-through（无匹配/歧义响亮报错，不静默）。
//! - 使 VM 与解释器一致（解释器按运行时值类型查表本就可用，VM 此前报「没有方法」）。
//!
//! 覆盖：
//! ① 泛型函数内 trait 方法（Circle/Rect 双 impl）三路径对拍
//! ② 泛型 + inherent 方法三路径对拍（同机制，顺带补齐 VM 缺口）
//! ③ 泛型函数体内嵌套（if/block/let 内 trait 方法）三路径对拍（walker 递归）
//! ④ 无 trait 匹配 → VM/解释器均响亮报错（不静默错值）
//! ⑤ 歧义（两 trait 同名方法）→ VM 响亮报错不静默；解释器维持既有 HashMap 序
//! ⑥ 改写结构性断言：实例化函数体（`area_of_Circle`）含 `__dyn_*` Call
//! ⑦ WASM 编译 smoke（改写产物 Call 可被 Rust 侧 WASM 后端编译）

use tenth::lexer::lexer::Lexer;
use tenth::parser::parser::Parser;
use tenth::hir::lower::Lowerer;
use tenth::hir::hir::{HirExpr, HirExprKind, HirStmt, HirStmtKind};
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

fn expect_float(v: &Value, expected: f64, tag: &str) {
    match v {
        Value::Float(n) => assert!((n - expected).abs() < 1e-9, "{}: 期望 {}，实际 {}", tag, expected, n),
        other => panic!("{}: 期望 Float({})，实际 {:?}", tag, expected, other),
    }
}

fn expect_str(v: &Value, expected: &str, tag: &str) {
    match v {
        Value::String(s) => assert_eq!(s.as_str(), expected, "{}: 期望 {:?}，实际 {:?}", tag, expected, s.as_str()),
        other => panic!("{}: 期望 Str，实际 {:?}", tag, other),
    }
}

/// 三路径对拍：解释器（语义基准）与 VM/JIT 结果一致。
fn assert_three_paths_float(src: &str, expected: f64, tag: &str) {
    let interp = run(src).unwrap();
    let vm = run_vm(src).unwrap();
    let jit = run_jit(src).unwrap();
    match (&interp, &vm, &jit) {
        (Some(Value::Float(i)), Value::Float(v), Value::Float(j)) => {
            assert!((i - expected).abs() < 1e-9, "{}: interp 期望 {}，实际 {}", tag, expected, i);
            assert!((i - v).abs() < 1e-9, "{}: interp/VM 不一致: {} vs {}", tag, i, v);
            assert!((i - j).abs() < 1e-9, "{}: interp/JIT 不一致: {} vs {}", tag, i, j);
        }
        (i, v, j) => panic!("{}: unexpected interp={:?} vm={:?} jit={:?}", tag, i, v, j),
    }
}

// ── ① 泛型函数内 trait 方法：Circle/Rect 双 impl 三路径对拍 ────────────

#[test]
fn test_generic_trait_method_three_paths() {
    // 泛型 `area_of<T>` 内调用 `x.area()`，实例化 `area_of<Circle>` /
    // `area_of<Rect>` 时 T 的具体类型各有恰一 Area impl → 改写为
    // `__dyn_Area_Circle_area` / `__dyn_Area_Rect_area`，三路径同通。
    let src = r#"
        struct Circle { radius: f64 }
        struct Rect { width: f64, height: f64 }
        trait Area {
            fn area(self) -> f64;
        }
        impl Area for Circle {
            fn area(self) -> f64 { 3.14159 * self.radius * self.radius }
        }
        impl Area for Rect {
            fn area(self) -> f64 { self.width * self.height }
        }
        fn area_of<T>(x: T) -> f64 {
            x.area()
        }
        let c = Circle { radius: 5.0 };
        let r = Rect { width: 4.0, height: 6.0 };
        area_of<Circle>(c) + area_of<Rect>(r)
    "#;
    // 78.53975 + 24.0 = 102.53975
    assert_three_paths_float(src, 102.53975, "generic trait 双 impl");
}

#[test]
fn test_generic_trait_string_method_parity() {
    // 泛型 + 返回 string 的 trait 方法。
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
        fn name_of<T>(x: T) -> string {
            x.name()
        }
        let c = Circle { radius: 1.0 };
        let s = Square { side: 2.0 };
        name_of<Circle>(c) + "-" + name_of<Square>(s)
    "#;
    let interp = run(src).unwrap();
    let vm = run_vm(src).unwrap();
    let jit = run_jit(src).unwrap();
    match (&interp, &vm, &jit) {
        (Some(Value::String(i)), Value::String(v), Value::String(j)) => {
            assert_eq!(i.as_str(), "circle-square", "interp 期望 circle-square，实际 {}", i.as_str());
            assert_eq!(i.as_str(), v.as_str(), "interp/VM 不一致: {} vs {}", i, v);
            assert_eq!(i.as_str(), j.as_str(), "interp/JIT 不一致: {} vs {}", i, j);
        }
        (i, v, j) => panic!("unexpected: interp={:?} vm={:?} jit={:?}", i, v, j),
    }
}

// ── ② 泛型 + inherent 方法三路径对拍（同机制，顺带补齐 VM 缺口）────────

#[test]
fn test_generic_inherent_method_three_paths() {
    // 泛型函数内调用 inherent 方法：实例化处按 `__{Type}_{method}` 改写
    // （inherent 优先于 trait，与普通 MethodCall 分支同源）。
    let src = r#"
        struct Circle { radius: f64 }
        impl Circle {
            fn desc(self) -> string { "inherent-circle" }
        }
        fn describe<T>(x: T) -> string {
            x.desc()
        }
        let c = Circle { radius: 5.0 };
        describe<Circle>(c)
    "#;
    let interp = run(src).unwrap();
    let vm = run_vm(src).unwrap();
    let jit = run_jit(src).unwrap();
    match (&interp, &vm, &jit) {
        (Some(Value::String(i)), Value::String(v), Value::String(j)) => {
            assert_eq!(i.as_str(), "inherent-circle");
            assert_eq!(i.as_str(), v.as_str(), "interp/VM 不一致: {} vs {}", i, v);
            assert_eq!(i.as_str(), j.as_str(), "interp/JIT 不一致: {} vs {}", i, j);
        }
        (i, v, j) => panic!("unexpected: interp={:?} vm={:?} jit={:?}", i, v, j),
    }
}

// ── ③ 泛型函数体内嵌套（if/block/let 内 trait 方法）——walker 递归 ─────

#[test]
fn test_generic_trait_method_nested_control_flow() {
    // 实例化 body 内含 if/block/let：walk 递归改写，三路径一致。
    let src = r#"
        struct Circle { radius: f64 }
        struct Rect { width: f64, height: f64 }
        trait Area {
            fn area(self) -> f64;
        }
        impl Area for Circle {
            fn area(self) -> f64 { 3.14159 * self.radius * self.radius }
        }
        impl Area for Rect {
            fn area(self) -> f64 { self.width * self.height }
        }
        fn area_if<T>(x: T, big: bool) -> f64 {
            if big {
                let a = x.area();
                a * 2.0
            } else {
                x.area() + 1.0
            }
        }
        let c = Circle { radius: 5.0 };
        let r = Rect { width: 4.0, height: 6.0 };
        area_if<Circle>(c, true) + area_if<Rect>(r, false)
    "#;
    // (78.53975 * 2) + (24.0 + 1.0) = 157.0795 + 25.0 = 182.0795
    assert_three_paths_float(src, 182.0795, "generic trait 嵌套控制流");
}

// ── ④ 无 trait 匹配 → VM/解释器均响亮报错（不静默错值）────────────────

#[test]
fn test_generic_trait_no_match_loud_error() {
    // Square 未实现 Area → 实例化 `area_of<Square>` 不改写，保持 fall-through，
    // VM 与解释器均响亮报错（文案不同：VM「没有方法」/ 解释器「未知的方法」）。
    let src = r#"
        struct Circle { radius: f64 }
        struct Square { side: f64 }
        trait Area {
            fn area(self) -> f64;
        }
        impl Area for Circle {
            fn area(self) -> f64 { 3.14159 * self.radius * self.radius }
        }
        fn area_of<T>(x: T) -> f64 {
            x.area()
        }
        let s = Square { side: 2.0 };
        area_of<Square>(s)
    "#;
    let interp = run(src).expect_err("解释器应报错（无匹配方法）");
    let vm = run_vm(src).expect_err("VM 应报错（无匹配方法）");
    assert!(interp.contains("area"), "解释器报错应提及方法名，实际: {}", interp);
    assert!(vm.contains("area"), "VM 报错应提及方法名，实际: {}", vm);
}

// ── ⑤ 歧义（两 trait 同名方法）→ VM 响亮报错不静默；解释器维持既有序 ──

#[test]
fn test_generic_trait_ambiguous_no_silent_dispatch() {
    // Circle 同时实现 Area 与 Draw 的 `area` → 实例化 `area_of<Circle>` 时
    // try_rewrite_trait_method 命中 2 个 trait → 不改写（不静默选一个）。
    // VM 报「没有方法」（响亮，不静默错值）；解释器维持既有 HashMap 序
    // （与批次2 C 具体值歧义场景同行为，非本任务引入）。
    let src = r#"
        struct Circle { radius: f64 }
        trait Area {
            fn area(self) -> f64;
        }
        trait Draw {
            fn area(self) -> f64;
        }
        impl Area for Circle {
            fn area(self) -> f64 { 3.14159 * self.radius * self.radius }
        }
        impl Draw for Circle {
            fn area(self) -> f64 { 99.0 }
        }
        fn area_of<T>(x: T) -> f64 {
            x.area()
        }
        let c = Circle { radius: 5.0 };
        area_of<Circle>(c)
    "#;
    // VM：响亮报错，不静默选一个。
    let vm = run_vm(src).expect_err("VM 应报错（歧义不静默选一个）");
    assert!(vm.contains("area"), "VM 报错应提及方法名，实际: {}", vm);
    // 解释器：维持既有 HashMap 序（可能返回某个 impl，但这是既有行为，非静默错值引入）。
    let _ = run(src).expect("解释器维持既有 HashMap 序行为");
}

// ── ⑥ 改写结构性断言：实例化函数体含 `__dyn_*` Call ─────────────────

/// lower 后收集指定函数（含泛型实例化 mangled 名）body 中所有 `__dyn_*` 调用名。
fn collect_dyn_calls_in_fn(src: &str, fn_name: &str) -> Result<Vec<String>, String> {
    let mut lexer = Lexer::new(src);
    let tokens = lexer.tokenize().map_err(|e| e.to_string())?;
    let mut parser = Parser::new(tokens);
    let program = parser.parse_program().map_err(|e| e.to_string())?;
    let mut lowerer = Lowerer::new();
    let hir = lowerer.lower_program(&program).map_err(|e| e.to_string())?;
    let mut out = Vec::new();
    for f in &hir.functions {
        if f.name == fn_name {
            walk_expr(&f.body, &mut out);
        }
    }
    Ok(out)
}

fn walk_expr(e: &HirExpr, out: &mut Vec<String>) {
    match &e.kind {
        HirExprKind::Call { func, args, .. } => {
            if let HirExprKind::Var(name) = &func.kind {
                // 收集所有 mangled 改写调用：trait 改写 `__dyn_*` 与
                // inherent 改写 `__{Type}_{method}` 均以 `__` 开头。
                if name.starts_with("__") {
                    out.push(name.clone());
                }
            }
            for a in args { walk_expr(a, out); }
        }
        HirExprKind::Block { stmts, final_expr } => {
            for s in stmts { walk_stmt(s, out); }
            if let Some(fe) = final_expr { walk_expr(fe, out); }
        }
        HirExprKind::If { then_branch, else_branch, .. } => {
            walk_expr(then_branch, out);
            if let Some(eb) = else_branch { walk_expr(eb, out); }
        }
        HirExprKind::Match { arms, .. } => {
            for arm in arms { walk_expr(&arm.body, out); }
        }
        HirExprKind::MethodCall { receiver, args, .. } => {
            walk_expr(receiver, out);
            for a in args { walk_expr(a, out); }
        }
        _ => {}
    }
}

fn walk_stmt(s: &HirStmt, out: &mut Vec<String>) {
    match &s.kind {
        HirStmtKind::Let { init, .. } => {
            if let Some(v) = init { walk_expr(v, out); }
        }
        HirStmtKind::Expr(e) => walk_expr(e, out),
        HirStmtKind::Return(Some(e)) => walk_expr(e, out),
        HirStmtKind::While { cond, body, .. } => {
            walk_expr(cond, out);
            walk_stmt(body, out);
        }
        HirStmtKind::For { iter, body, .. } => {
            walk_expr(iter, out);
            walk_stmt(body, out);
        }
        _ => {}
    }
}

#[test]
fn test_generic_inst_body_structural_rewrite() {
    // lower 后，实例化函数 `area_of_Circle` 的 body 应含 Call("__dyn_Area_Circle_area")
    // （而非 plain MethodCall）——证明 P3 改写发生在实例化点，VM 因此可用。
    let src = r#"
        struct Circle { radius: f64 }
        trait Area {
            fn area(self) -> f64;
        }
        impl Area for Circle {
            fn area(self) -> f64 { 3.14159 * self.radius * self.radius }
        }
        fn area_of<T>(x: T) -> f64 {
            x.area()
        }
        let c = Circle { radius: 5.0 };
        area_of<Circle>(c)
    "#;
    let calls = collect_dyn_calls_in_fn(src, "area_of_Circle")
        .expect("lower 应成功");
    assert_eq!(calls, vec!["__dyn_Area_Circle_area".to_string()],
        "实例化函数体应改写为 __dyn_Area_Circle_area，实际: {:?}", calls);
}

#[test]
fn test_generic_inherent_inst_body_structural_rewrite() {
    // inherent 优先：实例化函数 `describe_Circle` body 应含 Call("__Circle_desc")。
    let src = r#"
        struct Circle { radius: f64 }
        impl Circle {
            fn desc(self) -> string { "inherent-circle" }
        }
        fn describe<T>(x: T) -> string {
            x.desc()
        }
        let c = Circle { radius: 5.0 };
        describe<Circle>(c)
    "#;
    let mut lexer = Lexer::new(src);
    let tokens = lexer.tokenize().expect("lex");
    let mut parser = Parser::new(tokens);
    let program = parser.parse_program().expect("parse");
    let mut lowerer = Lowerer::new();
    let hir = lowerer.lower_program(&program).expect("lower");
    let mut found = false;
    for f in &hir.functions {
        if f.name == "describe_Circle" {
            found = true;
            let mut out = Vec::new();
            walk_expr(&f.body, &mut out);
            assert!(out.iter().any(|n| n == "__Circle_desc"),
                "实例化函数体应含 __Circle_desc Call，实际: {:?}", out);
        }
    }
    assert!(found, "应存在实例化函数 describe_Circle");
}

// ── ⑦ WASM 编译 smoke ────────────────────────────────────────────────

#[test]
fn test_generic_trait_wasm_compile_smoke() {
    // 改写产物是普通 Call("__dyn_*")，应可被 Rust 侧 WASM 后端编译。
    let src = r#"
        struct Circle { radius: f64 }
        trait Area {
            fn area(self) -> f64;
        }
        impl Area for Circle {
            fn area(self) -> f64 { 3.14159 * self.radius * self.radius }
        }
        fn area_of<T>(x: T) -> f64 {
            x.area()
        }
        let c = Circle { radius: 5.0 };
        area_of<Circle>(c)
    "#;
    let mut lexer = Lexer::new(src);
    let tokens = lexer.tokenize().expect("lex");
    let mut parser = Parser::new(tokens);
    let program = parser.parse_program().expect("parse");
    let mut lowerer = Lowerer::new();
    let hir = lowerer.lower_program(&program).expect("lower");
    let wasm = tenth::compile::compile_to_wasm(&hir)
        .expect("WASM 编译应成功（改写产物为普通 Call）");
    assert_eq!(&wasm[..4], b"\0asm", "WASM magic");
}
