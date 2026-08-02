//! 批次2 C：VM 具体值 trait 方法分派 —— 三路径（VM/JIT/解释器）对拍测试。
//!
//! 机制（编译器侧编译期改写，`lower_expr.rs`）：
//! - 具体值（`Circle { .. }`）上调用 trait 方法（`c.area()`）：MethodCall 分支
//!   按 receiver 静态类型查 `trait_impls`，**恰好 1 个** trait 命中 → 改写为对
//!   `__dyn_{trait}_{type}_{method}` 的普通 Call（与 inherent `__Type_method`
//!   改写同模式），VM/JIT/WASM/解释器四路径同通；
//! - 运算符重载（`a + b` → `a.add(b)`）降级处直接构造 MethodCall 绕过 MethodCall
//!   分支，按已知 trait（Add→add 固定映射）同样改写为 `__dyn_*` Call；
//! - **0 无匹配 / ≥2 歧义 → 不改写**：保持既有 fall-through 响亮报错，不静默选一个。
//!
//! 覆盖：
//! ① trait_demo 等价（area/name 多类型）三路径对拍
//! ② 运算符链式 `(a+b)+c` / `a+b+c+d` 三路径对拍
//! ③ 多 trait 多类型矩阵三路径对拍
//! ④ 方法与字段同名（`c.name()` 走 trait 方法，`c.name` 走 Field，三路径一致）
//! ⑤ 无 trait 匹配仍响亮报错「没有方法」/「不支持方法」
//! ⑥ 歧义（两 trait 同名方法）不改写仍响亮报错（VM；解释器维持既有 HashMap 序）
//! ⑦ 改写结构性断言：lower 后 main 中含 `__dyn_*` Call（而非 plain MethodCall）
//! ⑧ WASM 编译 smoke（改写产物 Call 可被 Rust 侧 WASM 后端编译）

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

// ── ① trait_demo 等价：具体值 trait 方法分派 三路径对拍 ─────────────────

#[test]
fn test_concrete_trait_dispatch_three_paths() {
    // 与 `Tenth实例/Trait示例/trait_demo.th` 同构：Circle/Rectangle 实现
    // Shape（area/name），具体值直接调用 trait 方法。
    let src = r#"
        struct Circle { radius: f64 }
        struct Rectangle { width: f64, height: f64 }
        trait Shape {
            fn area(self) -> f64;
            fn name(self) -> string;
        }
        impl Shape for Circle {
            fn area(self) -> f64 { 3.14159 * self.radius * self.radius }
            fn name(self) -> string { "Circle" }
        }
        impl Shape for Rectangle {
            fn area(self) -> f64 { self.width * self.height }
            fn name(self) -> string { "Rectangle" }
        }
        let c = Circle { radius: 5.0 };
        let r = Rectangle { width: 4.0, height: 6.0 };
        c.area() + r.area()
    "#;
    // 78.53975 + 24.0 = 102.53975
    let interp = run(src).unwrap();
    let vm = run_vm(src).unwrap();
    let jit = run_jit(src).unwrap();
    match (&interp, &vm, &jit) {
        (Some(Value::Float(i)), Value::Float(v), Value::Float(j)) => {
            assert!((i - 102.53975).abs() < 1e-6, "interp 期望 102.53975，实际 {}", i);
            assert!((i - v).abs() < 1e-9, "interp/VM 不一致: {} vs {}", i, v);
            assert!((i - j).abs() < 1e-9, "interp/JIT 不一致: {} vs {}", i, j);
        }
        (i, v, j) => panic!("unexpected: interp={:?} vm={:?} jit={:?}", i, v, j),
    }
}

#[test]
fn test_concrete_trait_string_method_parity() {
    // 返回 string 的 trait 方法（name），三路径结果一致（解释器为语义基准）。
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
        c.name() + ":" + s.name()
    "#;
    let interp = run(src).unwrap();
    let vm = run_vm(src).unwrap();
    let jit = run_jit(src).unwrap();
    match (&interp, &vm, &jit) {
        (Some(Value::String(i)), Value::String(v), Value::String(j)) => {
            assert_eq!(i, "circle:square");
            assert_eq!(i, v, "interp/VM 不一致");
            assert_eq!(i, j, "interp/JIT 不一致");
        }
        (i, v, j) => panic!("unexpected: interp={:?} vm={:?} jit={:?}", i, v, j),
    }
}

// ── ② 运算符重载链式：三路径对拍 ────────────────────────────────────────

#[test]
fn test_operator_chained_three_paths() {
    // 与 `Tenth实例/宏与自定义运算符/operator_overload.th` 同构：
    // `a + b` 降级为 `a.add(b)`，编译期按已知 trait（Add）改写为 `__dyn_*`。
    let src = r#"
        struct Point { x: f64 }
        trait Add { fn add(self, other: Point) -> Point; }
        impl Add for Point {
            fn add(self, other: Point) -> Point { Point { x: self.x + other.x } }
        }
        let a = Point { x: 1.5 };
        let b = Point { x: 2.5 };
        let c = Point { x: 1.0 };
        let d = Point { x: 2.0 };
        let f = (a + b) + c;      // (1.5+2.5)+1.0 = 5.0
        let g = a + b + c + d;    // 左结合：1.5+2.5+1.0+2.0 = 7.0
        f.x + g.x
    "#;
    // 5.0 + 7.0 = 12.0
    let interp = run(src).unwrap();
    let vm = run_vm(src).unwrap();
    let jit = run_jit(src).unwrap();
    match (&interp, &vm, &jit) {
        (Some(Value::Float(i)), Value::Float(v), Value::Float(j)) => {
            assert!((i - 12.0).abs() < 1e-9, "interp 期望 12.0，实际 {}", i);
            assert!((i - v).abs() < 1e-9, "interp/VM 不一致: {} vs {}", i, v);
            assert!((i - j).abs() < 1e-9, "interp/JIT 不一致: {} vs {}", i, j);
        }
        (i, v, j) => panic!("unexpected: interp={:?} vm={:?} jit={:?}", i, v, j),
    }
}

#[test]
fn test_operator_unary_three_paths() {
    // 一元运算符重载：`-a` → `a.neg()` → `__dyn_Neg_Point_neg`。
    let src = r#"
        struct Point { x: f64 }
        trait Neg { fn neg(self) -> Point; }
        impl Neg for Point {
            fn neg(self) -> Point { Point { x: -self.x } }
        }
        let a = Point { x: 3.0 };
        let b = -a;
        let c = -(-a);
        b.x + c.x
    "#;
    // -3.0 + 3.0 = 0.0
    let interp = run(src).unwrap();
    let vm = run_vm(src).unwrap();
    let jit = run_jit(src).unwrap();
    match (&interp, &vm, &jit) {
        (Some(Value::Float(i)), Value::Float(v), Value::Float(j)) => {
            assert!((i - 0.0).abs() < 1e-9, "interp 期望 0.0，实际 {}", i);
            assert!((i - v).abs() < 1e-9, "interp/VM 不一致: {} vs {}", i, v);
            assert!((i - j).abs() < 1e-9, "interp/JIT 不一致: {} vs {}", i, j);
        }
        (i, v, j) => panic!("unexpected: interp={:?} vm={:?} jit={:?}", i, v, j),
    }
}

// ── ③ 多 trait 多类型矩阵：三路径对拍 ───────────────────────────────────

#[test]
fn test_multi_trait_multi_type_matrix() {
    // 两个类型各实现两个 trait（area 在 Shape、label 在 Describe），
    // 同一类型上多个 trait 方法互不干扰（各恰一 trait 命中）。
    let src = r#"
        struct Circle { radius: f64 }
        struct Square { side: f64 }
        trait Shape { fn area(self) -> f64; }
        trait Describe { fn label(self) -> string; }
        impl Shape for Circle { fn area(self) -> f64 { 3.0 * self.radius * self.radius } }
        impl Shape for Square { fn area(self) -> f64 { self.side * self.side } }
        impl Describe for Circle { fn label(self) -> string { "circle" } }
        impl Describe for Square { fn label(self) -> string { "square" } }
        let c = Circle { radius: 2.0 };
        let s = Square { side: 3.0 };
        let a = c.area() + s.area();   // 12 + 9 = 21
        let l = c.label() + ":" + s.label();
        a
    "#;
    let interp = run(src).unwrap();
    let vm = run_vm(src).unwrap();
    let jit = run_jit(src).unwrap();
    match (&interp, &vm, &jit) {
        (Some(Value::Float(i)), Value::Float(v), Value::Float(j)) => {
            assert!((i - 21.0).abs() < 1e-9, "interp 期望 21.0，实际 {}", i);
            assert!((i - v).abs() < 1e-9, "interp/VM 不一致: {} vs {}", i, v);
            assert!((i - j).abs() < 1e-9, "interp/JIT 不一致: {} vs {}", i, j);
        }
        (i, v, j) => panic!("unexpected: interp={:?} vm={:?} jit={:?}", i, v, j),
    }
}

#[test]
fn test_multi_trait_string_matrix_parity() {
    let src = r#"
        struct Circle { radius: f64 }
        struct Square { side: f64 }
        trait Shape { fn area(self) -> f64; }
        trait Describe { fn label(self) -> string; }
        impl Shape for Circle { fn area(self) -> f64 { 3.0 * self.radius * self.radius } }
        impl Shape for Square { fn area(self) -> f64 { self.side * self.side } }
        impl Describe for Circle { fn label(self) -> string { "circle" } }
        impl Describe for Square { fn label(self) -> string { "square" } }
        let c = Circle { radius: 2.0 };
        let s = Square { side: 3.0 };
        c.label() + ":" + s.label()
    "#;
    let interp = run(src).unwrap();
    let vm = run_vm(src).unwrap();
    let jit = run_jit(src).unwrap();
    match (&interp, &vm, &jit) {
        (Some(Value::String(i)), Value::String(v), Value::String(j)) => {
            assert_eq!(i, "circle:square");
            assert_eq!(i, v, "interp/VM 不一致");
            assert_eq!(i, j, "interp/JIT 不一致");
        }
        (i, v, j) => panic!("unexpected: interp={:?} vm={:?} jit={:?}", i, v, j),
    }
}

// ── ④ 方法与字段同名：方法调用走 trait 方法，字段访问走 Field ──────────

#[test]
fn test_method_and_field_same_name() {
    // `Circle` 同时有字段 `name` 与 trait 方法 `name`：
    // - `c.name()`（方法调用）→ 编译期改写 → `__dyn_Shape_Circle_name` → "method-name"
    //   （与解释器一致：解释器 Value::Struct 分支先查方法表/遍历 trait_impls）
    // - `c.name`（Field 表达式）→ 字段访问 → "field-name"
    // 三路径一致，无静默错值。
    let src = r#"
        struct Circle { name: string, radius: f64 }
        trait Shape {
            fn name(self) -> string;
            fn area(self) -> f64;
        }
        impl Shape for Circle {
            fn name(self) -> string { "method-name" }
            fn area(self) -> f64 { 3.0 * self.radius * self.radius }
        }
        let c = Circle { name: "field-name", radius: 2.0 };
        let mn = c.name();
        let fv = c.name;
        mn + ":" + fv
    "#;
    let interp = run(src).unwrap();
    let vm = run_vm(src).unwrap();
    let jit = run_jit(src).unwrap();
    match (&interp, &vm, &jit) {
        (Some(Value::String(i)), Value::String(v), Value::String(j)) => {
            assert_eq!(i, "method-name:field-name");
            assert_eq!(i, v, "interp/VM 不一致");
            assert_eq!(i, j, "interp/JIT 不一致");
        }
        (i, v, j) => panic!("unexpected: interp={:?} vm={:?} jit={:?}", i, v, j),
    }
}

// ── ⑤ 无 trait 匹配：仍响亮报错（不改写、不静默） ───────────────────────

#[test]
fn test_no_trait_match_still_errors() {
    // 无任何 trait/inherent 实现该方法 → 改写不命中 → fall-through 响亮报错。
    let src = r#"
        struct Circle { radius: f64 }
        let c = Circle { radius: 2.0 };
        c.nonexistent()
    "#;
    let interp_err = run(src).unwrap_err();
    assert!(
        interp_err.contains("未知的方法") || interp_err.contains("没有方法") || interp_err.contains("不支持方法"),
        "解释器应响亮报错，实际: {}", interp_err
    );
    let vm_err = run_vm(src).unwrap_err();
    assert!(
        vm_err.contains("没有方法"),
        "VM 应响亮报错「没有方法」，实际: {}", vm_err
    );
}

// ── ⑥ 歧义：两 trait 同名方法 → 不改写（VM 响亮报错，不静默选一个） ────

#[test]
fn test_ambiguous_two_traits_same_method_vm_errors() {
    // 两 trait 同名方法、同一类型都实现 → 恰 2 命中 → 不改写。
    // VM：fall-through → call_method_priv 字段访问 → 响亮报错「没有方法」。
    // 解释器：维持既有行为（HashMap 序任选其一，非本任务范围），此处不断言。
    let src = r#"
        struct Circle { radius: f64 }
        trait A { fn m(self) -> f64; }
        trait B { fn m(self) -> f64; }
        impl A for Circle { fn m(self) -> f64 { 1.0 } }
        impl B for Circle { fn m(self) -> f64 { 2.0 } }
        let c = Circle { radius: 1.0 };
        c.m()
    "#;
    let vm_err = run_vm(src).unwrap_err();
    assert!(
        vm_err.contains("没有方法"),
        "VM 歧义应响亮报错（不静默选一个），实际: {}", vm_err
    );
    // 结构断言：lower 后 main 中不应出现 __dyn_ 改写（歧义不改写）
    let calls = collect_dyn_calls(src).unwrap();
    assert!(
        calls.is_empty(),
        "歧义场景不应改写为 __dyn_*，实际: {:?}", calls
    );
}

// ── ⑦ 改写结构性断言：lower 后 main 含 __dyn_* Call ─────────────────────

#[test]
fn test_rewrite_produces_dyn_call_structurally() {
    // 直接证明编译期改写发生：`c.area()` 在 lower 后是
    // Call("__dyn_Shape_Circle_area")，而非 plain MethodCall("area")。
    let src = r#"
        struct Circle { radius: f64 }
        trait Shape { fn area(self) -> f64; }
        impl Shape for Circle { fn area(self) -> f64 { 3.0 * self.radius * self.radius } }
        let c = Circle { radius: 2.0 };
        c.area()
    "#;
    let calls = collect_dyn_calls(src).unwrap();
    assert_eq!(calls, vec!["__dyn_Shape_Circle_area".to_string()],
        "期望改写为 __dyn_Shape_Circle_area，实际: {:?}", calls);
}

#[test]
fn test_rewrite_operator_produces_dyn_call_structurally() {
    // 运算符重载降级处同样改写：`a + b` → Call("__dyn_Add_Point_add")。
    let src = r#"
        struct Point { x: f64 }
        trait Add { fn add(self, other: Point) -> Point; }
        impl Add for Point {
            fn add(self, other: Point) -> Point { Point { x: self.x + other.x } }
        }
        let a = Point { x: 1.5 };
        let b = Point { x: 2.5 };
        a + b
    "#;
    let calls = collect_dyn_calls(src).unwrap();
    assert_eq!(calls, vec!["__dyn_Add_Point_add".to_string()],
        "期望改写为 __dyn_Add_Point_add，实际: {:?}", calls);
}

// ── ⑧ WASM 编译 smoke：改写产物 Call 可被 Rust 侧 WASM 后端编译 ─────────

#[test]
fn test_wasm_compile_smoke() {
    // 改写产物是普通 Call("__dyn_*")，与 inherent `__Type_method` 改写同构，
    // 应可被 Rust 侧 WASM 后端编译（编译 smoke，不运行）。
    let src = r#"
        struct Circle { radius: f64 }
        trait Shape { fn area(self) -> f64; }
        impl Shape for Circle { fn area(self) -> f64 { 3.0 * self.radius * self.radius } }
        fn main() -> f64 {
            let c = Circle { radius: 2.0 };
            c.area()
        }
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

// ── HIR 结构遍历：收集 main 中 `__dyn_*` 调用名 ──────────────────────────

/// lower 后收集 main（`<expr>` 或 `fn main` 函数体）中所有 `__dyn_*` 调用名。
fn collect_dyn_calls(src: &str) -> Result<Vec<String>, String> {
    let mut lexer = Lexer::new(src);
    let tokens = lexer.tokenize().map_err(|e| e.to_string())?;
    let mut parser = Parser::new(tokens);
    let program = parser.parse_program().map_err(|e| e.to_string())?;
    let mut lowerer = Lowerer::new();
    let hir = lowerer.lower_program(&program).map_err(|e| e.to_string())?;
    let mut out = Vec::new();
    if let Some(main) = &hir.main_expr {
        walk_expr(main, &mut out);
    }
    // `fn main() {}` 形式的程序：main 体在 functions 表
    for f in &hir.functions {
        if f.name == "main" {
            walk_expr(&f.body, &mut out);
        }
    }
    Ok(out)
}

fn walk_expr(e: &HirExpr, out: &mut Vec<String>) {
    match &e.kind {
        HirExprKind::Call { func, args, .. } => {
            if let HirExprKind::Var(name) = &func.kind {
                if name.starts_with("__dyn_") {
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
            // 未改写（歧义/无匹配）的 plain MethodCall：递归遍历 receiver/args
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
        HirStmtKind::Loop { body, .. } => {
            for s in body { walk_stmt(s, out); }
        }
        _ => {}
    }
}
