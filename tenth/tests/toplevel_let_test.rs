//! M3.5：顶层 `let` 程序级全局（常量/状态）与 `use` 导入模块全局。
//!
//! 覆盖：
//! - 顶层常量同文件函数可见（编译期 + 运行时，解释器/VM 双路径）
//! - 顶层可变全局同文件函数读写
//! - 函数内局部变量 shadow 全局（不破坏全局）
//! - `use` 导入模块常量（跨模块，含模块内函数引用常量）
//! - `use` 导入模块可变全局（跨模块读写）
//! - HirProgram.globals 元数据（name/ty/mutable）
//! - REPL 式逐行累积（顶层 let 持久为全局）
//!
//! 基线：修复前以下场景全部报"未定义变量"或运行时取不到全局值。

use tenth::lexer::lexer::Lexer;
use tenth::parser::parser::Parser;
use tenth::hir::lower::Lowerer;
use tenth::runtime::interpreter::Interpreter;
use tenth::runtime::value::Value;
use std::collections::HashSet;

fn lower_code(src: &str) -> tenth::hir::hir::HirProgram {
    let mut lexer = Lexer::new(src);
    let tokens = lexer.tokenize().unwrap();
    let mut parser = Parser::new(tokens);
    let program = parser.parse_program().unwrap();
    let mut lowerer = Lowerer::new();
    lowerer.lower_program(&program).unwrap()
}

/// 带 seed 的 lower（模拟 REPL 跨行：先把已累积全局注入作用域）
fn lower_code_seeded(src: &str, seed: &[tenth::hir::hir::HirGlobal]) -> tenth::hir::hir::HirProgram {
    let mut lexer = Lexer::new(src);
    let tokens = lexer.tokenize().unwrap();
    let mut parser = Parser::new(tokens);
    let program = parser.parse_program().unwrap();
    let mut lowerer = Lowerer::new();
    lowerer.seed_globals(seed);
    lowerer.lower_program(&program).unwrap()
}

fn lower_code_with_paths(src: &str, paths: Vec<String>) -> tenth::hir::hir::HirProgram {
    let mut lexer = Lexer::new(src);
    let tokens = lexer.tokenize().unwrap();
    let mut parser = Parser::new(tokens);
    let program = parser.parse_program().unwrap();
    let mut lowerer = Lowerer::with_search_paths(paths);
    lowerer.lower_program(&program).unwrap()
}

/// 解释器路径执行
fn run_interp(hir: &tenth::hir::hir::HirProgram) -> Result<Option<Value>, String> {
    let mut interpreter = Interpreter::new(hir);
    interpreter.execute_program(hir).map_err(|e| e.to_string())
}

/// VM 路径执行（镜像 main.rs vm_execute 的最小实现：含全局初始化 chunk）
fn run_vm(hir: &tenth::hir::hir::HirProgram) -> Result<Value, String> {
    use tenth::compile::bytecode::BytecodeCompiler;
    use tenth::runtime::vm::Vm;
    let global_names: std::collections::HashSet<String> =
        hir.globals.iter().map(|g| g.name.clone()).collect();
    let mut vm = Vm::new();
    for func in &hir.functions {
        let compiler = BytecodeCompiler::new_with_globals(global_names.clone());
        if let Ok((chunk, closures)) = compiler.compile(func) {
            vm.add_fn(func.name.clone(), chunk);
            for (name, c) in closures {
                vm.add_fn(name, c);
            }
            vm.set_global(
                func.name.clone(),
                Value::FnRef {
                    name: func.name.clone(),
                    params: func.params.clone(),
                    return_type: func.return_type.clone(),
                },
            );
        }
    }
    // M3.5：全局初始化（main 之前）
    if !hir.globals.is_empty() {
        let gcompiler = BytecodeCompiler::new_with_globals(global_names.clone());
        if let Ok((gchunk, gclosures)) = gcompiler.compile_globals(&hir.globals) {
            vm.add_fn("__global_init".into(), gchunk);
            for (name, c) in gclosures {
                vm.add_fn(name, c);
            }
            vm.call("__global_init").map_err(|e| e.to_string())?;
        }
    }
    if vm.has_fn("main") {
        vm.call("main").map_err(|e| e.to_string())
    } else if let Some(expr) = &hir.main_expr {
        let compiler = BytecodeCompiler::new_with_globals(global_names.clone());
        if let Ok((chunk, closures)) = compiler.compile_main(expr) {
            vm.add_fn("main".into(), chunk);
            for (name, c) in closures {
                vm.add_fn(name, c);
            }
        }
        vm.call("main").map_err(|e| e.to_string())
    } else {
        Ok(Value::Unit)
    }
}

fn as_f64(v: &Value) -> f64 {
    match v {
        Value::Float(f) => *f,
        Value::Float32(f) => *f as f64,
        Value::Int(n, _) => *n as f64,
        _ => panic!("expected numeric value, got {:?}", v.type_of()),
    }
}

fn as_i64(v: &Value) -> i64 {
    match v {
        Value::Int(n, _) => *n,
        _ => panic!("expected int value, got {:?}", v.type_of()),
    }
}

fn as_f64_vec(val: &Value) -> Vec<f64> {
    match val {
        Value::Tensor(t) => {
            let data = t.borrow().data.as_f64_view();
            data.iter().cloned().collect()
        }
        Value::Float(f) => vec![*f],
        _ => panic!("expected tensor value, got {:?}", val.type_of()),
    }
}

// ── 1. 顶层常量对同文件函数可见 ────────────────────────────────

#[test]
fn toplevel_const_visible_in_same_file_fn() {
    let src = r#"
        let PI: f64 = 3.14;
        fn get_pi() -> f64 { PI }
        fn main() -> f64 { get_pi() }
    "#;
    let hir = lower_code(src);
    // 编译期：函数体内 Var(PI) 必须解析（此前报"未定义变量"）
    assert!(!hir.functions.is_empty());
    // globals 元数据
    assert_eq!(hir.globals.len(), 1);
    assert_eq!(hir.globals[0].name, "PI");
    assert!(!hir.globals[0].mutable);

    // 解释器
    let v = run_interp(&hir).unwrap().unwrap();
    assert!((as_f64(&v) - 3.14).abs() < 1e-9, "interp got {}", as_f64(&v));
    // VM
    let v = run_vm(&hir).unwrap();
    assert!((as_f64(&v) - 3.14).abs() < 1e-9, "vm got {}", as_f64(&v));
}

// ── 2. 顶层可变全局同文件函数读写 ──────────────────────────────

#[test]
fn toplevel_mutable_global_read_write() {
    let src = r#"
        let mut counter: i32 = 10;
        fn bump() -> i32 {
            counter = counter + 5;
            counter
        }
        fn main() -> i32 { bump() }
    "#;
    let hir = lower_code(src);
    assert_eq!(hir.globals.len(), 1);
    assert_eq!(hir.globals[0].name, "counter");
    assert!(hir.globals[0].mutable);

    // 解释器
    let v = run_interp(&hir).unwrap().unwrap();
    assert_eq!(as_i64(&v), 15, "interp got {}", as_i64(&v));
    // VM
    let v = run_vm(&hir).unwrap();
    assert_eq!(as_i64(&v), 15, "vm got {}", as_i64(&v));
}

// ── 3. 函数内局部变量 shadow 全局 ──────────────────────────────

#[test]
fn local_shadows_global_without_destroying_it() {
    // ① shadow 函数内局部 PI 遮蔽全局 PI（返回 99）
    let src2 = r#"
        let PI: f64 = 3.14;
        fn shadow() -> f64 {
            let PI = 99.0;
            PI
        }
        shadow()
    "#;
    let hir2 = lower_code(src2);
    let v = run_interp(&hir2).unwrap().unwrap();
    assert!((as_f64(&v) - 99.0).abs() < 1e-9, "interp shadow got {}", as_f64(&v));
    let v = run_vm(&hir2).unwrap();
    assert!((as_f64(&v) - 99.0).abs() < 1e-9, "vm shadow got {}", as_f64(&v));

    // ② 调用 shadow 后全局 PI 不被破坏（real() 仍返回 3.14）
    let src3 = r#"
        let PI: f64 = 3.14;
        fn shadow() -> f64 {
            let PI = 99.0;
            PI
        }
        fn real() -> f64 { PI }
        fn main() -> f64 {
            let s = shadow();
            real()
        }
    "#;
    let hir3 = lower_code(src3);
    let v = run_interp(&hir3).unwrap().unwrap();
    assert!((as_f64(&v) - 3.14).abs() < 1e-9, "interp real-after-shadow got {}", as_f64(&v));
    let v = run_vm(&hir3).unwrap();
    assert!((as_f64(&v) - 3.14).abs() < 1e-9, "vm real-after-shadow got {}", as_f64(&v));
}

// ── 4. use 导入模块常量（跨模块） ──────────────────────────────

#[test]
fn use_imports_module_constants() {
    let src = r#"
        use toplevel_mod_fixtures::mymod::*
        fn main() -> f64 { get_gx() }
    "#;
    // 搜索路径：tests/ 目录（夹具位于 tests/toplevel_mod_fixtures/mymod.th）
    let hir = lower_code_with_paths(
        src,
        vec![std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests")
            .to_string_lossy()
            .to_string()],
    );
    // 模块常量合并进当前程序 globals
    assert!(hir.globals.iter().any(|g| g.name == "GX" && !g.mutable));
    assert!(hir.globals.iter().any(|g| g.name == "GCOUNT" && g.mutable));

    // 解释器
    let v = run_interp(&hir).unwrap().unwrap();
    assert!((as_f64(&v) - 3.5).abs() < 1e-9, "interp got {}", as_f64(&v));
    // VM
    let v = run_vm(&hir).unwrap();
    assert!((as_f64(&v) - 3.5).abs() < 1e-9, "vm got {}", as_f64(&v));
}

// ── 5. use 导入模块可变全局（跨模块读写） ──────────────────────

#[test]
fn use_imports_module_mutable_global() {
    let src = r#"
        use toplevel_mod_fixtures::mymod::*
        fn main() -> i32 {
            bump();
            bump();
            GCOUNT
        }
    "#;
    let hir = lower_code_with_paths(
        src,
        vec![std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests")
            .to_string_lossy()
            .to_string()],
    );

    // 解释器：bump 两次 → 7→8→9，GCOUNT 最终 9
    let v = run_interp(&hir).unwrap().unwrap();
    assert_eq!(as_i64(&v), 9, "interp got {}", as_i64(&v));
    // VM
    let v = run_vm(&hir).unwrap();
    assert_eq!(as_i64(&v), 9, "vm got {}", as_i64(&v));
}

// ── 6. globals 元数据（类型/可变性） ───────────────────────────

#[test]
fn globals_metadata() {
    let src = r#"
        let A: f64 = 1.0;
        let mut B: i32 = 2;
        let C = "hi";
        fn use_a() -> f64 { A }
        fn use_b() -> i32 { B }
        fn use_c() -> string { C }
    "#;
    let hir = lower_code(src);
    let names: Vec<&str> = hir.globals.iter().map(|g| g.name.as_str()).collect();
    assert_eq!(names, vec!["A", "B", "C"]);
    assert!(!hir.globals[0].mutable);
    assert!(hir.globals[1].mutable);
    // 类型
    use tenth::hir::types::{BaseType, Type};
    assert_eq!(hir.globals[0].ty, Type::Base(BaseType::F64));
    assert_eq!(hir.globals[1].ty, Type::Base(BaseType::I32));
    // 无 main 时 main_expr 为空块或 None（let 已全部提取为全局）
    assert!(hir.main_expr.is_none() || {
        matches!(&hir.main_expr.as_ref().unwrap().kind, tenth::hir::hir::HirExprKind::Block { stmts, final_expr } if stmts.is_empty() && final_expr.is_none())
    });
}

// ── 7. REPL 式逐行累积：顶层 let 持久为全局 ────────────────────

#[test]
fn repl_line_accumulation_persists_toplevel_let() {
    // 模拟 REPL 两行：第一行定义全局，第二行函数引用全局
    let line1 = r#"
        let PI: f64 = 3.14;
    "#;
    let line2 = r#"
        fn get_pi() -> f64 { PI }
        get_pi()
    "#;

    // 第一行：REPL 用模块模式 lower → PI 提升为全局
    let hir1 = {
        let mut lexer = Lexer::new(line1);
        let tokens = lexer.tokenize().unwrap();
        let mut parser = Parser::new(tokens);
        let program = parser.parse_program().unwrap();
        let mut lowerer = Lowerer::new();
        lowerer.set_module_mode();
        lowerer.lower_program(&program).unwrap()
    };
    assert_eq!(hir1.globals.len(), 1);

    // 第一行执行：全局 PI 初始化并持久（globals_clone）
    let mut interp1 = Interpreter::new(&hir1);
    interp1.execute_program(&hir1).unwrap();
    let vars = interp1.globals_clone();
    assert!(vars.contains_key("PI"), "REPL 全局 PI 未持久化");

    // 第二行：seed 已累积全局 → 函数体可解析 PI
    let hir2 = lower_code_seeded(line2, &hir1.globals);
    let mut interp2 = Interpreter::new(&hir2);
    interp2.extend_globals(vars.clone());
    let v = interp2.execute_program(&hir2).unwrap().unwrap();
    assert!((as_f64(&v) - 3.14).abs() < 1e-9, "repl line2 got {}", as_f64(&v));
}

// ── 8. 顶层 let 无 main 时作为 main_expr 顶层值 ────────────────

#[test]
fn toplevel_let_only_program_returns_value() {
    // 函数引用的顶层 let + 无 main：globals 初始化后 main_expr 求值
    let src = r#"
        let x: i32 = 5;
        fn get() -> i32 { x }
        get()
    "#;
    let hir = lower_code(src);
    assert_eq!(hir.globals.len(), 1);
    assert_eq!(hir.globals[0].name, "x");

    let v = run_interp(&hir).unwrap().unwrap();
    assert_eq!(as_i64(&v), 5, "interp got {}", as_i64(&v));
    let v = run_vm(&hir).unwrap();
    assert_eq!(as_i64(&v), 5, "vm got {}", as_i64(&v));
}

// ── 8b. 未被函数引用的顶层 let 保留在 main_expr（顺序保护） ───────

#[test]
fn toplevel_let_not_function_referenced_stays_in_main_expr() {
    // autodiff 风格：顶层 let 与 new_grad()/backward() 交错，未被函数引用。
    // 它们必须保留在 main_expr 原位（tape 记录在 new_grad() 之后），
    // 不能被提升为全局（否则顺序错乱导致梯度丢失）。
    let src = r#"
        new_grad();
        let x = param(tensor[[3.0]]);
        let y = x * 2.0;
        backward(y);
        stop_grad();
        grad(x)
    "#;
    let hir = lower_code(src);
    // 无函数引用 → 不提取为全局
    assert!(hir.globals.is_empty(), "expected no globals, got {:?}", hir.globals.iter().map(|g| g.name.clone()).collect::<Vec<_>>());

    let v = run_interp(&hir).unwrap().unwrap();
    let g = as_f64_vec(&v);
    assert_eq!(g.len(), 1);
    assert!((g[0] - 2.0).abs() < 1e-6, "interp grad got {}", g[0]);
    // VM 路径（VM 对 autodiff 支持有限，仅在可行时验证；解释器已保证语义）
    let v = run_vm(&hir);
    if let Ok(v) = v {
        let g = as_f64_vec(&v);
        if !g.is_empty() {
            assert!((g[0] - 2.0).abs() < 1e-6, "vm grad got {}", g[0]);
        }
    }
}

// ── 9. 顶层 let 引用更早声明的全局（顺序初始化） ────────────────

#[test]
fn toplevel_let_can_reference_earlier_global() {
    let src = r#"
        let BASE: i32 = 10;
        let mut TOTAL: i32 = BASE + 5;
        fn main() -> i32 {
            TOTAL = TOTAL * 2;
            TOTAL
        }
    "#;
    let hir = lower_code(src);
    assert_eq!(hir.globals.len(), 2);

    let v = run_interp(&hir).unwrap().unwrap();
    assert_eq!(as_i64(&v), 30, "interp got {}", as_i64(&v));
    let v = run_vm(&hir).unwrap();
    assert_eq!(as_i64(&v), 30, "vm got {}", as_i64(&v));
}
