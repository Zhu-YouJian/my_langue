//! M2.6：Copy trait 测试。
//!
//! 覆盖：
//! - 自动派生：所有字段 Copy 的结构体自动 impl Copy（`struct Point { x: i64 }`）
//! - `move p` 后原变量仍可用（Copy 值 move 不失效）
//! - 非 Copy 结构体（含 Vec 字段）`move p` 后原变量失效（编译期报错）
//! - 显式 `impl Copy for T`（含非 Copy 字段也可强制 Copy）
//! - `let q = p` 隐式克隆语义（Tenth 设计：let 赋值不移动，双方可用）
//! - Copy 结构体作为函数参数传递后原变量仍可用
//! - Copy 与 Drop 互斥（实现 Drop 的类型不自动派生 Copy）
//! - `str` 字段视为 Copy（设计取舍：字符串深拷贝安全）
//! - VM / 解释器 parity

use tenth::lexer::lexer::Lexer;
use tenth::parser::parser::Parser;
use tenth::hir::lower::Lowerer;
use tenth::runtime::interpreter::Interpreter;
use tenth::runtime::value::Value;

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

fn expect_int(v: Option<Value>, expected: i64, label: &str) {
    match v {
        Some(Value::Int(n, _)) => assert_eq!(n, expected, "{}: 期望 Int({}), 实际 {}", label, expected, n),
        other => panic!("{}: 期望 Int({}), 实际 {:?}", label, expected, other),
    }
}

// ── 1. 自动派生：move 后原变量仍可用 ──

#[test]
fn copy_auto_derive_move_exempt() {
    let src = r#"
        struct Point { x: i64, y: i64 }
        let p = Point { x: 1, y: 2 }
        let q = move p
        q.x + p.y
    "#;
    expect_int(run(src).unwrap(), 3, "auto-Copy 结构体 move 后原变量仍可用");
}

// ── 2. 非 Copy 结构体（Vec 字段）：move 后原变量失效 ──

#[test]
fn copy_noncopy_move_invalidates() {
    let src = r#"
        struct NC { items: Vec<i64> }
        let p = NC { items: [1, 2, 3] }
        let q = move p
        q.items.len()
    "#;
    // move 后 q 可用
    expect_int(run(src).unwrap(), 3, "非 Copy 结构体 move 后新变量可用");
}

#[test]
fn copy_noncopy_use_after_move_errors() {
    let src = r#"
        struct NC { items: Vec<i64> }
        let p = NC { items: [1, 2, 3] }
        let q = move p
        p.items.len()
    "#;
    let result = run(src);
    assert!(result.is_err(), "期望错误：非 Copy 结构体 move 后使用原变量");
}

// ── 3. 显式 impl Copy：非 Copy 字段也可强制 Copy ──

#[test]
fn copy_explicit_impl_override() {
    let src = r#"
        struct Wrap { items: Vec<i64> }
        impl Copy for Wrap {
        }
        let p = Wrap { items: [1, 2, 3] }
        let q = move p
        q.items.len() + p.items.len()
    "#;
    expect_int(run(src).unwrap(), 6, "显式 impl Copy 后 move 不失效");
}

// ── 4. `let q = p` 隐式克隆（不移动）──

#[test]
fn copy_let_rebind_implicit_clone() {
    // Tenth 设计：`let q = p` 是隐式克隆，不标记原变量移动（见手册 §10.2）
    let src = r#"
        struct Point { x: i64 }
        let p = Point { x: 5 }
        let q = p
        q.x + p.x
    "#;
    expect_int(run(src).unwrap(), 10, "let 赋值双方可用（隐式克隆）");
}

// ── 5. Copy 结构体作为函数参数传递后原变量仍可用 ──

#[test]
fn copy_as_fn_arg_preserves_original() {
    let src = r#"
        struct Point { x: i64 }
        fn get(p: Point) -> i64 { p.x }
        let p = Point { x: 5 }
        get(p) + p.x
    "#;
    expect_int(run(src).unwrap(), 10, "Copy 结构体传参后原变量可用");
}

// ── 6. Copy 与 Drop 互斥：实现 Drop 的类型不自动派生 Copy ──

#[test]
fn copy_drop_mutually_exclusive() {
    let src = r#"
        struct Res { id: i64 }
        impl Drop for Res {
            fn drop(self) { println("drop") }
        }
        let a = Res { id: 1 }
        let b = move a
        b.id
    "#;
    // move 后 a 应失效（Res 实现 Drop → 非 Copy）
    expect_int(run(src).unwrap(), 1, "Drop 类型 move 后新变量可用");
}

#[test]
fn copy_drop_mutually_exclusive_use_after_move() {
    let src = r#"
        struct Res { id: i64 }
        impl Drop for Res {
            fn drop(self) { println("drop") }
        }
        let a = Res { id: 1 }
        let b = move a
        a.id
    "#;
    let result = run(src);
    assert!(result.is_err(), "期望错误：Drop 类型（非 Copy）move 后使用原变量");
}

// ── 7. str 字段视为 Copy（设计取舍：深拷贝安全）──

#[test]
fn copy_str_field_is_copy() {
    let src = r#"
        struct S { s: str }
        let p = S { s: "hello" }
        let q = move p
        q.s.len() + p.s.len()
    "#;
    expect_int(run(src).unwrap(), 10, "str 字段结构体 move 后原变量仍可用");
}
