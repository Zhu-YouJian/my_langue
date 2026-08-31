//! 回归测试：AUDIT-11.4.12 —— `.shape()` 方法的历史演进。
//!
//! 历史：`hir/types.rs`（现 `hir/lower/types.rs`）曾把张量 `.shape()` 误标为
//! 返回 `Array<i64>`，但当时运行时无对应 native——`x.shape()` 类型检查能
//! 通过、运行时崩溃。修复（AUDIT-11.4.12）：删除 `"shape"` 分支 + 编译期
//! TypeError 拦截，引导用户改用 `.shape_tensor()`。
//!
//! 演进（2026-08-31，MINOR 兼容新增）：`.shape()` 已在 VM（vm/natives.rs）
//! 与解释器（interpreter/methods.rs）双侧注册，返回 `Vec<i64>`；类型推断与
//! 编译期拦截同步撤销。`.shape_tensor()`（f64 张量）保留不变。
//! 本文件同步更新：不再断言 `.shape()` 报错，改为断言其正常工作。

use tenth::lexer::lexer::Lexer;
use tenth::parser::parser::Parser;
use tenth::hir::lower::Lowerer;
use tenth::runtime::interpreter::Interpreter;
use tenth::runtime::value::Value;

/// 只做 lower（用于断言编译期错误）。
fn lower_error(src: &str) -> Result<(), String> {
    let mut lexer = Lexer::new(src);
    let tokens = lexer.tokenize().map_err(|e| e.to_string())?;
    let mut parser = Parser::new(tokens);
    let program = parser.parse_program().map_err(|e| e.to_string())?;
    let mut lowerer = Lowerer::new();
    lowerer.lower_program(&program).map(|_| ()).map_err(|e| e.to_string())
}

/// 解释器路径执行。
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

/// `x.shape()` 现已正常工作（双侧注册后）：返回 Vec<i64>，元素 = 各维大小。
/// （AUDIT-11.4.12 时期曾断言其编译期报错——运行时 native 补齐后语义演进。）
#[test]
fn shape_method_returns_vec() {
    let src = r#"
fn main() {
    let x = [[1.0, 2.0], [3.0, 4.0]];
    let s = x.shape();
    s
}
"#;
    // 先确认不再编译期报错
    assert!(lower_error(src).is_ok(), "x.shape() 应可编译（不再拦截）");
    let r = run(src).unwrap_or_else(|e| panic!("x.shape() 执行失败: {}", e));
    match r {
        Some(Value::Vec(items)) => {
            let items = items.borrow();
            assert_eq!(items.len(), 2, "2x2 张量的 shape() 应返回 2 个元素");
            assert!(matches!(&items[0], Value::Int(2, _)), "shape()[0] 应为 2");
            assert!(matches!(&items[1], Value::Int(2, _)), "shape()[1] 应为 2");
        }
        other => panic!("期望 Vec([2, 2]), 实际 {:?}", other),
    }
}

/// `x.shape_tensor()` 正常工作：返回 1D 维度张量（元素 = 各维大小）。
#[test]
fn shape_tensor_works() {
    let src = r#"
fn main() {
    let x = [[1.0, 2.0], [3.0, 4.0]];
    let s = x.shape_tensor();
    assert_eq(s.ndim(), 1);
    assert_eq(s[0], 2.0);
    assert_eq(s[1], 2.0);
    s.ndim()
}
"#;
    let r = run(src).unwrap_or_else(|e| panic!("shape_tensor 执行失败: {}", e));
    match r {
        Some(Value::Int(n, _)) => assert_eq!(n, 1, "shape_tensor() 应为 1D 张量"),
        other => panic!("期望 Int(1), 实际 {:?}", other),
    }
}

/// 用户自定义 struct 的 `shape` 方法不受影响（Tensor 专属检查不误伤）。
#[test]
fn user_defined_shape_method_ok() {
    let src = r#"
struct Box2 { w: i64, h: i64 }
impl Box2 {
    fn shape(self) -> i64 { self.w * self.h }
}
fn main() {
    let b = Box2 { w: 3, h: 4 };
    let s = b.shape();
    assert_eq(s, 12);
    s
}
"#;
    let r = run(src).unwrap_or_else(|e| panic!("用户自定义 shape 方法执行失败: {}", e));
    match r {
        Some(Value::Int(n, _)) => assert_eq!(n, 12, "用户自定义 shape 应返回 12"),
        other => panic!("期望 Int(12), 实际 {:?}", other),
    }
}
