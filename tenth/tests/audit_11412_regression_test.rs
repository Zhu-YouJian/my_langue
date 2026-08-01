//! 回归测试：AUDIT-11.4.12 —— 删除 `.shape()` 类型系统误标分支。
//!
//! 现象：`hir/types.rs`（现 `hir/lower/types.rs`）曾把张量 `.shape()` 误标为
//! 返回 `Array<i64>`，但运行时无对应 native——`x.shape()` 类型检查能通过、
//! 运行时崩溃。正确路径是 `.shape_tensor()`（返回 `Tensor[f64, ndim]`）。
//!
//! 修复：删除 `"shape"` 分支 + MethodCall 降级处对 Tensor receiver 的
//! `shape` 方法直接报编译期 TypeError，引导用户改用 `.shape_tensor()`。

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

/// `x.shape()` 必须在编译期报错，且提示使用 shape_tensor。
#[test]
fn shape_method_compile_error() {
    let src = r#"
fn main() {
    let x = [[1.0, 2.0], [3.0, 4.0]];
    let s = x.shape();
    s
}
"#;
    let err = lower_error(src).expect_err("x.shape() 应编译期报错");
    assert!(
        err.contains("shape_tensor"),
        "错误信息应提示使用 shape_tensor，实际: {}",
        err
    );
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
