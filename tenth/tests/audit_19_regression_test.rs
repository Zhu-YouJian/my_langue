//! 回归测试：AUDIT #19 —— 运算符重载降级支持链式（`(a+b)+c` / `a+b+c` / `a*b+c`）。
//!
//! 根因：trait 方法（`impl Add for Point` 的 `add`）不在 inherent 方法表中，
//! `resolve_method_type` 回退 `Unknown` → 单层 `a + b` 可用，但链式
//! `(a + b) + c` 的复合 receiver 类型丢失，外层降级检查断链 → 运行时
//! "加法类型不匹配"。修复：从 `trait_impls[trait][type][method].return_type`
//! 取真实返回类型，链式 receiver 保持为具体类型。
//!
//! 注：仅验证解释器路径——VM 对「具体值 trait 方法分派」有既有缺口
//! （Value::Struct 方法分派只做字段访问），属已知限制（见
//! custom_operator_test::test_coexist_with_builtin_overload 注释），
//! 链式降级本身（lower 层类型传播）已修复，VM 单层也不支持。

use tenth::lexer::lexer::Lexer;
use tenth::parser::parser::Parser;
use tenth::hir::lower::Lowerer;
use tenth::runtime::interpreter::Interpreter;
use tenth::runtime::value::Value;

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

fn expect_int(v: Option<Value>, expected: i64, label: &str) {
    match v {
        Some(Value::Int(n, _)) => assert_eq!(n, expected, "{}: 期望 Int({}), 实际 {}", label, expected, n),
        other => panic!("{}: 期望 Int({}), 实际 {:?}", label, expected, other),
    }
}

/// `(a + b) + c`：括号链式（原规避场景）。
#[test]
fn chained_add_parenthesized() {
    let src = r#"
struct V { x: i64 }
trait Add { fn add(self, other: V) -> V; }
impl Add for V {
    fn add(self, other: V) -> V { V { x: self.x + other.x } }
}
let a = V { x: 1 };
let b = V { x: 2 };
let c = V { x: 3 };
let r = (a + b) + c;
r.x
"#;
    let r = run(src).unwrap_or_else(|e| panic!("(a+b)+c 执行失败: {}", e));
    expect_int(r, 6, "(a+b)+c");
}

/// `a + b + c`：左结合链式（无括号）。
#[test]
fn chained_add_flat() {
    let src = r#"
struct V { x: i64 }
trait Add { fn add(self, other: V) -> V; }
impl Add for V {
    fn add(self, other: V) -> V { V { x: self.x + other.x } }
}
let a = V { x: 10 };
let b = V { x: 20 };
let c = V { x: 30 };
let d = V { x: 40 };
let r = a + b + c + d;
r.x
"#;
    let r = run(src).unwrap_or_else(|e| panic!("a+b+c+d 执行失败: {}", e));
    expect_int(r, 100, "a+b+c+d");
}

/// `a * b + c`：混合运算符链式（Mul 与 Add 都重载）。
#[test]
fn chained_mixed_mul_add() {
    let src = r#"
struct V { x: i64 }
trait Add { fn add(self, other: V) -> V; }
impl Add for V {
    fn add(self, other: V) -> V { V { x: self.x + other.x } }
}
trait Mul { fn mul(self, other: V) -> V; }
impl Mul for V {
    fn mul(self, other: V) -> V { V { x: self.x * other.x } }
}
let a = V { x: 2 };
let b = V { x: 3 };
let c = V { x: 4 };
let r = a * b + c;   // (2*3)+4 = 10
r.x
"#;
    let r = run(src).unwrap_or_else(|e| panic!("a*b+c 执行失败: {}", e));
    expect_int(r, 10, "a*b+c");
}

/// 链式结果可继续参与比较/嵌套（`(a+b) == (c+d)`，Eq 重载）。
#[test]
fn chained_result_reusable() {
    let src = r#"
struct V { x: i64 }
trait Add { fn add(self, other: V) -> V; }
impl Add for V {
    fn add(self, other: V) -> V { V { x: self.x + other.x } }
}
trait Eq { fn eq(self, other: V) -> bool; }
impl Eq for V {
    fn eq(self, other: V) -> bool { self.x == other.x }
}
let a = V { x: 1 };
let b = V { x: 2 };
let c = V { x: 3 };
let d = V { x: 0 };
let r = (a + b) == (c + d);
if r { 1 } else { 0 }
"#;
    let r = run(src).unwrap_or_else(|e| panic!("(a+b)==(c+d) 执行失败: {}", e));
    expect_int(r, 1, "(a+b)==(c+d)");
}
