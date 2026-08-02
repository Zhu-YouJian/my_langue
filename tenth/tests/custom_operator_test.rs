//! M3.1：自定义运算符测试。
//!
//! 设计（最小可行版本）：
//! - 字符集：`@` / `$` / `~` 的连续组合（如 `@@`、`@~`、`$@$`）。
//!   这三个字符在 lexer 中原本无任何用途（会报"意外字符"），与全部内置
//!   token 零冲突，因此不会破坏既有 token 化。
//! - 声明语法：`operator <op> = fn(a: T, b: T) -> R { ... }`
//!   绑定函数以合成名 `__custom_op_<op>` 注册为普通函数。
//! - 使用：`a <op> b` 解析为 CustomBinary，lower 阶段降级为对绑定函数的
//!   普通函数调用 → VM / 解释器双路径天然支持。
//! - 优先级：统一默认优先级 4（与 `+`/`-` 同级，左结合）。
//! - 边界：未声明即使用 → 编译期错误；重复声明 → 编译期错误。
//! - 与内置运算符重载（`impl Add for T`）可共存。

use tenth::lexer::lexer::Lexer;
use tenth::parser::parser::Parser;
use tenth::hir::lower::Lowerer;
use tenth::runtime::interpreter::Interpreter;
use tenth::runtime::vm::Vm;
use tenth::runtime::value::Value;
use tenth::compile::bytecode::BytecodeCompiler;
use tenth::compile::jit;
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
    vm.add_native("assert".into(), |_vm, args| {
        if let Some(Value::Bool(b)) = args.first() {
            assert!(*b, "assertion failed");
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
    } else {
        Ok(Value::Unit)
    }
}

/// JIT 路径：与 run_vm 相同，但通过 jit::run_jit 执行（fallback 到 VM）。
fn run_jit(src: &str) -> Result<Value, String> {
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
    vm.add_native("assert".into(), |_vm, args| {
        if let Some(Value::Bool(b)) = args.first() {
            assert!(*b, "assertion failed");
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
        jit::run_jit(&mut vm, "main").map_err(|e| e.to_string())
    } else {
        Ok(Value::Unit)
    }
}

fn expect_int(v: &Value, expected: i64, tag: &str) {
    match v {
        Value::Int(n, _) => assert_eq!(*n, expected, "{}: 期望 {}，实际 {}", tag, expected, n),
        other => panic!("{}: 期望 Int({})，实际 {:?}", tag, expected, other),
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
        Value::String(s) => {
            assert_eq!(s.as_str(), expected, "{}: 期望 {:?}，实际 {:?}", tag, expected, s.as_str());
        }
        other => panic!("{}: 期望 Str，实际 {:?}", tag, other),
    }
}

// ── 声明 + 使用 ────────────────────────────────────────────────────────────

#[test]
fn test_declare_and_use_int() {
    // 任务示例：`operator <|> = fn(a: i64, b: i64) -> i64 { a + b }`
    // 字符集限制下用 `@@` 等价表达。
    let src = r#"
    operator @@ = fn(a: i64, b: i64) -> i64 { a + b }
    { 1 @@ 2 }
    "#;
    let r = run(src).unwrap();
    expect_int(&r.unwrap(), 3, "解释器");

    let r = run_vm(src).unwrap();
    expect_int(&r, 3, "VM");
}

#[test]
fn test_declare_and_use_float() {
    let src = r#"
    operator @~ = fn(a: f64, b: f64) -> f64 { a * b + 1.0 }
    { 2.0 @~ 3.0 }
    "#;
    let r = run(src).unwrap();
    expect_float(&r.unwrap(), 7.0, "解释器");

    let r = run_vm(src).unwrap();
    expect_float(&r, 7.0, "VM");
}

#[test]
fn test_declare_and_use_string() {
    let src = r#"
    operator $@ = fn(a: str, b: str) -> str { a + "-" + b }
    "foo" $@ "bar"
    "#;
    let r = run(src).unwrap();
    expect_str(&r.unwrap(), "foo-bar", "解释器");

    let r = run_vm(src).unwrap();
    expect_str(&r, "foo-bar", "VM");
}

// ── 多个自定义运算符 ───────────────────────────────────────────────────────

#[test]
fn test_multiple_custom_operators() {
    let src = r#"
    operator @@ = fn(a: i64, b: i64) -> i64 { a + b }
    operator @~ = fn(a: i64, b: i64) -> i64 { a * b }
    operator $$ = fn(a: i64, b: i64) -> i64 { a - b }
    let x = 10 @@ 4;
    let y = x @~ 3;
    y $$ 5
    "#;
    // (10+4)*3 - 5 = 37
    let r = run(src).unwrap();
    expect_int(&r.unwrap(), 37, "解释器");

    let r = run_vm(src).unwrap();
    expect_int(&r, 37, "VM");
}

#[test]
fn test_custom_op_in_function_and_call() {
    let src = r#"
    operator @@ = fn(a: i64, b: i64) -> i64 { a + b }
    fn apply(x: i64, y: i64) -> i64 { x @@ y }
    apply(20, 22)
    "#;
    let r = run(src).unwrap();
    expect_int(&r.unwrap(), 42, "解释器");

    let r = run_vm(src).unwrap();
    expect_int(&r, 42, "VM");
}

// ── 优先级与结合性 ─────────────────────────────────────────────────────────

#[test]
fn test_precedence_default_like_plus() {
    // 自定义运算符默认优先级 4（与 + 同级，左结合）：
    // 1 + 2 @@ 3  →  (1 + 2) @@ 3
    let src = r#"
    operator @@ = fn(a: i64, b: i64) -> i64 { a * 10 + b }
    1 + 2 @@ 3
    "#;
    // (1+2) @@ 3 = 3*10+3 = 33
    let r = run(src).unwrap();
    expect_int(&r.unwrap(), 33, "解释器");
}

#[test]
fn test_mul_binds_tighter_than_custom_op() {
    // 乘法优先级 5 > 自定义运算符 4：1 @@ 2 * 3 → 1 @@ (2*3)
    let src = r#"
    operator @@ = fn(a: i64, b: i64) -> i64 { a + b }
    1 @@ 2 * 3
    "#;
    // 1 @@ 6 = 7
    let r = run(src).unwrap();
    expect_int(&r.unwrap(), 7, "解释器");

    let r = run_vm(src).unwrap();
    expect_int(&r, 7, "VM");
}

// ── 与内置运算符重载共存 ───────────────────────────────────────────────────

#[test]
fn test_coexist_with_builtin_overload() {
    // `a + b` 降级为 `a.add(b)` 方法调用。批次2 C 前，VM 路径的 Value::Struct
    // 方法分派只做字段访问（既有局限），本测试仅验证解释器路径；批次2 C
    // （编译期具体值 trait 方法改写 → `__dyn_Add_V_add`）后，VM/JIT 路径与
    // 解释器一致，此处补 VM/JIT 断言（与自定义运算符降级为顶层函数调用共存）。
    let src = r#"
    struct V { x: i64 }
    trait Add { fn add(self, other: V) -> V; }
    impl Add for V {
        fn add(self, other: V) -> V { V { x: self.x + other.x } }
    }
    operator @@ = fn(a: V, b: V) -> V { V { x: a.x * b.x } }
    let a = V { x: 2 };
    let b = V { x: 3 };
    let c = V { x: 4 };
    let s = a + b;      // 内置重载 Add::add → 2+3=5
    let m = s @@ c;     // 自定义运算符 → 5*4=20
    m.x
    "#;
    let r = run(src).unwrap();
    expect_int(&r.unwrap(), 20, "解释器");

    // 批次2 C：VM/JIT 路径与解释器一致（具体值 trait 方法编译期改写已打通）
    let r = run_vm(src).unwrap();
    expect_int(&r, 20, "VM");

    let r = run_jit(src).unwrap();
    expect_int(&r, 20, "JIT");
}

// ── 边界：未声明 / 重复声明 / 运算符名 ─────────────────────────────────────

#[test]
fn test_undeclared_operator_errors() {
    // 未声明即使用 → 编译期 TypeError
    let src = r#"
    { 1 @@ 2 }
    "#;
    let err = run(src).unwrap_err();
    assert!(err.contains("未声明的运算符") || err.contains("@@"),
        "期望未声明运算符错误，实际: {}", err);
}

#[test]
fn test_duplicate_declaration_errors() {
    let src = r#"
    operator @@ = fn(a: i64, b: i64) -> i64 { a + b }
    operator @@ = fn(a: i64, b: i64) -> i64 { a - b }
    { 1 @@ 2 }
    "#;
    let err = run(src).unwrap_err();
    assert!(err.contains("重复声明"), "期望重复声明错误，实际: {}", err);
}

#[test]
fn test_operator_must_be_custom_token() {
    // operator 后必须跟 @/$/~ 组合，不能跟标识符
    let src = r#"
    operator add = fn(a: i64, b: i64) -> i64 { a + b }
    "#;
    let err = run(src).unwrap_err();
    assert!(!err.is_empty(), "期望解析错误");
}

#[test]
fn test_builtin_token_names_unreachable_as_custom() {
    // `+` 不是 CustomOperator token（lexer 固定识别为 Plus），
    // 因此无法被 operator 声明捕获——这是设计边界（@$~ 集合外不开放）。
    // 验证 `operator +` 会得到解析错误而非静默成功。
    let src = r#"
    operator + = fn(a: i64, b: i64) -> i64 { a + b }
    "#;
    let err = run(src).unwrap_err();
    assert!(!err.is_empty(), "期望解析错误（+ 不是自定义运算符 token）");
}

// ── lexer 层面：@$~ 组合的 token 化 ───────────────────────────────────────

#[test]
fn test_lexer_custom_operator_tokens() {
    use tenth::lexer::token::TokenKind;
    let mut lexer = Lexer::new("a @@ b @~ c $$ d $@$ e");
    let tokens = lexer.tokenize().unwrap();
    let ops: Vec<String> = tokens.iter()
        .filter_map(|t| match &t.kind {
            TokenKind::CustomOperator(s) => Some(s.clone()),
            _ => None,
        })
        .collect();
    assert_eq!(ops, vec!["@@".to_string(), "@~".to_string(), "$$".to_string(), "$@$".to_string()]);
}

#[test]
fn test_builtin_tokens_unaffected_by_custom_op_lexing() {
    // 既有运算符 token 化不被破坏（@$~ 外字符仍按原规则）
    use tenth::lexer::token::TokenKind;
    let mut lexer = Lexer::new("a <= b && c || d == e << f");
    let tokens = lexer.tokenize().unwrap();
    let kinds: Vec<TokenKind> = tokens.iter().map(|t| t.kind.clone()).collect();
    assert!(kinds.contains(&TokenKind::LtEq));
    assert!(kinds.contains(&TokenKind::AndAnd));
    assert!(kinds.contains(&TokenKind::OrOr));
    assert!(kinds.contains(&TokenKind::EqEq));
    assert!(kinds.contains(&TokenKind::Shl));
    // 不应出现 CustomOperator
    assert!(!kinds.iter().any(|k| matches!(k, TokenKind::CustomOperator(_))));
}

// ── 与泛型函数 / 其他语言特性交互 ─────────────────────────────────────────

#[test]
fn test_custom_op_in_closure() {
    let src = r#"
    operator @@ = fn(a: i64, b: i64) -> i64 { a + b }
    let f = |x: i64| x @@ 1;
    f(41)
    "#;
    let r = run(src).unwrap();
    expect_int(&r.unwrap(), 42, "解释器");

    let r = run_vm(src).unwrap();
    expect_int(&r, 42, "VM");
}
