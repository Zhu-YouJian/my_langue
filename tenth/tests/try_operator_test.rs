//! `?` 操作符集成测试。
//!
//! 覆盖 `?` 操作符的四种场景：Ok 解包、Err 提前返回、链式 `?`、模拟 I/O 错误传播，
//! 以及 try 块捕获、多层 `?`。同时验证解释器（Interpreter）与字节码 VM（路径 A 默认后端）两条路径。
//!
//! `?` 语义：从 `Result<T>` 中提取 T；如果是 `Result::Err(e)` 则提前返回 `Result::Err(e)`。
//! - 解释器：通过 `TenthError::TryPropagate` 信号传递，函数边界 `unwrap_return` 与
//!   try 块捕获处均保持单层 `Result::Err(e)`（与 VM 一致，AUDIT-11.4.33 修复）。
//! - VM：`Op::Try` 通过 frame 恢复实现 early return（最外层函数直接返回 `Result::Err`）。
//!
//! 注意：`?` 只能在函数体内使用（依赖 frame 恢复机制），不能在顶层表达式使用。
//! `Result` 枚举为预定义内置：`Result::Ok(value)` / `Result::Err(error: str)`。

use tenth::compile::bytecode::BytecodeCompiler;
use tenth::hir::lower::Lowerer;
use tenth::lexer::lexer::Lexer;
use tenth::parser::parser::Parser;
use tenth::runtime::interpreter::Interpreter;
use tenth::runtime::value::Value;
use tenth::runtime::vm::Vm;

/// Run source through lexer → parser → HIR → interpreter.
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

/// Run source through the bytecode VM (path A default backend).
fn run_vm(src: &str) -> Result<Value, String> {
    let mut lexer = Lexer::new(src);
    let tokens = lexer.tokenize().map_err(|e| e.to_string())?;
    let mut parser = Parser::new(tokens);
    let program = parser.parse_program().map_err(|e| e.to_string())?;
    let mut lowerer = Lowerer::new();
    let hir = lowerer.lower_program(&program).map_err(|e| e.to_string())?;

    let mut vm = Vm::new();
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

/// 从 `Result::Err(...)` 中提取错误消息字符串。
/// AUDIT-11.4.33 修复后解释器与 VM 一致返回**单层** `Result::Err(String)`：
/// 内层必须是 `String`，若是 `Result::Err` 则说明 double-wrap 回归（测试失败）。
fn extract_err_msg(v: &Value) -> String {
    match v {
        Value::Enum { enum_name, variant, fields } if enum_name == "Result" && variant == "Err" => {
            let borrowed = fields.borrow();
            match borrowed.first() {
                Some((_, Value::String(s))) => s.clone(),
                Some((_, Value::Enum { enum_name, variant, .. }))
                    if enum_name == "Result" && variant == "Err" =>
                {
                    panic!("double-wrap 回归！解释器/VM 应返回单层 Result::Err(String)，got {:?}", v);
                }
                Some((_, inner)) => panic!("期望 Err(String), got {:?}", inner),
                None => panic!("Err 无字段, got {:?}", v),
            }
        }
        _ => panic!("期望 Result::Err, got {:?}", v),
    }
}

// ─── 1. `?` 解包 Ok：解释器路径 ─────────────────────────────────────────────

#[test]
fn test_try_ok() {
    let src = r#"
        fn main() -> i64 {
            let x = Result::Ok(42)?;
            x
        }
    "#;
    let result = run(src).unwrap();
    match result {
        Some(Value::Int(42, _)) => {}
        v => panic!("期望 Int(42), got {:?}", v),
    }
}

// ─── 2. `?` 传播 Err：解释器路径 ────────────────────────────────────────────
// 函数声明返回 i64，但 `?` 遇到 Err 会提前返回 Result::Err（通过 TryPropagate 信号）。
// AUDIT-11.4.33 后解释器返回**单层** `Result::Err(String)`（与 VM 一致）。

#[test]
fn test_try_err_propagation() {
    let src = r#"
        fn main() -> i64 {
            let x = Result::Err("error")?;
            x
        }
    "#;
    let result = run(src).unwrap();
    match result {
        Some(v) => {
            let msg = extract_err_msg(&v);
            assert_eq!(msg, "error", "期望单层 Err(\"error\")");
        }
        None => panic!("期望 Some(Result::Err), got None"),
    }
}

// ─── 2b. 多层 `?` 传播：解释器路径（AUDIT-11.4.33 守护）────────────────────
// inner 内 `?` 传播一次，main 内 `inner()?` 再传播一次——两层传播后仍是单层。

#[test]
fn test_try_multilayer_err_propagation() {
    let src = r#"
        fn inner() -> Result<i64, str> {
            Result::Err("deep")?
        }
        fn main() -> i64 {
            let x = inner()?;
            x
        }
    "#;
    let result = run(src).unwrap();
    match result {
        Some(v) => {
            let msg = extract_err_msg(&v);
            assert_eq!(msg, "deep", "多层 ? 期望单层 Err(\"deep\")");
        }
        None => panic!("期望 Some(Result::Err), got None"),
    }
}

// ─── 3. 链式 `?`：解释器路径 ────────────────────────────────────────────────
// 自定义 parse 函数返回 Result，连续 `?` 解包；第二个 parse 失败会传播 Err。

#[test]
fn test_try_chain_success() {
    let src = r#"
        fn parse(s: str) -> Result<i64, str> {
            if s == "42" { Result::Ok(42) } else { Result::Err("not 42") }
        }
        fn main() -> i64 {
            let a = parse("42")?;
            let b = parse("42")?;
            a + b
        }
    "#;
    let result = run(src).unwrap();
    match result {
        Some(Value::Int(84, _)) => {}
        v => panic!("期望 Int(84), got {:?}", v),
    }
}

#[test]
fn test_try_chain_err_propagation() {
    let src = r#"
        fn parse(s: str) -> Result<i64, str> {
            if s == "42" { Result::Ok(42) } else { Result::Err("not 42") }
        }
        fn main() -> i64 {
            let a = parse("42")?;
            let b = parse("10")?;
            a + b
        }
    "#;
    let result = run(src).unwrap();
    match result {
        Some(v) => {
            let msg = extract_err_msg(&v);
            assert_eq!(msg, "not 42", "期望 Err(\"not 42\")");
        }
        None => panic!("期望 Some(Result::Err), got None"),
    }
}

// ─── 4. 模拟 I/O 场景：解释器路径 ───────────────────────────────────────────
// 用自定义 simulate_read 函数模拟 I/O 错误传播（避免与 native read_file 同名冲突，
// read_file native 实际返回 str 而非 Result，遇到文件不存在会 RuntimeError panic）。

#[test]
fn test_try_with_io_error() {
    let src = r#"
        fn simulate_read(path: str) -> Result<str, str> {
            if path == "nonexistent" {
                Result::Err("file not found")
            } else {
                Result::Ok("hello")
            }
        }
        fn main() -> str {
            let content = simulate_read("nonexistent")?;
            content
        }
    "#;
    let result = run(src).unwrap();
    match result {
        Some(v) => {
            let msg = extract_err_msg(&v);
            assert_eq!(msg, "file not found", "期望 Err(\"file not found\")");
        }
        None => panic!("期望 Some(Result::Err), got None"),
    }
}

#[test]
fn test_try_with_io_ok() {
    let src = r#"
        fn simulate_read(path: str) -> Result<str, str> {
            if path == "nonexistent" {
                Result::Err("file not found")
            } else {
                Result::Ok("hello")
            }
        }
        fn main() -> str {
            let content = simulate_read("exists")?;
            content
        }
    "#;
    let result = run(src).unwrap();
    match result {
        Some(Value::String(s)) if s == "hello" => {}
        v => panic!("期望 String(\"hello\"), got {:?}", v),
    }
}

// ─── 5. try 块捕获 `?` 传播（AUDIT-11.4.33 守护：单层 Result::Err）──────────

#[test]
fn test_try_block_success() {
    let src = r#"
        fn parse(s: str) -> Result<i64, str> {
            if s == "42" { Result::Ok(42) } else { Result::Err("not 42") }
        }
        fn main() -> i64 {
            let r = try { parse("42")? };
            let ok = match r { Result::Ok(v) => v, _ => 0 };
            ok
        }
    "#;
    let result = run(src).unwrap();
    match result {
        Some(Value::Int(42, _)) => {}
        v => panic!("try 成功应解出 42, got {:?}", v),
    }
}

#[test]
fn test_try_block_catches_err() {
    let src = r#"
        fn parse(s: str) -> Result<i64, str> {
            if s == "42" { Result::Ok(42) } else { Result::Err("not 42") }
        }
        fn main() -> i64 {
            let r = try { parse("10")? };
            let msg = match r { Result::Err(m) => m, _ => "?" };
            if msg == "not 42" { 1 } else { 0 }
        }
    "#;
    let result = run(src).unwrap();
    // try 捕获 ? 传播 → 返回单层 Result::Err("not 42")；match 解出 msg 为 "not 42" → 1
    match result {
        Some(Value::Int(1, _)) => {}
        v => panic!("try 捕获后应得到单层 Err 且 match 解出消息, got {:?}", v),
    }
}

// ─── VM 路径（路径 A 默认后端）──────────────────────────────────────────────

#[test]
fn test_vm_try_ok() {
    let src = r#"
        fn main() -> i64 {
            let x = Result::Ok(42)?;
            x
        }
    "#;
    let result = run_vm(src).unwrap();
    match result {
        Value::Int(42, _) => {}
        v => panic!("VM: 期望 Int(42), got {:?}", v),
    }
}

#[test]
fn test_vm_try_err_propagation() {
    let src = r#"
        fn main() -> i64 {
            let x = Result::Err("error")?;
            x
        }
    "#;
    let result = run_vm(src).unwrap();
    let msg = extract_err_msg(&result);
    assert_eq!(msg, "error", "VM: 期望 Err(\"error\")");
}

#[test]
fn test_vm_try_chain_success() {
    let src = r#"
        fn parse(s: str) -> Result<i64, str> {
            if s == "42" { Result::Ok(42) } else { Result::Err("not 42") }
        }
        fn main() -> i64 {
            let a = parse("42")?;
            let b = parse("42")?;
            a + b
        }
    "#;
    let result = run_vm(src).unwrap();
    match result {
        Value::Int(84, _) => {}
        v => panic!("VM: 期望 Int(84), got {:?}", v),
    }
}

#[test]
fn test_vm_try_chain_err_propagation() {
    let src = r#"
        fn parse(s: str) -> Result<i64, str> {
            if s == "42" { Result::Ok(42) } else { Result::Err("not 42") }
        }
        fn main() -> i64 {
            let a = parse("42")?;
            let b = parse("10")?;
            a + b
        }
    "#;
    let result = run_vm(src).unwrap();
    let msg = extract_err_msg(&result);
    assert_eq!(msg, "not 42", "VM: 期望 Err(\"not 42\")");
}

#[test]
fn test_vm_try_with_io_error() {
    let src = r#"
        fn simulate_read(path: str) -> Result<str, str> {
            if path == "nonexistent" {
                Result::Err("file not found")
            } else {
                Result::Ok("hello")
            }
        }
        fn main() -> str {
            let content = simulate_read("nonexistent")?;
            content
        }
    "#;
    let result = run_vm(src).unwrap();
    let msg = extract_err_msg(&result);
    assert_eq!(msg, "file not found", "VM: 期望 Err(\"file not found\")");
}

#[test]
fn test_vm_try_with_io_ok() {
    let src = r#"
        fn simulate_read(path: str) -> Result<str, str> {
            if path == "nonexistent" {
                Result::Err("file not found")
            } else {
                Result::Ok("hello")
            }
        }
        fn main() -> str {
            let content = simulate_read("exists")?;
            content
        }
    "#;
    let result = run_vm(src).unwrap();
    match result {
        Value::String(s) if s == "hello" => {}
        v => panic!("VM: 期望 String(\"hello\"), got {:?}", v),
    }
}

// ─── VM 多层 `?` + try 块（AUDIT-11.4.33 守护，与解释器一致）────────────────

#[test]
fn test_vm_try_multilayer_err_propagation() {
    let src = r#"
        fn inner() -> Result<i64, str> {
            Result::Err("deep")?
        }
        fn main() -> i64 {
            let x = inner()?;
            x
        }
    "#;
    let result = run_vm(src).unwrap();
    let msg = extract_err_msg(&result);
    assert_eq!(msg, "deep", "VM 多层 ? 期望单层 Err(\"deep\")");
}

#[test]
fn test_vm_try_block_catches_err() {
    let src = r#"
        fn parse(s: str) -> Result<i64, str> {
            if s == "42" { Result::Ok(42) } else { Result::Err("not 42") }
        }
        fn main() -> i64 {
            let r = try { parse("10")? };
            let msg = match r { Result::Err(m) => m, _ => "?" };
            if msg == "not 42" { 1 } else { 0 }
        }
    "#;
    let result = run_vm(src).unwrap();
    match result {
        Value::Int(1, _) => {}
        v => panic!("VM try 捕获应得到单层 Err 且 match 解出消息, got {:?}", v),
    }
}

// ─── 6. AUDIT-11.4.67：`?` 作用于 Option ⇒ **编译期报错**（不再静默直通）────────
//
// 病灶（三路径"直通"）：VM `Op::Try`（opcode 52）只认 `enum_name == "Result"`
// （非 Result 原样压栈）、解释器 `Try` 同样只认 Result、类型层 `?` 对 Option 不脱壳
// ⇒ `let x = v.get_opt(i)?;` 在 Some/None 下都把**整个 Option** 绑给 x = 静默错值。
//
// 裁定（总师）：**不发明早退语义**（那需推断外层返回类型，另案），先让它**响亮**
// ⇒ `?` 的操作数静态类型是 Option 时，lower 阶段报 `TypeError`，提示 match / or_die。
//
// Result 的 `?` 行为**逐字不变**：本节之上的第 1-5 节（17 个用例，解释器 + VM
// 两路径）即回归守护，本改动未触碰 Result 分支的任何一行。

/// 断言源码在 **lower（编译）阶段**就报 `?`-on-Option 错误，且提示 match / or_die。
fn assert_try_on_option_is_compile_error(src: &str) {
    match run(src) {
        Err(msg) => {
            assert!(
                msg.contains("不支持 Option"),
                "错误应点名「`?` 不支持 Option」且为编译期错误，实际: {msg}"
            );
            assert!(
                msg.contains("match") && msg.contains("or_die"),
                "错误应提示用 match / or_die 消费，实际: {msg}"
            );
        }
        Ok(v) => panic!("期望编译期报错（`?`-on-Option），实际执行成功: {:?}", v),
    }
}

/// 用户函数返回 `Option<i64>`（注解 `Option<T>` → `Generic{base: TypeParam("Option")}`）。
#[test]
fn test_try_on_option_from_user_fn_is_compile_error() {
    assert_try_on_option_is_compile_error(
        r#"
        fn find() -> Option<i64> {
            Option::Some(1)
        }
        fn main() -> i64 {
            let x = find()?;
            x
        }
    "#,
    );
}

/// `Option::None` / `Option::Some(..)` 字面量（→ `Generic{base: Enum("Option")}`）。
#[test]
fn test_try_on_option_literal_is_compile_error() {
    assert_try_on_option_is_compile_error(
        r#"
        fn main() -> i64 {
            let x = Option::None?;
            x
        }
    "#,
    );
    assert_try_on_option_is_compile_error(
        r#"
        fn main() -> i64 {
            let x = Option::Some(1)?;
            x
        }
    "#,
    );
}

/// 红线原始复现：`let x = v.get_opt(i)?;`（真 Option：`Generic{Enum("Option"), [inner]}`）。
#[test]
fn test_try_on_get_opt_is_compile_error() {
    assert_try_on_option_is_compile_error(
        r#"
        let v = Vec::new();
        v.push(7);
        let x = v.get_opt(0)?;
        x
    "#,
    );
}

/// 裸 `Enum("Option")`（`parse_int`/`parse_float` 的静态标注；`Vec.get()/pop()` 同族）：
/// 静态既然说它是 Option，`?` 在其上就没有成立语义 ⇒ 一并响亮（宁响亮，不静默错值）。
#[test]
fn test_try_on_bare_option_native_is_compile_error() {
    assert_try_on_option_is_compile_error(
        r#"
        fn main() -> i64 {
            let x = parse_int("42")?;
            x
        }
    "#,
    );
}

/// 反向守护：`?` 作用在 **Result** 上必须仍然放行（编译期不报错）。
/// 与 `test_try_on_bare_option_native_is_compile_error` 构成对照——同一 native 家族
/// （`parse_int` = Option）与 Result 的处置必须**分开**。
#[test]
fn test_try_on_result_still_allowed() {
    let src = r#"
        fn main() -> i64 {
            let x = Result::Ok(42)?;
            x
        }
    "#;
    let out = run(src).expect("`?`-on-Result 不得变成编译期错误");
    match out {
        Some(Value::Int(42, _)) => {}
        v => panic!("期望 Int(42), got {:?}", v),
    }
}

