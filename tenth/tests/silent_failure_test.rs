//! 阶段1-静默失败（层1）集成测试。
//!
//! 核心命题：**「可能失败/为空」的值（Result/Option），不能静默丢弃——
//! 必须显式处理（or_die / ? / assume_ok）**。
//!
//! 覆盖：
//! 1. `or_die(x, msg)`：Ok/Some → 取出内部值；Err/None → panic（消息含自定义 msg）
//! 2. `assume_ok(x)`：不做检查直接取内部值（用户负责）
//! 3. `?` 传播不回归（解释器 + VM 两条路径）
//! 4. 丢弃 warning：Result/Option 值作为语句被丢弃 → 编译期 TenthWarning；
//!    被 `?`/or_die/assume_ok/match 消费、函数末尾表达式（返回值）→ 不触发
//!
//! 双侧一致性：or_die / assume_ok 在 VM（runtime/natives.rs::register_all_natives）
//! 与解释器（runtime/interpreter/natives.rs::call_named_fn）双重注册，
//! 本测试对两条路径都验证。

use tenth::compile::bytecode::BytecodeCompiler;
use tenth::error::TenthWarning;
use tenth::hir::lower::Lowerer;
use tenth::lexer::lexer::Lexer;
use tenth::parser::parser::Parser;
use tenth::runtime::interpreter::Interpreter;
use tenth::runtime::natives::register_all_natives;
use tenth::runtime::value::Value;
use tenth::runtime::vm::Vm;

/// 通过解释器执行 .th 源码，返回结果。
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

/// 通过字节码 VM 执行 .th 源码，返回结果（注册全部 native）。
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
                vm.set_global(func.name.clone(), Value::FnRef {
                    name: func.name.clone(),
                    params: func.params.clone(),
                    return_type: func.return_type.clone(),
                    captures: vec![],
                });
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

/// Lower 源码，返回 warnings 列表。
fn lower_warnings(src: &str) -> Vec<TenthWarning> {
    let mut lexer = Lexer::new(src);
    let tokens = lexer.tokenize().expect("lex");
    let mut parser = Parser::new(tokens);
    let program = parser.parse_program().expect("parse");
    let mut lowerer = Lowerer::new();
    let hir = lowerer.lower_program(&program).expect("lower");
    hir.warnings
}

/// 断言 warnings 中至少一条包含指定子串。
fn assert_has_warning(warnings: &[TenthWarning], part: &str) {
    let found = warnings.iter().any(|w| w.message.contains(part));
    assert!(
        found,
        "期望 warning 包含 '{}'\n实际 warnings: {:?}",
        part,
        warnings.iter().map(|w| &w.message).collect::<Vec<_>>()
    );
}

/// 断言 warnings 中没有一条包含指定子串。
fn assert_no_warning_containing(warnings: &[TenthWarning], part: &str) {
    let found = warnings.iter().any(|w| w.message.contains(part));
    assert!(
        !found,
        "不应触发含 '{}' 的 warning\n实际 warnings: {:?}",
        part,
        warnings.iter().map(|w| &w.message).collect::<Vec<_>>()
    );
}

// ══════════════════════════════════════════════════════════════════════
// M1a: or_die — 正常路径（Ok → 取出内部值），解释器 + VM
// ══════════════════════════════════════════════════════════════════════

#[test]
fn test_or_die_ok_interpreter() {
    let src = r#"
        fn main() -> i64 {
            let x = or_die(Result::Ok(42), "should not fail");
            x
        }
    "#;
    match run(src).unwrap() {
        Some(Value::Int(42, _)) => {}
        v => panic!("期望 Int(42), got {:?}", v),
    }
}

#[test]
fn test_or_die_ok_vm() {
    let src = r#"
        fn main() -> i64 {
            let x = or_die(Result::Ok(99), "should not fail");
            x
        }
    "#;
    match run_vm(src).unwrap() {
        Value::Int(99, _) => {}
        v => panic!("VM: 期望 Int(99), got {:?}", v),
    }
}

#[test]
fn test_or_die_option_some_interpreter() {
    let src = r#"
        fn main() -> i64 {
            let x = or_die(Option::Some(7), "no value");
            x
        }
    "#;
    match run(src).unwrap() {
        Some(Value::Int(7, _)) => {}
        v => panic!("期望 Int(7), got {:?}", v),
    }
}

#[test]
fn test_or_die_option_some_vm() {
    let src = r#"
        fn main() -> i64 {
            let x = or_die(Option::Some(8), "no value");
            x
        }
    "#;
    match run_vm(src).unwrap() {
        Value::Int(8, _) => {}
        v => panic!("VM: 期望 Int(8), got {:?}", v),
    }
}

#[test]
fn test_or_die_string_value_interpreter() {
    let src = r#"
        fn main() -> str {
            let s = or_die(Result::Ok("hello"), "no");
            s
        }
    "#;
    match run(src).unwrap() {
        Some(Value::String(s)) => assert_eq!(s, "hello"),
        v => panic!("期望 String(hello), got {:?}", v),
    }
}

// ══════════════════════════════════════════════════════════════════════
// M1b: or_die — 失败路径（Err/None → panic，消息含自定义 msg），解释器 + VM
// ══════════════════════════════════════════════════════════════════════

#[test]
fn test_or_die_err_panics_with_message_interpreter() {
    let src = r#"
        fn main() -> i64 {
            let x = or_die(Result::Err("boom"), "数据库查询失败");
            x
        }
    "#;
    let err = run(src).unwrap_err();
    assert!(
        err.contains("数据库查询失败"),
        "错误信息应包含自定义消息，实际: {}",
        err
    );
}

#[test]
fn test_or_die_err_panics_with_message_vm() {
    let src = r#"
        fn main() -> i64 {
            let x = or_die(Result::Err("boom"), "配置文件缺失");
            x
        }
    "#;
    let err = run_vm(src).unwrap_err();
    assert!(
        err.contains("配置文件缺失"),
        "VM: 错误信息应包含自定义消息，实际: {}",
        err
    );
}

#[test]
fn test_or_die_none_panics_with_message_interpreter() {
    let src = r#"
        fn main() -> i64 {
            let x = or_die(Option::None, "找不到元素");
            x
        }
    "#;
    let err = run(src).unwrap_err();
    assert!(
        err.contains("找不到元素"),
        "错误信息应包含自定义消息，实际: {}",
        err
    );
}

// ══════════════════════════════════════════════════════════════════════
// M1c: assume_ok — 不做检查直接取内部值，解释器 + VM
// ══════════════════════════════════════════════════════════════════════

#[test]
fn test_assume_ok_result_interpreter() {
    let src = r#"
        fn main() -> i64 {
            let x = assume_ok(Result::Ok(5));
            x
        }
    "#;
    match run(src).unwrap() {
        Some(Value::Int(5, _)) => {}
        v => panic!("期望 Int(5), got {:?}", v),
    }
}

#[test]
fn test_assume_ok_result_vm() {
    let src = r#"
        fn main() -> i64 {
            let x = assume_ok(Result::Ok(6));
            x
        }
    "#;
    match run_vm(src).unwrap() {
        Value::Int(6, _) => {}
        v => panic!("VM: 期望 Int(6), got {:?}", v),
    }
}

#[test]
fn test_assume_ok_option_some_vm() {
    let src = r#"
        fn main() -> i64 {
            let x = assume_ok(Option::Some(3));
            x
        }
    "#;
    match run_vm(src).unwrap() {
        Value::Int(3, _) => {}
        v => panic!("VM: 期望 Int(3), got {:?}", v),
    }
}

// ══════════════════════════════════════════════════════════════════════
// M1d: `?` 传播不回归（解释器 + VM）
// ══════════════════════════════════════════════════════════════════════

#[test]
fn test_try_operator_no_regression_interpreter() {
    let src = r#"
        fn parse(s: str) -> Result<i64, str> {
            if s == "42" { Result::Ok(42) } else { Result::Err("not 42") }
        }
        fn main() -> i64 {
            let a = parse("42")?;
            a
        }
    "#;
    match run(src).unwrap() {
        Some(Value::Int(42, _)) => {}
        v => panic!("期望 Int(42), got {:?}", v),
    }
}

#[test]
fn test_try_operator_err_propagation_vm() {
    let src = r#"
        fn parse(s: str) -> Result<i64, str> {
            if s == "42" { Result::Ok(42) } else { Result::Err("not 42") }
        }
        fn main() -> i64 {
            let a = parse("10")?;
            a
        }
    "#;
    let result = run_vm(src).unwrap();
    match result {
        Value::Enum { enum_name, variant, .. } => {
            assert_eq!(enum_name, "Result");
            assert_eq!(variant, "Err");
        }
        v => panic!("VM: 期望 Result::Err, got {:?}", v),
    }
}

// ══════════════════════════════════════════════════════════════════════
// M2: 丢弃 warning — 触发场景
// ══════════════════════════════════════════════════════════════════════

#[test]
fn test_discarded_result_native_triggers_warning() {
    // read_line 返回 Result；作为语句被丢弃 → warning
    let src = r#"
        fn main() {
            read_line();
            println("ok");
        }
    "#;
    let warnings = lower_warnings(src);
    assert_has_warning(&warnings, "Result 被忽略");
    assert_has_warning(&warnings, "or_die");
}

#[test]
fn test_discarded_result_from_user_fn_triggers_warning() {
    // 用户函数返回 Result<i64, str>；作为语句被丢弃 → warning
    let src = r#"
        fn db_query() -> Result<i64, str> {
            Result::Ok(42)
        }
        fn main() {
            db_query();
            println("ok");
        }
    "#;
    let warnings = lower_warnings(src);
    assert_has_warning(&warnings, "Result 被忽略");
}

#[test]
fn test_discarded_option_triggers_warning() {
    let src = r#"
        fn find() -> Option<i64> {
            Option::Some(1)
        }
        fn main() {
            find();
            println("ok");
        }
    "#;
    let warnings = lower_warnings(src);
    assert_has_warning(&warnings, "Option 被忽略");
}

#[test]
fn test_discarded_result_has_line_col() {
    let src = r#"
fn main() {
    let a = 1;
    read_line();
    let b = 2;
}
"#;
    let warnings = lower_warnings(src);
    let w = warnings.iter().find(|w| w.message.contains("Result 被忽略"))
        .expect("应有 Result 丢弃 warning");
    // 第 1 行为空；第 2 行 `fn main() {`，第 3 行 `let a = 1;`，
    // 第 4 行 `read_line();`（被丢弃的表达式语句）→ warning 应定位到第 4 行
    assert_eq!(w.line, 4, "warning 应定位到 read_line 行，实际 line={}", w.line);
    assert!(w.col >= 1, "warning 应有列号，实际 col={}", w.col);
}

// ══════════════════════════════════════════════════════════════════════
// M2: 丢弃 warning — 不触发场景
// ══════════════════════════════════════════════════════════════════════

#[test]
fn test_consumed_by_try_no_warning() {
    // `?` 消费 Result：不算丢弃
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
    let warnings = lower_warnings(src);
    assert_no_warning_containing(&warnings, "被忽略");
}

#[test]
fn test_consumed_by_or_die_no_warning() {
    let src = r#"
        fn main() -> i64 {
            let x = or_die(Result::Ok(42), "no");
            x
        }
    "#;
    let warnings = lower_warnings(src);
    assert_no_warning_containing(&warnings, "被忽略");
}

#[test]
fn test_consumed_by_assume_ok_no_warning() {
    let src = r#"
        fn main() -> i64 {
            let x = assume_ok(Result::Ok(42));
            x
        }
    "#;
    let warnings = lower_warnings(src);
    assert_no_warning_containing(&warnings, "被忽略");
}

#[test]
fn test_consumed_by_match_no_warning() {
    let src = r#"
        fn db_query() -> Result<i64, str> {
            Result::Ok(42)
        }
        fn main() -> i64 {
            let r = db_query();
            match r {
                Result::Ok(v) => v,
                Result::Err(_) => -1,
            }
        }
    "#;
    let warnings = lower_warnings(src);
    assert_no_warning_containing(&warnings, "被忽略");
}

#[test]
fn test_bound_to_variable_no_warning() {
    // Result 绑定到变量（let），不算"丢弃"
    let src = r#"
        fn main() {
            let r = read_line();
            println("ok");
        }
    "#;
    let warnings = lower_warnings(src);
    assert_no_warning_containing(&warnings, "被忽略");
}

#[test]
fn test_function_last_expr_is_return_value_no_warning() {
    // 函数最后一个表达式作为返回值：不算丢弃
    let src = r#"
        fn db_query() -> Result<i64, str> {
            Result::Ok(42)
        }
        fn main() -> Result<i64, str> {
            db_query()
        }
    "#;
    let warnings = lower_warnings(src);
    assert_no_warning_containing(&warnings, "被忽略");
}

#[test]
fn test_result_as_argument_no_warning() {
    // Result 作为 println 参数（被使用）：不算丢弃
    let src = r#"
        fn main() {
            println(read_line());
        }
    "#;
    let warnings = lower_warnings(src);
    assert_no_warning_containing(&warnings, "被忽略");
}

// ══════════════════════════════════════════════════════════════════════
// 端到端：or_die 修复"静默失败"演示（解释器 + VM 一致）
// ══════════════════════════════════════════════════════════════════════

#[test]
fn test_or_die_recovers_error_then_continues_vm() {
    // 第一个 or_die 失败 → panic；但用 Result::Ok 包装时正常继续
    let src = r#"
        fn main() -> i64 {
            let ok = or_die(Result::Ok(10), "no");
            let ok2 = or_die(Result::Ok(32), "no");
            ok + ok2
        }
    "#;
    match run_vm(src).unwrap() {
        Value::Int(42, _) => {}
        v => panic!("VM: 期望 Int(42), got {:?}", v),
    }
}

// ══════════════════════════════════════════════════════════════════════
// M3.4: "误用"拦截 warning — 触发场景（Result/Option 被当作普通值使用）
// ══════════════════════════════════════════════════════════════════════

#[test]
fn test_method_call_on_result_triggers_misuse_warning() {
    // db_query().len() 把 Result 当成功值使用 → warning
    let src = r#"
        fn db_query() -> Result<i64, str> {
            Result::Ok(42)
        }
        fn main() -> i64 {
            db_query().len()
        }
    "#;
    let warnings = lower_warnings(src);
    assert_has_warning(&warnings, "Result 值被当作普通值使用");
    assert_has_warning(&warnings, "方法 'len'");
}

#[test]
fn test_index_on_result_triggers_misuse_warning() {
    // db_query()[0] 索引访问 Result → warning
    let src = r#"
        fn db_query() -> Result<i64, str> {
            Result::Ok(42)
        }
        fn main() -> i64 {
            db_query()[0]
        }
    "#;
    let warnings = lower_warnings(src);
    assert_has_warning(&warnings, "Result 值被当作普通值使用");
    assert_has_warning(&warnings, "索引访问");
}

#[test]
fn test_field_access_on_result_triggers_misuse_warning() {
    // db_query().field 字段访问 Result → warning
    let src = r#"
        fn db_query() -> Result<i64, str> {
            Result::Ok(42)
        }
        fn main() -> i64 {
            db_query().field
        }
    "#;
    let warnings = lower_warnings(src);
    assert_has_warning(&warnings, "Result 值被当作普通值使用");
    assert_has_warning(&warnings, "字段访问");
}

#[test]
fn test_arithmetic_on_result_triggers_misuse_warning() {
    // db_query() + 1 算术运算（Result 参与）→ warning
    let src = r#"
        fn db_query() -> Result<i64, str> {
            Result::Ok(42)
        }
        fn main() -> i64 {
            db_query() + 1
        }
    "#;
    let warnings = lower_warnings(src);
    assert_has_warning(&warnings, "Result 值被当作普通值使用");
    assert_has_warning(&warnings, "算术运算");
}

#[test]
fn test_method_call_on_option_triggers_misuse_warning() {
    // find().len()：Option 接收者方法调用 → warning
    let src = r#"
        fn find() -> Option<i64> {
            Option::Some(1)
        }
        fn main() -> i64 {
            find().len()
        }
    "#;
    let warnings = lower_warnings(src);
    assert_has_warning(&warnings, "Option 值被当作普通值使用");
    assert_has_warning(&warnings, "方法 'len'");
}

#[test]
fn test_method_call_on_read_line_result_triggers_misuse_warning() {
    // read_line 返回裸 Type::Enum("Result")（resolve_builtin 注册）——同样拦截
    let src = r#"
        fn main() -> i64 {
            read_line().len()
        }
    "#;
    let warnings = lower_warnings(src);
    assert_has_warning(&warnings, "Result 值被当作普通值使用");
    assert_has_warning(&warnings, "方法 'len'");
}

#[test]
fn test_misuse_warning_has_line_col() {
    let src = r#"
fn db_query() -> Result<i64, str> {
    Result::Ok(42)
}
fn main() {
    let a = 1;
    db_query().len();
    let b = 2;
}
"#;
    let warnings = lower_warnings(src);
    let w = warnings.iter().find(|w| w.message.contains("被当作普通值使用"))
        .expect("应有误用 warning");
    // 第 1 行为空；第 2 行 `fn db_query()...`，第 5 行 `fn main() {`，
    // 第 6 行 `let a = 1;`，第 7 行 `db_query().len();` → warning 应定位到第 7 行
    assert_eq!(w.line, 7, "warning 应定位到 db_query().len() 行，实际 line={}", w.line);
    assert!(w.col >= 1, "warning 应有列号，实际 col={}", w.col);
}

// ══════════════════════════════════════════════════════════════════════
// M3.4: "误用"拦截 warning — 不触发场景（合法消费/传输/保守放行）
// ══════════════════════════════════════════════════════════════════════

#[test]
fn test_consumed_by_try_no_misuse_warning() {
    // `?` 消费 Result：不算误用
    let src = r#"
        fn db_query() -> Result<i64, str> {
            Result::Ok(42)
        }
        fn main() -> i64 {
            let a = db_query()?;
            a
        }
    "#;
    let warnings = lower_warnings(src);
    assert_no_warning_containing(&warnings, "被当作普通值使用");
}

#[test]
fn test_consumed_by_or_die_no_misuse_warning() {
    // or_die 消费 Result（其形参即 Result）：不算误用
    let src = r#"
        fn db_query() -> Result<i64, str> {
            Result::Ok(42)
        }
        fn main() -> i64 {
            or_die(db_query(), "db fail")
        }
    "#;
    let warnings = lower_warnings(src);
    assert_no_warning_containing(&warnings, "被当作普通值使用");
}

#[test]
fn test_consumed_by_assume_ok_no_misuse_warning() {
    let src = r#"
        fn db_query() -> Result<i64, str> {
            Result::Ok(42)
        }
        fn main() -> i64 {
            assume_ok(db_query())
        }
    "#;
    let warnings = lower_warnings(src);
    assert_no_warning_containing(&warnings, "被当作普通值使用");
}

#[test]
fn test_consumed_by_match_no_misuse_warning() {
    // match scrutinee 消费 Result：不算误用
    let src = r#"
        fn db_query() -> Result<i64, str> {
            Result::Ok(42)
        }
        fn main() -> i64 {
            match db_query() {
                Result::Ok(v) => v,
                Result::Err(_) => -1,
            }
        }
    "#;
    let warnings = lower_warnings(src);
    assert_no_warning_containing(&warnings, "被当作普通值使用");
}

#[test]
fn test_try_block_no_misuse_warning() {
    // try 块内 `?` 消费：不算误用
    let src = r#"
        fn db_query() -> Result<i64, str> {
            Result::Ok(42)
        }
        fn main() -> Result<i64, str> {
            try {
                let a = db_query()?;
                a + 1
            }
        }
    "#;
    let warnings = lower_warnings(src);
    assert_no_warning_containing(&warnings, "被当作普通值使用");
}

#[test]
fn test_return_transport_no_misuse_warning() {
    // 函数返回 Result 是传输（合法），不算误用
    let src = r#"
        fn db_query() -> Result<i64, str> {
            Result::Ok(42)
        }
        fn main() -> Result<i64, str> {
            db_query()
        }
    "#;
    let warnings = lower_warnings(src);
    assert_no_warning_containing(&warnings, "被当作普通值使用");
}

#[test]
fn test_let_binding_transport_no_misuse_warning() {
    // let 绑定 Result（保留值）是传输，不算误用
    let src = r#"
        fn db_query() -> Result<i64, str> {
            Result::Ok(42)
        }
        fn main() {
            let r = db_query();
            println("ok");
        }
    "#;
    let warnings = lower_warnings(src);
    assert_no_warning_containing(&warnings, "被当作普通值使用");
}

#[test]
fn test_container_transport_no_misuse_warning() {
    // Result 存入 Vec 是传输（合法），不算误用
    let src = r#"
        fn db_query() -> Result<i64, str> {
            Result::Ok(42)
        }
        fn main() {
            let v = Vec::new();
            v.push(db_query());
            println("ok");
        }
    "#;
    let warnings = lower_warnings(src);
    assert_no_warning_containing(&warnings, "被当作普通值使用");
}

#[test]
fn test_bare_enum_option_method_no_misuse_warning() {
    // 保守放行：裸 Type::Enum("Option")（parse_int/get/pop 等静态误标）上的
    // 方法调用不触发——`Vec.get(i).trim()` 依赖此行为，避免误报（宁可少报）
    let src = r#"
        fn main() -> str {
            "123".parse_int().to_string()
        }
    "#;
    let warnings = lower_warnings(src);
    assert_no_warning_containing(&warnings, "被当作普通值使用");
}

#[test]
fn test_comparison_on_result_no_misuse_warning() {
    // 比较运算语义模糊，保守放行（不报）
    let src = r#"
        fn db_query() -> Result<i64, str> {
            Result::Ok(42)
        }
        fn main() -> bool {
            db_query() == Result::Ok(42)
        }
    "#;
    let warnings = lower_warnings(src);
    assert_no_warning_containing(&warnings, "被当作普通值使用");
}

#[test]
fn test_discard_warning_regression_still_fires() {
    // 层1 丢弃 warning 回归：db_query(); 作为语句仍触发"被忽略"（非"误用"）
    let src = r#"
        fn db_query() -> Result<i64, str> {
            Result::Ok(42)
        }
        fn main() {
            db_query();
            println("ok");
        }
    "#;
    let warnings = lower_warnings(src);
    assert_has_warning(&warnings, "Result 被忽略");
    assert_no_warning_containing(&warnings, "被当作普通值使用");
}

// ══════════════════════════════════════════════════════════════════════
// M3.4: 运行行为零变化对拍（VM + 解释器，合法消费程序结果一致）
// ══════════════════════════════════════════════════════════════════════

#[test]
fn test_misuse_legit_consumption_runtime_parity() {
    // 合法消费（or_die + match）在 VM/解释器双侧运行行为一致：
    // M3.4 是纯编译期检查，不改变运行语义。
    let src = r#"
        fn parse(s: str) -> Result<i64, str> {
            if s == "42" { Result::Ok(42) } else { Result::Err("not 42") }
        }
        fn main() -> i64 {
            let a = or_die(parse("42"), "parse failed");
            a + 1
        }
    "#;
    match run(src).unwrap() {
        Some(Value::Int(43, _)) => {}
        v => panic!("解释器: 期望 Int(43), got {:?}", v),
    }
    match run_vm(src).unwrap() {
        Value::Int(43, _) => {}
        v => panic!("VM: 期望 Int(43), got {:?}", v),
    }
}
