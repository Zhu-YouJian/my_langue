//! 正则表达式原语集成测试。
//!
//! 覆盖 6 个 regex native（VM 路径 main.rs + 解释器路径 interpreter/natives.rs）：
//! - `regex_compile(pattern: String)` → `Result<i64>`（返回 1-based handle，0 表示无效）
//! - `regex_match(handle: i64, input: String)` → `bool`
//! - `regex_find(handle: i64, input: String)` → `String`（空串表无匹配）
//! - `regex_find_all(handle: i64, input: String)` → `Vec`
//! - `regex_replace(handle: i64, input: String, repl: String)` → `String`
//! - `regex_split(handle: i64, input: String)` → `Vec`
//!
//! 采用 VM 路径执行（参考 native_parity_test.rs 的 run_vm + register_test_natives 模式）。
//! VM 侧通过 `register_test_natives` 注册 6 个 regex native（复制自 main.rs::register_natives）。
//!
//! 注意：Tenth 字符串中 `\d` 会被转义，需用 `\\d+` 表示正则 `\d+`。
//!
//! 已知限制：解释器路径 `interpreter/mod.rs::eval_expr` 的 `Var` 白名单（约 587 行）
//! 暂未同步 6 个 regex native（仅含 tcp/http），导致解释器报"未定义变量"。
//! 待 runtime 部门补齐该白名单后，可扩展为 VM+解释器 parity 测试。

use std::cell::RefCell;
use std::rc::Rc;

use tenth::compile::bytecode::BytecodeCompiler;
use tenth::hir::lower::Lowerer;
use tenth::lexer::lexer::Lexer;
use tenth::parser::parser::Parser;
use tenth::runtime::value::Value;
use tenth::runtime::vm::Vm;

/// 构造 Result::Ok(value)
fn ok_result(value: Value) -> Value {
    Value::Enum {
        enum_name: "Result".to_string(),
        variant: "Ok".to_string(),
        fields: Rc::new(RefCell::new(vec![("_0".to_string(), value)])),
    }
}

/// 构造 Result::Err(message)
fn err_result(msg: impl Into<String>) -> Value {
    Value::Enum {
        enum_name: "Result".to_string(),
        variant: "Err".to_string(),
        fields: Rc::new(RefCell::new(vec![("_0".to_string(), Value::String(msg.into()))])),
    }
}

/// 注册测试所需的 native（复制自 main.rs::register_natives 的 regex 子集 + 辅助 native）。
/// 6 个 regex native 必须与 main.rs 完全一致；Vec::new 仅为辅助。
fn register_test_natives(vm: &mut Vm) {
    // ── 辅助 native ──
    vm.add_native("Vec::new".into(), |_vm, _args| {
        Ok(Value::Vec(Rc::new(RefCell::new(Vec::new()))))
    });

    // ── 6 个 regex native（复制自 main.rs，须保持同步）──
    vm.add_native("regex_compile".into(), |vm, args| {
        if let Some(Value::String(pattern)) = args.first() {
            match regex::Regex::new(pattern) {
                Ok(re) => {
                    vm.regexes.push(Some(re));
                    let handle = vm.regexes.len() as i64; // 1-based
                    Ok(ok_result(Value::Int(handle)))
                }
                Err(e) => Ok(err_result(format!("正则编译失败: {e}"))),
            }
        } else {
            Ok(err_result("regex_compile 需要 1 个 String 参数"))
        }
    });
    vm.add_native("regex_match".into(), |vm, args| {
        if args.len() < 2 {
            return Ok(Value::Bool(false));
        }
        if let (Value::Int(handle), Value::String(input)) = (&args[0], &args[1]) {
            let idx = *handle as usize;
            if idx == 0 || idx > vm.regexes.len() {
                return Ok(Value::Bool(false));
            }
            if let Some(ref re) = vm.regexes[idx - 1] {
                return Ok(Value::Bool(re.is_match(input)));
            }
            Ok(Value::Bool(false))
        } else {
            Ok(Value::Bool(false))
        }
    });
    vm.add_native("regex_find".into(), |vm, args| {
        if args.len() < 2 {
            return Ok(Value::String(String::new()));
        }
        if let (Value::Int(handle), Value::String(input)) = (&args[0], &args[1]) {
            let idx = *handle as usize;
            if idx == 0 || idx > vm.regexes.len() {
                return Ok(Value::String(String::new()));
            }
            if let Some(ref re) = vm.regexes[idx - 1] {
                if let Some(m) = re.find(input) {
                    return Ok(Value::String(m.as_str().to_string()));
                }
            }
            Ok(Value::String(String::new()))
        } else {
            Ok(Value::String(String::new()))
        }
    });
    vm.add_native("regex_find_all".into(), |vm, args| {
        if args.len() < 2 {
            return Ok(Value::Vec(Rc::new(RefCell::new(Vec::new()))));
        }
        if let (Value::Int(handle), Value::String(input)) = (&args[0], &args[1]) {
            let idx = *handle as usize;
            if idx == 0 || idx > vm.regexes.len() {
                return Ok(Value::Vec(Rc::new(RefCell::new(Vec::new()))));
            }
            if let Some(ref re) = vm.regexes[idx - 1] {
                let collected: Vec<Value> = re
                    .find_iter(input)
                    .map(|m| Value::String(m.as_str().to_string()))
                    .collect();
                return Ok(Value::Vec(Rc::new(RefCell::new(collected))));
            }
            Ok(Value::Vec(Rc::new(RefCell::new(Vec::new()))))
        } else {
            Ok(Value::Vec(Rc::new(RefCell::new(Vec::new()))))
        }
    });
    vm.add_native("regex_replace".into(), |vm, args| {
        if args.len() < 3 {
            return Ok(Value::String(String::new()));
        }
        if let (Value::Int(handle), Value::String(input), Value::String(replacement)) =
            (&args[0], &args[1], &args[2])
        {
            let idx = *handle as usize;
            if idx == 0 || idx > vm.regexes.len() {
                return Ok(Value::String(input.clone()));
            }
            if let Some(ref re) = vm.regexes[idx - 1] {
                let result = re.replace_all(input, replacement.as_str()).into_owned();
                return Ok(Value::String(result));
            }
            Ok(Value::String(input.clone()))
        } else {
            Ok(Value::String(String::new()))
        }
    });
    vm.add_native("regex_split".into(), |vm, args| {
        if args.len() < 2 {
            return Ok(Value::Vec(Rc::new(RefCell::new(Vec::new()))));
        }
        if let (Value::Int(handle), Value::String(input)) = (&args[0], &args[1]) {
            let idx = *handle as usize;
            if idx == 0 || idx > vm.regexes.len() {
                return Ok(Value::Vec(Rc::new(RefCell::new(Vec::new()))));
            }
            if let Some(ref re) = vm.regexes[idx - 1] {
                let collected: Vec<Value> = re
                    .split(input)
                    .map(|s| Value::String(s.to_string()))
                    .collect();
                return Ok(Value::Vec(Rc::new(RefCell::new(collected))));
            }
            Ok(Value::Vec(Rc::new(RefCell::new(Vec::new()))))
        } else {
            Ok(Value::Vec(Rc::new(RefCell::new(Vec::new()))))
        }
    });
}

/// 通过 VM 执行 .th 源码，返回结果。
fn run_vm(src: &str) -> Value {
    let mut lexer = Lexer::new(src);
    let tokens = lexer.tokenize().unwrap_or_else(|e| panic!("词法错误: {}", e));
    let mut parser = Parser::new(tokens);
    let program = parser.parse_program().unwrap_or_else(|e| panic!("语法错误: {}", e));
    let mut lowerer = Lowerer::new();
    let hir = lowerer.lower_program(&program).unwrap_or_else(|e| panic!("HIR 错误: {}", e));

    let mut vm = Vm::new();
    register_test_natives(&mut vm);

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
                });
            }
            Err(e) => panic!("字节码编译错误: {}", e),
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
            Err(e) => panic!("字节码编译错误: {}", e),
        }
        vm.call("main").unwrap_or_else(|e| panic!("VM 执行失败: {}", e))
    } else if vm.has_fn("main") {
        vm.call("main").unwrap_or_else(|e| panic!("VM 执行失败: {}", e))
    } else {
        Value::Unit
    }
}

// ══════════════════════════════════════════════════════════════════════
// 编译测试
// ══════════════════════════════════════════════════════════════════════

// ─── Test 1: regex_compile 成功返回 Ok(handle > 0) ───────────────────────

#[test]
fn test_regex_compile_success() {
    let src = r#"
        let r = regex_compile("\\d+");
        match r {
            Result::Ok(h) => h > 0,
            Result::Err(_) => false,
        }
    "#;
    let v = run_vm(src);
    assert!(matches!(v, Value::Bool(true)), "期望 Bool(true) 表示 handle > 0，got {:?}", v);
}

// ─── Test 2: regex_compile 无效正则返回 Err ───────────────────────────────

#[test]
fn test_regex_compile_invalid() {
    // "(" 是未闭合的分组，编译应失败
    let src = r#"
        let r = regex_compile("(");
        match r {
            Result::Ok(_) => false,
            Result::Err(_) => true,
        }
    "#;
    let v = run_vm(src);
    assert!(matches!(v, Value::Bool(true)), "期望 Bool(true) 表示 Err，got {:?}", v);
}

// ══════════════════════════════════════════════════════════════════════
// 匹配测试
// ══════════════════════════════════════════════════════════════════════

// ─── Test 3: regex_match 匹配成功返回 true ────────────────────────────────

#[test]
fn test_regex_match_true() {
    let src = r#"
        let r = regex_compile("\\d+");
        match r {
            Result::Ok(h) => regex_match(h, "hello123"),
            Result::Err(_) => false,
        }
    "#;
    let v = run_vm(src);
    assert!(matches!(v, Value::Bool(true)), "期望 Bool(true) 表示匹配，got {:?}", v);
}

// ─── Test 4: regex_match 不匹配返回 false ─────────────────────────────────

#[test]
fn test_regex_match_false() {
    let src = r#"
        let r = regex_compile("\\d+");
        match r {
            Result::Ok(h) => regex_match(h, "abc"),
            Result::Err(_) => false,
        }
    "#;
    let v = run_vm(src);
    assert!(matches!(v, Value::Bool(false)), "期望 Bool(false) 表示不匹配，got {:?}", v);
}

// ══════════════════════════════════════════════════════════════════════
// 查找测试
// ══════════════════════════════════════════════════════════════════════

// ─── Test 5: regex_find 返回第一个匹配 ────────────────────────────────────

#[test]
fn test_regex_find() {
    let src = r#"
        let r = regex_compile("\\d+");
        match r {
            Result::Ok(h) => regex_find(h, "hello123world456"),
            Result::Err(_) => "FAIL",
        }
    "#;
    let v = run_vm(src);
    match v {
        Value::String(s) => assert_eq!(s, "123", "期望 \"123\"，got \"{}\"", s),
        v => panic!("期望 String(\"123\")，got {:?}", v),
    }
}

// ─── Test 6: regex_find 无匹配返回空串 ────────────────────────────────────

#[test]
fn test_regex_find_no_match() {
    let src = r#"
        let r = regex_compile("\\d+");
        match r {
            Result::Ok(h) => regex_find(h, "abc"),
            Result::Err(_) => "FAIL",
        }
    "#;
    let v = run_vm(src);
    match v {
        Value::String(s) => assert_eq!(s, "", "期望空串，got \"{}\"", s),
        v => panic!("期望 String(\"\")，got {:?}", v),
    }
}

// ══════════════════════════════════════════════════════════════════════
// 查找全部测试
// ══════════════════════════════════════════════════════════════════════

// ─── Test 7: regex_find_all 返回所有匹配 ──────────────────────────────────

#[test]
fn test_regex_find_all() {
    let src = r#"
        let r = regex_compile("\\d+");
        match r {
            Result::Ok(h) => regex_find_all(h, "a1b2c3"),
            Result::Err(_) => Vec::new(),
        }
    "#;
    let v = run_vm(src);
    match v {
        Value::Vec(rc) => {
            let vec = rc.borrow();
            assert_eq!(vec.len(), 3, "期望 3 个匹配，got {}", vec.len());
            assert!(matches!(&vec[0], Value::String(s) if s == "1"), "第一个匹配期望 \"1\"，got {:?}", vec[0]);
            assert!(matches!(&vec[1], Value::String(s) if s == "2"), "第二个匹配期望 \"2\"，got {:?}", vec[1]);
            assert!(matches!(&vec[2], Value::String(s) if s == "3"), "第三个匹配期望 \"3\"，got {:?}", vec[2]);
        }
        v => panic!("期望 Vec，got {:?}", v),
    }
}

// ══════════════════════════════════════════════════════════════════════
// 替换测试
// ══════════════════════════════════════════════════════════════════════

// ─── Test 8: regex_replace 替换所有匹配 ────────────────────────────────────

#[test]
fn test_regex_replace() {
    // 用 r##"..."## 避免 "# 被解释为 r#"..."# 的结束符
    let src = r##"
        let r = regex_compile("\\d+");
        match r {
            Result::Ok(h) => regex_replace(h, "a1b2c3", "#"),
            Result::Err(_) => "FAIL",
        }
    "##;
    let v = run_vm(src);
    match v {
        Value::String(s) => assert_eq!(s, "a#b#c#", "期望 \"a#b#c#\"，got \"{}\"", s),
        v => panic!("期望 String(\"a#b#c#\")，got {:?}", v),
    }
}

// ══════════════════════════════════════════════════════════════════════
// 分割测试
// ══════════════════════════════════════════════════════════════════════

// ─── Test 9: regex_split 按分隔符分割 ──────────────────────────────────────

#[test]
fn test_regex_split() {
    let src = r#"
        let r = regex_compile(",");
        match r {
            Result::Ok(h) => regex_split(h, "a,b,c"),
            Result::Err(_) => Vec::new(),
        }
    "#;
    let v = run_vm(src);
    match v {
        Value::Vec(rc) => {
            let vec = rc.borrow();
            assert_eq!(vec.len(), 3, "期望 3 段，got {}", vec.len());
            assert!(matches!(&vec[0], Value::String(s) if s == "a"), "第一段期望 \"a\"，got {:?}", vec[0]);
            assert!(matches!(&vec[1], Value::String(s) if s == "b"), "第二段期望 \"b\"，got {:?}", vec[1]);
            assert!(matches!(&vec[2], Value::String(s) if s == "c"), "第三段期望 \"c\"，got {:?}", vec[2]);
        }
        v => panic!("期望 Vec，got {:?}", v),
    }
}

// ══════════════════════════════════════════════════════════════════════
// 无效 handle 测试
// ══════════════════════════════════════════════════════════════════════

// ─── Test 10: 无效 handle 返回安全默认值 ───────────────────────────────────
//
// handle=999 超出空 regexes 表的范围，三个 native 应返回各自的安全默认值：
// - regex_match → false
// - regex_find → ""（空串）
// - regex_replace → 原输入串

#[test]
fn test_regex_invalid_handle() {
    // match → false
    let v = run_vm(r#"regex_match(999, "abc")"#);
    assert!(matches!(v, Value::Bool(false)), "无效 handle match 期望 false，got {:?}", v);

    // find → ""
    let v = run_vm(r#"regex_find(999, "abc")"#);
    match v {
        Value::String(s) => assert_eq!(s, "", "无效 handle find 期望空串，got \"{}\"", s),
        v => panic!("期望 String(\"\")，got {:?}", v),
    }

    // replace → 原输入串
    let v = run_vm(r#"regex_replace(999, "abc", "X")"#);
    match v {
        Value::String(s) => assert_eq!(s, "abc", "无效 handle replace 期望原串 \"abc\"，got \"{}\"", s),
        v => panic!("期望 String(\"abc\")，got {:?}", v),
    }
}

// ══════════════════════════════════════════════════════════════════════
// 邮箱正则测试
// ══════════════════════════════════════════════════════════════════════

// ─── Test 11: 邮箱正则匹配验证 ─────────────────────────────────────────────
//
// 正则：[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}
// - "user@example.com" 应匹配 → m1 = true
// - "not-email" 应不匹配 → m2 = false
// 编码：m1=true & m2=false → 2；其他组合见嵌套 if

#[test]
fn test_regex_email() {
    let src = r#"
        let r = regex_compile("[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}");
        match r {
            Result::Ok(h) => {
                let m1 = regex_match(h, "user@example.com");
                let m2 = regex_match(h, "not-email");
                if m1 {
                    if m2 { 1 } else { 2 }
                } else {
                    if m2 { 0 } else { 1 }
                }
            },
            Result::Err(_) => -1,
        }
    "#;
    let v = run_vm(src);
    match v {
        Value::Int(n) => assert_eq!(n, 2, "期望 2（m1=true, m2=false），got {}", n),
        v => panic!("期望 Int(2)，got {:?}", v),
    }
}
