//! L2.3a 修复回归测试：语言层两个缺口
//!
//! a2（解释器）：`int + Vec.get(i)` 类型不匹配 → collections/iter.th 的
//!   sum/product 在解释器路径失败（VM 正常）。根因：解释器 Vec.push 用
//!   Value::Shared 包裹元素，Vec.get(i) 返回 Shared，eval_binary 加法未解壳。
//!   修复：eval_binary 入口对 Shared/Ref/MutRef 操作数统一解壳（binary.rs），
//!   对齐 VM 行为（VM 的 Vec 元素不包裹）。
//!
//! a3（VM）：VM 未注册 str_add native → f-string 在 VM 路径失败（报
//!   "未定义的函数 'str_add'"；bytecode.rs 编译 InterpolatedString 用
//!   CallN("str_add", 2)）。修复：runtime/natives.rs::register_all_natives
//!   补 str_add(String, String) 注册。
//!
//! 测试走真实运行环境：VM 用 register_all_natives（含 str_add），解释器
//! 走 Interpreter；std 模块通过 Lowerer::with_search_paths 加载。

use tenth::lexer::lexer::Lexer;
use tenth::parser::parser::Parser;
use tenth::hir::lower::Lowerer;
use tenth::runtime::vm::Vm;
use tenth::runtime::interpreter::Interpreter;
use tenth::runtime::value::Value;
use tenth::compile::bytecode::BytecodeCompiler;
use tenth::runtime::natives::register_all_natives;

/// 构造与 main.rs::source_to_hir 一致的 std 搜索路径。
/// 兼容两种 cwd：工作区根（tenth/std 存在）与 tenth/（std 存在）。
fn build_search_paths() -> Vec<String> {
    let mut search_paths = Vec::new();
    if let Ok(cwd) = std::env::current_dir() {
        search_paths.push(cwd.to_string_lossy().to_string());
    }
    let std_dev = std::path::Path::new("tenth/std");
    if std_dev.exists() {
        if let Some(parent) = std_dev.parent() {
            search_paths.push(parent.to_string_lossy().to_string());
        }
        search_paths.push(std_dev.to_string_lossy().to_string());
    }
    let std_local = std::path::Path::new("std");
    if std_local.exists() {
        if let Some(parent) = std_local.parent() {
            search_paths.push(parent.to_string_lossy().to_string());
        }
        search_paths.push(std_local.to_string_lossy().to_string());
    }
    search_paths
}

fn lower_with_std(src: &str) -> Result<tenth::hir::hir::HirProgram, String> {
    let mut lexer = Lexer::new(src);
    let tokens = lexer.tokenize().map_err(|e| e.to_string())?;
    let mut parser = Parser::new(tokens);
    let program = parser.parse_program().map_err(|e| e.to_string())?;
    let mut lowerer = Lowerer::with_search_paths(build_search_paths());
    lowerer.lower_program(&program).map_err(|e| e.to_string())
}

/// 通过 VM 执行 .th 源码（register_all_natives 注册全部 native，含 str_add）。
fn run_vm(src: &str) -> Result<Value, String> {
    let hir = lower_with_std(src)?;

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

/// 通过解释器执行 .th 源码。
fn run_interp(src: &str) -> Result<Value, String> {
    let hir = lower_with_std(src)?;
    let mut interp = Interpreter::new(&hir);
    interp.execute_program(&hir)
        .map(|v| v.unwrap_or(Value::Unit))
        .map_err(|e| e.to_string())
}

fn values_eq(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::Int(x, _), Value::Int(y, _)) => x == y,
        (Value::Float(x), Value::Float(y)) => (x - y).abs() < 1e-9,
        (Value::Float32(x), Value::Float32(y)) => (x - y).abs() < 1e-6,
        (Value::Bool(x), Value::Bool(y)) => x == y,
        (Value::String(x), Value::String(y)) => x == y,
        (Value::Unit, Value::Unit) => true,
        _ => false,
    }
}

fn assert_parity(src: &str) -> Value {
    let vm_res = run_vm(src).unwrap_or_else(|e| panic!("VM 执行失败: {}\n源码: {}", e, src));
    let interp_res = run_interp(src).unwrap_or_else(|e| panic!("解释器执行失败: {}\n源码: {}", e, src));
    assert!(
        values_eq(&vm_res, &interp_res),
        "VM 与解释器结果不一致\n源码: {}\nVM: {:?}\n解释器: {:?}",
        src, vm_res, interp_res
    );
    vm_res
}

// ══════════════════════════════════════════════════════════════════
// a2：解释器路径 `int + Vec.get(i)`（sum/product）—— Shared 解壳
// ══════════════════════════════════════════════════════════════════

const SUM_SRC: &str = r#"
use std::collections::iter::sum
use std::collections::collections::product
let v = Vec::new();
v.push(1);
v.push(2);
v.push(3);
let w = Vec::new();
w.push(2);
w.push(3);
w.push(4);
sum(v) + product(w)
"#;

#[test]
fn test_a2_sum_interp_path() {
    // 解释器路径（此前报"加法类型不匹配"）：sum([1,2,3]) == 6
    let src = r#"
use std::collections::iter::sum
let v = Vec::new();
v.push(1);
v.push(2);
v.push(3);
sum(v)
"#;
    let v = run_interp(src).expect("解释器执行失败");
    match v {
        Value::Int(n, _) => assert_eq!(n, 6, "sum 应为 6，实际 {}", n),
        other => panic!("期望 Int(6)，实际 {:?}", other),
    }
}

#[test]
fn test_a2_product_interp_path() {
    // 解释器路径：product([2,3,4]) == 24
    let src = r#"
use std::collections::collections::product
let w = Vec::new();
w.push(2);
w.push(3);
w.push(4);
product(w)
"#;
    let v = run_interp(src).expect("解释器执行失败");
    match v {
        Value::Int(n, _) => assert_eq!(n, 24, "product 应为 24，实际 {}", n),
        other => panic!("期望 Int(24)，实际 {:?}", other),
    }
}

#[test]
fn test_a2_sum_product_parity() {
    // VM 与解释器路径一致：sum([1,2,3]) + product([2,3,4]) == 30
    let v = assert_parity(SUM_SRC);
    match v {
        Value::Int(n, _) => assert_eq!(n, 30, "sum+product 应为 30，实际 {}", n),
        other => panic!("期望 Int(30)，实际 {:?}", other),
    }
}

// ══════════════════════════════════════════════════════════════════
// a3：VM 路径字符串拼接 / f-string / json —— str_add native 注册
// ══════════════════════════════════════════════════════════════════

#[test]
fn test_a3_str_add_vm() {
    // VM 路径 `"a" + "b"`（String+String 走 VM add_priv，回归守护）
    let src = r#"
"hello " + "world"
"#;
    let v = run_vm(src).expect("VM 执行失败");
    match v {
        Value::String(s) => assert_eq!(s, "hello world", "拼接结果应为 'hello world'，实际 '{}'", s),
        other => panic!("期望 String(\"hello world\")，实际 {:?}", other),
    }
}

#[test]
fn test_a3_fstring_vm() {
    // VM 路径 f-string（bytecode 编译为 str_add 调用，修复核心）
    let src = r#"
let name = "Tenth";
"f-string: {name}"
"#;
    let v = run_vm(src).expect("VM 执行失败");
    match v {
        Value::String(s) => assert_eq!(s, "f-string: Tenth", "f-string 结果应为 'f-string: Tenth'，实际 '{}'", s),
        other => panic!("期望 String(\"f-string: Tenth\")，实际 {:?}", other),
    }
}

#[test]
fn test_a3_fstring_multi_part_vm() {
    // 多段 f-string（literal + expr + literal，需要多次 str_add）
    let src = r#"
let a = "A";
let b = "B";
"[{a}-{b}]"
"#;
    let v = run_vm(src).expect("VM 执行失败");
    match v {
        Value::String(s) => assert_eq!(s, "[A-B]", "多段 f-string 结果应为 '[A-B]'，实际 '{}'", s),
        other => panic!("期望 String(\"[A-B]\")，实际 {:?}", other),
    }
}

#[test]
fn test_a3_fstring_parity() {
    // f-string VM 与解释器结果一致（字符串插值；数字插值 {n} 走 format 模板
    // 校验属既有 format native 限制，不在 L2.3a 边界，此处不覆盖）
    let src = r#"
let name = "Tenth";
let city = "Beijing";
"hello {name}, from {city}"
"#;
    let v = assert_parity(src);
    match v {
        Value::String(s) => assert_eq!(s, "hello Tenth, from Beijing", "f-string parity 结果应为 'hello Tenth, from Beijing'，实际 '{}'", s),
        other => panic!("期望 String，实际 {:?}", other),
    }
}

#[test]
fn test_a3_json_vm_path() {
    // VM 路径 json 字符串解析（json.th 纯 Tenth 实现，依赖 Vec.get/String 拼接）
    let src = r#"
use std::json::json::parse
let obj = parse("{\"name\": \"Alice\", \"age\": 30}");
obj.get("name")
"#;
    let v = run_vm(src).expect("VM 执行失败");
    match v {
        Value::String(s) => assert_eq!(s, "Alice", "json 解析 name 应为 'Alice'，实际 '{}'", s),
        other => panic!("期望 String(\"Alice\")，实际 {:?}", other),
    }
}

#[test]
fn test_a3_json_parity() {
    // json parse VM 与解释器一致（a2 解壳也保障解释器路径的字符串拼接）
    let src = r#"
use std::json::json::parse
let obj = parse("{\"name\": \"Alice\", \"age\": 30}");
obj.get("name")
"#;
    let v = assert_parity(src);
    match v {
        Value::String(s) => assert_eq!(s, "Alice", "json parity name 应为 'Alice'，实际 '{}'", s),
        other => panic!("期望 String(\"Alice\")，实际 {:?}", other),
    }
}

#[test]
fn test_a3_toml_vm_path() {
    // VM 路径 toml 解析（同样依赖字符串拼接/比较，回归守护）
    let src = r#"
use std::toml::toml::parse
let t = parse("name = \"Tenth\"\nversion = \"0.1\"\n");
t.get("name")
"#;
    let v = run_vm(src).expect("VM 执行失败");
    match v {
        Value::String(s) => assert_eq!(s, "Tenth", "toml 解析 name 应为 'Tenth'，实际 '{}'", s),
        other => panic!("期望 String(\"Tenth\")，实际 {:?}", other),
    }
}
