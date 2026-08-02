//! JIT 枚举字面量字段名/值颠倒 bug 回归测试（2026-08-01 运行时部修复）。
//!
//! 背景：CLI/JIT 路径下，Tenth 源码构造的枚举字面量 `Result::Ok(42)` /
//! `Option::Some(x)` 经 `or_die`/`assume_ok` 取内部值返回的是字段名 `_0`
//! 而非值（`tenth.exe run` 打印 `_0`，解释器路径打印 42）。
//!
//! 根因（已修复）：
//! - `compile/bytecode.rs` 对 EnumLiteral 每字段压 `[value, name]` 两值（2 个栈槽）；
//! - VM 的 MakeEnum 正确弹 2n 个；但 JIT translator 的 MakeEnum 只弹 field_count 个，
//!   导致 host_make_enum 拿到错误的 (name, value) 配对（把字段名当值）。
//! - 修复：translator 弹 2×field_count；host_make_enum 读 2×field_count 并
//!   按 [value,name] 配对成源码序（镜像 host_new_struct）；VM MakeEnum 去
//!   reverse() 统一三路径字段序。
//!
//! 本测试对 解释器 / 字节码 VM / JIT 三条路径逐一验证一致性。

use tenth::compile::bytecode::BytecodeCompiler;
use tenth::compile::jit;
use tenth::hir::lower::Lowerer;
use tenth::lexer::lexer::Lexer;
use tenth::parser::parser::Parser;
use tenth::runtime::interpreter::Interpreter;
use tenth::runtime::natives::register_all_natives;
use tenth::runtime::value::Value;
use tenth::runtime::vm::Vm;

/// 通过解释器执行 .th 源码，返回结果。
fn run_interp(src: &str) -> Result<Option<Value>, String> {
    let mut lexer = Lexer::new(src);
    let tokens = lexer.tokenize().map_err(|e| e.to_string())?;
    let mut parser = Parser::new(tokens);
    let program = parser.parse_program().map_err(|e| e.to_string())?;
    let mut lowerer = Lowerer::new();
    let hir = lowerer.lower_program(&program).map_err(|e| e.to_string())?;
    let mut interpreter = Interpreter::new(&hir);
    interpreter.execute_program(&hir).map_err(|e| e.to_string())
}

/// 编译 HIR 到 VM（公共部分）。
fn compile_to_vm(hir: &tenth::hir::hir::HirProgram, vm: &mut Vm) -> Result<(), String> {
    register_all_natives(vm);
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
    }
    Ok(())
}

/// 通过字节码 VM 执行 .th 源码，返回结果。
fn run_vm(src: &str) -> Result<Value, String> {
    let mut lexer = Lexer::new(src);
    let tokens = lexer.tokenize().map_err(|e| e.to_string())?;
    let mut parser = Parser::new(tokens);
    let program = parser.parse_program().map_err(|e| e.to_string())?;
    let mut lowerer = Lowerer::new();
    let hir = lowerer.lower_program(&program).map_err(|e| e.to_string())?;
    let mut vm = Vm::new();
    compile_to_vm(&hir, &mut vm)?;
    vm.call("main").map_err(|e| e.to_string())
}

/// 通过 JIT 执行 .th 源码（CLI 默认路径 `run_file` 同款入口）。
fn run_jit(src: &str) -> Result<Value, String> {
    let mut lexer = Lexer::new(src);
    let tokens = lexer.tokenize().map_err(|e| e.to_string())?;
    let mut parser = Parser::new(tokens);
    let program = parser.parse_program().map_err(|e| e.to_string())?;
    let mut lowerer = Lowerer::new();
    let hir = lowerer.lower_program(&program).map_err(|e| e.to_string())?;
    let mut vm = Vm::new();
    compile_to_vm(&hir, &mut vm)?;
    jit::run_jit(&mut vm, "main").map_err(|e| e.to_string())
}

/// 断言三路径对同一源码产出相同 i64 值。
fn assert_i64_consistent(src: &str, expected: i64, ctx: &str) {
    let interp = run_interp(src).unwrap().expect("interp result");
    let vm_val = run_vm(src).unwrap();
    let jit_val = run_jit(src).unwrap();
    for (name, v) in [("interp", &interp), ("vm", &vm_val), ("jit", &jit_val)] {
        match v {
            Value::Int(n, _) => assert_eq!(*n, expected, "{ctx}: {name} 期望 {expected}，实际 {n}"),
            other => panic!("{ctx}: {name} 期望 Int({expected})，实际 {:?}", other),
        }
    }
}

/// 断言三路径对同一源码产出相同字符串值。
fn assert_string_consistent(src: &str, expected: &str, ctx: &str) {
    let interp = run_interp(src).unwrap().expect("interp result");
    let vm_val = run_vm(src).unwrap();
    let jit_val = run_jit(src).unwrap();
    for (name, v) in [("interp", &interp), ("vm", &vm_val), ("jit", &jit_val)] {
        match v {
            Value::String(s) => assert_eq!(s, expected, "{ctx}: {name} 期望 {:?}，实际 {s:?}", expected),
            other => panic!("{ctx}: {name} 期望 String({expected:?})，实际 {:?}", other),
        }
    }
}

// ══════════════════════════════════════════════════════════════════════
// or_die：Tenth 源码构造的 Result::Ok / Option::Some → 取内部值（原始复现）
// ══════════════════════════════════════════════════════════════════════

#[test]
fn test_or_die_result_ok_int_three_paths() {
    // 原始复现：or_die(Result::Ok(42), "no") 在 JIT 下曾返回 "_0"
    let src = r#"
        fn main() -> i64 {
            let x = or_die(Result::Ok(42), "no");
            x
        }
    "#;
    assert_i64_consistent(src, 42, "or_die(Result::Ok(42))");
}

#[test]
fn test_or_die_option_some_int_three_paths() {
    let src = r#"
        fn main() -> i64 {
            let x = or_die(Option::Some(7), "no");
            x
        }
    "#;
    assert_i64_consistent(src, 7, "or_die(Option::Some(7))");
}

#[test]
fn test_or_die_result_ok_string_three_paths() {
    let src = r#"
        fn main() -> str {
            let s = or_die(Result::Ok("hello"), "no");
            s
        }
    "#;
    assert_string_consistent(src, "hello", "or_die(Result::Ok(str))");
}

// ══════════════════════════════════════════════════════════════════════
// assume_ok：Tenth 源码构造的枚举 → 取第一个字段值
// ══════════════════════════════════════════════════════════════════════

#[test]
fn test_assume_ok_result_ok_three_paths() {
    let src = r#"
        fn main() -> i64 {
            let x = assume_ok(Result::Ok(5));
            x
        }
    "#;
    assert_i64_consistent(src, 5, "assume_ok(Result::Ok(5))");
}

#[test]
fn test_assume_ok_option_some_three_paths() {
    let src = r#"
        fn main() -> i64 {
            let x = assume_ok(Option::Some(3));
            x
        }
    "#;
    assert_i64_consistent(src, 3, "assume_ok(Option::Some(3))");
}

// ══════════════════════════════════════════════════════════════════════
// 多字段枚举：match 取内部值（按字段名绑定，验证字段名/值配对正确）
// ══════════════════════════════════════════════════════════════════════

#[test]
fn test_multi_field_enum_match_three_paths() {
    let src = r#"
        enum Pair { AB(i64, i64), None }
        fn main() -> i64 {
            let p = Pair::AB(1, 2);
            match p {
                Pair::AB(a, b) => a + b,
                Pair::None => 0,
            }
        }
    "#;
    assert_i64_consistent(src, 3, "multi-field enum match a+b");
}

#[test]
fn test_multi_field_enum_or_die_first_field_three_paths() {
    // assume_ok 对任意枚举取 .first()——验证字段序为源码序（f0 在首）。
    // 修复前 VM/JIT 字段序为反源码序，此值会错（返回 2 而非 1）。
    let src = r#"
        enum Pair { AB(i64, i64), None }
        fn main() -> i64 {
            let p = Pair::AB(1, 2);
            let first = assume_ok(p);
            first
        }
    "#;
    assert_i64_consistent(src, 1, "assume_ok(Pair::AB(1,2)) 取源码首个字段");
}

// ══════════════════════════════════════════════════════════════════════
// 多字段枚举：直接返回枚举，检查 fields 向量序 = 源码声明序（三路径一致）
// ══════════════════════════════════════════════════════════════════════

#[test]
fn test_multi_field_enum_fields_order_three_paths() {
    let src = r#"
        enum Pair { AB(i64, i64), None }
        fn main() -> Pair {
            Pair::AB(1, 2)
        }
    "#;
    let interp = run_interp(src).unwrap().expect("interp result");
    let vm_val = run_vm(src).unwrap();
    let jit_val = run_jit(src).unwrap();
    for (name, v) in [("interp", &interp), ("vm", &vm_val), ("jit", &jit_val)] {
        match v {
            Value::Enum { enum_name, variant, fields } => {
                assert_eq!(enum_name, "Pair", "{name}: enum_name");
                assert_eq!(variant, "AB", "{name}: variant");
                let fields = fields.borrow();
                let names: Vec<&str> = fields.iter().map(|(n, _)| n.as_str()).collect();
                assert_eq!(names, vec!["_0", "_1"], "{name}: 字段名序应为源码序 _0,_1，实际 {names:?}");
                let vals: Vec<i64> = fields.iter()
                    .map(|(_, v)| match v { Value::Int(n, _) => *n, other => panic!("{name}: 字段值非 Int: {:?}", other) })
                    .collect();
                assert_eq!(vals, vec![1, 2], "{name}: 字段值序应为 1,2，实际 {vals:?}");
            }
            other => panic!("{name}: 期望 Enum，实际 {:?}", other),
        }
    }
}

// ══════════════════════════════════════════════════════════════════════
// 对照：native 构造的 Result（env_get 风格）JIT 下原本就正常，仍应保持
// ══════════════════════════════════════════════════════════════════════

#[test]
fn test_native_constructed_result_still_works_vm_jit() {
    // 用既有 native env_get（构造 Result::Ok(str)）验证 VM/JIT 两条路径
    // or_die 均正常——native 构造的 Result 不经 bytecode MakeEnum，
    // 在 JIT 下原本就正常（边界不回归）。
    // Rust 2024 edition：set_var 为 unsafe
    unsafe { std::env::set_var("TENTH_JIT_ENUM_TEST_VAR", "99") };
    let src = r#"
        fn main() -> str {
            let r = env_get("TENTH_JIT_ENUM_TEST_VAR");
            let v = or_die(r, "no");
            v
        }
    "#;
    for (name, use_jit) in [("vm", false), ("jit", true)] {
        let mut lexer = Lexer::new(src);
        let tokens = lexer.tokenize().unwrap();
        let mut parser = Parser::new(tokens);
        let program = parser.parse_program().unwrap();
        let mut lowerer = Lowerer::new();
        let hir = lowerer.lower_program(&program).unwrap();
        let mut vm = Vm::new();
        compile_to_vm(&hir, &mut vm).unwrap();
        let res = if use_jit {
            jit::run_jit(&mut vm, "main").unwrap()
        } else {
            vm.call("main").unwrap()
        };
        match res { Value::String(s) => assert_eq!(s, "99", "{name}: 期望 \"99\"，实际 {s:?}"), other => panic!("{name}: 期望 String(\"99\")，实际 {:?}", other) }
    }
}

// ══════════════════════════════════════════════════════════════════════
// 错误路径：or_die(Result::Err) 在 JIT 下应同样 panic（不回归）
// ══════════════════════════════════════════════════════════════════════

#[test]
fn test_or_die_err_panics_jit() {
    let src = r#"
        fn main() -> i64 {
            let x = or_die(Result::Err("boom"), "custom-msg");
            x
        }
    "#;
    let err = run_jit(src).unwrap_err();
    assert!(
        err.contains("custom-msg") || err.contains("boom"),
        "JIT or_die(Err) 应报错，实际错误: {err}"
    );
}
