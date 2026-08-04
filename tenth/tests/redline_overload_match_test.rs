//! 1.0 前红线修复轮测试：AUDIT-11.4.39（重载运行时分派三路径一致）+ AUDIT-11.4.34（VM match tuple 模式 + guard 回退）。
//!
//! AUDIT-11.4.39：`fn g(x: i64, y: i64)` + `fn g(x: str)` 这类同文件重载，编译期
//! `resolve_fn_overload` 按实参类型选中唯一签名，但运行时分派只按函数名——VM
//! `HashMap<String,usize>` 后注册覆盖、解释器取第一条同名，两路径可能选中不同
//! 签名（静默错值红线）。修复：lowering 完成后编译期确定性 mangling（定义改名
//! `__ovl_<name>_<idx>` + 调用点/函数值引用改写），三后端按 mangled 名解析天然一致。
//! 本套件对拍 解释器 = VM = JIT 三路径。
//!
//! AUDIT-11.4.34：VM `match` 的 tuple 模式 + guard——guard 失败后不试下一条
//! tuple 臂直接落 wildcard（解释器正确试下一条）。修复：有 guard 时保留一份
//! scrutinee 副本供下一条臂重试（对齐 EnumVariant/Struct 臂模式）。
//! 本套件把 AUDIT 登记的反例固化为正例。

use tenth::compile::bytecode::BytecodeCompiler;
use tenth::compile::jit;
use tenth::error::TenthError;
use tenth::hir::lower::Lowerer;
use tenth::lexer::lexer::Lexer;
use tenth::parser::parser::Parser;
use tenth::runtime::interpreter::Interpreter;
use tenth::runtime::natives::register_all_natives;
use tenth::runtime::value::Value;
use tenth::runtime::vm::Vm;

/// lex → parse → lower（共享）。
fn lower(src: &str) -> Result<tenth::hir::hir::HirProgram, String> {
    let mut lexer = Lexer::new(src);
    let tokens = lexer.tokenize().map_err(|e| e.to_string())?;
    let mut parser = Parser::new(tokens);
    let program = parser.parse_program().map_err(|e| e.to_string())?;
    let mut lowerer = Lowerer::new();
    lowerer.lower_program(&program).map_err(|e| e.to_string())
}

/// 解释器路径。
fn run_interp(src: &str) -> Result<Value, TenthError> {
    let hir = lower(src).map_err(|e| TenthError::RuntimeError { line: None, col: None, message: e })?;
    let mut interpreter = Interpreter::new(&hir);
    interpreter
        .execute_program(&hir)
        .map(|v| v.unwrap_or(Value::Unit))
}

/// 编译源码到 VM（含全部 natives + 函数全局 FnRef 注册，与 main.rs vm_execute 对齐）。
fn compile_vm(src: &str) -> Result<Vm, String> {
    let hir = lower(src)?;
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
    }
    Ok(vm)
}

/// 纯 VM 路径（不经 JIT）。
fn run_vm(src: &str) -> Result<Value, TenthError> {
    let mut vm = compile_vm(src).map_err(|e| TenthError::RuntimeError { line: None, col: None, message: e })?;
    if vm.has_fn("main") {
        vm.call("main")
    } else {
        Ok(Value::Unit)
    }
}

/// JIT 路径（`jit::run_jit`，内部失败自动回退 VM）。
fn run_jit(src: &str) -> Result<Value, TenthError> {
    let mut vm = compile_vm(src).map_err(|e| TenthError::RuntimeError { line: None, col: None, message: e })?;
    if vm.has_fn("main") {
        jit::run_jit(&mut vm, "main")
    } else {
        Ok(Value::Unit)
    }
}

fn int_of(v: &Value, label: &str) -> i64 {
    match v {
        Value::Int(n, _) => *n,
        other => panic!("[{label}] 期望 Int，实际 {:?}", other),
    }
}

/// 三路径对拍：解释器 / VM / JIT 均等于 expected Int 且互等。
fn assert_three_int(src: &str, expected: i64, label: &str) {
    let i = run_interp(src).unwrap_or_else(|e| panic!("[{label}] 解释器执行失败: {}", e));
    let v = run_vm(src).unwrap_or_else(|e| panic!("[{label}] VM 执行失败: {}", e));
    let j = run_jit(src).unwrap_or_else(|e| panic!("[{label}] JIT 执行失败: {}", e));
    let i_n = int_of(&i, label);
    let v_n = int_of(&v, label);
    let j_n = int_of(&j, label);
    assert_eq!(v_n, i_n, "[{label}] VM/解释器不一致: VM={} 解释器={}", v_n, i_n);
    assert_eq!(j_n, i_n, "[{label}] JIT/解释器不一致: JIT={} 解释器={}", j_n, i_n);
    assert_eq!(i_n, expected, "[{label}] 结果错误: {} != {}", i_n, expected);
}

// ════════════════════════════════════════════════════════════════════
// AUDIT-11.4.39：重载函数运行时分派三路径一致
// ════════════════════════════════════════════════════════════════════

/// 审计原始反例：`fn g(x: i64, y: i64) -> i64` + `fn g(x: str) -> str`，
/// `g(1, 2)` / `g("hi")`（类型正确但签名不同的重载调用）三路径一致。
const OVL_ARITY_DIFF_SRC: &str = r#"
fn g(x: i64, y: i64) -> i64 { x * 100 + y }
fn g(x: str) -> str { x + "_STR" }
let a = g(1, 2);              // 应命中 g(i64,i64) -> 102
let b = if g("hi") == "hi_STR" { 1 } else { 0 };  // 应命中 g(str)
let c = g(1, 2) + g(3, 4);    // 102 + 304 = 406
if a != 102 { 10 } else if b != 1 { 20 } else if c != 406 { 30 } else { 0 }
"#;

#[test]
fn ovl_arity_diff_dispatch_three_paths() {
    // 修复前：VM 后注册覆盖 → `g(1,2)` 可能命中 g(str) chunk 静默错值。
    assert_three_int(OVL_ARITY_DIFF_SRC, 0, "ovl_arity_diff");
}

/// 同参数数量、不同类型签名（更严苛：arity 相同但类型不同）。
/// 注：同参数量重载调用必须用显式类型字面量（`5i64`）才能精确匹配签名——
/// I32 字面量不精确匹配任一签名时编译期报「调用歧义」（既有正确行为，
/// 优于静默错值）。
const OVL_SAME_ARITY_DIFF_TYPE_SRC: &str = r#"
fn h(x: i64) -> i64 { x * 10 }
fn h(x: str) -> str { x + "!" }
let a = h(5i64);              // 应命中 h(i64) -> 50
let b = if h("hi") == "hi!" { 1 } else { 0 };  // 应命中 h(str)
if a != 50 { 11 } else if b != 1 { 21 } else { 0 }
"#;

#[test]
fn ovl_same_arity_diff_type_three_paths() {
    assert_three_int(OVL_SAME_ARITY_DIFF_TYPE_SRC, 0, "ovl_same_arity_diff_type");
}

/// 递归经过重载：g2 体内调用 g(i64,i64)。
const OVL_RECURSION_SRC: &str = r#"
fn g(x: i64, y: i64) -> i64 { x * 100 + y }
fn g(x: str) -> str { x + "_STR" }
fn g2(n: i64) -> i64 {
    if n <= 0 { return 0; }
    return g(n, n) + g2(n - 1);
}
g2(3)  // g(3,3)+g(2,2)+g(1,1) = 303+202+101 = 606
"#;

#[test]
fn ovl_recursion_through_overload_three_paths() {
    assert_three_int(OVL_RECURSION_SRC, 606, "ovl_recursion");
}

/// 函数值引用：`let f = g; f(7,8)` 取首个签名（与类型检查 lookup_fn 语义一致）。
const OVL_FN_VALUE_REF_SRC: &str = r#"
fn g(x: i64, y: i64) -> i64 { x * 100 + y }
fn g(x: str) -> str { x + "_STR" }
let f = g;
let e = f(7, 8);  // 应命中首个签名 g(i64,i64) -> 708
if e != 708 { 12 } else { 0 }
"#;

#[test]
fn ovl_fn_value_ref_three_paths() {
    assert_three_int(OVL_FN_VALUE_REF_SRC, 0, "ovl_fn_value_ref");
}

/// 三种重载混合 + 与普通函数协作 + 顺序无关（str 签名先声明也正确）。
const OVL_STR_FIRST_SRC: &str = r#"
fn m(x: str) -> str { "S:" + x }
fn m(x: i64) -> i64 { x + 1000 }
fn plain(x: i64) -> i64 { x + 1 }
let a = m("a");               // S:a
let b = m(5i64);              // 1005
let c = plain(b);             // 1006
if a != "S:a" { 13 } else if b != 1005 { 23 } else if c != 1006 { 33 } else { 0 }
"#;

#[test]
fn ovl_str_declared_first_three_paths() {
    // 修复前：解释器取第一条同名（str），`m(5)` 命中 m(str) 静默错值。
    assert_three_int(OVL_STR_FIRST_SRC, 0, "ovl_str_first");
}

/// 三路返回类型不同的重载（多签名 >2）。
const OVL_TRIPLE_SRC: &str = r#"
fn t(x: i64) -> i64 { x + 1 }
fn t(x: f64) -> f64 { x + 0.5 }
fn t(x: str) -> str { x + "!" }
let a = t(1i64);              // 2
let b = if t("hi") == "hi!" { 1 } else { 0 };
if a != 2 { 14 } else if b != 1 { 24 } else { 0 }
"#;

#[test]
fn ovl_triple_overload_three_paths() {
    assert_three_int(OVL_TRIPLE_SRC, 0, "ovl_triple");
}

/// 白盒守护：重载签名在 VM 中注册为独立 chunk（mangled 名），单签名函数名不变。
#[test]
fn ovl_mangled_chunks_registered() {
    let src = r#"
fn g(x: i64, y: i64) -> i64 { x * 100 + y }
fn g(x: str) -> str { x + "_STR" }
fn plain(x: i64) -> i64 { x + 1 }
plain(1)
"#;
    let vm = compile_vm(src).expect("编译应成功");
    // g(i64,i64) 与 g(str) 各自独立 chunk（AUDIT-11.4.39 修复核心）。
    assert!(vm.has_fn("__ovl_g_0"), "应注册 __ovl_g_0 (g(i64,i64))");
    assert!(vm.has_fn("__ovl_g_1"), "应注册 __ovl_g_1 (g(str))");
    // 单签名函数名保持原名（不 mangling，无行为变化）。
    assert!(vm.has_fn("plain"), "单签名函数名应保持 'plain'");
    // 原名字不再作为 chunk（杜绝运行时按原名选错）。
    assert!(!vm.has_fn("g"), "重载原名 'g' 不应再作为 chunk 存在");
}

/// 不回归：既有单签名重载调用语义不变（call_arg_type_check_test 的 overload 用例）。
#[test]
fn ovl_single_signature_unchanged() {
    let src = r#"
fn f(x: i64) -> i64 { x }
fn g(x: str) -> str { "s" }
f(5i64) + 1
"#;
    assert_three_int(src, 6, "ovl_single_signature");
}

// ════════════════════════════════════════════════════════════════════
// AUDIT-11.4.34：VM match tuple 模式 + guard 回退（反例固化为正例）
// ════════════════════════════════════════════════════════════════════

/// 审计原始反例：guard 失败后应尝试下一条 tuple 臂（而非落 wildcard）。
/// `(2, 3)`（guard 2>3=false）→ 修复前 VM 落 `_ => -1`，解释器正确返回 23。
const MGUARD_NEXT_TUPLE_SRC: &str = r#"
fn f1(t: (i64, i64)) -> i64 {
    match t {
        (a, b) if a > b => a * 100 + b,
        (a, b) => a * 10 + b,
        _ => -1,
    }
}
let r1 = f1((2, 3));   // guard false -> 下一条 tuple 臂 -> 2*10+3 = 23
let r2 = f1((5, 1));   // guard true  -> 第一条 -> 501
if r1 != 23 { 15 } else if r2 != 501 { 25 } else { 0 }
"#;

#[test]
fn mguard_fail_tries_next_tuple_arm_three_paths() {
    assert_three_int(MGUARD_NEXT_TUPLE_SRC, 0, "mguard_next_tuple");
}

/// 多 guard：第一条 guard 失败、第二条通过。
const MGUARD_MULTI_SRC: &str = r#"
fn f2(t: (i64, i64)) -> i64 {
    match t {
        (a, b) if a > 100 => 1000 + a,
        (a, b) if a < 0 => 2000 + b,
        (a, b) => a + b,
        _ => -1,
    }
}
let r1 = f2((150, 2));  // 第一条 guard 通过 -> 1150
let r2 = f2((-5, 7));   // 第一条 false、第二条通过 -> 2007
let r3 = f2((4, 6));    // 都 false -> 第三条 -> 10
if r1 != 1150 { 16 } else if r2 != 2007 { 26 } else if r3 != 10 { 36 } else { 0 }
"#;

#[test]
fn mguard_multi_guard_three_paths() {
    assert_three_int(MGUARD_MULTI_SRC, 0, "mguard_multi");
}

/// guard + wildcard 组合：guard 失败 → 落 wildcard。
const MGUARD_WILDCARD_SRC: &str = r#"
fn f3(t: (i64, i64)) -> i64 {
    match t {
        (a, b) if a > b => a - b,
        _ => -2,
    }
}
let r1 = f3((2, 5));   // guard false -> wildcard -> -2
let r2 = f3((5, 2));   // guard true  -> 3
if r1 != -2 { 17 } else if r2 != 3 { 27 } else { 0 }
"#;

#[test]
fn mguard_wildcard_fallback_three_paths() {
    assert_three_int(MGUARD_WILDCARD_SRC, 0, "mguard_wildcard");
}

/// 非 tuple 字面量臂在前 + tuple guard 臂 + wildcard（多形态组合）。
const MGUARD_MIXED_SRC: &str = r#"
fn f4(t: (i64, i64)) -> i64 {
    match t {
        (a, b) if a == b => a,
        (a, b) if a == 0 => 100 + b,
        (a, b) => a + b,
        _ => -3,
    }
}
let r1 = f4((2, 2));   // 第一条 guard 通过 -> 2
let r2 = f4((0, 5));   // 第一条 false、第二条通过 -> 105
let r3 = f4((3, 4));   // 都 false -> 第三条 -> 7
if r1 != 2 { 18 } else if r2 != 105 { 28 } else if r3 != 7 { 38 } else { 0 }
"#;

#[test]
fn mguard_mixed_arms_three_paths() {
    assert_three_int(MGUARD_MIXED_SRC, 0, "mguard_mixed");
}

/// 循环内 match guard（VM/JIT 热路径稳定性）。
const MGUARD_IN_LOOP_SRC: &str = r#"
fn f5(t: (i64, i64)) -> i64 {
    match t {
        (a, b) if a > b => a - b,
        (a, b) => b - a,
        _ => -1,
    }
}
fn main() {
    let mut acc = 0;
    let mut i = 0;
    while i < 10 {
        acc = acc + f5((i, i + 1));  // 每轮 guard false -> 下一条 tuple 臂 -> 1
        i = i + 1;
    }
    acc  // 10 * 1 = 10
}
"#;

#[test]
fn mguard_in_loop_three_paths() {
    assert_three_int(MGUARD_IN_LOOP_SRC, 10, "mguard_in_loop");
}

/// guard 引用的绑定变量在失败后不泄漏到下一条臂（绑定子作用域 AUDIT #18 族）。
const MGUARD_BINDING_ISOLATION_SRC: &str = r#"
fn f6(t: (i64, i64)) -> i64 {
    let outer = 99;
    match t {
        (a, b) if a > 100 => { let z = a + b; z },
        (a, b) => a * 10 + b,
        _ => -1,
    }
}
let r1 = f6((2, 3));  // 第一条 guard false -> 第二条 -> 23
let r2 = f6((200, 1));// 第一条 guard true -> 201
if r1 != 23 { 19 } else if r2 != 201 { 29 } else { 0 }
"#;

#[test]
fn mguard_binding_isolation_three_paths() {
    assert_three_int(MGUARD_BINDING_ISOLATION_SRC, 0, "mguard_binding_isolation");
}
