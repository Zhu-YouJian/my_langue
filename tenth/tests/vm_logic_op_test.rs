// 逻辑运算符 + 变量 shadowing 的 VM 字节码回归测试。
//
// 覆盖 2026-08-02 修复的两个 VM 字节码编译器 bug：
//   1. `Binary` 编译对 And/Or 无条件先编译右操作数，导致短路分支再次编译右操作数
//      （双重求值 + 栈污染）：`true || false` 错误返回 false、`false && true` 错误返回 true。
//      → 修复：And/Or 跳过急切右操作数编译，由短路分支编译。
//   2. 局部变量名→槽位用 `position`（首个匹配），同名 let/循环变量重绑定后读旧槽位：
//      `let x = n; let x = x + 10;` 读回首个 x；`for j` 重绑定后索引读旧 j。
//      → 修复：改用 `rposition`（最近绑定），与解释器/WASM 语义对齐。
use tenth::compile::bytecode::BytecodeCompiler;
use tenth::hir::lower::Lowerer;
use tenth::lexer::lexer::Lexer;
use tenth::parser::parser::Parser;
use tenth::runtime::natives::register_all_natives;
use tenth::runtime::value::Value;
use tenth::runtime::vm::Vm;

/// 纯 VM（无 JIT）执行 .th 源码，返回 main 结果。
fn run_plain_vm(src: &str) -> Result<Value, String> {
    let tokens = Lexer::new(src).tokenize().map_err(|e| e.to_string())?;
    let program = Parser::new(tokens).parse_program().map_err(|e| e.to_string())?;
    let mut lowerer = Lowerer::new();
    let hir = lowerer.lower_program(&program).map_err(|e| e.to_string())?;
    let mut vm = Vm::new();
    register_all_natives(&mut vm);
    for func in &hir.functions {
        let compiler = BytecodeCompiler::new();
        if let Ok((chunk, closures)) = compiler.compile(func) {
            vm.add_fn(func.name.clone(), chunk);
            for (name, closure_chunk) in closures {
                vm.add_fn(name, closure_chunk);
            }
        }
    }
    if let Some(ref expr) = hir.main_expr {
        let compiler = BytecodeCompiler::new();
        if let Ok((chunk, closures)) = compiler.compile_main(expr) {
            vm.add_fn("main".into(), chunk);
            for (name, closure_chunk) in closures {
                vm.add_fn(name, closure_chunk);
            }
        }
    }
    vm.call("main").map_err(|e| e.to_string())
}

#[test]
fn vm_logic_or_left_true() {
    // 修复前：`true || false` 在 VM 上错误返回 false（右操作数覆盖栈顶）
    let r = run_plain_vm("true || false").unwrap();
    assert!(matches!(r, Value::Bool(b) if b), "true || false 应为 true，实际 {:?}", r);
}

#[test]
fn vm_logic_or_eq_left_true() {
    // 修复前：`x == 0 || n == 0`（x=0）在 VM 上错误返回 false
    let r = run_plain_vm("let x = 0; let n = 3; x == 0 || n == 0").unwrap();
    assert!(matches!(r, Value::Bool(b) if b), "x==0||n==0 应为 true，实际 {:?}", r);
}

#[test]
fn vm_logic_and_right_true() {
    // 修复前：`false && true` 在 VM 上错误返回 true（右操作数覆盖栈顶）
    let r = run_plain_vm("false && true").unwrap();
    assert!(matches!(r, Value::Bool(b) if !b), "false && true 应为 false，实际 {:?}", r);
}

#[test]
fn vm_logic_all_combos() {
    let src = r#"
        fn main() -> i64 {
            let a = if true || false { 1 } else { 0 };
            let b = if false || true { 1 } else { 0 };
            let c = if false && true { 1 } else { 0 };
            let d = if true && false { 1 } else { 0 };
            let e = if true && true { 1 } else { 0 };
            let f = if false || false { 1 } else { 0 };
            a * 100000 + b * 10000 + c * 1000 + d * 100 + e * 10 + f
        }
    "#;
    let r = run_plain_vm(src).unwrap();
    // 期望: a=1 b=1 c=0 d=0 e=1 f=0 → 110010
    assert!(matches!(r, Value::Int(n, _) if n == 110010), "逻辑组合结果应为 110010，实际 {:?}", r);
}

#[test]
fn vm_shadowing_same_scope() {
    // 修复前：`let x = n; let x = x + 10; let x = x * 2;` 读回首个 x → n*2=10（错误）
    // 修复后：应逐次引用最近绑定 → (5+10)*2 = 30（与解释器/WASM 一致）
    let src = r#"
        fn shadow(n: i64) -> i64 { let x = n; let x = x + 10; let x = x * 2; x }
        fn main() -> i64 { shadow(5) }
    "#;
    let r = run_plain_vm(src).unwrap();
    assert!(matches!(r, Value::Int(n, _) if n == 30), "shadow(5) 应为 30，实际 {:?}", r);
}

#[test]
fn vm_loop_var_rebind() {
    // 修复前：两个 `for j` 循环复用同名循环变量，索引读旧槽位 → prev[j] 返回 Unit
    let src = r#"
        fn main() -> i64 {
            let mut prev = Vec::new();
            for j in 0..=3 { prev.push(j); };
            let mut s: i64 = 0;
            for j in 0..=3 { let v = prev[j]; s = s + v; };
            s
        }
    "#;
    let r = run_plain_vm(src).unwrap();
    // prev=[0,1,2,3]，求和 = 6
    assert!(matches!(r, Value::Int(n, _) if n == 6), "循环变量重绑定索引求和应为 6，实际 {:?}", r);
}
