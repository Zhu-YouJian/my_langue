//! M2.6-P1：纯标量函数固定开销消除守护测试（跳过 current_chunk_idx 切换）。
//!
//! P1 优化：特化调用点对「纯标量函数」（`compute_skip_chunk_ctx` 判定：body 仅
//! 纯标量 op + 每个 Call/CallN 站点内联或预测特化）省去 `current_chunk_idx`
//! 保存/切换/恢复 3 次内存操作；`host_check_error` 内联为直接 load
//! `vm.jit_error_flag`。本套件守护：
//! - 跳过判定正确性：fib（递归纯标量）/ even-odd（互递归纯标量）/ add3（多参，
//!   内联嵌套调用）→ skip=true 且结果正确
//! - 错误路径（跳过切换后仍保留 B2 红线）：除零 / 溢出带正确行号，不静默
//! - 不跳过场景：含 PushStr / MethodCall / TailCall（尾递归优化）的特化函数 →
//!   skip=false（保持切换），字符串表解析仍正确（证明「需要切换才不跳过」的判定生效）
//! - 全程 VM=JIT 对拍一致
//!
//! 注意：签名从 `Chunk.scalar_sig`（BytecodeCompiler 编译时从 HIR 推导）读取，
//! 与 main.rs 同构（compile → add_fn），特化自动生效。仅显式 `i64` 注解函数
//! 有 scalar_sig（`Int` 别名不纳入）。

use tenth::compile::bytecode::BytecodeCompiler;
use tenth::compile::jit;
use tenth::error::TenthError;
use tenth::hir::lower::Lowerer;
use tenth::lexer::lexer::Lexer;
use tenth::parser::parser::Parser;
use tenth::runtime::natives::register_all_natives;
use tenth::runtime::value::Value;
use tenth::runtime::vm::Vm;

/// 编译源码到 VM（含全部 natives；BytecodeCompiler 自动推导 scalar_sig）。
fn compile_vm(src: &str) -> Result<Vm, String> {
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
    }
    Ok(vm)
}

/// 纯 VM 路径执行 main（`vm.call`，不经 JIT——对拍基准）。
fn run_vm(src: &str) -> Result<Value, TenthError> {
    let mut vm = compile_vm(src)
        .map_err(|e| TenthError::RuntimeError { line: None, col: None, message: e })?;
    if vm.has_fn("main") {
        vm.call("main")
    } else {
        Ok(Value::Unit)
    }
}

/// JIT 路径执行 main（`jit::run_jit`，内部失败自动回退 VM）。
fn run_jit(src: &str) -> Result<Value, TenthError> {
    let mut vm = compile_vm(src)
        .map_err(|e| TenthError::RuntimeError { line: None, col: None, message: e })?;
    if vm.has_fn("main") {
        jit::run_jit(&mut vm, "main")
    } else {
        Ok(Value::Unit)
    }
}

/// JIT 执行并保留 Vm（供 skip_chunk_ctx_for 覆盖断言）。
fn run_jit_with_vm(src: &str) -> Result<(Value, Vm), String> {
    let mut vm = compile_vm(src)?;
    if vm.has_fn("main") {
        let r = jit::run_jit(&mut vm, "main").map_err(|e| e.to_string())?;
        Ok((r, vm))
    } else {
        Ok((Value::Unit, vm))
    }
}

/// JIT 执行，返回原始错误（供错误路径行号断言）。
fn run_jit_err(src: &str) -> Result<Value, TenthError> {
    let mut vm = compile_vm(src)
        .map_err(|e| TenthError::RuntimeError { line: None, col: None, message: e })?;
    if vm.has_fn("main") {
        jit::run_jit(&mut vm, "main")
    } else {
        Ok(Value::Unit)
    }
}

fn int_of(v: Value, label: &str) -> i64 {
    match v {
        Value::Int(n, _) => n,
        other => panic!("[{label}] 期望 Int，实际 {:?}", other),
    }
}

/// chunk 名 → chunk 索引。
fn chunk_idx(vm: &Vm, name: &str) -> usize {
    vm.chunk_index_of(name).unwrap_or_else(|| panic!("chunk {name} 未注册"))
}

/// chunk 的跳过判定（P1：skip=true = 特化调用点跳过 current_chunk_idx 切换）。
fn skip_ctx(vm: &Vm, name: &str) -> bool {
    let ctx = vm.jit_ctx.as_ref().expect("JIT 上下文应存在");
    ctx.skip_chunk_ctx_for(chunk_idx(vm, name))
}

/// 从错误中提取 RuntimeError 的 line 字段；非 RuntimeError 直接 panic。
fn runtime_line(err: &TenthError) -> Option<usize> {
    match err {
        TenthError::RuntimeError { line, .. } => *line,
        other => panic!("期望 RuntimeError，实际: {:?}", other),
    }
}

/// JIT 与 VM 均产出等于 `expected` 的 Int，且两侧互等。
fn assert_vm_jit_int(src: &str, expected: i64, label: &str) {
    let vm_res = run_vm(src).unwrap_or_else(|e| panic!("[{label}] VM 执行失败: {}", e));
    let jit_res = run_jit(src).unwrap_or_else(|e| panic!("[{label}] JIT 执行失败: {}", e));
    let vm_i = int_of(vm_res, label);
    let jit_i = int_of(jit_res, label);
    assert_eq!(vm_i, expected, "[{label}] VM 结果错误");
    assert_eq!(jit_i, expected, "[{label}] JIT 结果错误");
    assert_eq!(vm_i, jit_i, "[{label}] VM/JIT 不一致");
}

// ── 1. 跳过判定：递归纯标量函数（fib）────────────────────────────────────

/// fib（递归 + i64 注解）→ 判定为可跳过切换（skip=true）；结果正确。
#[test]
fn skip_eligible_fib_recursive() {
    let src = r#"
        fn fib(n: i64) -> i64 {
            if n < 2 { n } else { fib(n - 1) + fib(n - 2) }
        }
        fn main() -> i64 { fib(20) }
    "#;
    let (v, vm) = run_jit_with_vm(src).unwrap();
    assert_eq!(int_of(v, "fib"), 6765);
    assert!(skip_ctx(&vm, "fib"), "fib 应为纯标量、可跳过 chunk 切换（skip=true）");
}

#[test]
fn skip_eligible_fib_parity() {
    let src = r#"
        fn fib(n: i64) -> i64 {
            if n < 2 { n } else { fib(n - 1) + fib(n - 2) }
        }
        fn main() -> i64 { fib(28) }
    "#;
    assert_vm_jit_int(src, 317811, "fib28-parity");
}

// ── 2. 跳过判定：互递归纯标量（even/odd，嵌套 spec 调用）──────────────────

/// even/odd 互递归（非尾位置 → CallN，嵌套 spec 调用链）→ 双 skip=true。
#[test]
fn skip_eligible_even_odd_nested() {
    let src = r#"
        fn even(n: i64) -> i64 {
            if n == 0 { 1 } else { odd(n - 1) + 0 }
        }
        fn odd(n: i64) -> i64 {
            if n == 0 { 0 } else { even(n - 1) + 0 }
        }
        fn main() -> i64 { even(20) }
    "#;
    let (v, vm) = run_jit_with_vm(src).unwrap();
    assert_eq!(int_of(v, "even"), 1);
    assert!(skip_ctx(&vm, "even"), "even 应为纯标量（skip=true）");
    assert!(skip_ctx(&vm, "odd"), "odd 应为纯标量（skip=true）");
    let vm_res = run_vm(src).unwrap();
    assert_eq!(int_of(vm_res, "even"), 1);
}

// ── 3. 跳过判定：多参纯标量（add3，内联嵌套调用）────────────────────────

/// add3（3 个 i64 参数 + 内联嵌套调用 add2）→ 可跳过（skip=true）。
#[test]
fn skip_eligible_add3_multiparam() {
    let src = r#"
        fn add2(a: i64, b: i64) -> i64 { a + b }
        fn add3(a: i64, b: i64, c: i64) -> i64 { add2(a, b) + c }
        fn main() -> i64 { add3(10, 20, 30) }
    "#;
    let (v, vm) = run_jit_with_vm(src).unwrap();
    assert_eq!(int_of(v, "add3"), 60);
    assert!(skip_ctx(&vm, "add3"), "add3 应为纯标量、可跳过 chunk 切换（skip=true）");
    let vm_res = run_vm(src).unwrap();
    assert_eq!(int_of(vm_res, "add3"), 60);
}

// ── 4. 错误路径：跳过切换后仍保留 B2 红线 + 行号 ────────────────────────

/// 除零发生在「跳过切换」的纯标量特化函数体内 → 错误不静默、行号正确。
/// （成功路径验证 skip=true；错误路径验证行号——raw string 首行换行，`n / d` 在第 3 行）
#[test]
fn skip_error_div_zero_keeps_line() {
    let src_ok = r#"
        fn divrec(n: i64, d: i64) -> i64 {
            if n < 2 { n / d } else { divrec(n - 1, d) + 1 }
        }
        fn main() -> i64 { divrec(3, 2) }
    "#;
    let (v, vm) = run_jit_with_vm(src_ok).unwrap();
    assert_eq!(int_of(v, "divrec"), 2, "divrec(3,2) = 2（1/2=0 整数除法）");
    assert!(skip_ctx(&vm, "divrec"), "divrec 应为纯标量（skip=true）");
    let src_err = r#"
        fn divrec(n: i64, d: i64) -> i64 {
            if n < 2 { n / d } else { divrec(n - 1, d) + 1 }
        }
        fn main() -> i64 { divrec(3, 0) }
    "#;
    let err = run_jit_err(src_err).unwrap_err();
    assert_eq!(runtime_line(&err), Some(3), "跳过切换后除零行号应为第 3 行，实际 {:?}", err);
    let msg = format!("{}", err);
    assert!(msg.contains("整数除零"), "应报整数除零，实际: {msg}");
}

/// 溢出发生在「跳过切换」的纯标量特化函数体内 → 错误不静默、行号正确。
#[test]
fn skip_error_overflow_keeps_line() {
    let src_ok = r#"
        fn ovf(n: i64) -> i64 {
            if n < 2 { n + 2147483646 + 1 } else { ovf(n - 1) + 1 }
        }
        fn main() -> i64 { ovf(0) }
    "#;
    let (v, vm) = run_jit_with_vm(src_ok).unwrap();
    assert_eq!(int_of(v, "ovf"), 2147483647, "ovf(0) = i32::MAX（不溢出）");
    assert!(skip_ctx(&vm, "ovf"), "ovf 应为纯标量（skip=true）");
    let src_err = r#"
        fn ovf(n: i64) -> i64 {
            if n < 2 { n + 2147483646 + 1 } else { ovf(n - 1) + 1 }
        }
        fn main() -> i64 { ovf(1) }
    "#;
    let err = run_jit_err(src_err).unwrap_err();
    assert_eq!(runtime_line(&err), Some(3), "跳过切换后溢出行号应为第 3 行，实际 {:?}", err);
    let msg = format!("{}", err);
    assert!(msg.contains("溢出"), "应报整数溢出，实际: {msg}");
}

// ── 5. 不跳过场景：含字符串 / 方法调用 / 尾递归的特化函数（保持切换）──────

/// 特化函数体内含 PushStr + native 调用（parse_int）→ 需要字符串表 → 不跳过；
/// 字符串仍从被调 chunk 表解析（证明「需要切换才不跳过」判定生效）。
#[test]
fn no_skip_with_pushstr() {
    let src = r#"
        fn add_num(x: i64) -> i64 {
            let s = "10";
            x + parse_int(s)
        }
        fn main() -> i64 { add_num(5) }
    "#;
    let (v, vm) = run_jit_with_vm(src).unwrap();
    assert_eq!(int_of(v, "add_num"), 15, "add_num(5) = 5 + parse_int(\"10\") = 15");
    assert!(!skip_ctx(&vm, "add_num"), "含 PushStr 的特化函数不应跳过 chunk 切换");
    let vm_res = run_vm(src).unwrap();
    assert_eq!(int_of(vm_res, "add_num"), 15);
}

/// 特化函数体内含 MethodCall（str.len()）→ 需要字符串表 → 不跳过；结果正确。
#[test]
fn no_skip_with_methodcall() {
    let src = r#"
        fn mlen(x: i64) -> i64 {
            let s = "abc";
            x + s.len()
        }
        fn main() -> i64 { mlen(5) }
    "#;
    let (v, vm) = run_jit_with_vm(src).unwrap();
    assert_eq!(int_of(v, "mlen"), 8, "mlen(5) = 5 + \"abc\".len() = 8");
    assert!(!skip_ctx(&vm, "mlen"), "含 MethodCall 的特化函数不应跳过 chunk 切换");
    let vm_res = run_vm(src).unwrap();
    assert_eq!(int_of(vm_res, "mlen"), 8);
}

/// 尾递归（TCO → TailCall op，走 host_call 需字符串表）→ 不跳过，结果正确。
#[test]
fn no_skip_with_tailcall() {
    let src = r#"
        fn gcd(a: i64, b: i64) -> i64 {
            if b == 0 { a } else { gcd(b, a % b) }
        }
        fn main() -> i64 { gcd(48, 36) }
    "#;
    let (v, vm) = run_jit_with_vm(src).unwrap();
    assert_eq!(int_of(v, "gcd"), 12, "gcd(48,36) = 12");
    assert!(!skip_ctx(&vm, "gcd"), "尾递归（TailCall）不应跳过 chunk 切换");
    let vm_res = run_vm(src).unwrap();
    assert_eq!(int_of(vm_res, "gcd"), 12);
}

/// 特化函数含字符串字面量但仍经 spec 入口执行 → 字符串解析在「不跳过」的
/// 调用点下正确（跨 JIT 帧字符串表隔离）。
#[test]
fn no_skip_string_parity() {
    let src = r#"
        fn tag(x: i64) -> i64 {
            let t = "T";
            x + t.len()
        }
        fn main() -> i64 { tag(41) + tag(1) }
    "#;
    assert_vm_jit_int(src, 44, "tag-parity");
}
