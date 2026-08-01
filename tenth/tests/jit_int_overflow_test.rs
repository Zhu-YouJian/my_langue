//! AUDIT-11.4.17 回归测试：JIT / VM / 解释器 三路径整数溢出行为一致。
//!
//! 背景：`check_int_overflow()` 运行时溢出检测覆盖 VM/解释器路径，但 `overflow-checks
//! = true`（release）下 i64 层溢出（`i64::MAX + 1`、`i64::MIN / -1`）在算术原语里直接
//! panic；JIT 经 hostcall 调用同一原语，panic 穿越 `extern "C"` 边界导致进程 abort。
//! 修复：算术原语改用 `checked_*`，i64 层溢出与窄 dtype 范围溢出统一转为干净的
//! RuntimeError，三路径行为一致（不再 panic/abort/静默回绕）。
//!
//! 说明：
//! - JIT 整数算术全部经 hostcall（host_add 等）走与 VM 相同的 `add_priv` 等原语，
//!   因此本文件同时覆盖 VM 与 JIT 路径；解释器路径独立断言。
//! - i64 字面量后缀在运行时丢失为 I32（既有 dtype 保留问题，不在本 AUDIT 范围），
//!   故消息中出现 "i32 范围"；i64 层溢出由 `checked_*` 拦截，仍能干净报错。

use tenth::lexer::lexer::Lexer;
use tenth::parser::parser::Parser;
use tenth::hir::lower::Lowerer;
use tenth::runtime::vm::Vm;
use tenth::runtime::value::Value;
use tenth::runtime::interpreter::Interpreter;
use tenth::compile::bytecode::BytecodeCompiler;
use tenth::compile::jit;
use std::rc::Rc;
use std::cell::RefCell;

fn lower(src: &str) -> Result<tenth::hir::hir::HirProgram, String> {
    let mut lexer = Lexer::new(src);
    let tokens = lexer.tokenize().map_err(|e| e.to_string())?;
    let mut parser = Parser::new(tokens);
    let program = parser.parse_program().map_err(|e| e.to_string())?;
    let mut lowerer = Lowerer::new();
    lowerer.lower_program(&program).map_err(|e| e.to_string())
}

fn setup_vm(hir: &tenth::hir::hir::HirProgram) -> Vm {
    let mut vm = Vm::new();
    vm.add_native("println".into(), |_vm, args| {
        for a in args { print!("{a}"); }
        println!();
        Ok(Value::Unit)
    });
    vm.add_native("Vec::new".into(), |_vm, _args| {
        Ok(Value::Vec(Rc::new(RefCell::new(Vec::new()))))
    });
    for func in &hir.functions {
        let compiler = BytecodeCompiler::new();
        if let Ok((chunk, closures)) = compiler.compile(func) {
            vm.add_fn(func.name.clone(), chunk);
            for (name, closure_chunk) in closures { vm.add_fn(name, closure_chunk); }
        }
    }
    if let Some(ref expr) = hir.main_expr {
        let compiler = BytecodeCompiler::new();
        if let Ok((chunk, closures)) = compiler.compile_main(expr) {
            vm.add_fn("main".into(), chunk);
            for (name, closure_chunk) in closures { vm.add_fn(name, closure_chunk); }
        }
    }
    vm
}

/// VM 字节码路径（也是 JIT hostcall 最终调用的同一套算术原语）。
fn run_vm(src: &str) -> Result<Value, String> {
    let hir = lower(src)?;
    let mut vm = setup_vm(&hir);
    if vm.has_fn("main") {
        vm.call("main").map_err(|e| e.to_string())
    } else {
        Ok(Value::Unit)
    }
}

/// JIT 路径。
fn run_jit(src: &str) -> Result<Value, String> {
    let hir = lower(src)?;
    let mut vm = setup_vm(&hir);
    if vm.has_fn("main") {
        jit::run_jit(&mut vm, "main").map_err(|e| e.to_string())
    } else {
        Ok(Value::Unit)
    }
}

/// 树遍解释器路径。
fn run_interp(src: &str) -> Result<Value, String> {
    let hir = lower(src)?;
    let mut interpreter = Interpreter::new(&hir);
    interpreter.execute_program(&hir)
        .map_err(|e| e.to_string())
        .map(|opt| opt.unwrap_or(Value::Unit))
}

/// 断言三条路径对同一源程序行为一致（同 Ok 值 / 同 Err）。
fn assert_consistent(src: &str) -> (Result<Value, String>, Result<Value, String>, Result<Value, String>) {
    let vm_r = run_vm(src);
    let jit_r = run_jit(src);
    let interp_r = run_interp(src);
    let okness = |r: &Result<Value, String>| r.is_ok();
    assert_eq!(
        okness(&vm_r), okness(&jit_r),
        "VM 与 JIT 成功性不一致，src: {src}\nVM: {:?}\nJIT: {:?}", vm_r, jit_r
    );
    assert_eq!(
        okness(&vm_r), okness(&interp_r),
        "VM 与解释器成功性不一致，src: {src}\nVM: {:?}\nInterp: {:?}", vm_r, interp_r
    );
    if let (Ok(a), Ok(b), Ok(c)) = (&vm_r, &jit_r, &interp_r) {
        assert_eq!(a.to_string(), b.to_string(), "VM/JIT 结果值不一致，src: {src}");
        assert_eq!(a.to_string(), c.to_string(), "VM/解释器结果值不一致，src: {src}");
    }
    (vm_r, jit_r, interp_r)
}

/// 断言三路径错误消息均包含 `msg`。
fn assert_err_contains(vm_r: Result<Value, String>, jit_r: Result<Value, String>, interp_r: Result<Value, String>, msg: &str) {
    let vm_err = vm_r.unwrap_err();
    let jit_err = jit_r.unwrap_err();
    let interp_err = interp_r.unwrap_err();
    assert!(vm_err.contains(msg), "VM 消息不符: {vm_err}");
    assert!(jit_err.contains(msg), "JIT 消息不符: {jit_err}");
    assert!(interp_err.contains(msg), "Interp 消息不符: {interp_err}");
}

// ── 溢出报错（三路径一致）────────────────────────────────────────────

#[test]
fn test_int_add_overflow_consistent() {
    let src = "fn main() -> Int { let a = 2147483647; let b = 1; a + b }";
    let (vm_r, jit_r, interp_r) = assert_consistent(src);
    assert_err_contains(vm_r, jit_r, interp_r, "整数运算结果 2147483648 溢出 i32 范围");
}

#[test]
fn test_int_add_overflow_mid_chain() {
    // 溢出发生在链式中间：JIT 应立即中断并上报第一个错误（emit_err_check_abort）。
    let src = "fn main() -> Int { let a = 2147483647; let b = 1; let m = a + b; m * 2 }";
    let (vm_r, jit_r, interp_r) = assert_consistent(src);
    assert_err_contains(vm_r, jit_r, interp_r, "整数运算结果 2147483648 溢出 i32 范围");
}

#[test]
fn test_int_sub_overflow_consistent() {
    let src = "fn main() -> Int { let a = -2147483648; let b = 1; a - b }";
    let (vm_r, jit_r, interp_r) = assert_consistent(src);
    assert_err_contains(vm_r, jit_r, interp_r, "整数运算结果 -2147483649 溢出 i32 范围");
}

#[test]
fn test_int_mul_overflow_consistent() {
    let src = "fn main() -> Int { let a = 46341; let b = 46341; a * b }";
    let (vm_r, jit_r, interp_r) = assert_consistent(src);
    assert_err_contains(vm_r, jit_r, interp_r, "整数运算结果 2147488281 溢出 i32 范围");
}

#[test]
fn test_int_div_overflow_consistent() {
    // i32::MIN / -1 = 2147483648，超出 i32 范围 → 报错（与 VM 一致）
    let src = "fn main() -> Int { let a = -2147483648; let b = -1; a / b }";
    let (vm_r, jit_r, interp_r) = assert_consistent(src);
    assert_err_contains(vm_r, jit_r, interp_r, "整数运算结果 2147483648 溢出 i32 范围");
}

#[test]
fn test_int_neg_overflow_consistent() {
    // -i32::MIN = 2147483648，超出 i32 范围 → 报错
    let src = "fn main() -> Int { let a = -2147483648; -a }";
    let (vm_r, jit_r, interp_r) = assert_consistent(src);
    assert_err_contains(vm_r, jit_r, interp_r, "整数运算结果 2147483648 溢出 i32 范围");
}

#[test]
fn test_i64_layer_overflow_consistent() {
    // i64 字面量后缀在运行时丢失为 I32（既有 dtype 保留问题），但值本身触发
    // checked_add 的 i64 层溢出（9223372036854775807 + 1 超出 i64）→ 干净报错。
    let src = "fn main() -> Int { let a = 9223372036854775807i64; let b = 1i64; a + b }";
    let (vm_r, jit_r, interp_r) = assert_consistent(src);
    assert_err_contains(vm_r, jit_r, interp_r, "整数运算结果溢出");
}

#[test]
fn test_div_by_zero_consistent() {
    let src = "fn main() -> Int { let a = 10; a / 0 }";
    let (vm_r, jit_r, interp_r) = assert_consistent(src);
    assert_err_contains(vm_r, jit_r, interp_r, "除数为零");
}

// ── 正常整数算术（不溢出，三路径一致且正确）────────────────────────

#[test]
fn test_normal_arith_consistent() {
    let src = "fn main() -> Int { let a = 3; let b = 4; a + b * 2 }";
    let (vm_r, jit_r, interp_r) = assert_consistent(src);
    assert_eq!(vm_r.unwrap().to_string(), "11");
    assert_eq!(jit_r.unwrap().to_string(), "11");
    assert_eq!(interp_r.unwrap().to_string(), "11");
}

#[test]
fn test_mul_below_limit_consistent() {
    // 46340 * 46341 = 2147441940 < i32::MAX，不溢出 → 三路径同值
    let src = "fn main() -> Int { let a = 46340; let b = 46341; a * b }";
    let (vm_r, jit_r, interp_r) = assert_consistent(src);
    assert_eq!(vm_r.unwrap().to_string(), "2147441940");
    assert_eq!(jit_r.unwrap().to_string(), "2147441940");
    assert_eq!(interp_r.unwrap().to_string(), "2147441940");
}

#[test]
fn test_neg_normal_consistent() {
    let src = "fn main() -> Int { let a = 42; -a }";
    let (vm_r, jit_r, interp_r) = assert_consistent(src);
    assert_eq!(vm_r.unwrap().to_string(), "-42");
    assert_eq!(jit_r.unwrap().to_string(), "-42");
    assert_eq!(interp_r.unwrap().to_string(), "-42");
}

#[test]
fn test_mod_normal_consistent() {
    let src = "fn main() -> Int { let a = 17; let b = 5; a % b }";
    let (vm_r, jit_r, interp_r) = assert_consistent(src);
    assert_eq!(vm_r.unwrap().to_string(), "2");
    assert_eq!(jit_r.unwrap().to_string(), "2");
    assert_eq!(interp_r.unwrap().to_string(), "2");
}

#[test]
fn test_while_loop_fallback_still_ok() {
    // 回归：循环触发 JIT fallback 到 VM，行为不受影响。
    let src = "fn main() -> Int { let mut s = 0; let mut i = 0; while i < 10 { s = s + i; i = i + 1; } s }";
    let (vm_r, jit_r, _) = assert_consistent(src);
    assert_eq!(vm_r.unwrap().to_string(), "45");
    assert_eq!(jit_r.unwrap().to_string(), "45");
}
