//! M2.5-A6：入参标量 ABI（scalar-specialized call ABI）守护测试。
//!
//! 目标：把 JIT 函数调用从「Value 内存模型通用签名」推进到「静态类型寄存器传递」
//! （纯 i64 标量函数 → 特化入口 `(vm, i64 x MAX_SPEC_ARGS) -> i64`，参数/返回走
//! 寄存器，跳过整个 Value 装箱链）。本套件守护：
//! - 特化函数正确性（递归 fib / 多参 / 嵌套递归 / 含 hostcall）
//! - 混合签名走通用路径（f64 参数不特化，零行为变化）
//! - 错误路径（除零 / 溢出）带行号，不静默
//! - 特化函数被间接调用走通用入口（双入口共存）
//! - 覆盖断言（is_spec_compiled 且非 is_spec_failed）
//! - VM=JIT 对拍一致
//!
//! 注意：签名从 `Chunk.scalar_sig`（BytecodeCompiler 编译时从 HIR 推导）读取，
//! 测试 helper 与 main.rs 同构（compile → add_fn），因此特化自动生效。

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

/// 纯 VM 路径执行 main（`vm.call`，不经 JIT——VM 侧不受 A6 影响，作对拍基准）。
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

/// JIT 执行并保留 Vm（供 is_spec_compiled/is_spec_failed 覆盖断言）。
fn run_jit_with_vm(src: &str) -> Result<(Value, Vm), String> {
    let mut vm = compile_vm(src)?;
    if vm.has_fn("main") {
        let r = jit::run_jit(&mut vm, "main").map_err(|e| e.to_string())?;
        Ok((r, vm))
    } else {
        Ok((Value::Unit, vm))
    }
}

fn int_of(v: Value, label: &str) -> i64 {
    match v {
        Value::Int(n, _) => n,
        other => panic!("[{label}] 期望 Int，实际 {:?}", other),
    }
}

/// chunk 名 → chunk 索引（覆盖断言用）。
fn chunk_idx(vm: &Vm, name: &str) -> usize {
    vm.chunk_index_of(name).unwrap_or_else(|| panic!("chunk {name} 未注册"))
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

// ── 1. 递归 fib（特化全链路：参数/算术/调用/返回全寄存器）───────────────────

#[test]
fn spec_fib_recursive() {
    let src = r#"
        fn fib(n: i64) -> i64 {
            if n < 2 { n } else { fib(n - 1) + fib(n - 2) }
        }
        fn main() -> i64 { fib(20) }
    "#;
    let (v, vm) = run_jit_with_vm(src).unwrap();
    assert_eq!(int_of(v, "fib"), 6765);
    // 覆盖断言：fib 应已编译特化入口（递归调用点触发），且非特化失败。
    let fib = chunk_idx(&vm, "fib");
    let ctx = vm.jit_ctx.as_ref().expect("JIT 上下文应存在");
    assert!(ctx.is_spec_compiled(fib), "fib 应编译特化入口（is_spec_compiled）");
    assert!(!ctx.is_spec_failed(fib), "fib 特化不应失败");
}

#[test]
fn spec_fib_recursive_vm_jit_parity() {
    // 更大的递归深度（fib(28)），VM=JIT 对拍。
    let src = r#"
        fn fib(n: i64) -> i64 {
            if n < 2 { n } else { fib(n - 1) + fib(n - 2) }
        }
        fn main() -> i64 { fib(28) }
    "#;
    assert_vm_jit_int(src, 317811, "fib28-parity");
}

// ── 2. i64 多参函数（寄存器多参传递）─────────────────────────────────────

#[test]
fn spec_multi_arg_i64() {
    // add3 含调用（add2）→ 不可内联 → 特化入口编译（3 个 i64 寄存器参数）。
    let src = r#"
        fn add2(a: i64, b: i64) -> i64 { a + b }
        fn add3(a: i64, b: i64, c: i64) -> i64 { add2(a, b) + c }
        fn main() -> i64 { add3(10, 20, 30) }
    "#;
    let (v, vm) = run_jit_with_vm(src).unwrap();
    assert_eq!(int_of(v, "add3"), 60);
    let ctx = vm.jit_ctx.as_ref().unwrap();
    let add3 = chunk_idx(&vm, "add3");
    assert!(ctx.is_spec_compiled(add3), "add3 应编译特化入口");
    assert!(!ctx.is_spec_failed(add3), "add3 特化不应失败");
}

#[test]
fn spec_multi_arg_vm_jit_parity() {
    let src = r#"
        fn sub3(a: i64, b: i64, c: i64) -> i64 { a - b - c }
        fn main() -> i64 { sub3(100, 30, 7) }
    "#;
    assert_vm_jit_int(src, 63, "sub3-parity");
}

// ── 3. 嵌套递归（双函数互调，特化全链路）────────────────────────────────

#[test]
fn spec_nested_recursion() {
    // 递归调用后跟 `+ 0`（非尾位置）→ CallN 特化路径（不是 TailCall 保守路径）。
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
    assert_eq!(int_of(v, "even-odd"), 1);
    let ctx = vm.jit_ctx.as_ref().unwrap();
    let even = chunk_idx(&vm, "even");
    let odd = chunk_idx(&vm, "odd");
    assert!(ctx.is_spec_compiled(even), "even 应编译特化入口");
    assert!(ctx.is_spec_compiled(odd), "odd 应编译特化入口");
    assert!(!ctx.is_spec_failed(even) && !ctx.is_spec_failed(odd), "互递归特化不应失败");
}

// ── 4. 混合签名（P3：f64 纳入特化；f32/其他类型仍走通用，零行为变化）─────────

#[test]
fn spec_mixed_signature_generic() {
    // P3：scale 含 f64 参数 → **特化**（I64+F64 混合签名 [I64,F64]→I64）；含 to_float
    // 调用 → 不可内联 → 特化 ABI 路径（真实覆盖「f64 参数 → 特化」逻辑）。
    let src = r#"
        fn scale(a: i64, k: f64) -> i64 {
            let f = to_float(a) * k;
            if f > 100.0 { a } else { a + 1 }
        }
        fn main() -> i64 { scale(3, 2.0) + scale(1, 0.5) }
    "#;
    let (v, vm) = run_jit_with_vm(src).unwrap();
    assert_eq!(int_of(v, "scale"), 6);
    let ctx = vm.jit_ctx.as_ref().unwrap();
    let scale = chunk_idx(&vm, "scale");
    // P3：f64 参数 → 特化入口编译（I64+F64 混合签名）。
    assert!(ctx.is_spec_compiled(scale), "含 f64 参数的混合签名应编译特化入口（P3）");
    assert!(!ctx.is_spec_failed(scale), "scale 特化不应失败");
}

#[test]
fn spec_mixed_signature_vm_jit_parity() {
    let src = r#"
        fn scale(a: i64, k: f64) -> i64 {
            let f = to_float(a) * k;
            if f > 100.0 { a } else { a + 1 }
        }
        fn main() -> i64 { scale(5, 1.5) + scale(2, 200.0) }
    "#;
    // scale(5, 1.5): f=7.5 → 6; scale(2, 200.0): f=400 → 2 → 8
    assert_vm_jit_int(src, 8, "scale-parity");
}

/// 非标量返回（str）→ 不特化。
#[test]
fn spec_non_scalar_return_not_spec() {
    let src = r#"
        fn label(a: i64) -> str { to_string(a) }
        fn main() -> i64 { parse_int(label(7)) + parse_int(label(8)) }
    "#;
    let (v, vm) = run_jit_with_vm(src).unwrap();
    assert_eq!(int_of(v, "label"), 15);
    let ctx = vm.jit_ctx.as_ref().unwrap();
    let label = chunk_idx(&vm, "label");
    assert!(!ctx.is_spec_compiled(label), "str 返回不应编译特化入口");
}

// ── 5. 错误路径（除零 / 溢出）：不静默、带行号 ─────────────────────────────

/// 特化函数内整数除零 → 错误（不静默），带源码行号。
#[test]
fn spec_error_div_zero_with_line() {
    let src = "fn f(a: i64) -> i64 { 100 / (a - a) }\nfn main() -> i64 { f(5) }";
    let err = run_jit(src).unwrap_err();
    match &err {
        TenthError::RuntimeError { line, message, .. } => {
            assert!(line.is_some(), "[div0] 应携带行号，实际 message={}", message);
            assert!(
                message.contains("整数除零"),
                "[div0] 错误信息应含 '整数除零'，实际: {}",
                message
            );
        }
        other => panic!("[div0] 期望 RuntimeError，实际 {:?}", other),
    }
    // VM 侧同样报错（对拍：特化不改变错误语义）。
    let vm_err = run_vm(src).unwrap_err();
    match &vm_err {
        TenthError::RuntimeError { message, .. } => {
            assert!(message.contains("整数除零"), "[div0-vm] 实际: {}", message);
        }
        other => panic!("[div0-vm] 期望 RuntimeError，实际 {:?}", other),
    }
}

/// 特化函数内整数溢出 → 错误（不静默），带源码行号。
#[test]
fn spec_error_overflow_with_line() {
    let src = "fn f(a: i64) -> i64 { a + 1 }\nfn main() -> i64 { f(2147483647) }";
    let err = run_jit(src).unwrap_err();
    match &err {
        TenthError::RuntimeError { line, message, .. } => {
            assert!(line.is_some(), "[ovf] 应携带行号，实际 message={}", message);
            assert!(
                message.contains("溢出"),
                "[ovf] 错误信息应含 '溢出'，实际: {}",
                message
            );
        }
        other => panic!("[ovf] 期望 RuntimeError，实际 {:?}", other),
    }
}

// ── 6. 特化函数双入口（通用 + 特化，惰性编译）──────────────────────────────

#[test]
fn spec_dual_entry_generic_and_spec() {
    // fib 既被「非标量参数」调用（parse_int 结果 → 通用入口 A1），又被「标量参数」
    // 调用（5 → 特化入口）——双入口共存且各自正确。fib 不可内联（递归）。
    let src = r#"
        fn fib(n: i64) -> i64 {
            if n < 2 { n } else { fib(n - 1) + fib(n - 2) }
        }
        fn main() -> i64 {
            let n = parse_int("10");
            fib(n) + fib(5)
        }
    "#;
    let (v, vm) = run_jit_with_vm(src).unwrap();
    assert_eq!(int_of(v, "fib-dual"), 60);
    let ctx = vm.jit_ctx.as_ref().unwrap();
    let fib = chunk_idx(&vm, "fib");
    assert!(ctx.is_compiled(fib), "fib 通用入口应编译（非标量参数调用点走通用）");
    assert!(ctx.is_spec_compiled(fib), "fib 特化入口应编译（标量参数调用点走特化）");
    assert!(!ctx.is_spec_failed(fib), "fib 特化不应失败");
}

/// 特化函数经函数值间接调用（CallClosure → 通用/VM 路径）仍正确（双入口兼容）。
#[test]
fn spec_indirect_via_function_value() {
    let src = r#"
        fn sq(a: i64) -> i64 { a * a }
        fn apply(f: fn(i64) -> i64, x: i64) -> i64 { f(x) }
        fn main() -> i64 { apply(sq, 7) }
    "#;
    // 间接调用不触发 sq 特化入口（走 VM call_value），结果仍正确。
    assert_vm_jit_int(src, 49, "sq-indirect-fnval");
}

// ── 7. 特化函数含 hostcall（字符串/println）──────────────────────────────

#[test]
fn spec_with_hostcall_inside() {
    let src = r#"
        fn f(a: i64) -> i64 {
            let s = to_string(a);
            println(s);
            a + 1
        }
        fn main() -> i64 { f(41) }
    "#;
    let (v, vm) = run_jit_with_vm(src).unwrap();
    assert_eq!(int_of(v, "hostcall"), 42);
    let ctx = vm.jit_ctx.as_ref().unwrap();
    let f = chunk_idx(&vm, "f");
    assert!(ctx.is_spec_compiled(f), "含 hostcall 的 i64 函数仍可特化");
}

// ── 8. 特化调用点参数来自 native（非标量槽）→ 回退通用，结果仍正确 ───────

#[test]
fn spec_arg_from_native_falls_back_generic() {
    let src = r#"
        fn inc(a: i64) -> i64 { a + 1 }
        fn main() -> i64 {
            let n = parse_int("41");
            inc(n)
        }
    "#;
    // parse_int 返回 Value（分析期 Unknown）→ inc 调用点参数非标量槽 →
    // 回退通用 A1 路径；结果仍正确（不静默错值）。
    assert_vm_jit_int(src, 42, "native-arg");
}

// ── 9. 特化函数返回被进一步特化函数消费（链式特化调用）───────────────────

#[test]
fn spec_chained_calls() {
    let src = r#"
        fn double(a: i64) -> i64 { a * 2 }
        fn quad(a: i64) -> i64 { double(double(a)) }
        fn main() -> i64 { quad(5) + 1 }
    "#;
    let (v, vm) = run_jit_with_vm(src).unwrap();
    assert_eq!(int_of(v, "chain"), 21);
    let ctx = vm.jit_ctx.as_ref().unwrap();
    let quad = chunk_idx(&vm, "quad");
    // quad 含调用（不可内联）→ 编译特化入口；double 是平凡小函数 → 调用点内联。
    assert!(ctx.is_spec_compiled(quad), "quad 应编译特化入口");
    assert!(!ctx.is_spec_failed(quad), "quad 特化不应失败");
}

// ── 10. 参数跨条件分支（内联优先：纯分支小函数被内联；含调用才特化）───────

#[test]
fn spec_param_in_branches() {
    // classify 是纯分支小函数 → 调用点内联（A2 内联优先）。值正确验证
    // spec 参数跨分支语义（VM=JIT 对拍）。
    let src = r#"
        fn classify(x: i64) -> i64 {
            if x < 0 { -1 } else if x == 0 { 0 } else { 1 }
        }
        fn main() -> i64 { classify(-5) + classify(0) + classify(9) }
    "#;
    assert_vm_jit_int(src, 0, "classify-inline");
}

/// 参数跨多条件分支 + 嵌套调用（不可内联）→ 特化入口编译；参数经递归 helper
/// 流入分支条件。
#[test]
fn spec_param_branches_noninline() {
    let src = r#"
        fn step(x: i64, d: i64) -> i64 {
            if d <= 0 { x } else { step(x + 1, d - 1) }
        }
        fn classify(x: i64) -> i64 {
            let v = step(x, 0);
            if v < 0 { -1 } else if v == 0 { 0 } else { 1 }
        }
        fn main() -> i64 { classify(-5) + classify(0) + classify(9) }
    "#;
    let (v, vm) = run_jit_with_vm(src).unwrap();
    assert_eq!(int_of(v, "classify-ni"), 0);
    let ctx = vm.jit_ctx.as_ref().unwrap();
    let c = chunk_idx(&vm, "classify");
    assert!(ctx.is_spec_compiled(c), "classify（含调用）应编译特化入口");
    assert!(!ctx.is_spec_failed(c), "classify 特化不应失败");
    let s = chunk_idx(&vm, "step");
    assert!(ctx.is_spec_compiled(s), "step（递归）应编译特化入口");
}
