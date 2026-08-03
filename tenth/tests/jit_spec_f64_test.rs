//! M2.6-P3：f64 特化（scalar ABI 扩展）守护测试。
//!
//! P3 目标：把 A6 入参标量 ABI 从「仅 i64」扩展到：
//! 1. **f64 特化**——参数/返回含 `f64` 的函数推导混合/纯 f64 特化签名
//!    （`ChunkSig` 按参数顺序声明 I64/F64 种类；机器 ABI 保持 `(vm, i64×8) -> i64`
//!    位打包，f64 经 bitcast 塞 i64 寄存器——快/慢路径 ABI 统一，零性能损失）
//! 2. **`Int` 别名纳入**——`TypeParam("Int")` 与 `i64` 同为 `Value::Int`/I32 槽，
//!    推导为 I64 kind（扩大 i64 类覆盖）
//!
//! 守护内容：
//! - f64 递归（混合 i64+f64 签名 / 纯 f64 双递归）/ f64 多参 / f64 循环 /
//!   f64 返回消费链 / Int 别名纳入 / 混合签名
//! - **f64 错误路径语义对照 VM**：除零 → inf（IEEE，无错误）、NaN 传播——
//!   特化路径与 VM/解释器逐位一致
//! - **f64 相等 epsilon 语义**（P3 静默错值修复）：JIT 原生 F64 `==` 必须与
//!   VM/解释器的 `(x-y).abs() < 1e-10` 一致（如 0.1+0.2 == 0.3 → true）
//! - 覆盖断言（is_spec_compiled 且非 is_spec_failed）
//! - VM=JIT=解释器 三路径对拍（成功值逐位一致 + 错误消息/行号一致）

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

// ═══════════════════════════════════════════════════════════════════════════
// 进程内 helper（镜像 main.rs vm_execute：globals + __global_init + fn main 优先）
// ═══════════════════════════════════════════════════════════════════════════

fn lower(src: &str) -> Result<tenth::hir::hir::HirProgram, String> {
    let mut lexer = Lexer::new(src);
    let tokens = lexer.tokenize().map_err(|e| e.to_string())?;
    let mut parser = Parser::new(tokens);
    let program = parser.parse_program().map_err(|e| e.to_string())?;
    let mut lowerer = Lowerer::new();
    lowerer.lower_program(&program).map_err(|e| e.to_string())
}

fn compile_vm(src: &str) -> Result<Vm, String> {
    let hir = lower(src)?;
    let global_names: std::collections::HashSet<String> =
        hir.globals.iter().map(|g| g.name.clone()).collect();
    let mut vm = Vm::new();
    register_all_natives(&mut vm);
    for func in &hir.functions {
        let compiler = BytecodeCompiler::new_with_globals(global_names.clone());
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
    if !hir.globals.is_empty() {
        let gcompiler = BytecodeCompiler::new_with_globals(global_names.clone());
        match gcompiler.compile_globals(&hir.globals) {
            Ok((gchunk, gclosures)) => {
                vm.add_fn("__global_init".into(), gchunk);
                for (name, closure_chunk) in gclosures {
                    vm.add_fn(name, closure_chunk);
                }
                vm.call("__global_init").map_err(|e| format!("__global_init: {}", e))?;
            }
            Err(e) => return Err(format!("compile_globals error: {}", e)),
        }
    }
    if !vm.has_fn("main") {
        if let Some(ref expr) = hir.main_expr {
            let compiler = BytecodeCompiler::new_with_globals(global_names);
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
    }
    Ok(vm)
}

/// 纯 VM 字节码路径（不经 JIT）。
fn run_vm(src: &str) -> Result<Value, TenthError> {
    let mut vm = compile_vm(src)
        .map_err(|e| TenthError::RuntimeError { line: None, col: None, message: e })?;
    if vm.has_fn("main") { vm.call("main") } else { Ok(Value::Unit) }
}

/// JIT 路径（保留 Vm 供 is_spec_compiled 断言）。
fn run_jit_with_vm(src: &str) -> Result<(Value, Vm), TenthError> {
    let mut vm = compile_vm(src)
        .map_err(|e| TenthError::RuntimeError { line: None, col: None, message: e })?;
    if vm.has_fn("main") {
        let r = jit::run_jit(&mut vm, "main")?;
        Ok((r, vm))
    } else {
        Ok((Value::Unit, vm))
    }
}

/// 树遍解释器路径（= TENTH_NO_VM=1 的真实语义）。
fn run_interp(src: &str) -> Result<Value, TenthError> {
    let hir = lower(src)
        .map_err(|e| TenthError::RuntimeError { line: None, col: None, message: e })?;
    let mut interpreter = Interpreter::new(&hir);
    interpreter.execute_program(&hir)
        .map(|opt| opt.unwrap_or(Value::Unit))
}

fn float_of(v: Value, label: &str) -> f64 {
    match v { Value::Float(f) => f, other => panic!("[{label}] 期望 Float，实际 {:?}", other) }
}

fn int_of(v: Value, label: &str) -> i64 {
    match v { Value::Int(n, _) => n, other => panic!("[{label}] 期望 Int，实际 {:?}", other) }
}

/// main chunk 是否已 JIT 编译（防静默回退 VM）。
fn jit_compiled_main(vm: &Vm, label: &str) -> bool {
    let main_idx = vm.chunk_index_of("main").unwrap_or_else(|| panic!("[{label}] main chunk 未注册"));
    match vm.jit_ctx.as_ref() {
        Some(ctx) => ctx.is_compiled(main_idx),
        None => false,
    }
}

fn chunk_idx(vm: &Vm, name: &str) -> usize {
    vm.chunk_index_of(name).unwrap_or_else(|| panic!("chunk {name} 未注册"))
}

fn err_parts(err: &TenthError) -> (Option<usize>, String) {
    match err {
        TenthError::RuntimeError { line, message, .. } => (*line, message.clone()),
        other => (None, format!("{:?}", other)),
    }
}

/// 三路径对拍：成功性 + 值逐位一致 + 错误行号/消息一致。`require_compiled` = 断言 main 已编译。
fn assert_three_consistent(src: &str, label: &str, require_compiled: bool) {
    let vm_res = run_vm(src);
    let jit_pair = run_jit_with_vm(src);
    let interp_res = run_interp(src);

    let jit_res = match &jit_pair {
        Ok((v, vm)) => {
            if require_compiled {
                assert!(jit_compiled_main(vm, label),
                    "[{label}] main 未 JIT 编译（整函数回退 VM）——审计未覆盖 JIT 路径！\nsrc:\n{src}");
            }
            Ok(v.clone())
        }
        Err(e) => Err(e.clone()),
    };

    let okness = |r: &Result<Value, TenthError>| r.is_ok();
    assert_eq!(okness(&vm_res), okness(&jit_res),
        "[{label}] VM/JIT 成功性不一致\nVM: {:?}\nJIT: {:?}", vm_res, jit_res);
    assert_eq!(okness(&vm_res), okness(&interp_res),
        "[{label}] VM/解释器 成功性不一致\nVM: {:?}\nInterp: {:?}", vm_res, interp_res);

    match (&vm_res, &jit_res, &interp_res) {
        (Ok(a), Ok(b), Ok(c)) => {
            // 逐位一致（Debug 含 f64 全精度位模式；NaN/inf 来自同一计算 → 位一致）
            let sa = format!("{:?}", a);
            let sb = format!("{:?}", b);
            let sc = format!("{:?}", c);
            assert_eq!(sa, sb, "[{label}] VM/JIT 结果值不一致（静默错值红线！）\nVM={sa}\nJIT={sb}\nsrc:\n{src}");
            assert_eq!(sa, sc, "[{label}] VM/解释器 结果值不一致\nVM={sa}\nInterp={sc}\nsrc:\n{src}");
        }
        (Err(a), Err(b), Err(c)) => {
            let (la, ma) = err_parts(a);
            let (lb, mb) = err_parts(b);
            let (lc, mc) = err_parts(c);
            assert_eq!(la, lb, "[{label}] VM/JIT 错误行号不一致 VM={:?} JIT={:?}\nVM={ma}\nJIT={mb}\nsrc:\n{src}", la, lb);
            assert_eq!(la, lc, "[{label}] VM/解释器 错误行号不一致 VM={:?} Interp={:?}\nsrc:\n{src}", la, lc);
            assert_eq!(ma, mb, "[{label}] VM/JIT 错误消息不一致\nVM={ma}\nJIT={mb}\nsrc:\n{src}");
            assert_eq!(ma, mc, "[{label}] VM/解释器 错误消息不一致\nVM={ma}\nInterp={mc}\nsrc:\n{src}");
        }
        _ => panic!("[{label}] 三路径结果形态不一致\nVM: {:?}\nJIT: {:?}\nInterp: {:?}", vm_res, jit_res, interp_res),
    }
}

/// 断言指定函数已编译特化入口且未失败。
fn assert_spec_compiled(vm: &Vm, name: &str, label: &str) {
    let idx = chunk_idx(vm, name);
    let ctx = vm.jit_ctx.as_ref().expect("[{label}] JIT 上下文缺失");
    assert!(ctx.is_spec_compiled(idx), "[{label}] {name} 应编译特化入口（is_spec_compiled）");
    assert!(!ctx.is_spec_failed(idx), "[{label}] {name} 特化不应失败");
}

/// 断言指定函数**未**编译特化入口（如 f32/非标量类型不特化）。
fn assert_spec_not_compiled(vm: &Vm, name: &str, label: &str) {
    let idx = chunk_idx(vm, name);
    let ctx = vm.jit_ctx.as_ref().expect("[{label}] JIT 上下文缺失");
    assert!(!ctx.is_spec_compiled(idx), "[{label}] {name} 不应编译特化入口");
    assert!(!ctx.is_spec_failed(idx), "[{label}] {name} 不应特化失败（应走通用）");
}

// ═══════════════════════════════════════════════════════════════════════════
// 1. f64 递归（混合 i64+f64 签名 / 纯 f64 双递归）
// ═══════════════════════════════════════════════════════════════════════════

/// 混合 i64+f64 签名递归：`fn fp_sum(n: i64, x: f64) -> f64`。
/// 特化签名 [I64, F64] → F64；递归调用点实参 [I32, F64] 匹配 → 特化 ABI 全链路。
#[test]
fn spec_f64_recursive_mixed() {
    let src = r#"
        fn fp_sum(n: i64, x: f64) -> f64 {
            if n <= 0 { 0.0 } else { x + fp_sum(n - 1, x) }
        }
        fn main() -> f64 { fp_sum(28, 1.5) }
    "#;
    let (v, vm) = run_jit_with_vm(src).unwrap();
    assert_eq!(float_of(v, "fp_sum"), 42.0); // 28 * 1.5
    assert_spec_compiled(&vm, "fp_sum", "fp-sum-recursive");
    assert_three_consistent(src, "fp-sum-recursive-parity", true);
}

/// 纯 f64 双递归：`fn fp_fib(n: i64) -> f64`（i64 参 + f64 返回；递归永不内联）。
#[test]
fn spec_f64_recursive_fib() {
    let src = r#"
        fn fp_fib(n: i64) -> f64 {
            if n < 2 { 1.0 } else { fp_fib(n - 1) + fp_fib(n - 2) }
        }
        fn main() -> f64 { fp_fib(20) }
    "#;
    let (v, vm) = run_jit_with_vm(src).unwrap();
    // fp_fib(20) = fib(21) = 10946（fp_fib 定义为 fib(n+1)）
    assert_eq!(float_of(v, "fp_fib"), 10946.0);
    assert_spec_compiled(&vm, "fp_fib", "fp-fib-recursive");
    assert_three_consistent(src, "fp-fib-recursive-parity", true);
}

// ═══════════════════════════════════════════════════════════════════════════
// 2. 纯 f64 多参（>16 指令 → 非内联 → 特化 ABI）+ f64 循环
// ═══════════════════════════════════════════════════════════════════════════

/// 纯 f64 多参 + 大体内联不满足（>16 指令）→ 特化入口。结果 = a*b*c*d。
#[test]
fn spec_f64_multi_arg() {
    let src = r#"
        fn fp_prod(a: f64, b: f64, c: f64, d: f64) -> f64 {
            let p1 = a * b;
            let p2 = c * d;
            let p3 = p1 + p2;
            let q1 = p3 * 2.0;
            let q2 = q1 / 2.0;
            let q3 = q2 - p3;
            let r1 = q3 + a;
            let r2 = r1 + b;
            let r3 = r2 + c;
            let r4 = r3 + d;
            let s1 = r4 * 1.5;
            let s2 = s1 / 1.5;
            let t1 = s2 - r4;
            let u1 = t1 + a * b * c * d;
            u1
        }
        fn main() -> f64 { fp_prod(2.0, 3.0, 4.0, 5.0) }
    "#;
    // 各中间抵消：最终 u1 = a*b*c*d = 120
    let (v, vm) = run_jit_with_vm(src).unwrap();
    assert_eq!(float_of(v, "fp_prod"), 120.0);
    assert_spec_compiled(&vm, "fp_prod", "fp-prod-multi");
    assert_three_consistent(src, "fp-prod-multi-parity", true);
}

/// f64 循环（while 内 f64 累加 + i64 计数器）——混合签名 + 循环回边。
#[test]
fn spec_f64_loop() {
    let src = r#"
        fn fp_loop(n: i64, x: f64) -> f64 {
            let mut acc = 0.0;
            let mut p = 1.0;
            let mut i = 0;
            while i < n {
                p = p * x;
                acc = acc + p;
                i = i + 1;
            };
            acc
        }
        fn main() -> f64 { fp_loop(100, 0.5) }
    "#;
    // sum_{i=1..100} 0.5^i = 1 - 0.5^100 ≈ 1.0（浮点累加）
    let (v, vm) = run_jit_with_vm(src).unwrap();
    let f = float_of(v, "fp_loop");
    assert!((f - 1.0).abs() < 1e-6, "fp_loop(100, 0.5) 应 ≈ 1.0，实际 {f}");
    assert_spec_compiled(&vm, "fp_loop", "fp-loop");
    assert_three_consistent(src, "fp-loop-parity", true);
}

/// f64 返回结果在调用方被 f64 原生运算消费（特化结果槽种类 F64 → 下游原生 fadd/fmul）。
/// 体内联不满足（>16 指令）→ 走特化 ABI，返回 f64 标量槽供调用方原生消费。
#[test]
fn spec_f64_return_consumed() {
    let src = r#"
        fn half(x: f64) -> f64 {
            let a = x / 2.0;
            let b = a + 1.0;
            let c = b - 1.0;
            let d = c * 2.0;
            let e = d / 2.0;
            let f = e + 0.0;
            let g = f * 1.0;
            let h = g / 1.0;
            let j = h + a;
            let k = j - a;
            let m = k * 1.0;
            let n = m + 0.0;
            let o = n - 0.0;
            o
        }
        fn main() -> f64 { half(10.0) * 3.0 + half(8.0) }
    "#;
    // half(x) = x/2（b/c/d/e 抵消回 a，j/k 抵消回 a）；5*3 + 4 = 19
    let (v, vm) = run_jit_with_vm(src).unwrap();
    assert_eq!(float_of(v, "half-chain"), 19.0);
    assert_spec_compiled(&vm, "half", "half-return");
    assert_three_consistent(src, "half-return-parity", true);
}

/// 特化 f64 调用链（多级 spec 嵌套）。
#[test]
fn spec_f64_call_chain() {
    let src = r#"
        fn f1(x: f64) -> f64 { x + 1.0 }
        fn f2(x: f64) -> f64 { f1(x) * 2.0 }
        fn f3(x: f64) -> f64 { f2(x) - 3.0 }
        fn main() -> f64 { f3(5.0) }
    "#;
    // f3(5) = (6*2) - 3 = 9
    let (v, vm) = run_jit_with_vm(src).unwrap();
    assert_eq!(float_of(v, "f3-chain"), 9.0);
    assert_spec_compiled(&vm, "f3", "f3-chain");
    assert_three_consistent(src, "f3-chain-parity", true);
}

// ═══════════════════════════════════════════════════════════════════════════
// 3. f64 错误路径语义对照 VM：除零 → inf / NaN（IEEE，无错误）
// ═══════════════════════════════════════════════════════════════════════════

/// f64 除零 → inf（VM/解释器/JIT 均无错误，IEEE 语义；特化路径一致）。
/// 体内联不满足（>16 指令）→ 走特化 ABI。
#[test]
fn spec_f64_div_zero_inf() {
    let src = r#"
        fn fdiv(a: f64, b: f64) -> f64 {
            let d = a / b;
            let e = d + 0.0;
            let f = e * 1.0;
            let g = f / 1.0;
            let h = g - 0.0;
            let j = h + d;
            let k = j / 2.0;
            let m = k * 2.0;
            m
        }
        fn main() -> f64 { fdiv(1.0, 0.0) }
    "#;
    let (v, vm) = run_jit_with_vm(src).unwrap();
    assert!(float_of(v, "fdiv-inf").is_infinite(), "fdiv(1,0) 应得 inf");
    assert_spec_compiled(&vm, "fdiv", "fdiv-inf");
    assert_three_consistent(src, "fdiv-inf-parity", true);
}

/// f64 0/0 → NaN 传播（三路径逐位一致；特化路径）。
#[test]
fn spec_f64_nan_propagation() {
    let src = r#"
        fn fdiv(a: f64, b: f64) -> f64 {
            let d = a / b;
            let e = d + 0.0;
            let f = e * 1.0;
            let g = f / 1.0;
            let h = g - 0.0;
            let j = h + d;
            let k = j / 2.0;
            let m = k * 2.0;
            m
        }
        fn main() -> f64 { fdiv(0.0, 0.0) + 1.0 }
    "#;
    let (v, vm) = run_jit_with_vm(src).unwrap();
    assert!(float_of(v, "fdiv-nan").is_nan(), "fdiv(0,0) 应得 NaN");
    assert_spec_compiled(&vm, "fdiv", "fdiv-nan");
    assert_three_consistent(src, "fdiv-nan-parity", true);
}

/// 混合签名内的整数错误路径（i64 除零）——特化函数内报错带行号，不静默。
/// 体内联不满足（>16 指令）→ 特化入口内的原生 I32 Div 错误链。
#[test]
fn spec_f64_mixed_int_div_zero_error() {
    let src = r#"
        fn f(n: i64, x: f64) -> f64 {
            let a = n - n;
            let b = n / a;
            let c = b * 1;
            let d = c + 0;
            let e = d * 2;
            let g = e / 2;
            let h = g - 0;
            let j = h + 1;
            let m = j - 1;
            let p = m * 3;
            let q = p / 3;
            q + x
        }
        fn main() -> f64 { f(5, 1.5) }
    "#;
    let err = match run_jit_with_vm(src) {
        Err(e) => e,
        Ok(_) => panic!("[mixed-div0] 应报 RuntimeError"),
    };
    match &err {
        TenthError::RuntimeError { line, message, .. } => {
            assert!(line.is_some(), "[mixed-div0] 应携带行号，实际 message={}", message);
            assert!(message.contains("整数除零"),
                "[mixed-div0] 错误信息应含 '整数除零'，实际: {}", message);
        }
        other => panic!("[mixed-div0] 应报 RuntimeError，实际 {:?}", other),
    }
    assert_three_consistent(src, "mixed-div0-parity", true);
}

// ═══════════════════════════════════════════════════════════════════════════
// 4. f64 相等 epsilon 语义（P3 静默错值修复的回归守护）
// ═══════════════════════════════════════════════════════════════════════════

/// VM/解释器浮点 `==` 为 epsilon（|x-y| < 1e-10）；JIT 原生 F64 比较必须一致。
/// 0.1+0.2 与 0.3：精确不等但 epsilon 相等 → true（此前 JIT 精确比较会给 false）。
/// 用 f64 返回的包装函数（bool 返回无特化签名），且体内联不满足（>16 指令）——
/// 在**特化路径内**做比较。
#[test]
fn spec_f64_epsilon_eq_regression() {
    let src = r#"
        fn feq(a: f64, b: f64) -> f64 {
            let r = if a == b { 1.0 } else { 0.0 };
            let r2 = r + 0.0;
            let r3 = r2 * 1.0;
            let r4 = r3 / 1.0;
            let r5 = r4 - 0.0;
            let r6 = r5 + r;
            let r7 = r6 / 2.0;
            let r8 = r7 * 2.0;
            let r9 = r8 - r;
            r9
        }
        fn fneq(a: f64, b: f64) -> f64 {
            let r = if a != b { 1.0 } else { 0.0 };
            let r2 = r + 0.0;
            let r3 = r2 * 1.0;
            let r4 = r3 / 1.0;
            let r5 = r4 - 0.0;
            let r6 = r5 + r;
            let r7 = r6 / 2.0;
            let r8 = r7 * 2.0;
            let r9 = r8 - r;
            r9
        }
        fn main() -> f64 {
            feq(0.1 + 0.2, 0.3) * 100.0
                + fneq(0.1 + 0.2, 0.3) * 10.0
                + feq(1.5, 1.5) * 1.0
                + fneq(1.5, 1.5) * 0.1
                + feq(1.5, 1.6) * 0.01
                + fneq(1.5, 1.6) * 0.001
        }
    "#;
    // e=true→100；n=false→0；e2=true→1；n2=false→0；e3=false→0；n3=true→0.001
    let (v, vm) = run_jit_with_vm(src).unwrap();
    assert_eq!(float_of(v, "epsilon"), 101.001);
    assert_spec_compiled(&vm, "feq", "epsilon-feq");
    assert_spec_compiled(&vm, "fneq", "epsilon-fneq");
    assert_three_consistent(src, "epsilon-parity", true);
    // 核心回归：0.1+0.2 == 0.3 在特化路径必须为 1.0（epsilon 语义）
    let src2 = r#"
        fn feq(a: f64, b: f64) -> f64 {
            let r = if a == b { 1.0 } else { 0.0 };
            let r2 = r + 0.0;
            let r3 = r2 * 1.0;
            let r4 = r3 / 1.0;
            let r5 = r4 - 0.0;
            let r6 = r5 + r;
            let r7 = r6 / 2.0;
            let r8 = r7 * 2.0;
            let r9 = r8 - r;
            r9
        }
        fn main() -> f64 { feq(0.1 + 0.2, 0.3) }
    "#;
    let (v2, vm2) = run_jit_with_vm(src2).unwrap();
    assert_eq!(float_of(v2, "epsilon-core"), 1.0, "0.1+0.2 == 0.3 应为 true（epsilon 语义）");
    assert_spec_compiled(&vm2, "feq", "epsilon-core");
}

/// f64 有序比较（Lt/Gt/Lte/Gte）精确语义对照 VM。体非内联（>16 指令）→ 特化路径。
#[test]
fn spec_f64_ordered_cmp() {
    let src = r#"
        fn cmp(a: f64, b: f64) -> f64 {
            let lt = if a < b { 1.0 } else { 0.0 };
            let gt = if a > b { 1.0 } else { 0.0 };
            let le = if a <= b { 1.0 } else { 0.0 };
            let ge = if a >= b { 1.0 } else { 0.0 };
            let r = lt * 1000.0 + gt * 100.0 + le * 10.0 + ge;
            let r2 = r + 0.0;
            let r3 = r2 * 1.0;
            let r4 = r3 / 1.0;
            let r5 = r4 - 0.0;
            r5
        }
        fn main() -> f64 { cmp(1.5, 2.5) }
    "#;
    // lt=true, gt=false, le=true, ge=false → 1000 + 0 + 10 + 0 = 1010
    let (v, vm) = run_jit_with_vm(src).unwrap();
    assert_eq!(float_of(v, "ordered"), 1010.0);
    assert_spec_compiled(&vm, "cmp", "ordered-cmp");
    assert_three_consistent(src, "ordered-cmp-parity", true);
}

// ═══════════════════════════════════════════════════════════════════════════
// 5. Int 别名纳入（P3）+ 混合签名 + f32/其他类型不特化
// ═══════════════════════════════════════════════════════════════════════════

/// `Int` 别名（TypeParam("Int")）纳入特化：与 i64 同为 I32 槽 → 特化入口编译。
/// 体内联不满足（>16 指令）→ 走 Int 特化 ABI。
#[test]
fn spec_int_alias_included() {
    let src = r#"
        fn add_int(a: Int, b: Int) -> Int {
            let s = a + b;
            let t = s * 1;
            let u = t / 1;
            let v = u + 0;
            let w = v - 0;
            let x = w * 2;
            let y = x / 2;
            let z = y + s;
            let z2 = z - s;
            let z3 = z2 * 3;
            let z4 = z3 / 3;
            z4
        }
        fn main() -> Int { add_int(30, 12) }
    "#;
    let (v, vm) = run_jit_with_vm(src).unwrap();
    assert_eq!(int_of(v, "add-int"), 42);
    assert_spec_compiled(&vm, "add_int", "int-alias");
    assert_three_consistent(src, "int-alias-parity", true);
}

/// `Int` 递归纳入特化（扩大 i64 类覆盖）。
#[test]
fn spec_int_recursive() {
    let src = r#"
        fn fibi(n: Int) -> Int {
            if n < 2 { n } else { fibi(n - 1) + fibi(n - 2) }
        }
        fn main() -> Int { fibi(20) }
    "#;
    let (v, vm) = run_jit_with_vm(src).unwrap();
    assert_eq!(int_of(v, "int-fib"), 6765);
    assert_spec_compiled(&vm, "fibi", "int-recursive");
    assert_three_consistent(src, "int-recursive-parity", true);
}

/// 混合 i64+f64 签名（[I64, F64]→I64）特化。
#[test]
fn spec_mixed_i64_f64_sig() {
    let src = r#"
        fn scale(a: i64, k: f64) -> i64 {
            let f = to_float(a) * k;
            if f > 100.0 { a } else { a + 1 }
        }
        fn main() -> i64 { scale(3, 2.0) + scale(1, 0.5) }
    "#;
    let (v, vm) = run_jit_with_vm(src).unwrap();
    assert_eq!(int_of(v, "mixed-scale"), 6);
    assert_spec_compiled(&vm, "scale", "mixed-i64-f64");
    assert_three_consistent(src, "mixed-i64-f64-parity", true);
}

/// f32（Float32）不纳入特化（A2 native 路径无 f32；保留 hostcall 语义）。
#[test]
fn spec_f32_not_included() {
    let src = r#"
        fn f32add(a: f32, b: f32) -> f32 { a + b }
        fn main() -> f32 { f32add(1.5f32, 2.5f32) }
    "#;
    let (v, vm) = run_jit_with_vm(src).unwrap();
    match v {
        Value::Float32(f) => assert_eq!(f, 4.0f32),
        other => panic!("[f32] 期望 Float32，实际 {:?}", other),
    }
    assert_spec_not_compiled(&vm, "f32add", "f32-not-spec");
    assert_three_consistent(src, "f32-not-spec-parity", true);
}

/// 非标量返回（str）不特化。
#[test]
fn spec_f64_non_scalar_return_not_spec() {
    let src = r#"
        fn label(x: f64) -> str { to_string(x) }
        fn main() -> str { label(1.5) }
    "#;
    let (v, vm) = run_jit_with_vm(src).unwrap();
    match v {
        Value::String(s) => assert_eq!(s, "1.5"),
        other => panic!("[label] 期望 String，实际 {:?}", other),
    }
    assert_spec_not_compiled(&vm, "label", "str-return");
}

// ═══════════════════════════════════════════════════════════════════════════
// 6. 特化 f64 慢路径（首次调用编译特化入口 → host_jit_call_spec）
// ═══════════════════════════════════════════════════════════════════════════

/// 首次调用触发特化慢路径（spec 表未填充 → host_jit_call_spec → 编译 + 调用），
/// 后续走快路径。结果与 VM/解释器一致。
#[test]
fn spec_f64_slow_path_first_call() {
    let src = r#"
        fn fp_sum(n: i64, x: f64) -> f64 {
            if n <= 0 { 0.0 } else { x + fp_sum(n - 1, x) }
        }
        fn main() -> f64 {
            let a = fp_sum(5, 2.0);   // 首次 → 慢路径编译特化入口
            let b = fp_sum(5, 3.0);   // 快路径
            a + b
        }
    "#;
    // 5*2 + 5*3 = 10 + 15 = 25
    let (v, vm) = run_jit_with_vm(src).unwrap();
    assert_eq!(float_of(v, "slow-first"), 25.0);
    assert_spec_compiled(&vm, "fp_sum", "slow-first");
    assert_three_consistent(src, "slow-first-parity", true);
}
