//! M2-A5：VM-vs-JIT 系统性一致性对拍套件。
//!
//! 目标：把 A1-A3 全部新能力（直接调用 / 内联 / 标量专用化 / opcode 覆盖）在
//! 「VM = JIT」双路径下系统性对拍，每类能力 ≥3 用例，含错误路径与行号一致性。
//!
//! 与 `jit_test.rs` 的关系：jit_test 侧重单路径功能验证 + 覆盖断言（is_compiled）；
//! 本套件侧重**系统性边界**——互递归、深层嵌套、循环内调用、标量全比较集、
//! Try 多层/链式/循环、Tuple 深层/OOB、Struct 字段链/循环写、Spawn eager、
//! TailCall 链/深递归/混合，以及跨路径错误行号一致性（VM line == JIT line）。
//!
//! 策略：
//! - 成功路径：`assert_vm_jit_int`（两侧均等于期望值且互等）/ `assert_vm_jit_debug`
//!   （结构化 Debug 一致）/ `assert_vm_jit_float_parity`（浮点互等 + 近似期望）。
//! - 错误路径：`assert_vm_jit_err_line_parity`（两侧均报错、行号一致且等于期望行）。
//! - 热函数覆盖：`run_jit_with_vm` 保留 Vm，断言「被调函数已按需编译」（分层分析守护）。

use tenth::compile::bytecode::BytecodeCompiler;
use tenth::compile::jit;
use tenth::error::TenthError;
use tenth::hir::lower::Lowerer;
use tenth::lexer::lexer::Lexer;
use tenth::parser::parser::Parser;
use tenth::runtime::natives::register_all_natives;
use tenth::runtime::value::Value;
use tenth::runtime::vm::Vm;

/// 编译源码到 VM（含全部 natives）。
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

/// 纯 VM 路径执行 main（`vm.call`，不经 JIT）。
fn run_vm(src: &str) -> Result<Value, TenthError> {
    let mut vm = compile_vm(src).map_err(|e| TenthError::RuntimeError { line: None, col: None, message: e })?;
    if vm.has_fn("main") {
        vm.call("main")
    } else {
        Ok(Value::Unit)
    }
}

/// JIT 路径执行 main（`jit::run_jit`，内部失败自动回退 VM）。
fn run_jit(src: &str) -> Result<Value, TenthError> {
    let mut vm = compile_vm(src).map_err(|e| TenthError::RuntimeError { line: None, col: None, message: e })?;
    if vm.has_fn("main") {
        jit::run_jit(&mut vm, "main")
    } else {
        Ok(Value::Unit)
    }
}

/// 通过 JIT 执行并保留 Vm（供 is_compiled/is_failed 覆盖断言）。
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
    match v { Value::Int(n, _) => n, other => panic!("[{label}] 期望 Int，实际 {:?}", other) }
}

/// VM 与 JIT 均产出等于 `expected` 的 Int，且两侧互等。
fn assert_vm_jit_int(src: &str, expected: i64, label: &str) {
    let vm_res = run_vm(src).unwrap_or_else(|e| panic!("[{label}] VM 执行失败: {}", e));
    let jit_res = run_jit(src).unwrap_or_else(|e| panic!("[{label}] JIT 执行失败: {}", e));
    let vm_i = int_of(vm_res, label);
    let jit_i = int_of(jit_res, label);
    assert_eq!(vm_i, expected, "[{label}] VM 结果错误: {} != {}", vm_i, expected);
    assert_eq!(jit_i, expected, "[{label}] JIT 结果错误: {} != {}", jit_i, expected);
    assert_eq!(vm_i, jit_i, "[{label}] VM/JIT 不一致: {} != {}", vm_i, jit_i);
}

/// VM 与 JIT 的 Debug（→Display）结构化一致（Tuple/Struct/Result/Future 等）。
fn assert_vm_jit_debug(src: &str, label: &str) {
    let vm_res = run_vm(src).unwrap_or_else(|e| panic!("[{label}] VM 执行失败: {}", e));
    let jit_res = run_jit(src).unwrap_or_else(|e| panic!("[{label}] JIT 执行失败: {}", e));
    let vm_db = format!("{:?}", vm_res);
    let jit_db = format!("{:?}", jit_res);
    assert_eq!(vm_db, jit_db, "[{label}] VM/JIT Debug 不一致: VM={} JIT={}", vm_db, jit_db);
}

/// VM 与 JIT 均产出接近 `expected` 的 Float 且两侧互等（容差 1e-9）。
fn assert_vm_jit_float(src: &str, expected: f64, label: &str) {
    let vm_res = run_vm(src).unwrap_or_else(|e| panic!("[{label}] VM 执行失败: {}", e));
    let jit_res = run_jit(src).unwrap_or_else(|e| panic!("[{label}] JIT 执行失败: {}", e));
    let vm_f = match vm_res { Value::Float(f) => f, v => panic!("[{label}] VM 期望 Float，实际 {:?}", v) };
    let jit_f = match jit_res { Value::Float(f) => f, v => panic!("[{label}] JIT 期望 Float，实际 {:?}", v) };
    assert!((vm_f - expected).abs() < 1e-9, "[{label}] VM = {}, want {}", vm_f, expected);
    assert!((jit_f - expected).abs() < 1e-9, "[{label}] JIT = {}, want {}", jit_f, expected);
    assert!((vm_f - jit_f).abs() < 1e-9, "[{label}] VM/JIT 不一致: {} != {}", vm_f, jit_f);
}

fn err_line(err: &TenthError) -> Option<usize> {
    match err {
        TenthError::RuntimeError { line, .. } => *line,
        other => panic!("期望 RuntimeError，实际: {:?}", other),
    }
}

/// VM 与 JIT 均报错、行号一致，且等于 `expected_line`（跨路径错误行号一致性）。
fn assert_vm_jit_err_line_parity(src: &str, expected_line: usize, label: &str) {
    let vm_res = run_vm(src);
    let jit_res = run_jit(src);
    assert!(vm_res.is_err(), "[{label}] VM 应报错，实际 {:?}", vm_res);
    assert!(jit_res.is_err(), "[{label}] JIT 应报错，实际 {:?}", jit_res);
    let vm_line = err_line(&vm_res.unwrap_err());
    let jit_line = err_line(&jit_res.unwrap_err());
    assert_eq!(vm_line, Some(expected_line), "[{label}] VM 行号 = {:?}, 期望 {expected_line}", vm_line);
    assert_eq!(jit_line, Some(expected_line), "[{label}] JIT 行号 = {:?}, 期望 {expected_line}", jit_line);
    assert_eq!(vm_line, jit_line, "[{label}] VM/JIT 行号不一致: VM={:?} JIT={:?}", vm_line, jit_line);
}

// ═══════════════════════════════════════════════════════════════════════════
// A. 直接调用（A1）——递归/嵌套/多参/字符串返回/循环内调用/深表达式
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn consistency_direct_mutual_recursion() {
    // 互递归 even/odd：is_even(10)=1（10 偶），is_odd(7)=1（7 奇）→ 101
    assert_vm_jit_int(r#"
        fn is_even(n: Int) -> Int {
            if n == 0 { 1 } else { is_odd(n - 1) }
        }
        fn is_odd(n: Int) -> Int {
            if n == 0 { 0 } else { is_even(n - 1) }
        }
        fn main() -> Int { is_even(10) * 100 + is_odd(7) }
    "#, 101, "direct-mutual-recursion");
}

#[test]
fn consistency_direct_multi_param() {
    // 5 参数直接调用（实参编组顺序）→ 12345
    assert_vm_jit_int(r#"
        fn sum5(a: Int, b: Int, c: Int, d: Int, e: Int) -> Int {
            a * 10000 + b * 1000 + c * 100 + d * 10 + e
        }
        fn main() -> Int { sum5(1, 2, 3, 4, 5) }
    "#, 12345, "direct-multi-param");
}

#[test]
fn consistency_direct_string_return() {
    // 函数返回 String + 直接调用（跨 JIT 帧字符串表/值传递）→ "hello world".len() = 11
    assert_vm_jit_int(r#"
        fn greet(name: str) -> str { "hello " + name }
        fn main() -> Int {
            let s = greet("world");
            s.len()
        }
    "#, 11, "direct-string-return");
}

#[test]
fn consistency_direct_call_in_loop() {
    // 循环体内重复直接调用（热调用形态）：format3(i)="v"+to_string(i)，len=1+位数。
    // i 0..50：0-9 一位（10 个）、10-49 两位（40 个）→ 10*2 + 40*3 = 140
    assert_vm_jit_int(r#"
        fn format3(x: Int) -> str { "v" + to_string(x) }
        fn main() -> Int {
            let mut sum = 0;
            let mut i = 0;
            while i < 50 {
                let s = format3(i);
                sum = sum + s.len();
                i = i + 1;
            };
            sum
        }
    "#, 140, "direct-call-in-loop");
}

#[test]
fn consistency_direct_deep_expression() {
    // 深表达式嵌套调用（add 被内联，验证结构正确）→ 36
    assert_vm_jit_int(r#"
        fn add(a: Int, b: Int) -> Int { a + b }
        fn main() -> Int {
            add(add(add(1, 2), add(3, 4)), add(add(5, 6), add(7, 8)))
        }
    "#, 36, "direct-deep-expression");
}

// ═══════════════════════════════════════════════════════════════════════════
// B. 内联（A2）——多参/浮点/循环内/含调用回退/嵌套小函数
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn consistency_inline_multi_param_small() {
    assert_vm_jit_int(r#"
        fn calc(a: Int, b: Int, c: Int) -> Int { a * 100 + b * 10 + c }
        fn main() -> Int { calc(3, 4, 5) }
    "#, 345, "inline-multi-param");
}

#[test]
fn consistency_inline_float_params() {
    assert_vm_jit_float(r#"
        fn scale(x: Float, k: Float) -> Float { x * k }
        fn main() -> Float { scale(2.5, 4.0) }
    "#, 10.0, "inline-float-params");
}

#[test]
fn consistency_inline_in_loop() {
    // inc 在循环体内被内联（内联 + 回边交互）→ 1000
    assert_vm_jit_int(r#"
        fn inc(x: Int) -> Int { x + 1 }
        fn main() -> Int {
            let mut v = 0;
            let mut i = 0;
            while i < 1000 {
                v = inc(v);
                i = i + 1;
            };
            v
        }
    "#, 1000, "inline-in-loop");
}

#[test]
fn consistency_inline_fallback_contains_call() {
    // outer 含调用 → 不内联 → 直接调用；inner 小 → 内联进 outer → 12
    assert_vm_jit_int(r#"
        fn inner(x: Int) -> Int { x + 1 }
        fn outer(x: Int) -> Int { inner(x) * 2 }
        fn main() -> Int { outer(5) }
    "#, 12, "inline-fallback-contains-call");
}

#[test]
fn consistency_inline_nested_small_calls() {
    // b 调用 a（均小）：b 含调用不内联、a 内联进 b；main 内 b/a 混合 → 808
    assert_vm_jit_int(r#"
        fn a(x: Int) -> Int { x + 1 }
        fn b(x: Int) -> Int { a(x) * 2 }
        fn main() -> Int { b(3) * 100 + a(7) }
    "#, 808, "inline-nested-small");
}

// ───────────────────────────────────────────────────────────────────────────
// A2-AUDIT-11.4.35 回归：i64 注解小函数（可内联 + A6 可特化）在循环内被内联。
// 背景：发射端内联（A2）**优先于**特化（A6），内联结果是 Value 而非标量寄存器；
// 但标量分析只建模 spec 路径（结果 I32）。分析预测 I32/spec → 循环回边处分析把
// 该局部重置为 I32，而上一轮一般路径 Store 只写 Value 槽、局部标量槽残留过期值
// → 下一轮 Load 重专用化读过期标量 → 静默 0（溢出检查失效）。修复：分析期把
// 「可内联调用」预测为 Unknown（与发射端内联路径一致）+ 一般 Store 失效 local_scalars。
// 既有测试用 `Int`（无 spec 签名）未覆盖此路径——本组用显式 `i64` 注解触发。
// ───────────────────────────────────────────────────────────────────────────

#[test]
fn consistency_inline_i64_3param_loop_accumulate() {
    // 3 参小函数循环内联累加（i64 注解，触发 spec 分析路径）：s/t 分别累加 →
    // s=5050、t=5150 → 5055150。修复前 JIT 静默 0。
    assert_vm_jit_int(r#"
        fn f3(a: i64, b: i64, c: i64) -> i64 { a + b + c }
        fn main() -> i64 {
            let mut s = 0;
            let mut t = 0;
            let mut i = 0;
            while i < 100 {
                s = f3(s, i, 1);
                t = f3(t, i, 2);
                i = i + 1;
            };
            s * 1000 + t
        }
    "#, 5055150, "inline-i64-3param-loop-accumulate");
}

#[test]
fn consistency_inline_i64_4param_loop_accumulate() {
    // 4 参小函数循环内联累加：add4(s,i,1,2) → s += i+3 → 5250。
    assert_vm_jit_int(r#"
        fn add4(a: i64, b: i64, c: i64, d: i64) -> i64 { a + b + c + d }
        fn main() -> i64 {
            let mut s = 0;
            let mut i = 0;
            while i < 100 {
                s = add4(s, i, 1, 2);
                i = i + 1;
            };
            s
        }
    "#, 5250, "inline-i64-4param-loop-accumulate");
}

#[test]
fn consistency_inline_i64_2param_1param_loop_accumulate() {
    // 2 参/1 参小函数循环内联回归：s += i（4950）、t += 1（100）→ 104950。
    assert_vm_jit_int(r#"
        fn g2(a: i64, b: i64) -> i64 { a + b }
        fn h1(a: i64) -> i64 { a + 1 }
        fn main() -> i64 {
            let mut s = 0;
            let mut t = 0;
            let mut i = 0;
            while i < 100 {
                s = g2(s, i);
                t = h1(t);
                i = i + 1;
            };
            s + t * 1000
        }
    "#, 104950, "inline-i64-2param-1param-loop");
}

#[test]
fn consistency_inline_i64_3param_loop_overflow() {
    // 3 参小函数循环累加至溢出：JIT 与 VM 一致报「溢出 i32 范围」（修复前 JIT
    // 静默 0 无报错——静默错值红线）。行号均指向被调函数体内（第 2 行）。
    assert_vm_jit_err_line_parity(r#"fn f3(a: i64, b: i64, c: i64) -> i64 {
    a + b + c
}
fn main() -> i64 {
    let mut s = 0;
    let mut i = 0;
    while i < 3000000 {
        s = f3(s, i, 1);
        i = i + 1;
    };
    s
}
"#, 2, "err-inline-i64-3param-loop-overflow");
}

#[test]
fn consistency_inline_i64_multi_param_mixed_string_arg() {
    // 混合：≥3 参 + 字符串实参 → 含 PushStr/MethodCall 不内联（A6 特化也不适用，
    // 因非纯 i64）→ 走 A1 直接调用。对照：不内联路径保持正确 → 5050。
    assert_vm_jit_int(r#"
        fn f3mix(a: i64, b: i64, tag: str) -> i64 {
            let n = tag.len();
            a + b + n
        }
        fn main() -> i64 {
            let mut s = 0;
            let mut i = 0;
            while i < 100 {
                s = f3mix(s, i, "v");
                i = i + 1;
            };
            s
        }
    "#, 5050, "inline-i64-multi-param-mixed-string");
}

// ═══════════════════════════════════════════════════════════════════════════
// C. 标量专用化（A2b）——I32 全比较集/F64 混合/Bool 逻辑/溢出/除零/取模/取负
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn consistency_scalar_i32_all_compares() {
    // 六种比较各计数：eq=1 ne=99 lt=50 gt=49 le=51 ge=50 → 11454150
    assert_vm_jit_int(r#"
        fn main() -> Int {
            let mut eq = 0; let mut ne = 0; let mut lt = 0;
            let mut gt = 0; let mut le = 0; let mut ge = 0;
            let mut i = 0;
            while i < 100 {
                if i == 50 { eq = eq + 1 };
                if i != 50 { ne = ne + 1 };
                if i < 50 { lt = lt + 1 };
                if i > 50 { gt = gt + 1 };
                if i <= 50 { le = le + 1 };
                if i >= 50 { ge = ge + 1 };
                i = i + 1;
            };
            eq * 1000000 + ne * 100000 + lt * 10000 + gt * 1000 + le * 100 + ge
        }
    "#, 11454150, "scalar-i32-all-compares");
}

#[test]
fn consistency_scalar_f64_loop() {
    // F64 标量累加（0.5 精确二进制）→ 100.0
    assert_vm_jit_float(r#"
        fn main() -> Float {
            let mut acc = 0.0;
            let mut i = 0;
            while i < 200 {
                acc = acc + 0.5;
                i = i + 1;
            };
            acc
        }
    "#, 100.0, "scalar-f64-loop");
}

#[test]
fn consistency_scalar_bool_logic() {
    // (a||b) && !(b&&a) 恒真 → 100
    assert_vm_jit_int(r#"
        fn main() -> Int {
            let mut a = true;
            let mut b = false;
            let mut count = 0;
            let mut i = 0;
            while i < 100 {
                if (a || b) && !(b && a) { count = count + 1 };
                i = i + 1;
            };
            count
        }
    "#, 100, "scalar-bool-logic");
}

#[test]
fn consistency_scalar_overflow_in_loop() {
    // 循环内 I32 溢出：2147483640 + 45 > i32::MAX → 两侧报错 + 行号一致（第 5 行）
    assert_vm_jit_err_line_parity(r#"fn main() -> Int {
    let mut sum = 2147483640;
    let mut i = 0;
    while i < 10 {
        sum = sum + i;
        i = i + 1;
    };
    sum
}
"#, 5, "scalar-overflow-in-loop");
}

#[test]
fn consistency_scalar_div_zero_in_loop() {
    // 循环内除零（i=5 时 100/(5-i) → /0）：两侧报错 + 行号一致（第 5 行）
    assert_vm_jit_err_line_parity(r#"fn main() -> Int {
    let mut i = 0;
    let mut acc = 0;
    while i < 6 {
        acc = acc + 100 / (5 - i);
        i = i + 1;
    };
    acc
}
"#, 5, "scalar-div-zero-in-loop");
}

#[test]
fn consistency_scalar_neg_overflow() {
    // I32::MIN 取负溢出（-2147483647-1 = MIN；0 - MIN 溢出）→ 两侧报错 + 行号一致（第 3 行）
    assert_vm_jit_err_line_parity(r#"fn main() -> Int {
    let min = -2147483647 - 1;
    let b = 0 - min;
    b
}
"#, 3, "scalar-neg-overflow");
}

#[test]
fn consistency_scalar_mod_zero() {
    // 取模除零 → 两侧报错 + 行号一致（第 3 行）
    assert_vm_jit_err_line_parity(r#"fn main() -> Int {
    let y = 0;
    let z = 10 % y;
    z
}
"#, 3, "scalar-mod-zero");
}

// ═══════════════════════════════════════════════════════════════════════════
// D. Opcode 覆盖（A3）——Try / Tuple / Struct / Spawn / TailCall
// ═══════════════════════════════════════════════════════════════════════════

// ── Try：多层传播 / 链式 ? / 循环内 ? / try 块嵌套 / Ok 消费 ─────────────

#[test]
fn consistency_try_multi_layer_propagate() {
    // 三层嵌套传播：good=111（Ok 链）、bad=2（Err("e1").len()）→ 11102
    assert_vm_jit_int(r#"
        fn l1(ok: Bool) -> Result<Int, str> {
            if ok { Result::Ok(1) } else { Result::Err("e1") }
        }
        fn l2(ok: Bool) -> Result<Int, str> {
            let v = l1(ok)?;
            Result::Ok(v + 10)
        }
        fn l3(ok: Bool) -> Result<Int, str> {
            let v = l2(ok)?;
            Result::Ok(v + 100)
        }
        fn main() -> Int {
            let good = match l3(true) { Result::Ok(v) => v, _ => -1 };
            let bad = match l3(false) { Result::Err(m) => m.len(), _ => -2 };
            good * 100 + bad
        }
    "#, 11102, "try-multi-layer");
}

#[test]
fn consistency_try_chained_in_one_fn() {
    // 同一函数内连续两个 ?，第二个遇 Err → 早退返回 Result::Err("bad")（Debug 一致）
    assert_vm_jit_debug(r#"
        fn parse(s: str) -> Result<Int, str> {
            if s == "42" { Result::Ok(42) } else { Result::Err("bad") }
        }
        fn main() -> Int {
            let a = parse("42")?;
            let b = parse("99")?;
            a + b
        }
    "#, "try-chained");
}

#[test]
fn consistency_try_err_in_loop() {
    // 循环内 ? 遇 Err → 早退返回 Result::Err("hit3")（Debug 一致）
    assert_vm_jit_debug(r#"
        fn may_fail(i: Int) -> Result<Int, str> {
            if i == 3 { Result::Err("hit3") } else { Result::Ok(i * 10) }
        }
        fn main() -> Result<Int, str> {
            let mut i = 0;
            let mut acc = 0;
            while i < 5 {
                let v = may_fail(i)?;
                acc = acc + v;
                i = i + 1;
            };
            Result::Ok(acc)
        }
    "#, "try-err-in-loop");
}

#[test]
fn consistency_try_ok_in_loop() {
    // 循环内 ? 全成功 → Result::Ok(0+10+20=30)（Debug 一致）
    assert_vm_jit_debug(r#"
        fn may_fail(i: Int) -> Result<Int, str> {
            if i == 3 { Result::Err("hit3") } else { Result::Ok(i * 10) }
        }
        fn main() -> Result<Int, str> {
            let mut i = 0;
            let mut acc = 0;
            while i < 3 {
                let v = may_fail(i)?;
                acc = acc + v;
                i = i + 1;
            };
            Result::Ok(acc)
        }
    "#, "try-ok-in-loop");
}

#[test]
fn consistency_try_nested_try_block() {
    // try 块内 ?：成功包装 Ok、失败被 try 块捕获（不早退）→ v1=7, v2=-2 → 698
    assert_vm_jit_int(r#"
        fn parse(s: str) -> Result<Int, str> {
            if s == "7" { Result::Ok(7) } else { Result::Err("nope") }
        }
        fn main() -> Int {
            let r1 = try { parse("7")? };
            let r2 = try { parse("9")? };
            let v1 = match r1 { Result::Ok(v) => v, _ => -1 };
            let v2 = match r2 { Result::Ok(v) => v, _ => -2 };
            v1 * 100 + v2
        }
    "#, 698, "try-nested-try-block");
}

#[test]
fn consistency_try_ok_value_consumed() {
    // Ok 解包值被后续算术消费 → 42042
    assert_vm_jit_int(r#"
        fn parse(s: str) -> Result<Int, str> {
            if s == "42" { Result::Ok(42) } else { Result::Err("not 42") }
        }
        fn main() -> Int {
            let a = parse("42")?;
            let b = parse("42")?;
            a * 1000 + b
        }
    "#, 42042, "try-ok-consumed");
}

// ── Tuple：深层嵌套 / OOB / match guard / 跨函数返回 ─────────────────────

#[test]
fn consistency_tuple_nested_three_levels() {
    assert_vm_jit_int(r#"
        fn main() -> Int {
            let t = (1, (2, (3, 4)));
            let (a, inner1) = t;
            let (b, inner2) = inner1;
            let (c, d) = inner2;
            a * 1000 + b * 100 + c * 10 + d
        }
    "#, 1234, "tuple-nested-3");
}

#[test]
fn consistency_tuple_oob_nested() {
    // 嵌套 tuple 解构（扁平两步）+ 内层越界（c 越界 → Unit）→ 返回值 (1, 2)（Debug 一致）
    assert_vm_jit_debug(r#"
        fn main() -> (Int, Int) {
            let t = (1, (2, 3));
            let (a, inner) = t;
            let (b, c, d) = inner;
            (a, b)
        }
    "#, "tuple-oob-nested");
}

#[test]
fn consistency_tuple_match_single_arm_guard() {
    // 单 guard 臂 + wildcard（guard 不命中 → wildcard）→ -1
    // 注：guard 后接**第二个 tuple 臂**的形态存在预存 VM 缺陷（回退直接落 wildcard），
    // 已登记 AUDIT——本用例只用单 guard 臂 + wildcard 的稳定形态。
    assert_vm_jit_int(r#"
        fn main() -> Int {
            let t = (2, 3);
            match t {
                (a, b) if a > b => a * 100 + b,
                _ => -1,
            }
        }
    "#, -1, "tuple-guard-single-wildcard");
}

#[test]
fn consistency_tuple_return_from_fn_nested() {
    // 函数返回嵌套 tuple + 调用方扁平两步解构 → 123
    assert_vm_jit_int(r#"
        fn make() -> (Int, (Int, Int)) { (1, (2, 3)) }
        fn main() -> Int {
            let (a, inner) = make();
            let (b, c) = inner;
            a * 100 + b * 10 + c
        }
    "#, 123, "tuple-fn-return-nested");
}

// ── Struct：字段链 / 循环写 / tuple 字段 / 跨函数 match ──────────────────

#[test]
fn consistency_struct_nested_field_chain() {
    assert_vm_jit_int(r#"
        struct Inner { z: Int }
        struct Outer { x: Int, inner: Inner }
        fn main() -> Int {
            let o = Outer { x: 3, inner: Inner { z: 4 } };
            o.inner.z * 100 + o.x
        }
    "#, 403, "struct-field-chain");
}

#[test]
fn consistency_struct_field_mutation_loop() {
    // 循环内字段自增（StoreField 回写 + 循环）→ 100
    assert_vm_jit_int(r#"
        struct Counter { v: Int }
        fn main() -> Int {
            let mut c = Counter { v: 0 };
            let mut i = 0;
            while i < 100 {
                c.v = c.v + 1;
                i = i + 1;
            };
            c.v
        }
    "#, 100, "struct-field-mutation-loop");
}

#[test]
fn consistency_struct_with_tuple_field() {
    // struct 含 tuple 字段 + 解构 → 304
    assert_vm_jit_int(r#"
        struct P { name: str, coords: (Int, Int) }
        fn main() -> Int {
            let p = P { name: "pt", coords: (3, 4) };
            let (x, y) = p.coords;
            x * 100 + y
        }
    "#, 304, "struct-tuple-field");
}

#[test]
fn consistency_struct_cross_fn_match() {
    // struct 值跨函数传递 + match（IsStruct）→ 7*100 + 30 = 730
    assert_vm_jit_int(r#"
        struct Point { x: Int, y: Int }
        fn sum_pt(p: Point) -> Int {
            match p {
                Point { x, y } => x + y,
                _ => -1,
            }
        }
        fn main() -> Int {
            let a = sum_pt(Point { x: 3, y: 4 });
            let b = sum_pt(Point { x: 10, y: 20 });
            a * 100 + b
        }
    "#, 730, "struct-cross-fn-match");
}

// ── Spawn：eager Future（VM=JIT；解释器不支持 async，只对拍 VM/JIT）──────

#[test]
fn consistency_spawn_eager_literal() {
    assert_vm_jit_debug(r#"
        fn main() {
            let f = spawn 42;
            f
        }
    "#, "spawn-eager-literal");
}

#[test]
fn consistency_spawn_eager_expression() {
    // spawn 函数调用（eager 求值）→ Future<10>（Debug 一致）
    assert_vm_jit_debug(r#"
        fn double(x: Int) -> Int { x * 2 }
        fn main() {
            let f = spawn double(5);
            f
        }
    "#, "spawn-eager-expr");
}

#[test]
fn consistency_spawn_multi_future_tuple() {
    // 多个 spawn 组合为 tuple → Debug 一致
    assert_vm_jit_debug(r#"
        fn main() {
            let a = spawn 1;
            let b = spawn 2;
            (a, b)
        }
    "#, "spawn-multi-tuple");
}

// ── TailCall / TailCallClosure（D2 保守：host_call + 立即返回）───────────

#[test]
fn consistency_tailcall_chain() {
    // 尾调用链 a→b→c → 42
    assert_vm_jit_int(r#"
        fn a(x: Int) -> Int { b(x) }
        fn b(x: Int) -> Int { c(x) }
        fn c(x: Int) -> Int { x * 2 }
        fn main() -> Int { a(21) }
    "#, 42, "tailcall-chain");
}

#[test]
fn consistency_tailcall_recursive_deep() {
    // 深尾递归 sum_tail(1000, 0) = 500500（VM 帧堆式增长，深递归安全）
    assert_vm_jit_int(r#"
        fn sum_tail(n: Int, acc: Int) -> Int {
            if n <= 0 { acc } else { sum_tail(n - 1, acc + n) }
        }
        fn main() -> Int { sum_tail(1000, 0) }
    "#, 500500, "tailcall-recursive-deep");
}

#[test]
fn consistency_tailcall_in_loop() {
    // 循环内直接调用 classify（其 else 分支为 TailCall 到 neg）：奇数为负 → -5
    assert_vm_jit_int(r#"
        fn neg(x: Int) -> Int { 0 - x }
        fn classify(n: Int) -> Int {
            if n % 2 == 0 { n } else { neg(n) }
        }
        fn main() -> Int {
            let mut acc = 0;
            let mut i = 0;
            while i < 10 {
                acc = acc + classify(i);
                i = i + 1;
            };
            acc
        }
    "#, -5, "tailcall-in-loop");
}

#[test]
fn consistency_tailcall_closure_multi() {
    // 闭包尾调用（TailCallClosure）多次 → 42*100 + 2 = 4202
    assert_vm_jit_int(r#"
        fn apply(n: Int) -> Int {
            let dbl = |x: Int| x * 2;
            dbl(n)
        }
        fn main() -> Int { apply(21) * 100 + apply(1) }
    "#, 4202, "tailcall-closure-multi");
}

// ═══════════════════════════════════════════════════════════════════════════
// E. 跨路径错误行号一致性（VM line == JIT line）
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn consistency_err_line_div_zero_in_fn() {
    // compute 被内联（4 指令），除零在函数体内 → 两侧均第 2 行
    assert_vm_jit_err_line_parity(r#"fn compute(a: Int, b: Int) -> Int {
    let r = a / b;
    r * 2
}
fn main() -> Int {
    let x = compute(10, 0);
    x
}
"#, 2, "err-div-zero-in-fn");
}

#[test]
fn consistency_err_line_tensor_method_in_fn() {
    // reshape 元素数不匹配（含 MethodCall 不内联）→ 两侧均第 2 行
    assert_vm_jit_err_line_parity(r#"fn reshape_it(a: Tensor, n: Int, m: Int) -> Tensor {
    a.reshape(n, m)
}
fn main() {
    let a = ones(2, 3);
    let r = reshape_it(a, 4, 5);
    print(r);
}
"#, 2, "err-tensor-method-in-fn");
}

#[test]
fn consistency_err_line_index_oob_in_fn() {
    // 张量索引越界（IndexGet 不内联）→ 两侧均第 2 行
    assert_vm_jit_err_line_parity(r#"fn get(t: Tensor, i: Int) -> Float {
    t[i]
}
fn main() {
    let a = ones(2, 3);
    let v = get(a, 5);
    print(v);
}
"#, 2, "err-index-oob-in-fn");
}

#[test]
fn consistency_err_line_scalar_overflow_in_fn() {
    // 小函数内标量溢出（add_big 被内联）→ 两侧均第 2 行
    assert_vm_jit_err_line_parity(r#"fn add_big(a: Int, b: Int) -> Int {
    let c = a + b;
    c
}
fn main() -> Int {
    let x = add_big(2000000000, 2000000000);
    x
}
"#, 2, "err-scalar-overflow-in-fn");
}

// ═══════════════════════════════════════════════════════════════════════════
// F. 热函数覆盖验证（分层分析守护：被调函数按需 ensure 已编译）
// ═══════════════════════════════════════════════════════════════════════════
//
// 分层分析结论（M2-A5）：默认 JIT 路径下，main 编译后其**直接调用可达**的
// 用户函数在首次调用即被编译（host_jit_call 慢路径 → jit_call_chunk →
// get_or_compile → 注册进函数指针表），后续调用走快路径。这等价于「编译阈值
// = 1」——比「调用 N 次后再编译」更强，无需调用计数。
// 本用例把该不变量固化为测试：热函数（被 main 循环反复调用的函数）必须
// 出现在 JIT 编译表（is_compiled 且非 is_failed）。

#[test]
fn consistency_hot_functions_compiled_on_first_call() {
    // main 循环调用 hot2（hot2 调用 hot1）。hot1 含 PushStr → **不可内联**，
    // 必须以 chunk 形式编译（经 hot2 直接调用）；hot2 含调用不内联 → main 对 hot2
    // 为直接调用 → 首次调用编译。期望值 10100。
    // P3 后 `Int` 已纳入特化（热函数走 spec 入口）——本用例守护**通用入口**的
    // 按需编译，故用 `i32` 注解（无 scalar_sig → 通用路径）。
    let src = r#"
        fn hot1(x: i32) -> i32 { "n".len() + x }
        fn hot2(x: i32) -> i32 { hot1(x) * 2 }
        fn main() -> i32 {
            let mut acc = 0;
            let mut i = 0;
            while i < 100 {
                acc = acc + hot2(i);
                i = i + 1;
            };
            acc
        }
    "#;
    let (result, vm) = run_jit_with_vm(src).unwrap();
    assert_eq!(int_of(result, "hot-compiled"), 10100, "sum(2*(i+1), 0..100) = 10100");

    let ctx = vm.jit_ctx.as_ref().expect("JIT 上下文应存在");
    let main_idx = vm.chunk_index_of("main").expect("main chunk 应存在");
    let hot2_idx = vm.chunk_index_of("hot2").expect("hot2 chunk 应存在");
    let hot1_idx = vm.chunk_index_of("hot1").expect("hot1 chunk 应存在");
    assert!(ctx.is_compiled(main_idx), "main 应被 JIT 编译");
    assert!(ctx.is_compiled(hot2_idx), "热函数 hot2（直接调用可达）应被按需编译");
    assert!(ctx.is_compiled(hot1_idx), "热函数 hot1（不可内联，经 hot2 调用）应被按需编译");
    assert!(!ctx.is_failed(main_idx) && !ctx.is_failed(hot2_idx) && !ctx.is_failed(hot1_idx),
        "热函数不应整函数 fallback");
}
