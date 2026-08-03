//! M2-A5 / D4（已拍板）：性能基准固化进 CI 门槛。
//!
//! `perf_bench3.th` 的 fib(28)/loop 1e7/matmul150 固化进自动化测试：
//! - **默认 `#[ignore]`**：普通 `cargo test` 不跑（不拖慢）；显式
//!   `cargo test --release --test bench_gate_test -- --ignored` 才执行。
//! - **非 release 跳过**：`cfg!(debug_assertions)` 下直接返回（宽松处理环境差异，
//!   基准只在 release 有意义）。
//! - **阈值带裕量**（防 CI 抖动误报）：阈值 = 本机实测中位数的 3× 左右。
//!
//! 基准方法（与 `.trae/tmp/perf_bench3.th` 一致）：release 构建，JIT 默认路径
//! （不设 TENTH_NO_VM），`time_now_ms` 在 main 内部计时（JIT 编译发生在计时
//! 之前，计时不含编译开销），取 3 次中位数。基准源码内嵌于本文件（镜像
//! perf_bench3.th，输出改为 `FIB=<ms>` / `LOOP=<ms>` / `MATMUL=<ms>` 便于断言）。

use tenth::compile::bytecode::BytecodeCompiler;
use tenth::compile::jit;
use tenth::hir::lower::Lowerer;
use tenth::lexer::lexer::Lexer;
use tenth::parser::parser::Parser;
use tenth::runtime::natives::register_all_natives;
use tenth::runtime::value::Value;
use tenth::runtime::vm::Vm;

// ── 阈值（本机实测中位数 + ~3× 裕量）──────────────────────────────────────
// 实测（2026-08-03，release）：fib(28) = 26/28/30ms（中位数 28ms）；
// loop 1e7 = 66/72/60ms（中位数 66ms）；matmul150 = 0-1ms。
// M2 目标：fib <10ms / loop <100ms。CI 门槛留裕量防抖动：
const FIB_THRESHOLD_MS: u64 = 100;   // 28ms × 3.5
const LOOP_THRESHOLD_MS: u64 = 200;  // 66ms × 3.0
const MATMUL_THRESHOLD_MS: u64 = 20; // 1ms × 20（张量在 Rust 层，裕量最大）

/// 基准源码（镜像 `.trae/tmp/perf_bench3.th`；输出改为 FIB=/LOOP=/MATMUL=）。
const BENCH_SRC: &str = r#"
fn fib(n: Int) -> Int {
    if n < 2 { n } else { fib(n - 1) + fib(n - 2) }
}

fn main() {
    let t0 = time_now_ms();
    let f = fib(28);
    let t1 = time_now_ms();
    println("FIB=" + to_string(t1 - t0));

    let t2 = time_now_ms();
    let mut sum = 0;
    let mut i = 0;
    while i < 10000000 {
        sum = sum + (i % 7);
        i = i + 1;
    }
    let t3 = time_now_ms();
    println("LOOP=" + to_string(t3 - t2));

    let a = randn(150, 150);
    let b = randn(150, 150);
    let t4 = time_now_ms();
    let c = a.matmul(b);
    let t5 = time_now_ms();
    let s = c.sum();
    println("MATMUL=" + to_string(t5 - t4));
}
"#;

#[derive(Debug, Default, Clone, Copy)]
struct BenchSamples {
    fib_ms: u64,
    loop_ms: u64,
    matmul_ms: u64,
}

// println 输出捕获缓冲（NativeFn 是 `fn` 指针，不能捕获闭包环境，用 thread_local）。
thread_local! {
    static CAPTURED: std::cell::RefCell<Vec<String>> = const { std::cell::RefCell::new(Vec::new()) };
}

/// 编译基准源码并在 JIT 路径执行一次，捕获 `FIB=`/`LOOP=`/`MATMUL=` 计时行。
fn run_bench_once() -> Result<BenchSamples, String> {
    CAPTURED.with(|c| c.borrow_mut().clear());

    let mut lexer = Lexer::new(BENCH_SRC);
    let tokens = lexer.tokenize().map_err(|e| e.to_string())?;
    let mut parser = Parser::new(tokens);
    let program = parser.parse_program().map_err(|e| e.to_string())?;
    let mut lowerer = Lowerer::new();
    let hir = lowerer.lower_program(&program).map_err(|e| e.to_string())?;

    let mut vm = Vm::new();
    register_all_natives(&mut vm);
    // 覆盖 println：捕获输出行（避免污染测试 stdout，且便于解析计时）。
    vm.add_native("println".into(), |_vm, args| {
        let mut s = String::new();
        for a in args { s.push_str(&format!("{a}")); }
        CAPTURED.with(|c| c.borrow_mut().push(s));
        Ok(Value::Unit)
    });

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
        jit::run_jit(&mut vm, "main").map_err(|e| e.to_string())?;
    } else if vm.has_fn("main") {
        jit::run_jit(&mut vm, "main").map_err(|e| e.to_string())?;
    }

    // 解析计时行
    let mut samples = BenchSamples::default();
    let captured: Vec<String> = CAPTURED.with(|c| c.borrow().clone());
    for line in captured.iter() {
        // time_now_ms 返回 Float（如 "30.0"）——按 f64 解析后截断为 ms。
        if let Some(rest) = line.strip_prefix("FIB=") {
            samples.fib_ms = rest.trim().parse::<f64>().map_err(|e| format!("FIB 解析失败: {line} ({e})"))? as u64;
        } else if let Some(rest) = line.strip_prefix("LOOP=") {
            samples.loop_ms = rest.trim().parse::<f64>().map_err(|e| format!("LOOP 解析失败: {line} ({e})"))? as u64;
        } else if let Some(rest) = line.strip_prefix("MATMUL=") {
            samples.matmul_ms = rest.trim().parse::<f64>().map_err(|e| format!("MATMUL 解析失败: {line} ({e})"))? as u64;
        }
    }
    if samples.fib_ms == 0 || samples.loop_ms == 0 {
        return Err(format!("未捕获完整计时（FIB/LOOP 缺失），捕获行: {:?}", captured));
    }
    Ok(samples)
}

/// 跑 3 次取中位数（防单次抖动）。
fn run_bench_median() -> Result<BenchSamples, String> {
    let mut v: Vec<BenchSamples> = Vec::new();
    for _ in 0..3 {
        v.push(run_bench_once()?);
    }
    v.sort_by_key(|s| s.fib_ms);
    let mut fibs: Vec<u64> = v.iter().map(|s| s.fib_ms).collect();
    fibs.sort_unstable();
    let mut loops: Vec<u64> = v.iter().map(|s| s.loop_ms).collect();
    loops.sort_unstable();
    let mut mats: Vec<u64> = v.iter().map(|s| s.matmul_ms).collect();
    mats.sort_unstable();
    Ok(BenchSamples {
        fib_ms: fibs[1],
        loop_ms: loops[1],
        matmul_ms: mats[1],
    })
}

/// 基准门槛主测试：fib/loop/matmul 均须低于阈值。
/// 普通 `cargo test` 跳过（`#[ignore]`）；显式 `--ignored` + release 才执行。
#[test]
#[ignore = "性能基准门槛：普通 cargo test 跳过；cargo test --release --test bench_gate_test -- --ignored 运行"]
fn bench_gate_fib_loop_matmul() {
    // 非 release 宽松处理：直接通过（基准只在 release 有意义）。
    if cfg!(debug_assertions) {
        eprintln!("bench_gate: 非 release 构建，跳过基准门槛（仅 release 生效）");
        return;
    }
    let samples = run_bench_median().unwrap_or_else(|e| panic!("基准运行失败: {e}"));
    eprintln!(
        "bench_gate 中位数: fib(28) = {} ms / loop 1e7 = {} ms / matmul150 = {} ms",
        samples.fib_ms, samples.loop_ms, samples.matmul_ms
    );
    assert!(
        samples.fib_ms < FIB_THRESHOLD_MS,
        "fib(28) 基准超时: {} ms >= 阈值 {} ms（M2 目标 <10ms，CI 门槛留 3× 裕量）",
        samples.fib_ms, FIB_THRESHOLD_MS
    );
    assert!(
        samples.loop_ms < LOOP_THRESHOLD_MS,
        "loop 1e7 基准超时: {} ms >= 阈值 {} ms（M2 目标 <100ms，CI 门槛留 3× 裕量）",
        samples.loop_ms, LOOP_THRESHOLD_MS
    );
    assert!(
        samples.matmul_ms < MATMUL_THRESHOLD_MS,
        "matmul150 基准超时: {} ms >= 阈值 {} ms",
        samples.matmul_ms, MATMUL_THRESHOLD_MS
    );
}

/// 基准报告工具：手动跑 `-- --ignored bench_report` 打印 3 次中位数数据。
#[test]
#[ignore = "基准报告工具：手动执行，非 CI 门槛"]
fn bench_report() {
    if cfg!(debug_assertions) {
        eprintln!("bench_report: 非 release 构建，数据无意义（仅 release 生效）");
        return;
    }
    for i in 0..3 {
        let s = run_bench_once().unwrap_or_else(|e| panic!("基准运行失败: {e}"));
        eprintln!("run {}: fib(28) = {} ms / loop 1e7 = {} ms / matmul150 = {} ms",
            i + 1, s.fib_ms, s.loop_ms, s.matmul_ms);
    }
    let med = run_bench_median().unwrap_or_else(|e| panic!("基准运行失败: {e}"));
    eprintln!("median: fib(28) = {} ms / loop 1e7 = {} ms / matmul150 = {} ms",
        med.fib_ms, med.loop_ms, med.matmul_ms);
}
