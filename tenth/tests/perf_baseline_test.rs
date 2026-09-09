//! 性能基线巡检（G1–G6）——基准实现 + 结构化输出。
//!
//! 任务：性能基线巡检（档 B），权威契约见 `.agents/tmp/task_board.md`「总师契约 v1」。
//!
//! ## 为什么本文件全部 `#[test] #[ignore]`
//!
//! 工作规范 §七要求 `#[ignore]` 记录原因（MEMO 条目由文档部统一写，本处记录文件内理由）：
//!
//! 1. **不是正确性门槛，而是性能基线**：本文件的断言只有「结果与 VM 路径对拍一致」，
//!    不做任何阈值断言（硬阈值在 `bench_gate_test.rs`，本文件不重复也不降低标准）。
//! 2. **耗时以分钟计**：解释器路径的 `fib28`/`loop 1e7` 单次即 ~4s / ~9s，
//!    每场景 warm-up 1 + 测量 5 次，整份基线约 5 分钟；放进普通 `cargo test` 会拖慢
//!    每一次开发循环，且数据在共享 CI runner 上抖动无意义。
//! 3. **机器相关**：基线的价值是「同机可比」，普通 `cargo test` 在任意机器上跑毫无意义。
//! 4. **`jit_compile_share`/`cold_start_hello` 需要子进程 + 临时 `.th`**，属于基准专用行为。
//!
//! 运行方式（release 才有意义，debug 下直接跳过）。**规范跑法 = 每个测试一个独立进程**：
//! ```text
//! for t in perf_g1_scalar_controlflow perf_g2_tensor_ops perf_g3_autodiff \
//!          perf_g4_nn_optim perf_g5_compile_startup perf_g6_three_path_compare \
//!          perf_vm_reference perf_interp_reference; do
//!   cargo test --release --test perf_baseline_test -- --ignored --nocapture "$t"
//! done
//! ```
//! 为什么必须一进程一组：同进程内前一组的堆状态会把后续张量算子膨胀 2–3×
//!（实测 `matmul_512`：单独进程 5.6–6.5ms → G1 之后 15–20ms）；VM 长循环（loop ≈3–5s）
//! 也会膨胀后续 jit。单进程全量跑（`-- --ignored --test-threads=1`）仍可运行、`PERF_LOCK`
//! 保证串行，但 jit 绝对值偏保守，只适合看比值。
//!
//! ## 计时方法论（总师契约 v1 + 裁定 #10）
//!
//! - 计时一律 Rust 侧 `std::time::Instant`（ns 精度），**不用** Tenth 的 `time_now_ms`（ms 截断）。
//! - 每场景 **warm-up 1 次**（排除 JIT 编译 / 首调用缓存冷启）→ **测量 5 次** → 报 `min` 与 `median`。
//! - 张量短任务在 Tenth 源码内**内部重复 R 次**（见各场景 `repeats`），报单次耗时 = 总耗时 / R。
//! - **进程隔离**（总师裁定 #10① 的落地方式）：规范跑法一进程一组；各组 jit 测试内**不跑 VM**
//!   （VM 由 `perf_vm_reference` 独立进程测），避免 VM 长循环与同进程堆状态膨胀 jit 2–3×。
//! - **interp 隔离**（总师裁定 #10②）：解释器路径（≈4–9s/次，满核负载）移出主基线，
//!   由独立 `perf_interp_reference` 测量；数据保留但**标注「参考路径、仅比值可用」**。
//! - **频率预热**（总师裁定 #10③）：每组开始前 ~400ms 纯 CPU 忙循环把核心拉到稳态频率。
//!   实测：空闲 3s 后测 loop 1e7 = 157–255ms；不空闲（刚编译完）= 70ms（与 `bench_gate`
//!   内部计时 62–69ms 一致）——空闲使核心落入低频/深 C-state，短基准整段跑在升频过程中。
//!   故**绝对值为热态测量，`min` 才是可比统计量，跨次/跨机不可直接比**
//!   （发 `PERF-NOTE|...|methodology|...` 行）。
//! - **正确性**：每条路径内部做自洽性检查（warm-up + 5 次测量的校验和必须一致）；
//!   `perf_vm_reference` 以 VM 为参考对拍 JIT、`perf_interp_reference` 对拍 VM，
//!   不一致立即 fail——禁止「测了个错的」（三路径同源 Rust 实现，理论逐位一致，容差 1e-6）。
//! - 路径标识：`vm`（`Vm::call`）/ `jit`（`jit::run_jit`，默认路径）/ `interp`（`Interpreter`）/
//!   `cli`（G5 子进程端到端 `tenth.exe run <file>`，含进程启动 + 编译 + 执行）。
//! - **防静默回退**（总师裁定 #6②）：每个 `Path::Jit` 场景在 warm-up 后断言
//!   `vm.jit_ctx.is_compiled(bench_idx)`（判据照抄 `tenth/tests/jit_silent_audit_test.rs:138-144`）——
//!   否则 `run_jit` 编译失败会静默 `vm.call`，而 PERF 行仍标 `jit`。
//! - **JIT 混合体披露**（总师裁定 #6③）：G3 `transformer_block_fwd_bwd` / G4 `mha_fwd_bwd` 的
//!   callee `mha_block` 在 `new_grad()` recording 期经 JIT 安全门回退 VM，`jit` 列为
//!   「JIT 外壳 + VM 内层」——数据保留，另发 `PERF-NOTE|...` 披露（不谎称纯 JIT、也不丢数据）。
//! - 口径说明统一走 `PERF-NOTE|` 行（不污染 7 字段 `PERF|` 行）。
//! - 某场景在某路径不可用 → 输出 `n/a` + `PERF-NA` 原因行，**不静默跳过**。
//! - 输出行：`PERF|<group>|<scenario>|<path>|<metric>|<value>|<unit>`。
//!
//! ## 与 `bench_gate_test.rs` 的关系
//!
//! `bench_gate_test.rs` 保留为**硬回归门槛**（fib28<100ms / loop 1e7<200ms / matmul150<20ms），
//! 阈值不改、只增不删。本文件是它的**数据补充**：覆盖 6 组矩阵、报 min/median、给出
//! GFLOPS 与三路径比值，但不设阈值（CI 仅报告，见 `.github/workflows/perf.yml`）。

use std::collections::HashSet;
use std::path::{Path as FsPath, PathBuf};
use std::process::Command;
use std::time::{Duration, Instant};

use tenth::compile::bytecode::BytecodeCompiler;
use tenth::compile::jit;
use tenth::hir::hir::HirProgram;
use tenth::hir::lower::Lowerer;
use tenth::lexer::lexer::Lexer;
use tenth::parser::parser::Parser;
use tenth::runtime::interpreter::Interpreter;
use tenth::runtime::natives::register_all_natives;
use tenth::runtime::value::Value;
use tenth::runtime::vm::Vm;

// ── 方法论常量 ─────────────────────────────────────────────────────────────

/// warm-up 次数（总师契约 v1：1 次）。
const WARMUP_RUNS: usize = 1;
/// 测量次数（总师契约 v1：5 次，取 min / median）。
const MEASURE_RUNS: usize = 5;
/// 正确性对拍相对容差（VM 为参考；三路径张量算子走同一 Rust 实现，理论逐位一致）。
const CHECKSUM_RTOL: f64 = 1e-6;

/// 被测二进制（同 crate bin target；G5 子进程场景用）。
const TENTH_EXE: &str = env!("CARGO_BIN_EXE_tenth");
/// 包根（`tenth/`），子进程 cwd 与临时 `.th` 目录基准。
const TENTH_DIR: &str = env!("CARGO_MANIFEST_DIR");

/// 基准串行锁：cargo 默认并行跑测试函数，多个基准并发会互相争 CPU 使计时失真。
/// 每个测试函数全程持有该锁 → 无论 `--test-threads` 取何值都串行执行。
static PERF_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn serial_guard() -> std::sync::MutexGuard<'static, ()> {
    PERF_LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

// ── 路径 / 场景描述 ────────────────────────────────────────────────────────

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Path {
    Vm,
    Jit,
    Interp,
    /// 子进程端到端（`tenth.exe run <file>`，含进程启动 + 编译 + 执行）。
    /// 仅用于 G5 的 `emit` 路径标签（契约矩阵 G5 路径 = `cli`），不是进程内 runner。
    Cli,
}

impl Path {
    fn tag(self) -> &'static str {
        match self {
            Path::Vm => "vm",
            Path::Jit => "jit",
            Path::Interp => "interp",
            Path::Cli => "cli",
        }
    }
}

/// 主基线测量路径（JIT 先、VM 后；interp 已移至 `perf_interp_reference`，总师裁定 #10②）。
const VM_JIT: [Path; 2] = [Path::Vm, Path::Jit];

/// 每测试组开始前的**频率预热**（不是空闲冷却）。
///
/// 实测：3s 空闲后测 loop 1e7 → 157–255ms；不空闲（刚编译完）→ 70ms（与 `bench_gate`
/// 的内部计时 62–69ms 一致）。原因是空闲让核心落入低频/深 C-state，短基准整段跑在升频
/// 过程中，系统性偏慢。故改为 ~400ms 纯 CPU 忙循环把核心拉到稳态频率——测的是稳态性能，
/// 不是冷时钟。
const FREQ_WARMUP: Duration = Duration::from_millis(400);

fn cpu_freq_warmup() {
    let deadline = Instant::now() + FREQ_WARMUP;
    let mut x: u64 = 0x9E37_79B9_7F4A_7C15;
    while Instant::now() < deadline {
        for _ in 0..20_000 {
            x = x
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
        }
        std::hint::black_box(x);
    }
    std::hint::black_box(x);
}

/// 热态/可比性口径（总师裁定 #10③）：绝对值为热态测量，min 才是可比统计量。
fn emit_methodology_note(group: &str) {
    emit_note(
        group,
        "methodology",
        "all",
        "绝对值为热态测量（每组开始前 ~400ms CPU 频率预热）；min 为可比统计量；跨次/跨机不可直接比",
    );
}

/// 一个基准场景。
struct Scenario {
    group: &'static str,
    name: &'static str,
    src: &'static str,
    paths: &'static [Path],
    /// Tenth 源码内部重复次数 R（必须与源码内 `while r < R` 一致）；报单次 = 总耗时 / R。
    repeats: u32,
    /// 某路径不可用时的原因（`n/a`）；目前只有解释器无 TCO。
    interp_skip: Option<&'static str>,
}

#[derive(Clone, Copy)]
struct Stats {
    min_ms: f64,
    median_ms: f64,
}

// ── 输出 ───────────────────────────────────────────────────────────────────

fn emit(group: &str, scenario: &str, path: Path, metric: &str, value: f64, unit: &str) {
    println!("PERF|{group}|{scenario}|{}|{metric}|{value:.6}|{unit}", path.tag());
}

fn emit_na(group: &str, scenario: &str, path: Path, metric: &str, unit: &str, reason: &str) {
    println!(
        "PERF|{group}|{scenario}|{}|{metric}|n/a|{unit}",
        path.tag()
    );
    println!(
        "PERF-NA|{group}|{scenario}|{}|{reason}",
        path.tag()
    );
}

/// 披露/口径说明行（保持 7 字段 `PERF|` 行不被污染，单独前缀）。
fn emit_note(group: &str, scenario: &str, path_tag: &str, text: &str) {
    println!("PERF-NOTE|{group}|{scenario}|{path_tag}|{text}");
}

/// JIT 混合体披露（总师裁定 #6③）：callee 在 recording 期回退 VM——数据保留、必须披露。
fn jit_note_for(scenario: &str) -> Option<&'static str> {
    match scenario {
        "transformer_block_fwd_bwd" | "mha_fwd_bwd" => Some(
            "callee mha_block 在 new_grad() recording 期经 JIT 安全门（vm/mod.rs:368-370）回退 VM；\
             jit 列 = JIT 外壳（bench 已 is_compiled）+ VM 内层，非纯 JIT",
        ),
        _ => None,
    }
}

/// 需要「计时区间之外」先跑的 setup 函数（按场景名）。
///
/// 目前只有 `adamw_step_1e6`：契约场景名是纯优化器 step，梯度准备（new_grad/param/backward/grad）
/// 必须排除在计时之外，否则会把 fwd+bwd 算进「adamw step」。
fn setup_fn_for(scenario: &str) -> Option<&'static str> {
    match scenario {
        "adamw_step_1e6" => Some("setup"),
        _ => None,
    }
}

// ── 编译 / 执行 ────────────────────────────────────────────────────────────

fn lower(src: &str) -> Result<HirProgram, String> {
    let mut lexer = Lexer::new(src);
    let tokens = lexer.tokenize().map_err(|e| format!("lexer: {e}"))?;
    let mut parser = Parser::new(tokens);
    let program = parser.parse_program().map_err(|e| format!("parser: {e}"))?;
    let mut lowerer = Lowerer::new();
    lowerer
        .lower_program(&program)
        .map_err(|e| format!("lower: {e}"))
}

/// 构建 VM（注册全部 native + 编译全部函数 + 顶层全局初始化）。
fn build_vm(hir: &HirProgram) -> Result<Vm, String> {
    let global_names: HashSet<String> = hir.globals.iter().map(|g| g.name.clone()).collect();
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
                // 函数名全局 FnRef（递归/前向引用按名解析所需）。
                vm.set_global(
                    func.name.clone(),
                    Value::FnRef {
                        name: func.name.clone(),
                        params: func.params.clone(),
                        return_type: func.return_type.clone(),
                        captures: vec![],
                    },
                );
            }
            Err(e) => return Err(format!("compile error ({}): {}", func.name, e)),
        }
    }
    if !hir.globals.is_empty() {
        let gcompiler = BytecodeCompiler::new_with_globals(global_names.clone());
        let (gchunk, gclosures) = gcompiler
            .compile_globals(&hir.globals)
            .map_err(|e| format!("compile_globals: {e}"))?;
        vm.add_fn("__global_init".into(), gchunk);
        for (name, c) in gclosures {
            vm.add_fn(name, c);
        }
        vm.call("__global_init").map_err(|e| format!("global init: {e}"))?;
    }
    if let Some(ref expr) = hir.main_expr {
        let compiler = BytecodeCompiler::new_with_globals(global_names.clone());
        let (chunk, closures) = compiler
            .compile_main(expr)
            .map_err(|e| format!("compile_main: {e}"))?;
        vm.add_fn("main".into(), chunk);
        for (name, c) in closures {
            vm.add_fn(name, c);
        }
    }
    Ok(vm)
}

enum Runner {
    Vm(Vm),
    Interp(Box<Interpreter>),
}

fn make_runner(hir: &HirProgram, path: Path) -> Result<Runner, String> {
    match path {
        Path::Vm | Path::Jit => Ok(Runner::Vm(build_vm(hir)?)),
        // 解释器：execute_fn_test 首次调用初始化顶层全局（含张量），
        // 之后 `init_program_globals` 跳过已存在变量 → 全局张量跨测量复用。
        Path::Interp => Ok(Runner::Interp(Box::new(Interpreter::new(hir)))),
        Path::Cli => Err("cli 路径是子进程端到端计时，不走进程内 runner".into()),
    }
}

fn run_once(runner: &mut Runner, path: Path, fn_name: &str) -> Result<Value, String> {
    match (runner, path) {
        (Runner::Vm(vm), Path::Jit) => jit::run_jit(vm, fn_name).map_err(|e| format!("jit: {e}")),
        (Runner::Vm(vm), _) => vm.call(fn_name).map_err(|e| format!("vm: {e}")),
        (Runner::Interp(interp), _) => interp
            .execute_fn_test(fn_name)
            .map_err(|e| format!("interp: {e}"))
            .map(|opt| opt.unwrap_or(Value::Unit)),
    }
}

/// 把返回值压成标量校验和（张量取全元素和）。
fn value_to_f64(v: &Value) -> Result<f64, String> {
    Ok(match v {
        Value::Int(n, _) => *n as f64,
        Value::Float(f) => *f,
        Value::Float32(f) => *f as f64,
        Value::Bool(b) => {
            if *b {
                1.0
            } else {
                0.0
            }
        }
        Value::Unit => 0.0,
        Value::Tensor(t) => {
            let tb = t.borrow();
            tb.data.as_f64_view().iter().copied().sum()
        }
        other => return Err(format!("无法转为标量校验和: {:?}", other.type_of())),
    })
}

fn check_checksum(sc: &Scenario, path: Path, reference: f64, actual: f64, phase: &str) {
    let denom = reference.abs().max(1.0);
    let rel = (actual - reference).abs() / denom;
    assert!(
        rel <= CHECKSUM_RTOL,
        "[{}/{}] {phase} 校验和不一致: 基准={reference:.12} {}={actual:.12} 相对误差={rel:.3e}",
        sc.group,
        sc.name,
        path.tag()
    );
}

/// 先跑 VM 路径一次取参考校验和（G5 `jit_compile_share` 用；单次 VM fib28 ≈ 0.3s，
/// 且 G5 独立进程运行，不会污染其他组的 jit）。
fn run_reference(sc: &Scenario) -> Result<f64, String> {
    let hir = lower(sc.src)?;
    let mut runner = make_runner(&hir, Path::Vm)?;
    if let Some(setup) = setup_fn_for(sc.name) {
        run_once(&mut runner, Path::Vm, setup)?;
    }
    let v = run_once(&mut runner, Path::Vm, "bench")?;
    value_to_f64(&v)
}

/// 单路径计时：warm-up 1 次 + 测量 5 次，返回 (单次 min/median, 校验和)。
///
/// - `expected` 为 `Some` 时，每条记录（含 warm-up）都与该基准对拍；为 `None` 时
///   仅做**自洽性**检查（warm-up + 5 次测量的校验和必须一致，抓非确定性）。
/// - 跨路径正确性对拍由 `perf_vm_reference` / `perf_interp_reference` 以 VM 为参考完成
///   （各组 jit 测试内不跑 VM，避免污染本进程后续测量）。
fn measure_path(sc: &Scenario, path: Path, expected: Option<f64>) -> Result<(Stats, f64), String> {
    let hir = lower(sc.src)?;
    let mut runner = make_runner(&hir, path)?;
    let setup = setup_fn_for(sc.name);
    let mut first_checksum: Option<f64> = None;
    let mut all_checksums: Vec<f64> = Vec::with_capacity(WARMUP_RUNS + MEASURE_RUNS);

    for _ in 0..WARMUP_RUNS {
        if let Some(f) = setup {
            run_once(&mut runner, path, f)?;
        }
        let v = run_once(&mut runner, path, "bench")?;
        let c = value_to_f64(&v)?;
        if let Some(e) = expected {
            check_checksum(sc, path, e, c, "warm-up");
        }
        first_checksum.get_or_insert(c);
        all_checksums.push(c);
    }

    // 防静默回退（总师裁定 #6②）：JIT 路径若 `bench` 编译失败，`run_jit` 会静默
    // `vm.call`，而 PERF 行仍标 `jit` —— 这正是静默失败防护要拦的红线。
    // 照抄 `tenth/tests/jit_silent_audit_test.rs:138-144` 的判据。
    if path == Path::Jit {
        if let Runner::Vm(vm) = &runner {
            let idx = vm
                .chunk_index_of("bench")
                .ok_or_else(|| format!("[{}] bench chunk 未注册", sc.name))?;
            let ctx = vm
                .jit_ctx
                .as_ref()
                .ok_or_else(|| format!("[{}] JIT 上下文未建立", sc.name))?;
            assert!(
                ctx.is_compiled(idx),
                "[{}/{}] jit 路径静默回退：bench 未被 JIT 编译（run_jit 已 fallback 到 vm.call）",
                sc.group,
                sc.name
            );
        }
    }

    let mut samples: Vec<f64> = Vec::with_capacity(MEASURE_RUNS);
    for _ in 0..MEASURE_RUNS {
        if let Some(f) = setup {
            run_once(&mut runner, path, f)?;
        }
        let t0 = Instant::now();
        let v = run_once(&mut runner, path, "bench")?;
        let dt = t0.elapsed();
        let c = value_to_f64(&v)?;
        if let Some(e) = expected {
            check_checksum(sc, path, e, c, "measured");
        }
        first_checksum.get_or_insert(c);
        all_checksums.push(c);
        samples.push(dt.as_secs_f64() * 1000.0 / sc.repeats as f64);
    }
    // 自洽性：同一路径 6 次运行的校验和必须一致（抓非确定性/随机化）。
    let first = all_checksums[0];
    for (i, c) in all_checksums.iter().enumerate() {
        check_checksum(sc, path, first, *c, &format!("自洽性(第 {i} 次)"));
    }
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    Ok((
        Stats {
            min_ms: samples[0],
            median_ms: samples[MEASURE_RUNS / 2],
        },
        first,
    ))
}

/// 输出一条路径的 min/median（G2 的 matmul 附带 GFLOPS）。
fn emit_stats(sc: &Scenario, path: Path, st: Stats) {
    emit(sc.group, sc.name, path, "min", st.min_ms, "ms");
    emit(sc.group, sc.name, path, "median", st.median_ms, "ms");
    if sc.group == "G2" {
        match sc.name {
            "matmul_512" => emit_gflops(sc.group, sc.name, path, 512, st.median_ms),
            "matmul_1024" => emit_gflops(sc.group, sc.name, path, 1024, st.median_ms),
            _ => {}
        }
    }
}

/// 只测 JIT（产品默认路径）。各组 jit 测试内**不跑 VM**：
/// VM 长循环（G1 `loop_mod_1e7` ≈ 3–5s）会把后续组的 jit 膨胀 2–3×；同进程内
/// 前一组的堆状态也会把张量算子膨胀 2–3×（实测 matmul_512 单独进程 6.5ms → G1 之后 15–20ms）。
/// 故规范跑法是**每组一个独立进程**（见文件头），跨路径对拍由 `perf_vm_reference` 完成。
fn run_group_jit(scenarios: &[&Scenario]) {
    for sc in scenarios {
        assert!(
            sc.paths.contains(&Path::Jit),
            "[{}/{}] 主基线场景必须含 jit 路径",
            sc.group,
            sc.name
        );
        let (st, _ck) = measure_path(sc, Path::Jit, None)
            .unwrap_or_else(|e| panic!("[{}/{}] jit 路径计时失败: {e}", sc.group, sc.name));
        emit_stats(sc, Path::Jit, st);
        if let Some(note) = jit_note_for(sc.name) {
            emit_note(sc.group, sc.name, Path::Jit.tag(), note);
        }
    }
}

// ── 场景源码（Rust 字符串常量内嵌；不新建 .th）─────────────────────────────
//
// 注 1：G3/G4 的 nn/autodiff 场景**内联镜像** `tenth/std/nn/*.th` 与
// `tenth/std/optim/adamw.th` 的数学（linear / MHA / layer_norm / softmax-CE / AdamW），
// 因为内嵌源码经 `Lowerer::new()` 编译、无模块搜索路径，`use std::nn::...` 不可用；
// 内联版本与 std 实现逐行对齐（形状流、loss、更新公式），差异仅在不经过模块封装。
//
// 注 2：张量场景把大张量放**顶层 `let` 全局**，使构造发生在计时区间之外
//（VM 走 `__global_init`；解释器首次 `execute_fn_test` 初始化、之后复用）。
// autodiff 场景的 `param()` 必须每次 `new_grad()` 之后创建，故参数创建在计时区间内
//（占比已由 tape_overhead 场景单独量化）。

// ── G1 标量 / 控制流 ───────────────────────────────────────────────────────

const SRC_FIB28: &str = r#"
fn fib(n: i64) -> i64 {
    if n < 2 { n } else { fib(n - 1) + fib(n - 2) }
}
fn bench() -> i64 { fib(28) }
"#;

const SRC_LOOP_1E7: &str = r#"
fn bench() -> i64 {
    let mut sum = 0;
    let mut i = 0;
    while i < 10000000 {
        sum = sum + (i % 7);
        i = i + 1;
    }
    sum
}
"#;

const SRC_CLOSURE_1E6: &str = r#"
fn bench() -> i64 {
    let f = |x: i64| x * 2 + 1;
    let mut s = 0;
    let mut i = 0;
    while i < 1000000 {
        s = s + f(i % 100);
        i = i + 1;
    }
    s
}
"#;

const SRC_METHOD_VEC_1E6: &str = r#"
fn bench() -> i64 {
    let mut v = Vec::new();
    let mut i = 0;
    while i < 1000000 {
        v.push(i % 7);
        i = i + 1;
    }
    v.len()
}
"#;

const SRC_TAIL_REC_1E6: &str = r#"
fn tr(n: i64, acc: i64) -> i64 {
    if n <= 0 { acc } else { tr(n - 1, acc + (n % 7)) }
}
fn bench() -> i64 { tr(1000000, 0) }
"#;

// ── G2 张量算子 ────────────────────────────────────────────────────────────

const SRC_MATMUL_512: &str = r#"
let a = ones(512, 512);
let b = ones(512, 512) * 0.001;
fn bench_once() -> f64 { a.matmul(b).sum() }
fn bench() -> f64 {
    let mut acc = 0.0;
    let mut r = 0;
    while r < 2 { acc = acc + bench_once(); r = r + 1; }
    acc
}
"#;

const SRC_MATMUL_1024: &str = r#"
let a = ones(1024, 1024);
let b = ones(1024, 1024) * 0.001;
fn bench_once() -> f64 { a.matmul(b).sum() }
fn bench() -> f64 {
    let mut acc = 0.0;
    let mut r = 0;
    while r < 1 { acc = acc + bench_once(); r = r + 1; }
    acc
}
"#;

const SRC_ELEMENTWISE_MUL_1K: &str = r#"
let a = ones(1000, 1000);
let b = ones(1000, 1000) * 0.5;
fn bench_once() -> f64 { (a * b).sum() }
fn bench() -> f64 {
    let mut acc = 0.0;
    let mut r = 0;
    while r < 2 { acc = acc + bench_once(); r = r + 1; }
    acc
}
"#;

const SRC_BROADCAST_ADD_1K: &str = r#"
let a = ones(1000, 1000);
let b = ones(1000) * 0.25;
fn bench_once() -> f64 { (a + b).sum() }
fn bench() -> f64 {
    let mut acc = 0.0;
    let mut r = 0;
    while r < 2 { acc = acc + bench_once(); r = r + 1; }
    acc
}
"#;

const SRC_REDUCE_SUM_1K: &str = r#"
let a = ones(1000, 1000) * 0.5;
fn bench_once() -> f64 { a.sum() }
fn bench() -> f64 {
    let mut acc = 0.0;
    let mut r = 0;
    while r < 100 { acc = acc + bench_once(); r = r + 1; }
    acc
}
"#;

const SRC_TRANSPOSE_1K: &str = r#"
let a = ones(1000, 1000);
fn bench_once() -> f64 { a.transpose().sum() }
fn bench() -> f64 {
    let mut acc = 0.0;
    let mut r = 0;
    while r < 8 { acc = acc + bench_once(); r = r + 1; }
    acc
}
"#;

// ── G3 autodiff ────────────────────────────────────────────────────────────

/// 784→256→10 MLP，batch=64，CE loss，forward + backward 全流程。
const SRC_MLP_FWD_BWD: &str = r#"
fn bench() -> f64 {
    new_grad();
    let x = ones(64, 784) * 0.01;
    let w1 = param(ones(256, 784) * 0.01);
    let b1 = param(ones(256) * 0.01);
    let w2 = param(ones(10, 256) * 0.01);
    let b2 = param(ones(10) * 0.01);
    let target = ones(64, 10) * 0.1;
    let h = (x.matmul(w1.transpose()) + b1).relu();
    let logits = h.matmul(w2.transpose()) + b2;
    let loss = cross_entropy(logits, target);
    backward(loss);
    stop_grad();
    loss.sum() + grad(w1).sum() + grad(b1).sum() + grad(w2).sum() + grad(b2).sum()
}
"#;

/// Transformer block（MHA + FFN + 残差），d=256 / heads=8 / seq=32 / batch=8，
/// forward + backward 全流程；batch 维度用 Tenth 循环展开。
const SRC_TRANSFORMER_BLOCK_FWD_BWD: &str = r#"
fn mha_block(
    x: Tensor[f64, ..],
    wq: Tensor[f64, ..],
    wk: Tensor[f64, ..],
    wv: Tensor[f64, ..],
    wo: Tensor[f64, ..],
    n_heads: i64,
    seq_len: i64,
    d_model: i64
) -> Tensor[f64, ..] {
    let d_k = d_model / n_heads;
    let q = x.matmul(wq);
    let k = x.matmul(wk);
    let v = x.matmul(wv);
    let q3 = q.reshape(n_heads, seq_len, d_k);
    let k3 = k.reshape(n_heads, seq_len, d_k);
    let v3 = v.reshape(n_heads, seq_len, d_k);
    let scale = 1.0 / sqrt(to_f64(d_k));
    let scores = q3.bmm(k3.transpose()) * scale;
    let w = scores.softmax();
    let o3 = w.bmm(v3);
    let o = o3.reshape(seq_len, d_model);
    o.matmul(wo)
}

fn bench() -> f64 {
    new_grad();
    let n_heads = 8;
    let seq_len = 32;
    let d_model = 256;
    let batch = 8;
    let x = ones(seq_len, d_model) * 0.01;
    let wq = param(ones(d_model, d_model) * 0.01);
    let wk = param(ones(d_model, d_model) * 0.01);
    let wv = param(ones(d_model, d_model) * 0.01);
    let wo = param(ones(d_model, d_model) * 0.01);
    let w1 = param(ones(1024, d_model) * 0.01);
    let w2 = param(ones(d_model, 1024) * 0.01);
    let mut total = zeros(seq_len, d_model);
    let mut b = 0;
    while b < batch {
        let h = mha_block(x, wq, wk, wv, wo, n_heads, seq_len, d_model);
        let ff = (h.matmul(w1.transpose())).relu().matmul(w2.transpose()) + h;
        total = total + ff;
        b = b + 1;
    }
    backward(total);
    stop_grad();
    total.sum() + grad(wq).sum() + grad(w1).sum()
}
"#;

/// tape_overhead 对照：autodiff 前向+反向（含 param 创建）。
const SRC_TAPE_OVERHEAD_AD: &str = r#"
fn bench() -> f64 {
    new_grad();
    let w = param(ones(256, 256) * 0.01);
    let x = ones(64, 256) * 0.01;
    let y = x.matmul(w.transpose());
    let loss = (y * y).sum();
    backward(loss);
    stop_grad();
    loss.sum() + grad(w).sum()
}
"#;

/// tape_overhead 对照：等价手写前向（无 tape）。
const SRC_TAPE_OVERHEAD_MANUAL: &str = r#"
fn bench() -> f64 {
    let w = ones(256, 256) * 0.01;
    let x = ones(64, 256) * 0.01;
    let y = x.matmul(w.transpose());
    (y * y).sum()
}
"#;

// ── G4 nn / 优化器 ─────────────────────────────────────────────────────────

/// Linear 层 fwd+bwd（y = x @ W^T + b，loss = sum(y²)）。
const SRC_LINEAR_FWD_BWD: &str = r#"
fn bench() -> f64 {
    new_grad();
    let x = ones(64, 784) * 0.01;
    let w = param(ones(256, 784) * 0.01);
    let b = param(ones(256) * 0.01);
    let y = x.matmul(w.transpose()) + b;
    let loss = (y * y).sum();
    backward(loss);
    stop_grad();
    loss.sum() + grad(w).sum() + grad(b).sum()
}
"#;

/// MHA fwd+bwd（单 batch，d=256/heads=8/seq=32）。
const SRC_MHA_FWD_BWD: &str = r#"
fn mha_block(
    x: Tensor[f64, ..],
    wq: Tensor[f64, ..],
    wk: Tensor[f64, ..],
    wv: Tensor[f64, ..],
    wo: Tensor[f64, ..],
    n_heads: i64,
    seq_len: i64,
    d_model: i64
) -> Tensor[f64, ..] {
    let d_k = d_model / n_heads;
    let q = x.matmul(wq);
    let k = x.matmul(wk);
    let v = x.matmul(wv);
    let q3 = q.reshape(n_heads, seq_len, d_k);
    let k3 = k.reshape(n_heads, seq_len, d_k);
    let v3 = v.reshape(n_heads, seq_len, d_k);
    let scale = 1.0 / sqrt(to_f64(d_k));
    let scores = q3.bmm(k3.transpose()) * scale;
    let w = scores.softmax();
    let o3 = w.bmm(v3);
    let o = o3.reshape(seq_len, d_model);
    o.matmul(wo)
}

fn bench_once() -> f64 {
    new_grad();
    let seq_len = 32;
    let d_model = 256;
    let x = ones(seq_len, d_model) * 0.01;
    let wq = param(ones(d_model, d_model) * 0.01);
    let wk = param(ones(d_model, d_model) * 0.01);
    let wv = param(ones(d_model, d_model) * 0.01);
    let wo = param(ones(d_model, d_model) * 0.01);
    let h = mha_block(x, wq, wk, wv, wo, 8, seq_len, d_model);
    let loss = (h * h).sum();
    backward(loss);
    stop_grad();
    loss.sum() + grad(wq).sum()
}

fn bench() -> f64 {
    let mut acc = 0.0;
    let mut r = 0;
    while r < 4 { acc = acc + bench_once(); r = r + 1; }
    acc
}
"#;

/// LayerNorm fwd+bwd（64×256，gamma/beta 可学习）。
const SRC_LAYER_NORM_FWD_BWD: &str = r#"
fn bench_once() -> f64 {
    new_grad();
    let x = ones(64, 256) * 0.01;
    let gamma = param(ones(256));
    let beta = param(zeros(256));
    let y = x.layer_norm(gamma, beta, 0.00001);
    let loss = (y * y).sum();
    backward(loss);
    stop_grad();
    loss.sum() + grad(gamma).sum() + grad(beta).sum()
}

fn bench() -> f64 {
    let mut acc = 0.0;
    let mut r = 0;
    while r < 64 { acc = acc + bench_once(); r = r + 1; }
    acc
}
"#;

/// Softmax + CrossEntropy fwd+bwd（64×10）。
const SRC_SOFTMAX_CE_FWD_BWD: &str = r#"
fn bench_once() -> f64 {
    new_grad();
    let logits = param(ones(64, 10) * 0.5);
    let target = ones(64, 10) * 0.1;
    let loss = cross_entropy(logits, target);
    backward(loss);
    stop_grad();
    loss.sum() + grad(logits).sum()
}

fn bench() -> f64 {
    let mut acc = 0.0;
    let mut r = 0;
    while r < 1024 { acc = acc + bench_once(); r = r + 1; }
    acc
}
"#;

/// AdamW 单步（1e6 元素参数；镜像 `tenth/std/optim/adamw.th::adamw_step_w` 的数学）。
///
/// `setup()` 在计时区间**之外**准备参数与梯度（`new_grad`+`param`+`backward`+`grad`），
/// 计时区间只含优化器 step 本身——与契约场景名 `adamw_step_1e6` 对齐（不含 fwd/bwd）。
const SRC_ADAMW_STEP_1E6: &str = r#"
let m = zeros(1000, 1000);
let v = zeros(1000, 1000);
let mut w = zeros(1, 1);
let mut gw = zeros(1, 1);

fn setup() -> i64 {
    new_grad();
    w = param(ones(1000, 1000) * 0.01);
    let loss = (w * w).sum();
    backward(loss);
    stop_grad();
    gw = grad(w);
    0
}

fn bench() -> f64 {
    let new_m = 0.9 * m + 0.1 * gw;
    let new_v = 0.999 * v + 0.001 * gw * gw;
    let m_hat = new_m / (1.0 - 0.9);
    let v_hat = new_v / (1.0 - 0.999);
    let decayed_w = w * (1.0 - 0.001 * 0.01);
    let new_w = decayed_w - 0.001 * m_hat / (v_hat.sqrt() + 0.00000001);
    new_w.sum() + new_m.sum() + new_v.sum()
}
"#;

// ── 场景表 ─────────────────────────────────────────────────────────────────

fn scenario_g1() -> Vec<&'static Scenario> {
    vec![
        &Scenario {
            group: "G1",
            name: "fib28",
            src: SRC_FIB28,
            paths: &VM_JIT,
            repeats: 1,
            interp_skip: None,
        },
        &Scenario {
            group: "G1",
            name: "loop_mod_1e7",
            src: SRC_LOOP_1E7,
            paths: &VM_JIT,
            repeats: 1,
            interp_skip: None,
        },
        &Scenario {
            group: "G1",
            name: "closure_1e6",
            src: SRC_CLOSURE_1E6,
            paths: &VM_JIT,
            repeats: 1,
            interp_skip: None,
        },
        &Scenario {
            group: "G1",
            name: "method_vec_1e6",
            src: SRC_METHOD_VEC_1E6,
            paths: &VM_JIT,
            repeats: 1,
            interp_skip: None,
        },
        &Scenario {
            group: "G1",
            name: "tail_rec_1e6",
            src: SRC_TAIL_REC_1E6,
            paths: &VM_JIT,
            repeats: 1,
            interp_skip: Some(
                "解释器（tree-walk）未实现 TCO，1e6 深度尾递归会栈溢出；JIT/VM 有 TCO",
            ),
        },
    ]
}

fn scenario_g2() -> Vec<&'static Scenario> {
    vec![
        &Scenario {
            group: "G2",
            name: "matmul_512",
            src: SRC_MATMUL_512,
            paths: &VM_JIT,
            repeats: 2,
            interp_skip: None,
        },
        &Scenario {
            group: "G2",
            name: "matmul_1024",
            src: SRC_MATMUL_1024,
            paths: &VM_JIT,
            repeats: 1,
            interp_skip: None,
        },
        &Scenario {
            group: "G2",
            name: "elementwise_mul_1k",
            src: SRC_ELEMENTWISE_MUL_1K,
            paths: &VM_JIT,
            repeats: 2,
            interp_skip: None,
        },
        &Scenario {
            group: "G2",
            name: "broadcast_add_1k",
            src: SRC_BROADCAST_ADD_1K,
            paths: &VM_JIT,
            repeats: 2,
            interp_skip: None,
        },
        &Scenario {
            group: "G2",
            name: "reduce_sum_1k",
            src: SRC_REDUCE_SUM_1K,
            paths: &VM_JIT,
            repeats: 100,
            interp_skip: None,
        },
        &Scenario {
            group: "G2",
            name: "transpose_1k",
            src: SRC_TRANSPOSE_1K,
            paths: &VM_JIT,
            repeats: 8,
            interp_skip: None,
        },
    ]
}

fn scenario_g3() -> Vec<&'static Scenario> {
    vec![
        &Scenario {
            group: "G3",
            name: "mlp_fwd_bwd",
            src: SRC_MLP_FWD_BWD,
            paths: &VM_JIT,
            repeats: 1,
            interp_skip: None,
        },
        &Scenario {
            group: "G3",
            name: "transformer_block_fwd_bwd",
            src: SRC_TRANSFORMER_BLOCK_FWD_BWD,
            paths: &VM_JIT,
            repeats: 1,
            interp_skip: None,
        },
    ]
}

fn scenario_g4() -> Vec<&'static Scenario> {
    vec![
        &Scenario {
            group: "G4",
            name: "linear_fwd_bwd",
            src: SRC_LINEAR_FWD_BWD,
            paths: &VM_JIT,
            repeats: 1,
            interp_skip: None,
        },
        &Scenario {
            group: "G4",
            name: "mha_fwd_bwd",
            src: SRC_MHA_FWD_BWD,
            paths: &VM_JIT,
            repeats: 4,
            interp_skip: None,
        },
        &Scenario {
            group: "G4",
            name: "layer_norm_fwd_bwd",
            src: SRC_LAYER_NORM_FWD_BWD,
            paths: &VM_JIT,
            repeats: 64,
            interp_skip: None,
        },
        &Scenario {
            group: "G4",
            name: "softmax_ce_fwd_bwd",
            src: SRC_SOFTMAX_CE_FWD_BWD,
            paths: &VM_JIT,
            repeats: 1024,
            interp_skip: None,
        },
        &Scenario {
            group: "G4",
            name: "adamw_step_1e6",
            src: SRC_ADAMW_STEP_1E6,
            paths: &VM_JIT,
            repeats: 1,
            interp_skip: None,
        },
    ]
}

/// G6 三路径对比场景（interp 由 `perf_interp_reference` 单独测量）。
fn scenario_g6() -> Vec<&'static Scenario> {
    vec![
        &Scenario {
            group: "G6",
            name: "fib28",
            src: SRC_FIB28,
            paths: &VM_JIT,
            repeats: 1,
            interp_skip: None,
        },
        &Scenario {
            group: "G6",
            name: "loop_mod_1e7",
            src: SRC_LOOP_1E7,
            paths: &VM_JIT,
            repeats: 1,
            interp_skip: None,
        },
        &Scenario {
            group: "G6",
            name: "matmul_512",
            src: SRC_MATMUL_512,
            paths: &VM_JIT,
            repeats: 2,
            interp_skip: None,
        },
    ]
}

// ── G2 GFLOPS 补充 ─────────────────────────────────────────────────────────

/// matmul 的 GFLOPS = 2·n³ / t，从 median 计算（见 `emit` 的 min/median 行）。
fn emit_gflops(group: &str, scenario: &str, path: Path, n: u64, median_ms: f64) {
    if median_ms > 0.0 {
        let flops = 2.0 * (n as f64).powi(3);
        let gflops = flops / (median_ms / 1000.0) / 1e9;
        emit(group, scenario, path, "gflops_median", gflops, "GFLOPS");
    }
}

// ── 测试：G1 ───────────────────────────────────────────────────────────────

#[test]
#[ignore = "性能基线（非门槛）：需 release + --ignored --nocapture；interp 在 perf_interp_reference，见文件头 doc comment"]
fn perf_g1_scalar_controlflow() {
    let _g = serial_guard();
    if cfg!(debug_assertions) {
        eprintln!("perf_baseline: 非 release 构建，跳过（基线只在 release 有意义）");
        return;
    }
    cpu_freq_warmup();
    emit_methodology_note("G1");
    run_group_jit(&scenario_g1());
    emit_note(
        "G1",
        "interp_reference",
        "all",
        "interp 路径移至独立 perf_interp_reference（隔离热污染，总师裁定 #10②）；绝对值为热态、仅比值可用",
    );
}

// ── 测试：G2 ───────────────────────────────────────────────────────────────

#[test]
#[ignore = "性能基线（非门槛）：需 release + --ignored --nocapture；见文件头 doc comment"]
fn perf_g2_tensor_ops() {
    let _g = serial_guard();
    if cfg!(debug_assertions) {
        eprintln!("perf_baseline: 非 release 构建，跳过（基线只在 release 有意义）");
        return;
    }
    cpu_freq_warmup();
    emit_methodology_note("G2");
    run_group_jit(&scenario_g2());
    // 口径披露：GFLOPS 用 median 计算，而计时区间含 `.sum()` 归约（512² 约 0.2ms），
    // 故 GFLOPS 系统性低估约 1.5%（契约要求报 GFLOPS，未要求扣除归约）。
    emit_note(
        "G2",
        "matmul_512",
        "all",
        "gflops_median 计时含 .sum() 归约（512² 约 0.2ms），GFLOPS 低估约 1.5%",
    );
    emit_note(
        "G2",
        "matmul_1024",
        "all",
        "gflops_median 计时含 .sum() 归约，GFLOPS 低估约 1.5%",
    );
}

// ── 测试：G3 ───────────────────────────────────────────────────────────────

#[test]
#[ignore = "性能基线（非门槛）：需 release + --ignored --nocapture；见文件头 doc comment"]
fn perf_g3_autodiff() {
    let _g = serial_guard();
    if cfg!(debug_assertions) {
        eprintln!("perf_baseline: 非 release 构建，跳过（基线只在 release 有意义）");
        return;
    }
    cpu_freq_warmup();
    emit_methodology_note("G3");
    run_group_jit(&scenario_g3());

    // tape_overhead：autodiff（fwd+bwd）vs 等价手写前向（无 tape）。先 jit 后 vm（#10①）。
    let ad = Scenario {
        group: "G3",
        name: "tape_overhead",
        src: SRC_TAPE_OVERHEAD_AD,
        paths: &VM_JIT,
        repeats: 1,
        interp_skip: None,
    };
    let manual = Scenario {
        group: "G3",
        name: "tape_overhead",
        src: SRC_TAPE_OVERHEAD_MANUAL,
        paths: &VM_JIT,
        repeats: 1,
        interp_skip: None,
    };
    let (ad_jit, ad_ck) = measure_path(&ad, Path::Jit, None)
        .unwrap_or_else(|e| panic!("tape_overhead(ad) jit 失败: {e}"));
    let (ad_vm, _) = measure_path(&ad, Path::Vm, Some(ad_ck))
        .unwrap_or_else(|e| panic!("tape_overhead(ad) vm 失败: {e}"));
    let (man_jit, man_ck) = measure_path(&manual, Path::Jit, None)
        .unwrap_or_else(|e| panic!("tape_overhead(manual) jit 失败: {e}"));
    let (man_vm, _) = measure_path(&manual, Path::Vm, Some(man_ck))
        .unwrap_or_else(|e| panic!("tape_overhead(manual) vm 失败: {e}"));
    for (path, a, m) in [
        (Path::Jit, ad_jit, man_jit),
        (Path::Vm, ad_vm, man_vm),
    ] {
        emit("G3", "tape_overhead", path, "ad_min", a.min_ms, "ms");
        emit("G3", "tape_overhead", path, "ad_median", a.median_ms, "ms");
        emit("G3", "tape_overhead", path, "manual_min", m.min_ms, "ms");
        emit("G3", "tape_overhead", path, "manual_median", m.median_ms, "ms");
        if m.median_ms > 0.0 {
            emit(
                "G3",
                "tape_overhead",
                path,
                "ratio_ad_over_manual",
                a.median_ms / m.median_ms,
                "x",
            );
        }
    }
}

// ── 测试：G4 ───────────────────────────────────────────────────────────────

#[test]
#[ignore = "性能基线（非门槛）：需 release + --ignored --nocapture；见文件头 doc comment"]
fn perf_g4_nn_optim() {
    let _g = serial_guard();
    if cfg!(debug_assertions) {
        eprintln!("perf_baseline: 非 release 构建，跳过（基线只在 release 有意义）");
        return;
    }
    cpu_freq_warmup();
    emit_methodology_note("G4");
    run_group_jit(&scenario_g4());
    // 口径披露（总师裁定 #6④）：layer_norm 输入全 ones → 输出 ≈0，校验和弱。
    emit_note(
        "G4",
        "layer_norm_fwd_bwd",
        "all",
        "输入全 ones → 归一化输出 ≈0，跨路径校验和区分度弱（正确性另由既有 layer_norm 测试覆盖）；计时不受影响",
    );
}

// ── 测试：G5 编译 / 启动（子进程）──────────────────────────────────────────

fn bench_dir() -> PathBuf {
    PathBuf::from(TENTH_DIR).join("target").join("bench_tmp")
}

/// 跑一次 `tenth.exe run <file>`（子进程，含进程启动 + 编译 + 执行）。
fn run_cli(file: &FsPath, cwd: &FsPath) -> std::process::Output {
    Command::new(TENTH_EXE)
        .arg("run")
        .arg(file)
        .current_dir(cwd)
        .output()
        .expect("启动 tenth.exe 失败")
}

fn stats_from_durations(mut v: Vec<Duration>) -> Stats {
    v.sort();
    Stats {
        min_ms: v[0].as_secs_f64() * 1000.0,
        median_ms: v[v.len() / 2].as_secs_f64() * 1000.0,
    }
}

#[test]
#[ignore = "性能基线（非门槛）：需 release + --ignored --nocapture；含子进程与临时 .th，见文件头 doc comment"]
fn perf_g5_compile_startup() {
    let _g = serial_guard();
    if cfg!(debug_assertions) {
        eprintln!("perf_baseline: 非 release 构建，跳过（基线只在 release 有意义）");
        return;
    }
    cpu_freq_warmup();
    emit_methodology_note("G5");

    // ① cold_start_hello：子进程 tenth.exe run hello.th（启动 + 编译 + 执行）。
    let dir = bench_dir();
    std::fs::create_dir_all(&dir).expect("创建 target/bench_tmp 失败");
    let hello = dir.join("perf_cold_start_hello.th");
    std::fs::write(&hello, "fn main() {\n    println(\"hello\");\n}\n")
        .expect("写临时 hello.th 失败");
    let tenth_dir = PathBuf::from(TENTH_DIR);
    let warm = run_cli(&hello, &tenth_dir);
    assert!(
        warm.status.success() && String::from_utf8_lossy(&warm.stdout).contains("hello"),
        "cold_start_hello warm-up 失败: status={:?} stdout={} stderr={}",
        warm.status,
        String::from_utf8_lossy(&warm.stdout),
        String::from_utf8_lossy(&warm.stderr)
    );
    let mut samples = Vec::with_capacity(MEASURE_RUNS);
    for _ in 0..MEASURE_RUNS {
        let t0 = Instant::now();
        let out = run_cli(&hello, &tenth_dir);
        samples.push(t0.elapsed());
        assert!(
            out.status.success() && String::from_utf8_lossy(&out.stdout).contains("hello"),
            "cold_start_hello 失败: status={:?} stderr={}",
            out.status,
            String::from_utf8_lossy(&out.stderr)
        );
    }
    let _ = std::fs::remove_file(&hello);
    let st = stats_from_durations(samples);
    emit("G5", "cold_start_hello", Path::Cli, "min", st.min_ms, "ms");
    emit("G5", "cold_start_hello", Path::Cli, "median", st.median_ms, "ms");

    // ② selfhost_main：子进程 tenth.exe run tenthc/main.th（工作规范 §四：自举管线 <1s）。
    let repo_root = PathBuf::from(TENTH_DIR)
        .parent()
        .expect("tenth/ 的父目录（仓库根）不存在")
        .to_path_buf();
    let selfhost = repo_root.join("tenthc").join("main.th");
    assert!(selfhost.exists(), "未找到 {}", selfhost.display());
    let warm = run_cli(&selfhost, &repo_root);
    let warm_out = String::from_utf8_lossy(&warm.stdout).to_string();
    assert!(
        warm.status.success() && warm_out.contains("[OK]"),
        "selfhost_main warm-up 失败: status={:?} stdout={} stderr={}",
        warm.status,
        warm_out,
        String::from_utf8_lossy(&warm.stderr)
    );
    let mut samples = Vec::with_capacity(MEASURE_RUNS);
    for _ in 0..MEASURE_RUNS {
        let t0 = Instant::now();
        let out = run_cli(&selfhost, &repo_root);
        samples.push(t0.elapsed());
        assert!(
            out.status.success() && String::from_utf8_lossy(&out.stdout).contains("[OK]"),
            "selfhost_main 失败: status={:?} stderr={}",
            out.status,
            String::from_utf8_lossy(&out.stderr)
        );
    }
    let st = stats_from_durations(samples);
    emit("G5", "selfhost_main", Path::Cli, "min", st.min_ms, "ms");
    emit("G5", "selfhost_main", Path::Cli, "median", st.median_ms, "ms");

    // ③ jit_compile_share：首次 JIT 编译（含 Cranelift 编译 + 缓存建立）vs 已编译稳态。
    let sc = Scenario {
        group: "G5",
        name: "jit_compile_share",
        src: SRC_FIB28,
        paths: &[Path::Jit],
        repeats: 1,
        interp_skip: None,
    };
    let hir = lower(sc.src).expect("jit_compile_share lower 失败");
    let mut vm = build_vm(&hir).expect("jit_compile_share build_vm 失败");
    let mut checks: Vec<f64> = Vec::new();
    let t0 = Instant::now();
    let cold_v = jit::run_jit(&mut vm, "bench").expect("jit_compile_share 首次 JIT 失败");
    let cold = t0.elapsed().as_secs_f64() * 1000.0;
    checks.push(value_to_f64(&cold_v).expect("jit_compile_share 首次结果转换失败"));
    let mut steady = Vec::with_capacity(MEASURE_RUNS);
    for _ in 0..MEASURE_RUNS {
        let t0 = Instant::now();
        let v = jit::run_jit(&mut vm, "bench").expect("jit_compile_share 稳态 JIT 失败");
        steady.push(t0.elapsed());
        checks.push(value_to_f64(&v).expect("jit_compile_share 结果转换失败"));
    }
    // VM 参考放在 JIT 计时之后（G5 独立进程运行，fib28 vm ≈ 0.3s，不污染其他组）。
    let reference = run_reference(&sc).expect("jit_compile_share VM 参考失败");
    for (i, c) in checks.iter().enumerate() {
        check_checksum(
            &sc,
            Path::Jit,
            reference,
            *c,
            if i == 0 { "cold-jit" } else { "steady-jit" },
        );
    }
    let st = stats_from_durations(steady);
    emit("G5", "jit_compile_share", Path::Jit, "first_call", cold, "ms");
    emit("G5", "jit_compile_share", Path::Jit, "steady_min", st.min_ms, "ms");
    emit("G5", "jit_compile_share", Path::Jit, "steady_median", st.median_ms, "ms");
    // 编译占比用 first_call 对 steady_min（min 抗噪；median 在首次编译成本 < 抖动时会给出负占比）。
    if cold > 0.0 {
        emit(
            "G5",
            "jit_compile_share",
            Path::Jit,
            "compile_overhead",
            (cold - st.min_ms).max(0.0),
            "ms",
        );
        emit(
            "G5",
            "jit_compile_share",
            Path::Jit,
            "compile_share",
            (cold - st.min_ms).max(0.0) / cold * 100.0,
            "%",
        );
    }

    // 收尾：临时 .th 已删除；目录若非空（并行部门临时文件）保留。
    let _ = std::fs::remove_dir(&dir);
}

// ── 测试：G6 三路径对比 ────────────────────────────────────────────────────

#[test]
#[ignore = "性能基线（非门槛）：需 release + --ignored --nocapture；interp 列在 perf_interp_reference，见文件头 doc comment"]
fn perf_g6_three_path_compare() {
    let _g = serial_guard();
    if cfg!(debug_assertions) {
        eprintln!("perf_baseline: 非 release 构建，跳过（基线只在 release 有意义）");
        return;
    }
    cpu_freq_warmup();
    emit_methodology_note("G6");
    // 只测 JIT；VM 与 interp 分别在 perf_vm_reference / perf_interp_reference 测量。
    let scs = scenario_g6();
    for sc in &scs {
        let (st, _ck) = measure_path(sc, Path::Jit, None)
            .unwrap_or_else(|e| panic!("[G6/{}] jit 计时失败: {e}", sc.name));
        emit_stats(sc, Path::Jit, st);
    }
    emit_note(
        "G6",
        "vm_reference",
        "all",
        "vm 列与 jit/vm 比值见 perf_vm_reference（同 group 行）；interp 列见 perf_interp_reference",
    );
}

// ── 测试：VM 参考路径（独立进程跑，给出 vm 数据与跨路径对拍）───────────────

/// VM（字节码循环）路径：独立测量；每场景以 VM 校验和为参考，再跑一次 JIT 对拍
///（VM 为参考），并给出 G6 的 jit/vm 比值。
///
/// 规范跑法中本测试与各组 jit 测试**各占一个独立进程**（见文件头），故 VM 长循环
/// 不会污染任何 jit 测量。
#[test]
#[ignore = "性能基线（非门槛）：VM 参考路径，独立进程跑；需 release + --ignored --nocapture"]
fn perf_vm_reference() {
    let _g = serial_guard();
    if cfg!(debug_assertions) {
        eprintln!("perf_baseline: 非 release 构建，跳过（基线只在 release 有意义）");
        return;
    }
    cpu_freq_warmup();
    emit_methodology_note("vm");
    emit_note(
        "vm",
        "methodology",
        "vm",
        "VM 参考路径：绝对值为热态、仅作同组 jit/vm 比值与跨路径对拍（VM 为参考）",
    );

    let mut scs: Vec<&Scenario> = scenario_g1();
    scs.extend(scenario_g2());
    scs.extend(scenario_g3());
    scs.extend(scenario_g4());
    let g6 = scenario_g6();
    scs.extend(g6);

    // 阶段 1：先测全部 JIT（本进程内尚无 VM 运行 → jit 干净），作为对拍与比值基准。
    let mut jit_results: Vec<(Stats, f64)> = Vec::with_capacity(scs.len());
    for sc in &scs {
        let r = measure_path(sc, Path::Jit, None)
            .unwrap_or_else(|e| panic!("[{}/{}] jit 计时失败: {e}", sc.group, sc.name));
        jit_results.push(r);
    }
    // 阶段 2：测全部 VM（VM 为参考），与 JIT 校验和对拍；输出 vm 数据与 G6 比值。
    for (i, sc) in scs.iter().enumerate() {
        let (jit_st, jit_ck) = jit_results[i];
        let (st, vm_ck) = measure_path(sc, Path::Vm, None)
            .unwrap_or_else(|e| panic!("[{}/{}] vm 计时失败: {e}", sc.group, sc.name));
        check_checksum(sc, Path::Jit, vm_ck, jit_ck, "VM 参考对拍");
        emit_stats(sc, Path::Vm, st);
        if sc.group == "G6" && st.median_ms > 0.0 {
            emit(
                sc.group,
                sc.name,
                Path::Jit,
                "ratio_vs_vm_median",
                jit_st.median_ms / st.median_ms,
                "x",
            );
        }
    }
}

// ── 测试：interp 参考路径（独立，隔离热污染）───────────────────────────────

/// interp 参考路径：单独测量（总师裁定 #10②），不参与主基线的热态序列。
///
/// 契约 G1 的 interp 列与 G6 的 interp 列都在此输出；数据保留但**标注参考路径、仅比值可用**。
#[test]
#[ignore = "性能基线（非门槛）：interp 参考路径，单独跑以隔离热污染；需 release + --ignored --nocapture"]
fn perf_interp_reference() {
    let _g = serial_guard();
    if cfg!(debug_assertions) {
        eprintln!("perf_baseline: 非 release 构建，跳过（基线只在 release 有意义）");
        return;
    }
    cpu_freq_warmup();
    emit_methodology_note("interp");
    emit_note(
        "interp",
        "methodology",
        "interp",
        "参考路径：解释器长循环满核负载会致降频，绝对值为热态、仅比值可用；主基线 jit/vm 已隔离测量",
    );

    let g6 = scenario_g6();
    let mut scs: Vec<&Scenario> = scenario_g1();
    // G6 的 interp 列：matmul_512（scenario_g6 第 3 项）。
    scs.push(g6[2]);

    for sc in scs {
        if let Some(reason) = sc.interp_skip {
            emit_na(sc.group, sc.name, Path::Interp, "min", "ms", reason);
            continue;
        }
        // 比值基准：本测试内先测 vm（interp 为参考路径，vm 先测仅为给出同态比值）。
        let (vm_st, vm_ck) = measure_path(sc, Path::Vm, None)
            .unwrap_or_else(|e| panic!("[{}/{}] vm(比值基准) 失败: {e}", sc.group, sc.name));
        let (in_st, _) = measure_path(sc, Path::Interp, Some(vm_ck))
            .unwrap_or_else(|e| panic!("[{}/{}] interp 失败: {e}", sc.group, sc.name));
        emit_stats(sc, Path::Interp, in_st);
        if vm_st.median_ms > 0.0 {
            emit(
                sc.group,
                sc.name,
                Path::Interp,
                "ratio_vs_vm_median",
                in_st.median_ms / vm_st.median_ms,
                "x",
            );
        }
        emit_note(
            sc.group,
            sc.name,
            "interp",
            "参考路径：绝对值为热态、仅比值可用",
        );
    }
}
