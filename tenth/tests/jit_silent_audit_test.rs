//! M2.6-P2：JIT 静默错值系统性审计套件。
//!
//! 目标：系统性穷举 JIT 边界组合，抓出分析期（`analyze_scalar_kinds`/`transfer`）
//! 与发射期（`emit_op`/`emit_direct_call`）漂移导致的**静默错值**（红线，
//! AUDIT-11.4.35 同族：分析预测 spec 而发射内联 → 循环回边 Load 读过期标量槽）。
//!
//! 审计维度（每维度对应一节测试）：
//! - D1 分析/发射一致性：逐 opcode 三类程序（内联/特化/通用）对拍
//! - D2 标量槽生命周期：Store/Load/Dup/Pop/块边界/循环回边/内联进出
//! - D3 三路径一致性：通用(Value)/内联(A2)/特化(A6) 路径程序对拍
//! - D4 错误路径：溢出/除零/取模零/范围 在三路径 + 循环/内联/特化/跳过切换后
//!       报错一致（消息 + 行号），不静默
//! - D5 组合穷举（重点）：i64 注解 × 参数个数(1-5) × 循环内调用 ×
//!       内联/不内联 × 累加/比较/嵌套 → 生成组合矩阵程序
//!
//! 方法：
//! - 进程内三路径对拍（VM / JIT / 解释器）：成功值逐字节一致 + 错误消息/行号
//!   一致；**断言 main 已 JIT 编译**（`is_compiled`，防 JIT 静默回退 VM 掩盖问题）
//! - 子进程字节级对拍：真实二进制默认路径（JIT） vs `TENTH_NO_VM=1`（解释器），
//!   断言 exit_code + stdout 逐字节一致

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
// 进程内 helper：编译 / 三路径执行 / 对拍断言
// ═══════════════════════════════════════════════════════════════════════════

fn lower(src: &str) -> Result<tenth::hir::hir::HirProgram, String> {
    let mut lexer = Lexer::new(src);
    let tokens = lexer.tokenize().map_err(|e| e.to_string())?;
    let mut parser = Parser::new(tokens);
    let program = parser.parse_program().map_err(|e| e.to_string())?;
    let mut lowerer = Lowerer::new();
    lowerer.lower_program(&program).map_err(|e| e.to_string())
}

/// 编译源码到 VM（含全部 natives；BytecodeCompiler 自动推导 scalar_sig）。
/// 镜像 main.rs vm_execute：全局名集合 + `__global_init` 先于 main 执行。
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
    // M3.5：程序级顶层 let 全局初始化（main 之前）。
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
    // main 优先用 `fn main`（已从 hir.functions 注册）；无 fn main 才用 main_expr
    // （镜像 main.rs vm_execute 的 has_fn 优先，防 main_expr 覆盖 fn main）。
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

/// 纯 VM 字节码路径（`vm.call`，不经 JIT）。
fn run_vm(src: &str) -> Result<Value, TenthError> {
    let mut vm = compile_vm(src)
        .map_err(|e| TenthError::RuntimeError { line: None, col: None, message: e })?;
    if vm.has_fn("main") {
        vm.call("main")
    } else {
        Ok(Value::Unit)
    }
}

/// JIT 路径执行 main（保留 Vm 供 is_compiled 断言；错误保留行号）。
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

/// main chunk 是否已 JIT 编译（防静默回退 VM）。
fn jit_compiled_main(vm: &Vm, label: &str) -> bool {
    let main_idx = vm.chunk_index_of("main").unwrap_or_else(|| panic!("[{label}] main chunk 未注册"));
    match vm.jit_ctx.as_ref() {
        Some(ctx) => ctx.is_compiled(main_idx),
        None => false,
    }
}

/// 从 TenthError 提取 (行号, 消息)。
fn err_parts(err: &TenthError) -> (Option<usize>, String) {
    match err {
        TenthError::RuntimeError { line, message, .. } => (*line, message.clone()),
        other => (None, format!("{:?}", other)),
    }
}

/// 三路径成功性 + 值/错误逐字节一致对拍。`require_compiled` = 断言 main 已 JIT 编译。
///
/// 这是审计核心：若 JIT 与 VM/解释器任一不一致 → 潜在静默错值，立即 panic 暴露。
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

    // 成功性一致
    let okness = |r: &Result<Value, TenthError>| r.is_ok();
    assert_eq!(okness(&vm_res), okness(&jit_res),
        "[{label}] VM/JIT 成功性不一致\nVM: {:?}\nJIT: {:?}", vm_res, jit_res);
    assert_eq!(okness(&vm_res), okness(&interp_res),
        "[{label}] VM/解释器 成功性不一致\nVM: {:?}\nInterp: {:?}", vm_res, interp_res);

    match (&vm_res, &jit_res, &interp_res) {
        (Ok(a), Ok(b), Ok(c)) => {
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

/// 取 Int 值（供少数已知值 sanity 断言）。
fn int_of(v: Value, label: &str) -> i64 {
    match v { Value::Int(n, _) => n, other => panic!("[{label}] 期望 Int，实际 {:?}", other) }
}

// ═══════════════════════════════════════════════════════════════════════════
// D5：组合矩阵程序生成器
// ═══════════════════════════════════════════════════════════════════════════
//
// 触发三条 JIT 路径的形态（与 emit_direct_call 顺序一致：内联 A2 → 特化 A6 → 通用 A1）：
// - "inline" ：小 i64 函数（≤16 指令纯标量）→ 内联优先（分析预测 Unknown）
// - "spec"   ：大 i64 函数（>16 指令纯标量）→ 不内联 → 特化 ABI
// - "general"：大 Int 函数（无 scalar_sig、>16 指令）→ 不内联、不特化 → 通用 A1

fn i64_params(k: usize) -> String {
    (0..k).map(|i| format!("x{i}: i64")).collect::<Vec<_>>().join(", ")
}

fn int_params(k: usize) -> String {
    (0..k).map(|i| format!("x{i}: Int")).collect::<Vec<_>>().join(", ")
}

fn sum_body(k: usize) -> String {
    (0..k).map(|i| format!("x{i}")).collect::<Vec<_>>().join(" + ")
}

/// 大纯标量体（>16 指令、**线性增长**）：乘/除配对抵消增长，结果 ≈ base + 1。
/// 用途：非内联资格（>16 指令）且不因迭代累加意外溢出。
/// base = x0 + x1（k≥2）/ x0 + 1（k==1）。
fn big_slow_body(k: usize) -> String {
    let base = if k >= 2 { "x0 + x1" } else { "x0 + 1" };
    let mut body = String::new();
    body.push_str(&format!("let a = {base};\n"));
    body.push_str("let b = a * 2;\n");
    body.push_str("let c = b / 2;\n");
    body.push_str("let d = c + 5;\n");
    body.push_str("let e = d * 3;\n");
    body.push_str("let g = e / 3;\n");
    body.push_str("let h = g - 5;\n");
    body.push_str("let j = h * 4;\n");
    body.push_str("let k = j / 4;\n");
    body.push_str("let m = k + 100;\n");
    body.push_str("let n = m * 5;\n");
    body.push_str("let p = n / 5;\n");
    body.push_str("let q = p - 100;\n");
    body.push_str("let r = q + 1;\n");
    body.push_str("let s2 = r * 6;\n");
    body.push_str("let t = s2 / 6;\n");
    body.push_str("let u = t + 7;\n");
    body.push_str("let v = u * 2;\n");
    body.push_str("let w = v / 2;\n");
    body.push_str("let x2 = w - 7;\n");
    // 额外参数参与（防参数错位；线性相加不放大）
    for i in 2..k {
        body.push_str(&format!("let y{i} = x2 + x{i};\n"));
    }
    let final_expr = if k >= 3 {
        body.push_str(&format!("let z = x2 + x{};\n", k - 1));
        "z + 1".to_string()
    } else {
        "x2 + 1".to_string()
    };
    body.push_str(&final_expr);
    body
}

/// 大纯标量体，含对 `inner`（小 i64 函数）的调用（内联嵌套在特化/通用体内）。
/// 线性增长：inner(x0,2)=2x0、inner(x0,3)=3x0 → c=5x0 → 乘除配对 → 结果 ≈ x0+5。
fn nested_slow_body(k: usize) -> String {
    let mut body = String::new();
    body.push_str("let a = inner(x0, 2);\n");
    body.push_str("let b = inner(x0, 3);\n");
    body.push_str("let c = a + b;\n");
    body.push_str("let d = c / 5;\n");
    body.push_str("let e = d + 1;\n");
    body.push_str("let g = e * 2;\n");
    body.push_str("let h = g / 2;\n");
    body.push_str("let j = h + 1;\n");
    body.push_str("let m = j * 3;\n");
    body.push_str("let n = m / 3;\n");
    body.push_str("let p = n + 1;\n");
    body.push_str("let q = p * 4;\n");
    body.push_str("let r = q / 4;\n");
    body.push_str("let s2 = r + 1;\n");
    body.push_str("let t = s2 * 5;\n");
    body.push_str("let u = t / 5;\n");
    body.push_str("let v = u + 1;\n");
    body.push_str("let w = v * 2;\n");
    body.push_str("let z2 = w / 2;\n");
    for i in 1..k {
        body.push_str(&format!("let y{i} = z2 + x{i};\n"));
    }
    let final_expr = if k > 1 {
        body.push_str(&format!("let zz = z2 + x{};\n", k - 1));
        "zz".to_string()
    } else {
        "z2".to_string()
    };
    body.push_str(&final_expr);
    body
}

/// main 的循环累加实参（恰 k 个）：k=1 → `s`；k=2 → `s, i`；k≥3 → `s, i, 1, 2...`。
fn accum_args(k: usize) -> Vec<String> {
    let mut args = vec!["s".to_string()];
    if k >= 2 {
        args.push("i".to_string());
    }
    let mut c = 1;
    while args.len() < k {
        args.push(c.to_string());
        c += 1;
    }
    args
}

/// main 的循环累加体源码段。
fn accum_main(k: usize, n: usize) -> String {
    let args = accum_args(k).join(", ");
    format!(
        "let mut s = 0;\n    let mut i = 0;\n    while i < {n} {{\n        s = f({args});\n        i = i + 1;\n    }};\n    s"
    )
}

/// D5 组合程序。shape ∈ {"accum", "compare", "nested"}。
fn combo_program(path: &str, k: usize, shape: &str, n: usize) -> String {
    match (path, shape) {
        ("inline", "accum") => format!(
            "fn f({}) -> i64 {{ {} }}\nfn main() -> i64 {{\n    {}\n}}",
            i64_params(k), sum_body(k), accum_main(k, n)),
        ("inline", "compare") => {
            // f(i, 1000, fill...) = i < 1000 → 1/0；循环累加（恰 k 个实参）
            let mut args = vec!["i".to_string(), "1000".to_string()];
            let mut c = 7;
            while args.len() < k {
                args.push(c.to_string());
                c += 1;
            }
            let args = args.join(", ");
            let cond = if k >= 2 { "x0 < x1" } else { "x0 < 1000" };
            format!(
                "fn f({}) -> i64 {{ if {cond} {{ 1 }} else {{ 0 }} }}\n\
                 fn main() -> i64 {{\n    let mut s = 0;\n    let mut i = 0;\n    while i < {n} {{\n        s = s + f({args});\n        i = i + 1;\n    }};\n    s\n}}",
                i64_params(k))
        }
        ("spec", "accum") => format!(
            "fn f({}) -> i64 {{\n{}\n}}\nfn main() -> i64 {{\n    {}\n}}",
            i64_params(k), big_slow_body(k), accum_main(k, n)),
        ("spec", "nested") => format!(
            "fn inner(a: i64, b: i64) -> i64 {{ a * b }}\n\
             fn f({}) -> i64 {{\n{}\n}}\nfn main() -> i64 {{\n    {}\n}}",
            i64_params(k), nested_slow_body(k), accum_main(k, n)),
        ("general", "accum") => format!(
            "fn f({}) -> Int {{\n{}\n}}\nfn main() -> Int {{\n    {}\n}}",
            int_params(k), big_slow_body(k), accum_main(k, n)),
        ("general", "nested") => format!(
            "fn inner(a: i64, b: i64) -> i64 {{ a * b }}\n\
             fn f({}) -> Int {{\n{}\n}}\nfn main() -> Int {{\n    {}\n}}",
            int_params(k), nested_slow_body(k), accum_main(k, n)),
        _ => panic!("未知组合: path={path} shape={shape}"),
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// D5：组合穷举矩阵（i64 × 参数个数 1-5 × 循环调用 × 内联/特化/通用 × 累加/比较/嵌套）
// ═══════════════════════════════════════════════════════════════════════════

/// 组合矩阵三路径对拍 + **路径化覆盖断言**（防静默回退 VM 掩盖问题）：
/// - inline：`f` 被内联进 main → 通用入口**不应**单独编译（`!is_compiled` 反而
///   证明内联生效；若内联失败回退 A1/spec，f 会被单独编译 → 断言失败暴露）
/// - spec：`f` 特化入口必须已编译（spec ABI 生效）、未失败
/// - general：`f` 通用入口必须已编译（A1 直接调用生效）
fn assert_combo(path: &str, k: usize, shape: &str, n: usize) {
    let src = combo_program(path, k, shape, n);
    let label = format!("{path}-{shape}-k{k}");
    assert_three_consistent(&src, &label, true);
    if let Ok((_, vm)) = run_jit_with_vm(&src) {
        let f_idx = vm.chunk_index_of("f")
            .unwrap_or_else(|| panic!("[{label}] f chunk 未注册"));
        let ctx = vm.jit_ctx.as_ref().expect("[{label}] JIT 上下文缺失");
        match path {
            "inline" => {
                assert!(!ctx.is_compiled(f_idx),
                    "[{label}] f 被单独编译——内联未生效（走 A1/spec），审计未覆盖内联路径");
            }
            "spec" => {
                assert!(ctx.is_spec_compiled(f_idx),
                    "[{label}] f 特化入口未编译——spec ABI 未生效（审计盲区）");
                assert!(!ctx.is_spec_failed(f_idx),
                    "[{label}] f 特化编译失败（fallback 通用）");
            }
            "general" => {
                assert!(ctx.is_compiled(f_idx),
                    "[{label}] f 未 JIT 编译（整函数回退 VM）——审计未覆盖 JIT 调用路径");
            }
            _ => panic!("未知 path: {path}"),
        }
        if shape == "nested" {
            let inner_idx = vm.chunk_index_of("inner")
                .unwrap_or_else(|| panic!("[{label}] inner chunk 未注册"));
            // inner 是内联资格 → 被内联进 f 体 → 不应单独编译（证明嵌套内联生效）
            assert!(!ctx.is_compiled(inner_idx),
                "[{label}] inner 被单独编译——嵌套内联未生效");
        }
    }
}

#[test]
fn audit_d5_matrix_inline_accum() {
    // 内联（A2）优先路径：小 i64 函数 + 循环内调用 + 多参（11.4.35 同形态回归）。
    // 注意：11.4.35 的根因就是「分析预测 spec 而发射内联」——内联形态是本审计
    // 的重中之重（分析期必须预测 Unknown，循环回边 Load 不得读过期标量槽）。
    for k in 1..=5usize {
        assert_combo("inline", k, "accum", 1000);
    }
}

#[test]
fn audit_d5_matrix_inline_compare() {
    // 内联 + 比较 + 分支 + 循环回边（f 的结果参与 if 条件与累加）。
    // f(i,1000) 在 i<1000 时为 1 → n=1500 时 s=1000（已知值 sanity）。
    for k in 1..=5usize {
        assert_combo("inline", k, "compare", 1500);
    }
    // sanity：已知值 1000（全部路径一致）
    let src = combo_program("inline", 3, "compare", 1500);
    assert_eq!(int_of(run_interp(&src).unwrap(), "sanity"), 1000);
}

#[test]
fn audit_d5_matrix_spec_accum() {
    // 特化（A6）路径：大 i64 函数（>16 指令不内联）+ 循环内调用 + 实参全标量。
    // 触发 spec_target_qualifies + try_spec_call + skip_chunk 判定。
    for k in 1..=5usize {
        assert_combo("spec", k, "accum", 500);
    }
}

#[test]
fn audit_d5_matrix_spec_nested() {
    // 特化体内嵌内联调用：f(spec) 体内调用 inner（小 i64 → 内联）。
    // 组合：spec 外层 + 内联内层 + 循环回边。
    for k in 1..=5usize {
        assert_combo("spec", k, "nested", 300);
    }
}

#[test]
fn audit_d5_matrix_general_accum() {
    // 通用（A1）路径：大 Int 函数（无 scalar_sig）+ 循环内调用。
    for k in 1..=5usize {
        assert_combo("general", k, "accum", 500);
    }
}

#[test]
fn audit_d5_matrix_general_nested() {
    // 通用体内嵌内联调用：outer(general) 调 inner(i64 小 → 内联)。
    for k in 1..=5usize {
        assert_combo("general", k, "nested", 300);
    }
}

#[test]
fn audit_d5_matrix_large_iteration() {
    // 大迭代量（热路径形态）：内联/特化/通用各一个高迭代用例，防回边累积漂移。
    for (path, shape, n) in [
        ("inline", "accum", 20000),
        ("spec", "accum", 10000),
        ("general", "accum", 10000),
        ("inline", "compare", 10000),
    ] {
        assert_combo(path, 3, shape, n);
    }
}

#[test]
fn audit_d5_spec_body_native_call_mix() {
    // **对抗形态 A**：spec 函数体内含通用 native 方法调用（to_string + .len()）。
    // spec_body_pure_scalar 判定应为 false（MethodCall 不纯）→ 不跳过 chunk 切换；
    // spec 入口 + 通用 hostcall + 标量续算 + 返回值解包组合。
    assert_three_consistent(r#"
        fn f(x: i64) -> i64 {
            let a = x * 2;
            let b = a + 1;
            let c = b * 3;
            let d = c / 3;
            let e = d + 1;
            let g = e * 4;
            let h = g / 4;
            let j = h + 1;
            let m = j * 5;
            let n = m / 5;
            let p = n + 1;
            let q = p * 6;
            let r = q / 6;
            let s2 = r + 1;
            let t = s2 * 2;
            let u = t / 2;
            let v = u + 1;
            let w = v * 2;
            let z = w / 2;
            let len = to_string(z).len();
            z + len
        }
        fn main() -> i64 { f(123) }
    "#, "d5-spec-body-native", true);
    // sanity：z = 2*123+6 = 252，to_string(252).len()=3 → 255
    let src = r#"
        fn f(x: i64) -> i64 {
            let a = x * 2;
            let b = a + 1;
            let c = b * 3;
            let d = c / 3;
            let e = d + 1;
            let g = e * 4;
            let h = g / 4;
            let j = h + 1;
            let m = j * 5;
            let n = m / 5;
            let p = n + 1;
            let q = p * 6;
            let r = q / 6;
            let s2 = r + 1;
            let t = s2 * 2;
            let u = t / 2;
            let v = u + 1;
            let w = v * 2;
            let z = w / 2;
            let len = to_string(z).len();
            z + len
        }
        fn main() -> i64 { f(123) }
    "#;
    assert_eq!(int_of(run_interp(&src).unwrap(), "d5-spec-native"), 255);
}

#[test]
fn audit_d5_spec_body_internal_loop() {
    // **对抗形态 B**：spec 函数体内含 while 循环（spec 入口 + 内部回边 + 累加器）。
    // 分析 seed 使 x 为 I32；循环内 s/i 为 I32 标量 → 特化体内的标量槽跨回边。
    assert_three_consistent(r#"
        fn f(x: i64) -> i64 {
            let mut s = 0;
            let mut i = 0;
            while i < x {
                s = s + i;
                i = i + 1;
            };
            s
        }
        fn main() -> i64 {
            let mut t = 0;
            let mut k = 0;
            while k < 5 {
                t = t + f(k * 10);
                k = k + 1;
            };
            t
        }
    "#, "d5-spec-internal-loop", true);
    // sanity：f(n)=n(n-1)/2；f(0)+f(10)+f(20)+f(30)+f(40) = 0+45+190+435+780 = 1450
    let src = r#"
        fn f(x: i64) -> i64 {
            let mut s = 0;
            let mut i = 0;
            while i < x {
                s = s + i;
                i = i + 1;
            };
            s
        }
        fn main() -> i64 {
            let mut t = 0;
            let mut k = 0;
            while k < 5 {
                t = t + f(k * 10);
                k = k + 1;
            };
            t
        }
    "#;
    assert_eq!(int_of(run_interp(&src).unwrap(), "d5-spec-loop"), 1450);
}

#[test]
fn audit_d5_general_body_spec_calls() {
    // **对抗形态 C**：通用（Int）函数体内嵌 spec 调用（实参为 I32 标量 → 特化 ABI）。
    // 通用入口 + 特化内层 + 标量生命周期交错。
    assert_three_consistent(r#"
        fn f(x: i64) -> i64 {
            let a = x * 2;
            let b = a + 1;
            let c = b * 3;
            let d = c / 3;
            let e = d + 1;
            let g = e * 4;
            let h = g / 4;
            let j = h + 1;
            let m = j * 5;
            let n = m / 5;
            let p = n + 1;
            let q = p * 6;
            let r = q / 6;
            let s2 = r + 1;
            let t = s2 * 2;
            let u = t / 2;
            let v = u + 1;
            let w = v * 2;
            let z = w / 2;
            z * 2
        }
        fn g(_x: Int) -> Int {
            let t = 5;
            let a = f(t);
            let b = f(t + 1);
            a + b
        }
        fn main() -> Int { g(3) }
    "#, "d5-general-spec-calls", true);
    // sanity：f(n)=4n+12；f(5)=32, f(6)=36 → 68
    let src = r#"
        fn f(x: i64) -> i64 {
            let a = x * 2;
            let b = a + 1;
            let c = b * 3;
            let d = c / 3;
            let e = d + 1;
            let g = e * 4;
            let h = g / 4;
            let j = h + 1;
            let m = j * 5;
            let n = m / 5;
            let p = n + 1;
            let q = p * 6;
            let r = q / 6;
            let s2 = r + 1;
            let t = s2 * 2;
            let u = t / 2;
            let v = u + 1;
            let w = v * 2;
            let z = w / 2;
            z * 2
        }
        fn g(_x: Int) -> Int {
            let t = 5;
            let a = f(t);
            let b = f(t + 1);
            a + b
        }
        fn main() -> Int { g(3) }
    "#;
    assert_eq!(int_of(run_interp(&src).unwrap(), "d5-general-spec"), 68);
}

// ═══════════════════════════════════════════════════════════════════════════
// D3：三路径一致性（专门触发单一路径的程序）
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn audit_d3_pure_scalar_general_path() {
    // 纯标量但 >16 指令的 Int 函数 → 通用 A1（无 spec 签名）。多语句 + 局部。
    assert_three_consistent(r#"
        fn compute(x: Int, y: Int) -> Int {
            let a = x + y;
            let b = a * 2;
            let c = b - x;
            let d = c + y;
            let e = d * 3;
            let g = e / 2;
            let h = g + a;
            let j = h * 4;
            let k = j - 5;
            let m = k + x;
            let n = m * 2;
            let p = n / 3;
            p + a + d + h
        }
        fn main() -> Int { compute(10, 4) }
    "#, "d3-general-big", true);
}

#[test]
fn audit_d3_mutual_recursion() {
    // 互递归（非内联、非特化 → A1 通用直接调用）——调用路径跨 JIT 帧。
    assert_three_consistent(r#"
        fn is_even(n: Int) -> Int {
            if n == 0 { 1 } else { is_odd(n - 1) }
        }
        fn is_odd(n: Int) -> Int {
            if n == 0 { 0 } else { is_even(n - 1) }
        }
        fn main() -> Int { is_even(10) * 100 + is_odd(7) }
    "#, "d3-mutual-recursion", true);
}

#[test]
fn audit_d3_string_return_mixed() {
    // 混合：内联算术 + 通用字符串函数（PushStr 不内联 → A1）+ 循环。
    // 字符串结果跨 JIT 帧（current_chunk_idx 切换路径）。
    assert_three_consistent(r#"
        fn fmt(x: Int) -> str { "v" + to_string(x) }
        fn main() -> Int {
            let mut sum = 0;
            let mut i = 0;
            while i < 40 {
                let s = fmt(i);
                sum = sum + s.len();
                i = i + 1;
            };
            sum
        }
    "#, "d3-string-mixed", true);
}

// ═══════════════════════════════════════════════════════════════════════════
// D2：标量槽生命周期（Store/Load/Dup/Pop/块边界/循环回边/内联进出）
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn audit_d2_store_load_backedge_i64() {
    // 循环回边 Store/Load：i64 局部在回边处被分析重置为 I32，标量槽必须新鲜
    // （11.4.35 的精确形态：回边 Load 读过期标量槽 → 静默 0）。
    assert_three_consistent(r#"
        fn main() -> i64 {
            let mut s = 0;
            let mut i = 0;
            while i < 1000 {
                s = s + i;
                i = i + 1;
            };
            s
        }
    "#, "d2-backedge-i64", true);
    // sanity：1+...+999 = 499500
    let src = r#"
        fn main() -> i64 {
            let mut s = 0;
            let mut i = 0;
            while i < 1000 {
                s = s + i;
                i = i + 1;
            };
            s
        }
    "#;
    assert_eq!(int_of(run_interp(src).unwrap(), "d2-backedge"), 499500);
}

#[test]
fn audit_d2_store_general_then_load() {
    // 通用 Store 后 Load（标量槽必须失效——A2 修复的通用路径安全网）。
    // s 经字符串函数（通用）写回后再次 Load 参与算术。
    assert_three_consistent(r#"
        fn touch(x: Int) -> str { to_string(x) }
        fn main() -> Int {
            let mut s = 10;
            let t = touch(s).len();
            s = s + t;
            let u = s * 2;
            let v = u - 3;
            v
        }
    "#, "d2-general-store-load", true);
}

#[test]
fn audit_d2_dup_pop_block_boundary() {
    // 块边界 + 重复使用局部（触发 Dup/Load 组合）+ if/else 合并。
    assert_three_consistent(r#"
        fn main() -> Int {
            let mut a = 5;
            let b = a + a;
            let c = b * 2;
            let mut d = 0;
            if c > 10 {
                d = c - a;
            } else {
                d = c + a;
            };
            let e = d + b;
            let f = e / 3;
            f
        }
    "#, "d2-block-boundary", true);
    // sanity：a=5, b=10, c=20, d=15, e=25, f=8
    let src = r#"
        fn main() -> Int {
            let mut a = 5;
            let b = a + a;
            let c = b * 2;
            let mut d = 0;
            if c > 10 {
                d = c - a;
            } else {
                d = c + a;
            };
            let e = d + b;
            let f = e / 3;
            f
        }
    "#;
    assert_eq!(int_of(run_interp(src).unwrap(), "d2-block"), 8);
}

#[test]
fn audit_d2_inline_enter_exit_caller_locals() {
    // 内联体进出：调用方局部标量在内联点前后保持新鲜（内联体禁用标量、进出恢复）。
    assert_three_consistent(r#"
        fn inc(x: i64) -> i64 { x + 1 }
        fn main() -> i64 {
            let mut s = 0;
            let mut i = 0;
            while i < 500 {
                let t = inc(i);
                s = s + t;
                i = i + 1;
            };
            s
        }
    "#, "d2-inline-enter-exit", true);
    // sanity：sum(inc(i)) = sum(i+1), i=0..499 = 1+...+500 = 125250
    let src = r#"
        fn inc(x: i64) -> i64 { x + 1 }
        fn main() -> i64 {
            let mut s = 0;
            let mut i = 0;
            while i < 500 {
                let t = inc(i);
                s = s + t;
                i = i + 1;
            };
            s
        }
    "#;
    assert_eq!(int_of(run_interp(src).unwrap(), "d2-inline"), 125250);
}

#[test]
fn audit_d2_float_scalar_lifecycle() {
    // 浮点标量槽生命周期：F64 局部 + 循环 + 混合 int/float 边界。
    assert_three_consistent(r#"
        fn scale(x: Float, k: Float) -> Float { x * k }
        fn main() -> Float {
            let mut s = 0.0;
            let mut i = 0;
            while i < 200 {
                s = scale(s + 1.0, 1.0);
                i = i + 1;
            };
            s
        }
    "#, "d2-float-lifecycle", true);
    // sanity：每轮 s = (s+1)*1 = s+1 → 200.0
    let src = r#"
        fn scale(x: Float, k: Float) -> Float { x * k }
        fn main() -> Float {
            let mut s = 0.0;
            let mut i = 0;
            while i < 200 {
                s = scale(s + 1.0, 1.0);
                i = i + 1;
            };
            s
        }
    "#;
    let v = run_interp(src).unwrap();
    match v { Value::Float(f) => assert!((f - 200.0).abs() < 1e-9, "期望 200.0，实际 {f}"), o => panic!("期望 Float，实际 {o:?}") }
}

#[test]
fn audit_d2_char_bool_kinds() {
    // Char（I32 标量槽）/Bool 种类混合 + 分支 + 循环。
    assert_three_consistent(r#"
        fn main() -> Int {
            let mut s = 0;
            let mut i = 0;
            while i < 100 {
                let cond = i % 2 == 0;
                if cond {
                    s = s + 1;
                } else {
                    s = s + 2;
                };
                i = i + 1;
            };
            s
        }
    "#, "d2-char-bool", true);
    // sanity：50 偶 + 50 奇 → 50*1 + 50*2 = 150
    let src = r#"
        fn main() -> Int {
            let mut s = 0;
            let mut i = 0;
            while i < 100 {
                let cond = i % 2 == 0;
                if cond {
                    s = s + 1;
                } else {
                    s = s + 2;
                };
                i = i + 1;
            };
            s
        }
    "#;
    assert_eq!(int_of(run_interp(src).unwrap(), "d2-char-bool"), 150);
}

// ═══════════════════════════════════════════════════════════════════════════
// D4：错误路径（溢出/除零/取模零/范围——三路径一致，不静默）
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn audit_d4_overflow_inline_loop() {
    // 内联小函数 + 循环累加 → 溢出：JIT 必须报错（不得静默 0/回绕），行号一致。
    // 11.4.35 的原型：3 参 i64 小函数循环，应报「溢出 i32 范围」。
    assert_three_consistent(r#"
        fn f(a: i64, b: i64, c: i64) -> i64 { a + b + c }
        fn main() -> i64 {
            let mut s = 0;
            let mut i = 0;
            while i < 1000000 {
                s = f(s, i, 1);
                i = i + 1;
            };
            s
        }
    "#, "d4-overflow-inline-loop", true);
}

#[test]
fn audit_d4_div_zero_inline() {
    // 除零在内联体内：报错不静默，消息 + 行号三路径一致。
    assert_three_consistent(r#"
        fn div(a: i64, b: i64) -> i64 { a / b }
        fn main() -> i64 {
            let x = 10;
            let y = 0;
            div(x, y)
        }
    "#, "d4-div-zero-inline", true);
}

#[test]
fn audit_d4_mod_zero_spec() {
    // 取模零在特化函数体内（spec + 错误路径组合）；体线性增长防溢出掩盖。
    assert_three_consistent(r#"
        fn f(x: i64) -> i64 {
            let a = x + 1;
            let b = a * 2;
            let c = b / 2;
            let d = c + 5;
            let e = d * 3;
            let g = e / 3;
            let h = g - 5;
            let j = h * 4;
            let k = j / 4;
            let m = k + 100;
            let n = m * 5;
            let p = n / 5;
            let q = p - 100;
            let r = q + 1;
            let s2 = r * 6;
            let t = s2 / 6;
            let u = t + 7;
            let v = u * 2;
            let w = v / 2;
            let x2 = w - 7;
            x2 % 0
        }
        fn main() -> i64 { f(5) }
    "#, "d4-mod-zero-spec", true);
}

#[test]
fn audit_d4_overflow_spec_skip_chunk() {
    // 特化 + skip_chunk（纯标量体内不切换 chunk）+ 循环 → 溢出。
    // 倍增体 f(x)=4x+12（线性中间 + 末乘 2）：~16 轮过 2^31。
    // 覆盖「跳过 current_chunk_idx 切换后错误路径仍正确（B2 红线）」。
    assert_three_consistent(r#"
        fn f(x: i64) -> i64 {
            let a = x * 2;
            let b = a + 1;
            let c = b * 3;
            let d = c / 3;
            let e = d + 1;
            let g = e * 4;
            let h = g / 4;
            let j = h + 1;
            let m = j * 5;
            let n = m / 5;
            let p = n + 1;
            let q = p * 6;
            let r = q / 6;
            let s2 = r + 1;
            let t = s2 * 2;
            let u = t / 2;
            let v = u + 1;
            let w = v * 2;
            let z = w / 2;
            z * 2
        }
        fn main() -> i64 {
            let mut s = 1;
            let mut i = 0;
            while i < 30 {
                s = f(s);
                i = i + 1;
            };
            s
        }
    "#, "d4-overflow-spec-skip", true);
}

#[test]
fn audit_d4_div_zero_loop_general() {
    // 除零在循环内（通用路径 + 回边），i 到某值触发——报错不得延迟/静默。
    assert_three_consistent(r#"
        fn big(x: Int) -> Int {
            let a = x + 1;
            let b = a * 2;
            let c = b - 3;
            let d = c + 4;
            let e = d * 5;
            let g = e / 6;
            let h = g + 7;
            let j = h * 8;
            let k = j - 9;
            let m = k + 10;
            let n = m * 2;
            n / (x - 10)
        }
        fn main() -> Int {
            let mut s = 0;
            let mut i = 0;
            while i < 30 {
                s = big(i);
                i = i + 1;
            };
            s
        }
    "#, "d4-div-zero-loop-general", true);
}

#[test]
fn audit_d4_neg_overflow() {
    // Neg 溢出（-i32::MIN = 2147483648 超出 i32 范围）在特化体内——不静默。
    // 中性体（乘除配对）u == x，最后 -u 触发范围溢出。
    assert_three_consistent(r#"
        fn f(x: i64) -> i64 {
            let a = x + 1;
            let b = a - 1;
            let c = b * 2;
            let d = c / 2;
            let e = d + 3;
            let g = e - 3;
            let h = g * 4;
            let j = h / 4;
            let m = j + 5;
            let n = m - 5;
            let p = n * 6;
            let q = p / 6;
            let r = q + 7;
            let s2 = r - 7;
            let t = s2 * 2;
            let u = t / 2;
            -u
        }
        fn main() -> i64 { f(0 - 2147483648) }
    "#, "d4-neg-overflow", true);
}

#[test]
fn audit_d4_error_line_parity_after_skip() {
    // skip_chunk 特化 + 错误行号：JIT 报错行号 == VM == 解释器（逐字节）。
    // 行号来自 chunk 行号表（emit_line_hint 写 vm.current_line）。
    let src = r#"
fn g(a: i64) -> i64 {
    let x = a + 1;
    let y = x * 2;
    let z = y + a;
    let w = z * 3;
    let v = w / 2;
    let u = v + 1;
    let t = u * 4;
    let s = t - 2;
    let r = s + a;
    let q = r * 2;
    let p = q / 3;
    p / (a - 3)
}
fn main() -> i64 {
    let mut s = 0;
    let mut i = 0;
    while i < 20 {
        s = g(i);
        i = i + 1;
    };
    s
}
"#;
    // 三路径均报错且行号一致（i=3 时 a-3=0 → 第 13 行除零）
    assert_three_consistent(src, "d4-line-parity-skip", true);
}

// ═══════════════════════════════════════════════════════════════════════════
// D1：分析/发射一致性（逐 opcode 覆盖的代表性程序）
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn audit_d1_opcode_core_scalar() {
    // 核心标量 opcode 全集合（PushInt/PushFloat/PushBool/PushChar/算术/比较/一元）
    // 在标量专用化启用下的对拍——分析期种类预测 vs 发射期原生/通用二选一。
    assert_three_consistent(r#"
        fn main() -> Int {
            let a = 10;
            let b = 3;
            let c = a + b;
            let d = a - b;
            let e = a * b;
            let g = a / b;
            let h = a % b;
            let n = -g;
            let m = n + c + d + e + h;
            let cmp = (a > b) as Int + (a == b) as Int + (a < b) as Int;
            m + cmp
        }
    "#, "d1-core-scalar", true);
}

#[test]
fn audit_d1_pushfloat32_char() {
    // PushFloat32（不专用化）+ PushChar（I32 标量槽）与整数/布尔混合。
    assert_three_consistent(r#"
        fn main() -> Int {
            let c = 'A';
            let f = 1.5f32;
            let i = c as Int;
            let r = i + 10;
            r
        }
    "#, "d1-float32-char", true);
}

#[test]
fn audit_d1_global_store_load() {
    // LoadGlobal/StoreGlobal（通用 hostcall，清栈语义）与标量局部交错。
    assert_three_consistent(r#"
        let mut g = 0;
        fn bump() -> Int {
            g = g + 1;
            g
        }
        fn main() -> Int {
            let a = bump();
            let b = bump();
            let c = a + b;
            let d = c * 2;
            d + g
        }
    "#, "d1-global", true);
}



#[test]
fn audit_d1_vec_tuple_struct_mix() {
    // 复合 opcode（MakeVec/MakeTuple/NewStruct 等通用 hostcall）后接标量算术——
    // 清栈/标量槽生命周期交错（A3 覆盖的 opcode 族的静默错值回归）。
    assert_three_consistent(r#"
        struct Point { x: Int, y: Int }
        fn main() -> Int {
            let p = Point { x: 3, y: 4 };
            let v = [1, 2, 3];
            let t = (5, 6);
            let x = p.x;
            let len = v.len();
            let tv = t.0;
            let r = x + len + tv;
            r
        }
    "#, "d1-compound", true);
}

// ═══════════════════════════════════════════════════════════════════════════
// 子进程字节级对拍：真实二进制 默认(JIT) vs TENTH_NO_VM=1(解释器)
// ═══════════════════════════════════════════════════════════════════════════
// 断言 exit_code + stdout + stderr 逐字节一致（覆盖 println/print 输出形态）。

use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};

const TENTH_EXE: &str = env!("CARGO_BIN_EXE_tenth");
const TENTH_DIR: &str = env!("CARGO_MANIFEST_DIR");
static COUNTER: AtomicU64 = AtomicU64::new(0);

/// 运行 .th 程序：`use_vm=true` → 默认路径（JIT），`false` → TENTH_NO_VM（解释器）。
fn run_th(prog: &str, use_vm: bool) -> (i32, String, String) {
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!("tenth_p2audit_{}_{}", std::process::id(), n));
    std::fs::create_dir_all(&dir).unwrap();
    let file = dir.join("p2.th");
    std::fs::write(&file, prog).unwrap();
    let mut cmd = Command::new(TENTH_EXE);
    cmd.arg("run").arg(&file).current_dir(TENTH_DIR);
    if !use_vm {
        cmd.env("TENTH_NO_VM", "1");
    }
    let out = cmd.output().expect("运行 tenth.exe 失败");
    let stdout = String::from_utf8_lossy(&out.stdout).to_string();
    let stderr = String::from_utf8_lossy(&out.stderr).to_string();
    let code = out.status.code().unwrap_or(-1);
    let _ = std::fs::remove_dir_all(&dir);
    (code, stdout, stderr)
}

/// 双路径字节级对拍：exit_code + stdout + stderr 全部逐字节相等。
fn assert_byte_parity(name: &str, prog: &str) {
    let (cj, sj, ej) = run_th(prog, true);
    let (ci, si, ei) = run_th(prog, false);
    assert_eq!(cj, ci, "[{name}] exit 不一致: JIT={cj} Interp={ci}\n--- stderr JIT ---\n{ej}\n--- stderr Interp ---\n{ei}\nprog:\n{prog}");
    assert_eq!(sj, si, "[{name}] stdout 不一致\n--- JIT ---\n{sj}\n--- Interp ---\n{si}\nprog:\n{prog}");
    assert_eq!(ej, ei, "[{name}] stderr 不一致\n--- JIT ---\n{ej}\n--- Interp ---\n{ei}\nprog:\n{prog}");
}

/// 从 stderr 提取核心错误（行号, 消息）。
/// 说明：CLI 对 VM 错误额外打印 3 行 `[error] ...` 引导 + 解释器带列号——这些是
/// 既有的 CLI 展示差异（非 JIT 静默错值），归一化到核心错误比较。
/// 错误格式（error.rs runtime_error_format）：`第 L 行[第 C 列]：运行时错误 — <msg>`。
fn core_err_parts(stderr: &str) -> (Option<usize>, String) {
    let line = stderr.lines().rev().find(|l| l.contains("Error:") || l.contains("error:")).unwrap_or("");
    let line_no = {
        let start = line.find("第 ").map(|i| i + 3).unwrap_or(0);
        let digits: String = line[start..].chars().take_while(|c| c.is_ascii_digit()).collect();
        if digits.is_empty() { None } else { digits.parse().ok() }
    };
    let msg = match line.find("运行时错误 — ") {
        Some(i) => line[i + "运行时错误 — ".len()..].to_string(),
        None => line.to_string(),
    };
    (line_no, msg)
}

/// 双路径错误路径对拍：exit 非 0 但两侧一致；stdout 逐字节；stderr 归一化后
/// 核心错误（行号 + 「消息」）一致（CLI 对 VM 错误有额外的 `[error]` 引导行）。
fn assert_byte_parity_err(name: &str, prog: &str) {
    let (cj, sj, ej) = run_th(prog, true);
    let (ci, si, ei) = run_th(prog, false);
    assert!(cj != 0, "[{name}] JIT 应报错，exit={cj}\nprog:\n{prog}");
    assert!(ci != 0, "[{name}] Interp 应报错，exit={ci}\nprog:\n{prog}");
    assert_eq!(cj, ci, "[{name}] exit 不一致: JIT={cj} Interp={ci}\nprog:\n{prog}");
    assert_eq!(sj, si, "[{name}] stdout 不一致\n--- JIT ---\n{sj}\n--- Interp ---\n{si}\nprog:\n{prog}");
    let (lj, mj) = core_err_parts(&ej);
    let (li, mi) = core_err_parts(&ei);
    assert_eq!(lj, li, "[{name}] 核心错误行号不一致 JIT={lj:?} Interp={li:?}\n--- stderr JIT ---\n{ej}\n--- stderr Interp ---\n{ei}\nprog:\n{prog}");
    assert_eq!(mj, mi, "[{name}] 核心错误消息不一致\nJIT={mj}\nInterp={mi}\n--- stderr JIT ---\n{ej}\n--- stderr Interp ---\n{ei}\nprog:\n{prog}");
}

#[test]
fn audit_subprocess_byte_parity_stdout() {
    // 内联 + 循环 + 多参：println 输出累加结果（11.4.35 形态的 stdout 级验证）。
    assert_byte_parity("inline-accum-3", r#"
fn f(a: i64, b: i64, c: i64) -> i64 { a + b + c }
fn main() -> i64 {
    let mut s = 0;
    let mut i = 0;
    while i < 1000 {
        s = f(s, i, 1);
        i = i + 1;
    };
    println("s = ", s);
    s
}
"#);
    // 特化 + 循环累加 + println（线性体防意外溢出）。
    assert_byte_parity("spec-accum-2", r#"
fn f(x0: i64, x1: i64) -> i64 {
    let a = x0 + x1;
    let b = a * 2;
    let c = b / 2;
    let d = c + 5;
    let e = d * 3;
    let g = e / 3;
    let h = g - 5;
    let j = h * 4;
    let k = j / 4;
    let m = k + 100;
    let n = m * 5;
    let p = n / 5;
    let q = p - 100;
    let r = q + 1;
    let s2 = r * 6;
    let t = s2 / 6;
    let u = t + 7;
    let v = u * 2;
    let w = v / 2;
    let x2 = w - 7;
    x2 + 1
}
fn main() -> i64 {
    let mut s = 0;
    let mut i = 0;
    while i < 300 {
        s = f(s, i);
        i = i + 1;
    };
    println("s = ", s);
    s
}
"#);
    // 通用 + 字符串混合 + println。
    assert_byte_parity("general-string", r#"
fn fmt(x: Int) -> str { "v" + to_string(x) }
fn main() -> Int {
    let mut sum = 0;
    let mut i = 0;
    while i < 40 {
        let s = fmt(i);
        sum = sum + s.len();
        i = i + 1;
    };
    println("sum = ", sum);
    sum
}
"#);
    // 浮点 + 循环 + println。
    assert_byte_parity("float-loop", r#"
fn scale(x: Float, k: Float) -> Float { x * k }
fn main() -> Float {
    let mut s = 0.0;
    let mut i = 0;
    while i < 100 {
        s = scale(s + 1.0, 1.0);
        i = i + 1;
    };
    println("s = ", s);
    s
}
"#);
}

#[test]
fn audit_subprocess_byte_parity_error() {
    // 错误路径 stdout/stderr/exit 逐字节一致：内联溢出。
    assert_byte_parity_err("inline-overflow", r#"
fn f(a: i64, b: i64, c: i64) -> i64 { a + b + c }
fn main() -> i64 {
    let mut s = 0;
    let mut i = 0;
    while i < 1000000 {
        s = f(s, i, 1);
        i = i + 1;
    };
    println("s = ", s);
    s
}
"#);
    // 错误路径：特化除零（线性体，i=5 时 x-5=0 触发）。
    assert_byte_parity_err("spec-divzero", r#"
fn f(x: i64) -> i64 {
    let a = x + 1;
    let b = a * 2;
    let c = b / 2;
    let d = c + 5;
    let e = d * 3;
    let g = e / 3;
    let h = g - 5;
    let j = h * 4;
    let k = j / 4;
    let m = k + 100;
    let n = m * 5;
    let p = n / 5;
    let q = p - 100;
    let r = q + 1;
    let s2 = r * 6;
    let t = s2 / 6;
    let u = t + 7;
    let v = u * 2;
    let w = v / 2;
    let x2 = w - 7;
    x2 / (x - 5)
}
fn main() -> i64 {
    let mut s = 0;
    let mut i = 0;
    while i < 20 {
        s = f(i);
        i = i + 1;
    };
    s
}
"#);
}

#[test]
fn audit_subprocess_byte_parity_nested_mixed() {
    // 嵌套 + 比较 + 布尔/字符：组合形态的 stdout 级验证。
    assert_byte_parity("nested-compare", r#"
fn inner(a: i64, b: i64) -> i64 { a * b + a }
fn f(x0: i64) -> i64 {
    let a = inner(x0, 2);
    let b = inner(2, x0);
    let c = a + b;
    let d = c * 2;
    let e = d + x0;
    let g = e / 2;
    let h = g * 3;
    let j = h - 1;
    let m = j + a;
    let n = m * 2;
    n + c + e + h + m
}
fn main() -> i64 {
    let mut s = 0;
    let mut i = 0;
    while i < 200 {
        let v = f(i);
        if v > 1000 {
            s = s + 1;
        } else {
            s = s + v % 10;
        };
        i = i + 1;
    };
    println("s = ", s);
    s
}
"#);
}
