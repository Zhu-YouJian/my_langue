//! AUDIT-11.4.85 回归守护：**默认（JIT）路径下「调用作第二操作数 + 循环」的静默错值**。
//!
//! ## 缺陷（修复前）
//!
//! 默认执行路径 = Cranelift JIT（`main.rs: vm_execute → jit::run_jit`），不是字节码 VM。
//! `translator.rs::emit_direct_call` 的 A1 通用直接调用有两条分支：
//!
//! * **快分支**：目标 chunk 已编译（`vm.jit_table_ptr[callee] != 0`）→ 池内 `call_indirect`；
//! * **慢分支**：`host_jit_call` trampoline（首次遇到未编译目标 → 编译 + 注册 + 调用）。
//!
//! 慢分支经 `call_hostcall_call` → `invalidate_stack_scalars()`：**在发射期**于慢块内
//! 发「把调用前压入的操作数（仍是懒物化标量）写成 Value」的 hostcall + 清空本块剩余
//! 栈标量跟踪；快分支不做同样的事。于是：
//!
//! * 首次求值该语句（目标未编译 → 慢分支）⇒ 操作数 Value 槽被正确物化 ⇒ **结果正确**；
//! * 之后（目标已编译 → 快分支）⇒ 该操作数的 Value 槽**陈旧**（上一轮/上一表达式残留）
//!   ⇒ 后续通用消费者（`emit_binop` 通用路径）读到陈旧 Value。
//!
//! 故触发条件 = **① 默认（JIT）路径 ② 调用是运算符的第二操作数 ③ 该语句被重复求值
//! （循环）④ 被调函数不可内联（含控制流／嵌套调用／指令数 > 16）⑤ 保持栈槽布局**。
//! 症状面（比较只是唯一响亮形态，其余全静默）：
//!
//! * `while i < f()` ⇒ 第 2 轮 `compare()` 兜底 →**「运行时错误 — 无法比较」**（exit 1）；
//! * `i + f()` ⇒ **静默错值** `2/4/6`（真值 `2/3/4`）；`i - f()` ⇒ `-2/-4/-6`；
//! * `i != f()` ⇒ **恒 `true`**；`let s = i + f()` ⇒ 报「+ 类型不匹配」（换了个响亮形态）。
//!
//! 解释器（`TENTH_NO_VM=1`）一直是正确的。
//!
//! ## 修法（本文件守护）
//!
//! `translator.rs::emit_direct_call`：把这次物化提到**分叉之前**的公共块
//! （`self.materialize_all_stack()`），令快/慢两分支运行期内存状态一致，且与
//! `analyze_scalar_kinds` 对非特化 `Call/CallN` 的建模（`push(Unknown) + clear_stack`）
//! 一致。慢块内的 `invalidate_stack_scalars` 随后成为 no-op（跟踪已空，零重复发射）。
//!
//! ## 守护方式（两层，互补）
//!
//! 1. **子进程字节级两路径对拍**（真实二进制：默认 JIT vs `TENTH_NO_VM=1` 解释器）：
//!    断言 exit 码、stdout 逐字节、逐行内容、stderr 为空。夹具是**原样**的最小复现，
//!    保持原栈槽布局（不加任何额外语句，以免「布局变化掩盖缺陷」——审计 §2 遗留 1）。
//! 2. **进程内 JIT 覆盖断言**：`is_compiled`/`is_failed` 保证「默认路径真的走了 JIT 且
//!    被调函数真的注册进了函数指针表（快分支可达）」，防「JIT 静默回退 VM」把问题掩盖。
//!
//! 参考：`.agents/tmp/prep_audit_11485.md`（触发面 28 行收敛表）、`AUDIT.md` 的 `11.4.85`。

use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};

use tenth::compile::bytecode::BytecodeCompiler;
use tenth::compile::jit;
use tenth::hir::lower::Lowerer;
use tenth::lexer::lexer::Lexer;
use tenth::parser::parser::Parser;
use tenth::runtime::natives::register_all_natives;
use tenth::runtime::vm::Vm;

// ═══════════════════════════════════════════════════════════════════════════
// 夹具（单文件、无模块；与审计文档 §1 的最小复现逐字一致）
// ═══════════════════════════════════════════════════════════════════════════

/// `m1`：最小复现（15 行）。比较（响亮形态）。
const M1: &str = r#"fn f() -> i32 { let mut i = 0; while i < 2 { i = i + 1; }; 2 }
fn main() { let mut i = 0; while i < f() { println("k=" + format("{}", i)); i = i + 1; }; println("done"); }
"#;

/// `r1`：同一语句写两遍（`A=`/`B=`），循环 3 轮 —— 钉住「第一操作数读到上一表达式结果」。
const R1: &str = r#"fn f() -> i32 { let mut i = 0; while i < 2 { i = i + 1; }; 2 }
fn main() {
    let mut i = 0;
    while i < 3 {
        println("A=" + format("{}", i + f()));
        println("B=" + format("{}", i + f()));
        i = i + 1;
    };
    println("done");
}
"#;

/// `q1`：被调函数带 **1 个 i32 形参**（走 A6 特化 ABI 面；审计称此形态「更强」）。
const Q1: &str = r#"fn c2p(x: i32) -> i32 { let mut i = 0; while i < x { i = i + 1; }; 3 }
fn main() {
    let mut i = 0;
    while i < 3 { println("S=" + format("{}", i + c2p(1))); i = i + 1; };
    let mut k = 0;
    while k < c2p(1) { println("C=" + format("{}", k)); k = k + 1; };
    println("done");
}
"#;

/// 宽度形态：`i+f()` / `i-f()` / `i!=f()` / `let s = i+f()` 四种（比较只是响亮形态，
/// 其余是**静默错值**——本夹具是这条缺陷最重的证据面）。
const WIDE: &str = r#"fn f() -> i32 { let mut i = 0; while i < 2 { i = i + 1; }; 2 }

fn main() {
    let mut i = 0;
    while i < 4 {
        println("ADD=" + format("{}", i + f()));
        i = i + 1;
    };
    let mut j = 0;
    while j < 4 {
        println("SUB=" + format("{}", j - f()));
        j = j + 1;
    };
    let mut k = 0;
    while k < 4 {
        println("NEQ=" + format("{}", k != f()));
        k = k + 1;
    };
    let mut m = 0;
    while m < 4 {
        let s = m + f();
        println("BIND=" + format("{}", s));
        m = m + 1;
    };
    println("done");
}
"#;

// ═══════════════════════════════════════════════════════════════════════════
// 层 1：子进程字节级两路径对拍（真实二进制）
// ═══════════════════════════════════════════════════════════════════════════

const TENTH_EXE: &str = env!("CARGO_BIN_EXE_tenth");
const TENTH_DIR: &str = env!("CARGO_MANIFEST_DIR");
static COUNTER: AtomicU64 = AtomicU64::new(0);

/// 跑 `.th`：`use_vm=true` → 默认路径（JIT）；`false` → `TENTH_NO_VM=1`（解释器）。
fn run_th(prog: &str, use_vm: bool) -> (i32, String, String) {
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!("tenth_a11485_{}_{}", std::process::id(), n));
    std::fs::create_dir_all(&dir).unwrap();
    let file = dir.join("m.th");
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

/// 两路径对拍 + 期望行序列：默认路径 exit=0、stdout 与解释器**逐字节一致**、
/// 逐行等于 `expect_lines`、stderr 两路径皆空（修复前默认路径 exit=1 + 「无法比较」）。
fn assert_parity(name: &str, prog: &str, expect_lines: &[&str]) {
    let (cj, sj, ej) = run_th(prog, true);
    let (ci, si, ei) = run_th(prog, false);

    assert_eq!(cj, 0, "[{name}] 默认（JIT）路径应 exit 0，实际 {cj}\n--- stdout ---\n{sj}\n--- stderr ---\n{ej}\nprog:\n{prog}");
    assert_eq!(ci, 0, "[{name}] 解释器路径应 exit 0，实际 {ci}\n--- stdout ---\n{si}\n--- stderr ---\n{ei}\nprog:\n{prog}");
    assert_eq!(sj, si, "[{name}] 两路径 stdout 非逐字节一致\n--- JIT ---\n{sj}\n--- Interp ---\n{si}\nprog:\n{prog}");
    assert_eq!(ej, ei, "[{name}] 两路径 stderr 不一致\n--- JIT ---\n{ej}\n--- Interp ---\n{ei}\nprog:\n{prog}");
    let lines: Vec<&str> = sj.lines().collect();
    assert_eq!(lines, expect_lines, "[{name}] stdout 行序列不符\n--- 实际 ---\n{sj}\nprog:\n{prog}");
}

#[test]
fn audit_11485_m1_minimal_repro_parity() {
    // 审计 §1 最小复现：修复前默认路径 rc=1 + `k=0` + 「无法比较」。
    assert_parity("m1", M1, &["k=0", "k=1", "done"]);
}

#[test]
fn audit_11485_r1_two_same_exprs_in_loop() {
    // 修复前默认路径：`A=2/B=4/A=6/B=8/A=10/B=12`（第一操作数读到上一表达式结果）。
    assert_parity("r1", R1, &["A=2", "B=2", "A=3", "B=3", "A=4", "B=4", "done"]);
}

#[test]
fn audit_11485_q1_one_i32_arg_spec_call() {
    // 修复前默认路径：`S=3/6/9` + `C=0` + 「无法比较」。
    assert_parity("q1", Q1, &["S=3", "S=4", "S=5", "C=0", "C=1", "C=2", "done"]);
}

#[test]
fn audit_11485_wide_shape_add_second_operand() {
    // 修复前默认路径：`ADD=2/4/6/8`（**静默**错值；真值 2/3/4/5）。
    assert_parity(
        "wide-add",
        WIDE,
        &[
            "ADD=2", "ADD=3", "ADD=4", "ADD=5", "SUB=-2", "SUB=-1", "SUB=0", "SUB=1",
            "NEQ=true", "NEQ=true", "NEQ=false", "NEQ=true",
            "BIND=2", "BIND=3", "BIND=4", "BIND=5", "done",
        ],
    );
}

#[test]
fn audit_11485_shape_sub_second_operand() {
    // `i - f()`：修复前默认 `SUB=-2/-4/-6`。
    let src = r#"fn f() -> i32 { let mut i = 0; while i < 2 { i = i + 1; }; 2 }
fn main() { let mut i = 0; while i < 3 { println("SUB=" + format("{}", i - f())); i = i + 1; }; println("done"); }
"#;
    assert_parity("sub", src, &["SUB=-2", "SUB=-1", "SUB=0", "done"]);
}

#[test]
fn audit_11485_shape_neq_second_operand() {
    // `i != f()`：修复前默认**恒** `true`（真值 true/true/false）。
    let src = r#"fn f() -> i32 { let mut i = 0; while i < 2 { i = i + 1; }; 2 }
fn main() { let mut i = 0; while i < 3 { println("NEQ=" + format("{}", i != f())); i = i + 1; }; println("done"); }
"#;
    assert_parity("neq", src, &["NEQ=true", "NEQ=true", "NEQ=false", "done"]);
}

#[test]
fn audit_11485_shape_bind_local_second_operand() {
    // `let s = i + f()`：修复前默认报「+ 类型不匹配」（第 2 轮）。
    let src = r#"fn f() -> i32 { let mut i = 0; while i < 2 { i = i + 1; }; 2 }
fn main() { let mut i = 0; while i < 3 { let s = i + f(); println("BIND=" + format("{}", s)); i = i + 1; }; println("done"); }
"#;
    assert_parity("bind", src, &["BIND=2", "BIND=3", "BIND=4", "done"]);
}

// ═══════════════════════════════════════════════════════════════════════════
// 层 2：进程内 JIT 覆盖断言（防「JIT 静默回退 VM」掩盖回归）
// ═══════════════════════════════════════════════════════════════════════════

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
            }
            Err(e) => return Err(format!("compile error: {e}")),
        }
    }
    Ok(vm)
}

#[test]
fn audit_11485_jit_path_really_used_and_no_runtime_error() {
    // m1 原样夹具：修复前 `run_jit` 返回 Err（「无法比较」）；修复后 Ok(Unit)。
    // 同时断言 main 与 f **都**被 JIT 编译（f 注册进函数指针表 ⇒ 快分支可达），
    // 否则「默认路径静默回退 VM/只走慢分支」会让本用例变成假绿。
    let mut vm = compile_vm(M1).expect("编译失败");
    let r = jit::run_jit(&mut vm, "main");
    assert!(r.is_ok(), "JIT 路径应成功（修复前为「运行时错误 — 无法比较」）：{:?}", r.err());

    let ctx = vm.jit_ctx.as_ref().expect("JIT 上下文应存在");
    let main_idx = vm.chunk_index_of("main").expect("main chunk 应存在");
    let f_idx = vm.chunk_index_of("f").expect("f chunk 应存在");
    assert!(ctx.is_compiled(main_idx), "main 应被 JIT 编译（不得静默回退 VM）");
    assert!(ctx.is_compiled(f_idx), "f 应被按需 JIT 编译 ⇒ A1 快分支（call_indirect）可达");
    assert!(!ctx.is_failed(main_idx), "main 不应整函数 fallback");
    assert!(!ctx.is_failed(f_idx), "f 不应编译失败");
}

#[test]
fn audit_11485_jit_result_equals_interpreter_result() {
    // 取返回值形态（`m1` 的条件 + 计数），断言 JIT 路径结果与解释器一致。
    // 注：本形态只用于**值级**对拍；触发缺陷的布局敏感形态由层 1 的子进程对拍覆盖。
    let src = r#"fn f() -> i32 { let mut i = 0; while i < 2 { i = i + 1; }; 2 }
fn main() -> i32 {
    let mut i = 0;
    let mut n = 0;
    while i < f() { n = n + 1; i = i + 1; };
    n * 10 + i
}
"#;
    let mut vm_jit = compile_vm(src).expect("编译失败");
    let jit_v = jit::run_jit(&mut vm_jit, "main");
    let mut vm_interp = compile_vm(src).expect("编译失败");
    let interp_v = vm_interp.call("main");

    let jv = match jit_v {
        Ok(v) => v,
        Err(e) => panic!("JIT 执行失败: {e}"),
    };
    let iv = match interp_v {
        Ok(v) => v,
        Err(e) => panic!("解释器执行失败: {e}"),
    };
    assert_eq!(format!("{jv:?}"), format!("{iv:?}"), "JIT 与解释器结果不一致");
    match jv {
        tenth::runtime::value::Value::Int(n, _) => {
            assert_eq!(n, 22, "i 走完 0/1 两轮后为 2 ⇒ n=2, i=2 ⇒ 22");
        }
        other => panic!("期望 Int(22)，实际 {other:?}"),
    }
}
