//! `command_output_ex` —— 一次 spawn **同源**取回三轴 + 超时标志。
//!
//! 返回 **4 元组** `(stdout: str, stderr: str, exit_code: i64, timed_out: bool)`：
//!   - 元组返回已是一等（先例 `date_from_unix_days`），不动 VM/JIT/WASM 指令与 HIR 结构；
//!   - `timed_out` 是**值级**标志，**不走 `Err`**（卡死程序的部分输出正是定位证据；
//!     `Err` 会被 `?`/`or_die` 变 panic，用户就再也拿不到部分输出）；
//!   - `timeout_ms` **不得**接 `deadline_ms`：`with_timeout_ms` 是协作式的、native 内部
//!     不 tick ⇒ `with_timeout_ms(1000, || command_output_ex(...))` 形同无超时。
//!     本实现用「双读线程 + try_wait 轮询 + 到期 kill/wait」在 native 自身内界定。
//!
//! 覆盖：一次 spawn 拿齐三轴（stderr **非空**、退出码非 0）、超时用例（卡死子进程 →
//!        `timed_out=true` **且保留部分输出**）、VM/解释器两路径一致、静态类型是 4 元组。

use tenth::compile::bytecode::BytecodeCompiler;
use tenth::hir::lower::Lowerer;
use tenth::hir::types::{BaseType, Type};
use tenth::lexer::lexer::Lexer;
use tenth::parser::parser::Parser;
use tenth::runtime::interpreter::Interpreter;
use tenth::runtime::natives::register_all_natives;
use tenth::runtime::value::Value;
use tenth::runtime::vm::Vm;

fn lower(src: &str) -> Result<tenth::hir::hir::HirProgram, String> {
    let mut lexer = Lexer::new(src);
    let tokens = lexer.tokenize().map_err(|e| e.to_string())?;
    let mut parser = Parser::new(tokens);
    let program = parser.parse_program().map_err(|e| e.to_string())?;
    let mut lowerer = Lowerer::new();
    lowerer.lower_program(&program).map_err(|e| e.to_string())
}

fn run_vm(src: &str) -> Result<Value, String> {
    let hir = lower(src)?;
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
            Err(e) => return Err(format!("compile error: {e}")),
        }
    }
    if let Some(ref expr) = hir.main_expr {
        let compiler = BytecodeCompiler::new();
        let (chunk, closures) = compiler.compile_main(expr).map_err(|e| format!("compile error: {e}"))?;
        vm.add_fn("main".into(), chunk);
        for (name, closure_chunk) in closures {
            vm.add_fn(name, closure_chunk);
        }
        vm.call("main").map_err(|e| e.to_string())
    } else {
        Ok(Value::Unit)
    }
}

fn run_interp(src: &str) -> Result<Value, String> {
    let hir = lower(src)?;
    let mut interp = Interpreter::new(&hir);
    interp.execute_program(&hir)
        .map(|v| v.unwrap_or(Value::Unit))
        .map_err(|e| e.to_string())
}

/// 解 4 元组 `(stdout, stderr, exit_code, timed_out)`。
fn quad(v: &Value) -> (String, String, i64, bool) {
    match v {
        Value::Tuple(items) if items.len() == 4 => {
            let stdout = match &items[0] { Value::String(s) => s.clone(), o => format!("{o:?}") };
            let stderr = match &items[1] { Value::String(s) => s.clone(), o => format!("{o:?}") };
            let code = match &items[2] { Value::Int(n, _) => *n, o => panic!("exit_code 应为 Int，实际 {o:?}") };
            let to = match &items[3] { Value::Bool(b) => *b, o => panic!("timed_out 应为 Bool，实际 {o:?}") };
            (stdout, stderr, code, to)
        }
        other => panic!("期望 4 元组，实际 {other:?}"),
    }
}

/// 平台自适应的"带 stderr + 非零退出码"的探针命令。
/// Windows：`cmd /C "echo OUT& echo ERR 1>&2 & exit 3"`；POSIX：`sh -c`。
fn probe_src(timeout_ms: i64) -> String {
    let (prog, flag, script) = if cfg!(windows) {
        ("cmd.exe", "/C", "echo OUT& echo ERRLINE 1>&2 & exit 3")
    } else {
        ("sh", "-c", "echo OUT; echo ERRLINE 1>&2; exit 3")
    };
    format!(
        r#"
fn run_probe(p0: String, p1: String, p2: String, t: i64) -> (String, String, i64, bool) {{
    let h = match command_new(p0) {{
        Result::Ok(hh) => hh,
        Result::Err(e) => {{ return ("SPAWN_ERR", e, -1, false); }},
    }};
    command_arg(h, p1);
    command_arg(h, p2);
    command_output_ex(h, t)
}}
run_probe("{prog}", "{flag}", "{script}", {timeout_ms})
"#
    )
}

/// 平台自适应的"卡死子进程"探针：先打印一行，再长时间睡眠。
fn hang_src(timeout_ms: i64) -> String {
    let (prog, flag, script) = if cfg!(windows) {
        // ping 本机是 Windows 上无需额外依赖的可靠"睡一会儿"手段
        ("cmd.exe", "/C", "echo PARTIAL& ping -n 30 127.0.0.1 >NUL")
    } else {
        ("sh", "-c", "echo PARTIAL; sleep 30")
    };
    format!(
        r#"
fn run_hang(p0: String, p1: String, p2: String, t: i64) -> (String, String, i64, bool) {{
    let h = match command_new(p0) {{
        Result::Ok(hh) => hh,
        Result::Err(e) => {{ return ("SPAWN_ERR", e, -1, false); }},
    }};
    command_arg(h, p1);
    command_arg(h, p2);
    command_output_ex(h, t)
}}
run_hang("{prog}", "{flag}", "{script}", {timeout_ms})
"#
    )
}

// ── ① 一次 spawn 拿齐三轴（stdout + stderr 非空 + 退出码） ───────────────

#[test]
fn command_output_ex_captures_all_three_axes_vm() {
    let src = probe_src(0);
    let v = run_vm(&src).unwrap_or_else(|e| panic!("VM 执行失败: {e}"));
    let (out, err, code, timed_out) = quad(&v);
    assert!(out.contains("OUT"), "stdout 应含探针输出，实际 {out:?}");
    assert!(err.contains("ERRLINE"), "stderr 必须非空（旧 command_output 拿不到它），实际 {err:?}");
    assert_eq!(code, 3, "退出码应原样透传");
    assert!(!timed_out, "正常结束不应标记超时");
}

#[test]
fn command_output_ex_captures_all_three_axes_interp() {
    let src = probe_src(0);
    let v = run_interp(&src).unwrap_or_else(|e| panic!("解释器执行失败: {e}"));
    let (out, err, code, timed_out) = quad(&v);
    assert!(out.contains("OUT"), "stdout 应含探针输出，实际 {out:?}");
    assert!(err.contains("ERRLINE"), "stderr 必须非空，实际 {err:?}");
    assert_eq!(code, 3, "退出码应原样透传");
    assert!(!timed_out);
}

#[test]
fn command_output_ex_two_paths_identical() {
    let src = probe_src(0);
    let vm = quad(&run_vm(&src).expect("VM 失败"));
    let ip = quad(&run_interp(&src).expect("解释器失败"));
    assert_eq!(vm, ip, "VM 与解释器的 (stdout, stderr, code, timed_out) 必须逐字段一致");
}

/// 旧 `command_output` 行为不变（仍只回 stdout 的 Result）——本批只新增。
#[test]
fn legacy_command_output_unchanged() {
    let (prog, flag, script) = if cfg!(windows) {
        ("cmd.exe", "/C", "echo LEGACY")
    } else {
        ("sh", "-c", "echo LEGACY")
    };
    let src = format!(
        r#"
fn run_legacy(p0: String, p1: String, p2: String) -> String {{
    let h = or_die(command_new(p0));
    command_arg(h, p1);
    command_arg(h, p2);
    match command_output(h) {{
        Result::Ok(s) => s,
        Result::Err(e) => "ERR:" + e,
    }}
}}
run_legacy("{prog}", "{flag}", "{script}")
"#
    );
    for r in [run_vm(&src), run_interp(&src)] {
        let v = r.expect("执行失败");
        match v {
            Value::String(s) => assert!(s.contains("LEGACY"), "旧 command_output 应仍返回 stdout，实际 {s:?}"),
            other => panic!("期望 String，实际 {other:?}"),
        }
    }
}

// ── ② 超时：卡死子进程 → timed_out=true 且**保留部分输出** ────────────────

#[test]
fn command_output_ex_timeout_marks_flag_and_keeps_partial_output() {
    let src = hang_src(600);
    for (label, r) in [("VM", run_vm(&src)), ("解释器", run_interp(&src))] {
        let v = r.unwrap_or_else(|e| panic!("{label} 执行失败: {e}"));
        let (out, _err, code, timed_out) = quad(&v);
        assert!(timed_out, "{label}：卡死子进程必须标记 timed_out=true（若为 false 说明超时形同虚设）");
        assert!(out.contains("PARTIAL"),
            "{label}：超时必须**保留部分输出**，实际 stdout={out:?}");
        assert_eq!(code, -1, "{label}：超时的 exit_code 约定为 -1");
    }
}

/// 反向守护：`timeout_ms` **不能**接 `deadline_ms`。
///
/// `with_timeout_ms(1000, ...)` 只在 VM 主循环每 4096 指令检查 deadline，**native 内部不 tick**；
/// 若 `command_output_ex` 误用该机制，卡死子进程仍会挂住。这里用外层
/// `with_timeout_ms` 包住 `command_output_ex`，断言：真正的终止来自 native 自身的墙钟
/// （`timed_out=true` 而非外层超时吞成 `()`）。
#[test]
fn command_output_ex_timeout_is_not_deadline_ms() {
    let (prog, flag, script) = if cfg!(windows) {
        ("cmd.exe", "/C", "echo PARTIAL& ping -n 30 127.0.0.1 >NUL")
    } else {
        ("sh", "-c", "echo PARTIAL; sleep 30")
    };
    let src = format!(
        r#"
fn inner(p0: String, p1: String, p2: String) -> (String, String, i64, bool) {{
    let h = or_die(command_new(p0));
    command_arg(h, p1);
    command_arg(h, p2);
    command_output_ex(h, 600)
}}
with_timeout_ms(30000, |_| inner("{prog}", "{flag}", "{script}"))
"#
    );
    let v = run_vm(&src).unwrap_or_else(|e| panic!("VM 执行失败: {e}"));
    let (out, _err, _code, timed_out) = quad(&v);
    assert!(timed_out, "必须由 native 自身墙钟判定超时（外层 with_timeout_ms 不能替代）");
    assert!(out.contains("PARTIAL"), "应保留部分输出，实际 {out:?}");
}

// ── ③ 静态类型 = 4 元组（str, str, i64, bool） ───────────────────────────

#[test]
fn command_output_ex_static_type_is_quad() {
    let hir = lower(r#"command_output_ex(1, 100)"#).expect("lower 失败");
    let ty = hir.main_expr.expect("无 main_expr").ty;
    assert_eq!(
        ty,
        Type::Tuple(vec![
            Type::str_(),
            Type::str_(),
            Type::Base(BaseType::I64),
            Type::bool_(),
        ]),
        "静态返回类型必须是 4 元组 (str, str, i64, bool)，实际 {ty:?}"
    );
}

/// `let (a, b, c, d) = command_output_ex(...)` 的四个变量必须分别拿到元组元素类型
/// （AUDIT-11.4.65 的按位分解与此处联合验收）。
#[test]
fn command_output_ex_destructured_var_types() {
    let src = r#"
fn f() -> (String, String, i64, bool) {
    let (a, b, c, d) = command_output_ex(1, 100);
    (a, b, c, d)
}
"#;
    let hir = lower(src).expect("lower 失败");
    let body_ty = hir.functions.iter().find(|f| f.name == "f").expect("无 f").body.ty.clone();
    assert_eq!(
        body_ty,
        Type::Tuple(vec![
            Type::str_(),
            Type::str_(),
            Type::Base(BaseType::I64),
            Type::bool_(),
        ]),
        "解构后各变量类型必须正确（否则是 AUDIT-11.4.65 的静默错型），实际 {body_ty:?}"
    );
}
