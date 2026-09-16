//! 永久回归守护：tenthc 侧「响亮化」——AUDIT-11.4.57 / 11.4.58 / 11.4.59，
//! 以及 AUDIT-11.4.72（`[WASM] compile` 日志门控）。
//!
//! 背景：这四条都是「静默失败 → 响亮失败」的修复。它们的**唯一**可观测证据是
//! tenthc 自己打印的错误行（`println` → 宿主 `host.println`）。而 libtest 会把
//! 测试线程的 `println!` 捕获并在通过时丢弃，进程内**无法**断言这段文本。
//!
//! 因此本文件采用「自执行子进程 + 文件重定向」：
//!   * 子进程 = 同一个测试二进制，`TENTH_W7_CHILD=<group>` + `--exact w7_loud_child_runner
//!     --nocapture`，stdout/stderr 全部重定向到 `tenth/target/` 下的临时文件；
//!   * 子进程内部照抄既有 tenthc 测试的驱动方式（`tenthc_*_test.rs` / `parity_test.rs`
//!     的 `compile_via_tenthc`：Rust 母编译器编译 tenthc 源码 → WASM-A，wasmi 跑出
//!     WASM-B），并在每个用例边界打印 `W7-BEGIN/LEN/VALID/END` 结构化标记；
//!   * 父进程（真正的断言方）读回该文件，按用例分段断言：
//!     - **响亮**：分段文本含特征串；
//!     - **产物形状**：`W7-LEN` 为 0（57）或 >0 且 `W7-VALID ok`（58/59 占位与栈平衡）；
//!     - **防误报**：正常闭包/循环用例分段内不得出现 `[tenthc]`。
//!
//! 不使用 `#[ignore]`；不依赖外部命令（`tenth` 二进制 / 临时 .th 夹具）。

use std::fs;
use std::path::PathBuf;
use std::process::{Command, Stdio};

use tenth::compile::wasm::register_host_functions;
use wasmi::{Config, Engine, Linker, Module, StackLimits, Store};

const CHILD_ENV: &str = "TENTH_W7_CHILD";
const CHILD_TEST: &str = "w7_loud_child_runner";

// ── 用例表 ───────────────────────────────────────────────────────────────

/// 单个用例的期望（父进程据此断言）。
struct Expect {
    /// 子进程输出中必须出现的特征串（响亮化证据）。
    msg: Option<&'static str>,
    /// 产物必须为空（无 `\0asm` 魔数）：用于 11.4.57。
    empty: bool,
    /// 该用例必须是「正常用例」：分段内不得出现任何 `[tenthc]` 报错。
    silent: bool,
}

const fn loud(msg: &'static str) -> Expect {
    Expect { msg: Some(msg), empty: false, silent: false }
}

const fn quiet() -> Expect {
    Expect { msg: None, empty: false, silent: true }
}

const fn empty_product(msg: &'static str) -> Expect {
    Expect { msg: Some(msg), empty: true, silent: false }
}

struct Case {
    group: &'static str,
    name: &'static str,
    src: &'static str,
    expect: Expect,
}

const CASES: &[Case] = &[
    // ── 11.4.57：顶层 `let` 不再静默丢弃（产物变空 = 响亮） ──
    Case {
        group: "57",
        name: "top_level_let",
        src: "let mut H: i32 = 0;\nfn f() -> i64 { H = H + 1; H }",
        expect: empty_product("编译中止"),
    },
    // ── 11.4.58：4 处兜底点（Var / Assign / AssignOp / 闭包 env 装载） ──
    Case {
        group: "58",
        name: "var_read_unbound",
        src: "fn f(a: i64) -> i64 { a + zzz_var_missing }",
        expect: loud("未绑定变量 'zzz_var_missing'"),
    },
    Case {
        group: "58",
        name: "assign_unbound",
        src: "fn f(a: i64) -> i64 { zzz_assign_missing = 3; a }",
        expect: loud("对未绑定变量赋值 'zzz_assign_missing'"),
    },
    Case {
        group: "58",
        name: "assignop_unbound",
        src: "fn f(a: i64) -> i64 { zzz_assignop_missing += 3; a }",
        expect: loud("对未绑定变量赋值 'zzz_assignop_missing'"),
    },
    Case {
        group: "58",
        name: "closure_env_load_unbound",
        src: "fn f() -> i64 { let g = |u: i64| zzz_cap_missing + u; g(1) }",
        expect: loud("闭包捕获变量解析失败 'zzz_cap_missing'"),
    },
    // ── 11.4.59：闭包内写自由变量（Assign / AssignOp）响亮 ──
    // 注：报错路径刻意隔离在闭包体内（f 的尾语句是调用），以排除 tenthc
    // 既有缺陷「值语句后跟尾值不 drop」的干扰 —— 那条缺陷与本波改动无关，
    // 见 tenthc_loud_11_4_59_... 测试的说明。
    Case {
        group: "59",
        name: "closure_assign_capture",
        src: "fn f(seed: i64) -> i64 { let mut n = seed; let g = |u: i64| { n = n + u; u }; g(1) }",
        expect: loud("不能对捕获变量赋值 'n'"),
    },
    Case {
        group: "59",
        name: "closure_assignop_capture",
        src: "fn f(seed: i64) -> i64 { let mut n = seed; let g = |u: i64| { n += u; u }; g(1) }",
        expect: loud("不能对捕获变量赋值 'n'"),
    },
    // ── 防误报：正常闭包 / 正常循环必须安静且产物有效 ──
    Case {
        group: "59",
        name: "normal_closure_reads_outer_param",
        // 顺序敏感：闭包捕获的是**外层函数形参**，解析必须先查捕获再查形参。
        src: "fn make(n: i64) -> i64 { let f = |x: i64| x + n; f(5) }",
        expect: quiet(),
    },
    Case {
        group: "59",
        name: "normal_closure_reads_local_capture",
        src: "fn f() -> i64 { let n = 10; let g = |x: i64| x + n; g(5) }",
        expect: quiet(),
    },
    Case {
        group: "59",
        name: "normal_for_and_assign_after_errors",
        // 「后续正常程序仍能编译通过」：普通 for + 赋值，产物必须有效。
        src: "fn f(x: i64) -> i64 { let mut s: i64 = 0; for i in 0..x { s = s + i; }; s }",
        expect: quiet(),
    },
    Case {
        group: "59",
        name: "normal_block_closure_with_local_let",
        // tenthc 的 captures 会把闭包体内自己的 `let` 误收为「伪捕获」；
        // env 装载点必须识别出来、不得报成用户错误。
        src: "fn f(seed: i64) -> i64 { let g = |u: i64| { let mut m = u; m = m + 1; m }; g(seed) }",
        expect: quiet(),
    },
    // ── 11.4.72：只借这条用例触发 Rust 侧 compile_function 日志 ──
    Case {
        group: "72",
        name: "verbose_probe",
        src: "fn f() -> i64 { 1 }",
        expect: quiet(),
    },
];

// ── 子进程（同一测试二进制、文件重定向捕获） ─────────────────────────────

fn selfhost_config() -> Config {
    let mut config = Config::default();
    let limits = StackLimits::new(
        65536,       // initial_value_stack_height
        1_048_576,   // maximum_value_stack_height
        65536,       // maximum_recursion_depth
    )
    .expect("valid stack limits");
    config.set_stack_limits(limits);
    config
}

/// tenthc 管线（与 `tenthc_for_loop_test.rs` / `parity_test.rs` 同一驱动方式）：
/// parse_program → lower_program → compile_to_wasm，返回 tenthc 的产物字节。
fn compile_via_tenthc(test_src: &str) -> Vec<u8> {
    use tenth::compile;
    use tenth::hir::lower::Lowerer;
    use tenth::lexer::lexer::Lexer;
    use tenth::parser::parser::Parser;

    let selfhost_src = [
        include_str!("../../tenthc/lexer/token.th"),
        include_str!("../../tenthc/lexer/lexer.th"),
        include_str!("../../tenthc/parser/parser.th"),
        include_str!("../../tenthc/hir/hir.th"),
        include_str!("../../tenthc/hir/lower.th"),
        include_str!("../../tenthc/compile/wasm.th"),
    ]
    .join("\n");

    let escaped = test_src.replace('\\', "\\\\").replace('"', "\\\"");
    let main_src = format!(
        "fn main()->Vec<i64>{{let mut lex=lexer_new(\"{}\");let tokens=lexer_tokenize(&mut lex);let program=parse_program(tokens);let hir=lower_program(program);let wasm=compile_to_wasm(hir);wasm}}",
        escaped
    );
    let full_src = format!("{}\n{}", selfhost_src, main_src);

    // Stage 1：Rust 母编译器 → WASM-A（tenthc 本体）
    let mut lexer = Lexer::new(&full_src);
    let tokens = lexer.tokenize().expect("lex");
    let mut parser = Parser::new(tokens);
    let program = parser.parse_program().expect("parse");
    let mut lowerer = Lowerer::new();
    let hir = lowerer.lower_program(&program).expect("lower");
    let wasm_a = compile::compile_to_wasm(&hir).expect("compile");
    assert_eq!(&wasm_a[..4], b"\0asm", "WASM-A must have valid magic");

    // Stage 2：wasmi 运行 WASM-A → WASM-B（tenthc 对被测源码的产物）
    let engine = Engine::new(&selfhost_config());
    let module = Module::new(&engine, &wasm_a).expect("compile wasm-a");
    let mut store = Store::new(&engine, 8192u32);
    let mut linker = Linker::new(&engine);
    register_host_functions(&mut linker).expect("register host functions");
    let inst = linker
        .instantiate(&mut store, &module)
        .expect("inst")
        .start(&mut store)
        .expect("start");
    let main_fn = inst.get_func(&store, "main").expect("main");
    let mut r = [wasmi::Val::I32(0)];
    main_fn.call(&mut store, &[], &mut r).expect("call main");
    let vec_ptr = match r[0] {
        wasmi::Val::I32(v) => v as i64,
        wasmi::Val::I64(v) => v,
        _ => panic!("expected i32/i64 return from main, got {:?}", r[0]),
    };

    // 读取 Vec<i64>（元素即产物字节）。布局：cap(8) + len(8) + data_ptr(4) + data…
    let mem = inst.get_memory(&store, "memory").expect("memory");
    let data = mem.data(&store);
    let vp = vec_ptr as i32 as usize;
    assert!(vp + 20 <= data.len(), "vec ptr {} out of range", vp);
    let len = i64::from_le_bytes(data[vp + 8..vp + 16].try_into().unwrap());
    let dp = i32::from_le_bytes(data[vp + 16..vp + 20].try_into().unwrap()) as usize;
    let mut bytes = Vec::with_capacity(len as usize);
    for i in 0..len as usize {
        let pos = dp + i * 8;
        assert!(pos + 8 <= data.len(), "vec data {} out of range", pos);
        bytes.push(i64::from_le_bytes(data[pos..pos + 8].try_into().unwrap()) as u8);
    }
    bytes
}

/// 子进程入口：仅在 `TENTH_W7_CHILD=<group>` 时工作（否则立即返回）。
/// 它**不做断言**，只把 tenthc 自己的输出与结构化标记写进 stdout，
/// 由父进程重定向到文件后读回断言。
#[test]
fn w7_loud_child_runner() {
    let group = match std::env::var(CHILD_ENV) {
        Ok(g) => g,
        Err(_) => return, // 正常全量跑：这是空用例
    };
    let engine = Engine::new(&selfhost_config());
    for case in CASES.iter().filter(|c| c.group == group.as_str()) {
        println!("W7-BEGIN {}", case.name);
        let wasm = compile_via_tenthc(case.src);
        println!("W7-LEN {} {}", case.name, wasm.len());
        if wasm.len() >= 8 {
            match Module::new(&engine, &wasm) {
                Ok(_) => println!("W7-VALID {} ok", case.name),
                Err(e) => println!("W7-VALID {} err {}", case.name, e),
            }
        } else {
            println!("W7-VALID {} empty", case.name);
        }
        println!("W7-END {}", case.name);
    }
}

/// 起一个子进程跑 `group` 组的用例，返回它合并后的 stdout+stderr 文本。
fn child_report(group: &str, extra_env: &[(&str, &str)]) -> String {
    let exe = std::env::current_exe().expect("current_exe");
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("target");
    fs::create_dir_all(&dir).expect("create target dir");
    let path = dir.join(format!("w7_loud_child_{}_{}.log", std::process::id(), group));
    let file = fs::File::create(&path).expect("create capture file");
    let file2 = file.try_clone().expect("clone capture file");

    let mut cmd = Command::new(exe);
    cmd.arg("--exact")
        .arg(CHILD_TEST)
        .arg("--nocapture")
        .arg("--test-threads=1")
        .env(CHILD_ENV, group)
        .stdin(Stdio::null())
        .stdout(Stdio::from(file))
        .stderr(Stdio::from(file2));
    for (k, v) in extra_env {
        cmd.env(k, v);
    }
    let status = cmd.status().expect("spawn child test process");
    let out = fs::read_to_string(&path).unwrap_or_default();
    let _ = fs::remove_file(&path);
    assert!(status.success(), "子进程（group={group}）失败：\n{out}");
    out
}

// ── 父进程断言辅助 ───────────────────────────────────────────────────────

/// 取出某个用例在子进程输出中的分段（BEGIN..END 之间，含 tenthc 的错误行）。
fn segment<'a>(report: &'a str, name: &str) -> &'a str {
    let begin = format!("W7-BEGIN {name}");
    let end = format!("W7-END {name}");
    let b = report
        .find(&begin)
        .unwrap_or_else(|| panic!("子进程输出缺少 `{begin}`：\n{report}"));
    let tail = &report[b..];
    let e = tail
        .find(&end)
        .unwrap_or_else(|| panic!("子进程输出缺少 `{end}`：\n{report}"));
    &tail[..e]
}

/// 取出 `W7-<key> <name> <value>` 里的 value。
fn field<'a>(report: &'a str, name: &str, key: &str) -> &'a str {
    let prefix = format!("W7-{key} {name} ");
    let i = report
        .find(&prefix)
        .unwrap_or_else(|| panic!("子进程输出缺少 `{prefix}`：\n{report}"));
    let rest = &report[i + prefix.len()..];
    let end = rest.find('\n').unwrap_or(rest.len());
    rest[..end].trim()
}

/// 跑一组用例并逐条断言（响亮特征串 / 产物形状 / 正常用例安静）。
fn assert_group(group: &str) {
    let report = child_report(group, &[]);
    for case in CASES.iter().filter(|c| c.group == group) {
        let seg = segment(&report, case.name);
        let len: usize = field(&report, case.name, "LEN")
            .parse()
            .unwrap_or_else(|_| panic!("[{}] W7-LEN 非数字：\n{report}", case.name));

        if case.expect.empty {
            assert_eq!(
                len, 0,
                "[{}] 响亮化要求产物为空（无 \\0asm 魔数）：实际 len={len}\n{seg}",
                case.name
            );
        } else {
            assert!(len > 0, "[{}] 产物不应为空：\n{seg}", case.name);
            assert_eq!(
                field(&report, case.name, "VALID"),
                "ok",
                "[{}] 产物必须通过 wasmi 校验（占位不破校验 / 栈平衡未破）\n{seg}",
                case.name
            );
        }

        if let Some(msg) = case.expect.msg {
            assert!(
                seg.contains(msg),
                "[{}] 未出现响亮化特征串 {msg:?}（静默失败回归？）该用例实际输出：\n{seg}",
                case.name
            );
        }
        if case.expect.silent {
            assert!(
                !seg.contains("[tenthc]"),
                "[{}] 正常用例不应出现 tenthc 报错（误报）：\n{seg}",
                case.name
            );
        }
    }
}

// ── 永久回归测试 ─────────────────────────────────────────────────────────

/// AUDIT-11.4.57：顶层 `let` 曾与错误一起静默蒸发。现在必须**响亮**：
/// 打印解析错误 + 返回空产物（宿主/下游因缺少 `\0asm` 立即失败）。
#[test]
fn tenthc_loud_11_4_57_top_level_let_is_loud_and_empty() {
    assert_group("57");
}

/// AUDIT-11.4.58：未绑定名曾静默落 local 0（＝首形参/env_ptr）。
/// 4 处兜底点（Var 读 / Assign 写 / AssignOp / 闭包 env 装载）现均须响亮，
/// 且产物仍须通过 wasmi 校验（占位值不破坏模块结构）。
#[test]
fn tenthc_loud_11_4_58_unbound_name_four_sites() {
    assert_group("58");
}

/// AUDIT-11.4.59：闭包体内写自由变量曾写成 `env_ptr`（自读自写不一致 +
/// 覆写环境指针）。Assign 与 AssignOp 现均须响亮，且产物须通过 wasmi 校验
/// —— 这是「RHS 已 pop / 未发射 ⇒ 栈平衡未破」的机器判据。
/// 同组还断言正常闭包（读外层形参 / 读本地捕获 / 块体闭包局部 let）与
/// 正常 for+赋值**不报错**且产物有效（防误报 + 防栈失衡连带伤害）。
///
/// 注：本组刻意用「f 尾语句是调用」的形状隔离报错路径。tenthc 另有一处
/// **既有**缺陷（与本波无关）：值语句后跟尾值（`g(x); tail`）不 drop
/// 表达式语句的返回值 ⇒ 模块栈失衡、wasmi 报 "values remaining on stack"。
/// 该缺陷在改动前后同样存在，未纳入本文件的期望。
#[test]
fn tenthc_loud_11_4_59_capture_write_is_loud_and_stack_balanced() {
    assert_group("59");
}

/// AUDIT-11.4.72：`[WASM] compile` 刷屏改为受 `TENTH_WASM_VERBOSE` 控制。
/// 行为判据：默认（未设变量）子进程输出里不得出现该行；显式设 1 时应恢复。
/// 子进程编译 tenthc 源码会经过 Rust 侧 `compile_function`（数百次调用），
/// 因此这条断言直接反映门控是否还存在。
#[test]
fn tenthc_loud_11_4_72_verbose_gate_quiet_by_default() {
    let quiet = child_report("72", &[]);
    let quiet_hits = quiet.matches("[WASM] compile").count();
    assert_eq!(
        quiet_hits, 0,
        "默认（未设 TENTH_WASM_VERBOSE）不应出现 [WASM] compile 刷屏，实际 {quiet_hits} 行"
    );

    let loud = child_report("72", &[("TENTH_WASM_VERBOSE", "1")]);
    assert!(
        loud.contains("[WASM] compile"),
        "显式设 TENTH_WASM_VERBOSE=1 时应恢复 [WASM] compile 日志（门控被删？）"
    );
}
