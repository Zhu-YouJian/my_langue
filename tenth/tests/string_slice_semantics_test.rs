//! 永久回归守护：W13 能力波（AUDIT-11.4.89 / 11.4.93 / 11.4.96 / 11.4.78）。
//!
//! 三个**红线级**静默/分歧缺陷的护栏：
//! ① `s[0..99]`（end 越界）在 VM/JIT 静默 clamp 成 `"hello"`、解释器却报错
//!    ⇒ 同一源码两路径不同结果；且 `str_slice`/`str_len`/`str_at` 是**语言级名字**
//!    （前端白名单放行、`tenthc/**` 30+ 处在用）却只有 WASM host 有实现
//!    ⇒ VM/解释器路径"未定义函数"（11.4.93）。
//! ② `s[0..=2]` 的 `=` 被 parser **静默丢弃** ⇒ 得 `"he"`（真值 `"hel"`）。
//! ③ 张量 `t[0..1]` / `t[:]` 在解释器被**静默当成 `t[0]`**（VM 侧 Colon 被静默丢弃）。
//!
//! 另含 ④ `Vec<T>` 注解的方法表静态类型（AUDIT-11.4.78 窄修；`get`/`pop` 有意保持
//! `Unknown`，见 `hir/lower/types.rs` 的加固条件）。
//!
//! ## 判据口径（刻意如此，勿"收紧"）
//! * **运行时红线用例**：两条路径都必须**非零退出 + stderr 非空**（＝响亮）。
//!   不比对 stderr 文本（跨路径诊断文案存在既有差异），也**不要求 stdout 为空**：
//!   默认(JIT)路径存在既有缺陷 AUDIT-11.4.99（报错后调用方继续执行到下一个检查点，
//!   可能先落一行 `()`），把"响亮"判据钉在退出码 + stderr 上才是诚实口径；
//!   该缺陷属 JIT 作业面（W10），本波不许触碰。
//! * **正常路径用例**：两路径 stdout 必须**逐字节一致**且等于金标准
//!   （防"两路径一起错成一样"的假绿）。
//! * **编译期用例**：库内断言 `ParseError` / `TypeError`——注入漂移（把检查删掉）
//!   必然变红。

use std::path::PathBuf;
use std::process::{Command, Output};

use tenth::error::TenthError;
use tenth::hir::lower::Lowerer;
use tenth::lexer::lexer::Lexer;
use tenth::parser::parser::Parser;

const TENTH_EXE: &str = env!("CARGO_BIN_EXE_tenth");
const TENTH_DIR: &str = env!("CARGO_MANIFEST_DIR");

// ════════════════════════════════════════════════════════════════════
// 库内（编译期）辅助
// ════════════════════════════════════════════════════════════════════

fn lower(src: &str) -> Result<(), TenthError> {
    let mut lexer = Lexer::new(src);
    let tokens = lexer.tokenize()?;
    let mut parser = Parser::new(tokens);
    let program = parser.parse_program()?;
    let mut lowerer = Lowerer::new();
    lowerer.lower_program(&program).map(|_| ())
}

fn parse_only(src: &str) -> Result<(), TenthError> {
    let mut lexer = Lexer::new(src);
    let tokens = lexer.tokenize()?;
    let mut parser = Parser::new(tokens);
    parser.parse_program().map(|_| ())
}

/// 断言**语法层**响亮拒绝（含可读信息）。
fn assert_parse_error(src: &str, want: &str, why: &str) {
    match parse_only(src) {
        Err(TenthError::ParseError { message, .. }) => assert!(
            message.contains(want),
            "[{why}] 解析错误信息不含 {want:?}，实际：{message}"
        ),
        Err(other) => panic!("[{why}] 期望 ParseError，实际 {other:?}"),
        Ok(_) => panic!("[{why}] 期望**编译期**响亮报错，但解析通过了（静默错值回归？）"),
    }
}

/// 断言**类型层**响亮报错（TypeError）。
fn assert_type_error(src: &str, why: &str) {
    match lower(src) {
        Err(TenthError::TypeError { message, .. }) => {
            assert!(!message.trim().is_empty(), "[{why}] TypeError 信息为空")
        }
        Err(other) => panic!("[{why}] 期望 TypeError，实际 {other:?}"),
        Ok(_) => panic!("[{why}] 期望编译期 TypeError，但降级成功（静态类型未收窄？）"),
    }
}

fn assert_compiles(src: &str, why: &str) {
    if let Err(e) = lower(src) {
        panic!("[{why}] 期望编译通过，实际失败：{e:?}");
    }
}

// ════════════════════════════════════════════════════════════════════
// 跨路径（spawn 真实二进制；与 native_registry_guard_test.rs 同一驱动方式）
// ════════════════════════════════════════════════════════════════════

fn run_src(name: &str, src: &str, interpreter: bool) -> Output {
    let dir = PathBuf::from(TENTH_DIR).join("target").join("tmp_w13_slice_semantics");
    std::fs::create_dir_all(&dir).expect("创建临时目录失败");
    let file = dir.join(format!("{name}.th"));
    std::fs::write(&file, src).expect("写临时 .th 失败");

    let mut cmd = Command::new(TENTH_EXE);
    cmd.arg("run").arg(&file).current_dir(TENTH_DIR);
    if interpreter {
        // main.rs 用 `env::var(..).is_ok()` 判定 ⇒ 必须显式设值；默认路径必须显式清除，
        // 否则父进程环境会把"默认路径"静默跑成解释器（假绿）。
        cmd.env("TENTH_NO_VM", "1");
    } else {
        cmd.env_remove("TENTH_NO_VM");
    }
    cmd.output()
        .unwrap_or_else(|e| panic!("spawn 被测二进制失败：{TENTH_EXE}（{e}）"))
}

/// 红线口径：**两条路径都必须响亮失败**（非零退出 + stderr 非空）。
fn assert_loud_both_paths(name: &str, src: &str, why: &str) {
    for (label, interpreter) in [("默认路径(VM/JIT)", false), ("解释器路径(TENTH_NO_VM=1)", true)] {
        let out = run_src(name, src, interpreter);
        let stderr = String::from_utf8_lossy(&out.stderr).to_string();
        assert_ne!(
            out.status.code(),
            Some(0),
            "[{why}] {label} 必须响亮失败（此前是静默错值/静默当 0）。\n\
             --- stdout ---\n{}\n--- stderr ---\n{}",
            String::from_utf8_lossy(&out.stdout),
            stderr
        );
        assert!(
            !stderr.trim().is_empty(),
            "[{why}] {label} 非零退出但 stderr 为空（不响亮）"
        );
    }
}

/// 正常路径口径：两路径 stdout 逐字节一致 + 退出码 0 + 等于金标准。
fn assert_gold_parity(name: &str, src: &str, gold: &[&str], why: &str) {
    let def = run_src(name, src, false);
    let interp = run_src(name, src, true);
    let d_out = String::from_utf8_lossy(&def.stdout).to_string();
    let i_out = String::from_utf8_lossy(&interp.stdout).to_string();
    assert_eq!(
        def.status.code(),
        Some(0),
        "[{why}] 默认路径应成功。stderr={}",
        String::from_utf8_lossy(&def.stderr)
    );
    assert_eq!(
        interp.status.code(),
        Some(0),
        "[{why}] 解释器路径应成功。stderr={}",
        String::from_utf8_lossy(&interp.stderr)
    );
    assert_eq!(def.stdout, interp.stdout, "[{why}] 两路径 stdout 不一致（跨后端分歧）");
    let lines: Vec<&str> = d_out.lines().collect();
    assert_eq!(lines, gold.to_vec(), "[{why}] 输出不等于金标准（含「两路径一起错」）");
}

// ════════════════════════════════════════════════════════════════════
// ① str_slice / str_len / str_at（11.4.89 ① + 11.4.93）
// ════════════════════════════════════════════════════════════════════

/// 正常路径：native 自由函数 + 语法切片在两条路径上**逐字节一致**，且为**码点**语义。
const HAPPY_SRC: &str = r#"fn main() {
    println("A|" + str_slice("hello", 1, 3));
    println("B|" + str_slice("中文abc", 0, 3));
    println("C|" + format("{}", str_len("中文abc")));
    println("D|" + str_at("中文abc", 1));
    println("E|" + "hello"[1..3]);
    println("F|" + "中文abc"[0..3]);
    println("G|" + str_slice("hello", 0, 5));
    println("H|" + str_slice("hello", 3, 3));
}"#;

const HAPPY_GOLD: &[&str] = &[
    "A|el",
    "B|中文a",   // 码点：[0,3) = 中文a（按字节会切出非法 UTF-8 边界）
    "C|5",       // 码点数（不是字节数 8）
    "D|文",
    "E|el",
    "F|中文a",
    "G|hello",
    "H|",        // 零宽切片合法
];

#[test]
fn str_slice_native_and_syntax_gold_parity() {
    assert_gold_parity(
        "happy",
        HAPPY_SRC,
        HAPPY_GOLD,
        "11.4.89 ①/11.4.93：str_slice/str_len/str_at 两路径注册 + 码点语义",
    );
}

/// 红线①：`s[0..99]`（end 越界）此前默认路径静默 clamp 成 `"hello"`（rc=0）而解释器报错。
#[test]
fn str_slice_syntax_oob_end_is_loud_not_clamped() {
    let src = r#"fn main() {
    let s = "hello";
    let t = s[0..99];
    println(t);
}"#;
    assert_loud_both_paths(
        "syntax_oob",
        src,
        "11.4.89 ②：`s[0..99]` 不得再静默 clamp（默认路径此前 rc=0 输出 hello）",
    );
}

/// 红线①：native 形式的三类非法索引（end 越界 / 负索引 / start > end）都须响亮。
#[test]
fn str_slice_native_oob_negative_reversed_are_loud() {
    for (name, call) in [
        ("native_oob", "str_slice(\"hello\", 0, 99)"),
        ("native_neg", "str_slice(\"hello\", -1, 2)"),
        ("native_rev", "str_slice(\"hello\", 3, 1)"),
    ] {
        let src = format!("fn main() {{\n    println({call});\n}}");
        assert_loud_both_paths(
            name,
            &src,
            "11.4.89 ①：str_slice 严格索引（越界/负/start>end 一律响亮）",
        );
    }
}

/// 红线①：语法形式 `s[-1..2]` 的负索引此前 VM 报"类型不匹配"、解释器报巨型下标。
#[test]
fn str_slice_syntax_negative_is_loud() {
    let src = r#"fn main() {
    let s = "hello";
    let t = s[-1..2];
    println(t);
}"#;
    assert_loud_both_paths("syntax_neg", src, "11.4.89 ②：负索引必须响亮（此前两路径都不可读）");
}

// ════════════════════════════════════════════════════════════════════
// ② `..=` 索引（11.4.89 ③，红线级静默错值）
// ════════════════════════════════════════════════════════════════════

/// 红线②：`s[0..=2]` 此前静默等于 `s[0..2]`（得 `"he"`，真值 `"hel"`）——现编译期报错。
#[test]
fn index_inclusive_range_is_rejected_at_parse_time() {
    assert_parse_error(
        r#"fn main() {
    println("hello"[0..=2]);
}"#,
        "..=",
        "11.4.89 ③：索引 `..=` 必须编译期响亮报错（此前静默丢 '=' 得 'he'）",
    );
    // 开放端含端点 `s[..=2]`（DotDotEq 单 token）走的是另一条解析分支，同样必须拒绝。
    assert_parse_error(
        r#"fn main() {
    println("hello"[..=2]);
}"#,
        "..=",
        "11.4.89 ③：`s[..=2]` 形式同样必须拒绝",
    );
}

/// 对照组（防过度拒绝）：`..` 索引与 `..=` 的**非索引**用法（for 区间）必须仍然可用。
#[test]
fn index_exclusive_and_range_for_loop_still_work() {
    assert_compiles(
        r#"fn main() {
    println("hello"[0..2]);
}"#,
        "11.4.89 ③ 对照：`..` 索引不得被连带拒绝",
    );
    assert_compiles(
        r#"fn main() {
    let mut s: i64 = 0;
    for i in 1..=3 { s = s + i; }
    println(s);
}"#,
        "11.4.89 ③ 对照：for-in 的 `..=`（非索引位置）不得被连带拒绝",
    );
}

// ════════════════════════════════════════════════════════════════════
// ③ 张量 Colon / Range 索引（11.4.96，红线级静默错值）
// ════════════════════════════════════════════════════════════════════

/// 红线③：`t[0..1]` 在解释器此前**静默等于 `t[0]`**（rc=0 打印第一行），VM 侧响亮。
#[test]
fn tensor_range_index_is_loud_not_silently_t0() {
    let src = r#"fn main() {
    let t = tensor([[1.0, 2.0], [3.0, 4.0]]);
    let u = t[0..1];
    println(u);
}"#;
    assert_loud_both_paths(
        "tensor_range",
        src,
        "11.4.96：张量 Range 索引不得静默当 0（解释器此前 rc=0 得 t[0]）",
    );
}

/// 红线③：`t[:]` 此前能解析——VM 静默得"张量本身"、解释器静默得 `t[0]`（跨路径分歧）。
#[test]
fn tensor_colon_index_is_rejected_at_parse_time() {
    assert_parse_error(
        r#"fn main() {
    let t = tensor([[1.0, 2.0], [3.0, 4.0]]);
    println(t[:]);
}"#,
        ":",
        "11.4.96：Colon 下标（`t[:]`）必须响亮拒绝",
    );
    assert_loud_both_paths(
        "tensor_colon",
        r#"fn main() {
    let t = tensor([[1.0, 2.0], [3.0, 4.0]]);
    println(t[:]);
}"#,
        "11.4.96：`t[:]` 两路径都必须失败（此前分别静默得 整张量 / t[0]）",
    );
}

/// 对照组：张量整维索引（`t[0]`）与字符串 Colon 之外的索引必须仍可用。
#[test]
fn tensor_integer_index_still_works() {
    assert_gold_parity(
        "tensor_single",
        r#"fn main() {
    let t = tensor([[1.0, 2.0], [3.0, 4.0]]);
    println(t[0]);
}"#,
        &["[1.0, 2.0]"],
        "11.4.96 对照：`t[0]` 整维索引不得被连带拒绝",
    );
}

// ════════════════════════════════════════════════════════════════════
// ④ Vec<T> 注解的方法表静态类型（11.4.78 窄修 + L3-B 附录 A5 同族）
// ════════════════════════════════════════════════════════════════════
//
// 口径说明：`let x: T = <expr>` 的注解**不做**兼容性校验（非 Tensor 注解直接
// `annot.clone()`，见 `hir/lower/lower_stmt.rs`）⇒ 用"错型 let"当判据是无效的。
// 本组一律走**调用实参**这条已校验的通路（与 `call_arg_type_check_test.rs` 同口径）：
// 收窄后类型不再兼容 ⇒ 编译期 TypeError；收窄前是 `Unknown` ⇒ 一律放行（红）。

/// 生成"把 `call` 作为实参传给形参类型为 `param_ty` 的函数"的源码。
fn src_call_arg(param_ty: &str, call: &str) -> String {
    format!(
        "fn take(x: {param_ty}) -> i64 {{ 0 }}\n\
         fn take_i64(x: i64) -> i64 {{ x }}\n\
         fn take_str(x: str) -> i64 {{ 0 }}\n\
         fn take_bool(x: bool) -> i64 {{ 0 }}\n\
         fn main() {{\n\
         \x20   let v: Vec<i64> = Vec::new();\n\
         \x20   let r = {call};\n\
         \x20   println(r);\n\
         }}"
    )
}

/// ④ 反例（能变红）：收窄前 `get_opt` 是 `Unknown`（与任何形参兼容 ⇒ 放行）；
/// 收窄后 `or_die(v.get_opt(0))` 是 i64 ⇒ 传给 `str` 形参必须 TypeError。
#[test]
fn vec_annot_get_opt_narrows_to_i64() {
    assert_type_error(
        &src_call_arg("str", "take(or_die(v.get_opt(0)))"),
        "11.4.78：收窄后 get_opt 的内型 i64 传给 str 形参必须 TypeError（修前 Unknown 静默放行）",
    );
    // 正例（防过度收紧）：内型确实是 i64 ⇒ 传给 i64 形参必须通过。
    assert_compiles(
        &src_call_arg("i64", "take_i64(or_die(v.get_opt(0)))"),
        "11.4.78：`or_die(v.get_opt(0))` 必须是 i64（内型收窄到注解的 i64）",
    );
}

/// ④ 加固条件（总师裁定）：`get`/`pop` **有意**保持 `Unknown`——本用例把它钉住，
/// 谁若顺手把 `get` 改成 `Option`（会波及 std 约 178 处 `.get(`），这条会红，迫使复议。
#[test]
fn vec_get_and_pop_stay_unknown_by_design() {
    assert_compiles(
        &src_call_arg("str", "take(v.get(0))"),
        "11.4.78 加固条件：`Vec.get` 保持 Unknown（不得顺手收窄成 Option）",
    );
    assert_compiles(
        &src_call_arg("str", "take(v.pop())"),
        "11.4.78 加固条件：`Vec.pop` 保持 Unknown（不得顺手收窄成 Option）",
    );
}

/// ④ 同族（L3-B 附录 A5）：`slice`/`join`/`is_empty` 的静态返回类型补齐后，
/// 错型实参必须 TypeError（修前 `Unknown` ⇒ 一律放行）。
#[test]
fn vec_annot_other_methods_are_typed() {
    assert_type_error(
        &src_call_arg("bool", "take_bool(v.join(\",\"))"),
        "11.4.78 同族：`Vec.join` → str，传给 bool 形参必须 TypeError（修前 Unknown 放行）",
    );
    assert_type_error(
        &src_call_arg("i64", "take_i64(v.is_empty())"),
        "11.4.78 同族：`Vec.is_empty` → bool，传给 i64 形参必须 TypeError（修前 Unknown 放行）",
    );
    assert_type_error(
        &src_call_arg("str", "take_str(v.index_of(1))"),
        "11.4.78 同族：`Vec.index_of` → i64，传给 str 形参必须 TypeError",
    );
    assert_type_error(
        &src_call_arg("i64", "take_i64(v.contains(1))"),
        "11.4.78 同族：`Vec.contains` → bool，传给 i64 形参必须 TypeError",
    );
    assert_type_error(
        &src_call_arg("bool", "take_bool(v.remove(0))"),
        "11.4.78 同族：`Vec.remove` → 内型 i64，传给 bool 形参必须 TypeError",
    );
    // `slice`/`reverse` 的静态类型是**容器**（Array），而 Array vs 标量在
    // `types_compatible` 里是**保守放行**（`:540-546` 只比 Array-vs-Array）
    // ⇒ 不能用"Array 传给 str"当判据，改用**链式**判据：
    // `slice(...).is_empty()` 只有在 slice 已静态收窄成容器时才会解析到
    // `is_empty` → bool（若 slice 是 Unknown，兜底表里没有 is_empty ⇒ Unknown ⇒ 放行）。
    assert_type_error(
        &src_call_arg("i64", "take_i64(v.slice(0, 1).is_empty())"),
        "11.4.78 同族：`Vec.slice` 必须静态收窄为容器（否则 `.is_empty()` 落 Unknown 被放行）",
    );
    assert_type_error(
        &src_call_arg("i64", "take_i64(v.reverse().is_empty())"),
        "11.4.78 同族：`Vec.reverse` 必须静态收窄为容器",
    );
    // 正例：`join` 是 str、`is_empty`/`contains` 是 bool、`index_of` 是 i64、
    // `slice(...).is_empty()` 是 bool —— 类型正确时必须通过（防过度收紧）。
    assert_compiles(
        &src_call_arg("str", "take_str(v.join(\",\"))"),
        "11.4.78 同族：`Vec.join` 必须是 str",
    );
    assert_compiles(
        &src_call_arg("bool", "take_bool(v.is_empty())"),
        "11.4.78 同族：`Vec.is_empty` 必须是 bool",
    );
    assert_compiles(
        &src_call_arg("i64", "take_i64(v.index_of(1))"),
        "11.4.78 同族：`Vec.index_of` 必须是 i64",
    );
    assert_compiles(
        &src_call_arg("bool", "take_bool(v.slice(0, 1).is_empty())"),
        "11.4.78 同族：`Vec.slice(...).is_empty()` 必须是 bool",
    );
}
