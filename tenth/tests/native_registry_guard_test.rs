//! AUDIT-11.4.45 E — native 注册一致性守卫：VM ↔ 解释器 ↔ `resolve_builtin`。
//!
//! ## 为什么需要这个文件
//! native 名在**多处手工同步**（`runtime/natives.rs` 注册序列、
//! `interpreter/natives.rs` 的 `match name` 分派臂、`hir/lower/types.rs`
//! 的 `resolve_builtin` 返回类型表）。前轮审计（`.agents/tmp/prep_audit_native_registry.md`）
//! 实测：VM 侧 190 名、解释器侧 186 名，其中 **4 个公开 native**
//! （`to_utf8`/`to_utf16`/`utf16_to_str`/`bytes_to_str`）在解释器路径**完全不可用**，
//! 而**全仓测试没有一条覆盖解释器路径** ⇒ 套件全绿、缺口长期存在。
//! 本文件把「两侧注册集合必须一致」从**约定**变成**被守护的事实**。
//!
//! ## 设计（AUDIT-11.4.45 E 的三条断言 + 棘轮 + 行为对拍）
//! 1. **VM 侧零扫描**：`Vm::natives` 是 `pub HashMap<String, NativeFn>`
//!    （`runtime/vm/mod.rs:57`），`register_all_natives` 是公开入口 ⇒
//!    直接取**真注册表**的结构化键集合，不用正则猜源码。
//! 2. **解释器侧严格锚点扫描**（唯一可行形态：分派是 `match`，无数据结构）：
//!    只认「与 `match name {` 的臂**同缩进** + 行首为引号串或 `|`」的行，
//!    且只取 `=>` **之前**的引号串。前轮实证：`runtime/natives.rs` 曾在注册 `to_utf8`
//!    的同一行把错误文案写成 `"_to_utf8 ..."`，朴素 grep 会把**错误文案**误当注册项
//!    ⇒ 锚点必须严格（本文件用两个合成语料把这个性质也**测**住）。
//! 3. **断言方向**：解释器臂名集合 **⊇** VM 注册集合（VM 有、解释器无 = 真缺口）。
//! 4. **棘轮（ratchet）**：已知合法差异写进显式台账（`KNOWN_*`），且**双向**断言——
//!    ① 出现**新的**差集 ⇒ 报红（新 native 漏一端）；
//!    ② 台账里的差集**消失**（被修复/对齐）⇒ 也报红并提示更新台账
//!    （否则台账会变成掩盖真实修复的遮羞布）。
//! 5. **`VM ⊆ resolve_builtin`**：`hir/lower/types.rs` 的 native 返回类型表漏项是
//!    **静默**的（兜底 `_ => Ok(Type::Unknown)` ⇒ 静态类型丢失、shape 检查失能），
//!    `resolve_builtin` 是 `pub(super)`（测试无法直接调用）⇒ 只能用严格锚点扫描源码，
//!    配一份**已知未覆盖**台账做棘轮（同样双向断言）。该台账是「欠债清单」，
//!    不是第二份真相表：它只记录**残差**，每还一笔债都要显式从台账删除。
//! 6. **行为对拍（少量、代表性）**：一次 spawn 同源取三轴，**不**为每个 native
//!    各 spawn（190×2 次进程启动过贵）。详见 `parity_probe_*`。
//!
//! ## 不许 `#[ignore]`、不许静默跳过
//! 全部用例进默认套件；失败一律 assert 报红，消息含触发条件与实测值。
//!
//! ## 不可对拍名单（有副作用 / 需交互 / 非确定 ⇒ 不做行为对拍）
//! `exit`（终止进程）、`read_line` / `env_get`（依赖 stdin / 环境）、
//! `tcp_*` / `udp_*` / `http_*`（网络）、`file_*` / `read_file` / `read_bytes` /
//! `write_file` / `write_bytes` / `copy_file` / `rename_file` / `remove_file` /
//! `mkdir` / `list_dir` / `path_*` / `file_size`（文件系统）、
//! `command_*`（子进程）、`random_*` / `random_seed` / `rand` / `randn` /
//! `rand_f32` / `randn_f32`（随机）、`time_*` / `date_*`（时钟）、
//! `async_*`（解释器不支持 async）、`start_grad` / `stop_grad` / `zero_grad` /
//! `backward` / `grad` / `save_weights` / `load_weights`（状态机 / 落盘）、
//! `regex_*`（句柄表，需构造）、`cli_arg*`（依赖 argv）。
//! 名单内的 native 由上面 1-5 条的**集合断言**守护（结构面），不做值对拍（语义面）。

use std::collections::BTreeSet;
use std::path::PathBuf;
use std::process::Command;

use tenth::runtime::natives::register_all_natives;
use tenth::runtime::vm::Vm;

// ════════════════════════════════════════════════════════════════════
// 被测源文件（编译期嵌入：扫描的正是**本测试二进制所编译的**同一版源码）
// ════════════════════════════════════════════════════════════════════

/// 解释器 native 分派（`call_named_fn` 的 `match name`）。
const INTERPRETER_NATIVES_SRC: &str = include_str!("../src/runtime/interpreter/natives.rs");
/// HIR native 返回类型表（`resolve_builtin`）。
const RESOLVE_BUILTIN_SRC: &str = include_str!("../src/hir/lower/types.rs");

// ════════════════════════════════════════════════════════════════════
// 棘轮台账（AUDIT-11.4.45 E）
// ════════════════════════════════════════════════════════════════════

/// VM 注册但解释器**不注册**的名字——已知合法差异。
/// 逐条给理由；新增一条必须在此写明理由（否则守卫报红）。
const KNOWN_VM_ONLY: &[&str] = &[
    // 解释器硬编码拒绝 async：`interpreter/eval.rs` 的
    // "async/await/spawn 不支持解释器路径，请使用 VM"。这三个 native 是 async
    // 语法唯一入口 ⇒ 解释器不注册**不构成新增缺口**（属既定设计差异，且失败响亮）。
    "async_sleep_ms",
    "async_tcp_read",
    "async_tcp_write",
    // f-string 内部 codegen 专用（`compile/bytecode.rs` 为模板字符串发射 str_add），
    // 非公开 API（`tenth/std/prelude.th` 未收录）；解释器侧 String+String 直接走
    // `interpreter/binary.rs`，不需要该 native。
    "str_add",
];

/// 解释器注册但 VM **不注册**的名字——已知合法差异。
/// AUDIT-11.4.45 E 记录的历史差集（4 个 `_` 前缀内部名）已由前轮修复
/// （注册名与解释器臂名统一为公开名），故**当前为空**。
/// 棘轮：一旦重新出现差集 ⇒ 报红（要么补 VM 注册，要么在此写明理由）。
const KNOWN_INTERP_ONLY: &[&str] = &[];

/// `resolve_builtin`（native 返回类型表）**未覆盖**的 VM native——**欠债台账**。
///
/// 语义：这些名字能通过 `lower_expr.rs` 的裸标识符白名单（编译期不报错），
/// 但类型推断走 `resolve_builtin` 的兜底 `_ => Ok(Type::Unknown)` ⇒
/// 调用的静态类型丢失（shape 检查 / 泛型解构失能），**且完全静默**。
///
/// 本台账是**残差**（不是第二份 native 表）：新增 native 若不给返回类型 ⇒
/// 集合并集变大 ⇒ 报红；给某个名字补上返回类型 ⇒ 集合并集变小 ⇒ 报红并
/// 提示从此删除（棘轮，不许台账腐烂）。
const KNOWN_NO_STATIC_RETURN_TYPE: &[&str] = &[
    // 输出/断言：返回 Unit 或与实参同型，历史未登记
    "assert", "assert_eq", "print", "explain_error",
    // 编码/哈希族：返回 str 或 Result/Array，未登记（B批 native）
    "base64_decode", "base64_encode", "from_gbk", "from_utf16", "hex_decode", "hex_encode",
    "md5", "md5_str", "sha256", "sha256_str", "sha512", "sha512_str",
    "str_to_bytes", "str_to_utf16", "to_gbk", "to_utf16", "to_utf8", "utf16_to_str",
    "unicode_nfc", "unicode_nfd", "url_decode", "url_encode", "bytes_to_str",
    // 数值/大数/复数/十进制族：math_* 未登记（标量数学返回 f64）
    "bigint_add", "bigint_mul", "bigint_sub", "complex_add", "complex_div", "complex_mul",
    "complex_sub", "decimal_add", "decimal_div", "decimal_mul", "decimal_sub",
    "math_acos", "math_asin", "math_atan", "math_atan2", "math_ceil", "math_cosh", "math_exp",
    "math_floor", "math_log10", "math_log2", "math_pow", "math_round", "math_sinh", "math_tan",
    "math_tanh",
    // 时间族：返回 Int（epoch 毫秒 / 天数）
    "time_date", "time_datetime", "time_now", "time_now_ms", "time_sleep_ms", "time_time",
    // 随机族：返回 Int / 张量
    "random_float", "random_int", "random_seed",
    // JSON / 文件 / CLI / 张量散点
    "json_decode", "json_encode", "json_encode_pretty", "read_bytes", "rename_file",
    "cli_arg", "cli_args_count", "scatter",
    // f-string 内部 codegen 专用（与 KNOWN_VM_ONLY 同源）
    "str_add",
];

// ════════════════════════════════════════════════════════════════════
// 扫描器（严格锚点；两个合成语料把「不误报」也测住）
// ════════════════════════════════════════════════════════════════════

fn indent_of(line: &str) -> usize {
    line.len() - line.trim_start().len()
}

/// 取一段文本里的双引号字符串内容（这些表里不含转义引号）。
fn quoted(s: &str) -> Vec<String> {
    let cs: Vec<char> = s.chars().collect();
    let mut out = Vec::new();
    let mut i = 0;
    while i < cs.len() {
        if cs[i] == '"' {
            let start = i + 1;
            let mut j = start;
            while j < cs.len() && cs[j] != '"' {
                j += 1;
            }
            if j >= cs.len() {
                break;
            }
            out.push(cs[start..j].iter().collect());
            i = j + 1;
        } else {
            i += 1;
        }
    }
    out
}

/// `//` 之后的内容（行尾注释）——避免把注释里的名字当注册项。
fn cut_comment(line: &str) -> &str {
    match line.find("//") {
        Some(i) => &line[..i],
        None => line,
    }
}

/// `=>` 之前的部分（= 模式位）——避免把臂**体**里的字符串当注册项。
fn cut_body(code: &str) -> &str {
    match code.find("=>") {
        Some(i) => &code[..i],
        None => code,
    }
}

/// 严格锚点扫描 `match name { ... }` 的臂名。
///
/// 锚点规则（三条同时成立）：
/// 1. 与 `match name {` 的臂**同缩进**（`match` 缩进 + 4）；
/// 2. 行首（去缩进后）是引号串 `"..."` 或续行 `|`；
/// 3. 只取 `=>` 之前的名字；遇臂体（更深的缩进）与行尾注释一律不取。
///
/// 因此 arm 体内更深缩进的错误文案（如 `Err(... "_to_utf8 ...")`）与
/// `prelude.th` 风格的注释清单都**不会**被误认为注册项。
fn scan_match_arm_names(src: &str, fn_anchor: &str) -> BTreeSet<String> {
    let lines: Vec<&str> = src.lines().collect();
    let fn_line = lines
        .iter()
        .position(|l| l.contains(fn_anchor))
        .unwrap_or_else(|| panic!("源文件中找不到锚点 `{fn_anchor}`（扫描器需与源码结构同步）"));
    let match_line = lines
        .iter()
        .enumerate()
        .skip(fn_line)
        .find(|(_, l)| l.trim() == "match name {")
        .map(|(i, _)| i)
        .unwrap_or_else(|| panic!("`{fn_anchor}` 之后找不到 `match name {{`"));

    let arm_indent = indent_of(lines[match_line]) + 4;
    let mut names: BTreeSet<String> = BTreeSet::new();
    let mut pending: Vec<String> = Vec::new();

    for line in lines.iter().skip(match_line + 1) {
        let ind = indent_of(line);
        let trimmed = line.trim();
        if ind == arm_indent && trimmed.starts_with("_ =>") {
            break; // match 兜底 ⇒ 区域结束
        }
        if ind != arm_indent {
            pending.clear();
            continue;
        }
        let code = cut_comment(line);
        let pattern = cut_body(code);
        let t = pattern.trim();
        if !(t.starts_with('"') || t.starts_with('|')) {
            pending.clear();
            continue;
        }
        pending.extend(quoted(pattern));
        if code.contains("=>") {
            names.extend(pending.drain(..));
        }
    }
    names
}

/// 严格锚点扫描 `resolve_builtin` 的**模式位**名字
/// （区域 = `fn resolve_builtin` 到兜底 `_ => Ok(Type::Unknown),`）。
fn scan_resolve_builtin_names(src: &str) -> BTreeSet<String> {
    let lines: Vec<&str> = src.lines().collect();
    let start = lines
        .iter()
        .position(|l| l.contains("fn resolve_builtin"))
        .unwrap_or_else(|| panic!("源文件中找不到锚点 `fn resolve_builtin`"));
    let end = lines
        .iter()
        .enumerate()
        .skip(start)
        .find(|(_, l)| l.trim() == "_ => Ok(Type::Unknown),")
        .map(|(i, _)| i)
        .unwrap_or_else(|| panic!("`resolve_builtin` 区域找不到兜底 `_ => Ok(Type::Unknown),`"));

    let mut names: BTreeSet<String> = BTreeSet::new();
    for line in &lines[start..end] {
        names.extend(quoted(cut_body(cut_comment(line))));
    }
    names
}

/// VM 侧真注册表（结构化，零扫描）。
fn vm_native_names() -> BTreeSet<String> {
    let mut vm = Vm::new();
    register_all_natives(&mut vm);
    vm.natives.keys().cloned().collect()
}

fn sorted(v: &[&str]) -> Vec<String> {
    let mut out: Vec<String> = v.iter().map(|s| s.to_string()).collect();
    out.sort();
    out
}

// ════════════════════════════════════════════════════════════════════
// ① 解释器 ⊇ VM（结构面主守卫）
// ════════════════════════════════════════════════════════════════════

#[test]
fn interpreter_dispatch_superset_of_vm_registry() {
    let vm = vm_native_names();
    let interp = scan_match_arm_names(INTERPRETER_NATIVES_SRC, "fn call_named_fn");
    let vm_only: Vec<String> = vm.difference(&interp).cloned().collect();
    let known = sorted(KNOWN_VM_ONLY);
    let new_only: Vec<String> = vm_only.iter().filter(|n| !known.contains(n)).cloned().collect();
    assert!(
        new_only.is_empty(),
        "AUDIT-11.4.45 E 守卫报红：VM 已注册但这些 native **不在解释器分派臂**里：{:?}\n\
         （全量差集 = {:?}；已登记台账 = {:?}）\n\
         ⇒ 解释器路径（TENTH_NO_VM=1）调用它们会报 `undefined function`。\
         请补解释器臂，或在 KNOWN_VM_ONLY 里写明**为什么**该差异合法。\n\
         实测：VM {} 名 / 解释器 {} 臂名。",
        new_only,
        vm_only,
        known,
        vm.len(),
        interp.len()
    );
}

#[test]
fn interpreter_only_arms_are_whitelisted() {
    let vm = vm_native_names();
    let interp = scan_match_arm_names(INTERPRETER_NATIVES_SRC, "fn call_named_fn");
    let interp_only: Vec<String> = interp.difference(&vm).cloned().collect();
    let known = sorted(KNOWN_INTERP_ONLY);
    let new_only: Vec<String> = interp_only
        .iter()
        .filter(|n| !known.contains(n))
        .cloned()
        .collect();
    assert!(
        new_only.is_empty(),
        "AUDIT-11.4.45 E 守卫报红：解释器有分派臂但 VM **没注册**的名字：{:?}\n\
         （全量差集 = {:?}；已登记台账 = {:?}）\n\
         ⇒ 同一份 .th 在默认路径会失败、解释器路径能跑（或反之），跨后端不一致。\
         请补 VM 注册，或在 KNOWN_INTERP_ONLY 里写明理由。",
        new_only,
        interp_only,
        known
    );
}

// ════════════════════════════════════════════════════════════════════
// ② 棘轮：台账必须与现状**精确相等**（消失也要报红）
// ════════════════════════════════════════════════════════════════════

#[test]
fn ratchet_vm_only_ledger_has_no_stale_entries() {
    let vm = vm_native_names();
    let interp = scan_match_arm_names(INTERPRETER_NATIVES_SRC, "fn call_named_fn");
    let vm_only: BTreeSet<String> = vm.difference(&interp).cloned().collect();
    let stale: Vec<String> = KNOWN_VM_ONLY
        .iter()
        .filter(|n| !vm_only.contains(**n))
        .map(|s| s.to_string())
        .collect();
    assert!(
        stale.is_empty(),
        "AUDIT-11.4.45 E 棘轮报红：台账 KNOWN_VM_ONLY 里的这些名字**已不再**是差异：{:?}\n\
         （可能是被补齐 / 改名 / 删除）⇒ 请复验后从台账删除。\
         棘轮要求台账精确等于现状：否则它会在下一次漂移时掩盖真实差集。\n\
         实测差集 = {:?}",
        stale,
        vm_only
    );
}

#[test]
fn ratchet_interp_only_ledger_has_no_stale_entries() {
    let vm = vm_native_names();
    let interp = scan_match_arm_names(INTERPRETER_NATIVES_SRC, "fn call_named_fn");
    let interp_only: BTreeSet<String> = interp.difference(&vm).cloned().collect();
    let stale: Vec<String> = KNOWN_INTERP_ONLY
        .iter()
        .filter(|n| !interp_only.contains(**n))
        .map(|s| s.to_string())
        .collect();
    assert!(
        stale.is_empty(),
        "AUDIT-11.4.45 E 棘轮报红：台账 KNOWN_INTERP_ONLY 里的这些名字**已不再**是差异：{:?}\n\
         ⇒ 请复验后从台账删除。实测差集 = {:?}",
        stale,
        interp_only
    );
}

// ════════════════════════════════════════════════════════════════════
// ③ VM ⊆ resolve_builtin（静默返回类型的唯一守门）
// ════════════════════════════════════════════════════════════════════

#[test]
fn static_return_type_guard_covers_vm_natives() {
    let vm = vm_native_names();
    let covered = scan_resolve_builtin_names(RESOLVE_BUILTIN_SRC);
    let uncovered: Vec<String> = vm.difference(&covered).cloned().collect();
    let known = sorted(KNOWN_NO_STATIC_RETURN_TYPE);
    let new_uncovered: Vec<String> = uncovered
        .iter()
        .filter(|n| !known.contains(n))
        .cloned()
        .collect();
    assert!(
        new_uncovered.is_empty(),
        "AUDIT-11.4.45 E 守卫报红：这些 VM native **既不在 resolve_builtin 的返回类型表**、\
         也不在欠债台账 KNOWN_NO_STATIC_RETURN_TYPE 里：{:?}\n\
         ⇒ 调用它们的静态返回类型会静默回落 `_ => Ok(Type::Unknown)`\
         （shape 检查 / 泛型解构失能，且完全静默）。\
         请在 `hir/lower/types.rs::resolve_builtin` 补返回类型；若确实要给 Unknown，\
         请显式加入台账并写明理由。\n\
         实测：VM {} 名 / resolve_builtin 覆盖 {} 名 / 未覆盖 {} 名。",
        new_uncovered,
        vm.len(),
        covered.len(),
        uncovered.len()
    );
}

#[test]
fn ratchet_static_return_type_ledger_has_no_stale_entries() {
    let vm = vm_native_names();
    let covered = scan_resolve_builtin_names(RESOLVE_BUILTIN_SRC);
    let uncovered: BTreeSet<String> = vm.difference(&covered).cloned().collect();
    let stale: Vec<String> = KNOWN_NO_STATIC_RETURN_TYPE
        .iter()
        .filter(|n| !uncovered.contains(**n))
        .map(|s| s.to_string())
        .collect();
    assert!(
        stale.is_empty(),
        "AUDIT-11.4.45 E 棘轮报红：欠债台账里的这些名字**已被 resolve_builtin 覆盖**\
         （或已改名/删除）：{:?}\n\
         ⇒ 欠债已还，请从 KNOWN_NO_STATIC_RETURN_TYPE 删除对应条目\
         （棘轮：不许台账长期掩盖已修好的项）。实测未覆盖 {} 名。",
        stale,
        uncovered.len()
    );
}

// ════════════════════════════════════════════════════════════════════
// ④ 扫描器自身：严格锚点必须**不误报**（合成语料，含前轮实证的陷阱）
// ════════════════════════════════════════════════════════════════════

#[test]
fn scanner_strict_anchor_ignores_error_message_strings() {
    // 复刻前轮实证的陷阱：注册名是 `to_utf8`，而**错误文案**里含 `_to_utf8`。
    // 朴素 grep '"_to_utf8"' 会把错误文案误判成「已注册」⇒ 锚点必须严格。
    let synthetic = r#"
    pub(super) fn call_named_fn(
        &mut self, name: &str, args: &[Value], _span: &Span,
    ) -> TenthResult<Option<Value>> {
        match name {
            "to_utf8" => {
                if args.is_empty() {
                    return Err(TenthError::RuntimeError { line: None, col: None,
                        message: "_to_utf8 需要 1 个 String 参数".into() });
                }
                // 注释里的 "comment_only_name" 不是注册项
                return Ok(Some(Value::Vec(Rc::new(RefCell::new(bytes)))));
            }
            _ => {}
        }
    }
"#;
    let got = scan_match_arm_names(synthetic, "fn call_named_fn");
    let want: BTreeSet<String> = ["to_utf8".to_string()].into_iter().collect();
    assert_eq!(
        got, want,
        "严格锚点扫描器把臂体里的错误文案/注释误当注册项：扫描结果 {:?}（期望 {{to_utf8}}）\
         —— 朴素 grep 的误报机制（AUDIT-11.4.45 E 实证）必须被锚点排除",
        got
    );
}

#[test]
fn scanner_handles_grouped_and_continuation_arms() {
    let synthetic = r#"
    pub(super) fn call_named_fn(
        &mut self, name: &str, args: &[Value], _span: &Span,
    ) -> TenthResult<Option<Value>> {
        match name {
            "a" | "b" => {
                Ok(None)
            }
            "c" | "d"
            | "e" => Ok(None),
            _ => {}
        }
    }
"#;
    let got = scan_match_arm_names(synthetic, "fn call_named_fn");
    let want: BTreeSet<String> = ["a", "b", "c", "d", "e"]
        .iter()
        .map(|s| s.to_string())
        .collect();
    assert_eq!(
        got, want,
        "分组臂（`\"a\" | \"b\" =>`）与续行臂（`\"c\" | \"d\"` + `| \"e\" =>`）都必须被完整收集；\
         实测 {:?}",
        got
    );
}

#[test]
fn scanners_scan_live_regions() {
    // 防空转：扫描区域必须真的命中（锚点与源码结构一旦脱节，上面的守卫会「静默全绿」）。
    let interp = scan_match_arm_names(INTERPRETER_NATIVES_SRC, "fn call_named_fn");
    let covered = scan_resolve_builtin_names(RESOLVE_BUILTIN_SRC);
    assert!(
        interp.len() > 150,
        "解释器分派臂只扫到 {} 个名字 —— 锚点/区域判定已与源码结构脱节（守卫会静默失效）",
        interp.len()
    );
    assert!(
        covered.len() > 100,
        "resolve_builtin 只扫到 {} 个名字 —— 锚点/区域判定已与源码结构脱节（守卫会静默失效）",
        covered.len()
    );
    assert!(
        interp.contains("println") && covered.contains("println"),
        "两个扫描器都必须能扫到锚点样本 `println`（interp={} / resolve_builtin={}）",
        interp.contains("println"),
        covered.contains("println")
    );
}

// ════════════════════════════════════════════════════════════════════
// ⑤ 行为对拍（少量、代表性；一次 spawn 同源取三轴，全文件只 2 次 spawn）
// ════════════════════════════════════════════════════════════════════

const TENTH_EXE: &str = env!("CARGO_BIN_EXE_tenth");
const TENTH_DIR: &str = env!("CARGO_MANIFEST_DIR");

/// 一条语料内的调用（覆盖：W4 刚补齐的 4 个编码 native + 同族别名 + 哈希/Base64
/// 的确定性输出 + 一条纯算术对照）。**不含**不可对拍名单里的 native。
const PARITY_PROBE_SRC: &str = r#"fn main() {
    println("A1=" + hex_encode(to_utf8("AB")));
    println("A2=" + hex_encode(str_to_bytes("AB")));
    println("A3=" + bytes_to_str(str_to_bytes("AB")));
    println("A4=" + bytes_to_str(to_utf8("AB")));
    println("B1=" + utf16_to_str(str_to_utf16("AB")));
    println("B2=" + utf16_to_str(to_utf16("AB")));
    println("B3=" + from_utf16(str_to_utf16("AB")));
    println("C1=" + md5_str("abc"));
    println("C2=" + sha256_str("abc"));
    println("C3=" + base64_encode(str_to_bytes("abc")));
    println("C4=" + hex_encode(str_to_bytes("abc")));
    println("D1=" + format("{}", 1 + 2 * 3));
}
"#;

/// 金标准（语言语义应有的输出；「两条路径都错成一样」也会被这里抓到）。
const PARITY_GOLD: &[&str] = &[
    "A1=4142",
    "A2=4142",
    "A3=AB",
    "A4=AB",
    "B1=AB",
    "B2=AB",
    "B3=AB",
    "C1=900150983cd24fb0d6963f7d28e17f72",
    "C2=ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
    "C3=YWJj",
    "C4=616263",
    "D1=7",
];

fn run_probe(file: &PathBuf, interpreter: bool) -> std::process::Output {
    let mut cmd = Command::new(TENTH_EXE);
    cmd.arg("run").arg(file).current_dir(TENTH_DIR);
    if interpreter {
        // 显式设置（main.rs 用 `env::var(...).is_ok()` 判定，设成 "0" 也算已设置）。
        cmd.env("TENTH_NO_VM", "1");
    } else {
        // 显式移除，避免父进程环境把「默认路径」静默跑成解释器（假绿）。
        cmd.env_remove("TENTH_NO_VM");
    }
    cmd.output()
        .unwrap_or_else(|e| panic!("spawn 被测二进制失败：{}（{}）", TENTH_EXE, e))
}

#[test]
fn representative_natives_parity_across_backends() {
    let dir = PathBuf::from(TENTH_DIR).join("target").join("tmp_native_registry_guard");
    std::fs::create_dir_all(&dir).expect("创建临时目录失败");
    let file = dir.join("zz_probe_parity.th");
    std::fs::write(&file, PARITY_PROBE_SRC).expect("写临时 .th 失败");

    // 一次 spawn 同源取三轴（每后端各一次，全文件 2 次 spawn）。
    let def = run_probe(&file, false);
    let interp = run_probe(&file, true);

    let d_out = String::from_utf8_lossy(&def.stdout).to_string();
    let i_out = String::from_utf8_lossy(&interp.stdout).to_string();
    let d_err = String::from_utf8_lossy(&def.stderr).to_string();
    let i_err = String::from_utf8_lossy(&interp.stderr).to_string();

    // (a) 差分：stdout 原始字节 + 退出码必须一致。
    assert_eq!(
        def.stdout, interp.stdout,
        "AUDIT-11.4.45 E 对拍报红：两路径 stdout 不一致（VM 侧 {} 字节 / 解释器侧 {} 字节）\n\
         --- 默认路径(VM) stderr ---\n{}\n--- 解释器路径 stderr ---\n{}",
        def.stdout.len(),
        interp.stdout.len(),
        d_err,
        i_err
    );
    assert_eq!(
        def.status.code(),
        interp.status.code(),
        "AUDIT-11.4.45 E 对拍报红：两路径退出码不一致（VM {:?} / 解释器 {:?}）\n\
         --- 默认路径 stderr ---\n{}\n--- 解释器路径 stderr ---\n{}",
        def.status.code(),
        interp.status.code(),
        d_err,
        i_err
    );
    // (b) 金标准：两路径输出都必须等于语言语义应有的行（防「都错成一样」的假绿）。
    for (label, out, err) in [
        ("默认路径(VM)", &d_out, &d_err),
        ("解释器路径(TENTH_NO_VM=1)", &i_out, &i_err),
    ] {
        let lines: Vec<&str> = out.lines().collect();
        assert_eq!(
            lines,
            PARITY_GOLD.to_vec(),
            "AUDIT-11.4.45 E 金标准报红：{} 的输出不等于期望（含 W4 补齐的 to_utf8/to_utf16/\
             utf16_to_str/bytes_to_str 行为对拍）\nstderr = {}",
            label,
            err
        );
    }
}
