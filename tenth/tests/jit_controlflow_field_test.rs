//! QA-20260831 回归守护：JIT 控制流 block 管理 + VM 引用语义（用户反馈修复轮）
//!
//! 背景（用户反馈 deepseek-qa-20260831，运行时部任务 A/B）：
//! - 大量含循环/控制流的程序在默认 VM/JIT 路径打印 Cranelift panic
//!   （ssa.rs:349 `!is_sealed` 断言 / frontend.rs:626 "block already filled"），
//!   功能靠 M2-A5 catch_unwind fallback 兜底但 stderr 有噪音。根因：translator
//!   在 leader 入口即 seal 块，循环回边/continue 的后向 Jump 再声明前驱触发断言；
//!   以及 if-then 提前 Ret 后的死 Jump 被发射进已填充块。
//! - `let r = &mut p2; r.x = 5.0` 默认 VM 报「无法设置字段」（set_field 缺
//!   MutRef 写穿分支）；`&mut vec` 后 `vec[i]` 报「无法索引」（IndexGet 不解包
//!   MakeMutRef 回写的 Shared 包裹值）。
//! - 连带修复（同轮发现的同族缺口）：bytecode.rs tail_call_ok 泄漏（StructLiteral
//!   字段调用被误编译成 TailCall）、JIT host_load_field/host_store_field 与 VM
//!   get_field/set_field 语义对齐、JIT 栈标量陈旧跟踪覆盖 Load 值（adam 错值）、
//!   JIT Pop 空栈钳 0（AUDIT-11.4.43 TryFromIntError）。
//!
//! 守护方式：每个触发模式以子进程跑 `tenth.exe run <tmp.th>` 两条路径——
//! 默认（VM/JIT）与解释器（TENTH_NO_VM=1）——断言：
//! 1. 两条路径 exit 0；
//! 2. 默认路径 stderr 无 panic 噪音（不再是 fallback 兜底，而是 JIT 真编译）；
//! 3. 两条路径 stdout 逐字节一致（VM=解释器=JIT 对拍）。

use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};

const TENTH_EXE: &str = env!("CARGO_BIN_EXE_tenth");
const TENTH_DIR: &str = env!("CARGO_MANIFEST_DIR");

static COUNTER: AtomicU64 = AtomicU64::new(0);

/// 写临时 .th 并以指定模式运行，返回 (exit_code, stdout, stderr)。
fn run_th(prog: &str, interpreter: bool) -> (i32, String, String) {
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!("tenth_qa_20260831_{}_{}", std::process::id(), n));
    std::fs::create_dir_all(&dir).unwrap();
    let file = dir.join("case.th");
    std::fs::write(&file, prog).unwrap();

    let mut cmd = Command::new(TENTH_EXE);
    cmd.arg("run").arg(&file).current_dir(TENTH_DIR);
    if interpreter {
        cmd.env("TENTH_NO_VM", "1");
    }
    let out = cmd.output().expect("运行 tenth.exe 失败");
    let stdout = String::from_utf8_lossy(&out.stdout).to_string();
    let stderr = String::from_utf8_lossy(&out.stderr).to_string();
    let code = out.status.code().unwrap_or(-1);
    let _ = std::fs::remove_dir_all(&dir);
    (code, stdout, stderr)
}

/// 三路径对拍守护：默认（VM/JIT）= 解释器；默认路径无 panic；双 exit 0。
fn guard(name: &str, prog: &str) {
    let (code_j, out_j, err_j) = run_th(prog, false);
    let (code_i, out_i, err_i) = run_th(prog, true);

    assert_eq!(code_j, 0,
        "[{name}] 默认路径（VM/JIT）应 exit 0，实际 {code_j}\n--- stdout ---\n{out_j}\n--- stderr ---\n{err_j}");
    assert_eq!(code_i, 0,
        "[{name}] 解释器路径应 exit 0，实际 {code_i}\n--- stdout ---\n{out_i}\n--- stderr ---\n{err_i}");
    assert!(
        !err_j.contains("panicked") && !err_j.contains("stack overflow"),
        "[{name}] 默认路径不应有 panic 噪音（JIT 应真编译执行而非 fallback 兜底）\n--- stderr ---\n{err_j}"
    );
    assert!(
        !err_i.contains("panicked"),
        "[{name}] 解释器路径不应 panic\n--- stderr ---\n{err_i}"
    );
    assert_eq!(
        out_j.trim(),
        out_i.trim(),
        "[{name}] 默认路径与解释器输出不一致（VM=解释器=JIT 对拍失败）\n--- VM/JIT ---\n{out_j}\n--- 解释器 ---\n{out_i}"
    );
}

// ── 任务 1：JIT 控制流（循环回边 seal 时序 + 死代码）──────────────────────

/// `..=` 闭区间 for 循环（02b_for_inclusive.th 模式，原 ssa.rs:349 panic）。
#[test]
fn qa_for_inclusive_loop() {
    guard("for_inclusive", r#"
fn main() {
    let mut sum: i64 = 0;
    for i in 2..=5 {
        sum = sum + i;
    };
    println(sum);
}
"#);
}

/// for + break + continue（02c_for_break_continue.th 模式，原 ssa.rs:349 panic）。
#[test]
fn qa_for_break_continue() {
    guard("for_break_continue", r#"
fn main() {
    let mut count: i64 = 0;
    for i in 1..100 {
        if i == 50 { break; };
        if i % 2 == 0 { continue; };
        count = count + 1;
    };
    println(count);
}
"#);
}

/// while 循环 + 闭包捕获（while/闭包触发模式）。
#[test]
fn qa_while_with_closure() {
    guard("while_closure", r#"
fn main() {
    let factor = 3;
    let mul = |x| x * factor;
    let mut n = 0;
    let mut acc = 0;
    while n < 5 {
        acc = acc + mul(n);
        n = n + 1;
    };
    println(acc);
}
"#);
}

/// 嵌套控制流：嵌套 for + if + 提前 return 的被调函数（frontend.rs:626 死代码模式）。
#[test]
fn qa_nested_controlflow_and_early_return() {
    guard("nested_cf_early_return", r#"
fn safe_div(a: i64, b: i64) -> i64 {
    if b == 0 { return 0; };
    a / b
}

fn main() {
    println(safe_div(10, 0));
    println(safe_div(84, 2));
    let mut total = 0;
    for i in 0..3 {
        for j in 0..3 {
            if j > i { continue; };
            total = total + i * j;
        };
    };
    println(total);
}
"#);
}

/// 递归 + match 枚举字段访问（linked_list 模式：host_load_field 枚举语义）。
#[test]
fn qa_recursive_enum_field_access() {
    guard("recursive_enum_field", r#"
enum List {
    Cons(value: i64, next: List),
    Nil,
}

fn print_list(list: List) {
    match list {
        List::Nil => {},
        _ => {
            println(list.value);
            print_list(list.next);
        },
    };
}

fn main() {
    let l = List::Cons(value: 10, next: List::Cons(value: 20, next: List::Nil));
    print_list(l);
    println("done");
}
"#);
}

// ── 任务 2：VM 经 &mut 引用语义 ──────────────────────────────────────────

/// `let r = &mut p2; r.x = 5.0`（19_struct_default.th 第 28-31 行模式，
/// 原「无法设置字段」——set_field 缺 MutRef 写穿）。
#[test]
fn qa_mutref_field_set() {
    guard("mutref_field_set", r#"
struct Point {
    x: f64,
    y: f64,
}

fn main() {
    let mut p2 = Point { x: 0.0, y: 0.0 };
    let r = &mut p2;
    r.x = 5.0;
    println(p2.x);
}
"#);
}

/// `fn f(q: &mut Queue) { q.front = ... }`（queue.th 模式：&mut 参数字段赋值）。
#[test]
fn qa_mutref_param_field_set() {
    guard("mutref_param_field_set", r#"
struct Counter {
    n: i64,
    items: Vec<i64>,
}

fn bump(c: &mut Counter) {
    c.n = c.n + 1;
    c.items.push(c.n);
}

fn main() {
    let mut c = Counter { n: 0, items: Vec::new() };
    bump(&mut c);
    bump(&mut c);
    bump(&mut c);
    println(c.n);
    println(c.items.len());
}
"#);
}

/// `&mut vec` 之后索引 `vec[i]`（bintree.th 模式：IndexGet 解包 Shared）。
#[test]
fn qa_index_after_mutref() {
    guard("index_after_mutref", r#"
fn fill(v: &mut Vec<i64>) {
    v.push(1);
    v.push(2);
    v.push(3);
}

fn main() {
    let mut v = Vec::new();
    fill(&mut v);
    let mut total = 0;
    for i in 0..v.len() {
        total = total + v[i];
    };
    println(total);
}
"#);
}

// ── 同轮连带修复的守护 ───────────────────────────────────────────────────

/// tail_call_ok 泄漏：StructLiteral 字段内的 native 调用曾被误编译成 TailCall
/// （hashset `new()` 返回字段值而非 struct，JIT 路径「没有方法」错值）。
#[test]
fn qa_struct_literal_field_call() {
    guard("struct_literal_field_call", r#"
struct Holder {
    inner: HashMap,
}

fn make() -> Holder {
    Holder { inner: HashMap::new() }
}

fn main() {
    let h = make();
    h.inner.insert("a", 1);
    println(h.inner.contains_key("a"));
    println(h.inner.len());
}
"#);
}

/// 栈标量陈旧跟踪：(1.0 - 0.9) * t（adam.th 静默错值根因——原生 Sub 消费
/// 右操作数后该偏移跟踪残留，Load 压入非标量未清，物化把陈旧标量写回覆盖）。
#[test]
fn qa_scalar_sub_tensor_mix() {
    guard("scalar_sub_tensor_mix", r#"
fn main() {
    let t = tensor[[10.0]];
    let m = (1.0 - 0.9) * t;
    println(m.sum());
}
"#);
}

/// Union 字段修改（AUDIT-11.4.43：JIT Pop 空栈钳 0，原低化 TryFromIntError panic）。
#[test]
fn qa_union_field_assign() {
    guard("union_field_assign", r#"
union Number { integer: i64, float: f64 }

fn main() {
    let mut n3 = Number { integer: 1 };
    n3.integer = 100;
    println(n3.integer);
}
"#);
}

/// autodiff 训练循环（adam/07_autodiff 模式：循环内 new_grad/param/backward，
/// 守护 JIT 循环 + 标量/张量混合运算与解释器逐步一致）。
#[test]
fn qa_autodiff_training_loop() {
    guard("autodiff_training_loop", r#"
fn main() {
    let x = tensor[[1.0, 2.0, 3.0, 4.0]];
    let target = tensor[[3.0, 5.0, 7.0, 9.0]];
    let mut w = tensor[[0.0]];
    let mut b = tensor[[0.0]];
    for i in 0..5 {
        new_grad();
        zero_grad();
        w = param(w);
        b = param(b);
        let pred = w * x + b;
        let loss = ((pred - target) * (pred - target)).mean();
        backward(loss);
        stop_grad();
        let gw = grad(w);
        let gb = grad(b);
        println(loss.sum());
        w = w - 0.01 * gw;
        b = b - 0.01 * gb;
    };
    println(w.sum());
    println(b.sum());
}
"#);
}

/// QA-20260831（静默错值回归 A）：裸表达式 main + 前置 while + 模块函数调用 +
/// 4 层 `&&` 短路链、最外层 right 为复合表达式（Load+Const+Eq 三条指令）。
///
/// 根因：translator 分析器对比较指令（Eq/Neq/Lt/Gt/Lte/Gte）**无条件**预测结果
/// 为 Bool 标量，而发射端 emit_binop 仅在「两操作数均为同类 I32/F64 标量」时
/// 才走原生比较（emit_native_cmp 只支持 I32/F64）——`wsub.len() == 4` 的 len
/// 是 MethodCall 结果（无标量跟踪），发射走通用 hostcall，但分析仍说局部
/// `wsub_ok` 恒为 Bool → 块入口 cur_local_kinds=Bool → 后续 Load(8) 专用化读
/// local_scalars[8] ——该槽只由「运行期未执行的分支」内的专用化 Store 创建
/// （发射期建槽）→ 读未初始化槽（宿主栈残留），随二进制布局在 true/false
/// 间翻转（probe5 在部分布局输出 false、部分 true，TENTH_NO_VM=1 恒 true）。
/// 修复：分析端比较预测与发射端资格逐一对齐（仅同类 I32/F64 → Bool）。
/// 对拍断言：JIT 路径与解释器输出一致（均为 true）。
#[test]
fn qa_shortcut_chain_bare_expr_module_mix() {
    guard("shortcut_chain_bare_expr_module_mix", r#"
use std::data::sampler::*
use std::random::random::rand_seed

fn main() {
    rand_seed(42);
    let perm = shuffle_indices(5);
    let mut s = 0;
    let mut i = 0;
    while i < perm.len() {
        s = s + perm.get(i);
        i = i + 1;
    }
    let sub = random_sample([10, 20, 30, 40, 50], 3);
    let sl = sub.len();
    let repl = random_sample_with_replacement([10, 20, 30], 5);
    let rl = repl.len();
    let wsub = weighted_sample([10, 20, 30], [0.0, 0.0, 1.0], 4);
    let mut wsub_ok = wsub.len() == 4;
    let mut wi = 0;
    while wi < wsub.len() {
        let wv = wsub.get(wi);
        if wv != 30 {
            wsub_ok = false;
        }
        wi = wi + 1;
    }
    let strat = stratified_indices([0, 0, 1, 1, 2, 2, 2], 1);
    let stl = strat.len();
    s == 10 && sl == 3 && rl == 5 && wsub_ok && stl == 3
}
"#);
}

/// QA-20260831（静默错值回归 B）：TCO 泄漏——StructLiteral 字段调用被误编译成
/// TailCall（尾位置标记泄漏进容器构造的字段表达式）。`make()` 体内
/// `Holder { inner: HashMap::new() }` 的 `HashMap::new()`（0 参调用恰与外层
/// 0 参函数匹配）被误发射为 TailCall 终止指令 → PushStr/NewStruct/Ret 全部
/// 不可达 → make 返回 Map 而非 Struct → `h.inner.insert` 报「没有方法」。
/// 修复：compile_expr 入口消费 tail_call_ok（子表达式一律非尾位置），尾位置
/// 仅由 If/Block 分支与函数体 compile 显式转发。
/// （原 qa_struct_literal_field_call 已守护该场景；本测试以「函数返回带
/// 调用字段的 StructLiteral + 裸表达式 main」的复合形态加固，确保入口消费
/// 式修复不依赖单一形态。）
#[test]
fn qa_tco_leak_struct_literal_bare_expr() {
    guard("tco_leak_struct_literal_bare_expr", r#"
struct Holder {
    inner: HashMap,
}

fn make() -> Holder {
    Holder { inner: HashMap::new() }
}

fn main() {
    let h = make();
    h.inner.insert("a", 1);
    h.inner.insert("b", 2);
    h.inner.contains_key("a") && h.inner.len() == 2
}
"#);
}
