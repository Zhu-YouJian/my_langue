//! 常驻跨后端对拍守护：默认路径（VM/JIT）与解释器路径（`TENTH_NO_VM=1`）必须语义一致，
//! 且失败必须响亮。
//!
//! ## 立项理由
//! 全仓 137 个测试文件中，此前**没有任何一条常驻断言**「VM/JIT 与解释器语义必须一致，
//! 且失败必须响亮」——AUDIT-11.4.50 / 11.4.54 / 11.4.55 三条红线**全部由一次性探针
//! 发现**、探针用完即弃，且 CI 里 16 条大开销用例按构造不跑。本文件把这条防线变成
//! **被守护的事实**：任何一条语料的两路径行为分叉，默认套件立刻报红。
//!
//! ## 设计要点（逐条对应硬约束）
//! 1. **一次 spawn 同源取三轴**：每个后端只调用一次 `Command::output()`，同时拿到
//!    `(stdout, stderr 是否非空, 退出码)`；同一语料只写一份 `.th` 文件，两条路径读同一份。
//! 2. **二进制定位**：`env!("CARGO_BIN_EXE_tenth")`（cargo 为集成测试提供的被测二进制），
//!    不硬编码 `target/release/tenth.exe`。
//! 3. **显式控制子进程 env**：默认路径 `.env_remove("TENTH_NO_VM")`，解释器路径
//!    `.env("TENTH_NO_VM","1")`。原因（实测）：`tenth/src/main.rs:194` 用
//!    `std::env::var("TENTH_NO_VM").is_ok()` 判定，**设成 `0` 也算「已设置」**，会把
//!    「默认路径」静默跑成解释器 → 报告「全一致」实为**假绿**。此处不依赖父进程 env 干净。
//! 4. **防假绿双保险**：
//!    (a) **差分断言**——两路径的 `(stdout 原始字节, stderr 非空标志, 退出码)` 必须一致；
//!    (b) **金标准断言**——同时断言输出等于语言语义应有的期望值（`Expect::ParityLines` /
//!        `ParityLoud`），这样「两条路径都错成一样」也会被抓到。
//! 5. **不用「具体缺陷行为」当后端指纹**（如 char 在 VM 是 120、某条未修分歧）——那会在
//!    缺陷修复时误报。唯一的「两条路径确实不是同一后端」的哨兵是 `AUDIT-11.4.56` 那条
//!    **已知分歧**（VM 写穿全局表 vs 解释器按值捕获，见下）。**诚实说明**：目前没有找到
//!    「语义上稳定」的其它哨兵；一旦 11.4.56 被修好，本守护会按棘轮设计报红并提示从台账
//!    移除，届时该哨兵随之消失，必须另行确定一条稳定判据。
//! 6. **不许 `#[ignore]`、不许静默跳过**：所有语料都进默认套件；`check()` 内无 `continue`
//!    也无提前 `return`，失败一律 assert 报红。
//! 7. **失败必须响亮**：所有断言消息含 `AUDIT` 编号 + `slug` + **两路径原文输出**。
//! 8. **语料可增长**：新增一条探针 = 在文件末尾加一条 `guard_case!` declaration，
//!    **不需要改测试骨架**（`check` / `Case` / `Expect` 不动）。
//! 9. **不 flaky**：不比对耗时，不依赖网络 / 时间 / 随机数 / 工作目录；临时 `.th` 文件写在
//!    `tenth/target/tmp_backend_parity_guard/`（由 `CARGO_MANIFEST_DIR` 推导，与 cwd 无关）。
//!
//! ## 已知分歧台账（ratchet）
//! `AUDIT-11.4.56`（候选）**闭包体内写全局**：默认路径写穿全局表、解释器按值捕获，
//! **两侧退出码都是 0** 的静默分歧。本守护把它纳入语料但断言其**仍然分歧**：将来一旦被
//! 修好，守护会**报红并提示从台账移除**，形成棘轮——不许为了让守护今天全绿而剔除它。

use std::path::{Path, PathBuf};
use std::process::Command;

/// 被测二进制（同 crate 的 bin target，`cargo test` 会自动构建）。
const TENTH_EXE: &str = env!("CARGO_BIN_EXE_tenth");
/// 包根目录（= `tenth/`）：子进程 cwd 与临时目录推导基准。
const TENTH_DIR: &str = env!("CARGO_MANIFEST_DIR");

// ════════════════════════════════════════════════════════════════════
// 一次 spawn 取三轴
// ════════════════════════════════════════════════════════════════════

/// 一次子进程运行的**同源三轴**：退出码 / stdout / stderr 是否非空。
struct Run {
    /// 退出码；子进程异常终止（无退出码）时为 `None`。
    exit: Option<i32>,
    /// 原始 stdout 字节（差分断言用原始字节，避免 lossy 转换掩盖差异）。
    stdout_bytes: Vec<u8>,
    /// stdout 文本（仅用于报告与金标准逐行比对）。
    stdout: String,
    /// stderr 文本（仅用于报告与「响亮失败」的关键字断言）。
    stderr: String,
}

impl Run {
    /// 轴 2：stderr 是否非空（**只取标志**——VM 与解释器的报错前缀本就不同）。
    fn stderr_nonempty(&self) -> bool {
        !self.stderr.is_empty()
    }

    /// 报告用：把两路径的原文输出原样贴出（失败消息必须能定位）。
    fn dump(&self) -> String {
        format!(
            "  exit = {:?}\n  --- stdout ({} bytes) ---\n{}\n  --- stderr ({} bytes) ---\n{}",
            self.exit,
            self.stdout_bytes.len(),
            indent(&self.stdout),
            self.stderr.len(),
            indent(&self.stderr),
        )
    }
}

fn indent(s: &str) -> String {
    if s.is_empty() {
        return "  <empty>".to_string();
    }
    s.lines().map(|l| format!("  | {}", l)).collect::<Vec<_>>().join("\n")
}

/// **一次** `Command::output()` → 三轴。`interpreter=true` 走 `TENTH_NO_VM=1`；
/// `false` 显式 `env_remove("TENTH_NO_VM")`（不依赖父进程 env 干净，见文件头 §3）。
fn run_once(file: &Path, interpreter: bool) -> Run {
    let mut cmd = Command::new(TENTH_EXE);
    cmd.arg("run").arg(file).current_dir(TENTH_DIR);
    if interpreter {
        cmd.env("TENTH_NO_VM", "1");
    } else {
        cmd.env_remove("TENTH_NO_VM");
    }
    let out = cmd
        .output()
        .unwrap_or_else(|e| panic!("spawn 被测二进制失败：{}（{}）", TENTH_EXE, e));
    Run {
        exit: out.status.code(),
        stdout: String::from_utf8_lossy(&out.stdout).to_string(),
        stdout_bytes: out.stdout,
        stderr: String::from_utf8_lossy(&out.stderr).to_string(),
    }
}

// ════════════════════════════════════════════════════════════════════
// 语料 declaration + 期望
// ════════════════════════════════════════════════════════════════════

/// 期望类型——每条语料必须选一种，**没有「跳过」这一种**。
enum Expect {
    /// 两路径一致，且两侧 stdout **逐行等于**这些行（金标准：语言语义应有的输出）。
    ParityLines(&'static [&'static str]),
    /// 两路径一致，且两侧 **stdout 为空、退出码非 0、stderr 非空且含指定子串**
    /// （金标准：越界必须**响亮失败**，不许静默返回 `()`）。
    ParityLoud { stderr_contains: &'static [&'static str] },
    /// 已知分歧台账（ratchet）：两路径**必须仍然分歧**；一旦一致即报红。
    KnownDivergence,
}

struct Case {
    /// 语料标识（同时用作临时文件名，保证同一语料两条路径读同一份源码）。
    slug: &'static str,
    /// 人类可读语料名。
    name: &'static str,
    /// 审计编号（失败消息必须带）。
    audit: &'static str,
    /// .th 源码（内联最小程序，ASCII，避免控制台编码干扰）。
    src: &'static str,
    expect: Expect,
}

/// 写一份源码 → 两条路径各 spawn 一次（同源）→ 返回两路径三轴。
fn capture(case: &Case) -> (Run, Run) {
    let dir: PathBuf = Path::new(TENTH_DIR).join("target").join("tmp_backend_parity_guard");
    std::fs::create_dir_all(&dir)
        .unwrap_or_else(|e| panic!("创建临时语料目录失败 {}：{}", dir.display(), e));
    let file = dir.join(format!("{}.th", case.slug));
    std::fs::write(&file, case.src)
        .unwrap_or_else(|e| panic!("写入临时语料失败 {}：{}", file.display(), e));

    // 每个后端恰好一次 spawn；两条路径读同一份源码文件。
    let default_run = run_once(&file, false);
    let interp_run = run_once(&file, true);

    let _ = std::fs::remove_file(&file);
    (default_run, interp_run)
}

/// 失败消息统一格式：AUDIT 编号 + slug + 两路径原文输出。
fn ctx(case: &Case, d: &Run, i: &Run) -> String {
    format!(
        "\n  AUDIT   : {}\n  语料    : {} ({})\n  默认路径(VM/JIT) :\n{}\n  解释器路径(TENTH_NO_VM=1) :\n{}",
        case.audit,
        case.name,
        case.slug,
        d.dump(),
        i.dump(),
    )
}

/// 差分断言（保险 a）：三轴一致。
fn assert_axes_equal(case: &Case, d: &Run, i: &Run) {
    assert!(
        d.exit.is_some() && i.exit.is_some(),
        "两路径都必须有退出码（子进程异常终止视为失败）{}",
        ctx(case, d, i)
    );
    assert_eq!(
        d.stdout_bytes, i.stdout_bytes,
        "差分断言失败：两路径 stdout 不一致（跨后端语义分叉）{}",
        ctx(case, d, i)
    );
    assert_eq!(
        d.stderr_nonempty(),
        i.stderr_nonempty(),
        "差分断言失败：两路径 stderr 非空标志不一致（一侧响亮、一侧静默）{}",
        ctx(case, d, i)
    );
    assert_eq!(
        d.exit, i.exit,
        "差分断言失败：两路径退出码不一致{}",
        ctx(case, d, i)
    );
}

/// 语料检查入口（骨架——新增语料不需要改这里）。
fn check(case: &Case) {
    let (d, i) = capture(case);
    match &case.expect {
        Expect::ParityLines(gold) => {
            // (a) 差分
            assert_axes_equal(case, &d, &i);
            // (b) 金标准：两路径都必须输出语言语义应有的行——「都错成一样」也报红
            for (label, run) in [("默认路径(VM/JIT)", &d), ("解释器路径(TENTH_NO_VM=1)", &i)] {
                let lines: Vec<&str> = run.stdout.lines().collect();
                assert_eq!(
                    lines,
                    gold.to_vec(),
                    "金标准断言失败：{} 的输出不等于语言语义期望值 {:?}{}",
                    label,
                    gold,
                    ctx(case, &d, &i)
                );
            }
        }
        Expect::ParityLoud { stderr_contains } => {
            // (a) 差分
            assert_axes_equal(case, &d, &i);
            // (b) 金标准：响亮失败——非零退出 + stdout 为空 + stderr 报出越界原因
            for (label, run) in [("默认路径(VM/JIT)", &d), ("解释器路径(TENTH_NO_VM=1)", &i)] {
                assert!(
                    run.stdout_bytes.is_empty(),
                    "金标准断言失败：{} 应 stdout 为空（越界不许先静默输出 `()`）{}",
                    label,
                    ctx(case, &d, &i)
                );
                assert!(
                    matches!(run.exit, Some(code) if code != 0),
                    "金标准断言失败：{} 应以非零退出码响亮失败{}",
                    label,
                    ctx(case, &d, &i)
                );
                assert!(
                    run.stderr_nonempty(),
                    "金标准断言失败：{} 应把越界原因写到 stderr{}",
                    label,
                    ctx(case, &d, &i)
                );
                for needle in *stderr_contains {
                    assert!(
                        run.stderr.contains(needle),
                        "金标准断言失败：{} 的 stderr 应含 {:?}{}",
                        label,
                        needle,
                        ctx(case, &d, &i)
                    );
                }
            }
        }
        Expect::KnownDivergence => {
            let differs = d.stdout_bytes != i.stdout_bytes || d.exit != i.exit;
            assert!(
                differs,
                "ratchet 失败：AUDIT-11.4.56 已知分歧（闭包体内写全局）的两条路径**已经一致**\
                 （stdout 与退出码都相同）——该分歧已被修复或行为已改变；请复验后把它从\
                 「已知分歧台账」移除，并重新为「两条路径确实不是同一后端」确定一条语义上稳定的哨兵{}",
                ctx(case, &d, &i)
            );
            // 通过时也打印两侧原文（保留 CI 留痕，便于棘轮失效时对照）
            println!(
                "[ratchet] AUDIT-11.4.56 仍分歧（符合台账，两侧静默 exit 0）：{}",
                ctx(case, &d, &i)
            );
        }
    }
}

/// 新增一条语料 = 一条 declaration（骨架不变）。
macro_rules! guard_case {
    ($fn_name:ident, $slug:expr, $name:expr, $audit:expr, $src:expr, $expect:expr) => {
        #[test]
        fn $fn_name() {
            check(&Case {
                slug: $slug,
                name: $name,
                audit: $audit,
                src: $src,
                expect: $expect,
            });
        }
    };
}

// ════════════════════════════════════════════════════════════════════
// 语料
// ════════════════════════════════════════════════════════════════════

// ── AUDIT-11.4.55：函数 + main 双写全局 ─────────────────────────────
// 语义：全局标量被 main 与函数同时写，两侧读写共享同一份全局；期望
// N1=1 N2=1 N3=2 N4=3 N5=4 N6=2（等价于 tenth-lens/probes/gap012_vm_global_write.th 的最小化）。
guard_case!(
    guard_global_write_from_main_and_fn,
    "global_write_main_fn",
    "函数与 main 双写全局（gap012 等价最小程序）",
    "AUDIT-11.4.55",
    r#"
let mut A: i32 = 0
let mut B: i32 = 0

fn bumpA() {
    A = A + 1;
}

fn bumpB() {
    B = B + 1;
}

fn main() {
    bumpB();
    println("N1=" + format("{}", B));
    A = A + 1;
    println("N2=" + format("{}", A));
    bumpA();
    println("N3=" + format("{}", A));
    let cur = A;
    A = cur + 1;
    println("N4=" + format("{}", A));
    bumpA();
    println("N5=" + format("{}", A));
    bumpB();
    println("N6=" + format("{}", B));
}
"#,
    Expect::ParityLines(&["N1=1", "N2=1", "N3=2", "N4=3", "N5=4", "N6=2"])
);

// ── AUDIT-11.4.54：越界索引必须响亮失败（不许静默返回 ()） ───────────
// 修复前：VM 侧 `unwrap_or(Value::Unit)` 静默返回 ()、exit 0，与解释器（报错 exit 1）分叉。

guard_case!(
    guard_vec_index_out_of_bounds_is_loud,
    "vec_index_oob",
    "Vec 越界索引 [7] 必须两路径都响亮报错",
    "AUDIT-11.4.54",
    r#"
fn main() {
    let v = Vec::new();
    let x = v[7];
    println(x);
}
"#,
    Expect::ParityLoud { stderr_contains: &["越界"] }
);

guard_case!(
    guard_string_index_out_of_bounds_is_loud,
    "string_index_oob",
    "字符串越界索引 [7] 必须两路径都响亮报错（同族）",
    "AUDIT-11.4.54/字符串同族",
    r#"
fn main() {
    let s = "abc";
    let c = s[7];
    println(c);
}
"#,
    Expect::ParityLoud { stderr_contains: &["越界"] }
);

guard_case!(
    guard_vec_get_out_of_bounds_is_loud,
    "vec_get_oob",
    "Vec.get(7) 越界必须两路径都响亮报错（同族）",
    "AUDIT-11.4.54/Vec.get 同族",
    r#"
fn main() {
    let v = Vec::new();
    let x = v.get(7);
    println(x);
}
"#,
    Expect::ParityLoud { stderr_contains: &["越界"] }
);

// ── 反向守护：合法索引不许被「响亮失败」修过头 ─────────────────────
guard_case!(
    guard_vec_index_in_bounds_still_works,
    "vec_index_in_bounds",
    "Vec 合法索引 [0] / get(1) 仍返回元素",
    "AUDIT-11.4.54 反向守护",
    r#"
fn main() {
    let mut v = Vec::new();
    v.push(10);
    v.push(20);
    println("V0=" + format("{}", v[0]));
    println("G1=" + format("{}", v.get(1)));
}
"#,
    Expect::ParityLines(&["V0=10", "G1=20"])
);

guard_case!(
    guard_string_index_in_bounds_still_works,
    "string_index_in_bounds",
    "字符串合法索引 [0] / [2] 仍返回字符",
    "AUDIT-11.4.54 反向守护",
    r#"
fn main() {
    let s = "abc";
    println("S0=" + format("{}", s[0]));
    println("S2=" + format("{}", s[2]));
}
"#,
    Expect::ParityLines(&["S0=a", "S2=c"])
);

// ── AUDIT-11.4.50：元组 `==`（2 元 / 3 元 / 嵌套 / 混合类型，相等与不等） ──
// 修复前：VM `vm_eq` 无 (Tuple,Tuple) 分支，落入 `_ => false` → `(1,2)==(1,2)` 恒 false。

guard_case!(
    guard_tuple2_eq_and_ne,
    "tuple2_eq_ne",
    "2 元组相等 / 不等",
    "AUDIT-11.4.50",
    r#"
fn main() {
    println("E=" + format("{}", (1, 2) == (1, 2)));
    println("N=" + format("{}", (1, 2) == (1, 3)));
}
"#,
    Expect::ParityLines(&["E=true", "N=false"])
);

guard_case!(
    guard_tuple3_eq_and_ne,
    "tuple3_eq_ne",
    "3 元组相等 / 不等",
    "AUDIT-11.4.50",
    r#"
fn main() {
    println("E=" + format("{}", (1, 2, 3) == (1, 2, 3)));
    println("N=" + format("{}", (1, 2, 3) == (1, 2, 4)));
}
"#,
    Expect::ParityLines(&["E=true", "N=false"])
);

guard_case!(
    guard_tuple_nested_eq_and_ne,
    "tuple_nested_eq_ne",
    "嵌套元组相等 / 不等",
    "AUDIT-11.4.50",
    r#"
fn main() {
    println("E=" + format("{}", ((1, 2), 3) == ((1, 2), 3)));
    println("N=" + format("{}", ((1, 2), 3) == ((1, 9), 3)));
}
"#,
    Expect::ParityLines(&["E=true", "N=false"])
);

guard_case!(
    guard_tuple_mixed_types_eq_and_ne,
    "tuple_mixed_eq_ne",
    "混合类型元组（str,int,bool）相等 / 不等",
    "AUDIT-11.4.50",
    r#"
fn main() {
    println("E=" + format("{}", ("s", 1, true) == ("s", 1, true)));
    println("N=" + format("{}", ("s", 1, true) == ("s", 2, true)));
}
"#,
    Expect::ParityLines(&["E=true", "N=false"])
);

guard_case!(
    guard_tuple_arity_scalar_and_neq_operator,
    "tuple_arity_scalar_neq",
    "元数不匹配 / 跨类型 / 与非元组比较，以及 != 运算符",
    "AUDIT-11.4.50",
    r#"
fn main() {
    println("ARITY=" + format("{}", (1, 2) == (1, 2, 3)));
    println("SCALAR=" + format("{}", (1, 2) == 5));
    println("CROSSTYPE=" + format("{}", (1, 2) == (1, "x")));
    println("NEQ_T=" + format("{}", (1, 2) != (1, 3)));
    println("NEQ_F=" + format("{}", (1, 2) != (1, 2)));
    println("UNIT=" + format("{}", () == ()));
}
"#,
    Expect::ParityLines(&[
        "ARITY=false",
        "SCALAR=false",
        "CROSSTYPE=false",
        "NEQ_T=true",
        "NEQ_F=false",
        "UNIT=true",
    ])
);

// ── AUDIT-11.4.53：整型 dtype 贯通（i64 注解/后缀/形参/返回/混合提升/radix） ──
// 修复前：算术一律按 i32 检查；i64 后缀 / `let x: i64` / i64 形参 / i64 返回**全部无效**。
// 下列语料把「手册承诺」变成两路径共同守护的事实（差分 + 金标准双保险）。

guard_case!(
    guard_i64_annotation_suffix_and_promotion,
    "i64_annotation_suffix_promotion",
    "i64 注解 / i64 后缀 / 无后缀超 i32 自动提升，三路径同值",
    "AUDIT-11.4.53",
    r#"
fn main() {
    let x: i64 = 2000000000;
    println("A=" + format("{}", x * 100));
    let a = 3000000000;
    println("B=" + format("{}", a + a));
    println("C=" + format("{}", 2000000000i64 * 100i64));
    println("D=" + format("{}", a));
}
"#,
    Expect::ParityLines(&[
        "A=200000000000",
        "B=6000000000",
        "C=200000000000",
        "D=3000000000",
    ])
);

guard_case!(
    guard_i64_param_return_and_commutativity,
    "i64_param_return_commutativity",
    "i64 形参 / i64 返回 + 混合运算交换律（a op b == b op a）",
    "AUDIT-11.4.53/R4",
    r#"
fn scale(x: i64) -> i64 { x * 100 }
fn main() {
    println("P=" + format("{}", scale(2000000000)));
    let y: i64 = 2000000000;
    println("L=" + format("{}", y + 1));
    println("R=" + format("{}", 1 + y));
    println("ML=" + format("{}", y * 3));
    println("MR=" + format("{}", 3 * y));
}
"#,
    Expect::ParityLines(&[
        "P=200000000000",
        "L=2000000001",
        "R=2000000001",
        "ML=6000000000",
        "MR=6000000000",
    ])
);

guard_case!(
    guard_radix_literal_promotes_like_decimal,
    "radix_literal_promotion",
    "radix 字面量与十进制对齐（默认 i32、超范围提升 i64）",
    "AUDIT-11.4.53/R5",
    r#"
fn main() {
    println("H=" + format("{}", 0x1_0000_0000));
    let r = 0x1_0000_0000;
    println("R=" + format("{}", r + r));
    println("S=" + format("{}", 0xFF));
}
"#,
    Expect::ParityLines(&["H=4294967296", "R=8589934592", "S=255"])
);

guard_case!(
    guard_i32_overflow_is_loud,
    "i32_overflow_loud",
    "显式 i32 溢出必须两路径都响亮报错（不许回绕/饱和）",
    "AUDIT-11.4.53 反向守护（护城河：溢出必须响亮）",
    r#"
fn main() {
    let p: i32 = 2000000000;
    println("X=" + format("{}", p * 2));
}
"#,
    Expect::ParityLoud { stderr_contains: &["溢出 i32 范围"] }
);

guard_case!(
    guard_narrow_dtype_overflow_is_loud,
    "narrow_dtype_overflow_loud",
    "窄 dtype（i8）越界必须两路径都响亮报错",
    "AUDIT-11.4.53 反向守护（窄 dtype 范围检查）",
    r#"
fn main() {
    let s: i8 = 100;
    let t: i8 = 100;
    println("X=" + format("{}", s + t));
}
"#,
    Expect::ParityLoud { stderr_contains: &["溢出 i8 范围"] }
);

// ── 已知分歧台账（ratchet）：AUDIT-11.4.56 闭包体内写全局 ────────────
// 实测（波次 1 之后）：默认路径 VM 写穿全局表 → C=2；解释器按值捕获 → C=0；
// **两侧 exit 均 0**（静默分歧）。按台账断言「仍然分歧」：修好即报红，提示移除台账。
guard_case!(
    guard_known_divergence_closure_writes_global,
    "closure_write_global_ratchet",
    "已知分歧台账：闭包体内写全局（VM 写穿 vs 解释器按值捕获）",
    "AUDIT-11.4.56（候选，已知分歧，棘轮）",
    r#"
let mut C: i32 = 0

fn main() {
    let inc = |d| { C = C + d; };
    inc(1);
    inc(1);
    println("C=" + format("{}", C));
}
"#,
    Expect::KnownDivergence
);
