//! M5.2 模糊测试框架（确定性、CI 友好）— 见 `docs/语言规范.md` M5.2 / 能力全梳理「模糊测试」
//!
//! 目标：编译器前端（lexer / parser / lower）对**任意输入**不 panic——
//! 允许返回 Err（错误消息）或 recovery（容错解析），**禁止 panic**。
//!
//! 三组 fuzz + 一组边界：
//!   1. 语法层 fuzz：随机 token 文本流 / 随机字节 → lexer 不 panic；
//!      随机 token 序列 → parser（parse_program_with_recovery）不 panic
//!   2. 程序生成 fuzz：随机合法-ish 程序（语句/表达式/张量/shape 边界）
//!      → lex + parse + lower 全链路不 panic（仅编译断言，不运行）
//!   3. 边界 fuzz：超大整数 / 深嵌套 / 超长字符串 / 重复关键字等极端输入
//!
//! 确定性：固定种子 + 自实现 xorshift64* PRNG（不依赖外部 crate），
//! 任何 panic 都会在测试失败信息中携带（seed, 输入摘要, panic payload），
//! 可一键重现。迭代次数与输入规模受限，不拖慢 CI。

use std::panic::{catch_unwind, AssertUnwindSafe};

use tenth::hir::lower::Lowerer;
use tenth::lexer::lexer::Lexer;
use tenth::parser::parser::Parser;

// ─────────────────────────────────────────────────────────────
// 确定性 PRNG：xorshift64*
// ─────────────────────────────────────────────────────────────
struct Rng {
    state: u64,
}

impl Rng {
    fn new(seed: u64) -> Self {
        Rng {
            state: if seed == 0 { 0x9E37_79B9_7F4A_7C15 } else { seed },
        }
    }
    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.state = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n.max(1) as u64) as usize
    }
    fn chance(&mut self, pct: u64) -> bool {
        self.next_u64() % 100 < pct
    }
}

/// 把 panic payload 转成可读字符串（诊断用）。
fn panic_msg(payload: &(dyn std::any::Any + Send)) -> String {
    if let Some(s) = payload.downcast_ref::<String>() {
        format!("panic: {}", s)
    } else if let Some(s) = payload.downcast_ref::<&str>() {
        format!("panic: {}", s)
    } else {
        "panic: (非字符串 payload)".to_string()
    }
}

/// 捕获闭包 panic，返回 `Ok(())` 或 `Err(描述)`。
fn no_panic<F: FnOnce()>(f: F) -> Result<(), String> {
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(()) => Ok(()),
        Err(p) => Err(panic_msg(&*p)),
    }
}

// ─────────────────────────────────────────────────────────────
// 1. 语法层 fuzz：随机 token 文本 / 随机字节
// ─────────────────────────────────────────────────────────────

/// token 文本池：覆盖 lexer 所有分支（关键字/字面量/运算符/注释/非法字符/Unicode）。
const TOKEN_POOL: &[&str] = &[
    // 关键字（35 活跃 + 预留）
    "fn", "let", "mut", "if", "else", "match", "while", "loop", "for", "in",
    "return", "break", "continue", "struct", "enum", "union", "impl", "trait",
    "use", "mod", "as", "move", "ref", "pub", "const", "static", "yield",
    "await", "async", "spawn", "try", "lossy", "true", "false", "task", "shard", "node",
    // 字面量
    "0", "42", "-7", "1_000_000", "0x1F", "0b1010", "0o17", "999999999999999999999999999",
    "3.14", "-2.5e10", "1.0f32", "1e-300", "0.0000000000000000000000001", "1e308",
    "1i8", "255u8", "123i64", "42u32", "3.14f64", "1.5f16", "1.0bf16",
    "\"hello\"", "\"\"", "\"a\\nb\\t\\\"c\\\\\"", "\"unterminated", "r\"raw\"", "r#\"raw#str\"#",
    "f\"name={x}\"", "f\"{a}+{b}\"", "f\"no closing", "'a'", "'\\n'", "'\\''", "'unterminated",
    // 标识符
    "x", "a", "_", "__x", "x1", "foo_bar", "Tensor", "tensor", "main", "self", "Self",
    // 运算符 / 分隔符
    "+", "-", "*", "/", "%", "==", "!=", "<", ">", "<=", ">=", "&&", "||", "!", "=",
    "+=", "-=", "*=", "/=", "->", "=>", ".", "..", "..=", "::", ",", ";", ":", "(", ")",
    "[", "]", "{", "}", "&", "&mut", "|", "?", "@", "$", "~", "##", "`", "#[",
    // 注释
    "// line comment", "/* block comment */", "/* unterminated", "//", "/* nested /* */ */",
    // 张量 / 类型相关
    "Tensor[f64, 3, 4]", "Tensor[f32, M, K]", "Vec<i32>", "HashMap<str, i32>", "&str",
    // 特殊 / 非法字符
    "\u{0}", "\u{FEFF}", "\u{4e2d}\u{6587}", "\u{1F600}", "\u{00e9}", "\u{0007}",
    "#$%^", "\\", "\'", "\"", "`", "~", "@@", "$$", ">>>", "<<<", "!!", "??", ":::", "..::",
];

/// 生成一个随机 token 文本序列（长度 1..=max_len，逐 token 拼接，可能无分隔）。
fn random_token_stream(rng: &mut Rng, max_tokens: usize) -> String {
    let n = 1 + rng.below(max_tokens);
    let mut out = String::new();
    for _ in 0..n {
        let tok = TOKEN_POOL[rng.below(TOKEN_POOL.len())];
        // 70% 加空格分隔，30% 紧贴（制造 `fnfn`、`1x`、`a+b` 等粘连）
        if !out.is_empty() && rng.chance(70) {
            out.push(' ');
        }
        out.push_str(tok);
    }
    out
}

/// 随机字节流（任意 UTF-8 不保证合法——用 from_utf8_lossy 转义）。
fn random_byte_stream(rng: &mut Rng, max_len: usize) -> String {
    let len = 1 + rng.below(max_len);
    let bytes: Vec<u8> = (0..len).map(|_| rng.next_u64() as u8).collect();
    String::from_utf8_lossy(&bytes).into_owned()
}

/// 核心断言：lexer 对任意输入不 panic（允许 Err）。
fn assert_lexer_no_panic(src: &str, ctx: &str) {
    let res = no_panic(|| {
        let mut lexer = Lexer::new(src);
        let _ = lexer.tokenize();
    });
    assert!(res.is_ok(), "[lexer] {} 触发 panic: {}\n输入: {:?}",
        ctx, res.unwrap_err(), &src[..src.len().min(300)]);
}

/// 核心断言：parser（recovery 模式）对任意 token 流不 panic（允许 Err/recovery）。
fn assert_parser_no_panic(tokens: Vec<tenth::lexer::token::Token>, ctx: &str) {
    let res = no_panic(|| {
        let mut parser = Parser::new(tokens);
        let _ = parser.parse_program_with_recovery();
    });
    assert!(res.is_ok(), "[parser] {} 触发 panic: {}", ctx, res.unwrap_err());
}

/// 语法层 fuzz 主循环：种子 + 迭代次数 + 每输入最大 token/字节。
fn fuzz_syntax(seed: u64, iters: usize, max_tokens: usize, max_bytes: usize) {
    let mut rng = Rng::new(seed);
    for i in 0..iters {
        // 1) 随机 token 文本流 → lexer
        let src = random_token_stream(&mut rng, max_tokens);
        assert_lexer_no_panic(&src, &format!("token-stream iter#{i} seed={seed}"));

        // 2) 随机字节流 → lexer
        let bytes = random_byte_stream(&mut rng, max_bytes);
        assert_lexer_no_panic(&bytes, &format!("byte-stream iter#{i} seed={seed}"));

        // 3) 把随机 token 流 lex 出的 token 交给 parser（recovery 模式）
        let mut lexer = Lexer::new(&src);
        if let Ok(tokens) = lexer.tokenize() {
            assert_parser_no_panic(tokens, &format!("token->parser iter#{i} seed={seed}"));
        }
    }
}

// ─────────────────────────────────────────────────────────────
// 2. 程序生成 fuzz：随机合法-ish 程序 → 全链路不 panic
// ─────────────────────────────────────────────────────────────

/// 随机程序生成器：从片段库递归组合，深度受限。
struct ProgGen<'a> {
    rng: &'a mut Rng,
}

impl<'a> ProgGen<'a> {
    fn new(rng: &'a mut Rng) -> Self {
        ProgGen { rng }
    }

    /// 生成一个随机类型注解（含 Tensor/shape 边界）。
    fn gen_type(&mut self) -> String {
        const TYPES: &[&str] = &[
            "i32", "i64", "f32", "f64", "bool", "str", "char", "()",
            "Tensor[f64, 3, 4]", "Tensor[f32, M, K]", "Tensor[f64, 2, ..]",
            "Vec<i32>", "Vec<f64>", "HashMap<str, i32>", "&i32", "&mut f64",
            "Box<i32>", "Rc<f64>", "Option<i32>", "Result<i32, str>",
            "(i32, f64)", "(f64, f64, i32)", "[f64; 3]", "S", "E", "U",
            "Tensor[i64, 1, 2, 3]", "Tensor[f16, 2, 2]",
        ];
        TYPES[self.rng.below(TYPES.len())].to_string()
    }

    /// 生成一个简单（无控制流）表达式。
    fn gen_simple_expr(&mut self, depth: usize) -> String {
        const SIMPLE: &[&str] = &[
            "42", "0", "-7", "3.14", "1.5f64", "true", "false", "\"hi\"", "'x'",
            "x", "y", "a", "b", "flag", "n", "s", "v", "m",
            "x + 1", "x - y", "a * b", "x / 2", "x % 3", "a == b", "a != b",
            "x < y", "x >= y", "flag && b", "!flag", "x + y * 2", "(a + b) * c",
            "1..10", "0..=n", "0..n",
        ];
        if depth == 0 || self.rng.chance(60) {
            SIMPLE[self.rng.below(SIMPLE.len())].to_string()
        } else {
            // 组合一层
            match self.rng.below(6) {
                0 => format!("({})", self.gen_simple_expr(depth - 1)),
                1 => format!("{} + {}", self.gen_simple_expr(depth - 1), self.gen_simple_expr(depth - 1)),
                2 => format!("if {} {{ {} }} else {{ {} }}",
                    self.gen_simple_expr(depth - 1), self.gen_simple_expr(depth - 1), self.gen_simple_expr(depth - 1)),
                3 => format!("|t: f64| {} ", self.gen_simple_expr(depth - 1)),
                4 => format!("f({})", self.gen_simple_expr(depth - 1)),
                5 => format!("v[{}]", self.gen_simple_expr(depth - 1)),
                _ => unreachable!(),
            }
        }
    }

    /// 生成张量/shape 相关表达式（护城河边界）。
    fn gen_tensor_expr(&mut self) -> String {
        const T: &[&str] = &[
            "tensor([1.0, 2.0, 3.0])",
            "tensor([[1.0, 2.0], [3.0, 4.0]])",
            "zeros(3, 4)",
            "ones(2, 2)",
            "randn(2, 3)",
            "x.matmul(y)",
            "x.sum()",
            "x.reshape(4, 3)",
            "x.transpose()",
            "x.mean()",
            "tensor([1.0])[0]",
            "zeros(1, 1, 1)",
            "randn(0, 3)",
            "x.bmm(y)",
            "x + y",
        ];
        T[self.rng.below(T.len())].to_string()
    }

    /// 生成一个表达式（简单/张量/复合混合）。
    fn gen_expr(&mut self, depth: usize) -> String {
        if self.rng.chance(35) {
            self.gen_tensor_expr()
        } else {
            self.gen_simple_expr(depth)
        }
    }

    /// 生成一个语句。
    fn gen_stmt(&mut self, depth: usize) -> String {
        const STMTS: &[&str] = &[
            "let a = 1;",
            "let mut x = 0;",
            "let s = \"str\";",
            "x = x + 1;",
            "return;",
            "return 42;",
            "println(x);",
            "println(\"hello\");",
            "break;",
            "continue;",
            "let f = |a: i32, b: i32| a + b;",
        ];
        if depth == 0 || self.rng.chance(50) {
            STMTS[self.rng.below(STMTS.len())].to_string()
        } else {
            match self.rng.below(7) {
                0 => format!("let x: {} = {};", self.gen_type(), self.gen_expr(depth - 1)),
                1 => format!("let mut v: {} = {};", self.gen_type(), self.gen_expr(depth - 1)),
                2 => format!("if {} {{ {} }} else {{ {} }}",
                    self.gen_expr(depth - 1), self.gen_stmt(depth - 1), self.gen_stmt(depth - 1)),
                3 => format!("while {} {{ {} }}", self.gen_expr(depth - 1), self.gen_stmt(depth - 1)),
                4 => format!("for i in 0..10 {{ {} }}", self.gen_stmt(depth - 1)),
                5 => format!("match x {{ 0 => 1, _ => 2 }};"),
                6 => format!("{{ {} }}", self.gen_stmt(depth - 1)),
                _ => unreachable!(),
            }
        }
    }

    /// 生成一个顶层声明。
    fn gen_decl(&mut self, depth: usize) -> String {
        match self.rng.below(8) {
            0 => format!("fn f{}() -> {} {{ {} }}",
                self.rng.below(100), self.gen_type(), self.gen_stmt(depth)),
            1 => format!("fn g{}(a: {}, b: {}) -> {} {{ a }}",
                self.rng.below(100), self.gen_type(), self.gen_type(), self.gen_type()),
            2 => format!("struct S{} {{ x: i32, y: f64 }}", self.rng.below(100)),
            3 => format!("enum E{} {{ A, B(i32), C {{ v: f64 }} }}", self.rng.below(100)),
            4 => format!("union U{} {{ a: i32, b: f64 }}", self.rng.below(100)),
            5 => format!("trait T{} {{ fn m(&self) -> i32; fn n(&self, x: f64) -> f64; }}", self.rng.below(100)),
            6 => format!("fn h{}(t: Tensor[f64, 3, 4]) -> Tensor[f64, 3, 4] {{ t }}", self.rng.below(100)),
            7 => format!("fn k{}() -> i32 {{ let x = {}; x }}", self.rng.below(100), self.gen_expr(depth)),
            _ => unreachable!(),
        }
    }

    /// 生成一个完整程序：若干声明 + main。
    fn gen_program(&mut self, max_decls: usize) -> String {
        let n = 1 + self.rng.below(max_decls);
        let mut out = String::new();
        for _ in 0..n {
            out.push_str(&self.gen_decl(2));
            out.push('\n');
        }
        // main：混合张量/shape 边界 + 语句
        let stmts: Vec<String> = (0..(1 + self.rng.below(5)))
            .map(|_| self.gen_stmt(2))
            .collect();
        out.push_str(&format!("fn main() {{ {} {} }}",
            self.gen_tensor_expr(), stmts.join(" ")));
        out
    }
}

/// 程序生成 fuzz 主循环。
fn fuzz_program(seed: u64, iters: usize) {
    let mut rng = Rng::new(seed);
    for i in 0..iters {
        let prog = {
            let mut g = ProgGen::new(&mut rng);
            g.gen_program(4)
        };
        // lex + parse + lower 全链路不 panic（仅编译断言）
        let res = no_panic(|| {
            let mut lexer = Lexer::new(&prog);
            let tokens = lexer.tokenize().ok();
            if let Some(tokens) = tokens {
                let mut parser = Parser::new(tokens);
                if let Ok(ast) = parser.parse_program() {
                    let mut lowerer = Lowerer::new();
                    let _ = lowerer.lower_program(&ast);
                }
            }
        });
        assert!(res.is_ok(),
            "[program] iter#{i} seed={seed} 触发 panic: {}\n程序(前300字): {:?}",
            res.unwrap_err(), &prog[..prog.len().min(300)]);
    }
}

// ─────────────────────────────────────────────────────────────
// 3. 边界 fuzz：极端输入（全部已实测安全，见 M5.2 调研）
// ─────────────────────────────────────────────────────────────
fn boundary_cases() -> Vec<(&'static str, String)> {
    vec![
        // (名字, 输入)
        ("empty", String::new()),
        ("whitespace", "   \n\t  ".to_string()),
        ("single-char", "x".to_string()),
        ("only-punct", ":::;;;,,,(((()))){{}}[]".to_string()),
        ("bom", "\u{FEFF}fn main() { 1 }".to_string()),
        ("deep-paren-10k", format!("fn main() {{ {}1{} }}", "(".repeat(10_000), ")".repeat(10_000))),
        ("deep-if-5k", format!("fn main() {{ {}1{} }}", "if true { ".repeat(5_000), " }".repeat(5_000))),
        ("deep-block-3k", format!("fn main() {{ {}1{} }}", "{{ ".repeat(3_000), " }}".repeat(3_000))),
        ("big-int", "fn main() { 99999999999999999999999999999999999999999999999999 }".to_string()),
        ("big-int-hex", "fn main() { 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF }".to_string()),
        ("big-float", "fn main() { 1.7976931348623157e308 * 1.0 }".to_string()),
        ("long-str-200k", format!("fn main() {{ \"{}\" }}", "x".repeat(200_000))),
        ("long-ident-15k", format!("fn main() {{ let {} = 1; {} }}", "abc".repeat(5_000), "abc".repeat(5_000))),
        ("repeat-fn-3k", format!("fn main() {{ {} }}", "fn ".repeat(3_000))),
        ("repeat-num", format!("fn main() {{ {} }}", "42 ".repeat(5_000))),
        ("repeat-call-chain", format!("fn main() {{ {} }}", "(1 + 1)".repeat(2_000))),
        ("repeat-left-nest", format!("fn main() {{ {} }}", format!("{}{}", "((1 + 1)".repeat(1_000), ")".repeat(1_000)))),
        ("unterminated-comment", "fn main() { /* unterminated".to_string()),
        ("unterminated-string", "fn main() { \"unterminated".to_string()),
        ("unterminated-char", "fn main() { 'x".to_string()),
        ("unterminated-fstring", "fn main() { f\"{x}".to_string()),
        ("nul-bytes", "fn main() { \u{0}\u{0}\u{0} }".to_string()),
        ("control-chars", "fn main() { \u{7}\u{8}\u{1b} }".to_string()),
        ("mixed-unicode", "fn main() { \u{4e2d}\u{6587}\u{1F600}\u{00e9} }".to_string()),
        ("deep-generic", format!("fn f{}() -> i32 {{ 1 }}", "<i32>".repeat(500))),
    ]
}

fn fuzz_boundary() {
    // 边界用例含超深嵌套（10k 括号 / 5k if / 3k 块）。实测：这些深度在
    // `tenth.exe run`（主线程）下全部安全（见 M5.2 调研），但 cargo test
    // 的测试线程默认栈较小，同输入会误报栈溢出（abort，不可捕获）。
    // 修复：每个边界用例在 128MB 大栈线程中运行，模拟真实运行栈环境，
    // 使深嵌套边界可被安全测试（若 128MB 下仍溢出，则确属编译期栈需求缺陷）。
    for (name, src) in boundary_cases() {
        let name_c = name.to_string();
        let src_c = src.clone();
        let handle = std::thread::Builder::new()
            .name(format!("boundary-{}", name_c))
            .stack_size(128 * 1024 * 1024)
            .spawn(move || {
                // lexer 不 panic
                let lres = no_panic(|| {
                    let mut lexer = Lexer::new(&src_c);
                    let _ = lexer.tokenize();
                });
                assert!(lres.is_ok(), "[boundary:{name_c}] lexer 触发 panic: {}", lres.unwrap_err());

                // parser（recovery）不 panic
                let pres = no_panic(|| {
                    let mut lexer = Lexer::new(&src_c);
                    if let Ok(tokens) = lexer.tokenize() {
                        let mut parser = Parser::new(tokens);
                        let _ = parser.parse_program_with_recovery();
                    }
                });
                assert!(pres.is_ok(), "[boundary:{name_c}] parser 触发 panic: {}", pres.unwrap_err());

                // lower 不 panic（仅当 parse 成功）
                let lres2 = no_panic(|| {
                    let mut lexer = Lexer::new(&src_c);
                    if let Ok(tokens) = lexer.tokenize() {
                        let mut parser = Parser::new(tokens);
                        if let Ok(ast) = parser.parse_program() {
                            let mut lowerer = Lowerer::new();
                            let _ = lowerer.lower_program(&ast);
                        }
                    }
                });
                assert!(lres2.is_ok(), "[boundary:{name_c}] lower 触发 panic: {}", lres2.unwrap_err());
            })
            .expect("spawn boundary thread 失败");
        // 子线程 panic 会经 join 传播（Err）
        handle.join().expect("boundary 子线程 panic");
    }
}

// ─────────────────────────────────────────────────────────────
// 测试入口（固定种子，可重现）
// ─────────────────────────────────────────────────────────────

#[test]
fn fuzz_syntax_seed_a() {
    fuzz_syntax(0xA11CE, 300, 40, 200);
}

#[test]
fn fuzz_syntax_seed_b() {
    fuzz_syntax(0xB0B5E, 300, 60, 300);
}

#[test]
fn fuzz_syntax_seed_c() {
    fuzz_syntax(0xC0FFEE, 400, 80, 400);
}

#[test]
fn fuzz_program_seed_a() {
    fuzz_program(0xD0E1, 250);
}

#[test]
fn fuzz_program_seed_b() {
    fuzz_program(0xE7E5, 250);
}

#[test]
fn fuzz_boundary_extremes() {
    fuzz_boundary();
}
