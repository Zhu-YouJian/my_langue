# Tenth — 为 AI 研究而生的语言

> 代号 Tenth = Tensor + Zenith，意为「张量之巅」

## 现状

**v0.3.0-pre** — Rust 解释器 + 字节码 VM + 自举编译器。82 项测试全过。

| 组件 | 状态 |
|------|------|
| Lexer / Parser / AST | ✅ |
| HIR + 类型推断 + 借用检查 | ✅ |
| 树遍历解释器 | ✅ |
| **字节码 VM（栈式，33 指令）** | ✅ |
| 泛型函数 / 结构体 | ✅ |
| Trait 定义与实现 | ✅ |
| 引用 / 移动语义 | ✅ |
| struct / enum / match | ✅ |
| ~~MIR → C 编译~~ | ❌ 已移除 |
| REPL 交互环境 | ✅ |
| 内存护栏 (arena + limits) | ✅ |
| WASM 编译 (wasm-encoder + wasmi) | ✅ |
| 自动微分 (标量级 tape) | ✅ |
| Vec / HashMap / String 标准库 | ✅ |
| **自举编译器 (Tenth 编写)** | ✅ |

## 快速开始

```bash
# 克隆
git clone <repo-url> tenth-lang
cd tenth-lang

# 编译（release 模式，推荐）
cargo build --release --manifest-path tenth/Cargo.toml

# 运行 REPL
cargo run --release --manifest-path tenth/Cargo.toml

# 运行 .th 文件（默认走 VM，VM 不支持时自动回退解释器）
cargo run --release --manifest-path tenth/Cargo.toml run path/to/file.th

# 编译到 WASM 并用 wasmi 执行
cargo run --release --manifest-path tenth/Cargo.toml wasm path/to/file.th

# 编译到 .wasm 文件
cargo run --release --manifest-path tenth/Cargo.toml build path/to/file.th

# 运行测试
cargo test --manifest-path tenth/Cargo.toml
```

## 语言示例

```tenth
// Hello World
fn main() {
    println("Hello, Tenth!");
}

// 变量与控制流
fn factorial(n: i64) -> i64 {
    let mut result = 1;
    let mut i = 1;
    while i <= n {
        result = result * i;
        i = i + 1;
    };
    result
}

// 张量运算
fn main() {
    let a = tensor[[1.0, 2.0], [3.0, 4.0]];
    let b = a + 10.0;
    println(b.sum());  // => 50
}

// 结构体 + trait + 方法
struct Point { x: f64, y: f64 }

impl Point {
    fn dist(self) -> f64 {
        (self.x * self.x + self.y * self.y).sqrt()
    }
}

fn main() {
    let p = Point { x: 3.0, y: 4.0 };
    println(p.dist());  // => 5
}

// 泛型函数
fn identity<T>(x: T) -> T { x }
```

更多示例见 `Tenth实例/` 目录（21 个）和 `tenth/std/` 标准库。

## 自举

Tenth 编译器由 Tenth 自身编写（`tenthc/`），经自举管线验证：

| 路径 | 词法分析 | 语法分析 | 编译 | 状态 |
|------|---------|---------|------|------|
| A | Rust | Rust | compile_host | ✅ 秒级 |
| B | **Tenth** | **Tenth** | compile_program | ✅ 已验证 |
| C | WASM | wasmi | compile_host | ✅ 闭环 |

## 路线图

| Phase | 内容 | 状态 |
|-------|------|------|
| Phase 1 | Bootstrap 编译器 | ✅ |
| Phase 2 | 解释器夯实 | ✅ |
| Phase 3A | 类型系统深化 | ✅ |
| ~~Phase 3B~~ | ~~编译后端 (C)~~ | ❌ 已移除 |
| Phase 4 | GPU 与性能 | 🚧 |
| Phase 5 | AI 全栈 | 🚧 |
| Phase 6 | 生态与工具 | 🚧 |
| Phase 7 | 核心标准库 | ✅ |
| Phase 8 | 自举编译器 | ✅ |

详见 `docs/语言参考手册.md` 和 MEMO.md。

## 依赖

编译只需 Rust（≥1.95），所有 crate 依赖自动下载。详见 `DEPS.md`。