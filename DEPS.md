# 依赖清单

> 本文档记录项目编译运行所需的关键依赖及其位置，方便环境重建时查阅。

---

## Rust 工具链

| 项目 | 版本 | 位置 |
|------|------|------|
| rustc | 1.95.0 (2026-04-14) | `C:\Users\史蒂夫\.cargo\bin\rustc.exe` |
| cargo | 1.95.0 | `C:\Users\史蒂夫\.cargo\bin\cargo.exe` |
| rustup (toolchain 管理器) | — | `C:\Users\史蒂夫\.rustup\` |

### 使用方式

Rust 未加入系统 PATH，编译时需用完整路径：

```bash
# 编译
C:\Users\史蒂夫\.cargo\bin\cargo.exe build --manifest-path tenth/Cargo.toml

# 测试
C:\Users\史蒂夫\.cargo\bin\cargo.exe test --manifest-path tenth/Cargo.toml

# 运行 REPL
C:\Users\史蒂夫\.cargo\bin\cargo.exe run --manifest-path tenth/Cargo.toml
```

如果需要加入当前会话的 PATH（非全局，关终端即失效）：

```cmd
set PATH=%PATH%;C:\Users\史蒂夫\.cargo\bin
```

---

## 网络代理（GitHub / crates.io 访问）

国内网络直连 GitHub 经常超时，需要通过代理。

**代理地址**：`http://127.0.0.1:7892`（Clash）

### Cargo / curl

`set` 环境变量方式（关终端即失效，不影响系统全局）：

```cmd
set HTTP_PROXY=http://127.0.0.1:7892
set HTTPS_PROXY=http://127.0.0.1:7892
```

设置后 `cargo build`（下载 crate）走代理。**注意：此方式对 git push 不生效。**

### Git

git 需直接配置代理（持久生效，仅当前仓库）：

```cmd
git config http.proxy http://127.0.0.1:7892
git config https.proxy http://127.0.0.1:7892
```

设置后 `git push` / `git clone` 走代理。

如需取消：

```cmd
git config --unset http.proxy
git config --unset https.proxy
```

---

## Cargo 依赖（crates.io）

定义于 `tenth/Cargo.toml` 的 `[dependencies]` 段，`cargo build` 自动下载，无需手动安装。

| crate | 版本 | 用途 |
|-------|------|------|
| `ndarray` | 0.16 | 张量（多维数组）计算引擎 |
| `rustyline` | 15 | REPL 交互式行编辑 |
| `thiserror` | 2 | 错误类型派生宏 |
| `rand` | 0.8 | 随机数生成 |
| `rand_distr` | 0.4 | 分布采样（正态分布等，用于 `randn`） |
| `wasm-encoder` | 0.215 | WASM 二进制编码（`.th` → `.wasm`） |
| `wasmi` | 0.39 | WASM 内嵌解释器（wasm run） |

### 传递依赖树

上述直接依赖自动拉取的间接依赖：

| crate | 版本 | 上游 |
|-------|------|------|
| libm | 0.2 | ndarray |
| matrixmultiply | 0.3 | ndarray |
| rawpointer | 0.2 | ndarray |
| num-traits | 0.2 | ndarray |
| num-integer | 0.1 | num-traits |
| num-complex | 0.4 | ndarray |
| autocfg | 1.5 | num-traits |
| cfg-if | 1.0 | getrandom |
| zerocopy | 0.8 | rustyline |
| unicode-width | 0.2 | rustyline |
| unicode-segmentation | 1.13 | rustyline |
| clipboard-win | 5.4 | rustyline |
| endian-type | 0.1 | rustyline |
| nibble_vec | 0.1 | rustyline |
| radix_trie | 0.2 | rustyline |
| bitflags | 2.11 | rustyline |
| fd-lock | 4.0 | rustyline |
| home | 0.5 | fd-lock |
| nix | — | rustyline (Unix only) |
| winapi | — | clipboard-win (transitive) |
| ppv-lite86 | 0.2 | rand_chacha |
| rand_core | 0.6 | rand |
| rand_chacha | 0.3 | rand |
| getrandom | 0.2 | rand_core |
| smallvec | 1.15 | ndarray |
| proc-macro2 | 1.0 | thiserror-impl / syn |
| quote | 1.0 | thiserror-impl |
| syn | 2.0 | thiserror-impl |
| libc | 0.2 | getrandom |

---

## 项目目录结构

```
.
├── DEPS.md          ← 本文件
├── README.md
├── SECURITY.md
├── AUDIT.md
├── MEMO.md
├── docs/            ← 设计文档和实施计划
│   ├── 语言参考手册.md   ← 权威语言参考
│   └── superpowers/
│       └── plans/
│           ├── 2026-05-22-phase1-bootstrap-compiler.md
│           ├── 2026-05-22-phase2-interpreter-hardening.md
│           ├── 2026-05-22-phase3a-type-system.md
│           ├── 2026-05-22-phase3b-mlir-compilation.md
│           ├── 2026-05-26-phase4-gpu-performance.md
│           ├── 2026-05-26-phase5-ai-fullstack.md
│           ├── 2026-05-26-phase6-ecosystem-tools.md
│           ├── 2026-05-26-v0.3.0-standard-library-and-self-hosting.md
│           └── 2026-05-26-v0.3.1-language-gaps.md
├── tenthc/          ← Tenth 自举编译器 (.th 源码，通过 Rust 解释器运行)
│   ├── main.th
│   ├── lexer/
│   └── parser/
└── tenth/
    ├── Cargo.toml   ← 依赖声明 + feature flags
    ├── Cargo.lock   ← 锁定版本
    ├── src/         ← 编译器源码
    │   ├── main.rs
    │   ├── lib.rs
    │   ├── repl.rs
    │   ├── error.rs
    │   ├── lexer/
    │   ├── parser/
    │   ├── hir/
│   │   ├── compile/    ← WASM 编译后端
│   │   │   ├── mod.rs
│   │   │   ├── wasm.rs
│   │   │   └── bridge.rs
│   │   ├── runtime/     ← 解释器 + 值系统 + 内存管理
│   │   │   ├── mod.rs
│   │   │   ├── value.rs
│   │   │   ├── tensor.rs
│   │   │   ├── interpreter.rs
│   │   │   ├── arena.rs
│   │   │   ├── autodiff.rs
│   │   │   └── limits.rs    ← 资源限制 + 原子计数器
│   │   └── repl.rs       ← 交互环境
│   ├── std/             ← Tenth 标准库 (.th 源码)
│   │   ├── nn/
│   │   └── optim/
    └── tests/          ← 82 项测试（11 文件）
        ├── integration_test.rs
        ├── memory_test.rs
        ├── lexer_test.rs
        ├── parser_test.rs
        ├── enum_test.rs
        ├── generic_test.rs
        ├── module_test.rs
        ├── ownership_test.rs
        ├── stdlib_test.rs
        ├── struct_test.rs
        ├── trait_test.rs
        └── fixtures/
```

---

## Cargo Feature Flags

| flag | 含义 | 效果 |
|------|------|------|
| `mem-debug` | 启用内存追踪 | 全局原子计数器记录 arena 字节 / tensor 数 / 变量数 |
| `mem-strict` | 严格内存模式 | 超限直接 panic（隐含 `mem-debug`） |

```bash
# 严格模式编译
cargo build --features mem-strict --manifest-path tenth/Cargo.toml

# 带追踪运行 REPL
cargo run --features mem-debug --manifest-path tenth/Cargo.toml
```

---

## CLI 参数（main.rs）

| 参数 | 说明 |
|------|------|
| `--max-memory <MB>` | 限制最大内存（arena + tensor 元素数上限） |

```bash
# REPL 限制 256MB
cargo run -- --max-memory 256
```

---

## REPL 内置命令

| 命令 | 说明 |
|------|------|
| `:q` | 退出 |
| `:h` | 帮助 |
| `:vars` | 列出所有变量 |
| `:clear` | 🆕 清空全部状态（函数定义 + 变量） |
| `:mem` | 🆕 内存快照（arena / tensor / 变量数 / 上限） |
| `:print <var>` | 打印变量值 |

---

## 新环境搭建步骤

1. **安装 Rust**（如系统未装）
   ```cmd
   # 下载 rustup-init
   curl -o rustup-init.exe https://win.rustup.rs/x86_64
   # 安装
   rustup-init.exe -y --default-toolchain stable
   ```

2. **编译项目**
   ```bash
   cargo build --manifest-path tenth/Cargo.toml
   ```
   Cargo 会根据 `Cargo.toml` 和 `Cargo.lock` 自动下载所有 crate 依赖。

3. **运行测试**
   ```bash
   cargo test --manifest-path tenth/Cargo.toml
   ```
