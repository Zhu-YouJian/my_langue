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

## Cargo 依赖（crates.io）

定义于 `tenth/Cargo.toml` 的 `[dependencies]` 段，`cargo build` 自动下载，无需手动安装。

| crate | 版本 | 用途 |
|-------|------|------|
| `ndarray` | 0.16 | 张量（多维数组）计算引擎 |
| `rustyline` | 15 | REPL 交互式行编辑 |
| `thiserror` | 2 | 错误类型派生宏 |
| `rand` | 0.8 | 随机数生成 |
| `rand_distr` | 0.4 | 分布采样（正态分布等，用于 `randn`） |

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
├── docs/            ← 设计文档和实施计划
│   ├── tenth-language-reference.md
│   └── superpowers/
│       ├── plans/
│       │   ├── 2026-05-22-phase1-bootstrap-compiler.md
│       │   ├── 2026-05-22-phase2-interpreter-hardening.md
│       │   ├── 2026-05-22-phase3a-type-system.md
│       │   └── 2026-05-22-phase3b-mlir-compilation.md
│       └── specs/
│           └── 2026-05-22-tenth-language-design.md
└── tenth/
    ├── Cargo.toml   ← 依赖声明
    ├── Cargo.lock   ← 锁定版本
    ├── src/         ← 编译器源码
    │   ├── main.rs
    │   ├── lib.rs
    │   ├── repl.rs
    │   ├── error.rs
    │   ├── lexer/
    │   ├── parser/
    │   ├── hir/
    │   └── runtime/
    └── tests/       ← 测试文件
```

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
