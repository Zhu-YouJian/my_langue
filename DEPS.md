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

### tenthpm 包管理器依赖

定义于 `tools/tenthpm/Cargo.toml` 的 `[dependencies]` 段。

| crate | 版本 | 用途 |
|-------|------|------|
| `serde` | 1 | 序列化框架（derive） |
| `serde_json` | 1 | JSON 解析（包注册表通信） |
| `toml` | 0.8 | Tenth.toml 清单文件解析 |

### LSP 服务器依赖

定义于 `tools/lsp/Cargo.toml` 的 `[dependencies]` 段。

| crate | 版本 | 用途 |
|-------|------|------|
| `serde` | 1 | 序列化框架（derive） |
| `serde_json` | 1 | LSP 协议 JSON-RPC 消息解析 |
| `tenth` | path = "../../tenth" | 编译器前端（词法/语法/HIR） |

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

> 完整目录结构见 `CODE_WIKI.md` §1。

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

## CLI 子命令

> 完整 CLI 用法见 `CODE_WIKI.md` §3.2。

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
