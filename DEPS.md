# 依赖清单

> 本文档记录项目编译运行所需的关键依赖及其位置，方便环境重建时查阅。

---

## Rust 工具链

| 项目 | 版本 | 位置 |
|------|------|------|
| rustc | 1.95.0 (2026-04-14) | `%USERPROFILE%\.cargo\bin\rustc.exe` |
| cargo | 1.95.0 | `%USERPROFILE%\.cargo\bin\cargo.exe` |
| rustup (toolchain 管理器) | — | `%USERPROFILE%\.rustup\` |

### 使用方式

Rust 未加入系统 PATH，编译时需用完整路径：

```bash
# 编译
%USERPROFILE%\.cargo\bin\cargo.exe build --manifest-path tenth/Cargo.toml

# 测试
%USERPROFILE%\.cargo\bin\cargo.exe test --manifest-path tenth/Cargo.toml

# 运行 REPL
%USERPROFILE%\.cargo\bin\cargo.exe run --manifest-path tenth/Cargo.toml
```

如果需要加入当前会话的 PATH（非全局，关终端即失效）：

```cmd
set PATH=%PATH%;%USERPROFILE%\.cargo\bin
```

### 写边界与临时目录（受限沙箱下的机器保证）

> 本节取代 2026-09-29 之前的旧标题「已知环境陷阱：受限沙箱下测试写 `%TEMP%` 会「假红」」。**旧的手设 `$env:TMP`/`$env:TEMP` 绕行方案已作废**——现在由下述机器保证接管；仍引用旧标题的代码注释/守卫文案请改指本节。

**先记住结论**：本机（以及任何把可执行文件放在**工作区内**启动的环境）存在一条**写边界**——**只能写工作区内**。测试与示例的写落点必须落在仓库内；`%TEMP%`、家目录、桌面一律被拒。

改动前，12 个直接使用 `std::env::temp_dir()` 的测试目标因此**集体假红**（清单与证据见 `AUDIT-11.4.100`）。**这是环境问题、不是代码回归**：同一提交、同一二进制，只把 `TEMP`/`TMP` 指向工作区内即由红转绿。**不要把这类红当回归处理，也不要用它改写基线数字**（基线口径见 `能力梳理/能力全梳理.md` §统计基线）。

#### 写边界表（2026-09-29 实测）

| 写入目标 | 可写 | 典型症状 |
|---|---|---|
| 工作区内（`tenth/target/**`、仓库内任意相对路径） | ✅ | — |
| `%TEMP%`（系统临时目录） | ❌ | `Os { code: 5, kind: PermissionDenied }`「拒绝访问。」 |
| `~`（家目录） | ❌ | 同上 |
| `~\.cargo` / `~\.rustup` | ❌ | 同上 |
| 桌面 | ❌ | 同上 |

**为什么**：实测同一份 `cmd.exe`，在**工作区外**启动时令牌完整性为 **Medium**（`S-1-16-8192`），**复制进工作区后**启动变成 **Low**（`S-1-16-4096`）。`cargo.exe` / `rustc.exe` 本身在工作区外 ⇒ 不受限；**它们编译出来的东西落在工作区内 ⇒ 受限**（`tenth/target/**` 下的测试二进制、`tenth.exe`）。
⚠️ **口径**：上表是**已实测的行为与边界**；「按工作区路径生效」这一**机制**是**推断**，本文件不把它当已证因果使用。

#### 机器保证三件套（2026-09-29 起，不必手工设 `$env:TMP`/`$env:TEMP`）

1. **仓库根 `.cargo/config.toml` 的 `[env]`**（把临时目录钉进工作区）：

   ```toml
   TMP  = { value = "tenth/target/tmp", relative = true, force = true }
   TEMP = { value = "tenth/target/tmp", relative = true, force = true }
   ```

   - `relative = true` ⇒ 相对**仓库根**解析 ⇒ 从仓库根（`--manifest-path`）与 `cd tenth` 两种 cwd 得到**同一路径**（实测两 cwd 逐字相同）。
   - **`force = true` 必须带**：cargo `[env]` 默认**不覆盖已存在的变量**，日常 shell 里 `TEMP` 已被系统设好 ⇒ 非 force 版是**静默 no-op**。实测（独立对抗性复核）：保留两条、只删 `, force = true` ⇒ `module_search_path_test` **0/7 红**、`code: 5`。
2. **`tenth/build.rs` 构建期建目录**：cargo `[env]` 只设变量、**不建目录**，而部分测试（如 `mnist_loader_test`）是**直接**往 `temp_dir()` 写文件的 ⇒ 由构建脚本 `create_dir_all` 建出 `tenth/target/tmp`。**这不是「需要手工 `mkdir`」**——手工建只在当前这次有效，`fresh clone` / `cargo clean` 后会退回「缺目录」型假红。
3. **常驻守卫**：`tenth/tests/temp_dir_writable_test.rs`（1 例、默认套件、**无 `#[ignore]`**）——建/写/读/删探针；越界时报红，文案指向本节、`[env]` 两条与 `build.rs`。**它只断言「可写」，不断言 `temp_dir()` 等于 `tenth/target/tmp`**（Linux/macOS 读 `TMPDIR`，钉路径会在三平台 CI 假红）。

#### 4 条已知边界（独立对抗性复核实测，2026-09-29）

1. **只对 cargo 启动的进程生效**：绕过 cargo 直接跑 `tenth/target/release/deps/*.exe`（IDE「直接运行二进制」、非 cargo 的 runner 同理）会回到真实 `%TEMP%` ⇒ 本机沙箱下 `Os { code: 5, kind: PermissionDenied }`。此时须自行把 `TEMP`/`TMP` 指向工作区内目录（实测：同一个 exe，自设后 7 passed）。
2. **残窗（只在「过滤跑」咬人）**：`target/` 完好时**手工**删掉 `tenth/target/tmp`，构建脚本**不会**重跑（刻意不发 `rerun-if-changed`）⇒ 过滤跑（如 `cargo test --test mnist_loader_test`）会 4 failed / `Os { code: 3, kind: NotFound }`。**但整跑通常自愈**：多个测试的 `create_dir_all(temp_dir().join(...))` 会连带建出父目录，按字母序更早的目标先跑 ⇒ 全量套件常全绿。⇒ **本节不承诺「残窗时守卫一定会红」**；守卫只在过滤跑时可靠地喊。
   **有效修法**：`touch tenth/build.rs`，或**在 `tenth/` 下**执行 `cargo clean -p tenth --release`。**注意**：不带 `--release` 的 `cargo clean -p tenth` 实测**无效**（与产物 profile 不匹配 ⇒ 构建脚本不重跑、目录不回来）。
3. **Linux/macOS 是 no-op**：Rust 在那边读 `TMPDIR`，不读 `TMP`/`TEMP` ⇒ 该配置与常驻守卫对两平台 CI 无影响、行为不变（**静态判定**：本机无法执行 Linux/macOS 运行，未实测）。
4. **`tenth/target/tmp` 是持久草稿目录**：跨运行持久、debug/release **共用**、**不自动清理**（部分测试用固定文件名，会看到上次残留）；被根 `.gitignore` 的 `target/` 覆盖 ⇒ **不入库、不产生未跟踪文件**。可安全删除，删后按上一条的有效修法让构建脚本重跑。

#### 副作用（诚实披露）

- **`force = true` 会静默覆盖调用者显式导出的 `TMP`/`TEMP`**（这正是它的用途）。⇒ **有意让测试用别的临时目录的，改 shell 无效，必须改 `.cargo/config.toml`**（否则会得到「我明明设了却不生效」的反向困惑）。
- **生效前提是「cargo 从仓库内发起」**：cargo 按 **cwd 向上**发现 `.cargo/config.toml`，**不是**按 `--manifest-path` ⇒ 这也覆盖 `tenth/tools/tenthpm` 的 10 处 `temp_dir()` 单元测试。**未确证**：从仓库外发起 cargo 时是否仍被覆盖（未实测）。

#### `tenthpm` 的边界后果（**约束后果，不是缺陷**）

无 `Tenth.toml` 时 `tenthpm install <pkg>` 走全局安装，写家目录 `~/.tenth/packages`（`tenth/tools/tenthpm/src/commands/install.rs:284-285`）⇒ 在**本机受限沙箱内**运行仓库内构建的那份 tenthpm 时**必然失败**；错误经 `?` 传播 ⇒ **响亮**（用户看得见）。发布的 `bin/tenthpm.exe` 解压到**工作区外**运行不受此限。⇒ 按「**写边界的约束后果**」对待，**不是缺陷**；是否让全局目录可配置由后续裁定，本文件不改代码语义。

> **判据旁证（2026-09-29 实测）**：受限进程的令牌完整性级别为 `Low Mandatory Level (S-1-16-4096)`，工具 shell 为 `Medium`；同一 `cmd.exe` 在工作区外启动为 `Medium`、复制进工作区后启动为 `Low`。`%TEMP%` 本身不带低完整性标签 ⇒ 低完整性进程写入被拒。**基线口径 as-of 2026-09-29**：全量 = **154 结果行 / 2671 passed / 0 failed / 22 ignored**（153 个测试二进制 ＝ 2 unittests + 151 集成，另 1 个 Doc-tests；两个 cwd 逐字相同；C4 新增 `write_loud_guard_test.rs` 13 例 ⇒ `2658 + 13 = 2671`）。**唯一事实源＝`能力梳理/能力全梳理.md` §统计基线**，本节不重复维护该数字。

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
| `wasmtime` | 46.0.1 | WASM JIT 执行引擎（路径 C 闭环） |
| `cranelift` | 0.115 | Cranelift JIT 后端（热点编译） |
| `cranelift-module` | 0.115 | Cranelift 模块抽象 |
| `cranelift-jit` | 0.115 | Cranelift JIT 运行时 |
| `cranelift-native` | 0.115 | Cranelift 本机平台检测 |

### tenthpm 包管理器依赖

定义于 `tenth/tools/tenthpm/Cargo.toml` 的 `[dependencies]` 段。

| crate | 版本 | 用途 |
|-------|------|------|
| `serde` | 1 | 序列化框架（derive） |
| `serde_json` | 1 | JSON 解析（包注册表通信） |
| `toml` | 0.8 | Tenth.toml 清单文件解析 |

### LSP 服务器依赖

定义于 `tenth/tools/lsp/Cargo.toml` 的 `[dependencies]` 段。

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

## 供应链安全（L-6 建议）

项目存在三个独立 `Cargo.lock`：`tenth/Cargo.lock`、`tenth/tools/tenthpm/Cargo.lock`、`tenth/tools/lsp/Cargo.lock`。版本漂移可能引入已知漏洞（CVE）。建议在 CI 中加入以下两步扫描：

### `cargo audit`（RustSec 漏洞数据库）

```bash
# 安装（一次性）
cargo install cargo-audit

# 扫描主 crate
cargo audit --manifest-path tenth/Cargo.toml

# 扫描工具 crate
cargo audit --manifest-path tenth/tools/tenthpm/Cargo.toml
cargo audit --manifest-path tenth/tools/lsp/Cargo.toml
```

`cargo audit` 比对 `Cargo.lock` 与 [RustSec Advisory Database](https://rustsec.org/advisories/)，发现已知漏洞时非零退出。

### `cargo deny`（许可证/重复依赖/漏洞三合一）

```bash
# 安装（一次性）
cargo install cargo-deny

# 初始化配置（一次性，生成 deny.toml）
cargo deny init --manifest-path tenth/Cargo.toml

# 检查
cargo deny check --manifest-path tenth/Cargo.toml
```

`cargo deny` 同时检查：
- 漏洞（与 cargo-audit 同源）
- 许可证白名单（默认拒绝 GPL/AGPL 等 copyleft 许可证进入商业项目）
- 重复依赖（`duplicate` 版本漂移检测）
- banned 依赖（可在 `deny.toml` 中显式禁止某些 crate）

### CI 集成建议

在 GitHub Actions / GitLab CI 中添加 step：

```yaml
- name: cargo audit
  run: |
    cargo install cargo-audit
    cargo audit --manifest-path tenth/Cargo.toml
    cargo audit --manifest-path tenth/tools/tenthpm/Cargo.toml
    cargo audit --manifest-path tenth/tools/lsp/Cargo.toml
```

建议每周定时跑一次（依赖漏洞会随时间累积），并在每次 PR 中跑一次。

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
