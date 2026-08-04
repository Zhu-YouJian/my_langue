# Tenth 发布流程与跨平台产物（M5.3）

> 本文档是发布流程的权威说明 + M5.3 跨平台调研结论（真理源）。
> 版本：v1.0.0（2026-08-04）。涉及代码修改见 `MEMO.md` M5.3/M5.4 条目。

## 一、产物清单

| 产物 | 二进制名 | 来源 crate | 说明 |
|------|---------|-----------|------|
| 主编译器/运行时 | `tenth`(.exe) | `tenth/` | 语言前端 + VM/解释器/JIT + WASM 宿主（wasm-encoder 生成 + wasmi/wasmtime 执行） |
| 包管理器 | `tenthpm`(.exe) | `tenth/tools/tenthpm/` | 10 子命令（M4.1 完整） |
| 调试器 | `tenth-debug`(.exe) | `tenth/tools/debugger/` | 解释器路径 CLI 调试器（M4.4） |
| 剖析器 | `tenth-prof`(.exe) | `tenth/tools/profiler/` | VM 路径热点剖析（M4.4） |
| 语言服务器 | `tenth-lsp`(.exe) | `tenth/tools/lsp/` | LSP 13 项能力（M4.2） |
| 自举 WASM 产物 | `tenthc_full.wasm` | `tenthc/`（生成物） | 自举编译器 WASM 产物（路径 A/C 闭环，`tenthc_full.wasm` 不入 git，每次自举重新生成） |

## 二、平台落地程度（M5.3 实测/评估）

### Windows（✅ 已落地验证）
- 全部 5 产物 release 构建通过（`tenth.exe` 13.1MB / `tenthpm.exe` 11.3MB / `tenth-debug.exe` 7.9MB / `tenth-prof.exe` 8.0MB / `tenth-lsp.exe` 2.9MB，2026-08-04 实测）
- 自举 `[OK] Full compiler compiled to tenthc_full.wasm`（92.5KB，EXIT=0）
- 打包：`scripts/release/package.ps1`（zip + SHA-256）+ `verify.ps1`（校验）
- 平台相关代码仅 `#[cfg(windows)]` 的 `SetConsoleOutputCP(65001)`（`tenth/src/main.rs`、`debugger`、`profiler`）——条件编译，非 Windows 自动忽略，跨平台安全

### Linux / macOS（✅ CI 交付，本地未交叉）
- **交叉编译评估**：Windows（MSVC host）→ Linux 交叉编译需安装 `x86_64-unknown-linux-gnu` target + GNU 链接器（MSVC 产物无法链接到 ELF）；→ musl 需 `x86_64-unknown-linux-musl` target + musl-gcc 工具链。因项目依赖链含 wasmtime/cranelift（体积大、编译重），且本机无 Linux/macOS 工具链，**未做本地交叉**，改由 **GitHub Actions matrix 原生构建**（更可靠、无交叉工具链维护成本）。
- **交付物**：`.github/workflows/release.yml`——`ubuntu-latest / macos-latest / windows-latest` 三平台原生 `cargo build --release` 全产物 + 自举 [OK] + 关键测试（stdlib_smoke / integration / wasm_backend_minimal）+ 打包上传。tag `v*` 或手动触发。
- **关键修复（编译部）**：`tenth/build.rs` 原硬编码 `cargo:rustc-link-arg=/STACK:67108864`（MSVC 专用）——非 Windows 平台 GNU/Apple 链接器会把它当输入文件报 `cannot find /STACK:` 阻塞构建。已按 `CARGO_CFG_TARGET_OS`/`TARGET_ENV` 平台化：MSVC `/STACK:`、MinGW `-Wl,--stack,`、Linux `-Wl,-z,stack-size=`、macOS `-Wl,-stack_size,`（64 MiB 栈，解释器深递归所需）。
- **平台相关性调研结论**：所有 crate 均无 `[target.'cfg(...)'.dependencies]` 平台特定依赖；文件/进程 API（`std::path`/`std::process::Command`/`current_dir`/`current_exe`）均跨平台；无信号处理代码。Linux/macOS 原生构建无阻塞项（除上述 build.rs 已修）。

### WASM（✅ 已落地，作为宿主）
- **tenth 主 crate 无法编译到 wasm32 target（已实测）**：`rustup target add wasm32-unknown-unknown` 后对 LSP crate（依赖最少）执行 `cargo check --target wasm32-unknown-unknown` → **EXIT=101**，`cranelift-codegen v0.115.1` 的 build.rs panic：`error when identifying target: "no supported isa found for arch `wasm32`"`。依赖链含 `wasmtime`（WASM JIT 引擎，官方不支持 wasm 目标）/`cranelift`（JIT 后端，无 wasm32 ISA）/`rustyline`（REPL，需 tty）/`ndarray`。`tenth` 的 WASM 能力是**作为宿主**：`wasm-encoder` 生成 `.wasm`（`tenth build foo.th`）+ `wasmi` 解释执行（`tenth wasm foo.th`）+ `wasmtime` JIT 执行（路径 C 闭环）。
- **自举 WASM 产物 `tenthc_full.wasm`**：`tenthc/main.th` 拼接 `tenthc/*.th` 后调用 `compile_host` native 生成（相对路径写 cwd，`.gitignore` 忽略）。可复现：每平台 CI 自举步骤生成并验证 `[OK]`。产物依赖 host imports（`tenth_alloc`/`Vec::*` 等），**不能独立运行**（设计如此，见 `docs/论文/T14-自举管线封闭性与固定点.md` §F1）。
- 发布包不单独含 `.wasm`（由用户自举生成）；如需随包发布可在 CI 打包步骤附加。

## 三、发布步骤

```bash
# 1. 本地（任一平台）构建 + 冒烟
cargo build --release -j 2 --manifest-path tenth/Cargo.toml
cargo run --release --manifest-path tenth/Cargo.toml -- run tenthc/main.th   # 期望 [OK]

# 2. 全量测试（发布前，总师验收）
cargo test --release -j 2 --manifest-path tenth/Cargo.toml

# 3. 打包（Windows）
powershell -ExecutionPolicy Bypass -File scripts/release/package.ps1

# 4. 打包（Linux/macOS/CI）
bash scripts/release/package.sh

# 5. 校验
powershell -ExecutionPolicy Bypass -File scripts/release/verify.ps1
# 或： cd dist && sha256sum -c SHA256SUMS.txt

# 6. CI 发布（跨平台全产物）
git tag v1.0.0 && git push origin v1.0.0   # 触发 .github/workflows/release.yml
```

产物结构（`dist/tenth-<ver>-<platform>/`）：
```
bin/            # 5 个可执行产物
std/            # 标准库（运行时 use std::... 必需，随包分发）
docs/           # 语言参考手册 / 语言规范 / API冻结清单 / README
```
配套 `SHA256SUMS.txt`（zip/tar.gz 的 SHA-256）。

## 四、已知遗留（如实）

1. **本地交叉编译未做**（Windows→Linux/macOS 需 GNU/Apple 工具链，成本高、可靠性低于 CI 原生构建）——以 CI matrix 为准
2. **wasm32 target 编译未打通且明确不可行**（wasmtime/cranelift 依赖链）；浏览器端运行（wasm32-unknown-unknown + WASI）属远期（能力全梳理 §5.9 WebAssembly 推理 ⚠️）
3. **tenthc_full.wasm 依赖 host imports**，不能独立运行；路径 C 的"完全 WASM 闭环"实际是"WASM + 宿主运行时"联合体（T14 论文 F1 已记录）
4. **CI 未实测**（本机无 GitHub Actions 运行条件）——workflow 语法按 GitHub Actions 标准编写，首次运行需验证；`package.sh` 在 Linux/macOS 未实测（bash 语法已 review）
5. macOS `-Wl,-stack_size` 参数按 Apple ld 语法编写，未实测（64 MiB 满足页对齐）
