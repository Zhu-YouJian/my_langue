# 安全声明

> **版本**：v2.0 | **更新日期**：2026-06-29
> **本版本由全面安全审查（详见 `security_review.md`）驱动重写**，纠正了此前版本中关于 `unsafe` 数量、执行路径与内存护栏覆盖范围的失实声明。

---

## ⚠️ 整体威胁模型

**Tenth 是一门通用编程语言，不是沙箱**。运行任意 `.th` 文件等同于运行任意 Rust 程序——具备宿主进程的全部文件系统、子进程、网络能力。在运行不可信代码前，请在隔离环境（容器/VM）中执行，或使用 `--fs-root` 沙箱选项（详见下文）。

---

## 当前真实安全态势

| 项目 | 实际情况 | 备注 |
|------|---------|------|
| `unsafe` 代码块 | **41+ 处** | 集中在 `compile/jit/hostcalls.rs`、`compile/jit/context.rs`、`compile/jit/mod.rs`。JIT 是 `tenth run` 的默认执行路径，非可选项 |
| 执行路径 | **JIT / VM / 解释器 / WASM 四路径** | `main.rs::vm_execute` 默认调用 `jit::run_jit`，失败时 fallback 到 VM/解释器。"仅解释器路径"声明失实 |
| 内存护栏覆盖 | **仅 REPL 默认启用** | `run_file` 默认不应用 `MemoryConfig`；`mem-strict` feature（默认未启用）才硬阻断，否则仅 soft warning |
| 文件 I/O 原生函数 | **零沙箱、零白名单** | `read_file`/`write_file`/`remove_file`/`mkdir`/`copy_file`/`rename_file`/`list_dir` 等 14 个原生函数接受任意路径 |
| CPU/时间限制 | **无** | `while true {}` 即可永久挂起宿主 |
| 子进程调用 | **仅 `git clone/pull`**（tenthpm） | 路径参数已加包名校验，但 git 协议未限制 |
| 网络原生函数 | **无** | 项目本身不直接调用网络；运行时产生的张量运算等不联网 |

---

## 已修复的安全问题（2026-06-29）

下列问题已在本次安全审查后修复，详见 `security_review.md` 与 `MEMO.md`：

| ID | 严重度 | 摘要 | 修复位置 |
|----|--------|------|---------|
| C-1 | 🔴 致命 | tenthpm 包名注入导致路径穿越 / 任意目录删除 | `tools/tenthpm/src/manifest.rs`（新增 `validate_package_name`、`safe_package_name_from_git`、`ensure_within`、`safe_to_remove_dir`）、`commands/install.rs`、`commands/add.rs` |
| C-2 | 🔴 致命 | SECURITY.md 失实声明掩盖真实攻击面 | 本文件 |
| H-1 | 🟠 高危 | JIT hostcall `from_raw_parts` 缺乏溢出校验 | `compile/jit/hostcalls.rs` |
| H-3 | 🟠 高危 | `run_file` 默认无内存限制 | `main.rs`、`runtime/limits.rs` |
| H-5 | 🟠 高危 | `time_sleep_ms` 接受负数 → 永久 DoS | `main.rs`、`runtime/interpreter.rs` |
| H-6 | 🟠 高危 | JSON 解析器无深度限制 + 转义缺陷 | `main.rs`、`runtime/interpreter.rs` |
| H-7 | 🟠 高危 | `random_int`/`random_float` 使用可预测种子 | `main.rs`、`runtime/interpreter.rs` |
| H-8 | 🟠 高危 | WASM 宿主 `ptr as usize` 越界 panic | `compile/wasmtime_host.rs`、`compile/wasm.rs` |
| M-1 ~ M-8 | 🟡 中等 | transmute、mem-strict panic、release profile、checked 算术等 | 各文件 |

---

## `unsafe` 代码边界（透明披露）

`compile/jit/` 下的 `unsafe` 不可消除（JIT 必然涉及可执行内存与 FFI）。所有 `unsafe` 块须满足以下不变量，违反即 UB：

1. **`std::mem::transmute(raw_ptr)` → `JitFn`**（[context.rs:47](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/jit/context.rs#L47)）：`raw_ptr` 必须由 cranelift `JITModule::get_definition` 返回；声明的函数签名必须与 `translator.rs` 生成的函数签名一致。已加尺寸断言。
2. **`std::slice::from_raw_parts(args_ptr, count)`**（[hostcalls.rs:186, 198, 215, 220, 237, 297, 337](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/jit/hostcalls.rs#L186)）：`args_ptr` 必须指向 `count` 个 `Value` 的有效内存；`count` 必须 ≤ `MAX_HOSTCALL_ARGS`（1<<20），且所有 `count * N`、`rows * cols` 类运算用 `checked_mul` 防溢出。
3. **`vm as *mut Vm` 传入 JIT**（[mod.rs:81](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/jit/mod.rs#L81)）：JIT 调用期间 `Vm` 不可被移动；所有 hostcall 已用 `catch_unwind` 包裹，避免 unwind 跨 FFI 边界（UB）。
4. **`std::ptr::write`**（数十处）：偏移量来自翻译器，不得超过缓冲区容量。

---

## 内存与资源护栏

| 维度 | 默认行为 | 强制行为 |
|------|---------|---------|
| 内存（分配字节） | `run_file` 默认应用 `MemoryConfig::default()`，超限硬错误返回 | 同默认；`--no-limits` 显式关闭 |
| 张量元素数 | 受 `max_tensor_elements` 限制 | 同上 |
| 调用栈深度 | 受 `max_call_depth` 限制 | 同上 |
| CPU 时间 | 仍**无限制**（计划中） | 计划加 `--timeout <secs>` |
| 整数溢出 | release profile 启用 `overflow-checks = true` | 强制 |

> 历史 `mem-strict` feature 的 `panic!` 行为已改为返回 `TenthError`，避免长会话 REPL 因触发限制而被杀进程。

---

## 沙箱选项

```
tenth run --fs-root <dir> [--read-only] [--no-fs] file.th
```

- `--fs-root <dir>`：所有文件 I/O 路径规范化后必须 `starts_with(dir)`，符号链接逃逸被拒绝。
- `--read-only`：禁止 `write_file`/`remove_file`/`mkdir` 等修改类操作。
- `--no-fs`：禁用全部文件 I/O 原生函数。

> **示例**：运行 `Tenth实例/` 下不可信示例时建议 `tenth run --fs-root /tmp/sandbox --read-only demo.th`。

---

## 文件 I/O 原生函数清单（攻击面）

下列函数接受任意路径，是 `.th` 程序访问宿主文件系统的全部入口。沙箱模式下路径必须落在 `--fs-root` 之内：

- 读：`read_file`、`read_bytes`、`list_dir`、`path_exists`、`path_is_file`、`path_is_dir`、`path_metadata`
- 写：`write_file`、`write_bytes`、`mkdir`、`remove_file`、`copy_file`、`rename_file`

WASM 后端的 `read_file`/`write_file` 宿主导入同样受沙箱约束。

---

## 包管理器（tenthpm）安全

- **包名校验**：所有从 git URL / 本地路径 / 注册中心名称派生的目录名必须通过 `manifest::validate_package_name`，拒绝 `.`、`..`、含分隔符、Windows 保留名、控制字符等。
- **路径穿越防御**：`install_global`/`install_local`/`add_git_dependency` 在计算 `target_dir` 后调用 `ensure_within(deps_root, target_dir)` 二次校验。
- **删除闸门**：所有 `fs::remove_dir_all` 调用前调用 `safe_to_remove_dir`，拒绝删除根、用户主目录、`.ssh`/`.aws`/`.config` 等敏感目录。
- **git 协议**：默认放行 `https://`/`http://`/`git://`/`ssh://`；建议生产环境用 `--allow-ssh` 显式开关。

---

## 报告漏洞

发现安全漏洞请通过私有渠道报告（而非公开 issue），便于修复后再披露。
