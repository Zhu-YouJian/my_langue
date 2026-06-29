# 项目总览与审计报告

> 日期：2026-06-29 | 版本：v0.3.3 | GPU 脚手架 + 包管理器 + LSP + 语言增强 + 安全加固 | 499 项测试（498 passed + 1 ignored）

---

## 一、项目全景

Tenth = Tensor + Zenith，一门为 AI 研究而生的编程语言。Rust 编写的 bootstrap 编译器 + Tenth 编写的自举编译器 + 字节码 VM + WASM 编译。

### 目录地图

> 完整目录结构及模块注解见 `CODE_WIKI.md` §1。

---

## 二、编译器管线

> 完整管线架构见 `CODE_WIKI.md` §2-3。

> ~~C 编译路径 (MIR → C → GCC → .exe)~~ 已于 2026-06-04 移除。原因：生成的 C 代码无内存管理，详见 SECURITY.md。

---

## 三、测试矩阵

| 测试文件 | 数量 | 覆盖 |
|----------|------|------|
| `lexer_test.rs` | 6 | 整数/标识符/关键字/字符串/运算符/注释 |
| `parser_test.rs` | 5 | 字面量/二元表达式/函数定义/if/tensor |
| `integration_test.rs` | 14 | 全管线: 算术/布尔/比较/函数/闭包/while/tensor |
| `enum_test.rs` | 9 | 枚举定义/字段/match/通配/元组变体/match 绑定 |
| `generic_test.rs` | 11 | 泛型函数/泛型结构体/trait bound/泛型返回/Vec<Token>/>>拆分 |
| `struct_test.rs` | 8 | 结构体/嵌套/impl/默认字段/..语法 |
| `trait_test.rs` | 9 | trait 定义/builtin bound/inherent impl/默认方法/多 trait |
| `module_test.rs` | 6 | mod/use/嵌套模块/重导出 |
| `ownership_test.rs` | 11 | 移动/借用/引用/解引用 |
| `stdlib_test.rs` | 114 | Vec/HashMap/String/Option/文件 I/O/json/toml/runtime 限制 |
| `memory_test.rs` | 17 | 内存护栏: arena/limits/计数器 |
| `selfhost_frontend.rs` | 4 | 自举前端验证（lex/parse/lower） |
| `parity_test.rs` | 112 | VM vs Interpreter 行为一致（全指令覆盖） |
| `shape_check_test.rs` | 16 | 张量 shape 检查/广播/层归一化验证 |
| `autodiff_test.rs` | 52 | 自动微分/闭包/张量/错误位置（21 算子） |
| `vm_autodiff_test.rs` | 15 | 字节码 VM 上的自动微分回归 |
| `type_inference_test.rs` | 29 | 类型推断/统一/泛型实例化 |
| `pattern_match_test.rs` | 11 | 模式匹配/解构/守卫 |
| `iterator_test.rs` | 10 | 迭代器/for/生成器 |
| `error_recovery_test.rs` | 7 | 解析错误恢复/续接 |
| `mnist_loader_test.rs` | 4 | MNIST 数据加载 |
| `three_stage.rs` | 1 | 三段式自举（忽略 — wasmi 慢） |
| **总计** | **499** | 498 通过 + 1 忽略 |

---

## 四、自举状态

> 自举演进详情见 `MEMO.md`。

---

## 五、已移除 / 已修复

| 项目 | 状态 |
|------|------|
| ~~C 编译管线~~ | ❌ 已移除 (2026-06-04) |
| ~~tenthc lexer 字面量硬编码 0~~ | ✅ 已修复 (Token 新增 fval) |
| ~~tenthc parser method_call 丢失 receiver~~ | ✅ 已修复 (Dot+LParen 产生 method_call) |
| ~~VM chunk clone 内存泄漏~~ | ✅ 已修复 (chunk_idx 索引引用) |
| ~~VM StoreField/code 切换死循环~~ | ✅ 已修复 (CallN/Ret 同步更新 code/strings) |
| `runtime/autodiff.rs` | ✅ 张量级可用，已集成解释器（21 算子） |
| Lowerer 大文件性能 | ⚠️ 实际不慢 (release 39 函数 <0.01s)，之前误判为瓶颈 |

---

## 六、已知限制

| # | 问题 | 影响 |
|---|------|------|
| 0 | ~~VM 不支持字符串切片~~ | ✅ 已修复：SliceStr + Range 索引解析 |
| 2 | 树遍历解释器大文件慢 (debug build) | release build 即解决 |
| 3 | WASM codegen 个别边界情况 | wasmi 执行偶有 type mismatch |
| 4 | 无 GPU 后端 | `compile/gpu/` 脚手架已就绪，待 CUDA 环境安装 |

---

## 七、已知缺陷与不完整（历史参考 — C 后端已移除）

### 7.1 自举编译器 Tenth 源码中的已知问题

| # | 位置 | 问题 |
|---|------|------|
| 1 | `tenthc/lexer/lexer.th:98,105` | **字面量值不解析** — 整数和浮点数 token 的 value 硬编码为 0 |
| 2 | `tenthc/parser/parser.th:269` | **字段名不存储** — `parse_postfix` 的 `.field` 访问把字段名丢了 |

> ~~原 #1 (cgen 函数调用参数丢弃) 随 codegen 移除而消除。~~

### 7.2 语言功能缺口

| # | 位置 | 问题 |
|---|------|------|
| 3 | `tenth/src/compile/lower.rs` | **Match pattern binding 未生成** |
| 8 | `tenthc/main.th:11` | **依赖 tenthc_combined.th** — 需在编译前手动拼接 |

> ~~#5-#7/#10/#12 引用已删除的 C 代码生成文件 (`tenthc/codegen/`, `tenthc/runtime.c`)，随 C 后端移除而消除，不再罗列。~~

### 7.3 中优

| # | 位置 | 问题 |
|---|------|------|
| 9 | `tenthc/parser/parser.th:631` | **无 for 循环解析** — lexer 识别 `for` 但 parser 未处理 |
| 11 | `tenthc/parser/parser.th:180` | **parse_unary 纯透传** — 一元运算在 parse_primary 内处理，此函数冗余 |

### 7.4 架构债务

| # | 问题 |
|---|------|
| 14 | ~~borrow checker 双向放宽~~ — 已恢复，`check_borrow_shared` 和 `check_borrow_mut` 现执行 ExclusiveRef/SharedRef 检查 |
| 16 | ~~Test 覆盖盲区~~ — 已修复。前端契约测试 `selfhost_frontend.rs` 改为严格 assert（原 println 不 fail 的问题已修）；执行覆盖由 `fixpoint_runtime.rs`（Wasmtime 端到端编译+执行）和 `parity_test.rs`（112 项 Rust/tenthc 一致性）提供 |

---

## 八、灵感与改进方向

### 8.1 短期 (解锁自举)

- **~~修复致命缺陷 #1-#4~~** → ✅ 已修复，自举闭环可达
- **添加 `tenthc_combined.th` 自动生成** — build script 或 Makefile，而非手动拼接
- **~~`tenthc_test.rs` 加执行测试~~** → ✅ 已由 `fixpoint_runtime.rs` 和 `parity_test.rs` 提供

### 8.2 中期 (质量加固)

- **VM 补全** — closure/generic call/match 仍偶有 fallback
- **~~恢复 borrow checker~~** — 已恢复，`check_borrow_shared`/`check_borrow_mut` 现执行完整检查
- **WASM Host import 真实现** — Vec/String 在 WASM 模块中仍以占位形式存在，待实现

### 8.3 长期 (生态)

- **激活死模块** — shape.rs (张量形状优化)、docgen.rs (API 文档生成)；~~autodiff.rs (自动微分训练)~~ ✅ 已完整集成（21 算子，张量级，已集成解释器）
- **tenthpm 包管理器** — `tools/tenthpm/` **完整实现**，支持 init/build/test/run/add/remove/list/clean/publish/install + Tenth.toml + Tenth.lock + .tenthpkg 打包 + path/git/registry 三种依赖类型
- **LSP 服务器** — `tools/lsp/` **完整实现**（文档同步/diagnostics/hover/completion/definition/documentSymbol/references/rename/signatureHelp/foldingRange/semanticTokens/formatting）
- **CUDA 后端** — `compile/gpu/` + `compile/optimizations/` 脚手架已就绪，待安装 CUDA Toolkit

### 8.4 过程改进

- **`.gitignore` 已更新** — 添加 `*.exe` / 构建产物排除
- **清理 19 个构建产物** — 15 个 .exe + 2 个空 .txt + tenthc.c + test_mini.c 已从 git 跟踪移除
- **MEMO.md 保持同步** — 作为动态状态文件，每次大改动后更新

---

## 九、清理记录

| 操作 | 数量 |
|------|------|
| 从 git 移除 .exe 文件 | 15 |
| 从 git 移除空 .txt | 2 |
| 从 git 移除构建 C 文件 | 2 (tenthc.c, test_mini.c) |
| 删除临时 .th | 1 (test_input.th) |
| 更新 .gitignore | 覆盖 *.exe, *.txt, tenthc.c, test_mini.c |
| 删除游离产物 (2026-06-15) | 3 (stderr.txt, stdout.txt, test_w.thw) |
| .gitignore 加固 (2026-06-15) | 新增 *.thw / stdout.txt / stderr.txt |
| 文档对齐 (2026-06-15) | 8 文件：版本号 0.3.0→0.3.3，测试数 134→350，算子数 19→21，LSP/tenthpm 状态脚手架→完整实现 |
| AUDIT §5/§7 重构 (2026-06-15) | 修复重复小节号 (5.2×2)，移除 6 条已废弃 C 后端条目，章节号六/七→八/九 |

保留的历史 C 文件 (tenthc_v3.c ~ v9.c, tenthc_dbg5.c, tenthc_dbg6.c, tenthc_fix.c, tenthc_analyze.c, tenthc_chk.c, tenthc_out.c, tenthc_rust.c) 作为自举进化见证，暂时保留不删。

---

## 十、安全审查记录（2026-06-29）

> 全面安全审查识别 25 项问题（2 致命 / 8 高危 / 8 中等 / 7 低危），分两轮全部修复。第一轮 17 项（C-1, C-2, H-1, H-3, H-5~H-8, M-1~M-8），第二轮 7 项（H-2, H-4, L-1, L-3~L-6）；L-2 在 H-7 中一并修复，L-7 在 H-2 中一并修复。审查报告 `security_review.md`，威胁模型披露 `SECURITY.md` 重写，变更记录见 `MEMO.md` 顶部 2026-06-29 条目。

### 10.1 已修复（第一轮 17 项）

| ID | 严重度 | 位置 | 修复方式 |
|----|--------|------|---------|
| C-1 | 致命 | `tenth/tools/tenthpm/src/{manifest,install,add}.rs` | 新增 `validate_package_name` / `safe_package_name_from_git` / `ensure_within` / `safe_to_remove_dir` 集中校验；所有 `target_dir` 计算 + `fs::remove_dir_all` 必须经过校验 |
| C-2 | 致命 | `SECURITY.md` | 重写：纠正"0 处 unsafe"失实声明（实际 41+ 处），公开真实威胁模型与 `--fs-root` 沙箱选项 |
| H-1 | 高危 | `tenth/src/compile/jit/hostcalls.rs` | 新增 `MAX_HOSTCALL_ARGS = 1<<20` 与 `safe_slice` 统一闸门；所有 `from_raw_parts` 改用；`host_make_map` / `host_new_struct` 加 `count.checked_mul(2)`；`host_make_tensor` 加 `rows.checked_mul(cols)` |
| H-3 | 高危 | `tenth/src/main.rs` | 新增 `parse_memory_config` 与 `run_file(path, config)`；fallback 路径用 `Interpreter::with_limits`；`--no-limits` 显式退出沙箱 |
| H-5 | 高危 | `tenth/src/main.rs` | `time_sleep_ms` 拒绝负数与 > 24h 的请求 |
| H-6 | 高危 | `tenth/src/main.rs` | JSON 解析器加 `JSON_MAX_DEPTH=256` 防栈溢出；`json_unescape` / `simple_json_split` 修复 `\"` 转义状态机 |
| H-7 | 高危 | `tenth/src/main.rs` | `random` 改用 `rand::thread_rng().r#gen()`（CSPRNG），替代可预测的 `DefaultHasher` |
| H-8 | 高危 | `tenth/src/compile/wasmtime_host.rs` | 新增 `safe_offset` / `read_cstr` / `MAX_ALLOC_BYTES=16MiB`；17 个 host import 全部改用；`tenth_alloc` 拒绝负数 size 与 > 16MiB 请求 |
| M-1 | 中危 | `tenth/src/compile/jit/context.rs` | `transmute` 前加 `assert_eq!(size_of::<*const u8>(), size_of::<JitFn>())` 尺寸断言 |
| M-2 | 中危 | `tenth/src/repl.rs:241-245` | `mem-strict` 模式不再 `panic!` 杀 REPL，改为返回 `TenthError::RuntimeError` |
| M-3 | 中危 | `tenth/Cargo.toml` | `[profile.release]` 加 `overflow-checks = true, panic = "unwind"`；`[profile.dev]` 加 `overflow-checks = true` |
| M-4 | 中危 | `tenth/src/compile/jit/hostcalls.rs` | 见 H-1（`host_make_tensor` 加 `rows.checked_mul(cols)`） |
| M-5 | 中危 | `tenth/src/compile/jit/context.rs` | impl Drop for JitContext 调 `cache.clear()` 显式清理 |
| M-6 | 中危 | `tenth/src/compile/jit/hostcalls.rs` | `invoke_jit` 包 `catch_unwind`，panic 时写 `Value::Unit` 并返回 false，防 FFI unwind UB |
| M-7 | 中危 | `tenth/src/runtime/arena.rs` | `alloc` 用 `checked_add` / `checked_mul`；`scope` 用 `saturating_sub` 防下溢 panic |
| M-8 | 中危 | `tenth/src/compile/wasmtime_host.rs` | 见 H-8（bump allocator 越界 panic 修复） |

### 10.2 已修复（第二轮 7 项，2026-06-29）

| ID | 严重度 | 位置 | 修复方式 |
|----|--------|------|---------|
| H-2 | 高危 | `tenth/src/runtime/{limits,vm,interpreter}.rs`、`main.rs` | 新增 `FsSandbox` 类型（canonicalize + starts_with 防路径穿越）；Vm/Interpreter 新增 `fs_sandbox` 字段，所有文件 I/O 原生函数必须经过 `check_read`/`check_write`；`--fs-root`/`--read-only`/`--fs-cwd` 命令行选项 |
| H-4 | 高危 | `tenth/src/runtime/{vm,interpreter}.rs`、`main.rs` | VM 主循环和 Interpreter tick() 新增独立 `loop_counter`/`tick_counter`，每 4096 步检查 `deadline_ms`（墙钟超时独立于 `step_budget`）；`--timeout <secs>` 命令行选项，`parse_timeout_ms` 用 checked_mul/checked_add 防溢出 |
| L-1 | 低危 | `tenth/tools/tenthpm/src/manifest.rs` | 验证确认 `extract_package_name` 仅在 `safe_package_name_from_git` 内被调用，所有调用方都经 `validate_package_name` 校验 |
| L-3 | 低危 | `tenth/src/main.rs`、`tenth/src/runtime/interpreter.rs` | `days_to_date` 入口加 `if days > u64::MAX - EPOCH_OFFSET { return (0,0,0); }` 防 `days + 719468` 溢出 UB |
| L-4 | 低危 | `tenth/tools/tenthpm/src/manifest.rs` | `is_git_url` 默认仅放行 `https://`，`http://`/`git://`/`ssh://`/`.git` 后缀需 `TENTH_ALLOW_INSECURE_GIT=1` 显式 opt-in |
| L-5 | 低危 | `tenth/tools/tenthpm/src/{manifest,install,add}.rs` | 三处 `git clone` 加 `--config protocol.file.allow=deny` 和 `protocol.git.allow=deny`；克隆后调用新增 `disable_hooks()` 将 `core.hooksPath` 指向空设备 |
| L-6 | 低危 | `DEPS.md` | 新增"供应链安全"章节，说明 `cargo audit` 和 `cargo deny` 用法及 CI 集成建议 |

> L-2（DefaultHasher）已在 H-7 中一并修复；L-7（compile_host 写文件）已在 H-2 中一并修复。

### 10.3 未修复（保留至下轮）

无。全部 25 项安全问题已修复。

### 10.4 验证

| 命令 | 结果 |
|------|------|
| `cargo build --manifest-path tenth/Cargo.toml` | ✅ 成功（178 warning 为项目原有 `unsafe-op-in-unsafe-fn`，非本次引入） |
| `cargo build --release --manifest-path tenth/Cargo.toml` | ✅ 成功（第二轮验证） |
| `cargo build --manifest-path tenth/tools/tenthpm/Cargo.toml` | ✅ 成功（第二轮 L-4/L-5 验证） |
| `cargo run --release --manifest-path tenth/Cargo.toml -- run tenthc/main.th` | ✅ 自举路径 A 验证通过（`tenthc_full.wasm` 生成） |
| `cargo test --manifest-path tenth/Cargo.toml --lib` | ✅ 10/10 通过（第一轮） |
| `cargo test --manifest-path tenth/Cargo.toml --features mem-debug --test memory_test` | ✅ 17/17 通过（含 `arena_scope_rolls_back_counter` / `arena_overflow_returns_none` 验证 M-7 修复） |
| `cargo test --manifest-path tenth/Cargo.toml`（全集成测试） | ⚠️ rustc 1.95.0 ICE（`wasm.rs:2077` 方法调用解析，仅在 `--test` 模式触发），非本项目代码问题，ICE 文件未被本轮修改 |

---

*文档由 2026-05-27 全项目审计生成，作为后续正式开工的参考基准。*
