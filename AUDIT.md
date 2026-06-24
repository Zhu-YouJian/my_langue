# 项目总览与审计报告

> 日期：2026-06-24 | 版本：v0.3.3 | GPU 脚手架 + 包管理器 + LSP + 语言增强 | 499 项测试（498 passed + 1 ignored）

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
| 16 | **Test 覆盖盲区** — `tenthc_test.rs` 只测解析，未测执行；tenthc 子编译器的正确性无自动化验证 |

---

## 八、灵感与改进方向

### 8.1 短期 (解锁自举)

- **~~修复致命缺陷 #1-#4~~** → ✅ 已修复，自举闭环可达
- **添加 `tenthc_combined.th` 自动生成** — build script 或 Makefile，而非手动拼接
- **`tenthc_test.rs` 加执行测试** — 改为 VM/WASM 路径验证

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

*文档由 2026-05-27 全项目审计生成，作为后续正式开工的参考基准。*
