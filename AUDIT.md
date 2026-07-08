# 项目总览与审计报告

> 日期：2026-07-08 | 版本：v0.3.3 | GPU 脚手架 + 包管理器 + LSP + 语言增强（元组类型 + `?` 操作符）+ 安全加固 + Shape 检查 + Autograd 反向 Shape 校验 + 论文披露缺陷登记 + 同步 I/O 原语 + AUDIT 缺陷修复 + 异步 Phase 2（协程调度 + async I/O） | 790+ 项测试通过（55 个测试目标，含 6 个栈溢出崩溃预存问题）

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

> 数量列格式：`passed/failed/ignored`。"栈溢出" 表示编译通过但运行时触发 Windows STATUS_STACK_OVERFLOW (0xc00000fd)，无法获取具体用例数。统计日期：2026-07-08。

### 基础管线

| 测试文件 | 数量 | 覆盖 |
|----------|------|------|
| `lib`（单元测试） | 16/0/0 | HIR/VM/解释器/autodiff 等模块单元 |
| `lexer_test.rs` | 6/0/0 | 整数/标识符/关键字/字符串/运算符/注释 |
| `parser_test.rs` | 5/0/0 | 字面量/二元表达式/函数定义/if/tensor |
| `integration_test.rs` | 14/0/0 | 全管线: 算术/布尔/比较/函数/闭包/while/tensor |
| `error_recovery_test.rs` | 7/0/0 | 解析错误恢复/续接 |

### 类型系统

| 测试文件 | 数量 | 覆盖 |
|----------|------|------|
| `enum_test.rs` | 9/0/0 | 枚举定义/字段/match/通配/元组变体/match 绑定 |
| `struct_test.rs` | 8/0/0 | 结构体/嵌套/impl/默认字段/..语法 |
| `trait_test.rs` | 9/0/0 | trait 定义/builtin bound/inherent impl/默认方法/多 trait |
| `module_test.rs` | 6/0/0 | mod/use/嵌套模块/重导出 |
| `ownership_test.rs` | 11/0/0 | 移动/借用/引用/解引用 |
| `type_inference_test.rs` | 29/0/0 | 类型推断/统一/泛型实例化 |
| `generic_test.rs` | 11/0/0 | 泛型函数/泛型结构体/trait bound/泛型返回/Vec<Token>/>>拆分 |
| `pattern_match_test.rs` | 17/0/0 | 模式匹配/解构/守卫 |
| `iterator_test.rs` | 10/0/0 | 迭代器/for/生成器 |
| `tuple_test.rs` | 12/0/0 | 元组创建/解构/嵌套/函数返回/空元组（2026-07-08 新增） |
| `try_operator_test.rs` | 12/0/0 | `?` 操作符成功/错误传播/链式/I/O 模拟（2026-07-08 新增） |

### 张量与自动微分

| 测试文件 | 数量 | 覆盖 |
|----------|------|------|
| `autodiff_test.rs` | 54/0/0 | 自动微分/闭包/张量/错误位置（21 算子） |
| `autodiff_shape_test.rs` | 10/0/0 | Autograd 反向 shape 校验（护城河 A） |
| `vm_autodiff_test.rs` | 15/0/0 | 字节码 VM 上的自动微分回归 |
| `abs_test.rs` | 8/0/0 | abs 算子 |
| `select_test.rs` | 16/0/0 | select 算子 |
| `scatter_test.rs` | 16/0/0 | scatter 算子 |
| `gather_test.rs` | 13/0/0 | gather 算子 |
| `bmm_test.rs` | 11/0/0 | batched matmul (bmm) 算子 |
| `reshape_masked_fill_autodiff_test.rs` | 9/0/0 | reshape/masked_fill 自动微分 |
| `multihead_attention_test.rs` | 11/0/0 | MHA 真正分头计算 + 完整梯度链 |

### Shape 检查

| 测试文件 | 数量 | 覆盖 |
|----------|------|------|
| `shape_check_test.rs` | 16/0/0 | 张量 shape 检查/广播/层归一化验证 |
| `shape_check_compile_test.rs` | 74/0/7 | 编译期 shape 检查（7 项 ignored 占位） |

### f32 路线图

| 测试文件 | 数量 | 覆盖 |
|----------|------|------|
| `f32_tensor_test.rs` | 23/0/0 | f32 张量基础（Phase 1） |
| `f32_frontend_test.rs` | 13/0/0 | f32 前端贯通（Phase 2） |
| `f32_runtime_test.rs` | 7/0/0 | f32 运行时（Phase 3） |
| `f32_autodiff_test.rs` | 12/0/0 | f32 自动微分（Phase 4） |
| `f32_wasm_test.rs` | 14/0/0 | f32 WASM 后端（Phase 5.1） |
| `f32_stdlib_test.rs` | 10/0/0 | f32 标准库（Phase 5.4） |
| `f32_parity_test.rs` | 14/0/0 | f32/f64 一致性 |
| `f32_stdlib_parity_test.rs` | 8/0/0 | f32 标准库一致性 |

### 自举验证

| 测试文件 | 数量 | 覆盖 |
|----------|------|------|
| `three_stage.rs` | 1/0/2 | 三段式自举（2 项 ignored — wasmi 慢） |
| `selfhost_frontend.rs` | 栈溢出 | 自举前端验证（lex/parse/lower） |
| `parity_test.rs` | 栈溢出 | VM vs Interpreter 行为一致（全指令覆盖） |

### 一致性

| 测试文件 | 数量 | 覆盖 |
|----------|------|------|
| `native_parity_test.rs` | 35/0/0 | Rust 母编译器 / tenthc 一致性 |

### 标准库

| 测试文件 | 数量 | 覆盖 |
|----------|------|------|
| `stdlib_test.rs` | 114/0/0 | Vec/HashMap/String/Option/文件 I/O/json/toml/runtime 限制 |
| `relation_debugger_test.rs` | 10/0/0 | 关系调试器 |

### 安全与错误

| 测试文件 | 数量 | 覆盖 |
|----------|------|------|
| `memory_test.rs` | 17/0/0 | 内存护栏: arena/limits/计数器 |
| `memory_estimate_test.rs` | 32/0/3 | 编译期内存/算力预估（护城河 D，3 项 ignored） |

### 编译器后端

| 测试文件 | 数量 | 覆盖 |
|----------|------|------|
| `wasm_backend_minimal.rs` | 7/0/0 | WASM 后端最小用例 |
| `jit_test.rs` | 10/0/0 | JIT 编译器回归 |
| `jit_stack_overflow_test.rs` | 栈溢出 | JIT 栈溢出回归 |

### 神经网络

| 测试文件 | 数量 | 覆盖 |
|----------|------|------|
| `mnist_loader_test.rs` | 4/0/0 | MNIST 数据加载 |

### 异步

| 测试文件 | 数量 | 覆盖 |
|----------|------|------|
| `async_basic_test.rs` | 13/0/0 | 异步原语基础 + Phase 2 协程调度 + async I/O（sleep/TCP echo/无效句柄/yield/多任务切换；2026-07-08 从 6 项扩展到 13 项） |

### I/O

| 测试文件 | 数量 | 覆盖 |
|----------|------|------|
| `io_test.rs` | 4/0/0 | 同步 I/O 原语 |
| `net_test.rs` | 6/0/0 | 网络 I/O 原语 |

### 预存失败（不回归）

| 测试文件 | 数量 | 覆盖 |
|----------|------|------|
| `generic_tensor_test.rs` | 2/2/0 | 泛型张量构造函数（2 项预存失败：native_generic_ctor_f32_lowering, native_generic_ctor_f64_lowering） |
| `fixpoint_runtime.rs` | 0/1/0 | fixpoint 端到端编译+执行（1 项预存失败：fixpoint_runtime_benchmark） |

### 栈溢出崩溃（编译通过运行时栈溢出）

| 测试文件 | 数量 | 覆盖 |
|----------|------|------|
| `tenthc_generic_tensor_test.rs` | 栈溢出 | tenthc 泛型张量测试 |
| `tenthc_for_loop_test.rs` | 栈溢出 | tenthc for 循环测试 |
| `tenthc_dotdot_eq_test.rs` | 栈溢出 | tenthc `..=` lexer 测试 |

### 总计

| 测试目标 | 数量 | 说明 |
|----------|------|------|
| **总计** | **791 passed / 3 failed / 12 ignored** | 55 个测试目标（54 文件 + lib），6 个栈溢出崩溃预存问题 |

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
| 5 | three_stage wasmtime 路径 stub host 不完整 | `run_test_wasmtime` 中 `vec_new`/`vec_push`/`vec_len` 等 17 个 host import 均为返回 0 的占位实现，导致 WASM-B 为 0 字节。wasmi 路径已通过（WASM-B 460 bytes，add(3,4)=12）。已将 wasmtime 路径拆为 `#[ignore]` 的 `three_stage_selfhost_wasmtime` 独立测试（2026-07-06），默认 `cargo test` 仅跑 wasmi 路径。补全 stub 需参照 `register_host_functions` 实现 17 个 host import 的 wasmtime 版本，与 f32 路线图无关，登记为独立任务。 |
| 6 | Rust 母编译器 wasm.rs Call 分派缺 str_eq/str_add/str_int 函数调用分支 | `tenth/src/compile/wasm/compile.rs:251-372` 的 `HirExprKind::Call` 分派仅对 `str_len`/`str_at`/`str_cmp`/`str_slice` 有函数调用分支，`str_eq`/`str_add`/`str_int` 缺失。tenthc 源码若以函数调用形式 `str_eq(a,b)` 调用会产出 i64 → i32 类型不匹配的非法 WASM。2026-07-06 f32/f64 parity 路线图阶段 7 发现并规避（tenthc 侧 `parse_tensor_type` 用 `==`/`!=` 操作符走 BinOp 路径）。修复方案：在 `compile.rs` Call 分派里补齐三个分支，与 `str_len`/`str_at`/`str_cmp`/`str_slice` 对齐。 |
| 7 | 6 个测试文件栈溢出崩溃 | tenthc_generic_tensor_test / tenthc_for_loop_test / jit_stack_overflow_test / tenthc_dotdot_eq_test / selfhost_frontend / parity_test 编译通过但运行时触发 Windows STATUS_STACK_OVERFLOW (0xc00000fd)，需增大栈空间或改用 release 模式。**2026-07-08 更新**：`tenthc_for_loop_test` 的 Vec 迭代测试在设置 `RUST_MIN_STACK=268435456`（256MB）后已通过 3/3（vec_literal_basic / vec_literal_break_continue / vec_literal_empty），但默认栈空间下仍栈溢出，根因是 Windows debug 模式栈空间不足（git stash 验证为预存问题）。 |
| 8 | spawn 仍为 eager 语义（Phase 2 设计决策） | `runtime/vm.rs::Op::Spawn` 立即求值 inner 表达式并包装为 `Future(Ready(v))`，不制造真并行；真正的"并发"来自 async I/O 返回的 `Pending` Future（子线程工作期间调度器可切换到其他就绪任务）。**影响**：CPU 密集型 spawn 不会真并行执行；如需 CPU 并行需引入工作窃取调度器或绿色线程池（Phase 3 遗留）。**登记性质**：设计决策非缺陷——保持 Phase 1 兼容性 + 零新依赖原则。详见 MEMO.md 2026-07-08 feat 条目。 |
| 9 | yield 无语法层关键字（Phase 2 已就绪未接入） | VM 已支持 `Op::Yield`(opcode 53) 执行（让出控制权，task 回到 `ready_queue` 尾部），但 lower 阶段没有 `yield` 关键字的 AST→HIR→Op 路径，当前 `Op::Yield` 只能通过手动构造字节码或 native 内部触发。**影响**：用户代码无法直接使用 `yield` 表达式；未来添加 `yield` 关键字时直接接入（lexer + parser + lower 三层）。**登记性质**：能力已就绪未暴露，Phase 3 遗留。详见 MEMO.md 2026-07-08 feat 条目 Phase 3 遗留 (c)。 |

---

## 七、已知缺陷与不完整（历史参考 — C 后端已移除）

### 7.1 自举编译器 Tenth 源码中的已知问题

| # | 位置 | 问题 |
|---|------|------|
| ~~1~~ | ~~`tenthc/lexer/lexer.th:98,105`~~ | ~~**字面量值不解析** — 整数和浮点数 token 的 value 硬编码为 0~~ **已修复**：lexer.th:90 正确解析 `fval: fv + (ff / div)`，lexer.th:105-106 正确存储 `let fv: f64 = ival; ... fval: fv`（f32 后缀路径） |
| ~~2~~ | ~~`tenthc/parser/parser.th:269`~~ | ~~**字段名不存储** — `parse_postfix` 的 `.field` 访问把字段名丢了~~ **已修复**：parser.th:269 实为 `&` ref 处理（`if d == 40`），字段访问在 parser.th:628 `let field_name = field_tok.sval;` 正确存储 |

> ~~原 #1 (cgen 函数调用参数丢弃) 随 codegen 移除而消除。~~
> 2026-06-30 复核：#1/#2 均已修复，条目保留以记录历史。

### 7.2 语言功能缺口

| # | 位置 | 问题 |
|---|------|------|
| ~~3~~ | ~~`tenth/src/compile/lower.rs`~~ | ~~**Match pattern binding 未生成**~~ **已修复**：实现 match 表达式 struct 解构 pattern（如 `match p { Point { x, y } => ... }` 中的 x/y 绑定）。跨模块全链路：(1) `parser/ast.rs` + `hir/hir.rs` 新增 `Pattern::Struct { name, fields }` 变体；(2) `parser/parser.rs::parse_match_pattern` 加 `LBrace` 处理，支持 `Name { x, y }` 简写与 `Name { x: a, y: b }` 命名绑定；(3) `hir/lower/lower_expr.rs` `lower_pattern`/`bind_pattern_vars` 加 Struct 分支；(4) `runtime/interpreter/pattern.rs` 三处加 Struct 分支；(5) `runtime/vm.rs` 新增 `IsStruct(usize)` Op（opcode 46，与 `IsEnumVariant` 对称）；(6) `compile/bytecode.rs` 加 Struct 分支（IsStruct + LoadField）；(7) `compile/jit/translator.rs` 遇 IsStruct 返回 Err 触发 JIT→VM fallback。tenthc 不用 struct pattern，无需同步。测试：`tests/pattern_match_test.rs` 新增 6 项（4 解释器 + 2 VM），17/17 全绿，0 回归。 |
| 8 | `tenthc/main.th:11` | **依赖 tenthc_combined.th** — 需在编译前手动拼接 |

> ~~#5-#7/#10/#12 引用已删除的 C 代码生成文件 (`tenthc/codegen/`, `tenthc/runtime.c`)，随 C 后端移除而消除，不再罗列。~~

### 7.3 中优

| # | 位置 | 问题 |
|---|------|------|
| ~~9~~ | ~~`tenthc/parser/parser.th:631`~~ | ~~**无 for 循环解析** — lexer 识别 `for` 但 parser 未处理~~ **登记陈旧失效**（2026-07-06 复核）：for-in 解析已在 tenthc 全链路实现 — `parser.th:1100-1132`（`parse_stmt` 中 `disc==12` 分支）+ `lower.th:1258-1279`（`kind=="for"` 分支，lower 为 `disc=4`）+ `wasm.th:1404+`（`disc==4` 分支）。原引用行号 `parser.th:631` 实为 generic_call 创建 ident 节点逻辑，与 for 循环无关。tenthc 自举源码本身大量使用 for 循环（parser.th 中 13+ 处），自举路径 B/C 必须能编译 for 循环——这是功能可用的铁证。 |
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
- **tenthpm 包管理器** — `tenth/tools/tenthpm/` **完整实现**，支持 init/build/test/run/add/remove/list/clean/publish/install + Tenth.toml + Tenth.lock + .tenthpkg 打包 + path/git/registry 三种依赖类型
- **LSP 服务器** — `tenth/tools/lsp/` **完整实现**（文档同步/diagnostics/hover/completion/definition/documentSymbol/references/rename/signatureHelp/foldingRange/semanticTokens/formatting）
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

---

## 十一、论文披露的中等价值问题登记（2026-07-02）

> 本章节登记形式化分析论文与相关评审中披露的、尚未在历史章节登记的中等价值已知缺陷与健全性破口。每条含编号、标题、论文来源、影响、当前状态。**本批次为纯文档登记，未修改任何代码**——属于"诚实披露 → 缺陷登记"环节，修复方案分批推进。

### 11.1 形式化健全性缺陷（护城河相关）

| 编号 | 标题 | 论文来源 | 影响 | 当前状态 |
|------|------|---------|------|---------|
| AUDIT-11.1.1 | T19 定理 B6 跨语句借用 unsoundness | T19（形式化分析定理 B6） | `hir/lower/lower_stmt.rs:50/57/73` 与 `lower_expr.rs:437/439/441/464/656` 调用 `scope.release_borrows` 在语句边界过早释放借用；`lower_expr.rs:701/717` Deref 路径不检查别名。`scope.rs:67/81/113` 实现仅做单点 SharedRef/ExclusiveRef 检查，未跟踪跨语句别名。**存在 Tenth 接受但违反别名规则的程序**（如 `let r = &x; let m = &mut x; println(r);` 类构造）。 | ✅ 已修复（2026-07-07，`hir/lower/scope.rs` 新增 `borrow_holders: HashMap<String, Vec<String>>` 跟踪 holder→[borrowed] 关系；`release_borrows` 在释放前检查活跃 holder，保护对应 borrowed 变量不被过早释放；`hir/lower/lower_stmt.rs` Let 处理中记录 holder→borrowed（init 是 Ref/MutRef(Ident) 时）。验证：cargo check 通过 + lib 测试 16 passed（含 5 autodiff）+ integration_test 14 passed + 自举验证通过（`[OK] Full compiler compiled to tenthc_full.wasm`）。详见 MEMO.md 2026-07-07 fix 条目） |
| AUDIT-11.1.2 | T20 PB2 四类反例未覆盖 | T20（property-based testing 反例集） | Call/If/Block/Match 初始化表达式中的 Ref/MutRef 未跳过 `release_borrows`，四类 AST 节点的借用释放语义与 Let 不一致。属 PB2 反例集披露的 4 类漏洞，可能让通过 borrow checker 的程序在运行时仍违反别名。 | ✅ 已修复（2026-07-07，`hir/lower/mod.rs` 新增 `expr_may_produce_ref` 递归判断 If/Block/Match/Move 是否产生 Ref 值；`creates_persistent_borrow` 改用此函数；新增 `collect_persistent_borrowed_idents` 收集多分支中的所有被借变量；`hir/lower/lower_stmt.rs` Let 处理用新函数支持多变量借用记录。验证：cargo check 通过 + lib 测试 16 passed（含 5 autodiff）+ integration_test 14 passed + 自举验证通过（`[OK] Full compiler compiled to tenthc_full.wasm`）。详见 MEMO.md 2026-07-07 fix 条目） |
| AUDIT-11.1.3 | T42 LayerNorm per-feature γ + BatchNorm 多 channel dX 闭式 backward 缺陷 | T42（normalization 算子梯度推导） | LayerNorm 在 per-feature γ（每个特征独立缩放参数）场景下闭式 backward 不正确；BatchNorm 在多 channel dX 场景下闭式 backward 不正确。两者均属"已实现但有数值错误"——前向 ✅，反向在常见配置下 ✅，在论文 T42 披露的特定配置下产生错误梯度。 | ✅ 已修复（2026-07-07，`runtime/autodiff.rs` line 852-1015：(1) LayerNorm backward 把 `dX = γ·std_inv·(dY - mean(dY) - x̂·mean(dY·x̂))` 改为 `dX = std_inv·(gy - mean(gy) - x̂·mean(gy·x̂))`，其中 `gy = dY·γ`（γ 不能提到括号外）；(2) BatchNorm 重写为 per-channel 双遍循环，修正 d_gamma/d_beta shape（从 (N,C,H,W) 改为 (C,)）、mean 改为 per-channel、修正广播错误。验证：cargo check 通过 + autodiff 5/5 + lib 测试 16 passed + integration_test 14 passed + 自举验证通过（`[OK] Full compiler compiled to tenthc_full.wasm`）。详见 MEMO.md 2026-07-07 fix 条目） |
| AUDIT-11.1.4 | T50 multihead_attention 实为 single-head 等价 | T50（标准库 API 与论文承诺一致性审计） | `tenth/std/nn/multihead_attention.th` 文件头部注释自承："Simplified...single-head-equivalent attention...True multi-head attention requires either: 1. 3D/batched matmul support, or 2. A loop over heads with per-head slices"。`n_heads` 参数被读取但仅用于计算 `d_k = d_model / n_heads` 后做单次 attention，未真正分头。**与对外 API 名 MultiheadAttention 不符**。根因：受仅 2D matmul 限制（见 AUDIT-11.1.5）。 | ✅ 已修复。`tenth/std/nn/multihead_attention.th` 已基于 bmm 重写为真正分头计算版本：q/k/v 投影后 reshape 为 (n_heads, seq_len, d_k)，scores = q_3d.bmm(k_3d.transpose()) * scale，out_3d = weights.bmm(v_3d)，最后 reshape 回 (seq_len, d_model) 做 w_o 投影。Reshape/MaskedFill 已接入 autodiff（TapeOp::Reshape/TapeOp::MaskedFill），w_q/w_k/w_v/x 梯度完整传播。`tenth/tests/multihead_attention_test.rs` 11 项测试覆盖前向 shape/有限性/n_heads=1 等价/autodiff 无错误/w_q/w_k/w_v 梯度非零/x 梯度传播/完整反向链。详见 MEMO.md 2026-07-06 test: MHA 完整梯度测试补强 条目。 |
| AUDIT-11.1.5 | T18 body 直接 clone 健全性破口（泛型实例化） | T18（泛型实例化健全性） | 泛型函数实例化时 body 直接 clone 但未做替换健全性检查，可能让类型变量在 body 内未被一致替换，导致实例化后语义偏移。与 T12 双侧破口（tenthc 缺 shape 检查）联合暴露。 | ✅ 已修复（2026-07-07，`hir/lower/mod.rs` 新增 `substitute_expr` 递归替换 HirExpr 中所有 Type 字段；新增 `check_generic_instantiation_soundness` 健全性检查；`hir/lower/lower_expr.rs` GenericCall 实例化路径调用这两个函数，确保 body 内类型变量被一致替换。验证：cargo check 通过 + lib 测试 16 passed（含 5 autodiff）+ integration_test 14 passed + 自举验证通过（`[OK] Full compiler compiled to tenthc_full.wasm`）。详见 MEMO.md 2026-07-07 fix 条目） |

### 11.2 自举编译器双侧破口（T12）

| 编号 | 标题 | 论文来源 | 影响 | 当前状态 |
|------|------|---------|------|---------|
| AUDIT-11.2.1 | tenthc 缺 shape 检查 | T12（自举双侧对齐审计） | tenthc 完全无 shape 检查，Rust 母编译器已有 Phase 1+2+3 实现。两侧语义不对齐：在 tenthc 自举路径下无法享受 shape 检查带来的早期错误发现，且对 shape 不匹配的程序两侧行为可能不一致。 | ✅ 已修复（2026-07-02，`tenthc/hir/hir.th` HirType 新增 dim_count/dim0/dim1 三字段，`tenthc/hir/lower.th` 新增 8 个 shape 辅助函数与 3 处调用点检查（binary/call/method_call）；最小子集：matmul 2D 内侧 K 检查、二元广播检查、zeros/ones/rand/randn 字面量 shape 推断、transpose 2D 维度互换；跳过 let 注解/分支兼容/返回值合并（需 Tensor 注解解析，tenthc 不支持）；错误处理非致命（push 到 p.errors）；首轮 parity 全挂后重构 tuple 返回为 3 个单值 getter（WASM 后端 tuple 支持有限）；详见 MEMO.md 2026-07-02 fix 条目） |
| AUDIT-11.2.2 | tenthc 缺错误恢复 (panic mode) | T12 | tenthc 完全无 panic-mode，遇首个解析错误即终止。Rust 母编译器有 `error_recovery_test.rs` 7 项测试覆盖续接能力。两侧错误恢复语义不对齐。 | ✅ 已修复（2026-07-02，`tenthc/parser/parser.th` 新增 is_sync_token/synchronize/record_error_and_recover/is_expr_start_token，Parser 结构加 errors 字段，parse_program 主循环对无法识别的顶层 token 应用 panic-mode 恢复；error_recovery_test 7/7 通过，详见 MEMO.md 2026-07-02 fix 条目） |
| AUDIT-11.2.3 | tenthc 缺 expect_gt，嵌套泛型 `>>` 被整个吞掉 | T12 | tenthc lexer/parser 缺 `expect_gt`，把 `>>` 整个吞掉，无法解析 `Vec<HashMap<K, V>>`、`Pair<Pair<T, U>, V>` 等嵌套泛型。Rust 母编译器有 `generic_test.rs` 11 项含 `>>` 拆分测试。 | ✅ 已修复（2026-07-02，`tenthc/parser/parser.th` 新增 expect_gt 函数（遇 Gt consume / 遇 Shr consume 后插入合成 Gt，通过重建 tokens 数组实现 Vec::insert 语义），替换 parse_postfix 与 parse_generic_params 2 处闭合调用点，详见 MEMO.md 2026-07-02 fix 条目） |
| AUDIT-11.2.4 | tenthc 缺 i64 溢出检测 | T12 | tenthc 字面量与算术路径缺 i64 溢出检测，Rust 母编译器有 `overflow-checks = true`（M-3 安全修复）。两侧数值健壮性不对齐。 | ✅ 已修复（2026-07-02，`tenthc/lexer/lexer.th` parse_int 与 lexer_next 数字解析加入溢出检测，用数学等价条件 `ival > (i64_max - d) / 10` 检测，溢出时返回 i64_max 饱和值并打印错误，详见 MEMO.md 2026-07-02 fix 条目） |

### 11.3 双重 native 注册反模式（T37）

| 编号 | 标题 | 论文来源 | 影响 | 当前状态 |
|------|------|---------|------|---------|
| AUDIT-11.3.1 | VM 缺 17 项 native 函数 | T37（VM/解释器 native 注册对齐审计） | `main.rs::register_natives()` 注册到 VM 的 native 函数集合缺 17 项：`to_string`、`type_name`、`with_step_limit`、`with_timeout_ms`、`is_timeout`、`start_grad`、`f64_bits`、`f64_from_bits`、`sin`、`cos`、`ln`、`pow`、`save_weights`、`load_weights`、`format`、`parse_int`、`parse_float`。**`save_weights`/`load_weights` 为致命项**——VM 路径下模型保存/加载失效，与 2026-07-01 "护城河 D 演示补齐" 修复的 6 个构造函数（zeros/ones/rand 等）属同类问题的延续。 | ✅ 已修复（2026-07-02，`main.rs::register_natives()` 补全 17 项，save_weights/load_weights 复用 FsSandbox 校验，详见 MEMO.md 2026-07-02 fix 条目） |
| AUDIT-11.3.2 | 解释器缺 3 项 native 函数 | T37 | `interpreter/natives.rs` 注册集合缺 3 项：`print`、`to_f64`、`to_f32`。与 VM 路径不对称。 | ✅ 已修复（2026-07-02，`interpreter/natives.rs` 补全 print/to_f64/to_f32，并同步 HIR lower 白名单与解释器 Var fallback 白名单，详见 MEMO.md 2026-07-02 fix 条目） |

### 11.4 工程债务与潜在 UB

| 编号 | 标题 | 论文来源 | 影响 | 当前状态 |
|------|------|---------|------|---------|
| AUDIT-11.4.1 | T22 FV5 O(n²) 工程债务 | T22（闭包自由变量收集复杂度） | `hir/lower/closures.rs:9-15` 的 `free_vars_in` 用 `Vec<String>` + `sort` + `dedup` 实现，复杂度 O(n²)（每次 push 后 sort）。HashSet 实现可降到 O(n)。在深层嵌套闭包场景下编译期开销显著。 | ✅ 已修复（2026-07-06，`hir/lower/closures.rs` `free_vars_in`/`collect_free_vars`/`collect_free_vars_stmt` 改用 `HashSet<String>`，4 处 `push` 改 `insert`；API 兼容（仍返回 `Vec<String>` + `sort`），复杂度 O(n²)→O(n)，纯优化无语义影响。验证：autodiff 52/52 + 全量 0 回归。详见 MEMO.md 2026-07-06 fix 条目） |
| AUDIT-11.4.2 | T31 MAX_STACK_DEPTH=256 静默溢出潜在 UB | T31（VM/JIT 栈深度限制审计） | `compile/jit/translator.rs:32` `const MAX_STACK_DEPTH: u32 = 256;`，超过时静默截断而非报错，可能让 JIT 编译出的代码访问越界栈区域。属潜在未定义行为。VM 路径（`runtime/vm.rs`）的 `locals: Vec<Value>` 用 `resize` 安全增长，但 JIT 路径的固定 256 槽是硬上限。 | ✅ 已修复（2026-07-06，`compile/jit/translator.rs` 新增 `bump_sp()` 方法在 push 前检查 sp 上限，28 处 push 操作改为 `bump_sp()?`；超限返回 `Err` 触发既有的 JIT→VM fallback（`mod.rs:62-65`），不再静默截断。验证：jit_test 10/10 + 全量 0 回归。详见 MEMO.md 2026-07-06 fix 条目） |
| AUDIT-11.4.3 | T36 双重存储同步开销 | T36（解释器 scope 数据结构审计） | `runtime/interpreter/mod.rs:40` `pub scopes: Vec<HashMap<String, Value>>` 用 Vec+HashMap 双重存储，变量查找从最后一个 scope 向前遍历，在重嵌套场景下有平方风险。VM 路径用索引 `locals: Vec<Value>` 无此问题。 | ✅ 已修复（2026-07-08）。改为扁平化 `vars: HashMap<String, Vec<(usize, Value)>>` + `scope_depth` + `scope_vars: Vec<Vec<String>>`，`resolve_var`/`set_var` O(1)，`pop_scope` O(m)（m 为本层变量数） |
| AUDIT-11.4.4 | tenthc `..=` lexer 解析 bug | 回归测试发现（2026-07-06） | `tenthc/lexer/lexer.th:180-184` 解析 `.` 时 `if next == "."` 分支直接返回 `DotDot`，未检查第三个字符是否为 `=`。导致 `2..=4` 被错误分词为 `2` `..` `=` `4`。tenthc 路径下 inclusive range 实际不可用。 | ✅ 已修复（2026-07-06，`tenthc/lexer/lexer.th:180-190` 在 `if next == "."` 分支内 advance 后再 peek 一次检查 `=`，匹配则 advance 并返回 `DotDotEq`，与 Rust 侧 `lexer.rs` 对齐。验证：自举 + parity 129/129 + selfhost_frontend 4/4 + error_recovery 7/7；新增 `tenthc_dotdot_eq_test.rs` 3 项测试。详见 MEMO.md 2026-07-06 fix 条目） |
| AUDIT-11.4.5 | tenthc Vec 迭代未实现 | 回归测试发现（2026-07-06） | `tenthc/compile/wasm.th:1472` 明确注释 `// Non-range iterable: no-op for now`。tenthc WASM 后端不支持 `for x in [Vec]` 形式的迭代，仅支持 Range 迭代。 | ✅ 已修复（2026-07-06 codegen + 2026-07-08 测试 stub 补全。`tenthc/compile/wasm.th:1471-1546` For 语句新增 ArrayLiteral (disc=22) 分支，复用现有 `vec_new`/`vec_len`/`vec_push`/`vec_get` host imports 实现 Vec 迭代 codegen；后续修复：删除 `wasm.th:1492` 误加的 `wasm_drop`（type/drop 不匹配导致 wasmi 验证失败）。2026-07-08 补全 `tenth/tests/tenthc_for_loop_test.rs` 中 `env` 模块的 4 个 Vec host function stub（vec_new/vec_len/vec_push/vec_get），基于 WASM 内存 + bump allocator 实现真实 Vec 语义（24 字节 header = cap+len+dp，8 字节/元素，vec_push 含倍增扩容），与 `wasm/host.rs` 的 `Vec_new`/`Vec_push` 实现对齐。取消 2 项 `#[ignore]` 标记（tenthc_for_vec_literal_basic / tenthc_for_vec_literal_break_continue），3 项 Vec 迭代测试全部通过。验证：自举 + parity 129/129 + selfhost_frontend 4/4 + tenthc_for_loop_test Vec 迭代 3/3。ArrayLiteral 表达式本身 codegen 未改。详见 MEMO.md 2026-07-06/2026-07-08 fix 条目） |

### 11.5 登记元数据

| 字段 | 值 |
|------|----|
| 登记日期 | 2026-07-02（第 1 批）/ 2026-07-06（11.4.4/11.4.5 新增） |
| 登记批次 | 第 1 批（纯文档登记，无代码修改）+ 2026-07-06 回归测试发现新增 2 条 |
| 论文来源范围 | T12 / T18 / T19 / T20 / T22 / T31 / T36 / T37 / T42 / T50（共 10 个论文条目）+ 回归测试发现 2 条 |
| 总登记条目数 | 16（11.1 ×5、11.2 ×4、11.3 ×2、11.4 ×5） |
| 拆分说明 | T12 双侧破口在 11.2 拆 4 条（shape/panic-mode/expect_gt/i64 溢出）；T37 双重 native 在 11.3 拆 2 条（VM 缺 17 项/解释器缺 3 项）；其余 8 个论文条目各 1 条；2026-07-06 新增 11.4.4（tenthc `..=` lexer bug）和 11.4.5（tenthc Vec 迭代未实现）属回归测试发现 |
| 真理源同步 | `能力梳理/能力全梳理.md` 同步新增能力条目；`MEMO.md` 顶部新增 2026-07-02 与 2026-07-06 变更记录 |
| 修复方案 | 分批推进：第 1 批纯登记；2026-07-06 修复 11.4.1/11.4.2/11.4.4（✅）+ 11.4.5 codegen 修复；2026-07-08 补全 11.4.5 测试 host function stub（✅）；2026-07-08 修复 11.4.3（✅ 解释器 scope 扁平化索引） |

