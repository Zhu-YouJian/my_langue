# 开发备忘

> 各类待办、跳过项、环境依赖、注意事项均记录于此。
>
> **当前阶段：v0.3.3 → 阶段 1（可用）**
>
> 演进路线与阶段规划见 `CODE_WIKI.md` §10。
>
> **2026-06-25 架构决策**：自举长期路线图确立。重新审视 WASM 路径与"完全摆脱 Rust"终极目标的契合度，确认 WASM 路径的根本局限（执行运行时 wasmi/wasmtime 是 Rust 写的，"完全摆脱 Rust"在 WASM 路径上不可能达成）。决策采用两阶段策略：阶段 1（当前 spec）WASM 路径达成 C4 固定点，阶段 2（未来 spec）实现 native 后端达成运行时自举。Native 后端三条子路径评估：C2-直接机器码（5000-10000 行 Tenth，无外部依赖，推荐）、C2-LLVM IR（引入 C++ 依赖违背精神，不推荐）、C3-C 翻译（2026-06-04 已尝试并删除，内存管理未解决，重新评估需先解决所有权→C 内存映射）。新增 `docs/superpowers/self-hosting-roadmap.md` 长期路线图文档，spec.md §8 追加架构决策记录。
>
> **2026-06-25 更新**：自举固定点攻关 spec 重构。前置 spec `docs/superpowers/specs/2026-06-20-self-hosting-master-plan/` 已归档（Phase A-D 全部完成，D1-D7 共 129 用例全绿，仅 C4 固定点未达成）。新建 `docs/superpowers/specs/2026-06-25-self-hosting-fixpoint/`（含 spec.md / tasks.md / checklist.md），重新设计 C4 攻关方案：5 阶段架构（运行时迁移→确定性保证→端到端跑通→固定点达成→CI 集成），锁定 4 项设计决策（Wasmtime JIT / wasmi 保留 / 字节级等价不可达成则终止 / 大重构需求则终止），明确 4 项硬性退出条件（不允许降级实现）。同步修正旧 spec 文档不一致：spec.md 能力差距矩阵（Trait/泛型/借用/闭包/Tensor 已实现）、tasks.md D2/D6 验收 checkbox、checklist.md 状态描述。
>
> **2026-06-25 更新**：Phase D — D3（借用检查）+ D5（闭包 WASM 后端）完成，Phase D 全部 7 项（D1-D7）达成。D3：tenthc lower.th 新增 Ownership 状态存储（Owned/SharedRef/ExclusiveRef/Moved）、check_use/check_borrow_shared/check_borrow_mut 借用检查函数、release_borrows 借用释放；parser.th 补全 `&mut` 解析。D5：tenthc lower.th 实现 free_vars_in 递归捕获分析；wasm.th 新增 table/elem section + call_indirect 闭包调用 + env 装箱（tenth_alloc 分配 captures struct）；修复闭包解析双 advance bug（parse_primary 已消费 `|`，闭包块重复 advance 吞掉首个参数）；修复 `&&`/`||` 误用 i64 and/or 指令（WASM 比较结果为 i32，需用 I32And/I32Or）。新增 5 项 parity 测试（D3×3 + D5×2），parity_test 124→129 项全绿，499+ 测试无回归。
>
> **2026-06-25 更新**：Phase D — D2（泛型实例化）+ D6（Tensor WASM 后端）完成（阶段 1 并行）。D2：tenthc 新增泛型参数解析 `<T, U>`、泛型调用前瞻启发式 `looks_like_generic_call`、`substitute_type` 类型替换与 mangled name 实例化（`fn_name_T1_T2`），Rust 侧 lower.rs 同步将 GenericCall 改写为普通 Call。D6：tenthc wasm.th 新增 tensor_from_vec host import (idx 17) 与 TensorLiteral 编译，Rust 侧 wasm.rs 同步；修复 parser.th parse_primary 双 advance bug（tensor 字面量解析失败根因）。修复 three_stage.rs / wasm_backend_minimal.rs linker 缺失 tensor_from_vec 注册导致的回归。新增 4 项 parity 测试（D2×3 + D6×1），parity_test 120→124 项全绿，499+ 测试无回归，自举管道通过。
>
> **2026-06-25 更新**：Phase D — D1（Trait 系统）完成。tenthc 新增 trait 定义/inherent impl 解析与方法静态分派（mangled name `__<Type>_<method>`）；修复 lexer self token 缺少 sval、lowerer self 参数 type_ann 未覆盖为 impl 类型名两处 bug；Rust 母编译器同步实现 inherent impl 方法分派。新增 3 项 D1 parity 测试全部通过，parity_test 117→120 项全绿，499+ 测试无回归。
>
> **2026-06-25 更新**：Phase D — D4（native 函数对齐）完成。修复 tenthc lexer 数字解析双处理 bug（`lexer_peek` 返回当前字符导致 `5`→`55`）；修复 `char_to_ascii` 缺失空格等常见字符（补全 30+ ASCII 字符）。4 项 Slice parity 测试全部通过，parity_test 117 项全绿。
>
> **2026-06-24 更新**：测试数 350→499（新增 parity_test 112 项 VM/Interpreter 一致性、shape_check_test 16 项 shape 检查；selfhost_verify→selfhost_frontend 重构）。文档重复信息全面清理，交叉引用体系建立。
>
> **2026-06-15 更新**：文档维护 — 自动微分算子数修正为 21（新增 LayerNorm/GELU），标准库模块补全（collections/string/utils/nn 扩展），张量方法补全（gelu/layer_norm/cat/masked_fill/permute/broadcast_to/max_val）。
>
> **2026-06-04 重大变更**：`tenth/src/compile/`（MIR→C 编译管线）、`tenthc/codegen/`、`tenthc/runtime.c` 已删除。
> 原因：生成的 C 代码无内存管理（12 处 malloc / 0 处 free），导致系统级内存耗尽。详见 `SECURITY.md`。
> 自举编译器改为通过 Rust 解释器执行。

## 自举编译器现状（2026-06-10 更新）

自举管线全部由 Tenth 实现，全程走 VM（~0.2s），不再依赖解释器 fallback。

### 自举管线

```
Tenth 源码 → Lexer → Parser → Lowerer → WASM Compiler → .wasm → wasmi
   ✅          ✅       ✅         ✅           ✅           ✅      ✅
  (Tenth)   (370行)  (430行)    (294行)      (703行)     import   add(3,4)=7
```

### 性能

| 指标 | 优化前 | 优化后 |
|------|--------|--------|
| 自举管线执行时间 | ~200s (interpreter) | **~0.2s (VM)** |
| VM fallback 率 | Lexer/Parser 100% | **0%** |
| VM 指令数 | 33 | **45** (41→43→45, PushRange+MoveOp+MakeTensor+MakeClosure) |
| wasmi 加载验证 | ❌ | ✅ `add(3,4)=7` |

### 各层状态

| 层 | 文件 | 状态 |
|----|------|------|
| Token | `tenthc/lexer/token.th` | ✅ enum TokenKind (50+ 变体) |
| Lexer | `tenthc/lexer/lexer.th` | ✅ O(1) 源切片，**VM 全速** |
| Parser | `tenthc/parser/parser.th` | ✅ 递归下降 + method_call，**VM 全速** |
| HIR 类型 | `tenthc/hir/hir.th` | ✅ 紧凑表示 (104 行) |
| Lowerer | `tenthc/hir/lower.th` | ✅ AST→HIR 降级 (306 行) |
| WASM 编译器 | `tenthc/compile/wasm.th` | ✅ HIR→WASM + import 段 (703 行) |
| ~~C Codegen~~ | ~~`tenthc/codegen/cgen.th`~~ | ❌ 已移除 |
| **字节码 VM** | `tenth/src/runtime/vm.rs` | ✅ **45 指令** (33→41→43→45) |
| VM 编译器 | `tenth/src/compile/bytecode.rs` | ✅ HIR→bytecode (含 Enum/Match) |

### VM 指令列表（45 条）

```
0-3:   PushInt/Float/Bool/Str    20-23: Lt/Gt/Lte/Gte
4:     PushUnit                  24-26: Jump/JmpFalse/JmpTrue
5-6:   Pop/Dup                   27-28: Call/CallN
7-8:   Load/Store                29:    MethodCall
9-10:  LoadGlobal/StoreGlobal    30:    Ret
11-15: Add/Sub/Mul/Div/Mod       31-32: MakeVec/MakeMap
16-17: Neg/Not                   33:    NewStruct
18-19: Eq/Neq                    34-35: LoadField/StoreField
                                  36:    IndexGet
                                  37:    SliceStr
                                  38:    MakeEnum
                                  39:    IsEnumVariant
                                  40:    EnumGetField
                                  41:    PushRange       ← v0.3.0
                                  42:    MoveOp          ← v0.3.0
                                  43:    MakeTensor      ← v0.3.1
                                  44:    MakeClosure     ← v0.3.1
```

### 新增验证（2026-06-10）

- [x] 路径 C: Tenth Lexer+Parser+Lowerer+WASM 全链路 → wasmi `add(3,4)=7`
- [x] VM Enum/Match 支持：Lexer/Parser 不再 fallback
- [x] 永久回归测试：`tenth/tests/selfhost_verify.rs` (83/83 pass)
- [x] 性能基准：`tenth run tenthc/boot_full.th` → 0.2s

### 已知限制（当前活跃）

> 带编号的完整缺陷清单见 `AUDIT.md` §六/七。此处为逐版演化日志。

- [x] ~~Closure/GenericCall VM fallback~~ → GenericCall/Move/Range/MakeTensor/MakeClosure 已补全
- [ ] Host import (Vec/String) 为占位实现，WASM 模块需宿主提供真实运行时
- [ ] 三段式自举验证（输出 WASM 再编译自身）因栈溢出未跑通
- [x] ~~大文件 Lowerer 性能~~ → 已解决 (VM ~0.2s)

### v0.3.1 新增（2026-06-14）

#### 闭包捕获

| 组件 | 状态 |
|------|------|
| HIR `HirExprKind::Closure` 新增 `captures: Vec<String>` | ✅ |
| Lowerer `free_vars_in()` 递归分析自由变量 | ✅ |
| Interpreter 闭包创建时从 `resolve_var()` 捕获环境变量 | ✅ |
| VM `MakeClosure(params_count, chunk_idx)` 指令 | ✅ |
| BytecodeCompiler 闭包编译为 MakeClosure | ✅ |

#### VM 补全

| 组件 | 状态 |
|------|------|
| `MakeTensor(rows, cols)` 指令 (opcode 43) | ✅ |
| `MakeClosure(params_count, chunk_idx)` 指令 (opcode 44) | ✅ |
| BytecodeCompiler TensorLiteral 编译为 PushFloat+MakeTensor | ✅ |

#### 文件级导入

| 组件 | 状态 |
|------|------|
| Lowerer `search_paths` 字段 | ✅ |
| Lowerer `try_import_file()` / `load_and_compile_file()` | ✅ |
| `source_to_hir()` 使用 `with_search_paths(vec!["std"])` | ✅ |

#### 错误信息增强

| 组件 | 状态 |
|------|------|
| Scope `check_use/check_borrow_shared/check_borrow_mut` 带 span 参数 | ✅ |
| 消除 3 处 `line: 0, col: 0` 硬编码 | ✅ |

#### 标准库补全

| 模块 | 状态 |
|------|------|
| data/dataloader.th — DataLoader 完整实现 | ✅ |
| utils/serialization.th — save_model/load_model/save_checkpoint | ✅ |
| prelude.th — 更新所有模块索引 | ✅ |

#### 自举编译器同步

| 模块 | 状态 |
|------|------|
| hir/hir.th — captures_start/captures_count/range_inclusive 字段 | ✅ |
| hir/lower.th — closure/array/tensor/move/assign_op/deref 降低 | ✅ |
| parser/parser.th — 闭包/数组/张量/move/复合赋值解析 | ✅ |
| compile/wasm.th — disc 22-30 的 WASM 编译处理器 | ✅ |
| lexer/lexer.th — `/* */` 块注释支持（含嵌套） | ✅ |

#### 测试覆盖提升

| 测试文件 | 新增项 | 状态 |
|----------|--------|------|
| autodiff_test.rs | 20 项（autodiff 8 + closure 4 + tensor 7 + error span 1） | ✅ |

#### 示例集

| 变更 | 数量 |
|------|------|
| 新增示例 | 8（归并排序/二叉搜索树/闭包捕获/Softmax回归/Adam优化器/张量广播/词频统计/矩阵转置与运算） |
| 优化示例 | 18（while→for-in, 闭包捕获增强） |
| 总计 | 33 个示例 |

---

### v0.3.3 新增（2026-06-14）

#### GPU 后端脚手架

| 组件 | 状态 |
|------|------|
| `compile/gpu/` — CudaKernel 模板 + Device 抽象 | ✅ 脚手架 |
| `compile/optimizations/` — FusionPass / ParallelPass | ✅ 脚手架 |

#### tenthpm 包管理器

| 组件 | 状态 |
|------|------|
| `tools/tenthpm/` CLI (init/build/test/run/add/remove/list/clean/publish/install) | ✅ **完整实现** |
| Tenth.toml manifest 格式 (含 license 字段) | ✅ |
| 共享引擎模块 (engine.rs) — search_paths + in-process 编译/运行 | ✅ |
| 依赖类型：registry / path / git 三种 | ✅ |
| Tenth.lock 锁文件 (含 checksum) | ✅ |
| .tenthpkg 打包归档 (publish) | ✅ |
| 版本号校验 (X.Y.Z semver) | ✅ |

#### LSP 服务器

| 组件 | 状态 |
|------|------|
| `tools/lsp/` — 诊断/悬停/补全/定义/格式化 handler | ✅ 脚手架 |

#### 标准库补全

| 组件 | 状态 |
|------|------|
| optim/adam_step 实现 | ✅ |
| optim/adagrad_step 实现 | ✅ |
| optim/rmsprop_step 实现 | ✅ |
| nn/batchnorm 函数 | ✅ |
| nn/conv2d 函数 | ✅ |
| nn/embedding 函数 | ✅ |
| init/ 6 个初始化器 (zeros/ones/xavier_uniform/xavier_normal/kaiming_uniform/kaiming_normal) | ✅ |

#### 语言增强

| 组件 | 状态 |
|------|------|
| 结构体字段默认值 — `Expr { kind: "Int", ival: 42, .. }` 语法 | ✅ |
| 泛型返回类型 — `fn f() -> Vec<Token>` 正确解析，修复 `>>` 拆分 | ✅ |
| 枚举元组变体 — `enum TokenKind { IntLiteral(i64), Plus, Eof }` + match 绑定 | ✅ |

#### 测试覆盖提升

| 测试文件 | 变更 | 状态 |
|----------|------|------|
| enum_test.rs | 5→9（+4 枚举元组变体/match 绑定） | ✅ |
| generic_test.rs | 5→11（+6 泛型返回/Vec<Token>/>>拆分） | ✅ |
| struct_test.rs | 5→8（+3 字段默认值/..语法） | ✅ |
| 总计 | 121→499（+378，含 v0.3.1 autodiff + v0.3.3 LSP/tenthpm/类型推断/模式匹配/迭代器/错误恢复/MNIST/parity/shape_check） | ✅ 498 passed + 1 ignored |

---

### v0.3.0 后期新增（2026-06-14）

#### 自动微分

| 组件 | 状态 |
|------|------|
| 张量级 Wengert tape (21 算子) | ✅ |
| Backward 全链路 (DAG 遍历 + broadcast grad) | ✅ |
| 解释器 recording 模式 | ✅ |
| 7 个内置函数 (new_grad/param/backward/grad/stop_grad/zero_grad/cross_entropy) | ✅ |
| Mean/Sum 录制 (返回张量) | ✅ |
| Scalar-tensor 录制 (标量自动包装) | ✅ |

#### 张量运算

| 组件 | 状态 |
|------|------|
| 张量间四则运算 (广播语义) | ✅ |
| MatMul (1D/2D) | ✅ |
| Transpose | ✅ |
| Conv2D (im2col + backward) | ✅ |
| Dropout (inverted + backward) | ✅ |
| Softmax (逐行) | ✅ |
| BatchNorm (backward 就绪, forward 已包装) | ✅ |

#### 标准库

| 模块 | 文件数 | 状态 |
|------|--------|------|
| nn/ | 13 | 全部可运行 (linear/loss/activations/dropout/conv/batchnorm/embedding/attention/multihead_attention/layer_norm/positional_encoding/feedforward/transformer) |
| optim/ | 4 | 全部可运行 (sgd/adam/adagrad/rmsprop) |
| data/ | 1 | DataLoader (new/has_next/next_batch/reset/num_batches) |
| init/ | 1 | 6 个初始化器实现 (zeros/ones/xavier_uniform/xavier_normal/kaiming_uniform/kaiming_normal) |
| collections/ | 2 | iter (map/filter/reduce/zip/enumerate 等), collections (flat_map/partition 等) |
| string/ | 1 | 字符串工具 (join_lines/join_comma/indent/word_wrap/capitalize 等) |
| utils/ | 2 | 序列化 (save_model/load_model/save_checkpoint), math (min/max/clamp/signum 等) |
| math/ | 1 | 数学函数参考 |

#### 语言打磨

| 组件 | 状态 |
|------|------|
| 块注释 /* */ (支持嵌套) | ✅ |
| Vec: pop/set/clear | ✅ |
| String: trim/split/replace/substring/to_upper/to_lower | ✅ |
| REPL 多行输入 (自动续行) | ✅ |
| 错误源码上下文显示 | ✅ |
| 数组字面量 [1,2,3] | ✅ 已原生存在 |
| for-in 循环 (Range/Vec/Tensor) | ✅ |

---

## 跳过项：Phase 4-6 GPU / 分布式 / 生态

> 以下功能因环境限制暂跳，条件就绪后实施。

### Phase 4: GPU 与性能

- [ ] 安装 CUDA Toolkit 12.6（`nvidia-smi` 显示驱动已支持，RTX 4060 8GB）
- [x] ~~实现 CUDA kernel 模板代码生成~~ → `compile/gpu/` 脚手架已就绪 (CudaKernel 模板 + Device 抽象)
- [ ] MIR→CUDA 算子映射 + `tenth compile --target=cuda`
- [x] ~~算子融合~~ → `compile/optimizations/` 脚手架已就绪 (FusionPass/ParallelPass)
- [ ] 自动并行分解 / SPMD 降级（需多 GPU）

### Phase 5: AI 全栈

- [ ] SPMD 并行原语（数据并行、模型并行、流水线并行，需多 GPU）
- [ ] 分布式通信（MPI/NCCL：all_reduce, all_gather, send/recv）
- [ ] Autodiff 进阶：checkpointing、高阶微分、张量级 tape、解释器集成
- [ ] nn 进阶：LayerNorm, Attention, FlashAttention, GELU
- [ ] optim 进阶：AdamW, Lion, LAMB, lr_schedule, gradient clipping
- [ ] data 标准库：DataLoader, 数据增强, pipeline 宏

### Phase 6: 生态与工具

- [x] ~~包管理器 tenthpm~~ → `tools/tenthpm/` **完整实现** (CLI: init/build/test/run/add/remove/list/clean/publish/install + Tenth.toml + Tenth.lock + .tenthpkg 打包 + path/git/registry 依赖)
- [x] ~~LSP 服务器~~ → `tools/lsp/` **完整实现** (文档同步/diagnostics推送/hover/completion/definition/documentSymbol/references/rename/signatureHelp/foldingRange/semanticTokens/formatting)
- [ ] 调试器进阶（断点插桩、调用栈、条件断点）
- [ ] 官网 / 论坛 / RFC 流程 / 贡献指南

---

## 环境配置

> 环境依赖、网络代理、构建/测试命令详见 `DEPS.md`。

---


