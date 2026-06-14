# 开发备忘

> 各类待办、跳过项、环境依赖、注意事项均记录于此。
>
> **当前阶段：v0.3.3 — GPU 脚手架 + 包管理器 + LSP + 语言增强**
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

### 已知限制

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
| `tools/tenthpm/` CLI (init/build/test/run/add/publish/install) | ✅ 脚手架 |
| Tenth.toml manifest 格式 | ✅ |

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
| 总计 | 121→134（+13） | ✅ 133 passed + 1 ignored |

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

- [x] ~~包管理器 tenthpm~~ → `tools/tenthpm/` 脚手架已就绪 (CLI: init/build/test/run/add/publish/install + Tenth.toml)
- [x] ~~LSP 服务器~~ → `tools/lsp/` 脚手架已就绪 (诊断/悬停/补全/定义/格式化 handler)
- [ ] 调试器进阶（断点插桩、调用栈、条件断点）
- [ ] 官网 / 论坛 / RFC 流程 / 贡献指南

---

## 环境依赖速查

| 依赖 | 状态 | 路径 |
|------|------|------|
| Rust | ✅ 1.95.0 | `C:\Users\史蒂夫\.cargo\bin\` |
| GCC | ✅ 15.2.0 | `D:\msys64\mingw64\bin\gcc.exe` |
| Git | ✅ | 已配 github 代理 |
| CUDA Toolkit 12.6 | ❌ | https://developer.nvidia.com/cuda-downloads |
| NCCL | ❌ | 随 CUDA Toolkit |
| MPI | ❌ | MS-MPI / OpenMPI |

---

## 网络代理

Clash `127.0.0.1:7892`

```cmd
# cargo / curl
set HTTP_PROXY=http://127.0.0.1:7892
set HTTPS_PROXY=http://127.0.0.1:7892

# git（已配好）
git config http.proxy http://127.0.0.1:7892
git config https.proxy http://127.0.0.1:7892
```

---

## 测试命令

```bash
# 全量
cargo test --manifest-path tenth/Cargo.toml

# 单项
cargo test --manifest-path tenth/Cargo.toml -- autodiff

# 编译运行
cargo run --manifest-path tenth/Cargo.toml
```
