# 开发备忘

> 各类待办、跳过项、环境依赖、注意事项均记录于此。
>
> **当前阶段：v0.3.0 — C 编译后端已移除，聚焦 Rust 解释器路径**
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
| VM 指令数 | 33 | **40** |
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
| **字节码 VM** | `tenth/src/runtime/vm.rs` | ✅ **40 指令** (33→40) |
| VM 编译器 | `tenth/src/compile/bytecode.rs` | ✅ HIR→bytecode (含 Enum/Match) |

### VM 指令列表（40 条）

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
                                  38:    MakeEnum       ← 新增
                                  39:    IsEnumVariant   ← 新增
                                  40:    EnumGetField    ← 新增
```

### 新增验证（2026-06-10）

- [x] 路径 C: Tenth Lexer+Parser+Lowerer+WASM 全链路 → wasmi `add(3,4)=7`
- [x] VM Enum/Match 支持：Lexer/Parser 不再 fallback
- [x] 永久回归测试：`tenth/tests/selfhost_verify.rs` (83/83 pass)
- [x] 性能基准：`tenth run tenthc/boot_full.th` → 0.2s

### 已知限制

- [ ] Closure/GenericCall 仍在 VM 中 fallback（自举代码未使用）
- [ ] Host import (Vec/String) 为占位实现，WASM 模块需宿主提供真实运行时
- [ ] 三段式自举验证（输出 WASM 再编译自身）因栈溢出未跑通
- [ ] 大文件 Lowerer 性能由解释器瓶颈转为 VM 无关（已解决）

---

## 跳过项：Phase 4-6 GPU / 分布式 / 生态

> 以下功能因环境限制暂跳，条件就绪后实施。

### Phase 4: GPU 与性能

- [ ] 安装 CUDA Toolkit 12.6（`nvidia-smi` 显示驱动已支持，RTX 4060 8GB）
- [ ] 实现 CUDA kernel 模板代码生成（逐元素运算、matmul）
- [ ] MIR→CUDA 算子映射 + `tenth compile --target=cuda`
- [ ] 算子融合（bias+matmul+relu → fused kernel）
- [ ] 自动并行分解 / SPMD 降级（需多 GPU）

### Phase 5: AI 全栈

- [ ] SPMD 并行原语（数据并行、模型并行、流水线并行，需多 GPU）
- [ ] 分布式通信（MPI/NCCL：all_reduce, all_gather, send/recv）
- [ ] Autodiff 进阶：checkpointing、高阶微分、张量级 tape、解释器集成
- [ ] nn 进阶：Conv2D, Embedding, LayerNorm, Attention, FlashAttention, GELU
- [ ] optim 进阶：AdamW, Lion, LAMB, lr_schedule, gradient clipping
- [ ] data 标准库：DataLoader, 数据增强, pipeline 宏

### Phase 6: 生态与工具

- [ ] 包管理器 tenthpm（Tenth.toml, 依赖解析, 锁文件, 注册中心）
- [ ] LSP 服务器（补全、诊断、跳转、悬停、格式化、自动导入）
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
