# 开发备忘

> 各类待办、跳过项、环境依赖、注意事项均记录于此。
>
> **当前阶段：v0.3.0 — C 编译后端已移除，聚焦 Rust 解释器路径**
>
> **2026-06-04 重大变更**：`tenth/src/compile/`（MIR→C 编译管线）、`tenthc/codegen/`、`tenthc/runtime.c` 已删除。
> 原因：生成的 C 代码无内存管理（12 处 malloc / 0 处 free），导致系统级内存耗尽。详见 `SECURITY.md`。
> 自举编译器改为通过 Rust 解释器执行。

## 自举编译器现状

tenthc/ 保留 Tenth 编写的词法分析器和语法分析器（通过 Rust 解释器运行）。
`tenthc/main.th` 已合并全部源码，实现自举管线：Tenth Lexer → Parser → compile_program → WASM。

| 层 | 文件 | 状态 |
|----|------|------|
| Token | `tenthc/lexer/token.th` | ✅ enum TokenKind (50+ 变体) |
| Lexer | `tenthc/lexer/lexer.th` | ✅ 完整词法分析，字面量值正确解析 |
| Parser | `tenthc/parser/parser.th` | ✅ 递归下降 + 优先级爬山 + method_call 支持 |
| ~~Codegen~~ | ~~`tenthc/codegen/cgen.th`~~ | ❌ 已移除 |
| WASM 编译 | `tenth/src/compile/` | ✅ HIR→WASM |
| Bridge | `tenth/src/compile/bridge.rs` | ✅ compact→AST (含 method_call) |

**2026-06 进展：**
- [x] Lexer: Token 新增 fval 字段，FloatLiteral 值不再丢失
- [x] Parser: 修复 method_call — Dot+LParen 产生 "method_call" 节点 (含 receiver)
- [x] Bridge: 新增 "method_call" → ast::MethodCall 转换
- [x] 自举管线闭环：tenthc/main.th 通过 Rust 解释器编译 token.th → WASM (614 bytes)
- [x] 小函数自举验证通过："fn add" 18 tokens → 解析 → WASM 631 bytes

**已知限制：**
- [ ] 树遍历解释器性能：大文件 (~11000 字符) 的自举解析超时，需优化或分批处理
- [ ] WASM 执行验证：generated .wasm 的 wasmi 执行待测试

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
