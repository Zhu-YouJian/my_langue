# 开发备忘

> 各类待办、跳过项、环境依赖、注意事项均记录于此。
>
> **当前阶段：v0.3.0 — 标准库与自举**
> 详见：`docs/superpowers/plans/2026-05-26-v0.3.0-standard-library-and-self-hosting.md`

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

Clash `127.0.0.1:6454`

```cmd
# cargo / curl
set HTTP_PROXY=http://127.0.0.1:6454
set HTTPS_PROXY=http://127.0.0.1:6454

# git（已配好）
git config http.proxy http://127.0.0.1:6454
git config https.proxy http://127.0.0.1:6454
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
cargo run --manifest-path tenth/Cargo.toml -- compile input.th -o output.exe
```
