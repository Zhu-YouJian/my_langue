# Phase 4-6 待完成备忘

> 日期：2026-05-26 | 状态：以下功能因环境/硬件限制暂时跳过，待条件就绪后实施。

---

## Phase 4: GPU 与性能 — 跳过项

### GPU Kernel 生成（需 CUDA Toolkit）
- [ ] 安装 CUDA Toolkit 12.6（`nvidia-smi` 显示驱动已支持 CUDA 12.6）
- [ ] 实现 CUDA kernel 模板代码生成（逐元素运算、matmul）
- [ ] MIR→CUDA 算子映射
- [ ] 编译 flag：`tenth compile --target=cuda input.th`
- [ ] GPU kernel 端到端测试

### 算子融合 (Operator Fusion)
- [ ] bias + matmul + relu → fused kernel
- [ ] x * w + b (linear) → fused kernel
- [ ] 融合前后性能对比（kernel launch 次数、显存带宽）

### 自动并行分解 (SPMD 降级)
- [ ] `shard(batch across [GPU(0), GPU(1)])` 语法 → 多 GPU 代码生成
- [ ] NCCL all-reduce 通信代码插入
- [ ] 多 GPU 环境（至少 2 块 GPU）测试

---

## Phase 5: AI 全栈 — 跳过项

### SPMD 并行原语（需多 GPU）
- [ ] 数据并行（batch 分片）
- [ ] 模型并行（Tensor Parallelism）
- [ ] 流水线并行（Pipeline Parallelism）

### 分布式通信（需 MPI/NCCL）
- [ ] 点对点通信：`send(dest, data)`, `recv(src) -> data`
- [ ] 集合通信：`all_reduce`, `all_gather`, `reduce_scatter`, `broadcast`
- [ ] 异步消息：基于 async/await 的非阻塞通信
- [ ] 容错：节点故障检测与恢复

### Autodiff 进阶
- [ ] Gradient Checkpointing（`checkpoint(f)` 宏）
- [ ] 高阶微分（`grad(grad(f))` 二阶微分）
- [ ] 张量级 autodiff（当前仅标量级 tape）
- [ ] 与解释器集成（在 Tenth 源码中标注 `#[autodiff]` 自动生成反向）

### nn 标准库进阶
- [ ] Conv2D、Embedding、Dropout、LayerNorm、BatchNorm
- [ ] Multi-Head Attention（标准 Transformer）、Flash Attention
- [ ] GELU、Swish、Sigmoid、Tanh 激活函数
- [ ] CrossEntropy、BinaryCrossEntropy 损失函数

### optim 标准库进阶
- [ ] SGD Momentum / Nesterov
- [ ] AdamW、Lion、LAMB 优化器
- [ ] 学习率调度（cosine annealing、step decay、warmup、linear decay）
- [ ] Weight decay、gradient clipping

### data 标准库（未开始）
- [ ] DataLoader（多线程预取、批处理、shuffle）
- [ ] 数据增强（随机翻转/裁剪/颜色抖动）
- [ ] Mixup、CutMix
- [ ] Pipeline 宏（声明式数据预处理流水线）

---

## Phase 6: 生态与工具 — 跳过项

### 包管理器 (tenthpm)
- [ ] 包清单解析（Tenth.toml）
- [ ] 依赖解析（语义化版本约束求解，类似 Cargo）
- [ ] 锁文件（Tenth.lock）可重现构建
- [ ] 中央包注册中心（类似 crates.io）
- [ ] CLI：`tenthpm new/build/test/run/add/publish`

### LSP 服务器（需 VS Code 集成调试）
- [ ] 实时诊断（类型错误、借用冲突）
- [ ] 代码补全（变量/函数/方法/trait）
- [ ] 类型悬停（Hover）
- [ ] 跳转定义（Go-to-Definition）
- [ ] 代码格式化（AST-aware formatter）
- [ ] 自动导入（自动插入 `use` 语句）

### 调试器进阶
- [ ] 解释器断点模式（`eval_expr` 中插桩）
- [ ] 变量查看（暂停时显示作用域内所有变量）
- [ ] 调用栈显示
- [ ] 条件断点（`:break if x > 10`）

### 社区与基础设施
- [ ] 官网（tenth-lang.org）
- [ ] 论坛 / Discord 社区
- [ ] RFC 流程（语言变更提案）
- [ ] 贡献指南
- [ ] `rustc --explain` 风格的在线错误码查询

---

## 环境依赖速查

| 依赖 | 当前状态 | 安装方式 |
|------|----------|----------|
| CUDA Toolkit 12.6 | ❌ 未安装 | https://developer.nvidia.com/cuda-downloads |
| NCCL | ❌ 未安装 | 随 CUDA Toolkit 分发 |
| MPI | ❌ 未安装 | MS-MPI (Windows) / OpenMPI (Linux) |
| 多 GPU 硬件 | ❌ 单 GPU (RTX 4060) | 需第二块 GPU |
| VS Code | ✅ 已安装 | 用于 LSP 调试 |
