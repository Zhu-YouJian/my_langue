# Phase 5: AI 全栈 实施计划

> 状态：📋 计划中 | 前置条件：Phase 4 GPU 与性能完成
>
> 对应设计规格 §9 —「AI 全栈」

**Goal:** 实现自动微分（autodiff）、SPMD 并行原语、分布式通信原语，以及 nn / optim / data 第一方标准库，使 Tenth 成为完整的 AI 研究平台。

**Architecture:** 自动微分通过元编程（宏）在 HIR 层实现——编译器原生不懂微积分，只懂张量算子。SPMD 和分布式通信在 MIR 层降级。标准库（nn/optim/data）用 Tenth 自身编写，通过宏和编译器内在获得性能。

**Tech Stack:** Tenth 语言自身（标准库），Rust（编译器侧内在算子），NCCL/MPI（分布式通信）

---

## 文件结构（Phase 5 新增/修改）

```
tenth/std/
├── autodiff/
│   ├── mod.th           # grad, jacobian, vjp, jvp 等宏入口
│   ├── tape.th          # Wengert tape (计算图) 构建
│   ├── backward.th      # 反向传播代码生成
│   └── checkpoint.th    # gradient checkpointing 策略
├── nn/
│   ├── mod.th           # 神经网络层入口
│   ├── linear.th        # Linear (全连接)
│   ├── conv.th          # Conv2D
│   ├── norm.th          # LayerNorm, BatchNorm
│   ├── attention.th     # Multi-Head Attention
│   ├── activation.th    # ReLU, GELU, Swish, Sigmoid
│   └── loss.th          # MSE, CrossEntropy
├── optim/
│   ├── mod.th           # 优化器入口
│   ├── sgd.th           # SGD + Momentum
│   ├── adam.th          # Adam / AdamW
│   └── lr_schedule.th   # 学习率调度 (cosine, step, warmup)
├── data/
│   ├── mod.th           # 数据加载入口
│   ├── loader.th        # DataLoader
│   ├── augment.th       # 数据增强 (翻转, 裁剪, 颜色抖动)
│   └── pipeline.th      # 预处理流水线
├── spmd/
│   └── mod.th           # SPMD 分片原语
├── dist/
│   └── mod.th           # 分布式通信原语
└── tensor/
    └── mod.th           # 张量标准库 (Phase 4 中已有基础)
```

---

### Task 1: 自动微分宏 (autodiff)

**目标:** 通过标准库宏实现自动微分，编译器不需理解微积分。

- [ ] **Wengert Tape（计算图构建）**

`grad` 宏读取函数 HIR，生成前向传播时同时记录每步操作的计算图（tape）：
```
use std::autodiff::grad;

fn f(x: Tensor[f32, B, C]) -> Tensor[f32, B] {
    x.sum() * 2.0
}
// grad(f) 自动生成 df/dx
```

- [ ] **反向传播代码生成**

从 tape 反向遍历，应用每个算子的反向规则（vjp）：
- `+` 的 vjp 是对输入各传一份梯度
- `*` 的 vjp 是另一输入的缩放
- `sum` 的 vjp 是广播
- `matmul` 的 vjp 是转置后的 matmul

- [ ] **Gradient Checkpointing**

支持 `checkpoint(f)` 宏，在前向只保存每 N 层的激活，反向时重新计算中间激活（用计算换显存）。

- [ ] **高阶微分**

支持 `grad(grad(f))` 二阶微分。

---

### Task 2: SPMD 并行原语

**目标:** 提供数据并行、模型并行的声明式语法。

- [ ] **数据并行**

```
shard(batch across [GPU(0), GPU(1)])
fn train_step(model: &Model, batch: Tensor) -> Loss { ... }
```

编译器自动：分片输入 → 每设备独立前向 → all-reduce 梯度 → 每设备同步权重。

- [ ] **模型并行 (Tensor Parallelism)**

```
shard(weight across columns [GPU(0), GPU(1)])
fn large_linear(x: Tensor) -> Tensor { ... }
```

将权重矩阵按列拆分到多 GPU，每设备计算一部分，最后 concatenate。

- [ ] **流水线并行**

```
pipeline([stage_0, stage_1, stage_2])
fn deep_model(x: Tensor) -> Tensor { ... }
```

不同设备处理不同层，微批次流水线调度。

---

### Task 3: 分布式通信

- [ ] **点对点通信**：`send(dest, data)`, `recv(src) -> data`
- [ ] **集合通信**：`all_reduce`, `all_gather`, `reduce_scatter`, `broadcast`
- [ ] **异步消息**：基于 async/await 的非阻塞通信
- [ ] **容错**：节点故障检测与恢复

---

### Task 4: nn 标准库

**目标:** 提供生产级神经网络模块。

- [ ] **基础层**：Linear, Conv2D, Embedding, Dropout
- [ ] **归一化**：LayerNorm, BatchNorm, RMSNorm
- [ ] **注意力**：Multi-Head Attention (标准 Transformer), Flash Attention
- [ ] **激活函数**：ReLU, GELU, Swish, Sigmoid, Tanh
- [ ] **损失函数**：MSE, CrossEntropy, BinaryCrossEntropy

所有层通过 `use std::nn` 导入，纯 Tenth 实现（自动享受编译器融合和 GPU 加速）。

---

### Task 5: optim 与 data 标准库

- [ ] **优化器**：SGD (Momentum, Nesterov), Adam, AdamW, Lion, LAMB
  - 支持 weight decay、gradient clipping
- [ ] **学习率调度**：cosine annealing, step decay, warmup, linear decay
- [ ] **DataLoader**：多线程预取、批处理、shuffle
- [ ] **数据增强**：随机翻转/裁剪/颜色抖动/混合 (Mixup, CutMix)
- [ ] **Pipeline 宏**：声明式数据预处理流水线，编译期优化调度

---

### Task 6: 全量验收

- [ ] `grad(f)(x)` 对简单函数返回正确梯度（与数值微分对比）
- [ ] nn 层（Linear + ReLU + MSE）完整训练循环可运行
- [ ] SPMD 数据并行在 2 GPU 上正确收敛
- [ ] 分布式 all-reduce 在两节点间正确通信

---

## Phase 5 完成标准

- [ ] autodiff：`grad` 宏可用，至少支持逐元素运算 + matmul + reduce 的反向
- [ ] nn 库：至少 Linear, Conv2D, LayerNorm, Attention, ReLU, MSE 可用
- [ ] optim 库：至少 SGD, Adam 可用
- [ ] SPMD：数据并行在 2 GPU 上可运行
- [ ] 全部回归测试通过
