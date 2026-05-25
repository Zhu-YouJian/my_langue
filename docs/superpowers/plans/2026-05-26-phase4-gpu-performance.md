# Phase 4: GPU 与性能 实施计划

> 状态：📋 计划中 | 前置条件：Phase 3A + Phase 3B 完成
>
> 对应设计规格 §9 —「GPU 与性能」

**Goal:** 实现 GPU kernel 生成、算子融合优化、区域内存分配器（arena）、自动并行分解，使 Tenth 具备生产级的张量计算性能。

**Architecture:** 在现有 HIR→MIR→C 管线基础上，新增 GPU 后端路径。MIR 层新增设备感知（device placement）和并行分解 pass。Arena allocator 在 MIR 层通过编译期内存分析实现。

**Tech Stack:** Rust 2024 edition，CUDA Toolkit（GPU kernel），openmp/mpi（多核/分布式）

---

## 文件结构（Phase 4 新增/修改）

```
tenth/src/compile/
├── gpu/
│   ├── mod.rs           # GPU 后端入口
│   ├── cuda_kernel.rs   # CUDA kernel 模板生成
│   └── device.rs        # 设备抽象 (CPU/GPU)
├── optimizations/
│   ├── mod.rs           # 优化 pass 框架
│   ├── fusion.rs        # 算子融合 (bias + matmul + relu → fused kernel)
│   └── parallel.rs      # 自动并行分解 (SPMD 降级)
├── arena.rs             # Arena allocator 编译期支持
└── mod.rs               # 修改: compile 管线集成 GPU 路径
```

---

### Task 1: GPU Kernel 生成（CUDA 后端）

**目标:** 将 MIR 中的张量运算编译为 CUDA kernel。

- [ ] **CUDA C 代码模板系统**

定义常见张量算子的 CUDA kernel 模板：
```
// 逐元素加法 kernel 模板
__global__ void add_kernel_f32(const float* a, const float* b, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = a[idx] + b[idx];
}
```

- [ ] **MIR→CUDA 算子映射**

在 MIR 层识别可 GPU 加速的算子（BinaryOp, UnaryOp, matmul, reduce），生成对应 CUDA kernel 调用。

- [ ] **设备放置推断**

自动推断张量所在设备（CPU/GPU），插入必要的 `cudaMemcpy`。编译器对不必要的传输发出警告。

- [ ] **编译 flag**：`tenth compile --target=cuda input.th`

---

### Task 2: 算子融合 (Operator Fusion)

**目标:** 识别常见的算子组合模式并将其融合为单个 kernel，减少显存带宽消耗。

- [ ] **融合模式匹配**

在 MIR 层识别可融合的模式：
- `bias + matmul(x, w) + relu` → 单个 fused kernel
- `x * w + b` (linear) → 单个 fused kernel
- `x + y + z` → 逐个元素连加融合

- [ ] **融合 pass 实现**

遍历 MIR 基本块，检测连续的元素级运算，合并为 FusedOp。

- [ ] **性能验证**：对比融合前后的 kernel launch 次数和显存传输量

---

### Task 3: Arena Allocator（区域内存分配）

**目标:** 训练循环中每个 iteration 产生的临时张量批量分配/释放，避免频繁的 cudaMalloc/cudaFree。

- [ ] **Arena 内存管理器**

实现 arena 分配器，支持 CPU 和 GPU 显存：
```
let arena = Arena::new(Device::GPU(0));
for epoch in 0..num_epochs {
    arena.scope(|| {
        // 此 scope 内所有临时张量从 arena 分配
        let pred = model.forward(&batch);
        let loss = criterion(pred, labels);
    });
    // scope 结束时批量释放
}
```

- [ ] **编译器支持**

在编译期分析临时张量的生命周期，自动插入 arena 分配/释放指令。

- [ ] **测试**：验证 arena 模式下显存用量稳定（无泄漏）

---

### Task 4: 自动并行分解

**目标:** 将 SPMD 分片声明编译为多 GPU 并行代码。

```
shard(batch across [GPU(0), GPU(1)])
fn train_step(model: &Model, batch: Tensor[f32, B, C, H, W]) -> Loss { ... }
```

- [ ] **数据分片生成**：根据 `shard across` 声明，对输入张量按设备数均分
- [ ] **通信代码插入**：在需要跨设备同步处（如 weight update）插入 NCCL all-reduce
- [ ] **设备代码生成**：每个设备生成独立的 kernel launch 序列

---

### Task 5: 全量验收

- [ ] GPU kernel：`tenth compile --target=cuda matmul.th` 生成可运行的 CUDA 代码
- [ ] 算子融合：fused kernel 比非融合版本 kernel launch 次数减少
- [ ] Arena allocator：训练循环中显存稳定，无 OOM
- [ ] 并行分解：SPMD 声明生成的多 GPU 代码逻辑正确

---

## Phase 4 完成标准

- [ ] GPU kernel 生成（至少支持逐元素运算和 matmul）
- [ ] 至少 3 种算子融合模式
- [ ] Arena allocator 可工作
- [ ] SPMD 数据分片声明可编译为多 GPU 代码
- [ ] 全部现有回归测试继续通过
