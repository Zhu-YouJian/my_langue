# Tenth 语言设计规格

> 代号 Tenth = Tensor + Zenith，意为「张量之巅」
>
> 状态：原型实作阶段 (Phase 1-3B 完成) | 日期：2026-05-22
>
> 实现进度：Lexer/Parser/HIR/Interpreter 全管线可运行；泛型/trait/所有权/借用检查 已实现；MIR→C 编译管线已打通。60 项测试全部通过。

---

## 一、语言概要 & 设计哲学

### 一句话定位

Tenth 是一门为 AI 研究而生的通用静态语言 —— 张量和梯度是语言的内在概念，编译器理解你在计算什么并为你优化，同时不剥夺你对硬件每一寸的控制力。

### 四条设计铁律

**1. 张量是原生公民，不是库的附庸。**
数组操作不依赖任何外部库。`a + b` 天然支持任意维度的广播。编译器理解张量语义，能做融合、布局优化、自动并行分解。

**2. 编译期知道越多，运行时崩溃越少。**
强静态类型 + shape 感知。矩阵乘法维度不匹配在保存文件时就报错，而不是在训练第 5000 步才崩溃。类型系统是开发者的盟友，不是枷锁。

**3. 高层如写伪码，低层如写裸金属。**
用压缩的语法写前向传播，编译器生成优化的 CUDA kernel。但你随时可以下钻，精确控制内存布局、分配策略和同步粒度。抽象不遮蔽现实。

**4. 核心足够纯净，边界足够开放。**
编译器原生只理解张量运算和 shape 约束。神经网络层、优化器、自动微分 —— 这些通过元编程构建在编译器原语之上。语言不会过时，因为你可以在语言之上造任何新范式。

### 核心决策总结

| 维度 | 决策 |
|------|------|
| 定位 | AI/ML 研究语言，超越现有语言而非仅针对 Python |
| 野心 | 生产竞争级（C），创新是实现手段 |
| 痛点权重 | 70% 张量/autodiff 原生化 + 30% 性能体验统一 |
| 范式 | 多范式，通用高级语言能力不打折扣 |
| 生态 | 完全独立，从零构建 |
| 类型系统 | 强静态 + 优秀推断 + shape 感知（取 Rust 精神 + 部分依赖类型精华） |
| 内存管理 | 所有权 + 借用，ML 特化优化 |
| 编译策略 | MLIR 多层 IR |
| 并发模型 | 多层次显式并行（B） |
| 语法风格 | C 系简洁风 |
| 自动微分 | 标准库通过元编程/宏实现 |

---

## 二、类型系统

### 2.1 基础类型

```
i8, i16, i32, i64        // 有符号整数
u8, u16, u32, u64        // 无符号整数
f16, f32, f64, bf16      // 浮点数（含 AI 领域常用的 bf16）
bool, char, str           // 基本值类型
```

### 2.2 张量类型 —— 核心创新

张量的 shape 是类型的一部分，编译器在编译期做维度校验：

```
Tensor[f32, 3, 224, 224]        // 具体 shape：一个 3×224×224 的 f32 张量
Tensor[f32, ..]                  // 任意 rank，任意维度
Tensor[f32, B, C, H, W]          // 符号维度：B/C/H/W 是类型变量，编译期推导
Tensor[f32, B, 3, H, W]          // 混合：部分维度已知，部分符号化
```

**符号维度（symbolic dimensions）** 是该维的具体值在编译期未知、但可以通过类型变量追踪的维度。在函数签名中声明后，编译器保证所有调用处的维度兼容性：

```
fn matmul(a: Tensor[f32, M, K], b: Tensor[f32, K, N]) -> Tensor[f32, M, N] {
    // M、K、N 是符号维度
    // 编译器保证：a 的列数 == b 的行数（都等于 K）
    // 返回值的行数 = M，列数 = N
}
```

**Rank polymorphism**：当不需要约束具体维度时，用通配符让函数适用于任意 rank 的张量：

```
fn sum(x: Tensor[f32, ..]) -> Tensor[f32, []]      // 任意 rank → 标量
fn batch_norm(x: Tensor[f32, N, C, ..]) -> Tensor[f32, N, C, ..]
```

### 2.3 类型推断策略

- **函数签名**：必须显式标注（契约不能靠猜）
- **局部变量**：完全推断（`let x = a + b` 无需写类型）
- **泛型参数**：调用处推断（`matmul(a, b)` 自动推导 M, K, N）

### 2.4 Trait 系统

类似 Rust trait / Haskell typeclass，作为类型能力的契约：

```
trait Add {
    fn add(self, other: Self) -> Self;
}

trait Diffable {
    fn grad(self) -> Self;
}
```

标准库通过 trait 定义张量操作，编译器识别这些 trait 并在 IR 层做原生级优化。

### 2.5 代数数据类型（ADT）

```
enum Option[T] {
    Some(T),
    None,
}

enum Layer {
    Linear { in_dim: u32, out_dim: u32 },
    Conv2D { kernel: u32, stride: u32 },
    ReLU,
}
```

---

## 三、所有权与内存管理

### 3.1 基础所有权规则

继承 Rust 精神的核心规则：

```
let x = tensor.rand([B, C, H, W]);   // x 拥有这个张量
let y = x;                            // 所有权转移给 y，此后 x 失效
let z = &x;                           // 不可变借用
let w = &mut x;                       // 可变借用（独占）
```

### 3.2 ML 特化优化

**默认引用计数。** 这是与 Rust 的关键分歧。在 ML 训练中，同一份权重和中间激活常被多个函数同时持有引用，严格的 borrow checker 在探索性代码中是阻碍。Tenth 默认使用引用计数，灵活但有小量开销。在性能关键路径上可以声明式切换为独占所有权：

```
// 默认模式：引用计数，灵活
let weights = tensor.rand([1024, 512]);

// 性能路径：声明独占所有权，零开销
// （具体语法待定 - 见开放问题）
```

**区域/竞技场内存。** 训练循环中，每个 iteration 产生大量临时张量（中间激活、梯度），它们的生命周期恰好等于一个 iteration。Tenth 提供内置 arena allocator，一个 iteration 内分配的所有临时内存在该 iteration 结束时批量释放：

```
let arena = Arena::new();
for epoch in 0..num_epochs {
    arena.scope(|| {
        let batch = dataloader.next();
        let pred = model.forward(&batch);
        let loss = criterion(pred, batch.labels);
        loss.backward();
        optimizer.step();
    });
    // 该 iteration 的所有临时张量在此处批量释放
}
```

### 3.3 GPU 显存管理

GPU 显存是 AI 训练的真正瓶颈，Tenth 将其提升为语言概念：

```
let gpu_buf = tensor.rand([4096, 4096]).to_device(GPU(0));
// gpu_buf 的生命周期受所有权系统管理
// 变量离开作用域时自动释放 GPU 显存
```

编译器能静态追踪张量所在的设备，对不必要的 CPU↔GPU 数据传输发出警告。

---

## 四、编译架构

### 4.1 四层 IR 流水线

```
源代码
  │
  ▼
┌─────────────────────────────────┐
│  HIR (高层 IR)                   │
│  - 保留张量语义、shape 信息       │
│  - 类型检查、Shape 传播          │
│  - 自动微分变换（宏展开层面）     │
│  - 张量算子融合                  │
└─────────────────────────────────┘
  │
  ▼
┌─────────────────────────────────┐
│  MIR (中层 IR)                   │
│  - 所有权检查、借用验证          │
│  - 并行分解（SPMD 降级）         │
│  - 设备放置决策                  │
│  - 内存布局选择                  │
└─────────────────────────────────┘
  │
  ▼
┌─────────────────────────────────┐
│  LIR (低层 IR)                   │
│  - 循环展开、向量化              │
│  - GPU Kernel 生成               │
│  - 显存分配调度                  │
└─────────────────────────────────┘
  │
  ▼
┌─────────────────────────────────┐
│  LLVM IR / MLIR GPU Dialect     │
│  - LLVM 优化管线                 │
│  - 目标代码生成 (x86, ARM, CUDA) │
└─────────────────────────────────┘
```

### 4.2 各层职责

**HIR —— 语义之层。** 代码刚离开语法树的形态。类型检查、trait 解析、shape 推导全在这里完成。此层还能做算子融合：识别 `bias + matmul(x, w) + relu` 这个模式，标记为可融合，供下层处理。自动微分的宏展开也在此层执行。

**MIR —— 资源之层。** 所有权和借用的静态验证在这里执行。并行原语（SPMD 分片声明）在此层降级为具体的数据分布和通信指令。设备放置决策在此层做出 —— 哪些张量放 CPU、哪些放 GPU(0)、哪些需要跨设备复制。

**LIR —— 性能之层。** 做传统编译器优化：循环展开、向量化、内联。GPU kernel 在此层生成并调度。此时代码已经非常接近手写 CUDA kernel 的形态，但没有手写 CUDA 的痛苦。

**LLVM IR / GPU Dialect —— 目标之层。** 对接成熟的 LLVM 生态做最后一程优化和目标代码生成。GPU 路径走 MLIR 的 GPU dialect，映射到 CUDA/ROCm/Vulkan Compute。

### 4.3 编译模式

- **调试模式**：只跑 HIR → 直接解释执行。零编译时间，适合快速原型迭代
- **发布模式**：完整四层管线。首次编译慢，但生成的代码和手写 CUDA 同级
- **增量模式**：修改函数体后只重编译受影响的部分，类似 Rust 的增量编译

---

## 五、并发与并行模型

三层并行，各司其职，开发者显式控制每一层。

### 5.1 单机多核 —— 轻量级任务

类似 Go goroutine 或 Rust tokio task。适合并行数据加载、预处理流水线：

```
spawn task load_and_prefetch(path: str) -> Batch {
    let data = read_file(path);
    let processed = augment(data);
    return processed;
}

let handles = paths.map(|p| spawn task load_and_prefetch(p));
let batches = handles.map(|h| h.await());
```

M:N 调度的轻量级线程，无堆栈协程，零成本 async/await。

### 5.2 单机多 GPU —— SPMD 原语

AI 训练中最常用的并行模式。Tenth 提供显式的数据分片声明：

```
shard(batch across [GPU(0), GPU(1)])
fn train_step(model: &Model, batch: Tensor[f32, B, C, H, W]) -> Loss {
    let pred = model.forward(batch);
    let loss = criterion(pred, batch_labels);
    loss.backward();
    return loss;
}
```

编译器自动生成通信代码（NCCL all-reduce / all-gather），但并行策略由开发者显式决定。

### 5.3 多机分布式 —— 异步消息传递

类似 actor 模型或 MPI。多机场景下网络延迟不可忽略，需要用异步消息避免空等：

```
node(0)
fn coordinator() {
    send(node(1), WorkItem { ... });
    let result = recv(node(1));
}

node(1)
fn worker() {
    let item: WorkItem = recv(node(0));
    let result = process(item);
    send(node(0), result);
}
```

### 5.4 三层协作

三层可以组合。典型的分布式训练：

- 外层：多机消息协调
- 中层：每机内 GPU 间 SPMD 数据并行
- 内层：每个 GPU 内部多核并行数据预处理

语法上三层风格统一，语义上泾渭分明。

---

## 六、元编程与自动微分

### 6.1 元编程模型

采用类似 Rust proc macro 的机制，但更强大。宏在 HIR 层操作语法树，可以生成、变换、删除任意代码。宏本身用 Tenth 编写：

```
macro diff(fn_def: FnDef) -> FnDef {
    // 读取函数的 HIR 表示
    // 生成对应的反向传播计算图
    // 返回变换后的函数定义
    let fwd = fn_def;
    let bwd = generate_backward(fwd);
    return concat(fwd, bwd);
}
```

### 6.2 自动微分的宏实现

开发者使用时只需标注：

```
use std::autodiff;

fn my_model(x: Tensor[f32, B, C]) -> Tensor[f32, B, D] {
    ...
}

let (loss, grads) = grad(my_model)(input);
```

`grad` 是一个标准库宏，它读取 `my_model` 的 HIR、构建计算图（Wengert tape）、生成前向和反向传播代码。编译器在此之后继续执行 MIR/LIR 优化，包括算子融合和 GPU kernel 生成。

### 6.3 为什么是宏而非编译器内置

- 允许社区实现不同的 autodiff 策略（前向模式、反向模式、混合模式）
- 允许实现 checkpointing（重计算换显存）作为标准库特性而非编译器特性
- 允许未来新的微分范式（如隐函数微分）在不修改编译器的情况下加入

### 6.4 张量算子作为编译器内在

虽然 autodiff 在标准库层，但张量基础运算（加减乘除、矩阵乘、卷积、规约等）是编译器内在的。编译器"认识"它们，能在 IR 层做融合和优化。宏生成的代码会回退到这些内在运算，从而获得编译器优化。

### 6.5 元编程的其他用途

- **神经网络层定义**：通过宏将 `Linear`, `Conv2D`, `LayerNorm` 等定义为可组合的模块
- **数据管道**：`pipeline!` 宏将数据加载、预处理、增强声明为编译期优化的流水线
- **序列化/反序列化**：`derive(Serialize)` 宏自动生成模型权重的存储和加载代码
- **算子自定义**：允许用户定义新的张量算子，并通过宏注册到编译器优化管线中

---

## 七、语法概览

### 7.1 基本语法风格

C 系花括号，分号可选（语句以换行或分号终止），受 Rust/Go 影响：

```
// 变量声明
let x: i32 = 42;
let y = 3.14;           // 类型推断

// 可变变量
let mut z = 0;
z = z + 1;

// 函数定义
fn add(a: i32, b: i32) -> i32 {
    a + b               // 最后一个表达式为返回值
}
```

### 7.2 控制流

```
// if 是表达式
let result = if x > 0 { "positive" } else { "non-positive" };

// 模式匹配
match layer {
    Linear { in_dim, out_dim } => in_dim * out_dim,
    Conv2D { kernel, .. } => kernel * kernel,
    ReLU => 0,
}

// for 循环
for i in 0..10 {
    println(i);
}

// while 循环
while loss > threshold {
    loss = train_step(model, batch);
}
```

### 7.3 张量字面量与操作

```
// 张量字面量
let a = tensor[[1.0, 2.0], [3.0, 4.0]];     // 2×2
let b = tensor.ones([B, C, H, W]);
let c = tensor.randn([N, D]);

// 运算（自动广播）
let d = a + 1.0;                   // 标量广播
let e = matmul(a, b);              // 矩阵乘
let f = c.view([N, H, W, C]);     // reshape
let g = c[0..B, :, :, :];         // 切片

// 规约
let mean = x.mean(axis=0);
let total = x.sum();
```

### 7.4 模块与可见性

```
// 模块定义
mod layers {
    pub fn relu(x: Tensor[f32, ..]) -> Tensor[f32, ..] {
        x.maximum(0.0)
    }
}

// 导入
use layers::relu;
```

### 7.5 错误处理

采用 Result 类型而非异常，类似 Rust：

```
fn load_weights(path: str) -> Result<Tensor[f32, ..], IoError> {
    let data = try read_file(path);
    Ok(parse_weights(data))
}

// 使用
match load_weights("model.tenth") {
    Ok(weights) => model.load(weights),
    Err(e) => eprintln("Failed: {}", e),
}
```

---

## 八、标准库设计方向

### 8.1 第一方（随编译器分发）

这些是编译器理解并优化的模块：

- `std::tensor` —— 张量类型与基础运算（编译器内在）
- `std::autodiff` —— 自动微分（宏 + 编译器内在）
- `std::arena` —— 区域内存分配器
- `std::device` —— 设备抽象（CPU, GPU, TPU）
- `std::task` —— 轻量级任务系统
- `std::spmd` —— SPMD 并行原语
- `std::dist` —— 分布式通信

### 8.2 第二方（官方维护，单独发行）

- `nn` —— 神经网络层库（Linear, Conv2D, Attention, LayerNorm 等）
- `optim` —— 优化器（SGD, Adam, AdamW, Lion 等）
- `data` —— 数据加载与预处理管道
- `metrics` —— 评估指标
- `serialize` —— 模型序列化格式（Safetensors 风格）

### 8.3 第三方（社区空间）

- 视觉模型（ViT, ResNet 等）
- 语言模型（Transformer, LLaMA 架构等）
- 扩散模型
- 强化学习
- 科学计算扩展

---

## 九、开发路线图

### Phase 0：设计验证 ✅ 已完成
- 完成设计规格文档
- 收集反馈，迭代修改
- 产出语言参考手册初稿

### Phase 1：最小原型 ✅ 已完成
- Lexer → Parser(AST) → HIR Lowering + Type Check → Tree-walk Interpreter 完整管线
- 基本张量操作、变量绑定、控制流、函数定义、闭包
- 产出：22 项测试通过、可交互 REPL
- 详见：`docs/superpowers/plans/2026-05-22-phase1-bootstrap-compiler.md`

### Phase 2：夯实解释器 ✅ 已完成
- struct/enum/match/impl 完整类型系统
- mod/use 模块系统
- rand/randn/softmax/matmul 张量标准库扩展
- 改进错误诊断（源码片段 + 位置指示器）
- 产出：38 项测试通过，可编写非平凡 Tenth 程序
- 详见：`docs/superpowers/plans/2026-05-22-phase2-interpreter-hardening.md`

### Phase 3A：类型系统深化 ✅ 已完成
- 泛型系统（类型参数、泛型函数与泛型结构体）
- trait 系统（定义、实现、约束边界）
- 所有权与借用检查（引用、移动语义）
- 产出：类型系统完备的解释器，11 项所有权/借用测试通过
- 详见：`docs/superpowers/plans/2026-05-22-phase3a-type-system.md`

### Phase 3B：编译后端 ✅ 已完成
- HIR → MIR → C 编译管线
- shape 推导引擎
- CLI compile 命令（`tenth compile input.th -o output.c`）
- 偏差：以 C 代码生成替代原计划 LLVM 直编，降低 LLVM 安装依赖
- 产出：5 项编译测试通过
- 详见：`docs/superpowers/plans/2026-05-22-phase3b-mlir-compilation.md`

### Phase 4：GPU 与性能 🚧 部分完成
- ✅ Arena allocator — 内存池批量分配/释放
- ✅ MIR 优化 passes — 常量折叠 + 死代码消除
- ✅ 端到端 GCC 编译 — `.th → C → gcc → .exe`
- ❌ GPU kernel 生成 — 待装 CUDA Toolkit
- ❌ 算子融合 — 待 GPU 后端
- ❌ 自动并行分解 — 需多 GPU
- 详见：`docs/superpowers/plans/2026-05-26-phase4-gpu-performance.md`
- 跳过项备忘：`docs/superpowers/plans/2026-05-26-phase4-6-skipped.md`

### Phase 5：AI 全栈 🚧 部分完成
- ✅ Autodiff 基础 — tape + vjp + grad（标量级）
- ✅ nn 标准库原型 — Linear/ReLU/MSE/BCE（Tenth 实现）
- ✅ optim 标准库原型 — SGD/Adam（Tenth 实现）
- ❌ SPMD 并行原语 — 需多 GPU
- ❌ 分布式通信 — 需 MPI/NCCL
- ❌ data 标准库 — 未开始
- 详见：`docs/superpowers/plans/2026-05-26-phase5-ai-fullstack.md`
- 跳过项备忘：`docs/superpowers/plans/2026-05-26-phase4-6-skipped.md`

### Phase 6：生态与工具 🚧 部分完成
- ✅ REPL 调试器 — `:break` `:step` `:print` 命令
- ✅ 文档生成器 — HIR → Markdown API 文档
- ❌ 包管理器 (tenthpm) — 待开发
- ❌ LSP 服务器 — 待开发
- ❌ 社区建设 — 待启动
- 详见：`docs/superpowers/plans/2026-05-26-phase6-ecosystem-tools.md`
- 跳过项备忘：`docs/superpowers/plans/2026-05-26-phase4-6-skipped.md`

---

## 十、开放问题与待定决策

以下问题有待进一步讨论和决定：

1. **所有权模式切换语法**：如何在类型或声明层面区分「引用计数模式」和「独占所有权模式」？候选方案包括类型标注（`Owned<T>` vs `Shared<T>`）或关键字（`let own` vs `let shared`）。

2. **shape 符号维度的表示语法**：用大写字母（`Tensor[f32, M, K]`）还是其他约定来区分符号维度和字面量维度？

3. **自动广播规则的严格度**：是否像 NumPy 那样全面支持隐式广播，还是要求部分显式（类似 JAX 的 `lax.broadcast_in_dim`）？

4. **泛型的 rank 约束语法**：如何表达「此泛型参数必须是一个秩为 2 的张量」这类约束？`Tensor[T, R=2]`？`Tensor[T, ..2]`？

5. **标准库宏的安全模型**：proc macro 可以任意操作 HIR，是否需要沙箱机制或能力限制？

6. **GPU 后端优先级**：先支持 CUDA 还是同时考虑 ROCm（AMD）和 Vulkan Compute（跨平台）？

7. **调试模式解释器的性能目标**：调试模式下解释执行的最低可接受速度是多少？需要 JIT 辅助吗？

8. **async/await 与 GPU 操作的交互**：GPU kernel 是异步执行的，如何与语言级别的 async/await 模型统一？

---

## 附录：灵感与参考

- **Rust**：所有权系统、trait、enum、宏系统
- **Julia**：类型特化 JIT、多重派发、科学计算的易用性
- **JAX**：函数式变换（grad, vmap, pmap）、XLA 编译
- **Swift for TensorFlow**：将 autodiff 作为语言特性的尝试
- **APL / BQN**：数组编程、rank polymorphism
- **MLIR / XLA / TVM**：多层 IR 编译架构
- **Mojo**：ML 领域新语言的近期探索
- **Taichi**：多阶段编程、从 Python 生成 GPU kernel