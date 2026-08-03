# Tenth GPU 真正落地——可行性评估报告（M3「GPU 另评」）

| 项目 | 内容 |
|------|------|
| 版本 | v1.0 |
| 日期 | 2026-08-04 |
| 性质 | 纯调研 + 文档（不写实现代码） |
| 委派 | M3 规划「GPU 真正落地（另评）」——可行性评估后再决策 |
| 关联 | `docs/superpowers/specs/2026-06-25-gpu-backend/`（既有 CUDA spec）、`docs/shape-check-roadmap/综合分析.md` §2.1、AUDIT-11.4.6 |
| 结论速览 | **推荐 wgpu 优先做 MVP（跨平台验证管线）→ cudarc/CUDA 作为后续性能后端；建议 M4 先做，GPU MVP 紧随其后，全算子覆盖为长期主线** |

---

## 摘要（核心结论）

1. **技术路线**：推荐 **wgpu（MVP）→ cudarc/CUDA（性能后端）双后端**。与 `docs/shape-check-roadmap/综合分析.md:61` 既有建议（「先支持 WGPU（跨平台、易实现）验证管线，再上 CUDA（性能）」）一致，且与既有 spec（`2026-06-25-gpu-backend`）§5.1 决策 5 预留的「Device trait 多后端」设计兼容。
2. **f64 关键约束**：GPU 上 f64 **技术可行但性能弱**（消费级显卡 = f32 的 1/32；浏览器 WebGPU 无 f64）。与 Tenth 现有 f32 降级策略（语言规范 §3.2）天然协同：**GPU 路径强制 f32/f16，f64 保留 CPU**。这不是阻塞，而是必须写进架构的硬约束。
3. **集成点**：所有执行路径（VM/解释器/JIT）最终汇聚到 `Tensor` 方法层（`runtime/tensor/methods.rs`），**在 Tensor 方法层做设备分派是最小侵入方案**，无需改动三条执行路径各自的分派逻辑。
4. **优雅降级**：环境级（无 GPU → 全局回退 CPU + 用户可见通知）与算子级（有 GPU 但某算子未 GPU 化 → **显式报错**，禁止静默 fallback）两层语义必须分离，守护「静默失败防护」护城河不因 GPU 削弱。
5. **优先级**：**不建议 GPU 优先于 M4 生态工具链**。M4（tenthpm/LSP/datasets，2-4 周）工作量小、是「语言可用性」必需；GPU 全算子覆盖工程量极大（≥6000 行，5+ Phase），属长期主线（综合分析 P2）。**建议：M4 先做 → GPU MVP（约 4-6 周）紧随其后 → 全算子覆盖作为长期主线并行**。
6. **不做 GPU 的风险**：Tenth 停留 demo 阶段（综合分析.md:50「无 GPU 就不能训练真实模型」），M4「标准库 AI 生态」缺算力支撑，「AI 原生」名不副实，与竞品（PyTorch/JAX）的 GPU 差距持续。

---

## 一、现状（代码证据）

### 1.1 张量运行时：纯 CPU，ndarray 内核

| 事实 | 证据 |
|------|------|
| `TensorData` = `ArrayD`（F32/F64/F16/BF16 四变体，Wave 2 半精度已落地） | `runtime/tensor/mod.rs:18-28` |
| `Tensor { dtype, data, grad, tape_id }`，**无 device 字段** | `runtime/tensor/mod.rs:31-38` |
| 连续内存切片（`as_f32_slice`/`as_f64_slice`）仅 C 连续时可用，否则 None | `runtime/tensor/data.rs:70-85` |
| matmul 用 ndarray `dot()`（**纯 Rust，无 BLAS 加速**），2D/1D 多形态 | `runtime/tensor/methods.rs:774-830` |
| 广播：`broadcast_shape` + ndarray `broadcast()`（NumPy 语义） | `runtime/tensor/methods.rs:639,455-505` |
| `Cargo.toml` **无任何 GPU 依赖**（无 wgpu/cuda/candle） | `tenth/Cargo.toml:14-28` |

### 1.2 autodiff：Wengert tape，27 个 TapeOp

- `TapeOp` 枚举 27 个变体（Input/Add/Sub/Mul/Div/Neg/ReLU/MatMul/BatchedMatMul/Transpose/Sum/Mean/Exp/Log/Sigmoid/Softmax/CrossEntropy/Dropout/Conv2D/BatchNorm/LayerNorm/Gelu/Select/Abs/Scatter/Gather/Reshape/MaskedFill/MaxPool2D/AvgPool2D/Custom）——见 `runtime/autodiff/tape_op.rs:111+`
- backward 实现分文件：`runtime/autodiff/backward.rs`、`grad.rs:108`（`match op` 分发）

### 1.3 多执行路径与统一汇聚点

```
.tth → Lexer → Parser → HIR → VM(默认) / 解释器(TENTH_NO_VM) / JIT(VM 加速) / WASM
```

- VM 张量方法分派：`runtime/vm/natives.rs:30` `call_method_priv`
- 解释器张量方法分派：`runtime/interpreter/methods.rs:872` `eval_tensor_method`
- JIT 是 VM 的加速层（走 VM native 调用）
- **三者最终都调用 `Tensor` 的方法（methods.rs）** → Tensor 层是天然统一集成点

### 1.4 已有 GPU 基础设施（全部是「假 GPU」脚手架）

| 资产 | 现状 | 问题 |
|------|------|------|
| `compile/gpu/`（mod/device/cuda_kernel） | CUDA C 源码字符串生成 | `device.rs:79` `is_available()` 永真模拟；`from_hir_function` body 仅占位符；无 nvcc/NVRTC/launch |
| 既有 spec `2026-06-25-gpu-backend` | 5 Phase 完整计划（cudarc 路线，~6000 行上限 E3） | ① 假设 CUDA/cudarc；② 依赖 f32/Shape spec（**均已落地**，前置已满足）；③ spec §3 灵活修改条款允许「调研发现更优方案（如 wgpu 优先于 CUDA）」调整 |
| AUDIT-11.4.6 | CUDA 后端文档矛盾（已修措辞，代码仍是脚手架） | 已登记 |
| `compile/optimizations/fusion.rs` / `parallel.rs` | 字符串拼接 / 硬编码 1024×1024 | 玩具实现 |

### 1.5 项目状态与排期语境

- M3 暂缓（GPU 另评未启动），当前处于「性能深挖 + 健壮性」P 系列（`/memories/repo/project-status.md`）
- 测试 2341 passed / 0 failed；自举 OK；基准 matmul 0-1ms（样例矩阵小，CPU 已够快）
- M4 生态与工具链（tenthpm/LSP/AI 生态/调试器）未启动，预计 2-4 周

---

## 二、技术路线对比（带证据/调研）

### 2.1 候选路线总表

| 维度 | **wgpu** | **cudarc（CUDA 直连）** | **candle** | **arrayfire** | tch-rs / burn / dfdx |
|------|----------|------------------------|------------|---------------|----------------------|
| 最新版本/活跃度 | **v30.0.0（1 个月前），47K dependents，613 contributors** | v0.19.8（2 个月前，CUDA 13.3 支持） | v0.11.0（2 个月前），20.8k stars | 主库 v3.10.0（11 个月前）；**Rust 绑定 v3.8.0 停更 6 年** | tch-rs（绑定 libtorch）/ burn（后端抽象）/ dfdx（nightly 依赖） |
| Windows 支持 | ✅ DX12 一等支持 | ✅（需 NVIDIA GPU + CUDA toolkit） | ✅（CUDA 后端） | ✅（需安装 C++ 库 AF_PATH） | tch-rs ✅ 但引入整个 torch |
| 跨平台 | ✅ Vulkan/DX12/Metal/OpenGL/WebGPU(wasm) | ❌ NVIDIA only（macOS 无） | ✅ CPU/CUDA/Metal/WASM | ✅ CUDA/OpenCL/CPU | tch-rs 跨平台 |
| WASM | ✅ WebGPU（浏览器） | ❌ | ✅（有 wasm 示例） | ❌（Rust 绑定不支持） | tch-rs ❌ |
| f64 支持 | ⚠️ **实验性**（naga 支持 double，原生可用；浏览器 WebGPU 无 f64） | ✅ 硬件 double + cuBLAS DGEMM（但消费卡慢） | ⚠️ 弱（主 f32） | ✅（CUDA f64 后端） | torch f64 ✅ 但消费卡慢 |
| 是否自带 Tensor/autodiff | ❌ 无（自己写 kernel） | ❌ 无（绑定层） | ✅ 自带 `Tensor`（**与 Tenth 冲突**） | ✅ 自带 `af::array`（冲突） | torch 自带（冲突） |
| matmul 加速 | 手写 tiled WGSL kernel | ✅ cuBLAS（cublasSgemm/Dgemm） | ✅ 自带 | ✅ 自带 | ✅ cuDNN/cuBLAS |
| License | Apache-2.0/MIT | Apache-2.0/MIT | Apache-2.0/MIT | BSD-3-Clause | torch BSD / burn MIT / dfdx MIT |
| 安全 | ✅ 纯 Rust 无 unsafe | ⚠️ FFI unsafe 需封装 | ✅ | ⚠️ FFI | ⚠️ FFI |
| 开发环境依赖 | ✅ 零（DX12 即插即用） | ❌ 需 CUDA toolkit + NVIDIA GPU | ⚠️ CUDA 后端需工具链 | ❌ 需安装 C++ 库 | ❌ 需 torch |

### 2.2 逐路线结论

- **wgpu**（推荐 MVP）：
  - 证据：GitHub gfx-rs/wgpu 主页——「cross-platform, safe, pure-Rust graphics API. Runs natively on Vulkan, Metal, D3D12, OpenGL; and on top of WebGL2 and WebGPU on wasm」；MSRV 1.87（Tenth rustc 1.95 ✅）；Apache-2.0/MIT。
  - f64：naga（wgpu 的 shader 编译器）已支持 double-precision 构造（wgpu CHANGELOG：修复 GLSL double-precision overloads、支持 64-bit 浮点矩阵 WGSL 生成）——**原生路径 f64 可行但属 WGSL 实验性特性**；**浏览器 WebGPU 无 f64**（WASM 路径 GPU 只能 f32）。
  - 优势：跨平台全覆盖（含未来 M5 的 Win/Linux/macOS/WASM 产物）、纯 Rust 安全（与护城河「安全」叙事一致）、Windows 开发机零工具链依赖、无 unsafe 泄漏风险。
  - 代价：无现成 BLAS，matmul 需手写 tiled kernel（性能上限低于 cuBLAS，但对 MVP「验证管线」足够）；WGSL 生态偏图形/渲染，科学计算 kernel 库少。
- **cudarc / CUDA 直连**（推荐后续性能后端）：
  - 证据：GitHub cudarc 主页——v0.19.8，CUDA 11.4-13.3 支持；覆盖 driver/NVRTC/cuRAND/cuBLAS/cuBLASLt/NCCL/cuDNN；`dynamic-loading` 默认（构建时无需 CUDA 库）；Apache-2.0/MIT；已有 spec（2026-06-25）已选此路线。
  - f64：硬件 double 原生支持 + cuBLAS DGEMM，**但消费级显卡（GeForce）f64 吞吐 = f32 的 1/32**（专业卡 A100/H100 约 1/2）。
  - 优势：cuBLAS 提供工业级 matmul、NVRTC 复用现有 `cuda_kernel.rs` 的 `to_cuda_code()` 字符串生成。
  - 代价：NVIDIA only（放弃跨平台）、需 CUDA 工具链 + NVIDIA GPU（正是已有 spec 退出条件 E5）、FFI unsafe 需安全封装（E1 风险）。
- **candle**（排除）：
  - 证据：GitHub huggingface/candle——20.8k stars、多后端（CPU/CUDA/Metal/WASM）、`candle_core::Tensor` 自带 Tensor 与 autodiff。
  - 排除理由：**自带 Tensor 类型与 Tenth 现有 `Tensor`/ndarray/autodiff 架构深度冲突**——集成 = 替换张量内核而非渐进 GPU 化，破坏现有 2341 测试的适配面；f64 支持弱（Tenth 默认 f64，这是硬约束）；candle 是「完整框架」，与「自研语言 + 自研张量层」的架构哲学不符。可作参考实现，不作依赖。
- **arrayfire**（排除）：
  - 证据：arrayfire-rust 绑定「Rust wrapper for ArrayFire 3.8.0 · Latest 6 years ago」，最后提交 3 年前；依赖 C++ 库安装（AF_PATH/PATH）。
  - 排除理由：**Rust 绑定实质停滞**、Windows 安装配置复杂、BSD-3-Clause 附带商标条款。活跃度不足以作为长期依赖。
- **tch-rs / burn / dfdx**（排除）：tch-rs 引入整个 libtorch 运行时（体积/哲学冲突）；burn 是后端抽象框架（抽象层复杂度与 Tenth 现有架构重叠）；dfdx 需要 nightly 且 shape 编码进类型（与 Tenth 运行时动态 shape 冲突）。

### 2.3 f64 关键约束（决策级事实）

| 路径 | f64 可行性 | 性能 | 结论 |
|------|-----------|------|------|
| CUDA（cudarc） | ✅ 硬件原生 | ⚠️ 消费卡 1/32，专业卡 1/2 | 可行但默认 f64 用户收益低 |
| wgpu 原生（Vulkan/DX12/Metal） | ⚠️ 实验性（naga 支持） | ⚠️ 同样受硬件限制 | 可行但有实验性风险 |
| wgpu/WASM（浏览器 WebGPU） | ❌ 无 f64（仅 f32） | — | WASM 路径 GPU 只能 f32 |
| candle | ⚠️ 弱 | — | 不满足默认 f64 约束 |

**推论**：无论哪条路线，GPU 上 f64 都是「可行但慢」。这**不是阻塞**——Tenth 已有 f32 一等公民（`TensorData::F32`）+ f32 降级策略（语言规范 §3.2），与「GPU 路径强制 f32/f16、f64 留 CPU」的架构决策天然协同。**必须把「GPU 算子只承诺 f32/f16，f64 自动落 CPU 路径」写进 GPU 架构规范**，避免用户误以为 f64 上了 GPU。

---

## 三、与现有架构集成评估

### 3.1 内存布局桥接

- `TensorData` 底层是 ndarray `ArrayD`，默认 C 连续；`as_f32_slice`/`as_f64_slice`（data.rs:70-85）在连续时返回切片——**上传路径**：`ArrayD` → 连续 slice → `write_buffer`（GPU buffer）；**下载路径**：`map_buffer`/`copy_buffer_to_buffer` → `Vec` → `ArrayD::from_shape_vec`。
- 非连续视图（切片/转置视图）需先 `to_owned()` 规整为连续再上传（成本可控，MVP 阶段显式处理）。
- 广播：GPU 端两种方案——① host 端先 `broadcast()` 成完整 shape 再上传（简单、浪费显存）；② kernel 内按 strides 索引（省显存、需把广播元数据传 GPU）。MVP 用 ①，后续优化用 ②。

### 3.2 autodiff（TapeOp）映射

| TapeOp 类别 | GPU 化优先级 | 说明 |
|-------------|-------------|------|
| elementwise（Add/Sub/Mul/Div/Neg/ReLU/Exp/Log/Sigmoid/Gelu/Abs/Select） | **P0（MVP 随 forward 一起）** | kernel 简单，backward 是 elementwise 公式，成本低 |
| MatMul/Transpose/Sum/Mean/Softmax/CrossEntropy | **P1（MVP 第二波）** | matmul 需 tiled kernel；归约需多 block 两阶段 |
| Conv2D/BatchNorm/LayerNorm/Dropout/MaxPool2D/AvgPool2D/Scatter/Gather/Reshape/MaskedFill | **P2（长期主线）** | 复杂度高，backward 公式复杂（BN/LN 闭式解在 GPU 上需逐项 kernel） |
| Custom（用户自定义算子） | 长期 | 依赖用户提供 GPU 实现，MVP 不支持（显式报错） |

- **backward 映射原则**：GPU forward 产生的 tape 节点携带 device 信息，backward 在**同设备** replay（已有 spec P5.4 设计：`TapeNode` 增 `device` 字段）。f64 backward 一律落 CPU。
- **数据流**：GPU 算子输入/输出必须都在 GPU；跨设备（GPU↔CPU）操作显式报错或显式 `to_device`（禁止隐式传输——护城河「无隐藏性能陷阱」）。

### 3.3 多执行路径共享架构（建议分层）

```
VM(natives.rs) ─┐                     ┌─→ Tensor::add/mul/matmul/...（设备分派）
解释器(methods.rs)─┼→ Tensor 方法层 ──┤
JIT(VM 加速) ────┘                     └─→ GPU: Device::dispatch(op) ─→ WGSL kernel / CUDA kernel
                                       └─→ CPU: 现有 ndarray 实现（零改动）
```

- **关键结论**：VM/解释器/JIT 三路径的分派逻辑**一行不改**——它们已经统一调用 `Tensor` 方法；只需在 `Tensor` 方法内部（或方法入口）按 `Tensor.device` 分派。这使 GPU 集成对三条执行路径透明。
- **WASM 路径**：WASM 程序跑在宿主运行时（wasmtime）内，其 tensor 方法调用仍走 Rust 侧 `Tensor` 层——GPU 对 WASM 后端**正交**（WASM 程序能否用 GPU 取决于宿主进程是否有 GPU 设备，与 wasm 目标无关）。但浏览器 WebGPU 的 f64 限制意味着 WASM 上的 GPU 仅 f32。
- **JIT 交互**：JIT 已把 tensor 方法编译为 native 调用（走 VM `call_method_priv`）；GPU 化在 `Tensor` 层内部完成，JIT 无需感知。风险点是「JIT 特化假设张量在 CPU」——需在 GPU MVP 的 JIT 一致性套件中覆盖（见 §四风险 R6）。

### 3.4 优雅降级与护城河防线

| 场景 | 行为 | 原则 |
|------|------|------|
| 无 GPU 设备（`is_available()==false`） | **全局自动回退 CPU** + 启动时用户可见通知（`--gpu` 时警告「未检测到 GPU，回退 CPU」） | 环境级降级 = 允许 |
| 有 GPU，某算子未 GPU 化（如 Conv2D backward） | **显式报错**「GPU op not supported: X，请用 CPU 路径或 to_host()」 | 算子级降级 = **禁止静默**（守护「静默失败防护」护城河） |
| 跨设备操作（GPU+CPU 混合） | 显式报错「device mismatch」，要求显式 `to_device`/`to_host` | 无隐式传输（防性能陷阱） |
| shape 检查 / 静默失败防线 | **在 Tensor 方法层之前执行**（编译期 shape 检查在 HIR 层，不因 device 改变）；GPU 广播错误、shape 不匹配仍走既有错误通道 | GPU 不削弱既有防线 |

> 关键区分：**「环境级优雅降级」（无 GPU 回退 CPU，允许）≠「算子级静默 fallback」（有 GPU 却偷偷算 CPU，禁止）**。这是 GPU 架构最重要的语义约束，直接关联护城河「静默失败防护」。

---

## 四、工作量与风险

### 4.1 MVP 到全算子覆盖的路径

```
MVP（4-6 周）：Device 抽象 + 上传/下载 + elementwise 10 算子 + MatMul/Transpose/Sum/Mean + backward（elementwise/matmul）+ 自动回退 + 端到端训练对拍
  ↓
中期（M4 后，8-12 周）：Softmax/CrossEntropy/Dropout/BN/LN/Conv2D/Pool forward + 全 backward kernel + 广播 kernel 内优化 + 性能基准
  ↓
长期主线（P2，持续）：Scatter/Gather/MaskedFill/Custom + 算子融合（真 fusion，替换 fusion.rs 字符串拼接）+ 多流异步 + 多 GPU/NCCL（异步 spec 接入）
```

### 4.2 风险登记册

| 编号 | 风险 | 等级 | 缓解 |
|------|------|------|------|
| R1 | **驱动兼容**（Windows DX12/WGL/Vulkan 驱动差异、WGSL 编译器差异；CUDA 工具链版本） | 高 | wgpu 抽象掉驱动差异（一等支持 DX12）；`WGPU_BACKEND` 环境变量可强制后端调试；CUDA 路线有既有 spec E5 退出条件 |
| R2 | **调试难度**（GPU kernel 无 line-level 报错，数值错误难定位） | 高 | 严格数值对拍测试（f32 相对误差 ≤1e-5）；kernel 发射计数 + device 指针追踪（防假 GPU）；复用 `TENTH_NO_VM` 式开关做 CPU 对拍 |
| R3 | **f64 性能/支持**（消费卡 1/32；WASM 无 f64） | 中 | 架构规范明确「GPU 只承诺 f32/f16，f64 落 CPU」；文档向用户披露 |
| R4 | **CI 无法测 GPU**（本机/CI 无 GPU） | 高 | ① mock 后端（结构测试）；② kernel 发射计数守护（无 GPU 也验证调度逻辑）；③ 数值对拍测试标 `#[ignore]` 本地跑（仿 `memory_estimate_test.rs:446` 模式）；④ `--gpu` 端到端验证需 GPU 机器，登记为「需要 GPU 环境的测试」 |
| R5 | **wgpu matmul 性能上限**（手写 tiled kernel 低于 cuBLAS） | 中 | MVP 验收阈值设「≥CPU×3」而非「≥cuBLAS」；性能后端再上 cudarc/cuBLAS |
| R6 | **JIT 交互**（JIT 特化假设张量在 CPU；GPU 张量进 JIT 路径） | 中 | GPU MVP 的 jit_consistency 套件覆盖「GPU 张量经 VM/JIT 路径」；发现假设即补 `to_host` 或显式报错 |
| R7 | **自举三路径** | 低 | GPU 只动 Rust 运行时/张量层，不改 tenthc；既有 spec §2.3 已确认不破坏 A/B/C；仍跑 `run tenthc/main.th → [OK]` 守护 |
| R8 | **构建体积/时间**（wgpu 依赖链） | 低 | feature flag 门控（`gpu` 默认关闭），CI 非 GPU job 不启用；镜像既有 spec R9 设计 |
| R9 | **WGSL f64 实验性**（原生后端可用性漂移） | 中 | MVP 阶段 GPU 算子只实现 f32/f16 kernel，f64 一律走 CPU——天然规避实验性风险 |
| R10 | 行数/复杂度超估 | 中 | 沿用既有 spec E3（6000 行上限）；wgpu 路线 FFI 层更薄（无 unsafe），但 WGSL kernel 手写量大——按包拆分、每包独立验收 |

### 4.3 与 M 系列关系 / 排期建议

```
当前（性能深挖 P 系列收尾）→ M4 生态工具链（2-4 周，先做）→ GPU MVP（M4 后立即，作为 M3「另评」结论落地）→ 全算子覆盖（长期主线，M4 后与生态建设并行）
```

- **M3 关系**：GPU 是 M3 规划项（「另评」），本评估通过后 MVP 即 M3 的正式落地；但 M3 当前暂缓，MVP 顺延至 M4 后是自然时序。
- **是否优先于 M4？** **不建议**。理由：
  1. M4（tenthpm/LSP/datasets/调试器）2-4 周可交付，是「语言可用性」必需，成本低收益直接；
  2. GPU MVP 只覆盖 3-4 类算子，对「训练真实模型」的完整收益有限（多数模型的核心算子链仍慢），**全算子覆盖**才是生存级兑现——而那是长期主线；
  3. GPU 与 M4 互不依赖，M4 期间低强度并行 GPU MVP 的「包 1（抽象层+桥接）」无冲突。
- **折中**：若用户希望 GPU 尽快见效，可在 M4 启动时并行做 GPU MVP 包 1-2（抽象层 + elementwise），包 3-4（matmul/autodiff）在 M4 后集中。

---

## 五、建议

### 5.1 技术路线明确建议

**采用 wgpu 优先（MVP）+ cudarc/CUDA 作为后续性能后端 的双后端策略**：

1. **MVP 用 wgpu**（理由）：
   - 目标环境 Windows 本机 + 跨平台（Win/Linux/macOS/WASM）要求 → wgpu 天生满足，CUDA 仅 NVIDIA；
   - 纯 Rust 安全、无 unsafe 泄漏风险（与护城河「安全」叙事一致，规避既有 spec E1）；
   - Windows 开发机零工具链依赖（DX12 即插即用，规避既有 spec E5）；
   - WASM 路径与 M5「跨平台产物」对齐（WebGPU）；
   - MVP 只需 3-4 类算子，手写 WGSL kernel 工作量可控；
   - 与综合分析.md:61 既有建议一致。
2. **性能后端用 cudarc/CUDA**（理由）：cuBLAS 提供工业级 matmul；NVRTC 复用现有 `cuda_kernel.rs::to_cuda_code()`；既有 spec 的 5 Phase 计划基本可复用（Device trait 多后端设计已预留）；有 NVIDIA 机器时获得真正训练性能。
3. **架构上保持 `Device` trait 多后端抽象**（既有 spec §5.1 决策 5 已设计）：`DeviceKind { Cpu, Gpu }` + `Device::alloc/copy/launch`，wgpu 与 cudarc 各为实现。CPU 是默认零成本路径。
4. **f64 策略**：GPU 算子只承诺 f32/f16，f64 自动落 CPU 路径（写入架构规范 + 语言参考手册）。

### 5.2 MVP 工作包拆分

| 包 | 内容 | 估时 | 验收 |
|----|------|------|------|
| **包 1：GPU 抽象层 + 桥接** | wgpu 依赖（feature `gpu` 门控）；`DeviceKind`/`Device` trait；`Tensor.device` 字段（默认 Cpu）；`to_device`/`to_host`；`is_available()` 真实探测（替换 `device.rs:79` 永真模拟）；mock 后端；上传/下载桥接（ArrayD ↔ GPU buffer） | 1-1.5 周 | 无 GPU 自动回退 CPU；往返 bitwise 一致；跨设备操作显式报错；无 unsafe 泄漏；`cargo test` 全绿无回归 |
| **包 2：elementwise 算子 + 广播** | WGSL kernel：Add/Sub/Mul/Div/Neg/ReLU/Exp/Log/Sigmoid/Gelu；host 端广播展开；`Tensor` 方法层设备分派；kernel 发射计数 | 1-1.5 周 | 10 算子 GPU forward 数值一致（f32 相对误差 ≤1e-5，多 shape 含广播）；无假 GPU（发射计数>0）；CPU 零回归 |
| **包 3：MatMul + 归约** | matmul tiled WGSL kernel（f32，16×16 shared memory）；Transpose kernel；Sum/Mean 两阶段归约 | 1-1.5 周 | matmul/transpose/sum/mean 数值一致；matmul ≥CPU×3（1024²）；性能基准记录 |
| **包 4：autodiff 集成 + 端到端** | `TapeNode.device` 字段；backward 设备分派（同设备 replay）；elementwise/matmul backward kernel；`--gpu` CLI flag；MNIST 量级训练 CPU vs GPU 对拍 | 1-1.5 周 | forward+backward 数值一致；训练 loss 下降且与 CPU 容差内一致；`--gpu` 无 GPU 时警告并回退 CPU |

> 总计约 **4-6 周**（单人）。每包独立可验收，可随时中止（中止后 CPU 路径完整可用，不损失）。

### 5.3 不做 GPU 的风险

1. **生存级缺口持续**（综合分析.md:50）：无 GPU 不能训练真实模型，Tenth 停留在 demo 阶段——无论护城河多深都是「零」；
2. **M4 生态缺算力支撑**：「标准库 AI 生态（datasets·分布式·优化器）」在纯 CPU 上无意义，生态建设失去落脚点；
3. **护城河叙事不完整**：编译期 shape 检查 + 静默失败防护的差异化是「AI 原生」的上半场，运行期算力（GPU）是下半场；只有前者没有后者，「AI 原生」名不副实；
4. **竞品差距持续**（论文 T53 对比已披露 GPU 后端远不及 PyTorch/JAX），且随 PyTorch 生态继续扩大；
5. **机会成本**：GPU 是「入场券」不是「差异化」（综合分析 §2.1：AI 原生性弱），但入场券本身是启动一切训练场景的前提。

---

## 六、局限（诚实披露）

1. **外部库信息时效**：本报告基于 2026-08-04 可获取的 GitHub/crates 信息；wgpu v30 / cudarc v0.19.8 / candle v0.11 的 API 细节未在本机编译验证（纯调研任务，未写实现代码）。**包 1 的第一个验收动作必须是依赖引入 + 编译验证**（沿用既有 spec R1 的做法）。
2. **WGSL f64 支持的具体可用性**（原生后端 enable `f64` 特性后的行为、naga 对 double 的完整覆盖）**未实测**——本报告按「R9：GPU 算子只做 f32/f16、f64 走 CPU」规避，故不构成阻塞，但需在包 2 前做一次 spike 确认。
3. **性能数据（f64=1/32 消费卡等）** 为行业常识级估计（GeForce 架构 FP64 单元比例），未在本机实测；专业卡/新架构比例不同。建议在包 3 用真实基准校准。
4. **工作量估时为单人估算**（参考既有 spec 6000 行上限与 5 Phase 结构），未计入驱动问题排查的不可预期时间。
5. **wgpu matmul 性能上限**：手写 tiled kernel 大概率低于 cuBLAS；若未来性能主线要求达到 cuBLAS 量级，必须引入 cudarc/cuBLAS（本报告已建议）。
6. **未评估**：多 GPU / NCCL / 分布式（属异步 spec 范畴，仅提及）；算子融合的真实现（fusion.rs 替换）；WASM 浏览器端 WebGPU 的 f32 端到端验证（需浏览器环境）。

---

## 附录 A：证据索引

| 证据 | 位置 |
|------|------|
| Tensor 结构（无 device） | `tenth/src/runtime/tensor/mod.rs:31-38` |
| 连续切片仅限连续布局 | `tenth/src/runtime/tensor/data.rs:70-85` |
| matmul 纯 ndarray dot（无 BLAS） | `tenth/src/runtime/tensor/methods.rs:774-830` |
| 广播实现 | `tenth/src/runtime/tensor/methods.rs:639,455-505` |
| TapeOp 27 变体 | `tenth/src/runtime/autodiff/tape_op.rs:111+` |
| backward 分派 | `tenth/src/runtime/autodiff/grad.rs:108` |
| 解释器张量分派 | `tenth/src/runtime/interpreter/methods.rs:872` |
| VM 张量分派 | `tenth/src/runtime/vm/natives.rs:30` |
| GPU 脚手架（模拟设备） | `tenth/src/compile/gpu/device.rs:79`、`cuda_kernel.rs` |
| Cargo 无 GPU 依赖 | `tenth/Cargo.toml:14-28` |
| GPU 生存级必要性 | `docs/shape-check-roadmap/综合分析.md:46-61` |
| 既有 CUDA spec（5 Phase） | `docs/superpowers/specs/2026-06-25-gpu-backend/spec.md` |
| 既有 CUDA checklist（含退出条件） | `docs/superpowers/specs/2026-06-25-gpu-backend/checklist.md` |
| GPU 状态（未接 CUDA Runtime） | `docs/语言规范.md:1238`、AUDIT-11.4.6 |
| wgpu 跨平台/版本/license | GitHub gfx-rs/wgpu（v30.0.0，Apache-2.0/MIT，MSRV 1.87） |
| wgpu naga double 支持 | wgpu CHANGELOG（double-precision 修复条目） |
| cudarc 版本/CUDA 覆盖/license | GitHub chelsea0x3b/cudarc（v0.19.8，CUDA 11.4-13.3） |
| candle 版本/自带 Tensor/多后端 | GitHub huggingface/candle（v0.11.0，20.8k stars） |
| arrayfire Rust 绑定停滞 | GitHub arrayfire/arrayfire-rust（v3.8.0 停更 6 年） |
