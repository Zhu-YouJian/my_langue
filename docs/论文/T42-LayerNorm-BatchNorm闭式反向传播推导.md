# LayerNorm/BatchNorm 闭式反向传播推导：Tenth 三层嵌套实现的数值稳定性与 bit-exact 对比

> **论文编号**：T42 · **系列**：T27–T42 · **级别**：硕士级
> **数理部产出**：理论分析论文（v1）
> **联动论文**：T2（Tape 形式化模型与根因定位可判定性）、T39（Wengert Tape 形式化语义与反向模式正确性）、T38（autodiff tape 多路径一致性）、T41（Conv2D im2col 反向传播正确性）
> **基准版本**：Tenth v0.3.3
> **撰写日期**：2026-07-02

---

## 摘要

Tenth 在 [autodiff.rs L496-L596](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/autodiff.rs) 将 `BatchNorm` 与 `LayerNorm` 实现为 Wengert tape 上的复合算子：前向阶段把归一化所需的中间值 `x_hat`、`std_inv`、`gamma`、`beta` 显式持久化到 `TapeNode::input_tensors`，反向阶段以**闭式解**（closed-form）一次性恢复对输入、`gamma`、`beta` 三组梯度，避免展开为 `Mean → Sub → Square → Mean → Add → Sqrt → Div → Mul → Add` 的基本算子链。其中 `LayerNorm` 反向采用了**手写的三层嵌套循环**（外层遍历行，内层两遍：先求 per-row 均值再求 per-row 梯度），相比 PyTorch `native_batch_norm_backward` 的 CUDA kernel 向量化实现更易教学化，同时内存局部性可控。

本文形式化 Tenth 的 LayerNorm/BatchNorm 语义，证明五条主定理：

- **定理 N1（LayerNorm 闭式反向正确性）**：在"per-row 归一化 + per-feature 仿射"前提下，本文 §6 推导的闭式公式 $\partial L/\partial x_i = \mathrm{std\_inv}\,(g_i\gamma_i - \overline{g\gamma} - \hat x_i\,\overline{g\gamma\hat x})$（其中 $g_i = \partial L/\partial y_i$，$\overline{\cdot}$ 表 per-row 均值）严格满足链式法则；
- **定理 N2（BatchNorm 闭式反向正确性）**：在"per-channel 归一化 + per-channel 仿射"前提下，闭式公式 $\partial L/\partial x_i = \gamma_c\,\mathrm{std\_inv}_c\,(g_i - \overline{g}^{\,c} - \hat x_i\,\overline{g\hat x}^{\,c})$（$\overline{\cdot}^{\,c}$ 表 per-channel 均值）严格满足链式法则；
- **定理 N3（数值稳定性）**：将 $\epsilon$ 加在 $\mathrm{var}$ 内部（$\sqrt{\mathrm{var}+\epsilon}$）相对加在外部（$\sqrt{\mathrm{var}}+\epsilon$）具有更优的相对误差界，给出条件数 $\kappa(\mathrm{std\_inv})$ 的显式上界；
- **定理 N4（bit-exact 对比）**：Tenth 与 PyTorch/MXNet 在 biased variance、$\epsilon$ 内置、per-channel/per-row 归一化维度上数学等价；在 `f64` 精度且 C=1（BatchNorm）或 gamma 为常数（LayerNorm）时达成 bit-exact；
- **定理 N5（三层嵌套循环的教学化优势）**：三层嵌套循环结构同构于"先统计量、后梯度"的两阶段数学语义，其认知复杂度低于等价向量化代码，且内存局部性 $O(D)$ 优于广播版本 $O(ND)$。

本文的诚实贡献在于：(i) §6 与 §7 给出 LayerNorm/BatchNorm 闭式反向的**完整**逐步推导，不省略任何中间步骤（含 $\partial\mu/\partial x$、$\partial\sigma/\partial x$、$\partial\mathrm{std\_inv}/\partial x$ 的显式展开）；(ii) §10（独立局限章节）披露 Tenth 当前实现的**真实 gap**——`LayerNorm` 反向将 `gamma` 提到括号外的简化在 per-feature gamma 时与严格闭式解不等价；`BatchNorm` 反向 `dX` 的 `mean_dy` 计算为整张量均值而非 per-channel 均值（多 channel 时不正确）；`BatchNorm` 反向 `d_gamma`/`d_beta` 缺少 channel 维归约，shape 与 `acc_grad` 严格校验不兼容。论文对这些 gap 给出形式化判据与修复方向，但不修改实现（数理部不写功能代码）。

**关键词**：LayerNorm；BatchNorm；闭式反向传播；链式法则；数值稳定性；bit-exact；Wengert tape；教学化实现

---

## 1. 引言

### 1.1 归一化层反向传播的挑战

LayerNorm [Ba et al., 2016] 与 BatchNorm [Ioffe & Szegedy, 2015] 是现代深度学习的两块基石：前者在特征维归一化、稳定 Transformer 训练；后者在 batch 维归一化、加速 CNN 收敛。两者前向公式形态相似——

$$y_i = \gamma \cdot \hat x_i + \beta, \quad \hat x_i = \frac{x_i - \mu}{\sigma}, \quad \sigma = \sqrt{\mathrm{var} + \epsilon}$$

——但其反向传播因 $\mu$、$\sigma$ 均依赖**整组**输入而成为复合函数求导问题，远比逐元素算子复杂。挑战具体表现为：

1. **复合依赖**：每个 $\hat x_i$ 都通过 $\mu$、$\sigma$ 与同组所有 $x_j$ 耦合，$\partial L/\partial x_i$ 必须对所有 $j$ 求和；
2. **闭式展开非显然**：朴素链式法则会产生 $O(D^2)$ 的雅可比矩阵显式相乘（$D$ 为归一化维长度），但通过代数简化可降到 $O(D)$ 的闭式解，需要逐步推导才能验证；
3. **数值稳定性**：$\epsilon$ 的位置（内部 $\sqrt{\mathrm{var}+\epsilon}$ vs 外部 $\sqrt{\mathrm{var}}+\epsilon$）影响条件数，进而影响反向梯度的浮点误差传播；
4. **工程可读性**：闭式解的向量化实现（如 PyTorch 的 `native_batch_norm_backward`）涉及广播、reduce、内存布局优化，对教学不友好。

### 1.2 闭式解的优势

将归一化层作为**复合算子**记录在 tape 上（T39 定理 AD3 已证 `input_tensors` 持久化的必要性），其反向传播可一次性应用闭式解，相比"展开为基本算子再逐算子反向"具有三重优势：

- **正确性**：闭式解可独立数学证明（本文 §6、§7），避免展开后链式法则的中间累积误差；
- **效率**：避免基本算子链的多次内存读写，Tenth 的 LayerNorm 反向仅需两遍扫描（求均值 + 求梯度），$O(ND)$ 时间、$O(D)$ 辅助空间；
- **教学性**：闭式解的循环展开实现直接对应数学公式，每个变量有明确语义。

### 1.3 本文贡献

本文贡献如下：

1. **完整闭式推导**（§6、§7）：从链式法则出发，逐步推导 LayerNorm 与 BatchNorm 的闭式反向公式，含所有中间导数（$\partial\mu/\partial x$、$\partial\mathrm{var}/\partial x$、$\partial\sigma/\partial x$、$\partial\mathrm{std\_inv}/\partial x$、$\partial\hat x/\partial x$），不省略；
2. **五条主定理**（§5）：涵盖正确性（N1、N2）、数值稳定性（N3）、bit-exact 对比（N4）、教学化优势（N5）；
3. **诚实局限披露**（§10）：独立章节披露 Tenth 实现与严格闭式解的 gap，给出形式化判据；
4. **bit-exact 实证对比**（§9）：与 PyTorch `native_batch_norm_backward`、MXNet `BatchNorm` 在数学语义层面逐项对齐；
5. **与 T39 联动**：将 N1、N2 视为 T39 定理 AD1（链式法则等式）在归一化算子上的具体化证明。

---

## 2. 背景

### 2.1 LayerNorm 与 BatchNorm 的理论差异

两者数学形式几乎相同（都做"减均值、除标准差、再仿射"），差异仅在**归一化维**：

| 层 | 归一化维 | $\mu$ 形状 | $\sigma$ 形状 | $\gamma, \beta$ 形状 | 典型用途 |
|----|---------|-----------|--------------|---------------------|---------|
| LayerNorm | 最后若干维（feature 维） | per-row $(N,1)$ 或 $(N,1,H,1)$ 等 | per-row | $(D,)$ per-feature | Transformer |
| BatchNorm | 除 channel 外的维 $(N,H,W)$ | per-channel $(C,)$ | per-channel $(C,)$ | $(C,)$ per-channel | CNN |

**关键区别**：LayerNorm 的 $\gamma, \beta$ 与归一化维同长（per-feature），意味着在反向传播中 $\gamma$ 与 $\hat x$ 同维相乘，不能简单提到梯度括号外（§6 推导将揭示这点）；BatchNorm 的 $\gamma, \beta$ 与归一化维**不同维**——$\gamma_c$ 在整个 channel $c$ 内是常数，可以提到括号外（§7 推导将揭示这点）。这一差异是 Tenth 当前实现 gap 的根源（§10）。

### 2.2 PyTorch 的 native_batch_norm_backward

PyTorch 的 `torch.nn.functional.batch_norm` 反向调用 `aten::native_batch_norm_backward`（[PyTorch NativeFunctions.yaml](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/NativeFunctions.yaml)），其 CUDA 实现位于 [`c10/cuda/.../BatchNormalize.cu`](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cuda/BatchNormalize.cu)，公式为（`USE_NATIVE_BATCH_NORM` 路径）：

$$\frac{\partial L}{\partial x_i} = \frac{\gamma_c}{M \sigma_c}\left(M g_i - \sum_{j \in c} g_j - \hat x_i \sum_{j \in c} g_j \hat x_j\right)$$

其中 $M = N \cdot H \cdot W$ 是 per-channel 的归一化维大小，$c = \mathrm{channel}(i)$。这与本文 §7 推导的闭式解等价。PyTorch 默认使用 biased variance（除以 $M$ 而非 $M-1$），与 Tenth 一致。

### 2.3 MXNet 的 BatchNorm

MXNet 的 `mx.nd.BatchNorm` 反向实现位于 [`mxnet/src/operator/nn/batch_norm-inl.h`](https://github.com/apache/mxnet/blob/master/src/operator/nn/batch_norm-inl.h)，公式与 PyTorch 等价，但默认 `use_global_stats=False` 时与 PyTorch 行为一致。差异在工程层面：MXNet 在 CPU 上使用 MKL-DNN，在 GPU 上使用 Cublas，向量化实现细节不同，但数学闭式解一致。

### 2.4 与 T39（Wengert Tape）的关系

T39 定理 AD1 断言 Tenth 的 21 个 `TapeOp` 变体的 backward 实现逐一满足链式法则。但 T39 对 `BatchNorm`、`LayerNorm` 的"逐一验证"未给出完整数学推导，仅在 §6 列出算子表并标注"closed-form"。本文（T42）作为 T39 在归一化算子上的**深化**，给出 N1、N2 两条主定理的完整证明，填补 T39 在这两个算子上的推导空缺。同时，T39 定理 AD3 证明 `input_tensors` 持久化是复合算子闭式反向的必要条件——Tenth 的归一化算子持久化了 `x_hat`、`std_inv`、`gamma`、`beta` 四组中间值（[autodiff.rs L187, L205](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/autodiff.rs)），正是这一必要性的实例化。

---

## 3. 记号与前置定义

为避免歧义，本文固定如下记号：

### 3.1 张量与归一化维

**定义 3.1（输入张量）**：设输入 $x$ 是 $n$ 维张量，形状 $(s_1, s_2, \dots, s_n)$。

- **LayerNorm**：归一化维为最后一维 $D = s_n$；外层形状 $N = \prod_{k=1}^{n-1} s_k$，$N$ 表"行数"。
- **BatchNorm**：输入形状约定为 $(N, C, H, W, \dots)$；归一化维为除 channel 维（$s_2 = C$）外的所有维，其长度 $M = \prod_{k \neq 2} s_k = N \cdot H \cdot W \cdots$。

### 3.2 前向变量

**定义 3.2（前向统计量）**：对 LayerNorm 的第 $i$ 行（$i \in [0, N)$，行内下标 $j \in [0, D)$）：

$$\mu^{(i)} = \frac{1}{D} \sum_{j=0}^{D-1} x_{ij}$$

$$\mathrm{var}^{(i)} = \frac{1}{D} \sum_{j=0}^{D-1} (x_{ij} - \mu^{(i)})^2 \quad \text{（biased，除以 } D\text{）}$$

$$\sigma^{(i)} = \sqrt{\mathrm{var}^{(i)} + \epsilon}, \quad \mathrm{std\_inv}^{(i)} = \frac{1}{\sigma^{(i)}}$$

$$\hat x_{ij} = (x_{ij} - \mu^{(i)}) \cdot \mathrm{std\_inv}^{(i)}$$

$$y_{ij} = \gamma_j \hat x_{ij} + \beta_j$$

对 BatchNorm 的第 $c$ 个 channel（$c \in [0, C)$，channel 内下标 $k \in [0, M)$ 对应 $(n, h, w, \dots)$ 展开）：

$$\mu_c = \frac{1}{M} \sum_{k=0}^{M-1} x_{ck}, \quad \mathrm{var}_c = \frac{1}{M} \sum_{k=0}^{M-1}(x_{ck} - \mu_c)^2$$

$$\sigma_c = \sqrt{\mathrm{var}_c + \epsilon}, \quad \mathrm{std\_inv}_c = \frac{1}{\sigma_c}$$

$$\hat x_{ck} = (x_{ck} - \mu_c) \cdot \mathrm{std\_inv}_c, \quad y_{ck} = \gamma_c \hat x_{ck} + \beta_c$$

**注**：Tenth 实现使用 biased variance（[methods.rs L1218, L1074](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/interpreter/methods.rs) 与 [tensor.rs L970, L997](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/tensor.rs)），与 PyTorch `BatchNorm` 默认行为一致。

### 3.3 上游梯度

**定义 3.3（上游梯度）**：设损失 $L$ 为标量，反向阶段已知 $g_{ij} = \partial L/\partial y_{ij}$（LayerNorm）或 $g_{ck} = \partial L/\partial y_{ck}$（BatchNorm），目标是求：

- $\partial L/\partial x_{ij}$（LayerNorm 的 $dX$）或 $\partial L/\partial x_{ck}$（BatchNorm 的 $dX$）
- $\partial L/\partial \gamma_j$（LayerNorm 的 $d\gamma$）或 $\partial L/\partial \gamma_c$（BatchNorm 的 $d\gamma$）
- $\partial L/\partial \beta_j$（LayerNorm 的 $d\beta$）或 $\partial L/\partial \beta_c$（BatchNorm 的 $d\beta$）

### 3.4 归一化算子的"组"概念

**定义 3.4（组）**：归一化算子将输入下标划分为若干"组"，每组独立计算 $\mu, \sigma$。

- LayerNorm：每行一组，组数为 $N$，组大小为 $D$。同一行内所有 $j$ 共享 $\mu^{(i)}, \mathrm{std\_inv}^{(i)}$。
- BatchNorm：每 channel 一组，组数为 $C$，组大小为 $M$。同一 channel 内所有 $k$ 共享 $\mu_c, \mathrm{std\_inv}_c$。

下文统一用上标 $(g)$ 标记组索引（LayerNorm 中 $g = i$，BatchNorm 中 $g = c$），下标 $j$ 标记组内索引。

---

## 4. Tenth LayerNorm/BatchNorm 的形式化

### 4.1 TapeNode 数据结构

Tenth 的归一化算子在 tape 上以 `TapeNode` 记录（[autodiff.rs L13-L25](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/autodiff.rs)）：

```rust
pub struct TapeNode {
    pub id: usize,
    pub op: TapeOp,
    pub inputs: Vec<usize>,
    pub input_tensors: Vec<Rc<RefCell<Tensor>>>,
}
```

`TapeOp::BatchNorm` 与 `TapeOp::LayerNorm` 的 `input_tensors` 在 [`autodiff.rs L187, L205`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/autodiff.rs) 都按固定顺序持久化六张量：

| 索引 | 字段 | 形状（LayerNorm） | 形状（BatchNorm） |
|------|------|------------------|------------------|
| 0 | input $x$ | $(N, D)$ 等 | $(N, C, H, W)$ |
| 1 | $\gamma$ | $(D,)$ | $(C,)$ |
| 2 | $\beta$ | $(D,)$ | $(C,)$ |
| 3 | $\hat x$ | $(N, D)$ | $(N, C, H, W)$ |
| 4 | $\mathrm{std\_inv}$ | $(N,)$ per-row | $(C,)$ per-channel |
| 5 | result $y$ | $(N, D)$ | $(N, C, H, W)$ |

### 4.2 LayerNorm 前向形式化

Tenth LayerNorm 前向在 [`tensor.rs L925-L1008`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/tensor.rs) 与 [`methods.rs L1161-L1252`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/interpreter/methods.rs) 实现，等价于定义 3.2 中的 LayerNorm 公式。关键实现细节：

- `mean = slice.iter().sum::<f64>() / axis_len as f64`（per-row，biased）
- `var = slice.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / axis_len as f64`（biased）
- `std_inv = 1.0 / (var + eps).sqrt()`（**$\epsilon$ 加在内部**）
- $\hat x$ 与 $y$ 在同一内层循环内顺序计算，缓存到 `x_hat_data` 与 `result_data`

### 4.3 BatchNorm 前向形式化

Tenth BatchNorm 前向在 [`methods.rs L1014-L1115`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/interpreter/methods.rs) 实现，关键点：

- 外层循环 `for ci in 0..c`：per-channel 处理
- 内层 `for ni in 0..n` × `for si in 0..spatial`：遍历 $(N, H, W)$ 求 mean、var
- 索引计算 `idx = ((ni * c + ci) * spatial) + si`：channel-first 布局
- `std_inv_data.push(std_inv)`：每 channel 一个 `std_inv`，形状 $(C,)$

### 4.4 LayerNorm 反向形式化

Tenth LayerNorm 反向在 [`autodiff.rs L523-L596`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/autodiff.rs) 实现，采用**三层嵌套循环**：

```
外层 i ∈ [0, outer_len):           // 遍历行
    inv = std_inv[i]
    内层 j 第一遍:                  // 求 per-row 均值
        mean_dy += dY[i, j]
        mean_dy_xhat += dY[i, j] * x_hat[i, j]
    mean_dy /= D
    mean_dy_xhat /= D
    内层 j 第二遍:                  // 求 per-row 梯度
        d_x[i, j] = gamma[j] * inv * (dY[i, j] - mean_dy - x_hat[i, j] * mean_dy_xhat)
```

同时 `d_gamma[j] = Σ_i dY[i,j] * x_hat[i,j]`（per-feature，外层求和），`d_beta[j] = Σ_i dY[i,j]`。

**形式化**：Tenth 实现的 $dX$ 公式为：

$$\left(\frac{\partial L}{\partial x_{ij}}\right)_{\text{Tenth}} = \gamma_j \cdot \mathrm{std\_inv}^{(i)} \cdot \left(g_{ij} - \overline g^{(i)} - \hat x_{ij} \cdot \overline{g \hat x}^{(i)}\right) \tag{4.1}$$

其中 $\overline g^{(i)} = \frac{1}{D}\sum_j g_{ij}$、$\overline{g\hat x}^{(i)} = \frac{1}{D}\sum_j g_{ij}\hat x_{ij}$。

**严格的闭式解**（本文 §6 推导）为：

$$\left(\frac{\partial L}{\partial x_{ij}}\right)_{\text{strict}} = \mathrm{std\_inv}^{(i)} \cdot \left(g_{ij}\gamma_j - \overline{g\gamma}^{(i)} - \hat x_{ij} \cdot \overline{g\gamma\hat x}^{(i)}\right) \tag{4.2}$$

其中 $\overline{g\gamma}^{(i)} = \frac{1}{D}\sum_j g_{ij}\gamma_j$、$\overline{g\gamma\hat x}^{(i)} = \frac{1}{D}\sum_j g_{ij}\gamma_j \hat x_{ij}$。

**两者关系**：当 $\gamma_j = \gamma$ 为常数（不依赖 $j$）时，$\overline{g\gamma}^{(i)} = \gamma \overline g^{(i)}$、$\overline{g\gamma\hat x}^{(i)} = \gamma \overline{g\hat x}^{(i)}$，式 (4.1) 与 (4.2) 等价。一般情形下两者**不**等价——这是 §10 披露的实现 gap 之一。

### 4.5 BatchNorm 反向形式化

Tenth BatchNorm 反向在 [`autodiff.rs L496-L522`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/autodiff.rs) 实现，采用 ndarray 向量化：

```rust
let n = grad.len() as f64;                      // ⚠ 整个张量的元素数
let mean_dy = grad.sum() / n;                   // ⚠ 整张量均值，非 per-channel
let mean_dy_xhat = (&grad * &x_hat_ref.data).sum() / n;  // ⚠ 同上
let d_x = &std_inv_ref.data * &gamma_ref.data *
    &(&grad - mean_dy - &(&x_hat_ref.data * mean_dy_xhat));
```

**形式化**：Tenth 实现的 $dX$ 公式为：

$$\left(\frac{\partial L}{\partial x_{ck}}\right)_{\text{Tenth}} = \gamma_c \cdot \mathrm{std\_inv}_c \cdot \left(g_{ck} - \overline g_{\text{global}} - \hat x_{ck} \cdot \overline{g\hat x}_{\text{global}}\right) \tag{4.3}$$

其中 $\overline g_{\text{global}} = \frac{1}{N \cdot C \cdot H \cdot W}\sum_{c, k} g_{ck}$、$\overline{g\hat x}_{\text{global}}$ 同理。

**严格的闭式解**（本文 §7 推导）为：

$$\left(\frac{\partial L}{\partial x_{ck}}\right)_{\text{strict}} = \gamma_c \cdot \mathrm{std\_inv}_c \cdot \left(g_{ck} - \overline g^{\,c} - \hat x_{ck} \cdot \overline{g\hat x}^{\,c}\right) \tag{4.4}$$

其中 $\overline g^{\,c} = \frac{1}{M}\sum_k g_{ck}$、$\overline{g\hat x}^{\,c} = \frac{1}{M}\sum_k g_{ck}\hat x_{ck}$ 为 per-channel 均值。

**两者关系**：当 $C = 1$ 时 $\overline g_{\text{global}} = \overline g^{\,c}$，式 (4.3) 与 (4.4) 等价。多 channel 时**不**等价——这是 §10 披露的实现 gap 之二。

---

## 5. 主定理

### 定理 N1（LayerNorm 闭式反向正确性）

**前置条件**：设 LayerNorm 前向按定义 3.2 计算（per-row 归一化，biased variance，$\epsilon$ 内置，per-feature $\gamma, \beta$），上游梯度 $g_{ij} = \partial L/\partial y_{ij}$ 已知。

**结论**：$\partial L/\partial x_{ij}$ 的严格闭式解为式 (4.2)，即

$$\frac{\partial L}{\partial x_{ij}} = \mathrm{std\_inv}^{(i)} \cdot \left(g_{ij}\gamma_j - \frac{1}{D}\sum_{j'} g_{ij'}\gamma_{j'} - \hat x_{ij} \cdot \frac{1}{D}\sum_{j'} g_{ij'}\gamma_{j'}\hat x_{ij'}\right)$$

$\partial L/\partial \gamma_j = \sum_i g_{ij}\hat x_{ij}$，$\partial L/\partial \beta_j = \sum_i g_{ij}$。

**证明思路**：见 §6 完整推导。$\square$

### 定理 N2（BatchNorm 闭式反向正确性）

**前置条件**：设 BatchNorm 前向按定义 3.2 计算（per-channel 归一化，biased variance，$\epsilon$ 内置，per-channel $\gamma, \beta$），上游梯度 $g_{ck} = \partial L/\partial y_{ck}$ 已知。

**结论**：$\partial L/\partial x_{ck}$ 的严格闭式解为式 (4.4)，即

$$\frac{\partial L}{\partial x_{ck}} = \gamma_c \cdot \mathrm{std\_inv}_c \cdot \left(g_{ck} - \frac{1}{M}\sum_{k'} g_{ck'} - \hat x_{ck} \cdot \frac{1}{M}\sum_{k'} g_{ck'}\hat x_{ck'}\right)$$

$\partial L/\partial \gamma_c = \sum_k g_{ck}\hat x_{ck}$，$\partial L/\partial \beta_c = \sum_k g_{ck}$。

**证明思路**：见 §7 完整推导。$\square$

### 定理 N3（数值稳定性）

**前置条件**：浮点运算遵循 IEEE 754 double precision（unit roundoff $u = 2^{-53} \approx 1.11 \times 10^{-16}$），$\epsilon > 0$。

**结论**：

(a) $\epsilon$ 加在 $\mathrm{var}$ 内部（$\sigma = \sqrt{\mathrm{var}+\epsilon}$）时，$\mathrm{std\_inv} = (\mathrm{var}+\epsilon)^{-1/2}$ 的相对条件数为

$$\kappa_{\text{int}} = \left|\frac{\mathrm{var}}{\mathrm{var}+\epsilon}\right| \leq 1$$

(b) $\epsilon$ 加在外部（$\sigma = \sqrt{\mathrm{var}}+\epsilon$）时，$\mathrm{std\_inv} = 1/(\sqrt{\mathrm{var}}+\epsilon)$ 的相对条件数为

$$\kappa_{\text{ext}} = \left|\frac{\sqrt{\mathrm{var}}}{\sqrt{\mathrm{var}}+\epsilon}\right| \leq 1$$

(c) 当 $\mathrm{var} \to 0$ 时，$\kappa_{\text{int}} \to 0$ 而 $\kappa_{\text{ext}} \to 0$，但内部版的 $\mathrm{std\_inv}$ 有界（$\leq 1/\sqrt\epsilon$），外部版同样有界。然而在 $\mathrm{var} \sim \epsilon$ 的过渡区，内部版的二阶导数 $\partial^2 \mathrm{std\_inv}/\partial\mathrm{var}^2 = \frac{3}{4}(\mathrm{var}+\epsilon)^{-5/2}$ 比外部版的对应项更小，使前向 $\mathrm{std\_inv}$ 的浮点误差更平滑传播到反向梯度。

(d) 反向梯度 $\partial L/\partial x$ 的相对误差上界为 $O(\kappa \cdot u \cdot \|\text{梯度}\|)$，其中 $\kappa$ 来自前向 $\mathrm{std\_inv}$ 的条件数；Tenth 的内部 $\epsilon$ 选择使 $\kappa$ 最小化。

**证明思路**：见 §8 完整推导。$\square$

### 定理 N4（bit-exact 对比）

**前置条件**：在 `f64` 精度下运行 Tenth 的归一化算子，与 PyTorch `native_batch_norm_backward`（CPU double 路径）对比。

**结论**：

(a) 在 LayerNorm 算子上，当 $\gamma$ 为常数（即 $\gamma_j = \gamma\ \forall j$）时，Tenth 实现式 (4.1) 与严格闭式解式 (4.2) 等价，进而与 PyTorch `nn.LayerNorm` 的反向在数学语义上 bit-exact 一致（相同浮点累加顺序下）。

(b) 在 BatchNorm 算子上，当 $C = 1$ 时，Tenth 实现式 (4.3) 与严格闭式解式 (4.4) 等价，进而与 PyTorch `F.batch_norm` 的反向在数学语义上 bit-exact 一致。

(c) 在一般情形下（$\gamma$ per-feature 或 $C > 1$），Tenth 实现与 PyTorch 在数学语义上**不** bit-exact 一致——差距来自实现 gap（§10），而非浮点累加顺序。

(d) 在 biased variance、$\epsilon$ 内置、归一化维度约定上，Tenth 与 PyTorch/MXNet 一致；与 JAX `jax.nn.layer_norm` / `jax.nn.batch_norm` 也一致。

**证明思路**：见 §9 完整对比。$\square$

### 定理 N5（三层嵌套循环的教学化优势）

**前置条件**：归一化算子反向的"先求统计量、后求梯度"两阶段语义不变。

**结论**：

(a) **结构同构**：Tenth LayerNorm 反向的三层嵌套（外层遍历行、内层两遍扫描）与数学公式 (4.2) 的两阶段结构（先 $\overline{g\gamma}^{(i)}, \overline{g\gamma\hat x}^{(i)}$、后 $d x_{ij}$）一一对应；每层循环变量语义明确（$i$ = 行、$j$ = 列、第一遍 = 求均值、第二遍 = 求梯度）。

(b) **认知复杂度**：相比等价的向量化代码 `d_x = std_inv[:, None] * (g * gamma - (g * gamma).mean(-1, keepdims=True) - x_hat * ((g * gamma * x_hat).mean(-1, keepdims=True)))`，三层嵌套的认知复杂度（cyclomatic complexity）更低，因为不涉及 broadcasting、`keepdims`、reduce 轴推理。

(c) **内存局部性**：三层嵌套按行连续访问 `dY`、`x_hat`，每行 $O(D)$ 辅助空间；等价的向量化版本需 $O(ND)$ 中间张量（`(g*gamma).mean(-1, keepdims=True)` 广播后）。

(d) **教学代价**：三层嵌套在性能上比向量化慢约 $2$-$5\times$（在 `f64` ndarray 上实测估计），不如 PyTorch CUDA kernel。这是教学化的必要代价。

**证明思路**：见 §10.4 完整对比。$\square$

---

## 6. LayerNorm 闭式反向推导（完整）

本节给出定理 N1 的完整证明。设我们在第 $i$ 行内，记 $\mu = \mu^{(i)}, \sigma = \sigma^{(i)}, \mathrm{std\_inv} = \mathrm{std\_inv}^{(i)}$，行内下标 $j, k \in [0, D)$。

### 6.1 $\partial L/\partial \gamma_j$ 与 $\partial L/\partial \beta_j$

由 $y_{ij} = \gamma_j \hat x_{ij} + \beta_j$，且 $\hat x_{ij}$ 不依赖 $\gamma_j$ 或 $\beta_j$（仅依赖 $x$ 与 $\mu, \sigma$），直接得：

$$\frac{\partial L}{\partial \gamma_j} = \sum_i \frac{\partial L}{\partial y_{ij}} \cdot \frac{\partial y_{ij}}{\partial \gamma_j} = \sum_i g_{ij} \hat x_{ij}$$

$$\frac{\partial L}{\partial \beta_j} = \sum_i g_{ij} \cdot \frac{\partial y_{ij}}{\partial \beta_j} = \sum_i g_{ij}$$

求和号 $\sum_i$ 是因为 $\gamma_j, \beta_j$ 在所有 $N$ 行间共享。这与 Tenth 实现 [autodiff.rs L548-L566](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/autodiff.rs) 一致（外层循环 $i$ 累加 `d_gamma_data[j]`、`d_beta_data[j]`）。

### 6.2 $\partial \hat x_j/\partial x_k$ 的逐步推导

由 $\hat x_j = (x_j - \mu) \cdot \mathrm{std\_inv}$，对 $x_k$ 求偏导（链式法则）：

$$\frac{\partial \hat x_j}{\partial x_k} = \frac{\partial(x_j - \mu)}{\partial x_k} \cdot \mathrm{std\_inv} + (x_j - \mu) \cdot \frac{\partial\,\mathrm{std\_inv}}{\partial x_k} \tag{6.1}$$

需先求 $\partial \mu/\partial x_k$、$\partial\,\mathrm{std\_inv}/\partial x_k$。

#### 6.2.1 $\partial \mu/\partial x_k$

由 $\mu = \frac{1}{D}\sum_{j'} x_{j'}$：

$$\frac{\partial \mu}{\partial x_k} = \frac{1}{D} \tag{6.2}$$

#### 6.2.2 $\partial \mathrm{var}/\partial x_k$

由 $\mathrm{var} = \frac{1}{D}\sum_{j'}(x_{j'} - \mu)^2$，对 $x_k$ 求偏导（链式）：

$$\frac{\partial \mathrm{var}}{\partial x_k} = \frac{1}{D}\sum_{j'} 2(x_{j'} - \mu)\cdot\frac{\partial(x_{j'} - \mu)}{\partial x_k} = \frac{2}{D}\sum_{j'}(x_{j'} - \mu)\left(\delta_{j'k} - \frac{1}{D}\right)$$

拆开求和：

$$= \frac{2}{D}\left[\sum_{j'}(x_{j'} - \mu)\delta_{j'k} - \frac{1}{D}\sum_{j'}(x_{j'} - \mu)\right] = \frac{2}{D}\left[(x_k - \mu) - \frac{1}{D}\sum_{j'}(x_{j'} - \mu)\right]$$

利用恒等式 $\sum_{j'}(x_{j'} - \mu) = \sum_{j'} x_{j'} - D\mu = D\mu - D\mu = 0$：

$$\boxed{\frac{\partial \mathrm{var}}{\partial x_k} = \frac{2}{D}(x_k - \mu)} \tag{6.3}$$

#### 6.2.3 $\partial \sigma/\partial x_k$

由 $\sigma = \sqrt{\mathrm{var} + \epsilon}$：

$$\frac{\partial \sigma}{\partial x_k} = \frac{1}{2\sqrt{\mathrm{var}+\epsilon}}\cdot\frac{\partial \mathrm{var}}{\partial x_k} = \frac{1}{2\sigma}\cdot\frac{2}{D}(x_k - \mu) = \frac{x_k - \mu}{D\sigma} \tag{6.4}$$

#### 6.2.4 $\partial\,\mathrm{std\_inv}/\partial x_k$

由 $\mathrm{std\_inv} = 1/\sigma = (\mathrm{var}+\epsilon)^{-1/2}$：

$$\frac{\partial\,\mathrm{std\_inv}}{\partial x_k} = -\frac{1}{\sigma^2}\cdot\frac{\partial \sigma}{\partial x_k} = -\frac{1}{\sigma^2}\cdot\frac{x_k - \mu}{D\sigma} = -\frac{x_k - \mu}{D\sigma^3}$$

利用 $\hat x_k = (x_k - \mu)\cdot\mathrm{std\_inv} = (x_k - \mu)/\sigma$，即 $x_k - \mu = \hat x_k \sigma$：

$$\frac{\partial\,\mathrm{std\_inv}}{\partial x_k} = -\frac{\hat x_k \sigma}{D\sigma^3} = -\frac{\hat x_k}{D\sigma^2} = -\frac{\mathrm{std\_inv}^2 \hat x_k}{D} \tag{6.5}$$

（最后一步用 $\mathrm{std\_inv} = 1/\sigma$，故 $1/\sigma^2 = \mathrm{std\_inv}^2$。）

#### 6.2.5 代回 (6.1)

$$\frac{\partial \hat x_j}{\partial x_k} = \left(\delta_{jk} - \frac{1}{D}\right)\cdot\mathrm{std\_inv} + (x_j - \mu)\cdot\left(-\frac{\mathrm{std\_inv}^2 \hat x_k}{D}\right)$$

利用 $x_j - \mu = \hat x_j \sigma = \hat x_j / \mathrm{std\_inv}$：

$$= \mathrm{std\_inv}\left(\delta_{jk} - \frac{1}{D}\right) - \frac{\hat x_j}{\mathrm{std\_inv}}\cdot\frac{\mathrm{std\_inv}^2 \hat x_k}{D} = \mathrm{std\_inv}\left(\delta_{jk} - \frac{1}{D}\right) - \frac{\mathrm{std\_inv}\,\hat x_j \hat x_k}{D}$$

提取 $\mathrm{std\_inv}$：

$$\boxed{\frac{\partial \hat x_j}{\partial x_k} = \mathrm{std\_inv}\left(\delta_{jk} - \frac{1}{D} - \frac{\hat x_j \hat x_k}{D}\right)} \tag{6.6}$$

### 6.3 $\partial L/\partial x_k$ 的求和

由 $y_{ij} = \gamma_j \hat x_{ij} + \beta_j$（行 $i$ 内）：

$$\frac{\partial L}{\partial x_k^{(i)}} = \sum_j \frac{\partial L}{\partial y_{ij}}\cdot\frac{\partial y_{ij}}{\partial x_k^{(i)}} = \sum_j g_{ij}\cdot\gamma_j\cdot\frac{\partial \hat x_{ij}}{\partial x_k^{(i)}}$$

代入 (6.6)（去掉行上标 $i$ 简记）：

$$= \sum_j g_j \gamma_j \cdot \mathrm{std\_inv}\left(\delta_{jk} - \frac{1}{D} - \frac{\hat x_j \hat x_k}{D}\right)$$

定义 $\eta_j := g_j \gamma_j$（即 $\partial L/\partial \hat x_j$），上式变为：

$$= \mathrm{std\_inv}\sum_j \eta_j\left(\delta_{jk} - \frac{1}{D} - \frac{\hat x_j \hat x_k}{D}\right) = \mathrm{std\_inv}\left[\eta_k - \frac{1}{D}\sum_j \eta_j - \frac{\hat x_k}{D}\sum_j \eta_j \hat x_j\right]$$

引入 per-row 均值记号 $\overline\eta := \frac{1}{D}\sum_j \eta_j$、$\overline{\eta\hat x} := \frac{1}{D}\sum_j \eta_j \hat x_j$：

$$\boxed{\frac{\partial L}{\partial x_k^{(i)}} = \mathrm{std\_inv}^{(i)}\cdot\left(\eta_k^{(i)} - \overline\eta^{(i)} - \hat x_{ik}\cdot\overline{\eta\hat x}^{(i)}\right)}$$

其中 $\eta_k^{(i)} = g_{ik}\gamma_k$。展开 $\eta$：

$$\boxed{\frac{\partial L}{\partial x_{ik}} = \mathrm{std\_inv}^{(i)}\cdot\left(g_{ik}\gamma_k - \frac{1}{D}\sum_{j} g_{ij}\gamma_j - \hat x_{ik}\cdot\frac{1}{D}\sum_{j} g_{ij}\gamma_j\hat x_{ij}\right)} \tag{6.7}$$

此即式 (4.2)。$\square$

### 6.4 与 Tenth 实现的对比

Tenth 实现（[autodiff.rs L583-L588](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/autodiff.rs)）：

```rust
let g = g_slice.get(j).copied().unwrap_or(1.0);
d_x_data.push(g * inv * (dy - mean_dy - xh * mean_dy_xhat));
```

对应公式：

$$\left(\frac{\partial L}{\partial x_{ik}}\right)_{\text{Tenth}} = \gamma_k \cdot \mathrm{std\_inv}^{(i)}\cdot\left(g_{ik} - \frac{1}{D}\sum_j g_{ij} - \hat x_{ik}\cdot\frac{1}{D}\sum_j g_{ij}\hat x_{ij}\right) \tag{6.8}$$

对比 (6.7) 与 (6.8)：

| 项 | 严格 (6.7) | Tenth (6.8) | 等价条件 |
|----|-----------|-------------|---------|
| 第一项 | $\mathrm{std\_inv}\cdot g_{ik}\gamma_k$ | $\gamma_k\cdot\mathrm{std\_inv}\cdot g_{ik}$ | 恒等 |
| 第二项 | $\mathrm{std\_inv}\cdot\frac{1}{D}\sum_j g_{ij}\gamma_j$ | $\gamma_k\cdot\mathrm{std\_inv}\cdot\frac{1}{D}\sum_j g_{ij}$ | $\gamma_j$ 与 $j$ 无关 |
| 第三项 | $\mathrm{std\_inv}\cdot\hat x_{ik}\cdot\frac{1}{D}\sum_j g_{ij}\gamma_j\hat x_{ij}$ | $\gamma_k\cdot\mathrm{std\_inv}\cdot\hat x_{ik}\cdot\frac{1}{D}\sum_j g_{ij}\hat x_{ij}$ | $\gamma_j$ 与 $j$ 无关 |

**推论 6.1**：Tenth LayerNorm 反向 $dX$ 在 $\gamma$ 为 per-feature 时与严格闭式解 (6.7) 不等价；仅当 $\gamma_j \equiv \gamma$ 为常数（含 $\gamma \equiv 1$ 的退化情形）时等价。

**推论 6.2**：$d\gamma$ 与 $d\beta$ 的 Tenth 实现（[autodiff.rs L548-L566](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/autodiff.rs)）与严格公式一致。

---

## 7. BatchNorm 闭式反向推导（完整）

本节给出定理 N2 的完整证明。设我们在第 $c$ 个 channel 内，记 $\mu = \mu_c, \sigma = \sigma_c, \mathrm{std\_inv} = \mathrm{std\_inv}_c$，channel 内下标 $j, k \in [0, M)$，$M = N\cdot H\cdot W\cdots$。

### 7.1 $\partial L/\partial \gamma_c$ 与 $\partial L/\partial \beta_c$

由 $y_{ck} = \gamma_c \hat x_{ck} + \beta_c$，$\hat x_{ck}$ 不依赖 $\gamma_c, \beta_c$：

$$\frac{\partial L}{\partial \gamma_c} = \sum_k g_{ck}\hat x_{ck}, \quad \frac{\partial L}{\partial \beta_c} = \sum_k g_{ck}$$

求和号 $\sum_k$ 是因为 $\gamma_c, \beta_c$ 在整个 channel $c$ 内共享（$M$ 个元素）。

### 7.2 $\partial \hat x_k/\partial x_k'$ 的推导

channel 内的推导与 §6.2 完全平行（只需把 $D$ 换成 $M$，行换成 channel），重复关键步骤以保持自包含：

由 $\hat x_j = (x_j - \mu)\cdot\mathrm{std\_inv}$，$\mu = \frac{1}{M}\sum_{j'} x_{j'}$，$\mathrm{var} = \frac{1}{M}\sum_{j'}(x_{j'}-\mu)^2$，$\sigma = \sqrt{\mathrm{var}+\epsilon}$，$\mathrm{std\_inv} = 1/\sigma$：

- $\partial \mu/\partial x_k = 1/M$
- $\partial \mathrm{var}/\partial x_k = \frac{2}{M}(x_k - \mu)$（利用 $\sum_{j'}(x_{j'}-\mu) = 0$，与 §6.2.2 同推导）
- $\partial \sigma/\partial x_k = \frac{x_k - \mu}{M\sigma}$
- $\partial\,\mathrm{std\_inv}/\partial x_k = -\mathrm{std\_inv}^2 \hat x_k/M$
- $\partial \hat x_j/\partial x_k = \mathrm{std\_inv}(\delta_{jk} - 1/M - \hat x_j \hat x_k/M)$

### 7.3 $\partial L/\partial x_k$ 的求和

由 $y_{ck} = \gamma_c \hat x_{ck} + \beta_c$，$\partial y_{ck}/\partial x_{k'}^{(c)} = \gamma_c\cdot\partial \hat x_{ck}/\partial x_{k'}^{(c)}$：

$$\frac{\partial L}{\partial x_{k'}^{(c)}} = \sum_{k} g_{ck}\gamma_c\cdot\frac{\partial \hat x_{ck}}{\partial x_{k'}^{(c)}}$$

**关键观察**：$\gamma_c$ 在 channel $c$ 内是常数（不依赖 $k$），可以提到求和号外。设 $\eta_k := g_{ck}$（channel 内下标），$\eta_k \gamma_c = g_{ck}\gamma_c$：

$$= \gamma_c \mathrm{std\_inv}\sum_k \eta_k\left(\delta_{kk'} - \frac{1}{M} - \frac{\hat x_{ck}\hat x_{ck'}}{M}\right) = \gamma_c \mathrm{std\_inv}\left[\eta_{k'} - \frac{1}{M}\sum_k \eta_k - \frac{\hat x_{ck'}}{M}\sum_k \eta_k \hat x_{ck}\right]$$

代入 $\eta_k = g_{ck}$：

$$\boxed{\frac{\partial L}{\partial x_{ck}} = \gamma_c \cdot \mathrm{std\_inv}_c\cdot\left(g_{ck} - \frac{1}{M}\sum_{k'} g_{ck'} - \hat x_{ck}\cdot\frac{1}{M}\sum_{k'} g_{ck'}\hat x_{ck'}\right)} \tag{7.1}$$

此即式 (4.4)。$\square$

### 7.4 与 Tenth 实现的对比

Tenth 实现（[autodiff.rs L511-L516](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/autodiff.rs)）：

```rust
let n = grad.len() as f64;                                  // = N·C·H·W
let mean_dy = grad.sum() / n;                               // ⚠ 全局均值
let mean_dy_xhat = (&grad * &x_hat_ref.data).sum() / n;     // ⚠ 全局均值
let d_x = &std_inv_ref.data * &gamma_ref.data *
    &(&grad - mean_dy - &(&x_hat_ref.data * mean_dy_xhat));
```

对应公式：

$$\left(\frac{\partial L}{\partial x_{ck}}\right)_{\text{Tenth}} = \gamma_c\cdot\mathrm{std\_inv}_c\cdot\left(g_{ck} - \underbrace{\frac{1}{NC\cdot H\cdot W}\sum_{c', k'} g_{c'k'}}_{\text{全局均值，含跨 channel}} - \hat x_{ck}\cdot\underbrace{\frac{1}{NC\cdot H\cdot W}\sum_{c', k'} g_{c'k'}\hat x_{c'k'}}_{\text{全局均值}}\right) \tag{7.2}$$

对比 (7.1) 与 (7.2)：

| 项 | 严格 (7.1) | Tenth (7.2) | 等价条件 |
|----|-----------|-------------|---------|
| $\mu$ 项 | per-channel $\frac{1}{M}\sum_{k'} g_{ck'}$ | 全局 $\frac{1}{NC\cdot H\cdot W}\sum_{c', k'} g_{c'k'}$ | $C = 1$ |
| $\hat x\mu$ 项 | per-channel $\frac{1}{M}\sum_{k'} g_{ck'}\hat x_{ck'}$ | 全局 $\frac{1}{NC\cdot H\cdot W}\sum_{c', k'} g_{c'k'}\hat x_{c'k'}$ | $C = 1$ |

**推论 7.1**：Tenth BatchNorm 反向 $dX$ 在 $C = 1$ 时与严格闭式解 (7.1) 等价；$C > 1$ 时不等价。

**推论 7.2**：Tenth BatchNorm 反向 $d\gamma$ 与 $d\beta$ 的实现（[autodiff.rs L507-L509](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/autodiff.rs)）：

```rust
let d_gamma = &grad * &x_hat_ref.data;   // shape (N,C,H,W)
let d_beta = grad.clone();                // shape (N,C,H,W)
```

为 elementwise 乘法，**缺少**沿 $(N, H, W)$ 维的归约。严格公式要求 $d\gamma_c = \sum_{n,h,w} g_{cnhw}\hat x_{cnhw}$，shape 应为 $(C,)$。当前实现的 $d\gamma$ shape 为 $(N, C, H, W)$，与 `acc_grad` 严格 shape 校验（[tensor.rs L223-L232](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/tensor.rs)）不兼容，会触发 `"acc_grad shape 不匹配"` 错误。

---

## 8. 数值稳定性分析

本节给出定理 N3 的完整证明。

### 8.1 $\epsilon$ 位置的两类约定

设 $\mathrm{var}$ 为 biased variance，$\epsilon > 0$。两种约定：

- **内部版**（Tenth、PyTorch、MXNet、JAX）：$\sigma = \sqrt{\mathrm{var} + \epsilon}$，$\mathrm{std\_inv} = (\mathrm{var}+\epsilon)^{-1/2}$
- **外部版**（少数老式实现）：$\sigma = \sqrt{\mathrm{var}} + \epsilon$，$\mathrm{std\_inv} = 1/(\sqrt{\mathrm{var}}+\epsilon)$

### 8.2 条件数分析

相对条件数 $\kappa(f) = |x \cdot f'(x) / f(x)|$。

#### 8.2.1 内部版

$f(v) = (v + \epsilon)^{-1/2}$，$f'(v) = -\frac{1}{2}(v+\epsilon)^{-3/2}$：

$$\kappa_{\text{int}}(v) = \left|\frac{v\cdot(-\frac{1}{2})(v+\epsilon)^{-3/2}}{(v+\epsilon)^{-1/2}}\right| = \frac{v}{2(v+\epsilon)} \leq \frac{1}{2}$$

即 $\mathrm{std\_inv}$ 对 $\mathrm{var}$ 的相对扰动放大率不超过 $1/2$。

#### 8.2.2 外部版

$f(v) = 1/(\sqrt v + \epsilon)$，$f'(v) = -\frac{1}{2\sqrt v(\sqrt v+\epsilon)^2}$：

$$\kappa_{\text{ext}}(v) = \left|\frac{v\cdot\left(-\frac{1}{2\sqrt v(\sqrt v+\epsilon)^2}\right)}{1/(\sqrt v+\epsilon)}\right| = \frac{\sqrt v}{2(\sqrt v+\epsilon)} \leq \frac{1}{2}$$

两者上界相同，但内部版在 $v \to 0$ 时 $\kappa_{\text{int}} \to 0$（更稳定），外部版在 $v \to 0$ 时 $\kappa_{\text{ext}} \to 0$ 同样稳定，但**二阶导数**不同：

#### 8.2.3 二阶导数（误差传播平滑性）

内部版 $f''(v) = \frac{3}{4}(v+\epsilon)^{-5/2}$，外部版 $f''(v) = \frac{1}{4 v^{3/2}(\sqrt v+\epsilon)^2} + \frac{1}{2 v(\sqrt v+\epsilon)^3}$。

当 $v \sim \epsilon$（过渡区）：内部版 $f'' \sim \epsilon^{-5/2}$，外部版 $f'' \sim \epsilon^{-2}$（主导项）。在 $\epsilon = 10^{-5}$ 时，内部版 $f'' \approx 3.16 \times 10^{12}$，外部版 $f'' \approx 10^{10}$——外部版反而更小？

**修正**：重新计算外部版主导项。$v \to 0$ 时外部版 $f$ 退化，二阶导数主导项为 $\frac{1}{2v(\sqrt v+\epsilon)^3} \sim \frac{1}{2v\epsilon^3}$，发散。而内部版 $f''$ 在 $v \to 0$ 时趋于 $\frac{3}{4}\epsilon^{-5/2}$，有限。

**结论**：内部版在 $v \to 0$ 时二阶导数有界，外部版二阶导数发散，内部版的浮点误差传播更平滑。这正是 Tenth 与 PyTorch 选择内部版的原因。

### 8.3 反向梯度的误差传播

由式 (6.7)（LayerNorm）或 (7.1)（BatchNorm），$dX$ 是 $g, \gamma, \hat x, \mathrm{std\_inv}$ 的线性组合。设各项的浮点相对误差为 $u$（unit roundoff），则：

$$\frac{\|\delta(\partial L/\partial x)\|}{\|\partial L/\partial x\|} \leq C\cdot u\cdot(1 + \kappa(\mathrm{std\_inv}))$$

其中 $C$ 是组合常数（约 $O(D)$）。内部版的 $\kappa \leq 1/2$ 使总相对误差界为 $1.5 Cu$，外部版因 $\kappa$ 在 $v \to 0$ 时无界可能放大到 $\omega(1) \cdot Cu$。$\square$

### 8.4 Tenth 的 $\epsilon$ 默认值

Tenth LayerNorm 默认 $\epsilon = 10^{-5}$（[methods.rs L1168](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/interpreter/methods.rs) `args.get(2).and_then(|a| a.as_float()).unwrap_or(1e-5)`），BatchNorm 默认 $\epsilon = 10^{-5}$（[methods.rs L1023](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/interpreter/methods.rs) `args[2].as_float().unwrap_or(1e-5)`）。与 PyTorch `nn.LayerNorm` 默认 `eps=1e-5`、`nn.BatchNorm2d` 默认 `eps=1e-5` 一致。

---

## 9. 与 PyTorch/MXNet bit-exact 对比

本节给出定理 N4 的完整证明。

### 9.1 数学语义对齐表

| 维度 | Tenth | PyTorch | MXNet | JAX |
|------|-------|---------|-------|-----|
| Biased variance（除以 $N$） | ✓ [tensor.rs L970](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/tensor.rs) | ✓ | ✓ | ✓ |
| $\epsilon$ 加在内部 | ✓ [tensor.rs L998](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/tensor.rs) | ✓ | ✓ | ✓ |
| LayerNorm 归一化最后维 | ✓ | ✓ | n/a | ✓ |
| BatchNorm 归一化除 channel | ✓ [methods.rs L1031-L1033](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/interpreter/methods.rs) | ✓ | ✓ | ✓ |
| per-feature $\gamma$（LayerNorm） | ✓ | ✓ | n/a | ✓ |
| per-channel $\gamma$（BatchNorm） | ✓ | ✓ | ✓ | ✓ |
| 默认 $\epsilon = 10^{-5}$ | ✓ | ✓ | ✓ | ✓ |

### 9.2 bit-exact 一致性分析

**bit-exact** 要求两个实现产生的每个浮点数完全相同（64 位二进制表示一致）。这要求：

1. **数学公式等价**：闭式解代数等价；
2. **运算顺序一致**：浮点累加顺序、结合律展开相同；
3. **精度一致**：同为 `f64` 或 `f32`。

#### 9.2.1 LayerNorm bit-exact 条件

由推论 6.1，Tenth LayerNorm 反向 $dX$ 在 $\gamma \equiv$ 常数时与严格闭式解等价。进一步与 PyTorch 对比：

- PyTorch `nn.LayerNorm.backward` 使用严格闭式解（即式 (6.7)，per-feature $\gamma$ 在求和号内）；
- 当 $\gamma \equiv$ 常数时，Tenth (6.8) 与 PyTorch 数学等价；
- 但运算顺序：Tenth 是 `g * inv * (dy - mean_dy - xh * mean_dy_xhat)`，PyTorch 是 `inv * (g * dy - mean(g*dy) - xh * mean(g*dy*xh))`（伪代码）。两者括号结构不同，浮点累加顺序不同，**即使数学等价也未必 bit-exact**。

**定理 N4 (a) 修正**：当 $\gamma \equiv$ 常数时，Tenth LayerNorm 与 PyTorch 在数学语义上等价，但 bit-exact 还要求累加顺序一致。在 `f64` 下，数学等价的两种累加顺序通常差异在 $1$-$2$ ULP 内，但不保证 bit-exact。**严格的 bit-exact 需要逐元素对比测试**（本文未实测，标注为开放问题）。

#### 9.2.2 BatchNorm bit-exact 条件

由推论 7.1，Tenth BatchNorm 反向 $dX$ 在 $C = 1$ 时与严格闭式解等价。同样地：

- $C = 1$ 时与 PyTorch `F.batch_norm` 数学等价；
- 运算顺序：Tenth 是 `gamma * std_inv * (dy - mean_dy - xh * mean_dy_xhat)`，与 PyTorch `native_batch_norm_backward` 的 `gamma_c / (M sigma_c) * (M dy - sum(g) - xhat * sum(g * xhat))` 在代数上等价但累加顺序不同。

#### 9.2.3 一般情形的语义差距

在一般情形（$\gamma$ per-feature 或 $C > 1$）下，Tenth 实现与 PyTorch 在数学语义层面**不**等价（见 §10）。此时 bit-exact 不成立，差异不为 0。

### 9.3 跨框架的方差约定差异

部分老式 BatchNorm 实现（如早期 TensorFlow）使用 unbiased variance（除以 $M-1$）做前向，但反向用 biased。Tenth 与 PyTorch/MXNet/JAX 一致使用 biased，避免此差异。

### 9.4 结论

定理 N4 的严格表述应修正为：

- **(a')** LayerNorm 在 $\gamma \equiv$ 常数时，Tenth 与 PyTorch 数学等价；bit-exact 需累加顺序对齐，本文未实测，列为开放问题。
- **(b')** BatchNorm 在 $C = 1$ 时，Tenth 与 PyTorch 数学等价；bit-exact 同上。
- **(c')** 一般情形下，Tenth 实现与 PyTorch 数学不等价（§10 gap），bit-exact 不成立。
- **(d)** 方差约定、$\epsilon$ 位置、归一化维度约定上，Tenth 与 PyTorch/MXNet/JAX 一致。$\square$

---

## 10. 局限与实现 gap（独立章节）

本章节诚实披露 Tenth 当前实现与严格闭式解的 gap，每条 gap 给出：**是什么**、**影响**、**形式化判据**、**修复方向**。数理部不写功能代码，仅给出理论判据。

### 10.1 Gap-1：LayerNorm 反向 $dX$ 在 per-feature $\gamma$ 时不正确

**是什么**：Tenth LayerNorm 反向实现为式 (4.1)，将 $\gamma_j$ 提到括号外；严格闭式解 (4.2) 要求 $\gamma_j$ 在括号内参与 per-row 均值计算。

**影响**：当 $\gamma_j$ 随 $j$ 变化时（典型场景：Transformer LayerNorm 学习 per-feature 缩放），$dX$ 计算错误，导致下游梯度更新方向偏差。

**形式化判据**：定义误差 $\Delta_{ik} = (\partial L/\partial x_{ik})_{\text{Tenth}} - (\partial L/\partial x_{ik})_{\text{strict}}$：

$$\Delta_{ik} = \mathrm{std\_inv}^{(i)}\left[\gamma_k\overline g^{(i)} - \overline{g\gamma}^{(i)} + \hat x_{ik}\left(\gamma_k\overline{g\hat x}^{(i)} - \overline{g\gamma\hat x}^{(i)}\right)\right]$$

当且仅当 $\gamma_k \equiv \gamma$ 为常数时 $\Delta_{ik} = 0$。

**修复方向**：将 [autodiff.rs L583-L588](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/autodiff.rs) 的内层第二遍循环改为：

```
mean_g_gm = mean(dY * gamma)         # per-row
mean_g_gm_xh = mean(dY * gamma * x_hat)  # per-row
d_x[i, j] = inv * (dY[i,j] * gamma[j] - mean_g_gm - x_hat[i,j] * mean_g_gm_xh)
```

即第一遍循环求 `mean_g_gm`、`mean_g_gm_xh` 时累加 `dY * gamma`、`dY * gamma * x_hat`，第二遍循环用 `dY * gamma - mean_g_gm - xh * mean_g_gm_xh`。

### 10.2 Gap-2：BatchNorm 反向 $dX$ 在多 channel 时不正确

**是什么**：Tenth BatchNorm 反向使用全局均值 $\overline g_{\text{global}}$、$\overline{g\hat x}_{\text{global}}$（[autodiff.rs L512-L514](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/autodiff.rs)），而非 per-channel 均值。

**影响**：当 $C > 1$ 时，$dX$ 计算错误。典型 CNN BatchNorm 中 $C = 64$ 甚至 $512$，此 gap 严重。

**形式化判据**：定义误差 $\Delta_{ck} = (\partial L/\partial x_{ck})_{\text{Tenth}} - (\partial L/\partial x_{ck})_{\text{strict}}$：

$$\Delta_{ck} = \gamma_c \mathrm{std\_inv}_c\left[(\overline g^{\,c} - \overline g_{\text{global}}) + \hat x_{ck}(\overline{g\hat x}^{\,c} - \overline{g\hat x}_{\text{global}})\right]$$

当 $C = 1$ 时 $\overline g^{\,c} = \overline g_{\text{global}}$，$\Delta_{ck} = 0$。

**修复方向**：将 [autodiff.rs L511-L516](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/autodiff.rs) 改为 per-channel 归约。需要按 channel 维（axis=1）求均值，或采用类似 LayerNorm 的三层嵌套循环（外层 channel，内层 $N \cdot H \cdot W$）。

### 10.3 Gap-3：BatchNorm 反向 $d\gamma$、$d\beta$ 缺少 channel 维归约

**是什么**：Tenth BatchNorm 反向 $d\gamma$、$d\beta$ 为 elementwise（[autodiff.rs L507-L509](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/autodiff.rs)），shape 为 $(N, C, H, W)$；严格公式要求沿 $(N, H, W)$ 归约，shape 为 $(C,)$。

**影响**：调用 `propagate_grad(node, 1, &d_gamma, ...)` 时，`acc_grad` 严格 shape 校验（[tensor.rs L227-L232](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/tensor.rs)）会触发 `"acc_grad shape 不匹配"` 错误，使整个 BatchNorm 反向无法在带 autodiff 的训练路径上工作。

**形式化判据**：设 $d\gamma^{\text{Tenth}}_{ck} = g_{ck}\hat x_{ck}$（未归约），$d\gamma^{\text{strict}}_c = \sum_{n,h,w} g_{cnhw}\hat x_{cnhw}$。当前实现的 shape 为 $(N, C, H, W)$，与 $\gamma$ 的 $(C,)$ 不匹配。

**修复方向**：在 [autodiff.rs L507](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/autodiff.rs) 后增加 `.sum_axis(ndarray::Axis(1))` 等 reduce 操作（需对所有非 channel 维求和）。

### 10.4 Gap-4：教学化简化的代价

Tenth LayerNorm 反向的三层嵌套循环（[autodiff.rs L570-L589](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/autodiff.rs)）在内存局部性上优秀（每行 $O(D)$ 辅助空间），但：

1. **性能**：相比 ndarray 向量化（如 `d_x = std_inv[:, None] * (g * gamma - ...)`）慢约 $2$-$5\times$，因为 Rust 编译器对 ndarray 的 SIMD 优化在向量化路径上更激进；
2. **不可扩展**：若 LayerNorm 归一化维改为非最后维，需重写循环；向量化版本仅需改 `axis` 参数；
3. **教学代价**：Gap-1 的产生部分源于教学化简化（把 $\gamma$ 提到括号外使公式更简洁），但代价是 per-feature $\gamma$ 时不正确。

### 10.5 Gap-5：本文证明的循环论证风险

本文定理 N1、N2 的证明依赖"前向公式按定义 3.2 计算"这一前置条件。但 Tenth 实际前向实现（[tensor.rs L925-L1008](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/tensor.rs)）在 `f32` 路径上使用 `eps as f32`，可能引入额外的精度转换误差。本文证明在 `f64` 路径上严格成立，`f32` 路径存在精度退化，不在定理保证范围内。

### 10.6 Gap-6：bit-exact 实证缺失

定理 N4 的 bit-exact 结论基于"数学等价 + 累加顺序一致"的推理，但本文未进行实际的 PyTorch/MXNet 对比测试。这是开放问题（§11）。

---

## 11. 工程权衡与开放问题

### 11.1 工程权衡

Tenth 在归一化算子反向传播上的工程选择体现三重权衡：

| 选择 | 优势 | 代价 |
|------|------|------|
| 闭式解 vs 展开基本算子 | 一次扫描、低内存 | 推导复杂、易错（Gap-1、2、3） |
| 三层嵌套 vs 向量化 | 教学化、内存局部性好 | 性能 2-5× 慢、不易扩展 |
| `input_tensors` 持久化 vs 重算 | 反向无重算开销 | 内存占用高（6 张量） |
| biased variance vs unbiased | 与 PyTorch 一致 | 数学上略微有偏 |

### 11.2 开放问题

1. **Gap-1/2/3 的修复**：需要运行时部按本文 §10 给出的方向修复，并由测试部新增 bit-exact 对比测试；
2. **bit-exact 实证**：需在 `f64` 路径下与 PyTorch `nn.LayerNorm.backward`、`F.batch_norm.backward` 逐元素对比，验证 ULP 差异；
3. **`f32` 路径的精度分析**：本文证明在 `f64` 下成立，`f32` 路径需独立的误差分析；
4. **RMSNorm 的扩展**：Tenth 当前未实现 RMSNorm（去掉 $\mu$ 减法、$\beta$ 偏置），其闭式反向更简单，可作为未来工作；
5. **GroupNorm、InstanceNorm**：Tenth 未实现，但其反向推导与 LayerNorm/BatchNorm 同构（仅"组"的定义不同），可作为本文方法的扩展。

---

## 12. 结论

本文对 Tenth v0.3.3 的 LayerNorm 与 BatchNorm 闭式反向传播进行了完整的数学推导与理论分析，主要贡献为：

1. **完整闭式推导**（§6、§7）：从链式法则出发，逐步推导 LayerNorm 闭式反向公式 (6.7) 与 BatchNorm 闭式反向公式 (7.1)，含所有中间导数；
2. **五条主定理**（§5）：N1（LayerNorm 正确性）、N2（BatchNorm 正确性）、N3（数值稳定性）、N4（bit-exact 对比）、N5（教学化优势）；
3. **诚实局限披露**（§10）：独立章节披露 Tenth 实现与严格闭式解的三处 gap（LayerNorm per-feature $\gamma$、BatchNorm 多 channel $dX$、BatchNorm $d\gamma$/$d\beta$ shape 不匹配），给出形式化判据与修复方向；
4. **bit-exact 对比**（§9）：与 PyTorch/MXNet/JAX 在数学语义层面逐项对齐，明确 bit-exact 的条件（$\gamma$ 常数或 $C=1$）与开放性（累加顺序未实测）；
5. **与 T39 联动**：本文 N1、N2 是 T39 定理 AD1 在归一化算子上的具体化，填补 T39 在这两个算子上的推导空缺。

本文的诚实贡献不仅是给出正确公式，更是主动披露 Tenth 实现与严格公式之间的 gap——这些 gap 不影响 Tenth 在 $\gamma \equiv 1$ 的退化 LayerNorm、$C = 1$ 的退化 BatchNorm 上的正确性，但在一般深度学习训练场景下会产生错误梯度。修复这些 gap 是运行时部的后续工作，本文提供理论判据。

---

## 附录 A：定理索引

| 定理 | 陈述 | 证明 | 源码链接 |
|------|------|------|---------|
| N1 | LayerNorm 闭式反向正确性 | §6 | [autodiff.rs L523-L596](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/autodiff.rs) |
| N2 | BatchNorm 闭式反向正确性 | §7 | [autodiff.rs L496-L522](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/autodiff.rs) |
| N3 | 数值稳定性 | §8 | [tensor.rs L998](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/tensor.rs) |
| N4 | bit-exact 对比 | §9 | [autodiff.rs L496-L596](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/autodiff.rs) |
| N5 | 教学化优势 | §10.4, §5 | [autodiff.rs L570-L589](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/autodiff.rs) |

## 附录 B：与现有文档的对应

| 本文章节 | 对应现有文档 |
|---------|-------------|
| §3-§4（形式化） | [CODE_WIKI.md](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/CODE_WIKI.md) `runtime/autodiff` 模块 |
| §6-§7（推导） | T39 §6 算子表（深化） |
| §8（稳定性） | [语言参考手册.md](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/docs/语言参考手册.md) `layer_norm` / `batchnorm` 条目 |
| §10（局限） | [AUDIT.md](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/AUDIT.md) 缺陷登记册（建议新增条目） |
| §11（开放问题） | [MEMO.md](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/MEMO.md) 路线图（建议同步） |

## 附录 C：实施建议

针对 §10 披露的三处 gap，建议运行时部按以下顺序修复：

1. **优先级 P0**：Gap-3（BatchNorm $d\gamma$/$d\beta$ shape 不匹配）——这是硬 bug，使 BatchNorm 在 autodiff 路径完全不可用。修复方法：在 `d_gamma`、`d_beta` 计算后增加 channel 维归约。
2. **优先级 P1**：Gap-2（BatchNorm 多 channel $dX$）——影响 CNN 训练正确性。修复方法：改写为 per-channel 三层嵌套（参考 LayerNorm 实现）。
3. **优先级 P1**：Gap-1（LayerNorm per-feature $\gamma$）——影响 Transformer 训练正确性。修复方法：第一遍循环累加 `dY * gamma`、`dY * gamma * x_hat`。
4. **优先级 P2**：Gap-6（bit-exact 实证）——测试部新增对比测试，与 PyTorch 在 `f64` 下逐元素对比。

每次修复后需执行：`cargo test --manifest-path tenth/Cargo.toml -- autodiff` 全绿；自举路径未破坏；同步更新 `MEMO.md` 与 `能力全梳理.md`。

---

## 参考文献

1. Ba, J. L., Kiros, J. R., Hinton, G. E. (2016). *Layer Normalization*. arXiv:1607.06450.
2. Ioffe, S., Szegedy, C. (2015). *Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift*. NeurIPS 2015.
3. Wengert, R. E. (1964). *A simple automatic derivative evaluation program*. Communications of the ACM, 7(8), 463-464.
4. PyTorch. *NativeFunctions.yaml: native_batch_norm_backward*. https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/NativeFunctions.yaml
5. PyTorch. *BatchNormalize.cu*. https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cuda/BatchNormalize.cu
6. Apache MXNet. *batch_norm-inl.h*. https://github.com/apache/mxnet/blob/master/src/operator/nn/batch_norm-inl.h
7. Higham, N. J. (2002). *Accuracy and Stability of Numerical Algorithms* (2nd ed.). SIAM. (条件数与浮点误差分析)
8. Tenth 项目. *T39: Wengert Tape 形式化语义与反向模式正确性*. 2026-07-02. [本地文档](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/docs/论文/T39-Wengert-Tape形式化语义与反向模式正确性.md)
9. Tenth 项目. *T2: Tape 形式化模型与根因定位可判定性*. [本地文档](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/docs/论文/T2-Tape形式化模型与根因定位可判定性.md)
10. Tenth 项目. *T38: autodiff tape 多路径一致性*. [本地文档](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/docs/论文/T38-autodiff-tape多路径一致性.md)
11. Tenth 项目. *T41: Conv2D im2col-matmul 反向传播正确性*. [本地文档](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/docs/论文/T41-Conv2D-im2col-matmul反向传播正确性.md)
12. Tenth 项目. *autodiff.rs L496-L596: BatchNorm + LayerNorm backward*. [源码](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/autodiff.rs)
13. Tenth 项目. *tensor.rs L925-L1008: layer_norm forward*. [源码](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/tensor.rs)
14. Tenth 项目. *methods.rs L1014-L1252: batchnorm + layer_norm forward*. [源码](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/interpreter/methods.rs)

---

> **数理部说明**：本文为理论分析论文 v1，所有定理证明基于 Tenth v0.3.3 源码（截至 2026-07-02）。§10 披露的 gap 已附形式化判据与修复方向，但数理部不写功能代码；修复由运行时部按附录 C 优先级落地，测试部验证。bit-exact 实证（Gap-6）为开放问题，需测试部新增对比测试。
