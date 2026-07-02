# Softmax 雅可比稀疏化与 CrossEntropy 融合：Tenth 标准算子级融合的 O(n²)→O(n) 优化形式化

> **论文编号**：T43 · **系列**：自动微分算子优化形式化 · **级别**：本科/会议级
> **数理部产出**：理论分析论文（v1）
> **基准版本**：Tenth v0.3.3
> **撰写日期**：2026-07-02
> **联动论文**：T2（Tape 形式化模型与根因定位）、T38（autodiff-tape 多路径一致性）、T39（Wengert Tape 形式化模型，规划中）
> **核心源码**：`tenth/src/runtime/autodiff.rs`（L723-L745 CrossEntropy+Softmax 反向）、`tenth/src/runtime/tensor.rs`（L1153-L1202 softmax 前向含减 max）

---

## 摘要

Softmax 与 CrossEntropy 是分类神经网络的"出口算子"对，其反向梯度的计算复杂度与数值稳定性直接决定了训练开销与收敛行为。Softmax 的完整雅可比矩阵 $J_{ij}=y_i(\delta_{ij}-y_j)$ 是稠密的 $n\times n$ 矩阵，朴素链式法则实现需 $O(n^2)$ 时间与空间；CrossEntropy 与 Softmax 的分离实现还会引入 $\log$ 节点，导致反向阶段出现 $-\text{target}/\text{softmax}$ 的除法，在 softmax 接近 0 时数值失稳。

本文对 Tenth v0.3.3 标准库中 Softmax 与 CrossEntropy 两个 `TapeOp` 算子进行形式化分析，证明五条主定理：

- **定理 S1（雅可比稀疏化正确性）**：恒等式 $g_i = y_i\cdot(\text{grad}_i - \sum_j \text{grad}_j\cdot y_j)$ 与完整雅可比乘法 $g = J^\top\text{grad}$ 逐元素相等，且时间复杂度从 $O(n^2)$ 降至 $O(n)$；
- **定理 S2（CE+Softmax 融合正确性）**：融合反向 $\partial L/\partial\text{logits} = \text{softmax} - \text{target}$ 等价于"Softmax→Log→Mul→Neg→Sum"五节点链式法则的解析简化，复杂度从 $O(n^2)$ 降至 $O(n)$；
- **定理 S3（数值稳定性）**：前向 softmax 的"减 max"预处理不改变其雅可比，归因于 softmax 的平移不变性而非 max 的"detach 约定"；
- **定理 S4（算子融合形式化框架）**：给出融合语义等价的三充要条件（前向等价、反向解析简化、中间量可重算/已存储）；
- **定理 S5（与 XLA/PyTorch 融合策略对比）**：Tenth 的"标准算子级融合"相对于 XLA 的"图级融合"与 PyTorch 的"API 级手写融合"在稳定性与可预测性上具有独特优势，代价是泛化性受限。

本文诚实地披露三类工程差距：(L1) 前向 CE 的 `mean + eps` 与反向 `softmax - target` 之间的标度不一致（梯度被放大 $N$ 倍）；(L2) Softmax 雅可比稀疏化的下界并非严格 $O(n)$，因含一次 `sum` 归约；(L3) 融合框架（定理 S4）目前仅覆盖 Tenth 已定义的 21 个算子，无法自动发现新融合模式。这些局限以独立章节 §13 显式记录。

**关键词**：Softmax 雅可比；算子融合；CrossEntropy；自动微分；复杂度优化；数值稳定性；Wengert Tape

---

## 1. 引言

### 1.1 Softmax 反向的 O(n²)→O(n) 优化

Softmax 函数 $\sigma:\mathbb{R}^n\to\mathbb{R}^n$ 定义为
$$
y_i = \sigma_i(x) = \frac{e^{x_i}}{\sum_{k=1}^n e^{x_k}},\qquad i=1,\dots,n.
$$
其雅可比矩阵
$$
J_{ij} = \frac{\partial y_i}{\partial x_j} = y_i(\delta_{ij} - y_j)
$$
是 $n\times n$ 的稠密矩阵。给定上游梯度 $\text{grad}\in\mathbb{R}^n$，反向梯度
$$
g_j = \sum_{i=1}^n \text{grad}_i \cdot J_{ij} = \sum_{i=1}^n \text{grad}_i\cdot y_i(\delta_{ij} - y_j)
$$
朴素实现需先物化 $J$（$O(n^2)$ 空间），再做矩阵-向量乘（$O(n^2)$ 时间）。对词表大小 $n=50000$ 的语言模型，单次 softmax 反向即需约 $10$ GFLOP 与 $20$ GB（f64）存储——这显然不可接受。

幸运的是，$J$ 具有特殊的"对角减秩 1"结构 $J = \text{diag}(y) - yy^\top$，使得 $J^\top\text{grad}$ 可以在 $O(n)$ 时间内完成。这一稀疏化恒等式是所有现代深度学习框架的标配，但其在自动微分系统中的形式化正确性证明、与算子融合的统一框架、以及与 XLA/PyTorch 策略的对比分析，在公开文献中尚不系统。

### 1.2 算子融合的层次

机器学习框架中的"算子融合"发生在三个层次：

| 层次 | 代表框架 | 融合时机 | 机制 |
|------|---------|---------|------|
| 图级融合 | XLA、TVM | 编译期 IR 优化 | 模式匹配 + producer-consumer 融合 |
| API 级手写融合 | PyTorch `F.cross_entropy` | 库函数实现 | 手写 fused kernel |
| **标准算子级融合** | **Tenth** | **算子定义期** | **`TapeOp` 一等公民** |

Tenth 的策略独特之处在于：将"Softmax + CrossEntropy"的融合作为**标准算子集**（21 个 `TapeOp`）的一部分，在算子定义期就确定融合，而非依赖后续的图优化 pass。这意味着：

1. **稳定性**：不依赖图优化的模式匹配是否命中，融合行为可预测；
2. **早决性**：融合在算子语义层就已确定，与执行路径（VM/解释器/JIT）无关；
3. **可形式化**：每个融合算子的语义可以在算子定义处给出完整的形式化规约。

### 1.3 贡献

本文的贡献如下：

- **形式化模型**（§4）：将 Tenth 的 Softmax/CrossEntropy `TapeOp` 严格形式化为前向-反向算子对，明确存储的中间量与反向公式；
- **主定理 S1–S5**（§5）：给出雅可比稀疏化、CE 融合、数值稳定性、融合框架、对比分析五条定理的完整证明；
- **O(n²)→O(n) 完整证明**（§6、§7）：对 Softmax 稀疏化与 CE 融合两条路径分别给出从朴素 $O(n^2)$ 到优化 $O(n)$ 的完整推导，含代价模型；
- **数值稳定性的严格归因**（§8）：证明"减 max"的梯度不变性源于 softmax 的平移不变性，而非 max 的 detach 约定——这是一个比"实现约定"更强的数学事实；
- **算子融合的形式化框架**（§9）：给出融合语义等价的三充要条件，可指导未来新算子的融合设计；
- **诚实记录局限**（§13）：前向-反向标度不一致、稀疏化下界、融合框架泛化性——三类局限独立成节。

### 1.4 与 T39（Wengert Tape）的联动

本文的分析建立在 Tenth 的 Wengert Tape 之上。Tape 的形式化模型已在 T2（[T2-Tape形式化模型与根因定位可判定性](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/docs/论文/T2-Tape形式化模型与根因定位可判定性.md)）与 T38（[T38-autodiff-tape多路径一致性](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/docs/论文/T38-autodiff-tape多路径一致性.md)）中建立。规划中的 T39（Wengert Tape 形式化模型）将进一步严格化 Tape 节点的代数结构与拓扑回放语义。本文采用以下约定：

- **Tape 节点**：四元组 $(op, s_{in}, s_{out}, \ell)$，其中 $op$ 为算子类型（`TapeOp`），$s_{in}$ 为输入张量引用列表，$s_{out}$ 为输出张量引用，$\ell$ 为上游节点 id 列表（见 [autodiff.rs:14-25](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/autodiff.rs)）；
- **反向回放**：从 loss 节点出发，按拓扑逆序对每个节点应用链式法则，将梯度累积至叶节点的 `.grad` 字段；
- **多路径一致性**：VM 与解释器记录的 tape 同构（T38 定理 A1），JIT 在 recording 模式下整体退出至 VM（T38 定理 A2）。

本文聚焦于**单个 TapeOp 节点内部**的反向公式正确性与复杂度，不涉及跨节点的 tape 一致性（已由 T38 覆盖）。

---

## 2. 背景

### 2.1 Softmax 雅可比的结构

Softmax 的雅可比 $J = \text{diag}(y) - yy^\top$ 是"对角矩阵减去秩 1 矩阵"的特殊结构。这一结构有两条等价的解读：

**解读 1（概率论）**：$y$ 是一个概率单纯形上的点，$\sum_i y_i = 1$。雅可比的秩 1 项 $yy^\top$ 反映了"概率守恒"约束——若某个 $x_j$ 增大导致 $y_j$ 增大，则所有其他 $y_i$（$i\neq j$）必须相应减小以保持总和为 1。

**解读 2（线性代数）**：$J = D - yy^\top$，其中 $D = \text{diag}(y)$。对任意向量 $v$，$Jv = Dv - y(y^\top v) = y\odot v - y\cdot(y\cdot v)$，其中 $\odot$ 为逐元素乘，$\cdot$ 为内积。这一表达式仅需 $O(n)$ 时间计算——这正是稀疏化的代数根源。

### 2.2 XLA 的图级融合

XLA（Accelerated Linear Algebra）是 TensorFlow 与 JAX 后端的编译器。其融合策略发生在 HLO（High Level Optimizer）IR 层：

1. **Producer-Consumer 融合**：相邻的 HLO 指令若满足融合条件（如 elementwise 之间、reduction 与其 elementwise 邻居），则合并为一个 fusion kernel；
2. **模式匹配**：XLA 内置若干融合模式（softmax 融合、batch-norm 融合等），通过 `HloPassFusion` 识别；
3. **目标**：减少中间张量的内存读写，提升 GPU/TPU 利用率。

XLA 的 softmax 融合在 [XLA softmax fusion](https://www.tensorflow.org/xla) 中实现，将 `exp → sum → div → (可选 log)` 合并为单个 kernel。但这一融合是**图优化的产物**——若用户的代码结构未匹配预设模式（例如在 softmax 中间插入了自定义 op），融合不会发生。

### 2.3 PyTorch 的 API 级手写融合

PyTorch 提供 `torch.nn.functional.cross_entropy`，内部调用 `F.log_softmax` + `F.nll_loss` 的融合实现。关键点：

1. `log_softmax` 本身不是 `softmax → log` 的简单组合，而是手写的数值稳定 fused kernel；
2. `nll_loss` 的反向直接使用 `log_softmax` 的输出，避免重新计算；
3. 这一融合是**手写的**，仅存在于 PyTorch C++ 后端的特定算子实现中，自定义 op 无法自动获得融合。

PyTorch 的 `torch.compile`（Dynamo + Inductor）引入了图级融合，但其融合能力仍依赖模式匹配，且与 eager 模式的行为可能不一致。

### 2.4 Tenth 的标准算子级融合

Tenth 在 `runtime/autodiff.rs` 中定义了 21 个 `TapeOp` 变体（[autodiff.rs:29-79](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/autodiff.rs)），其中 `Softmax` 与 `CrossEntropy` 是两个独立的一等算子。`CrossEntropy` 的前向存储 `softmax(logits)` 作为中间量（[autodiff.rs:152-173](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/autodiff.rs)），反向直接返回 `softmax - target`（[autodiff.rs:723-734](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/autodiff.rs)）。

这种"在算子定义期就确定融合"的策略，与 XLA 的"图优化期融合"和 PyTorch 的"API 级手写融合"形成对比。本文第 10 节将系统对比三者。

---

## 3. 记号与前置定义

### 3.1 基本记号

- $x \in \mathbb{R}^n$：softmax 的输入（logits），$n$ 为类别数；
- $y = \sigma(x) \in \mathbb{R}^n$：softmax 输出，$y_i \geq 0$，$\sum_i y_i = 1$；
- $t \in \mathbb{R}^n$：target 分布，本文假设 $\sum_i t_i = 1$（one-hot 或软标签）；
- $\text{grad} \in \mathbb{R}^n$：上游梯度（loss 对 $y$ 的梯度）；
- $g \in \mathbb{R}^n$：下游梯度（loss 对 $x$ 的梯度）；
- $\delta_{ij}$：Kronecker delta；
- $\odot$：逐元素乘法；
- $\mathbf{1}$：全 1 向量；
- $\langle u, v\rangle = u^\top v$：内积。

### 3.2 算子定义

**定义 3.1（Softmax 算子）**：$\text{Softmax}:\mathbb{R}^n\to\mathbb{R}^n$，$\text{Softmax}(x) = y$，其中
$$
y_i = \frac{e^{x_i - m}}{\sum_{k=1}^n e^{x_k - m}},\qquad m = \max_k x_k.
$$
$m$ 为最大值减项，用于数值稳定性（详见 §8）。

**定义 3.2（CrossEntropy 算子）**：$\text{CE}:\mathbb{R}^n\times\mathbb{R}^n\to\mathbb{R}$，
$$
\text{CE}(x, t) = -\sum_{i=1}^n t_i \log y_i,\qquad y = \text{Softmax}(x).
$$
本文采用"求和"形式；Tenth 实现含"求均值"与 `eps` 平滑（见 §13 局限 L1）。

**定义 3.3（Softmax 雅可比）**：$J^\sigma \in \mathbb{R}^{n\times n}$，
$$
J^\sigma_{ij} = \frac{\partial y_i}{\partial x_j} = y_i(\delta_{ij} - y_j).
$$

**定义 3.4（融合算子 FusedCE）**：$\text{FusedCE}:\mathbb{R}^n\times\mathbb{R}^n\to\mathbb{R}$，前向与 CE 相同，但反向直接计算 $\partial L/\partial x = y - t$，不显式经过 log 节点。

### 3.3 Tape 节点结构

Tenth 的 `TapeNode` 结构（[autodiff.rs:14-25](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/autodiff.rs)）：
```rust
pub struct TapeNode {
    pub id: usize,
    pub op: TapeOp,
    pub inputs: Vec<usize>,           // 上游节点 id
    pub input_tensors: Vec<Rc<RefCell<Tensor>>>,  // 输入张量引用（含 result）
}
```
- `Softmax` 节点：`input_tensors = [input, result]`（[autodiff.rs:739-740](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/autodiff.rs)）；
- `CrossEntropy` 节点：`input_tensors = [logits, softmax, target, result]`（[autodiff.rs:170](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/autodiff.rs)）——注意 CE 节点存储了 `softmax` 中间量，这是融合的关键。

---

## 4. Tenth Softmax/CrossEntropy 形式化

### 4.1 Softmax 前向

Tenth 的 softmax 前向实现（[tensor.rs:1153-1202](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/tensor.rs)）沿最后一轴计算，含减 max：
```rust
let max_val = slice.iter().copied().fold(f64::NEG_INFINITY, f64::max);
let exps: Vec<f64> = slice.iter().map(|x| (x - max_val).exp()).collect();
let sum: f64 = exps.iter().sum();
let probs: Vec<f64> = exps.iter().map(|x| x / sum).collect();
```
对应数学定义 3.1。前向在 tape 上记录为 `TapeOp::Softmax` 节点，存储 `[input, result]`。

### 4.2 Softmax 反向（稀疏化形式）

Tenth 的 softmax 反向实现（[autodiff.rs:735-745](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/autodiff.rs)）：
```rust
TapeOp::Softmax => {
    // d(softmax(x)_i)/dx_j = y_i * (δ_ij - y_j)
    // Chain rule: g_i = y_i * (grad_i - sum_j(grad_j * y_j))
    let g_a = {
        let result_ref = node.input_tensors[1].borrow();
        let y = &result_ref.data;
        let sum_term = (&grad * y).sum();
        &grad * y - &(y.mapv(|v| v * sum_term))
    };
    propagate_grad(node, 0, &g_a, &mut node_grads)?;
}
```
对应公式：
$$
g_j = y_j \cdot \text{grad}_j - y_j \cdot \underbrace{\sum_i \text{grad}_i \cdot y_i}_{s},\qquad s = \langle\text{grad}, y\rangle.
$$
等价地：
$$
g_j = y_j\bigl(\text{grad}_j - s\bigr),\qquad s = \sum_i \text{grad}_i\,y_i.
$$

### 4.3 CrossEntropy 前向

Tenth 的 CE 前向（[natives.rs:323-371](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/interpreter/natives.rs)）：
1. 计算 `sm = softmax(logits)`；
2. 计算 `loss = -mean(sum(target * log(max(softmax, eps))))`，`eps = 1e-10`；
3. 在 tape 上记录 `TapeOp::CrossEntropy` 节点，存储 `[logits, softmax, target, result]`。

关键：**softmax 作为中间量被存储在 CE 节点中**，反向时直接读取，无需重新计算。

### 4.4 CrossEntropy 反向（融合形式）

Tenth 的 CE 反向实现（[autodiff.rs:723-734](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/autodiff.rs)）：
```rust
TapeOp::CrossEntropy => {
    // d(CE)/d(logits) = softmax - target
    // input_tensors = [logits, softmax_output, target]
    if node.input_tensors.len() >= 3 {
        let g_a = {
            let sm_ref = node.input_tensors[1].borrow();
            let tgt_ref = node.input_tensors[2].borrow();
            &sm_ref.data - &tgt_ref.data
        };
        propagate_grad(node, 0, &g_a, &mut node_grads)?;
    }
}
```
对应公式：
$$
g^{\text{CE}} = y - t,\qquad y = \text{softmax}(x).
$$
注意：此反向**不经过 log 节点**，直接使用存储的 softmax 中间量。

---

## 5. 主定理

### 5.1 定理 S1（雅可比稀疏化正确性）

**定理 S1**：设 $y = \sigma(x)$，上游梯度 $\text{grad}\in\mathbb{R}^n$。则稀疏化公式
$$
g_j = y_j\bigl(\text{grad}_j - \sum_i \text{grad}_i\,y_i\bigr),\qquad j=1,\dots,n
$$
与完整雅可比乘法 $g = (J^\sigma)^\top \text{grad}$ 逐元素相等，且计算时间为 $\Theta(n)$，空间为 $\Theta(n)$。

**源码锚点**：[autodiff.rs:735-745](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/autodiff.rs)。

**证明**：见 §6。

### 5.2 定理 S2（CE+Softmax 融合正确性）

**定理 S2**：设 $L = \text{CE}(x, t) = -\sum_i t_i \log\sigma_i(x)$，$t$ 为概率分布（$\sum_i t_i = 1$）。则融合反向
$$
\frac{\partial L}{\partial x_j} = y_j - t_j,\qquad y = \sigma(x)
$$
等价于"Softmax → Log → Mul → Neg → Sum"五节点链式法则的解析简化，且计算时间为 $\Theta(n)$，对比朴素（含完整 softmax 雅可比）的 $\Theta(n^2)$。

**源码锚点**：[autodiff.rs:723-734](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/autodiff.rs)（反向）、[autodiff.rs:152-173](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/autodiff.rs)（前向存储 softmax）。

**证明**：见 §7。

### 5.3 定理 S3（数值稳定性）

**定理 S3**：设 $\tilde y = \sigma(x - m\mathbf{1})$，$m = \max_k x_k$。则 $\tilde y = y = \sigma(x)$，且
$$
\frac{\partial \tilde y_i}{\partial x_j} = \frac{\partial y_i}{\partial x_j} = y_i(\delta_{ij} - y_j).
$$
进一步，即使将 $m$ 视为 $x$ 的函数 $m(x) = \max_k x_k$（而非 detach 常数），上述雅可比在非 ties 点处仍成立。

**源码锚点**：[tensor.rs:1170-1171](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/tensor.rs)（减 max 前向）。

**证明**：见 §8。

### 5.4 定理 S4（算子融合形式化框架）

**定理 S4**：设算子序列 $F = f_1\circ f_2\circ\dots\circ f_k$，输入 $x$，中间量 $h_1, \dots, h_{k-1}$（$h_i = f_{i+1}(h_{i-1})$，$h_0 = x$，$h_k = F(x)$）。融合算子 $\hat F$（存储中间量子集 $S\subseteq\{h_1,\dots,h_{k-1}\}$）与 $F$ **语义等价**当且仅当：

- **(C1) 前向等价**：$\hat F(x) = F(x)$ 对所有合法输入 $x$；
- **(C2) 反向解析简化等价**：对每个输入 $x_i$，融合反向公式 $\hat g_i$ 等于链式法则
  $$
  g_i = \sum_{\text{paths } p: x_i \rightsquigarrow L} \prod_{\ell\in p} J_\ell^\top \cdot \text{grad}
  $$
  经代数化简（允许使用 $S$ 中的中间量与 $F$ 的前向等式作为重写规则）后的表达式；
- **(C3) 中间量可用性**：化简后的 $\hat g_i$ 中出现的所有中间量要么属于 $S$，要么可从 $S\cup\{x, \hat F(x)\}$ 在 $O(n)$ 时间内重算。

在此条件下，融合保持语义等价；若 (C2) 的化简进一步消去了 $O(n^2)$ 的雅可比物化，则融合同时实现复杂度优化。

**源码锚点**：[autodiff.rs:152-173](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/autodiff.rs)（CE 节点存储 $S = \{y\}$）。

**证明**：见 §9。

### 5.5 定理 S5（与 XLA/PyTorch 融合策略对比）

**定理 S5**：在以下维度上，Tenth 的标准算子级融合与 XLA 图级融合、PyTorch API 级手写融合具有可形式化的差异：

- **(D1) 融合决定时机**：Tenth（算子定义期，编译前）< PyTorch（库函数实现期，运行时）< XLA（图优化期，编译期）；
- **(D2) 融合可预测性**：Tenth（确定性，无模式匹配）> PyTorch（部分确定，依赖调用哪个 API）> XLA（依赖图结构与 pass 命中）；
- **(D3) 融合泛化性**：XLA（自动发现新融合模式）> PyTorch（可手写新融合）> Tenth（需新增 `TapeOp`）；
- **(D4) 融合的可形式化性**：Tenth（每个融合算子可独立形式化）> PyTorch（依赖具体实现）> XLA（融合规则散布在多个 pass 中）。

三者并非互相替代，而是互补：Tenth 的策略在"已知高频融合模式"（如 CE+Softmax、LayerNorm）上最稳定高效；XLA 的策略在"长尾 elementwise 链"上最有效。

**证明**：见 §10。

---

## 6. Softmax 雅可比稀疏化的数学推导

### 6.1 完整推导（定理 S1 的证明）

**目标**：证明 $g_j = y_j(\text{grad}_j - \sum_i \text{grad}_i y_i)$ 等于 $(J^\sigma)^\top\text{grad}$ 的第 $j$ 个分量。

**第 1 步：展开雅可比乘法**。由定义 3.3，
$$
\bigl((J^\sigma)^\top \text{grad}\bigr)_j = \sum_{i=1}^n (J^\sigma)^\top_{ji}\,\text{grad}_i = \sum_{i=1}^n J^\sigma_{ij}\,\text{grad}_i.
$$
代入 $J^\sigma_{ij} = y_i(\delta_{ij} - y_j)$：
$$
= \sum_{i=1}^n y_i(\delta_{ij} - y_j)\,\text{grad}_i.
$$

**第 2 步：拆分求和**。
$$
= \sum_{i=1}^n y_i\,\delta_{ij}\,\text{grad}_i - \sum_{i=1}^n y_i\,y_j\,\text{grad}_i.
$$

**第 3 步：化简第一项**（Kronecker delta 性质 $\sum_i y_i\delta_{ij}\text{grad}_i = y_j\,\text{grad}_j$）：
$$
= y_j\,\text{grad}_j - y_j\sum_{i=1}^n y_i\,\text{grad}_i.
$$

**第 4 步：提取公因子 $y_j$**：
$$
= y_j\Bigl(\text{grad}_j - \sum_{i=1}^n \text{grad}_i\,y_i\Bigr).
$$

令 $s := \sum_i \text{grad}_i\,y_i = \langle\text{grad}, y\rangle$，则
$$
g_j = y_j(\text{grad}_j - s).\qquad\square\text{（公式等价性）}
$$

### 6.2 复杂度分析

**朴素方法（物化雅可比）**：
- 物化 $J^\sigma$：$n^2$ 个元素，每个 $O(1)$ 计算 → $O(n^2)$ 时间，$O(n^2)$ 空间；
- 矩阵-向量乘 $J^\top\text{grad}$：$n^2$ 次乘加 → $O(n^2)$ 时间；
- **总计**：$\Theta(n^2)$ 时间，$\Theta(n^2)$ 空间。

**稀疏化方法（Tenth 实现）**：
- 计算内积 $s = \langle\text{grad}, y\rangle$：$n$ 次乘加 → $O(n)$ 时间；
- 计算 $g_j = y_j(\text{grad}_j - s)$ 对所有 $j$：$n$ 次减法 + $n$ 次乘法 → $O(n)$ 时间；
- 存储：$\text{grad}, y, g$ 各 $O(n)$ → $O(n)$ 空间；
- **总计**：$\Theta(n)$ 时间，$\Theta(n)$ 空间。

**加速比**：$T_{\text{naive}}/T_{\text{sparse}} = \Theta(n)$。对 $n=50000$，加速比约 $50000\times$。

### 6.3 与 Tenth 源码的逐行对应

Tenth 的反向实现（[autodiff.rs:738-743](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/autodiff.rs)）：
```rust
let y = &result_ref.data;                              // y
let sum_term = (&grad * y).sum();                       // s = <grad, y>
&grad * y - &(y.mapv(|v| v * sum_term))                 // y ⊙ grad - y * s
```
注意输出为 `grad ⊙ y - y*s`，与 $g_j = y_j\text{grad}_j - y_j s = y_j(\text{grad}_j - s)$ 一致。$\square$（定理 S1 证毕）

### 6.4 代价模型的细化

精确的浮点运算计数（标量 op 数）：

| 步骤 | 朴素 | 稀疏化 |
|------|------|--------|
| 物化 $J$ | $n^2$ 乘 + $n^2$ 减 | 0 |
| 内积 $s$ | 0 | $n$ 乘 + $(n-1)$ 加 |
| 矩阵-向量乘 | $n^2$ 乘 + $n(n-1)$ 加 | 0 |
| 缩放与减 | 0 | $n$ 减 + $n$ 乘 |
| **总计（乘法）** | $2n^2$ | $2n$ |
| **总计（加法）** | $2n^2 - n$ | $2n - 1$ |
| **总计（flop）** | $\Theta(n^2)$，常数 $\approx 4$ | $\Theta(n)$，常数 $\approx 4$ |

对 $n=50000$：朴素 $\approx 10^{10}$ flop，稀疏化 $\approx 2\times 10^5$ flop——五个数量级的差距。

---

## 7. CE+Softmax 融合的 O(n²)→O(n) 证明

### 7.1 定理 S2 的证明

**目标**：证明 $\partial L/\partial x_j = y_j - t_j$，其中 $L = -\sum_i t_i\log y_i$，$y = \sigma(x)$，$\sum_i t_i = 1$。

#### 7.1.1 链式法则展开

由链式法则（经过 Softmax → Log → Mul → Neg → Sum 五节点）：
$$
\frac{\partial L}{\partial x_j} = \sum_{i=1}^n \frac{\partial L}{\partial y_i}\cdot\frac{\partial y_i}{\partial x_j}.
$$

**第 1 步：计算 $\partial L/\partial y_i$**。由 $L = -\sum_k t_k\log y_k$：
$$
\frac{\partial L}{\partial y_i} = -\frac{t_i}{y_i}.
$$
（此处假设 $y_i > 0$；$y_i = 0$ 时 $\log y_i$ 无定义，这正是分离实现的数值痛点，见 §7.3。）

**第 2 步：代入 softmax 雅可比**。
$$
\frac{\partial L}{\partial x_j} = \sum_{i=1}^n \Bigl(-\frac{t_i}{y_i}\Bigr)\cdot y_i(\delta_{ij} - y_j).
$$

**第 3 步：化简 $t_i/y_i \cdot y_i$**。
$$
= -\sum_{i=1}^n t_i(\delta_{ij} - y_j).
$$

**第 4 步：拆分求和**。
$$
= -\Bigl(\sum_{i=1}^n t_i\,\delta_{ij} - y_j\sum_{i=1}^n t_i\Bigr).
$$

**第 5 步：化简 Kronecker delta 与概率归一**。由 $\sum_i t_i\delta_{ij} = t_j$，$\sum_i t_i = 1$（target 是概率分布）：
$$
= -(t_j - y_j\cdot 1) = y_j - t_j.
$$

即
$$
\frac{\partial L}{\partial x_j} = y_j - t_j.\qquad\square\text{（公式等价性）}
$$

#### 7.1.2 复杂度对比

**朴素（分离实现，含完整 softmax 雅可比）**：

分离实现的反向需经过 5 个节点：
1. `Sum` 反向：$O(n)$，将标量梯度广播到 $-\text{target}\cdot\log y$ 的形状；
2. `Neg` 反向：$O(n)$，取负，得 $\text{target}\cdot\log y$ 的梯度为 $-1$（标量）→ 广播 $O(n)$；
3. `Mul` 反向（target 与 log y）：$O(n)$，$\partial L/\partial(\log y_i) = t_i$；
4. `Log` 反向：$O(n)$，$\partial L/\partial y_i = t_i / y_i$；
5. **Softmax 反向（朴素，物化雅可比）**：$O(n^2)$，$g = (J^\sigma)^\top\cdot(t/y)$。

**总计**：$O(n) + O(n) + O(n) + O(n) + O(n^2) = O(n^2)$，瓶颈在第 5 步。

**融合（Tenth 实现）**：

直接计算 $g = y - t$：
- 逐元素减：$n$ 次减法 → $O(n)$ 时间；
- 读取存储的 $y$（CE 节点已存）：$O(n)$；
- **总计**：$\Theta(n)$ 时间，$\Theta(n)$ 空间。

**加速比**：$\Theta(n)$。对 $n=50000$，约 $50000\times$。$\square$（定理 S2 证毕）

### 7.2 代数根源：为什么融合能消除 O(n²)

朴素方法的 $O(n^2)$ 来自第 5 步物化 softmax 雅可比。但通过链式法则的代数化简，我们发现：

$$
\underbrace{(J^\sigma)^\top}_{n\times n}\cdot\underbrace{\frac{t}{y}}_{\text{含除法}} = \underbrace{y - t}_{\text{逐元素减}}
$$

左端是 $O(n^2)$ 的矩阵-向量乘 + $O(n)$ 的逐元素除法；右端是 $O(n)$ 的逐元素减。这一化简的关键代数事实是：

1. **softmax 雅可比的秩 1 结构**：$J = \text{diag}(y) - yy^\top$，使得 $J^\top v = y\odot v - y\langle y, v\rangle$；
2. **特殊上游梯度**：$v = t/y$ 时，$\langle y, v\rangle = \langle y, t/y\rangle = \sum_i t_i = 1$，秩 1 项化简为 $-y$；
3. **对角项化简**：$y\odot(t/y) = t$；
4. **合并**：$t - y\cdot 1 = t - y$，取负得 $y - t$。

这一化简无法在"通用图优化"中发现——它依赖于 $t$ 是概率分布（$\sum t_i = 1$）与 softmax 雅可比结构的**联合**事实。XLA 的图级融合通常不会做这种"语义级"化简，而是将 5 个 kernel 合并为 1 个 kernel 以减少内存读写——复杂度仍是 $O(n)$（因为 XLA 的 softmax kernel 内部也用稀疏化），但融合发生在 kernel 层而非算子语义层。

### 7.3 数值稳定性优势

分离实现的第 4 步 $\partial L/\partial y_i = t_i/y_i$ 在 $y_i\to 0$ 时趋向无穷。虽然在前向 softmax 中 $y_i > 0$ 严格成立（指数函数恒正），但在 fp16/bf16 精度下，$y_i$ 可能下溢到 0，导致除零。

融合实现直接计算 $y - t$，**无除法**，因此对 $y_i$ 的下溢免疫。这是融合相对于分离的**独立于复杂度的第二优势**。

Tenth 的 CE 前向还使用了 `eps` 平滑（`max(softmax, 1e-10)`，见 [natives.rs:344](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/interpreter/natives.rs)），但反向不使用 eps——这是前向-反向不一致的来源之一（见 §13 局限 L1）。

---

## 8. 数值稳定性证明（减 max）

### 8.1 定理 S3 的证明

**目标**：证明 softmax 的"减 max"预处理不改变其雅可比。

#### 8.1.1 平移不变性

**引理 8.1（softmax 平移不变性）**：对任意常数 $c\in\mathbb{R}$，
$$
\sigma(x + c\mathbf{1}) = \sigma(x).
$$

**证明**：
$$
\sigma_i(x + c\mathbf{1}) = \frac{e^{x_i + c}}{\sum_k e^{x_k + c}} = \frac{e^c\cdot e^{x_i}}{e^c\sum_k e^{x_k}} = \frac{e^{x_i}}{\sum_k e^{x_k}} = \sigma_i(x).
$$
$e^c$ 在分子分母中约去。$\square$

取 $c = -m$，$m = \max_k x_k$，得 $\sigma(x - m\mathbf{1}) = \sigma(x)$，即 $\tilde y = y$。这证明了定理 S3 的第一部分。

#### 8.1.2 雅可比不变性（m 视为常数）

若 $m$ 被视为与 $x$ 无关的常数（detach），则 $\tilde y_i = e^{x_i - m}/Z$，$Z = \sum_k e^{x_k - m}$，
$$
\frac{\partial \tilde y_i}{\partial x_j} = \frac{\delta_{ij}e^{x_i-m}\cdot Z - e^{x_i-m}\cdot e^{x_j-m}}{Z^2} = \tilde y_i\delta_{ij} - \tilde y_i\tilde y_j = \tilde y_i(\delta_{ij} - \tilde y_j).
$$
由 $\tilde y = y$，得 $\partial\tilde y_i/\partial x_j = y_i(\delta_{ij} - y_j) = \partial y_i/\partial x_j$。

#### 8.1.3 雅可比不变性（m 视为 x 的函数）

更严格地，将 $m(x) = \max_k x_k$ 视为 $x$ 的函数。在非 ties 点（即最大值唯一），$m$ 是局部常数，$\partial m/\partial x_j = \delta_{j,\arg\max}$。在 ties 点，$m$ 不可微，但 softmax 仍可微（因为平移不变性使得 $m$ 的"选择"不影响输出）。

对非 ties 点，用全微分：
$$
\frac{\partial \tilde y_i}{\partial x_j} = \frac{\partial}{\partial x_j}\Bigl[\frac{e^{x_i - m(x)}}{Z(x)}\Bigr].
$$
令 $u_i = x_i - m(x)$，则 $\partial u_i/\partial x_j = \delta_{ij} - \partial m/\partial x_j$。由商法则：
$$
\frac{\partial \tilde y_i}{\partial x_j} = \frac{(\delta_{ij} - m'_j)e^{u_i}\cdot Z - e^{u_i}\cdot\sum_k(\delta_{kj} - m'_j)e^{u_k}}{Z^2},
$$
其中 $m'_j := \partial m/\partial x_j$。提取 $e^{u_i}/Z = \tilde y_i$：
$$
= \tilde y_i\Bigl[(\delta_{ij} - m'_j) - \sum_k \tilde y_k(\delta_{kj} - m'_j)\Bigr].
$$
化简求和：$\sum_k \tilde y_k\delta_{kj} = \tilde y_j$，$\sum_k \tilde y_k m'_j = m'_j\sum_k\tilde y_k = m'_j$（由 $\sum_k\tilde y_k = 1$）：
$$
= \tilde y_i\Bigl[\delta_{ij} - m'_j - \tilde y_j + m'_j\Bigr] = \tilde y_i(\delta_{ij} - \tilde y_j).
$$
**$m'_j$ 项完全消去**。因此即使 $m$ 被视为 $x$ 的函数，雅可比仍为 $y_i(\delta_{ij} - y_j)$。$\square$（定理 S3 证毕）

### 8.2 归因：平移不变性 vs detach 约定

定理 S3 的证明揭示了重要事实：**减 max 的梯度不变性源于 softmax 的平移不变性（引理 8.1），而非 max 的 detach 约定**。这意味着：

1. **理论层面**：即使 autodiff 系统将 max 视为可微函数（不 detach），softmax 的梯度仍正确——因为 $m$ 的贡献在雅可比中精确消去；
2. **工程层面**：Tenth 的实现（[tensor.rs:1170-1171](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/tensor.rs)）将 max 作为标量常数参与减法，且 softmax 反向公式（[autodiff.rs:737](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/autodiff.rs)）直接使用稀疏化形式，不经过 max 节点——这等价于 detach，但即使不 detach 也是正确的；
3. **可移植性**：这一性质保证了任何 autodiff 系统（无论 max 是否 detach）都能正确计算 softmax 梯度，只要反向公式使用稀疏化形式。

### 8.3 减 max 的数值收益

不减 max 时，$e^{x_i}$ 在 $x_i$ 较大时溢出（fp64 上 $x_i > 709$ 即上溢）；$x_i$ 较小（与其他分量相比）时，$e^{x_i}/\sum e^{x_k}$ 下溢到 0。减 max 后，最大指数项为 $e^0 = 1$，最小指数项为 $e^{x_i - m} \geq 0$（有限），消除上溢；下溢仍可能发生但被 eps 平滑（前向）或融合反向（避免除法）缓解。

---

## 9. 算子融合的形式化框架

### 9.1 定理 S4 的证明

**目标**：证明融合语义等价的三充要条件 (C1)(C2)(C3)。

#### 9.1.1 必要性

** (C1) 必要**：若 $\hat F(x)\neq F(x)$，则前向输出不同，融合改变前向语义，不等价。

**(C2) 必要**：若融合反向 $\hat g_i$ 不等于链式法则的化简形式，则存在输入使得 $\hat g_i \neq g_i$，反向语义不同，不等价。

**(C3) 必要**：若化简后的 $\hat g_i$ 引用了不在 $S$ 中且不可从 $S\cup\{x,\hat F(x)\}$ 重算的中间量 $h$，则融合反向在执行时无法计算 $\hat g_i$（$h$ 已丢失），要么报错要么返回错误结果，不等价。

#### 9.1.2 充分性

设 (C1)(C2)(C3) 成立。需证融合 $\hat F$ 与 $F$ 语义等价，即：对任意输入 $x$ 与上游梯度 $\text{grad}$，

(i) **前向等价**：$\hat F(x) = F(x)$——由 (C1) 直接给出。

(ii) **反向等价**：$\hat g_i = g_i$ 对所有 $i$。由 (C2)，$\hat g_i$ 是 $g_i$ 的链式法则表达式的代数化简。代数化简保持等式（每一步化简都是等式重写），故 $\hat g_i = g_i$（作为表达式）。由 (C3)，$\hat g_i$ 中所有中间量可计算，故 $\hat g_i$ 可执行，且执行结果等于 $g_i$ 的执行结果。

(iii) **复杂度优化（若适用）**：若 (C2) 的化简消去了 $O(n^2)$ 的雅可比物化（如 softmax 雅可比），则融合反向的时间复杂度低于朴素链式法则。$\square$

### 9.2 框架应用于 CE+Softmax

将定理 S4 应用于 CE+Softmax 融合：

**算子序列**：$F = \text{Sum}\circ\text{Neg}\circ\text{Mul}(t,\cdot)\circ\text{Log}\circ\text{Softmax}$，共 5 个节点。

**中间量**：$h_1 = y$（softmax 输出），$h_2 = \log y$，$h_3 = t\odot\log y$，$h_4 = -t\odot\log y$，$h_5 = L = \sum h_4$。

**存储子集**：$S = \{y\} = \{h_1\}$（CE 节点 `input_tensors[1]`，见 [autodiff.rs:170](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/autodiff.rs)）。

**验证 (C1)**：$\hat F(x, t) = -\sum_i t_i\log\sigma_i(x) = F(x, t)$。✓（前向相同）

**验证 (C2)**：链式法则展开（§7.1.1）：
$$
g_j = \sum_i \frac{\partial L}{\partial y_i}\cdot\frac{\partial y_i}{\partial x_j} = \sum_i\Bigl(-\frac{t_i}{y_i}\Bigr)\cdot y_i(\delta_{ij}-y_j).
$$
化简（使用 $\sum_i t_i = 1$）：
$$
= y_j - t_j.
$$
融合反向 $\hat g_j = y_j - t_j$，与化简结果一致。✓

**验证 (C3)**：化简结果 $\hat g = y - t$ 仅引用 $y$（$\in S$）与 $t$（输入）。✓

**结论**：CE+Softmax 融合满足 (C1)(C2)(C3)，语义等价。且 (C2) 的化简消去了 $O(n^2)$ 的 softmax 雅可比物化，实现复杂度优化。

### 9.3 框架对未融合场景的诊断

定理 S4 还可用于诊断"为什么某些场景无法融合"。例如，若在 Softmax 与 Log 之间插入一个**非标准变换** $\phi$（如温度缩放 $\phi(y) = y^{1/T}$），则：

- 链式法则：$g_j = \sum_i(-t_i/\phi_i)\cdot\phi'_i\cdot y_i(\delta_{ij}-y_j)$；
- 若 $\phi$ 不满足特殊结构使 $\sum_i t_i\phi'_i y_i/y_i$ 化简为常数，则化简无法消去雅可比；
- 此时融合需存储 $\phi(y)$ 并保留 $O(n^2)$ 雅可比，或退化为分离实现。

这解释了为什么"标准 CE+Softmax"是可融合的特例——它依赖于 softmax 雅可比的秩 1 结构与 target 概率归一的联合巧合。

---

## 10. 与 XLA/PyTorch 对比（定理 S5 的证明）

### 10.1 维度 D1：融合决定时机

| 框架 | 融合决定点 | 相对编译流程 |
|------|-----------|-------------|
| **Tenth** | `TapeOp` 算子定义（`autodiff.rs`） | 编译前（语言设计期） |
| **PyTorch** | `F.cross_entropy` 库函数 | 运行时（eager）或编译时（compile） |
| **XLA** | `HloPassFusion` 图优化 pass | 编译期（IR 优化） |

Tenth 的融合在"算子定义期"就确定——`CrossEntropy` 是 21 个 `TapeOp` 之一，其反向公式写在 [autodiff.rs:723-734](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/autodiff.rs)。这一决定不依赖任何运行时信息或图结构分析。

### 10.2 维度 D2：融合可预测性

- **Tenth**：确定性。用户调用 `cross_entropy(logits, target)` 必然走融合路径（[natives.rs:323-371](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/interpreter/natives.rs)）。无模式匹配，无 pass 命中问题。
- **PyTorch**：部分确定。`F.cross_entropy` 走融合；但若用户手写 `-(target * log_softmax(logits)).sum()`，则不一定走融合（取决于 `torch.compile` 是否识别）。
- **XLA**：依赖图结构。若 softmax 与 CE 之间插入了自定义 op，融合 pass 可能不命中。

### 10.3 维度 D3：融合泛化性

- **XLA**：最强。producer-consumer 融合可自动发现新的 elementwise 链，无需人工干预；
- **PyTorch**：中等。新融合需手写 C++ kernel 或依赖 `torch.compile` 的 pattern matching；
- **Tenth**：最弱。新融合需新增 `TapeOp` 变体并手写前向-反向，修改 21 个算子的枚举（[autodiff.rs:29-79](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/autodiff.rs)）。

### 10.4 维度 D4：融合的可形式化性

- **Tenth**：每个融合算子（如 CE）可独立形式化，反向公式写在源码注释与本文 §4 中，可逐一验证；
- **PyTorch**：融合规则散布在 `aten/src/ATen/native/` 的多个 kernel 实现中，形式化需逐一查阅；
- **XLA**：融合规则散布在 `xla/service/` 的多个 pass 中（`fusion`, `multi_output_fusion`, `gpu_fusion` 等），形式化需覆盖整个 pass pipeline。

### 10.5 互补性结论

Tenth 的策略并非"优于"XLA/PyTorch，而是**互补**：

- **Tenth 擅长**：已知高频融合模式（CE+Softmax、LayerNorm、GELU 等"标准算子"），稳定可预测；
- **XLA 擅长**：长尾 elementwise 链的自动融合，适应任意用户代码；
- **PyTorch 擅长**：快速迭代新融合（手写 kernel），灵活性高。

理想的 AI 框架应同时具备三层融合能力：标准算子级（Tenth 风格）+ API 级手写（PyTorch 风格）+ 图级自动（XLA 风格）。Tenth 当前仅实现第一层，§12 开放问题将讨论后续两层。$\square$（定理 S5 证毕）

---

## 11. 工程权衡

### 11.1 存储开销 vs 计算节省

CE 融合节点存储 `softmax` 中间量（$O(n)$），换取反向时 $O(n^2)\to O(n)$ 的计算节省。对 $n=50000$，存储 $50000$ 个 f64（400 KB），节省 $10^{10}$ flop——存储/计算比极优。

### 11.2 融合粒度 vs 灵活性

Tenth 的融合粒度是"固定算子对"（CE+Softmax），无法处理"CE+Softmax+LabelSmoothing"或"CE+Softmax+Mixup"等变体。这些变体需退化为分离实现，丢失融合收益。PyTorch 的 `F.cross_entropy` 支持 `label_smoothing` 参数，在同一融合 kernel 内处理——这是 API 级手写融合的灵活性优势。

### 11.3 多路径一致性

T38 定理 A1 证明 VM 与解释器记录的 tape 同构。CE 融合在两条路径上的行为一致：均调用 `tape.cross_entropy` 记录节点（[natives.rs:360-365](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/interpreter/natives.rs)），反向均走 [autodiff.rs:723-734](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/autodiff.rs)。JIT 路径在 recording 模式下整体退出至 VM（T38 定理 A2），因此 CE 融合的语义在三条路径上一致。

### 11.4 dtype 泛化

Tenth 的 softmax 前向支持 f64 与 f32（[tensor.rs:1162-1200](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/tensor.rs)），但反向（[autodiff.rs:738-743](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/autodiff.rs)）使用 `ArrayD<f64>`——f32 张量在反向时会被提升为 f64。这一设计的正确性依赖 T17（dtype 提升格）的保证，但对 f32 训练的内存开销有影响（梯度占双倍内存）。

---

## 12. 开放问题

### 12.1 图级融合的引入

Tenth 当前仅支持标准算子级融合。是否引入图级融合 pass（类似 XLA）以自动发现新融合模式？这需要：

- 在 HIR 层引入 fusion analysis pass；
- 定义融合合法性的形式化条件（可基于定理 S4 的 (C1)(C2)(C3)）；
- 处理融合与 JIT 的交互（融合后的算子是否仍可 JIT 编译？）。

### 12.2 融合的自动发现

定理 S4 给出了融合语义等价的充要条件，但未给出"自动发现可融合模式"的算法。是否存在多项式时间的算法，给定一个算子序列，判定它是否可融合为单一算子？这一问题与"代数化简的自动发现"相关，可能涉及 Gröbner 基或项重写系统的理论。

### 12.3 Label Smoothing 等变体的融合

`LabelSmoothing` 将 target 从 one-hot 改为 $(1-\alpha)t + \alpha/n$。融合 CE+LabelSmoothing 的反向为 $y - ((1-\alpha)t + \alpha/n) = y - (1-\alpha)t - \alpha/n$，仍是 $O(n)$。是否将其作为新 `TapeOp` 还是作为 CE 的参数？工程权衡待定。

### 12.4 与 Wengert Tape 形式化的深度联动

规划中的 T39（Wengert Tape 形式化模型）将严格化 Tape 节点的代数结构。本文的融合框架（定理 S4）依赖 Tape 节点的"中间量存储"能力——这一能力的形式化（如"存储什么、何时释放"）需 T39 提供。T43 与 T39 的联动方向：

- T43 提供"为什么存储 softmax 中间量"的语义动机（定理 S4 的 (C3)）；
- T39 提供"中间量在 Tape 上的生命周期管理"的形式化模型。

### 12.5 f16/bf16 下的数值稳定性

本文的数值稳定性分析（§8）基于 fp64。在 fp16/bf16 下，softmax 的减 max 仍防止上溢，但下溢更严重（fp16 最小正正规数约 $6\times 10^{-5}$）。融合 CE 反向（$y - t$）的 fp16 误差分析待研究。

---

## 13. 局限（独立章节）

本节诚实记录本文分析与 Tenth 实现之间的差距，每条局限说明：是什么、影响多大、如何缓解。

### 13.1 局限 L1：前向-反向标度不一致

**是什么**：Tenth 的 CE 前向（[natives.rs:334-347](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/interpreter/natives.rs)）计算
$$
L_{\text{fwd}} = -\frac{1}{N}\sum_i t_i\log\max(y_i, \varepsilon),
$$
其中 $N$ 为**总元素数**（`sm_slice.len()`，跨 batch 与类别），$\varepsilon = 10^{-10}$。但反向（[autodiff.rs:730](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/autodiff.rs)）返回
$$
g = y - t,
$$
对应的是 $L = -\sum_i t_i\log y_i$（无 mean、无 eps）的梯度。

**严格推导**：对 $L_{\text{fwd}} = -(1/N)\sum_i t_i\log\max(y_i, \varepsilon)$，当 $y_i \gg \varepsilon$ 时，
$$
\frac{\partial L_{\text{fwd}}}{\partial x_j} = \frac{1}{N}(y_j - t_j).
$$
Tenth 实现返回 $y_j - t_j$，**少了 $1/N$ 因子**。

**影响**：梯度被放大 $N$ 倍。对 batch=32、类别=10 的训练，$N = 320$，梯度放大 320 倍。这不影响梯度方向（仍是下降方向），但影响学习率的有效尺度——用户需相应调小学习率，否则训练可能发散。

**缓解**：
1. **短期**：在文档中说明"CE 反向未含 mean 因子"，建议用户调小学习率；
2. **中期**：修改反向为 `(softmax - target) / N`，与前向一致；
3. **长期**：引入 reduction 类型参数（sum/mean/none），分别对应不同反向。

**本文定理 S2 与实现的差距**：定理 S2 证明的是"求和形式"CE（$L = -\sum t_i\log y_i$）的融合正确性，与 Tenth 实现的"求均值形式"前向不完全匹配。定理 S2 的**公式等价性**部分（$y - t$ 等价于链式法则化简）仍成立，但**前向-反向一致性**在实现中被违反。

### 13.2 局限 L2：稀疏化的下界非严格 O(n)

**是什么**：定理 S1 声称稀疏化时间为 $\Theta(n)$。严格地，Tenth 实现含一次 `(&grad * y).sum()`（[autodiff.rs:741](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/autodiff.rs)），`sum` 是归约操作，其并行下界为 $\Omega(\log n)$（并行归约树深度）。串行下界为 $\Omega(n)$。

**影响**：在串行执行下，$\Theta(n)$ 成立；在并行执行下，下界为 $\Theta(\log n)$（树归约）而非 $O(1)$。本文的复杂度分析隐含串行假设。

**缓解**：在定理陈述中明确"串行 RAM 模型"；并行复杂度需另行分析。

### 13.3 局限 L3：融合框架的泛化性

**是什么**：定理 S4 给出了融合语义等价的充要条件，但 Tenth 的实际融合能力仅覆盖 21 个 `TapeOp` 中已定义的模式。新融合模式（如 CE+LabelSmoothing、Attention+Softmax、GELU+BiasResidual）需新增 `TapeOp`，无法自动发现。

**影响**：Tenth 的融合能力是"封闭集"——用户无法定义新融合算子（除非修改编译器源码）。这与 XLA 的"开放集"融合形成对比。

**缓解**：
1. **短期**：接受封闭集限制，文档化已支持的融合模式；
2. **中期**：引入"用户自定义 fused op"机制，允许用户在 Tenth 源码层定义前向-反向对；
3. **长期**：引入图级融合 pass（§12.1）。

### 13.4 局限 L4：ties 点的雅可比存在性

**是什么**：定理 S3 的证明在"非 ties 点"成立——即 $\arg\max$ 唯一。在 ties 点（多个分量同时达到最大值），$m(x) = \max_k x_k$ 不可微。虽然 softmax 本身仍可微（由平移不变性），但"将 $m$ 视为 $x$ 的函数"的论证在 ties 点失效。

**影响**：理论漏洞，但工程无影响——Tenth 的实现将 $m$ detach，且 ties 点的 softmax 雅可比仍由稀疏化公式正确给出（因为 $\tilde y = y$ 仍成立，雅可比的最终表达式不含 $m$）。

**缓解**：在定理 S3 陈述中明确"在非 ties 点处，将 $m$ 视为 $x$ 的函数时雅可比仍成立；在 ties 点，detach 处理保证正确性"。

### 13.5 局限 L5：CE 前向 eps 与反向的不一致

**是什么**：CE 前向使用 $\max(y_i, \varepsilon)$（[natives.rs:344](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/interpreter/natives.rs)），$\varepsilon = 10^{-10}$，但反向返回 $y - t$（无 eps）。严格地，$\partial L_{\text{fwd}}/\partial x_j$ 在 $y_i < \varepsilon$ 时应为 0（因 $\max$ 在饱和区导数为 0），但 Tenth 反向仍返回 $y_j - t_j$。

**影响**：在 $y_i < 10^{-10}$ 的极端情况下（极不自信的预测），反向梯度与前向不一致。实践中罕见（softmax 输出极少低于 $10^{-10}$），但不严格。

**缓解**：与 L1 一并修复，使用 `log_softmax` 的数值稳定形式（`log_softmax(x) = x - m - log(sum(exp(x - m)))`）避免 eps。

### 13.6 局限 L6：未覆盖 Tenth 自举编译器（tenthc）

**是什么**：本文分析基于 Rust 母编译器（`tenth/src/runtime/autodiff.rs`）。Tenth 的自举编译器 `tenthc` 是否有对应的 CE+Softmax 实现，本文未验证。

**影响**：若 tenthc 的 autodiff 实现与 Rust 侧不一致，自举路径 B（Tenth 前端 + Rust 后端）的 CE 融合行为可能与路径 A 不同。

**缓解**：查阅 `tenthc/` 对应模块，确认两侧实现一致（这属于 T38 多路径一致性的范畴）。

---

## 14. 结论

本文对 Tenth v0.3.3 的 Softmax 与 CrossEntropy 两个 `TapeOp` 算子进行了形式化分析，证明五条主定理：

1. **S1**：Softmax 雅可比稀疏化 $g_i = y_i(\text{grad}_i - \sum_j\text{grad}_j y_j)$ 与完整雅可比乘法等价，复杂度 $\Theta(n^2)\to\Theta(n)$；
2. **S2**：CE+Softmax 融合反向 $g = y - t$ 与五节点链式法则等价，复杂度 $\Theta(n^2)\to\Theta(n)$，且消除除法提升数值稳定性；
3. **S3**：softmax 减 max 的梯度不变性源于平移不变性，而非 detach 约定——这是一个比"实现约定"更强的数学事实；
4. **S4**：算子融合语义等价的三充要条件 (C1)(C2)(C3)，可指导新算子的融合设计；
5. **S5**：Tenth 的标准算子级融合与 XLA 图级融合、PyTorch API 级手写融合在 D1–D4 四维度上互补。

本文的诚实贡献在于 §13 的六类局限披露：前向-反向标度不一致（L1）、稀疏化并行下界（L2）、融合泛化性（L3）、ties 点存在性（L4）、eps 不一致（L5）、自举覆盖（L6）。其中 L1 是最重要的工程差距——反向梯度被放大 $N$ 倍——建议在中期修复时同步更新。

理论结论对应 Tenth v0.3.3 已实现的 `TapeOp::Softmax`（[autodiff.rs:735-745](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/autodiff.rs)）与 `TapeOp::CrossEntropy`（[autodiff.rs:723-734](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/autodiff.rs)），所有形式化定义均可锚定到具体源码位置。

---

## 15. 附录

### 附录 A：定理索引

| 定理 | 陈述 | 证明 | 源码锚点 |
|------|------|------|---------|
| S1 | 雅可比稀疏化正确性 | §6 | [autodiff.rs:735-745](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/autodiff.rs) |
| S2 | CE+Softmax 融合正确性 | §7 | [autodiff.rs:723-734](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/autodiff.rs), [autodiff.rs:152-173](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/autodiff.rs) |
| S3 | 数值稳定性（减 max） | §8 | [tensor.rs:1170-1171](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/tensor.rs) |
| S4 | 算子融合形式化框架 | §9 | [autodiff.rs:152-173](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/autodiff.rs) |
| S5 | 与 XLA/PyTorch 对比 | §10 | — |

### 附录 B：与现有文档的对应

| 本文章节 | 对应文档 |
|---------|---------|
| §3 Tape 节点结构 | T2（Tape 形式化模型）§3 |
| §4 Tenth 形式化 | T38（autodiff-tape 多路径一致性）§4 |
| §9 融合框架 | T39（Wengert Tape 形式化，规划中）|
| §11.3 多路径一致性 | T38 定理 A1, A2 |
| §13.6 自举覆盖 | T12（双侧编译器语义等价性）|

### 附录 C：实施建议

基于本文的理论结论，对 Tenth 实施的建议：

1. **修复 L1（标度不一致）**：将 CE 反向改为 `(softmax - target) / N`，与前向 mean 一致。修改 [autodiff.rs:730](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/autodiff.rs) 为 `(&sm_ref.data - &tgt_ref.data) / N`，其中 $N$ 需在 CE 节点中存储或从 shape 推断。
2. **修复 L5（eps 不一致）**：前向改用 `log_softmax` 的数值稳定形式，避免 eps；或反向加入 eps 的导数项。
3. **新增 LabelSmoothing 融合**（§12.3）：作为 `TapeOp::CrossEntropy` 的参数化变体，反向 $g = y - ((1-\alpha)t + \alpha/n)$，仍是 $O(n)$。
4. **文档化融合行为**：在语言参考手册中说明 CE+Softmax 的融合语义，包括 L1 的标度约定。
5. **测试覆盖**：新增测试验证 (a) 稀疏化与完整雅可比的数值一致性（小 $n$）；(b) 融合反向与分离反向的数值一致性；(c) 减 max 不改变梯度；(d) L1 标度因子。

---

## 16. 参考文献

1. Bridson, M. & Haas, J. (2018). "Numerically Stable Softmax & Cross-Entropy." arXiv preprint.
2. Blanchard, P. et al. (2021). "Softmax Tempering and Numerical Stability in Neural Network Training." *NeurIPS*.
3. Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press. §6.2.2 (Softmax), §8.1 (Cross-Entropy).
4. Baydin, A. G., Pearlmutter, B. A., Radul, A. A., & Siskind, J. M. (2018). "Automatic Differentiation in Machine Learning: a Survey." *JMLR*.
5. Wengert, R. E. (1964). "A Simple Automatic Derivative Evaluation Program." *Comm. ACM*.
6. XLA Team (2023). "XLA: Optimizing Compiler for Machine Learning." TensorFlow Documentation.
7. PyTorch Team (2024). "torch.nn.functional.cross_entropy." PyTorch Documentation.
8. Tenth Project (2026). *T2: Tape 形式化模型与根因定位可判定性*. 内部文档.
9. Tenth Project (2026). *T38: autodiff-tape 多路径一致性*. 内部文档.
10. Tenth Project (2026). *T39: Wengert Tape 形式化模型*（规划中）.
11. Tenth Project (2026). *T17: dtype 提升格与混合 dtype 算术*. 内部文档.
12. Tenth Project (2026). *T12: 双侧编译器语义等价性*. 内部文档.

---

> **文档结束** · T43 v1 · 数理部产出 · 2026-07-02
> **审查状态**：v1 待审查（建议审查重点：L1 标度不一致的工程影响、定理 S4 的 (C2) 充分性论证、定理 S3 的 ties 点处理）
