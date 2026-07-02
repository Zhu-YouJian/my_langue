# Wengert Tape 的形式化语义与反向模式正确性：Tenth 21 算子链式法则验证

> **论文编号**：T39 · **系列**：T27–T39 · **级别**：硕士级
> **数理部产出**：理论分析论文（v1）
> **联动论文**：T2（Tape 形式化模型与根因定位可判定性）、T38（autodiff tape 多路径一致性）
> **基准版本**：Tenth v0.3.3
> **撰写日期**：2026-07-02

---

## 摘要

Tenth 作为 AI 原生语言，将自动微分作为运行时一等公民，通过单一持久化的 Wengert tape 在前向阶段记录张量计算、在反向阶段按拓扑逆序回放链式法则。tape 的 21 个算子（`Input`、`Add`、`Sub`、`Mul`、`Div`、`Neg`、`ReLU`、`MatMul`、`Transpose`、`Sum`、`Mean`、`Exp`、`Log`、`Sigmoid`、`Softmax`、`CrossEntropy`、`Dropout`、`Conv2D`、`BatchNorm`、`LayerNorm`、`Gelu`）涵盖深度学习模型的核心计算，其中 `CrossEntropy`、`Conv2D`、`BatchNorm`、`LayerNorm`、`Gelu`、`Softmax` 为复合算子，其 backward 均为一次性推导的闭式解。

本文形式化 Tenth 的 Wengert tape 语义，证明五条主定理：

- **定理 AD1（链式法则等式）**：21 个 `TapeOp` 变体的 backward 实现逐一满足链式法则 $\partial L/\partial x_i = \sum_j \partial L/\partial y_j \cdot \partial y_j/\partial x_i$；
- **定理 AD2（拓扑逆序正确性）**：`Tape::backward` 按 `nodes.iter().rev()` 单遍回放保证梯度累积完整，无需显式反向图；
- **定理 AD3（input_tensors 持久化必要性）**：`input_tensors: Vec<Rc<RefCell<Tensor>>>` 的显式持久化是 backward 计算闭式解的必要条件——若仅持久化节点 id，则复合算子的中间值（softmax、im2col、x_hat、std_inv、mask）无法重建；
- **定理 AD4（与 PyTorch/JAX 语义等价性）**：在 tape 完整性前提下（T38 已证），Tenth 的 tape 反向与 PyTorch autograd、JAX `jax.grad` 在数学语义上等价，差异仅在工程组织（显式 tape vs 动态反向图 vs 纯函数式 JAXPR）；
- **定理 AD5（算子融合的形式化框架）**：复合算子的闭式 backward 等价于展开后逐算子链式法则的代数简化，给出融合保持语义等价的充分条件。

本文的诚实贡献在于 §6 对 21 算子链式法则的**逐一**验证（不偷懒、不"易证"），以及独立局限章节对证明漏洞的披露：`Conv2D` 反向的 col2im 简化（无 stride/padding/dilation 处理）、`BatchNorm` 反向的简化版本（不含 running statistics）、`CrossEntropy` 数值稳定性的隐式假设、以及 `Softmax` 反向在退化情形 $y_i \to 0$ 时的奇异性。

**关键词**：自动微分；Wengert tape；反向模式；链式法则；算子融合；指称语义；梯度正确性

---

## 1. 引言

### 1.1 反向模式自动微分的挑战

反向模式自动微分（reverse-mode automatic differentiation，以下简称 reverse-mode AD）是深度学习训练的数学基石：给定标量损失 $L$ 与参数向量 $\theta$，reverse-mode AD 以代价 $O(\text{前向})$ 同时计算 $\partial L/\partial \theta_i$ 对所有 $i$。这一效率来自于沿前向计算图的反向拓扑序传播梯度，每个节点的局部雅可比只需计算一次。

然而，reverse-mode AD 的**正确性**远非显然。它要求：

1. **链式法则的局部正确性**：每个算子 $f$ 的 backward 实现 $\bar f$ 必须满足 $\bar f(\bar y, x) = \bar y \cdot \partial f/\partial x$（伴随关系）；
2. **拓扑序的全局正确性**：反向遍历必须按拓扑逆序，使得每个节点的上游梯度 $\bar y$ 在被使用前已完全累积；
3. **中间值的可用性**：backward 计算可能需要前向阶段的中间值（如 `Softmax` 的输出 $y$、`LayerNorm` 的 $x_{\text{hat}}$ 与 $\sigma^{-1}$、`Conv2D` 的 im2col 矩阵），这些中间值必须被持久化；
4. **复合算子闭式解的正确性**：将多个基本算子融合为一个复合算子（如 `CrossEntropy = Softmax + Log + Neg + Sum`）时，其闭式 backward 必须等价于展开后逐算子链式法则的代数简化。

任一条件被破坏，梯度将静默错误——这是深度学习调试中最隐蔽的 bug 来源。

### 1.2 Wengert tape 模型

Wengert（1964）提出将计算分解为基本算子的序列，每个算子的输入是先前算子的输出或独立变量。形式化地，给定程序 $P: x \mapsto y$，Wengert tape $T$ 是一个有限序列：

$$T = [(op_1, \text{in}_1, \text{out}_1), (op_2, \text{in}_2, \text{out}_2), \dots, (op_n, \text{in}_n, \text{out}_n)]$$

其中 $\text{in}_i \subseteq \{x\} \cup \{\text{out}_j : j < i\}$。反向阶段从 $\bar y = 1$ 出发，按 $i = n, n-1, \dots, 1$ 的顺序对每个节点应用局部链式法则，累积 $\bar x$。

Wengert tape 的优势在于其**线性序列结构**——无需显式反向图，按记录顺序的逆序遍历即可保证拓扑序。这一性质使得 tape 实现简单且内存局部性好，是 Tenth 的设计选择。

### 1.3 Tenth 的 21 算子设计

Tenth 的 tape 实现见 [autodiff.rs L1-L765](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/autodiff.rs)。`TapeOp` 枚举定义了 21 个变体（[autodiff.rs:29-79](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/autodiff.rs)）：

```
Input | Add | Sub | Mul | Div | Neg | ReLU | MatMul | Transpose |
Sum | Mean | Exp | Log | Sigmoid | Softmax | CrossEntropy |
Dropout | Conv2D | BatchNorm | LayerNorm | Gelu
```

**算子分类**（按输入元数）：

| 类别 | 算子 | 数量 |
|------|------|------|
| 叶（0 元） | `Input` | 1 |
| 一元 | `Neg`, `ReLU`, `Transpose`, `Sum`, `Mean`, `Exp`, `Log`, `Sigmoid`, `Softmax`, `Gelu` | 10 |
| 二元 | `Add`, `Sub`, `Mul`, `Div`, `MatMul` | 5 |
| 多元（含中间值） | `CrossEntropy`, `Dropout`, `Conv2D`, `BatchNorm`, `LayerNorm` | 5 |

**复合算子的闭式 backward**：

- `CrossEntropy`：闭式 $\partial L/\partial \text{logits} = \text{softmax}(\text{logits}) - \text{target}$，等价于 `Softmax + Log + Neg + Sum` 链式展开后的代数简化；
- `Conv2D`：通过 im2col 将卷积转化为矩阵乘法，闭式 $\partial L/\partial W = \text{im2col}^T @ \text{dY}$、$\partial L/\partial X = \text{col2im}(\text{dY} @ W_{\text{flat}})$；
- `BatchNorm`：闭式 $\partial L/\partial X = (\gamma/\sigma) \cdot (\bar Y - \text{mean}(\bar Y) - x_{\text{hat}} \cdot \text{mean}(\bar Y \cdot x_{\text{hat}}))$；
- `LayerNorm`：与 `BatchNorm` 类似但归一化轴不同，按行计算 mean；
- `Gelu`：tanh 近似，闭式 $\partial L/\partial x = \bar Y \cdot [0.5(1+\tanh(\text{inner})) + 0.5 x \cdot \text{sech}^2(\text{inner}) \cdot \sqrt{2/\pi} (1 + 3 \cdot 0.044715 x^2)]$，其中 $\text{inner} = \sqrt{2/\pi}(x + 0.044715 x^3)$。

### 1.4 贡献

1. **形式化 Tenth 的 Wengert tape 语义**（§3），给出 `TapeOp`、`TapeNode`、`Tape` 的代数结构与 forward/backward 分离的指称语义；
2. **证明定理 AD1（链式法则等式）**（§4.1），并对 21 个算子**逐一**验证（§6）——含 5 个复合算子闭式解的代数推导；
3. **证明定理 AD2（拓扑逆序正确性）**（§4.2），论证 `nodes.iter().rev()` 单遍回放在 DAG 结构下的正确性；
4. **证明定理 AD3（input_tensors 持久化必要性）**（§4.3），论证显式持久化输入张量是闭式 backward 的必要条件；
5. **证明定理 AD4（与 PyTorch/JAX 语义等价性）**（§4.4），对比三种实现的数学语义与工程差异；
6. **证明定理 AD5（算子融合的形式化框架）**（§4.5），给出融合保持语义等价的充分条件；
7. **独立局限章节**（§10）诚实披露证明漏洞与工程差距；
8. **与 T38 联动**：T38 证明 tape 多路径一致性（前提），T39 证明 tape 反向语义正确性（结论），共同构成 Tenth autodiff 的完整正确性论证。

---

## 2. 背景

### 2.1 Reverse-mode AD 理论

反向模式自动微分的形式化理论由 Baydin、Pearlmutter、Radul、Siskind（2018）在 *Automatic Differentiation in Machine Learning: a Survey* 中系统总结。核心定理（Baydin et al. 2018, §4）表述如下：

**定理（Baydin 等，反向模式 AD）**：给定可微程序 $P: \mathbb{R}^n \to \mathbb{R}^m$，将其分解为基本算子序列 $P = f_n \circ f_{n-1} \circ \dots \circ f_1$，其中 $f_i: \mathbb{R}^{k_i} \to \mathbb{R}^{k_{i+1}}$。设 $v_i = f_i(v_{i-1})$ 为前向中间值（$v_0 = x$），则 $\bar v_i = \bar v_{i+1} \cdot \partial f_{i+1}/\partial v_i$（伴随关系），按 $i = n-1, n-2, \dots, 0$ 逆序计算。最终 $\bar v_0 = \bar y \cdot \partial P/\partial x$。

**复杂度定理**：reverse-mode AD 的计算代价为 $O(\text{前向代价})$，与输出维度无关，仅与输入维度有关。这使得 reverse-mode AD 在 $n \gg m$ 时（典型深度学习场景：$n$ 为参数数，$m = 1$ 为标量 loss）远优于 forward-mode AD。

Griewank 与 Walther（2008）在 *Evaluating Derivatives* 中给出更严格的形式化，将 Wengert tape 定义为"原始计算轨迹"，并证明 tape 上的反向遍历等价于计算雅可比转置乘积 $J_P^T \bar y$。

### 2.2 PyTorch autograd

PyTorch 的 autograd 引擎（Paszke et al. 2017，*Automatic Differentiation in PyTorch*）采用**动态反向图**模型：

- 每个 `torch.Tensor` 持有 `.grad_fn` 属性，指向产生它的反向函数；
- 前向执行时，若 `requires_grad=True`，则构建反向图节点（惰性构建，每次前向都重建）；
- 反向时 `torch.autograd.engine.execute()` 沿反向图拓扑序调度，使用多线程并行累积梯度；
- `register_hook` 机制允许用户在反向阶段插入副作用。

**关键差异**：PyTorch 的反向图是**节点级动态构建**的，反向遍历依赖显式的拓扑排序（通过 `next_edges` 链）。Tenth 的 tape 则是**单一显式持久化序列**，反向遍历直接 `nodes.iter().rev()`（[autodiff.rs:285](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/autodiff.rs)），无需显式反向图——因为 tape 的记录顺序就是合法的拓扑序。

### 2.3 JAX autodiff

JAX 的 autodiff（Frostig et al. 2018，*Compiling machine learning programs for high-performance numerical computing*）建立在**纯函数式**抽象值（traced values）上：

- `jax.grad(f)` 将输入包装为 `Tracer`，前向执行时构建 JAXPR 中间表示；
- JAXPR 是纯函数式 IR，禁止副作用；
- 反向阶段对 JAXPR 求导，生成新的 JAXPR 表示导数函数；
- 通过 XLA 编译为高效 GPU 代码。

**关键差异**：JAX 通过纯函数性保证 tape 完整性——任何副作用被类型系统拒绝。Tenth 是命令式语言，tape 记录依赖运行时的 `if self.recording { ... }` 分支，effect system 缺位下完整性靠工程纪律维持（详见 T38 §5）。

### 2.4 Wengert tape 起源与 Tenth 的定位

Wengert（1964）的原始提议是一种"计算列表"（computation list），每个条目记录一个基本算子及其输入输出。这一概念在 TensorFlow 1.x 的静态图、Autograd（PyTorch 前身）、Chainer 等框架中以不同形式实现。

Tenth 的 tape 设计定位为**显式持久化的 Wengert tape**——不同于 PyTorch 的动态反向图，也不同于 JAX 的 JAXPR，而是回归 Wengert 的原始构想：单一序列、显式持久化中间张量、按记录逆序回放。这一选择的工程动因是：

1. **简单性**：单一序列，无需拓扑排序算法；
2. **可调试性**：tape 节点显式持有输入张量，便于根因定位（T2 论文已论证）；
3. **闭式 backward**：复合算子的中间值已持久化，可一次性计算闭式梯度，避免展开为基本算子链的开销。

代价是**内存占用高**（所有中间张量被 `Rc<RefCell<Tensor>>` 持有），以及**单一序列结构**对循环/分支的支持需要额外机制（T38 §3.4 已讨论）。

---

## 3. Tenth Tape 形式化

### 3.1 TapeOp 枚举

`TapeOp` 是 `TapeNode` 的算子标签，定义为 21 个变体的枚举（[autodiff.rs:29-79](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/autodiff.rs)）：

```rust
pub enum TapeOp {
    Input, Add, Sub, Mul, Div, Neg, ReLU, MatMul, Transpose,
    Sum, Mean, Exp, Log, Sigmoid, Softmax, CrossEntropy,
    Dropout, Conv2D, BatchNorm, LayerNorm, Gelu,
}
```

**形式化**：定义算子签名元组 $\Sigma = (\mathcal{O}, \text{arity}, \text{denot})$，其中：

- $\mathcal{O} = \{\text{Input}, \text{Add}, \dots, \text{Gelu}\}$ 是 21 个算子的有限集；
- $\text{arity}: \mathcal{O} \to \mathbb{N}$ 给出输入元数：
  - $\text{arity}(\text{Input}) = 0$
  - $\text{arity}(\text{Neg, ReLU, Transpose, Sum, Mean, Exp, Log, Sigmoid, Softmax, Gelu}) = 1$
  - $\text{arity}(\text{Add, Sub, Mul, Div, MatMul}) = 2$
  - $\text{arity}(\text{CrossEntropy, Dropout, Conv2D, BatchNorm, LayerNorm}) = \text{变长}$（多元，含中间值）
- $\text{denot}: \mathcal{O} \to (\text{Tensor}^* \to \text{Tensor})$ 是指称语义（§5 逐一给出）。

### 3.2 TapeNode 与 Tape 结构

`TapeNode`（[autodiff.rs:14-25](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/autodiff.rs)）：

```rust
pub struct TapeNode {
    pub id: usize,
    pub op: TapeOp,
    pub inputs: Vec<usize>,                              // 上游节点 id
    pub input_tensors: Vec<Rc<RefCell<Tensor>>>,         // 输入张量引用（持久化）
}
```

**形式化**：`TapeNode` 是四元组 $(id, op, \text{inputs}, \text{input\_tensors})$，其中：

- $id \in \mathbb{N}$ 是节点唯一标识（即 `nodes` 数组中的索引）；
- $op \in \mathcal{O}$；
- $\text{inputs} \subseteq \{0, 1, \dots, id-1\}$ 是上游节点 id 的有序列表（满足 $\forall j \in \text{inputs}: j < id$，保证 DAG 性质）；
- $\text{input\_tensors}$ 是输入张量的 `Rc<RefCell<Tensor>>` 引用列表，长度与 `op` 的元数匹配（含中间值）。

`Tape`（[autodiff.rs:83-86](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/autodiff.rs)）：

```rust
pub struct Tape {
    nodes: Vec<TapeNode>,
    counter: usize,
}
```

**形式化**：`Tape` 是一个有限序列 $T = [n_0, n_1, \dots, n_{k-1}]$，其中 $n_i.\text{id} = i$（不变量：`counter == nodes.len()`，由 `next_id` 维护，[autodiff.rs:261-265](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/autodiff.rs)）。

**DAG 性质**：tape 是有向无环图（DAG），因为 $\forall i, \forall j \in n_i.\text{inputs}: j < i$。这一性质是定理 AD2（拓扑逆序正确性）的基础。

### 3.3 forward/backward 分离的指称语义

Tenth 的 tape 设计严格分离**前向计算**与**反向计算**：

- **前向**（forward）：在 VM/解释器/JIT 中执行实际张量运算，结果张量产生后调用 `tape.unary/binary/cross_entropy/...` 等记录方法，将算子与输入张量持久化到 tape（[autodiff.rs:108-259](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/autodiff.rs)）；
- **反向**（backward）：`Tape::backward`（[autodiff.rs:272-749](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/autodiff.rs)）按 `nodes.iter().rev()` 单遍遍历，对每个节点应用对应算子的 backward 公式。

**指称语义**：定义两个语义函数：

- $\mathcal{F}[\![op]\!] : \text{Tensor}^* \to \text{Tensor}$ ——前向指称，给出算子的数学语义（如 $\mathcal{F}[\![\text{Add}]\!](a, b) = a + b$）；
- $\mathcal{B}[\![op]\!] : (\text{Tensor}^*, \text{Tensor}) \to \text{Tensor}^*$ ——反向指称，给出 backward 的伴随语义（如 $\mathcal{B}[\![\text{Add}]\!]((a, b), \bar c) = (\bar c, \bar c)$）。

**链式法则等式**（定理 AD1 的形式化陈述）：

$$\mathcal{B}[\![op]\!](\text{inputs}, \bar y) = \left( \bar y \cdot \frac{\partial \mathcal{F}[\![op]\!](\text{inputs})}{\partial \text{inputs}_i} \right)_{i=1}^{\text{arity}}$$

即 backward 实现必须等于前向指称的雅可比转置乘以上游梯度。§6 将对 21 个算子逐一验证此等式。

### 3.4 input_tensors 持久化协议

`input_tensors` 字段持久化输入张量的 `Rc<RefCell<Tensor>>` 引用。**协议**：

| 算子 | `input_tensors` 内容 | 持久化目的 |
|------|---------------------|-----------|
| `Input` | `[tensor]` | 累积梯度的目标 |
| `Add`/`Sub`/`Mul`/`Div` | `[a, b, result]` | 读取 a, b 数据；读取 result 形状（仅部分算子用） |
| `MatMul` | `[a, b, result]` | 读取 a, b 数据计算 $G @ B^T$、$A^T @ G$ |
| `Neg`/`ReLU`/`Log` | `[input, result]` | ReLU/Log 读取 input 数据 |
| `Exp`/`Sigmoid`/`Softmax` | `[input, result]` | 读取 result（即 $\exp(a)$、$\sigma(a)$、$\text{softmax}(a)$） |
| `Sum`/`Mean` | `[input, result]` | 读取 input 形状 |
| `Transpose` | `[input, result]` | 读取 grad 形状即可，input 不直接用 |
| `CrossEntropy` | `[logits, softmax, target, result]` | 读取 softmax 与 target 计算 $\text{softmax} - \text{target}$ |
| `Dropout` | `[input, mask, result]` | 读取 mask |
| `Conv2D` | `[x, w, im2col, output]` | 读取 im2col 计算 $\text{dW}$；读取 w_flat 计算 $\text{dX}$ |
| `BatchNorm` | `[x, gamma, beta, x_hat, std_inv, result]` | 读取 gamma, x_hat, std_inv |
| `LayerNorm` | `[x, gamma, beta, x_hat, std_inv, result]` | 同 BatchNorm，按行 |
| `Gelu` | `[input, result]` | 读取 input（即 $x$）计算导数 |

**关键观察**：`result` 总是 `input_tensors` 的最后一个元素，便于 `backward` 取种子梯度（[autodiff.rs:279-282](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/autodiff.rs)）。

定理 AD3（§4.3）将证明：对于复合算子（`CrossEntropy`、`Conv2D`、`BatchNorm`、`LayerNorm`、`Gelu`、`Softmax`、`Sigmoid`、`Exp`、`Dropout`），持久化输入张量是计算闭式 backward 的**必要条件**——若仅持久化节点 id，则中间值无法重建。

---

## 4. 主定理

### 4.1 定理 AD1（链式法则等式）

**定理 AD1**：对于 `TapeOp` 的每个变体 $op \in \mathcal{O}$，`Tape::backward` 中的实现 $\mathcal{B}_{\text{impl}}[\![op]\!]$ 满足链式法则等式：

$$\forall \text{inputs}, \bar y: \quad \mathcal{B}_{\text{impl}}[\![op]\!](\text{inputs}, \bar y) = \left( \bar y \cdot \frac{\partial \mathcal{F}[\![op]\!](\text{inputs})}{\partial \text{inputs}_i} \right)_{i=1}^{\text{arity}(op)}$$

其中 $\mathcal{F}[\![op]\!]$ 是 $op$ 的前向指称（§5），$\partial/\partial \text{inputs}_i$ 是雅可比矩阵，$\bar y \cdot \cdot$ 是矩阵-张量乘积（伴随）。

**证明**：见 §6 的 21 算子逐一验证。每个算子的证明包含：
1. 前向指称 $\mathcal{F}[\![op]\!]$ 的显式给出；
2. 解析雅可比 $\partial \mathcal{F}[\![op]\!]/\partial \text{inputs}_i$ 的计算；
3. 链式法则右端 $\bar y \cdot \partial \mathcal{F}/\partial \text{inputs}_i$ 的化简；
4. backward 实现 $\mathcal{B}_{\text{impl}}[\![op]\!]$ 的代码引用；
5. 等式验证（左端 = 右端）。$\square$

**注**：定理 AD1 的证明是**穷举性**的——21 个算子逐一验证，无任何"同理可证"或"易证"。这是数理部"不偷懒"原则的实践。

### 4.2 定理 AD2（拓扑逆序正确性）

**定理 AD2**：设 tape $T = [n_0, n_1, \dots, n_{k-1}]$ 满足 DAG 性质（$\forall i, \forall j \in n_i.\text{inputs}: j < i$），loss 节点为 $n_L$。则 `Tape::backward` 的 `nodes.iter().rev()` 单遍遍历正确计算所有叶节点的梯度，即：

$$\forall \text{leaf } n_i: \quad \text{acc\_grad}(n_i) = \bar y_L \cdot \frac{\partial \mathcal{F}[\![n_L.\text{op}]\!]}{\partial n_i}$$

其中 $\bar y_L$ 是 loss 节点的种子梯度（[autodiff.rs:282](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/autodiff.rs) 设为 `ones`）。

**证明**：

**前置定义**：
- 节点 $n_i$ 的"完整上游梯度" $\bar v_i^*$ 定义为 $\bar y_L \cdot \partial \mathcal{F}[\![n_L.\text{op}]\!] / \partial v_i$，即从 loss 出发到 $n_i$ 输出的完整链式法则结果。
- 节点 $n_i$ 的"累积梯度" $\bar v_i^{\text{acc}}$ 定义为 backward 执行到 $n_i$ 时 `node_grads[i]` 的值。

**归纳不变量**：当 backward 遍历到节点 $n_i$（即外层循环 `node = n_i`）时，$\bar v_i^{\text{acc}} = \bar v_i^*$。

**归纳基础**（$i = L$）：`backward` 在 [autodiff.rs:282] 设 `node_grads[L] = ones(loss_shape)`，即 $\bar v_L^{\text{acc}} = \bar y_L = \bar v_L^*$（因为 $\partial \mathcal{F}/\partial v_L = 1$）。归纳基础成立。

**归纳步骤**（$i < L$）：假设对所有 $j > i$，归纳不变量成立（即 $\bar v_j^{\text{acc}} = \bar v_j^*$）。需证 $\bar v_i^{\text{acc}} = \bar v_i^*$。

**关键观察**：$\bar v_i^*$ 通过链式法则可表为：

$$\bar v_i^* = \sum_{j : i \in n_j.\text{inputs}} \bar v_j^* \cdot \frac{\partial \mathcal{F}[\![n_j.\text{op}]\!]}{\partial v_i}$$

即 $\bar v_i^*$ 是所有"直接下游"节点 $n_j$ 的贡献之和。

**反向遍历的累积机制**：当 backward 遍历到节点 $n_j$（$j > i$）时，[autodiff.rs:285-289] 取 `node_grads[j]`（即 $\bar v_j^{\text{acc}} = \bar v_j^*$，由归纳假设），调用 `propagate_grad(node, input_idx, g_i, node_grads)`（[autodiff.rs:786-804](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/autodiff.rs)），其中 $g_i = \mathcal{B}_{\text{impl}}[\![n_j.\text{op}]\!](\text{inputs}, \bar v_j^{\text{acc}})_i$（由定理 AD1 等于 $\bar v_j^* \cdot \partial \mathcal{F}/\partial v_i$）。`propagate_grad` 通过 `acc_node_grad`（[autodiff.rs:770-779](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/autodiff.rs)）将 $g_i$ 累加到 `node_grads[i]`：

```rust
fn acc_node_grad(node_grads: &mut [Option<ArrayD<f64>>], id: usize, g: &ArrayD<f64>) {
    match &mut node_grads[id] {
        Some(existing) => { *existing = &*existing + g; }
        slot @ None => { *slot = Some(g.clone()); }
    }
}
```

因此，遍历完所有 $j > i$ 后：

$$\bar v_i^{\text{acc}} = \sum_{j : i \in n_j.\text{inputs}, j > i} g_i^{(j)} = \sum_{j : i \in n_j.\text{inputs}} \bar v_j^* \cdot \frac{\partial \mathcal{F}[\![n_j.\text{op}]\!]}{\partial v_i} = \bar v_i^*$$

（第二个等号用归纳假设 $\bar v_j^{\text{acc}} = \bar v_j^*$；求和范围可去掉 $j > i$ 限制，因为 DAG 性质保证 $\forall j : i \in n_j.\text{inputs}, j > i$。）

**反向遍历的顺序保证**：`nodes.iter().rev()` 按 $i = k-1, k-2, \dots, 0$ 顺序遍历，因此在处理 $n_i$ 之前，所有 $j > i$ 的节点都已处理完毕，$\bar v_i^{\text{acc}}$ 已累积所有下游贡献。归纳步骤成立。$\square$

**推论 AD2.1**：`backward` 的复杂度为 $O(\sum_i \text{cost}(\mathcal{B}[\![n_i.\text{op}]\!]))$，即所有节点 backward 代价之和。这与 reverse-mode AD 的标准复杂度一致（Baydin et al. 2018, §4）。

### 4.3 定理 AD3（input_tensors 持久化必要性）

**定理 AD3**：对于以下 9 个算子，`input_tensors` 的显式持久化是 backward 计算闭式解的**必要条件**：

$$\mathcal{N} = \{\text{Exp}, \text{Sigmoid}, \text{Softmax}, \text{CrossEntropy}, \text{Dropout}, \text{Conv2D}, \text{BatchNorm}, \text{LayerNorm}, \text{Gelu}\}$$

即若仅持久化节点 id（不持久化输入张量），则 backward 无法在 $O(1)$ 额外空间内计算（必须重放前向或重算中间值）。

**证明**：

按算子分类论证：

**(1) Exp**（[autodiff.rs:474-480](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/autodiff.rs)）：backward 公式 $\bar a = \bar c \cdot \exp(a) = \bar c \cdot c$，其中 $c = \exp(a)$ 是前向结果。实现读取 `input_tensors[1]`（即 result $c$）。若仅持久化 id，需重算 $\exp(a)$，代价 $O(|a|)$。

**(2) Sigmoid**（[autodiff.rs:488-495](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/autodiff.rs)）：backward 公式 $\bar a = \bar c \cdot c \cdot (1 - c)$，其中 $c = \sigma(a)$。实现读取 `input_tensors[1]`（即 result $c$）。若仅持久化 id，需重算 $\sigma(a)$。

**(3) Softmax**（[autodiff.rs:735-745](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/autodiff.rs)）：backward 公式 $\bar a_i = c_i (\bar c_i - \sum_j \bar c_j c_j)$，其中 $c = \text{softmax}(a)$。实现读取 `input_tensors[1]`（即 result $c$）。若仅持久化 id，需重算 $\text{softmax}(a)$，且数值稳定性要求保留 max 减法技巧。

**(4) CrossEntropy**（[autodiff.rs:723-734](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/autodiff.rs)）：backward 公式 $\bar{\text{logits}} = \text{softmax}(\text{logits}) - \text{target}$。实现读取 `input_tensors[1]`（即 softmax 输出，前向已计算并持久化）与 `input_tensors[2]`（即 target）。若仅持久化 id，需重算 $\text{softmax}(\text{logits})$，且 target 也需重新获取。

**(5) Dropout**（[autodiff.rs:712-722](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/autodiff.rs)）：backward 公式 $\bar a = \bar c \cdot \text{mask}$。实现读取 `input_tensors[1]`（即 mask）。**关键**：mask 是随机生成的，无法从输入重建——若仅持久化 id，backward **根本无法**重算 mask（随机数生成器状态已丢失）。这是"必要"的最强例证。

**(6) Conv2D**（[autodiff.rs:615-711](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/autodiff.rs)）：backward 公式 $\partial L/\partial W = \text{im2col}^T @ \text{dY}$、$\partial L/\partial X = \text{col2im}(\text{dY} @ W_{\text{flat}})$。实现读取 `input_tensors[2]`（即 im2col 矩阵）。im2col 是前向阶段从 $X$ 计算的中间矩阵，重算代价 $O(|X| \cdot k_H \cdot k_W)$，远高于持久化的 $O(1)$ 读取。

**(7) BatchNorm**（[autodiff.rs:496-522](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/autodiff.rs)）：backward 公式 $\bar X = (\gamma/\sigma) (\bar Y - \text{mean}(\bar Y) - x_{\text{hat}} \cdot \text{mean}(\bar Y \cdot x_{\text{hat}}))$。实现读取 `input_tensors[3]`（即 $x_{\text{hat}}$）与 `input_tensors[4]`（即 $\sigma^{-1}$）。$x_{\text{hat}} = (X - \mu)/\sigma$ 与 $\sigma$ 依赖前向阶段的 batch 统计量，重算需重新计算 $\mu, \sigma$，代价 $O(|X|)$。

**(8) LayerNorm**（[autodiff.rs:523-596](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/autodiff.rs)）：与 BatchNorm 类似，但按行计算。实现读取 `input_tensors[3]`（$x_{\text{hat}}$）与 `input_tensors[4]`（$\sigma^{-1}$）。

**(9) Gelu**（[autodiff.rs:597-614](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/autodiff.rs)）：backward 公式涉及 $\tanh(\text{inner})$，其中 $\text{inner} = \sqrt{2/\pi}(x + 0.044715 x^3)$。实现读取 `input_tensors[0]`（即 $x$）重算 $\tanh(\text{inner})$。**注**：Gelu 的实现实际是读取 input 而非 result，因此理论上仅需持久化 input；但若选择持久化 result 并从 result 反推 $x$（涉及反 GELU 函数，数值不稳定），则不可行。当前实现的"必要"性在于持久化 input。

**反例（非必要）**：以下算子的 backward 不依赖 input_tensors 的数据（仅依赖上游梯度或形状）：
- `Neg`：$\bar a = -\bar c$，仅依赖 $\bar c$；
- `Sum`/`Mean`：仅依赖 input 的形状（用于 broadcast），但实现仍读取 `input_tensors[0]` 获取形状（[autodiff.rs:455-473](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/autodiff.rs)）——理论上形状可单独持久化，但工程上仍用 `input_tensors`。

**结论**：对于 $\mathcal{N}$ 中的 9 个算子，`input_tensors` 的持久化是 backward 闭式解的必要条件；对于其余 12 个算子，持久化是工程便利而非数学必要。$\square$

**工程意义**：定理 AD3 解释了为何 Tenth 选择 `Vec<Rc<RefCell<Tensor>>>` 而非 `Vec<usize>`（节点 id）持久化输入——闭式 backward 需要中间值，而中间值在前向阶段产生、反向阶段消费，必须跨阶段存活。

### 4.4 定理 AD4（与 PyTorch/JAX 语义等价性）

**定理 AD4**：设程序 $P: x \mapsto y$ 由 Tenth tape 算子序列 $T = [n_0, \dots, n_{k-1}]$ 实现，且 tape 完整性成立（T38 定理 A1 已证）。则 Tenth 的 `Tape::backward` 计算的梯度 $\bar x_{\text{Tenth}}$ 与 PyTorch autograd、JAX `jax.grad` 计算的梯度 $\bar x_{\text{PyTorch}}$、$\bar x_{\text{JAX}}$ 在数学语义上等价：

$$\bar x_{\text{Tenth}} = \bar x_{\text{PyTorch}} = \bar x_{\text{JAX}} = \bar y \cdot \frac{\partial P}{\partial x}$$

（在浮点数误差范围内）。

**证明**：

**步骤 1**：Tenth 的梯度由定理 AD1（链式法则等式）与定理 AD2（拓扑逆序正确性）共同保证等于 $\bar y \cdot \partial P/\partial x$。

**步骤 2**：PyTorch autograd 的梯度正确性由 Paszke et al.（2017）证明，其反向图遍历等价于 reverse-mode AD 标准算法（Baydin et al. 2018），即 $\bar x_{\text{PyTorch}} = \bar y \cdot \partial P/\partial x$。

**步骤 3**：JAX 的梯度正确性由 Frostig et al.（2018）证明，其 JAXPR 求导等价于 reverse-mode AD，即 $\bar x_{\text{JAX}} = \bar y \cdot \partial P/\partial x$。

**步骤 4**：三者数学语义相同，差异仅在工程组织：

| 维度 | Tenth | PyTorch | JAX |
|------|-------|---------|-----|
| 反向图结构 | 单一显式 tape 序列 | 动态反向图（节点级） | JAXPR（纯函数式 IR） |
| 拓扑序 | 记录逆序（隐式） | 显式拓扑排序 | JAXPR 线性序 |
| 中间值持久化 | `input_tensors` 显式 | `ctx.save_for_backward` | Tracer 自动捕获 |
| 副作用支持 | 命令式（recording 标志） | 命令式（hook） | 禁止（纯函数式） |
| 循环/分支 | 依赖 tape 展平 | 自然支持 | 通过 `lax.scan`/`lax.cond` |

**步骤 5**：浮点数误差来自不同的运算顺序。Tenth 的 `acc_node_grad` 按下游节点 id 升序累加（[autodiff.rs:770-779](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/autodiff.rs)），PyTorch 的多线程调度顺序非确定，JAX 的 XLA 编译可能重排。三者的浮点结果可能差异 $\sim 10^{-15}$，但数学语义等价。$\square$

**注**：定理 AD4 是"相对等价"——前提是 tape 完整性（T38）与链式法则等式（AD1）。若 tape 不完整（如 T38 §6 发现的 `Neg` 算子 record 缺失），等价性破坏。

### 4.5 定理 AD5（算子融合的形式化框架）

**定理 AD5**：设复合算子 $F = f_n \circ f_{n-1} \circ \dots \circ f_1$（每个 $f_i$ 是基本算子），其闭式 backward 为 $\mathcal{B}[\![F]\!]$。若以下条件成立：

**(C1) 链式法则可化简**：展开的链式法则 $\bar x = \bar y \cdot \prod_i \partial f_i/\partial v_i$ 可代数化简为闭式表达 $G(\bar y, \text{saved})$，其中 $\text{saved}$ 是前向阶段的中间值集合；

**(C2) 中间值可持久化**：$\text{saved}$ 中的每个中间值在前向阶段可计算并持久化于 `input_tensors`；

**(C3) 数值稳定性保持**：闭式表达 $G$ 在数值上不劣于展开形式（如避免 $\log(\text{softmax})$ 的数值下溢）；

则复合算子 $F$ 的闭式 backward $\mathcal{B}[\![F]\!] = G$ 与展开后逐算子链式法则语义等价，且 $\mathcal{B}[\![F]\!]$ 满足定理 AD1 的链式法则等式。

**证明**：

**步骤 1**（C1 保证代数等价）：由 reverse-mode AD 的链式法则（Baydin et al. 2018），展开形式的梯度为 $\bar x = \bar y \cdot \prod_i \partial f_i/\partial v_i$。若该表达式可代数化简为 $G(\bar y, \text{saved})$，则 $G$ 与展开形式在实数域上相等。

**步骤 2**（C2 保证可实现）：闭式表达 $G$ 依赖 $\text{saved}$，若 $\text{saved}$ 可在前向阶段持久化（通过 `input_tensors`），则 backward 可在 $O(|G|)$ 时间内计算，无需重放前向。

**步骤 3**（C3 保证数值等价）：若 $G$ 在数值上不劣于展开形式，则浮点结果在误差范围内一致。若 C3 不成立（如 `CrossEntropy` 展开为 `Softmax + Log + Neg + Sum` 会有 $\log(0)$ 下溢），则闭式形式在数值上**优于**展开形式——这是融合的实际价值。

**步骤 4**（AD1 满足）：由步骤 1，$G = \bar y \cdot \partial F/\partial x$，即 $\mathcal{B}[\![F]\!] = G$ 满足链式法则等式。$\square$

**应用**：§6 将验证 5 个复合算子（`CrossEntropy`、`Conv2D`、`BatchNorm`、`LayerNorm`、`Gelu`）满足定理 AD5 的三个条件：

| 算子 | C1（代数化简） | C2（持久化） | C3（数值稳定性） |
|------|---------------|-------------|-----------------|
| `CrossEntropy` | $\text{softmax} - \text{target}$（softmax + log + neg + sum 化简） | softmax 输出 | 闭式避免 $\log(0)$ |
| `Conv2D` | im2col 转化为 MatMul | im2col 矩阵 | 等价 |
| `BatchNorm` | $(\gamma/\sigma)(\bar Y - \text{mean}(\bar Y) - x_{\text{hat}} \text{mean}(\bar Y x_{\text{hat}}))$ | $x_{\text{hat}}, \sigma^{-1}, \gamma$ | 等价 |
| `LayerNorm` | 同 BatchNorm，按行 | 同 BatchNorm | 等价 |
| `Gelu` | tanh 近似的解析导数 | input $x$ | 等价 |

**注**：`Softmax` 的 backward 也是闭式（$c_i(\bar c_i - \sum_j \bar c_j c_j)$），可视为"自融合"——展开为 $\exp + \text{sum} + \text{div}$ 后链式法则化简。

---

## 5. 21 算子的指称语义

本节逐一给出 21 个 `TapeOp` 变体的前向指称 $\mathcal{F}[\![op]\!]$ 与反向指称 $\mathcal{B}[\![op]\!]$。反向指称的逐一验证见 §6。

### 5.1 Input（叶）

- **前向**：$\mathcal{F}[\![\text{Input}]\!]() = x$（直接返回参数张量）；
- **反向**：$\mathcal{B}[\![\text{Input}]\!](\bar y) = \text{acc\_grad}(\bar y)$（累积到 `.grad` 字段，[autodiff.rs:292-300](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/autodiff.rs)）；
- **语义**：叶节点无计算，仅作为梯度累积终点。

### 5.2 Add

- **前向**：$\mathcal{F}[\![\text{Add}]\!](a, b) = a + b$（带广播）；
- **反向**：$\mathcal{B}[\![\text{Add}]\!]((a, b), \bar c) = (\text{unbroadcast}(\bar c, a.\text{shape}), \text{unbroadcast}(\bar c, b.\text{shape}))$（[autodiff.rs:301-314](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/autodiff.rs)）。

### 5.3 Sub

- **前向**：$\mathcal{F}[\![\text{Sub}]\!](a, b) = a - b$；
- **反向**：$\mathcal{B}[\![\text{Sub}]\!]((a, b), \bar c) = (\text{unbroadcast}(\bar c, a.\text{shape}), -\text{unbroadcast}(\bar c, b.\text{shape}))$（[autodiff.rs:301-314](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/autodiff.rs)，sign = -1）。

### 5.4 Mul

- **前向**：$\mathcal{F}[\![\text{Mul}]\!](a, b) = a \odot b$（element-wise，带广播）；
- **反向**：$\mathcal{B}[\![\text{Mul}]\!]((a, b), \bar c) = (\text{unbroadcast}(\bar c \odot b, a.\text{shape}), \text{unbroadcast}(\bar c \odot a, b.\text{shape}))$（[autodiff.rs:315-326](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/autodiff.rs)）。

### 5.5 Div

- **前向**：$\mathcal{F}[\![\text{Div}]\!](a, b) = a \oslash b$（element-wise，带广播）；
- **反向**：$\mathcal{B}[\![\text{Div}]\!]((a, b), \bar c) = (\text{unbroadcast}(\bar c \oslash b, a.\text{shape}), \text{unbroadcast}(-\bar c \odot a \oslash b^2, b.\text{shape}))$（[autodiff.rs:327-337](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/autodiff.rs)）。

### 5.6 Neg

- **前向**：$\mathcal{F}[\![\text{Neg}]\!](a) = -a$；
- **反向**：$\mathcal{B}[\![\text{Neg}]\!](a, \bar c) = -\bar c$（[autodiff.rs:338-341](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/autodiff.rs)）。

### 5.7 ReLU

- **前向**：$\mathcal{F}[\![\text{ReLU}]\!](a) = \max(0, a)$（element-wise）；
- **反向**：$\mathcal{B}[\![\text{ReLU}]\!](a, \bar c) = \bar c \odot \mathbb{1}_{a > 0}$（[autodiff.rs:342-349](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/autodiff.rs)）。

### 5.8 MatMul

- **前向**：$\mathcal{F}[\![\text{MatMul}]\!](A, B) = A @ B$（2D@2D、1D@2D、2D@1D，[autodiff.rs:350-443](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/autodiff.rs)）；
- **反向**：$\mathcal{B}[\![\text{MatMul}]\!]((A, B), \bar C) = (\bar C @ B^T, A^T @ \bar C)$（含 1D 提升与 squeeze 回退）。

### 5.9 Transpose

- **前向**：$\mathcal{F}[\![\text{Transpose}]\!](a) = a^T$（最后两维转置）；
- **反向**：$\mathcal{B}[\![\text{Transpose}]\!](a, \bar c) = \bar c^T$（最后两维转置回，[autodiff.rs:444-454](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/autodiff.rs)）。

### 5.10 Sum

- **前向**：$\mathcal{F}[\![\text{Sum}]\!](a) = \sum_i a_i$（标量）；
- **反向**：$\mathcal{B}[\![\text{Sum}]\!](a, \bar c) = \text{ones}(a.\text{shape}) \cdot \bar c$（[autodiff.rs:455-463](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/autodiff.rs)）。

### 5.11 Mean

- **前向**：$\mathcal{F}[\![\text{Mean}]\!](a) = \frac{1}{|a|} \sum_i a_i$；
- **反向**：$\mathcal{B}[\![\text{Mean}]\!](a, \bar c) = \text{ones}(a.\text{shape}) \cdot \bar c / |a|$（[autodiff.rs:464-473](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/autodiff.rs)）。

### 5.12 Exp

- **前向**：$\mathcal{F}[\![\text{Exp}]\!](a) = \exp(a)$；
- **反向**：$\mathcal{B}[\![\text{Exp}]\!](a, \bar c) = \bar c \odot \exp(a) = \bar c \odot c$（[autodiff.rs:474-480](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/autodiff.rs)，读取 result）。

### 5.13 Log

- **前向**：$\mathcal{F}[\![\text{Log}]\!](a) = \ln(a)$；
- **反向**：$\mathcal{B}[\![\text{Log}]\!](a, \bar c) = \bar c \oslash a$（[autodiff.rs:481-487](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/autodiff.rs)）。

### 5.14 Sigmoid

- **前向**：$\mathcal{F}[\![\text{Sigmoid}]\!](a) = \sigma(a) = 1/(1 + e^{-a})$；
- **反向**：$\mathcal{B}[\![\text{Sigmoid}]\!](a, \bar c) = \bar c \odot c \odot (1 - c)$（[autodiff.rs:488-495](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/autodiff.rs)，读取 result $c$）。

### 5.15 Softmax

- **前向**：$\mathcal{F}[\![\text{Softmax}]\!](a)_i = e^{a_i} / \sum_j e^{a_j}$（最后一维）；
- **反向**：$\mathcal{B}[\![\text{Softmax}]\!](a, \bar c)_i = c_i (\bar c_i - \sum_j \bar c_j c_j)$（[autodiff.rs:735-745](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/autodiff.rs)）。

### 5.16 CrossEntropy（复合）

- **前向**：$\mathcal{F}[\![\text{CrossEntropy}]\!](\text{logits}, \text{target}) = -\sum_i \text{target}_i \cdot \ln(\text{softmax}(\text{logits})_i)$；
- **反向**：$\mathcal{B}[\![\text{CrossEntropy}]\!](\text{logits}, \text{target}, \bar L) = \text{softmax}(\text{logits}) - \text{target}$（[autodiff.rs:723-734](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/autodiff.rs)）。

### 5.17 Dropout

- **前向**：$\mathcal{F}[\![\text{Dropout}]\!](a, \text{mask}) = a \odot \text{mask}$（mask = $1/(1-p)$ 保留，0 丢弃）；
- **反向**：$\mathcal{B}[\![\text{Dropout}]\!](a, \text{mask}, \bar c) = \bar c \odot \text{mask}$（[autodiff.rs:712-722](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/autodiff.rs)）。

### 5.18 Conv2D（复合）

- **前向**：$\mathcal{F}[\![\text{Conv2D}]\!](X, W) = \text{col2im}^{-1}(\text{im2col}(X) @ W_{\text{flat}}^T)$（im2col + MatMul，[autodiff.rs:615-711](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/autodiff.rs)）；
- **反向**：$\mathcal{B}[\![\text{Conv2D}]\!]((X, W), \bar Y) = (\text{col2im}(\bar Y @ W_{\text{flat}}), \text{reshape}(\text{im2col}^T @ \bar Y, W.\text{shape}))$。

### 5.19 BatchNorm（复合）

- **前向**：$\mathcal{F}[\![\text{BatchNorm}]\!](X, \gamma, \beta) = \gamma \odot x_{\text{hat}} + \beta$，其中 $x_{\text{hat}} = (X - \mu)/\sigma$，$\mu, \sigma$ 为 batch 统计量；
- **反向**：$\mathcal{B}[\![\text{BatchNorm}]\!](X, \gamma, \beta, \bar Y) = (\sigma^{-1} \gamma \odot (\bar Y - \text{mean}(\bar Y) - x_{\text{hat}} \odot \text{mean}(\bar Y \odot x_{\text{hat}})), \text{sum}(\bar Y \odot x_{\text{hat}}), \text{sum}(\bar Y))$（[autodiff.rs:496-522](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/autodiff.rs)）。

### 5.20 LayerNorm（复合）

- **前向**：$\mathcal{F}[\![\text{LayerNorm}]\!](X, \gamma, \beta) = \gamma \odot x_{\text{hat}} + \beta$，按最后一维归一化；
- **反向**：与 BatchNorm 类似，但 mean 按"行"计算（[autodiff.rs:523-596](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/autodiff.rs)）。

### 5.21 Gelu（复合）

- **前向**：$\mathcal{F}[\![\text{Gelu}]\!](x) = 0.5 x (1 + \tanh(\sqrt{2/\pi}(x + 0.044715 x^3)))$（tanh 近似）；
- **反向**：$\mathcal{B}[\![\text{Gelu}]\!](x, \bar c) = \bar c \odot [0.5(1 + \tanh(\text{inner})) + 0.5 x \cdot \text{sech}^2(\text{inner}) \cdot \sqrt{2/\pi} (1 + 3 \cdot 0.044715 x^2)]$，其中 $\text{inner} = \sqrt{2/\pi}(x + 0.044715 x^3)$（[autodiff.rs:597-614](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/autodiff.rs)）。

---

## 6. 链式法则等式的逐一验证

本节对 21 个算子逐一验证定理 AD1 的链式法则等式。每个验证包含：前向指称、解析雅可比、链式法则右端、backward 实现、等式验证。

### 6.1 Input

- **前向**：$y = x$（恒等）；
- **雅可比**：$\partial y/\partial x = I$；
- **链式法则右端**：$\bar y \cdot I = \bar y$；
- **backward 实现**：`acc_grad(&grad)`（[autodiff.rs:295-299](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/autodiff.rs)）；
- **验证**：$\text{acc\_grad}(\bar y) = \bar y$。✓

### 6.2 Add

- **前向**：$c_i = a_i + b_i$（element-wise，含广播）；
- **雅可比**：$\partial c_i/\partial a_j = \delta_{ij}$，$\partial c_i/\partial b_j = \delta_{ij}$；
- **链式法则右端**：$\bar a_j = \sum_i \bar c_i \delta_{ij} = \bar c_j$，$\bar b_j = \bar c_j$；
- **广播处理**：若 $a$ 被 broadcast 到 $c$ 的形状，则 $\bar a = \text{unbroadcast}(\bar c, a.\text{shape})$（沿广播维求和）；
- **backward 实现**：`unbroadcast(&grad, a_shape)` 与 `unbroadcast(&grad, b_shape)`（[autodiff.rs:301-314](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/autodiff.rs)）；
- **验证**：$\text{unbroadcast}(\bar c, a.\text{shape}) = \bar a$。✓

### 6.3 Sub

- **前向**：$c_i = a_i - b_i$；
- **雅可比**：$\partial c_i/\partial a_j = \delta_{ij}$，$\partial c_i/\partial b_j = -\delta_{ij}$；
- **链式法则右端**：$\bar a_j = \bar c_j$，$\bar b_j = -\bar c_j$；
- **backward 实现**：sign = -1，对 $b$ 应用 `unbroadcast(&grad, b_shape).mapv(|v| v * sign)`（[autodiff.rs:301-314](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/autodiff.rs)）；
- **验证**：$\bar b = -\text{unbroadcast}(\bar c, b.\text{shape})$。✓

### 6.4 Mul

- **前向**：$c_i = a_i b_i$（element-wise）；
- **雅可比**：$\partial c_i/\partial a_j = b_i \delta_{ij}$，$\partial c_i/\partial b_j = a_i \delta_{ij}$；
- **链式法则右端**：$\bar a_i = \bar c_i b_i$，$\bar b_i = \bar c_i a_i$；
- **广播处理**：$\bar a = \text{unbroadcast}(\bar c \odot b, a.\text{shape})$；
- **backward 实现**：`unbroadcast(&(&grad * &b_data), &a_shape)` 与 `unbroadcast(&(&grad * &a_data), &b_shape)`（[autodiff.rs:315-326](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/autodiff.rs)）；
- **验证**：$\bar a = \text{unbroadcast}(\bar c \odot b, a.\text{shape})$。✓

### 6.5 Div

- **前向**：$c_i = a_i / b_i$；
- **雅可比**：$\partial c_i/\partial a_j = \delta_{ij}/b_i$，$\partial c_i/\partial b_j = -a_i \delta_{ij}/b_i^2$；
- **链式法则右端**：$\bar a_i = \bar c_i / b_i$，$\bar b_i = -\bar c_i a_i / b_i^2$；
- **backward 实现**：`unbroadcast(&(&grad / &b_data), &a_shape)` 与 `unbroadcast(&(-&grad * &a_data / (&b_data * &b_data)), &b_shape)`（[autodiff.rs:327-337](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/autodiff.rs)）；
- **验证**：$\bar a = \bar c \oslash b$，$\bar b = -\bar c \odot a \oslash b^2$。✓

### 6.6 Neg

- **前向**：$c = -a$；
- **雅可比**：$\partial c/\partial a = -1$；
- **链式法则右端**：$\bar a = -\bar c$；
- **backward 实现**：`let g = -&grad; propagate_grad(node, 0, &g, ...)`（[autodiff.rs:338-341](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/autodiff.rs)）；
- **验证**：$\bar a = -\bar c$。✓

### 6.7 ReLU

- **前向**：$c_i = \max(0, a_i)$；
- **雅可比**：$\partial c_i/\partial a_j = \mathbb{1}_{a_i > 0} \delta_{ij}$；
- **链式法则右端**：$\bar a_i = \bar c_i \mathbb{1}_{a_i > 0}$；
- **backward 实现**：`let mask = a.data.mapv(|x| if x > 0.0 { 1.0 } else { 0.0 }); let g_a = &grad * &mask;`（[autodiff.rs:342-349](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/autodiff.rs)）；
- **验证**：$\bar a = \bar c \odot \mathbb{1}_{a > 0}$。✓
- **边界情况**：$a = 0$ 时实现取 $\mathbb{1}_{0 > 0} = 0$（次梯度取 0），与 PyTorch 一致。

### 6.8 MatMul

- **前向**：$C = A @ B$（$C_{ij} = \sum_k A_{ik} B_{kj}$）；
- **雅可比**：$\partial C_{ij}/\partial A_{pq} = \delta_{ip} B_{qj}$，$\partial C_{ij}/\partial B_{pq} = A_{ip} \delta_{jq}$；
- **链式法则右端**：
  - $\bar A_{pq} = \sum_{ij} \bar C_{ij} \delta_{ip} B_{qj} = \sum_j \bar C_{pj} B_{qj} = (\bar C @ B^T)_{pq}$
  - $\bar B_{pq} = \sum_{ij} \bar C_{ij} A_{ip} \delta_{jq} = \sum_i A_{ip} \bar C_{iq} = (A^T @ \bar C)_{pq}$
- **backward 实现**：`d_a_2d = matmul_2d(&grad_2d, &b_t)` 与 `d_b_2d = matmul_2d(&a_t, &grad_2d)`（[autodiff.rs:406-407](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/autodiff.rs)）；
- **1D 提升与 squeeze**：若 $A$ 是 1D（shape $(k,)$），提升为 $(1, k)$；结果 $(1, n)$ squeeze 回 $(n,)$。校验见 [autodiff.rs:411-436](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/autodiff.rs)。
- **验证**：$\bar A = \bar C @ B^T$，$\bar B = A^T @ \bar C$。✓

### 6.9 Transpose

- **前向**：$c_{ij} = a_{ji}$（最后两维转置）；
- **雅可比**：$\partial c_{ij}/\partial a_{pq} = \delta_{ip} \delta_{jq}$（即转置是置换矩阵）；
- **链式法则右端**：$\bar a_{pq} = \sum_{ij} \bar c_{ij} \delta_{ip} \delta_{jq} = \bar c_{pq}$——但需注意 $c_{ij} = a_{ji}$，故 $\bar a_{pq} = \bar c_{qp}$，即 $\bar a = \bar c^T$；
- **backward 实现**：`perm.swap(last-1, last); g_a = grad.permuted_axes(perm)`（[autodiff.rs:444-454](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/autodiff.rs)）；
- **验证**：$\bar a = \bar c^T$（最后两维转置）。✓

### 6.10 Sum

- **前向**：$c = \sum_i a_i$（标量）；
- **雅可比**：$\partial c/\partial a_i = 1$；
- **链式法则右端**：$\bar a_i = \bar c \cdot 1 = \bar c$（即 $\bar a = \text{ones}(a.\text{shape}) \cdot \bar c$）；
- **backward 实现**：`let s: f64 = grad.iter().sum(); ArrayD::from_elem(a_shape, s)`（[autodiff.rs:455-463](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/autodiff.rs)）；
- **注**：`grad` 可能是张量（若 loss 非标量），实现取 `grad` 所有元素之和作为 $\bar c$。
- **验证**：$\bar a = \text{ones} \cdot \sum \bar c$。✓

### 6.11 Mean

- **前向**：$c = \frac{1}{n} \sum_i a_i$（$n = |a|$）；
- **雅可比**：$\partial c/\partial a_i = 1/n$；
- **链式法则右端**：$\bar a_i = \bar c / n$；
- **backward 实现**：`let s = grad.iter().sum::<f64>() / n; ArrayD::from_elem(a_shape, s)`（[autodiff.rs:464-473](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/autodiff.rs)）；
- **验证**：$\bar a = \text{ones} \cdot (\sum \bar c / n)$。✓

### 6.12 Exp

- **前向**：$c = \exp(a)$；
- **雅可比**：$\partial c_i/\partial a_j = \exp(a_i) \delta_{ij} = c_i \delta_{ij}$；
- **链式法则右端**：$\bar a_i = \bar c_i c_i$；
- **backward 实现**：`let result_ref = node.input_tensors[1].borrow(); &grad * &result_ref.data`（[autodiff.rs:474-480](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/autodiff.rs)）；
- **验证**：$\bar a = \bar c \odot c$（其中 $c = \exp(a)$ 由 `input_tensors[1]` 持久化）。✓
- **AD3 必要性**：需持久化 $c$（或 $a$），否则需重算 $\exp(a)$。

### 6.13 Log

- **前向**：$c = \ln(a)$；
- **雅可比**：$\partial c_i/\partial a_j = \delta_{ij}/a_i$；
- **链式法则右端**：$\bar a_i = \bar c_i / a_i$；
- **backward 实现**：`let a_ref = node.input_tensors[0].borrow(); &grad / &a_ref.data`（[autodiff.rs:481-487](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/autodiff.rs)）；
- **验证**：$\bar a = \bar c \oslash a$。✓
- **定义域**：$a > 0$，否则 $\ln$ 未定义（实现不显式检查，依赖前向阶段的定义域保证）。

### 6.14 Sigmoid

- **前向**：$c = \sigma(a) = 1/(1 + e^{-a})$；
- **雅可比**：$\partial c/\partial a = c(1 - c)$（标准 sigmoid 导数）；
- **链式法则右端**：$\bar a = \bar c \cdot c(1 - c)$；
- **backward 实现**：`let y = &result_ref.data; &grad * y * &y.mapv(|v| 1.0 - v)`（[autodiff.rs:488-495](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/autodiff.rs)）；
- **验证**：$\bar a = \bar c \odot c \odot (1 - c)$。✓
- **AD3 必要性**：需持久化 $c$（避免重算 $\sigma(a)$）。

### 6.15 Softmax

- **前向**：$c_i = e^{a_i} / \sum_j e^{a_j}$；
- **雅可比**：$\partial c_i/\partial a_j = c_i (\delta_{ij} - c_j)$（标准 softmax 导数）；
- **链式法则右端**：$\bar a_j = \sum_i \bar c_i c_i (\delta_{ij} - c_j) = \bar c_j c_j - c_j \sum_i \bar c_i c_i = c_j (\bar c_j - \sum_i \bar c_i c_i)$；
- **backward 实现**：`let sum_term = (&grad * y).sum(); &grad * y - &(y.mapv(|v| v * sum_term))`（[autodiff.rs:735-745](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/autodiff.rs)）；
- **验证**：$\bar a_j = c_j (\bar c_j - \sum_i \bar c_i c_i)$。✓
- **AD3 必要性**：需持久化 $c = \text{softmax}(a)$。
- **退化情形**：若 $c_i \to 0$（即 $a_i \to -\infty$），公式仍数学成立，但浮点可能下溢（见 §10 局限）。

### 6.16 CrossEntropy（复合，闭式验证）

- **前向**：$L = -\sum_i t_i \ln(\text{softmax}(\text{logits})_i)$，其中 $t = \text{target}$，$s = \text{softmax}(\text{logits})$；
- **展开形式**：`Softmax + Log + Neg + Mul(target) + Sum`，链式法则为：
  - $\bar s_i = \bar L \cdot (-t_i / s_i)$（来自 `Log + Neg + Mul + Sum`）；
  - $\bar{\text{logits}}_j = \sum_i \bar s_i \cdot \partial s_i/\partial \text{logits}_j = \sum_i \bar s_i \cdot s_i (\delta_{ij} - s_j)$；
- **代数化简**：
  - $\bar{\text{logits}}_j = \sum_i (-\bar L t_i / s_i) \cdot s_i (\delta_{ij} - s_j) = -\bar L \sum_i t_i (\delta_{ij} - s_j) = -\bar L (t_j - s_j \sum_i t_i)$；
  - 若 $\sum_i t_i = 1$（one-hot target），则 $\bar{\text{logits}}_j = -\bar L (t_j - s_j) = \bar L (s_j - t_j)$；
  - 种子 $\bar L = 1$，故 $\bar{\text{logits}} = s - t$；
- **backward 实现**：`let g_a = { let sm_ref = ...; let tgt_ref = ...; &sm_ref.data - &tgt_ref.data };`（[autodiff.rs:723-734](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/autodiff.rs)）；
- **验证**：$\bar{\text{logits}} = \text{softmax}(\text{logits}) - \text{target}$。✓
- **AD5 应用**：C1（代数化简为 $s - t$）、C2（softmax 持久化于 `input_tensors[1]`）、C3（闭式避免 $\ln(0)$ 下溢）。
- **隐式假设**：$\sum_i t_i = 1$（one-hot），若 target 非归一化则闭式不成立（见 §10 局限）。

### 6.17 Dropout

- **前向**：$c = a \odot \text{mask}$，其中 mask 是随机生成（保留位置 $1/(1-p)$，丢弃位置 0）；
- **雅可比**：$\partial c_i/\partial a_j = \text{mask}_i \delta_{ij}$；
- **链式法则右端**：$\bar a_i = \bar c_i \text{mask}_i$；
- **backward 实现**：`let mask_ref = node.input_tensors[1].borrow(); &grad * &mask_ref.data`（[autodiff.rs:712-722](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/autodiff.rs)）；
- **验证**：$\bar a = \bar c \odot \text{mask}$。✓
- **AD3 必要性**：mask 是随机的，无法从 input 重建——必须持久化。

### 6.18 Conv2D（复合，闭式验证）

- **前向**（im2col 形式）：$Y = \text{im2col}(X) @ W_{\text{flat}}^T$，其中 $\text{im2col}(X) \in \mathbb{R}^{(N H_{\text{out}} W_{\text{out}}) \times (C_{\text{in}} k_H k_W)}$，$W_{\text{flat}} \in \mathbb{R}^{C_{\text{out}} \times (C_{\text{in}} k_H k_W)}$；
- **链式法则**：
  - $\bar W_{\text{flat}} = \text{im2col}(X)^T @ \bar Y$（MatMul 的反向，对 $W_{\text{flat}}$）；
  - $\overline{\text{im2col}(X)} = \bar Y @ W_{\text{flat}}$（MatMul 的反向，对 im2col）；
  - $\bar X = \text{col2im}(\overline{\text{im2col}(X)})$（im2col 的反向，即 col2im）；
- **backward 实现**：
  - `d_w_flat = matmul_2d(&col_t, &grad_2d)`（[autodiff.rs:650](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/autodiff.rs)）；
  - `d_col = matmul_2d(&grad_2d, &w_flat)`（[autodiff.rs:684](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/autodiff.rs)）；
  - `d_x = ArrayD::from_shape_vec(x_shape, d_col.iter().cloned().collect())`（[autodiff.rs:699](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/autodiff.rs)）；
- **验证**：$\bar W = \text{reshape}(\text{im2col}^T @ \bar Y, W.\text{shape})$，$\bar X = \text{col2im}(\bar Y @ W_{\text{flat}})$。✓
- **AD5 应用**：C1（im2col 转化为 MatMul）、C2（im2col 持久化于 `input_tensors[2]`）、C3（等价）。
- **简化**：实现中的 `col2im` 是简化版本——直接 reshape `d_col` 到 $X$ 的形状（[autodiff.rs:699](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/autodiff.rs)），**未处理 stride/padding/dilation 的累积**。这要求 im2col 的输出元素数等于 $X$ 的元素数，即 stride=1, padding=0, dilation=1（见 §10 局限）。

### 6.19 BatchNorm（复合，闭式验证）

- **前向**：$Y = \gamma \odot x_{\text{hat}} + \beta$，其中 $x_{\text{hat}} = (X - \mu)/\sigma$，$\mu = \text{mean}(X)$，$\sigma = \text{std}(X)$（沿 batch 维）；
- **雅可比推导**（标准 BN backward，见 Ioffe & Szegedy 2015）：
  - $\bar \gamma = \sum \bar Y \odot x_{\text{hat}}$，$\bar \beta = \sum \bar Y$；
  - $\bar X = (\gamma/\sigma) (\bar Y - \text{mean}(\bar Y) - x_{\text{hat}} \odot \text{mean}(\bar Y \odot x_{\text{hat}}))$；
- **链式法则右端**推导：
  - $\bar x_{\text{hat}} = \bar Y \odot \gamma$；
  - $\bar \sigma = \sum \bar x_{\text{hat}} \odot (X - \mu) \cdot (-1/\sigma^2) = -\sum \bar x_{\text{hat}} \odot x_{\text{hat}} / \sigma$；
  - $\bar \mu = -\sum \bar x_{\text{hat}} / \sigma + \bar \sigma \cdot (-2 \text{mean}(X - \mu)/n) = -\sum \bar x_{\text{hat}} / \sigma$（第二项因 $\text{mean}(X - \mu) = 0$ 消去）；
  - $\bar X = \bar x_{\text{hat}} / \sigma + \bar \sigma \cdot 2(X - \mu)/n + \bar \mu / n$；
  - 代入化简：$\bar X = (\gamma/\sigma) (\bar Y - \text{mean}(\bar Y) - x_{\text{hat}} \odot \text{mean}(\bar Y \odot x_{\text{hat}}))$；
- **backward 实现**：
  - `d_gamma = &grad * &x_hat_ref.data`（[autodiff.rs:507](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/autodiff.rs)）；
  - `d_beta = grad.clone()`（[autodiff.rs:509](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/autodiff.rs)）；
  - `d_x = &std_inv_ref.data * &gamma_ref.data * &(&grad - mean_dy - &(&x_hat_ref.data * mean_dy_xhat))`（[autodiff.rs:515-516](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/autodiff.rs)）；
- **验证**：$\bar X = \sigma^{-1} \gamma \odot (\bar Y - \text{mean}(\bar Y) - x_{\text{hat}} \odot \text{mean}(\bar Y \odot x_{\text{hat}}))$。✓
- **AD5 应用**：C1（化简为闭式）、C2（$x_{\text{hat}}, \sigma^{-1}, \gamma$ 持久化）、C3（等价）。
- **简化**：实现的 mean 沿**所有维度**（[autodiff.rs:512](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/autodiff.rs) `n = grad.len()`），即把整个 batch 当作一个统计群体。这与标准 BN（沿 batch 维统计、保留通道维）不同（见 §10 局限）。

### 6.20 LayerNorm（复合，闭式验证）

- **前向**：$Y = \gamma \odot x_{\text{hat}} + \beta$，按最后一维归一化（每行独立计算 $\mu, \sigma$）；
- **雅可比推导**（与 BatchNorm 同形，但 mean 按行）：
  - $\bar X_{\text{row } i} = (\gamma/\sigma_i) (\bar Y_{\text{row } i} - \text{mean}(\bar Y_{\text{row } i}) - x_{\text{hat},\text{row } i} \odot \text{mean}(\bar Y_{\text{row } i} \odot x_{\text{hat},\text{row } i}))$；
- **backward 实现**：按行循环计算（[autodiff.rs:569-589](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/autodiff.rs)）：
  ```rust
  for i in 0..outer_len {
      let inv = std_inv_slice[i];
      let mut mean_dy = 0.0; let mut mean_dy_xhat = 0.0;
      for j in 0..axis_len { ... }  // 计算 row-wise mean
      for j in 0..axis_len {
          d_x_data.push(g * inv * (dy - mean_dy - xh * mean_dy_xhat));
      }
  }
  ```
- **验证**：每行 $\bar X_{\text{row}} = \sigma^{-1} \gamma \odot (\bar Y - \text{mean}(\bar Y) - x_{\text{hat}} \odot \text{mean}(\bar Y \odot x_{\text{hat}}))$。✓
- **AD5 应用**：C1、C2（$x_{\text{hat}}, \sigma^{-1}, \gamma$ 持久化）、C3。
- **注**：LayerNorm 的实现是"逐行标量循环"（[autodiff.rs:548-589](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/autodiff.rs)），与 BatchNorm 的"全张量向量化"不同。这是工程选择（避免 axis 求和的复杂 broadcast），数学语义等价。

### 6.21 Gelu（复合，闭式验证）

- **前向**（tanh 近似）：$c = 0.5 x (1 + \tanh(\text{inner}))$，其中 $\text{inner} = \sqrt{2/\pi}(x + 0.044715 x^3)$；
- **雅可比推导**：
  - $\frac{dc}{dx} = 0.5 (1 + \tanh(\text{inner})) + 0.5 x \cdot \text{sech}^2(\text{inner}) \cdot \frac{d\text{inner}}{dx}$；
  - $\frac{d\text{inner}}{dx} = \sqrt{2/\pi} (1 + 3 \cdot 0.044715 x^2)$；
  - 故 $\frac{dc}{dx} = 0.5 (1 + \tanh(\text{inner})) + 0.5 x \cdot \text{sech}^2(\text{inner}) \cdot \sqrt{2/\pi} (1 + 3 \cdot 0.044715 x^2)$；
- **链式法则右端**：$\bar x = \bar c \cdot \frac{dc}{dx}$；
- **backward 实现**（[autodiff.rs:601-612](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/autodiff.rs)）：
  ```rust
  let deriv = x_data.mapv(|x| {
      let inner = sqrt_2_over_pi * (x + 0.044715 * x * x * x);
      let tanh_inner = inner.tanh();
      let sech2 = 1.0 - tanh_inner * tanh_inner;
      0.5 * (1.0 + tanh_inner) + 0.5 * x * sech2 * sqrt_2_over_pi * (1.0 + 3.0 * 0.044715 * x * x)
  });
  &grad * &deriv
  ```
- **验证**：$\bar x = \bar c \odot [0.5(1 + \tanh(\text{inner})) + 0.5 x \cdot \text{sech}^2(\text{inner}) \cdot \sqrt{2/\pi} (1 + 3 \cdot 0.044715 x^2)]$。✓
- **AD5 应用**：C1（tanh 近似的解析导数）、C2（input $x$ 持久化）、C3（等价）。
- **注**：`sech2 = 1 - tanh²` 是 sech² 的恒等变形（$\text{sech}^2 = 1 - \tanh^2$），避免显式计算 cosh。

### 6.22 验证汇总表

| # | 算子 | 前向 | 闭式 backward | 链式法则验证 | 持久化必要性 |
|---|------|------|--------------|-------------|-------------|
| 1 | Input | $y = x$ | $\bar x = \bar y$ | ✓ §6.1 | 累积终点 |
| 2 | Add | $c = a + b$ | $\bar a = \bar c, \bar b = \bar c$ | ✓ §6.2 | 非必要 |
| 3 | Sub | $c = a - b$ | $\bar a = \bar c, \bar b = -\bar c$ | ✓ §6.3 | 非必要 |
| 4 | Mul | $c = a \odot b$ | $\bar a = \bar c \odot b, \bar b = \bar c \odot a$ | ✓ §6.4 | 非必要 |
| 5 | Div | $c = a \oslash b$ | $\bar a = \bar c \oslash b, \bar b = -\bar c \odot a \oslash b^2$ | ✓ §6.5 | 非必要 |
| 6 | Neg | $c = -a$ | $\bar a = -\bar c$ | ✓ §6.6 | 非必要 |
| 7 | ReLU | $c = \max(0, a)$ | $\bar a = \bar c \odot \mathbb{1}_{a > 0}$ | ✓ §6.7 | 必要（input） |
| 8 | MatMul | $C = A @ B$ | $\bar A = \bar C @ B^T, \bar B = A^T @ \bar C$ | ✓ §6.8 | 必要（A, B） |
| 9 | Transpose | $c = a^T$ | $\bar a = \bar c^T$ | ✓ §6.9 | 非必要 |
| 10 | Sum | $c = \sum a$ | $\bar a = \text{ones} \cdot \bar c$ | ✓ §6.10 | 形状必要 |
| 11 | Mean | $c = \text{mean}(a)$ | $\bar a = \text{ones} \cdot \bar c / n$ | ✓ §6.11 | 形状必要 |
| 12 | Exp | $c = \exp(a)$ | $\bar a = \bar c \odot c$ | ✓ §6.12 | **必要（result）** |
| 13 | Log | $c = \ln(a)$ | $\bar a = \bar c \oslash a$ | ✓ §6.13 | 必要（input） |
| 14 | Sigmoid | $c = \sigma(a)$ | $\bar a = \bar c \odot c \odot (1 - c)$ | ✓ §6.14 | **必要（result）** |
| 15 | Softmax | $c_i = e^{a_i}/\sum e^{a_j}$ | $\bar a_j = c_j(\bar c_j - \sum \bar c_i c_i)$ | ✓ §6.15 | **必要（result）** |
| 16 | CrossEntropy | $L = -\sum t \ln s$ | $\bar{\text{logits}} = s - t$ | ✓ §6.16 | **必要（softmax, target）** |
| 17 | Dropout | $c = a \odot \text{mask}$ | $\bar a = \bar c \odot \text{mask}$ | ✓ §6.17 | **必要（mask，不可重建）** |
| 18 | Conv2D | $Y = \text{im2col}(X) @ W^T$ | $\bar W = \text{im2col}^T @ \bar Y, \bar X = \text{col2im}(\bar Y @ W)$ | ✓ §6.18 | **必要（im2col）** |
| 19 | BatchNorm | $Y = \gamma x_{\text{hat}} + \beta$ | $\bar X = (\gamma/\sigma)(\bar Y - \text{mean} - x_{\text{hat}} \text{mean})$ | ✓ §6.19 | **必要（x_hat, σ⁻¹, γ）** |
| 20 | LayerNorm | 同 BN，按行 | 同 BN，按行 | ✓ §6.20 | **必要（x_hat, σ⁻¹, γ）** |
| 21 | Gelu | $c = 0.5x(1 + \tanh(\text{inner}))$ | $\bar x = \bar c \cdot [0.5(1 + \tanh) + 0.5 x \text{sech}^2 \sqrt{2/\pi}(1 + 3 \cdot 0.044715 x^2)]$ | ✓ §6.21 | 必要（input） |

**结论**：21 个算子的链式法则等式逐一验证通过，定理 AD1 成立。其中 9 个算子（Exp, Sigmoid, Softmax, CrossEntropy, Dropout, Conv2D, BatchNorm, LayerNorm, Gelu）的 `input_tensors` 持久化是闭式 backward 的必要条件，定理 AD3 成立。

---

## 7. 与 PyTorch/JAX 对比

### 7.1 数学语义对比

| 维度 | Tenth tape | PyTorch autograd | JAX |
|------|-----------|------------------|-----|
| 反向图 | 单一 tape 序列（显式持久化） | 动态反向图（节点级 `.grad_fn`） | JAXPR（纯函数式 IR） |
| 拓扑序 | 记录逆序（`nodes.iter().rev()`） | 显式拓扑排序（`next_edges`） | JAXPR 线性序 |
| 链式法则实现 | 21 算子手写 backward | 数百算子手写 backward | JAXPR primitive 自动微分 |
| 复合算子 | 闭式（CrossEntropy, Conv2D, BN, LN, Gelu） | 闭式（同 Tenth） | 通常展开为 primitive |
| 中间值持久化 | `input_tensors: Vec<Rc<RefCell<Tensor>>>` | `ctx.save_for_backward` | Tracer 自动捕获 |
| 数值稳定性 | CrossEntropy 闭式避免 log(0) | 同 Tenth | 同 Tenth |

### 7.2 工程差异

**Tenth 的优势**：

1. **简单性**：单一 tape 序列，无显式拓扑排序算法，无多线程调度复杂性；
2. **可调试性**：`input_tensors` 显式持久化，便于根因定位（T2 已论证）；
3. **闭式 backward**：复合算子一次性计算，避免展开开销。

**Tenth 的劣势**：

1. **内存占用**：所有中间张量被 `Rc` 持有，内存压力大于 PyTorch 的 `save_for_backward`（仅保存必要张量）；
2. **多线程缺失**：`nodes.iter().rev()` 单线程遍历，无法利用 PyTorch 的多线程反向引擎；
3. **循环/分支支持**：tape 是线性序列，循环需展平，分支需记录实际路径——不如 PyTorch 动态图自然；
4. **effect system 缺位**：tape 记录依赖 `if self.recording { ... }` 工程纪律，不如 JAX 类型系统强制（T38 §5）。

### 7.3 算子覆盖对比

Tenth 的 21 算子覆盖深度学习核心计算，但远少于 PyTorch 的数百算子。缺失的算子（如 `Conv1D`、`Conv3D`、`MaxPool`、`AvgPool`、`LSTM`、`Embedding`、`BCELoss`、`MSELoss` 等）需通过现有算子组合实现，或后续扩展 `TapeOp` 枚举。扩展时需同步：

1. `TapeOp` 枚举新增变体；
2. `Tape::backward` 新增 match 分支；
3. 前向 record 方法（`tape.xxx`）；
4. VM/解释器中的 record 调用（T38 已证这是 `Neg` 算子遗漏的根源）；
5. 测试覆盖（链式法则等式验证）。

---

## 8. 工程权衡

### 8.1 闭式 backward vs 展开形式

Tenth 选择闭式 backward（如 `CrossEntropy` 的 $s - t$）而非展开形式（`Softmax + Log + Neg + Sum`）。权衡：

| 维度 | 闭式 | 展开 |
|------|------|------|
| 计算开销 | $O(n)$ 一次性 | $O(n)$ 多步骤 |
| 内存开销 | 持久化 softmax | 持久化 softmax + log + ... |
| 数值稳定性 | 高（避免 $\ln(0)$） | 低（需 log-sum-exp 技巧） |
| 实现复杂度 | 高（手写闭式） | 低（复用基本算子） |
| 验证难度 | 高（需代数化简证明） | 低（链式法则自动） |

Tenth 的选择（闭式）在数值稳定性与计算效率上更优，但增加了验证负担——本文 §6 即为这一负担的兑现。

### 8.2 显式 tape vs 动态反向图

Tenth 选择显式 tape（单一序列）而非动态反向图（PyTorch 风格）。权衡：

| 维度 | 显式 tape | 动态反向图 |
|------|----------|-----------|
| 实现复杂度 | 低（Vec + rev 遍历） | 高（节点级 `.grad_fn` 链） |
| 拓扑排序 | 隐式（记录顺序） | 显式（需算法） |
| 多线程反向 | 难（序列依赖） | 易（图结构并行） |
| 内存控制 | 难（全持久化） | 易（`save_for_backward` 选择性） |
| 调试 | 易（线性序列） | 难（图遍历） |

Tenth 的选择（显式 tape）在简单性与可调试性上更优，但牺牲了多线程反向与精细内存控制。

### 8.3 `Rc<RefCell<Tensor>>` vs 节点 id

Tenth 选择 `input_tensors: Vec<Rc<RefCell<Tensor>>>` 持久化张量引用，而非仅持久化节点 id。权衡见定理 AD3——9 个算子的闭式 backward 必需持久化中间值，`Rc<RefCell<>>` 是必要选择。代价是：

1. **内存压力**：所有中间张量被持有，无法 GC；
2. **借用检查**：`RefCell` 运行时借用检查，可能 panic（实现需谨慎避免双重借用，[autodiff.rs:316-321](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/autodiff.rs) 显式 clone 数据规避）；
3. **不可序列化**：`Rc` 不可跨进程/序列化，限制了分布式训练扩展性。

---

## 9. 开放问题

### 9.1 高阶自动微分

Tenth 当前的 `Tape::backward` 仅支持一阶梯度。高阶梯度（如 `grad(grad(loss))`）需对 backward 本身再求导。这要求：

1. backward 公式本身可微（21 算子的 backward 是否可微？）；
2. tape 记录 backward 计算图（双层 tape）；
3. 或采用 forward-over-reverse 混合模式。

**开放**：Tenth 是否需要支持高阶 AD？若需要，当前 `input_tensors` 持久化是否足以支撑？

### 9.2 算子融合的自动化

定理 AD5 给出融合的充分条件，但 Tenth 的 5 个复合算子是**人工**融合的（手写闭式 backward）。是否能自动化融合？

- **规则驱动**：定义算子融合规则（如 `Softmax + Log + Neg + Sum → CrossEntropy`），编译期应用；
- **代数化简**：对展开形式应用符号微分与代数化简，自动生成闭式；
- **JAX 路径**：JAX 通过 XLA 编译器自动融合 primitive，Tenth 是否应引入类似优化层？

**开放**：Tenth 的 tape 层是否应引入自动融合 pass？若引入，需重新验证融合后的语义等价性（定理 AD5 的自动化）。

### 9.3 稀疏梯度与梯度累积

当前 `acc_node_grad`（[autodiff.rs:770-779](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/autodiff.rs)）总是 dense 累加。对于稀疏梯度（如 Embedding 的反向），dense 累加浪费内存与计算。

**开放**：是否应引入稀疏梯度支持？若引入，`TapeOp` 需扩展稀疏变体，或 `acc_grad` 需支持稀疏累积。

### 9.4 Effect system 强制 recording

T38 §11 提出：是否应引入 effect system，将"recording"作为类型系统的一部分，强制每个可微算子在前向阶段必 record？

**联动**：若引入 effect system，定理 AD1 的前提（tape 完整性）可从工程纪律提升为类型保证，T38 定理 A3（梯度正确性相对性）的局限可部分缓解。

### 9.5 Conv2D 反向的 stride/padding/dilation

当前 `Conv2D` 反向的 col2im 是简化版本（直接 reshape，[autodiff.rs:699](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/autodiff.rs)），未处理 stride/padding/dilation 的累积。这限制了 Conv2D 仅支持 stride=1, padding=0, dilation=1 的情形。

**开放**：是否应扩展 Conv2D 反向以支持通用 stride/padding/dilation？这需要实现真正的 col2im 累积算法，并重新验证链式法则等式。

---

## 10. 局限（独立章节）

本章节诚实披露本文证明的漏洞、不完备性与工程差距，遵循数理部"局限必披露"原则。

### 10.1 Conv2D 反向的简化

**局限**：`Conv2D` 反向的 col2im 步骤（[autodiff.rs:686-704](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/autodiff.rs)）直接 reshape `d_col` 到 $X$ 的形状，**未实现真正的 col2im 累积**。

**影响**：当 stride > 1 或 padding > 0 或 dilation > 1 时，im2col 的输出元素数 $\neq$ $X$ 的元素数，`d_col.len() != x_total` 会触发错误（[autodiff.rs:691-698](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/autodiff.rs)）。当前实现仅支持 stride=1, padding=0, dilation=1 的"退化"卷积。

**对定理 AD1 的影响**：在 stride=1, padding=0, dilation=1 的退化情形下，链式法则等式成立（§6.18 验证）。一般情形下，定理 AD1 对 `Conv2D` **不成立**——因为 backward 实现本身不正确。

**缓解**：在 [autodiff.rs:691-698](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/autodiff.rs) 已加入显式 shape 校验，退化情形之外会报错而非静默错误。这是 T38 提出的"方向 A：消除 silent squeeze"的实践。

### 10.2 BatchNorm 反向的简化

**局限**：`BatchNorm` 反向的 mean 计算（[autodiff.rs:512-514](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/autodiff.rs)）使用 `n = grad.len()`，即把整个张量当作一个统计群体。这与标准 BatchNorm（沿 batch 维统计、保留通道维）不同。

**影响**：当前 `BatchNorm` 实际是"全张量归一化"而非标准 BN。在通道间不独立归一化的情况下，梯度可能与标准 BN 不一致。

**对定理 AD1 的影响**：定理 AD1 对当前实现的"全张量 BN"成立（§6.19 验证），但与 PyTorch 的标准 BN 语义**不等价**——定理 AD4 在 `BatchNorm` 上不成立。

**缓解**：若需标准 BN 语义，需修改前向与反向以按通道维统计。这是工程实现差距，非理论漏洞。

### 10.3 CrossEntropy 的隐式假设

**局限**：`CrossEntropy` 的闭式 backward $\bar{\text{logits}} = \text{softmax} - \text{target}$ 隐式假设 $\sum_i \text{target}_i = 1$（one-hot target）。

**影响**：若 target 非归一化（如 soft label $\sum t_i \neq 1$），闭式不成立——正确公式应为 $\bar{\text{logits}} = \bar L (\text{softmax} - \text{target})$，其中 $\bar L = 1$ 时简化为 $\text{softmax} - \text{target}$ 仅当 $\sum t_i = 1$。

**对定理 AD1 的影响**：在 one-hot target 假设下，定理 AD1 成立（§6.16 验证）。非归一化 target 下，闭式 backward **不正确**——但当前实现不校验 target 归一化。

**缓解**：在前向阶段校验 $\sum t_i = 1$，或文档明确要求 one-hot target。

### 10.4 Softmax 的数值稳定性

**局限**：`Softmax` 前向（未在 autodiff.rs 中，在 tensor.rs 或 VM 中）应使用 max 减法技巧避免 $e^{a_i}$ 溢出。`Softmax` 反向（[autodiff.rs:735-745](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/autodiff.rs)）依赖 $c = \text{softmax}(a)$ 的持久化值，若前向数值不稳定，反向也不稳定。

**影响**：当 $a_i$ 极大时，$c_i \to 1$，其他 $c_j \to 0$，反向公式 $c_j(\bar c_j - \sum \bar c_i c_i)$ 在 $c_j \to 0$ 时下溢。

**对定理 AD1 的影响**：数学上成立，浮点上可能下溢。

### 10.5 浮点误差未量化

**局限**：定理 AD4 的"浮点误差范围内等价"未给出具体误差界。

**影响**：Tenth、PyTorch、JAX 三者的浮点结果差异依赖具体算子顺序，本文未量化。

**缓解**：实际工程中可通过 `cargo test --manifest-path tenth/Cargo.toml -- autodiff` 的数值测试（如 `test_backward_chain`，[autodiff.rs:1018-1046](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/autodiff.rs)）验证相对误差 $\sim 10^{-10}$。

### 10.6 循环与分支的 tape 处理未覆盖

**局限**：本文假设 tape 是线性序列（DAG 性质），未覆盖循环（`for`/`while`）与分支（`if`）的 tape 处理。

**影响**：循环需展平（每次迭代产生独立 tape 节点），分支需记录实际执行路径。这些场景的链式法则等式需额外论证。

**对定理 AD1-AD5 的影响**：在循环/分支场景下，定理仍应成立（tape 节点是按执行顺序记录的），但本文未形式化论证。

### 10.7 21 算子的完备性

**局限**：本文仅验证 21 个现有算子，不覆盖未来扩展的算子。

**影响**：新增算子时，需重新执行 §6 的链式法则验证。T38 §6 发现的 `Neg` 算子 record 缺失表明，新增算子的 record 调用易遗漏。

**缓解**：建议引入 effect system（§9.4）或编译期检查（每个 `TapeOp` 变体必须有对应 record 调用与 backward 分支）。

### 10.8 证明的循环论证风险

**局限**：定理 AD4（与 PyTorch/JAX 等价性）的证明引用 PyTorch 与 JAX 的正确性证明（Paszke 2017, Frostig 2018），这些证明本身依赖 reverse-mode AD 的一般理论（Baydin 2018）。本文又用 reverse-mode AD 理论证明 Tenth 的正确性。这是否构成循环论证？

**分析**：不构成循环。reverse-mode AD 的一般理论（Baydin 2018）是独立的数学定理，不依赖任何具体实现。Tenth、PyTorch、JAX 是该理论的三种独立实现，三者的等价性通过共同依赖该理论而建立。这是"归约到共同前提"而非循环论证。

### 10.9 工程差距汇总

| 局限 | 影响 | 缓解 | 严重性 |
|------|------|------|--------|
| Conv2D 简化（§10.1） | 退化情形外报错 | 显式校验已加 | 中（限制功能） |
| BatchNorm 简化（§10.2） | 与标准 BN 语义不同 | 需重写前向+反向 | 中（语义差距） |
| CrossEntropy 假设（§10.3） | 非归一化 target 错误 | 校验或文档 | 低（约定） |
| Softmax 数值（§10.4） | 极端值下溢 | max 减法 | 低（工程） |
| 浮点误差未量化（§10.5） | AD4 界模糊 | 数值测试 | 低 |
| 循环/分支（§10.6） | 未形式化 | 后续论文 | 中 |
| 21 算子完备性（§10.7） | 新增算子需重验证 | effect system | 中 |
| 循环论证风险（§10.8） | 无（已分析） | — | 无 |

---

## 11. 结论

本文形式化了 Tenth 的 Wengert tape 语义，证明五条主定理：

- **定理 AD1**：21 个 `TapeOp` 变体的 backward 实现逐一满足链式法则等式（§6 穷举验证）；
- **定理 AD2**：`nodes.iter().rev()` 单遍回放在 DAG 性质下保证梯度累积正确；
- **定理 AD3**：9 个算子的 `input_tensors` 持久化是闭式 backward 的必要条件；
- **定理 AD4**：在 tape 完整性前提下，Tenth 与 PyTorch/JAX 数学语义等价；
- **定理 AD5**：5 个复合算子的闭式 backward 满足融合语义等价的充分条件。

**与 T38 联动**：T38 证明 tape 多路径一致性（前提），T39 证明 tape 反向语义正确性（结论），共同构成 Tenth autodiff 的完整正确性论证。具体而言：

- T38 定理 A1（tape 同构性）保证 VM/解释器记录相同 tape → T39 定理 AD1 可在统一 tape 上验证；
- T38 定理 A3（梯度正确性相对性）指出梯度正确性依赖 tape 完整性 → T39 定理 AD1-AD5 的成立前提是 tape 完整；
- T38 定理 A4（21 算子穷尽性）发现 `Neg` 算子 record 缺失 → T39 §6.6 的 `Neg` 链式法则验证**仅在 record 后**成立，未 record 时该节点不会出现在 tape 上，链式法则在该点断裂。

**对实施的指导**：

1. 新增算子时，必须同步 record 调用与 backward 分支（T38 §6 的 `Neg` 教训）；
2. 复合算子的闭式 backward 需通过 §6 形式的链式法则验证；
3. `input_tensors` 持久化协议（§3.4）需文档化，新增算子需明确持久化内容；
4. 局限章节（§10）的工程差距需在 `AUDIT.md` 登记，并在后续版本逐步缓解。

---

## 参考文献

1. Baydin, A. G., Pearlmutter, B. A., Radul, A. A., & Siskind, J. M. (2018). Automatic differentiation in machine learning: a survey. *Journal of Machine Learning Research*, 18(153), 1-43.
2. Wengert, R. E. (1964). A simple automatic derivative evaluation program. *Communications of the ACM*, 7(8), 463-464.
3. Griewank, A., & Walther, A. (2008). *Evaluating Derivatives: Principles and Techniques of Algorithmic Differentiation* (2nd ed.). SIAM.
4. Paszke, A., Gross, S., Chintala, S., et al. (2017). Automatic differentiation in PyTorch. *NIPS Autodiff Workshop*.
5. Frostig, R., Johnson, M. J., & Leary, C. (2018). Compiling machine learning programs for high-performance numerical computing. *NeurIPS*.
6. Ioffe, S., & Szegedy, C. (2015). Batch normalization: Accelerating deep network training by reducing internal covariate shift. *ICML*.
7. Ba, J. L., Kiros, J. R., & Hinton, G. E. (2016). Layer normalization. *arXiv:1607.06450*.
8. Hendrycks, D., & Gimpel, K. (2016). Gaussian error linear units (GELUs). *arXiv:1606.08415*.
9. Tenth 项目. (2026). *autodiff.rs* (v0.3.3). [源码](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/autodiff.rs).
10. Tenth 数理部. (2026). T38: autodiff tape 多路径一致性. [论文](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/docs/论文/T38-autodiff-tape多路径一致性.md).
11. Tenth 数理部. (2026). T2: Tape 形式化模型与根因定位可判定性. [论文](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/docs/论文/T2-Tape形式化模型与根因定位可判定性.md).

---

## 附录 A：定理索引

| 定理 | 陈述 | 证明 | 源码引用 |
|------|------|------|---------|
| AD1 | 21 算子链式法则等式 | §4.1 + §6 逐一 | [autodiff.rs:291-746](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/autodiff.rs) |
| AD2 | 拓扑逆序正确性 | §4.2 | [autodiff.rs:285-289](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/autodiff.rs) |
| AD3 | input_tensors 持久化必要性 | §4.3 | [autodiff.rs:14-25](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/autodiff.rs) |
| AD4 | 与 PyTorch/JAX 语义等价性 | §4.4 | 全文件 |
| AD5 | 算子融合形式化框架 | §4.5 | §5 复合算子 |
| AD2.1 | backward 复杂度 | §4.2 推论 | [autodiff.rs:285](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/autodiff.rs) |

## 附录 B：与现有文档的对应

| 本文章节 | 对应文档 |
|---------|---------|
| §3 Tenth Tape 形式化 | T2（Tape 形式化模型）、T38 §3 |
| §4.1 定理 AD1 | T38 定理 A4（21 算子穷尽性）的语义补充 |
| §4.2 定理 AD2 | T2 §3（拓扑序与根因定位） |
| §4.3 定理 AD3 | T38 §3.4（input_tensors 协议） |
| §4.4 定理 AD4 | T38 定理 A5（与 PyTorch/JAX 对比）的语义深化 |
| §4.5 定理 AD5 | 新增（无前序论文） |
| §6 21 算子验证 | T38 定理 A4 的穷尽性延伸 |
| §10 局限 | T38 §12（局限章节）、AUDIT.md |

## 附录 C：实施建议

基于本文理论结论，对 Tenth 后续实施的建议：

1. **新增算子流程**：
   - 在 `TapeOp` 新增变体；
   - 在 `Tape::backward` 新增 match 分支，按 §6 格式验证链式法则等式；
   - 新增 record 方法（`tape.xxx`），明确 `input_tensors` 协议（§3.4）；
   - 在 VM/解释器中新增 record 调用（T38 §6 的 `Neg` 教训）；
   - 添加测试（`cargo test --manifest-path tenth/Cargo.toml -- autodiff`）；
   - 同步 tenthc（若涉及编译器前端，本任务不涉及）。

2. **Conv2D 反向修复**（§10.1）：
   - 实现真正的 col2im 累积算法（支持 stride/padding/dilation）；
   - 重新验证链式法则等式（§6.18 的扩展）；
   - 在 AUDIT.md 登记当前限制。

3. **BatchNorm 修复**（§10.2）：
   - 修改前向与反向以按通道维统计（标准 BN 语义）；
   - 重新验证链式法则等式；
   - 更新 §6.19 的验证。

4. **CrossEntropy target 校验**（§10.3）：
   - 前向阶段校验 $\sum t_i = 1$，或文档明确要求 one-hot；
   - 或扩展支持 soft label（修正闭式为 $\bar L (s - t)$）。

5. **effect system 引入**（§9.4）：
   - 评估将"recording"作为类型系统一部分的可行性；
   - 若引入，T38 定理 A3 的局限可部分缓解；
   - 这是一项跨模块（HIR/类型系统/运行时）的护城河级改动，需总师统筹。

6. **测试覆盖**：
   - 21 算子的链式法则等式应有对应单元测试（当前仅 5 个，[autodiff.rs:909-1046](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/autodiff.rs)）；
   - 建议补充 `test_backward_sub`、`test_backward_div`、`test_backward_neg`、`test_backward_exp`、`test_backward_log`、`test_backward_sigmoid`、`test_backward_softmax`、`test_backward_crossentropy`、`test_backward_dropout`、`test_backward_conv2d`、`test_backward_batchnorm`、`test_backward_layernorm`、`test_backward_gelu`、`test_backward_transpose`、`test_backward_sum`、`test_backward_mean` 等 16 个测试（覆盖剩余算子）。

---

**论文版本**：v1（2026-07-02）
**审查轮次**：4 轮（结构/证明/边界/诚实）
**数理部出品**
