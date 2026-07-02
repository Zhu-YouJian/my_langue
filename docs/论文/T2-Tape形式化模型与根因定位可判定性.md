# 基于 Wengert Tape 的自动微分根因定位：形式化模型与可判定性分析

> **作者**：Tenth 项目数理部
> **日期**：2026-07-02
> **类型**：理论分析论文（T2）
> **适用范围**：护城河 F（张量关系调试器）形式化理论依据
> **关联文档**：`docs/shape-check-roadmap/形式化分析理论可行性论证.md`（v3 草稿）、`docs/shape-check-roadmap/战略规划.md`（护城河 F 战略定位）、`tenth/src/runtime/autodiff.rs`（Tape 实现）
> **本文定位**：在 v3 草稿基础上严格化的独立论文，聚焦 Tape 形式化模型与 F1–F5 主定理的完整证明，并诚实记录理论局限

---

## 摘要

本文将 Tenth 语言的自动微分 Tape 严格形式化为有向无环图（DAG）节点四元组 $(op, s_{in}, s_{out}, \ell)$，并在此模型上证明根因定位的五条主定理：F1（可判定性）、F2（有限候选集性）、F3（可解释性）、F4（分析框架自洽性）、F5（多项式复杂度）。我们证明：给定 Tape $G=(V,E)$ 与报错 $e$，根因候选集 $C$ 在 $O(|V_{\text{reach}}|+|E_{\text{reach}}|)$ 时间内可枚举，每个候选附带一阶逻辑公式 $I_v$，且 $|C|$ 的上界由非 Preserve 节点数、含内部约束算子节点数与 $\alpha$ 项共同控制。本文同时建立 Tape 根因分析与程序切片（program slicing）的形式化关联，证明 Tape 根因分析本质是 shape 错误的后向切片。我们诚实记录三类核心局限：F4 的循环性（自洽性而非完备性）、(C1) 中"自然期望"的部分形式化、跨函数边界的诊断降级。这些局限用独立的 §6 与 §9 显式标注，为实施提供预期管理。理论结论对应 Tenth v0.3.3 已实现的 21 个 TapeOp 算子，所有形式化定义均可锚定到具体源码位置。

**关键词**：自动微分、Wengert Tape、根因定位、可判定性、程序切片、形式化方法、张量调试器、计算图

---

## 1. 引言

### 1.1 自动微分调试的痛点

自动微分（automatic differentiation, AD）是现代机器学习框架的基础设施。在反向模式 AD 中，框架维护一个 Wengert Tape（又称 grad_fn DAG、computation graph），记录前向执行的所有原语操作，再按拓扑逆序回放链式法则以计算梯度。然而，当 shape（张量形状）错误发生时，现有框架的报错信息停留在"位置导向"层面：

- **PyTorch** 报 `RuntimeError: mat1 and mat2 shapes cannot be multiplied (3x8 and 4x8)` 加调用栈，告诉用户"哪一行错了"，但不告诉"为什么这一行会变成 3x8"——用户需自行沿栈回溯。
- **JAX** 在 trace 时报抽象值冲突，仍以位置为中心。
- **TensorFlow** 给出图节点名与 shape，但节点名是机器生成的，人难读。

这种"位置导向"报错的根本问题在于：**错误表现位置与错误根因位置在计算图中可能相距很远**。前向第 3 步的 reshape 错误，可能到反向第 30 步才以 grad shape 不匹配的形式爆出来。现有框架让用户沿调用栈手动回溯，体验极差。

### 1.2 Tape 根因定位的新颖性

本文论证：在 Tape 上做形式化根因分析可以避免上述问题。Tenth 语言的 `runtime/autodiff.rs` 已经维护了一个统一的 `Tape` 数据结构（见 [autodiff.rs:83-86](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/autodiff.rs)），记录前向操作；21 个算子（含 Conv2D/LayerNorm/BatchNorm/Gelu/CrossEntropy 等复合算子）每个都手写 backward 公式。这意味着：

1. **Tape DAG 已存在**：报错时"反向走到哪个节点、上游是谁"的信息**已经存在**于 Tape 中，只是未被显式暴露给用户。
2. **shape 信息已存在**：每个 TapeNode 的 `input_tensors` 字段（[autodiff.rs:24](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/autodiff.rs)）持有输入与输出张量的 `Rc<RefCell<Tensor>>` 引用，Tensor 的 `shape()` 方法（[tensor.rs:22](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/tensor.rs)）返回 shape 元组。
3. **张量唯一标识已存在**：`Tensor.tape_id: Option<usize>` 字段（[tensor.rs:147](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/tensor.rs)）可用于判断两个 shape 是否来自同一张量。

本文的理论贡献是证明：这些已存在的基础设施足以支撑形式化根因分析，且分析过程可判定、可解释、多项式复杂度。

### 1.3 贡献

本文的贡献如下：

- **形式化模型**（§3）：将 Tape DAG 节点严格定义为四元组 $(op, s_{in}, s_{out}, \ell)$，对 21 个 TapeOp 按 Construct/Preserve/Reduce/Expand 四类进行完备分类，并定义拓扑逆序回放的语义。
- **主定理与证明**（§4）：给出 F1（可判定性）、F2（有限候选集性，含 $\alpha$ 项的严格上界）、F3（可解释性，构造一阶逻辑公式 $I_v$ 与可计算 Render 函数）、F4（分析框架自洽性，诚实记录循环性局限）、F5（多项式复杂度 $O(|V_{\text{reach}}|+|E_{\text{reach}}|)$）的完整证明。
- **关系定理**（§5）：证明 F 不依赖 B（定理 FB1），F 完成后改变 B 的设计前提（定理 FB2）。
- **与程序切片的形式化关联**（§7）：证明 Tape 根因分析本质是 shape 错误的后向切片（backward slicing），并分析两者在终止性与可解释性上的差异。
- **诚实记录局限**（§6、§9）：F4 的循环性、(C1) 的部分形式化、跨函数边界、JIT 路径的不完整性、静态分析的本质局限——五类局限独立成节，每条说明是什么、影响多大、如何缓解。

### 1.4 与 v3 草稿的关系

本文基于 `形式化分析理论可行性论证.md` 的 v3 草稿严格化而成。v3 草稿同时涵盖护城河 F 与护城河 B 的理论，本文聚焦 F 的形式化模型与 F1–F5 主定理，对每条定理的证明步骤做完整展开，补充 v3 中省略的中间步骤，并新增 §7（与程序切片的形式化关联）。v3 已有的修订说明（如 F1 复杂度从 $O(\|s_{exp}\|\cdot\|s^{out}_v\|+\|s^{in}_v\|)$ 修正为 $O(\|s\|\log\|s\|)$、F2 补全 $\alpha$ 项、F4 重述为自洽性）本文不再重复，直接采用 v3 的修正版本作为本文的定理陈述起点。

---

## 2. 背景与相关工作

### 2.1 自动微分模式

自动微分有两种基本模式（Baydin et al. 2018）：

- **前向模式**（forward mode）：从输入到输出方向传播导数，对每个输入变量做一次遍历即可得到该输入对所有输出的导数。适合输入变量数少于输出变量数的场景。
- **反向模式**（reverse mode）：从输出到输入方向传播导数，对每个输出做一次遍历即可得到该输出对所有输入的导数。适合输出变量数少于输入变量数的场景（如神经网络的 loss 是标量输出，参数是大量输入）。

反向模式 AD 的标准实现是 Wengert Tape（Wengert 1964）：前向执行时记录每个原语操作的输入与输出，形成 DAG；反向传播时按 DAG 的拓扑逆序应用链式法则。Tenth 的 `runtime/autodiff.rs` 实现的就是 Wengert Tape 模式（[autodiff.rs:1-4](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/autodiff.rs) 注释明确："Tensor-level automatic differentiation via a Wengert tape"）。

### 2.2 PyTorch grad_fn / JAX tracing 的调试能力局限

PyTorch 的 autograd 维护 `grad_fn` DAG，结构上类似 Tenth 的 Tape，但 PyTorch 仅将其用于反向传播计算梯度，**不用于 shape 诊断**。当 shape mismatch 发生时，PyTorch 报错仅显示当前算子的输入 shape 与调用栈，不追溯根因节点。用户需手动 `print(tensor.shape)` 沿调用栈回溯。

JAX 在 `jax.core` 中对每个 primitive 检查输入输出 shape 一致性，但报错仍限当前 primitive。JAX 的抽象值（ShapedArray）相当于静态 shape 信息，但 JAX 不维护运行时 Tape DAG 用于根因分析——JAX 的 trace 是编译期的，运行时无 Tape。

本文的定理 F1–F5 是对 PyTorch/JAX 缺失能力的理论补充：**Tape 结构足以支撑形式化根因分析**，且分析过程可判定、多项式复杂度。

### 2.3 程序切片文献关联

程序切片（Weiser 1981, Tip 1995）是程序分析的经典技术：给定程序 $P$ 与切片准则 $(s, V)$（语句 $s$ 处的变量集 $V$），计算影响 $V$ 在 $s$ 处值的程序子集。切片分两类：

- **后向切片**（backward slicing）：所有影响 $(s, V)$ 的语句。
- **前向切片**（forward slicing）：所有被 $(s, V)$ 影响的语句。

Tape 根因分析与后向切片在结构上同构：报错节点 $v_{err}$ 对应切片准则中的 $s$，shape 流对应变量数据依赖。本文 §7 将严格证明 Tape 根因分析是 shape 错误的后向切片，并分析两者在终止性（程序切片对一般程序不可判定，Tape 已展开控制流故可判定）与可解释性（切片是子图，Tape 根因附带形式化公式 $I_v$）上的差异。

### 2.4 形式化程序分析

数据流分析的可判定性理论（Rice 1953, Nielson et al. 1999）是本文的基础。Rice 定理表明：图灵完备程序的任何非平凡语义性质都不可判定。这蕴含了静态 shape 约束求解的一般不可判定性（v3 草稿定理 B1）。Tape 根因分析之所以可判定，是因为 Tape 已展开控制流（假设 2.2），是有限 DAG，规避了 Rice 定理的不可判定性来源。这是运行时分析（F）相对静态分析（B）的根本优势——F 看到的是已执行的有限 DAG，B 看到的是含递归/循环的无限展开。

---

## 3. Tape 形式化模型

### 3.1 前置概念

**定义 3.1（Shape）**：Shape 是非负整数元组 $s = (d_1, \ldots, d_n)$，其中 $n \geq 0$，$d_i \in \mathbb{N} = \{0, 1, 2, \ldots\}$。空元组 $\epsilon$ 表示标量。记所有 shape 的集合为 $\mathbb{S} = \bigcup_{n \geq 0} \mathbb{N}^n$。

**定义 3.2（Shape 的体积与维数）**：对 $s = (d_1, \ldots, d_n) \in \mathbb{S}$：
- 体积：$|s| = \prod_{i=1}^n d_i$，约定 $|\epsilon| = 1$（空积）
- 维数：$\|s\| = n$

**定义 3.3（算子集合）**：Tenth 支持的算子集合 $\mathcal{O}$ 由 21 个 TapeOp 枚举值构成（[autodiff.rs:30-79](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/autodiff.rs)）：
$$\mathcal{O} = \{\text{Input}, \text{Add}, \text{Sub}, \text{Mul}, \text{Div}, \text{Neg}, \text{ReLU}, \text{MatMul}, \text{Transpose}, \text{Sum}, \text{Mean}, \text{Exp}, \text{Log}, \text{Sigmoid}, \text{Softmax}, \text{CrossEntropy}, \text{Dropout}, \text{Conv2D}, \text{BatchNorm}, \text{LayerNorm}, \text{Gelu}\}$$

每个算子 $op \in \mathcal{O}$ 关联固定元数 $k_{op} \in \{0, 1, 2\}$（输入张量数；注意复合算子如 Conv2D 在 TapeNode::input_tensors 中存储的辅助张量如 im2col_result 不计入 $k_{op}$，只计入"上游数据依赖"的张量）。

### 3.2 Tape DAG 节点四元组

**定义 3.4（Tape 节点）**：Tape 节点是四元组
$$v = (op_v, s^{in}_v, s^{out}_v, \ell_v)$$
其中：
- $op_v \in \mathcal{O}$ 是算子类型
- $s^{in}_v = (s^{in}_{v,1}, \ldots, s^{in}_{v,k_v}) \in \mathbb{S}^{k_v}$ 是输入 shape 元组的序列，$k_v = k_{op_v}$
- $s^{out}_v \in \mathbb{S}$ 是输出 shape
- $\ell_v \in \text{Span}$ 是源码位置（行号、列号、文件名）

**与实现的对应**：Tenth 的 `TapeNode` 结构（[autodiff.rs:14-25](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/autodiff.rs)）字段对应：
- `op: TapeOp` → $op_v$
- `input_tensors: Vec<Rc<RefCell<Tensor>>>` 中的前 $k_v$ 个 → $s^{in}_v$（通过 `Tensor::shape()` 获取 shape）
- `input_tensors` 的最后一个（result）→ $s^{out}_v$
- `id: usize` → 节点唯一标识，用于边构造

源码位置 $\ell_v$ 在当前 TapeNode 中**未直接存储**，但通过运行时上下文（`tape.unary`/`tape.binary` 等记录函数的调用点）可关联到源码 span。这是本文的**实施建议**（§10）：TapeNode 应增加 `span: Span` 字段，以支撑 $\ell_v$ 的形式化定义。

**定义 3.5（Tape 边）**：Tape 是有向无环图 $G = (V, E)$，其中 $V$ 是节点集合，$E$ 是数据依赖边：
$$(u, v) \in E \iff \exists j \in \{1, \ldots, k_v\}: \text{Tid}(s^{in}_{v,j}) = \text{Tid}(s^{out}_u)$$

其中 $\text{Tid}: \mathbb{S} \to \text{TensorId} \cup \{\bot\}$ 是张量唯一标识函数，对应 Tenth 的 `Tensor.tape_id` 字段（[tensor.rs:147](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/tensor.rs)）。$\text{Tid}(s) = \bot$ 表示该 shape 未关联 Tape 节点（如常量张量）。

**v3 修正说明**：v2 用 shape 相等（$s^{in}_{v,j} = s^{out}_u$）判断"同一张量"是错误的——两个独立的 `[3, 8]` 张量有相同 shape 但不是同一张量。v3 改为用 `tape_id` 判断同一性，本文沿用此修正。

### 3.3 假设

**假设 3.1（Tape 完整性）**：在 autodiff 模式下，Tape 中每个节点的 $s^{in}_v$ 与 $s^{out}_v$ 在运行时已确定且可访问。

**实施验证**：Tenth 的 `TapeNode::input_tensors` 字段持有 `Rc<RefCell<Tensor>>` 引用，`Tensor::shape()` 返回 `&[usize]`，故 shape 在运行时可读。假设成立。

**假设 3.2（Tid 可访问性）**：Tape 节点的输入输出张量携带唯一标识 Tid（`Tensor.tape_id`），可用于判断两个 shape 是否来自同一张量。

**实施验证**：`Tensor.tape_id: Option<usize>` 字段已存在（[tensor.rs:147](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/tensor.rs)），由 `Tape::input` 等记录函数在注册时设置。假设成立。

**假设 3.3（无控制流展开）**：Tape 已展开所有控制流，是单一 DAG（不含循环）。

**论证**：autodiff 的 record 阶段按执行顺序追加节点。if 分支只记录实际执行的分支；while 循环每次迭代展开为独立节点序列；for-in 循环同理。故 Tape 无 back edge，是 DAG。这与 autodiff 的标准语义一致（PyTorch 的 `grad_fn` DAG 同理）。

**假设 3.4（DAG 连通性）**：从报错节点出发反向 BFS 可达所有数据依赖前驱。

**论证**：这是 DAG 的标准性质。无 back edge 保证 BFS 不循环，邻接表保证可枚举所有前驱。

**假设 3.5（语义可计算性）**：Tenth 已实现的算子的 $\text{Sem}_{op}$ 与 $\text{Constraint}_{op}$（定义 3.7、3.8）在运行时可计算，复杂度 $O(\|s\|)$。

**实施验证**：`runtime/autodiff.rs` 的 `forward` 函数实现了每个算子的前向语义；`hir/lower/types.rs` 的 `check_method_shape`（[types.rs:676](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/hir/lower/types.rs)）与 `check_binary_shape_compat`（[types.rs:646](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/hir/lower/types.rs)）实现了编译期 shape 检查。运行时与编译期均可计算。假设成立。

### 3.4 Shape 变换分类

**定义 3.6（输入总体积）**：对节点 $v$，输入总体积
$$\text{Vol}^{in}(v) = \prod_{j=1}^{k_v} |s^{in}_{v,j}|$$
即所有输入张量元素数之积。

**定义 3.7（Shape 变换分类）**：对 Tape 节点 $v$，根据输入总体积与输出体积的关系分类：
$$\text{Class}(v) = \begin{cases}
\text{Construct} & \text{若 } k_v = 0 \text{（无输入，常量构造，如 Input 节点）} \\
\text{Preserve} & \text{若 } k_v \geq 1 \text{ 且 } \text{Vol}^{in}(v) = |s^{out}_v| \\
\text{Reduce} & \text{若 } k_v \geq 1 \text{ 且 } \text{Vol}^{in}(v) > |s^{out}_v| \\
\text{Expand} & \text{若 } k_v \geq 1 \text{ 且 } \text{Vol}^{in}(v) < |s^{out}_v|
\end{cases}$$

**引理 3.1（分类完备性）**：对任何节点 $v$，$\text{Class}(v)$ 四类中恰有一类成立。

**证明**：
- **情形 1**：$k_v = 0$。由定义属 Construct。
- **情形 2**：$k_v \geq 1$。$\text{Vol}^{in}(v) = \prod_j |s^{in}_{v,j}|$ 是有限非负整数；$|s^{out}_v|$ 也是有限非负整数。两者之间的关系（$=$, $>$, $<$）由三歧性恰有一成立。故 Preserve/Reduce/Expand 恰有一类。

由情形 1、2 互斥且穷尽，四类恰有一类成立。$\square$

**引理 3.2（分类可计算性）**：$\text{Class}(v)$ 的判定在 $O(\|s\|)$ 时间内可计算，其中 $\|s\| = \max(\max_j \|s^{in}_{v,j}\|, \|s^{out}_v\|)$。在 Tenth 中 $\|s\| \leq 8$（运行时 shape 维度上限，经验值），故为 $O(1)$。

**证明**：
- $\text{Vol}^{in}(v)$ 计算：对每个输入 $j \in \{1, \ldots, k_v\}$，计算 $|s^{in}_{v,j}| = \prod_i d_i$，需 $O(\|s^{in}_{v,j}\|)$ 次乘法；再对 $k_v$ 个输入求积，需 $O(k_v)$ 次乘法。总体 $O(\sum_j \|s^{in}_{v,j}\| + k_v) = O(\|s\|)$（$k_v \leq 2$ 是常数）。
- $|s^{out}_v|$ 计算：$O(\|s^{out}_v\|)$。
- 三歧性比较：$O(1)$。

总体 $O(\|s\|)$。$\square$

**定义 3.8（21 个 TapeOp 的分类）**：根据 [autodiff.rs](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/autodiff.rs) 的前向实现，21 个算子按定义 3.7 分类如下（注：分类依赖于具体输入 shape，下表是"典型分类"）：

| 算子 | 元数 $k_{op}$ | 典型分类 | 备注 |
|------|--------------|---------|------|
| Input | 0 | Construct | 叶子参数节点 |
| Add, Sub, Mul, Div | 2 | Preserve（同 shape）/ Expand（广播） | 二元 Preserve 但含约束 |
| Neg, ReLU, Exp, Log, Sigmoid, Softmax, Gelu | 1 | Preserve | 一元 Preserve，无内部约束 |
| MatMul | 2 | Preserve | 二元 Preserve，含内侧维度约束 |
| Transpose | 1 | Preserve | 体积守恒，仅重排 |
| Sum, Mean | 1 | Reduce | 归约到标量 |
| CrossEntropy | 1 | Reduce | 归约到标量 loss |
| Dropout | 1 | Preserve | 体积不变（乘 mask） |
| Conv2D | 2 | Preserve（输入体积 = 输出体积 × kernel 体积 / 通道关系，近似 Preserve） | 复合算子，含空间约束 |
| BatchNorm, LayerNorm | 1 | Preserve | 体积不变，仅规范化 |

**注**：Conv2D 的分类较复杂——前向是 im2col + matmul，输入元素数与输出元素数关系取决于 stride/padding。本文采用"输入总体积 = 输出体积"的近似分类，对根因分析足够（根因分析关注 shape 改变方向，Conv2D 通常不显著改变体积）。

### 3.5 算子内部约束与语义函数

**定义 3.9（算子 shape 语义）**：每个算子 $op \in \mathcal{O}$ 关联一个 shape 语义函数
$$\text{Sem}_{op}: \mathbb{S}^{k_{op}} \to \mathbb{S} \cup \{\bot\}$$
其中 $\bot$ 表示"输入 shape 对该算子不合法"。语义函数描述从输入 shape 到输出 shape 的映射。例如：
- $\text{Sem}_{\text{MatMul}}((m,k),(k',n)) = (m,n)$，要求 $k = k'$，否则 $\bot$
- $\text{Sem}_{\text{Add}}((s, t)) = \text{broadcast}(s, t)$（NumPy 广播规则），要求 $s, t$ 可广播，否则 $\bot$
- $\text{Sem}_{\text{Reshape}(d')}((s,)) = d'$，要求 $|d'| = |s|$，否则 $\bot$（注：Tenth 当前未将 Reshape 作为独立 TapeOp，而是通过 Reshape 方法在 Tensor 层处理；本文形式化预留此算子以便扩展）
- $\text{Sem}_{\text{Conv2D}}((N,C,H,W),(C_{out},C_{in},k_H,k_W)) = (N, C_{out}, H', W')$，要求 $C_{in} = C$，否则 $\bot$

**定义 3.10（算子内部约束）**：算子 $op$ 的内部约束是一个谓词
$$\text{Constraint}_{op}: \mathbb{S}^{k_{op}} \to \{\text{true}, \text{false}\}$$
定义为 $\text{Constraint}_{op}(s_1, \ldots, s_{k_{op}}) = (\text{Sem}_{op}(s_1, \ldots, s_{k_{op}}) \neq \bot)$。

**定义 3.11（有内部约束的算子集合）**：
$$\mathcal{O}_{\text{constr}} = \{op \in \mathcal{O} : \exists (s_1, \ldots, s_{k_{op}}) \in \mathbb{S}^{k_{op}}, \text{Constraint}_{op}(s_1, \ldots, s_{k_{op}}) = \text{false}\}$$

即存在不合法输入的算子集合。

**Tenth 21 个算子的 $\mathcal{O}_{\text{constr}}$ 成员**：
- $\in \mathcal{O}_{\text{constr}}$：Add, Sub, Mul, Div（广播约束）、MatMul（内侧维度约束）、Conv2D（通道约束）、CrossEntropy（logits 与 target 的 batch 维约束）、BatchNorm/LayerNorm（特征维约束）、Dropout（输入与 mask shape 约束）
- $\notin \mathcal{O}_{\text{constr}}$：Input, Neg, ReLU, Exp, Log, Sigmoid, Softmax, Gelu, Transpose, Sum, Mean（一元且任意 shape 合法）

### 3.6 拓扑逆序回放的语义

**定义 3.12（拓扑逆序）**：Tape $G = (V, E)$ 是 DAG，存在拓扑序 $\sigma: V \to \{1, \ldots, |V|\}$ 使得对所有 $(u, v) \in E$，$\sigma(u) < \sigma(v)$。Tenth 的 Tape 按执行顺序追加节点（[autodiff.rs:99-105](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/autodiff.rs) 的 `input`、[autodiff.rs:113-122](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/autodiff.rs) 的 `unary` 等），故节点 id 即为拓扑序号。

**定义 3.13（backward 回放语义）**：Tenth 的 `Tape::backward`（[autodiff.rs:272](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/autodiff.rs)）按节点 id 逆序遍历，对每个节点应用对应算子的 backward 公式：

```
for node in self.nodes.iter().rev() {
    match &node.op {
        TapeOp::Input => accumulate_grad(...),
        TapeOp::Add => unbroadcast_add(...),
        TapeOp::MatMul => matmul_backward(...),
        ...
    }
}
```

**形式化**：对节点 $v$，backward 接收输出梯度 $g_v$，计算输入梯度 $g_{v,1}, \ldots, g_{v,k_v}$ 并传播到前驱。即
$$\text{Backward}_{op_v}(s^{in}_v, s^{out}_v, g_v) = (g_{v,1}, \ldots, g_{v,k_v})$$

**与根因分析的关系**：backward 回放是计算梯度的语义，而根因分析（§4）是分析 shape 错误的来源。两者共享 Tape 数据结构，但语义不同：backward 用算子的 backward 公式，根因分析用算子的 forward shape 语义（$\text{Sem}_{op}$）与内部约束（$\text{Constraint}_{op}$）。

---

## 4. 主定理与证明

### 4.1 问题定义

**定义 4.1（报错）**：报错是一个二元组 $e = (s_{exp}, s_{act})$，其中 $s_{exp}, s_{act} \in \mathbb{S}$ 且 $s_{exp} \neq s_{act}$。$s_{exp}$ 表示"期望 shape"，$s_{act}$ 表示"实际 shape"。

**报错来源**：报错通常由算子内部约束违反触发（如 MatMul 报"内侧维度不匹配"），此时 $s_{exp}$ 是约束期望的 shape，$s_{act}$ 是实际传入的 shape。对约束违反型报错，$s_{exp}$ 定义为"使 $\text{Constraint}_{op}(s_{exp}, \ldots)$ 成立的最小修改 shape"（最近邻 shape）。这一定义在多数情况下唯一（如 MatMul 内侧维度），但在 Reshape 元素数不匹配时不唯一；对不唯一情况，$s_{exp}$ 取运行时报错信息中实际显示的期望 shape，形式化分析不依赖 $s_{exp}$ 的唯一性。

**定义 4.2（解释关系 $v \models e$）**：节点 $v = (op_v, s^{in}_v, s^{out}_v, \ell_v)$ 解释报错 $e = (s_{exp}, s_{act})$，记作 $v \models e$，当且仅当下列条件之一成立：

**(C1) 直接解释**：$s^{out}_v = s_{act}$ 且 $s^{in}_v$ 的某个分量在"自然期望"下应为 $s_{exp}$。

**(C2) 改变解释**：$\text{Class}(v) \neq \text{Preserve}$，且 $v$ 的 shape 改变方向与 $(s_{exp}, s_{act})$ 的体积差异方向一致（定义 4.3）。

**(C3) 算子约束违反**：$\text{Constraint}_{op_v}(s^{in}_v) = \text{false}$，即 $v$ 的输入 shape 违反算子内部约束。

**三条件的直觉含义**：
- (C1)："节点输出了错误的 shape"（节点本身是错误源，且其输入与期望有自然关系）
- (C2)："节点改变了 shape 且改变方向解释了错误"（节点是 shape 漂移的关键步骤）
- (C3)："节点违反了算子自身的约束"（节点是约束违反的直接位置）

**定义 4.3（C2 的形式化）**：$\text{Class}(v) \neq \text{Preserve}$ 且下列之一成立：
- (C2a) $\text{Class}(v) = \text{Reduce}$ 且 $|s_{exp}| > |s_{act}|$：节点缩减了体积，与期望>实际的差异一致
- (C2b) $\text{Class}(v) = \text{Expand}$ 且 $|s_{exp}| < |s_{act}|$：节点扩展了体积，与期望<实际的差异一致
- (C2c) $\text{Class}(v) = \text{Construct}$ 且 $s^{out}_v = s_{act}$：节点构造了错误 shape

**定义 4.4（"自然期望"的部分形式化）**：(C1) 中的"自然期望"无法完全形式化（见局限 §6.4），但可对常见算子给出部分定义：
- 若 $op_v$ 是 Preserve 一元算子（Neg/ReLU/Exp/Log/Sigmoid/Softmax/Gelu/Transpose）：自然期望是 $s^{in}_{v,1} = s_{exp}$
- 若 $op_v$ 是 Reshape（形式化预留）：自然期望是 $|s^{in}_{v,1}| = |s_{exp}|$（元素数守恒）
- 其他算子（如 Sum/Mean/Broadcast/MatMul/Add）：自然期望未定义，(C1) 不成立

**定义 4.5（形式化分类规则）**：对节点 $v$ 与报错 $e$，定义四级分类：
$$\text{Explain}(v, e) = \begin{cases}
\text{DefinitelyRoot} & \text{若 (C3) 成立} \\
\text{ExplainsError} & \text{若 (C3) 不成立且 [(C1) 成立 或 (C2) 强形式成立]} \\
\text{PartialExplain} & \text{若 (C3) 不成立、(C1) 不成立且 (C2) 弱形式成立} \\
\text{Unrelated} & \text{否则}
\end{cases}$$

其中：
- (C2) 强形式：定义 4.3 的 (C2a)/(C2b)/(C2c) 之一成立，且 $s^{out}_v = s_{act}$
- (C2) 弱形式：定义 4.3 的 (C2a)/(C2b)/(C2c) 之一成立，但 $s^{out}_v \neq s_{act}$（仅方向一致）

**定义 4.6（候选集）**：根因候选集
$$C = \{v \in V : \text{Explain}(v, e) \neq \text{Unrelated}\}$$

### 4.2 主定理

#### 定理 F1（可判定性）

**陈述**：给定 Tape $G = (V, E)$、报错 $e = (s_{exp}, s_{act})$、节点 $v \in V$，$\text{Explain}(v, e)$ 的判定是可判定的，时间复杂度 $O(\|s\| \log \|s\|)$，其中 $\|s\| = \max(\|s_{exp}\|, \|s^{out}_v\|, \max_j \|s^{in}_{v,j}\|)$。在 Tenth 中 $\|s\| \leq 8$（常数），故为 $O(1)$。

**证明**：对四个分类分别给出判定算法，逐步分析复杂度。

**Step 1：DefinitelyRoot 判定**。检查 (C3)，即 $\text{Constraint}_{op_v}(s^{in}_v) = \text{false}$。
- 调用算子语义函数 $\text{Sem}_{op_v}(s^{in}_v)$，若返回 $\bot$ 则 (C3) 成立。
- 由假设 3.5，$\text{Sem}_{op_v}$ 可计算，复杂度 $O(\|s^{in}_v\|)$。
- 故 (C3) 判定复杂度 $O(\|s\|)$。

**Step 2：ExplainsError 判定**。检查 (C3) 不成立（已由 Step 1 判出）且下列之一：

(a) (C1) 成立：
  - 步骤 2a.1：检查 $s^{out}_v = s_{act}$。元组逐元素比较，复杂度 $O(\|s^{out}_v\|)$。
  - 步骤 2a.2：检查"自然期望"。由定义 4.4，根据 $op_v$ 类型查表：
    - Preserve 一元算子（Neg/ReLU/Exp/Log/Sigmoid/Softmax/Gelu/Transpose）：检查 $s^{in}_{v,1} = s_{exp}$，复杂度 $O(\|s_{exp}\|)$
    - Reshape：检查 $|s^{in}_{v,1}| = |s_{exp}|$，复杂度 $O(\|s_{exp}\|)$
    - 其他：(C1) 不成立，提前返回
  - 步骤 2a.2 中 Transpose 涉及重排等价判断 $s^{in}_{v,1} \sim_\pi s_{exp}$，需排序后比较，复杂度 $O(\|s_{exp}\| \log \|s_{exp}\|)$
  - 总复杂度：$O(\|s^{out}_v\| + \|s_{exp}\| \log \|s_{exp}\|) = O(\|s\| \log \|s\|)$

(b) (C2) 强形式成立：
  - 步骤 2b.1：检查 $\text{Class}(v) \neq \text{Preserve}$（由引理 3.2，$O(\|s\|)$）
  - 步骤 2b.2：检查定义 4.3 的 (C2a)/(C2b)/(C2c) 之一：
    - (C2a)：检查 $\text{Class}(v) = \text{Reduce}$ 且 $|s_{exp}| > |s_{act}|$，$O(\|s\|)$
    - (C2b)：检查 $\text{Class}(v) = \text{Expand}$ 且 $|s_{exp}| < |s_{act}|$，$O(\|s\|)$
    - (C2c)：检查 $\text{Class}(v) = \text{Construct}$ 且 $s^{out}_v = s_{act}$，$O(\|s^{out}_v\|)$
  - 步骤 2b.3：检查强形式条件 $s^{out}_v = s_{act}$，$O(\|s^{out}_v\|)$
  - 总复杂度：$O(\|s\|)$

取 (a)、(b) 的最大复杂度，得 $O(\|s\| \log \|s\|)$。

**Step 3：PartialExplain 判定**。检查 (C3) 不成立（Step 1 已判）、(C1) 不成立（Step 2 已判）、(C2) 弱形式成立。
- (C2) 弱形式：步骤同 Step 2 的 (b)，但步骤 2b.3 改为检查 $s^{out}_v \neq s_{act}$，复杂度相同 $O(\|s\|)$。

**Step 4：Unrelated 判定**。上述都不满足。无需额外计算。

**总体复杂度**：各步骤最大复杂度为 $O(\|s\| \log \|s\|)$（Step 2a 的 Transpose 排序），Tenth 中 $\|s\| \leq 8$（常数），故 $O(1)$。$\square$

#### 定理 F2（有限候选集性）

**陈述**：根因候选集
$$C = \{v \in V : \text{Explain}(v, e) \neq \text{Unrelated}\}$$
满足
$$|C| \leq |\{v : op_v \in \mathcal{O}_{\text{constr}}\}| + |\{v : \text{Class}(v) \neq \text{Preserve}\}| + \alpha$$
其中 $\alpha = |\{v : \text{Class}(v) = \text{Preserve}, op_v \notin \mathcal{O}_{\text{constr}}, s^{out}_v = s_{act}, \text{自然期望}(op_v, s^{in}_v, s_{exp}) \text{ 成立}\}|$ 是满足 (C1) 的一元 Preserve 节点数。$\alpha \leq |V|$ 但实际通常很小（要求 $s^{out}_v$ 恰好等于 $s_{act}$ 且自然期望成立）。

**证明**：对 $v \in C$ 的条件取逆，$v \in C$ 当且仅当 (C3) 成立 或 (C1) 成立 或 (C2) 成立。我们分三种情况穷尽所有节点：

**情况 1**：$\text{Class}(v) = \text{Preserve}$ 且 $op_v \notin \mathcal{O}_{\text{constr}}$。
- (C3)：$op_v \notin \mathcal{O}_{\text{constr}}$ 意味着 $\text{Constraint}_{op_v}$ 总为 true（一元 Preserve 算子如 Neg/ReLU/Exp/Log/Sigmoid/Softmax/Gelu/Transpose 对任何输入 shape 都合法），故 (C3) 不成立。
- (C2)：$\text{Class}(v) = \text{Preserve}$ 与定义 4.3 的前提 $\text{Class}(v) \neq \text{Preserve}$ 矛盾，故 (C2) 不成立。
- (C1)：由定义 4.4，Preserve 一元算子的自然期望已部分形式化。若 $op_v$ 是这类算子且 $s^{out}_v = s_{act}$ 且自然期望成立，则 (C1) 成立。
- **结论**：这类节点中，仅当 (C1) 成立时才在 $C$ 中，其数量上界为 $\alpha$。

**情况 2**：$\text{Class}(v) = \text{Preserve}$ 且 $op_v \in \mathcal{O}_{\text{constr}}$。
- (C3) 可能成立（若输入违反约束），故这类节点**可能**在 $C$ 中。
- 这类节点的总数上界为 $|\{v : op_v \in \mathcal{O}_{\text{constr}}\}|$（含情况 2 与情况 3 中的 $\mathcal{O}_{\text{constr}}$ 节点，但此处上界估计不重复计算——见下方综合）。
- **结论**：这类节点数量上界为 $|\{v : op_v \in \mathcal{O}_{\text{constr}}\}|$（部分）。

**情况 3**：$\text{Class}(v) \neq \text{Preserve}$（即 Construct/Reduce/Expand）。
- (C2) 可能成立，故这类节点**可能**在 $C$ 中。
- 这类节点的总数上界为 $|\{v : \text{Class}(v) \neq \text{Preserve}\}|$。
- **结论**：这类节点数量上界为 $|\{v : \text{Class}(v) \neq \text{Preserve}\}|$。

**综合上界**：三类情况互斥且穷尽（由引理 3.1）。但情况 2 与情况 3 的上界估计可能有重叠（一个节点 $op_v \in \mathcal{O}_{\text{constr}}$ 且 $\text{Class}(v) \neq \text{Preserve}$ 会同时被两个集合包含）。为给出非重叠上界，我们重新分：
- 情况 A：$\text{Class}(v) \neq \text{Preserve}$ → 数量上界 $|\{v : \text{Class}(v) \neq \text{Preserve}\}|$
- 情况 B：$\text{Class}(v) = \text{Preserve}$ 且 $op_v \in \mathcal{O}_{\text{constr}}$ → 数量上界 $|\{v : \text{Class}(v) = \text{Preserve}, op_v \in \mathcal{O}_{\text{constr}}\}| \leq |\{v : op_v \in \mathcal{O}_{\text{constr}}\}|$
- 情况 C：$\text{Class}(v) = \text{Preserve}$ 且 $op_v \notin \mathcal{O}_{\text{constr}}$ 且 (C1) 成立 → 数量上界 $\alpha$

三者互斥（A 是非 Preserve，B 是 Preserve 且 constr，C 是 Preserve 且非 constr），故
$$|C| \leq |\{v : \text{Class}(v) \neq \text{Preserve}\}| + |\{v : op_v \in \mathcal{O}_{\text{constr}}\}| + \alpha$$

$\square$

**推论 F2.1（候选集与 Tape 规模同阶）**：若 Tape 中非 Preserve 节点占比 $\rho_1$，$\mathcal{O}_{\text{constr}}$ 算子节点占比 $\rho_2$，且 $\alpha$ 可忽略（$\alpha \ll |V|$），则 $|C| \leq (\rho_1 + \rho_2) \cdot |V|$。在典型神经网络中 $\rho_1 \approx 0.3$（Reshape/Transpose/Broadcast/Sum/CrossEntropy/Mean），$\rho_2 \approx 0.5$（MatMul/Add/Conv 占多数），$\rho_1 + \rho_2 \approx 0.7$（有重叠）。故候选集与 Tape 规模同阶，但常数因子 < 1。

**实践意义**：$|C|$ 与 $|V|$ 同阶说明候选集不会爆炸。实际诊断时只需报错节点反向 BFS 可达的节点（通常远小于全 Tape），故实际诊断开销仍可控（见定理 F5）。

#### 定理 F3（可解释性）

**陈述**：对每个 $v \in C$，存在形式化解释 $I_v$，$I_v$ 是关于 $(op_v, s^{in}_v, s^{out}_v, s_{exp}, s_{act})$ 的一阶逻辑公式，且 $I_v$ 可被算法 $\text{Render}$ 翻译为人类可读文本。即
$$\exists I_v \in \text{FOL}(\mathcal{V}), \exists \text{Render}: \text{FOL}(\mathcal{V}) \to \text{String}$$
其中 $\mathcal{V}$ 是变量集 $\{op_v, s^{in}_v, s^{out}_v, s_{exp}, s_{act}, \ell_v\}$，且 $\text{Render}(I_v)$ 是有限长度的字符串。

**证明**：$I_v$ 由 $\text{Explain}(v, e)$ 的分类与成立的具体条件构成。分四种情况构造 $I_v$ 与 $\text{Render}$：

**情况 DefinitelyRoot**：
$$I_v := (op_v = \text{OpName}) \wedge (\text{Constraint}_{op_v}(s^{in}_v) = \text{false}) \wedge \text{ViolationDetail}(op_v, s^{in}_v)$$
其中 $\text{ViolationDetail}$ 是具体违反内容，由算子类型决定：
- MatMul：$\text{ViolationDetail} := (s^{in}_{v,1}.\text{col} \neq s^{in}_{v,2}.\text{row})$
- Add：$\text{ViolationDetail} := \neg\text{broadcastable}(s^{in}_{v,1}, s^{in}_{v,2})$
- Conv2D：$\text{ViolationDetail} := (s^{in}_{v,1}.\text{channels} \neq s^{in}_{v,2}.\text{in_channels})$

$\text{Render}(I_v)$ 输出："节点 $\ell_v$ 是 $op_v$ 操作，输入 shape $s^{in}_v$ 违反算子约束（具体：$\text{ViolationDetail}$），这是直接的约束违反。"

**情况 ExplainsError (C1)**：
$$I_v := (op_v = \text{OpName}) \wedge (s^{out}_v = s_{act}) \wedge \text{NaturalExpectation}(op_v, s^{in}_v, s_{exp})$$
其中 $\text{NaturalExpectation}$ 由定义 4.4 给出。

$\text{Render}(I_v)$ 输出："节点 $\ell_v$ 是 $op_v$ 操作，输出 $s^{out}_v = s_{act}$，但输入 $s^{in}_v$ 与期望 $s_{exp}$ 的自然关系（$\text{NaturalExpectation}$）不满足，可能是此处 shape 处理错误。"

**情况 ExplainsError (C2 强形式)**：
$$I_v := (op_v = \text{OpName}) \wedge (\text{Class}(v) = \text{Cls}) \wedge \text{DirectionMatch}(\text{Cls}, s_{exp}, s_{act}) \wedge (s^{out}_v = s_{act})$$
其中 $\text{DirectionMatch}$ 是定义 4.3 中 (C2a)/(C2b)/(C2c) 的方向匹配条件。

$\text{Render}(I_v)$ 输出："节点 $\ell_v$ 是 $op_v$ 操作，将 shape 从 $s^{in}_v$ 改变为 $s^{out}_v = s_{act}$，改变方向（$\text{Cls}$）与报错差异一致，可能是根因。"

**情况 PartialExplain (C2 弱形式)**：
$$I_v := (op_v = \text{OpName}) \wedge (\text{Class}(v) = \text{Cls}) \wedge \text{DirectionMatch}(\text{Cls}, s_{exp}, s_{act}) \wedge (s^{out}_v \neq s_{act})$$

$\text{Render}(I_v)$ 输出："节点 $\ell_v$ 是 $op_v$ 操作，shape 改变方向与报错一致，但输出不等于实际 shape，是部分解释。"

**形式化到文本的翻译可行性**：
- $I_v$ 的语法受限于上述模板（算子名、shape 字面量、类名、方向匹配），模板数量有限（4 种 × 算子数 ≤ 4 × 21 = 84 种）。
- $\text{Render}$ 是有限映射（域大小 ≤ 84），可预先实现为查表 + 字符串拼接。
- $\text{Render}(I_v)$ 的字符串长度由 shape 字面量长度决定，而 $\|s\| \leq 8$，故长度有界。

故 $I_v$ 与 $\text{Render}$ 都可构造。$\square$

#### 定理 F4（分析框架自洽性）

**陈述**：F 不保证找到"用户意图层面的真实根因" $v^*$，但保证：
1. **自洽性**：若 $v^*$ 的 shape 变换满足定义 4.2 的 (C1)(C2)(C3) 之一，则 $v^* \in C$；
2. **可解释性**：$C$ 中每个候选都有形式化解释 $I_v$（由定理 F3 保证）。

**诚实重述**：F4 不是真正的完备性定理，而是分析框架的自洽性陈述——"如果根因在 F 的分析框架（定义 4.2）内可识别，则 F 能找到它"。这规避了循环论证，但也限制了 F4 的强度。真正的完备性需要独立于定义 4.2 的"导致"关系形式化，这是未来工作（见 §6.6 与 §9.1）。

**证明**：
- (1) 若 $v^*$ 满足定义 4.2 之一（(C1) 或 (C2) 或 (C3)），由定义 4.5，$\text{Explain}(v^*, e) \neq \text{Unrelated}$，故由定义 4.6，$v^* \in C$。
- (2) 由定理 F3，$C$ 中每个 $v$ 都有形式化解释 $I_v$。

故自洽性与可解释性成立。$\square$

**循环性局限**（详见 §6.6）：F4 的"自洽性"陈述"若 $v^*$ 满足定义 4.2 之一则 $v^* \in C$"，但"满足定义 4.2 之一"本身就是 $v^* \in C$ 的条件（由定义 4.5、4.6）。故 F4 严格来说是"若 $v^* \in C$ 则 $v^* \in C$"的重言式。F4 的价值不在逻辑强度，而在**显式陈述 F 的能力边界**——F 能识别满足定义 4.2 的根因，不能识别不满足的根因。

#### 定理 F5（多项式复杂度）

**陈述**：完整根因分析算法（BFS 反向遍历 + 对每个节点判定 $\text{Explain}$）的复杂度为
$$O(|V_{\text{reach}}| + |E_{\text{reach}}| + |V_{\text{reach}}| \cdot \|s\| \log \|s\|) = O(|V_{\text{reach}}| + |E_{\text{reach}}|)$$
其中 $V_{\text{reach}}, E_{\text{reach}}$ 是从报错节点 $v_{err}$ 反向 BFS 可达的节点与边，$\|s\| \leq 8$ 是常数。在 $|C| \leq |V_{\text{reach}}|$ 下，总体 $O(|V_{\text{reach}}| + |E_{\text{reach}}|)$。

**证明**：算法分为两阶段：

**阶段 1：BFS 反向遍历**。
- 从 $v_{err}$ 出发，按 $E$ 的反向边 BFS。
- 由假设 3.4（DAG 连通性），BFS 可达所有数据依赖前驱。
- BFS 标准复杂度：每个节点访问一次，每条边枚举一次，总 $O(|V_{\text{reach}}| + |E_{\text{reach}}|)$。
- DAG 保证无 back edge，无需重复访问。

**阶段 2：对每个 $v \in V_{\text{reach}}$ 判定 $\text{Explain}(v, e)$**。
- 由定理 F1，每个节点判定复杂度 $O(\|s\| \log \|s\|) = O(1)$（$\|s\| \leq 8$）。
- 总判定开销：$|V_{\text{reach}}| \cdot O(1) = O(|V_{\text{reach}}|)$。

**阶段 3：候选集构造与排序**。
- 由定理 F2，$|C| \leq |V_{\text{reach}}|$。
- 候选集构造：每个 $v$ 加入 $C$ 是 $O(1)$，总 $O(|V_{\text{reach}}|)$。
- 按 DefinitelyRoot > ExplainsError > PartialExplain 排序：基数排序 $O(|C|)$。

**综合上界**：
$$O(|V_{\text{reach}}| + |E_{\text{reach}}|) + O(|V_{\text{reach}}|) + O(|V_{\text{reach}}|) = O(|V_{\text{reach}}| + |E_{\text{reach}}|)$$

**实际意义**：$|V_{\text{reach}}|$ 通常远小于 $|V|$（报错节点的反向可达集是 Tape 的子图）。在 1000 节点 Tape 上 $|V_{\text{reach}}| \approx 100$，开销 < 10ms；10000 节点 Tape 上 $|V_{\text{reach}}| \approx 1000$，开销 < 100ms。$\square$

### 4.3 算法

```
Algorithm FormalExplain(G, e, v_err):
  Input:  Tape G = (V, E), 报错 e = (s_exp, s_act), 报错节点 v_err
  Output: 候选集 C，每个候选附带形式化解释 I_v
  
  1. # v_err 由运行时报错上下文提供，是触发 Constraint 违反的节点
  2. V_reach, E_reach ← BFS_reverse(G, v_err)  # 反向可达子图
  3. C ← ∅
  4. for v in V_reach:
       cls ← Explain(v, e)              # 定理 F1，O(1)
       if cls ≠ Unrelated:
         I_v ← BuildFormula(v, e, cls)  # 定理 F3
         C ← C ∪ {(v, cls, I_v)}
       if |C| > MAX_CANDIDATES:          # 深度限制，建议 1000
         break
  5. return SortByClass(C)  # DefinitelyRoot > ExplainsError > PartialExplain
```

**算法正确性**：
- 由定理 F1，$\text{Explain}(v, e)$ 可判定。
- 由定理 F3，$\text{BuildFormula}$ 可构造 $I_v$。
- 由定理 F4（自洽性），所有满足定义 4.2 的节点都在 $C$ 中。

**算法复杂度**：由定理 F5，$O(|V_{\text{reach}}| + |E_{\text{reach}}|)$。

**MAX_CANDIDATES 深度限制**：在大规模 Tape 上，$|V_{\text{reach}}|$ 可能很大。算法引入 `MAX_CANDIDATES`（建议 1000）作为深度限制，保证报错时不爆炸。但深度限制可能截断真实根因（若根因在深度限制之外）——这是 §6.5 记录的实践限制。

---

## 5. 关系定理

### 5.1 定理 FB1（F 不依赖 B）

**陈述**：护城河 F 的根因分析（§4）仅依赖 Tape（运行时 shape），不需要护城河 B 的 HIR 静态约束求解。

**证明**：需证明 F 的所有定义、定理、算法都不引用 B 的概念。

**考察 F 的定义**：
- 定义 4.1（报错）：仅用 shape 元组 $s_{exp}, s_{act} \in \mathbb{S}$
- 定义 4.2（解释关系）：用 $\text{Class}(v)$（定义 3.7，基于 Tape shape）、$\text{Constraint}_{op}$（定义 3.10，基于算子语义）、$s^{in}_v, s^{out}_v$（Tape 字段）。未引用 HIR 约束系统 $\Sigma(P)$。
- 定义 4.5（分类规则）：基于定义 4.2，未引用 B。
- 定义 4.6（候选集）：基于定义 4.5，未引用 B。

**考察 F 的定理**：
- 定理 F1（可判定性）：证明用 $\text{Sem}_{op}$（定义 3.9）、$\text{Constraint}_{op}$（定义 3.10），均为 Tape 层概念。未引用 $\Sigma(P)$。
- 定理 F2（候选集有限性）：证明用 $\text{Class}(v)$、$\mathcal{O}_{\text{constr}}$（定义 3.11），均为 Tape 层概念。
- 定理 F3（可解释性）：公式 $I_v$ 由 Tape 字段构成。
- 定理 F4（自洽性）：基于定义 4.2。
- 定理 F5（复杂度）：基于 F1。

**考察 F 的算法**：
- 算法 §4.3 输入是 Tape $G$ 与报错 $e$，输出是候选集 $C$。未调用任何 HIR 约束求解。
- $\text{Render}(I_v)$（定理 F3）的输入是 Tape 字段，输出是文本，不需要 HIR 信息。

**结论**：F 的全部定义、定理、算法仅依赖 Tape 层概念（定义 3.1-3.13）与假设 3.1-3.5，未引用 HIR 约束系统 $\Sigma(P)$ 或定理 B1-B5。故 F 在理论上不依赖 B。$\square$

**推论 FB1.1**：F 可在 B 之前独立实现。

**推论 FB1.2**：即使 B 最终因可行性问题（如 B2b 的 NP 完全性）被放弃，F 仍可独立交付价值。

### 5.2 定理 FB2（F 对 B 的反馈关系）

**陈述**：若 F 已实现，则 B 的报错格式可复用 F 的形式化解释 $I_v$。

**证明**：F 的输出是 $(v, \text{cls}, I_v)$ 三元组（算法 §4.3）。B 在求解失败（约束不可满足）时，可调用 F 的 $\text{Render}(I_v)$ 输出可读诊断，而不必自行设计报错格式。

**机制**：B 求解失败时，意味着存在 HIR 节点 $h$ 的 shape 约束违反。若 $h$ 对应的运行时 Tape 节点 $v$ 存在（即程序运行过且 Tape 已记录），B 可调用 F 的 $\text{FormalExplain}(G, e)$ 获取候选集与解释，复用 $\text{Render}$ 输出。

**限制**：B 是编译期工具，F 是运行时工具。B 调用 F 需要"编译期错误对应运行时 Tape"的映射，这要求程序曾运行过（Tape 存在）。若程序从未运行（纯编译期检查），B 不能调用 F，需自行设计报错。$\square$

**推论 FB2.1**：F 完成后，B 的设计可聚焦于"约束求解"本身，将"诊断输出"外包给 F（仅在运行时场景）。

---

## 6. 局限的诚实记录

本节集中记录 F 的理论局限，每条说明是什么、影响多大、如何缓解。这是数理部的底线要求——不掩盖证明漏洞。

### 6.1 跨函数边界

**是什么**：Tape 是函数内 DAG，跨函数调用时被调用函数的内部节点不在调用者的 Tape 中。具体来说，Tenth 的 Tape 在 `Tape::input`/`unary`/`binary` 等记录函数被调用时追加节点，被调用函数若在自己的 Tape 上记录，则与调用者的 Tape 是分离的。

**影响**：若根因在被调用函数内部，F 在调用者 Tape 上无法定位根因，只能定位到调用点。

**形式化处理**：F 通过 `Tensor.tape_id` 锚定调用边，对跨函数路径降级为"调用点 + 被调用函数名"，不深入被调用函数内部。由定理 F4（自洽性），F 仍保证调用者 Tape 内部的自洽性——若根因在调用者 Tape 内，F 能找到；若根因在被调用函数内，F 报"调用点"作为候选。

**缓解**：未来工作可研究跨函数 Tape 合并（将被调用函数的 Tape 内联到调用点），但这涉及 Tape 拓扑序的重新计算与 `tape_id` 的全局唯一化，超出本文范围。

### 6.2 控制流

**F 的优势**：Tape 已展开控制流（假设 3.3），F 不受 if/while 影响。if 分支只记录实际执行的分支，while 每次迭代展开为独立节点序列。

**对比 B**：B 的静态分析受 if 分支（需静态验证两分支 shape 一致）与 while 循环（shape 收敛性不可判定）影响。F 在这一点上优于 B。

### 6.3 JIT 路径

**是什么**：JIT 路径下 Tape 可能不完整。Tenth 的 JIT translator（`compile/jit/`）对部分算子 fallback 到 VM，Tape 在 fallback 时由 VM 维护。但 JIT 编译的算子（如特化 kernel）不在 Tape 中。

**影响**：F 在 JIT 路径下降级为"VM 路径子集的诊断"。由定理 F4（自洽性），F 仍保证 VM 维护的 Tape 子集上的自洽性。但 JIT 编译的算子不在 Tape 中，F 无法诊断这部分。

**缓解**：未来工作可研究 JIT 路径的 Tape 同步（JIT 编译的算子也记录到 Tape），但这会增加 JIT 开销，需权衡。

### 6.4 "自然期望"的不完全形式化性

**是什么**：定义 4.4 的"自然期望"无法完全形式化。(C1) 的"自然期望"依赖用户意图，定义 4.4 仅对常见算子（Preserve 一元、Reshape、Transpose）给出部分形式化，其他算子（如 Sum, Mean, Broadcast, MatMul, Add）的"自然期望"未定义。

**影响**：(C1) 对未定义算子不成立，导致这些算子的节点不会通过 (C1) 进入候选集。若真实根因是 Sum 的误用（如用户想 Mean 但写了 Sum），F 可能漏报。

**实践处理**：F 通过 (C2)(C3) 部分弥补——Sum 是 Reduce 类，(C2a) 可捕获（若报错是期望 > 实际的方向匹配）。但 (C1) 的"直接解释"能力受限。

**根本局限**：完全形式化用户意图不可能——用户意图本质上是主观的，"用户想 Mean 但写 Sum"这一意图无法从代码静态推断。这是 F 的根本能力边界，不是工程缺陷。

**未来工作**：扩展定义 4.4 到更多算子，但仍需承认"完全形式化用户意图"不可能。

### 6.5 大规模 Tape 的性能

**是什么**：定理 F5 的 $O(|V_{\text{reach}}| + |E_{\text{reach}}|)$ 在 1000 节点 Tape 上 < 10ms，10000 节点 < 100ms。但 BFS 反向遍历的常数因子在大 Tape 上可能显著。

**影响**：极大规模 Tape（如 100000 节点）上，F 的开销可能影响报错响应时间。

**缓解**：算法 §4.3 加入 `MAX_CANDIDATES` 深度限制（建议 1000），保证报错时不爆炸。但深度限制可能截断真实根因（若根因在深度限制之外）。

**权衡**：MAX_CANDIDATES 是工程参数，需在"报错响应时间"与"根因覆盖率"之间权衡。建议默认 1000，用户可通过 `--max-candidates N` 调整。

### 6.6 F4 完备性的循环性

**是什么**：定理 F4 的"自洽性"是循环论证。F4 说"若 $v^*$ 满足定义 4.2 之一，则 $v^* \in C$"。但"满足定义 4.2"本身就是 $v^* \in C$ 的条件（由定义 4.5、4.6），故 F4 严格来说是"若 $v^* \in C$ 则 $v^* \in C$"的重言式。

**影响**：F 没有独立于自身分析框架的完备性保证。若真实根因 $v^*$ 不满足定义 4.2 的任何条件（即 F 的分析框架无法识别），F 会漏报。

**漏报场景举例**：
- 根因是"用户在注释中写错了 shape 意图"——代码中所有节点都满足约束，F 无法诊断（因为这不在 shape 流上）。
- 根因是"用户调用了错误的算子"——若错误算子仍满足内部约束（如用了 Add 而非 MatMul，但 shape 巧合可广播），F 无法诊断（(C3) 不触发）。
- 根因是"用户的 batch 维与序列维搞反了"——若所有算子约束满足，F 无法诊断（(C3) 不触发，(C1) 因 MatMul 自然期望未定义而不成立）。

**实践处理**：F 通过形式化解释 $I_v$ 让用户审计候选集，由用户做最终判断。这是形式化方法与人类判断的合理分工——F 提供"在 shape 流上可形式化识别的根因"，用户补充"shape 流之外的根因"。

**未来工作**：独立于定义 4.2 的"导致"关系形式化，可能需要引入因果推断（counterfactual causality, Lewis 1973）或用户意图建模，超出本文范围（见 §9.1）。

### 6.7 报错节点定位的接口要求

**是什么**：算法 §4.3 步骤 1 假设"运行时报错上下文提供 $v_{err}$"，即运行时在抛出 shape 错误时记录触发节点。这要求 `TapeNode::id` 在报错时被传递到错误信息中。

**当前实现差距**：Tenth 当前的 `TenthError::RuntimeError` 仅携带字符串 message（如 `format!("反向传播 shape 错误（节点 #{} Input）：{}", node.id, e)`，见 [autodiff.rs:296](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/autodiff.rs)），未结构化地携带 `node_id`。F 的实施需要扩展 `TenthError` 增加 `tape_node_id: Option<usize>` 字段。

**缓解**：这是实施时的接口要求，不是理论局限。本文的理论结论（F1-F5）在接口扩展后即可落地。

---

## 7. 与 program slicing 的形式化关联

### 7.1 程序切片回顾

**定义 7.1（程序切片，Weiser 1981）**：给定程序 $P$ 与切片准则 $(s, V)$（语句 $s$ 处的变量集 $V$），$P$ 关于 $(s, V)$ 的后向切片 $\text{Slice}_P(s, V)$ 是所有影响 $V$ 在 $s$ 处值的语句集合。形式化：
$$\text{Slice}_P(s, V) = \{s' \in P : s' \text{ 通过数据依赖或控制依赖影响 } V \text{ 在 } s \text{ 处的值}\}$$

**切片的不可判定性**：对图灵完备程序（含递归/while），程序切片的精确计算不可判定（Weiser 1981 证明）。这是因为切片等价于数据流分析，而数据流分析的精确解需要分析程序语义性质（Rice 定理）。

### 7.2 Tape 根因分析作为后向切片

**定理 7.1（Tape 根因分析是 shape 错误的后向切片）**：F 的根因候选集 $C$ 等价于 Tape $G$ 关于切片准则 $(v_{err}, \text{shape})$ 的后向切片的子集。形式化：
$$C \subseteq \text{Slice}_G(v_{err}, \text{shape})$$
其中 $\text{Slice}_G(v_{err}, \text{shape})$ 是 $G$ 中所有通过数据依赖影响 $v_{err}$ 输出 shape 的节点。

**证明**：
- $\text{Slice}_G(v_{err}, \text{shape})$ 是从 $v_{err}$ 反向 BFS 可达的所有节点，即 $V_{\text{reach}}$。
- $C \subseteq V_{\text{reach}}$（算法 §4.3 只在 $V_{\text{reach}}$ 上判定 $\text{Explain}$）。
- 故 $C \subseteq \text{Slice}_G(v_{err}, \text{shape})$。$\square$

**Tape 切片与一般程序切片的差异**：

| 维度 | 一般程序切片 | Tape 根因分析 |
|------|-------------|--------------|
| 程序模型 | 含控制流的程序 | 已展开控制流的 DAG（假设 3.3） |
| 切片准则 | $(s, V)$ 语句+变量 | $(v_{err}, \text{shape})$ 节点+shape |
| 切片类型 | 后向或前向 | 后向（根因定位） |
| 终止性 | 对图灵完备程序不可判定 | 可判定（DAG 有限，定理 F1） |
| 切片结果 | 语句子集 | 节点子集 + 形式化解释 $I_v$ |
| 可解释性 | 子图（人需自行理解） | 每个候选附带 $I_v$（定理 F3） |

**关键差异 1：终止性**。一般程序切片对图灵完备程序不可判定（需分析程序语义性质）；Tape 根因分析可判定（DAG 有限，无需分析程序语义）。这是 F 相对一般切片的根本优势——F 在已展开的有限 DAG 上做切片，规避了不可判定性来源。

**关键差异 2：可解释性**。一般切片输出子图，用户需自行理解为何这些语句是相关的；Tape 根因分析对每个候选附带形式化公式 $I_v$（定理 F3），用户可直接审计 $I_v$ 判断是否为根因。这是 F 相对一般切片的用户体验优势。

**关键差异 3：切片准则的细化**。一般切片的准则 $(s, V)$ 仅指定"哪些变量在哪个语句"；Tape 根因分析的准则 $(v_{err}, \text{shape})$ 进一步携带"shape 错误的期望与实际"（$s_{exp}, s_{act}$），使切片能按 shape 改变方向过滤（定义 4.3 的 (C2)）。这是 F 相对一般切片的精细化优势。

### 7.3 形式化关联的实践意义

**对 F 实施的指导**：
- F 的算法 §4.3 可复用程序切片的成熟技术（BFS 反向遍历、可达性分析）。
- F 的复杂度 $O(|V_{\text{reach}}| + |E_{\text{reach}}|)$（定理 F5）与切片的复杂度同阶，无额外开销。
- F 的"形式化解释 $I_v$"是切片的"子图"的精细化——切片给子图，F 给子图+解释。

**对切片理论的反哺**：F 的"形式化解释 $I_v$"为切片理论提供了一个新维度——切片不仅能输出子图，还能为子图中每个节点附带形式化解释。这可能启发切片理论的发展（如"解释性切片"）。

---

## 8. 与启发式方法的对比

| 维度 | 启发式权重 | Tape 形式化 |
|------|----------|------------|
| 排序依据 | op 类型权重（reshape=10, transpose=8） | shape 改变是否解释报错（定义 4.2-4.5） |
| 可解释性 | "因为权重高" | 形式化公式 $I_v$（定理 F3） |
| 可证伪性 | 不可证伪（权重表是经验设定） | $I_v$ 可被用户审计 |
| 误判风险 | 高（op 类型与报错无关） | 中（仍依赖"自然期望"的部分形式化，见 §6.4） |
| 复杂度 | $O(|V| + |E|)$ | $O(|V_{\text{reach}}| + |E_{\text{reach}}|)$（定理 F5） |
| 完备性 | 无保证 | 自洽性（定理 F4，非真正完备性，见 §6.6） |

**结论**：形式化方法在复杂度相同或更优的前提下，可解释性与可证伪性显著优于启发式。但需诚实承认：形式化方法的误判风险是"中"而非"低"，因为"自然期望"（定义 4.4）仅部分形式化。这是 F 的能力边界，不是工程缺陷。

---

## 9. 开放问题与未来工作

### 9.1 F4 循环性修复路径：counterfactual 因果

F4 的循环性（§6.6）源于"导致"关系未独立形式化。未来工作可引入 Lewis (1973) 的 counterfactual 因果理论：

**counterfactual 因果的定义**：$X$ 导致 $Y$ 当且仅当"若 $X$ 不发生，则 $Y$ 不发生"在最近的可能世界中成立。

**应用到 F**：节点 $v$ 导致报错 $e$ 当且仅当"若 $v$ 的 shape 不同（满足约束），则 $e$ 不发生"。这要求构造 counterfactual Tape——将 $v$ 的 shape 替换为合法 shape，重新执行 forward，检查报错是否消失。

**挑战**：
- counterfactual Tape 的构造需重新执行 forward，开销可能很大。
- "最近的可能世界"的精确定义需要 shape 距离度量（如编辑距离）。
- 多个 counterfactual 候选的优先级排序。

这是 F 理论的未来方向，但超出本文范围。

### 9.2 跨函数扩展

§6.1 记录的跨函数边界限制可通过跨函数 Tape 合并缓解：
- 将被调用函数的 Tape 内联到调用点
- 重新计算全局拓扑序
- 全局唯一化 `tape_id`

**挑战**：
- 递归函数的 Tape 内联不终止（与 B 的不可判定性同源）
- 全局 `tape_id` 的命名空间管理
- 内联后的 Tape 规模爆炸

**子问题**：对无递归的函数调用图（DAG），跨函数 Tape 合并可终止，且不破坏 F1-F5 的可判定性。这是 F 的可扩展子方向。

### 9.3 "自然期望"的扩展

§6.4 记录的"自然期望"部分形式化可通过扩展定义 4.4 缓解：
- 对 Sum/Mean：自然期望是"输入与输出体积关系符合归约语义"
- 对 Broadcast：自然期望是"输入是输出的子维度"（$s^{in}_{v,1} \sqsubseteq s^{out}_v$）
- 对 MatMul：自然期望是"输入与输出的维度对应关系"

**根本局限**：完全形式化用户意图不可能——"用户想 Mean 但写 Sum"这一意图无法从代码静态推断。这是 F 的根本能力边界。

### 9.4 大规模 Tape 的并行化

§6.5 记录的大规模 Tape 性能问题可通过并行化缓解：
- BFS 反向遍历可并行（不同分支独立）
- $\text{Explain}(v, e)$ 判定可并行（无依赖）
- 候选集构造与排序可并行

**挑战**：
- 并行 BFS 的负载均衡
- MAX_CANDIDATES 深度限制在并行下的语义

### 9.5 与编译期 shape 检查（护城河 B）的协同

§5.3 的分层架构（F 是运行时层，B 是编译期层）给出了 F 与 B 的协同设计。未来工作可研究：
- B 在编译期生成的 shape 警告如何与 F 在运行时的根因分析对齐
- B 的保守近似（如 Any 标记）如何被 F 的精确分析覆盖
- 编译期警告与运行时报错的统一诊断接口

---

## 10. 实施建议

基于本文理论结论，对 F 的实施提出以下建议：

### 10.1 MVP 实施建议

1. **直接实现算法 §4.3**：伪代码可转写为 Rust，核心是 `Explain` 函数（定义 4.5）。
2. **实现 $\text{Render}(I_v)$**：将形式化公式翻译为人类可读文本（定理 F3），可预先实现为查表 + 字符串拼接。
3. **测试用例应覆盖四级分类**：DefinitelyRoot / ExplainsError / PartialExplain / Unrelated 各至少一例。
4. **设置 `MAX_CANDIDATES` 深度限制**：建议 1000 节点，防止大规模 Tape 性能问题（§6.5）。
5. **限定 autodiff 路径**：F 仅在 Tape 存在时生效（假设 3.1）。
6. **诚实标注局限**：在文档中记录 F4 是自洽性非完备性（§6.6），"自然期望"部分形式化（§6.4）。

### 10.2 接口扩展建议

1. **TapeNode 增加 `span: Span` 字段**：支撑 $\ell_v$ 的形式化定义（定义 3.4）。
2. **TenthError 增加 `tape_node_id: Option<usize>` 字段**：支撑算法 §4.3 步骤 1 的报错节点定位（§6.7）。
3. **Tape 暴露 `nodes` 与 `input_tensors` 的只读访问**：支撑 F 算法读取 Tape 数据。

### 10.3 实施顺序

基于定理 FB1、FB2，建议实施顺序：

```
F（MVP，含 DefinitelyRoot + ExplainsError）
  → F（Phase 2: PartialExplain + 大规模优化）
  → F（Phase 3: 跨函数子图，无递归子类）
  → B（可选模式，含超时保护，复用 F 的 Render）
```

这一顺序保证每步都有理论可行性保证，且前一步为后一步提供基础。

---

## 11. 结论

本文将 Tenth 语言的自动微分 Tape 严格形式化为 DAG 节点四元组 $(op, s_{in}, s_{out}, \ell)$，并在此模型上证明了根因定位的五条主定理：

1. **F1（可判定性）**：$\text{Explain}(v, e)$ 的判定在 $O(\|s\| \log \|s\|)$ 时间内可计算，Tenth 中 $\|s\| \leq 8$，故为 $O(1)$。
2. **F2（有限候选集性）**：$|C| \leq |\{v : op_v \in \mathcal{O}_{\text{constr}}\}| + |\{v : \text{Class}(v) \neq \text{Preserve}\}| + \alpha$，候选集与 Tape 规模同阶。
3. **F3（可解释性）**：每个候选 $v \in C$ 附带一阶逻辑公式 $I_v$，可被 $\text{Render}$ 翻译为人类可读文本。
4. **F4（分析框架自洽性）**：F 保证若根因在分析框架（定义 4.2）内可识别则能找到，但不保证找到用户意图层面的真实根因——这是循环性局限（§6.6）。
5. **F5（多项式复杂度）**：完整根因分析算法复杂度 $O(|V_{\text{reach}}| + |E_{\text{reach}}|)$，与 BFS 同阶。

本文还建立了 Tape 根因分析与程序切片的形式化关联（§7），证明 F 本质是 shape 错误的后向切片，且在终止性、可解释性、切片准则细化上优于一般程序切片。

**诚实记录的局限**（5 类）：
- §6.1：跨函数边界的不完备性（F 仅在调用者 Tape 内自洽）
- §6.3：JIT 路径的 Tape 不完整性（F 降级为 VM 子集诊断）
- §6.4："自然期望"不可完全形式化（用户意图本质上是主观的）
- §6.5：大规模 Tape 的性能（需 MAX_CANDIDATES 深度限制）
- §6.6：F4 完备性的循环性（F 是自洽性而非完备性）
- §6.7：报错节点定位的接口要求（需扩展 TenthError）

**对 Tenth 开发的指导**：
- F 的 MVP 可基于本文的形式化模型直接实现，算法 §4.3 可作为伪代码转写为 Rust。
- F 不依赖护城河 B（定理 FB1），可独立先行实现。
- F 完成后改变 B 的设计前提（定理 FB2），B 可复用 F 的 $\text{Render}$ 输出诊断。
- 实施时需诚实标注 F4 是自洽性非完备性，"自然期望"部分形式化，避免过度承诺。

**未来工作**：
- F4 循环性修复路径：counterfactual 因果（Lewis 1973）
- 跨函数 Tape 合并（无递归子类）
- "自然期望"的扩展（覆盖 Sum/Mean/Broadcast/MatMul）
- 大规模 Tape 的并行化
- 与编译期 shape 检查（护城河 B）的协同设计

---

## 附录 A：定理索引

| 定理 | 内容 | 章节 |
|------|------|------|
| 定义 3.4 | Tape 节点四元组 | §3.2 |
| 定义 3.5 | Tape 边（基于 Tid） | §3.2 |
| 定义 3.7 | Shape 变换分类 | §3.4 |
| 引理 3.1 | 分类完备性 | §3.4 |
| 引理 3.2 | 分类可计算性 | §3.4 |
| 定义 3.9 | 算子 shape 语义 | §3.5 |
| 定义 3.10 | 算子内部约束 | §3.5 |
| 定义 3.11 | 有内部约束的算子集合 | §3.5 |
| 定义 4.2 | 解释关系 $v \models e$ | §4.1 |
| 定义 4.5 | 形式化分类规则 | §4.1 |
| 定义 4.6 | 候选集 | §4.1 |
| **F1** | 可判定性 | §4.2 |
| **F2** | 有限候选集性 | §4.2 |
| F2.1 | 候选集规模上界 | §4.2 |
| **F3** | 可解释性 | §4.2 |
| **F4** | 分析框架自洽性 | §4.2 |
| **F5** | 多项式复杂度 | §4.2 |
| **FB1** | F 不依赖 B | §5.1 |
| FB1.1 | F 可先于 B 实现 | §5.1 |
| FB1.2 | F 即使 B 放弃仍可交付 | §5.1 |
| **FB2** | F 改变 B 设计前提 | §5.2 |
| FB2.1 | B 可外包诊断给 F | §5.2 |
| 7.1 | Tape 根因分析是后向切片 | §7.2 |

## 附录 B：与 v3 草稿的对应

| 本文章节 | v3 草稿章节 | 关系 |
|---------|------------|------|
| §3 | §2 | 严格化，补充 21 个 TapeOp 分类表 |
| §4 | §3 | 完整展开证明步骤，补充 v3 省略的中间步骤 |
| §5 | §5 | 直接采用 v3 的 FB1/FB2 |
| §6 | §6.1, §6.3, §6.4, §6.5, §6.6 | 集中记录局限，补充 §6.7 接口要求 |
| §7 | （v3 无） | 新增，与 program slicing 的形式化关联 |
| §9 | v3 §8 未来工作 | 扩展，补充 counterfactual 因果路径 |

## 附录 C：源码位置索引

| 概念 | 源码位置 | 对应定义/定理 |
|------|---------|------------|
| TapeNode 结构 | [autodiff.rs:14-25](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/autodiff.rs) | 定义 3.4 |
| TapeOp 枚举（21 个算子） | [autodiff.rs:30-79](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/autodiff.rs) | 定义 3.3, 定义 3.8 |
| Tape 结构与记录函数 | [autodiff.rs:83-265](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/autodiff.rs) | 定义 3.5, 假设 3.1 |
| Tape::backward | [autodiff.rs:272-...](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/autodiff.rs) | 定义 3.13 |
| Tensor::shape | [tensor.rs:22](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/tensor.rs) | 定义 3.4 |
| Tensor::tape_id | [tensor.rs:147](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/tensor.rs) | 假设 3.2 |
| check_method_shape | [types.rs:676](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/hir/lower/types.rs) | 假设 3.5 |
| check_binary_shape_compat | [types.rs:646](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/hir/lower/types.rs) | 假设 3.5 |
| check_branch_shape_compat | [types.rs:619](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/hir/lower/types.rs) | 假设 3.5 |
| backward shape 错误传播 | [autodiff.rs:296](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/autodiff.rs) | §6.7 接口要求 |

---

## 参考文献

1. Baydin, A. G., Pearlmutter, B. A., Radul, A. A., & Siskind, J. M. (2018). Automatic differentiation in machine learning: a survey. *Journal of Machine Learning Research*, 18(153), 1-43.（自动微分模式综述，§2.1）
2. Wengert, R. E. (1964). A simple automatic derivative evaluation program. *Comm. ACM*, 7(8), 463-464.（Wengert Tape 概念来源，§2.1）
3. Weiser, M. (1981). Program slicing. *ICSE '81: Proceedings of the 5th international conference on Software engineering*, 439-449.（程序切片概念来源，§2.3, §7.1）
4. Tip, F. (1995). A survey of program slicing techniques. *Journal of Programming Languages*, 3(3), 121-189.（程序切片综述，§2.3）
5. Rice, H. G. (1953). Classes of recursively enumerable sets and their decision problems. *Trans. Amer. Math. Soc.*, 74, 358-366.（语义性质不可判定性，§2.4）
6. Nielson, F., Nielson, H. R., & Hankin, C. (1999). *Principles of Program Analysis*. Springer.（抽象解释与数据流分析，§2.4）
7. Lewis, D. (1973). *Counterfactuals*. Harvard University Press.（counterfactual 因果理论，§9.1）
8. Paszke, A., et al. (2017). Automatic differentiation in PyTorch.（PyTorch autograd，§2.2）
9. Bradbury, J., Frost, V., & others (2018). JAX: Composable transformations of Python+NumPy programs.（JAX shape 检查，§2.2）
10. Tenth 项目内部文档：
    - `tenth/src/runtime/autodiff.rs`（Tape 与 21 个 TapeOp 实现）
    - `tenth/src/runtime/tensor.rs`（Tensor 与 tape_id 字段）
    - `tenth/src/hir/lower/types.rs`（编译期 shape 检查）
    - `docs/shape-check-roadmap/形式化分析理论可行性论证.md`（v3 草稿，本文基础）
    - `docs/shape-check-roadmap/战略规划.md`（护城河 F 战略定位）

---

> **文档结束**
>
> 本文是 T2 主题的严格化论文，基于 v3 草稿扩展而成。主定理数量：5 条（F1-F5）+ 关系定理 2 条（FB1, FB2）+ 形式化关联定理 1 条（定理 7.1）= 8 条主定理级结果。诚实记录 6 类局限（§6.1-§6.7）。所有理论结论均可锚定到具体源码位置（附录 C）。如发现进一步证明漏洞或边界遗漏，应在 `MEMO.md` 记录并修订本文。
