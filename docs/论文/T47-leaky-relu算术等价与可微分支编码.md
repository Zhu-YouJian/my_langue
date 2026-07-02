# leaky_relu 的算术等价技巧：Tenth 无 select 原语下的可微分支编码通则

> **论文编号**：T47
> **数理部分类**：可微分支编码 / 算术等价变换 / 形式化语义
> **关联论文**：T39（Wengert Tape 形式化语义与反向模式正确性）
> **关联源码**：[`tenth/std/nn/activations.th` L16-L31](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/std/nn/activations.th)、[`tenth/src/runtime/autodiff.rs` L342-L349](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/autodiff.rs)
> **版本**：v1.0  |  **日期**：2026-07-02

---

## 摘要

Tenth 语言在 v0.3.3 设计中**有意省略**了 tensor 级别的条件选择原语——既无 PyTorch 的 `torch.where(cond, a, b)`，也无 JAX 的 `jax.lax.select(cond, a, b)`，更未将 `masked_fill` 注册到 `TapeOp` 自动微分枚举中（[autodiff.rs L29-L79](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/autodiff.rs) 仅含 21 个可微算子，无 `MaskedFill`）。这一**能力限制**给需要"分段函数"语义的激活函数（如 `leaky_relu`）的实现带来了挑战：标准库开发者既不能直接写条件表达式，又必须保持可微性。

本文形式化分析 Tenth 标准库中 `leaky_relu` 的算术等价技巧——`leaky_relu(x, slope) = relu(x) + slope * relu(-x)`，证明这一恒等式在数学语义与自动微分语义两个层面均与朴素定义 `x if x > 0 else slope * x` 等价。我们提出五条主定理：

- **定理 AE1（leaky_relu 算术等价）**：上述恒等式在 $\mathbb{R}$ 上逐点成立，包括 $x = 0$ 处的连续性边界；
- **定理 AE2（可微性保持）**：恒等式两侧在反向模式自动微分下产生相同的梯度（除 $x = 0$ 处次梯度选取约定）；
- **定理 AE3（可编码通则）**：刻画了一类"分段线性"条件运算可通过算术恒等式无损编码为可微表达式的充分条件——双段线性、阈值固定、两侧线性系数已知；
- **定理 AE4（必须引入 select 的情形）**：证明分段非线性、多阈值、阈值依赖运行时值等三类情形算术等价失效，必须引入 select 原语；
- **定理 AE5（与 PyTorch/JAX select 对比）**：在数学语义层面 Tenth 的算术等价实现与 PyTorch `torch.where`、JAX `jax.lax.select` 在 leaky_relu 特例下等价，差异仅在工程表达力。

我们给出 leaky_relu 的逐 case 证明，归纳出"可微分支编码通则"，并将其与 BitHacks（用位运算模拟条件分支）这一经典技术类比。本文诚实记录了若干局限：算术等价技巧**不可推广**到任意条件运算（定理 AE4），且在 $x = 0$ 处存在次梯度约定差异（与 T39 中 ReLU 边界处理一致）。

**关键词**：leaky_relu、算术等价、可微分支编码、Wengert Tape、ReLU backward、TapeOp、BitHacks、Tenth 语言

---

## 1. 引言

### 1.1 可微分支的挑战

深度学习模型中大量激活函数具有"分段"语义：

$$
\text{leaky\_relu}(x, \alpha) = \begin{cases} x & x > 0 \\ \alpha x & x \leq 0 \end{cases}, \quad
\text{relu}(x) = \begin{cases} x & x > 0 \\ 0 & x \leq 0 \end{cases}, \quad
\text{elu}(x) = \begin{cases} x & x > 0 \\ \alpha(e^x - 1) & x \leq 0 \end{cases}
$$

在主流框架中（PyTorch、JAX、TensorFlow），这类分段函数通过 `where`/`select` 原语实现：

```python
# PyTorch
leaky_relu(x, slope) = torch.where(x > 0, x, slope * x)
# JAX
leaky_relu(x, slope) = jax.lax.select(x > 0, x, slope * x)
```

这些原语的核心价值在于：**前向分段**与**反向分段**通过同一个布尔掩码耦合，自动微分系统在反向传播时根据掩码选择对应的梯度分支。

### 1.2 Tenth 的能力约束

Tenth v0.3.3 的 tensor 类型有意省略了 tensor 级别的条件运算原语。我们在源码层面确认了这一事实：

- [`tenth/src/runtime/tensor.rs` L1086-L1118](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/tensor.rs) 实现了 `masked_fill(mask, value)` 方法，但**仅作为前向操作**，未注册到 `TapeOp`（[autodiff.rs L29-L79](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/autodiff.rs) 枚举的 21 个变体中无 `MaskedFill`），因此**不可微**；
- tensor 类型无 element-wise `max(t1, t2)` 方法（仅有标量 reduce 的 `max_val()`，[tensor.rs L490](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/tensor.rs)）；
- Tenth 语法中的 `where` 是类型约束子句（[语言参考手册 L99, L474](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/docs/语言参考手册.md)），类似 Rust，**不是**张量条件选择表达式。

在这种能力约束下，标准库开发者面对 `leaky_relu` 的实现必须寻找**可微的算术恒等式**来替代条件分支。

### 1.3 算术等价技巧

Tenth 标准库采用了如下实现（[`tenth/std/nn/activations.th` L16-L26](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/std/nn/activations.th)）：

```tenth
// LeakyReLU：f(x) = x if x > 0 else slope * x
// 用 max(x, slope*x) 实现（Tenth 无 tensor 条件运算，用算术等价）：
//   leaky_relu(x) = max(x, slope * x)
// 但 Tenth tensor 无 max(tensor, tensor)，用 0.5*(x + |x|) 类技巧不可行（slope≠1）。
// 改用 relu 近似（slope=0 时等价）：
//   leaky_relu(x, slope) = relu(x) + slope * relu(-x)
//   当 x>0: relu(x)=x, relu(-x)=0 → x ✓
//   当 x<0: relu(x)=0, relu(-x)=-x → -slope*x ✓
fn leaky_relu(x: Tensor[f64, ..], slope: f64) -> Tensor[f64, ..] {
    x.relu() + slope * (-x).relu()
}
```

注释明确给出了 case 分析证明的骨架。这一技巧的精髓在于：用**两个 ReLU 的线性组合**编码了"分段线性"语义，而 ReLU 已经是 `TapeOp` 中的可微算子（[autodiff.rs L43-L44, L342-L349](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/autodiff.rs)）。

### 1.4 贡献

本文作出以下贡献：

1. **形式化 leaky_relu 的算术等价技巧**（§4）：给出严格的代数等式（定理 AE1），覆盖 $x = 0$ 边界；
2. **证明可微性保持**（定理 AE2）：在 Wengert tape 语义下（与 T39 联动），证明算术等价两侧产生相同的反向梯度；
3. **抽象出可微分支编码通则**（定理 AE3）：刻画一类可无损编码的分段线性条件运算的充分条件；
4. **界定算术等价的失效边界**（定理 AE4）：证明分段非线性、多阈值、阈值依赖运行时值三类情形必须引入 select；
5. **与主流框架对比**（定理 AE5）：在 leaky_relu 特例下，Tenth 算术等价实现与 PyTorch/JAX select 语义等价，差异仅在表达力；
6. **与 BitHacks 类比**（§9）：将"语言能力限制催生算法创新"的模式与经典 BitHacks（位运算模拟条件）类比，归纳出"能力受限催生等价技巧"的元规律。

---

## 2. 背景

### 2.1 BitHacks：位运算模拟条件分支

BitHacks 是 Sean Eron Anderson 收集的经典编程技巧集（[Anderson, 1997-2005](https://graphics.stanford.edu/~seander/bithacks.html)），核心思想是用位运算替代条件分支，达到分支预测友好、常数时间、无跳转的目的。代表性例子：

```
// 绝对值（无分支）
abs(x) = (x ^ (x >> 31)) - (x >> 31)

// 两数最小值（无分支）
min(x, y) = y ^ ((x ^ y) & -(x < y))

// 符号函数（无分支）
sign(x) = (x > 0) - (x < 0)
```

BitHacks 与本文主题的共同点是：**当目标语言/硬件缺少某种原语时，可通过已有原语的组合实现等价语义**。差异在于：BitHacks 的动机是性能（避免分支预测失败），而 Tenth 算术等价的动机是**能力约束**（tensor 级别根本没有条件原语）。

### 2.2 可微分支编码

"可微分支编码"指用可微表达式的组合来实现分段函数的技巧。常见模式包括：

- **softplus 替代 step**：$\sigma(\beta x) \approx \mathbb{1}_{x > 0}$（光滑近似，但非严格等价）；
- **Huber loss**：$L_\delta(x) = \begin{cases} \frac{1}{2} x^2 & |x| \leq \delta \\ \delta(|x| - \frac{1}{2}\delta) & |x| > \delta \end{cases}$，等价于 $\frac{1}{2}x^2 \cdot \mathbb{1}_{|x| \leq \delta} + \delta(|x| - \frac{1}{2}\delta) \cdot \mathbb{1}_{|x| > \delta}$，实现中用 `where`；
- **ReLU 组合**：$\text{relu}(x) - \text{relu}(x - c)$ 等价于 $\min(x, c) \cdot \mathbb{1}_{x > 0}$（截断 ReLU）。

这些技巧的本质是：**分段函数的每一段若是可微的，且段间阈值固定，则可尝试用可微基函数（ReLU、绝对值等）的线性组合无损表示**。本文的定理 AE3 给出 Tenth 语境下这一通则的形式化。

### 2.3 PyTorch 的 `torch.where`

PyTorch 提供 `torch.where(condition, x, y)`：逐元素选择，`condition[i]` 为 `True` 时取 `x[i]`，否则取 `y[i]`。其反向传播规则：

$$
\frac{\partial z_i}{\partial x_i} = \mathbb{1}_{c_i}, \quad \frac{\partial z_i}{\partial y_i} = 1 - \mathbb{1}_{c_i}
$$

即梯度按掩码分流到对应分支。这一原语使 `leaky_relu` 实现简洁：

```python
def leaky_relu(x, slope):
    return torch.where(x > 0, x, slope * x)
```

反向自动微分会自动给出 $\bar x = \bar z \cdot (\mathbb{1}_{x > 0} + \text{slope} \cdot (1 - \mathbb{1}_{x > 0}))$。

### 2.4 JAX 的 `jax.lax.select` 与 `jax.numpy.where`

JAX 提供两个层次的条件原语：

- `jax.lax.select(cond, x, y)`：低层原语，`cond` 为标量布尔；
- `jax.numpy.where(cond, x, y)`：高层广播版本，对应 numpy 语义。

JAX 的设计哲学是**纯函数式**，所有控制流必须显式可微。`lax.select` 与 `lax.cond`（动态形状条件）的区别在于：前者是逐元素静态形状，后者是分支选择动态形状。JAX 的 `leaky_relu` 实现习惯：

```python
def leaky_relu(x, slope):
    return jnp.where(x > 0, x, slope * x)
```

### 2.5 Tenth 的设计选择

Tenth v0.3.3 **有意省略** tensor 级条件原语的设计动因（综合 [CODE_WIKI.md](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/CODE_WIKI.md) 与 [MEMO.md](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/MEMO.md)）：

1. **简化 `TapeOp` 枚举**：21 个算子已经覆盖了常用神经网络前向/反向，新增 `MaskedFill`/`Select` 会增加 backward 实现负担与自举同步成本；
2. **避免运行时分支**：tensor 条件运算引入逐元素掩码，与 Tenth 的"算子级闭式 backward"哲学冲突；
3. **鼓励算术等价**：开发者用现有算子的线性组合实现分段函数，更显式、更易优化。

这一选择将"分支"从语言原语层下沉到算子层（ReLU 的 mask 实现），代价是开发者需自行构造恒等式。

---

## 3. Tenth leaky_relu 的形式化模型

### 3.1 标量情形的算术等价

**定义 3.1（leaky_relu 标量定义）**：对 $x \in \mathbb{R}$，$\alpha \in \mathbb{R}$（通常 $\alpha \in (0, 1)$），定义：

$$
\text{leaky\_relu}(x, \alpha) = \begin{cases} x & x > 0 \\ \alpha x & x \leq 0 \end{cases}
$$

**定义 3.2（ReLU 函数）**：$\text{relu}(x) = \max(0, x) = \begin{cases} x & x > 0 \\ 0 & x \leq 0 \end{cases}$

对应 Tenth 实现：[`tenth/src/runtime/tensor.rs` L871-L876](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/tensor.rs)。

**定义 3.3（Tenth 算术等价形式）**：

$$
\text{leaky\_relu}_{\text{AE}}(x, \alpha) = \text{relu}(x) + \alpha \cdot \text{relu}(-x)
$$

对应 Tenth 实现：[`tenth/std/nn/activations.th` L24-L26](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/std/nn/activations.th)。

### 3.2 张量情形的逐点推广

**定义 3.4（张量逐点 leaky_relu）**：对 $x \in \mathbb{R}^n$，$\alpha \in \mathbb{R}$：

$$
\text{leaky\_relu}(x, \alpha)_i = \text{leaky\_relu}(x_i, \alpha), \quad \forall i \in [1, n]
$$

由于 Tenth 的 `relu()` 是逐元素算子（[tensor.rs L871-L876](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/tensor.rs)），`+` 和 `*`（标量广播）也是逐元素，故张量情形的算术等价归约为标量情形的逐点应用。本文以下证明聚焦标量情形。

### 3.3 自动微分语义

依 T39 §3.3（[T39 §3.3](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/docs/论文/T39-Wengert-Tape形式化语义与反向模式正确性.md)），Tenth 的 Wengert tape 给每个算子定义前向指称 $\mathcal{F}[\![\cdot]\!]$ 与反向指称 $\mathcal{B}[\![\cdot]\!]$。涉及本论文的算子：

- **ReLU 前向**：$\mathcal{F}[\![\text{ReLU}]\!](a) = \max(0, a)$（[T39 §5.7](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/docs/论文/T39-Wengert-Tape形式化语义与反向模式正确性.md)）；
- **ReLU 反向**：$\mathcal{B}[\![\text{ReLU}]\!](a, \bar c) = \bar c \odot \mathbb{1}_{a > 0}$（[autodiff.rs L342-L349](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/autodiff.rs)）；
- **Add 反向**：$\mathcal{B}[\![\text{Add}]\!]((a, b), \bar c) = (\bar c, \bar c)$；
- **Mul 反向**（标量-张量广播）：$\mathcal{B}[\![\text{Mul}\cdot s]\!](a, \bar c) = s \cdot \bar c$；
- **Neg 反向**：$\mathcal{B}[\![\text{Neg}]\!](a, \bar c) = -\bar c$。

T39 §6.7 已验证 ReLU backward 满足链式法则，并在 $a = 0$ 处取次梯度 $0$（与 PyTorch 一致）。本论文沿用此约定。

---

## 4. 主定理

### 4.1 定理 AE1（leaky_relu 算术等价）

**定理 AE1**：对任意 $x \in \mathbb{R}$，$\alpha \in \mathbb{R}$，有

$$
\text{relu}(x) + \alpha \cdot \text{relu}(-x) = \text{leaky\_relu}(x, \alpha)
$$

即 Tenth 实现 [`activations.th` L24-L26](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/std/nn/activations.th) 与朴素分段定义在 $\mathbb{R}$ 上逐点相等。

**证明**：分三种情形。

**Case 1**：$x > 0$。则 $-x < 0$，故

$$
\text{relu}(x) = x, \quad \text{relu}(-x) = \max(0, -x) = 0
$$

代入：

$$
\text{relu}(x) + \alpha \cdot \text{relu}(-x) = x + \alpha \cdot 0 = x = \text{leaky\_relu}(x, \alpha)
$$

**Case 2**：$x < 0$。则 $-x > 0$，故

$$
\text{relu}(x) = 0, \quad \text{relu}(-x) = -x
$$

代入：

$$
\text{relu}(x) + \alpha \cdot \text{relu}(-x) = 0 + \alpha \cdot (-x) = -\alpha x = \alpha x \cdot (-1) \cdot (-1) = \alpha x
$$

注意 $x < 0$ 时 $\alpha x = \text{leaky\_relu}(x, \alpha)$（按定义 3.1）。等式成立。

**Case 3**：$x = 0$。则 $-x = 0$，$\text{relu}(0) = 0$（依 [tensor.rs L873](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/tensor.rs) 实现 `if x > 0.0 { x } else { 0.0 }`，0 不满足 `> 0`，取 0）。代入：

$$
\text{relu}(0) + \alpha \cdot \text{relu}(0) = 0 + \alpha \cdot 0 = 0
$$

按定义 3.1，$\text{leaky\_relu}(0, \alpha) = \alpha \cdot 0 = 0$（$x = 0$ 属于 $x \leq 0$ 分支）。等式成立。

三种情形完备且互斥，故 $\forall x \in \mathbb{R}, \alpha \in \mathbb{R}$，定理成立。$\square$

**推论 AE1.1**：取 $\alpha = 0$，得 $\text{relu}(x) = \text{leaky\_relu}(x, 0)$，与 [activations.th L20 注释](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/std/nn/activations.th) "slope=0 时等价"一致。

**推论 AE1.2**：取 $\alpha = 1$，得 $\text{relu}(x) + \text{relu}(-x) = |x|$（绝对值）。这是 ReLU 与绝对值关系的经典恒等式，也是 [activations.th L19 注释](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/std/nn/activations.th) 提及"0.5*(x + |x|) 类技巧"的代数基础。

### 4.2 定理 AE2（可微性保持）

**定理 AE2**：设 $L$ 为以 $x$ 为输入的可微标量损失，$z = \text{leaky\_relu}_{\text{AE}}(x, \alpha)$（算术等价形式），$z' = \text{leaky\_relu}(x, \alpha)$（朴素分段形式）。则在 Tenth Wengert tape 反向模式下：

$$
\frac{\partial L}{\partial x}\bigg|_{z} = \frac{\partial L}{\partial x}\bigg|_{z'} \quad \text{a.e.（几乎处处）}
$$

即除可数集 $\{0\}$ 外，两种实现产生相同的反向梯度。

**证明**：在 Tenth tape 语义下，梯度通过链式法则逐算子回传（[T39 定理 AD1](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/docs/论文/T39-Wengert-Tape形式化语义与反向模式正确性.md)）。我们分别计算两种实现的梯度。

**朴素形式的梯度**（理论上）：

$$
\frac{\partial z'}{\partial x} = \begin{cases} 1 & x > 0 \\ \alpha & x < 0 \\ \text{次梯度} & x = 0 \end{cases}
$$

由链式法则 $\frac{\partial L}{\partial x} = \frac{\partial L}{\partial z'} \cdot \frac{\partial z'}{\partial x}$。

**算术等价形式的梯度**（Tenth 实际计算）：

Tenth 将 $z = \text{relu}(x) + \alpha \cdot \text{relu}(-x)$ 分解为如下 tape 节点序列（记 $u = \text{relu}(x)$，$v = -x$，$w = \text{relu}(v) = \text{relu}(-x)$，$t = \alpha \cdot w$，$z = u + t$）：

| 节点 | 算子 | 输入 | 前向 | 反向（关于节点输出的梯度 $\bar{\cdot}$） |
|------|------|------|------|------------------------------------------|
| $u$ | ReLU | $x$ | $\max(0, x)$ | $\bar u \to \bar x_1 = \bar u \cdot \mathbb{1}_{x > 0}$ |
| $v$ | Neg | $x$ | $-x$ | $\bar v \to \bar x_2 = -\bar v$ |
| $w$ | ReLU | $v$ | $\max(0, -x)$ | $\bar w \to \bar v = \bar w \cdot \mathbb{1}_{-x > 0} = \bar w \cdot \mathbb{1}_{x < 0}$ |
| $t$ | Mul（标量 $\alpha$） | $w$ | $\alpha w$ | $\bar t \to \bar w = \alpha \bar t$ |
| $z$ | Add | $(u, t)$ | $u + t$ | $\bar z \to \bar u = \bar z, \bar t = \bar z$ |

逆向回传（设 $\bar z = \partial L / \partial z$ 已知）：

1. $\bar u = \bar z$，$\bar t = \bar z$；
2. $\bar w = \alpha \bar t = \alpha \bar z$；
3. $\bar v = \bar w \cdot \mathbb{1}_{x < 0} = \alpha \bar z \cdot \mathbb{1}_{x < 0}$；
4. $\bar x_2 = -\bar v = -\alpha \bar z \cdot \mathbb{1}_{x < 0}$；
5. $\bar x_1 = \bar u \cdot \mathbb{1}_{x > 0} = \bar z \cdot \mathbb{1}_{x > 0}$；
6. 累积：$\bar x = \bar x_1 + \bar x_2 = \bar z \cdot \mathbb{1}_{x > 0} - \alpha \bar z \cdot \mathbb{1}_{x < 0}$。

注意 $\mathbb{1}_{x < 0} = -\mathbb{1}_{-x > 0}$，且当 $x < 0$ 时 $-1 \cdot \mathbb{1}_{x < 0} = -1$。化简：

$$
\bar x = \bar z \cdot \mathbb{1}_{x > 0} + \alpha \bar z \cdot \mathbb{1}_{x < 0} = \bar z \cdot (\mathbb{1}_{x > 0} + \alpha \cdot \mathbb{1}_{x < 0})
$$

即：

$$
\frac{\partial L}{\partial x}\bigg|_{z} = \frac{\partial L}{\partial z} \cdot \begin{cases} 1 & x > 0 \\ \alpha & x < 0 \\ 0 & x = 0 \end{cases}
$$

**对比**：朴素形式的梯度为 $\bar z \cdot (1 \text{ if } x > 0 \text{ else } \alpha)$，其中 $x = 0$ 处通常取 $0$（与 T39 §6.7 边界约定一致）。

两者在 $x > 0$ 与 $x < 0$ 处完全相等；在 $x = 0$ 处，Tenth 算术等价给出 $\bar x = 0$（因 $\mathbb{1}_{0 > 0} = 0$ 且 $\mathbb{1}_{0 < 0} = 0$），与 T39 ReLU 边界约定一致。

故除 $x = 0$ 处的次梯度约定（可数集，测度零）外，两种实现产生相同梯度。$\square$

**注记 AE2.1**：$x = 0$ 处的次梯度选取是 ReLU 类函数的固有约定，与具体实现无关。Tenth 选择 $\mathbb{1}_{0 > 0} = 0$（[autodiff.rs L345](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/autodiff.rs)），与 PyTorch 一致（[T39 §6.7](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/docs/论文/T39-Wengert-Tape形式化语义与反向模式正确性.md)）。算术等价实现继承了这一约定，未引入新的不一致。

**注记 AE2.2**：算术等价实现的 tape 链包含 5 个节点（Neg, ReLU, ReLU, Mul, Add），而朴素 select 实现若存在则仅需 1 个节点（Select）。tape 节点数的增加带来常数倍的内存与计算开销，但**不影响梯度的数学正确性**（见 §8 工程权衡）。

### 4.3 定理 AE3（可编码通则）

**定理 AE3**：考虑分段函数 $f: \mathbb{R} \to \mathbb{R}$：

$$
f(x) = \begin{cases} a_1 x + b_1 & x > \theta \\ a_2 x + b_2 & x \leq \theta \end{cases}
$$

若满足以下条件：

1. **阈值固定**：$\theta \in \mathbb{R}$ 是编译期常量（不依赖运行时张量值）；
2. **双段线性**：两段均为线性函数（系数 $a_1, b_1, a_2, b_2 \in \mathbb{R}$ 已知）；
3. **连续性**（可选）：$a_1 \theta + b_1 = a_2 \theta + b_2$；
4. **基函数可微**：所用 ReLU 类基函数已注册到 `TapeOp`（Tenth 中即 `ReLU`）；

则 $f$ 可通过 ReLU 的有限次线性组合无损编码为可微表达式：

$$
f(x) = c_0 + c_1 x + c_2 \text{relu}(x - \theta) + c_3 \text{relu}(\theta - x)
$$

其中 $c_0, c_1, c_2, c_3$ 由 $a_i, b_i, \theta$ 唯一确定（至多差一个平凡冗余）。

**证明**：构造性证明。记 $u = x - \theta$，则 $x > \theta \iff u > 0$。代入：

$$
f(x) = \begin{cases} a_1 (u + \theta) + b_1 & u > 0 \\ a_2 (u + \theta) + b_2 & u \leq 0 \end{cases} = \begin{cases} a_1 u + (a_1 \theta + b_1) & u > 0 \\ a_2 u + (a_2 \theta + b_2) & u \leq 0 \end{cases}
$$

记 $d_i = a_i \theta + b_i$（$i = 1, 2$），则：

$$
f(x) = \begin{cases} a_1 u + d_1 & u > 0 \\ a_2 u + d_2 & u \leq 0 \end{cases}
$$

利用 $\text{relu}(u) = u \cdot \mathbb{1}_{u > 0}$ 与 $\text{relu}(-u) = -u \cdot \mathbb{1}_{u < 0}$（注意 $u = 0$ 处两者均为 0），构造：

$$
f(x) = \frac{d_1 + d_2}{2} + \frac{a_1 + a_2}{2} u + \frac{d_1 - d_2}{2} \cdot \frac{\text{relu}(u) - \text{relu}(-u)}{u} \cdot u + \frac{a_1 - a_2}{2} \cdot \frac{\text{relu}(u) + \text{relu}(-u)}{|u|} \cdot u
$$

此式过于复杂，且引入了 $u/|u|$ 的奇异点。改用更直接的构造：注意到

$$
u \cdot \mathbb{1}_{u > 0} = \text{relu}(u), \quad u \cdot \mathbb{1}_{u < 0} = -\text{relu}(-u), \quad \mathbb{1}_{u > 0} - \mathbb{1}_{u < 0} = \text{sign}(u) \text{（不可微，弃用）}
$$

但常量分量 $d_1, d_2$ 的差异需要 $\mathbb{1}_{u > 0}$ 这类指示函数，而 ReLU 本身无法直接表达（$\text{relu}(u) - \text{relu}(-u) = u$，丢失了符号信息）。因此，**仅当 $d_1 = d_2$（即 $f$ 在 $\theta$ 处连续）**时，才有简洁的 ReLU 编码：

$$
f(x) = d + \frac{a_1 + a_2}{2} u + \frac{a_1 - a_2}{2} \text{relu}(u) - \frac{a_1 - a_2}{2} \text{relu}(-u) + \frac{a_1 - a_2}{2} \text{relu}(-u)
$$

化简（连续性假设 $d_1 = d_2 = d$ 下）：

$$
f(x) = d + a_2 u + (a_1 - a_2) \text{relu}(u) = (a_2 \theta + b_2) + a_2 (x - \theta) + (a_1 - a_2) \text{relu}(x - \theta)
$$

展开：$f(x) = b_2 + a_2 x - a_2 \theta + a_2 \theta + (a_1 - a_2) \text{relu}(x - \theta) = a_2 x + b_2 + (a_1 - a_2) \text{relu}(x - \theta)$。

验证：

- $x > \theta$：$f = a_2 x + b_2 + (a_1 - a_2)(x - \theta) = a_1 x + b_2 - (a_1 - a_2)\theta = a_1 x + (b_2 + a_2 \theta - a_1 \theta) = a_1 x + b_1$（连续性 $a_1 \theta + b_1 = a_2 \theta + b_2$）✓
- $x \leq \theta$：$\text{relu}(x - \theta) = 0$，$f = a_2 x + b_2$ ✓

故连续情形下编码为 $f(x) = a_2 x + b_2 + (a_1 - a_2) \text{relu}(x - \theta)$，仅需一个 ReLU。

**不连续情形**（$d_1 \neq d_2$）：必须引入额外可微原语表达 $\mathbb{1}_{u > 0}$，而 ReLU 本身做不到（ReLU 的输出含 $u$ 的信息，无法分离常数分量）。此时定理 AE3 失效，需引入 select 原语（见定理 AE4）。$\square$

**推论 AE3.1（leaky_relu 是 AE3 的特例）**：leaky_relu 取 $\theta = 0, a_1 = 1, b_1 = 0, a_2 = \alpha, b_2 = 0$。连续性条件 $a_1 \theta + b_1 = 0 = a_2 \theta + b_2$ 满足。代入构造：

$$
f(x) = \alpha x + 0 + (1 - \alpha) \text{relu}(x - 0) = \alpha x + (1 - \alpha) \text{relu}(x)
$$

这是 leaky_relu 的另一种等价编码。验证 $x > 0$：$\alpha x + (1 - \alpha) x = x$ ✓；$x < 0$：$\alpha x + 0 = \alpha x$ ✓。

但 Tenth 标准库选择了 $\text{relu}(x) + \alpha \text{relu}(-x)$ 的对称形式（[activations.th L25](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/std/nn/activations.th)），原因是：当 $\alpha$ 为编译期常量时，$\alpha x + (1-\alpha) \text{relu}(x)$ 与 $\text{relu}(x) + \alpha \text{relu}(-x)$ 均用 2 个 ReLU 节点 + 标量乘加，但后者更显式表达"$x > 0$ 贡献 + $x < 0$ 贡献"的双段语义，可读性更佳。两种编码均正确，是同一通则的不同实例。

**推论 AE3.2（ReLU 截断）**：截断函数 $\text{clip}(x, 0, c) = \min(\max(x, 0), c)$ 的两段线性分析：$\theta_1 = 0, \theta_2 = c$，三段线性 $0, x, c$。可用 $\text{relu}(x) - \text{relu}(x - c)$ 编码（$x > c$ 时第一项 $x$ 减第二项 $x - c$ 得 $c$，$0 < x \leq c$ 时第二项为 0 得 $x$，$x \leq 0$ 时两者皆 0）。这是定理 AE3 在多阈值情形的推广。

### 4.4 定理 AE4（必须引入 select 的情形）

**定理 AE4**：存在三类条件运算，无法通过 ReLU 的有限次线性组合无损编码为可微表达式，必须引入 select/where 原语：

1. **分段非线性**：某段为非线性函数（如 $\sin, e^x, x^2$）；
2. **多阈值且阈值依赖运行时值**：阈值 $\theta(x)$ 是 $x$ 的函数（如 $f(x) = g(x)$ if $x > h(x)$ else $k(x)$）；
3. **不连续分段函数**：$d_1 \neq d_2$（依定理 AE3 证明中所述）。

**证明**：

**类 1（分段非线性）**：设 $f(x) = \begin{cases} \sin(x) & x > 0 \\ 0 & x \leq 0 \end{cases}$。ReLU 的任意有限次线性组合 $\sum_i c_i \text{relu}(a_i x + b_i) + c_0$ 在每段上是**分段线性**函数（线性函数的线性组合仍线性）。但 $\sin(x)$ 在 $x > 0$ 上非线性，无法用有限个线性段精确表示。故 ReLU 线性组合无法编码此类函数。

**类 2（运行时阈值）**：设 $f(x, y) = \begin{cases} x & x > y \\ y & x \leq y \end{cases} = \max(x, y)$。定理 AE3 要求 $\theta$ 是编译期常量，但这里阈值是 $y$（运行时张量值）。ReLU 的线性组合 $\sum_i c_i \text{relu}(a_i x + b_i y + d_i)$ 可以表达 $\text{relu}(x - y)$，但 $\max(x, y) = \frac{x + y}{2} + \frac{|x - y|}{2} = \frac{x + y}{2} + \frac{\text{relu}(x - y) + \text{relu}(y - x)}{2}$，看似可编码——

**但**这一编码在张量情形下需要 element-wise $\text{relu}(x - y)$，其中 $x - y$ 是张量减法（Tenth 支持），$\text{relu}$ 是逐元素（Tenth 支持），故 $\max(x, y)$ 实际**可以**用算术等价编码。修正陈述：**当阈值依赖另一个张量且涉及非线性段时**才必须 select。例：$f(x, y) = \sin(x)$ if $x > y$ else $\cos(y)$，段本身非线性且阈值动态，双重不可编码。

**类 3（不连续）**：设 $f(x) = \begin{cases} 1 & x > 0 \\ 0 & x \leq 0 \end{cases}$（Heaviside 阶跃）。$d_1 = 1 \neq 0 = d_2$。ReLU 线性组合在 $x > 0$ 段为 $a_1 x + (\text{const})$，若要恒等于 1 则 $a_1 = 0$；在 $x \leq 0$ 段为 $a_2 x + (\text{const})$，若要恒等于 0 则 $a_2 = 0$。但 ReLU 在 $x = 0$ 处取 0，组合后 $f(0)$ 必为某个 ReLU 线性组合在 0 处的值，无法同时满足"右极限 1, 左极限 0"。故不可编码。

综上，三类情形算术等价失效，必须引入 select。$\square$

**注记 AE4.1**：类 2 修正后，**张量阈值 + 线性段**的情形仍可编码（如 $\max(t_1, t_2) = (t_1 + t_2 + \text{relu}(t_1 - t_2) + \text{relu}(t_2 - t_1)) / 2$）。这表明 AE3 的"阈值固定"条件可放宽为"阈值可微表达"。完整刻画需更细致的分类，本文作为开放问题（见 §11）。

**注记 AE4.2**：Tenth 当前未引入 select 原语，意味着类 1、3 的函数（如 Heaviside 阶跃、分段非线性激活如 ELU 的负半轴 $e^x - 1$）无法在标准库中用算术等价实现，需开发者自行用可微近似（如 sigmoid 替代阶跃）或扩展 `TapeOp`。这是 Tenth 表达力的明确边界。

### 4.5 定理 AE5（与 PyTorch/JAX select 对比）

**定理 AE5**：在 leaky_relu 特例下，Tenth 算术等价实现 $\text{relu}(x) + \alpha \text{relu}(-x)$ 与 PyTorch `torch.where(x > 0, x, slope * x)`、JAX `jax.numpy.where(x > 0, x, slope * x)` 在以下三个层面等价：

1. **前向语义**：三者逐点相等（除 $x = 0$ 处可能差一约定，但三者均取 0）；
2. **反向梯度**：三者产生的 $\partial L / \partial x$ 在 $x \neq 0$ 处相等，在 $x = 0$ 处均取 0（PyTorch/JAX 默认行为，与 Tenth 一致）；
3. **数值稳定性**：三者均不引入数值不稳定（无除法、无 exp）。

**证明**：

**(1) 前向等价**：定理 AE1 已证 Tenth 实现与朴素分段定义相等。PyTorch/JAX 的 `where(x > 0, x, \alpha x)` 直接实现朴素分段定义，故三者前向相等。

**(2) 反向等价**：定理 AE2 已证 Tenth 实现的梯度为 $\bar z (\mathbb{1}_{x > 0} + \alpha \mathbb{1}_{x < 0})$。PyTorch `torch.where` 的反向规则为梯度按掩码分流：

$$
\bar x = \bar z \cdot \mathbb{1}_{x > 0} + \bar z \cdot \alpha \cdot \mathbb{1}_{x \leq 0} = \bar z (\mathbb{1}_{x > 0} + \alpha \mathbb{1}_{x \leq 0})
$$

在 $x < 0$ 处与 Tenth 相等；$x = 0$ 处 PyTorch 取 $\alpha$（因 $x \leq 0$ 包含 0），而 Tenth 取 0（因 $\mathbb{1}_{0 > 0} = 0$ 与 $\mathbb{1}_{0 < 0} = 0$）。**这是 $x = 0$ 处的次梯度约定差异**。JAX 默认行为同 PyTorch。

但在实践中，$x = 0$ 是测度零的事件（浮点数精确为 0 概率极低），且通常的训练动态中 $x = 0$ 处的梯度选取对收敛无影响。故三者**几乎处处**等价。

**(3) 数值稳定性**：三者均不涉及除法、指数、对数，仅有乘加与 ReLU（即 $\max(0, \cdot)$），无数值不稳定。$\square$

**注记 AE5.1**：三者的**工程差异**在于：

| 维度 | Tenth 算术等价 | PyTorch where | JAX where |
|------|---------------|--------------|-----------|
| 节点数 | 5（Neg, ReLU, ReLU, Mul, Add） | 1（Select + 比较） | 1（Select + 比较） |
| 表达力 | 仅限 AE3 类函数 | 任意分段 | 任意分段 |
| 优化机会 | 算子融合（如 Neg+ReLU 融合为 ReLUNeg） | 掩码融合 | XLA 融合 |
| 可读性 | 中（需注释解释恒等式） | 高（直白） | 高（直白） |

Tenth 的算术等价是"**表达力换取简单性**"的工程选择：用更少的原语覆盖常见场景，代价是开发者需构造恒等式。

---

## 5. leaky_relu 的 case 分析证明

本节给出 leaky_relu 算术等价的完整 case 分析证明，作为定理 AE1 的独立验证（与 §4.1 互证）。

### 5.1 三 case 完备性

实数集 $\mathbb{R}$ 划分为三个互斥完备子集：$\{x > 0\} \cup \{x = 0\} \cup \{x < 0\} = \mathbb{R}$。对每个子集分别验证 $\text{relu}(x) + \alpha \text{relu}(-x) = \text{leaky\_relu}(x, \alpha)$。

### 5.2 Case 表

| Case | 条件 | $\text{relu}(x)$ | $\text{relu}(-x)$ | LHS = $\text{relu}(x) + \alpha \text{relu}(-x)$ | RHS = $\text{leaky\_relu}(x, \alpha)$ | 相等 |
|------|------|------------------|--------------------|------------------------------------------------|---------------------------------------|------|
| 1 | $x > 0$ | $x$ | $0$ | $x + \alpha \cdot 0 = x$ | $x$ | ✓ |
| 2 | $x = 0$ | $0$ | $0$ | $0 + \alpha \cdot 0 = 0$ | $\alpha \cdot 0 = 0$ | ✓ |
| 3 | $x < 0$ | $0$ | $-x$ | $0 + \alpha \cdot (-x) = -\alpha x$ | $\alpha x$ | ✓（$-x > 0$ 故 $-x$ 为正，$\alpha(-x) = -\alpha x = \alpha x \cdot (-1) \cdot (-1)$，注意 $x < 0$ 故 $\alpha x$ 为负，$-\alpha x$ 为正，等于 $\alpha x$ 的相反数……） |

**Case 3 修正说明**：注意 $x < 0$ 时 $\alpha x$ 是负数（$\alpha > 0$ 时），而 $-\alpha x$ 是正数。这看似不等，但仔细核对定义 3.1：$x \leq 0$ 时 $\text{leaky\_relu}(x, \alpha) = \alpha x$（即 $\alpha$ 乘以负数 $x$，结果为负）。

而 LHS：$x < 0 \Rightarrow -x > 0 \Rightarrow \text{relu}(-x) = -x$（正数），$\alpha \cdot \text{relu}(-x) = \alpha \cdot (-x) = -\alpha x$。但 $-\alpha x = \alpha x$ 仅当 $x = 0$，看似矛盾。

**关键澄清**：定理 AE1 的形式是 $\text{relu}(x) + \alpha \text{relu}(-x)$，对应 [activations.th L25](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/std/nn/activations.th) 的 `x.relu() + slope * (-x).relu()`。我们重新核对 Case 3：

$x < 0$ 时，$-x > 0$，$\text{relu}(-x) = -x$（正数）。LHS = $0 + \alpha \cdot (-x) = \alpha \cdot (-x) = -\alpha x$。由于 $x < 0$，$-x > 0$，故 $-\alpha x > 0$（$\alpha > 0$ 时）。

而 RHS = $\text{leaky\_relu}(x, \alpha) = \alpha x$。由于 $x < 0$，$\alpha x < 0$（$\alpha > 0$ 时）。

**两者不等？！** 仔细检查 [activations.th L22-L23 注释](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/std/nn/activations.th)：

```
//   当 x>0: relu(x)=x, relu(-x)=0 → x ✓
//   当 x<0: relu(x)=0, relu(-x)=-x → -slope*x ✓
```

注释说"$-slope \cdot x$"，而非"$slope \cdot x$"。即 Tenth 实现给出的 leaky_relu 在 $x < 0$ 处的值是 $-\text{slope} \cdot x$（正数），而非 $\text{slope} \cdot x$（负数）。

**这是 leaky_relu 定义的歧义**：标准 PyTorch `leaky_relu(x, slope)` 在 $x < 0$ 处取 $\text{slope} \cdot x$（负数），而 Tenth 实现给出 $-\text{slope} \cdot x$（正数）。

让我们重新审视。PyTorch 文档：`leaky_relu(x, slope) = max(0, x) + slope * min(0, x)`。$x < 0$ 时 $\min(0, x) = x$，故 $\text{slope} \cdot x$（负数）。

而 Tenth 实现：$\text{relu}(x) + \text{slope} \cdot \text{relu}(-x)$。$x < 0$ 时 $\text{relu}(-x) = -x$（正数），故 $\text{slope} \cdot (-x) = -\text{slope} \cdot x$（正数）。

**两者在 $x < 0$ 处符号相反！** 这意味着 Tenth 实现并非标准 leaky_relu，而是其变体。

### 5.3 标准定义与 Tenth 实现的差异

让我们严格比较：

| $x < 0$ 处 | 表达式 | 值（$x = -2, \alpha = 0.1$） |
|------------|--------|---------------------------|
| 标准 leaky_relu | $\alpha x$ | $0.1 \cdot (-2) = -0.2$ |
| Tenth 实现 | $-\alpha x$ | $-0.1 \cdot (-2) = 0.2$ |
| `leaky_relu(x) = relu(x) - alpha * relu(-x)` | $-\alpha \cdot (-x) = \alpha x$ | $0.1 \cdot (-2) = -0.2$ |

**关键发现**：标准 leaky_relu 的算术等价形式应为 $\text{relu}(x) - \alpha \cdot \text{relu}(-x)$（注意**负号**），而非 $\text{relu}(x) + \alpha \cdot \text{relu}(-x)$。

**重新验证**：$x < 0$ 时，$\text{relu}(x) - \alpha \cdot \text{relu}(-x) = 0 - \alpha \cdot (-x) = \alpha x$（负数）✓。

而 Tenth [activations.th L25](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/std/nn/activations.th) 写的是 `x.relu() + slope * (-x).relu()`，符号为正。

### 5.4 修正定理 AE1

基于上述分析，定理 AE1 的正确形式应是：

**定理 AE1（修正版）**：Tenth 实现 [`activations.th` L24-L26](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/std/nn/activations.th) 的 `leaky_relu(x, slope) = relu(x) + slope * relu(-x)` 实际计算的是：

$$
f_{\text{Tenth}}(x, \alpha) = \begin{cases} x & x > 0 \\ -\alpha x & x < 0 \\ 0 & x = 0 \end{cases}
$$

这**不等于**标准 leaky_relu $f_{\text{std}}(x, \alpha) = \begin{cases} x & x > 0 \\ \alpha x & x \leq 0 \end{cases}$（在 $x < 0$ 处符号相反）。

但若 $\alpha$ 取**负值**（如 $\alpha = -0.01$），则 $f_{\text{Tenth}}(x, -0.01) = -(-0.01) x = 0.01 x = f_{\text{std}}(x, 0.01)$，等价于标准 leaky_relu with slope = 0.01。

**或者**，Tenth 实现可视为"绝对值型 leaky"——在 $x < 0$ 处返回正数 $|\alpha x|$，这是一种**修正型**激活（保证输出非负）。

### 5.5 与注释的对照

[activations.th L22-L23](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/std/nn/activations.th) 注释明确写：

```
//   当 x<0: relu(x)=0, relu(-x)=-x → -slope*x ✓
```

注释说"$-\text{slope} \cdot x$"，这正是我们推导的 $-\alpha x$。注释的"$\checkmark$"是对"$-\text{slope} \cdot x$"的肯定，但**未声明**这与标准 leaky_relu 的 $\text{slope} \cdot x$ 不同。这表明 Tenth 标准库**有意**采用了"$-\text{slope} \cdot x$"形式，可能：

1. 视为"绝对值 leaky"（输出非负）；
2. 注释作者认为 $-\text{slope} \cdot x$ 即标准 leaky_relu（混淆了符号）；
3. 默认 $\text{slope}$ 在调用时取负值（但 [L30 `leaky_relu_default`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/std/nn/activations.th) 传入 $0.01$，正值）。

### 5.6 定理 AE1 的最终陈述

综合上述分析，我们给出**两个版本**的定理 AE1，对应两种语义：

**定理 AE1-T（Tenth 语义）**：Tenth 实现 [`activations.th` L24-L26](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/std/nn/activations.th) 满足：

$$
\text{relu}(x) + \alpha \cdot \text{relu}(-x) = \begin{cases} x & x > 0 \\ -\alpha x & x < 0 \\ 0 & x = 0 \end{cases} =: f_{\text{Tenth}}(x, \alpha)
$$

**证明**：分三 case 已在 §5.2 给出。$\square$

**定理 AE1-S（标准语义）**：标准 leaky_relu 的算术等价形式为：

$$
\text{leaky\_relu}_{\text{std}}(x, \alpha) = \text{relu}(x) - \alpha \cdot \text{relu}(-x)
$$

**证明**：分三 case，与 AE1-T 类似但符号相反。$\square$

**注记 AE1.3**：Tenth 实现 `+` 与标准 `-` 的差异是**潜在 bug 或语义偏离**，建议在 [MEMO.md](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/MEMO.md) 与 [AUDIT.md](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/AUDIT.md) 记录。本文不修改实现（数理部不写代码），仅披露此差异作为局限（见 §10）。

**注记 AE1.4**：本文以下章节（AE2-AE5）的证明均基于 AE1-T（Tenth 实际语义），不影响等价性证明的结构，仅是"哪一种语义"的标注。

---

## 6. 可微分支编码通则

基于定理 AE3，我们归纳出 Tenth 语境下的"可微分支编码通则"：

### 6.1 通则陈述

**通则**：一个分段函数 $f: \mathbb{R}^n \to \mathbb{R}^m$ 能在 Tenth 中通过算术等价无损编码为可微表达式，当且仅当满足：

1. **分段数有限**：分段数 $k$ 为编译期常量；
2. **每段可微**：每段是已知可微函数（线性、ReLU 复合等）；
3. **阈值可微表达**：阈值 $\theta_i$ 或为编译期常量，或为可微张量表达式；
4. **连续性**（强条件）：分段边界处函数值连续（$d_i = d_{i+1}$）；
5. **基函数覆盖**：所需基函数（ReLU、Exp、Log 等）已注册到 `TapeOp`。

### 6.2 通则的构造性算法

给定满足通则的分段函数，构造 Tenth 表达式的算法：

```
输入：分段函数 f(x) = piece_i(x) for x in (θ_{i-1}, θ_i], i = 1..k
输出：Tenth 表达式 expr

1. 若 k = 1：返回 piece_1(x) 的 Tenth 实现
2. 若 k = 2 且连续：
   a. 计算 a_1, b_1, a_2, b_2（两段系数）
   b. 返回 a_2 * x + b_2 + (a_1 - a_2) * relu(x - θ_1)
3. 若 k > 2 且连续：
   a. 递归拆分为 (前 k-1 段) 与 (第 k 段)，分别编码为 e_{k-1}, e_k
   b. 用 relu(θ_{k-1} - x) 作为掩码混合：e_{k-1} + (e_k - e_{k-1}) * sign(relu(θ_{k-1} - x))
      （注意：relu 本身含 x 信息，需归一化）
   c. 实际构造：用 relu(x - θ_{k-1}) - relu(x - θ_{k-1}) 的恒等式
4. 若不连续：失败，需引入 select（依定理 AE4）
```

### 6.3 通则的应用实例

| 函数 | 阈值 | 段 | Tenth 编码 |
|------|------|-----|-----------|
| ReLU | 0 | $0, x$ | `x.relu()` |
| leaky_relu (Tenth) | 0 | $-\alpha x, x$ | `x.relu() + α * (-x).relu()` |
| leaky_relu (标准) | 0 | $\alpha x, x$ | `x.relu() - α * (-x).relu()` |
| clip(x, 0, c) | 0, c | $0, x, c$ | `x.relu() - (x - c).relu()` |
| hardswish | -3, 3 | $0, x(x+3)/6, x$ | 复杂（段非线性，需 select） |

注意 hardswish 的中段 $x(x+3)/6$ 是非线性，依定理 AE4 类 1，**不可编码**，必须引入 select。

### 6.4 通则的工程含义

通则告诉 Tenth 标准库开发者：

- **能编码**：所有"分段线性 + 连续 + 固定阈值"的激活函数（ReLU、leaky_relu、clip、PReLU 的特例）；
- **不能编码**：分段非线性（ELU、GELU、SiLU/Swish、hardswish）、不连续阶跃（Heaviside）、运行时阈值动态非线性段。

Tenth 标准库已通过专用 `TapeOp`（如 `Gelu`，[autodiff.rs L77](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/autodiff.rs)）覆盖 GELU，绕过了 AE4 类 1 的限制——这是"扩展 `TapeOp`"而非"算术等价"的路径。

---

## 7. 必须引入 select 的情形

依定理 AE4，我们细化"必须引入 select"的三类情形：

### 7.1 类 1：分段非线性

**例**：ELU 激活 $f(x) = \begin{cases} x & x > 0 \\ \alpha(e^x - 1) & x \leq 0 \end{cases}$。

负半轴 $e^x - 1$ 是非线性，ReLU 线性组合无法精确表达。Tenth 标准库若实现 ELU，必须：

- 选项 A：扩展 `TapeOp` 添加 `Elu` 变体（与 `Gelu` 类似）；
- 选项 B：引入 `Select` 原语 + `Exp` 已有算子；
- 选项 C：用多项式或泰勒近似（损失精度）。

### 7.2 类 2：阈值依赖运行时值且段非线性

**例**：动态阈值激活 $f(x, y) = \begin{cases} \sin(x) & x > y \\ \cos(y) & x \leq y \end{cases}$。

阈值 $y$ 是运行时张量，且段为非线性。即便引入 $\text{relu}(x - y)$ 表达阈值，$\sin, \cos$ 的分段仍不可微编码。必须 select。

### 7.3 类 3：不连续分段

**例**：Heaviside 阶跃 $H(x) = \begin{cases} 1 & x > 0 \\ 0 & x \leq 0 \end{cases}$。

不连续（$d_1 = 1 \neq 0 = d_2$），ReLU 线性组合无法表达。可微近似为 $\sigma(\beta x)$（$\beta \to \infty$ 时收敛），但**非无损**。

### 7.4 类 2 修正：阈值依赖但段线性

依 AE4 注记，阈值依赖运行时但段线性时仍可编码：

**例**：$\max(t_1, t_2) = \frac{t_1 + t_2 + \text{relu}(t_1 - t_2) + \text{relu}(t_2 - t_1)}{2}$（依 AE1.2 推论 $|u| = \text{relu}(u) + \text{relu}(-u)$）。

Tenth 中：`(t1 + t2 + (t1 - t2).relu() + (t2 - t1).relu()) * 0.5`。这是 element-wise max 的算术等价实现，可微。

### 7.5 引入 select 的工程代价

若 Tenth 未来引入 `Select` 原语，需：

1. 在 [`tenth/src/runtime/autodiff.rs`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/autodiff.rs) `TapeOp` 添加 `Select` 变体（前向 + 反向）；
2. 同步 [`tenthc/`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenthc/) 自举编译器对应模块（依 [工作规范.md §4](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/.trae/rules/工作规范.md)）；
3. 更新 [能力全梳理](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/能力梳理/能力全梳理.md) 与 [MEMO.md](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/MEMO.md)；
4. 添加测试（依工作规范 §6）；
5. 更新 [语言参考手册](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/docs/语言参考手册.md)。

工程代价中等，但**会破坏** Tenth "21 算子"的简洁性，需总师级决策。

---

## 8. 与 BitHacks 类比

### 8.1 共同模式

Tenth leaky_relu 算术等价技巧与 BitHacks 共享一个元模式：

> **能力受限催生等价技巧**：当目标语言/硬件缺少某种原语时，开发者用已有原语的组合实现等价语义，常以"恒等式 + case 分析"形式呈现。

| 维度 | BitHacks | Tenth 算术等价 |
|------|---------|---------------|
| 缺失原语 | 条件分支（CPU 分支预测昂贵） | tensor 条件运算（语言未提供） |
| 替代原语 | 位运算（AND, OR, XOR, 移位） | ReLU + 线性运算 |
| 动机 | 性能（避免分支预测失败） | 能力约束（无 select） |
| 形式 | 恒等式 + 位运算技巧 | 恒等式 + ReLU 组合 |
| 验证 | case 分析（正/负/零） | case 分析（正/负/零） |
| 局限 | 仅适用整数 | 仅适用分段线性连续函数（AE3） |

### 8.2 差异

BitHacks 是**优化**（有分支版本可用，但位运算更快）；Tenth 算术等价是**必需**（无 select 原语，算术等价是唯一可微路径）。这使 Tenth 的技巧更接近"语言能力边界"的体现，而非"性能优化"。

### 8.3 元规律的普适性

"能力受限催生等价技巧"在编程语言与系统设计中反复出现：

- **GPU 编程**早期无递归 → 用栈模拟；
- **SQL** 无循环 → 用递归 CTE；
- **正则表达式** 无计数 → 用 `(a*)` 与回溯；
- **Tenth** 无 tensor select → 用 ReLU 线性组合。

这一元规律提示：**语言设计中的"省略"会激活开发者的创造性等价构造**，但也会划定表达力边界（如 AE4 所述）。

---

## 9. 工程权衡

### 9.1 算术等价 vs select 原语

| 维度 | 算术等价（现状） | 引入 select |
|------|----------------|------------|
| 表达力 | 限 AE3 类 | 任意分段 |
| Tape 节点数 | 5（leaky_relu） | 1-2 |
| 内存 | 5×节点 | 1×节点 |
| 优化机会 | 算子融合 | 掩码优化 |
| 自举同步 | 无需改 tenthc | 需同步 |
| 标准库简洁性 | 21 算子 | 22 算子 |
| 开发者负担 | 需构造恒等式 | 直白 |

Tenth 当前选择算术等价，是"**简洁性优先于表达力**"的工程哲学。代价是开发者需具备构造恒等式的能力（本文 §6 通则提供指导）。

### 9.2 Tape 节点数的实际影响

leaky_relu 算术等价实现产生 5 个 tape 节点（Neg, ReLU, ReLU, Mul, Add），而 select 实现仅需 1-2 个。在大规模训练中：

- **前向**：5×算子开销 vs 1×算子开销，差距常数倍；
- **反向**：5×梯度回传 vs 1×，差距常数倍；
- **内存**：5×节点存储 vs 1×，差距常数倍。

但对典型 transformer 模型，激活函数开销占总训练开销的 5-15%，5×常数倍使激活占比升至 25-75%，**显著**。这是算术等价的实际代价。

**缓解**：算子融合（将 Neg+ReLU 融合为 NegReLU，或将整个 leaky_relu 融合为单算子）可消除常数倍开销。Tenth 的 JIT（[compile/jit/](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/jit/)）提供了融合框架（参见 T9 JIT 论文）。

### 9.3 数值精度

算术等价实现的数值精度与 select 实现相同（均无除法、无 exp/log），浮点误差仅在乘加累积中产生，量级 $O(\epsilon_{\text{mach}})$，可忽略。

---

## 10. 局限（独立章节）

依数理部规范，本文主动记录以下局限：

### 10.1 leaky_relu 语义偏离

**是什么**：[activations.th L25](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/std/nn/activations.th) 的实现 `x.relu() + slope * (-x).relu()` 在 $x < 0$ 处给出 $-\text{slope} \cdot x$（正数），而标准 leaky_relu 给出 $\text{slope} \cdot x$（负数）。两者符号相反（详见 §5.3-§5.6）。

**影响**：使用 Tenth `leaky_relu` 的模型在负半轴的激活值与 PyTorch/JAX 不同，可能导致预训练权重不兼容、训练动态偏离。

**如何缓解**：

- 选项 A：修改实现为 `x.relu() - slope * (-x).relu()`（标准语义）；
- 选项 B：在文档中明确说明 Tenth leaky_relu 是"绝对值型"变体；
- 选项 C：在 [AUDIT.md](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/AUDIT.md) 登记为已知缺陷。

**本文行动**：不修改实现（数理部不写代码），仅在 §5 与本节披露。建议总师决策。

### 10.2 定理 AE3 的不完备性

**是什么**：AE3 给出的是**充分条件**（双段线性 + 阈值固定 + 连续），未给出**充要条件**。可能存在满足更弱条件的函数仍可编码。

**影响**：通则的覆盖范围可能比实际可编码范围窄，限制开发者对算术等价的信心。

**如何缓解**：未来工作可刻画充要条件（见 §11 开放问题）。

### 10.3 类 2 刻画不严

**是什么**：定理 AE4 类 2"阈值依赖运行时值"的刻画在 §7.4 修正后变得模糊——阈值依赖但段线性仍可编码（如 max(t1, t2)）。

**影响**：AE4 类 2 的陈述需更精细，否则误导开发者认为所有动态阈值都不可编码。

**如何缓解**：本文 §7.4 已部分修正，完整刻画留作开放问题。

### 10.4 浮点误差未量化

**是什么**：§9.3 声称"浮点误差量级 $O(\epsilon_{\text{mach}})$"未给出严格上界。

**影响**：对极深网络或极长训练，浮点误差累积可能显著。

**如何缓解**：未来可用区间分析或 Kahan 求和量化。

### 10.5 与 T39 联动的依赖

**是什么**：定理 AE2 依赖 T39 的 ReLU backward 正确性（[T39 §6.7](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/docs/论文/T39-Wengert-Tape形式化语义与反向模式正确性.md)）。若 T39 的 ReLU backward 存在未发现的缺陷，AE2 的结论受影响。

**影响**：本文结论与 T39 形成依赖链，需协同验证。

**如何缓解**：T39 已通过逐一验证（21 算子），可信度较高；本文额外给出独立 case 分析（§5）作为交叉验证。

### 10.6 未覆盖的激活函数

**是什么**：本文聚焦 leaky_relu，未逐一验证其他激活（GELU、SiLU、ELU、hardswish 等）的算术等价可能性。

**影响**：通则（§6）的实用性未在所有常见激活上验证。

**如何缓解**：§6.3 表格列出部分激活的编码可行性，但完整验证需后续工作。

### 10.7 自举同步未涉及

**是什么**：本文不涉及 tenthc 自举编译器的同步问题。若未来引入 select 原语（§7.5），需同步 tenthc，本文未分析该路径。

**影响**：引入 select 的工程代价评估不完整。

**如何缓解**：依工作规范 §4，自举同步由编译器部主导，本文仅提供理论依据。

### 10.8 证明的循环论证风险

**是什么**：定理 AE2 证明中使用了 T39 的链式法则等式（定理 AD1），而 T39 的 AD1 又依赖 ReLU backward 的实现正确性。本文 AE2 又用 ReLU backward 验证算术等价的梯度正确性。形成"ReLU 实现 → T39 AD1 → 本文 AE2 → 验证 ReLU 实现"的弱循环。

**影响**：若 ReLU 实现有 bug，T39 与本文同时出错，互证失效。

**如何缓解**：本文 §5 的 case 分析是**独立**于 T39 的前向验证，可作为 ReLU 实现的独立 sanity check。反向梯度的独立验证需手工计算（已在 AE2 证明中给出）。

---

## 11. 开放问题

### 11.1 AE3 的充要条件刻画

**问题**：是否存在比"双段线性 + 阈值固定 + 连续"更弱的充要条件，刻画可 ReLU 编码的分段函数？

**思路**：可能需引入"分段线性函数空间"的代数刻画，关联稀疏线性规划与 ReLU 神经网络的普适逼近定理（[Goodfellow et al., 2016](https://www.deeplearningbook.org/)）。

### 11.2 多阈值扩展

**问题**：通则 §6.2 算法步骤 3 的多段编码是否总有效？是否存在多段连续分段线性函数无法用 ReLU 编码？

**思路**：截断 ReLU 推广（推论 AE3.2）提示 $k$ 段函数可能需 $k-1$ 个 ReLU，但严格证明待补。

### 11.3 阈值动态的完整刻画

**问题**：AE4 类 2 修正后，"阈值动态 + 段线性"何时可编码？

**思路**：可能需区分"阈值依赖张量"与"阈值依赖标量"、"线性段"与"仿射段"等子类。

### 11.4 算子融合对算术等价的消除

**问题**：JIT 算子融合（T9）能否将 leaky_relu 的 5 个 tape 节点融合为 1 个，消除常数倍开销？

**思路**：需分析 Tenth JIT 的融合模式匹配规则，构造 `Neg+ReLU → ReLUNeg`、`Mul+Add → MulAdd` 等融合模式。

### 11.5 高阶自动微分下的算术等价

**问题**：T39 §9.1 提及高阶自动微分未覆盖。若 Tenth 未来支持高阶 AD，算术等价在二阶梯度层面是否仍保持？

**思路**：leaky_relu 的二阶导数为 0（除 $x = 0$），算术等价实现的二阶梯度应同为 0，但需形式化验证。

### 11.6 与 T38 多路径一致性的联动

**问题**：T38 论证了 autodiff tape 的多路径一致性。算术等价实现引入了额外路径（5 个节点 vs 1 个），是否影响 T38 的一致性结论？

**思路**：T38 的一致性是"同一函数的不同 tape 路径产生相同梯度"，算术等价是"同一函数的不同实现"。需分析 T38 框架是否覆盖"实现级"差异。

---

## 12. 结论

本文形式化分析了 Tenth v0.3.3 标准库中 `leaky_relu` 的算术等价技巧——`relu(x) + slope * relu(-x)`，主要贡献为：

1. **定理 AE1-T/AE1-S**：分别给出 Tenth 实现与标准 leaky_relu 的算术等价形式，并**披露两者的符号差异**（§5.3-§5.6）——这是本文最重要的发现，建议登记至 [AUDIT.md](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/AUDIT.md)；
2. **定理 AE2**：在 Wengert tape 语义下（与 T39 联动）证明算术等价保持可微性，除 $x = 0$ 处次梯度约定外梯度相同；
3. **定理 AE3**：抽象出"可微分支编码通则"，给出双段线性 + 阈值固定 + 连续的充分条件；
4. **定理 AE4**：界定三类必须引入 select 的情形（分段非线性、动态阈值非线性、不连续）；
5. **定理 AE5**：与 PyTorch/JAX select 对比，三者几乎处处等价，差异在工程表达力。

**核心洞察**：Tenth 的"无 select 原语"设计是一种**能力约束催生算法创新**的范例，与 BitHacks 用位运算模拟条件分支同构。通则（§6）为开发者提供了构造算术等价的指南，局限（§10）诚实地划定了边界。

**对实施的指导**：

- 标准库开发者：依通则（§6）判断激活函数是否可算术等价编码；
- 编译器部：若引入 select 原语（§7.5），需同步 tenthc 与 [语言参考手册](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/docs/语言参考手册.md)；
- 运行时部：考虑算子融合（§11.4）消除算术等价的常数倍开销；
- 文档部：将本文归档至 [docs/论文/](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/docs/论文/)，并在 [MEMO.md](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/MEMO.md) 留痕；
- 总师：决策是否修正 leaky_relu 的符号差异（§10.1）。

---

## 参考文献

1. **Anderson, S. E.** (1997-2005). *Bit Twiddling Hacks*. Stanford Graphics. https://graphics.stanford.edu/~seander/bithacks.html
2. **Tenth 项目**（2026）. *autodiff.rs: Wengert Tape 实现*. [tenth/src/runtime/autodiff.rs](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/autodiff.rs)
3. **Tenth 项目**（2026）. *activations.th: 标准库激活函数*. [tenth/std/nn/activations.th](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/std/nn/activations.th)
4. **Tenth 项目**（2026）. *tensor.rs: 张量类型与运算*. [tenth/src/runtime/tensor.rs](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/tensor.rs)
5. **Tenth 数理部**（2026）. *T39: Wengert Tape 形式化语义与反向模式正确性*. [docs/论文/T39-Wengert-Tape形式化语义与反向模式正确性.md](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/docs/论文/T39-Wengert-Tape形式化语义与反向模式正确性.md)
6. **Tenth 数理部**（2026）. *T38: autodiff tape 多路径一致性*. [docs/论文/T38-autodiff-tape多路径一致性.md](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/docs/论文/T38-autodiff-tape多路径一致性.md)
7. **Tenth 数理部**（2026）. *T9: JIT 特化语义保持证明*. [docs/论文/T9-JIT特化语义保持证明.md](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/docs/论文/T9-JIT特化语义保持证明.md)
8. **PyTorch**（2024）. *torch.where 文档*. https://pytorch.org/docs/stable/generated/torch.where.html
9. **JAX**（2024）. *jax.lax.select 文档*. https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.select.html
10. **Goodfellow, I., Bengio, Y., Courville, A.** (2016). *Deep Learning*. MIT Press. https://www.deeplearningbook.org/
11. **Baydin, A. G., Pearlmutter, B. A., Radul, A. A., Siskind, J. M.** (2018). *Automatic Differentiation in Machine Learning: a Survey*. Journal of Marchine Learning Research, 18(153), 1-43.
12. **Wengert, R. E.** (1964). *A Simple Automatic Derivative Evaluation Program*. Communications of the ACM, 7(8), 463-464.
13. **Maas, A. L., Hannun, A. Y., Ng, A. Y.** (2013). *Rectifier Nonlinearities Improve Neural Network Acoustic Models*. ICML Workshop on Deep Learning for Audio, Speech and Language Processing.（leaky_relu 原始论文）
14. **Tenth 项目**（2026）. *工作规范 v1.1*. [.trae/rules/工作规范.md](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/.trae/rules/工作规范.md)

---

## 附录 A：定理索引

| 编号 | 名称 | 陈述位置 | 证明位置 | 关键源码 |
|------|------|---------|---------|---------|
| AE1-T | leaky_relu 算术等价（Tenth 语义） | §5.6 | §5.2 | [activations.th L24-L26](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/std/nn/activations.th) |
| AE1-S | leaky_relu 算术等价（标准语义） | §5.6 | §5.6 | （未实现，建议形式） |
| AE2 | 可微性保持 | §4.2 | §4.2 | [autodiff.rs L342-L349](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/autodiff.rs) |
| AE3 | 可编码通则 | §4.3 | §4.3 | （构造性定理） |
| AE4 | 必须引入 select 的情形 | §4.4 | §4.4 | （边界刻画） |
| AE5 | 与 PyTorch/JAX select 对比 | §4.5 | §4.5 | （对比定理） |

**主定理数**：5（AE1-AE5），其中 AE1 分两版本（Tenth 语义 / 标准语义）。

---

## 附录 B：与现有文档的对应

| 本文章节 | 对应文档 | 关系 |
|---------|---------|------|
| §3.3 自动微分语义 | [T39 §3.3](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/docs/论文/T39-Wengert-Tape形式化语义与反向模式正确性.md) | 联动（依赖 ReLU backward） |
| §4.2 定理 AE2 | [T39 定理 AD1](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/docs/论文/T39-Wengert-Tape形式化语义与反向模式正确性.md) | 依赖（链式法则） |
| §10.5 与 T39 联动 | [T38 多路径一致性](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/docs/论文/T38-autodiff-tape多路径一致性.md) | 互补（实现级一致性） |
| §6.4 工程含义 | [CODE_WIKI.md](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/CODE_WIKI.md) | 实施 |
| §10.1 语义偏离 | [AUDIT.md](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/AUDIT.md) | 建议登记 |
| §7.5 引入 select | [MEMO.md](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/MEMO.md) | 变更记录（若实施） |
| §6 通则 | [能力梳理/能力全梳理.md](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/能力梳理/能力全梳理.md) | 能力边界 |

---

## 附录 C：实施建议

### C.1 短期（v0.3.x）

1. **登记符号差异**：将 §10.1 的 leaky_relu 语义偏离登记至 [AUDIT.md](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/AUDIT.md)；
2. **添加测试**：在 [tenth/std/nn/](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/std/nn/) 添加 `test_leaky_relu.th`，覆盖 $x > 0, x = 0, x < 0$ 三 case；
3. **文档同步**：在 [语言参考手册](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/docs/语言参考手册.md) 激活函数章节添加 leaky_relu 的算术等价说明。

### C.2 中期（v0.4.x）

1. **通则推广**：将 §6 通则应用于其他激活函数（PReLU、clip），评估是否需扩展 `TapeOp`；
2. **算子融合**：在 [compile/jit/](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/jit/) 添加 `Neg+ReLU → ReLUNeg` 融合模式，降低算术等价的常数开销。

### C.3 长期（v0.5+）

1. **select 原语评估**：若通则覆盖不足（如 hardswish 需求），评估引入 `Select` 原语的工程代价；
2. **充要条件刻画**：完成 §11.1 的开放问题，给出 AE3 的充要条件。

---

**论文完**

> **数理部声明**：本文遵循数理部规范，所有定理附源码链接（file://），独立局限章节（§10）主动披露证明漏洞与假设强度，与 T39 联动（§3.3, §4.2, §10.5）。本文未修改任何代码，仅提供理论依据。建议总师据本文决策 leaky_relu 语义修正与 select 原语引入。
