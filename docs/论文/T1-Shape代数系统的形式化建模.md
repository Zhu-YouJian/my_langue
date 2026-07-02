# Shape 代数系统的形式化建模：基于 Tenth 语言三值维度类型的代数性质分析

> **作者**：Tenth 项目数理部
> **日期**：2026-07-02
> **类型**：理论分析论文（T1 理论点）
> **实证基础**：Tenth v0.3.3+ 源码（`hir/types.rs`、`hir/lower/types.rs`、`runtime/tensor.rs`、`runtime/autodiff.rs`）
> **关联文档**：`docs/shape-check-roadmap/形式化分析理论可行性论证.md`、`docs/语言参考手册.md`
> **版本**：v1（首轮分析，含 4 轮自审修正留痕）

---

## 摘要

本文对 Tenth 语言的维度类型系统进行形式化建模与代数性质分析。Tenth 的维度类型 `Dim` 不是经典整数,而是三值域 $\mathrm{Known}(n) \mid \mathrm{Symbol}(s) \mid \mathrm{Any}$ 配以 NumPy 风格广播规则构成的偏函数代数。我们将编译期 `broadcast_shapes` 与运行时 `broadcast_shape` 抽象为偏二元运算 $\oplus$,证明:(1) 运行时具体维度上的 $\oplus$ 构成**部分交换幂等么半群**(定理 1);(2) 编译期 $\mathrm{Known} \cup \{\mathrm{Any}\}$ 片段构成**有界半格**,$\oplus$ 即最小上界(定理 2);(3) 引入 $\mathrm{Symbol}$ 后 $\oplus$ **不满足结合律**(定理 3,给出反例),根源在于 Symbol-Known 保守混合破坏了信息单调性;(4) 编译期检查对全 $\mathrm{Known}$ 输入是**可靠且完备**的(定理 4);(5) 反向传播 `unbroadcast` 是前向 `broadcast` 线性算子的**伴随**(adjoint,定理 5);(6) 编译期是运行时的**保守过近似**(定理 6)。本文诚实记录 6 处理论局限,包括结合律破坏、负维度未校验、跨函数 shape 求解的非代数性等,为后续 Symbol unification(Phase 2/3)的设计提供形式化依据。

**关键词**：维度类型、广播代数、半格、偏函数、伴随算子、自动微分、形式化建模、Tenth 语言

---

## 1. 引言

### 1.1 动机

AI 原生编程语言的核心张力在于:**张量计算要求编译期 shape 推断以提供早期错误诊断,而真实程序中 shape 往往依赖运行时值(用户输入、数据集大小、动态 batch)**。经典整数维度类型(如 C++ 模板的 `std::integral_constant`)要求所有维度编译期已知,表达力不足;完全动态 shape(如 Python/NumPy)放弃编译期检查,错误延迟到运行时。

Tenth 语言采用折中方案——**三值维度域** $\mathrm{Dim}$,允许同一程序中混合静态已知维度、符号维度(命名但未赋值)、完全未知维度。这一设计自然引出一个理论问题:**这三类维度上的广播运算构成何种代数结构?其性质是否足以支撑可靠的编译期检查?**

该问题的答案直接关系 Tenth 的护城河能力:护城河 A(Autograd 反向 Shape 静态验证)、护城河 B(Shape 代数求解器)、护城河 D(编译期内存/算力预估)均依赖 `broadcast_shapes` 的代数性质。若运算不满足结合律,则跨多步算子的 shape 传播可能出现"顺序依赖"的虚假报错或漏报。

### 1.2 研究问题

本文回答以下四个研究问题:

- **RQ1**:Tenth 的广播运算 $\oplus$ 在单维度上构成何种代数结构?是否构成幺半群、格或范畴?
- **RQ2**:多维度广播(右对齐规则)的代数性质如何?是否保持单维度性质?
- **RQ3**:编译期 `broadcast_shapes` 与运行时 `broadcast_shape` 的语义关系是什么?编译期检查的可靠性与完备性如何?
- **RQ4**:反向传播中的 `unbroadcast` 与前向 `broadcast` 的代数/对偶关系是什么?

### 1.3 贡献

- **形式化模型**(§3):将 `Dim` 三值域与广播运算抽象为偏代数结构,定义子sumption 偏序 $\sqsubseteq$。
- **代数性质分析**(§4):证明 6 个主定理,覆盖运行时与编译期两个层级,并给出结合律失败的构造性反例(定理 3)。
- **可靠性论证**(§4.4):证明编译期检查对全 $\mathrm{Known}$ 输入的可靠性与完备性。
- **对偶性论证**(§4.5):证明 `unbroadcast` 是 `broadcast` 的线性伴随,梯度语义正确。
- **诚实局限**(§7):独立章节记录 6 处理论局限,包括结合律破坏的影响范围、负维度未校验的工程 gap、跨函数求解的非代数性。

### 1.4 v1 自审留痕

本文经历 4 轮自审,主要修正:

| 轮次 | 原始断言 | 修正 |
|------|---------|------|
| 第 1 轮(结构) | 声称 $\oplus$ 构成"幺半群" | 修正:仅在 $\mathrm{Known} \cup \{\mathrm{Any}\}$ 片段成立;含 Symbol 时不成立(定理 3) |
| 第 2 轮(证明) | 定理 2 初稿未证 $\sqsubseteq$ 传递性 | 补充传递性证明(引理 2.3) |
| 第 3 轮(边界) | 未处理空 shape(标量)情形 | 补充标量作为单位元的讨论(定义 3.6) |
| 第 4 轮(诚实) | 定理 4 初稿声称"完备" | 修正:仅对全 $\mathrm{Known}$ 输入完备;含 Symbol/Any 时仅可靠不完备(定理 4 推论 4.2) |

---

## 2. 背景与相关工作

### 2.1 NumPy 广播规则

NumPy 的广播规则是事实标准,定义于 [NumPy 文档](https://numpy.org/doc/stable/user/basics.broadcasting.html):两个 shape 从右向左对齐,每对维度 $(d_a, d_b)$ 兼容当且仅当 $d_a = d_b$ 或 $d_a = 1$ 或 $d_b = 1$,结果维度为 $\max(d_a, d_b)$;剩余维度直接附加。该运算在具体非负整数上构成**部分交换幂等么半群**(以 1 为单位元),但 NumPy 本身未将 shape 暴露为类型——shape 检查纯运行时。

### 2.2 JAX 的 shape 处理

JAX 将 shape 作为抽象值(`jax.core.ShapedArray`),shape 必须完全静态(ConcreteShape)或多维部分静态。JAX 的 `broadcast_shapes` 与 NumPy 语义一致,但引入 `Poly` 维度(多项式符号维度)用于 shape 计算,支持线性组合的 shape 推断(如 `jax.numpy.concatenate` 的拼接维相加)。JAX 的 Poly 维度**支持线性 unification**(同类项可合并),而 Tenth 的 `Symbol` **不做 unification**,这是关键差异。

### 2.3 Apache TVM 的 Tensor Expression

TVM 的 `te.Tensor` shape 必须编译期已知(整数或 `tvm.tir.Var` 符号),支持符号约束的隐式求解(基于 Z3)。TVM 的 shape 系统是**约束求解式**的——所有符号维度间约束被收集并交由 SMT 求解,表达力强但计算代价高。Tenth 选择了更轻量的"保守兼容"路径:不做求解,仅做局部广播检查。

### 2.4 与 Tenth 的定位差异

| 系统 | shape 表示 | 检查时机 | 符号处理 | 复杂度 |
|------|-----------|---------|---------|--------|
| NumPy | 运行时 `usize` | 运行时 | 无 | $O(n)$ |
| JAX | ConcreteShape / Poly | 追踪期 | 线性 unification | $O(n)$/求解 |
| TVM | 整数 / `tir.Var` | 编译期 | SMT 求解 | NP-hard(最坏) |
| **Tenth** | `Dim` 三值域 | **编译期 + 运行时** | **保守兼容,无 unification** | **$O(n)$** |

Tenth 的设计介于 NumPy(纯运行时)与 JAX/TVM(约束求解)之间:编译期做 $O(n)$ 的偏函数广播检查,**不引入求解器复杂度**,但代价是 Symbol 表达力的牺牲(见 §6)。

---

## 3. 形式化建模

### 3.1 前置定义:Shape 与维度

**定义 3.1(维度值域)**。Tenth 的维度类型 `Dim` 定义三值域:

$$\mathbb{D} = \{\,\mathrm{Known}(n) \mid n \in \mathbb{I}_{64}\,\} \;\cup\; \{\,\mathrm{Symbol}(s) \mid s \in \mathrm{String}\,\} \;\cup\; \{\mathrm{Any}\}$$

其中 $\mathbb{I}_{64} = [-2^{63}, 2^{63}-1] \cap \mathbb{Z}$ 为 `i64` 的值域。

**实现对应**:[`tenth/src/hir/types.rs:13-17`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/hir/types.rs) `enum Dim { Known(i64), Symbol(String), Any }`。

**注意(局限 L1)**:$\mathbb{I}_{64}$ 允许负数与零,但语义上维度应为非负(`usize`)。`static_numel` 对 `Known(n) < 0` 返回 `None`([types.rs:128](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/hir/types.rs)),但 `broadcast_shapes` 不校验维度非负。详见 §7。

**定义 3.2(运行时具体维度)**。运行时维度域 $\mathbb{D}_{\mathrm{rt}} = \mathbb{N}_0 = \{0, 1, 2, \ldots\}$(`usize`),无 Symbol/Any。存在嵌入 $\iota: \{\mathrm{Known}(n) \mid n \geq 0\} \hookrightarrow \mathbb{D}_{\mathrm{rt}}$, $\iota(\mathrm{Known}(n)) = n$。

**实现对应**:[`tenth/src/runtime/tensor.rs:552`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/tensor.rs) `fn broadcast_shape(a: &[usize], b: &[usize]) -> Option<Vec<usize>>`。

**定义 3.3(Shape)**。Shape 是维度序列 $s = (d_1, \ldots, d_k)$,$k \geq 0$,$d_i \in \mathbb{D}$(编译期)或 $d_i \in \mathbb{D}_{\mathrm{rt}}$(运行时)。空 shape $\epsilon = ()$ 表示标量。记编译期 shape 集合 $\mathbb{S} = \bigcup_{k \geq 0} \mathbb{D}^k$,运行时 shape 集合 $\mathbb{S}_{\mathrm{rt}} = \bigcup_{k \geq 0} \mathbb{D}_{\mathrm{rt}}^k$。

### 3.2 单维度广播运算

**定义 3.4(单维度广播 $\oplus$)**。偏函数 $\oplus: \mathbb{D} \times \mathbb{D} \rightharpoonup \mathbb{D}$ 定义为(对应 [`hir/lower/types.rs:23-30`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/hir/lower/types.rs)):

$$
\begin{aligned}
\mathrm{Any} \oplus x &= x \oplus \mathrm{Any} = \mathrm{Any} && \text{(吸收律)} \\
\mathrm{Known}(1) \oplus x &= x \oplus \mathrm{Known}(1) = x && \text{(单位元)} \\
\mathrm{Known}(a) \oplus \mathrm{Known}(b) &= \mathrm{Known}(a) && \text{if } a = b \geq 2 \\
\mathrm{Symbol}(s) \oplus \mathrm{Symbol}(s) &= \mathrm{Symbol}(s) && \text{(同名等价类)} \\
\mathrm{Symbol}(s) \oplus \mathrm{Known}(n) &= \mathrm{Symbol}(s) && \text{if } n \neq 1 \text{ (保守兼容)} \\
\mathrm{Known}(n) \oplus \mathrm{Symbol}(s) &= \mathrm{Symbol}(s) && \text{if } n \neq 1 \text{ (保守兼容)}
\end{aligned}
$$

其余情形(异名 Symbol、不等 Known $\geq 2$)未定义,记为 $\bot$(对应实现中的 `None`)。

**定义 3.5(运行时单维度广播 $\oplus_{\mathrm{rt}}$)**。偏函数 $\oplus_{\mathrm{rt}}: \mathbb{D}_{\mathrm{rt}} \times \mathbb{D}_{\mathrm{rt}} \rightharpoonup \mathbb{D}_{\mathrm{rt}}$:

$$
\begin{aligned}
1 \oplus_{\mathrm{rt}} x &= x \oplus_{\mathrm{rt}} 1 = x \\
a \oplus_{\mathrm{rt}} a &= a && \text{if } a \geq 2
\end{aligned}
$$

其余未定义。对应 [`runtime/tensor.rs:560-564`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/tensor.rs)。

### 3.3 多维度广播运算

**定义 3.6(多维度广播 $\mathrm{BCast}$)**。偏函数 $\mathrm{BCast}: \mathbb{S} \times \mathbb{S} \rightharpoonup \mathbb{S}$ 定义为:

对 $s = (d_1, \ldots, d_m)$, $t = (e_1, \ldots, e_n)$,令 $k = \max(m, n)$,定义左填充:

$$
\hat{d}_i = \begin{cases} \mathrm{Known}(1) & \text{if } i \leq k - m \\ d_{i - (k-m)} & \text{otherwise} \end{cases}, \quad
\hat{e}_i = \begin{cases} \mathrm{Known}(1) & \text{if } i \leq k - n \\ e_{i - (k-n)} & \text{otherwise} \end{cases}
$$

则 $\mathrm{BCast}(s, t) = (\hat{d}_1 \oplus \hat{e}_1, \ldots, \hat{d}_k \oplus \hat{e}_k)$ 当且仅当所有 $\hat{d}_i \oplus \hat{e}_i$ 有定义;否则 $\bot$。

**实现对应**:[`hir/lower/types.rs:18-41`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/hir/lower/types.rs) 与 [`runtime/tensor.rs:552-567`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/tensor.rs)。

**注 3.1(标量作为单位元)**。$m = 0$ 时(标量 $s = \epsilon$),所有 $\hat{d}_i = \mathrm{Known}(1)$,由单位元律 $\mathrm{Known}(1) \oplus x = x$,$\mathrm{BCast}(\epsilon, t) = t$。故**空 shape(标量)是多维度广播的左单位元**;由交换性(定理 1),亦为右单位元。

### 3.4 Subsumption 偏序

为分析 $\oplus$ 是否为最小上界运算,定义 subsumption 偏序:

**定义 3.7(Subsumption $\sqsubseteq$)**。$\mathbb{D}$ 上的二元关系 $\sqsubseteq$:

$$
\begin{aligned}
\mathrm{Known}(1) &\sqsubseteq x && \forall x \in \mathbb{D} \quad \text{(1 为底)} \\
x &\sqsubseteq x && \text{(自反)} \\
x &\sqsubseteq \mathrm{Any} && \forall x \in \mathbb{D} \quad \text{(Any 为顶)} \\
\mathrm{Known}(n) &\sqsubseteq \mathrm{Symbol}(s) && \forall n \geq 2, s \quad \text{(保守:Known 被 Symbol 包含)}
\end{aligned}
$$

其余二元组不可比较(尤其 $\mathrm{Known}(m) \not\sqsubseteq \mathrm{Known}(n)$,$m \neq n$,$m,n \geq 2$;$\mathrm{Symbol}(s) \not\sqsubseteq \mathrm{Symbol}(t)$,$s \neq t$)。

**引理 3.1($\sqsubseteq$ 自反性)**。由定义直接给出。$\square$

**引理 3.2($\sqsubseteq$ 反对称性)**。若 $x \sqsubseteq y$ 且 $y \sqsubseteq x$,则 $x = y$。

**证明**。分情形:
- 若 $x = \mathrm{Known}(1)$,则由 $y \sqsubseteq x$ 与"$x \sqsubseteq \mathrm{Any}$ 仅当 $x = \mathrm{Any}$ 不成立"——实际上 $\mathrm{Known}(1) \sqsubseteq y$ 对所有 $y$ 成立,但 $y \sqsubseteq \mathrm{Known}(1)$ 仅当 $y = \mathrm{Known}(1)$(因 $\mathrm{Known}(1)$ 是底,无其他元素能 $\sqsubseteq$ 它除非是其本身)。故 $y = \mathrm{Known}(1) = x$。
- 若 $x = \mathrm{Any}$,则 $x \sqsubseteq y$ 仅当 $y = \mathrm{Any}$(Any 为顶)。故 $y = \mathrm{Any} = x$。
- 若 $x = \mathrm{Known}(n)$, $n \geq 2$:由 $x \sqsubseteq y$,$y \in \{\mathrm{Known}(n), \mathrm{Symbol}(s), \mathrm{Any}\}$。若 $y = \mathrm{Known}(n)$ 则 $x = y$。若 $y = \mathrm{Symbol}(s)$,则 $y \sqsubseteq x$ 要求 $\mathrm{Symbol}(s) \sqsubseteq \mathrm{Known}(n)$,但定义中此关系不成立——矛盾。若 $y = \mathrm{Any}$,$y \sqsubseteq x$ 要求 $\mathrm{Any} \sqsubseteq \mathrm{Known}(n)$,不成立——矛盾。
- 若 $x = \mathrm{Symbol}(s)$:类似分析,$y \in \{\mathrm{Symbol}(s), \mathrm{Any}\}$。$y = \mathrm{Symbol}(s)$ 则 $x = y$;$y = \mathrm{Any}$ 则 $y \sqsubseteq x$ 不成立。$\square$

**引理 3.3($\sqsubseteq$ 传递性)**。若 $x \sqsubseteq y$ 且 $y \sqsubseteq z$,则 $x \sqsubseteq z$。

**证明**。分情形:
- $x = \mathrm{Known}(1)$:由定义 $\mathrm{Known}(1) \sqsubseteq z$ 对所有 $z$。$\checkmark$
- $x = \mathrm{Known}(n)$, $n \geq 2$:由 $x \sqsubseteq y$,$y \in \{\mathrm{Known}(n), \mathrm{Symbol}(s), \mathrm{Any}\}$。
  - $y = \mathrm{Known}(n)$:$y \sqsubseteq z$ 即 $x \sqsubseteq z$。$\checkmark$
  - $y = \mathrm{Symbol}(s)$:$y \sqsubseteq z$ 要求 $z \in \{\mathrm{Symbol}(s), \mathrm{Any}\}$(因 Symbol 仅自反或被 Any 包含)。若 $z = \mathrm{Symbol}(s)$,由定义 $\mathrm{Known}(n) \sqsubseteq \mathrm{Symbol}(s) = z$。$\checkmark$ 若 $z = \mathrm{Any}$,由 $x \sqsubseteq \mathrm{Any}$。$\checkmark$
  - $y = \mathrm{Any}$:$y \sqsubseteq z$ 要求 $z = \mathrm{Any}$(Any 为顶,仅自反)。则 $x \sqsubseteq \mathrm{Any} = z$。$\checkmark$
- $x = \mathrm{Symbol}(s)$:$y \in \{\mathrm{Symbol}(s), \mathrm{Any}\}$。
  - $y = \mathrm{Symbol}(s)$:$y \sqsubseteq z$ 同上分析。$\checkmark$
  - $y = \mathrm{Any}$:$z = \mathrm{Any}$。$\checkmark$
- $x = \mathrm{Any}$:$y = \mathrm{Any}$,$z = \mathrm{Any}$。$\checkmark$ $\square$

由引理 3.1–3.3,$(\mathbb{D}, \sqsubseteq)$ 是**偏序集**(poset),底为 $\mathrm{Known}(1)$,顶为 $\mathrm{Any}$。

---

## 4. 代数性质分析

### 4.1 运行时代数结构(定理 1)

**定理 1(运行时单维度广播构成部分交换幂等么半群)**。$(\mathbb{D}_{\mathrm{rt}}, \oplus_{\mathrm{rt}}, 1)$ 构成部分交换幂等么半群(partial commutative idempotent monoid),即:
- (a) **单位元**:$1 \oplus_{\mathrm{rt}} x = x \oplus_{\mathrm{rt}} 1 = x$ 对所有 $x \in \mathbb{D}_{\mathrm{rt}}$。
- (b) **交换律**:$x \oplus_{\mathrm{rt}} y = y \oplus_{\mathrm{rt}} x$(当任一侧有定义)。
- (c) **幂等律**:$x \oplus_{\mathrm{rt}} x = x$ 对所有 $x$。
- (d) **结合律**:$(x \oplus_{\mathrm{rt}} y) \oplus_{\mathrm{rt}} z = x \oplus_{\mathrm{rt}} (y \oplus_{\mathrm{rt}} z)$(当两侧均有定义时相等;若一侧有定义而另一侧无定义,则结论见推论 1.1)。

**证明**。
(a) 由定义 3.5 直接给出。
(b) $\oplus_{\mathrm{rt}}$ 的定义对称($1 \oplus x = x$ 与 $x \oplus 1 = x$ 同时给出,$a \oplus a = a$ 对称)。
(c) $a \oplus_{\mathrm{rt}} a = a$ 由定义直接给出。
(d) 分情形验证(设 $x, y, z \in \mathbb{D}_{\mathrm{rt}}$,记 $\bot$ 为未定义):

情形分析。若任一为 1(单位元),两侧化简后相等。设 $x, y, z \geq 2$:

- 若 $x = y = z$:$(x \oplus x) \oplus x = x \oplus x = x$;$x \oplus (x \oplus x) = x \oplus x = x$。$\checkmark$
- 若 $x = y \neq z$ 且 $x \neq 1$,$z \neq 1$:$x \oplus y = x$,而 $x \oplus z = \bot$(不等且非 1)。左侧 $= x \oplus z = \bot$;右侧 $= x \oplus \bot = \bot$。$\checkmark$
- 若 $x \neq y$ 且 $y \neq z$ 且 $x \neq z$(三者互异且非 1):$x \oplus y = \bot$,左侧 $= \bot \oplus z = \bot$;$y \oplus z = \bot$,右侧 $= x \oplus \bot = \bot$。$\checkmark$
- 若 $x = z \neq y$:$x \oplus y = \bot$,$y \oplus z = y \oplus x = \bot$。两侧均 $\bot$。$\checkmark$

所有情形验证完毕。$\square$

**推论 1.1(部分结合律的语义)**。当 $\oplus_{\mathrm{rt}}$ 一侧有定义而另一侧无定义时,对应"部分广播链中某一中间结果不兼容"的运行时失败,两侧均报告失败(返回 `None`),语义一致。

**定理 1'(运行时多维度广播)**。$\mathrm{BCast}_{\mathrm{rt}}$ 在 $\mathbb{S}_{\mathrm{rt}}$ 上满足:
- (a) **单位元**:$\mathrm{BCast}_{\mathrm{rt}}(\epsilon, s) = s$。
- (b) **交换律**:同秩 shape 交换;不同秩时,左填充 $\mathrm{Known}(1)$ 后等价于同秩情形,故交换。
- (c) **结合律**:$\mathrm{BCast}_{\mathrm{rt}}(\mathrm{BCast}_{\mathrm{rt}}(s, t), u) = \mathrm{BCast}_{\mathrm{rt}}(s, \mathrm{BCast}_{\mathrm{rt}}(t, u))$(当两侧均有定义)。

**证明思路**。多维度广播通过右对齐 + 左填充 $\mathrm{Known}(1)$ 归约为同秩逐维 $\oplus_{\mathrm{rt}}$。填充后秩为 $\max(\mathrm{rank}(s), \mathrm{rank}(t), \mathrm{rank}(u))$,逐维应用定理 1(d)。详细归纳省略(标准 NumPy 广播已知性质)。$\square$

**注 4.1(局限 L2)**。定理 1' 的结合律在运行时具体维度上成立,是 NumPy 广播的基础性质。但编译期引入 Symbol 后结合律破坏(见定理 3),这是编译期检查的固有局限。

### 4.2 编译期 Known-only 片段(定理 2)

**定理 2($\mathrm{Known} \cup \{\mathrm{Any}\}$ 片段构成有界半格)**。令 $\mathbb{D}_{\mathrm{KA}} = \{\mathrm{Known}(n) \mid n \in \mathbb{N}_0, n \geq 1\} \cup \{\mathrm{Any}\}$,则 $(\mathbb{D}_{\mathrm{KA}}, \oplus)$ 构成**有界半格**(bounded semilattice),即:
- (a) $\oplus$ 在 $\mathbb{D}_{\mathrm{KA}}$ 上**完全定义**(total,无 $\bot$)。
- (b) $\oplus$ 满足交换律、幂等律、结合律。
- (c) $\mathrm{Known}(1)$ 为底(bottom,$\forall x: \mathrm{Known}(1) \oplus x = x$)。
- (d) $\mathrm{Any}$ 为顶(top,$\forall x: x \oplus \mathrm{Any} = \mathrm{Any}$)。
- (e) $\oplus$ 即 subsumption 偏序 $\sqsubseteq$ 下的**最小上界**:对 $x, y \in \mathbb{D}_{\mathrm{KA}}$,$x \oplus y = \sup_{\sqsubseteq}\{x, y\}$。

**证明**。
(a) 对 $\mathbb{D}_{\mathrm{KA}}$ 中任意 $x, y$,验证 $\oplus$ 有定义:
- $\mathrm{Any} \oplus x = \mathrm{Any}$。$\checkmark$
- $\mathrm{Known}(1) \oplus x = x$。$\checkmark$
- $\mathrm{Known}(a) \oplus \mathrm{Known}(b)$, $a, b \geq 2$:若 $a = b$,得 $\mathrm{Known}(a)$;若 $a \neq b$,**未定义**。

**修正(局限 L3)**:严格地,$\oplus$ 在 $\mathbb{D}_{\mathrm{KA}}$ 上**不是完全定义的**——$\mathrm{Known}(2) \oplus \mathrm{Known}(3) = \bot$。故 $(\mathbb{D}_{\mathrm{KA}}, \oplus)$ 是**偏有界半格**(partial bounded semilattice),非完全半格。

(b) 交换律:定义对称。幂等律:$\mathrm{Known}(a) \oplus \mathrm{Known}(a) = \mathrm{Known}(a)$;$\mathrm{Any} \oplus \mathrm{Any} = \mathrm{Any}$;$\mathrm{Known}(1) \oplus \mathrm{Known}(1) = \mathrm{Known}(1)$。$\checkmark$

结合律:对 $\mathbb{D}_{\mathrm{KA}}$ 中三元组 $x, y, z$,若两侧均有定义,需证 $(x \oplus y) \oplus z = x \oplus (y \oplus z)$。分情形:

- 任一为 $\mathrm{Any}$:两侧均 $\mathrm{Any}$(吸收律可结合验证)。$\checkmark$
- 任一为 $\mathrm{Known}(1)$:由单位元律化简。$\checkmark$
- 全部 $\mathrm{Known}(a), \mathrm{Known}(b), \mathrm{Known}(c)$, $a, b, c \geq 2$:若 $a = b = c$,两侧 $= \mathrm{Known}(a)$;若 $a = b \neq c$,两侧 $= \bot$($\mathrm{Known}(a) \oplus \mathrm{Known}(c) = \bot$);若 $a \neq b$,左侧 $= \bot \oplus z = \bot$,右侧 $= \mathrm{Known}(a) \oplus (b \oplus c)$。若 $b = c$,$b \oplus c = \mathrm{Known}(b)$,右侧 $= a \oplus b = \bot$;若 $b \neq c$,$b \oplus c = \bot$,右侧 $= \bot$。$\checkmark$

(c) (d) 由定义直接。

(e) 需证 $\mathrm{Known}(a) \oplus \mathrm{Known}(b)$, $a, b \geq 2$, $a = b$ 时 $= \mathrm{Known}(a) = \sup\{\mathrm{Known}(a), \mathrm{Known}(b)\}$(上界为自身,因自反);$a \neq b$ 时无上界(返回 $\bot$)。

**验证 $\oplus$ 为最小上界**(当有定义时):对 $x, y \in \mathbb{D}_{\mathrm{KA}}$,$z = x \oplus y$ 有定义时需满足:
- (i) $x \sqsubseteq z$ 且 $y \sqsubseteq z$(上界性)
- (ii) 对任意 $w$ 满足 $x \sqsubseteq w$ 且 $y \sqsubseteq w$,有 $z \sqsubseteq w$(最小性)

分情形:
- $x = \mathrm{Known}(1)$:$z = y$, $x \sqsubseteq y$ ✓;最小性:$y \sqsubseteq w$ 即 $z \sqsubseteq w$ ✓。
- $x = \mathrm{Any}$:$z = \mathrm{Any}$, $x \sqsubseteq \mathrm{Any}$, $y \sqsubseteq \mathrm{Any}$ ✓;最小性:$\mathrm{Any} \sqsubseteq w$ 仅当 $w = \mathrm{Any}$ ✓。
- $x = \mathrm{Known}(a)$, $y = \mathrm{Known}(a)$, $a \geq 2$:$z = \mathrm{Known}(a)$, $x \sqsubseteq z$ 自反 ✓;最小性:若有 $w$ 使 $\mathrm{Known}(a) \sqsubseteq w$,则 $w \in \{\mathrm{Known}(a), \mathrm{Any}\}$(由 $\sqsubseteq$ 定义),$\mathrm{Known}(a) \sqsubseteq w$ ✓。
- $x = \mathrm{Known}(a)$, $y = \mathrm{Known}(b)$, $a \neq b$, $a, b \geq 2$:$z = \bot$,无上界——与"$\oplus$ 为最小上界"一致(返回 $\bot$ 表示无上界)。$\square$

**推论 2.1**。在 $\mathrm{Known} \cup \{\mathrm{Any}\}$ 片段上,$\oplus$ 与 $\sqsubseteq$ 的最小上界一致,故编译期广播检查等价于"求两 shape 在 $\sqsubseteq$ 下的最小上界,无上界则报错"。

### 4.3 引入 Symbol 后结合律失败(定理 3)

**定理 3(Symbol 破坏结合律)**。$\oplus$ 在完整 $\mathbb{D}$ 上**不满足结合律**:存在 $x, y, z \in \mathbb{D}$ 使得 $(x \oplus y) \oplus z \neq x \oplus (y \oplus z)$(一侧有定义而另一侧无定义,或两侧定义不同)。

**证明**(构造性反例)。取 $x = \mathrm{Symbol}(s)$, $y = \mathrm{Known}(2)$, $z = \mathrm{Known}(3)$。

**左侧**:$(x \oplus y) \oplus z = (\mathrm{Symbol}(s) \oplus \mathrm{Known}(2)) \oplus \mathrm{Known}(3)$。
- 由保守兼容规则,$\mathrm{Symbol}(s) \oplus \mathrm{Known}(2) = \mathrm{Symbol}(s)$。
- $\mathrm{Symbol}(s) \oplus \mathrm{Known}(3) = \mathrm{Symbol}(s)$(再次保守兼容)。
- 左侧 $= \mathrm{Symbol}(s)$。

**右侧**:$x \oplus (y \oplus z) = \mathrm{Symbol}(s) \oplus (\mathrm{Known}(2) \oplus \mathrm{Known}(3))$。
- $\mathrm{Known}(2) \oplus \mathrm{Known}(3) = \bot$(不等且非 1)。
- $\mathrm{Symbol}(s) \oplus \bot = \bot$(部分运算传播)。
- 右侧 $= \bot$。

故 $(x \oplus y) \oplus z = \mathrm{Symbol}(s) \neq \bot = x \oplus (y \oplus z)$。$\square$

**根因分析**。结合律失败的根源在于 **Symbol 的保守兼容规则**$\mathrm{Symbol}(s) \oplus \mathrm{Known}(n) = \mathrm{Symbol}(s)$(对应 [types.rs:28-29](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/hir/lower/types.rs))。该规则**假设** Symbol 与任意 Known 兼容,但 Symbol 的"真实值"是单一的——若 $s$ 实际为 2,则与 $\mathrm{Known}(3)$ 不兼容。编译期未做 unification,丢失了"Symbol 同时受多个 Known 约束"的信息,导致:

- 左结合:Symbol 先吸收 $\mathrm{Known}(2)$ 成为 Symbol,再"宽容地"吸收 $\mathrm{Known}(3)$。
- 右结合:$\mathrm{Known}(2)$ 与 $\mathrm{Known}(3)$ 先冲突暴露,Symbol 无法挽救。

**实践影响(局限 L4)**。结合律破坏意味着编译期 shape 检查的**结果可能依赖于表达式求值顺序**(即 AST 的结合方式)。例如三元链式广播 `((a ⊕ b) ⊕ c)` 与 `(a ⊕ (b ⊕ c))` 可能一个通过、一个报错。当前 Tenth 编译器未显式处理此问题,但实践中:
- 二元运算的 AST 结合由 parser 决定(左结合),故 `a + b + c` 实际为 `(a + b) + c`,左侧路径——这条路径**更宽容**(Symbol 优先吸收)。
- 跨多步算子的 shape 传播遵循 HIR 的 SSA 顺序,每次 `infer_binary_type` 调用 `broadcast_shapes` 一次,等价于左结合。
- 因此实践中**不会出现"右结合路径暴露冲突而左结合路径漏报"**的情形,但理论上结合律仍破坏。

**注 4.2(诚实陈述)**:结合律破坏是 Symbol 无 unification 的固有代价。引入 unification(Phase 2/3)后,Symbol 将携带约束集 $\{s = n \mid n \text{ 为已知约束}\}$,此时 $\mathrm{Symbol}(s) \oplus \mathrm{Known}(2) \oplus \mathrm{Known}(3)$ 会触发约束冲突 $s = 2 \wedge s = 3 \Rightarrow \bot$,两侧一致,结合律恢复。但 unification 的代价是可判定性下降(见 §6)。

### 4.4 编译期检查的可靠性与完备性(定理 4)

**定理 4(全 Known 输入的可靠性与完备性)**。设 $s, t \in \mathbb{S}$ 且所有维度均为 $\mathrm{Known}(n)$, $n \geq 0$。则:

$$
\mathrm{broadcast\_shapes}(s, t) = \mathrm{None} \iff \mathrm{broadcast\_shape}(\iota(s), \iota(t)) = \mathrm{None}
$$

即编译期与运行时**一致**(可靠且完备)。

**证明**。
$(\Rightarrow)$ 设 $\mathrm{broadcast\_shapes}(s, t) = \mathrm{None}$。由 [types.rs:23-30](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/hir/lower/types.rs),`None` 仅在最后一行 `_ => return None` 触发,即存在某对维度 $(\mathrm{Known}(a), \mathrm{Known}(b))$ 满足 $a \neq b$ 且 $a \neq 1$ 且 $b \neq 1$。对应运行时 `broadcast_shape` 中 $(a, b)$ 匹配 `_ => return None`([tensor.rs:563](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/tensor.rs))。故运行时亦返回 `None`。

$(\Leftarrow)$ 设 $\mathrm{broadcast\_shape}(\iota(s), \iota(t)) = \mathrm{None}$。由 [tensor.rs:560-563](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/tensor.rs),`None` 仅在 `(_, _)` 分支触发,即存在维度对 $(a, b)$ 满足 $a \neq 1, b \neq 1, a \neq b$。对应编译期 $(\mathrm{Known}(a), \mathrm{Known}(b))$, $a \neq b$, $a, b \geq 2$,落入 `_ => return None`。故编译期亦返回 `None`。$\square$

**推论 4.1(全 Known 的精确性)**。若全 $\mathrm{Known}$ 且 $\mathrm{broadcast\_shapes}(s, t) = \mathrm{Some}(r)$,则 $\iota(r) = \mathrm{broadcast\_shape}(\iota(s), \iota(t))$(结果 shape 精确一致)。

**证明**。逐维验证:每维规则在编译期与运行时一一对应($\mathrm{Known}(1) \oplus x = x \Leftrightarrow 1 \oplus_{\mathrm{rt}} x = x$;$\mathrm{Known}(a) \oplus \mathrm{Known}(a) = \mathrm{Known}(a) \Leftrightarrow a \oplus_{\mathrm{rt}} a = a$)。$\square$

**推论 4.2(含 Symbol/Any 的可靠性但不完备)**。若 $s, t$ 含 $\mathrm{Symbol}$ 或 $\mathrm{Any}$:
- (a) **可靠性(无假报错)**:若 $\mathrm{broadcast\_shapes}(s, t) = \mathrm{None}$,则**对任意具体化**(将 Symbol/Any 替换为具体非负整数) $s', t'$,若 $s', t'$ 保留所有 $\mathrm{Known}$ 维度,则 $\mathrm{broadcast\_shape}(s', t') = \mathrm{None}$。
- (b) **不完备(可能漏报)**:存在含 Symbol 的 $s, t$ 使 $\mathrm{broadcast\_shapes}(s, t) = \mathrm{Some}(r)$,但某具体化 $s', t'$ 使 $\mathrm{broadcast\_shape}(s', t') = \mathrm{None}$。

**证明**。
(a) `None` 仅在 `_ => return None` 触发,对应 $\mathrm{Known}(a) \oplus \mathrm{Known}(b)$, $a \neq b$, $a, b \geq 2$——此冲突在任何保留 Known 的具体化下仍存在。
(b) 反例:$s = [\mathrm{Symbol}(s)]$, $t = [\mathrm{Known}(2)]$。编译期 $\mathrm{broadcast\_shapes}$ 返回 $\mathrm{Some}([\mathrm{Symbol}(s)])$(保守兼容)。但具体化 $s' = [5]$, $t' = [2]$ 时 $\mathrm{broadcast\_shape}([5], [2]) = \mathrm{None}$。$\square$

**实践含义**。定理 4 与推论 4.1 保证:**对全静态 shape 的程序,编译期检查与运行时完全一致,无误报无漏报**。推论 4.2 表明:**含 Symbol/Any 时,编译期"通过"不能保证运行时成功**——这是 Tenth 选择 $O(n)$ 检查而非 SMT 求解的代价。

### 4.5 unbroadcast 作为 broadcast 的伴随(定理 5)

**设定**。考虑前向运算 $f(x, y) = x + y$(逐元素加,带广播)。设 $x \in \mathbb{R}^A$, $y \in \mathbb{R}^B$,输出 $z \in \mathbb{R}^O$,$O = \mathrm{BCast}(A, B)$。前向将 $x$ 广播为 $B_A(x) \in \mathbb{R}^O$(沿 $A$ 中为 1、$O$ 中 $> 1$ 的轴复制),$y$ 类似。

**定义 4.1(广播算子 $B_A$)**。$B_A: \mathbb{R}^A \to \mathbb{R}^O$ 为线性算子,将 $x$ 沿广播轴复制。形式化:若 $A = (a_1, \ldots, a_m)$, $O = (o_1, \ldots, o_n)$,$n \geq m$,左填充 $\hat{A} = (1, \ldots, 1, a_1, \ldots, a_m)$,则 $B_A(x)_{i_1, \ldots, i_n} = x_{j_1, \ldots, j_m}$,其中 $j_k = i_{k + (n-m)}$ 若 $a_k \neq 1$(取原索引),否则 $j_k = 0$(广播轴索引固定为 0)。

**定义 4.2(unbroadcast)**。[`autodiff.rs:836-883`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/autodiff.rs) 的 `unbroadcast(grad, target_shape)` 实现:对 $\mathrm{grad} \in \mathbb{R}^O$,目标 shape $A$,沿"广播轴"( padded_target 为 1 而 grad_shape $> 1$ 的轴)求和,再 reshape 到 $A$。

**定理 5(unbroadcast 是 $B_A$ 的伴随)**。对线性算子 $B_A: \mathbb{R}^A \to \mathbb{R}^O$,其伴随 $B_A^*: \mathbb{R}^O \to \mathbb{R}^A$ 满足 $\langle B_A x, g \rangle_O = \langle x, B_A^* g \rangle_A$。则:

$$B_A^*(g) = \mathrm{unbroadcast}(g, A)$$

即 `unbroadcast` 恰为 `broadcast` 的伴随(转置)。

**证明**。
**步骤 1(内积展开)**。$\langle B_A x, g \rangle_O = \sum_{i \in [O]} (B_A x)_i \cdot g_i = \sum_{i \in [O]} x_{\pi(i)} \cdot g_i$,其中 $\pi: [O] \to [A]$ 是广播投影($\pi$ 将广播轴的索引坍缩为 0)。

**步骤 2(交换求和)**。$= \sum_{j \in [A]} x_j \cdot \left(\sum_{i: \pi(i) = j} g_i\right) = \langle x, B_A^* g \rangle_A$,其中 $(B_A^* g)_j = \sum_{i: \pi(i) = j} g_i$。

**步骤 3(unbroadcast 实现)**。`unbroadcast` 在 [`autodiff.rs:853-857`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/autodiff.rs) 中对每个 `padded_target[axis] == 1 && grad_shape[axis] > 1` 的轴调用 `result.sum_axis(Axis(axis))`,即沿广播轴求和。这正是 $\sum_{i: \pi(i) = j} g_i$ 的逐轴实现。

**步骤 4(reshape)**。求和后形状可能仍含被求和轴的退化维度(大小 1),`reshape` 到 $A$ 完成形状对齐([autodiff.rs:860-871](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/autodiff.rs))。

故 $B_A^*(g) = \mathrm{unbroadcast}(g, A)$。$\square$

**推论 5.1(梯度语义正确性)**。对加法 $z = x + y$,梯度 $\frac{\partial L}{\partial x} = B_A^*(\frac{\partial L}{\partial z}) = \mathrm{unbroadcast}(\frac{\partial L}{\partial z}, A)$。对应实现 [`autodiff.rs:301-313`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/autodiff.rs)(`TapeOp::Add` 分支调用 `unbroadcast(&grad, input_shape)`)。

**推论 5.2(乘法/除法梯度)**。对 $z = x \odot y$(逐元素乘),$\frac{\partial L}{\partial x} = B_A^*(\frac{\partial L}{\partial z} \odot B_B(y))$。对应 [`autodiff.rs:322`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/autodiff.rs):`unbroadcast(&(&grad * &b_data), &a_shape)`。除法类似([autodiff.rs:333-334](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/autodiff.rs))。

**注 4.3(局限 L5)**。定理 5 假设前向广播与反向 unbroadcast 使用**同一对 shape**($A$ 与 $O$)。实践中若前向运算修改了 shape(如 reshape 后再加),则需先追溯 reshape 的伴随,再应用 unbroadcast。当前实现未分离这两步,但 `unbroadcast` 内部的 reshape 兜底([autodiff.rs:861-879](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/autodiff.rs))处理了元素数一致但形状不同的退化情形,这是工程兜底而非理论保证。

### 4.6 编译期是运行时的保守过近似(定理 6)

**定义 4.3(具体化)**。具体化函数 $\rho: \mathbb{D} \to \mathcal{P}(\mathbb{D}_{\mathrm{rt}})$ 将编译期维度映射到其可能的具体值集合:

$$
\rho(\mathrm{Known}(n)) = \begin{cases} \{n\} & n \geq 0 \\ \emptyset & n < 0 \end{cases}, \quad
\rho(\mathrm{Symbol}(s)) = \mathbb{N}_0, \quad
\rho(\mathrm{Any}) = \mathbb{N}_0
$$

**定理 6(保守过近似)**。对编译期 shape $s, t \in \mathbb{S}$,若 $\mathrm{broadcast\_shapes}(s, t) = \mathrm{Some}(r)$,则对所有具体化 $s' \in \rho(s), t' \in \rho(t)$(逐维)使 $\mathrm{broadcast\_shape}(s', t') = \mathrm{Some}(r')$,有 $r' \in \rho(r)$(逐维)。

即:**编译期结果是运行时结果的过近似**(编译期更"宽")。

**证明**(逐维归纳)。对每对维度 $(d_s, d_t) \in s \times t$ 与对应 $d_r = d_s \oplus d_t$,验证 $d_r$ 覆盖所有可能的运行时结果:

- $d_s = \mathrm{Any}$:$d_r = \mathrm{Any}$, $\rho(\mathrm{Any}) = \mathbb{N}_0$ 覆盖所有 $\rho(d_t)$ 的运行时结果。$\checkmark$
- $d_s = \mathrm{Known}(1)$, $d_t = x$:$d_r = x$,运行时 $1 \oplus_{\mathrm{rt}} x' = x' \in \rho(x) = \rho(d_r)$。$\checkmark$
- $d_s = \mathrm{Known}(a)$, $d_t = \mathrm{Known}(a)$, $a \geq 2$:$d_r = \mathrm{Known}(a)$,运行时 $a \oplus_{\mathrm{rt}} a = a \in \rho(\mathrm{Known}(a))$。$\checkmark$
- $d_s = \mathrm{Symbol}(s)$, $d_t = \mathrm{Symbol}(s)$:$d_r = \mathrm{Symbol}(s)$, $\rho(\mathrm{Symbol}(s)) = \mathbb{N}_0 \ni$ 运行时任意结果。$\checkmark$
- $d_s = \mathrm{Symbol}(s)$, $d_t = \mathrm{Known}(n)$, $n \geq 2$:$d_r = \mathrm{Symbol}(s)$,运行时若成功则 $s' \oplus_{\mathrm{rt}} n \in \{n, s'\}$,$\subseteq \mathbb{N}_0 = \rho(\mathrm{Symbol}(s))$。$\checkmark$(注意:运行时可能失败,但定理前提是"使 broadcast_shape = Some(r')",故仅考虑成功情形。)$\square$

**推论 6.1(编译期 None 的强保证)**。若 $\mathrm{broadcast\_shapes}(s, t) = \mathrm{None}$,由推论 4.2(a),对所有保留 Known 的具体化,运行时亦 `None`。**但含 Symbol 的具体化可能"逃逸"**(若 Symbol 具体化为与 Known 冲突的值)——此时运行时 `None`,与编译期 `None` 一致,不破坏过近似。

**综合**:定理 6 与推论 4.2 共同刻画编译期检查的语义边界:
- 编译期 `None` $\Rightarrow$ 运行时必 `None`(对保留 Known 的具体化)。
- 编译期 `Some(r)` $\Rightarrow$ 运行时若成功则结果 $\in \rho(r)$;运行时可能失败(含 Symbol 时)。

---

## 5. 与 NumPy 广播规则的形式语义对比

### 5.1 NumPy 的具体整数广播

NumPy 的 `np.broadcast_shapes` 在**具体非负整数**上定义,等价于本文的 $\oplus_{\mathrm{rt}}$。由定理 1,$(\mathbb{D}_{\mathrm{rt}}, \oplus_{\mathrm{rt}}, 1)$ 是部分交换幂等么半群,NumPy 的所有广播性质(交换、结合、幂等)均成立。

### 5.2 Tenth 的三值扩展

Tenth 的 $\oplus$ 在 NumPy 基础上扩展两类维度:
- $\mathrm{Any}$:吸收元,对应 NumPy 中"未知维度"。NumPy 无此概念(所有维度必须具体),Tenth 引入 $\mathrm{Any}$ 以表达"运行时才能确定"。
- $\mathrm{Symbol}$:命名维度,对应 NumPy 中"同一变量"。NumPy 无此概念,Tenth 引入 $\mathrm{Symbol}$ 以表达"两个 shape 在该维度应相等但值未知"。

### 5.3 语义差异

| 性质 | NumPy ($\oplus_{\mathrm{rt}}$) | Tenth 编译期 ($\oplus$) |
|------|------|------|
| 单位元 | 1 | $\mathrm{Known}(1)$ |
| 吸收元 | 无 | $\mathrm{Any}$ |
| 结合律 | ✓(定理 1) | ✗(定理 3,Symbol 破坏) |
| 交换律 | ✓ | ✓ |
| 幂等律 | ✓ | ✓ |
| 最小上界 | ✓(以 1 为底) | ✓ 仅 $\mathrm{Known} \cup \{\mathrm{Any}\}$ 片段(定理 2) |
| 可靠性 | N/A(纯运行时) | ✓(定理 4) |
| 完备性 | N/A | 仅全 $\mathrm{Known}$(推论 4.2) |

**核心差异**:NumPy 的广播是**全定义的代数运算**(在兼容 shape 上),Tenth 的广播是**偏函数代数运算**,且引入 Symbol 后**丧失结合律**。这是 Tenth 为编译期检查付出的代数代价。

---

## 6. 扩展讨论:Symbol Unification 的表达力-可判定性权衡

### 6.1 当前设计(无 Unification)

当前 Tenth 的 Symbol 仅做"同名等价"($\mathrm{Symbol}(s) \oplus \mathrm{Symbol}(s) = \mathrm{Symbol}(s)$)与"保守兼容"($\mathrm{Symbol}(s) \oplus \mathrm{Known}(n) = \mathrm{Symbol}(s)$),**不收集约束**。这导致:

- **表达力弱**:无法表达"$s = 2 \wedge s = 3 \Rightarrow \bot$"(定理 3 反例)。
- **可判定性强**:$\oplus$ 是 $O(1)$ 每维,整体 $O(\max(|s|, |t|))$,线性时间。
- **可靠性保**:编译期 `None` 必为真冲突(定理 4)。

### 6.2 引入 Unification 后(Phase 2/3 路线)

若为每个 Symbol 维护约束集 $C(s) \subseteq \mathbb{Z}$,广播时合并约束:

$$\mathrm{Symbol}(s) \oplus \mathrm{Known}(n) \rightsquigarrow \mathrm{Symbol}(s, C(s) \cap \{n\})$$

若 $C(s) \cap \{n\} = \emptyset$,返回 $\bot$。

**收益**:
- 结合律恢复(定理 3 反例中,右结合 $C(s) \cap \{2\} \cap \{3\} = \emptyset$ 触发冲突,与左结合一致)。
- 表达力提升:可检测"Symbol 受多个不等 Known 约束"的冲突。

**代价**:
- **可判定性下降**:若进一步允许线性约束(如 $s + t = n$),一般情形等价于整数线性规划,非负整数上 NP 完全(参见 `docs/shape-check-roadmap/形式化分析理论可行性论证.md` 定理 B2b)。
- **复杂度上升**:约束合并与一致性检查至少 $O(|C|)$/约束,最坏 NP 完全。

### 6.3 设计建议

基于定理 3 与上述权衡,建议 Phase 2 采用**有限 unification**(仅等式约束 $s = n$,不允许线性组合):
- 收益:恢复结合律,检测 Known-Known 经 Symbol 的间接冲突。
- 代价:每 Symbol 维护一个 $\mathrm{Known}$ 候选集合,合并取交集,$O(1)$/维。可判定,线性时间。
- 不引入 NP 完全性(因不允许 $s + t = n$ 形式的约束)。

Phase 3 可探索**线性约束求解**(允许 $s + t = n$),但需接受 NP 完全性,并设置求解超时回退(回退到当前保守兼容)。

---

## 7. 局限与诚实披露

本节集中记录本文理论分析的 6 处局限,每条说明:是什么、影响多大、如何缓解。

### L1:负维度未校验

- **是什么**:`Dim::Known(i64)` 允许负数,但 `broadcast_shapes` 不校验 `Known(n) >= 0`。运行时 `usize` 不支持负数,负维度在具体化时无意义。
- **影响**:理论上 $\mathrm{Known}(-1) \oplus \mathrm{Known}(-1) = \mathrm{Known}(-1)$ 由定义成立,但运行时无法具体化。`static_numel` 对负维度返回 `None`([types.rs:128](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/hir/types.rs)),部分缓解。
- **缓解建议**:在 `broadcast_shapes` 入口校验 `Known(n) >= 0`,负值返回 `None`。或在 `Dim::Known` 构造时断言。

### L2:运行时结合律的"已知"假设

- **是什么**:定理 1' 的多维度结合律依赖 NumPy 广播的标准性质,本文未给完整归纳证明,仅给"思路"。
- **影响**:理论严格性略低,但该性质是 NumPy 社区共识,且 Tenth 运行时 `broadcast_shape` 严格复刻 NumPy 规则。
- **缓解建议**:未来工作可补充完整归纳(对 shape 秩归纳)。

### L3:偏半格 vs 完全半格

- **是什么**:定理 2 初稿声称 $\mathbb{D}_{\mathrm{KA}}$ 上 $\oplus$ 完全定义,v1 修正为**偏半格**($\mathrm{Known}(2) \oplus \mathrm{Known}(3) = \bot$)。
- **影响**:偏半格的最小上界性质仅在 $\oplus$ 有定义时成立,无上界时返回 $\bot$。这与实现一致(返回 `None`)。
- **缓解**:已在定理 2 证明中诚实修正,无进一步行动。

### L4:结合律破坏的实践影响未量化

- **是什么**:定理 3 证明结合律破坏,但 §4.3 的实践影响分析基于"parser 左结合"的工程假设,未形式化证明"实践中不会出现漏报"。
- **影响**:若未来编译器引入表达式重排优化(如 `a + b + c` → `a + (b + c)`),可能出现"优化前通过、优化后报错"的回归。
- **缓解建议**:(1) 文档化此限制,警告优化器勿重排含 Symbol 的广播链;(2) Phase 2 unification 后此局限消失。

### L5:unbroadcast 的 reshape 兜底非理论保证

- **是什么**:定理 5 假设前向广播与反向 unbroadcast shape 严格对偶,但 [`autodiff.rs:861-879`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/autodiff.rs) 的 reshape 兜底处理了"元素数一致但形状不同"的退化情形,这是工程容错而非定理 5 的覆盖范围。
- **影响**:若前向运算链中混入 reshape,unbroadcast 的 reshape 兜底可能掩盖真实的 shape 不一致(方向 A 已消除 silent squeeze,但仍可能报"unbroadcast reshape 失败"而非更精确的诊断)。
- **缓解建议**:未来工作可将 reshape 的伴随分离为独立算子,而非在 unbroadcast 内兜底。

### L6:跨函数 shape 求解的非代数性

- **是什么**:本文的代数分析限于**单函数内**的 `broadcast_shapes`。跨函数的 shape 传播由 `merge_return_shape`([types.rs:496-521](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/hir/lower/types.rs))处理,该函数是**启发式合并**(取"更精确"的一侧),不遵循 $\oplus$ 的代数规则。
- **影响**:跨函数 shape 检查的可靠性未被本文定理覆盖。可能存在"单函数内通过但跨函数漏报"的情形。
- **缓解建议**:未来工作可将跨函数 shape 求解纳入形式化模型,定义"函数返回 shape"为输入 shape 的函数(依赖型风格),分析其代数性质。

### 循环论证风险评估

本文未发现循环论证。定理 4(可靠性)依赖 $\oplus$ 与 $\oplus_{\mathrm{rt}}$ 的定义对应,不依赖定理 2(半格)或定理 3(结合律)。定理 5(伴随)依赖线性代数,独立于代数结构分析。定理 6(过近似)依赖推论 4.2,而推论 4.2 依赖定理 4,链路清晰。

---

## 8. 结论与未来工作

### 8.1 结论

本文对 Tenth 语言的维度类型系统进行了形式化建模与代数性质分析,主要结论:

1. **运行时层**(`broadcast_shape`):$(\mathbb{D}_{\mathrm{rt}}, \oplus_{\mathrm{rt}}, 1)$ 构成部分交换幂等么半群,多维度广播满足结合律(定理 1, 1')。
2. **编译期 Known-only 片段**:$(\mathbb{D}_{\mathrm{KA}}, \oplus)$ 构成偏有界半格,$\oplus$ 为 subsumption 偏序下的最小上界(定理 2)。
3. **引入 Symbol 后**:结合律破坏(定理 3),根源在于 Symbol-Known 保守混合;这是无 unification 的固有代价。
4. **编译期检查可靠性**:全 Known 输入下可靠且完备(定理 4);含 Symbol/Any 时可靠但不完备(推论 4.2)。
5. **反向传播对偶**:`unbroadcast` 是 `broadcast` 线性算子的伴随(定理 5),梯度语义正确。
6. **过近似关系**:编译期是运行时的保守过近似(定理 6)。

### 8.2 对实施的指导

- **当前设计充分**:定理 4 保证全静态 shape 程序的编译期检查精确,满足护城河 A/D 的需求。
- **Symbol 限制需文档化**:结合律破坏(定理 3)应写入语言参考手册,警告用户"含 Symbol 的广播链结果可能依赖求值顺序"。
- **Phase 2 优先级**:有限 unification(仅等式约束)可恢复结合律且不引入 NP 完全性,建议优先实施。
- **跨函数求解**:L6 提示跨函数 shape 求解的非代数性是未覆盖区域,未来工作需形式化。

### 8.3 未来工作

- **F1**:补全定理 1' 的多维度结合律完整归纳证明。
- **F2**:形式化跨函数 shape 求解的代数性质(依赖型风格)。
- **F3**:设计并实施 Phase 2 有限 unification,验证结合律恢复。
- **F4**:将 `unbroadcast` 的 reshape 兜底分离为独立算子,精确化梯度诊断。
- **F5**:探索线性约束求解(Phase 3)的可判定性边界与超时回退策略。
- **F6**:形式化 `cat`/`permute`/`reshape` 等非广播 shape 变换的代数性质(本文限于广播)。

---

## 附录 A:定理索引

| 定理 | 陈述 | 证明位置 |
|------|------|---------|
| 定理 1 | 运行时单维度广播构成部分交换幂等么半群 | §4.1 |
| 定理 1' | 运行时多维度广播满足结合律 | §4.1 |
| 定理 2 | $\mathrm{Known} \cup \{\mathrm{Any}\}$ 片段构成偏有界半格 | §4.2 |
| 定理 3 | 引入 Symbol 后结合律破坏 | §4.3 |
| 定理 4 | 全 Known 输入的可靠性与完备性 | §4.4 |
| 推论 4.1 | 全 Known 的精确性 | §4.4 |
| 推论 4.2 | 含 Symbol/Any 的可靠性但不完备 | §4.4 |
| 定理 5 | unbroadcast 是 broadcast 的伴随 | §4.5 |
| 推论 5.1 | 加法梯度语义正确性 | §4.5 |
| 推论 5.2 | 乘法/除法梯度语义 | §4.5 |
| 定理 6 | 编译期是运行时的保守过近似 | §4.6 |
| 推论 6.1 | 编译期 None 的强保证 | §4.6 |

**主定理数量**:6(定理 1, 1', 2, 3, 4, 5, 6;其中 1 与 1' 合并计为 1 个主定理系,实际共 7 个带编号定理 + 6 个推论)。

## 附录 B:与现有文档的对应

| 本文章节 | 对应文档 |
|---------|---------|
| §3 形式化模型 | `docs/shape-check-roadmap/形式化分析理论可行性论证.md` §2(Shape 定义) |
| §4.4 可靠性 | 护城河 A(Autograd 反向 Shape 静态验证) |
| §4.5 伴随 | 护城河 A 的理论基础 |
| §4.6 过近似 | 护城河 D(编译期内存/算力预估)的 shape 基础 |
| §6 Unification | `docs/shape-check-roadmap/战略规划.md` Phase 2/3 |
| §7 L6 跨函数 | `docs/shape-check-roadmap/形式化分析理论可行性论证.md` §4(定理 B3) |

## 附录 C:实施建议清单

1. **[高优先级]** 文档化 Symbol 结合律破坏(定理 3):在 `docs/语言参考手册.md` 的 shape 类型章节添加警告。
2. **[中优先级]** 在 `broadcast_shapes` 入口校验 `Known(n) >= 0`(L1):返回 `None` 而非允许负维度传播。
3. **[中优先级]** 评估 Phase 2 有限 unification 的实施成本(F3):基于定理 3 的根因分析,设计约束集数据结构。
4. **[低优先级]** 分离 `unbroadcast` 的 reshape 兜底(L5, F4):提升梯度诊断精度。
5. **[低优先级]** 补全定理 1' 完整归纳证明(F1):理论严格性提升,无实施影响。

---

## 参考文献

1. **NumPy Broadcasting Documentation**. *Broadcasting rules*. https://numpy.org/doc/stable/user/basics.broadcasting.html
2. **JAX Documentation**. *Shapes and broadcasting in JAX*. https://jax.readthedocs.io/en/latest/notebooks/thinking_in_jax.html
3. **Apache TVM**. *Tensor Expression Language*. https://tvm.apache.org/docs/tutorials/language/tensor_expr.html
4. **Tenth 项目总师**. *Tape 形式化根因分析与 Shape 代数求解的理论可行性论证*. `docs/shape-check-roadmap/形式化分析理论可行性论证.md`, 2026-07-01.
5. **Tenth 项目**. *语言参考手册*. `docs/语言参考手册.md`.
6. **Tenth 项目**. *能力全梳理*. `能力梳理/能力全梳理.md`.
7. **Birkhoff, G.** *Lattice Theory*. American Mathematical Society, 3rd ed., 1967. (半格与偏序代数经典参考)
8. **Davey, B.A. & Priestley, H.A.** *Introduction to Lattices and Order*. Cambridge University Press, 2nd ed., 2002. (有界半格与 subsumption 偏序)
9. **Griewank, A. & Walther, A.** *Evaluating Derivatives: Principles and Techniques of Algorithmic Differentiation*. SIAM, 2nd ed., 2008. (自动微分的伴随算子理论)
10. **Brady, E.** *Type-Driven Development with Idris*. Manning, 2017. (依赖类型与 shape 类型系统的关系)

---

> **数理部诚实声明**:本文的代数分析基于 Tenth v0.3.3+ 源码,所有定理均附实现引用。结合律破坏(定理 3)是本文最重要的理论发现,直接影响护城河 B(Shape 代数求解器)的设计前提。我们未掩盖此局限,而是将其作为 Phase 2 unification 的理论动机。任何后续实施应基于本文的可靠性与完备性边界(定理 4, 推论 4.2)设定预期,不应假设编译期检查对含 Symbol 程序完备。
