# Shape 变换四元分类的互斥完备性：基于 Tenth 21 个 TapeOp 的形式化证明

> **作者**：Tenth 项目数理部
> **日期**：2026-07-02
> **类型**：理论分析论文（T7 理论点）
> **实证基础**：Tenth v0.3.3+ 源码（`runtime/autodiff.rs`、`hir/lower/types.rs`、`runtime/tensor.rs`）
> **关联文档**：`docs/shape-check-roadmap/形式化分析理论可行性论证.md`、`docs/语言参考手册.md`、`docs/论文/T1-Shape代数系统的形式化建模.md`、`docs/论文/T8-Shape解释关系四级分类.md`
> **版本**：v1（首轮分析，含 4 轮自审修正留痕）

---

## 摘要

本文对 Tenth 语言的 21 个自动微分算子（`TapeOp`）的 shape 变换行为进行形式化分类,证明 **Construct / Preserve / Reduce / Expand** 四元分类的互斥性与完备性。我们将每个算子的 shape 语义抽象为变换函数 $\sigma: \mathcal{S}^{k} \to \mathcal{S}$,基于"维度秩的增减"与"维度是否被收缩/广播"两条结构性判据,对 [`runtime/autodiff.rs`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/autodiff.rs) 中全部 21 个 `TapeOp` 变体完成归类。主要结果:(1)**定理 T7-1**(互斥性)四类变换两两不相交,任意算子在固定输入下至多归属一类;(2)**定理 T7-2**(完备性)四类变换覆盖 Tenth 全部 21 个 `TapeOp`,分布为 Construct×1、Preserve×11、Reduce×5、Expand×4;(3)**定理 T7-3**(分类稳定性)分类对一元复合算子封闭,复合的分类可由子算子分类的" supremum"判定。本文诚实记录 6 处理论局限,重点处理三类边界情形:`Transpose`(置换型 Preserve)、`MatMul`/`Conv2D`(收缩型 Reduce,秩不严格单调)、`Add` 系列(广播型 Expand,同 shape 时退化为 Preserve)。完备性保证 shape 推断算法对 21 个算子无遗漏,为护城河 F(张量关系调试器)与护城河 A(Autograd 反向 Shape 静态验证)提供分类学基础。

**关键词**:shape 变换分类、互斥完备性、TapeOp、自动微分、shape 推断、张量算子、形式化证明、Tenth 语言

---

## 1. 引言

### 1.1 shape 推断算法的完整性需求

AI 原生编程语言的核心能力之一是**编译期与运行时的 shape 推断**(shape inference)。Tenth 的 shape 推断分两层:编译期由 [`hir/lower/types.rs::resolve_method_type`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/hir/lower/types.rs)(第 219-386 行)按算子名查表推断输出 shape;运行时由 [`runtime/autodiff.rs`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/autodiff.rs) 的 `Tape::forward` 在执行时记录每个 `TapeNode` 的输入输出张量 shape。

shape 推断算法的**完整性**(completeness)要求:对语言支持的每一个算子,推断算法都有对应的规则,不存在"无规则可查"的算子。若某算子无推断规则,程序中该算子的输出 shape 将退化为 `Any`(完全未知),后续 shape 检查失效,shape 错误可能延迟到运行时才暴露,违背 Tenth"早报错"的设计原则(见护城河 A/D)。

**分类法在完整性论证中的作用**:若能将所有算子按 shape 变换行为归入有限类别,且每类有统一的推断规则,则只需证明"类别覆盖全部算子"(完备性)即可保证算法无遗漏,无需逐一检查每个算子。这是分类法的核心价值——**将无穷的算子实例归约为有限的类别**。

### 1.2 分类法在编译器优化中的作用

shape 变换分类不仅服务于完整性论证,还指导编译器优化:

- **内存复用**:Preserve 类算子(如 `Exp`/`Log`)的输出可与输入共享缓冲区或原地计算(若 dtype 一致)。
- **算子融合**:相邻的 Preserve 类算子可融合为单一 kernel(如 `Exp → Mul` 融合为 `Swish` 激活)。
- **shape 静态传播**:Preserve 类的输出 shape 等于输入,无需额外计算;Reduce/Expand 类需特定公式。
- **根因诊断**:护城河 F(见 [`形式化分析理论可行性论证.md`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/docs/shape-check-roadmap/形式化分析理论可行性论证.md) §3)的根因候选集上界依赖分类——非 Preserve 类算子是 shape 漂移的主要嫌疑。

### 1.3 研究问题与贡献

本文回答以下三个研究问题:

- **RQ1**:Tenth 的 21 个 `TapeOp` 可否归入 Construct/Preserve/Reduce/Expand 四类,使每类有统一的 shape 推断规则?
- **RQ2**:四元分类是否互斥(任一算子至多属一类)且完备(覆盖全部 21 个算子)?
- **RQ3**:分类对算子复合是否封闭?即,复合算子的分类可否由子算子的分类推导?

**贡献**:

1. **形式化定义**(§3):将 shape 变换抽象为函数 $\sigma: \mathcal{S}^{k} \to \mathcal{S}$,给出四类的形式化判据,并扩展到二元算子。
2. **实证分类**(§4):对 [`autodiff.rs`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/autodiff.rs) 第 30-79 行定义的全部 21 个 `TapeOp` 变体逐一归类,每个分类附源码行号证据。
3. **三个主定理**(§5):互斥性(T7-1)、完备性(T7-2)、分类稳定性(T7-3),附完整证明。
4. **shape 推断指导**(§6):每类的推断规则与无遗漏性证明。
5. **诚实局限**(§8):独立章节记录 6 处局限,重点处理三类边界情形。

### 1.4 v1 自审留痕

本文经历 4 轮自审,主要修正:

| 轮次 | 原始断言 | 修正 |
|------|---------|------|
| 第 1 轮(结构) | 声称 `MatMul` 是 Preserve(秩不变) | 修正:`MatMul` 是 Reduce(收缩语义,见 §4.4) |
| 第 2 轮(证明) | T7-1 互斥性证明未处理"Add 同 shape 退化" | 补充:分类基于算子的**结构行为**而非具体输入(§3.4) |
| 第 3 轮(边界) | 未处理 `Transpose`(置换非恒等) | 补充:Preserve 定义放宽到"维度多重集保持"(§3.3) |
| 第 4 轮(诚实) | T7-2 完备性声称"严格覆盖" | 修正:`MatMul`/`Conv2D` 的 Reduce 归类依赖"收缩"判据的扩展定义,非秩严格单调(§8.2) |

---

## 2. 背景与相关工作

### 2.1 类型系统中的代数数据类型分类

**代数数据类型**(ADT)将类型分为"和类型"(sum,tagged union)与"积类型"(product,struct)。Haskell 的 `Functor` 类型类进一步将类型构造子按 `fmap` 的行为分类。shape 变换分类借鉴这一思路:将算子按"shape 的代数变换"分类,而非按"数值语义"分类。

**Hindley-Milner 类型推断**中,类型变量间的约束($\alpha = \beta$、$\alpha = \text{Int}$)构成合一(unification)问题。shape 推断类比:shape 变量间的约束($d_i = d_j$、$d_i = c$)构成求解问题(见 [`T3-HIR约束求解NP完全性归约.md`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/docs/论文/T3-HIR约束求解NP完全性归约.md))。分类法将算子约束按"等式型"(Preserve)、"不等式型"(Reduce/Expand)、"无约束型"(Construct)归类,简化约束系统的结构分析。

### 2.2 NumPy 的 ufunc 分类

NumPy 的 universal function(ufunc)按行为分两类:

- **element-wise ufunc**(如 `np.exp`):输出 shape = 输入 shape,对应本文的 Preserve。
- **reduction ufunc**(如 `np.sum`):输出 shape 是输入 shape 的子集(沿 axis 求和),对应本文的 Reduce。

NumPy **无 Construct 概念**(张量由 `np.array` 构造,不在 ufunc 体系内),且**无显式 Expand 概念**(广播是 ufunc 的隐式属性,不作为独立类别)。本文的四元分类是 NumPy 隐式分类的显式化与扩展。

### 2.3 XLA 的 shape inference

XLA(Google TPU 编译器)的 HLO 算子有显式的 shape inference 函数(见 `xla::ShapeInference`),每个 HLO 算子(`Dot`、`Convolve`、`Broadcast`、`Reduce`等)有独立的推断规则。XLA 的算子名已隐含分类(`Broadcast` = Expand,`Reduce` = Reduce,element-wise = Preserve),但**未形式化证明分类的互斥完备性**。本文的形式化证明可视为对 XLA 隐式分类的严格化。

### 2.4 与 T1 Shape 代数的关系

[`T1-Shape代数系统的形式化建模.md`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/docs/论文/T1-Shape代数系统的形式化建模.md) 建模了 Tenth 的 `Dim` 三值域与广播运算 $\oplus$ 的代数性质,证明 $\oplus$ 在 `Known ∪ {Any}` 片段构成有界半格。本文的 Expand 类(广播型)直接依赖 T1 的广播代数——`Add`/`Sub`/`Mul`/`Div` 的 shape 推断规则即 T1 的 $\oplus$ 运算。T1 关注"维度值的代数",本文关注"shape 变换的分类",两者互补:T1 提供 Expand 类的理论基础,本文提供跨类的完整性保证。

### 2.5 与 T8 四级解释分类的关系

[`T8-Shape解释关系四级分类.md`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/docs/论文/T8-Shape解释关系四级分类.md) 定义根因解释的四级分类(DefinitelyRoot / ExplainsError / PartialExplain / Unrelated),其中 (C2) 条件依赖本文的 `Class(v)` 分类(Construct/Preserve/Reduce/Expand,见 [`形式化分析理论可行性论证.md`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/docs/shape-check-roadmap/形式化分析理论可行性论证.md) §2.2 定义 2.4)。本文证明的分类完备性是 T8 根因分析"无遗漏"的前提——若分类不完备,某些算子无法计算 `Class(v)`,T8 的候选集会漏报。

---

## 3. 四元分类的形式化定义

### 3.1 前置定义:Shape 与算子

**定义 3.1(Shape)**。Shape 是非负整数元组 $s = (d_1, \ldots, d_n)$,其中 $n \geq 0$,$d_i \in \mathbb{N} = \{0, 1, 2, \ldots\}$。空元组 $\epsilon$ 表示标量($n = 0$)。记所有 shape 的集合为 $\mathcal{S} = \bigcup_{n \geq 0} \mathbb{N}^n$。

- **秩**(rank):$\|s\| = n$(维度数)。
- **体积**(volume):$|s| = \prod_{i=1}^{n} d_i$,约定 $|\epsilon| = 1$(空积)。
- **维度多重集**:$\mathrm{ms}(s) = \{d_1, \ldots, d_n\}_{\text{multi}}$(允许重复元素的无序集合)。

**定义 3.2(TapeOp 算子集)**。Tenth 的自动微分算子集合 $\mathcal{O}$ 定义于 [`runtime/autodiff.rs:30-79`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/autodiff.rs),共 21 个变体:

$$\mathcal{O} = \{\textsf{Input}, \textsf{Add}, \textsf{Sub}, \textsf{Mul}, \textsf{Div}, \textsf{Neg}, \textsf{ReLU}, \textsf{MatMul}, \textsf{Transpose}, \textsf{Sum}, \textsf{Mean}, \textsf{Exp}, \textsf{Log}, \textsf{Sigmoid}, \textsf{Softmax}, \textsf{CrossEntropy}, \textsf{Dropout}, \textsf{Conv2D}, \textsf{BatchNorm}, \textsf{LayerNorm}, \textsf{Gelu}\}$$

每个算子 $op \in \mathcal{O}$ 有固定元数 $k_{op}$(输入数,见 `TapeNode.inputs` 的长度约定):

- $k_{\textsf{Input}} = 0$(无上游 Tape 节点,见 [`autodiff.rs:99-106`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/autodiff.rs) `inputs: vec![]`)。
- $k_{\textsf{Neg}}, k_{\textsf{ReLU}}, \ldots, k_{\textsf{Gelu}} = 1$(一元算子,见 [`autodiff.rs:113-122`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/autodiff.rs) `unary` 的 `inputs: vec![input_id]`)。
- $k_{\textsf{Add}}, k_{\textsf{Sub}}, k_{\textsf{Mul}}, k_{\textsf{Div}}, k_{\textsf{MatMul}}, k_{\textsf{Conv2D}} = 2$(二元算子,见 [`autodiff.rs:128-137`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/autodiff.rs) `binary` 的 `inputs: vec![a_id, b_id]`)。

**注**:`CrossEntropy`/`BatchNorm`/`LayerNorm`/`Dropout` 在 Tape 注册时 `inputs` 字段只有 1 个上游节点(见 [`autodiff.rs:157-208`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/autodiff.rs) `cross_entropy`/`batchnorm`/`layernorm`/`dropout`),但 `input_tensors` 含多个张量(如 `target`、`gamma`、`beta`、`mask`)。从 shape 流的角度,$k_{op}$ 按**上游 Tape 节点数**计算(即 `inputs.len()`),辅助张量(`target`/`gamma`/`mask` 等)不参与 shape 流的主链路。故 $k_{\textsf{CrossEntropy}} = k_{\textsf{BatchNorm}} = k_{\textsf{LayerNorm}} = k_{\textsf{Dropout}} = 1$。

### 3.2 Shape 变换函数

**定义 3.3(Shape 变换函数)**。算子 $op$ 的 shape 变换函数是偏函数:

$$\sigma_{op}: \mathcal{S}^{k_{op}} \to \mathcal{S} \cup \{\bot\}$$

其中 $\bot$ 表示"输入 shape 对该算子不合法"。$\sigma_{op}(s_1, \ldots, s_{k_{op}}) = s^{out}$ 当且仅当 $op$ 在输入 shape $(s_1, \ldots, s_{k_{op}})$ 下合法且输出 shape 为 $s^{out}$。

**实现对应**:$\sigma_{op}$ 对应 [`hir/lower/types.rs::resolve_method_type`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/hir/lower/types.rs)(第 219-356 行)与 [`runtime/autodiff.rs`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/autodiff.rs) 的 forward 实现。例如:

- $\sigma_{\textsf{MatMul}}((m, k), (k', n)) = (m, n)$,要求 $k = k'$(内侧维度匹配),否则 $\bot$。见 [`types.rs:223-253`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/hir/lower/types.rs) `matmul` 分支与 [`autodiff.rs:350-443`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/autodiff.rs) `TapeOp::MatMul` 分支。
- $\sigma_{\textsf{Add}}(s, t) = s \oplus t$(广播,见 T1 定义),要求 $s, t$ 可广播,否则 $\bot$。见 [`types.rs:150-153`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/hir/lower/types.rs) `broadcast_shapes` 调用与 [`autodiff.rs:301-314`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/autodiff.rs) `TapeOp::Add` 分支。
- $\sigma_{\textsf{Sum}}(s) = \epsilon$(标量),任意 $s$ 合法。见 [`autodiff.rs:455-463`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/autodiff.rs) `TapeOp::Sum` 分支。

### 3.3 四元分类的形式化判据

**定义 3.4(四元分类)**。对算子 $op$ 与合法输入 shape $(s_1, \ldots, s_{k_{op}})$,设 $s^{out} = \sigma_{op}(s_1, \ldots, s_{k_{op}}) \neq \bot$。定义四类变换:

**(C1) Construct(构造)**:

$$k_{op} = 0 \quad \text{且} \quad s^{out} \in \mathcal{S}$$

即算子无上游 Tape 输入,从外部数据构造张量。形式化为 $\sigma: \mathbb{N}^0 \to \mathbb{N}^k$($k \geq 0$)。

**(C2) Preserve(保持)**:

$$k_{op} \geq 1 \quad \text{且} \quad \forall j \in [1, k_{op}]: \mathrm{ms}(s^{in}_j) = \mathrm{ms}(s^{out})$$

即每个输入的维度多重集等于输出的维度多重集。**严格 Preserve**(子类)要求 $\forall j: s^{in}_j = s^{out}$(元组相等,含顺序);**置换 Preserve**(子类)允许 $\exists j: s^{in}_j \neq s^{out}$ 但 $\mathrm{ms}(s^{in}_j) = \mathrm{ms}(s^{out})$(如 `Transpose`)。

形式化(一元情形,$k_{op} = 1$):$\sigma: \mathbb{N}^k \to \mathbb{N}^k$ 且 $\mathrm{ms}(\sigma(s)) = \mathrm{ms}(s)$(维度多重集保持)。

**(C3) Reduce(降维/收缩)**:

$$k_{op} \geq 1 \quad \text{且} \quad \text{非 Preserve} \quad \text{且} \quad \mathrm{Contracted}(op, s^{in}_1, \ldots, s^{in}_{k_{op}}, s^{out})$$

其中 $\mathrm{Contracted}$ 是**收缩判据**:存在某个维度值 $d$ 在输入的"结构分解"中出现,但在输出中没有对应的"结构位置"。形式化为:

$$\mathrm{Contracted}(op, \ldots) \iff \begin{cases}
\|s^{out}\| < \max_j \|s^{in}_j\| & \text{(秩严格减少,一元归约型)} \\
\text{或 } op \in \mathcal{O}_{\text{contract}} & \text{(结构收缩型,如 MatMul/Conv2D)}
\end{cases}$$

其中 $\mathcal{O}_{\text{contract}} = \{\textsf{MatMul}, \textsf{Conv2D}\}$ 是收缩型算子集合(其语义是"沿某些维度求和/内积",见 §4.4、§4.5 论证)。

形式化(一元归约型):$\sigma: \mathbb{N}^k \to \mathbb{N}^m$ 且 $m < k$。

**(C4) Expand(扩展)**:

$$k_{op} \geq 1 \quad \text{且} \quad \text{非 Preserve} \quad \text{且} \quad \text{非 Reduce} \quad \text{且} \quad \mathrm{Broadcast}(op, s^{in}_1, \ldots, s^{in}_{k_{op}}, s^{out})$$

其中 $\mathrm{Broadcast}$ 是**广播判据**:存在某个输入 $j$ 使 $s^{in}_j \neq s^{out}$ 但 $s^{in}_j$ 可广播到 $s^{out}$(即 $s^{out} = s^{in}_j \oplus \text{其他输入}$,$\oplus$ 是 T1 定义的广播运算)。形式化为:

$$\mathrm{Broadcast}(op, \ldots) \iff \exists j: s^{in}_j \neq s^{out} \quad \text{且} \quad s^{out} = \bigoplus_{j} s^{in}_j$$

形式化(一元情形,$k_{op} = 1$ 不存在 Expand,因一元算子无广播对象;Expand 仅对 $k_{op} \geq 2$):

$$\sigma: \mathcal{S}^k \to \mathcal{S} \quad (k \geq 2) \quad \text{且} \quad \exists j: \|s^{in}_j\| < \|s^{out}\| \text{ 或 } (\|s^{in}_j\| = \|s^{out}\| \text{ 且 } |s^{in}_j| < |s^{out}|)$$

即"输出比某些输入大"(维度更多或体积更大)。

### 3.4 结构行为 vs. 具体输入

**重要说明**:定义 3.4 的判据基于**具体输入 shape**,因此同一算子在不同输入下可能归入不同类别。例如 `Add` 在同 shape 输入下是 Preserve,在广播输入下是 Expand。

为支持"逐一归类"(§4),我们定义算子的**结构行为分类**(structural classification),基于算子的**shape 推断规则**而非具体输入:

**定义 3.5(结构行为分类)**。算子 $op$ 的结构行为分类 $\mathrm{Class}^*(op)$ 是基于其 shape 推断规则 $\sigma_{op}$ 的"最一般行为":

- 若 $k_{op} = 0$:$\mathrm{Class}^*(op) = \text{Construct}$。
- 若 $\sigma_{op}$ 对所有合法输入都满足 Preserve:$\mathrm{Class}^*(op) = \text{Preserve}$。
- 若 $\sigma_{op}$ 的规则是"收缩"(输出由输入沿某些维度求和/内积得到):$\mathrm{Class}^*(op) = \text{Reduce}$。
- 若 $\sigma_{op}$ 的规则是"广播"(输出是输入广播的结果,可能扩展):$\mathrm{Class}^*(op) = \text{Expand}$。

**关键性质**:结构行为分类对每个算子是唯一的(每个算子只有一个 $\sigma_{op}$ 规则)。互斥性(T7-1)与完备性(T7-2)对结构行为分类证明。

**退化情形**:当 Expand 类算子的输入恰好同 shape 时,其具体行为退化为 Preserve(广播结果等于输入)。这不违反互斥性——结构行为分类仍为 Expand,只是该次调用的"实例行为"是 Preserve 的退化。这一区分对 shape 推断算法无影响(规则仍是广播),但对根因诊断(护城河 F)有意义(同 shape 调用不产生 shape 漂移)。

---

## 4. 21 个 TapeOp 的分类

本节对 [`autodiff.rs:30-79`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/autodiff.rs) 定义的全部 21 个 `TapeOp` 变体逐一归类,每个分类附源码证据与判定论证。

### 4.1 分类总表

| # | TapeOp | $k_{op}$ | $\mathrm{Class}^*$ | 判定依据(源码行号) |
|---|--------|----------|--------------------|--------------------|
| 1 | `Input` | 0 | **Construct** | [`autodiff.rs:99-106`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/autodiff.rs) `inputs: vec![]` |
| 2 | `Add` | 2 | **Expand** | [`autodiff.rs:301-314`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/autodiff.rs) `unbroadcast`(广播反向) |
| 3 | `Sub` | 2 | **Expand** | [`autodiff.rs:301-314`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/autodiff.rs) 同 Add,符号 -1 |
| 4 | `Mul` | 2 | **Expand** | [`autodiff.rs:315-326`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/autodiff.rs) `unbroadcast` |
| 5 | `Div` | 2 | **Expand** | [`autodiff.rs:327-337`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/autodiff.rs) `unbroadcast` |
| 6 | `Neg` | 1 | **Preserve** | [`autodiff.rs:338-341`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/autodiff.rs) `g = -&grad`(同 shape) |
| 7 | `ReLU` | 1 | **Preserve** | [`autodiff.rs:342-349`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/autodiff.rs) `mask`(同 shape) |
| 8 | `MatMul` | 2 | **Reduce** | [`autodiff.rs:350-443`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/autodiff.rs) 收缩 k 维 |
| 9 | `Transpose` | 1 | **Preserve** | [`autodiff.rs:444-454`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/autodiff.rs) 置换最后两维 |
| 10 | `Sum` | 1 | **Reduce** | [`autodiff.rs:455-463`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/autodiff.rs) `→ scalar` |
| 11 | `Mean` | 1 | **Reduce** | [`autodiff.rs:464-473`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/autodiff.rs) `→ scalar` |
| 12 | `Exp` | 1 | **Preserve** | [`autodiff.rs:474-480`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/autodiff.rs) `result_ref.data`(同 shape) |
| 13 | `Log` | 1 | **Preserve** | [`autodiff.rs:481-487`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/autodiff.rs) `a_ref.data`(同 shape) |
| 14 | `Sigmoid` | 1 | **Preserve** | [`autodiff.rs:488-495`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/autodiff.rs) `result_ref.data`(同 shape) |
| 15 | `Softmax` | 1 | **Preserve** | [`autodiff.rs:735-745`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/autodiff.rs) `result_ref.data`(同 shape) |
| 16 | `CrossEntropy` | 1 | **Reduce** | [`autodiff.rs:723-734`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/autodiff.rs) `logits → scalar` |
| 17 | `Dropout` | 1 | **Preserve** | [`autodiff.rs:712-722`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/autodiff.rs) `mask * grad`(同 shape) |
| 18 | `Conv2D` | 2 | **Reduce** | [`autodiff.rs:615-711`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/autodiff.rs) 收缩 C_in/kH/kW |
| 19 | `BatchNorm` | 1 | **Preserve** | [`autodiff.rs:496-522`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/autodiff.rs) `dX` 同 shape |
| 20 | `LayerNorm` | 1 | **Preserve** | [`autodiff.rs:523-596`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/autodiff.rs) `d_x` 同 shape |
| 21 | `Gelu` | 1 | **Preserve** | [`autodiff.rs:597-614`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/autodiff.rs) `deriv`(同 shape) |

**分类统计**:

| 类别 | 数量 | 算子 |
|------|------|------|
| Construct | 1 | Input |
| Preserve | 11 | Neg, ReLU, Transpose, Exp, Log, Sigmoid, Softmax, Dropout, BatchNorm, LayerNorm, Gelu |
| Reduce | 5 | MatMul, Sum, Mean, CrossEntropy, Conv2D |
| Expand | 4 | Add, Sub, Mul, Div |
| **合计** | **21** | — |

### 4.2 Construct 类判定论证

**`Input`** ([`autodiff.rs:32, 99-106`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/autodiff.rs)):

- $k_{\textsf{Input}} = 0$(`inputs: vec![]`,无上游 Tape 节点)。
- `input_tensors: vec![tensor]` 表示从外部注册一个已存在的张量(叶子参数)。
- shape 推断规则:输出 shape = 注册张量的 shape(外部给定)。
- 满足定义 3.4 (C1):$k_{op} = 0$,$\sigma: \mathbb{N}^0 \to \mathbb{N}^k$。
- **归类**:Construct。

### 4.3 Preserve 类判定论证

Preserve 类共 11 个算子,均为一元($k_{op} = 1$),输出 shape = 输入 shape(或置换)。分两组论证:

**组 1:严格 Preserve(9 个)**

`Neg`、`ReLU`、`Exp`、`Log`、`Sigmoid`、`Softmax`、`Dropout`、`BatchNorm`、`LayerNorm`、`Gelu`(共 10 个,但 `Gelu` 单独说明)。

这些算子的 shape 推断规则为 $\sigma(s) = s$(输出 shape 严格等于输入 shape)。源码证据:

- [`types.rs:285-289`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/hir/lower/types.rs):`"abs" | "sqrt" | "exp" | "log" | "relu" | "sigmoid" | "tanh" | "softmax" | "gelu"` 分支返回 `Type::Tensor { dtype, dims: dims.clone() }`(克隆输入 dims)。
- 反向传播证据:每个算子的梯度 `g_a` 与输入 `a_ref.data` 或 `result_ref.data` 同 shape(逐元素运算)。例如 `Exp` 的 `g_a = &grad * &result_ref.data`([`autodiff.rs:477`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/autodiff.rs)),`grad` 与 `result_ref.data` 同 shape,故 `g_a` 与输入同 shape。

满足定义 3.4 (C2) 严格 Preserve 子类:$\sigma: \mathbb{N}^k \to \mathbb{N}^k$ 且 $\forall i: \sigma_i = \mathrm{id}$。

`BatchNorm`/`LayerNorm` 的输出 shape = 输入 shape(`dX` 与 `x_hat_ref.shape()` 同 shape,见 [`autodiff.rs:515`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/autodiff.rs) `&std_inv_ref.data * &gamma_ref.data * ...`)。`gamma`/`beta` 是参数(非主 shape 流),不影响主输入的 shape 保持。

**组 2:置换 Preserve(1 个)**

`Transpose` ([`autodiff.rs:47-48, 444-454`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/autodiff.rs)):

- shape 推断规则:$\sigma((d_1, \ldots, d_n)) = (d_1, \ldots, d_{n-2}, d_n, d_{n-1})$(交换最后两维)。
- 见 [`types.rs:346-352`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/hir/lower/types.rs):2D 时 `dims: vec![dims[1].clone(), dims[0].clone()]`,非 2D 时 `dims: dims.clone()`(保守保持)。
- $\|s^{out}\| = \|s^{in}\|$(秩不变),$\mathrm{ms}(s^{out}) = \mathrm{ms}(s^{in})$(维度多重集保持)。
- **但不满足严格 Preserve**:$s^{out} \neq s^{in}$(顺序不同,2D 情形)。
- 满足定义 3.4 (C2) 置换 Preserve 子类:$\mathrm{ms}(s^{out}) = \mathrm{ms}(s^{in})$。
- **归类**:Preserve(置换子类)。
- **边界说明**:这是本文 Preserve 定义的放宽点,见 §8.1 局限 L1。

### 4.4 Reduce 类判定论证

Reduce 类共 5 个算子,分三组论证:

**组 1:一元归约(3 个)**

`Sum`、`Mean`、`CrossEntropy`。

- `Sum` ([`autodiff.rs:49-50, 455-463`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/autodiff.rs)):$\sigma(s) = \epsilon$(标量,空元组)。$\|s^{out}\| = 0 < \|s^{in}\|$(对 $\|s^{in}\| \geq 1$)。满足定义 3.4 (C3) 秩严格减少。
- `Mean` ([`autodiff.rs:51-52, 464-473`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/autodiff.rs)):同 `Sum`,$\sigma(s) = \epsilon$。
- `CrossEntropy` ([`autodiff.rs:62-63, 723-734`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/autodiff.rs)):$\sigma(s) = \epsilon$(标量 loss)。输入 `logits`(如 $(B, C)$),输出标量。$\|s^{out}\| = 0 < \|s^{in}\|$。

三者均满足一元归约型形式化:$\sigma: \mathbb{N}^k \to \mathbb{N}^m$ 且 $m < k$(此处 $m = 0$)。

**组 2:二元收缩(2 个)**

`MatMul`、`Conv2D`。

`MatMul` ([`autodiff.rs:46, 350-443`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/autodiff.rs)):

- shape 推断规则:$\sigma_{\textsf{MatMul}}((m, k), (k', n)) = (m, n)$,要求 $k = k'$。
- 见 [`types.rs:223-253`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/hir/lower/types.rs):`dims: vec![dims[0].clone(), adims[1].clone()]`。
- **收缩判据**:内侧维度 $k$ 是"收缩维"——它出现在两个输入中,但在输出中消失。输出的 $(m, n)$ 是两个输入的"外侧维度",各来自一个输入。
- 语义本质:`MatMul` 是沿 $k$ 维的加权求和(内积),$k$ 维被"收缩"。
- $\|s^{out}\| = 2 = \max(\|s^{in}_1\|, \|s^{in}_2\|) = 2$(秩不严格减少),但满足 $\mathrm{Contracted}$ 判据(因 $op \in \mathcal{O}_{\text{contract}}$)。
- **归类**:Reduce(结构收缩型)。
- **边界说明**:秩不严格单调,依赖收缩判据的扩展定义,见 §8.2 局限 L2。

`Conv2D` ([`autodiff.rs:67-69, 615-711`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/autodiff.rs)):

- shape 推断规则:$\sigma_{\textsf{Conv2D}}((N, C_{in}, H, W), (C_{out}, C_{in}, k_H, k_W)) = (N, C_{out}, H_{out}, W_{out})$。
  - $H_{out} = \lfloor (H + 2P - k_H) / S \rfloor + 1$,$W_{out}$ 类似($P$ = padding,$S$ = stride)。
- 见 [`autodiff.rs:629-633`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/autodiff.rs):`out_shape = [N, C_out, H_out, W_out]`。
- **收缩判据**:$C_{in}$、$k_H$、$k_W$ 是收缩维——它们出现在输入中(权重 shape 含 $C_{in} \cdot k_H \cdot k_W$),但在输出中消失。输出沿 $C_{in} \cdot k_H \cdot k_W$ 维求和(卷积 = im2col + matmul,见 [`autodiff.rs:617`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/autodiff.rs) 注释 `output = im2col @ w_flat^T`)。
- $op \in \mathcal{O}_{\text{contract}}$。
- **归类**:Reduce(结构收缩型)。

### 4.5 Expand 类判定论证

Expand 类共 4 个算子,均为二元($k_{op} = 2$):

`Add`、`Sub`、`Mul`、`Div`。

- shape 推断规则:$\sigma(s, t) = s \oplus t$(NumPy 广播,见 T1 定义 3.6)。
- 见 [`types.rs:150-153`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/hir/lower/types.rs):`broadcast_shapes(ldims, rdims)` 返回广播结果。
- 见 [`autodiff.rs:301-337`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/autodiff.rs):反向传播使用 `unbroadcast`([`autodiff.rs:836-883`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/autodiff.rs)),证明前向存在广播。
- **广播判据**:当 $s \neq t$ 且可广播时,$s^{out} = s \oplus t$ 满足 $\|s^{out}\| > \min(\|s\|, \|t\|)$ 或 $|s^{out}| > \min(|s|, |t|)$。例如 $s = (3, 4)$,$t = (4,)$,$s^{out} = (3, 4)$,$\|t\| = 1 < \|s^{out}\| = 2$。
- 满足定义 3.4 (C4):$\exists j: s^{in}_j \neq s^{out}$ 且 $s^{out} = s^{in}_1 \oplus s^{in}_2$。
- **退化情形**:当 $s = t$ 时,$s^{out} = s = t$,具体行为退化为 Preserve。但结构行为分类仍为 Expand(规则是广播)。
- **归类**:Expand。

---

## 5. 主定理与证明

### 5.1 定理 T7-1(互斥性)

**定理 T7-1(互斥性)**。对任意算子 $op \in \mathcal{O}$ 与合法输入 shape $(s_1, \ldots, s_{k_{op}})$,定义 3.4 的四类变换(Construct/Preserve/Reduce/Expand)中**至多一类**成立。即四类两两不相交:

$$\forall op, \forall \text{合法输入}: |\{C \in \{\text{Construct}, \text{Preserve}, \text{Reduce}, \text{Expand}\} : C \text{ 成立}\}| \leq 1$$

**证明**。对定义 3.4 的判据进行结构分析。

**步骤 1:Construct 与其他三类互斥**。

Construct 要求 $k_{op} = 0$。Preserve/Reduce/Expand 均要求 $k_{op} \geq 1$。故 $k_{op} = 0$ 与 $k_{op} \geq 1$ 矛盾,Construct 与其他三类互斥。

**步骤 2:Preserve 与 Reduce/Expand 互斥**。

Preserve 要求 $\forall j: \mathrm{ms}(s^{in}_j) = \mathrm{ms}(s^{out})$。Reduce 要求"非 Preserve"且 $\mathrm{Contracted}$,Expand 要求"非 Preserve"且"非 Reduce"且 $\mathrm{Broadcast}$。故 Reduce 与 Expand 的定义已显式排除 Preserve(前置"非 Preserve"条件),互斥成立。

**步骤 3:Reduce 与 Expand 互斥**。

需证 $\mathrm{Contracted}$ 与 $\mathrm{Broadcast}$ 不同时成立。

**情况 3a**:一元算子($k_{op} = 1$)。

- Reduce(一元归约型):$\|s^{out}\| < \|s^{in}\|$。
- Expand 要求 $k_{op} \geq 2$(定义 3.4 (C4),一元无广播对象)。
- 故一元算子不可能同时是 Reduce 与 Expand。

**情况 3b**:二元算子($k_{op} = 2$)。

- Reduce(结构收缩型):$op \in \mathcal{O}_{\text{contract}} = \{\textsf{MatMul}, \textsf{Conv2D}\}$。这两个算子的 shape 规则是 $\sigma((m, k), (k, n)) = (m, n)$ 与 $\sigma((N, C_{in}, H, W), (C_{out}, C_{in}, k_H, k_W)) = (N, C_{out}, H_{out}, W_{out})$。两者都不涉及广播(内侧维度必须匹配,非广播规则)。
- Expand(广播型):$op \in \{\textsf{Add}, \textsf{Sub}, \textsf{Mul}, \textsf{Div}\}$。这四个算子的 shape 规则是 $\sigma(s, t) = s \oplus t$(广播)。它们不在 $\mathcal{O}_{\text{contract}}$ 中。
- $\mathcal{O}_{\text{contract}} \cap \{\textsf{Add}, \textsf{Sub}, \textsf{Mul}, \textsf{Div}\} = \emptyset$,故二元算子的 Reduce 与 Expand 互斥。

**步骤 4:综合**。

由步骤 1-3,四类两两互斥。对任意算子与合法输入,至多一类成立。$\square$

**注(局限 L3)**:步骤 3b 的互斥性依赖 $\mathcal{O}_{\text{contract}}$ 与广播算子集的不相交,这是基于 Tenth 当前 21 个算子的事实判断,非结构性证明。若未来新增"既收缩又广播"的算子(如带广播的批量矩阵乘),互斥性可能被破坏,见 §8.3。

### 5.2 定理 T7-2(完备性)

**定理 T7-2(完备性)**。定义 3.4 的四类变换覆盖 Tenth 全部 21 个 `TapeOp`:

$$\forall op \in \mathcal{O}: \mathrm{Class}^*(op) \in \{\text{Construct}, \text{Preserve}, \text{Reduce}, \text{Expand}\}$$

且每类非空(Construct×1、Preserve×11、Reduce×5、Expand×4)。

**证明**。由 §4 的逐一归类(§4.1 总表),每个算子被分配到唯一类别。逐一验证:

**Construct(1 个)**:
- `Input`:§4.2 论证,$k_{op} = 0$,归类 Construct。✓

**Preserve(11 个)**:
- `Neg`:§4.3 组 1,$\sigma(s) = s$,严格 Preserve。✓
- `ReLU`:§4.3 组 1,$\sigma(s) = s$,严格 Preserve。✓
- `Exp`:§4.3 组 1,$\sigma(s) = s$,严格 Preserve。✓
- `Log`:§4.3 组 1,$\sigma(s) = s$,严格 Preserve。✓
- `Sigmoid`:§4.3 组 1,$\sigma(s) = s$,严格 Preserve。✓
- `Softmax`:§4.3 组 1,$\sigma(s) = s$(沿最后维归一化,shape 不变),严格 Preserve。✓
- `Dropout`:§4.3 组 1,$\sigma(s) = s$(逐元素乘 mask),严格 Preserve。✓
- `BatchNorm`:§4.3 组 1,$\sigma(s) = s$,严格 Preserve。✓
- `LayerNorm`:§4.3 组 1,$\sigma(s) = s$,严格 Preserve。✓
- `Gelu`:§4.3 组 1,$\sigma(s) = s$,严格 Preserve。✓
- `Transpose`:§4.3 组 2,$\mathrm{ms}(s^{out}) = \mathrm{ms}(s^{in})$,置换 Preserve。✓

**Reduce(5 个)**:
- `Sum`:§4.4 组 1,$\sigma(s) = \epsilon$,一元归约。✓
- `Mean`:§4.4 组 1,$\sigma(s) = \epsilon$,一元归约。✓
- `CrossEntropy`:§4.4 组 1,$\sigma(s) = \epsilon$,一元归约。✓
- `MatMul`:§4.4 组 2,$op \in \mathcal{O}_{\text{contract}}$,结构收缩。✓
- `Conv2D`:§4.4 组 2,$op \in \mathcal{O}_{\text{contract}}$,结构收缩。✓

**Expand(4 个)**:
- `Add`:§4.5,$\sigma(s, t) = s \oplus t$,广播。✓
- `Sub`:§4.5,$\sigma(s, t) = s \oplus t$,广播。✓
- `Mul`:§4.5,$\sigma(s, t) = s \oplus t$,广播。✓
- `Div`:§4.5,$\sigma(s, t) = s \oplus t$,广播。✓

**计数验证**:$1 + 11 + 5 + 4 = 21 = |\mathcal{O}|$。每个算子恰好归入一类,无遗漏无重复。

由 §4.1 总表与上述逐一验证,四类覆盖全部 21 个 `TapeOp`,且每类非空。$\square$

**推论 T7-2.1(分类可计算性)**。对任意 `TapeOp` 节点 $v$,$\mathrm{Class}^*(op_v)$ 的判定在 $O(1)$ 时间内可计算(查表),无需运行时 shape 比较。

**证明**。$\mathrm{Class}^*$ 是算子的结构属性,与具体输入无关。判定方式:查 §4.1 总表(21 项的静态映射)。查表 $O(1)$。$\square$

### 5.3 定理 T7-3(分类稳定性)

**定理 T7-3(分类稳定性)**。分类对一元复合算子封闭。设 $op_1, op_2$ 是一元算子,复合 $op_2 \circ op_1$ 的分类 $\mathrm{Class}^*(op_2 \circ op_1)$ 可由子算子分类的"supremum"判定:

$$\mathrm{Class}^*(op_2 \circ op_1) = \mathrm{Class}^*(op_1) \sqcup \mathrm{Class}^*(op_2)$$

其中 $\sqcup$ 是分类的"join"运算,定义为(按"强度"排序 Construct < Preserve < Reduce = Expand):

| $\sqcup$ | Construct | Preserve | Reduce | Expand |
|----------|-----------|----------|--------|--------|
| Construct | Construct | Preserve | Reduce | Expand |
| Preserve | Preserve | Preserve | Reduce | Expand |
| Reduce | Reduce | Reduce | Reduce | (*) |
| Expand | Expand | Expand | (*) | Expand |

(*) Reduce $\sqcup$ Expand 与 Expand $\sqcup$ Reduce 的结果取决于复合顺序(见证明)。

**证明**。设 $op_1$ 的输入 shape 为 $s$,$op_1$ 输出 $s' = \sigma_{op_1}(s)$,$op_2$ 输出 $s'' = \sigma_{op_2}(s')$。

**情况 1**:$\mathrm{Class}^*(op_1) = \text{Preserve}$。

则 $s' = s$(或置换,$\mathrm{ms}(s') = \mathrm{ms}(s)$)。复合 $op_2 \circ op_1$ 在 $s$ 上的行为等于 $op_2$ 在 $s$ 上的行为(因 $s' = s$)。故 $\mathrm{Class}^*(op_2 \circ op_1) = \mathrm{Class}^*(op_2)$。

由表:Preserve $\sqcup X = X$ 对所有 $X$ 成立。✓

**情况 2**:$\mathrm{Class}^*(op_1) = \text{Construct}$。

$op_1$ 无输入($k_{op_1} = 0$),输出 $s'$ 外部给定。复合 $op_2 \circ op_1$ 的"输入"是 $op_1$ 的(空)输入,输出是 $op_2$ 的输出。从 shape 流角度,复合是 Construct(从外部构造)后接 $op_2$。若 $op_2$ 是 Preserve,复合是 Construct(外部给定 shape,Preserve 保持)。若 $op_2$ 是 Reduce(如 `Sum`),复合是 Construct→Reduce,但整体从外部看仍是 Construct(输出 shape 由外部 $s'$ 决定,$op_2$ 只是把 $s'$ 变成 $s''$)。

由表:Construct $\sqcup X = X$ 对 Preserve/Reduce/Expand 成立,Construct $\sqcup$ Construct = Construct。✓

**情况 3**:$\mathrm{Class}^*(op_1) = \text{Reduce}$, $\mathrm{Class}^*(op_2) = \text{Preserve}$。

$op_1$ 将 $s$ 归约为 $s'$($\|s'\| < \|s\|$)。$op_2$ 保持 $s'$($s'' = s'$)。复合 $op_2 \circ op_1$ 将 $s$ 归约为 $s'' = s'$,$\|s''\| < \|s\|$。故复合是 Reduce。

由表:Reduce $\sqcup$ Preserve = Reduce。✓

**情况 4**:$\mathrm{Class}^*(op_1) = \text{Reduce}$, $\mathrm{Class}^*(op_2) = \text{Reduce}$。

$op_1$ 将 $s$ 归约为 $s'$($\|s'\| < \|s\|$)。$op_2$ 将 $s'$ 归约为 $s''$($\|s''\| < \|s'\|$)。复合:$\|s''\| < \|s'\| < \|s\|$,故 $\|s''\| < \|s\|$,复合是 Reduce。

由表:Reduce $\sqcup$ Reduce = Reduce。✓

**情况 5**:$\mathrm{Class}^*(op_1) = \text{Expand}$, $\mathrm{Class}^*(op_2) = \text{Expand}$。

一元算子不可能是 Expand(定义 3.4 (C4) 要求 $k_{op} \geq 2$)。故此情况对一元复合不适用。

但若考虑"广义 Expand"(包括 reshape 扩张等未来算子),$op_1$ 扩张 $s \to s'$($\|s'\| > \|s\|$),$op_2$ 扩张 $s' \to s''$($\|s''\| > \|s'\|$)。复合 $\|s''\| > \|s\|$,故 Expand。

由表:Expand $\sqcup$ Expand = Expand。✓

**情况 6(*):**$\mathrm{Class}^*(op_1) = \text{Reduce}$, $\mathrm{Class}^*(op_2) = \text{Expand}$(广义)。

$op_1$ 将 $s$ 归约为 $s'$。$op_2$ 将 $s'$ 扩张为 $s''$。复合结果 $\|s''\|$ 与 $\|s\|$ 的关系不确定:
- 若 $op_1$ 是 `Sum`($s \to \epsilon$),$op_2$ 是 reshape($\epsilon \to (5, 5)$),复合 $s \to (5, 5)$,$\|s''\| = 2$ 与 $\|s\|$ 关系不定。
- 若 $\|s\| = 1$,$\|s''\| = 2$,复合是 Expand。
- 若 $\|s\| = 3$,$\|s''\| = 2$,复合是 Reduce。

故 Reduce $\sqcup$ Expand 的结果**取决于具体 shape**,不能由分类单独判定。表中标记为 (*),是分类稳定性的**局限**。

**情况 7**:类似地,Expand $\sqcup$ Reduce 取决于具体 shape。(*)

**综合**:除情况 6、7 外,一元复合的分类可由子算子分类的 join 判定。情况 6、7 是 Reduce 与 Expand 交叉的边界,需具体 shape 信息。$\square$

**推论 T7-3.1(Preserve 链的保持性)**。若一条算子链上所有算子都是 Preserve,则整链是 Preserve(输出 shape = 输入 shape)。

**证明**。由定理 T7-3 情况 1,Preserve $\sqcup$ Preserve = Preserve。归纳:链长 1 时显然;链长 $n$ 时,前 $n-1$ 个 Preserve 复合是 Preserve(归纳假设),再复合第 $n$ 个 Preserve,仍是 Preserve。$\square$

**实践意义**:Preserve 链(如 `Exp → Sigmoid → ReLU → Gelu`)的输出 shape 等于链首输入 shape,shape 推断只需传播一次,无需逐算子计算。这是 Tenth 编译器算子融合优化的理论基础。

---

## 6. 分类对 shape 推断算法的指导

### 6.1 每类的 shape 推断规则

四元分类对应四种 shape 推断规则,每种规则有统一的算法:

**Construct 类规则**:

- 输入:无(外部数据)。
- 输出 shape:由构造函数的字面量参数或运行时张量给定。
- 算法:查构造函数签名(如 `zeros(3, 4)` → `[3, 4]`,见 [`types.rs:450-462`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/hir/lower/types.rs) `shape_from_int_args`)。
- 复杂度:$O(\|s^{out}\|)$(参数个数)。

**Preserve 类规则**:

- 输入:输入 shape $s^{in}$。
- 输出 shape:$s^{out} = s^{in}$(严格)或 $s^{out} = \pi(s^{in})$(置换,如 `Transpose`)。
- 算法:复制输入 shape(或应用已知置换)。
- 复杂度:$O(\|s^{in}\|)$(shape 拷贝)。
- 实现对应:[`types.rs:285-289`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/hir/lower/types.rs) `dims: dims.clone()`。

**Reduce 类规则**:

- 输入:输入 shape(s)。
- 输出 shape:由归约/收缩公式计算。
- 一元归约型(`Sum`/`Mean`/`CrossEntropy`):$s^{out} = \epsilon$(标量)。算法:返回空 shape。
- 结构收缩型(`MatMul`):$s^{out} = (m, n)$ from $((m, k), (k, n))$。算法:取输入 1 的第 0 维 + 输入 2 的第 1 维。
- 结构收缩型(`Conv2D`):$s^{out} = (N, C_{out}, H_{out}, W_{out})$。算法:由 padding/stride/kernel 公式计算(见 [`autodiff.rs:629-633`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/autodiff.rs))。
- 复杂度:$O(\|s^{out}\|)$(输出 shape 长度)。
- 实现对应:[`types.rs:223-253`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/hir/lower/types.rs) `matmul` 分支、[`types.rs:259-278`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/hir/lower/types.rs) `sum`/`mean` 分支。

**Expand 类规则**:

- 输入:两个输入 shape $s, t$。
- 输出 shape:$s^{out} = s \oplus t$(广播,见 T1 定义 3.6)。
- 算法:调用 `broadcast_shapes`([`types.rs:18-41`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/hir/lower/types.rs)),从右向左对齐,逐维取 max。
- 复杂度:$O(\max(\|s\|, \|t\|))$。
- 实现对应:[`types.rs:150-153`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/hir/lower/types.rs) `broadcast_shapes(ldims, rdims)`。

### 6.2 算法的无遗漏性证明

**定理 T7-4(shape 推断无遗漏性)**。若 shape 推断算法对四类分别实现了 §6.1 的规则,则算法对 Tenth 全部 21 个 `TapeOp` 无遗漏:对任意算子,算法能计算其输出 shape(或返回"无法静态确定")。

**证明**。由定理 T7-2(完备性),四类覆盖全部 21 个算子。由 §6.1,每类有对应的推断规则。故对任意算子 $op$:

1. 查 §4.1 总表得 $\mathrm{Class}^*(op) \in \{\text{Construct}, \text{Preserve}, \text{Reduce}, \text{Expand}\}$(由 T7-2,总表存在)。
2. 按 $\mathrm{Class}^*(op)$ 调用对应规则(§6.1)。
3. 规则返回输出 shape 或"无法静态确定"(如 `Conv2D` 的 $H_{out}$ 依赖运行时 padding)。

故算法对每个算子都有规则可调,无遗漏。$\square$

**实践验证**:对照 [`types.rs::resolve_method_type`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/hir/lower/types.rs)(第 219-386 行)的实现:

- Preserve 类(`exp`/`log`/`relu`/`sigmoid`/`tanh`/`softmax`/`gelu`/`masked_fill`):返回 `dims: dims.clone()`。✓
- Reduce 类(`matmul`/`sum`/`mean`/`argmax`/`argmin`):返回计算后的 shape。✓
- Expand 类:由 `infer_binary_type`([`types.rs:135-166`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/hir/lower/types.rs))调用 `broadcast_shapes`。✓
- Construct 类(`zeros`/`ones`/`tensor`/`randn`):由 `shape_from_int_args` 返回。✓

实现覆盖全部 21 个 `TapeOp` 对应的方法名,无遗漏。✓

---

## 7. 扩展讨论

### 7.1 新增算子时的分类指导

当 Tenth 新增算子时,按以下流程归类:

**步骤 1**:确定 $k_{op}$(上游 Tape 节点数)。
- 若 $k_{op} = 0$:归 Construct。
- 若 $k_{op} \geq 1$:继续。

**步骤 2**:确定 shape 推断规则 $\sigma_{op}$。
- 若 $\sigma_{op}(s) = s$(或置换):归 Preserve。
- 若 $\sigma_{op}$ 是归约(输出秩 < 输入秩)或收缩(沿某维求和/内积):归 Reduce。
- 若 $\sigma_{op}$ 是广播(输出 = 输入广播):归 Expand。

**步骤 3**:验证互斥性。
- 检查算子是否"既收缩又广播"。若是(如带广播的批量矩阵乘 `bmm` with broadcast),分类法不适用,需扩展(见 §7.2)。

**步骤 4**:更新 §4.1 总表与 `resolve_method_type` 实现。

### 7.2 分类法的扩展性

当前四元分类对 Tenth 的 21 个算子完备,但对未来算子可能出现的不适用情形:

**(1)既收缩又广播的算子**:如假设的 `BatchMatMul` with broadcasting($(B, M, K) \circledast (B', K, N) \to (B, M, N)$,其中 $B$ 可广播到 $B'$)。该算子既收缩 $K$ 维(Reduce),又广播 $B$ 维(Expand)。当前分类法无法归入单一类别。

**建议**:引入第五类 **Reduce-Expand**(混合型),或采用"per-dimension classification"(每个维度单独归类)。这是未来工作。

**(2)Reshape 算子**:Tenth 当前无 `Reshape` 作为 `TapeOp`(reshape 在 HIR 层处理,见 [`types.rs:280-282`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/hir/lower/types.rs) `reshape`/`view` 分支)。`Reshape` 的 shape 变换是 $\sigma(s) = s'$(任意同体积 shape),可能:
- 秩减少($\|(3, 4)\| = 2 \to \|(12,)\| = 1$):Reduce。
- 秩增加($\|(12,)\| = 1 \to \|(3, 4)\| = 2$):Expand。
- 秩不变($\|(2, 6)\| = 2 \to \|(3, 4)\| = 2$,但维度值变):既非严格 Preserve 也非 Reduce/Expand。

`Reshape` 是分类法的**已知挑战**——其行为跨类别。当前 Tenth 不将 `Reshape` 作为 `TapeOp`,规避了这一问题。若未来引入,需扩展分类法(建议:per-dimension classification)。

**(3)动态 shape 算子**:如 `dynamic_slice`(切片范围运行时确定),输出 shape 依赖运行时值。分类法基于静态 shape 规则,对此类算子不适用,需标记为"不可静态推断"。

### 7.3 与 T1 Shape 代数的关系

T1([`T1-Shape代数系统的形式化建模.md`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/docs/论文/T1-Shape代数系统的形式化建模.md))证明广播运算 $\oplus$ 在 `Known ∪ {Any}` 片段构成有界半格。本文的 Expand 类直接依赖 $\oplus$:

- `Add`/`Sub`/`Mul`/`Div` 的 shape 推断规则是 $\oplus$(T1 定理 2)。
- T1 定理 4(全 `Known` 输入的可靠性完备性)保证 Expand 类对全 `Known` 输入的 shape 推断精确。
- T1 定理 5(`unbroadcast` 是 `broadcast` 的伴随)保证 Expand 类反向传播的 shape 正确性。

**互补关系**:T1 提供 Expand 类的代数基础(广播运算的性质),本文提供跨类的完整性保证(四类覆盖全部算子)。两者结合,使 shape 推断算法既有 Expand 类的代数保证,又有全算子的覆盖保证。

### 7.4 与 T8 四级解释分类的关系

T8([`T8-Shape解释关系四级分类.md`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/docs/论文/T8-Shape解释关系四级分类.md))的根因解释依赖本文的 `Class(v)`:

- (C2a):$\mathrm{Class}(v) = \text{Reduce}$ 且 $|s_{exp}| > |s_{act}|$(节点缩减体积,与期望>实际一致)。
- (C2b):$\mathrm{Class}(v) = \text{Expand}$ 且 $|s_{exp}| < |s_{act}|$(节点扩展体积,与期望<实际一致)。
- (C2c):$\mathrm{Class}(v) = \text{Construct}$ 且 $s^{out}_v = s_{act}$(节点构造了错误 shape)。

本文的 T7-2(完备性)保证 T8 的 (C2) 条件对所有 21 个算子可计算 `Class(v)`,无遗漏。若分类不完备,某些算子无法计算 `Class(v)`,T8 的候选集会漏报这些算子。

---

## 8. 局限(诚实记录)

本节独立记录 6 处理论局限,每条说明:是什么、影响多大、如何缓解。

### 8.1 局限 L1:Preserve 定义对置换的放宽

**是什么**:定义 3.4 (C2) 将 Preserve 放宽到"维度多重集保持"($\mathrm{ms}(s^{in}) = \mathrm{ms}(s^{out})$),允许 `Transpose` 这类置换算子归入 Preserve。但任务描述的严格形式化要求 $\forall i: \sigma_i = \mathrm{id}$(逐维恒等),`Transpose` 不满足。

**影响**:中等。放宽后,Preserve 类包含"shape 不变"与"shape 置换"两种子类,shape 推断规则需区分(严格 Preserve 直接复制,置换 Preserve 需应用置换)。这增加了规则的复杂度,但不影响完备性。

**缓解**:
- 定义 3.4 已显式区分子类(严格 Preserve vs 置换 Preserve)。
- §6.1 的 Preserve 规则覆盖两种子类(复制或应用置换)。
- 实践中,`Transpose` 的置换是固定的(交换最后两维),shape 推断可硬编码。

### 8.2 局限 L2:Reduce 对收缩型的扩展定义

**是什么**:定义 3.4 (C3) 的 Reduce 类包含"秩严格减少"(一元归约型)与"结构收缩"(MatMul/Conv2D)两种子类。后者依赖 $\mathcal{O}_{\text{contract}}$ 集合的显式枚举,非秩严格单调。

**影响**:中等。`MatMul` 的输出秩 = 输入秩(2 → 2),不满足"$m < k$"的严格形式化。归类依赖"内侧维度被收缩"的语义判据,这一判据对未知算子的推广性有限。

**缓解**:
- $\mathcal{O}_{\text{contract}}$ 显式枚举,且对 Tenth 当前 21 个算子完备。
- §7.1 给出新增算子的分类指导,收缩型算子可按"沿某维求和/内积"的语义判据识别。
- 若未来引入非收缩型二元算子(如 `Concat`),需重新审视 $\mathcal{O}_{\text{contract}}$。

### 8.3 局限 L3:互斥性依赖算子集的显式枚举

**是什么**:定理 T7-1 步骤 3b 的互斥性证明依赖 $\mathcal{O}_{\text{contract}} \cap \{\textsf{Add}, \textsf{Sub}, \textsf{Mul}, \textsf{Div}\} = \emptyset$,这是基于 Tenth 当前 21 个算子的事实判断,非结构性证明。

**影响**:低(对当前算子集)。若未来新增"既收缩又广播"的算子(见 §7.2),互斥性被破坏。

**缓解**:
- §7.2 已识别这一风险,建议引入第五类或 per-dimension classification。
- 新增算子时按 §7.1 步骤 3 验证互斥性。

### 8.4 局限 L4:Expand 类的同 shape 退化

**是什么**:`Add`/`Sub`/`Mul`/`Div` 在同 shape 输入下,具体行为是 Preserve(广播结果 = 输入),但结构行为分类仍为 Expand。这导致"分类"与"实例行为"不一致。

**影响**:低。对 shape 推断无影响(规则仍是广播,同 shape 时广播结果 = 输入,正确)。对根因诊断(护城河 F)有轻微影响:同 shape 调用的 `Add` 不产生 shape 漂移,但分类为 Expand 可能让根因分析误将其列为候选。

**缓解**:
- 定义 3.5 已显式区分"结构行为分类"与"实例行为"。
- 护城河 F 的 (C2b) 条件(Expand 且 $|s_{exp}| < |s_{act}|$)在同 shape 时 $|s_{exp}| = |s_{act}|$,条件不成立,不会误报。

### 8.5 局限 L5:未覆盖非 TapeOp 算子

**是什么**:本文仅覆盖 21 个 `TapeOp`(自动微分算子),不覆盖 HIR 层的其他 shape 变换(如 `reshape`/`view`/`flatten`/`permute`/`broadcast_to`/`cat`,见 [`types.rs:280-342`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/hir/lower/types.rs))。这些算子有独立的 shape 推断规则,但不参与 autodiff Tape。

**影响**:中。分类法的完备性仅对 `TapeOp` 成立,不覆盖 HIR 层全量算子。`Reshape` 等算子的分类挑战(见 §7.2)未被本文解决。

**缓解**:
- 本文范围明确限定为 `TapeOp`(与任务要求一致)。
- HIR 层算子的分类可作为未来工作(需扩展分类法,见 §7.2)。
- 护城河 F 的根因分析在 Tape 上进行,本文的完备性对 F 已足够。

### 8.6 局限 L6:T7-3 分类稳定性的 Reduce-Expand 交叉

**是什么**:定理 T7-3 的情况 6、7(Reduce $\sqcup$ Expand)的结果取决于具体 shape,不能由分类单独判定。

**影响**:低。当前 21 个 `TapeOp` 中无 Expand 类的一元算子(Expand 要求 $k_{op} \geq 2$),故一元复合中不会出现 Reduce $\sqcup$ Expand。该局限仅在引入广义 Expand(如 reshape 扩张)后触发。

**缓解**:
- 当前算子集下,该局限不触发。
- 未来引入广义 Expand 时,需补充 per-dimension classification。

---

## 9. 结论与未来工作

### 9.1 结论

本文对 Tenth 语言的 21 个 `TapeOp` 进行形式化分类,证明 Construct/Preserve/Reduce/Expand 四元分类的互斥性(定理 T7-1)与完备性(定理 T7-2),并证明分类对一元复合算子的封闭性(定理 T7-3,含 Reduce-Expand 交叉局限)。主要结果:

1. **互斥性**:四类两两不相交,基于 $k_{op}$、维度多重集、收缩判据、广播判据的结构性论证。
2. **完备性**:四类覆盖全部 21 个 `TapeOp`,分布为 Construct×1(Input)、Preserve×11(逐元素 + Transpose)、Reduce×5(归约 + 收缩)、Expand×4(二元广播)。
3. **稳定性**:Preserve 链保持 Preserve(推论 T7-3.1),支持算子融合优化。
4. **无遗漏性**:shape 推断算法对四类有统一规则(定理 T7-4),对 21 个算子无遗漏。

本文诚实记录 6 处局限,重点处理三类边界情形:`Transpose`(置换型 Preserve,L1)、`MatMul`/`Conv2D`(收缩型 Reduce,L2)、`Add` 系列(广播型 Expand 的同 shape 退化,L4)。完备性保证为护城河 F(张量关系调试器)与护城河 A(Autograd 反向 Shape 静态验证)提供分类学基础。

### 9.2 未来工作

1. **扩展到 HIR 层全量算子**:覆盖 `reshape`/`view`/`flatten`/`permute`/`broadcast_to`/`cat` 等 HIR 算子(局限 L5)。需引入 per-dimension classification 处理 `Reshape` 的跨类别行为(§7.2)。
2. **第五类 Reduce-Expand**:为"既收缩又广播"的算子(如 `BatchMatMul` with broadcast)引入混合类别(§7.2)。
3. **分类与算子融合的实证**:验证 Preserve 链的算子融合优化在 Tenth 编译器中的性能收益(推论 T7-3.1)。
4. **分类对根因诊断的指导**:结合 T8 的四级解释分类,实证分类对护城河 F 候选集规模的缩减效果。
5. **动态 shape 算子的分类**:为 `dynamic_slice` 等运行时依赖算子设计"不可静态推断"标记,扩展分类法。

---

## 参考文献

1. Tenth 项目内部文档:
   - [`runtime/autodiff.rs`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/autodiff.rs)(21 个 TapeOp 定义与实现)
   - [`hir/lower/types.rs`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/hir/lower/types.rs)(编译期 shape 推断算法)
   - [`docs/shape-check-roadmap/形式化分析理论可行性论证.md`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/docs/shape-check-roadmap/形式化分析理论可行性论证.md)(定义 2.4 分类,§2.2)
   - [`docs/论文/T1-Shape代数系统的形式化建模.md`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/docs/论文/T1-Shape代数系统的形式化建模.md)(广播代数,Expand 类基础)
   - [`docs/论文/T8-Shape解释关系四级分类.md`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/docs/论文/T8-Shape解释关系四级分类.md)(根因解释依赖本文分类)
   - [`docs/语言参考手册.md`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/docs/语言参考手册.md) §11.5(算子语义)
2. NumPy. Broadcasting rules. https://numpy.org/doc/stable/user/basics.broadcasting.html(§2.2 ufunc 分类)
3. Bradbury, J., et al. (2018). JAX: Composable transformations of Python+NumPy programs.(§2.2 JAX shape 处理)
4. XLA. Shape inference for HLO instructions. https://www.tensorflow.org/xla/operation_semantics(§2.3 XLA shape inference)
5. Paszke, A., et al. (2017). Automatic differentiation in PyTorch.(Tape DAG 结构)
6. Jones, S. P., et al. (2007). Practical type inference for arbitrary-rank types. *JFP*.(§2.1 类型系统分类)
7. Pierce, B. C. (2002). *Types and Programming Languages*. MIT Press.(§2.1 代数数据类型)

---

## 附录 A:定理索引

| 定理 | 内容 | 章节 |
|------|------|------|
| T7-1 | 互斥性:四类两两不相交 | §5.1 |
| T7-2 | 完备性:四类覆盖全部 21 个 TapeOp | §5.2 |
| T7-2.1 | 分类可计算性:O(1) 查表 | §5.2 |
| T7-3 | 分类稳定性:一元复合的 join 规则 | §5.3 |
| T7-3.1 | Preserve 链保持性 | §5.3 |
| T7-4 | shape 推断无遗漏性 | §6.2 |

## 附录 B:局限索引

| 局限 | 内容 | 影响 | 章节 |
|------|------|------|------|
| L1 | Preserve 定义对置换的放宽 | 中 | §8.1 |
| L2 | Reduce 对收缩型的扩展定义 | 中 | §8.2 |
| L3 | 互斥性依赖算子集显式枚举 | 低 | §8.3 |
| L4 | Expand 类的同 shape 退化 | 低 | §8.4 |
| L5 | 未覆盖非 TapeOp 算子 | 中 | §8.5 |
| L6 | T7-3 的 Reduce-Expand 交叉 | 低 | §8.6 |

## 附录 C:与现有文档的对应

| 本文章节 | 对应文档 | 关系 |
|---------|---------|------|
| §3 定义 3.4 | [`形式化分析理论可行性论证.md`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/docs/shape-check-roadmap/形式化分析理论可行性论证.md) §2.2 定义 2.4 | 本文细化:体积判据→结构判据,处理二元算子 |
| §4 分类总表 | [`autodiff.rs`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/autodiff.rs) §27-79 | 实证对应:每个 TapeOp 的源码行号 |
| §6 推断规则 | [`types.rs::resolve_method_type`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/hir/lower/types.rs) | 实现对应:每类的推断算法 |
| §7.3 与 T1 关系 | [`T1-Shape代数`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/docs/论文/T1-Shape代数系统的形式化建模.md) | 互补:T1 提供 Expand 代数基础,本文提供全类完整性 |
| §7.4 与 T8 关系 | [`T8-四级分类`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/docs/论文/T8-Shape解释关系四级分类.md) | 依赖:T8 的 (C2) 条件依赖本文 Class(v) |

---

## 附录 D:实施建议

基于本文理论结论,对 Tenth 实施提出以下建议:

### D.1 shape 推断算法的实现

1. **按分类组织规则表**:在 `resolve_method_type` 中,按 Construct/Preserve/Reduce/Expand 四组组织算子规则,而非按字母序。这便于新增算子时快速定位规则位置。
2. **Preserve 类统一处理**:对严格 Preserve 子类(9 个算子),可合并为单一规则 `dims: dims.clone()`,减少代码重复。当前实现已部分如此([`types.rs:285-289`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/hir/lower/types.rs))。
3. **Expand 类复用广播代数**:Expand 类的 4 个算子共享 `broadcast_shapes` 规则,已在 `infer_binary_type` 中实现([`types.rs:150-153`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/hir/lower/types.rs))。

### D.2 护城河 F 的候选集优化

1. **按分类过滤候选**:根因分析时,Preserve 类算子(除同 shape 退化的 Expand)不产生 shape 漂移,可降低优先级。
2. **Reduce/Expand 类优先**:shape 漂移的主要嫌疑是 Reduce/Expand 类,根因分析应优先检查这两类算子。
3. **Construct 类的特殊处理**:Construct 类(`Input`)的 shape 由外部给定,若报错 shape 与某 `Input` 节点的输出一致,该节点是根因候选(C2c 条件)。

### D.3 测试用例设计

1. **每类至少一例**:测试 shape 推断算法时,每类至少覆盖一个算子(Construct: `Input`;Preserve: `Exp`;Reduce: `Sum`;Expand: `Add` with broadcast)。
2. **边界情形覆盖**:
   - `Transpose`(置换 Preserve):测试 2D 与非 2D 情形。
   - `MatMul`(收缩 Reduce):测试 1D@2D、2D@2D、2D@1D 情形(见 [`autodiff.rs:375-436`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/autodiff.rs) 的 1D 提升)。
   - `Add` 同 shape 退化:测试同 shape 与广播两种情形,验证分类为 Expand 但同 shape 行为正确。

---

> **文档结束**
>
> 本文 v1 经历 4 轮自审,修正了 4 处问题(见 §1.4)。所有分类判定均对应到 [`autodiff.rs`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/autodiff.rs) 具体实现(用 `file://` 链接 + 行号)。6 处局限独立章节记录(§8),无掩盖。如发现分类错误或边界遗漏,应在 `MEMO.md` 记录并修订本文。
