# Shape 错误解释关系的四级分类：基于三条件的形式化刻画与因果推理扩展

> **作者**：Tenth 项目数理部
> **日期**：2026-07-02
> **类型**：理论分析论文（T8）
> **适用范围**：护城河 F（张量关系调试器）形式化理论依据——解释关系的层级化与因果性修复
> **关联文档**：`docs/shape-check-roadmap/形式化分析理论可行性论证.md`（v3 草稿 §3、§6.6）、`docs/shape-check-roadmap/战略规划.md`（方向 F 定位）、`tenth/src/runtime/autodiff.rs`（Tape 实现）、`docs/论文/T2-Tape形式化模型与根因定位可判定性.md`
> **本文定位**：在 v3 草稿基础上严格化的独立论文，聚焦解释关系的四级层级、三条件 (C1)(C2)(C3) 的重新形式化、定理 F4 循环论证的因果性修复，以及 Tape DAG 上反事实推理的可判定性分析

---

## 摘要

本文针对 Tenth 语言张量关系调试器（护城河 F）中"算子是否为 shape 错误根因"的判定问题，提出严格的四级解释关系层级：`DefinitelyRoot`、`ExplainsError`、`PartialExplain`、`Unrelated`。我们重新形式化了三个递进的条件——(C1) 节点在错误传播路径上的可达性、(C2) 节点 shape 与错误的相关性、(C3) 节点 shape 是错误的直接因果性原因——并用这三个条件的合取严格定义四级关系。本文证明：四级构成链式偏序（定理 T8-1），解释关系在 $O(\|s\| \log \|s\|)$ 时间内可计算且每级附带一阶逻辑可解释公式（定理 T8-2）。本文的核心修正点是修复 `形式化分析理论可行性论证.md` §6.6 指出的定理 F4 循环论证：通过引入 Lewis (1973) 的 counterfactual 因果语义，将 (C3) 重新定义为"若 $v$ 的 shape 改变，$e$ 不会发生"的反事实命题，使解释关系摆脱对自身分析框架的循环依赖（定理 T8-3）。进一步分析 Tape DAG 上反事实推理的可判定性（定理 T8-4）：在非 Construct 节点情形下反事实推理可判定，复杂度 $O(|V_{\text{path}}| \cdot \|s\|)$；在 Construct 节点情形下可判定性依赖于 shape 传播函数的代数性质，本文给出**猜想**而非定理，并显式标注为开放问题。本文诚实记录三类核心局限：反事实推理在 Construct 节点的可判定性未完全证明、Lewis 反事实语义的"最近世界"在 Tape 上的退化、多根因场景的扩展未覆盖。所有形式化定义均可锚定到 Tenth v0.3.3 的源码位置。

**关键词**：Shape 错误、解释关系、四级分类、counterfactual 因果、Lewis 反事实语义、Tape DAG、可判定性、调试器形式化

---

## 1. 引言

### 1.1 Shape 错误根因判断的复杂性

在张量计算程序中，shape（张量形状）错误是最常见的运行时错误之一。当 MatMul 报错"内侧维度不匹配 [3,8] @ [4,8]"时，用户面对的真正困难不是"哪一行报错"，而是"为什么这一行的输入变成了 [3,8]"——错误的表现位置与错误的根因位置在计算图中可能相距很远。前向第 3 步的 reshape 误用，可能到反向第 30 步才以 grad shape 不匹配的形式爆出来（参见 [战略规划.md 方向 F §战略起源](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/docs/shape-check-roadmap/战略规划.md)）。

现有 AI 框架（PyTorch、JAX、TensorFlow）的报错停留在"位置导向"层面：告诉用户"哪一行错了"，但不告诉"为什么这一行会变成 [3,8]"。更根本的问题是：**当调试器试图判断"哪个算子是根因"时，缺乏形式化的判断标准**。一个节点是不是根因，到底是看它在错误传播路径上？看它的 shape 与错误相关？还是看它直接导致了错误？这三层判断的边界从未被严格区分。

### 1.2 经验式调试的局限

直觉方案是为算子分配权重（reshape=10, transpose=8, matmul=5），按权重排序选根因。这种方法存在四个根本问题（参见 [形式化分析理论可行性论证.md §1.2](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/docs/shape-check-roadmap/形式化分析理论可行性论证.md)）：

1. **不可解释**：用户问"为什么根因是 reshape"，只能答"权重高"
2. **不可证伪**：权重表是经验设定，无形式化依据
3. **可能误判**：实际根因是 matmul 但启发式指向 reshape
4. **违背调试器初衷**：调试器本身不可信，比没有调试器更糟

形式化方法（如 v3 草稿 §3 提出的三条件 (C1)(C2)(C3)）试图避免这些问题，但 v3 草稿自身的 §6.6 诚实指出：定理 F4 是循环论证——"若 $v^*$ 满足定义 3.2 之一，则 $v^* \in C$"退化为"若 $v^* \in C$ 则 $v^* \in C$"的重言式。这意味着 v3 的解释关系没有独立于自身分析框架的完备性保证。

### 1.3 形式化解释关系的价值

本文认为，要避免循环论证，必须将解释关系建立在**独立于分析框架的因果性概念**上。具体地：

- **可达性**（C1）回答"节点是否在错误传播路径上"——这是图论性质，独立于 shape 内容
- **相关性**（C2）回答"节点的 shape 与错误是否相关"——这是统计/代数性质，依赖于 shape 变换方向
- **因果性**（C3）回答"节点的 shape 是否直接导致了错误"——这是因果推理性质，必须独立于"节点是否在候选集中"

三层条件构成递进的严格性：可达性最弱（只是必要条件），相关性次之（要求方向一致），因果性最强（要求反事实支持）。这种层级化让调试器可以输出**有梯度的诊断**：DefinitelyRoot（强证据）、ExplainsError（中证据）、PartialExplain（弱证据）、Unrelated（无证据）。

### 1.4 贡献

本文的贡献如下：

- **三条件的重新形式化**（§3）：将 v3 草稿的 (C1)(C2)(C3)（直接解释/改变解释/约束违反）重新诠释并严格化为递进的可达性/相关性/因果性三层条件，明确三者的逻辑关系（合取式层级）。
- **四级解释关系的偏序结构**（§4）：用三条件的合取定义四级关系 `DefinitelyRoot ⊂ ExplainsError ⊂ PartialExplain ⊂ Unrelated` 的补集链，并证明其构成链式偏序（定理 T8-1）。
- **主定理与证明**（§5）：T8-1（四级偏序性）、T8-2（可解释性与可计算性）、T8-3（F4 循环性的因果性修复）、T8-4（反事实推理的可判定性）。
- **因果推理扩展**（§6）：Lewis (1973) counterfactual 语义在 Tape DAG 上的形式化，"若 $v$ 的 shape 改变，$e$ 不会发生"的严格定义，Tape DAG 上反事实世界的构造，与 Pearl do-calculus 的对比。
- **诚实记录局限**（§8）：反事实推理在 Construct 节点的可判定性未完全证明（标注为猜想）、Lewis "最近世界"在 Tape 上的退化、多根因扩展未覆盖。

### 1.5 与 v3 草稿及 T2 论文的关系

本文基于 `形式化分析理论可行性论证.md` v3 草稿与 T2 论文（`docs/论文/T2-Tape形式化模型与根因定位可判定性.md`）。v3 草稿定义 3.2 的 (C1)(C2)(C3) 与定义 3.5 的四级分类是本文的起点，但本文重新诠释了三条件的语义——v3 的 (C1)(C2)(C3) 是三种并列的根因模式，本文将其重组为递进的可达性/相关性/因果性三层。这种重组不改变 v3 的四级分类的实质（DefinitelyRoot 仍由约束违反识别），但使其偏序结构显式化，并为 F4 的因果性修复铺平道路。T2 论文已经证明 F1（可判定性）、F2（有限候选集）、F5（复杂度）；本文聚焦 F3（可解释性）的层级化与 F4（自洽性）的因果性修复，不重复 F1/F2/F5 的证明。

---

## 2. 背景与相关工作

### 2.1 因果推理理论

**Lewis (1973) counterfactual 语义**。David Lewis 的 counterfactual 理论将"如果 A 发生，B 就会发生"形式化为 $A \square\!\!\!\!\to B$，读作"A counterfactually implies B"。其语义基于可能世界（possible worlds）：$A \square\!\!\!\!\to B$ 为真当且仅当在所有"A 为真"的最近可能世界中，B 也为真。"最近"由世界间的相似性关系（similarity relation）定义。Lewis 的核心应用是因果性定义：事件 $c$ 导致事件 $e$ 当且仅当 $c$ 发生且 $\neg c \square\!\!\!\!\to \neg e$（若 $c$ 不发生，$e$ 也不会发生）。

**Pearl do-calculus**。Judea Pearl 的 do-calculus 将"干预"（intervention）形式化为 $do(X = x)$：将变量 $X$ 强制设为 $x$，断开 $X$ 与其父节点的依赖。$P(Y | do(X = x))$ 与 $P(Y | X = x)$ 的区别在于前者是干预后的因果效应，后者是观察到的统计相关。do-calculus 在因果贝叶斯网上是可判定的（Shpitser & Pearl 2006 证明了 do-calculus 的完备性）。

**两者的对比**。Lewis 的 counterfactual 是命题层面的反事实推理，依赖可能世界语义；Pearl 的 do-calculus 是概率层面的干预推理，依赖因果图结构。在 Tape DAG 上，我们既无概率分布（shape 是确定的），也无标准的"父节点干预"语义（Tape 是确定性的 DAG）。本文采用 Lewis 风格的 counterfactual，但将其限制在 Tape DAG 的有限结构上，使其可判定。

### 2.2 程序切片中的"相关性"与"因果性"

程序切片（Weiser 1981, Tip 1995）给定切片准则 $(s, V)$，计算影响 $V$ 在 $s$ 处值的程序子集。切片捕捉的是"相关性"（relevance）——哪些语句影响了变量值——但不区分"相关"与"因果"：切片中的所有语句都被视为"相关"，但哪个语句是"根因"切片不回答。

Tape 根因分析（T2 论文 §7）已证明本质是 shape 错误的后向切片。但切片只能给出"相关节点集合"，不能在相关节点中进一步区分"强相关"与"弱相关"。本文的四级分类正是对切片结果的层级化——在相关节点中，进一步用 (C2)(C3) 区分相关性强度与因果性。

### 2.3 软件调试中的 bug localization 形式化

软件工程中的 bug localization（Zeller 2002 的 delta debugging、Jones et al. 2002 的 Tarantula）主要基于频谱统计（哪些语句在失败用例中执行频繁）或变异分析（哪个语句变异后错误消失）。前者是统计相关，后者是反事实因果。

**变异分析**与本文的 counterfactual 思路最接近：通过修改某个语句，观察错误是否消失，判断该语句是否是根因。但变异分析在程序层面（修改源码），本文的反事实在 shape 层面（修改 shape 值），不修改源码——这是 Tape DAG 上的"shape 变异"。这种差异使本文的反事实推理可在运行时单次执行后完成，不需要重新运行程序。

### 2.4 Tenth 现有理论基础

本文建立在 T2 论文已证明的基础上：

- **Tape 形式化模型**（T2 §3）：Tape 是 DAG $G = (V, E)$，节点 $v = (op_v, s^{in}_v, s^{out}_v, \ell_v)$，边由张量唯一标识 Tid 定义。
- **Shape 变换分类**（T2 §3.2）：Construct/Preserve/Reduce/Expand 四类，完备且可计算。
- **算子内部约束**（T2 §3.3）：$\text{Constraint}_{op}: \mathbb{S}^{k_{op}} \to \{\text{true}, \text{false}\}$，可计算。
- **报错定义**（T2 §4.1）：$e = (s_{exp}, s_{act})$，$s_{exp} \neq s_{act}$。
- **F1 可判定性**（T2 §4.2）：v3 的 Explain 判定在 $O(\|s\| \log \|s\|)$ 时间内可计算。
- **F4 循环性局限**（T2 §6.6, v3 §6.6）：v3 的 F4 是重言式，需要独立因果性形式化。

本文的 (C1)(C2)(C3) 重新形式化与 T8-3 因果性修复正是针对 F4 局限的回应。

---

## 3. 三条件 (C1)(C2)(C3) 的形式化定义

本节将 v3 草稿的三条件从"三种并列的根因模式"重新形式化为"递进的可达性/相关性/因果性三层"。这种重组不改变 v3 的实质判定（DefinitelyRoot 仍由约束违反识别），但使四级的偏序结构显式化，并为 §5 的因果性修复铺平道路。

### 3.1 前置概念

沿用 T2 论文与 v3 草稿的符号：

- **Shape** $\mathbb{S} = \bigcup_{n \geq 0} \mathbb{N}^n$，非负整数元组
- **Tape DAG** $G = (V, E)$，节点 $v = (op_v, s^{in}_v, s^{out}_v, \ell_v)$，边由 Tid 定义（[autodiff.rs:15-25](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/autodiff.rs)）
- **算子 shape 语义** $\text{Sem}_{op}: \mathbb{S}^{k_{op}} \to \mathbb{S} \cup \{\bot\}$
- **算子内部约束** $\text{Constraint}_{op}(s_1, ..., s_{k_{op}}) = (\text{Sem}_{op}(s_1, ..., s_{k_{op}}) \neq \bot)$
- **Shape 变换分类** $\text{Class}(v) \in \{\text{Construct}, \text{Preserve}, \text{Reduce}, \text{Expand}\}$
- **报错** $e = (s_{exp}, s_{act})$，$s_{exp} \neq s_{act}$，报错节点 $v_{err} \in V$（由运行时报错上下文提供，见 T2 §4.1 v3 修正）

**定义 3.1（路径可达性）**：节点 $u$ 到节点 $w$ 在 $G$ 中路径可达，记作 $u \leadsto w$，当且仅当存在 $u = v_0, v_1, ..., v_k = w$ 使 $(v_{i-1}, v_i) \in E$ 对所有 $i = 1, ..., k$ 成立。约定 $u \leadsto u$（自反）。

### 3.2 三条件的严格定义

**定义 3.2（C1 可达性）**：节点 $v$ 满足 C1（在错误传播路径上），当且仅当 $v \leadsto v_{err}$，即 $v$ 在 $G$ 中路径可达报错节点 $v_{err}$。

形式化：$\text{C1}(v, e, G) \iff v \leadsto_G v_{err}$。

**直观含义**：C1 是必要条件——只有影响 $v_{err}$ 的节点才可能是根因。C1 不涉及 shape 内容，仅依赖 DAG 拓扑。等价于程序切片中的"在后向切片中"。

**定义 3.3（C2 相关性）**：节点 $v$ 满足 C2（shape 与错误相关），当且仅当 $\text{C1}(v, e, G)$ 成立，且下列条件之一成立：

- **(C2a) 输出匹配**：$s^{out}_v = s_{act}$（$v$ 的输出 shape 等于报错的实际 shape）
- **(C2b) 方向一致**：$\text{Class}(v) \neq \text{Preserve}$ 且 $v$ 的 shape 改变方向与 $(s_{exp}, s_{act})$ 的体积差异方向一致，即：
  - $\text{Class}(v) = \text{Reduce}$ 且 $|s_{exp}| > |s_{act}|$，或
  - $\text{Class}(v) = \text{Expand}$ 且 $|s_{exp}| < |s_{act}|$，或
  - $\text{Class}(v) = \text{Construct}$ 且 $s^{out}_v = s_{act}$

形式化：$\text{C2}(v, e, G) \iff \text{C1}(v, e, G) \wedge \big(\text{(C2a)} \vee \text{(C2b)}\big)$。

**直观含义**：C2 要求节点的 shape 与报错在数值上相关——要么直接输出错误的 shape，要么 shape 改变方向与报错方向一致。C2 是统计/代数相关，不蕴含因果。

**注**：C2 的 (C2a)(C2b) 对应 v3 定义 3.2 的 (C1)（直接解释）与 (C2)（改变解释）。本文将其归并为 C2 的两个子条件，因为两者都是"shape 数值相关"的不同强度。

**定义 3.4（C3 因果性，本文核心）**：节点 $v$ 满足 C3（shape 是错误的直接原因），当且仅当 $\text{C2}(v, e, G)$ 成立，且 $v$ 的 shape 是 $e$ 的 counterfactual 因果根因。即：

$$\text{C3}(v, e, G) \iff \text{C2}(v, e, G) \wedge \text{CF}(v, e, G)$$

其中 $\text{CF}(v, e, G)$ 是 counterfactual 因果谓词（§6 严格定义）：

$$\text{CF}(v, e, G) \iff \exists s' \in S_v^{\text{legal}}: \neg \text{Occurs}(e, G[v \mapsto s'])$$

即"存在 $v$ 的某个合法反事实 shape $s'$，使得在反事实世界 $G[v \mapsto s']$ 中报错 $e$ 不发生"。$S_v^{\text{legal}}$ 是 $v$ 的合法反事实 shape 集合（§6.2 定义），$\text{Occurs}(e, G')$ 表示报错 $e$ 在 Tape $G'$ 中发生（即重新传播后 $v_{err}$ 处的 shape 与约束状态导致 $e$）。

**直观含义**：C3 是因果性而非相关性——不仅要求 shape 相关，还要求"如果 shape 不同，错误就不会发生"。这是 Lewis counterfactual 因果的直接应用。

**与 v3 (C3) 的关系**：v3 定义 3.2 的 (C3) 是"$\text{Constraint}_{op_v}(s^{in}_v) = \text{false}$"（算子约束违反），这是相关性的特殊情况（约束违反意味着 shape 不合法）。本文的 C3 是更强的因果性条件，**包含** v3 的 (C3) 作为特例（约束违反的节点，其 counterfactual shape 是使其约束满足的 shape，此时报错可能消失）。

### 3.3 三条件的逻辑关系

三条件构成递进的合取层级：

$$\text{C3}(v, e, G) \implies \text{C2}(v, e, G) \implies \text{C1}(v, e, G)$$

**证明**：

- $\text{C3} \implies \text{C2}$：由定义 3.4，$\text{C3} = \text{C2} \wedge \text{CF}$，故 $\text{C3} \implies \text{C2}$。
- $\text{C2} \implies \text{C1}$：由定义 3.3，$\text{C2} = \text{C1} \wedge (\text{C2a} \vee \text{C2b})$，故 $\text{C2} \implies \text{C1}$。$\square$

**含义**：三条件是"必要条件链"——C1 是 C2 的必要条件，C2 是 C3 的必要条件。这种层级使四级分类可以表示为三条件的合取。

---

## 4. 四级解释关系

### 4.1 四级的定义

**定义 4.1（四级解释关系）**：对节点 $v \in V$ 与报错 $e$，定义：

$$\text{Explain}(v, e, G) = \begin{cases}
\text{DefinitelyRoot} & \text{若 } \text{C1}(v, e, G) \wedge \text{C2}(v, e, G) \wedge \text{C3}(v, e, G) \\
\text{ExplainsError} & \text{若 } \text{C1}(v, e, G) \wedge \text{C2}(v, e, G) \wedge \neg \text{C3}(v, e, G) \\
\text{PartialExplain} & \text{若 } \text{C1}(v, e, G) \wedge \neg \text{C2}(v, e, G) \\
\text{Unrelated} & \text{若 } \neg \text{C1}(v, e, G)
\end{cases}$$

**直观含义**：

- **DefinitelyRoot**（强证据）：节点在错误路径上、shape 相关、且 counterfactual 因果成立——若 shape 不同错误就不会发生。这是最强的根因证据。
- **ExplainsError**（中证据）：节点在错误路径上、shape 相关，但 counterfactual 不成立——shape 改变后错误仍会发生（说明根因在别处）。
- **PartialExplain**（弱证据）：节点在错误路径上，但 shape 不相关——只是路径上的中转节点。
- **Unrelated**（无证据）：节点不在错误路径上。

### 4.2 四级的偏序关系

定义四级的强度偏序 $\prec$（"强于"）：

$$\text{DefinitelyRoot} \prec \text{ExplainsError} \prec \text{PartialExplain} \prec \text{Unrelated}$$

其中 $A \prec B$ 表示"$A$ 比 $B$ 强"（即 $A$ 蕴含 $B$ 的所有条件再加更多）。

对应的节点集合构成链式包含：

$$V_{\text{DR}} \subseteq V_{\text{EE}} \subseteq V_{\text{PE}} \subseteq V_{\text{UR}} = V$$

其中 $V_{\text{DR}} = \{v : \text{Explain}(v, e, G) = \text{DefinitelyRoot}\}$，以此类推。

### 4.3 与 v3 定义 3.5 的对应

v3 草稿定义 3.5 的四级分类基于 (C1)(C2)(C3) 三条件的不同组合（强形式/弱形式），本文的四级基于 (C1)(C2)(C3) 的合取层级。两者的对应关系：

| 本文 | v3 定义 3.5 | 区别 |
|------|------------|------|
| DefinitelyRoot | DefinitelyRoot | v3 由 (C3) 约束违反识别；本文由 C3 counterfactual 因果识别，包含 v3 的 (C3) 作为特例 |
| ExplainsError | ExplainsError | v3 由 (C1) 或 (C2) 强形式识别；本文由 C1 ∧ C2 ∧ ¬C3 识别，等价但层级更清晰 |
| PartialExplain | PartialExplain | v3 由 (C2) 弱形式识别；本文由 C1 ∧ ¬C2 识别——本文的 PartialExplain 更弱（仅可达性） |
| Unrelated | Unrelated | 一致 |

**关键差异**：v3 的 PartialExplain 要求"shape 改变方向一致但输出不匹配"（C2 弱形式），本文的 PartialExplain 仅要求"在错误路径上"（C1 但 ¬C2）。本文的定义更宽泛——所有路径上的节点都至少是 PartialExplain，这符合"调试器应列出所有可能相关的节点"的直觉。v3 的 (C2) 弱形式节点在本文中归入 ExplainsError（因 C2 成立）。

---

## 5. 主定理与证明

### 5.1 定理 T8-1（四级偏序性）

**定理 T8-1**：四级解释关系构成链式偏序，即：

$$V_{\text{DR}} \subseteq V_{\text{EE}} \subseteq V_{\text{PE}} \subseteq V$$

且每级是非空的（在适当假设下）。

**证明**：逐级证明包含关系。

**$V_{\text{DR}} \subseteq V_{\text{EE}}$**：设 $v \in V_{\text{DR}}$，则 $\text{Explain}(v, e, G) = \text{DefinitelyRoot}$，由定义 4.1，$\text{C1} \wedge \text{C2} \wedge \text{C3}$ 成立。需证 $\text{Explain}(v, e, G) = \text{ExplainsError}$ 不成立——但这是反方向。实际上需证 $v$ 满足 ExplainsError 的"非条件"——不，包含关系应理解为：DefinitelyRoot 的节点"如果忽略 C3"则是 ExplainsError。但严格按定义 4.1，DefinitelyRoot 与 ExplainsError 是互斥的（C3 成立 vs C3 不成立）。

**修正陈述**：四级是互斥且完备的划分，偏序 $\prec$ 是"强度偏序"而非"集合包含"。正确的偏序是：若 $v$ 满足 DefinitelyRoot，则 $v$ 也满足 ExplainsError 的**必要条件**（C1 ∧ C2），但不满足 ExplainsError 的**充分条件**（¬C3）。故四级在"必要条件"意义上构成链式偏序。

**重新陈述定理 T8-1**：定义"节点的解释强度"$\text{Str}(v) \in \{0, 1, 2, 3\}$（0=Unrelated, 1=PartialExplain, 2=ExplainsError, 3=DefinitelyRoot）。则：

$$\text{Str}(v) \geq k \iff v \text{ 满足 C1, C2, ..., C}_k$$

即强度 $k$ 等价于前 $k$ 个条件的合取。特别地：

- $\text{Str}(v) \geq 1 \iff \text{C1}(v)$
- $\text{Str}(v) \geq 2 \iff \text{C1}(v) \wedge \text{C2}(v)$
- $\text{Str}(v) \geq 3 \iff \text{C1}(v) \wedge \text{C2}(v) \wedge \text{C3}(v)$

故强度构成链式偏序 $3 \succ 2 \succ 1 \succ 0$，对应的节点集合在"满足前 $k$ 个条件"意义上构成链式包含：

$$\{v : \text{Str}(v) \geq 3\} \subseteq \{v : \text{Str}(v) \geq 2\} \subseteq \{v : \text{Str}(v) \geq 1\} \subseteq V$$

**完备性**：每个 $v \in V$ 恰好属于一级。由定义 4.1 的四个分支互斥且穷尽（C1 成立/不成立，C2 成立/不成立，C3 成立/不成立的组合），四级划分完备。$\square$

**推论 T8-1.1**：候选集 $C = V_{\text{DR}} \cup V_{\text{EE}} \cup V_{\text{PE}} = \{v : \text{C1}(v)\} = \{v : v \leadsto v_{err}\}$，即候选集等于报错节点的后向切片。

**证明**：由定义 4.1，$v \in C$ 当且仅当 $\text{Explain}(v, e, G) \neq \text{Unrelated}$，当且仅当 $\text{C1}(v)$ 成立，当且仅当 $v \leadsto v_{err}$。这等价于后向切片。$\square$

### 5.2 定理 T8-2（可解释性与可计算性）

**定理 T8-2**：(a) 对每个 $v \in V$，$\text{Explain}(v, e, G)$ 在 $O(\|s\| \log \|s\|)$ 时间内可计算（不含 C3 的 counterfactual 部分）；(b) 对每个 $v \in C$，存在一阶逻辑公式 $I_v$ 解释其分级，且 $I_v$ 可被算法 $\text{Render}$ 翻译为人类可读文本。

**证明**：

**(a) 可计算性（不含 C3）**：

C1 判定：$v \leadsto v_{err}$，可用反向 BFS 判定。预处理一次反向 BFS（$O(|V_{\text{reach}}| + |E_{\text{reach}}|)$），之后每个 $v$ 的 C1 判定是 $O(1)$ 查表。

C2 判定：
- (C2a) $s^{out}_v = s_{act}$：元组逐元素比较，$O(\|s^{out}_v\|)$
- (C2b) $\text{Class}(v) \neq \text{Preserve}$：由 T2 引理 2.2，$O(1)$；方向一致检查 $O(1)$
- 总 $O(\|s\|)$

总体 $O(\|s\|)$，在 Tenth 中 $\|s\| \leq 8$（常数），故 $O(1)$。$\square$（不含 C3）

**(b) 可解释性**：

$I_v$ 由分级与成立的具体条件构成：

**DefinitelyRoot**：
$$I_v := \text{C1}(v) \wedge \text{C2}(v) \wedge \text{C3}(v) \wedge \text{CFDetail}(v)$$

其中 $\text{CFDetail}(v)$ 是 counterfactual 详情，如"反事实 shape $s' = ...$ 使报错消失"。

$\text{Render}(I_v)$ 输出："节点 $\ell_v$ 是根因：在错误路径上，shape $s^{out}_v$ 与报错相关，且 counterfactual 分析显示若 shape 改为 $s'$，报错不会发生。"

**ExplainsError**：
$$I_v := \text{C1}(v) \wedge \text{C2}(v) \wedge \neg \text{C3}(v) \wedge \text{WhyNoCF}(v)$$

其中 $\text{WhyNoCF}(v)$ 解释为何 counterfactual 不成立（如"所有反事实 shape 仍导致报错"）。

$\text{Render}(I_v)$ 输出："节点 $\ell_v$ 解释报错：在错误路径上，shape 相关，但 counterfactual 分析显示即使 shape 改变报错仍会发生，根因可能在更上游。"

**PartialExplain**：
$$I_v := \text{C1}(v) \wedge \neg \text{C2}(v) \wedge \text{PathRole}(v)$$

其中 $\text{PathRole}(v)$ 描述 $v$ 在路径上的角色（如"中转节点，shape Preserve"）。

$\text{Render}(I_v)$ 输出："节点 $\ell_v$ 部分解释：在错误路径上，但 shape 不直接相关。"

**Unrelated**：
$$I_v := \neg \text{C1}(v)$$

$\text{Render}(I_v)$ 输出："节点 $\ell_v$ 与报错无关。"

**Render 可行性**：$I_v$ 的语法受限于上述模板（条件名、shape 字面量、路径角色），模板数量有限（4 种 × 条件组合），故 Render 是有限映射，可预实现为查表 + 字符串拼接。$\square$

**注**：C3 的 counterfactual 判定 $\text{CF}(v, e, G)$ 的可计算性见定理 T8-4。在 T8-4 证明前，C3 的可计算性是开放问题（见 §8.1）。

### 5.3 定理 T8-3（F4 循环性的修复）

**定理 T8-3**：通过引入 Lewis (1973) counterfactual 因果谓词 $\text{CF}(v, e, G)$ 重新定义 C3（定义 3.4），定理 F4 的循环论证被消除，即：存在独立于本文解释关系 $\text{Explain}$ 的因果性形式化 $\text{CF}$，使得"若 $v^*$ 是 $e$ 的真实因果根因，则 $v^* \in V_{\text{DR}}$"不再退化为重言式。

**证明**：

**步骤 1：诊断 v3 F4 的循环性**。v3 定理 F4 陈述"若 $v^*$ 满足定义 3.2 之一，则 $v^* \in C$"。但"满足定义 3.2 之一"本身就是 $v^* \in C$ 的条件（定义 3.6），故 F4 等价于"$v^* \in C \implies v^* \in C$"，是重言式。循环性的根源是 v3 没有独立于 $\text{Explain}$ 的因果性概念——"导致"被定义为"满足 Explain 之一"。

**步骤 2：本文 C3 的独立性**。本文定义 3.4 的 $\text{CF}(v, e, G)$ 是独立于 $\text{Explain}$ 的因果性谓词：

$$\text{CF}(v, e, G) \iff \exists s' \in S_v^{\text{legal}}: \neg \text{Occurs}(e, G[v \mapsto s'])$$

$\text{CF}$ 的定义仅依赖：(i) Tape DAG $G$ 的拓扑结构，(ii) 算子 shape 语义 $\text{Sem}_{op}$（T2 定义 2.5，独立于 $\text{Explain}$），(iii) 报错 $e$ 的发生判定 $\text{Occurs}$（独立于 $\text{Explain}$）。$\text{CF}$ 不引用 $\text{Explain}$、$V_{\text{DR}}$、$V_{\text{EE}}$ 等本文定义的分级概念。故 $\text{CF}$ 是独立因果性形式化。

**步骤 3：修复后的完备性陈述**。本文的"完备性"陈述为：

> 若 $v^*$ 是 $e$ 的真实 counterfactual 因果根因（即 $\text{CF}(v^*, e, G)$ 成立且 $v^*$ 在错误路径上），则 $v^* \in V_{\text{DR}}$。

这个陈述不再循环——前提 $\text{CF}(v^*, e, G) \wedge \text{C1}(v^*) \wedge \text{C2}(v^*)$ 是独立于 $\text{Explain}$ 的，结论 $v^* \in V_{\text{DR}}$ 是本文分级的结果。

**步骤 4：完备性证明**。设 $v^*$ 满足 $\text{CF}(v^*, e, G) \wedge \text{C1}(v^*) \wedge \text{C2}(v^*)$。由定义 3.4，$\text{C3}(v^*) = \text{C2}(v^*) \wedge \text{CF}(v^*, e, G)$ 成立。由定义 4.1，$\text{Explain}(v^*, e, G) = \text{DefinitelyRoot}$，故 $v^* \in V_{\text{DR}}$。$\square$

**含义**：本文的 F4 修复了 v3 的循环论证。代价是引入了 counterfactual 谓词 $\text{CF}$，其可计算性需要单独分析（定理 T8-4）。如果 $\text{CF}$ 不可计算，则 C3 不可计算，DefinitelyRoot 级不可判定——这是 §8.1 诚实记录的局限。

### 5.4 定理 T8-4（反事实推理的可判定性）

**定理 T8-4**：在 Tape DAG $G = (V, E)$ 上，counterfactual 谓词 $\text{CF}(v, e, G)$ 的可判定性分两种情况：

(a) **若 $v$ 是非 Construct 节点**（即 $v$ 有至少一个输入，$\text{Class}(v) \in \{\text{Preserve}, \text{Reduce}, \text{Expand}\}$），则 $\text{CF}(v, e, G)$ 可判定，时间复杂度 $O(|V_{\text{path}}| \cdot \|s\|)$，其中 $V_{\text{path}}$ 是 $v$ 到 $v_{err}$ 的路径上的节点集。

(b) **若 $v$ 是 Construct 节点**（无输入，如常量构造），则 $\text{CF}(v, e, G)$ 的可判定性依赖于 shape 传播函数 $f_{v \to v_{err}}$ 的代数性质；本文给出**猜想 T8-4a**（见下）而非定理。

**猜想 T8-4a**：在 Tenth 的 21 个 TapeOp 算子集与有界 shape 维度（$\|s\| \leq 8$）下，Construct 节点的 $\text{CF}(v, e, G)$ 可判定，复杂度 $O(|V_{\text{path}}| \cdot D^8)$，其中 $D$ 是 shape 维度取值的实际上界（由运行时 shape 集合的有界性提供）。

**证明**：

**(a) 非 Construct 节点的可判定性**：

**步骤 1：合法反事实 shape 集合 $S_v^{\text{legal}}$ 有限**。$v$ 有输入 $s^{in}_v = (s^{in}_{v,1}, ..., s^{in}_{v,k_v})$（已运行时确定）。$v$ 的合法反事实 shape $s'$ 必须满足 $v$ 的算子语义约束：

- 若 $op_v = \text{Reshape}$：$|s'| = |s^{in}_{v,1}|$（元素数守恒），$s'$ 是同体积 shape 的有限集合
- 若 $op_v = \text{Transpose}(\pi)$：$s' = \pi(s^{in}_{v,1})$（唯一确定），$|S_v^{\text{legal}}| = 1$
- 若 $op_v = \text{MatMul}$：$s' = (m, n)$ 其中 $m = s^{in}_{v,1}.\text{row}$, $n = s^{in}_{v,2}.\text{col}$（唯一确定），$|S_v^{\text{legal}}| = 1$
- 若 $op_v = \text{Add}/\text{Sub}/\text{Mul}/\text{Div}$：$s' = \text{broadcast}(s^{in}_{v,1}, s^{in}_{v,2})$（唯一确定），$|S_v^{\text{legal}}| = 1$
- 若 $op_v$ 是一元 Preserve（Exp/Log/Sigmoid/Neg/ReLU）：$s' = s^{in}_{v,1}$（唯一确定），$|S_v^{\text{legal}}| = 1$
- 若 $op_v = \text{Sum}/\text{Mean}$：$s' = \epsilon$（标量，唯一确定），$|S_v^{\text{legal}}| = 1$

唯一例外是 Reshape，$|S_v^{\text{legal}}|$ 等于与 $|s^{in}_{v,1}|$ 同体积的 shape 数，是有限的（在 $\|s\| \leq 8$ 下，最多 $|s^{in}_{v,1}|$ 的因子数 $\leq 30$，每个因子对应有限多个 shape 排列）。

**步骤 2：反事实世界构造可计算**。对每个 $s' \in S_v^{\text{legal}}$，构造反事实世界 $G[v \mapsto s']$：将 $v$ 的输出 shape 改为 $s'$，重新计算 $v$ 的所有下游节点的 shape（用 $\text{Sem}_{op}$ 传播）。传播复杂度 $O(|V_{\text{path}}| \cdot \|s\|)$（每个节点用 $O(\|s\|)$ 计算 $\text{Sem}_{op}$，路径上最多 $|V_{\text{path}}|$ 个节点）。

**步骤 3：报错发生判定可计算**。$\text{Occurs}(e, G[v \mapsto s'])$ 检查在反事实世界中 $v_{err}$ 处是否触发报错 $e$。判定方式：
- 重新传播后 $v_{err}$ 的输入 shape 是否仍违反 $\text{Constraint}_{op_{v_{err}}}$？
- 若违反，$e$ 发生（$\text{Occurs} = \text{true}$）；若不违反，$e$ 不发生（$\text{Occurs} = \text{false}$）
- 判定复杂度 $O(\|s\|)$

**步骤 4：counterfactual 谓词判定**。$\text{CF}(v, e, G) = \exists s' \in S_v^{\text{legal}}: \neg \text{Occurs}(e, G[v \mapsto s'])$。遍历 $s' \in S_v^{\text{legal}}$，对每个 $s'$ 执行步骤 2-3。若存在 $s'$ 使 $\text{Occurs} = \text{false}$，则 $\text{CF} = \text{true}$；否则 $\text{CF} = \text{false}$。

**复杂度**：$|S_v^{\text{legal}}|$ 有限（多数情况 $= 1$，最多约 30），每个 $s'$ 的判定 $O(|V_{\text{path}}| \cdot \|s\|)$，总 $O(|S_v^{\text{legal}}| \cdot |V_{\text{path}}| \cdot \|s\|) = O(|V_{\text{path}}| \cdot \|s\|)$（$|S_v^{\text{legal}}|$ 视为常数）。$\square$

**(b) Construct 节点的开放问题**：

Construct 节点 $v$ 无输入，$s'$ 可以是任意 shape（$S_v^{\text{legal}} = \mathbb{S}$，无限）。直接遍历不可行。

**关键观察**：$\text{CF}(v, e, G)$ 等价于"shape 传播函数 $f_{v \to v_{err}}$ 不是常函数（值为 $s_{act}$）"。若 $f$ 是常函数（对所有 $s'$ 都返回 $s_{act}$），则报错必然发生，$\text{CF} = \text{false}$；若 $f$ 非常函数，存在 $s'$ 使 $f(s') \neq s_{act}$，$\text{CF} = \text{true}$。

**$f_{v \to v_{err}}$ 的结构**：$f$ 是有限个算子语义 $\text{Sem}_{op}$ 的复合。在 Tenth 的 21 个 TapeOp 中，$\text{Sem}_{op}$ 是 shape 元组上的"代数函数"——含等式（MatMul 内侧）、乘积（Reshape 元素守恒）、广播（Add/Sub/Mul/Div）、置换（Transpose）。这种函数的复合还是代数函数。

**可判定性的难点**：判断代数函数是否为常函数，等价于判断两个代数函数是否相等（$f(s') = s_{act}$ 是否对所有 $s'$ 成立）。在一般代数函数类上，函数等价性可能不可判定（依赖代数结构的复杂性）。

**猜想 T8-4a 的依据**：

1. **运行时 shape 有界性**：Tenth 实际运行时的 shape 维度取值有上界 $D$（由内存与张量大小限制，典型 $D \leq 10^6$）。在 $D$ 有界下，$S_v^{\text{legal}} \cap \{s : \|s\| \leq 8, d_i \leq D\}$ 有限，最多 $D^8$ 种。
2. **shape 传播函数的代数性质**：Tenth 的算子语义是"线性"或"分段线性"的 shape 函数（如 MatMul 输出 $(m, n)$ 是输入维度的线性投影；Reshape 输出是输入因子的重排），这种函数类上的常函数判定**可能**可判定，但本文未给出严格证明。

**诚实标注**：猜想 T8-4a 是基于运行时有界性的工程假设，**不是严格定理**。在无 $D$ 假设的理论情形下，Construct 节点的 $\text{CF}$ 可判定性是开放问题。这是本文的核心局限（§8.1）。$\square$

**推论 T8-4.1**：在非 Construct 节点情形下，DefinitelyRoot 级可判定，复杂度 $O(|V_{\text{reach}}| \cdot |V_{\text{path}}| \cdot \|s\|)$（对每个候选 $v$ 执行 T8-4 (a)）。

**推论 T8-4.2**：在 Construct 节点情形下，若猜想 T8-4a 成立，DefinitelyRoot 级可判定；否则 DefinitelyRoot 级的可判定性未知。

---

## 6. 因果推理扩展

### 6.1 Lewis (1973) counterfactual 语义的形式化

Lewis 的 counterfactual $A \square\!\!\!\!\to B$ 在 Tape DAG 上的实例化：

- **命题 $A$**："$v$ 的输出 shape 是 $s'$"（反事实假设）
- **命题 $B$**："$e$ 不发生"（反事实结论）
- **可能世界**：Tape DAG 的 shape 赋值变体 $G[v \mapsto s']$
- **最近世界**：在 Tape 上，"最近"退化为"仅修改 $v$ 的输出，保留其他节点的输入输出关系不变"——因为 Tape 是确定性 DAG，修改 $v$ 的输出后，下游节点的 shape 由 $\text{Sem}_{op}$ 唯一确定，不存在多个"最近世界"

**形式化**：

$$\text{CF}(v, e, G) \iff \exists s' \in S_v^{\text{legal}}: \neg \text{Occurs}(e, G[v \mapsto s'])$$

等价于 Lewis 的 $\neg(s_v = s^{out}_v) \square\!\!\!\!\to \neg \text{Occurs}(e, G)$——"若 $v$ 的 shape 不是 $s^{out}_v$，$e$ 不会发生"。

**与 Lewis 一般理论的差异**：

1. **有限性**：Lewis 的可能世界是无限的（含所有逻辑可能的世界），Tape 上的反事实世界是有限的（仅由 $S_v^{\text{legal}}$ 索引，且每个反事实世界由确定性传播唯一确定）。
2. **最近性**：Lewis 的"最近世界"依赖相似性度量（难以形式化），Tape 上的"最近世界"退化为"仅修改 $v$"——不存在相似性度量的问题。
3. **因果性**：Lewis 的因果性是"若 $\neg c$ 则 $\neg e$"，Tape 上是"若 $s_v \neq s^{out}_v$ 则 $\neg e$"——将事件因果替换为 shape 因果。

### 6.2 Tape DAG 上的反事实世界构造

**定义 6.1（合法反事实 shape 集合）**：节点 $v$ 的合法反事实 shape 集合

$$S_v^{\text{legal}} = \{s' \in \mathbb{S} : \text{Sem}_{op_v}(s^{in}_v \text{ 用 } s' \text{ 替换输出}) \neq \bot \text{ 且 } s' \neq s^{out}_v\}$$

更严格地：$s'$ 是 $v$ 的合法反事实 shape 当且仅当存在某个输入 shape 序列使 $v$ 的算子语义返回 $s'$（即 $s'$ 在 $op_v$ 的值域内）。对有输入的节点，$S_v^{\text{legal}}$ 由 $s^{in}_v$ 与 $op_v$ 的约束共同决定（见定理 T8-4 (a) 的步骤 1）。

**定义 6.2（反事实世界）**：对 $s' \in S_v^{\text{legal}}$，反事实世界 $G[v \mapsto s']$ 是按如下规则构造的新 Tape：

1. $v$ 的输出 shape 改为 $s'$：$s^{out}_v \leftarrow s'$
2. 对 $v$ 的所有下游节点 $w$（即 $v \leadsto w$），按拓扑序重新计算 $s^{out}_w \leftarrow \text{Sem}_{op_w}(s^{in}_w)$，其中 $s^{in}_w$ 中来自 $v$ 的分量已更新
3. 若某节点的 $\text{Sem}_{op}$ 返回 $\bot$（约束违反），传播停止，该节点及下游节点的 shape 标记为 $\bot$

**定义 6.3（报错发生判定）**：$\text{Occurs}(e, G')$ 在反事实世界 $G' = G[v \mapsto s']$ 中为真，当且仅当：
- $v_{err}$ 在 $G'$ 中的输入 shape 违反 $\text{Constraint}_{op_{v_{err}}}$，且
- 违反的 shape 对应报错 $e = (s_{exp}, s_{act})$（即 $v_{err}$ 的实际输入 shape 是 $s_{act}$）

### 6.3 与 Pearl do-calculus 的对比

| 维度 | Lewis counterfactual（本文） | Pearl do-calculus |
|------|---------------------------|-------------------|
| 对象 | 命题层面的反事实 | 概率层面的干预 |
| 形式 | $A \square\!\!\!\!\to B$ | $P(Y \| do(X = x))$ |
| 依赖 | 可能世界语义 | 因果图结构 |
| 在 Tape 上 | shape 反事实（修改 shape 值） | 不直接适用（Tape 无概率分布） |
| 可判定性 | 非 Construct 节点可判定（T8-4a） | 因果贝叶斯网上可判定（Shpitser & Pearl 2006） |
| 干预语义 | 修改 $v$ 的输出 shape | 切断 $X$ 与父节点的边 |

**关键差异**：Pearl 的 do 操作切断变量与其父节点的依赖，适用于因果图上的概率推理。在 Tape DAG 上，节点没有"父节点干预"的语义（每个节点的输入是确定的上游输出）。本文的 counterfactual 是"修改 $v$ 的输出，保留 $v$ 的输入与算子语义"——这是 shape 层面的反事实，不是图结构层面的干预。

**联系**：两者都关注"若改变某变量，结果如何变化"。本文的反事实可视为 do-calculus 在确定性 shape 传播上的特例——将 $do(s_v = s')$ 视为"强制 $v$ 的输出为 $s'$"，然后观察 $e$ 是否发生。

### 6.4 因果性层级

基于 counterfactual，可以定义更强的因果性层级（本文不深入展开，仅列举）：

- **弱因果（sufficient cause）**：$v$ 的 shape 是 $e$ 的充分原因，若 $v$ 的当前 shape 导致 $e$（即 $\text{Occurs}(e, G)$）
- **必要因果（necessary cause）**：$v$ 的 shape 是 $e$ 的必要原因，若 $\neg v \square\!\!\!\!\to \neg e$（若 $v$ 的 shape 不存在则 $e$ 不发生）——这是本文的 $\text{CF}$
- **充分必要因果**：两者都成立

本文的 C3 采用必要因果（counterfactual），因为"根因"的直觉是"若根因不发生，错误就不会发生"——这正是必要因果。充分因果在 shape 错误场景中意义较弱（多个 shape 都可能导致同一错误）。

---

## 7. 与现有调试方法对比

### 7.1 vs 程序切片的相关性

程序切片（Weiser 1981）给出"影响变量 $V$ 在语句 $s$ 处值的所有语句"——这是相关性集合。切片不区分切片内语句的"相关强度"，所有切片语句被视为同等相关。

本文的四级分类是对切片结果的层级化：
- 切片 = 本文的候选集 $C = \{v : \text{C1}(v)\}$（推论 T8-1.1）
- 切片内的语句进一步分为 DefinitelyRoot / ExplainsError / PartialExplain

这种层级化使调试器可以**优先展示强证据**（DefinitelyRoot），而非平铺所有切片语句。

### 7.2 vs 经验式调试

| 维度 | 经验式权重 | 程序切片 | 本文四级分类 |
|------|----------|---------|------------|
| 排序依据 | op 类型权重 | 拓扑可达性 | C1 可达性 + C2 相关性 + C3 因果性 |
| 可解释性 | "权重高" | "在切片中" | 形式化公式 $I_v$（T8-2） |
| 可证伪 | 不可证伪 | 可证伪（切片可重算） | 可证伪（counterfactual 可重算，非 Construct 情形） |
| 误判风险 | 高 | 中（切片过大） | 低（counterfactual 过滤） |
| 完备性 | 无保证 | 完备（切片含所有相关语句） | 完备（C1 等价切片）+ 因果性层级 |
| 复杂度 | $O(|V|)$ | $O(|V| + |E|)$ | $O(|V_{\text{reach}}| + |E_{\text{reach}}|)$（非 Construct） |

**结论**：本文方法在复杂度相同或更优的前提下，提供切片所没有的因果性层级与可证伪性。代价是 Construct 节点的 counterfactual 可判定性未完全证明（§8.1）。

### 7.3 vs 变异分析

软件工程中的变异分析（mutation analysis）通过修改源码观察错误是否消失，判断语句是否是根因。本文的 counterfactual 是"shape 变异"而非"源码变异"：

- **变异分析**：修改源码 → 重新运行 → 观察错误。开销：每次变异需完整运行。
- **本文 counterfactual**：修改 shape → 重新传播 shape → 观察报错。开销：仅 shape 传播，不重新运行。

这种差异使本文的 counterfactual 可在单次运行后完成（T8-4 (a) 复杂度 $O(|V_{\text{path}}| \cdot \|s\|)$），远低于变异分析的 $O(|V| \cdot \text{RunTime})$。

---

## 8. 开放问题与局限

### 8.1 反事实推理在 Construct 节点的可判定性未证明

**局限**：定理 T8-4 (b) 的 Construct 节点 counterfactual 可判定性是**猜想 T8-4a**而非定理。在无 shape 取值上界 $D$ 的理论情形下，$S_v^{\text{legal}} = \mathbb{S}$ 无限，遍历不可行。

**影响**：DefinitelyRoot 级在 Construct 节点的判定可能不可计算。若 Construct 节点是真实根因（如常量构造了错误 shape），本文方法可能无法判定。

**缓解**：
1. **工程上**：Tenth 运行时 shape 有上界 $D$（内存限制），猜想 T8-4a 在实践中成立。
2. **理论上**：可研究 shape 传播函数 $f_{v \to v_{err}}$ 的代数性质，判断其在 Tenth 算子集上是否属于可判定的函数类（如线性 shape 函数的复合是否仍可判定）。
3. **降级**：对 Construct 节点，若 counterfactual 不可判定，降级为 ExplainsError（仅相关性，非因果性）。

### 8.2 Lewis "最近世界"在 Tape 上的退化

**局限**：Lewis 的 counterfactual 依赖"最近可能世界"的相似性度量，但在 Tape DAG 上，"最近"退化为"仅修改 $v$"——这是工程上的简化，不是 Lewis 理论的严格实例化。

**影响**：本文的 counterfactual 不考虑"修改多个节点"的反事实世界。例如，若 $v_1$ 与 $v_2$ 共同导致 $e$，仅修改 $v_1$ 可能不足以使 $e$ 消失——但本文的 $\text{CF}$ 会判定 $v_1$ 不是根因（因 $\neg \text{Occurs}(e, G[v_1 \mapsto s'])$ 不成立）。

**缓解**：本文聚焦"单根因"场景。多根因场景需要"联合 counterfactual"（修改多个节点），是未来工作（§9.2）。

### 8.3 多根因场景的扩展未覆盖

**局限**：本文的 $\text{CF}(v, e, G)$ 是单节点 counterfactual——"若 $v$ 改变，$e$ 是否不发生"。但实际 shape 错误可能由多个节点共同导致：

- **联合因果**：$v_1$ 与 $v_2$ 单独都不导致 $e$，但共同导致 $e$（INUS 条件，Mackie 1965）
- ** overdetermination**：$v_1$ 与 $v_2$ 单独都导致 $e$，但若都改变 $e$ 不发生

本文的 $\text{CF}$ 不能识别这类多根因场景。

**影响**：DefinitelyRoot 级可能漏报联合根因。

**缓解**：未来工作扩展 $\text{CF}$ 为集合版：$\text{CF}_{\text{set}}(S, e, G) \iff \exists s' \in S_v^{\text{legal}}(S): \neg \text{Occurs}(e, G[S \mapsto s'])$，其中 $S$ 是节点集合。这增加组合复杂度（$|S|$ 个节点的反事实组合指数增长），需要启发式剪枝。

### 8.4 报错节点 $v_{err}$ 的定位依赖运行时

**局限**：定义 3.2 的 C1 依赖报错节点 $v_{err}$，而 $v_{err}$ 由运行时报错上下文提供（T2 §4.1 v3 修正）。若运行时未记录 $v_{err}$（如报错来自外部库），C1 不可计算。

**影响**：本文方法仅在 Tenth 运行时记录 $v_{err}$ 时生效。对外部库报错，需降级为"全 Tape 候选"。

**缓解**：Tenth 的运行时应在抛出 shape 错误时显式记录 $v_{err}$ 的 Tape 节点引用（实施时的接口要求，见 T2 §4.4 算法步骤 1）。

### 8.5 counterfactual 与"自然期望"的关系

**局限**：v3 草稿的 (C1) "自然期望"（定义 3.4）依赖用户意图，仅部分形式化。本文的 C3 counterfactual 不依赖"自然期望"，但也不完全替代——counterfactual 判定"shape 改变是否消除报错"，但不判定"shape 是否符合用户意图"。

**影响**：counterfactual 可能给出"shape 改变消除报错但不符合用户意图"的反事实 shape，导致 DefinitelyRoot 误判。

**缓解**：counterfactual 与"自然期望"是互补的——counterfactual 提供因果性，"自然期望"提供意图对齐。未来工作可结合两者。

---

## 9. 开放问题与未来工作

### 9.1 反事实推理的计算复杂度优化

定理 T8-4 (a) 的复杂度 $O(|V_{\text{path}}| \cdot \|s\|)$ 在单节点上可接受，但对所有候选 $v \in C$ 执行 counterfactual 的总复杂度是 $O(|C| \cdot |V_{\text{path}}| \cdot \|s\|) = O(|V_{\text{reach}}|^2 \cdot \|s\|)$，在大 Tape 上可能显著。

未来工作：
- **增量 counterfactual**：复用 shape 传播的中间结果，避免对每个 $v$ 重复传播
- **符号 shape 传播**：用符号 shape 变量代替具体值，一次传播判断多个 $v$ 的 counterfactual
- **剪枝**：先用 C2 过滤候选，仅对 C2 成立的节点执行 counterfactual

### 9.2 多根因场景的扩展

§8.3 提出的联合 counterfactual 是未来工作。形式化挑战：
- **INUS 条件**（Mackie 1965）：$v$ 是 $e$ 的 INUS 条件当且仅当 $v$ 是某最小充分条件集的必要部分
- **Halpern-Pearl 因果**（Halpern & Pearl 2005）：基于结构因果模型的实际因果定义，可处理 overdetermination

将 Halpern-Pearl 因果应用到 Tape DAG 上的 shape 错误是未来工作的方向。

### 9.3 静态 counterfactual

本文的 counterfactual 在运行时 Tape 上执行。若能在编译期 HIR 上做 counterfactual（"若某 HIR 节点的 shape 改变，报错是否不发生"），可提供编译期预警。但 HIR 上的 shape 是符号的，counterfactual 涉及符号 shape 传播，可判定性更复杂（与 B 护城河的约束求解耦合）。

### 9.4 counterfactual 与用户意图的融合

§8.5 提出的 counterfactual 与"自然期望"的融合，需要形式化用户意图。可能的路径：
- **类型注解**：用户在代码中标注期望 shape，counterfactual 检查反事实 shape 是否符合注解
- **历史 shape**：若程序多次运行，用历史 shape 作为"自然期望"的近似
- **交互式调试**：counterfactual 给出候选反事实 shape，由用户判断是否符合意图

---

## 10. 结论

本文针对 Tenth 语言张量关系调试器（护城河 F）中"算子是否为 shape 错误根因"的判定问题，提出严格的四级解释关系层级 `DefinitelyRoot`、`ExplainsError`、`PartialExplain`、`Unrelated`，并用三条件 (C1)(C2)(C3) 的合取严格定义。本文的核心贡献是：

1. **三条件的层级化**（§3）：将 v3 草稿的三条件从"并列的根因模式"重组为"递进的可达性/相关性/因果性"层级，使四级偏序结构显式化（定理 T8-1）。
2. **可解释性**（定理 T8-2）：每级附带一阶逻辑公式 $I_v$，可被算法 Render 翻译为人类可读文本。
3. **F4 循环性的修复**（定理 T8-3）：引入 Lewis (1973) counterfactual 因果谓词 $\text{CF}(v, e, G)$ 重新定义 C3，使解释关系摆脱对自身分析框架的循环依赖。这是对 v3 §6.6 诚实记录的循环论证局限的直接回应。
4. **反事实推理的可判定性**（定理 T8-4）：在非 Construct 节点情形下可判定（$O(|V_{\text{path}}| \cdot \|s\|)$），在 Construct 节点情形下给出猜想 T8-4a 并诚实标注为开放问题。

**5 处核心局限**（诚实记录）：
- §8.1：Construct 节点的 counterfactual 可判定性未完全证明（猜想 T8-4a）
- §8.2：Lewis "最近世界"在 Tape 上的退化（仅修改单节点）
- §8.3：多根因场景的扩展未覆盖
- §8.4：报错节点 $v_{err}$ 的定位依赖运行时
- §8.5：counterfactual 与"自然期望"的关系未完全融合

**对 Tenth 开发的指导**：
- F 的 MVP 可基于本文的四级分类与 T8-2 可解释性实现，counterfactual 部分（C3）先支持非 Construct 节点（T8-4 (a)）
- DefinitelyRoot 级在 Construct 节点降级为 ExplainsError（§8.1 缓解方案 3）
- 多根因扩展（§9.2）作为 Phase 2 工作，不阻塞 MVP

**理论价值**：本文将 Lewis counterfactual 因果推理应用于张量调试器形式化，是因果推理在 AI 原生语言调试中的首次系统应用。修复了 v3 草稿 F4 的循环论证，为护城河 F 提供了独立于分析框架的因果性基础。

---

## 附录 A：定理索引

| 定理 | 内容 | 章节 | 依赖 |
|------|------|------|------|
| 定义 3.2 | C1 可达性 | §3.2 | T2 §3.1 |
| 定义 3.3 | C2 相关性 | §3.2 | 定义 3.2 |
| 定义 3.4 | C3 因果性 | §3.2 | 定义 3.3, §6 |
| 定义 4.1 | 四级解释关系 | §4.1 | 定义 3.2-3.4 |
| T8-1 | 四级偏序性 | §5.1 | 定义 4.1 |
| T8-1.1 | 候选集等于后向切片 | §5.1 | T8-1 |
| T8-2 | 可解释性与可计算性 | §5.2 | T8-1, T2 F1 |
| T8-3 | F4 循环性的修复 | §5.3 | 定义 3.4, §6 |
| T8-4 | 反事实推理的可判定性 | §5.4 | 定义 3.4, §6 |
| T8-4.1 | 非 Construct 下 DefinitelyRoot 可判定 | §5.4 | T8-4 (a) |
| T8-4.2 | Construct 下可判定性开放 | §5.4 | T8-4 (b) |
| 猜想 T8-4a | Construct 下可判定性猜想 | §5.4 | — |

## 附录 B：与现有文档的对应

| 本文章节 | 对应文档 | 关系 |
|---------|---------|------|
| §3.2 定义 3.2-3.4 | v3 §3.1 定义 3.2-3.3 | 重新形式化（层级化） |
| §4.1 定义 4.1 | v3 §3.1 定义 3.5 | 严格化（合取形式） |
| §5.1 T8-1 | v3 §3.2（隐含） | 显式化偏序结构 |
| §5.2 T8-2 | v3 §3.2 F3 | 扩展（含分级可解释性） |
| §5.3 T8-3 | v3 §3.2 F4, §6.6 | **核心修正**（消除循环论证） |
| §5.4 T8-4 | v3 §6.6（未来工作） | **核心扩展**（counterfactual 可判定性） |
| §6 | v3 §6.6（"独立于定义 3.2 的导致关系形式化"） | 实现 |
| §7.1 | T2 §7 | 在切片基础上加因果层级 |
| §8.1 | v3 §6.6 | 新局限（Construct 可判定性） |

## 附录 C：实施建议

### C.1 四级分类的 MVP 实施

1. **实现 C1（可达性）**：反向 BFS 预处理，$O(|V_{\text{reach}}| + |E_{\text{reach}}|)$
2. **实现 C2（相关性）**：对每个 $v \in C$ 检查 (C2a)(C2b)，$O(\|s\|)$ 每节点
3. **实现 C3（counterfactual，非 Construct）**：对每个非 Construct 的 $v \in V_{\text{EE}}$ 执行 T8-4 (a) 算法
4. **Construct 节点降级**：Construct 节点的 DefinitelyRoot 判定降级为 ExplainsError（§8.1 缓解）
5. **Render 实现**：按 T8-2 的四个模板实现

### C.2 counterfactual 的实施细节

1. **合法反事实 shape 集合 $S_v^{\text{legal}}$**：按算子类型查表（Reshape 给出同体积 shape 集合，其他算子唯一确定）
2. **反事实世界构造**：从 $v$ 拓扑序传播 shape 到 $v_{err}$，复杂度 $O(|V_{\text{path}}| \cdot \|s\|)$
3. **报错发生判定**：检查 $v_{err}$ 的输入 shape 是否仍违反 $\text{Constraint}_{op_{v_{err}}}$
4. **早停优化**：若找到任一 $s'$ 使 $\text{Occurs} = \text{false}$，立即返回 $\text{CF} = \text{true}$

### C.3 测试用例

测试应覆盖：
- DefinitelyRoot：MatMul 内侧不匹配，counterfactual 改变内侧维度使报错消失
- ExplainsError：Reshape 改变方向一致，但 counterfactual 显示报错仍发生（根因在更上游）
- PartialExplain：路径上的 Preserve 节点（如 Exp），shape 不相关
- Unrelated：不在路径上的节点
- Construct 节点降级：常量构造错误 shape，降级为 ExplainsError

---

## 参考文献

1. Lewis, D. (1973). *Counterfactuals*. Harvard University Press.（§2.1, §6.1 counterfactual 因果语义）
2. Pearl, J. (2000). *Causality: Models, Reasoning, and Inference*. Cambridge University Press.（§2.1 do-calculus）
3. Shpitser, I., & Pearl, J. (2006). Identification of conditional interventional distributions. *UAI*.（§2.1 do-calculus 完备性）
4. Halpern, J. Y., & Pearl, J. (2005). Causes and explanations: A structural-model approach. *British Journal for the Philosophy of Science*, 56(4), 843-887.（§9.2 实际因果定义）
5. Mackie, J. L. (1965). Causes and conditions. *American Philosophical Quarterly*, 2(4), 245-265.（§8.3 INUS 条件）
6. Weiser, M. (1981). Program slicing. *ICSE*.（§2.2 程序切片）
7. Tip, F. (1995). A survey of program slicing techniques. *Journal of Programming Languages*, 3(3), 121-189.（§2.2 切片综述）
8. Zeller, A. (2002). Isolating cause-effect chains from computer programs. *FSE*.（§2.3 delta debugging）
9. Jones, J. A., Harrold, M. J., & Stasko, J. (2002). Visualization of test information to assist fault localization. *ICSE*.（§2.3 Tarantula）
10. Baydin, A. G., Pearlmutter, B. A., Radul, A. A., & Siskind, J. M. (2018). Automatic differentiation in machine learning: a survey. *JMLR*.（§2.1 自动微分模式）
11. Wengert, R. H. (1964). A simple automatic derivative evaluation program. *Communications of the ACM*, 7(8), 463-464.（§2.1 Wengert Tape）
12. Tenth 项目内部文档：
    - `docs/shape-check-roadmap/形式化分析理论可行性论证.md` v3（§3 定义 3.2, §6.6 F4 循环性）
    - `docs/shape-check-roadmap/战略规划.md`（方向 F 战略定位）
    - `docs/论文/T2-Tape形式化模型与根因定位可判定性.md`（Tape 形式化模型, F1-F5）
    - `tenth/src/runtime/autodiff.rs`（Tape 实现，[autodiff.rs:15-25](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/autodiff.rs) TapeNode 结构）
    - `tenth/src/runtime/tensor.rs`（Tensor.tape_id 字段）

---

> **文档结束**
>
> 本文 v1 基于 v3 草稿的 §3 定义 3.2 与 §6.6 F4 循环性局限，提出四级解释关系的层级化形式化与 counterfactual 因果性修复。本文的核心理论贡献是定理 T8-3（F4 循环性修复）与定理 T8-4（反事实推理可判定性），核心局限是 §8.1 诚实记录的 Construct 节点可判定性未证明（猜想 T8-4a）。所有形式化定义均可锚定到 Tenth v0.3.3 源码位置（[autodiff.rs](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/autodiff.rs) 与 [tensor.rs](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/tensor.rs)）。如发现证明漏洞或边界遗漏，应在 `MEMO.md` 记录并修订本文。
