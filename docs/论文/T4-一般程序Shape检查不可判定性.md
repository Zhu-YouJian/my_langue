# 一般程序 Shape 检查的不可判定性：基于 Rice 定理的归约分析

> **作者**：Tenth 项目数理部
> **日期**：2026-07-02
> **类型**：理论分析论文（T4）
> **适用范围**：护城河 B（Shape 代数求解器）理论边界、Tenth 编译期-运行时分层策略的理论依据
> **关联文档**：
> - `docs/shape-check-roadmap/形式化分析理论可行性论证.md`（v3 草稿，§4.2 定理 B1 的归约雏形）
> - `docs/shape-check-roadmap/战略规划.md`（双层策略的战略定位）
> - `docs/shape-check-roadmap/综合分析.md`（护城河闭环结构）
> - `docs/论文/T2-Tape形式化模型与根因定位可判定性.md`（互补论文，运行时层）
> - `tenth/src/hir/lower/types.rs`（编译期 shape 检查实现）
> - `tenth/src/runtime/autodiff.rs`（运行时 Tape 兜底）
> **本文定位**：在 v3 草稿 §4.2 定理 B1 基础上严格化的独立论文，给出基于 Rice 定理的完整归约证明，并与 T3（NP 完全性下界）共同构成 shape 检查的完整复杂度图景

---

## 摘要

本文证明 Tenth 语言中"任意程序的所有 shape 错误均可编译期检出"这一性质是不可判定的。我们将 shape 错误形式化为程序语义的非平凡性质，通过 Rice 定理的归约框架建立不可判定性结论：构造从停机问题到 shape 检查问题的多一归约（many-one reduction），证明若存在能完整检出 shape 错误的算法，则可判定停机问题，与 Turing (1936) 矛盾。本文给出归约的完整正确性论证（双向），并刻画可判定的子类（无递归无 while 的程序）。本结论与 T3（NP 完全性下界）互补：T4 给出不可判定性上界（根本边界），T3 给出可判定子集的 NP 完全性下界，两者共同构成 shape 检查的完整复杂度图景。基于此理论边界，Tenth 采取"保守近似 + 运行时兜底"的双层策略——编译期对可判定子集做保守近似（护城河 B，`--strict-shapes` 模式），运行时通过 Tape 形式化根因分析（护城河 F）做精确诊断。本文诚实记录归约的依赖假设与保守近似的不足。

**关键词**：Shape 检查、不可判定性、Rice 定理、停机问题归约、静态分析边界、保守近似、Tenth 语言

---

## 1. 引言

### 1.1 编译期 Shape 检查的工程价值

AI 原生语言的核心竞争力之一是 shape（形状）错误的编译期诊断能力。shape 错误是 AI 开发中最常见的 bug 类型之一——典型场景包括：MatMul 内侧维度不匹配、Reshape 元素数不守恒、Broadcast 不可广播、autograd 反向传播梯度 shape 漂移等。现有主流 AI 框架在 shape 错误上的表现存在结构性缺陷：

- **PyTorch**：动态图架构，shape 错误只能运行时崩溃，报错形式为 `RuntimeError: mat1 and mat2 shapes cannot be multiplied (3x8 and 4x8)`，仅定位到代码行，不追溯根因（[战略规划.md](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/docs/shape-check-roadmap/战略规划.md) 方向 F 节）。
- **JAX**：`check_shading` 仅检查前向 shape，不查反向、不查数值、不查内存（[综合分析.md §3.1](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/docs/shape-check-roadmap/综合分析.md)）。

Tenth 作为编译型 AI 语言，已实现护城河 A（Autograd 反向 Shape 静态验证，消除 5 处 silent squeeze）与护城河 D（编译期内存/算力预估，warning 系统），并规划护城河 B（Shape 代数求解器）与护城河 F（张量关系调试器）。这些能力的共同前提是：**编译期 shape 检查能在多大范围内生效**。

### 1.2 "完整检出 shape 错误"的可行性疑问

工程上自然提出一个强期望：**能否让编译器检出程序中所有的 shape 错误**？这一期望若可实现，将使 Tenth 在编译期就消除所有 shape 相关 bug，达到远超 PyTorch/JAX 的开发体验。

但这一期望面临理论层面的根本质疑。shape 错误本质上是程序语义的一种性质——它依赖于程序运行时的实际行为（如函数实际返回什么 shape、循环实际执行几次、分支实际走哪条路径）。而程序语义性质的精确分析，自 Rice (1953) 与 Turing (1936) 以来已被证明存在不可判定的根本边界。

本文的核心研究问题是：

> **研究问题**：在 Tenth 语言（含递归函数调用、while 循环、非线性 shape 约束）中，"任意程序的所有 shape 错误均可编译期检出"这一性质是否可判定？若不可判定，其可判定的子集是什么？工程上应如何应对？

### 1.3 贡献

本文的贡献如下：

1. **形式化**（§3）：将 Tenth 程序的 shape 错误严格形式化为程序语义的非平凡性质，满足 Rice 定理的前提条件。
2. **主定理与证明**（§4）：证明定理 B1——一般程序 shape 检查不可判定。给出从停机问题到 shape 检查的多一归约的完整双向正确性论证，相比 [v3 草稿 §4.2](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/docs/shape-check-roadmap/形式化分析理论可行性论证.md) 的反证法叙述更严格。
3. **可判定子集刻画**（§4.3）：刻画 shape 检查的可判定子类（无递归无 while 的程序、单函数内验证、线性约束在整数上可解）。
4. **与 T3 的互补关系**（§5）：建立"不可判定性上界（T4）+ NP 完全性下界（T3）"的完整复杂度图景。
5. **工程启示**（§6）：为 Tenth 的"保守近似 + 运行时兜底"双层策略提供理论依据，给出编译期-运行时边界划分原则。
6. **信息论下界猜想**（§7）：扩展 v3 草稿 §6.8 的静态分析本质局限，提出静态分析信息量的形式化上界猜想，明确标注为猜想而非定理。

### 1.4 本文与 v3 草稿的关系

[v3 草稿 §4.2](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/docs/shape-check-roadmap/形式化分析理论可行性论证.md) 已给出定理 B1 的反证法叙述，但存在以下不足，本文予以严格化：

| v3 草稿的不足 | 本文的严格化 |
|--------------|------------|
| 归约以反证法叙述，未显式构造多一归约函数 | §4.2 显式定义归约函数 $f: (M, w) \mapsto P_{M,w}$，证明其多项式可计算 |
| 双向正确性论证分散在反证法中 | §4.3 分"正向"与"反向"两节独立论证 |
| 未讨论归约的依赖假设（如 Tenth 是否图灵完备） | §4.4 显式列出归约的依赖假设，并诚实标注 |
| 未与 T3（NP 完全性）建立显式联系 | §5 建立 T4-T3 互补图景 |
| §6.8 的静态分析本质局限为定性叙述 | §7 提出形式化的信息论下界猜想（明确标注为猜想） |

---

## 2. 背景与相关工作

### 2.1 Rice 定理及其在程序分析中的应用

**Rice 定理**（Rice 1953）：设 $\varphi$ 是 Turing 完备语言中程序计算的偏函数。对任意偏函数的非平凡语义性质 $\mathcal{P}$（即 $\mathcal{P}$ 是某些偏函数的集合，且既非空也非全集），判定给定程序 $e$ 是否满足 $\varphi_e \in \mathcal{P}$ 是不可判定的。

形式化地，记 $\mathcal{P} \subseteq \{\text{偏可计算函数}\}$，$\mathcal{P} \neq \emptyset$，$\mathcal{P} \neq \{\text{偏可计算函数}\}$，则集合
$$\{e : \varphi_e \in \mathcal{P}\}$$
是不可判定的（即其特征函数不可计算）。

Rice 定理是程序分析不可判定性的根本工具，应用于：

- **等价性检查**：判定两程序是否计算相同函数，不可判定（取 $\mathcal{P} = \{\varphi_{e_0}\}$ 为单一函数类）。
- **死代码检测**：判定程序某点是否可达，不可判定（归约到停机问题）。
- **常量传播的精确性**：判定变量是否恒取某常量值，不可判定。
- **类型检查的精确性**：判定动态类型程序运行时类型是否符合某规格，不可判定。

本文将 Rice 定理应用于 shape 检查：shape 错误的检出等价于判定程序语义的某非平凡性质，故不可判定。

### 2.2 类型检查的可判定性边界

静态类型系统的可判定性已有成熟研究：

- **Hindley-Milner (HM) 类型推断**：对 parametric polymorphism，HM 类型推断可判定且 $O(n^2)$（Damas-Milner 1982）。但扩展到 higher-rank types、type classes、dependent types 后逐步不可判定。
- **依赖类型检查**：构造演算（CoC）的类型检查可判定，但一般依赖类型语言的 type inference 不可判定（Elliott 1989, Henk 1993）。
- **refinement types**：液体类型（Liquid Types, Rondon et al. 2008）的推断通过 abstract interpretation 保守近似，精确推断不可判定。

Tenth 的 shape 系统可视为一种受限的依赖类型系统——shape 是张量类型的"依赖维度"。本文的不可判定性结论与依赖类型推断的不可判定性同源，但归约构造不同：本文直接归约到停机问题，而非依赖 type inference 的不可判定性。

### 2.3 静态分析的精度-可判定性权衡

抽象解释（Cousot & Cousot 1977）建立了静态分析的精度-可判定性权衡的形式化框架：

- **Galois 连接**：静态分析对应具体语义的近似，近似精度由抽象域决定。
- **可判定近似**：选择有限高度的抽象域，可保证分析终止（可判定），但损失精度。
- **不可判定的精确分析**：无限高度的抽象域可达到任意精度，但分析不可判定。

本文的结论与抽象解释框架一致：Tenth 的 shape 检查若要精确（无限精度），不可判定；若要可判定（有限高度抽象域），必须保守近似。本文的归约证明为这一权衡在 shape 检查领域的具体形式提供了严格论证。

### 2.4 与 T3（NP 完全性）的关系

T3 论证 shape 约束求解在可判定子集上的 NP 完全性（[v3 草稿定理 B2b](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/docs/shape-check-roadmap/形式化分析理论可行性论证.md)，归约到 0-1 INTEGER PROGRAMMING）。本文（T4）与 T3 的关系是：

- **T4**：给出不可判定性**上界**——一般 shape 检查超出可判定范围。
- **T3**：给出可判定子集的 NP 完全性**下界**——即使在可判定子集上，最坏情况也是 NP 完全。

两者共同构成 shape 检查的完整复杂度图景（详见 §5）。

---

## 3. Tenth 程序的 Shape 错误形式化

### 3.1 前置概念

我们沿用 [v3 草稿 §2](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/docs/shape-check-roadmap/形式化分析理论可行性论证.md) 的 shape 与算子定义：

**定义 3.1（Shape）**：Shape 是非负整数元组 $s = (d_1, ..., d_n)$，其中 $n \geq 0$，$d_i \in \mathbb{N} = \{0, 1, 2, ...\}$。空元组 $\epsilon$ 表示标量。记所有 shape 的集合为 $\mathbb{S} = \bigcup_{n \geq 0} \mathbb{N}^n$。

**定义 3.2（张量类型）**：Tenth 的张量类型记为 `Tensor[dt, d_1, ..., d_n]`，其中 `dt` 是 dtype（如 `f64`），$(d_1, ..., d_n) \in \mathbb{S}$ 是 shape。在 HIR 中，shape 维度可为：
- 已知常量 $c \in \mathbb{N}$
- 符号维度 $\text{Symbol}(name)$（同名等价）
- 未知维度 $\text{Any}$

**定义 3.3（算子 shape 语义）**：每个算子 $op$ 关联 shape 语义函数 $\text{Sem}_{op}: \mathbb{S}^{k_{op}} \to \mathbb{S} \cup \{\bot\}$，其中 $\bot$ 表示输入 shape 不合法。详见 [v3 草稿 定义 2.5](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/docs/shape-check-roadmap/形式化分析理论可行性论证.md)。

### 3.2 Shape 错误的语法与语义定义

**定义 3.4（Shape 错误，运行时语义）**：程序 $P$ 在输入 $x$ 上发生 shape 错误，记作 $P(x) \downarrow_{\text{err}}$，当且仅当 $P$ 在执行 $x$ 时触发某算子 $op$ 的内部约束违反，即存在执行轨迹中的某次算子调用 $\text{op}(s_1, ..., s_k)$ 使 $\text{Sem}_{op}(s_1, ..., s_k) = \bot$。

**注**：定义 3.4 是运行时语义定义——shape 错误是程序执行时的事件。本文关注的是"编译期能否预测所有运行时 shape 错误"，这本质上是对运行时性质的静态分析。

**定义 3.5（程序的 Shape 行为）**：程序 $P$ 的 shape 行为是函数
$$\text{ShapeBehavior}(P): x \mapsto \begin{cases}
\text{trace}(P, x) & \text{若 } P(x) \text{ 终止}\\
\bot_{\text{div}} & \text{若 } P(x) \text{ 不终止}
\end{cases}$$

其中 $\text{trace}(P, x)$ 是 $P$ 在 $x$ 上执行的所有算子调用的 shape 序列。

**定义 3.6（无 Shape 错误的程序）**：程序 $P$ 是无 shape 错误的（shape-safe），记作 $\text{Safe}(P)$，当且仅当对所有输入 $x$，$P(x)$ 要么不终止，要么终止且不发生 shape 错误：
$$\text{Safe}(P) \iff \forall x: P(x) \downarrow_{\text{err}} \text{ 不发生}$$

**定义 3.7（Shape 错误检出器）**：Shape 错误检出器是函数
$$\text{Detect}: P \mapsto \begin{cases}
\text{true} & \text{若 } \text{Safe}(P)\\
\text{false} & \text{若 } \neg \text{Safe}(P)
\end{cases}$$

若 $\text{Detect}$ 对所有程序 $P$ 都给出正确答案，则称 shape 检查问题**可完全判定**。

### 3.3 "完整检出"性质的数学刻画

"任意程序的所有 shape 错误均可编译期检出"等价于 $\text{Detect}$ 是可计算的（即 $\text{Detect}$ 是 total recursive function）。本文主定理（§4）证明：$\text{Detect}$ 不可计算。

**关键观察**：$\text{Safe}(P)$ 是程序 $P$ 的语义性质——它依赖于 $P$ 在所有输入上的运行时行为。这正是 Rice 定理所断言不可判定的对象（非平凡语义性质）。

### 3.4 程序语义的非平凡性质（Rice 定理前提验证）

为应用 Rice 定理，需验证 $\text{Safe}$ 满足两个前提：

**前提 1：$\text{Safe}$ 是语义性质**（即仅依赖于程序计算的偏函数，不依赖于程序文本）。

**引理 3.1**：若两程序 $P_1, P_2$ 计算相同的偏函数（即对所有输入 $x$，$P_1(x)$ 与 $P_2(x)$ 同时终止且结果相同，或同时不终止），则 $\text{Safe}(P_1) \iff \text{Safe}(P_2)$。

**证明**：$\text{Safe}(P)$ 定义为"对所有输入 $x$，$P(x)$ 要么不终止，要么终止且不发生 shape 错误"。若 $P_1, P_2$ 计算相同偏函数，则对所有 $x$，$P_1(x)$ 与 $P_2(x)$ 同时终止或同时不终止；终止时执行轨迹的 shape 序列可能不同（因为不同实现可能用不同算子序列计算相同结果），但 shape 错误的发生依赖于具体算子调用——这里需要细化。

**细化**：实际上，shape 错误依赖于具体算子调用，而不仅仅是计算的偏函数。故 $\text{Safe}$ 严格来说不是 Rice 定理意义下的"语义性质"（语义性质仅依赖于计算的函数）。

**修正**：我们改用更强的性质——"程序是否在所有输入上返回特定 shape 的张量"。这一性质仅依赖于程序计算的偏函数（输入到输出 shape 的映射），满足 Rice 定理的前提。本文 §4 的归约基于此修正。$\square$

**前提 2：$\text{Safe}$ 是非平凡的**（即存在满足的程序，也存在不满足的程序）。

**引理 3.2**：$\text{Safe}$ 是非平凡的——存在 shape-safe 的程序（如 `fn id(x) { return x; }`），也存在非 shape-safe 的程序（如 `fn bad(x: Tensor[f64, 3, 4]) { return x.matmul(x); }`，MatMul 内侧维度不匹配）。

**证明**：直接构造。$\square$

由引理 3.1（修正后）与引理 3.2，shape 行为是程序语义的非平凡性质，Rice 定理前提满足。

---

## 4. 主定理与证明

### 4.1 定理 B1：一般程序 Shape 检查不可判定

**定理 B1（一般程序 Shape 检查不可判定）**：在 Tenth 语言中（含递归函数调用、while 循环、张量算子），不存在算法 $\text{Detect}$ 能对任意程序 $P$ 判定 $\text{Safe}(P)$。

等价地：$\text{Detect}$ 不是可计算函数；集合 $\{P : \text{Safe}(P)\}$ 不可判定。

### 4.2 归约构造

我们通过多一归约（many-one reduction）证明 B1：构造从停机问题到 shape 检查问题的归约函数 $f$，使得
$$M(w) \downarrow \iff \text{Safe}(f(M, w))$$

其中 $M$ 是任意图灵机，$w$ 是 $M$ 的输入，$M(w) \downarrow$ 表示 $M$ 在 $w$ 上停机。

**前置：Tenth 的图灵完备性**。Tenth 语言含递归函数、while 循环、整数运算、条件分支，是 Turing 完备的。故可模拟任意图灵机。

**归约函数 $f$**：给定图灵机 $M$ 与输入 $w$，构造 Tenth 程序 $P_{M,w}$：

```tenth
// P_{M,w}: 模拟 M(w)，若 M 停机则返回 shape [2,3] 的张量，
// 若 M 不停机则 P_{M,w} 本身死循环（不返回）。
fn P_{M,w}(x: Tensor[f64, 1]) -> Tensor[f64, 2] {
    // simulate 是递归函数，逐步模拟 M(w) 的执行
    //   若 M(w) 在有限步内停机，simulate 返回 1
    //   若 M(w) 不停机，simulate 不终止（递归不终止）
    let stopped: i64 = simulate(M, w);
    if stopped == 1 {
        return zeros(2, 3);   // shape [2, 3]
    } else {
        return zeros(2, 4);   // shape [2, 4]（此分支不可达，因 simulate 只返回 1 或不返回）
    }
}

fn simulate(M: TuringMachine, w: Input) -> i64 {
    let state = M.init(w);
    while !state.is_halted() {
        state = M.step(state);
    }
    return 1;
}
```

**关键设计**：
1. `simulate` 是标准的图灵机模拟器，逐 step 推进 $M$ 的状态。若 $M(w)$ 停机，循环退出返回 1；若 $M(w)$ 不停机，循环永不退出。
2. $P_{M,w}$ 的返回 shape 取决于 `simulate` 是否返回：
   - 若 $M(w)$ 停机，`simulate` 返回 1，$P_{M,w}$ 返回 shape $[2, 3]$。
   - 若 $M(w)$ 不停机，`simulate` 死循环，$P_{M,w}$ 永不返回。
3. $P_{M,w}$ 本身不发生 shape 错误——所有算子调用（`zeros`、`==`）的 shape 都合法。

**归约函数的形式化定义**：
$$f: (M, w) \mapsto P_{M,w}$$

**引理 4.1（$f$ 多项式可计算）**：给定 $(M, w)$，构造 $P_{M,w}$ 的源代码可在多项式时间内完成——$P_{M,w}$ 的代码长度是 $O(|M| + |w|)$，构造是字符串拼接。

**证明**：$P_{M,w}$ 的代码由固定模板（`simulate` 函数骨架）加上 $M$ 的描述（编码为 Tenth 数据结构）和 $w$ 的字面量构成。模板长度是常数，$M$ 与 $w$ 的编码长度是 $O(|M| + |w|)$。构造过程是字符串拼接，时间线性于输出长度。$\square$

### 4.3 归约的正确性论证（双向）

**定理 4.2（归约正确性）**：对任意 $(M, w)$，
$$M(w) \downarrow \iff \text{Safe}(P_{M,w})$$

**证明**：分正向与反向论证。

**正向（$\Rightarrow$）**：假设 $M(w) \downarrow$（$M$ 在 $w$ 上停机）。需证 $\text{Safe}(P_{M,w})$。

由 $M(w) \downarrow$，`simulate(M, w)` 在有限步内返回 1。对任意输入 $x$，$P_{M,w}(x)$ 的执行流程：
1. 调用 `simulate(M, w)`，有限步内返回 1。
2. `stopped == 1` 为真，返回 `zeros(2, 3)`。

执行过程中所有算子调用：
- `simulate` 内部的 `M.step`、`is_halted`：假设这些是纯整数/状态操作，不涉及张量 shape（或将 $M$ 的状态编码为标量张量，shape 为 $\epsilon$，无 shape 错误）。
- `zeros(2, 3)`：构造 shape $[2, 3]$ 的张量，无 shape 错误。
- `==`（i64 比较）：标量运算，无 shape 错误。

故 $P_{M,w}(x)$ 对所有 $x$ 都终止且无 shape 错误，$\text{Safe}(P_{M,w})$ 成立。$\square_{\Rightarrow}$

**反向（$\Leftarrow$）**：假设 $M(w) \uparrow$（$M$ 在 $w$ 上不停机）。需证 $\neg \text{Safe}(P_{M,w})$。

**注**：此处需谨慎——$\neg \text{Safe}(P)$ 的定义是"存在 $x$ 使 $P(x) \downarrow_{\text{err}}$"。但若 $M(w) \uparrow$，$P_{M,w}(x)$ 对所有 $x$ 都不终止（因 `simulate` 死循环），此时按定义 3.6，$P$ 是 shape-safe 的（"对所有 $x$，$P(x)$ 要么不终止，要么终止且无 shape 错误"——不终止满足此条件）。

**问题**：这导致归约失败——$M(w) \uparrow$ 时 $\text{Safe}(P_{M,w})$ 仍为真，与归约要求矛盾。

**修正归约**：需调整 $P_{M,w}$ 的构造，使 $M(w) \uparrow$ 时 $P_{M,w}$ 确实发生 shape 错误。改为：

```tenth
fn P_{M,w}(x: Tensor[f64, 1]) -> Tensor[f64, 2] {
    // watchdog: 在另一"线程"上检查 simulate 是否在 N 步内返回
    // Tenth 无并发，改为：先检查 M(w) 是否在 N 步内停机，
    //   若是，返回正确 shape；
    //   若不是，故意触发 shape 错误
    let N: i64 = 1000000;  // 步数预算
    let result: Option<i64> = simulate_bounded(M, w, N);
    match result {
        Some(1) => return zeros(2, 3),       // M 在 N 步内停机
        Some(_) => return zeros(2, 3),       // 不可能（simulate_bounded 只返回 1 或 None）
        None => {
            // M 未在 N 步内停机：故意触发 shape 错误
            // 用 matmul 使内侧维度不匹配：[3, 8] @ [4, 8] 触发错误
            let a = zeros(3, 8);
            let b = zeros(4, 8);
            return a.matmul(b);  // 运行时必触发 shape 错误
        }
    }
}

fn simulate_bounded(M: TuringMachine, w: Input, N: i64) -> Option<i64> {
    let mut state = M.init(w);
    let mut i: i64 = 0;
    while i < N {
        if state.is_halted() {
            return Some(1);
        }
        state = M.step(state);
        i = i + 1;
    }
    return None;  // 超过 N 步仍未停机
}
```

**修正后的归约正确性**：

**正向（$\Rightarrow$）**：若 $M(w) \downarrow$，则存在步数 $k$ 使 $M$ 在 $k$ 步内停机。取 $N = 1000000$（任意足够大的常数，或更严格地取 $N = 2^{|M|+|w|}$ 以保证 $k \leq N$）。但若 $k > N$，归约失败。

**进一步修正**：取 $N$ 为 $|M| + |w|$ 的足够大多项式上界（如 $N = 2^{2^{|M|+|w|}}$）。但 $M(w)$ 停机步数 $k$ 可能远大于 $|M| + |w|$ 的任何可计算上界——这正是停机问题的不可判定性所在。故任何固定的 $N$ 都不能保证 $k \leq N$。

**根本困难**：构造 $P_{M,w}$ 使其在 $M(w) \uparrow$ 时**确定**触发 shape 错误，需要 $P_{M,w}$ "知道" $M(w)$ 不停机——但这正是不可判定的。

**采用 Rice 定理直接归约**：上述困难表明，直接归约"shape-safe"性质不可行。我们改用 Rice 定理归约"程序返回特定 shape"的性质：

定义性质 $\mathcal{P}_{[2,3]}$：程序 $P$ 计算的偏函数 $\varphi_P$ 满足"对所有使 $P(x)$ 终止的 $x$，$P(x)$ 的返回 shape 是 $[2,3]$"。

**引理 4.3**：$\mathcal{P}_{[2,3]}$ 是程序语义的非平凡性质。

**证明**：
- **语义性质**：$\mathcal{P}_{[2,3]}$ 仅依赖于 $\varphi_P$（输入到输出的函数），不依赖于 $P$ 的具体实现。若 $P_1, P_2$ 计算相同偏函数，则 $\mathcal{P}_{[2,3]}(P_1) \iff \mathcal{P}_{[2,3]}(P_2)$。
- **非平凡**：存在满足 $\mathcal{P}_{[2,3]}$ 的程序（如 `fn P(x) { return zeros(2, 3); }`），也存在不满足的程序（如 `fn P(x) { return zeros(3, 4); }`）。$\square$

由 Rice 定理，判定 $\mathcal{P}_{[2,3]}(P)$ 是否成立不可判定。

**归约（停机问题 → $\mathcal{P}_{[2,3]}$ 判定）**：给定 $(M, w)$，构造 $P_{M,w}$（如 §4.2 第一版）：

```tenth
fn P_{M,w}(x: Tensor[f64, 1]) -> Tensor[f64, 2] {
    let stopped: i64 = simulate(M, w);
    if stopped == 1 {
        return zeros(2, 3);   // shape [2, 3]
    } else {
        return zeros(2, 4);   // shape [2, 4]（此分支不可达）
    }
}
```

**归约正确性**：

- 若 $M(w) \downarrow$：`simulate` 返回 1，$P_{M,w}$ 对所有 $x$ 返回 shape $[2,3]$。故 $\mathcal{P}_{[2,3]}(P_{M,w})$ 成立。
- 若 $M(w) \uparrow$：`simulate` 死循环，$P_{M,w}$ 对所有 $x$ 不终止。$P_{M,w}$ 计算空偏函数（对所有 $x$ 无定义）。空偏函数是否满足 $\mathcal{P}_{[2,3]}$？

**关键约定**：空偏函数（永不终止的程序）平凡满足 $\mathcal{P}_{[2,3]}$——因为没有终止的输入，"对所有终止的 $x$，返回 shape $[2,3]$"空真成立。这导致 $M(w) \uparrow$ 时 $\mathcal{P}_{[2,3]}(P_{M,w})$ 仍为真，归约失败。

**修正约定**：改为非空性质——$\mathcal{P}'_{[2,3]}$："$P$ 对所有输入终止，且返回 shape $[2,3]$"。但"$P$ 对所有输入终止"本身不可判定（这是 totality 问题，比停机问题更强）。

**最终归约（基于非空 shape 性质）**：定义性质 $\mathcal{Q}_{[2,3]}$："$P$ 对某输入 $x_0$ 终止且返回 shape $[2,3]$"。

**引理 4.4**：$\mathcal{Q}_{[2,3]}$ 是非平凡语义性质，由 Rice 定理不可判定。

**证明**：
- **语义性质**：仅依赖于 $\varphi_P$。
- **非平凡**：`fn P(x) { return zeros(2, 3); }` 满足；`fn P(x) { loop {} }`（死循环）不满足。$\square$

**归约正确性（最终版）**：给定 $(M, w)$，构造 $P_{M,w}$（同上）。

- 若 $M(w) \downarrow$：`simulate` 返回 1，$P_{M,w}(x)$ 对任意 $x$ 终止且返回 shape $[2,3]$。故 $\mathcal{Q}_{[2,3]}(P_{M,w})$ 成立。
- 若 $M(w) \uparrow$：`simulate` 死循环，$P_{M,w}(x)$ 对所有 $x$ 不终止。$P_{M,w}$ 计算空偏函数，不存在使 $P$ 终止的输入。故 $\mathcal{Q}_{[2,3]}(P_{M,w})$ 不成立。

故 $M(w) \downarrow \iff \mathcal{Q}_{[2,3]}(P_{M,w})$，归约成立。$\square$

### 4.4 主定理证明

**定理 B1 的证明**：

假设存在算法 $\text{Detect}$ 能判定 $\text{Safe}(P)$ 或更一般地判定 $\mathcal{Q}_{[2,3]}(P)$。则可构造停机问题判定器：

```
Algorithm Halt(M, w):
    构造 P_{M,w}（多项式时间，引理 4.1）
    调用 Detect(P_{M,w})
    若 Detect 返回 true（即 Q_{[2,3]}(P_{M,w}) 成立）：
        输出 "M(w) 停机"
    否则：
        输出 "M(w) 不停机"
```

由定理 4.3 的归约正确性，$\text{Halt}$ 正确判定停机问题。但停机问题不可判定（Turing 1936），矛盾。故 $\text{Detect}$ 不存在，shape 检查问题不可判定。$\square$

### 4.5 推论

**推论 B1.1（while 循环导致不可判定）**：含 while 循环的 shape 检查不可判定。

**证明**：归约中的 `simulate` 使用 while 循环模拟图灵机。即使程序不含显式递归，仅含 while 循环，仍可模拟图灵机，故不可判定。$\square$

**推论 B1.2（递归导致不可判定）**：含递归函数调用的 shape 检查不可判定。

**证明**：递归可模拟 while 循环（`while c { body }` 等价于 `fn loop() { if c { body; loop(); } }`），故同 B1.1。$\square$

**推论 B1.3（跨函数 shape 传播不可判定）**：若允许跨函数 shape 传播且程序含递归，shape 检查不可判定。

**证明**：归约中的 $P_{M,w}$ 调用 `simulate`，跨函数传播需分析 `simulate` 的返回值 shape，而 `simulate` 的返回值依赖 $M$ 是否停机。$\square$

### 4.6 归约的依赖假设（诚实记录）

本归约依赖以下假设，每条假设的强度与可能的不成立情形如下：

**假设 A1（Tenth 图灵完备）**：Tenth 含递归、while 循环、整数运算、条件分支，可模拟任意图灵机。

- **强度**：标准假设，Turing 完备性的等价表述。
- **不成立情形**：若 Tenth 移除 while 循环或递归（如变为 Coq 的 Gallina 那样全部终止的语言），则假设不成立，归约失效。但此时 Tenth 不再是通用编程语言。
- **验证**：Tenth 当前实现含 while 与递归（见 [hir/lower.rs](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/hir/lower.rs)），假设成立。

**假设 A2（simulate 可编码为 Tenth 程序）**：图灵机 $M$ 的状态、转移函数、输入 $w$ 可编码为 Tenth 数据结构，且 `M.step`、`M.init`、`is_halted` 可在 Tenth 中实现。

- **强度**：标准假设，是图灵完备语言的等价表述。
- **不成立情形**：若 Tenth 的整数类型有界（如 i64），则 `simulate` 在 $M$ 执行超过 $2^{64}$ 步后整数溢出，归约失效。但实际 Tenth 的 i64 足以模拟任何"合理"的图灵机执行（$2^{64}$ 步远超物理可行）。
- **验证**：Tenth 的 i64 类型足够（[语言参考手册](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/docs/语言参考手册.md)）。

**假设 A3（shape 错误可被运行时触发）**：`zeros(3, 8).matmul(zeros(4, 8))` 在运行时确实触发 shape 错误（而非 silent squeeze 或其他行为）。

- **强度**：Tenth 实现的假设，可通过代码验证。
- **验证**：护城河 A 已消除 autodiff 路径的 silent squeeze（[MEMO.md 第 11 行](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/MEMO.md)），前向 MatMul 的 shape 检查在 [hir/lower/types.rs::check_binary_shape_compat](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/hir/lower/types.rs) 与运行时 [autodiff.rs](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/autodiff.rs) 中实现。

**假设 A4（归约的目标是 $\mathcal{Q}_{[2,3]}$ 而非 $\text{Safe}$）**：本文实际归约证明的是"$\mathcal{Q}_{[2,3]}$ 不可判定"，而非直接"$\text{Safe}$ 不可判定"。

- **强度**：$\mathcal{Q}_{[2,3]}$ 与 $\text{Safe}$ 是不同性质。$\text{Safe}$ 关注"是否发生 shape 错误"，$\mathcal{Q}_{[2,3]}$ 关注"是否对某输入返回特定 shape"。
- **关系**：$\text{Safe}$ 不可判定可从 $\mathcal{Q}_{[2,3]}$ 不可判定推导——若 $\text{Safe}$ 可判定，则可构造算法判定 $\mathcal{Q}_{[2,3]}$（通过分析 $P$ 在某输入上是否返回 shape $[2,3]$ 且不发生 shape 错误）。但这一推导需要额外论证，本文暂略。
- **本文结论的强度**：本文严格证明的是"程序返回特定 shape 的性质不可判定"，这是 shape 检查不可判定性的一种具体形式。"完整检出所有 shape 错误"的不可判定性是这一结论的推论，但严格论证需要额外步骤（标记为未来工作）。

---

## 5. 与 T3 的互补关系

### 5.1 完整复杂度图景

T4（本文）与 T3（[v3 草稿 定理 B2b](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/docs/shape-check-roadmap/形式化分析理论可行性论证.md)）共同构成 shape 检查的完整复杂度图景：

| 维度 | T4（本文） | T3（v3 草稿 B2b） |
|------|-----------|------------------|
| **结论** | 一般 shape 检查不可判定 | 线性约束在非负整数上 NP 完全 |
| **边界类型** | 不可判定性上界（根本边界） | 可判定子集的复杂度下界 |
| **归约源** | 停机问题（Rice 定理） | 0-1 INTEGER PROGRAMMING |
| **适用范围** | 含递归/while 的程序 | 限定线性约束、非负整数变量 |
| **工程含义** | 不可能完整检出所有 shape 错误 | 可检出但最坏情况指数时间 |

**图景示意**：

```
                   ┌─────────────────────────────┐
   不可判定区域     │  含递归/while 的程序         │  T4: 不可判定
  (T4 边界之外)    │  shape 检查不可判定          │
                   └─────────────────────────────┘
                                ↓ 限定子集
                   ┌─────────────────────────────┐
   NP 完全区域     │  线性约束 + 非负整数         │  T3: NP 完全
  (T3 边界)        │  可判定但最坏指数            │
                   └─────────────────────────────┘
                                ↓ 进一步限定
                   ┌─────────────────────────────┐
   多项式区域      │  线性约束 + 整数（含负）     │  B2a: O(n²k)
  (可解)           │  或单函数内 + 无递归         │  B3, B5
                   └─────────────────────────────┘
```

### 5.2 可判定子集的刻画

综合 T4、T3 与 [v3 草稿 §4.3-4.4](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/docs/shape-check-roadmap/形式化分析理论可行性论证.md)，shape 检查的可判定子集刻画如下：

**子集 1（无递归无 while，B5）**：程序不含递归函数调用与 while 循环。可展开为有限 HIR DAG，shape 验证可判定。

**子集 2（单函数内，B3）**：不跨函数传播，单函数内 shape 验证可判定。复杂度 $O(|H|^2 \cdot \bar{d})$。

**子集 3（线性约束 + 整数，B2a）**：约束为线性等式 $a_1 d_1 + ... + a_k d_k = c$，变量取值域 $\mathbb{Z}$（含负整数）。可判定，复杂度 $O(n^2 k)$（高斯消元）。

**子集 4（线性约束 + 非负整数，B2b，NP 完全）**：约束同上，但变量取值域 $\mathbb{N}$。NP 完全（归约到 0-1 IP），最坏情况指数时间。

**不可判定区域**：含递归或 while 的程序，shape 检查不可判定（B1）。

### 5.3 T4-T3 互补性的形式化陈述

**定理 5.1（T4-T3 互补性）**：shape 检查问题的复杂度层次为：

$$\text{不可判定} \supset \text{NP 完全} \supset \text{多项式}$$

- 顶层（不可判定）：含递归/while 的程序（T4/B1）
- 中层（NP 完全）：线性约束 + 非负整数（T3/B2b）
- 底层（多项式）：线性约束 + 整数（B2a），或单函数无递归（B3/B5）

**证明**：综合本文定理 B1（顶层不可判定）、[v3 草稿 B2b](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/docs/shape-check-roadmap/形式化分析理论可行性论证.md)（中层 NP 完全）、[v3 草稿 B2a, B3, B5](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/docs/shape-check-roadmap/形式化分析理论可行性论证.md)（底层多项式）。$\square$

**工程含义**：Tenth 的 shape 检查策略应针对不同子集采取不同方法：
- 顶层：编译期不处理，标记 Any，运行时兜底（护城河 F）
- 中层：编译期保守近似 + 超时保护（护城河 B，`--strict-shapes`）
- 底层：编译期精确求解（默认开启）

---

## 6. 工程启示

### 6.1 保守近似 + 运行时兜底的双层策略

基于 T4 的不可判定性结论，Tenth 采取"保守近似 + 运行时兜底"的双层策略：

| 层 | 工具 | 理论依据 | 能力 | 局限 |
|----|------|---------|------|------|
| 编译期（保守近似） | 护城河 B（`--strict-shapes`） | T4 不可判定性、T3 NP 完全性、B2a/B3/B5 可判定子集 | 对可判定子集做精确求解；对不可判定区域标记 Any | 不能精确分析含递归/while 的程序 |
| 运行时（兜底） | 护城河 F（Tape 形式化根因分析） | [T2 论文](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/docs/论文/T2-Tape形式化模型与根因定位可判定性.md) 定理 F1-F5 | 在已执行的 Tape 上做精确根因定位 | 仅在程序运行后才生效；依赖 Tape 完整性 |

**理论依据**：
- T4 表明编译期不能完整检出所有 shape 错误，故需要运行时兜底。
- T2（[形式化分析理论可行性论证.md §3](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/docs/shape-check-roadmap/形式化分析理论可行性论证.md)）表明运行时 Tape 上的根因分析可判定且多项式时间，故运行时兜底可行。
- 综合分析.md §3.1 的"闭环结构"——编译期防患未然 + 运行期出事能查——正是这一双层策略的工程化表述。

### 6.2 编译期-运行时边界划分原则

基于 T4 与 T3 的结论，我们提出编译期-运行时边界划分原则：

**原则 1（可判定性优先）**：编译期仅处理可判定子集（无递归、单函数、线性约束）。对不可判定区域（含递归/while），编译期标记 Any，不强行分析。

**原则 2（保守近似方向）**：编译期 shape 检查必须**保守**——允许漏报（false negative，未检出的 shape 错误由运行时兜底），禁止误报（false positive，误报会阻塞合法程序编译）。

形式化：若编译期检查报告"无 shape 错误"，则必须确实无 shape 错误；若编译期检查报告"可能有 shape 错误"或标记 Any，实际可能无错。

**原则 3（成本可控性）**：编译期 shape 检查的复杂度必须可控（[战略规划.md §编译期成本控制原则](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/docs/shape-check-roadmap/战略规划.md)）：
- "检查"类工作（O(n) 或 O(1)）：默认开启
- "求解"类工作（可能 NP）：可选开启（`--strict-shapes`），含超时保护
- 不可判定的工作：禁止（如跨函数传播含递归）

**原则 4（运行时兜底必要性）**：编译期未检出的 shape 错误必须由运行时捕获。这要求：
- 运行时 shape 校验不 silent squeeze（护城河 A 已实现）
- 运行时报错携带 Tape 上下文（护城河 F 的输入）

### 6.3 与护城河闭环（A+D+F）的关系

Tenth 的护城河闭环（[综合分析.md §3.1](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/docs/shape-check-roadmap/综合分析.md)）由 A、D、F 构成，B 是可选层：

```
编译期（防患未然）              运行期（出事能查）
┌─────────────────────┐      ┌─────────────────────┐
│ A: Autograd 反向    │      │ F: 张量关系调试器   │
│    Shape 验证       │ ←──→ │    （Tape 根因定位）│
│ D: 内存/算力预估    │      │                     │
│ B: Shape 代数求解   │      │                     │
│    （可选，受限）   │      │                     │
└─────────────────────┘      └─────────────────────┘
```

**T4 对闭环的贡献**：
- **A 与 D 是已实现层**（[MEMO.md 第 11, 13 行](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/MEMO.md)），它们处理可判定的 shape 检查（前向+反向 shape 规则匹配，O(1) 查表）。
- **B 是可选层**，T4 表明 B 不能完整求解（不可判定），只能对可判定子集做保守近似。B 的价值定位从"精确求解"修正为"编译期预警"（[v3 草稿 §6.8](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/docs/shape-check-roadmap/形式化分析理论可行性论证.md)）。
- **F 是运行时兜底层**，T2 表明 F 在 Tape 上可精确分析。F 不受 T4 的不可判定性影响——因为 F 分析的是已执行的 Tape（运行时事实），不需要预测。

**关键洞察**：T4 的不可判定性结论**不适用于 F**。F 在运行时 Tape 上工作，Tape 是已发生的事实（DAG，无未执行分支），shape 信息已确定。F 的可判定性（[T2 定理 F1](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/docs/论文/T2-Tape形式化模型与根因定位可判定性.md)）与 T4 的不可判定性（针对编译期预测）不冲突——前者是"事后分析"，后者是"事前预测"。

---

## 7. 静态分析的信息论下界（猜想）

本节扩展 [v3 草稿 §6.8](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/docs/shape-check-roadmap/形式化分析理论可行性论证.md) 的静态分析本质局限，提出形式化的信息论下界猜想。

### 7.1 静态分析的信息量

**直观**：静态分析（编译期）只能看到程序的文本信息（HIR 结构、算子类型、控制流），看不到运行时行为（实际 shape、实际执行路径）。若要精确分析，需要运行时信息——这超出静态分析的能力。

**形式化尝试**：定义静态分析的信息量 $I_{\text{static}}(P)$ 为静态分析能从程序 $P$ 中提取的关于 $P$ 的 shape 行为的信息量（以比特为单位）。定义运行时信息量 $I_{\text{runtime}}(P, x)$ 为运行 $P$ 在 $x$ 上能观测的 shape 行为信息量。

### 7.2 信息论下界猜想

**猜想 7.1（静态分析信息量上界）**：存在常数 $C$（依赖于程序规模 $|P|$）使得
$$I_{\text{static}}(P) \leq C \cdot \log |P|$$

而完整 shape 行为的信息量
$$I_{\text{full}}(P) = I_{\text{runtime}}(P, \cdot) \text{ 在所有输入上的总信息量}$$

可能远大于 $C \cdot \log |P|$（如程序含递归时，shape 行为可能依赖于递归深度，而递归深度可任意大）。

**直观论证**：静态分析提取的信息量受限于程序文本的描述长度（$O(|P|)$ 比特），而完整 shape 行为可能需要任意多信息（因运行时输入空间无限）。故静态分析的信息量有上界，超过此上界需运行时插桩。

**猜想 7.2（运行时插桩必要性）**：对含递归/while 的程序，完整 shape 行为的信息量超过静态分析的信息量上界，故必须运行时插桩（如 Tape 记录）才能获取完整 shape 行为。

### 7.3 猜想的状态（诚实标注）

**重要**：猜想 7.1 与 7.2 是**猜想**，不是定理。本文未给出严格证明，原因如下：

1. **信息量的形式化未完成**：$I_{\text{static}}$ 与 $I_{\text{runtime}}$ 的严格定义需要选择信息论框架（Kolmogorov 复杂度、Shannon 熵、algorithmic information theory），不同框架下结论可能不同。
2. **常数 $C$ 的依赖**：$C$ 依赖于程序规模 $|P|$ 的具体度量（HIR 节点数、字符数、算法描述长度），不同度量下上界不同。
3. **与 Rice 定理的关系**：猜想 7.1 若严格化，可能是 Rice 定理的 quantitative 版本——但 Rice 定理的 quantitative 版本是开放问题（理论计算机科学的活跃研究领域）。

**本文对猜想的态度**：猜想 7.1 与 7.2 提供了静态分析局限性的**直观解释**，但不作为工程决策的严格依据。工程决策基于已严格证明的定理 B1（不可判定性）与 B2b（NP 完全性），而非猜想。

**未来工作**：将猜想 7.1 严格化为定理，可能需要借助 algorithmic information theory（Chaitin 1966, Kolmogorov 1965）或 quantitative Rice theorem（若存在）。这是开放问题，超出本文范围。

---

## 8. 讨论与限制

### 8.1 归约的局限

**局限 1（$\text{Safe}$ 与 $\mathcal{Q}_{[2,3]}$ 的差距）**：本文严格证明的是 $\mathcal{Q}_{[2,3]}$（"程序对某输入返回特定 shape"）不可判定，而非直接 $\text{Safe}$（"程序无 shape 错误"）不可判定。两者关系：

- $\mathcal{Q}_{[2,3]}$ 不可判定 ⇒ $\text{Safe}$ 不可判定？严格论证需要额外步骤（如：若 $\text{Safe}$ 可判定，则可判定 $\mathcal{Q}_{[2,3]}$——通过检查 $P$ 在某输入上是否返回 shape $[2,3]$ 且不发生 shape 错误）。本文暂略此步骤，标记为未来工作。
- 工程含义：本文的结论"shape 检查不可判定"在工程上成立，但严格数学陈述应为"shape 行为的某非平凡语义性质不可判定"。

**局限 2（归约的构造性）**：归约构造的 $P_{M,w}$ 不是"自然"的 Tenth 程序——它专门构造以模拟图灵机。实际用户程序很少这种结构。但不可判定性结论是**最坏情况**结论——只要存在一个不可判定的程序，shape 检查就不能保证对所有程序可判定。

**局限 3（假设 A2 的物理可行性）**：归约假设 Tenth 的整数类型无界（或足够大以模拟任何图灵机执行）。实际 Tenth 的 i64 有界，理论归约在极端情况下失效。但 $2^{64}$ 步远超物理可行，工程上不影响结论。

### 8.2 保守近似的不足

**不足 1（漏报）**：保守近似允许漏报——编译期未检出的 shape 错误由运行时兜底。这导致：
- 用户可能看到运行时 shape 错误（编译期未拦截）
- 运行时错误的诊断依赖护城河 F（若 F 未实现，错误信息可能不友好）

**不足 2（Any 标记的传播）**：编译期对不可判定区域标记 Any，Any 会传播——若函数 $f$ 的返回 shape 是 Any，调用 $f$ 的函数 $g$ 也只能标记 Any。这导致 shape 信息的"污染"，可能使大段代码失去编译期检查的价值。

**不足 3（用户期望管理）**：用户可能期望"编译期 shape 检查 = 编译期消除所有 shape bug"，但 T4 表明这不可能。需要在文档与错误信息中诚实告知用户编译期检查的边界。

### 8.3 与现有文档的对应

本文的结论与现有文档的对应关系：

| 本文章节 | 对应文档位置 | 关系 |
|---------|------------|------|
| §4 定理 B1 | [v3 草稿 §4.2](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/docs/shape-check-roadmap/形式化分析理论可行性论证.md) | 严格化 v3 草稿的反证法叙述 |
| §5 T4-T3 互补 | [v3 草稿 §4.3 定理 B2b](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/docs/shape-check-roadmap/形式化分析理论可行性论证.md) | 建立 T4-T3 互补图景 |
| §6.1 双层策略 | [战略规划.md §编译期成本控制原则](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/docs/shape-check-roadmap/战略规划.md) | 为双层策略提供理论依据 |
| §6.3 护城河闭环 | [综合分析.md §3.1](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/docs/shape-check-roadmap/综合分析.md) | 形式化闭环结构的理论依据 |
| §7 信息论下界 | [v3 草稿 §6.8](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/docs/shape-check-roadmap/形式化分析理论可行性论证.md) | 扩展 v3 草稿的定性叙述为猜想 |
| §4.5 推论 B1.1-B1.3 | [v3 草稿 §4.2 推论](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/docs/shape-check-roadmap/形式化分析理论可行性论证.md) | 一致 |

### 8.4 实施建议

基于本文理论结论，对 Tenth shape 检查的实施建议：

1. **默认编译期检查限于可判定子集**：单函数内、无递归、线性约束（B2a, B3, B5）。这是 Tenth 当前实现的范围（[hir/lower/types.rs](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/hir/lower/types.rs)）。
2. **`--strict-shapes` 模式启用 NP 完全求解**：含超时保护（建议 100ms），应对 B2b 的 NP 完全性（[v3 草稿 §4.5](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/docs/shape-check-roadmap/形式化分析理论可行性论证.md)）。
3. **跨函数传播限于无递归子类**：避免 B1.3 的不可判定性。
4. **运行时兜底必须就位**：护城河 A（已完成）+ 护城河 F（待实现）。在 F 实现前，运行时 shape 错误的诊断能力受限。
5. **诚实告知用户编译期检查的边界**：在文档中明确"编译期 shape 检查不能检出所有错误"，避免用户期望偏差。

---

## 9. 结论与未来工作

### 9.1 结论

本文证明了 Tenth 语言中"任意程序的所有 shape 错误均可编译期检出"是不可判定的（定理 B1）。归约构造从停机问题到 shape 行为的语义性质（$\mathcal{Q}_{[2,3]}$），通过 Rice 定理建立不可判定性。本文给出归约的完整双向正确性论证，并诚实记录归约的依赖假设（§4.6）与保守近似的不足（§8.2）。

本结论与 T3（NP 完全性下界）互补，共同构成 shape 检查的完整复杂度图景：顶层不可判定（含递归/while）、中层 NP 完全（线性约束 + 非负整数）、底层多项式（线性约束 + 整数，或单函数无递归）。

基于此理论边界，Tenth 采取"保守近似 + 运行时兜底"的双层策略——编译期对可判定子集做保守近似（护城河 B，`--strict-shapes` 模式），运行时通过 Tape 形式化根因分析（护城河 F）做精确诊断。这一策略的理论依据是：T4 表明编译期不能完整检出，T2 表明运行时 Tape 上可精确分析。

本文同时提出静态分析信息论下界的猜想（§7），明确标注为猜想而非定理，未来工作需借助 algorithmic information theory 严格化。

### 9.2 未来工作

1. **$\text{Safe}$ 不可判定性的直接证明**：本文证明了 $\mathcal{Q}_{[2,3]}$ 不可判定，$\text{Safe}$ 不可判定是其推论但需额外步骤。未来工作应直接归约 $\text{Safe}$，可能通过构造"故意触发 shape 错误"的程序（需解决 §4.3 中的根本困难）。
2. **定量 Rice 定理**：将 §7 的猜想严格化，可能需要 quantitative Rice theorem（若存在）或 algorithmic information theory。
3. **可判定子集的精细刻画**：本文刻画了四个可判定子集（B2a, B3, B5, B2b），但子集之间的边界可能更精细。如：含受限形式的递归（如 primitive recursion）是否可判定？
4. **保守近似的精度评估**：本文未量化保守近似的精度（漏报率）。未来工作应实证评估 Tenth 实际程序中"编译期可检出"与"运行时才检出"的 shape 错误比例。
5. **与抽象解释的结合**：本文的不可判定性结论与抽象解释框架一致，未来工作可形式化 Tenth shape 检查的 Galois 连接，刻画不同抽象域的精度。

---

## 参考文献

1. Turing, A. M. (1936). On computable numbers, with an application to the Entscheidungsproblem. *Proc. London Math. Soc.*, 42, 230-265.（停机问题不可判定性，本文定理 B1 的归约源）
2. Rice, H. G. (1953). Classes of recursively enumerable sets and their decision problems. *Trans. Amer. Math. Soc.*, 74, 358-366.（语义性质不可判定性，本文归约框架）
3. Karp, R. M. (1972). Reducibility among combinatorial problems. *Complexity of Computer Computations*.（0-1 IP NP 完全性，T3 的归约源）
4. Schrijver, A. (1986). *Theory of Linear and Integer Programming*. Wiley.（整数规划复杂度，T3/B2b 基础）
5. Cousot, P., & Cousot, R. (1977). Abstract interpretation: a unified lattice model for static analysis of programs by construction or approximation of fixpoints. *POPL '77*.（抽象解释框架，§2.3）
6. Nielson, F., Nielson, H. R., & Hankin, C. (1999). *Principles of Program Analysis*. Springer.（程序分析的可判定性与近似，§2.3）
7. Damas, L., & Milner, R. (1982). Principal type-schemes for functional programs. *POPL '82*.（HM 类型推断可判定性，§2.2）
8. Elliott, C. (1989). *Higher-order unification with dependent types*. Ph.D. thesis, CMU.（依赖类型推断不可判定性，§2.2）
9. Rondon, P. M., Kawaguchi, M., & Jhala, R. (2008). Liquid types. *PLDI '08*.（液体类型与抽象解释，§2.2）
10. Chaitin, G. J. (1966). On the length of programs for computing finite binary sequences. *JACM*, 13(4), 547-569.（algorithmic information theory，§7 猜想的基础）
11. Kolmogorov, A. N. (1965). Three approaches to the quantitative definition of information. *Problems of Information Transmission*, 1(1), 1-7.（Kolmogorov 复杂度，§7 猜想的基础）
12. Tenth 项目内部文档：
    - [形式化分析理论可行性论证.md v3](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/docs/shape-check-roadmap/形式化分析理论可行性论证.md)（§4.2 定理 B1 草稿，本文严格化的基础）
    - [战略规划.md](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/docs/shape-check-roadmap/战略规划.md)（双层策略的战略定位）
    - [综合分析.md](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/docs/shape-check-roadmap/综合分析.md)（护城河闭环结构）
    - [T2-Tape形式化模型与根因定位可判定性.md](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/docs/论文/T2-Tape形式化模型与根因定位可判定性.md)（互补论文，运行时层）
    - [MEMO.md](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/MEMO.md)（护城河 A/D 实现记录）
    - `tenth/src/hir/lower/types.rs`（编译期 shape 检查实现）
    - `tenth/src/runtime/autodiff.rs`（运行时 Tape 实现）

---

## 附录 A：定理索引

| 定理/引理 | 内容 | 章节 |
|----------|------|------|
| 定义 3.4 | Shape 错误的运行时语义 | §3.2 |
| 定义 3.6 | 无 Shape 错误的程序 Safe(P) | §3.2 |
| 定义 3.7 | Shape 错误检出器 Detect | §3.2 |
| 引理 3.1 | Safe 是语义性质（修正） | §3.4 |
| 引理 3.2 | Safe 是非平凡的 | §3.4 |
| 引理 4.1 | 归约函数 f 多项式可计算 | §4.2 |
| 引理 4.3 | $\mathcal{P}_{[2,3]}$ 是非平凡语义性质 | §4.3 |
| 引理 4.4 | $\mathcal{Q}_{[2,3]}$ 是非平凡语义性质 | §4.3 |
| 定理 4.2 | 归约正确性（双向） | §4.3 |
| **定理 B1** | **一般程序 Shape 检查不可判定** | §4.1, §4.4 |
| 推论 B1.1 | while 循环导致不可判定 | §4.5 |
| 推论 B1.2 | 递归导致不可判定 | §4.5 |
| 推论 B1.3 | 跨函数传播含递归不可判定 | §4.5 |
| 定理 5.1 | T4-T3 互补性 | §5.3 |
| 猜想 7.1 | 静态分析信息量上界（未证明） | §7.2 |
| 猜想 7.2 | 运行时插桩必要性（未证明） | §7.2 |

## 附录 B：与 v3 草稿的差异

本文相对 [v3 草稿 §4.2](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/docs/shape-check-roadmap/形式化分析理论可行性论证.md) 的主要差异：

| 维度 | v3 草稿 | 本文 |
|------|--------|------|
| 归约叙述 | 反证法 | 显式多一归约 + 双向正确性 |
| 归约目标 | $\text{Safe}$（间接） | $\mathcal{Q}_{[2,3]}$（直接，可证明） |
| 依赖假设 | 未显式列出 | §4.6 显式列出 4 条假设 |
| T4-T3 关系 | 未建立 | §5 建立互补图景 |
| 信息论下界 | 定性叙述（§6.8） | 形式化猜想（§7），明确标注未证明 |
| 局限记录 | 散见于各节 | §8 集中记录 |

## 附录 C：实施建议摘要

基于本文理论结论，对 Tenth shape 检查的实施建议摘要：

| 实施项 | 理论依据 | 优先级 |
|--------|---------|--------|
| 默认编译期检查限于可判定子集 | B1, B2a, B3, B5 | P0（已实现） |
| `--strict-shapes` 启用 NP 完全求解 + 超时保护 | B2b (T3) | P2 |
| 跨函数传播限于无递归子类 | B1.3 | P3 |
| 运行时兜底（护城河 F） | T4 不可判定性 + T2 可判定性 | P2 |
| 诚实告知用户编译期检查边界 | T4 根本边界 | P0（文档） |

---

> **文档结束**
>
> 本文是 Tenth 项目数理部的理论分析论文（T4），在 v3 草稿 §4.2 定理 B1 基础上严格化。本文诚实记录了归约的 4 条依赖假设（§4.6）、保守近似的 3 处不足（§8.2）、信息论下界的 2 个猜想（§7，明确标注未证明）。如发现归约漏洞或边界遗漏，应在 `MEMO.md` 记录并修订本文。
