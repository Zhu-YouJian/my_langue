# dtype 提升格与混合 dtype 算术：Tenth 类型提升代数的形式化与跨语言对比

> **作者**：Tenth 项目数理部
> **日期**：2026-07-02
> **类型**：理论分析论文（T17 理论点）
> **实证基础**：Tenth v0.3.3+ 源码（`hir/types.rs`、`hir/lower/types.rs`、`runtime/vm.rs`、`runtime/tensor.rs`、`runtime/interpreter/binary.rs`）
> **关联文档**：`docs/语言参考手册.md` §4.3 隐式转换规则、`MEMO.md`、`能力梳理/能力全梳理.md`
> **版本**：v1（首轮分析，含 4 轮自审修正留痕）

---

## 摘要

本文对 Tenth 语言的 dtype 提升规则进行形式化建模与代数性质分析。Tenth 的隐式类型提升函数 `promote_float_dtype` 通过 match 分支优先级定义了浮点 dtype 上的偏序关系 `BF16 < F16 < F32 < F64`，在 VM 的 `add_priv`/`sub_priv`/`mul_priv`/`div_priv` 四个算子中散布着共 56 个数值分支（每个算子 14 个有序类型对分支）。我们证明：(1) 浮点 dtype 子集构成**有限分配格**（定理 P1，实为 4 元链）；(2) 运行时标量域 {I64, F32, F64} 构成 3 元链上的分配格；(3) 提升规则在浮点子集上**无精度损失回路**（定理 P2）；(4) HIR 层整数 fallback 破坏交换性，全 `BaseType` 集上**不构成格**（定理 P3，给出反例）；(5) 跨语言对比显示 Tenth 的浮点提升规则与 JAX 同属"严格提升"家族，优于 C/Java 的"窄整数提升"与 NumPy 的"弱提升"（定理 P4）；(6) broadcast+promotion 复合代数在张量层级保持格性质，但标量-张量混合为**吸收运算**而非 join（定理 P5）。穷尽性验证发现：(Int, Tensor) 和 (Tensor, Int) 在全部 4 个算子中未实现（8 个缺失分支），F16/BF16 在 HIR 声明但运行时无表示（死代码路径）。本文诚实记录 7 处理论局限。

**关键词**：dtype 提升、有限分配格、混合精度算术、类型代数、broadcast、张量运算、跨语言对比、Tenth 语言

---

## 1. 引言

### 1.1 动机

在 AI 原生编程语言中，**混合 dtype 算术**是一个高频操作但工程上极易出错的领域。一个简单的 `a + b` 表达式，当 `a` 是 `f32` 张量而 `b` 是 `f64` 标量时，结果应该是 `f32` 还是 `f64`？不同语言给出了截然不同的答案：

- **C/Java**：执行"通常算术转换"，整数提升到浮点，但 `float` 与 `double` 运算时 `float` 提升为 `double`。
- **NumPy**：维护一张 promotion table，`f32 ⊕ f64 → f64`，但 `int64 ⊕ uint64` 会报错（无安全提升）。
- **PyTorch**：引入"弱提升"概念，允许 `tensor(f32) + python_int` 保持 `f32`，但 `tensor(f32) + tensor(f64)` 提升为 `f64`。
- **JAX**：采用严格提升，标量与张量运算时跟随张量 dtype，但标量间运算用 NumPy 规则。

Tenth 语言作为 AI 原生语言，其设计需要在**数值精度安全**与**使用便利性**之间取得平衡。当前 v0.3.3 的提升规则由 `promote_float_dtype` 函数（[hir/lower/types.rs:480-489](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/hir/lower/types.rs)）和 VM 中 4 个算子函数的 match 分支共同实现，总计 56+ 个数值分支散落在运行时代码中。这种分散式实现存在两个理论问题：

1. **代数性质不明**：提升运算是否构成格？是否满足结合律、交换律？是否存在精度损失回路？
2. **完备性边界不明**：56 个分支是否覆盖了所有合法类型组合？缺失的分支是设计决策还是 bug？

### 1.2 研究问题

本文回答以下五个研究问题：

- **RQ1**：Tenth 的 dtype 提升运算在数学上构成何种代数结构？是否为有限分配格？
- **RQ2**：提升规则是否存在精度损失回路（`a ≤ b ≤ a ⟹ a = b`）？
- **RQ3**：VM 中 56 个数值分支是否穷尽覆盖了所有合法类型组合？缺失分支的影响是什么？
- **RQ4**：Tenth 的提升规则与 C/Java/NumPy/PyTorch/JAX 相比，哪些更健全，哪些不健全？
- **RQ5**：broadcast 与 promotion 复合后，代数性质如何变化？

### 1.3 贡献

- **格论形式化**（§3）：将 `promote_float_dtype` 抽象为偏序集上的 join 运算，证明浮点子集构成 4 元链（有限分配格）。
- **健全性证明**（§4）：证明浮点提升无精度损失回路、无下溢。
- **非对称性实证**（§4）：发现并证明 HIR 层整数 fallback 破坏交换性，全 `BaseType` 集不构成格。
- **跨语言对比**（§4）：系统对比 6 种语言的提升规则，定位 Tenth 的健全性优势。
- **穷尽性验证**（§5）：逐分支核对 56 个数值分支，发现 (Int, Tensor) 的 8 个缺失分支和 F16/BF16 的死代码路径。
- **复合代数分析**（§6）：证明标量-张量混合为吸收运算，张量-张量为格 join。

### 1.4 v1 自审留痕

本文经历 4 轮自审，主要修正：

| 轮次 | 原始断言 | 修正 |
|------|---------|------|
| 第 1 轮（结构） | 声称"全 BaseType 集构成有限分配格" | 修正：仅浮点子集构成格；整数 fallback 破坏交换性（定理 P3） |
| 第 2 轮（证明） | 定理 P1 初稿未证分配律 | 补充：链上分配律平凡成立（引理 P1.4） |
| 第 3 轮（边界） | 未处理 (Int, Tensor) 缺失分支 | 补充：穷尽性验证发现 8 个缺失分支（§5.2） |
| 第 4 轮（诚实） | 声称"F16/BF16 提升规则健全" | 修正：运行时无 F16/BF16 表示，HIR 规则为前瞻性死代码（局限 L3） |

---

## 2. 背景与相关工作

### 2.1 C/Java 的隐式转换规则

C 语言（C11 §6.3.1）定义了"通常算术转换"（usual arithmetic conversions）：

1. **整数提升**：`char`、`short` 提升为 `int`。
2. **整数转换阶**：`int < unsigned int < long < unsigned long < long long`。
3. **浮点提升**：若一侧为 `double`，另一侧提升为 `double`；若一侧为 `float`，另一侧提升为 `float`（C99+）。
4. **整数与浮点**：整数转换为浮点。

Java 的规则类似（JLS §5.6.2），但不支持无符号整数。C/Java 的规则在**整数域内是非交换的**（`int ⊕ unsigned` 的结果取决于实现定义行为），但在浮点域内是交换的。

**健全性评价**：C 的 `int ⊕ unsigned` 可能产生非预期的无符号结果（如 `-1 + 1u == 0u` 在某些情况下为 `0` 而非 `-1`），这是**不健全**的。浮点提升规则是健全的。

### 2.2 NumPy 的 promotion table

NumPy 维护一张显式的 promotion table（`numpy.promote_types`）：

- 整数域：按位宽提升，`int8 ⊕ int16 → int16`，`int32 ⊕ uint32 → int64`。
- `int64 ⊕ uint64`：**报错**（无安全提升路径）。
- 浮点域：`float16 ⊕ float32 → float32`，`float32 ⊕ float64 → float64`。
- 整数与浮点：`int ⊕ float → float`，但 `int64 ⊕ float32 → float64`（避免精度损失）。

**健全性评价**：NumPy 的 `int64 ⊕ float32 → float64` 规则是**健全的**（避免 `2^53` 以上的整数损失精度），但与 PyTorch/JAX 的"弱提升"不同。

### 2.3 PyTorch 的 type promotion

PyTorch（`torch.result_type`）引入了**弱提升**（weak promotion）：

- `tensor(f32) + python_scalar(int)` → `tensor(f32)`（标量不提升张量）。
- `tensor(f32) + tensor(f64)` → `tensor(f64)`（张量间正常提升）。
- `tensor(f32) + tensor(i64)` → `tensor(f32)`（整数张量提升到浮点）。

**健全性评价**：弱提升对 `tensor(f32) + 2^60`（python int）是**不健全**的（`2^60` 超出 f32 精确表示范围，但结果仍为 f32）。但实际使用中，python 标量通常是小整数，损失可忽略。

### 2.4 JAX 的 dtype promotion

JAX 采用**严格提升**（strict promotion）：

- 标量与张量运算时，标量被视为 0 维数组，遵循张量提升规则。
- `jnp.float32 + jnp.float64 → float64`。
- 默认禁用 `x64` 模式（`jax.config.update("jax_enable_x64", True)` 才启用 f64）。

**健全性评价**：JAX 是最健全的——标量不享受"弱提升"特权，所有运算都走张量规则。代价是便利性降低。

### 2.5 Julia 的 type promotion

Julia 通过 `promote_rule` 机制实现可扩展的提升：

- 每个 dtype 对定义 `promote_rule(T1, T2)` 返回提升后的类型。
- `promote(Float32, Float64) → (Float64, Float64)`。
- 整数与浮点：`promote(Int64, Float32) → (Float32, Float32)`（注意：`Int64 → Float32` 可能损失精度！）。

**健全性评价**：Julia 的 `Int64 → Float32` 提升是**不健全**的（`Int64(2^40)` 无法精确表示为 `Float32`）。但 Julia 的设计哲学是"速度优先"。

### 2.6 Rust 的 From/Into trait

Rust 不做隐式类型提升，而是通过 `From`/`Into` trait 显式转换：

- `i32 as f64` 是显式 cast。
- `f32 as f64` 是无损 cast（`From<f32> for f64`）。
- `f64 as f32` 是有损 cast（无 `From` 实现，必须用 `as`）。

**健全性评价**：Rust 是最健全的——所有可能有损的转换都必须显式标注。代价是代码冗长。

### 2.7 Tenth 的定位

Tenth 的提升规则介于 JAX 和 PyTorch 之间：

- **标量间运算**：严格提升（`f32 ⊕ f64 → f64`，`i64 ⊕ f32 → f32`），与 JAX 一致。
- **标量与张量运算**：张量 dtype 优先（吸收规则），与 PyTorch 弱提升一致。
- **张量间运算**：严格提升（`Tensor[f32] ⊕ Tensor[f64] → Tensor[f64]`），与 JAX/NumPy 一致。

这一混合策略的理论性质将在 §4 详细分析。

---

## 3. Tenth 提升规则形式化

### 3.1 dtype 偏序集 (D, ≤)

**定义 3.1**（dtype 域）。Tenth 的 `BaseType` 枚举（[hir/types.rs:4-10](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/hir/types.rs)）定义了 16 种基础类型：

$$D_{\text{full}} = \{\text{I8}, \text{I16}, \text{I32}, \text{I64}, \text{U8}, \text{U16}, \text{U32}, \text{U64}, \text{F16}, \text{F32}, \text{F64}, \text{BF16}, \text{Bool}, \text{Char}, \text{Str}, \text{Unit}\}$$

其中参与算术提升的子集为：

$$D_{\text{arith}} = \{\text{I8}, \text{I16}, \text{I32}, \text{I64}, \text{U8}, \text{U16}, \text{U32}, \text{U64}, \text{F16}, \text{F32}, \text{F64}, \text{BF16}\}$$

**定义 3.2**（浮点子域）。浮点 dtype 子集为：

$$D_{\text{float}} = \{\text{BF16}, \text{F16}, \text{F32}, \text{F64}\} \subset D_{\text{arith}}$$

**定义 3.3**（浮点偏序）。在 $D_{\text{float}}$ 上定义偏序 $\leq_{\text{fp}}$：

$$\text{BF16} \leq_{\text{fp}} \text{F16} \leq_{\text{fp}} \text{F32} \leq_{\text{fp}} \text{F64}$$

即 $\text{BF16}$ 是最小元，$\text{F64}$ 是最大元。此偏序的直观含义是"精度/范围提升方向"，由 `promote_float_dtype` 的 match 分支优先级隐式定义（[hir/lower/types.rs:482-486](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/hir/lower/types.rs)）：F64 分支最先匹配，故 F64 是最大元；BF16 分支最后匹配（在浮点中），故 BF16 是最小元。

**注 3.1**（BF16 vs F16 的偏序争议）。从数值精度角度，BF16（8 位指数 + 7 位尾数）与 F16（5 位指数 + 10 位尾数）**不可比较**：BF16 范围更大但精度更低，F16 精度更高但范围更小。Tenth 的 `promote_float_dtype` 强制设定 $\text{BF16} \leq_{\text{fp}} \text{F16}$，这是一个**工程决策**（F16 的尾数更宽，更适合作为"中间精度"），而非数值上的自然偏序。此决策的影响在 §7.2 讨论。

### 3.2 提升函数 promote

**定义 3.4**（HIR 层提升函数）。`promote_float_dtype: BaseType × BaseType → BaseType`（[hir/lower/types.rs:480-489](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/hir/lower/types.rs)）定义为：

$$\text{promote}(l, r) = \begin{cases} \text{F64} & \text{if } l = \text{F64} \text{ or } r = \text{F64} \\ \text{F32} & \text{if } l = \text{F32} \text{ or } r = \text{F32} \\ \text{F16} & \text{if } l = \text{F16} \text{ or } r = \text{F16} \\ \text{BF16} & \text{if } l = \text{BF16} \text{ or } r = \text{BF16} \\ l & \text{otherwise (整数 + 整数)} \end{cases}$$

**关键观察**：前 4 个分支是**对称的**（`l` 或 `r` 任一为某浮点类型即返回该类型），但第 5 个分支（整数 fallback）返回 $l$（左操作数），**非对称**。

**定义 3.5**（运行时标量提升函数）。VM 中的标量提升通过 match 分支隐式实现（[runtime/vm.rs:817-871](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/vm.rs)）。运行时标量值域为：

$$V_{\text{scalar}} = \{\text{Int}(i64), \text{Float}(f64), \text{Float32}(f32)\}$$

运行时提升函数 $\text{promote}_{\text{vm}}$ 由以下规则定义（以 `add_priv` 为例，其余算子同构）：

| $l \setminus r$ | Int | Float | Float32 |
|:---:|:---:|:---:|:---:|
| **Int** | Int | Float | Float32 |
| **Float** | Float | Float | Float |
| **Float32** | Float32 | Float | Float32 |

此表是**对称的**（$\text{promote}_{\text{vm}}(l, r) = \text{promote}_{\text{vm}}(r, l)$），因为 VM 中整数只有 `i64` 一种，不存在整数间的非对称 fallback。

### 3.3 promote_float_dtype 的规则归纳

`promote_float_dtype` 的 5 个 match 分支可归纳为以下性质：

**引理 3.1**（浮点子集上的幂等性）。对任意 $d \in D_{\text{float}}$，$\text{promote}(d, d) = d$。

**证明**。由定义 3.4，$d$ 会匹配第一个以 $d$ 为模式的分支（如 $d = \text{F32}$ 匹配第 2 分支），返回 $d$。$\square$

**引理 3.2**（浮点子集上的交换性）。对任意 $d_1, d_2 \in D_{\text{float}}$，$\text{promote}(d_1, d_2) = \text{promote}(d_2, d_1)$。

**证明**。前 4 个分支使用 `(X, _) | (_, X)` 模式，显式对称。若 $d_1, d_2 \in D_{\text{float}}$，则必匹配前 4 个分支之一（不会到达第 5 分支）。设 $d_1 \leq_{\text{fp}} d_2$（即 $d_2$ 在 match 中优先级更高），则 $\text{promote}(d_1, d_2) = d_2 = \text{promote}(d_2, d_1)$。$\square$

**引理 3.3**（浮点子集上的 join 性质）。对任意 $d_1, d_2 \in D_{\text{float}}$，$\text{promote}(d_1, d_2) = \max_{\leq_{\text{fp}}}(d_1, d_2)$，即 $\text{promote}$ 是 $(D_{\text{float}}, \leq_{\text{fp}})$ 上的 join（最小上界）。

**证明**。由引理 3.2，$\text{promote}$ 对称。设 $d_1 \leq_{\text{fp}} d_2$（WLOG），则 $d_2$ 在 match 中的分支优先级高于 $d_1$，故 $\text{promote}(d_1, d_2) = d_2$。需证 $d_2 = \text{lub}(d_1, d_2)$：

- $d_2$ 是上界：$d_1 \leq_{\text{fp}} d_2$ 且 $d_2 \leq_{\text{fp}} d_2$（自反性）。
- $d_2$ 是最小上界：唯一上界（链上任意两元素的上界为较大者）。

故 $\text{promote}(d_1, d_2) = d_2 = \text{lub}(d_1, d_2)$。$\square$

### 3.4 VM 中 56 个分支的归纳

VM 的 4 个算子函数（`add_priv`、`sub_priv`、`mul_priv`、`div_priv`）各有以下 match 分支（以 `add_priv` 为例，[runtime/vm.rs:817-871](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/vm.rs)）：

| # | 分支 | 结果 dtype | 语义 |
|---|------|-----------|------|
| 1 | (Int, Int) | Int | 整数运算 |
| 2 | (Float, Float) | Float | f64 运算 |
| 3 | (Int, Float) | Float | 整数提升为 f64 |
| 4 | (Float, Int) | Float | 整数提升为 f64 |
| 5 | (Float32, Float32) | Float32 | f32 运算 |
| 6 | (Int, Float32) | Float32 | 整数提升为 f32 |
| 7 | (Float32, Int) | Float32 | 整数提升为 f32 |
| 8 | (Float32, Float) | Float | f32 提升为 f64 |
| 9 | (Float, Float32) | Float | f32 提升为 f64 |
| 10 | (String, String) | String | 字符串拼接（仅 add） |
| 11 | (Tensor, Float) | Tensor | 标量加到张量，保持张量 dtype |
| 12 | (Float, Tensor) | Tensor | 标量加到张量，保持张量 dtype |
| 13 | (Tensor, Float32) | Tensor | f32 标量先 cast 为 f64，再按张量 dtype |
| 14 | (Float32, Tensor) | Tensor | 同上 |
| 15 | (Tensor, Tensor) | Tensor | 逐元素运算，混合 dtype 提升为 f64 |
| — | _ | error | 类型不匹配 |

**计数**：每个算子有 14 个数值分支（+1 字符串分支仅 `add_priv`）+1 fallback。4 个算子共 $14 \times 4 = 56$ 个数值分支。

**注 3.2**（任务描述的"11 分支"辨析）。任务描述称"各有 11 个 match 分支覆盖 (Int, Float, Float32, Tensor) 的二元组合"。经核对源码，实际为 **14 个有序数值分支**（不含 String 和 fallback）。"11"可能指无序类型对数：$\binom{4+1}{2} = 10$ 个无序对 + 1 个 (String, String) = 11。但实现中 (Int, Float) 与 (Float, Int) 等有序对各自独立处理（因 cast 方向不同），故实际分支数为 14。

---

## 4. 主定理与证明

### 4.1 定理 P1（有限分配格）

**定理 P1**。$(D_{\text{float}}, \leq_{\text{fp}}, \text{promote})$ 构成有限分配格，其中 $\text{promote}$ 是 join（最小上界）。

**证明**。需证四个性质：

**（1）偏序性**。$\leq_{\text{fp}}$ 是 $D_{\text{float}}$ 上的偏序：
- 自反性：$\text{BF16} \leq_{\text{fp}} \text{BF16}$ 等，平凡成立（链定义蕴含自反性）。
- 反对称性：若 $d_1 \leq_{\text{fp}} d_2$ 且 $d_2 \leq_{\text{fp}} d_1$，则 $d_1 = d_2$（链上两元素可比，故相等）。
- 传递性：$\text{BF16} \leq_{\text{fp}} \text{F16} \leq_{\text{fp}} \text{F32} \leq_{\text{fp}} \text{F64}$，传递闭包封闭。

**（2）join 存在性**。由引理 3.3，$\text{promote}(d_1, d_2) = \max(d_1, d_2)$，且链上任意两元素有最大值，故 join 存在。

**（3）meet 存在性**。meet（最大下界）$d_1 \wedge d_2 = \min(d_1, d_2)$，链上任意两元素有最小值，故 meet 存在。

**（4）分配律**。需证 $d_1 \wedge (d_2 \vee d_3) = (d_1 \wedge d_2) \vee (d_1 \wedge d_3)$。

由于 $(D_{\text{float}}, \leq_{\text{fp}})$ 是**全序集**（链），全序集上的格**平凡地满足分配律**。证明如下：

设 $d_1, d_2, d_3 \in D_{\text{float}}$，WLOG 设 $d_1 \leq_{\text{fp}} d_2 \leq_{\text{fp}} d_3$（链上可排序）。

- 左侧：$d_2 \vee d_3 = d_3$，$d_1 \wedge d_3 = d_1$。故 $d_1 \wedge (d_2 \vee d_3) = d_1$。
- 右侧：$d_1 \wedge d_2 = d_1$，$d_1 \wedge d_3 = d_1$。故 $(d_1 \wedge d_2) \vee (d_1 \wedge d_3) = d_1 \vee d_1 = d_1$。
- 左 = 右 = $d_1$。$\square$

**推论 P1.1**。运行时标量域 $(V_{\text{scalar}}, \leq_{\text{vm}}, \text{promote}_{\text{vm}})$ 构成 3 元链上的有限分配格，其中 $\text{Int} \leq_{\text{vm}} \text{Float32} \leq_{\text{vm}} \text{Float}$。

**证明**。由定义 3.5 的提升表，$\text{promote}_{\text{vm}}$ 对称且 $\text{promote}_{\text{vm}}(l, r) = \max(l, r)$。3 元链是有限分配格（同定理 P1 的论证）。$\square$

**推论 P1.2**。张量 dtype 域 $(\{\text{F32}, \text{F64}\}, \leq_{\text{fp}}, \text{promote}_{\text{tensor}})$ 构成 2 元链上的有限分配格。

**证明**。`TensorData` 枚举（[runtime/tensor.rs:7-10](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/tensor.rs)）仅有 `F32` 和 `F64` 两个变体。张量-张量运算的 dtype 提升由 `add_tensor`/`sub_tensor`/`mul_tensor`/`div_tensor` 的 `_` 分支实现：混合 dtype 提升为 F64（[runtime/tensor.rs:587-596](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/tensor.rs)）。即 $\text{promote}_{\text{tensor}}(\text{F32}, \text{F64}) = \text{F64} = \max(\text{F32}, \text{F64})$。2 元链是有限分配格。$\square$

### 4.2 定理 P2（健全性：无精度损失回路）

**定理 P2**。在 $D_{\text{float}}$ 上，提升规则满足：
1. **无回路**：不存在 $d_1, d_2 \in D_{\text{float}}$ 使得 $d_1 \leq_{\text{fp}} d_2$ 且 $d_2 \leq_{\text{fp}} d_1$ 且 $d_1 \neq d_2$。
2. **无下溢**：$\text{promote}(d_1, d_2)$ 的精度 $\geq \max(\text{prec}(d_1), \text{prec}(d_2))$，其中 $\text{prec}$ 是 dtype 的尾数位数。

**证明**。

**（1）无回路**。$(D_{\text{float}}, \leq_{\text{fp}})$ 是全序集（链），链上偏序的反对称性保证无回路。具体地：若 $d_1 \leq_{\text{fp}} d_2$ 且 $d_2 \leq_{\text{fp}} d_1$，由反对称性 $d_1 = d_2$。

**（2）无下溢**。需验证每个提升方向的精度不降低。dtype 精度表：

| dtype | 指数位 | 尾数位 | 总位 |
|-------|--------|--------|------|
| BF16 | 8 | 7 | 16 |
| F16 | 5 | 10 | 16 |
| F32 | 8 | 23 | 32 |
| F64 | 11 | 52 | 64 |

提升方向为 $\text{BF16} \to \text{F16} \to \text{F32} \to \text{F64}$。需验证每步的尾数位数单调递增：

- $\text{BF16}(7) \to \text{F16}(10)$：$7 \leq 10$ ✓（尾数增加，但指数从 8 降到 5——**范围缩小**）
- $\text{F16}(10) \to \text{F32}(23)$：$10 \leq 23$ ✓（尾数增加，指数从 5 增到 8——范围增加）
- $\text{F32}(23) \to \text{F64}(52)$：$23 \leq 52$ ✓（尾数增加，指数从 8 增到 11——范围增加）

故尾数精度单调递增。但 $\text{BF16} \to \text{F16}$ 步骤中**指数范围缩小**（8 → 5），这意味着 `BF16` 能表示的大数值（如 $10^{38}$）在 `F16` 中会溢出（`F16` 最大值约 65504）。

**结论修正**：定理 P2 的"无下溢"性质**在 BF16 → F16 方向不成立**——这是 §3.3 注 3.1 所述偏序争议的数值后果。在 $D_{\text{float}} \setminus \{\text{BF16}\}$ 上无下溢成立。

**定理 P2（修正版）**。在 $D_{\text{float}}' = \{\text{F16}, \text{F32}, \text{F64}\}$ 上，提升规则无回路且无下溢。含 BF16 时，BF16 → F16 方向存在指数范围损失。$\square$

### 4.3 定理 P3（非对称性：全 BaseType 集不构成格）

**定理 P3**。$(D_{\text{arith}}, \text{promote})$ **不构成** join-半格，因为 `promote` 在整数 fallback 上不满足交换律。

**证明**。给出构造性反例。

取 $d_1 = \text{I32}, d_2 = \text{I64}$，两者均不属于 $D_{\text{float}}$，故匹配 `promote_float_dtype` 的第 5 分支（`_ => l`）：

$$\text{promote}(\text{I32}, \text{I64}) = \text{I32} \quad \text{(返回左操作数)}$$
$$\text{promote}(\text{I64}, \text{I32}) = \text{I64} \quad \text{(返回左操作数)}$$

$$\text{promote}(\text{I32}, \text{I64}) = \text{I32} \neq \text{I64} = \text{promote}(\text{I64}, \text{I32})$$

交换律被破坏，故不构成 join-半格，更不构成格。$\square$

**推论 P3.1**。整数 fallback 还破坏结合律。

**证明**。取 $d_1 = \text{I32}, d_2 = \text{I64}, d_3 = \text{F32}$：

$$\text{promote}(\text{promote}(\text{I32}, \text{I64}), \text{F32}) = \text{promote}(\text{I32}, \text{F32}) = \text{F32}$$

$$\text{promote}(\text{I32}, \text{promote}(\text{I64}, \text{F32})) = \text{promote}(\text{I32}, \text{F32}) = \text{F32}$$

此例结合律成立。但取 $d_1 = \text{I64}, d_2 = \text{I32}, d_3 = \text{F32}$：

$$\text{promote}(\text{promote}(\text{I64}, \text{I32}), \text{F32}) = \text{promote}(\text{I64}, \text{F32}) = \text{F32}$$

$$\text{promote}(\text{I64}, \text{promote}(\text{I32}, \text{F32})) = \text{promote}(\text{I64}, \text{F32}) = \text{F32}$$

仍成立。实际上，只要有一个操作数是浮点，就会命中前 4 个分支，整数 fallback 不生效，结合律成立。结合律**仅在三个操作数全为整数时**可能被破坏：

取 $d_1 = \text{I32}, d_2 = \text{I64}, d_3 = \text{I16}$：

$$\text{promote}(\text{promote}(\text{I32}, \text{I64}), \text{I16}) = \text{promote}(\text{I32}, \text{I16}) = \text{I32}$$

$$\text{promote}(\text{I32}, \text{promote}(\text{I64}, \text{I16})) = \text{promote}(\text{I32}, \text{I64}) = \text{I32}$$

此例成立（因最外层左操作数总是 I32）。但取 $d_1 = \text{I64}, d_2 = \text{I32}, d_3 = \text{I16}$：

$$\text{promote}(\text{promote}(\text{I64}, \text{I32}), \text{I16}) = \text{promote}(\text{I64}, \text{I16}) = \text{I64}$$

$$\text{promote}(\text{I64}, \text{promote}(\text{I32}, \text{I16})) = \text{promote}(\text{I64}, \text{I32}) = \text{I64}$$

仍成立。仔细分析可知：当三个操作数全为整数时，$\text{promote}$ 的结果总是**最外层的左操作数**（因整数 fallback 返回 $l$），而结合律的两侧最外层左操作数相同（均为 $d_1$），故结果相同。

**修正结论**：整数 fallback **破坏交换律**，但**不破坏结合律**（因结合律的最外层左操作数固定）。因此 $(D_{\text{arith}}, \text{promote})$ 是**非交换的幂等半群**（band），但不是 join-半格。$\square$

**注 3.3**（运行时不受影响）。VM 中整数只有 `Int`（即 `i64`）一种，不存在整数间的非对称 fallback。因此定理 P3 的非对称性**仅在 HIR 类型推断层显现**，不影响运行时行为。但若未来 Tenth 支持 `i32` 标量值（目前 `BaseType` 已声明 `I32` 但 VM `Value` 无 `Int32` 变体），则非对称性将变成运行时 bug。

### 4.4 定理 P4（跨语言对比）

**定理 P4**。Tenth 的浮点提升规则与 JAX 同属"严格提升"家族，在浮点域上健全；整数-浮点混合提升在 $D_{\text{float}}' = \{\text{F16}, \text{F32}, \text{F64}\}$ 上健全，含 BF16 时存在指数范围损失。

**对比表**：

| 语言 | `int ⊕ float` | `f32 ⊕ f64` | `scalar ⊕ tensor` | 健全性评级 |
|------|:---:|:---:|:---:|:---:|
| C/Java | int → float | f32 → f64 | N/A | ⚠️ 整数提升有实现定义行为 |
| NumPy | int → float | f32 → f64 | 跟随张量 | ✅ `int64 ⊕ float32 → float64`（避免精度损失） |
| PyTorch | int → float | f32 → f64 | 弱提升（跟随张量） | ⚠️ 弱提升对大整数不健全 |
| JAX | int → float | f32 → f64 | 严格（跟随张量） | ✅ 最健全 |
| Julia | int → float | f32 → f64 | N/A | ⚠️ `Int64 → Float32` 可能损失精度 |
| Rust | 显式 cast | 显式 cast | N/A | ✅ 最健全（无隐式提升） |
| **Tenth** | int → float | f32 → f64 | 跟随张量 | ✅ 浮点域健全；`i64 ⊕ f32 → f32` 与 Julia 同类风险 |

**证明**。

**（1）Tenth 浮点提升健全**。由定理 P2（修正版），$D_{\text{float}}'$ 上的提升无回路且尾数精度单调递增。

**（2）Tenth 整数-浮点提升**。`i64 ⊕ f32 → f32`（由 VM 分支 6/7）。`i64` 的精确表示范围是 $[-2^{63}, 2^{63}-1]$，而 `f32` 的精确整数表示范围仅 $[-2^{24}, 2^{24}]$。因此 `i64(2^30) + f32(1.0)` 会先将 `2^30` cast 为 `f32`（损失精度），再加 `1.0`。这与 Julia 的 `Int64 → Float32` 提升同类，是**潜在不健全**的。

**对比 NumPy**：NumPy 的 `int64 ⊕ float32 → float64` 更健全（自动提升到 f64 以容纳大整数）。Tenth 选择了 `i64 ⊕ f32 → f32`（跟随 f32），代价是大整数精度损失。

**（3）Tenth 标量-张量提升**。由 [hir/lower/types.rs:155-157](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/hir/lower/types.rs)，标量与张量运算时结果 dtype 跟随张量。这与 PyTorch 的弱提升一致，但 Tenth 的标量在进入 `add_scalar` 前先 cast 为 `f64`（[runtime/vm.rs:847-862](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/vm.rs)），再按张量 dtype 处理。

**特殊情况**：`Float(f64) + Tensor[f32]` → `add_scalar(s: f64)`，内部 `F32` 分支执行 `x + (s as f32)`（[runtime/tensor.rs:515](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/tensor.rs)）。若 $s$ 超出 `f32` 范围（如 $10^{300}$），`s as f32` 会溢出为 `inf`。这是**设计决策**（保持张量 dtype）而非 bug，但用户需知晓。$\square$

### 4.5 定理 P5（broadcast + promotion 复合代数）

**定理 P5**。设 $\mathcal{T}$ 为张量类型集（dtype × shape），定义复合运算 $\otimes: \mathcal{T} \times \mathcal{T} \to \mathcal{T}$ 为"broadcast shape + promote dtype"。则：

1. **张量-张量**：$\otimes$ 在 dtype 维度上是 $(D_{\text{float}}, \vee)$ 上的 join（格运算）。
2. **标量-张量**：$\otimes$ 是**吸收运算**（结果 dtype = 张量 dtype），不是格运算。
3. **复合代数**不是格，但满足"张量 dtype 守恒"性质。

**证明**。

**（1）张量-张量**。由 [runtime/tensor.rs:571-597](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/tensor.rs)（`add_tensor`），张量-张量运算分 3 分支：
- (F64, F64) → F64
- (F32, F32) → F32
- (mixed) → F64（通过 `as_f64_view` 提升）

dtype 提升结果 = $\text{promote}_{\text{tensor}}(d_1, d_2) = \max(d_1, d_2)$，由推论 P1.2 这是格 join。shape 提升由 `broadcast_shape` 实现（NumPy 规则），与 dtype 独立。故 $\otimes$ 在 dtype 维度上是 join。

**（2）标量-张量**。由 [hir/lower/types.rs:155-157](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/hir/lower/types.rs)：

```rust
(Type::Tensor { dtype, .. }, _) | (_, Type::Tensor { dtype, .. }) => {
    Type::Tensor { dtype: dtype.clone(), dims: vec![Dim::Any] }
}
```

标量的 dtype 被忽略，结果 dtype = 张量 dtype。这是**吸收运算**：$\text{scalar} \otimes \text{Tensor}[d] = \text{Tensor}[d]$。

吸收运算不满足 join 的交换性要求（$\text{Float32} \otimes \text{Tensor}[\text{F64}] = \text{Tensor}[\text{F64}]$，但若视为 join 应为 $\text{F64}$；$\text{Float} \otimes \text{Tensor}[\text{F32}] = \text{Tensor}[\text{F32}]$，但若视为 join 应为 $\text{F64}$——两者矛盾）。

**（3）复合代数**。标量-张量的吸收规则与张量-张量的 join 规则**不可统一**为一个格运算。具体反例：

$$\text{Float32}(s) \otimes \text{Tensor}[\text{F32}] = \text{Tensor}[\text{F32}] \quad \text{(吸收)}$$

$$\text{Tensor}[\text{F32}] \otimes \text{Tensor}[\text{F32}] = \text{Tensor}[\text{F32}] \quad \text{(join, F32 ∨ F32 = F32)}$$

两者结果相同（Tensor[F32]），但语义不同（前者是吸收，后者是 join）。

不一致反例：

$$\text{Float}(f64) \otimes \text{Tensor}[\text{F32}] = \text{Tensor}[\text{F32}] \quad \text{(吸收，dtype 降级)}$$

$$\text{Tensor}[\text{F64}] \otimes \text{Tensor}[\text{F32}] = \text{Tensor}[\text{F64}] \quad \text{(join, F64 ∨ F32 = F64)}$$

标量 F64 被张量 F32 吸收（降级），但张量 F64 与张量 F32 提升为 F64。**标量与张量在 dtype 提升中地位不对等**。$\square$

**推论 P5.1**。Tenth 的复合代数是**双轨制**：标量-张量用吸收规则（PyTorch 风格），张量-张量用 join 规则（JAX/NumPy 风格）。这一设计在数值实践中合理（避免标量意外提升张量 dtype 导致内存翻倍），但在代数上不统一。

---

## 5. bug 高发区分析

### 5.1 56 个分支的穷尽性验证

VM 的 4 个算子函数各覆盖以下有序类型对（取自 $\{\text{Int}, \text{Float}, \text{Float32}, \text{Tensor}\}^2$，共 $4 \times 4 = 16$ 个有序对）：

| # | 有序对 | add_priv | sub_priv | mul_priv | div_priv |
|---|--------|:---:|:---:|:---:|:---:|
| 1 | (Int, Int) | ✓ | ✓ | ✓ | ✓ |
| 2 | (Int, Float) | ✓ | ✓ | ✓ | ✓ |
| 3 | (Int, Float32) | ✓ | ✓ | ✓ | ✓ |
| 4 | (Int, Tensor) | **✗** | **✗** | **✗** | **✗** |
| 5 | (Float, Int) | ✓ | ✓ | ✓ | ✓ |
| 6 | (Float, Float) | ✓ | ✓ | ✓ | ✓ |
| 7 | (Float, Float32) | ✓ | ✓ | ✓ | ✓ |
| 8 | (Float, Tensor) | ✓ | ✓ | ✓ | ✓ |
| 9 | (Float32, Int) | ✓ | ✓ | ✓ | ✓ |
| 10 | (Float32, Float) | ✓ | ✓ | ✓ | ✓ |
| 11 | (Float32, Float32) | ✓ | ✓ | ✓ | ✓ |
| 12 | (Float32, Tensor) | ✓ | ✓ | ✓ | ✓ |
| 13 | (Tensor, Int) | **✗** | **✗** | **✗** | **✗** |
| 14 | (Tensor, Float) | ✓ | ✓ | ✓ | ✓ |
| 15 | (Tensor, Float32) | ✓ | ✓ | ✓ | ✓ |
| 16 | (Tensor, Tensor) | ✓ | ✓ | ✓ | ✓ |

**穷尽性结论**：16 个有序对中，14 个已实现，**2 个缺失**：(Int, Tensor) 和 (Tensor, Int)。4 个算子 × 2 个缺失 = **8 个缺失分支**。

### 5.2 缺失分支分析：(Int, Tensor)

**现象**：`Int + Tensor` 在 VM 中落入 `_` fallback，返回 `"+ 类型不匹配"` 错误（[runtime/vm.rs:870](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/vm.rs)）。

**影响**：用户无法直接写 `1 + tensor`，必须先转换：`1.0 + tensor` 或 `(1 as f64) + tensor`。

**对比**：
- NumPy：`1 + np.array([1.0])` → `array([2.0])`（自动提升）。
- PyTorch：`1 + torch.tensor([1.0])` → `tensor([2.0])`（弱提升）。
- JAX：`1 + jnp.array([1.0])` → `Array([2.0], dtype=float32)`（自动提升）。

**判定**：这是**设计决策**还是 **bug**？从 [hir/lower/types.rs:155-157](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/hir/lower/types.rs) 的 HIR 推断逻辑看，`Type::Tensor { dtype, .. }` 与 `_`（任意标量）的组合返回 `Tensor { dtype, .. }`——HIR 层**预期** Int + Tensor 是合法的（结果跟随 Tensor dtype）。但 VM 层未实现对应分支，导致 HIR 推断通过但运行时报错。这是 **HIR 与 VM 的语义不一致**，属于实现层面的缺陷（bug）。

**严重性**：中等。不影响已有代码（用户会自然写 `1.0 + tensor`），但对新用户有意外性。

### 5.3 潜在 bug 模式

#### 模式 1：f32 标量与 Tensor 的双重 cast

**现象**：`Float32(s) + Tensor[t]` 在 VM 中先将 `s` cast 为 `f64`，再调用 `t.borrow().add_scalar(s_f64)`（[runtime/vm.rs:855-862](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/vm.rs)）。`add_scalar` 内部对 F32 张量执行 `x + (scalar as f32)`（[runtime/tensor.rs:515](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/tensor.rs)）。

**cast 链**：`f32 → f64 → f32`。

**精度影响**：对于 f32 可表示的值，`f32 → f64 → f32` 是**无损的**（f64 是 f32 的超集）。但浪费了一次 cast 操作。

**autodiff 影响**：录制 Tape 时，`s_tensor` 的数据被存为 `f64`（`Tensor::from_vec(vec![*s as f64], vec![1])`），与原始 `f32` 标量的 dtype 不一致。反向传播时梯度 dtype 可能与正向不一致。

#### 模式 2：f64 标量与 f32 Tensor 的精度降级

**现象**：`Float(1e300) + Tensor[f32]` → `add_scalar(1e300)` → `F32` 分支执行 `x + (1e300 as f32)` → `1e300 as f32 = inf` → 结果为 `inf`。

**影响**：静默溢出，无警告。HIR 类型检查通过（结果 dtype = F32），但数值错误。

**建议**：在 `add_scalar` 的 F32 分支中添加溢出检查，或在 HIR 层对 `Float + Tensor[F32]` 发出精度降级警告。

#### 模式 3：HIR 与 VM 的 dtype 不一致

**现象**：HIR 层 `promote_float_dtype` 支持 12 种算术类型（含 I8-I64, U8-U64, F16, F32, F64, BF16），但 VM 的 `Value` 枚举仅有 `Int`(i64), `Float`(f64), `Float32`(f32) 三种标量。`TensorData` 仅有 `F32` 和 `F64` 两个变体。

**不一致**：
- HIR 声明 `i32 + i32 → i32`，但 VM 无 `Int32` 值，实际执行 `i64 + i64 → i64`。
- HIR 声明 `f16 + f16 → f16`，但 VM 无 `Float16` 值，**无法执行**。
- HIR 声明 `bf16 + bf16 → bf16`，同上**无法执行**。

**判定**：F16/BF16 的 HIR 提升规则是**前瞻性死代码**——类型系统承诺了支持，但运行时无对应实现。这不是 bug（不会产生错误行为），但属于**实现债务**（能力梳理应标记为 ⚠️ 而非 ✅）。

### 5.4 实证发现的 bug 汇总

| # | 描述 | 严重性 | 源码位置 | 状态 |
|---|------|--------|---------|------|
| B1 | (Int, Tensor) 4 个算子共 8 个缺失分支 | 中 | vm.rs:870 | HIR 推断通过但 VM 报错 |
| B2 | f64 标量 + f32 Tensor 静默溢出 | 低 | tensor.rs:515 | 设计决策，建议加警告 |
| B3 | f32 标量 + Tensor 双重 cast | 极低 | vm.rs:855-862 | 性能浪费，无精度影响 |
| B4 | F16/BF16 HIR 规则为死代码 | 低 | types.rs:485-486 | 实现债务，非运行时 bug |

---

## 6. 混合 dtype 张量运算

### 6.1 broadcast + promotion 的复合

张量-张量运算的复合代数由两个独立运算组成：

1. **Shape broadcast**：`broadcast_shape(a, b)`（[runtime/tensor.rs:552-566](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/tensor.rs)），NumPy 风格右对齐规则。
2. **Dtype promotion**：`promote_float_dtype(d1, d2)`，浮点格上的 join。

两者**独立**：shape 广播不影响 dtype 提升，dtype 提升不影响 shape 广播。这使得复合代数是两个代数的**直积**：

$$\mathcal{T} = (D_{\text{float}}, \vee) \times (\text{Shape}, \oplus_{\text{bc}})$$

其中 $\oplus_{\text{bc}}$ 是 broadcast 运算（由 T1 论文证明为部分交换幂等么半群）。

### 6.2 复合代数的代数性质

**定理 P6**（直积代数）。张量-张量运算的复合代数 $\mathcal{T}$ 是 dtype 格与 shape 半群的直积。dtype 维度满足分配律（定理 P1），shape 维度满足幂等性（T1 定理 1）。

**证明**。由 `add_tensor` 实现（[runtime/tensor.rs:571-597](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/tensor.rs)）：

- dtype 提升：3 分支（F64⊕F64→F64, F32⊕F32→F32, mixed→F64），与 `promote_float_dtype` 一致。
- shape 广播：3 分支共用 `broadcast_shape`，与标量运算无关。

两个维度独立计算，故为直积。$\square$

**推论 P6.1**。混合 dtype 张量运算（如 `Tensor[f32] + Tensor[f64]`）的 dtype 提升为 F64（join），shape 为 broadcast 结果。这与 NumPy/PyTorch/JAX 一致。

### 6.3 标量-张量运算的吸收语义

标量-张量运算不走直积代数，而走**吸收规则**：

$$\text{scalar}(d_s) \otimes \text{Tensor}[d_t, s_t] = \text{Tensor}[d_t, s_t]$$

标量的 dtype $d_s$ 被忽略，结果 dtype = 张量 dtype $d_t$。标量值先 cast 为 $f64$（统一中间表示），再按 $d_t$ 处理。

**实现路径**（以 `Tensor + Float` 为例，[runtime/vm.rs:830-837](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/vm.rs)）：

1. VM 调用 `t.borrow().add_scalar(*s)`（`s` 是 `f64`）。
2. `add_scalar` 内部按 Tensor 的 `data` 分支：
   - `F64(a)` → `a.mapv(|x| x + s)` → `TensorData::F64`
   - `F32(a)` → `a.mapv(|x| x + (s as f32))` → `TensorData::F32`

**autodiff 影响**：录制 Tape 时，标量被包装为 `Tensor::from_vec(vec![*s], vec![1])`（F64 张量）。反向传播时，梯度的 dtype 与原始张量可能不一致（标量梯度为 F64，张量梯度为 F32），需要在 `unbroadcast` 时处理。这一问题的详细分析超出本文范围，留作未来工作。

---

## 7. 开放问题与未来工作

### 7.1 f32/f64 混合训练对收敛性的实证

Tenth 的 `f32 ⊕ f64 → f64` 规则在混合精度训练中可能导致**意外的 dtype 升级**：若模型参数为 `f32`，但某一步计算引入了 `f64`（如 `loss *= 1.0` 其中 `1.0` 被 HIR 推断为 `f64`），则后续所有计算升级为 `f64`，内存翻倍。

**实证需求**：在标准模型（Transformer、ResNet）上测试 f32/f64 混合训练的收敛性与内存占用，量化"意外升级"的频率与影响。

### 7.2 BF16 的特殊处理

如 §3.3 注 3.1 所述，`BF16 < F16` 的偏序在数值上不自然（BF16 范围更大但精度更低）。两个可能的改进方向：

1. **BF16 与 F16 不可比**：提升规则改为 `BF16 ⊕ F16 → F32`（提升到两者都能容纳的最近类型），类似 NumPy 的 `int64 ⊕ uint64` 处理。
2. **BF16 > F16**：考虑到 BF16 的范围更大（与 F32 相同的指数位），将 BF16 排在 F16 之上。

当前实现选择了 `BF16 < F16`（F16 分支优先于 BF16 分支），这是**保守的**（优先保留精度而非范围）。需实证验证此选择对训练稳定性的影响。

### 7.3 (Int, Tensor) 缺失分支的补全

建议在 VM 的 4 个算子中补全 (Int, Tensor) 和 (Tensor, Int) 分支，实现方式：

- `Int(x) + Tensor(t)` → `t.borrow().add_scalar(*x as f64)`（与 `Float + Tensor` 同路径，但 cast 方向为 `i64 → f64`）。

这一补全与 HIR 推断（[hir/lower/types.rs:155-157](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/hir/lower/types.rs)）一致，消除 HIR-VM 语义不一致。

### 7.4 F16/BF16 运行时支持的实现路径

当前 F16/BF16 仅存在于 HIR 类型系统，运行时无对应表示。实现路径：

1. 在 `Value` 枚举中添加 `Float16(f16)` 和 `BFloat16(bf16)` 变体（需引入 `half` crate）。
2. 在 `TensorData` 中添加 `F16(ArrayD<f16>)` 和 `BF16(ArrayD<bf16>)` 变体。
3. 在 VM 的 4 个算子中添加 F16/BF16 的 match 分支（每个算子新增约 10 个分支）。

此为大型重构，建议在 Phase 6+ 规划。

### 7.5 整数 fallback 的对称化

当前 `promote_float_dtype(I32, I64) = I32`（返回左操作数）破坏交换性。建议改为按位宽提升：

- `I32 ⊕ I64 → I64`（按位宽大的提升）
- `I32 ⊕ U32 → I64`（有符号 + 无符号 → 更大的有符号）

这与 NumPy 的整数提升规则一致，但需要显式定义整数偏序。

---

## 8. 局限（诚实披露）

### L1. BF16 < F16 偏序的数值不自然性

**是什么**：`promote_float_dtype` 强制设定 `BF16 ≤ F16`，但 BF16 的指数范围（8 位）大于 F16（5 位）。

**影响**：`BF16(1e38) ⊕ F16(1.0) → F16`，但 `1e38` 无法表示为 F16（溢出为 inf）。当前运行时无 BF16/F16 实现，此问题不会显现，但未来实现时将成为真实 bug。

**缓解**：未来实现 BF16/F16 运行时时，应重新审视此偏序，考虑改为 `BF16 ⊕ F16 → F32`。

### L2. 整数 fallback 的非交换性未在运行时显现

**是什么**：HIR 层 `promote_float_dtype(I32, I64) = I32 ≠ I64 = promote_float_dtype(I64, I32)`，但 VM 仅有 `Int`(i64) 一种整数标量，非交换性不显现。

**影响**：若未来 VM 添加 `Int32` 值类型，非对称性将变成运行时 bug（`i32 + i64` 与 `i64 + i32` 结果类型不同）。

**缓解**：在添加多整数类型前，先对称化 `promote_float_dtype` 的整数 fallback（§7.5）。

### L3. F16/BF16 为前瞻性死代码

**是什么**：HIR 的 `promote_float_dtype` 处理 F16/BF16，但 VM `Value` 和 `TensorData` 无对应变体。

**影响**：`f16 + f16` 的 HIR 类型推断返回 `f16`，但运行时无法执行。若编译期不报错，运行时将 panic 或回退到错误路径。

**缓解**：在 HIR 层添加"运行时未实现"检查，对 F16/BF16 操作发出编译期错误或警告。

### L4. 标量-张量吸收规则的 autodiff 一致性未证明

**是什么**：标量与张量运算时，标量被 cast 为 f64 并包装为 F64 张量录制到 Tape。反向传播时，梯度 dtype 可能与原始张量不一致。

**影响**：梯度累加时可能出现 F64 梯度加到 F32 参数上，导致 dtype 不匹配或精度损失。

**缓解**：需在 `autodiff.rs` 中分析 `unbroadcast` 的 dtype 处理路径，证明梯度 dtype 一致性或添加显式 cast。留作未来工作。

### L5. 分配律仅在浮点链上证明

**是什么**：定理 P1 的分配律证明利用了"链上分配律平凡成立"的性质。若未来 Tenth 引入不可比的 dtype（如复数 `Complex32` 与 `Float32`），分配律可能不成立。

**影响**：当前无影响（dtype 集为链），但扩展 dtype 集时需重新验证。

**缓解**：扩展 dtype 集时，重新审视格结构，确认是否仍为分配格。

### L6. 跨语言对比基于文档而非实测

**是什么**：§4.4 的跨语言对比基于各语言的官方文档/规范，未进行实际代码测试。

**影响**：可能遗漏版本差异或实现 bug（如 NumPy 某版本的 promotion table 与文档不一致）。

**缓解**：未来进行实证测试，编写跨语言 dtype 提升测试套件。

### L7. (Int, Tensor) 缺失分支的严重性判定为主观

**是什么**：§5.2 将 (Int, Tensor) 缺失判定为"中等严重性"，这是基于经验判断而非用户报告数据。

**影响**：实际严重性可能更高（若用户频繁遇到）或更低（若用户自然写 `1.0 + tensor`）。

**缓解**：收集用户反馈或分析现有代码库中 `int + tensor` 的出现频率。

---

## 9. 结论

本文对 Tenth 语言的 dtype 提升规则进行了形式化建模与代数性质分析，得出以下核心结论：

1. **浮点子集构成有限分配格**（定理 P1）：$(D_{\text{float}}, \leq_{\text{fp}}, \text{promote})$ 是 4 元链，平凡满足分配律。
2. **运行时标量域构成 3 元链**（推论 P1.1）：$(V_{\text{scalar}}, \leq_{\text{vm}}, \text{promote}_{\text{vm}})$ 是 $\text{Int} < \text{Float32} < \text{Float}$ 的链。
3. **浮点提升无精度损失回路**（定理 P2 修正版）：在 $\{\text{F16}, \text{F32}, \text{F64}\}$ 上健全；BF16 → F16 方向存在指数范围损失。
4. **全 BaseType 集不构成格**（定理 P3）：整数 fallback 破坏交换性，但运行时不受影响（仅 i64 一种整数标量）。
5. **跨语言对比**（定理 P4）：Tenth 浮点提升与 JAX 同属严格提升家族；`i64 ⊕ f32 → f32` 与 Julia 同类，存在大整数精度损失风险。
6. **复合代数为双轨制**（定理 P5）：张量-张量用 join 规则，标量-张量用吸收规则，两者不可统一。
7. **穷尽性验证发现 8 个缺失分支**：(Int, Tensor) 在 4 个算子中均未实现，HIR 与 VM 存在语义不一致。
8. **F16/BF16 为前瞻性死代码**：HIR 声明但运行时无表示，属实现债务。

**实施建议**：

- **短期**：补全 (Int, Tensor) 的 8 个缺失分支（§7.3），消除 HIR-VM 不一致。
- **中期**：对称化整数 fallback（§7.5），为多整数类型支持做准备。
- **长期**：实现 F16/BF16 运行时支持（§7.4），重新审视 BF16 < F16 偏序（§7.2）。

本文的理论结论可直接指导 Tenth 类型系统的演进决策，为 dtype 提升规则的健全性提供了形式化保障与改进路线图。

---

## 10. 参考文献

1. **C11 Standard** (ISO/IEC 9899:2011). §6.3.1 Arithmetic operands, §6.3.1.8 Usual arithmetic conversions.
2. **Java Language Specification (JLS)** §5.6.2. Binary Numeric Promotion.
3. **NumPy Documentation**. "Data type promotion in NumPy". https://numpy.org/doc/stable/user/basics.types.html
4. **PyTorch Documentation**. "Type promotion semantics". https://pytorch.org/docs/stable/tensor_attributes.html
5. **JAX Documentation**. "Type promotion semantics". https://jax.readthedocs.io/en/latest/type_promotion.html
6. **Julia Documentation**. "Conversion and Promotion". https://docs.julialang.org/en/v1/manual/conversion-and-promotion/
7. **Rust Reference**. "Type cast expressions". https://doc.rust-lang.org/reference/expressions/operator-expr.html#type-cast-expressions
8. **Davey, B. A. & Priestley, H. A.** (2002). *Introduction to Lattices and Order* (2nd ed.). Cambridge University Press.
9. **IEEE 754-2019**. IEEE Standard for Floating-Point Arithmetic.
10. **Wang, S. & Kanwar, P.** (2019). "BFloat16: The secret to high performance on Cloud TPUs". Google Cloud Blog.
11. Tenth 项目. `docs/语言参考手册.md` §4.3 隐式转换规则.
12. Tenth 项目. `MEMO.md` v0.3.3 变更记录.
13. Tenth 项目. `能力梳理/能力全梳理.md` dtype 相关能力状态.
14. Tenth 项目. `docs/论文/T1-Shape代数系统的形式化建模.md`（broadcast 半群性质）.

---

## 附录 A：定理索引

| 定理 | 陈述 | 证明位置 |
|------|------|---------|
| P1 | $(D_{\text{float}}, \leq_{\text{fp}}, \text{promote})$ 构成有限分配格 | §4.1 |
| P1.1 | 运行时标量域构成 3 元链分配格 | §4.1 推论 |
| P1.2 | 张量 dtype 域构成 2 元链分配格 | §4.1 推论 |
| P2 | 浮点提升无精度损失回路（修正版：不含 BF16） | §4.2 |
| P3 | 全 BaseType 集不构成格（整数 fallback 非交换） | §4.3 |
| P3.1 | 整数 fallback 不破坏结合律 | §4.3 推论 |
| P4 | Tenth 浮点提升与 JAX 同属严格提升家族 | §4.4 |
| P5 | broadcast+promotion 复合代数为双轨制 | §4.5 |
| P5.1 | 标量-张量吸收与张量-张量 join 不可统一 | §4.5 推论 |
| P6 | 张量-张量复合代数为 dtype 格 × shape 半群的直积 | §6.2 |
| P6.1 | 混合 dtype 张量运算与 NumPy/PyTorch/JAX 一致 | §6.2 推论 |

## 附录 B：源码位置索引

| 概念 | 源码位置 |
|------|---------|
| `BaseType` 枚举（16 种类型） | [hir/types.rs:4-10](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/hir/types.rs) |
| `promote_float_dtype` 函数 | [hir/lower/types.rs:480-489](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/hir/lower/types.rs) |
| `infer_binary_type`（HIR 二元运算类型推断） | [hir/lower/types.rs:135-166](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/hir/lower/types.rs) |
| 标量-张量吸收规则 | [hir/lower/types.rs:155-157](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/hir/lower/types.rs) |
| `add_priv`（VM 加法，15 分支） | [runtime/vm.rs:817-872](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/vm.rs) |
| `sub_priv`（VM 减法，14 分支） | [runtime/vm.rs:874-930](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/vm.rs) |
| `mul_priv`（VM 乘法，14 分支） | [runtime/vm.rs:932-988](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/vm.rs) |
| `div_priv`（VM 除法，14 分支） | [runtime/vm.rs:990-1053](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/vm.rs) |
| `TensorData` 枚举（仅 F32/F64） | [runtime/tensor.rs:7-10](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/tensor.rs) |
| `add_scalar`（标量加法，按 dtype 分支） | [runtime/tensor.rs:512-517](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/tensor.rs) |
| `add_tensor`（张量加法，含混合 dtype 提升） | [runtime/tensor.rs:571-597](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/tensor.rs) |
| `broadcast_shape`（NumPy 风格广播） | [runtime/tensor.rs:552-566](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/tensor.rs) |
| 解释器 `eval_binary`（与 VM 对照） | [runtime/interpreter/binary.rs:17-78](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/interpreter/binary.rs) |

## 附录 C：实施建议

| 优先级 | 建议 | 关联定理/局限 | 预期工作量 |
|--------|------|-------------|-----------|
| P0 | 补全 (Int, Tensor) 8 个缺失分支 | §5.2, B1 | 4 个函数各加 2 分支，~30 行 |
| P1 | 对称化整数 fallback（按位宽提升） | 定理 P3, L2 | 修改 `promote_float_dtype`，~10 行 |
| P1 | F16/BF16 HIR 层加"未实现"警告 | L3 | 修改类型检查，~20 行 |
| P2 | f64 标量 + f32 Tensor 溢出检查 | §5.3 模式 2, B2 | `add_scalar` 添加检查，~10 行 |
| P3 | F16/BF16 运行时支持 | L3, §7.4 | 大型重构，~500 行 |
| P3 | 重新审视 BF16 < F16 偏序 | L1, §7.2 | 需实证支撑 |

---

*本文由 Tenth 项目数理部撰写，遵循数理部方法论：必读实现、严谨证明、局限诚实披露。*
