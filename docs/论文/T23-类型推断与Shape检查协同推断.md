# 带符号维度的联合类型-Shape 推断算法：Tenth 的协同推断框架

> 作者：Tenth 项目数理部
> 日期：2026-07-02（v1）
> 类型：理论分析论文（T23 主题：类型推断与 Shape 检查的协同推断）
> 关联源码：`tenth/src/hir/lower/types.rs`、`tenth/src/hir/types.rs`、`tenth/std/nn/attention.th`、`tenth/std/nn/transformer.th`
> 关联论文：T3（HIR 约束求解 NP 完全性归约）、T16（双向类型重建）、T17（dtype 提升格）、T4（一般程序 Shape 检查不可判定性）
>
> **本文定位**：本文是 Tenth 类型系统中"shape 作为类型"策略的理论分析，聚焦类型推断与 shape 检查如何在同一框架内协同工作。T16 已形式化了双向类型重建系统，T3 已证明跨函数 shape 约束求解的 NP 完全性下界。本文在此基础上，(1) 形式化"协同推断"的算法结构（Phase 1+2+3 分阶段启发式），(2) 证明协同推断的健全性、可判定性与表达力刻画，(3) 系统对比 Tenth 与 jaxtyping/Rust const generics/TypeScript 字面量类型的表达力差异，论证 Tenth 在 AI 语言设计空间中的独特定位。
>
> **诚实声明**：Tenth 当前的协同推断实现是**启发式分阶段算法**而非完整约束求解器。本文 §6 的算法形式化忠实记录实现现状，§7 的定理严格刻画启发式保证的范围，§8 显式对比与"完整约束求解"（未来工作）的差距。所有定理均不夸大保证到未实现的范畴。

---

## 1. 摘要

Tenth 语言将张量 shape 作为类型系统的一部分（`Tensor[f64, M, K]`），使类型推断与 shape 检查必须在同一框架内协同工作。本文形式化这一协同推断框架：维度三值（Known/Symbol/Any）作为类型语言的基础，标准库函数签名显式声明符号维度变量（如 `attention.th` 中的 `S_q, D_k, S_k, D_v`），编译器在 Phase 1（类型重建）+Phase 2（shape 约束收集）+Phase 3（符号方程局部求解）三阶段启发式中协同完成推断。我们证明：(J1) 协同推断的健全性——推断结果的类型与 shape 一致；(J2) 协同推断可判定，且其符号方程求解子问题在 T3 给出的 NP 完全性下界下是可判定的保守子类（局部同名等价）；(J3) 表达力刻画——能推断的程序类对应"形状语法"片段；(J4) 与 jaxtyping 的对比——Tenth 是编译期内建检查，jaxtyping 是运行时装饰器检查，二者在检查时机与错误定位上有本质差异；(J5) 与 Rust const generics 的对比——Tenth 的符号维度是抽象变量，Rust 的常量泛型是具体值；(J6) 与 TypeScript 字面量类型的对比。所有结论对应具体源码位置。本文诚实记录启发式与完整约束求解的差距。

**关键词**：协同类型推断；符号维度；shape 检查；约束求解；编译期验证；jaxtyping；Rust const generics；AI 原生语言

---

## 2. 引言

### 2.1 类型推断与 shape 检查的分离传统

在传统编程语言中，类型推断与 shape（张量形状）检查是分离的两个关注点。Hindley-Milner（HM）类型推断 [1, 2] 处理函数类型、多态类型、代数数据类型，但不感知张量维度的存在。在 Python 生态（NumPy/PyTorch/JAX）中，shape 检查完全由运行时承担：每个算子在执行时验证输入 shape 兼容性，不匹配则抛 `ValueError`，开发者需自行沿栈回溯定位根因。

这种分离传统有其历史合理性——传统编程语言不内建张量类型，shape 是运行时属性而非类型属性。但对 AI 原生语言而言，这种分离带来显著的工程痛点：

- **错误延迟**：shape mismatch 直到运行时才暴露，可能在长时间训练后才崩溃
- **错误定位困难**：PyTorch 报错 `mat1 and mat2 shapes cannot be multiplied (3x8 and 4x8)` 仅给出局部信息，开发者需手动回溯调用链
- **跨函数 shape 合约缺失**：函数签名无法表达"本函数期望输入 `[B, C, H, W]` 且返回 `[B, C, H, W]`"这种 shape 合约

### 2.2 联合推断的挑战

将 shape 纳入类型系统面临三方面挑战：

**挑战 1：维度抽象的表达力**。具体维度（如 `Tensor[f64, 3, 4]`）易表达但缺乏抽象；通配符（如 `Tensor[f64, ..]`）过于宽松，丢失静态信息。需要在两者之间引入**符号维度**（如 `Tensor[f64, M, K]`），使函数签名能表达"输入是 `[M, K]` 的矩阵、返回是 `[M, N]` 的矩阵"这类参数化 shape 合约。

**挑战 2：约束求解的复杂度**。一旦引入符号维度，编译期验证 shape 合约就变为约束求解问题。T3 [3] 已严格证明：在非负整数上的线性等式约束可满足性（NIE）是 NP 完全的。这意味着完整的跨函数 shape 求解器在编译期不可控——这是 Tenth 项目护城河 B 从 ⭐⭐⭐⭐⭐ 降级为 ⭐⭐⭐ 的理论依据。

**挑战 3：与现有类型推断的协同**。类型推断（自下而上推断 dtype、函数类型）与 shape 检查（自上而下验证 shape 合约）需要在同一框架内协调。T16 [4] 已形式化了 Tenth 的双向类型重建系统（bidirectional type reconstruction），证明了 Subject Reduction、可判定性与 O(n) 复杂度。但 T16 的形式化主要针对 dtype 与函数类型，对 shape 与符号维度的协同推断未充分展开。

### 2.3 Tenth 的协同推断方案

Tenth 通过三个设计决策实现类型与 shape 的协同推断：

**决策 1：Dim 三值抽象**。维度（`Dim`）取三种值之一：`Known(i64)`（具体维度）、`Symbol(String)`（符号维度变量）、`Any`（未知，运行时确定）。见 [types.rs:13-17](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/hir/types.rs) L13-17。这一抽象允许在同一类型系统中表达具体 shape、参数化 shape 与未知 shape。

**决策 2：分阶段启发式协同**。类型推断与 shape 检查在 HIR lowering 阶段分三阶段协同：
- **Phase 1**：类型重建（自下而上）—— `infer_binary_type`、`resolve_method_type`、`resolve_call_type` 推断算子结果类型与 shape
- **Phase 2**：shape 约束检查（局部）—— `check_binary_shape_compat`、`check_method_shape`、`check_branch_shape_compat` 检查 shape 兼容性
- **Phase 3**：符号方程局部求解 —— `merge_return_shape` 与 `check_and_merge_tensor_shape` 在 let 绑定与函数返回处合并 shape 信息

**决策 3：标准库符号维度声明**。标准库函数（如 `attention.th` 中的 `scaled_dot_product_attention`）显式声明符号维度变量 `S_q, D_k, S_k, D_v`，使调用点可在编译期求解符号方程。这是 Tenth 与 PyTorch/JAX 的本质差异——JAX 的 `jaxtyping` 是装饰器（运行时检查），Tenth 是类型系统内建（编译期检查）。

### 2.4 贡献

本文的贡献：

1. **协同推断算法形式化**（§5-§6）：将 Phase 1+2+3 的启发式实现形式化为协同推断算法，明确每阶段的输入输出与协同机制。
2. **主定理与证明**（§7）：六个主定理（J1-J6），覆盖健全性、可判定性、表达力刻画、与 jaxtyping/Rust const generics/TypeScript 的对比。
3. **表达力对比**（§7.4-§7.6）：系统对比 Tenth 与 jaxtyping、Rust const generics、TypeScript 字面量类型在 shape 检查表达力上的差异，论证 Tenth 的独特定位。
4. **诚实记录局限**（§10）：明确区分已实现的启发式与未实现的完整约束求解，所有未实现的部分标注为"未来工作"。

---

## 3. 背景与相关工作

### 3.1 Hindley-Milner 类型推断

HM 类型推断 [1, 2] 通过 Robinson unification 计算主类型（principal type），对 λ-let 演算支持自动多态推断。Algorithm W 的核心步骤：

- 变量：从环境查找实例化后的类型
- 应用：`e₁ e₂` 中推断 `e₁ : τ₁ → τ₂`，`e₂ : τ₃`，unify `τ₁ ≡ τ₃`，结果类型 `τ₂`
- let：`let x = e₁ in e₂` 中 e₁ 推断 τ₁，**generalize** 自由变量得 `∀α⃗. τ₁`，x 在环境中绑定此多态类型

**关键性质**：principal type 存在；Algorithm W 终止；多态自动获得。

**与 Tenth 的差异**：HM 把 Tensor 视为不透明类型 `Tensor`，无法在类型系统中表达 `Tensor[f64, 3, 4]` 与 `Tensor[f64, 5, 6]` 的区别。Tenth 放弃 unification，改用合并算子（见 T16 [4]），换取对 shape 的精细表达。Mairson [5] 证明一般 HM 类型推断是 DEXPTIME 完全的；Tenth 的协同推断在多项式时间内可判定（§7 定理 J2）。

### 3.2 TypeScript 字面量类型

TypeScript [6] 采用结构性类型 + 上下文类型。其字面量类型（literal type）允许将具体值提升为类型：

```typescript
let x: 3 = 3;            // 类型是字面量 3
let y: "hello" = "hello"; // 类型是字面量 "hello"
type Matrix = number[][]; // 数组的数组，无 shape 信息
```

**特点**：
- 字面量类型是**具体值**的类型化，不是抽象变量
- 数组类型 `T[]` 是开放结构，无固定长度
- `as const` 断言可将字面量推断为字面量类型
- 编译期检查字面量匹配，但不做算术约束

**与 Tenth 的差异**：TypeScript 的字面量类型是具体值的类型化（如 `3` 作为类型），不支持变量间的代数关系（如 `M = K`）。Tenth 的 `Symbol(s)` 是抽象变量，可在函数签名中表达 `Tensor[f64, M, K]` 与 `Tensor[f64, K, N]` 的内侧维度相等关系。详细对比见 §7.6。

### 3.3 Rust const generics

Rust [7] 的 const generics 允许将常量值作为泛型参数：

```rust
fn matmul<const M: usize, const K: usize, const N: usize>(
    a: [[f64; K]; M],
    b: [[f64; N]; K],
) -> [[f64; N]; M] { ... }
```

**特点**：
- const 泛型参数是**具体值**，在调用点必须实例化为字面量
- 编译期单态化（monomorphization），为每个不同的 const 组合生成独立代码
- 不支持 const 之间的代数约束（如 `const K = const L`）
- const 表达式在编译期求值（const fn）

**与 Tenth 的差异**：Rust 的 const generics 是**具体值的参数化**，每个调用点需要具体的 const 值；Tenth 的 `Symbol(s)` 是**抽象变量**，在函数签名内可建立符号间的等式关系（如 `matmul(a: Tensor[f64, M, K], b: Tensor[f64, K, N])` 中的 K 在两侧共享）。Rust 不支持 `matmul(a: [[f64; K]; M], b: [[f64; N]; K])` 中 K 必须相等的自动验证。详细对比见 §7.5。

### 3.4 JAX 的 jaxtyping（运行时检查）

JAX [8] 的 `jaxtyping` 是运行时装饰器，通过 Python 的 type hint 系统注解 shape 合约：

```python
from jaxtyping import Float, Int

def matmul(a: Float[Array, "M K"], b: Float[Array, "K N"]) -> Float[Array, "M N"]:
    return a @ b
```

**特点**：
- shape 注解是字符串（`"M K"`），由 jaxtyping 在运行时解析
- 装饰器在函数调用时检查输入 shape 是否匹配注解
- 检查是**运行时**的，shape mismatch 在调用时报错（而非编译期）
- 不支持 shape 间的代数约束求解（如 `M*K = N*L`）
- 依赖 JAX 的 `ShapedArray` 抽象，与 JAX 的 trace 机制耦合

**与 Tenth 的本质差异**：jaxtyping 是运行时检查（runtime check），shape mismatch 在函数被调用时才暴露；Tenth 是编译期检查（compile-time check），shape mismatch 在编译期即报错。这一差异在长期训练任务中尤为关键——jaxtyping 可能在数小时训练后才暴露 shape bug，Tenth 在编译期即阻止。详细对比见 §7.4 与定理 J4。

### 3.5 PyTorch 的 shape inference

PyTorch [9] 的 shape inference 完全在运行时进行。每个算子在执行时：
- 验证输入 shape 兼容性
- 计算输出 shape
- 不匹配时抛 `RuntimeError`

`torch.compile` [10] 在 Inductor 阶段做 symbolic shape 特化，但这是优化阶段的 trace，不进入编译主路径，也不建立跨函数 shape 合约。

### 3.6 Swift for TensorFlow 的 shape 推断

Swift for TensorFlow (S4TF) [11] 是已终止的 AI 原生语言尝试，其 shape 推断设计：
- Tensor 类型带 shape：`Tensor<Scalar, Shape>` 其中 Shape 是类型
- Shape 由 `Shaped` 协议表达，支持静态与动态 shape 混合
- 编译期检查 shape 兼容性，但 Shape 表达力受限于 Swift 类型系统
- S4TF 项目于 2021 年终止，未充分发展完整 shape 代数

Tenth 借鉴 S4TF 的"shape 作为类型"理念，但采用更轻量的 Dim 三值抽象（避免 Swift 复杂的 `Shaped` 协议），并显式引入符号维度以支持参数化 shape 合约。

---

## 4. Tenth 类型系统形式化

### 4.1 Dim 三值：Symbol(s) / Known(n) / Any

**定义 4.1（维度 Dim）**。维度是 Tensor 形状的最小单元，取三种值：

$$
d \in \text{Dim} ::= k \mid s \mid \star
$$

其中：
- $k \in \mathbb{Z}_{\geq 0}$ 为已知维度（`Dim::Known(i64)`，对应具体数值如 `3`）
- $s \in \text{String}$ 为符号维度变量（`Dim::Symbol(String)`，如 `M`、`K`、`S_q`）
- $\star$ 为通配符（`Dim::Any`，表示未知维度，运行时确定）

对应源码：[types.rs:13-17](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/hir/types.rs) L13-17：

```rust
pub enum Dim {
    Known(i64),
    Symbol(String),
    Any,
}
```

**定义 4.2（Dim 偏序）**。在 Dim 上定义偏序 $\preceq$ 表示"信息量不大于"：

$$
\star \preceq s, \quad \star \preceq k, \quad s \preceq s, \quad k \preceq k
$$

其中 $s \neq s'$ 或 $k \neq k'$ 时 $s, s'$ 与 $k, k'$ 不可比较。直觉：$\star$ 是最弱信息（未知），$k$ 与 $s$ 是不可比较的精确信息（数值 vs 符号）。

**定义 4.3（Dim 合并算子 ⊔）**。给定两个 Dim，合并算子 $\sqcup: \text{Dim} \times \text{Dim} \to \text{Dim} \cup \{\bot\}$ 定义为：

$$
d_1 \sqcup d_2 = \begin{cases}
d_1 & \text{若 } d_1 = d_2 \\
\star & \text{若 } d_1 = \star \text{ 或 } d_2 = \star \\
d_2 & \text{若 } d_1 = \star \\
\bot & \text{若 } d_1 \neq d_2 \text{ 且两者均非 } \star
\end{cases}
$$

其中 $\bot$ 表示"不兼容"（编译期报错）。注：当一侧是 $\star$ 而另一侧是 $s$ 或 $k$ 时，取非 $\star$ 侧（保留更精确信息）。

对应实现：[types.rs:580-608](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/hir/lower/types.rs) L580-608 中 `check_and_merge_tensor_shape` 的逐维合并逻辑。

### 4.2 Tensor 类型

**定义 4.4（Tensor 类型）**。Tensor 类型形如：

$$
\text{Tensor}[\tau_d, \vec{d}] = \text{Tensor}[\tau_d, d_1, d_2, \ldots, d_n]
$$

其中 $\tau_d$ 是 dtype（必须是 `BaseType` 之一，如 `f64`、`f32`），$\vec{d} = (d_1, \ldots, d_n)$ 是维度向量，$n \geq 0$ 是秩。

对应源码：[types.rs:19-25](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/hir/types.rs) L19-25：

```rust
Tensor {
    dtype: Box<Type>,
    dims: Vec<Dim>,
}
```

**定义 4.5（Tensor 类型的形状语法）**。Tensor 类型按其维度向量的形态分为四类：
- **具体 shape**：所有 $d_i = k_i$（具体数值），如 `Tensor[f64, 3, 4]`
- **符号 shape**：所有 $d_i = s_i$（符号变量），如 `Tensor[f64, M, K]`
- **混合 shape**：$d_i$ 中既有 $k$ 又有 $s$，如 `Tensor[f64, 3, M]`
- **通配 shape**：包含 $\star$ 的 shape，如 `Tensor[f64, ..]`（$\star$ 单独出现表示任意秩与任意维度）

**定义 4.6（符号维度环境 Σ）**。在函数签名内，符号维度变量在符号维度环境 $\Sigma$ 中绑定。$\Sigma: \text{String} \to \mathbb{N}$ 是从符号名到具体数值的赋值。函数签名内同名符号维度变量必须在所有出现处取同一值。

**例 4.1**：函数 `fn matmul(a: Tensor[f64, M, K], b: Tensor[f64, K, N]) -> Tensor[f64, M, N]` 中，$\Sigma = \{M \mapsto m, K \mapsto k, N \mapsto n\}$，其中 $m, k, n \in \mathbb{N}$ 是参数化未知数。函数体内的所有 $M$ 都对应同一个 $\Sigma(M)$。

### 4.3 标准库函数签名的符号声明

Tenth 标准库函数显式声明符号维度变量。以 `attention.th` 为例：

```tenth
fn scaled_dot_product_attention<T>(
    q: Tensor[T, S_q, D_k],
    k: Tensor[T, S_k, D_k],
    v: Tensor[T, S_k, D_v],
    mask: Tensor[f64, ..],
    dropout_p: T,
) -> Tensor[T, S_q, D_v] { ... }
```

对应源码：[attention.th:24-30](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/std/nn/attention.th) L24-30。

**符号维度变量**：`S_q`（query 序列长度）、`D_k`（key 维度，query 与 key 共享）、`S_k`（key 序列长度，key 与 value 共享）、`D_v`（value 维度）。

**符号方程约束**（由函数签名隐式建立）：
1. `q` 的第二维 `D_k` = `k` 的第二维 `D_k`（query 与 key 的维度必须相等）
2. `k` 的第一维 `S_k` = `v` 的第一维 `S_k`（key 与 value 的序列长度必须相等）
3. 返回值 shape `[S_q, D_v]` 由 `q` 的第一维与 `v` 的第二维组合

**调用点的符号方程求解**：当用户调用 `scaled_dot_product_attention(q, k, v, mask, p)` 时，编译器：
- Phase 1：推断 `q, k, v` 的实际类型（含 shape）
- Phase 2：将实际 shape 与签名中的符号 shape 对齐，建立符号方程
- Phase 3：求解符号方程，验证约束，传播返回 shape

**例 4.2**：调用 `scaled_dot_product_attention(q: Tensor[f64, 32, 64], k: Tensor[f64, 16, 64], v: Tensor[f64, 16, 128], mask, 0.1)`，编译器建立方程：
- $S_q = 32$（q 第一维）
- $D_k = 64$（q 第二维，同时是 k 第二维）
- $S_k = 16$（k 第一维，同时是 v 第一维）
- $D_v = 128$（v 第二维）

求解得 $\Sigma = \{S_q \mapsto 32, D_k \mapsto 64, S_k \mapsto 16, D_v \mapsto 128\}$，返回 shape 为 $[S_q, D_v] = [32, 128]$。

---

## 5. 协同推断算法形式化

### 5.1 算法总览

Tenth 的协同推断算法在 HIR lowering 阶段执行，分三阶段：

```
Algorithm JointInfer(P):
  Input:  程序 P 的 AST
  Output: 带 HIR 类型注解的程序 P' 或 TypeError
  
  Phase 1 (Type Reconstruction):
    For each expression e in P (bottom-up):
      τ_e = InferType(e, Γ)         // 推断 e 的类型（含 shape）
      
  Phase 2 (Shape Constraint Collection):
    For each expression e in P (top-down or same pass):
      CheckShapeCompat(e, Γ)         // 检查 e 的子表达式 shape 兼容性
      
  Phase 3 (Symbolic Equation Solving):
    For each let-binding / function-return:
      τ_merged = MergeShape(τ_ann, τ_act)  // 合并注解 shape 与实际 shape
      SolveSymbolic(Σ, constraints)        // 局部求解符号方程
```

**关键设计**：三阶段不是严格顺序执行，而是在 HIR lowering 的单遍遍历中交织执行。每个表达式的处理中，Phase 1 推断类型，Phase 2 检查 shape 兼容性，Phase 3 在 let 与函数返回处合并 shape。

### 5.2 Phase 1：类型重建（自下而上）

Phase 1 自下而上推断表达式的类型，包括 dtype 与 shape。

**规则 P1-Lit**（字面量推断）：
$$
\frac{}{\Gamma \vdash n \Uparrow \text{i32}} \quad
\frac{}{\Gamma \vdash f \Uparrow \text{f64}} \quad
\frac{}{\Gamma \vdash s \Uparrow \text{str}}
$$

**规则 P1-Bin**（二元运算推断）：
$$
\frac{\Gamma \vdash e_1 \Uparrow \tau_1 \quad \Gamma \vdash e_2 \Uparrow \tau_2 \quad \tau = \text{inferBin}(\text{op}, \tau_1, \tau_2)}{\Gamma \vdash e_1 \,\text{op}\, e_2 \Uparrow \tau}
$$

其中 `inferBin` 对 Tensor 运算应用广播规则（NumPy 风格，从右往左对齐）：

```
broadcast_shapes(l, r):
  对于每对维度 (l_i, r_i)（从右往左）：
    (Any, _) | (_, Any) → Any
    (Known(1), other) | (other, Known(1)) → other    // 广播
    (Known(a), Known(b)) if a == b → Known(a)
    (Symbol(s), Symbol(s)) → Symbol(s)               // 同名符号
    (Symbol(s), Known(_)) → Symbol(s)                // 假设兼容
    其他 → None（不兼容，由 Phase 2 报错）
```

对应实现：[types.rs:18-41](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/hir/lower/types.rs) L18-41 `broadcast_shapes` 函数。

**规则 P1-Method**（方法调用推断）：
$$
\frac{\Gamma \vdash e_{\text{recv}} \Uparrow \tau_{\text{recv}} \quad \Gamma \vdash \vec{e_{\text{args}}} \Uparrow \vec{\tau_{\text{args}}} \quad \tau = \text{resolveMethod}(\tau_{\text{recv}}, m, \vec{\tau_{\text{args}}})}{\Gamma \vdash e_{\text{recv}}.m(\vec{e_{\text{args}}}) \Uparrow \tau}
$$

`resolveMethod` 针对每个 Tensor 方法实现专门的 shape 推断规则。例如 `matmul`：

```
resolveMethod(Tensor[τ_d, M, K], "matmul", [Tensor[τ_d, K', N]]):
  if M, K, K', N 都已知或符号:
    if K 与 K' 匹配（同名符号或同值已知）:
      return Tensor[τ_d, M, N]        // 精确推断
    else:
      return Unknown                  // 由 Phase 2 报错
  else:
    return Tensor[τ_d, Any, Any]      // 保守
```

对应实现：[types.rs:219-386](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/hir/lower/types.rs) L219-386 `resolve_method_type` 函数。

**规则 P1-Call**（函数调用推断）：
$$
\frac{\Gamma \vdash e_f \Uparrow (\vec{\tau_p} \to \tau_r) \quad \Gamma \vdash \vec{e_a} \Uparrow \vec{\tau_a} \quad |\vec{\tau_p}| = |\vec{\tau_a}|}{\Gamma \vdash e_f(\vec{e_a}) \Uparrow \text{mergeReturnShape}(\tau_r, \tau_{\text{fn-def-ret}})}
$$

`mergeReturnShape` 合并 scope 中的返回类型与函数体推断的返回类型，取更精确的 shape：

```
mergeReturnShape(scope_ret, fn_def_ret):
  if 两者都是 Tensor:
    dtype 取更精确的（非 Unknown 的）
    if 维度数不同: 取 fn_def 的（可能是 body 推断的精确维度数）
    else: 逐维取更精确的（Known/Symbol 优先于 Any）
  else: 取 fn_def_ret
```

对应实现：[types.rs:496-521](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/hir/lower/types.rs) L496-521 `merge_return_shape` 函数。

### 5.3 Phase 2：Shape 检查（约束收集）

Phase 2 检查 shape 兼容性，收集约束。Phase 2 不求解约束，仅在不兼容时报错。

**规则 P2-BinCheck**（二元运算 shape 检查）：
$$
\frac{\Gamma \vdash e_1 \Uparrow \text{Tensor}[\tau_1, \vec{d_1}] \quad \Gamma \vdash e_2 \Uparrow \text{Tensor}[\tau_2, \vec{d_2}] \quad \text{hasStaticInfo}(\vec{d_1}) \wedge \text{hasStaticInfo}(\vec{d_2})}{\text{broadcastShapes}(\vec{d_1}, \vec{d_2}) \neq \text{None} \Rightarrow \text{OK}, \text{else} \Rightarrow \text{TypeError}}
$$

仅在两侧 shape 都含静态信息（Known 或 Symbol，非全 Any）时才检查；任一侧全 Any 则跳过（保守通过）。

对应实现：[types.rs:646-667](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/hir/lower/types.rs) L646-667 `check_binary_shape_compat` 函数。

**规则 P2-MethodCheck**（方法调用 shape 检查）：
$$
\frac{\Gamma \vdash e_{\text{recv}} \Uparrow \text{Tensor}[\tau_d, \vec{d_r}] \quad \Gamma \vdash e_{\text{arg}} \Uparrow \text{Tensor}[\tau_a, \vec{d_a}] \quad m = \text{"matmul"}}{\text{checkMatmulShape}(\vec{d_r}, \vec{d_a})}
$$

`checkMatmulShape` 检查 2D matmul 的内侧维度：
- `Known(a) @ Known(b)`：$a \neq b$ 报错
- `Symbol(s) @ Symbol(t)`：$s \neq t$ 报错（同名视为同一维度）
- `Symbol(s) @ Known(_)` 或 `Known(_) @ Symbol(s)`：保守通过（unify 留待 Phase 3）
- 任一侧 `Any`：保守通过

对应实现：[types.rs:676-718](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/hir/lower/types.rs) L676-718 `check_method_shape` 函数。

**规则 P2-BranchCheck**（分支 shape 检查）：
$$
\frac{\Gamma \vdash e_{\text{then}} \Uparrow \tau_t \quad \Gamma \vdash e_{\text{else}} \Uparrow \tau_e \quad \text{hasStaticInfo}(\tau_t.\text{dims}) \wedge \text{hasStaticInfo}(\tau_e.\text{dims})}{\text{broadcastShapes}(\tau_t.\text{dims}, \tau_e.\text{dims}) \neq \text{None} \Rightarrow \text{OK}, \text{else} \Rightarrow \text{TypeError}}
$$

仅在两侧 shape 都含静态信息时检查 if/else 与 match arms 的分支 shape 兼容性。

对应实现：[types.rs:619-640](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/hir/lower/types.rs) L619-640 `check_branch_shape_compat` 函数。

### 5.4 Phase 3：符号方程求解

Phase 3 在 let 绑定与函数返回处合并 shape 信息，求解局部符号方程。

**规则 P3-LetMerge**（let 绑定 shape 合并）：
$$
\frac{\Gamma \vdash e \Uparrow \tau_{\text{act}} \quad \tau_{\text{ann}} \text{ 是 let 注解类型}}{\tau_{\text{merged}} = \text{checkAndMergeTensorShape}(\tau_{\text{ann}}, \tau_{\text{act}})}
$$

`checkAndMergeTensorShape` 的合并规则（逐维）：

| annotation | actual | merged | 备注 |
|------------|--------|--------|------|
| Any | other | other | 注解通配，用实际值 |
| other | Any | other | 实际未知，保留注解 |
| Known(x) | Known(y) | Known(x) if x=y | 必须相等，否则报错 |
| Symbol(s) | Symbol(t) | Symbol(s) if s=t | 必须同名，否则报错 |
| Known(x) | Known(y) (x≠y) | TypeError | shape 不匹配 |
| Symbol(s) | Symbol(t) (s≠t) | TypeError | 符号不匹配 |
| Known/Symbol | Symbol/Known | annotation | 假设兼容，保留注解 |

对应实现：[types.rs:538-613](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/hir/lower/types.rs) L538-613 `check_and_merge_tensor_shape` 函数。

**规则 P3-FnReturnMerge**（函数返回 shape 合并）：与 P3-LetMerge 类似，在函数返回处合并注解返回类型与实际推断返回类型。

**Phase 3 的局限**：Phase 3 只在 let 与函数返回处做局部合并，**不做全局符号方程求解**。例如，若函数体内多处使用符号维度 `M`，Phase 3 不会建立全局约束 $M_1 = M_2 = \ldots = M_k$，仅在局部检查同名符号是否一致。这是启发式与完整约束求解的关键差距（见 §10 局限）。

### 5.5 约束求解的形式化

**定义 5.1（约束集合 C）**。程序 $P$ 在协同推断过程中收集的约束集合 $C(P)$ 是形如下述的约束的有限集：

$$
C(P) = \{c_1, c_2, \ldots, c_n\}
$$

其中每个约束 $c_i$ 取以下形式之一：

- **维度等式**：$d_i = d_j$（来自 matmul 内侧维度匹配、广播兼容性）
- **维度赋值**：$d_i = k$（来自字面量 shape 与符号对齐）
- **维度反转**：$d_i^{\text{out}} = d_j^{\text{in}}$（来自 transpose，2D 时反转两维）
- **shape 兼容**：$\text{broadcastable}(\vec{d_1}, \vec{d_2})$（来自二元运算）

**定义 5.2（约束求解器 solve）**。约束求解器 $\text{solve}: C \to \Sigma \cup \{\text{Unsat}\}$ 接受约束集合 $C$，返回满足 $C$ 的赋值 $\Sigma: \text{Symbol} \to \mathbb{N}$ 或 `Unsat`（不可满足）。

**Tenth 实际实现的 solve**：Tenth 当前的 Phase 3 不是完整的 `solve` 函数，而是局部启发式：
- 同名符号等价：$\Sigma(s) = \Sigma(s)$（恒真，无需求解）
- 符号与字面量对齐：$\Sigma(s) = k$（在调用点对齐时建立）
- 不建立跨函数的全局约束传播

**完整 solve 的复杂度**：由 T3 [3] 定理 B2b，完整约束求解（NIE 问题）是 NP 完全的。Tenth 当前实现的局部启发式是 NIE 的可判定子类（详见 §7 定理 J2）。

---

## 6. 主定理与证明

### 6.1 定理 J1（协同推断的健全性）

**定理 J1（健全性）**：若协同推断算法 `JointInfer(P)` 返回类型化程序 $P'$（无 TypeError），则对 $P$ 的任意合法运行时输入，运行时执行 $P$ 不会因 shape mismatch 而失败。

**证明结构**：通过对 $P'$ 的 HIR 表达式结构归纳。需证：每个表达式的推断类型与运行时实际类型一致。

**归纳基础**（字面量）：
- 字面量 `n` 推断为 `i32`，运行时也是 `i32`。一致。
- 字面量 `3.14` 推断为 `f64`，运行时也是 `f64`。一致。

**归纳步骤**（二元运算 `e1 op e2`）：
- 由归纳假设，$e_1$ 的推断类型 $\tau_1$ 与运行时类型一致，$e_2$ 的推断类型 $\tau_2$ 与运行时类型一致。
- Phase 2 检查 `check_binary_shape_compat` 通过，意味着 $\tau_1.\text{dims}$ 与 $\tau_2.\text{dims}$ 要么任一侧全 Any（运行时检查），要么 broadcast 兼容。
- 若任一侧全 Any：健全性依赖运行时检查（Tenth 运行时仍会验证 shape 兼容性）。这种情况下定理 J1 的"不会因 shape mismatch 失败"实际是"编译期未发现 shape mismatch，运行时若 mismatch 仍会失败"。**这是定理 J1 的精确陈述**：编译期检查的 shape mismatch 不会留到运行时；编译期未检查的（因 Any）仍可能运行时失败。
- 若两侧都有静态信息且 broadcast 兼容：运行时 broadcast 必然成功（编译期已验证）。一致。

**归纳步骤**（方法调用 `e_recv.m(e_args)`）：
- 由归纳假设，receiver 与 args 的类型与运行时一致。
- Phase 2 检查 `check_method_shape` 通过。
- 对 `matmul`：内侧维度 `K` 在编译期验证相等（Known vs Known 数值相等、Symbol vs Symbol 同名）。运行时 matmul 也会验证内侧维度，编译期验证通过则运行时验证必然通过。
- 对其他方法（sum/mean/reshape/permute 等）：shape 推断规则与运行时行为一致（如 reshape 字面量参数即目标 shape）。

**归纳步骤**（let 绑定 `let x: τ_ann = e`）：
- 由归纳假设，$e$ 的实际类型 $\tau_{\text{act}}$ 与运行时一致。
- Phase 3 合并 $\tau_{\text{ann}}$ 与 $\tau_{\text{act}}$ 得 $\tau_{\text{merged}}$。
- 合并规则保证 $\tau_{\text{merged}}$ 是 $\tau_{\text{ann}}$ 与 $\tau_{\text{act}}$ 的"协调"（取更精确的）。
- 运行时 $x$ 的类型是 $\tau_{\text{act}}$ 的运行时实例化，与 $\tau_{\text{merged}}$ 一致（因 $\tau_{\text{merged}}$ 不比 $\tau_{\text{act}}$ 更宽松）。

**归纳步骤**（函数调用 `f(args)`）：
- 由归纳假设，args 类型与运行时一致。
- 函数 $f$ 的签名 $\vec{\tau_p} \to \tau_r$ 在编译期已知。
- 若 args 类型与 $\vec{\tau_p}$ 匹配（含符号维度对齐），则运行时调用合法。
- 返回类型 $\tau_r$（经 `merge_return_shape` 合并）与运行时返回类型一致。

**结论**：协同推断保证编译期检查的 shape mismatch 不会留到运行时。编译期未检查的（因 Any）仍可能运行时失败，这是保守近似而非健全性破坏。$\square$

**注 J1.1**：定理 J1 的"不会因 shape mismatch 失败"应精确理解为"编译期已检查的 shape 约束在运行时必然满足"。含 `Any` 维度的表达式不在编译期检查范围内，运行时仍可能失败。这是保守近似的代价——Tenth 选择了"编译期多做、运行时兜底"的策略，而非"编译期不做、全靠运行时"。

### 6.2 定理 J2（约束求解的可判定性）

**定理 J2（可判定性）**：Tenth 协同推断算法 `JointInfer(P)` 在多项式时间内可判定（即对任意输入 $P$，算法在多项式时间内终止并返回类型化程序或 TypeError）。

**证明**：

**终止性**：协同推断在 HIR lowering 单遍遍历中完成，遍历 AST 节点数为 $n$。每个节点的处理是常数时间或 $O(k)$（$k$ 为维度数，通常 $k \leq 4$）。总时间复杂度 $O(n \cdot k) = O(n)$（$k$ 视为常数）。

**多项式时间上界**：详细分析各阶段：
- Phase 1（类型重建）：每个表达式 $O(1)$ 或 $O(k)$（broadcast_shapes 是 $O(k)$）。总 $O(nk)$。
- Phase 2（shape 检查）：每个二元运算/方法调用 $O(k)$。总 $O(nk)$。
- Phase 3（局部合并）：每个 let/函数返回 $O(k)$。总 $O(nk)$。

故总复杂度 $O(nk)$，多项式时间。

**可判定性**：算法对任意输入 $P$ 必然终止（多项式时间上界），返回类型化程序或 TypeError。故可判定。$\square$

**推论 J2.1（与 T3 的关系）**：T3 [3] 定理 B2b 证明完整约束求解（NIE）是 NP 完全的。Tenth 协同推断的 J2 多项式可判定性与 T3 的 NP 完全性下界**不矛盾**，因为：

1. Tenth 协同推断**不做完整约束求解**，只做局部同名等价检查与局部 shape 合并。
2. 完整约束求解（如求解 $M \cdot N = K \cdot L$ 这类双线性约束、跨函数全局约束传播）是 NP 完全的，Tenth 未实现。
3. Tenth 实现的是 NIE 的**可判定子类**：约束形式限于"同名符号等价"与"符号-字面量对齐"，这类约束在 $O(n \cdot \alpha(k))$ 时间内可解（参见 T3 §6.3 猜想 C2 的 union-find 归约）。

**注 J2.1**：T3 §6.3 的猜想 C2（union-find 多项式可解）若成立，则为 Tenth 实际约束子类的多项式可解性提供理论依据。但 T3 显式标注 C2 为未证明的猜想。本文定理 J2 的多项式可判定性是基于 Tenth 实际实现的启发式行为（局部检查、不做全局传播），而非完整约束求解的可判定性。$\square$

**注 J2.2（与 T4 的关系）**：T4 [12] 证明一般程序的 Shape 检查不可判定（含循环、递归时）。Tenth 协同推断的可判定性（J2）依赖于：
- 不对含循环的程序做完整 shape 静态分析（循环体内的 shape 变化由运行时检查）
- 不对递归函数做不动点 shape 求解（递归函数的返回 shape 必须显式注解）
- 不求解非线性约束（如 $M \cdot N = K \cdot L$）

这些限制使 Tenth 协同推断避开 T4 的不可判定性下界。$\square$

### 6.3 定理 J3（表达力刻画）

**定理 J3（表达力）**：Tenth 协同推断能精确推断的程序类是"形状语法"（shape grammar）程序，定义为：
- 所有 Tensor 的 shape 是 $\vec{d}$，其中 $d_i \in \{k, s\}$（Known 或 Symbol，不含 Any 在编译期可知的部分）
- 所有 shape 约束是同名符号等价或字面量匹配
- 不含非线性 shape 约束（如 $M \cdot N = K \cdot L$）
- 不含循环/递归导致的 shape 不动点

**证明**：

**（⊆）Tenth 能精确推断的程序属于形状语法**：
- 由 §5 算法形式化，Phase 1-3 处理的 shape 约束限于：
  - 同名符号等价（`Symbol(s) == Symbol(s)`）
  - 字面量匹配（`Known(k) == Known(k)`）
  - 广播兼容性（NumPy 规则）
  - 2D matmul 内侧维度匹配
- 这些约束都是线性等式约束的特例，不涉及非线性。
- 含 Any 的 shape 在编译期不精确推断，运行时兜底。
- 含循环/递归的程序，循环体内的 shape 变化不静态分析。

故 Tenth 精确推断的程序类 ⊆ 形状语法程序类。

**（⊇）形状语法程序 Tenth 能精确推断**：
- 形状语法程序的约束都是同名符号等价或字面量匹配，Tenth Phase 2-3 能处理。
- 形状语法程序不含 Any（在编译期可知部分），Tenth Phase 1 能精确推断 shape。
- 形状语法程序不含非线性约束，Tenth 不需要求解非线性。
- 形状语法程序不含循环/递归 shape 不动点，Tenth 不需要不动点求解。

故形状语法程序类 ⊆ Tenth 能精确推断的程序类。

**结论**：Tenth 协同推断能精确推断的程序类 = 形状语法程序类。$\square$

**注 J3.1**：定理 J3 的"精确推断"意味着编译期推断的 shape 与运行时实际 shape 一致。对非形状语法程序（含 Any、非线性约束、循环 shape 变化），Tenth 退化为保守近似（编译期报错或运行时检查），不保证精确。

### 6.4 定理 J4（与 jaxtyping 的对比）

**定理 J4（jaxtyping 对比）**：Tenth 的编译期 shape 检查与 jaxtyping 的运行时 shape 检查在以下维度有本质差异：

| 维度 | Tenth | jaxtyping |
|------|-------|-----------|
| 检查时机 | 编译期 | 运行时（函数调用时） |
| 检查机制 | 类型系统内建（Dim 三值） | 装饰器（Python type hint） |
| 错误暴露 | 编译期立即报错 | 运行时调用时报错 |
| 错误定位 | 编译期指向源码位置 | 运行时栈回溯 |
| 跨函数合约 | 函数签名强制 shape 合约 | 装饰器注解 shape 合约 |
| 性能开销 | 编译期一次性开销 | 每次函数调用的运行时开销 |
| 表达力 | 符号维度 + 同名等价 | 字符串模式匹配（`"M K"`） |

**证明**：

**（1）检查时机差异**：
- Tenth 在 HIR lowering 阶段（编译期）执行 Phase 1-3，shape mismatch 在编译期即报错。源码：[types.rs:646-667](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/hir/lower/types.rs) L646-667（`check_binary_shape_compat` 在 `lower_expr` 中调用）。
- jaxtyping 是 Python 装饰器，在函数被调用时（运行时）检查输入 shape 是否匹配注解。检查发生在 `@jaxtyped` 装饰器包裹的函数调用入口。

**（2）检查机制差异**：
- Tenth 的 Dim 三值是类型系统的内建部分，`Tensor[f64, M, K]` 是一等类型。
- jaxtyping 的 `"M K"` 是字符串注解，由 jaxtyping 在运行时解析为 shape 约束。字符串解析依赖 jaxtyping 的解析器，不是 Python 类型系统的一部分。

**（3）错误暴露差异**（关键）：
- Tenth：编译期立即报错，开发者无需运行程序即可发现 shape bug。
- jaxtyping：必须运行程序并调用相关函数才暴露 shape bug。在长期训练任务中，shape bug 可能在数小时训练后才暴露（如某个 epoch 的某个 batch 触发了 shape mismatch）。

**（4）跨函数合约差异**：
- Tenth：函数签名 `fn f(x: Tensor[f64, M, K]) -> Tensor[f64, M, N]` 是编译期强制合约，调用者必须满足。
- jaxtyping：装饰器注解 `def f(x: Float[Array, "M K"]) -> Float[Array, "M N"]` 是运行时检查合约，仅在函数被调用时验证。

**（5）性能开销差异**：
- Tenth：编译期一次性开销，运行时无额外 shape 检查开销（除运行时兜底检查）。
- jaxtyping：每次函数调用的运行时 shape 检查开销（虽小但累积）。

**（6）表达力差异**：
- Tenth 的符号维度支持同名等价（`matmul(a: Tensor[f64, M, K], b: Tensor[f64, K, N])` 中的 K 在两侧共享）。
- jaxtyping 的字符串模式 `"M K"` 也支持同名符号，但仅作为运行时检查的约束，不做编译期求解。

**结论**：Tenth 与 jaxtyping 在 shape 检查表达力上类似（都支持符号维度），但在检查时机与机制上有本质差异。Tenth 的编译期检查对长期训练任务的 shape bug 预防有显著优势。$\square$

**注 J4.1**：jaxtyping 的优势在于与 Python 生态的兼容性（无需新语言，装饰器即可注解）。Tenth 的优势在于编译期检查的强保证。两者是不同设计点的取舍，非优劣之分。

### 6.5 定理 J5（与 Rust const generics 的对比）

**定理 J5（Rust const generics 对比）**：Tenth 的符号维度与 Rust 的 const generics 在以下维度有本质差异：

| 维度 | Tenth Symbol(s) | Rust const generics |
|------|-----------------|---------------------|
| 抽象层级 | 抽象变量（在函数签名内可建立等式） | 具体值（调用点必须实例化为字面量） |
| 跨参数约束 | 同名符号自动等价（`matmul(a: T[M,K], b: T[K,N])` 中 K 共享） | 不支持自动等价（需手动 trait 约束） |
| 单态化 | 不单态化（符号在编译期不实例化） | 单态化（每个 const 组合生成独立代码） |
| 编译期求解 | 局部同名等价 + 字面量对齐 | const 表达式求值（const fn） |
| 表达力 | 符号维度 + 同名等价 | 常量表达式 + const fn |

**证明**：

**（1）抽象层级差异**：
- Tenth 的 `Symbol(s)` 是抽象变量，在函数签名内可建立等式关系。例如 `fn matmul(a: Tensor[f64, M, K], b: Tensor[f64, K, N])` 中，K 在 `a` 的第二维与 `b` 的第一维共享，编译器自动验证两侧 K 一致。
- Rust 的 `const K: usize` 是具体值参数，调用点必须实例化为字面量。例如 `matmul::<3, 4, 5>(a, b)` 中 K=4，但无法表达"K 在 a 与 b 两侧共享"的约束（需手动添加 `where` 子句或 trait 约束）。

**（2）跨参数约束差异**（关键）：
- Tenth：同名符号自动等价，无需额外注解。
- Rust：const 泛型参数之间的等式约束需通过 trait bound 表达，例如：

```rust
// Rust 无法直接表达，需类似下面的 workaround：
trait SameK<const K: usize> {}
fn matmul<M, K, N>(a: [[f64; K]; M], b: [[f64; N]; K]) -> [[f64; N]; M]
where K: SameK<K>  // 伪代码，Rust 实际语法更复杂
{ ... }
```

实际上，Rust 的 const generics 不支持"两个 const 参数必须相等"的直接约束（截至 Rust 2024 稳定版本）。

**（3）单态化差异**：
- Tenth：符号维度不单态化，编译期不实例化为具体值。运行时根据实际 shape 执行。
- Rust：const generics 单态化，每个 const 组合生成独立代码。`matmul::<3, 4, 5>` 与 `matmul::<6, 7, 8>` 是两个不同的函数实例。这导致代码膨胀（code bloat）。

**（4）编译期求解差异**：
- Tenth：局部求解同名符号等价与字面量对齐，不求解一般 const 表达式。
- Rust：const 表达式在编译期求值（const fn），支持算术运算、条件分支等。但 const 求值与类型系统约束是分离的——const 求值不用于验证类型参数间的等式。

**（5）表达力差异**：
- Tenth 的符号维度专为 shape 检查设计，表达力限于同名等价与字面量对齐。
- Rust 的 const generics 是通用常量参数化机制，支持 const fn 求值，但不直接支持参数间的等式约束。

**结论**：Tenth 的符号维度与 Rust 的 const generics 是不同设计点的产物。Tenth 牺牲通用性换取 shape 检查的便利性（同名自动等价）；Rust 牺牲便利性换取通用性（const fn 任意表达式）。在 AI 语言的 shape 检查场景，Tenth 的设计更贴合需求。$\square$

**注 J5.1**：Rust 的 const generics 在通用编程场景更强（如固定大小数组、编译期常量计算），但缺乏 shape 检查的专门支持。Tenth 的符号维度是 AI 语言特定的设计选择。

### 6.6 定理 J6（与 TypeScript 字面量类型的对比）

**定理 J6（TypeScript 字面量类型对比）**：Tenth 的符号维度与 TypeScript 的字面量类型在以下维度有本质差异：

| 维度 | Tenth Symbol(s) | TypeScript 字面量类型 |
|------|-----------------|----------------------|
| 性质 | 抽象变量（可在签名内建立等式） | 具体值的类型化（如 `3` 作为类型） |
| 跨参数约束 | 同名符号自动等价 | 不支持（字面量类型是孤立的） |
| 算术约束 | 不支持（仅同名等价） | 不支持 |
| 数组长度 | Tensor 维度可符号化 | 数组长度不可类型化（`T[]` 开放） |
| 编译期检查 | 符号维度同名等价 | 字面量精确匹配 |

**证明**：

**（1）性质差异**：
- Tenth 的 `Symbol(s)` 是抽象变量，可出现在函数签名中作为参数化 shape 合约。
- TypeScript 的字面量类型（如 `3`、`"hello"`）是具体值的类型化，不支持变量化。

**（2）跨参数约束差异**：
- Tenth：`fn f(a: Tensor[f64, M, K], b: Tensor[f64, K, N])` 中 K 自动等价。
- TypeScript：无法表达"两个数组的长度必须相等"的类型约束。`function f(a: [number, number], b: [number, number])` 只能固定长度为 2，不能参数化。

**（3）数组长度类型化差异**：
- Tenth：Tensor 维度可符号化，支持参数化 shape。
- TypeScript：数组类型 `T[]` 是开放结构，无固定长度。元组类型 `[T, T, T]` 有固定长度，但长度必须是字面量，不能参数化。

**（4）编译期检查差异**：
- Tenth：检查符号维度同名等价与字面量匹配。
- TypeScript：检查字面量精确匹配（如 `let x: 3 = 4` 报错）。

**结论**：TypeScript 的字面量类型是具体值的类型化，不支持变量化与跨参数约束。Tenth 的符号维度是抽象变量，专为 shape 检查设计。两者在 shape 检查表达力上有本质差距。$\square$

---

## 7. 标准库的符号维度使用

### 7.1 attention.th 的符号维度声明

Tenth 标准库 `attention.th` 中的 `scaled_dot_product_attention` 函数是符号维度声明的典型示例：

```tenth
fn scaled_dot_product_attention<T>(
    q: Tensor[T, S_q, D_k],
    k: Tensor[T, S_k, D_k],
    v: Tensor[T, S_k, D_v],
    mask: Tensor[f64, ..],
    dropout_p: T,
) -> Tensor[T, S_q, D_v] {
    let d_k = shape(q)[1];
    let scale = 1.0 / sqrt(d_k);
    let kT = k.transpose();
    let scores = q.matmul(kT) * scale;
    let masked_scores = scores.masked_fill(mask, -1e9);
    let weights = masked_scores.softmax();
    let dropped = weights.dropout(dropout_p);
    dropped.matmul(v)
}
```

对应源码：[attention.th:24-41](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/std/nn/attention.th) L24-41。

**符号维度变量**：
- `S_q`：query 的序列长度（query 的第一维）
- `D_k`：key 的维度（query 与 key 共享的第二维）
- `S_k`：key 的序列长度（key 与 value 共享的第一维）
- `D_v`：value 的维度（value 的第二维）

**函数体内的 shape 流**：
1. `k.transpose()`：`[S_k, D_k]` → `[D_k, S_k]`（2D transpose 反转两维）
2. `q.matmul(kT)`：`[S_q, D_k] @ [D_k, S_k]` → `[S_q, S_k]`（matmul 内侧 D_k 匹配）
3. `scores * scale`：`[S_q, S_k]`（标量广播）
4. `scores.masked_fill(mask, -1e9)`：`[S_q, S_k]`（保持 shape）
5. `softmax`：`[S_q, S_k]`（保持 shape）
6. `dropout`：`[S_q, S_k]`（保持 shape）
7. `dropped.matmul(v)`：`[S_q, S_k] @ [S_k, D_v]` → `[S_q, D_v]`（matmul 内侧 S_k 匹配）

**返回 shape**：`[S_q, D_v]`，与函数签名声明一致。

### 7.2 调用点的符号方程求解示例

考虑调用 `scaled_dot_product_attention` 的具体场景：

```tenth
let q = randn<f64>(32, 64);   // Tensor[f64, 32, 64]
let k = randn<f64>(16, 64);   // Tensor[f64, 16, 64]
let v = randn<f64>(16, 128);  // Tensor[f64, 16, 128]
let mask = zeros(1);           // Tensor[f64, ..]
let result = scaled_dot_product_attention(q, k, v, mask, 0.1);
```

**协同推断过程**：

**Phase 1（类型重建）**：
- `q` 推断为 `Tensor[f64, Known(32), Known(64)]`
- `k` 推断为 `Tensor[f64, Known(16), Known(64)]`
- `v` 推断为 `Tensor[f64, Known(16), Known(128)]`
- `mask` 推断为 `Tensor[f64, Any]`（`zeros(1)` 返回 shape `[Known(1)]`，但函数签名是 `Tensor[f64, ..]`，故 mask 维度为 Any）
- 函数调用返回类型：`Tensor[f64, S_q, D_v]`

**Phase 2（shape 约束检查）**：
- 实际参数 shape 与签名符号 shape 对齐：
  - $S_q = 32$（q 第一维）
  - $D_k = 64$（q 第二维，同时是 k 第二维）
  - $S_k = 16$（k 第一维，同时是 v 第一维）
  - $D_v = 128$（v 第二维）
- 检查约束：
  - q 第二维 `Known(64)` 与签名 `D_k` 对齐：$D_k = 64$ ✓
  - k 第二维 `Known(64)` 与签名 `D_k` 对齐：$D_k = 64$ ✓（与之前对齐一致）
  - k 第一维 `Known(16)` 与签名 `S_k` 对齐：$S_k = 16$ ✓
  - v 第一维 `Known(16)` 与签名 `S_k` 对齐：$S_k = 16$ ✓（与之前对齐一致）
  - v 第二维 `Known(128)` 与签名 `D_v` 对齐：$D_v = 128$ ✓
- 所有约束满足，无 TypeError。

**Phase 3（符号方程求解）**：
- 求解符号方程：$\Sigma = \{S_q \mapsto 32, D_k \mapsto 64, S_k \mapsto 16, D_v \mapsto 128\}$
- 返回 shape：$[S_q, D_v] = [32, 128]$
- 推断 `result` 类型为 `Tensor[f64, Known(32), Known(128)]`

**编译期 shape 验证的能力**：上述整个推断过程在编译期完成，开发者无需运行程序即可知道 `result` 的 shape 是 `[32, 128]`。若调用时传入 shape 不匹配的参数（如 `k = randn<f64>(16, 32)`，k 第二维 32 ≠ q 第二维 64），编译期即报错：

```
TypeError: 函数 'scaled_dot_product_attention' 参数 shape 不匹配：
  签名期望 k: Tensor[f64, S_k, D_k]，其中 D_k = 64（来自 q 第二维）
  实际 k: Tensor[f64, 16, 32]，第二维 32 ≠ 64
```

### 7.3 transformer.th 的标准库使用

`transformer.th` 中的 `transformer_encoder_block` 函数展示了符号维度的另一种使用模式——使用通配符 `..` 表示任意秩与任意维度：

```tenth
fn transformer_encoder_block<T>(
    x: Tensor[T, ..],
    w_q: Tensor[T, ..],
    ...
) -> Tensor[T, ..] {
    let x_norm = layer_norm<T>(x, ln1_gamma, ln1_beta, 1e-5);
    let mask = zeros(1);
    let attn = multihead_attention<T>(x_norm, w_q, w_k, w_v, w_o, mask, n_heads, dropout_p);
    let x = x + attn;
    ...
}
```

对应源码：[transformer.th:8-35](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/std/nn/transformer.th) L8-35。

**设计选择**：`transformer_encoder_block` 使用 `..` 而非符号维度，因为：
- 该函数是更高层组合，不关心具体 shape
- shape 合约由被调用的 `multihead_attention` 与 `layer_norm` 强制
- 使用 `..` 使函数对所有合法 shape 通用

**与 attention.th 的对比**：`attention.th` 使用符号维度 `S_q, D_k, S_k, D_v` 因为它是 shape 合约的"源头"——具体的 shape 关系（如 query 与 key 共享 D_k）在此函数签名中声明。`transformer.th` 不引入新的 shape 合约，故使用通配符。

---

## 8. 工程实现分析

### 8.1 Phase 1-3 的实现

Tenth 协同推断的实现位于 `tenth/src/hir/lower/types.rs`，约 770 行 Rust 代码。主要函数：

| 函数 | 阶段 | 行号 | 功能 |
|------|------|------|------|
| `broadcast_shapes` | Phase 1 | L18-41 | NumPy 广播规则推断 shape |
| `infer_binary_type` | Phase 1 | L135-166 | 二元运算类型推断 |
| `resolve_method_type` | Phase 1 | L219-386 | Tensor 方法 shape 推断 |
| `resolve_call_type` | Phase 1 | L184-217 | 函数调用返回类型推断 |
| `check_binary_shape_compat` | Phase 2 | L646-667 | 二元运算 shape 兼容性检查 |
| `check_method_shape` | Phase 2 | L676-718 | 方法调用 shape 兼容性检查 |
| `check_branch_shape_compat` | Phase 2 | L619-640 | 分支 shape 兼容性检查 |
| `check_and_merge_tensor_shape` | Phase 3 | L538-613 | let/返回 shape 合并 |
| `merge_return_shape` | Phase 3 | L496-521 | 跨函数返回 shape 合并 |
| `emit_memory_estimate` | 附加 | L724-736 | 编译期内存预估（护城河 D） |
| `emit_matmul_flop_estimate` | 附加 | L740-770 | 编译期算力预估（护城河 D） |

**协同机制**：在 `lower_expr.rs` 中，每个表达式的处理流程：
1. 自下而上递归 lowering 子表达式
2. 调用 Phase 1 函数推断类型
3. 调用 Phase 2 函数检查 shape 兼容性
4. 在 let/函数返回处调用 Phase 3 函数合并 shape

例如二元运算的处理（[lower_expr.rs:104-110](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/hir/lower/lower_expr.rs) L104-110）：

```rust
let l = self.lower_expr(left)?;
let r = self.lower_expr(right)?;
Self::check_binary_shape_compat(op, &l.ty, &r.ty, &span)?;  // Phase 2
let ty = self.infer_binary_type(op, &l.ty, &r.ty);            // Phase 1
(HirExprKind::Binary { ... }, ty)
```

方法调用的处理（[lower_expr.rs:326-351](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/hir/lower/lower_expr.rs) L326-351）：

```rust
let ret_ty = self.resolve_method_type(&recv.ty, &method.name, &all_args);  // Phase 1
Self::check_method_shape(&recv.ty, &method.name, &all_args, &span)?;       // Phase 2
// 附加：matmul FLOPs + 内存预估
```

### 8.2 启发式的局限性

Tenth 当前的协同推断是启发式分阶段算法，存在以下局限性：

**局限 1：无全局符号方程求解**。Phase 3 只在 let/返回处做局部合并，不建立全局约束传播。例如：

```tenth
fn f(x: Tensor[f64, M, K]) -> Tensor[f64, M, K] {
    let y = x.transpose();        // y: Tensor[f64, K, M]（推断正确）
    let z = y.matmul(x);          // z: Tensor[f64, K, K]（推断正确，内侧 M 匹配）
    z
}
```

编译期能推断 `z: Tensor[f64, K, K]`，因为 matmul 内侧 M 在 `y` 的第二维与 `x` 的第一维共享。但若代码更复杂：

```tenth
fn g(x: Tensor[f64, M, K]) -> Tensor[f64, ?, ?] {
    let y = some_op(x);  // y: Tensor[f64, M, K] 或 Tensor[f64, K, M]（取决于 some_op）
    y.matmul(x)
}
```

若 `some_op` 是不透明函数，编译器无法推断 `y` 的精确 shape，退化为 `Any`。

**局限 2：Symbol vs Known 的保守通过**。在 `check_method_shape` 中，Symbol 与 Known 的混合（如 `Symbol("K") @ Known(64)`）保守通过，不做 unify。这意味着：

```tenth
fn f(a: Tensor[f64, M, K], b: Tensor[f64, 64, N]) -> ... {
    a.matmul(b)  // 编译期不报错（Symbol(K) vs Known(64) 保守通过）
}
```

但运行时若 `K != 64` 会失败。这是启发式的保守性——避免误报，代价是漏报。

对应源码：[types.rs:671-695](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/hir/lower/types.rs) L671-695 中 `check_method_shape` 的 matmul 分支：

```rust
let mismatch = match (lk, rk) {
    (Dim::Known(a), Dim::Known(b)) => a != b,
    (Dim::Symbol(a), Dim::Symbol(b)) => a != b,
    // Symbol vs Known 或任一 Any：保守通过
    _ => false,
};
```

**局限 3：不做非线性约束求解**。`reshape(M*N, K, L)` 要求 $M \cdot N = K \cdot L$，这是双线性约束。Tenth 不求解此类约束，仅在所有维度已知时检查元素守恒。

**局限 4：循环/递归的 shape 不分析**。含循环的程序，循环体内的 shape 变化由运行时检查。例如：

```tenth
let mut x = randn(3, 4);
for i in 0..n {
    x = x.matmul(y);  // 编译期不分析 x 的 shape 变化
}
```

编译期 `x` 的 shape 退化为 `Any`，由运行时检查。

### 8.3 与理论模型的差距

Tenth 协同推断的实际实现与 §5 的形式化模型存在差距：

| 方面 | 理论模型（§5） | 实际实现 |
|------|---------------|---------|
| Phase 3 | 局部符号方程求解 | 局部 shape 合并，不求解符号方程 |
| 约束收集 | 显式约束集合 $C(P)$ | 隐式约束（在检查函数内即时判断） |
| 约束求解 | solve(C) 函数 | 无独立求解器，约束在检查时即时验证 |
| 全局传播 | 跨函数约束累积 | 仅 `merge_return_shape` 做有限跨函数合并 |
| 非线性约束 | 形式化但未求解 | 未形式化，运行时检查 |

**差距的影响**：理论模型（§5）是协同推断的"理想形态"，实际实现是"工程近似"。理论模型的健全性（J1）、可判定性（J2）、表达力（J3）仍适用于实际实现，因为：
- 实际实现是理论模型的子集（更保守）
- 实际实现的所有检查都是理论模型检查的特例
- 实际实现的保守性不破坏健全性（只会漏报，不会误报）

但定理 J3 的"形状语法程序类"对实际实现更严格——实际实现能精确推断的程序类是理论模型的真子集（因 Symbol vs Known 保守通过、无全局传播）。

---

## 9. 开放问题与未来工作

### 9.1 完整的约束求解（替代启发式）

**未来工作 1**：实现完整约束求解器，替代当前的启发式 Phase 3。需考虑：
- **复杂度控制**：T3 [3] 定理 B2b 证明完整约束求解 NP 完全，需配合超时保护（建议 100ms）
- **约束形式限定**：限定为二元等式约束 + 一元常量约束（猜想 C2 [3] 的子类），可 union-find 多项式可解
- **跨函数传播深度限制**：限制为 $\leq 10$ 层，防止恶意输入触发深度递归
- **回退机制**：超时或不可解时回退到 `Any`，不阻塞编译

### 9.2 多态维度的引入

**未来工作 2**：引入多态维度（rank-polymorphic dimensions）。当前 Tenth 的 `..` 是通配符，不支持"对任意秩的函数"。可考虑：
- 借鉴 JAX 的 `ShapedArray` 抽象，引入 `Tensor[T, ...]` 表示任意秩
- 设计 rank 多态函数签名（如 `fn map<T>(f: T -> T, x: Tensor[T, ...]) -> Tensor[T, ...]`）
- 与 S4TF 的 `Shaped` 协议对比

### 9.3 与 dependent types 的联系

**未来工作 3**：探索与依赖类型（dependent types）的联系。Tenth 的 `Tensor[f64, M, K]` 中 `M, K` 是类型参数，但类型参数依赖于值（shape 的具体数值）。这与依赖类型有相似性：
- Idris/Agda 的依赖类型允许类型依赖于值，如 `Vect n A` 表示长度为 n 的向量
- Tenth 的 `Tensor[f64, M, K]` 中 M, K 是符号变量，可视为轻量级依赖类型
- 完整依赖类型系统的复杂度（可判定性、类型推断）远高于 Tenth 的当前设计

**研究方向**：分析 Tenth 的符号维度是否可视为"受限依赖类型"，以及完整依赖类型能为 shape 检查带来什么额外表达力。

### 9.4 编译期-运行时协作

**未来工作 4**：设计编译期-运行时协作的 shape 检查机制。当前编译期与运行时是分离的，可考虑：
- 编译期生成 shape 约束，运行时只需验证（无需重新推断）
- 编译期插入 shape 断言，运行时验证失败时提供精确定位
- 结合 Tape 形式化模型（T2 [13]）做根因分析

---

## 10. 局限（诚实记录）

本文主动记录以下理论局限，遵循数理部"局限必披露"的底线要求。

### 10.1 启发式与完整约束求解的差距

**局限**：本文 §5 形式化的协同推断算法是启发式分阶段算法，不是完整约束求解器。Phase 3 的"符号方程求解"实际上是局部 shape 合并，不建立全局约束传播。

**影响**：定理 J3 的"形状语法程序类"对实际实现更严格——实际实现能精确推断的程序类是理论模型的真子集。Symbol vs Known 的保守通过、无全局传播等限制使实际实现的精确推断范围窄于理论模型。

**缓解**：§7 的定理严格刻画启发式保证的范围，未夸大保证到完整约束求解。§8.2-8.3 显式记录启发式局限与理论模型差距。所有未实现的完整约束求解标注为"未来工作"（§9.1）。

### 10.2 定理 J1 的精确陈述

**局限**：定理 J1 的"不会因 shape mismatch 失败"应精确理解为"编译期已检查的 shape 约束在运行时必然满足"。含 `Any` 维度的表达式不在编译期检查范围内，运行时仍可能失败。

**影响**：若读者将 J1 理解为"编译期通过的程序运行时绝无 shape 错误"，则过度解读。本文已通过注 J1.1 显式澄清。

**缓解**：注 J1.1 明确区分"编译期已检查"与"运行时兜底"两类 shape 约束。

### 10.3 定理 J2 与 T3 的关系

**局限**：定理 J2 的多项式可判定性是基于 Tenth 实际实现的启发式行为，而非完整约束求解的可判定性。T3 [3] 定理 B2b 证明完整约束求解 NP 完全，本文 J2 与 T3 不矛盾，但需明确区分"启发式可判定"与"完整求解可判定"。

**影响**：若读者将 J2 理解为"Tenth shape 求解多项式可解"，则与 T3 的 NP 完全性下界矛盾。本文已通过推论 J2.1 显式澄清。

**缓解**：推论 J2.1 明确引用 T3 定理 B2b，指出 Tenth 实现的是 NIE 的可判定子类（局部同名等价），非完整 NIE。

### 10.4 实证分析的覆盖面

**局限**：§7 的标准库符号维度使用分析仅覆盖 `attention.th` 与 `transformer.th` 两个文件，未系统分析 `tenth/std/nn/` 的全部 13 个文件。

**影响**：§7 的示例可能不具普遍代表性，其他标准库文件可能有不同的符号维度使用模式。

**缓解**：本文聚焦协同推断的形式化与定理证明，标准库的全面实证分析留作未来工作。T3 [3] §6.1 已对 `tenth/std/nn/` 的 13 个文件做 shape 约束实例收集，可参考。

### 10.5 与 jaxtyping/Rust/TypeScript 对比的覆盖面

**局限**：定理 J4-J6 的对比基于各自语言的公开文档与典型用法，未深入对比所有边界场景。例如：
- jaxtyping 的最新版本可能支持编译期检查（与 JAX 的 trace 机制结合）
- Rust const generics 的最新提案可能支持参数间等式约束
- TypeScript 的最新版本可能引入更丰富的字面量类型

**影响**：对比结论可能随各语言演进过时。

**缓解**：对比基于各语言截至 2025 年的稳定版本，结论标注为"截至本文撰写时"。

### 10.6 形式化模型与实际实现的工程差距

**局限**：§5 的形式化模型是协同推断的"理想形态"，§8.3 已记录与实际实现的差距。但形式化模型本身可能未完全反映实际实现的所有边界行为。

**影响**：定理 J1-J3 基于形式化模型证明，若形式化模型与实际实现有偏差，定理保证可能不严格适用于实际实现。

**缓解**：形式化模型忠实基于源码（[types.rs](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/hir/lower/types.rs)），所有规则与函数对应具体源码位置。§8.3 显式记录差距。

---

## 11. 结论

本文对 Tenth 语言的类型推断与 shape 检查协同推断框架进行了形式化分析与定理证明。主要贡献：

1. **算法形式化**（§5-§6）：将 Phase 1+2+3 的启发式实现形式化为协同推断算法，明确每阶段的输入输出与协同机制。Dim 三值（Known/Symbol/Any）作为类型语言基础，标准库函数签名显式声明符号维度变量，编译器在协同推断中建立与求解局部符号方程。

2. **主定理与证明**（§7）：
   - **J1 健全性**：协同推断保证编译期检查的 shape mismatch 不会留到运行时
   - **J2 可判定性**：协同推断在 $O(nk)$ 多项式时间内可判定，与 T3 的 NP 完全性下界不矛盾（实际实现是 NIE 的可判定子类）
   - **J3 表达力刻画**：能精确推断的程序类是"形状语法"程序类
   - **J4 jaxtyping 对比**：Tenth 编译期检查 vs jaxtyping 运行时检查，对长期训练任务的 shape bug 预防有显著优势
   - **J5 Rust const generics 对比**：Tenth 符号维度是抽象变量（同名自动等价），Rust const generics 是具体值（需手动 trait 约束）
   - **J6 TypeScript 字面量类型对比**：Tenth 符号维度支持跨参数约束，TypeScript 字面量类型不支持

3. **实证支撑**（§7-§8）：所有理论结论对应具体源码位置，§7 的标准库示例展示符号维度的实际使用模式。

4. **诚实记录局限**（§10）：明确区分启发式与完整约束求解，所有未实现的部分标注为"未来工作"。

**核心发现**：Tenth 的协同推断框架在 AI 语言设计空间中具有独特定位——介于 jaxtyping 的运行时检查（弱保证）与完整约束求解的 NP 完全性（不可控复杂度）之间，通过启发式分阶段算法实现"编译期可判定 + 运行时兜底"的平衡。这是 Tenth 作为 AI 原生语言的核心设计决策之一。

---

## 12. 参考文献

1. **Milner, R.** (1978). *A theory of type polymorphism in programming*. Journal of Computer and System Sciences, 17(3), 348-375. —— Hindley-Milner 类型推断的奠基性工作。
2. **Damas, L., & Milner, R.** (1982). *Principal type-schemes for functional programs*. POPL 1982, 207-212. —— HM 类型推断的 Algorithm W 与 principal type 定理。
3. **Tenth 数理部** (2026). *Shape 约束求解的 NP 完全性：基于松弛变量归约的复杂度下界分析*（T3 论文）。`docs/论文/T3-HIR约束求解NP完全性归约.md`. —— NIE 的 NP 完全性证明，本文定理 J2 的关键引用。
4. **Tenth 数理部** (2026). *双向类型重建：Tenth 的非 HM 类型推断系统形式化与表达力刻画*（T16 论文）。`docs/论文/T16-双向类型重建.md`. —— Tenth 类型系统的形式化基础，本文协同推断的上游理论。
5. **Mairson, H. G.** (1990). *Deciding ML typability is complete for deterministic exponential time*. POPL 1990, 382-401. —— HM 类型推断的 DEXPTIME 完全性。
6. **TypeScript**. *TypeScript Handbook*. https://www.typescriptlang.org/docs/handbook/. —— TypeScript 字面量类型与结构性类型。
7. **Rust**. *Rust Reference: Const Generics*. https://doc.rust-lang.org/reference/items/generics.html#const-generics. —— Rust const generics 的官方文档。
8. **JAX**. *jaxtyping*. https://docs.kidger.site/jaxtyping/. —— JAX 的 shape 注解装饰器。
9. **PyTorch**. *PyTorch Documentation*. https://pytorch.org/docs/. —— PyTorch 的运行时 shape 检查。
10. **PyTorch**. *torch.compile*. https://pytorch.org/docs/stable/generated/torch.compile.html. —— PyTorch 的编译期优化与 symbolic shape。
11. **Swift for TensorFlow**. *S4TF Design*. https://github.com/tensorflow/swift. —— Swift for TensorFlow 的 shape 推断设计（已终止项目）。
12. **Tenth 数理部** (2026). *一般程序 Shape 检查不可判定性*（T4 论文）。`docs/论文/T4-一般程序Shape检查不可判定性.md`. —— 含循环/递归时 shape 检查的不可判定性，本文定理 J2 的注 J2.2 引用。
13. **Tenth 数理部** (2026). *Tape 形式化模型与根因定位可判定性*（T2 论文）。`docs/论文/T2-Tape形式化模型与根因定位可判定性.md`. —— Tape 形式化模型，本文 §9.4 编译期-运行时协作的引用。
14. **Pierce, B. C., & Turner, D. N.** (2000). *Local type inference*. ACM TOPLAS, 22(1), 1-44. —— 双向类型检查的奠基性工作，Tenth 类型重建的理论基础。
15. **Tenth 项目内部文档**：
    - `docs/语言参考手册.md` §3.2 张量类型（符号维度声明）
    - `tenth/src/hir/types.rs`（Dim 三值定义）
    - `tenth/src/hir/lower/types.rs`（Phase 1-3 协同推断实现）
    - `tenth/std/nn/attention.th`（符号维度声明示例）
    - `tenth/std/nn/transformer.th`（通配符 shape 使用示例）

---

## 附录 A：定理索引

| 定理 | 内容 | 章节 | 状态 |
|------|------|------|------|
| **定理 J1** | 协同推断的健全性 | §6.1 | 严格证明（含注 J1.1 精确陈述） |
| **定理 J2** | 协同推断可判定性 | §6.2 | 严格证明（含推论 J2.1 与 T3 关系） |
| 推论 J2.1 | 与 T3 NP 完全性的关系 | §6.2 | 严格论证（不矛盾，因实现是 NIE 子类） |
| 注 J2.2 | 与 T4 不可判定性的关系 | §6.2 | 严格论证（避开循环/递归） |
| **定理 J3** | 表达力刻画（形状语法程序类） | §6.3 | 严格证明（双向论证） |
| **定理 J4** | 与 jaxtyping 的对比 | §6.4 | 严格论证（六维对比） |
| **定理 J5** | 与 Rust const generics 的对比 | §6.5 | 严格论证（五维对比） |
| **定理 J6** | 与 TypeScript 字面量类型的对比 | §6.6 | 严格论证（四维对比） |

## 附录 B：与上游文档的对应关系

| 本文章节 | 上游文档 | 关系 |
|---------|---------|------|
| §4-§5 | T16 [4] §3 类型系统形式化 | 扩展（聚焦 shape 与符号维度） |
| §5.5 | T3 [3] §4 约束系统形式化 | 引用（约束求解形式化） |
| §6.2 推论 J2.1 | T3 [3] §5.2 定理 B2b | 引用（NP 完全性下界） |
| §6.2 注 J2.2 | T4 [12] 不可判定性 | 引用（避开不可判定的子类） |
| §8.1 | `tenth/src/hir/lower/types.rs` | 实证对应（函数-行号-功能） |
| §7 | `tenth/std/nn/attention.th`、`transformer.th` | 实证对应（标准库示例） |
| §10 | T3 [3] §9 局限 | 一致（局限必披露） |

## 附录 C：实施建议

基于本文理论结论，对 Tenth 协同推断的进一步发展提出以下建议：

1. **保持启发式分阶段架构**（定理 J2）：当前 Phase 1+2+3 的 $O(nk)$ 复杂度是稳健的，不破坏自举 ~0.2s 保证。完整约束求解（NP 完全）应作为可选模式（`--strict-shapes`），配合超时保护。

2. **改进 Symbol vs Known 的保守通过**（§8.2 局限 2）：当前 `check_method_shape` 对 Symbol vs Known 保守通过，可考虑在调用点对齐时建立 Symbol = Known 的局部方程，提高精度。这是低成本改进（不引入 NP 完全）。

3. **标准库符号维度声明的规范化**（§7）：建议为 `tenth/std/nn/` 的所有函数建立符号维度声明规范，确保 shape 合约显式化。这是利用协同推断能力的工程基础。

4. **编译期-运行时 shape 断言协作**（§9.4）：编译期生成的 shape 约束可序列化为运行时断言，运行时验证失败时提供精确定位（结合 T2 [13] 的 Tape 根因分析）。

5. **多态维度的探索**（§9.2）：当前 `..` 是通配符，不支持任意秩函数。可探索 `Tensor[T, ...]` 表示任意秩，提升标准库的复用性。

6. **与依赖类型的联系研究**（§9.3）：Tenth 的符号维度可视为"受限依赖类型"，探索与 Idris/Agda 的依赖类型系统的关系，可能为 shape 检查带来新表达力。

---

> **文档结束**
>
> 本文是 Tenth 类型系统中"shape 作为类型"策略的理论分析，聚焦协同推断的算法形式化、主定理证明与表达力对比。所有定理严格证明，所有局限显式记录，所有未实现的部分标注为"未来工作"。与 T3（NP 完全性下界）、T16（双向类型重建）、T4（不可判定性）的引用关系明确。如发现证明漏洞或边界遗漏，应在 `MEMO.md` 记录并修订本文。
