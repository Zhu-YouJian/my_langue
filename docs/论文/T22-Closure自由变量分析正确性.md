# Closure 自由变量分析的正确性：Tenth 闭包转换的形式化与指称语义一致性

> **Tenth 数理部 · 理论分析论文 T22**
> 版本：v1.0 | 日期：2026-07-02
> 适用：Tenth v0.3.3+
> 定位：会议/期刊级（闭包转换理论分析）
> 诚实声明：本文对 `collect_free_vars` 的正确性证明刻意保持谨慎。我们证明"引用健全性"（每个被收集的变量确实被引用）无条件成立，但"非绑定健全性"与"完备性"均在受控假设下成立——本文不掩盖 Block 顺序绑定遮蔽、Match 模式绑定、While/Loop 内 Let 跟踪缺失等导致的反例，而是将其作为独立局限章节诚实披露。复杂度分析中 `Vec<String> + retain` 相对 `HashSet` 的效率劣势亦显式标注。

---

## 摘要

Tenth 语言的闭包转换依赖一个约 190 行 Rust 实现的递归自由变量收集器 `collect_free_vars`，其为每个 `Closure` 节点计算 `captures` 列表。本文对该收集器进行形式化建模，给出五条主定理：(FV1) 引用健全性——任何被收集的变量确实在表达式内被引用；(FV2) 受控完备性——在四条假设下，所有真自由变量均被收集；(FV3) Block 顺序绑定正确性——多 Let 顺序绑定的自由变量在无前置遮蔽引用时正确；(FV4) 嵌套闭包捕获穿透——内层闭包的自由变量正确传播至外层；(FV5) 复杂度上界——单闭包分析为 O(n²)，其中 `retain` 操作贡献 O(n·m)。本文**诚实披露**四条局限（L1–L4）：Block 内对外层变量的前置遮蔽引用导致完备性失败；Match 守卫未处理导致完备性失败；Match 模式绑定未移除导致非绑定健全性失败；While/Loop 体内 Let 绑定未跟踪导致非绑定健全性失败。与 OCaml、Haskell、Rust 的对比表明，Tenth 的递归遍历在结构上与经典方法一致，但其 `Vec<String> + retain` 实现相对 `HashSet` 存在 O(n)→O(1) 的渐近劣势，这是工程简单性的代价。

**关键词**：闭包转换；自由变量分析；指称语义；结构归纳；遮蔽；捕获穿透；复杂度分析

---

## 1. 引言

### 1.1 闭包转换的挑战

闭包转换（closure conversion）是函数式语言编译中的经典变换，其目标是将嵌套函数（可捕获外层变量）转换为不依赖词法作用域的"扁平"闭包对象 [Appel, 1992]。该变换的核心步骤是**自由变量分析**（free variable analysis）：对每个内嵌闭包，识别其引用但不绑定的变量，将其打包为捕获列表（capture list）。

自由变量分析的难点在于：

1. **作用域处理**：`Let`、`For`、`Closure`、`Block` 等绑定结构各有其作用域规则，必须精确区分绑定变量与自由变量。
2. **遮蔽（shadowing）**：同名变量可在不同作用域重复绑定，内层绑定遮蔽外层，必须正确处理遮蔽边界。
3. **嵌套闭包的捕获穿透**：内层闭包捕获的变量，若在外层闭包中未绑定，则成为外层闭包的自由变量——捕获穿透。
4. **顺序绑定**：`Block` 内多个 `Let` 按顺序绑定，后一个 `Let` 可引用前一个 `Let` 绑定的变量，但前者不可引用后者的变量。
5. **完备性与健全性的张力**：过近似（over-approximation）导致捕获冗余，欠近似（under-approximation）导致语义错误。

### 1.2 自由变量分析的核心地位

自由变量分析的正确性是闭包转换语义保持的**前提**：

- 若**欠近似**（漏掉真自由变量）：闭包运行时访问未捕获的变量，导致未定义行为或崩溃。
- 若**过近似**（多收集非自由变量）：捕获冗余，内存浪费，但不破坏语义。

因此，工业级编译器通常选择**过近似**策略以保证健全性。Tenth 的 `collect_free_vars`（[tenth/src/hir/lower/closures.rs:17-162](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/hir/lower/closures.rs)）采用了**精确匹配 + retain 移除**的混合策略，在大多数场景下既健全又完备，但存在若干边界场景下的失效（详见第 7 章局限分析）。

### 1.3 Tenth 的 collect_free_vars 实现

Tenth 的自由变量分析实现于 [tenth/src/hir/lower/closures.rs:9-189](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/hir/lower/closures.rs)。其调用点为 [tenth/src/hir/lower/lower_expr.rs:511](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/hir/lower/lower_expr.rs)：

```rust
let captures = Self::free_vars_in(&b);
```

该调用将闭包体 `b` 的自由变量收集为 `captures` 列表，存入 `HirExprKind::Closure { params, body, captures }`（[tenth/src/hir/hir.rs:74-78](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/hir/hir.rs)）。

算法的核心结构：

1. `free_vars_in(expr)`：入口，创建空 `Vec<String>`，调用 `collect_free_vars`，最后 `sort + dedup` 去重。
2. `collect_free_vars(expr, vars)`：对每种 `HirExprKind` 定义收集规则，递归遍历。
3. `collect_free_vars_stmt(stmt, vars)`：对 `HirStmtKind` 的补充处理。

数据结构选择：使用 `Vec<String>` 而非 `HashSet<String>`，依赖 `retain` 进行批量移除，依赖 `sort + dedup` 进行最终去重。

### 1.4 贡献

本文的贡献如下：

1. **形式化建模**：将 HIR 节点的指称语义、自由变量集合 FV(e)、作用域规则形式化为数学对象（第 3 章）。
2. **引用健全性证明**（定理 FV1）：证明 `collect_free_vars` 收集的每个变量确实在表达式内被引用，无条件成立。
3. **受控完备性证明**（定理 FV2）：在四条受控假设下，证明所有真自由变量被收集。
4. **Block 顺序绑定正确性**（定理 FV3）：证明多 Let 顺序绑定的自由变量在无前置遮蔽引用时正确。
5. **嵌套闭包捕获穿透**（定理 FV4）：证明内层闭包的自由变量正确传播至外层。
6. **复杂度上界**（定理 FV5）：证明单闭包分析为 O(n²)，其中 `retain` 贡献 O(n·m)。
7. **诚实披露四条局限**（L1–L4）：Block 前置遮蔽、Match 守卫、Match 模式绑定、While/Loop Let 跟踪。
8. **跨语言对比**：与 OCaml、Haskell、Rust 的闭包捕获机制对比，定位 Tenth 的工程取舍。

---

## 2. 背景与相关工作

### 2.1 闭包转换的经典理论

Appel [1992] 在 *Compiling with Continuations* 中系统阐述了闭包转换的理论框架。其核心思想是将每个 λ 抽象转换为 `(env, code)` 对，其中 `env` 是自由变量的记录，`code` 是接受 `env` 与参数的函数。自由变量分析是该变换的前提：

$$\text{FV}(\lambda x.e) = \text{FV}(e) \setminus \{x\}$$

Appel 的分析使用集合语义（set semantics），保证无重复且高效去重。其实现通常基于 `Set` 数据结构，渐近复杂度为 O(n)。

### 2.2 OCaml 的闭包转换

OCaml 编译器的闭包转换 [Leroy, 1992] 使用基于 `Set.t` 的自由变量分析。其 `free_vars` 函数对每种 AST 节点定义收集规则，递归遍历后用集合差集移除绑定变量。OCaml 的实现特点：

- 使用不可变集合，函数式风格。
- 支持 `let rec` 的递归绑定，需特殊处理。
- 复杂度为 O(n log n)（基于平衡树集合）。

### 2.3 Haskell 的 set-level analysis

Haskell 的 GHC 编译器使用更精细的 **set-level analysis** [Marlow & Peyton Jones, 2004]。其特点：

- 不仅分析自由变量，还分析闭包的**分配级别**（allocation level），决定闭包是否应分配在堆上或栈上。
- 基于 `VarSet`（IntMap 实现），O(1) 增删。
- 处理惰性求值的复杂性：thunk 的自由变量可能跨多个 suspension。

### 2.4 Rust 的闭包捕获（Fn/FnMut/FnOnce）

Rust 的闭包捕获机制 [Rust Reference, 2024] 与前三者本质不同：

- **Fn**：以不可变引用捕获，可多次调用。
- **FnMut**：以可变引用捕获，可多次调用但需独占。
- **FnOnce**：以值捕获（move），仅可调用一次。

Rust 的捕获分析在类型检查阶段完成，由 trait 解析器推断闭包应实现哪种 trait。其自由变量分析隐式嵌入于 trait 推断中，不像 Tenth/OCaml/Haskell 有显式的 `free_vars` 函数。Rust 还支持 `move` 关键字强制按值捕获。

### 2.5 Tenth 的定位

Tenth 的 `collect_free_vars` 在结构上最接近 OCaml 的 `free_vars`：递归遍历 + 作用域移除。但在数据结构上选择 `Vec<String>` 而非 `Set`，导致渐近复杂度从 O(n) 退化为 O(n²)（详见第 8 章复杂度分析）。这一选择的工程动机是简化实现、避免 `HashSet` 的哈希开销，但代价是理论上的渐近劣势。

---

## 3. HIR 自由变量的指称语义

### 3.1 HIR 节点的指称语义

我们采用指称语义（denotational semantics）框架，将每个 HIR 表达式 `e` 映射到其指称对象 `⟦e⟧`。对于自由变量分析，我们关心的指称是**变量引用集合**与**变量绑定集合**。

**定义 3.1（变量引用集 Refs）**：对 HIR 表达式 `e`，定义 `Refs(e)` 为 `e` 中所有变量引用的集合：

$$\text{Refs}(e) = \{ v \mid \text{e contains a Var}(v) \text{, Assign}\{target: v, \ldots\} \text{, AssignOp}\{target: v, \ldots\} \text{, or InterpPart::Expr}(v) \}$$

注意：`Refs` 不考虑绑定结构，仅收集所有"出现"的变量名。

**定义 3.2（绑定集 Binders）**：对 HIR 表达式 `e`，定义 `Binders(e)` 为 `e` 中所有绑定变量的集合，按绑定结构分类：

- `Block { stmts, ... }`：`Binders(Block) = ⋃_{s ∈ stmts} Binders_Stmt(s)`，其中 `Binders_Stmt(Let { names, ... }) = names`，其余为空。
- `Closure { params, ... }`：`Binders(Closure) = { n | (n, _) ∈ params }`。
- `For { var, ... }`（语句层）：`Binders(For) = { var }`。

**定义 3.3（作用域结构 Scope）**：作用域是一个二元组 `(Binders, Body)`，其中 `Binders` 是该作用域引入的绑定变量集合，`Body` 是绑定生效的表达式子树。

HIR 中有四类作用域：

1. **Let 作用域**：`Let { names, init, ... }` 中的 `names` 在 `init` 之后的同一 `Block` 内生效。
2. **For 作用域**：`For { var, iter, body }` 中的 `var` 在 `body` 内生效。
3. **Closure 作用域**：`Closure { params, body, ... }` 中的 `params` 在 `body` 内生效。
4. **Block 作用域**：`Block { stmts, final_expr }` 中的 `stmts` 内 `Let` 绑定的变量在 `Block` 内生效，**Block 结束后失效**。

### 3.2 自由变量的指称定义

**定义 3.4（自由变量集 FV）**：对 HIR 表达式 `e`，自由变量集 `FV(e)` 递归定义如下：

$$
\begin{aligned}
\text{FV}(\text{Var}(v)) &= \begin{cases} \emptyset & \text{if } v \text{ is builtin or } v \text{ contains } :: \\ \{v\} & \text{otherwise} \end{cases} \\
\text{FV}(\text{Literal}(\_)) &= \emptyset \\
\text{FV}(\text{Binary}\{l, r, \ldots\}) &= \text{FV}(l) \cup \text{FV}(r) \\
\text{FV}(\text{Unary}\{e, \ldots\}) &= \text{FV}(e) \\
\text{FV}(\text{Call}\{f, args, \ldots\}) &= \text{FV}(f) \cup \bigcup_{a \in args} \text{FV}(a) \\
\text{FV}(\text{Closure}\{params, body, \ldots\}) &= \text{FV}(body) \setminus \{n \mid (n, \_) \in params\} \\
\text{FV}(\text{Block}\{stmts, final\_expr\}) &= \left( \bigcup_{s \in stmts} \text{FV}_\text{stmt}(s, B_{<s}) \cup \text{FV}(final\_expr, B_\text{all}) \right) \setminus B_\text{all} \\
\text{FV}(\text{Assign}\{target, value\}) &= \{target\} \cup \text{FV}(value) \\
\text{FV}(\text{Match}\{scrutinee, arms\}) &= \text{FV}(scrutinee) \cup \bigcup_{arm} \left( \text{FV}(arm.body) \setminus \text{Binders}(arm.pattern) \cup \text{FV}(arm.guard) \right)
\end{aligned}
$$

其中 `B_<s` 表示 `s` 之前所有 `Let` 绑定的变量集合（顺序绑定语义），`B_all` 表示 `Block` 内所有 `Let` 绑定的变量集合。

**Block 的精确定义**：对于顺序绑定，`FV_stmt(s_i, B_<s_i)` 表示在处理 `s_i` 时，`B_<s_i` 中的变量已被绑定（不应视为自由）。最终从并集中移除 `B_all`，因为块内绑定的变量对块外不是自由的。

**定义 3.5（语句层自由变量 FV_stmt）**：

$$
\begin{aligned}
\text{FV}_\text{stmt}(\text{Let}\{names, init, \ldots\}, B) &= \text{FV}(init) \setminus B \\
\text{FV}_\text{stmt}(\text{Expr}(e), B) &= \text{FV}(e) \setminus B \\
\text{FV}_\text{stmt}(\text{Return}(e), B) &= \text{FV}(e) \setminus B \\
\text{FV}_\text{stmt}(\text{While}\{cond, body\}, B) &= (\text{FV}(cond) \cup \text{FV}_\text{stmt}(body, B)) \setminus B \\
\text{FV}_\text{stmt}(\text{For}\{var, iter, body\}, B) &= (\text{FV}(iter) \cup (\text{FV}_\text{stmt}(body, B \cup \{var\}) \setminus \{var\})) \setminus B \\
\text{FV}_\text{stmt}(\text{Loop}\{body\}, B) &= \left( \bigcup_{s \in body} \text{FV}_\text{stmt}(s, B) \right) \setminus B
\end{aligned}
$$

注意：`FV_stmt` 的第二个参数 `B` 是当前已绑定的变量集合，用于正确处理顺序绑定中的遮蔽。

### 3.3 遮蔽与捕获的形式化

**定义 3.6（遮蔽 Shadowing）**：变量 `v` 在表达式 `e` 中被遮蔽，若存在子表达式 `e' ⊆ e` 使得 `v ∈ Binders(e')` 且 `e'` 内引用的 `v` 指向 `e'` 的绑定而非外层绑定。

**定义 3.7（捕获 Capture）**：闭包 `c = Closure{params, body}` 捕获变量 `v`，当且仅当 `v ∈ FV(body) \ params`。`captures(c) = FV(body) \ params`。

**定义 3.8（捕获穿透 Capture穿透）**：对于嵌套闭包 `c_outer = Closure{p_o, body_o}`，其中 `body_o` 包含 `c_inner = Closure{p_i, body_i}`：

$$\text{captures}(c_\text{outer}) \supseteq \text{captures}(c_\text{inner}) \setminus \text{params}(c_\text{outer})$$

即内层闭包捕获的变量，若未被外层闭包参数绑定，则成为外层闭包的捕获。

### 3.4 内置名称集

**定义 3.9（内置名称集 BUILTINS）**：以下名称不视为自由变量：

$$\text{BUILTINS} = \{ \text{println, eprintln, tensor, rand, randn, randn\_f32, rand\_f32, zeros\_f32, ones\_f32, read\_file, write\_file, str\_at, Vec::new, HashMap::new, compile\_host, compile\_program, write\_bytes, start\_grad, new\_grad, stop\_grad, param, backward, grad, zero\_grad, cross\_entropy, abs, sqrt, sin, cos, ln, pow, zeros, ones, save\_weights, load\_weights, lexer\_new, lexer\_tokenize, parse\_program, lower\_program, compile\_to\_wasm, self} \}$$

此外，任何包含 `::` 的名称视为限定路径，不计入自由变量。

**定义 3.10（有效引用 ValidRef）**：`v` 是 `e` 中的有效引用，若 `v ∈ Refs(e)` 且 `v ∉ BUILTINS` 且 `v` 不含 `::`。

---

## 4. 主定理与证明

### 4.1 定理 FV1（引用健全性）

**定理 FV1（引用健全性）**：对任意 HIR 表达式 `e`，若 `v ∈ collect_free_vars(e)`，则 `v ∈ Refs(e)` 且 `v ∉ BUILTINS` 且 `v` 不含 `::`。

**证明**：对 `e` 的结构进行归纳。记 `CFV(e, vars)` 为 `collect_free_vars(e, vars)` 执行后 `vars` 的增量。

**基例**：

- `Literal(_)`：`collect_free_vars` 不向 `vars` 添加任何内容。`Refs(Literal) = ∅`。结论成立。✓
- `Var(name)`：
  - 若 `name` 含 `::`：直接 return，不添加。`Refs` 中虽有 `name`，但 `name ∉ CFV`。结论"若 `v ∈ CFV` 则 `v ∈ Refs`"的前件为假，空真成立。✓
  - 若 `name ∈ BUILTINS`：match 命中 builtin 分支，不添加。同上，空真。✓
  - 否则：`vars.push(name)`，故 `name ∈ CFV`。`Refs(Var(name)) = {name}`，故 `name ∈ Refs`。`name ∉ BUILTINS`（否则命中上一分支）。✓

**归纳步**：假设对所有子表达式 `e_i`，定理成立（归纳假设 IH）。

- `Binary{left, right, ..}`：`CFV = CFV(left) ∪ CFV(right)`。`Refs = Refs(left) ∪ Refs(right)`。由 IH，`CFV(left) ⊆ Refs(left)`，`CFV(right) ⊆ Refs(right)`。故 `CFV ⊆ Refs`。✓

- `Unary{expr, ..}`：`CFV = CFV(expr)`。由 IH，`CFV ⊆ Refs(expr) = Refs(Unary)`。✓

- `Call{func, args, ..}`、`GenericCall`、`MethodCall`：类似 Binary，对 `func` 与每个 `arg` 应用 IH。✓

- `Index{target, indices}`：对 `target` 与每个索引表达式应用 IH。✓

- `Field{target, ..}`：对 `target` 应用 IH。✓

- `TensorLiteral{data, ..}`、`ArrayLiteral{elements, ..}`：对每个元素应用 IH。✓

- `Range{start, end, ..}`：对 `start`、`end`（若存在）应用 IH。✓

- `If{cond, then, else, ..}`：对 `cond`、`then`、`else`（若存在）应用 IH。✓

- `Block{stmts, final_expr}`：
  - 算法遍历 `stmts`，对每个 `s` 调用 `collect_free_vars_stmt(s, vars)`，对 `final_expr` 调用 `collect_free_vars(e, vars)`。
  - 最终 `vars.retain(|v| !bound.contains(v))`，仅移除变量，不添加。
  - 故 `CFV(Block) ⊆ (∪_s CFV_stmt(s)) ∪ CFV(final_expr)`。
  - 由 IH（语句层引理 4.1），`CFV_stmt(s) ⊆ Refs(s) ⊆ Refs(Block)`。
  - 由 IH，`CFV(final_expr) ⊆ Refs(final_expr) ⊆ Refs(Block)`。
  - 故 `CFV(Block) ⊆ Refs(Block)`。✓
  - 关于 `v ∉ BUILTINS`：算法在 `Var` 节点过滤 builtin，其他节点不引入新变量名，仅传递。故 `CFV(Block) ∩ BUILTINS = ∅`。✓

- `Closure{params, body, ..}`：
  - 算法创建 `inner_vars`，调用 `collect_free_vars(body, &mut inner_vars)`。
  - `inner_vars.retain(|v| !param_names.contains(v))`：仅移除，不添加。
  - `vars.extend(inner_vars)`：将 `inner_vars` 的内容添加到 `vars`。
  - 故 `CFV(Closure) = CFV(body)`。
  - 由 IH，`CFV(body) ⊆ Refs(body) ⊆ Refs(Closure)`。✓

- `Assign{target, value}`：
  - `vars.push(target)`：`target ∈ Refs(Assign)`（由定义 3.1）。✓
  - `collect_free_vars(value, vars)`：由 IH，`CFV(value) ⊆ Refs(value) ⊆ Refs(Assign)`。✓
  - 但需检查 `target` 是否为 builtin：算法**未**对 `Assign` 的 `target` 进行 builtin 过滤！
  - **潜在问题**：若 `target` 是 builtin 名称（如 `self`），会被错误添加。
  - **实际影响**：`self` 不能作为赋值目标（语义限制），故该问题在实践中不触发。
  - 严格地说，定理 FV1 的"`v ∉ BUILTINS`"部分对 `Assign.target` 存在理论瑕疵。我们将其记录为**局限 L5**（理论瑕疵，实际不触发）。

- `AssignOp{target, ..}`：同 Assign。✓（同 L5）

- `StructLiteral`、`EnumLiteral`：对每个字段值应用 IH。✓

- `Match{scrutinee, arms}`：
  - 对 `scrutinee` 应用 IH：`CFV(scrutinee) ⊆ Refs(scrutinee)`。✓
  - 对每个 `arm.body` 应用 IH：`CFV(arm.body) ⊆ Refs(arm.body)`。✓
  - **但 `arm.guard` 未被处理**！算法未调用 `collect_free_vars` 对 `arm.guard`。
  - 这意味着 `arm.guard` 中的变量引用不会被收集。
  - 这是**完备性问题**（FV2 的反例），不是 FV1 的问题（FV1 只要求收集的是引用，guard 中的变量不被收集不影响 FV1）。✓

- `Ref(inner)`、`MutRef(inner)`、`Deref(inner)`、`Move(inner)`、`TryBlock(inner)`：对 `inner` 应用 IH。✓

- `InterpolatedString{parts}`：对每个 `InterpPart::Expr(name)`，`vars.push(name)`。`name ∈ Refs`（由定义 3.1）。但同样未过滤 builtin——若 `name` 是 builtin，会被错误添加。**同 L5**。✓（理论上）

- `Tuple(elems)`：对每个元素应用 IH。✓

- `DerefAssign{target, value}`、`DerefAssignOp{..}`、`FieldAssign{..}`：对 `target` 和 `value` 应用 IH（注意这里 `target` 是 `HirExpr` 而非 `String`，故走 `collect_free_vars(target, vars)` 路径，由 IH 成立）。✓

**语句层引理 4.1**：对任意语句 `s`，`CFV_stmt(s) ⊆ Refs(s)`。

证明类似，对 `HirStmtKind` 归纳：

- `Let{init, ..}`：`CFV = CFV(init) ⊆ Refs(init) = Refs(Let)`。✓
- `Expr(e)`：`CFV = CFV(e) ⊆ Refs(e) = Refs(Expr)`。✓
- `Return(e)`：类似。✓
- `While{cond, body}`：`CFV = CFV(cond) ∪ CFV_stmt(body)`。由 IH 成立。✓
- `For{var, iter, body}`：`CFV = CFV(iter) ∪ (CFV_stmt(body) \ {var})`。`CFV_stmt(body) \ {var} ⊆ CFV_stmt(body) ⊆ Refs(body)`。但 `var` 是绑定变量，`Refs(body)` 可能含 `var`。移除 `var` 后仍 ⊆ `Refs(body)`。✓
- `Break`、`Continue`：无变量。✓
- `Loop{body}`：对每个语句应用 IH。✓

综上，定理 FV1 成立（ modulo L5 的理论瑕疵，实际不触发）。$\square$

### 4.2 定理 FV2（受控完备性）

**定理 FV2（受控完备性）**：对任意 HIR 表达式 `e`，若以下四条假设成立：

- **假设 A1（无前置遮蔽引用）**：`e` 中不存在 `Block` 使得块内某 `Let` 绑定变量 `v`，且在该 `Let` 之前的语句中引用了外层 `v`。
- **假设 A2（无 Match 守卫）**：`e` 中的 `Match` 分支均无 `guard`，或守卫中的变量已被其他途径处理。
- **假设 A3（无 Match 模式绑定）**：`e` 中的 `Match` 分支模式均为 `Wildcard` 或不绑定变量的字面量模式，或模式绑定的变量不在 `arm.body` 中引用。
- **假设 A4（While/Loop body 为 Block 包装）**：`e` 中的 `While`、`Loop` 的 `body` 均为 `Expr(Block{...})` 形式，使 `Block` 的 bound 跟踪生效。

则对任意 `v`，若 `v ∈ FV(e)`（定义 3.4），则 `v ∈ collect_free_vars(e)`。

**证明**：对 `e` 的结构进行归纳。

**基例**：

- `Literal(_)`：`FV = ∅`。空真成立。✓
- `Var(name)`：
  - 若 `name` 含 `::` 或 ∈ BUILTINS：`FV = ∅`（定义 3.4 + 3.9）。空真。✓
  - 否则：`FV = {name}`。算法 `vars.push(name)`，故 `name ∈ CFV`。✓

**归纳步**：

- `Binary`、`Unary`、`Call`、`GenericCall`、`MethodCall`、`Index`、`Field`、`TensorLiteral`、`ArrayLiteral`、`Range`、`If`、`Tuple`、`Ref`、`MutRef`、`Deref`、`Move`、`TryBlock`：FV 为子表达式 FV 的并集，CFV 同理。由 IH 直接传递。✓

- `Assign{target, value}`：
  - `FV = {target} ∪ FV(value)`。
  - `{target}`：算法 `vars.push(target)`。但需检查 `target` 是否被后续 retain 移除——若 `Assign` 在某 `Block` 内且 `target` 被同块 `Let` 绑定，则会被移除。
  - 由 A1，若 `target` 是外层变量（非块内 Let 绑定），则不会被移除。
  - 若 `target` 是块内 Let 绑定的变量，则 `target ∉ FV(Assign)`（因 `target` 在块内绑定）。
  - 故 `{target} ⊆ CFV` 或 `{target} ∩ FV = ∅`。两种情况均满足 FV2。✓
  - `FV(value)`：由 IH，`FV(value) ⊆ CFV(value)`。`CFV(value)` 被加入 `vars`。若 `value` 在 `Block` 内，可能被 retain 移除——但被移除的是 `bound` 中的变量，而 `bound` 中的变量 ∉ FV(value)（因其在块内绑定，不在 `value` 中自由）。✓

- `AssignOp{target, ..}`：同 Assign。✓

- `Closure{params, body, ..}`：
  - `FV = FV(body) \ params`。
  - 算法：`inner_vars = CFV(body)`，然后 `inner_vars.retain(|v| !params.contains(v))`，最后 `vars.extend(inner_vars)`。
  - 由 IH（对 `body`），`FV(body) ⊆ CFV(body) = inner_vars`。
  - `retain` 移除 `params` 中的变量：`FV(body) \ params ⊆ inner_vars \ params`。
  - `extend` 将 `inner_vars \ params` 加入 `vars`。
  - 故 `FV(body) \ params ⊆ vars`，即 `FV(Closure) ⊆ CFV(Closure)`。✓

- `Block{stmts, final_expr}`：
  - `FV(Block) = (∪_s FV_stmt(s, B_<s) ∪ FV(final_expr, B_all)) \ B_all`。
  - 算法：遍历 `stmts`，对每个 `s` 调用 `collect_free_vars_stmt(s, vars)`，对 `final_expr` 调用 `collect_free_vars(e, vars)`，最后 `vars.retain(|v| !bound.contains(v))`。
  - `bound = B_all`（所有 Let 绑定的变量，无论位置）。
  - **关键观察**：算法在遍历前先收集 `bound`，然后遍历。这意味着：
    - 对 `s_i` 之前的 Let 绑定的变量 `v`（`v ∈ B_<s_i`）：算法将 `v` 加入 `bound`，但遍历 `s_i` 时 `v` 仍在 `vars` 中（若被引用）。最终 `retain` 移除 `v`。
    - 由 A1，`s_i` 中不引用外层 `v`（与块内 `v` 同名）。故 `s_i` 中对 `v` 的引用均指向块内 `v`，`v ∈ B_all`，`v ∉ FV_stmt(s_i, B_<s_i)`。
    - 由 IH（语句层），`FV_stmt(s_i, B_<s_i) ⊆ CFV_stmt(s_i)`（算法不考虑 `B_<s_i`，但由 A1，`B_<s_i` 中的变量不在 `s_i` 中作为自由变量出现）。
  - 综合：`∪_s FV_stmt(s, B_<s) ⊆ ∪_s CFV_stmt(s)`。
  - `FV(final_expr, B_all) ⊆ CFV(final_expr)`：由 IH，`FV(final_expr) ⊆ CFV(final_expr)`。`FV(final_expr, B_all) = FV(final_expr) \ B_all ⊆ FV(final_expr) ⊆ CFV(final_expr)`。
  - 故 `∪_s FV_stmt(s, B_<s) ∪ FV(final_expr, B_all) ⊆ ∪_s CFV_stmt(s) ∪ CFV(final_expr) = vars_before_retain`。
  - `FV(Block) = (vars_before_retain 的语义对应) \ B_all`。
  - 算法 `retain` 移除 `B_all`：`CFV(Block) = vars_before_retain \ B_all`。
  - 故 `FV(Block) ⊆ CFV(Block)`。✓
  - **注意**：此处 A1 是关键。若无 A1，`s_i` 中可能引用外层 `v`（`v ∈ FV_stmt(s_i, B_<s_i)`），但算法仍将 `v` 加入 `bound` 并最终移除，导致 `v ∉ CFV(Block)`——完备性失败。

- `Match{scrutinee, arms}`：
  - `FV(Match) = FV(scrutinee) ∪ ∪_arm (FV(arm.body) \ Binders(arm.pattern) ∪ FV(arm.guard))`。
  - 算法：`CFV(scrutinee)` + `∪_arm CFV(arm.body)`。
  - 由 IH，`FV(scrutinee) ⊆ CFV(scrutinee)`。✓
  - 由 IH，`FV(arm.body) ⊆ CFV(arm.body)`。但 `FV(Match)` 中是 `FV(arm.body) \ Binders(pattern)`。
    - 由 A3，`Binders(pattern)` 中的变量不在 `arm.body` 中引用，或模式为 Wildcard。故 `FV(arm.body) \ Binders(pattern) = FV(arm.body)`。
    - 故 `FV(arm.body) \ Binders(pattern) ⊆ CFV(arm.body)`。✓
  - 由 A2，`FV(arm.guard) = ∅` 或已处理。故 `FV(arm.guard)` 不贡献。✓
  - 综合：`FV(Match) ⊆ CFV(Match)`。✓

- `StructLiteral`、`EnumLiteral`：对字段值应用 IH。✓

- `InterpolatedString{parts}`：
  - `FV = {name | InterpPart::Expr(name), name ∉ BUILTINS, name 不含 ::}`。
  - 算法：对每个 `InterpPart::Expr(name)`，`vars.push(name)`。
  - 但算法**未**过滤 builtin！若 `name` 是 builtin，算法仍 push，而 `FV` 中不含。
  - 这是 FV1 的瑕疵（L5），但对 FV2 无影响：`FV` 中的变量（非 builtin）一定被 push。✓

- `DerefAssign`、`DerefAssignOp`、`FieldAssign`：对 `target` 和 `value` 应用 IH。✓

**语句层引理 4.2**：在假设 A1、A4 下，对任意语句 `s`，`FV_stmt(s, B) ⊆ CFV_stmt(s)`（算法不考虑 `B`，但由 A1、A4，`B` 中的变量不在 `s` 中作为自由变量出现）。

证明类似 4.1，关键点：

- `For{var, iter, body}`：
  - `FV_stmt = (FV(iter) ∪ (FV_stmt(body, B ∪ {var}) \ {var})) \ B`。
  - 算法：`CFV(iter) + (CFV_stmt(body) \ {var})`。
  - 由 IH，`FV(iter) ⊆ CFV(iter)`。✓
  - 由 IH（对 body），`FV_stmt(body, B ∪ {var}) ⊆ CFV_stmt(body)`。注意算法不传 `B ∪ {var}`，但 `var` 被 retain 移除。由 A4，body 为 Block 包装，Block 内 Let 由 Block 的 bound 跟踪处理。✓
  - `FV_stmt(body, B ∪ {var}) \ {var} ⊆ CFV_stmt(body) \ {var}`。✓

- `While{cond, body}`：
  - 由 A4，body 为 Block 包装。`FV_stmt(body, B) ⊆ CFV_stmt(body)`。✓

- `Loop{body}`：
  - 由 A4，body 中各语句若含 Let，应在 Block 内。但 `Loop.body` 是 `Vec<HirStmt>`，非 `Block`。
  - **问题**：若 `Loop.body` 直接含 `Let` 语句（非 Block 包装），算法不跟踪这些 Let。
  - 由 A4，`Loop.body` 中的 Let 应在 Block 内。但 A4 的表述是"While/Loop 的 body 均为 Expr(Block{...})"——这对 `Loop.body: Vec<HirStmt>` 不直接适用。
  - **修正 A4**：将 A4 改为"`e` 中不存在 `While`、`Loop`、`For` 的 body 直接含 `Let` 语句（非经 Block 包装）"。
  - 在修正的 A4 下，`Loop.body` 中的语句均为非 Let 语句（如 `Expr`、`Return` 等），或 `Expr(Block{...})`。由 IH 成立。✓

综上，在假设 A1、A2、A3、A4 下，定理 FV2 成立。$\square$

### 4.3 定理 FV3（Block 顺序绑定正确性）

**定理 FV3（Block 顺序绑定正确性）**：在假设 A1 下，对 `Block{stmts, final_expr}`，其中 `stmts = [Let(x_1, e_1), Let(x_2, e_2), ..., Let(x_k, e_k), ...]`，`collect_free_vars` 正确处理顺序绑定：

$$\text{CFV}(Block) = \left( \bigcup_{i=1}^{k} \text{FV}(e_i) \setminus \{x_1, ..., x_{i-1}\} \right) \cup \text{FV}(final\_expr) \setminus \{x_1, ..., x_k\}$$

即 `e_i` 可引用 `x_1, ..., x_{i-1}`（已绑定）但不引用 `x_i, ..., x_k`（未绑定），`final_expr` 可引用所有 `x_1, ..., x_k`。

**证明**：

算法对 `Block` 的处理：

1. **预收集 bound**：`bound = [x_1, x_2, ..., x_k]`（所有 Let 绑定的变量）。
2. **遍历 stmts**：对 `Let(x_i, e_i)`，调用 `collect_free_vars(e_i, vars)`。`e_i` 中引用的变量被加入 `vars`，包括：
   - 外层自由变量 `v`（`v ∉ bound`）：正确，应保留。
   - `x_1, ..., x_{i-1}`（已绑定）：被加入 `vars`，但应被移除（非自由）。
   - `x_i, ..., x_k`（未绑定）：由 A1，`e_i` 不引用 `x_i, ..., x_k`（否则是前置引用，违反顺序绑定语义；若引用同名外层变量，则违反 A1）。
3. **遍历 final_expr**：`collect_free_vars(final_expr, vars)`。`final_expr` 可引用 `x_1, ..., x_k`（均被加入 `vars`）和外层自由变量。
4. **retain 移除**：`vars.retain(|v| !bound.contains(v))`。移除 `x_1, ..., x_k`。

**分析**：

- 步骤 2 中，`e_i` 引用 `x_j`（`j < i`）：`x_j` 被加入 `vars`。步骤 4 移除 `x_j`（`x_j ∈ bound`）。**正确**：`x_j` 在块内绑定，非自由。✓
- 步骤 2 中，`e_i` 引用外层 `v`：`v` 被加入 `vars`。步骤 4 不移除 `v`（`v ∉ bound`）。**正确**：`v` 是自由变量。✓
- 步骤 3 中，`final_expr` 引用 `x_j`：`x_j` 被加入 `vars`。步骤 4 移除 `x_j`。**正确**：`x_j` 在块内绑定。✓
- 步骤 3 中，`final_expr` 引用外层 `v`：`v` 被加入 `vars`，不被移除。**正确**。✓

**反例（违反 A1）**：

考虑 `Block{stmts: [Expr(Var(x)), Let(x, e)], final_expr: None}`，其中外层存在 `x`。

- `FV(Block)`（语义）：`Expr(Var(x))` 中的 `x` 指向外层 `x`（在 `Let(x)` 之前），故 `x ∈ FV(Block)`。
- `CFV(Block)`（算法）：`bound = [x]`。`Expr(Var(x))` 将 `x` 加入 `vars`。`Let(x, e)` 处理 `e`。`retain` 移除 `x`。故 `x ∉ CFV(Block)`。
- **不完备**：`FV ≠ CFV`。

这正是 A1 排除的情形。$\square$

### 4.4 定理 FV4（嵌套闭包捕获穿透）

**定理 FV4（嵌套闭包捕获穿透）**：对嵌套闭包 `c_outer = Closure{p_o, body_o}`，其中 `body_o` 含 `c_inner = Closure{p_i, body_i}`，`collect_free_vars` 正确传播内层闭包的捕获：

$$\text{captures}(c_\text{inner}) \setminus p_o \subseteq \text{captures}(c_\text{outer})$$

其中 `captures(c) = CFV(body) \ params`。

**证明**：

算法对 `c_outer` 的处理（[closures.rs:107-114](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/hir/lower/closures.rs)）：

```rust
HirExprKind::Closure { params, body, .. } => {
    let mut inner_vars = Vec::new();
    Self::collect_free_vars(body, &mut inner_vars);
    let param_names: Vec<String> = params.iter().map(|(n, _)| n.clone()).collect();
    inner_vars.retain(|v| !param_names.contains(v));
    vars.extend(inner_vars);
}
```

设 `body_o` 含 `c_inner`。算法对 `body_o` 调用 `collect_free_vars(body_o, &mut inner_vars_o)`。

当遍历到 `c_inner` 时（`c_inner` 是 `body_o` 的子表达式）：

```rust
let mut inner_vars_i = Vec::new();
Self::collect_free_vars(body_i, &mut inner_vars_i);
let param_names_i = ...;
inner_vars_i.retain(|v| !param_names_i.contains(v));
inner_vars_o.extend(inner_vars_i);  // 内层捕获加入外层 inner_vars
```

故 `captures(c_inner) = inner_vars_i \ p_i` 被加入 `inner_vars_o`。

随后 `inner_vars_o.retain(|v| !p_o.contains(v))` 移除 `p_o`。

故 `captures(c_inner) \ p_o ⊆ inner_vars_o \ p_o = captures(c_outer)`。✓

**捕获穿透的层次性**：对任意深度 `d` 的嵌套闭包 `c_1 ⊃ c_2 ⊃ ... ⊃ c_d`：

$$\text{captures}(c_d) \setminus \bigcup_{i=1}^{d-1} p_i \subseteq \text{captures}(c_1)$$

由归纳法可证（对 `d` 归纳，每层应用上述论证）。✓

**实例**：

```tenth
let f = |a| {              // c_outer, p_o = [a]
    let g = |b| {          // c_inner, p_i = [b]
        a + b + outer_var  // 引用 a (外层参数), b (内层参数), outer_var (最外层)
    };
    g
};
```

- `captures(c_inner) = {a, outer_var}`（`b` 被 `p_i` 移除）。
- `captures(c_outer) = captures(c_inner) \ {a} = {outer_var}`（`a` 被 `p_o` 移除）。
- 算法：`inner_vars_i = [a, b, outer_var]`，retain 移除 `b` → `[a, outer_var]`。`inner_vars_o = [a, outer_var]`，retain 移除 `a` → `[outer_var]`。✓ $\square$

### 4.5 定理 FV5（复杂度）

**定理 FV5（复杂度）**：设 `n` 为 HIR 表达式的节点数，`m` 为 `collect_free_vars` 过程中 `vars` 累积器的最大长度，`k` 为 distinct 变量名数。则：

$$\text{Time}(collect\_free\_vars) = O(n^2) \quad \text{（最坏情况）}$$

其中 `retain` 操作贡献 $O(n \cdot m)$，`sort + dedup` 贡献 $O(k \log k)$。空间复杂度为 $O(n)$。

**证明**：

**时间复杂度分析**：

1. **节点遍历**：每个 HIR 节点被访问一次。每个节点的 match 分派为 O(1)。共 O(n)。

2. **Var 节点处理**：`vars.push(name.clone())`。`String::clone` 为 O(|name|)。设最大名称长度为 `L`（常数），则每次 push 为 O(L) = O(1)。共 O(n)。

3. **Builtin 过滤**：Var 节点的 match 检查名称是否为 builtin。match 列表长度为常数（约 40 项），每次检查 O(1)。共 O(n)。

4. **Block 的 retain**：`vars.retain(|v| !bound.contains(v))`。
   - `retain` 遍历 `vars`：O(|vars|)。
   - 对每个 `v`，`bound.contains(v)` 为 O(|bound|)（线性搜索 Vec）。
   - 单次 Block 的 retain：O(|vars| × |bound|)。
   - 设第 `i` 个 Block 的 `vars` 长度为 `m_i`，`bound` 长度为 `b_i`。则该 Block 的 retain 为 O(m_i × b_i)。
   - 总 retain 代价：$\sum_i m_i \cdot b_i$。
   - 上界：`m_i ≤ m`（累积器最大长度），$\sum_i b_i \leq n$（每个 Let 绑定至多常数个变量，总 Let 数 ≤ n）。
   - 故总 retain 代价 ≤ $m \cdot \sum_i b_i \leq m \cdot n$。
   - 由于 `m ≤ n`（累积器长度不超过节点数），总 retain 代价 ≤ $n \cdot n = O(n^2)$。

5. **Closure 的 retain**：`inner_vars.retain(|v| !param_names.contains(v))`。
   - 单次 Closure：O(|inner_vars| × |param_names|)。
   - `|param_names|` 通常为常数（闭包参数数有限）。
   - 单次 Closure：O(|inner_vars|)。
   - 总 Closure retain 代价：$\sum_i |inner\_vars_i| \leq n$。共 O(n)。

6. **For 的 retain**：同 Closure，O(n)。

7. **sort + dedup**（在 `free_vars_in` 入口）：
   - `vars.sort()`：O(k log k)（Rust 的 sort 为 O(n log n)）。
   - `vars.dedup()`：O(k)。
   - 共 O(k log k) ≤ O(n log n)。

**总时间复杂度**：

$$T(n) = O(n) + O(n) + O(n) + O(n^2) + O(n) + O(n) + O(n \log n) = O(n^2)$$

**`retain` 的贡献**：如上分析，`retain` 操作（Block + Closure + For）总代价为 O(n·m)（其中 m 为累积器最大长度）。这是复杂度从 O(n) 退化为 O(n²) 的根源。

**空间复杂度**：

- `vars` 累积器：O(m) ≤ O(n)。
- `bound` 列表（Block 内）：O(b) ≤ O(n)。
- `inner_vars`（Closure/For 内）：O(|inner|) ≤ O(n)。
- 递归栈：O(depth) ≤ O(n)。
- 共 O(n)。

**与 HashSet 的对比**：

若使用 `HashSet<String>` 替代 `Vec<String>`：
- `insert`：O(1) 平均。
- `remove`：O(1) 平均。
- 总时间：O(n) 平均。
- 但 `HashSet` 不保持顺序，需额外排序若需有序输出。

Tenth 选择 `Vec<String>` 的工程动机：
- 实现简单，无哈希开销。
- 小规模数据（闭包体通常节点数 < 1000）下，Vec 的常数因子优于 HashSet。
- `sort + dedup` 仅在入口调用一次。

代价：渐近复杂度从 O(n) 退化为 O(n²)。$\square$

---

## 5. collect_free_vars 的逐一分析

### 5.1 Let 绑定的作用域处理

**源码**（[closures.rs:92-106](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/hir/lower/closures.rs)）：

```rust
HirExprKind::Block { stmts, final_expr } => {
    let mut bound = Vec::new();
    for s in stmts {
        if let HirStmtKind::Let { names, .. } = &s.kind {
            for name in names {
                bound.push(name.clone());
            }
        }
        Self::collect_free_vars_stmt(s, vars);
    }
    if let Some(e) = final_expr { Self::collect_free_vars(e, vars); }
    vars.retain(|v| !bound.contains(v));
}
```

**分析**：

- `bound` 在遍历 `stmts` **之前**预收集所有 Let 绑定的变量名。
- 遍历时，对每个语句调用 `collect_free_vars_stmt`，将引用的变量加入 `vars`。
- 遍历后，`retain` 移除 `bound` 中的变量。

**正确性**：在假设 A1 下，预收集 `bound` 不影响正确性——因为 A1 排除了"外层变量在 Let 之前被引用且与 Let 同名"的情况。若无 A1，则预收集会导致外层同名变量被错误移除（见定理 FV3 的反例）。

**设计权衡**：预收集 `bound` 简化了实现（无需在遍历时维护"已绑定集合"），但牺牲了对前置遮蔽引用的处理能力。

### 5.2 For 循环变量的处理

**源码**（[closures.rs:177-183](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/hir/lower/closures.rs)）：

```rust
HirStmtKind::For { var, iter, body } => {
    Self::collect_free_vars(iter, vars);
    let mut inner = Vec::new();
    Self::collect_free_vars_stmt(body, &mut inner);
    inner.retain(|v| v != var);
    vars.extend(inner);
}
```

**分析**：

- `iter` 的自由变量直接加入外层 `vars`（iter 在 For 作用域外求值）。
- `body` 的自由变量先收集到 `inner`，然后移除 `var`（循环变量），最后加入外层 `vars`。
- 使用独立的 `inner` 隔离 body 的作用域，确保 `var` 仅在 body 内有效。

**正确性**：循环变量 `var` 被 `retain` 移除，正确反映其仅在 body 内绑定。✓

**潜在问题**：若 `body` 是 `Loop{body: [Let(x, ...), Expr(Var(x))]}`（非 Block 包装），则 `Let(x)` 绑定的 `x` 未被跟踪，会被错误加入 `vars`（见局限 L4）。

### 5.3 Closure 参数遮蔽的处理

**源码**（[closures.rs:107-114](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/hir/lower/closures.rs)）：

```rust
HirExprKind::Closure { params, body, .. } => {
    let mut inner_vars = Vec::new();
    Self::collect_free_vars(body, &mut inner_vars);
    let param_names: Vec<String> = params.iter().map(|(n, _)| n.clone()).collect();
    inner_vars.retain(|v| !param_names.contains(v));
    vars.extend(inner_vars);
}
```

**分析**：

- 使用独立 `inner_vars` 隔离闭包 body 的作用域。
- 收集 body 的自由变量后，移除参数名（参数在 body 内绑定）。
- 将剩余的（真自由变量）加入外层 `vars`，成为外层的捕获。

**正确性**：参数遮蔽被正确处理——参数在 body 内绑定，不是闭包的自由变量。✓

**嵌套闭包的捕获穿透**：当 body 含内层闭包时，内层闭包的 `inner_vars` 被加入外层 `inner_vars`（经 `extend`），然后外层参数被 `retain` 移除。这正确实现了捕获穿透（见定理 FV4）。✓

### 5.4 Block 内绑定的块结束后移除

**源码**：同 5.1，`vars.retain(|v| !bound.contains(v))` 在 Block 遍历结束后执行。

**分析**：

- Block 内 Let 绑定的变量在块结束后"失效"——不应作为外层的自由变量。
- `retain` 在遍历结束后一次性移除所有 `bound` 变量。
- 这正确反映了 Block 的词法作用域语义：块内绑定不泄漏到块外。✓

**与 For/Closure 的对比**：Block 使用"先收集 bound，遍历后 retain"的模式；For/Closure 使用"独立 inner，遍历后 retain"。两种模式都正确，但 Block 的模式在 A1 假设下成立，For/Closure 的模式无条件成立（因为 For 的循环变量、Closure 的参数在遍历前已知）。

### 5.5 嵌套 Closure 的捕获穿透

已在定理 FV4 中详述。关键机制：

- 内层 Closure 的 `inner_vars`（经参数 retain 后）通过 `vars.extend` 加入外层的 `inner_vars`。
- 外层 Closure 的参数 retain 在最后执行，移除外层参数。
- 故内层捕获中属于外层参数的部分被移除，其余成为外层捕获。

这一机制对任意深度嵌套成立（定理 FV4 的归纳证明）。✓

### 5.6 Assign 目标的处理

**源码**（[closures.rs:115-124](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/hir/lower/closures.rs)）：

```rust
HirExprKind::Assign { target, value } => {
    vars.push(target.clone());
    Self::collect_free_vars(value, vars);
}
```

**分析**：

- `target`（赋值目标变量名）被无条件加入 `vars`。
- 这是因为赋值目标可能是外层变量（如闭包捕获后修改），需要被识别为自由变量。
- 若 `target` 是局部变量（在同一 Block 内 Let 绑定），则会被 Block 的 `retain` 移除。

**正确性**：在 Block 上下文中，局部赋值目标被 retain 移除；外层赋值目标保留为自由变量。✓

**潜在问题**：若 `Assign` 不在 Block 内（如直接作为闭包 body），`target` 不会被 retain 移除。但闭包 body 通常是 Block，故实践中不触发。若 body 直接是 `Assign`，则 `target` 可能被误报为自由——但这通常意味着 `target` 确实是闭包需捕获的变量（闭包内赋值需捕获可变引用）。✓（语义合理）

### 5.7 内置名称的过滤

**源码**（[closures.rs:22-35](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/hir/lower/closures.rs)）：

```rust
match name.as_str() {
    "println" | "eprintln" | "tensor" | "rand" | ... | "self" => {}
    _ => { vars.push(name.clone()); }
}
```

**分析**：

- 硬编码的 builtin 列表，约 40 项。
- 含 `::` 的名称（限定路径）直接跳过。
- builtin 名称不视为自由变量（无需捕获，运行时直接解析）。

**正确性**：builtin 名称在运行时由 VM/解释器直接解析，无需通过捕获传递。✓

**潜在问题**：
- 列表硬编码，新增 builtin 需同步更新此列表。维护负担。
- 若用户定义了与 builtin 同名的变量（如 `let tensor = 5;`），会被错误跳过。但这通常被词法分析或类型系统禁止。
- `Assign.target` 和 `InterpPart::Expr` 未过滤 builtin（见 L5）。

### 5.8 Match 分支的处理

**源码**（[closures.rs:131-134](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/hir/lower/closures.rs)）：

```rust
HirExprKind::Match { scrutinee, arms } => {
    Self::collect_free_vars(scrutinee, vars);
    for arm in arms { Self::collect_free_vars(&arm.body, vars); }
}
```

**分析**：

- `scrutinee` 的自由变量被收集。✓
- 每个 `arm.body` 的自由变量被收集。✓
- **但 `arm.guard` 未被处理**！若 arm 有守卫 `if x > 0`，守卫中的 `x` 不会被收集。这是**完备性漏洞**（L2）。
- **`arm.pattern` 绑定的变量未被移除**！若模式 `Some(x)` 绑定 `x`，body 中引用 `x` 会被错误收集为自由变量。这是**非绑定健全性漏洞**（L3）。

**实际影响**：Tenth 的 Match 若使用守卫或模式绑定，闭包捕获分析可能不准确。但实践中，闭包体内直接使用 Match with guard 的情况较少，且过近似（L3）不破坏语义（仅捕获冗余）。

---

## 6. 与 OCaml/Haskell/Rust 的对比

### 6.1 相似性

| 方面 | Tenth | OCaml | Haskell (GHC) | Rust |
|------|-------|-------|---------------|------|
| 递归遍历 | ✓ | ✓ | ✓ | 隐式（trait 推断） |
| 作用域处理 | ✓ | ✓ | ✓ | ✓ |
| 参数移除 | ✓ | ✓ | ✓ | ✓ |
| 嵌套闭包穿透 | ✓ | ✓ | ✓ | ✓ |
| 内置排除 | 硬编码列表 | 预定义集 | 预定义集 | 预定义集 |

### 6.2 差异性

| 方面 | Tenth | OCaml | Haskell (GHC) | Rust |
|------|-------|-------|---------------|------|
| 数据结构 | `Vec<String>` | `Set.t` | `VarSet` (IntMap) | trait 推断 |
| 去重方式 | `sort + dedup` | 集合语义 | 集合语义 | N/A |
| 移除方式 | `retain` (线性) | 集合差 (O(log n)) | 集合差 (O(1)) | N/A |
| 复杂度 | O(n²) | O(n log n) | O(n) | N/A |
| 顺序绑定 | 预收集 bound | 增量维护 | 增量维护 | 增量维护 |
| Match 模式 | **未处理** | ✓ | ✓ | ✓ |
| Match 守卫 | **未处理** | ✓ | ✓ | ✓ |

### 6.3 复杂度对比

| 实现 | 时间复杂度 | 空间复杂度 | 备注 |
|------|-----------|-----------|------|
| Tenth (`Vec + retain`) | O(n²) | O(n) | `retain` 线性搜索 |
| OCaml (`Set.t`) | O(n log n) | O(n) | 平衡树 |
| Haskell (`VarSet`) | O(n) | O(n) | IntMap，O(1) 增删 |
| 理论最优 (`HashSet`) | O(n) 平均 | O(n) | 哈希 |

Tenth 的 O(n²) 是四者中最差的。但闭包体规模通常 < 1000 节点，O(n²) 在实践中可接受（< 1ms）。

### 6.4 表达力对比

Tenth 的 `collect_free_vars` 不支持：

- **Match 模式绑定**的变量移除（L3）。
- **Match 守卫**的变量收集（L2）。
- **前置遮蔽引用**的正确处理（L1）。

OCaml/Haskell/Rust 均支持上述场景。这是 Tenth 实现简化带来的表达力损失。

---

## 7. 局限（诚实披露）

### 7.1 局限 L1（Block 前置遮蔽引用）

**现象**：若 `Block` 内某 `Let` 绑定变量 `v`，且在该 `Let` 之前的语句中引用了外层同名变量 `v`，算法会将该外层 `v` 错误移除。

**反例**：

```tenth
// 假设外层有 let v = 10;
let block = {
    println(v);   // 引用外层 v
    let v = 20;   // 遮蔽
    v + 1
};
```

- **语义 FV**：`{v, println}` → 排除 builtin → `{v}`。
- **算法 CFV**：`bound = [v]`。`println(v)` 将 `v` 加入 vars。`retain` 移除 `v`。CFV = `{}`。
- **后果**：`v` 未被识别为自由变量。若此块在闭包内，`v` 不会被捕获，运行时访问未捕获变量。

**严重性**：**高**（若触发，导致运行时错误）。但实践中，前置遮蔽引用较少见（编程习惯通常先 Let 后引用）。

**根源**：`bound` 在遍历前预收集，未区分"Let 之前"与"Let 之后"。

**缓解**：改为增量维护 `bound`——在遍历到 `Let` 语句时才将 `names` 加入 `bound`。但此修改需重构 Block 的处理逻辑。

**现状**：未修复。依赖 A1 假设在实践中成立。

### 7.2 局限 L2（Match 守卫未处理）

**现象**：`Match` 分支的 `guard` 字段未被 `collect_free_vars` 处理，守卫中的变量引用被遗漏。

**反例**：

```tenth
let f = |x| {
    match x {
        Some(v) if v > threshold => v,  // threshold 在守卫中
        _ => 0
    }
};
```

- **语义 FV**：`{threshold}`（守卫引用 threshold）。
- **算法 CFV**：守卫未处理，CFV = `{}`。
- **后果**：`threshold` 未被捕获，运行时守卫求值失败。

**严重性**：**中**（仅影响使用守卫的闭包）。

**根源**：[closures.rs:131-134](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/hir/lower/closures.rs) 的 Match 分支未调用 `collect_free_vars` 对 `arm.guard`。

**缓解**：在 Match 分支添加 `if let Some(g) = &arm.guard { Self::collect_free_vars(g, vars); }`。

**现状**：未修复。

### 7.3 局限 L3（Match 模式绑定未移除）

**现象**：`Match` 分支的 `pattern` 绑定的变量未被 `collect_free_vars` 移除，body 中引用模式绑定变量会被错误收集为自由变量。

**反例**：

```tenth
let f = |opt| {
    match opt {
        Some(v) => v + 1,  // v 是模式绑定
        None => 0
    }
};
```

- **语义 FV**：`{}`（`v` 是模式绑定，非自由）。
- **算法 CFV**：`v` 被收集为自由变量。CFV = `{v}`。
- **后果**：`v` 被错误捕获。闭包运行时尝试捕获不存在的 `v`，可能导致错误或捕获冗余。

**严重性**：**中**（过近似，不破坏语义但浪费内存）。但若 `v` 在外层不存在，捕获会失败。

**根源**：[closures.rs:131-134](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/hir/lower/closures.rs) 未提取 `arm.pattern` 的绑定变量并 retain 移除。

**缓解**：实现 `pattern_binders(pattern) -> Vec<String>`，在 Match 分支对每个 arm 的 body 使用独立 inner_vars 并 retain 移除 pattern_binders。

**现状**：未修复。

### 7.4 局限 L4（While/Loop 内 Let 未跟踪）

**现象**：`While`、`Loop` 的 body 若直接含 `Let` 语句（非经 Block 包装），算法不跟踪这些 Let 绑定，导致 Let 绑定的变量被错误收集为自由变量。

**反例**：

```tenth
let f = |xs| {
    loop {
        let x = 10;       // Loop 内 Let
        break x + outer   // outer 是真自由变量
    }
};
```

- **语义 FV**：`{outer}`（`x` 在 Loop 内绑定）。
- **算法 CFV**：`Loop` 分支遍历 body，`Let(x)` 的 init 被处理，但 `x` 未加入任何 `bound`。`break x + outer` 将 `x` 和 `outer` 加入 vars。CFV = `{x, outer}`。
- **后果**：`x` 被错误捕获。

**严重性**：**中**（过近似，不破坏语义但捕获冗余）。

**根源**：[closures.rs:185-187](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/hir/lower/closures.rs) 的 `Loop` 分支和 [closures.rs:173-176](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/hir/lower/closures.rs) 的 `While` 分支未跟踪 body 中的 Let 绑定。

**缓解**：为 `While`、`Loop` 添加类似 `Block` 的 `bound` 跟踪，或要求 body 必须是 Block。

**现状**：未修复。依赖 A4 假设（body 为 Block 包装）在实践中成立。

### 7.5 局限 L5（Assign/InterpPart 未过滤 builtin）

**现象**：`Assign.target` 和 `InterpPart::Expr(name)` 未进行 builtin 过滤，若 `target` 或 `name` 是 builtin 名称，会被错误收集。

**严重性**：**极低**（builtin 名称不能作为赋值目标或插值变量，语义限制保证不触发）。

**根源**：[closures.rs:115-124](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/hir/lower/closures.rs) 和 [closures.rs:141-147](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/hir/lower/closures.rs) 未调用 builtin 过滤。

**缓解**：对 `target` 和 `name` 添加 builtin 检查。

**现状**：未修复，但实际不触发。

### 7.6 局限汇总表

| 局限 | 类型 | 严重性 | 是否触发 | 根源 |
|------|------|--------|---------|------|
| L1 | 完备性 | 高 | 实践少 | Block 预收集 bound |
| L2 | 完备性 | 中 | 守卫场景 | Match 未处理 guard |
| L3 | 健全性 | 中 | 模式绑定 | Match 未移除 pattern binders |
| L4 | 健全性 | 中 | Loop/While 直接含 Let | 未跟踪 Let |
| L5 | 健全性 | 极低 | 不触发 | Assign/Interp 未过滤 builtin |

---

## 8. 工程实现分析

### 8.1 Vec<String> + retain 的效率

**当前实现**：`Vec<String>` 作为累积器，`retain` 进行批量移除。

**优势**：
- 实现简单，无哈希开销。
- 小规模数据下常数因子小。
- `sort + dedup` 一次去重，结果有序（便于后续处理）。

**劣势**：
- `retain` 中的 `contains` 为线性搜索，O(|bound|)。
- 多次 `retain` 的总代价为 O(n·m)。
- 渐近复杂度 O(n²)，劣于 HashSet 的 O(n)。

**实测预估**：闭包体规模通常 < 1000 节点。O(n²) = O(10⁶)，在现代 CPU 上约 1ms。可接受。

### 8.2 去重的正确性

**当前实现**：`vars.sort()` 后 `vars.dedup()`。

**正确性**：`sort` 将相同元素相邻排列，`dedup` 移除连续重复。这是 Rust 标准库的惯用模式，正确去重。✓

**替代方案**：`HashSet` 天然去重，无需 sort+dedup。但需在最后转为有序 Vec（若需有序输出）。

### 8.3 潜在的优化（未来工作）

**优化 1：HashSet 替代 Vec**

将 `vars: &mut Vec<String>` 改为 `vars: &mut HashSet<String>`：
- `insert`：O(1) 平均。
- `remove`：O(1) 平均。
- 总复杂度：O(n) 平均。
- 代价：哈希开销（小规模下可能更慢）、无序输出（需排序若需有序）。

**优化 2：增量维护 bound**

修复 L1，将 Block 的 `bound` 改为增量维护：
- 遍历 `stmts` 时，遇到 `Let` 才将 `names` 加入 `bound`。
- 在 `Let` 之前引用的同名变量不被加入 `bound`，不被 `retain` 移除。

**优化 3：Match 模式与守卫处理**

修复 L2、L3：
- 实现 `pattern_binders(pattern) -> Vec<String>`。
- 在 Match 分支对每个 arm 的 body 使用独立 inner_vars，retain 移除 pattern_binders。
- 添加 `arm.guard` 的 `collect_free_vars` 调用。

**优化 4：While/Loop 的 Let 跟踪**

修复 L4：
- 为 `While`、`Loop` 添加 `bound` 跟踪，类似 `Block`。
- 或在 lowering 阶段强制要求 While/Loop body 为 Block。

**以上优化均未实现，标注为未来工作。**

---

## 9. 开放问题与未来工作

### 9.1 set-level analysis 的引入

Haskell GHC 的 set-level analysis 不仅分析自由变量，还分析闭包的分配级别，优化堆栈分配决策。Tenth 当前仅做基本的自由变量分析，未涉及分配级别优化。

**未来工作**：研究 set-level analysis 在 Tenth 中的应用，特别是在 AI 原生场景（张量计算、自动微分）中，闭包的分配级别可能影响性能。

### 9.2 机器验证的正确性

本文的证明为纸笔证明，未经机器验证。RustBelt [Jung et al., 2018] 使用 Iris 分离逻辑对 Rust 类型系统进行了机器验证。

**未来工作**：考虑使用 Lean 或 Coq 对 `collect_free_vars` 的正确性进行机器验证，特别是 FV1–FV4 的归纳证明。

### 9.3 局限修复的优先级

- **L1（高严重性）**：应优先修复，改为增量维护 bound。
- **L2、L3（中严重性）**：应修复 Match 守卫与模式绑定处理。
- **L4（中严重性）**：可强制 body 为 Block，或为 While/Loop 添加 bound 跟踪。
- **L5（极低严重性）**：可不修复，依赖语义保证。

### 9.4 性能优化的必要性

当前 O(n²) 在闭包体规模 < 1000 节点时可接受。若未来 Tenth 支持更大规模的闭包（如生成的 DSL 代码），应考虑 HashSet 优化。

---

## 10. 结论

本文对 Tenth 语言的闭包自由变量分析器 `collect_free_vars` 进行了形式化建模与正确性证明。核心结论：

1. **引用健全性（FV1）**无条件成立：每个被收集的变量确实在表达式内被引用（modulo L5 的理论瑕疵，实际不触发）。
2. **受控完备性（FV2）**在四条假设（A1–A4）下成立：所有真自由变量被收集。
3. **Block 顺序绑定（FV3）**在 A1 下正确：多 Let 顺序绑定的自由变量正确处理。
4. **嵌套闭包捕获穿透（FV4）**无条件成立：内层闭包的捕获正确传播至外层。
5. **复杂度（FV5）**为 O(n²)，其中 `retain` 贡献 O(n·m)，劣于 HashSet 的 O(n)。

**诚实披露的局限**：
- L1：Block 前置遮蔽引用导致完备性失败（高严重性，实践少）。
- L2：Match 守卫未处理导致完备性失败（中严重性）。
- L3：Match 模式绑定未移除导致非绑定健全性失败（中严重性，过近似）。
- L4：While/Loop 内 Let 未跟踪导致非绑定健全性失败（中严重性，过近似）。
- L5：Assign/InterpPart 未过滤 builtin（极低严重性，不触发）。

**与 OCaml/Haskell/Rust 的对比**：Tenth 的递归遍历结构与经典方法一致，但 `Vec<String> + retain` 的实现选择导致 O(n²) 复杂度，劣于 OCaml 的 O(n log n) 和 Haskell 的 O(n)。Match 模式与守卫的处理缺失是表达力损失。

**工程价值**：尽管存在局限，`collect_free_vars` 在 Tenth 的典型使用场景（闭包体规模 < 1000 节点、无复杂 Match 模式、无前置遮蔽引用）下足够正确且高效。其简单性（190 行 Rust）适合 AI 原生语言的快速迭代需求。

---

## 参考文献

1. Appel, A. W. (1992). *Compiling with Continuations*. Cambridge University Press.
2. Leroy, X. (1992). *The ZINC experiment: an economical implementation of the ML language*. INRIA Technical Report 117.
3. Marlow, S., & Peyton Jones, S. (2004). *Making a fast curry: push/enter vs. eval/apply for higher-order languages*. ICFP.
4. Jung, R., Jourdan, J.-H., Krebbers, R., & Dreyer, D. (2018). *RustBelt: Securing the foundations of the Rust programming language*. POPL.
5. Jung, R., Dang, H.-H., Kang, J., & Dreyer, D. (2017). *Stacked borrows: an aliasing model for Rust*. POPL.
6. Rust Reference. (2024). *Closure types: Fn, FnMut, FnOnce*. https://doc.rust-lang.org/reference/types/closure.html
7. Tenth 项目. (2026). *HIR 数据结构定义*. [tenth/src/hir/hir.rs](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/hir/hir.rs)
8. Tenth 项目. (2026). *collect_free_vars 实现*. [tenth/src/hir/lower/closures.rs](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/hir/lower/closures.rs)
9. Tenth 项目. (2026). *闭包转换调用点*. [tenth/src/hir/lower/lower_expr.rs:511](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/hir/lower/lower_expr.rs)
10. Tenth 项目. (2026). *工作规范 v1.1*. `.trae/rules/工作规范.md`

---

## 附录 A：定理索引

| 定理 | 名称 | 结论 | 证明完整性 |
|------|------|------|-----------|
| FV1 | 引用健全性 | 收集的变量均被引用 | 完整（modulo L5） |
| FV2 | 受控完备性 | A1–A4 下完备 | 完整（含假设声明） |
| FV3 | Block 顺序绑定 | A1 下正确 | 完整（含反例） |
| FV4 | 嵌套闭包捕获穿透 | 无条件正确 | 完整 |
| FV5 | 复杂度 | O(n²) | 完整 |

## 附录 B：局限索引

| 局限 | 类型 | 严重性 | 触发条件 | 缓解 |
|------|------|--------|---------|------|
| L1 | 完备性 | 高 | Block 前置遮蔽引用 | 增量维护 bound |
| L2 | 完备性 | 中 | Match 守卫 | 添加 guard 处理 |
| L3 | 健全性 | 中 | Match 模式绑定 | 添加 pattern_binders 移除 |
| L4 | 健全性 | 中 | While/Loop 直接含 Let | 添加 bound 跟踪 |
| L5 | 健全性 | 极低 | Assign/InterpPart 是 builtin | 添加 builtin 过滤 |

## 附录 C：实施建议

基于本文分析，对 Tenth 编译器部的实施建议（按优先级）：

1. **P0（高优先级）**：修复 L1。将 Block 的 `bound` 改为增量维护：
   ```rust
   for s in stmts {
       Self::collect_free_vars_stmt(s, vars);
       if let HirStmtKind::Let { names, .. } = &s.kind {
           for name in names { bound.push(name.clone()); }
       }
   }
   ```
   （先处理语句，后加入 bound——这样 Let 之前的引用不会被 retain 移除。）

2. **P1（中优先级）**：修复 L2、L3。在 Match 分支：
   ```rust
   for arm in arms {
       let mut inner = Vec::new();
       if let Some(g) = &arm.guard { Self::collect_free_vars(g, &mut inner); }
       Self::collect_free_vars(&arm.body, &mut inner);
       let pb = pattern_binders(&arm.pattern);
       inner.retain(|v| !pb.contains(v));
       vars.extend(inner);
   }
   ```

3. **P2（中优先级）**：修复 L4。为 While/Loop 添加 bound 跟踪，或强制 body 为 Block。

4. **P3（低优先级）**：考虑 HashSet 优化以将复杂度降至 O(n)。

5. **P4（低优先级）**：修复 L5（Assign/InterpPart 的 builtin 过滤）。

**注意**：以上均为未来工作，本文未实施任何代码修改。

---

> **数理部声明**：本文的理论结论基于对 [tenth/src/hir/lower/closures.rs](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/hir/lower/closures.rs)（v0.3.3+）的源码分析。所有源码引用均使用 `file://` 链接标注。局限章节诚实披露了证明的漏洞与假设的强度，未掩盖任何已知问题。实施建议附录将理论结论转化为可执行指导，但未实施任何代码修改——实施由编译器部负责。
