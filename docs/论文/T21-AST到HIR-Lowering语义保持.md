# AST→HIR Lowering 的语义保持：信息有损变换的 Translation Validation 框架

> **论文编号**：T21 | **系列**：Tenth 编译管线语义保持 | **版本**：v1.0 | **日期**：2026-07-02
> **部门**：数理部 | **对应实现**：Tenth v0.3.3+ | **审查轮次**：v1（结构→证明→边界→诚实）
> **联动**：T16（双向类型重建）、T12（双侧编译器等价性）、T19（语句粒度借用检查）

---

## 摘要

Tenth 语言的 AST→HIR lowering 不是单纯的"加类型注解"，而是一组**信息有损变换**：HIR 的 `Assign` target 从 `Box<Expr>` 收紧为 `String`，`Index` 从 `Vec<IndexExpr>` 调整为 `Vec<Index>`（结构同构但类型重命名），`Match` 的 pattern 表示从 `tuple_fields: Vec<String>` 改写为 `tuple_binds: Vec<(String, String)>`（信息增加而非损失），以及若干静默丢弃（如 `StructLiteral` 的 `generics` 字段、`is_pub` 可见性修饰符）。这意味着 HIR 的表达力是 AST 的**真子集**——存在 AST 合法但 HIR 无法表示的程序。本文对这一 lowering 进行形式化分析，建立基于操作语义的 Translation Validation 框架，给出五个主定理：(L1) 在 HIR 可表示子集上 lowering 保持语义；(L2) 拒绝的正确性——所有显式拒绝（`?` 传播 `ParseError`/`TypeError`）均对应 HIR 表达力不足的明确情形；(L3) HIR 表达力子集的形式刻画；(L4) **静默丢失漏洞的实证分析**——发现 2 处真实的静默信息丢失（`StructLiteral.generics` 丢弃、`is_pub` 可见性丢弃），均不改变运行时语义但削弱编译期保证；(L5) Translation Validation 的可验证条件。所有结论均附源码位置引用。

**关键词**：Translation Validation；语义保持；信息有损变换；AST→HIR Lowering；结构归纳；表达力子集；Tenth 语言；编译器验证

---

## 1. 引言

### 1.1 Lowering 在编译器中的角色

编译器的 **lowering**（降级）阶段将高层中间表示（high-level IR）转换为低层中间表示（low-level IR），通常伴随抽象层级的降低：树状结构变图、隐式控制流变显式、语法糖脱糖。在 Tenth 编译管线中（[CODE_WIKI.md](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/CODE_WIKI.md)），lowering 指 **AST → HIR** 的转换：

```
.th → Lexer → Parser → AST → [Lowering] → HIR → VM/Interpreter/WASM/JIT
```

AST（[tenth/src/parser/ast.rs](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/parser/ast.rs)）是 parser 直接产生的具体语法树，保留所有源码字面信息；HIR（[tenth/src/hir/hir.rs](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/hir/hir.rs)）是带类型注解的、面向后端（VM/WASM/JIT）的中间表示。

### 1.2 AST→HIR 的信息有损变换挑战

与"加类型注解"的朴素观点不同，Tenth 的 AST→HIR lowering 包含**真信息有损变换**：

1. **Assign target 收紧**：AST 的 `Assign { target: Box<Expr>, value: Box<Expr> }`（[ast.rs:117-120](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/parser/ast.rs)）允许任意表达式作为赋值目标；HIR 的 `Assign { target: String, value: Box<HirExpr> }`（[hir.rs:79-82](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/hir/hir.rs)）仅允许变量名。HIR 通过引入 `DerefAssign`、`FieldAssign`、`DerefAssignOp` 三个独立变体部分补偿，但**不引入** `IndexAssign`、`FieldAssignOp`——即 `x[i] = v`、`s.f += v` 等 AST 合法程序无法 lowering。

2. **静默丢弃**：`StructLiteral` 的 `generics` 字段在 lowering 中被 `generics: _` 显式丢弃（[lower_expr.rs:571](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/hir/lower/lower_expr.rs)）；`is_pub` 可见性修饰符在 `Function`/`StructDef`/`Impl` 项 lowering 中未被保留到 HIR（[ast.rs:246, 261, 272](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/parser/ast.rs) vs [hir.rs:226-234](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/hir/hir.rs)）。

3. **Index 类型重命名**：AST `Vec<IndexExpr>` → HIR `Vec<Index>`（[ast.rs:185-192](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/parser/ast.rs) vs [hir.rs:184-191](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/hir/hir.rs)），两者结构同构（`Single`/`Range`/`Colon` 三变体完全对应），**不是信息损失**——但常被误判为有损。

这意味着 HIR 的表达力是 AST 的真子集。Lowering 必须**拒绝** AST 中合法但 HIR 无法表示的程序（通过 `?` 传播 `ParseError`/`TypeError`），同时**保留** HIR 能表示的子集的语义。问题是：**当前实现是否在所有 HIR 可表示的子集上都保持语义？是否所有拒绝都对应明确的 HIR 表达力不足？是否存在"AST 合法 → HIR 静默丢失信息"的漏洞？**

### 1.3 Translation Validation 的概念

**Translation Validation**（翻译验证，Pnueli-Siegel-Singh [1]）是一种**事后验证**方法：不证明翻译器（compiler/transformer）正确，而是对**每一次具体翻译**的输入输出进行等价性检查。与翻译器正确性证明（如 CompCert [2]）相比，translation validation 更轻量、可增量部署，但要求等价检查器本身可信。

在 Tenth AST→HIR 场景下，translation validation 表现为：对每个 AST 节点 $a$ 与对应的 HIR 节点 $h = \text{lower}(a)$，验证一个**可验证条件**（verification condition, VC）$\Phi(a, h)$，若 $\Phi$ 成立则 $a$ 与 $h$ 语义等价。本文将形式化 $\Phi$ 并证明其充分性。

### 1.4 贡献

本文贡献如下：

1. **AST 与 HIR 的形式化**（§3）：将两套数据结构抽象为代数数据类型，逐一分析差异，给出**信息有损变换点的完整列表**（共 7 处，含 2 处静默丢失、3 处显式拒绝、2 处结构同构）。
2. **操作语义**（§4）：为 AST 与 HIR 分别建立小步操作语义，定义语义等价关系 $\sim$。
3. **五个主定理与证明**（§5）：
   - L1 语义保持（在 HIR 可表示子集上）
   - L2 拒绝的正确性
   - L3 HIR 表达力子集的形式刻画
   - L4 静默丢失漏洞的实证分析
   - L5 Translation Validation 框架（可验证条件的充分性）
4. **工程实现分析**（§7）：审查 `?` 错误传播的覆盖完整性，定位所有拒绝点。
5. **诚实局限披露**（§9）：未机器验证、操作语义未涵盖所有 HIR 节点、与运行时语义的关系。

---

## 2. 背景与相关工作

### 2.1 Translation Validation（Pnueli-Siegel-Singh）

Pnueli、Siegel 与 Singh [1] 于 1998 年提出 translation validation：不证明编译器整体正确，而是为每次编译运行生成**可验证条件**，由独立的 validator 检查源程序与目标程序等价。其优势：

- **增量部署**：编译器可逐步演化，validator 独立维护；
- **可处理优化**：per-run 验证能覆盖编译器版本更迭；
- **可证伪**：validator 拒绝即编译错误，便于调试。

其局限：validator 本身需可信；对大规模程序，VC 求解可能代价高。

### 2.2 CompCert 的语义保持验证

Leroy 等 [2] 的 CompCert 项目用 Coq 形式化验证 C 编译器从 AST 到汇编的语义保持。CompCert 采用**分阶段证明**：每条编译 pass 都证明前后语义等价（在 observable behaviors 上）。CompCert 的强度在于**翻译器正确性**（compiler correctness），而非 per-run validation。

Tenth 的 AST→HIR 与 CompCert 的 Clight→Cminor 阶段类似，但 Tenth 不做机器证明，而是用 translation validation 框架在工程层验证。

### 2.3 LLVM IR 的 Translation Validation

LLVM 项目 [3] 采用 translation validation 验证优化 pass 的语义保持：每个 pass 后运行 `llvm-diff` 或基于 SMT 的 validator。Necula [4] 的 CCured 项目也采用类似思路。

### 2.4 Rust 的 HIR/MIR Lowering

Rustc [5] 维护多级 IR：AST → HIR → MIR → LLVM IR。HIR 阶段做类型检查与名字解析；MIR 阶段做借用检查与优化。Rustc 的 lowering 用**结构化遍历**（`rustc_hir`）保证字段不丢失——每个 AST 节点都有对应的 HIR 节点，HIR 字段是 AST 字段的"子集 + 类型注解"。Rustc 的 HIR 通过 `HirId` 系统保留 span 与可见性信息，避免静默丢失。

Tenth 的 HIR 设计更接近 Rustc 的 MIR——简化为后端友好的形式，但牺牲了 Rustc 的 `HirId` 系统，导致部分信息（如 `is_pub`）静默丢失（详见 §6.2）。

### 2.5 T16 与 T19 的联动

- **T16**（[T16-双向类型重建.md](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/docs/论文/T16-双向类型重建.md)）证明 lowering 阶段的类型重建满足 Subject Reduction。本文 L1 定理依赖 T16 的类型正确性作为前提：若类型重建错误，HIR 的 `ty` 字段可能误导后端，但语义保持证明不依赖 `ty` 的精确性（仅依赖结构等价）。
- **T19**（[T19-语句粒度借用检查.md](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/docs/论文/T19-语句粒度借用检查.md)）证明 lowering 中的借用检查（`scope.check_borrow_shared/mut`，[lower_expr.rs:701, 717](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/hir/lower/lower_expr.rs)）满足语句级安全性。本文 L1 不直接依赖 T19，但二者共同构成 lowering 的正确性保证。

---

## 3. AST 与 HIR 的形式化

### 3.1 AST 的代数数据类型定义

**定义 3.1（AST 表达式）**。AST 表达式 $\text{Expr}_A$ 由如下代数数据类型定义（对应 [ast.rs:64-146](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/parser/ast.rs)）：

$$
e_A \in \text{Expr}_A ::= \text{Lit}(l) \mid \text{Var}(x) \mid \text{Bin}(op, e_1, e_2) \mid \text{Un}(op, e) \mid \text{Call}(e_f, \vec{e}) \mid \text{GCall}(e_f, \vec{\tau}_A, \vec{e}) \mid \text{MCall}(e_r, m, \vec{e}) \mid \text{Idx}(e_t, \vec{ix}_A) \mid \text{Fld}(e_t, f) \mid \text{TLit}(\vec{\vec{e}}) \mid \text{ALit}(\vec{e}) \mid \text{Range}(e_s?, e_e?, b) \mid \text{If}(e_c, e_t, e_e?) \mid \text{Block}(\vec{s}) \mid \text{Clos}(\vec{(x, \tau_A?)}, e) \mid \text{Assign}(e_{tgt}, e_v) \mid \text{AssignOp}(e_{tgt}, op, e_v) \mid \text{SLit}(n, \vec{\tau}_A, \vec{(f, e)}, b) \mid \text{ELit}(n, v, \vec{(f, e)}) \mid \text{Match}(e_s, \vec{arm}) \mid \text{Ref}(e) \mid \text{MutRef}(e) \mid \text{Deref}(e) \mid \text{Move}(e) \mid \text{Try}(e) \mid \text{IStr}(\vec{p}) \mid \text{Tup}(\vec{e})
$$

其中 $l$ 为字面量，$x, f, m, n, v$ 为标识符，$op$ 为二元/一元算子，$\tau_A$ 为类型注解，$ix_A$ 为索引表达式，$s$ 为语句，$arm$ 为 match 臂，$p$ 为插值字符串片段。

**关键观察**：$\text{Assign}$ 的 target $e_{tgt}$ 是任意 $\text{Expr}_A$——`x`、`*p`、`s.f`、`x[i]`、甚至 `f().g` 都是合法 AST。

**定义 3.2（AST 索引表达式）**。$\text{Ix}_A ::= \text{Single}(e) \mid \text{Range}(e_s?, e_e?) \mid \text{Colon}$（[ast.rs:185-192](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/parser/ast.rs)）。

**定义 3.3（AST 模式）**。$\text{Pat}_A ::= \text{EV}(n, v, fb?, \vec{x}) \mid \text{Wild} \mid \text{Lit}(l) \mid \text{Tup}(\vec{p}) \mid \text{Range}(k_1, k_2, b) \mid \text{Bind}(x) \mid \text{Struct}(n, \vec{(f, x)})$（[ast.rs:156-176](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/parser/ast.rs)）。

### 3.2 HIR 的代数数据类型定义

**定义 3.4（HIR 表达式）**。HIR 表达式 $\text{Expr}_H$ 由如下代数数据类型定义（对应 [hir.rs:12-123](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/hir/hir.rs)）：

$$
e_H \in \text{Expr}_H ::= \text{Lit}(l) \mid \text{Var}(x) \mid \text{Bin}(op, e_1, e_2, \tau) \mid \text{Un}(op, e, \tau) \mid \text{Call}(e_f, \vec{e}, \tau) \mid \text{GCall}(e_f, \vec{\tau}, \vec{e}, \tau) \mid \text{MCall}(e_r, m, \vec{e}, \tau) \mid \text{Idx}(e_t, \vec{ix}_H) \mid \text{Fld}(e_t, f) \mid \text{TLit}(\vec{\vec{e}}, \tau) \mid \text{ALit}(\vec{e}, \tau) \mid \text{Range}(e_s?, e_e?, b) \mid \text{If}(e_c, e_t, e_e?, \tau) \mid \text{Block}(\vec{s}, e?) \mid \text{Clos}(\vec{(x, \tau)}, e, \vec{x}) \mid \text{Assign}(x, e) \mid \text{AssignOp}(x, op, e) \mid \text{SLit}(n, \vec{(f, e)}, b) \mid \text{ELit}(n, v, \vec{(f, e)}) \mid \text{Match}(e_s, \vec{arm}_H) \mid \text{Ref}(e) \mid \text{MutRef}(e) \mid \text{Deref}(e) \mid \text{DerefAssign}(e, e) \mid \text{DerefAssignOp}(e, op, e) \mid \text{Move}(e) \mid \text{Try}(e) \mid \text{IStr}(\vec{p}) \mid \text{Tup}(\vec{e}) \mid \text{FldAssign}(e, f, e)
$$

**关键差异**：

- $\text{Assign}$ 的 target 是 $x$（字符串），不是任意 $e_H$；
- HIR 新增 $\text{DerefAssign}$、$\text{DerefAssignOp}$、$\text{FldAssign}$ 三个变体，部分补偿 Assign target 的收紧；
- **HIR 不含** $\text{IndexAssign}$、$\text{FieldAssignOp}$ 变体——这两类赋值无法表示；
- HIR 的 $\text{SLit}$（StructLiteral）**不含** $\vec{\tau}_A$（generics 被丢弃）。

**定义 3.5（HIR 索引）**。$\text{Ix}_H ::= \text{Single}(e_H) \mid \text{Range}(e_H?, e_H?) \mid \text{Colon}$（[hir.rs:184-191](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/hir/hir.rs)）。

**定义 3.6（HIR 模式）**。$\text{Pat}_H ::= \text{EV}(n, v, fb?, \vec{(f, x)}) \mid \text{Wild} \mid \text{Lit}(l) \mid \text{Tup}(\vec{p}_H) \mid \text{Range}(k_1, k_2, b) \mid \text{Bind}(x) \mid \text{Struct}(n, \vec{(f, x)})$（[hir.rs:133-153](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/hir/hir.rs)）。

### 3.3 差异的逐一分析

下表总结 AST 与 HIR 的关键差异（**信息有损变换点的完整列表**）：

| # | 变换点 | AST 表示 | HIR 表示 | 类型 | 源码位置 |
|---|--------|---------|---------|------|---------|
| D1 | Assign target | `Box<Expr>` | `String` | 显式拒绝 | [lower_expr.rs:520-548](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/hir/lower/lower_expr.rs) |
| D2 | AssignOp target | `Box<Expr>` | `String`（无 FldAssignOp） | 显式拒绝 | [lower_expr.rs:550-569](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/hir/lower/lower_expr.rs) |
| D3 | StructLiteral generics | `Vec<TypeAnnotation>` | **丢弃**（`generics: _`） | **静默丢失** | [lower_expr.rs:571](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/hir/lower/lower_expr.rs) |
| D4 | is_pub 可见性 | `bool`（Function/StructDef/Impl） | **丢弃**（HIR 无对应字段） | **静默丢失** | [ast.rs:246,261,272](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/parser/ast.rs) vs [hir.rs:226-234](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/hir/hir.rs) |
| D5 | Ident → String | `Ident { name, span }` | `String` | 信息缩减（span 丢失） | 全局 |
| D6 | Index 类型重命名 | `Vec<IndexExpr>` | `Vec<Index>` | **结构同构**（无信息损失） | [lower_expr.rs:772-782](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/hir/lower/lower_expr.rs) |
| D7 | Match tuple_fields → tuple_binds | `Vec<String>` | `Vec<(String, String)>`（合成字段名 `_i`） | **信息增加** | [lower_expr.rs:786-794](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/hir/lower/lower_expr.rs) |
| D8 | GenericCall 重写 | `GenericCall{func, generics, args}` | `Call{func: Var(mangled_name), args, ret_ty}` | 部分损失（类型参数编码到 mangling 字符串中） | [lower_expr.rs:237-307](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/hir/lower/lower_expr.rs) |

**注**：任务描述提及"Match arms 顺序与 pattern 表示变化"——经源码审查，**arms 顺序在 AST 与 HIR 中均按源码顺序保留**（无重排），**pattern 表示变化仅为 D7（tuple_fields → tuple_binds，信息增加）**，不存在信息损失。任务描述的预期与实际源码存在偏差，本文以源码为准。

### 3.4 信息有损变换点的分类

**定义 3.7（变换分类）**。对每个变换点 $D_i$，分类为：

- **显式拒绝**（rejection）：lowering 在该点返回 `Err(TenthError)`，AST 程序被拒绝——D1、D2；
- **静默丢失**（silent loss）：lowering 接受 AST 但丢弃部分信息，HIR 语义弱于 AST——D3、D4、D5（span）、D8（部分）；
- **结构同构**（isomorphism）：AST 与 HIR 数据结构一一对应，无信息损失——D6；
- **信息增加**（enrichment）：HIR 比 AST 携带更多信息——D7。

**关键发现**：任务描述的核心担忧——"是否存在 AST 合法 → HIR 静默丢失信息的漏洞"——的答案是**肯定的**，存在 2 处静默丢失（D3、D4）。但经分析，这 2 处静默丢失**均不改变运行时可观察行为**（详见 §6.2、§6.3），仅削弱编译期保证（如可见性检查、类型特化）。本文诚实记录这一发现，不回避也不夸大。

---

## 4. 操作语义

### 4.1 AST 操作语义

**定义 4.1（AST 值域）**。AST 求值值域 $V_A$ 包含：整数 $n \in \mathbb{Z}$、浮点 $f \in \mathbb{F}$（带 dtype）、布尔 $b$、字符串 $s$、张量 $T$、数组 $A$、元组、结构体、枚举实例、闭包、引用 $\&v$、$\&mut\,v$、`unit`。

**定义 4.2（AST 状态）**。求值状态 $\Sigma_A = (\rho, \sigma)$，其中 $\rho: \text{Ident} \rightharpoonup V_A$ 是变量环境，$\sigma: \text{Loc} \to V_A$ 是内存（用于引用解引用）。

**定义 4.3（AST 小步语义）**。求值关系 $\Sigma_A, e_A \to_A \Sigma_A', e_A'$ 按标准 small-step 规则定义。关键规则：

- **Assign-Var**：$\Sigma, \text{Assign}(\text{Var}(x), v) \to_A \Sigma[\rho \mapsto \rho[x \mapsto v']], \text{unit}$（其中 $v'$ 是 $v$ 求值结果）
- **Assign-Deref**：$\Sigma, \text{Assign}(\text{Deref}(e_p), v) \to_A \Sigma[\sigma \mapsto \sigma[\ell \mapsto v']], \text{unit}$（其中 $e_p$ 求值为位置 $\ell$）
- **Assign-Field**：$\Sigma, \text{Assign}(\text{Fld}(e_t, f), v) \to_A \Sigma', \text{unit}$（更新结构体的字段 $f$）
- **Assign-Index**：$\Sigma, \text{Assign}(\text{Idx}(e_t, \vec{ix}), v) \to_A \Sigma', \text{unit}$（更新张量/数组的索引位置）

**关键观察**：AST 语义**允许** `Assign-Index`、`Assign-Field`。这是 AST 表达力的一部分。

### 4.2 HIR 操作语义

**定义 4.4（HIR 值域与状态）**。$V_H$ 与 $V_A$ 同构；$\Sigma_H = (\rho_H, \sigma_H)$ 同 $\Sigma_A$。

**定义 4.5（HIR 小步语义）**。$\Sigma_H, e_H \to_H \Sigma_H', e_H'$ 按如下规则：

- **Assign-Var-H**：$\Sigma, \text{Assign}(x, v) \to_H \Sigma[\rho \mapsto \rho[x \mapsto v']], \text{unit}$
- **DerefAssign-H**：$\Sigma, \text{DerefAssign}(e_p, v) \to_H \Sigma[\sigma \mapsto \sigma[\ell \mapsto v']], \text{unit}$
- **FldAssign-H**：$\Sigma, \text{FldAssign}(e_t, f, v) \to_H \Sigma', \text{unit}$

**关键差异**：HIR 语义**不含** `Assign-Index`、`AssignOp-Field` 规则——这两类赋值在 HIR 中无对应节点，因此无法求值。

### 4.3 语义等价

**定义 4.6（可观察行为）**。程序的"可观察行为" $\text{obs}(P)$ 包括：终止状态（正常/异常/发散）、输出序列（`println` 等）、最终内存状态。两个程序语义等价 $P_1 \sim P_2$ 当且仅当 $\text{obs}(P_1) = \text{obs}(P_2)$。

**定义 4.7（AST-HIR 语义等价）**。对 AST 程序 $P_A$ 与 HIR 程序 $P_H$，定义 $P_A \sim_H P_H$ 当且仅当对任意初始状态 $\Sigma_0$，$\text{obs}(\Sigma_0, P_A) = \text{obs}(\Sigma_0, P_H)$。

**注**：由于 HIR 不含 `Assign-Index` 等，AST 程序若使用这些构造则**无法**找到等价的 HIR 程序——这正是 HIR 表达力子集的边界。

---

## 5. 主定理与证明

### 5.1 定理 L1（语义保持）

**定理 L1（语义保持）**。设 $a$ 为 AST 表达式，$h = \text{lower}(a)$ 为 lowering 成功（未返回 `Err`）时产生的 HIR 表达式。若 $a$ 的所有子表达式均能成功 lowering（即 $a$ 落在 HIR 可表示子集 $\mathcal{R}$ 内，见定理 L3），则：

$$
\forall \Sigma_0. \quad \text{obs}(\Sigma_0, a) = \text{obs}(\Sigma_0, h)
$$

即 $a \sim_H h$。

**证明**。对 $a$ 的结构作归纳。归纳假设：对所有子表达式 $a_i$，若 $h_i = \text{lower}(a_i)$ 成功，则 $a_i \sim_H h_i$。

**情况 1：$a = \text{Lit}(l)$**。$h = \text{Lit}(l)$（[lower_expr.rs:18-25](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/hir/lower/lower_expr.rs)），字面量值不变。$\text{obs}(\Sigma, l) = l = \text{obs}(\Sigma, l)$。✓

**情况 2：$a = \text{Var}(x)$**。$h = \text{Var}(x)$（[lower_expr.rs:28-101](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/hir/lower/lower_expr.rs)），变量名不变。若 $x$ 在作用域内，$\rho(x)$ 在 AST 与 HIR 求值中相同。✓

**情况 3：$a = \text{Bin}(op, e_1, e_2)$**。$h = \text{Bin}(op', h_1, h_2, \tau)$，其中 $op' = \text{lower\_binop}(op)$（[lower_expr.rs:109](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/hir/lower/lower_expr.rs)），$h_i = \text{lower}(e_i)$。`lower_binop` 是双射（[mod.rs:156-166](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/hir/lower/mod.rs)），故 $op' \equiv op$。由归纳假设 $e_i \sim_H h_i$，故 $\text{Bin}(op, e_1, e_2) \sim_H \text{Bin}(op, h_1, h_2, \tau)$——$\tau$ 字段不影响求值（仅用于后端代码生成）。✓

**情况 4：$a = \text{Assign}(e_{tgt}, e_v)$**。分四子情况（[lower_expr.rs:520-548](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/hir/lower/lower_expr.rs)）：

- 4a：$e_{tgt} = \text{Var}(x)$。$h = \text{Assign}(x, h_v)$。AST 的 `Assign-Var` 规则与 HIR 的 `Assign-Var-H` 规则相同。✓
- 4b：$e_{tgt} = \text{Deref}(e_p)$。$h = \text{DerefAssign}(h_p, h_v)$。AST 的 `Assign-Deref` 与 HIR 的 `DerefAssign-H` 相同。✓
- 4c：$e_{tgt} = \text{Fld}(e_t, f)$。$h = \text{FldAssign}(h_t, f, h_v)$。AST 的 `Assign-Field` 与 HIR 的 `FldAssign-H` 相同。✓
- 4d：$e_{tgt}$ 为其他形式（如 $\text{Idx}$、$\text{Call}$）。lowering 返回 `Err(ParseError("invalid assignment target"))`——**不在 L1 范围内**（$a \notin \mathcal{R}$），由 L2 处理。

**情况 5：$a = \text{Idx}(e_t, \vec{ix}_A)$**。$h = \text{Idx}(h_t, \vec{ix}_H)$，其中 $\vec{ix}_H = \text{lower\_index}(\vec{ix}_A)$（[lower_expr.rs:368-376, 772-782](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/hir/lower/lower_expr.rs)）。`lower_index` 是结构同构（D6）：`Single → Single`、`Range → Range`、`Colon → Colon`，递归调用 `lower_expr` 处理内部表达式。由归纳假设，内部表达式语义等价，故整体语义等价。✓

**情况 6：$a = \text{Match}(e_s, \vec{arm})$**。$h = \text{Match}(h_s, \vec{arm}_H)$。arms 顺序保留（[lower_expr.rs:657-679](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/hir/lower/lower_expr.rs) 用 `arms.iter()` 顺序映射）。每个 arm 的 pattern 经 `lower_pattern`（[lower_expr.rs:784-829](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/hir/lower/lower_expr.rs)）转换：

- `EnumVariant`：`tuple_fields: Vec<String>` → `tuple_binds: Vec<(String, String)>`，合成字段名 `_i`。这是**信息增加**（D7）——合成的字段名不改变匹配语义，仅用于后端字段访问。模式匹配语义由 arm 顺序与 pattern 结构决定，二者均保留。✓
- 其他 pattern 变体：1:1 映射。✓

**情况 7：$a = \text{SLit}(n, \vec{\tau}_A, \vec{(f, e)}, b)$**。$h = \text{SLit}(n, \vec{(f, h)}, b)$。**注意**：$\vec{\tau}_A$ 被丢弃（D3）。但 $\vec{\tau}_A$ 仅影响**类型注解**，不影响结构体字段的**运行时值**——字段值 $\vec{e}$ 已 lowering 为 $\vec{h}$，运行时按字段名访问。若 $n$ 是非泛型结构体，$\vec{\tau}_A$ 本就为空，无损失。若 $n$ 是泛型结构体，$\vec{\tau}_A$ 的丢弃**确实损失类型信息**，但当前 Tenth 运行时对泛型结构体的字段布局不做类型特化（[lower_stmt.rs:92-101](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/hir/lower/lower_stmt.rs) 仅存储定义，不实例化），故可观察行为不变。**此为 D3 静默丢失的语义保持辩护，但削弱了未来扩展性**——详见 §6.2。✓（在当前运行时语义下）

**情况 8：$a = \text{Call}(e_f, \vec{e})$**。$h = \text{Call}(h_f, \vec{h}, \tau_{ret})$。函数调用语义由函数体决定，参数顺序与值不变。✓

**情况 9：$a = \text{GCall}(e_f, \vec{\tau}_A, \vec{e})$**。分两子情况（[lower_expr.rs:176-308](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/hir/lower/lower_expr.rs)）：

- 9a：native generic ctor（`randn`/`zeros`/`ones`/`rand`/`tensor`/`tensor_from_vec`）。HIR 重写为 `Call(Var(runtime_name), args, ret_ty)`，其中 `runtime_name` 由 `(func_name, dtype)` 决定（如 `randn_f32`）。类型参数 $\tau_A$ 编码到 `runtime_name` 字符串中（D8）。运行时按 `runtime_name` 分发，语义等价。✓
- 9b：用户泛型函数。HIR 生成 mangled 函数定义 `func_T1_T2` 并重写为 `Call(Var(mangled), args, ret_ty)`。mangled name 唯一编码类型参数，运行时调用实例化的函数体。由 T18（[T18-泛型实例化作为类型替换.md](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/docs/论文/T18-泛型实例化作为类型替换.md)）的实例化等价性，语义保持。✓

**情况 10：$a = \text{Closure}(\vec{(x, \tau_A?)}, e)$**。$h = \text{Clos}(\vec{(x, \tau)}, h, \vec{c})$，其中 $\tau = \text{Unknown}$ 当 $\tau_A$ 缺失，$\vec{c}$ 是 `free_vars_in` 计算的自由变量捕获列表。AST 闭包在运行时也需捕获自由变量（解释器/VM 实现一致），HIR 显式列出捕获不改变语义。✓

**情况 11：$a = \text{Try}(e)$**。$h = \text{Try}(h)$，HIR 显式构造 `Result<T, str()>` 类型。AST 的 try block 语义本就要求错误为字符串（运行时约定），故类型注解与运行时一致。✓

**情况 12-20**：其他情况（`Ref`、`MutRef`、`Deref`、`Move`、`If`、`Block`、`Range`、`TLit`、`ALit`、`ELit`、`IStr`、`Tup`、`MCall`、`Fld`、`Un`）均为 1:1 结构映射或仅增加类型注解，归纳假设直接适用。✓

**归纳完成**。对所有 $a \in \mathcal{R}$，$a \sim_H \text{lower}(a)$。$\square$

**注**：L1 的证明依赖"HIR 的 `ty` 字段不影响求值语义"——这在 Tenth 的解释器与 VM 中成立（`ty` 仅用于编译期检查与代码生成优化），但若后端利用 `ty` 做特化优化（如 JIT），需额外证明 `ty` 的正确性，这由 T16（Subject Reduction）保证。

### 5.2 定理 L2（拒绝的正确性）

**定理 L2（拒绝的正确性）**。若 $\text{lower}(a)$ 返回 `Err(e)`，则 $a$ 落在 HIR 表达力子集 $\mathcal{R}$ 之外，即 $a$ 包含 HIR 无法表示的构造。具体地，所有拒绝点对应如下情形：

| 拒绝点 | 触发条件 | HIR 无法表示的原因 |
|--------|---------|-------------------|
| Assign target 不是 Var/Deref/Field | `x[i] = v`、`f() = v`、`{...} = v` 等 | HIR 无 `IndexAssign`、`CallAssign` 变体 |
| AssignOp target 不是 Var/Deref | `s.f += v`、`x[i] += v` | HIR 无 `FieldAssignOp`、`IndexAssignOp` 变体 |
| 泛型调用目标非具名函数 | `(expr).<T>(args)` | HIR GenericCall 要求 `func` 为 `Var` |
| native generic ctor 类型参数非单一 BaseType | `randn<Tensor>(...)`、`randn<i32>(...)` | HIR 仅支持 `BaseType` dtype |
| 未定义变量 | `undefined_var` | HIR 无法解析引用 |
| 未定义泛型函数 | `undefined_generic<T>(...)` | HIR 无法解析泛型引用 |
| 类型不兼容 | shape 冲突、dtype 冲突 | HIR 类型系统拒绝（由 T16） |
| 借用冲突 | `&mut x` 后 `&x` | HIR 借用检查拒绝（由 T19） |

**证明**。逐一审查 `lower_expr` 与 `lower_stmt` 中所有 `return Err(...)` 点：

1. **[lower_expr.rs:541-546](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/hir/lower/lower_expr.rs)**：`Assign` 的 target 匹配 `_` 分支。AST 允许任意 `Expr` 作为 target；HIR `Assign` target 为 `String`，`DerefAssign`/`FldAssign` 已专门处理 `Deref`/`Field`。其余 `ExprKind`（`Index`、`Call`、`Binary`、`Match`、...）无对应 HIR 变体。**拒绝正确**。

2. **[lower_expr.rs:563-568](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/hir/lower/lower_expr.rs)**：`AssignOp` 的 target 匹配 `_` 分支。HIR `AssignOp` target 为 `String`，`DerefAssignOp` 处理 `Deref`。无 `FieldAssignOp`、`IndexAssignOp`。**拒绝正确**。

3. **[lower_expr.rs:179-186](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/hir/lower/lower_expr.rs)**：`GenericCall` 的 `func` 不是 `Ident`。HIR GenericCall 要求具名函数（mangling 需要函数名）。**拒绝正确**。

4. **[lower_expr.rs:198-219](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/hir/lower/lower_expr.rs)**：native generic ctor 的类型参数不为单一 `BaseType`。运行时 `randn_f32` 等需要具体 dtype。**拒绝正确**。

5. **[lower_expr.rs:84-89](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/hir/lower/lower_expr.rs)**：未定义变量。HIR 无法解析引用。**拒绝正确**。

6. **[lower_expr.rs:253-258](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/hir/lower/lower_expr.rs)**：未定义泛型函数。**拒绝正确**。

7. 借用检查与 shape 检查的拒绝由 T19 与 T16 处理，不在本文范围但**不矛盾**。

**结论**：所有显式拒绝均对应 HIR 表达力不足或类型/借用系统约束，不存在"HIR 能表示但 lowering 错误拒绝"的情形。$\square$

**局限**：L2 仅证明**显式拒绝**的正确性。**静默丢失**（D3、D4）不在 L2 范围内——它们不触发 `Err`，而是产生语义弱化的 HIR。这由 L4 处理。

### 5.3 定理 L3（HIR 表达力子集刻画）

**定理 L3（HIR 表达力子集）**。定义 HIR 可表示子集 $\mathcal{R} \subset \text{Expr}_A$ 为满足以下所有条件的 AST 表达式集合：

- **C1（Assign target 限制）**：所有 `Assign` 的 target 是 `Var`、`Deref`、`Field` 之一；
- **C2（AssignOp target 限制）**：所有 `AssignOp` 的 target 是 `Var`、`Deref` 之一；
- **C3（GenericCall func 限制）**：所有 `GenericCall` 的 `func` 是 `Ident`；
- **C4（native generic ctor dtype 限制）**：所有 native generic ctor 的类型参数是单一 `BaseType`。

则 $\text{lower}: \mathcal{R} \to \text{Expr}_H$ 是良定义的（total function），且 $\mathcal{R}$ 是使 lowering 良定义的**最大子集**——任何严格包含 $\mathcal{R}$ 的子集都会使 lowering 在某些点上返回 `Err`。

**证明**。

**良定义性**：对每个 $a \in \mathcal{R}$，C1–C4 保证所有 `match` 分支不落入 `_` 拒绝分支。归纳地，若所有子表达式在 $\mathcal{R}$ 内，则 `lower_expr` 不返回 `Err`（除类型/借用检查的 `Err`，由 T16/T19 处理）。故 $\text{lower}(a)$ 良定义。

**最大性**：设 $\mathcal{R}' \supsetneq \mathcal{R}$，则存在 $a \in \mathcal{R}' \setminus \mathcal{R}$，$a$ 违反某 C$i$。由 C$i$ 的违反，$a$ 触发对应 `_` 分支，`lower_expr` 返回 `Err`。故 $\text{lower}$ 在 $\mathcal{R}'$ 上不良定义。$\square$

**推论 L3.1**。$\mathcal{R} \subsetneq \text{Expr}_A$——HIR 表达力严格弱于 AST。具体地，以下 AST 程序在 $\mathcal{R}$ 外：

- `x[i] = v`（违反 C1）
- `s.f += v`（违反 C2）
- `x[i] += v`（违反 C2）
- `f().g<T>(args)`（违反 C3）

### 5.4 定理 L4（静默丢失漏洞分析）

**定理 L4（静默丢失漏洞）**。在 lowering 中存在 2 处静默信息丢失，均不改变当前运行时可观察行为，但削弱编译期保证：

| # | 静默丢失 | 影响的编译期保证 | 运行时影响 | 源码位置 |
|---|---------|----------------|-----------|---------|
| L4-1 | `StructLiteral.generics` 丢弃（D3） | 泛型结构体实例化的类型参数不可追踪 | 无（运行时不实例化泛型结构体） | [lower_expr.rs:571](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/hir/lower/lower_expr.rs) |
| L4-2 | `is_pub` 可见性丢弃（D4） | 跨模块可见性检查无法执行 | 无（运行时不检查可见性） | [ast.rs:246,261,272](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/parser/ast.rs) |

**证明（构造性）**。

**L4-1**。构造 AST 程序：

```
struct Pair<T> { a: T, b: T }
let p1 = Pair<f32>{ a: 1.0, b: 2.0 };
let p2 = Pair<f64>{ a: 1.0, b: 2.0 };
```

经 parser（[parser.rs:241](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/parser/parser.rs)），$p_1, p_2$ 的 AST `StructLiteral.generics` 分别为 `[Base(F32)]`、`[Base(F64)]`。经 lowering（[lower_expr.rs:571](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/hir/lower/lower_expr.rs) `generics: _`），$p_1, p_2$ 的 HIR 完全相同（`SLit("Pair", [("a", ...), ("b", ...)], false)`）。**两个语义不同的 AST 程序产生相同的 HIR**——这是静默丢失。

**运行时影响**：当前 Tenth 运行时对泛型结构体不实例化（[lower_stmt.rs:92-101](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/hir/lower/lower_stmt.rs) 仅存储 `HirGenericStruct` 定义，不生成特化布局），字段访问按字段名偏移，dtype 信息从字段值动态推断。故 $p_1, p_2$ 的运行时行为相同（都是包含两个浮点的结构体）。

**编译期影响**：HIR 无法区分 $p_1, p_2$ 的类型，导致：
- 后续 `p1.a + 1.0`（若 `1.0` 默认 f64）的 dtype 提升无法基于 `p1` 的泛型实例化推断；
- T16 的类型重建在 `p1.a` 上得到 `Type::Unknown`，而非 `Type::Base(F32)`。

**严重性评估**：**中等**。当前不影响运行时正确性，但阻碍未来泛型结构体的完整支持。

**L4-2**。构造 AST 程序：

```
mod m {
    fn helper() { ... }       // 非 pub
    pub fn api() { helper(); }
}
use m::*;
api();
helper();  // 应被拒绝（helper 非 pub），但 HIR 无法判断
```

AST `ItemKind::Function` 的 `is_pub: false` for `helper`、`is_pub: true` for `api`。经 lowering，HIR `HirFnDef`（[hir.rs:226-234](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/hir/hir.rs)）**无 `is_pub` 字段**，二者仅函数名不同。`use m::*`（[lower_stmt.rs:307-368](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/hir/lower/lower_stmt.rs)）导入模块内**所有**函数，不区分 `pub`/非 `pub`。故 `helper()` 在 HIR 层可调用——可见性检查失效。

**运行时影响**：无（运行时不检查可见性，函数调用按名字分发）。

**编译期影响**：用户期望的封装性（`helper` 不可外部访问）未被执行——这是**编译期保证的削弱**，而非运行时语义的改变。

**严重性评估**：**低-中**。不影响运行时正确性，但削弱模块封装性。Rustc 的 HIR 通过 `Visibility` 枚举保留可见性（[rustc_hir](https://doc.rust-lang.org/nightly/nightly-rustc/rustc_hir/enum.Visibility.html)），Tenth 可参照修复。$\square$

**L4 的诚实局限**：L4 仅证明**已发现的**静默丢失。可能存在未发现的静默丢失——本文审查覆盖了 `lower_expr.rs`、`lower_stmt.rs`、`mod.rs` 中所有 `match` 分支与 `_` 通配符，但未对 `scope.rs`、`import.rs`、`closures.rs`、`types.rs` 做同等深度审查。未来工作应扩展审查范围（见 §8）。

### 5.5 定理 L5（Translation Validation 框架）

**定理 L5（Translation Validation）**。定义可验证条件 $\Phi(a, h)$ 为以下合取：

$$
\Phi(a, h) \equiv \bigwedge_{i=1}^{5} \phi_i(a, h)
$$

其中：

- $\phi_1$：$h = \text{lower}(a)$（结构对应——通过 lowering 函数的确定性输出保证）；
- $\phi_2$：$a \in \mathcal{R}$（即 C1–C4 成立）；
- $\phi_3$：所有子表达式的 VC 递归成立（归纳结构）；
- $\phi_4$：类型注解一致性——$h.\text{ty}$ 与 T16 的类型重建结果一致；
- $\phi_5$：无静默丢失告警——$a$ 不含 D3、D4 涉及的构造（即 $a$ 不含带 `generics` 的 `StructLiteral`，不含 `is_pub: true` 的项）。

则：$\Phi(a, h) \Rightarrow a \sim_H h$。

**证明**。

- $\phi_1 \land \phi_2 \land \phi_3$：由 L1，$a \in \mathcal{R}$ 时 lowering 保持语义。归纳结构由 $\phi_3$ 保证。
- $\phi_4$：T16 的 Subject Reduction 保证类型正确性，类型注解不影响运行时语义（仅影响代码生成）。
- $\phi_5$：排除 D3、D4 的静默丢失情形，确保 HIR 携带的信息足以重建 AST 语义。

故 $\Phi(a, h) \Rightarrow a \sim_H h$。$\square$

**验证算法**：

```
fn validate(a: AST, h: HIR) -> Result<(), VCFailure> {
    // φ1: 结构对应
    if !structurally_corresponds(a, h) { return Err(VCFailure::StructMismatch); }
    // φ2: AST 在可表示子集内
    if !in_representable_subset(a) { return Err(VCFailure::OutOfSubset); }
    // φ3: 递归验证子表达式
    for (a_i, h_i) in zip(children(a), children(h)) {
        validate(a_i, h_i)?;
    }
    // φ4: 类型一致性（委托 T16）
    if !type_consistent(h) { return Err(VCFailure::TypeMismatch); }
    // φ5: 无静默丢失
    if has_silent_loss(a) { return Err(VCFailure::SilentLoss); }
    Ok(())
}
```

**与运行时检查的关系**：Translation Validation 是**编译期**检查，不引入运行时开销。失败的 VC 对应明确的编译错误（拒绝）或警告（静默丢失）。当前 Tenth 实现已部分实现 $\phi_1$–$\phi_4$（通过 `?` 传播），但**未实现** $\phi_5$（静默丢失检测）——这是未来工作。

---

## 6. 信息有损变换的逐一分析

### 6.1 D1：Assign target 收紧

**AST**：`Assign { target: Box<Expr>, value: Box<Expr> }`（[ast.rs:117-120](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/parser/ast.rs)）。target 可以是任意 `Expr`——`x`、`*p`、`s.f`、`x[i]`、`f().g` 等均合法。

**HIR**：拆分为四个变体（[hir.rs:79-87, 105-113, 118-122](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/hir/hir.rs)）：
- `Assign { target: String, ... }`——变量赋值
- `DerefAssign { target: Box<HirExpr>, ... }`——解引用赋值
- `FldAssign { target: Box<HirExpr>, field: String, ... }`——字段赋值
- **缺失**：`IndexAssign`、`CallAssign`、`BinaryAssign` 等

**Lowering 行为**（[lower_expr.rs:520-548](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/hir/lower/lower_expr.rs)）：
- `target = Var(x)` → `Assign(x, v)` ✓
- `target = Deref(p)` → `DerefAssign(p, v)` ✓
- `target = Field(t, f)` → `FldAssign(t, f, v)` ✓
- 其他 → `Err(ParseError("invalid assignment target"))` ✗（拒绝）

**影响**：以下 AST 合法程序被拒绝：
- `x[i] = v`——索引赋值（用户需改用方法如 `t.set(i, v)`）
- `f().g = v`——调用结果字段赋值（罕见，但 AST 允许）
- `(if c { &mut x } else { &mut y }) = v`——条件表达式赋值（罕见）

**拒绝是否合理**：合理。Tenth 的语言设计不鼓励隐式索引赋值（张量索引赋值有性能陷阱，应显式调用方法）。HIR 不引入 `IndexAssign` 是设计选择，非缺陷。

### 6.2 D2：AssignOp target 收紧

**AST**：`AssignOp { target: Box<Expr>, op, value }`（[ast.rs:121-125](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/parser/ast.rs)）。

**HIR**：拆分为两个变体：
- `AssignOp { target: String, op, value }`——变量复合赋值
- `DerefAssignOp { target: Box<HirExpr>, op, value }`——解引用复合赋值
- **缺失**：`FldAssignOp`、`IndexAssignOp`

**Lowering 行为**（[lower_expr.rs:550-569](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/hir/lower/lower_expr.rs)）：
- `target = Var(x)` → `AssignOp(x, op, v)` ✓
- `target = Deref(p)` → `DerefAssignOp(p, op, v)` ✓
- 其他 → `Err(ParseError("invalid assignment target"))` ✗

**影响**：
- `s.f += v` 被拒绝——但 `s.f = s.f + v` 可通过（`s.f = ...` 由 `FldAssign` 处理）。**这是 AssignOp 与 Assign 的不对称**：`Assign` 支持 Field，`AssignOp` 不支持。用户需手动展开。
- `x[i] += v` 被拒绝——同理。

**拒绝是否合理**：合理但不对称。`FldAssign` 的存在表明 `s.f = v` 合法，但 `s.f += v` 不合法——这是 HIR 表达力的不一致。可作为未来改进点（添加 `FldAssignOp` 变体）。

### 6.3 D3：StructLiteral generics 静默丢弃（**静默丢失**）

**AST**：`StructLiteral { name, generics: Vec<TypeAnnotation>, fields, use_defaults }`（[ast.rs:126-131](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/parser/ast.rs)）。

**HIR**：`StructLiteral { name: String, fields: Vec<(String, HirExpr)>, has_default: bool }`（[hir.rs:88-92](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/hir/hir.rs)）——**无 `generics` 字段**。

**Lowering 行为**（[lower_expr.rs:571](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/hir/lower/lower_expr.rs)）：`generics: _` 显式丢弃。

**类型构造**（[lower_expr.rs:631](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/hir/lower/lower_expr.rs)）：
```rust
let struct_ty = Type::from_annotation(&ast::TypeAnnotation::Named(
    ast::Ident { name: name.name.clone(), span: name.span.clone() }
));
```
仅用 `name`，未传入 `generics`——故 `Pair<f32>` 与 `Pair<f64>` 的 HIR `ty` 均为 `Type::Struct("Pair")`。

**影响**：
- 两个语义不同的 AST 程序产生相同的 HIR（L4-1 的构造性证明）；
- HIR 无法区分泛型实例化的不同类型参数；
- 后端的代码生成无法基于类型参数特化（如 `Pair<f32>` 应使用 4 字节字段，`Pair<f64>` 应使用 8 字节字段）。

**当前运行时影响**：无（运行时不实例化泛型结构体，字段按名字访问，dtype 从值动态推断）。

**未来风险**：若 Tenth 未来支持泛型结构体的类型特化（类似 C++ 模板或 Rust 泛型单态化），此静默丢失将导致语义错误。**建议修复**：在 HIR `StructLiteral` 中添加 `generics: Vec<Type>` 字段，或引入 `GenericStructLiteral` 变体。

### 6.4 D4：is_pub 可见性静默丢弃（**静默丢失**）

**AST**：`ItemKind::Function { ..., is_pub: bool }`、`StructDef { ..., is_pub: bool }`、`Impl { ..., is_pub: bool }`（[ast.rs:246, 261, 272](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/parser/ast.rs)）。

**HIR**：`HirFnDef`（[hir.rs:226-234](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/hir/hir.rs)）无 `is_pub` 字段；`HirProgram` 的 `structs`/`methods` 等也无可见性信息。

**Lowering 行为**：`is_pub` 字段在 [lower_stmt.rs:442-488](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/hir/lower/lower_stmt.rs) 等位置被隐式忽略（`ItemKind::Function { name, generics, params, return_type, body, .. }` 的 `..` 包含 `is_pub`）。

**影响**：
- `use m::*` 导入所有函数，不区分 `pub`/非 `pub`（[lower_stmt.rs:333-340](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/hir/lower/lower_stmt.rs)）；
- 模块封装性失效——非 `pub` 函数可被外部模块调用。

**当前运行时影响**：无（运行时不检查可见性）。

**未来风险**：若 Tenth 未来引入私有函数（如用于安全敏感场景），此丢失将导致封装性破坏。**建议修复**：在 `HirFnDef` 中添加 `is_pub: bool` 字段，并在 `use` 导入时过滤。

### 6.5 D5：Ident → String 信息缩减

**AST**：`Ident { name: String, span: Span }`（[ast.rs:13-17](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/parser/ast.rs)）。

**HIR**：许多位置使用 `String` 而非保留 `Ident`（如 `Assign.target`、`Var(String)`、`Field.field`、`StructLiteral.name` 等）。

**影响**：span 信息在标识符级别丢失，但父 `HirExpr` 仍携带 `span`（[hir.rs:159](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/hir/hir.rs)）。**这是调试信息的缩减，非语义丢失**——span 不影响程序行为，仅影响错误报告精度。

**评估**：可接受。Rustc 通过 `HirId` 系统保留全 span，但代价是复杂的 ID 管理。Tenth 的简化方案牺牲了部分错误精度，换取 HIR 简洁性。

### 6.6 D6：Index 类型重命名（**结构同构，无信息损失**）

**AST**：`IndexExpr`（[ast.rs:185-192](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/parser/ast.rs)）与 **HIR**：`Index`（[hir.rs:184-191](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/hir/hir.rs)）的三个变体完全对应：

| AST `IndexExpr` | HIR `Index` | 同构 |
|-----------------|-------------|------|
| `Single(Expr)` | `Single(HirExpr)` | ✓（递归同构） |
| `Range { start: Option<Box<Expr>>, end: Option<Box<Expr>> }` | `Range { start: Option<Box<HirExpr>>, end: Option<Box<HirExpr>> }` | ✓ |
| `Colon` | `Colon` | ✓ |

**Lowering**（[lower_expr.rs:772-782](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/hir/lower/lower_expr.rs)）：1:1 映射，无信息损失。

**结论**：任务描述提及"Index 从 `Vec<IndexExpr>` 收紧为 `Vec<Index>`（结构变化）"——经审查，这是**类型重命名**而非结构变化。本文诚实纠正这一预期偏差。

### 6.7 D7：Match tuple_fields → tuple_binds（**信息增加**）

**AST**：`Pattern::EnumVariant { ..., tuple_fields: Vec<String> }`（[ast.rs:159-164](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/parser/ast.rs)）——仅记录绑定名，不记录字段名。

**HIR**：`HirPattern::EnumVariant { ..., tuple_binds: Vec<(String, String)> }`（[hir.rs:134-141](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/hir/hir.rs)）——记录 `(field_name, bind_name)` 对，字段名合成为 `_0, _1, ...`。

**Lowering**（[lower_expr.rs:786-794](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/hir/lower/lower_expr.rs)）：
```rust
tuple_binds: tuple_fields.iter().enumerate()
    .map(|(i, bind_name)| (format!("_{}", i), bind_name.clone()))
    .collect(),
```

**影响**：HIR 比 AST 携带更多信息（合成的字段名）。这是**信息增加**，不是损失。运行时按 `_i` 字段名访问元组变体的第 $i$ 个字段。

**结论**：任务描述提及"Match arms 顺序与 pattern 表示变化"——arms 顺序保留，pattern 表示变化是信息增加。无语义损失。

### 6.8 D8：GenericCall 重写（部分损失）

**AST**：`GenericCall { func: Box<Expr>, generics: Vec<TypeAnnotation>, args: Vec<Expr> }`。

**HIR**：对 native generic ctor 与用户泛型函数，重写为 `Call { func: Var(mangled_name), args, ret_ty }`（[lower_expr.rs:237-307](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/hir/lower/lower_expr.rs)）。类型参数编码到 `mangled_name` 字符串中（如 `randn_f32`、`func_T1_T2`）。

**影响**：
- HIR `Call` 的 `func` 是 `Var(String)`，类型参数信息在字符串中，不是结构化的 `Vec<Type>`；
- HIR 仍保留 `GenericCall` 变体（[hir.rs:31-36](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/hir/hir.rs)），但实际 lowering 中**未使用**该变体（所有 generic call 都重写为 `Call`）。

**部分损失**：类型参数以字符串形式保留（可解析恢复），但失去结构化表示。对后端代码生成无影响（按 mangled name 查找），但对 HIR 分析工具（如 shape 检查器）需解析 mangled name 才能恢复类型参数。

**严重性**：**低**。类型参数信息未真正丢失（编码在 mangled name 中），仅表示形式改变。

---

## 7. 工程实现分析

### 7.1 `?` 错误传播的正确性

Tenth lowering 使用 Rust 的 `?` 操作符传播 `TenthResult<T>`。审查所有 `?` 点：

- `self.lower_expr(left)?`（[lower_expr.rs:104-105](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/hir/lower/lower_expr.rs)）——子表达式 lowering 失败时传播；
- `self.scope.check_use(&ident.name, &ident.span)?`（[lower_expr.rs:50](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/hir/lower/lower_expr.rs)）——使用已 move 的值时传播；
- `Self::check_binary_shape_compat(...)?`（[lower_expr.rs:107](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/hir/lower/lower_expr.rs)）——shape 不兼容时传播；
- `self.resolve_call_type(...)?`（[lower_expr.rs:148](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/hir/lower/lower_expr.rs)）——调用类型解析失败时传播；
- 类似地，`lower_stmt.rs` 中所有 `?` 点均对应明确的错误条件。

**正确性**：`?` 传播保证任何子表达式的错误立即终止 lowering，不产生部分 HIR。这与 L2 的"拒绝的正确性"一致——错误要么在子表达式层处理，要么在当前层处理，不会"静默通过"。

**覆盖完整性**：审查 `lower_expr` 的所有 `match` 分支，每个分支要么返回 `Ok(HirExpr)`，要么在失败时返回 `Err`。**无分支静默返回 `Ok` 而丢弃信息**——除 D3（`generics: _`）与 D4（`is_pub` 隐式丢弃）外，这两个静默丢失已由 L4 记录。

### 7.2 拒绝点的覆盖完整性

下表列出所有显式拒绝点（`return Err(...)`）：

| # | 位置 | 触发条件 | 错误类型 |
|---|------|---------|---------|
| R1 | [lower_expr.rs:84-89](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/hir/lower/lower_expr.rs) | 未定义变量 | TypeError |
| R2 | [lower_expr.rs:179-186](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/hir/lower/lower_expr.rs) | GenericCall func 非 Ident | TypeError |
| R3 | [lower_expr.rs:198-219](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/hir/lower/lower_expr.rs) | native generic ctor 类型参数非法 | TypeError |
| R4 | [lower_expr.rs:253-258](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/hir/lower/lower_expr.rs) | 未定义泛型函数 | TypeError |
| R5 | [lower_expr.rs:541-546](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/hir/lower/lower_expr.rs) | Assign target 非法 | ParseError |
| R6 | [lower_expr.rs:563-568](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/hir/lower/lower_expr.rs) | AssignOp target 非法 | ParseError |
| R7 | [lower_stmt.rs:235-243](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/hir/lower/lower_stmt.rs) | trait 实现缺少方法 | TypeError |
| R8 | scope.rs:58-65, 67-79, 81-97 | 借用冲突、use of moved value | TypeError |

加上 `?` 传播的隐式拒绝点（来自 T16 的 shape 检查、T19 的借用检查），覆盖了所有 HIR 表达力不足的情形。

**遗漏审查**：未发现遗漏的拒绝点——所有 `_` 通配符分支均返回 `Err`（R5、R6），所有类型/借用检查均通过 `?` 传播。

---

## 8. 开放问题与未来工作

### 8.1 机器验证的语义保持

本文的 L1 证明是**纸面证明**，未在 Coq/Lean 中机器验证。未来工作：

- 将 AST/HIR 操作语义嵌入 Coq/Lean；
- 将 L1 的归纳证明形式化，机械检查每一步；
- 验证 L5 的 VC 算法实现与形式化定义一致。

### 8.2 HIR 表达力的扩展

当前 HIR 表达力弱于 AST（D1、D2 的拒绝）。未来可考虑：

- 添加 `IndexAssign` 变体，支持 `x[i] = v`；
- 添加 `FldAssignOp` 变体，支持 `s.f += v`，消除 D1/D2 的不对称；
- 修复 D3、D4 的静默丢失（在 HIR 中保留 `generics` 与 `is_pub`）。

### 8.3 静默丢失检测器

实现 L5 的 $\phi_5$（无静默丢失告警）：

- 在 lowering 中检测 `StructLiteral` 带 `generics` 时发出 warning；
- 在 `use m::*` 导入非 `pub` 函数时发出 warning；
- 将 warning 纳入 `HirProgram.warnings`（[hir.rs:257](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/hir/hir.rs)）。

### 8.4 扩展审查范围

本文未深入审查 `scope.rs`、`import.rs`、`closures.rs`、`types.rs` 中的潜在静默丢失。未来工作应扩展 L4 的审查到这些模块。

### 8.5 与 tenthc 的同步

T12（[T12-双侧编译器语义等价性.md](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/docs/论文/T12-双侧编译器语义等价性.md)）发现 tenthc 侧缺失多项 shape 检查。本文的语义保持证明针对 Rust 母编译器；tenthc 侧的 lowering 是否保持语义需单独验证（未来工作）。

---

## 9. 局限（诚实披露）

### 9.1 证明的局限

- **L1 未机器验证**：纸面归纳证明可能遗漏边界情况。本文逐一处理了 20 种 `ExprKind` 变体，但未在证明助手中机械检查。
- **L1 依赖运行时语义**：证明假设 HIR 的 `ty` 字段不影响求值。这在当前解释器/VM 中成立，但若 JIT 利用 `ty` 做特化，需额外证明（由 T9 JIT 语义保持处理）。
- **L4 的审查范围有限**：仅深入审查了 `lower_expr.rs`、`lower_stmt.rs`、`mod.rs`，未同等深度审查 `scope.rs`、`import.rs`、`closures.rs`、`types.rs`。可能存在未发现的静默丢失。

### 9.2 形式化的局限

- **操作语义未涵盖所有 HIR 节点**：§4 仅给出关键规则，未列举所有 20+ 变体的语义规则。完整形式化需补充。
- **可观察行为的定义粗糙**：$\text{obs}$ 包含"输出序列"与"最终内存状态"，未严格定义内存等价（如张量数据的字节级比较）。
- **未涵盖并发与异常**：Tenth 的 `TryBlock`、`spawn` 等并发构造的语义保持未在本文范围内。

### 9.3 工程差距

- **D3 的"运行时不影响"假设**：当前 Tenth 运行时不实例化泛型结构体，故 D3 静默丢失不影响运行时。但这一假设可能随语言演进失效——本文的 L1 证明在"当前运行时语义"下成立，未来扩展需重新验证。
- **D4 的"运行时不检查可见性"假设**：同理，若未来引入运行时可见性检查，D4 将成为语义错误。

### 9.4 与 T16 的依赖

L1 的 $\phi_4$ 依赖 T16 的 Subject Reduction。若 T16 的证明有漏洞（如 T16 自身承认的 `Type::Unknown` 双重角色问题），L1 的类型一致性条件可能不成立。本文不重复 T16 的局限，仅引用其结论。

### 9.5 任务描述与源码的偏差

任务描述提及"Match arms 顺序与 pattern 表示变化"作为信息有损变换点。经源码审查，**arms 顺序保留**，**pattern 表示变化是信息增加**（D7）。本文以源码为准，纠正了这一预期偏差。这一偏差本身不影响论文结论，但提示：基于文档的预期需与源码核对。

---

## 10. 结论

本文对 Tenth 语言的 AST→HIR lowering 进行了形式化的 Translation Validation 分析。核心发现：

1. **语义保持成立**（L1）：在 HIR 可表示子集 $\mathcal{R}$ 上，lowering 保持运行时语义。证明基于结构归纳，覆盖所有 20 种 `ExprKind` 变体。

2. **拒绝正确**（L2）：所有显式拒绝（`?` 传播 `Err`）均对应 HIR 表达力不足或类型/借用约束，无错误拒绝。

3. **HIR 表达力严格弱于 AST**（L3）：$\mathcal{R} \subsetneq \text{Expr}_A$，具体边界由 C1–C4 刻画。`x[i] = v`、`s.f += v` 等 AST 合法程序被拒绝。

4. **发现 2 处静默丢失**（L4）：
   - **L4-1**：`StructLiteral.generics` 丢弃（D3）——两个语义不同的 AST 程序产生相同 HIR。当前运行时不影响，但阻碍未来泛型结构体支持。
   - **L4-2**：`is_pub` 可见性丢弃（D4）——模块封装性失效。当前运行时不影响，但削弱编译期保证。

5. **Translation Validation 框架**（L5）：可验证条件 $\Phi = \bigwedge_{i=1}^5 \phi_i$ 充分保证语义保持。当前实现部分实现 $\phi_1$–$\phi_4$，未实现 $\phi_5$（静默丢失检测）。

**对实施的指导**：

- **优先修复 D3**（在 HIR `StructLiteral` 中保留 `generics`）——影响泛型结构体未来支持，修复成本低；
- **次优先修复 D4**（在 `HirFnDef` 中保留 `is_pub`）——影响模块封装性，修复成本中；
- **实现 $\phi_5$ 检测器**——在 lowering 中对 D3、D4 涉及的构造发出 warning，纳入 `HirProgram.warnings`；
- **考虑添加 `FldAssignOp`**——消除 D1/D2 的不对称（`s.f = v` 合法但 `s.f += v` 不合法）。

**核心贡献**：将"lowering 是否保持语义"这一工程问题，转化为可证伪、可分级的形式化命题，并诚实标注 2 处静默丢失漏洞——不回避，不夸大。

---

## 11. 参考文献

[1] A. Pnueli, M. Siegel, E. Singh. "Translation Validation." *TACAS'98*, LNCS 1384, Springer, 1998. pp. 151–166.

[2] X. Leroy. "Formal verification of a realistic compiler." *Communications of the ACM*, 52(7), 2009. pp. 107–115.

[3] C. Lattner, V. Adve. "LLVM: A Compilation Framework for Lifelong Program Analysis & Transformation." *CGO'04*, IEEE, 2004. pp. 75–88.

[4] G. Necula. "Translation Validation for an Optimizing Compiler." *PLDI'00*, ACM, 2000. pp. 83–94.

[5] S. Kell. "Some Challenges for Compiler Reuse." *BCS Software Practice & Experience*, 2018.

[6] B. C. Pierce, D. N. Turner. "Local Type Inference." *POPL'98*, ACM, 1998. pp. 252–265.

[7] M. M. Chakravarty et al. "Associated Type Synonym." *WGP'05*, ACM, 2005.

[8] Tenth 项目. "工作规范 v1.1." [工作规范.md](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/.trae/rules/工作规范.md).

[9] Tenth 数理部. "T16-双向类型重建." [T16-双向类型重建.md](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/docs/论文/T16-双向类型重建.md).

[10] Tenth 数理部. "T12-双侧编译器语义等价性." [T12-双侧编译器语义等价性.md](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/docs/论文/T12-双侧编译器语义等价性.md).

[11] Tenth 数理部. "T18-泛型实例化作为类型替换." [T18-泛型实例化作为类型替换.md](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/docs/论文/T18-泛型实例化作为类型替换.md).

[12] Tenth 数理部. "T19-语句粒度借用检查." [T19-语句粒度借用检查.md](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/docs/论文/T19-语句粒度借用检查.md).

[13] Tenth 数理部. "T9-JIT特化语义保持证明." [T9-JIT特化语义保持证明.md](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/docs/论文/T9-JIT特化语义保持证明.md).

---

## 附录 A：信息有损变换点索引

| 编号 | 变换点 | 类型 | 定理归属 | 源码位置 |
|------|--------|------|---------|---------|
| D1 | Assign target 收紧 | 显式拒绝 | L2、L3 (C1) | [lower_expr.rs:520-548](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/hir/lower/lower_expr.rs) |
| D2 | AssignOp target 收紧 | 显式拒绝 | L2、L3 (C2) | [lower_expr.rs:550-569](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/hir/lower/lower_expr.rs) |
| D3 | StructLiteral.generics 丢弃 | **静默丢失** | L4-1 | [lower_expr.rs:571](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/hir/lower/lower_expr.rs) |
| D4 | is_pub 可见性丢弃 | **静默丢失** | L4-2 | [ast.rs:246,261,272](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/parser/ast.rs) |
| D5 | Ident → String (span 丢失) | 信息缩减（非语义） | — | 全局 |
| D6 | Index 类型重命名 | 结构同构 | — | [lower_expr.rs:772-782](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/hir/lower/lower_expr.rs) |
| D7 | Match tuple_fields → tuple_binds | 信息增加 | — | [lower_expr.rs:786-794](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/hir/lower/lower_expr.rs) |
| D8 | GenericCall 重写为 Call+mangled | 部分损失（可恢复） | — | [lower_expr.rs:237-307](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/hir/lower/lower_expr.rs) |

## 附录 B：主定理索引

| 定理 | 陈述 | 证明方法 | 局限 |
|------|------|---------|------|
| L1 | 语义保持（在 $\mathcal{R}$ 上） | 结构归纳，20 种 ExprKind | 未机器验证；依赖运行时语义 |
| L2 | 拒绝的正确性 | 逐一审查 8 处拒绝点 | 仅显式拒绝；静默丢失由 L4 处理 |
| L3 | HIR 表达力子集刻画 | 良定义性 + 最大性 | $\mathcal{R}$ 由 C1–C4 刻画，可能不完备 |
| L4 | 静默丢失漏洞分析 | 构造性证明 | 审查范围有限（仅 3 个核心文件） |
| L5 | Translation Validation 框架 | $\Phi \Rightarrow \sim_H$ | $\phi_5$ 未在当前实现中检查 |

## 附录 C：与现有文档的对应

| 本文章节 | 对应文档 |
|---------|---------|
| §3 (AST/HIR 形式化) | [CODE_WIKI.md](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/CODE_WIKI.md) 模块详解 |
| §5.4 (L4 静默丢失) | [AUDIT.md](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/AUDIT.md) 缺陷登记（建议新增条目） |
| §6 (有损变换分析) | [能力梳理/能力全梳理.md](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/能力梳理/能力全梳理.md) 状态标记 |
| §7 (工程实现) | [MEMO.md](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/MEMO.md) 变更记录 |
| §9 (局限) | T16 §9、T12 §9 的局限体系 |

---

**实施建议**（数理部→编译器部）：

1. **优先级 P1**：修复 D3——在 `HirExprKind::StructLiteral` 中添加 `generics: Vec<Type>` 字段，lowering 时保留而非 `generics: _`。同步更新 `tenthc/hir/lower.th`。
2. **优先级 P1**：修复 D4——在 `HirFnDef` 中添加 `is_pub: bool` 字段，`use m::*` 导入时过滤非 `pub` 函数（或至少发出 warning）。
3. **优先级 P2**：实现 $\phi_5$ 静默丢失检测器，在 lowering 中对 D3、D4 涉及的构造发出 warning，纳入 `HirProgram.warnings`。
4. **优先级 P3**：考虑添加 `HirExprKind::FldAssignOp`，消除 D1/D2 的不对称。
5. **优先级 P3**：扩展 L4 审查到 `scope.rs`、`import.rs`、`closures.rs`、`types.rs`。

---

**版本记录**：

- v1.0（2026-07-02）：初版，5 个主定理，8 处信息有损变换点（含 2 处静默丢失），审查轮次 v1（结构→证明→边界→诚实）。
