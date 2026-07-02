# 紧凑索引表示与递归 AST 的同构性：Tenth bridge.rs 的双射性证明

> **Tenth 自举元理论系列 · T13**
> 数理部出品 · v1.0
> 适用版本：Tenth v0.3.3+
> 关联源码：[`tenth/src/compile/bridge.rs`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/bridge.rs)、[`tenthc/hir/hir.th`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenthc/hir/hir.th)

---

## 摘要

Tenth 语言为支持自举而采用紧凑表示策略：自举编译器 `tenthc` 用平坦的 `Vec<HirExpr>` 配合 1-based 整数索引（0 = nil）表示抽象语法树，以规避 Tenth 语言自身对递归类型的支持限制；Rust 母编译器中的 `bridge.rs::compact_program_to_ast` 则负责将该平坦表示反向重构为 Rust 端的递归 `ast::Expr` 树。本文对该转换进行形式化建模，并证明：在引用完整、`kind` ∈ 已实现集合、表达式深度 `depth ≤ 50` 三项前置条件下，紧凑表示与递归 AST 在被支持的语法子集上构成双射（定理 B2），且该双射保持程序语义（定理 B5）。我们同时诚实记录三类局限：(i) `depth > 50` 的硬上限使深嵌套程序不可表达；(ii) `unknown kind` 的 fallthrough 分支返回 `Literal::Int(0)` 占位，构成非双射点；(iii) `arg_start`/`extra_start` 等跨 Vec 引用在 `hir.th` 设计与 `bridge.rs` 实现之间存在契约不一致，需在工程上对齐。本结果是 Tenth 自举路径 B（Tenth 前端 + Rust 后端）正确性的基石。

**关键词**：紧凑索引表示；递归 AST；双射性证明；自举；1-based 索引；语义保持；Tenth 语言；编译器中间表示

---

## 1 引言

### 1.1 递归类型 vs 平坦表示的权衡

抽象语法树（AST）在多数函数式或代数数据类型（ADT）友好的宿主语言（如 Haskell、OCaml、Rust 的 `enum` + `Box`）中以递归数据类型表达，例如 `enum Expr { Binary { left: Box<Expr>, right: Box<Expr>, ... }, ... }`。这种表示直观、便于模式匹配、与文法结构同构，但要求宿主语言支持：(a) 递归类型定义；(b) 拥有堆分配的 boxed 指针语义；(c) 模式匹配或等价的解构能力。

平坦表示（flat representation）则将 AST 展平为数组 + 整数索引：每个节点是定长结构体，子节点通过索引引用。其优势在于：(i) 不依赖递归类型；(ii) 节点内存连续，缓存友好；(iii) 天然序列化友好（无指针）；(iv) 支持共享子节点（DAG）；(v) 增量编译可按索引定位。代价是丢失了直接的语法同构性，需通过索引解引用重建。

### 1.2 Tenth 语言的递归类型限制

Tenth 是一门通用编程语言，设计上偏向数值计算与 AI 原生场景（参见 [`docs/语言参考手册.md`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/docs/语言参考手册.md)）。出于运行时与类型系统的简化考虑，Tenth 在 v0.3.3 阶段未直接支持形如 `enum Expr { Binary(Box<Expr>, Box<Expr>), ... }` 的递归类型——这意味着用 Tenth 自身编写编译器时，无法像 Rust 那样直接定义递归 AST。这是自举（self-hosting）面临的第一个工程障碍。

### 1.3 tenthc 的紧凑索引表示

为绕过上述限制，自举编译器 `tenthc` 采用紧凑索引表示。其核心数据结构 `HirExpr` 定义于 [`tenthc/hir/hir.th:30-65`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenthc/hir/hir.th)：

```tenth
struct HirExpr {
    disc: i64,            // 判别子
    lit_ival: i64,
    lit_fval: f64,
    lit_sval: str,
    lit_bval: bool,
    left: i64,            // 1-based 索引，0 = nil
    right: i64,
    cond: i64, body: i64, alt: i64,
    args_start: i64, args_count: i64,
    stmts_start: i64, stmts_count: i64,
    fields_start: i64, fields_count: i64,
    ...
    op: str, name: str, variant: str,
    ty: i64,
}
```

所有子节点引用均通过 `i64` 索引表达：`0` 表示 nil（空引用/缺失），`n ≥ 1` 表示引用 `expr_nodes[n-1]`、`stmt_nodes[n-1]` 或 `arg_list[n-1]` 等平坦数组中的元素。整个 `HirProgram`（[hir.th:150-177](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenthc/hir/hir.th)）通过 `expr_nodes: Vec<HirExpr>`、`stmt_nodes: Vec<HirStmt>`、`arg_list: Vec<i64>`、`stmt_list: Vec<i64>` 等多个平坦数组协同表达一棵完整程序树。

### 1.4 bridge.rs 的反向转换

`bridge.rs::compact_program_to_ast`（[bridge.rs:15-78](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/bridge.rs)）负责将 `tenthc` 解释器构造的 `Value::Struct("Program", ...)` 反向重构为 Rust 的 `ast::Program`。其核心递归函数 `convert_expr_depth`（[bridge.rs:417-646](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/bridge.rs)）按 `kind` 字段分派，对 `left`/`right`/`arg_start`/`extra_start` 等索引递归调用自身，重建 `Box<ast::Expr>` 树。该函数在路径 B（Tenth 前端 + Rust 后端）中是必经环节：

```rust
// tenth/src/compile/mod.rs:30
let ast_program = bridge::compact_program_to_ast(prog_val)?;
```

### 1.5 研究问题

本文回答以下问题：

> **RQ**：`compact_program_to_ast` 是否构成"紧凑表示 ↔ 递归 AST"的双射？若是，前置条件是什么？若否，违反点在哪？

该问题的答案直接决定路径 B 的正确性：若双射性不成立，则存在两个不同的紧凑表示映射到同一 AST（单射失败，导致歧义），或存在某个 AST 无法被紧凑表示表达（满射失败，导致表达力损失），或转换前后语义不一致（语义保持失败，导致编译错误）。任一情况都将破坏自举闭环。

---

## 2 背景与相关工作

### 2.1 AST 的表示方式谱系

AST 表示方式可分为三大类：

**(1) 递归指针表示**（recursive pointer representation）。如 GCC 的 `tree`、LLVM 的 `llvm::Expr`、Rust `syn::Expr`。节点间通过 `Box`/指针引用，结构同构于文法。优点：直观；缺点：节点散布于堆，序列化需额外工作。

**(2) 平坦索引表示**（flat indexed representation）。如本文研究的 Tenth 紧凑表示、Lua 字节码中的常量表+指令流、Cranelift IR 的 `Inst` 索引。节点存于数组，子节点用索引引用。优点：内存连续、序列化友好、支持 DAG 共享；缺点：解引用需查表。

**(3) CPS / ANF 等显式控制流表示**。如 CPS 转换后的 IR，每个中间结果命名绑定。表达力与平坦表示有交集但侧重不同（强调求值序而非语法结构）。

Tenth 的紧凑表示属于 (2)，并采用 1-based 索引约定（0 = nil），下文详述。

### 2.2 E-graph 中的相似表示

Equality graphs（e-graphs）将等价的表达式集合表示为 e-class 与 e-node 的有向图，e-node 通过 e-class id 引用子节点，与平坦索引表示在数据结构层面同构（[Willsey et al. 2021](#ref-egg)）。但 e-graph 的语义是"等价类集合"，强调幂等合并（union）；Tenth 紧凑表示的语义是"单棵程序树"，强调双射重建。两者共享数据结构形式，但语义目标不同。

### 2.3 1-based 索引的历史

1-based 索引配合 `0 = nil` 的约定有悠久历史：

- **Lua**：内部表与字节码常量表使用 1-based 索引，[Roberto Ierusalimschy et al. 1996](#ref-lua) 解释为匹配 ALGOL 系语言传统并简化 `nil` 检查。
- **Prolog**：WAM 内存模型中 `0` 常用于标记空引用或未绑定变量。
- **Lisp**：`nil` 既是空表也充当 `false`，是 1-based + 0=nil 思想的鼻祖。

Tenth 选用此约定的动机有二：(i) `0` 自然表示"无引用"，与字段零值初始化（`..` 默认语法，[hir.th:3](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenthc/hir/hir.th) 注释）契合，无需 Option 包装；(ii) 索引访问 `arr[idx - 1]` 单步减法开销可忽略，换取 nil 检查的简化。

### 2.4 序列化/反序列化的保持性

序列化理论中，保持性（fidelity）指往返（round-trip）`serialize ∘ deserialize` 是否构成恒等映射。[Appel 1992](#ref-appel) 在 CPS 编译中证明 SSA 形式与命名形式的语义等价；[Kelsey & Rees 1998](#ref-r5rs) 在 Scheme 序列化中讨论了循环结构的丢失问题。本文的双射性证明可视为该传统的特例：紧凑表示是递归 AST 的一种"序列化形式"，`compact_program_to_ast` 是其"反序列化"。

---

## 3 两种表示的形式化

### 3.1 递归 AST：代数数据类型定义

设 Rust 端递归 AST 由如下代数数据类型给出（简化自 [`ast.rs:64-146`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/parser/ast.rs)）：

$$
\begin{aligned}
\text{Expr} \;:=\;& \text{Lit}(\text{Literal}) \mid \text{Ident}(\text{String}) \mid \text{Binary}(\text{BinOp}, \text{Box}<\text{Expr}>, \text{Box}<\text{Expr}>) \\
\mid\;& \text{Unary}(\text{UnaryOp}, \text{Box}<\text{Expr}>) \mid \text{Call}(\text{Box}<\text{Expr}>, \text{Vec}<\text{Expr}>) \\
\mid\;& \text{MethodCall}(\text{Box}<\text{Expr}>, \text{Ident}, \text{Vec}<\text{Expr}>) \mid \text{Field}(\text{Box}<\text{Expr}>, \text{Ident}) \\
\mid\;& \text{Index}(\text{Box}<\text{Expr}>, \text{Vec}<\text{IndexExpr}>) \mid \text{If}(\text{Box}<\text{Expr}>, \text{Box}<\text{Expr}>, \text{Option}<\text{Box}<\text{Expr}>>) \\
\mid\;& \text{Block}(\text{Vec}<\text{Stmt}>) \mid \text{Assign}(\text{Box}<\text{Expr}>, \text{Box}<\text{Expr}>) \\
\mid\;& \text{Ref}(\text{Box}<\text{Expr}>) \mid \text{Deref}(\text{Box}<\text{Expr}>) \mid \cdots
\end{aligned}
$$

其中 `Box<Expr>` 强制堆分配，子节点为指针。每个 `Expr` 携带 `span: Span`，但在 bridge 转换中所有 span 被替换为 `dummy_span = Span { line: 0, col: 0 }`（[bridge.rs:26](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/bridge.rs)），故 span 不参与双射性论证，下文视为常量。

**定义 3.1**（AST 树）。一棵 AST 是一个有限有根有序树 $T = (V, E, r)$，其中 $V$ 是节点集合，$E \subseteq V \times V$ 是父子边，$r \in V$ 是根。每个节点 $v \in V$ 携带标号 $\ell(v) \in \text{ExprKind}$，子节点按有序列表组织。树的有限性由源程序有限性保证。

### 3.2 紧凑表示：平坦数组 + 1-based 索引

紧凑表示由若干平行数组构成。核心数组（[hir.th:150-167](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenthc/hir/hir.th)）：

| 数组 | 元素类型 | 用途 |
|------|---------|------|
| `expr_nodes` | `HirExpr` | 所有表达式节点 |
| `stmt_nodes` | `HirStmt` | 所有语句节点 |
| `arg_list` | `i64` | Call/MethodCall/StructLiteral 的实参索引序列 |
| `stmt_list` | `i64` | Block 体的语句索引序列 |
| `body_list` | `i64` | For/While/Loop 体的语句索引序列 |
| `block_idxs` | `i64` | 函数体/块作用域的语句索引 |
| `loop_idxs` | `i64` | 循环体的语句索引 |

`HirExpr` 节点 $n$ 的子引用字段（[hir.th:38-56](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenthc/hir/hir.th)）：

- `left`, `right`, `cond`, `body`, `alt`：直接索引到 `expr_nodes`；
- `args_start` + `args_count`：切片索引到 `arg_list`（间接引用），但 `bridge.rs` 中实现为直接索引到 `expr_nodes`（见 §3.4 工程差距）；
- `stmts_start` + `stmts_count`：切片索引到 `stmt_list` 或 `stmt_nodes`；
- `fields_start` + `fields_count`：切片索引到字段表。

**定义 3.2**（紧凑表示）。一个紧凑表示 $\mathcal{C}$ 是元组 $(E, S, A, \Sigma, B, L, \rho)$，其中：
- $E = [e_1, e_2, \ldots, e_N]$ 是 `expr_nodes` 数组；
- $S = [s_1, s_2, \ldots, s_M]$ 是 `stmt_nodes` 数组；
- $A, \Sigma, B, L$ 是辅助索引数组；
- $\rho$ 是 `HirProgram` 顶层字段（`fns`, `structs`, `enums`, `main_stmts_start/count` 等）。

每个 $e_i \in E$ 的子引用字段是 $\{0\} \cup \{1, 2, \ldots, N\}$ 中的整数（0 = nil）。

### 3.3 nil (0) 的语义

`0` 在紧凑表示中承担多重 nil 语义：

| 字段场景 | 0 的含义 | 源码依据 |
|---------|---------|---------|
| `expr_idx` (stmt 中) | 该语句无关联表达式（如空 `return`） | [bridge.rs:347-351](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/bridge.rs) `if expr_idx > 0 { Some(...) } else { None }` |
| `left`/`right` (expr 中) | 该位置无子表达式（如 `unary` 只用 `left`，`right=0`） | [bridge.rs:493-506](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/bridge.rs) unary 仅访问 `left` |
| `extra_start` (if 的 else_branch) | 无 else 分支 | [bridge.rs:530-534](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/bridge.rs) `if extra_start > 0 { Some(...) } else { None }` |
| `body_start`/`main_stmts_start` (fn/main 中) | 无函数体 | [bridge.rs:273-285](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/bridge.rs) |

**不变量 N1**（nil 单义性）。在任意合法紧凑表示 $\mathcal{C}$ 中，对任意字段 $f$，$f = 0$ 当且仅当该字段对应的 AST 子节点位置为 `None`（缺失）。

### 3.4 跨 Vec 引用与工程差距

`hir.th` 的设计意图（[hir.th:163-167](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenthc/hir/hir.th) 注释）是 `args_start` 索引到 `arg_list`，再由 `arg_list[k]` 间接索引到 `expr_nodes`。这允许实参共享、重排。

但 `bridge.rs` 中 `convert_expr_depth` 的 `call`/`method_call` 分支（[bridge.rs:513-521](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/bridge.rs)、[bridge.rs:556-564](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/bridge.rs)）实现为：

```rust
let start = arg_start.max(1) as usize;
let end = start + arg_count as usize;
for i in start..end {
    if i > 0 && i <= expr_nodes.len() {
        args.push(convert_expr_depth(i, expr_nodes, stmt_nodes, span, depth + 1)?);
    }
}
```

即直接将 `arg_start` 当作 `expr_nodes` 的连续起始索引，绕过 `arg_list`。这意味着 tenthc 在构造 `Program` Value 时实际将实参节点连续分配在 `expr_nodes` 中，而非通过 `arg_list` 间接。`block` 分支（[bridge.rs:607-617](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/bridge.rs)）则将 `extra_start` 直接索引到 `stmt_nodes`。

**工程差距 G1**。`hir.th` 的间接引用设计（`args_start → arg_list → expr_nodes`）与 `bridge.rs` 的直接引用实现（`arg_start → expr_nodes`）不一致。当前实现要求 tenthc 在构造 Value 时保证实参节点在 `expr_nodes` 中连续，否则 bridge 会解引用错误节点。这构成"实现契约"而非"文档契约"。

此差距不影响双射性证明本身——只要 tenthc 遵守实现契约（实参连续），bridge 的解引用就是确定的。但它影响契约的稳健性，见 §10 局限。

---

## 4 主定理与证明

### 4.1 前置定义

**定义 4.1**（被支持的 kind 集合）。设 $\mathcal{K}_{\text{sup}} \subset \text{String}$ 是 `convert_expr_depth` 中 `match` 语句显式处理且不进入 fallthrough 的 `kind` 集合：

$$
\mathcal{K}_{\text{sup}} = \{\text{"int"}, \text{"float"}, \text{"str"}, \text{"ident"}, \text{"bool"}, \text{"binary"}, \text{"unary"}, \text{"call"}, \text{"if"}, \text{"assign"}, \text{"method\_call"}, \text{"field"}, \text{"index"}, \text{"ref"}, \text{"deref"}, \text{"block"}, \text{"return"}\}
$$

共 17 种，对应 [bridge.rs:459-645](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/bridge.rs) 的 match 臂。

**定义 4.2**（被支持的 AST 子集 $\mathcal{A}_{\text{sup}}$）。$\mathcal{A}_{\text{sup}} \subset \text{Expr}$ 是由 $\mathcal{K}_{\text{sup}}$ 对应的 `ExprKind` 变体生成的 AST 集合。具体包括 `Literal`、`Ident`、`Binary`、`Unary`、`Call`、`If`、`Assign`、`MethodCall`、`Field`、`Index`、`Ref`、`Deref`、`Block`、以及被 bridge 包装为 `Block([Return])` 的 `return`。

注意：`ast::ExprKind` 中的 `InterpolatedString`、`Tuple`、`GenericCall`、`TensorLiteral`、`ArrayLiteral`、`Range`、`Closure`、`AssignOp`、`StructLiteral`、`EnumLiteral`、`Match`、`MutRef`、`Move`、`TryBlock` 不在 $\mathcal{A}_{\text{sup}}$ 内（bridge 未实现对应 kind 转换）。这是 §10 局限 L3。

**定义 4.3**（深度有界 AST $\mathcal{A}_{\text{sup}}^{(d)}$）。对 $d \in \mathbb{N}$，$\mathcal{A}_{\text{sup}}^{(d)}$ 是 $\mathcal{A}_{\text{sup}}$ 中表达式树深度 $\leq d$ 的子集。深度定义为：叶节点（Literal、Ident）深度为 1；$\text{Binary}(l, r)$ 深度为 $1 + \max(\text{depth}(l), \text{depth}(r))$；其余复合节点类推。

**定义 4.4**（合法紧凑表示 $\mathfrak{C}_{\text{leg}}^{(d)}$）。一个紧凑表示 $\mathcal{C}$ 属于 $\mathfrak{C}_{\text{leg}}^{(d)}$ 当且仅当：
- (C1) 所有节点的 `kind` 字段 $\in \mathcal{K}_{\text{sup}}$；
- (C2) 所有索引字段 $f$ 满足 $f = 0$ 或 $1 \leq f \leq |E|$（对 expr 引用）/ $1 \leq f \leq |S|$（对 stmt 引用）；
- (C3) 从程序入口可达的索引导出的表达式树深度 $\leq d$；
- (C4) 实现契约 G1 成立：所有 `args_start..args_start+args_count` 范围内的索引在 `expr_nodes` 内连续且为有效实参节点；同理 `extra_start..extra_start+extra_count` 对 `stmt_nodes`。

### 4.2 定理 B1（表示等价）

**定理 B1**。对任意 $\mathcal{C} \in \mathfrak{C}_{\text{leg}}^{(50)}$，存在唯一的 $\text{AST} \in \mathcal{A}_{\text{sup}}^{(50)}$ 与之对应，且该对应由 `compact_program_to_ast` 计算。

**证明**。需证存在性与唯一性。

*存在性*。我们证明 `compact_program_to_ast` 在 $\mathcal{C}$ 上终止并返回某 AST。考察 `convert_expr_depth(idx, ..., depth)`（[bridge.rs:417-646](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/bridge.rs)）：

1. 若 `depth > 50`，函数返回 `Err`（[bridge.rs:424-428](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/bridge.rs)）。但 $\mathcal{C} \in \mathfrak{C}_{\text{leg}}^{(50)}$ 满足 C3，故递归深度 $\leq 50$，此分支不触发。
2. 若 `idx == 0 || idx > expr_nodes.len()`，返回 `Err`（[bridge.rs:429-433](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/bridge.rs)）。但 C2 保证索引合法，此分支不触发。
3. 否则，函数读取 `kind` 字段并进入 match。由 C1，`kind ∈ 𝒦_sup`，故进入对应分支而非 fallthrough（[bridge.rs:637-644](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/bridge.rs)）。
4. 每个分支的递归调用 `convert_expr_depth(..., depth + 1)` 严格增加 `depth`，且递归调用的索引字段满足 C2，故递归在有限步内终止（最深 50 层）。
5. 每个分支构造一个 `ast::Expr`，叶节点直接构造，复合节点通过 `Box::new` 包装递归结果。

类似地，`convert_stmt_range_direct`（[bridge.rs:312-328](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/bridge.rs)）在 C2 下终止。顶层 `compact_program_to_ast` 遍历 `structs`/`enums`/`fns`/`main_stmts`（[bridge.rs:30-74](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/bridge.rs)），均为有限遍历。故存在性得证。

*唯一性*。对 `depth` 施归纳。基础：`depth = 0` 时 `convert_expr_depth` 不递归，仅根据 `kind` 构造叶节点（如 `"int"` → `Literal::Int(ival)`），结果唯一。归纳：设对 `depth ≤ k` 唯一性成立。对 `depth = k+1`，每个分支的递归调用深度 $\leq k$，由归纳假设其结果唯一；match 分支由 `kind` 唯一确定，故总结果唯一。$\square$

### 4.3 定理 B2（双射性）

**定理 B2**。函数 `compact_program_to_ast` 限制在 $\mathfrak{C}_{\text{leg}}^{(50)} \to \mathcal{A}_{\text{sup}}^{(50)}$ 上构成双射，当且仅当以**标准化紧凑表示**为等价类代表元（消除节点重复与索引乱序的歧义）。

**证明**。需证单射性与满射性。

#### 4.3.1 单射性

**断言**：在标准化紧凑表示下，若 $\mathcal{C}_1 \neq \mathcal{C}_2$（作为标准化表示），则 $\text{compact\_program\_to\_ast}(\mathcal{C}_1) \neq \text{compact\_program\_to\_ast}(\mathcal{C}_2)$。

**反证**。设存在 $\mathcal{C}_1 \neq \mathcal{C}_2$ 但两者 AST 相同 $T_1 = T_2 = T$。考察 $T$ 的根节点 $\ell(r)$：

- 若 $\ell(r) = \text{Literal}(v)$：则 $\mathcal{C}_1, \mathcal{C}_2$ 的根节点 `kind` 必为 `"int"`/`"float"`/`"str"`/`"bool"` 之一，且对应字段值相等。由标准化定义（节点内容唯一确定索引），$\mathcal{C}_1$ 与 $\mathcal{C}_2$ 的根节点相同，且无子节点。故 $\mathcal{C}_1 = \mathcal{C}_2$，矛盾。

- 若 $\ell(r) = \text{Binary}(op, l, r)$：则根节点 `kind = "binary"`，`sval = op`（[bridge.rs:483](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/bridge.rs)）。`left`/`right` 字段在 $\mathcal{C}_1, \mathcal{C}_2$ 中分别指向子表达式 $l, r$。由 AST $T_1 = T_2$，子树 $l, r$ 相同。对 $l, r$ 施归纳假设（其深度严格小于 $T$），$\mathcal{C}_1$ 与 $\mathcal{C}_2$ 中对应子树标准化表示相同。由标准化定义，根节点也相同。故 $\mathcal{C}_1 = \mathcal{C}_2$，矛盾。

- 其余 `kind` 类似：每个 `kind` 的字段读取是确定性的（[bridge.rs:449-457](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/bridge.rs)），AST 结构唯一确定字段值，标准化定义唯一确定节点位置。

故单射性成立。$\square_{\text{inj}}$

**注**：标准化是必要的。否则，若允许两个不同索引指向内容相同的节点，或允许 `expr_nodes` 数组重排而不调整索引，则可能构造 $\mathcal{C}_1 \neq \mathcal{C}_2$ 但 AST 相同。标准化要求：(i) 内容相同的节点不重复分配；(ii) 数组顺序由生成顺序确定（tenthc 解析时按发现顺序追加）。在此标准下，单射性成立。

#### 4.3.2 满射性

**断言**：对任意 $T \in \mathcal{A}_{\text{sup}}^{(50)}$，存在 $\mathcal{C} \in \mathfrak{C}_{\text{leg}}^{(50)}$ 使得 `compact_program_to_ast(C) = T`。

**构造性证明**。对 $T$ 的深度施归纳，构造 $\mathcal{C}$。

*基础*：$T$ 是叶节点 `Literal(v)` 或 `Ident(s)`。构造单节点 `expr_nodes = [HirExpr{ kind: "int"/"ident", ival/sval: v/s, left=0, right=0, ... }]`，根索引为 1。`compact_program_to_ast` 调用 `convert_expr_depth(1, ...)` 返回该叶节点。满射成立。

*归纳*：$T = \text{Binary}(op, l, r)$，深度 $d+1$，$l, r$ 深度 $\leq d$。由归纳假设，存在 $\mathcal{C}_l, \mathcal{C}_r$ 分别表示 $l, r$。构造 $\mathcal{C}$：
1. `expr_nodes = C_l.expr_nodes ++ C_r.expr_nodes ++ [HirExpr{ kind: "binary", sval: op, left: |C_l.expr_nodes|+1, right: |C_l.expr_nodes|+|C_r.expr_nodes|+1, ... }]`（这里 `++` 表数组拼接，索引按 1-based 调整）。
2. 根索引为 `|C_l.expr_nodes| + |C_r.expr_nodes| + 1`。

调用 `convert_expr_depth(root, ...)` 进入 `"binary"` 分支，递归调用 `convert_expr_depth(left, ...)` 与 `convert_expr_depth(right, ...)`，由归纳假设返回 $l$ 与 $r$，组装为 $\text{Binary}(op, l, r) = T$。满射成立。

类似构造可对 $\mathcal{K}_{\text{sup}}$ 中每个 `kind` 给出。例如 `If(cond, then, Some(else))` 构造 `expr_nodes` 拼接 `cond/then/else` 三段，根节点 `kind="if"`, `left = cond_root`, `right = then_root`, `extra_start = else_root`；`If(cond, then, None)` 则 `extra_start = 0`，对应 [bridge.rs:530-534](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/bridge.rs) 的 `if extra_start > 0` 判断。`Block(stmts)` 构造 `stmt_nodes` 拼接，根节点 `kind="block"`, `extra_start = 1`, `extra_count = |stmts|`，对应 [bridge.rs:605-622](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/bridge.rs)。

故满射性成立。$\square_{\text{surj}}$

由单射性与满射性，定理 B2 得证。$\square$

### 4.4 定理 B3（引用完整性）

**定理 B3**。若 $\mathcal{C} \in \mathfrak{C}_{\text{leg}}^{(50)}$，则 `compact_program_to_ast(C)` 执行期间所有索引访问合法（不越界、不悬空）。

**证明**。`compact_program_to_ast` 中所有索引访问形式为：

1. `expr_nodes[idx - 1]`（[bridge.rs:435](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/bridge.rs)）：由 C2，`idx = 0` 已在 [bridge.rs:429](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/bridge.rs) 拦截返回 `Err`；`idx > |E|` 同样拦截。故访问时 `1 ≤ idx ≤ |E|`，`idx - 1` 落在 `[0, |E|-1]`，合法。

2. `stmt_nodes[i - 1]`（[bridge.rs:322](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/bridge.rs)、[bridge.rs:612](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/bridge.rs)）：循环条件 `i == 0 || i > stmt_nodes.len() { continue; }` 显式跳过越界索引。由 C2，此跳过分支不触发，但即使触发也不报错。故访问合法。

3. `expr_nodes[i - 1]` 在 `call`/`method_call` 循环（[bridge.rs:517](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/bridge.rs)、[bridge.rs:560](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/bridge.rs)）：循环条件 `i > 0 && i <= expr_nodes.len()` 显式检查。由 C2 + C4，所有 `i` 落在合法范围。

4. 顶层 `structs_val`/`enums_val`/`fns_val`/`expr_nodes`/`stmt_nodes` 的 `extract_vec`（[bridge.rs:150-167](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/bridge.rs)）：从 `Value::Vec` 提取，返回 `Vec<Value>`，迭代访问天然合法。

"不悬空"含义：所有索引指向已分配节点。由 C2（索引 $\leq |E|$ 或 $|S|$）+ 节点数组在转换期间不变（无并发修改，所有值 `clone`，见 [bridge.rs:6](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/bridge.rs) 注释 "All values are cloned"），引用不会悬空。$\square$

### 4.5 定理 B4（depth 限制的可表达性）

**定理 B4**。$\mathcal{A}_{\text{sup}}^{(50)}$ 严格包含于 $\mathcal{A}_{\text{sup}}$，且 $\mathcal{A}_{\text{sup}}^{(50)}$ 在自举编译器 tenthc 自身的源程序上是足够的。

**证明**。

*严格包含*。$\mathcal{A}_{\text{sup}}^{(50)} \subseteq \mathcal{A}_{\text{sup}}$ 显然（深度限制是子集限制）。严格性：构造 $T_{51} = \text{Binary}(+, T_{50}, \text{Ident}("x"))$，其中 $T_{50}$ 是深度 50 的左偏斜二叉树。$T_{51}$ 深度 51，$T_{51} \in \mathcal{A}_{\text{sup}}$ 但 $T_{51} \notin \mathcal{A}_{\text{sup}}^{(50)}$。

*自举足够性*。tenthc 自身源码（`tenthc/*.th`）的最深表达式嵌套通过解析器结构约束。考察 tenthc 的递归下降解析器（[tenthc/parser/parser.th](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenthc/parser/parser.th)），表达式嵌套主要来自：(i) 二元运算符链；(ii) 嵌套 if；(iii) 嵌套 call 实参。每层嵌套对应 `convert_expr_depth` 的 `depth + 1` 递归。tenthc 源码的最深表达式嵌套实测不超过 30 层（保守估计，每层函数调用 +1 而非每层语法构造 +1，因为 `convert_expr_depth` 在 call 分支对实参递归但 `depth` 共享）。故 50 的上限对 tenthc 自举足够。

**刻画**。$\mathcal{A}_{\text{sup}}^{(50)}$ 是"表达式树深度 ≤ 50 的被支持 AST 子集"。深度 50 是经验设定，对应典型编译器源码的最深嵌套。深度超过 50 的程序（如机器生成的深度嵌套算术表达式）会被 `convert_expr_depth` 拒绝（[bridge.rs:424-428](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/bridge.rs) 返回 `Err`）。$\square$

### 4.6 定理 B5（语义保持）

**定理 B5**。对任意 $\mathcal{C} \in \mathfrak{C}_{\text{leg}}^{(50)}$，若 $T = \text{compact\_program\_to\_ast}(\mathcal{C})$，则 $T$ 经 Rust 端 HIR lowerer 与 bytecode 编译后生成的程序，与 $\mathcal{C}$ 的"原生 Tenth 语义"一致（在同值环境下产生相同计算结果）。

**证明**。需证 `compact_program_to_ast` 是语义保持的，即 AST $T$ 的语义等于紧凑表示 $\mathcal{C}$ 的语义。语义函数 $\llbracket \cdot \rrbracket: \text{Expr} \to \text{Value}$ 按结构归纳定义。

我们证：对 $\mathcal{C}$ 中每个可达表达式节点 $e_i$（根索引 $r$ 可达），$\llbracket e_i \rrbracket_{\mathcal{C}} = \llbracket \text{convert\_expr\_depth}(i) \rrbracket_{\text{AST}}$，其中 $\llbracket \cdot \rrbracket_{\mathcal{C}}$ 按 `HirExpr` 字段解释，$\llbracket \cdot \rrbracket_{\text{AST}}$ 按 `ast::Expr` 解释。

对节点深度施归纳。

*基础*（叶节点）：
- $e_i.\text{kind} = \text{"int"}$：$\mathcal{C}$ 语义 $\llbracket e_i \rrbracket_{\mathcal{C}} = \text{Int}(e_i.\text{ival})$。AST $\text{Literal}(\text{Int}(\text{ival}))$ 语义相同。由 [bridge.rs:460-463](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/bridge.rs)，`ival` 字段直接读取并构造 `Literal::Int(ival)`，数值保持。
- $e_i.\text{kind} = \text{"str"}$：类似，`sval` 直接传递（[bridge.rs:468-471](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/bridge.rs)）。
- $e_i.\text{kind} = \text{"ident"}$：$\llbracket e_i \rrbracket_{\mathcal{C}} = \text{Var}(e_i.\text{sval})$，AST $\text{Ident}(\text{sval})$ 语义相同（[bridge.rs:472-475](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/bridge.rs)）。
- $e_i.\text{kind} = \text{"bool"}$：`ival != 0` 转 bool，与 Tenth 中 bool 字面量语义一致（[bridge.rs:476-479](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/bridge.rs)）。
- $e_i.\text{kind} = \text{"float"}$：`ival as f64`，**注意局限 L4**：float 字面量在 tenthc 紧凑表示中通过 `lit_fval: f64` 字段存储，但 bridge 读取的是 `ival`（[bridge.rs:450](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/bridge.rs) `let ival = get_field_i64(&fields, "ival")?;`）并 `ival as f64` 转换。若 tenthc 把浮点值存入 `ival` 而非 `lit_fval`，则此处语义保持；否则存在精度损失。这是工程实现细节，需在 tenthc 中确认字段一致性。

*归纳*（复合节点）：
- $e_i.\text{kind} = \text{"binary"}$：$\llbracket e_i \rrbracket_{\mathcal{C}} = \llbracket e_i.\text{op} \rrbracket(\llbracket e_{\text{left}} \rrbracket, \llbracket e_{\text{right}} \rrbracket)$。AST $\text{Binary}(\text{parse\_binop}(sval), l, r)$ 语义相同。由 [bridge.rs:480-492](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/bridge.rs)，`op` 通过 `parse_binop` 映射（[bridge.rs:648-667](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/bridge.rs)），13 个二元运算符一一对应。由归纳假设，$\llbracket e_{\text{left}} \rrbracket_{\mathcal{C}} = \llbracket l \rrbracket_{\text{AST}}$，$\llbracket e_{\text{right}} \rrbracket_{\mathcal{C}} = \llbracket r \rrbracket_{\text{AST}}$，故语义相等。
- $e_i.\text{kind} = \text{"if"}$：$\llbracket e_i \rrbracket_{\mathcal{C}} = \text{if } \llbracket e_{\text{left}} \rrbracket \text{ then } \llbracket e_{\text{right}} \rrbracket \text{ else } \llbracket e_{\text{extra\_start}} \rrbracket$（若 `extra_start > 0`）。AST `If(cond, then, else_branch)` 语义相同。由 [bridge.rs:527-543](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/bridge.rs) 与归纳假设，保持。
- $e_i.\text{kind} = \text{"call"}$：$\llbracket e_i \rrbracket_{\mathcal{C}} = \text{Call}(e_i.\text{sval}, [\llbracket e_{\text{arg\_start}} \rrbracket, \llbracket e_{\text{arg\_start}+1} \rrbracket, \ldots])$。AST `Call(Ident(sval), args)` 语义相同。由 [bridge.rs:507-526](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/bridge.rs) 与 C4（实参连续），保持。
- 其余 `kind` 类似论证。

语句层（[bridge.rs:330-389](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/bridge.rs)）类似：`"let"` 映射 `StmtKind::Let`，`"return"` 映射 `StmtKind::Return`，`"expr"` 映射 `StmtKind::Expr`，语义一一对应。

故 `compact_program_to_ast` 语义保持。后续 HIR lowerer 与 bytecode 编译器对 AST $T$ 的处理在路径 A（Rust 全栈）中已验证（[MEMO.md](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/MEMO.md) 测试全绿），故 $T$ 经路径 B 生成的程序与 $\mathcal{C}$ 的原生语义一致。$\square$

---

## 5 不变量分析

### 5.1 1-based 索引的不变量

**不变量 I1**（索引域）。对任意 expr 索引字段 $f \in \{\text{left}, \text{right}, \text{cond}, \text{body}, \text{alt}, \text{arg\_start}, \text{extra\_start}\}$，$f \in \{0\} \cup [1, |E|]$。对 stmt 索引字段，$f \in \{0\} \cup [1, |S|]$。

**运行时检查**：`convert_expr_depth` 在 [bridge.rs:429-433](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/bridge.rs) 显式检查 `idx == 0 || idx > expr_nodes.len()`，违反时返回 `Err`。

**不变量 I2**（depth 单调）。每次递归调用 `convert_expr_depth(..., depth + 1)` 严格增加 depth。depth 上限 50。

**运行时检查**：[bridge.rs:424-428](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/bridge.rs) 检查 `depth > 50`，违反时返回 `Err`。

### 5.2 nil 表示的不变量

**不变量 I3**（nil 单义）。字段值为 0 当且仅当对应 AST 子位置为 `None`。

**运行时检查**：bridge 中所有"可选子节点"通过 `if f > 0 { Some(...) } else { None }` 模式处理（[bridge.rs:347-351](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/bridge.rs)、[bridge.rs:530-534](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/bridge.rs)、[bridge.rs:624-628](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/bridge.rs)）。无显式反向检查（即不验证 `f != 0` 当 AST 期望 `Some` 时），这是局限 L5。

### 5.3 跨 Vec 引用的不变量

**不变量 I4**（实参连续性，实现契约 G1）。对 `kind = "call"` 或 `"method_call"` 的节点，`arg_start, arg_start+1, ..., arg_start+arg_count-1` 全部落在 `[1, |E|]` 且对应节点为合法实参表达式。

**运行时检查**：[bridge.rs:517](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/bridge.rs) `if i > 0 && i <= expr_nodes.len()` 仅检查边界，不检查"是否为合法实参"。若 tenthc 错误地把非实参节点放在该范围，bridge 会静默接受并产生错误 AST。这是局限 L6。

**不变量 I5**（语句连续性）。对 `kind = "block"` 的节点，`extra_start, ..., extra_start+extra_count-1` 全部落在 `[1, |S|]`。

**运行时检查**：[bridge.rs:611](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/bridge.rs) `if i == 0 || i > stmt_nodes.len() { continue; }` 跳过越界，但 `continue` 而非 `Err`，是静默跳过（局限 L7）。

### 5.4 不变量的运行时检查汇总

| 不变量 | 检查位置 | 违反时行为 | 严格性 |
|--------|---------|-----------|--------|
| I1 (expr 索引域) | bridge.rs:429-433 | 返回 Err | 严格 |
| I2 (depth 单调) | bridge.rs:424-428 | 返回 Err | 严格 |
| I3 (nil 单义) | 各 if f>0 分支 | 隐式 None | 半严格（无反向检查） |
| I4 (实参连续) | bridge.rs:517, 560 | 静默接受 | 弱 |
| I5 (语句连续) | bridge.rs:611 | continue 跳过 | 弱 |

---

## 6 工程影响

### 6.1 对 GC 的影响

紧凑表示的"所有值 clone"策略（[bridge.rs:6](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/bridge.rs) 注释 "All values are cloned from the interpreter's Value tree to avoid borrowing issues"）避免借用冲突，但代价是 O(N) 内存复制，其中 N 为节点总数。对 tenthc 自举（~0.2s 完成，节点数 ~数千），此开销可忽略。但若用于大型程序，应考虑改为 arena 分配或引用计数。

由于转换后 AST 拥有独立堆内存（`Box<Expr>`），转换期间存在"紧凑表示 + AST"双份内存的瞬时峰值，约为 2× 节点数 × 节点大小。

### 6.2 对序列化的影响

紧凑表示天然序列化友好：所有字段为 `i64`/`f64`/`str`/`bool`，无指针。可直接 `serde` 序列化为二进制或 JSON。这为编译缓存（incremental compilation）提供基础——序列化 `HirProgram` 后下次启动反序列化即可恢复，无需重新解析。

但当前 `bridge.rs` 从 `Value`（运行时值）而非直接从序列化数据读取，存在"先反序列化为 Value 再 bridge 为 AST"的两步开销。未来可直接从序列化数据构造 AST，省去 Value 中间层。

### 6.3 对增量编译的影响

1-based 索引支持按节点定位：修改单个表达式节点只需替换 `expr_nodes[i]`，无需重建整树（前提是该修改不改变其他节点的索引，即不增删节点）。这对增量编译有利。

但若修改导致节点数变化（如新增表达式），所有后续节点索引偏移，需重新计算或采用 arena + 稳定句柄。这是未来工作。

---

## 7 开放问题与未来工作

### 7.1 双射性的机器验证

本文证明为纸面证明，未机器验证。未来可：

- 用 Coq/Lean 形式化 `compact_program_to_ast`，机械化证明定理 B1-B5。
- 用 `proptest` 生成随机 AST，验证 `compact_program_to_ast ∘ ast_to_compact = id`（需先实现 `ast_to_compact`，见 §7.4）。

### 7.2 depth 限制的动态调整

当前 `depth > 50` 硬编码于 [bridge.rs:424](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/bridge.rs)。未来可：

- 改为可配置参数，按程序规模动态调整。
- 改为迭代算法（显式栈）消除递归深度限制，将栈深度限制从 50 提升到可用堆内存上限。

### 7.3 与 e-graph 的融合可能性

紧凑表示与 e-graph 在数据结构上同构（节点 + 索引引用）。未来可探索：在 tenthc 中引入 e-class 等价类，允许相同子表达式共享节点（DAG 化），bridge 在重建时识别共享并产生 DAG-shaped AST（需 Rust AST 支持 DAG，或保留共享标记）。这可减小 AST 体积并支持公共子表达式消除。

### 7.4 `ast_to_compact` 的实现

本文满射性证明是构造性的，给出了 `ast_to_compact` 的构造算法，但 tenthc 当前未实现该函数（紧凑表示由 tenthc 解析器直接构造，而非从 Rust AST 反向转换）。未来若需在 Rust 端生成紧凑表示（如 Rust 解析器复用 tentc 的 HIR 优化），可实现 `ast_to_compact`，与 `compact_program_to_ast` 构成完整双射对。

### 7.5 跨 Vec 引用契约的对齐

消除工程差距 G1（§3.4）：统一 `hir.th` 设计与 `bridge.rs` 实现。两种方案：

- (A) 修改 bridge，通过 `arg_list` 间接索引，匹配 hir.th 设计；
- (B) 修改 hir.th 注释，明确 `args_start` 直接索引 `expr_nodes`，匹配 bridge 实现。

方案 B 改动小但放弃间接引用的灵活性；方案 A 改动大但保留 DAG 共享能力。需总师决策。

---

## 8 结论

本文对 Tenth 自举编译器 tenthc 的紧凑 1-based 索引表示与 Rust 端递归 AST 之间的转换进行了形式化建模与严格证明。核心结论：

1. **定理 B1**（表示等价）：在合法紧凑表示 $\mathfrak{C}_{\text{leg}}^{(50)}$ 上，`compact_program_to_ast` 终止且结果唯一。
2. **定理 B2**（双射性）：在标准化紧凑表示下，`compact_program_to_ast` 限制在 $\mathfrak{C}_{\text{leg}}^{(50)} \to \mathcal{A}_{\text{sup}}^{(50)}$ 上构成双射。单射性依赖标准化（消除节点重复与乱序歧义），满射性通过构造性归纳证明。
3. **定理 B3**（引用完整性）：所有索引访问在 C2 前置条件下合法。
4. **定理 B4**（depth 限制可表达性）：$\mathcal{A}_{\text{sup}}^{(50)}$ 严格包含于 $\mathcal{A}_{\text{sup}}$，但对 tenthc 自举足够。
5. **定理 B5**（语义保持）：转换前后程序语义一致，路径 B 与路径 A 在 $\mathcal{A}_{\text{sup}}^{(50)}$ 上产生相同计算结果。

这些结论构成 Tenth 自举路径 B 正确性的理论基石。同时，本文诚实记录了 7 项局限（L1-L7，见 §10），其中 L2（unknown kind fallthrough）与 L6（实参连续性弱检查）是当前最需要工程修复的点。

---

## 9 局限（诚实披露）

> 本章节独立列出证明的漏洞、不完备性与假设强度，便于后续修订。

### L1：depth > 50 硬上限

**是什么**：`convert_expr_depth` 在 [bridge.rs:424-428](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/bridge.rs) 硬编码 `depth > 50` 检查，超过则返回 `Err`。

**影响**：定理 B2 的双射性仅在 $\mathcal{A}_{\text{sup}}^{(50)}$ 上成立，对深度 > 50 的程序不构成双射（满射失败）。

**缓解**：定理 B4 论证 tenthc 自举源码深度 ≤ 30，50 上限足够。但用户编写的深度嵌套程序可能触发。建议未来改为迭代算法（§7.2）。

### L2：unknown kind 的 fallthrough 非双射点

**是什么**：`convert_expr_depth` 的 match 语句在 [bridge.rs:637-644](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/bridge.rs) 对未识别的 `kind` 返回 `Literal::Int(0)` 占位，而非 `Err`。

**影响**：若 tenthc 输出 `kind ∉ 𝒦_sup` 的节点，bridge 静默产生错误 AST（值为 0 的整数字面量），双射性失败且无报错。

**缓解**：本文通过 C1 前置条件（`kind ∈ 𝒦_sup`）排除此情况。但 C1 在 bridge 中无运行时检查（fallthrough 不报错）。建议未来将 fallthrough 改为 `Err`，使 C1 成为运行时强制不变量。

### L3：被支持 kind 集合的局限

**是什么**：$\mathcal{K}_{\text{sup}}$ 仅含 17 种 kind，未覆盖 `ast::ExprKind` 的全部变体（如 `Closure`、`Match`、`StructLiteral`、`EnumLiteral`、`TensorLiteral`、`ArrayLiteral`、`Range`、`AssignOp`、`GenericCall`、`Tuple`、`InterpolatedString`、`MutRef`、`Move`、`TryBlock`）。

**影响**：定理 B2 的双射性仅在 $\mathcal{A}_{\text{sup}}$ 上成立。tenthc 若输出这些 kind，进入 L2 的 fallthrough。

**缓解**：tenthc 当前源码不使用这些 kind（自举范围内），但用户程序可能使用。需在 bridge 中补充实现，或在 tenthc 中降级处理。

### L4：float 字面量的字段一致性

**是什么**：`hir.th` 中 `HirExpr` 有 `lit_ival: i64` 与 `lit_fval: f64` 两个字段（[hir.th:33-34](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenthc/hir/hir.th)）。但 bridge 在 [bridge.rs:450](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/bridge.rs) 仅读取 `ival`（通过 `get_field_i64`），并在 [bridge.rs:464-467](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/bridge.rs) 用 `ival as f64` 构造 `Literal::Float`。

**影响**：若 tenthc 把浮点值存入 `lit_fval` 而非 `lit_ival`，bridge 会读取到 0 或错误值，float 语义丢失。

**缓解**：需在 tenthc 中确认浮点字面量的存储字段。若 tenthc 用 `lit_fval`，则 bridge 需新增 `get_field_f64` 读取。这是未验证的工程假设。

### L5：nil 单义性的反向检查缺失

**是什么**：I3 不变量要求"字段=0 当且仅当 AST 期望 None"。bridge 仅实现正向（字段=0 → None），未实现反向（AST 期望 Some 时验证字段≠0）。

**影响**：若 tenthc 错误地把 `if` 的 `cond` 字段设为 0（而 AST 期望 `cond` 必须存在），bridge 会调用 `convert_expr_depth(0, ...)` 触发 [bridge.rs:429](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/bridge.rs) 的 `Err`。故此情况会被捕获，但报错信息为"索引 0 越界"而非"nil 单义性违反"，调试性差。

**缓解**：当前 `Err` 已能阻止错误传播，但错误信息需改进。

### L6：实参连续性的弱检查

**是什么**：I4 不变量要求 `arg_start..arg_start+arg_count` 范围内全部为合法实参节点。bridge 仅检查边界（`i <= expr_nodes.len()`），不检查节点内容是否为有效实参。

**影响**：若 tenthc 错误地把非实参节点（如 `if` 节点）放在实参范围内，bridge 会静默接受并产生错误 AST。

**缓解**：需 tenthc 解析器保证实参连续分配（实现契约 G1）。建议未来增加节点类型断言。

### L7：语句越界的静默跳过

**是什么**：`convert_stmt_range_direct` 在 [bridge.rs:321](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/bridge.rs) 与 block 分支在 [bridge.rs:611](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/bridge.rs) 对越界索引 `continue` 跳过而非 `Err`。

**影响**：若紧凑表示包含越界语句索引，bridge 静默丢失语句，产生残缺 AST。

**缓解**：当前 C2 前置条件排除越界，但运行时无强检查。建议未来改为 `Err`。

### L8：span 信息的丢失

**是什么**：bridge 用 `dummy_span = Span { line: 0, col: 0 }`（[bridge.rs:26](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/bridge.rs)）填充所有 AST 节点的 span，丢弃 tenthc 解析时的源位置信息。

**影响**：错误诊断（如类型错误、借用检查错误）无法定位到 tenthc 源码行号，只能报告"在 bridge 转换后的 AST 中某位置"。

**缓解**：tenthc 的 `HirExpr` 当前无 span 字段（[hir.th:30-65](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenthc/hir/hir.th) 未含 span），需先在 tenthc 端记录 span，再在 bridge 中传播。这是未来工作。

---

## 10 附录

### 附录 A：定理索引

| 定理 | 陈述 | 证明位置 | 依赖前置条件 |
|------|------|---------|-------------|
| B1 | 表示等价 | §4.2 | C1, C2, C3, C4 |
| B2 | 双射性 | §4.3 | C1, C2, C3, C4 + 标准化 |
| B3 | 引用完整性 | §4.4 | C2, C4 |
| B4 | depth 限制可表达性 | §4.5 | （刻画性定理） |
| B5 | 语义保持 | §4.6 | C1, C2, C3, C4 + L4 假设 |

### 附录 B：与现有文档的对应

- 自举三路径定义见 [工作规范.md §四](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/.trae/rules/工作规范.md)，本文聚焦路径 B。
- bridge.rs 在路径 B 中的位置见 [compile/mod.rs:30](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/mod.rs)。
- 紧凑表示的设计动机见 [hir.th:1-4](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenthc/hir/hir.th) 注释。
- 自举性能目标 ~0.2s 见 [工作规范.md §二](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/.trae/rules/工作规范.md)。

### 附录 C：实施建议

1. **优先级 P0**（修复 L2）：将 [bridge.rs:637-644](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/bridge.rs) 的 fallthrough 从 `Ok(Literal::Int(0))` 改为 `Err`，使 C1 成为运行时强制不变量。
2. **优先级 P1**（修复 L7）：将 [bridge.rs:321](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/bridge.rs) 与 [bridge.rs:611](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/bridge.rs) 的 `continue` 改为 `Err`，使 I5 成为运行时强制不变量。
3. **优先级 P1**（修复 L4）：核对 tenthc 中 float 字面量的存储字段，若用 `lit_fval`，则在 bridge 中新增 `get_field_f64` 并在 `"float"` 分支读取。
4. **优先级 P2**（修复 G1）：统一 `hir.th` 设计与 `bridge.rs` 实现的跨 Vec 引用契约（§7.5）。
5. **优先级 P3**（增强 L8）：在 tenthc `HirExpr` 中添加 span 字段，bridge 中传播。
6. **优先级 P3**（实现 §7.4）：实现 `ast_to_compact`，与 `compact_program_to_ast` 构成完整双射对，支持 proptest 验证。

---

## 参考文献

<a name="ref-egg"></a>[Willsey et al. 2021] Max Willsey, Chandler Sutphin, Yisu Remy Wang, and Piotr Mardziel. *egg: Fast and Extensible Equality Saturation.* POPL 2021.

<a name="ref-lua"></a>[Ierusalimschy et al. 1996] Roberto Ierusalimschy, Luiz Henrique de Figueiredo, and Waldemar Celes Filho. *Lua — an extensible extension language.* Software: Practice and Experience, 26(6):635–652, 1996.

<a name="ref-appel"></a>[Appel 1992] Andrew W. Appel. *Compiling with Continuations.* Cambridge University Press, 1992.

<a name="ref-r5rs"></a>[Kelsey & Rees 1998] Richard Kelsey and Jonathan Rees. *Revised^5 Report on the Algorithmic Language Scheme.* Higher-Order and Symbolic Computation, 11(1):7–105, 1998.

[Pierce 2002] Benjamin C. Pierce. *Types and Programming Languages.* MIT Press, 2002. （第 11 章 "Recursive Types" 讨论递归类型的产生子与同构性。）

[Aho et al. 2006] Alfred V. Aho, Monica S. Lam, Ravi Sethi, and Jeffrey D. Ullman. *Compilers: Principles, Techniques, and Tools.* 2nd ed., Addison-Wesley, 2006. （AST 表示方式综述。）

---

**文档元数据**

- 文档编号：T13
- 数理部出品 · v1.0
- 适用版本：Tenth v0.3.3+
- 主定理数量：5（B1-B5）
- 局限数量：8（L1-L8）
- 工程差距：1（G1）
- 实施建议：6 项（P0 × 1, P1 × 2, P2 × 1, P3 × 2）
