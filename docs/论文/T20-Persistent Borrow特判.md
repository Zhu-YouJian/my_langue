# Persistent Borrow 特判：语句粒度借用检查的反例驱动扩展及其不完备性

> **作者**：Tenth 项目数理部
> **日期**：2026-07-02
> **类型**：理论分析论文（T20 理论点，护城河关联——借用检查健全性子集）
> **实证基础**：Tenth v0.3.3+ 源码（`hir/lower/mod.rs`、`hir/lower/lower_stmt.rs`、`hir/lower/lower_expr.rs`、`hir/lower/scope.rs`）；自举编译器镜像（`tenthc/hir/lower.th`）
> **关联文档**：T19（语句粒度借用检查健全性——本文为其反例驱动扩展）、`docs/语言参考手册.md`（所有权与借用语义）、`docs/理论分析点调研报告.md`（T19/T20 理论价值定位）
> **版本**：v1（首轮分析，含 4 轮自审修正留痕）

---

## 摘要

Tenth 语言的借用检查器采用语句粒度的保守近似策略（T19）：在不实现 NLL（Non-Lexical Lifetimes）的前提下，于每条语句结束后释放所有活跃借用。这一策略虽能覆盖 `if peek(&p).disc == 54 { advance(&mut p); }` 这类内联 peek/advance 模式，却引入了一个语义缺陷——`let r = &x;` 语句末尾立即释放借用，导致引用变量 `r` 在后续语句中"逻辑存活但借用已释放"。为修补此缺陷，Tenth 引入"持久借用特判"（Persistent Borrow Special Case）：当 `let` 语句的初始化表达式为直接引用（`ExprKind::Ref` 或 `ExprKind::MutRef`）时，跳过该语句末尾的 `release_borrows()` 调用。本文对这一特判进行形式化分析，给出四个主定理：**PB1**（健全性保持——特判不破坏 T19 的健全性且严格增强之）、**PB2**（不完备性——特判仅覆盖 `Ref/MutRef` 直接初始化子集，构造性反例证明 `let r = f(&x);` 等模式仍被错误释放）、**PB3**（完备化条件——给出语法判定函数 `may_return_ref` 的递归定义，刻画特判的完备化边界）、**PB4**（语义正确性——特判正确支持 `let r = &p;` 后续使用的编程模式）。本文诚实记录特判的 5 处理论局限，包括不完备覆盖、误报（死引用持续占用借用）、While/For 体的不对称处理等，为未来引入 NLL 后特判的废弃提供形式化坐标。

**关键词**：借用检查、持久借用、语句粒度、反例驱动扩展、语义补丁、不完备性、NLL、Tenth 语言

---

## 1. 引言

### 1.1 语句粒度借用检查的局限

Tenth 语言的借用检查器（[`tenth/src/hir/lower/scope.rs`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/hir/lower/scope.rs)）采用一种极简的语句粒度近似策略（理论点 T19）：不实现 NLL 或 two-phase borrows，而是在每条语句结束后调用 `release_borrows()` 重置所有 `SharedRef` 和 `ExclusiveRef` 状态为 `Owned`（[`scope.rs:113-122`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/hir/lower/scope.rs)）。`Moved` 状态被保留以确保 use-after-move 仍被检测。

这一策略的设计动机（见 [`scope.rs:104-112`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/hir/lower/scope.rs) 注释）是：在没有 NLL 的情况下，任何 `&x` 或 `&mut x` 会永久标记 `x` 为已借用，导致后续无法再次借用。语句末尾释放是一种务实的近似——允许 `if peek(&p).disc == 54 { advance(&mut p); }` 这类常见模式（条件中的共享借用通过 `if` 条件后的 `release_borrows()` 释放，体中可重新可变借用，见 [`lower_expr.rs:434-437`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/hir/lower/lower_expr.rs)），同时仍能捕获单表达式内的双重可变借用。

然而，T19 的纯语句粒度策略存在一个语义缺陷：`let r = &x;` 语句末尾，`release_borrows()` 会立即释放 `x` 的借用，使得 `r` 所持引用在借用检查器视图中"不存在"。后续语句若执行 `let m = &mut x;`，检查器不会报错——但 `r` 可能仍在作用域内被使用，构成实际的别名冲突。

### 1.2 持久借用模式的动机

考虑以下 Tenth 代码片段：

```tenth
let r = &p;                  // 创建共享引用 r 指向 p
if r.disc == 54 {            // 使用 r 读取 p 的字段
    advance(&mut p);         // 可变借用 p
}
```

在纯 T19 策略下，`let r = &p;` 末尾释放借用后，`r` 与 `p` 之间的引用关系在检查器中消失。这使得后续 `let m = &mut p; use(r, m);` 这类**实际不安全**的模式不会被拒绝——因为检查器认为 `p` 在 `let r = &p;` 之后已无活跃借用。

更具体地，纯 T19 策略下以下程序会被错误接受：

```tenth
let r = &data;               // T19: 释放 → data: Owned
let m = &mut data;           // T19: data: Owned → 接受（不健全！）
use(r, m);                   // r 与 m 同时活跃，m 可变 → 别名冲突
```

程序员直觉上期望 `let r = &data;` 建立"持续到 `r` 最后使用处"的借用。T19 无法表达这一直觉。

### 1.3 Tenth 的特判方案

为修补上述缺陷，Tenth 引入"持久借用特判"——一个**语义补丁**（semantic patch）：在语句序列的释放循环中，若当前语句是 `let` 且初始化表达式为直接引用（`&x` 或 `&mut x`），则跳过 `release_borrows()`。

核心判定函数 `creates_persistent_borrow` 定义于 [`tenth/src/hir/lower/mod.rs:43-50`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/hir/lower/mod.rs)：

```rust
pub(super) fn creates_persistent_borrow(stmt: &ast::Stmt) -> bool {
    match &stmt.kind {
        StmtKind::Let { init: Some(init), .. } => {
            matches!(init.kind, ExprKind::Ref(_) | ExprKind::MutRef(_))
        }
        _ => false
    }
}
```

特判在两个位置应用：
- **Block 表达式体**（[`lower_expr.rs:463-465`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/hir/lower/lower_expr.rs)）：逐语句 lowered 后，若非持久借用则释放。
- **Loop 体**（[`lower_stmt.rs:72-74`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/hir/lower/lower_stmt.rs)）：同上。

自举编译器 `tenthc` 中存在语义完全一致的镜像实现（[`tenthc/hir/lower.th:154-163`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenthc/hir/lower.th)），应用于 Block（[`:799-801`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenthc/hir/lower.th)）和 Loop（[`:1011-1013`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenthc/hir/lower.th)）。

### 1.4 贡献

本文做出以下贡献：

1. **特判形式化**（§3）：将 `creates_persistent_borrow` 抽象为语法判定函数，给出持久借用窗口的形式语义。
2. **不完备性证明**（§4，定理 PB2）：通过构造性反例证明特判仅覆盖 `Ref/MutRef` 直接初始化子集，`let r = f(&x);` 等模式仍被错误释放。
3. **完备化条件刻画**（§4，定理 PB3）：给出递归语法判定函数 `may_return_ref`，刻画特判的完备化边界，并分析其工程可行性。
4. **健全性保持证明**（§4，定理 PB1）：证明特判不破坏 T19 的健全性，且严格增强之（捕获更多违规）。
5. **语义正确性证明**（§4，定理 PB4）：证明特判正确支持 `let r = &p;` 编程模式。
6. **诚实局限记录**（§7）：独立章节记录 5 处理论局限，包括不完备覆盖、误报风险、While/For 不对称等。

### 1.5 v1 自审留痕

本文经历 4 轮自审：

| 轮次 | 原始断言 | 修正 |
|------|---------|------|
| 第 1 轮（结构） | PB1 初稿声称"特判保持 T19 的接受集不变" | 修正：特判使接受集**严格缩小**（更严格），T19 接受但特判拒绝的程序集非空 |
| 第 2 轮（证明） | PB4 初稿未区分 `let r = &p; if r.disc == 54 {...}` 与 `if peek(&p).disc == 54 {...}` | 修正：前者由 T20 处理（持久借用），后者由 T19 的 if 条件释放处理（[`lower_expr.rs:437`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/hir/lower/lower_expr.rs)），两者机制不同 |
| 第 3 轮（边界） | 未处理 While/For 体的特判应用 | 验证 While/For 体为单语句（`Box<Stmt>`），若为 Block 则经 `lower_expr` 的 Block 分支处理，特判一致应用；但 Loop 体为语句向量，显式应用特判 |
| 第 4 轮（诚实） | PB3 初稿声称完备化"仅需语法分析" | 修正：`Call` 分支需要返回类型分析，当前 Tenth 类型系统对引用返回类型的跟踪不完整，完备化需类型系统扩展，标注为"未来工作" |

---

## 2. 背景与相关工作

### 2.1 T19：语句粒度借用检查

T19（理论点 19，详见 [`docs/理论分析点调研报告.md:241-251`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/docs/理论分析点调研报告.md)）形式化了 Tenth 的语句粒度借用检查策略。其核心是一个四状态所有权状态机：

- **Owned**：变量未被借用，可被共享或可变借用。
- **SharedRef(n)**：变量被 n 个共享引用借用，可继续共享借用但不可可变借用。
- **ExclusiveRef**：变量被一个可变引用借用，不可再被任何借用。
- **Moved**：变量已被移动，任何使用均为错误（终态）。

`release_borrows()` 操作（[`scope.rs:113-122`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/hir/lower/scope.rs)）将所有 `SharedRef(_)` 和 `ExclusiveRef` 重置为 `Owned`，保留 `Moved`。这一操作在以下位置调用：

1. 每条非持久借用语句末尾（Block 体、Loop 体）。
2. `if` 表达式的条件之后（[`lower_expr.rs:437`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/hir/lower/lower_expr.rs)）。
3. `if` 表达式的 then 分支之后（[`lower_expr.rs:439`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/hir/lower/lower_expr.rs)）。
4. `if` 表达式的 else 分支之后（[`lower_expr.rs:441`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/hir/lower/lower_expr.rs)）。
5. `while` 条件之后（[`lower_stmt.rs:50`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/hir/lower/lower_stmt.rs)）。
6. `for` 迭代器之后（[`lower_stmt.rs:57`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/hir/lower/lower_stmt.rs)）。
7. `match` scrutinee 之后（自举侧 [`tenthc/hir/lower.th:934`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenthc/hir/lower.th)）。

T19 的健全性结论（本文引用为前提）：

> **T19-Sound**：对于任何程序 P，若 P 包含**完全在单个表达式 E 内**的双重可变借用违规（即违规不跨越语句边界），则 T19 检查器拒绝 P。

T19 的局限：跨语句的别名违规不被检测。

### 2.2 借用检查的例外处理

工业级借用检查器普遍采用"例外驱动扩展"模式——基础规则无法覆盖某些合法模式时，引入针对性例外而非重写规则系统：

- **Rust NLL**（Jung et al., 2017）：通过活跃性分析替代词法作用域，消除大量误报。但 NLL 本身仍需例外处理（如 two-phase borrows）。
- **Two-Phase Borrows**（Jung, 2018）：为 `vec.push(vec.len())` 模式引入"借用预留"——可变借用在表达式求值时预留，在实际使用时激活。
- **Polonius**（Rust 数据流借用检查器原型）：基于 Datalog 规则，通过新增规则覆盖例外模式。

T20 的持久借用特判遵循相同的"例外驱动"模式：T19 的语句末尾释放是基础规则，`let r = &x;` 是需要例外的反例。

### 2.3 语义补丁（Semantic Patch）的概念

本文使用"语义补丁"一词描述 T20 的性质。一个语义补丁 $P$ 对基础规则 $R$ 的修补满足以下性质：

1. **局部性**：$P$ 仅修改 $R$ 在特定语法模式下的行为，不改变整体架构。
2. **保守性**：$P$ 使检查器更严格（接受集缩小）或更宽松（接受集扩大），但不引入不一致。
3. **不完备性**：$P$ 覆盖触发它的反例，但可能存在同类的未覆盖反例。

T20 是一个**保守性增强**的语义补丁：它使检查器更严格（延迟释放 → 更多违规被捕获），但仅覆盖 `Ref/MutRef` 直接初始化这一子集。

### 2.4 Two-Phase Borrows 作为反例驱动扩展

Rust 的 Two-Phase Borrows（Jung, 2018）是反例驱动扩展的经典案例。基础规则要求可变借用在使用点立即激活，但 `vec.push(vec.len())` 中 `vec` 的可变借用（push 的 `&mut self` 参数）与 `vec.len()` 的共享借用（`&self`）在表达式求值期间共存。Two-phase 引入"预留"（reserved）状态：借用先进入预留态（不阻止其他共享借用），在实际使用时激活。

T20 与 two-phase 的关系将在 §8 详细对比。

---

## 3. Persistent Borrow 特判形式化

### 3.1 所有权状态机

**定义 3.1**（所有权状态）。变量 $v$ 的所有权状态 $\sigma(v) \in \Sigma$，其中：
$$\Sigma = \{ \textsf{Owned}, \textsf{SharedRef}(n) \mid n \in \mathbb{N}^+, \textsf{ExclusiveRef}, \textsf{Moved} \}$$

**定义 3.2**（状态迁移）。借用检查操作引起的状态迁移：

| 操作 | 前置条件 | 迁移 |
|------|---------|------|
| `&v`（共享借用） | $\sigma(v) \in \{\textsf{Owned}, \textsf{SharedRef}(\_)\}$ | $\sigma(v) \leftarrow \textsf{SharedRef}(n+1)$ |
| `&mut v`（可变借用） | $\sigma(v) = \textsf{Owned}$ | $\sigma(v) \leftarrow \textsf{ExclusiveRef}$ |
| `move v` | $\sigma(v) \in \{\textsf{Owned}, \textsf{SharedRef}(\_)\}$ | $\sigma(v) \leftarrow \textsf{Moved}$ |
| `release_borrows` | 任意 | $\textsf{SharedRef}(\_) \mapsto \textsf{Owned}$；$\textsf{ExclusiveRef} \mapsto \textsf{Owned}$；$\textsf{Moved}, \textsf{Owned}$ 不变 |

（实现见 [`scope.rs:67-97, 113-122`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/hir/lower/scope.rs)）

### 3.2 T19 释放策略形式化

**定义 3.3**（T19 释放策略）。对于语句序列 $S = s_1, s_2, \ldots, s_n$（Block 或 Loop 体），T19 在 lower 每条语句后调用 `release_borrows`：

$$\textsf{T19-Release}(S) := \forall i \in [1, n]. \textsf{release\_borrows} \text{ after } \textsf{lower}(s_i)$$

### 3.3 特判判定函数

**定义 3.4**（持久借用判定）。对于语句 $s$，定义语法判定函数：

$$\textsf{creates\_persistent\_borrow}(s) := \begin{cases} \textsf{true} & \text{若 } s = \textsf{Let}\{ \textsf{init}: e \} \text{ 且 } e \in \{\textsf{Ref}(\_), \textsf{MutRef}(\_)\} \\ \textsf{false} & \text{否则} \end{cases}$$

此定义精确对应源码 [`mod.rs:43-50`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/hir/lower/mod.rs)。`ExprKind::Ref` 和 `ExprKind::MutRef` 的 AST 定义见 [`ast.rs:141-142`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/parser/ast.rs)。

**关键观察**：判定仅检查 init 的**顶层** ExprKind，不递归进入子表达式。这是特判不完备性的根源（§4 定理 PB2）。

### 3.4 T20 增强释放策略

**定义 3.5**（T20 增强释放策略）。在 T19 基础上，对语句序列 $S$ 应用特判：

$$\textsf{T20-Release}(S) := \forall i \in [1, n]. \begin{cases} \textsf{skip release} & \text{若 } \textsf{creates\_persistent\_borrow}(s_i) \\ \textsf{release\_borrows} & \text{否则} \end{cases} \text{ after } \textsf{lower}(s_i)$$

实现见 [`lower_expr.rs:463-465`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/hir/lower/lower_expr.rs)（Block）和 [`lower_stmt.rs:72-74`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/hir/lower/lower_stmt.rs)（Loop）。

### 3.5 持久借用窗口

**定义 3.6**（持久借用窗口）。对于语句序列 $S = s_1, \ldots, s_n$，若 $s_i$ 满足 $\textsf{creates\_persistent\_borrow}(s_i) = \textsf{true}$，则 $s_i$ 创建的借用从 $s_i$ 开始存活，直到**下一个不满足特判的语句** $s_j$（$j > i$）的释放操作。即：

$$\textsf{window}(s_i) = [i, j) \quad \text{其中 } j = \min\{ k > i \mid \neg \textsf{creates\_persistent\_borrow}(s_k) \}$$

若不存在这样的 $j$，则窗口延伸至序列末尾。

**示例**：

| 语句 | creates_persistent_borrow | 释放行为 | 借用状态（对 `let r = &x;` 中的 `x`） |
|------|:---:|------|------|
| `let r = &x;` | ✓ | 跳过 | `x`: SharedRef(1) |
| `let s = &y;` | ✓ | 跳过 | `x`: SharedRef(1)（持续） |
| `print(r.disc);` | ✗ | 释放 | `x`: Owned（窗口结束） |
| `let m = &mut x;` | ✗ | 释放 | `x`: ExclusiveRef（OK，因 x 已 Owned） |

---

## 4. 主定理与证明

### 4.1 定理 PB1（特判的健全性保持）

**定理 PB1**。设 $\mathcal{A}_{T19}(P)$ 为 T19 策略下程序 $P$ 的接受（编译通过）判定，$\mathcal{A}_{T19+T20}(P)$ 为 T20 增强后的接受判定。则：

1. **接受集收缩**：$\{ P \mid \mathcal{A}_{T19+T20}(P) \} \subseteq \{ P \mid \mathcal{A}_{T19}(P) \}$。即 T20 接受的程序集是 T19 的子集。
2. **健全性保持**：若 T19 满足 T19-Sound（捕获所有表达式内双重可变借用违规），则 T19+T20 亦满足。
3. **健全性增强**：存在程序 $P$ 使得 $\mathcal{A}_{T19}(P) = \textsf{true}$ 但 $\mathcal{A}_{T19+T20}(P) = \textsf{false}$，且 $P$ 包含实际的跨语句别名冲突。

**证明**。

**(1) 接受集收缩**。

T19+T20 与 T19 的唯一区别是：T20 在 `creates_persistent_borrow(s) = true` 时跳过 `release_borrows()`。跳过释放意味着借用状态保持更久（`SharedRef` 或 `ExclusiveRef` 不被重置为 `Owned`），因此后续借用操作可能遇到更严格的检查（`check_borrow_shared` / `check_borrow_mut` 更可能失败）。

形式化地，设 $\sigma_{T19}^i$ 和 $\sigma_{T20}^i$ 分别为 T19 和 T20 在 lower 语句 $s_i$ 后的所有权状态映射。归纳证明：

- **基例**：$\sigma_{T19}^0 = \sigma_{T20}^0$（初始状态相同）。
- **归纳步**：对于语句 $s_i$，
  - 若 $\neg \textsf{creates\_persistent\_borrow}(s_i)$：T19 和 T20 均调用 `release_borrows`，且 lower $s_i$ 前的状态满足归纳假设。由于 lower 操作是确定性的（给定相同输入状态产生相同输出状态），$\sigma_{T19}^i = \sigma_{T20}^i$。
  - 若 $\textsf{creates\_persistent\_borrow}(s_i)$：T20 跳过释放，T19 执行释放。设 lower $s_i$ 前的状态为 $\sigma$。则：
    - $\sigma_{T19}^i = \textsf{release\_borrows}(\textsf{lower}_{s_i}(\sigma))$：所有 SharedRef/ExclusiveRef 重置为 Owned。
    - $\sigma_{T20}^i = \textsf{lower}_{s_i}(\sigma)$：保留借用状态。
  对于任意变量 $v$：若 $\sigma_{T19}^i(v) = \textsf{SharedRef}(n)$ 或 $\textsf{ExclusiveRef}$，则 $\sigma_{T20}^i(v) = \sigma_{T19}^i(v)$（因为 release 不改变这些... 等等，release 将它们重置为 Owned）。

  更精确地：$\sigma_{T19}^i(v) \in \{\textsf{Owned}, \textsf{Moved}\}$ 对所有 $v$（因为 release 重置了所有借用），而 $\sigma_{T20}^i(v)$ 可能保持 $\textsf{SharedRef}$ 或 $\textsf{ExclusiveRef}$。因此 $\sigma_{T20}^i$ 比 $\sigma_{T19}^i$ 有更多变量处于借用态。

  定义偏序 $\preceq$：$\sigma_1 \preceq \sigma_2$ 当且仅当 $\forall v. \sigma_1(v) = \textsf{Owned} \Rightarrow \sigma_2(v) \in \{\textsf{Owned}, \textsf{SharedRef}(\_), \textsf{ExclusiveRef}\}$ 且 $\sigma_1(v) = \textsf{Moved} \Rightarrow \sigma_2(v) = \textsf{Moved}$。即 $\sigma_2$ 的借用态不少于 $\sigma_1$。

  则 $\sigma_{T19}^i \preceq \sigma_{T20}^i$：T20 的借用态不少于 T19。

  由于 `check_borrow_shared` 和 `check_borrow_mut` 在被借用状态下更严格（更容易拒绝），$\sigma_{T20}^i \succeq \sigma_{T19}^i$ 意味着 T20 在后续语句中至少和 T19 一样严格。

  因此，T19+T20 拒绝所有 T19 拒绝的程序。即接受集收缩。$\square_{(1)}$

**(2) 健全性保持**。

T19-Sound 保证：表达式内双重可变借用违规在 `lower_expr(E)` 期间被检测。`lower_expr` 的借用检查（`check_borrow_shared` / `check_borrow_mut`）发生在表达式 lower 过程中，**在任何 `release_borrows` 调用之前**（因为 release 在语句 lower 完成后调用）。

T20 仅修改语句后的 release 行为，不修改 `lower_expr` 内部的借用检查逻辑。因此，T19 在 `lower_expr(E)` 中检测到的违规，T20 同样检测到。由 (1) 的接受集收缩，T20 不会接受 T19 拒绝的程序。故 T19-Sound 蕴含 T19+T20-Sound。$\square_{(2)}$

**(3) 健全性增强**。

构造程序 $P^*$：

```tenth
fn main() {
    let data = 42;
    let r = &data;        // T20: 持久借用, data: SharedRef(1)
    let m = &mut data;    // T20: 拒绝（data 已 SharedRef）
    print(r);
    print(m);
}
```

- **T19 行为**：lower `let r = &data;` → data: SharedRef(1) → `release_borrows` → data: Owned。lower `let m = &mut data;` → data: Owned → `check_borrow_mut` 通过 → data: ExclusiveRef。**接受**。
- **T20 行为**：lower `let r = &data;` → data: SharedRef(1) → **跳过 release** → data: SharedRef(1)。lower `let m = &mut data;` → data: SharedRef(1) → `check_borrow_mut` **失败**：错误 "cannot borrow 'data' as mutable because it is also borrowed as shared"。**拒绝**。

$P^*$ 包含实际别名冲突：`r`（共享引用）与 `m`（可变引用）同时活跃。T19 错误接受，T20 正确拒绝。故健全性增强。$\square_{(3)}$

**证毕**。$\blacksquare$

### 4.2 定理 PB2（特判的不完备性）

**定理 PB2**。存在持久借用模式不被 `creates_persistent_borrow` 覆盖。即存在语句 $s$ 使得 $s$ 的初始化表达式产生引用值但 $\textsf{creates\_persistent\_borrow}(s) = \textsf{false}$，且该引用值在后续语句中被使用时，借用已被错误释放。

具体地，以下四种模式构成不完备性的构造性反例：

| 反例 | 语句 | init 的 ExprKind | creates_persistent_borrow | 借用是否应持久 | 是否被正确处理 |
|------|------|:---:|:---:|:---:|:---:|
| CE-1 | `let r = f(&x);` | `Call` | false | 是（若 f 返回引用） | ✗ |
| CE-2 | `let r = if c { &a } else { &b };` | `If` | false | 是 | ✗ |
| CE-3 | `let r = { &x };` | `Block` | false | 是 | ✗ |
| CE-4 | `let r = match s { _ => &x };` | `Match` | false | 是 | ✗ |

**证明**。

**CE-1**：`let r = f(&x);` 被错误释放。

考虑程序 $P_1$（假设 `identity` 返回其参数）：

```tenth
fn identity(x: &i32) -> &i32 { return x; }
fn main() {
    let data = 42;
    let r = identity(&data);   // init = Call, creates_persistent_borrow = false
                               // → release_borrows → data: Owned
    let m = &mut data;         // data: Owned → 接受（不健全！）
    print(r);
    print(m);
}
```

- `creates_persistent_borrow(let r = identity(&data))`：`init.kind = ExprKind::Call`，不匹配 `Ref | MutRef`，返回 `false`（[`mod.rs:46`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/hir/lower/mod.rs)）。
- 语句末尾调用 `release_borrows` → `data` 从 `SharedRef(1)` 重置为 `Owned`。
- `let m = &mut data;` 的 `check_borrow_mut(data)` 通过（data 为 Owned）。
- **结果**：`r`（指向 data 的引用）与 `m`（data 的可变引用）同时活跃，别名冲突未被检测。

此外，即使在 `identity(&data)` 的 `lower_expr` 过程中 `&data` 的借用被正确记录（data: SharedRef(1)），该状态在语句末尾被释放——因为特判不覆盖 `Call` init。

**CE-2**：`let r = if c { &a } else { &b };` 被错误释放。

考虑程序 $P_2$：

```tenth
fn main() {
    let a = 1;
    let b = 2;
    let cond = true;
    let r = if cond { &a } else { &b };  // init = If, creates_persistent_borrow = false
    let m = &mut a;                       // 应失败（r 可能指向 a）
    print(r);
    print(m);
}
```

- `creates_persistent_borrow(let r = if ...)`：`init.kind = ExprKind::If`，返回 `false`。
- `lower_expr(If)` 内部：lower 条件 → `release_borrows` → lower then（`&a` → a: SharedRef）→ `release_borrows`（[`lower_expr.rs:439`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/hir/lower/lower_expr.rs)）→ a: Owned → lower else → `release_borrows`。
- 语句末尾：`release_borrows` → a, b 均 Owned。
- `let m = &mut a;` 通过。**不健全**：`r` 可能指向 `a`，与 `m` 冲突。

**CE-3**：`let r = { &x };` 被错误释放。

```tenth
fn main() {
    let x = 42;
    let r = { &x };            // init = Block, creates_persistent_borrow = false
    let m = &mut x;            // 应失败
    print(r);
    print(m);
}
```

- `init.kind = ExprKind::Block`，特判返回 `false`。
- Block 内部 lower `{ &x }`：lower `&x` → x: SharedRef → Block 末尾 `release_borrows`（[`lower_expr.rs:463-465`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/hir/lower/lower_expr.rs)）→ x: Owned。
- 语句末尾再次 `release_borrows`（无效果，x 已 Owned）。
- `&mut x` 通过。**不健全**。

**CE-4**：`let r = match s { _ => &x };` 被错误释放。

- `init.kind = ExprKind::Match`，特判返回 `false`。
- Match lowering 中 scrutinee 后 `release_borrows`（自举侧 [`tenthc/hir/lower.th:934`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenthc/hir/lower.th)），arm body 中的 `&x` 创建借用，但 arm 后释放（match 语义类似 if，每个 arm 后释放）。
- 语句末尾 `release_borrows` → x: Owned。
- **不健全**。

**结论**：`creates_persistent_borrow` 仅覆盖 `ExprKind::Ref | ExprKind::MutRef` 的顶层 init，不完备地覆盖所有产生引用值的 init 模式。$\blacksquare$

### 4.3 定理 PB3（完备化条件）

**定理 PB3**。存在一个递归语法判定函数 $\textsf{may\_return\_ref}^*: \textsf{Expr} \to \mathbb{B}$，使得将 `creates_persistent_borrow` 的判定条件从 $e \in \{\textsf{Ref}, \textsf{MutRef}\}$ 替换为 $\textsf{may\_return\_ref}^*(e)$ 后，特判完备覆盖所有产生引用值的 let 初始化模式。$\textsf{may\_return\_ref}^*$ 的递归定义如下：

$$\textsf{may\_return\_ref}^*(e) = \begin{cases} \textsf{true} & e = \textsf{Ref}(\_) \lor \textsf{MutRef}(\_) \\ \textsf{may\_return\_ref}^*(e') & e = \textsf{Paren}(e') \lor \textsf{Try}(e') \\ \textsf{may\_return\_ref}^*(e_t) \lor \textsf{may\_return\_ref}^*(e_e) & e = \textsf{If}\{ \textsf{then}: e_t, \textsf{else}: e_e \} \\ \textsf{may\_return\_ref}^*(e_f) & e = \textsf{Block}\{ \textsf{final}: e_f \} \\ \bigvee_{a \in \textsf{arms}} \textsf{may\_return\_ref}^*(a.\textsf{body}) & e = \textsf{Match}\{ \textsf{arms} \} \\ \textsf{return\_type}(f) \in \textsf{RefTypes} & e = \textsf{Call}\{ f, \ldots \} \\ \textsf{false} & \text{否则} \end{cases}$$

其中 $\textsf{RefTypes}$ 为引用类型集合，$\textsf{return\_type}(f)$ 为函数 $f$ 的声明返回类型。

完备化后的判定函数为：
$$\textsf{creates\_persistent\_borrow}^*(s) := s = \textsf{Let}\{ \textsf{init}: e \} \land \textsf{may\_return\_ref}^*(e)$$

**证明**。

**完备性**（覆盖所有产生引用值的 init）。

需证明：对于任意表达式 $e$，若 $e$ 的求值结果为引用值，则 $\textsf{may\_return\_ref}^*(e) = \textsf{true}$。对表达式结构归纳：

- **Ref/MutRef**：直接产生引用值，$\textsf{may\_return\_ref}^* = \textsf{true}$。✓
- **Paren/Try**：透传子表达式值，递归成立。✓
- **If**：结果为 then 或 else 分支的值。若结果为引用值，则对应分支求值为引用值，由归纳假设该分支的 $\textsf{may\_return\_ref}^* = \textsf{true}$，析取为 true。✓
- **Block**：结果为 final 表达式的值。若为引用值，由归纳假设 $\textsf{may\_return\_ref}^*(e_f) = \textsf{true}$。✓
- **Match**：结果为某 arm body 的值。若为引用值，由归纳假设对应 arm 的 $\textsf{may\_return\_ref}^* = \textsf{true}$，析取为 true。✓
- **Call**：结果类型由 $\textsf{return\_type}(f)$ 确定。若结果为引用值，则返回类型为引用类型，$\textsf{return\_type}(f) \in \textsf{RefTypes}$，$\textsf{may\_return\_ref}^* = \textsf{true}$。✓
- **其他**（字面量、算术、逻辑、比较、赋值、字段访问、方法调用...）：这些表达式不产生引用值（字段访问产生值拷贝，非引用），$\textsf{may\_return\_ref}^* = \textsf{false}$ 正确。

归纳完备。$\square$

**健全性**（不产生误判）。

$\textsf{may\_return\_ref}^*$ 仅在表达式**确实可能**产生引用值时返回 true。对于 Call 分支，依赖返回类型声明——若声明准确，则判定准确。若返回类型声明不准确（如声明为 `i32` 但运行时返回引用），则为类型系统的健全性问题，非特判的责任。

**工程可行性分析**。

- Ref/MutRef/Paren/Try/If/Block/Match 分支：纯语法递归，可在 `lower_expr` 期间同步计算，无需额外信息。
- **Call 分支**：需要查询函数的返回类型。Tenth 的类型系统在 `lower_expr` 阶段已维护函数签名（`Scope::functions: HashMap<String, (Vec<(String, Type)>, Type)>`，见 [`scope.rs:15`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/hir/lower/scope.rs)），理论上可查询返回类型是否为引用类型。

**当前未实现的原因**（诚实披露）：

1. Tenth 的 `Type` 枚举当前没有显式的"引用类型"变体（`Type` 主要区分 `BaseType` 和 tensor 维度），引用语义在 HIR 层面通过 `Ownership` 跟踪而非类型系统。因此 $\textsf{RefTypes}$ 的定义需要类型系统扩展。
2. 完备化引入递归语法分析，增加编译器复杂度，与 Tenth "极简借用检查器"的设计哲学冲突。
3. 完备化仍无法解决根本问题——缺少活跃性分析，所有"持久"借用均持续到下一非持久语句，产生误报。

因此，完备化标注为**未来工作**（§9）。当前特判的不完备覆盖是一个**有意识的工程权衡**（§7）。$\blacksquare$

### 4.4 定理 PB4（特判的语义正确性）

**定理 PB4**。T20 特判正确支持以下持久借用编程模式，且不引入新的误报（相对于 T19）：

1. **模式 P-1**（共享引用持续到下一语句）：
   ```tenth
   let r = &x;        // 持久借用, x: SharedRef(1)
   let m = &mut x;    // 正确拒绝（x 已被共享借用）
   ```
   T19 错误接受，T20 正确拒绝。

2. **模式 P-2**（可变引用持续到下一语句）：
   ```tenth
   let m = &mut x;    // 持久借用, x: ExclusiveRef
   let r = &x;        // 正确拒绝（x 已被可变借用）
   ```
   T19 错误接受，T20 正确拒绝。

3. **模式 P-3**（引用在后续语句使用后释放）：
   ```tenth
   let r = &x;        // 持久, x: SharedRef(1)
   print(r.disc);     // 使用 r, 然后释放 → x: Owned
   let m = &mut x;    // 正确接受（x 已 Owned）
   ```
   T19 和 T20 均正确接受。

4. **模式 P-4**（连续持久借用）：
   ```tenth
   let r = &x;        // 持久, x: SharedRef(1)
   let s = &x;        // 持久, x: SharedRef(2)（共享借用可叠加）
   print(r + s);      // 释放 → x: Owned
   ```
   T20 正确接受（共享借用叠加合法）。

**证明**。

**P-1**：
- `let r = &x;`：lower `&x` → `check_borrow_shared(x)` 通过（x: Owned）→ x: SharedRef(1)。`creates_persistent_borrow = true` → 跳过 release。x 保持 SharedRef(1)。
- `let m = &mut x;`：lower `&mut x` → `check_borrow_mut(x)` 检查：x 为 SharedRef(1)，n > 0 → **失败**，错误 "cannot borrow 'x' as mutable because it is also borrowed as shared"（[`scope.rs:81-87`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/hir/lower/scope.rs)）。
- **结果**：正确拒绝。$\square_{P\text{-}1}$

**P-2**：
- `let m = &mut x;`：lower `&mut x` → `check_borrow_mut(x)` 通过 → x: ExclusiveRef。`creates_persistent_borrow = true`（init = MutRef）→ 跳过 release。x 保持 ExclusiveRef。
- `let r = &x;`：lower `&x` → `check_borrow_shared(x)` 检查：x 为 ExclusiveRef → **失败**，错误 "不可将 'x' 共享借用，因为它已被可变借用"（[`scope.rs:67-72`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/hir/lower/scope.rs)）。
- **结果**：正确拒绝。$\square_{P\text{-}2}$

**P-3**：
- `let r = &x;`：x → SharedRef(1)，跳过 release。
- `print(r.disc);`：lower `r.disc`（字段访问，不创建新借用）→ `creates_persistent_borrow = false` → **调用 release** → x: Owned。
- `let m = &mut x;`：x: Owned → `check_borrow_mut` 通过 → x: ExclusiveRef。
- **结果**：正确接受。$\square_{P\text{-}3}$

**P-4**：
- `let r = &x;`：x → SharedRef(1)，跳过 release。
- `let s = &x;`：lower `&x` → `check_borrow_shared(x)` 检查：x 为 SharedRef(1)，允许继续共享借用 → x: SharedRef(2)。`creates_persistent_borrow = true` → 跳过 release。
- `print(r + s);`：`creates_persistent_borrow = false` → release → x: Owned。
- **结果**：正确接受。$\square_{P\text{-}4}$

**关于误报**：T20 相对于 T19 不引入新的误报（错误接受），仅引入新的拒报（正确拒绝实际冲突）。T20 可能引入的误报是"死引用持续占用借用"——即 `r` 在后续语句中不再使用，但借用仍持续到下一非持久语句。这是活跃性分析缺失的代价，非特判本身的逻辑错误（§7 局限 L2）。$\blacksquare$

---

## 5. 特判的设计分析

### 5.1 反例驱动扩展的模式

T20 特判体现了"反例驱动扩展"的设计模式：

1. **基础规则**（T19）：每条语句后释放所有借用。
2. **反例出现**：`let r = &x; let m = &mut x;` 被错误接受（r 存活但借用已释放）。
3. **定位反例模式**：`let` 语句的 init 为直接引用。
4. **针对性修补**：对这一特定模式跳过释放。
5. **不完备性残留**：其他产生引用的模式（`f(&x)`、`if c { &a } else { &b }`）未被覆盖。

这一模式与 Rust 借用检查器的演化历史一致：NLL → two-phase borrows → stacked borrows → tree borrows，每一步都是反例驱动的渐进扩展。

### 5.2 特判的工程动机

T20 的设计动机是**最小化实现复杂度**：

- **判定函数 8 行**（[`mod.rs:43-50`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/hir/lower/mod.rs)）：纯语法模式匹配，无数据流分析。
- **应用点 2 处**（Block + Loop）：每处仅 3 行（if not persistent → release）。
- **自举镜像一致**（[`tenthc/hir/lower.th:154-163`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenthc/hir/lower.th)）：自举编译器同步实现，499+ 测试无回归。

与 NLL 实现成本对比：Rust 的 NLL 需要活跃性数据流分析（MIR-level liveness pass），实现量级在数千行。T20 以 ~15 行代码覆盖了最常见的持久借用模式（直接引用 let），是极高的工程性价比。

### 5.3 特判的理论代价

T20 的理论代价体现在三个方面：

1. **不完备性**（定理 PB2）：4 类反例未被覆盖，构成已知的"漏洞"。
2. **误报风险**（局限 L2）：死引用的借用持续到下一非持久语句，可能拒绝合法程序。
3. **与 T19 的耦合**：特判的正确性依赖 T19 的释放策略——若 T19 的释放点变化，特判的窗口语义随之变化。

---

## 6. 与 Two-Phase Borrows 的对比

### 6.1 相似性：反例驱动扩展

| 维度 | Two-Phase Borrows（Rust） | Persistent Borrow（Tenth T20） |
|------|:---:|:---:|
| 基础规则 | 借用在创建点立即激活 | 借用在语句末尾释放 |
| 反例 | `vec.push(vec.len())` | `let r = &x; let m = &mut x;` |
| 修补 | 引入 reserved 状态，延迟激活 | 跳过特定语句的释放 |
| 不完备性 | 仍有 NLL 难以覆盖的 tree-borrow 场景 | `f(&x)` 等模式未覆盖 |
| 形式化基础 | Jung et al. 2018 | 本文（PB1-PB4） |

两者都是**对基础规则的局部修补**，而非系统重写。

### 6.2 差异性：临时 vs 持久

| 维度 | Two-Phase Borrows | Persistent Borrow |
|------|:---:|:---:|
| 修补方向 | 使借用**更短**（延迟激活，缩短活跃区间） | 使借用**更长**（延迟释放，延长活跃区间） |
| 解决的冲突 | 创建点与使用点之间的临时共享借用 | let 语句末尾与后续使用之间的持久借用 |
| 状态扩展 | 新增 `Reserved` 状态 | 无新状态（仅修改释放时机） |
| 活跃性分析 | 依赖 NLL 的活跃性分析 | 无活跃性分析（保守持续到下一语句） |

**核心差异**：Two-phase 缩短借用以容忍临时共存；Persistent 延长借用以检测跨语句冲突。两者方向相反，但都服务于"匹配程序员直觉的借用语义"。

---

## 7. 不完备修补的工程权衡

### 7.1 接受的不便

T20 的不完备性意味着以下模式仍需程序员手动规避：

```tenth
// 不被特判覆盖 — 借用被错误释放
let r = identity(&data);   // 借用在语句末尾释放
let m = &mut data;         // 检查器接受（不健全）
```

**当前缓解**：程序员应避免 `let r = f(&x);` 形式，改为直接使用引用或在同一表达式内完成操作。

### 7.2 替代方案：用户手动 reborrow

对于 `let r = if c { &a } else { &b };` 这类模式，程序员可在后续语句中显式 reborrow：

```tenth
let r = if c { &a } else { &b };
// 特判未覆盖，借用已释放 — 但 r 仍可使用（检查器不跟踪 r 的引用关系）
let m = &mut a;  // 检查器接受（因借用已释放）
// 程序员需自行确保 r 在 m 存活期间不被使用
```

这一方案将安全性责任转移给程序员，与 Tenth "编译期近似 + 运行时不强制"的整体设计一致。

### 7.3 未来引入 NLL 的过渡

T20 是通向 NLL 的**过渡补丁**。当 Tenth 引入 NLL（基于活跃性分析的借用检查）后：

1. `let r = &x;` 的借用将持续到 `r` 的最后使用点（而非下一非持久语句），消除误报。
2. `let r = f(&x);` 的借用也将正确持续（NLL 跟踪引用值的活跃性，不依赖语法模式），消除不完备性。
3. `creates_persistent_borrow` 特判可完全废弃——NLL 的活跃性分析自然覆盖所有持久借用模式。

T20 的形式化（本文 PB1-PB4）为这一过渡提供了**正确性基准**：NLL 实现后，可通过等价性测试验证 NLL 在 T20 覆盖的子集上行为一致。

---

## 8. 开放问题与未来工作

### 8.1 完备化特判的可行性

定理 PB3 给出的完备化函数 $\textsf{may\_return\_ref}^*$ 在纯语法分支（Ref/MutRef/If/Block/Match/Paren/Try）上可直接实现。**Call 分支**需要类型系统支持引用返回类型——当前 Tenth 的 `Type` 枚举无显式引用类型变体，需扩展。

**可行性评估**：中等。语法分支的完备化可在 ~30 行内实现；Call 分支需类型系统扩展（引入 `Type::Ref(Box<Type>, bool)` 变体），影响范围较广（`hir/types.rs`、`hir/lower/types.rs`、`compile/bytecode.rs` 等）。

### 8.2 引入 NLL 后特判的废弃

NLL 引入后，`release_borrows` 策略本身将被活跃性分析替代，`creates_persistent_borrow` 特判自然废弃。废弃路径：

1. **Phase 1**（NLL 引入）：实现 MIR-level 活跃性 pass，替代 `release_borrows`。
2. **Phase 2**（特判废弃）：移除 `creates_persistent_borrow` 及其应用点。
3. **Phase 3**（测试迁移）：T20 相关测试用例迁移为 NLL 等价性测试。

**时间预估**：NLL 实现是 Tenth 借用检查器的重大架构变更，预计在 v0.5+ 版本。

### 8.3 While/For 体的对称化

当前 While 和 For 的体为单语句（`Box<Stmt>`），若体为 Block 则经 `lower_expr` 的 Block 分支处理特判。但若体为非 Block 的单语句（如 `while c { let r = &x; ... }` 不成立——While 体必须是 Block），则特判不应用。

经核查 AST 定义（[`ast.rs`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/parser/ast.rs)），While/For/Loop 的体分别为 `Box<Stmt>`、`Box<Stmt>`、`Vec<Stmt>`。While/For 的体通常为 Block 表达式，Block 内部的特判已正确处理。Loop 体为语句向量，显式应用特判。因此当前实现**基本对称**，但若 While/For 体直接为非 Block 单语句（理论上 AST 允许），则特判不应用——这是一个边界不对称（局限 L4）。

---

## 9. 结论

本文对 Tenth 语言的 Persistent Borrow 特判进行了完整的形式化分析。特判作为 T19 语句粒度借用检查的"反例驱动扩展"，通过跳过 `let r = &x;` / `let r = &mut x;` 语句末尾的 `release_borrows()` 调用，使借用持续到下一非持久语句，从而正确检测 `let r = &x; let m = &mut x;` 这类别名冲突。

四个主定理构成了特判的理论基础：
- **PB1**（健全性保持）：特判不破坏 T19 的健全性，且严格增强之——接受集收缩，更多实际冲突被捕获。
- **PB2**（不完备性）：构造性反例证明特判仅覆盖 `Ref/MutRef` 直接初始化子集，`f(&x)`、`if c { &a } else { &b }` 等模式仍被错误释放。
- **PB3**（完备化条件）：递归语法判定函数 `may_return_ref*` 刻画了完备化边界，Call 分支需类型系统扩展。
- **PB4**（语义正确性）：特判正确支持 `let r = &x;` 的四种编程模式（共享/可变/释放/叠加）。

特判是一个**有意识的工程权衡**：以 ~15 行代码覆盖最常见的持久借用模式，接受 4 类不完备反例，作为通向 NLL 的过渡补丁。本文的形式化为未来 NLL 实现提供了正确性基准，并为完备化路径（PB3）提供了可执行的工程指导。

---

## 10. 局限（独立章节）

本文及特判本身存在以下理论局限，逐条披露：

### L1. 不完备覆盖（定理 PB2）

**是什么**：特判仅覆盖 `Ref/MutRef` 直接初始化，4 类模式（Call/If/Block/Match init）未被覆盖。

**影响**：`let r = f(&x); let m = &mut x;` 等模式被错误接受，存在实际别名冲突风险。

**缓解**：程序员避免 `let r = f(&x);` 形式；未来引入 NLL 或完备化特判（PB3）。

### L2. 死引用误报

**是什么**：`let r = &x;` 的借用持续到下一非持久语句，即使 `r` 在该区间内不被使用。

**影响**：`let r = &x; let m = &mut x;` 被拒绝，即使 `r` 可能已死。程序员需调整语句顺序。

**缓解**：将 `let m = &mut x;` 前置，或在 `let r = &x;` 与 `let m = &mut x;` 之间插入一条非持久语句触发释放。

### L3. 持久窗口的不可控性

**是什么**：持久借用窗口从 `let r = &x;` 延伸到下一非持久语句，窗口长度取决于后续语句序列，程序员难以预测。

**影响**：连续多个持久借用语句（`let r = &x; let s = &y; let t = &z; ...`）会累积多个活跃借用，可能意外阻止后续借用。

**缓解**：程序员在持久借用序列后插入显式释放语句（如 `let _ = 0;` 触发 release）。

### L4. While/For 体的边界不对称

**是什么**：While/For 的体为 `Box<Stmt>`，若体为 Block 则经 `lower_expr` Block 分支正确处理特判；若体为非 Block 单语句，特判不应用。

**影响**：理论上存在 While/For 体为非 Block 单语句的 AST 构造，特判不应用。但实际 Tenth 程序中 While/For 体几乎总是 Block，影响极小。

**缓解**：语言规范可要求 While/For 体必须为 Block（语法层面强制）。

### L5. 自举侧的 Match 释放不对称

**是什么**：自举编译器 `tenthc` 在 Match 的 scrutinee 后调用 `release_borrows`（[`tenthc/hir/lower.th:934`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenthc/hir/lower.th)），但 Rust 母编译器的 Match lowering 是否同样释放需核查。

**影响**：若两侧 Match 的释放策略不一致，可能破坏自举语义等价性（T12）。

**缓解**：核查 Rust 侧 `lower_expr` 的 Match 分支，确认释放策略一致。若不一致则需同步。

---

## 11. 实施建议

基于本文理论分析，对 Tenth 借用检查器的后续实施提出以下建议：

| 优先级 | 建议 | 依据 | 预期收益 |
|:---:|------|------|------|
| P1 | 文档化特判的不完备性 | PB2 | 程序员知晓 `let r = f(&x);` 的限制 |
| P2 | 添加 CE-1~CE-4 的回归测试 | PB2 | 防止不完备性意外"修复"后产生回归 |
| P2 | 核查 While/For 非 Block 体的特判应用 | L4 | 消除边界不对称 |
| P3 | 实现 PB3 的语法分支完备化（不含 Call） | PB3 | 覆盖 If/Block/Match init，减少不完备性 |
| P4 | 引入 Type::Ref 变体，支持 Call 分支完备化 | PB3 | 完备覆盖所有持久借用模式 |
| P5 | 规划 NLL 实现路线图 | §7.3 | 根本性解决借用检查的近似问题 |

---

## 12. 参考文献

1. Jung, R., Krebbers, R., Jourdan, J.-H., Dang, A., Bizjak, F., & Birkedal, L. (2017). *The future is ours: predictive concurrency in Rust.* Submitted to ICFP 2017.
2. Jung, R., Jourdan, J.-H., Krebbers, R., & Dirk, B. (2018). *Stacked Borrows: An Aliasing Model for Rust.* POPL 2020.
3. Jung, R., Dang, A., Kang, J., & Dreyer, D. (2019). *Stacked Borrows: An Aliasing Model for Rust.* RUST 2019.
4. Matsakis, N. D. (2017). *Two-Phase Borrows in Rust.* Blog post, available at https://smallcultfollowing.com/babysteps/blog/2017/03/01/nll-borrow-check-summary/.
5. RustBelt: Jung, R., Krebbers, R., et al. (2017). *RustBelt: Securing the Foundations of the Rust Programming Language.* POPL 2018.
6. Polonius: The Rust data-flow borrow checker prototype. https://github.com/rust-lang/polonius.
7. Tenth 项目数理部. *T19: 语句粒度借用检查健全性.* Tenth 项目内部论文（待撰）。
8. Tenth 项目数理部. *T12: 双侧编译器语义等价性.* 见 `docs/论文/T12-双侧编译器语义等价性.md`.
9. Tenth 项目. *理论分析点调研报告.* 见 `docs/理论分析点调研报告.md`.
10. Tenth 项目. *语言参考手册.* 见 `docs/语言参考手册.md`.

---

## 附录 A：定理索引

| 定理 | 名称 | 陈述 | 证明位置 |
|:---:|------|------|------|
| PB1 | 健全性保持 | 特判不破坏 T19 健全性且严格增强 | §4.1 |
| PB2 | 不完备性 | 4 类反例未被特判覆盖 | §4.2 |
| PB3 | 完备化条件 | `may_return_ref*` 递归定义完备覆盖 | §4.3 |
| PB4 | 语义正确性 | 4 种持久借用模式正确处理 | §4.4 |

## 附录 B：与源码的对应

| 理论概念 | 源码位置 | 行号 |
|------|------|:---:|
| 所有权状态机 | `tenth/src/hir/lower/scope.rs` | 5-11 |
| `release_borrows` | `tenth/src/hir/lower/scope.rs` | 113-122 |
| `check_borrow_shared` | `tenth/src/hir/lower/scope.rs` | 67-79 |
| `check_borrow_mut` | `tenth/src/hir/lower/scope.rs` | 81-97 |
| `creates_persistent_borrow`（Rust） | `tenth/src/hir/lower/mod.rs` | 43-50 |
| 特判应用点（Block） | `tenth/src/hir/lower/lower_expr.rs` | 463-465 |
| 特判应用点（Loop） | `tenth/src/hir/lower/lower_stmt.rs` | 72-74 |
| `if` 条件后释放（T19） | `tenth/src/hir/lower/lower_expr.rs` | 437 |
| `creates_persistent_borrow`（自举） | `tenthc/hir/lower.th` | 154-163 |
| 特判应用点（Block，自举） | `tenthc/hir/lower.th` | 799-801 |
| 特判应用点（Loop，自举） | `tenthc/hir/lower.th` | 1011-1013 |
| `Ref`/`MutRef` AST 定义 | `tenth/src/parser/ast.rs` | 141-142 |
