# Panic-Mode 错误恢复的完备性：Tenth SYNC_TOKENS 策略的形式化与双侧对比

> 系列编号：T25 | 难度：硕士级 | 子领域：编译前端 / 错误恢复 / 双侧编译器等价性
> 关联论文：T12《双侧编译器语义等价性》（L2 错误恢复层）

---

## 摘要

本文对 Tenth 语言 Rust 母编译器的 panic-mode 错误恢复策略进行形式化分析。该策略以 `SYNC_TOKENS = {Fn, Struct, Enum, Trait, Impl, Mod, Use, RBrace, Eof}` 为同步点，在解析失败后跳过非同步 token 直到下一个项边界。本文给出五条主定理：(R1) 捕获集刻画——精确描述可恢复错误类的语法特征；(R2) SYNC_TOKENS 覆盖性——证明该集合覆盖全部 7 个顶级 item 起始符；(R3) 终止性——`synchronize` 必然在至多 $O(n)$ 步内终止；(R4) 健全性——恢复后的解析树是合法语法树的子集近似；(R5) 双侧不等价——通过构造性反例证明 tenthc 自举 parser 完全缺失错误恢复，导致 T12 定义的 L2 错误行为层在"需要恢复的输入"上不成立等价。本文诚实记录 panic-mode 的固有局限（无法 phrase-level 恢复、可能丢失错误细节、双侧不对称未修补），并将 tenthc 错误恢复补全列为未来工作。

**关键词**：错误恢复、panic-mode、同步 token、双侧编译器、自举、parser 完备性、Tenth 语言、构造性反例

---

## 1. 引言

### 1.1 错误恢复的挑战

编译器前端的错误恢复是一个经典而棘手的问题。理想的恢复策略应当同时满足三个相互冲突的目标：

1. **健壮性**：在任何畸形输入上不崩溃、不无限循环；
2. **信息量**：尽可能多地报告独立错误，而非在首个错误后停止；
3. **保真度**：恢复后的语法树尽可能接近用户意图。

这三者难以兼得：激进的恢复可能引入"幻觉错误"（基于错误假设的虚假诊断），保守的恢复可能漏报后续错误。Aho 等人在 Dragon Book 中将主流策略分为四类：panic-mode、phrase-level、error production、error correction[^1]。

### 1.2 Panic-Mode 的经典性

Panic-mode 是最简单也最健壮的策略：遇到错误时，丢弃输入 token 直到遇到预先选定的"同步点"。它的优势在于：

- **实现简单**：仅需一个同步 token 集合与一个 skip 循环；
- **永不死循环**：同步点必然包含输入终止符；
- **多错误收集**：可在每个同步点后重启解析，收集多个错误。

其代价是：恢复粒度粗（item 级而非表达式级）、可能丢失错误细节、对深层嵌套结构恢复不佳。许多工业级编译器（rustc、clang）采用 panic-mode 为底座，叠加 phrase-level 与 error production 增强。

### 1.3 Tenth 的 SYNC_TOKENS 策略

Tenth Rust 母编译器采用经典 panic-mode。其同步 token 集合定义于 [`tenth/src/parser/parser.rs:15-25`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/parser/parser.rs)：

```rust
const SYNC_TOKENS: &[TokenKind] = &[
    TokenKind::Fn, TokenKind::Struct, TokenKind::Enum,
    TokenKind::Trait, TokenKind::Impl, TokenKind::Mod,
    TokenKind::Use, TokenKind::RBrace, TokenKind::Eof,
];
```

恢复函数 `synchronize` ([parser.rs:120-129](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/parser/parser.rs)) 实现为：循环 peek，若当前 token 属于 SYNC_TOKENS 则停止；若到 Eof 则停止；否则前进一位。带恢复的解析入口 `parse_program_with_recovery` ([parser.rs:141-188](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/parser/parser.rs)) 在每个 item 解析失败的分支调用 `synchronize`，将错误压入 `errors` 向量后继续主循环。

与之对照，tenthc 自举 parser ([`tenthc/parser/parser.th`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenthc/parser/parser.th)) **完全没有任何错误恢复机制**——所有 parse_* 函数返回值而非 `Result`，遇到意外 token 时仅通过 `parser_advance` 推进或在 `parse_primary` 的兜底分支返回一个伪 `int 0` 节点（[parser.th:446-449](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenthc/parser/parser.th)）。

### 1.4 贡献

本文做出四点贡献：

1. **捕获集形式化**：将 panic-mode 可恢复的错误类刻画为"通过 `?` 算子从 `parse_item` 子树传播出来的 `TenthError::ParseError`"，给出该类的语法特征与不可恢复类的反例。
2. **覆盖性证明**：证明 `SYNC_TOKENS` 覆盖全部 7 个顶级 item 起始符，且 `RBrace`/`Eof` 的纳入是必要的；同时识别 `Pub`/`Semicolon`/`RParen` 的有意省略及其后果。
3. **双侧不等价的构造性反例**：给出一个仅使用双侧共同词法/语法子集的输入，证明 Rust parser 与 tenthc parser 在该输入上产生不同的 AST 与错误集——精确将 T12 定义的 L2 等价失效边界定位到"需要错误恢复的输入"。
4. **最小修补集**：列出消除双侧不对称所需的最小修改清单，与 T12 §9 的修补表对齐。

---

## 2. 背景与相关工作

### 2.1 错误恢复策略分类

按 Dragon Book[^1] 的经典分类：

| 策略 | 思路 | 优势 | 劣势 |
|------|------|------|------|
| **Panic-mode** | 跳过到同步点 | 简单、健壮、不死循环 | 粒度粗、丢细节 |
| **Phrase-level** | 在错误点局部回退/插入/删除 | 粒度细、保真度高 | 实现复杂、易幻觉错误 |
| **Error production** | 在文法中预声明常见错误产生式 | 错误信息质量高 | 仅覆盖预想错误 |
| **Error correction** | 推断用户意图并修正 | 用户体验最好 | 修正可能错误、不可靠 |

### 2.2 LLVM clang 的错误恢复

clang 采用"**多层叠加**"策略[^2]：

- **基础层**：panic-mode，以 `;`、`}`、顶级声明关键字为同步点；
- **中间层**：`Parser::ExpectAndConsume` 在期望 token 缺失时尝试插入（保留 `}` 平衡）；
- **高层**：`Parser::Parse GNU` / `Parse Microsoft` 等 extension 表驱动的 error production；
- **AST 层**：用 `RecoveryExpr` 节点占位，保持 AST 结构完整以便下游分析。

clang 的恢复可在函数体内任意嵌套层级重启，错误信息质量极高。代价是 `Parser` 实现超过 8 万行，恢复逻辑遍布各处。

### 2.3 rustc 的错误恢复

rustc 的 `librustc_parse` 采用 panic-mode 为底座，叠加：

- **Token 插入**：`expect` 失败时根据上下文插入虚拟 token（如缺失的 `;`、`,`），记录为"added"诊断；
- **分隔符匹配**：`match` 臂、`struct` 字段等的分隔符错误时局部回退；
- **`Recovery` 状态**：parser 携带 `Recovery` 字段，标记"已恢复的括号缺失"等状态供后续判断；
- **`Error` 节点**：AST 中以 `Err` 变体占位，HIR 中以 `ExprKind::Err` 传递到 type checker。

rustc 的恢复侧重**错误信息质量**与**避免级联错误**，复杂度高于 panic-mode 但低于 clang。

### 2.4 GCC 的错误恢复

GCC 历史上采用 YACC 生成的 LALR(1) parser，错误恢复依赖 YACC 的 `error` 产生式（一种 error production 策略）。C++ 前端 (`cp/parser.c`) 后来手写了 panic-mode 与 phrase-level 混合策略，但其复杂度同样高达数万行。

### 2.5 双侧编译器的错误恢复对称性

文献中关于"同一语言的双侧编译器（host 与 self-host）的错误恢复等价性"分析极少。T12 是首批形式化讨论此问题的论文之一，提出 L1–L4 四级等价标准，其中 L2 专门刻画错误行为（错误集等价）。本文延续 T12 的框架，深入分析 Tenth 双侧 parser 的 L2 等价边界。

---

## 3. Panic-Mode 恢复形式化

### 3.1 基本记号

设输入 token 序列为 $T = \langle t_0, t_1, \dots, t_{n-1} \rangle$，其中 $t_i \in \mathcal{K}$（TokenKind 枚举的全集）。记 $|T| = n$。设 EOF 为特殊的 TokenKind，且对 $i \geq n$，$t_i = \text{EOF}$（与 [`parser.rs:27-30`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/parser/parser.rs) 的 `EOF_TOKEN` 一致）。

记 parser 状态为 $\sigma = (\text{pos}, \text{errors}, \text{items})$，初始 $\sigma_0 = (0, \emptyset, \emptyset)$。

### 3.2 SYNC_TOKENS 集合定义

**定义 3.1**（SYNC_TOKENS）。
$$\mathcal{S} := \{\text{Fn}, \text{Struct}, \text{Enum}, \text{Trait}, \text{Impl}, \text{Mod}, \text{Use}, \text{RBrace}, \text{Eof}\}$$

源码见 [parser.rs:15-25](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/parser/parser.rs)。$|\mathcal{S}| = 9$。

### 3.3 synchronize 函数的算法

`synchronize` ([parser.rs:120-129](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/parser/parser.rs)) 的形式语义：

$$
\text{sync}(\sigma) := \min\{k \geq \text{pos} \mid t_k \in \mathcal{S}\}
$$

即将 parser 的 pos 推进到不小于当前 pos 的、第一个属于 $\mathcal{S}$ 的 token 位置。实现为：

```rust
fn synchronize(&mut self) {
    loop {
        let kind = self.peek_kind();
        if SYNC_TOKENS.iter().any(|t| discriminant(kind) == discriminant(t)) { break; }
        if self.at_eof() { break; }
        self.pos += 1;
    }
}
```

**注**：`at_eof()` 检查与 `SYNC_TOKENS` 含 `Eof` 的检查冗余（peek 在 pos ≥ n 时返回 EOF_TOKEN，其 kind ∈ $\mathcal{S}$）。这是防御性设计，不改变终止性结论。

### 3.4 parse_program_with_recovery 的恢复流程

`parse_program_with_recovery` ([parser.rs:141-188](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/parser/parser.rs)) 的算法骨架：

```
输入: T = <t_0, ..., t_{n-1}>
输出: (Program, Vec<TenthError>)
1. items ← ∅; errors ← ∅; main_expr_stmts ← ∅
2. while pos < n:
3.   match parse_item():
4.     Ok(item)  => items ← items ∪ {item}
5.     Err(e)    => errors ← errors ∪ {e}; synchronize()
6. return (Program{items}, errors)
```

**关键性质**：
- 主循环每次迭代，pos 严格递增（`parse_item` 至少消费一个 token，或在 `Err` 分支后 `synchronize` 至少推进到下一个 sync token，最坏推进到 Eof）。
- 错误以 `TenthError::ParseError` 形式压入 `errors` 向量，最终返回给调用者。
- 已成功解析的 item 不会被回滚——这是 panic-mode 的"向前恢复"语义。

### 3.5 捕获集形式化定义

**定义 3.2**（捕获集 $\mathcal{C}$）。
称一个错误 $e$ 属于捕获集 $\mathcal{C}$，当且仅当存在某个 $\text{parse_item}$ 调用 $P$ 使得：
1. $P$ 在某个子解析函数（如 `parse_expr`、`parse_type`、`parse_param`、`parse_block_stmts`、`expect`、`expect_ident` 等）中通过 `?` 算子传播 $e$；
2. $e$ 是 `TenthError::ParseError` 变体；
3. $P$ 将 $e$ 作为 `Err(e)` 返回给 `parse_program_with_recovery`。

形式化：$\mathcal{C} := \{e \mid e \text{ 经由 } \texttt{parse\_item} \text{ 的 } \texttt{?} \text{ 传播路径返回}\}$。

直观解释：捕获集是"panic-mode 能感知并恢复的错误类"。任何不通过此路径返回的错误都不在捕获集内。

---

## 4. 主定理与证明

### 4.1 定理 R1（捕获集刻画）

**定理 R1**. 一个错误 $e$ 属于 $\mathcal{C}$ 当且仅当 $e$ 满足以下两条件：
1. （类型条件）$e$ 是 `TenthError::ParseError { line, col, message }` 变体；
2. （传播条件）$e$ 在 $\text{parse_item}$ 调用栈中由某个 `expect` / `expect_ident` / `expect_gt` / `parse_type` / `parse_expr` / `parse_param` / `parse_block_stmts` / `parse_match_pattern` 等子函数显式构造并通过 `?` 上抛。

且下列错误**不属于** $\mathcal{C}$：
- (a) Lexer 阶段的错误（在 `tokens` 进入 parser 之前已处理，不经过 `parse_item`）；
- (b) `parse_program_with_recovery` 主循环中 `<expr>` 处理逻辑的 panic（无 `?` 传播，主循环不返回 `Err`）；
- (c) `has_main_fn` 检测中的字段访问错误（不构造 `TenthError`）；
- (d) 程序运行时的 panic（如 `unreachable!`、数组越界）；
- (e) 语义错误（类型不匹配、借用违规）——这些在 HIR 阶段才出现，不属于解析错误。

**证明**.

（$\Rightarrow$）设 $e \in \mathcal{C}$。由定义 3.2，$e$ 经由 `parse_item` 的 `?` 传播路径返回。检查 `parse_item` 实现（[parser.rs:1501-1746](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/parser/parser.rs)），其 `?` 出现于：
- `parse_generic_params()` (L1526, L1570, L1635, L1682)
- `expect(LParen)` (L1527)
- `parse_param()` (L1531)
- `expect(RParen)` (L1542)
- `parse_type()` (L1546, L1576, L1604, L1613, L1711, L1128 等多处)
- `parse_block_or_expr()` / `parse_block_stmts()` (L1551)
- `expect(LBrace)` (L1571, L1589, L1655, L1683)
- `expect(RBrace)` (L1581, L1628, L1649, L1660, L1724)
- `expect_ident()` (L1569, L1587, L1634, L1654, L1690, L1696, 等)
- `expect_gt()` (L1172, L1784 等)
- `parse_expr()` (L1551, L1729, L233, L269 等)

每一个 `?` 都要求上游返回 `TenthResult<_> = Result<_, TenthError>`。检查 `TenthError` 枚举（在 `tenth/src/error.rs`），可知 parser 阶段只构造 `TenthError::ParseError` 变体。故条件 (1) 与 (2) 成立。

（$\Leftarrow$）设 $e$ 满足 (1)(2)。则 $e$ 在某子函数中构造为 `Err(TenthError::ParseError{...})`，通过 `?` 上抛到 `parse_item`，再由 `parse_item` 的某个 `?` 上抛到 `parse_program_with_recovery` 主循环。在主循环中（[parser.rs:147-168](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/parser/parser.rs)）落入 `Err(err)` 分支，被压入 `errors`。故 $e \in \mathcal{C}$。

（不可恢复类）
- (a) Lexer 错误在 `lexer.tokenize()` 返回前已处理（调用方 `parse_program_with_recovery` 接收的是已 tokenize 完的 `Vec<Token>`），不进入 parser，故不属于 $\mathcal{C}$。
- (b) 主循环的 `<expr>` 处理逻辑（[parser.rs:149-161](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/parser/parser.rs)）使用模式匹配与字段访问，不构造 `TenthError`，故不会进入 $\mathcal{C}$。
- (c) `has_main_fn` 检测（L160）仅设置布尔标志，不构造错误。
- (d) Rust panic（如 `unreachable!()` 在 [parser.rs:963](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/parser/parser.rs) `parse_binary` 的 `unreachable!()`）绕过 `Result` 机制，不属于 $\mathcal{C}$。
- (e) HIR 阶段错误在 `hir/lower.rs` 中构造，与 parser 无关。

$\square$

**定理 R1 的实践意义**：panic-mode 仅捕获"语法结构错误"，不捕获"词法错误"、"运行时 panic"、"语义错误"。这是 panic-mode 的固有边界，与 clang/rustc 的"多层叠加"策略形成对比——后两者通过 HIR/type checker 阶段的恢复机制扩展了捕获范围。

### 4.2 定理 R2（SYNC_TOKENS 覆盖性）

**定理 R2**. 设顶级 item 起始符集合为 $\mathcal{I}$。则 $\mathcal{I} \subseteq \mathcal{S}$，即 `SYNC_TOKENS` 覆盖所有顶级 item 起始符。

**证明**.

首先枚举 $\mathcal{I}$。由 `parse_program` ([parser.rs:1448-1499](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/parser/parser.rs)) 与 `parse_item` ([parser.rs:1501-1746](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/parser/parser.rs)) 的实现，顶级 item 的起始 TokenKind 为：

| item 类型 | 起始 TokenKind | 源码位置 |
|-----------|---------------|---------|
| 函数 | `Fn` | parser.rs:1511 |
| 结构体 | `Struct` | parser.rs:1567 |
| 枚举 | `Enum` | parser.rs:1585 |
| 实现 | `Impl` | parser.rs:1632 |
| 模块 | `Mod` | parser.rs:1652 |
| Use 声明 | `Use` | parser.rs:1663 |
| Trait | `Trait` | parser.rs:1679 |
| （修饰符） | `Pub` | parser.rs:1504 |

故 $\mathcal{I} = \{\text{Fn}, \text{Struct}, \text{Enum}, \text{Trait}, \text{Impl}, \text{Mod}, \text{Use}, \text{Pub}\}$，$|\mathcal{I}| = 8$。

逐一验证：
- $\text{Fn} \in \mathcal{S}$ ✓ ([parser.rs:16](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/parser/parser.rs))
- $\text{Struct} \in \mathcal{S}$ ✓ ([parser.rs:17](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/parser/parser.rs))
- $\text{Enum} \in \mathcal{S}$ ✓ ([parser.rs:18](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/parser/parser.rs))
- $\text{Trait} \in \mathcal{S}$ ✓ ([parser.rs:19](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/parser/parser.rs))
- $\text{Impl} \in \mathcal{S}$ ✓ ([parser.rs:20](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/parser/parser.rs))
- $\text{Mod} \in \mathcal{S}$ ✓ ([parser.rs:21](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/parser/parser.rs))
- $\text{Use} \in \mathcal{S}$ ✓ ([parser.rs:22](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/parser/parser.rs))
- $\text{Pub} \in \mathcal{S}$ ✗（**有意省略**）

前 7 个 item 起始符均属于 $\mathcal{S}$。`Pub` 不属于 $\mathcal{S}$，但 `Pub` 仅作为修饰符前缀，必后跟 7 个 item 起始符之一（由 `parse_item` [L1504-1509](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/parser/parser.rs) 的逻辑保证）。当 `synchronize` 在 `Pub` 处不停止时，下一轮 peek 必然落到 `Fn`/`Struct`/.../`Trait` 之一（属于 $\mathcal{S}$），故仍能正确恢复到 item 边界。

因此 $\mathcal{I} \setminus \{\text{Pub}\} \subseteq \mathcal{S}$，且 `Pub` 的省略不破坏恢复能力。

$\square$

**推论 R2.1**. `SYNC_TOKENS` 中 $\{\text{RBrace}, \text{Eof}\}$ 的纳入是必要的：
- `Eof` 保证 `synchronize` 在输入耗尽时终止；
- `RBrace` 允许在嵌套结构（impl/mod/trait/struct/enum 内部）出错时恢复到外层 item 的下一个成员，而非跳过整个外层块。

### 4.3 定理 R3（恢复的终止性）

**定理 R3**. 对任意输入 token 序列 $T$ 与任意初始状态 $\sigma = (\text{pos}, \cdot, \cdot)$，`synchronize` 函数必然在至多 $|T| - \text{pos}$ 次循环迭代内终止，且终止时 $\text{pos}' \leq |T|$。

**证明**.

考察 `synchronize` 的循环（[parser.rs:120-129](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/parser/parser.rs)）：

```
loop {
    if peek_kind() ∈ S: break
    if at_eof(): break
    pos += 1
}
```

定义循环不变量 $I(k)$：第 $k$ 次迭代开始时，$\text{pos} = \text{pos}_0 + k$，其中 $\text{pos}_0$ 为初始 pos。

- **基础**：$k=0$，$\text{pos} = \text{pos}_0$。$I(0)$ 成立。
- **归纳**：设 $I(k)$ 成立。若本轮 break，循环终止，结论成立。否则执行 `pos += 1`，故 $\text{pos} = \text{pos}_0 + k + 1$，即 $I(k+1)$ 成立。

**终止性**：考察 $k = |T| - \text{pos}_0$（假设 $\text{pos}_0 \leq |T|$）。此时 $\text{pos} = |T|$。`peek()` 调用 `tokens.get(pos).unwrap_or(&EOF_TOKEN)` ([parser.rs:41](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/parser/parser.rs))，因 pos ≥ n，返回 `EOF_TOKEN`，其 kind = `Eof`。`SYNC_TOKENS` 含 `Eof`，故第一个 if 触发 break。

因此循环至多在 $k = |T| - \text{pos}_0$ 次迭代后终止，且终止时 $\text{pos}' \leq |T|$（若提前遇到其他 sync token 则更早终止）。

$\square$

**推论 R3.1**. `parse_program_with_recovery` 主循环必然终止。

**证明**. 主循环每次迭代，pos 严格递增：要么 `parse_item` 成功消费至少 1 个 token，要么 `Err` 分支后 `synchronize` 至少将 pos 推进到下一个 sync token（由 R3，至少推进到 Eof）。由 R3 知 `synchronize` 终止，故主循环在至多 $O(|T|^2)$ 步内终止（实际为 $O(|T|)$，因为每个 token 至多被 `parse_item` 消费一次或被 `synchronize` 跳过一次）。$\square$

### 4.4 定理 R4（恢复的健全性）

**定理 R4**. 设输入 $T$ 的"真实意图"语法树为 $\text{Tree}^*$（假设存在），`parse_program_with_recovery` 实际产出的解析树为 $P' = (\text{items}', \text{errors}')$。则：

1. **合法性**：$\text{items}'$ 中的每个 item 都是语法合法的（即 `parse_item` 返回 `Ok` 的产物）。
2. **子集近似**：$\text{items}' \subseteq \text{Tree}^*.\text{items}$（在用户意图清晰、错误局限于被跳过区域的前提下）。
3. **错误覆盖**：每个 $e \in \text{errors}'$ 的 span 落在某个被 `synchronize` 跳过的 token 区间内。

**证明**.

(1) `parse_program_with_recovery` 仅在 `Ok(item)` 分支将 item 加入 `items'`（[parser.rs:148-163](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/parser/parser.rs)）。`parse_item` 返回 `Ok(item)` 当且仅当 item 的全部子解析（generic params、params、return type、body 等）均成功——即该 item 是语法合法的。故 $\text{items}'$ 中每个元素合法。

(2) 在用户意图清晰（无歧义）、错误局限于被跳过区域的前提下，`synchronize` 跳过的 token 区间恰对应"用户写错的部分"，保留下的 item 恰对应"用户写对的部分"。此时 $\text{items}' \subseteq \text{Tree}^*.\text{items}$。

**注**：此性质依赖"用户意图清晰"假设。若用户输入歧义（如忘记关闭一个 fn 块），`synchronize` 可能跳过本应保留的 item，导致 $\text{items}' \not\subseteq \text{Tree}^*.\text{items}$。这是 panic-mode 的固有局限（见 §10）。

(3) 每个 $e \in \text{errors}'$ 在 `Err(err)` 分支（[parser.rs:164-167](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/parser/parser.rs)）被压入，紧接着 `synchronize` 推进 pos 到下一个 sync token。$e$ 的 span（line, col）由 `parse_item` 内部子函数构造时记录的是错误发生位置，必然在当前 pos 之前（因为 `?` 触发时 pos 尚未推进到 sync token）。被 `synchronize` 跳过的区间是 $[\text{pos at error}, \text{pos after sync})$，包含 $e$ 的 span。

$\square$

**R4 的实践意义**：恢复后的解析树是"合法子集"，可作为下游 HIR/type checker 的输入；下游不需要处理非法 AST 节点。但需注意子集近似可能丢失用户意图的 item（如嵌套块未正确关闭时）。

### 4.5 定理 R5（双侧错误恢复不等价）

**定理 R5**. 存在一个输入 $T \in \Sigma_{\text{common}}^{\text{lex}}$（双侧 lexer 共同支持的词法子集），使得：
- Rust 母编译器 $\Phi_R$ 在 $T$ 上产出 $(P_R, E_R)$，其中 $E_R \neq \emptyset$（有错误报告），$P_R.\text{items}$ 包含 1 个合法 item；
- tenthc 自举编译器 $\Phi_S$ 在 $T$ 上产出 $(P_S, E_S)$，其中 $E_S = \emptyset$（无错误报告），$P_S.\text{items}$ 与 $P_R.\text{items}$ 不同。

即 T12 定义的 L2（错误行为等价）与 L3（HIR 等价）在"需要错误恢复的输入"上**不成立**。

**证明**（构造性）。

考虑如下输入 $T$（[counter-example.th](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/parser/parser.rs)）：

```
fn broken( -> i32 { 1 }
fn ok() -> i32 { 2 }
```

此输入仅使用 `Fn`、`Identifier`、`LParen`、`Arrow`、`Identifier`(`i32`)、`LBrace`、`IntLiteral`、`RBrace` 等 token，全部属于 tenthc lexer 支持的子集（见 T12 定义 4.5 的 $\Sigma_{\text{common}}$）。故 $T \in \Sigma_{\text{common}}^{\text{lex}}$。

**路径 A（Rust 母编译器）**：
1. 主循环 peek → `Fn`，调用 `parse_item`。
2. `parse_item` 进入 `Fn` 分支（[parser.rs:1511-1566](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/parser/parser.rs)）。
3. 消费 `fn`、`broken`、`(`。
4. 进入参数循环：peek = `->` (Arrow)，期望 `Identifier` 或 `Self_`（[parser.rs:1218-1234](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/parser/parser.rs) 的 `parse_param`）。`Arrow` 不匹配，构造 `Err(ParseError{message: "期望参数名"})` 并返回。
5. 错误经 `?` 传播到 `parse_program_with_recovery`，压入 `errors`。
6. 调用 `synchronize`：从 `->` 开始扫描，跳过 `->`、`i32`、`{`、`1`、`}`，停在 `fn`（属于 $\mathcal{S}$）。
7. 主循环继续：peek = `fn`，调用 `parse_item`，成功解析 `fn ok() -> i32 { 2 }`。
8. 返回 $(P_R, E_R)$：$P_R.\text{items} = [\text{ok}]$，$|E_R| = 1$，错误 span 在 `fn broken(` 处。

**路径 B（tenthc 自举编译器）**：
1. `parse_program` ([parser.th:1456-1523](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenthc/parser/parser.th)) 主循环 peek disc=4 (`Fn`)，调用 `parse_fn`。
2. `parse_fn` ([parser.th:1128-1240](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenthc/parser/parser.th))：
   - `parser_advance` 跳过 `fn`。
   - `name_tok = parser_advance` 得到 `broken`，`name = "broken"`。
   - `parse_generic_params`：peek 不是 `<`，返回空。
   - `parser_advance(p)` 跳过 `(`（[parser.th:1134](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenthc/parser/parser.th)）。
   - 参数循环（[parser.th:1136-1200](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenthc/parser/parser.th)）：peek = `->`。检查 `t.sval == "self"`：`->` 的 sval 是空字符串，false。检查 `t.disc == 38`（Ampersand）：`->` disc 不是 38。进入 else 分支：
     - `pname_tok = parser_advance` 得到 `->`，`pname = ""`（`->` 无 sval）。
     - `parser_advance(p)` 跳过 `:`——但实际是 `i32`！这里 `parser_advance` 直接消费 `i32` 作为冒号的占位（[parser.th:1185](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenthc/parser/parser.th)）。
     - 进入类型收集循环：从 `{` 开始，`is_end_of_paren` 检查 disc 48/61。`{` disc=49，不是。继续消费 `{`、`1`、`}`。下一 token 是 `fn`（disc=4），不是 48/61，继续消费 `fn`、`ok`、`(`、`)`、`->`、`i32`、`{`、`2`、`}`。到达 Eof（disc=61），break。
     - `params.push(Param{name: "", type_ann: "-> { 1 } fn ok ( ) -> i32 { 2 }"})`。
   - `parser_advance(p)` 跳过 `)`——但 pos 已到 Eof，返回 EOF token。
   - 返回类型：peek disc=58 (`:`)？实际是 Eof disc=61，跳过。
   - `parser_advance(p)` 跳过 `{`——Eof，返回 EOF。
   - 块语句循环：`is_end_of_block` 检查 disc 50/61，Eof 命中，break。
   - `parser_advance(p)` 跳过 `}`——Eof，返回 EOF。
   - 返回 `FnDef{name: "broken", params: [garbage], body_start: ..., body_count: 0}`。
3. 主循环：`parser_at_eof(&p)` 为 true，break。
4. 返回 $(P_S, E_S)$：$P_S$ 包含一个 `broken` 函数定义（params 字段为垃圾值，body 为空），$E_S = \emptyset$（tenthc 无错误收集机制）。

**对比**：

| 维度 | 路径 A (Rust) | 路径 B (tenthc) |
|------|--------------|----------------|
| 错误数 | 1 | 0 |
| 解析出的 item | `fn ok` | `fn broken`（垃圾 params） |
| 是否报告 L2 错误 | 是 | 否 |
| HIR 等价（L3） | — | 不成立（item 不同） |

故 T12 定义的 L2（错误集等价）失效：$E_R \neq E_S$（$|E_R|=1 \neq 0=|E_S|$）。L3（HIR 等价）亦失效：$P_R.\text{items} \neq P_S.\text{items}$。

$\square$

**R5 与 T12 的精确关系**：
- T12 定义 4.6 将 $\Sigma_{\text{common}}^{\text{parse}}$ 限制为"两侧 parser 都能正确解析**且不需要错误恢复**的源码子集"。
- 本文 R5 的反例 $T$ 正属于"需要错误恢复的输入"——按 T12 定义不在 $\Sigma_{\text{common}}^{\text{parse}}$ 内，故不直接反驳 T12 定理 S5（共同子集等价）。
- 但 R5 **精确刻画了 T12 等价保证的边界**：等价性仅在"无需恢复"的输入上成立；任何需要 panic-mode 恢复的输入都落在等价保证之外。这一边界此前在 T12 中仅作直觉陈述，本文给出严格构造性证明。

---

## 5. SYNC_TOKENS 选择的合理性分析

### 5.1 每个 SYNC_TOKEN 的选择理由

| SYNC_TOKEN | 选择理由 | 必要性 |
|-----------|---------|--------|
| `Fn` | 函数声明起始符 | ✅ 顶级 item 边界 |
| `Struct` | 结构体声明起始符 | ✅ 顶级 item 边界 |
| `Enum` | 枚举声明起始符 | ✅ 顶级 item 边界 |
| `Trait` | Trait 声明起始符 | ✅ 顶级 item 边界 |
| `Impl` | Impl 块起始符 | ✅ 顶级 item 边界 |
| `Mod` | 模块声明起始符 | ✅ 顶级 item 边界 |
| `Use` | Use 声明起始符 | ✅ 顶级 item 边界 |
| `RBrace` | 块闭合符 | ✅ 允许在嵌套结构内恢复到下一成员 |
| `Eof` | 输入终止符 | ✅ 保证终止性（R3） |

### 5.2 是否覆盖所有 item 起始符

由定理 R2，$\mathcal{I} \setminus \{\text{Pub}\} \subseteq \mathcal{S}$，且 `Pub` 的省略不破坏恢复能力。覆盖性成立。

### 5.3 遗漏的潜在同步点

以下 token **未纳入** $\mathcal{S}$，分析其后果：

| 候选 token | 未纳入的后果 | 是否应纳入？ |
|-----------|------------|------------|
| `Semicolon` | `use a::b;` 中若 `;` 后有错误，`synchronize` 跳过 `;` 到下一个 item 起始符。粒度略粗，但功能正确。 | 否（粒度损失可接受） |
| `RParen` | 表达式内部的 `)` 不作为同步点。但 panic-mode 仅在 item 级恢复，不需要表达式级同步点。 | 否（不符合 panic-mode 的 item 级粒度） |
| `LBrace` | 块开符。纳入会导致 `synchronize` 在块开处停下，但后续解析无法知道这是哪个 item 的块。 | 否（会破坏恢复语义） |
| `Type` | Trait 中的关联类型声明。属于 trait 内部成员，非顶级 item。 | 否（trait 内部恢复已由 `RBrace` 处理） |
| `Pub` | 修饰符前缀。省略不影响恢复（见 R2 证明）。 | 否（已分析） |
| `Const` | Tenth 当前**无** const item（见 `parse_item` 无 `Const` 分支）。 | 否（不存在该 item） |

**结论**：当前 $\mathcal{S}$ 的选择在 panic-mode 框架下是**合理的最小完备集**。任何增补要么冗余（如 `Semicolon`），要么破坏恢复语义（如 `LBrace`），要么对应不存在的 item（如 `Const`）。

---

## 6. 与 clang/rustc 的对比

### 6.1 clang 的错误恢复策略

clang 在 panic-mode 之上叠加：
- **括号匹配恢复**：`Parser::BalancedDelimiter` 跟踪 `()`/`{}`/`[]` 的开闭，缺失时插入虚拟闭合符；
- **Token 插入**：`ExpectAndConsume` 在期望 token 缺失时根据上下文插入（如缺失的 `,`）；
- **`RecoveryExpr` AST 节点**：保持 AST 结构完整；
- **错误产生式**：为常见错误（如 `if (x = 5)`）预声明产生式，给出针对性诊断。

clang 的恢复可在任意嵌套层级重启，错误信息质量极高，但实现复杂度约 8 万行 C++。

### 6.2 rustc 的错误恢复策略

rustc 在 panic-mode 之上叠加：
- **`Recovery` 字段**：parser 携带恢复状态（如"已插入缺失的 `;`"）；
- **`Err` AST 节点**：`ExprKind::Err`、`PatKind::Err` 等占位；
- **`Substitution` 建议**：诊断中附带"did you mean X?"建议；
- **`match` 臂分隔符恢复**：缺失 `,` 时尝试回退。

rustc 的恢复侧重错误信息质量与避免级联错误，复杂度约 3 万行 Rust。

### 6.3 Tenth 的简化假设

Tenth Rust 母编译器的恢复策略相对简化：

| 特性 | clang | rustc | Tenth |
|------|-------|-------|-------|
| Panic-mode 基础 | ✓ | ✓ | ✓ |
| 括号匹配恢复 | ✓ | ✓ | ✗ |
| Token 插入 | ✓ | ✓ | ✗ |
| AST 占位节点 | ✓ (`RecoveryExpr`) | ✓ (`Err`) | ✗ |
| Error production | ✓ | ✓ | ✗ |
| 错误建议 | ✓ | ✓ (`Substitution`) | ✗ |
| 多错误收集 | ✓ | ✓ | ✓ |
| 代码量 | ~8 万行 | ~3 万行 | ~10 行（`synchronize`） |

Tenth 的简化是工程权衡：以 10 行代码换取"不死循环 + 多错误收集"的核心能力，代价是错误信息质量较低、无错误建议、无 AST 占位。这一权衡对小规模项目可接受，但对工业级用户体验不足。

---

## 7. 双侧错误恢复不对称

### 7.1 tenthc 无错误恢复的影响

由 R5 的构造性反例，tenthc parser 在"需要错误恢复的输入"上：
1. **不报告错误**：`parse_program` 返回 `Program` 而非 `(Program, Vec<Error>)`，无错误收集机制（[parser.th:1456-1523](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenthc/parser/parser.th)）。
2. **产出垃圾 AST**：`parse_primary` 的兜底分支返回 `int 0` 节点（[parser.th:446-449](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenthc/parser/parser.th)），其他 parse_* 函数在意外 token 处直接 `parser_advance` 推进，导致 AST 结构与用户意图完全脱节。
3. **可能无限循环**：若 `parse_program` 主循环的某个分支既不消费 token 也不 break，理论上可能死循环。审查 tenthc 代码，主循环每次至少调用 `parse_stmt` 或 `parse_*_def`，这些函数至少 `parser_advance` 一次，故实际不死循环——但这是隐式性质，无形式化保证。

### 7.2 对自举三路径的影响（与 T12 联动）

T12 定义自举三路径：
- **路径 A**：Rust 全栈编译（日常使用）；
- **路径 B**：Tenth 前端 + Rust 后端（`bridge.rs`）；
- **路径 C**：全 WASM 闭环。

路径 A 使用 Rust 母编译器，享有 panic-mode 恢复。路径 B 使用 tenthc 前端，**无任何恢复**——若输入有语法错误，tenthc 产出垃圾 AST，`bridge.rs` 将垃圾 AST 翻译为 HIR，下游可能崩溃或产出错误结果。

T12 §4.5 将等价边界定位在 $\Sigma_{\text{common}}^{\text{parse}}$（无需恢复的子集），本文 R5 给出该边界的构造性证明。两篇论文共同结论：**自举三路径的"不可破坏"声明仅在 $\Sigma_{\text{common}}^{\text{parse}}$ 上成立**，需要错误恢复的输入会破坏路径 A 与路径 B 的等价性。

### 7.3 最小修补集

为消除双侧不对称，需要在 tenthc 侧补全错误恢复。最小修补集（按优先级递增）：

| 编号 | 修补项 | 优先级 | 依赖 | 工作量估计 |
|------|-------|--------|------|----------|
| P1 | 在 tenthc `Parser` 结构中添加 `errors: Vec<Error>` 字段 | 高 | 无 | 小 |
| P2 | 在 tenthc 中定义 `SYNC_TOKENS` 常量（与 Rust 侧对齐） | 高 | 无 | 小 |
| P3 | 实现 `synchronize` 函数（移植 Rust 侧逻辑） | 高 | P2 | 中 |
| P4 | 修改 `parse_program` 主循环为 `Result` 返回，遇错调用 `synchronize` | 高 | P1, P3 | 中 |
| P5 | 修改各 `parse_*` 函数返回 `Result`，错误经 `?` 传播 | 高 | P4 | 大（影响所有 parse_* 函数） |
| P6 | 在 `bridge.rs` 中处理 tenthc 传回的错误集 | 中 | P5 | 中 |
| P7 | 添加 tenthc 错误恢复测试，验证与 Rust 侧行为一致 | 中 | P5, P6 | 中 |

**注**：P5 是工作量最大的步骤，因为 tenthc 当前所有 parse_* 函数返回值类型而非 `Result`，需逐个改造。这与 T12 §9 列出的"双侧 parser 接口对齐"修补一致。

**当前状态**：以上修补**均未实施**，tenthc 侧仍无错误恢复。本文将 P1–P7 列为未来工作。

---

## 8. 工程权衡

### 8.1 panic-mode 的简单性

Tenth Rust 侧的 panic-mode 实现仅约 10 行核心代码（`synchronize` 函数）+ 9 个 sync token 常量。这一极简实现带来：

- **维护成本低**：无需随语法演进而调整（只要顶级 item 起始符不变）；
- **正确性易验证**：终止性（R3）与覆盖性（R2）的证明简短；
- **无副作用**：不修改 AST 结构、不插入虚拟 token，下游 HIR/type checker 无需感知恢复状态。

### 8.2 错误信息质量

panic-mode 的错误信息质量较低：
- **粒度粗**：错误 span 是 `parse_item` 失败时的当前位置，可能距实际错误位置较远；
- **无建议**：不提供"did you mean X?"建议；
- **可能漏报**：若 `synchronize` 跳过的区间内包含多个独立错误，仅报告首个。

测试 [`error_recovery_test.rs`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/tests/error_recovery_test.rs) 验证了多错误收集能力（`test_recovery_multiple_errors` 期望至少 2 个错误），但未验证错误信息的精确性。

### 8.3 多错误收集的能力

由 R1，每个被 `parse_item` 抛出的错误都会被收集。在测试 `test_recovery_multiple_errors` 中：

```
fn first_good() -> i32 { 1 }
fn bad1( -> i32 { 2 }
fn second_good() -> i32 { 3 }
fn bad2( -> i32 { 4 }
```

Rust 侧对每个 `bad1`/`bad2` 各报告一个错误，且 `first_good`/`second_good` 仍被正确解析。$|E_R| \geq 2$。tenthc 侧在相同输入上**不报告任何错误**，且 AST 结构错误（如 R5 反例所示）。

---

## 9. 开放问题与未来工作

### 9.1 phrase-level 恢复的引入

当前 panic-mode 的恢复粒度为 item 级。若需在表达式级恢复（如 `1 + + 2` 中恢复第二个 `+` 后的表达式），需引入 phrase-level 恢复。这要求：

- 在 `parse_binary`、`parse_postfix` 等函数中添加局部恢复点；
- 定义表达式级同步 token（如 `,`、`;`、`)`、`]`、`}`）；
- 处理括号平衡（避免恢复时跨过未闭合的括号）。

phrase-level 恢复**未实现**，列为未来工作。

### 9.2 tenthc 错误恢复的补全

如 §7.3 所述，P1–P7 是消除双侧不对称的最小修补集。其中 P5（改造所有 `parse_*` 函数返回 `Result`）工作量最大，且需同步更新 `bridge.rs` 以处理错误集。这一补全将扩展 T12 的 $\Sigma_{\text{common}}^{\text{parse}}$ 至包含"需要错误恢复的输入"，使 L2 等价性在更大子集上成立。

### 9.3 错误信息质量提升

未来可考虑：
- 在 `TenthError::ParseError` 中添加 `expected: Vec<TokenKind>` 与 `actual: TokenKind` 字段，支持"期望 X 但遇到 Y"的精确诊断；
- 添加 `suggestion: Option<String>` 字段，支持"did you mean X?"建议；
- 在 `synchronize` 中记录跳过的 token 范围，用于错误信息的"上下文展示"。

这些增强**未实现**，列为未来工作。

### 9.4 形式化验证

本文的定理 R1–R5 基于对源码的人工审查。未来可考虑：
- 用 Coq/Lean 形式化 `synchronize` 的终止性（R3）与覆盖性（R2）；
- 用 fuzzing 验证 R4（健全性）——随机生成畸形输入，检查恢复后的 AST 是否合法子集；
- 用等价性检查工具验证 R5——枚举小规模输入，自动检测双侧行为差异。

---

## 10. 局限（诚实披露）

本节集中记录本文分析的局限，符合数理部"局限必披露"原则。

### 10.1 panic-mode 的固有局限

- **粒度粗**：仅在 item 级恢复，不恢复表达式级错误（见 §9.1）。
- **可能丢失 item**：若 `synchronize` 跳过的区间内包含合法 item（如嵌套块未正确关闭），该 item 会被跳过（R4 的子集近似假设可能不成立）。
- **不报告被跳过区间的内部错误**：仅报告首个触发 `parse_item` 失败的错误，区间内的其他错误被吞没。
- **不处理词法错误**：R1 不可恢复类 (a)，词法错误在 parser 之前已处理。
- **不处理运行时 panic**：R1 不可恢复类 (d)，`unreachable!()` 等会绕过恢复机制。

### 10.2 形式化的不完备处

- **R4 的"用户意图清晰"假设**：R4(2) 依赖"用户意图清晰、错误局限于被跳过区域"的假设。这一假设无法形式化验证，实际可能不成立。
- **R5 的单一反例**：R5 仅给出一个反例，未穷举所有"需要恢复的输入"类。该类输入的完整刻画是开放问题。
- **未形式化 tenthc 的隐式终止性**：§7.1 声称 tenthc "实际不死循环"，但未给出形式化证明。

### 10.3 工程差距

- **测试覆盖不足**：`error_recovery_test.rs` 仅 7 个测试用例，未覆盖深层嵌套错误、跨 item 错误等场景。
- **双侧不对称未修补**：§7.3 的 P1–P7 均未实施，tenthc 侧仍无错误恢复。
- **无 fuzzing 验证**：R4 的健全性未经验证。

### 10.4 假设的强度

- **R2 的"item 起始符枚举完备"假设**：基于对 `parse_item` 的人工审查。若未来语法扩展新增 item 类型（如 `Const`、`Static`），需同步更新 $\mathcal{S}$ 与 R2 的证明。
- **R3 的"peek 在 pos ≥ n 时返回 EOF_TOKEN"假设**：基于 `tokens.get(pos).unwrap_or(&EOF_TOKEN)` 的实现。若实现变更（如改用 `tokens[pos]` 直接索引），R3 失效。

---

## 11. 结论

本文对 Tenth 语言 Rust 母编译器的 panic-mode 错误恢复策略进行了形式化分析，给出五条主定理：

1. **R1（捕获集刻画）**：精确描述可恢复错误类为"经由 `parse_item` 的 `?` 传播路径返回的 `TenthError::ParseError`"，并枚举不可恢复类（词法错误、主循环逻辑错误、运行时 panic、语义错误）。
2. **R2（SYNC_TOKENS 覆盖性）**：证明 $\mathcal{S}$ 覆盖全部 7 个顶级 item 起始符，`Pub` 的有意省略不破坏恢复能力。
3. **R3（终止性）**：`synchronize` 在至多 $O(|T|)$ 步内终止，`parse_program_with_recovery` 在至多 $O(|T|^2)$ 步内终止（实际 $O(|T|)$）。
4. **R4（健全性）**：恢复后的解析树是合法语法树的子集近似，每个错误的 span 落在被跳过区间内。
5. **R5（双侧不等价）**：通过构造性反例证明 tenthc parser 完全缺失错误恢复，导致 T12 的 L2/L3 等价性在"需要恢复的输入"上失效。

**对实施的指导**：
- 当前 `SYNC_TOKENS` 选择合理，无需调整；
- 双侧不对称（§7）是优先级最高的修补项，P1–P7 给出最小修补集；
- phrase-level 恢复与错误信息质量提升为可选增强。

**与 T12 的协同结论**：本文与 T12 共同确立了双侧编译器等价性的精确边界——等价性在 $\Sigma_{\text{common}}^{\text{parse}}$（无需恢复的子集）上成立，在需要错误恢复的输入上失效。这一边界的刻画为未来的双侧修补提供了明确目标。

---

## 12. 参考文献

[^1]: Aho, A. V., Lam, M. S., Sethi, R., & Ullman, J. D. (2006). *Compilers: Principles, Techniques, and Tools* (2nd ed.). Addison-Wesley. Chapter 4: Syntax Analysis, §4.4 Error Recovery.

[^2]: Clang Language Frontend Documentation. LLVM Project. https://clang.llvm.org/docs/ . 参见 `Parser::ExpectAndConsume`、`Parser::BalancedDelimiter` 等实现。

[^3]: rustc Parse Module. Rust Compiler. https://github.com/rust-lang/rust/tree/master/compiler/rustc_parse . 参见 `parser.rs` 的 `Recovery` 字段与 `ExprKind::Err` 占位。

[^4]: T12《双侧编译器语义等价性》. Tenth 项目内部论文. 参见 `docs/论文/T12-双侧编译器语义等价性.md`.

[^5]: Tenth 项目 `MEMO.md` v0.3.3 变更记录. 参见 `MEMO.md`.

[^6]: Tenth 项目 `AUDIT.md` 缺陷登记册. 参见 `AUDIT.md`，其中 `error_recovery_test.rs` 列出 7 项测试覆盖。

---

## 附录 A：定理索引

| 定理 | 名称 | 陈述 | 证明位置 |
|------|------|------|---------|
| R1 | 捕获集刻画 | $\mathcal{C}$ 由类型条件与传播条件刻画 | §4.1 |
| R2 | SYNC_TOKENS 覆盖性 | $\mathcal{I} \setminus \{\text{Pub}\} \subseteq \mathcal{S}$ | §4.2 |
| R3 | 终止性 | `synchronize` 在 $O(\|T\|)$ 步内终止 | §4.3 |
| R4 | 健全性 | 恢复后解析树是合法子集近似 | §4.4 |
| R5 | 双侧不等价 | 构造性反例证明 L2/L3 失效 | §4.5 |

## 附录 B：与现有文档的对应

| 论文节 | 对应源码/文档 |
|-------|--------------|
| §3.2 SYNC_TOKENS 定义 | [parser.rs:15-25](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/parser/parser.rs) |
| §3.3 synchronize 算法 | [parser.rs:120-129](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/parser/parser.rs) |
| §3.4 parse_program_with_recovery | [parser.rs:141-188](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/parser/parser.rs) |
| §4.5 R5 反例 | [tenthc/parser/parser.th:1128-1240](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenthc/parser/parser.th) |
| §6.3 测试 | [error_recovery_test.rs](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/tests/error_recovery_test.rs) |
| §7.2 与 T12 联动 | [T12 论文](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/docs/论文/T12-双侧编译器语义等价性.md) §4.5–4.6 |
| §7.3 修补集 | 与 T12 §9 修补表对齐 |

## 附录 C：实施建议

以下建议面向编译器部实施团队（数理部不写代码，仅给出指导）：

1. **短期（不变更双侧接口）**：
   - 维持当前 `SYNC_TOKENS` 不变（R2 验证其完备性）；
   - 在 `error_recovery_test.rs` 中增加跨 item 边界的恢复测试（如 R5 反例的 Rust 侧行为）；
   - 在 `MEMO.md` 记录"双侧错误恢复不对称"为已知限制。

2. **中期（补全 tenthc 恢复）**：
   - 按 §7.3 的 P1–P7 顺序实施修补；
   - 每步修补后跑自举验证：`cargo run --release --manifest-path tenth/Cargo.toml -- run tenthc/main.th`；
   - 在 `bridge.rs` 中处理 tenthc 传回的错误集，与 Rust 侧错误集做并集后报告。

3. **长期（错误信息质量）**：
   - 评估引入 phrase-level 恢复的必要性（取决于用户反馈）；
   - 考虑添加 `expected`/`actual`/`suggestion` 字段到 `TenthError::ParseError`；
   - 评估形式化验证（Coq/Lean）的投入产出比。

---

*文档版本：v1.0 | 完成日期：2026-07-02 | 数理部产出*
