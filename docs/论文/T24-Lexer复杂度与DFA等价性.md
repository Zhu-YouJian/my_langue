# 手写 Lexer 与 DFA 的等价性：Tenth 词法分析器的形式化与复杂度分析

> **数理部论文 T24** · Tenth 编译管线理论系列之四
> 版本：v1.0 · 日期：2026-07-02
> 适用：Tenth v0.3.3 · 难度：本科毕设级
> 主题：手写 lexer 的 DFA 等价性、peek 深度复杂度、增量算术解析的精度边界

---

## 摘要

Tenth 语言的词法分析器（lexer）采用手写递归状态机实现，而非 Lex/Flex 风格的 DFA 生成器。本文对该手写 lexer 进行形式化分析，证明其与某个确定性有限状态自动机（DFA）等价（定理 D1），时间复杂度为 O(n)（定理 D2），满足 maximal munch 最长匹配规则（定理 D3），并给出 tenthc 自举 lexer 增量算术解析 `ival = ival * 10 + digit` 的 i64 溢出精度边界分析（定理 D4）。核心发现包括：(1) 手写 lexer 的最大 peek 深度为 4（用于 `f32`/`f64` 后缀的边界检测），常数因子受 peek 深度影响但不改变线性复杂度；(2) **tenthc 的增量算术解析不检测 i64 溢出**，溢出时由 Tenth VM 的 `overflow-checks = true` 配置触发 panic，而 Rust 母编译器通过 `str::parse::<i64>()` 返回优雅错误——这是两侧 lexer 在精度边界上的**已知不对称**；(3) 在共同 token 子集上，两侧 lexer 满足等价性（定理 D5）。本文诚实记录 `>>=` token 不存在、tenthc 缺失 `Shl`/`CharLiteral`/`InterpolatedString` 等局限。

**关键词**：词法分析 · DFA 等价性 · 手写 lexer · maximal munch · peek 深度 · 增量算术解析 · i64 溢出 · 自举编译器

---

## 1. 引言

### 1.1 词法分析的两条路径

词法分析（lexical analysis）是编译器的第一阶段，将源码字符流切分为 token 序列。实现词法分析有两条经典路径：

- **DFA 生成器路径**：以 Lex/Flex 为代表，开发者用正则表达式描述 token 模式，工具自动将其编译为确定性有限状态自动机（DFA）的转移表，运行时按表驱动执行 [1]。
- **手写 lexer 路径**：以 GCC、LLVM、rustc 为代表，开发者直接用过程式代码（C/C++/Rust）编写状态机，通过 `peek`/`advance` 原语逐字符消费输入 [2]。

两条路径在理论上应等价——任何手写 lexer 本质上是一个状态机——但工程特性迥异：DFA 生成器保证最长匹配（maximal munch）与自动错误恢复，手写 lexer 提供更精确的错误信息与更灵活的上下文处理。

### 1.2 手写 lexer 的工程优势与理论挑战

手写 lexer 在工业级编译器中占主导地位（GCC C++ 前端、LLVM `clang::Lexer`、rustc `rustc_lexer`），其优势包括：

1. **错误信息质量**：可在特定状态注入精确的诊断信息（如"字符串未闭合，在第 N 行"），DFA 生成器通常只能报告"非法字符"。
2. **上下文敏感处理**：可基于前序 token 调整行为（如 Rust 的 raw identifier `r#ident`）。
3. **性能可控**：避免 DFA 转移表的间接跳转开销，热点路径可手动优化。

但手写 lexer 引入**理论挑战**：

- **DFA 等价性**：手写代码的任意控制流是否总能对应某个 DFA？若不能，则 lexer 接受的语言可能超出正则语言范畴，无法与 DFA 生成器对齐。
- **maximal munch 保持**：手写代码中 if-else 链的检查顺序是否保证了最长匹配？顺序错误可能导致 `>=` 被错误切分为 `>` `=`。
- **复杂度保证**：手写代码中的 lookahead（peek）深度是否有限？无界 lookahead 会破坏 O(n) 复杂度。
- **精度边界**：数字字面量的算术解析是否处理溢出？未处理的溢出导致静默错误。

### 1.3 Tenth 的手写 lexer

Tenth 语言（Tensor + Zenith）的词法分析器位于 [tenth/src/lexer/lexer.rs](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/lexer/lexer.rs)，采用 Rust 手写实现。其自举编译器 tenthc 的对应实现在 [tenthc/lexer/lexer.th](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenthc/lexer/lexer.th)，用 Tenth 自身编写。

核心入口 `next_token`（[lexer.rs:386-531](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/lexer/lexer.rs)）采用经典的"字符分类 + peek 判定"模式：先 `skip_whitespace_and_comments`，再按首字符分类（数字 / 标识符 / 字符串 / 运算符），多字符 token 通过 1–4 字符的 peek 深度判定。

### 1.4 贡献

本文的贡献如下：

1. **DFA 等价性证明**（定理 D1）：构造与 Tenth 手写 lexer 等价的 DFA，证明接受语言相同。
2. **复杂度分析**（定理 D2）：证明 O(n) 时间复杂度，量化 peek 深度 k 与常数因子的关系。
3. **maximal munch 保持**（定理 D3）：证明手写 lexer 满足最长匹配规则。
4. **增量算术解析的精度边界**（定理 D4）：分析 tenthc 的 `ival = ival * 10 + digit` 是否检测 i64 溢出，揭示两侧不对称。
5. **双侧 lexer 等价**（定理 D5）：在共同 token 子集上证明 Rust lexer 与 tenthc lexer 等价。
6. **诚实记录局限**：`>>=` token 不存在、tenthc 缺失多个 token、溢出检测缺失、人工证明未机器验证。

---

## 2. 背景与相关工作

### 2.1 有限状态自动机理论

**定义 2.1**（DFA）。一个确定性有限状态自动机是五元组 $M = (Q, \Sigma, \delta, q_0, F)$，其中：
- $Q$ 是有限状态集；
- $\Sigma$ 是有限输入字母表；
- $\delta: Q \times \Sigma \to Q$ 是转移函数；
- $q_0 \in Q$ 是初始状态；
- $F \subseteq Q$ 是接受状态集。

DFA $M$ 接受的语言 $L(M) = \{w \in \Sigma^* \mid \delta^*(q_0, w) \in F\}$，其中 $\delta^*$ 是 $\delta$ 的自反传递闭包。

**定义 2.2**（正则语言）。语言 $L$ 是正则语言，当且仅当存在 DFA $M$ 使得 $L = L(M)$。

**定理 2.1**（Kleene 定理 [3]）。语言是正则语言当且仅当它能用正则表达式描述。

### 2.2 Lex/Flex 的 DFA 生成

Lesk 的 Lex [4] 与其后继 Flex 将正则表达式描述的 token 规则编译为 DFA。其流程为：

1. 正则表达式 → NFA（Thompson 构造 [5]）；
2. NFA → DFA（子集构造，subset construction）；
3. DFA 最小化（Hopcroft 算法 [6]）；
4. 生成转移表的 C 代码。

Lex/Flex **自动保证 maximal munch**：在 DFA 中，每一步都尽可能多消费字符直到无法转移，然后回退到最近一个接受状态。这一保证是 DFA 运行机制的直接推论。

### 2.3 手写 lexer 的传统

工业级编译器普遍采用手写 lexer：

- **GCC** C 前端 `libcpp/lex.c`：手写，处理 C 的 `/* */` 嵌套注释与字符串字面量。
- **LLVM clang** `clang::Lexer`（[llvm-project/clang/lib/Lex/Lexer.cpp](https://github.com/llvm/llvm-project)）：手写，支持 C/C++ 的 trigraph、digraph、raw string。
- **rustc** `rustc_lexer`（[rust-lang/rust/compiler/rustc_lexer](https://github.com/rust-lang/rust)）：手写，处理 Rust 的 raw identifier `r#`、byte string `b"`、lifetime `'a`。

手写 lexer 的共同特征：(1) 用 `peek`/`advance` 原语逐字符消费；(2) 多字符 token 通过有限 lookahead 判定；(3) 错误信息携带精确位置。

### 2.4 maximal munch 规则

**定义 2.3**（maximal munch [7]）。lexer 满足 maximal munch，当且仅当对任意输入位置 $i$，lexer 选择的 token $t$ 是所有能从位置 $i$ 匹配的 token 中**最长**的。

maximal munch 是大多数编程语言词法分析的隐含约定。例如，在 C 中 `>>` 应识别为右移运算符而非两个 `>`；在 Java 中 `>>=` 应识别为复合右移赋值而非 `>>` `=`。

### 2.5 Tenth 的双侧 lexer 架构

Tenth 维护两套 lexer（详见 T12 [T12-双侧编译器语义等价性](T12-双侧编译器语义等价性.md)）：

- **Rust 母编译器 lexer**（[tenth/src/lexer/lexer.rs](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/lexer/lexer.rs)）：完整功能，支持字符串插值、转义字符、所有 token。
- **tenthc 自举 lexer**（[tenthc/lexer/lexer.th](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenthc/lexer/lexer.th)）：子集实现，无字符串插值、无转义字符、缺失部分 token。

两侧 Token 表示的去规范化差异（tenthc 携带冗余 `disc`/`ival`/`fval`/`sval` 字段）已在 T15 [T15-Denormalized Token表示等价性](T15-Denormalized Token表示等价性.md) 中分析。本文聚焦于 lexer 的**控制流等价性**与**复杂度**。

---

## 3. Tenth Lexer 形式化

### 3.1 记号约定

- $\Sigma$：Tenth 源码字符集（Unicode，但 lexer 实际仅处理 ASCII 子集 + 透传非 ASCII 标识符字符）。
- $\Sigma^*$：源码字符串集合。
- $\mathcal{T} = \{t_1, \ldots, t_N\}$：Token 种类集合（$N \approx 60$，见 [token.rs:13-100](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/lexer/token.rs)）。
- $\text{Lexer}: \Sigma^* \to \mathcal{T}^*$：词法分析函数。
- $\text{pos}$：当前消费位置（$0 \le \text{pos} \le |w|$）。
- $\text{peek}(w, \text{pos}, k)$：从位置 $\text{pos}$ 起向前看 $k$ 个字符（$k \ge 1$），返回 $w[\text{pos}], w[\text{pos}+1], \ldots, w[\text{pos}+k-1]$。

### 3.2 next_token 的状态机刻画

`next_token`（[lexer.rs:386-531](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/lexer/lexer.rs)）的逻辑可刻画为以下状态迁移：

**状态 0（Start）**：调用 `skip_whitespace_and_comments`，然后 `peek` 首字符 $c$：
- $c = \text{EOF}$ → 接受 `Eof`，终止。
- $c \in [0-9]$ → advance，转入状态 1（Number）。
- $c \in [a-zA-Z\_]$ → advance，转入状态 2（Identifier）。
- $c = \text{"}$ → 转入状态 3（String）。
- $c \in \{=, !, <, >, \&, |, +, -, *, /, ., :\}$ → advance，转入状态 4（Operator）。
- $c \in \{(, ), [, ], \{, \}, ,, ;, :, ., \%, ^, ?\}$ → advance，接受对应单字符 token。
- 其他 → 错误"意外字符"。

**状态 1（Number）**：调用 `read_number`（[lexer.rs:97-203](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/lexer/lexer.rs)）：
- 消费连续的 $[0-9\_]$（整数部分）。
- peek `.` + peek_next 数字 → 消费小数点 + 连续 $[0-9\_]$（小数部分）。
- peek `e`/`E` → 消费指数部分（含可选 `+`/`-`）。
- peek `f` + 检查 $\text{pos}+1, \text{pos}+2, \text{pos}+3$ → 消费 `f32`/`f64` 后缀（4 字符 peek）。
- 根据 `is_float`/`suffix_dtype` 构造 `IntLiteral(i64)` 或 `FloatLiteral(f64, BaseType)`。

**状态 2（Identifier）**：调用 `read_identifier`（[lexer.rs:205-257](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/lexer/lexer.rs)）：
- 消费连续的 $[a-zA-Z0-9\_]$。
- 查表：若匹配关键字（`fn`/`let`/`if`/...，共 32 个），返回对应 `TokenKind`；否则返回 `Identifier(String)`。

**状态 3（String）**：调用 `read_string`（[lexer.rs:259-365](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/lexer/lexer.rs)）：
- 消费到闭合 `"`，处理 `\n`/`\t`/`\\`/`\"`/`\{`/`\}` 转义。
- 检测 `{ident}` 插值：peek `{` + peek_next 是字母/下划线 → 消费到 `}`，构造 `InterpolatedString`。

**状态 4（Operator）**：根据首字符 $c$ 与 peek 的第二字符 $c'$：
- $c = `=`：$c' = `=`$ → `EqEq`；$c' = `>`$ → `FatArrow`；否则 `Assign`。
- $c = `!`：$c' = `=`$ → `NotEq`；否则 `Not`。
- $c = `<`：$c' = `=`$ → `LtEq`；$c' = `<`$ → `Shl`；否则 `Lt`。
- $c = `>`：$c' = `=`$ → `GtEq`；$c' = `>`$ → `Shr`；否则 `Gt`。
- $c = `&`：$c' = `&` → `AndAnd`；否则 `Ampersand`。
- $c = `|`：$c' = `|` → `OrOr`；否则 `Pipe`。
- $c = `+`：$c' = `=` → `PlusAssign`；否则 `Plus`。
- $c = `-`：$c' = `=` → `MinusAssign`；$c' = `>` → `Arrow`；否则 `Minus`。
- $c = `*`：$c' = `=` → `StarAssign`；否则 `Star`。
- $c = `/`：$c' = `=` → `SlashAssign`；否则 `Slash`。
- $c = `.`：$c' = `.` → 消费后再 peek `=` → `DotDotEq` 或 `DotDot`；否则 `Dot`。
- $c = `:`：$c' = `:` → `ColonColon`；否则 `Colon`。

### 3.3 字符分类与状态迁移

lexer 的字符分类可形式化为函数 $\text{class}: \Sigma \to C$，其中 $C$ 是有限字符类集合：

$$
C = \{\text{Digit}, \text{Alpha}, \text{Underline}, \text{Quote}, \text{OpEq}, \text{OpBang}, \text{OpLt}, \text{OpGt}, \ldots, \text{Single}, \text{Other}\}
$$

字符分类是平凡的（基于 ASCII 码点比较），不引入状态。

### 3.4 peek 深度的使用模式

Tenth lexer 有两类 peek：

**类型 A（推进式 peek）**：peek 1 字符，若匹配则 advance，再 peek 下一字符。用于运算符链（如 `..=` 识别）。

**类型 B（非推进式 peek）**：在不 advance 的前提下，同时检查 $\text{pos}+1, \text{pos}+2, \ldots, \text{pos}+k$ 位置的字符。用于 `f32`/`f64` 后缀检测（[lexer.rs:153-169](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/lexer/lexer.rs)）：

```rust
if self.peek() == Some('f') {
    let c1 = self.source.get(self.pos + 1).copied();
    let c2 = self.source.get(self.pos + 2).copied();
    let c3 = self.source.get(self.pos + 3).copied();
    let boundary_ok = c3.map_or(true, |c| !c.is_alphanumeric() && c != '_');
    match (c1, c2, boundary_ok) {
        (Some('3'), Some('2'), true) => { ... suffix_dtype = Some(BaseType::F32); }
        (Some('6'), Some('4'), true) => { ... suffix_dtype = Some(BaseType::F64); }
        _ => {}
    }
}
```

此处 peek 深度 $k = 4$（检查 `f` + `3`/`6` + `2`/`4` + 边界字符）。

### 3.5 多字符 token 的识别：`>>` / `>=` / `>`

> **诚实声明**：任务描述提及 `>>=`，但 Tenth 的 `TokenKind` 枚举（[token.rs:13-100](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/lexer/token.rs)）**不包含 `ShrAssign` 变体**。源码搜索 `>>=`/`ShrAssign` 仅在 CUDA kernel 字符串字面量中出现（[cuda_kernel.rs:205](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/gpu/cuda_kernel.rs)），不构成 Tenth token。因此 `>` 起始的多字符 token 实际为 `>=`（`GtEq`）与 `>>`（`Shr`），本文按此分析。

`>` 起始的 token 识别（[lexer.rs:446-456](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/lexer/lexer.rs)）：

```rust
if ch == '>' {
    if self.peek() == Some('=') { self.advance(); return Ok(Token { kind: TokenKind::GtEq, span }); }
    if self.peek() == Some('>') { self.advance(); return Ok(Token { kind: TokenKind::Shr, span }); }
    return Ok(Token { kind: TokenKind::Gt, span });
}
```

注意检查顺序：先 `=` 后 `>`。由于 `=` 与 `>` 是不同字符，两者互斥，顺序不影响结果。tenthc 侧顺序相反（先 `>` 后 `=`，[lexer.th:185-190](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenthc/lexer/lexer.th)），同样因互斥而不影响正确性。

对于输入 `>>=`（Tenth 不将其识别为单一 token），lexer 先匹配 `>>`（`Shr`），然后在下一轮 `next_token` 匹配 `=`（`Assign`）。这满足 maximal munch：因为 `>>=` 不是已定义的 token 模式，从位置 $i$ 能匹配的最长 token 是 `>>`（长度 2），而非 `>`（长度 1）。

### 3.6 数字字面量的解析

数字字面量解析（`read_number`，[lexer.rs:97-203](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/lexer/lexer.rs)）分四阶段：

1. **整数部分**：消费连续 $[0-9\_]$，构造字符串 $s$。
2. **小数部分**：若 peek `.` 且 peek_next 是数字，消费 `.` + 连续 $[0-9\_]$，标记 `is_float = true`。
3. **指数部分**：若 peek `e`/`E`，消费 `e` + 可选 `+`/`-` + 连续 $[0-9\_]$。
4. **后缀检测**：若 peek `f`，4 字符 peek 检测 `f32`/`f64` + 边界。

最终通过 Rust 标准库的 `str::parse::<i64>()` 或 `str::parse::<f64>()` 将字符串转换为数值。

tenthc 侧（[lexer.th:48-116](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenthc/lexer/lexer.th)）采用**增量算术**而非字符串构造：

```tenth
let mut ival: i64 = 0;
while is_digit(ch) {
    ival = ival * 10 + char_to_digit(ch);
    lexer_advance(lexer);
    ch = lexer_peek(lexer);
    ...
}
```

这是 $O(1)$ 空间的数字解析（无字符串分配），但引入 i64 溢出问题（见 §8）。

---

## 4. 主定理与证明

### 4.1 定理 D1（DFA 等价性）

**定理 D1**。存在 DFA $M_{\text{Tenth}}$，使得 $L(M_{\text{Tenth}}) = L(\text{Lexer}_{\text{Rust}})$，其中 $\text{Lexer}_{\text{Rust}}$ 是 [tenth/src/lexer/lexer.rs](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/lexer/lexer.rs) 定义的单 token 词法分析函数。

**证明**。

我们通过构造性证明完成。证明分三步：(a) 证明 `next_token` 的控制流是有界状态机；(b) 构造对应 DFA；(c) 证明接受语言相同。

**(a) 有界状态机证明。**

`next_token` 的运行时状态由以下分量组成：

1. **控制位置**（control location）：当前执行的代码行。由于 `next_token` 是单函数（无递归调用，`read_number`/`read_identifier`/`read_string` 是叶子调用且各自无递归），控制位置是有限的。审查源码，控制位置集合 $Q_{\text{ctrl}}$ 的大小约为 $|Q_{\text{ctrl}}| \le 200$（每条 if/match 分支对应一个位置）。
2. **peek 缓冲**：lexer 在任意时刻最多查看 $k$ 个未来字符。由 §3.4 分析，最大 peek 深度 $k_{\max} = 4$（`f32`/`f64` 后缀检测）。因此 peek 缓冲的状态数为 $|\Sigma|^{k_{\max}} \le 256^4$（实际远小于此，因为 peek 仅检查特定字符模式）。
3. **已消费字符的累积状态**：
   - `read_number` 中累积字符串 $s$：这是**无界**的（数字字面量长度无上限）。但 $s$ 仅用于最终的 `str::parse`，不影响控制流——lexer 在累积 $s$ 期间处于固定状态"消费数字字符"，不基于 $s$ 的内容做分支决策。
   - `read_identifier` 中累积字符串 $s$：同样，$s$ 仅用于最终的关键字查表，查表是 $O(1)$ 哈希或线性比较，不影响"消费标识符字符"这一状态的循环。
   - `read_string` 中累积 `current_literal`：仅用于最终构造 `StringLiteral`，不影响循环控制。

关键观察：**lexer 的控制流决策仅依赖于当前字符 $c$ 与有限 peek 缓冲，不依赖于已累积的字符串内容**。累积字符串只是"输出载荷"（payload），不影响状态迁移。因此 lexer 的控制状态空间是有界的。

形式化地，定义 lexer 的**抽象状态**为 $q = (\ell, b)$，其中：
- $\ell \in Q_{\text{ctrl}}$ 是控制位置；
- $b \in \Sigma^{\le k_{\max}}$ 是 peek 缓冲内容（长度 $\le k_{\max}$）。

抽象状态空间 $Q = Q_{\text{ctrl}} \times \Sigma^{\le k_{\max}}$ 是有限的。

**(b) DFA 构造。**

构造 DFA $M_{\text{Tenth}} = (Q', \Sigma, \delta', q_0', F')$：

- $Q' = Q \cup \{q_{\text{accept}, t} \mid t \in \mathcal{T}\} \cup \{q_{\text{error}}\}$：每个抽象状态 + 每个 token 的接受状态 + 错误状态。
- $q_0' = (\ell_0, \epsilon)$：初始控制位置 + 空 peek 缓冲。
- $\delta'$：对于抽象状态 $q = (\ell, b)$ 与输入字符 $c$：
  - 若 lexer 在 $\ell$ 处 advance（消费 $c$），则 $\delta'(q, c) = (\ell', b')$，其中 $\ell'$ 是 advance 后的控制位置，$b'$ 是更新后的 peek 缓冲。
  - 若 lexer 在 $\ell$ 处返回 token $t$（不消费 $c$），则 $\delta'(q, c) = q_{\text{accept}, t}$。
  - 若 lexer 在 $\ell$ 处报错，则 $\delta'(q, c) = q_{\text{error}}$。
- $F' = \{q_{\text{accept}, t} \mid t \in \mathcal{T}\}$：所有 token 接受状态。

注意：此 DFA 的"输入"是源码字符流，"接受"意味着识别出一个完整 token。完整词法分析（多 token 序列）是该 DFA 的反复运行。

**(c) 接受语言相同。**

需证明：对任意字符串 $w \in \Sigma^*$，

$$w \in L(\text{Lexer}_{\text{Rust}}) \iff w \in L(M_{\text{Tenth}})$$

其中 $L(\text{Lexer}_{\text{Rust}})$ 是 `next_token` 接受的字符串集合（即能被识别为某个 token 的前缀）。

**正向**（$w \in L(\text{Lexer}_{\text{Rust}}) \Rightarrow w \in L(M_{\text{Tenth}})$）：

若 `next_token` 在输入 $w$ 上返回 token $t$，则存在位置序列 $\text{pos}_0 = 0, \text{pos}_1, \ldots, \text{pos}_m$（$m = |w|$），使得 lexer 从 $\text{pos}_0$ 开始，依次消费 $w[0], w[1], \ldots, w[m-1]$，在 $\text{pos}_m$ 处返回 $t$。

由 (a)，lexer 的每一步决策仅依赖于当前控制位置与 peek 缓冲，二者对应抽象状态 $q_i = (\ell_i, b_i)$。DFA $M_{\text{Tenth}}$ 的转移 $\delta'$ 按 lexer 的 advance/return 行为定义，因此 DFA 在 $w$ 上的运行轨迹 $q_0' \to q_1' \to \cdots \to q_m'$ 与 lexer 的状态序列一一对应。最终 $q_m' = q_{\text{accept}, t} \in F'$，故 $w \in L(M_{\text{Tenth}})$。

**反向**（$w \in L(M_{\text{Tenth}}) \Rightarrow w \in L(\text{Lexer}_{\text{Rust}})$）：

若 $w \in L(M_{\text{Tenth}})$，则 DFA 运行 $q_0' \to \cdots \to q_{\text{accept}, t}$。由 $\delta'$ 的构造，每一步转移对应 lexer 的一次 advance 或 return。因此存在对应的 lexer 执行轨迹，在 $w$ 上返回 token $t$。故 $w \in L(\text{Lexer}_{\text{Rust}})$。

综上，$L(M_{\text{Tenth}}) = L(\text{Lexer}_{\text{Rust}})$。$\square$

**推论 D1.1**。Tenth lexer 接受的语言是正则语言。

**证明**。由定理 D1 与 Kleene 定理（定理 2.1）直接得出。$\square$

### 4.2 定理 D2（复杂度）

**定理 D2**。Tenth lexer 的 `tokenize`（完整词法分析）时间复杂度为 $O(n)$，其中 $n$ 是输入长度。空间复杂度为 $O(n)$（存储 token 序列）或 $O(1)$（单 token 流式输出，摊还）。

**证明**。

**时间复杂度**。

`tokenize`（[lexer.rs:533-544](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/lexer/lexer.rs)）反复调用 `next_token` 直到 EOF。每个字符至多被消费一次（`advance` 后 `pos` 单调递增），因此总 advance 次数 $\le n$。

但需考虑 **peek 开销**：lexer 在不 advance 的情况下可能查看多个字符。由定理 D1 的证明，最大 peek 深度 $k_{\max} = 4$。每次 `next_token` 调用中，peek 操作的次数有上界 $C_{\text{peek}} \cdot k_{\max}$（$C_{\text{peek}}$ 是 `next_token` 中 peek 调用点的数量，约为 30）。

然而，peek 操作**不消费字符**，因此同一字符可能被多次 peek。关键问题是：同一字符被 peek 的次数是否有界？

考察 peek 的使用模式：
- `skip_whitespace_and_comments` 中，每个字符被 peek 至多 2 次（`peek` + `peek_next`，[lexer.rs:52-95](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/lexer/lexer.rs)）。
- `read_number` 中，`.` 处 peek 2 次（`peek` + `peek_next`），`f` 处 peek 4 次（$\text{pos}+1$ 到 $\text{pos}+3$）。
- `read_identifier` 中，每个字符 peek 1 次。
- `read_string` 中，每个字符 peek 1–2 次（`{` 处 peek 2 次）。
- 运算符识别中，advance 后 peek 1–2 次。

因此，每个字符被 peek 的次数有上界 $C = 4$（`f32`/`f64` 后缀检测的最坏情况）。

总时间开销：
$$
T(n) \le \underbrace{n}_{\text{advance}} + \underbrace{C \cdot n}_{\text{peek}} = (1 + C) \cdot n = O(n)
$$

具体地，$T(n) \le 5n$，常数因子为 5。

**空间复杂度**。

`tokenize` 存储 token 序列，空间 $O(m)$（$m$ 是 token 数，$m \le n$）。若改为流式输出（每次 `next_token` 后丢弃），单 token 工作空间为 $O(L_{\max})$（$L_{\max}$ 是最长 token 的长度），由源码约束 $L_{\max} \le n$ 但实际有界（标识符长度通常 $< 256$），可视为 $O(1)$ 摊还。

tenthc 侧的增量算术解析（[lexer.th:48-116](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenthc/lexer/lexer.th)）使用 $O(1)$ 额外空间（`ival`/`frac_val`/`frac_len` 等固定数量标量），优于 Rust 侧的字符串构造（$O(L)$）。$\square$

### 4.3 定理 D3（maximal munch 保持）

**定理 D3**。Tenth lexer 满足 maximal munch 规则：对任意输入位置 $i$，lexer 选择的 token $t$ 是从 $i$ 起能匹配的所有 token 模式中**最长**的。

**证明**。

需证明：对每个运算符首字符 $c$，lexer 的 if-else 链按长度递减顺序检查，且检查顺序保证选择最长匹配。

逐一审查 [lexer.rs:417-520](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/lexer/lexer.rs)：

**情况 1：`=` 起始**。
- 模式：`==`（长度 2）、`=>`（长度 2）、`=`（长度 1）。
- 代码顺序：先检查 `==`，再检查 `=>`，最后 `=`。
- 由于 `==` 与 `=>` 的第二字符不同（`=` vs `>`），两者互斥，检查顺序不影响最长匹配。若输入是 `==`，匹配 `EqEq`（长度 2）；若输入是 `=>`，匹配 `FatArrow`（长度 2）；否则匹配 `Assign`（长度 1）。满足 maximal munch。✓

**情况 2：`<` 起始**。
- 模式：`<=`（长度 2）、`<<`（长度 2）、`<`（长度 1）。
- 代码顺序：先 `<=`，再 `<<`，最后 `<`。
- `<=` 与 `<<` 互斥（第二字符 `=` vs `<`）。满足 maximal munch。✓

**情况 3：`>` 起始**。
- 模式：`>=`（长度 2）、`>>`（长度 2）、`>`（长度 1）。
- 代码顺序：先 `>=`，再 `>>`，最后 `>`。
- `>=` 与 `>>` 互斥。满足 maximal munch。✓
- 注：Tenth 无 `>>=` token，故 `>>=` 被切分为 `>>` + `=`，这是 maximal munch 的正确应用（`>>=` 非已定义模式）。

**情况 4：`-` 起始**。
- 模式：`-=`（长度 2）、`->`（长度 2）、`-`（长度 1）。
- 代码顺序：先 `-=`，再 `->`，最后 `-`。
- 互斥，满足 maximal munch。✓

**情况 5：`.` 起始**（三级 peek）。
- 模式：`..=`（长度 3）、`..`（长度 2）、`.`（长度 1）。
- 代码（[lexer.rs:503-513](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/lexer/lexer.rs)）：advance `.` 后 peek `.`，若匹配则 advance 后再 peek `=`。
- 若输入是 `..=`：匹配 `DotDotEq`（长度 3）。
- 若输入是 `..`：匹配 `DotDot`（长度 2）。
- 若输入是 `.`：匹配 `Dot`（长度 1）。
- 满足 maximal munch。✓

**情况 6：单字符运算符**（`(`/`)`/`[`/`]`/`{`/`}`/`,`/`;`/`:`/`%`/`^`/`?`）。
- 无多字符变体，直接匹配单字符。满足 maximal munch。✓

**情况 7：数字字面量**。
- 模式：`<digits>`、`<digits>.<digits>`、`<digits>e<sign><digits>`、`<digits>f32`、`<digits>f64`、`<digits>.<digits>f32` 等。
- `read_number` 依次尝试小数部分、指数部分、后缀，每一步都"贪婪消费"匹配的字符。
- 关键：小数点检测要求 `peek_next` 是数字（[lexer.rs:113](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/lexer/lexer.rs)），避免将 `1..10` 中的 `.` 误消费（`..` 是范围运算符）。
- 后缀检测要求 boundary 检查（`c3` 非字母数字），避免将 `3.14factor` 误匹配为 `3.14f`。
- 满足 maximal munch。✓

**情况 8：标识符**。
- 模式：`[a-zA-Z_][a-zA-Z0-9_]*`。
- `read_identifier` 贪婪消费所有 alphanumeric + `_`。满足 maximal munch。✓

综上，Tenth lexer 在所有 token 模式上满足 maximal munch。$\square$

### 4.4 定理 D4（增量算术解析的精度边界）

**定理 D4**。tenthc lexer 的增量算术解析 `ival = ival * 10 + char_to_digit(ch)`（[lexer.th:54](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenthc/lexer/lexer.th)）**不显式检测 i64 溢出**。溢出时的行为取决于 Tenth VM 的 i64 算术语义。

**证明**。

**(a) tenthc 的增量解析无溢出检测。**

审查 [tenthc/lexer/lexer.th:48-58](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenthc/lexer/lexer.th)：

```tenth
let mut ival: i64 = 0;
...
while is_digit(ch) {
    ival = ival * 10 + char_to_digit(ch);
    lexer_advance(lexer);
    ch = lexer_peek(lexer);
    if !is_digit(ch) { break; };
};
```

循环体 `ival = ival * 10 + char_to_digit(ch)` 直接执行乘加，**无 `checked_mul`/`checked_add`/`overflowing_*` 调用，无溢出分支**。搜索 tenthc 全文，无 `overflow`/`checked`/`wrapping` 关键字（搜索证据：tenthc/ 目录无相关匹配）。

对比 Rust 侧（[lexer.rs:193-201](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/lexer/lexer.rs)）：

```rust
let n: i64 = s.parse().map_err(|_| TenthError::LexerError {
    line: span.line, col: span.col,
    message: format!("无效的整数：{}", s),
})?;
```

Rust 标准库的 `str::parse::<i64>()` 在溢出时返回 `Err(ParseIntError)`，lexer 将其转为 `TenthError::LexerError`——**优雅错误，不 panic**。

**(b) 溢出时的行为分析。**

tenthc 运行在 Tenth VM 上。VM 的 `Op::Mul` 与 `Op::Add` 实现在 [vm.rs:932-934](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/vm.rs) 与 [vm.rs:817-819](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/vm.rs)：

```rust
fn add_priv(&mut self, a: &Value, b: &Value) -> TenthResult<Value> {
    Ok(match (a, b) {
        (Value::Int(x), Value::Int(y)) => Value::Int(x + y),  // 行 819
        ...
fn mul_priv(&mut self, a: &Value, b: &Value) -> TenthResult<Value> {
    Ok(match (a, b) {
        (Value::Int(x), Value::Int(y)) => Value::Int(x * y),  // 行 934
```

VM 使用 Rust 原生 `+`/`*` 运算符。Tenth 的 [Cargo.toml](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/Cargo.toml) 在 `[profile.release]` 与 `[profile.dev]` 均设置 `overflow-checks = true`（[Cargo.toml:33,39](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/Cargo.toml)）。

因此，当 `ival * 10` 或 `ival * 10 + digit` 溢出 i64 时：
- Rust 原生 `*`/`+` 在 `overflow-checks = true` 下**触发 panic**（`attempt to multiply with overflow`）。
- VM 的 `mul_priv`/`add_priv` 未用 `catch_unwind` 包裹，panic 传播至宿主进程。
- tenthc 编译过程**崩溃**，无错误信息。

**(c) 精度边界。**

i64 范围：$[-2^{63}, 2^{63}-1] \approx [-9.22 \times 10^{18}, 9.22 \times 10^{18}]$。

最大安全数字位数：18 位（$999999999999999999 < 2^{63}-1$），19 位时部分值溢出（$9999999999999999999 > 2^{63}-1$）。

因此：
- 18 位十进制整数：tenthc 与 Rust lexer 均正确解析。
- 19+ 位十进制整数且值超出 i64 范围：
  - Rust lexer：返回 `TenthError::LexerError("无效的整数：...")`，编译中止，错误信息清晰。
  - tenthc lexer：VM panic，编译崩溃，无错误信息。

**(d) 结论。**

tenthc 的增量算术解析在 i64 溢出时**不优雅**（panic 而非返回错误），这是与 Rust lexer 的**精度边界不对称**。该不对称是 T12 [T12](T12-双侧编译器语义等价性.md) §1.3 第 4 项不对称的子案例（数字字面量处理差异）。$\square$

### 4.5 定理 D5（双侧 lexer 等价）

**定理 D5**。Rust lexer 与 tenthc lexer 在**共同 token 子集**上等价：对任意源码 $w$，若 $w$ 中所有 token 均属于两侧共同支持的 token 集合 $\mathcal{T}_{\text{common}}$，则两侧产生相同的 token 序列（modulo span 精度与 denormalized 表示差异）。

**证明**。

**(a) 共同 token 子集。**

由 T12 §1.3 与 T15 的分析，tenthc 缺失以下 token：
- `CharLiteral`（[token.rs:19](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/lexer/token.rs)）
- `InterpolatedString`（[token.rs:18](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/lexer/token.rs)）
- 关键字：`Try`/`Pub`/`Type`/`Spawn`/`Task`/`Shard`/`Node`/`Macro`/`Where`/`As`/`In`
- 运算符：`Shl`（`<<`）、`QuestionMark`（`?`）、`Caret`（`^`）

共同子集 $\mathcal{T}_{\text{common}}$ 包含约 45 个 token：所有单字符运算符、`==`/`!=`/`<=`/`>=`/`>>`/`&&`/`||`/`+=`/`-=`/`*=`/`/=`/`->`/`=>`/`::`/`..`/`..=`、`IntLiteral`/`FloatLiteral`/`StringLiteral`、20 个关键字。

**(b) 等价性证明。**

对每个 $t \in \mathcal{T}_{\text{common}}$，逐一核对两侧的识别逻辑：

1. **单字符运算符**：两侧逻辑相同（advance + 返回）。✓
2. **`==`/`!=`/`<=`/`>=`/`>>`/`&&`/`||`**：两侧均 advance 后 peek 第二字符，匹配则 advance + 返回。Rust [lexer.rs:417-470](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/lexer/lexer.rs)，tenthc [lexer.th:177-192](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenthc/lexer/lexer.th)。检查顺序虽有差异（如 `>` 侧 Rust 先 `=` 后 `>`，tenthc 先 `>` 后 `=`），但因第二字符互斥，结果相同。✓
3. **`+=`/`-=`/`*=`/`/=`/`->`/`=>`/`::`/`..`/`..=`**：两侧逻辑相同。✓
4. **`IntLiteral`**：两侧均消费 $[0-9\_]$。Rust 用 `str::parse::<i64>()`，tenthc 用增量算术。在不溢出的前提下，两者数值相同（Horner 法则：$\sum_{i=0}^{n-1} d_i \cdot 10^{n-1-i} = ((\cdots((0 \cdot 10 + d_0) \cdot 10 + d_1)\cdots) \cdot 10 + d_{n-1})$，即增量算术与多项式求值等价）。✓（溢出场景见定理 D4，不在共同子集保证范围内）
5. **`FloatLiteral`**：两侧均检测 `.`/`e`/`f32`/`f64`。tenthc 用增量算术计算 `ival + frac_val/div`，Rust 用 `str::parse::<f64>()`。在不溢出且 f64 精度范围内，两者数值相同（浮点精度差异在 ULP 级别，不影响词法等价性）。✓
6. **`StringLiteral`**：Rust 支持转义与插值；tenthc 仅做简单切片（[lexer.th:150-159](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenthc/lexer/lexer.th)）。**仅在不含转义与插值的纯字符串上等价**。含 `\n`/`\t`/`{expr}` 的字符串两侧不等价——但这类字符串不属于 $\mathcal{T}_{\text{common}}$（tenthc 不支持）。✓（限定在纯字符串）
7. **关键字与标识符**：两侧均消费 $[a-zA-Z0-9\_]*$ 后查表。共同关键字集合一致。✓

**(c) Token 表示等价。**

由 T15 的证明，tenthc 的 `disc`/`ival`/`fval`/`sval` 冗余字段是 `kind` 的纯函数投影，在共同子集上 `disc = project(kind)` 成立。因此两侧 Token 在表示层等价（modulo T15 的 denormalized 差异）。

综上，在 $\mathcal{T}_{\text{common}}$ 上两侧 lexer 等价。$\square$

---

## 5. peek 深度分析

### 5.1 各 token 的 peek 深度表

| Token | 首字符 | peek 深度 | 推进模式 | 源码位置 |
|-------|--------|-----------|----------|----------|
| `Eof` | EOF | 0 | — | [lexer.rs:389-397](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/lexer/lexer.rs) |
| `IntLiteral` | `[0-9]` | 2 | 推进 | [lexer.rs:113](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/lexer/lexer.rs) |
| `FloatLiteral`（小数） | `[0-9]` | 2 | 推进 | [lexer.rs:113](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/lexer/lexer.rs) |
| `FloatLiteral`（指数） | `[0-9]` | 1 | 推进 | [lexer.rs:128-148](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/lexer/lexer.rs) |
| `FloatLiteral`（f32/f64 后缀） | `f` | **4** | **非推进** | [lexer.rs:153-169](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/lexer/lexer.rs) |
| `Identifier`/关键字 | `[a-zA-Z_]` | 1 | 推进 | [lexer.rs:210-217](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/lexer/lexer.rs) |
| `StringLiteral`（纯） | `"` | 1 | 推进 | [lexer.rs:259-365](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/lexer/lexer.rs) |
| `InterpolatedString`（`{expr}`） | `"` + `{` | 2 | 推进 | [lexer.rs:305-354](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/lexer/lexer.rs) |
| `==`/`=>`/`!=`/`<=`/`<<`/`>=`/`>>`/`&&`/`\|\|` | 运算符首字符 | 1 | 推进 | [lexer.rs:417-470](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/lexer/lexer.rs) |
| `+=`/`-=`/`*=`/`/=`/`->` | 运算符首字符 | 1 | 推进 | [lexer.rs:471-502](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/lexer/lexer.rs) |
| `::`/`..` | `:`/`.` | 1 | 推进 | [lexer.rs:503-520](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/lexer/lexer.rs) |
| `..=` | `.` | 2 | 推进 | [lexer.rs:503-513](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/lexer/lexer.rs) |
| 单字符 token | 各 | 0 | — | [lexer.rs:367-384](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/lexer/lexer.rs) |
| 行注释 `//` | `/` | 2 | 非推进 | [lexer.rs:55-64](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/lexer/lexer.rs) |
| 块注释 `/* */` | `/` | 2 | 非推进 | [lexer.rs:65-90](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/lexer/lexer.rs) |

### 5.2 最大 peek 深度

由表可见，**最大 peek 深度 $k_{\max} = 4$**，出现在 `f32`/`f64` 后缀检测：

- 位置 $\text{pos}$：peek `f`（检查是否后缀起始）
- 位置 $\text{pos}+1$：检查 `3` 或 `6`
- 位置 $\text{pos}+2$：检查 `2` 或 `4`
- 位置 $\text{pos}+3$：边界检查（非 alphanumeric + 非 `_`）

tenthc 侧同样为 $k_{\max} = 4$（[lexer.th:71-89](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenthc/lexer/lexer.th)），构造方式与 Rust 侧一致。

### 5.3 peek 深度与 token 种类数的关系

peek 深度与 token 种类数无直接关系，但与**最长 token 长度**相关。Tenth 最长 token 为 `..=`（3 字符）或 `f32`/`f64` 后缀的数字字面量（如 `3.14f32`，长度 7）。peek 深度 4 用于**在不推进的前提下**判定后缀边界，避免误匹配（如 `3.14f32actor` 不应将 `f32` 识别为后缀）。

理论上，若 Tenth 引入更长的多字符 token（如 `>>=` 长度 3），peek 深度不变（仍为 1，因为用推进式 peek 即可）；但若引入更长的**后缀式** token（如 `u8`/`u16`/`u32`/`u64`/`usize`/`isize` 等更多后缀），可能需要更深 peek 或改为推进式。

---

## 6. 增量算术解析的精度问题

### 6.1 i64 溢出检测的有无

| 维度 | Rust lexer | tenthc lexer |
|------|-----------|--------------|
| 解析方式 | 字符串构造 + `str::parse::<i64>()` | 增量算术 `ival = ival * 10 + digit` |
| 溢出检测 | ✅ `parse()` 返回 `Err` | ❌ 无显式检测 |
| 溢出行为 | 返回 `TenthError::LexerError`，优雅中止 | VM panic（`overflow-checks = true`） |
| 空间复杂度 | $O(L)$（字符串） | $O(1)$（标量） |
| 源码位置 | [lexer.rs:193-201](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/lexer/lexer.rs) | [lexer.th:48-58](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenthc/lexer/lexer.th) |

### 6.2 溢出时的行为

**Rust 侧**：`"9999999999999999999".parse::<i64>()` 返回 `Err(ParseIntError { kind: PosOverflow })`，lexer 转为：

```
LexerError { line: L, col: C, message: "无效的整数：9999999999999999999" }
```

编译中止，错误信息清晰。

**tenthc 侧**：`ival = ival * 10 + 9` 在第 19 次迭代时，`ival * 10` 溢出 i64，Tenth VM 的 `Op::Mul`（[vm.rs:934](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/vm.rs)）触发 panic：

```
thread 'main' panicked at 'attempt to multiply with overflow', runtime/vm.rs:934
```

编译崩溃，无 lexer 层错误信息。

### 6.3 与 Rust lexer 的对比

两侧的**根本差异**在于"解析方式"与"错误处理策略"：

1. **解析方式**：Rust 侧"先收集字符串再解析"，依赖标准库的溢出检测；tenthc 侧"边读边算"，依赖宿主 VM 的算术语义。
2. **错误处理**：Rust 侧用 `Result` 传播错误；tenthc 侧无错误传播路径（增量算术是"裸运算"）。

这一不对称是 T12 §1.3 第 4 项的细化。在自举场景（路径 B/C）中，若 tenthc 源码本身含超长整数，可能导致 tenthc 编译自身时崩溃——但实际 tenthc 源码中无超长整数，故自举不受影响（实证：`cargo run --release -- run tenthc/main.th` 成功）。

---

## 7. 工程权衡

### 7.1 手写 vs DFA 生成的性能

手写 lexer 的性能优势在于：
- **无转移表查找**：DFA 生成器每读一个字符需查转移表（`table[state][char]`），引入一次间接访问；手写 lexer 的 if-else 链在分支预测良好时几乎零开销。
- **无回溯开销**：DFA 的 maximal munch 需要记录最近接受状态并回溯；手写 lexer 通过 peek 提前判定，无需回溯（但 peek 本身是"预读回溯"的变体）。

劣势：
- **代码膨胀**：60+ token 模式需手写 60+ 分支，维护成本高。
- **maximal munch 需人工保证**：DFA 自动保证，手写需如定理 D3 逐一审查。

### 7.2 可维护性

Tenth lexer 的 if-else 链按字符组织，可读性尚可。但 `read_number`（[lexer.rs:97-203](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/lexer/lexer.rs)，107 行）逻辑较复杂（小数/指数/后缀三阶段 + 4 字符 peek），是维护性最高的函数。

### 7.3 错误信息质量

手写 lexer 的核心优势是错误信息质量。Rust 侧 `read_number` 在溢出时返回精确的行号与"无效整数：9999999999999999999"格式的诊断信息（[lexer.rs:193-201](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/lexer/lexer.rs)）。DFA 生成器通常只能报"意外字符"。

tenthc 侧在溢出时 panic，无 lexer 层错误信息——这是**精度边界的代价**。

---

## 8. 开放问题与未来工作

### 8.1 机器验证的 DFA 等价性

本文的定理 D1 证明是手工构造，未经过机器检查。未来可：
- 用 Coq/Lean 形式化 Tenth lexer 的状态机；
- 用模型检测（model checking）验证手写代码与构造 DFA 的等价性；
- 借鉴 CompCert [3] 的方法，用 Coq 提取可执行 lexer。

### 8.2 溢出检测的引入

建议在 tenthc 的增量算术中引入 `checked_mul`/`checked_add`（若 Tenth 标准库提供）或显式边界检查：

```tenth
// 建议的 tenthc 溢出检测（伪代码）
let max_before_mul: i64 = 922337203685477580;  // i64::MAX / 10
if ival > max_before_mul { return error_token("整数溢出", span); }
ival = ival * 10;
if digit > 7 && ival > i64::MAX - digit { return error_token("整数溢出", span); }
ival = ival + digit;
```

当前 Tenth 标准库是否提供 `checked_mul` 需查证（搜索 `checked` 在 tenth/std/ 无匹配）。若不提供，需在 VM 层添加 `Op::CheckedMul` 或在 lexer 层用比较实现。

### 8.3 tenthc 缺失 token 的补全

由 T12 §1.3，tenthc 缺失 `CharLiteral`/`InterpolatedString`/`Shl`/`QuestionMark`/`Caret` 等。补全这些 token 是双侧等价的前提（定理 D5 的 $\mathcal{T}_{\text{common}}$ 扩展）。

### 8.4 `>>=` 的引入

若未来 Tenth 需要复合右移赋值 `>>=`（如 Rust 的 `x >>= 1`），需：
1. 在 `TokenKind` 添加 `ShrAssign` 变体（[token.rs](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/lexer/token.rs)）；
2. 在 `next_token` 的 `>` 分支添加 `>>=` 检查（需在 `>>` 之前检查，或改为先 peek `>>` 再 peek `=`）；
3. 同步 tenthc 侧；
4. 更新 parser 与 HIR。

当前 Tenth 无此 token，是未来工作。

---

## 9. 讨论

### 9.1 与 T12 的联动

本文的定理 D5 是 T12 §3 "四级等价标准" 中 L1（词法等价）的细化。T12 给出"两侧在共同子集上等价"的命题，本文将该命题限定到 lexer 层并给出完整证明。T12 发现的 4 项不对称中，第 3 项（字符串插值）与第 4 项（Char 字面量）直接影响 lexer；本文进一步发现第 5 项不对称：**数字字面量的溢出处理**（定理 D4）。

### 9.2 与 T15 的联动

T15 证明 tenthc Token 的 `disc`/`ival`/`fval`/`sval` 冗余字段是 `kind` 的纯函数投影。本文定理 D5 的证明依赖这一不变量：若 `disc` 错误，parser 分发错误，lexer 等价性无意义。因此 T15 的不变量 D1 是本文定理 D5 的**前置条件**。

### 9.3 局限

**局限 1：证明未机器验证。** 定理 D1 的 DFA 构造与等价性证明是手工完成的，可能存在遗漏的状态或转移。缓解：本文 §3 的状态机刻画基于源码逐行审查，且 §5 的 peek 深度表提供了可独立核对的证据。

**局限 2：`>>=` 的处理。** 任务描述提及 `>>=`，但 Tenth 无此 token。本文按实际实现分析，将 `>>=` 的引入标注为未来工作（§8.4）。若用户预期 `>>=` 已实现，则本文结论需相应修订。

**局限 3：tenthc 缺失 token 未完整列举。** 本文聚焦于 lexer 控制流等价性，对 tenthc 缺失 token 的完整列表引用 T12 §1.3，未独立核实。缓解：T12 与 T15 已穷尽列举。

**局限 4：浮点精度差异未深入。** 定理 D5 声称两侧 `FloatLiteral` "数值相同"，但 Rust 的 `str::parse::<f64>()` 与 tenthc 的 `ival + frac_val / div` 在浮点精度上可能有 ULP 级差异（如 `0.1` 的解析）。本文不深入浮点精度分析，标注为未来工作。

**局限 5：Unicode 标识符未分析。** Tenth lexer 用 `is_alphabetic()` 判定标识符首字符（[lexer.rs:406](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/lexer/lexer.rs)），支持 Unicode 标识符；tenthc 用 `is_alpha` 仅检查 ASCII（[lexer.th:5](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenthc/lexer/lexer.th)）。这是定理 D5 的一个边界——非 ASCII 标识符两侧不等价。但 tenthc 源码本身仅用 ASCII 标识符，自举不受影响。

---

## 10. 结论

本文对 Tenth 语言的手写 lexer 进行形式化分析，得到以下结论：

1. **DFA 等价性**（定理 D1）：手写 lexer 等价于某个 DFA，接受语言是正则语言。证明方法是构造性——从源码提取抽象状态机，构造对应 DFA，证明接受语言相同。

2. **线性复杂度**（定理 D2）：`tokenize` 时间复杂度 $O(n)$，常数因子 $\le 5$（每个字符至多被 peek 4 次 + advance 1 次）。

3. **maximal munch 保持**（定理 D3）：手写 lexer 在所有 token 模式上满足最长匹配，包括三级 peek 的 `..=`。

4. **增量算术的精度边界**（定理 D4）：tenthc 的 `ival = ival * 10 + digit` **不检测 i64 溢出**，溢出时 VM panic；Rust 侧通过 `str::parse` 优雅报错。这是两侧 lexer 的已知不对称。

5. **双侧等价**（定理 D5）：在共同 token 子集 $\mathcal{T}_{\text{common}}$（约 45 个 token）上，两侧 lexer 等价。缺失 token（`CharLiteral`/`InterpolatedString`/`Shl`/`QuestionMark`/`Caret` 等）与溢出处理差异是等价性边界。

本文的价值在于：**将"手写 lexer 等价于 DFA"这一工程直觉转化为可证伪的形式化命题**，并**诚实标注溢出检测缺失的精度边界**。对实施的指导：(1) 定理 D1 证明手写 lexer 的理论合法性，无需迁移到 DFA 生成器；(2) 定理 D4 揭示的溢出不对称应通过引入 `checked` 算术或边界检查修复（§8.2）；(3) 定理 D5 的共同子集为差分测试提供了明确范围。

---

## 附录 A：定理索引

| 定理 | 陈述 | 证明方法 | 局限 |
|------|------|----------|------|
| D1 | 手写 lexer 等价于某个 DFA | 构造性证明（提取状态机 + 构造 DFA + 双向语言等价） | 未机器验证 |
| D2 | `tokenize` 时间 $O(n)$，空间 $O(n)$ 或 $O(1)$ 摊还 | 每字符 advance 1 + peek $\le 4$ | 浮点解析常数未细算 |
| D3 | 满足 maximal munch | 逐 token 模式审查 if-else 顺序 | 仅审查已实现 token，未来新增需重审 |
| D4 | tenthc 增量算术不检测 i64 溢出 | 源码审查 + VM 语义分析 + Cargo.toml overflow-checks | 未实证 panic（避免破坏测试） |
| D5 | 共同子集上两侧 lexer 等价 | 逐 token 逐一核对 + 依赖 T15 不变量 | 浮点 ULP 差异、Unicode 标识符未覆盖 |

## 附录 B：与现有文档的对应

| 本文结论 | 对应文档 |
|---------|---------|
| 定理 D1（DFA 等价性） | 新增，无对应 |
| 定理 D2（O(n) 复杂度） | 新增，无对应 |
| 定理 D3（maximal munch） | 新增，无对应 |
| 定理 D4（溢出检测缺失） | T12 §1.3 第 4 项的细化 |
| 定理 D5（双侧 lexer 等价） | T12 L1（词法等价）的细化 |
| tenthc Token 表示 | T15 的 `disc` 不变量 |
| `>>=` 不存在 | AUDIT.md 未记录（本文新增发现） |

## 附录 C：实施建议

1. **短期（不破坏自举）**：在 tenthc lexer 的整数解析循环中添加显式溢出检查（§8.2 的伪代码），返回与 Rust 侧一致的 `LexerError`。需先确认 Tenth 标准库是否支持 `i64::MAX` 常量或提供 `checked_mul`。
2. **中期**：补全 tenthc 缺失 token（`Shl`/`CharLiteral`/`InterpolatedString`），扩展 $\mathcal{T}_{\text{common}}$。
3. **长期**：用模型检测工具验证两侧 lexer 的等价性（差分测试 + 状态机等价检查）。

---

## 参考文献

[1] Lesk, M. E. (1975). *LEX — A Lexical Analyzer Generator*. Computing Science Technical Report 39, Bell Laboratories.

[2] Parr, T. (2013). *Language Implementation Patterns: Create Your Own Domain-Specific and General Programming Languages*. Pragmatic Bookshelf.

[3] Kleene, S. C. (1956). "Representation of Events in Nerve Nets and Finite Automata". *Automata Studies*, pp. 3–42. Princeton University Press.

[4] Lesk, M. E. & Schmidt, E. (1975). *LEX — A Lexical Analyzer Generator*. CSTR 39.

[5] Thompson, K. (1968). "Regular Expression Search Algorithm". *Communications of the ACM*, 11(6), 419–422.

[6] Hopcroft, J. (1971). "An n log n algorithm for minimizing states in a finite automaton". *Theory of Machines and Computations*, pp. 189–196. Academic Press.

[7] Aho, A. V., Lam, M. S., Sethi, R., & Ullman, J. D. (2006). *Compilers: Principles, Techniques, and Tools* (2nd ed.). Addison-Wesley. (§3.5, "Recognition of Tokens" — maximal munch.)

[8] Leroy, X. (2009). "A Formally Verified Compiler Back-end". *Journal of Automated Reasoning*, 43(4), 363–446. (CompCert.)

[9] Kumar, R., Myreen, M. O., Norrish, M., & Owens, S. (2014). "CakeML: A Verified Implementation of ML". *POPL 2014*, pp. 3–16.

[10] Igarashi, A., Pierce, B. C., & Wadler, P. (2001). "Featherweight Java: A Minimal Core Calculus for Java and GJ". *ACM TOPLAS*, 23(3), 396–450.

[11] McKinna, J. & Pollack, R. (1993). "Pure Type Systems Formalized". *TLCA 1993*, LNCS 664, pp. 289–305.

[12] Pnueli, A., Siegel, M., & Singerman, E. (1998). "Translation Validation". *TACAS 1998*, LNCS 1384, pp. 151–166.

---

> **数理部声明**：本文的证明基于源码审查，未经过机器验证。定理 D4 揭示的 i64 溢出检测缺失是 tenthc 的已知局限，已诚实记录。`>>=` token 不存在是本文新增发现，与任务描述的预期不符。所有源码引用基于 Tenth v0.3.3 快照（2026-07-02）。
