# 双重 native 注册的协议一致性：Tenth 双源真相反模式的形式化与修补

> 编号：T37 · 数理部 · v1.0
> 撰写日期：2026-07-02
> 适用版本：Tenth v0.3.3+
> 关联论文：T34（栈式 VM 操作语义形式化）、T9（JIT 特化语义保持证明）、T12（双侧编译器语义等价性）、T32（hostcall trampoline FFI 安全性）
> 待关联：T35（双执行引擎等价性，待撰写）

---

## 摘要

Tenth 语言同时维护两条执行路径——基于字节码栈式 VM 的快速路径（默认 `tenth run`）与基于 tree-walk 解释器的回退路径。两条路径各自独立维护一份 native 函数注册表：VM 路径位于 `main.rs::register_natives(&mut Vm)`，解释器路径位于 `runtime/interpreter/natives.rs::call_named_fn`。这种**双源真相（dual source of truth）**结构构成一种系统性的反模式：任一 native 若仅在一侧注册，则在另一侧路径下会以"返回 `Unit`"或"undefined function"的形式静默失败，且故障无编译期信号。

本文对 Tenth 的双重 native 注册结构进行形式化建模，将其抽象为一对注册函数 $R_V, R_I$ 与一组原生算子签名 $\mathcal{N}$ 之间的的"双映射一致"关系，给出五个主定理：(N1) 双重注册不变量；(N2) 历史教训的形式化与实证（`zeros/ones/rand/randn` 曾缺失）；(N3) 当前注册的完备性检查（发现 VM 路径缺失 17 项、解释器路径缺失 3 项）；(N4) 与 Python C-extension/Lua C function 注册机制对比；(N5) 基于声明宏的自动双重注册方案（标注为未来工作）。

**关键发现**：截至 v0.3.3，双重注册协议**未完备**。VM 路径缺失 `to_string`、`type_name`、`with_step_limit`、`with_timeout_ms`、`is_timeout`、`start_grad`、`f64_bits`、`f64_from_bits`、`sin`、`cos`、`ln`、`pow`、`save_weights`、`load_weights`、`format`、`parse_int`、`parse_float` 共 17 项；解释器路径缺失 `to_f64`、`to_f32`、`print` 共 3 项。论文以独立"局限"章节披露形式化方法无法覆盖的边界与潜在的循环论证风险。

**关键词**：Tenth 语言；native 注册；双源真相；执行引擎等价性；声明宏；FFI 协议

---

## 1 引言

### 1.1 双重注册的反模式

在多执行引擎语言实现中，"native 函数"（即由宿主语言 Rust 直接实现、暴露给 Tenth 程序调用的内置函数）需要在每条执行路径上独立绑定。Tenth 当前维护两条路径：

- **路径 A（默认 VM 路径）**：`.th → Lexer → Parser → HIR → Bytecode → 栈式 VM`，由 `main.rs::register_natives` 注入。
- **路径 B（解释器回退路径）**：当 VM 编译失败或显式请求解释器时，HIR 直接由 tree-walk 解释器执行，native 由 `interpreter::natives.rs::call_named_fn` 分派。

由于两份注册表是物理上分离的 Rust 代码（一份是命令式 `vm.add_native(name, closure)` 调用序列，另一份是 `match name { ... }` 分派函数），它们构成两个独立的真相源。这种结构违反"单一真相源（single source of truth, SSoT）"原则，是经典的**双源真相反模式**。

### 1.2 历史教训

Tenth 项目在 v0.3.x 演化过程中实际遭遇过此反模式引发的故障：张量构造函数 `zeros/ones/rand/randn` 早期仅在解释器路径实现，VM 路径下程序调用 `zeros(256,256,256)` 会得到 `Unit` 而非张量，进而触发下游 `.numel()` 等方法的运行时崩溃。源码注释（见 [tenth/src/main.rs:1042-1044](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/main.rs)）显式记录了这一历史：

```rust
// ── Tensor 构造函数（与 interpreter::natives 对齐，支持任意 shape）──
// 历史：这些函数仅在 interpreter 实现，JIT/VM 路径下返回 Unit。
// 补齐后 zeros(256,256,256).numel() 等才能在默认 tenth run 路径下正常工作。
```

该注释本身即是双源真相反模式在工程实践中留下的"化石证据"。

### 1.3 贡献

本文贡献如下：

1. **形式化建模**：将 native 注册机制抽象为代数对象 $(\mathcal{N}, R_V, R_I)$，定义"双映射一致"关系 $\cong_{\text{reg}}$（第 4 节）。
2. **不变量定理**：给出双重注册不变量 N1，明确两侧协议一致性的充要条件（第 5.1 节）。
3. **历史教训实证**：将 `zeros/ones/rand/randn` 故障形式化为定理 N2，给出故障路径的可执行归约（第 5.2 节）。
4. **完备性审计**：基于 v0.3.3 源码统计两侧注册表，定理 N3 给出当前完备性差距：VM 路径缺 17 项、解释器路径缺 3 项（第 5.3 节、第 8 节）。
5. **横向对比**：与 Python C-extension、Lua C function 的注册机制对比（定理 N4，第 5.4 节、第 9 节）。
6. **修补方案**：基于 Rust 声明宏的自动双重注册方案 N5，标注为未来工作（第 5.5 节、第 10 节）。
7. **诚实局限**：独立章节披露形式化方法的边界与循环论证风险（第 12 节）。

---

## 2 背景：宿主语言绑定机制

### 2.1 Python C-extension 注册

Python 通过 `PyMethodDef` 数组在模块初始化时注册 C 扩展函数（见 CPython `Include/methodobject.h`）。每个 C 函数以统一签名 `PyObject* func(PyObject* self, PyObject* args, PyObject* kwargs)` 出现，由解释器主循环统一调度。**关键性质**：Python 仅维护一份注册表，CPython 解释器是唯一执行引擎（即使是 Cython/PyPy，也通过同一 C-API 暴露）。因此 Python 不存在 Tenth 式的双重注册问题。

### 2.2 Lua C function 注册

Lua 通过 `luaL_Reg` 数组 + `luaL_setfuncs` 在栈上注册 C 函数（见 Lua `lauxlib.c`）。所有 C 函数签名为 `int (*)(lua_State*)`，由 Lua VM 统一调度。Lua 也不存在 Tenth 式的双重注册问题——Lua 的"解释器"与"VM"是同一对象。

### 2.3 Tenth 的特殊性

Tenth 同时维护栈式 VM（`runtime/vm.rs`）与 tree-walk 解释器（`runtime/interpreter/`），二者并存于同一二进制中。VM 是默认路径，解释器在 VM 编译失败时回退（见 [tenth/src/main.rs:1278-1285](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/main.rs)）。这种"双引擎同存"的设计强制 native 必须双注册，是 Python/Lua 所不具备的工程约束。

---

## 3 预备符号

| 符号 | 含义 |
|------|------|
| $\mathcal{N}$ | Tenth 语言规范定义的全部 native 函数名集合 |
| $\Sigma$ | 值域（含 `Int/Float/Float32/String/Unit/Tensor/Vec/...`） |
| $\text{Sig}$ | 签名函数 $\text{Sig}: \mathcal{N} \to \Sigma^* \rightharpoonup \Sigma$（部分函数） |
| $R_V$ | VM 路径注册函数，$R_V: \mathcal{N} \rightharpoonup (\text{Vm} \times \Sigma^*) \to \Sigma$ |
| $R_I$ | 解释器路径注册函数，$R_I: \mathcal{N} \rightharpoonup (\text{Interp} \times \Sigma^*) \to \Sigma$ |
| $\text{dom}(R)$ | 注册函数的定义域，即已注册的 native 名集合 |
| $\text{eval}_V, \text{eval}_I$ | VM 与解释器的求值函数 |
| $\cong_{\text{reg}}$ | 双映射一致关系（定义 4.4） |
| $\text{sem}_V, \text{sem}_I$ | 两侧路径的程序语义函数 |

记号约定：$R(n) = \bot$ 表示 $n \notin \text{dom}(R)$；$R(n) \downarrow$ 表示求值终止且有结果。

---

## 4 Tenth 双重注册形式化

### 4.1 native 函数签名

**定义 4.1（native 函数）**：一个 native 函数是元组 $(n, \sigma, f_V, f_I)$，其中：
- $n \in \mathcal{N}$ 是函数名（字符串）；
- $\sigma$ 是参数/返回类型签名；
- $f_V: \text{Vm} \times \Sigma^* \rightharpoonup \Sigma$ 是 VM 路径实现；
- $f_I: \text{Interp} \times \Sigma^* \rightharpoonup \Sigma$ 是解释器路径实现。

### 4.2 注册函数

**定义 4.2（注册函数 $R_V$）**：VM 路径的注册函数 $R_V$ 由 `main.rs::register_natives` 定义，是一张从 native 名到闭包的映射：

$$R_V(n) = \begin{cases} f_V & \text{若 } \texttt{vm.add\_native}(n, f_V) \text{ 在 [main.rs:322-1149](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/main.rs) 中被调用} \\ \bot & \text{否则} \end{cases}$$

**定义 4.3（注册函数 $R_I$）**：解释器路径的注册函数 $R_I$ 由 `interpreter::natives.rs::call_named_fn` 的 `match name { ... }` 分支定义：

$$R_I(n) = \begin{cases} f_I & \text{若 } n \text{ 是 [natives.rs:40-1221](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/interpreter/natives.rs) 的某个 match 分支} \\ \bot & \text{否则（fallthrough 至用户函数查找）} \end{cases}$$

### 4.3 双映射一致

**定义 4.4（双映射一致 $\cong_{\text{reg}}$）**：native 名 $n \in \mathcal{N}$ 关于 $(R_V, R_I)$ 双映射一致，记 $n \cong_{\text{reg}} (R_V, R_I)$，当且仅当：
1. **域一致**：$n \in \text{dom}(R_V) \Leftrightarrow n \in \text{dom}(R_I)$；
2. **签名一致**：若 $n \in \text{dom}(R_V) \cap \text{dom}(R_I)$，则 $\text{Sig}_{R_V(n)} = \text{Sig}_{R_I(n)}$；
3. **语义一致**：对所有合法输入 $\bar v \in \Sigma^*$ 与状态 $s$，
$$\text{sem}_V(s, R_V(n), \bar v) = \text{sem}_I(s, R_I(n), \bar v)$$

**定义 4.5（全局双映射一致）**：注册协议 $(R_V, R_I)$ 关于 $\mathcal{N}$ 全局一致，记 $\mathcal{N} \cong_{\text{reg}} (R_V, R_I)$，当且仅当 $\forall n \in \mathcal{N}: n \cong_{\text{reg}} (R_V, R_I)$。

### 4.4 故障模式

**定义 4.6（单侧缺失故障）**：若 $n \in \text{dom}(R_I) \setminus \text{dom}(R_V)$，则在 VM 路径下调用 $n$ 会触发"undefined function $n$"运行时错误（[natives.rs:1268-1270](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/interpreter/natives.rs) 的等价分支），或返回 `Unit`（取决于调用点的字节码生成方式）。

**定义 4.7（语义偏移故障）**：若 $n \in \text{dom}(R_V) \cap \text{dom}(R_I)$ 但 $R_V(n)(\bar v) \neq R_I(n)(\bar v)$ 对某 $\bar v$ 成立，则双路径同输入产生不同结果，破坏执行引擎等价性（待 T35 形式化）。

---

## 5 主定理

### 5.1 定理 N1（双重注册不变量）

**定理 N1**：Tenth 的双执行引擎等价性（即 $\text{sem}_V = \text{sem}_I$ 在所有合法程序上成立）蕴含 $\mathcal{N} \cong_{\text{reg}} (R_V, R_I)$。

**证明**（反证）：

假设 $\text{sem}_V = \text{sem}_I$ 但 $\mathcal{N} \not\cong_{\text{reg}} (R_V, R_I)$。则存在 $n \in \mathcal{N}$ 使得至少下列其一成立：

**情形 1**：$n \in \text{dom}(R_I) \setminus \text{dom}(R_V)$。构造程序 $P_n = \texttt{let x = } n\texttt{(...); return x;}$，参数取 $R_I(n)$ 的合法输入 $\bar v$。则：
- $\text{sem}_I(P_n, \bar v) = R_I(n)(\bar v) \downarrow$（因 $n \in \text{dom}(R_I)$）；
- $\text{sem}_V(P_n, \bar v)$ 要么报"undefined function"运行时错误（若调用走 native 查找路径），要么返回 `Unit`（若 VM 编译期未报错且 native 表查找返回默认值）。

两种情况均与 $\text{sem}_V = \text{sem}_I$ 矛盾。

**情形 2**：$n \in \text{dom}(R_V) \setminus \text{dom}(R_I)$。对称论证。

**情形 3**：$n \in \text{dom}(R_V) \cap \text{dom}(R_I)$ 但存在 $\bar v$ 使 $R_V(n)(\bar v) \neq R_I(n)(\bar v)$。直接构造 $P_n$ 即得 $\text{sem}_V(P_n, \bar v) \neq \text{sem}_I(P_n, \bar v)$，矛盾。

三种情形均矛盾，故 $\mathcal{N} \cong_{\text{reg}} (R_V, R_I)$。$\square$

**逆定理**：$\mathcal{N} \cong_{\text{reg}} (R_V, R_I) \Rightarrow \text{sem}_V = \text{sem}_I$ 仅在"native 调用是唯一双路径分歧源"时成立。该假设较强，第 12 节将单独讨论其局限。

**源码引用**：
- VM 注册：[tenth/src/main.rs:322-1149](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/main.rs)
- 解释器注册：[tenth/src/runtime/interpreter/natives.rs:37-1271](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/interpreter/natives.rs)

---

### 5.2 定理 N2（历史教训的形式化与实证）

**定理 N2**：在 v0.3.x 早期版本中，$n \in \{\texttt{zeros}, \texttt{ones}, \texttt{rand}, \texttt{randn}\}$ 满足 $n \in \text{dom}(R_I) \setminus \text{dom}(R_V)$，故 VM 路径下程序 `zeros(256,256,256).numel()` 退化为 `Unit.numel()`，违反 N1，导致运行时崩溃或语义偏移。

**实证证据**：

1. **化石注释**：[tenth/src/main.rs:1042-1044](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/main.rs) 明确记载：

   > 历史：这些函数仅在 interpreter 实现，JIT/VM 路径下返回 Unit。
   > 补齐后 zeros(256,256,256).numel() 等才能在默认 tenth run 路径下正常工作。

2. **当前实现已补齐**：[main.rs:1045-1062](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/main.rs) 现含 `vm.add_native("zeros", ...)`、`vm.add_native("ones", ...)`、`vm.add_native("rand", ...)`，与解释器分支 [natives.rs:393-449](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/interpreter/natives.rs) 对齐。补齐后 $n \in \text{dom}(R_V) \cap \text{dom}(R_I)$，N1 不变量恢复。

3. **同期补齐的同类**：`zeros_f32`、`ones_f32`、`rand_f32`、`randn`、`randn_f32` 均在 [main.rs:1016-1080](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/main.rs) 补齐，且 `randn`/`randn_f32` 在 VM 路径使用 Box-Muller 变换，与解释器路径（委托 `Tensor::randn`）的随机数发生器**不同**——这是一个潜在的语义偏移（见 N3 与第 12 节局限 L3）。

**证明**（形式化还原）：

设 $P = \texttt{zeros(256,256,256).numel()}$，记 $P$ 在引擎 $E$ 下的求值为 $\llbracket P \rrbracket_E$。在补齐前的 VM 路径下：

$$\llbracket P \rrbracket_V = \llbracket \texttt{.numel()} \rrbracket_V(\llbracket \texttt{zeros(256,256,256)} \rrbracket_V) = \llbracket \texttt{.numel()} \rrbracket_V(\text{Unit})$$

由于 `.numel()` 在张量方法分派表上不接受 `Unit` 接收者，运行时进入错误处理路径或返回 `Unit`。而解释器路径：

$$\llbracket P \rrbracket_I = \llbracket \texttt{.numel()} \rrbracket_I(\text{Tensor}(\bar 0, [256,256,256])) = 16777216$$

故 $\llbracket P \rrbracket_V \neq \llbracket P \rrbracket_I$，与 $\text{sem}_V = \text{sem}_I$ 矛盾。由 N1 逆否，原协议不一致。$\square$

---

### 5.3 定理 N3（当前注册的完备性检查）

**定理 N3**：截至 v0.3.3，$\mathcal{N} \not\cong_{\text{reg}} (R_V, R_I)$，具体差距如下：

| 缺失方向 | 缺失数量 | 缺失项 |
|---------|---------|--------|
| $n \in \text{dom}(R_I) \setminus \text{dom}(R_V)$ | 17 | `to_string`, `type_name`, `with_step_limit`, `with_timeout_ms`, `is_timeout`, `start_grad`, `f64_bits`, `f64_from_bits`, `sin`, `cos`, `ln`, `pow`, `save_weights`, `load_weights`, `format`, `parse_int`, `parse_float` |
| $n \in \text{dom}(R_V) \setminus \text{dom}(R_I)$ | 3 | `to_f64`, `to_f32`, `print` |

**实证方法**：

通过对 [main.rs:322-1149](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/main.rs) 中 `vm.add_native(...)` 调用与 [natives.rs:40-1221](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/interpreter/natives.rs) 中 `match name { ... }` 分支的逐行枚举，得：

- $|\text{dom}(R_V)| = 70$
- $|\text{dom}(R_I)| = 84$
- $|\text{dom}(R_V) \cap \text{dom}(R_I)| = 67$
- $|\text{dom}(R_V) \setminus \text{dom}(R_I)| = 3$
- $|\text{dom}(R_I) \setminus \text{dom}(R_V)| = 17$

**证明**：由集合差集的对称性 $|\text{dom}(R_V) \triangle \text{dom}(R_I)| = 17 + 3 = 20$。由定义 4.4 条件 1（域一致）对这 20 项失效，故 $\mathcal{N} \not\cong_{\text{reg}} (R_V, R_I)$。$\square$

**关键风险项分析**：

1. **`save_weights` / `load_weights`**：解释器有（[natives.rs:450-574](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/interpreter/natives.rs)），VM 无。VM 路径下任何使用权重持久化的训练脚本都会静默失败——这是 ML 训练流程的关键路径。
2. **`with_step_limit` / `with_timeout_ms` / `is_timeout`**：解释器有（[natives.rs:74-140](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/interpreter/natives.rs)），VM 无。VM 路径下用户写的超时控制代码完全失效，可能导致无限循环。
3. **`sin` / `cos` / `ln` / `pow`**：解释器有（[natives.rs:271-322](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/interpreter/natives.rs)），VM 无。这些是基础数学函数。注意 VM 路径有 `math_pow`/`math_log2` 等带前缀版本，但裸名 `sin/cos/ln/pow` 缺失。
4. **`format` / `parse_int` / `parse_float`**：解释器有（[natives.rs:944-1013](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/interpreter/natives.rs)），VM 无。影响字符串处理程序。
5. **`start_grad`**：解释器有（[natives.rs:147-151](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/interpreter/natives.rs)），VM 无。注意 `start_grad` 与 `new_grad` 在解释器中行为不同——`start_grad` 仅设 `recording = true` 不创建新 Tape，而 `new_grad` 同时创建 Tape。VM 路径下 `start_grad` 会让程序直接报"undefined function"错误，而非退化为 `new_grad`。

---

### 5.4 定理 N4（与 Python/Lua 注册机制对比）

**定理 N4**：Tenth 的双重注册反模式在 Python C-extension 与 Lua C function 中均不存在，因其执行引擎唯一；但 Tenth 的反模式提供了 Python/Lua 所没有的"双执行引擎容错"能力，本质上是工程取舍。

**对比表**：

| 维度 | Python C-ext | Lua C function | Tenth native |
|------|--------------|----------------|--------------|
| 执行引擎数 | 1（CPython VM） | 1（Lua VM） | 2（栈式 VM + tree-walk 解释器） |
| 注册表数 | 1（`PyMethodDef[]`） | 1（`luaL_Reg[]`） | 2（`R_V` + `R_I`） |
| C-API 签名统一 | 是（`PyObject*` 三参签名） | 是（`int (*)(lua_State*)`） | 否（VM 闭包 vs 解释器方法分支） |
| 缺失注册的故障模式 | 链接错误（编译期） | 运行时"no field" | 运行时"undefined"或返回 `Unit` |
| 是否存在双源真相 | 否 | 否 | **是** |
| 容错性 | 单点失败 | 单点失败 | 双引擎可互为回退 |

**论证**：

Python 的 `PyMethodDef` 数组在 C 模块初始化时被解释器读取，注册到模块字典。由于只有一条执行路径，不存在"另一侧"的概念。Lua 同理。Tenth 的双引擎设计要求 native 必须双注册，本质上是**双源真相的成本换取了双引擎的容错收益**——当 VM 编译失败时，解释器仍可执行（见 [main.rs:1278-1285](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/main.rs) 的 fallback 路径）。这是 Tenth 在 ML 训练场景（需要快速迭代且不能因编译失败阻塞实验）下的工程取舍。

**推论 N4.1**：Tenth 的双重注册反模式无法通过"模仿 Python/Lua 单注册表"消除，因双引擎同存是设计目标。正确修补方向是**声明宏自动双重注册**（N5）。$\square$

---

### 5.5 定理 N5（宏自动双重注册方案，未来工作）

**定理 N5（设计性）**：存在一个 Rust 声明宏 `native!`，可将单点函数实现展开为同时注册到 $R_V$ 与 $R_I$ 的双份代码，使得 $\text{dom}(R_V) = \text{dom}(R_I)$ 在编译期被强制保证，且签名一致（定义 4.4 条件 1、2）自动满足。语义一致（条件 3）仍需人工或测试保证。

**展开示意**（不实现，仅描述目标）：

```rust
// 期望的宏调用
native! {
    name: "zeros",
    sig: (Variadic Int) -> Tensor,
    impl: |state, args| {
        let shape = args.iter().map(|a| a.as_int().unwrap_or(1) as usize).collect();
        Tensor::zeros(&shape)
    }
}

// 期望展开为
vm.add_native("zeros".into(), /* 上述 impl 适配为 VM 闭包 */);
// 同时
match name { "zeros" => { /* 上述 impl 适配为解释器分支 */ } }
```

**约束**：由于 `vm.add_native` 接受 `impl Fn(&mut Vm, &[Value]) -> Result<Value>`，而 `call_named_fn` 是 `Interpreter` 的方法（接受 `&mut self`），两路的状态类型不同。宏必须生成两个适配层：

- $R_V$ 适配：从 `&mut Vm` 提取状态（如 `vm.tape`、`vm.recording`）；
- $R_I$ 适配：从 `&mut Interpreter` 提取等价状态（如 `self.tape`、`self.recording`）。

**强制力分析**：

| 不变量 | 宏能否保证 | 失效模式 |
|--------|-----------|---------|
| 域一致（定义 4.4.1） | **能**（编译期展开到两侧） | 若宏调用漏写则双侧都缺，但不会单侧缺 |
| 签名一致（定义 4.4.2） | **能**（宏从单一 `sig` 注解生成两侧类型） | 无 |
| 语义一致（定义 4.4.3） | **否**（实现体在两侧共享，但状态适配层不同） | 适配层 bug 导致 `vm.tape` 与 `self.tape` 行为分歧 |

**为何标注为未来工作**：当前 `register_natives` 与 `call_named_fn` 的代码结构差异较大（前者是命令式调用序列，后者是 match 表达式），宏改造涉及两侧代码生成模板的设计，且需在不破坏自举三路径（[DEPS.md 路径 A/B/C]）的前提下进行。这超出本文范围。$\square$

**源码引用**：
- VM 注册入口：[tenth/src/main.rs:322](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/main.rs)
- 解释器注册入口：[tenth/src/runtime/interpreter/natives.rs:37](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/interpreter/natives.rs)

---

## 6 双重注册的协议形式化

### 6.1 协议三要素

将双重注册视为一个协议 $\Pi = (\mathcal{N}, R_V, R_I)$，要求满足三要素：

**要素 1（域闭包）**：$\text{dom}(R_V) = \text{dom}(R_I) = \mathcal{N}$。

**要素 2（签名同构）**：对每个 $n \in \mathcal{N}$，$R_V(n)$ 与 $R_I(n)$ 接受相同的参数类型序列、返回相同的值类型。

**要素 3（语义等价）**：对每个 $n \in \mathcal{N}$ 与每个合法状态-输入对 $(s, \bar v)$，$R_V(n)(s_V, \bar v) = R_I(n)(s_I, \bar v)$，其中 $s_V, s_I$ 是同一抽象状态 $s$ 在两侧的具体表示。

### 6.2 协议破坏的故障谱系

| 破坏类型 | 故障表现 | 实例（v0.3.3） |
|---------|---------|---------------|
| 域不一致（VM 缺） | "undefined function" 或返回 `Unit` | `save_weights`、`sin`、`format` 等 17 项 |
| 域不一致（解释器缺） | "undefined function" | `to_f64`、`to_f32`、`print` 3 项 |
| 签名不一致 | 类型错误或语义偏移 | 暂未发现（待 T35 系统审计） |
| 语义不一致 | 同输入双路径不同结果 | `randn` 随机源差异（见 6.3） |

### 6.3 语义偏移实例：`randn` 的双路径随机源

VM 路径 [main.rs:1016-1028](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/main.rs)：

```rust
vm.add_native("randn".into(), |_vm, args| {
    // 直接 Box-Muller，使用 rand::thread_rng()
    let mut rng = rand::thread_rng();
    let data: Vec<f64> = (0..rows*cols).map(|_| {
        let u1: f64 = rng.r#gen::<f64>().max(1e-10);
        let u2: f64 = rng.r#gen::<f64>();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }).collect();
    Tensor::from_vec(data, vec![rows, cols])
});
```

解释器路径 [natives.rs:400-406](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/interpreter/natives.rs)：

```rust
"randn" => {
    let t = Tensor::randn(&shape);  // 委托给 Tensor::randn
    return Ok(Some(Value::Tensor(Rc::new(RefCell::new(t)))));
}
```

二者均使用 Box-Muller，但实现路径不同（VM 内联实现 vs 委托 `Tensor::randn`）。若 `Tensor::randn` 的实现与 VM 内联版本在数值稳定性处理（如 `u1.max(1e-10)`）上不一致，则产生语义偏移。这构成定义 4.4 条件 3 的潜在破坏。**审计建议**：统一为 `Tensor::randn` 单点实现，VM 闭包仅做适配调用。

---

## 7 历史教训的实证分析

### 7.1 化石注释的考古学

[tenth/src/main.rs:1042-1044](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/main.rs) 的注释是少有的"反模式化石"——它直接记录了"这些函数仅在 interpreter 实现"的历史状态。这类注释在工程实践中价值极高：

1. **可追溯性**：注释本身证明了反模式曾经存在，不是论文虚构；
2. **修补证据**：注释下方的 `vm.add_native("zeros", ...)` 等代码即是修补 commit 的产物；
3. **可审计性**：未来开发者可通过此注释理解为何该段代码必须与 [natives.rs:436-449](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/interpreter/natives.rs) 对齐。

### 7.2 故障的潜伏期

`zeros/ones/rand/randn` 缺失期间，为何测试套件未捕获？合理推断：

- 单元测试可能默认走解释器路径（更快、更稳定）；
- VM 路径的端到端测试可能未覆盖张量构造；
- 即使有覆盖，"返回 Unit"的故障模式在简单程序中可能不立即触发崩溃（因 Unit 是合法 Value）。

这暴露了**单侧测试覆盖**的盲区：测试通过 ≠ 协议一致。审计建议见 [AUDIT.md] 测试覆盖矩阵。

### 7.3 同期反模式实例

除 `zeros/ones/rand/randn` 外，N3 揭示的 17 项 VM 缺失均为同一反模式的延续。其中最危险的是 `save_weights/load_weights`——训练脚本通常以 `tenth run` 走 VM 路径，权重持久化失败会导致训练成果丢失，且故障静默（解释器路径才生效）。

---

## 8 完备性检查

### 8.1 两侧注册表全量枚举

#### 8.1.1 VM 路径 $R_V$（70 项，源自 [main.rs:322-1149](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/main.rs)）

按功能分类：

- **I/O（5）**：`println`, `print`, `read_file`, `write_bytes`, `read_bytes`
- **容器（2）**：`Vec::new`, `HashMap::new`
- **张量构造（10）**：`tensor`, `zeros`, `ones`, `rand`, `randn`, `zeros_f32`, `ones_f32`, `rand_f32`, `randn_f32`, `tensor_from_vec`
- **数值转换（4）**：`to_float`, `to_f64`, `to_f32`, `abs`, `sqrt`
- **自动微分（7）**：`new_grad`, `param`, `backward`, `grad`, `stop_grad`, `zero_grad`, `cross_entropy`
- **数学（15）**：`math_tan/asin/acos/atan/atan2/sinh/cosh/tanh/log10/log2/exp/pow/floor/ceil/round`
- **时间（6）**：`time_now/now_ms/date/time/datetime/sleep_ms`
- **随机（2）**：`random_int`, `random_float`
- **JSON（3）**：`json_encode/encode_pretty/decode`
- **文件系统（11）**：`write_file`, `path_join/exists/is_file/is_dir`, `mkdir`, `list_dir`, `file_size`, `remove_file`, `copy_file`, `rename_file`
- **CLI（2）**：`cli_args_count`, `cli_arg`
- **编译（2）**：`compile_host`, `compile_program`

注：分类计数 5+2+10+5+7+15+6+2+3+11+2+2 = 70，与枚举一致。

#### 8.1.2 解释器路径 $R_I$（84 项，源自 [natives.rs:40-1221](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/interpreter/natives.rs)）

按功能分类：

- **I/O（4）**：`println`, `read_file`, `write_bytes`, `read_bytes`（无 `print`）
- **元信息（3）**：`to_string`, `type_name`, `format`
- **解析（2）**：`parse_int`, `parse_float`
- **执行控制（3）**：`with_step_limit`, `with_timeout_ms`, `is_timeout`
- **容器（2）**：`Vec::new`, `HashMap::new`
- **张量构造（10）**：`tensor`, `zeros`, `ones`, `rand`, `randn`, `zeros_f32`, `ones_f32`, `rand_f32`, `randn_f32`, `tensor_from_vec`
- **数值转换（3）**：`to_float`, `abs`, `sqrt`（无 `to_f64`、`to_f32`）
- **位运算（2）**：`f64_bits`, `f64_from_bits`
- **基础数学（4）**：`sin`, `cos`, `ln`, `pow`
- **自动微分（8）**：`new_grad`, `start_grad`, `param`, `backward`, `grad`, `stop_grad`, `zero_grad`, `cross_entropy`
- **数学（15）**：`math_tan/...round`（同 VM）
- **时间（6）**：同 VM
- **随机（2）**：同 VM
- **JSON（3）**：同 VM
- **文件系统（11）**：同 VM
- **权重持久化（2）**：`save_weights`, `load_weights`
- **CLI（2）**：同 VM
- **编译（2）**：同 VM

注：4+3+2+3+2+10+3+2+4+8+15+6+2+3+11+2+2+2 = 84，与枚举一致。

### 8.2 对称差集详表

| $n$ | $\in R_V$ | $\in R_I$ | 风险等级 | 影响 |
|-----|-----------|-----------|---------|------|
| `to_string` | ❌ | ✅ | 中 | VM 路径字符串化失效 |
| `type_name` | ❌ | ✅ | 低 | VM 路径类型查询失效 |
| `with_step_limit` | ❌ | ✅ | **高** | VM 路径无步数预算保护，可能死循环 |
| `with_timeout_ms` | ❌ | ✅ | **高** | VM 路径无超时保护 |
| `is_timeout` | ❌ | ✅ | 中 | 与上二者配套使用 |
| `start_grad` | ❌ | ✅ | 中 | VM 路径无此 API（已有 `new_grad` 替代） |
| `f64_bits` | ❌ | ✅ | 低 | VM 路径无位运算 |
| `f64_from_bits` | ❌ | ✅ | 低 | 同上 |
| `sin` | ❌ | ✅ | 中 | VM 路径缺基础数学（有 `math_*` 前缀版本但无裸名） |
| `cos` | ❌ | ✅ | 中 | 同上 |
| `ln` | ❌ | ✅ | 中 | 同上 |
| `pow` | ❌ | ✅ | 中 | 同上 |
| `save_weights` | ❌ | ✅ | **致命** | VM 路径训练脚本权重不落盘 |
| `load_weights` | ❌ | ✅ | **致命** | VM 路径训练脚本权重不加载 |
| `format` | ❌ | ✅ | 中 | VM 路径字符串模板失效 |
| `parse_int` | ❌ | ✅ | 中 | VM 路径整数解析失效 |
| `parse_float` | ❌ | ✅ | 中 | VM 路径浮点解析失效 |
| `to_f64` | ✅ | ❌ | 低 | 解释器路径无此别名（已有 `to_float`） |
| `to_f32` | ✅ | ❌ | 低 | 解释器路径缺 f32 显式转换 |
| `print` | ✅ | ❌ | 低 | 解释器路径无（已有 `println`） |

### 8.3 完备性结论

**当前协议不完备**：$\mathcal{N} \not\cong_{\text{reg}} (R_V, R_I)$，对称差 20 项，其中 **2 项致命**（`save_weights`、`load_weights`）、**2 项高风险**（`with_step_limit`、`with_timeout_ms`）、**1 项语义偏移隐患**（`randn` 双路径实现差异）。

---

## 9 与 Python C-extension / Lua C function 对比

### 9.1 注册表结构的代数差异

- **Python/Lua**：注册表是单射 $R: \mathcal{N} \hookrightarrow \text{CFn}$，$\text{CFn}$ 是统一签名的 C 函数集。一致性条件平凡：$R$ 存在即可。
- **Tenth**：注册表是双射对 $(R_V, R_I)$，一致性条件非平凡：要求 $\text{dom}(R_V) = \text{dom}(R_I)$ 且实现语义等价。

### 9.2 故障检测时机

- **Python**：未注册的 C 函数在模块加载时报 `AttributeError`，编译期/加载期即可发现。
- **Lua**：未注册的 C 函数在调用时报 `attempt to call a nil value`，运行时但故障明确。
- **Tenth**：未注册的 native 在调用时报"undefined function"或静默返回 `Unit`，故障可能潜伏至生产环境。**这是双源真相的最大代价**。

### 9.3 可演化性

- **Python/Lua**：新增 native 仅需修改一处注册表，演化成本低。
- **Tenth**：新增 native 需同步修改两处，演化成本翻倍。但 N5 的宏方案可将成本降回单点。

### 9.4 容错性

- **Python/Lua**：单引擎，无回退能力。VM 崩溃即程序崩溃。
- **Tenth**：双引擎可互为回退（[main.rs:1278-1285](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/main.rs)），但回退前提是 $R_I$ 完备。若 $R_I$ 也不完备（如缺 `to_f64`），则回退路径同样失败。**双源真相的容错收益只在协议完备时兑现**。

---

## 10 工程权衡

### 10.1 当前架构的取舍

Tenth 选择双源真相换取：
1. **双引擎容错**：VM 编译失败可回退解释器；
2. **演进自由度**：VM 路径可独立优化（如 JIT），不受解释器约束；
3. **代码局部性**：VM 闭包与解释器方法分支各自贴近其执行上下文，可读性较好。

代价：
1. **协议维护成本**：每次新增 native 需双写；
2. **故障潜伏期**：单侧缺失无编译期信号；
3. **审计复杂度**：完备性需人工或工具定期审计（如本文 N3）。

### 10.2 修补路径优先级

| 优先级 | 修补项 | 理由 |
|--------|--------|------|
| P0 | 补齐 `save_weights`/`load_weights` 至 VM | 致命，影响训练流程 |
| P0 | 补齐 `with_step_limit`/`with_timeout_ms`/`is_timeout` 至 VM | 高风险，影响程序安全性 |
| P1 | 统一 `randn` 实现为委托 `Tensor::randn` | 消除语义偏移隐患 |
| P1 | 补齐 `sin/cos/ln/pow` 至 VM | 基础数学完备性 |
| P2 | 补齐 `format/parse_int/parse_float/to_string/type_name` 至 VM | 字符串/元信息完备性 |
| P2 | 补齐 `to_f64/to_f32/print` 至解释器 | 别名/便利函数完备性 |
| P3 | 引入 `native!` 宏（N5） | 根治反模式 |

### 10.3 自举路径影响

按 [DEPS.md] 自举三路径：
- **路径 A**（Rust 全栈）：受 N3 影响最大（默认走 VM）；
- **路径 B**（Tenth 前端 + Rust 后端）：bridge.rs 不涉及 native 注册，不受影响；
- **路径 C**（全 WASM 闭环）：WASM runtime 的 native 注册独立，本文未审计（见局限 L4）。

修补 P0/P1 项需保证自举三路径不破坏，建议按 [DEPS.md] 验证命令逐项跑通。

---

## 11 开放问题

### 11.1 签名一致性的自动化验证

定义 4.4 条件 2（签名一致）目前依赖人工核对。能否引入编译期类型注解（如 `native!` 宏的 `sig` 字段）自动检查？

### 11.2 语义一致性的可证伪测试

定义 4.4 条件 3（语义一致）难以穷举证明。能否设计一组差分测试（differential testing），对每个 native 在两侧路径下跑同输入、对比输出？

### 11.3 WASM 路径的第三重注册

路径 C（WASM 闭环）的 native 注册是否构成"三重源真相"？若是，本文的双映射模型需扩展为三映射。这超出本文范围，待 T29（HIR 到 WASM 语义保持）后续工作覆盖。

### 11.4 与 T35（双执行引擎等价性）的联动

N1 给出的是"双引擎等价 ⇒ 注册一致"的单向蕴含。完整的等价性定理（待 T35）需要"注册一致 ⇒ 双引擎等价"的逆向蕴含，但后者依赖"native 调用是唯一双路径分歧源"这一强假设（见局限 L1）。建议 T35 形式化时显式分层：
- **层 1**：注册一致性（本文 N1）；
- **层 2**：字节码与 AST 求值的等价（T34 已部分覆盖）；
- **层 3**：JIT 特化与 VM 的等价（T9 覆盖）；
- **层 4**：全引擎等价（T35 总成）。

### 11.5 `vm_run` 遗留函数的隐患

[main.rs:1180-1311](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/main.rs) 存在 `#[allow(dead_code)] fn vm_run`，内部独立注册了 5 个 native（`println, read_file, Vec::new, compile_host, compile_program`），构成**第三处潜在注册点**。虽当前为 dead code，但若未来被误启用，将引入"三源真相"。建议清理或显式标记 `#[deprecated]`。

---

## 12 局限（独立章节）

本节诚实披露本文形式化方法的边界与潜在漏洞。

### L1：N1 逆定理的强假设

N1 给出 $\text{sem}_V = \text{sem}_I \Rightarrow \mathcal{N} \cong_{\text{reg}} (R_V, R_I)$。逆定理 $\mathcal{N} \cong_{\text{reg}} (R_V, R_I) \Rightarrow \text{sem}_V = \text{sem}_I$ **未证明**，因其依赖"native 调用是唯一双路径分歧源"——这显然不成立（字节码生成与 AST 求值的语义差异、JIT 特化、闭包捕获方式等都是分歧源）。本文不声称逆定理成立，T35 需综合多层等价性。

### L2：完备性审计的时点局限

N3 的统计基于 v0.3.3 源码快照（2026-07-02）。若后续 commit 补齐了部分项，本文结论需更新。建议在 [AUDIT.md] 维护"协议一致性矩阵"的活文档。

### L3：`randn` 语义偏移未实测

第 6.3 节指出 `randn` 在 VM 与解释器路径的实现差异，但未做差分测试。本文声称"潜在语义偏移"是基于代码阅读的推断，非实测证据。修补前应先实测确认是否真有偏移。

### L4：WASM 路径未审计

N3 仅审计 VM 与解释器两路径。WASM 路径（`compile/wasm.rs` + wasmi 执行）的 native 注册独立，本文未覆盖。若 WASM 路径也有自己的 native 表，则实际是"三源真相"，N1 的双映射模型需扩展。

### L5：语义一致性的不可判定性

定义 4.4 条件 3 要求"对所有合法输入 $R_V(n)(\bar v) = R_I(n)(\bar v)$"。这等价于程序等价性，一般不可判定（Rice 定理）。故 N1 的"语义一致"条件在实践中只能通过测试部分验证，无法形式化完备证明。本文接受此局限，将语义一致性留给差分测试而非形式化证明。

### L6：宏方案（N5）的展开可行性未验证

N5 仅给出宏的设计目标，未实现原型。宏展开为双份代码的可行性依赖 Rust 声明宏的表达力，可能需要过程宏（proc-macro）。这超出数理部职责，建议编译器部评估。

### L7：循环论证风险

N1 的证明构造程序 $P_n = \texttt{let x = } n\texttt{(...); return x;}$，假设该程序在 VM 路径下"会调用 native $n$"。但若 VM 编译期将 `n(...)` 编译为内联字节码（如 `Op::MakeTensor` 对 `tensor[[...]]` 的特化，见 [main.rs:350-359](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/main.rs) 注释），则调用不走 native 表，N1 的反证不成立。本文假设"native 调用走运行时分派"，这在大多数 native 上成立，但对有编译期特化的 native（如 `tensor`）不成立。**缓解**：N1 应排除有编译期特化的 native，或在求值模型中显式区分"编译期特化"与"运行时分派"两类调用。这是本文形式化最需要补强之处。

### L8：分类计数的机械性

8.1.1、8.1.2 的分类计数是机械枚举，可能因分类口径不同而重复或遗漏（如 `tensor_from_vec` 既是张量构造又是类型转换）。本文的分类仅用于完备性核对，不声称分类本身互斥完备（这是 T7 shape 变换分类的议题）。

---

## 13 结论

本文对 Tenth 的双重 native 注册结构进行了形式化建模，给出五个主定理：

1. **N1**：双重注册不变量——双引擎等价蕴含注册一致；
2. **N2**：历史教训实证——`zeros/ones/rand/randn` 曾违反 N1，已在 [main.rs:1042-1080](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/main.rs) 补齐；
3. **N3**：当前完备性审计——VM 路径缺 17 项（含 2 项致命：`save_weights/load_weights`）、解释器路径缺 3 项，协议不完备；
4. **N4**：与 Python/Lua 对比——Tenth 的反模式是双引擎设计的代价，无法通过模仿单注册表消除；
5. **N5**：宏自动双重注册方案——标注为未来工作，可强制保证域一致与签名一致，语义一致仍需测试。

**核心结论**：Tenth 的双重 native 注册是反模式，但有其工程合理性。当前协议不完备，存在 2 项致命缺失，建议按 P0/P1 优先级修补。长期方向是引入 `native!` 宏（N5）根治。

**对实施的指导**：
- 短期：按第 10.2 节优先级补齐缺失项；
- 中期：在 [AUDIT.md] 维护协议一致性矩阵，每次新增 native 同步更新；
- 长期：实施 N5 宏方案，从结构上消除双源真相。

---

## 14 参考文献

1. Tenth 项目. *工作规范 v1.1*. `d:\史蒂夫\Desktop\AI开发新语言：头脑风暴与评估\.trae\rules\工作规范.md`
2. Tenth 项目. *DEPS.md*. `d:\史蒂夫\Desktop\AI开发新语言：头脑风暴与评估\DEPS.md`
3. Tenth 项目. *CODE_WIKI.md*. `d:\史蒂夫\Desktop\AI开发新语言：头脑风暴与评估\CODE_WIKI.md`
4. Tenth 项目. *MEMO.md*. `d:\史蒂夫\Desktop\AI开发新语言：头脑风暴与评估\MEMO.md`
5. Tenth 项目. *AUDIT.md*. `d:\史蒂夫\Desktop\AI开发新语言：头脑风暴与评估\AUDIT.md`
6. Tenth 项目. *能力梳理/能力全梳理.md*. `d:\史蒂夫\Desktop\AI开发新语言：头脑风暴与评估\能力梳理\能力全梳理.md`
7. Tenth 项目. *docs/语言参考手册.md*. `d:\史蒂夫\Desktop\AI开发新语言：头脑风暴与评估\docs\语言参考手册.md`
8. T34 论文. *栈式 VM 操作语义形式化*. `d:\史蒂夫\Desktop\AI开发新语言：头脑风暴与评估\docs\论文\T34-栈式VM操作语义形式化.md`
9. T9 论文. *JIT 特化语义保持证明*. `d:\史蒂夫\Desktop\AI开发新语言：头脑风暴与评估\docs\论文\T9-JIT特化语义保持证明.md`
10. T12 论文. *双侧编译器语义等价性*. `d:\史蒂夫\Desktop\AI开发新语言：头脑风暴与评估\docs\论文\T12-双侧编译器语义等价性.md`
11. T32 论文. *hostcall trampoline FFI 安全性*. `d:\史蒂夫\Desktop\AI开发新语言：头脑风暴与评估\docs\论文\T32-hostcall-trampoline-FFI安全性.md`
12. CPython. *methodobject.h*. https://github.com/python/cpython/blob/main/Include/methodobject.h
13. Lua. *lauxlib.c*. https://www.lua.org/source/5.4/lauxlib.c.html
14. Rice, H. G. (1953). *Classes of Recursively Enumerable Sets and Their Decision Problems*. Transactions of the American Mathematical Society, 74, 358-366.（用于 L5 语义等价性不可判定性引用）

---

## 附录 A：定理索引

| 定理 | 简称 | 源码锚点 |
|------|------|---------|
| N1 | 双重注册不变量 | [main.rs:322-1149](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/main.rs) + [natives.rs:37-1271](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/interpreter/natives.rs) |
| N2 | 历史教训实证 | [main.rs:1042-1080](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/main.rs) + [natives.rs:393-449](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/interpreter/natives.rs) |
| N3 | 当前完备性检查 | 第 8 节全量枚举 |
| N4 | 与 Python/Lua 对比 | 第 9 节 |
| N5 | 宏自动双重注册（未来工作） | 设计性，无源码锚点 |

## 附录 B：与现有文档的对应

| 本文定理 | 同步动作 |
|---------|---------|
| N3 完备性差距 | 建议在 `AUDIT.md` 增加"协议一致性矩阵"小节 |
| N5 宏方案 | 建议在 `MEMO.md` 记录"未来工作：native! 宏" |
| L4 WASM 路径 | 建议在 `CODE_WIKI.md` WASM 模块小节标注"native 注册待审计" |

## 附录 C：实施建议清单

按优先级排序（与第 10.2 节对齐）：

1. **P0-a**：在 [main.rs::register_natives](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/main.rs) 末尾添加 `save_weights`、`load_weights` 的 VM 闭包，参考 [natives.rs:450-574](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/interpreter/natives.rs)。
2. **P0-b**：添加 `with_step_limit`、`with_timeout_ms`、`is_timeout` 的 VM 闭包。注意 VM 路径需在 `Vm` 结构体上增加 `step_budget`、`deadline_ms` 字段（若不存在）。
3. **P1-a**：将 `randn`/`randn_f32` 的 VM 闭包改为委托 `Tensor::randn`/`Tensor::randn_f32`，与解释器一致。
4. **P1-b**：补齐 `sin/cos/ln/pow` 的 VM 闭包。
5. **P2-a**：补齐 `format/parse_int/parse_float/to_string/type_name` 的 VM 闭包。
6. **P2-b**：补齐 `to_f64/to_f32/print` 的解释器分支。
7. **P3**：设计 `native!` 宏原型（编译器部评估）。

每步完成后按 [工作规范.md] 第五章验证闭环：
- `cargo test --manifest-path tenth/Cargo.toml`（全测试通过）
- `cargo run --release --manifest-path tenth/Cargo.toml -- run tenthc/main.th`（自举路径 A 未破坏）

---

*本文为数理部产出，遵循"严谨性、完备性边界、局限诚实"三原则。所有定理附源码引用，所有局限独立披露。实施建议附于附录 C，留待编译器部/运行时部落地。*
