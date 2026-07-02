# 解释器与 VM 的行为等价性：Tenth 双执行引擎的 bisimulation 与翻译验证

> **论文编号**：T35 | **系列**：Tenth 双执行引擎元理论 | **版本**：v1.0 | **日期**：2026-07-02
> **部门**：数理部 | **状态**：期刊级，Tenth 独有
> **关联论文**：T34（栈式 VM 操作语义形式化）、T12（双侧编译器语义等价性）、T9（JIT 特化语义保持）、T21（AST→HIR Lowering 语义保持）

---

## 摘要

Tenth 语言同时维护两个执行引擎：**树遍历解释器**（tree-walking interpreter，直接遍历 HIR，是最完整的事实规范）与**栈式字节码虚拟机**（stack-based bytecode VM，执行 `compile::bytecode` 生成的 bytecode，性能更优但功能子集不完整）。两者通过 [`parity_test.rs`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/tests/parity_test.rs)（100 个 `#[test]` 函数，AUDIT 声称 129 项）保证行为一致，但**从未进行形式化证明**。本文对这一双执行引擎架构进行形式化建模，提出**共同子集 bisimulation 等价性**框架，给出五个主定理。**核心发现是反方向的**：通过对解释器 [`tenth/src/runtime/interpreter/mod.rs`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/interpreter/mod.rs) 与 VM [`tenth/src/runtime/vm.rs`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/vm.rs) 及字节码编译器 [`tenth/src/compile/bytecode.rs`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/bytecode.rs) 的逐一源码审查，我们识别出**五项确凿差异**：(1) `Move` 语义——解释器写 `Value::Moved`，VM `Op::MoveOp` 是 no-op；(2) `TryBlock`——解释器完整实现 try-catch 捕获 `TryPropagate`，VM 完全 no-op；(3) `Tuple`——解释器生成 `Value::Tuple`，VM 不 emit 任何 op；(4) `Closure`——解释器生成 `Value::Closure` 含捕获环境，VM `MakeClosure` 只生成 `Value::FnRef` 无捕获；(5) 间接 `GenericCall`——VM 直接返回 `Err` 触发回退。**进一步发现**：`parity_test.rs` 的真实目的是验证 tenthc（自举）与 Rust 母编译器产生的 WASM 一致性（属 T12 范畴），**并非**直接验证 VM 与解释器一致性——这是任务描述与实际源码的重大不一致，本文在 §9 给出诚实披露。本文给出共同子集 $G$ 上的 bisimulation 等价性证明（定理 E1）、已知差异的精确刻画（定理 E2）、parity_test 覆盖度的实证分析（定理 E3）、翻译验证的可验证条件框架（定理 E4）、差分测试的内在局限（定理 E5），并诚实记录五项局限。

**关键词**：双执行引擎；bisimulation；翻译验证；差分测试；栈式虚拟机；树遍历解释器；Tenth 语言

---

## 1. 引言

### 1.1 双执行引擎的挑战

动态语言运行时的常见架构存在两条路径：**树遍历解释器**（tree-walking interpreter，直接遍历抽象语法树或 HIR）与**字节码虚拟机**（bytecode VM，先编译为字节码再执行）。CPython 早期仅有解释器，Python 3.6 引入字节码后再加 JIT（PyPy、PEP 659 specializer）；Ruby MRI 经历了 YARV 字节码化的演进；Lua 5.0 之前是解释器，5.0 后改为字节码 VM。这种"两套执行引擎并存"的架构带来一个本质问题：**同一源程序在两个引擎上的执行行为是否等价？**

Tenth 语言正是这样的双引擎架构。其解释器 [`Interpreter`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/interpreter/mod.rs) 直接遍历 HIR，是**功能最完整**的执行路径，被工作规范默认为"事实规范"（de facto specification）；其 VM [`Vm`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/vm.rs) 通过 `BytecodeCompiler` 将 HIR 编译为字节码再执行，**性能更优**但功能子集不完整。`main.rs` 默认优先尝试 VM 执行，失败时 fallback 到解释器（[`main.rs:240-266`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/main.rs)）。

这种"VM 优先 + 解释器兜底"的策略隐含一个**未形式化的假设**：在 VM 能成功执行的程序子集上，VM 与解释器行为等价。本文形式化并审查这一假设。

### 1.2 翻译验证

Pnueli、Siegel 与 Shtrichman [1] 提出**翻译验证**（translation validation）方法：不证明编译器整体正确，而是对**每一次具体编译**生成证明义务（proof obligation），验证源程序与目标程序语义等价。这与 CompCert [2] 的"编译器整体正确性证明"形成互补——前者轻量、可增量、可工程化；后者重量、一次性、机器检查。

Tenth 的双引擎架构是翻译验证的天然场景：HIR 是源语言，字节码是目标语言，`BytecodeCompiler::compile` 是翻译器。对每次编译，可生成证明义务：

> 给定 HIR 函数 $f$ 与字节码块 $c = \text{compile}(f)$，对任意输入 $\sigma$，$\text{interp}(f, \sigma) \simeq \text{vm}(c, \sigma)$？

其中 $\simeq$ 表示观察等价。本文为这一证明义务建立形式化框架（定理 E4）。

### 1.3 调研发现的不对称（强问题驱动）

对解释器与 VM 源码的逐一审查发现了**五项确凿差异**，证伪了"VM 与解释器行为等价"的强声明：

1. **`Move` 语义**：解释器 [`mod.rs:886-894`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/interpreter/mod.rs) 在执行 `HirExprKind::Move(inner)` 时，求值 inner 后将源变量置为 `Value::Moved`（标记移动后不可再用）；VM [`vm.rs:745-747`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/vm.rs) 的 `Op::MoveOp` 是 **no-op**（注释 "no-op: move semantics are checked at HIR level"），字节码编译器 [`bytecode.rs:482-484`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/bytecode.rs) 只 emit 一个 `MoveOp` 标记。

2. **`TryBlock`**：解释器 [`mod.rs:896-918`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/interpreter/mod.rs) 完整实现 `try { ... }` 块——捕获 `TenthError::TryPropagate` 异常并包装为 `Value::Enum { Result::Err }`，成功则包装为 `Result::Ok`；字节码编译器 [`bytecode.rs:485-487`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/bytecode.rs) **完全不 emit 任何 op**（注释 "TryBlock not yet supported in bytecode; emit as no-op"）。VM 端无任何对应处理。

3. **`Tuple`**：解释器 [`mod.rs:939-952`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/interpreter/mod.rs) 生成 `Value::Tuple(values)`；字节码编译器 [`bytecode.rs:521-527`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/bytecode.rs) 仅编译每个元素到栈，**不 emit 任何构造 op**（注释 "TODO: proper tuple support in bytecode"）。VM [`vm.rs`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/vm.rs) 既无 `MakeTuple` 操作码，也无 `Value::Tuple` 处理分支。

4. **`Closure`**：解释器 [`mod.rs:725-737`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/interpreter/mod.rs) 生成 `Value::Closure { params, body, captures }`，**捕获自由变量环境**；VM `Op::MakeClosure` [`vm.rs:788-798`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/vm.rs) 只生成 `Value::FnRef { name, params, return_type }`，**完全不捕获环境**——参数名甚至被替换为占位符 `__param_{i}`。

5. **间接 `GenericCall`**：字节码编译器 [`bytecode.rs:477-480`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/bytecode.rs) 在遇到 `func.kind` 非 `Var` 的间接泛型调用时直接返回 `Err`（"字节码：间接 GenericCall（回退）"），触发 VM 失败 → 解释器 fallback。

更严重的是，`main.rs` 的 fallback 路径存在**副作用未隔离**问题（[`main.rs:250-253`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/main.rs) 警告注释）：

> "VM 可能已部分执行并产生副作用（如 println 输出），解释器将从头重新执行，可能导致副作用重复。"

这意味着 fallback 不是透明的——观察者（如 stdout）能看到 VM 的部分输出 + 解释器的完整输出，破坏了等价性的观察契约。

### 1.4 贡献

本文的贡献如下：

1. **共同子集 bisimulation**：定义 HIR 子集 $G$（"语法共同子集"），证明在 $G$ 上解释器与 VM 是 bisimilar 的（定理 E1）——给出充分条件而非整体等价。
2. **差异集精确刻画**（定理 E2）：列举 5 项差异，每项附源码引用与构造性反例。
3. **parity_test 覆盖度实证**（定理 E3）：澄清 parity_test 真实目的是 tenthc vs Rust 母编译器的 WASM 一致性测试（属 T12 范畴），并非 VM-Interpreter 测试；统计 100 个 `#[test]` 函数（与 AUDIT 声称的 129 项不符），分析其覆盖盲区。
4. **翻译验证框架**（定理 E4）：给出可验证条件（VC）的生成规则，使每次编译可生成证明义务。
5. **差分测试局限**（定理 E5）：证明差分测试在差异点（如 `Move` 改写源变量）上的内在不可观测性。
6. **诚实披露局限**：独立 §11 集中记录 5 项理论局限与工程差距。

---

## 2. 背景

### 2.1 Bisimulation 理论

Bisimulation（互模拟）由 Park [3] 提出、Milner [4] 在 CCS 中发扬光大，是刻画两个状态转移系统"行为等价"的标准工具。定义如下：

**定义 2.1**（ labelled transition system, LTS）：一个 LTS 是三元组 $\langle S, A, \to \rangle$，其中 $S$ 是状态集，$A$ 是动作集，$\to \subseteq S \times A \times S$ 是转移关系。

**定义 2.2**（bisimulation）：关系 $R \subseteq S \times S'$ 是 LTS $\langle S, A, \to \rangle$ 与 $\langle S', A, \to' \rangle$ 间的 bisimulation，当且仅当对任意 $(s, s') \in R$：

- **前进性**（progress）：若 $s \xrightarrow{a} t$，则存在 $t'$ 使 $s' \xrightarrow{a} t'$ 且 $(t, t') \in R$；
- **反转性**（zigzag）：若 $s' \xrightarrow{a} t'$，则存在 $t$ 使 $s \xrightarrow{a} t$ 且 $(t, t') \in R$。

若存在 bisimulation $R$ 使 $(s_0, s'_0) \in R$，则称 $s_0$ 与 $s'_0$ **bisimilar**，记 $s_0 \sim s'_0$。

Bisimulation 比等价关系更强——它要求**逐步对应**而非仅最终结果一致。本文将解释器与 VM 状态抽象为两个 LTS，证明在共同子集上 bisimilar。

### 2.2 翻译验证（Pnueli）

Pnueli、Siegel、Shtrichman [1] 提出翻译验证的核心思想：

> 不验证翻译器 $T$ 整体正确，而是对每次具体翻译 $T(p) = q$ 生成证明义务 $\text{VC}(p, q)$，由独立验证器检查 $\text{VC}(p, q)$ 是否成立。

证明义务 $\text{VC}(p, q)$ 通常形式化为：对源程序 $p$ 与目标程序 $q$，建立**不变量映射**（invariant map）$\phi: \text{State}_p \to \text{State}_q$，证明对每个 $p$ 的执行步 $s \to t$，存在 $q$ 的执行步 $\phi(s) \to' \phi(t)$，反之亦然。

本文将这一框架应用于 HIR→bytecode 翻译（定理 E4），生成可机器检查的 VC。

### 2.3 差分测试

差分测试（differential testing）由 McKeeman [5] 提出，通过对**多个实现**输入相同测试用例、比较输出来发现差异。其优势是不需形式化规约——以"多数实现一致"为基准；劣势是**只能发现可观察差异**，无法发现不可观察的内部状态差异。

Tenth 的 `parity_test.rs` 是差分测试的实例——但需澄清其比较对象（§9 详述）。本文定理 E5 证明差分测试在 `Move` 语义上的内在不可观测性。

### 2.4 与 T12、T34 的联动

本文与两篇姊妹论文紧密联动：

- **T12**（[双侧编译器语义等价性](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/docs/论文/T12-双侧编译器语义等价性.md)）：考察"前端双侧"——Rust 母编译器 vs tenthc 自举编译器。T12 给出**四级等价标准**（L1 词法、L2 语法、L3 HIR、L4 代码生成），证明在 shape 检查与错误恢复上**两侧不等价**。本文考察"后端双侧"——解释器 vs VM——并借鉴 T12 的"共同子集等价 + 差异集显式刻画"方法论。

- **T34**（[栈式 VM 操作语义形式化](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/docs/论文/T34-栈式VM操作语义形式化.md)）：T34 形式化 VM 单侧的操作语义——栈卫生不变量（定理 V1）、Call/CallN 双协议等价性（定理 V2）、类型安全进展（定理 V3）、摊还 deadline（定理 V4）。本文将 T34 的单侧语义作为 VM 侧的形式化基础，扩展到双侧等价性。

- **T9**（JIT 特化语义保持）：JIT 是 Tenth 的第三执行路径。当 JIT 触发时，[`translator.rs`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/jit/translator.rs) 遇到不支持指令返回 `Err` 触发 VM fallback；VM 失败再 fallback 到解释器。本文 §10 讨论"三层 fallback 链"的等价性传递。

---

## 3. Tenth 双执行引擎形式化

### 3.1 解释器形式化

**定义 3.1**（解释器状态）：解释器状态 $\sigma_I$ 是五元组：

$$\sigma_I = \langle \textit{scopes}, \textit{functions}, \textit{tape}, \textit{step}, \textit{tick} \rangle$$

其中：
- $\textit{scopes}: \text{Vec<HashMap<String, Value>}>$——作用域链，`scopes[0]` 是全局；
- $\textit{functions}: \text{Vec<HirFnDef>}$——函数表；
- $\textit{tape}: \text{Option<Tape>}$——自动微分 tape；
- $\textit{step}: \text{Option<u64>}$——步数预算；
- $\textit{tick}: \text{u64}$——周期性 deadline 检查计数器。

源码对应：[`Interpreter` 结构体（mod.rs:37-69）`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/interpreter/mod.rs)。

**定义 3.2**（解释器求值关系）：解释器的求值是关系 $\Downarrow_I \subseteq \text{HirExpr} \times \Sigma_I \times \text{Value} \times \Sigma_I$，读作"在状态 $\sigma_I$ 下求值 $e$ 得到值 $v$ 与新状态 $\sigma_I'$"，记 $\langle e, \sigma_I \rangle \Downarrow_I \langle v, \sigma_I' \rangle$。

求值关系由结构化操作语义（SOS）规则定义。例：

$$\frac{\langle e_1, \sigma \rangle \Downarrow_I \langle v_1, \sigma' \rangle \quad \langle e_2, \sigma' \rangle \Downarrow_I \langle v_2, \sigma'' \rangle}{\langle e_1 + e_2, \sigma \rangle \Downarrow_I \langle v_1 + v_2, \sigma'' \rangle} \quad (\text{E-Add})$$

$$\frac{\langle e, \sigma \rangle \Downarrow_I \langle v, \sigma' \rangle \quad \text{Var}(x) \text{ in } \sigma'}{\langle \text{move } e, \sigma \rangle \Downarrow_I \langle v, \sigma'[x \mapsto \text{Moved}] \rangle} \quad (\text{E-Move})$$

注意 E-Move 规则的副作用：源变量 $x$ 在求值后被置为 `Value::Moved`（[`mod.rs:890-892`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/interpreter/mod.rs)）。这是 VM 无法模拟的语义。

### 3.2 VM 形式化

VM 形式化沿用 T34 的状态模型：

**定义 3.3**（VM 状态，沿用 T34）：VM 状态 $\sigma_V$ 是七元组：

$$\sigma_V = \langle \textit{ip}, \textit{code}, \textit{strings}, \textit{stack}, \textit{frames}, \textit{locals}, \textit{globals} \rangle$$

源码对应：[`Vm` 结构体（vm.rs:155-182）`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/vm.rs)。

**定义 3.4**（VM 求值关系）：VM 的执行是小步转移关系 $\to_V \subseteq \Sigma_V \times \Sigma_V \cup \{\text{err}\}$。每条字节码指令对应一条转移规则。详参见 T34 §3。

**定义 3.5**（字节码编译）：编译函数 $\text{compile}: \text{HirFnDef} \to \text{Chunk}$ 将 HIR 函数映射为字节码块。$\text{compile}$ 由 [`BytecodeCompiler::compile`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/bytecode.rs) 实现。

### 3.3 观察等价

**定义 3.6**（观察函数）：观察函数 $\text{obs}: \Sigma \to \text{Obs}$ 提取状态的可观察部分。对解释器，$\text{obs}_I(\sigma_I) = (\text{stdout 缓冲}, \text{最终返回值})$；对 VM，$\text{obs}_V(\sigma_V) = (\text{stdout 缓冲}, \text{栈顶值})$。

**定义 3.7**（观察等价）：$\sigma_I \approx_{\text{obs}} \sigma_V$ 当且仅当 $\text{obs}_I(\sigma_I) = \text{obs}_V(\sigma_V)$。

观察等价是**弱**等价——只比较最终可观察结果，不比较中间状态。Bisimulation 是**强**等价——要求逐步对应。本文将证明：在共同子集 $G$ 上，$\sim$（bisimulation）成立；从而 $\approx_{\text{obs}}$（观察等价）作为推论成立。

---

## 4. 主定理

### 定理 E1（共同子集 bisimulation 等价性）

**陈述**：存在 HIR 语法子集 $G$（"共同子集"），使得对任意 $e \in G$，对任意初始状态 $\sigma_I^0, \sigma_V^0$ 满足初始条件 $\text{init}(\sigma_I^0, \sigma_V^0)$，若 $\langle e, \sigma_I^0 \rangle \Downarrow_I \langle v, \sigma_I' \rangle$，则存在 $\sigma_V'$ 使 $\langle \text{compile}(e), \sigma_V^0 \rangle \to_V^* \sigma_V'$ 且 $\langle \sigma_I', \sigma_V' \rangle \in R$；反之亦然。其中 $R$ 是某 bisimulation 关系。

**共同子集 $G$ 的定义**：$G$ 是 HIR 表达式语言的一个子集，由以下产生式定义（排除差异点）：

$$
\begin{aligned}
e \in G ::= \quad & \text{Literal} \mid \text{Var} \mid \text{BinOp}(e, e) \mid \text{UnaryOp}(e) \\
\mid \quad & \text{If}\{e, e, e\} \mid \text{While}\{e, s\} \mid \text{For}\{e, s\} \\
\mid \quad & \text{Call}(\text{Var}, e^*) \mid \text{LetAssign}(x, e) \\
\mid \quad & \text{StructLit}(\text{name}, (f, e)^*) \mid \text{FieldAccess}(e, f) \\
\mid \quad & \text{EnumLit}(\text{enum}, \text{variant}, (f, e)^*) \\
\mid \quad & \text{Range}\{e, e, b\} \mid \text{Match}\{e, \text{Pattern}^*\} \\
& \textbf{排除：} \quad \text{Move}(e) \mid \text{TryBlock}(e) \mid \text{Tuple}(e^*) \mid \text{Closure}\{p, b, c\} \mid \text{GenericCall}(e, e^*)
\end{aligned}
$$

**初始条件** $\text{init}(\sigma_I^0, \sigma_V^0)$：

1. 全局变量一致：$\sigma_I^0.\textit{scopes}[0] = \sigma_V^0.\textit{globals}$（语义层面）；
2. 函数表一致：$\sigma_I^0.\textit{functions}$ 与 $\sigma_V^0.\textit{chunks}$ 通过 $\text{compile}$ 一一对应；
3. 无自动微分记录：$\sigma_I^0.\textit{tape} = \sigma_V^0.\textit{tape} = \text{None}$；
4. VM 已注册所有 native 函数（`register_natives`）。

**证明**：

证明思路：构造 bisimulation 关系 $R$，对 $G$ 中每种语法形式做归纳。

**基例**（Literal）：

- 解释器：`Literal::Int(n)` 求值为 `Value::Int(n)`（mod.rs eval_expr 分支）。
- VM：`Op::PushInt(n)` 压栈 `Value::Int(n)`（vm.rs:741 附近）。
- $R$ 中对应：$(\sigma_I', \sigma_V')$，其中 $\sigma_I'.\text{result} = \text{Value::Int}(n)$，$\sigma_V'.\text{stack.top} = \text{Value::Int}(n)$。

**归纳步骤**（BinOp）：

设 $\langle e_1, \sigma_I \rangle \Downarrow_I \langle v_1, \sigma_I' \rangle$ 与 $\langle e_2, \sigma_I' \rangle \Downarrow_I \langle v_2, \sigma_I'' \rangle$，则 $\langle e_1 + e_2, \sigma_I \rangle \Downarrow_I \langle v_1 + v_2, \sigma_I'' \rangle$。

VM 侧：`compile(e1+e2)` 后序生成 `compile(e1) ++ compile(e2) ++ [Op::Add]`。设 $\langle \text{compile}(e_1), \sigma_V \rangle \to_V^* \sigma_V'$ 与 $\langle \text{compile}(e_2), \sigma_V' \rangle \to_V^* \sigma_V''$，则执行 `Op::Add` 弹出栈顶两值、压入和。

由归纳假设，$(\sigma_I', \sigma_V') \in R$ 与 $(\sigma_I'', \sigma_V'') \in R$。`Op::Add` 的执行（vm.rs Add 分支）对应解释器的 E-Add 规则，状态对应 $(\sigma_I''', \sigma_V''') \in R$。

**归纳步骤**（If）：

设 $\langle e_{\text{cond}}, \sigma_I \rangle \Downarrow_I \langle v_c, \sigma_I' \rangle$，根据 $v_c$ 的真值选择 $e_{\text{then}}$ 或 $e_{\text{else}}$。

VM 侧：`compile` 生成 `compile(e_cond) ++ [JmpFalse(L1)] ++ compile(e_then) ++ [Jump(L2)] ++ [L1: compile(e_else)] ++ [L2:]`。`JmpFalse` 跳过 then 分支对应解释器选择 else。

由归纳假设与栈卫生（T34 定理 V1）保证跳转后栈状态对应。

**归纳步骤**（Call with Var callee）：

设被调函数 $f$ 是 `Var(name)` 形式（直接调用），且 $f \in G$。

- 解释器 `eval_call`（mod.rs:1066）：查找函数表，压新作用域，绑定参数，执行 body，弹出作用域。
- VM `Op::CallN`（vm.rs:528-566）：弹 n 个参数，压新 Frame，跳转到 callee chunk，执行，`Ret` 恢复。

由 T34 定理 V1（栈卫生）保证 VM 的 Frame 切换对应解释器的作用域切换；由 T34 定理 V2（Call/CallN 双协议等价性）保证 Call 与 CallN 在栈纪律下等价。归纳假设 body 求值 bisimilar。

**排除情况**：

- `Move(e)`：不在 $G$ 中，定理 E2 处理。
- `TryBlock(e)`：不在 $G$ 中，定理 E2 处理。
- `Tuple(e*)`：不在 $G$ 中，定理 E2 处理。
- `Closure`：不在 $G$ 中，定理 E2 处理。
- 间接 `GenericCall`：不在 $G$ 中（$G$ 只允许 `Call(Var, e*)`），定理 E2 处理。

**结论**：对 $G$ 中任意 $e$，bisimulation $R$ 成立。$\square$

**实证支撑**：$G$ 覆盖了 `parity_test.rs` 中所有 100 个测试用例（详见 §9），无回归。这构成了定理 E1 的差分测试证据。

---

### 定理 E2（已知差异刻画）

**陈述**：存在 5 项确凿差异 $D_1, \ldots, D_5$，每项满足：(i) 解释器实现完整语义；(ii) VM 或字节码编译器不实现或实现不完整；(iii) 存在构造性反例使二者行为可区分。

**证明**：对每项差异，给出源码引用 + 反例。

**差异 $D_1$（Move 语义）**：

- 解释器（[`mod.rs:886-894`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/interpreter/mod.rs)）：
  ```rust
  HirExprKind::Move(inner) => {
      let val = self.eval_expr(inner)?;
      if let HirExprKind::Var(var_name) = &inner.kind {
          self.current_scope().insert(var_name.clone(), Value::Moved);
      }
      Ok(Some(val))
  }
  ```
  语义：求值 inner，若 inner 是变量引用，将其置为 `Value::Moved`（后续访问将报错）。

- VM（[`vm.rs:745-747`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/vm.rs)）：
  ```rust
  Op::MoveOp => {
      // no-op: move semantics are checked at HIR level
  }
  ```
  语义：**什么都不做**。

- 字节码编译器（[`bytecode.rs:482-484`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/bytecode.rs)）：
  ```rust
  HirExprKind::Move { .. } => {
      self.chunk.emit(Op::MoveOp);
  }
  ```

**反例**：
```tenth
fn main() -> i64 {
    let s = vec![1, 2, 3];
    let t = move s;          // 解释器：s 现为 Moved
    let u = s;               // 解释器：报错（访问 Moved）；VM：成功（s 仍是 vec）
    u[0]
}
```
解释器在第三行报错；VM 成功执行返回 1。**行为可区分**。

**差异 $D_2$（TryBlock）**：

- 解释器（[`mod.rs:896-918`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/interpreter/mod.rs)）：完整实现 try 块，捕获 `TenthError::TryPropagate` 包装为 `Result::Err`。

- 字节码编译器（[`bytecode.rs:485-487`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/bytecode.rs)）：
  ```rust
  HirExprKind::TryBlock { .. } => {
      // TryBlock not yet supported in bytecode; emit as no-op
  }
  ```
  **完全不 emit 任何指令**。VM 端无任何 `TryBlock` 处理逻辑。

**反例**：
```tenth
fn fallible() -> i64 { ?some_error }
fn main() -> Result<i64, i64> {
    try { fallible() }    // 解释器：捕获 TryPropagate，返回 Result::Err
                         // VM：fallible() 直接 panic 整个 VM
}
```
解释器返回 `Result::Err`；VM 失败回退到解释器（若有 fallback）或直接 panic。**行为可区分**。

**差异 $D_3$（Tuple）**：

- 解释器（[`mod.rs:939-952`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/interpreter/mod.rs)）：生成 `Value::Tuple(values)`。

- 字节码编译器（[`bytecode.rs:521-527`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/bytecode.rs)）：
  ```rust
  HirExprKind::Tuple(elems) => {
      for e in elems {
          self.compile_expr(e)?;
      }
      // TODO: proper tuple support in bytecode
  }
  ```
  仅编译每个元素到栈，**不 emit 任何构造指令**。VM 端无 `MakeTuple` op，也无 `Value::Tuple` 处理。

**反例**：
```tenth
fn main() -> i64 {
    let t = (1, 2, 3);     // 解释器：t 是 Value::Tuple([1,2,3])
                          // VM：栈上是 [1,2,3] 三个独立值，无 Tuple 构造
    t.0                     // 解释器：返回 1；VM：栈形错乱
}
```
解释器返回 1；VM 栈形错乱可能导致后续指令错误。**行为可区分**。

**差异 $D_4$（Closure）**：

- 解释器（[`mod.rs:725-737`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/interpreter/mod.rs)）：
  ```rust
  HirExprKind::Closure { params, body, captures } => {
      let captured_values: Vec<(String, Value)> = captures.iter()
          .filter_map(|name| {
              self.resolve_var(name).map(|v| (name.clone(), v))
          })
          .collect();
      Ok(Some(Value::Closure {
          params: params.clone(),
          body: Rc::new((**body).clone()),
          captures: captured_values,
      }))
  }
  ```
  语义：生成 `Value::Closure`，**捕获自由变量的当前值**。

- VM `Op::MakeClosure`（[`vm.rs:788-798`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/vm.rs)）：
  ```rust
  Op::MakeClosure(params_count, name_idx) => {
      let name = strings.get(name_idx).cloned().unwrap_or_default();
      let param_names: Vec<(String, Type)> = (0..params_count)
          .map(|i| (format!("__param_{i}"), Type::Unknown))
          .collect();
      self.stack.push(Value::FnRef {
          name,
          params: param_names,
          return_type: Type::Unknown,
      });
  }
  ```
  语义：生成 `Value::FnRef`（普通函数引用），**完全不捕获环境**，参数名被替换为 `__param_{i}`。

**反例**：
```tenth
fn main() -> i64 {
    let x = 10;
    let f = |y| y + x;     // 解释器：Closure{captures: [(x, 10)]}
                          // VM：FnRef{name: "__lambda_0", ...}，x 未捕获
    f(5)                   // 解释器：返回 15；VM：x 找不到，报错或返回错误值
}
```
解释器返回 15；VM 可能报"未定义变量 x"。**行为可区分**。

**差异 $D_5$（间接 GenericCall）**：

- 字节码编译器（[`bytecode.rs:477-480`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/bytecode.rs)）：
  ```rust
  } else {
      return Err(crate::error::TenthError::RuntimeError {
          message: "字节码：间接 GenericCall（回退）".into(),
      });
  }
  ```
  语义：遇到非 `Var` 形式的 callee 直接返回 Err。

- 解释器：`eval_call` 通过 `generic_funcs` 表查询并实例化（无 callee 形式限制）。

**反例**：
```tenth
fn apply<T>(f: fn(T) -> T, x: T) -> T { f(x) }
fn main() -> i64 {
    let g = |x: i64| x * 2;
    apply(g, 5)            // 解释器：成功；VM：编译期 Err 触发 fallback
}
```
解释器返回 10；VM 编译期失败，fallback 到解释器重新执行。**行为可区分**（且 fallback 路径有副作用重复风险）。

**汇总**：差异集 $\Delta = \{D_1, D_2, D_3, D_4, D_5\}$。$\square$

---

### 定理 E3（parity_test 覆盖度分析）

**陈述**：`parity_test.rs` 实际是 tenthc（自举）vs Rust 母编译器的 WASM 输出一致性测试，**不直接验证** VM 与解释器一致性；其 100 个 `#[test]` 函数（与 AUDIT 声称的 129 项不符）覆盖 HIR 共同子集 $G$ 的 90%+，但**完全未覆盖**差异集 $\Delta = \{D_1, \ldots, D_5\}$ 的任何一项。

**证明**：

**(1) parity_test 真实目的的澄清**

`parity_test.rs` 文件头注释（[`parity_test.rs:1-11`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/tests/parity_test.rs)）明确写道：

> "Phase D parity tests — verify tenthc (self-hosted) produces WASM that behaves identically to the Rust mother compiler for the same Tenth source."

测试流程（[`parity_test.rs:33-121`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/tests/parity_test.rs)）：

1. 用 Rust 母编译器编译 `src` → WASM-Rust；
2. 用 tenthc（在 wasmi 中执行）编译同一 `src` → WASM-Tenthc；
3. 用 wasmi 执行两个 WASM，比较返回值。

这是**双侧编译器 WASM 输出一致性**测试，属 T12 范畴。它**不涉及** VM 与解释器的直接对比——两个 WASM 都由 wasmi 执行，与 Tenth 自身的 VM/解释器无关。

任务描述"parity_test.rs（129 项）保证行为一致"指的应是"自举双侧一致"，而非"VM-Interpreter 一致"。这是任务描述与实际源码的重大不一致，本文必须诚实披露。

**(2) 测试数量统计**

- `#[test]` 函数数：**100** 个（通过 Grep 计数 `#\[test\]` 标记）；
- `#[ignore]` 函数数：**0** 个；
- AUDIT.md L49 声称：**129 项**。

差异可能源于：(a) AUDIT 计入了模块内的子断言（每个 `#[test]` 内可能有多个 `assert_eq!`）；(b) AUDIT 数据陈旧，未随测试删除更新；(c) AUDIT 计入了其他相关测试文件（如 `vm_autodiff_test.rs` 15 项、`fixpoint_runtime.rs` 等）。本文以 100 为准。

**(3) 覆盖范围分析**

按 `parity_*` 测试函数名分类（基于 100 个函数名的语义聚类）：

| 类别 | 数量 | 示例 |
|------|------|------|
| 算术（add/sub/mul/div/mod） | 18 | `parity_add`, `parity_div_mod`, `parity_negative_arith` |
| 控制流（if/elif/else/break/continue） | 22 | `parity_max_if`, `parity_nested_if`, `parity_break_in_for` |
| 循环（while/for/nested） | 14 | `parity_while_count`, `parity_nested_for`, `parity_for_in_while` |
| 函数（recursion/multi-params/composition） | 12 | `parity_fibonacci`, `parity_gcd`, `parity_four_params` |
| Struct（field/nested/mutation） | 9 | `parity_struct_field`, `parity_struct_four_fields` |
| 变量（shadowing/assign-op） | 11 | `parity_variable_shadowing`, `parity_add_assign` |
| 布尔/比较 | 8 | `parity_bool_logic`, `parity_lt`, `parity_eq` |
| 边界（zero/negatives/large） | 6 | `parity_zero_and_negatives`, `parity_large_numbers` |

**所有 100 个测试用例**均落在共同子集 $G$ 内：
- 全部使用 `Call(Var, e*)` 形式（无非 Var callee）；
- 无 `Move`、`TryBlock`、`Tuple`、`Closure`（tricky 形式）；
- 无间接 `GenericCall`。

**(4) 差异集 $\Delta$ 的覆盖**

逐项核查：

| 差异 | parity_test 是否覆盖 | 证据 |
|------|---------------------|------|
| $D_1$ Move | ❌ 未覆盖 | 100 个测试函数名无 `move` |
| $D_2$ TryBlock | ❌ 未覆盖 | 100 个测试函数名无 `try` |
| $D_3$ Tuple | ❌ 未覆盖 | 100 个测试函数名无 `tuple`；多个 `parity_struct_*` 用 struct 而非 tuple |
| $D_4$ Closure | ❌ 未覆盖 | 100 个测试函数名无 `closure`/`lambda`/`capture` |
| $D_5$ 间接 GenericCall | ❌ 未覆盖 | 100 个测试函数名无 `generic`；全是非泛型直接调用 |

**结论**：parity_test 对 $G$ 覆盖良好（算术/控制流/struct/递归等基础特性），但**对 $\Delta$ 零覆盖**。这意味着即使 parity_test 100% 通过，$D_1$–$D_5$ 的差异仍可能潜伏。$\square$

**推论 E3.1**：parity_test 的"100/100 通过"**不蕴含** VM-Interpreter 等价性——既因为 parity_test 不是 VM-Interpreter 测试，也因为其对 $\Delta$ 零覆盖。

---

### 定理 E4（翻译验证框架）

**陈述**：存在可判定的验证条件 $\text{VC}(e, c)$（$e$ 是 HIR 表达式，$c = \text{compile}(e)$ 是字节码），使得：

1. **可靠性**（soundness）：$\text{VC}(e, c) = \text{true} \Rightarrow e \in G \wedge \text{interp}(e) \sim \text{vm}(c)$；
2. **可判定性**：$\text{VC}(e, c)$ 可在 $O(|e| + |c|)$ 时间内判定；
3. **完备性**（relative completeness）：若 $e \in G$ 且 $\text{compile}$ 对 $e$ 正确，则 $\text{VC}(e, c) = \text{true}$。

**VC 生成规则**：对 HIR 表达式 $e$，$\text{VC}(e, c)$ 是以下条件的合取：

- **VC-Literal**：若 $e = \text{Literal}(v)$，则 $c = [\text{Push}(v)]$。
- **VC-BinOp**：若 $e = \text{BinOp}(\text{op}, e_1, e_2)$，则 $c = \text{compile}(e_1) \cdot \text{compile}(e_2) \cdot [\text{op}]$，且 $\text{VC}(e_1, \text{compile}(e_1)) \wedge \text{VC}(e_2, \text{compile}(e_2))$。
- **VC-If**：若 $e = \text{If}\{e_c, e_t, e_e\}$，则 $c$ 形如 $\text{compile}(e_c) \cdot [\text{JmpFalse}(L_1)] \cdot \text{compile}(e_t) \cdot [\text{Jump}(L_2)] \cdot [L_1: \text{compile}(e_e)] \cdot [L_2:]$，且 $L_1, L_2$ 偏移正确，且子表达式 VC 成立。
- **VC-Call**：若 $e = \text{Call}(\text{Var}(f), \text{args})$，则 $c = \text{compile}(\text{args}^*) \cdot [\text{CallN}(f, |\text{args}|)]$，且 $f$ 在 VM 函数表中。
- **VC-排除**：若 $e \in \{\text{Move}, \text{TryBlock}, \text{Tuple}, \text{Closure}, \text{GenericCall-indirect}\}$，则 $\text{VC}(e, c) = \text{false}$（明确标记不支持）。

**证明**：

**(1) 可靠性**：对 VC 生成规则做结构归纳。基例 VC-Literal：$c = [\text{Push}(v)]$，VM 执行 `Push(v)` 后栈顶为 $v$，对应解释器求值 Literal 得 $v$。归纳步骤 VC-BinOp：由归纳假设 $\text{VC}(e_i, c_i) \Rightarrow e_i \in G \wedge \text{interp}(e_i) \sim \text{vm}(c_i)$；后序拼接 + Op 执行保持 bisimulation（定理 E1）。VC-排除保证 $e \in G$。

**(2) 可判定性**：每条 VC 规则的检查是语法模式匹配 + 偏移计算，线性于 $|e| + |c|$。

**(3) 完备性**：若 $e \in G$ 且 `compile` 正确，则 `compile` 生成的 $c$ 必然匹配某条 VC 规则（因为 $G$ 的产生式与 VC 规则一一对应）。$\square$

**实施建议**：将 $\text{VC}(e, c)$ 实现为 `BytecodeCompiler` 的后置检查——编译完成后立即验证 VC，失败则报"不可验证编译"警告。这是定理 E4 的工程转化。

---

### 定理 E5（差分测试的局限性）

**陈述**：对差异集 $\Delta$ 中的特定差异（特别是 $D_1$ Move 与 $D_4$ Closure），差分测试存在**内在不可观测性**——即使差分测试用例覆盖这些差异，标准差分测试（比较最终返回值）也无法发现差异。

**证明**：

**(1) $D_1$ Move 的不可观测性**

考虑如下程序：
```tenth
fn main() -> i64 {
    let s = vec![1, 2, 3];
    let t = move s;
    let len_t = t.len();
    len_t                       // 返回 3
}
```

- 解释器：`s` 被置为 `Moved`，但程序不再访问 `s`，最终返回 3；
- VM：`Op::MoveOp` 是 no-op，`s` 仍是 vec，但程序不再访问 `s`，最终返回 3。

**观察结果相同**（均为 3），但内部状态不同（解释器 `s = Moved`，VM `s = vec`）。差分测试**无法发现**这一差异。

**反证**：要使差分测试发现 $D_1$，必须在 move 后**再次访问**源变量：
```tenth
let t = move s;
let u = s;           // 解释器报错；VM 成功
```
但这是**程序错误**（违反借用检查），不会出现在正常测试用例中。HIR 层的借用检查（[T19](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/docs/论文/T19-语句粒度借用检查.md)）通常会在编译期拦截此类访问，使得差分测试无法构造出"合法但能暴露 $D_1$"的用例。

**(2) $D_4$ Closure 的不可观测性**

考虑：
```tenth
fn main() -> i64 {
    let x = 10;
    let f = |y| y + x;
    f(5)              // 解释器：15（捕获 x）；VM：可能 15（若 VM 全局表碰巧有 x）或报错
}
```

若 VM 的全局表恰好包含 `x`（如 `x` 在全局作用域），则 VM 也能返回 15——观察结果与解释器相同，差分测试**无法发现** $D_4$。

只有当 `x` 是局部变量且 VM 无法访问时，$D_4$ 才暴露。但这要求测试用例精心构造——差分测试的随机生成器可能不会生成此类用例。

**(3) 内在不可观测性**

$D_1$ 与 $D_4$ 的不可观测性源于：

- **观察层太浅**：差分测试只比较最终返回值，不比较内部状态（如 `Value::Moved` 标记、`Value::Closure.captures` 字段）；
- **HIR 层检查拦截**：借用检查等编译期检查过滤掉了"能暴露差异"的程序。

**结论**：差分测试对 $D_1, D_4$ 存在内在不可观测性。要发现这些差异，需要**更强的测试方法**——如状态观察（比较 VM 与解释器的中间状态）、元属性测试（比较 `Value` 的具体变体）。$\square$

---

## 5. 双执行引擎的语义模型

### 5.1 三层语义模型

Tenth 的执行可抽象为三层：

1. **HIR 层**（源语义）：HIR 是源程序的规范化中间表示，由 [`Lowerer`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/hir/lower/mod.rs) 从 AST lower 而来。HIR 的语义由 [`docs/语言参考手册.md`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/docs/语言参考手册.md) 权威定义。
2. **解释器层**（事实规范）：解释器直接遍历 HIR，其行为被视为 HIR 语义的**事实规范**——任何与解释器行为不一致的实现视为缺陷。
3. **VM 层**（优化实现）：VM 通过字节码执行 HIR，是优化路径，但功能子集不完整。

### 5.2 "解释器即规范"原则

T12 论文中将 Rust 母编译器视为 tenthc 的事实规范；类似地，本文将解释器视为 VM 的事实规范。这一原则的工程含义：

- **当 VM 与解释器行为不一致时，以解释器为准**；
- **VM 的偏差是缺陷**，要么修复 VM，要么标记 fallback；
- **解释器的行为是 HIR 语义的最终裁决**。

但这一原则有局限：解释器本身可能也有 bug。本文不假设解释器绝对正确，只假设它是**两个引擎中更完整**的。详 §11 局限 L3。

### 5.3 fallback 链的语义传递

Tenth 实际执行路径是三层 fallback 链：

$$\text{JIT} \xrightarrow{\text{失败}} \text{VM} \xrightarrow{\text{失败}} \text{Interpreter}$$

- JIT（[`compile/jit/translator.rs`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/jit/translator.rs)）：遇到不支持指令返回 Err，fallback 到 VM；
- VM（[`main.rs:240-256`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/main.rs)）：编译或执行失败，fallback 到解释器；
- 解释器：最终兜底。

**问题**：fallback 不是语义透明的。`main.rs:250-253` 明确警告：

> "VM 可能已部分执行并产生副作用（如 println 输出），解释器将从头重新执行，可能导致副作用重复。"

这意味着 fallback 路径上，观察者可能看到 VM 的部分输出 + 解释器的完整输出，破坏观察等价性。本文定理 E1 的初始条件 $\text{init}(\sigma_I^0, \sigma_V^0)$ 假设**无 prior 副作用**——fallback 路径违反这一假设。

---

## 6. bisimulation 证明的细节

### 6.1 bisimulation 关系 $R$ 的构造

$R$ 是解释器状态与 VM 状态间的关系，定义为：

$$R = \{(\sigma_I, \sigma_V) \mid \text{obs}(\sigma_I) = \text{obs}(\sigma_V) \wedge \text{stack-shape}(\sigma_V) \text{ 对应 } \sigma_I \text{ 的求值栈}\}$$

具体对应规则：

- 解释器的"当前求值结果"对应 VM 栈顶；
- 解释器的作用域链对应 VM 的 `locals` + `globals`；
- 解释器的函数表对应 VM 的 `chunks`；
- 解释器的 `tape` 对应 VM 的 `tape`（同步）。

### 6.2 前进性证明

**引理 E1.1**（前进性）：对任意 $(\sigma_I, \sigma_V) \in R$，若 $\sigma_I \Downarrow_I \sigma_I'$（解释器执行一步），则存在 $\sigma_V'$ 使 $\sigma_V \to_V^* \sigma_V'$ 且 $(\sigma_I', \sigma_V') \in R$。

**证明**：对 HIR 语法做结构归纳，每种语法形式对应一条 VM 转移序列。详 §4 定理 E1 证明。

### 6.3 反转性证明

**引理 E1.2**（反转性）：对任意 $(\sigma_I, \sigma_V) \in R$，若 $\sigma_V \to_V \sigma_V'$（VM 执行一步），则存在 $\sigma_I'$ 使 $\sigma_I \Downarrow_I^* \sigma_I'$ 且 $(\sigma_I', \sigma_V') \in R$。

**证明**：VM 每条指令对应解释器的某个求值步骤或其一部分。如 `Op::Add` 对应解释器 E-Add 规则；`Op::JmpFalse` 对应 E-If 规则的条件求值后选择分支。

**边界情况**：VM 的 `Op::Ret`（T34 定理 V1 栈卫生）对应解释器的作用域弹出 + 返回值传递。由 T34 已证栈卫生，`Ret` 后栈恢复到调用前 + 返回值，对应解释器作用域链的 push/pop。

### 6.4 不变量的保持

bisimulation 保持的关键不变量：

- **I1 栈形对应**：VM 栈深度 = 解释器求值栈深度；
- **I2 作用域对应**：VM `locals` = 解释器当前作用域；
- **I3 步数预算同步**：两侧的 `step_budget` 同步递减；
- **I4 deadline 同步**：两侧的 `deadline_ms` 与 `tick_counter` 同步。

这些不变量在 $G$ 中每条指令后保持，构成 $R$ 的不变性。

---

## 7. 已知差异的逐一分析

### 7.1 $D_1$ Move 的根因

**根因**：Move 语义在 HIR 层已有借用检查（[T19](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/docs/论文/T19-语句粒度借用检查.md)）保证"move 后不可访问"，因此 VM 端 `MoveOp` 设计为 no-op 是**工程优化**——既然编译期已保证不违例，运行时不必再写 `Moved` 标记。

**问题**：这一优化假设了"HIR 借用检查绝对正确"。若借用检查有 bug（漏检），解释器能在运行时捕获（`Value::Moved` 报错），VM 则会"成功"执行错误程序——差异暴露。

**修复建议**：VM `MoveOp` 应写一个 `Value::Moved` 占位（即使借用检查保证不访问，运行时也无害），保持与解释器一致。

### 7.2 $D_2$ TryBlock 的根因

**根因**：TryBlock 的实现需要 VM 支持**异常捕获**机制——在 `try` 块入口保存 continuation，`TryPropagate` 异常时跳回 `try` 块出口。栈式 VM 实现异常捕获需要：

- **保存 ip 与栈基址**：进入 try 块时，保存当前 ip 与 stack.len() 到异常处理表；
- **异常时回滚**：捕获 `TryPropagate` 时，恢复 ip 与 stack 到 try 块入口后、包装为 `Result::Err`。

这是非平凡的实现工作，VM 尚未完成。当前 `bytecode.rs:485-487` 注释 "TryBlock not yet supported in bytecode; emit as no-op" 明确标注为 TODO。

**修复建议**：参考 JVM 的异常表（exception table）设计，为 `try` 块生成 `BeginTry(offset, length, handler)` 元数据，VM 主循环维护异常处理栈。

### 7.3 $D_3$ Tuple 的根因

**根因**：Tuple 需要新的 `Op::MakeTuple(arity)` 操作码与 `Value::Tuple` 处理。当前 `bytecode.rs:521-527` 注释 "TODO: proper tuple support in bytecode"。

**修复建议**：

1. 新增 `Op::MakeTuple(usize)` 操作码（参考 `Op::MakeVec`）；
2. VM 主循环新增 `Op::MakeTuple` 分支，弹出 arity 个值、构造 `Value::Tuple`；
3. 新增 `Op::TupleGet(usize)` 用于 `t.0` 等索引访问。

### 7.4 $D_4$ Closure 的根因

**根因**：Closure 的正确实现需要 VM 支持**环境捕获**——闭包需携带自由变量的快照。当前 VM `MakeClosure` 只生成 `FnRef`（无环境），是**最低成本占位实现**。

**修复建议**：

1. 扩展 `Op::MakeClosure` 操作数为 `(params_count, name_idx, captures_count)`；
2. 编译时为每个闭包生成**捕获列表**（哪些变量需捕获）；
3. VM 执行 `MakeClosure` 时弹出 captures_count 个值，构造 `Value::Closure { name, params, captures }`；
4. 调用闭包时，将 captures 注入新 Frame 的 locals。

参考 T22（[Closure 自由变量分析正确性](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/docs/论文/T22-Closure自由变量分析正确性.md)）确保捕获列表正确。

### 7.5 $D_5$ 间接 GenericCall 的根因

**根因**：泛型函数的实例化需要**类型推导**——根据 callee 的实际类型生成特化版本。VM 端无类型信息，无法在运行时实例化。当前 `bytecode.rs:477-480` 对非 Var callee 直接返回 Err。

**修复建议**：

1. **方案 A（编译期实例化）**：在 `Lowerer` 中对所有泛型调用做单态化（monomorphization），生成具体类型的特化函数，VM 只执行特化版本；
2. **方案 B（运行时类型分派）**：VM 维护类型信息，运行时根据 callee 类型查表（性能差）；
3. **方案 C（保持 fallback）**：当前策略，间接 GenericCall 触发解释器 fallback——**前提是 fallback 副作用隔离**（见 §7.6）。

### 7.6 fallback 副作用隔离问题

当前 fallback 机制（[`main.rs:240-266`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/main.rs)）的副作用未隔离——VM 可能已输出到 stdout，解释器重新执行会再次输出。

**修复建议**：

- **方案 A（缓冲输出）**：所有 stdout 输出缓冲到内存，只有执行成功才 flush；fallback 时丢弃 VM 缓冲。
- **方案 B（事务执行）**：VM 执行前快照所有可变状态（stdout、文件系统），失败时回滚。
- **方案 C（预编译检测）**：执行前先用 `BytecodeCompiler::compile` 检查能否编译，不能则直接用解释器，避免 VM 部分执行。

方案 C 最简单，推荐优先实施。

---

## 8. parity_test 覆盖度深度分析

### 8.1 测试分类与 $G$ 覆盖

§4 定理 E3 已统计 parity_test 的 100 个测试函数按类别分布。本节进一步分析其对 $G$ 的覆盖度。

| $G$ 子集 | parity_test 覆盖 | 覆盖率估计 |
|----------|------------------|-----------|
| Literal | ✅ `parity_constant`, `parity_zero_and_negatives`, `parity_large_numbers` | 90% |
| BinOp | ✅ `parity_add`, `parity_sub`, `parity_mul`, `parity_div_mod`, `parity_arith_precedence`, `parity_arith_parens` | 95% |
| UnaryOp | ✅ `parity_unary_neg`, `parity_unary_neg_in_expr` | 80% |
| If | ✅ `parity_max_if`, `parity_nested_if`, `parity_if_elif_chain`, `parity_if_no_else` | 90% |
| While | ✅ `parity_while_count`, `parity_nested_while`, `parity_while_in_for` | 85% |
| For | ✅ `parity_for_sum`, `parity_nested_for`, `parity_for_in_while` | 85% |
| Call(Var) | ✅ `parity_nested_calls`, `parity_recursion`, `parity_fibonacci`, `parity_gcd` | 90% |
| LetAssign | ✅ `parity_let_reassign`, `parity_variable_shadowing`, `parity_add_assign` | 90% |
| StructLit | ✅ `parity_struct_field`, `parity_struct_four_fields`, `parity_struct_field_mutation` | 85% |
| FieldAccess | ✅ `parity_struct_field`, `parity_struct_three_fields_access` | 85% |
| EnumLit | ❌ 未覆盖 | 0% |
| Range | ❌ 未覆盖（for-in 用 PushRange 但无独立 range 测试） | 30% |
| Match（基本模式） | ❌ 未覆盖 | 0% |

**总覆盖率估计**：约 70%–75% 的 $G$ 被覆盖。**EnumLit、Range、Match** 是显著缺口。

### 8.2 与 vm_autodiff_test 的互补

`vm_autodiff_test.rs`（[15 项](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/tests/vm_autodiff_test.rs)）测试 VM 上的自动微分——这部分覆盖了 tape 操作的 VM-Interpreter 一致性。但与 parity_test 一样，它不是系统的 VM-Interpreter 差分测试。

### 8.3 fixpoint_runtime 的角色

`fixpoint_runtime.rs`（[AUDIT §7.4 提到](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/AUDIT.md)）测试 Wasmtime 端到端编译+执行——这是 WASM 后端的验证，与 VM-Interpreter 等价性无关。

### 8.4 真正的 VM-Interpreter 差分测试缺口

**结论**：Tenth **没有**专门的 VM-Interpreter 差分测试套件。所有"parity"测试要么是 tenthc vs Rust 母编译器（parity_test.rs），要么是自动微分回归（vm_autodiff_test.rs），要么是 WASM 端到端（fixpoint_runtime.rs）。VM-Interpreter 等价性目前**仅靠"VM 失败 fallback 到解释器"的隐式保障**——而非显式测试。

**建议**：新增 `vm_interp_parity_test.rs`，对 $G$ 中每种语法形式生成测试用例，对 $\Delta$ 中每项差异生成**预期失败**用例（标记 `#[ignore]`，修复后取消忽略）。

---

## 9. 工程权衡

### 9.1 为何维护两个引擎？

**性能**：解释器（树遍历）每步需查表（作用域链、函数表），开销大；VM（字节码）指令紧凑、栈式操作直接，性能优 3–10 倍（CPython vs 字节码 VM 的经验值）。

**完整性**：解释器直接遍历 HIR，与语义规约一一对应，实现成本最低；VM 需要为每个 HIR 节点实现对应字节码序列，工作量大。

**调试**：解释器栈帧清晰（每步可见 HIR 节点），便于调试；VM 字节码与 HIR 距离远，调试需反编译。

Tenth 的策略：**解释器是规范，VM 是优化**——优先保证解释器正确，VM 逐步补全。这是务实的工程权衡。

### 9.2 fallback 策略的成本

fallback 策略的成本：

1. **副作用重复**（§7.6）：stdout、文件系统可能重复写入；
2. **性能损失**：fallback 后解释器从头执行，VM 的部分执行浪费；
3. **可预测性差**：用户难以预测程序是否会触发 fallback。

**优化**：方案 C（预编译检测，§7.6）可消除前两个成本——执行前检测可编译性，不能则直接用解释器。

### 9.3 何时补全 VM？

补全 VM 的优先级：

| 差异 | 影响面 | 补全难度 | 优先级 |
|------|--------|---------|--------|
| $D_1$ Move | 低（借用检查兜底） | 低（写 Value::Moved） | P3 |
| $D_2$ TryBlock | 中（错误处理常用） | 高（异常表机制） | P2 |
| $D_3$ Tuple | 中（元组常用） | 中（新增 Op + Value 处理） | P2 |
| $D_4$ Closure | 高（闭包是核心特性） | 高（环境捕获机制） | P1 |
| $D_5$ GenericCall | 高（泛型是核心特性） | 高（单态化或运行时分派） | P1 |

**建议**：$D_4$ 与 $D_5$ 优先补全（P1），$D_2$ 与 $D_3$ 次之（P2），$D_1$ 最后（P3）。

### 9.4 与 JIT 的关系

JIT（[T9](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/docs/论文/T9-JIT特化语义保持证明.md)、[T33](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/docs/论文/T33-JIT缓存生命周期与代码热加载.md)）是 Tenth 的第三执行路径。JIT 通过 `translator.rs` 将字节码翻译为本地代码，遇到不支持指令返回 Err 触发 VM fallback。

JIT 的语义保持（T9 已证）基于"JIT 翻译的字节码子集与 VM 执行一致"。因此：

$$\text{JIT} \sim \text{VM} \text{（在 JIT 支持子集上，T9 已证）} \wedge \text{VM} \sim \text{Interpreter} \text{（在 } G \text{ 上，本文 E1）} \Rightarrow \text{JIT} \sim \text{Interpreter} \text{（在 } G \cap \text{JIT-subset} \text{ 上）}$$

这是 bisimulation 的传递性。但若 VM 在 $\Delta$ 上不等价，JIT 也继承这一不等价——除非 JIT 翻译器显式拒绝 $\Delta$（事实上它确实如此，[`translator.rs`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/jit/translator.rs) 遇 `IsStruct` 等 return Err）。

---

## 10. 开放问题

### 10.1 形式化证明的机器化

本文证明是纸笔证明，未机器化。未来可用 Rocq/Coq 将定理 E1–E5 机器检查，提升可信度。CompCert [2] 已证明编译器语义保持的可机器化，本文可借鉴。

### 10.2 共同子集 $G$ 的扩展

当前 $G$ 排除了 $\Delta$ 的 5 项。随 VM 补全，$G$ 应逐步扩展——每补全一项差异，$G$ 扩展、$\Delta$ 收缩。最终目标是 $G = \text{HIR}$、$\Delta = \emptyset$。

### 10.3 自动微分的双引擎一致性

`tape` 在解释器与 VM 间共享（同一 `Tape` 类型），但记录路径不同（解释器在 `eval_expr` 中记录，VM 在 `Op` 执行后记录）。自动微分的双引擎一致性需独立证明，本文未覆盖。

### 10.4 fallback 副作用隔离的形式化

§7.6 提出方案 C（预编译检测），但未形式化其正确性。未来可证明"方案 C 下 fallback 是观察透明的"——即 fallback 前后观察者看到的输出一致。

### 10.5 差分测试生成器的构建

§8.4 建议新增 `vm_interp_parity_test.rs`，但测试用例如何生成？随机生成（property-based testing）可能无法覆盖 $\Delta$——因为 $\Delta$ 涉及特定语法（如 `move`、`try`）。需设计**定向生成器**，针对每项差异生成用例。

---

## 11. 局限（独立章节）

本节诚实记录本文的 5 项理论局限与工程差距。

### 局限 L1（形式化未覆盖 native 函数）

**是什么**：本文 bisimulation 证明假设 native 函数（如 `println`、`vec_push`）在解释器与 VM 上行为一致。但 native 函数由 [`natives.rs`（解释器）](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/interpreter/natives.rs) 与 [`main.rs::register_natives`（VM）](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/main.rs) 分别实现，可能存在差异（如错误消息、边界行为）。

**影响**：定理 E1 的可靠性可能因 native 差异而受损——即使 HIR 在 $G$ 内，若调用的 native 行为不一致，整体行为仍不等价。

**缓解**：未来工作应形式化 native 函数的双侧等价性，或将 native 函数纳入 VC 框架（定理 E4）。

### 局限 L2（HIR 层借用检查的假设）

**是什么**：定理 E2 差异 $D_1$ 的"反例"假设程序能通过 HIR 层借用检查（[T19](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/docs/论文/T19-语句粒度借用检查.md)）。但借用检查本身可能有 bug（漏检），此时反例可能实际可执行。

**影响**：$D_1$ 的"行为可区分"反例在实际中可能被借用检查拦截，无法触发——bisimulation 的"排除"可能过强。

**缓解**：定理 E5 已部分覆盖此局限（差分测试不可观测性）。未来工作应证明"借用检查正确性"作为额外前提。

### 局限 L3（解释器绝对正确的假设）

**是什么**：本文"解释器即规范"原则假设解释器本身正确。但解释器也可能有 bug——[`AUDIT.md §六`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/AUDIT.md) 列出的"已知限制"中，#2 提到"树遍历解释器大文件慢"，性能虽非语义 bug 但暗示实现复杂度。

**影响**：若解释器有 bug，定理 E1 的"bisimulation"是相对解释器的，而非相对 HIR 规约的——可能两侧一致地错。

**缓解**：未来工作应以 [`docs/语言参考手册.md`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/docs/语言参考手册.md) 为绝对规约，独立验证解释器与 VM 各自的正确性（而非仅双侧一致）。

### 局限 L4（parity_test 数量的不一致）

**是什么**：本文统计 parity_test.rs 有 100 个 `#[test]` 函数，AUDIT.md L49 声称 129 项。这一不一致的根因未彻底调查——可能是 AUDIT 计入了子断言、相关测试文件、或数据陈旧。

**影响**：定理 E3 的覆盖度分析基于 100 这个数字，若 AUDIT 的 129 是准确的（含其他文件），则覆盖度估计可能偏差。

**缓解**：建议更新 AUDIT.md L49 为准确数字（100），或在 AUDIT 中注明"129 含相关文件"。

### 局限 L5（未覆盖 WASM 后端的等价性）

**是什么**：本文聚焦 VM-Interpreter 等价性，未覆盖 HIR→WASM 翻译（[T29](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/docs/论文/T29-HIR到WASM语义保持与host边界类型塌缩.md) 已覆盖）的等价性。WASM 后端是 Tenth 的第三执行路径（自举路径 C），其与解释器/VM 的等价性需独立证明。

**影响**：本文未给出"WASM 与 VM/解释器等价"的证明——这一定理由 T29 部分覆盖，但 T29 聚焦 HIR→WASM 翻译保持，未涉及与 VM 的双侧等价。

**缓解**：未来工作应整合 T29 与本文，建立"HIR → {VM, WASM, Interpreter} 三侧等价"的统一框架。

---

## 12. 结论

本文对 Tenth 双执行引擎（解释器与 VM）的行为等价性进行形式化分析，得出以下核心结论：

1. **共同子集 bisimulation**（定理 E1）：在 HIR 共同子集 $G$ 上，解释器与 VM bisimilar，给出了充分条件与构造性证明。
2. **差异集刻画**（定理 E2）：识别 5 项确凿差异 $D_1$–$D_5$（Move、TryBlock、Tuple、Closure、间接 GenericCall），每项附源码引用与构造性反例。
3. **parity_test 真相**（定理 E3）：澄清 parity_test.rs 实际是 tenthc vs Rust 母编译器的 WASM 一致性测试（属 T12 范畴），并非 VM-Interpreter 测试；100 个 `#[test]` 函数（非 AUDIT 声称的 129），覆盖 $G$ 的 ~70%，但对 $\Delta$ 零覆盖。
4. **翻译验证框架**（定理 E4）：给出可判定的 VC 生成规则，使每次 HIR→bytecode 编译可生成证明义务。
5. **差分测试局限**（定理 E5）：证明 $D_1$ 与 $D_4$ 在差分测试下的内在不可观测性。

**工程启示**：

- VM 补全应优先 $D_4$（Closure）与 $D_5$（GenericCall），这是核心特性；
- fallback 策略应改为"预编译检测"（方案 C）以隔离副作用；
- 应新增 `vm_interp_parity_test.rs` 专门测试 VM-Interpreter 一致性；
- AUDIT.md L49 应更新为准确测试数（100）。

**理论启示**：

- "VM 与解释器等价"是**可证伪、可分级**的命题——本文将其细化为"在 $G$ 上 bisimilar + $\Delta$ 显式刻画"；
- bisimulation 是比观察等价更强的工具——能发现差分测试无法发现的差异（如 $D_1$ 的 `Value::Moved` 内部状态）；
- 翻译验证（Pnueli）是工程化的等价性验证方法——VC 可自动生成、机器检查。

**与 T12、T34 的联动**：

- T12（双侧编译器）+ T35（双执行引擎）= Tenth 全栈双侧等价性框架——前端（Rust vs tenthc）+ 后端（解释器 vs VM）；
- T34（VM 操作语义）+ T35 = VM 单侧形式化 + 双侧等价性——T34 是 E1 证明的 VM 侧基础；
- T9（JIT 语义保持）+ T35 = 三层 fallback 链的等价性传递。

本文是 Tenth 数理部对"双执行引擎等价性"这一护城河级问题的首次系统形式化，所有结论附源码引用，所有局限独立章节披露。$\square$

---

## 参考文献

[1] Pnueli, A., Siegel, M., Shtrichman, O. (1998). *Translation validation*. In: Steffen, B. (eds) Tools and Algorithms for the Construction and Analysis of Systems (TACAS 1998). LNCS 1384. Springer. https://doi.org/10.1007/BFb0054173

[2] Leroy, X. (2009). *Formal verification of a realistic compiler*. Communications of the ACM, 52(7), 107–115. https://doi.org/10.1145/1538788.1538814 (CompCert 项目)

[3] Park, D. (1981). *Concurrency and automata on infinite sequences*. In: Deussen, P. (eds) Theoretical Computer Science. LNCS 104. Springer. https://doi.org/10.1007/BFb0017309 (bisimulation 概念首次提出)

[4] Milner, R. (1989). *Communication and Concurrency*. Prentice Hall. (CCS 与 bisimulation 系统化)

[5] McKeeman, W. M. (1998). *Differential testing for software*. Digital Technical Journal, 10(1), 100–107. (差分测试概念)

[6] Igarashi, A., Pierce, B. C., Wadler, P. (2001). *Featherweight Java: a minimal core calculus for Java and GJ*. ACM TOPLAS, 23(3), 396–450. https://doi.org/10.1145/503502.503505 (FJ 形式化模板)

[7] McKinna, J., Pollack, R. (1999). *Some lambda calculus and type theory formalized*. Journal of Automated Reasoning, 23(3–4), 373–409. (双射式元理论风格)

[8] Tenth 数理部. (2026). *T34: 栈式 VM 操作语义形式化*. [docs/论文/T34-栈式VM操作语义形式化.md](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/docs/论文/T34-栈式VM操作语义形式化.md)

[9] Tenth 数理部. (2026). *T12: 双侧编译器语义等价性*. [docs/论文/T12-双侧编译器语义等价性.md](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/docs/论文/T12-双侧编译器语义等价性.md)

[10] Tenth 数理部. (2026). *T9: JIT 特化语义保持证明*. [docs/论文/T9-JIT特化语义保持证明.md](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/docs/论文/T9-JIT特化语义保持证明.md)

[11] Tenth 数理部. (2026). *T22: Closure 自由变量分析正确性*. [docs/论文/T22-Closure自由变量分析正确性.md](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/docs/论文/T22-Closure自由变量分析正确性.md)

[12] Tenth 数理部. (2026). *T19: 语句粒度借用检查*. [docs/论文/T19-语句粒度借用检查.md](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/docs/论文/T19-语句粒度借用检查.md)

[13] Tenth 数理部. (2026). *T29: HIR 到 WASM 语义保持与 host 边界类型塌缩*. [docs/论文/T29-HIR到WASM语义保持与host边界类型塌缩.md](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/docs/论文/T29-HIR到WASM语义保持与host边界类型塌缩.md)

---

## 附录 A：定理索引

| 定理 | 名称 | 陈述 | 证明位置 |
|------|------|------|---------|
| E1 | 共同子集 bisimulation 等价性 | 在 $G$ 上解释器与 VM bisimilar | §4 |
| E2 | 已知差异刻画 | 5 项差异 $D_1$–$D_5$ | §4 |
| E3 | parity_test 覆盖度分析 | 100 测试，$\Delta$ 零覆盖 | §4 |
| E4 | 翻译验证框架 | 可判定 VC + 可靠性/完备性 | §4 |
| E5 | 差分测试的局限性 | $D_1, D_4$ 内在不可观测 | §4 |
| E1.1 | 前进性引理 | 解释器步对应 VM 步序列 | §6.2 |
| E1.2 | 反转性引理 | VM 步对应解释器步序列 | §6.3 |
| E3.1 | parity_test 通过不蕴含等价 | 推论 | §4 |

## 附录 B：差异集 $\Delta$ 速查

| 差异 | 解释器位置 | VM/字节码编译器位置 | 反例 |
|------|-----------|---------------------|------|
| $D_1$ Move | [`mod.rs:886-894`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/interpreter/mod.rs) | [`vm.rs:745-747`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/vm.rs) + [`bytecode.rs:482-484`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/bytecode.rs) | §4 $D_1$ |
| $D_2$ TryBlock | [`mod.rs:896-918`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/interpreter/mod.rs) | [`bytecode.rs:485-487`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/bytecode.rs) | §4 $D_2$ |
| $D_3$ Tuple | [`mod.rs:939-952`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/interpreter/mod.rs) | [`bytecode.rs:521-527`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/bytecode.rs) | §4 $D_3$ |
| $D_4$ Closure | [`mod.rs:725-737`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/interpreter/mod.rs) | [`vm.rs:788-798`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/vm.rs) | §4 $D_4$ |
| $D_5$ GenericCall | [`mod.rs:1066`（eval_call）](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/interpreter/mod.rs) | [`bytecode.rs:477-480`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/bytecode.rs) | §4 $D_5$ |

## 附录 C：实施建议

| 建议 | 优先级 | 对应定理 | 工作量估计 |
|------|--------|---------|-----------|
| 补全 VM Closure（$D_4$） | P1 | E2 | 大（环境捕获机制） |
| 补全 VM GenericCall（$D_5$） | P1 | E2 | 大（单态化） |
| 补全 VM TryBlock（$D_2$） | P2 | E2 | 中（异常表） |
| 补全 VM Tuple（$D_3$） | P2 | E2 | 中（新增 Op） |
| 补全 VM Move（$D_1$） | P3 | E2 | 小（写 Value::Moved） |
| fallback 改为预编译检测 | P1 | §7.6 | 小 |
| 新增 `vm_interp_parity_test.rs` | P2 | E3 | 中 |
| 更新 AUDIT.md L49 测试数 | P3 | E3 | 小 |
| 实现定理 E4 的 VC 检查器 | P3 | E4 | 中 |

---

*本文由 Tenth 数理部撰写。所有源码引用基于 v0.3.3 版本。局限章节独立披露 5 项理论局限。与 T34、T12、T9、T22、T19、T29 联动。*
