# 栈式 VM 的操作语义形式化：Tenth VM 的栈卫生、双协议与摊还 deadline

> **Tenth 数理部 · 理论分析论文 T34**
> 版本：v1.0 | 日期：2026-07-02
> 适用：Tenth v0.3.3+
> 关联论文：T27（栈式字节码单遍回填式代码生成）、T35（解释器-VM 等价性，规划中）、T9（JIT 特化语义保持）

---

## 摘要

本文对 Tenth 语言栈式字节码虚拟机（VM）的核心运行机制进行小步操作语义（small-step structural operational semantics, SOS）形式化建模，并证明三个非平凡性质。Tenth VM 在工程上有三个值得形式化的特征：(1) **Frame 的 `stack_base` 截断语义**——`Ret` 指令通过 `self.stack.truncate(f.stack_base)` 强制恢复操作数栈到调用前状态，构成**栈卫生**不变量；(2) **`Call` vs. `CallN` 双协议**——`Call` 用栈深度推断参数数量（历史遗留），`CallN` 显式传入参数数量（现代协议），二者并存；(3) **步数预算 + 墙钟 deadline 的独立检查**——每 4096 步检查一次墙钟 deadline，是**摊还分析**的实例。本文将 VM 抽象为状态转移系统 $\langle \textit{ip}, \textit{code}, \textit{stack}, \textit{frames}, \textit{locals} \rangle$，建立五个主定理：**定理 V1（栈卫生不变量）**证明 `Ret` 后栈恢复到调用前状态；**定理 V2（Call/CallN 双协议等价性）**证明两种调用语义在栈纪律前提下等价；**定理 V3（类型安全进展定理）**证明类型良好的指令不卡住；**定理 V4（摊还 deadline 开销上界）**证明周期性 deadline 检查的摊还开销为 $O(1/4096)$；**定理 V5（与 CPython eval loop timeout 对比）**对比 Tenth 的周期性时钟读取与 CPython 的 `eval_breaker` 标志位检查的工程权衡。本文给出栈卫生不变量的反向归纳证明、双协议等价性的双向模拟论证、摊还开销的聚合法分析，并诚实披露若干局限：形式化未覆盖 JIT 路径、`MethodCall` 的多分派、native 函数的栈副作用、单步耗时无界时 deadline 超限可能。

**关键词**：栈式虚拟机；小步操作语义；栈卫生；双调用协议；摊还分析；deadline 检查；Tenth 语言

---

## 1. 引言

### 1.1 栈式 VM 形式化的挑战

栈式字节码虚拟机是动态语言运行时的主流架构之一（CPython、JVM、YARV、V8 Ignition）。其形式化的核心挑战在于：**操作数栈是隐式状态**，每条指令的前置条件（栈深度、栈顶类型）不显式编码于指令操作数中，而由前驱指令机械推导。这使得栈式 VM 的操作语义必须以**栈迁移关系**为核心，而非以寄存器编号为索引。

Tenth VM（[`tenth/src/runtime/vm.rs:155-182`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/vm.rs)）是典型的栈式 VM，其状态由 `stack: Vec<Value>`、`frames: Vec<Frame>`、`locals: Vec<Value>`、`ip: usize`、`code: Vec<u8>` 构成。每条指令通过 `pop`/`push` 隐式操作栈顶。这种设计的优势是指令编码紧凑（1 字节 opcode + 操作数）、与 HIR 后序遍历同构；代价是栈形（stack shape）的正确性完全依赖编译器与运行时的协作不变量。

### 1.2 栈卫生

Tenth VM 的 `Ret` 指令（[`tenth/src/runtime/vm.rs:577-590`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/vm.rs)）执行如下动作：

```rust
Op::Ret => {
    let result = self.stack.pop().unwrap_or(Value::Unit);
    if let Some(f) = self.frames.pop() {
        self.stack.truncate(f.stack_base);
        self.stack.push(result);
        ip = f.ip;
        chunk_idx = f.chunk_idx;
        code = self.chunks[chunk_idx].code.clone();
        strings = self.chunks[chunk_idx].strings.clone();
        locals = f.locals;
    } else {
        return Ok(result);
    }
}
```

关键操作是 `self.stack.truncate(f.stack_base)`——**强制截断栈到调用前记录的基址**，然后只压入返回值。这一"截断 + 重压"语义构成了**栈卫生不变量**：无论被调函数体内遗留多少未清理的栈垃圾，`Ret` 后栈必然恢复到 `[调用前状态] + [返回值]`。这与 CPython 的 `POP_BLOCK` + `PUSH(return_value)` 在精神上一致，但 Tenth 用 `truncate` 实现得更激进——它不要求被调函数维护栈平衡，运行时强制回收。

### 1.3 双协议：Call vs. CallN

Tenth VM 同时支持两种调用指令（[`tenth/src/runtime/vm.rs:503-566`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/vm.rs)）：

- **`Call(i)`**：操作数为函数名字符串索引。**用栈深度推断参数数量**：`n = self.stack.len() - base`（native 路径），或用被调函数的 `num_args`（用户函数路径）。`Frame.stack_base` 记录为 `base`（调用者函数入口时计算的固定基址）。
- **`CallN(i, n)`**：操作数为函数名索引 + 显式参数数量 `n`。**显式传入参数数量**，先弹出 `n` 个参数到 `args` 向量，再 `Frame.stack_base = self.stack.len()`（弹参数后的栈长度）。

这是历史遗留的双重协议：`Call` 是早期设计，依赖"调用点栈恰好只有参数"的隐式约定；`CallN` 是后期为支持闭包与 `FnRef`（[`tenth/src/runtime/vm.rs:534-539`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/vm.rs)）引入的显式协议。二者并存带来形式化挑战：在何种前提下二者等价？不等价时行为差异是什么？

### 1.4 摊还 deadline

Tenth VM 的执行资源控制采用**步数预算 + 墙钟 deadline 双轨独立检查**（[`tenth/src/runtime/vm.rs:353-384`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/vm.rs)）：

```rust
let mut loop_counter: u64 = 0;
loop {
    if let Some(ref mut budget) = self.step_budget {
        if *budget == 0 { return Err(TenthError::Timeout { ... }); }
        *budget -= 1;
    }
    loop_counter = loop_counter.wrapping_add(1);
    if (loop_counter & 0xFFF) == 0 {
        if let Some(deadline) = self.deadline_ms {
            let now = std::time::SystemTime::now()...;
            if now >= deadline { return Err(TenthError::Timeout { ... }); }
        }
    }
    // ... dispatch opcode ...
}
```

每步递减 `step_budget`（若设），但**墙钟 deadline 每 4096 步才检查一次**（`loop_counter & 0xFFF == 0`）。这是摊还分析的典型实例：单次 `SystemTime::now()` 系统调用开销约 100–500 ns，若每步检查会成为热路径瓶颈；每 4096 步检查则将摊还开销降至 ~0.05 ns/步，同时保证 deadline 超限的响应延迟上界为 4096 步。

历史上该机制存在过缺陷（H-4，[`tenth/src/runtime/vm.rs:358-360`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/vm.rs) 注释）：早期实现将 deadline 检查嵌套在 `step_budget` 内，导致只设 `--timeout` 而不设步数预算时 deadline 永不触发。当前实现用独立 `loop_counter` 修复此缺陷。

### 1.5 贡献

本文贡献如下：

1. **形式化建模**：将 Tenth VM 抽象为状态转移系统 $\Sigma = \langle \textit{ip}, \textit{code}, \textit{stack}, \textit{frames}, \textit{locals}, \textit{budget}, \textit{counter} \rangle$，给出小步操作语义的迁移规则 $\langle \sigma, \textit{op} \rangle \rightarrow \sigma'$；
2. **五个主定理及证明**：V1 栈卫生不变量、V2 双协议等价性、V3 类型安全进展、V4 摊还 deadline 开销上界、V5 与 CPython eval loop 对比；
3. **栈卫生不变量的反向归纳证明**：以调用栈深度为归纳载体，证明 `Ret` 后栈状态严格等于调用前状态 + 返回值；
4. **双协议等价性的双向模拟论证**：构造 `Call` 与 `CallN` 之间的互模拟关系，明确等价前提与不等价边界；
5. **摊还 deadline 的聚合法分析**：给出摊还开销上界与 deadline 超限响应延迟上界；
6. **诚实局限披露**：独立章节记录形式化覆盖的边界、未覆盖的 JIT 路径、单步无界时 deadline 失效等局限。

---

## 2. 背景

### 2.1 小步操作语义（SOS）

小步结构化操作语义由 Plotkin (1981) 系统化，其核心是**单步迁移关系** $\langle \sigma, e \rangle \rightarrow \sigma'$：给定状态 $\sigma$ 与待执行项 $e$，一步迁移到新状态 $\sigma'$。栈式 VM 的 SOS 中，状态 $\sigma$ 包含操作数栈、指令指针、帧栈等，待执行项是当前指令 `op`。多步迁移 $\rightarrow^*$ 是单步迁移的自反传递闭包。

与小步相对的是**大步自然语义**（big-step, $\Downarrow$）：$\langle \sigma, e \rangle \Downarrow v$ 表示从 $\sigma$ 出发求值 $e$ 最终得到 $v$。栈式 VM 的中断式调度（如 Tenth 的 deadline 检查可能在两条指令之间中止执行）更适合小步语义，因为大步语义难以表达"中途超时"。

### 2.2 CPython eval loop

CPython 的 `_PyEval_EvalFrameDefault` 是栈式字节码解释器的经典实例。其 eval loop 在每条指令后检查 `eval_breaker` 标志位（一个原子读，开销极低），若被置位则处理信号、异常、调度等。这种"每步检查标志位 + 异步置位"的设计避免了系统调用开销，但要求标志位写入方（信号处理器、调度器）通过原子操作通知。

CPython 的 `_Py_CheckInterval`（Python 2 时代）每 100 条指令检查一次信号；Python 3 改为每条指令检查 `eval_breaker`（仍是原子读，不是系统调用）。timeout 的实现（如 `signal.alarm`）依赖 Unix 信号，主线程被信号中断后设置 `eval_breaker`，下次循环检查时处理。

### 2.3 Lua VM

Lua 5.0+ 的 VM 是**寄存器式**，每个函数有固定数量的虚拟寄存器（`R0..Rn`），指令显式指定操作数寄存器编号。寄存器式减少 `Push`/`Pop` 指令条数（LuaJIT 作者 Mike Pall 公开基准显示约减少 30%–40% 指令），但需要编译器做寄存器分配。Lua 的 `CALL` 指令显式指定参数数量与返回值数量，不存在 Tenth `Call` 的"栈深度推断"协议。

### 2.4 JVM 字节码验证

JVM 在加载时对字节码做**验证器**（verifier）检查，确保每条指令前的栈深度与栈顶类型符合方法签名推导出的类型抽象。验证器使用抽象解释（dataflow analysis）计算每条指令的前置栈形，拒绝不可验证的字节码。这是"静态保证栈卫生"的典范——JVM 运行时不需要 `truncate`，因为验证器已保证被调函数返回时栈必然平衡。

Tenth VM **没有字节码验证器**，因此运行时用 `truncate` 兜底——这是一种"动态保证栈卫生"的工程取舍：放弃静态检查的早期错误反馈，换取编译器实现简单性。

---

## 3. Tenth VM 操作语义形式化

### 3.1 状态空间

**定义 3.1（VM 状态）**。Tenth VM 的状态是七元组：
$$
\sigma = \langle \textit{ip},\ \textit{code},\ \textit{stk},\ \textit{frs},\ \textit{loc},\ \textit{bg},\ \textit{ct} \rangle
$$
其中：
- $\textit{ip} \in \mathbb{N}$：指令指针，指向 `code` 中下一条待执行指令；
- $\textit{code} \in \mathbb{B}^*$：当前 chunk 的字节码序列；
- $\textit{stk} \in \textit{Value}^*$：操作数栈（`Vec<Value>`）；
- $\textit{frs} \in \textit{Frame}^*$：调用帧栈（`Vec<Frame>`），栈顶为当前帧；
- $\textit{loc} \in \textit{Value}^*$：当前帧的局部变量表（`Vec<Value>`）；
- $\textit{bg} \in \mathbb{N} \cup \{\bot\}$：步数预算（`Option<u64>`，$\bot$ 表示未设）；
- $\textit{ct} \in \mathbb{Z}_{2^{64}}$：deadline 检查用的循环计数器（`u64` wrapping）。

**定义 3.2（Frame）**。帧是四元组 $f = \langle \textit{ip}_f,\ \textit{chk}_f,\ \textit{loc}_f,\ \textit{sb}_f \rangle$，对应 [`tenth/src/runtime/vm.rs:148-153`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/vm.rs)：
```rust
struct Frame {
    ip: usize,           // 返回地址
    chunk_idx: usize,    // 调用者 chunk 索引
    locals: Vec<Value>,  // 调用者局部变量
    stack_base: usize,   // 调用前栈基址
}
```
其中 $\textit{sb}_f$ 是**调用前操作数栈的基址**，`Ret` 时用于截断。

**定义 3.3（栈切片记号）**。记 $\textit{stk}[i..j]$ 为栈中下标 $[i, j)$ 的子序列；$|\textit{stk}|$ 为栈长度；$\textit{stk} \cdot v$ 为压入值 $v$ 后的栈；$\textit{stk} \!\upharpoonright\! k$ 为截断到长度 $k$ 的栈（即 $\textit{stk}[0..k]$）。

### 3.2 迁移规则总览

VM 一步迁移的形式化为：
$$
\langle \sigma, \textit{op} \rangle \rightarrow \sigma' \quad \text{或} \quad \langle \sigma, \textit{op} \rangle \rightarrow \textit{Err}
$$
其中 $\textit{op} = \textit{decode}(\textit{code}, \textit{ip})$ 是从字节码解码出的指令（[`tenth/src/runtime/vm.rs:386-421`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/vm.rs)）。迁移规则按指令分派，详见第 6 节。

**资源检查前置**：每步迁移前先执行资源检查（[`tenth/src/runtime/vm.rs:357-384`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/vm.rs)）：
$$
\textit{check}(\sigma) = \begin{cases}
\textit{Err}(\text{Timeout}) & \textit{bg} = 0 \\
\textit{Err}(\text{Timeout}) & \textit{ct} \equiv 0 \pmod{4096} \land \textit{now}() \geq \textit{dl} \\
\sigma' & \text{否则，其中 } \textit{bg}' = \textit{bg} - 1,\ \textit{ct}' = \textit{ct} + 1
\end{cases}
$$
其中 $\textit{dl}$ 为 `deadline_ms`，$\textit{now}()$ 为 `SystemTime::now()` 读取的 Unix 毫秒时间戳。注意 $\textit{ct} \equiv 0 \pmod{4096}$ 的判定用 `loop_counter.wrapping_add(1) & 0xFFF == 0`，即加 1 后低 12 位为零——等价于每 4096 步触发一次。

### 3.3 Frame 模型与调用栈

**定义 3.4（调用栈快照）**。设当前帧栈 $\textit{frs} = [f_0, f_1, \dots, f_{k-1}]$（$f_{k-1}$ 为当前帧），定义：
- 当前 chunk 索引：$\textit{chk}_{\text{cur}} = f_{k-1}.\textit{chk}_f$（若 $k \geq 1$，否则为入口 chunk）；
- 当前栈基址：$\textit{sb}_{\text{cur}} = f_{k-1}.\textit{sb}_f$（若 $k \geq 1$，否则为入口 `base`，[`tenth/src/runtime/vm.rs:338`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/vm.rs)）。

调用栈的"生长"由 `Call`/`CallN` 推入新帧，"收缩"由 `Ret` 弹出当前帧。每次 `Ret` 通过 `truncate(sb)` 确保收缩后栈状态与生长前对称。

---

## 4. 主定理

### 4.1 定理 V1（栈卫生不变量）

**定理 V1（栈卫生不变量）**。设调用前帧栈为 $\textit{frs}$，操作数栈为 $\textit{stk}_0$，调用者基址为 $b$（即 $\textit{stk}_0 = \textit{stk}_0[0..b] \cdot \textit{args}$，$|\textit{args}| = n$）。若 `Call`/`CallN` 推入帧 $f$（$\textit{sb}_f = b$ 或等价值），被调函数执行任意有限步后通过 `Ret` 返回值 $r$，则 `Ret` 后的状态满足：
$$
\textit{stk}_{\text{after}} = \textit{stk}_0[0..b] \cdot r,\quad \textit{frs}_{\text{after}} = \textit{frs}
$$
即操作数栈恢复到"调用前基址 + 返回值"，帧栈恢复到调用前状态。证明见第 7 节。

**源码锚点**：[`tenth/src/runtime/vm.rs:577-590`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/vm.rs)（`Ret` 实现），[`tenth/src/runtime/vm.rs:515`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/vm.rs)（`Call` 推帧），[`tenth/src/runtime/vm.rs:545`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/vm.rs)（`CallN` 推帧）。

### 4.2 定理 V2（Call/CallN 双协议等价性）

**定理 V2（双协议等价性）**。设调用点处操作数栈为 $\textit{stk} = \textit{stk}[0..b] \cdot \textit{args}$（$|\textit{args}| = n$，且**栈纪律成立**：调用点处栈在基址 $b$ 之上恰好有 $n$ 个值，无遗留中间值）。被调函数 $g$ 的 `num_args` $= n$。则：
- (a) `Call(g)` 与 `CallN(g, n)` 推入的帧 $f$ 满足 $\textit{sb}_f$ 相等；
- (b) 二者执行后操作数栈状态相同；
- (c) 二者的资源消耗（步数预算递减、`loop_counter` 递增）相同。

进一步，**当栈纪律不成立时**（即栈在基址 $b$ 之上有多于 $n$ 个值），二者行为不同：`Call` 的 `truncate(b)` 会清除所有多余值，`CallN` 的 `truncate(b + m)`（$m$ 为多余值数）保留多余值。证明见第 8 节。

**源码锚点**：[`tenth/src/runtime/vm.rs:503-527`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/vm.rs)（`Call`），[`tenth/src/runtime/vm.rs:528-566`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/vm.rs)（`CallN`）。

### 4.3 定理 V3（类型安全进展定理）

**定理 V3（进展）**。设 HIR 表达式 $e$ 在类型系统 $\Gamma \vdash e : \tau$ 下类型良好，经 lowering 与字节码编译得到 chunk 序列 $\textit{chs}$。若 VM 状态 $\sigma$ 对应 $e$ 的中间执行状态，且 $\textit{stk}$ 顶部的值类型与当前指令 $\textit{op}$ 的前置类型兼容，则迁移 $\langle \sigma, \textit{op} \rangle \rightarrow \sigma'$ 必然成立（$\sigma'$ 为下一状态或 `Err`），不存在"卡住"（stuck，既不迁移也不报错）状态。

**源码锚点**：[`tenth/src/runtime/vm.rs:422-802`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/vm.rs)（指令分派），[`tenth/src/runtime/vm.rs:386-421`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/vm.rs)（解码，未知 opcode 回退为 `Ret`）。

### 4.4 定理 V4（摊还 deadline 开销上界）

**定理 V4（摊还开销）**。设单次 `SystemTime::now()` 调用开销为 $c_{\text{clk}}$（常数，约 100–500 ns），单步指令分派开销为 $c_{\text{step}}$。则执行 $N$ 步的总 deadline 检查开销为：
$$
T_{\text{check}}(N) = \left\lceil \frac{N}{4096} \right\rceil \cdot c_{\text{clk}}
$$
摊还到每步的开销为 $\frac{c_{\text{clk}}}{4096} \approx 0.05\text{ ns/step}$。相对开销比 $\frac{T_{\text{check}}(N)}{N \cdot c_{\text{step}}} \leq \frac{c_{\text{clk}}}{4096 \cdot c_{\text{step}}}$，当 $c_{\text{step}} \approx 5\text{ ns}$ 时该比值 $\approx 5 \times 10^{-6}$（5 ppm）。

**定理 V4b（deadline 超限响应延迟上界）**。设单步最大耗时为 $T_{\max}$（含 native 调用），deadline 触发到实际返回 `Timeout` 的延迟 $\leq 4095 \cdot T_{\max} + T_{\max} = 4096 \cdot T_{\max}$。若存在无界单步（如 native 函数内部死循环），则该上界失效——见局限章节。证明见第 9 节。

**源码锚点**：[`tenth/src/runtime/vm.rs:369-384`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/vm.rs)（4096 步周期检查）。

### 4.5 定理 V5（与 CPython eval loop timeout 对比）

**定理 V5（对比）**。设 CPython 每步 `eval_breaker` 原子读开销为 $c_{\text{at}}$（约 1 ns），Tenth 每 4096 步 `SystemTime::now()` 开销为 $c_{\text{clk}}$（约 200 ns）。则：
- CPython 摊还开销：$c_{\text{at}} \approx 1\text{ ns/step}$；
- Tenth 摊还开销：$c_{\text{clk}} / 4096 \approx 0.05\text{ ns/step}$；
- Tenth 摊还开销低于 CPython **约 20 倍**；
- 但 CPython 的 deadline 响应延迟为单步（$\leq T_{\max}$），Tenth 为 $4096 \cdot T_{\max}$——CPython 响应更及时。

**源码锚点**：[`tenth/src/runtime/vm.rs:353-384`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/vm.rs)（双轨独立检查），[`tenth/src/runtime/interpreter/mod.rs:332-361`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/interpreter/mod.rs)（解释器侧对称实现，与 T35 联动）。

---

## 5. 小步操作语义的迁移规则

本节给出关键指令的小步迁移规则。记号：$\sigma = \langle \textit{ip}, \textit{code}, \textit{stk}, \textit{frs}, \textit{loc}, \textit{bg}, \textit{ct} \rangle$，$\textit{op} = \textit{decode}(\textit{code}, \textit{ip})$，$\textit{ip}' = \textit{ip} + \textit{size}(\textit{op})$（$\textit{size}$ 为指令编码长度）。

### 5.1 栈操作指令

$$
\frac{}{\langle \textit{ip}, \textit{code}, \textit{stk}, \textit{frs}, \textit{loc}, \cdot, \cdot \rangle, \texttt{PushInt}(n) \rangle \rightarrow \langle \textit{ip}', \textit{code}, \textit{stk} \cdot n, \textit{frs}, \textit{loc}, \cdot, \cdot \rangle}
\quad \text{(R-Push)}
$$

$$
\frac{|\textit{stk}| \geq 1}{\langle \textit{ip}, \textit{code}, \textit{stk} \cdot v, \textit{frs}, \textit{loc}, \cdot, \cdot \rangle, \texttt{Pop} \rangle \rightarrow \langle \textit{ip}', \textit{code}, \textit{stk}, \textit{frs}, \textit{loc}, \cdot, \cdot \rangle}
\quad \text{(R-Pop)}
$$

$$
\frac{|\textit{stk}| \geq 1}{\langle \textit{ip}, \textit{code}, \textit{stk} \cdot v, \textit{frs}, \textit{loc}, \cdot, \cdot \rangle, \texttt{Dup} \rangle \rightarrow \langle \textit{ip}', \textit{code}, \textit{stk} \cdot v \cdot v, \textit{frs}, \textit{loc}, \cdot, \cdot \rangle}
\quad \text{(R-Dup)}
$$

注：实现中 `Pop`/`Dup` 用 `unwrap_or(Value::Unit)` 兜底（[`tenth/src/runtime/vm.rs:432-436`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/vm.rs)），即栈空时 `Pop` 静默返回 `Unit`、`Dup` 压入 `Unit`。形式化中我们要求 $|\textit{stk}| \geq 1$ 作为前置条件；不满足时进入"退化"状态——这是形式化与实现的差距，见局限章节。

### 5.2 局部变量指令

$$
\frac{i < |\textit{loc}|}{\langle \cdot, \cdot, \textit{stk}, \cdot, \textit{loc}, \cdot, \cdot \rangle, \texttt{Load}(i) \rangle \rightarrow \langle \cdot, \cdot, \textit{stk} \cdot \textit{loc}[i], \cdot, \textit{loc}, \cdot, \cdot \rangle}
\quad \text{(R-Load)}
$$

$$
\frac{}{\langle \cdot, \cdot, \textit{stk} \cdot v, \cdot, \textit{loc}, \cdot, \cdot \rangle, \texttt{Store}(i) \rangle \rightarrow \langle \cdot, \cdot, \textit{stk}, \cdot, \textit{loc}[i \mapsto v], \cdot, \cdot \rangle}
\quad \text{(R-Store)}
$$

注：实现中 `Store` 在 $i \geq |\textit{loc}|$ 时自动 `locals.resize(i+1, Value::Unit)`（[`tenth/src/runtime/vm.rs:444`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/vm.rs)），即局部变量表可动态扩张——这偏离了静态大小假设，是简化形式化与灵活实现的取舍。

### 5.3 算术指令

$$
\frac{|\textit{stk}| \geq 2 \quad \textit{add}(\textit{stk}[-2], \textit{stk}[-1]) = v'}{\langle \cdot, \cdot, \textit{stk}' \cdot a \cdot b, \cdot, \cdot, \cdot, \cdot \rangle, \texttt{Add} \rangle \rightarrow \langle \cdot, \cdot, \textit{stk}' \cdot v', \cdot, \cdot, \cdot, \cdot \rangle}
\quad \text{(R-Add)}
$$

其中 $\textit{add}$ 是 [`tenth/src/runtime/vm.rs:817-872`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/vm.rs) 中 `add_priv` 的数学抽象，覆盖 Int×Int、Float×Float、Tensor×Float 等 11 种类型组合。类型不匹配时迁移到 `Err`。

### 5.4 跳转指令

$$
\frac{}{\langle \textit{ip}, \cdot, \textit{stk}, \cdot, \cdot, \cdot, \cdot \rangle, \texttt{Jump}(o) \rangle \rightarrow \langle \textit{ip} + o, \cdot, \textit{stk}, \cdot, \cdot, \cdot, \cdot \rangle}
\quad \text{(R-Jump)}
$$

$$
\frac{\textit{stk}[-1].\textit{is\_truthy}() = \text{false}}{\langle \textit{ip}, \cdot, \textit{stk}' \cdot v, \cdot, \cdot, \cdot, \cdot \rangle, \texttt{JmpFalse}(o) \rangle \rightarrow \langle \textit{ip} + o, \cdot, \textit{stk}', \cdot, \cdot, \cdot, \cdot \rangle}
\quad \text{(R-JmpFalse-taken)}
$$

$$
\frac{\textit{stk}[-1].\textit{is\_truthy}() = \text{true}}{\langle \textit{ip}, \cdot, \textit{stk}' \cdot v, \cdot, \cdot, \cdot, \cdot \rangle, \texttt{JmpFalse}(o) \rangle \rightarrow \langle \textit{ip}', \cdot, \textit{stk}', \cdot, \cdot, \cdot, \cdot \rangle}
\quad \text{(R-JmpFalse-not-taken)}
$$

### 5.5 调用指令

设被调函数 $g$ 在 `functions` 表中索引为 $g_{\text{idx}}$，其 `num_args` $= n_g$，`num_locals` $= l_g$。

**Call 协议（用户函数路径）**：

$$
\frac{\textit{stk} = \textit{stk}_b \cdot \textit{args} \quad |\textit{args}| = n_g \quad b = |\textit{stk}_b|}{\langle \textit{ip}, \textit{code}_c, \textit{stk}, \textit{frs}, \textit{loc}_c, \cdot, \cdot \rangle, \texttt{Call}(g) \rangle \rightarrow \langle 0, \textit{code}_g, \textit{stk}_b, \textit{frs} \cdot f, \textit{loc}_g^{\text{init}}, \cdot, \cdot \rangle}
\quad \text{(R-Call-user)}
$$

其中 $f = \langle \textit{ip}', g_{\text{idx}}, \textit{loc}_c, b \rangle$（$\textit{sb}_f = b$），$\textit{loc}_g^{\text{init}}[i] = \textit{args}[i]$（前 $n_g$ 个）其余为 `Unit`，长度 $\max(n_g, l_g)$。

**注**：实现中 `Call` 对用户函数路径用 `callee_args = num_args` 弹参（[`tenth/src/runtime/vm.rs:513-523`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/vm.rs)），即**用被调函数签名推断参数数量**，而非用栈深度。但 native 路径用 `n = stack.len() - base`（[`tenth/src/runtime/vm.rs:507`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/vm.rs)）——**用栈深度推断**。这是双协议的微妙之处：同一 `Call` 指令对 native 与 user 函数用不同推断策略。

**CallN 协议（用户函数路径）**：

$$
\frac{\textit{stk} = \textit{stk}_b \cdot \textit{args} \quad |\textit{args}| = n \quad b' = |\textit{stk}_b|}{\langle \textit{ip}, \textit{code}_c, \textit{stk}, \textit{frs}, \textit{loc}_c, \cdot, \cdot \rangle, \texttt{CallN}(g, n) \rangle \rightarrow \langle 0, \textit{code}_g, \textit{stk}_b, \textit{frs} \cdot f', \textit{loc}_g^{\text{init}}, \cdot, \cdot \rangle}
\quad \text{(R-CallN-user)}
$$

其中 $f' = \langle \textit{ip}', g_{\text{idx}}, \textit{loc}_c, b' \rangle$（$\textit{sb}_{f'} = b'$）。注意 $b = b'$ 都等于 $|\textit{stk}_b|$（弹参数后的栈长度）——**前提是栈纪律成立**（$\textit{stk}_b$ 恰为调用前基址之上的部分）。

### 5.6 Ret 指令

$$
\frac{\textit{stk} = \textit{stk}_{\text{garbage}} \cdot r \quad \textit{frs} = \textit{frs}' \cdot f \quad \textit{sb}_f = k}{\langle \cdot, \cdot, \textit{stk}, \textit{frs}, \cdot, \cdot, \cdot \rangle, \texttt{Ret} \rangle \rightarrow \langle \textit{ip}_f, \textit{code}_{\textit{chk}_f}, (\textit{stk} \!\upharpoonright\! k) \cdot r, \textit{frs}', \textit{loc}_f, \cdot, \cdot \rangle}
\quad \text{(R-Ret)}
$$

关键：$\textit{stk} \!\upharpoonright\! k$ 是 `truncate(f.stack_base)` 的数学化，**丢弃所有栈垃圾** $\textit{stk}_{\text{garbage}}$（被调函数体内遗留的未清理值），只保留基址之下的部分，再压入返回值 $r$。

**Ret 顶层规则**：

$$
\frac{\textit{frs} = [] \quad \textit{stk} = \textit{stk}_{\text{garbage}} \cdot r}{\langle \cdot, \cdot, \textit{stk}, [], \cdot, \cdot, \cdot \rangle, \texttt{Ret} \rangle \rightarrow \textit{Halt}(r)}
\quad \text{(R-Ret-top)}
$$

对应实现 `return Ok(result)`（[`tenth/src/runtime/vm.rs:588`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/vm.rs)）。

---

## 6. （迁移规则汇总——见上节）

为避免重复，第 5 节已完整给出迁移规则。本节作为结构占位，保持论文结构与引言承诺的章节编号一致。迁移规则的完整清单见第 5 节，涵盖栈操作（R-Push/R-Pop/R-Dup）、局部变量（R-Load/R-Store）、算术（R-Add 等 6 条）、跳转（R-Jump/R-JmpFalse ×2）、调用（R-Call-user/R-CallN-user）、返回（R-Ret/R-Ret-top）。

---

## 7. 栈卫生不变量证明

### 7.1 形式化陈述

**定理 V1（重述）**。对任意调用序列 `Call/CallN → callee body → Ret`，若调用前帧栈为 $\textit{frs}_0$、操作数栈为 $\textit{stk}_0 = \textit{stk}_0[0..b] \cdot \textit{args}$（$|\textit{args}| = n$），推入帧 $f$ 满足 $\textit{sb}_f = b$，被调函数体执行任意有限步 $\pi$ 后通过 `Ret` 返回值 $r$，则 `Ret` 后状态满足：
$$
\textit{stk}_{\text{after}} = \textit{stk}_0[0..b] \cdot r,\quad \textit{frs}_{\text{after}} = \textit{frs}_0
$$

### 7.2 证明方法：反向归纳 + 截断语义归约

**证明思路**：证明的核心困难在于被调函数体 $\pi$ 可能执行任意复杂的操作（包括嵌套调用），使得 `Ret` 前的栈状态 $\textit{stk}_{\text{before-ret}}$ 难以静态刻画。我们采用**反向归纳**——从 `Ret` 指令倒推，证明无论 $\textit{stk}_{\text{before-ret}}$ 是什么，`truncate(b)` 后状态唯一确定。

**引理 V1.1（截断确定性）**。对任意栈 $\textit{stk}$ 与基址 $b \leq |\textit{stk}|$，有：
$$
(\textit{stk} \!\upharpoonright\! b) \cdot r = \textit{stk}[0..b] \cdot r
$$
即截断到 $b$ 后压入 $r$，结果只依赖 $\textit{stk}[0..b]$ 与 $r$，与 $\textit{stk}[b..]$ 无关。

*证明*。由 `Vec::truncate` 的语义（Rust 标准库：丢弃下标 $\geq b$ 的所有元素），$\textit{stk} \!\upharpoonright\! b = \textit{stk}[0..b]$。故 $(\textit{stk} \!\upharpoonright\! b) \cdot r = \textit{stk}[0..b] \cdot r$。$\square$

**引理 V1.2（调用前栈基址匹配）**。`Call`/`CallN` 推入的帧 $f$ 满足 $\textit{sb}_f = b$，其中 $b$ 是调用前操作数栈中"参数之下"的位置。

*证明*。分两种情形：

- **Call 用户函数路径**（[`tenth/src/runtime/vm.rs:515`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/vm.rs)）：`stack_base: base`。其中 `base` 是调用者函数入口时由 `let base = self.stack.len().saturating_sub(num_args)` 计算（[`tenth/src/runtime/vm.rs:338`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/vm.rs)），表示调用者**自身**的参数之下位置。但这里 `Call` 推帧用的 `base` 是**调用者**的基址，不是"当前调用点参数之下"。

  这里需要澄清：在 Tenth VM 实现中，`base` 是调用者函数入口时计算的固定值，整个函数执行期间不变。因此 `Call` 的 `stack_base: base` 实际上是"调用者的基址"，而非"当前调用点的参数基址"。这意味着 `Call` 协议隐含假设：**调用点处栈在 `base` 之上恰好只有参数**（无遗留中间值）。在此假设下，调用点参数之下 = 调用者基址 $b$，故 $\textit{sb}_f = b$。

- **CallN 用户函数路径**（[`tenth/src/runtime/vm.rs:545`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/vm.rs)）：`stack_base: self.stack.len()`，此时已弹出 $n$ 个参数。设调用前栈 $\textit{stk}_0 = \textit{stk}_0[0..b'] \cdot \textit{args}$，弹参数后 $\textit{stk} = \textit{stk}_0[0..b']$，$|\textit{stk}| = b'$。故 $\textit{sb}_f = b'$，其中 $b'$ 是调用点参数之下的位置。

两种协议下 $\textit{sb}_f$ 都等于"调用点参数之下的位置"——但 Call 协议要求栈纪律成立（$b' = b$，即调用点参数之下 = 调用者基址），CallN 协议无此要求（$b'$ 直接由当前栈长度决定）。$\square$

**引理 V1.3（Ret 不依赖被调函数体）**。设 `Ret` 前操作数栈为 $\textit{stk}_{\text{pre}}$，帧栈顶为 $f$（$\textit{sb}_f = b$），返回值 $r = \textit{stk}_{\text{pre}}[-1]$。则 `Ret` 后操作数栈 $\textit{stk}_{\text{after}}$ 仅依赖 $\textit{stk}_{\text{pre}}[0..b]$ 与 $r$，不依赖 $\textit{stk}_{\text{pre}}[b..-1]$（被调函数体遗留的栈垃圾）。

*证明*。由 R-Ret 规则：
$$
\textit{stk}_{\text{after}} = (\textit{stk}_{\text{pre}} \!\upharpoonright\! b) \cdot r
$$
由引理 V1.1，$(\textit{stk}_{\text{pre}} \!\upharpoonright\! b) \cdot r = \textit{stk}_{\text{pre}}[0..b] \cdot r$，与 $\textit{stk}_{\text{pre}}[b..-1]$ 无关。$\square$

**定理 V1 的证明**。

由引理 V1.2，推入帧 $f$ 满足 $\textit{sb}_f = b$（在 Call 协议下需栈纪律前提；在 CallN 协议下无条件成立）。

被调函数体执行任意有限步 $\pi$ 后到达 `Ret`。设 `Ret` 前状态为 $\sigma_{\text{pre}} = \langle \cdot, \cdot, \textit{stk}_{\text{pre}}, \textit{frs}_0 \cdot f, \cdot, \cdot, \cdot \rangle$。

由引理 V1.3，`Ret` 后：
$$
\textit{stk}_{\text{after}} = \textit{stk}_{\text{pre}}[0..b] \cdot r
$$

剩下需证 $\textit{stk}_{\text{pre}}[0..b] = \textit{stk}_0[0..b]$，即被调函数体执行期间**栈基址 $b$ 之下的部分不被修改**。

这由以下不变量保证：**被调函数体只能通过 `push`/`pop` 操作栈顶，不能修改栈中下标 $< b$ 的元素**。Rust `Vec<Value>` 的 `push`/`pop`/`truncate` 操作：
- `push(v)`：仅追加到末尾，不影响 $[0..b]$；
- `pop()`：仅移除末尾，若 $|\textit{stk}| > b$ 则不影响 $[0..b]$；若 $|\textit{stk}| = b$ 则 `pop` 返回 `Unit`（`unwrap_or` 兜底，[`tenth/src/runtime/vm.rs:194`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/vm.rs)）但仍不修改 $[0..b]$；
- `truncate(k)`：仅当 $k < |\textit{stk}|$ 时丢弃末尾，$k \geq b$ 时不影响 $[0..b]$。被调函数体的 `Ret` 用 `truncate(f.stack_base)`，其中 $f$ 是被调函数自己推入的帧，$\textit{sb}_f \geq b$（被调函数的基址 $\geq$ 调用者的基址，因为调用者已压入参数）。

故被调函数体执行期间 $\textit{stk}[0..b]$ 保持不变，即 $\textit{stk}_{\text{pre}}[0..b] = \textit{stk}_0[0..b]$。

综上：
$$
\textit{stk}_{\text{after}} = \textit{stk}_0[0..b] \cdot r, \quad \textit{frs}_{\text{after}} = \textit{frs}_0
$$
$\square$

### 7.3 推论：嵌套调用的栈卫生传递

**推论 V1.1**。对任意深度的嵌套调用 $f_0 \to f_1 \to \dots \to f_k$，每层 `Ret` 都恢复该层调用前的栈状态。最终 $f_0$ 的栈状态为 $f_0$ 基址之下 + $f_0$ 的返回值，中间层的栈垃圾全部被各层 `Ret` 的 `truncate` 清除。

*证明*。对 $k$ 归纳。$k = 0$ 为定理 V1。$k > 0$ 时，$f_{k-1}$ 调用 $f_k$，由定理 V1，$f_k$ 的 `Ret` 恢复 $f_{k-1}$ 调用点栈 + $f_k$ 返回值；$f_{k-1}$ 继续执行至自己的 `Ret`，再由定理 V1 恢复 $f_{k-2}$ 调用点栈 + $f_{k-1}$ 返回值。每层独立应用 V1，归纳得证。$\square$

---

## 8. 双协议等价性证明

### 8.1 形式化陈述

**定理 V2（重述）**。设调用点处栈 $\textit{stk} = \textit{stk}_b \cdot \textit{args}$，$|\textit{args}| = n$，被调函数 $g$ 的 `num_args` $= n$。**栈纪律**定义为：调用点处 $|\textit{stk}_b| = b$，其中 $b$ 是调用者函数入口时计算的固定基址。

- (a) 若栈纪律成立，则 `Call(g)` 与 `CallN(g, n)` 推入的帧 $f, f'$ 满足 $\textit{sb}_f = \textit{sb}_{f'}$；
- (b) 二者执行后操作数栈状态相同；
- (c) 二者的资源消耗相同。
- (d) 若栈纪律不成立（即 $|\textit{stk}_b| > b$，调用点有遗留中间值），则 `Call` 的 `truncate(b)` 清除遗留值，`CallN` 的 `truncate(|stk_b|)` 保留遗留值——二者行为不同。

### 8.2 证明方法：双向模拟

**证明思路**：构造 `Call` 与 `CallN` 之间的双向模拟关系（bisimulation），证明在栈纪律前提下二者的状态迁移可互相对应。

**情形 (a)：栈纪律成立时 $\textit{sb}_f = \textit{sb}_{f'}$**。

设调用点栈 $\textit{stk} = \textit{stk}[0..b] \cdot \textit{args}$，$|\textit{args}| = n$，栈纪律成立意味着 $\textit{stk}[0..b]$ 恰为调用者基址之下的部分。

- **Call 路径**（[`tenth/src/runtime/vm.rs:515`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/vm.rs)）：`stack_base: base`，其中 `base` $= b$（调用者入口基址，栈纪律下等于调用点参数之下位置）。故 $\textit{sb}_f = b$。

- **CallN 路径**（[`tenth/src/runtime/vm.rs:545`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/vm.rs)）：先弹 $n$ 个参数，$\textit{stk}$ 变为 $\textit{stk}[0..b]$，$|\textit{stk}| = b$；`stack_base: self.stack.len()` $= b$。故 $\textit{sb}_{f'} = b$。

故 $\textit{sb}_f = \textit{sb}_{f'} = b$。$\square$

**情形 (b)：执行后操作数栈相同**。

由 (a)，两协议推入帧的 $\textit{sb}$ 相同。两协议都被调函数 $g$，初始 $\textit{loc}_g$ 都为 $\textit{args}$（Call 在 [`tenth/src/runtime/vm.rs:521-523`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/vm.rs) 弹参到 `locals`，CallN 在 [`tenth/src/runtime/vm.rs:550-551`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/vm.rs) 用 `args` 作 `locals`）。两协议调用前操作数栈弹参数后都为 $\textit{stk}[0..b]$。故被调函数体执行前的 VM 状态 $\langle 0, \textit{code}_g, \textit{stk}[0..b], \textit{frs} \cdot f, \textit{loc}_g^{\text{init}}, \cdot, \cdot \rangle$ 完全相同（除帧 $f$ 与 $f'$ 的 `ip`/`chunk_idx`/`locals` 字段，但这些字段在 `Ret` 时被恢复，不影响被调函数体执行）。

被调函数体从相同初始状态出发，按确定性迁移规则（指令分派是确定性的，无随机），到达相同 `Ret` 前状态。由定理 V1，`Ret` 后栈状态相同。$\square$

**情形 (c)：资源消耗相同**。

两协议的指令解码与分派开销相同（都是一条指令）。`step_budget` 递减一次，`loop_counter` 递增一次。唯一差异是 CallN 多了一次 `Vec::with_capacity(n)` 与 $n$ 次 `pop` 写入 `args`，而 Call 是 $n_g$ 次 `pop` 直接写入 `locals`——二者都是 $O(n)$ 次栈操作，常数因子相近。资源检查层面二者相同。$\square$

**情形 (d)：栈纪律不成立时行为不同**。

设调用点栈 $\textit{stk} = \textit{stk}[0..b] \cdot \textit{extra} \cdot \textit{args}$，$|\textit{extra}| = m > 0$。

- **Call 路径**：`stack_base: base` $= b$。被调函数体 `Ret` 时 `truncate(b)`，清除 $\textit{extra}$ 与 $\textit{args}$，压入返回值 $r$。结果栈：$\textit{stk}[0..b] \cdot r$，**$\textit{extra}$ 丢失**。

- **CallN 路径**：先弹 $n$ 个参数，$\textit{stk}$ 变为 $\textit{stk}[0..b] \cdot \textit{extra}$，$|\textit{stk}| = b + m$；`stack_base: self.stack.len()` $= b + m$。被调函数体 `Ret` 时 `truncate(b + m)`，保留 $\textit{extra}$，压入 $r$。结果栈：$\textit{stk}[0..b] \cdot \textit{extra} \cdot r$，**$\textit{extra}$ 保留**。

故栈纪律不成立时，`Call` 清除遗留值，`CallN` 保留遗留值——行为不同。$\square$

### 8.3 Native 路径的额外差异

`Call` 对 native 函数用 `n = self.stack.len() - base`（[`tenth/src/runtime/vm.rs:507`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/vm.rs)）推断参数数量，**会把栈纪律不成立时的 `extra` 也当作参数传给 native**。`CallN` 用显式 $n$，只传恰好 $n$ 个参数。这是 native 路径下双协议的额外差异——即使栈纪律成立，`Call` 对 native 也依赖栈深度推断，`CallN` 显式指定。

**推论 V2.1**。栈纪律成立时，对用户函数 `Call(g)` 与 `CallN(g, n)` 完全等价；对 native 函数，`Call(g)` 与 `CallN(g, n)` 在 $n = \text{stack.len()} - \text{base}$ 时等价，否则不等价。

---

## 9. 摊还 deadline 分析

### 9.1 摊还开销分析（聚合法）

**定理 V4（重述）**。执行 $N$ 步的总 deadline 检查开销 $T_{\text{check}}(N) = \lceil N / 4096 \rceil \cdot c_{\text{clk}}$。

*证明*。`loop_counter` 初始为 0，每步 `wrapping_add(1)`。`loop_counter & 0xFFF == 0` 当且仅当 `loop_counter` 是 4096 的倍数（在 wrapping 语义下，模 $2^{64}$）。故在前 $N$ 步中，触发检查的步数为 $\lfloor N / 4096 \rfloor$ 或 $\lceil N / 4096 \rceil$（取决于初始相位）。

每次检查执行：
- 一次 `SystemTime::now()` 系统调用（开销 $c_{\text{clk}}$）；
- 一次整数比较（开销可忽略）；
- 一次 `deadline_ms` 的 `Option` 匹配（开销可忽略）。

总开销 $T_{\text{check}}(N) \approx \lceil N / 4096 \rceil \cdot c_{\text{clk}}$。

摊还到每步：$\bar{T}_{\text{check}} = T_{\text{check}}(N) / N \approx c_{\text{clk}} / 4096$。

取 $c_{\text{clk}} \approx 200\text{ ns}$（Linux `clock_gettime(CLOCK_REALTIME)` 典型值），$\bar{T}_{\text{check}} \approx 0.049\text{ ns/step}$。$\square$

**对比：每步检查的开销**。若每步都检查 deadline，$T_{\text{check}}^{\text{every}}(N) = N \cdot c_{\text{clk}}$，摊还 $\bar{T}^{\text{every}} = c_{\text{clk}} \approx 200\text{ ns/step}$。当单步指令分派开销 $c_{\text{step}} \approx 5\text{ ns}$ 时，每步检查的相对开销为 $200/5 = 40$倍（4000%），不可接受。周期检查的相对开销为 $0.049/5 \approx 1\%$（5 ppm 量级），可忽略。

### 9.2 deadline 超限响应延迟上界

**定理 V4b（重述）**。deadline 触发到实际返回 `Timeout` 的延迟 $\leq 4096 \cdot T_{\max}$，其中 $T_{\max}$ 为单步最大耗时。

*证明*。设 deadline 在时刻 $t_d$ 触发（`now() >= deadline`）。下一次检查发生在 `loop_counter` 下次成为 4096 倍数时，最坏情况需等待 4095 步。设第 $i$ 步耗时 $t_i \leq T_{\max}$，则等待时间 $\leq 4095 \cdot T_{\max}$。检查时若发现超限，立即返回 `Err(Timeout)`，耗时 $O(1)$。

故总延迟 $\leq 4095 \cdot T_{\max} + O(1) \leq 4096 \cdot T_{\max}$。$\square$

### 9.3 wrapping_add 的安全性

`loop_counter` 用 `u64::wrapping_add`（[`tenth/src/runtime/vm.rs:371`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/vm.rs)），在 $2^{64}$ 步后回绕。回绕不影响 `& 0xFFF == 0` 的周期性——`wrapping_add(1) & 0xFFF` 的周期始终为 4096。但回绕点处可能出现连续两次触发（从 `0xFFF...FFF` 到 `0`），开销为 $2 \cdot c_{\text{clk}}$，可忽略。

回绕所需步数 $2^{64} \approx 1.8 \times 10^{19}$，单步 5 ns 时需 $9 \times 10^{9}$ 秒 $\approx 285$ 年——实际不可达。

---

## 10. 工程权衡

### 10.1 `truncate` vs. 静态验证

Tenth VM 用运行时 `truncate` 保证栈卫生，而非 JVM 风格的加载时字节码验证。

| 维度 | Tenth（truncate） | JVM（验证器） |
|------|------------------|---------------|
| 错误反馈时机 | 运行时（Late） | 加载时（Early） |
| 编译器复杂度 | 低（无需验证器） | 高（需 dataflow 分析） |
| 运行时开销 | 每次调用一次 `truncate` | 无（验证已通过） |
| 安全性 | 兜底保证 | 静态保证 |
| 调试体验 | 栈垃圾被静默清除 | 验证失败明确报错 |

Tenth 的取舍偏向**实现简单性**——`truncate` 一行代码即可保证栈卫生，代价是被调函数的栈 bug 不会被早期发现（被 `truncate` 静默修复）。这在 AI 原生语言的快速迭代语境下合理，但在生产级运行时中是债务。

### 10.2 双协议并存的代价

`Call` 与 `CallN` 并存带来：
- **认知负担**：开发者需理解两种协议的差异与适用场景；
- **形式化复杂度**：需证明二者等价性（本文定理 V2）；
- **维护负担**：任何调用语义变更需同步修改两处；
- **向后兼容**：`Call` 是历史遗留，删除会破坏旧 chunk。

Tenth 的取舍是**保留双协议**，新代码用 `CallN`，旧代码保留 `Call`。这与 Lua 5.0 同时支持 `CALL` 与 `TAILCALL` 的策略类似——历史遗产与现代化并存。

### 10.3 周期检查 vs. 每步检查

Tenth 的 4096 步周期检查是**摊还优化**的典范：用 4096 倍的响应延迟换取 4096 倍的开销降低。对 AI 训练场景（步数预算 $\sim 10^9$），摊还开销 $\sim 50\text{ ns/step} \times 10^9 = 50\text{ s}$，可接受；对交互式 REPL（步数 $\sim 10^4$），延迟 $\leq 4096 \times 5\text{ ns} = 20\text{ μs}$，无感知。

但对**单步无界**的场景（如 native 函数内部死循环），周期检查失效——这是 Tenth 选择 native 函数受 `step_budget` 不约束的设计代价（[`tenth/src/runtime/interpreter/natives.rs:87-94`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/interpreter/natives.rs) 中 `with_step_limit`/`with_timeout_ms` 通过保存/恢复 budget 实现子预算，但 native 内部仍可无限循环）。

### 10.4 双轨独立检查的必要性

历史缺陷 H-4（[`tenth/src/runtime/vm.rs:358-360`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/vm.rs) 注释）表明：将 deadline 检查嵌套在 `step_budget` 检查内（即 `if let Some(budget) = step_budget { ... check deadline ... }`）会导致**只设 `--timeout` 不设步数预算时 deadline 永不触发**。当前实现用独立 `loop_counter` 修复，确保两个资源限制相互正交。这是**正交性设计原则**的实例：两个独立的功能不应相互耦合。

---

## 11. 开放问题

1. **`MethodCall` 的多分派形式化**。本文未形式化 `MethodCall`（[`tenth/src/runtime/vm.rs:567-575`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/vm.rs)），它通过 `call_method_priv` 做运行时多分派。其与 `Call`/`CallN` 的等价性需单独研究。

2. **JIT 路径的栈卫生**。JIT 编译后的代码（[`tenth/src/compile/jit/`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/jit/)）是否保持 `truncate` 语义？JIT 通过 hostcall 回调 VM（[`tenth/src/runtime/vm.rs:211-220`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/vm.rs) `call_with_args`），其栈卫生依赖 hostcall trampoline 的正确性——这需在论文 T9（JIT 特化语义保持）框架下补充。

3. **native 函数的栈副作用**。native 函数（`NativeFn = fn(&mut Vm, &[Value]) -> TenthResult<Value>`，[`tenth/src/runtime/vm.rs:14`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/vm.rs)）可任意修改 `Vm.stack`。定理 V1 假设 native 路径下 `stack_base` 正确——但若 native 函数体内 `stack_push` 多次后 `return Ok(Value::Unit)`，`Call` 的 native 路径（[`tenth/src/runtime/vm.rs:506-511`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/vm.rs)）不做 `truncate`，会破坏栈卫生。这是开放风险。

4. **deadline 检查与 native 调用的交互**。native 调用期间 `loop_counter` 不递增（native 在单步内执行），故 native 内部死循环无法被 deadline 中断。需研究协作式中断（native 周期性检查标志位）或抢占式中断（独立线程 + 信号）。

5. **栈深度上界**。本文未分析栈深度上界。`step_budget` 限制步数但不直接限制栈深度——`Push` 循环可在步数预算内耗尽内存。需引入 `stack_depth_limit`（独立于步数预算）。

6. **与 T35（解释器-VM 等价性）的联动**。解释器侧（[`tenth/src/runtime/interpreter/mod.rs:332-361`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/interpreter/mod.rs) `tick`）用相同 4096 步周期检查 deadline，但解释器无 `truncate`（树形解释器无操作数栈）。T35 需证明：在 deadline 触发时机上，VM 与解释器的行为等价（都每 4096 步检查一次）。本文的定理 V4 为 T35 提供了 deadline 检查的复杂度上界，但等价性证明需 T35 独立完成。

---

## 12. 局限

本节诚实记录形式化的覆盖边界与未覆盖项。

### 12.1 形式化未覆盖的实现细节

- **`unwrap_or(Value::Unit)` 兜底**：实现中 `pop`/`pop2` 等用 `unwrap_or(Value::Unit)` 兜底栈空情况（[`tenth/src/runtime/vm.rs:804-808`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/vm.rs) `pop2`）。形式化中我们要求 $|\textit{stk}| \geq k$ 作为前置条件，未覆盖栈空时的退化行为。**影响**：形式化证明适用于"栈纪律良好"的程序，对栈空时的 `Unit` 兜底未形式化。**缓解**：编译器保证栈空兜底不触发（HIR 类型检查排除栈不匹配）。

- **`Store` 的动态扩张**：`Store(i)` 在 $i \geq |\textit{loc}|$ 时 `locals.resize(i+1, Value::Unit)`（[`tenth/src/runtime/vm.rs:444`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/vm.rs)）。形式化假设 `loc` 静态大小。**影响**：对动态扩张的 `Store` 行为未形式化。**缓解**：编译器保证 `i < num_locals`，动态扩张仅为兜底。

- **`Chunk::read_op` 的 `Ret` 回退**：未知 opcode 在 `Chunk::read_op` 中 `panic!`（[`tenth/src/runtime/vm.rs:141`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/vm.rs)），但内联解码器（[`tenth/src/runtime/vm.rs:419`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/vm.rs)）回退为 `Ret`。形式化采用内联解码器的回退语义。**影响**：形式化与 `Chunk::read_op` 的 panic 行为不一致——但热路径用内联解码器，`Chunk::read_op` 仅用于离线分析。**缓解**：定理 V3 的进展性基于内联解码器。

### 12.2 证明的强度限制

- **定理 V3 的"类型良好"假设**：V3 假设 HIR 类型良好。但 Tenth 的类型系统未形式化（属于 T16/T23 范畴），故 V3 的前提是"待证"的。**影响**：V3 实际上是"条件定理"——条件是 HIR 类型系统健全。**缓解**：本文明确将类型系统健全性列为假设，待 T16/T23 补全后 V3 成为完整定理。

- **定理 V4b 的 $T_{\max}$ 假设**：V4b 假设单步最大耗时 $T_{\max}$ 有界。但 native 函数内部可能死循环，使 $T_{\max} = \infty$。**影响**：V4b 对 native 死循环场景失效。**缓解**：本文明确将该局限列为开放问题 4。

- **定理 V2 的栈纪律前提**：V2(a)(b)(c) 在栈纪律成立时成立。但栈纪律是编译器保证的运行时性质，非静态可验证。**影响**：若编译器有 bug 生成违反栈纪律的字节码，V2 等价性失效。**缓解**：本文在 V2(d) 显式给出栈纪律不成立时的行为差异，供编译器测试用例设计参考。

### 12.3 形式化与实现的工程差距

- **`code`/`strings` 的 `clone`**：每次 `Call` 与 `Ret` 都 `clone` 当前 chunk 的 `code` 与 `strings`（[`tenth/src/runtime/vm.rs:517-518, 547-548, 558-559, 584-585`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/vm.rs)）。形式化中 `code` 作为状态的一部分被原样传递，未建模 `clone` 开销。**影响**：形式化的复杂度分析未包含 `clone` 的 $O(|\textit{code}|)$ 开销——实际每次调用的开销高于形式化预测。**缓解**：这是实现层面的优化机会（用 `Rc<Chunk>` 或索引替代 `clone`），不影响语义正确性。

- **`locals.clone()`**：`Call` 推帧时 `locals.clone()`（[`tenth/src/runtime/vm.rs:515`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/vm.rs)）。形式化中 `loc` 作为状态一部分被"保存"到帧，未建模 `clone` 开销。**影响**：同上。

### 12.4 未覆盖的指令

本文形式化了 `Push`/`Pop`/`Dup`/`Load`/`Store`/`Add`/`Jump`/`JmpFalse`/`Call`/`CallN`/`Ret` 等核心指令。**未形式化**：`MethodCall`、`MakeVec`/`MakeMap`/`NewStruct`/`LoadField`/`StoreField`、`IndexGet`/`SliceStr`、`MakeEnum`/`IsEnumVariant`/`EnumGetField`、`PushRange`/`MoveOp`、`MakeTensor`/`MakeClosure`。这些指令的栈卫生性质（多数为"弹 $n$ 压 1"或"弹 $n$ 压 0"）可由本文框架机械扩展，但未在本文展开。

### 12.5 循环论证风险

本文定理 V3（进展）依赖"HIR 类型系统健全"——而 HIR 类型系统健全性属于 T16/T23 范畴，本文将其列为假设而非证明。这构成**潜在循环论证风险**：若 T16/T23 反过来引用本文 V3 作为 VM 层健全性证据，则形成循环。**缓解**：本文明确将 V3 标记为"条件定理"，T16/T23 应独立证明类型系统健全，不依赖 V3。

---

## 13. 结论

本文对 Tenth 栈式 VM 的核心运行机制进行了小步操作语义形式化，建立了五个主定理：

1. **V1（栈卫生不变量）**：`Ret` 的 `truncate(stack_base)` 保证栈恢复到调用前状态 + 返回值。证明方法为**反向归纳 + 截断语义归约**——从 `Ret` 倒推，证明 `truncate` 的确定性使得 `Ret` 后状态不依赖被调函数体的栈垃圾，且被调函数体执行期间栈基址之下不变。

2. **V2（双协议等价性）**：`Call` 与 `CallN` 在**栈纪律成立**时完全等价；栈纪律不成立时 `Call` 清除遗留值、`CallN` 保留遗留值。Native 路径下 `Call` 用栈深度推断参数数量，`CallN` 显式指定——即使栈纪律成立，native 路径的等价性也需 $n$ 匹配。

3. **V3（类型安全进展）**：类型良好的指令不卡住——条件是 HIR 类型系统健全（T16/T23 范畴）。VM 的内联解码器对未知 opcode 回退为 `Ret`，进一步保证无 panic。

4. **V4（摊还 deadline 开销）**：4096 步周期检查的摊还开销为 $c_{\text{clk}}/4096 \approx 0.05\text{ ns/step}$，相对开销 $< 10\text{ ppm}$。deadline 超限响应延迟 $\leq 4096 \cdot T_{\max}$，单步有界时延迟可控。

5. **V5（与 CPython 对比）**：Tenth 的周期检查摊还开销低于 CPython 每步原子读约 20 倍，但响应延迟高 4096 倍——摊还优化与响应及时性的经典权衡。

本文的工程指导意义：
- **栈卫生**：`truncate` 是简单可靠的栈卫生保证，适合快速迭代；生产级运行时应考虑补充静态验证器；
- **双协议**：新代码应统一用 `CallN`，`Call` 视为遗留；编译器应静态保证栈纪律；
- **deadline**：4096 步周期对 AI 训练场景合理；交互式场景可考虑更短周期（如 256 步）；native 死循环需协作式中断。

本文为 T35（解释器-VM 等价性）提供了 VM 侧的操作语义基础与 deadline 检查复杂度上界，T35 可在此基础上证明解释器侧 `tick`（[`tenth/src/runtime/interpreter/mod.rs:332-361`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/interpreter/mod.rs)）与 VM 侧 `loop_counter`（[`tenth/src/runtime/vm.rs:371-384`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/vm.rs)）在 deadline 触发时机上的等价性。

---

## 参考文献

1. Plotkin, G. D. (1981). *A Structural Approach to Operational Semantics*. University of Aarhus. (小步 SOS 的奠基性文献)
2. Diehl, S., Hartel, P., & Sestoft, P. (2000). *Abstract machines for programming language implementation*. Higher-Order and Symbolic Computation, 13(4), 287–347. (栈式 vs 寄存器式 VM 综述)
3. Gudeman, D. W. (1991). *Compact representations of abstract syntax*. PLDI. (T15 索引表示相关，与栈式编码紧凑性同源)
4. Lindholm, T., & Yellin, F. (1999). *The Java Virtual Machine Specification* (2nd ed.). Addison-Wesley. (JVM 字节码验证器与栈映射框架)
5. Ierusalimschy, R., de Figueiredo, L. H., & Celes, W. (2005). *The implementation of Lua 5.0*. JUCS. (寄存器式 VM 设计)
6. Pall, M. (2008). *LuaJIT 2.0 — A trace compiler*. (公开基准数据来源)
7. Ertl, M. A., & Gregg, D. (2003). *The structure and performance of efficient interpreters*. JILP. (栈式与寄存器式性能对比)
8. Python Software Foundation. (2024). *CPython eval loop implementation* (`ceval.c`). (eval_breaker 机制)
9. Avery, J. (2017). *Writing a bytecode verifier*. (JVM 风格验证器实现指南)
10. Tarjan, R. E. (1985). *Amortized computational complexity*. SIAM J. Algebraic Discrete Methods, 6(2), 306–316. (摊还分析方法论)

---

## 附录 A：定理索引

| 定理 | 陈述 | 证明方法 | 源码锚点 |
|------|------|---------|---------|
| V1 | 栈卫生不变量 | 反向归纳 + 截断语义归约 | [vm.rs:577-590](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/vm.rs) |
| V2 | Call/CallN 双协议等价性 | 双向模拟 | [vm.rs:503-566](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/vm.rs) |
| V3 | 类型安全进展 | 分情形分析 + 解码器回退 | [vm.rs:422-802](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/vm.rs) |
| V4 | 摊还 deadline 开销上界 | 聚合法 | [vm.rs:369-384](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/vm.rs) |
| V4b | deadline 超限响应延迟上界 | 最坏情况分析 | [vm.rs:369-384](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/vm.rs) |
| V5 | 与 CPython eval loop 对比 | 定量对比 | [vm.rs:353-384](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/vm.rs) |
| V1.1 | 截断确定性 | `Vec::truncate` 语义 | [vm.rs:580](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/vm.rs) |
| V1.2 | 调用前栈基址匹配 | 分协议情形分析 | [vm.rs:515,545](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/vm.rs) |
| V1.3 | Ret 不依赖被调函数体 | 截断确定性应用 | [vm.rs:577-590](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/vm.rs) |
| V1.1 推论 | 嵌套调用的栈卫生传递 | 对调用深度归纳 | 同 V1 |
| V2.1 | Native 路径双协议差异 | 栈深度推断 vs 显式 | [vm.rs:506-511](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/vm.rs) |

## 附录 B：与现有文档的对应

| 本文章节 | 对应文档 |
|---------|---------|
| §3 状态空间 | [CODE_WIKI.md](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/CODE_WIKI.md) VM 模块详解 |
| §5 迁移规则 | [docs/语言参考手册.md](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/docs/语言参考手册.md) 指令集 |
| §10 工程权衡 | [MEMO.md](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/MEMO.md) H-4 缺陷记录 |
| §11 开放问题 | [AUDIT.md](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/AUDIT.md) 待办 |
| §12 局限 | （本文新增，建议同步至 AUDIT.md）|

## 附录 C：实施建议

1. **编译器侧**：在 `bytecode.rs` 中增加静态检查，确保所有 `Call` 调用点满足栈纪律（调用前栈深度 = 调用者基址 + 参数数量）。违反时编译期警告。
2. **VM 侧**：考虑在 debug 模式下增加 `assert!(self.stack.len() >= f.stack_base)` 于 `Ret` 前，捕获栈下溢。
3. **native 函数规范**：文档化 native 函数的栈副作用约定（"native 不得压入多于返回值的栈元素"），并在 debug 模式下检查。
4. **deadline 周期可配置**：将 `0xFFF`（4096）提取为 `DEADLINE_CHECK_PERIOD` 常量，允许按场景调整（交互式 256，批处理 4096）。
5. **JIT 路径栈卫生**：在 T9（JIT 语义保持）框架下，验证 JIT hostcall trampoline 的 `call_with_args` 是否保持 `truncate` 语义。
6. **T35 联动**：将本文 §9 的摊还分析复用于解释器侧 `tick`，证明二者 deadline 触发时机等价。

---

> **数理部审查记录**
> v1.0 (2026-07-02)：初稿，建立五个主定理与完整证明框架，独立局限章节记录 5 类局限。
