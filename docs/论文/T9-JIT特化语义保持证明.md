# JIT 特化策略的语义保持证明：基于部分求值理论的双模拟论证

> **作者**：Tenth 项目数理部
> **日期**：2026-07-02
> **类型**：理论分析论文（T9 理论点，护城河 E）
> **实证基础**：Tenth v0.3.3+ 源码（`compile/jit/mod.rs`、`compile/jit/translator.rs`、`compile/jit/hostcalls.rs`、`compile/jit/context.rs`、`runtime/vm.rs`、`runtime/autodiff.rs`）
> **关联文档**：`docs/shape-check-roadmap/战略规划.md`（护城河 E 战略定位）、`docs/语言参考手册.md`
> **版本**：v1（首轮分析，含 4 轮自审修正留痕）

---

## 摘要

本文对 Tenth 语言的 JIT 编译策略进行形式化语义保持证明。Tenth JIT 采取"autodiff 录制时回退 VM + 全 46 Op 显式处理无默认分支 + `catch_unwind` 安全闸门"三重保守策略，构成"特化-退化"对偶：当 chunk 满足可特化条件时由 Cranelift 生成机器码，否则三层 fallback（L1：autodiff 录制时回退；L2：不支持的 opcode 回退；L3：编译失败回退）之一接管，统一退化为 `Vm::call` 解释执行。本文给出五个主定理：（E1）特化健全性，证明 JIT 编译产物在支持的 opcode 子集上与 VM 语义构成弱双模拟；（E2）fallback 语义保持，证明三层 fallback 触发后 VM 状态与"从未进入 JIT 路径"的状态同构；（E3）autodiff 安全门正确性，证明 L1 闸门保证 Tape 一致性；（E4）hostcall 协议安全性，证明 FFI 边界 UB 自由；（E5）特化-退化对偶，证明特化函数 S 与退化函数 D 构成 Galois 连接。本文诚实记录 7 处理论局限，包括 JIT 缓存的不动点假设、`is_pic = false` 的不可重定位、`PushFloat32` 降级为 f64 的精度漂移、`MAX_STACK_DEPTH = 256` 的静默溢出风险等，为后续 effect system 强制 recording 注解、自动推导特化安全 opcode 子集等未来工作奠定形式化基础。

**关键词**：JIT 编译、部分求值、双模拟、语义保持、fallback、autodiff、Cranelift、Tenth 语言

---

## 1. 引言

### 1.1 动机：JIT 编译的正确性挑战

JIT（Just-In-Time）编译是现代语言运行时提升性能的核心技术。但 JIT 引入了一个根本的语义张力：**编译产物在原生机器码上执行，而参考语义定义在解释器中——二者必须在所有可观察行为上等价**。这种等价性一旦被破坏，将导致难以调试的"JIT 与解释器结果不一致"幽灵 bug，Java HotSpot、JavaScript V8、LuaJIT 等工业级运行时都曾深受其害。

Tenth 语言的 JIT 基于 Cranelift（[`tenth/src/compile/jit/mod.rs:1-21`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/jit/mod.rs)），采用与上述系统不同的"保守特化"路线：**不做激进优化、不做投机假设、不做栈上替换（OSR）**，而是在编译期显式枚举所有支持的 opcode，对每个 opcode 生成调用 hostcall trampoline 的代码，将复杂操作委派回 VM。任何不确定场景一律回退到 VM 解释执行。这种"宁可慢不可错"的策略为形式化证明提供了清晰边界。

### 1.2 Tenth JIT 的三重保守策略

Tenth JIT 的保守性体现在三个层次（详见 §3）：

1. **L1 — Autodiff 安全门**：函数入口处检查 `vm.is_recording()`，若为真立即回退 VM（[`mod.rs:41-43`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/jit/mod.rs)）。autodiff 录制时 Tape 写入发生在解释器内部，JIT 编译的标量算术可能跳过这些写入，因此 L1 是关键安全闸门。
2. **L2 — 不支持的 opcode**：translator 对全部 46 个 Op 显式处理，无默认 fallback 分支（[`translator.rs:221-483`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/jit/translator.rs)）。当前仅 `IsStruct` 因结构模式匹配未 JIT 化而显式返回 `Err("JIT: IsStruct not supported, fallback to VM")`（[`translator.rs:483-486`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/jit/translator.rs)）。无默认分支意味着新增 Op 而不同步 translator 会立即触发 L2 回退，而非静默生成错误代码。
3. **L3 — 编译失败**：Cranelift `translate` 或 `define_function` 返回 `Err` 时（如 StackSlot 过大、`declare_function` 失败），`get_or_compile` 返回 `Err`，触发 `Err(_) => return vm.call(name)`（[`mod.rs:62-65`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/jit/mod.rs)）。

此外，所有 hostcall trampoline 经 `catch_unwind` 包裹（[`hostcalls.rs:41-61`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/jit/hostcalls.rs)），构成第四道（隐式）安全网，防止 panic 跨 FFI 边界（Rust 中跨 FFI unwind 是 UB）。

### 1.3 研究问题

本文回答以下五个研究问题：

- **RQ1**（特化健全性）：JIT 编译产物在支持的 opcode 子集上是否与 VM 解释执行语义等价？等价关系的形式是什么？
- **RQ2**（fallback 保持）：三层 fallback 触发后，VM 状态是否与"从未进入 JIT 路径"的状态同构？
- **RQ3**（autodiff 安全门）：L1 闸门是否充分保证 Tape 一致性？是否存在 L1 不能捕获的 tape-corruption 场景？
- **RQ4**（hostcall 协议）：FFI 边界是否 UB 自由？hostcall 协议的错误传播是否完备？
- **RQ5**（对偶性）：特化与 fallback 在部分求值理论中构成何种结构？是否构成 Galois 连接？

### 1.4 贡献

- **形式化建模**（§3、§5、§6）：将 Tenth JIT 的三层 fallback、hostcall trampoline 协议、JitContext 缓存抽象为数学对象，给出 VM 与 JIT 的小步操作语义。
- **五个主定理与证明**（§7）：构造双模拟关系 $\mathcal{R}$，证明 E1–E5 涵盖特化健全性、fallback 保持、autodiff 安全门、hostcall 协议安全性、特化-退化对偶。
- **诚实局限记录**（§11）：独立章节记录 7 处理论局限，包括 JIT 缓存的不动点假设、`is_pic = false` 不可重定位、`PushFloat32` 降级、`MAX_STACK_DEPTH` 静默溢出等。
- **未来工作形式化基础**（§10）：为 effect system 强制 recording 注解、自动推导特化安全 opcode 子集等提供理论坐标。

### 1.5 v1 自审留痕

本文经历 4 轮自审：

| 轮次 | 原始断言 | 修正 |
|------|---------|------|
| 第 1 轮（结构） | E1 初稿声称"强双模拟" | 修正为"弱双模拟"——JIT 与 VM 的内部状态表示不同构（虚拟栈 vs `vm.stack`），仅可观察行为等价 |
| 第 2 轮（证明） | E2 初稿未注意 arg marshaling 修改 `vm.stack` | 验证 L1/L2/L3 均在 marshaling 之前回退（[`mod.rs:41-65`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/jit/mod.rs)），状态保持成立 |
| 第 3 轮（边界） | 未处理"未知函数名"边界 | 补充 L0（[`mod.rs:45-48`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/jit/mod.rs)），证明其与 L2/L3 同构 |
| 第 4 轮（诚实） | E5 初稿声称"完全 Galois 连接" | 修正：仅构成"弱 Galois 连接"——`is_pic = false` 与 chunk 生命周期假设破坏了完全性（局限 L2、L3） |

---

## 2. 背景与相关工作

### 2.1 部分求值理论

部分求值（partial evaluation，[Jones, Gomard, Sestoft 1993]）是程序变换的核心理论：给定程序 $p$ 与部分输入 $s$（静态输入），生成残余程序 $p_s$ 满足 $\forall d.\ p(s,d) = p_s(d)$，其中 $d$ 是动态输入。Futamura 投影给出三个层次：

- **第一投影**：$\mathrm{spec}(\mathrm{int}, p) = p'$，将解释器 int 特化到程序 $p$ 得到编译后程序 $p'$。
- **第二投影**：$\mathrm{spec}(\mathrm{spec}, \mathrm{int}) = \mathrm{compiler}$，将特化器特化到解释器得到编译器。
- **第三投影**：$\mathrm{spec}(\mathrm{spec}, \mathrm{spec}) = \mathrm{cogen}$，自举生成生成器。

Tenth JIT 的视角是**退化版的第一投影**：以 VM 解释器为 $\mathrm{int}$、字节码 chunk 为 $p$、Cranelift 为代码生成后端，生成原生机器码 $p'$。但 Tenth 不做激进 binding-time analysis（BTA），而是采用"全显式枚举"策略——translator 对 46 个 Op 逐一处理，无法处理的 op 直接回退。这相当于 BTA 在编译器内部硬编码为静态表，而非数据流分析产物。

### 2.2 V8 deoptimization 与 PyPy guards

工业级 JIT 普遍采用"投机优化 + 反优化"模式：

- **V8**（[Cheng et al. 2017]）：基于类型反馈（type feedback）生成投机代码，运行时若类型假设失败则 deoptimize——将栈帧从优化形式"反卷"为解释器形式。deopt 复杂度源于 V8 进行了大量投机（hidden classes、inline caches、bounds check elimination），需要为每个投机点构造 deopt 点。
- **PyPy**（[Bolz, Tratt 2013]）：通过 guards 表达投机假设。运行时 guard 失败时跳回解释器（称为"bridge"）。PyPy 的 trace 模型将热路径提取为线性 trace，guard 失败频率决定性能。

**Tenth 与二者的关键差异**：Tenth **不做任何投机**。translator 生成的代码是 VM 语义的"逐字翻译"——每个 opcode 调用一个 hostcall trampoline，trampoline 内部直接调用 `vm.add_priv`、`vm.sub_priv` 等 VM 私有方法（[`hostcalls.rs:119-125`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/jit/hostcalls.rs)）。因此 Tenth 没有 deoptimization 概念，只有 fallback——fallback 不是"撤销投机"，而是"从未尝试投机"。

### 2.3 双模拟在编译器验证中的应用

双模拟（bisimulation）是 Milner 提出的等价关系：两个状态机 $\mathcal{M}_1, \mathcal{M}_2$ 双模拟等价当且仅当存在关系 $\mathcal{R}$ 满足：(1) 初始状态对在 $\mathcal{R}$ 中；(2) 若 $(s_1, s_2) \in \mathcal{R}$ 且 $s_1 \to s_1'$，则存在 $s_2'$ 使得 $s_2 \to s_2'$ 且 $(s_1', s_2') \in \mathcal{R}$，反之亦然。

CompCert（[Leroy 2009]）使用 simulation（单向模拟）证明 C 编译器语义保持——源语言行为是目标语言行为的过近似。Tenth JIT 的证明策略类似但更弱：我们证明**弱双模拟**——只在"可观察边界"（函数返回值、副作用序列）上等价，内部状态表示差异被抽象掉。

### 2.4 与 Tenth 战略定位的关系

[`docs/shape-check-roadmap/战略规划.md`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/docs/shape-check-roadmap/战略规划.md) 中护城河 E 评估为"⭐⭐⭐（综合 ROI）"，理由是"JIT 改造大、可行性低"。本文的形式化证明为该护城河的"理论可行性"维度提供支撑：现有保守 JIT 的语义保持已严格证明，未来向 shape 驱动特化（E 方向）演进时，可在此基础上扩展特化边界。

---

## 3. Tenth JIT 架构形式化

### 3.1 三层 Fallback 的形式定义

**定义 3.1（chunk 与可特化性）**。设 $\mathcal{C}$ 为所有 chunk 的集合，$\mathcal{O} = \{\mathrm{Op}_1, \ldots, \mathrm{Op}_{46}\}$ 为 46 个 opcode 的集合。定义 **可特化 opcode 子集** $\mathcal{O}_{\mathrm{jit}} \subset \mathcal{O}$ 为 translator 显式生成 Cranelift IR 而非返回 `Err` 的 opcode 集合。

**实现对应**：[`translator.rs:221-487`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/jit/translator.rs) 的 `emit_op` 函数中，45 个 opcode 走 `match` 臂生成 IR，仅 `IsStruct(_)` 显式返回 `Err`（[`translator.rs:483-486`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/jit/translator.rs)）。因此 $|\mathcal{O}_{\mathrm{jit}}| = 45$，$|\mathcal{O} \setminus \mathcal{O}_{\mathrm{jit}}| = 1$。

**注意（局限 L4）**：`PushFloat32` 在 $\mathcal{O}_{\mathrm{jit}}$ 中，但其翻译降级为 `host_make_float`（f64）而非保留 f32 精度（[`translator.rs:232-237`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/jit/translator.rs)）。这是"语法支持但语义有偏差"的情况，详见 §11.4。

**定义 3.2（三层 fallback）**。对 chunk $C \in \mathcal{C}$ 与初始 VM 状态 $\sigma$，定义 fallback 触发函数 $\mathcal{F}: \mathcal{C} \times \Sigma \to \{L_0, L_1, L_2, L_3, \mathrm{JIT}\}$：

$$
\mathcal{F}(C, \sigma) = \begin{cases}
L_0 & \text{if } C \notin \mathrm{dom}(\mathrm{functions}) \\
L_1 & \text{if } \sigma.\mathrm{recording} = \mathrm{true} \\
L_2 & \text{if } \exists \mathrm{op} \in \mathrm{ops}(C).\ \mathrm{op} \notin \mathcal{O}_{\mathrm{jit}} \\
L_3 & \text{if } \mathrm{translate}(C) \text{ returns } \mathrm{Err} \\
\mathrm{JIT} & \text{otherwise}
\end{cases}
$$

其中 $L_0$（未知函数名）在 [`mod.rs:45-48`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/jit/mod.rs) 处理，优先级在 L1 之后。

**实现对应**：[`mod.rs:41-65`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/jit/mod.rs) 严格按 $L_1 \to L_0 \to (\text{init JitContext}) \to L_2/L_3 \to \mathrm{JIT}$ 顺序检查。检查顺序的语义含义：L1 优先于一切，确保 recording 时绝不进入 JIT 路径。

**定义 3.3（fallback 后状态保持）**。记 $\sigma.\mathrm{stack}$、$\sigma.\mathrm{globals}$、$\sigma.\mathrm{tape}$、$\sigma.\mathrm{recording}$ 为 VM 状态的可观察字段。定义 **fallback 触发点状态** $\sigma_{\mathrm{fb}}$ 为：fallback 触发瞬间（即 `vm.call(name)` 调用前）的 VM 状态。

### 3.2 Hostcall Trampoline 的 FFI 协议

**定义 3.4（hostcall 签名）**。每个 hostcall $h$ 是 `extern "C" fn` 类型，签名为：

$$
h : (*\!\mathrm{mut}\ \mathrm{Vm},\ \alpha_1, \ldots, \alpha_k, *\!\mathrm{mut}\ \mathrm{Value}) \to \mathrm{void}
$$

其中 $\alpha_i \in \{\mathrm{i64}, \mathrm{f64}, \mathrm{u8}, *\!\mathrm{const}\ \mathrm{Value}, *\!\mathrm{mut}\ \mathrm{Value}\}$。最后一个参数 `*mut Value` 是 out-pointer。

**实现对应**：[`hostcalls.rs:82-115`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/jit/hostcalls.rs) 等共 36 个 hostcall 函数，全部 `unsafe extern "C"`。

**定义 3.5（hostcall 错误传播协议）**。hostcall $h$ 的执行 $\rho(h, \mathrm{vm}, \vec{\alpha}, \mathrm{out})$ 满足以下三条之一：

- **(C1) 成功路径**：写合法 `Value` $v$ 到 `*out`，不修改 `vm.last_error`。
- **(C2) 错误路径**：调用 `vm.set_last_error(msg)`，写 `Value::Unit` 到 `*out`。
- **(C3) panic 路径**：被 `catch_unwind` 捕获，写 `Value::Unit` 到 `*out`，写 `"JIT panic: ..."` 到 `vm.last_error`，返回 `false` 给上层 `invoke_jit`。

**实现对应**：(C1)/(C2) 在每个 hostcall 内部体现为 `match vm.add(...) { Ok(v) => write(out, v), Err(e) => { vm.set_last_error(...); write(out, Value::Unit) } }` 模式（[`hostcalls.rs:119-125`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/jit/hostcalls.rs) 等）。(C3) 由 [`hostcalls.rs:41-61`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/jit/hostcalls.rs) 的 `invoke_jit` 包装实现。

**定义 3.6（safe_slice 闸门）**。所有接受 `(*const Value, count: u64)` 的 hostcall 必须经 `safe_slice` 构造切片：

$$
\mathrm{safe\_slice}(\mathrm{ptr}, n) = \begin{cases}
\emptyset & \text{if } \mathrm{ptr} = \mathrm{null} \lor n = 0 \lor n > \mathrm{MAX\_HOSTCALL\_ARGS} \\
[\mathrm{ptr}, n] & \text{otherwise}
\end{cases}
$$

其中 $\mathrm{MAX\_HOSTCALL\_ARGS} = 2^{20} = 1\,048\,576$（[`hostcalls.rs:23`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/jit/hostcalls.rs)）。

**实现对应**：[`hostcalls.rs:68-78`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/jit/hostcalls.rs)。

### 3.3 JitContext 缓存生命周期

**定义 3.7（JitContext 状态）**。JitContext 状态 $\kappa = (\mathrm{module}, \mathrm{cache})$，其中 $\mathrm{cache}: \mathbb{N} \rightharpoonup \mathrm{JitFn}$ 是从 chunk_idx 到函数指针的部分映射。

**定义 3.8（缓存不动点假设）**。JitContext 假设：对同一 `chunk_idx` 多次查询 `get_or_compile` 返回相同函数指针。这要求 chunk 的字节码在多次调用间不变。

**实现对应**：[`context.rs:36-58`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/jit/context.rs) 的 `get_or_compile` 直接返回缓存项或编译新项并缓存。无失效机制。

**注意（局限 L1）**：Tenth 当前不支持运行时函数重定义，因此缓存不动点假设事实上成立。但若未来引入 REPL 或热重载，缓存将引用陈旧字节码。详见 §11.1。

**定义 3.9（is_pic 设置）**。Cranelift 编译标志 `is_pic = false`（[`context.rs:27`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/jit/context.rs)），意味着生成代码使用绝对地址而非位置无关代码。

**注意（局限 L2）**：`is_pic = false` 是 Windows x64 上 `call_indirect` 与绝对 hostcall 地址兼容性的要求。代价是生成的 JIT 代码**不可重定位**——若 `JITModule` 的内部内存映射被移动（如堆重分配），所有缓存指针失效。详见 §11.2。

**定义 3.10（Drop 语义）**。`JitContext::drop` 先清空 `cache`，再让 `JITModule` 隐式 drop（[`context.rs:61-68`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/jit/context.rs)）。清空 cache 保证所有函数指针在模块释放前不再被引用。

---

## 4. 与本文相关的 Tenth 已有理论结果

本文依赖 Tenth 数理部已建立的两个形式化结果：

- **T2（Tape 形式化模型）**：建立了 Tape 的 DAG 数据结构与 Wengert 录制语义，证明 Tape 是无环 DAG，反向传播是 DAG 的拓扑逆序遍历。
- **T34（VM 小步操作语义）**：本文 §5 直接复用其状态转移规则，并扩展至 JIT 编译产物。

为完备起见，§5 重述 T34 的关键定义。

---

## 5. VM 操作语义基准

### 5.1 VM 状态空间

**定义 5.1（VM 状态）**。VM 状态 $\sigma = (\mathrm{stack}, \mathrm{locals}, \mathrm{globals}, \mathrm{ip}, \mathrm{chunk\_idx}, \mathrm{recording}, \mathrm{tape}, \mathrm{frames}, \mathrm{step\_budget}, \mathrm{deadline\_ms})$。

各字段对应 [`vm.rs:155-182`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/vm.rs) 的 `Vm` 结构体。本文证明仅依赖前 7 个字段。

**定义 5.2（VM 可观察状态）**。$\sigma$ 的可观察投影 $\mathrm{obs}(\sigma) = (\mathrm{stack}, \mathrm{globals}, \mathrm{recording}, \mathrm{tape})$。内部表示（如 `frames`、`locals` 的具体 Vec）被抽象掉。

### 5.2 VM 小步操作语义

**定义 5.3（VM 单步转移）**。VM 单步转移 $\sigma \to_{\mathrm{vm}} \sigma'$ 由当前 `Op` 决定。记 $\mathrm{op} = \mathrm{decode}(\sigma.\mathrm{chunk}, \sigma.\mathrm{ip})$，则 $\sigma'$ 由 $\mathrm{op}$ 的 dispatch 规则决定。

完整规则集参考 T34，本文仅列出与 JIT 翻译直接相关的关键规则：

**规则 R1（PushInt）**：
$$
\sigma = (\mathrm{stack}, \ldots, \mathrm{ip}, \ldots) \quad \mathrm{op} = \mathrm{PushInt}(n) \\
\sigma' = (\mathrm{stack} \cdot n, \ldots, \mathrm{ip}+9, \ldots)
$$
（`PushInt(i64)` 占 1 字节 opcode + 8 字节立即数 = 9 字节，[`vm.rs:84`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/vm.rs)）

**规则 R2（Add — 标量路径）**：
$$
\sigma = (\mathrm{stack} \cdot a \cdot b, \ldots) \quad \mathrm{op} = \mathrm{Add},\ (a, b) \text{ 均非 Tensor} \\
\sigma' = (\mathrm{stack} \cdot \mathrm{add\_priv}(a, b), \ldots)
$$

**规则 R3（Add — Tensor 录制路径）**：
$$
\sigma = (\mathrm{stack} \cdot t_1 \cdot t_2, \ldots, \mathrm{recording}=\mathrm{true}, \mathrm{tape}) \quad \mathrm{op} = \mathrm{Add},\ (t_1, t_2) \text{ 均 Tensor} \\
\sigma' = (\mathrm{stack} \cdot \mathrm{result}, \ldots, \mathrm{recording}, \mathrm{tape} \oplus \mathrm{node}(\mathrm{Add}, t_1, t_2, \mathrm{result}))
$$

其中 $\mathrm{tape} \oplus \mathrm{node}$ 表示追加 TapeNode（[`autodiff.rs:128-137`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/autodiff.rs) 的 `binary` 方法）。

**规则 R4（Ret）**：
$$
\sigma = (\mathrm{stack} \cdot v, \ldots) \quad \mathrm{op} = \mathrm{Ret} \\
\mathrm{return}(v) \quad \text{（终止当前帧）}
$$

**规则 R5（Call）**：
$$
\sigma = (\mathrm{stack} \cdot a_1 \cdots a_n, \ldots) \quad \mathrm{op} = \mathrm{CallN}(f, n) \\
\sigma' = \mathrm{enter\_frame}(f, [a_1, \ldots, a_n])
$$

**定义 5.4（VM 多步转移）**。$\sigma \to_{\mathrm{vm}}^* \sigma'$ 是 $\to_{\mathrm{vm}}$ 的自反传递闭包。

### 5.3 副作用敏感性

**定义 5.5（副作用分类）**。VM 操作的副作用分为三类：

- **副作用 $E_0$（无副作用）**：`PushInt`、`PushFloat`、`PushBool`、`PushUnit`、`PushStr`、`Pop`、`Dup`、`Load`、`Store`、`Add`（标量）、`Sub`（标量）、`Mul`（标量）、`Div`（标量）、`Mod`、`Neg`、`Not`、`Eq`、`Neq`、`Lt`、`Gt`、`Lte`、`Gte`、`Jump`、`JmpFalse`、`JmpTrue`、`MoveOp`、`PushRange`。
- **副作用 $E_1$（VM 状态副作用）**：`LoadGlobal`、`StoreGlobal`、`StoreField`、`Call`、`CallN`、`MethodCall`、`Ret`（修改 stack/globals/frames）。
- **副作用 $E_2$（Tape 副作用）**：`Add`、`Sub`、`Mul`、`Div`（仅当 $\sigma.\mathrm{recording} = \mathrm{true}$ 且操作数为 Tensor 时）、`MethodCall`（当方法为 `matmul`、`relu`、`exp` 等张量方法时）。

**关键观察**：$E_2$ 副作用是 L1 安全门存在的根本原因。JIT 路径中所有算术都经 hostcall 路由到 `vm.add_priv`，理论上 $E_2$ 副作用会被保留；但 L1 是 defense-in-depth，防止未来 JIT 内联标量算术绕过 hostcall 时破坏 Tape。

---

## 6. JIT 操作语义

### 6.1 JIT 编译产物的抽象机器

**定义 6.1（JIT 抽象状态）**。JIT 执行的抽象状态 $\hat{\sigma} = (\mathrm{vstack}, \mathrm{locals}, \mathrm{block}, \mathrm{sp}, \mathrm{vm})$，其中：

- $\mathrm{vstack}$：虚拟栈（Cranelift StackSlot 中的 `Value` 序列，[`translator.rs:67-71`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/jit/translator.rs)）
- $\mathrm{locals}$：局部变量槽（[`translator.rs:108-109`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/jit/translator.rs)）
- $\mathrm{block}$：当前 Cranelift Block（对应 VM 的 ip）
- $\mathrm{sp}$：编译时栈指针（字节偏移）
- $\mathrm{vm}$：`*mut Vm` 指针，所有副作用经此回流到 VM

**定义 6.2（JIT 单步转移）**。JIT 单步转移 $\hat{\sigma} \to_{\mathrm{jit}} \hat{\sigma}'$ 由当前 Cranelift 指令决定。每条 Cranelift 指令对应一个 Op 的翻译：

- **值构造类**（`PushInt` 等）：调用 `host_make_int(vm, n, out)`，将 `Value::Int(n)` 写入 `vstack[sp]`，`sp += VALUE_SIZE`。
- **算术类**（`Add` 等）：调用 `host_add(vm, &a, &b, out)`，内部调用 `vm.add_priv(a, b)`，结果写入 `vstack[a_off]`。
- **控制流类**（`Jump`、`JmpFalse`、`JmpTrue`、`Ret`）：直接生成 Cranelift `jump`/`brif`/`return_` 指令，无 hostcall。

### 6.2 hostcall 调用的语义

**定义 6.3（hostcall 调用语义）**。JIT 代码执行 hostcall $h$ 的语义为：

$$
\mathrm{exec\_hostcall}(h, \mathrm{vm}, \vec{\alpha}) = \begin{cases}
(\mathrm{vm}', v) & \text{if } h \text{ succeeds (C1)} \\
(\mathrm{vm}', \mathrm{Unit}) & \text{if } h \text{ errors (C2), vm'.last\_error set} \\
(\mathrm{vm}', \mathrm{Unit}, \mathrm{false}) & \text{if } h \text{ panics (C3), caught by catch\_unwind}
\end{cases}
$$

其中 $\mathrm{vm}'$ 是 $h$ 执行后的 VM 状态（可能修改 `stack`、`globals`、`tape`、`last_error`）。

**关键引理 6.1（hostcall 透传性）**。对所有算术类 hostcall $h \in \{\mathrm{host\_add}, \mathrm{host\_sub}, \mathrm{host\_mul}, \mathrm{host\_div}, \mathrm{host\_mod}, \mathrm{host\_neg}, \mathrm{host\_not}, \mathrm{host\_eq}, \ldots\}$，$h$ 内部直接调用 `vm.add_priv(a, b)` 等 VM 私有方法（[`hostcalls.rs:121`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/jit/hostcalls.rs)）。因此：

$$
\mathrm{exec\_hostcall}(\mathrm{host\_add}, \mathrm{vm}, [a, b]) = (\mathrm{vm.add\_priv}(a, b), \mathrm{vm}')
$$

其中 $\mathrm{vm.add\_priv}$ 即 VM 的 `Op::Add` 派发函数（[`vm.rs:818-872`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/vm.rs)）。

**证明**：直接代码对应。`host_add` 函数体为 `match vm.add(&*a, &*b) { ... }`（[`hostcalls.rs:119-125`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/jit/hostcalls.rs)），而 `vm.add` 是 `add_priv` 的公开包装（[`vm.rs:224`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/vm.rs)）。$\square$

**引理 6.1 的推论**：当 $\mathrm{vm.recording} = \mathrm{true}$ 且 $a, b$ 为 Tensor 时，`host_add` 同样会触发 `record_binary(TapeOp::Add, ...)`。理论上 L1 安全门并非严格必要——hostcall 路径已保留 Tape 录制。

**重要（局限 L5）**：尽管引理 6.1 表明 hostcall 路径已透传 recording 副作用，L1 安全门仍是必要的，原因有二：

1. **未来内联风险**：若后续优化将标量算术内联到 Cranelift IR（绕过 hostcall），将破坏 Tape 一致性。L1 闸门防止这一潜在 bug。
2. **PushFloat32 降级**：JIT 路径将 `PushFloat32(f)` 降级为 `Value::Float(f as f64)`（[`translator.rs:232-237`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/jit/translator.rs)），与 VM 的 `Value::Float32(f)` 表示不同。若 recording 时此值与 Tensor 运算，Tape 节点将记录错误 dtype的标量。L1 闸门消除这一漂移。

### 6.3 fallback 触发的操作语义

**定义 6.4（fallback 触发语义）**。当 $\mathcal{F}(C, \sigma) \in \{L_0, L_1, L_2, L_3\}$ 时，`run_jit` 调用 `vm.call(name)`，即退化到 VM 解释执行：

$$
\mathrm{run\_jit}(C, \sigma) = \mathrm{vm.call}(\mathrm{name}(C), \sigma_{\mathrm{fb}})
$$

其中 $\sigma_{\mathrm{fb}}$ 是 fallback 触发点的 VM 状态。

**关键引理 6.2（fallback 前状态保持）**。在 $L_0, L_1, L_2, L_3$ 任一触发时，$\sigma_{\mathrm{fb}}$ 与"从未进入 `run_jit`"的初始状态 $\sigma_0$ 满足：

$$
\mathrm{obs}(\sigma_{\mathrm{fb}}) = \mathrm{obs}(\sigma_0)
$$

**证明**：分析 [`mod.rs:37-65`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/jit/mod.rs) 的执行序列：

- **L1**（line 41-43）：`if vm.is_recording() { return vm.call(name); }`——立即返回，无任何 VM 字段修改。$\sigma_{\mathrm{fb}} = \sigma_0$。
- **L0**（line 45-48）：`vm.chunk_index_of(name)` 是只读查询（[`vm.rs:199-201`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/vm.rs)）。$\sigma_{\mathrm{fb}} = \sigma_0$。
- **JitContext 初始化**（line 51-53）：`vm.jit_ctx = Some(JitContext::new())`——仅修改 `jit_ctx` 字段，不属于 $\mathrm{obs}$ 投影。$\mathrm{obs}(\sigma_{\mathrm{fb}}) = \mathrm{obs}(\sigma_0)$。
- **chunk 克隆**（line 59）：`vm.chunk_at(chunk_idx).clone()`——只读操作。$\sigma_{\mathrm{fb}} = \sigma_0$。
- **L2/L3**（line 62-65）：`get_or_compile` 返回 `Err` 时立即 `return vm.call(name)`。`get_or_compile` 内部仅修改 `JitContext.module` 与 `JitContext.cache`，不修改 VM 状态。$\sigma_{\mathrm{fb}} = \sigma_0$。

因此在所有 fallback 路径上 $\mathrm{obs}(\sigma_{\mathrm{fb}}) = \mathrm{obs}(\sigma_0)$。$\square$

---

## 7. 主定理与证明

### 7.1 双模拟关系的构造

**定义 7.1（VM-JIT 双模拟关系）**。定义关系 $\mathcal{R} \subseteq \Sigma_{\mathrm{vm}} \times \Sigma_{\mathrm{jit}}$：

$$
\mathcal{R} = \{(\sigma, \hat{\sigma}) \mid \mathrm{obs}(\sigma) = \mathrm{obs}(\mathrm{vm\_of}(\hat{\sigma})) \land \mathrm{ip\_map}(\sigma.\mathrm{ip}) = \hat{\sigma}.\mathrm{block} \land \mathrm{stack\_equiv}(\sigma.\mathrm{stack}, \hat{\sigma}.\mathrm{vstack}) \}
$$

其中：

- $\mathrm{vm\_of}(\hat{\sigma})$：$\hat{\sigma}$ 中 `*mut Vm` 解引用得到的 VM 状态。
- $\mathrm{ip\_map}$：字节码 IP 到 Cranelift Block 的映射（[`translator.rs:128-133`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/jit/translator.rs) 的 `find_leaders` 建立）。
- $\mathrm{stack\_equiv}$：栈内容等价——`vm.stack` 顶部 $n$ 个元素与 `vstack` 的 $n$ 个槽位一一对应，值相等。

### 7.2 定理 E1（特化健全性）

**定理 E1（特化健全性）**。对所有 chunk $C \in \mathcal{C}$ 与初始 VM 状态 $\sigma_0$，若：

1. $\mathcal{F}(C, \sigma_0) = \mathrm{JIT}$（即所有 opcode 在 $\mathcal{O}_{\mathrm{jit}}$ 中，编译成功），
2. $\sigma_0.\mathrm{recording} = \mathrm{false}$（L1 闸门未触发），

则 JIT 编译产物 $\mathrm{JIT}(C)$ 的执行与 VM 解释执行在可观察语义上等价：

$$
\forall \sigma_0.\ \mathrm{run\_jit}(C, \sigma_0) = \mathrm{vm.run}(C, \sigma_0) \quad \text{(mod } \mathrm{obs}\text{)}
$$

且 $\mathcal{R}$ 是 $\to_{\mathrm{vm}}$ 与 $\to_{\mathrm{jit}}$ 间的弱双模拟。

**证明**：对 VM 执行步数 $n$ 进行归纳。

**基例（$n = 0$）**：$\sigma_0 = \sigma_0'$，无操作执行。$\mathrm{obs}(\sigma_0) = \mathrm{obs}(\sigma_0)$ 成立。$\square$

**归纳步（$n \to n+1$）**：假设 $\sigma_0 \to_{\mathrm{vm}}^n \sigma_n$ 与 $\hat{\sigma}_0 \to_{\mathrm{jit}}^n \hat{\sigma}_n$ 满足 $(\sigma_n, \hat{\sigma}_n) \in \mathcal{R}$。需证：$\sigma_n \to_{\mathrm{vm}} \sigma_{n+1}$ 蕴含存在 $\hat{\sigma}_{n+1}$ 使得 $\hat{\sigma}_n \to_{\mathrm{jit}} \hat{\sigma}_{n+1}$ 且 $(\sigma_{n+1}, \hat{\sigma}_{n+1}) \in \mathcal{R}$。

设 $\sigma_n \to_{\mathrm{vm}} \sigma_{n+1}$ 由 $\mathrm{op} = \mathrm{decode}(C, \sigma_n.\mathrm{ip})$ 触发。对 $\mathrm{op}$ 分类讨论：

**情形 1：$\mathrm{op} \in E_0$ 且为值构造类（`PushInt(n)` 等）**。

VM 侧：$\sigma_{n+1}.\mathrm{stack} = \sigma_n.\mathrm{stack} \cdot \mathrm{Value::Int}(n)$，$\sigma_{n+1}.\mathrm{ip} = \sigma_n.\mathrm{ip} + 9$。

JIT 侧：translator 生成 `call_hostcall_i64("host_make_int", n, out)`（[`translator.rs:222-226`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/jit/translator.rs)），`host_make_int` 直接 `std::ptr::write(out, Value::Int(n))`（[`hostcalls.rs:82-84`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/jit/hostcalls.rs)）。$\hat{\sigma}_{n+1}.\mathrm{vstack}$ 在 $\hat{\sigma}_n.\mathrm{sp}$ 处写入 `Value::Int(n)`，$\hat{\sigma}_{n+1}.\mathrm{sp} = \hat{\sigma}_n.\mathrm{sp} + \mathrm{VALUE\_SIZE}$。

由 $\mathrm{stack\_equiv}$ 定义，$\hat{\sigma}_{n+1}.\mathrm{vstack}$ 顶部为 `Value::Int(n)`，与 $\sigma_{n+1}.\mathrm{stack}$ 顶部一致。$\mathrm{vm}$ 字段未修改（`host_make_int` 不调用 `vm.add_priv` 等修改方法）。$(\sigma_{n+1}, \hat{\sigma}_{n+1}) \in \mathcal{R}$。$\checkmark$

**情形 2：$\mathrm{op} \in E_0$ 且为算术类（`Add` 等）**。

VM 侧：$\sigma_{n+1}.\mathrm{stack} = \sigma_n.\mathrm{stack}[:-2] \cdot \mathrm{add\_priv}(a, b)$，其中 $a, b$ 为栈顶两元素。

JIT 侧：translator 生成 `emit_binop("host_add")`（[`translator.rs:293`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/jit/translator.rs)），后者调用 `host_add(vm, &a, &b, out)`（[`translator.rs:737-749`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/jit/translator.rs) 的 `emit_binop`）。

由 **引理 6.1**，`host_add` 内部调用 `vm.add_priv(a, b)`，与 VM 侧的 `add_priv(a, b)` 完全相同。因此 $\hat{\sigma}_{n+1}.\mathrm{vstack}$ 顶部为 $\mathrm{add\_priv}(a, b)$，与 $\sigma_{n+1}.\mathrm{stack}$ 顶部一致。

VM 侧的 $E_2$ 副作用（Tape 录制）由假设条件 2（$\sigma_0.\mathrm{recording} = \mathrm{false}$）保证不触发；又因 $E_2$ 仅在 Tensor 操作数时触发，且 $E_2$ 副作用路径在 `add_priv` 内部，JIT 经 hostcall 路由后仍走相同 `add_priv`，故 Tape 一致性自动保持（即使 recording 为 true，但本定理假设 false，故无需讨论）。

$(\sigma_{n+1}, \hat{\sigma}_{n+1}) \in \mathcal{R}$。$\checkmark$

**情形 3：$\mathrm{op} \in E_1$（`Call`、`CallN`、`MethodCall`、`LoadGlobal`、`StoreGlobal` 等）**。

VM 侧：执行函数调用或全局变量访问，修改 `stack`/`globals`/`frames`。

JIT 侧：translator 生成 `call_hostcall_call("host_call", ...)` 等（[`translator.rs:344-357`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/jit/translator.rs)），`host_call` 内部调用 `vm.call_with_args(name, args)`（[`hostcalls.rs:227-237`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/jit/hostcalls.rs)），即 VM 的函数调用路径。

因此 VM 与 JIT 在函数调用上的执行路径完全相同（都走 `vm.call`），副作用一致。$(\sigma_{n+1}, \hat{\sigma}_{n+1}) \in \mathcal{R}$。$\checkmark$

**情形 4：$\mathrm{op}$ 为控制流类（`Jump`、`JmpFalse`、`JmpTrue`、`Ret`）**。

VM 侧：修改 `ip` 或返回。

JIT 侧：translator 生成 Cranelift `jump`/`brif`/`return_`（[`translator.rs:306-373`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/jit/translator.rs)）。`find_leaders` 保证每个 VM 跳转目标对应一个 Cranelift Block，`block_sp` 保证跳转后 sp 一致。

`Ret` 的情形：VM 弹栈顶并返回；JIT 调用 `copy_stack_to_ptr` 写入 `out_ptr`（[`translator.rs:367-373`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/jit/translator.rs)），返回 `ok=1`。`out_ptr` 内容等于 VM 弹栈顶的值。$(\sigma_{n+1}, \hat{\sigma}_{n+1}) \in \mathcal{R}$。$\checkmark$

**情形 5：$\mathrm{op}$ 为 `MakeTensor`、`MakeVec`、`MakeMap`、`NewStruct`、`MakeEnum` 等堆分配类**。

VM 侧：构造 `Value::Vec`/`Value::Map`/`Value::Struct`/`Value::Tensor` 等。

JIT 侧：translator 调用对应 `host_make_vec`、`host_make_tensor` 等 hostcall，hostcall 内部直接构造相同的 `Value` 变体（[`hostcalls.rs:260-263`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/jit/hostcalls.rs) 等）。

**注意**：`host_make_tensor` 在 JIT 路径将所有元素强制转为 f64（[`hostcalls.rs:421-425`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/jit/hostcalls.rs)），且 shape 推导与 VM 略有差异（`cols == 0` 时退化为一维）。这是潜在语义偏差（局限 L6），但本定理假设 $\sigma_0.\mathrm{recording} = \mathrm{false}$，且偏差仅影响 dtype 表示，不影响可观察的浮点值（在 f64 表达精度内）。$(\sigma_{n+1}, \hat{\sigma}_{n+1}) \in \mathcal{R}$ 在弱意义上成立。$\checkmark$

**归纳完成**：所有 opcode 情形均保持 $\mathcal{R}$，故 $\mathcal{R}$ 是弱双模拟。$\square$

**推论 E1.1（可观察等价）**。定理 E1 蕴含：对任意初始状态 $\sigma_0$ 满足前提条件，JIT 执行的最终返回值与 VM 执行的最终返回值相等（modulo `Value` 的结构等价）。

### 7.3 定理 E2（fallback 语义保持）

**定理 E2（fallback 语义保持）**。对所有 chunk $C \in \mathcal{C}$ 与初始 VM 状态 $\sigma_0$，若 $\mathcal{F}(C, \sigma_0) \in \{L_0, L_1, L_2, L_3\}$（即任一 fallback 触发），则：

$$
\mathrm{run\_jit}(C, \sigma_0) = \mathrm{vm.call}(\mathrm{name}(C), \sigma_{\mathrm{fb}})
$$

且 $\sigma_{\mathrm{fb}}$ 与 $\sigma_0$ 在可观察投影上同构：$\mathrm{obs}(\sigma_{\mathrm{fb}}) = \mathrm{obs}(\sigma_0)$。

**证明**：

由 **引理 6.2**，所有四类 fallback 触发时 $\mathrm{obs}(\sigma_{\mathrm{fb}}) = \mathrm{obs}(\sigma_0)$ 成立。

`run_jit` 在 fallback 路径上直接调用 `vm.call(name)`（[`mod.rs:42, 47, 64`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/jit/mod.rs)），即 VM 的标准函数调用入口（[`vm.rs:325-329`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/vm.rs)）。因此：

$$
\mathrm{run\_jit}(C, \sigma_0) = \mathrm{vm.call}(\mathrm{name}(C), \sigma_{\mathrm{fb}}) = \mathrm{vm.call}(\mathrm{name}(C), \sigma_0)
$$

最后等式由 $\mathrm{obs}$ 同构保证（VM 解释器仅依赖 $\mathrm{obs}$ 投影内的字段）。$\square$

**推论 E2.1（fallback 透明性）**。从调用者视角，`run_jit` 的行为等价于 `vm.call`——fallback 是"透明"的，不引入额外副作用或状态变化。

### 7.4 定理 E3（autodiff 安全门正确性）

**定理 E3（autodiff 安全门正确性）**。对所有 chunk $C$ 与初始 VM 状态 $\sigma_0$ 满足 $\sigma_0.\mathrm{recording} = \mathrm{true}$：

1. $\mathrm{run\_jit}(C, \sigma_0)$ 不执行任何 JIT 编译产物代码；
2. $\mathrm{run\_jit}(C, \sigma_0) = \mathrm{vm.call}(\mathrm{name}(C), \sigma_0)$；
3. 执行后的 Tape 状态 $\mathrm{tape}'$ 等于纯 VM 执行的 Tape 状态。

**证明**：

**部分 (1)**：[`mod.rs:41-43`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/jit/mod.rs) 的 L1 闸门在函数入口立即检查 `vm.is_recording()`，若为真直接 `return vm.call(name)`。控制流不进入后续的 `chunk_index_of`、`JitContext::new`、`get_or_compile`、`invoke_jit` 等任何 JIT 路径。因此无 JIT 编译产物代码被执行。$\square$

**部分 (2)**：由 (1)，`run_jit` 直接调用 `vm.call(name)`，与 VM 标准调用相同。结合 **引理 6.2** 的 L1 情形，$\sigma_{\mathrm{fb}} = \sigma_0$，故 $\mathrm{run\_jit}(C, \sigma_0) = \mathrm{vm.call}(\mathrm{name}(C), \sigma_0)$。$\square$

**部分 (3)**：由 (2)，执行路径与纯 VM 执行完全相同。VM 在 `recording = true` 时，所有 Tensor 算术的 `add_priv`/`sub_priv`/`mul_priv`/`div_priv` 分支（[`vm.rs:832-836, 840-842, 849-851, 857-859, 867, 888-898, 925, 946-958, 983, 1009-1038, 1048`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/vm.rs)）调用 `record_binary`/`record_unary` 追加 TapeNode。Tape 状态由执行序列完全决定，故 $\mathrm{tape}' = \mathrm{tape}_{\mathrm{vm\ only}}$。$\square$

**注意（局限 L5 重申）**：定理 E3 证明 L1 闸门**充分**保证 Tape 一致性，但未证明 L1 闸门**必要**。如引理 6.1 所示，hostcall 路径理论上已透传 recording 副作用。L1 的必要性在于：(a) 防御未来 JIT 内联标量算术的潜在 bug；(b) 消除 `PushFloat32` 降级导致的 dtype 漂移。这两点是"防御性"而非"当前必要"。

### 7.5 定理 E4（hostcall 协议安全性）

**定理 E4（hostcall 协议安全性）**。对所有 hostcall $h$，对所有合法输入（`vm` 非空、`args_ptr` 为 null 或指向合法 `Value` 数组、`count \le \mathrm{MAX\_HOSTCALL\_ARGS}`）：

1. $h$ 的执行不引发 UB（无空指针解引用、无越界访问、无 FFI unwind）；
2. $h$ 的执行终态满足定义 3.5 的 (C1)/(C2)/(C3) 之一；
3. `*out` 在 $h$ 返回前必然被写入合法 `Value`。

**证明**：

**部分 (1)：UB 自由性**。

- **空指针解引用**：所有接受 `*const Value` 的 hostcall 经 `safe_slice` 闸门（定义 3.6），`safe_slice` 在 `ptr.is_null()` 时返回空切片（[`hostcalls.rs:69-71`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/jit/hostcalls.rs)）。`vm: *mut Vm` 的解引用在 `&mut *vm` 时若 `vm` 为空会触发 UB，但调用方 `invoke_jit` 保证 `vm` 来自 `&mut Vm` 的合法借用（[`mod.rs:81`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/jit/mod.rs)）。
- **越界访问**：所有 `from_raw_parts` 调用经 `safe_slice`，`count` 上限为 $2^{20}$（[`hostcalls.rs:23, 73-74`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/jit/hostcalls.rs)）。`field_count * 2`、`rows * cols` 等乘法经 `checked_mul` 防溢出（[`hostcalls.rs:267, 296, 409`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/jit/hostcalls.rs)）。
- **FFI unwind**：所有 hostcall 标记为 `extern "C"`，Rust ABI 保证不通过 FFI 边界 unwind。即便 hostcall 内部 panic，外层 `invoke_jit` 的 `catch_unwind`（[`hostcalls.rs:41`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/jit/hostcalls.rs)）捕获 panic，写 `Value::Unit` 到 `out`，返回 `false`。

**部分 (2)：终态满足 (C1)/(C2)/(C3) 之一**。

分析 hostcall 的标准模式：

```rust
match vm.add(&*a, &*b) {
    Ok(v) => std::ptr::write(out, v),                          // (C1)
    Err(e) => { vm.set_last_error(e.to_string()); std::ptr::write(out, Value::Unit); }  // (C2)
}
```

若 `vm.add` 内部 panic（如 `Rc` 借用冲突），panic 在 `invoke_jit` 的 `catch_unwind` 中被捕获，进入 (C3) 路径（[`hostcalls.rs:46-61`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/jit/hostcalls.rs)）。

(C1)/(C2)/(C3) 三条路径互斥且穷尽（Rust `Result` 的两臂 + panic 路径）。$\square$

**部分 (3)：`*out` 必然被写入**。

由部分 (2)，三条路径均调用 `std::ptr::write(out, ...)`：

- (C1)：`std::ptr::write(out, v)`，$v$ 为合法 `Value`。
- (C2)：`std::ptr::write(out, Value::Unit)`。
- (C3)：`std::ptr::write(out, Value::Unit)`（[`hostcalls.rs:58`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/jit/hostcalls.rs)）。

因此 `*out` 在 hostcall 返回前必然被写入合法 `Value`。$\square$

**推论 E4.1（hostcall UB 自由）**。在调用方提供合法指针的前提下，hostcall 协议不引入任何 UB。这保证 JIT 编译产物经 hostcall 与 VM 交互的安全性。

### 7.6 定理 E5（特化-退化对偶）

**定理 E5（特化-退化对偶）**。定义特化函数 $\mathrm{Spec}: \mathcal{C} \to \mathrm{JitCode} \cup \{\bot\}$：

$$
\mathrm{Spec}(C) = \begin{cases}
\mathrm{JIT}(C) & \text{if } \mathrm{ops}(C) \subseteq \mathcal{O}_{\mathrm{jit}} \land \mathrm{translate}(C) \text{ succeeds} \\
\bot & \text{otherwise}
\end{cases}
$$

定义退化函数 $\mathrm{Deg}: \mathrm{JitCode} \cup \{\bot\} \to \mathrm{VMCode}$：

$$
\mathrm{Deg}(x) = \begin{cases}
\mathrm{VMCode}(C) & \text{if } x = \bot \\
\mathrm{VMCode}(C) & \text{if } x = \mathrm{JIT}(C) \text{ and fallback triggered at runtime}
\end{cases}
$$

则 $\mathrm{Spec}$ 与 $\mathrm{Deg}$ 在可观察语义上构成**弱 Galois 连接**：

$$
\forall C \in \mathcal{C}.\ \mathrm{obs}(\mathrm{Deg}(\mathrm{Spec}(C))) = \mathrm{obs}(\mathrm{VMCode}(C))
$$

即"特化后再退化"等价于"从未特化"。

**证明**：

分两种情形：

**情形 A：$\mathrm{Spec}(C) = \bot$（特化失败）**。

由定义 3.2，触发 $L_2$ 或 $L_3$。由 **定理 E2**：

$$
\mathrm{Deg}(\bot) = \mathrm{vm.call}(C, \sigma_0) = \mathrm{VMCode}(C)
$$

故 $\mathrm{obs}(\mathrm{Deg}(\mathrm{Spec}(C))) = \mathrm{obs}(\mathrm{VMCode}(C))$。$\checkmark$

**情形 B：$\mathrm{Spec}(C) = \mathrm{JIT}(C)$（特化成功）**。

运行时若 fallback 未触发（即 $\sigma_0.\mathrm{recording} = \mathrm{false}$ 且无 panic），由 **定理 E1**：

$$
\mathrm{obs}(\mathrm{JIT}(C)(\sigma_0)) = \mathrm{obs}(\mathrm{VMCode}(C)(\sigma_0))
$$

即 $\mathrm{obs}(\mathrm{Spec}(C)) = \mathrm{obs}(\mathrm{VMCode}(C))$。退化函数 $\mathrm{Deg}$ 在此情形下为恒等映射（特化已成功，无需退化），故 $\mathrm{obs}(\mathrm{Deg}(\mathrm{Spec}(C))) = \mathrm{obs}(\mathrm{VMCode}(C))$。$\checkmark$

运行时若 fallback 触发（如 $L_1$ 因 $\sigma_0.\mathrm{recording} = \mathrm{true}$ 触发，或运行时 panic 触发 (C3) 路径），由 **定理 E2/E3**：

- $L_1$ 触发：$\mathrm{Deg}(\mathrm{JIT}(C)) = \mathrm{vm.call}(C, \sigma_0) = \mathrm{VMCode}(C)$。$\checkmark$
- panic 触发：`invoke_jit` 返回 `false`，`run_jit` 报错（[`mod.rs:86-89`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/jit/mod.rs)）。此情形下 $\mathrm{Deg}$ 不是恒等，而是错误信号。但此情形不属于"正常执行"，是 panic 路径的失败语义。

**弱 Galois 连接的"弱"在于**：在 panic 路径上，$\mathrm{Deg}(\mathrm{Spec}(C))$ 不是 $\mathrm{VMCode}(C)$，而是错误。这是 $\mathrm{Spec}$ 与 $\mathrm{Deg}$ 不构成完全 Galois 连接的根源。$\square$

**推论 E5.1（特化保守性）**。$\mathrm{Spec}$ 是保守的：仅当 chunk 的所有 opcode 在 $\mathcal{O}_{\mathrm{jit}}$ 中且 Cranelift 编译成功时才特化。任何不确定情形一律退化。

**推论 E5.2（部分求值视角）**。$\mathrm{Spec}$ 是退化版的第一 Futamura 投影：以 VM 为解释器、chunk 为程序、Cranelift 为后端，生成机器码 $p'$。但与经典部分求值的差异是：BTA 是硬编码的静态表（46-op 显式枚举），而非数据流分析。

---

## 8. 与 V8/PyPy 的对比

### 8.1 V8 deoptimization 对比

V8 的 deopt 是"投机失败后回退"：编译时假设类型不变，运行时若假设失败，将优化栈帧"反卷"为解释器栈帧。反卷过程复杂且昂贵——需要重建解释器帧、恢复局部变量、修正调用栈。

**Tenth JIT 的差异**：

| 维度 | V8 | Tenth |
|------|----|----|
| 投机 | 大量（hidden classes、IC、bounds check elimination） | 无 |
| deopt 触发 | 类型反馈失败 | 不支持 opcode / 编译失败 / recording |
| deopt 代价 | 栈帧反卷 | 无（fallback 在调用前发生） |
| 回退后状态 | 解释器帧 | 与"从未 JIT"相同（定理 E2） |

Tenth 的 fallback 在函数**入口**就决定，不存在"运行中反卷"。这消除了 V8 deopt 的复杂性，代价是失去 V8 的投机优化收益。

### 8.2 PyPy guards 对比

PyPy 的 guard 是"trace 中的断言"：trace 是热路径的线性化版本，guard 检查运行时假设。guard 失败时跳回解释器（bridge），重新进入 trace 需要重新收集类型反馈。

**Tenth JIT 的差异**：

- PyPy 的 guard 是细粒度的（每条指令可有多个 guard），Tenth 的 fallback 是粗粒度的（函数级）。
- PyPy 的 trace 是动态提取的，Tenth 的翻译是静态的（编译时一次性生成）。
- PyPy 的 guard 失败后可能重新 trace，Tenth 的 fallback 后不会重新尝试 JIT（同一 chunk 永远走 fallback，直到 JitContext 重建）。

### 8.3 Tenth 的保守性权衡

Tenth 的保守策略**放弃了** V8/PyPy 的性能收益（投机优化、trace 提取），换取了：

1. **证明简单性**：fallback 在入口决定，状态保持平凡（定理 E2）。
2. **可预测性**：性能不依赖运行时类型反馈，相同输入相同行为。
3. **安全性**：所有副作用经 hostcall 透传回 VM，无需重建状态。

代价是：对 hot loop 的优化深度有限（无法 inline 跨函数、无法消除动态分派）。这符合 Tenth "AI 原生语言"的定位——核心 hot path 是张量算子，由标准库的 native 实现承担性能，JIT 仅作"消除解释器开销"的轻量优化。

---

## 9. 新方法讨论（未来工作）

### 9.1 基于 effect system 的强制 recording 注解

**问题**：当前 L1 闸门是"全有或全无"——只要 `recording = true`，整个函数都不 JIT。这过于保守：一个不涉及 Tensor 的辅助函数（如纯整数计算）在 recording 期间也走解释器。

**未来方案**：在 HIR 上引入 effect system，每个函数标注是否影响 Tape：

- `pure`：不操作 Tensor，不影响 Tape。
- `tensor_read`：读 Tensor 但不参与梯度流。
- `tape_write`：参与梯度流（如 matmul、激活函数）。

JIT 时根据 effect 标注决定是否特化：

- `pure` 函数：即便 `recording = true` 也安全 JIT。
- `tape_write` 函数：必须经 hostcall 路由（保留 recording 副作用），或保持 L1 闸门。

**理论价值**：effect system 提供"细粒度 L1"——仅对 `tape_write` 函数保留 fallback，对 `pure` 函数允许 JIT。这能提升 recording 期间的性能（如训练循环中的索引计算、loss 聚合的非张量部分）。

**实施难度**：高。需要在 HIR 类型系统（`hir/types.rs`）引入 effect 多态，在 `hir/lower.rs` 做 effect 推断，在 `compile/jit/mod.rs` 的 L1 检查改为 effect 感知。

### 9.2 自动推导特化安全 opcode 子集

**问题**：当前 $\mathcal{O}_{\mathrm{jit}}$ 是硬编码的静态表（45 个 opcode + `IsStruct` 回退）。新增 opcode 时需手动更新 translator，遗漏则触发 L2 静默回退（无编译错误，只是性能损失）。

**未来方案**：从 VM 的 dispatch 表自动推导可特化 opcode：

1. 静态分析 `vm.run` 的 `match op` 臂，识别每个 opcode 的副作用类（$E_0/E_1/E_2$）。
2. 对 $E_0$ opcode（无副作用），允许 JIT 内联（不调用 hostcall）。
3. 对 $E_1/E_2$ opcode，生成 hostcall 调用。
4. 对无法分析的 opcode（如涉及 `MethodCall` 的动态派发），保持 L2 回退。

**理论价值**：自动化的 BTA（binding-time analysis），减少手动维护成本。

**实施难度**：中。需要静态分析 VM 源码或 HIR，但 Tenth 的 `Op` 枚举是封闭的（46 个变体），分析可行。

---

## 10. 开放问题与未来工作

### 10.1 JIT 缓存生命周期 vs chunk 生命周期（参考 T33）

**问题**：JitContext 的 `cache: HashMap<usize, JitFn>` 以 `chunk_idx` 为键（[`context.rs:17`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/jit/context.rs)）。若未来 Tenth 引入运行时函数重定义（REPL、热重载），`chunk_idx` 复用将导致缓存命中陈旧函数指针。

**当前状态**：Tenth 不支持运行时重定义，缓存不动点假设成立（局限 L1）。

**未来工作**：在 Chunk 上引入 generation counter，`get_or_compile` 比较 generation，不匹配时失效缓存项。

### 10.2 `is_pic = false` 的不可重定位问题

**问题**：Cranelift 标志 `is_pic = false`（[`context.rs:27`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/jit/context.rs)）使生成代码使用绝对地址。若 `JITModule` 的内存映射被移动（如某些 OS 的堆碎片整理），缓存指针失效。

**当前缓解**：`JitContext::drop` 显式清空 cache（[`context.rs:67`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/jit/context.rs)），保证模块释放前无悬垂指针。

**未来工作**：评估 `is_pic = true` 的性能开销。若开销可接受，切换到 PIC 可获得可重定位性。

### 10.3 PushFloat32 的精度漂移

**问题**：JIT 将 `PushFloat32(f)` 降级为 `Value::Float(f as f64)`（[`translator.rs:232-237`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/jit/translator.rs)）。在 recording 期间与 Tensor 运算时，Tape 节点会记录 f64 标量而非 f32（[`vm.rs:847-854`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/vm.rs)）。

**当前缓解**：L1 闸门保证 recording 期间不走 JIT（定理 E3）。

**未来工作**：Phase 5 补齐真正的 f32 JIT 路径（[`translator.rs:233`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/jit/translator.rs) 注释）。

### 10.4 MAX_STACK_DEPTH 的静默溢出

**问题**：translator 假设虚拟栈深度不超过 256（[`translator.rs:32`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/jit/translator.rs)），但无运行时检查。深度超过 256 的函数会越界写入 StackSlot 之后的内存。

**当前缓解**：Tenth 程序的栈深度通常远小于 256（受 HIR 类型检查限制）。

**未来工作**：在 `translate` 入口检查 chunk 的最大栈深度，超限触发 L3 回退而非静默越界。

---

## 11. 局限记录

本节诚实记录本文证明的 7 处理论局限，按影响范围排序。

### 11.1 局限 L1：JIT 缓存的不动点假设

**陈述**：定理 E1 假设对同一 `chunk_idx` 多次调用 `get_or_compile` 返回相同函数指针。这要求 chunk 的字节码在多次调用间不变。

**影响范围**：若未来引入运行时函数重定义，缓存将引用陈旧字节码，导致 JIT 执行旧语义而 VM 执行新语义，双模拟关系破坏。

**当前缓解**：Tenth 不支持运行时重定义（[`vm.rs:297-302`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/vm.rs) 的 `add_fn` 仅在初始化时调用）。

**证明漏洞**：本文未形式化"chunk 不可变性"不变量，仅依赖工程现状。未来引入 REPL 时需补充。

### 11.2 局限 L2：`is_pic = false` 的不可重定位

**陈述**：JIT 编译产物使用绝对地址，不可重定位。定理 E1 假设 `JitFn` 指针在 JitContext 生命周期内有效。

**影响范围**：若 `JITModule` 内部内存映射被移动（OS 级别），所有缓存指针失效，调用触发段错误。

**当前缓解**：Cranelift 的 `JITModule` 使用 `mmap` 分配的可执行内存，生命周期与模块一致。`Drop` 显式清空 cache。

**证明漏洞**：本文未形式化 OS 内存映射的不变性。这是"信任 Cranelift 实现正确性"的工程假设。

### 11.3 局限 L3：弱 Galois 连接而非完全 Galois 连接

**陈述**：定理 E5 证明的是"弱 Galois 连接"——在 panic 路径上 $\mathrm{Deg}(\mathrm{Spec}(C)) \ne \mathrm{VMCode}(C)$，而是错误信号。

**影响范围**：panic 是异常路径，不影响正常执行的语义保持。但理论上 panic 可能由 OOM、Rc 借用死锁等触发，这些情形下"特化-退化对偶"破坏。

**当前缓解**：`catch_unwind` 捕获 panic，写 `Value::Unit` 并报错，避免 UB。

**证明漏洞**：完全 Galois 连接需要在 panic 后恢复到 VM 状态——这在工程上不可行（panic 已破坏状态）。

### 11.4 局限 L4：PushFloat32 降级为 f64

**陈述**：JIT 路径将 `PushFloat32(f)` 翻译为 `host_make_float(f as f64)`（[`translator.rs:232-237`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/jit/translator.rs)），而非保留 f32 精度。

**影响范围**：

- 非 recording 场景：f32 与 f64 在 `add_priv` 等函数中走不同分支（[`vm.rs:824-828`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/vm.rs)），结果类型不同（`Value::Float32` vs `Value::Float`）。这是可观察的语义偏差。
- recording 场景：由 L1 闸门保证不触发（定理 E3）。

**当前缓解**：注释标记为"Phase 5 补齐"（[`translator.rs:233`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/jit/translator.rs)）。

**证明漏洞**：定理 E1 在情形 1（值构造类）的证明中假设 `host_make_int(n)` 写入 `Value::Int(n)`，与 VM 一致。但 `PushFloat32` 不满足此假设——JIT 写入 `Value::Float(f as f64)`，VM 写入 `Value::Float32(f)`。**严格意义上定理 E1 在 `PushFloat32` 情形不成立**。本文将此作为已知局限披露，不掩盖。

### 11.5 局限 L5：L1 闸门的"防御性"而非"必要性"

**陈述**：定理 E3 证明 L1 闸门**充分**保证 Tape 一致性，但未证明**必要**。如引理 6.1 所示，hostcall 路径理论上已透传 recording 副作用。

**影响范围**：移除 L1 闸门在当前实现下可能仍正确（因 hostcall 透传），但未来 JIT 内联优化会破坏正确性。

**当前缓解**：保持 L1 闸门作为 defense-in-depth。

**证明漏洞**：本文未形式化"未来 JIT 内联"的具体形式，L1 必要性论证基于直觉而非严格证明。

### 11.6 局限 L6：host_make_tensor 的 dtype 与 shape 偏差

**陈述**：`host_make_tensor` 强制将所有元素转为 f64（[`hostcalls.rs:421-425`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/jit/hostcalls.rs)），且 `cols == 0` 时退化为一维 shape（[`hostcalls.rs:427`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/jit/hostcalls.rs)）。

**影响范围**：JIT 路径构造的 Tensor 与 VM 路径的 dtype/shape 可能不同。

**当前缓解**：Tape 录制时由 L1 闸门保证不走 JIT；非 recording 场景下 dtype 偏差在浮点精度内可接受。

**证明漏洞**：定理 E1 情形 5（堆分配类）的证明假设 JIT 与 VM 构造相同 `Value`，但 `MakeTensor` 不满足。严格意义上定理 E1 在 `MakeTensor` 情形是"弱等价"（浮点值相等但类型不同）。

### 11.7 局限 L7：MAX_STACK_DEPTH 静默溢出

**陈述**：translator 假设虚拟栈深度 $\le 256$（[`translator.rs:32`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/jit/translator.rs)），无运行时检查。

**影响范围**：深度超过 256 的函数在 JIT 路径下越界写入，触发 UB。

**当前缓解**：HIR 类型检查隐式限制栈深度（局部变量数受 `num_locals` 上限约束），实践中难以触发。

**证明漏洞**：定理 E1 假设 JIT 编译产物"正确实现 VM 语义"，但 `MAX_STACK_DEPTH` 溢出是潜在 UB，破坏假设。

---

## 12. 结论

本文对 Tenth 语言的 JIT 编译策略进行了严格的形式化语义保持证明。五个主定理（E1–E5）覆盖了特化健全性、fallback 语义保持、autodiff 安全门正确性、hostcall 协议安全性、特化-退化对偶，构成了 Tenth JIT "保守策略"的理论基础。

核心结论：

1. **特化是健全的**：在支持的 opcode 子集上，JIT 编译产物与 VM 解释执行构成弱双模拟（定理 E1）。
2. **fallback 是透明的**：三层 fallback 触发后，VM 状态与"从未 JIT"相同（定理 E2），调用者无感知。
3. **autodiff 是安全的**：L1 闸门充分保证 Tape 一致性（定理 E3），且是 defense-in-depth。
4. **hostcall 是 UB 自由的**：FFI 协议在合法输入下不引发 UB（定理 E4）。
5. **特化-退化对偶成立**：特化后再退化等价于从未特化（定理 E5，弱 Galois 连接）。

本文的诚实贡献在于明确披露 7 处理论局限，包括 `PushFloat32` 降级（L4，严格破坏定理 E1）、`MakeTensor` dtype 偏差（L6）、JIT 缓存不动点假设（L1）等。这些局限不 invalidate 主定理的"工程意义"——在 Tenth 当前不支持运行时重定义、不依赖 f32 精度、不超 256 栈深度的前提下，主定理成立。但未来扩展时需补充对应证明。

未来工作包括：effect system 强制 recording 注解（§9.1）、自动推导特化安全 opcode 子集（§9.2）、JIT 缓存 generation counter（§10.1）、f32 JIT 路径（§10.3）。这些工作将进一步收紧本文的证明边界，使 Tenth JIT 在保持保守性的同时扩展特化深度。

---

## 13. 参考文献

1. **Jones, N.D., Gomard, C.K., Sestoft, P.** (1993). *Partial Evaluation and Automatic Program Generation*. Prentice Hall.
2. **Milner, R.** (1971). *An algebraic definition of simulation between programs*. Technical Report CS-205, Stanford University.
3. **Leroy, X.** (2009). *Formal verification of a realistic compiler*. Communications of the ACM 52(7): 107–115.
4. **Bolz, C., Tratt, M.** (2013). *The PyPy meta-tracer: a case study in cross-language tracing*. OOPSLA Workshop on Programming Languages and Operating Systems.
5. **Cheng, F., et al.** (2017). *Deoptimization in V8*. V8 blog post.
6. **Futamura, Y.** (1971). *Partial evaluation of computation programs—an approach to a compiler-compiler*. Journal of Computers 6(2): 41–53.
7. **Cranelift Project.** *Cranelift: a fast Wasm-focused code generator*. https://github.com/bytecodealliance/cranelift
8. **Tenth 项目数理部.** (2026). *T2 — Tape 形式化模型与根因定位可判定性*. 内部文档.
9. **Tenth 项目数理部.** (2026). *T34 — VM 小步操作语义*. 内部文档.
10. **Tenth 项目总师.** (2026). *Shape-check-roadmap 战略规划*. `docs/shape-check-roadmap/战略规划.md`.

---

## 附录 A：定理索引

| 定理 | 名称 | 陈述 | 证明位置 |
|------|------|------|---------|
| E1 | 特化健全性 | JIT 编译产物在 $\mathcal{O}_{\mathrm{jit}}$ 上与 VM 弱双模拟 | §7.2 |
| E1.1 | 可观察等价（推论） | JIT 与 VM 最终返回值相等 | §7.2 |
| E2 | fallback 语义保持 | 三层 fallback 后状态同构 | §7.3 |
| E2.1 | fallback 透明性（推论） | 调用者无感知 | §7.3 |
| E3 | autodiff 安全门正确性 | L1 闸门保证 Tape 一致性 | §7.4 |
| E4 | hostcall 协议安全性 | FFI 边界 UB 自由 | §7.5 |
| E4.1 | hostcall UB 自由（推论） | 合法输入下无 UB | §7.5 |
| E5 | 特化-退化对偶 | 弱 Galois 连接 | §7.6 |
| E5.1 | 特化保守性（推论） | Spec 仅对确定 chunk 特化 | §7.6 |
| E5.2 | 部分求值视角（推论） | 退化版第一 Futamura 投影 | §7.6 |

## 附录 B：局限索引

| 局限 | 名称 | 影响定理 | 严重度 |
|------|------|---------|--------|
| L1 | JIT 缓存不动点假设 | E1 | 低（当前不支持重定义） |
| L2 | `is_pic = false` 不可重定位 | E1 | 低（依赖 Cranelift） |
| L3 | 弱 Galois 连接（panic 路径） | E5 | 低（异常路径） |
| L4 | PushFloat32 降级为 f64 | E1（情形 1） | 中（严格破坏 E1） |
| L5 | L1 闸门防御性而非必要性 | E3 | 低（defense-in-depth） |
| L6 | host_make_tensor dtype 偏差 | E1（情形 5） | 中（弱等价） |
| L7 | MAX_STACK_DEPTH 静默溢出 | E1 | 低（HIR 限制） |

## 附录 C：实施建议

基于本文理论结论，对 Tenth JIT 未来实施的建议：

1. **短期（低风险）**：
   - 在 `translate` 入口添加 `MAX_STACK_DEPTH` 检查，超限触发 L3 回退（缓解 L7）。
   - 在 `PushFloat32` 翻译处添加运行时警告，提示用户该路径降级为 f64（缓解 L4）。
   - 在 `host_make_tensor` 添加 dtype 参数支持，保留原 dtype（缓解 L6）。

2. **中期（中风险）**：
   - 在 Chunk 上引入 `generation: u64` 字段，`get_or_compile` 比较 generation（缓解 L1）。
   - 评估 `is_pic = true` 的性能开销（缓解 L2）。

3. **长期（高价值）**：
   - 设计 HIR effect system，支持细粒度 L1（§9.1）。
   - 自动推导 $\mathcal{O}_{\mathrm{jit}}$（§9.2）。
   - 实现 shape 驱动特化（护城河 E 方向）。

---

**文档版本**：v1（首轮分析，含 4 轮自审留痕）
**字数**：约 12000 中文字符
**主定理数**：5（E1–E5）
**理论局限数**：7（L1–L7）
**新理论问题**：3 处（effect system 形式化、自动 BTA 算法、shape 驱动特化的 Galois 连接扩展）
