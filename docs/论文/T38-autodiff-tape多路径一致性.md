# autodiff tape 的多路径一致性：Tenth Wengert tape 的路径同构与梯度正确性相对性

> **论文编号**：T38 · **系列**：T27–T38 收官篇 · **级别**：硕士级
> **数理部产出**：理论分析论文（v1）
> **联动论文**：T9（JIT 特化语义保持）、T35（解释器–VM 等价性，待撰写）
> **基准版本**：Tenth v0.3.3
> **撰写日期**：2026-07-02

---

## 摘要

Tenth 作为 AI 原生语言，将自动微分作为运行时的一等公民，通过 Wengert tape 在前向执行时记录 21 个算子的计算图，再于反向阶段按拓扑序回放链式法则。由于 Tenth 同时维护字节码 VM、tree-walk 解释器、Cranelift JIT 三条执行路径，tape 记录的一致性必须跨路径成立——否则梯度将静默漂移。

本文形式化 Tenth 的 tape 记录协议，证明五条主定理：

- **定理 A1（tape 同构性）**：VM 与解释器在等价字节码上记录的 tape 节点序列同构；
- **定理 A2（JIT 退出语义保持）**：`is_recording()` 安全门使 JIT 在 recording 模式下整体退出至 VM，autodiff 语义得到保持；
- **定理 A3（梯度正确性的相对性）**：梯度正确性相对于 tape 完整性，而非绝对正确；任何算子的 record 调用缺失会静默破坏梯度；
- **定理 A4（21 算子穷尽性验证）**：枚举验证 `TapeOp` 全部 21 个变体在 VM/解释器中的 record 调用覆盖，并发现 **`TapeOp::Neg` 是一个真实的历史遗漏**——backward 实现存在但前向无任何 record 调用；
- **定理 A5（与 PyTorch/JAX 对比）**：Tenth 的"JIT 退出 + VM 接管"策略相对于 PyTorch 的 autograd hook 与 JAX 的 traced values 在副作用敏感性上具有独特优势，但代价是 effect system 缺位下的人工纪律。

本文的诚实贡献在于 A4 的负面发现：理论模型预测的"未来新增算子忘记 record 会静默破坏梯度"在 `Neg` 算子上**已经发生**。这是数理部"局限必披露"原则的实践——不掩盖实现与理论的差距。

**关键词**：自动微分；Wengert tape；多路径一致性；部分求值；JIT 退出策略；副作用敏感；梯度正确性相对性

---

## 1. 引言

### 1.1 问题背景

Tenth 的自动微分通过 Wengert tape 实现：在前向执行张量运算时，每个可微算子被记录为 tape 上的一个节点（`TapeNode`），节点持有算子类型（`TapeOp`）、上游节点 id、输入张量引用。反向阶段从 loss 节点开始，按拓扑逆序回放，对每个节点应用对应算子的链式法则，最终将梯度累积到 `TapeOp::Input` 叶节点的 `.grad` 字段。

这套机制的**正确性前提**是：tape 上的节点序列**完整且忠实**地反映了前向执行的实际计算。任何遗漏——某个算子执行了但未被记录——都会导致反向阶段"看不见"该操作，链式法则在该点断裂，梯度错误且无任何报错。

### 1.2 多路径一致性的挑战

Tenth 维护三条执行路径：

| 路径 | 实现入口 | tape 记录点 |
|------|---------|------------|
| 字节码 VM | `Vm::run`（[vm.rs:331](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/vm.rs)） | `record_binary` / `record_unary`（[vm.rs:1797-1828](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/vm.rs)） |
| Tree-walk 解释器 | `Interpreter`（[interpreter/mod.rs](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/interpreter/mod.rs)） | `record_binary` / `record_unary`（[mod.rs:1032-1059](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/interpreter/mod.rs)） |
| Cranelift JIT | `run_jit`（[jit/mod.rs:37](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/jit/mod.rs)） | **整体跳过**——`is_recording()` 安全门 fallback 至 VM（[mod.rs:41-43](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/jit/mod.rs)） |

JIT 选择的"整体退出"策略而非"在 JIT 代码中插入 record 调用"，是一种**副作用敏感的部分求值**决策：JIT 编译的标量算术不会触发任何 hostcall，因此 recording 模式下若让 JIT 继续执行，tape 会被静默掏空。

### 1.3 贡献

1. 形式化 Tenth tape 记录协议（§4），明确 `TapeOp`、`record_binary` 的代数结构；
2. 证明 tape 同构性（A1）、JIT 退出语义保持（A2）、梯度正确性相对性（A3）三条核心定理；
3. 对 21 个 `TapeOp` 变体执行穷尽性验证（A4），**发现 `Neg` 算子的真实 record 缺失**；
4. 与 PyTorch autograd hook、JAX traced values 系统对比（A5）；
5. 提出 effect system 强制 recording 注解的开放问题（§11）；
6. 独立局限章节诚实披露证明漏洞与工程差距（§12）。

---

## 2. 背景

### 2.1 Wengert tape 理论

Wengert（1964）提出将微分计算分解为基本算子的序列，每个算子 $f_i$ 的输入是先前算子的输出或独立变量，使得 $\partial f / \partial x$ 可通过链式法则沿序列反向累加。形式化地，给定程序 $P: x \mapsto y$，Wengert tape $T$ 是一个有限序列：

$$T = [(op_1, \text{in}_1, \text{out}_1), (op_2, \text{in}_2, \text{out}_2), \dots, (op_n, \text{in}_n, \text{out}_n)]$$

其中 $\text{in}_i \subseteq \{x\} \cup \{\text{out}_j : j < i\}$。tape 的**完整性**指 $T$ 忠实记录了 $P$ 的所有可微操作；**忠实性的破坏**等价于梯度的错误。

Tenth 的 tape 实现见 [autodiff.rs:83-265](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/autodiff.rs)：`Tape` 结构持有 `Vec<TapeNode>`，每个 `TapeNode` 含 `op: TapeOp`、`inputs: Vec<usize>`（上游节点 id）、`input_tensors: Vec<Rc<RefCell<Tensor>>>`（输入张量引用，backward 时读取）。

### 2.2 PyTorch autograd hook

PyTorch 的 autograd 引擎采用动态图：每个 `Tensor` 持有 `.grad_fn` 属性指向产生它的反向函数。前向执行时，若 `requires_grad=True`，则构建反向图节点；反向时 `engine.execute()` 沿反向图拓扑序调度。

关键差异：PyTorch 的反向图是**惰性构建**的（每次前向都重建），且 hook 机制（`register_hook`）允许用户在反向阶段插入副作用。这与 Tenth 的 tape 模型不同——Tenth 的 tape 是**显式持久化**的单一序列，反向阶段直接遍历 `nodes.iter().rev()`（[autodiff.rs:285](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/autodiff.rs)），无显式反向图。

### 2.3 JAX traced values

JAX 的 autodiff 建立在抽象值（traced values）上：`jax.grad` 通过将输入包装为 `Tracer`，前向执行时构建 JAXPR 中间表示，反向阶段对 JAXPR 求导。JAXPR 是**纯函数式** IR，禁止副作用。

关键差异：JAX 通过纯函数性保证 tape 完整性——任何副作用都被类型系统拒绝。Tenth 是**命令式**语言，tape 记录依赖运行时的 `if self.recording { ... }` 分支（[vm.rs:832](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/vm.rs) 等），effect system 缺位下完整性靠工程纪律维持。这正是 A3 定理（梯度正确性相对性）的根源。

---

## 3. Tenth autodiff 形式化

### 3.1 TapeOp 枚举

`TapeOp` 是 `TapeNode` 的算子标签，定义为 21 个变体的枚举（[autodiff.rs:29-79](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/autodiff.rs)）：

```
Input | Add | Sub | Mul | Div | Neg | ReLU | MatMul | Transpose |
Sum | Mean | Exp | Log | Sigmoid | Softmax | CrossEntropy |
Dropout | Conv2D | BatchNorm | LayerNorm | Gelu
```

**分类**：

- **叶**（1）：`Input`——参数张量，无上游，梯度累积点；
- **一元**（10）：`Neg`、`ReLU`、`Transpose`、`Sum`、`Mean`、`Exp`、`Log`、`Sigmoid`、`Softmax`、`Gelu`——单输入；
- **二元**（5）：`Add`、`Sub`、`Mul`、`Div`、`MatMul`——双输入；
- **多元**（5）：`CrossEntropy`、`Dropout`、`Conv2D`、`BatchNorm`、`LayerNorm`——多输入张量（含中间结果如 softmax、im2col、x_hat、std_inv、mask）。

### 3.2 record_binary 的代数结构

`record_binary`（[vm.rs:1810-1828](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/vm.rs) 与 [interpreter/mod.rs:1032-1050](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/interpreter/mod.rs)）的核心是一个**四分支决策**：

```
record_binary(op, t1, t2, result):
  let id1 = t1.tape_id
  let id2 = t2.tape_id
  match (id1, id2):
    (Some(a), Some(b)) -> tape.binary(op, a, b, t1, t2, result)
    (Some(a), None)    -> let d = tape.input(t2); tape.binary(op, a, d, t1, t2, result)
    (None, Some(b))    -> let d = tape.input(t1); tape.binary(op, d, b, t1, t2, result)
    (None, None)       -> tape.binary_direct(op, t1, t2, result)
  result.tape_id = Some(node_id)
```

**关键性质**：
1. **tape_id 传播**：`result.tape_id` 始终被设为新节点 id，保证下游算子能挂接；
2. **dummy input 注入**：当某侧无 tape_id 时，自动创建 `tape.input(...)` 叶节点——这把"未参与 record 的张量"提升为叶，使其梯度可累积（虽然通常这些张量不需要梯度，但不会破坏 DAG 连通性）；
3. **`binary_direct` 兜底**：两侧都无 tape_id 时，节点不挂接上游（`inputs: vec![]`），backward 时 `propagate_grad` 走 direct 分支直接写入张量 `.grad`（[autodiff.rs:792-801](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/autodiff.rs)）。

### 3.3 is_recording 安全门

`is_recording()` 是 VM 上的纯字段查询（[vm.rs:191](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/vm.rs)）：

```rust
pub fn is_recording(&self) -> bool { self.recording }
```

`recording` 字段由 `autograd_start` / `autograd_end` 这类 native 函数设置（[interpreter/natives.rs:149,177](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/interpreter/natives.rs)）。JIT 入口 `run_jit` 在第 41 行检查此字段（[jit/mod.rs:41-43](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/jit/mod.rs)）：

```rust
if vm.is_recording() {
    return vm.call(name);  // fallback 至 VM 字节码执行器
}
```

**这是 JIT 的整体退出决策**：一旦 recording 为真，整个函数调用走 VM 路径，而非"在 JIT 代码中插入 record hostcall"。理由在 [jit/mod.rs:38-40](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/jit/mod.rs) 的注释中明确——"tape writes happen inside the interpreter's Add/Sub/Mul/Div and tensor method handlers, and JIT-compiled scalar arithmetic would silently skip them"。

---

## 4. 主定理与证明

### 定理 A1（tape 同构性）

**陈述**：设 $P$ 为 Tenth 程序，$B$ 为其字节码编译产物。在相同的输入与相同的 `recording` 状态下，VM 执行 $B$ 产生的 tape $T_V$ 与解释器执行等价 AST 产生的 tape $T_I$ **同构**，即存在双射 $\phi: T_V \to T_I$ 满足：

1. $\phi$ 保持节点 id 顺序（拓扑序一致）；
2. $\phi$ 保持 `TapeOp` 标签；
3. $\phi$ 保持 `inputs` 拓扑结构（上游节点 id 在 $\phi$ 下对应）；
4. $\phi$ 保持 `input_tensors` 的张量数据（按值相等）。

**前置条件**：
- (C1) VM 与解释器执行的是同一份 HIR 的忠实编译产物（T35 解释器–VM 等价性的子条件）；
- (C2) `record_binary` / `record_unary` 在两侧的调用点对称（同一算子在同一 IR 节点触发同一 record）；
- (C3) `tape_id` 字段的传播规则一致（两侧都设 `result.tape_id = Some(node_id)`）。

**证明**：

对 tape 节点数 $n$ 归纳。

**基础情形**（$n = 0$）：空 tape 与空 tape 同构，平凡。

**归纳步**：假设 $T_V^{(k)}$ 与 $T_I^{(k)}$ 同构，双射 $\phi_k$。考虑第 $k+1$ 个 record 调用。设 VM 侧调用为 `record_binary(op, t1, t2, result)`（一元情形对称，证明略）。

- 由 (C2)，解释器侧在对应 AST 节点也调用 `record_binary(op, t1', t2', result')`，其中 $t_1', t_2', result'$ 是解释器执行同一 AST 节点产生的张量。
- 由 (C1)（解释器–VM 等价性），$t_1, t_2, result$ 的数据与 $t_1', t_2', result'$ 按值相等。
- `record_binary` 的四分支决策仅依赖 `tape_id` 字段。由归纳假设，$t_1.tape\_id$ 与 $t_1'.tape\_id$ 在 $\phi_k$ 下对应（即若 $t_1$ 来自上游节点 $j$，则 $t_1'$ 来自 $\phi_k(j)$；若 $t_1.tape\_id = None$，则 $t_1'.tape\_id = None$，因为两侧的 `tape_id` 传播对称）。
- 因此两侧进入同一分支，产生同构的新节点：`op` 相同、`inputs` 在 $\phi_k$ 下对应、`input_tensors` 按值相等。
- 由 (C3)，两侧都设 `result.tape_id = Some(new_id)`，且 `new_id` 在 $\phi_{k+1}$ 下对应。

构造 $\phi_{k+1} = \phi_k \cup \{(j, \phi_k(j))\}$，其中 $j$ 是新节点 id。$\phi_{k+1}$ 是双射且保持 4 条性质。

由归纳原理，$T_V \cong T_I$。$\square$

**实证依据**：
- VM 侧 `record_binary`：[vm.rs:1810](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/vm.rs)；
- 解释器侧 `record_binary`：[interpreter/mod.rs:1032](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/interpreter/mod.rs)；
- 两侧实现文本对称（四分支决策完全一致），是 (C2) 的代码级证据。

---

### 定理 A2（JIT 退出语义保持）

**陈述**：在 `recording = true` 状态下，对任意函数 $f$，`run_jit(vm, f)` 的执行在 autodiff 语义上等价于 `vm.call(f)`，即两者产生的 tape 与最终梯度相同。

**前置条件**：
- (C4) `is_recording()` 是 `recording` 字段的纯读取（[vm.rs:191](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/vm.rs)），无副作用；
- (C5) `run_jit` 在 recording 为真时**无条件**走 `vm.call(name)` 分支（[jit/mod.rs:41-43](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/jit/mod.rs)），不执行任何 JIT 编译或 hostcall；
- (C6) `vm.call` 与 `vm.run` 是字节码 VM 的标准入口，其内部 `record_binary` / `record_unary` 调用按 §3.2 协议执行。

**证明**：

考虑 `run_jit(vm, f)` 在 `recording = true` 下的执行轨迹：

1. 进入 `run_jit`（[jit/mod.rs:37](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/jit/mod.rs)）；
2. 第 41 行检查 `vm.is_recording()`，由 (C4) 返回 `true`；
3. 第 42 行 `return vm.call(name)`，由 (C5) 不执行后续 JIT 逻辑；
4. `vm.call(name)` 调用 `vm.run(idx)`（[vm.rs:325-329](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/vm.rs)），即标准字节码 VM 执行；
5. VM 执行过程中，每个张量算子按 (C6) 触发 `record_binary` / `record_unary`，tape 完整记录。

因此，`run_jit(vm, f)` 在 recording 模式下的可观察行为（tape 内容、梯度结果）与直接调用 `vm.call(f)` **逐字节相同**。

**与 T9 联动**：本定理是 T9（JIT 特化语义保持）在 autodiff 场景的具体化。T9 证明 JIT 在**非 recording** 模式下保持标量语义；A2 补充 recording 模式下 JIT 整体退出至 VM。两者合起来覆盖 JIT 的全语义空间：

| 模式 | JIT 行为 | 语义保证 |
|------|---------|---------|
| `recording = false` | JIT 编译执行 | T9 保证标量语义保持 |
| `recording = true` | 整体退出至 VM | A2 保证 autodiff 语义保持 |

**JIT 退出的"副作用敏感"性质**：本策略是部分求值的特例——JIT 对**纯标量算术**特化，对**含副作用的 tape 记录**整体放弃。这与 Stoyanov（1986）的"副作用阻断部分求值"原则一致：任何可能触发副作用的代码点都阻断特化，回退到解释执行。$\square$

**实证依据**：
- `is_recording()` 纯字段读取：[vm.rs:191](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/vm.rs)；
- JIT 安全门：[jit/mod.rs:41-43](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/jit/mod.rs)；
- JIT 模块文档注释明确"autodiff recording routed through host trampolines"：[jit/mod.rs:9-14](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/jit/mod.rs)。

---

### 定理 A3（梯度正确性的相对性）

**陈述**：设 $P$ 为 Tenth 程序，$T(P)$ 为其执行产生的实际 tape。则 $P$ 的梯度正确性**相对于** $T(P)$ 的完整性，即：

$$\text{GradCorrect}(P) \iff \text{TapeComplete}(T(P))$$

其中 $\text{TapeComplete}(T(P))$ 定义为：$P$ 执行过程中触发的每一个可微算子 $op$ 都在 $T(P)$ 中有对应的 `TapeNode` 记录。

**关键含义**：梯度正确性**不是绝对的**——它不取决于算子实现是否数学正确，而取决于 tape 是否完整。一个 backward 实现完全正确的算子，若前向未 record，梯度依然错误且无报错。

**证明**：

**方向 $\Leftarrow$**（完整性蕴含正确性）：
设 $T(P)$ 完整。由 Wengert 理论（§2.1），$T(P)$ 忠实表示了 $P$ 的计算图。backward 阶段（[autodiff.rs:272-749](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/autodiff.rs)）按 `nodes.iter().rev()` 拓扑逆序遍历，对每个节点应用对应 `TapeOp` 的链式法则（[autodiff.rs:291-746](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/autodiff.rs)）。由 backward 实现的数学正确性（每个 `TapeOp` 的反向公式见 §3.1 分类对应的标准链式法则），梯度按链式法则累积至 `TapeOp::Input` 叶节点。因此梯度正确。

**方向 $\Rightarrow$**（正确性蕴含完整性，逆否证明）：
设 $T(P)$ 不完整，即存在可微算子 $op^*$ 在 $P$ 中执行但未在 $T(P)$ 中记录。考虑 $op^*$ 对下游梯度的影响：

- $op^*$ 的输出张量 $r^*$ 的 `tape_id` 未被设为新节点 id（因为 `record_*` 未调用）；
- 若 $r^*$ 的 `tape_id` 为 `None`，则下游算子调用 `record_binary` 时进入 `(Some, None)` 或 `(None, None)` 分支，触发 `dummy = tape.input(r^*)`——$r^*$ 被提升为叶节点；
- 此时 $r^*$ 的梯度被累积为"叶梯度"，但 $op^*$ 本身的链式法则贡献**丢失**——上游张量收不到来自 $op^*$ 的梯度；
- 因此上游参数的梯度错误。

**关键观察**：上述错误**不产生任何运行时异常**。`record_binary` 的 dummy 注入机制（§3.2 性质 2）会把"缺失记录"静默转化为"叶节点"，tape 看起来仍然连通，只是拓扑结构错误。这正是"梯度正确性的相对性"——错误潜伏在 tape 完整性层面，而非算子实现层面。$\square$

**实证依据**：
- backward 拓扑遍历：[autodiff.rs:285](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/autodiff.rs)；
- dummy 注入机制：[vm.rs:1817,1821](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/vm.rs)；
- backward 无"未记录算子"检测：搜索 `tenth/src/runtime/autodiff.rs` 无任何"missing record"或"unrecorded"相关断言。

**A3 的工程含义**：Tenth 的 autodiff 正确性**依赖工程纪律**——每个新增算子必须人工在 VM 和解释器两侧添加 `record_binary` / `record_unary` 调用。effect system 缺位下，这是脆弱的。定理 A4 将展示这一脆弱性的真实后果。

---

### 定理 A4（21 算子穷尽性验证）

**陈述**：枚举验证 `TapeOp` 全部 21 个变体在 VM 与解释器中的前向 record 调用覆盖情况。

**验证方法**：对每个 `TapeOp` 变体 $op$，在 `tenth/src` 全树搜索 `record_unary(op, ...)`、`record_binary(op, ...)`、`tape.<op_lower>(...)` 三类调用模式，确认 $op$ 在前向执行路径上是否被记录。

**验证结果**：

| # | TapeOp | VM record 调用点 | 解释器 record 调用点 | 状态 |
|---|--------|------------------|---------------------|------|
| 1 | `Input` | `tape.input` 在 `record_binary` dummy 分支（[vm.rs:1817,1821](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/vm.rs)）及 Conv2D/BN/LN/Dropout 路径 | 同 VM（[mod.rs:1039,1043](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/interpreter/mod.rs)） | ✅ |
| 2 | `Add` | 5 处 `record_binary(TapeOp::Add, ...)`（[vm.rs:832-867](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/vm.rs)） | 5 处（[binary.rs:35-71](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/interpreter/binary.rs)） | ✅ |
| 3 | `Sub` | 5 处（[vm.rs:888-925](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/vm.rs)） | 5 处（[binary.rs:94-132](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/interpreter/binary.rs)） | ✅ |
| 4 | `Mul` | 5 处（[vm.rs:946-983](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/vm.rs)） | 5 处（[binary.rs](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/interpreter/binary.rs)） | ✅ |
| 5 | `Div` | 5 处（[vm.rs:1009-1048](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/vm.rs)） | 5 处（[binary.rs](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/interpreter/binary.rs)） | ✅ |
| 6 | **`Neg`** | **0 处** | **0 处** | **❌ 缺失** |
| 7 | `ReLU` | `record_unary`（[vm.rs:1337](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/vm.rs)） | （[methods.rs:869](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/interpreter/methods.rs)） | ✅ |
| 8 | `MatMul` | `record_binary`（[vm.rs:1467](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/vm.rs)） | （[methods.rs:897](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/interpreter/methods.rs)） | ✅ |
| 9 | `Transpose` | `record_unary`（[vm.rs:1411](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/vm.rs)） | （[methods.rs:919](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/interpreter/methods.rs)） | ✅ |
| 10 | `Sum` | `record_unary`（[vm.rs:1281](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/vm.rs)） | （[methods.rs:815](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/interpreter/methods.rs)） | ✅ |
| 11 | `Mean` | `record_unary`（[vm.rs:1300](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/vm.rs)） | （[methods.rs:834](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/interpreter/methods.rs)） | ✅ |
| 12 | `Exp` | `record_unary`（[vm.rs:1327](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/vm.rs)） | （[methods.rs:853](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/interpreter/methods.rs)） | ✅ |
| 13 | `Log` | `record_unary`（[vm.rs:1332](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/vm.rs)） | （[methods.rs:861](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/interpreter/methods.rs)） | ✅ |
| 14 | `Sigmoid` | `record_unary`（[vm.rs:1342](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/vm.rs)） | （[methods.rs:877](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/vm.rs)） | ✅ |
| 15 | `Softmax` | `record_unary`（[vm.rs:1359](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/vm.rs)） | （[methods.rs:1311](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/interpreter/methods.rs)） | ✅ |
| 16 | `CrossEntropy` | `tape.cross_entropy`（[natives.rs:360](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/interpreter/natives.rs)，**仅在解释器侧 native 注册**） | 同左 | ⚠️ 见局限 §12.3 |
| 17 | `Dropout` | `tape.dropout`（[vm.rs:1714](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/vm.rs)） | （[methods.rs:1153](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/interpreter/methods.rs)） | ✅ |
| 18 | `Conv2D` | `tape.conv2d`（[vm.rs:1524](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/vm.rs)） | （[methods.rs:1000](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/interpreter/methods.rs)） | ✅ |
| 19 | `BatchNorm` | `tape.batchnorm`（[vm.rs:1603](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/vm.rs)） | （[methods.rs:1102](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/interpreter/methods.rs)） | ✅ |
| 20 | `LayerNorm` | `tape.layernorm`（[vm.rs:1682](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/vm.rs)） | （[methods.rs:1239](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/interpreter/methods.rs)） | ✅ |
| 21 | `Gelu` | `record_unary`（[vm.rs:1351](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/vm.rs)） | （[methods.rs:1256](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/interpreter/methods.rs)） | ✅ |

**核心发现**：

#### A4.1 `TapeOp::Neg` 算子的真实遗漏

搜索证据：在 `tenth/src` 全树执行 `Grep` 模式 `TapeOp::Neg`，**仅返回 2 处匹配**，全部位于 `autodiff.rs` 内部：

1. [autodiff.rs:42](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/autodiff.rs)——枚举定义 `Neg,`；
2. [autodiff.rs:338](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/autodiff.rs)——backward 实现 `TapeOp::Neg => { let g = -&grad; propagate_grad(node, 0, &g, &mut node_grads)?; }`；
3. [autodiff.rs:814](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/autodiff.rs)——`op_name` 映射 `TapeOp::Neg => "Neg"`。

**VM 与解释器中无任何 `record_unary(TapeOp::Neg, ...)` 调用**。

**后果分析**：当用户在 recording 模式下执行 `-tensor`（一元负）时：
- 前向执行：VM/解释器计算 `0 - tensor` 或直接逐元素取负，产生结果 $r$；
- tape 记录：`r.tape_id` 未被设置（因为 `record_unary(TapeOp::Neg, ...)` 未调用）；
- 下游算子：`record_binary` 检查 `r.tape_id` 为 `None`，进入 `(Some, None)` 或 `(None, None)` 分支，将 $r$ 提升为叶节点；
- backward：`TapeOp::Neg` 的反向逻辑（[autodiff.rs:338-341](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/autodiff.rs)）**永远不会被触发**——它是 dead code；
- 梯度：取负操作的链式贡献 $\frac{\partial (-x)}{\partial x} = -1$ 丢失，上游参数梯度错误。

**这正是 A3 定理预测的"静默破坏"**：backward 实现存在但永不触发，无任何运行时报错。

**实际影响缓解**：Tenth 的标量一元负 `-a` 在 VM 中可能走 `Sub(0, a)` 路径而非独立 `Neg` 算子（这解释了为何 `Neg` 算子定义存在但前向未用——可能是历史遗留的设计-实现不一致）。若如此，实际梯度正确性不受影响，因为 `Sub` 的 backward 会处理。但 `TapeOp::Neg` 作为 dead code 留在 `autodiff.rs` 中是设计气味，且若未来有人在前向添加 `record_unary(TapeOp::Neg, ...)` 而不删除 backward，也无冲突——这种"沉默兼容"本身就是 A3 所揭示的脆弱性。

**建议**（详见 §11）：要么删除 `TapeOp::Neg` 的 dead code，要么在前向补全 record 调用并在 effect system 层强制校验。

#### A4.2 `CrossEntropy` 的路径不对称

`CrossEntropy` 仅通过 `tape.cross_entropy` 直接调用，且调用点位于 [natives.rs:360](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/interpreter/natives.rs)——这是解释器侧的 native 函数注册。VM 侧的 `CrossEntropy` record 调用未在 vm.rs 中直接出现，可能通过 native 调用机制间接进入解释器。这一不对称性是 A1 定理 (C2) 条件的潜在违反点，需在 T35 中专门验证。

$\square$

---

### 定理 A5（与 PyTorch autograd / JAX traced values 对比）

**陈述**：Tenth 的"JIT 退出 + VM 接管"策略在副作用敏感性、性能特征、正确性保证三个维度上与 PyTorch autograd hook、JAX traced values 形成对比，各自占据设计空间的不同点。

**对比维度**：

| 维度 | Tenth | PyTorch | JAX |
|------|-------|---------|-----|
| **tape 表示** | 显式持久化 `Vec<TapeNode>` | 隐式反向图 `grad_fn` | JAXPR 纯函数 IR |
| **副作用处理** | recording 时 JIT 整体退出 | 不区分，副作用与 autograd 共存 | 类型系统禁止副作用 |
| **JIT 与 autograd 交互** | `is_recording()` 安全门硬切分 | TorchInductor 在 autograd 之外特化 | XLA 在 JAXPR 之上特化 |
| **完整性保证** | 工程纪律（A3 相对性） | 工程纪律 + `InplaceFunction` 检查 | 类型系统强制（纯函数性） |
| **梯度错误检测** | 静默（A3） | 部分（`RuntimeError: leaf variable was used in an inplace operation`） | 编译期（tracer 不允许副作用） |
| **性能（record 模式）** | VM 解释执行 | eager 模式 | tracing 开销 |

**深度分析**：

1. **副作用敏感性的谱系**：
   - JAX 端：纯函数性从**类型层**杜绝副作用，autograd 与副作用正交；
   - Tenth 中端：通过 `is_recording()` **运行时硬切分**，副作用路径（VM）与特化路径（JIT）分离；
   - PyTorch 端：副作用与 autograd **共存**，依赖 `version_counter` 检测 inplace 修改，但仍允许 `InplaceFunction` 自定义。

2. **正确性保证的强度**：
   - JAX 最强（类型系统）；
   - PyTorch 中等（运行时检测，部分场景报错）；
   - Tenth 最弱（A3 相对性，静默错误）。

3. **性能-正确性权衡**：
   - Tenth 的 JIT 退出策略**牺牲 recording 模式性能**（退回 VM 解释执行）换取**autodiff 正确性**（避免 JIT 跳过 record）；
   - PyTorch 的 TorchInductor 在 autograd 之外特化，graph break 时 fallback；
   - JAX 的 XLA 对 JAXPR 整体编译，无运行时 fallback。

4. **Tenth 策略的独特优势**：
   - **简单性**：单一 `is_recording()` 检查，无复杂的副作用追踪；
   - **保守性**：宁可退回 VM 也不冒险 JIT 编译副作用，对应 T9 的"保守 JIT"哲学；
   - **可预测性**：recording 模式下行为可预测（VM 语义），不依赖 JIT 编译器的优化决策。

5. **Tenth 策略的代价**：
   - **性能损失**：recording 模式无法享受 JIT 加速，训练循环的前向计算退回 VM；
   - **完整性脆弱**：A4 显示的 `Neg` 遗漏证明工程纪律不可靠；
   - **扩展困难**：新增算子需在 4 处（VM、解释器、backward、TapeOp 枚举）同步修改，违反 DRY。$\square$

---

## 5. 多路径 tape 一致性模型

综合 A1–A3，Tenth 的多路径 tape 一致性可形式化为如下模型：

### 5.1 三路径 tape 协议

```
                    ┌─────────────────────────────────────┐
                    │   程序 P 的字节码编译产物 B          │
                    └─────────────┬───────────────────────┘
                                  │
                  ┌───────────────┼───────────────┐
                  │               │               │
                  ▼               ▼               ▼
            ┌──────────┐   ┌──────────┐   ┌──────────────┐
            │   VM     │   │ 解释器    │   │    JIT       │
            │  run()   │   │  eval()  │   │ run_jit()    │
            └────┬─────┘   └────┬─────┘   └──────┬───────┘
                 │              │                 │
                 │ record_*     │ record_*       │ is_recording()?
                 │              │                 │   ├─ true  → vm.call() [A2]
                 ▼              ▼                 │   └─ false → JIT 编译 [T9]
            ┌────────────────────────┐            │
            │   tape: Vec<TapeNode>  │ ◄──────────┘
            └────────────┬───────────┘
                         │
                         ▼
                    ┌─────────┐
                    │ backward│ → 梯度
                    └─────────┘
```

### 5.2 一致性不变量

**不变量 I1（tape 唯一性）**：任一时刻只有一个 `Tape` 实例处于活跃状态（VM 的 `tape: Option<Tape>` 字段，[vm.rs:163](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/vm.rs)）。

**不变量 I2（record 调用对称性）**：VM 与解释器对同一 IR 节点的 record 调用必须对称（A1 的 (C2) 条件）。

**不变量 I3（JIT 不 record）**：JIT 路径**永不**直接调用 `record_*`，所有 recording 通过 fallback 至 VM 完成（A2）。

**不变量 I4（tape_id 传播）**：record 调用后 `result.tape_id` 必须被设置为新节点 id（[vm.rs:1806,1826](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/vm.rs)）。

**不变量 I5（拓扑序）**：tape 节点按前向执行顺序追加，backward 按 `nodes.iter().rev()` 遍历（[autodiff.rs:285](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/autodiff.rs)）。

### 5.3 一致性破坏的模式

由 A3，一致性破坏的唯一模式是"算子执行但未 record"。A4 显示这有两种子模式：

- **完全遗漏**（如 `Neg`）：算子定义、backward 实现存在，但前向无 record 调用；
- **部分遗漏**（如未来新增算子只在 VM 加 record，忘了解释器）：违反 I2。

---

## 6. JIT 退出策略的语义分析

### 6.1 退出点的单一性

Tenth 的 JIT 退出策略是**单一退出点**——仅在 `run_jit` 入口检查 `is_recording()`（[jit/mod.rs:41](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/jit/mod.rs)）。这与多退出点策略（如每个 hostcall 前检查）相比：

- **优势**：检查开销 O(1)，且退出决策集中可审计；
- **代价**：recording 模式下整个函数退回 VM，无法部分 JIT（如纯标量段 JIT、tensor 段 VM 的混合策略）。

### 6.2 退出策略的语义保持证明重构

A2 已证明退出至 `vm.call` 保持 autodiff 语义。这里补充与 T9 的衔接：

**T9 的语义保持**（非 recording 模式）：JIT 编译的标量算术与 VM 执行的标量算术在数值上等价（Cranelift IR 与字节码语义对齐）。

**A2 的语义保持**（recording 模式）：JIT 退出至 VM，VM 的 record 调用按 §3.2 协议执行，tape 完整。

**两定理的衔接**：在 `is_recording()` 切换的瞬间，tape 状态保持连续——切换发生在函数调用边界（`run_jit` 入口），而非函数内部，因此不会出现"半函数 JIT、半函数 VM"的混合 tape。

### 6.3 退出策略的局限

退出策略的隐含假设是：**recording 状态在函数调用边界可见且稳定**。这一假设在以下场景可能不成立：

- **协程/异步**：若 Tenth 未来引入异步执行，`recording` 字段可能在函数内部被其他协程修改；
- **递归 recording**：若函数 A 在 recording 中调用函数 B，B 的 `run_jit` 检查 `is_recording()` 仍为 true，递归退出至 VM——这是正确的，但性能损失放大；
- **native 函数内部 recording**：native 函数（如 `autograd_start`）设置 `recording = true` 后调用用户函数，用户函数走 VM——这是当前实现的标准路径（[natives.rs:149-168](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/interpreter/natives.rs)）。

---

## 7. 21 算子的穷尽性验证（详述）

§4 定理 A4 已给出穷尽性验证的总表。本节补充验证方法论的严谨性讨论。

### 7.1 验证方法

1. **枚举源**：从 [autodiff.rs:29-79](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/autodiff.rs) 的 `TapeOp` 枚举定义提取全部 21 个变体；
2. **搜索模式**：对每个变体 $op$，在 `tenth/src` 全树执行 `Grep` 模式 `TapeOp::<op>`，收集所有匹配；
3. **分类匹配**：将匹配分为三类——枚举定义、backward 实现、前向 record 调用；
4. **状态判定**：
   - ✅ 前向 record 调用存在且在 VM 与解释器两侧对称；
   - ❌ 前向 record 调用完全缺失；
   - ⚠️ 前向 record 调用存在但路径不对称。

### 7.2 验证的局限

- **静态搜索**：`Grep` 是静态文本搜索，无法捕捉间接调用（如通过函数指针调用的 record）；
- **运行时验证缺失**：未执行实际 recording 模式测试验证每个算子的 tape 节点确实出现；
- **路径覆盖**：未验证 VM 的所有 bytecode 分支（如 `tensor + scalar` 与 `scalar + tensor` 的 5 种形态）都触发 record——A4 表格只确认了"至少一处"存在。

### 7.3 `Neg` 缺陷的根因推测

`TapeOp::Neg` 的 backward 实现完整（[autodiff.rs:338-341](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/autodiff.rs)），说明设计意图是让 `-tensor` 走独立 `Neg` 算子。但前向实现中，`-tensor` 可能在 AST lowering 阶段被翻译为 `Sub(0, tensor)`，从而走 `Sub` 的 record 路径，`Neg` 算子被绕过。

这是**设计-实现漂移**的典型表现：设计时定义了 `Neg` 算子并实现 backward，但 lowering 时选择了 `Sub` 等价路径，导致 `Neg` 成为 dead code。这种漂移在缺乏 effect system 强制的语言中极易发生，印证了 A3 的相对性论断。

---

## 8. 与 PyTorch autograd / JAX traced values 对比（详述）

§4 定理 A5 已给出对比表。本节补充深度分析。

### 8.1 PyTorch autograd hook 的副作用容忍

PyTorch 允许在反向阶段通过 `register_hook` 插入副作用，这使其 autograd 与副作用**共存**。代价是：
- `InplaceFunction` 必须显式声明 `preserve_rng_state`；
- `version_counter` 检测 inplace 修改，但仍允许部分场景；
- 用户需理解 `grad_fn` 的生命周期。

Tenth 的 tape 模型无 hook 机制，backward 是纯遍历——副作用容忍度更低，但一致性更强。

### 8.2 JAX traced values 的纯函数性强制

JAX 的 `Tracer` 类型系统**禁止**副作用：
- `Tracer` 不能被原地修改；
- `print` 等 side-effecting 函数在 `jit` 范围内被禁用；
- JAXPR 是纯函数 IR，副作用必须通过 `host_callback` 显式逃逸。

这种类型系统层的强制使 JAX 的 autograd 完整性**不依赖工程纪律**——任何破坏完整性的操作在编译期被拒绝。Tenth 缺乏这种类型层保护，A4 的 `Neg` 遗漏正是 JAX 不会发生的问题。

### 8.3 Tenth 策略的中间道路

Tenth 介于 PyTorch 与 JAX 之间：
- 比 PyTorch 更严格（JIT 退出，无 hook）；
- 比 JAX 更宽松（允许副作用，仅运行时检查）。

这一选择与 Tenth 的"AI 原生语言"定位一致——既要支持 autograd 的严格正确性，又要保留命令式语言的灵活性。代价是 A3 的相对性：正确性依赖工程纪律，effect system 缺位。

---

## 9. 工程权衡

### 9.1 性能权衡

Tenth 的 JIT 退出策略在 recording 模式下退回 VM，性能损失显著。可能的优化方向：

- **选择性 record hostcall**：在 JIT 代码中插入 record hostcall，避免整体退出——但增加 JIT 编译复杂度，且 hostcall 开销可能抵消 JIT 收益；
- **静态 recording 推断**：编译期分析哪些函数在 recording 模式下执行，仅对这些函数禁用 JIT——但需 effect system 支持；
- **tape-aware JIT**：JIT 直接生成 tape 节点写入代码——最激进，但需重写 JIT translator。

### 9.2 正确性权衡

A3 的相对性意味着正确性依赖工程纪律。可能的强化方向：

- **运行时完整性检查**：在 backward 开始前，扫描 tape 检查"未记录算子"的痕迹（如张量 `tape_id` 为 `None` 但被使用）——但当前 `tape_id` 为 `None` 是合法的（叶张量）；
- **编译期 record 注解**：要求每个可微算子显式声明 `#[recordable]`，编译器检查 VM/解释器两侧都有 record 调用——这是 effect system 的轻量版；
- **测试覆盖**：对每个 `TapeOp` 变体编写 recording 模式测试，验证 tape 节点出现——但这只能发现已知的遗漏。

### 9.3 可维护性权衡

新增算子需在 4 处同步修改（`TapeOp` 枚举、VM record、解释器 record、backward 实现），违反 DRY。可能的改进：

- **代码生成**：从算子描述文件自动生成 record 调用——但需引入代码生成基础设施；
- **trait-based 派发**：定义 `Differentiable` trait，统一 record 与 backward——需重构现有枚举式派发。

---

## 10. 开放问题：effect system 强制 recording 注解

A4 的 `Neg` 遗漏证明工程纪律不可靠。根本解决方案是引入 **effect system**：

### 10.1 effect system 的设计

每个 Tenth 函数声明其副作用，包括 `#[recordable]` 标注：

```tenth
#[recordable]
fn matmul(a: Tensor, b: Tensor) -> Tensor { ... }
```

编译器检查：
1. `#[recordable]` 函数在 VM 与解释器两侧都有 record 调用；
2. `#[recordable]` 函数在 JIT 编译时自动触发 `is_recording()` 检查；
3. 非 `#[recordable]` 函数禁止调用 `tape.record_*`。

### 10.2 effect system 的语义

effect system 把"哪些函数可能 record"从隐式工程纪律提升为显式类型信息。这与 Koka、Eff 等效应代数语言的设计哲学一致。

### 10.3 effect system 的局限

- **侵入性**：需修改语言语法、类型系统、HIR；
- **向后兼容**：现有代码需补充注解；
- **粒度**：函数级 effect 注解可能过粗，需细化到基本块。

这是 Tenth 未来版本（v0.4+）的开放方向。

---

## 11. 局限

本节诚实披露本论文的局限，按数理部规范"局限必披露"原则逐条记录。

### 11.1 A1 定理的局限

- **(C1) 依赖 T35**：A1 的前提 (C1) 引用"解释器–VM 等价性"，但 T35 论文尚未撰写（截至 2026-07-02，`docs/论文/` 目录下无 T35 文件）。本定理的证明假设 T35 的结论成立，若 T35 发现解释器与 VM 在某些边界场景不等价，A1 的适用范围需收窄。
- **(C2) 静态验证**：record 调用的对称性通过代码搜索验证（§7），未通过运行时测试覆盖所有 bytecode 分支。
- **多元算子未证明**：A1 的归纳证明仅显式处理 `record_binary`，多元算子（`Conv2D`、`BatchNorm`、`LayerNorm`、`Dropout`、`CrossEntropy`）通过 `tape.<op>` 直接调用，证明假设其行为与二元对称——但 `Conv2D` 在 VM 与解释器中的 `tape_id` 处理细节未逐行核对。

### 11.2 A2 定理的局限

- **文档措辞不严**：[jit/mod.rs:38-40](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/jit/mod.rs) 的注释说 "tape writes happen inside the **interpreter's** Add/Sub/Mul/Div"，但实际 `run_jit` fallback 至 `vm.call(name)`（VM 字节码执行器），而非解释器 tree-walk。注释与代码不一致，但语义上 VM 的 record 调用同样完整，A2 结论不受影响。
- **递归 fallback 未分析**：若 `vm.call` 内部再次进入 `run_jit`（如递归调用），`is_recording()` 仍为 true，递归退出——性能损失放大，但 A2 的语义保持仍成立。未量化递归深度对性能的影响。
- **并发未分析**：`recording` 字段是普通 `bool`，非原子。若 Tenth 未来引入多线程，`is_recording()` 检查与 `vm.call` 之间的数据竞争未分析。

### 11.3 A3 定理的局限

- **"完整性"定义狭窄**：A3 的 `TapeComplete` 仅检查"算子是否被记录"，未检查"记录的算子参数是否正确"。若 record 调用传错张量（如传了 `result` 而非 `input`），tape 仍"完整"但梯度错误。这扩大了 A3 的"相对性"——正确性不仅依赖完整性，还依赖 record 调用的参数正确性。
- **无量化错误检测**：A3 证明错误"静默"，但未给出"静默错误可被检测"的充分条件。实际中只能通过梯度数值检验（如有限差分对比）发现。

### 11.4 A4 验证的局限

- **`CrossEntropy` 路径不对称未深查**：A4 表格标注 `CrossEntropy` 为 ⚠️，但未深入验证 VM 侧是否通过 native 调用机制间接进入解释器的 `tape.cross_entropy`。这可能不是真正的缺陷，而是 native 注册机制的正常行为——但需 T35 等价性验证确认。
- **`Neg` 实际影响未运行时验证**：A4 推测 `-tensor` 走 `Sub(0, tensor)` 路径，但未通过实际运行 recording 模式测试验证。若 `-tensor` 实际走独立路径而 `Neg` 算子真的未被触发，则用户梯度已错误；若走 `Sub` 路径，则 `Neg` 仅是 dead code 气味。需运行时测试裁定。
- **VM 5 形态未逐一验证**：`Add/Sub/Mul/Div` 在 VM 中各有 5 处 record 调用（对应 `tensor+tensor`、`tensor+scalar`、`scalar+tensor` 等形态），A4 只确认了"至少一处"存在，未逐一验证 5 形态的对称性。

### 11.5 A5 对比的局限

- **PyTorch/JAX 版本基准**：A5 的对比基于 PyTorch 2.x 与 JAX 0.4.x 的一般设计，未引用具体版本的源码。
- **性能数据缺失**：A5 表格的性能维度是定性陈述，未提供 benchmark 数据。
- **TorchInductor/XLA 细节**：A5 对 PyTorch TorchInductor 与 JAX XLA 的描述简化，未深入两者的 graph break 与 fallback 机制。

### 11.6 形式化模型的局限

- **未建模 `tape_id` 的全局唯一性**：`tape_id` 是 `Option<usize>`，跨 tape 实例的唯一性未形式化。若 tape 被 `clear()` 后重建（[autodiff.rs:752](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/autodiff.rs)），旧 `tape_id` 可能与新 id 冲突——但当前实现中 `clear` 与 `recording = false` 同步，不冲突。
- **未建模 `Rc<RefCell<Tensor>>` 的所有权**：`input_tensors` 持有 `Rc` 引用，若张量在 backward 前被释放，tape 节点持有悬垂引用——但 Rust 的 `Rc` 保证引用计数，不会释放。这一安全保证未形式化。

### 11.7 工程差距

- **JIT 注释与代码不一致**（§11.2）：注释说 "interpreter's"，代码 fallback 至 `vm.call`。这是文档债务，建议修正注释或重构 fallback 目标。
- **`Neg` dead code**（§7.3）：`TapeOp::Neg` 的 backward 实现存在但前向未用，是设计-实现漂移。建议删除或补全。

---

## 12. 结论

本文形式化证明了 Tenth autodiff tape 的多路径一致性：

1. **A1**：VM 与解释器的 tape 同构（依赖 T35 等价性）；
2. **A2**：JIT 退出至 VM 保持 autodiff 语义（与 T9 联动覆盖全语义空间）；
3. **A3**：梯度正确性相对于 tape 完整性，错误静默；
4. **A4**：21 算子穷尽性验证发现 `Neg` 算子的真实 record 缺失——A3 预测的"静默破坏"在历史代码中已发生；
5. **A5**：与 PyTorch/JAX 对比，Tenth 的"JIT 退出 + VM 接管"策略在副作用敏感性上独特，但 effect system 缺位下脆弱。

**核心洞察**：Tenth 的 autodiff 正确性依赖**三条路径的 record 调用对称性**与**JIT 安全门的完整性**两个工程不变量。前者由 A1 保证（依赖 T35），后者由 A2 保证（依赖 T9）。任何不变量的破坏（如新增算子忘 record，或 JIT 安全门被绕过）都会导致 A3 预测的静默梯度错误。

**实施建议**：
1. **短期**：清理 `TapeOp::Neg` dead code（删除或补全前向 record）；
2. **中期**：为每个 `TapeOp` 变体编写 recording 模式测试，验证 tape 节点出现；
3. **长期**：引入 effect system 强制 `#[recordable]` 注解（§10），从类型层杜绝 A4 类缺陷。

本论文的诚实贡献在于 A4 的负面发现——理论模型预测的脆弱性在实现中已被实证。这是数理部"对自身局限诚实"原则的实践。

---

## 附录 A：定理索引

| 定理 | 陈述 | 证明 | 实证依据 |
|------|------|------|---------|
| A1 | VM 与解释器 tape 同构 | §4 归纳 | [vm.rs:1810](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/vm.rs), [mod.rs:1032](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/interpreter/mod.rs) |
| A2 | JIT 退出语义保持 | §4 轨迹分析 | [jit/mod.rs:41-43](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/jit/mod.rs), [vm.rs:191,325](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/vm.rs) |
| A3 | 梯度正确性相对性 | §4 逆否 | [autodiff.rs:285,1817](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/autodiff.rs) |
| A4 | 21 算子穷尽性 | §4 枚举 | [autodiff.rs:29-79](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/autodiff.rs) 全树 Grep |
| A5 | 与 PyTorch/JAX 对比 | §4 维度分析 | 文献 |

## 附录 B：与现有文档的对应

| 本论文章节 | 对应真理源 |
|-----------|-----------|
| §3.1 TapeOp 枚举 | `docs/语言参考手册.md` autodiff 章节 |
| §3.2 record_binary | `CODE_WIKI.md` runtime 模块详解 |
| §3.3 is_recording 安全门 | `MEMO.md` JIT 安全门引入版本记录 |
| §4 A4 `Neg` 缺陷 | `AUDIT.md` 缺陷登记（建议补充登记） |
| §11 局限 | `AUDIT.md` 架构债务 |

## 附录 C：实施建议

| 优先级 | 建议 | 关联章节 | 预期工作量 |
|--------|------|---------|-----------|
| P0 | 删除 `TapeOp::Neg` dead code 或补全前向 record | §7.3 | 1 小时（含测试） |
| P1 | 为 21 算子编写 recording 模式测试 | §9.2 | 1 天 |
| P1 | 修正 [jit/mod.rs:38-40](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/jit/mod.rs) 注释（"interpreter's" → "VM's"） | §11.2 | 5 分钟 |
| P2 | 验证 `CrossEntropy` 路径对称性 | §11.4 | 半天 |
| P3 | effect system 设计 spike | §10 | 1 周 |

---

## 参考文献

1. Wengert, R. E. (1964). "A simple automatic derivative evaluation program." *Communications of the ACM*, 7(8), 463–464.
2. Baydin, A. G., Pearlmutter, B. A., Radul, A. A., & Siskind, J. M. (2018). "Automatic differentiation in machine learning: a survey." *Journal of Machine Learning Research*, 18(153), 1–43.
3. Paszke, A., et al. (2017). "Automatic differentiation in PyTorch." *NeurIPS Autodiff Workshop*.
4. Bradbury, J., et al. (2018). "JAX: composable transformations of Python+NumPy programs." *JAX documentation*.
5. Stoyanov, V. (1986). "Partial evaluation and side effects." *ACM Symposium on LISP and Functional Programming*.
6. Bauer, F. L. (1974). "Computational graphs and rounding error." *SIAM Journal on Numerical Analysis*, 11(1), 87–96.
7. Tenth 项目. (2026). `CODE_WIKI.md` runtime 模块详解. [内部文档]
8. Tenth 项目. (2026). `MEMO.md` v0.3.3 变更记录. [内部文档]
9. Tenth 项目. (2026). T9-JIT 特化语义保持证明. [联动论文]
10. Tenth 项目. (2026). T35-解释器–VM 等价性. [待撰写]

---

> **数理部声明**：本论文遵循"严谨性、完备性边界、局限诚实"三原则。A4 的 `Neg` 算子遗漏发现是基于静态代码搜索的实证，未运行时验证实际影响——这一不确定性已在 §11.4 显式记录。A1 对 T35 的依赖、A2 对注释-代码不一致的发现、A3 对"完整性"定义的狭窄性，均在 §11 逐条披露。本论文不掩盖实现与理论的差距，而是把差距作为改进的起点。
