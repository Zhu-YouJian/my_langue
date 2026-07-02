# JIT 缓存的生命周期与代码热加载：Tenth JitContext 的不动点假设与不可重定位性

> **作者**：Tenth 项目数理部
> **日期**：2026-07-02
> **类型**：理论分析论文（T33 理论点，护城河 E：JIT 编译体系）
> **实证基础**：Tenth v0.3.3+ 源码（`compile/jit/context.rs`、`compile/jit/mod.rs`、`compile/jit/translator.rs`、`compile/jit/hostcalls.rs`、`runtime/vm.rs`）
> **关联文档**：`docs/论文/T9-JIT特化语义保持证明.md`（JIT 系列姊妹篇，本文扩展其 §11 局限 L2、L3）；JIT 系列规划文档 T31/T32（撰写时未见诸 `docs/论文/` 目录，本文以"前瞻引用"方式标注联动关系，详见 §11 局限 L6）
> **版本**：v1（首轮分析，含 4 轮自审修正留痕）

---

## 摘要

本文对 Tenth 语言 JIT 编译子系统 `JitContext` 的缓存生命周期与代码热加载能力进行形式化建模与正确性证明。Tenth JIT 通过 `HashMap<usize, JitFn>` 按 `chunk_idx` 缓存 Cranelift 编译产物，`Drop` 时显式 `cache.clear()` 再由 `JITModule` 释放；同时为支持 `call_indirect` 经绝对地址调用 hostcall trampoline，将 Cranelift 编译标志 `is_pic` 显式置为 `false`。这两个工程决策隐含两条强假设：**不动点假设**——`chunk_idx` 一经分配永不回收，缓存索引空间单调增长；**不可重定位性假设**——JIT 机器码内嵌绝对地址，一旦生成即与加载地址绑定，无法迁移。本文给出五个主定理：（K1）不动点假设的实证性成立——通过源码搜索证明 `Vm::chunks` 仅追加不删除；（K2）不可重定位性定理——`is_pic = false` 与 `hostcall_addr` 的 `iconst` 嵌入共同导致 JIT 代码不可重定位，并给出迁移失败的形式化构造；（K3）Drop 正确性定理——`cache.clear()` 先于 `JITModule` 释放保证无悬垂指针；（K4）与 V8 code aging / LuaJIT mcode 的对比定理——Tenth 的"永不回收"策略在热加载维度上严格弱于 V8/LuaJIT；（K5）引用计数回收方案的形式化刻画——作为未来工作，给出 `Arc<JitFn>` + 弱引用回收的可行性骨架与未解决的并发难题。本文诚实记录 6 处理论局限，包括 `Module::finish` 的 Drop 语义不完备、跨平台 `is_pic` 行为差异、T31/T32 联动缺口等，为后续 JIT 缓存淘汰策略、热重载子系统、PIC 重定位改造奠定形式化坐标。

**关键词**：JIT 缓存、生命周期管理、代码热加载、不动点假设、不可重定位性、位置无关代码、Cranelift、V8 code aging、LuaJIT mcode、Tenth 语言

---

## 1. 引言

### 1.1 动机：JIT 缓存的生命周期张力

JIT（Just-In-Time）编译在提升运行时性能的同时，引入一个根本的工程张力：**编译产物占据可执行内存，但其生命周期必须与被引用期严格对齐**。一旦编译产物被释放而引用尚存，即产生悬垂指针（dangling pointer），调用即触发段错误；反之，编译产物永不释放则造成内存泄漏，长期运行的进程将耗尽地址空间。

工业级 JIT 运行时（V8、LuaJIT、HotSpot、PyPy）均为此设计复杂的缓存生命周期策略：V8 的 code aging 将长期未执行的机器码标记为"冷"并最终回收；LuaJIT 的 mcode 区按页管理，支持整体释放但不易细粒度回收；HotSpot 的 nmethod 由 sweeper 增量清理。这些策略的共同前提是**索引可回收**与**代码可重定位**两条假设中至少一条成立。

Tenth 语言的 JIT 子系统（基于 Cranelift）采取了一条与上述系统截然不同的保守路线：**索引永不回收 + 代码不可重定位**。这一选择极大地简化了实现，但同时也关闭了细粒度缓存淘汰与代码热加载的可能性。本文对这两条隐含假设进行形式化分析，明确其边界、影响与未来改造路径。

### 1.2 Tenth JIT 缓存的工程现状

Tenth JIT 的核心数据结构 `JitContext` 定义于 [`tenth/src/compile/jit/context.rs:14-18`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/jit/context.rs)：

```rust
pub struct JitContext {
    module: JITModule,
    /// Cached compiled function pointers, keyed by chunk index.
    cache: HashMap<usize, JitFn>,
}
```

缓存键为 `chunk_idx: usize`，即字节码 chunk 在 `Vm::chunks: Vec<Chunk>` 中的下标（[`runtime/vm.rs:150`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/vm.rs)）。`Drop` 实现于 [`context.rs:61-69`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/jit/context.rs)：

```rust
impl Drop for JitContext {
    fn drop(&mut self) {
        self.cache.clear();
    }
}
```

Cranelift 编译标志中 `is_pic` 被显式置为 `false`（[`context.rs:27`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/jit/context.rs)），其注释明确说明动机：

> PIC must be disabled for `call_indirect` to work correctly with absolute hostcall addresses on Windows x64.

与之配套，translator 通过 `hostcall_addr` 将 hostcall 函数的绝对地址以 `iconst` 指令硬编码进 JIT 机器码（[`translator.rs:583-587`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/jit/translator.rs)），随后由 `call_indirect` 经该绝对地址发起间接调用（[`translator.rs:606`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/jit/translator.rs) 等共 21 处）。

### 1.3 两条隐含假设

上述工程决策隐含两条强假设：

- **不动点假设（Fixed-Point Assumption, FPA）**：`chunk_idx` 一经分配即永不回收。若 `chunk_idx` 被回收并重新分配给新 chunk，缓存命中将返回指向旧机器码的指针，而旧机器码对应已不存在的字节码——语义断裂。
- **不可重定位性假设（Non-Relocatability Assumption, NRA）**：JIT 机器码一旦生成就与加载地址绑定，无法迁移到其他地址空间。若强制迁移，内嵌的绝对地址将指向无效位置——调用即崩溃。

这两条假设在 T9 论文（[`docs/论文/T9-JIT特化语义保持证明.md`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/docs/论文/T9-JIT特化语义保持证明.md)）的 §11 局限章节中被首次披露（局限 L2、L3），但未展开分析。本文承接 T9，对 FPA 与 NRA 进行完整的形式化建模、正确性证明与对比分析。

### 1.4 研究问题

本文回答以下五个研究问题：

- **RQ1**（不动点假设）：`chunk_idx` 是否真的永不回收？该假设的实证强度如何？
- **RQ2**（不可重定位性）：`is_pic = false` 在何种形式化意义上导致 JIT 代码不可重定位？不可重定位性的边界条件是什么？
- **RQ3**（Drop 正确性）：`cache.clear()` + `JITModule` 隐式 Drop 的释放顺序是否保证无悬垂指针？是否存在 `Module::finish` 语义不完备导致的漏洞？
- **RQ4**（对比定位）：Tenth 的"永不回收 + 不可重定位"策略在 V8/LuaJIT 谱系中处于何种位置？热加载能力缺失的代价是什么？
- **RQ5**（回收方案可行性）：引用计数回收方案（`Arc<JitFn>` + 弱引用）在 Tenth 当前架构下是否可行？哪些并发难题阻碍其落地？

### 1.5 贡献

- **形式化建模**（§4、§6）：将 `JitContext` 缓存、`chunk_idx` 索引空间、JIT 机器码地址嵌入抽象为数学对象，给出缓存生命周期的状态迁移语义。
- **五个主定理与证明**（§5）：K1（不动点假设实证）、K2（不可重定位性）、K3（Drop 正确性）、K4（与 V8/LuaJIT 对比）、K5（引用计数回收方案形式化）。
- **诚实局限记录**（§11）：独立章节记录 6 处理论局限，包括 `Module::finish` Drop 语义、跨平台 `is_pic` 差异、T31/T32 联动缺口等。
- **未来工作形式化基础**（§10）：为 JIT 缓存 LRU 淘汰、热重载子系统、PIC 重定位改造提供理论坐标。

### 1.6 v1 自审留痕

本文经历 4 轮自审：

| 轮次 | 原始断言 | 修正 |
|------|---------|------|
| 第 1 轮（结构） | K2 初稿声称"完全不可重定位" | 修正为"对 hostcall 地址不可重定位，对 chunk 内部逻辑地址仍可相对寻址"——`is_pic=false` 仅影响绝对地址嵌入，不影响 PC 相对跳转 |
| 第 2 轮（证明） | K3 初稿声称"`cache.clear()` 是充分必要" | 修正为"仅充分不必要"——即便不清空 cache，Rust 借用规则也保证 `module` 与 `cache` 同时被 drop；但显式 clear 提供"防御性编程"语义，对未来 cranelift 版本变更 Drop 行为具有韧性 |
| 第 3 轮（边界） | K1 初稿未处理"chunks 容量回绕"边界 | 补充：`Vec::len` 返回 `usize`，在 64 位平台上理论容量 $2^{64}-1$，实际不会回绕；但若未来引入 `Vec::shrink_to_fit` 或 `Vec::swap_remove` 则假设破坏——已加入局限 L1 |
| 第 4 轮（诚实） | K5 初稿声称"引用计数方案可落地" | 修正：标注为"未来工作"，并发难题（§10.3）尚未解决，仅给出形式化骨架 |

---

## 2. 背景与相关工作

### 2.1 V8 code aging

V8（Google Chrome 的 JavaScript 引擎）引入 code aging 机制以应对 JIT 代码的内存累积问题。其核心思想是：**为每个 JIT 函数附加"年龄"计数器，长期未执行的函数被标记为冷代码（cold），最终被回收以释放可执行内存**。

V8 code aging 的关键机制包括：

- **Age counter**：每个 Code 对象维护一个 8 位年龄字段，函数入口处递增；GC 时根据年龄分级标记。
- **Flush policy**： Major GC 时，超过阈值年龄的 Code 对象被 flush（释放机器码），保留字节码以便重新 JIT。
- **Deoptimization**：flush 触发后，下次调用经栈帧回退到解释器（Ignition），重新触发 JIT。

V8 code aging 的前提是：**Code 对象可独立释放，且引用方可通过 deoptimization 重新进入解释执行**。这与 Tenth 的"索引永不回收"形成鲜明对比——Tenth 没有 deoptimization 路径，缓存命中的指针直接被调用，无法回退。

### 2.2 LuaJIT mcode 管理

LuaJIT 的机器码（mcode）区采用页级管理：

- **整体分配**：mcode 区按 64KB 页分配，所有 JIT 函数共享同一页池。
- **Mcode limit**：默认 512MB（`luaJIT.mcode.size`），达到上限后停止 JIT。
- **页释放**：`lj_mcode_free` 在 trace 退出时释放整页，但**单函数粒度的释放不支持**——页内活跃函数阻止整页释放。
- **PIC 处理**：LuaJIT 在 x64 上默认生成 PIC 机器码，通过 GOT（全局偏移表）间接寻址，支持 mcode 区整体迁移。

LuaJIT 的策略是"页级回收 + PIC 可迁移"，与 Tenth 的"永不回收 + 不可重定位"恰为两个极端。

### 2.3 HotSpot nmethod sweeper

HotSpot JVM 的 nmethod（已编译的 Java 方法）由 sweeper 线程增量清理：

- **Mark-sweep**：GC 标记不可达 nmethod，sweeper 异步释放。
- **Unloading barrier**：内联缓存（IC）在 nmethod 卸载时被 patch 为解释器入口，保证调用方安全。
- **OSR（栈上替换）**：支持从解释执行切换到 JIT 执行，反之亦然（deoptimization）。

HotSpot 的 unloading barrier 是关键：**卸载 nmethod 前必须 patch 所有引用方**。Tenth 缺乏这一基础设施——`JitFn` 是裸函数指针，无 patch 机制，因此无法安全卸载。

### 2.4 T9 论文的局限披露

T9 论文（[`docs/论文/T9-JIT特化语义保持证明.md`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/docs/论文/T9-JIT特化语义保持证明.md)）在 §11 局限章节首次披露：

> **局限 L2**：JIT 缓存的不动点假设——`chunk_idx` 永不回收，未来若引入 chunk 淘汰机制需重新审视。
>
> **局限 L3**：`is_pic = false` 导致 JIT 代码不可重定位，未来若需 mcode 区迁移或热重载需改造为 PIC。

本文承接 T9，将这两条局限从"提及"升级为"完整证明"，并扩展至 Drop 正确性、对比定位、回收方案三个新维度。

---

## 3. 预备知识与符号约定

### 3.1 符号表

| 符号 | 含义 |
|------|------|
| $\mathbb{N}$ | 自然数集 $\{0, 1, 2, \dots\}$ |
| $\mathcal{C}$ | chunk 索引空间，$\mathcal{C} \subseteq \mathbb{N}$ |
| $\mathcal{K}$ | 缓存键空间，$\mathcal{K} \subseteq \mathbb{N}$ |
| $\mathcal{F}$ | JIT 函数指针空间，$\mathcal{F} \subseteq \mathbb{N}$（地址） |
| $\text{cache}: \mathcal{K} \rightharpoonup \mathcal{F}$ | 缓存偏函数 |
| $\text{chunks}: \mathcal{C} \to \text{Chunk}$ | chunk 存储函数 |
| $\text{idx}: \text{Chunk} \rightharpoonup \mathcal{C}$ | chunk 到索引的逆映射 |
| $\text{addr}: \text{Hostcall} \to \mathcal{F}$ | hostcall 到地址的映射 |
| $\text{embed}: \mathcal{F} \to \text{MachineCode}$ | 地址嵌入函数（生成 `iconst` 指令） |
| $\sigma \in \Sigma$ | 缓存状态，$\sigma = (\text{cache}, \text{chunks}, \text{module})$ |
| $\to \subseteq \Sigma \times \Sigma$ | 状态迁移关系 |
| $\sqsubseteq$ | 偏序关系（子集关系） |

### 3.2 关键定义

**定义 3.1（chunk 索引单调性）**：称 chunk 索引空间 $\mathcal{C}$ 是单调增长的，若其演化序列 $\mathcal{C}_0 \subseteq \mathcal{C}_1 \subseteq \dots \subseteq \mathcal{C}_t \dots$，且对任意 $t_1 < t_2$，$\mathcal{C}_{t_1} \subseteq \mathcal{C}_{t_2}$。

**定义 3.2（不动点假设 FPA）**：称 JIT 缓存满足不动点假设，若：
1. $\mathcal{C}$ 单调增长；
2. 对任意 $i \in \mathcal{C}$，$\text{chunks}(i)$ 在 $i$ 被分配后不再变更（即 $\text{chunks}_t(i) = \text{chunks}_{t'}(i)$ 对所有 $t' > t$ 成立，其中 $t$ 是 $i$ 被分配的时刻）。

**定义 3.3（地址嵌入）**：JIT 机器码 $m$ 的地址嵌入函数 $\text{embed}(a)$ 生成将地址 $a \in \mathcal{F}$ 作为立即数嵌入的指令序列。在 Cranelift IR 层面表现为 `iconst $ptr, a`（[`translator.rs:586`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/jit/translator.rs)）。

**定义 3.4（可重定位性）**：称机器码 $m$ 在迁移 $\rho: \mathcal{F} \to \mathcal{F}$（地址重映射）下可重定位，若存在有效算法 $R$ 使得 $R(m, \rho)$ 产生新机器码 $m'$，且 $m'$ 在 $\rho$ 后的地址空间中执行语义等价于 $m$ 在原地址空间中的执行。即：
$$\forall \rho.\ \exists R.\ \text{sem}(R(m, \rho)) = \text{sem}(m) \circ \rho^{-1}$$

**定义 3.5（悬垂指针）**：称缓存 $\text{cache}$ 在状态 $\sigma$ 下存在悬垂指针，若存在 $k \in \text{dom}(\text{cache})$ 使得 $\text{cache}(k)$ 指向的内存已被释放或不可执行。

**定义 3.6（缓存生命周期）**：缓存 $\text{cache}$ 的生命周期是状态序列 $\sigma_0 \to \sigma_1 \to \dots \to \sigma_n$，其中 $\sigma_0$ 是 `JitContext::new()` 后的初始状态（$\text{cache} = \emptyset$），$\sigma_n$ 是 `Drop` 后的终态（$\text{cache} = \emptyset$ 且 $\text{module}$ 已释放）。

---

## 4. Tenth JIT 缓存的形式化建模

### 4.1 JitContext 的形式化

`JitContext` 的形式化模型为三元组：

$$\text{JitContext} = (\text{module}: \text{JITModule},\ \text{cache}: \mathcal{K} \rightharpoonup \mathcal{F},\ \text{flags}: \text{Settings})$$

其中 $\text{flags}$ 包含 `is_pic = false`、`use_colocated_libcalls = false` 两个关键设置（[`context.rs:24-27`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/jit/context.rs)）。

### 4.2 状态迁移语义

JIT 缓存的状态迁移由以下三条规则定义：

**规则 T1（编译并缓存）**：

$$\frac{\text{chunk\_idx} \notin \text{dom}(\text{cache}) \quad \text{translate}(\text{chunk}) = f \quad \text{finalize}(f) = a}{(\text{cache}, \text{chunks}, \text{module}) \to (\text{cache}[\text{chunk\_idx} \mapsto a], \text{chunks}, \text{module})}$$

对应源码 `get_or_compile`（[`context.rs:36-58`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/jit/context.rs)）：若 `chunk_idx` 不在缓存中，调用 `translator::translate` 编译，`finalize_definitions` 后通过 `get_finalized_function` 获取函数指针，插入缓存。

**规则 T2（缓存命中）**：

$$\frac{\text{chunk\_idx} \in \text{dom}(\text{cache})}{(\text{cache}, \text{chunks}, \text{module}) \to (\text{cache}, \text{chunks}, \text{module})}$$

对应源码 `if let Some(f) = self.cache.get(&chunk_idx) { return Ok(*f); }`（[`context.rs:37-39`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/jit/context.rs)）：直接返回缓存的函数指针，状态不变。

**规则 T3（Drop 释放）**：

$$\frac{}{(\text{cache}, \text{chunks}, \text{module}) \to_{\text{drop}} (\emptyset, \text{chunks}, \bot)}$$

对应源码 `Drop::drop`（[`context.rs:61-69`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/jit/context.rs)）：先 `cache.clear()` 清空缓存，再由 `JITModule` 的隐式 Drop 释放机器码内存。

### 4.3 chunk 索引空间的形式化

`Vm::chunks` 是 `Vec<Chunk>`（[`runtime/vm.rs:150`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/vm.rs)），其索引空间 $\mathcal{C} = \{0, 1, \dots, |\text{chunks}| - 1\}$。索引分配的唯一入口是 `add_fn`（[`runtime/vm.rs:297-302`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/vm.rs)）：

```rust
pub fn add_fn(&mut self, name: String, chunk: Chunk) {
    let idx = self.chunks.len();
    self.chunks.push(chunk);
    self.chunk_names.push(name.clone());
    self.functions.insert(name, idx);
}
```

形式化：`add_fn` 执行迁移 $\sigma \to \sigma'$，其中 $\mathcal{C}' = \mathcal{C} \cup \{|\text{chunks}|\}$，$\text{chunks}'(|\text{chunks}|) = \text{chunk}$。

### 4.4 地址嵌入的形式化

`hostcall_addr` 函数（[`translator.rs:583-587`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/jit/translator.rs)）将 hostcall 函数地址嵌入 JIT 机器码：

```rust
fn hostcall_addr(&mut self, name: &str) -> Result<Value_, String> {
    let addr = super::hostcalls::hostcall_addr(name)
        .ok_or_else(|| format!("unknown hostcall: {name}"))?;
    Ok(self.builder.ins().iconst(self.ptr, addr as i64))
}
```

形式化：$\text{embed}(\text{addr}(h)) = \text{iconst}(\text{ptr}, \text{addr}(h))$，其中 $h \in \text{Hostcall}$。生成的机器码包含立即数 $\text{addr}(h)$，这是一个绝对地址（在 `is_pic = false` 下）。

后续 `call_indirect`（[`translator.rs:606`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/jit/translator.rs) 等共 21 处）以该绝对地址为 callee 发起间接调用。形式化：$\text{call\_indirect}(\text{sig}, \text{embed}(\text{addr}(h)), \text{args})$。

---

## 5. 主定理与证明

### 5.1 定理 K1（不动点假设的实证性成立）

**定理 K1**：在 Tenth v0.3.3+ 源码中，`Vm::chunks` 索引空间 $\mathcal{C}$ 满足不动点假设 FPA（定义 3.2），即：
1. $\mathcal{C}$ 单调增长；
2. 对任意 $i \in \mathcal{C}$，$\text{chunks}(i)$ 在 $i$ 被分配后不再变更。

**证明**：

**Part 1（$\mathcal{C}$ 单调增长）**：

`Vm::chunks` 的类型为 `Vec<Chunk>`（[`runtime/vm.rs:150`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/vm.rs)）。`Vec` 的语义保证：`push` 操作使 `len` 单调递增，且不改变既有元素的索引与值。`add_fn` 是唯一向 `chunks` 添加元素的方法（搜索证据：`Grep "chunks\.push|chunks\.remove|chunks\.swap_remove"` 在 `vm.rs` 中仅匹配到 `chunks.push(chunk)` 一处，[`runtime/vm.rs:299`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/vm.rs)）。

形式化：设 $\sigma_t = (\text{cache}_t, \text{chunks}_t, \text{module}_t)$ 为时刻 $t$ 的状态。`add_fn` 执行迁移 $\sigma_t \to \sigma_{t+1}$，其中 $\text{chunks}_{t+1} = \text{chunks}_t \cup \{|\text{chunks}_t| \mapsto \text{chunk}\}$，故 $\text{dom}(\text{chunks}_t) \subseteq \text{dom}(\text{chunks}_{t+1})$。由归纳法，$\mathcal{C}_t = \text{dom}(\text{chunks}_t)$ 单调增长。$\square$

**Part 2（$\text{chunks}(i)$ 不变更）**：

搜索证据：`Grep "chunks\.remove|chunks\.swap_remove|chunks\.truncate|chunks\.clear|chunks\.insert"` 在 `vm.rs` 中无匹配（除 `add_fn` 的 `push` 与只读的 `get` 外，`chunks` 字段无其他修改入口）。`Vec::push` 不改变既有元素。

形式化：对任意 $i \in \mathcal{C}_t$，设 $i$ 在时刻 $t_i$ 被分配（即 $i = |\text{chunks}_{t_i}|$ 且 $i \notin \mathcal{C}_{t_i - 1}$）。对任意 $t' > t_i$，由 Part 1，$\text{chunks}$ 仅经 `push` 增长，故 $\text{chunks}_{t'}(i) = \text{chunks}_{t_i}(i)$。$\square$

**综合**：FPA 的两个条件均成立，故 K1 成立。$\blacksquare$

**实证强度说明**：K1 的证明依赖"源码中无 chunks 删除操作"的搜索证据。这一证据是**实证性**（empirical）的，而非**逻辑必然**（logical necessity）。未来若引入 `chunks.swap_remove` 或 `Vec::truncate`，FPA 立即破坏（详见局限 L1）。

### 5.2 定理 K2（不可重定位性）

**定理 K2**：在 `is_pic = false` 的设置下（[`context.rs:27`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/jit/context.rs)），Tenth JIT 生成的机器码 $m$ 对 hostcall 地址不可重定位。即存在地址迁移 $\rho: \mathcal{F} \to \mathcal{F}$，使得不存在有效算法 $R$ 满足定义 3.4 的可重定位性条件。

具体地，设 $m$ 通过 $\text{call\_indirect}(\text{sig}, \text{embed}(\text{addr}(h)), \text{args})$ 调用 hostcall $h$（[`translator.rs:606`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/jit/translator.rs)）。若 hostcall $h$ 的地址从 $\text{addr}(h)$ 迁移至 $\rho(\text{addr}(h)) \neq \text{addr}(h)$，则未重定位的 $m$ 在原地址空间执行时将调用 $\text{addr}(h)$（已失效），触发未定义行为。

**证明**：

**Step 1（地址嵌入是绝对地址）**：

`is_pic = false`（[`context.rs:27`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/jit/context.rs)）使 Cranelift 生成非 PIC 机器码。在非 PIC 模式下，`iconst $ptr, addr`（[`translator.rs:586`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/jit/translator.rs)）将 `addr` 作为**绝对地址**立即数嵌入机器码，而非 PC 相对偏移或 GOT 间接引用。

形式化：$\text{embed}(\text{addr}(h))$ 生成的机器码字节序列包含 $\text{addr}(h)$ 的二进制表示，且该表示与加载地址无关（即非 PIC）。

**Step 2（迁移破坏嵌入地址的有效性）**：

设 $\rho: \mathcal{F} \to \mathcal{F}$ 是非常迁移（$\exists a.\ \rho(a) \neq a$），且 hostcall $h$ 的地址被迁移：$\rho(\text{addr}(h)) \neq \text{addr}(h)$。这对应 hostcall 函数被重新加载到新地址的场景（如动态链接器重定位、mmap 区域调整）。

未重定位的机器码 $m$ 仍包含立即数 $\text{addr}(h)$。当 $m$ 执行 `call_indirect` 时，控制流跳转到地址 $\text{addr}(h)$——但该地址已被迁移，可能：
- (a) 指向已释放内存 → 段错误；
- (b) 指向其他函数 → 语义错误；
- (c) 指向无效指令 → 非法指令异常。

三种情况均违反定义 3.4 的语义等价条件。

**Step 3（重定位算法的不存在性）**：

为使 $m$ 在迁移后语义等价，需要算法 $R$ 扫描 $m$ 中所有嵌入的绝对地址，将 $\text{addr}(h)$ 替换为 $\rho(\text{addr}(h))$。但 $R$ 的存在性依赖以下条件：

1. **地址识别**：$R$ 必须能识别 $m$ 中哪些字节是嵌入的地址，哪些是指令操作码或非地址立即数。在非 PIC 机器码中，地址与非地址立即数在字节层面无区别，$R$ 必须反汇编 $m$ 并理解其语义。
2. **完备地址集**：$R$ 必须知道 $m$ 中嵌入了哪些 hostcall 地址。这要求 $R$ 访问 translator 的元数据（即哪些 `iconst` 指令嵌入了 hostcall 地址）。
3. **原子性**：$R$ 必须在 $m$ 不被执行时完成重写，否则并发执行可能观察到部分重写的中间状态。

Tenth JIT 的当前实现不提供上述任何条件：
- 机器码无重定位表（relocation table），Cranelift 在 `is_pic = false` 下不生成 `.rela` 段；
- translator 不记录嵌入地址的元数据；
- JIT 机器码区无写保护切换机制（无法在执行时安全重写）。

因此，不存在有效的 $R$。形式化：

$$\nexists R.\ \forall \rho.\ \text{sem}(R(m, \rho)) = \text{sem}(m) \circ \rho^{-1}$$

故 $m$ 不可重定位。$\blacksquare$

**推论 K2.1**：Tenth JIT 机器码无法被迁移到不同的加载地址。具体影响：
- (i) 无法将 mcode 区整体迁移以解决碎片化；
- (ii) 无法在进程 fork 后子进程共享 mcode（因 ASLR 使子进程加载地址不同）；
- (iii) 无法实现代码热重载（hot reload）——新版本函数必须生成新机器码而非替换旧机器码；
- (iv) 无法支持 mmap 区域重映射（mremap）以压缩内存。

**边界说明**：K2 仅断言"对 hostcall 地址不可重定位"。chunk 内部的相对跳转（如 `jump` 指令生成的 PC 相对分支）在 `is_pic = false` 下仍是 PC 相对的，理论上可随 mcode 区整体迁移。但由于 hostcall 地址嵌入的绝对性，整体迁移仍不可行——除非 chunk 完全不调用 hostcall，但 Tenth JIT 的所有复杂操作（call、tensor、autodiff、堆分配）均经 hostcall（[`mod.rs:7-11`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/jit/mod.rs)），故实际所有 chunk 均受 K2 约束。

### 5.3 定理 K3（Drop 的正确性）

**定理 K3**：`JitContext::drop` 的释放顺序（`cache.clear()` 先于 `JITModule` 隐式 Drop）保证无悬垂指针。即对任意缓存状态 $\sigma = (\text{cache}, \text{chunks}, \text{module})$，$\sigma \to_{\text{drop}} \sigma'$ 后，不存在 $k \in \text{dom}(\text{cache})$ 使得 $\text{cache}(k)$ 指向已释放内存。

**证明**：

`Drop::drop` 的实现（[`context.rs:61-69`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/jit/context.rs)）：

```rust
fn drop(&mut self) {
    self.cache.clear();
}
```

释放顺序分析：

**Step 1（cache.clear() 的效果）**：

`HashMap::clear` 释放 HashMap 内部的 bucket 数组，但**不调用值的 Drop**——因为 `JitFn` 是 `unsafe extern "C" fn` 指针，是 `Copy` 类型，无 Drop 实现。形式化：`cache.clear()` 后，$\text{cache} = \emptyset$，但 `cache` 中曾存储的函数指针值（指向 JIT 机器码）在内存层面仍存在（直到 HashMap 的 bucket 被覆盖）。

**Step 2（JITModule 的隐式 Drop）**：

`JitContext` 的字段顺序为 `module: JITModule` 在前，`cache: HashMap` 在后（[`context.rs:14-18`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/jit/context.rs)）。Rust 的 Drop 顺序是**字段声明逆序**：先 `cache.drop()`（即 `Drop::drop` 中的 `cache.clear()`），再 `module.drop()`（`JITModule` 的隐式 Drop，释放机器码内存）。

**Step 3（无悬垂指针的保证）**：

悬垂指针的定义（定义 3.5）要求存在 $k \in \text{dom}(\text{cache})$ 使得 $\text{cache}(k)$ 指向已释放内存。在 $\sigma \to_{\text{drop}} \sigma'$ 后：
- $\text{cache}' = \emptyset$（由 Step 1）；
- $\text{module}' = \bot$（由 Step 2，机器码已释放）。

由于 $\text{dom}(\text{cache}') = \emptyset$，定义 3.5 的存在量词 $\exists k \in \text{dom}(\text{cache}')$ 不成立。故无悬垂指针。$\square$

**Step 4（防御性编程的语义）**：

注释（[`context.rs:63-66`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/jit/context.rs)）说明：

> 显式释放编译产物与代码映射，避免依赖 JITModule 的隐式 Drop 语义（未来 cranelift 版本变更 Drop 行为时不易察觉）。

由 Step 2，即便没有 `cache.clear()`，Rust 字段逆序 Drop 仍保证 `cache` 先于 `module` 释放。但 `cache.clear()` 提供两层防御：
- (i) 若未来 cranelift 变更 `JITModule` 的 Drop 语义（如改为懒释放），`cache` 中的指针可能在 `module` 真正释放前被外部引用——`cache.clear()` 确保 `cache` 在 `drop` 函数返回时为空，外部无法经 `JitContext` 获取指针。
- (ii) 若未来在 `Drop::drop` 中插入日志或 panic，`cache.clear()` 在最前确保即使后续代码 panic，`cache` 仍为空。

**综合**：K3 成立。$\blacksquare$

**局限说明**：K3 仅证明"`cache.clear()` + `JITModule` Drop"在当前 Rust 语义下无悬垂指针。但 `Module::finish` 的注释（[`context.rs:65`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/jit/context.rs)）提及"`Module::finish` 消费 self，这里只能尽力清理"——这暗示 `JITModule::drop` 可能未完整释放机器码内存，存在内存泄漏风险（详见局限 L2）。

### 5.4 定理 K4（与 V8/LuaJIT 对比）

**定理 K4**：Tenth JIT 的"永不回收 + 不可重定位"策略在热加载能力上严格弱于 V8 code aging 与 LuaJIT mcode 管理。具体地：
1. Tenth 不支持单函数级缓存淘汰，V8 支持（code aging flush）；
2. Tenth 不支持 mcode 区迁移，LuaJIT 支持（PIC + 页级重映射）；
3. Tenth 不支持 deoptimization 回退，V8/HotSpot 支持。

**证明**：

**Part 1（单函数级缓存淘汰）**：

V8 的 code aging 机制允许 Major GC 时 flush 年龄超过阈值的 Code 对象（§2.1）。flush 后，该函数下次调用经 deoptimization 回退到解释器，重新触发 JIT。

Tenth 的缓存淘汰策略：**无**。`JitContext::cache` 仅在 `Drop` 时清空（[`context.rs:67`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/jit/context.rs)），运行期间缓存只增不减。形式化：对任意时刻 $t$，$\text{dom}(\text{cache}_t) \subseteq \text{dom}(\text{cache}_{t+1})$（缓存键集单调增长）。

形式化对比：设 $f$ 为某 chunk 对应的 JIT 函数。在 V8 中，存在时刻 $t$ 使 $f \notin \text{dom}(\text{cache}_t)$（被 flush）。在 Tenth 中，对任意 $t$，$f \in \text{dom}(\text{cache}_t) \Rightarrow f \in \text{dom}(\text{cache}_{t+1})$（永不淘汰）。故 Tenth 的热加载能力严格弱于 V8。$\square$

**Part 2（mcode 区迁移）**：

LuaJIT 在 x64 上默认生成 PIC 机器码（§2.2），通过 GOT 间接引用外部符号。GOT 是数据段中的指针表，迁移时仅需更新 GOT 项，机器码本身不变。故 LuaJIT 的 mcode 区可整体迁移。

Tenth 的 `is_pic = false`（[`context.rs:27`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/jit/context.rs)）使机器码内嵌绝对地址（K2），无 GOT 间接层。迁移需重写机器码中的立即数，但不存在有效重定位算法（K2 Step 3）。故 Tenth 的 mcode 区不可迁移。$\square$

**Part 3（deoptimization 回退）**：

V8 的 deoptimization 在 flush 触发时将栈帧从 JIT 代码回退到解释器（Ignition）。这要求：(i) 栈帧携带足够元数据以重建解释器状态；(ii) 调用方检查目标函数是否已 flush。

Tenth 的 JIT 调用经 `hostcalls::invoke_jit`（[`hostcalls.rs:33`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/jit/hostcalls.rs)）直接调用函数指针，无 deoptimization 检查。形式化：调用路径为 `run_jit → get_or_compile → invoke_jit(fn_ptr, ...)`（[`mod.rs:62-81`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/jit/mod.rs)），其中 `fn_ptr` 是缓存命中的指针，无中间检查。

若实现 deoptimization，需在 `get_or_compile` 后插入"目标函数是否有效"检查，但这与"缓存命中即有效"的不动点假设冲突——FPA 保证缓存命中即有效，故无需检查。反之，若要支持 deoptimization，必须先破坏 FPA。$\square$

**综合**：Tenth 在三个维度上均严格弱于 V8/LuaJIT。$\blacksquare$

**注释**：K4 的"严格弱"是**功能维度**的对比，非**性能维度**。Tenth 的"永不回收"策略在稳态性能上可能优于 V8（无 flush 后的重新 JIT 开销），但代价是内存累积。这一权衡在 §8 工程权衡中详细讨论。

### 5.5 定理 K5（引用计数回收方案，未来工作）

**定理 K5（形式化骨架，非可落地证明）**：若将 `cache: HashMap<usize, JitFn>` 改造为 `cache: HashMap<usize, Arc<Weak<JitFn>>>`，并引入"chunk 引用计数归零即回收"的协议，则可在不破坏 FPA 的前提下实现单函数级缓存淘汰。但该方案面临三个未解决的并发难题，标注为未来工作。

**形式化骨架**：

**Step 1（数据结构改造）**：

$$\text{cache}: \mathcal{K} \rightharpoonup \text{Arc}^{\text{weak}}(\mathcal{F})$$

其中 $\text{Arc}^{\text{weak}}$ 是弱引用 Arc。`run_jit` 调用时升级为 `Arc::upgrade`，若失败则重新编译。

**Step 2（回收协议）**：

- 当某 chunk 的强引用计数归零（即无 `Frame` 引用该 chunk），其 cache 项的 `Weak` 升级失败，触发重新编译。
- `chunk_idx` 仍不回收（保持 FPA），但 cache 项可被回收。

**Step 3（未解决的并发难题）**：

1. **Frame 引用计数维护**：`Frame`（[`runtime/vm.rs:515`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/vm.rs)）需携带 `Arc<Chunk>` 以维持强引用。但 `Frame` 是栈帧，频繁创建销毁，Arc 的原子操作开销可能抵消 JIT 收益。
2. **TOCTOU 竞态**：`upgrade` 成功后、`invoke_jit` 调用前，若另一线程回收该 chunk，将调用已释放机器码。需引入锁或 hazard pointer，但 Tenth VM 当前是单线程模型（`Vm: !Sync`），多线程扩展需重构。
3. **JITModule 释放顺序**：`JITModule` 持有所有机器码的所有权。单函数回收要求 `JITModule` 支持细粒度释放（`free_function`），但 Cranelift 的 `JITModule` 不提供此 API（仅支持 `finish` 整体释放）。

**结论**：K5 给出引用计数回收的形式化骨架，但三个并发难题（Frame 引用计数开销、TOCTOU 竞态、JITModule 细粒度释放）使其在当前架构下不可落地。标注为未来工作。$\blacksquare$

---

## 6. JIT 缓存生命周期模型

基于 §4 的形式化建模，本节给出 Tenth JIT 缓存的完整生命周期模型。

### 6.1 生命周期的五阶段

**阶段 1（初始化）**：`JitContext::new()`（[`context.rs:20-33`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/jit/context.rs)）创建空的 `JITModule` 与空 `cache`。状态：$\sigma_0 = (\emptyset, \text{chunks}_0, \text{module}_0)$。

**阶段 2（按需编译）**：首次调用 `get_or_compile(chunk_idx, chunk)`（[`context.rs:36-58`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/jit/context.rs)）触发规则 T1，编译并缓存。

**阶段 3（缓存命中）**：后续调用同一 `chunk_idx` 触发规则 T2，直接返回缓存指针。

**阶段 4（持续增长）**：新 chunk 经 `add_fn` 加入 `Vm::chunks`（[`runtime/vm.rs:297-302`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/vm.rs)），其首次 JIT 调用触发规则 T1。cache 单调增长。

**阶段 5（Drop 释放）**：`JitContext` 被丢弃时触发规则 T3，`cache.clear()` 后 `JITModule` 释放。

### 6.2 状态迁移图

```
   σ_0 (空缓存)
    │
    │ T1 (编译并缓存)
    ▼
   σ_1 (cache = {i₁ ↦ f₁})
    │
    │ T2 (缓存命中)  ──自循环
    │ T1 (新 chunk 编译)
    ▼
   σ_2 (cache = {i₁ ↦ f₁, i₂ ↦ f₂})
    │
    │ ... (单调增长)
    ▼
   σ_n (cache = {i₁ ↦ f₁, ..., i_n ↦ f_n})
    │
    │ T3 (Drop)
    ▼
   σ_∞ (cache = ∅, module = ⊥)
```

**性质 6.1（无回退迁移）**：状态迁移图中无"从 $\sigma_{k}$ 回退到 $\sigma_{k-1}$"的边。即 cache 一旦增长即不收缩（直到 Drop）。这是 FPA 的直接推论。

**性质 6.2（终态唯一性）**：所有生命周期路径均终止于 $\sigma_\infty$。这是 Drop 语义的保证。

### 6.3 与代码热加载的不兼容性

代码热加载（hot reload）要求：**在运行时替换某函数的实现，且不影响已运行的调用栈**。形式化，热加载操作 $H(i, \text{chunk}')$ 将 $\text{chunks}(i)$ 替换为 $\text{chunk}'$，并使后续调用 $i$ 执行 $\text{chunk}'$。

在 Tenth 的生命周期模型中，$H$ 的实现需：
1. 更新 $\text{chunks}(i) := \text{chunk}'$；
2. 使 $\text{cache}(i)$ 失效（因旧 cache 指针对应旧 chunk 的机器码）；
3. 重新编译 $\text{chunk}'$ 并更新 $\text{cache}(i)$。

但 FPA（K1）假设 $\text{chunks}(i)$ 不变更，故 $H$ 直接破坏 FPA。即便强行实现 $H$，需解决：
- (a) 已运行的 JIT 栈帧如何处理（其机器码对应旧 chunk）；
- (b) `chunk_idx` 重用如何与 cache 失效协调；
- (c) `JITModule` 不支持单函数释放（K5 Step 3），旧机器码无法回收。

故 Tenth 当前架构不支持代码热加载。$\square$

---

## 7. 不动点假设的分析

### 7.1 FPA 的实证强度

K1 的证明依赖"源码搜索无 chunks 删除操作"的实证证据。这一证据的强度分析：

**支持证据**：
- `Grep "chunks\.push|chunks\.remove|chunks\.swap_remove"` 在 `vm.rs` 中仅匹配 `chunks.push(chunk)`（[`runtime/vm.rs:299`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/vm.rs)）。
- `add_fn` 是 `Vm` 的公开 API 中唯一向 `chunks` 添加元素的方法。
- Tenth 的 chunk 模型是"编译期确定 + 运行期不可变"，无动态卸载需求。

**反例风险**：
- 若未来引入 `Vm::remove_fn` 或 `Vm::reload_fn`，FPA 破坏。
- 若 `chunks` 改为 `HashMap<usize, Chunk>` 以支持稀疏索引，FPA 的"索引单调增长"仍成立，但"索引永不回收"需重新审视。
- 若引入热重载子系统（§6.3），FPA 必须破坏。

**结论**：FPA 在当前 v0.3.3 源码中实证成立，但**非逻辑必然**。任何引入 chunk 卸载/重载的改造均需重新审视 FPA，并同步更新 K1 的证明。

### 7.2 FPA 破坏的后果

假设 FPA 被破坏，即存在某 $i \in \mathcal{C}$，$\text{chunks}(i)$ 在 $t_1$ 时刻被替换为 $\text{chunk}'$。则：

- $\text{cache}(i)$（若存在）指向旧 chunk 的机器码 $m$；
- $m$ 的语义对应旧 chunk，与 $\text{chunks}(i) = \text{chunk}'$ 不一致；
- 后续 `run_jit` 调用 $i$ 时，`get_or_compile` 缓存命中（规则 T2），返回 $m$，执行旧语义——**静默错误**。

形式化：设 $\text{sem}(m) = \text{sem}(\text{chunk})$（旧），但 $\text{chunks}(i) = \text{chunk}'$（新）。则 `run_jit` 的执行结果为 $\text{sem}(\text{chunk})(\text{args})$，而非 $\text{sem}(\text{chunk}')(\text{args})$。两者语义不等价（除非 $\text{chunk} \equiv \text{chunk}'$）。

**后果严重性**：静默错误是最严重的 bug 类型——无崩溃、无异常，仅结果错误。在 autodiff 场景下，可能导致 Tape 记录错误梯度，进而使训练发散。

### 7.3 FPA 的工程保证

为防止 FPA 破坏，建议以下工程保证：

1. **API 审计**：`Vm` 的公开 API 中不应出现 `remove_fn`、`reload_fn`、`swap_fn` 等修改 `chunks` 既有元素的方法。
2. **类型层面保证**：考虑将 `chunks: Vec<Chunk>` 改为 `chunks: AppendOnlyVec<Chunk>`（类型层面保证只追加）。但需评估对 `Vm` 其他方法的影响。
3. **测试覆盖**：添加测试验证 `chunks.len()` 在 `add_fn` 后单调增长，且无 API 能使其减少。
4. **文档警示**：在 `Vm::chunks` 字段的文档注释中明确"FPA 假设依赖此字段只追加不删除"。

---

## 8. 不可重定位性的影响

### 8.1 影响范围

K2 的不可重定位性影响所有调用 hostcall 的 JIT 机器码。搜索证据：`translator.rs` 中 `hostcall_addr` 被调用 19 处（[`translator.rs:583-587`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/jit/translator.rs) 等），`call_indirect` 被调用 21 处（[`translator.rs:606`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/jit/translator.rs) 等）。覆盖的 opcode 包括：
- 字面量构造（`host_make_int`、`host_make_float`、`host_make_bool`、`host_make_str`、`host_make_unit`）
- 全局变量（`host_load_global`、`host_store_global`）
- 控制流（`host_truthy`、`host_call`、`host_method_call`）
- 数据结构（`host_make_vec`、`host_make_map`、`host_new_struct`、`host_load_field`、`host_store_field`）
- 索引与切片（`host_index_get`、`host_slice_str`）

即 Tenth JIT 支持的全部 opcode（除纯标量算术外）均经 hostcall，故均受 K2 约束。

### 8.2 影响维度

**维度 1（内存碎片化）**：mcode 区不可迁移，长期运行后可能产生碎片。但 Tenth 的 `JITModule` 内部管理 mcode 区，碎片化程度取决于 Cranelift 的实现。

**维度 2（进程 fork）**：子进程继承父进程的 mcode 区，但 ASLR 使子进程的 hostcall 地址不同。由于 K2，子进程的 JIT 机器码内嵌父进程的 hostcall 地址，调用即崩溃。**Tenth JIT 不支持 fork 后子进程复用 mcode**。

**维度 3（代码热重载）**：§6.3 已分析，热重载需破坏 FPA 并解决 K5 的并发难题。

**维度 4（mmap 重映射）**：`mremap(2)` 系统调用可重映射内存区域，但 K2 使 mcode 区的重映射无效（机器码内嵌入旧地址）。

**维度 5（跨进程共享）**：共享内存（`shm_open`）允许多进程共享 mcode 区，但 K2 使共享无效（各进程的 hostcall 地址不同）。

### 8.3 缓解策略

**策略 1（PIC 改造）**：将 `is_pic` 改为 `true`，使 Cranelift 生成 PIC 机器码。但这需重新设计 `hostcall_addr`——PIC 模式下应通过 GOT 间接引用。Cranelift 的 `SymbolValue` 机制可生成 GOT 引用，但需评估对 `call_indirect` 的影响。**标注为未来工作**。

**策略 2（trampoline 间接层）**：在 mcode 区外维护一个 trampoline 表，每个 trampoline 跳转到对应 hostcall。JIT 机器码内嵌 trampoline 的地址（而非 hostcall 地址），trampoline 表可迁移。但这增加一次间接跳转的开销，且 trampoline 表本身仍需管理。

**策略 3（重定位表）**：在编译时记录所有嵌入地址的位置，迁移时扫描并重写。这需修改 Cranelift 或在 translator 层维护元数据。Cranelift 在 `is_pic = false` 下不生成重定位表，需自行维护。

**结论**：三种策略均需显著工程投入，且与 Tenth 的"保守 JIT"路线冲突。当前阶段，接受 K2 的不可重定位性是合理的工程权衡。

---

## 9. 与 V8 code aging / LuaJIT mcode 对比

### 9.1 对比矩阵

| 维度 | Tenth | V8 | LuaJIT | HotSpot |
|------|-------|-----|--------|---------|
| 缓存淘汰 | 无（Drop 时清空） | code aging flush | 页级释放 | sweeper 增量清理 |
| 索引回收 | 无（FPA） | 有 | 有 | 有 |
| 代码重定位 | 不支持（NRA） | 支持（PIC） | 支持（PIC + GOT） | 支持（PIC） |
| Deoptimization | 无 | 有 | 有 | 有 |
| OSR | 无 | 有 | 有 | 有 |
| 单函数释放 | 不支持 | 支持 | 不支持（页级） | 支持 |
| mcode 区迁移 | 不支持 | 支持 | 支持 | 支持 |
| fork 后复用 | 不支持 | 支持 | 支持 | 支持 |
| 热重载 | 不支持 | 支持 | 支持 | 支持 |

### 9.2 设计哲学对比

**Tenth 的"保守极简"哲学**：
- 优先正确性，牺牲功能（无淘汰、无重定位、无 deopt）；
- 依赖 FPA 简化缓存语义（缓存命中即有效，无需检查）；
- 适合嵌入式、短生命周期场景（脚本、REPL、CI）。

**V8 的"激进优化"哲学**：
- 优先性能，支持复杂淘汰与 deopt；
- 依赖 deoptimization 保证正确性（flush 后回退解释器）；
- 适合长期运行的浏览器场景。

**LuaJIT 的"中间路线"**：
- 页级管理，平衡复杂度与功能；
- PIC 支持迁移，但不支持单函数释放；
- 适合嵌入式 Lua 场景。

### 9.3 适用场景分析

Tenth 的策略在以下场景下合理：
- **短生命周期进程**：CI 脚本、REPL 会话，进程结束后 mcode 自然释放。
- **函数集稳定**：编译期确定的函数集，运行期不增不减。
- **内存预算充足**：mcode 累积不构成瓶颈。

Tenth 的策略在以下场景下不足：
- **长期运行服务**：mcode 累积导致内存泄漏。
- **动态代码生成**：eval、REPL 中的函数定义，每次定义新增 chunk，cache 单调增长。
- **热重载需求**：开发期的代码热替换。

---

## 10. 工程权衡与开放问题

### 10.1 工程权衡

**权衡 1（简化 vs 功能）**：FPA + NRA 极大简化了缓存语义（无需淘汰、无需重定位、无需 deopt），但关闭了热加载、动态卸载等高级功能。这是"保守 JIT"路线的必然代价。

**权衡 2（性能 vs 内存）**：永不回收策略在稳态性能上最优（无 flush 后重新 JIT 的开销），但内存累积。短期进程受益于性能，长期进程受害于内存。

**权衡 3（绝对地址 vs PIC）**：`is_pic = false` 使 `call_indirect` 能直接用绝对地址（[`context.rs:25-27`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/jit/context.rs) 注释），避免一次 GOT 间接引用。在 Windows x64 上，PIC 的 GOT 引用与 `call_indirect` 的兼容性存在问题（注释明示）。绝对地址方案在性能上略优，但牺牲可重定位性。

### 10.2 开放问题

**问题 1（mcode 累积的阈值）**：长期运行进程中，mcode 累积到何种阈值会构成瓶颈？需实测 `JITModule` 的内存占用随 chunk 数量的增长曲线。

**问题 2（PIC 改造的可行性）**：将 `is_pic` 改为 `true` 后，`call_indirect` 与 GOT 的兼容性如何？Cranelift 在 Windows x64 上的 PIC 支持是否成熟？

**问题 3（热重载的最小改造）**：若仅需支持"开发期热重载"（非生产环境），是否可在不破坏 FPA 的前提下实现？例如，每次重载分配新 `chunk_idx`，旧 idx 永不回收（FPA 保持），但 cache 项可被新 idx 覆盖。

**问题 4（多线程 VM 的 cache 一致性）**：若未来 Tenth VM 支持多线程，`JitContext` 的 `cache` 需线程安全。`HashMap` 非线程安全，需改造为 `RwLock<HashMap>` 或 `DashMap`。但这与 K5 的并发难题相关。

**问题 5（`Module::finish` 的 Drop 语义）**：`context.rs:65` 注释提及"`Module::finish` 消费 self，这里只能尽力清理"。`JITModule::drop` 是否完整释放机器码内存？需查阅 Cranelift 源码确认。

---

## 11. 局限

本节诚实记录本文的 6 处理论局限，每条说明是什么、影响多大、如何缓解。

### 局限 L1（FPA 的实证性而非逻辑必然）

**是什么**：K1 的证明依赖"源码搜索无 chunks 删除操作"的实证证据，而非类型系统或逻辑的必然保证。

**影响多大**：中等。当前 v0.3.3 源码中 FPA 成立，但未来引入 chunk 卸载/重载即破坏。FPA 破坏后，K1 失效，缓存可能出现静默错误（§7.2）。

**如何缓解**：
- 短期：在 `Vm::chunks` 字段的文档注释中明确"FPA 依赖此字段只追加不删除"。
- 中期：考虑将 `Vec<Chunk>` 改为 `AppendOnlyVec<Chunk>`（类型层面保证）。
- 长期：若引入热重载，需重新设计缓存协议（K5）。

### 局限 L2（`Module::finish` 的 Drop 语义不完备）

**是什么**：K3 证明"`cache.clear()` + `JITModule` Drop"无悬垂指针，但未证明 `JITModule::drop` 完整释放机器码内存。`context.rs:65` 注释明示"`Module::finish` 消费 self，这里只能尽力清理；失败可忽略"。

**影响多大**：低-中。即便 `JITModule::drop` 未完整释放，结果仅是内存泄漏（进程结束后由 OS 回收），非悬垂指针。但长期运行进程的内存占用可能超预期。

**如何缓解**：
- 短期：在 `Drop::drop` 中添加日志，记录 `cache.len()` 与 `module` 状态，便于排查。
- 中期：查阅 Cranelift `JITModule::drop` 源码，确认其释放行为。
- 长期：若 Cranelift 提供显式释放 API（如 `Module::finish`），改用显式释放。

### 局限 L3（K2 的跨平台 `is_pic` 行为差异）

**是什么**：K2 证明"`is_pic = false` 导致不可重定位"，但 `is_pic` 的实际行为在不同平台（x86、x64、ARM64）与不同 Cranelift 版本上可能不同。`context.rs:25-27` 注释明示"on Windows x64"，但 Linux/macOS 行为未明确。

**影响多大**：中。若 Linux/macOS 上 `is_pic = false` 的行为与 Windows x64 不同（如某些平台仍生成 PIC），K2 的适用范围需调整。

**如何缓解**：
- 短期：在论文中明确 K2 的适用平台（Windows x64）。
- 中期：在 Linux/macOS 上实测 `is_pic = false` 的机器码，验证 K2 的适用性。
- 长期：若需跨平台一致，改用 PIC + GOT 方案。

### 局限 L4（K5 的并发难题未解决）

**是什么**：K5 给出引用计数回收的形式化骨架，但三个并发难题（Frame 引用计数开销、TOCTOU 竞态、JITModule 细粒度释放）未解决，标注为未来工作。

**影响多大**：高（对热重载需求）。若需支持热重载，K5 的难题必须解决。当前 Tenth 无热重载需求，故不影响生产。

**如何缓解**：
- 短期：接受 K5 不可落地，明确 Tenth 不支持热重载。
- 中期：若需热重载，优先探索"问题 3（热重载的最小改造）"——分配新 `chunk_idx` 而非回收旧 idx。
- 长期：若需生产级热重载，解决 K5 的三个并发难题。

### 局限 L5（K4 对比的功能维度而非性能维度）

**是什么**：K4 证明 Tenth 在功能维度上严格弱于 V8/LuaJIT，但未在性能维度上对比。Tenth 的"永不回收"策略在稳态性能上可能优于 V8（无 flush 后重新 JIT 的开销）。

**影响多大**：低。本文定位为"生命周期与热加载"分析，性能对比非核心。但读者可能误读 K4 为"Tenth 整体弱于 V8/LuaJIT"。

**如何缓解**：
- 短期：在 K4 的注释中明确"功能维度对比，非性能维度"。
- 中期：补充性能对比实验（mcode 累积曲线、稳态吞吐量）。

### 局限 L6（T31/T32 联动缺口）

**是什么**：任务描述要求本文与"T31/T32（JIT 系列）"联动，但撰写时 `docs/论文/` 目录中未见 T31、T32 论文（搜索证据：`Glob "docs/论文/T3*.md"` 仅返回 T30 与 T3，未返回 T31/T32）。本文仅能与已存在的 T9 联动。

**影响多大**：低。本文与 T9 的联动已覆盖 JIT 系列的核心理论点。T31/T32 的缺失不影响本文的独立性。

**如何缓解**：
- 短期：本文以"前瞻引用"方式提及 T31/T32，待 T31/T32 撰写后补充双向链接。
- 中期：在 `MEMO.md` 中记录 T31/T32 的规划，确保后续撰写。

---

## 12. 结论

本文对 Tenth 语言 JIT 编译子系统的缓存生命周期与代码热加载能力进行了完整的形式化分析。核心结论：

1. **K1（不动点假设实证性成立）**：`chunk_idx` 在当前 v0.3.3 源码中永不回收，`Vm::chunks` 仅追加不删除（[`runtime/vm.rs:297-302`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/vm.rs)）。FPA 是实证性成立，非逻辑必然。

2. **K2（不可重定位性）**：`is_pic = false`（[`context.rs:27`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/jit/context.rs)）+ `hostcall_addr` 的 `iconst` 嵌入（[`translator.rs:583-587`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/jit/translator.rs)）共同导致 JIT 机器码对 hostcall 地址不可重定位。影响包括：不支持 mcode 迁移、fork 后子进程复用、代码热重载、mmap 重映射、跨进程共享。

3. **K3（Drop 正确性）**：`cache.clear()` 先于 `JITModule` 隐式 Drop（[`context.rs:61-69`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/jit/context.rs)）保证无悬垂指针。`cache.clear()` 提供"防御性编程"语义，对未来 cranelift 版本变更 Drop 行为具有韧性。

4. **K4（与 V8/LuaJIT 对比）**：Tenth 在单函数级缓存淘汰、mcode 区迁移、deoptimization 回退三个功能维度上严格弱于 V8/LuaJIT。这是"保守 JIT"路线的必然代价。

5. **K5（引用计数回收方案，未来工作）**：给出 `Arc<Weak<JitFn>>` 回收方案的形式化骨架，但三个并发难题（Frame 引用计数开销、TOCTOU 竞态、JITModule 细粒度释放）使其在当前架构下不可落地。

本文的结论对实施的指导：
- **短期**：接受 FPA + NRA 的工程权衡，在 `Vm::chunks` 文档注释中明确 FPA 依赖；在 `Drop::drop` 中添加日志便于排查。
- **中期**：若需热重载，优先探索"分配新 `chunk_idx` 而非回收旧 idx"的最小改造（问题 3）。
- **长期**：若需生产级热重载或多线程 VM，解决 K5 的并发难题，并评估 PIC 改造的可行性。

本文为 Tenth JIT 缓存淘汰策略、热重载子系统、PIC 重定位改造等未来工作奠定了形式化基础。

---

## 参考文献

[1] Jones, N.D., Gomard, C.K., Sestoft, P. *Partial Evaluation and Automatic Program Generation*. Prentice Hall, 1993.

[2] Chromium Project. *V8 Code Aging*. https://v8.dev/blog/code-aging, 2013.

[3] Pall, M. *LuaJIT 2.0 JIT Compiler*. http://luajit.org/luajit.html, 2011.

[4] Kotzmann, T., Mössenböck, H. *Run-time Support for Optimizations in the Java HotSpot VM*. Concurrency and Computation: Practice and Experience, 2007.

[5] Cranelift Project. *Cranelift JIT Module Documentation*. https://docs.rs/cranelift-jit/, 2024.

[6] Tenth 项目数理部. *T9-JIT特化语义保持证明：基于部分求值理论的双模拟论证*. `docs/论文/T9-JIT特化语义保持证明.md`, 2026.

[7] Tenth 项目数理部. *T3-HIR约束求解NP完全性归约*. `docs/论文/T3-HIR约束求解NP完全性归约.md`, 2026.

[8] Tenth 项目. *工作规范 v1.1*. `.trae/rules/工作规范.md`, 2026.

[9] Tenth 项目. *MEMO.md：逐版变更记录*. `MEMO.md`, 2026.

[10] Tenth 项目. *CODE_WIKI.md：模块架构*. `CODE_WIKI.md`, 2026.

---

## 附录 A：定理索引

| 定理 | 名称 | 核心结论 | 源码引用 |
|------|------|---------|---------|
| K1 | 不动点假设实证性 | `chunk_idx` 永不回收 | [`runtime/vm.rs:297-302`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/vm.rs) |
| K2 | 不可重定位性 | `is_pic=false` 导致 JIT 代码不可重定位 | [`context.rs:27`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/jit/context.rs), [`translator.rs:583-587`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/jit/translator.rs) |
| K3 | Drop 正确性 | `cache.clear()` + `JITModule` Drop 无悬垂指针 | [`context.rs:61-69`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/jit/context.rs) |
| K4 | 与 V8/LuaJIT 对比 | Tenth 热加载能力严格弱于 V8/LuaJIT | （对比分析，无单一源码引用） |
| K5 | 引用计数回收方案 | 形式化骨架，并发难题未解决 | （未来工作，无源码引用） |

## 附录 B：与现有文档的对应

| 本文章节 | 对应文档 | 关系 |
|---------|---------|------|
| §1.3 两条隐含假设 | T9 §11 局限 L2、L3 | 本文扩展 T9 的局限披露为完整证明 |
| §4 形式化建模 | T9 §3 形式化模型 | 本文聚焦缓存生命周期，T9 聚焦特化语义 |
| §5.2 K2 不可重定位性 | T9 §11 局限 L3 | 本文给出完整证明，T9 仅提及 |
| §5.4 K4 对比 | T9 §2.2 V8 deoptimization | 本文扩展对比至 LuaJIT、HotSpot |
| §11 局限 L6 | 任务描述 T31/T32 联动 | 诚实披露 T31/T32 缺失 |

## 附录 C：实施建议

### C.1 短期（v0.3.x）

1. 在 `Vm::chunks` 字段的文档注释中明确"FPA 依赖此字段只追加不删除"。
2. 在 `JitContext::drop` 中添加 `log::debug!("JitContext dropping, cache size: {}", self.cache.len())` 日志。
3. 在 `MEMO.md` 中记录 FPA + NRA 的工程权衡。

### C.2 中期（v0.4.x）

1. 若需热重载，实现"分配新 `chunk_idx` 而非回收旧 idx"的最小改造。
2. 评估 `AppendOnlyVec<Chunk>` 的可行性。
3. 实测 mcode 累积曲线，确认内存瓶颈阈值。

### C.3 长期（v0.5+）

1. 若需多线程 VM，改造 `cache` 为线程安全结构。
2. 若需生产级热重载，解决 K5 的三个并发难题。
3. 评估 PIC 改造（`is_pic = true` + GOT）的可行性与性能影响。

---

> **数理部自审留痕**：本文经历 4 轮自审（§1.6），核心修正包括 K2 从"完全不可重定位"修正为"对 hostcall 地址不可重定位"、K3 从"充分必要"修正为"仅充分不必要"、K5 从"可落地"修正为"未来工作"。所有定理的源码引用均已核对，行号基于 v0.3.3 源码。局限章节独立成节，6 处局限均含"是什么、影响多大、如何缓解"三要素。
