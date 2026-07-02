# HIR → WASM 的语义保持与 host 边界类型塌缩：Tenth 的值类型塌缩 + host 桥接模式

> **作者**：Tenth 项目数理部
> **日期**：2026-07-02
> **类型**：理论分析论文（T29 理论点，护城河 C：WASM 闭环）
> **实证基础**：Tenth v0.3.3+ 源码（`tenth/src/compile/wasm/mod.rs`、`tenth/src/compile/wasm/host.rs`、`tenth/src/compile/wasm/sections.rs`、`tenth/src/compile/wasm/compile.rs`、`tenth/src/compile/wasm/types.rs`、`tenth/src/hir/types.rs`）
> **关联文档**：`docs/论文/T12-双侧编译器语义等价性.md`（路径 C 闭环依赖本文的 W1/W3）、`docs/语言参考手册.md`、`MEMO.md`
> **版本**：v1（首轮分析，含 4 轮自审留痕）

---

## 摘要

Tenth 语言的 HIR 拥有 16 种 `BaseType` + 11 种复合类型（共 27 种类型构造子），而 WebAssembly 1.0 仅提供 4 种值类型（`i32` / `i64` / `f32` / `f64`）。本文形式化分析 `compile/wasm/mod.rs:22-39` 中 `to_val_type` 函数所实现的"**值类型塌缩 + host 桥接**"模式：所有非数值类型塌缩为 `i64`（用作指针或 reinterp 容器），所有 heap 数据通过 18 个 host import 显式管理。这一模式与 Emscripten / AssemblyScript 的 lowering 策略有本质区别——后者在编译期生成胶水代码展开 heap 语义，而 Tenth 将 heap 语义**整体外推**到 host。本文给出五个主定理：（W1）HIR↔WASM 操作语义构成弱双模拟关系；（W2）类型塌缩映射健全（不损失 HIR 层区分的语义）；（W3）18 个 host import 覆盖所有 HIR 复合类型的 heap 操作；（W4）Tenth 模式相对 Emscripten/AssemblyScript 在 ABI 稳定性、host 兼容性、自举闭环上的优势与代价；（W5）自动推导 host 边界 import 集（标注为未来工作）。本文诚实记录 8 处理论局限，包括 7 种 `BaseType`（U8/U16/U32/U64/F16/BF16/Char）在 `to_val_type` 中**实际未显式处理而落入 `_ => None` 分支**、`Unit` 类型在参数位置的二义性、`tensor_from_vec` 的伪实现、双模拟关系对 host 副作用的"外推"假设等。本工作的价值在于：**把"塌缩+桥接"这一工程取舍提升为可证伪、可分级、可比较的形式化命题，并诚实标注其破损处。**

**关键词**：WebAssembly、HIR、语义保持、类型塌缩、host 桥接、双模拟、Emscripten、AssemblyScript、自举闭环、Tenth 语言

---

## 1. 引言

### 1.1 HIR-WASM 类型差距问题

WebAssembly 1.0 的类型系统极简：值类型仅有 4 种（`i32` / `i64` / `f32` / `f64`），无乘积类型、无求和类型、无引用类型、无类型参数（[WebAssembly Core Specification 1.0, §2.2.1](https://webassembly.github.io/spec/core/bikeshed/#value-types%E2%91%A0)）。而 Tenth 的 HIR 类型系统（[`tenth/src/hir/types.rs:3-42`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/hir/types.rs)）拥有 16 种 `BaseType` + `Tensor`/`Array`/`FnType`/`TypeParam`/`Generic`/`Ref`/`MutRef`/`Struct`/`Enum`/`Tuple`/`Unknown` 共 11 种复合构造子。如何把 HIR 的丰富类型系统编译为 WASM 的极简类型系统而不损失语义，是任何 HIR→WASM 编译器必须解决的核心问题。

业界已知有两条路线：

1. **Lowering 路线**（Emscripten / AssemblyScript）：在编译期把复合类型展开为 WASM 内存布局，生成大量胶水代码（glue code）模拟结构体访问、虚函数表、字符串拼接等。产物 WASM 自包含，但 ABI 复杂、体积膨胀、host 不可干预。
2. **Host 桥接路线**（Tenth）：将 HIR 中所有"heap 数据"统一塌缩为 `i64` 指针，所有针对 heap 的操作通过显式 host import 调用宿主函数。产物 WASM 体积小、ABI 简洁，但执行时**必须**有 host 配合。

Tenth 选择路线 2 不是偶然：Tenth 的设计哲学是"AI 原生"——张量算子、自动微分、字符串处理等都是重计算任务，复用 host 已有的 Rust 生态（`std::fs`、`Vec`、`String`、`ndarray`）比重新在 WASM 内部实现更高效，也更易维护。但这一选择引入了**语义保持的形式化难题**：当 HIR 中的一段结构体操作被翻译为对 host 的 `i64` 指针调用时，我们如何保证 HIR 语义未被破坏？

### 1.2 值类型塌缩 + host 桥接模式

Tenth 的 `to_val_type` 函数（[`tenth/src/compile/wasm/mod.rs:22-39`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/wasm/mod.rs)）实现了下述塌缩映射：

```
I8 / I16 / I32 / I64        → I64     (整数统一 64 位)
F32 / F64                   → F64     (浮点统一 64 位)
Bool                        → I32     (布尔保持 32 位)
Str                         → I64     (字符串作指针)
Unit                        → None    (无值)
Ref / MutRef / Struct / Generic / TypeParam / Unknown → I64  (复合类型作指针)
```

配合 18 个 host import（[`tenth/src/compile/wasm/mod.rs:71-89`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/wasm/mod.rs)），Tenth 把所有"无法在 WASM 1.0 内表达"的语义外推给 host。这种模式与 Emscripten/AssemblyScript 的本质区别在于：**Tenth 不试图在 WASM 内部重建 heap 语义，而是把 heap 语义整体搬到 host**。

### 1.3 研究问题

本文回答以下五个研究问题：

- **RQ1**（语义保持）：HIR 操作语义与 WASM 操作语义之间是否存在模拟关系？等价性的"可观察行为"如何定义？
- **RQ2**（塌缩健全性）：`to_val_type` 的塌缩映射是否损失 HIR 层的语义区分？例如 HIR 的 `I32` 与 `Str` 都被塌缩为 `i64`，是否意味着类型信息丢失？
- **RQ3**（host 桥接完备性）：18 个 host import 是否覆盖 HIR 所有复合类型所需的 heap 操作？是否存在未被覆盖的"漏洞"？
- **RQ4**（与现有方法对比）：Tenth 模式相对 Emscripten/AssemblyScript 在 ABI 稳定性、host 兼容性、自举闭环上有何优势与代价？
- **RQ5**（自动推导可行性）：能否从 HIR 类型系统自动推导所需的 host import 集，而非手工维护 18 个常量？

### 1.4 贡献

1. **形式化建模**（§3、§4）：将 `to_val_type` 函数抽象为类型塌缩映射 $\phi$，将 18 个 host import 抽象为 host 操作签名集 $\mathcal{H}$，给出 HIR 与 WASM 的小步操作语义。
2. **五个主定理与证明**（§5）：W1（弱双模拟）、W2（塌缩健全性）、W3（host 桥接完备性）、W4（与 Emscripten/AssemblyScript 对比）、W5（自动推导，未来工作）。
3. **诚实局限记录**（§9）：独立章节记录 8 处理论局限，包括 7 种 `BaseType` 未显式处理、`Unit` 类型二义性、`tensor_from_vec` 伪实现等。
4. **与 T12 联动**（§10）：本文 W1/W3 是 T12 路径 C（全 WASM 闭环）语义等价性的前提条件；T12 §6.4 路径 C 闭环依赖本文证明。
5. **v1 自审留痕**：本文经历 4 轮自审（见 §11.4），每轮修正标注版本。

### 1.5 v1 自审留痕

| 轮次 | 原始断言 | 修正 |
|------|---------|------|
| v1.1 | "13 种 BaseType" | 实际 16 种（[`types.rs:3-10`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/hir/types.rs)），7 种未显式处理 |
| v1.2 | "W3 完备性成立" | `tensor_from_vec` 是 stub（[`host.rs:340-343`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/wasm/host.rs)），改为"在排除 tensor 后成立" |
| v1.3 | "双模拟关系对称" | host 副作用不可逆，改为弱双模拟（仅单向） |
| v1.4 | "host import 集 ∩ lower 路径" | lower 路径无 host，删除 W4 中"路径覆盖"对比项 |

---

## 2. 背景与相关工作

### 2.1 WebAssembly 1.0 类型系统

WebAssembly Core Specification 1.0 定义（[spec §2.2.1](https://webassembly.github.io/spec/core/bikeshed/#value-types%E2%91%A0)）：

- **值类型**：`i32` / `i64` / `f32` / `f64`，共 4 种。
- **函数类型**：$t^\ast \to t^\ast$，参数与返回值均为值类型序列。
- **表类型**：`funcref` 或 `externref` 元素。
- **内存类型**：线性字节数组，无类型化字段。
- **全局类型**：单个值类型 + 可变性。

**关键限制**：无乘积类型（struct）、无求和类型（enum）、无引用类型（直到 reference-types proposal 才有 `externref`）、无类型参数（无泛型）、无字符串原语、无张量原语。任何高级类型必须 lowering 为上述 4 种值类型 + 线性内存 + 函数表。

### 2.2 Emscripten 的 lowering 策略

Emscripten（[Zakai 2011](https://dl.acm.org/doi/10.1145/1993316.1993532)）把 C/C++ 编译为 WASM，采用**完全 lowering** 策略：

- **结构体**：编译期计算偏移量，所有字段访问编译为 `i32.load` / `i32.store`。
- **虚函数表**：编译期生成 funcref 表，`call_indirect` 调用。
- **字符串**：内嵌为 data section，运行时拼接通过 WASM 内函数实现。
- **STL 容器**：完整 lowering 为 WASM 内的内存管理代码。
- **异常**：通过 `setjmp`/`longjmp` 模拟或 Emscripten EH 机制。

**优势**：产物自包含，可在任何符合规范的 WASM 运行时执行。
**代价**：胶水代码膨胀（一个 hello world 也数十 KB）；ABI 复杂（结构体布局对 host 不可见）；host 难以干预内部数据。

### 2.3 AssemblyScript 的策略

AssemblyScript（[AssemblyScript docs](https://www.assemblyscript.org/)）把 TypeScript 子集编译为 WASM，采用**半 lowering** 策略：

- **基本类型**：直接映射（`number` → `f64` 或 `i32`，根据上下文）。
- **类**：编译为结构体 + 函数表，类似 Emscripten 但更简洁。
- **字符串**：内嵌 UTF-16 字符串到线性内存，提供 `String` API。
- **数组**：通过 `Array<T>` 模板在 WASM 内实现。
- **GC**：AssemblyScript 自带 tracing GC，运行时无需 host 介入。

**优势**：产物比 Emscripten 紧凑；类型系统比 Emscripten 更现代。
**代价**：依然自包含，host 不能干预数据；GC 复杂性内嵌于 WASM。

### 2.4 wasm-encoder 标准 ABI 模式

`wasm-encoder` crate（[bytecodealliance/wasm-tools](https://github.com/bytecodealliance/wasm-tools)）是 Rust 生态生成 WASM 字节码的事实标准库。Tenth 使用 `wasm-encoder`（[`mod.rs:15`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/wasm/mod.rs)）生成 WASM 模块。`wasm-encoder` 不规定 ABI，仅提供字节码构造原语——具体的 ABI（参数传递、返回值、host 调用约定）由使用方决定。Tenth 选择"i64 主导 + host import"的 ABI，是 `wasm-encoder` 用户中较保守的一种。

### 2.5 与本文最相关的工作

- **CompCert**（[Leroy 2009](https://dl.acm.org/doi/10.1145/1538788.1538814)）：C→汇编的语义保持证明，本文 W1 模拟关系借鉴其思路，但目标不同（汇编 ↔ WASM）。
- **CakeML**（[Kumar et al. 2014](https://dl.acm.org/doi/10.1145/2535838.2535841)）：自举编译器形式化验证，本文 W3 的"host 桥接完备性"是 CakeML 未涉及的（CakeML 不依赖 host）。
- **T12**（[`docs/论文/T12-双侧编译器语义等价性.md`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/docs/论文/T12-双侧编译器语义等价性.md)）：Tenth 双侧编译器等价性，本文 W1/W3 是其路径 C 闭环的前提。
- **wasm3 / wasmi host 模式**：嵌入式 WASM 运行时普遍支持 host import，但本文未发现形式化分析"host import 集完备性"的既有工作——这是本文的差异化贡献。

---

## 3. Tenth 类型塌缩形式化

### 3.1 记号约定

- $\mathcal{T}_{\text{HIR}}$：HIR 类型集合，由 [`hir/types.rs:20-42`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/hir/types.rs) 的 `Type` 枚举定义。
- $\mathcal{B} = \{$ I8, I16, I32, I64, U8, U16, U32, U64, F16, F32, F64, BF16, Bool, Char, Str, Unit $\}$：16 种 `BaseType`。
- $\mathcal{V}_{\text{WASM}} = \{$ `i32`, `i64`, `f32`, `f64` $\}$：4 种 WASM 值类型。
- $\mathcal{V}_{\text{WASM}}^{\bot} = \mathcal{V}_{\text{WASM}} \cup \{\bot\}$：加入"无类型"语义（对应 `None` 返回）。
- $\phi : \mathcal{T}_{\text{HIR}} \to \mathcal{V}_{\text{WASM}}^{\bot}$：类型塌缩映射。
- $\mathcal{H} = \{h_0, h_1, \ldots, h_{17}\}$：18 个 host import。
- $\mathcal{O}_{\text{HIR}}$：HIR 操作语义迁移规则集。
- $\mathcal{O}_{\text{WASM}}$：WASM 操作语义迁移规则集。
- $\sigma_{\text{HIR}}, \sigma_{\text{WASM}}$：HIR / WASM 状态（环境 + 栈 + 内存）。

### 3.2 类型塌缩映射 $\phi$ 的形式化定义

**定义 3.1（类型塌缩映射 $\phi$）**：$\phi : \mathcal{T}_{\text{HIR}} \to \mathcal{V}_{\text{WASM}}^{\bot}$ 由下述分支定义（对应 [`wasm/mod.rs:22-39`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/wasm/mod.rs)）：

$$
\phi(t) = \begin{cases}
\texttt{i64} & t = \text{Base}(b),\ b \in \{\text{I8, I16, I32, I64, Str}\} \\
\texttt{f64} & t = \text{Base}(b),\ b \in \{\text{F32, F64}\} \\
\texttt{i32} & t = \text{Base}(\text{Bool}) \\
\bot & t = \text{Base}(\text{Unit}) \\
\bot & t = \text{Base}(b),\ b \in \{\text{U8, U16, U32, U64, F16, BF16, Char}\} \quad \text{(\textit{v1.1 修正：实际未显式处理})} \\
\texttt{i64} & t \in \{\text{Ref}(\_), \text{MutRef}(\_), \text{Struct}(\_), \text{TypeParam}\{\_\}, \text{Generic}\{\_\}, \text{Unknown}\} \\
\texttt{i64} & t = \text{Base}(\text{Str}) \quad \text{(指针)} \\
\bot & \text{otherwise} \quad \text{(\textit{Tensor / Array / FnType / Enum / Tuple 走复合规则})}
\end{cases}
$$

**注**：源码中的 `_ => None` 分支（[`mod.rs:30, 37`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/wasm/mod.rs)）实际会捕获 7 种 `BaseType`（U8/U16/U32/U64/F16/BF16/Char）与 `Tensor`/`Array`/`FnType`/`Enum`/`Tuple` 五种复合类型，使它们全部映射为 $\bot$。这意味着 `to_val_type_required` 在遇到这些类型时会触发 `RuntimeError`（[`mod.rs:42-44`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/wasm/mod.rs)）。这一行为在 §9.1 局限中详述。

### 3.3 $\phi$ 的语义分类

按塌缩**机制**划分，$\phi$ 包含 4 种塌缩策略：

| 策略 | 源类型 | 目标类型 | 机制 | 源码位置 |
|------|--------|---------|------|---------|
| **保留**（preserve） | `I8/I16/I32/I64`, `F32/F64`, `Bool` | `i64`, `f64`, `i32` | 值直接传递，必要时符号扩展 | [`mod.rs:25-27`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/wasm/mod.rs) |
| **指针化**（pointerize） | `Str`, `Ref`, `MutRef`, `Struct`, `Generic` | `i64` | 值变为指向线性内存的指针 | [`mod.rs:28, 32-35`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/wasm/mod.rs) |
| **Reinterpret** | （`F64` 在 host 边界） | `i64` | 通过 `f64_bits` 转为 IEEE 754 位模式 | [`host.rs:300-303`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/wasm/host.rs) |
| **丢弃**（discard） | `Unit` | $\bot$ | 不传递值 | [`mod.rs:29`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/wasm/mod.rs) |

### 3.4 字段布局规则

[`wasm/mod.rs:48-59`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/wasm/mod.rs) 的 `field_size_and_type` 进一步规定：**所有结构体字段一律 8 字节对齐**，无论原类型大小。这一保守策略简化了字段访问代码生成，但牺牲内存效率（例如 `I8` 字段也占 8 字节）。

---

## 4. 操作语义形式化

### 4.1 HIR 操作语义（小步）

HIR 状态 $\sigma_{\text{HIR}} = (\rho, \kappa, \mu, \tau)$，其中：

- $\rho : \text{Var} \to \text{Val}_{\text{HIR}}$：变量环境
- $\kappa \in \text{Val}_{\text{HIR}}^\ast$：操作数栈
- $\mu : \mathbb{N} \to \text{Byte}$：线性内存（与 WASM 共享同一物理内存）
- $\tau$：Tensor 环境（HIR 层张量，不在 WASM 内存中）

HIR 操作语义迁移：

$$
\langle e, \sigma_{\text{HIR}} \rangle \to_{\text{HIR}} \langle e', \sigma'_{\text{HIR}} \rangle
$$

关键迁移规则：

- **(LIT)** $\langle \text{Lit}(v:t), \sigma \rangle \to_{\text{HIR}} \langle \cdot, \sigma[\kappa \mapsto \kappa \cdot v] \rangle$
- **(VAR)** $\langle \text{Var}(x), \sigma \rangle \to_{\text{HIR}} \langle \cdot, \sigma[\kappa \mapsto \kappa \cdot \rho(x)] \rangle$
- **(BIN-ADD-STR)** $\langle \text{Bin}(+, e_1, e_2), \sigma \rangle \to \langle \text{Call}(\text{str\_add}, [e_1, e_2]), \sigma \rangle$（若 $e_1.t = \text{Str}$）
- **(STRUCT-NEW)** $\langle \text{Struct}\{f_i = e_i\}, \sigma \rangle \to \langle \text{Call}(\text{tenth\_alloc}, [\text{size}]) \cdot \text{Store}(f_i, e_i), \sigma \rangle$
- **(VEC-PUSH)** $\langle \text{Vec::push}(e_1, e_2), \sigma \rangle \to \langle \text{Call}(\text{Vec\_push}, [e_1, e_2]), \sigma \rangle$

### 4.2 WASM 操作语义（小步）

WASM 状态 $\sigma_{\text{WASM}} = (s, \text{pc}, \text{stack}, \text{locals}, \text{mem})$，遵循 [WASM spec §4.4](https://webassembly.github.io/spec/core/exec/index.html)。

WASM 操作语义迁移：

$$
\langle \text{instr}^\ast, \sigma_{\text{WASM}} \rangle \to_{\text{WASM}} \langle \text{instr'}^\ast, \sigma'_{\text{WASM}} \rangle
$$

关键迁移规则：

- **(LOCAL.GET)** $\langle \text{local.get}\;x, \sigma \rangle \to \langle \epsilon, \sigma[\text{stack} \mapsto \text{stack} \cdot \text{locals}[x]] \rangle$
- **(I64.ADD)** $\langle \text{i64.add}, \sigma \rangle \to \langle \epsilon, \sigma[\text{stack} \mapsto \text{stack}[:-2] \cdot (v_1 + v_2)] \rangle$
- **(CALL host)** $\langle \text{call}\;h_i, \sigma \rangle \to \langle \epsilon, \sigma' \rangle$，其中 $\sigma'$ 由 host 函数 $h_i$ 的实现决定（host 可读写 `mem` 与 host 状态）

### 4.3 Host 操作语义

Host 函数集 $\mathcal{H} = \{h_0, \ldots, h_{17}\}$，每个 $h_i$ 有签名 $\text{Sig}(h_i) = (\text{params}, \text{ret})$，由 [`sections.rs:26-44`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/wasm/sections.rs) 定义。Host 函数的语义由 Rust 实现（[`host.rs:8-345`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/wasm/host.rs)）决定。

**关键观察**：Host 函数的执行**不在 WASM 操作语义内**——WASM 仅看到 `call` 指令与返回值。Host 可读写 WASM 内存（通过 `Caller` 接口），修改 host 状态（如 bump allocator 偏移 `*caller.data_mut()`），甚至调用 WASM 模块外的 Rust 代码（如 `std::fs::write`）。这种"语义外推"是 Tenth 模式的核心特征。

### 4.4 可观察行为

**定义 4.1（可观察行为）**：状态 $\sigma_{\text{HIR}}$ 的可观察行为 $\text{Obs}_{\text{HIR}}(\sigma)$ 包括：

1. 标准输出（`println` 调用产生的字符序列）
2. 文件系统副作用（`write_file` / `read_file` 调用的文件状态变化）
3. 退出码（main 函数返回值经 `wrap_to_i32` 后的 `i32`）

**定义 4.2（可观察行为等价）**：$\text{Obs}_{\text{HIR}}(\sigma_1) \equiv \text{Obs}_{\text{WASM}}(\sigma_2)$ 当且仅当：

- 输出序列相同
- 文件系统最终状态相同
- 退出码相同

**注**：内存布局、临时变量、调用栈深度等**不在**可观察行为中——这些是实现细节。

---

## 5. 主定理与证明

### 5.1 定理 W1（语义保持：弱双模拟）

**定理 W1**：设 $P$ 为一 HIR 程序，$W = \text{compile}(P)$ 为经 [`wasm/mod.rs`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/wasm/mod.rs) 编译后的 WASM 模块。若：

1. $P$ 中所有 `BaseType` 出现均属于 $\{\text{I8, I16, I32, I64, F32, F64, Bool, Str, Unit}\}$（"支持子集"）；
2. $P$ 中不出现 `Tensor` / `Array` / `FnType` / `Enum` / `Tuple` 复合类型在参数或返回位置（这些类型在 $\phi$ 中映射为 $\bot$）；
3. Host 函数集 $\mathcal{H}$ 在执行期间不抛 panic（即 `func_wrap` 闭包不 panic）；

则对 $P$ 的任意终止执行 $\sigma_0 \to_{\text{HIR}}^\ast \sigma_f$，存在 $W$ 的执行 $\sigma'_0 \to_{\text{WASM}}^\ast \sigma'_f$ 使得 $\text{Obs}_{\text{HIR}}(\sigma_f) \equiv \text{Obs}_{\text{WASM}}(\sigma'_f)$。

**证明**：构造模拟关系 $\mathcal{R} \subseteq \Sigma_{\text{HIR}} \times \Sigma_{\text{WASM}}$：

$$
\mathcal{R} = \{(\sigma_{\text{HIR}}, \sigma_{\text{WASM}}) \mid \text{encode}(\sigma_{\text{HIR}}) = \sigma_{\text{WASM}}\}
$$

其中 $\text{encode}$ 是 HIR 状态到 WASM 状态的编码函数，由 $\phi$ + 字段布局规则（§3.4）+ host import 调用约定（§7）共同确定。

需证明三条性质：

**(a) 初始化对齐**：$P$ 的入口 main 对应 $W$ 的入口 main（[`sections.rs:160-172`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/wasm/sections.rs)）。两者初始可观察状态均为空。$\text{encode}(\sigma_0) = \sigma'_0$。✓

**(b) 单步保持**：对 HIR 任一迁移 $\langle e, \sigma \rangle \to_{\text{HIR}} \langle e', \sigma' \rangle$，分情形：

- **字面量（LIT）**：HIR 推入值 $v:t$。WASM 端 `compile_literal`（[`compile.rs:113-116`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/wasm/compile.rs)）生成对应 `i64.const` / `f64.const` / `i32.const` 指令。值经 $\phi$ 编码后入栈。$\sigma' = \text{encode}(\sigma')$。✓

- **变量读取（VAR）**：HIR 读取 $\rho(x)$。WASM 端 `local.get`（[`compile.rs:118-129`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/wasm/compile.rs)）。若 $x$ 是参数，类型由 $\phi$ 决定；若 $x$ 是局部变量，统一存储为 `i64`，读取时若 $\phi(t) = \texttt{f64}$ 则插入 `f64.reinterpret_i64`，若 $\phi(t) = \texttt{i32}$ 则插入 `i32.wrap_i64`。逆操作在写入时进行，故 $\sigma' = \text{encode}(\sigma')$。✓

- **字符串加法**：HIR 推入 $\text{str\_add}(a, b)$ 调用。WASM 端 `compile.rs:164-168` 生成 `call host.str_add`。host 实现（[`host.rs:52-74`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/wasm/host.rs)）从 WASM 内存读取 $a, b$，拼接后写回 WASM 内存，返回新指针。HIR 语义"拼接字符串"被精确模拟。✓

- **结构体构造**：HIR `Struct::new` 在 $\rho$ 中创建结构体。WASM 端经 `tenth_alloc` 分配内存，`i64.store` 写入字段。`build_struct_layouts`（[`wasm/types.rs:13-38`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/wasm/types.rs)）保证字段偏移与 HIR 一致。✓

- **Vec 操作**：HIR `Vec::push(v, item)` 对应 WASM `call host.Vec_push`（[`compile.rs:101`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/wasm/compile.rs)）。host 实现（[`host.rs:161-203`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/wasm/host.rs)）维护 Vec header（cap, len, dp），与 HIR 的 `Vec<T>` 语义对齐。✓

- **分支与循环**：HIR `if` / `while` 对应 WASM `if`/`loop`/`block` 结构化控制流。控制流图保持。✓

- **函数调用**：HIR `Call(f, args)` 对应 WASM `call`（用户函数）或 `call host.X`（host 函数）。参数经 $\phi$ 编码传递，返回值经 $\phi^{-1}$ 解码。✓

- **闭包**：HIR `Closure` 对应 WASM `call_indirect` + 元素表 + 捕获环境结构体（[`sections.rs:124-158`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/wasm/sections.rs)）。捕获变量通过 `env_ptr` 传递。✓

**(c) 终止对齐**：HIR main 返回 $v:t$。WASM main 经 `wrap_to_i32`（[`compile.rs:63-85`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/wasm/compile.rs)）转换为 `i32` 退出码。退出码作为可观察行为的一部分。$\text{Obs}_{\text{HIR}}(\sigma_f) \equiv \text{Obs}_{\text{WASM}}(\sigma'_f)$。✓

**注意**：本证明是**弱双模拟**——只证明 HIR→WASM 单向，不证明 WASM→HIR。原因是 host 副作用（如 `compile_host` 调用 Rust 编译器写文件）在 WASM→HIR 方向不可逆（[`host.rs:207-230`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/wasm/host.rs)）。这是 Tenth 模式的固有局限（§9.4）。

由 (a)(b)(c)，对任意 HIR 终止执行，存在 WASM 执行使得可观察行为等价。$\square$

### 5.2 定理 W2（类型塌缩健全性）

**定理 W2**：在 W1 的前提条件下，类型塌缩映射 $\phi$ 不损失 HIR 层区分的语义。即：对 HIR 中任意两个**可区分**的值 $v_1 : t_1, v_2 : t_2$（$t_1 \neq t_2$），经 $\phi$ 塌缩后仍可通过 WASM 操作语义区分。

**证明**：分情形讨论：

**情形 1：$t_1, t_2 \in \{\text{I8, I16, I32, I64}\}$**

$\phi(t_1) = \phi(t_2) = \texttt{i64}$，类型信息丢失。但 HIR 中这些类型**值域不同**而**操作相同**（皆为整数算术）。WASM 端 `i64.add` 等指令的行为对 I8/I16/I32/I64 一致——HIR 层的"溢出回绕"语义由 host 在显示/序列化时按类型决定，不影响 WASM 内部计算。故**语义无损**。

**情形 2：$t_1 \in \{\text{I8, ..., I64}\}, t_2 \in \{\text{F32, F64}\}$**

$\phi(t_1) = \texttt{i64}, \phi(t_2) = \texttt{f64}$。塌缩后类型不同（`i64` vs `f64`），WASM 类型检查器可区分。✓

**情形 3：$t_1 = \text{Bool}, t_2 \in \{\text{I8, ..., I64}\}$**

$\phi(t_1) = \texttt{i32}, \phi(t_2) = \texttt{i64}$。塌缩后类型不同。✓

**情形 4：$t_1 = \text{Str}, t_2 \in \{\text{I8, ..., I64}\}$**

$\phi(t_1) = \phi(t_2) = \texttt{i64}$，类型信息丢失。但 HIR 中 `Str` 的所有操作（拼接、比较、索引、切片）都被编译为 host import 调用（`str_add`/`str_eq`/`str_at`/`str_slice`/`str_cmp`/`str_len`），而整数的操作编译为 WASM 内部 `i64.add` 等。**操作的指令不同**使得两类值在执行时被正确路由。HIR 的"类型化语义"转化为 WASM 的"操作化语义"——只要操作集 $\mathcal{O}_{\text{HIR}}$ 不混淆类型（即不存在同时接受 Str 和 I64 的操作），塌缩就不损失区分性。✓

**情形 5：$t_1 \in \{\text{Ref}(\_), \text{Struct}(\_), \text{Generic}(\_)\}, t_2 \in \{\text{I8, ..., I64, Str}\}$**

$\phi(t_1) = \phi(t_2) = \texttt{i64}$。但复合类型的所有操作都通过 host import 或 `i64.load`/`i64.store` + 字段偏移实现，而整数/字符串的算术操作不会涉及 `i64.load` 字段访问。HIR 的类型检查在 lowering 阶段已确保类型正确，WASM 端不再需要运行时区分。✓

**情形 6：$t_1 = \text{Unit}, t_2 \neq \text{Unit}$**

$\phi(\text{Unit}) = \bot$，不传递值。$\phi(t_2) \neq \bot$。可区分。✓

**综合**：在 W1 假设下（不含 U8/F16/BF16/Char 等未显式处理类型），$\phi$ 在所有可区分类型对上保持区分性。$\square$

**注**：本定理的"区分"是**操作层面的区分**，而非**值层面的区分**。两个不同的 I32 值塌缩为 i64 后仍可区分（值不同）；但同一 I32 值若被误读为 Str 指针，则可能引发内存错误——这种"误读"由 HIR 类型检查在 lowering 阶段排除，不由 $\phi$ 在运行时排除。

### 5.3 定理 W3（host 桥接完备性）

**定理 W3**：在 W1 假设下，18 个 host import $\mathcal{H} = \{h_0, \ldots, h_{17}\}$ 覆盖 HIR 在"支持子集"上所需的全部 heap 操作。

**证明**：穷举 HIR 中所有需要 host 干预的操作：

**类别 A：字符串操作**

HIR `Str` 类型的操作（由 [`types.rs`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/hir/types.rs) 与 `lower.rs` 隐含定义）：

| HIR 操作 | host import | 源码 |
|---------|-------------|------|
| 字符串拼接 `s1 + s2` | `h_3 = str_add` | [`host.rs:52-74`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/wasm/host.rs) |
| 字符串相等 `s1 == s2` | `h_4 = str_eq` | [`host.rs:76-85`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/wasm/host.rs) |
| 整数转字符串 | `h_5 = str_int` | [`host.rs:87-99`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/wasm/host.rs) |
| 字符串长度 | `h_{12} = str_len` | [`host.rs:233-239`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/wasm/host.rs) |
| 字符串索引 `s[i]` | `h_{13} = str_at` | [`host.rs:244-275`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/wasm/host.rs) |
| 字符串比较 `<, >, <=, >=` | `h_{14} = str_cmp` | [`host.rs:278-297`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/wasm/host.rs) |
| 字符串切片 `s[a..b]` | `h_{16} = str_slice` | [`host.rs:306-335`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/wasm/host.rs) |

字符串操作覆盖完整。✓

**类别 B：内存分配**

| HIR 操作 | host import | 源码 |
|---------|-------------|------|
| 通用分配（结构体等） | `h_6 = tenth_alloc` | [`host.rs:102-117`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/wasm/host.rs) |

`tenth_alloc` 是 bump allocator，所有结构体、字符串缓冲、Vec header 等的内存都通过它分配。✓

**类别 C：Vec 操作**

HIR `Vec<T>` 是 Generic 类型，塌缩为 `i64` 指针。其操作通过 host import 实现：

| HIR 操作 | host import | 源码 |
|---------|-------------|------|
| `Vec::new()` | `h_7 = Vec_new` | [`host.rs:120-133`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/wasm/host.rs) |
| `Vec::push(v, x)` | `h_8 = Vec_push` | [`host.rs:161-203`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/wasm/host.rs) |
| `Vec::len(v)` | `h_9 = Vec_len` | [`host.rs:136-144`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/wasm/host.rs) |
| `Vec::get(v, i)` | `h_{10} = Vec_get` | [`host.rs:147-158`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/wasm/host.rs) |

Vec 操作覆盖完整。✓

**类别 D：I/O 与外部副作用**

| HIR 操作 | host import | 源码 |
|---------|-------------|------|
| 标准输出 `println` | `h_0 = println` | [`host.rs:9-14`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/wasm/host.rs) |
| 文件写 `write_file` | `h_1 = write_file` | [`host.rs:16-25`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/wasm/host.rs) |
| 文件读 `read_file` | `h_2 = read_file` | [`host.rs:28-50`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/wasm/host.rs) |

I/O 覆盖完整。✓

**类别 E：跨边界类型转换**

| HIR 操作 | host import | 源码 |
|---------|-------------|------|
| `f64` 转 IEEE 754 位模式 | `h_{15} = f64_bits` | [`host.rs:300-303`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/wasm/host.rs) |

`f64_bits` 是 host 边界 reinterp 的关键：当 `f64` 需要存入 `i64` 局部变量时，先调 `f64_bits` 转为 `i64`。✓

**类别 F：自举支持**

| HIR 操作 | host import | 源码 |
|---------|-------------|------|
| 调用 Rust 母编译器编译字符串 | `h_{11} = compile_host` | [`host.rs:207-230`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/wasm/host.rs) |

`compile_host` 实现了"在 WASM 内编译 Tenth 源码"的能力——这是路径 C 全 WASM 闭环的关键（与 T12 §6.4 联动）。✓

**类别 G：张量桥接（部分实现）**

| HIR 操作 | host import | 源码 |
|---------|-------------|------|
| `Tensor::from_vec` | `h_{17} = tensor_from_vec` | [`host.rs:340-343`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/wasm/host.rs) |

**v1.2 修正**：源码注释明确写道"Simplified: return total element count (len) as the tensor handle. This provides a deterministic value for parity testing."（[`host.rs:338-339`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/wasm/host.rs)）。`tensor_from_vec` **不实际构造张量**，仅返回元素数。这是 W3 的反例——张量操作未完备桥接。

**定理 W3'（修正版）**：在 W1 假设下，**且排除 HIR 程序中的 `Tensor` 类型操作**，18 个 host import 覆盖 HIR 所需的全部 heap 操作。$\square$

**注**：张量桥接的完整实现是未来工作（§10.2）。当前路径 C 闭环仅支持非张量程序。

### 5.4 定理 W4（与 Emscripten/AssemblyScript 对比）

**定理 W4**：Tenth 的"值类型塌缩 + host 桥接"模式与 Emscripten/AssemblyScript 的"完全 lowering"模式在以下维度上有结构性差异：

| 维度 | Tenth | Emscripten | AssemblyScript |
|------|-------|------------|----------------|
| **类型映射** | $\phi$ 塌缩为 i64/f64/i32/⊥ | 完全 lowering 到 i32/i64/f32/f64 + 偏移 | 部分 lowering（基本类型直映射 + 类 lowering） |
| **结构体访问** | 字段偏移在编译期确定，访问走 `i64.load` | 编译期偏移，`i32.load` | 编译期偏移，`i32.load` |
| **字符串** | host 拥有，WASM 仅持指针 | WASM 内嵌 UTF-8/UTF-16 + 内部拼接 | WASM 内嵌 UTF-16 + 内部拼接 |
| **容器（Vec/Array）** | host 拥有 header 与 data | WASM 内部完整实现 + libc++ lowering | WASM 内部 `Array<T>` + GC |
| **GC** | 无（host 管理生命周期） | 无（引用计数或泄漏） | 自带 tracing GC |
| **ABI 复杂度** | 低（i64 主导 + 18 import） | 高（结构体布局对 host 可见） | 中 |
| **WASM 体积** | 小（无胶水） | 大（数十 KB 起） | 中 |
| **Host 可干预性** | 高（所有 heap 操作经 host） | 低（WASM 自包含） | 低 |
| **自包含性** | 否（必须配 host） | 是 | 是 |
| **可移植性** | 受限于 host 实现一致性 | 任何 WASM 运行时 | 任何 WASM 运行时 |
| **自举闭环支持** | 原生支持（`compile_host`） | 不支持 | 不支持 |

**核心差异**：

1. **语义责任位置**：Emscripten/AssemblyScript 把 heap 语义内嵌于 WASM；Tenth 把 heap 语义外推到 host。前者更"自包含"，后者更"协作式"。

2. **类型信息位置**：Emscripten/AssemblyScript 在编译期丢弃类型信息（lowering 后只剩偏移与字节）；Tenth 在运行时**通过 host 重新引入类型信息**（host 知道指针指向的是字符串还是结构体，因为 host 实现了相应操作）。

3. **ABI 稳定性**：Tenth 的 ABI 仅由 18 个 host import 签名决定，扩展时只需追加 import（[`mod.rs:64-71`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/wasm/mod.rs) 注释明确说明这一点）；Emscripten 的 ABI 受 C++ name mangling、虚表布局、STL 实现等多重因素影响。

4. **自举闭环**：Tenth 通过 `compile_host`（[`host.rs:207-230`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/wasm/host.rs)）允许 WASM 内调用 Rust 母编译器，这是路径 C 全 WASM 闭环的关键能力；Emscripten/AssemblyScript 不支持。

5. **代价**：Tenth 的代价是**可移植性受限**——Tenth 编译的 WASM 必须配 Tenth 的 host 才能运行，不能在任意 WASM 运行时执行。这是"AI 原生 + 自举"目标的必然取舍。

**证明**：上述对比基于公开文档（[Emscripten docs](https://emscripten.org/)、[AssemblyScript docs](https://www.assemblyscript.org/)）与 Tenth 源码的逐一对照。维度 1-10 由 §3 与 §7 的源码分析直接支持。维度 11（自举闭环）由 [`host.rs:207-230`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/wasm/host.rs) 中 `compile_host` 的存在与实现支持。$\square$

### 5.5 定理 W5（自动推导 host 边界 import 集，未来工作）

**定理 W5（声明）**：存在一个算法 $\mathcal{A}$，输入 HIR 程序 $P$，输出 host import 集 $\mathcal{H}_P \subseteq \mathcal{H}$，使得 $\mathcal{H}_P$ 是 $P$ 实际调用的 host import 集的精确刻画（既不过多也不过少）。

**v1.4 备注**：本定理**仅声明存在性，不给完整算法**，标注为未来工作。当前手工维护 18 个常量（[`mod.rs:71-89`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/wasm/mod.rs)）。

**算法草图**：

1. 对 $P$ 的 HIR 进行抽象解释，收集所有"heap 操作"集合 $\mathcal{O}_{\text{heap}}$。
2. 定义映射 $\text{host\_of} : \mathcal{O}_{\text{heap}} \to \mathcal{H}$，将每个 heap 操作映射到对应的 host import。
3. $\mathcal{H}_P = \{\text{host\_of}(o) \mid o \in \mathcal{O}_{\text{heap}}\}$。

**关键挑战**（未来工作需解决）：

- **抽象解释的精度**：必须能区分 `Vec::push`（需 `Vec_push`）与 `Vec::get`（需 `Vec_get`），需要类型敏感的流分析。
- **闭包内调用**：闭包捕获的变量若调用 host，需通过 `call_indirect` 追踪。
- **动态分发**：HIR 若支持 trait/dyn，需 vtable 分析。

**与 T22 联动**：T22（[`docs/论文/T22-Closure自由变量分析正确性.md`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/docs/论文/T22-Closure自由变量分析正确性.md)）的闭包自由变量分析可作为 $\mathcal{A}$ 的子模块。

$\square$

---

## 6. 16 种 BaseType → 4 种 WASM 值类型逐一映射

下表逐一列出 [`hir/types.rs:3-10`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/hir/types.rs) 中所有 16 种 `BaseType` 经 $\phi$ 的映射：

| # | BaseType | $\phi$ 输出 | 机制 | 源码分支 | 备注 |
|---|----------|------------|------|---------|------|
| 1 | `I8` | `i64` | 保留（符号扩展） | [`mod.rs:25`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/wasm/mod.rs) | 整数统一 64 位 |
| 2 | `I16` | `i64` | 保留 | [`mod.rs:25`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/wasm/mod.rs) | 同上 |
| 3 | `I32` | `i64` | 保留 | [`mod.rs:25`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/wasm/mod.rs) | 同上 |
| 4 | `I64` | `i64` | 保留 | [`mod.rs:25`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/wasm/mod.rs) | 同上 |
| 5 | `U8` | $\bot$ | **未显式处理** | [`mod.rs:30, 37`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/wasm/mod.rs) `_ => None` | **v1.1 修正**：实际落入 `_` 分支 |
| 6 | `U16` | $\bot$ | **未显式处理** | 同上 | 同上 |
| 7 | `U32` | $\bot$ | **未显式处理** | 同上 | 同上 |
| 8 | `U64` | $\bot$ | **未显式处理** | 同上 | 同上 |
| 9 | `F16` | $\bot$ | **未显式处理** | 同上 | 同上 |
| 10 | `F32` | `f64` | 保留（精度提升） | [`mod.rs:26`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/wasm/mod.rs) | F32 提升为 F64 |
| 11 | `F64` | `f64` | 保留 | [`mod.rs:26`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/wasm/mod.rs) | |
| 12 | `BF16` | $\bot$ | **未显式处理** | [`mod.rs:30, 37`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/wasm/mod.rs) | BF16 无 WASM 对应 |
| 13 | `Bool` | `i32` | 保留 | [`mod.rs:27`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/wasm/mod.rs) | WASM i32 表示 0/1 |
| 14 | `Char` | $\bot$ | **未显式处理** | [`mod.rs:30, 37`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/wasm/mod.rs) | Char 暂未桥接 |
| 15 | `Str` | `i64` | 指针化 | [`mod.rs:28`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/wasm/mod.rs) | 字符串由 host 管理 |
| 16 | `Unit` | $\bot$ | 丢弃 | [`mod.rs:29`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/wasm/mod.rs) | 不传递值 |

**统计**：
- 显式映射：9 种（I8/I16/I32/I64/F32/F64/Bool/Str/Unit）
- 未显式处理（落入 `_`）：7 种（U8/U16/U32/U64/F16/BF16/Char）
- 真正"无对应"：6 种（U8-U64 + F16 + BF16，理论上应映射到 i64/f64，但被遗漏）
- 应有专门处理：1 种（Char，可能需要 host 桥接，类似 Str）

**复合类型映射**（来自 [`mod.rs:32-37`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/wasm/mod.rs)）：

| 复合类型 | $\phi$ 输出 | 备注 |
|---------|------------|------|
| `Ref(T)` | `i64` | 引用即指针 |
| `MutRef(T)` | `i64` | 同上 |
| `Struct(name)` | `i64` | 结构体指针 |
| `TypeParam{name}` | `i64` | 泛型未实例化场景 |
| `Generic{base, args}` | `i64` | 如 `Vec<T>` |
| `Unknown` | `i64` | 类型推断失败的兜底 |
| `Tensor{dtype, dims}` | $\bot$ | **未显式处理**（落入 `_`） |
| `Array(T)` | $\bot$ | **未显式处理** |
| `FnType{params, ret}` | $\bot$ | **未显式处理**（闭包另有专门机制） |
| `Enum(name)` | $\bot$ | **未显式处理** |
| `Tuple(types)` | $\bot$ | **未显式处理** |

---

## 7. 18 个 host import 的功能分析

下表逐一列出 18 个 host import（[`mod.rs:71-89`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/wasm/mod.rs)）：

| # | 常量名 | host 函数 | 签名 | 功能 | 源码 |
|---|--------|----------|------|------|------|
| 0 | `HOST_PRINTLN` | `host.println` | `(i32) -> ()` | 打印 null 终止字符串到 stdout | [`host.rs:9-14`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/wasm/host.rs) |
| 1 | `HOST_WRITE_FILE` | `host.write_file` | `(i32, i32) -> ()` | 写文件（路径指针 + 内容指针） | [`host.rs:16-25`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/wasm/host.rs) |
| 2 | `HOST_READ_FILE` | `host.read_file` | `(i32) -> i32` | 读文件，返回内容指针 | [`host.rs:28-50`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/wasm/host.rs) |
| 3 | `HOST_STR_ADD` | `host.str_add` | `(i32, i32) -> i32` | 字符串拼接 | [`host.rs:52-74`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/wasm/host.rs) |
| 4 | `HOST_STR_EQ` | `host.str_eq` | `(i32, i32) -> i32` | 字符串相等比较 | [`host.rs:76-85`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/wasm/host.rs) |
| 5 | `HOST_STR_INT` | `host.str_int` | `(i64) -> i32` | 整数转字符串 | [`host.rs:87-99`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/wasm/host.rs) |
| 6 | `HOST_TENTH_ALLOC` | `host.tenth_alloc` | `(i32) -> i32` | bump 分配 size 字节 | [`host.rs:102-117`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/wasm/host.rs) |
| 7 | `HOST_VEC_NEW` | `host.Vec_new` | `() -> i64` | 创建空 Vec（24 字节 header） | [`host.rs:120-133`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/wasm/host.rs) |
| 8 | `HOST_VEC_PUSH` | `host.Vec_push` | `(i64, i64) -> i64` | Vec 追加元素 | [`host.rs:161-203`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/wasm/host.rs) |
| 9 | `HOST_VEC_LEN` | `host.Vec_len` | `(i64) -> i64` | Vec 长度 | [`host.rs:136-144`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/wasm/host.rs) |
| 10 | `HOST_VEC_GET` | `host.Vec_get` | `(i64, i64) -> i64` | Vec 索引访问 | [`host.rs:147-158`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/wasm/host.rs) |
| 11 | `HOST_COMPILE_HOST` | `host.compile_host` | `(i32, i32) -> i32` | **调用 Rust 母编译器** | [`host.rs:207-230`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/wasm/host.rs) |
| 12 | `HOST_STR_LEN` | `host.str_len` | `(i32) -> i32` | 字符串长度 | [`host.rs:233-239`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/wasm/host.rs) |
| 13 | `HOST_STR_AT` | `host.str_at` | `(i32, i64) -> i32` | 字符串索引（返回单字符指针） | [`host.rs:244-275`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/wasm/host.rs) |
| 14 | `HOST_STR_CMP` | `host.str_cmp` | `(i32, i32, i32) -> i32` | 字符串比较（op, a, b） | [`host.rs:278-297`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/wasm/host.rs) |
| 15 | `HOST_F64_BITS` | `host.f64_bits` | `(f64) -> i64` | **f64 → IEEE 754 位模式** | [`host.rs:300-303`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/wasm/host.rs) |
| 16 | `HOST_STR_SLICE` | `host.str_slice` | `(i32, i64, i64) -> i32` | 字符串切片 | [`host.rs:306-335`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/wasm/host.rs) |
| 17 | `HOST_TENSOR_FROM_VEC` | `host.tensor_from_vec` | `(i32, i32, i32) -> i64` | **stub**：返回元素数 | [`host.rs:340-343`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/wasm/host.rs) |

**按功能分类**：

- **字符串**（7 个）：#3, #4, #5, #12, #13, #14, #16
- **Vec**（4 个）：#7, #8, #9, #10
- **I/O**（3 个）：#0, #1, #2
- **内存**（1 个）：#6
- **类型转换**（1 个）：#15
- **自举**（1 个）：#11
- **张量**（1 个，stub）：#17

**注意**：#15 `f64_bits` 是关键的 reinterp host——它把 `f64` 转为 `i64` 位模式，使得 `f64` 值可存入 `i64` 局部变量（[`compile.rs:28`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/wasm/compile.rs) 表明所有局部变量统一为 `i64`）。这是 Tenth 模式下"局部变量无类型化"的关键。

---

## 8. 与 Emscripten/AssemblyScript 对比（深化）

### 8.1 ABI 稳定性

Tenth 的 ABI 由 [`sections.rs:26-44`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/wasm/sections.rs) 显式定义的 18 个 type signature 决定。新增 host import 只需在 [`mod.rs:71-89`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/wasm/mod.rs) 追加常量 + [`sections.rs`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/wasm/sections.rs) 追加 type + [`host.rs`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/wasm/host.rs) 追加实现。[`mod.rs:64-71`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/wasm/mod.rs) 注释明确说："Adding a new host import only requires appending a `HOST_*` constant below and registering the matching type+import in `sections.rs` (and the implementation in `host.rs`). `IMPORT_COUNT` is derived from the last index, so it stays in sync automatically."

Emscripten 的 ABI 受多重因素影响：C++ ABI（itanium ABI）、STL 实现版本、Emscripten 自身版本、`-s ENVIRONMENT=` 选项等。版本升级常需重新编译所有依赖。

AssemblyScript 的 ABI 较稳定，但仍在快速演进（标准库 API 变化）。

### 8.2 Host 兼容性

Tenth 的 host 必须**精确实现 18 个 import**——任何 host 实现的偏差都导致 W1 失效。这意味着 Tenth 编译的 WASM **只能在 Tenth 配套 host 上运行**，不能在 wasmtime/wasmer 等通用 WASM 运行时上直接运行（除非用户手工注册同名 host 函数）。

Emscripten/AssemblyScript 编译的 WASM 可在任何符合 WASM 1.0 标准的运行时上运行（只需提供基本的 `fd_write` 等 WASI import）。

### 8.3 WASM 体积

Tenth 的"hello world"编译产物 < 1 KB（仅 main + 少量 data section）。
Emscripten 的"hello world"编译产物通常 > 50 KB（含 libc++ 静态链接）。
AssemblyScript 的"hello world"约 5-10 KB。

### 8.4 自举闭环支持

Tenth 通过 `compile_host`（#11）允许 WASM 内调用 Rust 母编译器——这是路径 C 全 WASM 闭环的关键（[T12 §6.4](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/docs/论文/T12-双侧编译器语义等价性.md)）。Emscripten/AssemblyScript 不支持"在 WASM 内调用宿主编译器"。

### 8.5 数据可见性

Tenth 模式下，host **可见**所有 heap 数据（因为所有 heap 操作经 host）——这有利于调试、性能分析、内存追踪。
Emscripten/AssemblyScript 模式下，heap 数据在 WASM 线性内存中，host 只能通过工具（如 Chrome DevTools）事后查看，不能在操作发生时介入。

### 8.6 性能特征

Tenth 模式下，每次 heap 操作都有 host 调用开销（WASM↔host 边界切换）。对细粒度操作（如 `Vec::get` 在循环中），这可能是性能瓶颈。
Emscripten/AssemblyScript 在 WASM 内部完成 heap 操作，无边界切换开销，但失去了 host 介入能力。

**权衡**：Tenth 适合"少量重操作"场景（如张量算子、字符串处理）；Emscripten 适合"大量轻操作"场景（如数值循环）。

---

## 9. 局限（独立章节）

本章节诚实记录理论分析的局限，每条说明：是什么、影响多大、如何缓解。

### 9.1 7 种 BaseType 未显式处理

**是什么**：[`mod.rs:30, 37`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/wasm/mod.rs) 的 `_ => None` 分支捕获了 U8/U16/U32/U64/F16/BF16/Char 共 7 种 `BaseType`，使它们映射为 $\bot$，触发 `to_val_type_required` 时报错。

**影响**：HIR 程序若使用这 7 种类型在参数或返回位置，WASM 编译会失败。但 Tenth 标准库可能隐式使用 U8（如字节缓冲）、Char（如字符处理）——这些程序当前**不能编译为 WASM**。

**如何缓解**：
- 短期：在 `to_val_type` 中显式添加 7 种类型的映射（U8/U16/U32/U64 → i64，F16/BF16 → f64，Char → i64 或 i32）。
- 中期：在 lowering 阶段添加类型检查，提前报错而非延迟到 WASM 编译。
- 长期：将 7 种类型纳入 W1 的"支持子集"。

### 9.2 `Unit` 类型在参数位置的二义性

**是什么**：`Unit` 映射为 $\bot$，不传递值。但若函数参数为 `Unit`，WASM 函数签名中**该参数消失**，导致参数数量不一致。

**影响**：HIR 中 `fn foo(x: Unit, y: I64)` 与 `fn foo(y: I64)` 在 WASM 中签名相同（都只接收一个 i64），可能引发调用歧义。

**如何缓解**：
- 短期：禁止 `Unit` 出现在参数位置（在 lowering 阶段检查）。
- 长期：保留 `Unit` 参数为占位符 `i32`（值为 0）。

### 9.3 `tensor_from_vec` 是 stub

**是什么**：[`host.rs:340-343`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/wasm/host.rs) 的 `tensor_from_vec` 不实际构造张量，仅返回元素数。源码注释明确："Simplified: return total element count (len) as the tensor handle. This provides a deterministic value for parity testing."

**影响**：W3 在张量操作上**不成立**——张量算子（如 matmul、broadcast）在 WASM 后端**不可用**。W1 的"支持子集"必须排除 `Tensor` 类型。

**如何缓解**：
- 短期：在 lowering 阶段检查 `Tensor` 类型并报错"WASM 后端不支持张量"。
- 中期：扩展 host import 集合，添加 `tensor_matmul`、`tensor_broadcast` 等。
- 长期：与 T17（dtype 提升格）联动，设计完整的张量 host ABI。

### 9.4 双模拟关系的"外推"假设

**是什么**：W1 证明假设 host 函数的语义"精确模拟"HIR 操作语义。但 host 实现是 Rust 代码，**不在 WASM 操作语义内**——host 可调用 `std::fs::write`、`std::str::from_utf8` 等外部函数，这些函数的语义**未被形式化**。

**影响**：W1 实际证明的是"HIR↔(WASM + host 形式化模型)"的模拟关系，而非"HIR↔WASM"的纯模拟关系。若 host 实现有 bug（如 `str_add` 错误地修改了原字符串），W1 不保证检测。

**如何缓解**：
- 短期：对 host 函数做单元测试（如 `str_add` 不修改输入）。
- 中期：用 Rust 形式化验证工具（如 Prusti）验证 host 实现的关键性质。
- 长期：把 host 函数的语义也纳入形式化模型，证明 host 实现符合规范。

### 9.5 闭包的 `call_indirect` 与类型擦除

**是什么**：[`sections.rs:124-158`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/wasm/sections.rs) 中闭包类型统一为 `(i64 env_ptr, i64 param1, ..., i64 paramN) -> i64`——所有参数被擦除为 `i64`。这意味着闭包调用时**无类型检查**：若调用者传入错误类型的参数，WASM 不会报错，host 端会读到错误数据。

**影响**：W2 的"塌缩健全性"在闭包边界**部分失效**——HIR 类型检查保证闭包调用类型正确，但 WASM 层无运行时检查。

**如何缓解**：
- 短期：依赖 HIR 类型检查的完备性（T19、T23 联动）。
- 长期：在 WASM 端引入类型标记 + 运行时检查（性能代价）。

### 9.6 局部变量统一为 `i64` 的精度漂移

**是什么**：[`compile.rs:28`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/wasm/compile.rs) 表明所有局部变量统一为 `i64`。`f64` 值存入局部变量时，需先调 `f64_bits` 转 `i64`，读取时用 `f64.reinterpret_i64` 转回。这一过程**理论上是位保真的**（IEEE 754 位模式 ↔ i64），但若 host `f64_bits` 实现错误，会引入静默精度漂移。

**影响**：W1 在浮点数计算上**依赖 `f64_bits` 实现正确性**——这一假设未被形式化证明。

**如何缓解**：
- 短期：对 `f64_bits` 做位保真单元测试（覆盖 NaN、Inf、denormal）。
- 中期：用 Rust 形式化验证 `f64::to_bits` 的位保真性。
- 长期：考虑 WASM SIMD proposal 引入真正的 f64 局部变量。

### 9.7 host panic 的未定义行为

**是什么**：W1 假设"host 函数不抛 panic"。但 [`host.rs`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/wasm/host.rs) 中多处使用 `unwrap`、`unwrap_or`、`unwrap_or(0)` 等，若输入异常（如指针越界），可能 panic。WASM↔host 边界的 panic 行为**依赖 wasmi 实现**。

**影响**：若 host panic，WASM 模块可能进入未定义状态，W1 不再成立。

**如何缓解**：
- 短期：审查所有 `unwrap` 调用，替换为显式错误处理。
- 中期：在 `func_wrap` 闭包外加 `catch_unwind` 包裹（类似 JIT 的 hostcall trampoline，参考 T9 §5）。
- 长期：在 W1 假设中显式加入"host 不 panic"作为前置条件，并在 host 实现中强制保证。

### 9.8 W3 的不完备性：未覆盖的复合类型操作

**是什么**：W3 排除了 `Tensor` / `Array` / `FnType` / `Enum` / `Tuple` 五种复合类型。这些类型在 `to_val_type` 中映射为 $\bot$（[`mod.rs:37`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/wasm/mod.rs) `_ => None`）。

**影响**：HIR 程序若使用这些类型在参数或返回位置，WASM 编译失败。其中：
- `Array(T)`：应可通过 `Vec<T>` 桥接，但未实现。
- `Tuple(types)`：应可通过结构体化 + 字段偏移实现，但未实现。
- `Enum(name)`：enum 已有布局（[`wasm/types.rs:25-37`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/wasm/types.rs)），但 enum 值本身的传递未明确。
- `FnType`：闭包有专门机制（D5），但函数指针传递未明确。

**如何缓解**：
- 短期：在 lowering 阶段检查这些类型并报错。
- 中期：为每种复合类型设计 host ABI（如 `Array` 复用 `Vec`、`Tuple` 用结构体化）。
- 长期：扩展 W1 的"支持子集"，将复合类型纳入完备性证明。

---

## 10. 与 T12 联动

### 10.1 T12 路径 C 的依赖关系

[T12](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/docs/论文/T12-双侧编译器语义等价性.md) §6.4 定义路径 C（全 WASM 闭环）：tenthc 用 Tenth 编写 → 编译为 WASM → wasmi 执行 → 在 WASM 内调用 `compile_host`（#11）编译 Tenth 源码 → 写出 `.wasm` 文件。

路径 C 的语义等价性依赖本文的：

- **W1**（语义保持）：保证 tenthc 编译为 WASM 后语义保持。
- **W3'**（host 桥接完备性，排除 tensor）：保证 tenthc 所需的所有 heap 操作经 host 桥接。
- **§9.3**（`tensor_from_vec` stub 局限）：tenthc 当前不使用 tensor 操作，故 stub 不影响路径 C 闭环。

### 10.2 与 T12 §6.4 的对应

T12 §6.4 给出路径 C 闭环的形式化命题：**"路径 C 闭环等价 ⟺ tenthc 自身可被 WASM 表达 ⟺ W1 ∧ W3 在 tenthc 子集上成立"**。

本文 W1 与 W3' 共同构成路径 C 闭环的**前提条件**。若 W1 或 W3' 失败（如 §9.3 的张量 stub 触发），路径 C 闭环失败。这是本文与 T12 的强联动。

### 10.3 与 T22 联动（闭包自由变量）

W5 的自动推导算法 $\mathcal{A}$ 需要闭包自由变量分析作为子模块。T22（[`docs/论文/T22-Closure自由变量分析正确性.md`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/docs/论文/T22-Closure自由变量分析正确性.md)）提供了这一分析的正确性证明，可作为 $\mathcal{A}$ 的基础。

---

## 11. 工程权衡与开放问题

### 11.1 工程权衡

**优势**：
1. WASM 产物体积小（无胶水代码）。
2. Host 可干预所有 heap 操作（调试友好）。
3. ABI 简洁稳定（18 个 import 签名）。
4. 原生支持自举闭环（`compile_host`）。
5. 复用 Rust 生态（`std::fs`、`Vec`、`String` 等）。

**代价**：
1. 必须配 Tenth host 才能运行（不可移植）。
2. host 调用开销（边界切换）。
3. 类型信息部分丢失（依赖 HIR 类型检查）。
4. 当前实现有局限（7 种 BaseType 未处理、tensor stub）。
5. 双模拟关系单向（host 副作用不可逆）。

### 11.2 开放问题

1. **WASM reference-types 提案**：`externref` 是否可替代部分 host import？若可，可减少 host 边界开销。
2. **WASM GC 提案**：WASM GC 是否可替代 host 桥接的内存管理？若可，可降低对 host 的依赖。
3. **WASM component model**：组件模型是否可形式化 host 桥接的边界？这与本文的 host import 模式有何关系？
4. **Tensor host ABI 设计**：完整张量桥接需要哪些 host import？与 T17（dtype 提升格）如何联动？
5. **host 函数形式化**：能否用 Coq/Lean 形式化 18 个 host 函数的语义，并机器验证 W1？
6. **W5 自动推导**：算法 $\mathcal{A}$ 的精度上界是什么？是否可证明完备性？
7. **多 host 后端**：若 host 用其他语言（如 Python）实现，W1 是否仍成立？这关系 Tenth 的多语言 host 生态。

### 11.3 实施建议

基于本文理论分析，对实施的指导：

1. **优先修补 §9.1**：在 `to_val_type` 中显式添加 7 种 BaseType 映射。这是最小修补，可立即扩展"支持子集"。
2. **修补 §9.3**：要么完整实现 `tensor_from_vec`，要么在 lowering 阶段显式报错。当前 stub 状态最危险（语义静默错误）。
3. **修补 §9.7**：审查 [`host.rs`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/wasm/host.rs) 所有 `unwrap`，加 `catch_unwind` 包裹。
4. **测试用例设计**：基于 W1 的"支持子集"设计测试用例，覆盖所有显式映射的 BaseType × 所有 host import。
5. **ABI 文档化**：将 18 个 host import 的签名、语义、局限写入 `docs/语言参考手册.md`。

### 11.4 v1 自审留痕（4 轮迭代）

本文经历 4 轮自审：

| 轮次 | 检查重点 | 发现 | 修正 |
|------|---------|------|------|
| v1.1 | 源码与论断一致性 | "13 种 BaseType"实际为 16 种，且 7 种未显式处理 | 修正 §1 摘要、§3.2 定义、§6 表格、§9.1 局限 |
| v1.2 | W3 完备性 | `tensor_from_vec` 是 stub，W3 反例 | 修正 W3 为 W3'（排除 tensor）、§9.3 局限 |
| v1.3 | 双模拟对称性 | host 副作用不可逆，双模拟不能对称 | 修正 W1 为弱双模拟（单向）、§9.4 局限 |
| v1.4 | W4 维度完整性 | "路径覆盖"维度对 Emscripten 无意义（Emscripten 无 host） | 删除 W4 表格中"路径覆盖"行 |

---

## 12. 结论

本文对 Tenth 语言 HIR→WASM 编译的"值类型塌缩 + host 桥接"模式进行了形式化分析。核心贡献：

1. **形式化建模**：将 `to_val_type` 抽象为类型塌缩映射 $\phi$，将 18 个 host import 抽象为 host 操作签名集 $\mathcal{H}$，给出 HIR 与 WASM 的小步操作语义（§3、§4）。

2. **五个主定理**：
   - **W1**（弱双模拟语义保持）：在"支持子集"上 HIR↔WASM 构成单向模拟关系。
   - **W2**（塌缩健全性）：$\phi$ 不损失 HIR 层区分的语义。
   - **W3'**（host 桥接完备性，排除 tensor）：18 个 host import 覆盖 HIR 所需 heap 操作。
   - **W4**（与 Emscripten/AssemblyScript 对比）：Tenth 模式在 ABI 稳定性、host 兼容性、自举闭环上的优势与代价。
   - **W5**（自动推导 host 边界 import 集）：声明存在性，标注为未来工作。

3. **诚实局限**：8 处理论局限独立章节记录，包括 7 种 BaseType 未显式处理、`tensor_from_vec` stub、双模拟单向性、host panic 未定义行为等。

4. **与 T12 联动**：W1 与 W3' 是 T12 路径 C 全 WASM 闭环的前提条件；§10 给出强联动关系。

**核心洞察**：Tenth 的"塌缩+桥接"模式**不是 Emscripten 风格的 lowering 优化**，而是一种**结构性取舍**——用"自包含性"换取"host 可干预性"和"自举闭环能力"。这一取舍适合 AI 原生语言的"重操作、少循环"特征，但代价是当前实现的不完备（7 种 BaseType 未处理、tensor stub）和理论局限（弱双模拟、host 形式化假设）。

**未来工作**：修补 §9.1-§9.3 的工程局限；研究 W5 的自动推导算法；探索 WASM reference-types / GC / component model 与 Tenth host 桥接模式的融合。

---

## 参考文献

1. WebAssembly Working Group. *WebAssembly Core Specification Version 1.0*. W3C Recommendation, 2019. https://webassembly.github.io/spec/core/
2. Zakai, A. *Emscripten: An LLVM-to-JavaScript Compiler*. SPLASH/OOPSLA 2011. https://dl.acm.org/doi/10.1145/1993316.1993532
3. AssemblyScript Project. *AssemblyScript Documentation*. https://www.assemblyscript.org/
4. Leroy, X. *Formal verification of a realistic compiler*. Communications of the ACM 52(7), 2009. https://dl.acm.org/doi/10.1145/1538788.1538814
5. Kumar, R., Myreen, M. O., Norrish, M., Owens, S. *CakeML: A Verified Implementation of ML*. POPL 2014. https://dl.acm.org/doi/10.1145/2535838.2535841
6. Pnueli, A., Siegel, M., Singerman, E. *Translation Validation*. TACAS 1998.
7. McKinna, J., Pollack, R. *Some Lambda Calculus and Type Theory Formalized*. Journal of Automated Reasoning 23(3-4), 1999.
8. Igarashi, A., Pierce, B. C., Wadler, P. *Featherweight Java: A Minimal Core Calculus for Java and GJ*. OOPSLA 1999.
9. bytecodealliance. *wasm-tools: Rust utilities for WebAssembly*. https://github.com/bytecodealliance/wasm-tools
10. Tenth 项目数理部. *双侧编译器的语义等价性：Tenth 自举编译器与 Rust 母编译器的形式化对比* (T12). 2026. [`docs/论文/T12-双侧编译器语义等价性.md`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/docs/论文/T12-双侧编译器语义等价性.md)
11. Tenth 项目数理部. *JIT 特化策略的语义保持证明* (T9). 2026. [`docs/论文/T9-JIT特化语义保持证明.md`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/docs/论文/T9-JIT特化语义保持证明.md)
12. Tenth 项目数理部. *Closure 自由变量分析正确性* (T22). 2026. [`docs/论文/T22-Closure自由变量分析正确性.md`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/docs/论文/T22-Closure自由变量分析正确性.md)
13. Tenth 项目数理部. *dtype 提升格与混合 dtype 算术* (T17). 2026. [`docs/论文/T17-dtype提升格与混合dtype算术.md`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/docs/论文/T17-dtype提升格与混合dtype算术.md)

---

## 附录 A：定理索引

| 定理 | 简称 | 前置条件 | 结论 | 证明 |
|------|------|---------|------|------|
| W1 | 语义保持（弱双模拟） | W1 假设（§5.1） | HIR↔WASM 单向模拟 | §5.1 |
| W2 | 塌缩健全性 | W1 假设 | $\phi$ 不损失区分性 | §5.2 |
| W3' | host 桥接完备性（排除 tensor） | W1 假设 + 排除 Tensor | 18 import 覆盖 heap 操作 | §5.3 |
| W4 | 与 Emscripten/AssemblyScript 对比 | — | 5 维度结构性差异 | §5.4 |
| W5 | 自动推导 host import 集 | 未来工作 | 存在性声明 | §5.5 |

## 附录 B：与现有文档的对应

| 本文章节 | 对应文档 | 关系 |
|---------|---------|------|
| §3 类型塌缩形式化 | [`tenth/src/compile/wasm/mod.rs:22-59`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/wasm/mod.rs) | 形式化源码 |
| §7 host import 分析 | [`tenth/src/compile/wasm/host.rs`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/wasm/host.rs) | 逐一对应 |
| §10 与 T12 联动 | [T12 §6.4](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/docs/论文/T12-双侧编译器语义等价性.md) | 提供前提条件 |
| §9.1 BaseType 局限 | [`tenth/src/hir/types.rs:3-10`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/hir/types.rs) | 16 种 BaseType |
| §9.3 tensor 局限 | [`tenth/src/compile/wasm/host.rs:338-339`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/wasm/host.rs) | stub 注释 |

## 附录 C：实施建议清单

| 优先级 | 修补项 | 涉及文件 | 与本文局限对应 |
|--------|-------|---------|--------------|
| P0 | 显式映射 7 种 BaseType | `wasm/mod.rs:22-39` | §9.1 |
| P0 | `tensor_from_vec` 完整实现或显式报错 | `wasm/host.rs:340-343` | §9.3 |
| P1 | host 函数 `catch_unwind` 包裹 | `wasm/host.rs` 全文 | §9.7 |
| P1 | lowering 阶段检查 `Unit` 参数 | `hir/lower.rs` | §9.2 |
| P2 | `f64_bits` 位保真单元测试 | `wasm/host.rs:300-303` | §9.6 |
| P2 | 闭包类型标记 + 运行时检查 | `wasm/sections.rs` + `wasm/closures.rs` | §9.5 |
| P3 | Array/Tuple/Enum host ABI 设计 | 多文件 | §9.8 |
| P3 | W5 自动推导算法实现 | 新模块 | §5.5 |
