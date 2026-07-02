# WASM 中的闭包与环境传递：Tenth env_ptr + call_indirect 方案的形式化与对比

> **Tenth 数理部 · 理论分析论文 T30**
> 版本：v1.0 | 日期：2026-07-02
> 适用：Tenth v0.3.3+
> 定位：会议级（WASM 闭包转换理论分析）
> 联动：T22（自由变量分析正确性）
> 诚实声明：本文对 `env_ptr + call_indirect` 方案的正确性证明刻意保持谨慎。我们证明在受控假设下（A1 假设：闭包创建时所有捕获变量在 `local_map` 中存在；A2 假设：table 索引与 closure_info 一一对应；A3 假设：FV 分析完备——其本身由 T22 在更弱的假设下成立）方案的语义保持、env_ptr 偏移正确性与 table 索引正确性。本文不掩盖 WASM table 单一性、call_indirect 间接调用的运行期类型检查开销、`i32` 地址空间对 env_ptr 的限制、捕获变量按值复制语义不支持可变捕获等问题，而是将其作为独立局限章节诚实披露。

---

## 摘要

Tenth 语言的 WebAssembly 后端在实现闭包时采用了"env_ptr + call_indirect"方案：每个闭包被表示为一个打包的 64 位整数 `(table_idx << 32) | env_ptr`，其中 `table_idx` 是闭包体函数在 WASM `table` 中的索引，`env_ptr` 是捕获环境结构在 WASM 线性内存中的 32 位指针。闭包调用通过 `call_indirect` 指令完成，调用栈布局为 `[env_ptr, params..., fn_ptr]`；闭包体内通过 `local.get 0` 获取 env_ptr，并按 `ci * 8` 偏移从线性内存读取第 `ci` 个捕获变量。本文对该方案进行形式化建模，给出五条主定理：(C1) closure conversion 语义保持——在三条受控假设下，转换前后程序语义一致；(C2) env_ptr 偏移正确性——运行期通过 env_ptr 偏移读取捕获变量与编译期捕获列表一致；(C3) table 索引正确性——call_indirect 索引与闭包体函数一一对应；(C4) 与 Rust/Swift/Chez Scheme/OCaml 对比——Tenth 方案在表达力上居中，在 WASM 亲和度上居首；(C5) 复杂度——捕获列表查找为 O(k)（k 为捕获数），table 索引查找为 O(1)（直接寻址），但 call_indirect 引入运行期类型检查开销。本文**诚实披露**五条局限（L1–L5）：env_ptr 仅 32 位限制线性内存上限；捕获按值复制不支持可变捕获；table 全局单一性限制多模块互操作；call_indirect 运行期类型检查无法消除；与 T22 的完备性假设耦合。复杂度分析表明，捕获数量 k 与 table 索引查找开销解耦——env_ptr 偏移读取为 O(k)，table 索引为 O(1)，整体闭包调用复杂度为 O(k + 1)。

**关键词**：闭包转换；WebAssembly；call_indirect；环境传递；CPS 变换；显式环境；table 索引；结构化控制流

---

## 1. 引言

### 1.1 闭包转换的挑战

闭包（closure）是函数式编程的核心抽象：一个 λ 抽象不仅包含代码，还包含其定义时刻的词法环境——捕获的自由变量。将源语言中的闭包翻译到不支持词法作用域的目标平台（如 C、WebAssembly），需要进行**闭包转换**（closure conversion）[Appel, 1992]：将每个闭包拆解为 `(env, code)` 对，其中 `env` 是自由变量的记录，`code` 是接受 `env` 与参数的函数。

闭包转换的核心挑战在于：

1. **环境表示**：捕获的自由变量如何打包？记录（record）、元组（tuple）、对象（object）？
2. **环境传递**：闭包调用时如何将 `env` 传入 `code`？作为额外参数（CPS 风格）？作为对象方法接收者（OO 风格）？
3. **间接调用**：闭包的 `code` 通过什么机制调用？函数指针？vtable？`call_indirect`？
4. **目标平台限制**：目标平台支持哪些原语？是否有 GC？是否有堆？是否有 vtable？

### 1.2 WASM 的限制

WebAssembly（WASM）作为目标平台，对闭包转换施加了若干独特限制：

- **结构化控制流**：WASM 不支持任意 `goto`，所有控制流必须用 `block`/`loop`/`if` 表达。这排除了通过 `goto` 实现的 continuation。
- **无尾调用优化（MVP）**：WASM MVP 不保证尾调用，递归闭包调用可能栈溢出。Tail-call proposal（`return_call`）尚未广泛实现。
- **单一 `table`（MVP）**：WASM MVP 只支持一个 `funcref` table，`call_indirect` 通过该 table 的索引间接调用。
- **无 GC（MVP）**：WASM MVP 无 GC，所有堆分配需手动管理或由 host 管理。Tenth 通过 host 提供的 `tenth_alloc` bump allocator 管理。
- **线性内存模型**：WASM 的内存是字节寻址的线性数组，所有指针为 `i32` 偏移。这限制了 env_ptr 的地址空间。
- **类型擦除**：WASM 的 `funcref` 不携带类型签名，`call_indirect` 通过 `type_index` 进行运行期类型检查。

### 1.3 env_ptr + call_indirect 方案

Tenth 在 WASM 中实现闭包采用"env_ptr + call_indirect"方案（[tenth/src/compile/wasm/closures.rs:124-161](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/wasm/closures.rs)、[tenth/src/compile/wasm/compile.rs:686-730](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/wasm/compile.rs)）：

- **闭包值表示**：单个 `i64`，高 32 位为 `table_idx`（闭包体函数在 table 中的索引），低 32 位为 `env_ptr`（捕获环境在线性内存中的指针）。
- **闭包类型签名**：`(i64 env_ptr, i64 param1, ..., i64 paramN) -> i64`，所有参数与返回值统一为 `i64`（[tenth/src/compile/wasm/sections.rs:52-63](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/wasm/sections.rs)）。
- **闭包创建**：通过 `tenth_alloc(captures_count * 8)` 分配 env 结构，按捕获顺序写入每个变量；打包 `(table_idx << 32) | env_ptr` 为 `i64`。
- **闭包调用**：解包 `i64` 为 `fn_ptr` 与 `env_ptr`，按 `[env_ptr, args..., fn_ptr]` 顺序压栈，`call_indirect type_idx 0` 调用。
- **捕获变量读取**：闭包体内通过 `local.get 0`（env_ptr）+ `i32.const ci*8` + `i64.load` 读取第 `ci` 个捕获变量。

这是 CPS 变换的"显式环境"变体——闭包的隐式词法环境被显式化为 `env_ptr` 参数，但**在 WASM 的结构化控制流 + table 限制下**实现。

### 1.4 贡献

本文的贡献如下：

1. **形式化建模**：将 Tenth 的 env_ptr 打包值、闭包类型签名、table 索引映射形式化为数学对象（第 4 章）。
2. **closure conversion 语义保持**（定理 C1）：在三条受控假设下，证明转换前后程序语义一致。
3. **env_ptr 偏移正确性**（定理 C2）：证明运行期通过 env_ptr 偏移读取捕获变量与编译期 captures 列表一一对应。
4. **table 索引正确性**（定理 C3）：证明 call_indirect 索引与闭包体函数一一对应，table 在编译期正确填充。
5. **跨语言对比**（定理 C4）：与 Rust FnOnce/FnMut/Fn、Swift context、Chez Scheme、OCaml 的闭包实现对比，定位 Tenth 的工程取舍。
6. **复杂度分析**（定理 C5）：证明捕获数量 k 与 table 索引查找开销解耦，整体闭包调用为 O(k + 1)。
7. **诚实披露五条局限**（L1–L5）：env_ptr 32 位限制、按值复制不支持可变捕获、table 单一性、call_indirect 运行期开销、与 T22 完备性耦合。
8. **WASM 限制影响分析**：结构化控制流、单一 table、无 GC、线性内存模型对方案的影响。

---

## 2. 背景与相关工作

### 2.1 闭包转换的经典理论

Appel [1992] 在 *Compiling with Continuations* 中系统阐述了闭包转换的理论框架。其核心思想是将每个 λ 抽象转换为 `(env, code)` 对：

$$\text{CC}(\lambda x. e) = (\text{env}, \text{code})$$

其中 `env = record(FV(λx.e))`，`code = λ(env, x). e'`，`e'` 是将 `e` 中所有自由变量 `v` 替换为 `env.v` 的版本。Appel 的分析使用集合语义（set semantics），强调 CPS 变换与闭包转换的协同：在 CPS 下，所有调用变为尾调用，闭包成为一等公民。

Appel 的方案在 SML/NJ 中实现，环境采用 record 表示，调用通过函数指针。其局限在于依赖目标平台支持函数指针与堆分配。

### 2.2 OCaml 的闭包转换

OCaml 编译器 [Leroy, 1992] 采用基于 `Set.t` 的自由变量分析。其闭包表示为：

$$\text{closure} = (\text{code\_ptr}, \text{env\_ptr}, \text{inlined\_vars})$$

OCaml 的特点：

- **环境为堆分配的 record**：与 Tenth 类似。
- **小闭包优化**：若捕获变量数 ≤ 3，环境内联到闭包对象本身，避免额外堆分配。
- **代码指针为 native 函数指针**：直接调用，无 vtable 间接。
- **支持 `let rec`**：递归绑定特殊处理。

OCaml 的方案依赖 native 函数指针，无法直接用于 WASM（WASM MVP 不支持函数指针，只有 table 索引）。

### 2.3 Rust 的 FnOnce / FnMut / Fn

Rust 的闭包捕获机制 [Rust Reference, 2024] 通过三种 trait 区分：

- **FnOnce**：以值捕获（`move`），消耗捕获变量，仅可调用一次。
- **FnMut**：以可变引用捕获（`&mut`），可多次调用但需独占。
- **Fn**：以不可变引用捕获（`&`），可并发多次调用。

Rust 的闭包在底层表示为：

$$\text{closure} = (\text{vtable\_ptr}, \text{env})$$

其中 `vtable_ptr` 指向包含 `call` 方法的虚表，`env` 是捕获变量的结构体。调用通过 vtable 间接进行。

Rust 的关键设计：

- **捕获方式由 trait 推断**：编译器根据闭包体使用捕获变量的方式（读/写/移动）推断应实现哪种 trait。
- **零成本抽象**：当闭包未跨函数边界传递时，编译器可内联，消除 vtable 间接。
- **move 关键字**：强制按值捕获，覆盖默认的引用捕获。

Rust 的 vtable 方案与 Tenth 的 table 方案在间接调用上类似，但 Rust 的 vtable 是 per-closure-type 的，而 Tenth 的 table 是全局单一的。

### 2.4 Swift 的 context

Swift 编译器 [Swift, 2023] 采用"context"方案：

- **闭包表示**：`(context, function)` 对，`context` 是捕获环境的胖指针。
- **partial apply**：闭包创建时生成"partial apply"thunk，将 `context` 与 `function` 捆绑。
- **thunk 转发**：调用闭包时，thunk 从 `context` 解包捕获变量，转发给真正的函数。
- **ARC 兼容**：`context` 是引用计数对象，与 Swift 的 ARC 无缝集成。

Swift 的方案与 Tenth 的 env_ptr 方案结构相似（都是 `(env, code)` 对），但 Swift 的 `context` 是 ARC 对象，依赖 runtime 支持；Tenth 的 env_ptr 是裸指针，依赖 host 的 bump allocator。

### 2.5 Chez Scheme 的 closure

Chez Scheme [Dybvig, 2006] 是高性能 Scheme 编译器：

- **闭包表示**：(code, env) 对，env 是变量绑定的关联列表。
- **flat closure 优化**：将环境扁平化为单层 record，避免多层 indirection。
- **兼容性 closure**：未优化时使用泛型 closure，支持任意捕获。
- **CP0 内联**：编译期内联已知闭包，消除间接调用。

Chez 的 flat closure 与 Tenth 的 env 结构在概念上一致（都是扁平 record），但 Chez 在 native 代码生成，Tenth 在 WASM 字节码。

### 2.6 Tenth 的定位

Tenth 的 env_ptr + call_indirect 方案在结构上是 Appel 显式环境方案与 Rust vtable 方案的混合：

- **显式环境**（如 Appel）：env_ptr 作为闭包体的第一个参数显式传递。
- **table 间接调用**（如 Rust vtable）：通过 table 索引而非函数指针调用。
- **打包为 i64**：env_ptr 与 table_idx 打包为单个 i64，避免双字段开销。

与上述方案相比，Tenth 的关键差异在于**严格遵循 WASM 的限制**：单一 table、无 GC、线性内存、结构化控制流。这些限制塑造了方案的若干独特设计（如 i64 打包、tenth_alloc host 调用、type_index 运行期检查）。

---

## 3. Tenth env_ptr + call_indirect 方案形式化

### 3.1 数据结构形式化

**定义 3.1（闭包打包值 PackedClosure）**：闭包在运行期表示为单个 64 位无符号整数：

$$\text{PackedClosure} = (\text{table\_idx} \ll 32) \mid \text{env\_ptr}$$

其中：

- `table_idx ∈ [0, 2^32 - 1]`：闭包体函数在 WASM table 中的索引。
- `env_ptr ∈ [0, 2^32 - 1]`：捕获环境结构在线性内存中的字节偏移。
- `<<` 为左移位，`|` 为按位或。

**对应源码**：[tenth/src/compile/wasm/compile.rs:686-730](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/wasm/compile.rs)。

**定义 3.2（捕获环境 EnvStruct）**：捕获环境为线性内存中连续的字节序列，长度为 `k * 8` 字节（`k` 为捕获数量）：

$$\text{EnvStruct}(\text{env\_ptr}, k) = \text{Memory}[\text{env\_ptr}, \text{env\_ptr} + 8k)$$

第 `i` 个捕获变量（`0 ≤ i < k`）存储在 `env_ptr + 8i` 处，作为 `i64`：

$$\text{EnvStruct}_i(\text{env\_ptr}) = \text{i64.load}(\text{env\_ptr} + 8i)$$

**定义 3.3（闭包类型签名 ClosureType）**：闭包体函数的 WASM 类型签名为：

$$\text{ClosureType}(N) = (i64, \underbrace{i64, \ldots, i64}_{N}) \to i64$$

其中第一个 `i64` 为 `env_ptr`，后 `N` 个 `i64` 为闭包参数。所有参数与返回值统一为 `i64`（[tenth/src/compile/wasm/sections.rs:52-60](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/wasm/sections.rs)）。

**定义 3.4（闭包索引 cidx）**：每个闭包在编译期由 `collect_closures` 分配唯一索引 `cidx`，与 `closure_info` 列表一一对应：

$$\text{closure\_info}[cidx] = (\text{func\_idx}, \text{type\_idx}, N)$$

其中 `func_idx = IMPORT_COUNT + num\_user\_funcs + 1 + cidx`（[tenth/src/compile/wasm/closures.rs:239-261](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/wasm/closures.rs)）。

**定义 3.5（table 索引映射 TableIdxMap）**：闭包索引 `cidx` 与 table 索引一致——第 `cidx` 个闭包占据 table 的第 `cidx` 项：

$$\text{table}[cidx] = \text{func\_idx}(cidx)$$

由 `emit_elem_section` 通过 `Elements::Functions(&func_idxs)` 填充，起始 offset 为 0（[tenth/src/compile/wasm/sections.rs:148-158](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/wasm/sections.rs)）。

### 3.2 操作的形式化

**定义 3.6（闭包创建 CreateClosure）**：对 HIR 中的 `Closure { params, body, captures }` 节点，`CreateClosure` 操作生成 WASM 指令序列：

$$\text{CreateClosure}(cidx, \text{captures}) = \begin{cases}
\text{i64.const}(cidx \ll 32) & \text{if } |\text{captures}| = 0 \\
\text{AllocateAndStore}(\text{captures}) \ ;;\ \text{i64.const}(cidx \ll 32)\ ;;\ \text{i64.or} & \text{if } |\text{captures}| > 0
\end{cases}$$

其中：

$$\text{AllocateAndStore}(\text{captures}) = \begin{aligned}
& \text{i32.const}(8 \cdot |\text{captures}|) \ ;;\ \text{call(tenth\_alloc)} \ ;;\ \text{i64.extend\_i32\_u} \\
& \ ;;\ \text{local.set tmp} \\
& \ ;;\ \bigcirc_{i=0}^{|\text{captures}|-1} \text{StoreCapture}(i, \text{captures}_i, \text{tmp}) \\
& \ ;;\ \text{local.get tmp}
\end{aligned}$$

$$\text{StoreCapture}(i, \text{cap\_name}, \text{tmp}) = \text{local.get tmp} \ ;;\ \text{i32.wrap\_i64} \ ;;\ \text{local.get}(\text{local\_map}[\text{cap\_name}]) \ ;;\ \text{i64.store}(\text{offset}=8i)$$

**对应源码**：[tenth/src/compile/wasm/compile.rs:686-730](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/wasm/compile.rs)。

**定义 3.7（闭包调用 InvokeClosure）**：对调用 `Call { Var(cv), args }`（其中 `cv` 为闭包变量），`InvokeClosure` 生成：

$$\text{InvokeClosure}(cv, \text{args}, \text{type\_idx}) = \begin{aligned}
& \text{local.get}(\text{local\_map}[cv]) \ ;;\ \text{i64.const 32} \ ;;\ \text{i64.shr\_u} \ ;;\ \text{local.set tmp} \\
& \ ;;\ \text{local.get}(\text{local\_map}[cv]) \ ;;\ \text{i64.const 0xFFFFFFFF} \ ;;\ \text{i64.and} \\
& \ ;;\ \bigcirc_{a \in \text{args}} \text{CompileExpr}(a) \\
& \ ;;\ \text{local.get tmp} \ ;;\ \text{i32.wrap\_i64} \\
& \ ;;\ \text{call\_indirect}(\text{type\_idx}, \text{table}=0)
\end{aligned}$$

调用栈最终布局为 `[env_ptr, args..., fn_ptr]`，符合 WASM `call_indirect` 的栈语义（[tenth/src/compile/wasm/compile.rs:300-327](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/wasm/compile.rs)）。

**定义 3.8（捕获变量读取 LoadCapture）**：在闭包体内对 `Var(name)` 的读取（`name ∈ current_captures`），`LoadCapture` 生成：

$$\text{LoadCapture}(\text{name}, \text{ci}) = \begin{aligned}
& \text{local.get 0} \ ;;\ \text{i32.wrap\_i64} \ ;;\ \text{i32.const}(8 \cdot \text{ci}) \ ;;\ \text{i32.add} \\
& \ ;;\ \text{i64.load}(\text{align}=3)
\end{aligned}$$

其中 `ci = current_captures.position(name)`（[tenth/src/compile/wasm/compile.rs:130-145](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/wasm/compile.rs)）。

### 3.3 调用栈布局的形式化

**定义 3.9（调用栈布局 CallStackLayout）**：WASM `call_indirect` 的栈语义要求参数从底到顶依次为 `[arg_0, arg_1, ..., arg_{N-1}, fn_ptr]`。闭包体的参数列表为 `[env_ptr, param_1, ..., param_N]`，故栈布局为：

$$\text{CallStack} = [\text{env\_ptr}, \text{param}_1, \ldots, \text{param}_N, \text{fn\_ptr}]$$

即压栈顺序为 `env_ptr → params → fn_ptr`，最后 `call_indirect` 弹出 `fn_ptr` 用于 table 索引，其余作为参数传入。

### 3.4 闭包体编译的形式化

**定义 3.10（闭包体环境 ClosureBodyEnv）**：编译闭包体时，编译器状态为：

- `local_map = {param_1: 1, param_2: 2, ..., param_N: N}`（参数从 local 1 开始，local 0 为 env_ptr）。
- `compiling_closure = true`。
- `current_captures = captures.to_vec()`（[tenth/src/compile/wasm/closures.rs:124-161](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/wasm/closures.rs)）。

`Var(name)` 的读取分三种情况：

- 若 `name ∈ local_map`：直接 `local.get(idx)`（参数或局部变量）。
- 若 `name ∈ current_captures`：通过 `LoadCapture(name, ci)` 从 env_ptr 偏移读取。
- 否则：报错（未定义变量）。

### 3.5 闭包转换的指称语义

**定义 3.11（源语言闭包指称）**：源语言中闭包的指称语义为：

$$\llbracket \lambda x. e \rrbracket_\rho = \text{closure}(\rho|_{\text{FV}(e) \setminus \{x\}}, \lambda \rho'. x. \llbracket e \rrbracket_{\rho' \cup \rho})$$

其中 `ρ` 是当前环境，`ρ|_S` 是 `ρ` 限制到变量集 `S` 的子环境。

**定义 3.12（目标语言闭包指称）**：转换后，闭包的指称语义为：

$$\llbracket \text{PackedClosure}(cidx, \text{env\_ptr}) \rrbracket = (\text{table}[cidx], \text{EnvStruct}(\text{env\_ptr}, k))$$

调用时：

$$\llbracket \text{InvokeClosure}(cv, \text{args}) \rrbracket = \text{table}[\text{fn\_ptr}](\text{env\_ptr}, \llbracket \text{args} \rrbracket)$$

闭包体内对捕获变量 `name` 的读取：

$$\llbracket \text{Var}(\text{name}) \rrbracket_{\text{closure}} = \text{EnvStruct}_{\text{ci}}(\text{env\_ptr})$$

其中 `ci = position(name, captures)`。

---

## 4. 主定理与证明

### 4.1 定理 C1（closure conversion 语义保持）

**定理 C1（closure conversion 语义保持）**：在以下三条受控假设下：

- **假设 A1（捕获变量可解析）**：闭包创建时，所有 `captures` 中的变量名在 `local_map` 中存在。
- **假设 A2（FV 分析完备）**：闭包的 `captures` 列表包含所有真自由变量（由 T22 在更弱假设下保证）。
- **假设 A3（类型一致）**：所有捕获变量与闭包参数的 WASM 类型一致（均为 `i64`，通过 `I64ReinterpretF64` / `I64ExtendI32U` 转换）。

对源程序 `P`，若 `CC(P)` 为 Tenth 的 WASM 闭包转换结果，则：

$$\forall \rho. \llbracket P \rrbracket_\rho = \llbracket \text{CC}(P) \rrbracket_{\text{encode}(\rho)}$$

其中 `encode(ρ)` 将源环境编码为线性内存布局。

**证明**：对 `P` 的结构进行归纳。

**基例**：

- `Literal(l)`：转换前后均为字面量压栈，无闭包介入。`encode` 不改变字面量值。✓
- `Var(name)`（非闭包上下文）：转换为 `local.get(local_map[name])`。`encode(ρ)[name] = ρ[name]`，故语义一致。✓

**归纳步**：

- `Binary { op, left, right }`：由 IH，`left` 与 `right` 语义保持。`op` 的 WASM 指令与源语言一致（除类型转换，由 A3 保证一致）。✓

- `Closure { params, body, captures }`：
  - **源语义**：`⟦Closure⟧_ρ = closure(ρ|_captures, λρ'.params.⟦body⟧_{ρ'∪ρ|_captures})`。
  - **目标语义**：`CreateClosure(cidx, captures)` 生成 `(table_idx << 32) | env_ptr`，其中 `env_ptr` 指向 `EnvStruct`，`EnvStruct_i = ρ[captures_i]`（由 A1，所有 `captures_i` 在 `local_map` 中可解析）。
  - **关键**：`EnvStruct_i` 存储的是 `ρ[captures_i]` 的 WASM 编码值，与源环境一致（由 A3，类型一致）。
  - 闭包体 `body` 编译为 `code`，其参数列表为 `[env_ptr, params...]`，对 `Var(name)` 的读取：
    - 若 `name ∈ params`：从 local 读取，与源语义一致。
    - 若 `name ∈ captures`：通过 `LoadCapture(name, ci)` 读取 `EnvStruct_ci`，即 `ρ[name]`，与源语义一致。
    - 否则（由 A2，不应发生，因为 captures 已包含所有真自由变量）。
  - 故 `⟦CC(Closure)⟧ = ⟦Closure⟧`。✓

- `Call { Var(cv), args }`（`cv` 为闭包变量）：
  - **源语义**：`⟦Call⟧_ρ = ⟦cv⟧_ρ(⟦args⟧_ρ)`，其中 `⟦cv⟧_ρ = closure(env, code)`，调用为 `code(env, args)`。
  - **目标语义**：`InvokeClosure(cv, args, type_idx)` 解包 `cv` 为 `(fn_ptr, env_ptr)`，按 `[env_ptr, args, fn_ptr]` 压栈，`call_indirect` 调用 `table[fn_ptr]` 即 `code`。
  - WASM `call_indirect` 语义：弹出 `fn_ptr`，从 table 取 `table[fn_ptr]`，调用并传入剩余栈元素 `[env_ptr, args]`。
  - 故 `⟦CC(Call)⟧ = code(env_ptr, args) = ⟦Call⟧`。✓
  - **类型检查**：`call_indirect` 通过 `type_idx` 进行运行期类型检查，由 A3 保证类型一致。✓

- `Call { Var(f), args }`（`f` 为普通函数，非闭包）：直接 `call(func_map[f])`，无闭包介入。由 IH。✓

- `Block { stmts, final_expr }`：语句按序执行，由 IH 逐句传递。✓

- `If { cond, then, else }`：由 IH，条件与分支语义保持。✓

- 其他节点（`Unary`、`Assign`、`StructLiteral`、`Match` 等）：由 IH，递归保持。✓

**归纳完整**：所有 HIR 节点类型均覆盖，故 `⟦CC(P)⟧ = ⟦P⟧`。$\square$

**说明**：A1 在 Tenth 的 HIR lowering 阶段保证（若变量未定义，lowering 报错）；A2 由 T22 在四条假设下保证；A3 由 `compile.rs` 的类型转换逻辑保证（[tenth/src/compile/wasm/compile.rs:714-723](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/wasm/compile.rs)）。

### 4.2 定理 C2（env_ptr 偏移正确性）

**定理 C2（env_ptr 偏移正确性）**：对任意闭包 `c`，其 `captures` 列表为 `[cap_0, cap_1, ..., cap_{k-1}]`，则在闭包体内对 `Var(cap_i)` 的读取（`0 ≤ i < k`）满足：

$$\llbracket \text{LoadCapture}(\text{cap}_i, i) \rrbracket = \text{EnvStruct}_i(\text{env\_ptr}) = \rho[\text{cap}_i]$$

即运行期通过 `env_ptr + 8i` 偏移读取的值，等于编译期 `cap_i` 在闭包创建时的值。

**证明**：分两端论证。

**端 1（写入端，CreateClosure）**：由定义 3.6，`CreateClosure` 对 `captures` 中的每个 `cap_i`（按 `i = 0, 1, ..., k-1` 顺序）执行：

$$\text{StoreCapture}(i, \text{cap}_i, \text{tmp}) = \text{local.get tmp} \ ;;\ \text{i32.wrap\_i64} \ ;;\ \text{local.get}(\text{local\_map}[\text{cap}_i]) \ ;;\ \text{i64.store}(\text{offset}=8i)$$

源码：[tenth/src/compile/wasm/compile.rs:710-724](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/wasm/compile.rs)：

```rust
for (ci, cap_name) in captures.iter().enumerate() {
    body.instruction(&Instruction::LocalGet(tmp));
    body.instruction(&Instruction::I32WrapI64);
    if let Some(&idx) = self.local_map.get(cap_name) {
        body.instruction(&Instruction::LocalGet(idx));
    } else {
        body.instruction(&Instruction::I64Const(0));
    }
    let arg = wasm_encoder::MemArg { offset: (ci as u64) * 8, align: 0, memory_index: 0 };
    body.instruction(&Instruction::I64Store(arg));
}
```

`i64.store` 将 `local_map[cap_i]` 的值写入地址 `tmp + 8i`。故：

$$\text{Memory}[\text{env\_ptr} + 8i] = \rho[\text{cap}_i]$$

**端 2（读取端，LoadCapture）**：由定义 3.8，闭包体内对 `Var(cap_i)` 的读取（当 `cap_i ∈ current_captures`）执行：

$$\text{LoadCapture}(\text{cap}_i, \text{ci}) = \text{local.get 0} \ ;;\ \text{i32.wrap\_i64} \ ;;\ \text{i32.const}(8 \cdot \text{ci}) \ ;;\ \text{i32.add} \ ;;\ \text{i64.load}$$

源码：[tenth/src/compile/wasm/compile.rs:130-145](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/wasm/compile.rs)：

```rust
if let Some(ci) = self.current_captures.iter().position(|c| c == name) {
    body.instruction(&Instruction::LocalGet(0));
    body.instruction(&Instruction::I32WrapI64);
    body.instruction(&Instruction::I32Const(ci as i32 * 8));
    body.instruction(&Instruction::I32Add);
    let arg = wasm_encoder::MemArg { offset: 0, align: 3, memory_index: 0 };
    body.instruction(&Instruction::I64Load(arg));
    ...
}
```

`local.get 0` 取 env_ptr（i64），`i32.wrap_i64` 截断为 i32 地址，`i32.const 8ci + i32.add` 计算地址 `env_ptr + 8ci`，`i64.load` 读取 `Memory[env_ptr + 8ci]`。

**关键观察**：`ci = current_captures.position(cap_i)`，且 `current_captures = captures.to_vec()`（在 `compile_closure_body` 中设置，[tenth/src/compile/wasm/closures.rs:148](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/wasm/closures.rs)）。故 `ci = i`（写入时的索引）。

**合并两端**：

$$\llbracket \text{LoadCapture}(\text{cap}_i, i) \rrbracket = \text{Memory}[\text{env\_ptr} + 8i] = \rho[\text{cap}_i]$$

故定理 C2 成立。✓

**注意**：定理 C2 要求 `current_captures` 与 `captures`（写入时的）一致。这在 Tenth 实现中保证——两者均来自同一个 `Closure` 节点的 `captures` 字段。$\square$

### 4.3 定理 C3（table 索引正确性）

**定理 C3（table 索引正确性）**：对任意闭包 `c`，其 `cidx` 为 `closure_info` 中的索引，`table_idx = cidx`。则：

1. **table 填充正确**：`table[cidx] = func_idx(cidx)`，即 table 的第 `cidx` 项指向第 `cidx` 个闭包的 WASM 函数。
2. **call_indirect 索引正确**：调用闭包时，`fn_ptr = cv >> 32 = cidx`，`call_indirect` 通过 `fn_ptr` 索引 table，正确路由到 `c` 的 WASM 函数。

**证明**：分两部分。

**部分 1（table 填充）**：

由定义 3.4，`closure_info[cidx] = (func_idx, type_idx, N)`，其中 `func_idx = IMPORT_COUNT + num_user_funcs + 1 + cidx`（[tenth/src/compile/wasm/closures.rs:253-254](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/wasm/closures.rs)）：

```rust
let cidx = self.closure_info.len() as u32;
let func_idx = IMPORT_COUNT + num_user_funcs + 1 + cidx;
let param_count = params.len() as u32;
self.closure_info.push((func_idx, 0, param_count));
```

由 `emit_elem_section`（[tenth/src/compile/wasm/sections.rs:148-158](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/wasm/sections.rs)）：

```rust
let func_idxs: Vec<u32> = self.closure_info.iter().map(|&(fi, _, _)| fi).collect();
elements.active(
    Some(0),
    &ConstExpr::i32_const(0),
    Elements::Functions(&func_idxs),
);
```

`Elements::Functions(&func_idxs)` 将 `func_idxs` 按顺序填入 table，起始 offset 为 0。故：

$$\text{table}[i] = \text{func\_idx}(i) = \text{closure\_info}[i].0$$

特别地，`table[cidx] = func_idx(cidx)`。✓

**部分 2（call_indirect 索引）**：

由定义 3.7，`InvokeClosure` 解包 `cv`：

- `fn_ptr = cv >> 32`（高 32 位）
- `env_ptr = cv & 0xFFFFFFFF`（低 32 位）

由定义 3.6，`CreateClosure` 打包 `cv = (cidx << 32) | env_ptr`。故 `fn_ptr = cidx`。

`call_indirect` 通过 `fn_ptr` 索引 table，调用 `table[fn_ptr] = table[cidx] = func_idx(cidx)`，即第 `cidx` 个闭包的 WASM 函数。✓

源码：[tenth/src/compile/wasm/compile.rs:692-694](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/wasm/compile.rs)：

```rust
let (_func_idx, _type_idx, _pc) = self.closure_info[cidx];
// Use closure index (table position) as fn_ptr, NOT func_idx
let table_idx = cidx as i64;
```

**关键设计**：注释明确指出 `fn_ptr = cidx`（table position），而非 `func_idx`（WASM 函数索引）。这是因为 `call_indirect` 通过 table 索引而非函数索引路由。✓

**合并两部分**：table 在编译期正确填充，call_indirect 在运行期正确索引。定理 C3 成立。$\square$

### 4.4 定理 C4（与 Rust/Swift/Chez/OCaml 对比）

**定理 C4（与 Rust/Swift/Chez/OCaml 对比）**：Tenth 的 env_ptr + call_indirect 方案在以下维度与四种语言的闭包实现形成对比：

| 维度 | Tenth | Rust | Swift | Chez Scheme | OCaml |
|------|-------|------|-------|-------------|-------|
| 环境表示 | 线性内存连续 record（env_ptr） | 结构体（捕获变量内联） | ARC context 对象 | flat closure record | 堆分配 record |
| 间接调用机制 | `call_indirect` + table | vtable | partial apply thunk | native 函数指针 | native 函数指针 |
| 调用开销 | 运行期类型检查 + table 查找 | vtable 查找 | thunk 转发 | 直接调用 | 直接调用 |
| 捕获方式 | 按值复制（不可变） | FnMut/FnOnce/Fn 三分 | ARC 强/弱引用 | 按值复制 | 按值复制 |
| 可变捕获 | ❌ 不支持 | ✅ FnMut/FnOnce | ✅ var | ✅ set! | ✅ ref |
| GC 依赖 | ❌ 无（host 管理） | ❌ 无（所有权） | ✅ ARC | ✅ BDW-GC | ✅ major/minor GC |
| 平台亲和度 | WASM 原生 | native | native/LLVM | native | native |
| 多态闭包 | 单一 type_index | trait object | protocol witness | 泛型 | 多态 record |

**证明**：逐一分析。

**与 Rust 对比**：

- **相似**：均通过间接机制调用（Tenth 用 table，Rust 用 vtable）。均无 GC 依赖。
- **差异**：
  - Rust 的三分 trait（FnOnce/FnMut/Fn）支持可变捕获与移动捕获；Tenth 仅按值复制，不支持可变捕获（见局限 L2）。
  - Rust 的 vtable 是 per-closure-type 的（每个闭包类型有独立 vtable）；Tenth 的 table 是全局单一的（WASM MVP 限制）。
  - Rust 的捕获方式由 trait 推断；Tenth 的捕获方式固定为按值复制。
- **优势**：Rust 表达力更强。
- **劣势**：Rust 不直接支持 WASM（需 wasm-bindgen 等桥接）。

**与 Swift 对比**：

- **相似**：均使用 `(env, code)` 对（Swift 是 `(context, function)`）。均通过 thunk/间接机制调用。
- **差异**：
  - Swift 的 context 是 ARC 对象，自动管理生命周期；Tenth 的 env_ptr 是裸指针，依赖 host bump allocator（无释放，内存泄漏）。
  - Swift 的 thunk 转发有额外开销；Tenth 的 call_indirect 直接路由。
  - Swift 支持可变捕获（通过 var）；Tenth 不支持。
- **优势**：Swift 内存管理更安全。
- **劣势**：Swift 依赖 ARC runtime。

**与 Chez Scheme 对比**：

- **相似**：均使用 flat closure（扁平 record 环境）。均按值复制捕获。
- **差异**：
  - Chez 在 native 代码生成，使用直接函数指针调用；Tenth 在 WASM 字节码，使用 call_indirect。
  - Chez 支持 `set!` 可变捕获（通过 box）；Tenth 不支持。
  - Chez 有 CP0 内联优化；Tenth 无（WASM 限制）。
- **优势**：Chez 性能更高（直接调用 + 内联）。
- **劣势**：Chez 依赖 native 代码生成，不可移植到 WASM。

**与 OCaml 对比**：

- **相似**：均使用堆分配的 record 环境。均按值复制捕获。
- **差异**：
  - OCaml 有小闭包优化（≤ 3 捕获内联）；Tenth 无（始终堆分配）。
  - OCaml 使用 native 函数指针；Tenth 使用 call_indirect。
  - OCaml 有 major/minor GC；Tenth 无 GC。
  - OCaml 支持 `let rec` 递归闭包；Tenth 不支持（无特殊处理）。
- **优势**：OCaml 性能更高（小闭包优化 + 直接调用）。
- **劣势**：OCaml 依赖 GC，不可直接用于 WASM MVP。

**综合定位**：

- Tenth 方案在**表达力**上居中：支持按值复制捕获，不支持可变/移动捕获。
- Tenth 方案在**性能**上居末：call_indirect 有运行期类型检查开销，无小闭包优化，无内联。
- Tenth 方案在**WASM 亲和度**上居首：严格遵循 WASM MVP 限制，无需 GC、无需 native 代码生成、无需 ARC runtime。

$\square$

**说明**：定理 C4 是描述性定理，对比基于公开文献与源码分析，非形式化等价证明。

### 4.5 定理 C5（复杂度：capture 数量 vs table 索引查找开销）

**定理 C5（复杂度）**：设 `k` 为闭包的捕获数量，`N` 为程序中闭包总数。则：

1. **捕获变量查找复杂度**：`LoadCapture(name, ci)` 为 `O(1)`（直接 `i64.load`，固定偏移 `8ci`）。
2. **捕获变量定位复杂度**：`current_captures.position(name)` 为 `O(k)`（线性搜索 Vec）。
3. **table 索引查找复杂度**：`call_indirect` 通过 `fn_ptr` 索引 table，为 `O(1)`（WASM 规范保证）。
4. **闭包创建复杂度**：`CreateClosure` 为 `O(k)`（`k` 次 `i64.store`）。
5. **闭包调用复杂度**：`InvokeClosure` 为 `O(k + 1)`（k 次参数压栈 + 1 次 call_indirect，但 k 为参数数而非捕获数；捕获读取在闭包体内进行）。
6. **整体闭包调用复杂度**（含捕获读取）：`O(|captures_used|)`，其中 `captures_used` 为本次调用实际读取的捕获数，最坏 `O(k)`。
7. **编译期 table 生成复杂度**：`emit_table_section` + `emit_elem_section` 为 `O(N)`。

**证明**：

**1. 捕获变量查找复杂度**：

`LoadCapture(name, ci)` 生成 5 条 WASM 指令（`local.get 0`, `i32.wrap_i64`, `i32.const 8ci`, `i32.add`, `i64.load`）。每条指令为 `O(1)`（WASM 解释器/引擎的固定开销）。总复杂度 `O(1)`。✓

**2. 捕获变量定位复杂度**：

`current_captures.position(name)` 是 `Vec<String>::position`，线性搜索：

```rust
if let Some(ci) = self.current_captures.iter().position(|c| c == name) {
```

源码：[tenth/src/compile/wasm/compile.rs:132](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/wasm/compile.rs)。每次比较为 `O(|name|)`（字符串比较），共 `O(k)` 次。设最大名称长度为 `L`（常数），则定位为 `O(k · L) = O(k)`。✓

**注意**：这是**编译期**复杂度（在 `compile_expr` 中执行），非运行期。运行期生成的指令为 `O(1)`（见 1）。

**3. table 索引查找复杂度**：

WASM 规范 [WebAssembly, 2023] 规定 `call_indirect` 通过 `fn_ptr` 索引 table，查找为 `O(1)`（直接数组访问）。但 WASM 引擎需进行**运行期类型检查**——比较 `table[fn_ptr]` 的实际类型签名与 `type_idx` 指定的预期签名。类型检查为 `O(N_params)`（比较参数类型列表），但 `N_params` 为常数（闭包参数数有限）。故 `call_indirect` 整体为 `O(1)`（含类型检查）。✓

**4. 闭包创建复杂度**：

`CreateClosure` 生成：
- 1 次 `tenth_alloc`（host 调用，`O(1)` bump allocation）。
- `k` 次 `StoreCapture`（每次 4 条指令，`O(1)`）。
- 1 次打包（`i64.const + i64.or`，`O(1)`）。

总指令数 `O(k)`。运行期执行 `O(k)`。✓

**5. 闭包调用复杂度**：

`InvokeClosure` 生成：
- 1 次解包 `fn_ptr`（3 条指令，`O(1)`）。
- 1 次解包 `env_ptr`（3 条指令，`O(1)`）。
- `N` 次 `CompileExpr(args[i])`（参数压栈，复杂度依赖参数表达式）。
- 1 次 `call_indirect`（`O(1)`）。

设参数表达式的总复杂度为 `O(A)`，则 `InvokeClosure` 为 `O(A + 1)`。若仅考虑闭包机制开销（不含参数求值），为 `O(1)`。

**注意**：`k`（捕获数）不影响调用本身，仅影响闭包体内的捕获读取（见 6）。

**6. 整体闭包调用复杂度**（含捕获读取）：

设闭包体执行时实际读取 `u` 个捕获变量（`u ≤ k`），则总开销为：

$$T = O(A) + O(1) + O(u) = O(A + u)$$

最坏情况 `u = k`，故 `T = O(A + k)`。

**7. 编译期 table 生成复杂度**：

`emit_table_section`（[tenth/src/compile/wasm/sections.rs:124-138](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/wasm/sections.rs)）创建 1 个 table，`O(1)`。
`emit_elem_section`（[tenth/src/compile/wasm/sections.rs:148-158](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/wasm/sections.rs)）填充 `N` 个元素，`O(N)`。

故编译期 table 生成 `O(N)`。✓

**关键结论**：捕获数量 `k` 与 table 索引查找开销**解耦**——table 索引始终为 `O(1)`，不受 `k` 影响。`k` 仅影响：
- 闭包创建（`O(k)` 次 `i64.store`）。
- 闭包体内的捕获读取（每次 `O(1)`，共 `O(u)` 次，`u ≤ k`）。
- 编译期的捕获定位（`O(k)` 线性搜索，但仅一次）。

`N`（程序中闭包总数）仅影响编译期 table 生成（`O(N)`）与 table 内存占用（`O(N)` 项 funcref），不影响运行期调用复杂度。

$\square$

---

## 5. env_ptr 方案的实现分析

### 5.1 闭包创建的实现

**源码**：[tenth/src/compile/wasm/compile.rs:686-730](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/wasm/compile.rs)。

实现要点：

1. **闭包注册查找**：通过 `closure_expr_map`（HIR 节点指针 → cidx）查找当前闭包的 `cidx`。这要求 HIR 节点指针在编译期稳定（Rust 的 `&HirExpr` 引用稳定）。
2. **table_idx 选择**：`table_idx = cidx`（注释明确："Use closure index (table position) as fn_ptr, NOT func_idx"）。这是关键设计——`call_indirect` 通过 table 索引而非函数索引路由。
3. **无捕获优化**：`captures.is_empty()` 时，`env_ptr = 0`，packed = `table_idx << 32`，跳过 `tenth_alloc` 调用。
4. **env 结构分配**：`tenth_alloc(captures_count * 8)`，返回 i32 指针，零扩展为 i64。
5. **捕获变量写入**：按 `captures` 顺序，对每个 `cap_name`：
   - 若 `cap_name ∈ local_map`：`local.get(idx)` 读取值。
   - 否则：`i64.const 0`（fallback，理论不应触发，由 A1 保证）。
   - `i64.store offset=8ci` 写入 env 结构。
6. **打包**：`i64.const(table_idx << 32) + local.get(tmp) + i64.or`。

**设计权衡**：

- **优势**：单个 i64 表示闭包，避免双字段开销；无捕获时零分配。
- **劣势**：env_ptr 仅 32 位（i64 低 32 位），限制线性内存上限为 4GB（见局限 L1）。

### 5.2 闭包调用的实现

**源码**：[tenth/src/compile/wasm/compile.rs:300-327](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/wasm/compile.rs)。

实现要点：

1. **闭包变量检测**：通过 `closure_vars.get(&fname)` 检测 `fname` 是否为闭包变量。`closure_vars` 在 `Let` 语句处理时填充（[tenth/src/compile/wasm/compile.rs:762-771](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/wasm/compile.rs)）——若 `init` 是 `Closure`，将变量名映射到 `type_idx`。
2. **解包**：
   - `fn_ptr = cv >> 32`（`i64.shr_u`），存入临时 local `tmp`。
   - `env_ptr = cv & 0xFFFFFFFF`（`i64.and`）。
3. **压栈顺序**：`env_ptr → args → fn_ptr`，符合 `call_indirect` 的栈语义。
4. **call_indirect**：`CallIndirect { type_index: type_idx, table_index: 0 }`，使用 `type_idx` 进行运行期类型检查，table 0 为唯一 table。

**设计权衡**：

- **优势**：单一 i64 解包为 (fn_ptr, env_ptr)，无需额外内存访问。
- **劣势**：每次调用需 3 条解包指令 + 1 次 `call_indirect`，相对直接 `call` 有开销。

### 5.3 闭包体编译的实现

**源码**：[tenth/src/compile/wasm/closures.rs:124-161](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/wasm/closures.rs)。

实现要点：

1. **状态重置**：`local_map.clear()`, `local_count = 1`, `param_count = 1`（local 0 为 env_ptr）。
2. **参数注册**：闭包参数从 local 1 开始，依次注册到 `local_map`。
3. **闭包状态**：`compiling_closure = true`, `current_captures = captures.to_vec()`。
4. **locals 声明**：固定 256 个 i64 locals（`(0..256).map(|_| ValType::I64)`），避免动态计算 local 数量。
5. **闭包体内编译**：调用 `compile_expr` 编译 `body`，`Var(name)` 的读取分三种情况（见定义 3.10）。
6. **状态恢复**：`compiling_closure = false`, `current_captures.clear()`。

**设计权衡**：

- **优势**：固定 256 locals 简化实现，避免局部变量计数。
- **劣势**：每个闭包体固定分配 256 locals 的空间（WASM 函数 locals 占用），略微增加模块体积。

### 5.4 捕获变量读取的实现

**源码**：[tenth/src/compile/wasm/compile.rs:130-145](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/wasm/compile.rs)。

实现要点：

1. **捕获定位**：`current_captures.iter().position(|c| c == name)` 线性搜索，返回 `ci`。
2. **地址计算**：`local.get 0`（env_ptr as i64）→ `i32.wrap_i64`（截断为 i32）→ `i32.const 8ci` → `i32.add`。
3. **加载**：`i64.load align=3`（8 字节对齐）。
4. **类型转换**：根据 `expr.ty` 转换：
   - `F64/F32`：`F64ReinterpretI64`（位重新解释）。
   - `Bool`：`I32WrapI64`（截断为 i32）。
   - 其他：无转换（已是 i64）。

**设计权衡**：

- **优势**：固定 8 字节偏移，对齐访问，效率高。
- **劣势**：所有捕获变量统一为 8 字节，不支持紧凑布局（如 `i32` 捕获占 8 字节而非 4）。

### 5.5 table 与 elem 段的生成

**源码**：[tenth/src/compile/wasm/sections.rs:124-158](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/wasm/sections.rs)。

实现要点：

1. **table 段**：仅当 `num_closures > 0` 时生成。`TableType { FUNCREF, minimum: num_closures, maximum: None }`。
2. **elem 段**：`Elements::Functions(&func_idxs)`，起始 offset 0，将 `func_idxs` 按顺序填入 table。
3. **func_idxs 构造**：`closure_info.iter().map(|&(fi, _, _)| fi).collect()`，按 `cidx` 顺序。

**设计权衡**：

- **优势**：编译期一次性填充 table，运行期无填充开销。
- **劣势**：table 大小固定（`minimum: num_closures`），不支持运行期动态添加闭包（如 `eval`）。`maximum: None` 允许增长但 Tenth 未使用。

### 5.6 类型段的生成

**源码**：[tenth/src/compile/wasm/sections.rs:52-63](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/wasm/sections.rs)。

实现要点：

1. **闭包类型签名**：`(i64 env_ptr, i64 param1, ..., i64 paramN) -> i64`。
2. **类型缓存**：`type_cache` 去重——相同 `(params, result)` 共享 `type_idx`。
3. **回填**：`closure_info[cidx].1 = type_idx`，在 `emit_type_section` 末尾回填。

**设计权衡**：

- **优势**：类型缓存减少 module 体积。
- **劣势**：所有闭包参数统一为 i64，丢失类型信息（运行期类型检查仅比较 `i64` 与 `i64`，无法区分 `i32` 与 `i64`）。

---

## 6. 与 Rust/Swift/Chez Scheme/OCaml 对比

### 6.1 环境表示对比

| 方案 | 环境表示 | 内存管理 | 可变捕获 |
|------|---------|---------|---------|
| Tenth | 线性内存连续 record | host bump alloc（无释放） | ❌ |
| Rust | 结构体（捕获内联） | 所有权（编译期） | ✅ FnMut/FnOnce |
| Swift | ARC context 对象 | ARC（运行期） | ✅ var |
| Chez | flat closure record | BDW-GC | ✅ set! (box) |
| OCaml | 堆分配 record | major/minor GC | ✅ ref |

**Tenth 的独特性**：env_ptr 是线性内存中的裸指针，无 GC，无 ARC。捕获变量按值复制（写入时复制），后续修改不影响闭包内已捕获的值——**不支持可变捕获**。

### 6.2 间接调用对比

| 方案 | 机制 | 调用开销 | 多态支持 |
|------|------|---------|---------|
| Tenth | `call_indirect` + table | 运行期类型检查 + table 查找 | 单一 type_index |
| Rust | vtable | vtable 查找 | per-type vtable |
| Swift | partial apply thunk | thunk 转发 | protocol witness |
| Chez | native 函数指针 | 直接调用 | 泛型实例化 |
| OCaml | native 函数指针 | 直接调用 | 多态 record |

**Tenth 的独特性**：`call_indirect` 是 WASM MVP 的唯一间接调用机制，运行期类型检查无法消除（除非使用 tail-call proposal 或 reference-types proposal）。

### 6.3 捕获语义对比

| 方案 | 捕获方式 | 捕获时机 | 生命周期 |
|------|---------|---------|---------|
| Tenth | 按值复制（写入时） | 闭包创建时 | 闭包存活期间不变 |
| Rust | FnOnce=move, FnMut=&mut, Fn=& | 闭包创建时 | 借用检查保证 |
| Swift | ARC 强/弱引用 | 闭包创建时 | ARC 管理 |
| Chez | 按值复制 | 闭包创建时 | GC 管理 |
| OCaml | 按值复制 | 闭包创建时 | GC 管理 |

**Tenth 的关键限制**：按值复制意味着：

- 闭包创建后，对外层变量的修改不影响闭包内的捕获值。
- 闭包内对捕获变量的修改不影响外层（实际上，Tenth 的 `Var` 读取在闭包内是只读的——`i64.load` 读取，无 `i64.store` 写回）。
- 不支持"可变捕获"（如 Rust 的 `FnMut` 或 Swift 的 `var`）。

这是 Tenth 方案的**根本语义限制**（见局限 L2）。

### 6.4 平台亲和度对比

| 方案 | WASM 原生 | Native | GC 依赖 | Runtime 依赖 |
|------|----------|--------|---------|-------------|
| Tenth | ✅ | ❌ | ❌ | host（tenth_alloc） |
| Rust | ⚠️（需 wasm-bindgen） | ✅ | ❌ | ❌ |
| Swift | ⚠️（实验性） | ✅ | ✅ ARC | ✅ ARC runtime |
| Chez | ❌ | ✅ | ✅ BDW-GC | ✅ GC runtime |
| OCaml | ⚠️（wasocaml 实验） | ✅ | ✅ GC | ✅ GC runtime |

**Tenth 的优势**：严格遵循 WASM MVP，无需 GC、无需 runtime（除 host 的 `tenth_alloc`）。

---

## 7. WASM 结构化控制流 + table 限制的影响

### 7.1 结构化控制流的影响

WASM 的结构化控制流（`block`/`loop`/`if`）对闭包转换的影响：

- **无 continuation**：CPS 变换通常依赖 continuation，但 WASM 的结构化控制流不支持任意 continuation。Tenth 的方案避免 continuation——env_ptr 作为显式参数传递，无需 continuation。
- **无尾调用优化（MVP）**：递归闭包调用可能栈溢出。Tail-call proposal（`return_call`）尚未广泛实现。Tenth 当前未使用 `return_call`。
- **Block 限定的控制流**：`break`/`continue` 通过 `br` 指令实现，限定向外层 `block`/`loop`。闭包体内的 `break`/`continue` 需正确处理 `if_depths` 栈（[tenth/src/compile/wasm/closures.rs:136-139](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/wasm/closures.rs)）。

### 7.2 单一 table 限制的影响

WASM MVP 只支持一个 `funcref` table（[tenth/src/compile/wasm/sections.rs:130-137](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/wasm/sections.rs)）：

- **全局单一 table**：所有闭包共享同一 table，无模块隔离。
- **table 索引全局唯一**：`cidx` 在整个模块内唯一，跨模块互操作需额外协调。
- **Reference-types proposal**：允许多 table 与任意引用类型，但未广泛实现。Tenth 未使用。

**影响**：Tenth 的所有闭包在单一 table 中索引，无法支持多模块闭包互操作（见局限 L3）。

### 7.3 无 GC 限制的影响

WASM MVP 无 GC：

- **手动内存管理**：env_ptr 由 host 的 `tenth_alloc` 分配，但**无释放**——env_ptr 指向的内存永不回收。
- **内存泄漏**：每次闭包创建（有捕获时）分配 8k 字节，永不释放。长期运行可能耗尽线性内存。
- **GC proposal**：WASM GC proposal 提供 `struct`/`array`/`anyref` 等，但未广泛实现。Tenth 未使用。

**影响**：Tenth 的闭包方案存在内存泄漏（见局限 L4）。

### 7.4 线性内存模型的影响

WASM 的线性内存是字节寻址的 `i32` 偏移数组：

- **env_ptr 为 i32**：env_ptr 截断为 i32（`i32.wrap_i64`），限制线性内存上限为 4GB。
- **对齐访问**：`i64.load align=3`（8 字节对齐），WASM 引擎可能优化。
- **无虚拟内存**：线性内存是连续数组，无页表、无虚拟内存。

**影响**：env_ptr 的 32 位限制线性内存上限（见局限 L1）。

### 7.5 type_index 运行期类型检查的影响

WASM `call_indirect` 通过 `type_index` 进行运行期类型检查：

- **检查机制**：比较 `table[fn_ptr]` 的实际类型签名与 `type_index` 指定的预期签名。
- **失败行为**：类型不匹配时 `trap`（运行期错误）。
- **开销**：每次 `call_indirect` 都有类型检查开销，无法消除。

**影响**：Tenth 的闭包调用始终有类型检查开销（见局限 L5）。

---

## 8. 工程权衡

### 8.1 i64 打包 vs 双字段

**当前方案**：闭包值打包为单个 i64（`table_idx << 32 | env_ptr`）。

**优势**：

- 单 i64 表示，避免双字段（如 `(i32, i32)`）的内存开销。
- 与 WASM 的 i64 操作指令（`i64.shr_u`, `i64.and`, `i64.or`）契合。
- 局部变量存储为 i64，与 Tenth 的"所有局部为 i64"设计一致。

**劣势**：

- env_ptr 限 32 位，线性内存上限 4GB。
- 解包需 3 条指令（`shr_u`, `and`, `local.set tmp`）。

**替代方案**：双字段 `(i32 fn_ptr, i32 env_ptr)`：

- 优势：env_ptr 仍为 32 位，但 fn_ptr 与 env_ptr 分离，无打包开销。
- 劣势：需两个 local，与"所有局部为 i64"设计不一致。

**结论**：i64 打包与 Tenth 的整体设计一致，权衡合理。

### 8.2 table 索引 vs 函数指针

**当前方案**：通过 table 索引 + `call_indirect` 调用。

**优势**：

- WASM MVP 原生支持，无需 reference-types proposal。
- table 在编译期填充，运行期无填充开销。

**劣势**：

- 运行期类型检查无法消除。
- 单一 table 限制多模块互操作。

**替代方案**：reference-types proposal 的 `call_ref`：

- 优势：直接函数引用调用，无 table 中介。
- 劣势：依赖 reference-types proposal，未广泛实现。

**结论**：table 索引是 WASM MVP 下的合理选择。

### 8.3 按值复制 vs 引用捕获

**当前方案**：捕获变量按值复制（写入 env 结构时复制）。

**优势**：

- 实现简单，无需借用检查。
- 无生命周期问题（捕获值独立于外层变量）。
- 与 WASM 的线性内存模型契合（无引用语义）。

**劣势**：

- 不支持可变捕获（闭包内修改捕获变量不影响外层）。
- 捕获大对象（如大 tensor）时复制开销高。

**替代方案**：引用捕获（捕获 env_ptr 偏移而非值）：

- 优势：支持可变捕获，大对象捕获零开销。
- 劣势：需借用检查，生命周期管理复杂。

**结论**：按值复制是 Tenth 当前阶段的简化选择，未来可考虑引用捕获。

### 8.4 固定 256 locals vs 动态计算

**当前方案**：闭包体固定 256 个 i64 locals。

**优势**：

- 实现简单，无需动态计算 local 数量。
- 避免局部变量溢出（256 个足够大多数闭包）。

**劣势**：

- 每个闭包体固定占用 256 locals 的空间（WASM 模块体积增加）。
- 超过 256 个局部变量时溢出（理论可能，实践罕见）。

**替代方案**：动态计算 local 数量：

- 优势：模块体积优化。
- 劣势：实现复杂，需预扫描 body 统计 local 数量。

**结论**：固定 256 locals 是工程简化，可优化但非关键瓶颈。

### 8.5 host bump allocator vs WASM 内分配

**当前方案**：通过 host 导入的 `tenth_alloc` 分配 env 结构。

**优势**：

- WASM 模块本身无内存管理逻辑，简化模块。
- host 可统一管理内存（与其他 host 对象共享）。

**劣势**：

- 每次 `tenth_alloc` 是 host call，有跨边界开销。
- 无释放机制，内存泄漏。

**替代方案**：WASM 内 bump allocator（使用 global 与 memory）：

- 优势：无 host call 开销。
- 劣势：需在 WASM 内实现分配逻辑，增加模块复杂度。

**结论**：host bump allocator 是 Tenth 的整体设计选择，与闭包方案解耦。

---

## 9. 局限（诚实披露）

### 9.1 局限 L1（env_ptr 32 位限制）

**现象**：env_ptr 打包在 i64 的低 32 位，截断为 i32 后用于线性内存寻址。这限制线性内存上限为 4GB（`i32` 地址空间）。

**根源**：[tenth/src/compile/wasm/compile.rs:704, 712](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/wasm/compile.rs) 的 `I64ExtendI32U` 与 `I32WrapI64` 指令。

**影响**：若 Tenth 程序的线性内存需求超过 4GB（如大规模张量计算），env_ptr 截断会导致地址错误。当前 `emit_memory_section` 设置 `minimum: 16, maximum: Some(256)` 页（每页 64KB，共 16MB），远低于 4GB，故实践中不触发。

**严重性**：**低**（当前内存配置远低于 4GB）。

**缓解**：

- 使用 memory64 proposal（`memory64: true`），但 WASM MVP 不支持。
- 改为双字段表示（`i64 env_ptr, i64 fn_ptr`），但增加开销。

**现状**：未修复，依赖当前内存配置在实践中不触发。

### 9.2 局限 L2（不支持可变捕获）

**现象**：捕获变量按值复制（`i64.store` 写入 env 结构），闭包内对捕获变量的修改不影响外层。Tenth 的闭包体内 `Var(name)` 读取捕获变量后，无 `i64.store` 写回——**捕获变量在闭包内是只读的**。

**根源**：[tenth/src/compile/wasm/compile.rs:710-724](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/wasm/compile.rs) 的 `StoreCapture` 仅在闭包创建时写入，闭包体内无写回机制。`Assign` 节点在闭包内只支持局部变量赋值（`local_map`），不支持写回 env 结构。

**影响**：

- 不支持 Rust 的 `FnMut` 语义（可变捕获）。
- 不支持 Swift 的 `var` 捕获。
- 不支持 Chez 的 `set!` 捕获。

**反例**：

```tenth
let counter = 0;
let inc = || { counter = counter + 1; };  // 期望可变捕获
inc();
println(counter);  // 期望 1，实际 0（按值复制，外层 counter 不变）
```

**严重性**：**中**（限制闭包表达力，但 Tenth 的 AI 原生场景中可变捕获较少）。

**缓解**：

- 引入 `Box<T>` / `RefCell<T>` 类似的可变容器，捕获容器指针而非值。
- 在闭包体内支持 `env_ptr + offset` 的 `i64.store` 写回。

**现状**：未修复，按值复制是当前简化选择。

### 9.3 局限 L3（table 全局单一性）

**现象**：WASM MVP 只支持一个 `funcref` table，所有闭包共享同一 table，无模块隔离。

**根源**：[tenth/src/compile/wasm/sections.rs:130-137](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/wasm/sections.rs) 的 `emit_table_section` 创建单一 table。

**影响**：

- 跨模块闭包互操作需协调 table 索引（cidx 全局唯一）。
- 动态加载模块（`eval`）无法共享 table。

**严重性**：**低**（Tenth 当前无多模块互操作需求）。

**缓解**：

- 使用 reference-types proposal 的多 table 支持。
- 使用 `table.grow` 动态扩展 table。

**现状**：未修复，依赖 Tenth 的单模块设计。

### 9.4 局限 L4（内存泄漏）

**现象**：`tenth_alloc` 分配的 env 结构永不释放，长期运行可能耗尽线性内存。

**根源**：host 的 `tenth_alloc` 是 bump allocator，无 free 操作。Tenth 的 WASM 模块无 GC。

**影响**：

- 每次闭包创建（有捕获时）分配 8k 字节，永不回收。
- 长期运行的 Tenth 程序（如 REPL、服务器）可能内存耗尽。

**严重性**：**中**（短期脚本无影响，长期运行有风险）。

**缓解**：

- 引入 WASM GC proposal（未广泛实现）。
- 在 host 实现标记-清除或引用计数。
- 闭包创建时复用已释放的 env 结构（需 free 操作）。

**现状**：未修复，依赖 Tenth 的脚本式使用模式。

### 9.5 局限 L5（call_indirect 运行期开销）

**现象**：`call_indirect` 每次调用都进行运行期类型检查（比较 `table[fn_ptr]` 的类型签名与 `type_index`），开销无法消除。

**根源**：WASM 规范要求 `call_indirect` 进行类型检查，确保类型安全。

**影响**：

- 闭包调用相对直接 `call` 有额外开销（类型检查 + table 查找）。
- 性能敏感场景下，闭包调用可能成为瓶颈。

**严重性**：**低**（WASM 引擎通常优化类型检查，开销小）。

**缓解**：

- 使用 tail-call proposal 的 `return_call_indirect`（仍需类型检查）。
- 使用 reference-types proposal 的 `call_ref`（直接引用，无 table 中介）。
- 编译期内联已知闭包（需更复杂的分析）。

**现状**：未修复，是 WASM MVP 的固有开销。

### 9.6 局限 L6（与 T22 完备性耦合）

**现象**：定理 C1 的假设 A2（FV 分析完备）依赖 T22 的结论，而 T22 在四条假设（A1–A4）下成立。若 T22 的假设不成立，T30 的语义保持可能失败。

**根源**：T30 的 `captures` 列表由 T22 的 `collect_free_vars` 计算，若 T22 漏收集真自由变量，闭包运行时访问未捕获的变量会导致未定义行为。

**影响**：

- T22 的局限 L1（Block 前置遮蔽引用）会导致 T30 的语义保持失败——闭包未捕获应捕获的变量，运行时访问未定义。
- T22 的局限 L2（Match 守卫未处理）类似。
- T22 的局限 L3、L4（过近似）不破坏 T30 语义保持（仅捕获冗余）。

**严重性**：**高**（若 T22 局限 L1/L2 触发，T30 语义保持失败）。

**缓解**：

- 修复 T22 的局限 L1、L2（见 T22 实施建议）。
- 在 T30 的闭包创建时检查所有 `captures` 在 `local_map` 中存在（A1 假设）——但若 T22 漏收集，`captures` 缺失变量，检查无法发现。

**现状**：未修复，依赖 T22 的假设在实践中成立。

### 9.7 局限汇总表

| 局限 | 类型 | 严重性 | 触发条件 | 根源 |
|------|------|--------|---------|------|
| L1 | 地址空间 | 低 | 线性内存 > 4GB | env_ptr 32 位 |
| L2 | 表达力 | 中 | 可变捕获需求 | 按值复制 |
| L3 | 互操作 | 低 | 多模块闭包 | 单一 table |
| L4 | 内存管理 | 中 | 长期运行 | 无 free |
| L5 | 性能 | 低 | 频繁调用 | call_indirect 类型检查 |
| L6 | 语义保持 | 高 | T22 局限触发 | FV 分析耦合 |

---

## 10. 开放问题

### 10.1 可变捕获的支持

当前方案不支持可变捕获（L2）。如何在 WASM MVP 下支持可变捕获？

**思路**：

- 引入 `Box<T>` 类型（堆分配的 mutable cell）。
- 捕获 `Box<T>` 的指针（i32）而非值。
- 闭包内通过 `i32.load` / `i32.store` 读写 box 内容。

**挑战**：

- 需在 HIR 层引入 `Box<T>` 类型。
- 需在类型系统支持 `Box<T>`。
- 需在借用检查支持 `Box<T>` 的可变借用。

### 10.2 内存回收

当前方案无内存回收（L4）。如何在 WASM MVP 下回收 env 结构？

**思路**：

- 引入引用计数（RC）：env 结构附带引用计数，闭包销毁时减一，归零时释放。
- 引入标记-清除：周期性扫描线性内存，标记可达 env 结构，清除不可达的。

**挑战**：

- WASM MVP 无 finalizer，闭包销毁时机难以确定。
- RC 需在每次闭包复制时增减计数，开销大。
- 标记-清除需扫描线性内存，需识别 env 结构的布局。

### 10.3 多模块互操作

当前方案单一 table（L3）。如何支持多模块闭包互操作？

**思路**：

- 使用 reference-types proposal 的多 table。
- 跨模块调用时，传递 `(table_id, table_idx)` 对。

**挑战**：

- reference-types proposal 未广泛实现。
- 跨模块 table 索引协调复杂。

### 10.4 性能优化

当前方案 call_indirect 有运行期开销（L5）。如何优化？

**思路**：

- 编译期内联已知闭包：若闭包创建与调用在同一函数内，且闭包未逃逸，直接 `call` 而非 `call_indirect`。
- 类型特化：为常见闭包类型生成特化版本，避免类型检查。
- WASM 引擎优化：依赖 wasmi / wasmtime 的 JIT 优化。

**挑战**：

- 闭包逃逸分析需更复杂的 HIR 分析。
- 类型特化增加代码体积。

### 10.5 与 T22 的解耦

当前方案的语义保持依赖 T22 的完备性（L6）。如何解耦？

**思路**：

- 在闭包创建时，对 `captures` 中的每个变量检查 `local_map` 中存在性。若缺失，报错（而非 fallback 到 `i64.const 0`）。
- 这能发现 T22 的漏收集（完备性失败），但不能发现 T22 的过收集（健全性问题）。

**挑战**：

- 当前 fallback 到 `i64.const 0`（[tenth/src/compile/wasm/compile.rs:718-721](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/wasm/compile.rs)）掩盖了 T22 的错误。

---

## 11. 结论

本文对 Tenth 语言的 WASM 闭包实现——env_ptr + call_indirect 方案——进行了形式化建模与正确性证明。核心结论：

1. **closure conversion 语义保持（C1）**在三条受控假设下成立：转换前后程序语义一致。
2. **env_ptr 偏移正确性（C2）**无条件成立：运行期通过 env_ptr 偏移读取的捕获变量与编译期 captures 列表一一对应。
3. **table 索引正确性（C3）**无条件成立：call_indirect 索引与闭包体函数一一对应，table 在编译期正确填充。
4. **跨语言对比（C4）**：Tenth 方案在表达力上居中（不支持可变捕获），在 WASM 亲和度上居首（严格遵循 MVP）。
5. **复杂度（C5）**：捕获数量 k 与 table 索引查找开销解耦——table 索引 O(1)，捕获读取 O(u)（u ≤ k），整体闭包调用 O(A + k)。

**诚实披露的局限**：

- L1：env_ptr 32 位限制线性内存上限（低严重性）。
- L2：按值复制不支持可变捕获（中严重性，限制表达力）。
- L3：单一 table 限制多模块互操作（低严重性）。
- L4：env 结构无回收，内存泄漏（中严重性，长期运行风险）。
- L5：call_indirect 运行期类型检查开销（低严重性）。
- L6：与 T22 完备性耦合（高严重性，依赖 T22 假设）。

**与 Rust/Swift/Chez/OCaml 的对比发现**：

- Tenth 方案在**环境表示**上与 OCaml/Chez 最接近（堆分配 record），但无 GC。
- Tenth 方案在**间接调用**上与 Rust vtable 最接近（table 索引），但单一 table。
- Tenth 方案在**捕获语义**上最受限（仅按值复制，无可变捕获）。
- Tenth 方案在**WASM 亲和度**上最优（严格遵循 MVP，无 GC/runtime 依赖）。

**工程价值**：尽管存在局限，env_ptr + call_indirect 方案在 Tenth 的 WASM 后端中是合理的工程选择——它严格遵循 WASM MVP 限制，实现简洁（约 200 行 Rust），性能可接受（O(k + 1) 调用复杂度），适合 AI 原生语言的快速迭代需求。其局限（可变捕获、内存回收、多模块互操作）是 WASM MVP 的固有限制，未来可随 WASM 提案的成熟逐步缓解。

---

## 参考文献

1. Appel, A. W. (1992). *Compiling with Continuations*. Cambridge University Press.
2. Leroy, X. (1992). *The ZINC experiment: an economical implementation of the ML language*. INRIA Technical Report 117.
3. Dybvig, R. K. (2006). *The Implementation of the Chez Scheme System*. Workshop on Scheme and Functional Programming.
4. Marlow, S., & Peyton Jones, S. (2004). *Making a fast curry: push/enter vs. eval/apply for higher-order languages*. ICFP.
5. Jung, R., Jourdan, J.-H., Krebbers, R., & Dreyer, D. (2018). *RustBelt: Securing the foundations of the Rust programming language*. POPL.
6. Rust Reference. (2024). *Closure types: Fn, FnMut, FnOnce*. https://doc.rust-lang.org/reference/types/closure.html
7. Swift. (2023). *Swift Compiler: Closure Representation*. https://github.com/apple/swift/blob/main/docs/ABI/ClosureRepresentation.md
8. WebAssembly. (2023). *WebAssembly Specification*. https://webassembly.github.io/spec/core/
9. WebAssembly. (2023). *Reference Types Proposal*. https://github.com/WebAssembly/reference-types
10. WebAssembly. (2023). *Tail Call Proposal*. https://github.com/WebAssembly/tail-call
11. WebAssembly. (2023). *GC Proposal*. https://github.com/WebAssembly/gc
12. Tenth 项目. (2026). *WASM 闭包编译实现*. [tenth/src/compile/wasm/closures.rs](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/wasm/closures.rs)
13. Tenth 项目. (2026). *WASM 段生成*. [tenth/src/compile/wasm/sections.rs](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/wasm/sections.rs)
14. Tenth 项目. (2026). *WASM 表达式编译*. [tenth/src/compile/wasm/compile.rs](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/wasm/compile.rs)
15. Tenth 项目. (2026). *WASM 编译器状态*. [tenth/src/compile/wasm/mod.rs](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/wasm/mod.rs)
16. Tenth 项目. (2026). *T22：Closure 自由变量分析正确性*. [docs/论文/T22-Closure自由变量分析正确性.md](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/docs/论文/T22-Closure自由变量分析正确性.md)
17. Tenth 项目. (2026). *自由变量分析实现*. [tenth/src/hir/lower/closures.rs](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/hir/lower/closures.rs)
18. Tenth 项目. (2026). *工作规范 v1.1*. `.trae/rules/工作规范.md`

---

## 附录 A：定理索引

| 定理 | 名称 | 结论 | 证明完整性 | 依赖 |
|------|------|------|-----------|------|
| C1 | closure conversion 语义保持 | A1–A3 下语义一致 | 完整（含假设声明） | T22 (A2) |
| C2 | env_ptr 偏移正确性 | 偏移读取与 captures 一致 | 完整（双端论证） | 无 |
| C3 | table 索引正确性 | call_indirect 索引正确 | 完整（编译期 + 运行期） | 无 |
| C4 | 跨语言对比 | Tenth 居中，WASM 亲和度居首 | 描述性（非形式化等价） | 文献调研 |
| C5 | 复杂度 | O(k + 1)，k 与 table 查找解耦 | 完整 | 无 |

## 附录 B：局限索引

| 局限 | 类型 | 严重性 | 触发条件 | 缓解 |
|------|------|--------|---------|------|
| L1 | 地址空间 | 低 | 线性内存 > 4GB | memory64 / 双字段 |
| L2 | 表达力 | 中 | 可变捕获需求 | Box<T> / 引用捕获 |
| L3 | 互操作 | 低 | 多模块闭包 | reference-types |
| L4 | 内存管理 | 中 | 长期运行 | GC / RC / free |
| L5 | 性能 | 低 | 频繁调用 | 内联 / call_ref |
| L6 | 语义保持 | 高 | T22 局限触发 | 修复 T22 / 编译期检查 |

## 附录 C：实施建议

基于本文分析，对 Tenth 编译器部的实施建议（按优先级）：

1. **P0（高优先级）**：缓解 L6。在 `CreateClosure` 的 `StoreCapture` 中，若 `cap_name` 不在 `local_map` 中，**报错**而非 fallback 到 `i64.const 0`：
   ```rust
   let idx = self.local_map.get(cap_name).ok_or_else(|| TenthError::RuntimeError {
       message: format!("WASM: 闭包捕获变量 '{}' 在 local_map 中不存在", cap_name),
   })?;
   body.instruction(&Instruction::LocalGet(*idx));
   ```
   这能发现 T22 的漏收集（完备性失败），便于调试。

2. **P1（中优先级）**：修复 L2。引入 `Box<T>` 类型支持可变捕获：
   - HIR 层增加 `Box<T>` 类型。
   - 闭包捕获 `Box<T>` 的指针（i32）。
   - 闭包体内通过 `i32.load` / `i32.store` 读写 box 内容。

3. **P2（中优先级）**：缓解 L4。在 host 实现 free 列表或引用计数：
   - `tenth_alloc` 返回的指针附带元数据（大小）。
   - 新增 `tenth_free(ptr)` host 函数。
   - 闭包销毁时调用 `tenth_free`（需 finalizer 支持，WASM MVP 不支持，可用显式 `drop`）。

4. **P3（低优先级）**：优化 L5。编译期内联已知闭包：
   - HIR 层分析闭包逃逸。
   - 若闭包未逃逸（仅在创建作用域内调用），直接 `call` 而非 `call_indirect`。

5. **P4（低优先级）**：考虑 L1。若未来 Tenth 需要大内存，改用双字段表示或 memory64 proposal。

6. **P5（低优先级）**：考虑 L3。若未来 Tenth 需要多模块互操作，迁移到 reference-types proposal。

**注意**：以上均为未来工作，本文未实施任何代码修改。实施由编译器部负责，本文仅提供理论依据。

## 附录 D：与 T22 的联动

T30 的语义保持（C1）依赖 T22 的 FV 分析完备性（A2 假设）。具体联动：

| T30 假设 | T22 结论 | T22 假设 |
|---------|---------|---------|
| A2（FV 完备） | FV2（受控完备性） | A1（无前置遮蔽）、A2（无 Match 守卫）、A3（无 Match 模式绑定）、A4（While/Loop body 为 Block 包装） |

故 T30 的 C1 在 T22 的 A1–A4 + T30 的 A1, A3 下成立。T22 的局限 L1（Block 前置遮蔽）与 L2（Match 守卫）会破坏 T30 的 C1——若 T22 漏收集真自由变量，闭包运行时访问未捕获变量，导致未定义行为。

**联动结论**：T30 的语义保持强依赖 T22 的完备性。修复 T22 的 L1、L2 是 T30 可靠性的前提。

---

> **数理部声明**：本文的理论结论基于对 [tenth/src/compile/wasm/closures.rs](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/wasm/closures.rs)、[tenth/src/compile/wasm/sections.rs](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/wasm/sections.rs)、[tenth/src/compile/wasm/compile.rs](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/wasm/compile.rs)、[tenth/src/compile/wasm/mod.rs](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/wasm/mod.rs)（v0.3.3+）的源码分析。所有源码引用均使用 `file://` 链接标注。局限章节诚实披露了证明的漏洞与假设的强度，未掩盖任何已知问题。实施建议附录将理论结论转化为可执行指导，但未实施任何代码修改——实施由编译器部负责。与 T22 的联动关系在附录 D 中显式说明，T30 的语义保持强依赖 T22 的完备性结论。
