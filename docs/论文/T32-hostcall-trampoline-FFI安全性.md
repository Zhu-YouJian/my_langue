# hostcall trampoline 的 FFI 安全性：Tenth JIT 与 Rust 边界的 UB 自由性证明

> **理论分析点**：T32 | **难度**：会议论文级 | **版本**：v1
> **关联**：T31（基于 Cranelift 的栈区设计——无 phi 节点的栈式 JIT）、T9（JIT 特化语义保持）
> **核心源码**：
> - [`tenth/src/compile/jit/hostcalls.rs`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/jit/hostcalls.rs)
> - [`tenth/src/compile/jit/context.rs`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/jit/context.rs)
> - [`tenth/src/compile/jit/mod.rs`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/jit/mod.rs)
> - [`tenth/src/compile/jit/translator.rs`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/jit/translator.rs)

---

## 摘要

Tenth 语言使用 Rust 编写主编译器与运行时，使用 Cranelift 进行 JIT 编译。JIT 产物（机器码）与 Rust 运行时（`Vm`、`Value`）之间通过一组 `extern "C"` 的 **hostcall trampoline** 函数相互调用，构成一个非平凡的 FFI 边界。该边界若设计不当，将立即引入未定义行为（UB）：跨 FFI 的 panic unwind 是 UB；通过 Cranelift 返回槽传递 32+ 字节含 `Rc`/`Vec`/`String` 的 `enum Value` 可能违反 Rust ABI 约定；`from_raw_parts` 在 `count` 溢出时是 UB；`*const u8` 与函数指针尺寸不一致时 `transmute` 是 UB。

本文形式化 Tenth hostcall 协议的 **4 个 FFI 不变量**：(I1) 所有 `Value` 经 `*mut Value` out-pointer 传递；(I2) `catch_unwind` 包裹 JIT 调用阻止 panic 跨 FFI；(I3) `safe_slice` 限制 `from_raw_parts` 的 `count`；(I4) `JitFn` 类型断言 `size_of::<*const u8>() == size_of::<JitFn>()`。我们证明 **主定理 F1**：在 I1–I4 共同成立的前提下，Tenth JIT 与 Rust 边界上的所有交互均 UB 自由；并给出 **定理 F2–F5** 分别对应 panic 隔离、切片溢出防护、函数指针 transmute 安全、以及与 Julia `ccall`/PyPy `rffi` 的对比。最后以独立章节诚实披露形式化模型的局限：模型仅覆盖 I1–I4 已识别的 UB 类，不覆盖恶意 JIT 产物、`vm` 指针被篡改、并发访问等场景。

**关键词**：FFI 安全性；未定义行为；Rust ABI；JIT；Cranelift；panic unwind；out-pointer 调用约定

---

## 1. 引言

### 1.1 FFI 边界的 UB 风险

Rust 的安全性保证在 `unsafe` 边界之外成立，而 FFI（Foreign Function Interface）天然是 `unsafe` 的：跨语言调用涉及两套调用约定（calling convention）、两套对 panic/unwind 的处理方式、两套类型布局假设。Rust Reference 明确规定 [RustRef]：

> 若 unwind 跨越 `extern "C"` 边界（且 `C-unwind` ABI 未使用），行为未定义。

Cranelift 生成的机器码不参与 Rust 类型系统，其返回值通过平台 ABI 的返回槽（return slot）传递；当 Rust 端以 Rust ABI 假设解读 Cranelift 返回值时，任何布局不一致都构成 UB。

### 1.2 Rust + Cranelift 集成的挑战

Tenth 的 JIT 路径（[`compile/jit/mod.rs`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/jit/mod.rs)）使用 Cranelift 把字节码 `Chunk` 翻译为机器码（[§3.1, T31]），编译产物函数指针通过 `transmute` 转换为类型化 `JitFn`（[context.rs:55](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/jit/context.rs)）。JIT 产物反过来通过 `call_indirect` 调用 Rust 端的 hostcall trampoline（[translator.rs:583-660](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/jit/translator.rs)）。这一双向 FFI 边界上的 UB 风险点至少有四类（见 §4 详述）。

### 1.3 贡献

1. **形式化**：将 Tenth hostcall 协议抽象为 4 个 FFI 不变量 I1–I4，每个不变量对应一类潜在 UB（§4）。
2. **证明**：给出主定理 F1——I1–I4 共同蕴涵 UB 自由性；并分定理 F2/F3/F4 各自证明单一不变量阻止的 UB 类（§5）。
3. **对比**：与 Julia `ccall`、PyPy `rffi` 对比，指出 Tenth 的 out-pointer 协议在 Rust ABI 下的独特性（§8）。
4. **诚实局限**：独立 §10 章节披露模型不完备之处——仅覆盖已知 UB 类，不覆盖恶意 JIT、并发、`vm` 篡改。

---

## 2. 背景

### 2.1 Rust FFI 安全规则

Rust 的 FFI 安全性由以下规则构成（[Nomicon]）：

- **(R1) ABI 一致性**：跨 FFI 函数必须使用统一 ABI（`extern "C"` 或 `extern "C-unwind"`）。
- **(R2) 布局一致性**：两侧对同一类型的 `size_of`/`align_of` 必须一致；非 `#[repr(C)]` 的 Rust 类型布局未定义，跨 FFI 传递是 `unsafe`。
- **(R3) Unwind 隔离**：默认 `extern "C"` 不允许 unwind；若 panic 跨越边界是 UB，除非使用 `C-unwind`（Rust 1.51+）。
- **(R4) 生存期与所有权**：通过裸指针传递时，调用方与被调方需对生存期与所有权达成显式约定。

### 2.2 Panic 跨 FFI 的 UB

`std::panic::catch_unwind` 捕获当前线程的 panic payload，将其转为 `Err` 返回。在 `extern "C"` 边界上，Rust 实现将 unwind 实现为平台异常（如 Linux 的 DWARF、Windows 的 SEH）。若 JIT 产物不是 Rust 编译的，其异常表与 Rust unwind 机制不兼容，跨边界 unwind 可能：

- 在 JIT 端触发段错误（未注册 unwind 表）；
- 跳过析构函数导致资源泄漏；
- 在 ABI 不匹配时进入未定义状态。

### 2.3 Julia `ccall` 与 PyPy `rffi`

**Julia `ccall`** [JuliaDoc]：通过 `ccall((symbol, "lib"), RetType, (ArgTypes...), args...)` 调用 C 函数。Julia 自身使用 `libunwind` 实现 GC 安全点；`ccall` 调用方需保证被调函数不会抛 Julia 异常，否则触发 `jlbacktrace` 终止。

**PyPy `rffi`** [PyPyDoc]：通过 `rffi.llexternal(...)` 声明外部 C 函数，PyPy RPython 工具链生成包装代码。PyPy 的 GC 是增量式，跨 FFI 时需显式 `rgc.ll_writebarrier`；panic 由 RPython 转为进程退出。

二者均**不**涉及把含引用计数的 enum 直接跨 FFI 传递——Julia 用 `jl_value_t*` 不透明指针，PyPy 用 `PyObject*`。Tenth 直接传 `*const Value` 是更激进的设计（§8）。

---

## 3. Tenth hostcall 协议概述

### 3.1 调用方向

Tenth JIT 路径有两条 FFI 调用方向：

**方向 A（Rust → JIT）**：[`hostcalls.rs:33-62 invoke_jit`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/jit/hostcalls.rs)。Rust 调用 `JitFn` 类型的函数指针，参数为 `(vm: *mut Vm, args: *const Value, n: usize, out: *mut Value)`。

**方向 B（JIT → Rust）**：[`translator.rs:603-732`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/jit/translator.rs)。JIT 产物通过 `call_indirect` 调用 hostcall trampoline（如 `host_add`、`host_make_vec`），trampoline 是 `unsafe extern "C" fn`（[hostcalls.rs:82-451](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/jit/hostcalls.rs)）。

### 3.2 `Value` 的非平凡布局

[`runtime/value.rs:68-106`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/runtime/value.rs) 定义 `Value` 为含 18 个变体的 `enum`，部分变体包含 `Rc<RefCell<...>>`、`Vec<Value>`、`String`、`HashMap`。其 `size_of::<Value>()` 在 64 位平台上为 32 字节或更多（具体取决于派生 `Debug`/`Clone` 的开销，实测为 32 字节，[`translator.rs:28`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/jit/translator.rs) 的 `VALUE_SIZE` 常量）。该 enum **未标注 `#[repr(C)]`**，故其布局由 Rust 编译器自由选择（tag 位置、padding、变体排序），**跨 FFI 直接传递是 UB**（违反 R2）。

### 3.3 协议要点

按 [`hostcalls.rs:1-13`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/jit/hostcalls.rs) 的注释：

- `vm: *mut Vm` — VM 上下文。
- 输入 `Value` 以 `*const Value` 传递（只读）。
- 输出 `Value` 通过 `*mut Value` 写入。
- 错误：trampoline 设置 `vm.last_error` 并写入 `Value::Unit` 到 out-pointer。
- 所有 trampoline 为 `extern "C"`（"no unwinding across FFI"）。

---

## 4. 形式化：4 个 FFI 不变量

我们定义以下四个不变量。每个不变量是一个**运行时/编译时条件**，由 Tenth hostcall 协议保证。

### 定义 I1（out-pointer 传递）

> 对每个 hostcall trampoline `h` 与每个 JIT 函数 `f`，所有 `Value` 类型的输入与输出参数**均**通过裸指针 `*const Value` / `*mut Value` 传递；**禁止**通过函数返回槽返回 `Value`。

**实现位置**：[`hostcalls.rs:8-13`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/jit/hostcalls.rs)；[`translator.rs:43-47`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/jit/translator.rs)（Cranelift 签名只声明 `bool` 作为返回值，参数全为指针）；`invoke_jit` 返回 `bool` 而非 `Value`（[hostcalls.rs:34, 42](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/jit/hostcalls.rs)）。

### 定义 I2（panic 隔离）

> 对方向 A 的每次 JIT 调用，调用表达式 `f(vm, args, n, out)` 必须被 `catch_unwind` 包裹；捕获的 panic payload 写入 `vm.last_error`、`*out` 写为 `Value::Unit`、返回 `false`。方向 B 中，trampoline 函数为 `extern "C"`，且 Rust 编译器按 `extern "C"` 语义生成代码（无 unwind 表跨越边界）。

**实现位置**：[`hostcalls.rs:41-61`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/jit/hostcalls.rs)；所有 trampoline 标注 `unsafe extern "C"`（如 [`hostcalls.rs:82`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/jit/hostcalls.rs)）。

### 定义 I3（safe_slice 闸门）

> 所有 `from_raw_parts(ptr, count)` 调用必须经 `safe_slice` 或其等价闸门；闸门满足：
> (a) `ptr.is_null()` 时返回空切片；
> (b) `count == 0` 时返回空切片；
> (c) `count > MAX_HOSTCALL_ARGS`（= $2^{20}$）时返回空切片；
> (d) `count` 与 `count * k`（$k \in \{1, 2\}$）的乘积用 `checked_mul` 验证不溢出。

**实现位置**：[`hostcalls.rs:23, 68-78`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/jit/hostcalls.rs)（`safe_slice` 与 `MAX_HOSTCALL_ARGS`）；[`hostcalls.rs:267-273, 296-303, 409-415`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/jit/hostcalls.rs)（`host_make_map`/`host_new_struct`/`host_make_tensor` 的 `checked_mul`）。

### 定义 I4（JitFn 类型断言）

> 在 `JitContext::get_or_compile` 中，对 `raw_ptr: *const u8` 与目标类型 `JitFn` 调用 `transmute` 之前，必须执行运行时断言 `assert_eq!(size_of::<*const u8>(), size_of::<JitFn>())`；断言失败时进程 abort。

**实现位置**：[`context.rs:50-55`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/jit/context.rs)。

---

## 5. 主定理与证明

### 5.1 定理 F1（FFI 边界 UB 自由性）

**陈述**：设 Tenth JIT 与 Rust 边界上的所有交互均满足 I1、I2、I3、I4。设 `invoke_jit` 与所有 hostcall trampoline 的实现遵循 [`hostcalls.rs`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/jit/hostcalls.rs) 与 [`context.rs`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/jit/context.rs) 中的源码。则在以下前置条件下：

- (P1) `vm` 指针非空且指向有效的 `Vm` 实例；
- (P2) JIT 产物由 Tenth translator 合法生成（未被恶意篡改）；
- (P3) 单线程执行（无并发访问 `vm`）；
- (P4) `args: &[Value]` 在 `invoke_jit` 调用期间生存期有效；

`invoke_jit` 与每个 trampoline 的执行不引入 Rust 语言意义上的 UB。

**证明**：按 FFI 边界上的 UB 来源分类证明。

**(A) 布局 UB（R2 违反）**：由 I1，所有 `Value` 经指针传递。Rust 端通过 `*const Value` / `*mut Value` 引用 `Value`，其布局由 Rust 编译器决定，与 JIT 端无关——JIT 端只把指针作为不透明 `i64`/`ptr` 处理（[`translator.rs:43-46`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/jit/translator.rs)）。`Value` 的内部布局从未跨边界。返回值仅 `bool`（1 字节，`#[repr(C)]` 保证布局），符合 Cranelift 的 `I8` 返回槽（[`translator.rs:47`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/jit/translator.rs)）。故布局一致，无 R2 违反。

**(B) Unwind UB（R3 违反）**：方向 A 由 I2 `catch_unwind` 包裹（[`hostcalls.rs:41`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/jit/hostcalls.rs)）；即使 JIT 产物内部 panic（理论上 trampoline 的 Rust 代码 panic），unwind 在 `catch_unwind` 处终止，不跨越 `extern "C"` 边界。方向 B 中 trampoline 自身是 `extern "C"`，其内部 panic 若未被 trampoline 自身捕获，按 Rust 语义在 `extern "C"` 边界 abort（Rust 1.51+ 默认行为）；由 I2 隐含——`extern "C"` 不允许 unwind。故无 R3 违反。

**(C) 切片 UB（`from_raw_parts` 溢出）**：由 I3，所有 `from_raw_parts` 经 `safe_slice` 或 `checked_mul` 闸门。`safe_slice` 在 `count > MAX_HOSTCALL_ARGS` 时返回空切片（[`hostcalls.rs:74`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/jit/hostcalls.rs)）；`MAX_HOSTCALL_ARGS = 2^{20}` 远小于 `usize::MAX`，`count * 2` 不溢出（$2^{21} \ll 2^{64}$）。`checked_mul` 在溢出时返回 `None`，被 `host_make_map` 等显式处理（[`hostcalls.rs:267-273`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/jit/hostcalls.rs)）。`ptr.is_null()` 提前返回（[`hostcalls.rs:69-71`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/jit/hostcalls.rs)）。故无 `from_raw_parts` UB。

**(D) Transmute UB（`*const u8` → `JitFn`）**：由 I4，`transmute` 前断言 `size_of::<*const u8>() == size_of::<JitFn>()`（[`context.rs:50-54`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/jit/context.rs)）。在 64 位平台上两者均为 8 字节，断言成立；若未来 `JitFn` 签名变更导致尺寸不符（理论上不可能因函数指针总是 1 字），断言触发 abort 而非静默 UB。Rust 文档明确：`transmute` 在两侧 `size_of` 相等时是安全（虽 `unsafe`）；不等时 UB。故 I4 阻止 transmute UB。

**(E) 生存期 UB（R4 违反）**：`args: &[Value]` 由 P4 保证生存期；`vm: *mut Vm` 由 P1 保证。`out: &mut Value` 由 `invoke_jit` 调用方（[`mod.rs:80-81`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/jit/mod.rs)）在栈上分配 `let mut out = Value::Unit` 并传入 `&mut out`，生存期覆盖整个调用。trampoline 内 `std::ptr::write(out, ...)`（如 [`hostcalls.rs:83`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/jit/hostcalls.rs)）写入的是已初始化的 `Value` 内存，无 drop 旧值的需要（旧值是 `Value::Unit`，无堆分配），故无 drop UB。

由 (A)–(E) 四类 UB 均被 I1–I4 阻止，且无其他 UB 来源（P2/P3 排除恶意与并发场景），定理成立。$\square$

### 5.2 定理 F2（panic 不跨越 FFI）

**陈述**：在 I2 成立且 `invoke_jit` 的实现为 [`hostcalls.rs:33-62`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/jit/hostcalls.rs) 的前提下，若 `fn_ptr(vm, args.as_ptr(), args.len(), out as *mut Value)` 的求值引发 panic，则：
(a) panic 不跨越 `extern "C"` 边界进入 JIT 产物；
(b) `vm.last_error` 被设置为形如 `"JIT panic: <msg>"` 的字符串（若 `vm` 非空）；
(c) `*out` 被写为 `Value::Unit`；
(d) `invoke_jit` 返回 `false`。

**证明**：`catch_unwind(AssertUnwindSafe(|| { fn_ptr(...) }))` 是 Rust 标准库的 panic 捕获原语。其语义（[RustStd]）为：若闭包求值 panic，`catch_unwind` 返回 `Err(payload)`，且 unwind 在闭包边界停止。`AssertUnwindSafe` 是对 `FnOnce` 的 `UnwindSafe` 约束的显式放宽，不影响捕获语义。

`fn_ptr` 调用的 ABI 为 `extern "C"`（[`hostcalls.rs:34`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/jit/hostcalls.rs)）。即使 JIT 产物内部代码或 hostcall trampoline 内的 Rust 代码触发 panic，unwind 栈展开在 `catch_unwind` 处被截获，不进入 JIT 产物的调用帧（unwind 在 Rust 端截获，与 JIT 产物的 unwind 表无关——事实上 JIT 产物无 Rust 风格 unwind 表，故跨边界 unwind 本身即 UB，但 `catch_unwind` 在 Rust 端先于该 UB 发生之前截获，因 unwind 由 Rust 代码发起）。

捕获后，`match result { Err(payload) => ... }` 分支执行：
- (b) 通过 `payload.downcast_ref::<&'static str>()` 或 `String` 提取消息（[`hostcalls.rs:48-54`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/jit/hostcalls.rs)），写入 `(*vm).set_last_error(format!("JIT panic: {}", msg))`（若 `!vm.is_null()`）；
- (c) `std::ptr::write(out, Value::Unit)`（[`hostcalls.rs:58`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/jit/hostcalls.rs)）；
- (d) 返回 `false`（[`hostcalls.rs:59`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/jit/hostcalls.rs)）。

故 (a)–(d) 成立。$\square$

**注**：`AssertUnwindSafe` 的使用意味着 `vm`/`args`/`out` 等指针的非 `UnwindSafe` 性被显式忽略——这是安全的，因为 panic 后这些指针指向的对象要么被 `invoke_jit` 显式覆盖（`out`），要么仍处于一致状态（`vm`，其字段 `last_error` 被显式设置）。

### 5.3 定理 F3（safe_slice 的溢出防护）

**陈述**：在 I3 成立且 `safe_slice` 实现为 [`hostcalls.rs:68-78`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/jit/hostcalls.rs) 的前提下，对任意输入 `ptr: *const Value` 与 `count: u64`：
(a) 若 `ptr.is_null()`，返回空切片 `&[]`，不调用 `from_raw_parts`；
(b) 若 `count == 0`，返回空切片，不调用 `from_raw_parts`；
(c) 若 `count > MAX_HOSTCALL_ARGS = 2^{20}`，返回空切片，不调用 `from_raw_parts`；
(d) 否则，调用 `from_raw_parts(ptr, count as usize)`，且 `count as usize` 在 64 位平台上无截断（`count \le 2^{20} < 2^{64}`）。

**证明**：

`safe_slice` 控制流（[`hostcalls.rs:69-77`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/jit/hostcalls.rs)）：

```
if ptr.is_null() { return &[]; }            // (a)
let n = match count {
    0 => return &[],                          // (b)
    c if c as usize > MAX_HOSTCALL_ARGS => return &[],  // (c)
    c => c as usize,                          // (d)
};
from_raw_parts(ptr, n)
```

(a) `ptr.is_null()` 分支显式返回，不执行 `from_raw_parts`。$\checkmark$

(b) `count == 0` 分支显式返回。$\checkmark$

(c) `c as usize > MAX_HOSTCALL_ARGS` 分支：在 64 位平台 `usize = u64`，故 `c as usize` 无截断，比较直接。`MAX_HOSTCALL_ARGS = 1 << 20 = 1_048_576`（[`hostcalls.rs:23`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/jit/hostcalls.rs)）。$\checkmark$

(d) 由 (c) 进入此分支的条件是 `c \le 2^{20}`，故 `c as usize` 在 32 位与 64 位平台均无截断（$2^{20} < 2^{32}$）。`from_raw_parts(ptr, n)` 的契约（[RustStd]）：`ptr` 必须非空（由 (a) 保证）、`n` 个 `Value` 内存必须可读（由 P4 + 调用方契约保证）。在 I3 + P1–P4 下，安全。$\checkmark$

对 `count * 2` 场景（`host_make_map`/`host_new_struct`），`checked_mul` 在 `count * 2` 溢出时返回 `None`，进入降级分支（写空 `Map` 或 `Value::Unit`，[`hostcalls.rs:267-273, 296-303`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/jit/hostcalls.rs)）。对 `host_make_tensor` 的 `rows * cols`，同理 `checked_mul` + `MAX_HOSTCALL_ARGS` 上限（[`hostcalls.rs:409-415`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/jit/hostcalls.rs)）。$\square$

**注**：`MAX_HOSTCALL_ARGS = 2^{20}` 的选择是工程权衡——足够大以覆盖合理的 Tenth 函数参数数（实际函数通常 < 100 参数），足够小以使 `count * 2 = 2^{21}` 远低于 `usize::MAX`，避免任何溢出。

### 5.4 定理 F4（JitFn 类型断言）

**陈述**：在 I4 成立且 `JitContext::get_or_compile` 实现为 [`context.rs:36-58`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/jit/context.rs) 的前提下：
(a) 若 `size_of::<*const u8>() != size_of::<JitFn>()`，则 `assert_eq!` 触发 panic，进程 abort（因 panic 在非 `catch_unwind` 上下文中触发，默认 abort on panic 不影响，但即使 unwind，由于 `get_or_compile` 不在 `catch_unwind` 内，将沿调用栈上溯直到 abort）；
(b) 若 `size_of::<*const u8>() == size_of::<JitFn>()`，则 `transmute(raw_ptr)` 在 Rust 语义下不引入 UB。

**证明**：

(a) `assert_eq!(x, y, "...")` 在 `x != y` 时 `panic!`。`JitContext::get_or_compile` 不被 `catch_unwind` 包裹（[`mod.rs:62-65`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/jit/mod.rs) 直接 `match`），故 panic 沿调用栈上溯。Rust 默认 panic 策略为 unwind，但若 `Cargo.toml` 配置 `panic = "abort"` 则直接 abort；无论哪种，进程不会进入 UB 状态。

(b) Rust 文档 [Nomicon]：`transmute<T, U>` 在 `size_of::<T>() == size_of::<U>()` 时是布局安全的（仍 `unsafe`，因可能违反其他不变量，如非 `Pod` 类型的位模式）。这里 `T = *const u8`，`U = JitFn = unsafe extern "C" fn(...)`，二者在所有支持的平台上均为 `size_of::<usize>()`（[`context.rs:50-54`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/jit/context.rs)）。`raw_ptr` 由 `module.get_finalized_function(fn_id)` 返回（[`context.rs:43`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/jit/context.rs)），Cranelift 保证其为合法的可执行函数地址。故 `transmute` 后的 `JitFn` 可安全调用（在 P2 假设下）。$\square$

**注**：`size_of::<JitFn>() == size_of::<*const u8>()` 在所有 Rust 支持的平台上成立，I4 的断言是防御性编程——若未来 Rust 引入"fat function pointer"或 Cranelift 变更返回类型，断言会立即触发而非静默 UB。

### 5.5 定理 F5（与 Julia `ccall` / PyPy `rffi` 对比）

**陈述**：Tenth hostcall 协议（I1–I4）与 Julia `ccall` / PyPy `rffi` 在 FFI 安全设计上有以下结构差异：

| 维度 | Tenth hostcall | Julia `ccall` | PyPy `rffi` |
|------|---------------|---------------|-------------|
| 复杂 enum 传递 | 直接 `*const Value`（I1 保证布局不跨边界） | 不透明 `jl_value_t*` | 不透明 `PyObject*` |
| Panic 隔离 | `catch_unwind` 显式包裹（I2） | Julia 异常不能跨 `ccall`（约定） | RPython 异常转进程退出 |
| 切片溢出防护 | `safe_slice` + `MAX_HOSTCALL_ARGS` + `checked_mul`（I3） | `ccall` 内部不切片，由调用方 `unsafe_load` | `rffi` 不提供切片 |
| Transmute 安全 | `size_of` 断言 + `transmute`（I4） | `cglobal` + `ccall` 类型签名一致 | RPython 工具链生成包装 |
| 调用方向 | 双向（A: Rust→JIT, B: JIT→Rust） | 单向（Julia→C） | 单向（PyPy→C） |

**证明（结构性论证）**：

- **复杂 enum**：Julia 与 PyPy 通过不透明指针绕过布局问题；Tenth 选择直接传 `*const Value`，但因 I1 保证 `Value` 内部布局不跨边界（仅指针跨边界），故等价安全。
- **Panic**：Tenth 是双向 FFI，且 JIT 产物不是 Rust 编译，故必须显式 `catch_unwind`（I2）；Julia/PyPy 的 `ccall` 是单向，被调方是 C 代码不会 panic，故无需。
- **切片**：Julia/PyPy 不在 FFI 边界切片；Tenth 因 `host_make_vec`/`host_make_map` 等需要从 JIT 端接收变长参数列表，必须切片，故需 I3。
- **Transmute**：Tenth 必须把 Cranelift 返回的 `*const u8` 转为类型化函数指针；Julia/PyPy 通过函数名动态解析，不 transmute。$\square$

---

## 6. 4 个 FFI 不变量的逐一分析

### 6.1 I1：out-pointer 传递

**机制**：所有 `Value` 类型参数以 `*const Value`（输入）或 `*mut Value`（输出）传递。返回值仅 `bool`（[`hostcalls.rs:34`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/jit/hostcalls.rs)）。

**为什么必须如此**：`Value` 是 32 字节非 `#[repr(C)]` enum。Rust 的 enum 布局（tag 位置、niche 优化）由编译器自由选择，跨 FFI 时 JIT 端无法可靠地按 Rust 布局填充返回槽。Cranelift 的 `Signature::returns` 仅支持基本类型（`I8`/`I64`/`F64`/指针），不支持 32 字节聚合体（除非用 `struct` 返回，但 `struct` 返回 ABI 在不同平台不一致——System V AMD64 与 Microsoft x64 不同）。

**额外好处**：错误信号统一——返回 `bool` 表示成功/失败，错误细节通过 `vm.last_error` 传递（[`hostcalls.rs:12`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/jit/hostcalls.rs)）。

### 6.2 I2：panic 隔离

**机制**：`invoke_jit` 用 `catch_unwind(AssertUnwindSafe(|| { fn_ptr(...) }))` 包裹方向 A 调用（[`hostcalls.rs:41-43`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/jit/hostcalls.rs)）。方向 B 的 trampoline 标注 `extern "C"`（如 [`hostcalls.rs:82`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/jit/hostcalls.rs)），按 Rust 语义其内部 panic 在边界 abort（或被调用方 `catch_unwind` 捕获）。

**为什么必须如此**：JIT 产物由 Cranelift 生成，无 Rust unwind 表。若 Rust 端 trampoline panic，unwind 进入 JIT 帧时找不到 landing pad，触发 UB（段错误或进程 abort，取决于平台）。`catch_unwind` 在 Rust 端截获，避免此情形。

**边界情况**：`AssertUnwindSafe` 的使用意味着我们显式声明 `vm`/`args`/`out` 在 panic 后仍可用——这是安全的，因为 panic 后 `vm` 的状态仍一致（除了可能未完成的 hostcall 副作用），`out` 被显式覆盖为 `Value::Unit`。

### 6.3 I3：safe_slice 闸门

**机制**：所有 `from_raw_parts` 调用经 `safe_slice` 或等价 `checked_mul` 闸门（§4 定义 I3）。

**为什么必须如此**：JIT 端传来的 `count: u64` 是不可信的——若 JIT 产物被恶意篡改或 translator 有 bug，`count` 可能是 `u64::MAX`。`from_raw_parts(ptr, u64::MAX as usize)` 立即 UB（读取越界内存）。`safe_slice` 在 `count > 2^{20}` 时拒绝，`checked_mul` 在 `count * 2` 溢出时拒绝。

**额外考虑**：`MAX_HOSTCALL_ARGS = 2^{20} = 1_048_576` 是工程上限。任何 Tenth 函数实际参数数远小于此值；超出即视为翻译器 bug 或恶意输入，安全降级（返回空切片，调用方 hostcall 进入错误路径）。

### 6.4 I4：JitFn 类型断言

**机制**：[`context.rs:50-54`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/jit/context.rs) 在 `transmute` 前断言 `size_of::<*const u8>() == size_of::<JitFn>()`。

**为什么必须如此**：`std::mem::transmute<T, U>` 在 `size_of::<T>() != size_of::<U>()` 时是 UB（[Nomicon]）。Cranelift 的 `get_finalized_function` 返回 `*const u8`，Tenth 需转为 `JitFn`。在所有当前支持的平台上二者尺寸相等（8 字节），但断言是防御性编程——若未来 Rust 引入 fat function pointer 或 Cranelift 变更返回类型，断言触发而非静默 UB。

**为何不直接用 `as` 转换**：Rust 不允许 `*const u8 as JitFn`（函数指针类型转换需 `transmute`）。`transmute` 是唯一路径。

---

## 7. 每个不变量对应的潜在 UB

下表列出每个不变量阻止的具体 UB：

| 不变量 | 阻止的 UB | Rust 文档依据 | 触发场景（反例） |
|--------|----------|--------------|----------------|
| **I1** | 跨 FFI 传递非 `#[repr(C)]` enum 的布局 UB | [Nomicon] "unwinding and FFI" + 类型布局章节 | 若 `Value` 通过返回槽传递，JIT 端按 Cranelift `Signature::returns` 填充 32 字节，Rust 端按 Rust ABI 读取——tag 位置/变体顺序不一致即 UB |
| **I2** | Panic 跨 `extern "C"` 边界的 UB | [RustRef] §"FFI and unwinding" | 若无 `catch_unwind`，hostcall 内 `vm.add()` 触发 panic（如内部 `unwrap` 失败），unwind 进入 JIT 帧，无 landing pad，段错误或 abort |
| **I3** | `from_raw_parts` count 溢出导致的越界读 UB | [RustStd] `slice::from_raw_parts` 安全契约 | 若 `count = u64::MAX`，`count as usize = 18446744073709551615`，`from_raw_parts` 读取 32 × count 字节，立即段错误 |
| **I4** | `transmute` 尺寸不匹配的 UB | [Nomicon] `transmute` 章节 | 若未来 `JitFn` 改为 fat pointer（16 字节），`transmute(*const u8)` 只填充 8 字节，剩余 8 字节是 garbage，调用时通过 garbage 间接寻址——UB |

**未覆盖的 UB 类（见 §10 局限）**：
- `vm` 指针被 JIT 产物篡改后解引用的 UB；
- 并发调用 `invoke_jit` 导致 `Vm` 内部 `Vec`/`HashMap` 数据竞争的 UB；
- JIT 产物在 `args_ptr` 指向已释放内存的 UB（P4 假设排除）。

---

## 8. 与 Julia `ccall` / PyPy `rffi` 对比

### 8.1 Julia `ccall` 的安全模型

Julia 的 `ccall` 设计假设被调方是 C 代码（[JuliaDoc]）。安全机制：
- 类型签名静态检查（`RetType` 与 `ArgTypes` 必须是 `convert` 兼容的 C 类型）；
- 不支持直接传递 Julia 复合类型（需通过 `Ptr{T}` 不透明指针）；
- 异常不能跨越 `ccall`——若 C 代码回调 Julia 并抛异常，进程 abort。

**对比 Tenth**：Tenth 的 `Value` 是 Julia 的 `jl_value_t*` 等价物，但 Tenth 选择直接传 `*const Value`（而非不透明指针），由 I1 保证布局不跨边界。差异：Tenth 在 Rust 端可直接 `&*v` 解引用访问 `Value` 字段，Julia 必须通过 `unsafe_pointer_to_objref` 显式转换。

### 8.2 PyPy `rffi` 的安全模型

PyPy 的 `rffi.llexternal` 生成 RPython 包装代码（[PyPyDoc]）。安全机制：
- 类型必须是 C 兼容（`Signed`/`Unsigned`/`Ptr`）；
- GC 障碍由 RPython 编译器自动插入；
- 异常转为进程退出（RPython 无 unwind）。

**对比 Tenth**：PyPy 是单向 FFI（PyPy→C），Tenth 是双向（JIT↔Rust）。PyPy 的 GC 障碍在 Tenth 中等价于 `Rc` 引用计数——`Value::clone()` 增加 `Rc` 计数，但 `*const Value` 不增加（由调用方契约保证生存期，P4）。

### 8.3 Tenth 设计的独特性

Tenth hostcall 协议的独特之处：
1. **双向 FFI**：方向 A（Rust→JIT）与方向 B（JIT→Rust）均存在，Julia/PyPy 仅单向。
2. **Rust ABI 感知**：Tenth 利用 Rust 编译器对 `Value` 布局的自由选择，但通过 I1 避免布局跨边界——这与 Julia/PyPy 的"不透明指针"殊途同归，但 Tenth 在 Rust 端无需 `unsafe_pointer_to_objref` 转换。
3. **panic 显式隔离**：Tenth 是唯一显式 `catch_unwind` 的——Julia/PyPy 假设 C 代码不会 panic，Tenth 不能假设 JIT 产物内部 Rust trampoline 不会 panic。

---

## 9. 工程权衡

### 9.1 out-pointer 的代价

每次 hostcall 需在调用方分配 `Value` 内存（栈或堆），通过指针写入。相比直接返回 `Value`：
- **优点**：避免布局 UB（I1），统一错误信号（`bool` + `last_error`）；
- **代价**：每次 hostcall 多一次 `std::ptr::write`（约 1 ns），且 `Value` 的 drop 在调用方完成（不在 trampoline 内）。

对 Tenth 而言，hostcall 本身的 Rust 操作（如 `vm.add()`）开销远大于一次 `ptr::write`，故代价可忽略。

### 9.2 `catch_unwind` 的代价

`catch_unwind` 在 panic 不发生时几乎零开销（Rust 实现为 zero-cost exception）。在 panic 发生时，unwind 比 abort 慢约 100×，但 panic 是异常路径，可接受。

代价主要在 `AssertUnwindSafe` 的语义放宽——开发者必须显式确认 panic 后状态仍可用。Tenth 通过 `out = Value::Unit` 覆盖与 `vm.last_error` 设置保证。

### 9.3 `MAX_HOSTCALL_ARGS` 的选择

$2^{20} = 1\,048\,576$ 是工程权衡：
- 足够大：覆盖任何合理的 Tenth 函数参数数（实际 < 100）；
- 足够小：`count * 2 = 2^{21} \ll 2^{64}`，无溢出风险；
- 是 2 的幂：便于位运算检查（虽然 `safe_slice` 用比较而非位运算）。

若设为 $2^{32}$，则 32 位平台溢出；若设为 $2^{10}$，可能拒绝合法大参数列表。$2^{20}$ 是合理折衷。

### 9.4 类型断言的代价

`assert_eq!(size_of::<*const u8>(), size_of::<JitFn>())` 是编译期常量比较（`size_of` 是 `const fn`），无运行时开销。代价仅在断言失败时——但这种情况意味着 Rust/Cranelift ABI 变化，应立即 abort 而非继续。

---

## 10. 局限

本节诚实披露形式化模型的局限。

### L1. 不覆盖恶意 JIT 产物

**陈述**：定理 F1 的前置条件 P2 假设 JIT 产物由 Tenth translator 合法生成。若 JIT 产物被恶意篡改（如直接修改机器码），可违反 I1–I4 中的任意不变量——例如，JIT 端可向 trampoline 传入 `count = u64::MAX`，绕过 `safe_slice`（虽然 `safe_slice` 仍拒绝，但 JIT 可传 `args_ptr = 0` 后 `safe_slice` 返回 `&[]`，trampoline 进入错误路径，但若 JIT 进一步篡改 `vm` 指针，`(*vm).set_last_error` 解引用空指针 UB）。

**影响**：定理 F1 不保证"对抗恶意 JIT"的安全性。Tenth 的安全模型假设 translator 是可信计算基（TCB）的一部分。

**缓解**：translator 的输入是 Tenth 字节码（`Chunk`），由 Rust 编译器生成；若字节码本身可信，translator 的输出可信。这构成"translator 作为 TCB"的安全边界。

### L2. 不覆盖并发

**陈述**：P3 假设单线程执行。若多线程同时调用 `invoke_jit` 共享同一 `Vm`，`Vm` 内部的 `Vec`/`HashMap`（`stack`、`globals`、`chunks`）数据竞争，立即 UB。

**影响**：Tenth 当前不支持并发 JIT。`Vm` 是 `!Sync`（含 `Rc`）。

**缓解**：单线程是 Tenth 的设计选择（AI 工作负载通常单线程主导，多线程通过 batch 实现）。未来若支持并发，需重构 `Vm` 为 `Arc<Mutex<Vm>>` 或线程局部实例。

### L3. 不覆盖 `vm` 指针的生存期

**陈述**：P1 假设 `vm` 非空且指向有效 `Vm`。若 JIT 产物传入空 `vm`，`host_call` 等依赖 `vm.string_at` 的 trampoline 在 `&mut *vm` 时立即 UB。

**影响**：定理 F1 不覆盖 `vm = null` 场景。

**缓解**：`invoke_jit` 的调用方（[`mod.rs:81`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/jit/mod.rs)）传入 `vm as *mut Vm`，由 `&mut vm` 借用保证非空。但 trampoline 自身不检查 `vm.is_null()`——除 `invoke_jit` 的 panic 分支显式检查（[`hostcalls.rs:55`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/jit/hostcalls.rs)）。其他 trampoline 假设 `vm` 非空，是 P1 的隐含契约。

### L4. 形式化不完备

**陈述**：定理 F1 的证明按 UB 类别（A–E）分类，但未严格证明这五类是完备的——可能存在未被识别的 UB 类（如 `Value` 的 `Drop` 实现中的 UB，若 `out` 指向的旧 `Value` 不是 `Unit` 而是含 `Rc` 的变体，`std::ptr::write` 会跳过 drop 导致 `Rc` 泄漏——虽然不是 UB 但是资源泄漏）。

**影响**：定理 F1 是"已知 UB 类的自由性"，非"所有 UB 类的自由性"。

**缓解**：`mod.rs:80` 的 `let mut out = Value::Unit` 保证 `out` 初始为 `Unit`（无堆资源），故 `ptr::write` 不泄漏。但若未来调用方改变（如重用 `out` 内存），需重新审视。

### L5. `extern "C"` 与 `extern "C-unwind"` 的选择

**陈述**：Tenth 使用 `extern "C"` 而非 `extern "C-unwind"`（Rust 1.51+）。若使用 `C-unwind`，panic 可安全跨 FFI，无需 `catch_unwind`。但 Tenth 选择 `extern "C"` + `catch_unwind`，理由是 Cranelift JIT 产物的 unwind 表与 Rust `C-unwind` 不兼容。

**影响**：`catch_unwind` 的 `AssertUnwindSafe` 是显式放宽 `UnwindSafe` 约束，可能掩盖状态不一致——例如，若 hostcall 在 `vm.set_last_error` 前 panic，`vm.last_error` 仍是上次的值，但 `out` 被覆盖为 `Unit`，状态部分不一致。

**缓解**：`invoke_jit` 在 panic 分支显式设置 `last_error`（[`hostcalls.rs:56`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/jit/hostcalls.rs)），覆盖上次值。但若 `set_last_error` 自身 panic（理论上 `String` 操作可能 OOM panic），状态仍不一致——这是未覆盖的边界。

### L6. 与 T31（栈区设计）联动的局限

**陈述**：T31 论证 JIT 翻译器的"栈区设计"（单个大 `StackSlot`，无 phi 节点）与 SSA 的语义等价性。T31 的栈区是 `Value` 大小的内存区域（`VALUE_SIZE * MAX_STACK_DEPTH`，[`translator.rs:28-32`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/jit/translator.rs)），JIT 端通过 `stack_store`/`stack_load` 操作。这些 `Value` 内存的初始化由翻译器保证（push 总是先于 pop）。

**联动局限**：T32 的 I1 保证 hostcall 边界上的 `Value` 通过指针传递，但不覆盖 JIT 端栈区内部的 `Value` 布局——若 T31 的栈区设计在某种边界情况下（如未初始化 slot 被读取），`Value` 内存是未初始化的 `MaybeUninit`，传给 hostcall 的 `*const Value` 解引用 UB。T31 的语义等价性证明需保证"所有 read 之前有对应 write"，否则 T32 的 I1 不成立。

**缓解**：T31 的翻译器对每个 `Op` 显式处理（[`translator.rs:221-483`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/jit/translator.rs)），每个 pop 前有对应 push。但形式化证明（T31 待完成）需明确此点。

### L7. `transmute` 的语义假设

**陈述**：定理 F4 证明 (b) 称 `transmute` 在尺寸相等时"布局安全"。但 Rust 文档 [Nomicon] 进一步要求：若 `U` 是非 `Pod` 类型，`transmute` 的位模式必须有效。`JitFn` 是函数指针类型，其位模式约束是"必须是有效函数地址或 null"。Cranelift 的 `get_finalized_function` 返回非 null 有效地址，故满足。

**局限**：若未来 `JitFn` 改为含数据字段（如 fat pointer 含数据指针与 vtable），`transmute` 的位模式约束需重新审视。I4 的 `size_of` 断言只能阻止尺寸不匹配，不能阻止位模式不匹配。

---

## 11. 开放问题

1. **基于类型系统的 hostcall 协议自动生成**：当前 I1–I4 由人工维护。能否设计一个类型系统（如 Rust 过程宏 + Cranelift 签名生成器），使得 hostcall 协议的违反在编译期被检测？例如，宏自动生成 `extern "C"` trampoline 与对应的 Cranelift `Signature`，保证两侧签名一致。

2. **`extern "C-unwind"` 的可行性**：若 Cranelift 未来支持生成与 Rust `C-unwind` 兼容的 unwind 表，可移除 `catch_unwind`，简化 I2。但这要求 Cranelift 集成平台 unwind 机制（DWARF/SEH）。

3. **形式化验证**：能否用 RustBelt [RustBelt] 或 Miri [Miri] 对 I1–I4 进行机器检查？Miri 可检测 `from_raw_parts` 溢出与 `transmute` 尺寸不匹配，但不检测跨 FFI unwind UB（Miri 不模拟 JIT）。

4. **并发 JIT**：若 Tenth 未来支持多线程 JIT（如 batch 处理），如何扩展 I1–I4？`Vm` 需重构为线程安全，hostcall 协议需增加同步不变量。

5. **与 T31 联动的形式化**：T31 的栈区设计语义等价性证明（待完成）需与 T32 的 I1 联合——证明"栈区中的 `Value` 在传给 hostcall 前已初始化"。这是 T31 + T32 的联合定理，待未来工作。

---

## 12. 结论

本文形式化了 Tenth JIT 与 Rust 边界上的 4 个 FFI 不变量 I1–I4，并证明主定理 F1——在 P1–P4 前置条件下，I1–I4 共同保证 UB 自由性。分定理 F2–F5 分别证明 panic 隔离、切片溢出防护、transmute 安全、与 Julia/PyPy 的对比。本文的贡献在于将 Tenth hostcall 协议的工程实践提升为形式化不变量集合，每个不变量对应一类潜在 UB。

本文的局限（§10）诚实披露了模型的不完备性——仅覆盖已知 UB 类，不覆盖恶意 JIT、并发、`vm` 篡改等场景。这些局限指向未来工作：基于类型系统的协议自动生成、`C-unwind` 集成、RustBelt/Miri 机器检查、与 T31 联合形式化。

Tenth hostcall 协议的设计哲学是"**每个 FFI 不变量对应一个潜在 UB**"——通过显式编码 UB 防御，使 JIT↔Rust 边界的安全性可形式化、可论证、可审查。这一哲学对其他 Rust + JIT 集成项目（如 Wasmer、Wasmtime 的 host 函数协议）有参考价值。

---

## 参考文献

- [RustRef] Rust Reference. *Foreign Function Interface*. https://doc.rust-lang.org/reference/unsafe-blocks.html#behavior-considered-undefined
- [Nomicon] Rustonomicon. *Transmutes* and *Unwinding*. https://doc.rust-lang.org/nomicon/
- [RustStd] Rust Standard Library. *std::panic::catch_unwind*, *std::slice::from_raw_parts*. https://doc.rust-lang.org/std/
- [JuliaDoc] Julia Documentation. *Calling C and Fortran Code (ccall)*. https://docs.julialang.org/en/v1/manual/calling-c-and-fortran-code/
- [PyPyDoc] PyPy Documentation. *rffi module*. https://doc.pypy.org/en/latest/
- [RustBelt] Jung, R. et al. *RustBelt: Securing the Foundations of the Rust Programming Language*. POPL 2018.
- [Miri] Rust Miri. *An interpreter for Rust's MIR*. https://github.com/rust-lang/miri
- [Cranelift] Bytecode Alliance. *Cranelift Code Generator*. https://github.com/bytecodealliance/wasmtime/tree/main/cranelift

---

## 附录 A：定理索引

| 定理 | 陈述 | 证明位置 | 对应不变量 |
|------|------|---------|-----------|
| F1 | FFI 边界 UB 自由性 | §5.1 | I1, I2, I3, I4 |
| F2 | panic 不跨越 FFI | §5.2 | I2 |
| F3 | safe_slice 溢出防护 | §5.3 | I3 |
| F4 | JitFn 类型断言 | §5.4 | I4 |
| F5 | 与 Julia/PyPy 对比 | §5.5 | (结构性) |

## 附录 B：4 个不变量与源码位置对照

| 不变量 | 源码位置 | 关键行 |
|--------|---------|--------|
| I1 | [`hostcalls.rs`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/jit/hostcalls.rs) | L8-13（约定注释）、L34（`invoke_jit` 签名）、L82-451（所有 trampoline 签名） |
| I2 | [`hostcalls.rs`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/jit/hostcalls.rs) | L41-43（`catch_unwind`）、L46-61（panic 处理） |
| I3 | [`hostcalls.rs`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/jit/hostcalls.rs) | L23（`MAX_HOSTCALL_ARGS`）、L68-78（`safe_slice`）、L267-273/L296-303/L409-415（`checked_mul`） |
| I4 | [`context.rs`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/jit/context.rs) | L50-54（`assert_eq!`）、L55（`transmute`） |

## 附录 C：实施建议

基于本文的形式化结论，对 Tenth JIT 后续维护的建议：

1. **保持 I1–I4 的显式编码**：未来若新增 hostcall，必须遵循 out-pointer 协议（I1）、不在 trampoline 内 panic 或被 `catch_unwind` 覆盖（I2）、所有 `from_raw_parts` 经 `safe_slice`/`checked_mul`（I3）。
2. **`MAX_HOSTCALL_ARGS` 的版本化**：若调整该常量，需重新评估 I3 的溢出边界（`count * 2` 是否仍不溢出）。
3. **`JitFn` 签名变更的同步**：若 `JitFn` 签名变更（如增加参数），I4 的断言会触发；此时需同步更新 `translator.rs` 的 `Signature` 声明（[`translator.rs:43-47`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/src/compile/jit/translator.rs)）。
4. **新增 hostcall 的检查清单**：每个新 hostcall 需在 PR 中确认 (a) 签名为 `unsafe extern "C" fn(..., out: *mut Value)` 或返回基本类型；(b) 内部不 panic 或 panic 被上层 `catch_unwind` 覆盖；(c) 任何 `from_raw_parts` 经 `safe_slice`；(d) 不引入新的 `transmute`。
5. **与 T31 的联合验证**：当 T31 的栈区设计形式化完成后，需联合 T32 的 I1 验证"栈区 `Value` 在传给 hostcall 前已初始化"——这是 T31 + T32 的联合不变量。
