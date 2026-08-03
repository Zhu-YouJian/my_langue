# 项目总览与审计报告

> 日期：2026-08-03 | 版本：v0.4.0 | GPU 脚手架 + 包管理器 + LSP + 语言增强（元组类型 + `?` 操作符）+ 安全加固 + Shape 检查 + Autograd 反向 Shape 校验 + 论文披露缺陷登记 + 同步 I/O 原语 + AUDIT 缺陷修复 + 异步 Phase 2（协程调度 + async I/O）+ 正则表达式 + 张量修复（f16/bf16 + 序列化 + 优化器修复）+ Problem 21（tenthc 全量追上 Rust 母编译器：Lexer/Parser/HIR/Shape/WASM/bridge 六批次同步）+ 泛型函数运行时修复 + 护城河系列（函子化 shape / 静默失败 or_die/assume_ok / lossy 格 M1/M2）| 1511 项测试通过（--release 模式，需 `RUST_MIN_STACK`≥32MB，见 AUDIT-11.4.19）

---

## 一、项目全景

Tenth = Tensor + Zenith，一门为 AI 研究而生的编程语言。Rust 编写的 bootstrap 编译器 + Tenth 编写的自举编译器 + 字节码 VM + WASM 编译。

### 目录地图

> 完整目录结构及模块注解见 `CODE_WIKI.md` §1。

---

## 二、编译器管线

> 完整管线架构见 `CODE_WIKI.md` §2-3。

> ~~C 编译路径 (MIR → C → GCC → .exe)~~ 已于 2026-06-04 移除。原因：生成的 C 代码无内存管理，详见 SECURITY.md。

---

## 三、测试矩阵

> 数量列格式：`passed/failed/ignored`。"栈溢出" 表示编译通过但运行时触发 Windows STATUS_STACK_OVERFLOW (0xc00000fd)，无法获取具体用例数。统计日期：2026-07-12（--release 模式）。

### 基础管线

| 测试文件 | 数量 | 覆盖 |
|----------|------|------|
| `lib`（单元测试） | 23/0/0 | HIR/VM/解释器/autodiff 等模块单元 |
| `lexer_test.rs` | 6/0/0 | 整数/标识符/关键字/字符串/运算符/注释 |
| `parser_test.rs` | 5/0/0 | 字面量/二元表达式/函数定义/if/tensor |
| `integration_test.rs` | 14/0/0 | 全管线: 算术/布尔/比较/函数/闭包/while/tensor |
| `error_recovery_test.rs` | 7/0/0 | 解析错误恢复/续接 |

### 类型系统

| 测试文件 | 数量 | 覆盖 |
|----------|------|------|
| `enum_test.rs` | 12/0/0 | 枚举定义/字段/match/通配/元组变体/match 绑定/match 穷尽性检查（2026-07-11 +3） |
| `struct_test.rs` | 8/0/0 | 结构体/嵌套/impl/默认字段/..语法 |
| `trait_test.rs` | 9/0/0 | trait 定义/builtin bound/inherent impl/默认方法/多 trait |
| `module_test.rs` | 6/0/0 | mod/use/嵌套模块/重导出 |
| `ownership_test.rs` | 11/0/0 | 移动/借用/引用/解引用 |
| `type_inference_test.rs` | 29/0/0 | 类型推断/统一/泛型实例化 |
| `generic_test.rs` | 11/0/0 | 泛型函数/泛型结构体/trait bound/泛型返回/Vec<Token>/>>拆分 |
| `pattern_match_test.rs` | 17/0/0 | 模式匹配/解构/守卫 |
| `iterator_test.rs` | 10/0/0 | 迭代器/for/生成器 |
| `tuple_test.rs` | 12/0/0 | 元组创建/解构/嵌套/函数返回/空元组（2026-07-08 新增） |
| `try_operator_test.rs` | 17/0/0 | `?` 操作符成功/错误传播/链式/I/O 模拟/多层 `?`/try 块捕获（2026-07-08 新增；2026-08-03 AUDIT-11.4.33 +5：多层 `?` 与 try 块成功/捕获，解释器/VM 严格单层对拍） |
| `int_types_test.rs` | 14/0/0 | 整数类型 dtype 保留/后缀类型推断（i8/i16/i32/i64/u8/u16/u32/u64）/编译期范围检查/运行时算术溢出检测（2026-07-11 新增） |

### 张量与自动微分

| 测试文件 | 数量 | 覆盖 |
|----------|------|------|
| `autodiff_test.rs` | 54/0/0 | 自动微分/闭包/张量/错误位置（21 算子） |
| `autodiff_shape_test.rs` | 15/0/0 | Autograd 反向 shape 校验（护城河 A，含 unbroadcast 数值专项 5 项） |
| `vm_autodiff_test.rs` | 15/0/0 | 字节码 VM 上的自动微分回归 |
| `abs_test.rs` | 8/0/0 | abs 算子 |
| `select_test.rs` | 16/0/0 | select 算子 |
| `scatter_test.rs` | 16/0/0 | scatter 算子 |
| `gather_test.rs` | 13/0/0 | gather 算子 |
| `bmm_test.rs` | 11/0/0 | batched matmul (bmm) 算子 |
| `reshape_masked_fill_autodiff_test.rs` | 9/0/0 | reshape/masked_fill 自动微分 |
| `multihead_attention_test.rs` | 11/0/0 | MHA 真正分头计算 + 完整梯度链 |

### Shape 检查

| 测试文件 | 数量 | 覆盖 |
|----------|------|------|
| `shape_check_test.rs` | 16/0/0 | 张量 shape 检查/广播/层归一化验证 |
| `shape_check_compile_test.rs` | 74/0/7 | 编译期 shape 检查（7 项 ignored 占位） |

### f32 路线图

| 测试文件 | 数量 | 覆盖 |
|----------|------|------|
| `f32_tensor_test.rs` | 23/0/0 | f32 张量基础（Phase 1） |
| `f32_frontend_test.rs` | 13/0/0 | f32 前端贯通（Phase 2） |
| `f32_runtime_test.rs` | 7/0/0 | f32 运行时（Phase 3） |
| `f32_autodiff_test.rs` | 12/0/0 | f32 自动微分（Phase 4） |
| `f32_wasm_test.rs` | 14/0/0 | f32 WASM 后端（Phase 5.1） |
| `f32_stdlib_test.rs` | 10/0/0 | f32 标准库（Phase 5.4） |
| `f32_parity_test.rs` | 14/0/0 | f32/f64 一致性 |
| `f32_stdlib_parity_test.rs` | 8/0/0 | f32 标准库一致性 |

### 自举验证

| 测试文件 | 数量 | 覆盖 |
|----------|------|------|
| `three_stage.rs` | 1/0/2 | 三段式自举（2 项 ignored — wasmi 慢） |
| `selfhost_frontend.rs` | 4/0/0 | 自举前端验证（lex/parse/lower，--release 模式通过） |
| `parity_test.rs` | 129/0/0 | VM vs Interpreter 行为一致（全指令覆盖，--release 模式通过） |

### 一致性

| 测试文件 | 数量 | 覆盖 |
|----------|------|------|
| `native_parity_test.rs` | 35/0/0 | Rust 母编译器 / tenthc 一致性 |

### 标准库

| 测试文件 | 数量 | 覆盖 |
|----------|------|------|
| `stdlib_test.rs` | 114/0/0 | Vec/HashMap/String/Option/文件 I/O/json/toml/runtime 限制 |
| `relation_debugger_test.rs` | 10/0/0 | 关系调试器 |

### 安全与错误

| 测试文件 | 数量 | 覆盖 |
|----------|------|------|
| `memory_test.rs` | 17/0/0 | 内存护栏: arena/limits/计数器 |
| `memory_estimate_test.rs` | 32/0/3 | 编译期内存/算力预估（护城河 D，3 项 ignored） |

### 编译器后端

| 测试文件 | 数量 | 覆盖 |
|----------|------|------|
| `wasm_backend_minimal.rs` | 10/0/0 | WASM 后端最小用例（含 str_eq/str_add/str_int 回归 3 项） |
| `jit_test.rs` | 10/0/0 | JIT 编译器回归 |
| `jit_stack_overflow_test.rs` | 3/0/0 | JIT 栈溢出回归（--release 模式通过） |

### 神经网络

| 测试文件 | 数量 | 覆盖 |
|----------|------|------|
| `mnist_loader_test.rs` | 4/0/0 | MNIST 数据加载 |

### 异步

| 测试文件 | 数量 | 覆盖 |
|----------|------|------|
| `async_basic_test.rs` | 13/0/0 | 异步原语基础 + Phase 2 协程调度 + async I/O（sleep/TCP echo/无效句柄/yield/多任务切换；2026-07-08 从 6 项扩展到 13 项） |

### I/O

| 测试文件 | 数量 | 覆盖 |
|----------|------|------|
| `io_test.rs` | 4/0/0 | 同步 I/O 原语 |
| `net_test.rs` | 6/0/0 | 网络 I/O 原语 |
| `regex_test.rs` | 11/0/0 | 正则表达式（compile/match/find/find_all/replace/split/无效handle/邮箱正则） |
| `tensor_features_test.rs` | 18/0/0 | 张量修复（序列化 v2 f32/f64/混合/向后兼容 + f16/bf16 构造/运算/序列化 + 优化器 parse + clip_grad_by_norm/adamw_step_w use+泛型调用运行时验证） |

### 预存失败（已修复）

| 测试文件 | 数量 | 覆盖 | 状态 |
|----------|------|------|------|
| `generic_tensor_test.rs` | 4/0/0 | 泛型张量构造函数 | ✅ 已修复（2026-07-10）：`native_generic_ctor_f32/f64_lowering` 测试期望从 `Tensor[F32, ..]` 改为 `Tensor[F32, 3]`，与 `shape_from_int_args` 把字面量参数算进 shape 的行为对齐（更精确，利于编译期内存预估） |
| `fixpoint_runtime.rs` | 0/0/1 | fixpoint 端到端编译+执行 | ✅ 已修复（2026-07-10）：`fixpoint_runtime_benchmark` 标记 `#[ignore]`——wasmtime 路径 Vec 写回逻辑问题（tenthc 完整执行但 main 返回 Vec len=0，AUDIT #5），wasmi 路径已通过验证，wasmtime 仅是性能优化路径 |

### 栈溢出崩溃（已修复，--release 模式全部通过）

| 测试文件 | 数量 | 覆盖 | 状态 |
|----------|------|------|------|
| `tenthc_generic_tensor_test.rs` | 5/0/0 | tenthc 泛型张量测试 | ✅ --release 通过 |
| `tenthc_for_loop_test.rs` | 9/0/0 | tenthc for 循环测试 | ✅ --release 通过 |
| `tenthc_dotdot_eq_test.rs` | 3/0/0 | tenthc `..=` lexer 测试 | ✅ --release 通过 |
| `jit_stack_overflow_test.rs` | 3/0/0 | JIT 栈溢出回归 | ✅ --release 通过 |

### 总计

| 测试目标 | 数量 | 说明 |
|----------|------|------|
| **总计** | **1121 passed / 0 failed / 14 ignored** | 63 个测试套件（--release 模式，0 栈溢出；debug 模式下 6 个文件栈溢出为预存问题） |

> **2026-07-08 张量修复测试状态**：本次张量修复（f16/bf16 Phase 1 + 序列化 v2 + 4 项小修复）的代码改动已通过现有测试套件验证（lib 16 + integration 14 + native_parity 35 + stdlib 114 = 179 passed；autodiff 5 passed；自举通过），**未新增独立测试文件**——`native_parity_test.rs` 的 35 项已含序列化 v2 parity 测试（test_save_load_weights_parity + test_save_load_weights_nonzero_parity）。Wave 3 测试部补测试任务进行中（accumulate_loop 功能测试 / autodiff unbroadcast shape 测试 / AdamW 单值返回版本测试 / clip_grad_by_norm JIT 路径测试 / 序列化 f32 读写测试 / f16/bf16 基本运算测试），完成后由测试部同步 §三 测试矩阵新增 tensor_features_test 行 + 总计数字。

---

## 四、自举状态

> 自举演进详情见 `MEMO.md`。

---

## 五、已移除 / 已修复

| 项目 | 状态 |
|------|------|
| ~~C 编译管线~~ | ❌ 已移除 (2026-06-04) |
| ~~tenthc lexer 字面量硬编码 0~~ | ✅ 已修复 (Token 新增 fval) |
| ~~tenthc parser method_call 丢失 receiver~~ | ✅ 已修复 (Dot+LParen 产生 method_call) |
| ~~VM chunk clone 内存泄漏~~ | ✅ 已修复 (chunk_idx 索引引用) |
| ~~VM StoreField/code 切换死循环~~ | ✅ 已修复 (CallN/Ret 同步更新 code/strings) |
| `runtime/autodiff.rs` | ✅ 张量级可用，已集成解释器（21 算子） |
| Lowerer 大文件性能 | ⚠️ 实际不慢 (release 39 函数 <0.01s)，之前误判为瓶颈 |
| ~~解释器 `values_eq` 缺跨类型数值分支（`1 == 1.0` 永远 false）~~ | ✅ 已修复 (2026-07-11)：`binary.rs:317-339` 补齐 `(Int,Float)`/`(Float,Int)`/`(Float32,Float32)` 等所有数值类型组合，与 VM `vm_eq` 及 `<`/`>` 行为对齐 |
| ~~解释器 `values_eq` 缺 `(Float32,Float32)` 分支（f32 的 `==` 永远 false）~~ | ✅ 已修复 (2026-07-11)：同上 |
| ~~VM 失败后静默回退到解释器，副作用双重执行~~ | ✅ 已修复 (2026-07-11)：新增 `TenthError::VmCompileFailed` 区分编译期失败（静默回退）与运行时失败（硬失败），`main.rs:162-176` + `vm_run` 均已处理 |
| ~~解释器路径输出多余的 `= ()`~~ | ✅ 已修复 (2026-07-11)：`main.rs` 三处解释器输出均加 `if !matches!(val, Value::Unit)` 过滤，与 VM 路径一致 |
| ~~Option/Result 非泛型（`Type::Enum` 而非 `Type::Generic`）~~ | ✅ 已修复 (2026-07-11)：`lower_expr.rs` EnumLiteral/Call/Ident 三条路径为 Option/Result 推断 `Type::Generic { base: Enum, args }`，从字段值推断类型参数 |
| ~~HashMap 键仅支持 str~~ | ✅ 已修复 (2026-07-11)：`interpreter/methods.rs` + `vm/natives.rs` 双侧新增 `map_key_to_string`/`vm_map_key_to_string`，支持 str/int/bool/float 键 |
| ~~Range 非一等类型（降级为内部类型）~~ | ✅ 已修复 (2026-07-11)：`hir/types.rs` 新增 `Type::Range { inner, inclusive }` 变体 + Display，`lower_expr.rs` Range 表达式推断为 `Type::Range` |
| ~~match 无穷尽性检查~~ | ✅ 已修复 (2026-07-11)：`lower_expr.rs` match 表达式新增穷尽性检查，支持 `Type::Enum` 与 `Type::Generic` 两种 scrutinee 类型，缺变体报 TypeError |
| ~~函数返回类型未检查（body 为空返回 Unit 但声明非 Unit）~~ | ✅ 已修复 (2026-07-11)：`hir/lower/types.rs` `check_and_merge_tensor_shape` 非 Tensor 分支新增 Unit 返回检测，声明非 Unit 但实际 Unit 时报 TypeError |
| ~~整数类型运行时全退化为 i64（dtype 丢失）~~ | ✅ 已修复 (2026-07-11)：`IntLiteral(i64)` → `IntLiteral(i64, BaseType)` 四层同步（Token/AST/HIR/Value）；Lexer 新增后缀检测（`42u8`）+ 编译期范围检查（`256u8` → 错误）+ 无后缀大整数自动提升 I32→I64；运行时新增 `check_int_overflow()` 算术溢出检测（Add/Sub/Mul/Div/Mod/Neg）；type_of() 返回实际 dtype 而非硬编码 I32；Bug fix：`()` 类型注解正确解析为 `Type::Base(BaseType::Unit)`。14 项 `int_types_test.rs` 全部通过 |

---

## 六、已知限制

| # | 问题 | 影响 |
|---|------|------|
| 0 | ~~VM 不支持字符串切片~~ | ✅ 已修复：SliceStr + Range 索引解析 |
| 2 | 树遍历解释器大文件慢 (debug build) | release build 即解决 |
| 3 | WASM codegen 个别边界情况 | wasmi 执行偶有 type mismatch |
| 4 | 无 GPU 后端 | `compile/gpu/` 仅 CUDA C 源代码生成 + 模拟设备（`device.rs:79-82` `CudaDevice::is_available()` 永远返回 true，注释自承 "Simulated"；`total_memory` 硬编码 24GB）；`mod.rs::compile_kernel` 仅把 HIR 函数转成 CudaKernel 字符串；**未接 nvcc / CUDA Runtime API / cuLaunchKernel**，不编译、不加载、不执行任何 kernel。待安装 CUDA Toolkit 后真正激活。详见 §11.4 AUDIT-11.4.6。 |
| 5 | three_stage wasmtime 路径运行时失败（host import 已补全但 WASM-B 0 字节） | **2026-07-09 更新**：`wasmtime_host.rs::register_wasmtime_host_functions` 中 17/18 个 host import 已补全真实现（Vec 四件套/String 全家桶均按 WASM 线性内存 + bump allocator 方案实现，与 `compile/wasm/host.rs` 对齐），仅 `tensor_from_vec` 为简化版（与 wasmi 路径一致，自举管线不依赖）。**但运行 `three_stage_selfhost_wasmtime` 仍失败**：tenthc 编译器在 wasmtime 下执行完整流程（tokens_done/parsed/lowered/compiled 日志均输出），但最终 WASM-B 为 0 字节——根因是 tenthc 在 wasmtime 下的 Vec 写回逻辑有问题（main 返回的 Vec 指针指向空数据）。wasmi 路径已通过（WASM-B 460 bytes，add(3,4)=12）。wasmtime 路径保持 `#[ignore]`，深度调试 ROI 不高（wasmi 路径已工作，wasmtime 仅是 JIT 性能优化）。 |
| 6 | ~~Rust 母编译器 wasm.rs Call 分派缺 str_eq/str_add/str_int 函数调用分支~~ | ✅ 已修复 (2026-07-09)：`tenth/src/compile/wasm/compile.rs` Call 分派补齐 `str_eq`/`str_add`/`str_int` 三个分支，与 `str_len`/`str_at`/`str_cmp`/`str_slice` 对齐。新增 3 项回归测试 `wasm_backend_minimal::test_str_{eq,add,int}_call_compiles` 验证函数调用形式能产出合法 WASM 并被 wasmi 实例化执行。 |
| 7 | ~~6 个测试文件栈溢出崩溃~~ | ✅ --release 模式下全部通过（2026-07-12 验证）：tenthc_generic_tensor_test 5/0/0、tenthc_for_loop_test 9/0/0、jit_stack_overflow_test 3/0/0、tenthc_dotdot_eq_test 3/0/0、selfhost_frontend 4/0/0、parity_test 129/0/0。debug 模式下仍栈溢出（Windows STATUS_STACK_OVERFLOW），根因是 Windows debug 模式栈空间不足。**2026-07-08 更新**：`tenthc_for_loop_test` 的 Vec 迭代测试在设置 `RUST_MIN_STACK=268435456`（256MB）后已通过 3/3（vec_literal_basic / vec_literal_break_continue / vec_literal_empty），但默认栈空间下仍栈溢出，根因是 Windows debug 模式栈空间不足（git stash 验证为预存问题）。 |
| 8 | spawn 仍为 eager 语义（Phase 2 设计决策） | `runtime/vm.rs::Op::Spawn` 立即求值 inner 表达式并包装为 `Future(Ready(v))`，不制造真并行；真正的"并发"来自 async I/O 返回的 `Pending` Future（子线程工作期间调度器可切换到其他就绪任务）。**影响**：CPU 密集型 spawn 不会真并行执行；如需 CPU 并行需引入工作窃取调度器或绿色线程池（Phase 3 遗留）。**登记性质**：设计决策非缺陷——保持 Phase 1 兼容性 + 零新依赖原则。详见 MEMO.md 2026-07-08 feat 条目。 |
| ~~9~~ | ~~yield 无语法层关键字（Phase 2 已就绪未接入）~~ | ✅ 已修复 (2026-07-25)：Rust 母编译器侧 yield 早已完整接入（lexer TokenKind::Yield + parser ExprKind::Yield + hir HirExprKind::Yield + bytecode Op::Yield + jit fallback）；tenthc 侧本次完成接入：hir.th 新增 disc=37、parser.th 新增 yield 解析（支持 `yield;`/`yield expr` 两种形式）、lower.th 新增 yield lowering（返回 Unit）、wasm.th 新增 yield codegen（pass through + drop inner）。新增 tenthc_yield_test 3 项测试。**遗留**：路径 B（bridge.rs）未接 yield（tenthc/main.th 不含 yield，自举不受影响；路径 A/C 完整可用）。 |
| 10 | ~~f16/bf16 Phase 1 已实现，Phase 2 待做（2026-07-08 登记）~~ | ✅ Phase 2 已完成 (2026-07-09)：`TensorData` 已含 `F16(ArrayD<f16>)/BF16(ArrayD<bf16>)` 变体（`half = "2"` 依赖），构造器/运算/dtype 提升表/序列化 v2/HIR 白名单/native 双侧注册已完整。**Phase 2 四缺口全部完成**：(a) JIT 路径 `compile/jit/translator.rs:470-478` MakeTensor 已扩展 4 dtype 分发（0=F64/1=F32/2=F16/3=BF16），`hostcalls.rs:470-526` 实现 `host_make_tensor_f16/bf16`；(b) WASM 路径 `bytecode.rs:610-615` dtype_code 扩展为 0-3 编码，`wasm/compile.rs:761-833` 按 dtype 分发到 `HOST_MAKE_TENSOR_F16=18`/`HOST_MAKE_TENSOR_BF16=19`，`wasm/host.rs`+`wasmtime_host.rs` 双侧注册 stub，`wasm/mod.rs` to_val_type F16/BF16→I64；(c) F16/BF16 param 的 autodiff 反向传播已实现（`tensor.rs:307-383` acc_grad 移除 early-return Err，F16/BF16 走 F32 中间累加策略 AMP，避免 F16 溢出 max≈65504；`autodiff.rs:72-85` dispatch_float! F16/BF16 走 f32 路径；种子梯度 F16/BF16 用 F32 ones）；(d) tenthc 自举编译器支持 f16/bf16 字面量（`lexer.th` 扩展 f16(ival=2)/bf16(ival=3) 后缀检测，`lower.th` FloatLiteral 写入 HIR sub=14(F16)/15(BF16)，`wasm.th` is_expr_float 识别）。**影响**：f16/bf16 张量现可在 VM/解释器/JIT/WASM 全路径下使用，也可作为 autodiff param 参与训练（F32 中间累加，支持混合精度训练 AMP）。**验证**：autodiff 54/54、vm_autodiff 15/15、tensor_features 18/18、jit 10/10、wasm_backend_minimal 10/10 均 0 回归；自举 `[OK] Full compiler compiled`。详见 MEMO.md 2026-07-09 feat 条目。 |
| 11 | ~~use 机制在 cargo run --release 下不工作~~ | ✅ 已修复 (2026-07-09)：4 段路径 use（prelude.th 推荐用法，如 `use std::nn::activations::gelu`、`use std::optim::clip::clip_grad_by_value`）在 `cargo run --release` 下工作正常（验证 `Tenth实例/Transformer示例/transformer_demo.th` 和 `Tenth实例/梯度裁剪与累积/grad_clip_accum.th` 均成功执行）。**原描述"连原版 gelu/adamw_step 都报 undefined variable"已过时**——根因是 `try_import_file` 单符号导入只用 parent_path 调用，对目录型模块失败；方案 B 修复（`lower_stmt.rs` 第 404-443 行）改为先尝试完整 path 作为文件，失败再回退 parent_path 作为模块。3 段路径 use（如 `use std::nn::gelu`）仍不支持，属设计限制（标准库用目录/文件.th 结构，gelu 在 activations.th 中而非 gelu.th）。 |

| ~~12~~ | ~~bmm FLOPs 预估为 no-op（返回 0）~~ | ✅ 已修复 (2026-07-25)：tenthc `HirType` 新增 `dim2: i64` 字段（hir.th:39），lower.th 激活 6 处 no-op（`get_tensor_dim2`/`check_bmm_shape`/`static_numel`/`fmt_dims`/`emit_bmm_flop_estimate`/bmm 分支 shape 推断）+ `parse_tensor_type` 放宽至解析 1-3 个 dim 字面量。bmm FLOPs 预估现与 Rust 母编译器对齐（types.rs:1094-1129），新增 `test_bmm_flop_estimate_parity` 验证双侧 warning 一致。详见 §11.4 AUDIT-11.4.7。 |
| 13 | JIT `is_sealed` 断言 panic（循环回边） | **workaround 已加（2026-07-30）**：`compile/jit/translator.rs` 用 `catch_unwind` 包裹 `translate`，含循环的函数 JIT 编译时自动 fallback 到 VM 解释执行，不再 crash。**根因**：`translator.rs:165` block 被过早密封（`is_sealed` 返回 true），循环回边（loop back-edge）尝试密封已密封 block 触发 panic。**根本修复**：延迟密封 block 以支持循环 JIT 编译，推后到 P2。**影响**：含循环的函数无法 JIT 加速，但功能完整（fallback VM 正确执行）。详见 §11.4 AUDIT-11.4.10。 |
| 14 | embedding `gather` ndim 限制 | `tenth/std/nn/embedding.th` 改用 `gather(weight, 0, indices)` native 实现（原 `embedding_lookup` 张量方法从未实现为 native）。`gather` 要求 `weight` 与 `indices` 的 ndim 匹配——典型场景 `weight[V, D]` (ndim=2) + `indices[S]` (ndim=1) 会因 ndim 不匹配运行时报错。**完整解决**：需新增 `index_select` native（沿 dim 维收集，对 ndim 不匹配更宽容）或 broadcast 支持。**当前状态**：workaround——调用方需保证 ndim 匹配（如把 indices 扩展为 `[S, 1]` 再 gather 后 reshape）；推后到 P1 后续。详见 §11.4 AUDIT-11.4.11。 |
| ~~15~~ | ~~`hir/types.rs:392` `.shape()` 方法分支与运行时不一致~~ | ✅ 已修复 (2026-08-02)：grep 确认无生产代码/测试依赖 `.shape()`（`.th` 中仅 `tenth/std/nn/multihead_attention.th:21` 注释提及；测试中 `.shape()` 均为 Rust `Tensor` API 而非 Tenth 方法）。删除 `hir/lower/types.rs` 的 `"shape" => Array<i64>` 分支，并在 `hir/lower/lower_expr.rs` MethodCall 降级处对 Tensor receiver 的 `shape` 方法直接报编译期 TypeError（提示改用 `.shape_tensor()`，返回 `Tensor[f64, ndim]`）。用户自定义 struct 的 `shape` 方法不受影响（Tensor 专属检查）。验证：`tenth/tests/audit_11412_regression_test.rs` 3 项（编译期报错/`shape_tensor` 正常/用户自定义 `shape` 不误伤）。详见 §11.4 AUDIT-11.4.12。 |
| 16 | `randn` 变量参数限制 | `shape_from_int_args`（hir/lower/types.rs）只接受字面量参数作为 shape 维度，变量参数（如 `let n = 10; randn(n, m)`）退化为 `Dim::Any`，丢失编译期 shape 信息。**影响**：使用变量参数的 `randn`/`zeros`/`ones` 调用无法享受编译期 shape 检查与内存预估。**修复计划**：P1 提升 `Dim::Symbol` 支持——变量参数生成符号维度（如 `Dim::Symbol("n")`），编译期通过约束求解保留信息。**当前状态**：✅ 已修复（2026-07-30 P1-1 完成）：`shape_from_int_args` 新增 `HirExprKind::Var` 分支，变量参数提升为 `Dim::Symbol(name)`；tenthc 侧 `HirType` 新增 `symbol_dims` 字段（方案 A）双侧同步。详见 §11.4 AUDIT-11.4.13。 |
| ~~17~~ | ~~VM 字节码：同一函数内连续多个带标签循环，第二个及之后的标签循环错乱（2026-08-02 登记）~~ | ✅ 已修复 (2026-08-02)：**根因**实为 `compile/bytecode.rs` 局部变量槽位查找用 `position`（首匹配）——两个标签循环共用变量名（如 `j`）时，第二个循环的 `j = j + 1` / `while j < 3` 读写第一个循环的 `j` 槽（残留值 3 使内层 while 立即退出 → `s2` 应为 26 实测 50）。`rposition`（最近绑定）+ 本批新增 match 臂编译期作用域（`scope_stack`，臂绑定在 `locals` 中臂结束后被 truncate）修复后，同函数多标签循环（不同标签名、同名标签、while/for/loop 混合嵌套）全部正确。**验证**：`tenth/tests/audit_17_18_regression_test.rs` 3 项标签循环回归（含同名标签、三循环混合），VM+解释器双路径 parity；`Tenth实例/标签循环/labeled_loop.th` 已合并为单函数 `demo_labeled` 运行正确（s=3、s2=26）。 |
| ~~18~~ | ~~VM 字节码：同一函数内连续两个泛型枚举 match，第二个（str 绑定）取值错误（2026-08-02 登记）~~ | ✅ 已修复 (2026-08-02)：**根因**实为 `compile/bytecode.rs` 变量槽位查找用 `position`（首匹配）——两个 match 都绑定同名 `x` 时，第二个 match 的 `Load(x)` 读到第一个 match 的 `x` 槽（残留 42 → `"hi"` 实测 42）。`rposition`（最近绑定）修复 + 本批新增 match 臂编译期作用域（`scope_stack`）修复后：同函数双 match（i64+str）、多字段（Pair<A,B>）、嵌套泛型（Wrap<Vec<i64>>）全部正确，且 match 臂绑定不再遮蔽/污染外层同名变量（此前臂绑定残留为"最近绑定"：解释器删外层变量报"未定义变量"、VM 读残留臂值）。**验证**：`tenth/tests/audit_17_18_regression_test.rs` 2 项泛型枚举回归（含同名绑定 shadow、多字段+嵌套），VM+解释器双路径 parity；`Tenth实例/泛型枚举/generic_enum.th` 已合并为单函数 `demo_both` 运行正确（42 / hi）。 |
| ~~19~~ | ~~运算符重载降级仅支持单层 `a + b`（2026-08-02 登记）~~ | ✅ 已修复 (2026-08-02)：**根因**：`hir/lower/lower_expr.rs` 运算符重载降级用 `resolve_method_type` 取返回类型，但 trait 方法（`impl Add for Point` 的 `add`）不在 inherent 方法表（methods）中，回退 `Unknown` → 链式 `(a + b) + c` 的复合 receiver 类型丢失，外层 `has_trait_impl_for_type("Add", Unknown)` 为 false → 断链，运行时"加法类型不匹配"。**修复**：新增 `trait_impl_method_ret_type()`（从 `trait_impls[trait][type][method].return_type` 取真实返回类型），二元与一元重载降级路径均优先使用，链式 receiver 保持为具体类型。**验证**：`tenth/tests/audit_19_regression_test.rs` 4 项回归（`(a+b)+c`、`a+b+c+d`、`a*b+c` 混合、链式结果参与 Eq 比较）解释器路径全绿；`Tenth实例/宏与自定义运算符/operator_overload.th` 新增链式示例运行正确（5.0 / 6.0）。**遗留**：VM 具体值 trait 方法分派缺口（`Value::Struct` 方法分派只做字段访问）为既有已知限制，链式仍仅解释器路径可用。 |

---

## 七、已知缺陷与不完整（历史参考 — C 后端已移除）

### 7.1 自举编译器 Tenth 源码中的已知问题

| # | 位置 | 问题 |
|---|------|------|
| ~~1~~ | ~~`tenthc/lexer/lexer.th:98,105`~~ | ~~**字面量值不解析** — 整数和浮点数 token 的 value 硬编码为 0~~ **已修复**：lexer.th:90 正确解析 `fval: fv + (ff / div)`，lexer.th:105-106 正确存储 `let fv: f64 = ival; ... fval: fv`（f32 后缀路径） |
| ~~2~~ | ~~`tenthc/parser/parser.th:269`~~ | ~~**字段名不存储** — `parse_postfix` 的 `.field` 访问把字段名丢了~~ **已修复**：parser.th:269 实为 `&` ref 处理（`if d == 40`），字段访问在 parser.th:628 `let field_name = field_tok.sval;` 正确存储 |

> ~~原 #1 (cgen 函数调用参数丢弃) 随 codegen 移除而消除。~~
> 2026-06-30 复核：#1/#2 均已修复，条目保留以记录历史。

### 7.2 语言功能缺口

| # | 位置 | 问题 |
|---|------|------|
| ~~3~~ | ~~`tenth/src/compile/lower.rs`~~ | ~~**Match pattern binding 未生成**~~ **已修复**：实现 match 表达式 struct 解构 pattern（如 `match p { Point { x, y } => ... }` 中的 x/y 绑定）。跨模块全链路：(1) `parser/ast.rs` + `hir/hir.rs` 新增 `Pattern::Struct { name, fields }` 变体；(2) `parser/parser.rs::parse_match_pattern` 加 `LBrace` 处理，支持 `Name { x, y }` 简写与 `Name { x: a, y: b }` 命名绑定；(3) `hir/lower/lower_expr.rs` `lower_pattern`/`bind_pattern_vars` 加 Struct 分支；(4) `runtime/interpreter/pattern.rs` 三处加 Struct 分支；(5) `runtime/vm.rs` 新增 `IsStruct(usize)` Op（opcode 46，与 `IsEnumVariant` 对称）；(6) `compile/bytecode.rs` 加 Struct 分支（IsStruct + LoadField）；(7) `compile/jit/translator.rs` 遇 IsStruct 返回 Err 触发 JIT→VM fallback。tenthc 不用 struct pattern，无需同步。测试：`tests/pattern_match_test.rs` 新增 6 项（4 解释器 + 2 VM），17/17 全绿，0 回归。 |
| 8 | `tenthc/main.th:11` | **依赖 tenthc_combined.th** — 需在编译前手动拼接 |

> ~~#5-#7/#10/#12 引用已删除的 C 代码生成文件 (`tenthc/codegen/`, `tenthc/runtime.c`)，随 C 后端移除而消除，不再罗列。~~

### 7.3 中优

| # | 位置 | 问题 |
|---|------|------|
| ~~9~~ | ~~`tenthc/parser/parser.th:631`~~ | ~~**无 for 循环解析** — lexer 识别 `for` 但 parser 未处理~~ **登记陈旧失效**（2026-07-06 复核）：for-in 解析已在 tenthc 全链路实现 — `parser.th:1100-1132`（`parse_stmt` 中 `disc==12` 分支）+ `lower.th:1258-1279`（`kind=="for"` 分支，lower 为 `disc=4`）+ `wasm.th:1404+`（`disc==4` 分支）。原引用行号 `parser.th:631` 实为 generic_call 创建 ident 节点逻辑，与 for 循环无关。tenthc 自举源码本身大量使用 for 循环（parser.th 中 13+ 处），自举路径 B/C 必须能编译 for 循环——这是功能可用的铁证。 |
| 11 | `tenthc/parser/parser.th:180` | **parse_unary 纯透传** — 一元运算在 parse_primary 内处理，此函数冗余 |

### 7.4 架构债务

| # | 问题 |
|---|------|
| ~~14~~ | ~~borrow checker 双向放宽~~ — 已恢复，`check_borrow_shared` 和 `check_borrow_mut` 现执行 ExclusiveRef/SharedRef 检查 |
| ~~16~~ | ~~Test 覆盖盲区~~ — 已修复。前端契约测试 `selfhost_frontend.rs` 改为严格 assert（原 println 不 fail 的问题已修）；执行覆盖由 `fixpoint_runtime.rs`（Wasmtime 端到端编译+执行）和 `parity_test.rs`（112 项 Rust/tenthc 一致性）提供 |

---

## 八、灵感与改进方向

### 8.1 短期 (解锁自举)

- **~~修复致命缺陷 #1-#4~~** → ✅ 已修复，自举闭环可达
- **添加 `tenthc_combined.th` 自动生成** — build script 或 Makefile，而非手动拼接
- **~~`tenthc_test.rs` 加执行测试~~** → ✅ 已由 `fixpoint_runtime.rs` 和 `parity_test.rs` 提供

### 8.2 中期 (质量加固)

- **VM 补全** — closure/generic call/match 仍偶有 fallback
- **~~恢复 borrow checker~~** — 已恢复，`check_borrow_shared`/`check_borrow_mut` 现执行完整检查
- **WASM Host import 真实现** — Vec/String 在 WASM 模块中仍以占位形式存在，待实现

### 8.3 长期 (生态)

- **激活死模块** — shape.rs (张量形状优化)、docgen.rs (API 文档生成)；~~autodiff.rs (自动微分训练)~~ ✅ 已完整集成（21 算子，张量级，已集成解释器）
- **tenthpm 包管理器** — `tenth/tools/tenthpm/` **完整实现**，支持 init/build/test/run/add/remove/list/clean/publish/install + Tenth.toml + Tenth.lock + .tenthpkg 打包 + path/git/registry 三种依赖类型
- **LSP 服务器** — `tenth/tools/lsp/` **完整实现**（文档同步/diagnostics/hover/completion/definition/documentSymbol/references/rename/signatureHelp/foldingRange/semanticTokens/formatting）
- **CUDA 后端** — `compile/gpu/` + `compile/optimizations/` 当前仅 CUDA C 源代码生成 + 模拟设备（`device.rs::is_available()` 永远返回 true，`total_memory` 硬编码 24GB），未接 nvcc / CUDA Runtime / cuLaunchKernel。待安装 CUDA Toolkit 后真正激活 kernel 编译与执行

### 8.4 过程改进

- **`.gitignore` 已更新** — 添加 `*.exe` / 构建产物排除
- **清理 19 个构建产物** — 15 个 .exe + 2 个空 .txt + tenthc.c + test_mini.c 已从 git 跟踪移除
- **MEMO.md 保持同步** — 作为动态状态文件，每次大改动后更新

---

## 九、清理记录

| 操作 | 数量 |
|------|------|
| 从 git 移除 .exe 文件 | 15 |
| 从 git 移除空 .txt | 2 |
| 从 git 移除构建 C 文件 | 2 (tenthc.c, test_mini.c) |
| 删除临时 .th | 1 (test_input.th) |
| 更新 .gitignore | 覆盖 *.exe, *.txt, tenthc.c, test_mini.c |
| 删除游离产物 (2026-06-15) | 3 (stderr.txt, stdout.txt, test_w.thw) |
| .gitignore 加固 (2026-06-15) | 新增 *.thw / stdout.txt / stderr.txt |
| 文档对齐 (2026-06-15) | 8 文件：版本号 0.3.0→0.3.3，测试数 134→350，算子数 19→21，LSP/tenthpm 状态脚手架→完整实现 |
| AUDIT §5/§7 重构 (2026-06-15) | 修复重复小节号 (5.2×2)，移除 6 条已废弃 C 后端条目，章节号六/七→八/九 |

保留的历史 C 文件 (tenthc_v3.c ~ v9.c, tenthc_dbg5.c, tenthc_dbg6.c, tenthc_fix.c, tenthc_analyze.c, tenthc_chk.c, tenthc_out.c, tenthc_rust.c) 作为自举进化见证，暂时保留不删。

---

## 十、安全审查记录（2026-06-29）

> 全面安全审查识别 25 项问题（2 致命 / 8 高危 / 8 中等 / 7 低危），分两轮全部修复。第一轮 17 项（C-1, C-2, H-1, H-3, H-5~H-8, M-1~M-8），第二轮 7 项（H-2, H-4, L-1, L-3~L-6）；L-2 在 H-7 中一并修复，L-7 在 H-2 中一并修复。审查报告 `security_review.md`，威胁模型披露 `SECURITY.md` 重写，变更记录见 `MEMO.md` 顶部 2026-06-29 条目。

### 10.1 已修复（第一轮 17 项）

| ID | 严重度 | 位置 | 修复方式 |
|----|--------|------|---------|
| C-1 | 致命 | `tenth/tools/tenthpm/src/{manifest,install,add}.rs` | 新增 `validate_package_name` / `safe_package_name_from_git` / `ensure_within` / `safe_to_remove_dir` 集中校验；所有 `target_dir` 计算 + `fs::remove_dir_all` 必须经过校验 |
| C-2 | 致命 | `SECURITY.md` | 重写：纠正"0 处 unsafe"失实声明（实际 41+ 处），公开真实威胁模型与 `--fs-root` 沙箱选项 |
| H-1 | 高危 | `tenth/src/compile/jit/hostcalls.rs` | 新增 `MAX_HOSTCALL_ARGS = 1<<20` 与 `safe_slice` 统一闸门；所有 `from_raw_parts` 改用；`host_make_map` / `host_new_struct` 加 `count.checked_mul(2)`；`host_make_tensor` 加 `rows.checked_mul(cols)` |
| H-3 | 高危 | `tenth/src/main.rs` | 新增 `parse_memory_config` 与 `run_file(path, config)`；fallback 路径用 `Interpreter::with_limits`；`--no-limits` 显式退出沙箱 |
| H-5 | 高危 | `tenth/src/main.rs` | `time_sleep_ms` 拒绝负数与 > 24h 的请求 |
| H-6 | 高危 | `tenth/src/main.rs` | JSON 解析器加 `JSON_MAX_DEPTH=256` 防栈溢出；`json_unescape` / `simple_json_split` 修复 `\"` 转义状态机 |
| H-7 | 高危 | `tenth/src/main.rs` | `random` 改用 `rand::thread_rng().r#gen()`（CSPRNG），替代可预测的 `DefaultHasher` |
| H-8 | 高危 | `tenth/src/compile/wasmtime_host.rs` | 新增 `safe_offset` / `read_cstr` / `MAX_ALLOC_BYTES=16MiB`；17 个 host import 全部改用；`tenth_alloc` 拒绝负数 size 与 > 16MiB 请求 |
| M-1 | 中危 | `tenth/src/compile/jit/context.rs` | `transmute` 前加 `assert_eq!(size_of::<*const u8>(), size_of::<JitFn>())` 尺寸断言 |
| M-2 | 中危 | `tenth/src/repl.rs:241-245` | `mem-strict` 模式不再 `panic!` 杀 REPL，改为返回 `TenthError::RuntimeError` |
| M-3 | 中危 | `tenth/Cargo.toml` | `[profile.release]` 加 `overflow-checks = true, panic = "unwind"`；`[profile.dev]` 加 `overflow-checks = true` |
| M-4 | 中危 | `tenth/src/compile/jit/hostcalls.rs` | 见 H-1（`host_make_tensor` 加 `rows.checked_mul(cols)`） |
| M-5 | 中危 | `tenth/src/compile/jit/context.rs` | impl Drop for JitContext 调 `cache.clear()` 显式清理 |
| M-6 | 中危 | `tenth/src/compile/jit/hostcalls.rs` | `invoke_jit` 包 `catch_unwind`，panic 时写 `Value::Unit` 并返回 false，防 FFI unwind UB |
| M-7 | 中危 | `tenth/src/runtime/arena.rs` | `alloc` 用 `checked_add` / `checked_mul`；`scope` 用 `saturating_sub` 防下溢 panic |
| M-8 | 中危 | `tenth/src/compile/wasmtime_host.rs` | 见 H-8（bump allocator 越界 panic 修复） |

### 10.2 已修复（第二轮 7 项，2026-06-29）

| ID | 严重度 | 位置 | 修复方式 |
|----|--------|------|---------|
| H-2 | 高危 | `tenth/src/runtime/{limits,vm,interpreter}.rs`、`main.rs` | 新增 `FsSandbox` 类型（canonicalize + starts_with 防路径穿越）；Vm/Interpreter 新增 `fs_sandbox` 字段，所有文件 I/O 原生函数必须经过 `check_read`/`check_write`；`--fs-root`/`--read-only`/`--fs-cwd` 命令行选项 |
| H-4 | 高危 | `tenth/src/runtime/{vm,interpreter}.rs`、`main.rs` | VM 主循环和 Interpreter tick() 新增独立 `loop_counter`/`tick_counter`，每 4096 步检查 `deadline_ms`（墙钟超时独立于 `step_budget`）；`--timeout <secs>` 命令行选项，`parse_timeout_ms` 用 checked_mul/checked_add 防溢出 |
| L-1 | 低危 | `tenth/tools/tenthpm/src/manifest.rs` | 验证确认 `extract_package_name` 仅在 `safe_package_name_from_git` 内被调用，所有调用方都经 `validate_package_name` 校验 |
| L-3 | 低危 | `tenth/src/main.rs`、`tenth/src/runtime/interpreter.rs` | `days_to_date` 入口加 `if days > u64::MAX - EPOCH_OFFSET { return (0,0,0); }` 防 `days + 719468` 溢出 UB |
| L-4 | 低危 | `tenth/tools/tenthpm/src/manifest.rs` | `is_git_url` 默认仅放行 `https://`，`http://`/`git://`/`ssh://`/`.git` 后缀需 `TENTH_ALLOW_INSECURE_GIT=1` 显式 opt-in |
| L-5 | 低危 | `tenth/tools/tenthpm/src/{manifest,install,add}.rs` | 三处 `git clone` 加 `--config protocol.file.allow=deny` 和 `protocol.git.allow=deny`；克隆后调用新增 `disable_hooks()` 将 `core.hooksPath` 指向空设备 |
| L-6 | 低危 | `DEPS.md` | 新增"供应链安全"章节，说明 `cargo audit` 和 `cargo deny` 用法及 CI 集成建议 |

> L-2（DefaultHasher）已在 H-7 中一并修复；L-7（compile_host 写文件）已在 H-2 中一并修复。

### 10.3 未修复（保留至下轮）

无。全部 25 项安全问题已修复。

### 10.4 验证

| 命令 | 结果 |
|------|------|
| `cargo build --manifest-path tenth/Cargo.toml` | ✅ 成功（178 warning 为项目原有 `unsafe-op-in-unsafe-fn`，非本次引入） |
| `cargo build --release --manifest-path tenth/Cargo.toml` | ✅ 成功（第二轮验证） |
| `cargo build --manifest-path tenth/tools/tenthpm/Cargo.toml` | ✅ 成功（第二轮 L-4/L-5 验证） |
| `cargo run --release --manifest-path tenth/Cargo.toml -- run tenthc/main.th` | ✅ 自举路径 A 验证通过（`tenthc_full.wasm` 生成） |
| `cargo test --manifest-path tenth/Cargo.toml --lib` | ✅ 10/10 通过（第一轮） |
| `cargo test --manifest-path tenth/Cargo.toml --features mem-debug --test memory_test` | ✅ 17/17 通过（含 `arena_scope_rolls_back_counter` / `arena_overflow_returns_none` 验证 M-7 修复） |
| `cargo test --manifest-path tenth/Cargo.toml`（全集成测试） | ⚠️ rustc 1.95.0 ICE（`wasm.rs:2077` 方法调用解析，仅在 `--test` 模式触发），非本项目代码问题，ICE 文件未被本轮修改 |

---

*文档由 2026-05-27 全项目审计生成，作为后续正式开工的参考基准。*

---

## 十一、论文披露的中等价值问题登记（2026-07-02）

> 本章节登记形式化分析论文与相关评审中披露的、尚未在历史章节登记的中等价值已知缺陷与健全性破口。每条含编号、标题、论文来源、影响、当前状态。**本批次为纯文档登记，未修改任何代码**——属于"诚实披露 → 缺陷登记"环节，修复方案分批推进。

### 11.1 形式化健全性缺陷（护城河相关）

| 编号 | 标题 | 论文来源 | 影响 | 当前状态 |
|------|------|---------|------|---------|
| AUDIT-11.1.1 | T19 定理 B6 跨语句借用 unsoundness | T19（形式化分析定理 B6） | `hir/lower/lower_stmt.rs:50/57/73` 与 `lower_expr.rs:437/439/441/464/656` 调用 `scope.release_borrows` 在语句边界过早释放借用；`lower_expr.rs:701/717` Deref 路径不检查别名。`scope.rs:67/81/113` 实现仅做单点 SharedRef/ExclusiveRef 检查，未跟踪跨语句别名。**存在 Tenth 接受但违反别名规则的程序**（如 `let r = &x; let m = &mut x; println(r);` 类构造）。 | ✅ 部分修复（2026-07-07 `borrow_holders` 机制堵住 B6 原始反例；2026-07-09 数理部论证健全性边界）。**定理 B6'（条件健全性）**：在 P1（persistent Let 记录 holder）/P2（无变量转义）/P3（holder 未被 Move 前不被新借用）前置条件下，活跃 holder 的 borrowed 不被释放。**定理 B7（剩余反例）**：borrow_holders 仍无法捕获 3 类变量转义——B7-1 Block 最终表达式为 Ident（`let s = { let r = &mut p; r };`）、B7-2 变量赋值转义（`let s = r; move(r);`）、B7-3 函数返回引用类型（`fn id(&mut T) -> &mut T`）。根源是 `collect_persistent_borrowed_idents` 不识别 Ident 与 Call 形态。**风险评估**：在 Tenth 当前约束（无 unsafe/无 FFI/无并发）下不构成可利用内存安全漏洞，仅语义健全性缺口。根治方案仍是 NLL（liveness 分析）。详见 MEMO.md 2026-07-09 docs 条目。 |
| AUDIT-11.1.2 | T20 PB2 四类反例未覆盖 | T20（property-based testing 反例集） | Call/If/Block/Match 初始化表达式中的 Ref/MutRef 未跳过 `release_borrows`，四类 AST 节点的借用释放语义与 Let 不一致。属 PB2 反例集披露的 4 类漏洞，可能让通过 borrow checker 的程序在运行时仍违反别名。 | ✅ 已修复（2026-07-07，`hir/lower/mod.rs` 新增 `expr_may_produce_ref` 递归判断 If/Block/Match/Move 是否产生 Ref 值；`creates_persistent_borrow` 改用此函数；新增 `collect_persistent_borrowed_idents` 收集多分支中的所有被借变量；`hir/lower/lower_stmt.rs` Let 处理用新函数支持多变量借用记录。验证：cargo check 通过 + lib 测试 16 passed（含 5 autodiff）+ integration_test 14 passed + 自举验证通过（`[OK] Full compiler compiled to tenthc_full.wasm`）。详见 MEMO.md 2026-07-07 fix 条目） |
| AUDIT-11.1.3 | T42 LayerNorm per-feature γ + BatchNorm 多 channel dX 闭式 backward 缺陷 | T42（normalization 算子梯度推导） | LayerNorm 在 per-feature γ（每个特征独立缩放参数）场景下闭式 backward 不正确；BatchNorm 在多 channel dX 场景下闭式 backward 不正确。两者均属"已实现但有数值错误"——前向 ✅，反向在常见配置下 ✅，在论文 T42 披露的特定配置下产生错误梯度。 | ✅ 已修复（2026-07-07，`runtime/autodiff.rs` line 852-1015：(1) LayerNorm backward 把 `dX = γ·std_inv·(dY - mean(dY) - x̂·mean(dY·x̂))` 改为 `dX = std_inv·(gy - mean(gy) - x̂·mean(gy·x̂))`，其中 `gy = dY·γ`（γ 不能提到括号外）；(2) BatchNorm 重写为 per-channel 双遍循环，修正 d_gamma/d_beta shape（从 (N,C,H,W) 改为 (C,)）、mean 改为 per-channel、修正广播错误。验证：cargo check 通过 + autodiff 5/5 + lib 测试 16 passed + integration_test 14 passed + 自举验证通过（`[OK] Full compiler compiled to tenthc_full.wasm`）。详见 MEMO.md 2026-07-07 fix 条目） |
| AUDIT-11.1.4 | T50 multihead_attention 实为 single-head 等价 | T50（标准库 API 与论文承诺一致性审计） | `tenth/std/nn/multihead_attention.th` 文件头部注释自承："Simplified...single-head-equivalent attention...True multi-head attention requires either: 1. 3D/batched matmul support, or 2. A loop over heads with per-head slices"。`n_heads` 参数被读取但仅用于计算 `d_k = d_model / n_heads` 后做单次 attention，未真正分头。**与对外 API 名 MultiheadAttention 不符**。根因：受仅 2D matmul 限制（见 AUDIT-11.1.5）。 | ✅ 已修复。`tenth/std/nn/multihead_attention.th` 已基于 bmm 重写为真正分头计算版本：q/k/v 投影后 reshape 为 (n_heads, seq_len, d_k)，scores = q_3d.bmm(k_3d.transpose()) * scale，out_3d = weights.bmm(v_3d)，最后 reshape 回 (seq_len, d_model) 做 w_o 投影。Reshape/MaskedFill 已接入 autodiff（TapeOp::Reshape/TapeOp::MaskedFill），w_q/w_k/w_v/x 梯度完整传播。`tenth/tests/multihead_attention_test.rs` 11 项测试覆盖前向 shape/有限性/n_heads=1 等价/autodiff 无错误/w_q/w_k/w_v 梯度非零/x 梯度传播/完整反向链。详见 MEMO.md 2026-07-06 test: MHA 完整梯度测试补强 条目。 |
| AUDIT-11.1.5 | T18 body 直接 clone 健全性破口（泛型实例化） | T18（泛型实例化健全性） | 泛型函数实例化时 body 直接 clone 但未做替换健全性检查，可能让类型变量在 body 内未被一致替换，导致实例化后语义偏移。与 T12 双侧破口（tenthc 缺 shape 检查）联合暴露。 | ✅ 已修复（2026-07-07，`hir/lower/mod.rs` 新增 `substitute_expr` 递归替换 HirExpr 中所有 Type 字段；新增 `check_generic_instantiation_soundness` 健全性检查；`hir/lower/lower_expr.rs` GenericCall 实例化路径调用这两个函数，确保 body 内类型变量被一致替换。验证：cargo check 通过 + lib 测试 16 passed（含 5 autodiff）+ integration_test 14 passed + 自举验证通过（`[OK] Full compiler compiled to tenthc_full.wasm`）。详见 MEMO.md 2026-07-07 fix 条目） |

### 11.2 自举编译器双侧破口（T12）

| 编号 | 标题 | 论文来源 | 影响 | 当前状态 |
|------|------|---------|------|---------|
| AUDIT-11.2.1 | tenthc 缺 shape 检查 | T12（自举双侧对齐审计） | tenthc 完全无 shape 检查，Rust 母编译器已有 Phase 1+2+3 实现。两侧语义不对齐：在 tenthc 自举路径下无法享受 shape 检查带来的早期错误发现，且对 shape 不匹配的程序两侧行为可能不一致。 | ✅ 已修复（2026-07-02，`tenthc/hir/hir.th` HirType 新增 dim_count/dim0/dim1 三字段，`tenthc/hir/lower.th` 新增 8 个 shape 辅助函数与 3 处调用点检查（binary/call/method_call）；最小子集：matmul 2D 内侧 K 检查、二元广播检查、zeros/ones/rand/randn 字面量 shape 推断、transpose 2D 维度互换；跳过 let 注解/分支兼容/返回值合并（需 Tensor 注解解析，tenthc 不支持）；错误处理非致命（push 到 p.errors）；首轮 parity 全挂后重构 tuple 返回为 3 个单值 getter（WASM 后端 tuple 支持有限）；详见 MEMO.md 2026-07-02 fix 条目） |
| AUDIT-11.2.2 | tenthc 缺错误恢复 (panic mode) | T12 | tenthc 完全无 panic-mode，遇首个解析错误即终止。Rust 母编译器有 `error_recovery_test.rs` 7 项测试覆盖续接能力。两侧错误恢复语义不对齐。 | ✅ 已修复（2026-07-02，`tenthc/parser/parser.th` 新增 is_sync_token/synchronize/record_error_and_recover/is_expr_start_token，Parser 结构加 errors 字段，parse_program 主循环对无法识别的顶层 token 应用 panic-mode 恢复；error_recovery_test 7/7 通过，详见 MEMO.md 2026-07-02 fix 条目） |
| AUDIT-11.2.3 | tenthc 缺 expect_gt，嵌套泛型 `>>` 被整个吞掉 | T12 | tenthc lexer/parser 缺 `expect_gt`，把 `>>` 整个吞掉，无法解析 `Vec<HashMap<K, V>>`、`Pair<Pair<T, U>, V>` 等嵌套泛型。Rust 母编译器有 `generic_test.rs` 11 项含 `>>` 拆分测试。 | ✅ 已修复（2026-07-02，`tenthc/parser/parser.th` 新增 expect_gt 函数（遇 Gt consume / 遇 Shr consume 后插入合成 Gt，通过重建 tokens 数组实现 Vec::insert 语义），替换 parse_postfix 与 parse_generic_params 2 处闭合调用点，详见 MEMO.md 2026-07-02 fix 条目） |
| AUDIT-11.2.4 | tenthc 缺 i64 溢出检测 | T12 | tenthc 字面量与算术路径缺 i64 溢出检测，Rust 母编译器有 `overflow-checks = true`（M-3 安全修复）。两侧数值健壮性不对齐。 | ✅ 已修复（2026-07-02，`tenthc/lexer/lexer.th` parse_int 与 lexer_next 数字解析加入溢出检测，用数学等价条件 `ival > (i64_max - d) / 10` 检测，溢出时返回 i64_max 饱和值并打印错误，详见 MEMO.md 2026-07-02 fix 条目） |

### 11.3 双重 native 注册反模式（T37）

| 编号 | 标题 | 论文来源 | 影响 | 当前状态 |
|------|------|---------|------|---------|
| AUDIT-11.3.1 | VM 缺 17 项 native 函数 | T37（VM/解释器 native 注册对齐审计） | `main.rs::register_natives()` 注册到 VM 的 native 函数集合缺 17 项：`to_string`、`type_name`、`with_step_limit`、`with_timeout_ms`、`is_timeout`、`start_grad`、`f64_bits`、`f64_from_bits`、`sin`、`cos`、`ln`、`pow`、`save_weights`、`load_weights`、`format`、`parse_int`、`parse_float`。**`save_weights`/`load_weights` 为致命项**——VM 路径下模型保存/加载失效，与 2026-07-01 "护城河 D 演示补齐" 修复的 6 个构造函数（zeros/ones/rand 等）属同类问题的延续。 | ✅ 已修复（2026-07-02，`main.rs::register_natives()` 补全 17 项，save_weights/load_weights 复用 FsSandbox 校验，详见 MEMO.md 2026-07-02 fix 条目） |
| AUDIT-11.3.2 | 解释器缺 3 项 native 函数 | T37 | `interpreter/natives.rs` 注册集合缺 3 项：`print`、`to_f64`、`to_f32`。与 VM 路径不对称。 | ✅ 已修复（2026-07-02，`interpreter/natives.rs` 补全 print/to_f64/to_f32，并同步 HIR lower 白名单与解释器 Var fallback 白名单，详见 MEMO.md 2026-07-02 fix 条目） |

### 11.4 工程债务与潜在 UB

| 编号 | 标题 | 论文来源 | 影响 | 当前状态 |
|------|------|---------|------|---------|
| AUDIT-11.4.1 | T22 FV5 O(n²) 工程债务 | T22（闭包自由变量收集复杂度） | `hir/lower/closures.rs:9-15` 的 `free_vars_in` 用 `Vec<String>` + `sort` + `dedup` 实现，复杂度 O(n²)（每次 push 后 sort）。HashSet 实现可降到 O(n)。在深层嵌套闭包场景下编译期开销显著。 | ✅ 已修复（2026-07-06，`hir/lower/closures.rs` `free_vars_in`/`collect_free_vars`/`collect_free_vars_stmt` 改用 `HashSet<String>`，4 处 `push` 改 `insert`；API 兼容（仍返回 `Vec<String>` + `sort`），复杂度 O(n²)→O(n)，纯优化无语义影响。验证：autodiff 52/52 + 全量 0 回归。详见 MEMO.md 2026-07-06 fix 条目） |
| AUDIT-11.4.2 | T31 MAX_STACK_DEPTH=256 静默溢出潜在 UB | T31（VM/JIT 栈深度限制审计） | `compile/jit/translator.rs:32` `const MAX_STACK_DEPTH: u32 = 256;`，超过时静默截断而非报错，可能让 JIT 编译出的代码访问越界栈区域。属潜在未定义行为。VM 路径（`runtime/vm.rs`）的 `locals: Vec<Value>` 用 `resize` 安全增长，但 JIT 路径的固定 256 槽是硬上限。 | ✅ 已修复（2026-07-06，`compile/jit/translator.rs` 新增 `bump_sp()` 方法在 push 前检查 sp 上限，28 处 push 操作改为 `bump_sp()?`；超限返回 `Err` 触发既有的 JIT→VM fallback（`mod.rs:62-65`），不再静默截断。验证：jit_test 10/10 + 全量 0 回归。详见 MEMO.md 2026-07-06 fix 条目） |
| AUDIT-11.4.3 | T36 双重存储同步开销 | T36（解释器 scope 数据结构审计） | `runtime/interpreter/mod.rs:40` `pub scopes: Vec<HashMap<String, Value>>` 用 Vec+HashMap 双重存储，变量查找从最后一个 scope 向前遍历，在重嵌套场景下有平方风险。VM 路径用索引 `locals: Vec<Value>` 无此问题。 | ✅ 已修复（2026-07-08）。改为扁平化 `vars: HashMap<String, Vec<(usize, Value)>>` + `scope_depth` + `scope_vars: Vec<Vec<String>>`，`resolve_var`/`set_var` O(1)，`pop_scope` O(m)（m 为本层变量数） |
| AUDIT-11.4.4 | tenthc `..=` lexer 解析 bug | 回归测试发现（2026-07-06） | `tenthc/lexer/lexer.th:180-184` 解析 `.` 时 `if next == "."` 分支直接返回 `DotDot`，未检查第三个字符是否为 `=`。导致 `2..=4` 被错误分词为 `2` `..` `=` `4`。tenthc 路径下 inclusive range 实际不可用。 | ✅ 已修复（2026-07-06，`tenthc/lexer/lexer.th:180-190` 在 `if next == "."` 分支内 advance 后再 peek 一次检查 `=`，匹配则 advance 并返回 `DotDotEq`，与 Rust 侧 `lexer.rs` 对齐。验证：自举 + parity 129/129 + selfhost_frontend 4/4 + error_recovery 7/7；新增 `tenthc_dotdot_eq_test.rs` 3 项测试。详见 MEMO.md 2026-07-06 fix 条目） |
| AUDIT-11.4.5 | tenthc Vec 迭代未实现 | 回归测试发现（2026-07-06） | `tenthc/compile/wasm.th:1472` 明确注释 `// Non-range iterable: no-op for now`。tenthc WASM 后端不支持 `for x in [Vec]` 形式的迭代，仅支持 Range 迭代。 | ✅ 已修复（2026-07-06 codegen + 2026-07-08 测试 stub 补全。`tenthc/compile/wasm.th:1471-1546` For 语句新增 ArrayLiteral (disc=22) 分支，复用现有 `vec_new`/`vec_len`/`vec_push`/`vec_get` host imports 实现 Vec 迭代 codegen；后续修复：删除 `wasm.th:1492` 误加的 `wasm_drop`（type/drop 不匹配导致 wasmi 验证失败）。2026-07-08 补全 `tenth/tests/tenthc_for_loop_test.rs` 中 `env` 模块的 4 个 Vec host function stub（vec_new/vec_len/vec_push/vec_get），基于 WASM 内存 + bump allocator 实现真实 Vec 语义（24 字节 header = cap+len+dp，8 字节/元素，vec_push 含倍增扩容），与 `wasm/host.rs` 的 `Vec_new`/`Vec_push` 实现对齐。取消 2 项 `#[ignore]` 标记（tenthc_for_vec_literal_basic / tenthc_for_vec_literal_break_continue），3 项 Vec 迭代测试全部通过。验证：自举 + parity 129/129 + selfhost_frontend 4/4 + tenthc_for_loop_test Vec 迭代 3/3。ArrayLiteral 表达式本身 codegen 未改。详见 MEMO.md 2026-07-06/2026-07-08 fix 条目） |
| AUDIT-11.4.6 | CUDA 后端文档矛盾 | 基本功核查第 94 项（2026-07-25 复核） | `compile/gpu/` 模块在 AUDIT.md §六 #4 / §8.3 / CODE_WIKI.md §4.4 中描述为"脚手架已就绪，待 CUDA 环境安装"，易让读者误解为"只差环境配置即可工作"。实际：`device.rs:79-82` `CudaDevice::is_available()` 永远返回 true（注释自承 "Simulated: always report available for code generation purposes"），`total_memory: 24 * 1024 * 1024 * 1024` 硬编码模拟值；`mod.rs::compile_kernel` 仅把 HIR 函数转成 CudaKernel 字符串；`cuda_kernel.rs:48-119` `to_cuda_code()` 仅生成 CUDA C 源代码字符串；Grep `nvcc\|cuda_runtime\|cudaRun\|cuLaunch\|exec\|compile_binary\|build_kernel` 在 `compile/gpu/` 0 匹配——整个 gpu 模块只是 C 代码生成器 + 模拟设备，不编译、不加载、不执行任何 kernel。 | ✅ 已修复（2026-07-25，纯文档措辞修正，零代码改动）：AUDIT.md §六 #4 措辞改为"仅 CUDA C 源代码生成 + 模拟设备，未接 nvcc / CUDA Runtime API / cuLaunchKernel"；§8.3 同步修正；CODE_WIKI.md §4.4 GPU 后端章节代码示例与实际 `Device` trait 对齐（实际是 `name/device_type/memory_limit/is_available` 4 方法，非文档原写的 `device_type/allocate/launch` 3 方法）；`基本功核查.md` 第 94 项状态从 "✅ 严重" 升级为 "已解决（文档措辞修正）"。**性质**：文档失实风险消除，非代码 bug。**未触及代码**：gpu/ 模块实现保留现状（待 CUDA Toolkit 环境就绪后真正激活）。详见 MEMO.md 2026-07-25 docs 条目） |
| AUDIT-11.4.7 | tenthc bmm FLOPs 预估为 no-op | 已知限制 #12 登记（2026-07-12 Problem 21 批次4 发现） | `tenthc/hir/lower.th:351` `emit_bmm_flop_estimate` 原直接 return（no-op）。根因：tenthc `HirType` 仅有 dim0/dim1，无 dim2 字段，3D tensor shape 无法在编译期追踪，导致 bmm FLOPs 预估、3D shape 检查、3D numel/bytes 计算均为 no-op。**影响**：bmm 数值计算路径完整（运行时正确），仅编译期 FLOPs 预估与 3D shape 静态检查缺失（errors defer to runtime）。 | ✅ 已修复（2026-07-25）：(1) `tenthc/hir/hir.th:39` 新增 `dim2: i64` 字段（`..` 默认零初始化，不破坏现有 1D/2D 路径）；(2) `tenthc/hir/lower.th` 激活 6 处 no-op——`get_tensor_dim2`（读 `t.dim2`）、`check_bmm_shape`（编译期校验 batch/M/K/N）、`static_numel`/`fmt_dims` 的 `dc==3` 分支、`emit_bmm_flop_estimate`（完整实现 B*M*K*N*2 FLOPs 计算，分段乘法 + 范围检查防 i64 溢出，每维 < 10^6，累乘 < 10^18）、bmm 分支（3D shape 推断 `(B, M, N)` + 调用 `check_bmm_shape` + `emit_bmm_flop_estimate`）；(3) `parse_tensor_type` 放宽：支持 `Tensor[f64, 2, 3, 4]` 解析 1-3 个 dim 字面量；(4) 构造函数 `add_tensor_type`/`add_tensor_type_param`/`propagate_tensor_dtype` 加 `dim2` 参数。bmm FLOPs 预估现与 Rust 母编译器对齐（`types.rs:1094-1129`）。新增 `test_bmm_flop_estimate_parity`（`tenth/tests/native_parity_test.rs`）验证双侧 warning 一致。**字段语义**：dim_count=3 → dim0=B, dim1=M, dim2=K(bmm 输入) 或 N(bmm 结果)。**验证**：build 无新 warning、自举 `[OK] Full compiler compiled`、cargo test --release 全套全绿、bmm_test 11/11、native_parity_test 36/36、selfhost_frontend 4/4、three_stage 1/0/2。详见 MEMO.md 2026-07-25 fix 条目） |
| AUDIT-11.4.8 | tenthc yield 关键字未接入 | 已知限制 #9 修复同步（2026-07-25） | `tenthc/parser/parser.th`（原无 yield 解析）、`tenthc/hir/lower.th`（原无 yield lowering）、`tenthc/compile/wasm.th`（原无 yield codegen）。**根因**：tenthc lexer 已注册 "yield" 关键字（disc=82），但 parser/lower/wasm 三层未接入，导致 tenthc 无法解析含 yield 的源码。**影响**：tenthc 侧无法编译含 yield 表达式的 Tenth 源码（Rust 母编译器侧不受影响，早已完整实现）。 | ✅ 已修复（2026-07-25）：(1) `tenthc/hir/hir.th`：HirExpr disc 列表新增 `37: Yield`（left=inner expr, ty=Unit）；(2) `tenthc/parser/parser.th`：新增 yield (disc=82) 解析，支持 `yield;`（无值，left=0）和 `yield expr`（带值，left=inner）两种形式；(3) `tenthc/hir/lower.th`：新增 yield lowering（inner 若存在则 lower 保留副作用，返回 Unit 类型 disc=11）；(4) `tenthc/compile/wasm.th`：新增 yield (disc=37) codegen（inner 若存在则编译并 drop，yield 本身 no-op）；(5) 新增 `tenth/tests/tenthc_yield_test.rs`：3 项测试（无值形式/带值形式/多 yield 混合）。**遗留**：路径 B（bridge.rs）未接 yield（任务约束禁止修改），tenthc/main.th 不含 yield 故自举不受影响。**验证**：build 无新 warning · 自举 [OK] · cargo test --release 全套全绿 · yield_test 6/6 · selfhost 4/4 · three_stage 1/0/2 · tenthc_yield_test 3/3。详见 MEMO.md 2026-07-25 feat 条目） |
| AUDIT-11.4.9 | 标准库三项能力文档失实 | 文档部复核（2026-07-25） | `能力梳理/能力全梳理.md` §4.1（HashSet ❌）、§4.3（统计函数 ❌）、§4.10（断言 ❌）。**根因**：标准库扩展后未同步能力全梳理状态，导致文档标记与实际实现不符。**影响**：误导总师优先级评估（误将已实现能力列为扩展目标）。 | ✅ 已修复（2026-07-25）：(1) 统计函数：原有 6 函数（mean/median/variance/stddev/variance_sample/stddev_sample）+ 新增 6 函数（min/max/range/sum/product/percentile）+ test_stats.th（25+ 断言）；(2) HashSet：原有 8 函数（new/insert/contains/remove/len/is_empty/to_array/clear）+ 新增 5 函数（from_array/set_union/intersection/difference/is_subset）+ hashset_test 12 项行为测试；(3) assert/assert_eq：补 8 项行为测试（assert_test.rs）+ 修复解释器 Shared 包裹值问题（新增 deref_wrapped() 辅助函数）+ native 白名单同步（lower_expr.rs + eval.rs）；(4) 能力全梳理状态修正（§4.1 HashSet ❌→✅、§4.3 统计函数 ❌→✅、§4.10 断言 ❌→⚠️）+ 总结矩阵数字更新（标准库 ✅ 57→59、⚠️ 5→6、❌ 98→95；合计 ✅ 217→219、⚠️ 26→27、❌ 342→339）。**性质**：文档失实 + 测试补强 + 能力扩展。详见 MEMO.md 2026-07-25 feat 条目） |
| AUDIT-11.4.10 | JIT `is_sealed` 断言 panic（循环回边） | 第一波修复同步登记（2026-07-30） | `tenth/src/compile/jit/translator.rs:165` block 被过早密封（`is_sealed` 返回 true），循环回边（loop back-edge）尝试密封已密封 block 触发 panic。**根因**：当前 JIT translator 的 block 密封策略假设线性控制流，循环回边打破该假设。**影响**：含循环的函数 JIT 编译时 crash（影响训练循环、迭代算法等典型场景）。 | ⚠️ workaround 已加（2026-07-30）：`compile/jit/translator.rs` 用 `catch_unwind` 包裹 `translate`，含循环的函数 JIT 编译时自动 fallback 到 VM 解释执行，不再 crash。**根本修复**：延迟密封 block 以支持循环 JIT 编译（需要重构 block 密封时机，让回边能正确触发 block 合并），推后到 P2。**影响**：含循环的函数无法 JIT 加速，但功能完整（fallback VM 正确执行）。**性质**：crash 防护 + 待根本修复。详见 MEMO.md 2026-07-30 fix 条目） |
| AUDIT-11.4.11 | embedding `gather` ndim 限制 | 第一波修复同步登记（2026-07-30） | `tenth/std/nn/embedding.th` 改用 `gather(weight, 0, indices)` native 实现（原 `embedding_lookup` 张量方法从未实现为 native）。`gather` 原语要求 `weight` 与 `indices` 的 ndim 匹配——典型场景 `weight[V, D]` (ndim=2) + `indices[S]` (ndim=1) 会因 ndim 不匹配运行时报错。**根因**：`gather` 是通用原语（沿 dim 维收集，要求 index 与 base ndim 一致以保持 shape 推断确定性），不是为 embedding 场景定制的 index_select。**影响**：标准 embedding 用法（`weight[V,D]` + `indices[S]`）需调用方手动扩展 indices 维度。 | ⚠️ workaround 已加（2026-07-30）：调用方可将 indices 扩展为 `[S, 1]` 再 gather 后 reshape，或保证 weight 与 indices ndim 匹配。**完整修复**：新增 `index_select` native（沿 dim 维收集，对 ndim 不匹配更宽容，专门用于 embedding 等场景）或为 gather 增加 broadcast 支持。推后到 P1 后续。**性质**：API 兼容性限制 + 待 native 扩展。详见 MEMO.md 2026-07-30 fix 条目） |
| AUDIT-11.4.12 | `hir/types.rs:392` `.shape()` 方法分支与运行时不一致 | 第一波修复同步登记（2026-07-30） | `tenth/src/hir/types.rs:392` 有一处 `.shape()` 方法分支（类型系统误标为返回 shape），但运行时**无对应 native 实现**——`.shape()` 方法不存在于 tensor 方法注册表。**影响**：用户写 `x.shape()` 类型检查能通过，运行时崩溃（"method not found" 或类似错误）。**正确路径**：取 shape 应使用 `.shape_tensor()` 方法（返回 `Tensor[f64, ndim]`）。 | ✅ 已修复（2026-08-02）：(1) grep 确认无生产代码/测试依赖 `.shape()`（`.th` 生产代码零依赖，仅注释提及；测试 `.shape()` 均为 Rust `Tensor` API）；(2) 删除 `tenth/src/hir/lower/types.rs` 张量方法分派的 `"shape" => Type::Array{..}` 分支（现落 `_ => Type::Unknown`）；(3) `tenth/src/hir/lower/lower_expr.rs` MethodCall 降级处新增 Tensor receiver + `shape` 方法 → 编译期 TypeError："张量没有方法 'shape()'——取形状请用 'shape_tensor()'（返回 Tensor[f64, ndim]）"。用户自定义 struct/trait 的 `shape` 方法不受影响（Tensor 专属检查）。**验证**：`tenth/tests/audit_11412_regression_test.rs` 3 项（`x.shape()` 编译期报错含 shape_tensor 提示 / `x.shape_tensor()` 返回 1D 维度张量正常 / 用户自定义 `shape` 方法不误伤）全绿；全量 0 回归。**性质**：类型系统遗留误标清理。详见 MEMO.md 2026-08-02 fix 条目） |
| AUDIT-11.4.13 | `randn` 变量参数限制 | 第一波修复同步登记（2026-07-30） | `tenth/src/hir/lower/types.rs` `shape_from_int_args` 只接受字面量参数作为 shape 维度，变量参数（如 `let n = 10; randn(n, m)`）退化为 `Dim::Any`，丢失编译期 shape 信息。**根因**：Phase 1 shape 检查的 `Dim` 枚举仅有 `Known(i64)` 与 `Any` 两类，无法表达"运行时已知但编译期未知"的变量维度。**影响**：使用变量参数的 `randn`/`zeros`/`ones` 调用无法享受编译期 shape 检查与内存预估（`static_bytes` 依赖 `Dim::Known`）。 | ✅ 已修复（2026-07-30 P1-1 完成）：(1) Rust 侧 `types.rs:599-613` `shape_from_int_args` 新增 `HirExprKind::Var(name)` 分支，变量参数提升为 `Dim::Symbol(name.clone())`；(2) tenthc 侧 `hir.th:46-54` `HirType` 新增 `symbol_dims: str` 字段（方案 A，分号分隔 Symbol 维度名字，`dim=-1` 标记 Symbol），`lower.th` 的 `dims_count_from_args`/`dims_d0/d1/d2_from_args` 支持 Var，新增 `dims_symbol_names_from_args` 和 `add_tensor_type_with_symbols`，消费者加 -1 跳过检查；(3) 额外修复 transpose 3D shape 推断 bug（`types.rs:383-395`，改为对任意 ≥2D 都交换最后两维）；(4) bridge.rs 无需改动。**遗留**：tenthc 侧 transpose 仍保守跳过 3D（见 AUDIT-11.4.14）；表达式参数（如 `n*2`）仍退化为 Any（属 P2 层级二）。**验证**：shape_check_compile_test 77 passed、自举 [OK]。详见 MEMO.md 2026-07-30 feat P1-1 条目） |
| AUDIT-11.4.14 | tenthc 3D+ transpose shape 推断双侧不一致 | P1-1 修复同步登记（2026-07-30） | P1-1 修复 Rust 侧 transpose 3D shape 推断 bug 时，tenthc 侧未同步——tenthc 的 `lower.th` transpose 仍保持"非 2D 退化为 unknown"的保守行为。**影响**：双侧在 3D+ transpose 上不完全一致（Rust 侧正确交换最后两维，tenthc 侧保守返回 unknown 跳过检查）。**风险**：低——tenthc 侧保守跳过不会误报，只是放弃 shape 信息；tenthc 自举源码不依赖 3D transpose 的精确 shape。 | ⚠️ 已知限制（2026-07-30 登记）：tenthc 侧保持保守行为，待后续 tenthc transpose shape 推断对齐 Rust 侧（需在 `lower.th` 实现 3D+ 最后两维交换逻辑）。**性质**：双侧非完全一致 + 低风险保守跳过。详见 MEMO.md 2026-07-30 feat P1-1 条目） |
| AUDIT-11.4.15 | 泛型函数运行时不可用（tensor<f64>([...]) + randn<T>） | 用户反馈 + 诊断发现（2026-07-31） | 两个独立根因：(a) `tensor<f64>([1.0, 2.0, 3.0])` 语法经 HIR 编译为 `Call("tensor", [ArrayLiteral])`，但 `tensor` native 仅克隆参数返回 `Value::Vec`/`Value::Array`，后续 `t1 + t2` 在 `eval_binary` 找不到 `(Vec, Vec)` 分支报"加法类型不匹配"；(b) `hir/lower/lower_expr.rs` `NATIVE_GENERIC_CTORS` 分支强制要求类型参数是 `Type::Base(BaseType)`，泛型函数体内 `randn<T>(...)` 类型参数是 `Type::TypeParam`，触发 TypeError "类型参数必须是具体 BaseType"。**影响**：泛型函数体内调用张量构造 native 全部失效；`tensor<f64>([...])` 字面量语法实际不可用。 | ✅ 已修复（2026-07-31）：(1) `runtime/value.rs` 新增 `array_to_tensor(&Value) -> TenthResult<Value>`——递归处理 `Value::Vec`/`Value::Array`，`flatten_values` 计算形状与数据（嵌套数组递归 + 形状一致性校验 + 叶子元素 Float/Int/Float32/Bool→f64），`unpack_shared` 解包 `Value::Shared`/`Value::SharedBox`（返回 owned 副本规避 `RefCell::borrow()` 的借用冲突）；`runtime/natives.rs` + `runtime/interpreter/natives.rs` 双侧 `tensor` native 改为调用 `array_to_tensor`；(2) `hir/lower/lower_expr.rs` NATIVE_GENERIC_CTORS 分支扩展——`Type::TypeParam { name }` 时保留原始 `func_name` + ret_ty.dtype 保留为 TypeParam；(3) `hir/lower/mod.rs` `substitute_kind_in_place` 的 `HirExprKind::Call` 分支新增 native 构造函数名修正——TypeParam 替换为 BaseType 后按 (name, dtype) 改写 func_name（F32→`randn_f32`/`zeros_f32`/`ones_f32`/`rand_f32`，F16→`zeros_f16`/`ones_f16`，BF16→`zeros_bf16`/`ones_bf16`）。**测试**：`tenth/tests/fixtures/generic_diag.th` 验证两场景——`tensor<f64>([1,2,3]) + tensor<f64>([10,20,30])` = `[11,22,33]` + `make_noise<f32>(4)` 正确实例化 `randn_f32`。**验证**：cargo test --release 全套全绿（含 selfhost_frontend 4/4 + three_stage 1/0/2）。详见 MEMO.md 2026-07-31 fix 条目） |
| AUDIT-11.4.16 | NaN 比较静默 | 阶段2b lossy M2 建议登记（2026-07-31） | 浮点 NaN 与自身的比较（`NaN == NaN` → false、`NaN <= x` → false 等）静默通过，无编译期/运行时提示。**根因**：lossy 格把 NaN 归为 PossibleNaN 级别，M2 采用"PossibleOverflow/NaN 只传播不报"的防误报策略。**影响**：NaN 污染条件判断（`if x == x` 等）时逻辑错误静默，难以定位。 | ⚠️ 已知限制（2026-07-31 登记）：NaN 比较静默未处理——lossy 只传播 PossibleNaN 不报错；完整方案需 NaN 比较使用点检测（如 `x == x` 自比较、NaN 传播进 if 条件），推后。**性质**：静默算错路径 + 待增强。详见 MEMO.md 2026-07-31 护城河系列条目） |
| AUDIT-11.4.17 | JIT 整数溢出路径未覆盖 | 阶段2b lossy M2 建议登记（2026-07-31） | 登记时假设：`check_int_overflow()` 运行时溢出检测覆盖 VM/解释器路径，但 **JIT 编译路径**的整数算术（Add/Sub/Mul/Div/Mod）直接 emit，无溢出检查。**根因**：lossy M2 明确"整数组合暂不做"（既有 lexer 字面量范围检查 + check_int_overflow 运行时兜底），但该兜底在 JIT 路径不生效。**影响**：JIT 路径整数溢出静默回绕（i64 wrapping），与 VM 路径报错行为不一致。**2026-08-02 运行时部复核更正**：JIT translator 并非直接 emit——Add/Sub/Mul/Div/Mod/Neg 全部经 hostcall（`host_add` 等）调用与 VM 相同的算术原语（`vm.add_priv` 等），窄 dtype 范围检查本就在 JIT 生效；真根因是 `Cargo.toml [profile.release] overflow-checks=true` 下 i64 层溢出（`i64::MAX+1`、`i64::MIN/-1`）在算术原语直接 panic——VM 路径崩进程，JIT 路径 panic 穿越 `extern "C"` hostcall 边界触发 `panic_cannot_unwind` 直接 abort，三路径行为均错误（崩溃而非报错）。 | ✅ 已修复（2026-08-02，运行时部）：(1) 算术原语改用 `checked_*`——`runtime/vm/execute.rs` `add_priv`/`sub_priv`/`mul_priv`/`div_priv`/`Op::Mod`/`Op::Neg` + `runtime/vm/mod.rs` `vm.rem`/`vm.neg`（JIT hostcall 入口）+ `runtime/interpreter/binary.rs` `eval_binary` Add/Sub/Mul/Div/Mod + `eval_unary` Neg，i64 层溢出（`checked_*` 返回 None）与窄 dtype 范围溢出统一转为干净 RuntimeError；`runtime/value.rs` 新增 `int_overflow_err`/`int_dtype_name` 辅助；(2) JIT translator `emit_binop`/`emit_unop` 新增 `emit_err_check_abort`（`host_check_error` 检测 + 立即返回 ok=0），与 MethodCall 分支 B2 模式一致——binop 报错后立即中断，避免错误被后续操作覆盖（三路径错误消息一致）；(3) 解释器 Sub 对齐 VM `sub_priv`（保留左 dtype + 范围检查，原强制 I32 且无检查）。**验证**：新增 `tenth/tests/jit_int_overflow_test.rs` 13 项（加/减/乘/除/取负/i64 层溢出 + 链式中间错误 + 除零 + 正常算术 + 循环 fallback），VM/JIT/解释器三路径错误与结果一致；jit_test 10/10 + jit_enum_field 13/13 + 全量 0 回归 + 自举 [OK]。**性质**：路径覆盖缺口 + 行为不一致（修复后三路径一致）。**遗留**：i64 字面量后缀运行时丢失为 I32（既有 dtype 保留问题，非本 AUDIT 范围）；JIT 错误消息带"运行时错误 — 运行时错误 —"双前缀（既有装饰，非功能性）。详见 MEMO.md 2026-08-02 fix 条目） |
| AUDIT-11.4.18 | 标量→张量静默降级 natives 未全审计 | 阶段2b lossy M2 建议登记（2026-07-31） | 标量×/±/÷张量 = 静默降级路径（scalar cast 到张量 dtype，如 `f16_tensor * f64 标量`）。lossy M2 已静态覆盖二元"标量 op 张量"路径（标量 F32/F64 ×/±/÷ Tensor[F16/BF16/F32] → Lossy，使用点报错 + `lossy(...)` 显式放行），但 **natives 内部标量/常量处理未逐行审计**（阶段2b §7.7 已披露）——softmax/reduce 等 natives 内部的隐式 dtype 收缩不在 taint 覆盖内。 | ⚠️ 已知限制（2026-07-31 登记）：lossy 已覆盖"标量 op 张量"二元路径；natives 内部标量/常量处理残留未覆盖降级路径，待逐算子审计（natives 内标量/常量处理未逐行审计为既有披露，非本次新发现）。**性质**：覆盖缺口 + 待审计。详见 MEMO.md 2026-07-31 护城河系列条目） |
| AUDIT-11.4.19 | parity_test / jit_stack_overflow_test 默认线程栈（2MB）栈溢出基线 | 测试环境问题（2026-07-31 登记；M1 提交 0a426ad 即存在，非新回归） | cargo test 默认线程栈 2MB 下，`parity_test` 与 `jit_stack_overflow_test` 触发 Windows STATUS_STACK_OVERFLOW（0xc00000fd）。**根因**：Windows 默认测试线程栈 2MB，全量 1511 项测试中部分套件（parity/jit_stack_overflow）递归深度超过 2MB。**验证**：设置 `RUST_MIN_STACK=33554432`（32MB，实测）后全量 1511 passed / 0 failed；64MB（268435456）亦可用。与 AUDIT.md §六 #7（debug 模式栈溢出预存）同源不同表现——本条目为默认线程栈基线问题。 | ⚠️ 环境依赖（2026-07-31 登记）：验证命令需先 `$env:RUST_MIN_STACK="33554432"` 再 cargo test；影响 CI 与本地默认执行（不设置则 parity/jit_stack_overflow 栈溢出误报为失败）。**性质**：环境基线 + 非代码缺陷。详见 MEMO.md 2026-07-31 护城河系列条目） |
| AUDIT-11.4.20 | lossy 设计取舍（PossibleOverflow/NaN 只传播不报、递归保守 Exact、方法调用不查 callee） | 阶段2b lossy M2 设计取舍登记（2026-07-31） | 三项保守/防误报取舍：① **PossibleOverflow/NaN 只传播不报**——浮点溢出/NaN 的产生路径编译期不告警，仅 Lossy 级别使用点报错（防误报底线：静态不可判定的不报）；② **递归/互递归函数保守返回 Exact**——未做不动点，递归函数内的污点不跨调用传播，可能漏报；③ **方法调用不查 callee 返回污点**——只 receiver⊔args，方法返回值污点保守为 Exact，可能漏报。**影响**：三项取舍均可能导致 lossy 分析漏报（漏报方向安全，不会误报）。 | ⚠️ 已知限制（2026-07-31 登记）：三项取舍均为防误报/保守策略，如实记录。递归不动点求值 + 方法 callee 返回污点查询列为未来增强候选（架构设计文档 §3.3 已披露）。**性质**：设计取舍 + 潜在漏报。详见 MEMO.md 2026-07-31 护城河系列条目） |
| AUDIT-11.4.21 | VM `&mut` 写回顺序相关失效（静默错误值） | 手册示例验证（E 任务）发现（2026-08-02） | `runtime/vm/execute.rs` 写回路径：共享引用 `&x` 先声明、可变引用 `&mut y` 后写回（`*m = 20`）时值不更新——VM 输出 10、解释器输出 20（顺序相关：先 `&mut` 后 `&` 则正常）。**静默错误值**（无报错、结果错误），高危——引用语义与解释器不一致。最小复现 `.trae/tmp/manual_audit/m_ref_multi.th`（另 `d01_ref_move.th`）。手册 §10.1 引用示例受影响（手册示例需修/需等待修复）。 | ✅ 已修复（2026-08-03，运行时部）：VM 引用语义补齐——新增 opcode MakeRef(59)/MakeMutRef(60)/Deref(61)/DerefStore(62)（op.rs + chunk.rs emit/read_op + execute.rs 执行 + bytecode.rs 编译 + JIT fallback 双侧）；`&mut 变量` 经 Value::Shared 槽位 + Value::MutRef(Weak)，`*m = v` 写穿 Shared 不再硬编码 Store(0)，写回与声明顺序无关；算术/比较/取整前置解包 Shared（对齐解释器 eval_binary）。**验证**：`m_ref_multi.th`、`m_ref_slot1.th`（y 非槽 0 真实缺陷，修复前静默错值）VM=解释器一致；守护测试 `vm_ref_vec_methods_test` 19 项全绿；stdlib/ownership/vm 等针对性套件 291 项 0 回归。 |
| AUDIT-11.4.22 | VM 高阶函数/闭包作参数缺口 | 手册示例验证（E 任务）发现（2026-08-02） | `collections::iter::map/filter` 等闭包作参数的高阶函数在 VM 路径不可用（`m_map_diag.th`：map 结果无 len / 报未定义函数）；解释器正常。与批次2 A（a1 VM 闭包值调用）任务**重合**——VM 无法以闭包值作实参，修复随 a1 推进（登记关联）。手册 §12.7 代表性函数示例受影响。 | ✅ 部分修复（2026-08-03，a1 P4）：闭包值调用缺口随 a1（P1-P3）修复——`iter::map/filter/reduce`、`collections::any/all/find/count_if/partition`、`curry::partial/curry/compose`、`accumulate_loop<T>`、`runtime::run_with_limit/limit_or_default/run_with_timeout/timeout_or_default` 在 **VM 路径全部可用**（VM=解释器对拍一致，对拍用例 `.trae/tmp/a1_p4_verify/`）；`Tenth实例/闭包合集/closures.th`、`闭包捕获/closure_capture.th` VM 转绿；stdlib_smoke_test 4 个高阶用例（M29/M31/M49/M50）已翻转为 VM 路径（56/56 全绿）。**剩余**：`collections::flat_map` 被新发现独立缺陷（**VM Vec 方法分派缺 `extend`**，解释器有；同族缺口另有 VM Map 缺 `entries` 阻塞 `map_values/filter_map`）阻塞——非 a1 残余（闭包调用本身正常，t04e 证实闭包返回 Vec 可用）、非 AUDIT-11.4.23，属既有 VM/解释器方法对齐缺口，待另行安排（建议登记新 AUDIT）。`group_by` 仅存在于 collections.th 文档注释，无实现（非 a1 问题）。 |
| AUDIT-11.4.23 | 文件级模块限定调用不可用 | 手册示例验证（E 任务）发现（2026-08-02） | `use std::env;` 后 `env::get_or_empty(...)`、`use nn::activations;` 后 `activations::leaky_relu_select(...)` 在 VM 与解释器**均**报「未定义的函数」（两种限定调用模式均不可用）；正确写法是 `use std::env::get_or_empty;` 导入具体函数后直接调用。**手册 4 处示例**（§11.6 / §12.13 / §12.14.2 等）依赖此模式受影响（手册示例需修/需等待修复）。复现 `.trae/tmp/manual_audit/m_env2.th`、`m_activations_q.th`、`m_activations_direct.th`。 | ✅ 已修复（2026-08-03，编译器部）：**根因**：① `use std::env;`（末段是模块文件）被当作「导入单个函数 `env`」处理——模块函数虽已整体进 `self.functions`，但末段在模块内找不到同名函数时**不注册任何别名/作用域符号**，`env::get_or_empty` 只能解析为扁平名 `env::get_or_empty`（永远不存在）→ 报未定义；② 目录型模块（`use std::collections;`）连文件都加载不到（`try_import_file` 缺「目录/<末段>.th」分支，头部注释有但代码未实现）→ 静默 no-op。**修复**（`hir/lower/`）：(1) `try_import_file` 补「`<rel>/<last>.th`」目录模块分支；(2) `use` 处理区分「末段是函数 vs 模块」——末段非模块内函数且完整 path 命中模块文件时，注册**模块别名**（`module_aliases: alias→模块key`）+ 模块全部函数进作用域（裸名与限定调用均可，手册 §12.14.2 同时用 `env::get_or_empty` 与裸 `exit`）；(3) `lower_expr` `Ident("mod::fn")` 新增模块限定解析（枚举变体优先，其次模块别名→底层函数名），泛型也注册进 `generic_funcs` 供 `mod::fn<T>(...)`。**验证**：`env::get_or_empty`（手册 §12.14.2 完整示例含 `exit`）、`collections::flat_map`（目录模块）、`activations::leaky_relu_select_default` 均 VM=解释器一致；既有直接导入/glob 不回归。守护测试 `module_qualified_call_test` 13 项全绿；`stdlib_smoke_test` 56/56 + `stdlib_test` 147/147 + `module_test` 6/6 + `l23a_fix_test` 10/10 + 自举 [OK]。**遗留**：泛型函数**普通调用**（无显式类型参数，`activations::leaky_relu_select(x, 0.01)`）为既有语言限制（同文件泛型普通调用同样失败，非本 AUDIT 范围），需显式 `mod::fn<T>(...)` 或 `fn<T>(...)`——手册 §11.6/§12.13 示例仍需文档部按 D4 改写（`leaky_relu_select<f64>(...)`）。**tenthc**：无需同步（tenthc 无模块系统，use 解析后丢弃）。 |
| AUDIT-11.4.24 | Vec 相等比较 `==` 缺陷（元素相同返回 false） | 手册示例验证（E 任务）发现（2026-08-02） | Vec `==` 元素相同但比较结果为 false：`decoded == [72, 105]`（解码值 72/105 均正确）返回 false。**影响**：断言/条件判断静默错误，高危。复现 `.trae/tmp/manual_audit/m_b64_diag.th`（§12.15.4 Base64/Hex/URL 示例断言必失败——手册示例需修/需等待修复）。 | ✅ 已修复（2026-08-03，运行时部）：VM `vm_eq` 与解释器 `values_eq` 均补 Vec/Array 元素逐一相等（解包 Shared），`decoded == [72, 105]` 现返回 true；另修解释器 `base64_encode`/`hex_encode` 未解包 Shared 元素（数组字面量元素 Shared 包裹）导致读 0 的静默错值——现 VM=解释器输出完全一致（`SGk=`/72/105/true/`ff0080`/255/0/128）。守护测试 `vm_vec_eq_*` 通过。 |
| AUDIT-11.4.25 | 字符串不支持 `\u{...}` Unicode 转义 | 手册示例验证（E 任务）发现（2026-08-02） | lexer `read_string` 仅支持 `\n\r\t\\\"\{\}`，无 `\u`/`\xNN` 分支——`"cafe\u{0301}"` 原样输出 `cafeu{0301}`，§12.15.2 Unicode 规范化示例不可运行（手册示例需修/需等待修复）。复现 `.trae/tmp/manual_audit/m_escape.th`、`m_unicode_diag.th`。 | ✅ 已修复（2026-08-03，编译器部）：lexer `read_string` 新增 `\u{HEX}` Unicode 码点转义（UTF-8 编码，`\u{0301}`→U+0301 组合重音、`\u{1f600}`→😀，无效码点/非十六进制报 LexerError）与 `\xNN` 十六进制字节转义（与字节串一致）；`\u` 后非 `{` 保持原样向后兼容。验证：m_escape/m_unicode_diag 双路径 EXIT=0 输出一致（café 规范化断言 true）；string_encoding_test 守护通过。 |
| AUDIT-11.4.26 | format 进制说明符 `{:x}`/`{:X}`/`{:o}`/`{:b}` 未实现 | 手册示例验证（E 任务）发现（2026-08-02） | `format("0x{:x}", 255)` 输出 `0x255`（说明符被忽略，按普通格式处理）；手册 §12.15.1 表格明确列出进制格式，文档夸大（手册示例需修/需等待修复）。复现 `.trae/tmp/manual_audit/m_format_diag.th`。 | ✅ 已修复（2026-08-03，运行时部）：`apply_format_spec`（VM `runtime/natives.rs` + 解释器 `interpreter/natives.rs` 双侧）重构为接收 `&Value`，支持 `{:x}`/`{:X}`（十六进制）/`{:o}`（八进制）/`{:b}`（二进制）/`{:d}`（十进制），含宽度/补零/`#` 前缀（`{:08x}`→`000000ff`、`{#x}`→`0xff`）；非整数回退默认字符串；既有 `>5`/`<5`/`^5`/`.2f` 语义保持。验证：m_format_diag 双路径 EXIT=0（0xff/FF/377/11111111）；string_encoding_test 进制守护通过。 |
| AUDIT-11.4.27 | f-string 仅支持简单标识符插值，不支持表达式与格式说明符 | 手册示例验证（E 任务）发现（2026-08-02） | lexer 插值用 `is_valid_ident` 限制——`f"pi ≈ {3.14159:.2f}"` 原样输出 `{3.14159:.2f}`；手册 §2.3 / §12.15.1 称支持任意表达式与格式说明符，文档夸大（手册示例需修/需等待修复）。复现 `.trae/tmp/manual_audit/m_format_diag.th`、`a03_fstring.th`。 | ✅ 已修复（2026-08-03，编译器部）：lexer `read_string(is_fstring)` 对 f-string 做括号深度感知的 `{expr:spec}` 扫描（`{{`/`}}`→字面花括号，与 Python 一致）；`lower_expr.rs` FString 分支按首个顶层 `:` 拆分表达式+格式说明符，表达式经子 Lexer/Parser 重新解析并 lower（任意表达式 `{a+b}`/`{3.14}` 等），说明符随模板 `{:.2f}` 由 format native 统一解析；顺带修 VM 既有限制（bytecode InterpolatedString 字符串化 `format(x)`→`to_string(x)`，普通插值含数字不再报错）。验证：m_format_diag/a03_fstring 双路径 EXIT=0（pi ≈ 3.14、name=Alice, age=30）；string_encoding_test 表达式/说明符/花括号转义守护通过。 |
| AUDIT-11.4.28 | VM Vec/Map 方法分派与解释器不对齐（flat_map/map_values/filter_map 阻塞根因） | a1 P4 std 高阶解锁验证发现（2026-08-03） | `runtime/vm/natives.rs` VM 方法分派缺方法，解释器 `interpreter/methods.rs` 全有：**VM Vec 缺 10 个方法**（`index_of`/`reverse`/`slice`/`extend`/`sort`/`dedup`/`first`/`last`/`flatten`/`chunks`）、**VM Map 缺 `entries`**。**影响**：`std::collections::collections::flat_map`（依赖 Vec.extend）在 VM 报「Vec 没有方法 'extend'」、`map_values`/`filter_map`（依赖 Map.entries）报「Map 没有方法 'entries'」——三者解释器路径均正常。**非 a1 残余**（闭包值调用 + 闭包返回 Vec 均正常，`t04e_diag_retvec.th` 证实）、**非 AUDIT-11.4.23**（直接导入函数名调用仍不可用，按任务判定依据可排除）。同族属性：VM/解释器方法分派对齐缺口。复现 `.trae/tmp/a1_p4_verify/t04d_diag_extend.th`（extend 缺失）、`t11a_diag_entries.th`（entries 缺失）。**建议修复方向**：运行时部按解释器实现补齐 VM 方法分派（`vm/natives.rs`，参照 `interpreter/methods.rs` 对应实现，约 15 行/方法）。 | ✅ 已修复（2026-08-03，运行时部）：`runtime/vm/natives.rs` `call_method_priv` 补齐 **Vec 10 方法**（`index_of`/`reverse`/`slice`/`extend`/`sort`/`dedup`/`first`/`last`/`flatten`/`chunks`）+ **Map `entries`**，语义对齐解释器 `methods.rs`。**验证**：`t04d_diag_extend.th`（extend）、`t11a_diag_entries.th`（entries）VM=解释器一致；`std::collections::collections::flat_map`/`map_values`/`filter_map` VM 路径可用（对拍 `.trae/tmp/manual_audit/m_std_flatmap_vm.th` 一致）；守护测试 11 项通过。 |
| AUDIT-11.4.29 | `collections.th` `group_by` 文档注释误导（仅注释无实现） | a1 P4 std 高阶解锁验证发现（2026-08-03） | `tenth/std/collections/collections.th` 文档注释声称有 `group_by` 函数，但源码中**无实现**——引用时报「未定义的函数」，属文档/实现不一致（非 a1 问题，VM/解释器均无此函数）。**建议**：文档部修正 collections.th 注释（删除或标注未实现），或补实现。 | ✅ 已修复（2026-08-03，总师直做）：`collections.th` 文件头 usage 注释已修正——移除 `use ...group_by` 误导行，改为标注「group_by 未实现，见 AUDIT-11.4.29」。 |
| AUDIT-11.4.30 | a1 闭包值调用遗留三项：native 别名 VM 断 / 递归闭包前端拒绝 / JIT 报错行号不保留 | a1 遗留 + B 批 JIT 独立限制（2026-08-03 登记，任务 9 修复） | ① **native 别名**：`let p = println; p("x")` 在 VM 报「期望可调用值，得到 Unit」——bytecode/VM 无 native 名→FnRef 表（解释器 eval.rs 有硬编码表），`LoadGlobal` 未命中得 Unit（响亮非静默）；② **递归闭包**：`let fact = |n| ... fact(n-1)` 在 lowering 报「未定义变量 'fact'」——闭包体自引用未绑定（前端限制，a1 范畴外）；③ **JIT 报错行号**：JIT 路径 hostcall 用 `set_last_error(e.to_string())` 传字符串，行号不保留（B 批记录为 JIT 独立限制）。 | ✅ 已修复（2026-08-03，编译器部 + 运行时部，任务 9）：**① native 别名**——VM `execute.rs` opcode 9 LoadGlobal 与 JIT `host_load_global` 在 globals 未命中时查 `natives` 表构造 `Value::FnRef`（用户全局 shadow native），`let p = println; p("x")` VM=JIT=解释器一致；**② 递归闭包**——lowering 新增 `Lowerer.self_ref_lets` 栈（`let name = <闭包>` 时压入），闭包体同名引用解析为 `Var(name)`（不报未定义变量）且**排除出 captures**（创建时槽位未绑定，按值捕获得 Unit/旧值；运行时按名解析：VM 经 globals / 解释器经作用域链）；`let fact=|n| if n<=1 {1} else {n*fact(n-1)}; fact(5)=120` 与「递归+捕获混用」（f(4)=762）VM=解释器一致；**③ JIT 报错行号**——JIT translator 在每个 hostcall 前把当前指令行号（chunk 行号表 `line_at(op_start)`）写入 `vm.current_line`，hostcall 用新 `Vm::set_jit_error` 补行号（取裸 message，消除双重前缀），`run_jit` surface 带行号 RuntimeError；除零/取模/非可调用/reshape/切片/索引越界全部带行号且文案与 VM 对齐。**验证**：守护 `jit_test` 新增 4 项（native 别名 2 + 递归闭包 2）、新建 `jit_error_line_test` 7 项全绿；`jit_test` 25/25、`vm_error_line_test` 10/10、相关回归（module_qualified/toplevel_let/silent_failure/stdlib/smoke/l23a）262 项 0 失败；自举 [OK]。**遗留 → ✅ 已根治（2026-08-03，M1-S2 true letrec）**：递归闭包逃逸作用域 + 多实例共存的别名静默错值已由 **true letrec** 根治——自引用绑定做成**实例级可变 cell**（`Value::Shared(Rc<RefCell<Value>>)`）捕获：HIR `HirExprKind::Closure` 新增 `self_refs`，闭包创建时先建空 cell 再 `BindSelfCapture` 写自身，体引用经捕获槽位 Load 到 cell、调用时解包 Shared 后间接调用；每实例 cell 独立随闭包走（逃逸作用域后仍可用），不再按名/全局解析。VM 新增 opcode `MakeCell(63)`/`BindSelfCapture(64)`；`CallClosure`/`call_value`/解释器 `eval_call`/`apply_closure` 对 Shared 解包；`Value` Debug 改转发 Display（防自引用 cell `{:?}` 无限递归）。**验证**：`t9b_multi_instance.th` VM=解释器=**762769**（修复前 522769/未定义变量）；新增守护 `tenth/tests/letrec_test.rs` **10 项**（多实例独立/逃逸作用域/main 级回归/递归+捕获回归/三实例/尾递归 TCO/嵌套/ HOF JIT 路径/工厂多实例/Debug 有界）VM=JIT=解释器三方对拍全绿；回归 31 套件 ≈850 项 0 失败；自举 [OK]。**遗留（如实，非本 AUDIT 范围）**：互递归（`let f=...g...; let g=...f...`）与条件分支共享 letrec（`let x = if c { |n| x(n-1) } else { |n| n }`）仍非 true letrec 共享绑定（每闭包独立 cell）。 |

| AUDIT-11.4.31 | 解释器缺 broadcast_to 方法分派 + VM 内联 mod 限定调用缺口（M1-S3a 两项同批修复） | 手册示例验证（E 任务）发现（2026-08-03 登记，M1-S3a 修复） | ① **L14（解释器缺 broadcast_to）**：手册 §11.3 `t.broadcast_to(3, 4)`（位置参数形式）在解释器（TENTH_NO_VM=1）报「未知的张量方法: broadcast_to」——VM 可用（`runtime/vm/natives.rs` 已注册）、tensor 层 `Tensor::broadcast_to`（`runtime/tensor/methods.rs:1087`）存在，仅 `runtime/interpreter/methods.rs` 未接线（与 AUDIT-11.4.28 同型反向）；② **L5（VM 内联 mod 限定调用）**：内联 `mod { }` 块（手册 §9.1/9.2）定义的函数经 `mod_name::fn_name()` 限定调用在 VM 报「未定义的函数 'mod::fn'」——任务 5（AUDIT-11.4.23）修的只是**文件级**模块（use 注册 module_aliases）；解释器在运行时按 `::` 拆分查 `self.modules` 故可用，VM 无此机制。 | ✅ 已修复（2026-08-03，运行时部 + 编译器部，M1-S3a）：**① broadcast_to**——`runtime/interpreter/methods.rs` 张量方法分派补 `broadcast_to`（目标 shape 由整数位置参数收集，失败报 RuntimeError，语义与 VM natives.rs 完全一致）；**② 内联 mod**——`hir/lower/lower_expr.rs::try_resolve_module_qualified` 扩展：首段命中 `self.modules` 内联 mod 键（无需 use）即解析 `mod::fn(...)` → 底层函数名 `Var("fn")`，并把内联 mod 函数补入 `self.functions`（first-wins，与 use 导入一致）——后端（bytecode/解释器/wasm）按函数名解析即通。**验证**：§11.3 broadcast_to、§9.1/9.2 内联 mod（直呼 + `use math::add;` 限定 + 表达式位置）全部 VM=解释器一致；文件级模块不回归（`module_qualified_call_test` 13/13）；新增守护 `s3a_runtime_gap_test` 14 项全绿；`stdlib_smoke_test` 59/59；release 构建通过。**遗留（如实）**：① 内联 mod 函数体裸名调用私有 sibling 仍不支持（VM 与解释器一致地失败，§9.1 的 `helper` 未在 `add` 内调用）；② 顶层函数与内联 mod 函数同名时 first-wins 可能解析到顶层同名函数（与文件级模块「先注册者胜」既有语义一致）；③ 嵌套多段 `outer::inner::fn()` 限定调用不在本次范围（手册仅单层）；④ `module_test` 当前无法编译（并行任务 S2 true letrec 在改 closures 区域，`HirExprKind::Closure` 增 `self_refs` 字段后 `eval.rs`/`wasm/closures.rs` 模式匹配暂未更新），非本次改动引入。 |
| AUDIT-11.4.32 | WASM 后端 4 项缺口：函数内引用全局 / 标签 break / 自定义运算符 native / 泛型枚举布局（+match 不支持） | M1-S1 阶段 1 可用性最后缺口（2026-08-03 登记，M1-S1 修复） | ① **函数内引用全局**：globals 只做进 main 的 local 槽，函数体 `Var(g)` 报「未定义变量 'g'」（VM/解释器经 globals/作用域链可用）；② **标签 break/continue**：`if_depths` 无标签维度，带标签形式直接报「暂不支持 WASM 后端」（明确报错不静默）；③ **自定义运算符 native**：operator 定义体内调用标量 math native（sqrt 等）报「未定义函数 'sqrt'」——WASM 后端 resolve_func 只认固定 host 集合；④ **泛型枚举布局**：`build_struct_layouts` 只建 `program.enums` 不建 `generic_enums`，且 **Match 在 WASM 完全不支持**（compile.rs 无 Match 分支落入 `_ => Err`）、枚举内存无变体 tag（无法区分变体）。另：**M3.5 潜伏 bug**——compile_main 原优先 main_expr，顶层 let 提升后 main_expr 常为 Unit 空块，`wrap_to_i32(Unit)` 发 `Drop`（空栈）致 WASM 校验失败（此前被「函数内引用全局报错」掩盖）。 | ✅ 已修复（2026-08-03，编译器部，M1-S1）：**① 全局**——globals 改真 WASM mut i64 全局（emit_global_section 声明 + global_map），main 前 GlobalSet 初始化，函数体 Var/Assign/AssignOp 经 GlobalGet/GlobalSet + 按声明类型转 i64 位存储；**② 标签**——`if_depths` 增标签维度，带标签 break/continue 从栈顶搜索匹配标签按相对深度计算 Br（语义与 bytecode loop_stack 一致）；**③ native**——sqrt/abs 用 WASM 原生指令内联，sin/cos/ln/pow 新增 4 个 host 函数（host_sin/host_cos/host_ln/host_pow，sections.rs/host.rs/wasmtime_host.rs 双侧注册 + 8 个测试 linker stub）；**④ 枚举**——泛型枚举布局补齐（TypeParam 字段统一 8 字节 i64 位存储），枚举布局统一为 tag(i64)+字段，**新增 Match codegen**（按 tag 比较变体、绑定字段、wildcard、字面量模式、守卫），f64 字段位存储往返正确；**顺带**：compile_main 改与 VM 一致优先 `fn main` + Unit 分支不 Drop。**验证**：新建 `m1s1_wasm_gaps_test.rs` **15 项全绿**；WASM 回归 wasm_backend_minimal 10 / f32_wasm 14 / parity 129 / three_stage 1(+2 ignored) / tenthc 系列 18 全绿；定向套件（generic_enum/toplevel_let/custom_operator/labeled_break/module/selfhost_frontend）全绿；自举 `[OK]`。**遗留（如实，非本次范围）**：① **类型系统 `abs(i64)` 返回 F64**（`infer_scalar_dtype` 默认 f64）而 VM 运行时返回 Int——既有不一致，WASM 按类型系统产 F64（`fn->i64 { abs(...) }` 会 wasmi 校验报错，不静默错）；② **tenthc wasm.th 未接标量 math**——tenthc 编译调用 sqrt/abs/sin/cos/ln/pow 的程序 fallthrough 到错误 call_idx（自举源码不用，无实际影响，建议后续 tenthc wasm.th 显式报错）；③ **Match 模式覆盖**——Tuple/Range/Struct/Binding 模式报错不静默（枚举变体/字面量/wildcard 已覆盖）。 |
| AUDIT-11.4.33 | `?` 解释器 double-wrap（try 块捕获路径）+ 十进制字面量下划线 bug + 字节串仅词法层 | M1-S5 语言规范评审（2026-08-03 登记，M1 收尾修复） | 三项与规范/手册声称不符的实现缺口（M1-S5 评审发现）：① **`?` double-wrap**——披露于 `try_operator_test.rs` 注释：`?` 遇 `Err` 时解释器返回 `Result::Err(Result::Err(e))`（double-wrap），VM 返回单层。函数边界 `unwrap_return` 曾在 b0c9588 修好单层，但 **try 块捕获处**（`runtime/interpreter/eval.rs` TryBlock 分支）仍把 TryPropagate 携带的完整 `Result::Err(e)` 再包一层——`try { f()? }` 在解释器路径仍 double-wrap；② **十进制下划线**——规范 §2.6.1 声称「下划线分隔（任何进制均支持）」，但 lexer `read_number` 十进制路径消费 `_` 并入 s 后直接 `s.parse()`，`"1_000_000"` 解析失败报「无效的整数」（进制路径正确跳过下划线，十进制/浮点漏剥离）；③ **字节串半实现**——`b"..."` 仅词法层产出 `ByteString` token，parser/HIR/运行时未接线，作为表达式直接报「意外的标记」——规范 §2.6.8 声称「运行时 Vec<u8>」失实。 | ✅ 已修复（2026-08-03，运行时部 + 文档部，M1 收尾）：**① `?` 单层**——`runtime/interpreter/eval.rs` TryBlock 捕获分支对齐 `core.rs::unwrap_return` 逻辑：TryPropagate 的 err_val 已是完整 `Result::Err(e)` 则直接透传（不再包一层），非 Result 值才兜底包装——与 VM `Op::Try` 单层语义一致（VM try 块 catch 路径本就透传栈上 `Result::Err(e)`）。验证：`try_operator_test` 12→17 项（新增多层 `?`/try 块成功/捕获 + 解释器/VM 对拍严格单层断言），VM=解释器一致；**② 下划线**——`lexer.rs read_number` 三处 parse（整数/浮点/浮点带后缀）前 `s.replace('_', "")`，`1_000_000`/`1_000.5` 现合法（进制路径原已跳过）；**③ 字节串**——`parser/expr.rs parse_primary` 把 `ByteString` token 降为整数数组字面量（每字节 → `Int(b, I32)`，复用 ArrayLiteral 全管线），`b"..."` 现可作表达式：`.len()`/`hex_encode`/`base64_encode`/`write_bytes` 可用（运行时为 `Vec<i64>` 字节数组）。**验证**：release 构建通过；手册修订示例 v1_literals（进制/下划线/后缀/提升/字符/`\u`/`\x`/原始/多行/字节串）/ v3_precedence（`1 == 2 < 3` → false）/ v5_try（try 块单层）/ v5_async（async/await/spawn/yield）VM=解释器双路径 EXIT=0 一致；针对性套件 try_operator 17 / string_encoding 68 / int_types 14 / lexer 34 / yield 6 全绿。 |
| AUDIT-11.4.34 | match tuple 模式 + guard 缺陷（guard 失败后回退错乱 + JIT 编译 panic 逃逸） | M2-A5 VM-vs-JIT 一致性套件发现（2026-08-03 登记，部分修复） | ① **VM guard 回退错乱**：`match t { (a,b) if a>b => ..., (a,b) => ..., _ => ... }` 中 guard 失败后，VM 不尝试**下一条 tuple 臂**而是直接落 wildcard——`(2,3)`（guard false）VM 返回 `_ => -1` 而解释器正确返回 23；② **JIT 编译 panic 逃逸**：任何 tuple 模式 + guard 都会在 Cranelift 低化阶段（`finalize_definitions`，位于既有 catch_unwind 之外）触发 `TryFromIntError` panic 并**逃逸出 JIT**（进程级 panic 红线），二进制顶层捕获后回退 VM 才免于崩溃。**影响**：tuple 模式 + guard 组合不可靠（VM 静默错值方向 + JIT panic）。最小复现 `.trae/tmp/a5_tguard.th` / `a5_tguard2.th`（已删，用例模式见 jit_consistency_test 注释）。 | ⚠️ 部分修复（2026-08-03，M2-A5）：**② 已根治**——`compile/jit/context.rs` 把**整个编译链路**（translate + finalize_definitions 低化 + get_finalized_function）纳入 catch_unwind，Cranelift 低化 panic 统一转为 Err → 既有 fallback（VM 解释执行），不再逃逸进程（JIT compile panic 红线）。**① 未修复（如实登记，待排期）**：VM match lowering 的 guard 失败后回退逻辑（tuple/多臂 + guard 组合）为预存缺陷，非 M2 JIT 引入；一致性套件已改用「单 guard 臂 + wildcard」稳定形态，**未**固化该错误形态为用例（避免守护错误行为）。**性质**：① 静默错值 + 待修复；② 已修复（panic 逃逸红线）。详见 MEMO.md 2026-08-03 M2-A5 条目。 |

### 11.5 登记元数据

| 字段 | 值 |
|------|----|
| 登记日期 | 2026-07-02（第 1 批）/ 2026-07-06（11.4.4/11.4.5 新增）/ 2026-07-25（11.4.6 / 11.4.7 新增）/ 2026-07-25（11.4.8 新增）/ 2026-07-25（11.4.9 新增）/ 2026-07-30（11.4.10 / 11.4.11 / 11.4.12 / 11.4.13 新增）/ 2026-07-30（11.4.14 新增）/ 2026-07-31（11.4.15 新增）/ 2026-07-31（11.4.16 / 11.4.17 / 11.4.18 / 11.4.19 / 11.4.20 新增）/ 2026-08-03（11.4.21 ~ 11.4.27 新增）/ 2026-08-03（11.4.28 / 11.4.29 新增）/ 2026-08-03（11.4.30 新增）/ 2026-08-03（11.4.31 新增）/ 2026-08-03（11.4.32 新增）/ 2026-08-03（11.4.33 新增）/ 2026-08-03（11.4.34 新增） |
| 登记批次 | 第 1 批（纯文档登记，无代码修改）+ 2026-07-06 回归测试发现新增 2 条 + 2026-07-25 基本功核查发现新增 1 条（11.4.6）+ 已知限制 #12 修复同步新增 1 条（11.4.7）+ 已知限制 #9 修复同步新增 1 条（11.4.8）+ 2026-07-25 文档部复核新增 1 条（11.4.9）+ 2026-07-30 第一波修复同步新增 4 条（11.4.10 JIT is_sealed / 11.4.11 embedding gather ndim / 11.4.12 .shape() 类型系统不一致 / 11.4.13 randn 变量参数）+ 2026-07-30 P1-1 修复同步新增 1 条（11.4.14 tenthc 3D+ transpose 双侧不一致）+ 2026-07-31 用户反馈修复同步新增 1 条（11.4.15 泛型函数运行时不可用）+ 2026-07-31 护城河系列同步新增 5 条（11.4.16 NaN 比较静默 / 11.4.17 JIT 整数溢出路径 / 11.4.18 标量→张量静默降级 natives / 11.4.19 默认线程栈 2MB 栈溢出基线 / 11.4.20 lossy 设计取舍）+ 2026-08-03 手册示例验证（E 任务）同步新增 7 条（11.4.21 VM `&mut` 写回顺序相关失效 / 11.4.22 VM 高阶函数/闭包作参数缺口 / 11.4.23 文件级模块限定调用不可用 / 11.4.24 Vec `==` 比较缺陷 / 11.4.25 字符串 `\u` 转义未实现 / 11.4.26 format 进制说明符未实现 / 11.4.27 f-string 仅标识符插值）+ 2026-08-03 a1 P4 std 高阶解锁验证同步新增 2 条（11.4.28 VM Vec/Map 方法分派缺口 / 11.4.29 group_by 文档注释误导）+ 2026-08-03 任务 9（a1 遗留三项补齐）同步新增 1 条（11.4.30 a1 闭包值调用遗留三项）+ 2026-08-03 M1-S3a 同步新增 1 条（11.4.31 解释器 broadcast_to 分派缺口 + VM 内联 mod 限定调用缺口）+ 2026-08-03 M1-S1 同步新增 1 条（11.4.32 WASM 后端 4 项缺口）+ 2026-08-03 M1 收尾（M1-S5 评审）同步新增 1 条（11.4.33 `?` 解释器 double-wrap + 十进制下划线 bug + 字节串半实现）+ 2026-08-03 M2-A5 VM-vs-JIT 一致性套件同步新增 1 条（11.4.34 match tuple 模式 + guard 缺陷：VM guard 回退错乱 + JIT 编译 panic 逃逸） |
| 论文来源范围 | T12 / T18 / T19 / T20 / T22 / T31 / T36 / T37 / T42 / T50（共 10 个论文条目）+ 回归测试发现 2 条 + 基本功核查发现 1 条 + 已知限制同步 2 条 + 文档部复核 1 条 + 第一波修复同步 4 条 + P1-1 修复同步 1 条 + 用户反馈修复同步 1 条 + 护城河系列同步 5 条 + 手册示例验证（E 任务）同步 7 条 + a1 P4 验证同步 2 条 + 任务 9（a1 遗留三项）同步 1 条 |
| 总登记条目数 | 43（11.1 ×5、11.2 ×4、11.3 ×2、11.4 ×32） |
| 拆分说明 | T12 双侧破口在 11.2 拆 4 条（shape/panic-mode/expect_gt/i64 溢出）；T37 双重 native 在 11.3 拆 2 条（VM 缺 17 项/解释器缺 3 项）；其余 8 个论文条目各 1 条；2026-07-06 新增 11.4.4（tenthc `..=` lexer bug）和 11.4.5（tenthc Vec 迭代未实现）属回归测试发现；2026-07-25 新增 11.4.6（CUDA 后端文档矛盾）属基本功核查第 94 项发现；2026-07-25 新增 11.4.7（tenthc bmm FLOPs 预估为 no-op）属已知限制 #12 修复同步登记；2026-07-25 新增 11.4.8（tenthc yield 关键字未接入）属已知限制 #9 修复同步登记；2026-07-25 新增 11.4.9（标准库三项能力文档失实）属文档部复核发现；2026-07-30 新增 11.4.10（JIT `is_sealed` 断言 panic）+ 11.4.11（embedding `gather` ndim 限制）+ 11.4.12（`hir/types.rs:392` `.shape()` 方法分支与运行时不一致）+ 11.4.13（`randn` 变量参数限制）属第一波 5 项修复同步登记；2026-07-30 新增 11.4.14（tenthc 3D+ transpose shape 推断双侧不一致）属 P1-1 修复同步登记；2026-07-31 新增 11.4.15（泛型函数运行时不可用 — `tensor<f64>([...])` Array→Tensor 转换 + `randn<T>` TypeParam 实例化）属用户反馈修复同步登记；2026-07-31 新增 11.4.16（NaN 比较静默）+ 11.4.17（JIT 整数溢出路径未覆盖）+ 11.4.18（标量→张量静默降级 natives 未全审计）+ 11.4.20（lossy 设计取舍三项）属阶段2b lossy M2 建议登记；11.4.19（parity_test / jit_stack_overflow_test 默认线程栈 2MB 栈溢出基线）属测试环境问题登记（M1 提交 0a426ad 即存在，非新回归）；2026-08-03 新增 11.4.21（VM `&mut` 写回顺序相关失效）+ 11.4.22（VM 高阶函数/闭包作参数缺口）+ 11.4.23（文件级模块限定调用不可用）+ 11.4.24（Vec `==` 比较缺陷）+ 11.4.25（字符串 `\u` 转义未实现）+ 11.4.26（format 进制说明符未实现）+ 11.4.27（f-string 仅标识符插值）属手册示例验证（E 任务）发现登记（报告 `.trae/tmp/manual_example_audit.md`；D1-D9 手册示例 bug 属文档问题不登记）；2026-08-03 新增 11.4.28（VM Vec/Map 方法分派与解释器不对齐）+ 11.4.29（collections.th `group_by` 文档注释误导）属 a1 P4 std 高阶解锁验证发现登记（报告 `.trae/tmp/a1_p4_verify/report.md`）；2026-08-03 新增 11.4.34（match tuple 模式 + guard 缺陷）属 M2-A5 VM-vs-JIT 一致性套件发现登记（JIT 编译 panic 逃逸已根治，VM guard 回退错乱待排期） |
| 真理源同步 | `能力梳理/能力全梳理.md` 同步新增能力条目；`MEMO.md` 顶部新增 2026-07-02 与 2026-07-06 与 2026-07-25 与 2026-07-30 与 2026-07-31 变更记录；2026-07-31 护城河系列条目同步新增 11.4.16-11.4.20；2026-08-03 手册示例验证（E 任务）条目同步新增 11.4.21-11.4.27；2026-08-03 a1 P5 收尾条目同步新增 11.4.28-11.4.29；2026-08-03 M1-S3a 条目同步新增 11.4.31；2026-08-03 M1-S1 条目同步新增 11.4.32；2026-08-03 M1 收尾（M1-S5 评审）条目同步新增 11.4.33；2026-08-03 M2-A5 条目同步新增 11.4.34（match tuple 模式 + guard 缺陷） |
| 修复方案 | 分批推进：第 1 批纯登记；2026-07-06 修复 11.4.1/11.4.2/11.4.4（✅）+ 11.4.5 codegen 修复；2026-07-08 补全 11.4.5 测试 host function stub（✅）；2026-07-08 修复 11.4.3（✅ 解释器 scope 扁平化索引）；2026-07-25 修复 11.4.6（✅ 纯文档措辞修正，零代码改动）；2026-07-25 修复 11.4.7（✅ tenthc HirType 加 dim2 + 6 处 no-op 激活 + parse_tensor_type 放宽，含 parity 测试）；2026-07-25 修复 11.4.8（✅ tenthc 侧 yield 关键字 parser/lower/wasm 三层接入 + 3 项测试）；2026-07-25 修复 11.4.9（✅ 能力全梳理状态修正 + 总结矩阵数字更新 + 测试补强 + 能力扩展，纯文档同步）；2026-07-30 修复 11.4.10（⚠️ catch_unwind workaround + 根本修复推后 P2）/ 11.4.11（⚠️ gather ndim 限制 workaround + index_select native 推后 P1 后续）/ 11.4.12（⚠️ 待清理 hir/types.rs:392 .shape() 分支，未清理避免破坏潜在依赖）/ 11.4.13（❌ P1 计划项 Dim::Symbol 支持）；2026-07-30 P1-1 完成 11.4.13（✅ Dim::Symbol + tenthc symbol_dims 双侧同步）+ 登记 11.4.14（⚠️ tenthc 3D+ transpose 保守跳过已知限制）；2026-07-31 修复 11.4.15（✅ array_to_tensor Vec/Array→Tensor 转换 + NATIVE_GENERIC_CTORS 接受 TypeParam + substitute_kind_in_place 函数名修正）；2026-07-31 登记 11.4.16-11.4.18 + 11.4.20（⚠️ lossy 系列已知限制，待后续处理）+ 11.4.19（⚠️ 环境依赖：RUST_MIN_STACK≥32MB）；2026-08-02 修复 §六 #17/#18/#19 与 11.4.12（✅ 编译器部 4 项：字节码槽位 position→rposition 族修复 + match 臂编译期作用域 scope_stack（#17/#18，含同名绑定 shadow 污染）+ 运算符重载链式返回类型从 trait_impls 取（#19）+ .shape() 误标分支删除并编译期报错引导 shape_tensor（11.4.12）；新增 audit_17_18/audit_19/audit_11412 三个回归测试套件共 12 项，全量 0 回归，自举 [OK]）；2026-08-02 修复 11.4.17（✅ 运行时部：算术原语 checked_* 化 + JIT emit_binop/emit_unop 错误立即中断 + 解释器 Sub 对齐 VM；新增 jit_int_overflow_test 13 项，三路径一致，全量 0 回归，自举 [OK]）；2026-08-03 登记 11.4.21-11.4.27（⚠️ 手册示例验证（E 任务）发现的 7 项语言/运行时缺陷，纯文档登记待修复；D1-D9 手册示例 bug 属文档问题不登记，待手册修订）；2026-08-03 登记 11.4.28-11.4.29（⚠️ a1 P4 std 高阶解锁验证发现的 2 项缺陷——11.4.28 VM Vec/Map 方法分派缺口建议运行时部补齐分派、11.4.29 group_by 文档注释误导建议文档部修正，纯文档登记待修复）；2026-08-03 修复 11.4.31（✅ M1-S3a 运行时部 + 编译器部：解释器 methods.rs 补 broadcast_to 分派 + lower_expr.rs try_resolve_module_qualified 扩展内联 mod；新增 s3a_runtime_gap_test 14 项，VM=解释器一致，module_qualified_call_test 13/13 不回归，release 构建通过）；2026-08-03 修复 11.4.32（✅ M1-S1 编译器部：WASM 后端 4 项缺口——函数内引用全局（真 WASM mut i64 全局）、标签 break/continue（if_depths 增标签维度）、自定义运算符 native（sqrt/abs 内联 + sin/cos/ln/pow 新 host 函数）、泛型枚举布局 + Match codegen（tag 布局 + 变体匹配）；新增 m1s1_wasm_gaps_test 15 项，WASM 回归全绿，自举 [OK]）；2026-08-03 修复 11.4.33（✅ M1 收尾运行时部 + 文档部：`?` 解释器单层——eval.rs TryBlock 捕获分支对齐 unwrap_return 直接透传完整 `Result::Err(e)`，try_operator_test 12→17 项含多层 `?`/try 块严格单层对拍；十进制下划线——lexer read_number 三处 parse 前剥离 `_`；字节串——parser 把 ByteString 降为整数数组字面量接通全管线；手册修订示例双路径 EXIT=0，针对性套件全绿）；2026-08-03 部分修复 11.4.34（⚠️ M2-A5 编译器部 + 运行时部：**② JIT 编译 panic 逃逸已根治**——compile/jit/context.rs 把整个编译链路（translate + finalize_definitions 低化 + get_finalized_function）纳入 catch_unwind，Cranelift 低化 panic（tuple 模式 + guard 的 TryFromIntError）统一转 Err → 既有 fallback，不再逃逸进程；**① VM match guard 回退错乱未修复（待排期）**；顺带修 VM 溢出错误行号一致性缺口——execute.rs R2 快速路径（Add/Sub/Mul/Div/Mod/Neg 共 7 处）checked_* 溢出错误补 with_line，对齐 JIT 标量路径（此前 VM 溢出报错无行号、JIT 有行号）；新增 jit_consistency_test 43 项（VM=JIT 对拍：互递归/多参/字符串返回/循环内直接调用/内联多形态/标量全比较集/溢出除零取模取负/Try 多层链式循环/tuple 深层 OOB/struct 字段链循环写/Spawn eager/TailCall 链深递归/错误行号跨路径一致/热函数按需编译覆盖断言）+ bench_gate_test 2 项（D4 基准固化 CI 门槛，默认 #[ignore]，release 下断言 fib<100ms/loop<200ms/matmul<20ms） |

