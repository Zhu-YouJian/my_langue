# 开发备忘

> 各类待办、跳过项、环境依赖、注意事项均记录于此。
>
> **当前阶段：v0.3.3 → 阶段 1（可用）**
>
> 演进路线与阶段规划见 `CODE_WIKI.md` §10。
>
> **2026-07-01 Autograd 反向 Shape 静态验证（护城河 A）**：实现 `docs/shape-check-roadmap/战略规划.md` 方向 A（评级 ⭐⭐⭐⭐⭐，"JAX 都没做好"）。**核心问题**：autodiff.rs 存在 5 处 silent squeeze 静默修正梯度 shape，掩盖真实 shape 错误，导致参数 grad 字段可能写入错误 shape 的数据（典型症状：模型 loss 不降反升、梯度数值异常但无报错）。**5 处 silent squeeze 全部消除**：(1) `tensor.rs::acc_grad` 加 shape 校验——梯度 shape 必须与参数 shape 一致，不匹配返回 `Err`，签名改 `-> Result<(), String>`；(2) `autodiff.rs::unbroadcast` 末尾不再 `unwrap_or(result)` 静默保留错误 shape，reshape 失败/元素数不匹配均报错，签名改 `-> Result<ArrayD<f64>, TenthError>`；(3) `autodiff.rs::matmul_2d` 非 2D 输入不再返回零数组，改 `map_err` 报错，签名改 `-> Result<ArrayD<f64>, TenthError>`；(4) `autodiff.rs::MatMul` backward 1D squeeze 前校验 `d_a_2d.shape[0]==1` / `d_b_2d.shape[1]==1`，`grad.ndim() > 2` 直接报错而非静默 clone；(5) `autodiff.rs::Conv2D` backward dW/dX reshape 失败与元素数不匹配均报错，移除 3 处 `unwrap_or_else(|_| ...)` 零填充兜底。**全链路 Result 传播**：`backward` 签名改 `-> Result<(), TenthError>`，`propagate_grad` 改 `-> Result<(), TenthError>`，所有 Add/Sub/Mul/Div/MatMul/Transpose/Sum/Mean/Exp/Log/Sigmoid/Softmax/CrossEntropy/Dropout/Conv2D/BatchNorm/LayerNorm/Gelu 分支的 `unbroadcast(...)` 与 `propagate_grad(...)` 加 `?` 传播；7 个 backward 调用点全部更新（natives.rs、main.rs、5 处 autodiff.rs 测试）。**新增 op_name 辅助函数**：返回 TapeOp 的人类可读名称（用于错误信息定位节点类型）。**编译期 param() warning**（`lower_expr.rs` Call 分支）：对 `param(t)` 调用，若 `t.static_bytes() ≥ 1GB` 发 warning 提示"反向传播将分配同等大小的梯度，可能触发 OOM"——让用户在编译期就意识到梯度内存开销。**测试** `tests/autodiff_shape_test.rs` 10 项：正确 shape 无回归 3 项（simple/matmul/broadcast gradient）、silent squeeze 边界 1 项（1D tensor 反向传播通过）、单元测试 smoke 1 项、编译期 param() warning 3 项（large ≥1GB 触发/small 不触发/消息格式）、边界 2 项（非 param 调用不警告/多 param 各自警告）。验证：autodiff_shape_test 10/10、autodiff_test 52/52、vm_autodiff_test 15/15、shape_check_compile_test 66 passed + 7 ignored、memory_estimate_test 32 passed + 3 ignored、autodiff lib tests 5/5、stdlib_test 114/114、selfhost_frontend --release 4/4 全绿；自举路径 A 通过（tenthc/main.th → Full compiler compiled to tenthc_full.wasm）。**设计取舍**：选"运行时 shape 校验 + 编译期 warning"方案而非"全静态推导"，因为 autodiff 反向 shape 依赖运行时数据流（如 unbroadcast 结果 shape 取决于前向输入 shape），静态推导需完整抽象解释器成本过高；运行时校验在 backward 出口（acc_grad）和 5 处 silent squeeze 位置加零成本检查，编译期 warning 用已有 shape 信息做 best-effort 提示。
>
> **2026-07-01 编译期内存/算力预估（护城河 D）**：实现 `docs/shape-check-roadmap/战略规划.md` 方向 D（评级 ⭐⭐⭐⭐⭐），用编译期 shape 信息换开发者真实的时间，AI 原生卖点（无竞品）。**warning 基础设施**：`error.rs` 新增 `TenthWarning` 结构体（line/col/message + `display_with_source`，复用 TenthError 的源码定位逻辑），非致命，独立于 `TenthError`。**静态预估数据基础**：`hir/types.rs` 为 `Type` 新增 `static_numel() -> Option<u64>`（所有维 Known 时返回乘积，任一 Symbol/Any/溢出/负数返回 None，用 `checked_mul` 防溢出）和 `static_bytes() -> Option<u64>`（numel × dtype 字节数，F64/I64/U64=8、F32/I32/U32=4、F16/I16/U16/BF16=2、I8/U8/Bool/Char=1）。**warning 收集管线**：`Lowerer` 新增 `warnings: Vec<TenthWarning>` 字段；`HirProgram` 携带 `warnings` 传递到 main.rs；`lower_program` 末尾 `std::mem::take` 转移；`main.rs::run_file` 在 source_to_hir 后 `eprintln!` 输出所有 warnings。**预估插入点**（`hir/lower/types.rs` 末尾两个方法）：`emit_memory_estimate(ty, span, context)`——bytes ≥ 1GB 时发 warning（如"函数调用 创建约 2.00 GB 的 tensor（编译期预估，可能触发 OOM）"）；`emit_matmul_flop_estimate(recv_ty, arg_ty, span)`——2D (M,K)@(K,N) 且 K 匹配时算 M*K*N 乘加，≥ 1 GFLOP 时发 warning。**4 个调用点**（`lower_expr.rs`）：Call 分支（函数调用返回大 tensor）、GenericCall native 构造函数路径（`randn<T>`/`zeros<T>` 等）、GenericCall 泛型函数实例化路径、MethodCall 分支（matmul FLOPs + 方法结果 bytes）。**测试** `tests/memory_estimate_test.rs` 35 项（32 passed + 3 ignored）：static_numel/static_bytes 单元测试 11 项（all_known/single_dim/symbol/any/non_tensor/overflow/negative/f64_2d/f32_2d/f64_3d_large/i8）、内存预估 warning 7 项（large zeros/randn/ones/randn 构造触发、small/medium 不触发、tensor literal）、matmul FLOPs 预估 7 项（large/huge/with_transpose 触发、small/medium 不触发、组合 both）、边界 4 项（dynamic/symbol/non_tensor/line_col）、消息格式 2 项、泛型构造函数 2 项、3 项 `#[ignore]` 高开销占位（actual 8GB allocation、huge matmul execution、deep large tensor chain）。**阈值**：内存 1GB（`1024*1024*1024` bytes）、FLOPs 1 GFLOP（`1_000_000_000` 乘加）。**默认开启**，不新增 CLI 标志（"检查多做、求解少做"原则，O(n) 检查零编译成本）。验证：memory_estimate_test 32 passed + 3 ignored；shape_check_compile_test 66 passed + 7 ignored、autodiff 52/52、stdlib 114/114、shape_check_test 16/16、vm_autodiff 15/15、selfhost_frontend --release 4/4 全绿；自举路径 A 通过（tenthc/main.th → Full compiler compiled to tenthc_full.wasm）。**已知限制**：仅静态 shape 全 Known 时预估，动态维度（Symbol/Any）静默跳过；matmul FLOPs 仅 2D，batched matmul 未覆盖；阈值为硬编码常量，未来可按硬件配置调整。
>
> **2026-07-01 编译期 Shape 检查短期规划（四项方向）**：完成 `docs/shape-check-roadmap/短期规划.md` 全部四项。**方向 1（类型注解强制化）**：`hir/lower/types.rs` 新增 `check_and_merge_tensor_shape(annot, actual, span, context)`——let 语句同时存在 type_ann 和 init 时强制检查 shape 兼容性并合并；wildcard `[..]` 合并 actual dims、actual 单 Any 保留 annotation、维度数不等报错、逐维 Known/Symbol 不等报错；`lower_stmt.rs` `StmtKind::Let` 分支调用。**方向 4（跨分支一致性）**：`types.rs` 新增 `check_branch_shape_compat(then_ty, else_ty, span, context)`——if/else 与 match arms 两侧 shape 都含静态信息（Known/Symbol）时必须可广播，否则报 TypeError；`lower_expr.rs` If 分支与 Match 分支调用。**方向 3（标准库符号维度标注）**：`std/nn/linear.th`、`std/nn/attention.th`、`std/nn/feedforward.th` 三个文件改用符号维度标注（如 `Tensor[f64, M, K]`、`Tensor[T, S_q, D_k]`），让标准库 API 自文档化。**方向 2（跨函数 shape 求解）**：`types.rs` 新增 `merge_return_shape(scope_ret, fn_def_ret)`——`resolve_call_type` 调用函数时若 self.functions 中有更精确的 return_type（body lower 后合并的）用它；`lower_stmt.rs` 函数定义 body lower 后调用 `check_and_merge_tensor_shape(&ret_ty, &lowered_body.ty, "函数返回值")?`（原 unwrap_or 吞错改为 `?` 传播，让 body 与注解 shape 静态冲突时报错）。**测试** `tests/shape_check_compile_test.rs` 33→73 项（+40 项）：方向 1 ×9、方向 4 ×9、方向 3 ×7（含 attention/feedforward 去 `<T>` 泛型以便注册到 scope）、方向 2 ×10、组合 ×3；7 项 `#[ignore]` 占位（3 项高开销：大 tensor shape 推断/深层跨函数链/大量 match arms；4 项等待战略方向 B 参数 shape unification：linear/attention/feedforward 错误参数 shape 检测）。验证：shape_check_compile_test 66 passed + 7 ignored；autodiff 52/52、stdlib 114/114、shape_check_test 16/16、vm_autodiff 15/15、selfhost_frontend --release 4/4 全绿；自举路径 A 通过（tenthc/main.th → Full compiler compiled to tenthc_full.wasm）；cargo build --release exit 0。**已知限制**：(1) 跨函数 shape 求解只做返回值传播，参数 shape unification 未实现（属战略方向 B 范畴）；(2) selfhost_frontend debug build 栈溢出（tenthc_lowers_own_source），release build 通过，属预存 debug 栈空间问题；(3) generic_test/memory_test 预存编译错误（LiveCounter/RuntimeLimits/MemoryConfig 未找到，与本次改动无关）。
>
> **2026-07-01 编译期 Shape 检查 Phase 3（算子覆盖扩展）**：补齐运行时已有但 shape 未推断的算子。**新增 shape 推断**：`permute(dims...)` 按字面量索引重排原 dims（如 `[3,8,5].permute(2,0,1)`→`[5,3,8]`，新增 `literal_int_args` 辅助函数提取字面量整数列表）；`broadcast_to(shape...)` 字面量参数即目标 shape；`cat(other, dim=0)` 沿 dim 拼接、dim 维相加（如 `[2,3].cat([3,3],0)`→`[5,3]`）；`argmax`/`argmin` 返回 i64 标量；`gelu`/`masked_fill` 保持原 shape。**新增测试** `tests/shape_check_compile_test.rs` 新增 8 项（共 25→33 项）：permute 重排、broadcast_to 推断、cat dim=0/dim=1 相加、argmax 标量、gelu/masked_fill 保持、flatten 1D。验证：`cargo build --release` exit 0；自举路径 A 通过（tenthc/main.th → Full compiler compiled）；shape_check_compile_test 33/33、autodiff 52/52、stdlib 114/114、f32_autodiff 12/12、f32_runtime 7/7、generic_tensor 4/4、vm_autodiff 15/15 全绿，0 回归。
>
> **2026-07-01 编译期 Shape 检查 Phase 2**：在 Phase 1 基础上扩展符号维度传播与更多算子 shape 规则。**符号维度同名等价检查**：`check_method_shape` 的 matmul 分支新增 `Symbol vs Symbol` 比较——同名 Symbol（如 `Tensor[f64, M, K] @ Tensor[f64, K, N]` 的 K）视为兼容，不同名（如 K ≠ P）报 `TenthError::TypeError`，错误信息含内侧维度对比。**归约算子 axis 降维**：`sum`/`mean`/`max`/`min` 合并为统一分支，新增 `literal_axis_arg` 辅助函数提取字面量 axis 参数——`x.sum(0)` 移除第 0 维（[3,4]→[4]）、`x.sum(1)` 移除第 1 维（[3,4]→[3]）、无参数 `x.sum()` 全部降维到标量、变量参数（如 keepdim 标志）保守保持原 shape。**reshape/view 字面量参数推断**：`x.reshape(3, 4)` 改用 `shape_from_int_args` 推断新 shape 为 [Known(3), Known(4)]，动态参数 `x.reshape(n, m)` 返回 [Any]。**Phase 2 验证已自动工作的能力**：let 传播（let 从 init.ty 继承 shape，scope.lookup_var 返回完整 Type 含 dims）、函数参数 shape 约束（参数类型从签名注入 scope，函数体内 matmul 检查自动生效）。**新增测试** `tests/shape_check_compile_test.rs` 新增 7 项（共 19→25 项）：sum(0)/sum(1)/sum() 三项、mean(0) 一项、reshape 字面量/动态两项、let 传播/参数约束/符号维度同名/不同名四项。**文档更新**：能力全梳理符号维度从 ⚠️ 数据结构就绪→✅ Phase 2 已实现同名等价检查。验证：`cargo build --release` exit 0；自举路径 A 通过（tenthc/main.th → Full compiler compiled）；shape_check_compile_test 25/25、autodiff_test 52/52、stdlib_test 114/114 全绿；fixpoint_runtime_benchmark 为 pre-existing 失败（stash 验证确认，与本轮改动无关）。
>
> **2026-06-30 编译期 Shape 检查 Phase 1**：实现路线图阶段 3 核心卖点（不可替代性）的第一阶段。**数据结构层已就绪**：`Type::Tensor { dtype, dims }` + `Dim::Known/Symbol/Any` + `TypeAnnotation::Tensor` + Parser 已能解析 `Tensor[f64, 3, 4]`/`Tensor[f64, M, K]`/`Tensor[f64, ..]`。**本轮新增**：`hir/lower/types.rs` 新增 `broadcast_shapes(l, r)` 函数（NumPy 广播规则，从右往左对齐，支持 Known/Symbol/Any）；`infer_binary_type` 增强——Tensor+Tensor 二元运算尝试 broadcast 推断结果 shape（兼容则返回精确 dims，否则保守 Any）；`resolve_method_type` 新增 `matmul` 分支——2D (M,K)@(K,N)→(M,N) 静态推断，K 不匹配返回 Unknown；`transpose` 2D 时反转两维（修复 attention 测试回归：k.transpose() 现被正确推断为 [8,3] 而非 [3,8]）；`resolve_builtin` 构造函数（zeros/ones/randn/rand/tensor/tensor_from_vec）新增 `shape_from_int_args`——字面量参数 `zeros(3,4)`→[Known(3), Known(4)]，动态参数 `zeros(n)`→[Any]；新增 `check_binary_shape_compat` 和 `check_method_shape` 两个编译期检查函数——shape 不兼容时返回 `TenthError::TypeError`（含人类可读错误信息如"编译期 matmul shape 不兼容：[3, 8] @ [3, 8]（内侧维度 8 ≠ 3 必须相等）"）；`lower_expr.rs` Binary 分支和 MethodCall 分支（两处）调用 check 函数。**新增测试** `tests/shape_check_compile_test.rs` 12 项：matmul K 不匹配编译期报错、matmul 正确 shape 编译通过、matmul + transpose 编译通过（attention 模式）、忘记 transpose 编译期报错、二元加法不兼容 shape 报错、同 shape/标量广播/行广播编译通过、zeros/randn 字面量 shape 推断、动态参数返回 Any、transpose 2D 反转。**文档更新**：能力全梳理 编译期 Shape 检查从 ❌→⚠️ Phase 1 已实现、符号维度从 ❌→⚠️ 数据结构就绪；语言参考手册/README 阶段 3 状态从 📋 规划中→⚠️ Phase 1 已实现。验证：`cargo build --release` exit 0；自举路径 A 通过；`cargo test --release -- --skip fixpoint_runtime --skip three_stage` 全绿（shape_check_compile_test 12/12、autodiff_test 52/52、generic_tensor_test 4/4、stdlib_test 114/114 等）。
>
> **2026-06-30 第三类技术债扫尾 + native 构造函数泛型化**：完成外部评价者第三类批评（小杂项）+ 第一类遗留的 native 泛型化。**native 构造函数泛型化**：`hir/lower/lower_expr.rs` 的 `ExprKind::GenericCall` 分支新增 `NATIVE_GENERIC_CTORS` 列表（randn/zeros/ones/rand/tensor/tensor_from_vec），支持 `randn<f32>(d)` 语法；类型参数必须是具体 BaseType，f32 dtype 映射到运行时名字 `randn_f32`/`zeros_f32`/`ones_f32`/`rand_f32`，f64 保持原名。新增 2 个测试（`native_generic_ctor_f32_lowering` / `native_generic_ctor_f64_lowering`），generic_tensor_test 4/4 通过。**make_* 构造函数改泛型**：`std/nn/layer_norm.th`（make_layer_norm）、`std/nn/feedforward.th`（make_feedforward_params）、`std/nn/transformer.th`（make_transformer_block_params）、`std/nn/positional_encoding.th`（positional_encoding）四个函数改泛型 `<T>`，统一用 `randn<T>(...)`/`zeros<T>(...)`/`ones<T>(...)`；positional_encoding 注释更新（原"kept as f64 because randn native is f64-only"已过时）。**IMPORT_COUNT 魔数修复**：`compile/wasm/mod.rs` 新增 18 个 `HOST_*` 具名常量（HOST_PRINTLN=0 ... HOST_TENSOR_FROM_VEC=17），`IMPORT_COUNT` 改为 `HOST_TENSOR_FROM_VEC + 1` 推导；`compile/wasm/compile.rs` 中 33 处 `Call(N)` 魔数全部替换为 `Call(HOST_*)` 引用，`host_call_index` 函数的 match 分支也用具名常量；新增 import 只需在 mod.rs 加一行常量 + sections.rs/host.rs 注册。**AUDIT §7.1 过时条目**：#1（lexer.th 字面量硬编码 0）和 #2（parser.th 字段名不存储）经复核均已修复，标记为 ~~已修复~~ 并附证据行号。**tools 路径统一**：8 个文件（AUDIT/CODE_WIKI/DEPS/语言参考手册/MEMO/SECURITY/security_review/engine.rs）共 15 处 `tools/tenthpm/`、`tools/lsp/` 统一为 `tenth/tools/tenthpm/`、`tenth/tools/lsp/`（与实际磁盘路径一致）。验证：`cargo build --release` exit 0；`cargo test --release -- --skip fixpoint_runtime --skip three_stage` 全绿（generic_tensor 4/4、stdlib 114/114、wasm_backend_minimal 7/7、vm_autodiff 15/15 等）；fixpoint_runtime 和 three_stage 两个 wasmtime 路径测试为 pre-existing 失败（stash 后仍失败，WASM-B 0 bytes，与本轮改动无关）。
>
> **2026-06-29 安全审查修复（第二轮）**：完成剩余 9 项中的 7 项（H-2, H-4, L-1, L-3, L-4, L-5, L-6）；L-2（DefaultHasher）已在 H-7 中一并修复，L-7（compile_host 写文件）已在 H-2 中一并修复。**H-2（文件 I/O 沙箱）**：`tenth/src/runtime/limits.rs` 新增 `FsSandbox` 类型（`check_read`/`check_write` 用 `canonicalize` + `starts_with` 防路径穿越，不存在路径规范化父目录再拼接文件名）；`Vm`/`Interpreter` 新增 `fs_sandbox` 字段，所有文件 I/O 原生函数（read_file/write_file/write_bytes/read_bytes/path_exists/path_is_file/path_is_dir/mkdir/list_dir/file_size/remove_file/copy_file/rename_file/compile_host/compile_program/load_weights/save_weights）必须经过沙箱校验；`main.rs` 新增 `--fs-root <dir>`/`--read-only`/`--fs-cwd` 命令行选项。**H-4（CPU/时间限制）**：VM 主循环和 Interpreter `tick()` 新增独立 `loop_counter`/`tick_counter`，每 4096 步检查 `deadline_ms`（墙钟超时独立于 `step_budget`，用户可只设 `--timeout` 不设步数预算）；`main.rs` 新增 `--timeout <secs>` 命令行选项，`parse_timeout_ms` 用 `checked_mul`/`checked_add` 防溢出。**L-1（extract_package_name 清理）**：验证确认 `extract_package_name` 仅在 `safe_package_name_from_git` 内被调用，所有调用方（install_local/install_global/add_git_dependency）都经 `validate_package_name` 校验，security_review.md 引用的 install.rs:177 为过时信息。**L-3（days_to_date 溢出）**：`main.rs` 和 `interpreter.rs` 两处 `days_to_date` 入口加 `if days > u64::MAX - EPOCH_OFFSET { return (0,0,0); }` 防 `days + 719468` 溢出 UB。**L-4（git URL 协议限制）**：`manifest.rs::is_git_url` 默认仅放行 `https://`，`http://`/`git://`/`ssh://`/`.git` 后缀需环境变量 `TENTH_ALLOW_INSECURE_GIT=1` 显式 opt-in。**L-5（git clone hooks 禁用）**：三处 `git clone` 调用（install_local/install_global/add_git_dependency）加 `--config protocol.file.allow=deny` 和 `--config protocol.git.allow=deny`，克隆后调用新增 `manifest::disable_hooks()` 将 `core.hooksPath` 指向空设备（Unix `/dev/null` / Windows `nul`）。**L-6（cargo-audit 建议）**：`DEPS.md` 新增"供应链安全"章节，说明 `cargo audit` 和 `cargo deny` 用法及 CI 集成建议。验证：`cargo build`（dev+release）exit 0，自举路径 A 验证通过（`tenthc/main.th` → `tenthc_full.wasm`），tenthpm 编译通过。**已知限制**：`cargo test` 因 rustc 1.95.0 ICE（`wasm.rs:2077` 方法调用解析，仅在 `--test` 模式触发，非本项目代码问题）无法编译测试二进制；ICE 文件未被本轮修改，属预存在 rustc bug。
>
> **2026-06-29 安全审查修复（第一轮）**：以最严谨安全员身份完成全面审查，识别 25 项问题（2 致命 / 8 高危 / 8 中等 / 7 低危），本轮修复 17 项（C-1, C-2, H-1, H-3, H-5, H-6, H-7, H-8, M-1 ~ M-8）。审查报告 `security_review.md`，威胁模型披露 `SECURITY.md` 重写。**C-1（tenthpm 路径穿越）**：`tenth/tools/tenthpm/src/manifest.rs` 新增 `validate_package_name` / `safe_package_name_from_git` / `ensure_within` / `safe_to_remove_dir` 四个集中校验函数；`install.rs` / `add.rs` 全路径调用，所有 `fs::remove_dir_all` 前必须通过 `safe_to_remove_dir`，所有 `target_dir` 计算后必须通过 `ensure_within`。**C-2（SECURITY.md 失实声明）**：原声称"0 处 unsafe"实为 41+ 处，JIT 为默认执行路径，重写公开真实威胁模型与沙箱选项 `--fs-root`。**H-1（JIT hostcall 越界）**：`tenth/src/compile/jit/hostcalls.rs` 引入 `MAX_HOSTCALL_ARGS = 1<<20` 与 `safe_slice` 统一闸门，所有 `from_raw_parts` 改用；`host_make_map` / `host_new_struct` 加 `count.checked_mul(2)`；`host_make_tensor` 加 `rows.checked_mul(cols)`。**H-3（run_file 无限制）**：`tenth/src/main.rs` 新增 `parse_memory_config` 与 `run_file(path, config)`，fallback 路径用 `Interpreter::with_limits`；`--no-limits` 显式退出沙箱。**H-5（time_sleep_ms 负数 DoS）**：拒绝负数与 > 24h 的请求。**H-6（JSON 解析器递归/转义）**：`JSON_MAX_DEPTH=256` 防栈溢出；`json_unescape` 与 `simple_json_split` 修复 `\"` 状态机。**H-7（DefaultHasher 可预测）**：`random` 改用 `rand::thread_rng().r#gen()`（CSPRNG）。**H-8（WASM 宿主 ptr as usize 符号扩展）**：`tenth/src/compile/wasmtime_host.rs` 新增 `safe_offset` / `read_cstr` / `MAX_ALLOC_BYTES=16MiB`，17 个 host import 全部改用；`tenth_alloc` 拒绝负数 size 与 > 16MiB 请求。**M-1（transmute 尺寸断言）**：`tenth/src/compile/jit/context.rs` 加 `assert_eq!(size_of::<*const u8>(), size_of::<JitFn>())`。**M-2（mem-strict panic 杀 REPL）**：`tenth/src/repl.rs:241-245` 改为返回 `TenthError::RuntimeError`。**M-3（release 溢出检查）**：`tenth/Cargo.toml` 加 `[profile.release] overflow-checks = true, panic = "unwind"`。**M-4（host_make_tensor 溢出）**：见 H-1。**M-5（JitContext Drop 显式清理）**：`context.rs` impl Drop 调 `cache.clear()`。**M-6（FFI unwind UB）**：`hostcalls.rs::invoke_jit` 包 `catch_unwind`，panic 时写 `Value::Unit` 并返回 false。**M-7（Arena 算术溢出）**：`tenth/src/runtime/arena.rs` `alloc` 用 `checked_add` / `checked_mul`；`scope` 用 `saturating_sub` 防下溢 panic。**M-8（WASM 宿主 bump allocator 越界）**：见 H-8。**验证**：`cargo build --release` 成功；`cargo test --lib` 10/10 通过；`cargo test --features mem-debug --test memory_test` 17/17 通过（含 arena_scope_rolls_back_counter / arena_overflow_returns_none 验证 M-7 修复）。**未引入回归**：stash 验证原始代码下 `cargo test --no-run` 同样因 wasmtime/cranelift 依赖 rlib 格式问题失败，与本次修复无关。**未修复项**：L-1 ~ L-7 低危项（命名约定、文档同步等），保留至下轮。
>
> **2026-06-25 优先级调整**：自举固定点 spec 暂停，转向能力梳理第一梯队。经评估，自举固定点（C4）是纯工程内部指标，对语言采用率无影响；而能力梳理第一梯队 4 项（f32 张量、编译期 Shape 检查、GPU 算子、异步/并发）是"生存级"缺口，直接决定 Tenth 能否称为"AI 原生语言"。决策：暂停自举固定点 spec（Phase 1 代码 + Phase 2 审计结论作为存量资产保留），分四个独立 spec 推进第一梯队。自举固定点重启条件：第一梯队完成 或 用户明确指示。
>
> **2026-06-25 更新**：f32 张量 spec Phase 1 完成。`TensorData` enum（F32/F64 变体）替换原 `ArrayD<f64>` 单一存储，`Tensor` 结构新增 `dtype` 字段。新增 8 个 f32 构造器（zeros_f32/ones_f32/full_f32/rand_f32/randn_f32/eye_f32/arange_f32/from_vec_f32）+ 2 个 dtype 通用构造（zeros_with_dtype/ones_with_dtype）。运算方法（add/sub/mul/div_tensor + 标量运算）按 dtype 分支：f32⊙f32→f32 保持精度，f32⊙f64→f64 提升。`BaseType` 添加 `Copy` trait。兼容层策略：TensorData 实现 mapv(FnMut)/view/as_standard_layout/iter/broadcast 等 ArrayD<f64> 接口（F32 自动 cast 为 f64 视图），外加 Mul/Add/Sub/Div/Neg/Index 算术 trait impl，让 autodiff.rs/interpreter.rs/vm.rs/main.rs 外部代码零改动通过编译（Phase 1 语义降级：F32 经兼容方法后变为 f64，真正 f32 路径在 Phase 3/4 恢复）。跨模块影响：autodiff.rs:360/365 `data.clone()`→`as_f64_view()`；main.rs:650 + vm_autodiff_test.rs:80 `into_raw_vec()`→`from_tensor_data()`；repl.rs Display 通过 TensorData Display impl 解决。新增 `tests/f32_tensor_test.rs`（23 用例，覆盖构造器/dtype标记/运算保持/提升/标量/reduction/Display/broadcast）。全量 539 测试通过（516+23），0 回归，自举三路径未破坏。
>
> **2026-06-25 更新**：f32 张量 spec Phase 2 完成（前端贯通）。`TokenKind::FloatLiteral(f64, BaseType)` 携带 dtype，Lexer `read_number` 末尾检测 `f32`/`f64` 后缀（三字符边界检查避免 `3.14factor` 误匹配）。AST/HIR `Literal::Float(f64, BaseType)` 双字段贯通。Lower 新增三个推断辅助函数：`infer_tensor_dtype(args)`（任一参数 F32 → F32，否则默认 F64）、`infer_scalar_dtype(args, fallback)`（标量函数 dtype 跟随输入）、`promote_float_dtype(l, r)`（F64 优先 > F32 > F16 > BF16 > 整数）；`resolve_builtin` 重写为参数化推断（`tensor/rand/randn/zeros/ones/param/new_grad/cross_entropy` 全部走推断，不再硬编码 F64）；`infer_binary_type` 重写支持 Tensor-Tensor dtype 合并、Tensor-标量保持、标量-标量提升。字节码：`Op::PushFloat32(f32)` 新增（opcode 45），`Op::MakeTensor(rows, cols, dtype)` 第三参数 dtype_code（0=F64, 1=F32）；序列化/反序列化 MakeTensor 增加 1 字节 dtype。`Value::Float32(f32)` 新运行时值类型，`type_of(Value::Tensor)` 改为读 `t.dtype()`（修复 f32 Tensor 类型推断错误）；VM `add_priv/sub_priv/mul_priv/div_priv/compare/vm_eq/neg` 各增加 5 个 Float32 分支（含 Int↔Float32、Float↔Float32 混合提升）；`sum/mean/max_val` natives 按 tensor.is_f32() 分支返回。新增 `to_f32` / `to_f64` 内置函数（lower + native + interpreter），作为 `as f32` / `as f64` 表达式的 API 替代（AST 层未扩展 `as` 语法）。WASM 后端 `compile_literal` Float 分支按 dtype 选择 F32Const/F64Const。JIT 路径 `PushFloat32` 暂降级为 f64 调用 `host_make_float`（Phase 5 补齐真正 f32 JIT）。新增 `tests/f32_frontend_test.rs`（13 用例，覆盖 G1 lexer 后缀、G2 HIR dtype 贯通、G4 Value::Float32 算术、E5 回归）。全量 552 测试通过（539+13），0 回归，自举路径 A 验证通过（`cargo run --release -- run tenthc/main.th` → Full compiler compiled to tenthc_full.wasm）。
>
> **2026-06-25 更新**：f32 张量 spec Phase 3 完成（运行时贯通）。interpreter.rs 三处硬编码 F64 return_type（tensor/zeros/ones 内置函数）改为 `Type::Unknown`，让 VM 运行时按实际 Value 推断 dtype。`lower.rs:596-608` TensorLiteral dtype 推断重写：原硬编码 `dtype: BaseType::F64`，现遍历 lowered 元素，任一为 F32 → Tensor dtype=F32，否则 F64（修复 `[[1.0f32, 2.0f32]]` 被误标记为 F64 的 bug）。interpreter.rs TensorLiteral 评估按 HIR `ty` 字段 dtype 分支构造 f32/f64 Tensor。`to_f32` / `to_f64` 加入 lower.rs 和 interpreter.rs builtin 白名单（原仅含 `to_float`，导致 `to_f32(2.0)` 被当作 undefined variable 报错）。main.rs native 函数 f32 支持：`randn_f32`（f32 Box-Muller）、`tensor_from_vec` 按 Vec 元素 dtype 判断、`grad` 按参数 dtype 返回零张量（f32→`zeros_f32`）、`cross_entropy` 按 logits dtype 构造 loss tensor；所有 `math_*` 函数（math_tan/asin/acos/atan/atan2/sinh/cosh/tanh/log10/log2/exp/pow/floor/ceil/round）+ abs/sqrt 增加 Float32 分支。新增 `tests/f32_runtime_test.rs`（7 用例：VM 算术 f32 分支、混合提升、Int+Float32 提升、MakeTensor dtype、randn_f32 native、to_f32 native、sqrt f32）。全量 559 测试通过（552+7），0 回归，自举路径 A 验证通过。
>
> **2026-06-25 更新**：f32 张量 spec Phase 4 完成（自动微分 f32 支持）。**关键发现：方案 B 天然实现，无需改造 autodiff.rs**。调研发现 `backward` 函数 `node_grads: Vec<Option<ArrayD<f64>>>` 固定 f64 计算（反向用 f64），`acc_grad(&ArrayD<f64>)` 已在 Phase 1 按 tensor.dtype 转换存储（f32 参数→grad 存为 F32），`grad` native 已在 Phase 3 按 dtype 返回——方案 B 链路（前向 f32 + 反向 f64 + 梯度按参数 dtype 写回）已完整。误差测量：纯算术（Add/Sub/Mul/Neg/ReLU/MatMul/Sigmoid/Exp/Log）误差 < 1e-10；涉及 f32 前向 exp/sum 的算子（Softmax/CrossEntropy）误差 ~1e-7（f32 机器精度，spec 允许 < 1e-5），E4 退出条件未触发。唯一代码改动：vm.rs 补齐 `Float32 × Tensor` / `Tensor × Float32` 算术分支（add/sub/mul/div 各 2 个，Phase 2 遗漏修复，`2.0f32 * f32_tensor` 原报"* 类型不匹配"）。新增 `tests/f32_autodiff_test.rs`（12 用例：new_grad/param 注册、simple gradient、ReLU 正负梯度、matmul、softmax/sigmoid/cross_entropy backward、exp/log chain、grad dtype 保持、extreme input stability、chain rule）。全量 571 测试通过（559+12），0 回归，自举路径 A 验证通过。
>
> **2026-06-25 更新**：f32 张量 spec Phase 5.0（泛型化可行性验证）+ Phase 5.1（WASM 后端 f32→f64 提升）完成。**Phase 5.0 调研结论**：用户原选泛型化路线（`Tensor[T, ..]`）在当前类型系统下不可行——`Type::Tensor.dtype: BaseType` 是硬约束（types.rs:23），`from_annotation` 对非 Base dtype fallback 到 F32（types.rs:141），`substitute_type` 不递归 Tensor（lower.rs:1991）；路径 B（扩展支持）需改 Type 核心枚举 + ~20 处 match + 编译后端 + 自举三路径对齐，工作量属独立专项级。用户确认降级为路径 C（后缀复制）策略推进 Phase 5.4 标准库 f32 化。**Phase 5.1 实际执行策略 A（f32→f64 提升）**：原 spec 计划策略 B（完整 f32 指令集 + 4 字节 stride + Type section 重构），但调研后发现 wasm.rs 已有 15 处 F32→F64 提升路径，唯一破坏策略 A 的是行 1417 `F32Const`（死代码 bug：压入 f32 栈但下游期望 f64，导致 WASM 验证失败，无 e2e 测试覆盖）。修复：`wasm.rs:1416-1420` 将 `Literal::Float(n, dt)` 的 F32 分支改为统一 `F64Const`，与下游 F64 处理对齐。新增 `tests/f32_wasm_test.rs`（14 用例：f32 字面量算术 add/sub/mul/div、f32 后缀解析、一元负号、比较 eq/lt、f32-f64 混合、f32-i64 混合、let 绑定、链式算术、main wrap 路径）。全量 219 测试通过（f32_tensor 23 + f32_frontend 13 + f32_runtime 7 + f32_autodiff 12 + f32_wasm 14 + wasm_backend_minimal 7 + parity 129 + selfhost 4 + 其他 10），0 回归，自举路径 A 验证通过（tenthc/main.th 编译成功）。
>
> **2026-06-25 更新**：f32 张量 spec Phase 5.2（tenthc 自举同步）完成。**实际方案：复用 Token.ival 字段存 dtype**，非原 spec 计划的扩展 Token/HirExpr 结构。调研发现 FloatLiteral 时 ival 字段原本未使用，复用为 dtype 标记（0=F64, 1=F32）避免扩展两侧结构 + bridge.rs 对齐工作；HirType.sub 字段语义化（0=I64, 2=F64, 3=F32, 4=Bool, 5=Str），与 Rust 侧 `Literal::Float(f64, BaseType)` 语义对齐。4 step 改动 + 5 步渐进验证（每 step 跑 `cargo run --release -- run tenthc/main.th` + `cargo test --test selfhost_frontend` 4/4）：step 1 lower.th 加 `is_type_f32`/`is_type_float` + Binary 算术 F64>F32>I64 优先级（与 Rust lower.rs:1286-1290 infer_binary_type 对齐）+ float 字面量按 e.ival 决定 dtype；step 2 lexer.th 加 f32/f64 后缀检测（FloatLiteral + IntLiteral 两路径，三字符边界检查避免 `3.14factor` 误匹配；整数后跟 f32 也变 FloatLiteral，与 Rust 语义一致）+ parser.th 保留 tok.ival 到 Expr.ival；step 3 wasm.th `is_expr_float` 识别 sub==3 (F32)，策略 A 统一发 f64 指令；step 4 parser.th let 语句检测类型注解是否为 `f32`（单 token 类型），若且初始表达式是 FloatLiteral 则把 ival 强制改为 1（让 `let x: f32 = 3.14` 字面量升级为 F32 dtype）。自举三路径未破坏：自举成功（Full compiler compiled）+ selfhost_frontend 4/4 + Phase 5.1 e2e 无回归（f32_wasm_test 14/14）。退出条件 E6 未触发，E2 已记录（4 step 改动 ~150 行，远未达 800 行）。
>
> **2026-06-26 更新**：f32 张量 spec Phase 5.4（标准库 nn/optim/init f32 化，精简版）+ Phase 5.5（f32 native 构造函数补全）完成。**路径 C 后缀复制策略**：用户选精简版，仅复制「带硬编码 f64 标量参数」+「构造函数调用 f64 内置」两类文件，不复制纯方法包装。共 14 个 std 文件新增 26 个 `_f32` 副本函数：optim/{sgd,adam,adagrad,rmsprop}.th（4 文件，标量参数 `lr/momentum/decay: f32`、字面量 `1.0f32 - decay`）；init/initializers.th（6 个构造函数 `xavier_uniform_f32`/`xavier_normal_f32`/`he_normal_f32`/`he_uniform_f32`/`zeros_init_f32`/`constant_init_f32`，调用 `randn_f32`/`zeros_f32`、字面量 `6.0f32`/`2.0f32`）；nn/{dropout,batchnorm,loss,attention,multihead_attention,layer_norm}.th（标量 `rate/eps/dropout_p: f32`、`scale = 1.0f32 / sqrt(d_k)`、`-1000000000.0f32` 替代 lexer 不支持的科学计数法 `-1e9`）；nn/{positional_encoding,feedforward,transformer}.th（构造函数 `make_*_f32` 调用 `randn_f32`/`zeros_f32`/`ones_f32`、`0.00001f32` 替代 `1e-5`）。**Phase 5.5 内置补全**：`zeros_f32`/`ones_f32`/`rand_f32` Tensor 层方法已存在（tensor.rs:318-338）但 native 未注册——lower.rs 3 处（行 352 builtin 列表、行 1217 类型推断、行 1815 变量收集跳过列表）+ interpreter.rs 2 处（行 525 builtin 列表、行 3499+ native 实现）补全注册。**Interpreter f32 路径修复**（测试驱动发现 4 处缺失）：(1) Add/Sub/Mul/Div 各补 5 个 Float32 标量×标量分支（Float32+Float32→Float32、Int+Float32→Float32、Float32+Int→Float32、Float+Float32→Float、Float32+Float→Float，与 lower.rs:1286 promote_float_dtype 规则对齐）；(2) Add/Sub/Mul/Div 各补 2 个 Float32×Tensor / Tensor×Float32 分支（`*s as f64` 转换后调用 scalar 方法，模式与 vm.rs Phase 3 一致）；(3) `grad` native 对 f32 张量返回 f32 zeros（原 `Tensor::zeros` 硬编码 f64，改用 `Tensor::zeros_with_dtype(&shape, p.dtype())`）；(4) `dropout` 方法按 tensor dtype 分支保持精度（f32 路径用 `as_f32().mapv` + f32 随机 + `from_data_f32`，f64 路径不变）。新增 `tests/f32_stdlib_test.rs`（10 用例：3 native 构造函数、5 内联 _f32 副本端到端、2 标量×tensor dtype 行为验证）。全量测试 0 回归：f32_stdlib 10/10、stdlib 114/114、autodiff 5/5、vm_autodiff 15/15、f32_runtime 7/7、f32_autodiff 12/12、integration 14/14、selfhost_frontend 4/4。自举路径 A 验证通过。**关键发现**：`Tensor::mul_scalar(scalar: f64)` 内部按 tensor dtype 分支（F32 分支把 scalar cast 为 f32），故 f64 标量 × f32 tensor 结果仍是 f32（不提升）——`_f32` 副本的主要价值在于使用 f32 字面量保持标量运算精度，而非防止 tensor dtype 提升。
>
> **2026-06-25 架构决策**：自举长期路线图确立。重新审视 WASM 路径与"完全摆脱 Rust"终极目标的契合度，确认 WASM 路径的根本局限（执行运行时 wasmi/wasmtime 是 Rust 写的，"完全摆脱 Rust"在 WASM 路径上不可能达成）。决策采用两阶段策略：阶段 1（当前 spec）WASM 路径达成 C4 固定点，阶段 2（未来 spec）实现 native 后端达成运行时自举。Native 后端三条子路径评估：C2-直接机器码（5000-10000 行 Tenth，无外部依赖，推荐）、C2-LLVM IR（引入 C++ 依赖违背精神，不推荐）、C3-C 翻译（2026-06-04 已尝试并删除，内存管理未解决，重新评估需先解决所有权→C 内存映射）。新增 `docs/superpowers/self-hosting-roadmap.md` 长期路线图文档，spec.md §8 追加架构决策记录。
>
> **2026-06-25 更新**：自举固定点攻关 spec 重构。前置 spec `docs/superpowers/specs/2026-06-20-self-hosting-master-plan/` 已归档（Phase A-D 全部完成，D1-D7 共 129 用例全绿，仅 C4 固定点未达成）。新建 `docs/superpowers/specs/2026-06-25-self-hosting-fixpoint/`（含 spec.md / tasks.md / checklist.md），重新设计 C4 攻关方案：5 阶段架构（运行时迁移→确定性保证→端到端跑通→固定点达成→CI 集成），锁定 4 项设计决策（Wasmtime JIT / wasmi 保留 / 字节级等价不可达成则终止 / 大重构需求则终止），明确 4 项硬性退出条件（不允许降级实现）。同步修正旧 spec 文档不一致：spec.md 能力差距矩阵（Trait/泛型/借用/闭包/Tensor 已实现）、tasks.md D2/D6 验收 checkbox、checklist.md 状态描述。
>
> **2026-06-25 更新**：Phase D — D3（借用检查）+ D5（闭包 WASM 后端）完成，Phase D 全部 7 项（D1-D7）达成。D3：tenthc lower.th 新增 Ownership 状态存储（Owned/SharedRef/ExclusiveRef/Moved）、check_use/check_borrow_shared/check_borrow_mut 借用检查函数、release_borrows 借用释放；parser.th 补全 `&mut` 解析。D5：tenthc lower.th 实现 free_vars_in 递归捕获分析；wasm.th 新增 table/elem section + call_indirect 闭包调用 + env 装箱（tenth_alloc 分配 captures struct）；修复闭包解析双 advance bug（parse_primary 已消费 `|`，闭包块重复 advance 吞掉首个参数）；修复 `&&`/`||` 误用 i64 and/or 指令（WASM 比较结果为 i32，需用 I32And/I32Or）。新增 5 项 parity 测试（D3×3 + D5×2），parity_test 124→129 项全绿，499+ 测试无回归。
>
> **2026-06-25 更新**：Phase D — D2（泛型实例化）+ D6（Tensor WASM 后端）完成（阶段 1 并行）。D2：tenthc 新增泛型参数解析 `<T, U>`、泛型调用前瞻启发式 `looks_like_generic_call`、`substitute_type` 类型替换与 mangled name 实例化（`fn_name_T1_T2`），Rust 侧 lower.rs 同步将 GenericCall 改写为普通 Call。D6：tenthc wasm.th 新增 tensor_from_vec host import (idx 17) 与 TensorLiteral 编译，Rust 侧 wasm.rs 同步；修复 parser.th parse_primary 双 advance bug（tensor 字面量解析失败根因）。修复 three_stage.rs / wasm_backend_minimal.rs linker 缺失 tensor_from_vec 注册导致的回归。新增 4 项 parity 测试（D2×3 + D6×1），parity_test 120→124 项全绿，499+ 测试无回归，自举管道通过。
>
> **2026-06-25 更新**：Phase D — D1（Trait 系统）完成。tenthc 新增 trait 定义/inherent impl 解析与方法静态分派（mangled name `__<Type>_<method>`）；修复 lexer self token 缺少 sval、lowerer self 参数 type_ann 未覆盖为 impl 类型名两处 bug；Rust 母编译器同步实现 inherent impl 方法分派。新增 3 项 D1 parity 测试全部通过，parity_test 117→120 项全绿，499+ 测试无回归。
>
> **2026-06-25 更新**：Phase D — D4（native 函数对齐）完成。修复 tenthc lexer 数字解析双处理 bug（`lexer_peek` 返回当前字符导致 `5`→`55`）；修复 `char_to_ascii` 缺失空格等常见字符（补全 30+ ASCII 字符）。4 项 Slice parity 测试全部通过，parity_test 117 项全绿。
>
> **2026-06-24 更新**：测试数 350→499（新增 parity_test 112 项 VM/Interpreter 一致性、shape_check_test 16 项 shape 检查；selfhost_verify→selfhost_frontend 重构）。文档重复信息全面清理，交叉引用体系建立。
>
> **2026-06-15 更新**：文档维护 — 自动微分算子数修正为 21（新增 LayerNorm/GELU），标准库模块补全（collections/string/utils/nn 扩展），张量方法补全（gelu/layer_norm/cat/masked_fill/permute/broadcast_to/max_val）。
>
> **2026-06-04 重大变更**：`tenth/src/compile/`（MIR→C 编译管线）、`tenthc/codegen/`、`tenthc/runtime.c` 已删除。
> 原因：生成的 C 代码无内存管理（12 处 malloc / 0 处 free），导致系统级内存耗尽。详见 `SECURITY.md`。
> 自举编译器改为通过 Rust 解释器执行。

## 自举编译器现状（2026-06-10 更新）

自举管线全部由 Tenth 实现，全程走 VM（~0.2s），不再依赖解释器 fallback。

### 自举管线

```
Tenth 源码 → Lexer → Parser → Lowerer → WASM Compiler → .wasm → wasmi
   ✅          ✅       ✅         ✅           ✅           ✅      ✅
  (Tenth)   (370行)  (430行)    (294行)      (703行)     import   add(3,4)=7
```

### 性能

| 指标 | 优化前 | 优化后 |
|------|--------|--------|
| 自举管线执行时间 | ~200s (interpreter) | **~0.2s (VM)** |
| VM fallback 率 | Lexer/Parser 100% | **0%** |
| VM 指令数 | 33 | **45** (41→43→45, PushRange+MoveOp+MakeTensor+MakeClosure) |
| wasmi 加载验证 | ❌ | ✅ `add(3,4)=7` |

### 各层状态

| 层 | 文件 | 状态 |
|----|------|------|
| Token | `tenthc/lexer/token.th` | ✅ enum TokenKind (50+ 变体) |
| Lexer | `tenthc/lexer/lexer.th` | ✅ O(1) 源切片，**VM 全速** |
| Parser | `tenthc/parser/parser.th` | ✅ 递归下降 + method_call，**VM 全速** |
| HIR 类型 | `tenthc/hir/hir.th` | ✅ 紧凑表示 (104 行) |
| Lowerer | `tenthc/hir/lower.th` | ✅ AST→HIR 降级 (306 行) |
| WASM 编译器 | `tenthc/compile/wasm.th` | ✅ HIR→WASM + import 段 (703 行) |
| ~~C Codegen~~ | ~~`tenthc/codegen/cgen.th`~~ | ❌ 已移除 |
| **字节码 VM** | `tenth/src/runtime/vm.rs` | ✅ **45 指令** (33→41→43→45) |
| VM 编译器 | `tenth/src/compile/bytecode.rs` | ✅ HIR→bytecode (含 Enum/Match) |

### VM 指令列表（45 条）

```
0-3:   PushInt/Float/Bool/Str    20-23: Lt/Gt/Lte/Gte
4:     PushUnit                  24-26: Jump/JmpFalse/JmpTrue
5-6:   Pop/Dup                   27-28: Call/CallN
7-8:   Load/Store                29:    MethodCall
9-10:  LoadGlobal/StoreGlobal    30:    Ret
11-15: Add/Sub/Mul/Div/Mod       31-32: MakeVec/MakeMap
16-17: Neg/Not                   33:    NewStruct
18-19: Eq/Neq                    34-35: LoadField/StoreField
                                  36:    IndexGet
                                  37:    SliceStr
                                  38:    MakeEnum
                                  39:    IsEnumVariant
                                  40:    EnumGetField
                                  41:    PushRange       ← v0.3.0
                                  42:    MoveOp          ← v0.3.0
                                  43:    MakeTensor      ← v0.3.1
                                  44:    MakeClosure     ← v0.3.1
```

### 新增验证（2026-06-10）

- [x] 路径 C: Tenth Lexer+Parser+Lowerer+WASM 全链路 → wasmi `add(3,4)=7`
- [x] VM Enum/Match 支持：Lexer/Parser 不再 fallback
- [x] 永久回归测试：`tenth/tests/selfhost_verify.rs` (83/83 pass)
- [x] 性能基准：`tenth run tenthc/boot_full.th` → 0.2s

### 已知限制（当前活跃）

> 带编号的完整缺陷清单见 `AUDIT.md` §六/七。此处为逐版演化日志。

- [x] ~~Closure/GenericCall VM fallback~~ → GenericCall/Move/Range/MakeTensor/MakeClosure 已补全
- [ ] Host import (Vec/String) 为占位实现，WASM 模块需宿主提供真实运行时
- [ ] 三段式自举验证（输出 WASM 再编译自身）因栈溢出未跑通
- [x] ~~大文件 Lowerer 性能~~ → 已解决 (VM ~0.2s)

### v0.3.1 新增（2026-06-14）

#### 闭包捕获

| 组件 | 状态 |
|------|------|
| HIR `HirExprKind::Closure` 新增 `captures: Vec<String>` | ✅ |
| Lowerer `free_vars_in()` 递归分析自由变量 | ✅ |
| Interpreter 闭包创建时从 `resolve_var()` 捕获环境变量 | ✅ |
| VM `MakeClosure(params_count, chunk_idx)` 指令 | ✅ |
| BytecodeCompiler 闭包编译为 MakeClosure | ✅ |

#### VM 补全

| 组件 | 状态 |
|------|------|
| `MakeTensor(rows, cols)` 指令 (opcode 43) | ✅ |
| `MakeClosure(params_count, chunk_idx)` 指令 (opcode 44) | ✅ |
| BytecodeCompiler TensorLiteral 编译为 PushFloat+MakeTensor | ✅ |

#### 文件级导入

| 组件 | 状态 |
|------|------|
| Lowerer `search_paths` 字段 | ✅ |
| Lowerer `try_import_file()` / `load_and_compile_file()` | ✅ |
| `source_to_hir()` 使用 `with_search_paths(vec!["std"])` | ✅ |

#### 错误信息增强

| 组件 | 状态 |
|------|------|
| Scope `check_use/check_borrow_shared/check_borrow_mut` 带 span 参数 | ✅ |
| 消除 3 处 `line: 0, col: 0` 硬编码 | ✅ |

#### 标准库补全

| 模块 | 状态 |
|------|------|
| data/dataloader.th — DataLoader 完整实现 | ✅ |
| utils/serialization.th — save_model/load_model/save_checkpoint | ✅ |
| prelude.th — 更新所有模块索引 | ✅ |

#### 自举编译器同步

| 模块 | 状态 |
|------|------|
| hir/hir.th — captures_start/captures_count/range_inclusive 字段 | ✅ |
| hir/lower.th — closure/array/tensor/move/assign_op/deref 降低 | ✅ |
| parser/parser.th — 闭包/数组/张量/move/复合赋值解析 | ✅ |
| compile/wasm.th — disc 22-30 的 WASM 编译处理器 | ✅ |
| lexer/lexer.th — `/* */` 块注释支持（含嵌套） | ✅ |

#### 测试覆盖提升

| 测试文件 | 新增项 | 状态 |
|----------|--------|------|
| autodiff_test.rs | 20 项（autodiff 8 + closure 4 + tensor 7 + error span 1） | ✅ |

#### 示例集

| 变更 | 数量 |
|------|------|
| 新增示例 | 8（归并排序/二叉搜索树/闭包捕获/Softmax回归/Adam优化器/张量广播/词频统计/矩阵转置与运算） |
| 优化示例 | 18（while→for-in, 闭包捕获增强） |
| 总计 | 33 个示例 |

---

### v0.3.3 新增（2026-06-14）

#### GPU 后端脚手架

| 组件 | 状态 |
|------|------|
| `compile/gpu/` — CudaKernel 模板 + Device 抽象 | ✅ 脚手架 |
| `compile/optimizations/` — FusionPass / ParallelPass | ✅ 脚手架 |

#### tenthpm 包管理器

| 组件 | 状态 |
|------|------|
| `tenth/tools/tenthpm/` CLI (init/build/test/run/add/remove/list/clean/publish/install) | ✅ **完整实现** |
| Tenth.toml manifest 格式 (含 license 字段) | ✅ |
| 共享引擎模块 (engine.rs) — search_paths + in-process 编译/运行 | ✅ |
| 依赖类型：registry / path / git 三种 | ✅ |
| Tenth.lock 锁文件 (含 checksum) | ✅ |
| .tenthpkg 打包归档 (publish) | ✅ |
| 版本号校验 (X.Y.Z semver) | ✅ |

#### LSP 服务器

| 组件 | 状态 |
|------|------|
| `tenth/tools/lsp/` — 诊断/悬停/补全/定义/格式化 handler | ✅ 脚手架 |

#### 标准库补全

| 组件 | 状态 |
|------|------|
| optim/adam_step 实现 | ✅ |
| optim/adagrad_step 实现 | ✅ |
| optim/rmsprop_step 实现 | ✅ |
| nn/batchnorm 函数 | ✅ |
| nn/conv2d 函数 | ✅ |
| nn/embedding 函数 | ✅ |
| init/ 6 个初始化器 (zeros/ones/xavier_uniform/xavier_normal/kaiming_uniform/kaiming_normal) | ✅ |

#### 语言增强

| 组件 | 状态 |
|------|------|
| 结构体字段默认值 — `Expr { kind: "Int", ival: 42, .. }` 语法 | ✅ |
| 泛型返回类型 — `fn f() -> Vec<Token>` 正确解析，修复 `>>` 拆分 | ✅ |
| 枚举元组变体 — `enum TokenKind { IntLiteral(i64), Plus, Eof }` + match 绑定 | ✅ |

#### 测试覆盖提升

| 测试文件 | 变更 | 状态 |
|----------|------|------|
| enum_test.rs | 5→9（+4 枚举元组变体/match 绑定） | ✅ |
| generic_test.rs | 5→11（+6 泛型返回/Vec<Token>/>>拆分） | ✅ |
| struct_test.rs | 5→8（+3 字段默认值/..语法） | ✅ |
| 总计 | 121→499（+378，含 v0.3.1 autodiff + v0.3.3 LSP/tenthpm/类型推断/模式匹配/迭代器/错误恢复/MNIST/parity/shape_check） | ✅ 498 passed + 1 ignored |

---

### v0.3.3 f32 真泛型改造（2026-06-29）

#### 问题根因
`Type::Tensor { dtype: BaseType, .. }` 的 dtype 字段是 `BaseType` 枚举值，无法表达 TypeParam。导致 std/nn/ 下 13 个文件出现 `_f32` 双胞胎副本（且副本的 Tensor 参数仍是 f64——f32 支持是假的）。这是类型系统表达力不足的系统性 DRY 违反。

#### 阶段 1：Rust 侧 Type 枚举改造
- `types.rs`：`Type::Tensor.dtype` 从 `BaseType` 改为 `Box<Type>`，dtype 现在可以是 TypeParam
- 新增 `Type::tensor_dtype() -> Option<BaseType>` helper（运行时取回 BaseType，TypeParam 场景返回 None）
- `from_annotation`：删除非 Base dtype 强制 fallback 到 F32 的逻辑，TypeParam 保留
- `substitute_type`：递归处理 Tensor 的 dtype 字段（+ Array/Generic）
- 适配 ~20 处 match 模式：lower/types.rs(15)、lower_expr.rs、mod.rs、bytecode.rs、value.rs、interpreter/mod.rs
- 采用 `dtype.as_ref().clone()` 保留 TypeParam 信息，比硬 fallback 到 F64 更准确

#### 阶段 2：tenthc 侧（无需改造）
- 验证发现 tenthc parser 不解析 `Tensor[T, ..]` 泛型类型注解（只处理具体 dtype）
- tenthc 自身的 .th 源码不用泛型 Tensor
- parity_test 129/129 + selfhost_frontend 4/4 全通过 → 路径 B/C 未断
- 结论：tenthc 侧 HirType.sub 整数编码足以处理当前所有 tenthc 能编译的代码

#### 阶段 3：std/nn 重写
- 删除 11 个 `_f32` 副本函数，7 个数据计算函数改泛型：
  - `transformer_encoder_block<T>`、`multihead_attention<T>`、`scaled_dot_product_attention<T>`
  - `layer_norm<T>`、`feedforward<T>`、`dropout<T>`、`batchnorm<T>`
- `make_*` 构造函数保留 f64（内部依赖 `randn`/`zeros` native，不支持泛型实例化）
- `positional_encoding` 保留 f64（同理）
- `binary_cross_entropy` 保留 f64（标量函数，Tenth 不支持 `<T: Float>` trait bound 强制检查）
- mask 参数保留 `Tensor[f64, ..]`（元数据语义，非数据）
- 顺带修正：transformer.th 第 57 行注释"lexer 不支持科学计数法"已过时（lexer.rs 第 127-147 行实际支持 `1e-5`），删除该注释
- prelude.th 索引同步：删除已删除的 _f32 引用，标注泛型函数

#### 验证（全绿）
- cargo build：0 error（177 pre-existing warnings，非本轮引入）
- 自举路径 A：`[OK] Full compiler compiled to tenthc_full.wasm`
- parity_test：129/129（路径 B/C 未断）
- selfhost_frontend：4/4（真实 assert 通过）
- stdlib_test：114/114（nn 模块解析全通过）
- generic_tensor_test：2/2（新增，验证泛型 Tensor 实例化）

#### 新增测试
- `tenth/tests/generic_tensor_test.rs`：验证 `fn foo<T>(x: Tensor[T, ..])` 调用 `foo<f64>(a)` 被正确实例化为 `foo_F64(x: Tensor[f64, ..])`，TypeParam T 被正确替换到 Tensor.dtype 字段

#### 已知限制（非本轮引入）
- native 构造函数（`randn`/`zeros`/`ones`）不支持泛型实例化（`GenericCall` 查 `generic_funcs` 表，native 不在表中）
- trait bound（`<T: Float>`）语法支持但语义不强制检查（`generics_bounds: HashMap::new()` 空表）

---

### v0.3.3 技术债清理第一类（2026-06-29）

#### selfhost_frontend.rs 测试 assert 化
- 原状：`tenthc_parses_own_source`/`tenthc_lowers_own_source` 在 Err 分支只 `println!` 不 `assert!`（注释"Phase B is about identifying gaps"），测试永远不会失败，等于没有护栏
- 修复：Err 分支改为 `.expect("...self-hosting frontend contract")`，让 parse/lower 失败真正触发测试红
- 头注释更新：移除过时的"Phase B"措辞，指明执行覆盖由 `fixpoint_runtime.rs`/`parity_test.rs` 提供
- 验证：4/4 通过（parse 182 items、lower 145 functions，真实 assert 通过而非 println 假绿）

#### AUDIT §7.4 #16 测试盲区条目已过时 → 标记修复
- AUDIT 原写"tenthc_test.rs 只测解析未测执行"——名字对不上（实为 selfhost_frontend.rs），且执行覆盖已由 `fixpoint_runtime.rs`（Wasmtime 端到端）+ `parity_test.rs`（112 项 Rust/tenthc 一致性）提供
- §7.4 #16 和 §8.1 "tenthc_test.rs 加执行测试" 均标记为已修复

#### lower.rs（2017 行）上帝文件拆分
- 拆为 `tenth/src/hir/lower/` 目录模块 7 文件：mod.rs(144) / scope.rs(120) / import.rs(56) / lower_expr.rs(691) / types.rs(240) / lower_stmt.rs(475) / closures.rs(187)
- 纯重构不改行为；Lowerer 公开 API（new/with_search_paths/lower_program）签名不变
- 修复 3 个编译错误：Ownership 导入路径、Type 私有枚举改用 types 模块、Scope.parent 字段改 pub(super)

#### wasm.rs（2089 行）上帝文件拆分
- 拆为 `tenth/src/compile/wasm/` 目录模块 6 文件：mod.rs(128) / types.rs(69) / sections.rs(181) / compile.rs(942) / closures.rs(340) / host.rs(355)
- 纯重构不改行为；`register_host_functions`/`run_wasm_module` 通过 `pub use` 重导出，外部引用零改动
- IMPORT_COUNT=18 注释完整保留；自举路径 C（全 WASM 闭环）未受影响

#### 文档漂移批量修复
- `能力全梳理.md` §2.2：JIT 状态从"未实际使用"改为"默认启用，保守策略（autodiff/复杂操作回退 VM）"——与 main.rs:292 默认调用 `jit::run_jit` 的事实对齐
- `CODE_WIKI.md`：删除不存在的 `boot.th` 引用（项目根目录 Glob 确认无此文件），main.th 描述改为"拼接 6 个模块源码后调用 compile_host"
- `README.md`：shape 检查卖点标注"规划中，当前未实现"（与能力全梳理 ❌ 状态对齐，消除内部矛盾）；transformer 改为 transformer_encoder_block（与实际文件内容对齐）；LSP 从 ✅ 降为 ⚠️ 并注明"AST 符号+关键字，非语义补全"
- `AUDIT.md` §7.4 #16 和 §8.1 tenthc_test.rs 条目标记已修复

**验证**：cargo build（debug）通过，177 pre-existing warnings（hostcalls.rs 函数指针转换，非本轮引入）；selfhost_frontend 4/4 真实 assert 通过；自举路径 A `[OK] Full compiler compiled to tenthc_full.wasm`。未运行全量 cargo test（rustc 1.95.0 ICE bug，非本项目问题）。

**未触及**：f32 双胞胎（14 个文件）属"伪装成技术债的设计决策"，清理等于做 A 类类型系统改造，本轮冻结不处理。

---

### v0.3.3 重构（2026-06-29）

#### interpreter.rs 上帝文件拆分

| 组件 | 状态 |
|------|------|
| `runtime/interpreter.rs`（4655 行）→ `runtime/interpreter/` 目录模块（8 个文件） | ✅ 纯重构 |
| `mod.rs`（1202 行）— Interpreter 结构体、eval_expr/eval_call/eval_stmt | ✅ |
| `binary.rs`（426 行）— eval_binary/eval_unary/values_eq/value_to_string | ✅ |
| `pattern.rs`（180 行）— eval_field/pattern_matches/bind_pattern/unbind_pattern | ✅ |
| `methods.rs`（1312 行）— 方法分派（String/Vec/Map/Range/Iterator/Tensor/Scalar） | ✅ |
| `index.rs`（121 行）— eval_index（String/Tensor/Vec 索引与切片） | ✅ |
| `natives.rs`（1253 行）— call_named_fn（原生函数分派） | ✅ |
| `json.rs`（205 行）— JSON 编解码（H-6 安全版本，带深度闸门+转义状态机修复） | ✅ |
| `datetime.rs`（28 行）— days_to_date（Unix 天数→年月日） | ✅ |
| main.rs 删除 days_to_date/json_encode_value/json_decode_string 等函数定义（226 行） | ✅ |
| main.rs 调用点改为 `datetime::days_to_date` / `json::json_encode_value` 等模块前缀 | ✅ |
| H-6 漏洞修复 — interpreter 内部 JSON 解析改用 main.rs 安全版本 | ✅ |
| cargo build（debug + release）无新增 warning | ✅ |
| 自举路径 A 验证 — `[OK] Full compiler compiled to tenthc_full.wasm` | ✅ |

**重构原因**：interpreter.rs 是 4655 行的上帝文件，难以维护和导航。拆分为目录模块后，每个文件聚焦单一职责，便于定位和修改。

**范围**：纯重构，不改任何行为（唯一例外：JSON 解析改用安全版本，顺带修复 H-6 漏洞）。方法签名保持不变，只改可见性（`fn` → `pub(super) fn`）。Interpreter 结构体字段可见性保持不变。

---

### v0.3.0 后期新增（2026-06-14）

#### 自动微分

| 组件 | 状态 |
|------|------|
| 张量级 Wengert tape (21 算子) | ✅ |
| Backward 全链路 (DAG 遍历 + broadcast grad) | ✅ |
| 解释器 recording 模式 | ✅ |
| 7 个内置函数 (new_grad/param/backward/grad/stop_grad/zero_grad/cross_entropy) | ✅ |
| Mean/Sum 录制 (返回张量) | ✅ |
| Scalar-tensor 录制 (标量自动包装) | ✅ |

#### 张量运算

| 组件 | 状态 |
|------|------|
| 张量间四则运算 (广播语义) | ✅ |
| MatMul (1D/2D) | ✅ |
| Transpose | ✅ |
| Conv2D (im2col + backward) | ✅ |
| Dropout (inverted + backward) | ✅ |
| Softmax (逐行) | ✅ |
| BatchNorm (backward 就绪, forward 已包装) | ✅ |

#### 标准库

| 模块 | 文件数 | 状态 |
|------|--------|------|
| nn/ | 13 | 全部可运行 (linear/loss/activations/dropout/conv/batchnorm/embedding/attention/multihead_attention/layer_norm/positional_encoding/feedforward/transformer) |
| optim/ | 4 | 全部可运行 (sgd/adam/adagrad/rmsprop) |
| data/ | 1 | DataLoader (new/has_next/next_batch/reset/num_batches) |
| init/ | 1 | 6 个初始化器实现 (zeros/ones/xavier_uniform/xavier_normal/kaiming_uniform/kaiming_normal) |
| collections/ | 2 | iter (map/filter/reduce/zip/enumerate 等), collections (flat_map/partition 等) |
| string/ | 1 | 字符串工具 (join_lines/join_comma/indent/word_wrap/capitalize 等) |
| utils/ | 2 | 序列化 (save_model/load_model/save_checkpoint), math (min/max/clamp/signum 等) |
| math/ | 1 | 数学函数参考 |

#### 语言打磨

| 组件 | 状态 |
|------|------|
| 块注释 /* */ (支持嵌套) | ✅ |
| Vec: pop/set/clear | ✅ |
| String: trim/split/replace/substring/to_upper/to_lower | ✅ |
| REPL 多行输入 (自动续行) | ✅ |
| 错误源码上下文显示 | ✅ |
| 数组字面量 [1,2,3] | ✅ 已原生存在 |
| for-in 循环 (Range/Vec/Tensor) | ✅ |

---

## 跳过项：Phase 4-6 GPU / 分布式 / 生态

> 以下功能因环境限制暂跳，条件就绪后实施。

### Phase 4: GPU 与性能

- [ ] 安装 CUDA Toolkit 12.6（`nvidia-smi` 显示驱动已支持，RTX 4060 8GB）
- [x] ~~实现 CUDA kernel 模板代码生成~~ → `compile/gpu/` 脚手架已就绪 (CudaKernel 模板 + Device 抽象)
- [ ] MIR→CUDA 算子映射 + `tenth compile --target=cuda`
- [x] ~~算子融合~~ → `compile/optimizations/` 脚手架已就绪 (FusionPass/ParallelPass)
- [ ] 自动并行分解 / SPMD 降级（需多 GPU）

### Phase 5: AI 全栈

- [ ] SPMD 并行原语（数据并行、模型并行、流水线并行，需多 GPU）
- [ ] 分布式通信（MPI/NCCL：all_reduce, all_gather, send/recv）
- [ ] Autodiff 进阶：checkpointing、高阶微分、张量级 tape、解释器集成
- [ ] nn 进阶：LayerNorm, Attention, FlashAttention, GELU
- [ ] optim 进阶：AdamW, Lion, LAMB, lr_schedule, gradient clipping
- [ ] data 标准库：DataLoader, 数据增强, pipeline 宏

### Phase 6: 生态与工具

- [x] ~~包管理器 tenthpm~~ → `tenth/tools/tenthpm/` **完整实现** (CLI: init/build/test/run/add/remove/list/clean/publish/install + Tenth.toml + Tenth.lock + .tenthpkg 打包 + path/git/registry 依赖)
- [x] ~~LSP 服务器~~ → `tenth/tools/lsp/` **完整实现** (文档同步/diagnostics推送/hover/completion/definition/documentSymbol/references/rename/signatureHelp/foldingRange/semanticTokens/formatting)
- [ ] 调试器进阶（断点插桩、调用栈、条件断点）
- [ ] 官网 / 论坛 / RFC 流程 / 贡献指南

---

## 环境配置

> 环境依赖、网络代理、构建/测试命令详见 `DEPS.md`。

---


