# f32/f64 双精度对等路线图

> 目标：消除 f32 与 f64 支持的所有不对称点，使 f32 成为 f64 的真正等价选项（精度更低的等价选项）
>
> 现状：f64 成熟度 5/5，f32 成熟度 3/5（运行时层对齐，编译后端与 autodiff 内部策略性降级）
>
> 创建：2026-07-06
>
> 执行策略：全部 7 阶段连续执行，每阶段完成后验证再推进下一阶段

---

## 当前不对称点全景

### A. 策略性降级（编译后端 + autodiff 内部）

| 编号 | 位置 | 降级内容 |
|------|------|---------|
| A-1 | `compile/wasm/mod.rs:26,52` | WASM 后端 F32 → ValType::F64 塌缩（策略 A） |
| A-2 | `compile/wasm/compile.rs:895-899` | F32 字面量统一发 F64Const |
| A-3 | `tenthc/compile/wasm.th:94-105` | tenthc WASM 后端不实现 wasm_f32_* 指令族 |
| A-4 | `compile/jit/translator.rs:229-235` | JIT PushFloat32 降级为 f64 hostcall |
| A-5 | `compile/jit/translator.rs:474` | JIT dtype 降级为 F64 |
| A-6 | `runtime/autodiff.rs:434` | node_grads 全 ArrayD<f64>（策略 B：前向 f32 + 反向 f64） |

### B. 标准库 DRY 违反（13 个 _f32 副本）

| 位置 | 副本数 |
|------|--------|
| `std/init/initializers.th` | 6 个（xavier/he/zeros/constant _f32） |
| `std/optim/sgd.th` | 3 个 |
| `std/optim/{adam,adamw,adagrad,rmsprop}.th` | 4 个 |
| `std/optim/{clip,accumulate}.th` | 3 个 |

### C. 标准库硬编码 f64（5 个模块）

| 位置 | 硬编码内容 |
|------|-----------|
| `std/nn/activations.th` | relu/sigmoid/tanh/softmax/exp/log/gelu/leaky_relu 全 f64 |
| `std/nn/linear.th` | linear 函数全 f64 |
| `std/nn/conv.th` | conv2d 全 f64 |
| `std/nn/embedding.th` | embedding 全 f64 |
| `std/nn/loss.th` | mse/bce/l1/huber 全 f64 |

### D. 实现细节降级

| 位置 | 降级内容 |
|------|---------|
| `runtime/tensor.rs:91-128` | 降级 helper（mapv/view/iter/as_slice）F32 自动 cast f64 |
| `runtime/tensor.rs:916` | clamp F32 路径走 f64 |
| `runtime/tensor.rs:1105` | cat 即便 F32+F32 也走 as_f64_view() |
| `std/math/constants.th` | 11 常量 + 19 函数全 f64 |

### E. 文档滞后

| 位置 | 问题 |
|------|------|
| `docs/语言参考手册.md:607` | §11 开篇"所有张量操作均为 f64"——与代码相反 |
| `docs/语言参考手册.md:162-163,609-621` | 张量类型示例与创建表无 f32 |
| 全文 | 函数签名示例几乎全 f64 |

### F. 测试缺口

| 缺口 | 现状 |
|------|------|
| f32 vs f64 parity 系统性测试 | 仅 2 项 to_f32 parity |
| f32 autodiff 梯度一致性 | 只验证误差 < 1e-10，未对比梯度方向/幅度 |
| JIT f32 路径 | 无测试（已降级为 f64） |

---

## 阶段 1：文档同步（低风险）

**目标**：消除文档与代码的严重不一致，新增 f32 支持范围说明

**涉及文件**：
- `docs/语言参考手册.md`
- `能力梳理/能力全梳理.md`（如需）
- `MEMO.md`

**任务清单**：
1. 修订 `语言参考手册.md:607` §11 开篇陈述——删除"所有张量操作均为 f64"
2. §3.2 张量类型增加 `Tensor[f32, 3, 224, 224]` 示例
3. §11.1 张量创建表增加 `zeros_f32`/`ones_f32`/`rand_f32`/`randn_f32` 行
4. 新增 §11.X "f32 支持范围与限制"章节：
   - 已支持：前端字面量后缀、类型推断、VM/解释器、TensorData 双变体、标准库 natives
   - 策略性降级：WASM（策略 A）、JIT、autodiff backward（策略 B）
   - 已知不对称：标准库 nn 部分模块硬编码 f64、math 全 f64
5. 全文函数签名示例选择性增加 f32 版本（至少 §3.2、§11.1）
6. MEMO.md 顶部新增 docs 变更记录

**验证**：
- 交叉引用指向正确章节
- 日期为 2026-07-06
- 无新文件创建（仅编辑现有）

**风险**：纯文档改动，零代码风险

---

## 阶段 2：测试补强（中风险）

**目标**：建立 f32 vs f64 系统性 parity 测试基线，为后续阶段提供回归守护

**涉及文件**：
- 新建 `tenth/tests/f32_parity_test.rs`
- 扩展 `tenth/tests/f32_autodiff_test.rs`

**任务清单**：
1. 新建 `f32_parity_test.rs`，对比 f32 与 f64 路径在全后端下的结果一致性：
   - 算术：add/sub/mul/div（f32 vs f64 误差 < 1e-6）
   - 张量构造：zeros_f32 vs zeros、ones_f32 vs ones
   - 激活函数：relu/sigmoid/tanh（f32 路径 vs f64 路径）
   - 矩阵乘：matmul_f32 vs matmul
   - reduction：sum/mean
2. 扩展 `f32_autodiff_test.rs`：
   - f32 vs f64 梯度方向一致性（cosine similarity > 0.9999）
   - f32 vs f64 梯度幅度比（0.99 < ratio < 1.01）
   - 复合网络（linear → relu → mse_loss）端到端 f32 vs f64 梯度对比
3. 新建 `f32_stdlib_parity_test.rs`：对比泛型标准库函数在 `<f32>` 与 `<f64>` 实例化下的输出一致性（覆盖 dropout/batchnorm/layer_norm/feedforward/attention/mha/transformer）

**验证**：
- `cargo test --test f32_parity_test --release` 全绿
- `cargo test --test f32_autodiff_test --release` 全绿
- `cargo test --test f32_stdlib_parity_test --release` 全绿
- 全量回归 0 失败

**风险**：可能发现 f32 路径的潜在 bug（这是价值，不是风险）

---

## 阶段 3：标准库 DRY 清理（中风险）

**目标**：删除 13 个 `*_f32` 副本，5 个硬编码 f64 模块改泛型

**涉及文件**：
- `std/init/initializers.th`
- `std/optim/{sgd,adam,adamw,adagrad,rmsprop,clip,accumulate}.th`
- `std/nn/{activations,linear,conv,embedding,loss}.th`
- `std/math/constants.th`（评估是否泛型化）
- `std/prelude.th`（索引同步）

**任务清单**：
1. `nn/activations.th` 8 个激活函数改泛型 `<T>`（参考 `nn/dropout.th` 模式）
2. `nn/linear.th` linear 函数改泛型
3. `nn/conv.th` conv2d 改泛型
4. `nn/embedding.th` embedding 改泛型
5. `nn/loss.th` mse/l1/huber 改泛型；`binary_cross_entropy` 保留 f64（标量函数，无 trait bound）
6. 删除 `init/initializers.th` 6 个 `*_f32` 副本，原函数改泛型（需解决 `randn`/`zeros` native 泛型实例化——若不支持则保留 _f32 副本，标注原因）
7. 删除 `optim/*.th` 9 个 `*_f32` 副本，原函数改泛型
8. `math/constants.th` 评估：19 个函数改泛型 `<T>`（PI/E 等常量需按 T 返回）
9. `prelude.th` 索引同步：移除 _f32 副本引用
10. 验证 `binary_cross_entropy` 保留 f64 的决策是否可接受（若用户需要 f32 版本，需先实现 `<T: Float>` trait bound 或显式 cast）

**验证**：
- `cargo test --manifest-path tenth/Cargo.toml -- stdlib` 全绿
- `cargo test --test f32_stdlib_test --release` 全绿（验证 _f32 副本删除后 f32 路径仍可用）
- 全量回归 0 失败
- 自举验证通过

**风险**：
- `randn`/`zeros` native 不支持泛型实例化可能导致 init/ 无法完全泛型化——若如此则保留 _f32 副本并标注
- 标准库泛型化可能暴露类型推断的边界 case

**回退策略**：git revert 单个 commit

---

## 阶段 4：autodiff backward f32 化（高风险）

**目标**：消除策略 B，实现真正的 f32 反向传播（前向 f32 + 反向 f32）

**涉及文件**：
- `runtime/autodiff.rs`（核心重写）
- `runtime/tensor.rs`（acc_grad 适配）

**任务清单**：
1. `TapeNode` 增加 `dtype: BaseType` 字段记录前向 dtype（默认 F64）
2. `node_grads` 从 `Vec<Option<ArrayD<f64>>>` 改为 `Vec<Option<TensorData>>`（按 dtype 存储）
3. 重写 `backward` 函数所有 21+ 算子的反向公式，增加 f32 分支：
   - Add/Sub/Mul/Div/Neg/ReLU/Sigmoid/Tanh
   - MatMul/Sum/Mean
   - LayerNorm/BatchNorm/Gelu
   - Select/Scatter/Gather/MaskedFill/Reshape
   - Conv2D/BatchedMatMul
4. 重写辅助函数 `propagate_grad`/`unbroadcast`/`matmul_2d`/`acc_node_grad` 为双 dtype 版本
5. `acc_grad` 适配：从 `&ArrayD<f64>` 改为 `&TensorData`，按 tensor.dtype 写回
6. 保留策略 B 作为 fallback：若反向过程中 dtype 不一致（混合精度），fallback 到 f64 计算

**验证**：
- `cargo test --test autodiff_test --release` 全绿（f64 路径不回归）
- `cargo test --test f32_autodiff_test --release` 全绿
- 新增 `f32_autodiff_precision_test.rs`：对比 f32 backward 与 f64 backward 的精度差异（应 < 1e-6，而非策略 B 的 < 1e-10）
- 全量回归 0 失败
- 自举验证通过

**风险**：
- f32 反向传播可能引入数值稳定性问题（梯度消失/爆炸）
- 重写面大（21+ 算子），bug 风险高
- TapeNode 结构变更波及全部分类/调试器

**回退策略**：保留策略 B 代码路径作为 `#[cfg(feature = "f32_backward_f64")]` feature flag，出问题可快速回退

---

## 阶段 5：WASM 后端 f32 化（高风险）

**目标**：消除策略 A，实现真正的 f32 WASM 代码生成

**涉及文件**：
- `compile/wasm/mod.rs`
- `compile/wasm/compile.rs`
- `compile/wasm/host.rs`（如需）
- `tenthc/compile/wasm.th`（双侧同步）
- `compile/wasm/sections.rs`（如需）

**任务清单**：
1. `wasm/mod.rs:26` 增加 `BaseType::F32 => ValType::F32` 映射
2. `wasm/mod.rs:52` `size_and_type`：F32 返回 (4, ValType::F32)
3. `wasm/compile.rs:895-899` FloatLiteral 按 dtype 分支：F32 → F32Const，F64 → F64Const
4. `wasm/compile.rs` 算术指令：F32 路径发 F32Add/F32Sub/F32Mul/F32Div
5. `wasm/compile.rs` 比较指令：F32 路径发 F32Eq/F32Lt/...
6. `wasm/compile.rs` TensorLiteral：F32 元素按 4 字节存储
7. `tenthc/compile/wasm.th` 实现 `wasm_f32_const`/`wasm_f32_add`/... 11 个 f32 指令
8. `tenthc/compile/wasm.th` `is_expr_float` 区分 f32 与 f64 路径
9. 修改 host imports：如需 f32 版本的 host function（如 `host_make_float32`），双侧同步注册
10. 更新 `compile/wasm/types.rs` 类型段生成

**验证**：
- `cargo test --test f32_wasm_test --release` 全绿（更新期望：f32 现在真正走 f32 路径）
- `cargo test --test parity_test --release` 全绿
- `cargo test --test selfhost_frontend --release` 全绿
- 自举验证通过（路径 C 全 WASM 闭环）
- WASM 模块大小验证：f32 tensor 占 4 字节而非 8 字节

**风险**：
- 双侧同步（Rust + tenthc）工作量大
- host imports 签名变更可能破坏现有 WASM 模块
- wasmi 验证器对 f32/f64 混合类型检查严格

**回退策略**：保留策略 A 代码路径，通过编译期 flag 切换

---

## 阶段 6：JIT 后端 f32 化（高风险）

**目标**：消除 JIT 路径的 f32 降级，实现真正的 f32 hostcall

**涉及文件**：
- `compile/jit/translator.rs`
- `compile/jit/hostcalls.rs`
- `compile/jit/context.rs`（如需）
- `compile/jit/mod.rs`（如需）

**任务清单**：
1. `hostcalls.rs` 新增 `host_make_float32(_vm, f: f32, out)` hostcall
2. `translator.rs:229-235` PushFloat32 实现真正的 f32 hostcall（不再 `as f64`）
3. `translator.rs:474` dtype 字段保留 F32（不再降级为 F64）
4. `translator.rs` 算术指令：F32 路径调用 f32 版本的 hostcall
5. `call_hostcall_f64` 重构为 `call_hostcall_typed(dtype, ...)` 或新增 `call_hostcall_f32`
6. Tensor 元素提取：`hostcalls.rs:421-423` 增加 f32 分支
7. 验证 JIT fallback 机制在 f32 路径下仍正常工作

**验证**：
- `cargo test --test jit_test --release` 全绿
- 新增 `f32_jit_test.rs`：f32 字面量在 JIT 路径下精度保留
- `cargo test --test parity_test --release` 全绿
- 全量回归 0 失败

**风险**：
- JIT hostcall 签名变更可能引入内存安全问题
- f32/f64 混合运算的栈布局可能出错

**回退策略**：保留降级路径，通过编译期 flag 切换

---

## 阶段 7：tenthc 泛型 Tensor 支持（高风险）

**目标**：tenthc parser 支持 `Tensor[T, ..]` 泛型类型注解，使 tenthc 能编译 Tenth 泛型代码

**涉及文件**：
- `tenthc/parser/parser.th`
- `tenthc/hir/lower.th`
- `tenthc/hir/hir.th`（如需）

**任务清单**：
1. `parser.th` 修改类型注解解析：支持 `Tensor[T, ..]` 形式（T 为类型参数）
2. `hir.th` 类型表示：TypeParam 在 Tensor.dtype 字段中的编码
3. `lower.th` 处理 TypeParam 在 Tensor.dtype 字段中（参考 Rust 侧 `hir/types.rs` 的 `Box<Type>` 改造）
4. `lower.th` 适配 ~15 处 match 模式（参考 Rust 侧 v0.3.3 改造记录）
5. 验证 tenthc 能编译 `tenth/std/nn/transformer.th` 等泛型代码

**验证**：
- `cargo test --test selfhost_frontend --release` 全绿
- `cargo test --test parity_test --release` 全绿
- 自举验证通过（路径 B：Tenth 前端 + Rust 后端）
- 新增测试：tenthc 编译泛型 Tensor 代码不报错

**风险**：
- tenthc parser 改动可能破坏自举
- TypeParam 编码可能与现有 sub 整数编码冲突

**回退策略**：git revert

---

## 依赖关系

```
阶段 1（文档）─────────────────────────────────────────────┐
阶段 2（测试）─────────────────────────────────────────────┤
                                                          │
阶段 3（标准库 DRY）──── depends on ──→ 阶段 2（测试守护）─┤
                                                          │
阶段 4（autodiff f32）── depends on ──→ 阶段 2（测试守护）─┤
                                                          │
阶段 5（WASM f32）────── depends on ──→ 阶段 4（backward）─┤
                                                          │
阶段 6（JIT f32）─────── depends on ──→ 阶段 4（backward）─┤
                                                          │
阶段 7（tenthc 泛型）─── independent ─────────────────────┘
```

- 阶段 1、2 可并行
- 阶段 3、4 依赖阶段 2（需要测试守护）
- 阶段 5、6 依赖阶段 4（backward f32 化后才有意义）
- 阶段 7 独立，可与 5、6 并行

## 验证总结

每阶段完成后统一执行：
- `cargo build --release` 无 error
- `cargo test --release -- --skip fixpoint_runtime --skip three_stage --skip native_generic_ctor` 全绿
- `cargo run --release -- run tenthc/main.th` 自举成功
- MEMO.md + 能力全梳理.md 同步更新

## 风险控制

| 阶段 | 风险等级 | 回退策略 |
|------|---------|---------|
| 1 | 低 | git revert |
| 2 | 低 | git revert |
| 3 | 中 | git revert 单 commit |
| 4 | 高 | feature flag 保留策略 B |
| 5 | 高 | 编译期 flag 保留策略 A |
| 6 | 高 | 编译期 flag 保留降级 |
| 7 | 中 | git revert |
