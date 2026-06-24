# Tenth 自举总体规划（Self-Hosting Master Plan）

> 版本：v1.2 | 日期：2026-06-25
> 状态：**Phase A-C 已完成，Phase D 进行中（D4/D7 已完成，D1-D3/D5/D6 待实施）**
> 范围：tenthc 自举编译器的完整自举路线图，从当前"前端自举 + 后端委托"的混合架构演进到真正的自举闭环。
>
> **v1.2 变更**：基于 2026-06-25 代码复核，修正过时行号（parse_program 1133→1121、闭包占位 887→960、tensor 占位 957→1030），修正 D7 用例数（112→117 实测），补充 Rust 侧 lower.rs 准确行号引用（Ownership@11、trait_defs@151、generic_funcs@144、substitute_type@1871），标注 D4 已完成。
>
> **v1.1 变更**：基于 2026-06-24 代码调查，更新能力差距矩阵（附行号证据），细化 Phase D 实施步骤与依赖关系，补充实施优先级。

---

## 1. 背景与现状

### 1.1 当前架构（As-Is）

```
tenthc/*.th (Tenth 源码)
       │
       ├─[路径 A]─→ compile_host() ──→ Rust Lexer/Parser/Lowerer/WasmCompiler ──→ WASM
       │             (main.th 使用)     (实际是 Rust 母编译器全栈处理)
       │
       ├─[路径 B]─→ compile_program() ──→ bridge.rs ──→ Rust Lowerer + Rust WasmCompiler ──→ WASM
       │             (selfhost_verify)   (Tenth 前端 + Rust 后端)
       │
       └─[路径 C]─→ tenthc/compile/wasm.th ──→ WASM (仅 i64 子集)
                     (three_stage.rs, 被 ignore)
```

**核心问题**：tenthc 的代码生成完全委托给 Rust，tenthc 自身的 `wasm.th` 能力不足以编译 tenthc 自身，自举闭环未真正验证。

### 1.2 能力差距矩阵（2026-06-24 实测）

> 以下数据基于对 tenthc/*.th 和 tenth/src/**/*.rs 的逐行调查，行号为证据。

| 维度 | Rust 母编译器 | tenthc | 差距 | 证据 |
|------|--------------|--------|------|------|
| Lexer | 63+ TokenKind，含插值字符串/科学计数法 | 63 TokenKind，无插值/科学计数法 | 中 | `tenthc/lexer/lexer.th:87-88` 已识别 trait/impl 关键字 |
| Parser | 24 ExprKind + 8 StmtKind + 8 ItemKind | 22 Expr + 8 Stmt + 3 Item（无 impl/trait/use/mod） | 大 | `tenthc/parser/parser.th:1131-1156` parse_program 仅处理 Struct/Enum/Fn |
| HIR Lowerer | 完整类型推断 + 借用检查 + 泛型 + Trait | 无类型推断，ty 字段全填 0；generics 始终空 | 严重 | `tenthc/hir/lower.th:662` `generics: Vec::new()` |
| Trait 系统 | trait_defs + trait_impls + 预置 Display/Eq/Clone | 完全未实现 | 严重 | Rust: `lower.rs:151-152,198-220,1327,1392-1398`；tenthc: 无 |
| 泛型实例化 | generic_funcs + substitute_type | 完全未实现 | 严重 | Rust: `lower.rs:144,464,1871`；tenthc: 无 |
| 借用检查 | Ownership enum + check_use/borrow_shared/borrow_mut | 完全未实现 | 严重 | Rust: `lower.rs:11-16,21,62,72,86,833-872`；tenthc: 无 |
| WASM 后端 import | 17 import（含 f64_bits/str_slice） | 17 import（已对齐） | 已对齐 | Rust: `wasm.rs:60,243-244`；tenthc: `wasm.th:1239,1321+` |
| 闭包 WASM | 未实现（返回错误） | 占位返回 0 | 双方均不支持 | Rust: `wasm.rs:840`；tenthc: `wasm.th:960-963` |
| Tensor WASM | 未实现（返回错误） | 占位返回 0 | 双方均不支持 | Rust: `wasm.rs:840`；tenthc: `wasm.th:1030-1031` |
| 模块系统 | use/mod + try_import_file | ✅ 已实现（Phase B） | 已对齐 | — |
| 自举验证 | — | three_stage.rs 小程序通过，完整自编译受 wasmi 性能限制 | 部分 | `tenth/tests/three_stage.rs` |

### 1.3 tenthc wasm.th 可工作的最小子集

当前 tenthc 的 wasm.th 仅能正确编译：
- i64 整数运算（`+ - * / == != < > <= >=`）
- i64 函数参数变量
- `if/else` 表达式
- `return` 语句
- 函数调用（用户函数 + 5 个 import）
- `println` / `read_file` / `write_bytes` / `Vec::new` / `Vec.len` / `Vec.push` / `Vec.get`
- `break` / `continue`（但无可用循环体）
- Ref/Deref 指针透传

**无法编译 tenthc 自身源码**，因为 tenthc 源码使用了：字符串字面量/拼接/比较、for 循环、struct 字段访问、Vec 操作（部分可用）。

---

## 2. 目标架构（To-Be）

```
┌─────────────────────────────────────────────────────────────────┐
│  Stage 0: Rust 母编译器（tenth/）                                │
│  职责：孵化 Stage 1，提供 host import，最终可退役               │
│  产出：tenthc_stage1.wasm（用 Rust 编译 tenthc/*.th 得到）       │
└─────────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│  Stage 1: 自举编译器（tenthc/，编译为 tenthc_stage1.wasm）       │
│  前端：Lexer + Parser + HIR Lowerer（含类型推断最小集）         │
│  后端：独立 WASM emitter（不依赖 Rust bridge）                  │
│  能力：能编译 tenthc 自身源码                                   │
└─────────────────────────────────────────────────────────────────┘
                            │ 编译自身源码
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│  Stage 2: 固定点验证                                             │
│  tenthc_stage1 编译 tenthc/*.th → tenthc_stage2.wasm            │
│  tenthc_stage2 编译 tenthc/*.th → tenthc_stage3.wasm            │
│  验证：tenthc_stage2 ≡ tenthc_stage3（字节级或语义级等价）       │
└─────────────────────────────────────────────────────────────────┘
```

### 2.1 设计原则

1. **渐进式自举**：每补齐一项 wasm.th 能力，立即用 tenthc 自身源码验证，确保持续向自举闭环收敛。
2. **最小类型推断**：不追求完整类型系统，只实现 WASM 后端正确选操作码所需的最小类型推断（i64/f64/str/struct 名）。
3. **host import 对齐**：tenthc wasm.th 的 import 列表与 Rust wasm.rs 的 15 个 import 对齐，复用 Rust 的 wasmi host 实现。
4. **测试驱动**：每个阶段都有可执行的验收测试，three_stage.rs 最终取消 ignore。
5. **不破坏现有**：Rust 母编译器的 375+ 测试必须持续通过，tenthc 修改不能影响 Rust 侧。

### 2.2 关键设计决策

**决策 1：后端目标选择 WASM 而非 C**

理由：
- Rust 母编译器已有完整 WASM 后端（wasm.rs）和 wasmi host 实现，可直接复用
- WASM 是可验证的字节码格式，便于固定点比较
- C 代码生成需要额外的 C 编译器依赖，增加自举复杂度
- three_stage.rs 已有 WASM 三阶段验证框架

**决策 2：类型推断最小集而非完整类型系统**

理由：
- 完整类型推断（含泛型实例化、Trait 解析、借用检查）工程量巨大
- WASM 后端只需要知道：操作数是 i64 还是 f64（选操作码）、struct 名（算字段偏移）、str 还是 i64（选 import）
- 这些信息可以通过局部类型推断获得，无需完整 TypeEnv
- 完整类型系统留待 Phase D（能力对等）

**决策 3：保留 bridge.rs 作为过渡路径**

理由：
- 在 wasm.th 能力补齐之前，bridge.rs 提供了"前端自举 + 后端委托"的过渡能力
- bridge.rs 不需要删除，但需要明确标记为"过渡路径"
- 最终自举闭环达成后，bridge.rs 可保留用于调试或退役

**决策 4：模块系统采用 use + 文件解析**

理由：
- tenthc 当前用 read_file + 字符串拼接加载 6 个文件，无法支持相对路径、无法防循环导入
- Rust 母编译器已有 try_import_file / load_and_compile_file 实现
- tenthc 需要实现简化版：`use path::name` → 查找 `<base>/path/name.th` 或 `<base>/path/mod.th` → 递归 lower
- 这是自举编译器组织自身源码的前提

---

## 3. 分阶段路线图

### Phase A：WASM 后端最小可用（解除自举阻断）✅ 已完成

**目标**：补齐 wasm.th 的核心能力，使其能编译一个包含字符串、for 循环、struct 的最小 Tenth 程序。

**验收**：新增测试 `tenth/tests/wasm_backend_minimal.rs`，用 tenthc wasm.th 编译并执行一个非平凡的 Tenth 程序（如字符串拼接 + for 循环累加）。

**关键工作项**：
- A1: f64 运算支持（f64 操作码 + FloatLiteral）
- A2: 字符串驻留（string_data + string_offsets）
- A3: 字符串 import 对齐（str_add/str_eq/str_int/str_len/str_at/str_cmp）
- A4: InterpolatedString 全链路（lexer 插值 + parser + lower + wasm）
- A5: for 循环（lower_stmt + compile_stmt）
- A6: loop 循环（compile_stmt 补 disc 7）
- A7: while 循环修复（parser 存 cond/body + lower + compile）
- A8: Match 表达式（lower arm body + compile）
- A9: export 名编码修复（补全大写字母 B-Z）
- A10: local 槽位动态分配（替代固定 16 个）
- A11: 类型推断最小集（infer_binary_type + resolve_call_type + struct 字段类型）

### Phase B：前端对齐（tenthc 能正确解析自身源码）✅ 已完成

**目标**：tenthc 的 parser + lowerer 能正确处理 tenthc 自身的 6 个 .th 文件，产出正确的 HIR。

**验收**：新增测试 `tenth/tests/selfhost_frontend.rs`，用 tenthc 的 lexer/parser/lowerer 处理 tenthc/*.th，验证 HIR 节点数和结构符合预期。

**关键工作项**：
- B1: 模块系统（use + try_import_file 简化版）
- B2: impl 块解析（parse_impl_block + 收集方法到 methods map）
- B3: enum 定义收集（enums map + variant 类型）
- B4: struct 字段类型推断（struct_layouts）
- B5: 闭包捕获分析（free_vars_in + collect_free_vars）
- B6: Range 表达式 lower（补 "range" 分支）
- B7: AssignOp 修复（load + op + store 而非直接 local.set）
- B8: Var/Let 变量解析修复（let 绑定的局部变量可被 Var/Assign 查找）

### Phase C：自举闭环（tenthc 编译自身，固定点验证）✅ 已完成（固定点受 wasmi 性能限制）

**目标**：tenthc wasm.th 能编译 tenthc 自身源码，产出可执行的 WASM，且 Stage N ≡ Stage N+1。

**验收**：取消 three_stage.rs 的 ignore，测试通过。

**关键工作项**：
- C1: tenthc 源码自举适配（确保 tenthc/*.th 只使用 wasm.th 支持的特性）
- C2: boot.th 同步或废弃（统一到模块化版本）
- C3: three_stage.rs 修复与优化（降低 wasmi 执行时间）
- C4: 固定点验证（Stage 2 ≡ Stage 3）— 小程序通过，完整自编译待 JIT 运行时

### Phase D：能力对等（tenthc 与 Rust 母编译器功能对等）

**目标**：tenthc 能编译 Rust 母编译器测试套件中的程序，产出语义等价的 WASM；补齐 tenthc 在 Trait/泛型/借用检查三大核心特性上的差距。

**验收**：扩展 `tenth/tests/parity_test.rs`，覆盖 Trait 方法分派、泛型实例化、借用检查场景；对同一输入，tenthc 和 Rust 编译器产出的 WASM 执行结果一致。

**当前状态（2026-06-25 实测）**：
- D4 native 函数对齐已完成（wasm.th:1239 `num_imports=17`，与 Rust wasm.rs:60 `IMPORT_COUNT=17` 对齐）
- D7 parity_test.rs 已完成（117 用例，覆盖算术/控制流/struct/递归等基础特性）
- D1/D2/D3/D5/D6 均未开始，tenthc 与 Rust 母编译器在高级特性上存在严重差距

#### D1: Trait 系统

**现状**：tenthc `lexer.th:87-88` 已识别 `trait`(disc=19)/`impl`(disc=18) 关键字，但 `parser.th:1131-1156` 的 parse_program 不处理这两个 token；`hir.th` 无 HirTraitDef/HirTraitImpl 结构（HirProgram 仅含 fns/structs/enums，见 `hir.th:120-138`）；`lower.th` 无 trait_defs/trait_impls。

**Rust 母编译器参考**：`lower.rs:151-152` trait_defs/trait_impls HashMap；`lower.rs:198-220` 预置 Display/Eq/Clone；`lower.rs:1327` trait 定义解析；`lower.rs:1392-1398` trait impl 方法解析。

**实施步骤**：
1. `hir.th` 新增 HirTraitDef（name, method_names, method_sigs）和 HirTraitImpl（trait_name, type_name, methods）
2. `parser.th` 新增 parse_trait（解析 `trait Name { fn method(...) -> ...; ... }`）和 parse_impl_for（解析 `impl Trait for Type { fn method(...) {...} ... }`）
3. `parser.th` parse_program 新增 trait(disc=19)/impl(disc=18) 分支
4. `lower.th` 新增 trait_defs/trait_impls map；lower_program 收集 trait 定义和实现
5. `lower.th` 方法调用分派：按 receiver 类型查 trait_impls 找到具体方法
6. `wasm.th` 无需改动（方法调用已编译为普通 Call）

**依赖**：无前置依赖，可独立实施

#### D2: 泛型实例化

**现状**：tenthc `hir.th:98` HirFnDef 有 `generics: Vec<str>` 字段，但 `lower.th:662` 始终设为 `Vec::new()`；`parser.th` parse_fn 不解析 `<T>` 语法；无 substitute_type 函数。

**Rust 母编译器参考**：`lower.rs:144` generic_funcs HashMap；`lower.rs:464` 泛型函数实例化；`lower.rs:1871` substitute_type 函数。

**实施步骤**：
1. `parser.th` parse_fn 新增泛型参数解析：`fn name<T, U>(...) -> ...` 中 `<T, U>` 部分
2. `parser.th` parse_struct 新增泛型参数解析：`struct Pair<T, U> { ... }`
3. `hir.th` HirFnDef.generics 和 HirStructDef.generics 填充实际值
4. `lower.th` 新增 generic_funcs map：第一遍收集所有泛型函数模板
5. `lower.th` 新增 substitute_type(ty, type_map)：将类型参数替换为具体类型
6. `lower.th` GenericCall 处理：根据调用点实参推断 type_map，实例化函数体
7. `wasm.th` 无需改动（实例化后是普通函数）

**依赖**：建议在 D1 之后实施（Trait bound 常与泛型配合）

#### D3: 借用检查

**现状**：tenthc `lower.th` 完全无 Ownership/check_borrow 相关代码。

**Rust 母编译器参考**：`lower.rs:11-16` Ownership enum（Owned/SharedRef/ExclusiveRef/Moved）；`lower.rs:21` Scope.ownership HashMap；`lower.rs:62` check_use；`lower.rs:72` check_borrow_shared；`lower.rs:86` check_borrow_mut；`lower.rs:833-872` ref/mutref/move 状态跟踪。

**实施步骤**：
1. `lower.th` 新增 Ownership enum（0=Owned, 1=SharedRef, 2=ExclusiveRef, 3=Moved）
2. `lower.th` Scope 结构新增 ownership: Vec<(String, i64)> 字段
3. `lower.th` 新增 check_use：检查变量是否被 move，若 moved 则报错
4. `lower.th` 新增 check_borrow_shared：检查共享借用冲突（已有 ExclusiveRef 则报错）
5. `lower.th` 新增 check_borrow_mut：检查独占借用冲突（已有任何借用则报错）
6. `lower.th` lower_expr 的 Ref/MutRef/Move 分支：更新 ownership 状态
7. `lower.th` lower_stmt 的 Let/Assign 分支：更新 ownership 状态

**依赖**：无前置依赖，可与 D1/D2 并行；但建议在 D1/D2 之后，避免类型信息不足影响借用分析

#### D4: native 函数对齐（最小差距）✅ 已完成

**现状**：tenthc `wasm.th:1239` num_imports=17，已补齐 `f64_bits`(import 15) 和 `str_slice`(import 16)。Rust 侧 `wasm.rs:60` IMPORT_COUNT=17。两侧已对齐，命名差异（tenthc 用 `env`/小写，Rust 用 `host`/大写）由 parity_test.rs 兼容处理。

**实施步骤**：
1. `wasm.th` num_imports 从 15 改为 17
2. `wasm.th` import section 新增 idx 15: `env.f64_bits` (f64)->i64
3. `wasm.th` import section 新增 idx 16: `env.str_slice` (i32,i64,i64)->i32
4. `wasm.th` wasm_f64_const（第 87-97 行）：将 f64_bits 调用改为 import 15
5. `wasm.th` compile_expr 的 Slice 分支（第 967-978 行）：实现 str_slice 调用，移除 TODO 占位
6. `tenth/src/compile/wasm.rs` host 侧：确认 f64_bits/str_slice 已实现（第 243-244 行已存在）

**依赖**：无前置依赖，工作量最小，可立即实施

#### D5: 闭包 WASM 后端实现

**现状**：tenthc `wasm.th:960-963` 闭包占位返回 0；`lower.th:427-434` 有闭包 lowering 但 captures_count 始终 0（第 429 行 `let captures_count: i64 = 0;` 注释 "simplified: no deep analysis"）；`parser.th` 有闭包解析。Rust 侧 `wasm.rs:840` 闭包返回错误 — **双方 WASM 后端均不支持闭包**。

**实施步骤**：
1. `lower.th` 第 427 行：实现 free_vars_in 递归分析，填充 captures_start/captures_count
2. `wasm.th` 闭包表示设计：struct { fn_ptr: i64, env_ptr: i64 }（两 i64 打包为一个 i64 pair 或两个 local）
3. `wasm.th` compile_expr disc 23 (Closure) 分支：
   - 分配 env struct（tenth_alloc），填充捕获变量
   - 将闭包体编译为独立函数（mangled name）
   - 返回 (fn_ptr, env_ptr)
4. `wasm.th` 闭包调用：调用 fn_ptr(env_ptr, args)
5. Rust 侧 `wasm.rs` 同步实现闭包后端（第 840 行 _ 分支替换为 Closure 处理）
6. parity_test.rs 新增闭包测试用例

**依赖**：需要 D4（tenth_alloc import）就位；建议在 D1-D3 之后

**风险**：闭包的 WASM 表示复杂，env 捕获需要堆分配；建议先实现无捕获闭包，再扩展到有捕获

#### D6: Tensor WASM 后端实现

**现状**：tenthc `wasm.th:1030-1031` Tensor 占位返回 0；`lower.th` 有 tensor lowering；`parser.th` 有 tensor 字面量解析。Rust 侧 `wasm.rs:840` tensor 返回错误 — **双方 WASM 后端均不支持 Tensor**。

**实施步骤**：
1. `wasm.th` Tensor 通过 host import 实现：新增 tensor_create/dot/matmul 等 import
2. `wasm.th` compile_expr disc 29 (TensorLiteral) 分支：
   - 将元素展开为 i64 数组
   - 调用 tensor_from_vec import 传入元素指针 + shape
3. Rust 侧 `wasm.rs` 同步实现 Tensor 后端（新增对应 import 和 host 实现）
4. parity_test.rs 新增 tensor 测试用例

**依赖**：需要 D4（import 对齐）就位；建议在 D5 之后

**风险**：Tensor 操作涉及 host 运行时（`runtime/tensor.rs`），WASM 侧需要完整的 host import 套件；工程量大，可考虑延后或标记为可选

#### D7: parity_test.rs（已完成）

**现状**：117 个测试用例通过，覆盖算术/变量/控制流/struct/递归等基础特性。未覆盖 Trait/泛型/借用/闭包/Tensor/Match/Float（详见 `parity_test.rs` 注释）。

**后续扩展**：随 D1-D6 完成逐步新增对应测试用例

#### Phase D 实施优先级与依赖

```
优先级 1（最小补齐，立即可做）：
  D4 (native 对齐) ✅ 已完成 — 17 个 import 两侧对齐

优先级 2（核心高级特性，前端对齐）：
  D1 (Trait 系统) ──┐
  D2 (泛型实例化) ──┤ (D2 建议 D1 之后，Trait bound 常配泛型)
  D3 (借用检查) ────┘ (可与 D1/D2 并行，建议之后)

优先级 3（WASM 后端扩展，双方均需实现）：
  D5 (闭包 WASM) ── (依赖 D4 ✅)
  D6 (Tensor WASM) ─ (依赖 D4 ✅，建议 D5 之后)

优先级 4（测试覆盖扩展）：
  D7 扩展 ── 随 D1-D6 完成逐步新增用例（当前 117 用例）
```

---

## 4. 风险与缓解

| 风险 | 概率 | 影响 | 缓解措施 |
|------|------|------|----------|
| wasm.th 手写字节码易出错 | 高 | 高 | 每项能力配套单元测试；用 wasm-encoder 验证生成的 WASM 合法性 |
| 类型推断最小集不够用 | 中 | 高 | 优先支持 tenthc 自身源码用到的类型组合；遇到无法推断的情况回退到 i64 |
| wasmi 执行 tenthc 编译器太慢 | 高 | 中 | three_stage.rs 用单独 thread + 64MB 栈；考虑用 wasm3 替代 wasmi |
| tenthc 源码使用了未支持的特性 | 中 | 高 | 逐文件审计 tenthc/*.th 的特性使用；必要时重构 tenthc 源码避开未支持特性 |
| boot.th 与模块化版本不一致 | 中 | 中 | Phase C 统一到模块化版本，废弃 boot.th |
| 工程量过大 | 高 | 高 | 严格分阶段，每阶段独立可交付；Phase A 完成即可解锁最小自举 |
| **D1 Trait 方法分派复杂** | 中 | 高 | 参考 Rust 侧 `lower.rs:1392-1398` 实现；先支持静态分派（inherent impl），再扩展 trait impl |
| **D2 泛型实例化类型推断不足** | 中 | 高 | 参考 Rust 侧 `lower.rs:1871` substitute_type；类型推断失败时回退到 i64 |
| **D3 借用检查误报/漏报** | 中 | 中 | 参考 Rust 侧 `lower.rs:62-86`；先实现 move 检查，再扩展 ref/mutref |
| **D5 闭包 env 捕获需要堆分配** | 高 | 高 | 先实现无捕获闭包（env 为空），再扩展到有捕获；用 tenth_alloc import |
| **D6 Tensor host import 套件庞大** | 高 | 高 | 可标记为可选；Tensor 操作主要走 VM/解释器路径，WASM 侧非自举必需 |
| **parity_test.rs Float 测试被禁用** | 中 | 中 | tenthc 将所有函数声明为 `(i64...) -> i64`，f64 参数类型不匹配；需修复 tenthc 函数签名推断 |

---

## 5. 成功标准

### 5.1 Phase A 成功标准 ✅ 已达成
- [x] tenthc wasm.th 能编译并执行包含 f64 运算、字符串拼接、for 循环的程序
- [x] 新增 `wasm_backend_minimal.rs` 测试通过
- [x] Rust 母编译器 375+ 测试无回归

### 5.2 Phase B 成功标准 ✅ 已达成
- [x] tenthc lexer/parser/lowerer 能正确处理 tenthc/*.th 全部 6 个文件
- [x] 新增 `selfhost_frontend.rs` 测试通过
- [x] HIR 节点数和结构符合预期

### 5.3 Phase C 成功标准（自举达成）✅ 已达成（固定点除外）
- [x] three_stage.rs 取消 ignore 并通过（小程序 add(3,4)=12）
- [x] tenthc_stage1.wasm 能编译测试程序产出 tenthc_stage2.wasm
- [x] tenthc_stage2.wasm 能执行 add(3,4)=12
- [ ] 固定点：tenthc_stage2.wasm ≡ tenthc_stage3.wasm（受 wasmi 性能限制，待 JIT 运行时）

### 5.4 Phase D 成功标准（能力对等）

**D4 native 函数对齐** ✅ 已达成：
- [x] tenthc wasm.th import 数量从 15 扩展到 17（补齐 f64_bits/str_slice）
- [x] Slice 表达式通过 str_slice import 正确编译（移除 TODO 占位）
- [x] parity_test.rs 新增 Slice 测试用例通过

**D1 Trait 系统**：
- [ ] tenthc 能解析 `trait Name { fn method(...); }` 和 `impl Trait for Type { fn method(...) {...} }`
- [ ] tenthc 能正确分派 trait 方法调用（静态分派）
- [ ] parity_test.rs 新增 Trait 方法调用测试用例通过

**D2 泛型实例化**：
- [ ] tenthc 能解析 `fn name<T>(...) -> ...` 和 `struct Pair<T, U> { ... }`
- [ ] tenthc 能根据调用点实参实例化泛型函数
- [ ] parity_test.rs 新增泛型函数调用测试用例通过

**D3 借用检查**：
- [ ] tenthc 能检测 use-after-move 错误
- [ ] tenthc 能检测双重独占借用错误
- [ ] parity_test.rs 新增借用检查测试用例通过

**D5 闭包 WASM 后端**：
- [ ] tenthc wasm.th 能编译无捕获闭包并正确调用
- [ ] tenthc wasm.th 能编译有捕获闭包（env 通过 tenth_alloc 分配）
- [ ] parity_test.rs 新增闭包测试用例通过

**D6 Tensor WASM 后端**（可选）：
- [ ] tenthc wasm.th 能编译 TensorLiteral 并通过 host import 创建 Tensor
- [ ] parity_test.rs 新增 Tensor 测试用例通过

**全局验收**：
- [ ] tenthc 能编译 Rust 母编译器测试套件中 90%+ 的程序
- [ ] parity_test.rs 用例数从 117 扩展到 150+（覆盖 Trait/泛型/借用/闭包）
- [ ] Rust 母编译器 499+ 测试无回归
- [ ] AUDIT.md §4 的"自举验证通过"表述与实际一致

---

## 6. 不在范围内（Out of Scope）

以下内容不在本规划范围内，留待后续版本：

- JIT 编译器的自举（Cranelift JIT 是 Rust 侧优化，无需自举）
- GPU 后端的自举（Phase 4 GPU 支持是独立方向）
- 并发原语的自举（Spawn/Task/Shard/Node）
- 完整标准库的自举（仅补齐自举所需最小集）
- IDE 工具的自举（LSP/格式化器等）
