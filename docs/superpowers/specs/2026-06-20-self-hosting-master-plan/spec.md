# Tenth 自举总体规划（Self-Hosting Master Plan）

> 版本：v1.0 | 日期：2026-06-20
> 状态：**DRAFT — 待用户批准**
> 范围：tenthc 自举编译器的完整自举路线图，从当前"前端自举 + 后端委托"的混合架构演进到真正的自举闭环。

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

### 1.2 能力差距矩阵

| 维度 | Rust 母编译器 | tenthc | 差距 |
|------|--------------|--------|------|
| Lexer | 63+ TokenKind，含插值字符串/科学计数法 | 63 TokenKind，无插值/科学计数法 | 中 |
| Parser | 24 ExprKind + 8 StmtKind + 8 ItemKind | 22 Expr + 8 Stmt + 3 Item（无 impl/trait/use/mod） | 大 |
| HIR Lowerer | 完整类型推断 + 借用检查 + 泛型 + Trait | 无类型推断，ty 字段全填 0 | 严重 |
| WASM 后端 | 15 import，f64/str/struct/match 支持 | 7 import，仅 i64，无 str/f64/match/for/loop | 严重 |
| 闭包 | Interpreter 支持，WASM 不支持 | 不支持 | 大 |
| 模块系统 | use/mod + try_import_file | 手动字符串拼接 | 严重 |
| 自举验证 | — | three_stage.rs 被 ignore | 严重 |

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

### Phase A：WASM 后端最小可用（解除自举阻断）

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

### Phase B：前端对齐（tenthc 能正确解析自身源码）

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

### Phase C：自举闭环（tenthc 编译自身，固定点验证）

**目标**：tenthc wasm.th 能编译 tenthc 自身源码，产出可执行的 WASM，且 Stage N ≡ Stage N+1。

**验收**：取消 three_stage.rs 的 ignore，测试通过。

**关键工作项**：
- C1: tenthc 源码自举适配（确保 tenthc/*.th 只使用 wasm.th 支持的特性）
- C2: boot.th 同步或废弃（统一到模块化版本）
- C3: three_stage.rs 修复与优化（降低 wasmi 执行时间）
- C4: 固定点验证（Stage 2 ≡ Stage 3）

### Phase D：能力对等（tenthc 与 Rust 母编译器功能对等）

**目标**：tenthc 能编译 Rust 母编译器测试套件中的所有程序，产出语义等价的 WASM。

**验收**：新增测试 `tenth/tests/parity_test.rs`，对同一输入，tenthc 和 Rust 编译器产出的 WASM 执行结果一致。

**关键工作项**：
- D1: Trait 系统（HirTraitDef/HirTraitImpl/默认方法体）
- D2: 泛型实例化（substitute_type + generic_funcs）
- D3: 借用检查（Ownership + check_*）
- D4: 完整 native 函数对齐（85+ 个）
- D5: 闭包 WASM 后端实现
- D6: Tensor WASM 后端实现
- D7: 自动微分 WASM 后端实现（可选）

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

---

## 5. 成功标准

### 5.1 Phase A 成功标准
- [ ] tenthc wasm.th 能编译并执行包含 f64 运算、字符串拼接、for 循环的程序
- [ ] 新增 `wasm_backend_minimal.rs` 测试通过
- [ ] Rust 母编译器 375+ 测试无回归

### 5.2 Phase B 成功标准
- [ ] tenthc lexer/parser/lowerer 能正确处理 tenthc/*.th 全部 6 个文件
- [ ] 新增 `selfhost_frontend.rs` 测试通过
- [ ] HIR 节点数和结构符合预期

### 5.3 Phase C 成功标准（自举达成）
- [ ] three_stage.rs 取消 ignore 并通过
- [ ] tenthc_stage1.wasm 能编译 tenthc/*.th 产出 tenthc_stage2.wasm
- [ ] tenthc_stage2.wasm 能编译 tenthc/*.th 产出 tenthc_stage3.wasm
- [ ] tenthc_stage2 ≡ tenthc_stage3（固定点达成）

### 5.4 Phase D 成功标准（能力对等）
- [ ] tenthc 能编译 Rust 母编译器测试套件中 90%+ 的程序
- [ ] parity_test.rs 通过
- [ ] AUDIT.md §4 的"自举验证通过"表述与实际一致

---

## 6. 不在范围内（Out of Scope）

以下内容不在本规划范围内，留待后续版本：

- JIT 编译器的自举（Cranelift JIT 是 Rust 侧优化，无需自举）
- GPU 后端的自举（Phase 4 GPU 支持是独立方向）
- 并发原语的自举（Spawn/Task/Shard/Node）
- 完整标准库的自举（仅补齐自举所需最小集）
- IDE 工具的自举（LSP/格式化器等）
