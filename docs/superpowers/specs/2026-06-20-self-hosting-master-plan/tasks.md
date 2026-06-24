# Tenth 自举任务清单（Tasks）

> 配套文档：[spec.md](./spec.md) | [checklist.md](./checklist.md)
> 执行顺序：Phase A → Phase B → Phase C → Phase D（严格顺序，每阶段验收通过后进入下一阶段）

---

## Phase A：WASM 后端最小可用

> **目标**：补齐 tenthc wasm.th 的核心能力，使其能编译包含字符串、for 循环、struct 的最小 Tenth 程序。
> **验收测试**：`tenth/tests/wasm_backend_minimal.rs`

### Task A1: f64 运算支持

**文件**：`tenthc/compile/wasm.th`

**步骤**：
1. 新增 f64 WASM 操作码发射函数：
   - `wasm_f64_const(val: f64)` — 操作码 0x44，f64 以 IEEE 754 小端 8 字节编码
   - `wasm_f64_add()` — 0xA0
   - `wasm_f64_sub()` — 0xA1
   - `wasm_f64_mul()` — 0xA2
   - `wasm_f64_div()` — 0xA3
   - `wasm_f64_eq()` — 0x63
   - `wasm_f64_ne()` — 0x64
   - `wasm_f64_lt()` — 0x65
   - `wasm_f64_gt()` — 0x66
   - `wasm_f64_le()` — 0x67
   - `wasm_f64_ge()` — 0x68
2. `compile_expr` 的 disc 1 (FloatLiteral) 分支：调用 `wasm_f64_const(e.lit_fval)`
3. `compile_expr` 的 disc 5 (Binary) 分支：根据操作数类型选择 i64 或 f64 操作码
   - 需要类型推断支持（依赖 A11），暂时用启发式：若左操作数是 FloatLiteral 则用 f64
4. `to_val_type` 辅助函数：根据 HirExpr.ty 返回 0x7F (i64) 或 0x7C (f64)

**验收**：能编译并执行 `fn f(x:f64,y:f64)->f64{x+y}`，返回 3.5

---

### Task A2: 字符串驻留

**文件**：`tenthc/compile/wasm.th`

**步骤**：
1. 在 `WasmCtx` 新增字段：
   - `string_data: Vec<u8>` — 字符串数据段
   - `string_offsets: Vec<(String, i32)>` — (内容, 偏移) 驻留表
2. 新增函数 `intern_string(ctx: &mut WasmCtx, s: str) -> i32`：
   - 查 `string_offsets`，命中则返回已有偏移
   - 未命中：记录当前 `string_data.len()` 为偏移，追加 s 的字节 + 终止符 0，返回偏移
3. `compile_to_wasm` 末尾：构建 data section，将 `string_data` 写入 memory 偏移 4（跳过 bump pointer global）
4. `compile_expr` 的 disc 3 (StringLiteral) 分支：`wasm_i32_const(intern_string(ctx, e.lit_sval))`

**验收**：能编译并执行 `fn main()->i64{ let s = "hello"; println(s); 0 }`，println 收到 "hello" 字符串指针

---

### Task A3: 字符串 import 对齐

**文件**：`tenthc/compile/wasm.th`、`tenth/src/compile/wasm.rs`（host 侧无需改，复用现有）

**步骤**：
1. 将 tenthc wasm.th 的 import section 从 7 个扩展到 15 个，与 Rust wasm.rs 对齐：
   ```
   idx 0: host.println     (i32) -> ()
   idx 1: host.write_file  (i32, i32) -> ()
   idx 2: host.read_file   (i32) -> i32
   idx 3: host.str_add     (i32, i32) -> i32
   idx 4: host.str_eq      (i32, i32) -> i32
   idx 5: host.str_int     (i64) -> i32
   idx 6: host.tenth_alloc (i32) -> i32
   idx 7: host.Vec_new     () -> i64
   idx 8: host.Vec_push    (i64, i64) -> i64
   idx 9: host.Vec_len     (i64) -> i64
   idx 10: host.Vec_get    (i64, i64) -> i64
   idx 11: host.compile_host (i32, i32) -> i32
   idx 12: host.str_len    (i32) -> i32
   idx 13: host.str_at     (i32, i64) -> i32
   idx 14: host.str_cmp    (i32, i32, i32) -> i32
   ```
2. 更新 `compile_expr` 中 Call 分支的 import 索引映射：
   - println → import 0
   - read_file → import 2
   - write_bytes/write_file → import 1
   - Vec::new → import 7
   - HashMap::new → import 7（复用）
   - compile_host → import 11
   - compile_program → import 11（复用）
3. 更新 MethodCall 分支：
   - len (Vec) → import 9
   - len (str) → import 12
   - push → import 8
   - get (Vec) → import 10
   - get (str/char) → import 13

**验收**：生成的 WASM 能被 wasmi 用 15 个 host import 实例化

---

### Task A4: InterpolatedString 全链路

**文件**：`tenthc/lexer/lexer.th`、`tenthc/parser/parser.th`、`tenthc/hir/lower.th`、`tenthc/compile/wasm.th`

**步骤**：
1. **Lexer**：`lexer_next` 中字符串字面量分支，遇到 `{ident}` 时：
   - 收集前缀为 StringLiteral token
   - 收集 `{ident}` 为 Identifier token（标记为插值变量）
   - 收集后续部分为 StringLiteral token
   - 或：简化方案 — 保留为单个 StringLiteral，在 parser 层处理插值
   - **采用简化方案**：lexer 不改，parser 在 parse_primary 的 "str" 分支检测 `{...}` 并拆分为 InterpolatedString AST 节点
2. **Parser**：`parse_primary` 的 "str" 分支，扫描 `sval` 中的 `{...}`：
   - 无 `{`：普通 StringLiteral
   - 有 `{`：拆分为 parts 数组，存入 Expr kind="interp"
3. **HIR**：`HirExpr` 新增 disc 31 (InterpolatedString)，字段 `parts_start` + `parts_count` 指向 `string_parts` 数组
4. **Lowerer**：`lower_expr` 新增 "interp" 分支，将每个 part 降级：
   - Literal part → StringLiteral
   - Expr part → 调用 `to_string(expr)` 或 `str_int(expr)`（根据类型）
   - 用 `str_add` 链式拼接
5. **WASM**：`compile_expr` 新增 disc 31 分支：
   - 第一个 part 作为初始字符串指针
   - 后续每个 part：调用 str_add(prev, part_ptr)

**验收**：能编译并执行 `fn main()->i64{ let x=42; println("x={x}"); 0 }`，println 收到 "x=42"

---

### Task A5: for 循环

**文件**：`tenthc/hir/lower.th`、`tenthc/compile/wasm.th`

**步骤**：
1. **Lowerer**：`lower_stmt` 新增 "for" 分支：
   - 降级 iterable（Range 或 Vec）
   - 降级 body stmts
   - 存入 HirStmt disc 4 (For)，字段：`name`, `iter_expr`, `body_start`, `body_count`
2. **WASM**：`compile_stmt` 新增 disc 4 (For) 分支：
   - 若 iterable 是 Range：用 block/loop + 计数器实现
     ```
     local.set iter_start
     local.set iter_end
     block
       loop
         local.get iter_start
         local.get iter_end
         i64.ge_s
         br_if 1  ; 退出 block
         ; body
         ; iter_start += 1
         br 0  ; 循环
       end
     end
     ```
   - 若 iterable 是 Vec：用 Vec_len + Vec_get + 索引实现

**验收**：能编译并执行 `fn sum(n:i64)->i64{ let s=0; for i in 0..n { s=s+i; } s }`，sum(5)=10

---

### Task A6: loop 循环

**文件**：`tenthc/compile/wasm.th`

**步骤**：
1. **WASM**：`compile_stmt` 新增 disc 7 (Loop) 分支：
   ```
   block
     loop
       ; body
       br 0  ; 循环
     end
   end
   ```
   - body 中的 break → br 1（退出 block）
   - body 中的 continue → br 0（跳到 loop 头）

**验收**：能编译并执行 `fn count()->i64{ let i=0; loop { if i>=10 { break; } i=i+1; } i }`，返回 10

---

### Task A7: while 循环修复

**文件**：`tenthc/parser/parser.th`、`tenthc/hir/lower.th`、`tenthc/compile/wasm.th`

**步骤**：
1. **Parser**：`parse_stmt` 的 "while" 分支修复：
   - 解析 cond expr 并存入 `stmt.expr_idx`
   - 解析 body block 并存入 `stmt.body_start` + `stmt.body_count`
2. **Lowerer**：`lower_stmt` 新增 "while" 分支：
   - 降级 cond → HirStmt disc 3 (While)，字段 `cond`, `body_start`, `body_count`
3. **WASM**：`compile_stmt` 的 disc 3 (While) 分支已有代码，验证正确性：
   ```
   block
     loop
       ; cond
       i64.eqz
       br_if 1  ; 退出 block
       ; body
       br 0  ; 循环
     end
   end
   ```

**验收**：能编译并执行 `fn count(n:i64)->i64{ let i=0; while i<n { i=i+1; } i }`，count(5)=5

---

### Task A8: Match 表达式

**文件**：`tenthc/hir/lower.th`、`tenthc/compile/wasm.th`

**步骤**：
1. **Lowerer**：`lower_expr` 的 "match" 分支修复：
   - 降级 scrutinee
   - 降级每个 arm 的 pattern 和 body
   - 存入 HirExpr disc 17 (Match)，字段 `arms_start` + `arms_count` 指向 `match_arms` 数组
   - 每个 HirMatchArm：`pattern_kind` (0=wildcard, 1=enum_variant), `variant_name`, `body_expr`
2. **WASM**：`compile_expr` 新增 disc 17 (Match) 分支：
   - 编译 scrutinee，结果留在栈上
   - 对每个 arm：
     - wildcard：drop scrutinee，编译 body
     - enum_variant：用 `i64.eq` 比较 scrutinee 与 variant disc，`br_if` 跳到对应 body
   - 用 block/br_if 结构实现跳转

**验收**：能编译并执行包含 enum match 的程序

---

### Task A9: export 名编码修复

**文件**：`tenthc/compile/wasm.th`

**步骤**：
1. 修复 export section 的 ASCII 编码：
   - 当前仅支持 a-z (97-122), A (65), 0-9 (48-57), _ (95)
   - 补全 A-Z (65-90) 的完整映射
   - 其他字符仍用 `?` (63) 占位
2. 验证：函数名 `Vec_new`、`HashMap_new` 等含大写字母的函数名能正确 export

**验收**：`tenthc_full.wasm` 中所有函数名能正确 export

---

### Task A10: local 槽位动态分配

**文件**：`tenthc/compile/wasm.th`

**步骤**：
1. 在 `WasmCtx` 新增 `local_map: Vec<(String, i64)>` 和 `local_count: i64`
2. `compile_stmt` 的 Let 分支：
   - 分配新 local index = `param_count + local_count`
   - `local_count += 1`
   - 记录 `(var_name, local_index)` 到 `local_map`
   - 编译 init expr，`local.set local_index`
3. `compile_expr` 的 Var 分支：
   - 先查 `hir.fns[].params`（参数）
   - 再查 `local_map`（let 绑定的局部变量）
   - 命中则 `local.get index`
4. `compile_expr` 的 Assign 分支：
   - 同上查找逻辑，`local.set index`
5. 函数声明 local：`local_count` 个 i64 local（替代固定 16 个）

**验收**：能编译包含 20+ 个 let 绑定的函数，无 local 溢出

---

### Task A11: 类型推断最小集

**文件**：`tenthc/hir/lower.th`

**步骤**：
1. 新增 `TypeEnv` struct（简化版）：
   ```
   struct TypeEnv {
     vars: Vec<(String, i64)>,    // (var_name, type_disc)
     fns: Vec<(String, i64)>,     // (fn_name, return_type_disc)
     structs: Vec<(String, Vec<String>)>,  // (struct_name, field_names)
   }
   ```
   type_disc: 0=unknown, 1=i64, 2=f64, 3=bool, 4=str, 5=struct(name_idx)
2. `lower_program` 第一遍：收集所有 fn 签名（返回类型）和 struct 定义（字段名列表）到 TypeEnv
3. `lower_expr` 设置 ty 字段：
   - IntLiteral → ty=1 (i64)
   - FloatLiteral → ty=2 (f64)
   - BoolLiteral → ty=3 (bool)
   - StringLiteral → ty=4 (str)
   - Var → 查 TypeEnv.vars
   - Binary → `infer_binary_type(left_ty, right_ty, op)`
   - Call → 查 TypeEnv.fns
   - Field → 查 TypeEnv.structs 找到 struct_name，设置 ty=5 (struct)
4. `infer_binary_type(l, r, op)`：
   - 比较运算 (==/!=/</>/<=/>=) → 3 (bool)
   - 算术运算：若 l==2 或 r==2 → 2 (f64)；否则 → 1 (i64)
5. `lower_stmt` 的 Let 分支：将 `(name, init_ty)` 加入 TypeEnv.vars

**验收**：
- struct 字段访问能正确计算偏移
- f64 运算能正确选择 f64 操作码
- Binary 比较运算返回 bool 类型

---

### Task A12: Phase A 集成测试

**文件**：`tenth/tests/wasm_backend_minimal.rs`（新建）

**步骤**：
1. 编写测试：用 Rust 母编译器编译 tenthc/*.th 拼接 + 测试程序，得到 WASM
2. 用 wasmi 执行 WASM，验证结果
3. 测试用例：
   - `fn add(a:i64,b:i64)->i64{a+b}` → add(3,4)=7
   - `fn fadd(a:f64,b:f64)->f64{a+b}` → fadd(1.5,2.0)=3.5
   - `fn str_test()->i64{ let s="hello"; let t=" world"; println(s+t); 0 }`
   - `fn for_sum(n:i64)->i64{ let s=0; for i in 0..n { s=s+i; } s }` → for_sum(5)=10
   - `fn loop_count()->i64{ let i=0; loop { if i>=10 { break; } i=i+1; } i }` → 10
   - `fn while_count(n:i64)->i64{ let i=0; while i<n { i=i+1; } i }` → while_count(5)=5

**验收**：全部测试通过

---

## Phase B：前端对齐

> **目标**：tenthc 的 parser + lowerer 能正确处理 tenthc 自身的 6 个 .th 文件。
> **验收测试**：`tenth/tests/selfhost_frontend.rs`

### Task B1: 模块系统

**文件**：`tenthc/parser/parser.th`、`tenthc/hir/lower.th`

**步骤**：
1. Parser 新增 `parse_use`：解析 `use path::name` 或 `use path::*`
2. Parser 新增 `parse_mod`：解析 `mod name { ... }`
3. Lowerer 新增 `try_import_file`：
   - 根据 use path 查找文件：`<base>/<path>.th` 或 `<base>/<path>/mod.th`
   - 递归 lex → parse → lower
   - `imported_files` HashSet 防循环
4. Lowerer 的 `lower_program` 支持 use 声明：将导入的符号加入 scope

**验收**：tenthc 能解析 `use std::vec::Vec;` 并识别 Vec 类型

---

### Task B2: impl 块解析

**文件**：`tenthc/parser/parser.th`、`tenthc/hir/lower.th`

**步骤**：
1. Parser 新增 `parse_impl_block`：
   - 解析 `impl TypeName { fn method(...) {...} ... }`
   - 将方法存入 Program 的 `methods` 数组
2. Lowerer 新增 `methods: Vec<(String, String, HirFnDef)>`（type_name, method_name, fn_def）
3. `lower_program` 第二遍：处理 impl 块，收集方法到 methods

**验收**：tenthc 能解析 `impl Foo { fn bar(&self) -> i64 { 0 } }`

---

### Task B3: enum 定义收集

**文件**：`tenthc/hir/lower.th`

**步骤**：
1. Lowerer 新增 `enums: Vec<(String, Vec<String>)>`（enum_name, variant_names）
2. `lower_program`：收集所有 enum 定义到 enums
3. Match 表达式的 enum_variant pattern 能查 enums 验证

**验收**：tenthc 能处理 `enum Color { Red, Green, Blue }` + `match c { Color::Red => 1, _ => 0 }`

---

### Task B4: struct 字段类型推断

**文件**：`tenthc/hir/lower.th`

**步骤**：
1. 扩展 TypeEnv.structs 为 `Vec<(String, Vec<(String, i64)>)>`（struct_name, [(field_name, field_type_disc)]）
2. `lower_program` 第一遍：从 StructDef 收集字段名和类型
3. `lower_expr` 的 Field 分支：根据 `e.ty` 查 struct 名，再查字段偏移

**验收**：struct 字段访问能正确计算偏移

---

### Task B5: 闭包捕获分析

**文件**：`tenthc/hir/lower.th`

**步骤**：
1. 新增 `free_vars_in(expr, scope) -> Vec<String>`
2. `lower_expr` 的 Closure 分支：调用 `free_vars_in` 分析捕获
3. 将捕获变量存入 HirExpr 的 `captures_start` + `captures_count`

**验收**：闭包能正确识别捕获的变量

---

### Task B6: Range 表达式 lower

**文件**：`tenthc/hir/lower.th`

**步骤**：
1. `lower_expr` 新增 "range" 分支：
   - 降级 start 和 end
   - 存入 HirExpr disc 28 (Range)，字段 `left` (start), `right` (end), `range_inclusive`

**验收**：`0..n` 能正确降级为 Range HIR 节点

---

### Task B7: AssignOp 修复

**文件**：`tenthc/compile/wasm.th`

**步骤**：
1. `compile_expr` 的 disc 24 (AssignOp) 分支修复：
   - 当前：直接 `local.set`，未执行运算
   - 修复：`local.get` → 编译 rhs → 运算 → `local.set`
2. disc 27 (DerefAssignOp) 同理修复

**验收**：`x += 5` 正确执行加法后赋值

---

### Task B8: Var/Let 变量解析修复

**文件**：`tenthc/compile/wasm.th`

**步骤**：
1. 依赖 A10（local 槽位动态分配）已完成
2. 验证 Var 分支能查找 let 绑定的变量
3. 验证 Assign 分支能赋值给 let 绑定的变量

**验收**：`let x = 1; x = x + 2;` 正确执行

---

### Task B9: Phase B 集成测试

**文件**：`tenth/tests/selfhost_frontend.rs`（新建）

**步骤**：
1. 用 tenthc 的 lexer/parser/lowerer 处理 tenthc/*.th 全部 6 个文件
2. 验证 HIR 节点数和结构符合预期
3. 验证无解析错误、无 lower 错误

**验收**：tenthc 能正确解析自身源码

---

## Phase C：自举闭环

> **目标**：tenthc wasm.th 能编译 tenthc 自身源码，Stage N ≡ Stage N+1。
> **验收测试**：`tenth/tests/three_stage.rs`（取消 ignore）

### Task C1: tenthc 源码自举适配

**文件**：`tenthc/*.th`

**步骤**：
1. 审计 tenthc/*.th 的特性使用，确保只使用 wasm.th 支持的特性
2. 必要时重构 tenthc 源码：
   - 将不支持的特性替换为支持的等价写法
   - 例如：闭包 → 提取为独立函数 + 显式传参
3. 确保 tenthc 源码能用 tenthc 自身的 wasm.th 编译

**验收**：tenthc wasm.th 能编译 tenthc/*.th 产出合法 WASM

---

### Task C2: boot.th 同步或废弃

**文件**：`tenthc/boot.th`、`tenthc/main.th`

**步骤**：
1. 评估 boot.th 是否仍需要：
   - 若模块化版本 + 模块系统（B1）可用 → 废弃 boot.th
   - 否则 → 将模块化版本同步到 boot.th
2. 更新 main.th 的文件加载逻辑

**验收**：tenthc 入口能正确加载所有源码

---

### Task C3: three_stage.rs 修复与优化

**文件**：`tenth/tests/three_stage.rs`

**步骤**：
1. 取消 `#[ignore]`
2. 优化 wasmi 执行：
   - 使用 `Config::default().set_compilation_mode(CompilationMode::Eager)` 预编译
   - 或考虑用 wasm3 替代 wasmi（更快）
3. 增加 stack_size 到 128MB
4. 验证三阶段流程：
   - Stage 1: Rust 编译 tenthc → WASM-A
   - Stage 2: WASM-A 执行，编译 `fn add(a,b){a+b}` → WASM-B
   - Stage 3: 验证 WASM-B 能执行 add(3,4)=7

**验收**：three_stage_selfhost 测试通过

---

### Task C4: 固定点验证

**文件**：`tenth/tests/three_stage.rs`

**步骤**：
1. 扩展 three_stage.rs：
   - Stage 2: tenthc_stage1 编译 tenthc/*.th → tenthc_stage2.wasm
   - Stage 3: tenthc_stage2 编译 tenthc/*.th → tenthc_stage3.wasm
   - 验证：tenthc_stage2.wasm ≡ tenthc_stage3.wasm（字节级比较）
2. 若字节级不等（因时间戳/地址等），改为语义级验证：
   - 两者编译同一测试程序，结果一致

**验收**：固定点达成

---

## Phase D：能力对等

> **目标**：tenthc 与 Rust 母编译器功能对等。
> **验收测试**：`tenth/tests/parity_test.rs`（扩展）
> **实施优先级**：D4 → D1 → D2 → D3 → D5 → D6（可选）
> **当前状态（2026-06-25）**：D1/D4/D7 已完成（120 用例），D2/D3/D5/D6 未开始

### Task D4: 完整 native 函数对齐 ✅ 已完成

**文件**：`tenthc/compile/wasm.th`、`tenth/src/compile/wasm.rs`（host 侧已就绪）

**现状证据**（2026-06-25 复核）：
- tenthc `wasm.th:1239` — `let num_imports: i64 = 17;`（已从 15 扩展到 17）
- tenthc `wasm.th:1298-1304` — Type 10 (f64_bits) 和 Type 11 (str_slice) 已声明
- tenthc `wasm.th:1321+` — 17 个 import 已全部写入 import section
- Rust `wasm.rs:60` — `const IMPORT_COUNT: u32 = 17;`
- Rust `wasm.rs:243-244` — `host.f64_bits` (idx 15) 和 `host.str_slice` (idx 16) 已存在
- Rust `wasm.rs:515-555` — f64_bits/str_slice host 实现已就绪
- Rust `wasm.rs:850-868` — Slice 表达式已编译为 str_slice 调用

**完成状态**：D4.1-D4.5 全部完成，两侧 import 对齐，Slice 表达式正确编译。

**步骤**：
1. `wasm.th:1239` — `num_imports` 从 15 改为 17（已完成）
2. `wasm.th` import section（第 1235-1315 行附近）新增：
   - idx 15: `env.f64_bits` (f64)->i64 — 模块名 `env`，函数名 `f64_bits`，1 个 f64 参数，返回 i64
   - idx 16: `env.str_slice` (i32,i64,i64)->i32 — 3 个参数（str_ptr, start, end），返回 i32
3. `wasm.th:87-97` wasm_f64_const 函数：将 `f64_bits(val)` 调用改为 `call_import(15, val)`（使用 import 15）
4. `wasm.th:967-978` compile_expr 的 Slice 分支：
   - 编译 target 表达式（str 指针）
   - 编译 start 表达式（i64）
   - 编译 end 表达式（i64）
   - 调用 `call_import(16, target, start, end)` 返回切片指针
   - 移除 `TODO: needs host str_slice import` 注释和占位 `wasm_i64_const(out, 0)`
5. Rust 侧 `wasm.rs:243-244` 已有 f64_bits/str_slice 的 host 实现，无需改动
6. `parity_test.rs` 新增 Slice 测试用例：`fn slice_test() -> i64 { let s = "hello"; let t = s[1..3]; str_len(t) }`

**验收**：
- tenthc 生成的 WASM import 数量 = 17
- Slice 表达式正确编译并执行
- Rust 母编译器 499+ 测试无回归

---

### Task D1: Trait 系统 ✅ 已完成

**文件**：`tenthc/hir/hir.th`、`tenthc/parser/parser.th`、`tenthc/hir/lower.th`、`tenth/src/hir/lower.rs`（Rust 侧同步方法分派）

**完成证据**（2026-06-25）：
- tenthc `hir.th` — 新增 HirTraitMethod/HirTraitDef/HirTraitImpl 结构 + HirProgram 6 个 trait 相关字段
- tenthc `parser.th` — 新增 parse_trait_def/parse_impl_block，支持 `self`/`&self`/`&mut self` 参数语法
- tenthc `lexer.th:99` — 修复 self token 缺少 sval 的 bug
- tenthc `lower.th` — 收集 trait 定义/impl 方法（mangled name `__<Type>_<method>`），预置 Display/Eq/Clone
- tenthc `lower.th:700-708` — impl 方法 lowering 时 self 参数 type_ann 覆盖为 imp.type_name（修复字段偏移计算）
- tenthc `lower.th:280-310` — MethodCall 分派：resolve_user_method 命中则改写为常规 Call(disc 7)
- Rust `lower.rs:1449` — inherent impl 方法以 mangled name 注册到 functions
- Rust `lower.rs:505` — MethodCall 分派：receiver 类型为 Struct/TypeParam 且存在 mangled 函数时改写为 Call
- `parity_test.rs` — 新增 3 个 D1 测试用例全部通过

**步骤**：

**D1.1: hir.th 新增 Trait 结构**
1. `hir.th` 新增 `struct HirTraitDef { name: str, method_names: Vec<str>, method_sigs: Vec<HirFnSig> }`
2. `hir.th` 新增 `struct HirTraitImpl { trait_name: str, type_name: str, methods: Vec<HirFnDef> }`
3. `hir.th` HirProgram 新增字段：`trait_defs: Vec<HirTraitDef>` + `trait_impls: Vec<HirTraitImpl>`

**D1.2: parser.th 新增 parse_trait/parse_impl_for**
1. `parser.th` 新增 `fn parse_trait() -> Item`：
   - 消费 `trait` token (disc=19)
   - 解析 trait 名（Identifier）
   - 解析 `{ fn method(...); ... }` 方法签名列表（无方法体，以 `;` 结尾）
2. `parser.th` 新增 `fn parse_impl_for() -> Item`：
   - 消费 `impl` token (disc=18)
   - 解析 trait 名 + `for` + 类型名
   - 解析 `{ fn method(...) {...} ... }` 方法定义列表（有方法体）
3. 参考 Rust 侧 `parser.rs` 的 trait/impl 解析逻辑

**D1.3: parser.th parse_program 新增分支**
1. `parser.th:1131-1156` parse_program 新增：
   - disc=19 (trait) → 调用 parse_trait
   - disc=18 (impl) → 调用 parse_impl_for

**D1.4: lower.th 收集 trait 定义和实现**
1. `lower.th` 新增 `trait_defs: Vec<(String, HirTraitDef)>` 和 `trait_impls: Vec<(String, String, Vec<HirFnDef>)>`（trait_name, type_name, methods）
2. `lower.th` lower_program 第一遍：收集所有 trait 定义到 trait_defs
3. `lower.th` lower_program 第二遍：收集所有 trait impl 到 trait_impls
4. 参考 Rust `lower.rs:1327`（trait 定义）和 `lower.rs:1392-1398`（trait impl）

**D1.5: lower.th 实现 trait 方法静态分派**
1. `lower.th` lower_expr 的 MethodCall 分支：
   - 查 receiver 类型
   - 查 trait_impls 找到 (trait_name, type_name) 匹配的方法
   - 将方法调用改写为普通 Call（mangled name: `type_name_method_name`）
2. 参考 Rust `lower.rs:1392-1398` 的方法解析逻辑

**验收**：
- [x] tenthc 能解析 `trait MyTrait { fn foo(self) -> i64; }`
- [x] tenthc 能解析 `impl Pair { fn sum(self) -> i64 { self.a + self.b } }`
- [x] inherent impl 方法调用 `p.sum()` 能正确静态分派到 `__Pair_sum`
- [x] parity_test.rs 新增 3 个 Trait 测试用例通过
- [x] Rust 母编译器 499+ 测试无回归

---

### Task D2: 泛型实例化

**文件**：`tenthc/parser/parser.th`、`tenthc/hir/hir.th`、`tenthc/hir/lower.th`

**现状证据**：
- tenthc `hir.th:98` — HirFnDef 有 `generics: Vec<str>` 字段
- tenthc `lower.th:662` — `generics: Vec::new()` 始终为空
- tenthc `parser.th` parse_fn — 不解析 `<T>` 语法
- Rust `lower.rs:144` — `generic_funcs: HashMap<String, HirFnDef>`
- Rust `lower.rs:464` — 泛型函数实例化 `self.generic_funcs.get(&func_name)`
- Rust `lower.rs:1871` — `substitute_type` 函数

**步骤**：

**D2.1: parser.th 解析泛型参数**
1. `parser.th` parse_fn 新增：解析 `fn` 后可选的 `<T, U>` 部分
   - 遇到 `<` (disc=lt) 开始解析泛型参数列表
   - 收集 Identifier 列表到 fn.generics
   - 遇到 `>` 结束
2. `parser.th` parse_struct 同理：解析 `struct Name<T, U> { ... }` 中的 `<T, U>`

**D2.2: hir.th 填充 generics 字段**
1. `lower.th:662` — 将 `generics: Vec::new()` 改为从 AST 读取实际的泛型参数列表

**D2.3: lower.th 新增 generic_funcs map**
1. `lower.th` 新增 `generic_funcs: Vec<(String, HirFnDef)>`（函数名, 模板定义）
2. lower_program 第一遍：收集所有带 generics 的函数到 generic_funcs（不立即 lower 函数体）

**D2.4: lower.th 新增 substitute_type**
1. `lower.th` 新增 `fn substitute_type(ty: HirType, type_map: Vec<(String, HirType)>) -> HirType`：
   - 若 ty 是类型参数（在 type_map 中找到），返回映射的具体类型
   - 否则原样返回
2. 参考 Rust `lower.rs:1871`

**D2.5: lower.th GenericCall 实例化**
1. `lower.th` lower_expr 的 Call 分支：
   - 若函数名在 generic_funcs 中且调用点有泛型实参：
     - 根据实参类型构建 type_map
     - 用 substitute_type 实例化参数和返回类型
     - lower 实例化后的函数体（mangled name: `fn_name_T1_T2`）
2. 参考 Rust `lower.rs:464`

**验收**：
- tenthc 能解析 `fn id<T>(x: T) -> T { x }`
- tenthc 能解析 `struct Pair<T, U> { first: T, second: U }`
- 泛型函数调用 `id(42)` 能正确实例化
- parity_test.rs 新增 ≥3 个泛型测试用例通过

---

### Task D3: 借用检查

**文件**：`tenthc/hir/lower.th`

**现状证据**：
- tenthc `lower.th` — 完全无 Ownership/check_borrow 相关代码
- Rust `lower.rs:11-16` — `Ownership` enum（Owned/SharedRef(usize)/ExclusiveRef/Moved）
- Rust `lower.rs:21` — `Scope.ownership: HashMap<String, Ownership>`
- Rust `lower.rs:62` — `check_use`：检查变量是否被 move
- Rust `lower.rs:72` — `check_borrow_shared`：检查共享借用冲突
- Rust `lower.rs:86` — `check_borrow_mut`：检查独占借用冲突
- Rust `lower.rs:833-872` — ref/mutref/move 表达式更新 ownership

**步骤**：

**D3.1: lower.th 新增 Ownership enum**
1. `lower.th` 新增 Ownership 常量：`const OWNED: i64 = 0; const SHARED_REF: i64 = 1; const EXCLUSIVE_REF: i64 = 2; const MOVED: i64 = 3;`

**D3.2: lower.th Scope 新增 ownership 字段**
1. `lower.th` Scope 结构新增 `ownership: Vec<(String, i64)>`（变量名, Ownership 状态）

**D3.3: lower.th 实现 check_use**
1. `lower.th` 新增 `fn check_use(scope: Scope, name: str)`：
   - 查 scope.ownership 找到变量的 ownership 状态
   - 若为 MOVED，报 use-after-move 错误
2. 参考 Rust `lower.rs:62`

**D3.4: lower.th 实现 check_borrow_shared/check_borrow_mut**
1. `lower.th` 新增 `fn check_borrow_shared(scope: Scope, name: str)`：
   - 若已有 EXCLUSIVE_REF，报借用冲突错误
2. `lower.th` 新增 `fn check_borrow_mut(scope: Scope, name: str)`：
   - 若已有 SHARED_REF 或 EXCLUSIVE_REF，报借用冲突错误
3. 参考 Rust `lower.rs:72,86`

**D3.5: lower.th 更新 ownership 状态**
1. `lower.th` lower_expr 的 Ref 分支：调用 check_borrow_shared，设置 ownership=SHARED_REF
2. `lower.th` lower_expr 的 MutRef 分支：调用 check_borrow_mut，设置 ownership=EXCLUSIVE_REF
3. `lower.th` lower_expr 的 Move 分支：设置 ownership=MOVED
4. `lower.th` lower_stmt 的 Let/Assign 分支：调用 check_use 验证变量未被 move
5. 参考 Rust `lower.rs:833-872`

**验收**：
- `let a = String::new(); let b = a; a.len();` 报 use-after-move 错误
- `let mut a = 1; let r = &mut a; let s = &mut a;` 报双重借用错误
- parity_test.rs 新增 ≥3 个借用检查测试用例通过

---

### Task D5: 闭包 WASM 后端实现

**文件**：`tenthc/hir/lower.th`、`tenthc/compile/wasm.th`、`tenth/src/compile/wasm.rs`

**现状证据**：
- tenthc `wasm.th:960-963` — 闭包占位：`if d == 23 { wasm_i64_const(out, 0); return; }`
- tenthc `lower.th:427-434` — 有闭包 lowering，但第 429 行 `let captures_count: i64 = 0;` 注释 `Captures — simplified: no deep analysis in self-hosting compiler`，captures_count 始终为 0
- tenthc `parser.th` — 有闭包解析（`|params| body` 语法）
- Rust `wasm.rs:840` — `_ => return Err(...)` 闭包返回错误

**步骤**：

**D5.1: lower.th 实现闭包捕获分析**
1. `lower.th:429` — 实现 `fn free_vars_in(expr: Expr, scope: Vec<str>) -> Vec<str>`：
   - 递归遍历表达式，收集不在 params 中的变量引用
2. `lower.th:427-434` Closure 分支：调用 free_vars_in 填充 captures_start/captures_count

**D5.2: wasm.th 闭包表示设计**
1. 闭包表示为两个 i64：`fn_ptr`（函数索引）+ `env_ptr`（捕获环境指针）
2. 无捕获闭包：env_ptr = 0
3. 有捕获闭包：env_ptr 指向 tenth_alloc 分配的 struct，字段为捕获变量

**D5.3: wasm.th compile_expr disc 23 (Closure) 实现**
1. `wasm.th:960-963` 替换占位代码：
   - 将闭包体编译为独立函数（mangled name: `closure_N`）
   - 若有捕获：调用 tenth_alloc 分配 env struct，填充捕获变量
   - 压入 fn_ptr（i64.const function_index）
   - 压入 env_ptr（i64.const env 地址或 0）

**D5.4: wasm.th 闭包调用实现**
1. 闭包调用：从栈上取出 fn_ptr + env_ptr
2. 调用 fn_ptr(env_ptr, args...)
3. 需要间接调用支持（call_indirect 或函数表）

**D5.5: Rust 侧 wasm.rs 同步实现**
1. `wasm.rs:840` _ 分支替换为 Closure 处理（与 tenthc 对齐）
2. 新增闭包相关 host import（若需要）

**验收**：
- 无捕获闭包 `let f = |x: i64| x + 1; f(2)` 返回 3
- 有捕获闭包 `let n = 10; let f = |x: i64| x + n; f(5)` 返回 15
- parity_test.rs 新增 ≥2 个闭包测试用例通过

**风险**：闭包 env 捕获需要堆分配；建议先实现无捕获闭包，再扩展到有捕获

---

### Task D6: Tensor WASM 后端实现（可选）

**文件**：`tenthc/compile/wasm.th`、`tenth/src/compile/wasm.rs`

**现状证据**：
- tenthc `wasm.th:1030-1031` — Tensor 占位：`if d == 29 { wasm_i64_const(out, 0); return; }`
- tenthc `lower.th` — 有 tensor lowering（e.ival 存储行数）
- tenthc `parser.th` — 有 tensor 字面量解析（`[[...], [...]]` 语法）
- Rust `wasm.rs:840` — `_ => return Err(...)` tensor 返回错误

**步骤**：

**D6.1: wasm.th 新增 tensor host import**
1. `wasm.th` import section 新增：
   - `env.tensor_from_vec` (i32, i32, i32) -> i64 — (data_ptr, len, rank) -> tensor_handle
   - `env.tensor_dot` (i64, i64) -> i64 — (a, b) -> result
   - `env.tensor_matmul` (i64, i64) -> i64
   - 其他必要操作（zeros/ones/shape 等）
2. Rust 侧 `wasm.rs` 同步新增对应 import 和 host 实现（复用 `runtime/tensor.rs`）

**D6.2: wasm.th compile_expr disc 29 实现**
1. `wasm.th:1030-1031` 替换占位代码：
   - 将 tensor 元素展开为 i64 数组（存入 memory）
   - 调用 `tensor_from_vec(data_ptr, len, rank)` 创建 tensor
   - 压入 tensor_handle

**D6.3: Rust 侧 wasm.rs 同步实现**
1. `wasm.rs:840` _ 分支替换为 TensorLiteral 处理
2. 新增 tensor 相关 host import 实现

**验收**：
- `let t = [[1, 2], [3, 4]];` 能编译为 tensor_from_vec 调用
- parity_test.rs 新增 ≥1 个 Tensor 测试用例通过

**风险**：Tensor host import 套件庞大；可标记为可选，Tensor 操作主要走 VM/解释器路径

---

### Task D7: parity_test.rs 扩展（已完成基础，待扩展）

**文件**：`tenth/tests/parity_test.rs`

**现状**：117 个测试用例通过，覆盖算术/变量/控制流/struct/递归等基础特性

**未覆盖特性（待 D1-D6 完成后扩展）**：
- Trait 方法分派（待 D1）
- 泛型函数实例化（待 D2）
- 借用检查场景（待 D3）
- Slice 表达式（待 D4）
- 闭包创建和调用（待 D5）
- Tensor 字面量（待 D6，可选）
- Match 表达式（Rust 侧 WASM 不支持，待后续）
- Float 算术（tenthc 函数签名推断问题，待修复）

**步骤**：
1. 随 D1-D6 完成逐步新增对应测试用例
2. 每完成一项 D 任务，新增 ≥2 个 parity 测试用例
3. 目标：用例数从 117 扩展到 150+

**验收**：90%+ 测试用例通过

---

## 执行顺序与依赖关系

```
Phase A (WASM 后端最小可用) ✅ 已完成
Phase B (前端对齐) ✅ 已完成
Phase C (自举闭环) ✅ 已完成（固定点除外）

Phase D (能力对等) — 当前阶段
  优先级 1（最小补齐）：
    D4 (native 对齐) ✅ 已完成 ─── 17 个 import 两侧对齐

  优先级 2（核心高级特性）：
    D1 (Trait 系统) ✅ 已完成 ─── trait 定义/inherent impl/方法静态分派（120 用例）
    D2 (泛型实例化) ──────┐ (D2 建议 D1 之后，Trait bound 常配泛型)
    D3 (借用检查) ──────── (可与 D1/D2 并行，建议之后)

  优先级 3（WASM 后端扩展）：
    D5 (闭包 WASM) ─────── (依赖 D4 ✅ 的 tenth_alloc import)
    D6 (Tensor WASM) ──── (依赖 D4 ✅，可选，建议 D5 之后)

  优先级 4（测试覆盖）：
    D7 扩展 ────────────── 随 D1-D6 完成逐步新增用例（当前 117 用例）
```
