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
> **验收测试**：`tenth/tests/parity_test.rs`

### Task D1: Trait 系统

**文件**：`tenthc/hir/hir.th`、`tenthc/hir/lower.th`

**步骤**：
1. HIR 新增 HirTraitDef、HirTraitImpl 结构
2. Lowerer 新增 trait_defs、trait_impls map
3. `lower_program`：收集 trait 定义和实现
4. 方法调用分派：按 receiver 类型查 methods map

---

### Task D2: 泛型实例化

**文件**：`tenthc/hir/lower.th`

**步骤**：
1. 新增 `substitute_type(ty, type_map)`
2. Lowerer 新增 generic_funcs map
3. GenericCall 处理：构建 type_map，实例化参数和返回类型

---

### Task D3: 借用检查

**文件**：`tenthc/hir/lower.th`

**步骤**：
1. 新增 Ownership enum
2. Scope 维护 ownership map
3. check_use / check_borrow_shared / check_borrow_mut

---

### Task D4: 完整 native 函数对齐

**文件**：`tenthc/compile/wasm.th`、`tenth/src/compile/wasm.rs`

**步骤**：
1. 扩展 WASM import 到覆盖 85+ 个 native 函数
2. 或：通过 compile_host 委托给 Rust 解释器执行（过渡方案）

---

### Task D5: 闭包 WASM 后端实现

**文件**：`tenthc/compile/wasm.th`

**步骤**：
1. 闭包表示：struct { fn_ptr, env_ptr }
2. Closure 节点：分配 env struct，填充捕获变量，返回 fn_ptr+env_ptr
3. 闭包调用：调用 fn_ptr(env_ptr, args)

---

### Task D6: Tensor WASM 后端实现

**文件**：`tenthc/compile/wasm.th`

**步骤**：
1. Tensor 通过 host import 实现（tensor/tensor_from_vec/zeros/ones 等作为 import）
2. TensorLiteral → 调用 tensor_from_vec import

---

### Task D7: parity_test.rs

**文件**：`tenth/tests/parity_test.rs`（新建）

**步骤**：
1. 对同一 Tenth 程序，分别用 Rust 母编译器和 tenthc 编译
2. 用 wasmi 执行两者产出的 WASM
3. 验证结果一致

**验收**：90%+ 测试用例通过

---

## 执行顺序与依赖关系

```
Phase A (WASM 后端最小可用)
  A1 (f64) ──────────────────────┐
  A2 (字符串驻留) ──┐             │
  A3 (import 对齐) ─┤             │
  A4 (InterpString) ┤ (依赖 A2,A3)│
  A5 (for) ─────────┤             │
  A6 (loop) ────────┤             │
  A7 (while) ───────┤             │
  A8 (Match) ───────┤             │
  A9 (export 编码) ─┤             │
  A10 (local 动态) ─┤             │
  A11 (类型推断) ───┘ (依赖 A1)   │
  A12 (集成测试) ──── (依赖全部 A) │
                                   │
Phase B (前端对齐)                 │
  B1 (模块系统) ──────┐            │
  B2 (impl 块) ───────┤            │
  B3 (enum 收集) ─────┤            │
  B4 (struct 类型) ───┤ (依赖 A11) │
  B5 (闭包捕获) ──────┤            │
  B6 (Range lower) ───┤            │
  B7 (AssignOp 修复) ─┤            │
  B8 (Var/Let 修复) ──┘ (依赖 A10) │
  B9 (集成测试) ────── (依赖全部 B)│
                                   │
Phase C (自举闭环)                 │
  C1 (源码适配) ──┐                │
  C2 (boot.th) ───┤ (依赖 B1)      │
  C3 (three_stage)┤ (依赖 A,B)     │
  C4 (固定点) ────┘ (依赖 C3)      │
                                   │
Phase D (能力对等)                 │
  D1-D7 (独立任务，可并行) ────────┘
```
