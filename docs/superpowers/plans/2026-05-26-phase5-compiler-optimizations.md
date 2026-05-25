# Phase 5: 编译优化 实施计划

> 状态：📋 计划中 | 前置条件：Phase 4 标准库完成

**Goal:** 在 MIR 层实现优化 pass，包括常量折叠、死代码消除、函数内联、基本块合并，使编译产物（C 代码 / 未来 LLVM IR）质量接近手写。

**Architecture:** 新增 `compile/optimize.rs` 模块，提供 MIR→MIR 的优化 pass 框架。每个 pass 是 `fn(&mut MirProgram)` 的形式，可组合执行。

**Tech Stack:** Rust 2024 edition，纯逻辑无外部依赖。

---

## 文件结构（Phase 5 新增/修改）

```
tenth/src/compile/
├── optimize.rs          # 新增: 优化 pass 框架与各 pass 实现
├── mod.rs               # 修改: compile_to_c 插入优化阶段
```

---

### Task 1: 优化 Pass 框架

- [ ] 定义 `OptimizationPass` trait
- [ ] 实现 pass 管道：`run_passes(program, passes)`
- [ ] 添加编译 flag：`tenth compile --opt-level=1 input.th`

```
pub trait OptimizationPass {
    fn name(&self) -> &str;
    fn run(&self, program: &mut MirProgram);
}
```

---

### Task 2: 常量折叠 (Constant Folding)

**目标:** 编译期计算常量表达式。

```
let x = 2 + 3 * 4;  // → let x = 14;
if true { ... }      // → 直接取 then 分支
```

- [ ] 遍历 MirRvalue，对全是 Literal 的 BinaryOp/UnaryOp 求值替换
- [ ] If 条件为常量 Bool 时，消除死分支
- [ ] 测试：编译 `2 + 3 * 4` → C 代码中出现 `14` 而非 `(2 + (3 * 4))`

---

### Task 3: 死代码消除 (Dead Code Elimination)

**目标:** 移除不可达的基本块和未使用的局部变量。

- [ ] 活跃性分析：标记所有使用的变量
- [ ] 移除未被任何块跳转到的基本块
- [ ] 移除 `StorageLive` 后无 `StorageDead` 的局部变量
- [ ] 测试：编译 `{ let x = 42; 10 }` → x 被消除

---

### Task 4: 函数内联 (Function Inlining)

**目标:** 对小函数在调用处内联展开，消除调用开销。

```
fn add(a: i32, b: i32) -> i32 { a + b }
let x = add(1, 2);  // → let x = 1 + 2;
```

- [ ] 启发式：函数体 < 5 条语句时内联
- [ ] 参数替换：形参→实参
- [ ] 局部变量重命名避免冲突
- [ ] 测试：编译后 C 代码中无 `add(` 调用

---

### Task 5: 基本块合并与跳转优化

- [ ] 合并连续无条件跳转的基本块
- [ ] 消除 `goto next_block` 后的冗余标签
- [ ] 跳转线程化：`goto A; A: goto B` → `goto B`

---

### Task 6: 公共子表达式消除 (CSE — 可选)

- [ ] 检测同一基本块内的重复计算
- [ ] `let t1 = a + b; ... let t2 = a + b;` → 复用 t1

---

### Task 7: 回归测试与性能验证

- [ ] 运行全部 60+ 测试，确保优化不改变语义
- [ ] 对比优化前后 C 代码行数
- [ ] 编写优化专项测试（验证折叠/内联/死代码消除）

---

## Phase 5 完成标准

- [ ] 至少 4 个优化 pass 可用（常量折叠 + 死代码消除 + 内联 + 块合并）
- [ ] `--opt-level=1` flag 生效
- [ ] 优化后 C 代码行数减少 20%+（相比未优化）
- [ ] 全部 60+ 回归测试通过
