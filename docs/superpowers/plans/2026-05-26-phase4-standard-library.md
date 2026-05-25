# Phase 4: 标准库与运行时 实施计划

> 状态：📋 计划中 | 前置条件：Phase 3A + Phase 3B 完成

**Goal:** 将当前散落在解释器中的内置函数（println、tensor、rand 等）提取为标准库模块，补齐基础类型的方法（字符串操作、数组操作）、建立 I/O 体系，让 Tenth 具备独立编写实用程序的能力。

**Architecture:** 标准库以 `.th` 源文件形式存放在 `tenth/std/` 目录下，通过模块系统加载。编译器和解释器在启动时预加载核心模块。内置函数逐步迁移为标准库方法。

**Tech Stack:** Tenth 语言自身（标准库用 Tenth 编写），Rust（编译器/解释器侧少量原生函数）。

---

## 文件结构（Phase 4 新增/修改）

```
tenth/
├── std/
│   ├── prelude.th        # 预加载：自动导入 std::io, std::math, std::tensor
│   ├── io.th             # println, eprintln, read_line, read_file, write_file
│   ├── math.th           # abs, sqrt, exp, log, sin, cos, pow
│   ├── tensor.th         # 张量创建/运算/方法（tensor, rand, randn, zeros, ones）
│   ├── collections.th    # Array 方法 (len, push, pop, map, filter)
│   ├── string.th         # String 方法 (len, concat, split, contains)
│   ├── convert.th        # 类型转换 (to_string, parse_int, parse_float)
│   └── test.th           # 简易测试框架 (assert, assert_eq, test 宏)
├── src/
│   ├── stdlib.rs         # Rust 侧标准库加载器
│   └── lib.rs            # 导出 stdlib 模块
```

---

### Task 1: 标准库基础设施

**目标:** 建立模块加载体系，让 `.th` 文件能在编译/解释时被预加载。

- [ ] **Step 1: 核心原生函数注册表**

在 Rust 侧定义"原生函数"接口，允许标准库调用 Rust 实现的高性能函数：
```rust
// stdlib.rs
pub struct NativeFn {
    pub name: String,
    pub handler: fn(&[Value]) -> TenthResult<Value>,
}

pub fn native_functions() -> Vec<NativeFn> { ... }
```

- [ ] **Step 2: 预加载机制**

解释器/编译器启动时自动加载 `std/prelude.th`，执行其中的 `use` 导入。

- [ ] **Step 3: 模块路径解析**

`use std::io::println` 解析为 `tenth/std/io.th` 中的 `println` 函数。

- [ ] **Step 4: 编写测试**

验证 `use std::io` 能正确导入并使用。

---

### Task 2: IO 模块

**目标:** 实现基础输入输出。

```
// std/io.th
fn println(msg: str) { /* native */ }
fn eprintln(msg: str) { /* native */ }
fn read_line() -> str { /* native */ }
fn read_file(path: str) -> str { /* native */ }
fn write_file(path: str, content: str) { /* native */ }
```

- [ ] 移除解释器中硬编码的 `println`/`eprintln`
- [ ] 实现为原生函数，在 std/io.th 中声明
- [ ] 测试：REPL 中 `use std::io; io.println("hello")`

---

### Task 3: Tensor 模块

**目标:** 将张量操作统一到标准库。

```
// std/tensor.th
fn tensor(data: [[f64]]) -> Tensor<f64, ..> { /* native */ }
fn zeros(shape: [i64]) -> Tensor<f64, ..> { /* native */ }
fn ones(shape: [i64]) -> Tensor<f64, ..> { /* native */ }
fn rand(shape: [i64]) -> Tensor<f64, ..> { /* native */ }
fn randn(shape: [i64]) -> Tensor<f64, ..> { /* native */ }
fn eye(n: i64) -> Tensor<f64, ..> { /* native */ }
fn arange(start: f64, end: f64, step: f64) -> Tensor<f64, ..> { /* native */ }
```

- [ ] 将 `tensor` / `rand` / `randn` 等从解释器硬编码迁移到原生函数表
- [ ] 张量方法（`.sum()`, `.relu()`, `.reshape()` 等）通过 trait impl 提供
- [ ] 测试：`use std::tensor; tensor.rand([3, 224, 224])`

---

### Task 4: Math 模块

```
// std/math.th
fn abs(x: f64) -> f64 { /* native */ }
fn sqrt(x: f64) -> f64 { /* native */ }
fn exp(x: f64) -> f64 { /* native */ }
fn log(x: f64) -> f64 { /* native */ }
fn sin(x: f64) -> f64 { /* native */ }
fn cos(x: f64) -> f64 { /* native */ }
fn tan(x: f64) -> f64 { /* native */ }
fn pow(base: f64, exp: f64) -> f64 { /* native */ }
```

- [ ] 注册 Rust 侧 `f64::sin` / `f64::cos` 等为原生函数
- [ ] 测试：`use std::math; math.sqrt(16.0)` → `4.0`

---

### Task 5: String 与 Collections 模块

```
// std/string.th
impl String {
    fn len(self) -> i64 { /* native */ }
    fn concat(self, other: str) -> str { /* native */ }
    fn split(self, delim: str) -> [str] { /* native */ }
    fn contains(self, substr: str) -> bool { /* native */ }
}

// std/collections.th
impl Array<T> {
    fn len(self) -> i64 { /* native */ }
    fn push(mut self, item: T) { /* native */ }
    fn pop(mut self) -> T { /* native */ }
}
```

---

### Task 6: Test 框架

```
// std/test.th
fn assert(cond: bool) { if !cond { eprintln("assertion failed") } }
fn assert_eq<T: Eq>(a: T, b: T) { assert(a == b) }
```

- [ ] 编写单元测试宏或函数
- [ ] 用 Tenth 自身的 test 框架测试标准库

---

### Task 7: 全量验收

- [ ] 解释器加载 prelude.th 后 `println("hello")` 可用
- [ ] `cargo test` 全部通过（含新的 std 测试）
- [ ] REPL 中 `use std::tensor; tensor.rand([2,2])` 正常工作
- [ ] Commit + push

---

## Phase 4 完成标准

- [ ] `std/` 目录下的标准库模块可被 `use` 导入
- [ ] 解释器中不再有硬编码的内置函数名
- [ ] IO / 张量 / 数学 三大核心模块有 Tenth 测试
- [ ] 42+ 项测试通过（现有 60 项 + 新增 std 测试）
