# Phase 3B: MLIR 原生编译 实施计划

> **✅ COMPLETED** — 2026-05-22  
> 交付物：MIR 定义与 HIR→MIR lowering (compile/mir.rs + compile/lower.rs)，C 代码生成器 (compile/cgen.rs)，Shape 推导引擎 (compile/shape.rs)，CLI compile 命令 (main.rs)。  
> 5 项 compile_test 全部通过。  
> 偏差：原计划用 inkwell/LLVM 直接生成原生二进制。实现改为 C 代码生成方案——MIR→C→调用系统 C 编译器（gcc/clang），降低了 LLVM 安装依赖。LLVM 直编留在远期计划。

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 打通 HIR → MIR → LIR → LLVM IR 四层编译管线，产出能编译 `.th` 文件到原生可执行文件的编译器。

**Architecture:** 新增编译后端。复用 Phase 3A 的类型系统（泛型/trait/所有权信息在编译期静态化）。前端（Lexer/Parser/HIR Lowering）不变。新增 MIR（中端优化 IR）、LIR（线性化 IR）、LLVM IR codegen。

**Tech Stack:** Rust 2024 edition, [`llvm-sys`](https://crates.io/crates/llvm-sys) 或 [`inkwell`](https://crates.io/crates/inkwell) （LLVM Rust 绑定）。

**前置条件:** Phase 3A 完成（类型系统完备）。

**注:** 此计划为远期规划，随着 Phase 3A 的进展可能需要调整。此处描述的是目标架构。

---

## 编译器管线

```
.th 源码
  │
  ▼
Lexer → Parser → AST           [Phase 1, 不变]
  │
  ▼
HIR Lowering + Type Check       [Phase 1/2/3A, 不变]
  │
  ▼
MIR Lowering                    [Phase 3B 新增]
  │  - 控制流图 (CFG) 构建
  │  - SSA 形式转换
  │  - 基础块划分
  │  - 常量折叠
  │
  ▼
LIR (Linear IR)                 [Phase 3B 新增]
  │  - 虚拟寄存器分配
  │  - 线性指令序列
  │  - 调用约定处理
  │
  ▼
LLVM IR Codegen                 [Phase 3B 新增]
  │  - 类型映射 (Tenth → LLVM)
  │  - 内存布局
  │  - 运行时链接
  │
  ▼
LLVM → 原生二进制 (.o / 可执行文件)
```

## 文件结构（Phase 3B 新增）

```
tenth/src/
├── compile/
│   ├── mod.rs           # 编译入口：compile_program()
│   ├── mir.rs           # MIR 定义与 lowering
│   ├── lir.rs           # LIR 定义与 lowering
│   ├── llvm_gen.rs      # LLVM IR 代码生成
│   └── shape.rs         # Shape 推导引擎
tenth/Cargo.toml         # 新增: inkwell 依赖
```

---

### Task 1: MIR（中端 IR）定义与 Lowering

**MIR 核心概念:**
```rust
struct MirFunction {
    name: String,
    basic_blocks: Vec<BasicBlock>,
    locals: Vec<Local>,
}

struct BasicBlock {
    id: usize,
    statements: Vec<Statement>,
    terminator: Terminator,
}

enum Statement {
    Assign(Place, Rvalue),
    StorageLive(usize),
    StorageDead(usize),
}

enum Rvalue {
    Use(Operand),
    BinaryOp(BinOp, Operand, Operand),
    UnaryOp(UnOp, Operand),
    Call(String, Vec<Operand>),
    Ref(Operand),
}

enum Terminator {
    Return(Option<Operand>),
    Goto(usize),
    If(Operand, usize, usize),
}
```

- [ ] MIR 数据结构定义
- [ ] HIR → MIR：控制流图构建（if/while/match → 基本块）
- [ ] 简单常量折叠优化
- [ ] 测试：basic block 划分正确性

---

### Task 2: LIR（线性 IR）与虚拟寄存器

**LIR 核心概念:**
```rust
struct LirFunction {
    name: String,
    instructions: Vec<LirInst>,
    num_vregs: usize,
}

enum LirInst {
    Mov(usize, Operand),
    BinaryOp(BinOp, usize, Operand, Operand),
    Call(usize, String, Vec<Operand>),
    Return(Operand),
    Load(usize, Operand),
    Store(Operand, Operand),
}
```

- [ ] MIR → LIR：基本块线性化
- [ ] 虚拟寄存器分配（简单线性扫描）
- [ ] 栈帧布局（local variables → stack slots）
- [ ] 测试：LIR 输出可读性

---

### Task 3: LLVM IR 代码生成（inkwell）

使用 `inkwell` crate 作为 LLVM 的 Rust 安全绑定。

- [ ] 添加 `inkwell` 依赖到 Cargo.toml
- [ ] 类型映射：Tenth Type → LLVM BasicType
  - `i32` → `i32`, `f64` → `double`, `bool` → `i1`
  - `struct Point { x: f64, y: f64 }` → `{ double, double }`
  - Tensor → 运行时库调用（暂不直接编译）
- [ ] 函数编译：LIR → LLVM Function
- [ ] 控制流编译：基本块 → LLVM BasicBlock
- [ ] 二进制输出：`compile_to_object()` → `.o` 文件

---

### Task 4: Shape 推导引擎

**职责:** 在编译期推断张量的符号维度，消除运行时 shape 检查。

- [ ] 符号维度表示：`Dim::Known(n)`, `Dim::Symbolic("N")`, `Dim::Unknown`
- [ ] 表达式 shape 推导：`(N×M) × (M×K) → N×K`
- [ ] 向后传播：从输出 shape 反推输入约束
- [ ] 与 LLVM codegen 集成：shape 参数 → runtime shape descriptor

---

### Task 5: 运行时库（libtenth）

编译后的二进制需要运行时支持：

- [ ] `tenth/runtime/` C 代码（libtenth）
  - 张量内存分配 / 释放
  - BLAS / 矩阵乘法（调用系统 BLAS 或内置实现）
  - `rand` / `randn` 随机数
- [ ] 链接运行时库到编译产物
- [ ] 测试：`compile("42 + 10")` → 运行输出 `52`

---

### Task 6: 编译驱动程序

- [ ] CLI 新增 `tenth compile input.th -o output` 命令
- [ ] 编译管道集成测试：端到端 .th → 可执行文件
- [ ] 错误处理：编译阶段的错误格式统一

---

### Task 7: 全量验收

- [ ] `tenth compile examples/hello.th -o hello && ./hello` 输出正确
- [ ] `tenth compile examples/matmul.th -o matmul && ./matmul` 精度验证
- [ ] 运行 `cargo test` 确保回归
- [ ] 提交

---

## Phase 3B 完成标准

- [ ] `.th` 源文件 → 原生可执行文件，端到端打通
- [ ] MIR/LIR/LLVM IR 三层中间表示齐全
- [ ] 基本张量运算（创建、四则运算、matmul）可用 LLVM 编译
- [ ] Shape 推导引擎能推断符号维度
- [ ] 运行时库提供内存管理和基本 BLAS
- [ ] 编译后的二进制精度与解释器一致（浮点误差容忍）