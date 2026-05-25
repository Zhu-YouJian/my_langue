# Phase 6: 自举编译器 实施计划

> 状态：📋 计划中 | 前置条件：Phase 4 + Phase 5 完成

**Goal:** 用 Tenth 语言自身重写 Tenth 编译器，实现自举（self-hosting）。产出 `tenthc` —— 一个由 Tenth 编写、能编译自身源代码的编译器，彻底脱离 Rust 依赖。

**Architecture:** 分两步走：
1. **Phase 6A — Tenth 编写的编译器前端**：用 Tenth 重写 Lexer + Parser + HIR Lowering + Type Check，通过 Rust 版解释器运行
2. **Phase 6B — 脱靴（bootstrap）**：用 Rust 版编译器编译 Tenth 版编译器为 C 代码，再用 gcc/clang 编译 C 代码为可执行文件。此后 T3 编译器可编译自身。

**Tech Stack:** Tenth 语言自身（编译器源码），Rust（bootstrap 阶段的编译驱动），gcc/clang（C 代码编译）。

---

## 编译器演进路线

```
T1: Rust 编写 (Phase 1-5)
  │
  ▼ 解释执行
T2: Tenth 编写 (Phase 6)
  │
  ▼ T1 编译 T2 → C → gcc → 可执行文件
T3: 自举 (Phase 6 终点)
  │
  ▼ T3 编译自身 → C → gcc → 可执行文件 (循环闭合)
```

---

## 文件结构（Phase 6 新增）

```
tenth/
├── src/                     # T1: Rust 编译器 (不变，作为 bootstrap 驱动)
├── tenthc/                  # T2/T3: Tenth 自举编译器
│   ├── main.th              # 入口
│   ├── lexer/
│   │   ├── token.th         # Token 类型定义
│   │   └── lexer.th         # 词法分析器
│   ├── parser/
│   │   ├── ast.th           # AST 节点定义
│   │   └── parser.th        # 递归下降解析器
│   ├── hir/
│   │   ├── types.th         # 类型系统
│   │   ├── hir.th           # HIR 定义
│   │   └── lower.th         # AST→HIR lowering + 类型检查
│   ├── codegen/
│   │   └── cgen.th          # HIR→C 代码生成
│   └── std/                 # 标准库的 Tenth 实现
└── tests/
    └── bootstrap_test.rs    # Rust 侧 bootstrap 测试
```

---

### Task 1: 基础能力补齐

**目标:** 确保 Tenth 语言自身具备写编译器的能力。必要时在标准库中添加缺失特性。

- [ ] 递归函数调用（编译器天然递归）
- [ ] 字符串切片与模式匹配（用于 Lexer）
- [ ] Vec/List 数据结构（Token 流、AST 列表）
- [ ] HashMap 数据结构（符号表）
- [ ] Result/Option 枚举（错误处理）
- [ ] 文件 I/O（读源文件、写输出）

---

### Task 2: Token 与 Lexer（T2）

**目标:** 用 Tenth 重写词法分析器。

```
// token.th
enum TokenKind {
    IntLiteral(i64),
    FloatLiteral(f64),
    Identifier(str),
    Fn, Let, Mut, If, Else, // ... all keywords
    Plus, Minus, Star, Slash,
    Eof,
}

struct Token { kind: TokenKind, line: i64, col: i64 }
```

- [ ] 实现 `Lexer::new(source: str) -> Lexer`
- [ ] 实现 `Lexer::next_token() -> Token`
- [ ] 实现 `Lexer::tokenize() -> [Token]`
- [ ] 测试：与 Rust 版 Lexer 输出一致

---

### Task 3: Parser（T2）

**目标:** 用 Tenth 重写递归下降解析器。

- [ ] 实现 AST 节点类型（Expr、Stmt、Item、Program）
- [ ] 实现 Pratt 解析器或递归下降
- [ ] 实现所有语法构造：表达式、语句、函数、结构体、枚举、trait、模块
- [ ] 测试：与 Rust 版 Parser 输出一致

---

### Task 4: HIR Lowering + Type Check（T2）

**目标:** 用 Tenth 重写类型检查与 HIR 降级。

- [ ] 实现 HIR 类型定义
- [ ] 实现 AST→HIR 降级
- [ ] 实现类型推断与检查
- [ ] 实现借用检查（所有权跟踪）

---

### Task 5: C 代码生成（T2）

**目标:** 用 Tenth 重写 HIR→C 生成器。

- [ ] 实现类型映射：Tenth Type → C type
- [ ] 实现表达式编译
- [ ] 实现控制流编译
- [ ] 实现函数编译

---

### Task 6: Bootstrap — T1 编译 T2

**目标:** 用 Rust 编译器将 T2 源码编译为可执行文件。

```bash
# 步骤
cargo run -- compile tenthc/main.th -o tenthc.c
gcc tenthc.c -o tenthc
./tenthc compile tenthc/main.th -o tenthc_v2.c  # 自举循环！
diff tenthc.c tenthc_v2.c  # 应一致
```

---

### Task 7: 验证与庆祝

- [ ] T3 编译自身：输出与上一代一致
- [ ] T3 编译测试用例：与 T1 输出一致
- [ ] 全部 60+ 测试以 T3 为编译器跑通
- [ ] 庆祝自举完成 🎉

---

## Phase 6 完成标准

- [ ] `tenthc/` 目录下的 Tenth 编译器源码完成
- [ ] T1 能编译 T2 为 C 代码
- [ ] gcc 能将 C 代码编译为可执行文件
- [ ] T3 能编译自身，且输出与上一代一致（自举循环闭合）
- [ ] 全部现有测试在 T3 下通过
