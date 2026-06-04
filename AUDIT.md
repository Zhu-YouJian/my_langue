# 项目总览与审计报告

> 日期：2026-06-04 | 版本：v0.3.0-pre | C 编译后端已移除

---

## 一、项目全景

Tenth = Tensor + Zenith，一门为 AI 研究而生的编程语言。Rust 编写的 bootstrap 编译器 + Tenth 编写的自举编译器（通过 Rust 解释器执行）。

### 目录地图

```
├── tenth/                    ← Rust bootstrap 编译器
│   ├── Cargo.toml            ← ndarray, rustyline, thiserror, rand
│   ├── src/
│   │   ├── main.rs           ← CLI: REPL 入口
│   │   ├── lib.rs            ← 导出 6 个顶层模块
│   │   ├── error.rs          ← TenthError 统一错误类型
│   │   ├── lexer/            ← 词法分析 (token.rs + lexer.rs)
│   │   ├── parser/           ← 递归下降解析 (ast.rs + parser.rs)
│   │   ├── hir/              ← HIR + 类型推断 + 借用检查 (hir.rs + types.rs + lower.rs)
│   │   ├── runtime/          ← 解释器 (interpreter.rs + value.rs + tensor.rs + arena.rs + autodiff.rs + limits.rs)
│   │   └── repl.rs           ← 交互环境
│   ├── tests/                ← 测试
│   ├── std/                  ← Tenth 标准库 (.th 源码: nn/, optim/)
│   └── target/               ← (gitignored)
├── tenthc/                   ← Tenth 自举编译器 (.th 源码，通过解释器运行)
│   ├── main.th               ← 入口
│   ├── lexer/token.th        ← TokenKind 枚举 (50+ 变体)
│   ├── lexer/lexer.th        ← 手写词法分析器
│   └── parser/parser.th      ← 递归下降解析器 (arena AST)
├── docs/                     ← 设计文档与实施计划
├── README.md / MEMO.md / DEPS.md / SECURITY.md
└── .gitignore
```

---

## 二、编译器管线（仅解释器路径）

```
源码 (.th)
  ↓ Lexer (lexer.rs)
Token 流
  ↓ Parser (parser.rs)
AST (ast.rs)
  ↓ Lowerer (hir/lower.rs)
HIR + 类型推断 + 借用检查
  ↓ Interpreter (runtime/interpreter.rs)
运行时值 (Value) / 张量 (Tensor)
```

> ~~C 编译路径 (MIR → C → GCC → .exe) 已于 2026-06-04 移除。原因：生成的 C 代码无内存管理，导致系统级内存耗尽。详见 SECURITY.md。~~

---

## 三、测试矩阵

| 测试文件 | 数量 | 覆盖 |
|----------|------|------|
| `lexer_test.rs` | 6 | 整数/标识符/关键字/字符串/运算符/注释 |
| `parser_test.rs` | 5 | 字面量/二元表达式/函数定义/if/tensor |
| `integration_test.rs` | 14 | 全管线: 算术/布尔/比较/函数/闭包/while/tensor |
| `enum_test.rs` | 5 | 枚举定义/字段/match/通配 |
| `generic_test.rs` | 5 | 泛型函数/泛型结构体/trait bound |
| `struct_test.rs` | 5 | 结构体/嵌套/impl/默认字段 |
| `trait_test.rs` | 4 | trait 定义/builtin bound/inherent impl |
| `module_test.rs` | 2 | mod/use |
| `ownership_test.rs` | 11 | 移动/借用/引用/解引用 |
| `stdlib_test.rs` | 8 | Vec/HashMap/String/Option/文件 I/O |
| `memory_test.rs` | 17 | 内存护栏: arena/limits/计数器 |
| **总计** | **82** | |

---

## 四、已移除模块

| 模块 | 说明 |
|------|------|
| ~~`tenth/src/compile/` (7 文件)~~ | MIR→C 编译管线，因内存安全问题于 2026-06-04 移除 |
| ~~`tenthc/codegen/cgen.th`~~ | C 代码生成器（Tenth 编写），随上移除 |
| ~~`tenthc/runtime.c`~~ | C 运行时库，随上移除 |
| ~~`tenth/tests/compile_test.rs`~~ | C 编译测试 (6 项) |
| ~~`tenth/tests/tenthc_test.rs`~~ | 自举编译器测试 (3 项) |

保留但暂未集成：
| 模块 | 位置 | 说明 |
|------|------|------|
| `runtime/arena.rs` | 池化分配器 | 已接入 limits 追踪 |
| `runtime/autodiff.rs` | 计算图自动微分 | 解释器未集成 |

---

## 五、已知缺陷与不完整

### 5.1 自举编译器 Tenth 源码中的已知问题

| # | 位置 | 问题 |
|---|------|------|
| 1 | `tenthc/lexer/lexer.th:98,105` | **字面量值不解析** — 整数和浮点数 token 的 value 硬编码为 0 |
| 2 | `tenthc/parser/parser.th:269` | **字段名不存储** — `parse_postfix` 的 `.field` 访问把字段名丢了 |

> ~~原 #1 (cgen 函数调用参数丢弃) 随 codegen 移除而消除。~~

### 5.2 语言功能缺口
| 4 | `tenth/src/compile/lower.rs` | **Match pattern binding 未生成** — `TokenKind::Identifier(name: s) => { ... s ... }` 中 `s` 变量在 C 中未声明 |

### 5.2 高优

| # | 位置 | 问题 |
|---|------|------|
| 5 | `tenthc/runtime.c:50-54` | **HashMap 纯桩** — `HashMap_new` 返回 Vec，无 insert/get |
| 6 | `tenthc/codegen/cgen.th:178` | **结构体字段类型硬编码 int** — 忽略 `type_ann` |
| 7 | `tenthc/codegen/cgen.th:123` | **变量类型硬编码 int** — 忽略类型标注 |
| 8 | `tenthc/main.th:11` | **依赖 tenthc_combined.th** — 需在编译前手动拼接 |

### 5.3 中优

| # | 位置 | 问题 |
|---|------|------|
| 9 | `tenthc/parser/parser.th:631` | **无 for 循环解析** — lexer 识别 `for` 但 parser 未处理 |
| 10 | `tenthc/codegen/cgen.th:209` | **float_to_str 截断** — 浮点数被截为整数 |
| 11 | `tenthc/parser/parser.th:180` | **parse_unary 纯透传** — 一元运算在 parse_primary 内处理，此函数冗余 |
| 12 | `tenthc/runtime.c` | **Vec_push 未检查 realloc 失败** — 失败时泄漏旧内存 |

### 5.4 架构债务

| # | 问题 |
|---|------|
| 13 | **解释器与 C 编译路径分歧** — interpreter 支持 match on enum，但 MIR→C 的 match 降低仍有 bug (#4) |
| 14 | **borrow checker 双向放宽** — `check_borrow_shared` 和 `check_borrow_mut` 均跳过 ExclusiveRef/SharedRef 检查，为自举临时放宽 |
| 15 | **C 类型系统薄弱** — `c_type_name` 大量依赖启发式 (`void*` → 查 struct_fields → `int64_t`)，应为 MIR 值附加精确类型 |
| 16 | **Test 覆盖盲区** — `tenthc_test.rs` 只测解析，未测执行；tenthc 子编译器的正确性无自动化验证 |

---

## 六、灵感与改进方向

### 6.1 短期 (解锁自举)

- **修复致命缺陷 #1-#4** → 自举闭环可达
- **添加 `tenthc_combined.th` 自动生成** — build script 或 Makefile，而非手动拼接
- **`tenthc_test.rs` 加执行测试** — 编译简单 .th → C → GCC → 运行验证

### 6.2 中期 (质量加固)

- **消除解释器/C 编译路径分歧** — 将 match/enum 支持统一到 MIR 层
- **恢复 borrow checker** — 自举完成后移除去掉的检查
- **泛型类型传播到 C 代码** — `Type::Generic` 的信息当前在 C 输出中被丢弃
- **HashMap 真实现** — `runtime.c` 和解释器都需要

### 6.3 长期 (生态)

- **激活死模块** — shape.rs (张量形状优化)、docgen.rs (API 文档生成)、autodiff.rs (自动微分训练)
- **tenthpm 包管理器** — Phase 6 预留
- **LSP 服务器** — Phase 6 预留
- **CUDA 后端** — Phase 4 预留 (需安装 CUDA Toolkit)

### 6.4 过程改进

- **`.gitignore` 已更新** — 添加 `*.exe` / 构建产物排除
- **清理 19 个构建产物** — 15 个 .exe + 2 个空 .txt + tenthc.c + test_mini.c 已从 git 跟踪移除
- **MEMO.md 保持同步** — 作为动态状态文件，每次大改动后更新

---

## 七、清理记录

| 操作 | 数量 |
|------|------|
| 从 git 移除 .exe 文件 | 15 |
| 从 git 移除空 .txt | 2 |
| 从 git 移除构建 C 文件 | 2 (tenthc.c, test_mini.c) |
| 删除临时 .th | 1 (test_input.th) |
| 更新 .gitignore | 覆盖 *.exe, *.txt, tenthc.c, test_mini.c |

保留的历史 C 文件 (tenthc_v3.c ~ v9.c, tenthc_dbg5.c, tenthc_dbg6.c, tenthc_fix.c, tenthc_analyze.c, tenthc_chk.c, tenthc_out.c, tenthc_rust.c) 作为自举进化见证，暂时保留不删。

---

*文档由 2026-05-27 全项目审计生成，作为后续正式开工的参考基准。*
