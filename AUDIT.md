# 项目总览与审计报告

> 日期：2026-05-27 | 版本：v0.3.0-pre | 86 测试全绿

---

## 一、项目全景

Tenth = Tensor + Zenith，一门为 AI 研究而生的编程语言。Rust 编写的 bootstrap 编译器 + Tenth 编写的自举编译器（进行中）。

### 目录地图

```
├── tenth/                    ← Rust bootstrap 编译器
│   ├── Cargo.toml            ← ndarray, rustyline, thiserror, rand
│   ├── src/
│   │   ├── main.rs           ← CLI: REPL / compile 命令
│   │   ├── lib.rs            ← 导出 7 个顶层模块
│   │   ├── error.rs          ← TenthError 统一错误类型
│   │   ├── lexer/            ← 词法分析 (token.rs + lexer.rs)
│   │   ├── parser/           ← 递归下降解析 (ast.rs + parser.rs)
│   │   ├── hir/              ← HIR + 类型推断 + 借用检查 (hir.rs + types.rs + lower.rs)
│   │   ├── runtime/          ← 解释器 (interpreter.rs + value.rs + tensor.rs + arena.rs + autodiff.rs)
│   │   ├── compile/          ← MIR→C 编译 (mir.rs + lower.rs + cgen.rs + optimize.rs + shape.rs* + docgen.rs*)
│   │   └── repl.rs           ← 交互环境
│   ├── tests/                ← 86 项测试 (12 文件)
│   ├── std/                  ← Tenth 标准库 (.th 源码: nn/, optim/)
│   └── target/               ← (gitignored)
├── tenthc/                   ← Tenth 自举编译器 (.th 源码)
│   ├── main.th               ← 入口
│   ├── lexer/token.th        ← TokenKind 枚举 (50+ 变体)
│   ├── lexer/lexer.th        ← 手写词法分析器
│   ├── parser/parser.th      ← 递归下降解析器 (arena AST)
│   ├── codegen/cgen.th       ← C 代码生成器
│   └── runtime.c             ← C 运行时 (Vec, I/O, 字符串)
├── tenthc_combined.th        ← 上述 .th 文件的拼接版 (43KB, 自举输入)
├── docs/                     ← 设计文档与实施计划
├── README.md / MEMO.md / DEPS.md
└── .gitignore                ← 已更新
```

*标记: `shape.rs` 和 `docgen.rs` 未被编译管线调用。

---

## 二、编译器管线

### 2.1 Rust bootstrap 编译器 (解释器路径)

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

### 2.2 Rust bootstrap 编译器 (C 编译路径)

```
源码 (.th)
  ↓ Lexer → Parser → HIR
  ↓ MirLowerer (compile/lower.rs)
MIR (mir.rs)
  ↓ [Optimize (optimize.rs)]
CGenerator (compile/cgen.rs)
  ↓
C 代码 → GCC → .exe
```

### 2.3 自举编译器 (Tenth → C)

```
tenthc_combined.th
  ↓ Rust bootstrap (compile 命令)
tenthc.c → GCC → tenthc.exe
  ↓ (理想状态)
tenthc.exe 编译 tenthc_combined.th → tenthc_v2.c → GCC → tenthc_v2.exe
  ↓ (验证)
diff tenthc.c tenthc_v2.c → 一致 = 自举闭环
```

---

## 三、测试矩阵

| 测试文件 | 数量 | 覆盖 |
|----------|------|------|
| `lexer_test.rs` | 6 | 整数/标识符/关键字/字符串/运算符/注释 |
| `parser_test.rs` | 5 | 字面量/二元表达式/函数定义/if/tensor |
| `compile_test.rs` | 6 | 算术/变量/if-else/函数/结构体/常量折叠 |
| `integration_test.rs` | 14 | 全管线: 算术/布尔/比较/函数/闭包/while/tensor |
| `enum_test.rs` | 5 | 枚举定义/字段/match/通配 |
| `generic_test.rs` | 5 | 泛型函数/泛型结构体/trait bound |
| `struct_test.rs` | 5 | 结构体/嵌套/impl/默认字段 |
| `trait_test.rs` | 4 | trait 定义/builtin bound/inherent impl |
| `module_test.rs` | 2 | mod/use |
| `ownership_test.rs` | 11 | 移动/借用/引用/解引用 (已放宽) |
| `stdlib_test.rs` | 8 | Vec/HashMap/String/Option/文件 I/O |
| `tenthc_test.rs` | 3 | token.th/lexer.th/pipeline 解析通过 |
| **总计** | **86** | |

---

## 四、死代码 & 未使用模块

| 模块 | 位置 | 说明 |
|------|------|------|
| `compile/shape.rs` | `tenth/src/compile/` | 张量形状推断引擎，编译管线未调用 |
| `compile/docgen.rs` | `tenth/src/compile/` | Markdown API 文档生成，无入口点 |
| `runtime/arena.rs` | `tenth/src/runtime/` | 池化分配器，解释器/Tensor 均未使用 |
| `runtime/autodiff.rs` | `tenth/src/runtime/` | 计算图自动微分，解释器未集成 |

这 4 个模块有完整实现和测试，但缺少管线入口。属于 Phase 4-5 的预留基础设施，暂保留。

---

## 五、已知缺陷与不完整

### 5.1 致命 (阻断自举)

| # | 位置 | 问题 |
|---|------|------|
| 1 | `tenthc/codegen/cgen.th:23-25` | **函数调用参数全部丢弃** — `if e.kind == "call"` 分支 `args = ""` 后直接返回，所有参数丢失 |
| 2 | `tenthc/lexer/lexer.th:98,105` | **字面量值不解析** — 整数和浮点数 token 的 value 硬编码为 0 |
| 3 | `tenthc/parser/parser.th:269` | **字段名不存储** — `parse_postfix` 的 `.field` 访问把字段名丢了 |
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
