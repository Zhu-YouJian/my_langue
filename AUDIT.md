# 项目总览与审计报告

> 日期：2026-06-10 | 版本：v0.3.0 | 自举完成 | 83 项测试全过（共 84 项/13 文件，1 项忽略）

---

## 一、项目全景

Tenth = Tensor + Zenith，一门为 AI 研究而生的编程语言。Rust 编写的 bootstrap 编译器 + Tenth 编写的自举编译器 + 字节码 VM + WASM 编译。

### 目录地图

```
├── tenth/                    ← Rust bootstrap 编译器
│   ├── Cargo.toml            ← ndarray, rustyline, thiserror, rand, wasm-encoder, wasmi
│   ├── src/
│   │   ├── main.rs           ← CLI: REPL / run / build / wasm
│   │   ├── lib.rs            ← 导出 7 个顶层模块
│   │   ├── error.rs          ← TenthError 统一错误类型
│   │   ├── lexer/            ← 词法分析 (token.rs + lexer.rs)
│   │   ├── parser/           ← 递归下降解析 (ast.rs + parser.rs)
│   │   ├── hir/              ← HIR + 类型推断 + 借用检查 (hir.rs + types.rs + lower.rs)
│   │   ├── compile/          ← WASM 编译 + 字节码编译 (wasm.rs + bytecode.rs + bridge.rs)
│   │   ├── runtime/          ← 解释器 + VM + 值系统 (interpreter.rs + vm.rs + value.rs + tensor.rs + arena.rs + autodiff.rs + limits.rs)
│   │   └── repl.rs           ← 交互环境
│   ├── tests/                ← 测试 (13 文件, 84 项 — 83 激活 + 1 忽略)
│   ├── std/                  ← Tenth 标准库 (.th 源码: nn/, optim/)
│   └── target/               ← (gitignored)
├── tenthc/                   ← Tenth 自举编译器 (.th 源码, 自举验证通过)
│   ├── main.th               ← 入口 (编排脚本, ~500B)
│   ├── lexer/token.th        ← TokenKind 枚举 (50+ 变体)
│   ├── lexer/lexer.th        ← O(1) 源切片词法分析器
│   └── parser/parser.th      ← 递归下降解析器 (method_call 支持)
├── docs/                     ← 语言参考手册 + 实施计划
├── Tenth实例/                ← 21 个语言示例程序
├── README.md / MEMO.md / DEPS.md / SECURITY.md / AUDIT.md
└── .gitignore
```

---

## 二、编译器管线

### 路径 A：VM 字节码执行（默认）
```
源码 (.th) → Lexer → Parser → AST → Lowerer → HIR
  → BytecodeCompiler → Chunk (字节码)
  → Vm::run() → 运行时值
  (不支持的特性自动回退到路径 C)
```

### 路径 B：WASM 编译
```
源码 (.th) → Lexer → Parser → AST → Lowerer → HIR
  → WasmCompiler → .wasm 文件
  → wasmi 加载执行
```

### 路径 C：树遍历解释器（兼容性保障）
```
源码 (.th) → Lexer → Parser → AST → Lowerer → HIR
  → Interpreter (tree-walk) → 运行时值
```

> ~~C 编译路径 (MIR → C → GCC → .exe)~~ 已于 2026-06-04 移除。原因：生成的 C 代码无内存管理，详见 SECURITY.md。

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
| `selfhost_verify.rs` | 1 | WASM 自举闭环验证 |
| `three_stage.rs` | 1 | 三段式自举（忽略 — wasmi 慢） |
| **总计** | **84** | 83 激活 + 1 忽略 |

---

## 四、自举状态

Tenth 编译器由 Tenth 自身编写，三路径验证通过：

| 路径 | 词法 | 语法 | 编译 | 速度 | 验证 |
|------|------|------|------|------|------|
| A (VM) | Rust | Rust | compile_host | 秒级 | 36 函数 → WASM ✅ |
| B (真正) | Tenth | Tenth | compile_program | ~30s | 14 tokens → WASM ✅ |
| C (wasmi) | WASM | wasmi | 内嵌编译 | 秒级 | 36 函数闭环 ✅ |

---

## 五、已移除 / 已修复

| 项目 | 状态 |
|------|------|
| ~~C 编译管线~~ | ❌ 已移除 (2026-06-04) |
| ~~tenthc lexer 字面量硬编码 0~~ | ✅ 已修复 (Token 新增 fval) |
| ~~tenthc parser method_call 丢失 receiver~~ | ✅ 已修复 (Dot+LParen 产生 method_call) |
| ~~VM chunk clone 内存泄漏~~ | ✅ 已修复 (chunk_idx 索引引用) |
| ~~VM StoreField/code 切换死循环~~ | ✅ 已修复 (CallN/Ret 同步更新 code/strings) |
| `runtime/autodiff.rs` | ⚠️ 标量级可用，未集成解释器 |
| Lowerer 大文件性能 | ⚠️ 实际不慢 (release 39 函数 <0.01s)，之前误判为瓶颈 |

---

## 六、已知限制

| # | 问题 | 影响 |
|---|------|------|
| 1 | VM 不支持字符串切片/闭包/match/for | 自动回退到树遍历解释器 |
| 2 | 树遍历解释器大文件慢 (debug build) | release build 即解决 |
| 3 | WASM codegen 个别边界情况 | wasmi 执行偶有 type mismatch |
| 4 | 无 GPU 后端 | Phase 4 待 CUDA 环境就绪 |

---

## 七、已知缺陷与不完整（历史参考 — C 后端已移除）

### 5.1 自举编译器 Tenth 源码中的已知问题

| # | 位置 | 问题 |
|---|------|------|
| 1 | `tenthc/lexer/lexer.th:98,105` | **字面量值不解析** — 整数和浮点数 token 的 value 硬编码为 0 |
| 2 | `tenthc/parser/parser.th:269` | **字段名不存储** — `parse_postfix` 的 `.field` 访问把字段名丢了 |

> ~~原 #1 (cgen 函数调用参数丢弃) 随 codegen 移除而消除。~~

### 5.2 语言功能缺口
| 3 | `tenth/src/compile/lower.rs` | **Match pattern binding 未生成**  |

> ~~以下条目 (#5-#8) 引用已删除的 C 代码生成文件 (`tenthc/codegen/`, `tenthc/runtime.c`)，保留作为历史记录。~~

### 5.2 高优（已废弃 — C 后端移除）

| # | 位置 | 问题 |
|---|------|------|
| 5 | ~~`tenthc/runtime.c:50-54`~~ | ~~HashMap 纯桩~~ — 文件已删除 |
| 6 | ~~`tenthc/codegen/cgen.th:178`~~ | ~~结构体字段类型硬编码 int~~ — 文件已删除 |
| 7 | ~~`tenthc/codegen/cgen.th:123`~~ | ~~变量类型硬编码 int~~ — 文件已删除 |
| 8 | `tenthc/main.th:11` | **依赖 tenthc_combined.th** — 需在编译前手动拼接 |

### 5.3 中优

| # | 位置 | 问题 |
|---|------|------|
| 9 | `tenthc/parser/parser.th:631` | **无 for 循环解析** — lexer 识别 `for` 但 parser 未处理 |
| 10 | ~~`tenthc/codegen/cgen.th:209`~~ | ~~float_to_str 截断~~ — 文件已删除 |
| 11 | `tenthc/parser/parser.th:180` | **parse_unary 纯透传** — 一元运算在 parse_primary 内处理，此函数冗余 |
| 12 | ~~`tenthc/runtime.c`~~ | ~~Vec_push 未检查 realloc~~ — 文件已删除 |

### 5.4 架构债务

| # | 问题 |
|---|------|
| 13 | ~~解释器与 C 编译路径分歧~~ — C 后端已移除，不再适用 |
| 14 | **borrow checker 双向放宽** — `check_borrow_shared` 和 `check_borrow_mut` 均跳过 ExclusiveRef/SharedRef 检查，为自举临时放宽 |
| 15 | ~~C 类型系统薄弱~~ — C 后端已移除，不再适用 |
| 16 | **Test 覆盖盲区** — `tenthc_test.rs` 只测解析，未测执行；tenthc 子编译器的正确性无自动化验证 |

---

## 六、灵感与改进方向

### 6.1 短期 (解锁自举)

- **修复致命缺陷 #1-#4** → 自举闭环可达
- **添加 `tenthc_combined.th` 自动生成** — build script 或 Makefile，而非手动拼接
- **`tenthc_test.rs` 加执行测试** — 改为 VM/WASM 路径验证

### 6.2 中期 (质量加固)

- **VM 补全** — closure/generic call/match 仍偶有 fallback
- **恢复 borrow checker** — 自举完成后移除去掉的检查
- **WASM Host import 真实现** — Vec/String 在 WASM 模块中以占位形式存在

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
