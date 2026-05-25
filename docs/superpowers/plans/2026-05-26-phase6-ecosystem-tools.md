# Phase 6: 生态与工具 实施计划

> 状态：📋 计划中 | 前置条件：Phase 5 AI 全栈完成
>
> 对应设计规格 §9 —「生态与工具」

**Goal:** 建立 Tenth 语言的完整开发者生态——包管理器、LSP 服务器、调试器、文档生成器，以及社区建设基础设施。

**Architecture:** 包管理器（tenthpm）用 Tenth 自举编写，集成到编译管线。LSP 服务器复用编译器的 HIR/类型信息。调试器通过解释器模式配合断点支持实现。

**Tech Stack:** Tenth（包管理器、CLI 工具），Rust（LSP 服务器性能关键路径），LSP Protocol

---

## 文件结构（Phase 6 新增）

```
tenth/
├── tools/
│   ├── tenthpm/             # 包管理器 (Tenth 实现)
│   │   ├── main.th
│   │   ├── registry.th     # 包注册中心客户端
│   │   ├── resolve.th      # 依赖解析 (版本约束求解)
│   │   ├── install.th      # 包下载与安装
│   │   └── lock.th         # 锁文件管理
│   ├── lsp/                # LSP 服务器 (Rust 实现)
│   │   ├── main.rs
│   │   ├── server.rs       # LSP 协议处理
│   │   ├── completion.rs   # 代码补全
│   │   ├── hover.rs        # 类型悬停
│   │   ├── goto_def.rs     # 跳转定义
│   │   ├── diagnostics.rs  # 实时诊断
│   │   └── formatting.rs   # 代码格式化
│   ├── debugger/           # 调试器
│   │   ├── main.th
│   │   └── breakpoint.th
│   └── docgen/             # 文档生成器
│       ├── main.th
│       └── markdown.th
├── registry/               # 包注册中心 (未来可独立仓库)
│   └── README.md
└── website/                # 官方网站
    └── ...
```

---

### Task 1: 包管理器 (tenthpm)

**目标:** 让 Tenth 项目能声明依赖并从注册中心下载。

- [ ] **包清单 (Tenth.toml)**

```
[package]
name = "my-model"
version = "0.1.0"
edition = "2026"

[dependencies]
std = "0.2"
nn = "0.1"
optim = "0.1"
```

- [ ] **依赖解析**

语义化版本约束求解（类似 Cargo）。生成锁文件（Tenth.lock）确保可重现构建。

- [ ] **注册中心**

中央包注册中心（类似 crates.io）。支持包发布：`tenthpm publish`。

- [ ] **CLI 命令**

```
tenthpm new my-project      # 脚手架
tenthpm build               # 编译 (调用 tenthc)
tenthpm test                # 测试
tenthpm run                 # 运行
tenthpm add nn              # 添加依赖
tenthpm publish             # 发布
```

---

### Task 2: LSP 服务器

**目标:** 提供 IDE 级开发体验（VS Code / Vim / Emacs 集成）。

- [ ] **实时诊断**

保存文件时即时反馈类型错误、借用冲突、未使用变量。复用编译器的 HIR lowering 和类型检查信息。

- [ ] **代码补全**

基于 HIR 的符号表提供变量、函数、方法、字段补全。trait 方法在 `.` 后自动列出。

- [ ] **类型悬停 (Hover)**

鼠标悬停在变量/表达式上时显示推断类型和文档注释。

- [ ] **跳转定义 (Go-to-Definition)**

点击符号跳转到其定义位置。基于 HIR 的 use-def 链。

- [ ] **代码格式化**

AST-aware 的格式化器（类似 rustfmt），统一 Tenth 代码风格。

- [ ] **自动导入**

自动插入 `use` 语句补全未导入的类型。

---

### Task 3: 调试器

**目标:** 支持断点、单步执行、变量查看。

- [ ] **解释器断点模式**

在树遍历解释器中插入断点检查。遇到断点时暂停执行并进入交互模式。

- [ ] **变量查看**

暂停时查看当前作用域中所有变量的值。

- [ ] **调用栈显示**

显示当前函数调用链。

- [ ] **REPL 内嵌调试**

在 REPL 中直接设置断点并调试代码片段。

---

### Task 4: 文档生成器

- [ ] **文档注释语法**

```
/// Computes the mean squared error between predictions and targets.
/// 
/// # Parameters
/// - `pred`: model predictions, shape [B, D]
/// - `target`: ground truth, shape [B, D]
///
/// # Returns
/// Scalar loss value
fn mse(pred: Tensor[f32, B, D], target: Tensor[f32, B, D]) -> Tensor[f32, []] { ... }
```

- [ ] **Markdown 生成**

从源码中的文档注释生成静态文档站点。自动交叉链接类型引用。

- [ ] **文档测试 (Doc-tests)**

文档注释中的代码示例自动作为测试运行（类似 Rust doc-tests）。

---

### Task 5: CI/CD 与基础设施

- [ ] **GitHub Actions 集成**

`tenthpm test` 和 `tenthpm build` 在 CI 中自动运行。

- [ ] **版本管理**

编译器版本与语言版本解耦。`edition = "2026"` 机制类似 Rust editions。

- [ ] **错误索引**

在线错误码查询（类似 `rustc --explain`）。

---

### Task 6: 社区建设

- [ ] **官网**（tenth-lang.org）：语言概述、快速开始、文档、博客
- [ ] **论坛 / Discord**：社区讨论与技术支持
- [ ] **RFC 流程**：语言变更提案的标准化流程
- [ ] **贡献指南**：为编译器/标准库/工具贡献代码的指南

---

### Task 7: 全量验收

- [ ] `tenthpm new && tenthpm build && tenthpm test` 端到端可用
- [ ] LSP 在 VS Code 中提供补全和诊断
- [ ] 调试器中可设置断点并查看变量
- [ ] 文档生成器产出完整的 API 文档

---

## Phase 6 完成标准

- [ ] 包管理器可安装依赖并构建项目
- [ ] LSP 提供补全、悬停、诊断、跳转定义
- [ ] 调试器支持断点和变量查看
- [ ] 文档生成器产出 Markdown API 文档
- [ ] 官网和社区基础设施上线
