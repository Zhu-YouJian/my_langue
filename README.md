# Tenth — 为 AI 研究而生的语言

> 代号 Tenth = Tensor + Zenith，意为「张量之巅」

## 现状

**v0.3.0-pre** — Rust 解释器 + 标准库。C 编译后端已移除（内存安全原因，详见 `SECURITY.md`）。

| 组件 | 状态 |
|------|------|
| Lexer / Parser / AST | ✅ |
| HIR + 类型推断 | ✅ |
| 树遍历解释器 | ✅ |
| 泛型函数 / 结构体 | ✅ |
| Trait 定义与实现 | ✅ |
| 引用 / 移动语义 | ✅ |
| 编译期借用检查 | ✅ |
| ~~MIR → C 编译~~ | ❌ 已移除 |
| Shape 推导引擎 | ✅ |
| REPL 交互环境 | ✅ |
| 内存护栏 (arena + limits) | ✅ |

## 快速开始

```bash
# 编译
cargo build --manifest-path tenth/Cargo.toml

# 运行 REPL
cargo run --manifest-path tenth/Cargo.toml

# 运行测试
cargo test --manifest-path tenth/Cargo.toml
```

## 路线图

| Phase | 内容 | 状态 |
|-------|------|------|
| Phase 1 | Bootstrap 编译器 | ✅ |
| Phase 2 | 解释器夯实 | ✅ |
| Phase 3A | 类型系统深化 | ✅ |
| ~~Phase 3B~~ | ~~编译后端 (C)~~ | ❌ 已移除 |
| Phase 4 | GPU 与性能 | 🚧 |
| Phase 5 | AI 全栈 | 🚧 |
| Phase 6 | 生态与工具 | 🚧 |
| Phase 7 | 核心标准库 | ✅ |
| Phase 8 | 自举编译器 (解释执行) | 🚧 |

详见 `docs/superpowers/plans/`。MEMO.md 记录跳过项备忘。

## 依赖

见 `DEPS.md`。