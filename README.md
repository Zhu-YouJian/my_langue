# Tenth — 为 AI 研究而生的语言

> 代号 Tenth = Tensor + Zenith，意为「张量之巅」

## 现状

**v0.2.0** — Phase 1-6 全部完成，86 项测试全部通过。Phase 7 标准库完工，Phase 8 自举各层补全完毕，4 项语言阻塞全部解决。

| 组件 | 状态 |
|------|------|
| Lexer / Parser / AST | ✅ |
| HIR + 类型推断 | ✅ |
| 树遍历解释器 | ✅ |
| 泛型函数 / 结构体 | ✅ |
| Trait 定义与实现 | ✅ |
| 引用 / 移动语义 | ✅ |
| 编译期借用检查 | ✅ |
| MIR → C 编译 | ✅ |
| Shape 推导引擎 | ✅ |
| REPL 交互环境 | ✅ |

## 快速开始

```bash
# 编译
cargo build --manifest-path tenth/Cargo.toml

# 运行 REPL
cargo run --manifest-path tenth/Cargo.toml

# 运行测试 (86 项)
cargo test --manifest-path tenth/Cargo.toml

# 编译 .th 文件为 C 代码
cargo run --manifest-path tenth/Cargo.toml -- compile input.th -o output.c
```

## 路线图

| Phase | 内容 | 状态 |
|-------|------|------|
| Phase 1 | Bootstrap 编译器 | ✅ |
| Phase 2 | 解释器夯实 | ✅ |
| Phase 3A | 类型系统深化 | ✅ |
| Phase 3B | 编译后端 | ✅ |
| Phase 4 | GPU 与性能 | 🚧 部分完成 |
| Phase 5 | AI 全栈 | 🚧 部分完成 |
| Phase 6 | 生态与工具 | 🚧 部分完成 |
| **Phase 7** | **核心标准库** | **✅** |
| **Phase 8** | **自举编译器** | **🚧 各层补全完毕** |
| **Phase 9** | **v0.3.0 收尾** | **📋** |

详见 `docs/superpowers/plans/`。MEMO.md 记录跳过项备忘。

## 依赖

见 `DEPS.md`。