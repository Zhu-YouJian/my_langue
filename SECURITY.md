# 安全审查报告：C 编译路径已移除

> **审查日期**：2026-06-04  
> **决议**：`tenth/src/compile/`、`tenthc/codegen/`、`tenthc/runtime.c` 已删除。  
> **当前状态**：✅ 项目仅保留 Rust 解释器路径，0 处 `unsafe`，内存安全由 Rust 保证。

---

## 移除原因

原 C 编译路径（`tenthc.c` + `tenthc/runtime.c`）存在以下致命问题：

| # | 问题 | 严重度 |
|---|------|--------|
| 1 | `str_add` 每次调用分配新内存，从不释放 | 🔴 致命 |
| 2 | AST 节点全部 `malloc`，零 `free`（12 处 malloc） | 🔴 致命 |
| 3 | `Vec` 不递归释放元素 | 🔴 致命 |
| 4 | `HashMap` 无析构函数 | 🟠 高危 |
| 5 | `Vec_push` 的 `realloc` 未检查 NULL | 🟡 中等 |

详细分析见本文档历史版本（git log -- SECURITY.md）。

---

## Rust 侧评估

| 项目 | 结果 |
|------|------|
| `unsafe` 代码块 | **0 处** |
| 内存泄漏 | 无 — Rust 所有权系统保证 |
| 张量运算 | `ndarray` (成熟库，安全) |
| 内存护栏 | `limits.rs` + `arena.rs` — 全局原子计数器 + 作用域回滚 |

---

## 当前安全态势

✅ **安全**：项目仅依赖 Rust 解释器路径执行。所有 Tenth 代码（包括自举编译器 `tenthc/`）通过 Rust 解释器运行，内存安全由 Rust 类型系统和所有权模型保证。
