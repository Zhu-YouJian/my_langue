# Tenth 语言工具

> 版本: v0.3.0-pre | 字节码 VM + 树遍历 fallback | 自举验证通过

## 快速开始

```cmd
REM 安装到 PATH (以管理员运行)
install.bat

REM 或直接使用
tenth.bat run file.th
```

## 命令

| 命令 | 说明 |
|------|------|
| `tenth run file.th` | 解释执行 .th 文件 (VM优先, tree-walk兜底) |
| `tenth build file.th` | 编译 .th → .wasm |
| `tenth wasm file.th` | wasmi 加载执行 |
| `tenth` | 启动 REPL 交互环境 |
| `tenth --max-memory N` | REPL 内存限制 (MB) |

## 语言特性

| 特性 | 状态 |
|------|------|
| Lexer / Parser / HIR 类型系统 | ✅ |
| 张量 (ndarray) 运算 | ✅ |
| struct / enum / match | ✅ |
| trait / impl / 泛型 | ✅ |
| 引用 / 移动 / 借用检查 | ✅ |
| Vec / HashMap / String 标准库 | ✅ |
| 字节码 VM (33 指令) | ✅ |
| WASM 编译 (wasm-encoder) | ✅ |
| 自举 (Tenth 编译器编译自身) | ✅ |

## 标准库

见 `tenth/std/` — nn (Linear, ReLU, Sigmoid, MSE, BCE), optim (SGD, Adam)

## 示例

见 `Tenth实例/` — 21 个示例程序 (排序/查找/链表/神经网络/梯度下降等)

## 更多

- 语言参考: `docs/语言参考手册.md`
- 开发备忘: `MEMO.md`
