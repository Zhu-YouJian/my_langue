# Tenth — 为 AI 研究而生的语言

> 代号 Tenth = Tensor + Zenith，意为「张量之巅」
>
> [![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
> [![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-2.1-4baaaa.svg)](CODE_OF_CONDUCT.md)

## 现状

**v0.3.0** — 字节码 VM (41 指令) + 自举编译器 (Tenth 全链路) + WASM 输出 + **张量级自动微分**。88 项测试全过（共 89 项，1 项忽略）。

| 组件 | 状态 |
|------|------|
| Lexer / Parser / AST | ✅ |
| HIR + 类型推断 + 借用检查 | ✅ |
| 树遍历解释器 | ✅ (VM fallback) |
| **字节码 VM（栈式，43 指令）** | ✅ **默认路径** |
| 泛型函数 / 结构体 | ✅ |
| Trait 定义与实现 | ✅ |
| 引用 / 移动语义 | ✅ |
| struct / enum / match | ✅ **VM 全支持** |
| ~~MIR → C 编译~~ | ❌ 已移除 |
| REPL 交互环境 | ✅ 多行输入支持 |
| 内存护栏 (arena + limits) | ✅ |
| WASM 编译 (wasm-encoder + wasmi) | ✅ |
| **张量级自动微分 (19 算子)** | ✅ **backward 全链路** |
| **张量间运算 (matmul/广播/转置)** | ✅ |
| **Conv2D / Dropout / BatchNorm** | ✅ |
| Vec / HashMap / String 标准库 | ✅ **pop/split/trim 等 10+ 方法** |
| **自举编译器 (Tenth 编写，全链路)** | ✅ **~0.2s** |
| **WASM import 输出** | ✅ wasmi 验证通过 |
| **块注释 /* */** | ✅ 支持嵌套 |

## 快速开始

```bash
# 编译（release 模式，推荐）
cargo build --release --manifest-path tenth/Cargo.toml

# 运行 REPL
cargo run --release --manifest-path tenth/Cargo.toml

# 运行 .th 文件
cargo run --release --manifest-path tenth/Cargo.toml run path/to/file.th

# 运行测试
cargo test --manifest-path tenth/Cargo.toml
```

## 语言示例

```tenth
// 张量运算 + 自动微分
fn main() {
    let x = tensor[[1.0, 2.0, 3.0, 4.0]];
    let target = tensor[[3.0, 5.0, 7.0, 9.0]];

    let mut w = tensor[[0.0]];
    let mut b = tensor[[0.0]];
    let lr = 0.02;
    let mut epoch = 0;

    while epoch < 500 {
        new_grad();
        zero_grad();
        w = param(w);
        b = param(b);

        let pred = w * x + b;
        let loss = ((pred - target) * (pred - target)).mean();

        backward(loss);

        stop_grad();
        w = w - lr * grad(w);
        b = b - lr * grad(b);
        epoch = epoch + 1;
    };

    println(w.sum());   // ~2.0
    println(b.sum());   // ~1.0
}

// 矩阵乘法 + 广播
fn main() {
    let a = tensor[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];  // 2×3
    let b = tensor[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];  // 3×2
    let c = a.matmul(b);  // (2×3) @ (3×2) = (2×2)
    println(c.sum());      // => 163
}

// 神经网络（XOR）
fn main() {
    let x = tensor[[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]];
    let target = tensor[[0.0], [1.0], [1.0], [0.0]];

    let mut w1 = randn(2, 4);
    let mut w2 = randn(4, 1);
    // ... training loop with backward(loss) ...
    // Converges to [0.03, 0.94, 0.93, 0.08]
}
```

更多示例见 `Tenth实例/` 目录（25 个）和 `tenth/std/` 标准库（16 个文件）。

## 自动微分

Tenth 内置张量级自动微分，通过 7 个内置函数控制：

| 函数 | 作用 |
|------|------|
| `new_grad()` | 创建新计算图，开启录制 |
| `param(tensor)` | 注册可训练参数 |
| `backward(loss)` | 反向传播，梯度写入 `param` 的 `.grad` |
| `grad(param)` | 读取参数梯度 |
| `stop_grad()` | 关闭录制（SGD 更新时不录） |
| `zero_grad()` | 清零所有参数梯度 |
| `cross_entropy(logits, target)` | 交叉熵损失（融合 softmax） |

支持的算子（19 个，全部有正确的 backward）：
`Add / Sub / Mul / Div / Neg / ReLU / MatMul / Transpose / Sum / Mean / Exp / Log / Sigmoid / Softmax / CrossEntropy / Dropout / Conv2D / BatchNorm / Input`

## 标准库

```
tenth/std/
├── nn/          ← linear, loss (MSE/L1/BCE), activations, dropout
├── optim/       ← SGD (vanilla/momentum/decay), Adam, AdaGrad, RMSProp
├── data/        ← DataLoader (规划中)
├── init/        ← 初始化指南
├── utils/       ← 序列化 (规划中)
├── math/        ← 数学函数参考
└── prelude.th   ← 可用项总目录
```

## 自举

Tenth 编译器由 Tenth 自身编写（`tenthc/`），经自举管线验证：

| 路径 | 词法分析 | 语法分析 | 编译 | 状态 |
|------|---------|---------|------|------|
| A | Rust | Rust | compile_host | ✅ 秒级 |
| B | **Tenth** | **Tenth** | compile_program | ✅ 已验证 |
| C | WASM | wasmi | compile_host | ✅ 闭环 |

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
| Phase 8 | 自举编译器 | ✅ |

详见 `docs/语言参考手册.md` 和 MEMO.md。

## 依赖

编译只需 Rust（≥1.95），所有 crate 依赖自动下载。详见 `DEPS.md`。
