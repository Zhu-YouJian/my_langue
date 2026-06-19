# Tenth — 通用语言，AI 原生

> 代号 Tenth = Tensor + Zenith，意为「张量之巅」
>
> [![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
> [![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-2.1-4baaaa.svg)](CODE_OF_CONDUCT.md)

## 最终目标

Tenth 的定位是**通用编程语言的超集**——用户可以把它当作日常编程语言来写脚本、构建工具、开发服务，同时发现 AI 能力就在手边，无需切换语言或引入外部框架。

具体而言：

- **作为通用语言**：Tenth 提供完整的控制流、数据结构、模块系统、错误处理、包管理，足以胜任日常编程任务
- **作为 AI 语言**：张量是内置类型，自动微分是一等公民，训练循环无需外部框架
- **不可替代性**：编译期 shape 检查——`Tensor[f64, 3, 4] @ Tensor[f64, 5, 6]` 在编译期报错，而非运行时崩溃

这意味着 Tenth 不是 Python + PyTorch 的替代品，而是**一种新的可能性**：一门语言同时覆盖通用编程和 AI 研究，且在类型安全上超越两者。

### 演进路线

| 阶段 | 目标 | 核心指标 | 状态 |
|------|------|----------|------|
| **阶段 1：可用** | 补全 VM 张量方法 + 日常标准库 + tenthpm 基本可用 | 能写日常脚本，能跑小模型训练 | 🚧 进行中 |
| **阶段 2：好用** | Cranelift JIT 编译热函数 | 标量/控制流性能接近 Go（慢 5-10x C） | 📋 规划中 |
| **阶段 3：不可替代** | 编译期张量 shape 检查 | shape 不匹配在编译期报错，而非运行时 | 📋 规划中 |

每个阶段独立可用——阶段 1 完成后即使不做 JIT，用户也能用 Tenth 写脚本和训练小模型；阶段 2 完成后即使不做 shape 检查，Tenth 也是一门性能不错的通用语言。

## 现状

**v0.3.3** — 字节码 VM (45 指令) + 自举编译器 (Tenth 全链路) + WASM 输出 + **张量级自动微分** + **闭包捕获** + **文件级导入** + **GPU 脚手架** + **tenthpm 包管理器** + **LSP 服务器**。349 项测试通过（共 350 项，1 项忽略）。

| 组件 | 状态 |
|------|------|
| Lexer / Parser / AST | ✅ |
| HIR + 类型推断 + 借用检查 | ✅ |
| 树遍历解释器 | ✅ (VM fallback) |
| **字节码 VM（栈式，45 指令）** | ✅ **默认路径**（含 MakeTensor/MakeClosure） |
| 泛型函数 / 结构体 | ✅ |
| Trait 定义与实现 | ✅ |
| 引用 / 移动语义 | ✅ |
| struct / enum / match | ✅ **VM 全支持** |
| ~~MIR → C 编译~~ | ❌ 已移除 |
| REPL 交互环境 | ✅ 多行输入支持 |
| 内存护栏 (arena + limits) | ✅ |
| WASM 编译 (wasm-encoder + wasmi) | ✅ |
| **张量级自动微分 (21 算子)** | ✅ **backward 全链路** |
| **张量间运算 (matmul/广播/转置)** | ✅ |
| **Conv2D / Dropout / BatchNorm** | ✅ |
| Vec / HashMap / String 标准库 | ✅ **pop/split/trim 等 10+ 方法** |
| **自举编译器 (Tenth 编写，全链路)** | ✅ **~0.2s** |
| **WASM import 输出** | ✅ wasmi 验证通过 |
| **VM for-in 循环** | ✅ **Range/Vec 迭代编译** |
| **VM 闭包调用** | ✅ **MakeClosure + FnRef 全局查找** |
| **VM 字符串切片** | ✅ **SliceStr + Range 索引解析** |
| **严格借用检查** | ✅ **check_borrow_shared/check_borrow_mut 恢复** |
| **闭包捕获环境变量** | ✅ **free_vars_in() 自动分析** |
| **文件级导入（use 自动搜索 std/）** | ✅ **search_paths + try_import_file()** |
| **错误信息增强（源码位置）** | ✅ **span 信息** |
| **块注释 /* */** | ✅ 支持嵌套 |
| **GPU 后端脚手架** | ✅ CudaKernel 模板 + Device 抽象 + 算子融合/并行分解 |
| **tenthpm 包管理器** | ✅ init/build/test/run/add/remove/list/clean/publish/install |
| **LSP 服务器** | ✅ 文档同步/diagnostics/hover/completion/definition/documentSymbol/references/rename/signatureHelp/foldingRange/semanticTokens/formatting |
| **结构体字段默认值 (..)** | ✅ 全管线支持 |
| **泛型返回类型** | ✅ Vec<Token> / HashMap<str, Vec<i64>> |
| **枚举元组变体** | ✅ Some(T) 构造 + match 绑定 |

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

更多示例见 `Tenth实例/` 目录（33 个）和 `tenth/std/` 标准库（16 个文件）。

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

支持的算子（21 个，全部有正确的 backward）：
`Add / Sub / Mul / Div / Neg / ReLU / MatMul / Transpose / Sum / Mean / Exp / Log / Sigmoid / Softmax / CrossEntropy / Dropout / Conv2D / BatchNorm / LayerNorm / GELU / Input`

## 标准库

```
tenth/std/
├── nn/          ← linear, loss (MSE/L1/BCE), activations, dropout, batchnorm, conv2d, embedding, attention, multihead_attention, layer_norm, positional_encoding, feedforward, transformer
├── optim/       ← SGD (vanilla/momentum/decay), Adam, AdaGrad, RMSProp (全部可运行)
├── data/        ← DataLoader (new/has_next/next_batch/reset/num_batches)
├── init/        ← xavier_uniform/xavier_normal/he_normal/he_uniform/zeros_init/constant_init
├── collections/ ← iter (map/filter/reduce/zip/enumerate 等), collections (flat_map/partition 等)
├── string/      ← join_lines/join_comma/repeat_sep/indent/word_wrap/capitalize 等
├── utils/       ← 序列化 (save_model/load_model/save_checkpoint), math (min/max/clamp 等)
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
| Phase 4 | GPU 与性能 | 🚧 脚手架就绪 |
| Phase 5 | AI 全栈 | 🚧 |
| Phase 6 | 生态与工具 | 🚧 tenthpm+LSP 脚手架就绪 |
| Phase 7 | 核心标准库 | ✅ |
| Phase 8 | 自举编译器 | ✅ |

详见 `docs/语言参考手册.md` 和 MEMO.md。

## 依赖

编译只需 Rust（≥1.95），所有 crate 依赖自动下载。详见 `DEPS.md`。
