# Tenth — 通用语言，AI 原生

> 代号 Tenth = Tensor + Zenith，意为「张量之巅」
>
> [![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
> [![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-2.1-4baaaa.svg)](CODE_OF_CONDUCT.md)
> [![Rust: 1.95+](https://img.shields.io/badge/Rust-1.95+-orange.svg)](https://www.rust-lang.org/)

Tenth 是一门**通用编程语言的超集**——你可以把它当作日常语言来写脚本、构建工具、开发服务，同时发现 AI 能力就在手边，无需切换语言或引入外部框架。

- **作为通用语言**：完整的控制流、数据结构、模块系统、错误处理、包管理
- **作为 AI 语言**：张量是内置类型，自动微分是一等公民，训练循环无需外部框架
- **不可替代性**：编译期 + 运行时双层 shape 防御，在 shape 安全上超越 PyTorch 与 JAX

这不是 Python + PyTorch 的替代品，而是**一种新的可能性**：一门语言同时覆盖通用编程和 AI 研究，且在类型安全上超越两者。

## 为什么需要 Tenth

主流 AI 开发的痛点是「**运行时才发现错误**」：shape 不匹配要等到前向传播崩溃、内存 OOM 要等到训练中途、梯度 shape 错误被静默 squeeze 掩盖（典型症状是 loss 不降反升却无报错）。Tenth 把这些错误前移到编译期，并在运行时兜底：

| 防御层 | 能力 | 等价物 |
|--------|------|--------|
| **编译期 shape 检查** | `Tensor[f64, 3, 4] @ Tensor[f64, 5, 6]` 编译期报错 | 优于 PyTorch（运行时崩溃） |
| **编译期内存/算力预估** | ≥1GB tensor 或 ≥1 GFLOP matmul 发 warning | 无竞品 |
| **运行时 autodiff shape 校验** | `backward()` 返回 `Result`，梯度 shape 不匹配显式报错 | JAX 都没做好 |

三层防御共同构成 Tenth 的「**AI 原生护城河**」——用编译期信息换开发者真实的时间。

## Features

**语言核心**
- 张量是一等类型：`Tensor[f64, M, K]` / `Tensor[f32, ..]`，符号维度编译期追踪
- 21 个可微算子：Add/Sub/Mul/Div/MatMul/Conv2D/BatchNorm/LayerNorm/GELU/Softmax/CrossEntropy 等
- 完整控制流 + struct/enum/match + 泛型 + Trait + 闭包捕获 + 借用检查
- f32 / f64 双精度张量，自动微分方案 B 天然支持（前向 f32 + 反向 f64 + 梯度按参数 dtype 写回）

**编译管线**
- 字节码 VM（45 指令，默认路径，~0.2s 自举）
- 树遍历解释器（fallback 路径）
- Cranelift JIT（热点编译）
- WASM 后端（wasm-encoder + wasmi 闭环）

**自举**
- Tenth 编译器由 Tenth 自身编写（`tenthc/`，7 个 `.th` 文件，5000+ 行）
- 三条自举路径全部通过验证（Rust 全栈 / Tenth 前端 + Rust 后端 / 全 WASM 闭环）

**工具链**
- `tenthpm` 包管理器：init/build/test/run/add/remove/list/clean/publish/install
- LSP 服务器：hover/completion/definition/references/rename/formatting 等 13 项能力
- GPU 后端脚手架：CudaKernel 模板 + Device 抽象 + 算子融合/并行分解

**标准库**：36 个源文件覆盖 nn/optim/data/init/collections/string/utils/fs/json/toml/cli/logging/time/random/math

**测试**：700+ 项测试，0 回归

## Quick Start

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

依赖：Rust ≥ 1.95（所有 crate 依赖自动下载，详见 `DEPS.md`）。

## Examples

### 张量运算 + 自动微分

```tenth
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

        backward(loss);  // backward 返回 Result，shape 错误会显式报错

        stop_grad();
        w = w - lr * grad(w);
        b = b - lr * grad(b);
        epoch = epoch + 1;
    };

    println(w.sum());   // ~2.0
    println(b.sum());   // ~1.0
}
```

### 编译期 Shape 检查

```tenth
fn main() {
    let a: Tensor[f64, 2, 3] = tensor[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
    let b: Tensor[f64, 3, 2] = tensor[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
    let c = a.matmul(b);  // (2×3) @ (3×2) = (2×2) ✅ 编译通过

    // let bad = a.matmul(a);  // ❌ 编译期报错：(2×3) @ (2×3) 内侧维度 3 ≠ 2
}
```

### 矩阵乘法 + 广播

```tenth
fn main() {
    let a = tensor[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];  // 2×3
    let b = tensor[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];  // 3×2
    let c = a.matmul(b);  // (2×3) @ (3×2) = (2×2)
    println(c.sum());      // => 163
}
```

更多示例见 `Tenth实例/` 目录（49 个）和 `tenth/std/` 标准库（36 个源文件）。

## 自动微分

Tenth 内置张量级自动微分，通过 7 个内置函数控制：

| 函数 | 作用 |
|------|------|
| `new_grad()` | 创建新计算图，开启录制 |
| `param(tensor)` | 注册可训练参数（≥1GB 时编译期发梯度 OOM warning） |
| `backward(loss)` | 反向传播，返回 `Result`，梯度 shape 不匹配时显式报错 |
| `grad(param)` | 读取参数梯度 |
| `stop_grad()` | 关闭录制（SGD 更新时不录） |
| `zero_grad()` | 清零所有参数梯度 |
| `cross_entropy(logits, target)` | 交叉熵损失（融合 softmax） |

支持的算子（21 个，全部有正确的 backward）：
`Add / Sub / Mul / Div / Neg / ReLU / MatMul / Transpose / Sum / Mean / Exp / Log / Sigmoid / Softmax / CrossEntropy / Dropout / Conv2D / BatchNorm / LayerNorm / GELU / Input`

**护城河 A：反向 shape 静态验证**。`backward()` 全链路返回 `Result`，消除 autodiff 中 5 处 silent squeeze 静默修正梯度 shape 的代码（`acc_grad` / `unbroadcast` / `matmul_2d` / MatMul 1D squeeze / Conv2D 零填充兜底全部改为报错）。这是 JAX 都没做好的能力——梯度 shape 错误会显式报错而非静默写入错误数据。

## 标准库

```
tenth/std/
├── nn/          ← linear, loss (MSE/L1/BCE), activations, dropout, batchnorm, conv2d, embedding, attention, multihead_attention, layer_norm, positional_encoding, feedforward, transformer_encoder_block
├── optim/       ← SGD (vanilla/momentum/decay), Adam, AdaGrad, RMSProp (全部可运行)
├── data/        ← DataLoader, MNIST 加载器
├── init/        ← xavier_uniform/xavier_normal/he_normal/he_uniform/zeros_init/constant_init
├── collections/ ← iter (map/filter/reduce/zip/enumerate 等), collections (flat_map/partition 等)
├── string/      ← join_lines/join_comma/repeat_sep/indent/word_wrap/capitalize 等
├── utils/       ← 序列化 (save_model/load_model/save_checkpoint), math (min/max/clamp 等)
├── fs/          ← 文件系统操作 (exists/is_file/is_dir/mkdir/list_dir/remove/copy 等)
├── json/        ← JSON 编解码 (encode/decode/encode_pretty/load/save)
├── toml/        ← TOML 解析
├── cli/         ← 命令行参数处理
├── logging/     ← 日志 (debug/info/warn/error + set_level)
├── time/        ← 时间工具 (now/now_ms/date/datetime/sleep_ms/timer)
├── random/      ← 随机数 (rand_int/rand_float/choice/shuffle 等)
├── math/        ← 数学函数与常量
├── runtime.th   ← 资源限制 (with_step_limit/with_timeout_ms)
└── prelude.th   ← 可用项总目录
```

## 自举

Tenth 编译器由 Tenth 自身编写（`tenthc/`），三条自举路径全部通过验证：

| 路径 | 词法分析 | 语法分析 | 编译 | 状态 |
|------|---------|---------|------|------|
| A | Rust | Rust | compile_host | ✅ 秒级 |
| B | **Tenth** | **Tenth** | compile_program | ✅ 已验证 |
| C | WASM | wasmi | compile_host | ✅ 闭环 |

自举管线执行时间 ~0.2s（VM 路径，无 fallback）。

## Documentation

| 文档 | 内容 |
|------|------|
| `CODE_WIKI.md` | 模块架构、编译管线、依赖关系 |
| `MEMO.md` | 逐版变更记录、已知限制演化、重大决策 |
| `能力梳理/能力全梳理.md` | 479 项能力的逐条完成状态（✅/⚠️/❌） |
| `docs/语言参考手册.md` | 语言语法、类型系统、标准库 API |
| `docs/shape-check-roadmap/` | Shape 检查战略规划与短期规划 |
| `AUDIT.md` | 缺陷登记册、测试覆盖矩阵、架构债务 |
| `DEPS.md` | 环境配置、构建命令、依赖清单 |
| `SECURITY.md` | 威胁模型与沙箱选项 |

## Roadmap

| 阶段 | 目标 | 状态 |
|------|------|------|
| **阶段 1：可用** | VM 张量方法 + 日常标准库 + tenthpm 基本可用 | 🚧 进行中 |
| **阶段 2：好用** | Cranelift JIT 编译热函数，标量性能接近 Go | 📋 规划中 |
| **阶段 3：不可替代** | 编译期 shape 检查 + 运行时 autodiff 校验 + 内存预估 | ✅ Phase 1+2+3 + 护城河 A/D 已实现 |

每个阶段独立可用——阶段 1 完成后即使不做 JIT，用户也能用 Tenth 写脚本和训练小模型；阶段 2 完成后即使不做 shape 检查，Tenth 也是一门性能不错的通用语言。

详见 `CODE_WIKI.md` §10 与 `MEMO.md`。

## License

MIT License，详见 [LICENSE](LICENSE)。
