# Tenth

**Tenth 是一门 AI 原生的通用编程语言**：张量与自动微分是语言内置能力，同时具备通用语言所需的完整设施——控制流、结构体/枚举/match、泛型、Trait、闭包、模块系统与包管理器。用 Tenth 写脚本、工具或服务时，不需要切换到另一门语言，也不需要引入外部 AI 框架。

> 代号 Tenth = Tensor + Zenith，意为「张量之巅」。
>
> [![Version: 1.0.0](https://img.shields.io/badge/Version-1.0.0-blue.svg)](RELEASE_NOTES.md)
> [![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
> [![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-2.1-4baaaa.svg)](CODE_OF_CONDUCT.md)
> [![Rust: 1.95+](https://img.shields.io/badge/Rust-1.95+-orange.svg)](https://www.rust-lang.org/)

## 快速开始

前置条件：Rust **1.95+**。所有 crate 依赖由 cargo 自动下载；环境细节（含 Windows 下 Rust 未加入 PATH 时的写法）见 `DEPS.md`。

```bash
# 构建（release）
cargo build --release --manifest-path tenth/Cargo.toml

# 运行 .th 文件
cargo run --release --manifest-path tenth/Cargo.toml -- run path/to/file.th

# 启动 REPL
cargo run --release --manifest-path tenth/Cargo.toml

# 运行测试
cargo test --manifest-path tenth/Cargo.toml
```

最小程序：

```tenth
fn main() {
    println("Hello, Tenth");
}
```

## 它解决的问题

AI 开发中最常见的故障模式是**错误到运行期才暴露**：矩阵维度不匹配要等到前向传播失败，内存不足要等到训练中途，梯度形状错误被静默修正——典型表现是 loss 不下降，却没有任何报错。Tenth 把这类错误前移到编译期，并在运行期保留校验：

| 防护层 | 机制 |
|--------|------|
| 编译期 shape 检查 | `Tensor[f64, 3, 4] @ Tensor[f64, 5, 6]` 在编译期报错 |
| 编译期内存与算力预估 | ≥ 1 GB 的张量、≥ 1 GFLOP 的矩阵乘法发出 warning |
| 运行期自动微分校验 | `backward()` 返回 `Result`，梯度 shape 不匹配时显式报错 |

## 示例

### 编译期 shape 检查

```tenth
fn main() {
    let a: Tensor[f64, 2, 3] = tensor[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
    let b: Tensor[f64, 3, 2] = tensor[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];

    let c = a.matmul(b);  // (2×3) @ (3×2) = (2×2)，编译通过

    // let bad = a.matmul(a);  // 编译期报错：(2×3) @ (2×3)，内侧维度 3 ≠ 2
}
```

### 自动微分：线性回归

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

        backward(loss);  // 返回 Result；梯度 shape 不匹配会显式报错

        stop_grad();
        w = w - lr * grad(w);
        b = b - lr * grad(b);
        epoch = epoch + 1;
    };

    println(w.sum());   // ≈ 2.0
    println(b.sum());   // ≈ 1.0
}
```

### 矩阵乘法与广播

```tenth
fn main() {
    let a = tensor[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];  // 2×3
    let b = tensor[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];  // 3×2
    let c = a.matmul(b);  // (2×3) @ (3×2) = (2×2)
    println(c.sum());      // 163
}
```

更多可运行实例见 `Tenth实例/`（每个实例含 `task.txt`）。

## 语言核心

- **张量为内置类型**：`Tensor[f64, M, K]`、`Tensor[f32, ..]`，维度是可编译期追踪的符号。
- **可微算子**：Add / Sub / Mul / Div / Neg / ReLU / MatMul / Transpose / Sum / Mean / Exp / Log / Sigmoid / Softmax / CrossEntropy / Dropout / Conv2D / BatchNorm / LayerNorm / GELU / Input，全部实现反向传播。
- **通用语言设施**：控制流、struct / enum / match、泛型、Trait（含 trait 对象）、闭包捕获、借用检查、模块与包管理。
- **双精度张量**：f32 与 f64；自动微分采用「前向 f32 + 反向 f64，梯度按参数 dtype 写回」。

## 自动微分

Tenth 内置张量级自动微分，由 7 个内置函数控制：

| 函数 | 作用 |
|------|------|
| `new_grad()` | 创建新的计算图并开启录制 |
| `param(tensor)` | 注册可训练参数（≥ 1 GB 时编译期发出梯度 OOM warning） |
| `backward(loss)` | 反向传播；返回 `Result`，梯度 shape 不匹配时显式报错 |
| `grad(param)` | 读取参数梯度 |
| `stop_grad()` | 关闭录制（参数更新阶段不录制） |
| `zero_grad()` | 清零所有参数梯度 |
| `cross_entropy(logits, target)` | 交叉熵损失（融合 softmax） |

`backward()` 全链路返回 `Result`：梯度 shape 不匹配时显式报错，而不是静默 squeeze 成某个能写回的形状。

## 标准库

```
tenth/std/
├── nn/            # linear, loss (MSE/L1/BCE), activations, dropout, batchnorm, conv2d,
│                  # embedding, attention, multihead_attention, layer_norm,
│                  # positional_encoding, feedforward, transformer_encoder_block
├── optim/         # SGD (vanilla/momentum/decay), Adam, AdaGrad, RMSProp
├── data/          # DataLoader, MNIST 加载器
├── init/          # xavier_uniform / xavier_normal / he_normal / he_uniform / zeros / constant
├── collections/   # iter (map/filter/reduce/zip/enumerate), collections (flat_map/partition)
├── string/        # join_lines / join_comma / repeat_sep / indent / word_wrap / capitalize
├── utils/         # 序列化 (save_model / load_model / save_checkpoint)、math (min/max/clamp)
├── fs/            # exists / is_file / is_dir / mkdir / list_dir / remove / copy
├── json/          # encode / decode / encode_pretty / load / save
├── toml/          # TOML 解析
├── cli/           # 命令行参数处理
├── logging/       # debug / info / warn / error + set_level
├── time/          # now / now_ms / date / datetime / sleep_ms / timer
├── random/        # rand_int / rand_float / choice / shuffle
├── math/          # 数学函数与常量
├── crypto/
├── distributed/
├── async.th autograd.th curry.th date.th duration.th env.th io.th
├── http.th net.th process.th regex.th
├── runtime.th     # 资源限制 (with_step_limit / with_timeout_ms)
└── prelude.th     # 可用项总目录
```

模块数、测试数与运行实例数以 `能力梳理/能力全梳理.md` §统计基线为唯一事实源。

## 工具链

| 可执行产物 | 用途 |
|------------|------|
| `tenth` | 编译器与运行时（`run` 子命令、REPL） |
| `tenthpm` | 包管理器：init / build / test / run / add / remove / list / clean / publish / install |
| `tenth-lsp` | 语言服务器：hover / completion / definition / references / rename / formatting 等 13 项能力 |
| `tenth-debug` | CLI 调试器：断点、单步、变量查看 |
| `tenth-prof` | 热点剖析器：函数级耗时与调用次数 |

## 执行管线与自举

源文件经 `Lexer → Parser → HIR` 后，可选四条执行路径：

| 路径 | 说明 |
|------|------|
| 字节码 VM | 默认路径 |
| 树遍历解释器 | 备选路径，`TENTH_NO_VM=1` 时启用 |
| Cranelift JIT | 热点函数编译，含标量参数 ABI |
| WASM 后端 | wasm-encoder 生成 + wasmi 执行 |

Tenth 编译器本身由 Tenth 编写（`tenthc/`），三条自举路径均通过验证：

| 路径 | 词法分析 | 语法分析 | 编译 | 状态 |
|------|---------|---------|------|------|
| A | Rust | Rust | `compile_host` | 通过（约 0.2 秒） |
| B | **Tenth** | **Tenth** | `compile_program` | 通过 |
| C | WASM | wasmi | `compile_host` | 通过（闭环） |

## 项目状态

- **版本**：v1.0.0（2026-08-04 发布）。API 已冻结，权威清单见 `docs/API冻结清单.md`；版本语义为 `PATCH` = 缺陷修复、`MINOR` = 兼容新增、`MAJOR` = 破坏性变更。
- **已知限制**：见 `RELEASE_NOTES.md` §五，逐项列出 1.0 门槛遗留项与绕行方式。
- **规模数据**：测试数、可运行实例数、标准库模块数以 `能力梳理/能力全梳理.md` §统计基线为唯一事实源。
- **后续方向**：编译期 shape 检查与静态分析规划见 `docs/shape-check-roadmap/`。
- **GPU 后端**：目前为 CUDA 代码生成脚手架（`tenth/src/compile/gpu/`），尚未接入运行时与真实设备。

## 从源码构建与发布产物

- **从源码构建**（推荐，任一平台）：`cargo build --release --manifest-path tenth/Cargo.toml`
- **GitHub Releases**（CI 三平台原生构建，tag `v1.0.0`）：下载 `tenth-1.0.0-<windows-x86_64.zip | linux-x86_64.tar.gz | macos-x86_64.tar.gz>`。解压后 `bin/` 含 5 个可执行产物，`std/` 为运行时必需的标准库，`docs/` 含手册、语言规范与 API 冻结清单。
- **校验**：`sha256sum -c SHA256SUMS.txt`（Windows 使用 `scripts/release/verify.ps1`）
- **发布说明**：`RELEASE_NOTES.md`；发布流程：`scripts/release/README.md`

## 文档

| 文档 | 内容 |
|------|------|
| `docs/语言参考手册.md` | 语言语法、类型系统、标准库 API（对外权威） |
| `docs/语言规范.md` | 语言规范（稳定契约，v1.0.0 冻结） |
| `docs/教程.md` | 入门与进阶教程 |
| `docs/API冻结清单.md` | API 冻结范围与 semver 承诺 |
| `RELEASE_NOTES.md` | 1.0 发布说明：里程碑、关键能力、已知限制、升级指引 |
| `CODE_WIKI.md` | 模块架构、编译管线、依赖关系 |
| `AUDIT.md` | 缺陷登记册、测试覆盖矩阵、架构债务 |
| `DEPS.md` | 环境配置、构建命令、依赖清单 |
| `SECURITY.md` | 威胁模型与沙箱选项 |
| `docs/性能基线.md` | 性能数据（唯一事实源） |

## License

MIT License，详见 [LICENSE](LICENSE)。
