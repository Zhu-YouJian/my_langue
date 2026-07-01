# Tenth 能力-场景映射深度调研

> 版本：v1.0 | 日期：2026-07-01
> 目的：评估 Tenth 的"AI 原生"能力在非 AI 场景下的扩展价值
> 关联文档：[战略规划.md](战略规划.md)（护城河 A/D/F）、[综合分析.md](综合分析.md)

---

## 一、调研背景与动机

### 1.1 战略命题

Tenth 定位为"AI 原生通用编程语言"，但若其核心创新特性（autograd、编译期 shape 检查、内存预估）**仅服务于 AI 任务**，本质上与 PyTorch/JAX + Python 调包无异，甚至因额外的语言运行时而更占安装空间。

**核心问题**：Tenth 的创新特性能否在非 AI 场景下发挥"精巧、出其不意、强于现有方案"的价值？

### 1.2 创新特性的去 AI 化抽象

将"AI 语境"的特性重新抽象为"语言级通用能力"：

| AI 语境命名 | 通用语言级能力抽象 | 现有主流语言痛点 |
|------------|-------------------|----------------|
| Autograd / `backward()` | **原生自动链式法则 + 反向依赖追踪** | C++/Rust 要手写 adjoint 或引入 Adept/CoDiPack 重库；Python 靠 PyTorch/JAX 绑定 |
| 编译期 Shape 检查 | **编译期多维约束系统 + 符号维度泛型** | C++ 模板元编程极其痛苦；Fortran 完全没有；Ada 有维度但非多维 |
| 护城河 D 内存预估 | **编译期资源预算 warning** | 几乎所有语言都只在运行时 OOM 崩溃 |
| 护城河 A 反向 shape 静态验证 | **自动推导结果的维度正确性** | JAX 形状检查在运行时/trace 时，非编译期 |
| Tensor 统一类型 | **原生多维同质数据 + 广播语义** | C++ 要选 Eigen/xtensor/glm，各自割裂 |

**关键洞察**：Tenth 实质是"**维度安全 + 自动可微 + 编译期资源感知**"的通用数值语言。AI 是第一应用，不是唯一应用。

### 1.3 调研方法

1. 盘点 Tenth 当前能力（autodiff 算子、shape 检查覆盖、内存预估）
2. 选定候选非 AI 场景，逐一分析"核心计算模式 → 所需算子 → Tenth 能力映射 → 竞品短板"
3. 为每个场景设计"30-60 行可验证 demo"
4. 评估"能力就绪度 / 惊喜度 / 可行性 / ROI"

---

## 二、Tenth 当前能力盘点（基线）

### 2.1 Autodiff 算子清单（21 个 TapeOp，全部支持反向传播）

来源：[`tenth/src/runtime/autodiff.rs`](../../tenth/src/runtime/autodiff.rs) 第 29-79 行 `pub enum TapeOp`，`backward` 函数第 272-749 行逐分支实现。

| # | 变体 | 数学语义 | backward 实现行号 |
|---|------|---------|-----------------|
| 1 | `Input` | 叶子参数（梯度累积起点） | 292-300 |
| 2 | `Add` | 元素级加法（带广播） | 301-314 |
| 3 | `Sub` | 元素级减法 | 301-314 |
| 4 | `Mul` | 元素级乘法 | 315-326 |
| 5 | `Div` | 元素级除法 | 327-337 |
| 6 | `Neg` | 取负 `-a` | 338-341 |
| 7 | `ReLU` | `max(0, a)` | 342-349 |
| 8 | `MatMul` | 矩阵乘法 `a @ b` | 350-443 |
| 9 | `Transpose` | 转置最后两维 | 444-454 |
| 10 | `Sum` | 全元素求和→标量 | 455-463 |
| 11 | `Mean` | 全元素求均值→标量 | 464-473 |
| 12 | `Exp` | `exp(a)` | 474-480 |
| 13 | `Log` | `ln(a)` | 481-487 |
| 14 | `Sigmoid` | `1/(1+exp(-a))` | 488-495 |
| 15 | `Softmax` | 最后一维 softmax | 735-745 |
| 16 | `CrossEntropy` | 交叉熵损失 | 723-734 |
| 17 | `Dropout` | Dropout | 712-722 |
| 18 | `Conv2D` | 2D 卷积（im2col + matmul） | 615-711 |
| 19 | `BatchNorm` | 批归一化 | 496-522 |
| 20 | `LayerNorm` | 层归一化（最后一维） | 523-596 |
| 21 | `Gelu` | GELU 激活（tanh 近似） | 597-614 |

**结论**：autodiff 覆盖**完整**，21/21 算子均有反向实现。

### 2.2 Tensor 公共方法清单（67 个 pub fn）

来源：[`tenth/src/runtime/tensor.rs`](../../tenth/src/runtime/tensor.rs)。

| 类别 | 方法数 | 代表方法 |
|------|-------|---------|
| 构造器（f64/f32/通用） | 19 | `zeros`/`ones`/`rand`/`randn`/`full`/`eye`/`arange` 及 f32 变体 |
| dtype/属性访问 | 6 | `dtype`/`is_f32`/`data_f64`/`data_as_f64_view` |
| 梯度相关 | 2 | `zero_grad`/`acc_grad` |
| shape/访问 | 4 | `shape`/`ndim`/`size`/`get` |
| 归约 | 5 | `sum`/`sum_axis`/`mean`/`max_val`/`argmax` |
| 标量运算 | 5 | `add_scalar`/`sub_scalar`/`mul_scalar`/`div_scalar`/`div_scalar_inv` |
| 元素级运算 | 4 | `add_tensor`/`sub_tensor`/`mul_tensor`/`div_tensor` |
| 矩阵乘法 | 1 | `matmul` |
| 转置/广播/重塑 | 4 | `transpose`/`broadcast_to`/`reshape`/`flatten` |
| 一元元素级 | 9 | `neg`/`abs`/`clip_scalar`/`sqrt`/`exp`/`log`/`relu`/`sigmoid`/`tanh` |
| Transformer/NN | 6 | `layer_norm`/`gelu`/`cat`/`masked_fill`/`permute`/`softmax` |
| 其他 | 2 | `im2col`/`assign_` |

**注**：`sin`/`cos`/`tan` **不在** Tensor 方法中，这是物理/几何场景的硬缺口。

### 2.3 编译期 Shape 检查覆盖

来源：[`tenth/src/hir/lower/types.rs`](../../tenth/src/hir/lower/types.rs)。

**`check_method_shape`（第 676-718 行）**：仅覆盖 **`matmul`** 一个方法。
- 检查 2D Tensor `(M, K)` @ `(K, N)` 的内侧 K 维相等
- `Known(a)` vs `Known(b)`：数值不等则报错
- `Symbol(s)` vs `Symbol(t)`：名字不等则报错（同名视为同一维度）
- 其他情况保守通过

**`check_binary_shape_compat`（第 646-667 行）**：覆盖 5 个二元算术（`+`/`-`/`*`/`/`/`%`）的广播兼容性。
- 仅当两侧 Tensor dims 都含静态信息（Known/Symbol）时检查
- 任一侧全 Any（运行时构造）则跳过

**关键发现**：编译期 shape 检查覆盖面**极窄**——只有 matmul + 5 个二元算术有校验。其他方法（`reshape`/`cat`/`permute`/`layer_norm`/`sum_axis` 等）只有 shape 推断，错误只能等运行时报。

### 2.4 护城河 D 内存预估

已实现：编译期 numel 计算 + warning 系统。来源：[`tenth/src/hir/lower/lower_expr.rs`](../../tenth/src/hir/lower/lower_expr.rs)。

---

## 三、候选场景分析

### 3.1 场景 1：金融期权 Greeks（AAD 自动伴随微分）⭐⭐⭐⭐⭐

#### 3.1.1 场景描述

期权定价 V = f(S, K, σ, r, T) 对各参数求偏导，得到风险敏感度 Greeks：

| Greek | 定义 | 用途 |
|-------|------|------|
| Delta | ∂V/∂S | 标的对冲 |
| Gamma | ∂²V/∂S² | Delta 变化率 |
| Vega | ∂V/∂σ | 波动率风险 |
| Theta | ∂V/∂t | 时间衰减 |
| Rho | ∂V/∂r | 利率风险 |

#### 3.1.2 Black-Scholes 公式（所需算子分析）

```
d1 = (log(S/K) + (r + σ²/2) * T) / (σ * sqrt(T))
d2 = d1 - σ * sqrt(T)
N(x) = 标准正态 CDF（可用 sigmoid(x * sqrt(2/π)) 近似）
V_call = S * N(d1) - K * exp(-r*T) * N(d2)
```

**所需算子 → Tenth 支持映射**：

| 算子 | Tenth 支持 | 备注 |
|------|-----------|------|
| `exp`（折现） | ✅ TapeOp::Exp | 直接可用 |
| `log`（对数正态） | ✅ TapeOp::Log | 直接可用 |
| `sqrt`（波动率时间） | ✅ Tensor::sqrt | 直接可用 |
| `+`/`-`/`*`/`/` | ✅ TapeOp 全覆盖 | 直接可用 |
| `matmul`（批量定价） | ✅ TapeOp::MatMul | 向量化多合约 |
| `mean`（损失聚合） | ✅ TapeOp::Mean | 训练损失 |
| `sigmoid`（N(x) 近似） | ✅ TapeOp::Sigmoid | 数值近似 |

**结论**：能力零缺口，所有算子已就绪。

#### 3.1.3 竞品短板证据

| 竞品 | 短板 | 证据 |
|------|------|------|
| **Adept (C++)** | 库侵入式，需 `Stack`/`aVector`/`aReal` 样板；操作符重载增加编译开销；数组库与 AD 库耦合但与其他库不兼容 | [Adept 官网](https://www.met.reading.ac.uk/clouds/adept/) 示例需 `#include <adept_arrays.h>` + `Stack stack;` + `aVector`/`aReal` 类型 |
| **CoDiPack (C++)** | 表达式模板元编程，编译慢、调试难、错误信息冗长 | 设计基于表达式模板，C++ 模板错误是经典痛点 |
| **PyTorch autograd** | Python 解释器开销，金融高频场景性能不够；为 ML 设计，金融语义需包装 | 通用共识 |
| **手写解析 adjoint** | 每个定价模型重写，易错，维护噩梦 | 金融工程教科书通病 |

**Tenth 相对优势**：
- 语言原生 `backward()`，无库侵入
- 编译型语言性能（vs PyTorch）
- 编译期 shape 检查防止"波动率曲面维度不匹配"经典 bug
- 护城河 F 关系调试器：Greeks 出错时定位"哪个参数影响哪个 Greek"

#### 3.1.4 最小可验证 demo 设计（约 40 行）

```tenth
// Black-Scholes + 自动 Greeks
fn black_scholes(S: Tensor, K: Tensor, sigma: Tensor, r: Tensor, T: Tensor) -> Tensor {
    let sqrtT = T.sqrt();
    let d1 = (S / K).log() + (r + sigma * sigma * 0.5) * T;
    let d1 = d1 / (sigma * sqrtT);
    let d2 = d1 - sigma * sqrtT;
    // N(x) ≈ sigmoid(x * 1.5957691)  // sqrt(2/π)
    let Nd1 = (d1 * 1.5957691).sigmoid();
    let Nd2 = (d2 * 1.5957691).sigmoid();
    S * Nd1 - K * (r * T * -1.0).exp() * Nd2
}

fn main() {
    new_grad();
    let S = param(100.0);
    let K = param(100.0);
    let sigma = param(0.2);
    let r = param(0.05);
    let T = param(1.0);
    let V = black_scholes(S, K, sigma, r, T);
    backward(V);
    stop_grad();
    println("V = ", V);
    println("Delta (dV/dS) = ", grad(S));      // 应 ≈ 0.5398
    println("Vega  (dV/dσ) = ", grad(sigma));  // 应 ≈ 37.55
    println("Rho   (dV/dr) = ", grad(r));      // 应 ≈ 53.23
}
```

**验证标准**：与解析解 delta = N(d1) ≈ 0.5398、vega = S*φ(d1)*sqrt(T) ≈ 37.55 数值吻合。

#### 3.1.5 评估

| 维度 | 评分 | 说明 |
|------|------|------|
| 能力就绪度 | ⭐⭐⭐⭐⭐ | 零缺口，所有算子已支持 |
| 惊喜度 | ⭐⭐⭐⭐⭐ | "AI 语言写期权定价"反差强 |
| 可行性 | ⭐⭐⭐⭐⭐ | 无新算子需求 |
| 与护城河协同 | ⭐⭐⭐⭐ | F 关系调试器天然适配 |
| **综合 ROI** | **⭐⭐⭐⭐⭐** | **P0 优先级** |

---

### 3.2 场景 2：物理仿真逆向运动学（IK）⭐⭐⭐⭐

#### 3.2.1 场景描述

给定末端执行器目标位置，反解机器人关节角度。需要雅可比矩阵：

```
J = ∂(末端位置) / ∂(关节角度)  ∈ R^{m×n}  (m=任务维度, n=关节自由度)
```

迭代更新：`θ ← θ - lr * J⁺ * (当前末端位置 - 目标位置)`

#### 3.2.2 2 连杆平面臂雅可比推导

```
forward_kinematics(θ1, θ2):
    x = L1*cos(θ1) + L2*cos(θ1+θ2)
    y = L1*sin(θ1) + L2*sin(θ1+θ2)

雅可比（解析）:
    J = [∂x/∂θ1  ∂x/∂θ2]   [-L1*sin(θ1)-L2*sin(θ1+θ2)   -L2*sin(θ1+θ2)]
        [∂y/∂θ1  ∂y/∂θ2] = [ L1*cos(θ1)+L2*cos(θ1+θ2)    L2*cos(θ1+θ2)]
```

**Tenth 的 autodiff 路径**：对 `forward_kinematics` 调用 `backward(误差)`，自动得到 J 的列（梯度），无需手写解析雅可比。

#### 3.2.3 所需算子 → Tenth 支持映射

| 算子 | Tenth 支持 | 备注 |
|------|-----------|------|
| `sin`/`cos` | ❌ **缺失** | **硬需求**，运动学本质是三角函数 |
| `matmul`（齐次变换） | ✅ TapeOp::MatMul | 直接可用 |
| `+`/`-`/`*` | ✅ TapeOp 全覆盖 | 直接可用 |
| `reshape`（向量化） | ✅ Tensor::reshape | 直接可用 |
| `mean`（误差聚合） | ✅ TapeOp::Mean | 直接可用 |

**关键缺口**：`sin`/`cos`/`tan` 不在 TapeOp 中。

#### 3.2.4 补齐 sin/cos 的实现路径

参照 [`autodiff.rs`](../../tenth/src/runtime/autodiff.rs) 第 474-480 行 `Exp` 的实现：

```rust
// 1. TapeOp 新增变体（autodiff.rs 第 21 行后）
Sin, Cos,

// 2. forward 记录（interpreter/VM 类似 Exp）
TapeOp::Sin => { let v = a.sin(); tape.push(TapeOp::Sin, &[a_id], v.clone()); v }

// 3. backward 实现（autodiff.rs 第 480 行后）
TapeOp::Sin => { *acc_grad = acc_grad * a.cos(); }  // d/dx sin(x) = cos(x)
TapeOp::Cos => { *acc_grad = acc_grad * -a.sin(); } // d/dx cos(x) = -sin(x)

// 4. Tensor 方法（tensor.rs 第 885 行 tanh 后）
pub fn sin(&self) -> Tensor { self.map_f(|x| x.sin()) }
pub fn cos(&self) -> Tensor { self.map_f(|x| x.cos()) }
```

**工作量估算**：约 30-50 行 Rust（含测试），类比 `Exp`/`Log` 的实现模式。

#### 3.2.5 竞品短板证据

| 竞品 | 短板 | 证据 |
|------|------|------|
| **手写解析雅可比** | 每个机器人模型重写，符号推导易错 | 教科书通病 |
| **有限差分** | 精度差（O(h²)），n 关节要 n 次前向计算 | 数值方法经典问题 |
| **PyTorch autograd** | Python 性能限制实时控制；需把运动学写成 tensor 运算 | [IK 综述](https://blog.csdn.net/hiwangwenbing/article/details/159324049) 指出"目前并没有一个万能库" |
| **Pinocchio (C++)** | 库侵入式，Rigid Body Dynamics 库与 AD 解耦需手动集成 | 生态割裂 |

#### 3.2.6 最小可验证 demo 设计（约 50 行）

```tenth
// 2 连杆平面臂 IK
fn forward_kinematics(theta: Tensor, L1: Tensor, L2: Tensor) -> Tensor {
    let t1 = theta[0];
    let t2 = theta[1];
    let x = L1 * t1.cos() + L2 * (t1 + t2).cos();
    let y = L1 * t1.sin() + L2 * (t1 + t2).sin();
    // cat([x, y])  → 2D 位置
}

fn main() {
    let target = tensor[1.0, 0.5];
    let L1 = 1.0;
    let L2 = 1.0;
    new_grad();
    let theta = param(tensor[0.5, 0.5]);  // 初始猜测
    for i in 0..100 {
        let pos = forward_kinematics(theta, L1, L2);
        let err = pos - target;
        let loss = (err * err).mean();
        backward(loss);
        stop_grad();
        let g = grad(theta);
        theta = theta - 0.1 * g;  // 梯度下降
    }
    println("最终关节角度: ", theta);
}
```

**验证标准**：最终末端位置与目标距离 < 1e-3；与解析雅可比数值解吻合。

#### 3.2.7 评估

| 维度 | 评分 | 说明 |
|------|------|------|
| 能力就绪度 | ⭐⭐⭐⭐ | 90%，仅缺 sin/cos |
| 惊喜度 | ⭐⭐⭐⭐ | autodiff 用于 IK 不新，语言原生是新点 |
| 可行性 | ⭐⭐⭐⭐⭐ | 补 sin/cos 是小工作 |
| 与护城河协同 | ⭐⭐⭐ | F 可定位"哪个关节雅可比行维度不对" |
| **综合 ROI** | **⭐⭐⭐⭐** | **P1 优先级** |

---

### 3.3 场景 3：科学计算维度安全（FEM/FDM 网格）⭐⭐⭐⭐

#### 3.3.1 场景描述

PDE 离散化流程：
1. 单元刚度矩阵 Ke（n×n，n=节点自由度）
2. 组装全局刚度矩阵 K（N×N，N=总自由度）
3. 解 Ku=F

经典 bug：Ke 维度与全局编号不匹配 → 运行时崩溃或静默错误。

#### 3.3.2 1D Poisson 方程示例

```
PDE: -u''(x) = f(x),  x ∈ [0,1],  u(0)=u(1)=0
离散化: 单元 [x_i, x_{i+1}], 单元长度 h
单元刚度矩阵: Ke = (1/h) * [[1, -1], [-1, 1]]
全局组装: K[i,i] += 1/h, K[i,i+1] -= 1/h, K[i+1,i] -= 1/h, K[i+1,i+1] += 1/h
载荷: F[i] = f(x_i) * h
解: U = K^{-1} F
```

#### 3.3.3 所需算子 → Tenth 支持映射

| 算子 | Tenth 支持 | 状态 |
|------|-----------|------|
| matmul 维度检查 | ✅ 编译期校验 | 已就绪 |
| 二元算术广播检查 | ✅ 编译期校验 | 已就绪 |
| reshape 维度检查 | ⚠️ 仅推断，无校验 | 需扩展 |
| cat 维度检查 | ⚠️ 仅推断，无校验 | 需扩展 |
| 内存预估（大网格 OOM） | ✅ 护城河 D | 已就绪 |
| 线性方程组求解 | ❌ 缺失 | 需迭代法（CG/GMRES）或绕开 |

#### 3.3.4 竞品短板证据

| 竞品 | 短板 | 证据 |
|------|------|------|
| **Fortran** | 无 shape 检查，维度不匹配静默错误或运行时崩溃 | 语言级缺失 |
| **C++ Eigen** | 运行时 assert，调试版才报，release 版静默错误 | 编译期无维度约束 |
| **Python NumPy** | 运行时 shape mismatch，是顶级调试痛点 | [搜索证据](https://blog.csdn.net/weixin_42376614/article/details/157950204)：shape mismatch 是高频 RuntimeError |
| **PyTorch** | 运行时报错，错误位置与根因距离远 | [腾讯云调试指南](https://cloud.tencent.com/developer/article/2469191) |

#### 3.3.5 最小可验证 demo 设计（约 60 行）

```tenth
fn assemble_stiffness(N: i64) -> Tensor {
    let h = 1.0 / (N - 1);
    let mut K = zeros(N, N);
    let Ke = (1.0 / h) * tensor[[1.0, -1.0], [-1.0, 1.0]];
    for i in 0..(N-1) {
        // 编译期应警告：Ke(2x2) 与 K(NxN) 子块维度约束
        K[i:i+2, i:i+2] = K[i:i+2, i:i+2] + Ke;  // 需切片赋值
    }
    K
}

fn main() {
    let N = 1000;
    let K = assemble_stiffness(N);
    // 护城河 D 应警告：N=10000 时 K 占 800MB
    let F = zeros(N);
    // ... 边界条件处理 + 求解 Ku=F（需补求解器）
}
```

**阻断点**：
1. shape 检查需扩展到 reshape/cat（约 50 行 Rust，参照 [`types.rs`](../../tenth/src/hir/lower/types.rs) 第 676 行 matmul 检查模式）
2. 线性方程组求解器缺失——可先用 Jacobi 迭代绕开，或先只演示组装+内存预估

#### 3.3.6 评估

| 维度 | 评分 | 说明 |
|------|------|------|
| 能力就绪度 | ⭐⭐⭐ | 核心检查有，但 reshape/cat/求解器缺 |
| 惊喜度 | ⭐⭐⭐ | 维度安全是显式卖点，不算意外但实用 |
| 可行性 | ⭐⭐⭐⭐ | 扩展 shape 检查是小工作 |
| 与护城河协同 | ⭐⭐⭐⭐⭐ | 与护城河 A（shape 检查扩展）天然合并 |
| **综合 ROI** | **⭐⭐⭐⭐** | **P2 优先级** |

---

### 3.4 场景 4（探索性）：数据库向量化查询引擎 ⭐⭐⭐

#### 3.4.1 场景描述

列式数据库（DuckDB/ClickHouse）的向量化执行内核：列式批处理、批次大小与列数维度全靠运行时检查。

#### 3.4.2 潜在优势

- Tensor 天然 = 列式批数据
- 编译期符号维度可推断"join 后行数"
- 内存预估防止大 hash join OOM

#### 3.4.3 风险与不确定点

- **类型系统扩展需求**：Tensor 当前仅支持 f32/f64，数据库列可能是字符串/日期/嵌套类型
- **生态壁垒高**：数据库内核需要 SQL 解析器、查询优化器、存储引擎，Tenth 全栈重写不现实
- **性能要求严苛**：DuckDB/ClickHouse 已经过极度优化

#### 3.4.4 评估

| 维度 | 评分 | 说明 |
|------|------|------|
| 能力就绪度 | ⭐⭐ | 类型系统不支持非数值列 |
| 惊喜度 | ⭐⭐⭐⭐⭐ | "用 AI 语言写数据库"反直觉 |
| 可行性 | ⭐⭐ | 需要大量基础设施 |
| **综合 ROI** | **⭐⭐⭐** | **探索性，不建议立即投入** |

---

## 四、能力-场景映射矩阵

| 能力 \ 场景 | 金融 Greeks | 物理 IK | FEM 维度安全 | 数据库（探索） |
|------------|------------|---------|-------------|---------------|
| TapeOp::Exp/Log | ✅ 核心 | ❌ 不需要 | ❌ 不需要 | ❌ 不需要 |
| TapeOp::MatMul | ✅ 批量定价 | ✅ 齐次变换 | ✅ 矩阵组装 | ⚠️ 列式运算 |
| TapeOp::Add/Sub/Mul/Div | ✅ 核心 | ✅ 核心 | ✅ 核心 | ⚠️ 算术列 |
| TapeOp::Sum/Mean | ✅ 损失聚合 | ✅ 误差 | ✅ 载荷聚合 | ✅ 聚合查询 |
| **sin/cos**（缺失） | ❌ | ✅ **硬需求** | ❌ | ❌ |
| **sigmoid**（N(x) 近似） | ✅ 核心 | ❌ | ❌ | ❌ |
| 编译期 matmul 检查 | ✅ 防曲面维度错 | ✅ 防变换维度错 | ✅ 防组装错 | ⚠️ 防 join 错 |
| **reshape/cat 检查**（缺失） | ⚠️ | ⚠️ | ✅ **核心价值** | ⚠️ |
| 护城河 D 内存预估 | ⚠️ 曲面不大 | ⚠️ 模型小 | ✅ **核心价值**（大网格 OOM） | ✅ **核心价值**（hash join） |
| backward() 一键求导 | ✅ **核心价值** | ✅ **核心价值** | ❌ 不需要 | ❌ 不需要 |
| 护城河 F 关系调试器 | ✅ Greeks 出错定位 | ✅ 雅可比维度定位 | ✅ 组装错误定位 | ⚠️ 查询计划调试 |

**关键洞察**：
- **金融 Greeks** 与 Tenth 现有能力**零缺口**，即插即用
- **物理 IK** 仅缺 sin/cos，补齐后即就绪
- **FEM** 不需要 autodiff，但需要**扩展 shape 检查覆盖**——这恰好是护城河 A 的待办
- **数据库** 惊喜度最高但能力缺口最大，仅作探索

---

## 五、推荐优先级与实施路径

### 5.1 优先级排序

| 优先级 | 场景 | 理由 |
|--------|------|------|
| **P0（立即）** | 金融 Greeks | 能力零缺口、惊喜度高、demo 40 行可验证、与护城河 F 协同 |
| **P1（补算子后）** | 物理 IK | 仅缺 sin/cos（约 30 行 Rust）、demo 50 行可验证、机器人赛道叙事强 |
| **P2（扩检查后）** | FEM 维度安全 | 需扩展 reshape/cat shape 检查、惊喜度中但实用、与护城河 A 自然协同 |
| **P3（探索）** | 数据库向量化 | 能力缺口大、生态壁垒高，仅作长期探索 |

### 5.2 实施路径与依赖

```
阶段 1（P0）：金融 Greeks demo
    无依赖 → 直接写 demo → 验证数值吻合
    交付物：Tenth实例/金融Greeks/black_scholes.th
        ↓
阶段 2（P1）：补 sin/cos → 物理 IK demo
    依赖：autodiff.rs 补 Sin/Cos TapeOp（30-50 行）
    交付物：tenth/std/math/trig.th + Tenth实例/逆向运动学/ik_demo.th
        ↓
阶段 3（P2）：扩展 shape 检查 → FEM demo
    依赖：types.rs 扩展 reshape/cat 检查（约 50 行）
    交付物：Tenth实例/FEM维度安全/poisson_1d.th
        ↓
阶段 4（探索）：数据库向量化调研
    仅做可行性报告，不立即实现
```

### 5.3 demo 设计原则

所有 demo 遵循"**30-60 行可验证**"标准：
1. **可运行**：用 Tenth 现有 `tenth run` 路径
2. **可对比**：输出与解析解/竞品结果数值对比
3. **可演示**：展示 Tenth 相对竞品的代码简洁性 + 报错体验

---

## 六、风险评估

### 6.1 技术风险

| 风险 | 概率 | 影响 | 缓解 |
|------|------|------|------|
| Black-Scholes 的 N(x) sigmoid 近似误差过大 | 中 | 中 | 改用 erf 近似或查表；先验证误差 < 1e-4 |
| JIT 路径已知限制（tuple 解构、tensor 索引）影响 demo | 高 | 中 | 用内联公式/伪代码规避，记录于 demo 注释 |
| sin/cos 的 backward 实现有数值边界问题（大角度） | 低 | 低 | 参照 Exp/Log 成熟实现，加单元测试 |
| FEM 求解器缺失阻断 demo | 高 | 中 | 先只演示组装+内存预估，求解用 Jacobi 迭代绕开 |

### 6.2 战略风险

| 风险 | 概率 | 影响 | 缓解 |
|------|------|------|------|
| 非 AI 场景分散核心 AI 能力开发精力 | 中 | 中 | 限定 demo 规模（< 100 行），不深入生态 |
| 金融/物理场景验证不充分反而损害声誉 | 低 | 高 | 必须与解析解数值对比后才发布 |
| 竞品（Adept/CoDiPack）快速跟进 | 低 | 低 | Tenth 的语言级集成是结构性优势，库难复制 |

---

## 七、与战略护城河的协同关系

本调研的场景与 [战略规划.md](战略规划.md) 的护城河形成协同：

```
护城河 A（shape 检查） ← 场景 3（FEM）驱动扩展 reshape/cat 检查
护城河 D（内存预估）   ← 场景 3（FEM 大网格）+ 场景 4（数据库 hash join）驱动
护城河 F（关系调试器） ← 场景 1（Greeks 出错定位）+ 场景 2（雅可比维度定位）驱动
```

**关键洞察**：非 AI 场景不是"分散精力"，而是**为护城河提供具体验证场景**。每个场景的 demo 都同时验证一条护城河的价值。

特别地，**护城河 F（关系调试器）在非 AI 场景的价值更高**：
- AI 场景：张量关系调试（已有 PyTorch 对比）
- 金融场景：Greeks 依赖关系调试（无竞品，独家价值）
- 物理场景：雅可比维度关系调试（无竞品，独家价值）

---

## 八、下一步行动建议

### 8.1 立即可做（P0）

1. **写金融 Greeks demo**（`Tenth实例/金融Greeks/black_scholes.th`）
   - 验证 sigmoid 近似 N(x) 的误差是否可接受
   - 与解析 Greeks 对比数值
   - 40 行内完成

2. **若 demo 成功**：
   - 更新 `MEMO.md` 记录"非 AI 场景首次验证"
   - 更新 `能力梳理/能力全梳理.md` 新增"金融工程"能力项
   - 考虑写一篇短文"用 AI 语言写期权定价"作为对外宣传素材

### 8.2 短期可做（P1）

1. **补 sin/cos 到 TapeOp**（约 30-50 行 Rust）
   - 参照 `Exp`/`Log` 实现模式
   - 加单元测试
   - 同步 tenthc 自举验证

2. **写物理 IK demo**（`Tenth实例/逆向运动学/ik_demo.th`）
   - 2 连杆平面臂
   - 与解析雅可比对比

### 8.3 中期可做（P2）

1. **扩展 shape 检查到 reshape/cat**（约 50 行 Rust）
   - 与护城河 A 合并推进
   - 补单元测试

2. **写 FEM demo**（`Tenth实例/FEM维度安全/poisson_1d.th`）
   - 1D Poisson 方程
   - 演示编译期维度警告 + 内存预估

---

## 九、附录：调研依据

### 9.1 Tenth 源码依据

- [`tenth/src/runtime/autodiff.rs`](../../tenth/src/runtime/autodiff.rs) 第 29-79 行：TapeOp 枚举
- [`tenth/src/runtime/autodiff.rs`](../../tenth/src/runtime/autodiff.rs) 第 272-749 行：backward 实现
- [`tenth/src/runtime/tensor.rs`](../../tenth/src/runtime/tensor.rs)：67 个 pub fn 方法
- [`tenth/src/hir/lower/types.rs`](../../tenth/src/hir/lower/types.rs) 第 646-718 行：shape 检查

### 9.2 竞品资料

- [Adept 官网](https://www.met.reading.ac.uk/clouds/adept/)：C++ AD 库，库侵入式
- CoDiPack：表达式模板 C++ AD 库
- PyTorch autograd：Python ML 框架 AD
- [IK 综述](https://blog.csdn.net/hiwangwenbing/article/details/159324049)：机器人逆向运动学库现状
- [Shape Mismatch 调试指南](https://cloud.tencent.com/developer/article/2469191)：PyTorch/NumPy 维度错误调试痛点

### 9.3 待深入研究的问题

1. sigmoid 近似 N(x) 的最大误差是多少？是否需要改用 erf？
2. sin/cos 的 backward 在大角度（θ > 2π）是否有数值问题？
3. FEM 求解器是否值得在 Tenth 标准库实现（CG 迭代法约 50 行）？
4. 数据库向量化场景的字符串列处理——Tenth 类型系统是否值得扩展？
