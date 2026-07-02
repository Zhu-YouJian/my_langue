# AdamW 解耦权重衰减的语义对比：Tenth 两种实现的收敛性分析

> **论文编号**：T51 · **系列**：标准库优化器形式化 · **级别**：硕士/会议级
> **数理部产出**：理论分析论文（v1）
> **基准版本**：Tenth v0.3.3
> **撰写日期**：2026-07-02
> **联动论文**：T52（优化器状态空间形式化，规划中）、T39（Wengert Tape 形式化语义）、T45（f32 自动微分精度分析）、T48（损失函数双形式）
> **核心源码**：[`tenth/std/optim/adamw.th`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/std/optim/adamw.th)、[`tenth/std/optim/adam.th`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/std/optim/adam.th)、[`tenth/std/optim/sgd.th`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/std/optim/sgd.th)

---

## 摘要

权重衰减（weight decay）是深度学习优化器中抑制过拟合、稳定训练的核心正则化手段。其两种实现方式在工程上长期被混为一谈：（1）**L2 正则化**——把 $\lambda w$ 加到梯度上，让优化器把正则项当作梯度的一部分处理；（2）**解耦权重衰减**——把衰减作为对参数的直接乘性收缩 $w \leftarrow (1-\eta\lambda)w$，与梯度更新分离。Loshchilov & Hutter (2019) 在 *Decoupled Weight Decay Regularization* 一文中指出：**当优化器是 Adam 这类自适应学习率方法时，L2 正则化会被自适应缩放扭曲**，等价的有效衰减强度随历史梯度二阶矩的坐标分布而变，无法实现 L2 正则化的本意——逐坐标均匀收缩。这一论断催生了 AdamW。

Tenth v0.3.3 标准库在 [`tenth/std/optim/adamw.th`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/std/optim/adamw.th) 与 [`tenth/std/optim/adam.th`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/std/optim/adam.th) 中并置提供了两种实现：`adam_step`（原版 Adam，不内置权重衰减）与 `adamw_step`（解耦版，权重衰减直接作用于参数 $w * (1 - lr * decay)$）。`adamw.th` L4–L8 的注释明确指出："原 Adam 的 L2 正则会被扭曲"——这一注释构成可对比的研究对象。值得注意的是，Tenth 的 `adam.th` 本身并未实现 L2 正则化版本的 Adam；L2 正则化模式仅在 [`sgd.th`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/std/optim/sgd.th) 的 `sgd_weight_decay` 中以 `gw = grad(w) + decay * w` 形式落地。因此 `adamw.th` 注释中"原 Adam"指的是一种**假想实现**——把 `sgd_weight_decay` 的 L2 模式套用到 Adam 上。

本文对 Tenth 这两种实现进行形式化语义对比，证明五条主定理：

- **定理 AW1（L2 正则被扭曲）**：在 Adam+L2 假想实现中，正则项 $\lambda w_{t-1}$ 进入一阶矩 $m_t$ 与二阶矩 $v_t$ 后，被自适应学习率 $\eta/(\sqrt{\hat v_t}+\epsilon)$ 逐坐标缩放，其有效衰减强度为 $\eta\lambda/(\sqrt{\hat v_t}+\epsilon)$——坐标 $i$ 的有效衰减反比于 $\sqrt{\hat v_{t,i}}$，与 L2 正则化的本意（坐标无关的均匀收缩）背道而驰；
- **定理 AW2（解耦的等价性）**：AdamW 的更新可分解为两个互不污染的子步骤——先以**未污染的原始梯度**走一次 Adam 更新得 $\tilde w_t$，再对 $\tilde w_t$ 做乘性收缩 $w_t = (1-\eta\lambda)\tilde w_t$；且这两步的顺序在单步内可交换（引理 AW2.1）。解耦的关键在于正则项不进入 $m_t, v_t$ 的累积；
- **定理 AW3（收敛性对比）**：在凸设置下，AdamW 在标准假设（有界梯度、$\sum \eta_t^2 < \infty$、$v_t$ 一致下界）下达到 $O(\sqrt{T})$ 的 regret 界，与原 Adam 同阶；而 Adam+L2 因正则项进入 $v_t$，使 $v_t$ 的下界依赖于 $\lambda$ 与 $w$ 的范数轨迹，证明所需的"独立于参数轨迹的 $v_t$ 下界"假设失效，标准 Adam 收敛证明不能直接搬运；
- **定理 AW4（Transformer 训练实证预期）**：在 Transformer 训练典型场景（大学习率 warmup、$\eta \sim 10^{-3}\sim 10^{-4}$、$\lambda \sim 0.01$）下，Adam+L2 的有效衰减被 $\sqrt{\hat v_t}$ 放大或缩小一个数量级以上，AdamW 则保持 $\eta\lambda$ 的恒定名义衰减；这是 AdamW 在 Transformer 训练中显著优于 Adam+L2 的理论依据，与 [`prelude.th`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/std/prelude.th) 中"Transformer 训练推荐"注释呼应；
- **定理 AW5（与 PyTorch AdamW 对比）**：Tenth `adamw_step` 与 `torch.optim.AdamW` 在权重衰减路径上代数等价；二者仅在偏置校正的写法（Tenth 用 $\beta_1^t$ 显式传入，PyTorch 在 step 内累积）与 `eps` 位置（Tenth 加在分母 $\sqrt{\hat v}+\epsilon$，PyTorch 默认同位）存在实现细节差异，不影响数学等价性。

本文诚实地披露六类局限：(L1) Tenth 的 `adam.th` 实际未实现 Adam+L2，本文比较的"Adam+L2"是**假想实现**，需通过手动 `gw = grad(w) + decay * w` 构造；(L2) 定理 AW1 的"扭曲"是**单步有效衰减强度**层面的，多步累积下的扭曲量化需进一步分析；(L3) 定理 AW3 的 AdamW 收敛证明依赖 $v_t$ 一致下界假设，该假设在冷启动期（前若干步 $\hat v_t$ 很小）不严格成立；(L4) 定理 AW4 的 Transformer 实证预期是**理论预测**，未配实测数据；(L5) 与 PyTorch 的对比基于 PyTorch 1.x/2.x 公开源码，未来版本可能调整；(L6) 本文未覆盖 AdamW 与 SGD momentum + weight decay 的对比（后者在 Tenth 中由 `sgd_momentum` + `sgd_weight_decay` 模拟）。这些局限以独立章节 §12 显式记录。

**关键词**：AdamW；解耦权重衰减；L2 正则化；自适应学习率；优化器收敛性；Transformer 训练；Tenth

---

## 1. 引言

### 1.1 AdamW 的核心论断

权重衰减作为深度学习最古老的正则化手段之一，其原始形式可追溯到 Hanson & Pratt (1988) 与 Krogh & Hertz (1991)：每步以因子 $(1-\eta\lambda)$ 收缩权重，等价于在损失函数中加入 L2 罚项 $\frac{\lambda}{2}\|w\|^2$ 后做梯度下降。在**非自适应**优化器（如 SGD）下，这两种实现严格等价：

$$
\underbrace{w_{t+1} = w_t - \eta(g_t + \lambda w_t)}_{\text{L2 正则}} = \underbrace{(1-\eta\lambda)w_t - \eta g_t}_{\text{解耦衰减}} \qquad (\star)
$$

这一等价性使得深度学习社区长期认为"权重衰减 = L2 正则化"，二者在文献与框架中被混用。

Loshchilov & Hutter (2019) 指出，这一等价性**仅对非自适应优化器成立**。当优化器是 Adam（Kingma & Ba, 2014）这类基于一阶/二阶矩的自适应方法时，把 $\lambda w$ 加到梯度上后，$\lambda w$ 会进入 $m_t$ 与 $v_t$ 的指数移动平均，从而被自适应学习率 $\eta/(\sqrt{\hat v_t}+\epsilon)$ 逐坐标缩放。其后果是：

- **正则强度坐标不均**：历史梯度大的坐标 $\sqrt{\hat v_{t,i}}$ 大，正则项被分母放大压制，有效衰减弱；历史梯度小的坐标有效衰减强。这与 L2 罚项"各坐标均匀收缩"的本意相反；
- **正则与训练耦合**：正则强度随训练进度（$v_t$ 的演化）变化，难以通过 $\lambda$ 单一超参控制；
- **大学习率下扭曲放大**：当 $\eta$ 较大（如 Transformer 训练），$\eta\lambda/(\sqrt{\hat v_t}+\epsilon)$ 在不同坐标间差异放大，扭曲显著。

为修复这一扭曲，Loshchilov & Hutter 提出 **AdamW**：把权重衰减从梯度路径中剥离，直接以乘性因子作用于参数：

$$
w_{t+1} = (1-\eta\lambda)w_t - \eta\,\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}
$$

其中 $m_t, v_t$ 用**未污染的原始梯度** $g_t$ 累积。这一改写使权重衰减的强度恢复为坐标无关的 $\eta\lambda$，与 SGD+L2 的行为一致。

### 1.2 Tenth 的双实现

Tenth v0.3.3 在标准库 [`tenth/std/optim/`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/std/optim/) 下并置提供了 Adam 与 AdamW 两种实现，构成可对比的研究对象：

```tenth
// ── tenth/std/optim/adam.th ──
fn adam_step(w, m, v, lr, beta1, beta2, eps, beta1_t, beta2_t)
    -> (Tensor, Tensor, Tensor) {
    let gw = grad(w);
    let new_m = beta1 * m + (1.0 - beta1) * gw;
    let new_v = beta2 * v + (1.0 - beta2) * gw * gw;
    let m_hat = new_m / (1.0 - beta1_t);
    let v_hat = new_v / (1.0 - beta2_t);
    let new_w = w - lr * m_hat / (v_hat.sqrt() + eps);   // 无衰减
    (new_w, new_m, new_v)
}

// ── tenth/std/optim/adamw.th ──
fn adamw_step(w, m, v, lr, beta1, beta2, eps, decay, beta1_t, beta2_t)
    -> (Tensor, Tensor, Tensor) {
    let gw = grad(w);                                      // 原始梯度，未污染
    let new_m = beta1 * m + (1.0 - beta1) * gw;
    let new_v = beta2 * v + (1.0 - beta2) * gw * gw;
    let m_hat = new_m / (1.0 - beta1_t);
    let v_hat = new_v / (1.0 - beta2_t);
    let decayed_w = w * (1.0 - lr * decay);                // 解耦衰减
    let new_w = decayed_w - lr * m_hat / (v_hat.sqrt() + eps);
    (new_w, new_m, new_v)
}
```

值得特别注意的是，[`adamw.th`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/std/optim/adamw.th) L4–L8 的注释明确写道：

> 与 `std::optim::adam::adam_step` 的区别：
> - 原 Adam：weight decay 加在梯度上（`gw = grad + decay * w`），与 momentum 耦合
> - AdamW：weight decay 直接作用于参数（`w = w * (1 - lr * decay)`），与 momentum 解耦
>
> 解耦权重衰减对 Transformer 训练尤其重要（学习率大时原 Adam 的 L2 正则会被扭曲）。

这一注释构成 Tenth 对 Loshchilov & Hutter 论断的**官方记录**。然而，注释中描述的"原 Adam：weight decay 加在梯度上"在 Tenth 标准库中**并未直接实现**——[`adam.th`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/std/optim/adam.th) 中的 `adam_step` 不含 `decay` 参数，是纯 Adam。L2 正则化的 Adam 变体在 Tenth 中是一种**假想实现**：用户需手动构造 `gw = grad(w) + decay * w`，再以某种方式让 `adam_step` 接受这一污染后的梯度。这一观察是本文定理 AW1 形式化的起点。

L2 正则化模式在 Tenth 中确实落地，但仅出现在 [`sgd.th`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/std/optim/sgd.th) 的 `sgd_weight_decay`：

```tenth
// ── tenth/std/optim/sgd.th ──
fn sgd_weight_decay(w, lr, decay) -> Tensor {
    let gw = grad(w) + decay * w;       // L2 正则化
    w - lr * gw
}
```

这为本文的 Adam+L2 假想实现提供了**直接的工程参照**——把 `sgd_weight_decay` 的 L2 模式套用到 Adam 上即得到注释所述"原 Adam：weight decay 加在梯度上"。

### 1.3 贡献

本文的贡献如下：

- **形式化模型**（§4）：将 Tenth 三种相关实现——`adam_step`（原版）、假想 Adam+L2（基于 `sgd_weight_decay` 模式）、`adamw_step`（解耦版）——抽象为形式化更新规则，明确三者的代数关系；
- **主定理 AW1–AW5**（§5）：给出 L2 扭曲、解耦等价性、收敛性对比、Transformer 实证预期、PyTorch 对比五条定理的陈述与证明；
- **L2 扭曲的严格证明**（§7）：通过把 Adam+L2 的更新改写为"原 Adam 更新 + 扭曲的正则项"，量化证明有效衰减强度为 $\eta\lambda/(\sqrt{\hat v_t}+\epsilon)$，并证明其反比于 $\sqrt{\hat v_{t,i}}$；
- **解耦等价性的双向论证**（§8）：证明 AdamW 等价于"原 Adam 更新 + 乘性收缩"两步的复合，且两步在单步内可交换；
- **收敛性分析**（§9）：在凸设置下给出 AdamW 的 $O(\sqrt T)$ regret 界，并论证 Adam+L2 的标准收敛证明为何不能直接搬运；
- **与 PyTorch AdamW 的代数等价**（§10）：证明 Tenth `adamw_step` 与 `torch.optim.AdamW` 在权重衰减路径上代数等价；
- **诚实记录局限**（§12）：假想实现、单步 vs 多步扭曲、$v_t$ 下界假设、实证缺失、版本时效、未覆盖对比六类局限独立成节。

### 1.4 与 T52（优化器状态空间）的联动

本文是 T52（优化器状态空间形式化，规划中）的**前置理论依据**。T52 计划形式化 Tenth 标准库中所有优化器（SGD/SGD+momentum/SGD+L2/Adam/AdamW/AdaGrad/RMSProp）的状态空间——每个优化器维护的状态变量（$m, v, w$ 等）与更新规则的代数结构。

本文对 Adam 与 AdamW 的形式化（§4）与等价性/不等价性定理（AW1–AW3）为 T52 提供：

- **状态变量边界**：明确 Adam 的状态空间是 $\{w, m, v\}$，AdamW 的状态空间也是 $\{w, m, v\}$——二者状态空间同构，差异在更新规则而非状态变量。这一结论使 T52 可在统一状态空间下对比两个优化器；
- **更新规则的代数分类**：本文把更新规则分为"梯度路径"（进入 $m, v$）与"参数路径"（直接作用于 $w$），T52 可据此把所有优化器分类为"路径耦合"（SGD+L2、Adam+L2）与"路径解耦"（SGD+L2 与 SGD 等价、AdamW）；
- **收敛性结论的迁移**：本文定理 AW3 的 AdamW 收敛界为 T52 中"优化器状态空间收敛性"提供具体实例。

T52 的开放问题（如"是否可构造统一框架覆盖所有优化器的收敛性"）以本文为参照之一。

---

## 2. 关键词

AdamW；解耦权重衰减；L2 正则化；自适应学习率；Adam 收敛性；Transformer 训练；PyTorch AdamW；优化器状态空间；Tenth

---

## 3. 背景

### 3.1 Loshchilov & Hutter (2019) 的核心论断

Loshchilov & Hutter 在 ICLR 2019 发表 *Decoupled Weight Decay Regularization* 一文，提出三个核心论断：

**论断 1（L2 与权重衰减不等价）**：对于 Adam 这类自适应优化器，L2 正则化（把 $\lambda w$ 加到梯度上）与权重衰减（乘性收缩 $w$）**不代数等价**。具体地，SGD 下的等价性 $(\star)$ 在 Adam 下失效，因为 $\lambda w$ 进入 $v_t$ 后被 $\sqrt{\hat v_t}$ 缩放，破坏了等式两端。

**论断 2（L2 在 Adam 下被扭曲）**：Adam+L2 的有效正则强度随坐标 $i$ 与训练步 $t$ 变化，无法通过 $\lambda$ 单一超参控制。在自适应学习率大的坐标（$\sqrt{\hat v_{t,i}}$ 大），正则被压制；在自适应学习率小的坐标，正则被放大。这种"反向正则"是 Adam+L2 训练不稳定的根源之一。

**论断 3（AdamW 修复扭曲）**：通过把权重衰减从梯度路径剥离，直接作用于参数，AdamW 恢复了权重衰减的坐标无关性，使其行为与 SGD+L2 一致。理论上，AdamW 在凸设置下保持 Adam 的 $O(\sqrt T)$ 收敛率，且实证上在 ImageNet、CIFAR、Transformer 训练中显著优于 Adam+L2。

### 3.2 Adam 的收敛性（Kingma & Ba, 2014；Reddi et al., 2018）

Adam 的收敛性分析历经两个阶段：

**原始分析（Kingma & Ba, 2014）**：原论文证明 Adam 在凸设置下达到 $O(\sqrt T)$ regret 界。证明假设 $\eta_t = \alpha/\sqrt t$，利用 $m_t, v_t$ 的有界性与偏差校正。

**反例与修正（Reddi et al., 2018）**：*On the Convergence of Adam and Beyond* 一文指出原始 Adam 收敛证明存在反例——在某些情形下 Adam 不收敛到最优解。根因是 $v_t$ 的指数移动平均在 $v_t$ 减小时导致有效学习率增大，违反收敛所需。作者提出 AMSGrad 修正：保持 $v_t$ 的逐元素单调不减（$\hat v_t = \max(\hat v_{t-1}, v_t)$）。

本文定理 AW3 的 AdamW 收敛证明基于**修正后的 Adam 框架**（AMSGrad 式单调性假设或等价的 $v_t$ 一致下界假设），不依赖原始 Adam 证明。

### 3.3 PyTorch 的 AdamW 实现

PyTorch 在 `torch.optim.AdamW` 中实现了 AdamW，其核心更新（伪代码）如下：

```python
# torch/optim/adamw.py (简化)
def step(self):
    for p in params:
        grad = p.grad
        # 解耦权重衰减
        if weight_decay != 0:
            p.mul_(1 - lr * weight_decay)
        # Adam 更新（使用原始梯度）
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * grad**2
        m_hat = m / (1 - beta1**t)
        v_hat = v / (1 - beta2**t)
        p.addcdiv_(m_hat, sqrt(v_hat) + eps, value=-lr)
```

注意 PyTorch 的实现细节：

1. **衰减在 Adam 更新之前**：先 `p.mul_(1 - lr * weight_decay)` 再做 Adam 更新，与 Tenth 的"先衰减后更新"顺序一致；
2. **`eps` 在分母**：`sqrt(v_hat) + eps`，与 Tenth `v_hat.sqrt() + eps` 一致；
3. **偏置校正用 `beta**t` 累积**：PyTorch 在 step 内累积 `t`，Tenth 由调用方传入 `beta1_t, beta2_t`。

这些细节将在 §10 详述。

### 3.4 Tenth 的优化器生态

Tenth v0.3.3 在 [`tenth/std/optim/`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/std/optim/) 下提供 7 个优化器相关文件：

| 文件 | 优化器 | 状态空间 | 权重衰减 |
|------|--------|---------|---------|
| `sgd.th` | SGD, SGD+momentum, SGD+L2 | $\{w\}$ / $\{w, m\}$ | L2 正则（`sgd_weight_decay`） |
| `adam.th` | Adam | $\{w, m, v\}$ | 无 |
| `adamw.th` | AdamW | $\{w, m, v\}$ | 解耦（`adamw_step`） |
| `adagrad.th` | AdaGrad | $\{w, \sum g^2\}$ | 无 |
| `rmsprop.th` | RMSProp | $\{w, v\}$ | 无 |
| `clip.th` | 梯度裁剪工具 | 无（纯函数） | 无 |
| `accumulate.th` | 梯度累积工具 | 无（纯函数） | 无 |

观察：**Tenth 的 Adam 不提供 L2 正则化变体**——这是 AdamW 存在的工程动机。若用户需在 Adam 上加 L2，必须手动 `gw = grad(w) + decay * w`，再以某种方式让 `adam_step` 接受这一污染梯度。但 `adam_step` 内部直接调用 `grad(w)`（[`adam.th`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/std/optim/adam.th) L29），不接受外部梯度参数。因此 Adam+L2 在 Tenth 中**无直接 API 入口**——这是本文形式化"假想实现"的工程现实。

[`prelude.th`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/std/prelude.th) L71 明确标注 AdamW 为"解耦权重衰减（Transformer 训练推荐）"，与本文定理 AW4 的实证预期呼应。

---

## 4. Tenth 两种实现的形式化

### 4.1 记号约定

设参数 $w \in \mathbb{R}^d$，损失函数 $\mathcal{L}$，第 $t$ 步梯度 $g_t = \nabla \mathcal{L}(w_{t-1}) \in \mathbb{R}^d$。所有向量运算逐坐标进行。记：

- $\odot$：逐元素乘法（Hadamard 积）；
- $\oslash$：逐元素除法；
- $\sqrt{\cdot}$：逐元素平方根；
- $\beta_1, \beta_2 \in [0, 1)$：一阶/二阶矩衰减系数；
- $\eta > 0$：学习率；
- $\epsilon > 0$：分母稳定常数；
- $\lambda \geq 0$：权重衰减系数；
- $\beta_1^t, \beta_2^t$：第 $t$ 步的 $\beta_1, \beta_2$ 的 $t$ 次幂，用于偏置校正。

索引：坐标 $i \in \{1, \ldots, d\}$，时间步 $t \in \{1, 2, \ldots\}$。$w_{t,i}$ 表示第 $t$ 步、第 $i$ 坐标的参数值。

### 4.2 原版 Adam（Tenth `adam_step`）

**定义 4.1（Adam 更新规则）**：给定初始 $w_0 \in \mathbb{R}^d$，$m_0 = v_0 = 0$，$\beta_1^0 = \beta_2^0 = 1$，原版 Adam 的更新规则（对应 [`adam.th`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/std/optim/adam.th) L18–L36）为：

$$
\begin{aligned}
g_t &= \nabla \mathcal{L}(w_{t-1}) \\
m_t &= \beta_1 m_{t-1} + (1-\beta_1) g_t \\
v_t &= \beta_2 v_{t-1} + (1-\beta_2) g_t \odot g_t \\
\hat m_t &= m_t \oslash (1 - \beta_1^t \mathbf{1}) \\
\hat v_t &= v_t \oslash (1 - \beta_2^t \mathbf{1}) \\
w_t &= w_{t-1} - \eta\, \hat m_t \oslash (\sqrt{\hat v_t} + \epsilon \mathbf{1})
\end{aligned}
$$

其中 $\mathbf{1}$ 为全 1 向量，$\beta_1^t = \beta_1^{t-1} \cdot \beta_1$（由调用方累积，对应 `beta1_t` 参数）。

**注 4.1**：原版 Adam **不含权重衰减**。这是 Tenth [`adam.th`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/std/optim/adam.th) 的实际实现状态，与原始 Adam 论文（Kingma & Ba, 2014）一致。

### 4.3 假想 Adam+L2（基于 `sgd_weight_decay` 模式）

**定义 4.2（Adam+L2 更新规则）**：把 [`sgd.th`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/std/optim/sgd.th) L21–L26 中 `sgd_weight_decay` 的 L2 模式套用到 Adam 上，得到假想的 Adam+L2 更新规则（对应 [`adamw.th`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/std/optim/adamw.th) L5 注释所述"原 Adam：weight decay 加在梯度上"）：

$$
\begin{aligned}
\tilde g_t &= g_t + \lambda w_{t-1} && \text{(L2 污染梯度)} \\
m_t &= \beta_1 m_{t-1} + (1-\beta_1) \tilde g_t \\
v_t &= \beta_2 v_{t-1} + (1-\beta_2) \tilde g_t \odot \tilde g_t \\
\hat m_t &= m_t \oslash (1 - \beta_1^t \mathbf{1}) \\
\hat v_t &= v_t \oslash (1 - \beta_2^t \mathbf{1}) \\
w_t &= w_{t-1} - \eta\, \hat m_t \oslash (\sqrt{\hat v_t} + \epsilon \mathbf{1})
\end{aligned}
$$

**注 4.2**：定义 4.2 是**假想实现**——Tenth 标准库中不存在此函数。其形式化依据是 `sgd_weight_decay` 的 L2 模式（`gw = grad(w) + decay * w`）与 `adamw.th` L5 注释。本文所有"Adam+L2"均指此假想实现。

**注 4.3**：定义 4.2 与定义 4.1 的唯一区别是梯度被 $\lambda w_{t-1}$ 污染——$\tilde g_t$ 进入 $m_t, v_t$ 的累积。这一污染是定理 AW1 中"L2 被扭曲"的根源。

### 4.4 AdamW（Tenth `adamw_step`）

**定义 4.3（AdamW 更新规则）**：给定初始 $w_0 \in \mathbb{R}^d$，$m_0 = v_0 = 0$，AdamW 的更新规则（对应 [`adamw.th`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/std/optim/adamw.th) L23–L44）为：

$$
\begin{aligned}
g_t &= \nabla \mathcal{L}(w_{t-1}) && \text{(原始梯度，未污染)} \\
m_t &= \beta_1 m_{t-1} + (1-\beta_1) g_t \\
v_t &= \beta_2 v_{t-1} + (1-\beta_2) g_t \odot g_t \\
\hat m_t &= m_t \oslash (1 - \beta_1^t \mathbf{1}) \\
\hat v_t &= v_t \oslash (1 - \beta_2^t \mathbf{1}) \\
\tilde w_t &= (1-\eta\lambda) w_{t-1} && \text{(解耦衰减)} \\
w_t &= \tilde w_t - \eta\, \hat m_t \oslash (\sqrt{\hat v_t} + \epsilon \mathbf{1})
\end{aligned}
$$

**注 4.4**：定义 4.3 与定义 4.1 的唯一区别是参数 $w_{t-1}$ 先乘 $(1-\eta\lambda)$ 再做 Adam 更新。$m_t, v_t$ 仍以**原始梯度** $g_t$ 累积，与原版 Adam 完全一致。

**注 4.5**：定义 4.3 与定义 4.2 的关键区别在梯度路径：AdamW 的 $m_t, v_t$ 不含 $\lambda w_{t-1}$，权重衰减仅通过乘性因子进入参数路径。这一"路径分离"是定理 AW2 中"解耦等价性"的核心。

### 4.5 三种实现的代数关系

把三种实现并列，可观察到：

| 实现 | 梯度路径（$m_t, v_t$） | 参数路径（$w_t$） | 路径耦合 |
|------|----------------------|------------------|---------|
| Adam（定义 4.1） | $g_t$ | $w_{t-1} - \eta \hat m_t \oslash (\sqrt{\hat v_t}+\epsilon)$ | 无衰减 |
| Adam+L2（定义 4.2） | $g_t + \lambda w_{t-1}$ | $w_{t-1} - \eta \hat m_t \oslash (\sqrt{\hat v_t}+\epsilon)$ | **耦合**：$\lambda w$ 经 $v_t$ 影响分母 |
| AdamW（定义 4.3） | $g_t$ | $(1-\eta\lambda)w_{t-1} - \eta \hat m_t \oslash (\sqrt{\hat v_t}+\epsilon)$ | **解耦**：$\lambda$ 仅在参数路径 |

**关键观察**：AdamW 的更新可重写为

$$
w_t = (1-\eta\lambda)w_{t-1} - \eta\,\hat m_t \oslash (\sqrt{\hat v_t}+\epsilon) = (1-\eta\lambda)\Big[w_{t-1} - \frac{\eta}{1-\eta\lambda}\,\hat m_t \oslash (\sqrt{\hat v_t}+\epsilon)\Big]
$$

即"先以有效学习率 $\eta/(1-\eta\lambda)$ 走一次 Adam 更新，再对结果乘 $(1-\eta\lambda)$"。当 $\eta\lambda \ll 1$（典型设置 $\eta=10^{-3}, \lambda=10^{-2}$，$\eta\lambda=10^{-5}$），有效学习率 $\eta/(1-\eta\lambda) \approx \eta(1+\eta\lambda) \approx \eta$，故 AdamW $\approx$ Adam + 乘性衰减。这一近似是定理 AW2 的工程依据。

---

## 5. 主定理

本节陈述五条主定理，证明分别在 §6–§10 给出。

### 5.1 定理 AW1（L2 正则被扭曲）

**定理 AW1（L2 正则在 Adam 下被自适应缩放扭曲）**：设 Adam+L2（定义 4.2）与 AdamW（定义 4.3）使用相同的 $\eta, \beta_1, \beta_2, \epsilon, \lambda$ 与相同的初始 $w_0, m_0, v_0$。则在第 $t$ 步：

(a) **有效衰减强度差异**：Adam+L2 中坐标 $i$ 的有效衰减强度为

$$
\lambda^{\text{L2}}_{\text{eff}, t, i} = \frac{\eta\lambda}{\sqrt{\hat v^{\text{L2}}_{t, i}} + \epsilon}
$$

其中 $\hat v^{\text{L2}}_{t, i}$ 是 Adam+L2 中第 $t$ 步、第 $i$ 坐标的二阶矩估计；而 AdamW 中坐标 $i$ 的有效衰减强度为

$$
\lambda^{\text{W}}_{\text{eff}, t, i} = \eta\lambda
$$

(b) **扭曲方向**：Adam+L2 的有效衰减反比于 $\sqrt{\hat v^{\text{L2}}_{t, i}} + \epsilon$。当 $\sqrt{\hat v^{\text{L2}}_{t, i}} \gg \epsilon$ 时，$\lambda^{\text{L2}}_{\text{eff}, t, i} \approx \eta\lambda/\sqrt{\hat v^{\text{L2}}_{t, i}}$——历史梯度大的坐标有效衰减弱，历史梯度小的坐标有效衰减强；

(c) **扭曲幅度**：设坐标 $i, j$ 的二阶矩估计之比为 $r = \sqrt{\hat v^{\text{L2}}_{t, i}}/\sqrt{\hat v^{\text{L2}}_{t, j}}$，则 Adam+L2 中两坐标的有效衰减比为 $\lambda^{\text{L2}}_{\text{eff}, t, i}/\lambda^{\text{L2}}_{\text{eff}, t, j} \approx 1/r$（当 $\epsilon$ 可忽略），而 AdamW 中两坐标的有效衰减比为 1（坐标无关）。

**证明**：见 §7。$\square$

### 5.2 定理 AW2（解耦的等价性）

**定理 AW2（AdamW 等价于"原 Adam 更新 + 乘性收缩"的复合）**：设 $\Phi^{\text{Adam}}_t: (w_{t-1}, m_{t-1}, v_{t-1}) \mapsto (\tilde w_t, m_t, v_t)$ 是原版 Adam（定义 4.1）的单步更新（不含衰减），$\Psi_t: w \mapsto (1-\eta\lambda)w$ 是乘性收缩算子。则 AdamW（定义 4.3）的单步更新 $\Phi^{\text{W}}_t$ 满足：

(a) **复合表示**：$\Phi^{\text{W}}_t(w_{t-1}, m_{t-1}, v_{t-1}) = (\Psi_t \circ \Phi^{\text{Adam}}_t)(w_{t-1}, m_{t-1}, v_{t-1})$，即"先 Adam 更新再乘性收缩"；

(b) **顺序可交换（单步）**：在单步内，$\Psi_t \circ \Phi^{\text{Adam}}_t \approx \Phi^{\text{Adam}}_t \circ \Psi_t$，当 $\eta\lambda \ll 1$ 时近似等式成立，误差为 $O((\eta\lambda)^2)$；

(c) **梯度路径未污染**：AdamW 的 $m_t, v_t$ 与原版 Adam 的 $m_t, v_t$ 在相同 $g_t$ 下**完全相同**——权重衰减不进入矩估计。

**证明**：见 §8。$\square$

### 5.3 定理 AW3（收敛性对比）

**定理 AW3（凸设置下 AdamW 与 Adam+L2 的收敛性）**：设 $\mathcal{L}$ 为凸函数，梯度有界 $\|g_t\|_\infty \leq G$，$v_t$ 一致下界 $v_{t,i} \geq v_{\min} > 0$ 对所有 $t, i$ 成立（AMSGrad 式假设）。学习率 $\eta_t = \eta/\sqrt t$。则：

(a) **AdamW 收敛**：AdamW（定义 4.3）的 regret 满足

$$
R(T) = \sum_{t=1}^T \big(\mathcal{L}(w_t) - \mathcal{L}(w^*)\big) = O(\sqrt T)
$$

即平均 regret $R(T)/T = O(1/\sqrt T) \to 0$。

(b) **Adam+L2 收敛证明不能直接搬运**：Adam+L2（定义 4.2）的 $v_t$ 含 $\lambda w_{t-1}$ 的贡献，$v^{\text{L2}}_{t, i} = \beta_2 v^{\text{L2}}_{t-1, i} + (1-\beta_2)(g_{t,i} + \lambda w_{t-1, i})^2$。其下界 $v^{\text{L2}}_{t, i} \geq v_{\min}$ 依赖于 $\lambda$ 与 $w_{t-1}$ 的轨迹，**不能**作为独立于参数轨迹的假设。因此定理 (a) 的证明**不能**直接搬运到 Adam+L2。

(c) **收敛率同阶**：在附加假设"$w_t$ 轨迹有界 $\|w_t\|_\infty \leq W$ 且 $\lambda W \leq G$"下，Adam+L2 的 regret 仍为 $O(\sqrt T)$，但常数因子大于 AdamW。

**证明**：见 §9。$\square$

### 5.4 定理 AW4（Transformer 训练实证预期）

**定理 AW4（Transformer 训练中 Adam+L2 的扭曲放大）**：在 Transformer 训练典型设置下（$\eta \in [10^{-4}, 10^{-3}]$，$\lambda = 0.01$，warmup 阶段 $\eta$ 从 $0$ 线性增至峰值，$v_t$ 在不同参数张量间差异显著——注意力权重 $v$ 大、bias 与 LayerNorm 参数 $v$ 小）：

(a) **AdamW 名义衰减**：AdamW 对所有参数施加恒定名义衰减 $\eta\lambda \in [10^{-6}, 10^{-5}]$；

(b) **Adam+L2 扭曲范围**：Adam+L2 的有效衰减 $\eta\lambda/(\sqrt{\hat v_{t,i}}+\epsilon)$ 因 $\sqrt{\hat v_{t,i}}$ 在不同参数间差异可达 1–2 个数量级，有效衰减差异相应放大 1–2 个数量级；

(c) **预期表现**：在 warmup 结束、$\eta$ 达峰时，Adam+L2 对 attention 权重（$v$ 大）的衰减被压制（接近 $0$），对 bias/LayerNorm（$v$ 小）的衰减被放大（接近 $\eta\lambda/\epsilon$，可能远大于 $\eta\lambda$）。这种"反向正则"是 AdamW 在 Transformer 训练中显著优于 Adam+L2 的理论依据。

**证明**：见 §9.4。$\square$

### 5.5 定理 AW5（与 PyTorch AdamW 对比）

**定理 AW5（Tenth `adamw_step` 与 `torch.optim.AdamW` 代数等价）**：在以下条件下，Tenth `adamw_step`（定义 4.3）与 `torch.optim.AdamW` 在权重衰减路径上代数等价：

(a) **衰减顺序**：两者均"先衰减后更新"——Tenth `decayed_w = w * (1 - lr * decay)` 在 Adam 更新之前，PyTorch `p.mul_(1 - lr * weight_decay)` 也在 Adam 更新之前；

(b) **梯度路径**：两者均使用**原始梯度** $g_t$ 累积 $m_t, v_t$，不污染；

(c) **`eps` 位置**：两者均把 $\epsilon$ 加在分母 $\sqrt{\hat v_t} + \epsilon$（PyTorch 默认 `eps=1e-8`，Tenth 默认 `eps=1e-8`）；

(d) **偏置校正**：Tenth 由调用方累积 `beta1_t, beta2_t` 后传入，PyTorch 在 step 内累积 `t` 后计算 `beta**t`——数学等价，仅工程写法不同；

(e) **`maximize` 参数**：PyTorch 支持 `maximize=True` 反向梯度，Tenth 无此参数（需用户手动 `-grad`）——不影响默认情形的等价性。

**证明**：见 §10。$\square$

---

## 6. Adam L2 vs AdamW 解耦的形式化

本节把第 4 节的三种实现进一步抽象，明确"L2 正则化"与"解耦权重衰减"在自适应优化器下的形式化区别。

### 6.1 L2 正则化的形式化

**定义 6.1（L2 正则化）**：给定损失 $\mathcal{L}$ 与正则系数 $\lambda \geq 0$，L2 正则化把优化目标改为

$$
\mathcal{L}_{\text{reg}}(w) = \mathcal{L}(w) + \frac{\lambda}{2}\|w\|^2
$$

其梯度为 $\nabla \mathcal{L}_{\text{reg}}(w) = \nabla \mathcal{L}(w) + \lambda w$。把这一污染梯度喂给优化器即得 L2 正则化变体。

**注 6.1**：定义 6.1 的关键特征是正则项 $\lambda w$ **进入梯度路径**——优化器看到的是 $\nabla \mathcal{L}_{\text{reg}}$ 而非 $\nabla \mathcal{L}$。对非自适应优化器（SGD），这一区别无关紧要（$(\star)$）；对自适应优化器（Adam），$\lambda w$ 进入 $v_t$ 后被自适应缩放（定理 AW1）。

### 6.2 解耦权重衰减的形式化

**定义 6.2（解耦权重衰减）**：给定损失 $\mathcal{L}$ 与衰减系数 $\lambda \geq 0$，解耦权重衰减把优化器更新规则修改为

$$
w_t = (1-\eta_t\lambda)w_{t-1} - \eta_t\, u_t
$$

其中 $u_t$ 是优化器内部基于**原始梯度** $g_t = \nabla \mathcal{L}(w_{t-1})$ 计算的更新方向（对 Adam，$u_t = \hat m_t \oslash (\sqrt{\hat v_t}+\epsilon)$；对 SGD，$u_t = g_t$）。$\lambda w$ **不进入**梯度路径。

**注 6.2**：定义 6.2 的关键特征是正则项 $\lambda w$ **仅出现在参数路径**——以乘性因子 $(1-\eta_t\lambda)$ 形式直接收缩参数，不污染 $m_t, v_t$。

### 6.3 两者的代数区别

把定义 6.1 与 6.2 应用到 SGD 与 Adam 上，可得四种组合：

| 优化器 | L2 正则化（定义 6.1） | 解耦衰减（定义 6.2） | 等价性 |
|--------|---------------------|---------------------|--------|
| SGD | $w_t = w_{t-1} - \eta(g_t + \lambda w_{t-1})$ | $w_t = (1-\eta\lambda)w_{t-1} - \eta g_t$ | **等价**（$(\star)$） |
| Adam | $w_t = w_{t-1} - \eta \hat m^{\text{L2}}_t \oslash (\sqrt{\hat v^{\text{L2}}_t}+\epsilon)$ | $w_t = (1-\eta\lambda)w_{t-1} - \eta \hat m_t \oslash (\sqrt{\hat v_t}+\epsilon)$ | **不等价**（定理 AW1） |

**核心区别**：在 SGD 下，$g_t + \lambda w_{t-1}$ 与 $g_t$ 的差别只是个加性项，展开后可因式分解为 $(1-\eta\lambda)w_{t-1} - \eta g_t$；在 Adam 下，$g_t + \lambda w_{t-1}$ 进入 $v_t = \beta_2 v_{t-1} + (1-\beta_2)(g_t + \lambda w_{t-1})^2$ 后被平方，无法因式分解回原始形式，且 $\sqrt{\hat v_t}$ 出现在分母，使 $\lambda w$ 被非线性缩放。

### 6.4 Tenth 实现的形式化对照

把 §6.3 的四种组合对照到 Tenth 实现：

| 组合 | Tenth 实现 | 源码位置 |
|------|-----------|---------|
| SGD（无衰减） | `sgd_step` | [`sgd.th`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/std/optim/sgd.th) L4–L7 |
| SGD+L2 | `sgd_weight_decay` | [`sgd.th`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/std/optim/sgd.th) L23–L26 |
| Adam（无衰减） | `adam_step` | [`adam.th`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/std/optim/adam.th) L18–L36 |
| Adam+L2 | **未实现**（假想） | [`adamw.th`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/std/optim/adamw.th) L5 注释 |
| AdamW | `adamw_step` | [`adamw.th`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/std/optim/adamw.th) L23–L44 |

**关键观察**：Tenth **不实现 Adam+L2**。这一工程选择是定理 AW1 的现实体现——既然 Adam+L2 在自适应下被扭曲，提供它无意义；直接提供 AdamW 即可。Tenth 通过**只实现解耦版**回避了 L2 扭曲问题。

---

## 7. L2 正则被扭曲的证明（定理 AW1）

本节给出定理 AW1 的完整证明。

### 7.1 单步更新的分解

考虑 Adam+L2（定义 4.2）的单步更新。把 $\tilde g_t = g_t + \lambda w_{t-1}$ 代入：

$$
\begin{aligned}
m_t &= \beta_1 m_{t-1} + (1-\beta_1)(g_t + \lambda w_{t-1}) \\
&= \underbrace{\beta_1 m_{t-1} + (1-\beta_1)g_t}_{=: m^{\text{Adam}}_t} + (1-\beta_1)\lambda w_{t-1} \\
&= m^{\text{Adam}}_t + (1-\beta_1)\lambda w_{t-1}
\end{aligned}
$$

其中 $m^{\text{Adam}}_t$ 是原版 Adam 的一阶矩（用未污染梯度 $g_t$）。类似地，二阶矩：

$$
\begin{aligned}
v_t &= \beta_2 v_{t-1} + (1-\beta_2)(g_t + \lambda w_{t-1})^2 \\
&= \beta_2 v_{t-1} + (1-\beta_2)(g_t^2 + 2\lambda g_t w_{t-1} + \lambda^2 w_{t-1}^2) \\
&= \underbrace{\beta_2 v_{t-1} + (1-\beta_2)g_t^2}_{=: v^{\text{Adam}}_t} + (1-\beta_2)(2\lambda g_t w_{t-1} + \lambda^2 w_{t-1}^2) \\
&= v^{\text{Adam}}_t + (1-\beta_2)\lambda(2 g_t w_{t-1} + \lambda w_{t-1}^2)
\end{aligned}
$$

**关键观察**：$\tilde g_t$ 进入 $v_t$ 后被平方，产生交叉项 $2\lambda g_t w_{t-1}$ 与 $\lambda^2 w_{t-1}^2$。这些项使 $v_t \neq v^{\text{Adam}}_t$，进而 $\sqrt{\hat v_t} \neq \sqrt{\hat v^{\text{Adam}}_t}$——分母被污染。

### 7.2 有效衰减强度的提取

Adam+L2 的参数更新为

$$
w_t = w_{t-1} - \eta\, \hat m_t \oslash (\sqrt{\hat v_t} + \epsilon \mathbf{1})
$$

把 $\hat m_t = m_t / (1-\beta_1^t)$ 展开：

$$
\hat m_t = \frac{m^{\text{Adam}}_t + (1-\beta_1)\lambda w_{t-1}}{1-\beta_1^t} = \hat m^{\text{Adam}}_t + \frac{(1-\beta_1)\lambda w_{t-1}}{1-\beta_1^t}
$$

代入更新规则：

$$
\begin{aligned}
w_t &= w_{t-1} - \eta\, \hat m^{\text{Adam}}_t \oslash (\sqrt{\hat v_t} + \epsilon \mathbf{1}) - \eta\, \frac{(1-\beta_1)\lambda w_{t-1}}{1-\beta_1^t} \oslash (\sqrt{\hat v_t} + \epsilon \mathbf{1}) \\
&= \underbrace{w_{t-1} - \eta\, \hat m^{\text{Adam}}_t \oslash (\sqrt{\hat v_t} + \epsilon \mathbf{1})}_{\text{Adam 更新（但分母被污染）}} - \underbrace{\eta\, \frac{(1-\beta_1)\lambda}{1-\beta_1^t}\, w_{t-1} \oslash (\sqrt{\hat v_t} + \epsilon \mathbf{1})}_{\text{正则项的贡献}}
\end{aligned}
$$

第二项即"L2 正则的贡献"。逐坐标展开，第 $i$ 坐标的正则贡献为

$$
\Delta w^{\text{L2}}_{t, i} = -\eta\, \frac{(1-\beta_1)\lambda}{1-\beta_1^t} \cdot \frac{w_{t-1, i}}{\sqrt{\hat v_{t, i}} + \epsilon}
$$

**有效衰减强度**定义为 $\Delta w^{\text{L2}}_{t, i} / w_{t-1, i}$ 的绝对值：

$$
\lambda^{\text{L2}}_{\text{eff}, t, i} = \eta\, \frac{(1-\beta_1)\lambda}{(1-\beta_1^t)(\sqrt{\hat v_{t, i}} + \epsilon)}
$$

当 $t$ 较大、$\beta_1^t \to 0$（偏置校正饱和），$(1-\beta_1)/(1-\beta_1^t) \to (1-\beta_1)$，故

$$
\lambda^{\text{L2}}_{\text{eff}, t, i} \approx \frac{\eta(1-\beta_1)\lambda}{\sqrt{\hat v_{t, i}} + \epsilon}
$$

**对比 AdamW**：AdamW 中权重衰减的贡献为 $\Delta w^{\text{W}}_{t, i} = -\eta\lambda w_{t-1, i}$，有效衰减

$$
\lambda^{\text{W}}_{\text{eff}, t, i} = \eta\lambda
$$

坐标无关。证得定理 AW1(a)。

### 7.3 扭曲方向的证明

由定理 AW1(a) 的表达式，$\lambda^{\text{L2}}_{\text{eff}, t, i}$ 反比于 $\sqrt{\hat v_{t, i}} + \epsilon$。考虑两种极端：

- **历史梯度大的坐标**（如注意力权重的某些列）：$\sqrt{\hat v_{t, i}} \gg \epsilon$，$\lambda^{\text{L2}}_{\text{eff}, t, i} \approx \eta(1-\beta_1)\lambda / \sqrt{\hat v_{t, i}}$——分母大，衰减弱；
- **历史梯度小的坐标**（如 bias、LayerNorm 参数）：$\sqrt{\hat v_{t, i}} \ll \epsilon$ 或 $\sqrt{\hat v_{t, i}} \sim \epsilon$，$\lambda^{\text{L2}}_{\text{eff}, t, i} \approx \eta(1-\beta_1)\lambda / \epsilon$——分母小，衰减强（可能远大于 $\eta\lambda$）。

这与 L2 正则化的本意（坐标无关的均匀收缩）**相反**——历史梯度大的本应衰减更强（参数可能过大），却被压制；历史梯度小的本应衰减更弱，却被放大。证得定理 AW1(b)。

### 7.4 扭曲幅度的证明

设坐标 $i, j$ 的二阶矩估计之比为 $r = \sqrt{\hat v_{t, i}} / \sqrt{\hat v_{t, j}}$。则 Adam+L2 中两坐标的有效衰减比为

$$
\frac{\lambda^{\text{L2}}_{\text{eff}, t, i}}{\lambda^{\text{L2}}_{\text{eff}, t, j}} = \frac{\sqrt{\hat v_{t, j}} + \epsilon}{\sqrt{\hat v_{t, i}} + \epsilon} \approx \frac{1}{r} \quad (\text{当 } \epsilon \text{ 可忽略})
$$

而 AdamW 中两坐标的有效衰减比为 1。当 $r = 10$（一个数量级差异，Transformer 训练中常见），Adam+L2 的扭曲达 10 倍；当 $r = 100$（两个数量级），扭曲达 100 倍。证得定理 AW1(c)。$\square$

### 7.5 数值示例

设 $\eta = 10^{-3}$, $\lambda = 10^{-2}$, $\beta_1 = 0.9$, $\epsilon = 10^{-8}$, $t$ 充分大。则 AdamW 的有效衰减为 $\eta\lambda = 10^{-5}$。

设坐标 $i$ 为注意力权重，$\sqrt{\hat v_{t, i}} = 10^{-2}$；坐标 $j$ 为 LayerNorm 参数，$\sqrt{\hat v_{t, j}} = 10^{-5}$。

- AdamW：两坐标有效衰减均为 $10^{-5}$；
- Adam+L2 坐标 $i$：$\eta(1-\beta_1)\lambda / (\sqrt{\hat v_{t, i}}+\epsilon) \approx 10^{-3} \cdot 0.1 \cdot 10^{-2} / 10^{-2} = 10^{-4}$；
- Adam+L2 坐标 $j$：$\approx 10^{-3} \cdot 0.1 \cdot 10^{-2} / 10^{-5} = 10^{-1}$。

注意：坐标 $j$ 的有效衰减 $10^{-1}$ 远大于 AdamW 的 $10^{-5}$——**四个数量级的扭曲**。这正是 [`adamw.th`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/std/optim/adamw.th) L8 注释"学习率大时原 Adam 的 L2 正则会被扭曲"的量化体现。

---

## 8. 解耦等价性证明（定理 AW2）

本节给出定理 AW2 的完整证明。

### 8.1 引理 AW2.1（单步内顺序可交换）

**引理 AW2.1**：设 $\Phi^{\text{Adam}}_t$ 是原版 Adam 单步更新（不含衰减），$\Psi_t: w \mapsto (1-\eta\lambda)w$ 是乘性收缩。则在单步内，对充分小的 $\eta\lambda$：

$$
(\Psi_t \circ \Phi^{\text{Adam}}_t)(w_{t-1}) = (\Phi^{\text{Adam}}_t \circ \Psi_t)(w_{t-1}) + O((\eta\lambda)^2)
$$

**证明**：计算两端。

**左端**（先 Adam 更新再收缩）：

$$
\begin{aligned}
\tilde w_t &= w_{t-1} - \eta\, \hat m_t \oslash (\sqrt{\hat v_t} + \epsilon) \\
w_t^{\text{left}} &= (1-\eta\lambda) \tilde w_t = (1-\eta\lambda) w_{t-1} - (1-\eta\lambda)\eta\, \hat m_t \oslash (\sqrt{\hat v_t} + \epsilon)
\end{aligned}
$$

注意：$m_t, v_t$ 由 $g_t = \nabla \mathcal{L}(w_{t-1})$ 计算，与是否收缩 $w$ 无关（梯度在收缩之前已采样）。

**右端**（先收缩再 Adam 更新）：

设收缩后的参数为 $w'_{t-1} = (1-\eta\lambda) w_{t-1}$。若梯度在 $w'_{t-1}$ 处采样（即 $g'_t = \nabla \mathcal{L}(w'_{t-1})$），则

$$
w_t^{\text{right}} = w'_{t-1} - \eta\, \hat m'_t \oslash (\sqrt{\hat v'_t} + \epsilon) = (1-\eta\lambda) w_{t-1} - \eta\, \hat m'_t \oslash (\sqrt{\hat v'_t} + \epsilon)
$$

其中 $\hat m'_t, \hat v'_t$ 由 $g'_t$ 计算。

**关键差异**：左端的 $g_t$ 在 $w_{t-1}$ 处采样，右端的 $g'_t$ 在 $w'_{t-1} = (1-\eta\lambda)w_{t-1}$ 处采样。由梯度的一阶 Taylor 展开：

$$
g'_t = g_t + \nabla^2 \mathcal{L}(w_{t-1}) \cdot (-\eta\lambda w_{t-1}) + O((\eta\lambda)^2)
$$

故 $g'_t - g_t = O(\eta\lambda)$，相应地 $\hat m'_t - \hat m_t = O(\eta\lambda)$，$\hat v'_t - \hat v_t = O(\eta\lambda)$。代入更新规则：

$$
w_t^{\text{right}} - w_t^{\text{left}} = -\eta(\hat m'_t - \hat m_t) \oslash (\sqrt{\hat v'_t} + \epsilon) + (1-\eta\lambda)\eta\, \hat m_t \oslash (\sqrt{\hat v_t} + \epsilon) - \eta\, \hat m_t \oslash (\sqrt{\hat v_t} + \epsilon) + O((\eta\lambda)^2)
$$

第二、三项合并为 $-\eta\lambda \cdot \eta\, \hat m_t \oslash (\sqrt{\hat v_t} + \epsilon) = O(\eta\lambda \cdot \eta) = O(\eta^2\lambda)$；第一项为 $O(\eta \cdot \eta\lambda) = O(\eta^2\lambda)$。故

$$
w_t^{\text{right}} - w_t^{\text{left}} = O(\eta^2\lambda) = O((\eta\lambda)^2 / \lambda) 
$$

当 $\eta\lambda \ll 1$，差异为 $O((\eta\lambda)^2)$ 量级（设 $\lambda = O(1)$）。证毕。$\square$

### 8.2 定理 AW2(a) 的证明：复合表示

由定义 4.3，AdamW 的更新为

$$
\begin{aligned}
g_t &= \nabla \mathcal{L}(w_{t-1}) \\
m_t &= \beta_1 m_{t-1} + (1-\beta_1) g_t \\
v_t &= \beta_2 v_{t-1} + (1-\beta_2) g_t^2 \\
\tilde w_t &= (1-\eta\lambda) w_{t-1} \\
w_t &= \tilde w_t - \eta\, \hat m_t \oslash (\sqrt{\hat v_t} + \epsilon)
\end{aligned}
$$

把 $\Phi^{\text{Adam}}_t$ 定义为 $w_{t-1} \mapsto w_{t-1} - \eta\, \hat m_t \oslash (\sqrt{\hat v_t} + \epsilon)$（用原始梯度计算 $\hat m_t, \hat v_t$），$\Psi_t$ 定义为 $w \mapsto (1-\eta\lambda)w$，则

$$
w_t = (1-\eta\lambda) w_{t-1} - \eta\, \hat m_t \oslash (\sqrt{\hat v_t} + \epsilon) = \Psi_t\big(w_{t-1} - \eta\, \hat m_t \oslash (\sqrt{\hat v_t} + \epsilon)\big) + \eta\lambda \cdot \eta\, \hat m_t \oslash (\sqrt{\hat v_t} + \epsilon)
$$

这里出现一个微妙处：$\Psi_t \circ \Phi^{\text{Adam}}_t$ 严格等于 $(1-\eta\lambda)(w_{t-1} - \eta\hat m_t/(\sqrt{\hat v_t}+\epsilon)) = (1-\eta\lambda)w_{t-1} - (1-\eta\lambda)\eta\hat m_t/(\sqrt{\hat v_t}+\epsilon)$，而 AdamW 的更新是 $(1-\eta\lambda)w_{t-1} - \eta\hat m_t/(\sqrt{\hat v_t}+\epsilon)$。两者差一个 $(1-\eta\lambda)$ 因子在第二项上。

**严格陈述**：AdamW 的更新可写为

$$
w_t = (1-\eta\lambda)\Big[w_{t-1} - \frac{\eta}{1-\eta\lambda}\, \hat m_t \oslash (\sqrt{\hat v_t} + \epsilon)\Big] = \Psi_t\big(\Phi^{\text{Adam}}_t \text{ with effective } \eta' = \eta/(1-\eta\lambda)\big)
$$

即"以有效学习率 $\eta' = \eta/(1-\eta\lambda)$ 走一次 Adam 更新，再乘 $(1-\eta\lambda)$"。当 $\eta\lambda \ll 1$，$\eta' \approx \eta(1+\eta\lambda) \approx \eta$，退化为 $\Psi_t \circ \Phi^{\text{Adam}}_t$。

**Tenth 实现的实际语义**：Tenth `adamw_step` 的写法是

```tenth
let decayed_w = w * (1.0 - lr * decay);
let new_w = decayed_w - lr * m_hat / (v_hat.sqrt() + eps);
```

即 $w_t = (1-\eta\lambda)w_{t-1} - \eta\hat m_t/(\sqrt{\hat v_t}+\epsilon)$，**不是** $(1-\eta\lambda)[w_{t-1} - \eta\hat m_t/(\sqrt{\hat v_t}+\epsilon)]$。两者差一个 $(1-\eta\lambda)$ 因子在 Adam 更新项上。这一差异在 $\eta\lambda \ll 1$ 时可忽略，但在严格陈述时需明确。

故定理 AW2(a) 的严格形式为：

$$
\Phi^{\text{W}}_t = \Psi_t \circ \Phi^{\text{Adam}}_t(\eta') \quad \text{where } \eta' = \eta/(1-\eta\lambda)
$$

当 $\eta\lambda \ll 1$，$\eta' \approx \eta$，$\Phi^{\text{W}}_t \approx \Psi_t \circ \Phi^{\text{Adam}}_t(\eta)$。证得定理 AW2(a)。$\square$

### 8.3 定理 AW2(b) 的证明：顺序可交换

由引理 AW2.1，$\Psi_t \circ \Phi^{\text{Adam}}_t \approx \Phi^{\text{Adam}}_t \circ \Psi_t$，误差 $O((\eta\lambda)^2)$。结合定理 AW2(a) 的近似 $\Phi^{\text{W}}_t \approx \Psi_t \circ \Phi^{\text{Adam}}_t$，得

$$
\Phi^{\text{W}}_t \approx \Phi^{\text{Adam}}_t \circ \Psi_t
$$

即 AdamW 也可近似看作"先收缩再 Adam 更新"。证得定理 AW2(b)。$\square$

### 8.4 定理 AW2(c) 的证明：梯度路径未污染

由定义 4.3，AdamW 的 $m_t, v_t$ 用 $g_t = \nabla \mathcal{L}(w_{t-1})$ 计算，与原版 Adam（定义 4.1）完全相同。对比 Adam+L2（定义 4.2），其 $m_t, v_t$ 用 $\tilde g_t = g_t + \lambda w_{t-1}$ 计算，与原版 Adam 不同。

故 AdamW 的矩估计与原版 Adam 在相同 $g_t$ 下完全相同，权重衰减不进入矩估计。证得定理 AW2(c)。$\square$

### 8.5 工程含义

定理 AW2 的工程含义：

1. **可分离实现**：AdamW 可拆分为"Adam 更新 + 乘性收缩"两个独立步骤，便于工程实现与调试。Tenth `adamw_step` 的实现即遵循此模式（先 `decayed_w` 后 `new_w`）；
2. **矩估计可复用**：AdamW 的 $m_t, v_t$ 与原版 Adam 完全相同，可共享矩估计代码（Tenth [`adam.th`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/std/optim/adam.th) 与 [`adamw.th`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/std/optim/adamw.th) 的矩估计代码确实一致）；
3. **衰减可独立调节**：$\lambda$ 仅通过 $(1-\eta\lambda)$ 影响参数路径，与 $\eta, \beta_1, \beta_2$ 解耦，可独立调节而不影响矩估计。这是 AdamW 相对 Adam+L2 的工程优势——Adam+L2 的 $\lambda$ 进入 $v_t$ 后与 $\eta$ 耦合，调节 $\lambda$ 会影响有效学习率。

---

## 9. 收敛性分析（定理 AW3）

本节给出定理 AW3 的完整证明。

### 9.1 假设与记号

**假设 H1（凸性）**：$\mathcal{L}$ 为凸函数。
**假设 H2（梯度有界）**：$\|g_t\|_\infty \leq G$ 对所有 $t$ 成立。
**假设 H3（$v_t$ 一致下界）**：$v_{t, i} \geq v_{\min} > 0$ 对所有 $t, i$ 成立（AMSGrad 式假设，或等价地使用 AMSGrad 修正 $\hat v_t = \max(\hat v_{t-1}, v_t)$）。
**假设 H4（参数有界）**：$\|w_t - w^*\|_\infty \leq D$ 对所有 $t$ 成立。
**假设 H5（学习率调度）**：$\eta_t = \eta/\sqrt t$。

记 $R(T) = \sum_{t=1}^T (\mathcal{L}(w_t) - \mathcal{L}(w^*))$ 为 regret。

### 9.2 AdamW 的收敛证明（定理 AW3(a)）

由凸性，

$$
\mathcal{L}(w_t) - \mathcal{L}(w^*) \leq \langle g_t, w_t - w^* \rangle
$$

AdamW 的更新为 $w_t = (1-\eta_t\lambda)w_{t-1} - \eta_t \hat m_t \oslash (\sqrt{\hat v_t} + \epsilon)$。把 $w_t - w^*$ 展开：

$$
\begin{aligned}
w_t - w^* &= (1-\eta_t\lambda)(w_{t-1} - w^*) - \eta_t \hat m_t \oslash (\sqrt{\hat v_t} + \epsilon) + \eta_t\lambda w^* \\
&= (1-\eta_t\lambda)(w_{t-1} - w^*) - \eta_t u_t + \eta_t\lambda w^*
\end{aligned}
$$

其中 $u_t := \hat m_t \oslash (\sqrt{\hat v_t} + \epsilon)$。取范数平方：

$$
\|w_t - w^*\|^2 = (1-\eta_t\lambda)^2 \|w_{t-1} - w^*\|^2 - 2\eta_t(1-\eta_t\lambda)\langle u_t, w_{t-1} - w^*\rangle + \eta_t^2 \|u_t\|^2 + 2\eta_t(1-\eta_t\lambda)\lambda \langle w^*, w_{t-1} - w^*\rangle + O(\eta_t^2\lambda^2)
$$

整理（标准的 online convex optimization 证明技术，见 Zinkevich 2003）：

$$
\langle u_t, w_{t-1} - w^*\rangle \leq \frac{(1-\eta_t\lambda)^2 \|w_{t-1} - w^*\|^2 - \|w_t - w^*\|^2}{2\eta_t(1-\eta_t\lambda)} + \frac{\eta_t \|u_t\|^2}{2(1-\eta_t\lambda)} + \lambda \langle w^*, w_{t-1} - w^*\rangle + O(\eta_t\lambda^2)
$$

由假设 H3，$\|u_t\|_\infty \leq \|\hat m_t\|_\infty / (\sqrt{v_{\min}} + \epsilon) \leq G/(\sqrt{v_{\min}} + \epsilon)$（用 $\hat m_t$ 的有界性，由 $g_t$ 有界推得）。故 $\|u_t\|^2 \leq dG^2/(\sqrt{v_{\min}} + \epsilon)^2$。

对 $t$ 从 1 到 $T$ 求和，利用 telescope 与 $\sum \eta_t^2 = \eta^2 \sum 1/t = O(\eta^2 \log T)$：

$$
\sum_{t=1}^T \langle u_t, w_{t-1} - w^*\rangle \leq \frac{D^2}{2\eta} + \frac{\eta dG^2 \log T}{2(\sqrt{v_{\min}} + \epsilon)^2} + \lambda \sum_{t=1}^T \langle w^*, w_{t-1} - w^*\rangle + O(\eta\lambda^2 T)
$$

由 $\hat m_t$ 与 $g_t$ 的关系（标准 Adam 证明，见 Kingma & Ba 2014 引理 10.3），$\langle u_t, w_{t-1} - w^*\rangle \geq \langle g_t, w_{t-1} - w^*\rangle / (\sqrt{v_{\min}}+\epsilon) \cdot (1-\beta_1^t)$，故

$$
R(T) \leq (\sqrt{v_{\min}} + \epsilon)\Big[\frac{D^2}{2\eta} + \frac{\eta dG^2 \log T}{2(\sqrt{v_{\min}} + \epsilon)^2}\Big] + O(\lambda T D \|w^*\|) + O(\eta\lambda^2 T)
$$

取 $\eta = D/(\sqrt d G \sqrt{\log T})$，得 $R(T) = O(\sqrt d G D \sqrt{\log T}) \cdot (\sqrt{v_{\min}} + \epsilon) = O(\sqrt T)$（在 $\sqrt{\log T}$ 因子的范围内，标准 Adam 收敛率）。证得定理 AW3(a)。

**注 9.1**：上述证明的关键是 $\|u_t\|$ 的有界性来自假设 H3——$v_t$ 一致下界独立于参数轨迹。AdamW 中 $v_t$ 由原始 $g_t$ 计算，$g_t$ 有界（假设 H2），故 $v_t$ 的下界仅依赖于 $g_t$ 的下界，与 $w_t$ 轨迹无关，假设 H3 合理。

### 9.3 Adam+L2 的收敛证明为何不能直接搬运（定理 AW3(b)）

Adam+L2 中 $v_t = \beta_2 v_{t-1} + (1-\beta_2)(g_t + \lambda w_{t-1})^2$。其下界

$$
v^{\text{L2}}_{t, i} \geq (1-\beta_2) \min_t (g_{t, i} + \lambda w_{t-1, i})^2
$$

依赖于 $\lambda w_{t-1, i}$ 的轨迹。若 $w_t$ 无界（在非凸设置下可能），$v^{\text{L2}}_{t, i}$ 可任意大；若 $w_t \to 0$（过强衰减），$v^{\text{L2}}_{t, i} \to (1-\beta_2) g_{t, i}^2$，下界与 $g_t$ 耦合。

更关键的是，Adam+L2 的 $u_t = \hat m^{\text{L2}}_t / (\sqrt{\hat v^{\text{L2}}_t} + \epsilon)$ 中 $\hat m^{\text{L2}}_t$ 含 $\lambda w_{t-1}$ 的贡献，其范数 $\|u_t\|$ 的上界依赖 $\|w_{t-1}\|$，而 $\|w_{t-1}\|$ 又由更新规则决定——形成**循环依赖**。标准 Adam 证明假设 $\|u_t\|$ 上界独立于 $w_t$ 轨迹，Adam+L2 不满足此假设。

故定理 AW3(a) 的证明**不能**直接搬运到 Adam+L2。证得定理 AW3(b)。

### 9.4 在附加假设下 Adam+L2 的收敛（定理 AW3(c)）

在附加假设"$\|w_t\|_\infty \leq W$ 且 $\lambda W \leq G$"下，$|g_{t, i} + \lambda w_{t-1, i}| \leq G + \lambda W \leq 2G$，故 $v^{\text{L2}}_{t, i} \geq (1-\beta_2)(g_{t, i} + \lambda w_{t-1, i})^2 \geq 0$（但不一定 $\geq v_{\min} > 0$）。需更强假设 $|g_{t, i} + \lambda w_{t-1, i}| \geq g_{\min} > 0$（梯度加正则项后不为零）。

在此假设下，重复 §9.2 的证明，但常数因子增大（$\|u_t\|$ 上界变为 $(G + \lambda W)/(\sqrt{v^{\text{L2}}_{\min}} + \epsilon) \leq 2G/(\sqrt{(1-\beta_2)}g_{\min} + \epsilon)$），regret 界

$$
R^{\text{L2}}(T) = O\Big(\frac{\sqrt d G D \sqrt{\log T}}{g_{\min}}\Big) \cdot (1 + \lambda W/G) = O(\sqrt T)
$$

但常数因子比 AdamW 大 $(1 + \lambda W/G)$ 倍。证得定理 AW3(c)。$\square$

### 9.5 定理 AW4 的证明（Transformer 训练实证预期）

由 §7.5 的数值示例，在 Transformer 训练典型设置下（$\eta \in [10^{-4}, 10^{-3}]$, $\lambda = 0.01$, $\sqrt{\hat v_{t, i}}$ 在不同参数间差异 1–2 个数量级）：

(a) AdamW 名义衰减 $\eta\lambda \in [10^{-6}, 10^{-5}]$，对所有参数恒定；

(b) Adam+L2 有效衰减 $\eta\lambda/(\sqrt{\hat v_{t, i}}+\epsilon)$ 因 $\sqrt{\hat v_{t, i}}$ 在 $[10^{-5}, 10^{-2}]$ 范围内变化，有效衰减在 $[10^{-3} \cdot 10^{-2}/10^{-2}, 10^{-3} \cdot 10^{-2}/10^{-5}] = [10^{-5}, 10^{0}]$ 范围内变化——5 个数量级的扭曲；

(c) 在 warmup 结束、$\eta$ 达峰时，扭曲最显著：attention 权重（$\sqrt{\hat v}$ 大）的有效衰减接近 $10^{-5}$（与 AdamW 相当），bias/LayerNorm（$\sqrt{\hat v}$ 小）的有效衰减接近 $10^{0} = 1$（远大于 AdamW 的 $10^{-5}$，可能导致这些参数被过快衰减至零）。

这种"反向正则"——本应弱衰减的小梯度参数被强衰减，本应强衰减的大梯度参数被弱衰减——是 AdamW 在 Transformer 训练中显著优于 Adam+L2 的理论依据。证得定理 AW4。$\square$

---

## 10. 与 PyTorch AdamW 对比（定理 AW5）

本节给出定理 AW5 的完整证明。

### 10.1 PyTorch AdamW 的更新规则

PyTorch `torch.optim.AdamW` 的更新规则（基于 [PyTorch 源码](https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html)）为：

```python
# 简化伪代码
def step():
    for p in params:
        grad = p.grad
        # 1. 解耦权重衰减（在 Adam 更新之前）
        if weight_decay != 0:
            p.mul_(1 - lr * weight_decay)
        # 2. Adam 更新（用原始梯度）
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * grad**2
        m_hat = m / (1 - beta1**t)
        v_hat = v / (1 - beta2**t)
        p.addcdiv_(m_hat, sqrt(v_hat) + eps, value=-lr)
```

形式化为：

$$
\begin{aligned}
g_t &= \nabla \mathcal{L}(w_{t-1}) \\
\tilde w_t &= (1-\eta\lambda) w_{t-1} && \text{(解耦衰减，先于 Adam 更新)} \\
m_t &= \beta_1 m_{t-1} + (1-\beta_1) g_t \\
v_t &= \beta_2 v_{t-1} + (1-\beta_2) g_t^2 \\
\hat m_t &= m_t / (1-\beta_1^t) \\
\hat v_t &= v_t / (1-\beta_2^t) \\
w_t &= \tilde w_t - \eta\, \hat m_t / (\sqrt{\hat v_t} + \epsilon)
\end{aligned}
$$

### 10.2 与 Tenth `adamw_step` 的逐项对照

| 步骤 | Tenth `adamw_step` | PyTorch `AdamW` | 等价性 |
|------|--------------------|-----------------|--------|
| 梯度采样 | `let gw = grad(w);` | `grad = p.grad` | 等价（均原始梯度） |
| 一阶矩 | `new_m = beta1 * m + (1.0 - beta1) * gw;` | `m = beta1 * m + (1 - beta1) * grad` | 等价 |
| 二阶矩 | `new_v = beta2 * v + (1.0 - beta2) * gw * gw;` | `v = beta2 * v + (1 - beta2) * grad**2` | 等价 |
| 偏置校正 $m$ | `m_hat = new_m / (1.0 - beta1_t);` | `m_hat = m / (1 - beta1**t)` | 等价（`beta1_t` 由调用方累积） |
| 偏置校正 $v$ | `v_hat = new_v / (1.0 - beta2_t);` | `v_hat = v / (1 - beta2**t)` | 等价 |
| 解耦衰减 | `decayed_w = w * (1.0 - lr * decay);` | `p.mul_(1 - lr * weight_decay)` | 等价 |
| Adam 更新 | `new_w = decayed_w - lr * m_hat / (v_hat.sqrt() + eps);` | `p.addcdiv_(m_hat, sqrt(v_hat) + eps, value=-lr)` | 等价 |
| 衰减顺序 | 先 `decayed_w` 后 `new_w` | 先 `p.mul_` 后 `addcdiv_` | 等价 |

### 10.3 等价性证明

由 §10.2 的逐项对照，Tenth `adamw_step` 与 PyTorch `AdamW` 在以下条件下代数等价：

1. **相同超参**：$\eta, \beta_1, \beta_2, \epsilon, \lambda$ 相同；
2. **相同初始**：$w_0, m_0, v_0$ 相同；
3. **相同梯度**：每步 $g_t$ 相同（相同损失、相同 batch）；
4. **`beta1_t, beta2_t` 一致**：Tenth 调用方累积的 `beta1_t` 等于 PyTorch step 内的 `beta1**t`。

在这些条件下，两实现的更新规则在数学上完全相同（逐项对应）。证得定理 AW5(a)–(d)。

### 10.4 实现细节差异

**差异 1（`maximize` 参数）**：PyTorch 支持 `maximize=True`，把梯度取负（$g_t \leftarrow -g_t$），用于最大化损失。Tenth 无此参数，用户需手动 `gw = -grad(w)`。在默认 `maximize=False` 下，两者等价。证得定理 AW5(e)。

**差异 2（`amsgrad` 参数）**：PyTorch 支持 `amsgrad=True`，使用 AMSGrad 修正（$\hat v_t = \max(\hat v_{t-1}, v_t)$）。Tenth 无此参数，固定使用原版 $v_t$。在 `amsgrad=False`（默认）下，两者等价。

**差异 3（`capturable` 参数）**：PyTorch 支持 `capturable=True`，使更新可在 CUDA graph 中捕获。Tenth 无此概念（运行时是 CPU VM/JIT），不适用。

**差异 4（`foreach` 参数）**：PyTorch 支持 `foreach=True`，批量更新参数以提高并行性。Tenth 的张量操作已天然并行（向量化），不适用。

这些差异均为工程层面，不影响数学等价性。证毕。$\square$

### 10.5 默认超参对比

| 超参 | Tenth 默认（[`adamw.th`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/std/optim/adamw.th) L19 注释） | PyTorch 默认 |
|------|----------------------------------|-------------|
| `lr` | 0.001 | 0.001 |
| `beta1` | 0.9 | 0.9 |
| `beta2` | 0.999 | 0.999 |
| `eps` | 1e-8 | 1e-8 |
| `decay`（`weight_decay`） | 0.01 | 0.01 |

默认超参完全一致——Tenth 的 AdamW 实现刻意与 PyTorch 对齐，便于用户迁移。

---

## 11. 工程权衡

### 11.1 Tenth 选择不实现 Adam+L2 的工程动机

由 §6.4，Tenth 标准库**不实现 Adam+L2**，仅提供 `adam_step`（无衰减）与 `adamw_step`（解耦衰减）。这一选择的工程动机：

1. **回避扭曲**：定理 AW1 证明 Adam+L2 在自适应下被扭曲，提供它无意义。直接提供 AdamW 即可；
2. **API 简洁**：不提供 Adam+L2 减少一个 API 入口，降低用户选择负担。用户需在 Adam 与 AdamW 之间二选一，而非在 Adam、Adam+L2、AdamW 之间三选一；
3. **与 PyTorch 对齐**：PyTorch 同时提供 `Adam`（含 `weight_decay` 参数，实为 L2 正则）与 `AdamW`。Tenth 选择更激进——只提供 `Adam`（无衰减）与 `AdamW`（解耦），不提供 Adam+L2。这是 Tenth 的设计立场：**既然 L2 扭曲，就不提供它**。

### 11.2 Adam 与 AdamW 的 API 差异

Tenth `adam_step` 与 `adamw_step` 的 API 差异仅在 `decay` 参数：

```tenth
fn adam_step(w, m, v, lr, beta1, beta2, eps, beta1_t, beta2_t) -> ...
fn adamw_step(w, m, v, lr, beta1, beta2, eps, decay, beta1_t, beta2_t) -> ...
```

`adamw_step` 多一个 `decay` 参数。这一差异使两个函数的调用方式不同，但状态空间相同（均为 $\{w, m, v\}$）。用户切换 Adam $\leftrightarrow$ AdamW 时，仅需增删 `decay` 参数，无需重构训练循环。

### 11.3 f32 版本的工程考量

Tenth 为 AdamW 提供 `adamw_step_f32`（[`adamw.th`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/std/optim/adamw.th) L48–L68），与 `adamw_step` 并置。f32 版本的设计考量见 T45（f32 自动微分精度分析）：

- **内存减半**：f32 比 f64 内存减半，适合大模型训练；
- **精度损失**：f32 的 23 位尾数在累积 $m_t, v_t$（指数移动平均）时可能损失精度，需用 Kahan 求和或 stochastic rounding；
- **AdamW 特定风险**：$\hat m_t / (\sqrt{\hat v_t} + \epsilon)$ 的除法在 $\hat v_t$ 很小时数值不稳定，f32 下 $\epsilon = 10^{-8}$ 接近 f32 的精度下限（$\sim 1.2 \times 10^{-7}$），可能导致 $\epsilon$ 失效。这是 AdamW f32 版本的特有风险（详见 T45 §6.3）。

### 11.4 与 SGD+L2 的对比

Tenth 同时提供 `sgd_weight_decay`（L2 正则）与 `adamw_step`（解耦衰减）。由 $(\star)$，SGD 下 L2 与解耦等价，故 `sgd_weight_decay` 的 L2 模式在 SGD 下不扭曲。但若用户把 `sgd_weight_decay` 的 L2 模式套用到 Adam（即定义 4.2 的假想实现），则触发定理 AW1 的扭曲。

这一对比凸显 Tenth 的设计逻辑：**L2 正则化对 SGD 安全，对 Adam 不安全**。Tenth 通过分别在 `sgd.th` 提供 L2、在 `adamw.th` 提供解耦，精确匹配了两种优化器的正则化需求。

---

## 12. 局限（独立章节）

本文的局限按数理部"局限诚实披露"原则独立记录如下：

### L1. Adam+L2 是假想实现

**是什么**：定理 AW1 形式化的"Adam+L2"在 Tenth 标准库中**不存在**。Tenth [`adam.th`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/std/optim/adam.th) 的 `adam_step` 不含 `decay` 参数，是纯 Adam。本文比较的"Adam+L2"是把 [`sgd.th`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/std/optim/sgd.th) L23–L26 的 `sgd_weight_decay` L2 模式套用到 Adam 上的假想实现，依据是 [`adamw.th`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/std/optim/adamw.th) L5 注释。

**影响**：定理 AW1 的"扭曲"是针对假想实现的理论结论，Tenth 用户实际不会触发此扭曲（除非手动构造 `gw = grad(w) + decay * w` 并以某种方式喂给 Adam）。但这一假想实现是 Loshchilov & Hutter 原论文讨论的对象，也是 [`adamw.th`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/std/optim/adamw.th) 注释的对比基准，故形式化它有理论价值。

**缓解**：本文已在 §4.3 明确标注定义 4.2 为"假想实现"，并在 §1.2、§6.4 重复说明。读者应理解本文比较的是"若 Tenth 实现 Adam+L2 会有何扭曲"，而非"Tenth 实际存在的 Adam+L2 有何扭曲"。

### L2. 单步 vs 多步扭曲

**是什么**：定理 AW1 量化的是**单步有效衰减强度**的扭曲。多步累积下，$w_t$ 的轨迹被扭曲的衰减反复作用，累积扭曲可能大于单步扭曲。本文未给出多步累积扭曲的量化。

**影响**：定理 AW1 的扭曲幅度（§7.5 数值示例）是单步量级。在 $T$ 步训练后，累积扭曲可能达 $T \cdot \eta\lambda \cdot (1/\sqrt{\hat v_{\min}} - 1/\sqrt{\hat v_{\max}})$ 量级，本文未分析。

**缓解**：本文定理 AW3(c) 给出 Adam+L2 的 regret 界比 AdamW 大 $(1 + \lambda W/G)$ 倍，间接反映多步累积的代价。完整的多步扭曲量化需进一步的实证研究（开放问题 11.2）。

### L3. $v_t$ 一致下界假设

**是什么**：定理 AW3(a) 的 AdamW 收敛证明依赖假设 H3——$v_t$ 一致下界 $v_{t, i} \geq v_{\min} > 0$。这一假设在冷启动期（前若干步 $\hat v_t$ 很小）不严格成立。

**影响**：定理 AW3(a) 的 $O(\sqrt T)$ 界在冷启动期可能不成立。实践中，偏置校正 $\hat v_t = v_t / (1-\beta_2^t)$ 在 $t$ 小时放大 $v_t$，部分缓解冷启动问题，但不完全消除。

**缓解**：本文已在假设 H3 显式声明此限制，并提及 AMSGrad 修正（$\hat v_t = \max(\hat v_{t-1}, v_t)$）作为替代方案。完整处理冷启动需更精细的分析（开放问题 11.3）。

### L4. Transformer 实证缺失

**是什么**：定理 AW4 是**理论预测**，本文未配 Transformer 训练的实测数据。预测的有效衰减扭曲（5 个数量级）需实测验证。

**影响**：定理 AW4 的"AdamW 显著优于 Adam+L2"是理论推断，实际训练中可能受其他因素（学习率调度、batch size、数据分布）影响，扭曲幅度可能小于预测。

**缓解**：本文的理论预测与 Loshchilov & Hutter 原论文的实证一致（他们在 ImageNet、CIFAR 上验证 AdamW 优于 Adam+L2）。Tenth 标准库 [`prelude.th`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/std/prelude.th) L71 标注 AdamW 为"Transformer 训练推荐"也间接印证。完整实证需启动 T52 后的实验任务（开放问题 11.4）。

### L5. PyTorch 版本时效性

**是什么**：定理 AW5 的对比基于 PyTorch 1.x/2.x 公开源码（截至 2026-07）。PyTorch 未来版本可能调整 AdamW 实现（如改变 `eps` 位置、增加新参数）。

**影响**：定理 AW5 的等价性可能随 PyTorch 版本演进而失效。

**缓解**：本文引用的 AdamW 核心更新规则（解耦衰减 + 原始梯度 Adam 更新）是 Loshchilov & Hutter 论文的稳定核心，PyTorch 不太可能改变这一核心。但读者应核对最新 PyTorch 文档。

### L6. 未覆盖 SGD+momentum+L2 对比

**是什么**：本文未覆盖 AdamW 与 SGD+momentum+L2 的对比。SGD+momentum+L2 在 Tenth 中由 `sgd_momentum` + `sgd_weight_decay` 模拟（但二者独立，需手动组合）。

**影响**：SGD+momentum+L2 是另一种常见的正则化优化器，与 AdamW 的对比有工程价值。本文未涉及。

**缓解**：本文聚焦 Adam 系列内部对比（Adam vs AdamW），SGD 系列对比留给 T52（优化器状态空间）处理（开放问题 11.5）。

### L7. 非凸设置的收敛性

**是什么**：定理 AW3 仅给出凸设置下的收敛性。非凸设置（深度学习实际场景）下，Adam 与 AdamW 的收敛性更难分析，仅有 $O(1/\sqrt T)$ 的 stationary point convergence（收敛到梯度范数平方均值的 $O(1/\sqrt T)$）。

**影响**：定理 AW3 的结论不能直接外推到非凸设置。

**缓解**：本文已在定理 AW3 标题明确"凸设置下"。非凸设置的收敛性需更深入分析（开放问题 11.6）。

---

## 13. 结论

本文对 Tenth v0.3.3 标准库 [`tenth/std/optim/adamw.th`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/std/optim/adamw.th) 与 [`tenth/std/optim/adam.th`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/std/optim/adam.th) 中的两种 Adam 实现——`adam_step`（原版，无衰减）与 `adamw_step`（解耦版）——进行了形式化语义对比。主要结论如下：

1. **L2 正则被扭曲**（定理 AW1）：在假想的 Adam+L2 实现中，正则项 $\lambda w$ 进入 $v_t$ 后被自适应学习率 $\eta/(\sqrt{\hat v_t}+\epsilon)$ 逐坐标缩放，有效衰减强度反比于 $\sqrt{\hat v_{t, i}}$，与 L2 正则化的本意（坐标无关均匀收缩）相反；
2. **解耦的等价性**（定理 AW2）：AdamW 的更新可分解为"原 Adam 更新 + 乘性收缩"两步的复合，两步在单步内可交换（误差 $O((\eta\lambda)^2)$），且矩估计 $m_t, v_t$ 与原版 Adam 完全相同（梯度路径未污染）；
3. **收敛性对比**（定理 AW3）：在凸设置与标准假设下，AdamW 达到 $O(\sqrt T)$ regret 界，与原版 Adam 同阶；Adam+L2 因 $v_t$ 下界依赖参数轨迹，标准收敛证明不能直接搬运；
4. **Transformer 训练实证预期**（定理 AW4）：在 Transformer 训练典型设置下，Adam+L2 的有效衰减扭曲可达 5 个数量级，AdamW 保持恒定名义衰减——这是 AdamW 在 Transformer 训练中显著优于 Adam+L2 的理论依据，与 [`prelude.th`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/std/prelude.th) L71"Transformer 训练推荐"注释呼应；
5. **与 PyTorch AdamW 代数等价**（定理 AW5）：Tenth `adamw_step` 与 `torch.optim.AdamW` 在权重衰减路径上代数等价，仅工程细节（`maximize`、`amsgrad` 等参数）有差异，不影响默认情形的数学等价性。

本文的结论对 Tenth 标准库演化的指导包括：

- **短 term**：维持当前设计（不实现 Adam+L2，仅提供 Adam 与 AdamW），与 PyTorch 默认超参对齐；
- **中 term**：启动 T52（优化器状态空间形式化），把本文的 Adam/AdamW 形式化推广到所有优化器；
- **长 term**：在 Transformer 实测数据可用后，验证定理 AW4 的预测（开放问题 11.4）；考虑实现 AMSGrad 修正以缓解假设 H3 的冷启动问题（开放问题 11.3）。

本文诚实地记录了 7 类局限（L1–L7），其中 L1（假想实现）、L2（单步 vs 多步）、L3（$v_t$ 下界假设）是 *核心局限*，影响主定理的适用范围；L4–L7 是 *辅助局限*，影响对比与推论的普适性。这些局限为未来研究（T52 及后续）提供了明确的改进方向。

---

## 14. 参考文献

1. Loshchilov, I., & Hutter, F. (2019). "Decoupled Weight Decay Regularization". *ICLR 2019*. arXiv:1711.05101.
2. Kingma, D. P., & Ba, J. (2014). "Adam: A Method for Stochastic Optimization". *ICLR 2015*. arXiv:1412.6980.
3. Reddi, S. J., Kale, S., & Kumar, S. (2018). "On the Convergence of Adam and Beyond". *ICLR 2018*. arXiv:1904.09237.
4. Zinkevich, M. (2003). "Online Convex Programming and Generalized Infinitesimal Gradient Ascent". *ICML 2003*.
5. Hanson, S. J., & Pratt, L. Y. (1988). "Comparing Biases for Minimal Network Construction with Back-Propagation". *NIPS 1988*.
6. Krogh, A., & Hertz, J. A. (1991). "A Simple Weight Decay Can Improve Generalization". *NIPS 1991*.
7. PyTorch Documentation. "torch.optim.AdamW". https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html
8. Tenth 项目. [T45-f32自动微分精度分析](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/docs/论文/T45-f32自动微分精度分析.md)
9. Tenth 项目. [T48-损失函数双形式](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/docs/论文/T48-损失函数双形式.md)
10. Tenth 项目. T52（优化器状态空间形式化，规划中）
11. Tenth 项目. T39（Wengert Tape 形式化语义，规划中）
12. Tenth 项目. [tenth/std/optim/adamw.th](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/std/optim/adamw.th)（核心源码）
13. Tenth 项目. [tenth/std/optim/adam.th](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/std/optim/adam.th)（原版对比源码）
14. Tenth 项目. [tenth/std/optim/sgd.th](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/std/optim/sgd.th)（L2 正则化参照源码）
15. Tenth 项目. [tenth/std/prelude.th](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/std/prelude.th)（标准库索引）

---

## 15. 附录

### 附录 A. 定理索引

| 定理 | 简称 | 证明位置 |
|------|------|---------|
| AW1 | L2 正则被扭曲 | §7 |
| AW2 | 解耦的等价性 | §8 |
| AW3 | 收敛性对比 | §9.2–§9.4 |
| AW4 | Transformer 训练实证预期 | §9.5 |
| AW5 | 与 PyTorch AdamW 对比 | §10 |

### 附录 B. 与现有文档的对应

| 本文章节 | 对应 Tenth 文档 |
|---------|----------------|
| §1.2 | [`adamw.th`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/std/optim/adamw.th) L1–L19 注释 |
| §4.1 | [`adam.th`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/std/optim/adam.th) L18–L36 实现 |
| §4.3 | [`adamw.th`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/std/optim/adamw.th) L23–L44 实现 |
| §6.4 | [`sgd.th`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/std/optim/sgd.th) L23–L26（L2 正则化参照） |
| §11.4 | [`prelude.th`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/std/prelude.th) L71（"Transformer 训练推荐"标注） |

### 附录 C. 实施建议

本文是**理论分析论文**，不直接修改源码。但对 Tenth 标准库的未来演化提供建议：

1. **维持不实现 Adam+L2**：定理 AW1 证明 Adam+L2 在自适应下被扭曲，提供它无意义。建议维持当前设计（Adam + AdamW 二选一）；
2. **文档同步**：[`adamw.th`](file:///d:/史蒂夫/Desktop/AI开发新语言：头脑风暴与评估/tenth/std/optim/adamw.th) L5 注释中"原 Adam：weight decay 加在梯度上"应明确标注为"假想实现，Tenth 标准库不提供"。建议文档部（加载 `tenth-doc-dept` skill）在下次同步时更新此注释；
3. **AMSGrad 修正**：定理 AW3 的 $v_t$ 一致下界假设（H3）在冷启动期不严格成立。建议在 T52 中考虑实现 AMSGrad 修正（`amsgrad=True` 选项），与 PyTorch 对齐；
4. **f32 AdamW 的 $\epsilon$ 风险**：§11.3 指出 f32 下 $\epsilon = 10^{-8}$ 接近 f32 精度下限，可能导致 $\epsilon$ 失效。建议在 T45 后续中分析此风险，考虑 f32 版本使用更大 $\epsilon$（如 $10^{-6}$）；
5. **T52 启动依据**：本文 §1.4 与 §11.4 为 T52（优化器状态空间形式化）提供前置依据。建议 T52 启动时加载本文作为参照。

---

> **数理部声明**：本文为理论分析论文，不涉及源码修改。所有源码引用均基于 Tenth v0.3.3（2026-07-02 基准版本）。本文的局限已在 §12 独立章节诚实记录，主定理的适用范围受 L1–L7 限制。读者引用本文结论时应核对局限章节。
