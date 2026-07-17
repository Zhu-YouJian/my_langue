//! Tape 数据结构定义：FloatElem trait、TapeNode、TapeOp 枚举。
//!
//! 从 `autodiff.rs` 拆分而来（T3c 架构重构），保持原有可见性与语义不变。

use std::rc::Rc;
use std::cell::RefCell;
use ndarray::ArrayD;
use crate::runtime::tensor::{Tensor, TensorData};
use crate::hir::types::BaseType;

// ── Float element trait ────────────────────────────────────────────────

/// 抽象 f32/f64 的公共运算，让 backward 可按 node.dtype 分发。
/// 阶段 4：autodiff backward f32 化，消除策略 B，实现真正的 f32 反向传播。
pub(super) trait FloatElem: 'static + Copy + Send + Sync +
    ndarray::ScalarOperand + ndarray::LinalgScalar +
    std::ops::Add<Output=Self> + std::ops::Sub<Output=Self> +
    std::ops::Mul<Output=Self> + std::ops::Div<Output=Self> +
    std::ops::AddAssign + std::ops::SubAssign + std::ops::MulAssign + std::ops::DivAssign +
    std::ops::Neg<Output=Self> +
    PartialOrd +
    std::iter::Sum<Self> {
    fn from_f64(x: f64) -> Self;
    fn to_f64(self) -> f64;
    fn sqrt_(self) -> Self;
    fn tanh_(self) -> Self;
    fn from_tensor_data(td: &TensorData) -> ArrayD<Self>;
    fn into_tensor_data(arr: ArrayD<Self>) -> TensorData;
}

impl FloatElem for f32 {
    fn from_f64(x: f64) -> Self { x as f32 }
    fn to_f64(self) -> f64 { self as f64 }
    fn sqrt_(self) -> Self { self.sqrt() }
    fn tanh_(self) -> Self { self.tanh() }
    fn from_tensor_data(td: &TensorData) -> ArrayD<f32> {
        match td {
            TensorData::F32(a) => a.clone(),
            TensorData::F64(a) => a.mapv(|v| v as f32),
            // Phase 2: F16/BF16 转 f32（dispatch_float! 将 F16/BF16 走 f32 路径）
            TensorData::F16(a) => a.mapv(|v| v.to_f32()),
            TensorData::BF16(a) => a.mapv(|v| v.to_f32()),
        }
    }
    fn into_tensor_data(arr: ArrayD<f32>) -> TensorData { TensorData::F32(arr) }
}

impl FloatElem for f64 {
    fn from_f64(x: f64) -> Self { x }
    fn to_f64(self) -> f64 { self }
    fn sqrt_(self) -> Self { self.sqrt() }
    fn tanh_(self) -> Self { self.tanh() }
    fn from_tensor_data(td: &TensorData) -> ArrayD<f64> {
        match td {
            TensorData::F64(a) => a.clone(),
            TensorData::F32(a) => a.mapv(|v| v as f64),
            // Wave 2: F16/BF16 转 f64
            TensorData::F16(a) => a.mapv(|v| v.to_f64()),
            TensorData::BF16(a) => a.mapv(|v| v.to_f64()),
        }
    }
    fn into_tensor_data(arr: ArrayD<f64>) -> TensorData { TensorData::F64(arr) }
}

/// 按 node.dtype 分发到 f32 或 f64 路径。
/// `$E` 是类型别名（f32 或 f64），`$body` 是使用 `E` 的代码块。
/// 块内的 `?` 和 `return` 会从外层函数 early-return（语义与未分发版本一致）。
/// Phase 2：F16/BF16 走 f32 路径（F32 中间累加策略，AMP）。
/// F16/BF16 精度低（F16: 10 位尾数，BF16: 7 位尾数），用 F32 中间表示计算
/// 可避免溢出和精度损失，最终 grad 写回时由 acc_grad 转 F32 buffer。
macro_rules! dispatch_float {
    ($dtype:expr, $E:ident, $body:block) => {{
        match $dtype {
            BaseType::F32 | BaseType::F16 | BaseType::BF16 => {
                type E = f32;
                $body
            }
            _ => {
                type E = f64;
                $body
            }
        }
    }};
}
pub(super) use dispatch_float;

// ── Tape node ─────────────────────────────────────────────────────────

/// A node in the computation graph.
#[derive(Debug, Clone)]
pub struct TapeNode {
    /// Unique node id (index into Tape::nodes).
    pub id: usize,
    /// The operation that produced this node.
    pub op: TapeOp,
    /// IDs of the input nodes (these are the *nodes* that fed this op).
    pub inputs: Vec<usize>,
    /// `Rc` references to the input *tensors* — kept alive so backward
    /// can read their data and write gradients.
    pub input_tensors: Vec<Rc<RefCell<Tensor>>>,
    /// 辅助整数参数（如 Scatter/Gather 的 dim；其他算子默认 0）。
    pub aux: usize,
    /// 前向运算的 dtype。backward 按此 dtype 分发到 f32/f64 路径。
    /// 由 result tensor 的 dtype 决定（反映前向实际产出的 dtype）。
    pub dtype: BaseType,
}

// ── Operations ────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub enum TapeOp {
    /// Leaf parameter (no inputs).  Gradient accumulated here.
    Input,
    /// Element-wise addition (with broadcasting).
    Add,
    /// Element-wise subtraction.
    Sub,
    /// Element-wise multiplication.
    Mul,
    /// Element-wise division.
    Div,
    /// Negation: -a
    Neg,
    /// ReLU: max(0, a)
    ReLU,
    /// Matrix multiplication: a @ b
    MatMul,
    /// Batched matrix multiplication: (B, M, K) @ (B, K, N) -> (B, M, N).
    /// Backward: d_a = bmm(grad, b^T), d_b = bmm(a^T, grad)（沿 batch 循环）。
    BatchedMatMul,
    /// Transpose last two dims.
    Transpose,
    /// Sum over all elements → scalar.
    Sum,
    /// Mean over all elements → scalar.
    Mean,
    /// Exponential: exp(a)
    Exp,
    /// Natural log: ln(a)
    Log,
    /// Sigmoid: 1 / (1 + exp(-a))
    Sigmoid,
    /// Softmax (over last dim, treated as flatten).
    Softmax,
    /// Cross-entropy loss: -sum(target * log(softmax(logits))).
    /// Forward stores softmax(logits); backward = softmax - target.
    CrossEntropy,
    /// Dropout: randomly zeroes elements during training.
    /// Forward stores the mask; backward multiplies gradient by mask.
    Dropout,
    /// 2D convolution (im2col + matmul).
    /// input_tensors = [input, weight, im2col_result, output]
    Conv2D,
    /// Batch normalization.
    /// Forward stores normalized x, std, gamma, beta; backward is standard BN grad.
    BatchNorm,
    /// Layer normalization (over last dim).
    /// input_tensors = [input, gamma, beta, x_hat, std_inv, result]
    LayerNorm,
    /// GELU activation (tanh approximation).
    /// input_tensors = [input, result]
    Gelu,
    /// Element-wise select: result = cond ? then : else.
    /// cond is non-differentiable (bool semantics, encoded as f64 0.0/1.0).
    /// inputs = [then_id?, else_id?] (cond 阻断链式传播，不写入 inputs)
    /// input_tensors = [cond, then, else, result]
    Select,
    /// Element-wise absolute value: |a|.
    /// Backward: d|x|/dx = sign(x)，x=0 处取 0（次梯度中点）。
    Abs,
    /// Scatter: out = base.clone(); out[dim][index[idx]] = src[idx]（不可变语义，PyTorch 对齐）。
    /// 支持任意 dim + 多维 index/src。index 不可微。
    /// input_tensors = [base, src, index, result]
    /// inputs = [base_id, src_id]（index 阻断链式传播，不写入 inputs）
    /// dim 存于 TapeNode.aux
    Scatter,
    /// Gather: out[i,j,...] = base[index[i,j,...], j, ...]（沿 dim 维按 index 取值，与 PyTorch 对齐）。
    /// index 不可微；out.shape == index.shape。
    /// input_tensors = [base, index, result]
    /// inputs = [base_id]（index 阻断链式传播，不写入 inputs）
    Gather,
    /// Reshape: result = input.reshape(target_shape)（元素数不变，仅重排）。
    /// input_tensors = [input, result]（原始 shape 从 input.shape() 读取）
    /// backward: d_input = grad.reshape(input.shape())
    Reshape,
    /// MaskedFill: result = input.masked_fill(mask, value)（mask=true 位置覆盖为 value）。
    /// mask 不可微（bool 语义，0/1 张量），inputs = [input_id]
    /// input_tensors = [input, mask, result]
    /// backward: d_input = grad * (1 - mask)（被覆盖位置不传梯度）
    MaskedFill,
    /// MaxPool2D: result = input.max_pool2d(kH, kW, sH, sW, pH, pW)。
    /// input_tensors = [input, result]（不含 argmax_mask；backward 时从 input 重新计算 argmax，
    /// 避免 mask 在重叠 window 下产生歧义——同一 input 位置可能是多个 window 的 argmax）
    /// inputs = [input_id]
    /// aux 编码 6 个 pool 参数（kh, kw, sh, sw, ph, pw，各 10 bits，见 Tape::encode_pool_params）
    /// backward: 根据 input 数据重新计算 argmax，将 d_output 路由到 argmax 位置
    MaxPool2D,
    /// AvgPool2D: result = input.avg_pool2d(kH, kW, sH, sW, pH, pW)。
    /// input_tensors = [input, result]
    /// inputs = [input_id]
    /// aux 编码 6 个 pool 参数（同 MaxPool2D）
    /// backward: d_input[window 内每个有效位置] += d_output / valid_count（均分到窗口，
    /// count_include_pad=False 语义，padding 位置不分梯度）
    AvgPool2D,
    /// 自定义可微算子（PROJ-006）。
    /// `op_id` 是 `CustomOpRegistry::register` 分配的 id，可在 registry 中查到
    /// 用户的 `CustomBackward` 实现（含 forward/backward/op_class/name）。
    ///
    /// 约定：`input_tensors = [input1, ..., inputN, result]`（最后一个为输出）。
    /// `inputs = [input1_id, ..., inputN_id]`（每个输入对应上游节点 id）。
    ///
    /// backward：通过 `CustomOpRegistry::get(op_id)` 查找用户实现，调用
    /// `backward(&[&input1, ..., &inputN], &grad)` 获取各输入梯度；
    /// 运行时强制校验梯度数量与 shape（与对应输入一致）。
    ///
    /// TapeOpClass 分类由用户在注册时通过 `CustomBackward::op_class()` 声明，
    /// `classify_tape_op` 通过 registry 查询（论证 4 保证 T7 完备性不破坏）。
    Custom(usize),
}
