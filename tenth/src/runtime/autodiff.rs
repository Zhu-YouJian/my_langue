//! Tensor-level automatic differentiation via a Wengert tape.
//!
//! Records operations on `Tensor`s during forward execution, then replays
//! the chain rule backward to populate each parameter tensor's `.grad` field.

use std::rc::Rc;
use std::cell::RefCell;
use ndarray::{ArrayD, IxDyn};
use super::tensor::{Tensor, TensorData};
use crate::hir::types::BaseType;

// ── Float element trait ────────────────────────────────────────────────

/// 抽象 f32/f64 的公共运算，让 backward 可按 node.dtype 分发。
/// 阶段 4：autodiff backward f32 化，消除策略 B，实现真正的 f32 反向传播。
trait FloatElem: 'static + Copy + Send + Sync +
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
        }
    }
    fn into_tensor_data(arr: ArrayD<f64>) -> TensorData { TensorData::F64(arr) }
}

/// 按 node.dtype 分发到 f32 或 f64 路径。
/// `$E` 是类型别名（f32 或 f64），`$body` 是使用 `E` 的代码块。
/// 块内的 `?` 和 `return` 会从外层函数 early-return（语义与未分发版本一致）。
macro_rules! dispatch_float {
    ($dtype:expr, $E:ident, $body:block) => {{
        match $dtype {
            BaseType::F32 => {
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
}

// ── Tape ──────────────────────────────────────────────────────────────

pub struct Tape {
    nodes: Vec<TapeNode>,
    counter: usize,
}

impl Tape {
    pub fn new() -> Self {
        Tape { nodes: Vec::new(), counter: 0 }
    }

    /// Number of recorded nodes.
    pub fn len(&self) -> usize { self.nodes.len() }

    /// 按节点 id 取得 TapeNode 引用（护城河 F：relation_debugger 使用）。
    /// 节点 id 与 self.nodes 索引对齐（不变量：node.id == self.nodes 索引）。
    pub fn node(&self, id: usize) -> Option<&TapeNode> {
        self.nodes.get(id)
    }

    /// 取得所有节点（只读视图，护城河 F：relation_debugger 使用）。
    pub fn nodes(&self) -> &[TapeNode] {
        &self.nodes
    }

    /// Register a leaf (parameter) tensor.  Returns the node id.
    pub fn input(&mut self, tensor: Rc<RefCell<Tensor>>) -> usize {
        let dtype = tensor.borrow().dtype;
        let id = self.next_id();
        self.nodes.push(TapeNode {
            id,
            op: TapeOp::Input,
            inputs: vec![],
            input_tensors: vec![tensor],
            aux: 0,
            dtype,
        });
        id
    }

    /// Record a unary operation.
    /// `input_id` — upstream node id (for chain-rule traversal).
    /// `input_tensor` — the tensor that was the *input* to this op (needed
    ///   by backward to read saved values).
    /// `result` — the output tensor (the forward result).
    pub fn unary(&mut self, op: TapeOp, input_id: usize, input_tensor: Rc<RefCell<Tensor>>, result: Rc<RefCell<Tensor>>) -> usize {
        let dtype = result.borrow().dtype;
        let id = self.next_id();
        self.nodes.push(TapeNode {
            id,
            op,
            inputs: vec![input_id],
            input_tensors: vec![input_tensor, result],
            aux: 0,
            dtype,
        });
        id
    }

    /// Record a binary operation.
    /// `a_id` / `b_id` — upstream node ids.
    /// `a` / `b` — the actual input tensors (for reading values in backward).
    /// `result` — the output tensor.
    pub fn binary(&mut self, op: TapeOp, a_id: usize, b_id: usize, a: Rc<RefCell<Tensor>>, b: Rc<RefCell<Tensor>>, result: Rc<RefCell<Tensor>>) -> usize {
        let dtype = result.borrow().dtype;
        let id = self.next_id();
        self.nodes.push(TapeNode {
            id,
            op,
            inputs: vec![a_id, b_id],
            input_tensors: vec![a, b, result],
            aux: 0,
            dtype,
        });
        id
    }

    /// Record a unary operation that has no upstream tape node
    /// (e.g. a leaf tensor is passed directly).
    pub fn unary_direct(&mut self, op: TapeOp, input_tensor: Rc<RefCell<Tensor>>, result: Rc<RefCell<Tensor>>) -> usize {
        let dtype = result.borrow().dtype;
        let id = self.next_id();
        self.nodes.push(TapeNode {
            id,
            op,
            inputs: vec![],
            input_tensors: vec![input_tensor, result],
            aux: 0,
            dtype,
        });
        id
    }

    /// Record a cross-entropy loss node.
    /// `logits_id` — upstream node id for the logits tensor.
    /// `softmax` — pre-computed softmax(logits) (needed by backward).
    /// `target` — one-hot target tensor.
    /// `result` — scalar loss tensor.
    pub fn cross_entropy(
        &mut self,
        logits_id: usize,
        logits: Rc<RefCell<Tensor>>,
        softmax: Rc<RefCell<Tensor>>,
        target: Rc<RefCell<Tensor>>,
        result: Rc<RefCell<Tensor>>,
    ) -> usize {
        let dtype = result.borrow().dtype;
        let id = self.next_id();
        self.nodes.push(TapeNode {
            id,
            op: TapeOp::CrossEntropy,
            inputs: vec![logits_id],
            input_tensors: vec![logits, softmax, target, result],
            aux: 0,
            dtype,
        });
        id
    }

    /// Record a batchnorm node.
    pub fn batchnorm(
        &mut self, x_id: usize, x: Rc<RefCell<Tensor>>,
        gamma: Rc<RefCell<Tensor>>, beta: Rc<RefCell<Tensor>>,
        x_hat: Rc<RefCell<Tensor>>, std_inv: Rc<RefCell<Tensor>>,
        result: Rc<RefCell<Tensor>>,
    ) -> usize {
        let dtype = result.borrow().dtype;
        let id = self.next_id();
        self.nodes.push(TapeNode {
            id,
            op: TapeOp::BatchNorm,
            inputs: vec![x_id],
            input_tensors: vec![x, gamma, beta, x_hat, std_inv, result],
            aux: 0,
            dtype,
        });
        id
    }

    /// Record a layernorm node.
    /// input_tensors = [input, gamma, beta, x_hat, std_inv, result]
    pub fn layernorm(
        &mut self, x_id: usize, x: Rc<RefCell<Tensor>>,
        gamma: Rc<RefCell<Tensor>>, beta: Rc<RefCell<Tensor>>,
        x_hat: Rc<RefCell<Tensor>>, std_inv: Rc<RefCell<Tensor>>,
        result: Rc<RefCell<Tensor>>,
    ) -> usize {
        let dtype = result.borrow().dtype;
        let id = self.next_id();
        self.nodes.push(TapeNode {
            id,
            op: TapeOp::LayerNorm,
            inputs: vec![x_id],
            input_tensors: vec![x, gamma, beta, x_hat, std_inv, result],
            aux: 0,
            dtype,
        });
        id
    }

    /// Record a conv2d node.
    pub fn conv2d(
        &mut self, x_id: usize, x: Rc<RefCell<Tensor>>,
        w_id: usize, w: Rc<RefCell<Tensor>>,
        im2col: Rc<RefCell<Tensor>>,
        result: Rc<RefCell<Tensor>>,
    ) -> usize {
        let dtype = result.borrow().dtype;
        let id = self.next_id();
        self.nodes.push(TapeNode {
            id,
            op: TapeOp::Conv2D,
            inputs: vec![x_id, w_id],
            input_tensors: vec![x, w, im2col, result],
            aux: 0,
            dtype,
        });
        id
    }

    /// Record a dropout node.
    /// `input_id` — upstream node id for the input tensor.
    /// `input` — the input tensor.
    /// `mask` — the dropout mask (1/(1-rate) for kept, 0 for dropped).
    /// `result` — the output tensor (input * mask).
    pub fn dropout(
        &mut self,
        input_id: usize,
        input: Rc<RefCell<Tensor>>,
        mask: Rc<RefCell<Tensor>>,
        result: Rc<RefCell<Tensor>>,
    ) -> usize {
        let dtype = result.borrow().dtype;
        let id = self.next_id();
        self.nodes.push(TapeNode {
            id,
            op: TapeOp::Dropout,
            inputs: vec![input_id],
            input_tensors: vec![input, mask, result],
            aux: 0,
            dtype,
        });
        id
    }

    /// Record a binary operation with no upstream tape nodes.
    pub fn binary_direct(&mut self, op: TapeOp, a: Rc<RefCell<Tensor>>, b: Rc<RefCell<Tensor>>, result: Rc<RefCell<Tensor>>) -> usize {
        let dtype = result.borrow().dtype;
        let id = self.next_id();
        self.nodes.push(TapeNode {
            id,
            op,
            inputs: vec![],
            input_tensors: vec![a, b, result],
            aux: 0,
            dtype,
        });
        id
    }

    /// Record a select node: result = cond ? then : else.
    /// `then_id` / `else_id` — upstream node ids（cond 阻断链式传播，不写入 inputs）。
    /// `cond` / `then` / `else` — 实际输入张量（cond 用于反向 mask 计算）。
    /// `result` — 输出张量。
    /// inputs 固定为 [then_id, else_id]（若为 None 则用 dummy input 占位），
    /// 保证 backward 时 propagate_grad(node, 0, d_then) / propagate_grad(node, 1, d_else) 索引对齐。
    pub fn select(
        &mut self,
        then_id: Option<usize>, else_id: Option<usize>,
        cond: Rc<RefCell<Tensor>>, then: Rc<RefCell<Tensor>>, else_: Rc<RefCell<Tensor>>,
        result: Rc<RefCell<Tensor>>,
    ) -> usize {
        // 先创建 dummy input（若需要），再分配本节点 id，
        // 保证 id == self.nodes 索引（backward 依赖此不变量：self.nodes[loss_node_id] 与 node_grads[node.id]）
        let tid = then_id.unwrap_or_else(|| self.input(then.clone()));
        let eid = else_id.unwrap_or_else(|| self.input(else_.clone()));
        let dtype = result.borrow().dtype;
        let id = self.next_id();
        self.nodes.push(TapeNode {
            id,
            op: TapeOp::Select,
            inputs: vec![tid, eid],
            input_tensors: vec![cond, then, else_, result],
            aux: 0,
            dtype,
        });
        id
    }

    /// Record a scatter node: out = base.clone(); out[dim][index[i]] = src[i].
    /// `base_id` / `src_id` — 上游节点 id（index 阻断链式传播，不写入 inputs）。
    /// `base` / `src` / `index` — 实际输入张量（index 用于反向 gather 语义）。
    /// `result` — 输出张量。
    /// inputs 固定为 [base_id, src_id]（若为 None 则用 dummy input 占位），
    /// 保证 backward 时 propagate_grad(node, 0, d_base) / propagate_grad(node, 1, d_src) 索引对齐。
    pub fn scatter(
        &mut self,
        base_id: Option<usize>, src_id: Option<usize>,
        base: Rc<RefCell<Tensor>>, src: Rc<RefCell<Tensor>>,
        index: Rc<RefCell<Tensor>>,
        result: Rc<RefCell<Tensor>>,
        dim: usize,
    ) -> usize {
        let bid = base_id.unwrap_or_else(|| self.input(base.clone()));
        let sid = src_id.unwrap_or_else(|| self.input(src.clone()));
        let dtype = result.borrow().dtype;
        let id = self.next_id();
        self.nodes.push(TapeNode {
            id,
            op: TapeOp::Scatter,
            inputs: vec![bid, sid],
            input_tensors: vec![base, src, index, result],
            aux: dim,
            dtype,
        });
        id
    }

    /// Record a gather node: out = base.gather(dim, index)（沿 dim 维按 index 取值）。
    /// `base_id` — 上游节点 id（index 阻断链式传播，不写入 inputs）。
    /// `base` / `index` — 实际输入张量。
    /// `result` — 输出张量（shape == index.shape）。
    /// inputs 固定为 [base_id]（若为 None 则用 dummy input 占位），
    /// 保证 backward 时 propagate_grad(node, 0, d_base) 索引对齐。
    pub fn gather(
        &mut self,
        base_id: Option<usize>,
        base: Rc<RefCell<Tensor>>,
        index: Rc<RefCell<Tensor>>,
        result: Rc<RefCell<Tensor>>,
        dim: usize,
    ) -> usize {
        let bid = base_id.unwrap_or_else(|| self.input(base.clone()));
        let dtype = result.borrow().dtype;
        let id = self.next_id();
        self.nodes.push(TapeNode {
            id,
            op: TapeOp::Gather,
            inputs: vec![bid],
            input_tensors: vec![base, index, result],
            aux: dim,
            dtype,
        });
        id
    }

    /// Record a masked_fill node: result = input.masked_fill(mask, value).
    /// `input_id` — 上游节点 id（mask 阻断链式传播，不写入 inputs）。
    /// `input` / `mask` — 实际输入张量（mask 用于反向 0/1 屏蔽）。
    /// `result` — 输出张量。
    /// inputs 固定为 [input_id]（若为 None 则用 dummy input 占位），
    /// 保证 backward 时 propagate_grad(node, 0, d_input) 索引对齐。
    pub fn masked_fill(
        &mut self,
        input_id: Option<usize>,
        input: Rc<RefCell<Tensor>>,
        mask: Rc<RefCell<Tensor>>,
        result: Rc<RefCell<Tensor>>,
    ) -> usize {
        let iid = input_id.unwrap_or_else(|| self.input(input.clone()));
        let dtype = result.borrow().dtype;
        let id = self.next_id();
        self.nodes.push(TapeNode {
            id,
            op: TapeOp::MaskedFill,
            inputs: vec![iid],
            input_tensors: vec![input, mask, result],
            aux: 0,
            dtype,
        });
        id
    }

    fn next_id(&mut self) -> usize {
        let id = self.counter;
        self.counter += 1;
        id
    }

    // ── backward pass ─────────────────────────────────────────────────

    /// Run backward pass starting from `loss_node_id`.
    /// Writes gradients into the `.grad` field of every `TapeOp::Input` tensor.
    /// 返回 Err 当反向传播发现 shape 不匹配（方向 A：消除 silent squeeze）。
    ///
    /// 阶段 4：按 node.dtype 分发到 f32/f64 路径，实现真正的 f32 反向传播。
    /// 混合 dtype 场景（前向 f32+f64）回退为 f64（策略 B 兜底）。
    pub fn backward(&self, loss_node_id: usize) -> Result<(), crate::error::TenthError> {
        let n = self.nodes.len();
        // Per-node upstream gradient (Option so we can .take() it).
        // 阶段 4：node_grads 改为 Vec<Option<TensorData>>，按 node.dtype 存储。
        let mut node_grads: Vec<Option<TensorData>> = vec![None; n];

        // Seed: ∂loss/∂loss = 1 (or ones if loss is a tensor).
        // The result tensor is always the LAST entry in input_tensors.
        let result_idx = self.nodes[loss_node_id].input_tensors.len() - 1;
        let (loss_shape, loss_dtype) = {
            let loss_tensor = &self.nodes[loss_node_id].input_tensors[result_idx].borrow();
            (loss_tensor.shape(), loss_tensor.dtype)
        };
        // 种子梯度按 loss tensor 的 dtype 构造（f32 → F32 ones，f64 → F64 ones）
        let seed = match loss_dtype {
            BaseType::F32 => TensorData::F32(ArrayD::ones(IxDyn(&loss_shape))),
            _ => TensorData::F64(ArrayD::ones(IxDyn(&loss_shape))),
        };
        node_grads[loss_node_id] = Some(seed);

        // Walk nodes in reverse order (topological by construction).
        for node in self.nodes.iter().rev() {
            let grad = match node_grads[node.id].take() {
                Some(g) => g,
                None => continue,
            };

            match &node.op {
                TapeOp::Input => {
                    // Leaf: accumulate gradient into the parameter tensor.
                    // 方向 A：此处校验梯度 shape 与参数 shape 一致（消除 silent squeeze）
                    node.input_tensors[0].borrow_mut().acc_grad(&grad).map_err(|e| {
                        crate::error::TenthError::RuntimeError {
                            message: format!("反向传播 shape 错误（节点 #{} Input）：{}", node.id, e),
                        }
                    })?;
                }
                TapeOp::Add | TapeOp::Sub => {
                    let sign_f64: f64 = if node.op == TapeOp::Add { 1.0 } else { -1.0 };
                    let shapes: Vec<Vec<usize>> = (0..node.input_tensors.len().min(2))
                        .map(|i| node.input_tensors[i].borrow().shape())
                        .collect();
                    dispatch_float!(node.dtype, E, {
                        let grad_arr = E::from_tensor_data(&grad);
                        let sign = E::from_f64(sign_f64);
                        for (i, input_shape) in shapes.iter().enumerate() {
                            let g_i = if i == 0 {
                                unbroadcast(&grad_arr, input_shape)?
                            } else {
                                unbroadcast(&grad_arr, input_shape)?.mapv(|v| v * sign)
                            };
                            propagate_grad(node, i, &E::into_tensor_data(g_i), &mut node_grads)?;
                        }
                    });
                }
                TapeOp::Mul => {
                    // Clone input data first to avoid holding RefCell borrows.
                    let (a_data, a_shape, b_data, b_shape) = {
                        let a = node.input_tensors[0].borrow();
                        let b = node.input_tensors[1].borrow();
                        (a.data.clone(), a.shape(), b.data.clone(), b.shape())
                    };
                    dispatch_float!(node.dtype, E, {
                        let a_arr = E::from_tensor_data(&a_data);
                        let b_arr = E::from_tensor_data(&b_data);
                        let grad_arr = E::from_tensor_data(&grad);
                        let ga = unbroadcast(&(&grad_arr * &b_arr), &a_shape)?;
                        let gb = unbroadcast(&(&grad_arr * &a_arr), &b_shape)?;
                        propagate_grad(node, 0, &E::into_tensor_data(ga), &mut node_grads)?;
                        propagate_grad(node, 1, &E::into_tensor_data(gb), &mut node_grads)?;
                    });
                }
                TapeOp::Div => {
                    let (a_data, a_shape, b_data, b_shape) = {
                        let a = node.input_tensors[0].borrow();
                        let b = node.input_tensors[1].borrow();
                        (a.data.clone(), a.shape(), b.data.clone(), b.shape())
                    };
                    dispatch_float!(node.dtype, E, {
                        let a_arr = E::from_tensor_data(&a_data);
                        let b_arr = E::from_tensor_data(&b_data);
                        let grad_arr = E::from_tensor_data(&grad);
                        let ga = unbroadcast(&(&grad_arr / &b_arr), &a_shape)?;
                        let gb = unbroadcast(&(-&grad_arr * &a_arr / (&b_arr * &b_arr)), &b_shape)?;
                        propagate_grad(node, 0, &E::into_tensor_data(ga), &mut node_grads)?;
                        propagate_grad(node, 1, &E::into_tensor_data(gb), &mut node_grads)?;
                    });
                }
                TapeOp::Neg => {
                    dispatch_float!(node.dtype, E, {
                        let grad_arr = E::from_tensor_data(&grad);
                        let g = -&grad_arr;
                        propagate_grad(node, 0, &E::into_tensor_data(g), &mut node_grads)?;
                    });
                }
                TapeOp::ReLU => {
                    let input_data = {
                        let a = node.input_tensors[0].borrow();
                        a.data.clone()
                    };
                    dispatch_float!(node.dtype, E, {
                        let mask = E::from_tensor_data(&input_data).mapv(|x| if x > E::from_f64(0.0) { E::from_f64(1.0) } else { E::from_f64(0.0) });
                        let grad_arr = E::from_tensor_data(&grad);
                        let g_a = &grad_arr * &mask;
                        propagate_grad(node, 0, &E::into_tensor_data(g_a), &mut node_grads)?;
                    });
                }
                TapeOp::MatMul => {
                    // Backward for a @ b:
                    //   d_a = grad @ b^T,  d_b = a^T @ grad
                    // Supports 2D@2D, 1D@2D, 2D@1D by promoting 1D to 2D
                    // and squeezing the gradient back afterwards.
                    if node.input_tensors.len() >= 3 {
                        let (a_data, a_ndim, b_data, b_ndim) = {
                            let a_ref = node.input_tensors[0].borrow();
                            let b_ref = node.input_tensors[1].borrow();
                            (a_ref.data.clone(), a_ref.ndim(), b_ref.data.clone(), b_ref.ndim())
                        };

                        // 方向 A：校验输入维度（仅支持 1D/2D，更高维报错而非静默）
                        if a_ndim > 2 {
                            return Err(crate::error::TenthError::RuntimeError {
                                message: format!("MatMul 反向传播：a ndim={} > 2 不支持（方向 A：不再静默处理）", a_ndim),
                            });
                        }
                        if b_ndim > 2 {
                            return Err(crate::error::TenthError::RuntimeError {
                                message: format!("MatMul 反向传播：b ndim={} > 2 不支持（方向 A：不再静默处理）", b_ndim),
                            });
                        }

                        let (d_a, d_b) = dispatch_float!(node.dtype, E, {
                            let a_arr = E::from_tensor_data(&a_data);
                            let b_arr = E::from_tensor_data(&b_data);
                            let grad_arr = E::from_tensor_data(&grad);

                            // Promote 1D inputs to 2D for uniform handling.
                            let a_2d: ArrayD<E> = if a_ndim == 1 {
                                a_arr.view().insert_axis(ndarray::Axis(0)).into_owned().into_dyn()
                            } else {
                                a_arr
                            };
                            let b_2d: ArrayD<E> = if b_ndim == 1 {
                                b_arr.view().insert_axis(ndarray::Axis(1)).into_owned().into_dyn()
                            } else {
                                b_arr
                            };
                            let grad_2d: ArrayD<E> = if grad_arr.ndim() == 1 {
                                if a_ndim == 1 {
                                    // result of (1,k)@(k,n) squeezed to (n,) → promote to (1,n)
                                    grad_arr.view().insert_axis(ndarray::Axis(0)).into_owned().into_dyn()
                                } else {
                                    // result of (m,k)@(k,1) squeezed to (m,) → promote to (m,1)
                                    grad_arr.view().insert_axis(ndarray::Axis(1)).into_owned().into_dyn()
                                }
                            } else if grad_arr.ndim() == 2 {
                                grad_arr.clone()
                            } else {
                                // 方向 A：grad.ndim() > 2 不再静默 clone
                                return Err(crate::error::TenthError::RuntimeError {
                                    message: format!("MatMul 反向传播：grad ndim={} > 2 不支持（方向 A：不再静默兜底）", grad_arr.ndim()),
                                });
                            };

                            // b_2d^T and a_2d^T
                            let b_t = b_2d.view().reversed_axes().to_owned();
                            let a_t = a_2d.view().reversed_axes().to_owned();

                            let d_a_2d = matmul_2d(&grad_2d, &b_t)?;
                            let d_b_2d = matmul_2d(&a_t, &grad_2d)?;

                            // Squeeze gradients back to match original input shapes.
                            // 方向 A：1D squeeze 前校验 shape 符合预期（避免静默 squeeze 错误 shape）
                            let d_a: ArrayD<E> = if a_ndim == 1 {
                                if d_a_2d.shape().get(0).copied() != Some(1) {
                                    return Err(crate::error::TenthError::RuntimeError {
                                        message: format!(
                                            "MatMul 反向 1D squeeze 失败：d_a_2d shape = {:?}，期望第 0 维为 1（方向 A：不再静默 squeeze）",
                                            d_a_2d.shape()
                                        ),
                                    });
                                }
                                d_a_2d.view().index_axis_move(ndarray::Axis(0), 0).into_owned().into_dyn()
                            } else {
                                d_a_2d
                            };
                            let d_b: ArrayD<E> = if b_ndim == 1 {
                                if d_b_2d.shape().get(1).copied() != Some(1) {
                                    return Err(crate::error::TenthError::RuntimeError {
                                        message: format!(
                                            "MatMul 反向 1D squeeze 失败：d_b_2d shape = {:?}，期望第 1 维为 1（方向 A：不再静默 squeeze）",
                                            d_b_2d.shape()
                                        ),
                                    });
                                }
                                d_b_2d.view().index_axis_move(ndarray::Axis(1), 0).into_owned().into_dyn()
                            } else {
                                d_b_2d
                            };
                            (E::into_tensor_data(d_a), E::into_tensor_data(d_b))
                        });

                        propagate_grad(node, 0, &d_a, &mut node_grads)?;
                        propagate_grad(node, 1, &d_b, &mut node_grads)?;
                    }
                }
                TapeOp::BatchedMatMul => {
                    // Batched matmul backward:
                    //   forward: (B, M, K) @ (B, K, N) -> (B, M, N)
                    //   d_a = bmm(grad, b^T)  // (B,M,N) @ (B,N,K) -> (B,M,K)
                    //   d_b = bmm(a^T, grad)  // (B,K,M) @ (B,M,N) -> (B,K,N)
                    // 通过 tensor 的 transpose（仅转最后两维）+ bmm 组合实现。
                    // input_tensors = [a, b, result]
                    if node.input_tensors.len() >= 3 {
                        let (d_a, d_b) = {
                            let a_ref = node.input_tensors[0].borrow();
                            let b_ref = node.input_tensors[1].borrow();
                            let a_ndim = a_ref.ndim();
                            let b_ndim = b_ref.ndim();
                            if a_ndim != 3 || b_ndim != 3 {
                                return Err(crate::error::TenthError::RuntimeError {
                                    message: format!(
                                        "BatchedMatMul 反向传播：a ndim={}, b ndim={}（期望均为 3）",
                                        a_ndim, b_ndim
                                    ),
                                });
                            }
                            if grad.ndim() != 3 {
                                return Err(crate::error::TenthError::RuntimeError {
                                    message: format!(
                                        "BatchedMatMul 反向传播：grad ndim={}（期望 3）",
                                        grad.ndim()
                                    ),
                                });
                            }
                            let b_t = b_ref.transpose().ok_or_else(|| crate::error::TenthError::RuntimeError {
                                message: "BatchedMatMul 反向：b.transpose() 失败".into(),
                            })?;
                            let a_t = a_ref.transpose().ok_or_else(|| crate::error::TenthError::RuntimeError {
                                message: "BatchedMatMul 反向：a.transpose() 失败".into(),
                            })?;
                            // grad 是 TensorData，转为 Tensor 才能调用 bmm
                            let grad_t = Tensor::from_tensor_data(grad.clone());
                            let d_a_t = grad_t.bmm(&b_t).map_err(|e| crate::error::TenthError::RuntimeError {
                                message: format!("BatchedMatMul 反向 d_a：{}", e),
                            })?;
                            let d_b_t = a_t.bmm(&grad_t).map_err(|e| crate::error::TenthError::RuntimeError {
                                message: format!("BatchedMatMul 反向 d_b：{}", e),
                            })?;
                            // d_a_t/d_b_t 的 data 是 TensorData，直接保留 dtype
                            (d_a_t.data.clone(), d_b_t.data.clone())
                        };
                        propagate_grad(node, 0, &d_a, &mut node_grads)?;
                        propagate_grad(node, 1, &d_b, &mut node_grads)?;
                    }
                }
                TapeOp::Transpose => {
                    dispatch_float!(node.dtype, E, {
                        let grad_arr = E::from_tensor_data(&grad);
                        let g_a = {
                            let mut perm: Vec<usize> = (0..grad_arr.ndim()).collect();
                            if perm.len() >= 2 {
                                let last = perm.len() - 1;
                                perm.swap(last - 1, last);
                            }
                            grad_arr.view().permuted_axes(perm).to_owned()
                        };
                        propagate_grad(node, 0, &E::into_tensor_data(g_a), &mut node_grads)?;
                    });
                }
                TapeOp::Sum => {
                    let a_shape = node.input_tensors[0].borrow().shape();
                    dispatch_float!(node.dtype, E, {
                        let grad_arr = E::from_tensor_data(&grad);
                        let s: E = grad_arr.iter().copied().sum();
                        let g_a = ArrayD::from_elem(IxDyn(&a_shape), s);
                        propagate_grad(node, 0, &E::into_tensor_data(g_a), &mut node_grads)?;
                    });
                }
                TapeOp::Mean => {
                    let (a_shape, a_size) = {
                        let a = node.input_tensors[0].borrow();
                        (a.shape(), a.size())
                    };
                    dispatch_float!(node.dtype, E, {
                        let grad_arr = E::from_tensor_data(&grad);
                        let n = E::from_f64(a_size as f64);
                        let s: E = grad_arr.iter().copied().sum::<E>() / n;
                        let g_a = ArrayD::from_elem(IxDyn(&a_shape), s);
                        propagate_grad(node, 0, &E::into_tensor_data(g_a), &mut node_grads)?;
                    });
                }
                TapeOp::Exp => {
                    let result_data = node.input_tensors[1].borrow().data.clone();
                    dispatch_float!(node.dtype, E, {
                        let grad_arr = E::from_tensor_data(&grad);
                        let y = E::from_tensor_data(&result_data);
                        let g_a = &grad_arr * &y;
                        propagate_grad(node, 0, &E::into_tensor_data(g_a), &mut node_grads)?;
                    });
                }
                TapeOp::Log => {
                    let a_data = node.input_tensors[0].borrow().data.clone();
                    dispatch_float!(node.dtype, E, {
                        let grad_arr = E::from_tensor_data(&grad);
                        let a = E::from_tensor_data(&a_data);
                        let g_a = &grad_arr / &a;
                        propagate_grad(node, 0, &E::into_tensor_data(g_a), &mut node_grads)?;
                    });
                }
                TapeOp::Sigmoid => {
                    let result_data = node.input_tensors[1].borrow().data.clone();
                    dispatch_float!(node.dtype, E, {
                        let grad_arr = E::from_tensor_data(&grad);
                        let y = E::from_tensor_data(&result_data);
                        let one = E::from_f64(1.0);
                        let g_a = &grad_arr * &y * &y.mapv(|v| one - v);
                        propagate_grad(node, 0, &E::into_tensor_data(g_a), &mut node_grads)?;
                    });
                }
                TapeOp::BatchNorm => {
                    // Simplified BN backward:
                    // d(gamma) = sum(dY * x_hat), d(beta) = sum(dY)
                    // dX = (gamma / std) * (dY - mean(dY) - x_hat * mean(dY * x_hat))
                    // input_tensors = [input, gamma, beta, x_hat, std_inv, result]
                    if node.input_tensors.len() >= 5 {
                        let (gamma_data, x_hat_data, std_inv_data) = {
                            let gamma_ref = node.input_tensors[1].borrow();
                            let x_hat_ref = node.input_tensors[3].borrow();
                            let std_inv_ref = node.input_tensors[4].borrow();
                            (gamma_ref.data.clone(), x_hat_ref.data.clone(), std_inv_ref.data.clone())
                        };
                        dispatch_float!(node.dtype, E, {
                            let grad_arr = E::from_tensor_data(&grad);
                            let gamma = E::from_tensor_data(&gamma_data);
                            let x_hat = E::from_tensor_data(&x_hat_data);
                            let std_inv = E::from_tensor_data(&std_inv_data);

                            // d(gamma)
                            let d_gamma = &grad_arr * &x_hat;
                            // d(beta)
                            let d_beta = grad_arr.clone();

                            // dX = gamma * std_inv * (dY - mean(dY) - x_hat * mean(dY * x_hat))
                            let n = E::from_f64(grad_arr.len() as f64);
                            let mean_dy = grad_arr.iter().copied().sum::<E>() / n;
                            let mean_dy_xhat = (&grad_arr * &x_hat).iter().copied().sum::<E>() / n;
                            let d_x = &std_inv * &gamma *
                                &(&grad_arr - mean_dy - &(&x_hat * mean_dy_xhat));

                            propagate_grad(node, 0, &E::into_tensor_data(d_x), &mut node_grads)?;
                            propagate_grad(node, 1, &E::into_tensor_data(d_gamma), &mut node_grads)?;
                            propagate_grad(node, 2, &E::into_tensor_data(d_beta), &mut node_grads)?;
                        });
                    }
                }
                TapeOp::LayerNorm => {
                    // LayerNorm backward (over last dim):
                    // d(gamma) = sum_over_outer(dY * x_hat), d(beta) = sum_over_outer(dY)
                    // dX = gamma * std_inv * (dY - mean(dY) - x_hat * mean(dY * x_hat))  [per-row]
                    // input_tensors = [input, gamma, beta, x_hat, std_inv, result]
                    if node.input_tensors.len() >= 5 {
                        let (x_hat_shape, x_hat_data, std_inv_data, gamma_data) = {
                            let x_hat_ref = node.input_tensors[3].borrow();
                            let std_inv_ref = node.input_tensors[4].borrow();
                            let gamma_ref = node.input_tensors[1].borrow();
                            (x_hat_ref.shape(), x_hat_ref.data.clone(), std_inv_ref.data.clone(), gamma_ref.data.clone())
                        };
                        let ndim = x_hat_shape.len();
                        let axis_len = x_hat_shape[ndim - 1];
                        let outer_len: usize = x_hat_shape[..ndim - 1].iter().product();

                        dispatch_float!(node.dtype, E, {
                            let x_hat_arr = E::from_tensor_data(&x_hat_data);
                            let std_inv_arr = E::from_tensor_data(&std_inv_data);
                            let gamma_arr = E::from_tensor_data(&gamma_data);
                            let grad_arr = E::from_tensor_data(&grad);

                            let x_hat_flat = x_hat_arr.as_standard_layout().to_owned();
                            let x_hat_slice = x_hat_flat.as_slice().unwrap_or(&[]);
                            let std_inv_flat = std_inv_arr.as_standard_layout().to_owned();
                            let std_inv_slice = std_inv_flat.as_slice().unwrap_or(&[]);
                            let grad_flat = grad_arr.as_standard_layout().to_owned();
                            let grad_slice = grad_flat.as_slice().unwrap_or(&[]);
                            let g_flat = gamma_arr.as_standard_layout().to_owned();
                            let g_slice = g_flat.as_slice().unwrap_or(&[]);

                            // d(gamma): sum over outer dims of dY * x_hat
                            let mut d_gamma_data = vec![E::from_f64(0.0); axis_len];
                            for i in 0..outer_len {
                                let start = i * axis_len;
                                for j in 0..axis_len {
                                    d_gamma_data[j] = d_gamma_data[j]
                                        + grad_slice.get(start + j).copied().unwrap_or(E::from_f64(0.0))
                                        * x_hat_slice.get(start + j).copied().unwrap_or(E::from_f64(0.0));
                                }
                            }
                            let d_gamma = ArrayD::from_shape_vec(IxDyn(&[axis_len]), d_gamma_data).unwrap();

                            // d(beta): sum over outer dims of dY
                            let mut d_beta_data = vec![E::from_f64(0.0); axis_len];
                            for i in 0..outer_len {
                                let start = i * axis_len;
                                for j in 0..axis_len {
                                    d_beta_data[j] = d_beta_data[j]
                                        + grad_slice.get(start + j).copied().unwrap_or(E::from_f64(0.0));
                                }
                            }
                            let d_beta = ArrayD::from_shape_vec(IxDyn(&[axis_len]), d_beta_data).unwrap();

                            // dX per row: gamma * std_inv * (dY - mean(dY) - x_hat * mean(dY * x_hat))
                            let mut d_x_data = Vec::with_capacity(grad_slice.len());
                            for i in 0..outer_len {
                                let start = i * axis_len;
                                let inv = std_inv_slice.get(i).copied().unwrap_or_else(|| E::from_f64(1.0));
                                let mut mean_dy = E::from_f64(0.0);
                                let mut mean_dy_xhat = E::from_f64(0.0);
                                for j in 0..axis_len {
                                    let dy = grad_slice.get(start + j).copied().unwrap_or(E::from_f64(0.0));
                                    let xh = x_hat_slice.get(start + j).copied().unwrap_or(E::from_f64(0.0));
                                    mean_dy = mean_dy + dy;
                                    mean_dy_xhat = mean_dy_xhat + dy * xh;
                                }
                                let n_inv = E::from_f64(axis_len as f64);
                                mean_dy = mean_dy / n_inv;
                                mean_dy_xhat = mean_dy_xhat / n_inv;
                                for j in 0..axis_len {
                                    let dy = grad_slice.get(start + j).copied().unwrap_or(E::from_f64(0.0));
                                    let xh = x_hat_slice.get(start + j).copied().unwrap_or(E::from_f64(0.0));
                                    let g = g_slice.get(j).copied().unwrap_or_else(|| E::from_f64(1.0));
                                    d_x_data.push(g * inv * (dy - mean_dy - xh * mean_dy_xhat));
                                }
                            }
                            let d_x = ArrayD::from_shape_vec(IxDyn(&x_hat_shape), d_x_data).unwrap();

                            propagate_grad(node, 0, &E::into_tensor_data(d_x), &mut node_grads)?;
                            propagate_grad(node, 1, &E::into_tensor_data(d_gamma), &mut node_grads)?;
                            propagate_grad(node, 2, &E::into_tensor_data(d_beta), &mut node_grads)?;
                        });
                    }
                }
                TapeOp::Gelu => {
                    // GELU backward: d(gelu(x))/dx = 0.5 * (1 + tanh(inner)) + 0.5 * x * sech^2(inner) * sqrt(2/pi) * (1 + 3*0.044715*x^2)
                    // inner = sqrt(2/pi) * (x + 0.044715 * x^3)
                    // input_tensors = [input, result]
                    let x_data = node.input_tensors[0].borrow().data.clone();
                    dispatch_float!(node.dtype, E, {
                        let grad_arr = E::from_tensor_data(&grad);
                        let x = E::from_tensor_data(&x_data);
                        let sqrt_2_over_pi = E::from_f64((2.0 / std::f64::consts::PI).sqrt());
                        let c1 = E::from_f64(0.044715);
                        let c3 = E::from_f64(3.0 * 0.044715);
                        let half = E::from_f64(0.5);
                        let one = E::from_f64(1.0);
                        let deriv = x.mapv(|xv| {
                            let inner = sqrt_2_over_pi * (xv + c1 * xv * xv * xv);
                            let tanh_inner = inner.tanh_();
                            let sech2 = one - tanh_inner * tanh_inner;
                            half * (one + tanh_inner) + half * xv * sech2 * sqrt_2_over_pi * (one + c3 * xv * xv)
                        });
                        let g_a = &grad_arr * &deriv;
                        propagate_grad(node, 0, &E::into_tensor_data(g_a), &mut node_grads)?;
                    });
                }
                TapeOp::Select => {
                    // Select backward: result = cond ? then : else
                    // d_then = unbroadcast(grad * cond_mask, then.shape)
                    // d_else = unbroadcast(grad * (1 - cond_mask), else.shape)
                    // cond 不可微（bool 语义），不传播梯度
                    // input_tensors = [cond, then, else_, result]
                    // inputs = [then_id, else_id]（dummy 占位保证对齐）
                    if node.input_tensors.len() >= 4 {
                        let (cond_data, then_shape, else_shape) = {
                            let cond_ref = node.input_tensors[0].borrow();
                            let then_ref = node.input_tensors[1].borrow();
                            let else_ref = node.input_tensors[2].borrow();
                            (cond_ref.data.clone(), then_ref.shape(), else_ref.shape())
                        };
                        let grad_shape = grad.shape().to_vec();
                        dispatch_float!(node.dtype, E, {
                            let grad_arr = E::from_tensor_data(&grad);
                            let cond_view = E::from_tensor_data(&cond_data);
                            // cond 广播到 result（grad）shape，再转为 0/1 mask
                            let cond_mask: ArrayD<E> = if cond_view.shape() == grad_arr.shape() {
                                cond_view.mapv(|v| if v > E::from_f64(0.5) { E::from_f64(1.0) } else { E::from_f64(0.0) })
                            } else {
                                let bcast_view = cond_view.broadcast(IxDyn(grad_arr.shape()))
                                    .unwrap_or_else(|| cond_view.view());
                                bcast_view.mapv(|v| if v > E::from_f64(0.5) { E::from_f64(1.0) } else { E::from_f64(0.0) }).into_owned()
                            };
                            let one = E::from_f64(1.0);
                            // d_then = unbroadcast(grad * cond_mask, then.shape)
                            let d_then = unbroadcast(&(&grad_arr * &cond_mask), &then_shape)?;
                            // d_else = unbroadcast(grad * (1 - cond_mask), else.shape)
                            let inv_mask = cond_mask.mapv(|v| one - v);
                            let d_else = unbroadcast(&(&grad_arr * &inv_mask), &else_shape)?;

                            propagate_grad(node, 0, &E::into_tensor_data(d_then), &mut node_grads)?;
                            propagate_grad(node, 1, &E::into_tensor_data(d_else), &mut node_grads)?;
                        });
                    }
                }
                TapeOp::Abs => {
                    // |x| backward: d|x|/dx = sign(x)，x=0 处取 0（次梯度中点，工程惯例）
                    // input_tensors = [input, result]
                    let a_data = node.input_tensors[0].borrow().data.clone();
                    dispatch_float!(node.dtype, E, {
                        let grad_arr = E::from_tensor_data(&grad);
                        let a = E::from_tensor_data(&a_data);
                        let zero = E::from_f64(0.0);
                        let one = E::from_f64(1.0);
                        let neg_one = E::from_f64(-1.0);
                        let sign = a.mapv(|x| if x > zero { one } else if x < zero { neg_one } else { zero });
                        let g_a = &grad_arr * &sign;
                        propagate_grad(node, 0, &E::into_tensor_data(g_a), &mut node_grads)?;
                    });
                }
                TapeOp::Scatter => {
                    // Scatter backward（支持任意 dim + 多维 index/src，PyTorch 对齐）:
                    //   forward: out = base.clone();
                    //            对每个 multi-index idx（遍历 index）:
                    //              actual = idx; actual[dim] = index[idx] as usize
                    //              out[actual] = src[idx]
                    //   d_src[idx] = grad[actual]              (gather 语义)
                    //   d_base = grad.clone()，但所有 actual 位置置 0（被 src 覆盖，梯度不传回 base）
                    //   index 不可微（无梯度）
                    // input_tensors = [base, src, index, result]
                    // inputs = [base_id, src_id]
                    // dim 存于 node.aux
                    if node.input_tensors.len() >= 4 {
                        let dim = node.aux;
                        let (base_shape, index_data, index_shape) = {
                            let base_ref = node.input_tensors[0].borrow();
                            let index_ref = node.input_tensors[2].borrow();
                            (base_ref.shape(), index_ref.data.clone(), index_ref.shape().to_vec())
                        };
                        // Scatter 是 index-based 操作，用 f64 视图计算，最后按 node.dtype 转换存储
                        let grad_view = grad.as_f64_view();
                        let index_view = index_data.as_f64_view();
                        let total: usize = index_shape.iter().product();
                        let unflatten = |flat: usize| -> Vec<usize> {
                            let mut multi = vec![0usize; index_shape.len()];
                            let mut rem = flat;
                            for i in (0..index_shape.len()).rev() {
                                multi[i] = rem % index_shape[i];
                                rem /= index_shape[i];
                            }
                            multi
                        };
                        // d_src[idx] = grad[actual]，actual[dim]=index[idx]
                        let mut d_src_data = Vec::with_capacity(total);
                        for flat in 0..total {
                            let multi = unflatten(flat);
                            let mut actual = multi.clone();
                            let v = index_view[IxDyn(&multi)];
                            actual[dim] = v as usize;
                            let g = grad_view.get(IxDyn(&actual)).copied().unwrap_or(0.0);
                            d_src_data.push(g);
                        }
                        // d_base = grad.clone()，但所有 actual 位置置 0
                        let mut d_base_data: Vec<f64> = grad_view.iter().copied().collect();
                        for flat in 0..total {
                            let multi = unflatten(flat);
                            let mut actual = multi.clone();
                            let v = index_view[IxDyn(&multi)];
                            actual[dim] = v as usize;
                            let actual_flat = flatten_index(&actual, &base_shape);
                            if let Some(slot) = d_base_data.get_mut(actual_flat) {
                                *slot = 0.0;
                            }
                        }
                        // 按 node.dtype 构造 TensorData
                        let (d_base, d_src) = match node.dtype {
                            BaseType::F32 => {
                                let d_base = TensorData::F32(
                                    ArrayD::from_shape_vec(IxDyn(&base_shape), d_base_data.iter().map(|v| *v as f32).collect())
                                        .map_err(|_| crate::error::TenthError::RuntimeError {
                                            message: "Scatter 反向 d_base reshape 失败".into(),
                                        })?
                                );
                                let d_src = TensorData::F32(
                                    ArrayD::from_shape_vec(IxDyn(&index_shape), d_src_data.iter().map(|v| *v as f32).collect())
                                        .map_err(|_| crate::error::TenthError::RuntimeError {
                                            message: "Scatter 反向 d_src reshape 失败".into(),
                                        })?
                                );
                                (d_base, d_src)
                            }
                            _ => {
                                let d_base = TensorData::F64(
                                    ArrayD::from_shape_vec(IxDyn(&base_shape), d_base_data)
                                        .map_err(|_| crate::error::TenthError::RuntimeError {
                                            message: "Scatter 反向 d_base reshape 失败".into(),
                                        })?
                                );
                                let d_src = TensorData::F64(
                                    ArrayD::from_shape_vec(IxDyn(&index_shape), d_src_data)
                                        .map_err(|_| crate::error::TenthError::RuntimeError {
                                            message: "Scatter 反向 d_src reshape 失败".into(),
                                        })?
                                );
                                (d_base, d_src)
                            }
                        };
                        propagate_grad(node, 0, &d_base, &mut node_grads)?;
                        propagate_grad(node, 1, &d_src, &mut node_grads)?;
                    }
                }
                TapeOp::Gather => {
                    // Gather backward（支持任意 dim + 多维 index，PyTorch 对齐）:
                    //   forward: out[idx] = base[actual]，actual[dim]=index[idx]，其他维同 idx
                    //   d_base = zeros_like(base)
                    //   对每个 idx: d_base[actual] += grad[idx]   (scatter-add 语义，重复 index 累加)
                    //   index 不可微（无梯度）
                    // input_tensors = [base, index, result]
                    // inputs = [base_id]
                    // dim 存于 node.aux
                    if node.input_tensors.len() >= 3 {
                        let dim = node.aux;
                        let (base_shape, index_data, index_shape) = {
                            let base_ref = node.input_tensors[0].borrow();
                            let index_ref = node.input_tensors[1].borrow();
                            (base_ref.shape(), index_ref.data.clone(), index_ref.shape().to_vec())
                        };
                        // Gather 是 index-based 操作，用 f64 视图计算，最后按 node.dtype 转换存储
                        let grad_view = grad.as_f64_view();
                        let index_view = index_data.as_f64_view();
                        let total: usize = index_shape.iter().product();
                        let unflatten = |flat: usize| -> Vec<usize> {
                            let mut multi = vec![0usize; index_shape.len()];
                            let mut rem = flat;
                            for i in (0..index_shape.len()).rev() {
                                multi[i] = rem % index_shape[i];
                                rem /= index_shape[i];
                            }
                            multi
                        };
                        let base_total: usize = base_shape.iter().product();
                        let mut d_base_data: Vec<f64> = vec![0.0; base_total];
                        for flat in 0..total {
                            let multi = unflatten(flat);
                            let mut actual = multi.clone();
                            let v = index_view[IxDyn(&multi)];
                            actual[dim] = v as usize;
                            let actual_flat = flatten_index(&actual, &base_shape);
                            let g = grad_view.get(IxDyn(&multi)).copied().unwrap_or(0.0);
                            if let Some(slot) = d_base_data.get_mut(actual_flat) {
                                *slot += g;
                            }
                        }
                        let d_base = match node.dtype {
                            BaseType::F32 => TensorData::F32(
                                ArrayD::from_shape_vec(IxDyn(&base_shape), d_base_data.iter().map(|v| *v as f32).collect())
                                    .map_err(|_| crate::error::TenthError::RuntimeError {
                                        message: "Gather 反向 d_base reshape 失败".into(),
                                    })?
                            ),
                            _ => TensorData::F64(
                                ArrayD::from_shape_vec(IxDyn(&base_shape), d_base_data)
                                    .map_err(|_| crate::error::TenthError::RuntimeError {
                                        message: "Gather 反向 d_base reshape 失败".into(),
                                    })?
                            ),
                        };
                        propagate_grad(node, 0, &d_base, &mut node_grads)?;
                    }
                }
                TapeOp::Reshape => {
                    // Reshape backward: d_input = grad.reshape(input.shape())
                    // input_tensors = [input, result]（原始 shape 从 input.shape() 读取）
                    // 元素数必须一致（reshape 不改变元素数）
                    // Reshape 是 dtype 无关操作，直接在 TensorData 上 reshape
                    let orig_shape = node.input_tensors[0].borrow().shape();
                    let total: usize = orig_shape.iter().product();
                    if grad.len() != total {
                        return Err(crate::error::TenthError::RuntimeError {
                            message: format!(
                                "Reshape 反向元素数不匹配：grad {} 元素，原始 shape {:?} 期望 {} 元素",
                                grad.len(), orig_shape, total
                            ),
                        });
                    }
                    // 注意：grad 可能不是连续内存（如经过 MatMul/Transpose 后的视图），
                    // 用 from_shape_vec 重新构造保证连续，避免 into_shape_with_order 失败。
                    let g_a = match grad {
                        TensorData::F32(a) => {
                            let data: Vec<f32> = a.iter().cloned().collect();
                            TensorData::F32(
                                ArrayD::from_shape_vec(IxDyn(&orig_shape), data)
                                    .map_err(|_| crate::error::TenthError::RuntimeError {
                                        message: format!("Reshape 反向 reshape grad 到 {:?} 失败", orig_shape),
                                    })?
                            )
                        },
                        TensorData::F64(a) => {
                            let data: Vec<f64> = a.iter().cloned().collect();
                            TensorData::F64(
                                ArrayD::from_shape_vec(IxDyn(&orig_shape), data)
                                    .map_err(|_| crate::error::TenthError::RuntimeError {
                                        message: format!("Reshape 反向 reshape grad 到 {:?} 失败", orig_shape),
                                    })?
                            )
                        },
                    };
                    propagate_grad(node, 0, &g_a, &mut node_grads)?;
                }
                TapeOp::MaskedFill => {
                    // MaskedFill backward: d_input = grad * (1 - mask)
                    // mask=true 位置被 value 覆盖，不传梯度回输入
                    // input_tensors = [input, mask, result]
                    let mask_data = node.input_tensors[1].borrow().data.clone();
                    dispatch_float!(node.dtype, E, {
                        let grad_arr = E::from_tensor_data(&grad);
                        let mask_view = E::from_tensor_data(&mask_data);
                        let one = E::from_f64(1.0);
                        let zero = E::from_f64(0.0);
                        let inv_mask = mask_view.mapv(|v| if v > E::from_f64(0.5) { zero } else { one });
                        let g_a = &grad_arr * &inv_mask;
                        propagate_grad(node, 0, &E::into_tensor_data(g_a), &mut node_grads)?;
                    });
                }
                TapeOp::Conv2D => {
                    // input_tensors = [input(4D), weight(4D), im2col(2D), output(4D)]
                    // Forward: output = im2col @ w_flat^T  where w_flat = weight.reshape(C_out, C_in*kH*kW)
                    // dW_flat = im2col^T @ dY        → reshape back to (C_out, C_in, kH, kW)
                    // d(im2col) = dY @ w_flat        → col2im back to input shape
                    // dY (upstream grad) has output shape (N, C_out, H_out, W_out);
                    // we reshape it to 2D (N*H_out*W_out, C_out) for matmul.
                    if node.input_tensors.len() >= 4 {
                        let (x_shape, w_shape, out_shape) = {
                            let x_ref = node.input_tensors[0].borrow();
                            let w_ref = node.input_tensors[1].borrow();
                            let out_ref = node.input_tensors[3].borrow();
                            (x_ref.shape(), w_ref.shape(), out_ref.shape())
                        };
                        // output is (N, C_out, H_out, W_out)
                        let n = out_shape[0];
                        let c_out = out_shape[1];
                        let hw_out = out_shape[2] * out_shape[3];

                        let (d_x, d_w) = {
                            let w_ref = node.input_tensors[1].borrow();
                            let col_ref = node.input_tensors[2].borrow();
                            dispatch_float!(node.dtype, E, {
                                let w_arr = E::from_tensor_data(&w_ref.data);
                                let col_arr = E::from_tensor_data(&col_ref.data);
                                let grad_arr = E::from_tensor_data(&grad);

                                let grad_2d: ArrayD<E> = {
                                    let v: Vec<E> = grad_arr.iter().cloned().collect();
                                    ArrayD::from_shape_vec(IxDyn(&[hw_out * n, c_out]), v).map_err(|_| {
                                        crate::error::TenthError::RuntimeError {
                                            message: "Conv2D 反向 reshape grad 失败".into(),
                                        }
                                    })?
                                };

                                // dW_flat = im2col^T @ dY
                                let col_t = col_arr.view().reversed_axes().to_owned();
                                let d_w_flat = matmul_2d(&col_t, &grad_2d)?;
                                let d_w_flat_t = d_w_flat.view().reversed_axes().to_owned();
                                let total_w: usize = w_shape.iter().product();
                                if d_w_flat_t.len() != total_w {
                                    return Err(crate::error::TenthError::RuntimeError {
                                        message: format!("Conv2D 反向 dW 元素数不匹配：{} != {}", d_w_flat_t.len(), total_w),
                                    });
                                }
                                let d_w = ArrayD::from_shape_vec(IxDyn(&w_shape), d_w_flat_t.iter().cloned().collect()).map_err(|_| {
                                    crate::error::TenthError::RuntimeError {
                                        message: "Conv2D 反向 dW reshape 失败".into(),
                                    }
                                })?;

                                // d(im2col) = dY @ w_flat
                                let w_flat: ArrayD<E> = {
                                    let v: Vec<E> = w_arr.iter().cloned().collect();
                                    ArrayD::from_shape_vec(IxDyn(&[c_out, w_shape[1] * w_shape[2] * w_shape[3]]), v).map_err(|_| {
                                        crate::error::TenthError::RuntimeError {
                                            message: "Conv2D 反向 w_flat reshape 失败".into(),
                                        }
                                    })?
                                };
                                let d_col = matmul_2d(&grad_2d, &w_flat)?;

                                // col2im: accumulate d_col back into input shape
                                let x_total: usize = x_shape.iter().product();
                                if d_col.len() != x_total {
                                    return Err(crate::error::TenthError::RuntimeError {
                                        message: format!("Conv2D 反向 dX 元素数不匹配：d_col {} != x_total {}", d_col.len(), x_total),
                                    });
                                }
                                let d_x = ArrayD::from_shape_vec(IxDyn(&x_shape), d_col.iter().cloned().collect()).map_err(|_| {
                                    crate::error::TenthError::RuntimeError {
                                        message: "Conv2D 反向 dX reshape 失败".into(),
                                    }
                                })?;

                                (E::into_tensor_data(d_x), E::into_tensor_data(d_w))
                            })
                        };

                        propagate_grad(node, 0, &d_x, &mut node_grads)?;
                        propagate_grad(node, 1, &d_w, &mut node_grads)?;
                    }
                }
                TapeOp::Dropout => {
                    // d(dropout(x))/dx = mask * dY
                    // input_tensors = [input, mask, result]
                    if node.input_tensors.len() >= 2 {
                        let mask_data = node.input_tensors[1].borrow().data.clone();
                        dispatch_float!(node.dtype, E, {
                            let grad_arr = E::from_tensor_data(&grad);
                            let mask = E::from_tensor_data(&mask_data);
                            let g_a = &grad_arr * &mask;
                            propagate_grad(node, 0, &E::into_tensor_data(g_a), &mut node_grads)?;
                        });
                    }
                }
                TapeOp::CrossEntropy => {
                    // d(CE)/d(logits) = softmax - target
                    // input_tensors = [logits, softmax_output, target]
                    if node.input_tensors.len() >= 3 {
                        let (sm_data, tgt_data) = {
                            let sm_ref = node.input_tensors[1].borrow();
                            let tgt_ref = node.input_tensors[2].borrow();
                            (sm_ref.data.clone(), tgt_ref.data.clone())
                        };
                        dispatch_float!(node.dtype, E, {
                            let sm = E::from_tensor_data(&sm_data);
                            let tgt = E::from_tensor_data(&tgt_data);
                            let g_a = &sm - &tgt;
                            propagate_grad(node, 0, &E::into_tensor_data(g_a), &mut node_grads)?;
                        });
                    }
                }
                TapeOp::Softmax => {
                    // d(softmax(x)_i)/dx_j = y_i * (δ_ij - y_j)
                    // Chain rule: g_i = y_i * (grad_i - sum_j(grad_j * y_j))
                    let result_data = node.input_tensors[1].borrow().data.clone();
                    dispatch_float!(node.dtype, E, {
                        let grad_arr = E::from_tensor_data(&grad);
                        let y = E::from_tensor_data(&result_data);
                        let sum_term: E = (&grad_arr * &y).iter().copied().sum();
                        let g_a = &grad_arr * &y - &y.mapv(|v| v * sum_term);
                        propagate_grad(node, 0, &E::into_tensor_data(g_a), &mut node_grads)?;
                    });
                }
            }
        }
        Ok(())
    }

    /// Clear all nodes and reset the counter.
    pub fn clear(&mut self) {
        self.nodes.clear();
        self.counter = 0;
    }

    /// Zero out the `.grad` field of every Input (leaf) tensor on the tape.
    pub fn zero_grad(&self) {
        for node in &self.nodes {
            if node.op == TapeOp::Input && !node.input_tensors.is_empty() {
                node.input_tensors[0].borrow_mut().zero_grad();
            }
        }
    }
}

// ── helpers ───────────────────────────────────────────────────────────

/// Accumulate `g` into `node_grads[id]`, adding if a gradient already exists.
/// 阶段 4：支持 TensorData 累加（同 dtype 直接加，异 dtype 提升为 f64）。
fn acc_node_grad(node_grads: &mut [Option<TensorData>], id: usize, g: &TensorData) {
    let existing = node_grads[id].take();
    node_grads[id] = match (existing, g) {
        (Some(TensorData::F64(cur)), TensorData::F64(g2)) => Some(TensorData::F64(&cur + g2)),
        (Some(TensorData::F32(cur)), TensorData::F32(g2)) => Some(TensorData::F32(&cur + g2)),
        (Some(TensorData::F64(cur)), TensorData::F32(g2)) => Some(TensorData::F64(&cur + &g2.mapv(|v| v as f64))),
        (Some(TensorData::F32(cur)), TensorData::F64(g2)) => Some(TensorData::F64(&cur.mapv(|v| v as f64) + g2)),
        (None, _) => Some(g.clone()),
    };
}

/// Propagate gradient to input `input_idx` of a node.
/// If the node has upstream node ids, write to `node_grads` so DAG traversal
/// continues.  Otherwise, write directly to the tensor's `.grad` field
/// (used by `_direct` variants that bypass the node-graph).
/// 返回 Err 当 direct 路径的 acc_grad 报告 shape 不匹配（方向 A）。
/// 阶段 4：g 参数从 &ArrayD<f64> 改为 &TensorData，支持按 dtype 存储。
fn propagate_grad(
    node: &TapeNode,
    input_idx: usize,
    g: &TensorData,
    node_grads: &mut [Option<TensorData>],
) -> Result<(), crate::error::TenthError> {
    if input_idx < node.inputs.len() {
        acc_node_grad(node_grads, node.inputs[input_idx], g);
    } else {
        if let Some(t) = node.input_tensors.get(input_idx) {
            t.borrow_mut().acc_grad(g).map_err(|e| {
                crate::error::TenthError::RuntimeError {
                    message: format!("反向传播 shape 错误（节点 #{} {} direct input {}）：{}", node.id, op_name(&node.op), input_idx, e),
                }
            })?;
        }
    }
    Ok(())
}

/// 把多维索引（row-major / C order）展平为线性索引。
/// `multi` 长度必须与 `shape` 一致；每个维度值 < shape[d]。
fn flatten_index(multi: &[usize], shape: &[usize]) -> usize {
    let mut flat = 0usize;
    let mut stride = 1usize;
    for d in (0..multi.len()).rev() {
        flat += multi[d] * stride;
        stride *= shape[d];
    }
    flat
}

/// 人类可读的 TapeOp 名称（用于错误信息）。
fn op_name(op: &TapeOp) -> &'static str {
    match op {
        TapeOp::Input => "Input",
        TapeOp::Add => "Add",
        TapeOp::Sub => "Sub",
        TapeOp::Mul => "Mul",
        TapeOp::Div => "Div",
        TapeOp::Neg => "Neg",
        TapeOp::ReLU => "ReLU",
        TapeOp::MatMul => "MatMul",
        TapeOp::BatchedMatMul => "BatchedMatMul",
        TapeOp::Transpose => "Transpose",
        TapeOp::Sum => "Sum",
        TapeOp::Mean => "Mean",
        TapeOp::Exp => "Exp",
        TapeOp::Log => "Log",
        TapeOp::Sigmoid => "Sigmoid",
        TapeOp::Softmax => "Softmax",
        TapeOp::CrossEntropy => "CrossEntropy",
        TapeOp::Dropout => "Dropout",
        TapeOp::Conv2D => "Conv2D",
        TapeOp::BatchNorm => "BatchNorm",
        TapeOp::LayerNorm => "LayerNorm",
        TapeOp::Gelu => "Gelu",
        TapeOp::Select => "Select",
        TapeOp::Abs => "Abs",
        TapeOp::Scatter => "Scatter",
        TapeOp::Gather => "Gather",
        TapeOp::Reshape => "Reshape",
        TapeOp::MaskedFill => "MaskedFill",
    }
}

/// Reduce `grad` from the output shape down to `target_shape` by summing
/// over broadcast dimensions.  Follows numpy-style broadcasting rules.
/// 返回 Err 当 reshape 失败（方向 A：不再静默保留错误 shape）。
/// 阶段 4：泛型化，支持 f32 和 f64。
fn unbroadcast<E: FloatElem>(grad: &ArrayD<E>, target_shape: &[usize]) -> Result<ArrayD<E>, crate::error::TenthError> {
    let grad_shape = grad.shape();
    if grad_shape == target_shape {
        return Ok(grad.clone());
    }

    let mut result = grad.clone();

    // Align shapes from the right.
    let g_ndim = grad_shape.len();
    let t_ndim = target_shape.len();

    // Pad target shape with 1s on the left to match grad ndim.
    let mut padded_target: Vec<usize> = vec![1; g_ndim.saturating_sub(t_ndim)];
    padded_target.extend_from_slice(target_shape);

    // For each axis where target is 1 and grad > 1, sum over that axis.
    for axis in (0..g_ndim).rev() {
        if padded_target[axis] == 1 && grad_shape[axis] > 1 {
            result = result.sum_axis(ndarray::Axis(axis));
        }
    }

    // Reshape to target if needed (sum_axis may keep trailing dims).
    let current_shape: Vec<usize> = result.shape().to_vec();
    if current_shape != target_shape {
        let total: usize = target_shape.iter().product();
        if total == result.len() {
            result = result.clone().into_shape_with_order(IxDyn(target_shape)).map_err(|_| {
                crate::error::TenthError::RuntimeError {
                    message: format!(
                        "unbroadcast reshape 失败：梯度 shape {:?} 无法 reshape 到目标 shape {:?}（方向 A：不再静默保留错误 shape）",
                        current_shape, target_shape
                    ),
                }
            })?;
        } else {
            return Err(crate::error::TenthError::RuntimeError {
                message: format!(
                    "unbroadcast 元素数不匹配：梯度 {} 元素，目标 {} 元素（shape {:?} → {:?}）",
                    result.len(), total, current_shape, target_shape
                ),
            });
        }
    }

    Ok(result)
}

/// Pure 2-D matrix multiplication returning an owned ArrayD.
/// 返回 Err 当输入非 2D（方向 A：不再静默返回零数组掩盖错误）。
/// 阶段 4：泛型化，支持 f32 和 f64。
fn matmul_2d<E: FloatElem>(a: &ArrayD<E>, b: &ArrayD<E>) -> Result<ArrayD<E>, crate::error::TenthError> {
    let a2 = a.view().into_dimensionality::<ndarray::Ix2>().map_err(|_| {
        crate::error::TenthError::RuntimeError {
            message: format!("matmul_2d 期望 2D 输入，实际 a shape = {:?}（方向 A：不再静默返回零数组）", a.shape()),
        }
    })?;
    let b2 = b.view().into_dimensionality::<ndarray::Ix2>().map_err(|_| {
        crate::error::TenthError::RuntimeError {
            message: format!("matmul_2d 期望 2D 输入，实际 b shape = {:?}（方向 A：不再静默返回零数组）", b.shape()),
        }
    })?;
    Ok(a2.dot(&b2).into_dyn())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_tensor(data: Vec<f64>, shape: Vec<usize>) -> Rc<RefCell<Tensor>> {
        Rc::new(RefCell::new(Tensor::from_vec(data, shape)))
    }

    #[test]
    fn test_backward_mul_same_shape() {
        // f(a, b) = a * b (element-wise)
        let a = make_tensor(vec![2.0, 3.0], vec![2]);
        let b = make_tensor(vec![4.0, 5.0], vec![2]);

        let mut tape = Tape::new();
        let _a_id = tape.input(a.clone());
        let _b_id = tape.input(b.clone());

        // Compute a * b
        let a_data = a.borrow().data.clone();
        let b_data = b.borrow().data.clone();
        let result_data = &a_data * &b_data;
        let result = Rc::new(RefCell::new(Tensor::from_data(result_data)));
        let r_id = tape.binary_direct(TapeOp::Mul, a.clone(), b.clone(), result.clone());

        tape.backward(r_id).unwrap();

        // d(a*b)/da = b = [4, 5]
        let a_grad = a.borrow().grad.clone().unwrap();
        let a_grad_f64 = a_grad.as_f64_view();
        assert!((a_grad_f64[[0]] - 4.0).abs() < 1e-10, "a grad[0] = {}", a_grad_f64[[0]]);
        assert!((a_grad_f64[[1]] - 5.0).abs() < 1e-10, "a grad[1] = {}", a_grad_f64[[1]]);

        // d(a*b)/db = a = [2, 3]
        let b_grad = b.borrow().grad.clone().unwrap();
        let b_grad_f64 = b_grad.as_f64_view();
        assert!((b_grad_f64[[0]] - 2.0).abs() < 1e-10, "b grad[0] = {}", b_grad_f64[[0]]);
        assert!((b_grad_f64[[1]] - 3.0).abs() < 1e-10, "b grad[1] = {}", b_grad_f64[[1]]);
    }

    #[test]
    fn test_backward_add_broadcast() {
        // f(w, b) = w + b where w=[2,3], b=[1] (broadcast)
        let w = make_tensor(vec![10.0, 20.0], vec![2]);
        let b = make_tensor(vec![5.0], vec![1]);

        let mut tape = Tape::new();
        tape.input(w.clone());
        tape.input(b.clone());

        let w_data = w.borrow().data.clone();
        let b_data = b.borrow().data.clone();
        let b_br = b_data.broadcast(w_data.shape()).unwrap();
        let result_data = &w_data + &b_br;
        let result = Rc::new(RefCell::new(Tensor::from_data(result_data)));
        let r_id = tape.binary_direct(TapeOp::Add, w.clone(), b.clone(), result.clone());

        tape.backward(r_id).unwrap();

        let w_grad = w.borrow().grad.clone().unwrap();
        let w_grad_f64 = w_grad.as_f64_view();
        assert!((w_grad_f64[[0]] - 1.0).abs() < 1e-10);
        assert!((w_grad_f64[[1]] - 1.0).abs() < 1e-10);

        // b had shape [1], gradient should be sum of upstream over broadcast dim: 1+1=2
        let b_grad = b.borrow().grad.clone().unwrap();
        let b_grad_f64 = b_grad.as_f64_view();
        assert!((b_grad_f64[[0]] - 2.0).abs() < 1e-10, "b_grad = {}", b_grad_f64[[0]]);
    }

    #[test]
    fn test_backward_relu() {
        let x = make_tensor(vec![-1.0, 2.0, -3.0, 4.0], vec![4]);

        let mut tape = Tape::new();
        tape.input(x.clone());

        let x_data = x.borrow().data.clone();
        let relu_data = x_data.mapv(|v| if v > 0.0 { v } else { 0.0 });
        let result = Rc::new(RefCell::new(Tensor::from_data(relu_data)));
        let r_id = tape.unary_direct(TapeOp::ReLU, x.clone(), result.clone());

        tape.backward(r_id).unwrap();

        let grad = x.borrow().grad.clone().unwrap();
        let grad_f64 = grad.as_f64_view();
        assert!((grad_f64[[0]] - 0.0).abs() < 1e-10);
        assert!((grad_f64[[1]] - 1.0).abs() < 1e-10);
        assert!((grad_f64[[2]] - 0.0).abs() < 1e-10);
        assert!((grad_f64[[3]] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_backward_matmul() {
        // a (2x3), b (3x2), result (2x2)
        let a = make_tensor(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let b = make_tensor(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![3, 2]);

        let mut tape = Tape::new();
        tape.input(a.clone());
        tape.input(b.clone());

        let a_data = a.borrow().data.clone();
        let b_data = b.borrow().data.clone();
        let a2 = a_data.view().into_dimensionality::<ndarray::Ix2>().unwrap();
        let b2 = b_data.view().into_dimensionality::<ndarray::Ix2>().unwrap();
        let result_data = a2.dot(&b2).into_dyn();
        let result = Rc::new(RefCell::new(Tensor::from_data(result_data)));
        let r_id = tape.binary_direct(TapeOp::MatMul, a.clone(), b.clone(), result.clone());

        tape.backward(r_id).unwrap();

        // d(a @ b)/da = 1 @ b^T (since upstream grad is ones)
        // b^T is (2x3), so da/dy (2x2) @ b^T (2x3) → should give (2x3)
        let a_grad = a.borrow().grad.clone().unwrap();
        let a_grad_f64 = a_grad.as_f64_view();
        assert_eq!(a_grad_f64.shape(), &[2, 3]);

        // d(a @ b)/db = a^T @ 1, a^T (3x2) @ 1 (2x2) → (3x2)
        let b_grad = b.borrow().grad.clone().unwrap();
        let b_grad_f64 = b_grad.as_f64_view();
        assert_eq!(b_grad_f64.shape(), &[3, 2]);
    }

    #[test]
    fn test_backward_chain() {
        // f(x) = relu(x) * 2, x = [-2, 3]
        // Expected: df/dx = [0, 2]
        let x = make_tensor(vec![-2.0, 3.0], vec![2]);

        let mut tape = Tape::new();
        let x_id = tape.input(x.clone());

        // relu(x)
        let x_data = x.borrow().data.clone();
        let relu_data = x_data.mapv(|v| if v > 0.0 { v } else { 0.0 });
        let relu = Rc::new(RefCell::new(Tensor::from_data(relu_data)));
        let relu_id = tape.unary(TapeOp::ReLU, x_id, x.clone(), relu.clone());

        // relu * 2 — two is constant (not a parameter), register as input then ignore its grad
        let two = make_tensor(vec![2.0, 2.0], vec![2]);
        let two_id = tape.input(two.clone());
        let r_data = &relu.borrow().data * &two.borrow().data;
        let result = Rc::new(RefCell::new(Tensor::from_data(r_data)));
        let r_id = tape.binary(TapeOp::Mul, relu_id, two_id, relu.clone(), two.clone(), result);

        tape.backward(r_id).unwrap();

        let grad = x.borrow().grad.clone().unwrap();
        let grad_f64 = grad.as_f64_view();
        // d(relu(x)*2)/dx = 2 if x > 0 else 0
        assert!((grad_f64[[0]] - 0.0).abs() < 1e-10, "x[0] grad = {}", grad_f64[[0]]);
        assert!((grad_f64[[1]] - 2.0).abs() < 1e-10, "x[1] grad = {}", grad_f64[[1]]);
    }
}
