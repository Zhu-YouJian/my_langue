//! Tensor-level automatic differentiation via a Wengert tape.
//!
//! Records operations on `Tensor`s during forward execution, then replays
//! the chain rule backward to populate each parameter tensor's `.grad` field.

use std::rc::Rc;
use std::cell::RefCell;
use ndarray::{ArrayD, IxDyn};
use super::tensor::Tensor;

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

    /// Register a leaf (parameter) tensor.  Returns the node id.
    pub fn input(&mut self, tensor: Rc<RefCell<Tensor>>) -> usize {
        let id = self.next_id();
        self.nodes.push(TapeNode {
            id,
            op: TapeOp::Input,
            inputs: vec![],
            input_tensors: vec![tensor],
        });
        id
    }

    /// Record a unary operation.
    /// `input_id` — upstream node id (for chain-rule traversal).
    /// `input_tensor` — the tensor that was the *input* to this op (needed
    ///   by backward to read saved values).
    /// `result` — the output tensor (the forward result).
    pub fn unary(&mut self, op: TapeOp, input_id: usize, input_tensor: Rc<RefCell<Tensor>>, result: Rc<RefCell<Tensor>>) -> usize {
        let id = self.next_id();
        self.nodes.push(TapeNode {
            id,
            op,
            inputs: vec![input_id],
            input_tensors: vec![input_tensor, result],
        });
        id
    }

    /// Record a binary operation.
    /// `a_id` / `b_id` — upstream node ids.
    /// `a` / `b` — the actual input tensors (for reading values in backward).
    /// `result` — the output tensor.
    pub fn binary(&mut self, op: TapeOp, a_id: usize, b_id: usize, a: Rc<RefCell<Tensor>>, b: Rc<RefCell<Tensor>>, result: Rc<RefCell<Tensor>>) -> usize {
        let id = self.next_id();
        self.nodes.push(TapeNode {
            id,
            op,
            inputs: vec![a_id, b_id],
            input_tensors: vec![a, b, result],
        });
        id
    }

    /// Record a unary operation that has no upstream tape node
    /// (e.g. a leaf tensor is passed directly).
    pub fn unary_direct(&mut self, op: TapeOp, input_tensor: Rc<RefCell<Tensor>>, result: Rc<RefCell<Tensor>>) -> usize {
        let id = self.next_id();
        self.nodes.push(TapeNode {
            id,
            op,
            inputs: vec![],
            input_tensors: vec![input_tensor, result],
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
        let id = self.next_id();
        self.nodes.push(TapeNode {
            id,
            op: TapeOp::CrossEntropy,
            inputs: vec![logits_id],
            input_tensors: vec![logits, softmax, target, result],
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
        let id = self.next_id();
        self.nodes.push(TapeNode {
            id,
            op: TapeOp::BatchNorm,
            inputs: vec![x_id],
            input_tensors: vec![x, gamma, beta, x_hat, std_inv, result],
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
        let id = self.next_id();
        self.nodes.push(TapeNode {
            id,
            op: TapeOp::LayerNorm,
            inputs: vec![x_id],
            input_tensors: vec![x, gamma, beta, x_hat, std_inv, result],
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
        let id = self.next_id();
        self.nodes.push(TapeNode {
            id,
            op: TapeOp::Conv2D,
            inputs: vec![x_id, w_id],
            input_tensors: vec![x, w, im2col, result],
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
        let id = self.next_id();
        self.nodes.push(TapeNode {
            id,
            op: TapeOp::Dropout,
            inputs: vec![input_id],
            input_tensors: vec![input, mask, result],
        });
        id
    }

    /// Record a binary operation with no upstream tape nodes.
    pub fn binary_direct(&mut self, op: TapeOp, a: Rc<RefCell<Tensor>>, b: Rc<RefCell<Tensor>>, result: Rc<RefCell<Tensor>>) -> usize {
        let id = self.next_id();
        self.nodes.push(TapeNode {
            id,
            op,
            inputs: vec![],
            input_tensors: vec![a, b, result],
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
    pub fn backward(&self, loss_node_id: usize) -> Result<(), crate::error::TenthError> {
        let n = self.nodes.len();
        // Per-node upstream gradient (Option so we can .take() it).
        let mut node_grads: Vec<Option<ArrayD<f64>>> = vec![None; n];

        // Seed: ∂loss/∂loss = 1 (or ones if loss is a tensor).
        // The result tensor is always the LAST entry in input_tensors.
        let result_idx = self.nodes[loss_node_id].input_tensors.len() - 1;
        let loss_tensor = &self.nodes[loss_node_id].input_tensors[result_idx].borrow();
        let loss_shape = loss_tensor.shape();
        node_grads[loss_node_id] = Some(ArrayD::ones(IxDyn(&loss_shape)));

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
                    let sign: f64 = if node.op == TapeOp::Add { 1.0 } else { -1.0 };
                    let shapes: Vec<Vec<usize>> = (0..node.input_tensors.len().min(2))
                        .map(|i| node.input_tensors[i].borrow().shape())
                        .collect();
                    for (i, input_shape) in shapes.iter().enumerate() {
                        let g_i = if i == 0 {
                            unbroadcast(&grad, input_shape)?
                        } else {
                            unbroadcast(&grad, input_shape)?.mapv(|v| v * sign)
                        };
                        propagate_grad(node, i, &g_i, &mut node_grads)?;
                    }
                }
                TapeOp::Mul => {
                    // Clone input data first to avoid holding RefCell borrows.
                    let (a_data, a_shape, b_data, b_shape) = {
                        let a = node.input_tensors[0].borrow();
                        let b = node.input_tensors[1].borrow();
                        (a.data.clone(), a.shape(), b.data.clone(), b.shape())
                    };
                    let ga = unbroadcast(&(&grad * &b_data), &a_shape)?;
                    let gb = unbroadcast(&(&grad * &a_data), &b_shape)?;
                    propagate_grad(node, 0, &ga, &mut node_grads)?;
                    propagate_grad(node, 1, &gb, &mut node_grads)?;
                }
                TapeOp::Div => {
                    let (a_data, a_shape, b_data, b_shape) = {
                        let a = node.input_tensors[0].borrow();
                        let b = node.input_tensors[1].borrow();
                        (a.data.clone(), a.shape(), b.data.clone(), b.shape())
                    };
                    let ga = unbroadcast(&(&grad / &b_data), &a_shape)?;
                    let gb = unbroadcast(&(-&grad * &a_data / (&b_data * &b_data)), &b_shape)?;
                    propagate_grad(node, 0, &ga, &mut node_grads)?;
                    propagate_grad(node, 1, &gb, &mut node_grads)?;
                }
                TapeOp::Neg => {
                    let g = -&grad;
                    propagate_grad(node, 0, &g, &mut node_grads)?;
                }
                TapeOp::ReLU => {
                    let mask = {
                        let a = node.input_tensors[0].borrow();
                        a.data.mapv(|x| if x > 0.0 { 1.0 } else { 0.0 })
                    };
                    let g_a = &grad * &mask;
                    propagate_grad(node, 0, &g_a, &mut node_grads)?;
                }
                TapeOp::MatMul => {
                    // Backward for a @ b:
                    //   d_a = grad @ b^T,  d_b = a^T @ grad
                    // Supports 2D@2D, 1D@2D, 2D@1D by promoting 1D to 2D
                    // and squeezing the gradient back afterwards.
                    if node.input_tensors.len() >= 3 {
                        let (d_a, d_b) = {
                            let a_ref = node.input_tensors[0].borrow();
                            let b_ref = node.input_tensors[1].borrow();
                            let a_ndim = a_ref.ndim();
                            let b_ndim = b_ref.ndim();

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

                            // Promote 1D inputs to 2D for uniform handling.
                            let a_2d: ArrayD<f64> = if a_ndim == 1 {
                                a_ref.data.view().insert_axis(ndarray::Axis(0)).into_owned().into_dyn()
                            } else {
                                a_ref.data.as_f64_view()
                            };
                            let b_2d: ArrayD<f64> = if b_ndim == 1 {
                                b_ref.data.view().insert_axis(ndarray::Axis(1)).into_owned().into_dyn()
                            } else {
                                b_ref.data.as_f64_view()
                            };
                            let grad_2d: ArrayD<f64> = if grad.ndim() == 1 {
                                if a_ndim == 1 {
                                    // result of (1,k)@(k,n) squeezed to (n,) → promote to (1,n)
                                    grad.view().insert_axis(ndarray::Axis(0)).into_owned().into_dyn()
                                } else {
                                    // result of (m,k)@(k,1) squeezed to (m,) → promote to (m,1)
                                    grad.view().insert_axis(ndarray::Axis(1)).into_owned().into_dyn()
                                }
                            } else if grad.ndim() == 2 {
                                grad.clone()
                            } else {
                                // 方向 A：grad.ndim() > 2 不再静默 clone（原代码会走进 matmul_2d 兜底成零数组）
                                return Err(crate::error::TenthError::RuntimeError {
                                    message: format!("MatMul 反向传播：grad ndim={} > 2 不支持（方向 A：不再静默兜底）", grad.ndim()),
                                });
                            };

                            // b_2d^T and a_2d^T
                            let b_t = b_2d.view().reversed_axes().to_owned();
                            let a_t = a_2d.view().reversed_axes().to_owned();

                            let d_a_2d = matmul_2d(&grad_2d, &b_t)?;
                            let d_b_2d = matmul_2d(&a_t, &grad_2d)?;

                            // Squeeze gradients back to match original input shapes.
                            // 方向 A：1D squeeze 前校验 shape 符合预期（避免静默 squeeze 错误 shape）
                            let d_a: ArrayD<f64> = if a_ndim == 1 {
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
                            let d_b: ArrayD<f64> = if b_ndim == 1 {
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
                            (d_a, d_b)
                        };

                        propagate_grad(node, 0, &d_a, &mut node_grads)?;
                        propagate_grad(node, 1, &d_b, &mut node_grads)?;
                    }
                }
                TapeOp::Transpose => {
                    let g_a = {
                        let mut perm: Vec<usize> = (0..grad.ndim()).collect();
                        if perm.len() >= 2 {
                            let last = perm.len() - 1;
                            perm.swap(last - 1, last);
                        }
                        grad.view().permuted_axes(perm).to_owned()
                    };
                    propagate_grad(node, 0, &g_a, &mut node_grads)?;
                }
                TapeOp::Sum => {
                    let g_a = {
                        let a = node.input_tensors[0].borrow();
                        let a_shape = a.shape();
                        let s: f64 = grad.iter().sum::<f64>();
                        ArrayD::from_elem(IxDyn(&a_shape), s)
                    };
                    propagate_grad(node, 0, &g_a, &mut node_grads)?;
                }
                TapeOp::Mean => {
                    let g_a = {
                        let a = node.input_tensors[0].borrow();
                        let a_shape = a.shape();
                        let n = a.size() as f64;
                        let s: f64 = grad.iter().sum::<f64>() / n;
                        ArrayD::from_elem(IxDyn(&a_shape), s)
                    };
                    propagate_grad(node, 0, &g_a, &mut node_grads)?;
                }
                TapeOp::Exp => {
                    let g_a = {
                        let result_ref = node.input_tensors[1].borrow();
                        &grad * &result_ref.data
                    };
                    propagate_grad(node, 0, &g_a, &mut node_grads)?;
                }
                TapeOp::Log => {
                    let g_a = {
                        let a_ref = node.input_tensors[0].borrow();
                        &grad / &a_ref.data
                    };
                    propagate_grad(node, 0, &g_a, &mut node_grads)?;
                }
                TapeOp::Sigmoid => {
                    let g_a = {
                        let result_ref = node.input_tensors[1].borrow();
                        let y = &result_ref.data;
                        &grad * y * &y.mapv(|v| 1.0 - v)
                    };
                    propagate_grad(node, 0, &g_a, &mut node_grads)?;
                }
                TapeOp::BatchNorm => {
                    // Simplified BN backward:
                    // d(gamma) = sum(dY * x_hat), d(beta) = sum(dY)
                    // dX = (gamma / std) * (dY - mean(dY) - x_hat * mean(dY * x_hat))
                    // input_tensors = [input, gamma, beta, x_hat, std_inv, result]
                    if node.input_tensors.len() >= 5 {
                        let gamma_ref = node.input_tensors[1].borrow();
                        let x_hat_ref = node.input_tensors[3].borrow();
                        let std_inv_ref = node.input_tensors[4].borrow();

                        // d(gamma)
                        let d_gamma = &grad * &x_hat_ref.data;
                        // d(beta)
                        let d_beta = grad.clone();

                        // dX = gamma * std_inv * (dY - mean(dY) - x_hat * mean(dY * x_hat))
                        let n = grad.len() as f64;
                        let mean_dy = grad.sum() / n;
                        let mean_dy_xhat = (&grad * &x_hat_ref.data).sum() / n;
                        let d_x = &std_inv_ref.data * &gamma_ref.data *
                            &(&grad - mean_dy - &(&x_hat_ref.data * mean_dy_xhat));

                        propagate_grad(node, 0, &d_x, &mut node_grads)?;
                        propagate_grad(node, 1, &d_gamma, &mut node_grads)?;
                        propagate_grad(node, 2, &d_beta, &mut node_grads)?;
                    }
                }
                TapeOp::LayerNorm => {
                    // LayerNorm backward (over last dim):
                    // d(gamma) = sum_over_outer(dY * x_hat), d(beta) = sum_over_outer(dY)
                    // dX = gamma * std_inv * (dY - mean(dY) - x_hat * mean(dY * x_hat))  [per-row]
                    // input_tensors = [input, gamma, beta, x_hat, std_inv, result]
                    if node.input_tensors.len() >= 5 {
                        let x_hat_ref = node.input_tensors[3].borrow();
                        let std_inv_ref = node.input_tensors[4].borrow();
                        let gamma_ref = node.input_tensors[1].borrow();

                        let x_hat_shape = x_hat_ref.shape();
                        let ndim = x_hat_shape.len();
                        let axis_len = x_hat_shape[ndim - 1];
                        let outer_len: usize = x_hat_shape[..ndim - 1].iter().product();

                        let x_hat_flat = x_hat_ref.data.as_standard_layout().to_owned();
                        let x_hat_slice = x_hat_flat.as_slice().unwrap_or(&[]);
                        let std_inv_flat = std_inv_ref.data.as_standard_layout().to_owned();
                        let std_inv_slice = std_inv_flat.as_slice().unwrap_or(&[]);
                        let grad_flat = grad.as_standard_layout().to_owned();
                        let grad_slice = grad_flat.as_slice().unwrap_or(&[]);
                        let g_flat = gamma_ref.data.as_standard_layout().to_owned();
                        let g_slice = g_flat.as_slice().unwrap_or(&[]);

                        // d(gamma): sum over outer dims of dY * x_hat
                        let mut d_gamma_data = vec![0.0f64; axis_len];
                        for i in 0..outer_len {
                            let start = i * axis_len;
                            for j in 0..axis_len {
                                d_gamma_data[j] += grad_slice.get(start + j).copied().unwrap_or(0.0)
                                    * x_hat_slice.get(start + j).copied().unwrap_or(0.0);
                            }
                        }
                        let d_gamma = ArrayD::from_shape_vec(IxDyn(&[axis_len]), d_gamma_data).unwrap();

                        // d(beta): sum over outer dims of dY
                        let mut d_beta_data = vec![0.0f64; axis_len];
                        for i in 0..outer_len {
                            let start = i * axis_len;
                            for j in 0..axis_len {
                                d_beta_data[j] += grad_slice.get(start + j).copied().unwrap_or(0.0);
                            }
                        }
                        let d_beta = ArrayD::from_shape_vec(IxDyn(&[axis_len]), d_beta_data).unwrap();

                        // dX per row: gamma * std_inv * (dY - mean(dY) - x_hat * mean(dY * x_hat))
                        let mut d_x_data = Vec::with_capacity(grad_slice.len());
                        for i in 0..outer_len {
                            let start = i * axis_len;
                            let inv = std_inv_slice.get(i).copied().unwrap_or(1.0);
                            let mut mean_dy = 0.0;
                            let mut mean_dy_xhat = 0.0;
                            for j in 0..axis_len {
                                let dy = grad_slice.get(start + j).copied().unwrap_or(0.0);
                                let xh = x_hat_slice.get(start + j).copied().unwrap_or(0.0);
                                mean_dy += dy;
                                mean_dy_xhat += dy * xh;
                            }
                            mean_dy /= axis_len as f64;
                            mean_dy_xhat /= axis_len as f64;
                            for j in 0..axis_len {
                                let dy = grad_slice.get(start + j).copied().unwrap_or(0.0);
                                let xh = x_hat_slice.get(start + j).copied().unwrap_or(0.0);
                                let g = g_slice.get(j).copied().unwrap_or(1.0);
                                d_x_data.push(g * inv * (dy - mean_dy - xh * mean_dy_xhat));
                            }
                        }
                        let d_x = ArrayD::from_shape_vec(IxDyn(&x_hat_shape), d_x_data).unwrap();

                        propagate_grad(node, 0, &d_x, &mut node_grads)?;
                        propagate_grad(node, 1, &d_gamma, &mut node_grads)?;
                        propagate_grad(node, 2, &d_beta, &mut node_grads)?;
                    }
                }
                TapeOp::Gelu => {
                    // GELU backward: d(gelu(x))/dx = 0.5 * (1 + tanh(inner)) + 0.5 * x * sech^2(inner) * sqrt(2/pi) * (1 + 3*0.044715*x^2)
                    // inner = sqrt(2/pi) * (x + 0.044715 * x^3)
                    // input_tensors = [input, result]
                    let g_a = {
                        let x_ref = node.input_tensors[0].borrow();
                        let sqrt_2_over_pi = (2.0 / std::f64::consts::PI).sqrt();
                        let x_data = &x_ref.data;
                        let deriv = x_data.mapv(|x| {
                            let inner = sqrt_2_over_pi * (x + 0.044715 * x * x * x);
                            let tanh_inner = inner.tanh();
                            let sech2 = 1.0 - tanh_inner * tanh_inner;
                            0.5 * (1.0 + tanh_inner) + 0.5 * x * sech2 * sqrt_2_over_pi * (1.0 + 3.0 * 0.044715 * x * x)
                        });
                        &grad * &deriv
                    };
                    propagate_grad(node, 0, &g_a, &mut node_grads)?;
                }
                TapeOp::Conv2D => {
                    // input_tensors = [input(4D), weight(4D), im2col(2D), output(4D)]
                    // Forward: output = im2col @ w_flat^T  where w_flat = weight.reshape(C_out, C_in*kH*kW)
                    // dW_flat = im2col^T @ dY        → reshape back to (C_out, C_in, kH, kW)
                    // d(im2col) = dY @ w_flat        → col2im back to input shape
                    // dY (upstream grad) has output shape (N, C_out, H_out, W_out);
                    // we reshape it to 2D (N*H_out*W_out, C_out) for matmul.
                    if node.input_tensors.len() >= 4 {
                        let (d_x, d_w) = {
                            let x_ref = node.input_tensors[0].borrow();
                            let w_ref = node.input_tensors[1].borrow();
                            let col_ref = node.input_tensors[2].borrow();
                            let out_ref = node.input_tensors[3].borrow();

                            let out_shape = out_ref.shape();
                            // output is (N, C_out, H_out, W_out)
                            let n = out_shape[0];
                            let c_out = out_shape[1];
                            let hw_out = out_shape[2] * out_shape[3];

                            // Reshape grad to 2D (N*H_out*W_out, C_out)
                            // 方向 A：reshape 失败不再静默 fallback，直接报错
                            let grad_2d: ArrayD<f64> = {
                                let v: Vec<f64> = grad.iter().cloned().collect();
                                ArrayD::from_shape_vec(IxDyn(&[hw_out * n, c_out]), v).map_err(|_| {
                                    crate::error::TenthError::RuntimeError {
                                        message: format!("Conv2D 反向 reshape grad 失败（方向 A：不再静默 fallback）"),
                                    }
                                })?
                            };

                            // im2col is (N*H_out*W_out, C_in*kH*kW)
                            let col_data = &col_ref.data;
                            // dW_flat = im2col^T @ dY  → (C_in*kH*kW, C_out)
                            let col_t = col_data.view().reversed_axes().to_owned();
                            let d_w_flat = matmul_2d(&col_t, &grad_2d)?;

                            // Reshape d_w_flat back to weight shape (C_out, C_in, kH, kW)
                            // Note: d_w_flat is (C_in*kH*kW, C_out), so transpose first
                            let d_w_flat_t = d_w_flat.view().reversed_axes().to_owned();
                            let w_shape = w_ref.shape();
                            // 方向 A：dW reshape 失败不再静默保留错误 shape，直接报错
                            let d_w: ArrayD<f64> = {
                                let total: usize = w_shape.iter().product();
                                if d_w_flat_t.len() != total {
                                    return Err(crate::error::TenthError::RuntimeError {
                                        message: format!(
                                            "Conv2D 反向 dW 元素数不匹配：{} != {}（方向 A：不再静默保留错误 shape）",
                                            d_w_flat_t.len(), total
                                        ),
                                    });
                                }
                                ArrayD::from_shape_vec(IxDyn(&w_shape), d_w_flat_t.iter().cloned().collect()).map_err(|_| {
                                    crate::error::TenthError::RuntimeError {
                                        message: format!("Conv2D 反向 dW reshape 失败（方向 A）"),
                                    }
                                })?
                            };

                            // d(im2col) = dY @ w_flat  → (N*H_out*W_out, C_in*kH*kW)
                            // w_flat = weight.reshape(C_out, C_in*kH*kW)
                            let w_flat: ArrayD<f64> = {
                                let v: Vec<f64> = w_ref.data.iter().collect();
                                ArrayD::from_shape_vec(IxDyn(&[c_out, w_shape[1] * w_shape[2] * w_shape[3]]), v).map_err(|_| {
                                    crate::error::TenthError::RuntimeError {
                                        message: format!("Conv2D 反向 w_flat reshape 失败（方向 A）"),
                                    }
                                })?
                            };
                            let d_col = matmul_2d(&grad_2d, &w_flat)?;

                            // col2im: accumulate d_col back into input shape (N, C_in, H, W)
                            // 方向 A：d_col 元素数不匹配时不再静默返回零数组，直接报错
                            let x_shape = x_ref.shape();
                            let d_x: ArrayD<f64> = {
                                let x_total: usize = x_shape.iter().product();
                                if d_col.len() != x_total {
                                    return Err(crate::error::TenthError::RuntimeError {
                                        message: format!(
                                            "Conv2D 反向 dX 元素数不匹配：d_col {} != x_total {}（方向 A：不再静默返回零数组）",
                                            d_col.len(), x_total
                                        ),
                                    });
                                }
                                ArrayD::from_shape_vec(IxDyn(&x_shape), d_col.iter().cloned().collect()).map_err(|_| {
                                    crate::error::TenthError::RuntimeError {
                                        message: format!("Conv2D 反向 dX reshape 失败（方向 A）"),
                                    }
                                })?
                            };
                            (d_x, d_w)
                        };

                        propagate_grad(node, 0, &d_x, &mut node_grads)?;
                        propagate_grad(node, 1, &d_w, &mut node_grads)?;
                    }
                }
                TapeOp::Dropout => {
                    // d(dropout(x))/dx = mask * dY
                    // input_tensors = [input, mask, result]
                    if node.input_tensors.len() >= 2 {
                        let g_a = {
                            let mask_ref = node.input_tensors[1].borrow();
                            &grad * &mask_ref.data
                        };
                        propagate_grad(node, 0, &g_a, &mut node_grads)?;
                    }
                }
                TapeOp::CrossEntropy => {
                    // d(CE)/d(logits) = softmax - target
                    // input_tensors = [logits, softmax_output, target]
                    if node.input_tensors.len() >= 3 {
                        let g_a = {
                            let sm_ref = node.input_tensors[1].borrow();
                            let tgt_ref = node.input_tensors[2].borrow();
                            &sm_ref.data - &tgt_ref.data
                        };
                        propagate_grad(node, 0, &g_a, &mut node_grads)?;
                    }
                }
                TapeOp::Softmax => {
                    // d(softmax(x)_i)/dx_j = y_i * (δ_ij - y_j)
                    // Chain rule: g_i = y_i * (grad_i - sum_j(grad_j * y_j))
                    let g_a = {
                        let result_ref = node.input_tensors[1].borrow();
                        let y = &result_ref.data;
                        let sum_term = (&grad * y).sum();
                        &grad * y - &(y.mapv(|v| v * sum_term))
                    };
                    propagate_grad(node, 0, &g_a, &mut node_grads)?;
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
fn acc_node_grad(node_grads: &mut [Option<ArrayD<f64>>], id: usize, g: &ArrayD<f64>) {
    match &mut node_grads[id] {
        Some(existing) => {
            *existing = &*existing + g;
        }
        slot @ None => {
            *slot = Some(g.clone());
        }
    }
}

/// Propagate gradient to input `input_idx` of a node.
/// If the node has upstream node ids, write to `node_grads` so DAG traversal
/// continues.  Otherwise, write directly to the tensor's `.grad` field
/// (used by `_direct` variants that bypass the node-graph).
/// 返回 Err 当 direct 路径的 acc_grad 报告 shape 不匹配（方向 A）。
fn propagate_grad(
    node: &TapeNode,
    input_idx: usize,
    g: &ArrayD<f64>,
    node_grads: &mut [Option<ArrayD<f64>>],
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
    }
}

/// Reduce `grad` from the output shape down to `target_shape` by summing
/// over broadcast dimensions.  Follows numpy-style broadcasting rules.
/// 返回 Err 当 reshape 失败（方向 A：不再静默保留错误 shape）。
fn unbroadcast(grad: &ArrayD<f64>, target_shape: &[usize]) -> Result<ArrayD<f64>, crate::error::TenthError> {
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
fn matmul_2d(a: &ArrayD<f64>, b: &ArrayD<f64>) -> Result<ArrayD<f64>, crate::error::TenthError> {
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
        assert!((a_grad[[0]] - 4.0).abs() < 1e-10, "a grad[0] = {}", a_grad[[0]]);
        assert!((a_grad[[1]] - 5.0).abs() < 1e-10, "a grad[1] = {}", a_grad[[1]]);

        // d(a*b)/db = a = [2, 3]
        let b_grad = b.borrow().grad.clone().unwrap();
        assert!((b_grad[[0]] - 2.0).abs() < 1e-10, "b grad[0] = {}", b_grad[[0]]);
        assert!((b_grad[[1]] - 3.0).abs() < 1e-10, "b grad[1] = {}", b_grad[[1]]);
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
        assert!((w_grad[[0]] - 1.0).abs() < 1e-10);
        assert!((w_grad[[1]] - 1.0).abs() < 1e-10);

        // b had shape [1], gradient should be sum of upstream over broadcast dim: 1+1=2
        let b_grad = b.borrow().grad.clone().unwrap();
        assert!((b_grad[[0]] - 2.0).abs() < 1e-10, "b_grad = {}", b_grad[[0]]);
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
        assert!((grad[[0]] - 0.0).abs() < 1e-10);
        assert!((grad[[1]] - 1.0).abs() < 1e-10);
        assert!((grad[[2]] - 0.0).abs() < 1e-10);
        assert!((grad[[3]] - 1.0).abs() < 1e-10);
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
        assert_eq!(a_grad.shape(), &[2, 3]);

        // d(a @ b)/db = a^T @ 1, a^T (3x2) @ 1 (2x2) → (3x2)
        let b_grad = b.borrow().grad.clone().unwrap();
        assert_eq!(b_grad.shape(), &[3, 2]);
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
        // d(relu(x)*2)/dx = 2 if x > 0 else 0
        assert!((grad[[0]] - 0.0).abs() < 1e-10, "x[0] grad = {}", grad[[0]]);
        assert!((grad[[1]] - 2.0).abs() < 1e-10, "x[1] grad = {}", grad[[1]]);
    }
}