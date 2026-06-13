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
    pub fn backward(&self, loss_node_id: usize) {
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
                    node.input_tensors[0].borrow_mut().acc_grad(&grad);
                }
                TapeOp::Add | TapeOp::Sub => {
                    let sign: f64 = if node.op == TapeOp::Add { 1.0 } else { -1.0 };
                    let shapes: Vec<Vec<usize>> = (0..node.input_tensors.len().min(2))
                        .map(|i| node.input_tensors[i].borrow().shape())
                        .collect();
                    for (i, input_shape) in shapes.iter().enumerate() {
                        let g_i = if i == 0 {
                            unbroadcast(&grad, input_shape)
                        } else {
                            unbroadcast(&grad, input_shape).mapv(|v| v * sign)
                        };
                        propagate_grad(node, i, &g_i, &mut node_grads);
                    }
                }
                TapeOp::Mul => {
                    // Clone input data first to avoid holding RefCell borrows.
                    let (a_data, a_shape, b_data, b_shape) = {
                        let a = node.input_tensors[0].borrow();
                        let b = node.input_tensors[1].borrow();
                        (a.data.clone(), a.shape(), b.data.clone(), b.shape())
                    };
                    let ga = unbroadcast(&(&grad * &b_data), &a_shape);
                    let gb = unbroadcast(&(&grad * &a_data), &b_shape);
                    propagate_grad(node, 0, &ga, &mut node_grads);
                    propagate_grad(node, 1, &gb, &mut node_grads);
                }
                TapeOp::Div => {
                    let (a_data, a_shape, b_data, b_shape) = {
                        let a = node.input_tensors[0].borrow();
                        let b = node.input_tensors[1].borrow();
                        (a.data.clone(), a.shape(), b.data.clone(), b.shape())
                    };
                    let ga = unbroadcast(&(&grad / &b_data), &a_shape);
                    let gb = unbroadcast(&(-&grad * &a_data / (&b_data * &b_data)), &b_shape);
                    propagate_grad(node, 0, &ga, &mut node_grads);
                    propagate_grad(node, 1, &gb, &mut node_grads);
                }
                TapeOp::Neg => {
                    let g = -&grad;
                    propagate_grad(node, 0, &g, &mut node_grads);
                }
                TapeOp::ReLU => {
                    let mask = {
                        let a = node.input_tensors[0].borrow();
                        a.data.mapv(|x| if x > 0.0 { 1.0 } else { 0.0 })
                    };
                    let g_a = &grad * &mask;
                    propagate_grad(node, 0, &g_a, &mut node_grads);
                }
                TapeOp::MatMul => {
                    if node.input_tensors.len() >= 3 {
                        let result = {
                            let a = node.input_tensors[0].borrow();
                            let b = node.input_tensors[1].borrow();
                            let b_t = b.transpose();
                            let a_t = a.transpose();
                            match (b_t, a_t) {
                                (Some(b_t), Some(a_t)) => {
                                    let d_a = matmul_2d(&grad, &b_t.data);
                                    let d_b = matmul_2d(&a_t.data, &grad);
                                    Some((d_a, d_b))
                                }
                                _ => None,
                            }
                        };
                        if let Some((d_a, d_b)) = result {
                            propagate_grad(node, 0, &d_a, &mut node_grads);
                            propagate_grad(node, 1, &d_b, &mut node_grads);
                        }
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
                    propagate_grad(node, 0, &g_a, &mut node_grads);
                }
                TapeOp::Sum => {
                    let g_a = {
                        let a = node.input_tensors[0].borrow();
                        let a_shape = a.shape();
                        let s: f64 = grad.iter().sum::<f64>();
                        ArrayD::from_elem(IxDyn(&a_shape), s)
                    };
                    propagate_grad(node, 0, &g_a, &mut node_grads);
                }
                TapeOp::Mean => {
                    let g_a = {
                        let a = node.input_tensors[0].borrow();
                        let a_shape = a.shape();
                        let n = a.size() as f64;
                        let s: f64 = grad.iter().sum::<f64>() / n;
                        ArrayD::from_elem(IxDyn(&a_shape), s)
                    };
                    propagate_grad(node, 0, &g_a, &mut node_grads);
                }
                TapeOp::Exp => {
                    let g_a = {
                        let result_ref = node.input_tensors[1].borrow();
                        &grad * &result_ref.data
                    };
                    propagate_grad(node, 0, &g_a, &mut node_grads);
                }
                TapeOp::Log => {
                    let g_a = {
                        let a_ref = node.input_tensors[0].borrow();
                        &grad / &a_ref.data
                    };
                    propagate_grad(node, 0, &g_a, &mut node_grads);
                }
                TapeOp::Sigmoid => {
                    let g_a = {
                        let result_ref = node.input_tensors[1].borrow();
                        let y = &result_ref.data;
                        &grad * y * &y.mapv(|v| 1.0 - v)
                    };
                    propagate_grad(node, 0, &g_a, &mut node_grads);
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

                        propagate_grad(node, 0, &d_x, &mut node_grads);
                        propagate_grad(node, 1, &d_gamma, &mut node_grads);
                        propagate_grad(node, 2, &d_beta, &mut node_grads);
                    }
                }
                TapeOp::Conv2D => {
                    // input_tensors = [input, weight, im2col, output]
                    // dY = upstream gradient (same shape as output: N*H_out*W_out, C_out)
                    // dW = im2col^T @ dY
                    // d(im2col) = dY @ W^T → then col2im back to input shape
                    if node.input_tensors.len() >= 3 {
                        let d_w = {
                            let col_ref = node.input_tensors[2].borrow();
                            let col_t = col_ref.transpose().map(|t| t.data).unwrap_or_default();
                            matmul_2d(&col_t, &grad)
                        };
                        let d_x = {
                            let w_ref = node.input_tensors[1].borrow();
                            let w_t = w_ref.transpose();
                            if let Some(w_t) = w_t {
                                let d_col = matmul_2d(&grad, &w_t.data);
                                d_col
                            } else {
                                ArrayD::zeros(IxDyn(&[1, 1]))
                            }
                        };
                        propagate_grad(node, 0, &d_x, &mut node_grads);
                        propagate_grad(node, 1, &d_w, &mut node_grads);
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
                        propagate_grad(node, 0, &g_a, &mut node_grads);
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
                        propagate_grad(node, 0, &g_a, &mut node_grads);
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
                    propagate_grad(node, 0, &g_a, &mut node_grads);
                }
            }
        }
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
fn propagate_grad(
    node: &TapeNode,
    input_idx: usize,
    g: &ArrayD<f64>,
    node_grads: &mut [Option<ArrayD<f64>>],
) {
    if input_idx < node.inputs.len() {
        acc_node_grad(node_grads, node.inputs[input_idx], g);
    } else {
        if let Some(t) = node.input_tensors.get(input_idx) {
            t.borrow_mut().acc_grad(g);
        }
    }
}

/// Reduce `grad` from the output shape down to `target_shape` by summing
/// over broadcast dimensions.  Follows numpy-style broadcasting rules.
fn unbroadcast(grad: &ArrayD<f64>, target_shape: &[usize]) -> ArrayD<f64> {
    let grad_shape = grad.shape();
    if grad_shape == target_shape {
        return grad.clone();
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
            result = result.clone().into_shape_with_order(IxDyn(target_shape)).unwrap_or(result);
        }
    }

    result
}

/// Pure 2-D matrix multiplication returning an owned ArrayD.
fn matmul_2d(a: &ArrayD<f64>, b: &ArrayD<f64>) -> ArrayD<f64> {
    let a2 = a.view().into_dimensionality::<ndarray::Ix2>().unwrap();
    let b2 = b.view().into_dimensionality::<ndarray::Ix2>().unwrap();
    a2.dot(&b2).into_dyn()
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

        tape.backward(r_id);

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

        tape.backward(r_id);

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

        tape.backward(r_id);

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

        tape.backward(r_id);

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

        tape.backward(r_id);

        let grad = x.borrow().grad.clone().unwrap();
        // d(relu(x)*2)/dx = 2 if x > 0 else 0
        assert!((grad[[0]] - 0.0).abs() < 1e-10, "x[0] grad = {}", grad[[0]]);
        assert!((grad[[1]] - 2.0).abs() < 1e-10, "x[1] grad = {}", grad[[1]]);
    }
}