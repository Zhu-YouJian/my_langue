//! Tensor-level automatic differentiation via a Wengert tape.
//!
//! Records operations on `Tensor`s during forward execution, then replays
//! the chain rule backward to populate each parameter tensor's `.grad` field.
//!
//! 模块拆分（T3c 架构重构）：
//! - `tape_op`：FloatElem trait + dispatch_float! 宏 + TapeNode + TapeOp 枚举
//! - `grad`：acc_node_grad + propagate_grad + op_name 辅助函数
//! - `backward`：impl Tape::backward + unbroadcast/matmul_2d/flatten_index
//! - `mod.rs`（本文件）：Tape 结构体 + impl Tape 的非 backward 方法 + clear/zero_grad + pub use + 测试

use std::rc::Rc;
use std::cell::RefCell;
use super::tensor::Tensor;
use crate::hir::types::BaseType;

mod tape_op;
mod grad;
mod backward;

pub use tape_op::{TapeNode, TapeOp};

// 护城河 F Phase 1：op_name 去重——re-export 供 relation_debugger 复用。
// grad 模块本身是私有子模块，通过此 re-export 暴露 op_name 到 crate 内。
pub(crate) use grad::op_name;

// ── Tape ──────────────────────────────────────────────────────────────

pub struct Tape {
    /// 节点列表。`pub(super)` 以便 `backward.rs` 的 `impl Tape::backward` 读取。
    /// 不变量：node.id == self.nodes 索引（护城河 F：relation_debugger 依赖）。
    pub(super) nodes: Vec<TapeNode>,
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

    // backward 方法实现在 `backward.rs` 中（`impl Tape` 跨文件扩展）。

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
