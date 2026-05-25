use std::collections::HashMap;
use std::cell::RefCell;
use crate::runtime::value::Value;

/// A node in the computation graph (tape).
#[derive(Debug, Clone)]
pub struct TapeNode {
    /// Unique ID for this node
    pub id: usize,
    /// The operation that produced this node
    pub op: TapeOp,
    /// Input node IDs that this operation depends on
    pub inputs: Vec<usize>,
    /// The value computed during forward pass (cached)
    pub value: Value,
}

#[derive(Debug, Clone)]
pub enum TapeOp {
    /// Leaf: a parameter or input (no inputs)
    Input,
    /// Addition: a + b
    Add,
    /// Subtraction: a - b
    Sub,
    /// Multiplication: a * b
    Mul,
    /// Division: a / b
    Div,
    /// Negation: -a
    Neg,
    /// ReLU: max(0, a)
    ReLU,
    /// Sum over all elements
    Sum,
    /// Matrix multiplication: a @ b
    MatMul,
    /// Exponential: exp(a)
    Exp,
    /// Natural log: ln(a)
    Log,
}

/// A computation tape (Wengert tape).
pub struct Tape {
    nodes: Vec<TapeNode>,
    counter: RefCell<usize>,
}

impl Tape {
    pub fn new() -> Self {
        Tape {
            nodes: Vec::new(),
            counter: RefCell::new(0),
        }
    }

    /// Record an input/leaf node.
    pub fn input(&mut self, value: Value) -> TapeNode {
        let id = self.next_id();
        let node = TapeNode {
            id,
            op: TapeOp::Input,
            inputs: vec![],
            value,
        };
        self.nodes.push(node.clone());
        node
    }

    /// Record a unary operation.
    pub fn unary(&mut self, op: TapeOp, input: &TapeNode, value: Value) -> TapeNode {
        let id = self.next_id();
        let node = TapeNode {
            id,
            op,
            inputs: vec![input.id],
            value,
        };
        self.nodes.push(node.clone());
        node
    }

    /// Record a binary operation.
    pub fn binary(&mut self, op: TapeOp, a: &TapeNode, b: &TapeNode, value: Value) -> TapeNode {
        let id = self.next_id();
        let node = TapeNode {
            id,
            op,
            inputs: vec![a.id, b.id],
            value,
        };
        self.nodes.push(node.clone());
        node
    }

    fn next_id(&self) -> usize {
        let mut c = self.counter.borrow_mut();
        let id = *c;
        *c += 1;
        id
    }

    /// Run backward pass: compute gradients of `loss` w.r.t all inputs.
    /// Returns a map from node id to gradient.
    pub fn backward(&self, loss_node: &TapeNode) -> HashMap<usize, f64> {
        let mut grads: HashMap<usize, f64> = HashMap::new();
        // Initialize loss gradient to 1.0
        grads.insert(loss_node.id, 1.0);

        // Process nodes in reverse topological order
        for node in self.nodes.iter().rev() {
            let grad = *grads.get(&node.id).unwrap_or(&0.0);
            if grad == 0.0 {
                continue;
            }

            match &node.op {
                TapeOp::Input => {
                    // Input nodes don't need additional accumulation — 
                    // their gradient is already accumulated from downstream uses.
                    // Only pass through if there are no downstream nodes.
                    if node.inputs.is_empty() {
                        // Leaf: gradient already set by downstream
                    }
                }
                TapeOp::Add => {
                    // d(a+b)/da = 1, d(a+b)/db = 1
                    for &input_id in &node.inputs {
                        *grads.entry(input_id).or_insert(0.0) += grad;
                    }
                }
                TapeOp::Sub => {
                    // d(a-b)/da = 1, d(a-b)/db = -1
                    if node.inputs.len() == 2 {
                        *grads.entry(node.inputs[0]).or_insert(0.0) += grad;
                        *grads.entry(node.inputs[1]).or_insert(0.0) -= grad;
                    }
                }
                TapeOp::Mul => {
                    if node.inputs.len() == 2 {
                        let a_val = self.get_node_value(node.inputs[0]);
                        let b_val = self.get_node_value(node.inputs[1]);
                        // d(a*b)/da = b, d(a*b)/db = a
                        *grads.entry(node.inputs[0]).or_insert(0.0) += grad * b_val;
                        *grads.entry(node.inputs[1]).or_insert(0.0) += grad * a_val;
                    }
                }
                TapeOp::Div => {
                    if node.inputs.len() == 2 {
                        let a_val = self.get_node_value(node.inputs[0]);
                        let b_val = self.get_node_value(node.inputs[1]);
                        // d(a/b)/da = 1/b, d(a/b)/db = -a/b²
                        *grads.entry(node.inputs[0]).or_insert(0.0) += grad / b_val;
                        *grads.entry(node.inputs[1]).or_insert(0.0) -= grad * a_val / (b_val * b_val);
                    }
                }
                TapeOp::Neg => {
                    // d(-a)/da = -1
                    for &input_id in &node.inputs {
                        *grads.entry(input_id).or_insert(0.0) -= grad;
                    }
                }
                TapeOp::ReLU => {
                    // d(relu(a))/da = 1 if a > 0 else 0
                    if !node.inputs.is_empty() {
                        let a_val = self.get_node_value(node.inputs[0]);
                        if a_val > 0.0 {
                            *grads.entry(node.inputs[0]).or_insert(0.0) += grad;
                        }
                    }
                }
                TapeOp::Sum => {
                    // d(sum(a))/da = 1 for each element
                    for &input_id in &node.inputs {
                        *grads.entry(input_id).or_insert(0.0) += grad;
                    }
                }
                TapeOp::Exp => {
                    // d(exp(a))/da = exp(a) = node.value
                    for &input_id in &node.inputs {
                        *grads.entry(input_id).or_insert(0.0) += grad * self.get_node_value(node.id);
                    }
                }
                TapeOp::Log => {
                    // d(ln(a))/da = 1/a
                    if !node.inputs.is_empty() {
                        let a_val = self.get_node_value(node.inputs[0]);
                        if a_val != 0.0 {
                            *grads.entry(node.inputs[0]).or_insert(0.0) += grad / a_val;
                        }
                    }
                }
                TapeOp::MatMul => {
                    // Simplified: treat matmul like mul for now
                    // Real implementation needs tensor shapes
                    for &input_id in &node.inputs {
                        *grads.entry(input_id).or_insert(0.0) += grad;
                    }
                }
            }
        }
        grads
    }

    fn get_node_value(&self, id: usize) -> f64 {
        self.nodes.iter()
            .find(|n| n.id == id)
            .and_then(|n| n.value.as_float())
            .unwrap_or(0.0)
    }
}

/// Compute the gradient of a scalar function f(x) at point x.
/// Returns (f(x), df/dx).
pub fn grad_scalar<F>(f: F, x: f64) -> (f64, f64)
where
    F: Fn(&mut Tape, &TapeNode) -> TapeNode,
{
    let mut tape = Tape::new();
    let x_node = tape.input(Value::Float(x));
    let y_node = f(&mut tape, &x_node);
    let y_val = y_node.value.as_float().unwrap_or(0.0);
    let grads = tape.backward(&y_node);
    let dx = *grads.get(&x_node.id).unwrap_or(&0.0);
    (y_val, dx)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_grad_square() {
        // f(x) = x * x, df/dx = 2x
        let (y, dx) = grad_scalar(|tape: &mut Tape, x| {
            tape.binary(TapeOp::Mul, x, x, Value::Float(x.value.as_float().unwrap() * x.value.as_float().unwrap()))
        }, 3.0);
        assert!((y - 9.0).abs() < 1e-10, "f(3) should be 9.0, got {}", y);
        assert!((dx - 6.0).abs() < 1e-10, "f'(3) should be 6.0, got {}", dx);
    }

    #[test]
    fn test_grad_add_mul() {
        // f(x) = x * x + x, df/dx = 2x + 1
        let (y, dx) = grad_scalar(|tape: &mut Tape, x| {
            let x2 = tape.binary(TapeOp::Mul, x, x,
                Value::Float(x.value.as_float().unwrap() * x.value.as_float().unwrap()));
            tape.binary(TapeOp::Add, &x2, x,
                Value::Float(x2.value.as_float().unwrap() + x.value.as_float().unwrap()))
        }, 3.0);
        assert!((y - 12.0).abs() < 1e-10, "f(3) should be 12.0, got {}", y);
        assert!((dx - 7.0).abs() < 1e-10, "f'(3) should be 7.0, got {}", dx);
    }

    #[test]
    fn test_grad_relu() {
        // f(x) = relu(x), df/dx = 1 if x > 0 else 0
        let (y, dx) = grad_scalar(|tape: &mut Tape, x| {
            let v = x.value.as_float().unwrap();
            let r = if v > 0.0 { v } else { 0.0 };
            tape.unary(TapeOp::ReLU, x, Value::Float(r))
        }, 3.0);
        assert!((y - 3.0).abs() < 1e-10);
        assert!((dx - 1.0).abs() < 1e-10);

        let (y2, dx2) = grad_scalar(|tape: &mut Tape, x| {
            let v = x.value.as_float().unwrap();
            let r = if v > 0.0 { v } else { 0.0 };
            tape.unary(TapeOp::ReLU, x, Value::Float(r))
        }, -2.0);
        assert!((y2 - 0.0).abs() < 1e-10);
        assert!((dx2 - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_grad_exp() {
        // f(x) = exp(x), df/dx = exp(x)
        let x = 1.0;
        let (y, dx) = grad_scalar(|tape: &mut Tape, x_node| {
            let v = x_node.value.as_float().unwrap();
            tape.unary(TapeOp::Exp, x_node, Value::Float(v.exp()))
        }, x);
        assert!((y - x.exp()).abs() < 1e-10);
        assert!((dx - x.exp()).abs() < 1e-10);
    }
}
