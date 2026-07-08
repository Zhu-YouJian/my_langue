//! 自动微分记录辅助函数：`record_binary` / `record_unary`。
//!
//! 从 `mod.rs` 拆出（架构重构 T3e），在 tape 录制开启时把张量运算记录到
//! 计算图节点上。供 `binary.rs` 与 `methods.rs` 中的张量算子调用。
//!
//! 注意：vm.rs 中的 `Vm` 也有同名方法 `record_binary` / `record_unary`
//! （在 vm.rs 内部 impl Vm），属于 VM 路径独立的实现，互不影响。

use std::rc::Rc;
use std::cell::RefCell;
use crate::runtime::tensor::Tensor;
use crate::runtime::autodiff::TapeOp;

impl super::Interpreter {
    pub(super) fn record_binary(&mut self, op: TapeOp, t1: &Rc<RefCell<Tensor>>, t2: &Rc<RefCell<Tensor>>, result: &Rc<RefCell<Tensor>>) {
        if let Some(ref mut tape) = self.tape {
            let id1 = t1.borrow().tape_id;
            let id2 = t2.borrow().tape_id;
            let node_id = match (id1, id2) {
                (Some(a), Some(b)) => tape.binary(op, a, b, t1.clone(), t2.clone(), result.clone()),
                (Some(a), None) => {
                    let dummy = tape.input(t2.clone());
                    tape.binary(op, a, dummy, t1.clone(), t2.clone(), result.clone())
                }
                (None, Some(b)) => {
                    let dummy = tape.input(t1.clone());
                    tape.binary(op, dummy, b, t1.clone(), t2.clone(), result.clone())
                }
                (None, None) => tape.binary_direct(op, t1.clone(), t2.clone(), result.clone()),
            };
            result.borrow_mut().tape_id = Some(node_id);
        }
    }

    pub(super) fn record_unary(&mut self, op: TapeOp, input: &Rc<RefCell<Tensor>>, result: &Rc<RefCell<Tensor>>) {
        if let Some(ref mut tape) = self.tape {
            let node_id = match input.borrow().tape_id {
                Some(input_id) => tape.unary(op, input_id, input.clone(), result.clone()),
                None => {
                    // Create dummy input so the DAG stays connected
                    let dummy = tape.input(input.clone());
                    tape.unary(op, dummy, input.clone(), result.clone())
                }
            };
            result.borrow_mut().tape_id = Some(node_id);
        }
    }
}
