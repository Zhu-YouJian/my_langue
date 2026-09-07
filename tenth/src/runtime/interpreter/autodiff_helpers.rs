//! 自动微分记录辅助函数：`record_binary` / `record_unary`。
//!
//! 从 `mod.rs` 拆出（架构重构 T3e），在 tape 录制开启时把张量运算记录到
//! 计算图节点上。供 `binary.rs` 与 `methods.rs` 中的张量算子调用。
//!
//! P2/B4：`record_binary` / `record_unary` 的实现已收敛到共享模块
//! `runtime::autodiff::record`（`crate::runtime::autodiff::{record_binary, record_unary}`），
//! 解释器与 VM 均委托到该单点实现——`TapeOp` / `tape.input` 语义调整只需改一处。

use std::rc::Rc;
use std::cell::RefCell;
use crate::runtime::tensor::Tensor;
use crate::runtime::autodiff::TapeOp;
use crate::runtime::autodiff::record_binary as tape_record_binary;
use crate::runtime::autodiff::record_unary as tape_record_unary;

impl super::Interpreter {
    pub(super) fn record_binary(&mut self, op: TapeOp, t1: &Rc<RefCell<Tensor>>, t2: &Rc<RefCell<Tensor>>, result: &Rc<RefCell<Tensor>>) {
        tape_record_binary(&mut self.tape, op, t1, t2, result);
    }

    pub(super) fn record_unary(&mut self, op: TapeOp, input: &Rc<RefCell<Tensor>>, result: &Rc<RefCell<Tensor>>) {
        tape_record_unary(&mut self.tape, op, input, result);
    }
}
