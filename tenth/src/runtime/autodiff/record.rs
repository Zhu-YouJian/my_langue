//! 共享自动微分记录辅助函数：`record_binary` / `record_unary`。
//!
//! 从 `interpreter/autodiff_helpers.rs` 与 `vm/execute.rs` 收敛而来（P2/B4）。
//! 此前两者几乎逐字复制（含 dummy input 与 binary_direct/input 分支），
//! 现统一到此处——`TapeOp` / `tape.input` 语义一旦调整只需改一处，
//! 解释器（Interpreter）与 VM 的 `record_*` 均委托到本实现。
//!
//! 这些是自由函数（非方法）：`self.tape` 由调用方以 `&mut Option<Tape>` 传入，
//! 因此不依赖 `Interpreter` / `Vm` 的具体类型，天然可作为单点共享。

use std::rc::Rc;
use std::cell::RefCell;
use super::Tape;
use super::TapeOp;
use crate::runtime::tensor::Tensor;

/// 在 tape 录制开启时记录一元运算。`tape` 为 `None`（未录制）时不做任何事。
/// 录制成功后把 node_id 写入 `result.tape_id`（与调用方原语义一致）。
pub(crate) fn record_unary(
    tape: &mut Option<Tape>,
    op: TapeOp,
    input: &Rc<RefCell<Tensor>>,
    result: &Rc<RefCell<Tensor>>,
) {
    if let Some(tape) = tape.as_mut() {
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

/// 在 tape 录制开启时记录二元运算。`tape` 为 `None`（未录制）时不做任何事。
/// 按两个输入的 tape_id 组合分派：双 Some → `binary`；单 Some → 为 None 侧补
/// dummy input（保持 DAG 连通）后 `binary`；双 None → `binary_direct`。
/// 录制成功后把 node_id 写入 `result.tape_id`（与调用方原语义一致）。
pub(crate) fn record_binary(
    tape: &mut Option<Tape>,
    op: TapeOp,
    t1: &Rc<RefCell<Tensor>>,
    t2: &Rc<RefCell<Tensor>>,
    result: &Rc<RefCell<Tensor>>,
) {
    if let Some(tape) = tape.as_mut() {
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
