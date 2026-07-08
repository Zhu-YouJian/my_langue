//! 梯度累积与传播辅助函数。
//!
//! 从 `autodiff.rs` 拆分而来（T3c 架构重构），保持原有可见性与语义不变。
//! 这些函数原本是模块私有 `fn`，拆分后改为 `pub(super) fn` 以便 `backward.rs` 调用。

use crate::runtime::tensor::TensorData;
use super::tape_op::{TapeNode, TapeOp};

/// Accumulate `g` into `node_grads[id]`, adding if a gradient already exists.
/// 阶段 4：支持 TensorData 累加（同 dtype 直接加，异 dtype 提升为 f64）。
/// Phase 2：F16/BF16 grad 统一提升为 F32 中间表示累加（AMP 策略），
/// 避免 F16 溢出（max≈65504）和 BF16 精度损失。
pub(super) fn acc_node_grad(node_grads: &mut [Option<TensorData>], id: usize, g: &TensorData) {
    let existing = node_grads[id].take();
    // Phase 2: F16/BF16 统一提升为 F32 中间表示（AMP 策略）
    let existing = existing.map(|cur| match cur {
        TensorData::F16(a) => TensorData::F32(a.mapv(|v| v.to_f32())),
        TensorData::BF16(a) => TensorData::F32(a.mapv(|v| v.to_f32())),
        other => other,
    });
    let g_normalized: TensorData = match g {
        TensorData::F16(a) => TensorData::F32(a.mapv(|v| v.to_f32())),
        TensorData::BF16(a) => TensorData::F32(a.mapv(|v| v.to_f32())),
        other => other.clone(),
    };
    node_grads[id] = match (existing, &g_normalized) {
        (Some(TensorData::F64(cur)), TensorData::F64(g2)) => Some(TensorData::F64(&cur + g2)),
        (Some(TensorData::F32(cur)), TensorData::F32(g2)) => Some(TensorData::F32(&cur + g2)),
        (Some(TensorData::F64(cur)), TensorData::F32(g2)) => Some(TensorData::F64(&cur + &g2.mapv(|v| v as f64))),
        (Some(TensorData::F32(cur)), TensorData::F64(g2)) => Some(TensorData::F64(&cur.mapv(|v| v as f64) + g2)),
        // 兜底：异 dtype 混合场景提升为 f64
        (Some(cur), g2) => Some(TensorData::F64(cur.as_f64_view() + g2.as_f64_view())),
        (None, _) => Some(g_normalized),
    };
}

/// Propagate gradient to input `input_idx` of a node.
/// If the node has upstream node ids, write to `node_grads` so DAG traversal
/// continues.  Otherwise, write directly to the tensor's `.grad` field
/// (used by `_direct` variants that bypass the node-graph).
/// 返回 Err 当 direct 路径的 acc_grad 报告 shape 不匹配（方向 A）。
/// 阶段 4：g 参数从 &ArrayD<f64> 改为 &TensorData，支持按 dtype 存储。
pub(super) fn propagate_grad(
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
