//! 梯度累积与传播辅助函数。
//!
//! 从 `autodiff.rs` 拆分而来（T3c 架构重构），保持原有可见性与语义不变。
//! 这些函数原本是模块私有 `fn`，拆分后改为 `pub(super) fn` 以便 `backward.rs` 调用。

use std::borrow::Cow;

use crate::runtime::tensor::TensorData;
use super::tape_op::{TapeNode, TapeOp};

/// Accumulate `g` into `node_grads[id]`, adding if a gradient already exists.
/// 阶段 4：支持 TensorData 累加（同 dtype 直接加，异 dtype 提升为 f64）。
/// Phase 2：F16/BF16 grad 统一提升为 F32 中间表示累加（AMP 策略），
/// 避免 F16 溢出（max≈65504）和 BF16 精度损失。
///
/// 护城河 F Phase 2：DtypeConflict 检测（保守策略）。
/// 当 existing 与 g 的 dtype 不一致（F32 vs F64，经 AMP 提升后）时，
/// 输出 warning 到 stderr。不改为 error 以避免破坏现有测试——保持现有的
/// 静默提升为 f64 的行为，仅增加可观测性。formal_explain 侧通过
/// `classify_error_type` 检测 error_msg 中的 "dtype"/"f32/f64" 关键词
/// 标记 DtypeConflict，与本检测互补。
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
    // 护城河 F Phase 2：DtypeConflict 检测（保守策略，仅 warning 不改 error）
    // AMP 提升后 existing/g_normalized 只剩 F32/F64；二者不一致即为 dtype 冲突。
    if let Some(ref cur) = existing {
        let cur_is_f64 = matches!(cur, TensorData::F64(_));
        let g_is_f64 = matches!(g_normalized, TensorData::F64(_));
        if cur_is_f64 != g_is_f64 {
            let cur_dt = if cur_is_f64 { "f64" } else { "f32" };
            let g_dt = if g_is_f64 { "f64" } else { "f32" };
            eprintln!(
                "警告（护城河 F DtypeConflict）：节点 #{} 梯度累积 dtype 不一致（existing={} vs new={}），已静默提升为 f64",
                id, cur_dt, g_dt
            );
        }
    }
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
            // 护城河 F Phase 1：结构化提取 (v_err=node.id, op=op_name, expected=input.shape, actual=grad.shape)
            // 注意：先取 input shape 再 borrow_mut，避免 RefCell 二次 borrow。
            let input_shape = t.borrow().shape();
            let grad_shape = g.shape().to_vec();
            let op_str = op_name(&node.op);
            t.borrow_mut().acc_grad(g).map_err(|e| {
                crate::error::TenthError::ShapeMismatch {
                    context: crate::error::TapeErrorContext {
                        tape_node_id: node.id,
                        op: op_str.to_string(),
                        expected_shape: input_shape,
                        actual_shape: grad_shape,
                    },
                    message: format!("反向传播 shape 错误（节点 #{} {} direct input {}）：{}", node.id, op_str, input_idx, e),
                }
            })?;
        }
    }
    Ok(())
}

/// 人类可读的 TapeOp 名称（用于错误信息）。
///
/// 护城河 F：`pub(crate)` 以便 `relation_debugger` 复用，避免重复实现
///（原先 `relation_debugger.rs` 有一份同步副本，现已删除）。
///
/// PROJ-006：返回类型从 `&'static str` 改为 `Cow<'static, str>`，
/// 因为 `TapeOp::Custom(op_id)` 需要动态生成名称（`Custom#{op_id}`）。
/// 调用方通过 `op_name(...).as_ref()` 或 `&op_name(...)` 取 `&str` 视图；
/// 算子实际名称（如 "square"）需在调用点通过 `CustomOpRegistry::get(op_id).name()` 获取，
/// 本函数仅返回 `Custom#id` 形式（无法访问 registry）。
pub(crate) fn op_name(op: &TapeOp) -> Cow<'static, str> {
    match op {
        TapeOp::Input => Cow::Borrowed("Input"),
        TapeOp::Add => Cow::Borrowed("Add"),
        TapeOp::Sub => Cow::Borrowed("Sub"),
        TapeOp::Mul => Cow::Borrowed("Mul"),
        TapeOp::Div => Cow::Borrowed("Div"),
        TapeOp::Neg => Cow::Borrowed("Neg"),
        TapeOp::ReLU => Cow::Borrowed("ReLU"),
        TapeOp::MatMul => Cow::Borrowed("MatMul"),
        TapeOp::BatchedMatMul => Cow::Borrowed("BatchedMatMul"),
        TapeOp::Transpose => Cow::Borrowed("Transpose"),
        TapeOp::Sum => Cow::Borrowed("Sum"),
        TapeOp::Mean => Cow::Borrowed("Mean"),
        TapeOp::Exp => Cow::Borrowed("Exp"),
        TapeOp::Log => Cow::Borrowed("Log"),
        TapeOp::Sigmoid => Cow::Borrowed("Sigmoid"),
        TapeOp::Softmax => Cow::Borrowed("Softmax"),
        TapeOp::CrossEntropy => Cow::Borrowed("CrossEntropy"),
        TapeOp::Dropout => Cow::Borrowed("Dropout"),
        TapeOp::Conv2D => Cow::Borrowed("Conv2D"),
        TapeOp::BatchNorm => Cow::Borrowed("BatchNorm"),
        TapeOp::LayerNorm => Cow::Borrowed("LayerNorm"),
        TapeOp::Gelu => Cow::Borrowed("Gelu"),
        TapeOp::Select => Cow::Borrowed("Select"),
        TapeOp::Abs => Cow::Borrowed("Abs"),
        TapeOp::Scatter => Cow::Borrowed("Scatter"),
        TapeOp::Gather => Cow::Borrowed("Gather"),
        TapeOp::Reshape => Cow::Borrowed("Reshape"),
        TapeOp::MaskedFill => Cow::Borrowed("MaskedFill"),
        TapeOp::MaxPool2D => Cow::Borrowed("MaxPool2D"),
        TapeOp::AvgPool2D => Cow::Borrowed("AvgPool2D"),
        // Custom 算子：返回 Custom#id（实际名称需通过 registry 查询）
        TapeOp::Custom(op_id) => Cow::Owned(format!("Custom#{}", op_id)),
    }
}
