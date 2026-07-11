//! 编译期反向 shape 验证（护城河 A 深化 Phase 1）。
//!
//! 在 lowering 阶段验证每个可微算子的反向梯度 shape 与前向输入 shape 兼容，
//! 把运行时 RuntimeError 提升到编译期 TypeError。
//!
//! 核心思路：对每个可微算子，根据前向输入/输出 shape 推导反向梯度 shape，
//! 并检查梯度 shape 能否 unbroadcast 回输入 shape（或满足算子特定的 shape 约束）。
//!
//! 保守策略：任一 shape 全 Any 时跳过检查（与现有 check_binary_shape_compat 一致）。
//! 参考运行时实现：`runtime/autodiff/backward.rs` 的 `unbroadcast` 逻辑。

use crate::error::{TenthError, TenthResult};
use crate::hir::types::Dim;
use crate::lexer::token::Span;
use super::types::{has_static_info, fmt_dims, fmt_dim};

/// 计算可微算子的反向梯度 shapes。
///
/// 输入：算子名、前向输入 shapes、前向输出 shape
/// 输出：每个可微输入的梯度 shape（不可微输入返回空 vec）
/// 返回 `Err(msg)` 表示反向 shape 与前向输入 shape 不兼容。
///
/// 规则表详见护城河 A Phase 1 任务说明，此处按算子逐一实现。
/// 跳过检查的算子（element-wise 一元 / softmax / dropout / transpose / conv2d）
/// 直接返回输入 shapes（梯度 shape 天然与输入一致）。
pub(super) fn backward_shape(
    op: &str,
    fwd_in_shapes: &[Vec<Dim>],
    fwd_out_shape: &[Dim],
) -> Result<Vec<Vec<Dim>>, String> {
    match op {
        // ── 二元算术（add/sub/mul/div）──────────────────────────────────
        // 反向：两个输入都得到 output_shape，需 unbroadcast 到各自 input_shape
        // 不兼容条件：unbroadcast(output_shape, input_shape) 不可行
        "add" | "sub" | "mul" | "div" => {
            let mut grads = Vec::with_capacity(fwd_in_shapes.len());
            for in_shape in fwd_in_shapes {
                if in_shape.is_empty() {
                    grads.push(vec![]);
                    continue;
                }
                unbroadcast_feasible(fwd_out_shape, in_shape)?;
                grads.push(fwd_out_shape.to_vec());
            }
            Ok(grads)
        }

        // ── matmul ──────────────────────────────────────────────────────
        // 前向: a(M,K) @ b(K,N) → out(M,N)
        // 反向: d_a(M,K), d_b(K,N) — 梯度 shape 天然与输入 shape 一致
        // 编译期假设 grad == output_shape（backward 从 loss 传播）；
        // 前向 K 一致性检查已由 check_method_shape 完成，此处总是可行
        "matmul" => {
            Ok(fwd_in_shapes.to_vec())
        }

        // ── bmm ─────────────────────────────────────────────────────────
        // 前向: a(B,M,K) @ b(B,K,N) → out(B,M,N)
        // 反向: d_a(B,M,K), d_b(B,K,N) — 同 matmul
        "bmm" => {
            Ok(fwd_in_shapes.to_vec())
        }

        // ── sum/mean ───────────────────────────────────────────────────
        // 反向: 梯度 broadcast 回 input_shape — 总是可行
        "sum" | "mean" => {
            Ok(fwd_in_shapes.to_vec())
        }

        // ── cross_entropy ─────────────────────────────────────────────
        // 前向: logits(B,V), target → scalar
        // 反向: d_logits = logits_shape；target 不传梯度
        // 不兼容条件：target shape 不是 [B] 或 [B,V]（当 logits 是 [B,V] 时）
        "cross_entropy" => {
            if fwd_in_shapes.len() < 2 {
                return Ok(vec![]);
            }
            let logits_shape = &fwd_in_shapes[0];
            let target_shape = &fwd_in_shapes[1];
            // 仅在 logits 是 2D [B, V] 且两侧都有静态信息时检查
            if logits_shape.len() == 2
                && has_static_info(logits_shape)
                && has_static_info(target_shape)
            {
                let b_dim = &logits_shape[0];
                let v_dim = &logits_shape[1];
                let valid = match target_shape.len() {
                    1 => dims_compatible(&target_shape[0], b_dim),
                    2 => {
                        dims_compatible(&target_shape[0], b_dim)
                            && dims_compatible(&target_shape[1], v_dim)
                    }
                    _ => false,
                };
                if !valid {
                    return Err(format!(
                        "cross_entropy target shape {} 与 logits shape {} 不兼容（应为 [{}] 或 [{}, {}]）",
                        fmt_dims(target_shape),
                        fmt_dims(logits_shape),
                        fmt_dim(b_dim),
                        fmt_dim(b_dim),
                        fmt_dim(v_dim)
                    ));
                }
            }
            // logits 得到自身 shape 的梯度；target 不传梯度（空 vec）
            Ok(vec![logits_shape.clone(), vec![]])
        }

        // ── reshape/view ──────────────────────────────────────────────
        // 反向: d_input = grad.reshape(input_shape)
        // 不兼容条件: output numel != input numel
        "reshape" | "view" => {
            if let Some(in_shape) = fwd_in_shapes.first() {
                if !numel_compatible(in_shape, fwd_out_shape) {
                    return Err(format!(
                        "reshape 反向 shape 不兼容：输入 {} 元素数与输出 {} 不一致",
                        fmt_dims(in_shape),
                        fmt_dims(fwd_out_shape)
                    ));
                }
                Ok(vec![in_shape.clone()])
            } else {
                Ok(vec![])
            }
        }

        // ── scatter ───────────────────────────────────────────────────
        // 前向: base, src, index → result（result shape == base shape）
        // 反向: d_base = base_shape, d_src = src_shape, index 不传梯度
        // 兼容性: grad shape（== output == base shape）天然与 base 一致；
        //         d_src 通过 gather 语义从 grad 提取，shape == src shape
        "scatter" => {
            let mut grads = Vec::with_capacity(fwd_in_shapes.len());
            for (i, in_shape) in fwd_in_shapes.iter().enumerate() {
                if i == 2 {
                    // index 不可微
                    grads.push(vec![]);
                } else {
                    grads.push(in_shape.clone());
                }
            }
            Ok(grads)
        }

        // ── gather ────────────────────────────────────────────────────
        // 前向: base, index → result（result shape == index shape）
        // 反向: d_base = base_shape（通过 scatter-add），index 不传梯度
        "gather" => {
            let mut grads = Vec::with_capacity(fwd_in_shapes.len());
            for (i, in_shape) in fwd_in_shapes.iter().enumerate() {
                if i == 1 {
                    // index 不可微
                    grads.push(vec![]);
                } else {
                    grads.push(in_shape.clone());
                }
            }
            Ok(grads)
        }

        // ── masked_fill ───────────────────────────────────────────────
        // 前向: input, mask → result（result shape == input shape）
        // 反向: d_input = grad * (1 - mask)，shape == input shape
        // 兼容性: grad shape == input shape（output == input，天然满足）
        "masked_fill" => {
            if let Some(in_shape) = fwd_in_shapes.first() {
                // mask 不可微
                Ok(vec![in_shape.clone(), vec![]])
            } else {
                Ok(vec![])
            }
        }

        // ── select ────────────────────────────────────────────────────
        // 前向: cond, then, else → result（broadcast 三者）
        // 反向: d_then = unbroadcast(grad, then_shape), d_else = 同
        //       cond 不可微
        // 不兼容条件: unbroadcast(output, then/else) 不可行
        "select" => {
            if fwd_in_shapes.len() < 3 {
                return Ok(vec![]);
            }
            let then_shape = &fwd_in_shapes[1];
            let else_shape = &fwd_in_shapes[2];
            unbroadcast_feasible(fwd_out_shape, then_shape)?;
            unbroadcast_feasible(fwd_out_shape, else_shape)?;
            // cond 不传梯度
            Ok(vec![vec![], fwd_out_shape.to_vec(), fwd_out_shape.to_vec()])
        }

        // ── 跳过检查的算子（反向 shape 天然与输入一致）──────────────────
        // element-wise 一元 / softmax / dropout / batch_norm / layer_norm / gelu
        // transpose / conv2d（保守通过，运行时兜底）
        "neg" | "relu" | "exp" | "log" | "sigmoid" | "abs"
        | "softmax" | "dropout" | "batch_norm" | "layer_norm" | "gelu"
        | "transpose" | "conv2d" => Ok(fwd_in_shapes.to_vec()),

        // 未知算子：保守返回 Ok（不检查，运行时兜底）
        _ => Ok(fwd_in_shapes.to_vec()),
    }
}

/// 验证反向 shape 与前向输入 shape 兼容（unbroadcast 可行性）。
///
/// 保守策略：
/// - 对于依赖 output shape 的算子，output 全 Any 时跳过（返回 Ok）
/// - 对于 cross_entropy 等不依赖 output 的算子，直接检查
/// - `backward_shape` 内部 `unbroadcast_feasible` 也会对 target 全 Any 跳过
///
/// 报错时返回 `TenthError::TypeError`（把运行时 RuntimeError 提升到编译期）。
pub(super) fn check_backward_shape_compat(
    op: &str,
    fwd_in_shapes: &[Vec<Dim>],
    fwd_out_shape: &[Dim],
    span: &Span,
) -> TenthResult<()> {
    // 对于依赖 output shape 的算子，output 全 Any 时跳过
    // cross_entropy/sum/mean/gather/scatter/reshape 的检查不依赖 output 静态信息
    let depends_on_output = !matches!(
        op,
        "cross_entropy" | "sum" | "mean" | "gather" | "scatter" | "reshape" | "view"
    );
    if depends_on_output && !fwd_out_shape.is_empty() && !has_static_info(fwd_out_shape) {
        return Ok(());
    }

    match backward_shape(op, fwd_in_shapes, fwd_out_shape) {
        Ok(_grads) => Ok(()),
        Err(msg) => Err(TenthError::TypeError {
            line: span.line,
            col: span.col,
            message: format!("编译期反向 shape 验证失败（{}）：{}", op, msg),
        }),
    }
}

// ── 辅助函数 ─────────────────────────────────────────────────────────

/// 模拟 unbroadcast 可行性检查（编译期 Dim 版本）。
///
/// 参考 `runtime/autodiff/backward.rs` 的 `unbroadcast` 逻辑，
/// 在编译期用 `Dim` 模拟。规则（从右往左对齐）：
/// - target 维为 `Known(1)` 且 grad 维 > 1 → 需要求和（可行）
/// - target 维与 grad 维相等 → 可行
/// - target 维为 `Any` 或 `Symbol` → 可行（保守）
/// - grad 维为 `Any` 或 `Symbol` → 可行（保守）
/// - target 维 != 1 且 != grad 维 → 不可行（报错）
///
/// target 全 Any 时直接返回 Ok（无法检查）。
fn unbroadcast_feasible(grad_shape: &[Dim], target_shape: &[Dim]) -> Result<(), String> {
    // target 全 Any 时跳过
    if !has_static_info(target_shape) {
        return Ok(());
    }
    let g_ndim = grad_shape.len();
    let t_ndim = target_shape.len();
    // 对齐：target 左侧补 Known(1)（与运行时 padded_target 逻辑一致）
    let pad = g_ndim.saturating_sub(t_ndim);
    for i in 0..g_ndim {
        let g_dim = &grad_shape[i];
        let t_dim: Dim = if i < pad {
            Dim::Known(1)
        } else {
            target_shape[i - pad].clone()
        };
        let compatible = match (&t_dim, g_dim) {
            (Dim::Known(1), _) => true, // target=1，可以 sum
            (Dim::Any, _) | (_, Dim::Any) => true, // 任一 Any，保守
            (Dim::Symbol(_), _) | (_, Dim::Symbol(_)) => true, // 任一 Symbol，保守
            (Dim::Known(t), Dim::Known(g)) => t == g, // 都 Known，必须相等
        };
        if !compatible {
            return Err(format!(
                "unbroadcast 不可行：梯度 {} 无法还原到目标 {}（第 {} 维 {} ≠ {}）",
                fmt_dims(grad_shape),
                fmt_dims(target_shape),
                i,
                fmt_dim(&t_dim),
                fmt_dim(g_dim)
            ));
        }
    }
    Ok(())
}

/// 判断两个维度是否兼容（相等/任一 Any/任一 Symbol 视为兼容）。
fn dims_compatible(a: &Dim, b: &Dim) -> bool {
    match (a, b) {
        (Dim::Any, _) | (_, Dim::Any) => true,
        (Dim::Symbol(_), _) | (_, Dim::Symbol(_)) => true,
        (Dim::Known(x), Dim::Known(y)) => x == y,
    }
}

/// 检查两个 shape 的元素数是否兼容（用于 reshape）。
///
/// 仅在两侧所有维度都 `Known` 时检查；含 `Any`/`Symbol` 时跳过（保守）。
fn numel_compatible(a: &[Dim], b: &[Dim]) -> bool {
    match (static_numel(a), static_numel(b)) {
        (Some(x), Some(y)) => x == y,
        // 含动态维度，无法静态判断，保守视为兼容
        _ => true,
    }
}

/// 计算 shape 的静态元素数（所有维度都 `Known` 时返回 `Some`，否则 `None`）。
fn static_numel(dims: &[Dim]) -> Option<u64> {
    let mut prod: u64 = 1;
    for d in dims {
        match d {
            Dim::Known(n) => {
                if *n < 0 {
                    return None;
                }
                prod = prod.checked_mul(*n as u64)?;
            }
            _ => return None,
        }
    }
    Some(prod)
}
