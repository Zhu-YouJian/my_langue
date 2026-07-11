//! 护城河 F Phase 1：结构化错误信息验证测试。
//!
//! 验证 backward 抛出的 ShapeMismatch 错误携带真实的 TapeErrorContext，
//! 包括 v_err（报错节点 id，非 loss_id）、op（算子名，非 "backward"）、
//! expected/actual shape（非空）。
//! 同时验证 formal_explain 接收真实 v_err 后的根因分析和边级归因。
//!
//! 测试项：
//! 1. ShapeMismatch 携带 TapeErrorContext（v_err 是真实报错节点，非 loss_id）
//! 2. formal_explain 接收真实 v_err，RootCause 列表非空且关联报错节点
//! 3. 边级归因：链式 Tape 中 RootCause.edge 非None
//! 4. op 字段为算子名（"MatMul"），非 "backward"

use tenth::runtime::autodiff::{Tape, TapeOp};
use tenth::runtime::tensor::Tensor;
use tenth::error::TenthError;

use std::rc::Rc;
use std::cell::RefCell;

// ── 辅助函数 ────────────────────────────────────────────────────────────

fn make_tensor(data: Vec<f64>, shape: Vec<usize>) -> Rc<RefCell<Tensor>> {
    Rc::new(RefCell::new(Tensor::from_vec(data, shape)))
}

// ── 1. 验证 ShapeMismatch 携带 TapeErrorContext（v_err 是真实报错节点） ─────

#[test]
fn test_shape_mismatch_carries_real_v_err() {
    // 构造链：Input A[2,3] → Reshape(result [4]) → Sum → loss
    // Reshape 的 result 只有 4 元素，但 A 有 6 元素 → backward 在 Reshape 失败。
    // 错误的 v_err 应是 Reshape 节点（非 loss/Sum 节点）。
    let a = make_tensor(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]); // 6 元素
    let reshape_result = make_tensor(vec![0.0; 4], vec![4]); // 仅 4 元素，与 A 的 6 不一致

    let mut tape = Tape::new();
    let a_id = tape.input(a.clone());
    let reshape_id = tape.unary(TapeOp::Reshape, a_id, a.clone(), reshape_result.clone());

    // Sum 作为 loss
    let sum_result = make_tensor(vec![0.0], vec![1]);
    let sum_id = tape.unary(TapeOp::Sum, reshape_id, reshape_result.clone(), sum_result);

    // 触发 backward — 应在 Reshape 节点失败
    let err = tape.backward(sum_id).unwrap_err();

    // 断言错误类型是 ShapeMismatch
    let context = match &err {
        TenthError::ShapeMismatch { context, .. } => context.clone(),
        _ => panic!("期望 ShapeMismatch 错误，实际: {:?}", err),
    };

    // 断言 v_err 是 Reshape 节点（非 loss/Sum 节点）
    assert_eq!(
        context.tape_node_id, reshape_id,
        "v_err 应是 Reshape 节点 #{}，而非 loss 节点 #{}",
        reshape_id, sum_id
    );
    assert_ne!(
        context.tape_node_id, sum_id,
        "v_err 不应是 loss 节点"
    );

    // 断言 expected_shape 和 actual_shape 非空
    assert!(
        !context.expected_shape.is_empty(),
        "expected_shape 不应为空，实际: {:?}",
        context.expected_shape
    );
    assert!(
        !context.actual_shape.is_empty(),
        "actual_shape 不应为空，实际: {:?}",
        context.actual_shape
    );

    // 断言具体值正确
    assert_eq!(context.expected_shape, vec![2, 3], "expected_shape 应为 A 的 shape");
    assert_eq!(context.actual_shape, vec![4], "actual_shape 应为 grad 的 shape");
}

// ── 2. 验证 formal_explain 接收真实 v_err ───────────────────────────────

#[test]
fn test_formal_explain_receives_real_v_err() {
    // 构造 MatMul 3D 输入的 backward 错误：
    // Input A[2,3,4] → MatMul(A, B[4,5]) → Sum → loss
    // MatMul backward 因 a_ndim=3 > 2 报错，v_err = MatMul 节点。
    let a = make_tensor(vec![1.0; 24], vec![2, 3, 4]); // 3D
    let b = make_tensor(vec![1.0; 20], vec![4, 5]); // 2D
    let mm_result = make_tensor(vec![0.0; 30], vec![2, 3, 5]);

    let mut tape = Tape::new();
    let a_id = tape.input(a.clone());
    let b_id = tape.input(b.clone());
    let mm_id = tape.binary(TapeOp::MatMul, a_id, b_id, a.clone(), b.clone(), mm_result.clone());

    // Sum 作为 loss
    let sum_result = make_tensor(vec![0.0], vec![1]);
    let sum_id = tape.unary(TapeOp::Sum, mm_id, mm_result.clone(), sum_result);

    // 触发 backward — 应在 MatMul 节点失败
    let err = tape.backward(sum_id).unwrap_err();

    // 从错误中提取真实 v_err/expected/actual/error_msg
    let (v_err, expected, actual, error_msg) = match &err {
        TenthError::ShapeMismatch { context, message } => (
            context.tape_node_id,
            context.expected_shape.as_slice(),
            context.actual_shape.as_slice(),
            message.as_str(),
        ),
        _ => panic!("期望 ShapeMismatch 错误，实际: {:?}", err),
    };

    // v_err 应是 MatMul 节点（非 loss 节点）
    assert_eq!(v_err, mm_id, "v_err 应是 MatMul 节点，而非 loss 节点");
    assert_ne!(v_err, sum_id, "v_err 不应是 loss 节点");

    // 用真实 v_err 调用 formal_explain
    let causes = tape.formal_explain(v_err, expected, actual, error_msg);

    // RootCause 列表应非空
    assert!(!causes.is_empty(), "formal_explain 应返回非空根因列表");

    // RootCause 的 tape_node_id 应与报错节点相关（上游可达节点）
    let cause_ids: Vec<usize> = causes.iter().map(|c| c.tape_node_id).collect();
    assert!(cause_ids.contains(&a_id), "根因应包含 A（MatMul 的上游输入）");
    assert!(cause_ids.contains(&b_id), "根因应包含 B（MatMul 的上游输入）");
    assert!(
        !cause_ids.contains(&mm_id),
        "v_err 自身不应在根因候选中"
    );
}

// ── 3. 验证边级归因（RootCause.edge 非None）──────────────────────────────

#[test]
fn test_edge_level_attribution() {
    // 构造链式 Tape：Input A → MatMul(A, W) → Add(mm, B) → Reshape → Sum → loss
    // Reshape 的 result shape 与 input 不一致 → backward 在 Reshape 失败。
    // formal_explain 应返回带边级归因的 RootCause（edge 字段非 None）。
    let a = make_tensor(vec![1.0; 4], vec![2, 2]);
    let w = make_tensor(vec![1.0; 4], vec![2, 2]);
    let b = make_tensor(vec![1.0; 4], vec![2, 2]);

    let mut tape = Tape::new();
    let a_id = tape.input(a.clone());
    let w_id = tape.input(w.clone());

    // MatMul(A, W) → [2, 2]
    let mm_result = make_tensor(vec![0.0; 4], vec![2, 2]);
    let mm_id = tape.binary(TapeOp::MatMul, a_id, w_id, a.clone(), w.clone(), mm_result.clone());

    // Add(mm, B) → [2, 2]
    let b_id = tape.input(b.clone());
    let add_result = make_tensor(vec![0.0; 4], vec![2, 2]);
    let add_id = tape.binary(TapeOp::Add, mm_id, b_id, mm_result.clone(), b.clone(), add_result.clone());

    // Reshape(add_result [2,2] = 4 元素) → result [3]（3 ≠ 4，不一致！）
    let reshape_result = make_tensor(vec![0.0; 3], vec![3]);
    let reshape_id = tape.unary(TapeOp::Reshape, add_id, add_result.clone(), reshape_result.clone());

    // Sum(reshape_result) → 标量
    let sum_result = make_tensor(vec![0.0], vec![1]);
    let sum_id = tape.unary(TapeOp::Sum, reshape_id, reshape_result.clone(), sum_result);

    // 触发 backward — 应在 Reshape 节点失败
    let err = tape.backward(sum_id).unwrap_err();

    let (v_err, expected, actual, error_msg) = match &err {
        TenthError::ShapeMismatch { context, message } => (
            context.tape_node_id,
            context.expected_shape.as_slice(),
            context.actual_shape.as_slice(),
            message.as_str(),
        ),
        _ => panic!("期望 ShapeMismatch 错误，实际: {:?}", err),
    };

    assert_eq!(v_err, reshape_id, "v_err 应是 Reshape 节点");

    // 调用 formal_explain
    let causes = tape.formal_explain(v_err, expected, actual, error_msg);
    assert!(!causes.is_empty(), "formal_explain 应返回非空根因列表");

    // 断言至少部分 RootCause 的 edge 字段非 None
    let with_edges: Vec<_> = causes.iter().filter(|c| c.edge.is_some()).collect();
    assert!(
        !with_edges.is_empty(),
        "至少部分 RootCause 应有 edge 归因（非 None）"
    );

    // 验证边结构：edge = (src, dst)，dst 应等于 tape_node_id
    for cause in &causes {
        if let Some((src, dst)) = cause.edge {
            assert_eq!(
                dst, cause.tape_node_id,
                "edge 的 dst 应等于 tape_node_id"
            );
            assert_ne!(
                src, cause.tape_node_id,
                "edge 的 src 应不同于 dst（src 更靠近 v_err）"
            );
        }
    }

    // 验证链式结构中的关键节点都在根因候选中
    let cause_ids: Vec<usize> = causes.iter().map(|c| c.tape_node_id).collect();
    assert!(cause_ids.contains(&add_id), "根因应包含 Add 节点");
    assert!(cause_ids.contains(&mm_id), "根因应包含 MatMul 节点");
    assert!(!cause_ids.contains(&reshape_id), "v_err 自身不应在根因中");
}

// ── 4. 验证 op 字段为算子名（"MatMul"），非 "backward" ──────────────────

#[test]
fn test_op_field_is_operator_name() {
    // 触发 MatMul backward 错误，断言 context.op == "MatMul"（非 "backward"）。
    // 这验证了 Phase 1 的改进：backward 各分支抛错时填充真实算子名，
    // 而非 Phase 0 的兜底值 "backward"。
    let a = make_tensor(vec![1.0; 24], vec![2, 3, 4]); // 3D，触发 a_ndim > 2 错误
    let b = make_tensor(vec![1.0; 20], vec![4, 5]);
    let mm_result = make_tensor(vec![0.0; 30], vec![2, 3, 5]);

    let mut tape = Tape::new();
    let a_id = tape.input(a.clone());
    let b_id = tape.input(b.clone());
    let mm_id = tape.binary(TapeOp::MatMul, a_id, b_id, a.clone(), b.clone(), mm_result.clone());

    // Sum 作为 loss
    let sum_result = make_tensor(vec![0.0], vec![1]);
    let sum_id = tape.unary(TapeOp::Sum, mm_id, mm_result.clone(), sum_result);

    // 触发 backward
    let err = tape.backward(sum_id).unwrap_err();

    let context = match &err {
        TenthError::ShapeMismatch { context, .. } => context.clone(),
        _ => panic!("期望 ShapeMismatch 错误，实际: {:?}", err),
    };

    // 核心断言：op 应是 "MatMul"（算子名），而非 "backward"（兜底值）
    assert_eq!(
        context.op, "MatMul",
        "op 字段应是算子名 'MatMul'，而非兜底值 'backward'"
    );
    assert_ne!(
        context.op, "backward",
        "op 字段不应是 'backward'（Phase 0 兜底值）"
    );

    // 额外验证：tape_node_id 应是 MatMul 节点
    assert_eq!(
        context.tape_node_id, mm_id,
        "tape_node_id 应是 MatMul 节点"
    );
}
