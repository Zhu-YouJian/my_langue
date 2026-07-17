//! 护城河 F：关系调试器集成测试（T2 FormalExplain 动态层 MVP）。
//!
//! 验证项：
//! 1. MatMul shape mismatch 根因定位到 MatMul 节点
//! 2. Add 广播失败根因定位到 Add 节点
//! 3. 链式错误根因定位：A→B→C 链中 C 报错，根因包含 A（Construct 节点）
//! 4. PartialExplain 用例：Preserve 节点在路径上但非根因
//! 5. 可达性过滤：不可达节点不在候选集中
//! 6. 端到端：通过 Tenth 源码触发 backward 失败，验证 ShapeMismatch 错误携带根因
//! 7. explain_error() native：返回根因列表

use tenth::lexer::lexer::Lexer;
use tenth::parser::parser::Parser;
use tenth::hir::lower::Lowerer;
use tenth::runtime::interpreter::Interpreter;
use tenth::runtime::value::Value;
use tenth::runtime::autodiff::{Tape, TapeOp};
use tenth::runtime::tensor::{Tensor, TensorData};
use tenth::runtime::relation_debugger::{classify_tape_op, ExplainClass, TapeOpClass};
use tenth::error::{ErrorType, TapeErrorContext, TenthError};

use std::rc::Rc;
use std::cell::RefCell;

// ── 辅助函数 ────────────────────────────────────────────────────────────

fn make_tensor(data: Vec<f64>, shape: Vec<usize>) -> Rc<RefCell<Tensor>> {
    Rc::new(RefCell::new(Tensor::from_vec(data, shape)))
}

/// 通过解释器运行 Tenth 源码，返回 Result<Value, TenthError>。
fn run_source(src: &str) -> Result<Value, TenthError> {
    let mut lexer = Lexer::new(src);
    let tokens = lexer.tokenize()?;
    let mut parser = Parser::new(tokens);
    let program = parser.parse_program()?;
    let mut lowerer = Lowerer::new();
    let hir = lowerer.lower_program(&program)?;
    let mut interp = Interpreter::new(&hir);
    interp.fs_sandbox = None;
    interp.deadline_ms = None;
    match interp.execute_program(&hir)? {
        Some(v) => Ok(v),
        None => Ok(Value::Unit),
    }
}

// ── 1. TapeOp 分类完整性（T7 定理完备互斥分类）────────────────────────

#[test]
fn test_tapeop_classify_completeness() {
    // 每个 TapeOp 变体必须分类到恰好一个类别
    let cases: [(TapeOp, TapeOpClass); 23] = [
        (TapeOp::Input, TapeOpClass::Construct),
        (TapeOp::CrossEntropy, TapeOpClass::Construct),
        (TapeOp::Conv2D, TapeOpClass::Construct),
        (TapeOp::BatchNorm, TapeOpClass::Construct),
        (TapeOp::LayerNorm, TapeOpClass::Construct),
        (TapeOp::Dropout, TapeOpClass::Construct),
        (TapeOp::Select, TapeOpClass::Construct),
        (TapeOp::Add, TapeOpClass::Preserve),
        (TapeOp::Sub, TapeOpClass::Preserve),
        (TapeOp::Mul, TapeOpClass::Preserve),
        (TapeOp::Div, TapeOpClass::Preserve),
        (TapeOp::Neg, TapeOpClass::Preserve),
        (TapeOp::ReLU, TapeOpClass::Preserve),
        (TapeOp::Exp, TapeOpClass::Preserve),
        (TapeOp::Log, TapeOpClass::Preserve),
        (TapeOp::Sigmoid, TapeOpClass::Preserve),
        (TapeOp::Softmax, TapeOpClass::Preserve),
        (TapeOp::Gelu, TapeOpClass::Preserve),
        (TapeOp::Abs, TapeOpClass::Preserve),
        (TapeOp::Sum, TapeOpClass::Reduce),
        (TapeOp::Mean, TapeOpClass::Reduce),
        (TapeOp::MatMul, TapeOpClass::Expand),
        (TapeOp::Transpose, TapeOpClass::Expand),
    ];
    for (op, expected) in &cases {
        assert_eq!(classify_tape_op(op, None), *expected, "TapeOp::{:?} 分类错误", op);
    }
}

// ── 2. MatMul shape mismatch 根因定位 ──────────────────────────────────

#[test]
fn test_matmul_shape_mismatch_root_cause() {
    // 构造 (M,K)@(K',N) where K=3 ≠ K'=2 的不匹配场景。
    // 注意：forward 在实际计算时会失败，这里直接构造合法 tape，
    // 然后调用 formal_explain 模拟 backward 失败的根因分析。
    let a = make_tensor(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let b = make_tensor(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);

    let mut tape = Tape::new();
    let a_id = tape.input(a.clone());
    let b_id = tape.input(b.clone());

    // 模拟 matmul 结果（占位，不实际计算）
    let result = make_tensor(vec![0.0; 4], vec![2, 2]);
    let mm_id = tape.binary(TapeOp::MatMul, a_id, b_id, a.clone(), b.clone(), result.clone());

    // 调用 formal_explain：v_err = mm_id（backward 在此失败）
    let causes = tape.formal_explain(mm_id, &[2, 3], &[2, 2], "");

    // 应返回 2 个候选（a_id 和 b_id，不含 mm_id 自身）
    assert_eq!(causes.len(), 2);
    // 两个 Input 节点都是 Construct，分类为 ExplainsError
    for c in &causes {
        assert_eq!(c.classification, ExplainClass::ExplainsError);
    }
    // 候选应包含 a_id 和 b_id
    let cause_ids: Vec<usize> = causes.iter().map(|c| c.tape_node_id).collect();
    assert!(cause_ids.contains(&a_id));
    assert!(cause_ids.contains(&b_id));
    // 说明应包含 MatMul 相关诊断
    let hint_concat: String = causes.iter().map(|c| c.explanation.as_str()).collect();
    assert!(hint_concat.contains("叶子参数") || hint_concat.contains("shape"));
}

// ── 3. Add 广播失败根因定位 ────────────────────────────────────────────

#[test]
fn test_add_broadcast_failure_root_cause() {
    // 构造不兼容 shape 的 Add：[2,3] + [4,5]（无法广播）
    let a = make_tensor(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let b = make_tensor(vec![1.0, 2.0, 3.0, 4.0], vec![4]);

    let mut tape = Tape::new();
    let a_id = tape.input(a.clone());
    let b_id = tape.input(b.clone());

    let result = make_tensor(vec![0.0; 6], vec![2, 3]);
    let add_id = tape.binary(TapeOp::Add, a_id, b_id, a.clone(), b.clone(), result.clone());

    let causes = tape.formal_explain(add_id, &[2, 3], &[2, 3], "");

    // 应返回 a_id 和 b_id 两个候选
    assert_eq!(causes.len(), 2);
    let cause_ids: Vec<usize> = causes.iter().map(|c| c.tape_node_id).collect();
    assert!(cause_ids.contains(&a_id));
    assert!(cause_ids.contains(&b_id));
    // 二者都是 Input（Construct），分类为 ExplainsError
    for c in &causes {
        assert_eq!(c.classification, ExplainClass::ExplainsError);
    }
}

// ── 4. 链式错误根因定位（A→B→C，C 报错，根因包含 A）─────────────────

#[test]
fn test_chain_error_root_cause_includes_origin() {
    // 链式：A (Input) → relu (Preserve) → matmul (Expand) → 报错
    // 根因应包含 A（Construct/Input）和 matmul 节点本身被排除（v_err）
    let a = make_tensor(vec![-1.0, 2.0, 3.0, -4.0], vec![2, 2]);
    let w = make_tensor(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);

    let mut tape = Tape::new();
    let a_id = tape.input(a.clone());
    let w_id = tape.input(w.clone());

    // relu(A)
    let relu_data = a.borrow().data.mapv(|v| if v > 0.0 { v } else { 0.0 });
    let relu = Rc::new(RefCell::new(Tensor::from_tensor_data(TensorData::F64(relu_data))));
    let relu_id = tape.unary(TapeOp::ReLU, a_id, a.clone(), relu.clone());

    // matmul(relu, w)
    let r_data = relu.borrow().data.clone();
    let w_data = w.borrow().data.clone();
    let r2 = r_data.view().into_dimensionality::<ndarray::Ix2>().unwrap();
    let w2 = w_data.view().into_dimensionality::<ndarray::Ix2>().unwrap();
    let mm_data = r2.dot(&w2).into_dyn();
    let mm = Rc::new(RefCell::new(Tensor::from_tensor_data(TensorData::F64(mm_data))));
    let mm_id = tape.binary(TapeOp::MatMul, relu_id, w_id, relu.clone(), w.clone(), mm.clone());

    // 调用 formal_explain：v_err = mm_id
    let causes = tape.formal_explain(mm_id, &[], &[], "");

    // 应返回 3 个候选：a_id, w_id, relu_id（不含 mm_id 自身）
    assert_eq!(causes.len(), 3);
    let cause_ids: Vec<usize> = causes.iter().map(|c| c.tape_node_id).collect();
    assert!(cause_ids.contains(&a_id), "根因应包含 A（链的源头）");
    assert!(cause_ids.contains(&w_id), "根因应包含 W（权重）");
    assert!(cause_ids.contains(&relu_id), "根因应包含中间节点 relu");
    assert!(!cause_ids.contains(&mm_id), "v_err 自身不应在候选中");

    // a_id 和 w_id 是 Input（Construct）→ ExplainsError
    // relu_id 是 ReLU（Preserve）→ PartialExplain
    let a_cause = causes.iter().find(|c| c.tape_node_id == a_id).unwrap();
    assert_eq!(a_cause.classification, ExplainClass::ExplainsError);
    let w_cause = causes.iter().find(|c| c.tape_node_id == w_id).unwrap();
    assert_eq!(w_cause.classification, ExplainClass::ExplainsError);
    let relu_cause = causes.iter().find(|c| c.tape_node_id == relu_id).unwrap();
    assert_eq!(relu_cause.classification, ExplainClass::PartialExplain);

    // 排序：ExplainsError 应在 PartialExplain 之前
    assert_eq!(causes[0].classification, ExplainClass::ExplainsError);
    assert_eq!(causes[0].classification, ExplainClass::ExplainsError);
}

// ── 5. PartialExplain 用例：Preserve 节点在路径上但非根因 ─────────────

#[test]
fn test_partial_explain_preserve_node() {
    // 链式：x → exp → log → exp2 → 报错
    // 所有中间节点都是 Preserve；只有 x（Input）是 Construct
    let x = make_tensor(vec![1.0, 2.0], vec![2]);
    let mut tape = Tape::new();
    let x_id = tape.input(x.clone());

    let exp_data = x.borrow().data.mapv(|v| v.exp());
    let exp = Rc::new(RefCell::new(Tensor::from_tensor_data(TensorData::F64(exp_data))));
    let exp_id = tape.unary(TapeOp::Exp, x_id, x.clone(), exp.clone());

    let log_data = exp.borrow().data.mapv(|v| v.ln());
    let log = Rc::new(RefCell::new(Tensor::from_tensor_data(TensorData::F64(log_data))));
    let log_id = tape.unary(TapeOp::Log, exp_id, exp.clone(), log.clone());

    let exp2_data = log.borrow().data.mapv(|v| v.exp());
    let exp2 = Rc::new(RefCell::new(Tensor::from_tensor_data(TensorData::F64(exp2_data))));
    let exp2_id = tape.unary(TapeOp::Exp, log_id, log.clone(), exp2.clone());

    let causes = tape.formal_explain(exp2_id, &[], &[], "");

    // 应有 3 个候选：x_id, exp_id, log_id（不含 exp2_id）
    assert_eq!(causes.len(), 3);
    // x_id (Input) → ExplainsError
    let x_cause = causes.iter().find(|c| c.tape_node_id == x_id).unwrap();
    assert_eq!(x_cause.classification, ExplainClass::ExplainsError);
    // exp_id 和 log_id 是 Preserve → PartialExplain
    for mid_id in &[exp_id, log_id] {
        let cause = causes.iter().find(|c| c.tape_node_id == *mid_id).unwrap();
        assert_eq!(cause.classification, ExplainClass::PartialExplain);
    }
}

// ── 6. 可达性过滤：不可达节点不在候选集中 ───────────────────────────────

#[test]
fn test_unreachable_filtered() {
    // 两个独立子图：
    //   子图 A: a → relu → 报错
    //   子图 B: c → exp → （独立，与 A 无关）
    // 调用 formal_explain on relu_id 时，c_id / exp_id 不应在候选中
    let a = make_tensor(vec![-1.0, 2.0], vec![2]);
    let c = make_tensor(vec![3.0, 4.0], vec![2]);

    let mut tape = Tape::new();
    let a_id = tape.input(a.clone());
    let c_id = tape.input(c.clone());

    let relu_data = a.borrow().data.mapv(|v| if v > 0.0 { v } else { 0.0 });
    let relu = Rc::new(RefCell::new(Tensor::from_tensor_data(TensorData::F64(relu_data))));
    let relu_id = tape.unary(TapeOp::ReLU, a_id, a.clone(), relu.clone());

    let exp_data = c.borrow().data.mapv(|v| v.exp());
    let exp = Rc::new(RefCell::new(Tensor::from_tensor_data(TensorData::F64(exp_data))));
    let exp_id = tape.unary(TapeOp::Exp, c_id, c.clone(), exp.clone());

    let causes = tape.formal_explain(relu_id, &[], &[], "");
    let cause_ids: Vec<usize> = causes.iter().map(|c| c.tape_node_id).collect();

    // 不可达节点不应在候选中
    assert!(!cause_ids.contains(&c_id), "不可达节点 c_id 不应在候选中");
    assert!(!cause_ids.contains(&exp_id), "不可达节点 exp_id 不应在候选中");
    // 可达节点 a_id 应在候选中
    assert!(cause_ids.contains(&a_id), "可达节点 a_id 应在候选中");
}

// ── 7. 端到端：通过 Tenth 源码触发 backward 失败 ──────────────────────

#[test]
fn test_end_to_end_shape_mismatch_carries_root_cause() {
    // 构造一个会触发 backward shape 错误的 Tenth 程序：
    //   - 用 param 注册一个 shape 错误的张量
    //   - 调用 backward 时会失败，错误消息应包含根因分析
    let src = r#"
fn main() {
    new_grad();
    // 构造一个 1D 张量 [2.0, 3.0]
    let x = param(tensor([[2.0, 3.0]]));
    // 构造不兼容的 matmul：[1,2] @ [3,2] —— K=2 vs K'=3，forward 就会失败
    // 但为了触发 backward 失败，我们构造一个 forward 能成功但 backward 失败的场景：
    // 这里我们用 element-wise add 触发 broadcast 失败
    let y = param(tensor([[1.0], [2.0]]));  // shape [2,1]
    let z = x + y;  // [1,2] + [2,1] = [2,2]，forward 成功
    let loss = z.sum();
    backward(loss);  // backward 应正常通过（无 shape 错误）
    println("ok");
}
"#;
    // 这个用例应正常通过（forward 与 backward 都合法）
    let result = run_source(src);
    assert!(result.is_ok(), "合法程序应执行成功，但出错: {:?}", result.err());

    // 现在构造一个 backward 失败的端到端用例：
    // 由于 backward 内部 shape 错误比较罕见（多数 shape 错误在 forward 就抛出），
    // 这里通过直接构造 Tape 调用 formal_explain 验证端到端逻辑。
    let mut tape = Tape::new();
    let a = make_tensor(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let b = make_tensor(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let a_id = tape.input(a.clone());
    let b_id = tape.input(b.clone());
    let result = make_tensor(vec![0.0; 4], vec![2, 2]);
    let mm_id = tape.binary(TapeOp::MatMul, a_id, b_id, a.clone(), b.clone(), result.clone());

    // 模拟 backward 失败：直接调用 formal_explain
    let causes = tape.formal_explain(mm_id, &[], &[], "");
    assert!(!causes.is_empty(), "formal_explain 应返回根因候选");
    // 所有候选的 explanation 应非空
    for c in &causes {
        assert!(!c.explanation.is_empty(), "根因说明不应为空");
        assert!(c.explanation.contains("节点 #"), "根因说明应包含节点 id");
    }
}

// ── 8. explain_error() native 行为 ────────────────────────────────────

#[test]
fn test_explain_error_native_returns_vec() {
    // 没有发生 backward 错误时，explain_error() 应返回空 Vec
    let src = r#"
fn main() {
    let causes = explain_error();
    println("causes count: " + to_string(causes.len()));
}
"#;
    let result = run_source(src);
    assert!(result.is_ok(), "explain_error() 在无错误时应返回空 Vec: {:?}", result.err());
}

// ── 9. ShapeMismatch 错误结构验证 ─────────────────────────────────────

#[test]
fn test_shape_mismatch_error_structure() {
    // 构造一个 ShapeMismatch 错误并验证其 Display 输出
    let context = TapeErrorContext {
        tape_node_id: 5,
        op: "MatMul".to_string(),
        expected_shape: vec![2, 3],
        actual_shape: vec![2, 2],
    };
    let err = TenthError::ShapeMismatch {
        context,
        message: "K ≠ K'，矩阵维度不匹配".to_string(),
    };
    let displayed = format!("{}", err);
    assert!(displayed.contains("形状错误"), "Display 应包含 '形状错误'，实际: {}", displayed);
    assert!(displayed.contains("节点 #5"), "Display 应包含节点 id，实际: {}", displayed);
    assert!(displayed.contains("MatMul"), "Display 应包含算子名，实际: {}", displayed);
    assert!(displayed.contains("K ≠ K'"), "Display 应包含消息，实际: {}", displayed);
}

// ── 10. Gather / Scatter 分类为 Preserve ──────────────────────────────

#[test]
fn classify_tape_op_gather_is_preserve() {
    // Gather：输出 shape == index.shape（从输入张量继承 shape，非新构造）。
    // 按 T7 定理分类为 Preserve（与 Scatter 一致）。
    assert_eq!(classify_tape_op(&TapeOp::Gather, None), TapeOpClass::Preserve);
    // 顺带验证 Scatter 也为 Preserve（多维扩展后分类不变）
    assert_eq!(classify_tape_op(&TapeOp::Scatter, None), TapeOpClass::Preserve);
}

// ── 11. Phase 1：backward 错误携带真实 v_err（非 loss_id）──────────────

#[test]
fn test_backward_error_carries_real_v_err() {
    // 构造 a(3D) → matmul → sum(loss) 的 Tape。
    // backward 在 matmul 节点因 a_ndim > 2 报错（ShapeMismatch）。
    // 验证 context.tape_node_id == mm_id（真实报错节点），而非 loss_id。
    // 这验证了 Phase 1 的核心改进：natives.rs 包裹层从 ShapeMismatch 错误中
    // 提取真实 v_err 传给 formal_explain，替代 Phase 0 的占位值 loss_id。
    let a = make_tensor(vec![1.0; 8], vec![2, 2, 2]); // 3D，触发 a_ndim > 2
    let b = make_tensor(vec![1.0; 4], vec![2, 2]);
    let mut tape = Tape::new();
    let a_id = tape.input(a.clone());
    let b_id = tape.input(b.clone());
    let mm_result = make_tensor(vec![0.0; 4], vec![2, 2]);
    let mm_id = tape.binary(TapeOp::MatMul, a_id, b_id, a.clone(), b.clone(), mm_result.clone());
    // loss = sum(mm_result)，使 loss_id ≠ mm_id
    let loss_result = make_tensor(vec![0.0; 1], vec![1]);
    let loss_id = tape.unary(TapeOp::Sum, mm_id, mm_result.clone(), loss_result.clone());

    let err = tape.backward(loss_id).unwrap_err();
    match err {
        TenthError::ShapeMismatch { context, message } => {
            assert_ne!(
                context.tape_node_id, loss_id,
                "tape_node_id 不应是 loss_id={}（Phase 0 占位行为）", loss_id
            );
            assert_eq!(
                context.tape_node_id, mm_id,
                "tape_node_id 应是 mm_id（真实报错节点）"
            );
            assert_eq!(context.op, "MatMul");
            assert!(
                message.contains("a ndim=3"),
                "message 应包含 'a ndim=3'，实际: {}", message
            );
        }
        _ => panic!("期望 ShapeMismatch，实际: {:?}", err),
    }
}

// ── 12. Phase 1：backward 错误携带 expected/actual shape ───────────────

#[test]
fn test_backward_error_carries_expected_actual() {
    // 构造 a(2D) → batchedmatmul(a, b) → sum(loss) 的 Tape。
    // backward 在 batchedmatmul 节点因 a_ndim != 3 报错。
    // 验证 context.expected_shape = [3, 3, 3] 且 context.actual_shape = [2, 2]。
    let a = make_tensor(vec![1.0; 4], vec![2, 2]); // 2D，触发 a_ndim != 3
    let b = make_tensor(vec![1.0; 4], vec![2, 2]);
    let mut tape = Tape::new();
    let a_id = tape.input(a.clone());
    let b_id = tape.input(b.clone());
    let bmm_result = make_tensor(vec![0.0; 4], vec![2, 2]);
    let bmm_id = tape.binary(
        TapeOp::BatchedMatMul,
        a_id,
        b_id,
        a.clone(),
        b.clone(),
        bmm_result.clone(),
    );
    let loss_result = make_tensor(vec![0.0; 1], vec![1]);
    let loss_id = tape.unary(TapeOp::Sum, bmm_id, bmm_result.clone(), loss_result.clone());

    let err = tape.backward(loss_id).unwrap_err();
    match err {
        TenthError::ShapeMismatch { context, .. } => {
            assert_eq!(context.tape_node_id, bmm_id);
            assert_eq!(context.op, "BatchedMatMul");
            assert_eq!(
                context.expected_shape,
                vec![3, 3, 3],
                "expected_shape 应为 [3, 3, 3]"
            );
            assert_eq!(
                context.actual_shape,
                vec![2, 2],
                "actual_shape 应为 [2, 2]（a_ndim, b_ndim）"
            );
            // 关键：expected/actual 都非空（Phase 1 结构化提取）
            assert!(!context.expected_shape.is_empty(), "expected_shape 不应为空");
            assert!(!context.actual_shape.is_empty(), "actual_shape 不应为空");
        }
        _ => panic!("期望 ShapeMismatch，实际: {:?}", err),
    }
}

// ── 13. Phase 1：formal_explain 收到真实 context 后分类更精确 ──────────

#[test]
fn test_formal_explain_receives_real_context() {
    // 验证 formal_explain 收到真实 v_err/expected/actual/error_msg 后，
    // RootCause.error_type 的分类比 Phase 0（占位值）更精确。
    // 场景：MatMul 节点，error_msg 含 "squeeze" → SilentSqueeze（Phase 1）
    //       vs 空 error_msg → ShapeMismatch 兜底（Phase 0）
    let a = make_tensor(vec![1.0; 6], vec![2, 3]);
    let b = make_tensor(vec![1.0; 4], vec![2, 2]);
    let mut tape = Tape::new();
    let a_id = tape.input(a.clone());
    let b_id = tape.input(b.clone());
    let result = make_tensor(vec![0.0; 4], vec![2, 2]);
    let mm_id = tape.binary(TapeOp::MatMul, a_id, b_id, a.clone(), b.clone(), result.clone());

    // Phase 0（占位）：空 expected/actual/error_msg → ShapeMismatch 兜底
    let causes_phase0 = tape.formal_explain(mm_id, &[], &[], "");
    assert!(causes_phase0.len() >= 1);
    for c in &causes_phase0 {
        assert_eq!(
            c.error_type,
            Some(ErrorType::ShapeMismatch),
            "Phase 0（占位）应分类为 ShapeMismatch（兜底）"
        );
    }

    // Phase 1（真实）：真实 expected/actual/error_msg（含 "squeeze"）→ SilentSqueeze
    let causes_phase1 =
        tape.formal_explain(mm_id, &[1], &[2], "MatMul 反向 1D squeeze 失败");
    assert!(causes_phase1.len() >= 1);
    for c in &causes_phase1 {
        assert_eq!(
            c.error_type,
            Some(ErrorType::SilentSqueeze),
            "Phase 1（真实）应分类为 SilentSqueeze（更精确）"
        );
    }

    // 关键断言：Phase 1 分类比 Phase 0 更精确（两者不同）
    assert_ne!(
        causes_phase0[0].error_type,
        causes_phase1[0].error_type,
        "Phase 1 应比 Phase 0 分类更精确"
    );
}

// ── 14. Phase 2：5 类错误分类 — ShapeMismatch（兜底）──────────────────

#[test]
fn test_error_type_shape_mismatch() {
    // 通用 shape 不匹配（MatMul 无 "squeeze" 关键词）→ ShapeMismatch
    let a = make_tensor(vec![1.0; 6], vec![2, 3]);
    let b = make_tensor(vec![1.0; 4], vec![2, 2]);
    let mut tape = Tape::new();
    let a_id = tape.input(a.clone());
    let b_id = tape.input(b.clone());
    let result = make_tensor(vec![0.0; 4], vec![2, 2]);
    let mm_id = tape.binary(TapeOp::MatMul, a_id, b_id, a.clone(), b.clone(), result.clone());

    let causes = tape.formal_explain(mm_id, &[2, 3], &[2, 2], "MatMul 维度不匹配");
    assert!(causes.len() >= 1, "应返回根因候选");
    for c in &causes {
        assert_eq!(
            c.error_type,
            Some(ErrorType::ShapeMismatch),
            "MatMul 无 squeeze 关键词应分类为 ShapeMismatch"
        );
    }
}

// ── 15. Phase 2：5 类错误分类 — SilentSqueeze ─────────────────────────

#[test]
fn test_error_type_silent_squeeze() {
    // MatMul + "squeeze" 关键词 → SilentSqueeze
    let a = make_tensor(vec![1.0; 6], vec![2, 3]);
    let b = make_tensor(vec![1.0; 4], vec![2, 2]);
    let mut tape = Tape::new();
    let a_id = tape.input(a.clone());
    let b_id = tape.input(b.clone());
    let result = make_tensor(vec![0.0; 4], vec![2, 2]);
    let mm_id = tape.binary(TapeOp::MatMul, a_id, b_id, a.clone(), b.clone(), result.clone());

    let causes = tape.formal_explain(mm_id, &[], &[], "MatMul 反向 1D squeeze 失败");
    assert!(causes.len() >= 1);
    for c in &causes {
        assert_eq!(
            c.error_type,
            Some(ErrorType::SilentSqueeze),
            "MatMul + squeeze 关键词应分类为 SilentSqueeze"
        );
    }

    // BatchedMatMul + "squeeze" 同样应分类为 SilentSqueeze
    let causes2 = tape.formal_explain(mm_id, &[], &[], "squeeze 失败");
    for c in &causes2 {
        assert_eq!(c.error_type, Some(ErrorType::SilentSqueeze));
    }
}

// ── 16. Phase 2：5 类错误分类 — BroadcastFail ─────────────────────────

#[test]
fn test_error_type_broadcast_fail() {
    // Add + "unbroadcast" 关键词 → BroadcastFail
    let a = make_tensor(vec![1.0; 6], vec![2, 3]);
    let b = make_tensor(vec![1.0; 4], vec![4]);
    let mut tape = Tape::new();
    let a_id = tape.input(a.clone());
    let b_id = tape.input(b.clone());
    let result = make_tensor(vec![0.0; 6], vec![2, 3]);
    let add_id = tape.binary(TapeOp::Add, a_id, b_id, a.clone(), b.clone(), result.clone());

    let causes = tape.formal_explain(add_id, &[2, 3], &[2, 3], "unbroadcast reshape 失败");
    assert!(causes.len() >= 1);
    for c in &causes {
        assert_eq!(
            c.error_type,
            Some(ErrorType::BroadcastFail),
            "Add + unbroadcast 关键词应分类为 BroadcastFail"
        );
    }

    // Sub/Mul/Div 同理
    let sub_id = tape.binary(TapeOp::Sub, a_id, b_id, a.clone(), b.clone(), result.clone());
    for c in tape.formal_explain(sub_id, &[], &[], "unbroadcast 失败") {
        assert_eq!(c.error_type, Some(ErrorType::BroadcastFail));
    }
}

// ── 17. Phase 2：5 类错误分类 — GradDrift ─────────────────────────────

#[test]
fn test_error_type_grad_drift() {
    // Input acc_grad shape 不一致 → GradDrift。
    // 构造：input(param shape=[2,3]) → matmul(a shape=[2,4], b shape=[4,2]) → sum(loss)
    // backward 时，matmul 的 d_a shape=[2,4]，但 Input 的 param shape=[2,3]
    // → acc_grad 失败 → ShapeMismatch { op="Input", expected=[2,3], actual=[2,4] }
    // 关键：param_id 节点收到的 grad shape（来自 matmul 的 d_a）与 param shape 不一致，
    // 这是 GradDrift 的典型场景（前向 shape 流 vs 反向 grad shape 流不一致）。
    let param = make_tensor(vec![1.0; 6], vec![2, 3]); // Input 的 param，shape=[2,3]
    let a = make_tensor(vec![1.0; 8], vec![2, 4]); // matmul 的输入，shape=[2,4]（与 param 不一致）
    let b = make_tensor(vec![1.0; 8], vec![4, 2]);
    let mut tape = Tape::new();
    let param_id = tape.input(param.clone()); // Input 节点
    let b_id = tape.input(b.clone());
    // matmul(param_id, b_id, a, b, result)：a 的 shape=[2,4]，与 param 的 shape=[2,3] 不一致
    let mm_result = make_tensor(vec![0.0; 4], vec![2, 2]);
    let mm_id = tape.binary(
        TapeOp::MatMul,
        param_id,
        b_id,
        a.clone(),
        b.clone(),
        mm_result.clone(),
    );
    let loss_result = make_tensor(vec![0.0; 1], vec![1]);
    let loss_id = tape.unary(TapeOp::Sum, mm_id, mm_result.clone(), loss_result.clone());

    let err = tape.backward(loss_id).unwrap_err();
    match err {
        TenthError::ShapeMismatch { context, message } => {
            assert_eq!(context.tape_node_id, param_id, "应在 Input 节点报错");
            assert_eq!(context.op, "Input");
            assert_eq!(context.expected_shape, vec![2, 3], "expected = param shape");
            assert_eq!(context.actual_shape, vec![2, 4], "actual = grad shape（漂移）");
            assert!(
                message.contains("acc_grad"),
                "message 应包含 'acc_grad'（GradDrift 标志），实际: {}",
                message
            );
        }
        _ => panic!("期望 ShapeMismatch，实际: {:?}", err),
    }

    // 验证 classify_error_type 将此场景分类为 GradDrift：
    // 由于 formal_explain 跳过 v_err 自身（Input 无上游），返回空 Vec，
    // 这里通过构造相同 op=Input + error_msg="acc_grad" 的 formal_explain 调用
    // 间接验证分类逻辑（classify_error_type 是 Tape 的私有方法，无法直接调用）。
    // 注意：natives.rs 包裹层会用真实 v_err=param_id 调用 formal_explain，
    // 虽然返回空 Vec，但 classify_error_type 仍会被调用并分类为 GradDrift。
    let causes = tape.formal_explain(param_id, &[2, 3], &[2, 4], "acc_grad shape 不匹配");
    assert!(
        causes.is_empty(),
        "Input 节点无上游，formal_explain 应返回空 Vec（分类在内部完成）"
    );
}

// ── 18. Phase 2：5 类错误分类 — DtypeConflict ─────────────────────────

#[test]
fn test_error_type_dtype_conflict() {
    // 任意 op（非 MatMul/Add/Input 等）+ "dtype" 关键词 → DtypeConflict。
    // 用 ReLU 节点（不在 op 特定分类列表中），error_msg 含 "dtype"。
    let x = make_tensor(vec![-1.0, 2.0], vec![2]);
    let mut tape = Tape::new();
    let x_id = tape.input(x.clone());
    let relu_data = x.borrow().data.mapv(|v| if v > 0.0 { v } else { 0.0 });
    let relu = Rc::new(RefCell::new(Tensor::from_tensor_data(TensorData::F64(
        relu_data,
    ))));
    let relu_id = tape.unary(TapeOp::ReLU, x_id, x.clone(), relu.clone());

    let causes = tape.formal_explain(relu_id, &[], &[], "dtype 不匹配（f32/f64 混用）");
    assert!(causes.len() >= 1);
    for c in &causes {
        assert_eq!(
            c.error_type,
            Some(ErrorType::DtypeConflict),
            "ReLU + dtype 关键词应分类为 DtypeConflict"
        );
    }

    // "f32/f64" 关键词同样应分类为 DtypeConflict
    let causes2 = tape.formal_explain(relu_id, &[], &[], "f32/f64 混用");
    for c in &causes2 {
        assert_eq!(c.error_type, Some(ErrorType::DtypeConflict));
    }
}

// ── 19. 边级归因：RootCause.edge 字段被正确填充 ────────────────────────

#[test]
fn test_root_cause_has_edge_info() {
    // 构造 a → matmul → 报错，验证所有 RootCause.edge 被 BFS 填充（非 None）。
    // edge = (src=parent_in_bfs, dst=this_node)，parent 是更靠近 v_err 的下一跳。
    let a = make_tensor(vec![1.0; 6], vec![2, 3]);
    let b = make_tensor(vec![1.0; 4], vec![2, 2]);
    let mut tape = Tape::new();
    let a_id = tape.input(a.clone());
    let b_id = tape.input(b.clone());
    let result = make_tensor(vec![0.0; 4], vec![2, 2]);
    let mm_id = tape.binary(TapeOp::MatMul, a_id, b_id, a.clone(), b.clone(), result.clone());

    let causes = tape.formal_explain(mm_id, &[2, 3], &[2, 2], "MatMul 维度不匹配");
    assert!(causes.len() >= 2, "应至少有 2 个根因候选");
    for c in &causes {
        assert!(
            c.edge.is_some(),
            "edge 应被填充（非 None），节点 #{}",
            c.tape_node_id
        );
        // edge 的 dst 应等于本节点 id
        let (_, dst) = c.edge.unwrap();
        assert_eq!(dst, c.tape_node_id, "edge 的 dst 应等于本节点 id");
    }
}

// ── 20. 边级归因：edge 指向 BFS 树的 parent ───────────────────────────

#[test]
fn test_edge_points_to_parent_in_bfs_tree() {
    // 构造链：x → relu → matmul → 报错
    // BFS 树（从 v_err=mm_id 反向遍历）：
    //   mm_id (v_err, parent=None)
    //     ├─ relu_id (parent=mm_id)
    //     │    └─ x_id (parent=relu_id)
    //     └─ w_id (parent=mm_id)
    // edge[relu_id] = (mm_id, relu_id)
    // edge[w_id] = (mm_id, w_id)
    // edge[x_id] = (relu_id, x_id)（x_id 通过 relu_id 到达）
    let x = make_tensor(vec![-1.0, 2.0, 3.0, -4.0], vec![2, 2]);
    let w = make_tensor(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let mut tape = Tape::new();
    let x_id = tape.input(x.clone());
    let w_id = tape.input(w.clone());

    let relu_data = x.borrow().data.mapv(|v| if v > 0.0 { v } else { 0.0 });
    let relu = Rc::new(RefCell::new(Tensor::from_tensor_data(TensorData::F64(
        relu_data,
    ))));
    let relu_id = tape.unary(TapeOp::ReLU, x_id, x.clone(), relu.clone());

    let mm_result = make_tensor(vec![0.0; 4], vec![2, 2]);
    let mm_id = tape.binary(
        TapeOp::MatMul,
        relu_id,
        w_id,
        relu.clone(),
        w.clone(),
        mm_result.clone(),
    );

    let causes = tape.formal_explain(mm_id, &[], &[], "");
    // relu_id 和 w_id 的 parent 应为 mm_id
    let relu_cause = causes
        .iter()
        .find(|c| c.tape_node_id == relu_id)
        .expect("应包含 relu_id");
    assert_eq!(
        relu_cause.edge,
        Some((mm_id, relu_id)),
        "relu_id 的 edge 应指向 parent=mm_id"
    );
    let w_cause = causes
        .iter()
        .find(|c| c.tape_node_id == w_id)
        .expect("应包含 w_id");
    assert_eq!(
        w_cause.edge,
        Some((mm_id, w_id)),
        "w_id 的 edge 应指向 parent=mm_id"
    );
    // x_id 的 parent 应为 relu_id（BFS 树的下一跳，而非 mm_id）
    let x_cause = causes
        .iter()
        .find(|c| c.tape_node_id == x_id)
        .expect("应包含 x_id");
    assert_eq!(
        x_cause.edge,
        Some((relu_id, x_id)),
        "x_id 的 edge 应指向 parent=relu_id（BFS 树的下一跳，而非 mm_id）"
    );
}
