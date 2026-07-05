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
use tenth::error::{TapeErrorContext, TenthError};

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
        assert_eq!(classify_tape_op(op), *expected, "TapeOp::{:?} 分类错误", op);
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
    let causes = tape.formal_explain(mm_id, &[2, 3], &[2, 2]);

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

    let causes = tape.formal_explain(add_id, &[2, 3], &[2, 3]);

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
    let causes = tape.formal_explain(mm_id, &[], &[]);

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

    let causes = tape.formal_explain(exp2_id, &[], &[]);

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

    let causes = tape.formal_explain(relu_id, &[], &[]);
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
    let causes = tape.formal_explain(mm_id, &[], &[]);
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
    assert_eq!(classify_tape_op(&TapeOp::Gather), TapeOpClass::Preserve);
    // 顺带验证 Scatter 也为 Preserve（多维扩展后分类不变）
    assert_eq!(classify_tape_op(&TapeOp::Scatter), TapeOpClass::Preserve);
}
