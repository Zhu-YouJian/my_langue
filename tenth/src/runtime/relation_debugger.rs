//! 护城河 F：关系调试器（T2 FormalExplain 动态层 MVP）。
//!
//! 在 autodiff backward 因 shape 不匹配失败时，基于 Wengert tape 的 DAG 结构
//! 解释根因。本模块实现 T2 §4.3 FormalExplain 算法的 C1（可达）+ C2（相关）
//! 两级分类；C3 counterfactual 留作 Phase 2 / T8-4 开放问题。
//!
//! ## 算法复杂度
//! BFS 反向可达 + 节点分类均为 O(|V|+|E|)（T2 F5），不改变 backward 的
//! 渐近复杂度。
//!
//! ## 输出分类
//! - `ExplainsError`：节点既可达（C1），又与错误形状相关（C2a 或 C2b）
//! - `PartialExplain`：节点可达（C1），但与错误形状不直接相关
//! - `Unrelated`：节点不可达（不在候选集中）
//!
//! ## TapeOp 分类（T2 引理 3.1 / T7 定理完备分类）
//! - `Construct`：产生新 shape（Input / CrossEntropy / Conv2D / BatchNorm /
//!   LayerNorm / Dropout / Select）
//! - `Preserve`：shape 不变（Add / Sub / Mul / Div / Neg / ReLU / Exp / Log /
//!   Sigmoid / Softmax / Gelu / Abs）
//! - `Reduce`：维度减少（Sum / Mean）
//! - `Expand`：维度增加（MatMul / Transpose）

use std::collections::HashSet;
use crate::runtime::autodiff::{Tape, TapeNode, TapeOp};

/// 根因分类（T2 §4.3）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExplainClass {
    /// C1 ∧ C2：可达且与错误形状相关——强根因候选。
    ExplainsError,
    /// C1 ∧ ¬C2：可达但与错误形状不直接相关——弱根因候选。
    PartialExplain,
    /// ¬C1：不可达——非根因。
    Unrelated,
}

/// TapeOp 的形状变换分类（T2 引理 3.1 / T7 完备互斥分类）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TapeOpClass {
    /// 产生新 shape（不继承输入的形状语义）。
    Construct,
    /// 保留输入 shape（元素级运算）。
    Preserve,
    /// 减少维度（归约）。
    Reduce,
    /// 增加维度或重排（矩阵乘 / 转置）。
    Expand,
}

/// 单个根因候选节点及其人类可读说明。
#[derive(Debug, Clone)]
pub struct RootCause {
    pub tape_node_id: usize,
    pub op: TapeOp,
    /// 各输入张量的 shape（用于诊断）。
    pub s_in: Vec<Vec<usize>>,
    /// 输出张量的 shape。
    pub s_out: Vec<usize>,
    pub classification: ExplainClass,
    /// 人类可读说明（含节点 id / 算子名 / shape / 分类理由）。
    pub explanation: String,
}

/// 对 TapeOp 进行形状变换分类（T2 引理 3.1 / T7 定理）。
///
/// 该分类是完备且互斥的：每个 TapeOp 变体唯一映射到一个类别。
/// 用于 FormalExplain 算法的 C2b 判定（节点是否改变 shape 方向）。
pub fn classify_tape_op(op: &TapeOp) -> TapeOpClass {
    match op {
        // Construct：产生新 shape（不继承输入的形状语义）
        TapeOp::Input
        | TapeOp::CrossEntropy
        | TapeOp::Conv2D
        | TapeOp::BatchNorm
        | TapeOp::LayerNorm
        | TapeOp::Dropout
        | TapeOp::Select => TapeOpClass::Construct,

        // Preserve：shape 不变（元素级运算 + softmax/gelu 等保持输入 shape）
        TapeOp::Add
        | TapeOp::Sub
        | TapeOp::Mul
        | TapeOp::Div
        | TapeOp::Neg
        | TapeOp::ReLU
        | TapeOp::Exp
        | TapeOp::Log
        | TapeOp::Sigmoid
        | TapeOp::Softmax
        | TapeOp::Gelu
        | TapeOp::Abs => TapeOpClass::Preserve,

        // Reduce：维度减少（归约到标量）
        TapeOp::Sum
        | TapeOp::Mean => TapeOpClass::Reduce,

        // Expand：维度增加或重排（MatMul 改变 shape 维度，Transpose 重排）
        TapeOp::MatMul
        | TapeOp::Transpose => TapeOpClass::Expand,
    }
}

impl Tape {
    /// T2 §4.3 FormalExplain 算法。
    ///
    /// 输入：
    /// - `v_err`：报错节点 id（backward 失败的节点）
    /// - `expected`：期望的 shape（若未知则传空切片）
    /// - `actual`：实际的 shape（若未知则传空切片）
    ///
    /// 输出：所有可达且可能相关的根因候选列表，按 ExplainsError > PartialExplain
    /// 排序。复杂度 O(|V|+|E|)。
    pub fn formal_explain(
        &self,
        v_err: usize,
        expected: &[usize],
        actual: &[usize],
    ) -> Vec<RootCause> {
        // 1. BFS 反向 reachable 集合（C1）
        let reachable = self.bfs_reverse_reachable(v_err);

        // 2. 对每个 reachable 节点判定 C2，构造 RootCause
        let mut causes: Vec<RootCause> = Vec::new();
        for &node_id in &reachable {
            // 跳过 v_err 自身（它是错误发生点，不是根因候选）
            if node_id == v_err {
                continue;
            }
            let node = match self.node(node_id) {
                Some(n) => n,
                None => continue,
            };
            let cls = self.classify_node(node, expected, actual);
            let explanation = self.render_explanation(node, &cls, expected, actual);
            let s_in: Vec<Vec<usize>> = node
                .input_tensors
                .iter()
                .map(|t| t.borrow().shape())
                .collect();
            // s_out：input_tensors 的最后一个元素是 result（约定）
            let s_out: Vec<usize> = node
                .input_tensors
                .last()
                .map(|t| t.borrow().shape())
                .unwrap_or_default();
            causes.push(RootCause {
                tape_node_id: node_id,
                op: node.op.clone(),
                s_in,
                s_out,
                classification: cls,
                explanation,
            });
        }

        // 3. 按 ExplainsError > PartialExplain 排序（Unrelated 已被 reachable 过滤）
        causes.sort_by(|a, b| {
            let rank = |c: &ExplainClass| match c {
                ExplainClass::ExplainsError => 0,
                ExplainClass::PartialExplain => 1,
                ExplainClass::Unrelated => 2,
            };
            rank(&a.classification).cmp(&rank(&b.classification))
        });

        causes
    }

    /// BFS 反向可达集合（C1）：从 v_err 出发，沿 inputs 反向遍历，返回所有可达节点。
    /// 复杂度 O(|V|+|E|)。
    fn bfs_reverse_reachable(&self, v_err: usize) -> HashSet<usize> {
        let mut visited: HashSet<usize> = HashSet::new();
        let mut queue: Vec<usize> = vec![v_err];
        visited.insert(v_err);
        while let Some(node_id) = queue.pop() {
            if let Some(node) = self.node(node_id) {
                for &input_id in &node.inputs {
                    if visited.insert(input_id) {
                        queue.push(input_id);
                    }
                }
            }
        }
        visited
    }

    /// C2 相关性判定（T2 §4.3）。
    ///
    /// - C2a：节点的输出 shape 与 actual shape 直接匹配
    /// - C2b：节点的 TapeOp 类别为 Construct / Reduce / Expand（非 Preserve / Input）
    ///
    /// C1 已由 bfs_reverse_reachable 保证。返回 ExplainsError（C1∧C2）
    /// 或 PartialExplain（C1∧¬C2）。
    fn classify_node(
        &self,
        node: &TapeNode,
        _expected: &[usize],
        actual: &[usize],
    ) -> ExplainClass {
        // C2a：output shape == actual shape
        let c2a = if !actual.is_empty() {
            node.input_tensors
                .last()
                .map(|t| t.borrow().shape() == actual.to_vec())
                .unwrap_or(false)
        } else {
            false
        };

        // C2b：op 类别为 Construct / Reduce / Expand（非 Preserve）
        // 注意：Input 节点是 Construct，但作为叶子参数它"产生"了 shape，
        // 因此视为根因候选（C2b 满足）。
        let op_class = classify_tape_op(&node.op);
        let c2b = matches!(
            op_class,
            TapeOpClass::Construct | TapeOpClass::Reduce | TapeOpClass::Expand
        );

        if c2a || c2b {
            ExplainClass::ExplainsError
        } else {
            ExplainClass::PartialExplain
        }
    }

    /// 生成人类可读的根因说明。
    fn render_explanation(
        &self,
        node: &TapeNode,
        cls: &ExplainClass,
        expected: &[usize],
        actual: &[usize],
    ) -> String {
        let op_name = op_name(&node.op);
        let s_out: Vec<usize> = node
            .input_tensors
            .last()
            .map(|t| t.borrow().shape())
            .unwrap_or_default();
        let s_in: Vec<Vec<usize>> = node
            .input_tensors
            .iter()
            .map(|t| t.borrow().shape())
            .collect();

        let class_str = match cls {
            ExplainClass::ExplainsError => "强根因",
            ExplainClass::PartialExplain => "弱根因（路径上但非形状来源）",
            ExplainClass::Unrelated => "无关",
        };

        let shape_info = if !actual.is_empty() {
            format!("实际 shape={:?} ", actual)
        } else {
            String::new()
        };
        let expected_info = if !expected.is_empty() {
            format!("期望 shape={:?} ", expected)
        } else {
            String::new()
        };

        format!(
            "节点 #{} ({}) [{}]: 输入 shape={:?}, 输出 shape={:?}；{}{}——{}",
            node.id, op_name, class_str, s_in, s_out, shape_info, expected_info,
            self.root_cause_hint(&node.op)
        )
    }

    /// 针对特定算子的根因提示（人类可读的诊断建议）。
    fn root_cause_hint(&self, op: &TapeOp) -> &'static str {
        match op {
            TapeOp::MatMul => "可能是矩阵维度不匹配（K ≠ K'），检查权重矩阵的第二维与输入的第一维",
            TapeOp::Add | TapeOp::Sub | TapeOp::Mul | TapeOp::Div => {
                "可能是广播失败，检查两个操作数的 shape 是否可广播"
            }
            TapeOp::Conv2D => "可能是卷积参数（input channels / kernel size）与权重不匹配",
            TapeOp::Transpose => "可能是转置后维度与下游算子期望不一致",
            TapeOp::Sum | TapeOp::Mean => "归约后 shape 改变，检查下游是否期望归约前的 shape",
            TapeOp::Input => "叶子参数 shape 可能与算子期望不一致",
            TapeOp::CrossEntropy => "交叉熵的 logits 与 target shape/类别数可能不匹配",
            TapeOp::BatchNorm => "BatchNorm 的 num_features 可能与输入通道数不一致",
            TapeOp::LayerNorm => "LayerNorm 的 normalized_shape 可能与输入最后一维不一致",
            TapeOp::Dropout => "Dropout 不应改变 shape，若失败检查输入 shape 是否合法",
            TapeOp::Select => "Select 的 then/else 分支 shape 可能不一致",
            _ => "检查该节点的输入 shape 与算子语义是否匹配",
        }
    }
}

// 复用 autodiff.rs 的 op_name 实现，避免重复定义。
// 由于 autodiff::op_name 是私有 fn，这里重新实现一份用于展示。
// 注意：两份必须保持同步——若 TapeOp 新增变体，此处也要更新。
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
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::rc::Rc;
    use std::cell::RefCell;
    use crate::runtime::tensor::Tensor;
    use crate::runtime::autodiff::Tape;

    fn make_tensor(data: Vec<f64>, shape: Vec<usize>) -> Rc<RefCell<Tensor>> {
        Rc::new(RefCell::new(Tensor::from_vec(data, shape)))
    }

    #[test]
    fn test_classify_tape_op_completeness() {
        // T7 定理：四类互斥完备。每个 TapeOp 变体必须落入恰好一类。
        let all_ops = [
            TapeOp::Input, TapeOp::Add, TapeOp::Sub, TapeOp::Mul, TapeOp::Div,
            TapeOp::Neg, TapeOp::ReLU, TapeOp::MatMul, TapeOp::Transpose,
            TapeOp::Sum, TapeOp::Mean, TapeOp::Exp, TapeOp::Log, TapeOp::Sigmoid,
            TapeOp::Softmax, TapeOp::CrossEntropy, TapeOp::Dropout, TapeOp::Conv2D,
            TapeOp::BatchNorm, TapeOp::LayerNorm, TapeOp::Gelu, TapeOp::Select,
            TapeOp::Abs,
        ];
        for op in &all_ops {
            let _cls = classify_tape_op(op); // 不 panic 即可
        }
        // Construct: 7 个
        assert_eq!(classify_tape_op(&TapeOp::Input), TapeOpClass::Construct);
        assert_eq!(classify_tape_op(&TapeOp::CrossEntropy), TapeOpClass::Construct);
        assert_eq!(classify_tape_op(&TapeOp::Conv2D), TapeOpClass::Construct);
        assert_eq!(classify_tape_op(&TapeOp::BatchNorm), TapeOpClass::Construct);
        assert_eq!(classify_tape_op(&TapeOp::LayerNorm), TapeOpClass::Construct);
        assert_eq!(classify_tape_op(&TapeOp::Dropout), TapeOpClass::Construct);
        assert_eq!(classify_tape_op(&TapeOp::Select), TapeOpClass::Construct);
        // Preserve: 12 个
        assert_eq!(classify_tape_op(&TapeOp::Add), TapeOpClass::Preserve);
        assert_eq!(classify_tape_op(&TapeOp::Sub), TapeOpClass::Preserve);
        assert_eq!(classify_tape_op(&TapeOp::Mul), TapeOpClass::Preserve);
        assert_eq!(classify_tape_op(&TapeOp::Div), TapeOpClass::Preserve);
        assert_eq!(classify_tape_op(&TapeOp::Neg), TapeOpClass::Preserve);
        assert_eq!(classify_tape_op(&TapeOp::ReLU), TapeOpClass::Preserve);
        assert_eq!(classify_tape_op(&TapeOp::Exp), TapeOpClass::Preserve);
        assert_eq!(classify_tape_op(&TapeOp::Log), TapeOpClass::Preserve);
        assert_eq!(classify_tape_op(&TapeOp::Sigmoid), TapeOpClass::Preserve);
        assert_eq!(classify_tape_op(&TapeOp::Softmax), TapeOpClass::Preserve);
        assert_eq!(classify_tape_op(&TapeOp::Gelu), TapeOpClass::Preserve);
        assert_eq!(classify_tape_op(&TapeOp::Abs), TapeOpClass::Preserve);
        // Reduce: 2 个
        assert_eq!(classify_tape_op(&TapeOp::Sum), TapeOpClass::Reduce);
        assert_eq!(classify_tape_op(&TapeOp::Mean), TapeOpClass::Reduce);
        // Expand: 2 个
        assert_eq!(classify_tape_op(&TapeOp::MatMul), TapeOpClass::Expand);
        assert_eq!(classify_tape_op(&TapeOp::Transpose), TapeOpClass::Expand);
    }

    #[test]
    fn test_bfs_reverse_reachable() {
        // 构造链：x → relu → matmul → loss
        let x = make_tensor(vec![-1.0, 2.0], vec![2]);
        let w = make_tensor(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);

        let mut tape = Tape::new();
        let x_id = tape.input(x.clone());
        let w_id = tape.input(w.clone());

        // relu(x)
        let relu_data = x.borrow().data.mapv(|v| if v > 0.0 { v } else { 0.0 });
        let relu = Rc::new(RefCell::new(Tensor::from_data(relu_data)));
        let relu_id = tape.unary(TapeOp::ReLU, x_id, x.clone(), relu.clone());

        // matmul(relu, w)
        let r_data = relu.borrow().data.view().insert_axis(ndarray::Axis(0)).into_owned().into_dyn();
        let w_data = w.borrow().data.clone();
        let r2 = r_data.view().into_dimensionality::<ndarray::Ix2>().unwrap();
        let w2 = w_data.view().into_dimensionality::<ndarray::Ix2>().unwrap();
        let mm_data = r2.dot(&w2).into_dyn();
        let mm = Rc::new(RefCell::new(Tensor::from_data(mm_data)));
        let mm_id = tape.binary(TapeOp::MatMul, relu_id, w_id, relu.clone(), w.clone(), mm.clone());

        let reachable = tape.bfs_reverse_reachable(mm_id);
        // mm_id 自身、relu_id、x_id、w_id 都应可达
        assert!(reachable.contains(&mm_id));
        assert!(reachable.contains(&relu_id));
        assert!(reachable.contains(&x_id));
        assert!(reachable.contains(&w_id));
    }

    #[test]
    fn test_formal_explain_matmul_root_cause() {
        // 构造 (M,K)@(K',N) where K ≠ K'——但 forward 已会失败，
        // 这里直接构造合法 forward 的 tape，然后调用 formal_explain 模拟 backward 失败。
        let a = make_tensor(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let b = make_tensor(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]); // 注意 K'=2 ≠ K=3
        let mut tape = Tape::new();
        let a_id = tape.input(a.clone());
        let b_id = tape.input(b.clone());

        // 模拟 forward 已完成（不实际计算 matmul，只记录节点）
        // 这里我们直接构造一个 result tensor 占位
        let result = make_tensor(vec![0.0; 4], vec![2, 2]);
        let mm_id = tape.binary(TapeOp::MatMul, a_id, b_id, a.clone(), b.clone(), result.clone());

        // 调用 formal_explain：v_err = mm_id, actual = [2,2]
        let causes = tape.formal_explain(mm_id, &[2, 3], &[2, 2]);
        // 应该返回 a_id 和 b_id 两个候选（不含 mm_id 自身）
        assert_eq!(causes.len(), 2);
        // 由于 Input 是 Construct，二者都应分类为 ExplainsError
        for c in &causes {
            assert_eq!(c.classification, ExplainClass::ExplainsError);
        }
    }

    #[test]
    fn test_formal_explain_unreachable_filtered() {
        // 构造两个独立子图：a→b 和 c→d，验证 c 不在 b 的根因候选集中
        let a = make_tensor(vec![1.0, 2.0], vec![2]);
        let c = make_tensor(vec![3.0, 4.0], vec![2]);

        let mut tape = Tape::new();
        let a_id = tape.input(a.clone());
        let c_id = tape.input(c.clone());

        // a → relu
        let relu_data = a.borrow().data.mapv(|v| if v > 0.0 { v } else { 0.0 });
        let relu = Rc::new(RefCell::new(Tensor::from_data(relu_data)));
        let relu_id = tape.unary(TapeOp::ReLU, a_id, a.clone(), relu.clone());

        // c → exp（独立子图）
        let exp_data = c.borrow().data.mapv(|v| v.exp());
        let exp = Rc::new(RefCell::new(Tensor::from_data(exp_data)));
        let exp_id = tape.unary(TapeOp::Exp, c_id, c.clone(), exp.clone());

        // 调用 formal_explain on relu_id：c_id 不应出现在结果中
        let causes = tape.formal_explain(relu_id, &[], &[]);
        let cause_ids: Vec<usize> = causes.iter().map(|c| c.tape_node_id).collect();
        assert!(!cause_ids.contains(&c_id));
        assert!(!cause_ids.contains(&exp_id));
        // a_id 应该出现（reachable）
        assert!(cause_ids.contains(&a_id));
    }

    #[test]
    fn test_formal_explain_preserve_is_partial() {
        // 链式：x → relu → relu2 → 报错
        // relu 是 Preserve 节点，应分类为 PartialExplain
        let x = make_tensor(vec![-1.0, 2.0], vec![2]);
        let mut tape = Tape::new();
        let x_id = tape.input(x.clone());

        let relu_data = x.borrow().data.mapv(|v| if v > 0.0 { v } else { 0.0 });
        let relu = Rc::new(RefCell::new(Tensor::from_data(relu_data)));
        let relu_id = tape.unary(TapeOp::ReLU, x_id, x.clone(), relu.clone());

        let relu2_data = relu.borrow().data.mapv(|v| if v > 0.0 { v } else { 0.0 });
        let relu2 = Rc::new(RefCell::new(Tensor::from_data(relu2_data)));
        let relu2_id = tape.unary(TapeOp::ReLU, relu_id, relu.clone(), relu2.clone());

        // 调用 formal_explain on relu2_id（不提供 actual，C2a 失效）
        let causes = tape.formal_explain(relu2_id, &[], &[]);
        // x_id 是 Input（Construct）→ ExplainsError
        // relu_id 是 ReLU（Preserve）→ PartialExplain
        let x_cause = causes.iter().find(|c| c.tape_node_id == x_id).unwrap();
        assert_eq!(x_cause.classification, ExplainClass::ExplainsError);
        let relu_cause = causes.iter().find(|c| c.tape_node_id == relu_id).unwrap();
        assert_eq!(relu_cause.classification, ExplainClass::PartialExplain);
    }

    #[test]
    fn test_formal_explain_sorting() {
        // 验证 ExplainsError 排在 PartialExplain 之前
        let x = make_tensor(vec![-1.0, 2.0], vec![2]);
        let mut tape = Tape::new();
        let x_id = tape.input(x.clone());

        let relu_data = x.borrow().data.mapv(|v| if v > 0.0 { v } else { 0.0 });
        let relu = Rc::new(RefCell::new(Tensor::from_data(relu_data)));
        let relu_id = tape.unary(TapeOp::ReLU, x_id, x.clone(), relu.clone());

        let relu2_data = relu.borrow().data.mapv(|v| if v > 0.0 { v } else { 0.0 });
        let relu2 = Rc::new(RefCell::new(Tensor::from_data(relu2_data)));
        let relu2_id = tape.unary(TapeOp::ReLU, relu_id, relu.clone(), relu2.clone());

        let causes = tape.formal_explain(relu2_id, &[], &[]);
        // 第一个应为 ExplainsError（x_id），第二个为 PartialExplain（relu_id）
        assert_eq!(causes[0].classification, ExplainClass::ExplainsError);
        assert_eq!(causes[1].classification, ExplainClass::PartialExplain);
    }
}
