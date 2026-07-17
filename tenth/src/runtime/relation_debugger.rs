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

use std::collections::{HashMap, HashSet};
use crate::error::ErrorType;
use crate::runtime::autodiff::{Tape, TapeNode, TapeOp, op_name};

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
    /// 护城河 F Phase 1：5 类错误分类（ShapeMismatch / SilentSqueeze /
    /// BroadcastFail / GradDrift / DtypeConflict）。
    /// 由 backward 抛错时填充；`formal_explain` 本身不主动推断错误类型，
    /// 保留 `None` 表示"未分类"（由 runtime 部门在抛 `RelationError` 时填充）。
    pub error_type: Option<ErrorType>,
    /// 护城河 F Phase 1：边级归因 `(src_node, dst_node)`。
    /// 表示该根因候选节点是通过哪条边被 BFS 反向遍历到的：
    /// - `dst_node` == `tape_node_id`（本候选节点）
    /// - `src_node` == 从 v_err 反向到达本节点的"下一跳"（更靠近 v_err 的节点）
    /// `None` 表示本节点是 v_err 自身或无上游边（如直接作为根因）。
    pub edge: Option<(usize, usize)>,
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
        | TapeOp::Select
        // MaxPool2D / AvgPool2D: (N,C,H,W) → (N,C,H_out,W_out)，产生新空间维度
        | TapeOp::MaxPool2D
        | TapeOp::AvgPool2D => TapeOpClass::Construct,

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

        // Expand：维度增加或重排（MatMul 改变 shape 维度，Transpose/Reshape 重排）
        TapeOp::MatMul
        | TapeOp::BatchedMatMul
        | TapeOp::Transpose
        | TapeOp::Reshape => TapeOpClass::Expand,
        // Scatter：保留 base shape（覆盖部分位置，shape 不变）
        | TapeOp::Scatter
        // Gather：输出 shape == index.shape（从输入张量继承 shape，非新构造）
        | TapeOp::Gather
        // MaskedFill：保留 input shape（仅覆盖部分位置，shape 不变）
        | TapeOp::MaskedFill => TapeOpClass::Preserve,
    }
}

impl Tape {
    /// T2 §4.3 FormalExplain 算法。
    ///
    /// 输入：
    /// - `v_err`：报错节点 id（backward 失败的节点）
    /// - `expected`：期望的 shape（若未知则传空切片）
    /// - `actual`：实际的 shape（若未知则传空切片）
    /// - `error_msg`：backward 抛出的错误消息（用于 5 类错误分类）
    ///
    /// 输出：所有可达且可能相关的根因候选列表，按 ExplainsError > PartialExplain
    /// 排序。复杂度 O(|V|+|E|)。
    ///
    /// 护城河 F Phase 2：根据 v_err 节点的 op 与 error_msg 调用
    /// `classify_error_type` 推断 5 类错误类型，填入每个 RootCause.error_type。
    pub fn formal_explain(
        &self,
        v_err: usize,
        expected: &[usize],
        actual: &[usize],
        error_msg: &str,
    ) -> Vec<RootCause> {
        // 1. BFS 反向 reachable 集合（C1）+ 边级归因（parent 映射）
        //    parent[node_id] = Some(next_hop) 表示从 v_err 反向到达 node_id 的下一跳；
        //    parent[v_err] = None（起点）。
        let parent = self.bfs_reverse_reachable_with_edges(v_err);

        // 护城河 F Phase 2：对错误进行 5 类分类（针对错误发生点 v_err 一次性分类）
        // v_err 节点的 op 决定分类方向；error_msg 关键词决定具体类型。
        let v_err_op = self
            .node(v_err)
            .map(|n| n.op.clone())
            .unwrap_or(TapeOp::Input);
        let error_type = Self::classify_error_type(&v_err_op, expected, actual, error_msg);

        // 2. 对每个 reachable 节点判定 C2，构造 RootCause
        let mut causes: Vec<RootCause> = Vec::new();
        for &node_id in parent.keys() {
            // 跳过 v_err 自身（它是错误发生点，不是根因候选）
            if node_id == v_err {
                continue;
            }
            let node = match self.node(node_id) {
                Some(n) => n,
                None => continue,
            };
            let cls = self.classify_node(node, expected, actual);
            let explanation =
                self.render_explanation(node, &cls, expected, actual, Some(error_type.clone()));
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
            // 边级归因：(src=parent 下一跳, dst=本节点)
            // parent[dst]=Some(src) 表示 src 是更靠近 v_err 的那一端。
            let edge = parent
                .get(&node_id)
                .copied()
                .flatten()
                .map(|src| (src, node_id));
            causes.push(RootCause {
                tape_node_id: node_id,
                op: node.op.clone(),
                s_in,
                s_out,
                classification: cls,
                explanation,
                error_type: Some(error_type.clone()),
                edge,
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

    /// BFS 反向可达集合 + 边级归因（护城河 F Phase 1）。
    ///
    /// 与 `bfs_reverse_reachable` 相同的遍历，但额外记录每个可达节点
    /// 的"下一跳" parent（更靠近 v_err 的节点），用于 `RootCause::edge`。
    ///
    /// 返回 `HashMap<node_id, Option<parent>>`：
    /// - `parent[v_err] = None`（起点，无下一跳）
    /// - `parent[node] = Some(next_hop)` 表示从 v_err 反向到达 node 的下一跳是 next_hop
    ///
    /// 复杂度 O(|V|+|E|)，与原 BFS 一致（T2 F5 不变量保持）。
    fn bfs_reverse_reachable_with_edges(&self, v_err: usize) -> HashMap<usize, Option<usize>> {
        let mut parent: HashMap<usize, Option<usize>> = HashMap::new();
        let mut queue: Vec<usize> = vec![v_err];
        parent.insert(v_err, None);
        while let Some(node_id) = queue.pop() {
            if let Some(node) = self.node(node_id) {
                for &input_id in &node.inputs {
                    // 仅在首次访问时记录 parent（BFS 树的第一条边）
                    if !parent.contains_key(&input_id) {
                        parent.insert(input_id, Some(node_id));
                        queue.push(input_id);
                    }
                }
            }
        }
        parent
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
        error_type: Option<ErrorType>,
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
            self.root_cause_hint(&node.op, error_type)
        )
    }

    /// 针对特定算子+错误类型的根因提示（人类可读的诊断建议）。
    ///
    /// 护城河 F Phase 2：优先按 `error_type` 给出 4 类特殊错误的针对性提示；
    /// 若 `error_type` 为 `None` 或 `ShapeMismatch`，退化为按 op 分类的通用提示。
    fn root_cause_hint(&self, op: &TapeOp, error_type: Option<ErrorType>) -> &'static str {
        // 优先按 error_type 给出针对性提示（4 类特殊错误）
        if let Some(et) = error_type {
            match et {
                ErrorType::SilentSqueeze => {
                    return "MatMul 1D squeeze 被拒绝——这通常意味着输入 shape 设计错误，梯度无法 squeeze 回原始 shape";
                }
                ErrorType::BroadcastFail => {
                    return "广播失败——两个张量的 shape 无法对齐，检查上游算子的输出 shape";
                }
                ErrorType::GradDrift => {
                    return "梯度 shape 漂移——反向传播到参数时 grad shape 与参数 shape 不一致，根因可能在前向的 reshape/transpose";
                }
                ErrorType::DtypeConflict => {
                    return "dtype 冲突——f32/f64 混用导致精度提升，检查上游算子的 dtype";
                }
                ErrorType::ShapeMismatch => {
                    // 退化为按 op 分类的通用提示
                }
            }
        }
        // 按 op 分类的通用提示（ShapeMismatch 或未分类）
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
            TapeOp::BatchedMatMul => "可能是 batched matmul 维度不匹配（B/K/N 不一致），检查两侧 batch 维与内侧 K 维",
            TapeOp::Scatter => "Scatter 的 index 越界或与 src/base shape 不匹配",
            TapeOp::Gather => "Gather 的 index 越界或与 base shape 不匹配（除 dim 维外必须一致）",
            TapeOp::Reshape => "Reshape 的目标 shape 元素数与输入不一致",
            TapeOp::MaskedFill => "MaskedFill 的 mask shape 与输入不一致",
            _ => "检查该节点的输入 shape 与算子语义是否匹配",
        }
    }

    /// 护城河 F Phase 2：根据算子、shape 和错误消息推断 5 类错误类型。
    ///
    /// 分类规则（按优先级，先匹配先返回）：
    /// 1. op 是 MatMul/BatchedMatMul 且 error_msg 含 "squeeze" 或 "1D" → SilentSqueeze
    /// 2. op 是 Add/Sub/Mul/Div 且 error_msg 含 "broadcast"（覆盖 "unbroadcast"）→ BroadcastFail
    /// 3. op 是 Input 且 error_msg 含 "acc_grad"/"反向传播 shape 错误"/"梯度 shape" → GradDrift
    /// 4. error_msg 含 "dtype"/"f32/f64" → DtypeConflict（与 op 无关）
    /// 5. 其他 shape 不匹配 → ShapeMismatch（兜底）
    ///
    /// 说明：
    /// - DtypeConflict 排在 op 特定分类之后——当 op 特定关键词命中时，
    ///   优先返回更具体的类型（如 MatMul+squeeze → SilentSqueeze 而非 DtypeConflict）。
    ///   若 error_msg 同时含 "dtype" 但未命中 op 特定关键词，则返回 DtypeConflict。
    /// - GradDrift 使用精准关键词（"反向传播 shape 错误"）而非泛化的 "shape 错误"，
    ///   避免误匹配非 GradDrift 场景（如 "参数 shape 错误" 应归类为 ShapeMismatch）。
    /// - BroadcastFail 检测 "broadcast" 子串，自然覆盖 "unbroadcast"（含 "broadcast"）。
    fn classify_error_type(
        op: &TapeOp,
        _expected: &[usize],
        _actual: &[usize],
        error_msg: &str,
    ) -> ErrorType {
        // 1-3：op 特定关键词匹配
        match op {
            TapeOp::MatMul | TapeOp::BatchedMatMul => {
                // SilentSqueeze：1D 输入 squeeze 失败
                if error_msg.contains("squeeze") || error_msg.contains("1D") {
                    return ErrorType::SilentSqueeze;
                }
            }
            TapeOp::Add | TapeOp::Sub | TapeOp::Mul | TapeOp::Div => {
                // BroadcastFail：广播失败（"broadcast" 覆盖 "unbroadcast"）
                if error_msg.contains("broadcast") {
                    return ErrorType::BroadcastFail;
                }
            }
            TapeOp::Input => {
                // GradDrift：梯度累积时 shape 漂移
                // 用精准关键词避免误匹配（"shape 错误" 过于泛化）
                if error_msg.contains("acc_grad")
                    || error_msg.contains("反向传播 shape 错误")
                    || error_msg.contains("梯度 shape")
                {
                    return ErrorType::GradDrift;
                }
            }
            _ => {}
        }
        // 4：DtypeConflict 独立检测（不依赖 op）
        if error_msg.contains("dtype") || error_msg.contains("f32/f64") {
            return ErrorType::DtypeConflict;
        }
        // 5：兜底
        ErrorType::ShapeMismatch
    }
}

// op_name 实现已统一到 `crate::runtime::autodiff::grad::op_name`（pub(crate)），
// 本模块通过 `use` 导入复用，不再维护重复副本。
// 历史：原先此处有一份同步副本（27 变体），护城河 F Phase 1 去重时删除。

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
            TapeOp::Abs, TapeOp::Scatter, TapeOp::Gather, TapeOp::Reshape, TapeOp::MaskedFill,
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
        // Preserve: 15 个（含 Scatter / Gather / MaskedFill）
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
        assert_eq!(classify_tape_op(&TapeOp::Scatter), TapeOpClass::Preserve);
        assert_eq!(classify_tape_op(&TapeOp::Gather), TapeOpClass::Preserve);
        assert_eq!(classify_tape_op(&TapeOp::MaskedFill), TapeOpClass::Preserve);
        // Reduce: 2 个
        assert_eq!(classify_tape_op(&TapeOp::Sum), TapeOpClass::Reduce);
        assert_eq!(classify_tape_op(&TapeOp::Mean), TapeOpClass::Reduce);
        // Expand: 4 个（含 BatchedMatMul / Reshape）
        assert_eq!(classify_tape_op(&TapeOp::MatMul), TapeOpClass::Expand);
        assert_eq!(classify_tape_op(&TapeOp::BatchedMatMul), TapeOpClass::Expand);
        assert_eq!(classify_tape_op(&TapeOp::Transpose), TapeOpClass::Expand);
        assert_eq!(classify_tape_op(&TapeOp::Reshape), TapeOpClass::Expand);
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
        let causes = tape.formal_explain(mm_id, &[2, 3], &[2, 2], "");
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
        let causes = tape.formal_explain(relu_id, &[], &[], "");
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
        let causes = tape.formal_explain(relu2_id, &[], &[], "");
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

        let causes = tape.formal_explain(relu2_id, &[], &[], "");
        // 第一个应为 ExplainsError（x_id），第二个为 PartialExplain（relu_id）
        assert_eq!(causes[0].classification, ExplainClass::ExplainsError);
        assert_eq!(causes[1].classification, ExplainClass::PartialExplain);
    }

    // ── 护城河 F Phase 2：5 类错误分类测试 ──────────────────────────────

    #[test]
    fn test_classify_error_type_silent_squeeze() {
        // MatMul + "squeeze" → SilentSqueeze
        assert_eq!(
            Tape::classify_error_type(&TapeOp::MatMul, &[], &[], "MatMul 反向 1D squeeze 失败"),
            ErrorType::SilentSqueeze
        );
        // BatchedMatMul + "squeeze" → SilentSqueeze
        assert_eq!(
            Tape::classify_error_type(&TapeOp::BatchedMatMul, &[], &[], "squeeze 失败"),
            ErrorType::SilentSqueeze
        );
        // MatMul + "1D"（无 "squeeze"）→ SilentSqueeze（Phase 2 增强关键词）
        assert_eq!(
            Tape::classify_error_type(&TapeOp::MatMul, &[], &[], "MatMul 反向 1D 输入处理失败"),
            ErrorType::SilentSqueeze
        );
    }

    #[test]
    fn test_classify_error_type_broadcast_fail() {
        // Add + "unbroadcast" → BroadcastFail
        assert_eq!(
            Tape::classify_error_type(&TapeOp::Add, &[], &[], "unbroadcast reshape 失败"),
            ErrorType::BroadcastFail
        );
        // Sub + "unbroadcast" → BroadcastFail
        assert_eq!(
            Tape::classify_error_type(&TapeOp::Sub, &[], &[], "unbroadcast 元素数不匹配"),
            ErrorType::BroadcastFail
        );
        // Mul/Div 同理
        assert_eq!(
            Tape::classify_error_type(&TapeOp::Mul, &[], &[], "unbroadcast 失败"),
            ErrorType::BroadcastFail
        );
        assert_eq!(
            Tape::classify_error_type(&TapeOp::Div, &[], &[], "unbroadcast"),
            ErrorType::BroadcastFail
        );
        // Add + "broadcast"（无 "un" 前缀）→ BroadcastFail（Phase 2 增强关键词）
        assert_eq!(
            Tape::classify_error_type(&TapeOp::Add, &[], &[], "broadcast 失败，shape 无法对齐"),
            ErrorType::BroadcastFail
        );
    }

    #[test]
    fn test_classify_error_type_grad_drift() {
        // Input + "acc_grad" → GradDrift
        assert_eq!(
            Tape::classify_error_type(&TapeOp::Input, &[], &[], "acc_grad shape 不匹配"),
            ErrorType::GradDrift
        );
        // Input + "反向传播 shape 错误" → GradDrift
        assert_eq!(
            Tape::classify_error_type(&TapeOp::Input, &[], &[], "反向传播 shape 错误（节点 #0 Input）"),
            ErrorType::GradDrift
        );
        // Input + "梯度 shape" → GradDrift（任务要求关键词）
        assert_eq!(
            Tape::classify_error_type(&TapeOp::Input, &[], &[], "梯度 shape 漂移"),
            ErrorType::GradDrift
        );
    }

    #[test]
    fn test_classify_error_type_dtype_conflict() {
        // 任意 op + "dtype" → DtypeConflict（当 op 特定关键词未命中时）
        assert_eq!(
            Tape::classify_error_type(&TapeOp::ReLU, &[], &[], "dtype 不匹配"),
            ErrorType::DtypeConflict
        );
        // 任意 op + "f32/f64" → DtypeConflict
        assert_eq!(
            Tape::classify_error_type(&TapeOp::Exp, &[], &[], "f32/f64 混用"),
            ErrorType::DtypeConflict
        );
        // MatMul + "dtype"（无 "squeeze"）→ DtypeConflict
        assert_eq!(
            Tape::classify_error_type(&TapeOp::MatMul, &[], &[], "dtype 冲突"),
            ErrorType::DtypeConflict
        );
    }

    #[test]
    fn test_classify_error_type_shape_mismatch_default() {
        // MatMul 无 "squeeze"/"1D" → ShapeMismatch
        assert_eq!(
            Tape::classify_error_type(&TapeOp::MatMul, &[], &[], "MatMul 维度不匹配"),
            ErrorType::ShapeMismatch
        );
        // Add 无 "broadcast" → ShapeMismatch
        assert_eq!(
            Tape::classify_error_type(&TapeOp::Add, &[], &[], "shape 不一致"),
            ErrorType::ShapeMismatch
        );
        // Input 无 "acc_grad"/"反向传播 shape 错误"/"梯度 shape" → ShapeMismatch
        //（"参数 shape 错误" 不含精准关键词，不误判为 GradDrift）
        assert_eq!(
            Tape::classify_error_type(&TapeOp::Input, &[], &[], "参数 shape 错误"),
            ErrorType::ShapeMismatch
        );
        // 空 error_msg → ShapeMismatch
        assert_eq!(
            Tape::classify_error_type(&TapeOp::Conv2D, &[], &[], ""),
            ErrorType::ShapeMismatch
        );
    }

    #[test]
    fn test_classify_error_type_priority() {
        // op 特定关键词优先于 DtypeConflict
        // MatMul + "squeeze" + "dtype" → SilentSqueeze（op 特定优先）
        assert_eq!(
            Tape::classify_error_type(&TapeOp::MatMul, &[], &[], "squeeze dtype"),
            ErrorType::SilentSqueeze
        );
        // Add + "unbroadcast" + "dtype" → BroadcastFail
        assert_eq!(
            Tape::classify_error_type(&TapeOp::Add, &[], &[], "unbroadcast dtype"),
            ErrorType::BroadcastFail
        );
        // Input + "acc_grad" + "dtype" → GradDrift
        assert_eq!(
            Tape::classify_error_type(&TapeOp::Input, &[], &[], "acc_grad dtype"),
            ErrorType::GradDrift
        );
    }

    #[test]
    fn test_formal_explain_fills_error_type() {
        // 验证 formal_explain 正确填充 RootCause.error_type
        // 构造 MatMul 节点，error_msg 含 "squeeze" → SilentSqueeze
        let a = make_tensor(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let b = make_tensor(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let mut tape = Tape::new();
        let a_id = tape.input(a.clone());
        let b_id = tape.input(b.clone());
        let result = make_tensor(vec![0.0; 4], vec![2, 2]);
        let mm_id = tape.binary(TapeOp::MatMul, a_id, b_id, a.clone(), b.clone(), result.clone());

        // SilentSqueeze：MatMul + "squeeze"
        let causes = tape.formal_explain(mm_id, &[1], &[2], "MatMul 反向 1D squeeze 失败");
        assert!(causes.len() >= 1);
        for c in &causes {
            assert_eq!(c.error_type, Some(ErrorType::SilentSqueeze));
        }

        // ShapeMismatch：MatMul + 无 "squeeze"
        let causes2 = tape.formal_explain(mm_id, &[], &[], "MatMul 维度不匹配");
        for c in &causes2 {
            assert_eq!(c.error_type, Some(ErrorType::ShapeMismatch));
        }
    }
}
