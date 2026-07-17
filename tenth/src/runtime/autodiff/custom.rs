//! 用户自定义可微算子（PROJ-006 选项 2：Rust 端注册版）。
//!
//! 提供 `CustomBackward` trait 与 `CustomOpRegistry` 注册表。
//! forward/backward 由 Rust 端实现（论证 1 证明 .th 闭包在 VM 路径不可行）。
//!
//! ## 设计要点
//! - `CustomOpRegistry` 作为 `Vm` / `Interpreter` 的字段（论证 3 证明单线程无需锁）。
//! - `TapeOp::Custom(usize)` 的 `usize` 是 registry 分配的 op_id（论证 4 证明影响可控）。
//! - 用户在注册时声明 `op_class`（TapeOpClass），保证 T7 完备性定理不破坏。
//! - 运行时强制 shape 检查：backward 返回的梯度 shape 必须与对应输入 shape 一致。
//! - 编译期 shape 检查（护城河 A）走 `backward_shapes.rs` 的未知算子兜底，无需修改。
//!
//! ## API 示例（Rust 端）
//! ```ignore
//! #[derive(Debug)]
//! struct SquareOp;
//! impl CustomBackward for SquareOp {
//!     fn forward(&self, inputs: &[&Tensor]) -> Result<Tensor, TenthError> {
//!         let x = inputs[0];
//!         let data = x.data_as_f64_view().mapv(|v| v * v);
//!         Ok(Tensor::from_data(data))
//!     }
//!     fn backward(&self, inputs: &[&Tensor], grad: &Tensor) -> Result<Vec<Tensor>, TenthError> {
//!         let x = inputs[0];
//!         let g = grad.data_as_f64_view();
//!         let x_view = x.data_as_f64_view();
//!         let d_x = (&g * &x_view * 2.0).into();
//!         Ok(vec![Tensor::from_data(d_x)])
//!     }
//!     fn op_class(&self) -> TapeOpClass { TapeOpClass::Preserve }
//!     fn name(&self) -> &str { "square" }
//! }
//! // 注册：
//! let op_id = vm.register_custom_op(Box::new(SquareOp))?;
//! ```
//!
//! ## .th 端调用
//! ```tenth
//! // 通过 __call_custom_op native 调用，op_name 是注册时声明的 name
//! let y = __call_custom_op("square", x)
//! ```

use std::collections::HashMap;

use crate::error::TenthError;
use crate::runtime::relation_debugger::TapeOpClass;
use crate::runtime::tensor::Tensor;

/// 用户自定义可微算子的 trait。
///
/// forward_fn / backward_fn 由 Rust 端实现（论证 1 证明 .th 闭包在 VM 路径不可行：
/// VM 路径下闭包脱糖为 `Value::FnRef`，native 函数签名 `fn(&mut Vm, &[Value])`
/// 无法访问 HIR 解释器）。
///
/// 实现者必须声明 `op_class`（论证 4 证明否则破坏 T7 完备性定理）。
pub trait CustomBackward: std::fmt::Debug {
    /// 前向计算。
    ///
    /// `inputs`：输入张量切片（按注册时约定的顺序）。
    /// 返回：输出张量（单个；多输出场景请返回打包张量或拆分为多个 op）。
    fn forward(&self, inputs: &[&Tensor]) -> Result<Tensor, TenthError>;

    /// 反向传播。
    ///
    /// `inputs`：前向输入张量（用于计算梯度，顺序与 `forward` 相同）。
    /// `grad`：输出梯度（shape 与 forward 输出一致）。
    /// 返回：各输入的梯度（长度必须与 `inputs` 相同，shape 必须与对应输入一致）。
    ///
    /// 运行时会强制校验返回梯度的数量与 shape；不匹配将抛出 `ShapeMismatch`。
    fn backward(
        &self,
        inputs: &[&Tensor],
        grad: &Tensor,
    ) -> Result<Vec<Tensor>, TenthError>;

    /// 算子分类（用于 relation_debugger 的 T7 完备性）。
    /// 必须由用户声明（论证 4 证明否则破坏 T7 完备性定理）。
    fn op_class(&self) -> TapeOpClass;

    /// 算子名（用于 op_name 显示、注册查重、`__call_custom_op` 查找）。
    /// 必须唯一（同名校验在 `register` 时进行）。
    fn name(&self) -> &str;

    /// 可选：前向 shape 函数（护城河 A 的编译期 hint）。
    /// 默认返回 `None`（跳过编译期 shape 检查，运行时兜底——与现有未知算子策略一致）。
    /// 若实现，编译期会作为 hint 用于 shape 推断；运行时仍会强制检查实际输出 shape。
    fn forward_shape(&self, _input_shapes: &[Vec<usize>]) -> Option<Vec<usize>> {
        None
    }

    /// 可选：反向 shape 函数（护城河 A 的编译期 hint）。
    /// 默认返回 `None`（跳过编译期 shape 检查，运行时兜底）。
    /// 若实现，编译期会作为 hint；运行时仍会强制检查实际梯度 shape。
    fn backward_shapes(
        &self,
        _input_shapes: &[Vec<usize>],
        _output_shape: &[usize],
    ) -> Option<Vec<Vec<usize>>> {
        None
    }
}

/// 自定义算子注册表。
///
/// 作为 `Vm` / `Interpreter` 的字段（论证 3 证明 Tenth 单线程无需锁，
/// `Rc<RefCell<...>>` 也不满足 Send + Sync；故直接作为字段持有）。
///
/// ## 不变量
/// - `op_id` 由 `register` 单调递增分配，从 0 开始。
/// - `name_to_id` 与 `ops` 保持双向一致（同一 name ↔ 同一 id）。
/// - 同名算子只能注册一次（`register` 返回 `Err`）。
#[derive(Default)]
pub struct CustomOpRegistry {
    /// op_id → 算子实现。
    ops: HashMap<usize, Box<dyn CustomBackward>>,
    /// name → op_id（用于按名查找，避免线性扫描）。
    name_to_id: HashMap<String, usize>,
    /// 下一个 op_id（单调递增）。
    next_id: usize,
}

impl std::fmt::Debug for CustomOpRegistry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CustomOpRegistry")
            .field("op_count", &self.ops.len())
            .field("next_id", &self.next_id)
            .field("names", &self.name_to_id.keys().collect::<Vec<_>>())
            .finish()
    }
}

impl CustomOpRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    /// 注册自定义算子。
    ///
    /// 返回 `op_id`（用于 `TapeOp::Custom(op_id)`）。
    /// 若同名算子已注册，返回 `Err`。
    pub fn register(&mut self, op: Box<dyn CustomBackward>) -> Result<usize, String> {
        let name = op.name().to_string();
        if self.name_to_id.contains_key(&name) {
            return Err(format!("自定义算子 '{}' 已注册", name));
        }
        let id = self.next_id;
        self.next_id += 1;
        self.name_to_id.insert(name, id);
        self.ops.insert(id, op);
        Ok(id)
    }

    /// 按 id 查找算子。
    pub fn get(&self, id: usize) -> Option<&dyn CustomBackward> {
        self.ops.get(&id).map(|b| b.as_ref())
    }

    /// 按名查找算子 id。
    pub fn get_id_by_name(&self, name: &str) -> Option<usize> {
        self.name_to_id.get(name).copied()
    }

    /// 已注册算子数量。
    pub fn len(&self) -> usize {
        self.ops.len()
    }

    /// 是否为空。
    pub fn is_empty(&self) -> bool {
        self.ops.is_empty()
    }
}
