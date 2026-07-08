//! 张量模块：双精度张量数据存储 + 张量类型。
//!
//! 子模块组织：
//! - `data`: `TensorData` 的方法实现
//! - `methods`: `Tensor` 的方法实现（67 个 pub fn）
//! - `display`: `Display` 实现
//! - `index`: `Index` trait 实现
//! - `ops`: 算术运算符重载

use crate::hir::types::BaseType;
use half::{bf16, f16};
use ndarray::ArrayD;

mod data;
mod display;
mod index;
mod methods;
mod ops;

/// 双精度张量数据存储。f32 与 f64 各占一个变体，避免 f32 退化为「语法糖 f64」。
/// Wave 2：新增 F16/BF16 半精度变体（前向运算时转 f32 计算后回写）。
#[derive(Debug, Clone)]
pub enum TensorData {
    F32(ArrayD<f32>),
    F64(ArrayD<f64>),
    F16(ArrayD<f16>),
    BF16(ArrayD<bf16>),
}

#[derive(Debug, Clone)]
pub struct Tensor {
    pub dtype: BaseType,
    pub data: TensorData,
    /// Accumulated gradient (populated by autodiff backward pass).
    /// grad 的 dtype 与 data 保持一致。
    pub grad: Option<TensorData>,
    /// Tape node id set by the interpreter during recording mode.
    /// Used to link tensors back to their computation-graph nodes.
    pub tape_id: Option<usize>,
}
