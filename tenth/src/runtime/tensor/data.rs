use super::TensorData;
use crate::hir::types::BaseType;
use half::{bf16, f16};
use ndarray::{ArrayD, IxDyn};

impl TensorData {
    /// 返回数据元素 dtype。
    pub fn dtype(&self) -> BaseType {
        match self {
            TensorData::F32(_) => BaseType::F32,
            TensorData::F64(_) => BaseType::F64,
            TensorData::F16(_) => BaseType::F16,
            TensorData::BF16(_) => BaseType::BF16,
        }
    }

    /// 返回数据的 ndarray 形状（与 dtype 无关）。
    pub fn shape(&self) -> &[usize] {
        match self {
            TensorData::F32(a) => a.shape(),
            TensorData::F64(a) => a.shape(),
            TensorData::F16(a) => a.shape(),
            TensorData::BF16(a) => a.shape(),
        }
    }

    /// 元素总数。
    pub fn len(&self) -> usize {
        match self {
            TensorData::F32(a) => a.len(),
            TensorData::F64(a) => a.len(),
            TensorData::F16(a) => a.len(),
            TensorData::BF16(a) => a.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn ndim(&self) -> usize {
        match self {
            TensorData::F32(a) => a.ndim(),
            TensorData::F64(a) => a.ndim(),
            TensorData::F16(a) => a.ndim(),
            TensorData::BF16(a) => a.ndim(),
        }
    }

    /// 以 f64 视图访问（用于通用打印、与现有 f64 代码兼容）。
    /// 若实际为 F32/F16/BF16，则逐元素 cast 为 f64（损失精度但保证可用）。
    pub fn as_f64_view(&self) -> ArrayD<f64> {
        match self {
            TensorData::F64(a) => a.clone(),
            TensorData::F32(a) => a.mapv(|v| v as f64),
            TensorData::F16(a) => a.mapv(|v| v.to_f64()),
            TensorData::BF16(a) => a.mapv(|v| v.to_f64()),
        }
    }

    /// 以 f64 切片访问（若底层不是连续 f64，返回 None）。
    /// 仅当 F64 且内存连续时返回切片；否则返回 None（不做 layout 转换以避免临时值）。
    pub fn as_f64_slice(&self) -> Option<&[f64]> {
        match self {
            TensorData::F64(a) => a.as_slice(),
            _ => None,
        }
    }

    /// 以 f32 切片访问（若底层不是连续 f32，返回 None）。
    /// 仅当 F32 且内存连续时返回切片；否则返回 None。
    pub fn as_f32_slice(&self) -> Option<&[f32]> {
        match self {
            TensorData::F32(a) => a.as_slice(),
            _ => None,
        }
    }

    /// 以 f16 切片访问（仅 F16 且连续时返回切片）。
    pub fn as_f16_slice(&self) -> Option<&[f16]> {
        match self {
            TensorData::F16(a) => a.as_slice(),
            _ => None,
        }
    }

    /// 以 bf16 切片访问（仅 BF16 且连续时返回切片）。
    pub fn as_bf16_slice(&self) -> Option<&[bf16]> {
        match self {
            TensorData::BF16(a) => a.as_slice(),
            _ => None,
        }
    }

    /// 获取 f32 引用（dtype 不匹配返回 None）。
    pub fn as_f32(&self) -> Option<&ArrayD<f32>> {
        match self {
            TensorData::F32(a) => Some(a),
            _ => None,
        }
    }

    /// 获取 f64 引用（dtype 不匹配返回 None）。
    pub fn as_f64(&self) -> Option<&ArrayD<f64>> {
        match self {
            TensorData::F64(a) => Some(a),
            _ => None,
        }
    }

    /// 获取 f16 引用（dtype 不匹配返回 None）。
    pub fn as_f16(&self) -> Option<&ArrayD<f16>> {
        match self {
            TensorData::F16(a) => Some(a),
            _ => None,
        }
    }

    /// 获取 bf16 引用（dtype 不匹配返回 None）。
    pub fn as_bf16(&self) -> Option<&ArrayD<bf16>> {
        match self {
            TensorData::BF16(a) => Some(a),
            _ => None,
        }
    }

    // ── ArrayD<f64> 兼容方法（F32/F16/BF16 自动 cast 为 f64，让外部代码零改动）──
    // 这些方法模拟 ArrayD<f64> 接口，F32/F16/BF16 路径自动 cast 为 f64 视图。
    // 语义降级：F32/F16/BF16 张量经过这些方法后变为 f64，但 Phase 1 目标是隔离爆破半径，
    // 真正的 f32/f16/bf16 路径在 Phase 3/4 改造外部代码后恢复。

    /// 模拟 ArrayD<f64>::mapv。F32/F16/BF16 时先 cast 为 f64 再应用 f。
    /// 使用 FnMut 以匹配 ndarray::mapv 的签名（允许闭包内部可变借用）。
    pub fn mapv<U, F: FnMut(f64) -> U>(&self, mut f: F) -> ArrayD<U> {
        match self {
            TensorData::F64(a) => a.mapv(f),
            TensorData::F32(a) => a.mapv(|v| f(v as f64)),
            TensorData::F16(a) => a.mapv(|v| f(v.to_f64())),
            TensorData::BF16(a) => a.mapv(|v| f(v.to_f64())),
        }
    }

    /// 模拟 ArrayD<f64>::view，返回 owned ArrayD<f64>（F32/F16/BF16 cast）。
    /// 注意：返回 owned 而非借用 view，但接口名兼容，让 .view().insert_axis() 等链式调用工作。
    pub fn view(&self) -> ArrayD<f64> {
        self.as_f64_view()
    }

    /// 模拟 ArrayD<f64>::as_standard_layout，返回 owned ArrayD<f64>。
    pub fn as_standard_layout(&self) -> ArrayD<f64> {
        match self {
            TensorData::F64(a) => a.as_standard_layout().to_owned(),
            TensorData::F32(a) => a.mapv(|v| v as f64).as_standard_layout().to_owned(),
            TensorData::F16(a) => a.mapv(|v| v.to_f64()).as_standard_layout().to_owned(),
            TensorData::BF16(a) => a.mapv(|v| v.to_f64()).as_standard_layout().to_owned(),
        }
    }

    /// 迭代 f64 值（F32/F16/BF16 cast）。替代 .iter().cloned() 模式。
    /// 注意：返回 Box<dyn Iterator<Item=f64>>，调用方不需要 .cloned()。
    pub fn iter(&self) -> Box<dyn Iterator<Item = f64> + '_> {
        match self {
            TensorData::F64(a) => Box::new(a.iter().copied()),
            TensorData::F32(a) => Box::new(a.iter().map(|v| *v as f64)),
            TensorData::F16(a) => Box::new(a.iter().map(|v| v.to_f64())),
            TensorData::BF16(a) => Box::new(a.iter().map(|v| v.to_f64())),
        }
    }

    /// 模拟 ArrayD<f64>::as_slice，仅 F64 时返回切片；其他返回 None。
    /// （已在上方定义 as_f64_slice，此处不重复）

    /// 模拟 ArrayD<f64>::broadcast，返回 owned ArrayD<f64>（broadcast 后 to_owned）。
    pub fn broadcast(&self, shape: &[usize]) -> Option<ArrayD<f64>> {
        let view = self.as_f64_view();
        view.broadcast(IxDyn(shape)).map(|v| v.to_owned())
    }
}
