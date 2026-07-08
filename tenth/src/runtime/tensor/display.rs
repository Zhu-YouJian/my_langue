use super::{Tensor, TensorData};
use ndarray::ArrayViewD;
use std::fmt;

/// 格式化 f64，确保整数值显示 `.0` 后缀（如 `2.0` 而非 `2`）。
fn format_tensor_f64(n: f64) -> String {
    let s = format!("{}", n);
    if n.is_finite() && !s.contains('.') && !s.contains('e') {
        format!("{}.0", s)
    } else {
        s
    }
}

/// 递归格式化 ArrayViewD<f64>，确保每个元素都显示 `.0` 后缀（如 `[[1.0, 2.0]]`）。
fn format_array_f64(f: &mut fmt::Formatter, a: ArrayViewD<f64>) -> fmt::Result {
    let shape = a.shape();
    if shape.is_empty() {
        // 0D 标量张量
        let v = a.iter().next().copied().unwrap_or(0.0);
        return write!(f, "{}", format_tensor_f64(v));
    }
    if shape.len() == 1 {
        write!(f, "[")?;
        for (i, v) in a.iter().enumerate() {
            if i > 0 { write!(f, ", ")?; }
            write!(f, "{}", format_tensor_f64(*v))?;
        }
        return write!(f, "]");
    }
    // N-D：递归格式化外层每一片
    write!(f, "[")?;
    for (i, sub) in a.outer_iter().enumerate() {
        if i > 0 { write!(f, ", ")?; }
        format_array_f64(f, sub.into_dyn())?;
    }
    write!(f, "]")
}

impl fmt::Display for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.data {
            TensorData::F64(a) => format_array_f64(f, a.view()),
            TensorData::F32(a) => format_array_f64(f, a.mapv(|v| v as f64).view()),
            // F16/BF16 cast 为 f64 视图后打印（保留精度信息）
            TensorData::F16(a) => format_array_f64(f, a.mapv(|v| v.to_f64()).view()),
            TensorData::BF16(a) => format_array_f64(f, a.mapv(|v| v.to_f64()).view()),
        }
    }
}

impl fmt::Display for TensorData {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TensorData::F64(a) => {
                // 简洁打印：形状 + 前/后少量元素
                write!(f, "f64{:?}", a.shape())?;
                if a.len() <= 8 {
                    write!(f, "[")?;
                    for (i, v) in a.iter().copied().enumerate() {
                        if i > 0 { write!(f, ", ")?; }
                        write!(f, "{}", v)?;
                    }
                    write!(f, "]")
                } else {
                    write!(f, "[{}, {}, ..., {}, {}]",
                        a.first().copied().unwrap_or(0.0),
                        a.iter().copied().nth(1).unwrap_or(0.0),
                        a.iter().copied().nth(a.len().saturating_sub(2)).unwrap_or(0.0),
                        a.last().copied().unwrap_or(0.0))
                }
            }
            TensorData::F32(a) => {
                write!(f, "f32{:?}", a.shape())?;
                if a.len() <= 8 {
                    write!(f, "[")?;
                    for (i, v) in a.iter().copied().enumerate() {
                        if i > 0 { write!(f, ", ")?; }
                        write!(f, "{}", v)?;
                    }
                    write!(f, "]")
                } else {
                    write!(f, "[{}, {}, ..., {}, {}]",
                        a.first().copied().unwrap_or(0.0),
                        a.iter().copied().nth(1).unwrap_or(0.0),
                        a.iter().copied().nth(a.len().saturating_sub(2)).unwrap_or(0.0),
                        a.last().copied().unwrap_or(0.0))
                }
            }
            TensorData::F16(a) => {
                write!(f, "f16{:?}", a.shape())?;
                if a.len() <= 8 {
                    write!(f, "[")?;
                    for (i, v) in a.iter().enumerate() {
                        if i > 0 { write!(f, ", ")?; }
                        write!(f, "{}", v.to_f64())?;
                    }
                    write!(f, "]")
                } else {
                    write!(f, "[{}, {}, ..., {}, {}]",
                        a.first().map(|v| v.to_f64()).unwrap_or(0.0),
                        a.iter().map(|v| v.to_f64()).nth(1).unwrap_or(0.0),
                        a.iter().map(|v| v.to_f64()).nth(a.len().saturating_sub(2)).unwrap_or(0.0),
                        a.last().map(|v| v.to_f64()).unwrap_or(0.0))
                }
            }
            TensorData::BF16(a) => {
                write!(f, "bf16{:?}", a.shape())?;
                if a.len() <= 8 {
                    write!(f, "[")?;
                    for (i, v) in a.iter().enumerate() {
                        if i > 0 { write!(f, ", ")?; }
                        write!(f, "{}", v.to_f64())?;
                    }
                    write!(f, "]")
                } else {
                    write!(f, "[{}, {}, ..., {}, {}]",
                        a.first().map(|v| v.to_f64()).unwrap_or(0.0),
                        a.iter().map(|v| v.to_f64()).nth(1).unwrap_or(0.0),
                        a.iter().map(|v| v.to_f64()).nth(a.len().saturating_sub(2)).unwrap_or(0.0),
                        a.last().map(|v| v.to_f64()).unwrap_or(0.0))
                }
            }
        }
    }
}
