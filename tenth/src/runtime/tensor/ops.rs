// ── TensorData 算术 trait impl（让外部代码 &x.data * &y.data 等表达式自动工作）──
// F32/F16/BF16 路径自动 cast 为 f64 视图后参与运算，结果为 ArrayD<f64>。
// 这是 Phase 1 兼容层：F32/F16/BF16 张量在外部算术中表现为 f64，dtype 信息在此层丢失。
// Phase 3/4 改造外部代码使用真正的 f32/f16/bf16 路径后，这些 trait impl 可移除。

use super::TensorData;
use ndarray::ArrayD;
use std::ops::{Add, Div, Mul, Neg, Sub};

impl<'a, 'b> Mul<&'b TensorData> for &'a TensorData {
    type Output = ArrayD<f64>;
    fn mul(self, rhs: &'b TensorData) -> ArrayD<f64> {
        &self.as_f64_view() * &rhs.as_f64_view()
    }
}

impl<'a, 'b> Mul<&'b TensorData> for &'a ArrayD<f64> {
    type Output = ArrayD<f64>;
    fn mul(self, rhs: &'b TensorData) -> ArrayD<f64> {
        self * &rhs.as_f64_view()
    }
}

impl Mul<&TensorData> for ArrayD<f64> {
    type Output = ArrayD<f64>;
    fn mul(self, rhs: &TensorData) -> ArrayD<f64> {
        self * &rhs.as_f64_view()
    }
}

impl<'a, 'b> Mul<&'b ArrayD<f64>> for &'a TensorData {
    type Output = ArrayD<f64>;
    fn mul(self, rhs: &'b ArrayD<f64>) -> ArrayD<f64> {
        &self.as_f64_view() * rhs
    }
}

impl<'a, 'b> Add<&'b TensorData> for &'a TensorData {
    type Output = ArrayD<f64>;
    fn add(self, rhs: &'b TensorData) -> ArrayD<f64> {
        &self.as_f64_view() + &rhs.as_f64_view()
    }
}

impl<'a, 'b> Add<&'b TensorData> for &'a ArrayD<f64> {
    type Output = ArrayD<f64>;
    fn add(self, rhs: &'b TensorData) -> ArrayD<f64> {
        self + &rhs.as_f64_view()
    }
}

impl Add<&TensorData> for ArrayD<f64> {
    type Output = ArrayD<f64>;
    fn add(self, rhs: &TensorData) -> ArrayD<f64> {
        self + &rhs.as_f64_view()
    }
}

impl<'a, 'b> Add<&'b ArrayD<f64>> for &'a TensorData {
    type Output = ArrayD<f64>;
    fn add(self, rhs: &'b ArrayD<f64>) -> ArrayD<f64> {
        &self.as_f64_view() + rhs
    }
}

impl<'a, 'b> Sub<&'b TensorData> for &'a TensorData {
    type Output = ArrayD<f64>;
    fn sub(self, rhs: &'b TensorData) -> ArrayD<f64> {
        &self.as_f64_view() - &rhs.as_f64_view()
    }
}

impl<'a, 'b> Sub<&'b TensorData> for &'a ArrayD<f64> {
    type Output = ArrayD<f64>;
    fn sub(self, rhs: &'b TensorData) -> ArrayD<f64> {
        self - &rhs.as_f64_view()
    }
}

impl Sub<&TensorData> for ArrayD<f64> {
    type Output = ArrayD<f64>;
    fn sub(self, rhs: &TensorData) -> ArrayD<f64> {
        self - &rhs.as_f64_view()
    }
}

impl<'a, 'b> Sub<&'b ArrayD<f64>> for &'a TensorData {
    type Output = ArrayD<f64>;
    fn sub(self, rhs: &'b ArrayD<f64>) -> ArrayD<f64> {
        &self.as_f64_view() - rhs
    }
}

impl<'a, 'b> Div<&'b TensorData> for &'a TensorData {
    type Output = ArrayD<f64>;
    fn div(self, rhs: &'b TensorData) -> ArrayD<f64> {
        &self.as_f64_view() / &rhs.as_f64_view()
    }
}

impl<'a, 'b> Div<&'b TensorData> for &'a ArrayD<f64> {
    type Output = ArrayD<f64>;
    fn div(self, rhs: &'b TensorData) -> ArrayD<f64> {
        self / &rhs.as_f64_view()
    }
}

impl Div<&TensorData> for ArrayD<f64> {
    type Output = ArrayD<f64>;
    fn div(self, rhs: &TensorData) -> ArrayD<f64> {
        self / &rhs.as_f64_view()
    }
}

impl<'a, 'b> Div<&'b ArrayD<f64>> for &'a TensorData {
    type Output = ArrayD<f64>;
    fn div(self, rhs: &'b ArrayD<f64>) -> ArrayD<f64> {
        &self.as_f64_view() / rhs
    }
}

impl<'a> Neg for &'a TensorData {
    type Output = ArrayD<f64>;
    fn neg(self) -> ArrayD<f64> {
        -&self.as_f64_view()
    }
}

// ── 标量算术（&TensorData op f64）──
// 用于 autodiff 等外部代码中 &tensor.data * scalar_f64 模式。
// F32 自动 cast 为 f64 视图后运算，结果为 f64（Phase 1 降级策略）。

impl<'a> Mul<f64> for &'a TensorData {
    type Output = ArrayD<f64>;
    fn mul(self, rhs: f64) -> ArrayD<f64> {
        &self.as_f64_view() * rhs
    }
}

impl<'a> Add<f64> for &'a TensorData {
    type Output = ArrayD<f64>;
    fn add(self, rhs: f64) -> ArrayD<f64> {
        &self.as_f64_view() + rhs
    }
}

impl<'a> Sub<f64> for &'a TensorData {
    type Output = ArrayD<f64>;
    fn sub(self, rhs: f64) -> ArrayD<f64> {
        &self.as_f64_view() - rhs
    }
}

impl<'a> Div<f64> for &'a TensorData {
    type Output = ArrayD<f64>;
    fn div(self, rhs: f64) -> ArrayD<f64> {
        &self.as_f64_view() / rhs
    }
}
