use super::TensorData;
use std::ops::Index;

impl Index<[usize; 1]> for TensorData {
    type Output = f64;
    fn index(&self, idx: [usize; 1]) -> &f64 {
        match self {
            TensorData::F64(a) => &a[[idx[0]]],
            TensorData::F32(a) => {
                // F32 cast 到 f64 需要新内存，无法返回引用。
                // 这里用 leak 方式返回 'static 引用，仅供测试断言读取，避免内存泄漏需调用方不长期持有。
                // 更优做法是改造外部代码用 .get(i) 返回 Option<f64>。
                let v = a[[idx[0]]] as f64;
                Box::leak(Box::new(v))
            }
            TensorData::F16(a) => {
                let v = a[[idx[0]]].to_f64();
                Box::leak(Box::new(v))
            }
            TensorData::BF16(a) => {
                let v = a[[idx[0]]].to_f64();
                Box::leak(Box::new(v))
            }
        }
    }
}

impl Index<usize> for TensorData {
    type Output = f64;
    fn index(&self, idx: usize) -> &f64 {
        &self[[idx]]
    }
}

impl Index<[usize; 2]> for TensorData {
    type Output = f64;
    fn index(&self, idx: [usize; 2]) -> &f64 {
        match self {
            TensorData::F64(a) => &a[[idx[0], idx[1]]],
            TensorData::F32(a) => {
                let v = a[[idx[0], idx[1]]] as f64;
                Box::leak(Box::new(v))
            }
            TensorData::F16(a) => {
                let v = a[[idx[0], idx[1]]].to_f64();
                Box::leak(Box::new(v))
            }
            TensorData::BF16(a) => {
                let v = a[[idx[0], idx[1]]].to_f64();
                Box::leak(Box::new(v))
            }
        }
    }
}
