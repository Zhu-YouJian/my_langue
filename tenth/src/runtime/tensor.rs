use crate::hir::types::BaseType;
use ndarray::{ArrayD, IxDyn};
use std::fmt;

/// 双精度张量数据存储。f32 与 f64 各占一个变体，避免 f32 退化为「语法糖 f64」。
#[derive(Debug, Clone)]
pub enum TensorData {
    F32(ArrayD<f32>),
    F64(ArrayD<f64>),
}

impl TensorData {
    /// 返回数据元素 dtype。
    pub fn dtype(&self) -> BaseType {
        match self {
            TensorData::F32(_) => BaseType::F32,
            TensorData::F64(_) => BaseType::F64,
        }
    }

    /// 返回数据的 ndarray 形状（与 dtype 无关）。
    pub fn shape(&self) -> &[usize] {
        match self {
            TensorData::F32(a) => a.shape(),
            TensorData::F64(a) => a.shape(),
        }
    }

    /// 元素总数。
    pub fn len(&self) -> usize {
        match self {
            TensorData::F32(a) => a.len(),
            TensorData::F64(a) => a.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn ndim(&self) -> usize {
        match self {
            TensorData::F32(a) => a.ndim(),
            TensorData::F64(a) => a.ndim(),
        }
    }

    /// 以 f64 视图访问（用于通用打印、与现有 f64 代码兼容）。
    /// 若实际为 F32，则逐元素 cast 为 f64（损失精度但保证可用）。
    pub fn as_f64_view(&self) -> ArrayD<f64> {
        match self {
            TensorData::F64(a) => a.clone(),
            TensorData::F32(a) => a.mapv(|v| v as f64),
        }
    }

    /// 以 f64 切片访问（若底层不是连续 f64，返回 None）。
    /// 仅当 F64 且内存连续时返回切片；否则返回 None（不做 layout 转换以避免临时值）。
    pub fn as_f64_slice(&self) -> Option<&[f64]> {
        match self {
            TensorData::F64(a) => a.as_slice(),
            TensorData::F32(_) => None,
        }
    }

    /// 以 f32 切片访问（若底层不是连续 f32，返回 None）。
    /// 仅当 F32 且内存连续时返回切片；否则返回 None。
    pub fn as_f32_slice(&self) -> Option<&[f32]> {
        match self {
            TensorData::F32(a) => a.as_slice(),
            TensorData::F64(_) => None,
        }
    }

    /// 获取 f32 引用（dtype 不匹配返回 None）。
    pub fn as_f32(&self) -> Option<&ArrayD<f32>> {
        match self {
            TensorData::F32(a) => Some(a),
            TensorData::F64(_) => None,
        }
    }

    /// 获取 f64 引用（dtype 不匹配返回 None）。
    pub fn as_f64(&self) -> Option<&ArrayD<f64>> {
        match self {
            TensorData::F64(a) => Some(a),
            TensorData::F32(_) => None,
        }
    }

    // ── ArrayD<f64> 兼容方法（F32 自动 cast 为 f64，让外部代码零改动）──
    // 这些方法模拟 ArrayD<f64> 接口，F32 路径自动 cast 为 f64 视图。
    // 语义降级：F32 张量经过这些方法后变为 f64，但 Phase 1 目标是隔离爆破半径，
    // 真正的 f32 路径在 Phase 3/4 改造外部代码后恢复。

    /// 模拟 ArrayD<f64>::mapv。F32 时先 cast 为 f64 再应用 f。
    /// 使用 FnMut 以匹配 ndarray::mapv 的签名（允许闭包内部可变借用）。
    pub fn mapv<U, F: FnMut(f64) -> U>(&self, mut f: F) -> ArrayD<U> {
        match self {
            TensorData::F64(a) => a.mapv(f),
            TensorData::F32(a) => a.mapv(|v| f(v as f64)),
        }
    }

    /// 模拟 ArrayD<f64>::view，返回 owned ArrayD<f64>（F32 cast）。
    /// 注意：返回 owned 而非借用 view，但接口名兼容，让 .view().insert_axis() 等链式调用工作。
    pub fn view(&self) -> ArrayD<f64> {
        self.as_f64_view()
    }

    /// 模拟 ArrayD<f64>::as_standard_layout，返回 owned ArrayD<f64>。
    pub fn as_standard_layout(&self) -> ArrayD<f64> {
        match self {
            TensorData::F64(a) => a.as_standard_layout().to_owned(),
            TensorData::F32(a) => a.mapv(|v| v as f64).as_standard_layout().to_owned(),
        }
    }

    /// 迭代 f64 值（F32 cast）。替代 .iter().cloned() 模式。
    /// 注意：返回 Box<dyn Iterator<Item=f64>>，调用方不需要 .cloned()。
    pub fn iter(&self) -> Box<dyn Iterator<Item = f64> + '_> {
        match self {
            TensorData::F64(a) => Box::new(a.iter().copied()),
            TensorData::F32(a) => Box::new(a.iter().map(|v| *v as f64)),
        }
    }

    /// 模拟 ArrayD<f64>::as_slice，仅 F64 时返回切片；F32 返回 None。
    /// （已在上方定义 as_f64_slice，此处不重复）

    /// 模拟 ArrayD<f64>::broadcast，返回 owned ArrayD<f64>（broadcast 后 to_owned）。
    pub fn broadcast(&self, shape: &[usize]) -> Option<ArrayD<f64>> {
        let view = self.as_f64_view();
        view.broadcast(IxDyn(shape)).map(|v| v.to_owned())
    }
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

impl Tensor {
    // ── helpers ────────────────────────────────────────────────────────

    /// 从 f64 数据构造 F64 Tensor（内部便利构造器，保持向后兼容）。
    pub(crate) fn from_data(data: ArrayD<f64>) -> Self {
        Tensor {
            dtype: BaseType::F64,
            data: TensorData::F64(data),
            grad: None,
            tape_id: None,
        }
    }

    /// 从 f32 数据构造 F32 Tensor。
    pub(crate) fn from_data_f32(data: ArrayD<f32>) -> Self {
        Tensor {
            dtype: BaseType::F32,
            data: TensorData::F32(data),
            grad: None,
            tape_id: None,
        }
    }

    /// 从指定 dtype 的 TensorData 构造 Tensor（dtype 自动从 data 推断）。
    pub fn from_tensor_data(data: TensorData) -> Self {
        let dtype = data.dtype();
        Tensor {
            dtype,
            data,
            grad: None,
            tape_id: None,
        }
    }

    /// 返回 Tensor 元素 dtype。
    pub fn dtype(&self) -> BaseType {
        self.dtype
    }

    /// 是否为 f32 张量。
    pub fn is_f32(&self) -> bool {
        matches!(self.dtype, BaseType::F32)
    }

    /// 是否为 f64 张量。
    pub fn is_f64(&self) -> bool {
        matches!(self.dtype, BaseType::F64)
    }

    /// 获取 f64 数据引用（dtype 不匹配返回 None）。
    pub fn data_f64(&self) -> Option<&ArrayD<f64>> {
        self.data.as_f64()
    }

    /// 获取 f32 数据引用（dtype 不匹配返回 None）。
    pub fn data_f32(&self) -> Option<&ArrayD<f32>> {
        self.data.as_f32()
    }

    /// 以 f64 视图获取数据（若为 f32 则 cast，损失精度但保证可用）。
    /// 用于打印、通用比较等不关心精度的场景。
    pub fn data_as_f64_view(&self) -> ArrayD<f64> {
        self.data.as_f64_view()
    }

    /// Zero-initialise the gradient buffer to match the tensor shape.
    pub fn zero_grad(&mut self) {
        self.grad = None;
    }

    /// Accumulate `g` into `self.grad`.
    /// g 的 dtype 必须与 self.dtype 一致；若 self.grad 为 None 则按 dtype 初始化。
    /// 返回 Err 当 g.shape() 与 self.data.shape() 不一致（防止 silent broadcast 掩盖梯度 shape 错误）。
    pub fn acc_grad(&mut self, g: &ArrayD<f64>) -> Result<(), String> {
        // shape 校验：梯度 shape 必须与参数 shape 一致（方向 A：消除 silent squeeze）
        let self_shape = self.data.shape();
        let g_shape = g.shape();
        if self_shape != g_shape {
            return Err(format!(
                "acc_grad shape 不匹配：参数 shape {:?}，梯度 shape {:?}（可能反向传播 silent squeeze 掩盖了 shape 错误）",
                self_shape, g_shape
            ));
        }
        // 保持现有签名兼容：g 视为 f64，按 self.dtype 转换存储
        let g_converted = match self.dtype {
            BaseType::F32 => TensorData::F32(g.mapv(|v| v as f32)),
            BaseType::F64 => TensorData::F64(g.clone()),
            _ => TensorData::F64(g.clone()),
        };
        match &mut self.grad {
            Some(TensorData::F64(cur)) => {
                if let TensorData::F64(g2) = &g_converted {
                    *cur = &*cur + g2;
                } else {
                    // dtype 不一致，回退为 f64
                    let merged = cur.clone() + g;
                    self.grad = Some(TensorData::F64(merged));
                }
            }
            Some(TensorData::F32(cur)) => {
                if let TensorData::F32(g2) = &g_converted {
                    *cur = &*cur + g2;
                } else {
                    let merged = cur.mapv(|v| v as f64) + g;
                    self.grad = Some(TensorData::F64(merged));
                }
            }
            None => {
                self.grad = Some(g_converted);
            }
        }
        Ok(())
    }

    // ── constructors (f64, 保持向后兼容) ──────────────────────────────

    pub fn from_vec(data: Vec<f64>, shape: Vec<usize>) -> Self {
        let array = ArrayD::from_shape_vec(IxDyn(&shape), data)
            .expect("invalid tensor shape");
        Tensor::from_data(array)
    }

    pub fn zeros(shape: &[usize]) -> Self {
        Tensor::from_data(ArrayD::zeros(IxDyn(shape)))
    }

    pub fn ones(shape: &[usize]) -> Self {
        Tensor::from_data(ArrayD::ones(IxDyn(shape)))
    }

    pub fn rand(shape: &[usize]) -> Self {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let size: usize = shape.iter().product();
        let data: Vec<f64> = (0..size).map(|_| rng.r#gen::<f64>()).collect();
        Tensor::from_vec(data, shape.to_vec())
    }

    pub fn randn(shape: &[usize]) -> Self {
        use rand_distr::{Normal, Distribution};
        let mut rng = rand::thread_rng();
        let normal = Normal::new(0.0, 1.0).unwrap();
        let size: usize = shape.iter().product();
        let data: Vec<f64> = (0..size).map(|_| normal.sample(&mut rng)).collect();
        Tensor::from_vec(data, shape.to_vec())
    }

    pub fn full(shape: &[usize], value: f64) -> Self {
        Tensor::from_data(ArrayD::from_elem(IxDyn(shape), value))
    }

    pub fn eye(n: usize) -> Self {
        let mut array = ArrayD::zeros(IxDyn(&[n, n]));
        for i in 0..n {
            array[[i, i]] = 1.0;
        }
        Tensor::from_data(array)
    }

    pub fn arange(start: f64, end: f64, step: f64) -> Self {
        let mut data = Vec::new();
        let mut x = start;
        while x < end {
            data.push(x);
            x += step;
        }
        let len = data.len();
        Tensor::from_vec(data, vec![len])
    }

    // ── constructors (f32) ───────────────────────────────────────────

    pub fn from_vec_f32(data: Vec<f32>, shape: Vec<usize>) -> Self {
        let array = ArrayD::from_shape_vec(IxDyn(&shape), data)
            .expect("invalid tensor shape");
        Tensor::from_data_f32(array)
    }

    pub fn zeros_f32(shape: &[usize]) -> Self {
        Tensor::from_data_f32(ArrayD::zeros(IxDyn(shape)))
    }

    pub fn ones_f32(shape: &[usize]) -> Self {
        Tensor::from_data_f32(ArrayD::ones(IxDyn(shape)))
    }

    pub fn full_f32(shape: &[usize], value: f32) -> Self {
        Tensor::from_data_f32(ArrayD::from_elem(IxDyn(shape), value))
    }

    pub fn rand_f32(shape: &[usize]) -> Self {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let size: usize = shape.iter().product();
        let data: Vec<f32> = (0..size).map(|_| rng.r#gen::<f32>()).collect();
        Tensor::from_vec_f32(data, shape.to_vec())
    }

    pub fn randn_f32(shape: &[usize]) -> Self {
        use rand_distr::{Normal, Distribution};
        let mut rng = rand::thread_rng();
        let normal = Normal::new(0.0f32, 1.0f32).unwrap();
        let size: usize = shape.iter().product();
        let data: Vec<f32> = (0..size).map(|_| normal.sample(&mut rng)).collect();
        Tensor::from_vec_f32(data, shape.to_vec())
    }

    pub fn eye_f32(n: usize) -> Self {
        let mut array = ArrayD::zeros(IxDyn(&[n, n]));
        for i in 0..n {
            array[[i, i]] = 1.0;
        }
        Tensor::from_data_f32(array)
    }

    pub fn arange_f32(start: f32, end: f32, step: f32) -> Self {
        let mut data = Vec::new();
        let mut x = start;
        while x < end {
            data.push(x);
            x += step;
        }
        let len = data.len();
        Tensor::from_vec_f32(data, vec![len])
    }

    // ── constructors (通用 dtype) ─────────────────────────────────────

    /// 按指定 dtype 构造全零张量。当前支持 F32/F64，其他 dtype 返回 F64 兜底。
    pub fn zeros_with_dtype(shape: &[usize], dtype: BaseType) -> Self {
        match dtype {
            BaseType::F32 => Tensor::zeros_f32(shape),
            _ => Tensor::zeros(shape),
        }
    }

    /// 按指定 dtype 构造全一张量。
    pub fn ones_with_dtype(shape: &[usize], dtype: BaseType) -> Self {
        match dtype {
            BaseType::F32 => Tensor::ones_f32(shape),
            _ => Tensor::ones(shape),
        }
    }

    // ── shape / access ─────────────────────────────────────────────────

    pub fn shape(&self) -> Vec<usize> {
        self.data.shape().to_vec()
    }

    pub fn ndim(&self) -> usize {
        self.data.ndim()
    }

    pub fn size(&self) -> usize {
        self.data.len()
    }

    pub fn get(&self, index: &[usize]) -> Option<f64> {
        match &self.data {
            TensorData::F64(a) => a.get(IxDyn(index)).copied(),
            TensorData::F32(a) => a.get(IxDyn(index)).map(|v| *v as f64),
        }
    }

    // ── 内部 f32/f64 分支辅助 ─────────────────────────────────────────

    /// 对每个元素应用 f32 → f32 映射（仅 F32 路径）。
    fn map_f32(&self, f: impl Fn(f32) -> f32) -> Tensor {
        let a = self.data.as_f32().expect("map_f32 requires F32 tensor");
        Tensor::from_data_f32(a.mapv(f))
    }

    /// 对每个元素应用 f64 → f64 映射（仅 F64 路径）。
    fn map_f64(&self, f: impl Fn(f64) -> f64) -> Tensor {
        let a = self.data.as_f64().expect("map_f64 requires F64 tensor");
        Tensor::from_data(a.mapv(f))
    }

    /// 对两个同 dtype 张量做元素级 f32 运算（仅 F32 路径，需广播）。
    fn zip_f32(&self, other: &Tensor, f: impl Fn(f32, f32) -> f32) -> Result<Tensor, String> {
        let a = self.data.as_f32().expect("zip_f32 self requires F32");
        let b = other.data.as_f32().expect("zip_f32 other requires F32");
        let out_shape = Self::broadcast_shape(a.shape(), b.shape())
            .ok_or_else(|| format!("cannot broadcast shapes {:?} + {:?}", self.shape(), other.shape()))?;
        let a_br = a.broadcast(IxDyn(&out_shape)).unwrap();
        let b_br = b.broadcast(IxDyn(&out_shape)).unwrap();
        let mut out = ArrayD::zeros(IxDyn(&out_shape));
        out.zip_mut_with(&a_br.to_owned(), |o, &x| { *o = x; });
        let b_owned = b_br.to_owned();
        out.zip_mut_with(&b_owned, |o, &y| { *o = f(*o, y); });
        Ok(Tensor::from_data_f32(out))
    }

    /// 对两个同 dtype 张量做元素级 f64 运算（仅 F64 路径，需广播）。
    fn zip_f64(&self, other: &Tensor, f: impl Fn(f64, f64) -> f64) -> Result<Tensor, String> {
        let a = self.data.as_f64().expect("zip_f64 self requires F64");
        let b = other.data.as_f64().expect("zip_f64 other requires F64");
        let out_shape = Self::broadcast_shape(a.shape(), b.shape())
            .ok_or_else(|| format!("cannot broadcast shapes {:?} + {:?}", self.shape(), other.shape()))?;
        let a_br = a.broadcast(IxDyn(&out_shape)).unwrap();
        let b_br = b.broadcast(IxDyn(&out_shape)).unwrap();
        let mut out = ArrayD::zeros(IxDyn(&out_shape));
        out.zip_mut_with(&a_br.to_owned(), |o, &x| { *o = x; });
        let b_owned = b_br.to_owned();
        out.zip_mut_with(&b_owned, |o, &y| { *o = f(*o, y); });
        Ok(Tensor::from_data(out))
    }

    // ── reductions ─────────────────────────────────────────────────────

    pub fn sum(&self) -> f64 {
        match &self.data {
            TensorData::F64(a) => a.sum(),
            TensorData::F32(a) => a.sum() as f64,
        }
    }

    pub fn sum_axis(&self, axis: usize) -> Result<Tensor, String> {
        let ndim = self.ndim();
        if axis >= ndim {
            return Err(format!(
                "sum_axis: axis {} out of bounds for {}-D tensor",
                axis, ndim
            ));
        }
        let result = match &self.data {
            TensorData::F64(a) => Tensor::from_data(a.sum_axis(ndarray::Axis(axis))),
            TensorData::F32(a) => Tensor::from_data_f32(a.sum_axis(ndarray::Axis(axis))),
        };
        Ok(result)
    }

    pub fn mean(&self) -> f64 {
        match &self.data {
            TensorData::F64(a) => a.mean().unwrap_or(0.0),
            TensorData::F32(a) => a.mean().map(|v| v as f64).unwrap_or(0.0),
        }
    }

    pub fn max_val(&self) -> f64 {
        match &self.data {
            TensorData::F64(a) => a.iter().copied().fold(f64::NEG_INFINITY, f64::max),
            TensorData::F32(a) => a.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64,
        }
    }

    /// Return the index of the maximum value (flat index).
    /// Returns -1 for an empty tensor.
    pub fn argmax(&self) -> i64 {
        let iter: Box<dyn Iterator<Item = f64>> = match &self.data {
            TensorData::F64(a) => Box::new(a.iter().copied()),
            TensorData::F32(a) => Box::new(a.iter().map(|v| *v as f64)),
        };
        iter.enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i as i64)
            .unwrap_or(-1)
    }

    // ── scalar ops (按 dtype 分支) ────────────────────────────────────

    pub fn add_scalar(&self, scalar: f64) -> Tensor {
        match &self.data {
            TensorData::F64(a) => Tensor::from_data(a.mapv(|x| x + scalar)),
            TensorData::F32(a) => Tensor::from_data_f32(a.mapv(|x| x + scalar as f32)),
        }
    }

    pub fn sub_scalar(&self, scalar: f64) -> Tensor {
        match &self.data {
            TensorData::F64(a) => Tensor::from_data(a.mapv(|x| x - scalar)),
            TensorData::F32(a) => Tensor::from_data_f32(a.mapv(|x| x - scalar as f32)),
        }
    }

    pub fn mul_scalar(&self, scalar: f64) -> Tensor {
        match &self.data {
            TensorData::F64(a) => Tensor::from_data(a.mapv(|x| x * scalar)),
            TensorData::F32(a) => Tensor::from_data_f32(a.mapv(|x| x * scalar as f32)),
        }
    }

    pub fn div_scalar(&self, scalar: f64) -> Tensor {
        match &self.data {
            TensorData::F64(a) => Tensor::from_data(a.mapv(|x| x / scalar)),
            TensorData::F32(a) => Tensor::from_data_f32(a.mapv(|x| x / scalar as f32)),
        }
    }

    /// Scalar divided by tensor: scalar / self (element-wise).
    pub fn div_scalar_inv(&self, scalar: f64) -> Tensor {
        match &self.data {
            TensorData::F64(a) => Tensor::from_data(a.mapv(|x| scalar / x)),
            TensorData::F32(a) => Tensor::from_data_f32(a.mapv(|x| scalar as f32 / x)),
        }
    }

    // ── tensor-tensor element-wise ops (with broadcasting) ─────────────

    /// Compute the numpy-style broadcast shape of two shapes.
    /// Returns `None` if the shapes are not broadcast-compatible.
    fn broadcast_shape(a: &[usize], b: &[usize]) -> Option<Vec<usize>> {
        let na = a.len();
        let nb = b.len();
        let n = na.max(nb);
        let mut out = vec![0usize; n];
        for i in 0..n {
            let da = if i < na { a[na - 1 - i] } else { 1 };
            let db = if i < nb { b[nb - 1 - i] } else { 1 };
            out[n - 1 - i] = match (da, db) {
                (1, x) | (x, 1) => x,
                (x, y) if x == y => x,
                _ => return None,
            };
        }
        Some(out)
    }

    /// Element-wise addition with broadcasting.  Errors if shapes are incompatible.
    /// dtype 提升规则：f32 + f32 → f32；f64 + f64 → f64；混合 → f64（提升）。
    pub fn add_tensor(&self, other: &Tensor) -> Result<Tensor, String> {
        match (&self.data, &other.data) {
            (TensorData::F64(a), TensorData::F64(b)) => {
                let out_shape = Self::broadcast_shape(a.shape(), b.shape())
                    .ok_or_else(|| format!("cannot broadcast shapes {:?} + {:?}", self.shape(), other.shape()))?;
                let a_br = a.broadcast(IxDyn(&out_shape)).unwrap();
                let b_br = b.broadcast(IxDyn(&out_shape)).unwrap();
                Ok(Tensor::from_data(&a_br.to_owned() + &b_br.to_owned()))
            }
            (TensorData::F32(a), TensorData::F32(b)) => {
                let out_shape = Self::broadcast_shape(a.shape(), b.shape())
                    .ok_or_else(|| format!("cannot broadcast shapes {:?} + {:?}", self.shape(), other.shape()))?;
                let a_br = a.broadcast(IxDyn(&out_shape)).unwrap();
                let b_br = b.broadcast(IxDyn(&out_shape)).unwrap();
                Ok(Tensor::from_data_f32(&a_br.to_owned() + &b_br.to_owned()))
            }
            // 混合 dtype：提升为 f64
            _ => {
                let a = self.data.as_f64_view();
                let b = other.data.as_f64_view();
                let out_shape = Self::broadcast_shape(a.shape(), b.shape())
                    .ok_or_else(|| format!("cannot broadcast shapes {:?} + {:?}", self.shape(), other.shape()))?;
                let a_br = a.broadcast(IxDyn(&out_shape)).unwrap();
                let b_br = b.broadcast(IxDyn(&out_shape)).unwrap();
                Ok(Tensor::from_data(&a_br.to_owned() + &b_br.to_owned()))
            }
        }
    }

    pub fn sub_tensor(&self, other: &Tensor) -> Result<Tensor, String> {
        match (&self.data, &other.data) {
            (TensorData::F64(a), TensorData::F64(b)) => {
                let out_shape = Self::broadcast_shape(a.shape(), b.shape())
                    .ok_or_else(|| format!("cannot broadcast shapes {:?} - {:?}", self.shape(), other.shape()))?;
                let a_br = a.broadcast(IxDyn(&out_shape)).unwrap();
                let b_br = b.broadcast(IxDyn(&out_shape)).unwrap();
                Ok(Tensor::from_data(&a_br.to_owned() - &b_br.to_owned()))
            }
            (TensorData::F32(a), TensorData::F32(b)) => {
                let out_shape = Self::broadcast_shape(a.shape(), b.shape())
                    .ok_or_else(|| format!("cannot broadcast shapes {:?} - {:?}", self.shape(), other.shape()))?;
                let a_br = a.broadcast(IxDyn(&out_shape)).unwrap();
                let b_br = b.broadcast(IxDyn(&out_shape)).unwrap();
                Ok(Tensor::from_data_f32(&a_br.to_owned() - &b_br.to_owned()))
            }
            _ => {
                let a = self.data.as_f64_view();
                let b = other.data.as_f64_view();
                let out_shape = Self::broadcast_shape(a.shape(), b.shape())
                    .ok_or_else(|| format!("cannot broadcast shapes {:?} - {:?}", self.shape(), other.shape()))?;
                let a_br = a.broadcast(IxDyn(&out_shape)).unwrap();
                let b_br = b.broadcast(IxDyn(&out_shape)).unwrap();
                Ok(Tensor::from_data(&a_br.to_owned() - &b_br.to_owned()))
            }
        }
    }

    pub fn mul_tensor(&self, other: &Tensor) -> Result<Tensor, String> {
        match (&self.data, &other.data) {
            (TensorData::F64(a), TensorData::F64(b)) => {
                let out_shape = Self::broadcast_shape(a.shape(), b.shape())
                    .ok_or_else(|| format!("cannot broadcast shapes {:?} * {:?}", self.shape(), other.shape()))?;
                let a_br = a.broadcast(IxDyn(&out_shape)).unwrap();
                let b_br = b.broadcast(IxDyn(&out_shape)).unwrap();
                Ok(Tensor::from_data(&a_br.to_owned() * &b_br.to_owned()))
            }
            (TensorData::F32(a), TensorData::F32(b)) => {
                let out_shape = Self::broadcast_shape(a.shape(), b.shape())
                    .ok_or_else(|| format!("cannot broadcast shapes {:?} * {:?}", self.shape(), other.shape()))?;
                let a_br = a.broadcast(IxDyn(&out_shape)).unwrap();
                let b_br = b.broadcast(IxDyn(&out_shape)).unwrap();
                Ok(Tensor::from_data_f32(&a_br.to_owned() * &b_br.to_owned()))
            }
            _ => {
                let a = self.data.as_f64_view();
                let b = other.data.as_f64_view();
                let out_shape = Self::broadcast_shape(a.shape(), b.shape())
                    .ok_or_else(|| format!("cannot broadcast shapes {:?} * {:?}", self.shape(), other.shape()))?;
                let a_br = a.broadcast(IxDyn(&out_shape)).unwrap();
                let b_br = b.broadcast(IxDyn(&out_shape)).unwrap();
                Ok(Tensor::from_data(&a_br.to_owned() * &b_br.to_owned()))
            }
        }
    }

    pub fn div_tensor(&self, other: &Tensor) -> Result<Tensor, String> {
        match (&self.data, &other.data) {
            (TensorData::F64(a), TensorData::F64(b)) => {
                let out_shape = Self::broadcast_shape(a.shape(), b.shape())
                    .ok_or_else(|| format!("cannot broadcast shapes {:?} / {:?}", self.shape(), other.shape()))?;
                let a_br = a.broadcast(IxDyn(&out_shape)).unwrap();
                let b_br = b.broadcast(IxDyn(&out_shape)).unwrap();
                Ok(Tensor::from_data(&a_br.to_owned() / &b_br.to_owned()))
            }
            (TensorData::F32(a), TensorData::F32(b)) => {
                let out_shape = Self::broadcast_shape(a.shape(), b.shape())
                    .ok_or_else(|| format!("cannot broadcast shapes {:?} / {:?}", self.shape(), other.shape()))?;
                let a_br = a.broadcast(IxDyn(&out_shape)).unwrap();
                let b_br = b.broadcast(IxDyn(&out_shape)).unwrap();
                Ok(Tensor::from_data_f32(&a_br.to_owned() / &b_br.to_owned()))
            }
            _ => {
                let a = self.data.as_f64_view();
                let b = other.data.as_f64_view();
                let out_shape = Self::broadcast_shape(a.shape(), b.shape())
                    .ok_or_else(|| format!("cannot broadcast shapes {:?} / {:?}", self.shape(), other.shape()))?;
                let a_br = a.broadcast(IxDyn(&out_shape)).unwrap();
                let b_br = b.broadcast(IxDyn(&out_shape)).unwrap();
                Ok(Tensor::from_data(&a_br.to_owned() / &b_br.to_owned()))
            }
        }
    }

    // ── matrix multiplication ─────────────────────────────────────────

    /// Matrix multiplication.  Supports 2D @ 2D and 1D @ 2D / 2D @ 1D.
    /// 保留输入 dtype：f32@f32 → f32；f64@f64 → f64；混合 → f64（提升）。
    pub fn matmul(&self, other: &Tensor) -> Result<Tensor, String> {
        let a_ndim = self.ndim();
        let b_ndim = other.ndim();
        let use_f32 = self.is_f32() && other.is_f32();

        if use_f32 {
            self.matmul_f32(other)
        } else {
            // f64 路径（含混合提升：将 f32 视图 cast 为 f64 后走原 f64 逻辑）
            let a_view = self.data.as_f64_view();
            let b_view = other.data.as_f64_view();

            if a_ndim == 2 && b_ndim == 2 {
                let a = a_view.view().into_dimensionality::<ndarray::Ix2>()
                    .map_err(|_| "matmul: self must be 2D")?;
                let b = b_view.view().into_dimensionality::<ndarray::Ix2>()
                    .map_err(|_| "matmul: other must be 2D")?;
                if a.shape()[1] != b.shape()[0] {
                    return Err(format!(
                        "matmul shape mismatch: {:?} @ {:?}",
                        a.shape(), b.shape()
                    ));
                }
                let result = a.dot(&b);
                Ok(Tensor::from_data(result.into_dyn()))
            } else if a_ndim == 1 && b_ndim == 2 {
                let a = a_view.view().into_dimensionality::<ndarray::Ix1>()
                    .map_err(|_| "matmul: self must be 1D")?;
                let b = b_view.view().into_dimensionality::<ndarray::Ix2>()
                    .map_err(|_| "matmul: other must be 2D")?;
                if a.shape()[0] != b.shape()[0] {
                    return Err(format!("matmul shape mismatch: {:?} @ {:?}", a.shape(), b.shape()));
                }
                let a_2d = a.insert_axis(ndarray::Axis(0));
                let result = a_2d.dot(&b);
                let squeezed = result.index_axis_move(ndarray::Axis(0), 0);
                Ok(Tensor::from_data(squeezed.into_dyn()))
            } else if a_ndim == 2 && b_ndim == 1 {
                let a = a_view.view().into_dimensionality::<ndarray::Ix2>()
                    .map_err(|_| "matmul: self must be 2D")?;
                let b = b_view.view().into_dimensionality::<ndarray::Ix1>()
                    .map_err(|_| "matmul: other must be 1D")?;
                if a.shape()[1] != b.shape()[0] {
                    return Err(format!("matmul shape mismatch: {:?} @ {:?}", a.shape(), b.shape()));
                }
                let result = a.dot(&b);
                Ok(Tensor::from_data(result.into_dyn()))
            } else {
                Err(format!("matmul requires 1D/2D tensors, got {:?}D and {:?}D", a_ndim, b_ndim))
            }
        }
    }

    /// f32 专用 matmul 路径，保持 f32 dtype。
    fn matmul_f32(&self, other: &Tensor) -> Result<Tensor, String> {
        let a_ndim = self.ndim();
        let b_ndim = other.ndim();
        let a = self.data.as_f32().expect("matmul_f32: self must be F32");
        let b = other.data.as_f32().expect("matmul_f32: other must be F32");

        if a_ndim == 2 && b_ndim == 2 {
            let a = a.view().into_dimensionality::<ndarray::Ix2>()
                .map_err(|_| "matmul: self must be 2D")?;
            let b = b.view().into_dimensionality::<ndarray::Ix2>()
                .map_err(|_| "matmul: other must be 2D")?;
            if a.shape()[1] != b.shape()[0] {
                return Err(format!(
                    "matmul shape mismatch: {:?} @ {:?}",
                    a.shape(), b.shape()
                ));
            }
            let result = a.dot(&b);
            Ok(Tensor::from_data_f32(result.into_dyn()))
        } else if a_ndim == 1 && b_ndim == 2 {
            let a = a.view().into_dimensionality::<ndarray::Ix1>()
                .map_err(|_| "matmul: self must be 1D")?;
            let b = b.view().into_dimensionality::<ndarray::Ix2>()
                .map_err(|_| "matmul: other must be 2D")?;
            if a.shape()[0] != b.shape()[0] {
                return Err(format!("matmul shape mismatch: {:?} @ {:?}", a.shape(), b.shape()));
            }
            let a_2d = a.insert_axis(ndarray::Axis(0));
            let result = a_2d.dot(&b);
            let squeezed = result.index_axis_move(ndarray::Axis(0), 0);
            Ok(Tensor::from_data_f32(squeezed.into_dyn()))
        } else if a_ndim == 2 && b_ndim == 1 {
            let a = a.view().into_dimensionality::<ndarray::Ix2>()
                .map_err(|_| "matmul: self must be 2D")?;
            let b = b.view().into_dimensionality::<ndarray::Ix1>()
                .map_err(|_| "matmul: other must be 1D")?;
            if a.shape()[1] != b.shape()[0] {
                return Err(format!("matmul shape mismatch: {:?} @ {:?}", a.shape(), b.shape()));
            }
            let result = a.dot(&b);
            Ok(Tensor::from_data_f32(result.into_dyn()))
        } else {
            Err(format!("matmul requires 1D/2D tensors, got {:?}D and {:?}D", a_ndim, b_ndim))
        }
    }

    // ── transpose ─────────────────────────────────────────────────────

    /// Transpose last two dimensions.  For 2D tensors this is the usual matrix transpose.
    pub fn transpose(&self) -> Option<Tensor> {
        if self.ndim() < 2 {
            return None;
        }
        let mut perm: Vec<usize> = (0..self.ndim()).collect();
        let last = perm.len() - 1;
        perm.swap(last - 1, last);
        match &self.data {
            TensorData::F64(a) => {
                let result = a.view().permuted_axes(perm);
                Some(Tensor::from_data(result.to_owned()))
            }
            TensorData::F32(a) => {
                let result = a.view().permuted_axes(perm);
                Some(Tensor::from_data_f32(result.to_owned()))
            }
        }
    }

    // ── broadcasting / reshaping ───────────────────────────────────────

    /// Broadcast to a target shape.  Returns None if not broadcastable.
    pub fn broadcast_to(&self, target_shape: &[usize]) -> Option<Tensor> {
        match &self.data {
            TensorData::F64(a) => {
                let broadcasted = a.broadcast(target_shape)?;
                Some(Tensor::from_data(broadcasted.to_owned()))
            }
            TensorData::F32(a) => {
                let broadcasted = a.broadcast(target_shape)?;
                Some(Tensor::from_data_f32(broadcasted.to_owned()))
            }
        }
    }

    // ── unary elementwise ──────────────────────────────────────────────

    pub fn neg(&self) -> Tensor {
        match &self.data {
            TensorData::F64(a) => Tensor::from_data(a.mapv(|x| -x)),
            TensorData::F32(a) => Tensor::from_data_f32(a.mapv(|x| -x)),
        }
    }

    pub fn abs(&self) -> Tensor {
        match &self.data {
            TensorData::F64(a) => Tensor::from_data(a.mapv(|x| x.abs())),
            TensorData::F32(a) => Tensor::from_data_f32(a.mapv(|x| x.abs())),
        }
    }

    /// 元素级裁剪到 [min_val, max_val]（用于梯度裁剪）。
    pub fn clip_scalar(&self, min_val: f64, max_val: f64) -> Tensor {
        match &self.data {
            TensorData::F64(a) => Tensor::from_data(a.mapv(|x| x.clamp(min_val, max_val))),
            TensorData::F32(a) => Tensor::from_data_f32(a.mapv(|x| (x as f64).clamp(min_val, max_val) as f32)),
        }
    }

    pub fn sqrt(&self) -> Tensor {
        match &self.data {
            TensorData::F64(a) => Tensor::from_data(a.mapv(|x| x.sqrt())),
            TensorData::F32(a) => Tensor::from_data_f32(a.mapv(|x| x.sqrt())),
        }
    }

    pub fn exp(&self) -> Tensor {
        match &self.data {
            TensorData::F64(a) => Tensor::from_data(a.mapv(|x| x.exp())),
            TensorData::F32(a) => Tensor::from_data_f32(a.mapv(|x| x.exp())),
        }
    }

    pub fn log(&self) -> Tensor {
        match &self.data {
            TensorData::F64(a) => Tensor::from_data(a.mapv(|x| x.ln())),
            TensorData::F32(a) => Tensor::from_data_f32(a.mapv(|x| x.ln())),
        }
    }

    pub fn relu(&self) -> Tensor {
        match &self.data {
            TensorData::F64(a) => Tensor::from_data(a.mapv(|x| if x > 0.0 { x } else { 0.0 })),
            TensorData::F32(a) => Tensor::from_data_f32(a.mapv(|x| if x > 0.0 { x } else { 0.0 })),
        }
    }

    pub fn sigmoid(&self) -> Tensor {
        match &self.data {
            TensorData::F64(a) => Tensor::from_data(a.mapv(|x| 1.0 / (1.0 + (-x).exp()))),
            TensorData::F32(a) => Tensor::from_data_f32(a.mapv(|x| 1.0 / (1.0 + (-x).exp()))),
        }
    }

    pub fn tanh(&self) -> Tensor {
        match &self.data {
            TensorData::F64(a) => Tensor::from_data(a.mapv(|x| x.tanh())),
            TensorData::F32(a) => Tensor::from_data_f32(a.mapv(|x| x.tanh())),
        }
    }

    pub fn reshape(&self, shape: &[usize]) -> Option<Tensor> {
        match &self.data {
            TensorData::F64(a) => {
                let array = a.clone().into_shape_with_order(IxDyn(shape)).ok()?;
                Some(Tensor::from_data(array))
            }
            TensorData::F32(a) => {
                let array = a.clone().into_shape_with_order(IxDyn(shape)).ok()?;
                Some(Tensor::from_data_f32(array))
            }
        }
    }

    pub fn flatten(&self) -> Tensor {
        let size = self.data.len();
        match &self.data {
            TensorData::F64(a) => {
                let array = a.clone().into_shape_with_order(IxDyn(&[size])).unwrap();
                Tensor::from_data(array)
            }
            TensorData::F32(a) => {
                let array = a.clone().into_shape_with_order(IxDyn(&[size])).unwrap();
                Tensor::from_data_f32(array)
            }
        }
    }

    // ── Transformer ops ──────────────────────────────────────────────

    /// Layer normalization along the last dimension.
    /// x: (..., D), gamma: (D,), beta: (D,), eps: f64
    /// Returns (x - mean) / sqrt(var + eps) * gamma + beta
    /// 输入 dtype 决定输出 dtype；gamma/beta 自动 cast 到对应精度。
    pub fn layer_norm(&self, gamma: &Tensor, beta: &Tensor, eps: f64) -> Result<Tensor, String> {
        let shape = self.shape();
        let ndim = shape.len();
        if ndim == 0 || shape[ndim - 1] == 0 {
            // 空张量：保持 dtype 返回
            return Ok(match self.dtype {
                BaseType::F32 => self.clone(),
                _ => self.clone(),
            });
        }
        let axis_len = shape[ndim - 1];
        let outer_len: usize = shape[..ndim - 1].iter().product();

        let g_shape = gamma.shape();
        let b_shape = beta.shape();
        if g_shape.len() != 1 || g_shape[0] != axis_len {
            return Err(format!(
                "layer_norm: gamma shape {:?} does not match last axis length {}",
                g_shape, axis_len
            ));
        }
        if b_shape.len() != 1 || b_shape[0] != axis_len {
            return Err(format!(
                "layer_norm: beta shape {:?} does not match last axis length {}",
                b_shape, axis_len
            ));
        }

        if self.is_f32() {
            let a = self.data.as_f32().unwrap();
            let contiguous = a.as_standard_layout().to_owned();
            let flat: Vec<f32> = match contiguous.as_slice() {
                Some(s) => s.to_vec(),
                None => a.iter().copied().collect(),
            };
            let g_contig = gamma.data.as_f64_view().as_standard_layout().to_owned();
            let b_contig = beta.data.as_f64_view().as_standard_layout().to_owned();
            let g_slice: Vec<f32> = g_contip_iter_as_f32(&g_contig);
            let b_slice: Vec<f32> = g_contip_iter_as_f32(&b_contig);

            let mut result_data = Vec::with_capacity(flat.len());
            for i in 0..outer_len {
                let start = i * axis_len;
                let slice = &flat[start..start + axis_len];
                let mean: f32 = slice.iter().sum::<f32>() / axis_len as f32;
                let var: f32 = slice.iter().map(|x| (x - mean) * (x - mean)).sum::<f32>() / axis_len as f32;
                let std_inv = 1.0 / (var + eps as f32).sqrt();
                for j in 0..axis_len {
                    let x_hat = (slice[j] - mean) * std_inv;
                    let g = g_slice[j];
                    let b = b_slice[j];
                    result_data.push(g * x_hat + b);
                }
            }
            Ok(Tensor::from_vec_f32(result_data, shape))
        } else {
            let a = self.data.as_f64().unwrap();
            let contiguous = a.as_standard_layout().to_owned();
            let flat: Vec<f64> = match contiguous.as_slice() {
                Some(s) => s.to_vec(),
                None => a.iter().copied().collect(),
            };
            let g_contig = gamma.data.as_f64_view().as_standard_layout().to_owned();
            let b_contig = beta.data.as_f64_view().as_standard_layout().to_owned();
            let g_slice = g_contig.as_slice().unwrap_or(&[]);
            let b_slice = b_contig.as_slice().unwrap_or(&[]);

            let mut result_data = Vec::with_capacity(flat.len());
            for i in 0..outer_len {
                let start = i * axis_len;
                let slice = &flat[start..start + axis_len];
                let mean: f64 = slice.iter().sum::<f64>() / axis_len as f64;
                let var: f64 = slice.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / axis_len as f64;
                let std_inv = 1.0 / (var + eps).sqrt();
                for j in 0..axis_len {
                    let x_hat = (slice[j] - mean) * std_inv;
                    let g = g_slice[j];
                    let b = b_slice[j];
                    result_data.push(g * x_hat + b);
                }
            }
            Ok(Tensor::from_vec(result_data, shape))
        }
    }

    /// GELU activation (tanh approximation).
    /// 0.5 * x * (1.0 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    pub fn gelu(&self) -> Tensor {
        match &self.data {
            TensorData::F64(a) => {
                let sqrt_2_over_pi = (2.0 / std::f64::consts::PI).sqrt();
                Tensor::from_data(a.mapv(|x| {
                    let inner = sqrt_2_over_pi * (x + 0.044715 * x * x * x);
                    0.5 * x * (1.0 + inner.tanh())
                }))
            }
            TensorData::F32(a) => {
                let sqrt_2_over_pi = (2.0 / std::f32::consts::PI).sqrt();
                Tensor::from_data_f32(a.mapv(|x| {
                    let inner = sqrt_2_over_pi * (x + 0.044715 * x * x * x);
                    0.5 * x * (1.0 + inner.tanh())
                }))
            }
        }
    }

    /// Concatenate two 2D tensors along the given dimension.
    /// Only supports dim=0 or dim=1 for 2D tensors.
    /// dtype 提升规则：同 dtype 保留；混合 → f64。
    pub fn cat(&self, other: &Tensor, dim: usize) -> Result<Tensor, String> {
        let a_shape = self.shape();
        let b_shape = other.shape();
        if a_shape.len() != 2 || b_shape.len() != 2 {
            return Err(format!("cat only supports 2D tensors, got {:?}D and {:?}D", a_shape.len(), b_shape.len()));
        }
        let use_f32 = self.is_f32() && other.is_f32();
        match dim {
            0 => {
                if a_shape[1] != b_shape[1] {
                    return Err(format!("cat dim=0: column mismatch {} vs {}", a_shape[1], b_shape[1]));
                }
                let a_flat = self.data.as_f64_view().as_standard_layout().to_owned();
                let b_flat = other.data.as_f64_view().as_standard_layout().to_owned();
                let a_slice = a_flat.as_slice().unwrap_or(&[]);
                let b_slice = b_flat.as_slice().unwrap_or(&[]);
                let mut data = Vec::with_capacity(a_slice.len() + b_slice.len());
                data.extend_from_slice(a_slice);
                data.extend_from_slice(b_slice);
                if use_f32 {
                    Ok(Tensor::from_vec_f32(data.iter().map(|v| *v as f32).collect(), vec![a_shape[0] + b_shape[0], a_shape[1]]))
                } else {
                    Ok(Tensor::from_vec(data, vec![a_shape[0] + b_shape[0], a_shape[1]]))
                }
            }
            1 => {
                if a_shape[0] != b_shape[0] {
                    return Err(format!("cat dim=1: row mismatch {} vs {}", a_shape[0], b_shape[0]));
                }
                let a_flat = self.data.as_f64_view().as_standard_layout().to_owned();
                let b_flat = other.data.as_f64_view().as_standard_layout().to_owned();
                let a_slice = a_flat.as_slice().unwrap_or(&[]);
                let b_slice = b_flat.as_slice().unwrap_or(&[]);
                let mut data = Vec::with_capacity(a_slice.len() + b_slice.len());
                for i in 0..a_shape[0] {
                    let a_start = i * a_shape[1];
                    data.extend_from_slice(&a_slice[a_start..a_start + a_shape[1]]);
                    let b_start = i * b_shape[1];
                    data.extend_from_slice(&b_slice[b_start..b_start + b_shape[1]]);
                }
                if use_f32 {
                    Ok(Tensor::from_vec_f32(data.iter().map(|v| *v as f32).collect(), vec![a_shape[0], a_shape[1] + b_shape[1]]))
                } else {
                    Ok(Tensor::from_vec(data, vec![a_shape[0], a_shape[1] + b_shape[1]]))
                }
            }
            _ => Err(format!("cat only supports dim=0 or dim=1, got dim={}", dim)),
        }
    }

    /// Masked fill: where mask is truthy (>0.5), fill with value.
    /// mask must have the same shape as self.
    pub fn masked_fill(&self, mask: &Tensor, value: f64) -> Result<Tensor, String> {
        if self.shape() != mask.shape() {
            return Err(format!("masked_fill: shape mismatch {:?} vs {:?}", self.shape(), mask.shape()));
        }
        let mask_data = mask.data.as_f64_view().as_standard_layout().to_owned();
        let mask_slice = mask_data.as_slice().unwrap_or(&[]);
        match &self.data {
            TensorData::F64(a) => {
                let self_data = a.as_standard_layout().to_owned();
                let self_slice = self_data.as_slice().unwrap_or(&[]);
                let mut result = Vec::with_capacity(self_slice.len());
                for i in 0..self_slice.len() {
                    if mask_slice.get(i).copied().unwrap_or(0.0) > 0.5 {
                        result.push(value);
                    } else {
                        result.push(self_slice[i]);
                    }
                }
                Ok(Tensor::from_vec(result, self.shape()))
            }
            TensorData::F32(a) => {
                let self_data = a.as_standard_layout().to_owned();
                let self_slice = self_data.as_slice().unwrap_or(&[]);
                let mut result = Vec::with_capacity(self_slice.len());
                let value_f32 = value as f32;
                for i in 0..self_slice.len() {
                    if mask_slice.get(i).copied().unwrap_or(0.0) > 0.5 {
                        result.push(value_f32);
                    } else {
                        result.push(self_slice[i]);
                    }
                }
                Ok(Tensor::from_vec_f32(result, self.shape()))
            }
        }
    }

    /// Element-wise select: result = cond ? then : else.
    /// cond truthy (>0.5) selects then, else selects else.
    /// 三输入按 NumPy 广播规则对齐。then/else dtype 决定结果 dtype
    /// （f32 与 f64 混合提升为 f64；与 cond dtype 无关）。
    pub fn select(cond: &Tensor, then: &Tensor, else_: &Tensor) -> Result<Tensor, String> {
        // 计算广播后的目标 shape
        let cond_shape = cond.shape();
        let then_shape = then.shape();
        let else_shape = else_.shape();
        let target_shape = broadcast_shape(&[
            cond_shape.as_slice(), then_shape.as_slice(), else_shape.as_slice(),
        ]).ok_or_else(|| format!(
            "select: shapes cond={:?}, then={:?}, else={:?} 不兼容广播",
            cond_shape, then_shape, else_shape
        ))?;
        // dtype 提升：then/else 决定（f32 + f64 → f64）
        let result_dtype = match (then.dtype, else_.dtype) {
            (BaseType::F32, BaseType::F32) => BaseType::F32,
            (BaseType::F64, BaseType::F64) => BaseType::F64,
            // f32 与 f64 混合或与其它 → f64
            _ => BaseType::F64,
        };
        // 广播三个输入到 target_shape 并逐元素选择
        let cond_b = broadcast_to_owned(&cond.data, &target_shape);
        let then_b = broadcast_to_owned(&then.data, &target_shape);
        let else_b = broadcast_to_owned(&else_.data, &target_shape);
        let cond_slice = cond_b.as_slice().unwrap_or(&[]);
        let then_slice = then_b.as_slice().unwrap_or(&[]);
        let else_slice = else_b.as_slice().unwrap_or(&[]);
        let n = target_shape.iter().product::<usize>();
        match result_dtype {
            BaseType::F32 => {
                // cond/then/else 都 cast 到 f64 视图后再转 f32
                let mut result = Vec::with_capacity(n);
                for i in 0..n {
                    let c = cond_slice.get(i).copied().unwrap_or(0.0);
                    let t = then_slice.get(i).copied().unwrap_or(0.0);
                    let e = else_slice.get(i).copied().unwrap_or(0.0);
                    result.push((if c > 0.5 { t } else { e }) as f32);
                }
                Ok(Tensor::from_vec_f32(result, target_shape))
            }
            _ => {
                let mut result = Vec::with_capacity(n);
                for i in 0..n {
                    let c = cond_slice.get(i).copied().unwrap_or(0.0);
                    let t = then_slice.get(i).copied().unwrap_or(0.0);
                    let e = else_slice.get(i).copied().unwrap_or(0.0);
                    result.push(if c > 0.5 { t } else { e });
                }
                Ok(Tensor::from_vec(result, target_shape))
            }
        }
    }

    /// Permute dimensions. Only supports 2D/3D/4D tensors.
    pub fn permute(&self, dims: &[usize]) -> Result<Tensor, String> {
        let ndim = self.ndim();
        if dims.len() != ndim {
            return Err(format!("permute: expected {} dims, got {}", ndim, dims.len()));
        }
        if ndim < 2 || ndim > 4 {
            return Err(format!("permute only supports 2D/3D/4D tensors, got {}D", ndim));
        }
        let mut sorted = dims.to_vec();
        sorted.sort();
        let expected: Vec<usize> = (0..ndim).collect();
        if sorted != expected {
            return Err(format!("permute: {:?} is not a valid permutation of {} dims", dims, ndim));
        }
        match &self.data {
            TensorData::F64(a) => {
                let result = a.view().permuted_axes(dims.to_vec()).to_owned();
                Ok(Tensor::from_data(result))
            }
            TensorData::F32(a) => {
                let result = a.view().permuted_axes(dims.to_vec()).to_owned();
                Ok(Tensor::from_data_f32(result))
            }
        }
    }

    /// Softmax along the last axis (per-row for 2D).
    /// For a tensor of shape [..., N], each slice along the last dimension
    /// is independently softmaxed. 保留输入 dtype。
    pub fn softmax(&self) -> Option<Tensor> {
        let shape = self.shape().to_vec();
        let ndim = shape.len();
        if ndim == 0 || shape[ndim - 1] == 0 {
            return None;
        }
        let axis_len = shape[ndim - 1];
        let outer_len: usize = shape[..ndim - 1].iter().product();

        match &self.data {
            TensorData::F64(a) => {
                let contiguous = a.as_standard_layout().to_owned();
                let flat = contiguous.as_slice()?;
                let mut result_data = Vec::with_capacity(flat.len());
                for i in 0..outer_len {
                    let start = i * axis_len;
                    let slice = &flat[start..start + axis_len];
                    let max_val = slice.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                    let exps: Vec<f64> = slice.iter().map(|x| (x - max_val).exp()).collect();
                    let sum: f64 = exps.iter().sum();
                    let probs: Vec<f64> = if sum == 0.0 {
                        vec![1.0 / axis_len as f64; axis_len]
                    } else {
                        exps.iter().map(|x| x / sum).collect()
                    };
                    result_data.extend(probs);
                }
                Some(Tensor::from_vec(result_data, shape))
            }
            TensorData::F32(a) => {
                let contiguous = a.as_standard_layout().to_owned();
                let flat = contiguous.as_slice()?;
                let mut result_data = Vec::with_capacity(flat.len());
                for i in 0..outer_len {
                    let start = i * axis_len;
                    let slice = &flat[start..start + axis_len];
                    let max_val = slice.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                    let exps: Vec<f32> = slice.iter().map(|x| (x - max_val).exp()).collect();
                    let sum: f32 = exps.iter().sum();
                    let probs: Vec<f32> = if sum == 0.0 {
                        vec![1.0 / axis_len as f32; axis_len]
                    } else {
                        exps.iter().map(|x| x / sum).collect()
                    };
                    result_data.extend(probs);
                }
                Some(Tensor::from_vec_f32(result_data, shape))
            }
        }
    }

    // ── conv2d helpers ───────────────────────────────────────────────

    /// im2col: extract sliding windows from a 4D tensor (N, C, H, W)
    /// into a 2D matrix (N*H_out*W_out, C*K_H*K_W).
    /// Returns (col_matrix, output_height, output_width).
    /// 保留输入 dtype。
    pub fn im2col(&self, kernel_h: usize, kernel_w: usize, stride: usize, pad: usize) -> Option<(Tensor, usize, usize)> {
        let shape = self.shape();
        if shape.len() != 4 { return None; }
        let (n, c, h, w) = (shape[0], shape[1], shape[2], shape[3]);
        let h_out = (h + 2 * pad - kernel_h) / stride + 1;
        let w_out = (w + 2 * pad - kernel_w) / stride + 1;

        let mut cols = Vec::with_capacity(n * h_out * w_out * c * kernel_h * kernel_w);
        match &self.data {
            TensorData::F64(a) => {
                let flat = a.as_standard_layout().to_owned();
                let slice = flat.as_slice()?;
                for ni in 0..n {
                    for hi in 0..h_out {
                        for wi in 0..w_out {
                            for ci in 0..c {
                                for kh in 0..kernel_h {
                                    let ih = hi * stride + kh;
                                    for kw in 0..kernel_w {
                                        let iw = wi * stride + kw;
                                        if ih >= pad && ih < h + pad && iw >= pad && iw < w + pad {
                                            let ih_adj = ih - pad;
                                            let iw_adj = iw - pad;
                                            let idx = ((ni * c + ci) * h + ih_adj) * w + iw_adj;
                                            cols.push(slice.get(idx).copied().unwrap_or(0.0));
                                        } else {
                                            cols.push(0.0);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                let col_tensor = Tensor::from_vec(cols, vec![n * h_out * w_out, c * kernel_h * kernel_w]);
                Some((col_tensor, h_out, w_out))
            }
            TensorData::F32(a) => {
                let flat = a.as_standard_layout().to_owned();
                let slice = flat.as_slice()?;
                let mut cols_f32: Vec<f32> = Vec::with_capacity(n * h_out * w_out * c * kernel_h * kernel_w);
                for ni in 0..n {
                    for hi in 0..h_out {
                        for wi in 0..w_out {
                            for ci in 0..c {
                                for kh in 0..kernel_h {
                                    let ih = hi * stride + kh;
                                    for kw in 0..kernel_w {
                                        let iw = wi * stride + kw;
                                        if ih >= pad && ih < h + pad && iw >= pad && iw < w + pad {
                                            let ih_adj = ih - pad;
                                            let iw_adj = iw - pad;
                                            let idx = ((ni * c + ci) * h + ih_adj) * w + iw_adj;
                                            cols_f32.push(slice.get(idx).copied().unwrap_or(0.0));
                                        } else {
                                            cols_f32.push(0.0);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                let col_tensor = Tensor::from_vec_f32(cols_f32, vec![n * h_out * w_out, c * kernel_h * kernel_w]);
                Some((col_tensor, h_out, w_out))
            }
        }
    }

    /// In-place element-wise assignment: `self[i..] = src`.
    /// This mutates the underlying ArrayD in-place.
    /// Returns `Err` if shapes are incompatible (neither equal nor broadcastable to self's shape).
    pub fn assign_(&mut self, src: &Tensor) -> Result<(), String> {
        if self.shape() == src.shape() {
            match (&mut self.data, &src.data) {
                (TensorData::F64(s), TensorData::F64(o)) => {
                    s.zip_mut_with(o, |s, &x| *s = x);
                    return Ok(());
                }
                (TensorData::F32(s), TensorData::F32(o)) => {
                    s.zip_mut_with(o, |s, &x| *s = x);
                    return Ok(());
                }
                _ => {
                    // dtype 不一致，按 self dtype cast 后赋值
                    let o_view = src.data.as_f64_view();
                    match &mut self.data {
                        TensorData::F64(s) => {
                            s.zip_mut_with(&o_view, |s, &x| *s = x);
                            return Ok(());
                        }
                        TensorData::F32(s) => {
                            let o_f32 = o_view.mapv(|v| v as f32);
                            s.zip_mut_with(&o_f32, |s, &x| *s = x);
                            return Ok(());
                        }
                    }
                }
            }
        }
        // 形状不一致：尝试广播
        let src_view = src.data.as_f64_view();
        if let Some(src_br) = src_view.broadcast(self.shape().as_slice()) {
            match &mut self.data {
                TensorData::F64(s) => {
                    s.zip_mut_with(&src_br, |s, &x| *s = x);
                    return Ok(());
                }
                TensorData::F32(s) => {
                    let src_br_f32: ArrayD<f32> = src_br.mapv(|v| v as f32);
                    s.zip_mut_with(&src_br_f32, |s, &x| *s = x);
                    return Ok(());
                }
            }
        }
        Err(format!(
            "assign_: shape mismatch, cannot broadcast {:?} into {:?}",
            src.shape(),
            self.shape()
        ))
    }
}

/// 将任意 ndarray（f64 视图）按 f32 收集为 Vec<f32>。
/// layer_norm 中 gamma/beta cast 用，避免命名冲突。
fn g_contip_iter_as_f32(arr: &ArrayD<f64>) -> Vec<f32> {
    arr.iter().map(|v| *v as f32).collect()
}

/// 计算 NumPy 风格广播后的目标 shape（从右向左对齐）。
/// 任一维度不兼容（非 1 且不相等）返回 None。
fn broadcast_shape(shapes: &[&[usize]]) -> Option<Vec<usize>> {
    if shapes.is_empty() { return Some(vec![]); }
    let max_ndim = shapes.iter().map(|s| s.len()).max()?;
    let mut result = vec![1usize; max_ndim];
    for s in shapes {
        let offset = max_ndim - s.len();
        for (i, &dim) in s.iter().enumerate() {
            let target_idx = offset + i;
            let cur = result[target_idx];
            if cur == 1 {
                result[target_idx] = dim;
            } else if dim != 1 && dim != cur {
                return None;
            }
        }
    }
    Some(result)
}

/// 将 TensorData 广播到 target_shape（返回 f64 owned ArrayD）。
/// 用于 select 等需要三输入广播的算子。失败时返回 1 元素标量广播。
fn broadcast_to_owned(data: &TensorData, target_shape: &[usize]) -> ArrayD<f64> {
    let view = data.as_f64_view();
    if view.shape() == target_shape {
        return view.clone();
    }
    // 用 ndarray broadcast 视图，再 to_owned 实际化
    if let Some(bcast) = view.broadcast(IxDyn(target_shape)) {
        bcast.to_owned()
    } else {
        // 广播失败：保守返回 target_shape 的零张量（前向校验应已拦截）
        ArrayD::zeros(IxDyn(target_shape))
    }
}

impl fmt::Display for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.data {
            TensorData::F64(a) => write!(f, "{}", a),
            TensorData::F32(a) => write!(f, "{}", a),
        }
    }
}

// ── TensorData 算术 trait impl（让外部代码 &x.data * &y.data 等表达式自动工作）──
// F32 路径自动 cast 为 f64 视图后参与运算，结果为 ArrayD<f64>。
// 这是 Phase 1 兼容层：F32 张量在外部算术中表现为 f64，dtype 信息在此层丢失。
// Phase 3/4 改造外部代码使用真正的 f32 路径后，这些 trait impl 可移除。

use std::ops::{Add, Sub, Mul, Div, Neg, Index};

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
        }
    }
}

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

// ── Display ──

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
        }
    }
}
