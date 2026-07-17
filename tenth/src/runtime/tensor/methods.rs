use super::{Tensor, TensorData};
use crate::hir::types::BaseType;
use half::{bf16, f16};
use ndarray::{ArrayD, IxDyn};

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

    /// 从 f16 数据构造 F16 Tensor（Wave 2）。
    pub(crate) fn from_data_f16(data: ArrayD<f16>) -> Self {
        Tensor {
            dtype: BaseType::F16,
            data: TensorData::F16(data),
            grad: None,
            tape_id: None,
        }
    }

    /// 从 bf16 数据构造 BF16 Tensor（Wave 2）。
    pub(crate) fn from_data_bf16(data: ArrayD<bf16>) -> Self {
        Tensor {
            dtype: BaseType::BF16,
            data: TensorData::BF16(data),
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

    /// 是否为 f16 张量（Wave 2）。
    pub fn is_f16(&self) -> bool {
        matches!(self.dtype, BaseType::F16)
    }

    /// 是否为 bf16 张量（Wave 2）。
    pub fn is_bf16(&self) -> bool {
        matches!(self.dtype, BaseType::BF16)
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
    /// g 可为 F32/F64/F16/BF16；按 self.dtype 转换存储（f32 参数→F32 grad，f64 参数→F64 grad）。
    /// 返回 Err 当 g.shape() 与 self.data.shape() 不一致（防止 silent broadcast 掩盖梯度 shape 错误）。
    /// 阶段 4：签名从 `&ArrayD<f64>` 改为 `&TensorData`，支持真正的 f32 反向传播。
    /// Phase 2：F16/BF16 param 的 grad 累加使用 F32 中间表示（AMP 策略），
    /// 避免 F16 溢出（max≈65504）和 BF16 精度损失。grad buffer 存储为 F32，
    /// 优化器读取时可转回原 dtype。
    pub fn acc_grad(&mut self, g: &TensorData) -> Result<(), String> {
        // shape 校验：梯度 shape 必须与参数 shape 一致（方向 A：消除 silent squeeze）
        let self_shape = self.data.shape();
        let g_shape = g.shape();
        if self_shape != g_shape {
            return Err(format!(
                "acc_grad shape 不匹配：参数 shape {:?}，梯度 shape {:?}（可能反向传播 silent squeeze 掩盖了 shape 错误）",
                self_shape, g_shape
            ));
        }
        // Phase 2：F16/BF16 param 使用 F32 中间累加策略（AMP）。
        // grad buffer 存储为 F32，避免 F16/BF16 溢出和精度损失。
        // 输入 g 可能是 F32/F64/F16/BF16，需先转 F32 再累加。
        match self.dtype {
            BaseType::F32 | BaseType::F16 | BaseType::BF16 => {
                let g_f32 = match g {
                    TensorData::F32(a) => a.clone(),
                    TensorData::F64(a) => a.mapv(|v| v as f32),
                    TensorData::F16(a) => a.mapv(|v| v.to_f32()),
                    TensorData::BF16(a) => a.mapv(|v| v.to_f32()),
                };
                match &mut self.grad {
                    Some(TensorData::F32(cur)) => {
                        *cur = &*cur + &g_f32;
                    }
                    // 已有 F64 grad（混合 dtype 场景），回退为 f64 累加（策略 B 残留）
                    Some(TensorData::F64(cur)) => {
                        let merged = &*cur + &g_f32.mapv(|v| v as f64);
                        *cur = merged;
                    }
                    // 已有 F16/BF16 grad（旧数据或混合场景），提升为 F32 中间表示
                    Some(TensorData::F16(cur)) => {
                        let merged = cur.mapv(|v| v.to_f32()) + &g_f32;
                        self.grad = Some(TensorData::F32(merged));
                    }
                    Some(TensorData::BF16(cur)) => {
                        let merged = cur.mapv(|v| v.to_f32()) + &g_f32;
                        self.grad = Some(TensorData::F32(merged));
                    }
                    None => {
                        self.grad = Some(TensorData::F32(g_f32));
                    }
                }
            }
            _ => {
                let g_f64 = match g {
                    TensorData::F64(a) => a.clone(),
                    TensorData::F32(a) => a.mapv(|v| v as f64),
                    TensorData::F16(a) => a.mapv(|v| v.to_f64()),
                    TensorData::BF16(a) => a.mapv(|v| v.to_f64()),
                };
                match &mut self.grad {
                    Some(TensorData::F64(cur)) => {
                        *cur = &*cur + &g_f64;
                    }
                    // 已有 F32 grad（混合 dtype 场景），回退为 f64 累加（策略 B 残留）
                    Some(TensorData::F32(cur)) => {
                        let merged = cur.mapv(|v| v as f64) + &g_f64;
                        self.grad = Some(TensorData::F64(merged));
                    }
                    // F16/BF16 grad 不应到达 F64 param；回退为 f64 累加
                    Some(TensorData::F16(cur)) => {
                        let merged = cur.mapv(|v| v.to_f64()) + &g_f64;
                        self.grad = Some(TensorData::F64(merged));
                    }
                    Some(TensorData::BF16(cur)) => {
                        let merged = cur.mapv(|v| v.to_f64()) + &g_f64;
                        self.grad = Some(TensorData::F64(merged));
                    }
                    None => {
                        self.grad = Some(TensorData::F64(g_f64));
                    }
                }
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

    // ── constructors (f16, Wave 2) ────────────────────────────────────

    pub fn from_vec_f16(data: Vec<f16>, shape: Vec<usize>) -> Self {
        let array = ArrayD::from_shape_vec(IxDyn(&shape), data)
            .expect("invalid tensor shape");
        Tensor::from_data_f16(array)
    }

    pub fn zeros_f16(shape: &[usize]) -> Self {
        Tensor::from_data_f16(ArrayD::from_elem(IxDyn(shape), f16::from_f32(0.0)))
    }

    pub fn ones_f16(shape: &[usize]) -> Self {
        Tensor::from_data_f16(ArrayD::from_elem(IxDyn(shape), f16::from_f32(1.0)))
    }

    pub fn full_f16(shape: &[usize], value: f16) -> Self {
        Tensor::from_data_f16(ArrayD::from_elem(IxDyn(shape), value))
    }

    // ── constructors (bf16, Wave 2) ───────────────────────────────────

    pub fn from_vec_bf16(data: Vec<bf16>, shape: Vec<usize>) -> Self {
        let array = ArrayD::from_shape_vec(IxDyn(&shape), data)
            .expect("invalid tensor shape");
        Tensor::from_data_bf16(array)
    }

    pub fn zeros_bf16(shape: &[usize]) -> Self {
        Tensor::from_data_bf16(ArrayD::from_elem(IxDyn(shape), bf16::from_f32(0.0)))
    }

    pub fn ones_bf16(shape: &[usize]) -> Self {
        Tensor::from_data_bf16(ArrayD::from_elem(IxDyn(shape), bf16::from_f32(1.0)))
    }

    pub fn full_bf16(shape: &[usize], value: bf16) -> Self {
        Tensor::from_data_bf16(ArrayD::from_elem(IxDyn(shape), value))
    }

    // ── constructors (通用 dtype) ─────────────────────────────────────

    /// 按指定 dtype 构造全零张量。支持 F32/F64/F16/BF16，其他 dtype 返回 F64 兜底。
    pub fn zeros_with_dtype(shape: &[usize], dtype: BaseType) -> Self {
        match dtype {
            BaseType::F32 => Tensor::zeros_f32(shape),
            BaseType::F16 => Tensor::zeros_f16(shape),
            BaseType::BF16 => Tensor::zeros_bf16(shape),
            _ => Tensor::zeros(shape),
        }
    }

    /// 按指定 dtype 构造全一张量。
    pub fn ones_with_dtype(shape: &[usize], dtype: BaseType) -> Self {
        match dtype {
            BaseType::F32 => Tensor::ones_f32(shape),
            BaseType::F16 => Tensor::ones_f16(shape),
            BaseType::BF16 => Tensor::ones_bf16(shape),
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
            TensorData::F16(a) => a.get(IxDyn(index)).map(|v| v.to_f64()),
            TensorData::BF16(a) => a.get(IxDyn(index)).map(|v| v.to_f64()),
        }
    }

    /// 沿第 0 维索引，返回降维后的子张量（NumPy 语义）。
    /// 例如 shape [2,3] 的 t[0] → shape [3] 的 1D 张量；
    /// shape [3] 的 t[0] → 标量（1D 张量，shape []）。
    /// 索引越界返回 Err。
    pub fn index_dim(&self, idx: usize) -> Result<Tensor, String> {
        let ndim = self.data.ndim();
        if ndim == 0 {
            return Err("无法对 0D 张量索引".into());
        }
        let dim_len = self.data.shape()[0];
        if idx >= dim_len {
            return Err(format!("索引 {} 越界，第 0 维长度为 {}", idx, dim_len));
        }
        match &self.data {
            TensorData::F64(a) => {
                let sub = a.index_axis(ndarray::Axis(0), idx).to_owned();
                Ok(Tensor::from_data(sub))
            }
            TensorData::F32(a) => {
                let sub = a.index_axis(ndarray::Axis(0), idx).to_owned();
                Ok(Tensor::from_data_f32(sub))
            }
            TensorData::F16(a) => {
                let sub = a.index_axis(ndarray::Axis(0), idx).to_owned();
                Ok(Tensor::from_data_f16(sub))
            }
            TensorData::BF16(a) => {
                let sub = a.index_axis(ndarray::Axis(0), idx).to_owned();
                Ok(Tensor::from_data_bf16(sub))
            }
        }
    }

    // ── 内部 f32/f64/f16/bf16 分支辅助 ────────────────────────────────

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

    /// 对每个元素应用 f16 → f16 映射（仅 F16 路径，Wave 2）。
    /// 实现策略：转 f32 计算，结果转回 f16（f16 精度太低不适合直接运算）。
    fn map_f16(&self, f: impl Fn(f32) -> f32) -> Tensor {
        let a = self.data.as_f16().expect("map_f16 requires F16 tensor");
        let result = a.mapv(|v| f16::from_f32(f(v.to_f32())));
        Tensor::from_data_f16(result)
    }

    /// 对每个元素应用 bf16 → bf16 映射（仅 BF16 路径，Wave 2）。
    /// 实现策略：转 f32 计算，结果转回 bf16。
    fn map_bf16(&self, f: impl Fn(f32) -> f32) -> Tensor {
        let a = self.data.as_bf16().expect("map_bf16 requires BF16 tensor");
        let result = a.mapv(|v| bf16::from_f32(f(v.to_f32())));
        Tensor::from_data_bf16(result)
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

    /// 对两个同 dtype 张量做元素级 f16 运算（仅 F16 路径，Wave 2）。
    /// 实现策略：转 f32 计算，结果转回 f16。
    fn zip_f16(&self, other: &Tensor, f: impl Fn(f32, f32) -> f32) -> Result<Tensor, String> {
        let a = self.data.as_f16().expect("zip_f16 self requires F16");
        let b = other.data.as_f16().expect("zip_f16 other requires F16");
        let out_shape = Self::broadcast_shape(a.shape(), b.shape())
            .ok_or_else(|| format!("cannot broadcast shapes {:?} + {:?}", self.shape(), other.shape()))?;
        let a_br = a.broadcast(IxDyn(&out_shape)).unwrap();
        let b_br = b.broadcast(IxDyn(&out_shape)).unwrap();
        let mut out: ArrayD<f16> = ArrayD::from_elem(IxDyn(&out_shape), f16::from_f32(0.0));
        out.zip_mut_with(&a_br.to_owned(), |o, &x| { *o = x; });
        let b_owned = b_br.to_owned();
        out.zip_mut_with(&b_owned, |o, &y| { *o = f16::from_f32(f(o.to_f32(), y.to_f32())); });
        Ok(Tensor::from_data_f16(out))
    }

    /// 对两个同 dtype 张量做元素级 bf16 运算（仅 BF16 路径，Wave 2）。
    /// 实现策略：转 f32 计算，结果转回 bf16。
    fn zip_bf16(&self, other: &Tensor, f: impl Fn(f32, f32) -> f32) -> Result<Tensor, String> {
        let a = self.data.as_bf16().expect("zip_bf16 self requires BF16");
        let b = other.data.as_bf16().expect("zip_bf16 other requires BF16");
        let out_shape = Self::broadcast_shape(a.shape(), b.shape())
            .ok_or_else(|| format!("cannot broadcast shapes {:?} + {:?}", self.shape(), other.shape()))?;
        let a_br = a.broadcast(IxDyn(&out_shape)).unwrap();
        let b_br = b.broadcast(IxDyn(&out_shape)).unwrap();
        let mut out: ArrayD<bf16> = ArrayD::from_elem(IxDyn(&out_shape), bf16::from_f32(0.0));
        out.zip_mut_with(&a_br.to_owned(), |o, &x| { *o = x; });
        let b_owned = b_br.to_owned();
        out.zip_mut_with(&b_owned, |o, &y| { *o = bf16::from_f32(f(o.to_f32(), y.to_f32())); });
        Ok(Tensor::from_data_bf16(out))
    }

    // ── reductions ─────────────────────────────────────────────────────

    pub fn sum(&self) -> f64 {
        match &self.data {
            TensorData::F64(a) => a.sum(),
            TensorData::F32(a) => a.sum() as f64,
            TensorData::F16(a) => a.iter().map(|v| v.to_f64()).sum(),
            TensorData::BF16(a) => a.iter().map(|v| v.to_f64()).sum(),
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
            // F16/BF16 sum_axis 转 f64 计算（Phase 1 简化）
            TensorData::F16(a) => Tensor::from_data(a.mapv(|v| v.to_f64()).sum_axis(ndarray::Axis(axis))),
            TensorData::BF16(a) => Tensor::from_data(a.mapv(|v| v.to_f64()).sum_axis(ndarray::Axis(axis))),
        };
        Ok(result)
    }

    pub fn mean(&self) -> f64 {
        match &self.data {
            TensorData::F64(a) => a.mean().unwrap_or(0.0),
            TensorData::F32(a) => a.mean().map(|v| v as f64).unwrap_or(0.0),
            TensorData::F16(a) => {
                let n = a.len() as f64;
                if n == 0.0 { 0.0 } else { a.iter().map(|v| v.to_f64()).sum::<f64>() / n }
            }
            TensorData::BF16(a) => {
                let n = a.len() as f64;
                if n == 0.0 { 0.0 } else { a.iter().map(|v| v.to_f64()).sum::<f64>() / n }
            }
        }
    }

    pub fn max_val(&self) -> f64 {
        match &self.data {
            TensorData::F64(a) => a.iter().copied().fold(f64::NEG_INFINITY, f64::max),
            TensorData::F32(a) => a.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64,
            TensorData::F16(a) => a.iter().map(|v| v.to_f64()).fold(f64::NEG_INFINITY, f64::max),
            TensorData::BF16(a) => a.iter().map(|v| v.to_f64()).fold(f64::NEG_INFINITY, f64::max),
        }
    }

    /// Return the index of the maximum value (flat index).
    /// Returns -1 for an empty tensor.
    pub fn argmax(&self) -> i64 {
        let iter: Box<dyn Iterator<Item = f64>> = match &self.data {
            TensorData::F64(a) => Box::new(a.iter().copied()),
            TensorData::F32(a) => Box::new(a.iter().map(|v| *v as f64)),
            TensorData::F16(a) => Box::new(a.iter().map(|v| v.to_f64())),
            TensorData::BF16(a) => Box::new(a.iter().map(|v| v.to_f64())),
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
            TensorData::F16(a) => Tensor::from_data_f16(a.mapv(|x| f16::from_f64(x.to_f64() + scalar))),
            TensorData::BF16(a) => Tensor::from_data_bf16(a.mapv(|x| bf16::from_f64(x.to_f64() + scalar))),
        }
    }

    pub fn sub_scalar(&self, scalar: f64) -> Tensor {
        match &self.data {
            TensorData::F64(a) => Tensor::from_data(a.mapv(|x| x - scalar)),
            TensorData::F32(a) => Tensor::from_data_f32(a.mapv(|x| x - scalar as f32)),
            TensorData::F16(a) => Tensor::from_data_f16(a.mapv(|x| f16::from_f64(x.to_f64() - scalar))),
            TensorData::BF16(a) => Tensor::from_data_bf16(a.mapv(|x| bf16::from_f64(x.to_f64() - scalar))),
        }
    }

    pub fn mul_scalar(&self, scalar: f64) -> Tensor {
        match &self.data {
            TensorData::F64(a) => Tensor::from_data(a.mapv(|x| x * scalar)),
            TensorData::F32(a) => Tensor::from_data_f32(a.mapv(|x| x * scalar as f32)),
            TensorData::F16(a) => Tensor::from_data_f16(a.mapv(|x| f16::from_f64(x.to_f64() * scalar))),
            TensorData::BF16(a) => Tensor::from_data_bf16(a.mapv(|x| bf16::from_f64(x.to_f64() * scalar))),
        }
    }

    pub fn div_scalar(&self, scalar: f64) -> Tensor {
        match &self.data {
            TensorData::F64(a) => Tensor::from_data(a.mapv(|x| x / scalar)),
            TensorData::F32(a) => Tensor::from_data_f32(a.mapv(|x| x / scalar as f32)),
            TensorData::F16(a) => Tensor::from_data_f16(a.mapv(|x| f16::from_f64(x.to_f64() / scalar))),
            TensorData::BF16(a) => Tensor::from_data_bf16(a.mapv(|x| bf16::from_f64(x.to_f64() / scalar))),
        }
    }

    /// Scalar divided by tensor: scalar / self (element-wise).
    pub fn div_scalar_inv(&self, scalar: f64) -> Tensor {
        match &self.data {
            TensorData::F64(a) => Tensor::from_data(a.mapv(|x| scalar / x)),
            TensorData::F32(a) => Tensor::from_data_f32(a.mapv(|x| scalar as f32 / x)),
            TensorData::F16(a) => Tensor::from_data_f16(a.mapv(|x| f16::from_f64(scalar / x.to_f64()))),
            TensorData::BF16(a) => Tensor::from_data_bf16(a.mapv(|x| bf16::from_f64(scalar / x.to_f64()))),
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

    /// Wave 2 dtype 提升规则：
    /// - F64 + 任何 → F64
    /// - F32 + F32 → F32；F32 + F16/BF16 → F32
    /// - F16 + F16 → F16；BF16 + BF16 → BF16；F16 + BF16 → F32
    fn promote_dtype(a: BaseType, b: BaseType) -> BaseType {
        use BaseType::*;
        match (a, b) {
            (F64, _) | (_, F64) => F64,
            (F32, _) | (_, F32) => F32,
            (F16, F16) => F16,
            (BF16, BF16) => BF16,
            // F16 + BF16 → F32（半精度混合提到 f32）
            (F16, BF16) | (BF16, F16) => F32,
            // 兜底（理论上不会到这里，因为上面已覆盖所有浮点组合）
            _ => F64,
        }
    }

    /// 通用元素级二元运算（带广播 + dtype 提升）。
    /// `f32_op` / `f64_op` 接收 f32/f64 返回 f32/f64，按提升后的 dtype 分发。
    /// F16/BF16 路径：转 f32 计算，结果转回 F16/BF16。
    fn elementwise_binary(
        &self,
        other: &Tensor,
        f32_op: impl Fn(f32, f32) -> f32,
        f64_op: impl Fn(f64, f64) -> f64,
        op_symbol: &str,
    ) -> Result<Tensor, String> {
        let result_dtype = Self::promote_dtype(self.dtype, other.dtype);
        let a_shape = self.data.shape();
        let b_shape = other.data.shape();
        let out_shape = Self::broadcast_shape(a_shape, b_shape)
            .ok_or_else(|| format!("cannot broadcast shapes {:?} {} {:?}", self.shape(), op_symbol, other.shape()))?;

        match result_dtype {
            BaseType::F64 => {
                let a = self.data.as_f64_view();
                let b = other.data.as_f64_view();
                let a_br = a.broadcast(IxDyn(&out_shape)).unwrap();
                let b_br = b.broadcast(IxDyn(&out_shape)).unwrap();
                let mut out: ArrayD<f64> = ArrayD::zeros(IxDyn(&out_shape));
                out.zip_mut_with(&a_br.to_owned(), |o, &x| { *o = x; });
                let b_owned = b_br.to_owned();
                out.zip_mut_with(&b_owned, |o, &y| { *o = f64_op(*o, y); });
                Ok(Tensor::from_data(out))
            }
            BaseType::F32 => {
                let a = self.data.as_f64_view().mapv(|v| v as f32);
                let b = other.data.as_f64_view().mapv(|v| v as f32);
                let a_br = a.broadcast(IxDyn(&out_shape)).unwrap();
                let b_br = b.broadcast(IxDyn(&out_shape)).unwrap();
                let mut out: ArrayD<f32> = ArrayD::zeros(IxDyn(&out_shape));
                out.zip_mut_with(&a_br.to_owned(), |o, &x| { *o = x; });
                let b_owned = b_br.to_owned();
                out.zip_mut_with(&b_owned, |o, &y| { *o = f32_op(*o, y); });
                Ok(Tensor::from_data_f32(out))
            }
            BaseType::F16 => {
                // 仅 F16+F16 走此路径；转 f32 计算，结果转回 f16
                let a = self.data.as_f16().expect("F16 op requires F16 self").mapv(|v| v.to_f32());
                let b = other.data.as_f16().expect("F16 op requires F16 other").mapv(|v| v.to_f32());
                let a_br = a.broadcast(IxDyn(&out_shape)).unwrap();
                let b_br = b.broadcast(IxDyn(&out_shape)).unwrap();
                let mut out: ArrayD<f16> = ArrayD::from_elem(IxDyn(&out_shape), f16::from_f32(0.0));
                out.zip_mut_with(&a_br.to_owned(), |o, &x| { *o = f16::from_f32(x); });
                let b_owned = b_br.to_owned();
                out.zip_mut_with(&b_owned, |o, &y| { *o = f16::from_f32(f32_op(o.to_f32(), y)); });
                Ok(Tensor::from_data_f16(out))
            }
            BaseType::BF16 => {
                let a = self.data.as_bf16().expect("BF16 op requires BF16 self").mapv(|v| v.to_f32());
                let b = other.data.as_bf16().expect("BF16 op requires BF16 other").mapv(|v| v.to_f32());
                let a_br = a.broadcast(IxDyn(&out_shape)).unwrap();
                let b_br = b.broadcast(IxDyn(&out_shape)).unwrap();
                let mut out: ArrayD<bf16> = ArrayD::from_elem(IxDyn(&out_shape), bf16::from_f32(0.0));
                out.zip_mut_with(&a_br.to_owned(), |o, &x| { *o = bf16::from_f32(x); });
                let b_owned = b_br.to_owned();
                out.zip_mut_with(&b_owned, |o, &y| { *o = bf16::from_f32(f32_op(o.to_f32(), y)); });
                Ok(Tensor::from_data_bf16(out))
            }
            _ => {
                // 整数 dtype 等：兜底走 f64
                let a = self.data.as_f64_view();
                let b = other.data.as_f64_view();
                let a_br = a.broadcast(IxDyn(&out_shape)).unwrap();
                let b_br = b.broadcast(IxDyn(&out_shape)).unwrap();
                let mut out: ArrayD<f64> = ArrayD::zeros(IxDyn(&out_shape));
                out.zip_mut_with(&a_br.to_owned(), |o, &x| { *o = x; });
                let b_owned = b_br.to_owned();
                out.zip_mut_with(&b_owned, |o, &y| { *o = f64_op(*o, y); });
                Ok(Tensor::from_data(out))
            }
        }
    }

    /// Element-wise addition with broadcasting.  Errors if shapes are incompatible.
    /// Wave 2 dtype 提升规则：见 promote_dtype。
    pub fn add_tensor(&self, other: &Tensor) -> Result<Tensor, String> {
        self.elementwise_binary(other, |a, b| a + b, |a, b| a + b, "+")
    }

    pub fn sub_tensor(&self, other: &Tensor) -> Result<Tensor, String> {
        self.elementwise_binary(other, |a, b| a - b, |a, b| a - b, "-")
    }

    pub fn mul_tensor(&self, other: &Tensor) -> Result<Tensor, String> {
        self.elementwise_binary(other, |a, b| a * b, |a, b| a * b, "*")
    }

    pub fn div_tensor(&self, other: &Tensor) -> Result<Tensor, String> {
        self.elementwise_binary(other, |a, b| a / b, |a, b| a / b, "/")
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

    // ── batched matrix multiplication ─────────────────────────────────

    /// Batched matrix multiplication: (B, M, K) @ (B, K, N) -> (B, M, N).
    /// Both tensors must be 3D with matching batch dimension.
    /// 保留输入 dtype：f32@f32 → f32；f64@f64 → f64；混合 → f64（提升）。
    pub fn bmm(&self, other: &Tensor) -> Result<Tensor, String> {
        if self.ndim() != 3 || other.ndim() != 3 {
            return Err(format!(
                "bmm requires 3D tensors, got {:?}D and {:?}D",
                self.ndim(), other.ndim()
            ));
        }
        if self.shape()[0] != other.shape()[0] {
            return Err(format!(
                "bmm batch mismatch: self batch={}, other batch={}",
                self.shape()[0], other.shape()[0]
            ));
        }
        if self.shape()[2] != other.shape()[1] {
            return Err(format!(
                "bmm inner dim mismatch: self K={}, other K={}",
                self.shape()[2], other.shape()[1]
            ));
        }

        let use_f32 = self.is_f32() && other.is_f32();
        if use_f32 {
            self.bmm_f32(other)
        } else {
            // f64 路径（含混合提升：将 f32 视图 cast 为 f64 后走原 f64 逻辑）
            let a_view = self.data.as_f64_view();
            let b_view = other.data.as_f64_view();
            let a3 = a_view.view().into_dimensionality::<ndarray::Ix3>()
                .map_err(|_| "bmm: self must be 3D")?;
            let b3 = b_view.view().into_dimensionality::<ndarray::Ix3>()
                .map_err(|_| "bmm: other must be 3D")?;
            let batch = a3.shape()[0];
            let m = a3.shape()[1];
            let n = b3.shape()[2];
            let mut result = ndarray::Array3::<f64>::zeros((batch, m, n));
            for i in 0..batch {
                let a_slice = a3.index_axis(ndarray::Axis(0), i);
                let b_slice = b3.index_axis(ndarray::Axis(0), i);
                let r = a_slice.dot(&b_slice);
                result.index_axis_mut(ndarray::Axis(0), i).assign(&r);
            }
            Ok(Tensor::from_data(result.into_dyn()))
        }
    }

    /// f32 专用 bmm 路径，保持 f32 dtype。
    fn bmm_f32(&self, other: &Tensor) -> Result<Tensor, String> {
        let a = self.data.as_f32().expect("bmm_f32: self must be F32");
        let b = other.data.as_f32().expect("bmm_f32: other must be F32");
        let a3 = a.view().into_dimensionality::<ndarray::Ix3>()
            .map_err(|_| "bmm: self must be 3D")?;
        let b3 = b.view().into_dimensionality::<ndarray::Ix3>()
            .map_err(|_| "bmm: other must be 3D")?;
        let batch = a3.shape()[0];
        let m = a3.shape()[1];
        let n = b3.shape()[2];
        let mut result = ndarray::Array3::<f32>::zeros((batch, m, n));
        for i in 0..batch {
            let a_slice = a3.index_axis(ndarray::Axis(0), i);
            let b_slice = b3.index_axis(ndarray::Axis(0), i);
            let r = a_slice.dot(&b_slice);
            result.index_axis_mut(ndarray::Axis(0), i).assign(&r);
        }
        Ok(Tensor::from_data_f32(result.into_dyn()))
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
            TensorData::F16(a) => {
                let result = a.view().permuted_axes(perm);
                Some(Tensor::from_data_f16(result.to_owned()))
            }
            TensorData::BF16(a) => {
                let result = a.view().permuted_axes(perm);
                Some(Tensor::from_data_bf16(result.to_owned()))
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
            TensorData::F16(a) => {
                let broadcasted = a.broadcast(target_shape)?;
                Some(Tensor::from_data_f16(broadcasted.to_owned()))
            }
            TensorData::BF16(a) => {
                let broadcasted = a.broadcast(target_shape)?;
                Some(Tensor::from_data_bf16(broadcasted.to_owned()))
            }
        }
    }

    // ── unary elementwise ──────────────────────────────────────────────

    pub fn neg(&self) -> Tensor {
        match &self.data {
            TensorData::F64(a) => Tensor::from_data(a.mapv(|x| -x)),
            TensorData::F32(a) => Tensor::from_data_f32(a.mapv(|x| -x)),
            // F16/BF16 转 f64 计算，结果转回原 dtype
            TensorData::F16(a) => Tensor::from_data_f16(a.mapv(|x| f16::from_f64(-x.to_f64()))),
            TensorData::BF16(a) => Tensor::from_data_bf16(a.mapv(|x| bf16::from_f64(-x.to_f64()))),
        }
    }

    pub fn abs(&self) -> Tensor {
        match &self.data {
            TensorData::F64(a) => Tensor::from_data(a.mapv(|x| x.abs())),
            TensorData::F32(a) => Tensor::from_data_f32(a.mapv(|x| x.abs())),
            TensorData::F16(a) => Tensor::from_data_f16(a.mapv(|x| f16::from_f64(x.to_f64().abs()))),
            TensorData::BF16(a) => Tensor::from_data_bf16(a.mapv(|x| bf16::from_f64(x.to_f64().abs()))),
        }
    }

    /// 元素级裁剪到 [min_val, max_val]（用于梯度裁剪）。
    pub fn clip_scalar(&self, min_val: f64, max_val: f64) -> Tensor {
        match &self.data {
            TensorData::F64(a) => Tensor::from_data(a.mapv(|x| x.clamp(min_val, max_val))),
            TensorData::F32(a) => Tensor::from_data_f32(a.mapv(|x| (x as f64).clamp(min_val, max_val) as f32)),
            TensorData::F16(a) => Tensor::from_data_f16(a.mapv(|x| f16::from_f64(x.to_f64().clamp(min_val, max_val)))),
            TensorData::BF16(a) => Tensor::from_data_bf16(a.mapv(|x| bf16::from_f64(x.to_f64().clamp(min_val, max_val)))),
        }
    }

    pub fn sqrt(&self) -> Tensor {
        match &self.data {
            TensorData::F64(a) => Tensor::from_data(a.mapv(|x| x.sqrt())),
            TensorData::F32(a) => Tensor::from_data_f32(a.mapv(|x| x.sqrt())),
            TensorData::F16(a) => Tensor::from_data_f16(a.mapv(|x| f16::from_f64(x.to_f64().sqrt()))),
            TensorData::BF16(a) => Tensor::from_data_bf16(a.mapv(|x| bf16::from_f64(x.to_f64().sqrt()))),
        }
    }

    pub fn exp(&self) -> Tensor {
        match &self.data {
            TensorData::F64(a) => Tensor::from_data(a.mapv(|x| x.exp())),
            TensorData::F32(a) => Tensor::from_data_f32(a.mapv(|x| x.exp())),
            TensorData::F16(a) => Tensor::from_data_f16(a.mapv(|x| f16::from_f64(x.to_f64().exp()))),
            TensorData::BF16(a) => Tensor::from_data_bf16(a.mapv(|x| bf16::from_f64(x.to_f64().exp()))),
        }
    }

    pub fn log(&self) -> Tensor {
        match &self.data {
            TensorData::F64(a) => Tensor::from_data(a.mapv(|x| x.ln())),
            TensorData::F32(a) => Tensor::from_data_f32(a.mapv(|x| x.ln())),
            TensorData::F16(a) => Tensor::from_data_f16(a.mapv(|x| f16::from_f64(x.to_f64().ln()))),
            TensorData::BF16(a) => Tensor::from_data_bf16(a.mapv(|x| bf16::from_f64(x.to_f64().ln()))),
        }
    }

    pub fn relu(&self) -> Tensor {
        match &self.data {
            TensorData::F64(a) => Tensor::from_data(a.mapv(|x| if x > 0.0 { x } else { 0.0 })),
            TensorData::F32(a) => Tensor::from_data_f32(a.mapv(|x| if x > 0.0 { x } else { 0.0 })),
            TensorData::F16(a) => Tensor::from_data_f16(a.mapv(|x| f16::from_f64(if x.to_f64() > 0.0 { x.to_f64() } else { 0.0 }))),
            TensorData::BF16(a) => Tensor::from_data_bf16(a.mapv(|x| bf16::from_f64(if x.to_f64() > 0.0 { x.to_f64() } else { 0.0 }))),
        }
    }

    pub fn sigmoid(&self) -> Tensor {
        match &self.data {
            TensorData::F64(a) => Tensor::from_data(a.mapv(|x| 1.0 / (1.0 + (-x).exp()))),
            TensorData::F32(a) => Tensor::from_data_f32(a.mapv(|x| 1.0 / (1.0 + (-x).exp()))),
            TensorData::F16(a) => Tensor::from_data_f16(a.mapv(|x| {
                let v = x.to_f64();
                f16::from_f64(1.0 / (1.0 + (-v).exp()))
            })),
            TensorData::BF16(a) => Tensor::from_data_bf16(a.mapv(|x| {
                let v = x.to_f64();
                bf16::from_f64(1.0 / (1.0 + (-v).exp()))
            })),
        }
    }

    pub fn tanh(&self) -> Tensor {
        match &self.data {
            TensorData::F64(a) => Tensor::from_data(a.mapv(|x| x.tanh())),
            TensorData::F32(a) => Tensor::from_data_f32(a.mapv(|x| x.tanh())),
            TensorData::F16(a) => Tensor::from_data_f16(a.mapv(|x| f16::from_f64(x.to_f64().tanh()))),
            TensorData::BF16(a) => Tensor::from_data_bf16(a.mapv(|x| bf16::from_f64(x.to_f64().tanh()))),
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
            TensorData::F16(a) => {
                let array = a.clone().into_shape_with_order(IxDyn(shape)).ok()?;
                Some(Tensor::from_data_f16(array))
            }
            TensorData::BF16(a) => {
                let array = a.clone().into_shape_with_order(IxDyn(shape)).ok()?;
                Some(Tensor::from_data_bf16(array))
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
            TensorData::F16(a) => {
                let array = a.clone().into_shape_with_order(IxDyn(&[size])).unwrap();
                Tensor::from_data_f16(array)
            }
            TensorData::BF16(a) => {
                let array = a.clone().into_shape_with_order(IxDyn(&[size])).unwrap();
                Tensor::from_data_bf16(array)
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
            // F16/BF16 转 f64 计算，结果转回原 dtype
            TensorData::F16(a) => {
                let sqrt_2_over_pi = (2.0 / std::f64::consts::PI).sqrt();
                Tensor::from_data_f16(a.mapv(|x| {
                    let v = x.to_f64();
                    let inner = sqrt_2_over_pi * (v + 0.044715 * v * v * v);
                    f16::from_f64(0.5 * v * (1.0 + inner.tanh()))
                }))
            }
            TensorData::BF16(a) => {
                let sqrt_2_over_pi = (2.0 / std::f64::consts::PI).sqrt();
                Tensor::from_data_bf16(a.mapv(|x| {
                    let v = x.to_f64();
                    let inner = sqrt_2_over_pi * (v + 0.044715 * v * v * v);
                    bf16::from_f64(0.5 * v * (1.0 + inner.tanh()))
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
            TensorData::F16(a) => {
                let self_data = a.as_standard_layout().to_owned();
                let self_slice = self_data.as_slice().unwrap_or(&[]);
                let mut result = Vec::with_capacity(self_slice.len());
                let value_f16 = f16::from_f64(value);
                for i in 0..self_slice.len() {
                    if mask_slice.get(i).copied().unwrap_or(0.0) > 0.5 {
                        result.push(value_f16);
                    } else {
                        result.push(self_slice[i]);
                    }
                }
                Ok(Tensor::from_vec_f16(result, self.shape()))
            }
            TensorData::BF16(a) => {
                let self_data = a.as_standard_layout().to_owned();
                let self_slice = self_data.as_slice().unwrap_or(&[]);
                let mut result = Vec::with_capacity(self_slice.len());
                let value_bf16 = bf16::from_f64(value);
                for i in 0..self_slice.len() {
                    if mask_slice.get(i).copied().unwrap_or(0.0) > 0.5 {
                        result.push(value_bf16);
                    } else {
                        result.push(self_slice[i]);
                    }
                }
                Ok(Tensor::from_vec_bf16(result, self.shape()))
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

    // ── tensor-tensor comparison ops (返回 F64 张量，0.0/1.0 编码 bool) ─────
    //
    // Wave 2 第 4 项：张量比较运算。不引入新的 Bool TensorData 变体——
    // 沿用 f64 0.0/1.0 编码以保持与 select backward 的 `> 0.5` 判定兼容。
    // 输入 a/b dtype 任意（F32/F64/F16/BF16）；先 cast 到 f64 视图，再广播比较。
    // 不可微：比较结果是布尔掩码，不参与梯度计算（若需可微掩码，用 select 直接耦合）。

    /// 通用元素级比较（带广播，结果固定 F64 0.0/1.0）。
    /// `cmp` 接收 f64 返回 bool；结果 1.0/0.0。
    fn compare_binary(&self, other: &Tensor, cmp: impl Fn(f64, f64) -> bool, op_symbol: &str) -> Result<Tensor, String> {
        let a_shape = self.data.shape();
        let b_shape = other.data.shape();
        let out_shape = Self::broadcast_shape(a_shape, b_shape)
            .ok_or_else(|| format!("cannot broadcast shapes {:?} {} {:?}", self.shape(), op_symbol, other.shape()))?;
        let a = self.data.as_f64_view();
        let b = other.data.as_f64_view();
        let a_br = a.broadcast(IxDyn(&out_shape)).unwrap();
        let b_br = b.broadcast(IxDyn(&out_shape)).unwrap();
        let mut out: ArrayD<f64> = ArrayD::zeros(IxDyn(&out_shape));
        out.zip_mut_with(&a_br.to_owned(), |o, &x| { *o = x; });
        let b_owned = b_br.to_owned();
        out.zip_mut_with(&b_owned, |o, &y| { *o = if cmp(*o, y) { 1.0 } else { 0.0 }; });
        Ok(Tensor::from_data(out))
    }

    /// 逐元素 a > b，返回 F64 张量（1.0/0.0）。
    pub fn gt(&self, other: &Tensor) -> Result<Tensor, String> {
        self.compare_binary(other, |a, b| a > b, ">")
    }

    /// 逐元素 a < b，返回 F64 张量（1.0/0.0）。
    pub fn lt(&self, other: &Tensor) -> Result<Tensor, String> {
        self.compare_binary(other, |a, b| a < b, "<")
    }

    /// 逐元素 a >= b，返回 F64 张量（1.0/0.0）。
    pub fn ge(&self, other: &Tensor) -> Result<Tensor, String> {
        self.compare_binary(other, |a, b| a >= b, ">=")
    }

    /// 逐元素 a <= b，返回 F64 张量（1.0/0.0）。
    pub fn le(&self, other: &Tensor) -> Result<Tensor, String> {
        self.compare_binary(other, |a, b| a <= b, "<=")
    }

    /// 逐元素 a == b，返回 F64 张量（1.0/0.0）。
    pub fn eq(&self, other: &Tensor) -> Result<Tensor, String> {
        self.compare_binary(other, |a, b| a == b, "==")
    }

    /// 逐元素 a != b，返回 F64 张量（1.0/0.0）。
    pub fn ne(&self, other: &Tensor) -> Result<Tensor, String> {
        self.compare_binary(other, |a, b| a != b, "!=")
    }

    /// Scatter values from `src` into a copy of `base` at positions given by
    /// `index` along `dim`（PyTorch scatter_ but immutable，支持任意 dim + 多维 index/src）.
    ///   out = base.clone();
    ///   对每个 multi-index idx（遍历 index）:
    ///     actual = idx; actual[dim] = index[idx] as usize
    ///     out[actual] = src[idx]
    /// - index.ndim() == base.ndim()；除 dim 维外，index.shape == base.shape
    /// - index.shape == src.shape
    /// - index 值范围 [0, base.shape()[dim])
    /// dtype 保留：与 base.dtype 一致（src 自动 cast）。
    pub fn scatter(base: &Tensor, dim: usize, index: &Tensor, src: &Tensor) -> Result<Tensor, String> {
        let base_shape = base.shape();
        if base_shape.is_empty() {
            return Err("scatter: base 必须为非标量张量".into());
        }
        if dim >= base.ndim() {
            return Err(format!(
                "scatter: dim={} 越界（base ndim={}）",
                dim,
                base.ndim()
            ));
        }
        if index.ndim() != base.ndim() {
            return Err(format!(
                "scatter: index.ndim()={} 必须等于 base.ndim()={}",
                index.ndim(),
                base.ndim()
            ));
        }
        // 除 dim 维外，其他维度 shape 与 base 一致
        for d in 0..base.ndim() {
            if d == dim {
                continue;
            }
            if index.shape()[d] != base_shape[d] {
                return Err(format!(
                    "scatter: 维度 {} 上 index shape {} 与 base shape {} 不一致（除 dim 维外必须一致）",
                    d, index.shape()[d], base_shape[d]
                ));
            }
        }
        // index.shape == src.shape
        if index.shape() != src.shape() {
            return Err(format!(
                "scatter: index shape {:?} 与 src shape {:?} 不匹配（必须一致）",
                index.shape(), src.shape()
            ));
        }
        let dim_len = base_shape[dim];
        let index_view = index.data.as_f64_view();
        // 校验 index 值范围（取整数）
        for v in index_view.iter() {
            let idx = *v as i64;
            if idx < 0 || (idx as usize) >= dim_len {
                return Err(format!(
                    "scatter: index 值 {} 越界（base 第 {} 维长度={}）",
                    v, dim, dim_len
                ));
            }
        }

        // out = base.clone()，按 index 散布 src
        let mut out = base.clone();
        let index_shape: Vec<usize> = index.shape().to_vec();
        let total: usize = index_shape.iter().product();
        // 把 flat（row-major / C order）反推为多维索引
        // 与 flatten_index 对偶：从最后一维开始，stride 递增
        let unflatten = |flat: usize| -> Vec<usize> {
            let mut multi = vec![0usize; index_shape.len()];
            let mut rem = flat;
            for i in (0..index_shape.len()).rev() {
                multi[i] = rem % index_shape[i];
                rem /= index_shape[i];
            }
            multi
        };
        let src_view = src.data.as_f64_view();
        match &mut out.data {
            TensorData::F64(a) => {
                for flat in 0..total {
                    let multi = unflatten(flat);
                    let mut actual = multi.clone();
                    let v = index_view[IxDyn(&multi)];
                    actual[dim] = v as usize;
                    let s = src_view[IxDyn(&multi)];
                    a[IxDyn(&actual)] = s;
                }
            }
            TensorData::F32(a) => {
                for flat in 0..total {
                    let multi = unflatten(flat);
                    let mut actual = multi.clone();
                    let v = index_view[IxDyn(&multi)];
                    actual[dim] = v as usize;
                    let s = src_view[IxDyn(&multi)] as f32;
                    a[IxDyn(&actual)] = s;
                }
            }
            TensorData::F16(a) => {
                for flat in 0..total {
                    let multi = unflatten(flat);
                    let mut actual = multi.clone();
                    let v = index_view[IxDyn(&multi)];
                    actual[dim] = v as usize;
                    let s = src_view[IxDyn(&multi)];
                    a[IxDyn(&actual)] = f16::from_f64(s);
                }
            }
            TensorData::BF16(a) => {
                for flat in 0..total {
                    let multi = unflatten(flat);
                    let mut actual = multi.clone();
                    let v = index_view[IxDyn(&multi)];
                    actual[dim] = v as usize;
                    let s = src_view[IxDyn(&multi)];
                    a[IxDyn(&actual)] = bf16::from_f64(s);
                }
            }
        }
        Ok(out)
    }

    /// Gather values from `base` along `dim` at positions given by `index`.
    /// 与 PyTorch gather 对齐：
    ///   out[i,j,...] = base[index[i,j,...], j, ...]  (dim=0)
    ///   out[i,j,...] = base[i, index[i,j,...], ...]  (dim=1)
    /// - out.shape == index.shape
    /// - index.ndim() == base.ndim()；除 dim 维外，其他维度 shape 与 base 一致
    /// - index 必须为整数张量（f64/f32 存储，取整）；值范围 [0, base.shape()[dim])
    /// - dtype 保留：与 base.dtype 一致
    pub fn gather(base: &Tensor, dim: usize, index: &Tensor) -> Result<Tensor, String> {
        let base_shape = base.shape();
        if base_shape.is_empty() {
            return Err("gather: base 必须为非标量张量".into());
        }
        if dim >= base.ndim() {
            return Err(format!(
                "gather: dim={} 越界（base ndim={}）",
                dim,
                base.ndim()
            ));
        }
        if index.ndim() != base.ndim() {
            return Err(format!(
                "gather: index.ndim()={} 必须等于 base.ndim()={}",
                index.ndim(),
                base.ndim()
            ));
        }
        // 除 dim 维外，其他维度 shape 一致
        for d in 0..base.ndim() {
            if d == dim {
                continue;
            }
            if index.shape()[d] != base_shape[d] {
                return Err(format!(
                    "gather: 维度 {} 上 index shape {} 与 base shape {} 不一致（除 dim 维外必须一致）",
                    d, index.shape()[d], base_shape[d]
                ));
            }
        }
        // 校验 index 值范围（取整数）
        let index_view = index.data.as_f64_view();
        let dim_len = base_shape[dim];
        for v in index_view.iter() {
            let idx = *v as i64;
            if idx < 0 || (idx as usize) >= dim_len {
                return Err(format!(
                    "gather: index 值 {} 越界（base 第 {} 维长度={}）",
                    v, dim, dim_len
                ));
            }
        }

        // 构造 out：遍历 index 每个元素，从 base 对应位置取值
        let index_shape: Vec<usize> = index.shape().to_vec();
        let total: usize = index_shape.iter().product();
        // 把 flat（row-major / C order）反推为多维索引
        // 与 flatten_index 对偶：从最后一维开始，stride 递增
        let unflatten = |flat: usize| -> Vec<usize> {
            let mut multi = vec![0usize; index_shape.len()];
            let mut rem = flat;
            for i in (0..index_shape.len()).rev() {
                multi[i] = rem % index_shape[i];
                rem /= index_shape[i];
            }
            multi
        };
        match base.data {
            TensorData::F64(_) => {
                let base_view = base.data.as_f64_view();
                let mut out_data = Vec::with_capacity(total);
                for flat in 0..total {
                    let multi = unflatten(flat);
                    let mut actual = multi.clone();
                    let v = index_view[IxDyn(&multi)];
                    actual[dim] = v as usize;
                    out_data.push(base_view[IxDyn(&actual)]);
                }
                Ok(Tensor::from_vec(out_data, index_shape))
            }
            TensorData::F32(_) => {
                // 保留 f32 dtype：直接从 f32 数组取值
                let base_f32 = base.data.as_f32().ok_or_else(|| {
                    "gather: base dtype 不一致（期望 f32）".to_string()
                })?;
                let mut out_data: Vec<f32> = Vec::with_capacity(total);
                for flat in 0..total {
                    let multi = unflatten(flat);
                    let mut actual = multi.clone();
                    let v = index_view[IxDyn(&multi)];
                    actual[dim] = v as usize;
                    out_data.push(base_f32[IxDyn(&actual)]);
                }
                Ok(Tensor::from_vec_f32(out_data, index_shape))
            }
            TensorData::F16(_) => {
                let base_f16 = base.data.as_f16().ok_or_else(|| {
                    "gather: base dtype 不一致（期望 f16）".to_string()
                })?;
                let mut out_data: Vec<f16> = Vec::with_capacity(total);
                for flat in 0..total {
                    let multi = unflatten(flat);
                    let mut actual = multi.clone();
                    let v = index_view[IxDyn(&multi)];
                    actual[dim] = v as usize;
                    out_data.push(base_f16[IxDyn(&actual)]);
                }
                Ok(Tensor::from_vec_f16(out_data, index_shape))
            }
            TensorData::BF16(_) => {
                let base_bf16 = base.data.as_bf16().ok_or_else(|| {
                    "gather: base dtype 不一致（期望 bf16）".to_string()
                })?;
                let mut out_data: Vec<bf16> = Vec::with_capacity(total);
                for flat in 0..total {
                    let multi = unflatten(flat);
                    let mut actual = multi.clone();
                    let v = index_view[IxDyn(&multi)];
                    actual[dim] = v as usize;
                    out_data.push(base_bf16[IxDyn(&actual)]);
                }
                Ok(Tensor::from_vec_bf16(out_data, index_shape))
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
            TensorData::F16(a) => {
                let result = a.view().permuted_axes(dims.to_vec()).to_owned();
                Ok(Tensor::from_data_f16(result))
            }
            TensorData::BF16(a) => {
                let result = a.view().permuted_axes(dims.to_vec()).to_owned();
                Ok(Tensor::from_data_bf16(result))
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
            // Wave 2: F16/BF16 softmax 转 f64 计算，结果转回原 dtype
            TensorData::F16(a) => {
                let contiguous = a.as_standard_layout().to_owned();
                let flat = contiguous.as_slice()?;
                let mut result_data: Vec<f16> = Vec::with_capacity(flat.len());
                for i in 0..outer_len {
                    let start = i * axis_len;
                    let slice = &flat[start..start + axis_len];
                    let max_val = slice.iter().map(|v| v.to_f64()).fold(f64::NEG_INFINITY, f64::max);
                    let exps: Vec<f64> = slice.iter().map(|x| (x.to_f64() - max_val).exp()).collect();
                    let sum: f64 = exps.iter().sum();
                    let probs: Vec<f16> = if sum == 0.0 {
                        vec![f16::from_f64(1.0 / axis_len as f64); axis_len]
                    } else {
                        exps.iter().map(|x| f16::from_f64(x / sum)).collect()
                    };
                    result_data.extend(probs);
                }
                Some(Tensor::from_vec_f16(result_data, shape))
            }
            TensorData::BF16(a) => {
                let contiguous = a.as_standard_layout().to_owned();
                let flat = contiguous.as_slice()?;
                let mut result_data: Vec<bf16> = Vec::with_capacity(flat.len());
                for i in 0..outer_len {
                    let start = i * axis_len;
                    let slice = &flat[start..start + axis_len];
                    let max_val = slice.iter().map(|v| v.to_f64()).fold(f64::NEG_INFINITY, f64::max);
                    let exps: Vec<f64> = slice.iter().map(|x| (x.to_f64() - max_val).exp()).collect();
                    let sum: f64 = exps.iter().sum();
                    let probs: Vec<bf16> = if sum == 0.0 {
                        vec![bf16::from_f64(1.0 / axis_len as f64); axis_len]
                    } else {
                        exps.iter().map(|x| bf16::from_f64(x / sum)).collect()
                    };
                    result_data.extend(probs);
                }
                Some(Tensor::from_vec_bf16(result_data, shape))
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
            // Wave 2: F16/BF16 im2col 转 f64 计算（Phase 1 简化）
            TensorData::F16(a) => {
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
                                            cols.push(slice.get(idx).map(|v| v.to_f64()).unwrap_or(0.0));
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
            TensorData::BF16(a) => {
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
                                            cols.push(slice.get(idx).map(|v| v.to_f64()).unwrap_or(0.0));
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
        }
    }

    // ── pool2d ───────────────────────────────────────────────────────

    /// 池化输出维度计算（PyTorch 语义，向下取整）：
    /// `H_out = (H + 2*pad - kernel) / stride + 1`
    fn pool_out_dim(input: usize, kernel: usize, stride: usize, pad: usize) -> usize {
        (input + 2 * pad).saturating_sub(kernel) / stride + 1
    }

    /// MaxPool2D 前向 + argmax mask。
    /// 输入：4D NCHW。返回 `(output, argmax_mask)`：
    /// - `output` shape = `[N, C, H_out, W_out]`，dtype 与输入一致（F16/BF16 转 f64 计算后按输入 dtype 输出）。
    /// - `argmax_mask` shape = `[N, C, H, W]`（与输入同 shape），每个 window 的 argmax 位置标 1。
    ///
    /// 语义对齐 PyTorch：同值取第一个（row-major 顺序，严格 `>` 才更新 argmax）。
    /// padding 位置既不参与 max 比较也不参与 argmax（视为 -∞）。
    pub fn max_pool2d_with_argmax(
        &self,
        kernel_h: usize, kernel_w: usize,
        stride_h: usize, stride_w: usize,
        padding_h: usize, padding_w: usize,
    ) -> Result<(Tensor, Tensor), String> {
        let shape = self.shape();
        if shape.len() != 4 {
            return Err(format!("max_pool2d 需要 4D NCHW 输入，得到 {}D", shape.len()));
        }
        let (n, c, h, w) = (shape[0], shape[1], shape[2], shape[3]);
        let h_out = Self::pool_out_dim(h, kernel_h, stride_h, padding_h);
        let w_out = Self::pool_out_dim(w, kernel_w, stride_w, padding_w);

        // 统一转 f64 计算（F16/BF16 Phase 1 简化）
        let x_f64: Vec<f64> = match &self.data {
            TensorData::F64(a) => a.iter().copied().collect(),
            TensorData::F32(a) => a.iter().map(|v| *v as f64).collect(),
            TensorData::F16(a) => a.iter().map(|v| v.to_f64()).collect(),
            TensorData::BF16(a) => a.iter().map(|v| v.to_f64()).collect(),
        };
        let mut out_f64: Vec<f64> = vec![0.0; n * c * h_out * w_out];
        let mut mask_f64: Vec<f64> = vec![0.0; n * c * h * w];
        for ni in 0..n {
            for ci in 0..c {
                for ho in 0..h_out {
                    for wo in 0..w_out {
                        let mut best_val = f64::NEG_INFINITY;
                        let mut best_idx: Option<(usize, usize)> = None;
                        for kh in 0..kernel_h {
                            let ih = ho * stride_h + kh;
                            if ih < padding_h { continue; }
                            let ih_adj = ih - padding_h;
                            if ih_adj >= h { continue; }
                            for kw in 0..kernel_w {
                                let iw = wo * stride_w + kw;
                                if iw < padding_w { continue; }
                                let iw_adj = iw - padding_w;
                                if iw_adj >= w { continue; }
                                let in_idx = ((ni * c + ci) * h + ih_adj) * w + iw_adj;
                                let v = x_f64[in_idx];
                                if v > best_val {
                                    best_val = v;
                                    best_idx = Some((ih_adj, iw_adj));
                                }
                            }
                        }
                        let out_idx = ((ni * c + ci) * h_out + ho) * w_out + wo;
                        out_f64[out_idx] = best_val;
                        if let Some((ih_adj, iw_adj)) = best_idx {
                            let mask_idx = ((ni * c + ci) * h + ih_adj) * w + iw_adj;
                            mask_f64[mask_idx] = 1.0;
                        }
                    }
                }
            }
        }
        // 按输入 dtype 构造输出（F16/BF16 输入也产出 F64，Phase 1 简化）
        let out_tensor = match self.dtype {
            BaseType::F32 => {
                let data: Vec<f32> = out_f64.iter().map(|v| *v as f32).collect();
                Tensor::from_vec_f32(data, vec![n, c, h_out, w_out])
            }
            _ => Tensor::from_vec(out_f64, vec![n, c, h_out, w_out]),
        };
        // mask 统一用 F64（仅供 backward 内部使用，backward 已统一转 f64 路径）
        let mask_tensor = Tensor::from_vec(mask_f64, vec![n, c, h, w]);
        Ok((out_tensor, mask_tensor))
    }

    /// MaxPool2D 前向（仅输出，丢弃 argmax mask）。
    pub fn max_pool2d(
        &self,
        kernel_h: usize, kernel_w: usize,
        stride_h: usize, stride_w: usize,
        padding_h: usize, padding_w: usize,
    ) -> Result<Tensor, String> {
        let (out, _mask) = self.max_pool2d_with_argmax(
            kernel_h, kernel_w, stride_h, stride_w, padding_h, padding_w,
        )?;
        Ok(out)
    }

    /// AvgPool2D 前向。
    /// `count_include_pad=False`：分母为 `valid_count`（window 内非 padding 位置数）。
    /// padding 位置既不计入分子也不计入分母。
    /// F16/BF16 转 f64 计算（Phase 1 简化）；结果按输入 dtype 构造。
    pub fn avg_pool2d(
        &self,
        kernel_h: usize, kernel_w: usize,
        stride_h: usize, stride_w: usize,
        padding_h: usize, padding_w: usize,
    ) -> Result<Tensor, String> {
        let shape = self.shape();
        if shape.len() != 4 {
            return Err(format!("avg_pool2d 需要 4D NCHW 输入，得到 {}D", shape.len()));
        }
        let (n, c, h, w) = (shape[0], shape[1], shape[2], shape[3]);
        let h_out = Self::pool_out_dim(h, kernel_h, stride_h, padding_h);
        let w_out = Self::pool_out_dim(w, kernel_w, stride_w, padding_w);

        let x_f64: Vec<f64> = match &self.data {
            TensorData::F64(a) => a.iter().copied().collect(),
            TensorData::F32(a) => a.iter().map(|v| *v as f64).collect(),
            TensorData::F16(a) => a.iter().map(|v| v.to_f64()).collect(),
            TensorData::BF16(a) => a.iter().map(|v| v.to_f64()).collect(),
        };
        let mut out_f64: Vec<f64> = vec![0.0; n * c * h_out * w_out];
        for ni in 0..n {
            for ci in 0..c {
                for ho in 0..h_out {
                    for wo in 0..w_out {
                        let mut sum = 0.0f64;
                        let mut valid_count: usize = 0;
                        for kh in 0..kernel_h {
                            let ih = ho * stride_h + kh;
                            if ih < padding_h { continue; }
                            let ih_adj = ih - padding_h;
                            if ih_adj >= h { continue; }
                            for kw in 0..kernel_w {
                                let iw = wo * stride_w + kw;
                                if iw < padding_w { continue; }
                                let iw_adj = iw - padding_w;
                                if iw_adj >= w { continue; }
                                let in_idx = ((ni * c + ci) * h + ih_adj) * w + iw_adj;
                                sum += x_f64[in_idx];
                                valid_count += 1;
                            }
                        }
                        let out_idx = ((ni * c + ci) * h_out + ho) * w_out + wo;
                        out_f64[out_idx] = if valid_count > 0 {
                            sum / (valid_count as f64)
                        } else {
                            0.0
                        };
                    }
                }
            }
        }
        let out_tensor = match self.dtype {
            BaseType::F32 => {
                let data: Vec<f32> = out_f64.iter().map(|v| *v as f32).collect();
                Tensor::from_vec_f32(data, vec![n, c, h_out, w_out])
            }
            _ => Tensor::from_vec(out_f64, vec![n, c, h_out, w_out]),
        };
        Ok(out_tensor)
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
                (TensorData::F16(s), TensorData::F16(o)) => {
                    s.zip_mut_with(o, |s, &x| *s = x);
                    return Ok(());
                }
                (TensorData::BF16(s), TensorData::BF16(o)) => {
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
                        TensorData::F16(s) => {
                            let o_f16 = o_view.mapv(|v| f16::from_f64(v));
                            s.zip_mut_with(&o_f16, |s, &x| *s = x);
                            return Ok(());
                        }
                        TensorData::BF16(s) => {
                            let o_bf16 = o_view.mapv(|v| bf16::from_f64(v));
                            s.zip_mut_with(&o_bf16, |s, &x| *s = x);
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
                TensorData::F16(s) => {
                    let src_br_f16: ArrayD<f16> = src_br.mapv(|v| f16::from_f64(v));
                    s.zip_mut_with(&src_br_f16, |s, &x| *s = x);
                    return Ok(());
                }
                TensorData::BF16(s) => {
                    let src_br_bf16: ArrayD<bf16> = src_br.mapv(|v| bf16::from_f64(v));
                    s.zip_mut_with(&src_br_bf16, |s, &x| *s = x);
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
