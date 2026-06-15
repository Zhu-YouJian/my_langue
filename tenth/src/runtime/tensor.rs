use ndarray::{ArrayD, IxDyn};
use std::fmt;

#[derive(Debug, Clone)]
pub struct Tensor {
    pub data: ArrayD<f64>,
    /// Accumulated gradient (populated by autodiff backward pass).
    pub grad: Option<ArrayD<f64>>,
    /// Tape node id set by the interpreter during recording mode.
    /// Used to link tensors back to their computation-graph nodes.
    pub tape_id: Option<usize>,
}

impl Tensor {
    // ── helpers ────────────────────────────────────────────────────────

    pub(crate) fn from_data(data: ArrayD<f64>) -> Self {
        Tensor { data, grad: None, tape_id: None }
    }

    /// Zero-initialise the gradient buffer to match the tensor shape.
    pub fn zero_grad(&mut self) {
        self.grad = None;
    }

    /// Accumulate `g` into `self.grad` (broadcasting if needed).
    pub fn acc_grad(&mut self, g: &ArrayD<f64>) {
        match &mut self.grad {
            Some(cur) => {
                *cur = &*cur + g;
            }
            None => {
                self.grad = Some(g.clone());
            }
        }
    }

    // ── constructors ───────────────────────────────────────────────────

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
        self.data.get(IxDyn(index)).copied()
    }

    // ── reductions ─────────────────────────────────────────────────────

    pub fn sum(&self) -> f64 {
        self.data.sum()
    }

    pub fn sum_axis(&self, axis: usize) -> Tensor {
        Tensor::from_data(self.data.sum_axis(ndarray::Axis(axis)))
    }

    pub fn mean(&self) -> f64 {
        self.data.mean().unwrap_or(0.0)
    }

    pub fn max_val(&self) -> f64 {
        *self.data.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap_or(&0.0)
    }

    // ── scalar ops (keep for compatibility) ────────────────────────────

    pub fn add_scalar(&self, scalar: f64) -> Tensor {
        Tensor::from_data(&self.data + scalar)
    }

    pub fn sub_scalar(&self, scalar: f64) -> Tensor {
        Tensor::from_data(&self.data - scalar)
    }

    pub fn mul_scalar(&self, scalar: f64) -> Tensor {
        Tensor::from_data(&self.data * scalar)
    }

    pub fn div_scalar(&self, scalar: f64) -> Tensor {
        Tensor::from_data(&self.data / scalar)
    }

    /// Scalar divided by tensor: scalar / self (element-wise).
    pub fn div_scalar_inv(&self, scalar: f64) -> Tensor {
        Tensor::from_data(scalar / &self.data)
    }

    // ── tensor-tensor element-wise ops (with broadcasting) ─────────────

    /// Element-wise addition with broadcasting.  Errors if shapes are incompatible.
    pub fn add_tensor(&self, other: &Tensor) -> Result<Tensor, String> {
        let a = self.data.view();
        let b = other.data.view();
        if a.broadcast(b.shape()).is_some() {
            let a_br = a.broadcast(b.shape()).unwrap();
            Ok(Tensor::from_data(a_br.to_owned() + &b))
        } else if b.broadcast(a.shape()).is_some() {
            let b_br = b.broadcast(a.shape()).unwrap();
            Ok(Tensor::from_data(&a + b_br.to_owned()))
        } else {
            Err(format!("cannot broadcast shapes {:?} + {:?}", self.shape(), other.shape()))
        }
    }

    pub fn sub_tensor(&self, other: &Tensor) -> Result<Tensor, String> {
        let a = self.data.view();
        let b = other.data.view();
        if a.broadcast(b.shape()).is_some() {
            let a_br = a.broadcast(b.shape()).unwrap();
            Ok(Tensor::from_data(a_br.to_owned() - &b))
        } else if b.broadcast(a.shape()).is_some() {
            let b_br = b.broadcast(a.shape()).unwrap();
            Ok(Tensor::from_data(&a - b_br.to_owned()))
        } else {
            Err(format!("cannot broadcast shapes {:?} - {:?}", self.shape(), other.shape()))
        }
    }

    pub fn mul_tensor(&self, other: &Tensor) -> Result<Tensor, String> {
        let a = self.data.view();
        let b = other.data.view();
        if a.broadcast(b.shape()).is_some() {
            let a_br = a.broadcast(b.shape()).unwrap();
            Ok(Tensor::from_data(a_br.to_owned() * &b))
        } else if b.broadcast(a.shape()).is_some() {
            let b_br = b.broadcast(a.shape()).unwrap();
            Ok(Tensor::from_data(&a * b_br.to_owned()))
        } else {
            Err(format!("cannot broadcast shapes {:?} * {:?}", self.shape(), other.shape()))
        }
    }

    pub fn div_tensor(&self, other: &Tensor) -> Result<Tensor, String> {
        let a = self.data.view();
        let b = other.data.view();
        if a.broadcast(b.shape()).is_some() {
            let a_br = a.broadcast(b.shape()).unwrap();
            Ok(Tensor::from_data(a_br.to_owned() / &b))
        } else if b.broadcast(a.shape()).is_some() {
            let b_br = b.broadcast(a.shape()).unwrap();
            Ok(Tensor::from_data(&a / b_br.to_owned()))
        } else {
            Err(format!("cannot broadcast shapes {:?} / {:?}", self.shape(), other.shape()))
        }
    }

    // ── matrix multiplication ─────────────────────────────────────────

    /// Matrix multiplication.  Supports 2D @ 2D and 1D @ 2D / 2D @ 1D.
    pub fn matmul(&self, other: &Tensor) -> Result<Tensor, String> {
        let a_ndim = self.ndim();
        let b_ndim = other.ndim();

        if a_ndim == 2 && b_ndim == 2 {
            let a = self.data.view().into_dimensionality::<ndarray::Ix2>()
                .map_err(|_| "matmul: self must be 2D")?;
            let b = other.data.view().into_dimensionality::<ndarray::Ix2>()
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
            // vector @ matrix  →  broadcast row-vector to match
            let a = self.data.view().into_dimensionality::<ndarray::Ix1>()
                .map_err(|_| "matmul: self must be 1D")?;
            let b = other.data.view().into_dimensionality::<ndarray::Ix2>()
                .map_err(|_| "matmul: other must be 2D")?;
            if a.shape()[0] != b.shape()[0] {
                return Err(format!("matmul shape mismatch: {:?} @ {:?}", a.shape(), b.shape()));
            }
            // a (k,) → (1,k); b (k,n); result (1,n) → (n,)
            let a_2d = a.insert_axis(ndarray::Axis(0)); // (1, k)
            let result = a_2d.dot(&b);
            let squeezed = result.index_axis_move(ndarray::Axis(0), 0);
            Ok(Tensor::from_data(squeezed.into_dyn()))
        } else if a_ndim == 2 && b_ndim == 1 {
            let a = self.data.view().into_dimensionality::<ndarray::Ix2>()
                .map_err(|_| "matmul: self must be 2D")?;
            let b = other.data.view().into_dimensionality::<ndarray::Ix1>()
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

    // ── transpose ─────────────────────────────────────────────────────

    /// Transpose last two dimensions.  For 2D tensors this is the usual matrix transpose.
    pub fn transpose(&self) -> Option<Tensor> {
        if self.ndim() < 2 {
            return None;
        }
        let mut perm: Vec<usize> = (0..self.ndim()).collect();
        let last = perm.len() - 1;
        perm.swap(last - 1, last);
        let result = self.data.view().permuted_axes(perm);
        Some(Tensor::from_data(result.to_owned()))
    }

    // ── broadcasting / reshaping ───────────────────────────────────────

    /// Broadcast to a target shape.  Returns None if not broadcastable.
    pub fn broadcast_to(&self, target_shape: &[usize]) -> Option<Tensor> {
        let view = self.data.view();
        let broadcasted = view.broadcast(target_shape)?;
        Some(Tensor::from_data(broadcasted.to_owned()))
    }

    // ── unary elementwise ──────────────────────────────────────────────

    pub fn neg(&self) -> Tensor {
        Tensor::from_data(-&self.data)
    }

    pub fn abs(&self) -> Tensor {
        Tensor::from_data(self.data.mapv(|x| x.abs()))
    }

    pub fn sqrt(&self) -> Tensor {
        Tensor::from_data(self.data.mapv(|x| x.sqrt()))
    }

    pub fn exp(&self) -> Tensor {
        Tensor::from_data(self.data.mapv(|x| x.exp()))
    }

    pub fn log(&self) -> Tensor {
        Tensor::from_data(self.data.mapv(|x| x.ln()))
    }

    pub fn relu(&self) -> Tensor {
        Tensor::from_data(self.data.mapv(|x| if x > 0.0 { x } else { 0.0 }))
    }

    pub fn sigmoid(&self) -> Tensor {
        Tensor::from_data(self.data.mapv(|x| 1.0 / (1.0 + (-x).exp())))
    }

    pub fn tanh(&self) -> Tensor {
        Tensor::from_data(self.data.mapv(|x| x.tanh()))
    }

    pub fn reshape(&self, shape: &[usize]) -> Option<Tensor> {
        let array = self.data.clone().into_shape_with_order(IxDyn(shape)).ok()?;
        Some(Tensor::from_data(array))
    }

    pub fn flatten(&self) -> Tensor {
        let size = self.data.len();
        let array = self.data.clone().into_shape_with_order(IxDyn(&[size])).unwrap();
        Tensor::from_data(array)
    }

    // ── Transformer ops ──────────────────────────────────────────────

    /// Layer normalization along the last dimension.
    /// x: (..., D), gamma: (D,), beta: (D,), eps: f64
    /// Returns (x - mean) / sqrt(var + eps) * gamma + beta
    pub fn layer_norm(&self, gamma: &Tensor, beta: &Tensor, eps: f64) -> Tensor {
        let shape = self.shape();
        let ndim = shape.len();
        if ndim == 0 || shape[ndim - 1] == 0 {
            return self.clone();
        }
        let axis_len = shape[ndim - 1];
        let outer_len: usize = shape[..ndim - 1].iter().product();

        let contiguous = self.data.as_standard_layout().to_owned();
        let flat = match contiguous.as_slice() {
            Some(s) => s.to_vec(),
            None => self.data.iter().cloned().collect(),
        };

        let g_flat = gamma.data.as_standard_layout().to_owned();
        let b_flat = beta.data.as_standard_layout().to_owned();
        let g_slice = g_flat.as_slice().unwrap_or(&[]);
        let b_slice = b_flat.as_slice().unwrap_or(&[]);

        let mut result_data = Vec::with_capacity(flat.len());
        for i in 0..outer_len {
            let start = i * axis_len;
            let slice = &flat[start..start + axis_len];
            let mean: f64 = slice.iter().sum::<f64>() / axis_len as f64;
            let var: f64 = slice.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / axis_len as f64;
            let std_inv = 1.0 / (var + eps).sqrt();
            for j in 0..axis_len {
                let x_hat = (slice[j] - mean) * std_inv;
                let g = g_slice.get(j).copied().unwrap_or(1.0);
                let b = b_slice.get(j).copied().unwrap_or(0.0);
                result_data.push(g * x_hat + b);
            }
        }
        Tensor::from_vec(result_data, shape)
    }

    /// GELU activation (tanh approximation).
    /// 0.5 * x * (1.0 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    pub fn gelu(&self) -> Tensor {
        let sqrt_2_over_pi = (2.0 / std::f64::consts::PI).sqrt();
        Tensor::from_data(self.data.mapv(|x| {
            let inner = sqrt_2_over_pi * (x + 0.044715 * x * x * x);
            0.5 * x * (1.0 + inner.tanh())
        }))
    }

    /// Concatenate two 2D tensors along the given dimension.
    /// Only supports dim=0 or dim=1 for 2D tensors.
    pub fn cat(&self, other: &Tensor, dim: usize) -> Result<Tensor, String> {
        let a_shape = self.shape();
        let b_shape = other.shape();
        if a_shape.len() != 2 || b_shape.len() != 2 {
            return Err(format!("cat only supports 2D tensors, got {:?}D and {:?}D", a_shape.len(), b_shape.len()));
        }
        match dim {
            0 => {
                if a_shape[1] != b_shape[1] {
                    return Err(format!("cat dim=0: column mismatch {} vs {}", a_shape[1], b_shape[1]));
                }
                let a_flat = self.data.as_standard_layout().to_owned();
                let b_flat = other.data.as_standard_layout().to_owned();
                let a_slice = a_flat.as_slice().unwrap_or(&[]);
                let b_slice = b_flat.as_slice().unwrap_or(&[]);
                let mut data = Vec::with_capacity(a_slice.len() + b_slice.len());
                data.extend_from_slice(a_slice);
                data.extend_from_slice(b_slice);
                Ok(Tensor::from_vec(data, vec![a_shape[0] + b_shape[0], a_shape[1]]))
            }
            1 => {
                if a_shape[0] != b_shape[0] {
                    return Err(format!("cat dim=1: row mismatch {} vs {}", a_shape[0], b_shape[0]));
                }
                let a_flat = self.data.as_standard_layout().to_owned();
                let b_flat = other.data.as_standard_layout().to_owned();
                let a_slice = a_flat.as_slice().unwrap_or(&[]);
                let b_slice = b_flat.as_slice().unwrap_or(&[]);
                let mut data = Vec::with_capacity(a_slice.len() + b_slice.len());
                for i in 0..a_shape[0] {
                    let a_start = i * a_shape[1];
                    data.extend_from_slice(&a_slice[a_start..a_start + a_shape[1]]);
                    let b_start = i * b_shape[1];
                    data.extend_from_slice(&b_slice[b_start..b_start + b_shape[1]]);
                }
                Ok(Tensor::from_vec(data, vec![a_shape[0], a_shape[1] + b_shape[1]]))
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
        let mask_data = mask.data.as_standard_layout().to_owned();
        let mask_slice = mask_data.as_slice().unwrap_or(&[]);
        let self_data = self.data.as_standard_layout().to_owned();
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

    /// Permute dimensions. Only supports 2D/3D/4D tensors.
    pub fn permute(&self, dims: &[usize]) -> Result<Tensor, String> {
        let ndim = self.ndim();
        if dims.len() != ndim {
            return Err(format!("permute: expected {} dims, got {}", ndim, dims.len()));
        }
        if ndim < 2 || ndim > 4 {
            return Err(format!("permute only supports 2D/3D/4D tensors, got {}D", ndim));
        }
        // Validate dims is a valid permutation
        let mut sorted = dims.to_vec();
        sorted.sort();
        let expected: Vec<usize> = (0..ndim).collect();
        if sorted != expected {
            return Err(format!("permute: {:?} is not a valid permutation of {} dims", dims, ndim));
        }
        let result = self.data.view().permuted_axes(dims.to_vec()).to_owned();
        Ok(Tensor::from_data(result))
    }

    /// Softmax along the last axis (per-row for 2D).
    /// For a tensor of shape [..., N], each slice along the last dimension
    /// is independently softmaxed.
    pub fn softmax(&self) -> Option<Tensor> {
        let shape = self.shape().to_vec();
        let ndim = shape.len();
        if ndim == 0 || shape[ndim - 1] == 0 {
            return None;
        }
        let axis_len = shape[ndim - 1];
        let outer_len: usize = shape[..ndim - 1].iter().product();

        // Ensure contiguous for safe slicing
        let contiguous = self.data.as_standard_layout().to_owned();
        let flat = contiguous.as_slice()?;

        let mut result_data = Vec::with_capacity(flat.len());
        for i in 0..outer_len {
            let start = i * axis_len;
            let slice = &flat[start..start + axis_len];
            let max_val = slice.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
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

    // ── conv2d helpers ───────────────────────────────────────────────

    /// im2col: extract sliding windows from a 4D tensor (N, C, H, W)
    /// into a 2D matrix (N*H_out*W_out, C*K_H*K_W).
    /// Returns (col_matrix, output_height, output_width).
    pub fn im2col(&self, kernel_h: usize, kernel_w: usize, stride: usize, pad: usize) -> Option<(Tensor, usize, usize)> {
        let shape = self.shape();
        if shape.len() != 4 { return None; }
        let (n, c, h, w) = (shape[0], shape[1], shape[2], shape[3]);
        let h_out = (h + 2 * pad - kernel_h) / stride + 1;
        let w_out = (w + 2 * pad - kernel_w) / stride + 1;

        let mut cols = Vec::with_capacity(n * h_out * w_out * c * kernel_h * kernel_w);
        let flat = self.data.as_standard_layout().to_owned();
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
                                    cols.push(0.0); // zero padding
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

    /// In-place element-wise assignment: `self[i..] = src`.
    /// This mutates the underlying ArrayD in-place.
    pub fn assign_(&mut self, src: &Tensor) {
        // Use zip_mut_with for element-wise assignment with broadcasting
        if self.shape() == src.shape() {
            self.data.zip_mut_with(&src.data, |s, &x| *s = x);
        } else if let Some(src_br) = src.data.view().broadcast(self.shape().as_slice()) {
            self.data.zip_mut_with(&src_br, |s, &x| *s = x);
        }
        // otherwise no-op (shapes incompatible)
    }
}

impl fmt::Display for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.data)
    }
}