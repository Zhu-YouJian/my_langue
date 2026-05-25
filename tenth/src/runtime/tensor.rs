use ndarray::{ArrayD, IxDyn};
use std::fmt;

#[derive(Debug, Clone)]
pub struct Tensor {
    data: ArrayD<f64>,
}

impl Tensor {
    pub fn from_vec(data: Vec<f64>, shape: Vec<usize>) -> Self {
        let array = ArrayD::from_shape_vec(IxDyn(&shape), data)
            .expect("invalid tensor shape");
        Tensor { data: array }
    }

    pub fn zeros(shape: &[usize]) -> Self {
        let array = ArrayD::zeros(IxDyn(shape));
        Tensor { data: array }
    }

    pub fn ones(shape: &[usize]) -> Self {
        let array = ArrayD::ones(IxDyn(shape));
        Tensor { data: array }
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
        let array = ArrayD::from_elem(IxDyn(shape), value);
        Tensor { data: array }
    }

    pub fn eye(n: usize) -> Self {
        let mut array = ArrayD::zeros(IxDyn(&[n, n]));
        for i in 0..n {
            array[[i, i]] = 1.0;
        }
        Tensor { data: array }
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

    pub fn shape(&self) -> Vec<usize> {
        self.data.shape().to_vec()
    }

    pub fn get(&self, index: &[usize]) -> Option<f64> {
        self.data.get(IxDyn(index)).copied()
    }

    pub fn sum(&self) -> f64 {
        self.data.sum()
    }

    pub fn sum_axis(&self, axis: usize) -> Tensor {
        let summed = self.data.sum_axis(ndarray::Axis(axis));
        Tensor { data: summed }
    }

    pub fn mean(&self) -> f64 {
        self.data.mean().unwrap_or(0.0)
    }

    pub fn add_scalar(&self, scalar: f64) -> Tensor {
        Tensor { data: &self.data + scalar }
    }

    pub fn sub_scalar(&self, scalar: f64) -> Tensor {
        Tensor { data: &self.data - scalar }
    }

    pub fn mul_scalar(&self, scalar: f64) -> Tensor {
        Tensor { data: &self.data * scalar }
    }

    pub fn div_scalar(&self, scalar: f64) -> Tensor {
        Tensor { data: &self.data / scalar }
    }

    pub fn neg(&self) -> Tensor {
        Tensor { data: -&self.data }
    }

    pub fn abs(&self) -> Tensor {
        Tensor { data: self.data.mapv(|x| x.abs()) }
    }

    pub fn sqrt(&self) -> Tensor {
        Tensor { data: self.data.mapv(|x| x.sqrt()) }
    }

    pub fn exp(&self) -> Tensor {
        Tensor { data: self.data.mapv(|x| x.exp()) }
    }

    pub fn log(&self) -> Tensor {
        Tensor { data: self.data.mapv(|x| x.ln()) }
    }

    pub fn relu(&self) -> Tensor {
        Tensor { data: self.data.mapv(|x| if x > 0.0 { x } else { 0.0 }) }
    }

    pub fn sigmoid(&self) -> Tensor {
        Tensor { data: self.data.mapv(|x| 1.0 / (1.0 + (-x).exp())) }
    }

    pub fn tanh(&self) -> Tensor {
        Tensor { data: self.data.mapv(|x| x.tanh()) }
    }

    pub fn reshape(&self, shape: &[usize]) -> Option<Tensor> {
        let array = self.data.clone().into_shape_with_order(IxDyn(shape)).ok()?;
        Some(Tensor { data: array })
    }

    pub fn flatten(&self) -> Tensor {
        let size = self.data.len();
        let array = self.data.clone().into_shape_with_order(IxDyn(&[size])).unwrap();
        Tensor { data: array }
    }

    pub fn softmax(&self) -> Option<Tensor> {
        let shape = self.shape().to_vec();
        let self_data = self.data.as_slice().unwrap_or(&[]);
        let max_val = self_data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exps: Vec<f64> = self_data.iter().map(|x| (x - max_val).exp()).collect();
        let sum: f64 = exps.iter().sum();
        let result: Vec<f64> = exps.iter().map(|x| x / sum).collect();
        Some(Tensor::from_vec(result, shape))
    }
}

impl fmt::Display for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.data)
    }
}