use crate::hir::types::{Type, Dim};

/// Shape inference engine for tensor operations.
pub struct ShapeEngine;

impl ShapeEngine {
    pub fn new() -> Self { ShapeEngine }

    /// Infer the result shape of a binary tensor operation (element-wise).
    pub fn infer_binary(&self, left: &Type, right: &Type) -> Type {
        match (left, right) {
            (Type::Tensor { dtype, dims }, Type::Base(_)) => {
                Type::Tensor { dtype: dtype.clone(), dims: dims.clone() }
            }
            (Type::Base(_), Type::Tensor { dtype, dims }) => {
                Type::Tensor { dtype: dtype.clone(), dims: dims.clone() }
            }
            (Type::Tensor { dtype, dims: ldims }, Type::Tensor { dims: rdims, .. }) => {
                let result_dims = self.broadcast_shapes(ldims, rdims);
                Type::Tensor { dtype: dtype.clone(), dims: result_dims }
            }
            _ => left.clone(),
        }
    }

    /// Broadcast two shape lists.
    fn broadcast_shapes(&self, a: &[Dim], b: &[Dim]) -> Vec<Dim> {
        let max_len = a.len().max(b.len());
        let a_padded = pad_dims(a, max_len);
        let b_padded = pad_dims(b, max_len);

        a_padded.iter().zip(b_padded.iter()).map(|(da, db)| {
            match (da, db) {
                (Dim::Known(x), Dim::Known(y)) if x == y => Dim::Known(*x),
                (Dim::Known(1), other) | (other, Dim::Known(1)) => other.clone(),
                (Dim::Known(_), Dim::Known(_)) => Dim::Any, // mismatched
                (Dim::Any, d) | (d, Dim::Any) => d.clone(),
                (Dim::Symbol(s), _) | (_, Dim::Symbol(s)) => Dim::Symbol(s.clone()),
            }
        }).collect()
    }

    /// Infer matmul result shape: (M,K) × (K,N) → (M,N).
    pub fn infer_matmul(&self, left: &Type, right: &Type) -> Option<Type> {
        match (left, right) {
            (Type::Tensor { dtype, dims: ldims }, Type::Tensor { dims: rdims, .. }) => {
                let llen = ldims.len();
                let rlen = rdims.len();
                if llen < 2 || rlen < 2 {
                    return None;
                }
                let _lr = &ldims[llen - 1];
                let ll = &ldims[llen - 2];
                let rr = &rdims[rlen - 1];
                let rl = &rdims[rlen - 2];

                match (ll, rl) {
                    (Dim::Known(k1), Dim::Known(k2)) if k1 != k2 => return None,
                    _ => {}
                }

                let mut result_dims = Vec::new();
                for _i in 0..(llen - 2).max(rlen - 2) {
                    result_dims.push(Dim::Any);
                }
                result_dims.push(ll.clone());
                result_dims.push(rr.clone());

                Some(Type::Tensor { dtype: dtype.clone(), dims: result_dims })
            }
            _ => None,
        }
    }
}

fn pad_dims(dims: &[Dim], target_len: usize) -> Vec<Dim> {
    let pad = target_len - dims.len();
    let mut result = vec![Dim::Known(1); pad];
    result.extend_from_slice(dims);
    result
}
