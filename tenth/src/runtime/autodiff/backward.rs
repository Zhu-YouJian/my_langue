//! 反向传播实现：`impl Tape::backward` + 专用辅助函数。
//!
//! 从 `autodiff.rs` 拆分而来（T3c 架构重构），保持原有可见性与语义不变。
//! `backward` 方法原本是 `impl Tape` 的 `pub fn`，拆分后保持 `pub` 不变。
//! 辅助函数（`unbroadcast`/`matmul_2d`/`flatten_index`）原本是模块私有 `fn`，
//! 拆分后改为 `pub(super) fn` 以便本文件内部和未来扩展使用。

use ndarray::{ArrayD, IxDyn};
use super::tape_op::{FloatElem, TapeOp, dispatch_float};
use super::grad::{op_name, propagate_grad};
use crate::runtime::tensor::{Tensor, TensorData};
use crate::hir::types::BaseType;
use super::Tape;
use crate::error::TapeErrorContext;

impl Tape {
    // ── backward pass ─────────────────────────────────────────────────

    /// Run backward pass starting from `loss_node_id`.
    /// Writes gradients into the `.grad` field of every `TapeOp::Input` tensor.
    /// 返回 Err 当反向传播发现 shape 不匹配（方向 A：消除 silent squeeze）。
    ///
    /// 阶段 4：按 node.dtype 分发到 f32/f64 路径，实现真正的 f32 反向传播。
    /// 混合 dtype 场景（前向 f32+f64）回退为 f64（策略 B 兜底）。
    pub fn backward(&self, loss_node_id: usize) -> Result<(), crate::error::TenthError> {
        let n = self.nodes.len();
        // Per-node upstream gradient (Option so we can .take() it).
        // 阶段 4：node_grads 改为 Vec<Option<TensorData>>，按 node.dtype 存储。
        let mut node_grads: Vec<Option<TensorData>> = vec![None; n];

        // Seed: ∂loss/∂loss = 1 (or ones if loss is a tensor).
        // The result tensor is always the LAST entry in input_tensors.
        let result_idx = self.nodes[loss_node_id].input_tensors.len() - 1;
        let (loss_shape, loss_dtype) = {
            let loss_tensor = &self.nodes[loss_node_id].input_tensors[result_idx].borrow();
            (loss_tensor.shape(), loss_tensor.dtype)
        };
        // 种子梯度按 loss tensor 的 dtype 构造。
        // Phase 2：F16/BF16 loss 使用 F32 种子（AMP 策略，F32 中间表示）。
        let seed = match loss_dtype {
            BaseType::F32 | BaseType::F16 | BaseType::BF16 => TensorData::F32(ArrayD::ones(IxDyn(&loss_shape))),
            _ => TensorData::F64(ArrayD::ones(IxDyn(&loss_shape))),
        };
        node_grads[loss_node_id] = Some(seed);

        // Walk nodes in reverse order (topological by construction).
        for node in self.nodes.iter().rev() {
            let grad = match node_grads[node.id].take() {
                Some(g) => g,
                None => continue,
            };

            match &node.op {
                TapeOp::Input => {
                    // Leaf: accumulate gradient into the parameter tensor.
                    // 方向 A：此处校验梯度 shape 与参数 shape 一致（消除 silent squeeze）
                    // 护城河 F Phase 1：结构化提取 (v_err=node.id, op="Input", expected=param.shape, actual=grad.shape)
                    let param_shape = node.input_tensors[0].borrow().shape();
                    let grad_shape = grad.shape().to_vec();
                    node.input_tensors[0].borrow_mut().acc_grad(&grad).map_err(|e| {
                        crate::error::TenthError::ShapeMismatch {
                            context: TapeErrorContext {
                                tape_node_id: node.id,
                                op: "Input".to_string(),
                                expected_shape: param_shape,
                                actual_shape: grad_shape,
                            },
                            message: format!("反向传播 shape 错误（节点 #{} Input）：{}", node.id, e),
                        }
                    })?;
                }
                TapeOp::Add | TapeOp::Sub => {
                    let sign_f64: f64 = if node.op == TapeOp::Add { 1.0 } else { -1.0 };
                    let shapes: Vec<Vec<usize>> = (0..node.input_tensors.len().min(2))
                        .map(|i| node.input_tensors[i].borrow().shape())
                        .collect();
                    let op_str = op_name(&node.op);
                    dispatch_float!(node.dtype, E, {
                        let grad_arr = E::from_tensor_data(&grad);
                        let sign = E::from_f64(sign_f64);
                        for (i, input_shape) in shapes.iter().enumerate() {
                            let g_i = if i == 0 {
                                unbroadcast(&grad_arr, input_shape, node.id, op_str)?
                            } else {
                                unbroadcast(&grad_arr, input_shape, node.id, op_str)?.mapv(|v| v * sign)
                            };
                            propagate_grad(node, i, &E::into_tensor_data(g_i), &mut node_grads)?;
                        }
                    });
                }
                TapeOp::Mul => {
                    // Clone input data first to avoid holding RefCell borrows.
                    let (a_data, a_shape, b_data, b_shape) = {
                        let a = node.input_tensors[0].borrow();
                        let b = node.input_tensors[1].borrow();
                        (a.data.clone(), a.shape(), b.data.clone(), b.shape())
                    };
                    let op_str = op_name(&node.op);
                    dispatch_float!(node.dtype, E, {
                        let a_arr = E::from_tensor_data(&a_data);
                        let b_arr = E::from_tensor_data(&b_data);
                        let grad_arr = E::from_tensor_data(&grad);
                        let ga = unbroadcast(&(&grad_arr * &b_arr), &a_shape, node.id, op_str)?;
                        let gb = unbroadcast(&(&grad_arr * &a_arr), &b_shape, node.id, op_str)?;
                        propagate_grad(node, 0, &E::into_tensor_data(ga), &mut node_grads)?;
                        propagate_grad(node, 1, &E::into_tensor_data(gb), &mut node_grads)?;
                    });
                }
                TapeOp::Div => {
                    let (a_data, a_shape, b_data, b_shape) = {
                        let a = node.input_tensors[0].borrow();
                        let b = node.input_tensors[1].borrow();
                        (a.data.clone(), a.shape(), b.data.clone(), b.shape())
                    };
                    let op_str = op_name(&node.op);
                    dispatch_float!(node.dtype, E, {
                        let a_arr = E::from_tensor_data(&a_data);
                        let b_arr = E::from_tensor_data(&b_data);
                        let grad_arr = E::from_tensor_data(&grad);
                        let ga = unbroadcast(&(&grad_arr / &b_arr), &a_shape, node.id, op_str)?;
                        let gb = unbroadcast(&(-&grad_arr * &a_arr / (&b_arr * &b_arr)), &b_shape, node.id, op_str)?;
                        propagate_grad(node, 0, &E::into_tensor_data(ga), &mut node_grads)?;
                        propagate_grad(node, 1, &E::into_tensor_data(gb), &mut node_grads)?;
                    });
                }
                TapeOp::Neg => {
                    dispatch_float!(node.dtype, E, {
                        let grad_arr = E::from_tensor_data(&grad);
                        let g = -&grad_arr;
                        propagate_grad(node, 0, &E::into_tensor_data(g), &mut node_grads)?;
                    });
                }
                TapeOp::ReLU => {
                    let input_data = {
                        let a = node.input_tensors[0].borrow();
                        a.data.clone()
                    };
                    dispatch_float!(node.dtype, E, {
                        let mask = E::from_tensor_data(&input_data).mapv(|x| if x > E::from_f64(0.0) { E::from_f64(1.0) } else { E::from_f64(0.0) });
                        let grad_arr = E::from_tensor_data(&grad);
                        let g_a = &grad_arr * &mask;
                        propagate_grad(node, 0, &E::into_tensor_data(g_a), &mut node_grads)?;
                    });
                }
                TapeOp::MatMul => {
                    // Backward for a @ b:
                    //   d_a = grad @ b^T,  d_b = a^T @ grad
                    // Supports 2D@2D, 1D@2D, 2D@1D by promoting 1D to 2D
                    // and squeezing the gradient back afterwards.
                    if node.input_tensors.len() >= 3 {
                        let (a_data, a_ndim, b_data, b_ndim) = {
                            let a_ref = node.input_tensors[0].borrow();
                            let b_ref = node.input_tensors[1].borrow();
                            (a_ref.data.clone(), a_ref.ndim(), b_ref.data.clone(), b_ref.ndim())
                        };

                        // 方向 A：校验输入维度（仅支持 1D/2D，更高维报错而非静默）
                        // 护城河 F Phase 1：结构化提取 (v_err=node.id, op="MatMul", expected=空, actual=输入 shape)
                        if a_ndim > 2 {
                            return Err(crate::error::TenthError::ShapeMismatch {
                                context: TapeErrorContext {
                                    tape_node_id: node.id,
                                    op: "MatMul".to_string(),
                                    expected_shape: vec![],
                                    actual_shape: node.input_tensors[0].borrow().shape(),
                                },
                                message: format!("MatMul 反向传播：a ndim={} > 2 不支持（方向 A：不再静默处理）", a_ndim),
                            });
                        }
                        if b_ndim > 2 {
                            return Err(crate::error::TenthError::ShapeMismatch {
                                context: TapeErrorContext {
                                    tape_node_id: node.id,
                                    op: "MatMul".to_string(),
                                    expected_shape: vec![],
                                    actual_shape: node.input_tensors[1].borrow().shape(),
                                },
                                message: format!("MatMul 反向传播：b ndim={} > 2 不支持（方向 A：不再静默处理）", b_ndim),
                            });
                        }

                        let (d_a, d_b) = dispatch_float!(node.dtype, E, {
                            let a_arr = E::from_tensor_data(&a_data);
                            let b_arr = E::from_tensor_data(&b_data);
                            let grad_arr = E::from_tensor_data(&grad);

                            // Promote 1D inputs to 2D for uniform handling.
                            let a_2d: ArrayD<E> = if a_ndim == 1 {
                                a_arr.view().insert_axis(ndarray::Axis(0)).into_owned().into_dyn()
                            } else {
                                a_arr
                            };
                            let b_2d: ArrayD<E> = if b_ndim == 1 {
                                b_arr.view().insert_axis(ndarray::Axis(1)).into_owned().into_dyn()
                            } else {
                                b_arr
                            };
                            let grad_2d: ArrayD<E> = if grad_arr.ndim() == 1 {
                                if a_ndim == 1 {
                                    // result of (1,k)@(k,n) squeezed to (n,) → promote to (1,n)
                                    grad_arr.view().insert_axis(ndarray::Axis(0)).into_owned().into_dyn()
                                } else {
                                    // result of (m,k)@(k,1) squeezed to (m,) → promote to (m,1)
                                    grad_arr.view().insert_axis(ndarray::Axis(1)).into_owned().into_dyn()
                                }
                            } else if grad_arr.ndim() == 2 {
                                grad_arr.clone()
                            } else {
                                // 方向 A：grad.ndim() > 2 不再静默 clone
                                return Err(crate::error::TenthError::RuntimeError { line: None, col: None,
                                    message: format!("MatMul 反向传播：grad ndim={} > 2 不支持（方向 A：不再静默兜底）", grad_arr.ndim()),
                                });
                            };

                            // b_2d^T and a_2d^T
                            let b_t = b_2d.view().reversed_axes().to_owned();
                            let a_t = a_2d.view().reversed_axes().to_owned();

                            let d_a_2d = matmul_2d(&grad_2d, &b_t)?;
                            let d_b_2d = matmul_2d(&a_t, &grad_2d)?;

                            // Squeeze gradients back to match original input shapes.
                            // 方向 A：1D squeeze 前校验 shape 符合预期（避免静默 squeeze 错误 shape）
                            // 护城河 F Phase 1：结构化提取 (v_err=node.id, op="MatMul", expected=[1], actual=d_2d.shape)
                            let d_a: ArrayD<E> = if a_ndim == 1 {
                                if d_a_2d.shape().get(0).copied() != Some(1) {
                                    return Err(crate::error::TenthError::ShapeMismatch {
                                        context: TapeErrorContext {
                                            tape_node_id: node.id,
                                            op: "MatMul".to_string(),
                                            expected_shape: vec![1],
                                            actual_shape: d_a_2d.shape().to_vec(),
                                        },
                                        message: format!(
                                            "MatMul 反向 1D squeeze 失败：d_a_2d shape = {:?}，期望第 0 维为 1（方向 A：不再静默 squeeze）",
                                            d_a_2d.shape()
                                        ),
                                    });
                                }
                                d_a_2d.view().index_axis_move(ndarray::Axis(0), 0).into_owned().into_dyn()
                            } else {
                                d_a_2d
                            };
                            let d_b: ArrayD<E> = if b_ndim == 1 {
                                if d_b_2d.shape().get(1).copied() != Some(1) {
                                    return Err(crate::error::TenthError::ShapeMismatch {
                                        context: TapeErrorContext {
                                            tape_node_id: node.id,
                                            op: "MatMul".to_string(),
                                            expected_shape: vec![1],
                                            actual_shape: d_b_2d.shape().to_vec(),
                                        },
                                        message: format!(
                                            "MatMul 反向 1D squeeze 失败：d_b_2d shape = {:?}，期望第 1 维为 1（方向 A：不再静默 squeeze）",
                                            d_b_2d.shape()
                                        ),
                                    });
                                }
                                d_b_2d.view().index_axis_move(ndarray::Axis(1), 0).into_owned().into_dyn()
                            } else {
                                d_b_2d
                            };
                            (E::into_tensor_data(d_a), E::into_tensor_data(d_b))
                        });

                        propagate_grad(node, 0, &d_a, &mut node_grads)?;
                        propagate_grad(node, 1, &d_b, &mut node_grads)?;
                    }
                }
                TapeOp::BatchedMatMul => {
                    // Batched matmul backward:
                    //   forward: (B, M, K) @ (B, K, N) -> (B, M, N)
                    //   d_a = bmm(grad, b^T)  // (B,M,N) @ (B,N,K) -> (B,M,K)
                    //   d_b = bmm(a^T, grad)  // (B,K,M) @ (B,M,N) -> (B,K,N)
                    // 通过 tensor 的 transpose（仅转最后两维）+ bmm 组合实现。
                    // input_tensors = [a, b, result]
                    if node.input_tensors.len() >= 3 {
                        let (d_a, d_b) = {
                            let a_ref = node.input_tensors[0].borrow();
                            let b_ref = node.input_tensors[1].borrow();
                            let a_ndim = a_ref.ndim();
                            let b_ndim = b_ref.ndim();
                            if a_ndim != 3 || b_ndim != 3 {
                                // 护城河 F Phase 1：结构化提取 (v_err=node.id, op="BatchedMatMul", expected=[3,3,3], actual=[a_ndim, b_ndim])
                                return Err(crate::error::TenthError::ShapeMismatch {
                                    context: TapeErrorContext {
                                        tape_node_id: node.id,
                                        op: "BatchedMatMul".to_string(),
                                        expected_shape: vec![3, 3, 3],
                                        actual_shape: vec![a_ndim, b_ndim],
                                    },
                                    message: format!(
                                        "BatchedMatMul 反向传播：a ndim={}, b ndim={}（期望均为 3）",
                                        a_ndim, b_ndim
                                    ),
                                });
                            }
                            if grad.ndim() != 3 {
                                return Err(crate::error::TenthError::RuntimeError { line: None, col: None,
                                    message: format!(
                                        "BatchedMatMul 反向传播：grad ndim={}（期望 3）",
                                        grad.ndim()
                                    ),
                                });
                            }
                            let b_t = b_ref.transpose().ok_or_else(|| crate::error::TenthError::RuntimeError { line: None, col: None,
                                message: "BatchedMatMul 反向：b.transpose() 失败".into(),
                            })?;
                            let a_t = a_ref.transpose().ok_or_else(|| crate::error::TenthError::RuntimeError { line: None, col: None,
                                message: "BatchedMatMul 反向：a.transpose() 失败".into(),
                            })?;
                            // grad 是 TensorData，转为 Tensor 才能调用 bmm
                            let grad_t = Tensor::from_tensor_data(grad.clone());
                            let d_a_t = grad_t.bmm(&b_t).map_err(|e| crate::error::TenthError::RuntimeError { line: None, col: None,
                                message: format!("BatchedMatMul 反向 d_a：{}", e),
                            })?;
                            let d_b_t = a_t.bmm(&grad_t).map_err(|e| crate::error::TenthError::RuntimeError { line: None, col: None,
                                message: format!("BatchedMatMul 反向 d_b：{}", e),
                            })?;
                            // d_a_t/d_b_t 的 data 是 TensorData，直接保留 dtype
                            (d_a_t.data.clone(), d_b_t.data.clone())
                        };
                        propagate_grad(node, 0, &d_a, &mut node_grads)?;
                        propagate_grad(node, 1, &d_b, &mut node_grads)?;
                    }
                }
                TapeOp::Transpose => {
                    dispatch_float!(node.dtype, E, {
                        let grad_arr = E::from_tensor_data(&grad);
                        let g_a = {
                            let mut perm: Vec<usize> = (0..grad_arr.ndim()).collect();
                            if perm.len() >= 2 {
                                let last = perm.len() - 1;
                                perm.swap(last - 1, last);
                            }
                            grad_arr.view().permuted_axes(perm).to_owned()
                        };
                        propagate_grad(node, 0, &E::into_tensor_data(g_a), &mut node_grads)?;
                    });
                }
                TapeOp::Sum => {
                    let a_shape = node.input_tensors[0].borrow().shape();
                    dispatch_float!(node.dtype, E, {
                        let grad_arr = E::from_tensor_data(&grad);
                        let s: E = grad_arr.iter().copied().sum();
                        let g_a = ArrayD::from_elem(IxDyn(&a_shape), s);
                        propagate_grad(node, 0, &E::into_tensor_data(g_a), &mut node_grads)?;
                    });
                }
                TapeOp::Mean => {
                    let (a_shape, a_size) = {
                        let a = node.input_tensors[0].borrow();
                        (a.shape(), a.size())
                    };
                    dispatch_float!(node.dtype, E, {
                        let grad_arr = E::from_tensor_data(&grad);
                        let n = E::from_f64(a_size as f64);
                        let s: E = grad_arr.iter().copied().sum::<E>() / n;
                        let g_a = ArrayD::from_elem(IxDyn(&a_shape), s);
                        propagate_grad(node, 0, &E::into_tensor_data(g_a), &mut node_grads)?;
                    });
                }
                TapeOp::Exp => {
                    let result_data = node.input_tensors[1].borrow().data.clone();
                    dispatch_float!(node.dtype, E, {
                        let grad_arr = E::from_tensor_data(&grad);
                        let y = E::from_tensor_data(&result_data);
                        let g_a = &grad_arr * &y;
                        propagate_grad(node, 0, &E::into_tensor_data(g_a), &mut node_grads)?;
                    });
                }
                TapeOp::Log => {
                    let a_data = node.input_tensors[0].borrow().data.clone();
                    dispatch_float!(node.dtype, E, {
                        let grad_arr = E::from_tensor_data(&grad);
                        let a = E::from_tensor_data(&a_data);
                        let g_a = &grad_arr / &a;
                        propagate_grad(node, 0, &E::into_tensor_data(g_a), &mut node_grads)?;
                    });
                }
                TapeOp::Sigmoid => {
                    let result_data = node.input_tensors[1].borrow().data.clone();
                    dispatch_float!(node.dtype, E, {
                        let grad_arr = E::from_tensor_data(&grad);
                        let y = E::from_tensor_data(&result_data);
                        let one = E::from_f64(1.0);
                        let g_a = &grad_arr * &y * &y.mapv(|v| one - v);
                        propagate_grad(node, 0, &E::into_tensor_data(g_a), &mut node_grads)?;
                    });
                }
                TapeOp::BatchNorm => {
                    // BN backward (T42 修正公式 — 多 channel dX 场景):
                    // x: (N, C, H, W, ...)，gamma/beta/std_inv: (C,)
                    // 每 channel c 独立计算（M = N * H * W * ... 为每 channel 元素数）：
                    //   d_gamma[c] = sum_{n,h,w}(dY * x_hat)   （仅 channel c 的元素）
                    //   d_beta[c]  = sum_{n,h,w}(dY)           （仅 channel c 的元素）
                    //   mean_dY[c]      = d_beta[c]  / M
                    //   mean_dY_xhat[c] = d_gamma[c] / M
                    //   dX[..., c, ...] = gamma[c] * std_inv[c] * (dY - mean_dY[c] - x_hat * mean_dY_xhat[c])
                    // 旧实现 3 处错误：
                    //   1) d_gamma/d_beta 形状为 (N,C,H,W) 而 gamma/beta 形状为 (C,)，acc_grad 拒绝；
                    //   2) mean_dY/mean_dY_xhat 算成全局标量，而非 per-channel；
                    //   3) std_inv/gamma 形状 (C,) 与 (N,C,H,W) 相乘时 ndarray 从右对齐广播，
                    //      仅当 W==C 时巧合正确，多 channel 普遍错误。
                    // input_tensors = [input, gamma, beta, x_hat, std_inv, result]
                    if node.input_tensors.len() >= 5 {
                        let (x_shape, gamma_data, x_hat_data, std_inv_data) = {
                            let x_ref = node.input_tensors[0].borrow();
                            let gamma_ref = node.input_tensors[1].borrow();
                            let x_hat_ref = node.input_tensors[3].borrow();
                            let std_inv_ref = node.input_tensors[4].borrow();
                            (x_ref.shape(), gamma_ref.data.clone(), x_hat_ref.data.clone(), std_inv_ref.data.clone())
                        };
                        // x 至少 2D（forward 已校验）：(N, C, H, W, ...)
                        let c = x_shape[1];
                        let n = x_shape[0];
                        let spatial: usize = x_shape[2..].iter().product();
                        // 2D 输入时 spatial 为 empty product = 1，与 forward 一致
                        let m_per_channel: usize = n * spatial;

                        dispatch_float!(node.dtype, E, {
                            let grad_arr = E::from_tensor_data(&grad);
                            let gamma = E::from_tensor_data(&gamma_data);
                            let x_hat = E::from_tensor_data(&x_hat_data);
                            let std_inv = E::from_tensor_data(&std_inv_data);

                            let grad_flat = grad_arr.as_standard_layout().to_owned();
                            let grad_slice = grad_flat.as_slice().unwrap_or(&[]);
                            let xhat_flat = x_hat.as_standard_layout().to_owned();
                            let xhat_slice = xhat_flat.as_slice().unwrap_or(&[]);
                            let g_flat = gamma.as_standard_layout().to_owned();
                            let g_slice = g_flat.as_slice().unwrap_or(&[]);
                            let s_flat = std_inv.as_standard_layout().to_owned();
                            let s_slice = s_flat.as_slice().unwrap_or(&[]);

                            // 第 1 遍：每 channel 累加
                            //   d_beta[c]  = sum_{n,h,w}(dY)
                            //   d_gamma[c] = sum_{n,h,w}(dY * x_hat)
                            let mut d_beta_data = vec![E::from_f64(0.0); c];
                            let mut d_gamma_data = vec![E::from_f64(0.0); c];
                            for ci in 0..c {
                                let mut sum_dy = E::from_f64(0.0);
                                let mut sum_dy_xhat = E::from_f64(0.0);
                                for ni in 0..n {
                                    for si in 0..spatial {
                                        let idx = (ni * c + ci) * spatial + si;
                                        let dy = grad_slice.get(idx).copied().unwrap_or(E::from_f64(0.0));
                                        let xh = xhat_slice.get(idx).copied().unwrap_or(E::from_f64(0.0));
                                        sum_dy = sum_dy + dy;
                                        sum_dy_xhat = sum_dy_xhat + dy * xh;
                                    }
                                }
                                d_beta_data[ci] = sum_dy;
                                d_gamma_data[ci] = sum_dy_xhat;
                            }
                            let d_gamma = ArrayD::from_shape_vec(IxDyn(&[c]), d_gamma_data.clone()).unwrap();
                            let d_beta = ArrayD::from_shape_vec(IxDyn(&[c]), d_beta_data.clone()).unwrap();

                            // 第 2 遍：按 row-major (n, c, spatial) 顺序填 d_x_data
                            // 每 channel 用自己的 mean_dY[c]、mean_dY_xhat[c] 与 gamma[c] * std_inv[c]
                            let m_inv = E::from_f64(m_per_channel as f64);
                            let mut d_x_data = Vec::with_capacity(grad_slice.len());
                            for ni in 0..n {
                                for ci in 0..c {
                                    let mean_dy = d_beta_data[ci] / m_inv;
                                    let mean_dy_xhat = d_gamma_data[ci] / m_inv;
                                    let g = g_slice.get(ci).copied().unwrap_or_else(|| E::from_f64(1.0));
                                    let inv = s_slice.get(ci).copied().unwrap_or_else(|| E::from_f64(1.0));
                                    let scale = g * inv;
                                    for si in 0..spatial {
                                        let idx = (ni * c + ci) * spatial + si;
                                        let dy = grad_slice.get(idx).copied().unwrap_or(E::from_f64(0.0));
                                        let xh = xhat_slice.get(idx).copied().unwrap_or(E::from_f64(0.0));
                                        d_x_data.push(scale * (dy - mean_dy - xh * mean_dy_xhat));
                                    }
                                }
                            }
                            let d_x = ArrayD::from_shape_vec(IxDyn(&x_shape), d_x_data).unwrap();

                            propagate_grad(node, 0, &E::into_tensor_data(d_x), &mut node_grads)?;
                            propagate_grad(node, 1, &E::into_tensor_data(d_gamma), &mut node_grads)?;
                            propagate_grad(node, 2, &E::into_tensor_data(d_beta), &mut node_grads)?;
                        });
                    }
                }
                TapeOp::LayerNorm => {
                    // LayerNorm backward (over last dim, T42 修正 — per-feature γ 场景):
                    // d(gamma) = sum_over_outer(dY * x_hat), d(beta) = sum_over_outer(dY)
                    // 令 gy_j = dY_j * gamma_j（per-feature γ 乘入上游梯度），则 per-row：
                    //   dX_j = std_inv * (gy_j - mean(gy) - x_hat_j * mean(gy * x_hat))
                    // 注意：gamma 不能提到括号外，因为 mean(gy) 与 mean(gy * x_hat) 依赖 per-feature gamma_j 求和。
                    // input_tensors = [input, gamma, beta, x_hat, std_inv, result]
                    if node.input_tensors.len() >= 5 {
                        let (x_hat_shape, x_hat_data, std_inv_data, gamma_data) = {
                            let x_hat_ref = node.input_tensors[3].borrow();
                            let std_inv_ref = node.input_tensors[4].borrow();
                            let gamma_ref = node.input_tensors[1].borrow();
                            (x_hat_ref.shape(), x_hat_ref.data.clone(), std_inv_ref.data.clone(), gamma_ref.data.clone())
                        };
                        let ndim = x_hat_shape.len();
                        let axis_len = x_hat_shape[ndim - 1];
                        let outer_len: usize = x_hat_shape[..ndim - 1].iter().product();

                        dispatch_float!(node.dtype, E, {
                            let x_hat_arr = E::from_tensor_data(&x_hat_data);
                            let std_inv_arr = E::from_tensor_data(&std_inv_data);
                            let gamma_arr = E::from_tensor_data(&gamma_data);
                            let grad_arr = E::from_tensor_data(&grad);

                            let x_hat_flat = x_hat_arr.as_standard_layout().to_owned();
                            let x_hat_slice = x_hat_flat.as_slice().unwrap_or(&[]);
                            let std_inv_flat = std_inv_arr.as_standard_layout().to_owned();
                            let std_inv_slice = std_inv_flat.as_slice().unwrap_or(&[]);
                            let grad_flat = grad_arr.as_standard_layout().to_owned();
                            let grad_slice = grad_flat.as_slice().unwrap_or(&[]);
                            let g_flat = gamma_arr.as_standard_layout().to_owned();
                            let g_slice = g_flat.as_slice().unwrap_or(&[]);

                            // d(gamma): sum over outer dims of dY * x_hat
                            let mut d_gamma_data = vec![E::from_f64(0.0); axis_len];
                            for i in 0..outer_len {
                                let start = i * axis_len;
                                for j in 0..axis_len {
                                    d_gamma_data[j] = d_gamma_data[j]
                                        + grad_slice.get(start + j).copied().unwrap_or(E::from_f64(0.0))
                                        * x_hat_slice.get(start + j).copied().unwrap_or(E::from_f64(0.0));
                                }
                            }
                            let d_gamma = ArrayD::from_shape_vec(IxDyn(&[axis_len]), d_gamma_data).unwrap();

                            // d(beta): sum over outer dims of dY
                            let mut d_beta_data = vec![E::from_f64(0.0); axis_len];
                            for i in 0..outer_len {
                                let start = i * axis_len;
                                for j in 0..axis_len {
                                    d_beta_data[j] = d_beta_data[j]
                                        + grad_slice.get(start + j).copied().unwrap_or(E::from_f64(0.0));
                                }
                            }
                            let d_beta = ArrayD::from_shape_vec(IxDyn(&[axis_len]), d_beta_data).unwrap();

                            // dX per row (T42 修正公式 — per-feature γ 场景):
                            // 令 gy_j = dY_j * gamma_j（per-feature γ 乘入上游梯度），
                            // dX_j = std_inv * (gy_j - mean(gy) - x_hat_j * mean(gy * x_hat))
                            // 注意：gamma 不能提到括号外——mean(gy) 与 mean(gy * x_hat) 都依赖 per-feature gamma_j 求和。
                            // 旧实现写成 `gamma_j * std_inv * (dY_j - mean(dY) - x_hat_j * mean(dY * x_hat))`，
                            // 仅在 gamma 为标量/全 1 时与正确公式等价；per-feature γ 下产生错误梯度。
                            let mut d_x_data = Vec::with_capacity(grad_slice.len());
                            for i in 0..outer_len {
                                let start = i * axis_len;
                                let inv = std_inv_slice.get(i).copied().unwrap_or_else(|| E::from_f64(1.0));
                                let mut sum_gy = E::from_f64(0.0);
                                let mut sum_gy_xhat = E::from_f64(0.0);
                                for j in 0..axis_len {
                                    let dy = grad_slice.get(start + j).copied().unwrap_or(E::from_f64(0.0));
                                    let xh = x_hat_slice.get(start + j).copied().unwrap_or(E::from_f64(0.0));
                                    let g = g_slice.get(j).copied().unwrap_or_else(|| E::from_f64(1.0));
                                    let gy = dy * g;
                                    sum_gy = sum_gy + gy;
                                    sum_gy_xhat = sum_gy_xhat + gy * xh;
                                }
                                let n_inv = E::from_f64(axis_len as f64);
                                let mean_gy = sum_gy / n_inv;
                                let mean_gy_xhat = sum_gy_xhat / n_inv;
                                for j in 0..axis_len {
                                    let dy = grad_slice.get(start + j).copied().unwrap_or(E::from_f64(0.0));
                                    let xh = x_hat_slice.get(start + j).copied().unwrap_or(E::from_f64(0.0));
                                    let g = g_slice.get(j).copied().unwrap_or_else(|| E::from_f64(1.0));
                                    let gy = dy * g;
                                    d_x_data.push(inv * (gy - mean_gy - xh * mean_gy_xhat));
                                }
                            }
                            let d_x = ArrayD::from_shape_vec(IxDyn(&x_hat_shape), d_x_data).unwrap();

                            propagate_grad(node, 0, &E::into_tensor_data(d_x), &mut node_grads)?;
                            propagate_grad(node, 1, &E::into_tensor_data(d_gamma), &mut node_grads)?;
                            propagate_grad(node, 2, &E::into_tensor_data(d_beta), &mut node_grads)?;
                        });
                    }
                }
                TapeOp::Gelu => {
                    // GELU backward: d(gelu(x))/dx = 0.5 * (1 + tanh(inner)) + 0.5 * x * sech^2(inner) * sqrt(2/pi) * (1 + 3*0.044715*x^2)
                    // inner = sqrt(2/pi) * (x + 0.044715 * x^3)
                    // input_tensors = [input, result]
                    let x_data = node.input_tensors[0].borrow().data.clone();
                    dispatch_float!(node.dtype, E, {
                        let grad_arr = E::from_tensor_data(&grad);
                        let x = E::from_tensor_data(&x_data);
                        let sqrt_2_over_pi = E::from_f64((2.0 / std::f64::consts::PI).sqrt());
                        let c1 = E::from_f64(0.044715);
                        let c3 = E::from_f64(3.0 * 0.044715);
                        let half = E::from_f64(0.5);
                        let one = E::from_f64(1.0);
                        let deriv = x.mapv(|xv| {
                            let inner = sqrt_2_over_pi * (xv + c1 * xv * xv * xv);
                            let tanh_inner = inner.tanh_();
                            let sech2 = one - tanh_inner * tanh_inner;
                            half * (one + tanh_inner) + half * xv * sech2 * sqrt_2_over_pi * (one + c3 * xv * xv)
                        });
                        let g_a = &grad_arr * &deriv;
                        propagate_grad(node, 0, &E::into_tensor_data(g_a), &mut node_grads)?;
                    });
                }
                TapeOp::Select => {
                    // Select backward: result = cond ? then : else
                    // d_then = unbroadcast(grad * cond_mask, then.shape)
                    // d_else = unbroadcast(grad * (1 - cond_mask), else.shape)
                    // cond 不可微（bool 语义），不传播梯度
                    // input_tensors = [cond, then, else_, result]
                    // inputs = [then_id, else_id]（dummy 占位保证对齐）
                    if node.input_tensors.len() >= 4 {
                        let (cond_data, then_shape, else_shape) = {
                            let cond_ref = node.input_tensors[0].borrow();
                            let then_ref = node.input_tensors[1].borrow();
                            let else_ref = node.input_tensors[2].borrow();
                            (cond_ref.data.clone(), then_ref.shape(), else_ref.shape())
                        };
                        let _grad_shape = grad.shape().to_vec();
                        let op_str = op_name(&node.op);
                        dispatch_float!(node.dtype, E, {
                            let grad_arr = E::from_tensor_data(&grad);
                            let cond_view = E::from_tensor_data(&cond_data);
                            // cond 广播到 result（grad）shape，再转为 0/1 mask
                            let cond_mask: ArrayD<E> = if cond_view.shape() == grad_arr.shape() {
                                cond_view.mapv(|v| if v > E::from_f64(0.5) { E::from_f64(1.0) } else { E::from_f64(0.0) })
                            } else {
                                let bcast_view = cond_view.broadcast(IxDyn(grad_arr.shape()))
                                    .unwrap_or_else(|| cond_view.view());
                                bcast_view.mapv(|v| if v > E::from_f64(0.5) { E::from_f64(1.0) } else { E::from_f64(0.0) }).into_owned()
                            };
                            let one = E::from_f64(1.0);
                            // d_then = unbroadcast(grad * cond_mask, then.shape)
                            let d_then = unbroadcast(&(&grad_arr * &cond_mask), &then_shape, node.id, op_str)?;
                            // d_else = unbroadcast(grad * (1 - cond_mask), else.shape)
                            let inv_mask = cond_mask.mapv(|v| one - v);
                            let d_else = unbroadcast(&(&grad_arr * &inv_mask), &else_shape, node.id, op_str)?;

                            propagate_grad(node, 0, &E::into_tensor_data(d_then), &mut node_grads)?;
                            propagate_grad(node, 1, &E::into_tensor_data(d_else), &mut node_grads)?;
                        });
                    }
                }
                TapeOp::Abs => {
                    // |x| backward: d|x|/dx = sign(x)，x=0 处取 0（次梯度中点，工程惯例）
                    // input_tensors = [input, result]
                    let a_data = node.input_tensors[0].borrow().data.clone();
                    dispatch_float!(node.dtype, E, {
                        let grad_arr = E::from_tensor_data(&grad);
                        let a = E::from_tensor_data(&a_data);
                        let zero = E::from_f64(0.0);
                        let one = E::from_f64(1.0);
                        let neg_one = E::from_f64(-1.0);
                        let sign = a.mapv(|x| if x > zero { one } else if x < zero { neg_one } else { zero });
                        let g_a = &grad_arr * &sign;
                        propagate_grad(node, 0, &E::into_tensor_data(g_a), &mut node_grads)?;
                    });
                }
                TapeOp::Scatter => {
                    // Scatter backward（支持任意 dim + 多维 index/src，PyTorch 对齐）:
                    //   forward: out = base.clone();
                    //            对每个 multi-index idx（遍历 index）:
                    //              actual = idx; actual[dim] = index[idx] as usize
                    //              out[actual] = src[idx]
                    //   d_src[idx] = grad[actual]              (gather 语义)
                    //   d_base = grad.clone()，但所有 actual 位置置 0（被 src 覆盖，梯度不传回 base）
                    //   index 不可微（无梯度）
                    // input_tensors = [base, src, index, result]
                    // inputs = [base_id, src_id]
                    // dim 存于 node.aux
                    if node.input_tensors.len() >= 4 {
                        let dim = node.aux;
                        let (base_shape, index_data, index_shape) = {
                            let base_ref = node.input_tensors[0].borrow();
                            let index_ref = node.input_tensors[2].borrow();
                            (base_ref.shape(), index_ref.data.clone(), index_ref.shape().to_vec())
                        };
                        // Scatter 是 index-based 操作，用 f64 视图计算，最后按 node.dtype 转换存储
                        let grad_view = grad.as_f64_view();
                        let index_view = index_data.as_f64_view();
                        let total: usize = index_shape.iter().product();
                        let unflatten = |flat: usize| -> Vec<usize> {
                            let mut multi = vec![0usize; index_shape.len()];
                            let mut rem = flat;
                            for i in (0..index_shape.len()).rev() {
                                multi[i] = rem % index_shape[i];
                                rem /= index_shape[i];
                            }
                            multi
                        };
                        // d_src[idx] = grad[actual]，actual[dim]=index[idx]
                        let mut d_src_data = Vec::with_capacity(total);
                        for flat in 0..total {
                            let multi = unflatten(flat);
                            let mut actual = multi.clone();
                            let v = index_view[IxDyn(&multi)];
                            actual[dim] = v as usize;
                            let g = grad_view.get(IxDyn(&actual)).copied().unwrap_or(0.0);
                            d_src_data.push(g);
                        }
                        // d_base = grad.clone()，但所有 actual 位置置 0
                        let mut d_base_data: Vec<f64> = grad_view.iter().copied().collect();
                        for flat in 0..total {
                            let multi = unflatten(flat);
                            let mut actual = multi.clone();
                            let v = index_view[IxDyn(&multi)];
                            actual[dim] = v as usize;
                            let actual_flat = flatten_index(&actual, &base_shape);
                            if let Some(slot) = d_base_data.get_mut(actual_flat) {
                                *slot = 0.0;
                            }
                        }
                        // 按 node.dtype 构造 TensorData
                        let (d_base, d_src) = match node.dtype {
                            BaseType::F32 => {
                                let d_base = TensorData::F32(
                                    ArrayD::from_shape_vec(IxDyn(&base_shape), d_base_data.iter().map(|v| *v as f32).collect())
                                        .map_err(|_| crate::error::TenthError::RuntimeError { line: None, col: None,
                                            message: "Scatter 反向 d_base reshape 失败".into(),
                                        })?
                                );
                                let d_src = TensorData::F32(
                                    ArrayD::from_shape_vec(IxDyn(&index_shape), d_src_data.iter().map(|v| *v as f32).collect())
                                        .map_err(|_| crate::error::TenthError::RuntimeError { line: None, col: None,
                                            message: "Scatter 反向 d_src reshape 失败".into(),
                                        })?
                                );
                                (d_base, d_src)
                            }
                            _ => {
                                let d_base = TensorData::F64(
                                    ArrayD::from_shape_vec(IxDyn(&base_shape), d_base_data)
                                        .map_err(|_| crate::error::TenthError::RuntimeError { line: None, col: None,
                                            message: "Scatter 反向 d_base reshape 失败".into(),
                                        })?
                                );
                                let d_src = TensorData::F64(
                                    ArrayD::from_shape_vec(IxDyn(&index_shape), d_src_data)
                                        .map_err(|_| crate::error::TenthError::RuntimeError { line: None, col: None,
                                            message: "Scatter 反向 d_src reshape 失败".into(),
                                        })?
                                );
                                (d_base, d_src)
                            }
                        };
                        propagate_grad(node, 0, &d_base, &mut node_grads)?;
                        propagate_grad(node, 1, &d_src, &mut node_grads)?;
                    }
                }
                TapeOp::Gather => {
                    // Gather backward（支持任意 dim + 多维 index，PyTorch 对齐）:
                    //   forward: out[idx] = base[actual]，actual[dim]=index[idx]，其他维同 idx
                    //   d_base = zeros_like(base)
                    //   对每个 idx: d_base[actual] += grad[idx]   (scatter-add 语义，重复 index 累加)
                    //   index 不可微（无梯度）
                    // input_tensors = [base, index, result]
                    // inputs = [base_id]
                    // dim 存于 node.aux
                    if node.input_tensors.len() >= 3 {
                        let dim = node.aux;
                        let (base_shape, index_data, index_shape) = {
                            let base_ref = node.input_tensors[0].borrow();
                            let index_ref = node.input_tensors[1].borrow();
                            (base_ref.shape(), index_ref.data.clone(), index_ref.shape().to_vec())
                        };
                        // Gather 是 index-based 操作，用 f64 视图计算，最后按 node.dtype 转换存储
                        let grad_view = grad.as_f64_view();
                        let index_view = index_data.as_f64_view();
                        let total: usize = index_shape.iter().product();
                        let unflatten = |flat: usize| -> Vec<usize> {
                            let mut multi = vec![0usize; index_shape.len()];
                            let mut rem = flat;
                            for i in (0..index_shape.len()).rev() {
                                multi[i] = rem % index_shape[i];
                                rem /= index_shape[i];
                            }
                            multi
                        };
                        let base_total: usize = base_shape.iter().product();
                        let mut d_base_data: Vec<f64> = vec![0.0; base_total];
                        for flat in 0..total {
                            let multi = unflatten(flat);
                            let mut actual = multi.clone();
                            let v = index_view[IxDyn(&multi)];
                            actual[dim] = v as usize;
                            let actual_flat = flatten_index(&actual, &base_shape);
                            let g = grad_view.get(IxDyn(&multi)).copied().unwrap_or(0.0);
                            if let Some(slot) = d_base_data.get_mut(actual_flat) {
                                *slot += g;
                            }
                        }
                        let d_base = match node.dtype {
                            BaseType::F32 => TensorData::F32(
                                ArrayD::from_shape_vec(IxDyn(&base_shape), d_base_data.iter().map(|v| *v as f32).collect())
                                    .map_err(|_| crate::error::TenthError::RuntimeError { line: None, col: None,
                                        message: "Gather 反向 d_base reshape 失败".into(),
                                    })?
                            ),
                            _ => TensorData::F64(
                                ArrayD::from_shape_vec(IxDyn(&base_shape), d_base_data)
                                    .map_err(|_| crate::error::TenthError::RuntimeError { line: None, col: None,
                                        message: "Gather 反向 d_base reshape 失败".into(),
                                    })?
                            ),
                        };
                        propagate_grad(node, 0, &d_base, &mut node_grads)?;
                    }
                }
                TapeOp::Reshape => {
                    // Reshape backward: d_input = grad.reshape(input.shape())
                    // input_tensors = [input, result]（原始 shape 从 input.shape() 读取）
                    // 元素数必须一致（reshape 不改变元素数）
                    // Reshape 是 dtype 无关操作，直接在 TensorData 上 reshape
                    // 护城河 F Phase 1：结构化提取 (v_err=node.id, op="Reshape", expected=input.shape, actual=grad.shape)
                    let orig_shape = node.input_tensors[0].borrow().shape();
                    let total: usize = orig_shape.iter().product();
                    if grad.len() != total {
                        return Err(crate::error::TenthError::ShapeMismatch {
                            context: TapeErrorContext {
                                tape_node_id: node.id,
                                op: "Reshape".to_string(),
                                expected_shape: orig_shape.clone(),
                                actual_shape: grad.shape().to_vec(),
                            },
                            message: format!(
                                "Reshape 反向元素数不匹配：grad {} 元素，原始 shape {:?} 期望 {} 元素",
                                grad.len(), orig_shape, total
                            ),
                        });
                    }
                    // 注意：grad 可能不是连续内存（如经过 MatMul/Transpose 后的视图），
                    // 用 from_shape_vec 重新构造保证连续，避免 into_shape_with_order 失败。
                    let g_a = match grad {
                        TensorData::F32(a) => {
                            let data: Vec<f32> = a.iter().cloned().collect();
                            TensorData::F32(
                                ArrayD::from_shape_vec(IxDyn(&orig_shape), data)
                                    .map_err(|_| crate::error::TenthError::RuntimeError { line: None, col: None,
                                        message: format!("Reshape 反向 reshape grad 到 {:?} 失败", orig_shape),
                                    })?
                            )
                        },
                        TensorData::F64(a) => {
                            let data: Vec<f64> = a.iter().cloned().collect();
                            TensorData::F64(
                                ArrayD::from_shape_vec(IxDyn(&orig_shape), data)
                                    .map_err(|_| crate::error::TenthError::RuntimeError { line: None, col: None,
                                        message: format!("Reshape 反向 reshape grad 到 {:?} 失败", orig_shape),
                                    })?
                            )
                        },
                        // Phase 2: F16/BF16 grad 转 f32（dispatch_float! 走 f32 路径）
                        TensorData::F16(a) => {
                            let data: Vec<f32> = a.iter().map(|v| v.to_f32()).collect();
                            TensorData::F32(
                                ArrayD::from_shape_vec(IxDyn(&orig_shape), data)
                                    .map_err(|_| crate::error::TenthError::RuntimeError { line: None, col: None,
                                        message: format!("Reshape 反向 reshape grad 到 {:?} 失败", orig_shape),
                                    })?
                            )
                        },
                        TensorData::BF16(a) => {
                            let data: Vec<f32> = a.iter().map(|v| v.to_f32()).collect();
                            TensorData::F32(
                                ArrayD::from_shape_vec(IxDyn(&orig_shape), data)
                                    .map_err(|_| crate::error::TenthError::RuntimeError { line: None, col: None,
                                        message: format!("Reshape 反向 reshape grad 到 {:?} 失败", orig_shape),
                                    })?
                            )
                        },
                    };
                    propagate_grad(node, 0, &g_a, &mut node_grads)?;
                }
                TapeOp::MaskedFill => {
                    // MaskedFill backward: d_input = grad * (1 - mask)
                    // mask=true 位置被 value 覆盖，不传梯度回输入
                    // input_tensors = [input, mask, result]
                    let mask_data = node.input_tensors[1].borrow().data.clone();
                    dispatch_float!(node.dtype, E, {
                        let grad_arr = E::from_tensor_data(&grad);
                        let mask_view = E::from_tensor_data(&mask_data);
                        let one = E::from_f64(1.0);
                        let zero = E::from_f64(0.0);
                        let inv_mask = mask_view.mapv(|v| if v > E::from_f64(0.5) { zero } else { one });
                        let g_a = &grad_arr * &inv_mask;
                        propagate_grad(node, 0, &E::into_tensor_data(g_a), &mut node_grads)?;
                    });
                }
                TapeOp::Conv2D => {
                    // input_tensors = [input(4D), weight(4D), im2col(2D), output(4D)]
                    // Forward: output = im2col @ w_flat^T  where w_flat = weight.reshape(C_out, C_in*kH*kW)
                    // dW_flat = im2col^T @ dY        → reshape back to (C_out, C_in, kH, kW)
                    // d(im2col) = dY @ w_flat        → col2im back to input shape
                    // dY (upstream grad) has output shape (N, C_out, H_out, W_out);
                    // we reshape it to 2D (N*H_out*W_out, C_out) for matmul.
                    if node.input_tensors.len() >= 4 {
                        let (x_shape, w_shape, out_shape) = {
                            let x_ref = node.input_tensors[0].borrow();
                            let w_ref = node.input_tensors[1].borrow();
                            let out_ref = node.input_tensors[3].borrow();
                            (x_ref.shape(), w_ref.shape(), out_ref.shape())
                        };
                        // output is (N, C_out, H_out, W_out)
                        let n = out_shape[0];
                        let c_out = out_shape[1];
                        let hw_out = out_shape[2] * out_shape[3];

                        let (d_x, d_w) = {
                            let w_ref = node.input_tensors[1].borrow();
                            let col_ref = node.input_tensors[2].borrow();
                            dispatch_float!(node.dtype, E, {
                                let w_arr = E::from_tensor_data(&w_ref.data);
                                let col_arr = E::from_tensor_data(&col_ref.data);
                                let grad_arr = E::from_tensor_data(&grad);

                                let grad_2d: ArrayD<E> = {
                                    let v: Vec<E> = grad_arr.iter().cloned().collect();
                                    ArrayD::from_shape_vec(IxDyn(&[hw_out * n, c_out]), v).map_err(|_| {
                                        crate::error::TenthError::RuntimeError { line: None, col: None,
                                            message: "Conv2D 反向 reshape grad 失败".into(),
                                        }
                                    })?
                                };

                                // dW_flat = im2col^T @ dY
                                let col_t = col_arr.view().reversed_axes().to_owned();
                                let d_w_flat = matmul_2d(&col_t, &grad_2d)?;
                                let d_w_flat_t = d_w_flat.view().reversed_axes().to_owned();
                                let total_w: usize = w_shape.iter().product();
                                if d_w_flat_t.len() != total_w {
                                    return Err(crate::error::TenthError::RuntimeError { line: None, col: None,
                                        message: format!("Conv2D 反向 dW 元素数不匹配：{} != {}", d_w_flat_t.len(), total_w),
                                    });
                                }
                                let d_w = ArrayD::from_shape_vec(IxDyn(&w_shape), d_w_flat_t.iter().cloned().collect()).map_err(|_| {
                                    crate::error::TenthError::RuntimeError { line: None, col: None,
                                        message: "Conv2D 反向 dW reshape 失败".into(),
                                    }
                                })?;

                                // d(im2col) = dY @ w_flat
                                let w_flat: ArrayD<E> = {
                                    let v: Vec<E> = w_arr.iter().cloned().collect();
                                    ArrayD::from_shape_vec(IxDyn(&[c_out, w_shape[1] * w_shape[2] * w_shape[3]]), v).map_err(|_| {
                                        crate::error::TenthError::RuntimeError { line: None, col: None,
                                            message: "Conv2D 反向 w_flat reshape 失败".into(),
                                        }
                                    })?
                                };
                                let d_col = matmul_2d(&grad_2d, &w_flat)?;

                                // col2im: accumulate d_col back into input shape
                                let x_total: usize = x_shape.iter().product();
                                if d_col.len() != x_total {
                                    return Err(crate::error::TenthError::RuntimeError { line: None, col: None,
                                        message: format!("Conv2D 反向 dX 元素数不匹配：d_col {} != x_total {}", d_col.len(), x_total),
                                    });
                                }
                                let d_x = ArrayD::from_shape_vec(IxDyn(&x_shape), d_col.iter().cloned().collect()).map_err(|_| {
                                    crate::error::TenthError::RuntimeError { line: None, col: None,
                                        message: "Conv2D 反向 dX reshape 失败".into(),
                                    }
                                })?;

                                (E::into_tensor_data(d_x), E::into_tensor_data(d_w))
                            })
                        };

                        propagate_grad(node, 0, &d_x, &mut node_grads)?;
                        propagate_grad(node, 1, &d_w, &mut node_grads)?;
                    }
                }
                TapeOp::Dropout => {
                    // d(dropout(x))/dx = mask * dY
                    // input_tensors = [input, mask, result]
                    if node.input_tensors.len() >= 2 {
                        let mask_data = node.input_tensors[1].borrow().data.clone();
                        dispatch_float!(node.dtype, E, {
                            let grad_arr = E::from_tensor_data(&grad);
                            let mask = E::from_tensor_data(&mask_data);
                            let g_a = &grad_arr * &mask;
                            propagate_grad(node, 0, &E::into_tensor_data(g_a), &mut node_grads)?;
                        });
                    }
                }
                TapeOp::CrossEntropy => {
                    // d(CE)/d(logits) = softmax - target
                    // input_tensors = [logits, softmax_output, target]
                    if node.input_tensors.len() >= 3 {
                        let (sm_data, tgt_data) = {
                            let sm_ref = node.input_tensors[1].borrow();
                            let tgt_ref = node.input_tensors[2].borrow();
                            (sm_ref.data.clone(), tgt_ref.data.clone())
                        };
                        dispatch_float!(node.dtype, E, {
                            let sm = E::from_tensor_data(&sm_data);
                            let tgt = E::from_tensor_data(&tgt_data);
                            let g_a = &sm - &tgt;
                            propagate_grad(node, 0, &E::into_tensor_data(g_a), &mut node_grads)?;
                        });
                    }
                }
                TapeOp::Softmax => {
                    // d(softmax(x)_i)/dx_j = y_i * (δ_ij - y_j)
                    // Chain rule: g_i = y_i * (grad_i - sum_j(grad_j * y_j))
                    let result_data = node.input_tensors[1].borrow().data.clone();
                    dispatch_float!(node.dtype, E, {
                        let grad_arr = E::from_tensor_data(&grad);
                        let y = E::from_tensor_data(&result_data);
                        let sum_term: E = (&grad_arr * &y).iter().copied().sum();
                        let g_a = &grad_arr * &y - &y.mapv(|v| v * sum_term);
                        propagate_grad(node, 0, &E::into_tensor_data(g_a), &mut node_grads)?;
                    });
                }
            }
        }
        Ok(())
    }
}

// ── backward 辅助函数 ─────────────────────────────────────────────────

/// 把多维索引（row-major / C order）展平为线性索引。
/// `multi` 长度必须与 `shape` 一致；每个维度值 < shape[d]。
fn flatten_index(multi: &[usize], shape: &[usize]) -> usize {
    let mut flat = 0usize;
    let mut stride = 1usize;
    for d in (0..multi.len()).rev() {
        flat += multi[d] * stride;
        stride *= shape[d];
    }
    flat
}

/// Reduce `grad` from the output shape down to `target_shape` by summing
/// over broadcast dimensions.  Follows numpy-style broadcasting rules.
/// 返回 Err 当 reshape 失败（方向 A：不再静默保留错误 shape）。
/// 阶段 4：泛型化，支持 f32 和 f64。
/// 护城河 F Phase 1：新增 `node_id` / `op` 参数，错误抛出时结构化填充 TapeErrorContext
/// （expected=target_shape, actual=grad/当前 shape），供 formal_explain 根因分析使用。
fn unbroadcast<E: FloatElem>(
    grad: &ArrayD<E>,
    target_shape: &[usize],
    node_id: usize,
    op: &str,
) -> Result<ArrayD<E>, crate::error::TenthError> {
    let grad_shape = grad.shape();
    if grad_shape == target_shape {
        return Ok(grad.clone());
    }

    let mut result = grad.clone();

    // Align shapes from the right.
    let g_ndim = grad_shape.len();
    let t_ndim = target_shape.len();

    // Pad target shape with 1s on the left to match grad ndim.
    let mut padded_target: Vec<usize> = vec![1; g_ndim.saturating_sub(t_ndim)];
    padded_target.extend_from_slice(target_shape);

    // For each axis where target is 1 and grad > 1, sum over that axis.
    for axis in (0..g_ndim).rev() {
        if padded_target[axis] == 1 && grad_shape[axis] > 1 {
            result = result.sum_axis(ndarray::Axis(axis));
        }
    }

    // Reshape to target if needed (sum_axis may keep trailing dims).
    let current_shape: Vec<usize> = result.shape().to_vec();
    if current_shape != target_shape {
        let total: usize = target_shape.iter().product();
        if total == result.len() {
            result = result.clone().into_shape_with_order(IxDyn(target_shape)).map_err(|_| {
                crate::error::TenthError::ShapeMismatch {
                    context: TapeErrorContext {
                        tape_node_id: node_id,
                        op: op.to_string(),
                        expected_shape: target_shape.to_vec(),
                        actual_shape: current_shape.clone(),
                    },
                    message: format!(
                        "unbroadcast reshape 失败：梯度 shape {:?} 无法 reshape 到目标 shape {:?}（方向 A：不再静默保留错误 shape）",
                        current_shape, target_shape
                    ),
                }
            })?;
        } else {
            return Err(crate::error::TenthError::ShapeMismatch {
                context: TapeErrorContext {
                    tape_node_id: node_id,
                    op: op.to_string(),
                    expected_shape: target_shape.to_vec(),
                    actual_shape: current_shape.clone(),
                },
                message: format!(
                    "unbroadcast 元素数不匹配：梯度 {} 元素，目标 {} 元素（shape {:?} → {:?}）",
                    result.len(), total, current_shape, target_shape
                ),
            });
        }
    }

    Ok(result)
}

/// Pure 2-D matrix multiplication returning an owned ArrayD.
/// 返回 Err 当输入非 2D（方向 A：不再静默返回零数组掩盖错误）。
/// 阶段 4：泛型化，支持 f32 和 f64。
fn matmul_2d<E: FloatElem>(a: &ArrayD<E>, b: &ArrayD<E>) -> Result<ArrayD<E>, crate::error::TenthError> {
    let a2 = a.view().into_dimensionality::<ndarray::Ix2>().map_err(|_| {
        crate::error::TenthError::RuntimeError { line: None, col: None,
            message: format!("matmul_2d 期望 2D 输入，实际 a shape = {:?}（方向 A：不再静默返回零数组）", a.shape()),
        }
    })?;
    let b2 = b.view().into_dimensionality::<ndarray::Ix2>().map_err(|_| {
        crate::error::TenthError::RuntimeError { line: None, col: None,
            message: format!("matmul_2d 期望 2D 输入，实际 b shape = {:?}（方向 A：不再静默返回零数组）", b.shape()),
        }
    })?;
    Ok(a2.dot(&b2).into_dyn())
}
