//! 索引操作。
//!
//! 从 `interpreter.rs` 第 3101-3212 行迁移而来。包含 `eval_index`，
//! 处理 String / Tensor / Vec 的下标与切片访问。

use crate::error::{TenthError, TenthResult};
use crate::hir::hir::*;
use crate::runtime::value::Value;

impl super::Interpreter {
    pub(super) fn eval_index(&mut self, target: &Value, indices: &[Index]) -> TenthResult<Value> {
        match target {
            Value::String(s) => {
                if indices.len() != 1 {
                    return Err(TenthError::RuntimeError {
                        message: "字符串索引只需要 1 个索引".into(),
                    });
                }
                match &indices[0] {
                    Index::Single(e) => {
                        let v = self.eval_expr(e)?.ok_or_else(|| TenthError::RuntimeError {
                            message: "索引为空值".into(),
                        })?;
                        let idx = v.as_int().unwrap_or(0) as usize;
                        s.chars().nth(idx).map(|c| Value::String(c.to_string())).ok_or_else(|| {
                            TenthError::RuntimeError {
                                message: format!("字符串索引 {} 越界", idx),
                            }
                        })
                    }
                    Index::Range { start, end } => {
                        let s_val = s.clone();
                        let start_idx = match start {
                            Some(e) => {
                                let v = self.eval_expr(e)?.ok_or_else(|| TenthError::RuntimeError {
                                    message: "范围起始为空值".into(),
                                })?;
                                v.as_int().unwrap_or(0) as usize
                            }
                            None => 0,
                        };
                        let end_idx = match end {
                            Some(e) => {
                                let v = self.eval_expr(e)?.ok_or_else(|| TenthError::RuntimeError {
                                    message: "范围结束为空值".into(),
                                })?;
                                v.as_int().unwrap_or(0) as usize
                            }
                            None => s_val.chars().count(),
                        };
                        let chars: Vec<char> = s_val.chars().collect();
                        if start_idx > chars.len() || end_idx > chars.len() || start_idx > end_idx {
                            return Err(TenthError::RuntimeError {
                                message: format!("字符串切片 {}..{} 越界", start_idx, end_idx),
                            });
                        }
                        let slice: String = chars[start_idx..end_idx].iter().collect();
                        Ok(Value::String(slice))
                    }
                    _ => Err(TenthError::RuntimeError {
                        message: "字符串索引必须是整数或范围".into(),
                    }),
                }
            }
            Value::Tensor(t) => {
                let tensor = t.borrow();
                let shape = tensor.shape();
                let mut idx: Vec<usize> = Vec::new();
                for (i, index_expr) in indices.iter().enumerate() {
                    match index_expr {
                        Index::Single(e) => {
                            let v = self.eval_expr(e)?.ok_or_else(|| TenthError::RuntimeError {
                                message: "索引为空值".into(),
                            })?;
                            idx.push(v.as_int().unwrap_or(0) as usize);
                        }
                        _ => {
                            if i < shape.len() {
                                idx.push(0);
                            }
                        }
                    }
                }
                match tensor.get(&idx) {
                    Some(val) => Ok(Value::Float(val)),
                    None => Err(TenthError::RuntimeError {
                        message: format!("索引 {:?} 越界，形状为 {:?}", idx, shape),
                    }),
                }
            }
            Value::Vec(items) => {
                if indices.len() != 1 {
                    return Err(TenthError::RuntimeError {
                        message: "Vec 索引只需要 1 个索引".into(),
                    });
                }
                match &indices[0] {
                    Index::Single(e) => {
                        let v = self.eval_expr(e)?.ok_or_else(|| TenthError::RuntimeError {
                            message: "索引为空值".into(),
                        })?;
                        let idx = v.as_int().unwrap_or(0) as usize;
                        // Elements are stored as Shared; return the Shared so
                        // field assignment can mutate through it.
                        match items.borrow().get(idx) {
                            Some(Value::Shared(rc)) => Ok(Value::Shared(rc.clone())),
                            Some(other) => Ok(other.clone()),
                            None => Err(TenthError::RuntimeError {
                                message: format!("Vec 索引 {} 越界", idx),
                            }),
                        }
                    }
                    _ => Err(TenthError::RuntimeError {
                        message: "Vec 索引必须是整数".into(),
                    }),
                }
            }
            _ => Err(TenthError::RuntimeError {
                message: "此类型不支持索引".into(),
            }),
        }
    }
}
