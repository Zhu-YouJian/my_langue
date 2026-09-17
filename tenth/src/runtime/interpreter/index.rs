//! 索引操作。
//!
//! 从 `interpreter.rs` 第 3101-3212 行迁移而来。包含 `eval_index`，
//! 处理 String / Tensor / Vec 的下标与切片访问。

use crate::error::{TenthError, TenthResult};
use crate::hir::hir::*;
use crate::runtime::value::Value;
use std::cell::RefCell;
use std::rc::Rc;

impl super::Interpreter {
    pub(super) fn eval_index(&mut self, target: &Value, indices: &[Index]) -> TenthResult<Value> {
        match target {
            Value::String(s) => {
                if indices.len() != 1 {
                    return Err(TenthError::RuntimeError { line: None, col: None,
                        message: "字符串索引只需要 1 个索引".into(),
                    });
                }
                match &indices[0] {
                    Index::Single(e) => {
                        let v = self.eval_expr(e)?.ok_or_else(|| TenthError::RuntimeError { line: None, col: None,
                            message: "索引为空值".into(),
                        })?;
                        let idx = v.as_int().unwrap_or(0) as usize;
                        s.chars().nth(idx).map(|c| Value::String(c.to_string())).ok_or_else(|| {
                            TenthError::RuntimeError { line: None, col: None,
                                message: format!("字符串索引 {} 越界", idx),
                            }
                        })
                    }
                    Index::Range { start, end } => {
                        let s_val = s.clone();
                        let start_i = match start {
                            Some(e) => {
                                let v = self.eval_expr(e)?.ok_or_else(|| TenthError::RuntimeError { line: None, col: None,
                                    message: "范围起始为空值".into(),
                                })?;
                                v.as_int().unwrap_or(0)
                            }
                            None => 0,
                        };
                        let end_i = match end {
                            Some(e) => {
                                let v = self.eval_expr(e)?.ok_or_else(|| TenthError::RuntimeError { line: None, col: None,
                                    message: "范围结束为空值".into(),
                                })?;
                                v.as_int().unwrap_or(0)
                            }
                            None => s_val.chars().count() as i64,
                        };
                        // AUDIT-11.4.89 / 11.4.93：字符串切片的单一权威实现与 VM/JIT/
                        // `str_slice` native 共用（码点 + 严格：越界/负索引/start>end 一律
                        // 响亮报错，不再各自手写一份）。此前 `as usize` 会把 `-1` 变成
                        // `usize::MAX`（报错文案里出现 18446744073709551615）。
                        crate::runtime::value::str_slice_codepoints(&s_val, start_i, end_i)
                            .map(Value::String)
                            .map_err(|msg| TenthError::RuntimeError { line: None, col: None, message: msg })
                    }
                    _ => Err(TenthError::RuntimeError { line: None, col: None,
                        message: "字符串索引必须是整数或范围".into(),
                    }),
                }
            }
            Value::Tensor(t) => {
                let tensor = t.borrow();
                let shape = tensor.shape();
                let ndim = shape.len();
                // 收集 Single 索引；Range/Colon 暂按 0 处理（保持原行为）。
                let mut idx: Vec<usize> = Vec::new();
                for index_expr in indices.iter() {
                    match index_expr {
                        Index::Single(e) => {
                            let v = self.eval_expr(e)?.ok_or_else(|| TenthError::RuntimeError { line: None, col: None,
                                message: "索引为空值".into(),
                            })?;
                            idx.push(v.as_int().unwrap_or(0) as usize);
                        }
                        // AUDIT-11.4.96（**红线级静默错值**）：此前 Range/Colon 走
                        // `_ => { if i < ndim { idx.push(0) } }` ⇒ `t[0..2]` / `t[:]`
                        // **静默等于 `t[0]`**（注释自称"暂按 0 处理，保持原行为"）。
                        // 张量切片四个后端都没有实现（VM/JIT 的 SliceStr 对张量目标
                        // 响亮报错、WASM 显式拒绝 `Vec range slicing`）⇒ 本波裁定为
                        // **响亮拒绝**（实现切片需 AST/HIR 多下标语义 + 四后端，且 JIT
                        // 属 W10 作业面不可动）——绝不允许继续静默当 0。
                        Index::Range { .. } | Index::Colon => {
                            let form = match index_expr {
                                Index::Colon => ":",
                                _ => "a..b",
                            };
                            return Err(TenthError::RuntimeError { line: None, col: None,
                                message: format!(
                                    "张量切片（索引 `{}`）未实现：请改用 index_select(t, dim, idx) 或显式整维索引",
                                    form
                                ),
                            });
                        }
                    }
                }
                if idx.is_empty() {
                    // 无有效索引：返回张量本身
                    return Ok(Value::Tensor(t.clone()));
                }
                if idx.len() > ndim {
                    return Err(TenthError::RuntimeError { line: None, col: None,
                        message: format!("索引数 {} 大于张量维度数 {}", idx.len(), ndim),
                    });
                }
                if idx.len() == ndim {
                    // 全索引：返回标量
                    match tensor.get(&idx) {
                        Some(val) => Ok(Value::Float(val)),
                        None => Err(TenthError::RuntimeError { line: None, col: None,
                            message: format!("索引 {:?} 越界，形状为 {:?}", idx, shape),
                        }),
                    }
                } else {
                    // 部分索引（idx.len() < ndim）：迭代沿第 0 维降维，返回子张量。
                    // NumPy 语义：t[0] 对 N-D 张量返回 (N-1)-D 子张量。
                    let mut sub = tensor.clone();
                    for i in &idx {
                        match sub.index_dim(*i) {
                            Ok(s) => sub = s,
                            Err(msg) => {
                                return Err(TenthError::RuntimeError { line: None, col: None, message: msg });
                            }
                        }
                    }
                    Ok(Value::Tensor(Rc::new(RefCell::new(sub))))
                }
            }
            Value::Vec(items) => {
                if indices.len() != 1 {
                    return Err(TenthError::RuntimeError { line: None, col: None,
                        message: "Vec 索引只需要 1 个索引".into(),
                    });
                }
                match &indices[0] {
                    Index::Single(e) => {
                        let v = self.eval_expr(e)?.ok_or_else(|| TenthError::RuntimeError { line: None, col: None,
                            message: "索引为空值".into(),
                        })?;
                        let idx = v.as_int().unwrap_or(0) as usize;
                        // Elements are stored as Shared; return the Shared so
                        // field assignment can mutate through it.
                        match items.borrow().get(idx) {
                            Some(Value::Shared(rc)) => Ok(Value::Shared(rc.clone())),
                            Some(other) => Ok(other.clone()),
                            None => Err(TenthError::RuntimeError { line: None, col: None,
                                message: format!("Vec 索引 {} 越界", idx),
                            }),
                        }
                    }
                    _ => Err(TenthError::RuntimeError { line: None, col: None,
                        message: "Vec 索引必须是整数".into(),
                    }),
                }
            }
            _ => Err(TenthError::RuntimeError { line: None, col: None,
                message: "此类型不支持索引".into(),
            }),
        }
    }
}
