//! VM 方法分派：call_method_priv（String/Vec/Map/Tensor/Struct/Float/Int 方法）。
//!
//! 从 runtime/vm.rs 拆分而来（T3b 架构重构）。

use std::rc::Rc;
use crate::hir::types::BaseType;
use std::cell::RefCell;
use crate::error::{TenthError, TenthResult};
use crate::runtime::value::Value;
use crate::runtime::autodiff::TapeOp;
use crate::runtime::tensor::Tensor;

use super::Vm;
use super::err;

/// 问题2：将 Value 键转换为 HashMap 内部存储的 String 键（VM 端）。
/// 支持 String / Int / Bool / Float，其他类型返回错误。
fn vm_map_key_to_string(v: &Value) -> TenthResult<String> {
    match v {
        Value::String(s) => Ok(s.clone()),
        Value::Int(n, _) => Ok(n.to_string()),
        Value::Bool(b) => Ok(b.to_string()),
        Value::Float32(f) => Ok(format!("{}", f)),
        Value::Float(f) => Ok(format!("{}", f)),
        _ => err(&format!("Map 的键类型不支持: {:?}（仅支持 str/int/bool/float）", v)),
    }
}

impl Vm {
    pub(super) fn call_method_priv(&mut self, receiver: &Value, method: &str, args: &[Value]) -> TenthResult<Value> {
        // Auto-deref via cloning (avoids borrow issues)
        let recv = match receiver {
            Value::Ref(rc) => rc.borrow().clone(),
            Value::MutRef(w) => w.upgrade().map(|rc| rc.borrow().clone()).unwrap_or(Value::Moved),
            Value::Shared(rc) => rc.borrow().clone(),
            v => v.clone(),
        };
        match recv {
            Value::String(s) => match method {
                "len" => Ok(Value::Int(s.chars().count() as i64, BaseType::I32)),
                "trim" => Ok(Value::String(s.trim().to_string())),
                "to_upper" => Ok(Value::String(s.to_uppercase())),
                "to_lower" => Ok(Value::String(s.to_lowercase())),
                "replace" => {
                    if args.len() >= 2 {
                        if let (Value::String(from), Value::String(to)) = (&args[0], &args[1]) {
                            Ok(Value::String(s.replace(from.as_str(), to.as_str())))
                        } else { err("replace() 需要 2 个字符串参数") }
                    } else { err("replace() 需要 2 个字符串参数") }
                }
                "split" => {
                    if let Some(Value::String(delim)) = args.first() {
                        let parts: Vec<Value> = s.split(delim.as_str()).map(|p| Value::String(p.to_string())).collect();
                        Ok(Value::Vec(Rc::new(RefCell::new(parts))))
                    } else { err("split() 需要一个字符串分隔符") }
                }
                "substring" => {
                    if args.len() >= 2 {
                        let start = args[0].as_int().unwrap_or(0).max(0) as usize;
                        let len = args[1].as_int().unwrap_or(0).max(0) as usize;
                        let chars: Vec<char> = s.chars().collect();
                        let end = (start + len).min(chars.len());
                        let sub: String = chars[start..end].iter().collect();
                        Ok(Value::String(sub))
                    } else { err("substring() 需要起始位置和长度") }
                }
                "contains" => {
                    if let Some(Value::String(sub)) = args.first() {
                        Ok(Value::Bool(s.contains(sub.as_str())))
                    } else { err("contains() 需要一个字符串参数") }
                }
                "find" => {
                    if let Some(Value::String(sub)) = args.first() {
                        Ok(Value::Int(s.find(sub.as_str()).map(|i| i as i64).unwrap_or(-1), BaseType::I32))
                    } else { err("find() 需要一个字符串参数") }
                }
                "starts_with" => {
                    if let Some(Value::String(prefix)) = args.first() {
                        Ok(Value::Bool(s.starts_with(prefix.as_str())))
                    } else { err("starts_with() 需要一个字符串参数") }
                }
                "ends_with" => {
                    if let Some(Value::String(suffix)) = args.first() {
                        Ok(Value::Bool(s.ends_with(suffix.as_str())))
                    } else { err("ends_with() 需要一个字符串参数") }
                }
                "parse_int" => Ok(Value::Int(s.trim().parse::<i64>().unwrap_or(0), BaseType::I32)),
                "parse_float" => Ok(Value::Float(s.trim().parse::<f64>().unwrap_or(0.0))),
                "is_empty" => Ok(Value::Bool(s.is_empty())),
                "repeat" => {
                    if let Some(arg) = args.first() {
                        let n = arg.as_int().unwrap_or(0).max(0) as usize;
                        Ok(Value::String(s.repeat(n)))
                    } else { err("repeat() 需要一个整数参数") }
                }
                "chars" => {
                    let chars: Vec<Value> = s.chars().map(|c| Value::String(c.to_string())).collect();
                    Ok(Value::Vec(Rc::new(RefCell::new(chars))))
                }
                "bytes" => {
                    let bytes: Vec<Value> = s.bytes().map(|b| Value::Int(b as i64, BaseType::I32)).collect();
                    Ok(Value::Vec(Rc::new(RefCell::new(bytes))))
                }
                "trim_start" => Ok(Value::String(s.trim_start().to_string())),
                "trim_end" => Ok(Value::String(s.trim_end().to_string())),
                "strip_prefix" => {
                    if let Some(Value::String(prefix)) = args.first() {
                        Ok(match s.strip_prefix(prefix.as_str()) {
                            Some(rest) => Value::String(rest.to_string()),
                            None => Value::String(s.to_string()),
                        })
                    } else { err("strip_prefix() 需要一个字符串参数") }
                }
                "strip_suffix" => {
                    if let Some(Value::String(suffix)) = args.first() {
                        Ok(match s.strip_suffix(suffix.as_str()) {
                            Some(rest) => Value::String(rest.to_string()),
                            None => Value::String(s.to_string()),
                        })
                    } else { err("strip_suffix() 需要一个字符串参数") }
                }
                _ => err(&format!("字符串没有方法 '{}'", method)),
            },
            Value::Vec(items) => match method {
                "len" => Ok(Value::Int(items.borrow().len() as i64, BaseType::I32)),
                "push" => {
                    if args.len() == 1 {
                        items.borrow_mut().push(args[0].clone());
                        Ok(Value::Unit)
                    } else { err("push 需要 1 个参数") }
                }
                "get" => {
                    if args.len() == 1 {
                        let idx = args[0].as_int().unwrap_or(0) as usize;
                        Ok(items.borrow().get(idx).cloned().unwrap_or(Value::Unit))
                    } else { err("get 需要 1 个参数") }
                }
                "pop" => {
                    let mut vec = items.borrow_mut();
                    match vec.pop() {
                        Some(v) => Ok(v),
                        None => err("对空 Vec 调用 pop()"),
                    }
                }
                "set" => {
                    if args.len() != 2 { return err("set() 需要 2 个参数 (索引, 值)"); }
                    let idx = args[0].as_int().unwrap_or(0) as usize;
                    let mut vec = items.borrow_mut();
                    if idx < vec.len() {
                        vec[idx] = args[1].clone();
                        Ok(Value::Unit)
                    } else { err(&format!("Vec 索引 {} 越界", idx)) }
                }
                "clear" => {
                    items.borrow_mut().clear();
                    Ok(Value::Unit)
                }
                "contains" => {
                    if args.len() != 1 { return err("contains() 需要 1 个参数"); }
                    let vec = items.borrow();
                    let found = vec.iter().any(|v| self.vm_eq(v, &args[0]));
                    Ok(Value::Bool(found))
                }
                "insert" => {
                    if args.len() != 2 { return err("insert() 需要 2 个参数 (索引, 值)"); }
                    let idx = args[0].as_int().unwrap_or(0) as usize;
                    items.borrow_mut().insert(idx, args[1].clone());
                    Ok(Value::Unit)
                }
                "remove" => {
                    if args.len() != 1 { return err("remove() 需要 1 个参数 (索引)"); }
                    let idx = args[0].as_int().unwrap_or(0) as usize;
                    let vec_len = items.borrow().len();
                    if idx < vec_len {
                        Ok(items.borrow_mut().remove(idx))
                    } else { err(&format!("Vec 索引 {} 越界", idx)) }
                }
                "join" => {
                    if args.len() != 1 { return err("join() 需要 1 个参数 (分隔符)"); }
                    if let Value::String(delim) = &args[0] {
                        let vec = items.borrow();
                        let parts: Vec<String> = vec.iter().map(|v| match v {
                            Value::String(s) => s.clone(),
                            other => format!("{:?}", other),
                        }).collect();
                        Ok(Value::String(parts.join(delim)))
                    } else { err("join() 分隔符必须是字符串") }
                }
                "is_empty" => Ok(Value::Bool(items.borrow().is_empty())),
                // AUDIT-11.4.28：补齐 VM Vec 方法（语义对齐解释器 methods.rs eval_vec_method）。
                "index_of" => {
                    if args.len() != 1 { return err("index_of() 需要 1 个参数"); }
                    let vec = items.borrow();
                    for (i, v) in vec.iter().enumerate() {
                        if self.vm_eq(v, &args[0]) {
                            return Ok(Value::Int(i as i64, BaseType::I32));
                        }
                    }
                    Ok(Value::Int(-1, BaseType::I32))
                }
                "reverse" => {
                    let vec = items.borrow();
                    let reversed: Vec<Value> = vec.iter().rev().cloned().collect();
                    Ok(Value::Vec(Rc::new(RefCell::new(reversed))))
                }
                "slice" => {
                    if args.len() != 2 { return err("slice() 需要 2 个参数 (起始, 结束)"); }
                    let start = args[0].as_int().unwrap_or(0).max(0) as usize;
                    let end = args[1].as_int().unwrap_or(0).max(0) as usize;
                    let vec = items.borrow();
                    let end = end.min(vec.len());
                    if start > end {
                        return Ok(Value::Vec(Rc::new(RefCell::new(Vec::new()))));
                    }
                    let sliced: Vec<Value> = vec[start..end].to_vec();
                    Ok(Value::Vec(Rc::new(RefCell::new(sliced))))
                }
                "extend" => {
                    if args.len() != 1 { return err("extend() 需要 1 个参数 (Vec)"); }
                    if let Value::Vec(other) = &args[0] {
                        let other_vals = other.borrow().clone();
                        let mut vec = items.borrow_mut();
                        for v in other_vals { vec.push(v); }
                        return Ok(Value::Unit);
                    }
                    err("extend() 参数必须是 Vec")
                }
                "sort" => {
                    let mut vec = items.borrow_mut();
                    vec.sort_by(|a, b| {
                        match (a, b) {
                            (Value::Int(x, _), Value::Int(y, _)) => x.cmp(y),
                            (Value::Float(x), Value::Float(y)) => x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal),
                            (Value::String(x), Value::String(y)) => x.cmp(y),
                            _ => std::cmp::Ordering::Equal,
                        }
                    });
                    Ok(Value::Unit)
                }
                "dedup" => {
                    let mut vec = items.borrow_mut();
                    vec.dedup_by(|a, b| self.vm_eq(a, b));
                    Ok(Value::Unit)
                }
                "first" => {
                    let vec = items.borrow();
                    match vec.first() {
                        Some(v) => Ok(v.clone()),
                        None => Ok(Value::Unit),
                    }
                }
                "last" => {
                    let vec = items.borrow();
                    match vec.last() {
                        Some(v) => Ok(v.clone()),
                        None => Ok(Value::Unit),
                    }
                }
                "flatten" => {
                    let vec = items.borrow();
                    let mut result = Vec::new();
                    for v in vec.iter() {
                        match v {
                            Value::Vec(inner) => {
                                for item in inner.borrow().iter() {
                                    result.push(item.clone());
                                }
                            }
                            other => result.push(other.clone()),
                        }
                    }
                    Ok(Value::Vec(Rc::new(RefCell::new(result))))
                }
                "chunks" => {
                    if args.len() != 1 { return err("chunks() 需要 1 个参数 (大小)"); }
                    let size = args[0].as_int().unwrap_or(1).max(1) as usize;
                    let vec = items.borrow();
                    let mut result = Vec::new();
                    for chunk in vec.chunks(size) {
                        let c: Vec<Value> = chunk.to_vec();
                        result.push(Value::Vec(Rc::new(RefCell::new(c))));
                    }
                    Ok(Value::Vec(Rc::new(RefCell::new(result))))
                }
                _ => err(&format!("Vec 没有方法 '{}'", method)),
            },
            Value::Map(m) => match method {
                "len" => Ok(Value::Int(m.borrow().len() as i64, BaseType::I32)),
                "insert" => {
                    if args.len() != 2 { return err("insert() 需要 2 个参数 (键, 值)"); }
                    // 问题2：支持 str/int/bool/float 键（内部统一转 String 存储）
                    let key = vm_map_key_to_string(&args[0])?;
                    m.borrow_mut().insert(key, args[1].clone());
                    Ok(Value::Unit)
                }
                "get" => {
                    if args.len() != 1 { return err("get() 需要 1 个参数 (键)"); }
                    let key = vm_map_key_to_string(&args[0])?;
                    Ok(m.borrow().get(&key).cloned().unwrap_or(Value::Unit))
                }
                "contains_key" => {
                    if args.len() != 1 { return err("contains_key() 需要 1 个参数 (键)"); }
                    let key = vm_map_key_to_string(&args[0])?;
                    Ok(Value::Bool(m.borrow().contains_key(&key)))
                }
                "remove" => {
                    if args.len() != 1 { return err("remove() 需要 1 个参数 (键)"); }
                    let key = vm_map_key_to_string(&args[0])?;
                    Ok(m.borrow_mut().remove(&key).unwrap_or(Value::Unit))
                }
                "keys" => {
                    let keys: Vec<Value> = m.borrow().keys().map(|k| Value::String(k.clone())).collect();
                    Ok(Value::Vec(Rc::new(RefCell::new(keys))))
                }
                "values" => {
                    let values: Vec<Value> = m.borrow().values().cloned().collect();
                    Ok(Value::Vec(Rc::new(RefCell::new(values))))
                }
                "is_empty" => Ok(Value::Bool(m.borrow().is_empty())),
                // AUDIT-11.4.28：Map.entries（语义对齐解释器 methods.rs eval_map_method）——
                // 返回 [[key, value], ...] 的 Vec。阻塞 map_values/filter_map 的根因。
                "entries" => {
                    let entries: Vec<Value> = m.borrow().iter().map(|(k, v)| {
                        Value::Vec(Rc::new(RefCell::new(vec![
                            Value::String(k.clone()),
                            v.clone(),
                        ])))
                    }).collect();
                    Ok(Value::Vec(Rc::new(RefCell::new(entries))))
                }
                _ => err(&format!("Map 没有方法 '{}'", method)),
            },
            Value::Tensor(t) => {
                let tensor = t.borrow();
                match method {
                    // ── Reductions ──
                    "sum" => {
                        if args.is_empty() {
                            if self.recording {
                                let scalar = tensor.sum();
                                let result = Rc::new(RefCell::new(Tensor::from_vec(vec![scalar], vec![1])));
                                self.record_unary(TapeOp::Sum, &t, &result);
                                Ok(Value::Tensor(result))
                            } else if tensor.is_f32() {
                                Ok(Value::Float32(tensor.sum() as f32))
                            } else {
                                Ok(Value::Float(tensor.sum()))
                            }
                        } else {
                            let axis = args[0].as_int().unwrap_or(0) as usize;
                            match tensor.sum_axis(axis) {
                                Ok(t) => Ok(Value::Tensor(Rc::new(RefCell::new(t)))),
                                Err(msg) => err(&msg),
                            }
                        }
                    }
                    "mean" => {
                        if self.recording {
                            let scalar = tensor.mean();
                            let result = Rc::new(RefCell::new(Tensor::from_vec(vec![scalar], vec![1])));
                            self.record_unary(TapeOp::Mean, &t, &result);
                            Ok(Value::Tensor(result))
                        } else if tensor.is_f32() {
                            Ok(Value::Float32(tensor.mean() as f32))
                        } else {
                            Ok(Value::Float(tensor.mean()))
                        }
                    }
                    "max_val" => {
                        if tensor.is_f32() {
                            Ok(Value::Float32(tensor.max_val() as f32))
                        } else {
                            Ok(Value::Float(tensor.max_val()))
                        }
                    }

                    // ── Elementwise unary ──
                    "abs" => {
                        let result = Rc::new(RefCell::new(tensor.abs()));
                        if self.recording { self.record_unary(TapeOp::Abs, &t, &result); }
                        Ok(Value::Tensor(result))
                    }
                    "sqrt" => {
                        let result = Rc::new(RefCell::new(tensor.sqrt()));
                        Ok(Value::Tensor(result))
                    }
                    "exp" => {
                        let result = Rc::new(RefCell::new(tensor.exp()));
                        if self.recording { self.record_unary(TapeOp::Exp, &t, &result); }
                        Ok(Value::Tensor(result))
                    }
                    "log" => {
                        let result = Rc::new(RefCell::new(tensor.log()));
                        if self.recording { self.record_unary(TapeOp::Log, &t, &result); }
                        Ok(Value::Tensor(result))
                    }
                    "relu" => {
                        let result = Rc::new(RefCell::new(tensor.relu()));
                        if self.recording { self.record_unary(TapeOp::ReLU, &t, &result); }
                        Ok(Value::Tensor(result))
                    }
                    "sigmoid" => {
                        let result = Rc::new(RefCell::new(tensor.sigmoid()));
                        if self.recording { self.record_unary(TapeOp::Sigmoid, &t, &result); }
                        Ok(Value::Tensor(result))
                    }
                    "tanh" => {
                        let result = Rc::new(RefCell::new(tensor.tanh()));
                        Ok(Value::Tensor(result))
                    }
                    "gelu" => {
                        let result = Rc::new(RefCell::new(tensor.gelu()));
                        if self.recording { self.record_unary(TapeOp::Gelu, &t, &result); }
                        Ok(Value::Tensor(result))
                    }
                    "softmax" => {
                        let result_tensor = tensor.softmax().ok_or_else(|| {
                            TenthError::RuntimeError { line: None, col: None, message: "softmax 计算失败".into() }
                        })?;
                        let result = Rc::new(RefCell::new(result_tensor));
                        if self.recording { self.record_unary(TapeOp::Softmax, &t, &result); }
                        Ok(Value::Tensor(result))
                    }
                    "argmax" => Ok(Value::Int(tensor.argmax(), BaseType::I32)),
                    // 梯度裁剪辅助：元素级裁剪到 [min_val, max_val]（与 interpreter 同步）
                    "clip_scalar" => {
                        if args.len() < 2 {
                            return Err(TenthError::RuntimeError { line: None, col: None,
                                message: "clip_scalar() 需要 min_val 和 max_val".into(),
                            });
                        }
                        let min_val = args[0].as_float().unwrap_or(f64::NEG_INFINITY);
                        let max_val = args[1].as_float().unwrap_or(f64::INFINITY);
                        let clipped = tensor.clip_scalar(min_val, max_val);
                        Ok(Value::Tensor(Rc::new(RefCell::new(clipped))))
                    }
                    // 张量属性查询（配合护城河 D 内存预估，与 interpreter 同步）
                    // len = 第 0 维长度（行数），NumPy 语义；VM 通用 for-in 迭代张量依赖它
                    "len" => Ok(Value::Int(tensor.shape().first().copied().unwrap_or(0) as i64, BaseType::I32)),
                    "numel" => Ok(Value::Int(tensor.data.len() as i64, BaseType::I32)),
                    "nbytes" | "bytes" => {
                        let n = tensor.data.len() as i64;
                        let bytes_per_elem: i64 = match &tensor.data {
                            crate::runtime::tensor::TensorData::F64(_) => 8,
                            crate::runtime::tensor::TensorData::F32(_) => 4,
                            crate::runtime::tensor::TensorData::F16(_) => 2,
                            crate::runtime::tensor::TensorData::BF16(_) => 2,
                        };
                        Ok(Value::Int(n * bytes_per_elem, BaseType::I32))
                    }
                    "ndim" | "rank" => Ok(Value::Int(tensor.data.ndim() as i64, BaseType::I32)),
                    "shape_tensor" => {
                        // 返回 shape 作为 f64 tensor（便于运行时查询）
                        let shape: Vec<f64> = tensor.data.shape().iter().map(|&d| d as f64).collect();
                        let len = shape.len();
                        Ok(Value::Tensor(Rc::new(RefCell::new(
                            Tensor::from_vec(shape, vec![len])
                        ))))
                    }

                    // ── Shape operations ──
                    "reshape" | "view" => {
                        let shape: Vec<usize> = args.iter()
                            .map(|a| a.as_int().unwrap_or(1) as usize)
                            .collect();
                        let result_tensor = tensor.reshape(&shape).ok_or_else(|| {
                            TenthError::RuntimeError { line: None, col: None, message: format!("无法重塑形状为 {:?}", shape) }
                        })?;
                        let result = Rc::new(RefCell::new(result_tensor));
                        if self.recording { self.record_unary(TapeOp::Reshape, &t, &result); }
                        Ok(Value::Tensor(result))
                    }
                    "flatten" => Ok(Value::Tensor(Rc::new(RefCell::new(tensor.flatten())))),
                    "transpose" => {
                        let result_tensor = tensor.transpose().ok_or_else(|| {
                            TenthError::RuntimeError { line: None, col: None, message: "转置至少需要 2 个维度".into() }
                        })?;
                        let result = Rc::new(RefCell::new(result_tensor));
                        if self.recording { self.record_unary(TapeOp::Transpose, &t, &result); }
                        Ok(Value::Tensor(result))
                    }
                    "permute" => {
                        let dims: Vec<usize> = args.iter()
                            .map(|a| a.as_int().unwrap_or(0) as usize)
                            .collect();
                        let result = tensor.permute(&dims)
                            .map_err(|msg| TenthError::RuntimeError { line: None, col: None, message: msg })?;
                        Ok(Value::Tensor(Rc::new(RefCell::new(result))))
                    }
                    "broadcast_to" => {
                        let target_shape: Vec<usize> = args.iter()
                            .map(|a| a.as_int().unwrap_or(1) as usize)
                            .collect();
                        let result = tensor.broadcast_to(&target_shape).ok_or_else(|| {
                            TenthError::RuntimeError { line: None, col: None, message: format!("无法广播到 {:?}", target_shape) }
                        })?;
                        Ok(Value::Tensor(Rc::new(RefCell::new(result))))
                    }
                    "cat" => {
                        if args.is_empty() {
                            return err("cat() 至少需要 1 个参数 (other, [dim])");
                        }
                        let dim = args.get(1).and_then(|a| a.as_int()).unwrap_or(0) as usize;
                        if let Value::Tensor(other) = &args[0] {
                            let result = tensor.cat(&other.borrow(), dim)
                                .map_err(|msg| TenthError::RuntimeError { line: None, col: None, message: msg })?;
                            Ok(Value::Tensor(Rc::new(RefCell::new(result))))
                        } else {
                            err("cat() 第一个参数必须是张量")
                        }
                    }
                    "masked_fill" => {
                        if args.len() < 2 {
                            return err("masked_fill() 需要 mask 和 value 参数");
                        }
                        let value = args[1].as_float().unwrap_or(0.0);
                        if let Value::Tensor(mask_rc) = &args[0] {
                            let result_tensor = tensor.masked_fill(&mask_rc.borrow(), value)
                                .map_err(|msg| TenthError::RuntimeError { line: None, col: None, message: msg })?;
                            let result = Rc::new(RefCell::new(result_tensor));
                            if self.recording {
                                if let Some(ref mut tape) = self.tape {
                                    let input_id = t.borrow().tape_id;
                                    let node_id = tape.masked_fill(input_id, t.clone(), mask_rc.clone(), result.clone());
                                    result.borrow_mut().tape_id = Some(node_id);
                                }
                            }
                            Ok(Value::Tensor(result))
                        } else {
                            err("masked_fill() 的 mask 必须是张量")
                        }
                    }

                    // ── Matrix / NN operations ──
                    "matmul" => {
                        if args.len() != 1 {
                            return err("matmul() 需要 1 个参数");
                        }
                        if let Value::Tensor(other) = &args[0] {
                            let result_tensor = tensor.matmul(&other.borrow())
                                .map_err(|msg| TenthError::RuntimeError { line: None, col: None, message: msg })?;
                            let result = Rc::new(RefCell::new(result_tensor));
                            if self.recording { self.record_binary(TapeOp::MatMul, &t, &other, &result); }
                            Ok(Value::Tensor(result))
                        } else {
                            err("matmul() 参数必须是张量")
                        }
                    }
                    "bmm" => {
                        // batched matmul: (B, M, K) @ (B, K, N) -> (B, M, N)
                        if args.len() != 1 {
                            return err("bmm() 需要 1 个参数");
                        }
                        if let Value::Tensor(other) = &args[0] {
                            let result_tensor = tensor.bmm(&other.borrow())
                                .map_err(|msg| TenthError::RuntimeError { line: None, col: None, message: msg })?;
                            let result = Rc::new(RefCell::new(result_tensor));
                            if self.recording { self.record_binary(TapeOp::BatchedMatMul, &t, &other, &result); }
                            Ok(Value::Tensor(result))
                        } else {
                            err("bmm() 参数必须是张量")
                        }
                    }
                    "conv2d" => {
                        // x.conv2d(w, kernel_h, kernel_w, stride, pad)
                        if args.len() < 5 {
                            return err("conv2d() 需要 5 个参数: w, kH, kW, stride, pad");
                        }
                        let k_h = args[1].as_int().unwrap_or(3) as usize;
                        let k_w = args[2].as_int().unwrap_or(3) as usize;
                        let stride = args[3].as_int().unwrap_or(1) as usize;
                        let pad = args[4].as_int().unwrap_or(0) as usize;
                        if let Value::Tensor(w_rc) = &args[0] {
                            let w_data = w_rc.borrow();
                            let w_shape = w_data.shape();
                            // Validate weight shape: must be 4D (C_out, C_in, kH, kW)
                            if w_shape.len() != 4 {
                                return err(&format!(
                                    "conv2d: 权重必须是 4D (C_out, C_in, kH, kW)，得到 {:?}D",
                                    w_shape.len()
                                ));
                            }
                            if w_shape[2] != k_h || w_shape[3] != k_w {
                                return err(&format!(
                                    "conv2d: 权重 kernel 尺寸 {:?} 与参数 kH={}, kW={} 不匹配",
                                    &w_shape[2..4], k_h, k_w
                                ));
                            }
                            let (cols, h_out, w_out) = tensor.im2col(k_h, k_w, stride, pad)
                                .ok_or_else(|| TenthError::RuntimeError { line: None, col: None,
                                    message: "im2col 失败（输入必须是 4D）".into(),
                                })?;
                            let c_out = w_shape[0];
                            let w_flat = w_data.reshape(&[c_out, w_shape[1] * w_shape[2] * w_shape[3]])
                                .ok_or_else(|| TenthError::RuntimeError { line: None, col: None,
                                    message: "权重重塑失败".into(),
                                })?;
                            let output_2d = cols.matmul(&w_flat.transpose()
                                .ok_or_else(|| TenthError::RuntimeError { line: None, col: None,
                                    message: "权重转置失败".into(),
                                })?).map_err(|msg| TenthError::RuntimeError { line: None, col: None, message: msg })?;
                            let n = tensor.shape()[0];
                            let result_tensor = output_2d.reshape(&[n, c_out, h_out, w_out])
                                .ok_or_else(|| TenthError::RuntimeError { line: None, col: None,
                                    message: "输出重塑失败".into(),
                                })?;
                            let result = Rc::new(RefCell::new(result_tensor));
                            if self.recording {
                                let cols_rc = Rc::new(RefCell::new(cols));
                                if let Some(ref mut tape) = self.tape {
                                    let x_id = t.borrow().tape_id
                                        .unwrap_or_else(|| tape.input(t.clone()));
                                    let w_id = w_rc.borrow().tape_id
                                        .unwrap_or_else(|| tape.input(w_rc.clone()));
                                    let node_id = tape.conv2d(
                                        x_id, t.clone(),
                                        w_id, w_rc.clone(),
                                        cols_rc, result.clone(),
                                    );
                                    result.borrow_mut().tape_id = Some(node_id);
                                }
                            }
                            Ok(Value::Tensor(result))
                        } else {
                            err("conv2d: 权重必须是张量")
                        }
                    }
                    "max_pool2d" => {
                        // x.max_pool2d(kH, kW, sH, sW, pH, pW) — PyTorch 语义
                        if args.len() < 6 {
                            return err("max_pool2d() 需要 6 个参数: kH, kW, sH, sW, pH, pW");
                        }
                        let k_h = args[0].as_int().unwrap_or(2) as usize;
                        let k_w = args[1].as_int().unwrap_or(2) as usize;
                        let s_h = args[2].as_int().unwrap_or(2) as usize;
                        let s_w = args[3].as_int().unwrap_or(2) as usize;
                        let p_h = args[4].as_int().unwrap_or(0) as usize;
                        let p_w = args[5].as_int().unwrap_or(0) as usize;
                        let (output, _argmax_mask) = tensor.max_pool2d_with_argmax(k_h, k_w, s_h, s_w, p_h, p_w)
                            .map_err(|msg| TenthError::RuntimeError { line: None, col: None, message: msg })?;
                        let result = Rc::new(RefCell::new(output));
                        if self.recording {
                            if let Some(ref mut tape) = self.tape {
                                let x_id = t.borrow().tape_id
                                    .unwrap_or_else(|| tape.input(t.clone()));
                                let node_id = tape.max_pool2d(
                                    Some(x_id), t.clone(), result.clone(),
                                    k_h, k_w, s_h, s_w, p_h, p_w,
                                );
                                result.borrow_mut().tape_id = Some(node_id);
                            }
                        }
                        Ok(Value::Tensor(result))
                    }
                    "avg_pool2d" => {
                        // x.avg_pool2d(kH, kW, sH, sW, pH, pW) — count_include_pad=False
                        if args.len() < 6 {
                            return err("avg_pool2d() 需要 6 个参数: kH, kW, sH, sW, pH, pW");
                        }
                        let k_h = args[0].as_int().unwrap_or(2) as usize;
                        let k_w = args[1].as_int().unwrap_or(2) as usize;
                        let s_h = args[2].as_int().unwrap_or(2) as usize;
                        let s_w = args[3].as_int().unwrap_or(2) as usize;
                        let p_h = args[4].as_int().unwrap_or(0) as usize;
                        let p_w = args[5].as_int().unwrap_or(0) as usize;
                        let output = tensor.avg_pool2d(k_h, k_w, s_h, s_w, p_h, p_w)
                            .map_err(|msg| TenthError::RuntimeError { line: None, col: None, message: msg })?;
                        let result = Rc::new(RefCell::new(output));
                        if self.recording {
                            if let Some(ref mut tape) = self.tape {
                                let x_id = t.borrow().tape_id
                                    .unwrap_or_else(|| tape.input(t.clone()));
                                let node_id = tape.avg_pool2d(
                                    Some(x_id), t.clone(), result.clone(),
                                    k_h, k_w, s_h, s_w, p_h, p_w,
                                );
                                result.borrow_mut().tape_id = Some(node_id);
                            }
                        }
                        Ok(Value::Tensor(result))
                    }
                    "batchnorm" => {
                        // x.batchnorm(gamma, beta, eps)
                        if args.len() < 3 {
                            return err("batchnorm() 需要 gamma, beta, eps 参数");
                        }
                        let eps = args[2].as_float().unwrap_or(1e-5);
                        if let (Value::Tensor(gamma_rc), Value::Tensor(beta_rc)) = (&args[0], &args[1]) {
                            let x_shape = tensor.shape();
                            if x_shape.len() < 2 {
                                return err("batchnorm 至少需要 2D 输入");
                            }
                            let c = x_shape[1];
                            let n = x_shape[0];
                            let spatial: usize = x_shape[2..].iter().product();
                            let x_flat = tensor.data.as_standard_layout().to_owned();
                            let x_slice = x_flat.as_slice().unwrap_or(&[]);
                            let gamma_ref = gamma_rc.borrow();
                            let beta_ref = beta_rc.borrow();
                            let g_flat = gamma_ref.data.as_standard_layout().to_owned();
                            let b_flat = beta_ref.data.as_standard_layout().to_owned();
                            let g_slice = g_flat.as_slice().unwrap_or(&[]);
                            let b_slice = b_flat.as_slice().unwrap_or(&[]);
                            let mut result_data = Vec::with_capacity(x_slice.len());
                            let mut x_hat_data = Vec::with_capacity(x_slice.len());
                            let mut std_inv_data = Vec::with_capacity(c);
                            for ci in 0..c {
                                let mut sum = 0.0;
                                let mut count = 0;
                                for ni in 0..n {
                                    for si in 0..spatial {
                                        let idx = ((ni * c + ci) * spatial) + si;
                                        if idx < x_slice.len() { sum += x_slice[idx]; count += 1; }
                                    }
                                }
                                let mean = if count > 0 { sum / count as f64 } else { 0.0 };
                                let mut var_sum = 0.0;
                                for ni in 0..n {
                                    for si in 0..spatial {
                                        let idx = ((ni * c + ci) * spatial) + si;
                                        if idx < x_slice.len() { let d = x_slice[idx] - mean; var_sum += d * d; }
                                    }
                                }
                                let var = if count > 0 { var_sum / count as f64 } else { 1.0 };
                                let std_inv = 1.0 / (var + eps).sqrt();
                                std_inv_data.push(std_inv);
                                let g = g_slice.get(ci).copied().unwrap_or(1.0);
                                let b = b_slice.get(ci).copied().unwrap_or(0.0);
                                for ni in 0..n {
                                    for si in 0..spatial {
                                        let idx = ((ni * c + ci) * spatial) + si;
                                        if idx < x_slice.len() {
                                            let x_hat = (x_slice[idx] - mean) * std_inv;
                                            x_hat_data.push(x_hat);
                                            result_data.push(g * x_hat + b);
                                        }
                                    }
                                }
                            }
                            let result = Rc::new(RefCell::new(Tensor::from_vec(result_data, x_shape.clone())));
                            if self.recording {
                                let x_hat = Rc::new(RefCell::new(Tensor::from_vec(x_hat_data, x_shape.clone())));
                                let std_inv_tensor = Rc::new(RefCell::new(Tensor::from_vec(std_inv_data, vec![c])));
                                if let Some(ref mut tape) = self.tape {
                                    let x_id = t.borrow().tape_id
                                        .unwrap_or_else(|| tape.input(t.clone()));
                                    let node_id = tape.batchnorm(
                                        x_id, t.clone(),
                                        gamma_rc.clone(), beta_rc.clone(),
                                        x_hat, std_inv_tensor, result.clone(),
                                    );
                                    result.borrow_mut().tape_id = Some(node_id);
                                }
                            }
                            Ok(Value::Tensor(result))
                        } else {
                            err("batchnorm: gamma 和 beta 必须是张量")
                        }
                    }
                    "layer_norm" => {
                        // x.layer_norm(gamma, beta, [eps])
                        if args.len() < 2 {
                            return err("layer_norm() 需要 gamma, beta, [eps] 参数");
                        }
                        let eps = args.get(2).and_then(|a| a.as_float()).unwrap_or(1e-5);
                        if let (Value::Tensor(gamma_rc), Value::Tensor(beta_rc)) = (&args[0], &args[1]) {
                            let x_shape = tensor.shape();
                            let ndim = x_shape.len();
                            if ndim == 0 || x_shape[ndim - 1] == 0 {
                                return Ok(Value::Tensor(Rc::new(RefCell::new(tensor.clone()))));
                            }
                            let axis_len = x_shape[ndim - 1];
                            // Validate gamma/beta shapes
                            let g_shape = gamma_rc.borrow().shape();
                            let b_shape = beta_rc.borrow().shape();
                            if g_shape.len() != 1 || g_shape[0] != axis_len {
                                return err(&format!(
                                    "layer_norm: gamma shape {:?} does not match last axis length {}",
                                    g_shape, axis_len
                                ));
                            }
                            if b_shape.len() != 1 || b_shape[0] != axis_len {
                                return err(&format!(
                                    "layer_norm: beta shape {:?} does not match last axis length {}",
                                    b_shape, axis_len
                                ));
                            }
                            let outer_len: usize = x_shape[..ndim - 1].iter().product();
                            let contiguous = tensor.data.as_standard_layout().to_owned();
                            let flat = match contiguous.as_slice() {
                                Some(s) => s.to_vec(),
                                None => tensor.data.iter().collect(),
                            };
                            let gamma_ref = gamma_rc.borrow();
                            let beta_ref = beta_rc.borrow();
                            let g_flat = gamma_ref.data.as_standard_layout().to_owned();
                            let b_flat = beta_ref.data.as_standard_layout().to_owned();
                            let g_slice = g_flat.as_slice().unwrap_or(&[]);
                            let b_slice = b_flat.as_slice().unwrap_or(&[]);
                            let mut result_data = Vec::with_capacity(flat.len());
                            let mut x_hat_data = Vec::with_capacity(flat.len());
                            let mut std_inv_data = Vec::with_capacity(outer_len);
                            for i in 0..outer_len {
                                let start = i * axis_len;
                                let slice = &flat[start..start + axis_len];
                                let mean: f64 = slice.iter().sum::<f64>() / axis_len as f64;
                                let var: f64 = slice.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / axis_len as f64;
                                let std_inv = 1.0 / (var + eps).sqrt();
                                std_inv_data.push(std_inv);
                                for j in 0..axis_len {
                                    let x_hat = (slice[j] - mean) * std_inv;
                                    x_hat_data.push(x_hat);
                                    let g = g_slice[j];
                                    let b = b_slice[j];
                                    result_data.push(g * x_hat + b);
                                }
                            }
                            let result = Rc::new(RefCell::new(Tensor::from_vec(result_data, x_shape.clone())));
                            if self.recording {
                                let x_hat = Rc::new(RefCell::new(Tensor::from_vec(x_hat_data, x_shape.clone())));
                                let std_inv_tensor = Rc::new(RefCell::new(Tensor::from_vec(std_inv_data, vec![outer_len])));
                                if let Some(ref mut tape) = self.tape {
                                    let x_id = t.borrow().tape_id
                                        .unwrap_or_else(|| tape.input(t.clone()));
                                    let node_id = tape.layernorm(
                                        x_id, t.clone(),
                                        gamma_rc.clone(), beta_rc.clone(),
                                        x_hat, std_inv_tensor, result.clone(),
                                    );
                                    result.borrow_mut().tape_id = Some(node_id);
                                }
                            }
                            Ok(Value::Tensor(result))
                        } else {
                            err("layer_norm: gamma 和 beta 必须是张量")
                        }
                    }
                    "dropout" => {
                        if args.is_empty() {
                            return err("dropout() 需要 1 个参数 (rate)");
                        }
                        let rate = args[0].as_float().unwrap_or(0.5);
                        use rand::Rng;
                        let mut rng = rand::thread_rng();
                        let scale = 1.0 / (1.0 - rate);
                        let mask_data = tensor.data.mapv(|_| {
                            if rng.r#gen::<f64>() < rate { 0.0 } else { scale }
                        });
                        let mask = Rc::new(RefCell::new(Tensor::from_data(mask_data)));
                        let result_tensor = Tensor::from_data(&tensor.data * &mask.borrow().data);
                        let result = Rc::new(RefCell::new(result_tensor));
                        if self.recording {
                            if let Some(ref mut tape) = self.tape {
                                let input_id = t.borrow().tape_id
                                    .unwrap_or_else(|| tape.input(t.clone()));
                                let _mask_id = tape.input(mask.clone());
                                let node_id = tape.dropout(input_id, t.clone(), mask.clone(), result.clone());
                                result.borrow_mut().tape_id = Some(node_id);
                            }
                        }
                        Ok(Value::Tensor(result))
                    }

                    _ => err(&format!("张量没有方法 '{}'", method)),
                }
            }
            Value::Struct { name: _, fields } => {
                // Try field-like access (e.g. .len on Vec field)
                for (fname, fval) in fields.borrow().iter() {
                    if fname == method { return Ok(fval.clone()); }
                }
                err(&format!("没有方法 '{}'", method))
            }
            // 标量数学方法（与 interpreter::eval_scalar_method 对齐）
            // 使 `f64.sqrt()` / `f64.abs()` 等方法调用在 VM 路径可用
            Value::Float(f) => match method {
                "sqrt" => Ok(Value::Float(f.sqrt())),
                "abs" => Ok(Value::Float(f.abs())),
                "exp" => Ok(Value::Float(f.exp())),
                "log" | "ln" => Ok(Value::Float(f.ln())),
                "sin" => Ok(Value::Float(f.sin())),
                "cos" => Ok(Value::Float(f.cos())),
                _ => err(&format!("Float 没有方法 '{}'", method)),
            },
            Value::Float32(f) => match method {
                "sqrt" => Ok(Value::Float32(f.sqrt())),
                "abs" => Ok(Value::Float32(f.abs())),
                "exp" => Ok(Value::Float32(f.exp())),
                "log" | "ln" => Ok(Value::Float32(f.ln())),
                "sin" => Ok(Value::Float32(f.sin())),
                "cos" => Ok(Value::Float32(f.cos())),
                _ => err(&format!("Float32 没有方法 '{}'", method)),
            },
            Value::Int(n, dtype) => match method {
                // ── 返回 Int ──
                "abs" => Ok(Value::Int(n.abs(), dtype)),
                "signum" => Ok(Value::Int(if n > 0 { 1 } else if n < 0 { -1 } else { 0 }, dtype)),
                "min" => {
                    if let Some(Value::Int(other, _)) = args.first() {
                        Ok(Value::Int(std::cmp::min(n, *other), dtype))
                    } else {
                        err("Int.min() 需要整数参数")
                    }
                }
                "max" => {
                    if let Some(Value::Int(other, _)) = args.first() {
                        Ok(Value::Int(std::cmp::max(n, *other), dtype))
                    } else {
                        err("Int.max() 需要整数参数")
                    }
                }
                "clamp" => {
                    let lo = match args.first() {
                        Some(Value::Int(v, _)) => *v,
                        _ => return err("Int.clamp() 需要整数参数 (lo)"),
                    };
                    let hi = match args.get(1) {
                        Some(Value::Int(v, _)) => *v,
                        _ => return err("Int.clamp() 需要整数参数 (hi)"),
                    };
                    Ok(Value::Int(n.clamp(lo, hi), dtype))
                }
                "bit_length" => Ok(Value::Int((64 - n.leading_zeros()) as i64, BaseType::I64)),
                "count_ones" => Ok(Value::Int(n.count_ones() as i64, BaseType::I64)),

                // ── 返回 Float ──
                "sqrt" => Ok(Value::Float((n as f64).sqrt())),
                "exp" => Ok(Value::Float((n as f64).exp())),
                "log" | "ln" => Ok(Value::Float((n as f64).ln())),
                "sin" => Ok(Value::Float((n as f64).sin())),
                "cos" => Ok(Value::Float((n as f64).cos())),
                "pow" => {
                    let exp = match args.first() {
                        Some(Value::Float(f)) => *f,
                        Some(Value::Int(i, _)) => *i as f64,
                        _ => return err("Int.pow() 需要数值参数"),
                    };
                    Ok(Value::Float((n as f64).powf(exp)))
                }
                "to_float" => Ok(Value::Float(n as f64)),

                // ── 返回 Str ──
                "to_string" => Ok(Value::String(n.to_string())),

                // ── 返回 Bool ──
                "is_even" => Ok(Value::Bool(n % 2 == 0)),
                "is_odd" => Ok(Value::Bool(n % 2 != 0)),

                _ => err(&format!("Int 没有方法 '{}'", method)),
            },
            // 问题29：智能指针容器方法（Box/Rc/Arc/Pin）。
            // 与解释器 eval_smart_ptr_method（interpreter/methods.rs）语义一致：
            // - deref/deref_mut：返回内部值（Value 克隆）
            // - clone：HeapBox/Pin 深拷贝内部值重新包装；SharedBox 用 Rc::clone 共享
            Value::HeapBox(v) => match method {
                "deref" | "deref_mut" => Ok((*v).clone()),
                "clone" => Ok(Value::HeapBox(Box::new((*v).clone()))),
                _ => err(&format!("Box 没有方法 '{}'", method)),
            },
            Value::SharedBox(rc) => match method {
                "deref" | "deref_mut" => Ok(rc.borrow().clone()),
                "clone" => Ok(Value::SharedBox(Rc::clone(&rc))),
                _ => err(&format!("Rc 没有方法 '{}'", method)),
            },
            Value::Pin(v) => match method {
                "deref" | "deref_mut" => Ok((*v).clone()),
                "clone" => Ok(Value::Pin(Box::new((*v).clone()))),
                _ => err(&format!("Pin 没有方法 '{}'", method)),
            },
            // M3.4：Weak 弱引用——不能直接解引用（必须先 weak_upgrade 取强引用），
            // 仅支持 clone（Weak::clone 共享同一弱句柄）。
            Value::Weak(w) => match method {
                "clone" => Ok(Value::Weak(w.clone())),
                _ => err(&format!("Weak 没有方法 '{}'", method)),
            },
            // M1.3：dyn Trait 动态分派——通过 `__dyn_{trait}_{type}_{method}`
            // 字节码函数调用（lowerer 在 `impl Trait for Type` 时注册这些函数）。
            // 与解释器（trait_impls 直接查表 + eval HIR body）语义一致。
            Value::Dyn { trait_name, type_name, value } => {
                let mangled = format!("__dyn_{}_{}_{}", trait_name, type_name, method);
                if !self.has_fn(&mangled) {
                    return err(&format!(
                        "dyn {} 值（具体类型 {}）没有方法 '{}'",
                        trait_name, type_name, method
                    ));
                }
                let mut dyn_args = Vec::with_capacity(args.len() + 1);
                dyn_args.push((*value).clone());
                dyn_args.extend(args.iter().cloned());
                self.call_with_args(&mangled, &dyn_args)
            }
            _ => err(&format!("没有方法 '{}'", method)),
        }
    }
}

