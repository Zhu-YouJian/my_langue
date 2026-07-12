//! 方法分派：String / Vec / Map / Range / Iterator / Tensor / Scalar 方法实现。
//!
//! 从 `interpreter.rs` 第 1780-3099 行迁移而来。包含：
//! - `eval_method_call` / `eval_native_method`：方法分派入口
//! - `eval_string_method` / `eval_vec_method` / `eval_map_method` /
//!   `eval_range_method` / `eval_iterator_method`：各类型方法
//! - `apply_closure`：闭包调用
//! - `find_methods_for_type` / `call_method_impl`：用户自定义方法查找与调用
//! - `eval_tensor_method` / `eval_scalar_method`：张量与标量方法

use std::collections::HashMap;
use crate::hir::types::BaseType;
use std::rc::Rc;
use std::cell::RefCell;
use ndarray::{IxDyn, ArrayD};
use crate::error::{TenthError, TenthResult};
use crate::hir::hir::*;
use crate::runtime::value::{Value, LazyIterator, IteratorTransform};
use crate::runtime::tensor::Tensor;
use crate::runtime::autodiff::TapeOp;

/// 问题2：将 Value 键转换为 HashMap 内部存储的 String 键。
/// 支持 String / Int / Bool / Float（浮点键按整数部分转字符串，仅推荐整数场景）。
/// 其他类型返回 TypeError。
fn map_key_to_string(v: &Value) -> TenthResult<String> {
    match v {
        Value::String(s) => Ok(s.clone()),
        Value::Int(n, _) => Ok(n.to_string()),
        Value::Bool(b) => Ok(b.to_string()),
        Value::Float32(f) => Ok(format!("{}", f)),
        Value::Float(f) => Ok(format!("{}", f)),
        _ => Err(TenthError::RuntimeError { line: None, col: None,
            message: format!("HashMap 键类型不支持: {:?}（仅支持 str/int/bool/float）", v),
        }),
    }
}

impl super::Interpreter {
    pub(super) fn eval_method_call(&mut self, recv: &Value, method: &str, args: &[Value]) -> TenthResult<Option<Value>> {
        // Auto-dereference Ref/MutRef/Shared to reach the inner value
        let recv = match recv {
            Value::Ref(rc) => {
                let inner = rc.borrow();
                return self.eval_method_call(&inner, method, args);
            }
            Value::MutRef(weak) => {
                if let Some(rc) = weak.upgrade() {
                    let inner = rc.borrow();
                    return self.eval_method_call(&inner, method, args);
                }
                return Err(TenthError::RuntimeError { line: None, col: None,
                    message: format!("悬垂 &mut 引用上的方法 '{}'", method),
                });
            }
            Value::Shared(rc) => {
                let inner = rc.borrow();
                return self.eval_method_call(&inner, method, args);
            }
            other => other,
        };
        match recv {
            Value::Struct { name, .. } => {
                if let Some(type_methods) = self.find_methods_for_type(name) {
                    if let Some(method_fn) = type_methods.get(method) {
                        return self.call_method_impl(recv, method_fn, args);
                    }
                }
                for (_trait_name, type_impls) in &self.trait_impls {
                    if let Some(methods) = type_impls.get(name) {
                        if let Some(method_fn) = methods.get(method) {
                            let fn_def = method_fn.clone();
                            return self.call_method_impl(recv, &fn_def, args);
                        }
                    }
                }
                self.eval_tensor_method(recv, method, args).map(Some)
            }
            Value::Enum { .. } => {
                self.eval_tensor_method(recv, method, args).map(Some)
            }
            Value::Tensor(_) => {
                self.eval_tensor_method(recv, method, args).map(Some)
            }
            Value::Float(f) => {
                self.eval_scalar_method(*f, method, args).map(Some)
            }
            Value::Int(i, _) => {
                let f = *i as f64;
                self.eval_scalar_method(f, method, args).map(Some)
            }
            Value::String(_) | Value::Vec(_) | Value::Map(_) | Value::Range { .. } | Value::Iterator(_) => {
                self.eval_native_method(recv, method, args)
            }
            _ => Err(TenthError::RuntimeError { line: None, col: None,
                message: format!("此类型不支持方法 '{}'", method),
            }),
        }
    }

    pub(super) fn eval_native_method(&mut self, recv: &Value, method: &str, args: &[Value]) -> TenthResult<Option<Value>> {
        match recv {
            Value::String(s) => self.eval_string_method(s, method, args),
            Value::Vec(items) => self.eval_vec_method(items, method, args),
            Value::Map(m) => self.eval_map_method(m, method, args),
            Value::Range { start, end, inclusive } => self.eval_range_method(*start, *end, *inclusive, method, args),
            Value::Iterator(iter) => self.eval_iterator_method(iter, method, args),
            _ => Err(TenthError::RuntimeError { line: None, col: None,
                message: format!("原生方法 '{}' 不可用", method),
            }),
        }
    }

    pub(super) fn eval_string_method(&self, s: &str, method: &str, args: &[Value]) -> TenthResult<Option<Value>> {
        match method {
            "len" => Ok(Some(Value::Int(s.chars().count() as i64, BaseType::I32))),
            "trim" => Ok(Some(Value::String(s.trim().to_string()))),
            "to_upper" => Ok(Some(Value::String(s.to_uppercase()))),
            "to_lower" => Ok(Some(Value::String(s.to_lowercase()))),
            "replace" => {
                if args.len() >= 2 {
                    if let (Value::String(from), Value::String(to)) = (&args[0], &args[1]) {
                        return Ok(Some(Value::String(s.replace(from.as_str(), to.as_str()))));
                    }
                }
                Err(TenthError::RuntimeError { line: None, col: None,
                    message: "replace() 需要 2 个字符串参数".into(),
                })
            }
            "split" => {
                if let Some(Value::String(delim)) = args.first() {
                    let parts: Vec<Value> = s.split(delim.as_str())
                        .map(|p| Value::String(p.to_string()))
                        .collect();
                    return Ok(Some(Value::Vec(Rc::new(RefCell::new(parts)))));
                }
                Err(TenthError::RuntimeError { line: None, col: None,
                    message: "split() 需要一个字符串分隔符".into(),
                })
            }
            "substring" => {
                if args.len() >= 2 {
                    let start = args[0].as_int().unwrap_or(0).max(0) as usize;
                    let len = args[1].as_int().unwrap_or(0).max(0) as usize;
                    let chars: Vec<char> = s.chars().collect();
                    let end = (start + len).min(chars.len());
                    let sub: String = chars[start..end].iter().collect();
                    return Ok(Some(Value::String(sub)));
                }
                Err(TenthError::RuntimeError { line: None, col: None,
                    message: "substring() 需要起始位置和长度".into(),
                })
            }
            "contains" => {
                if let Some(Value::String(sub)) = args.first() {
                    return Ok(Some(Value::Bool(s.contains(sub.as_str()))));
                }
                Err(TenthError::RuntimeError { line: None, col: None,
                    message: "contains() 需要一个字符串参数".into(),
                })
            }
            "find" => {
                if let Some(Value::String(sub)) = args.first() {
                    return Ok(Some(Value::Int(s.find(sub.as_str()).map(|i| i as i64).unwrap_or(-1), BaseType::I32)));
                }
                Err(TenthError::RuntimeError { line: None, col: None,
                    message: "find() 需要一个字符串参数".into(),
                })
            }
            "starts_with" => {
                if let Some(Value::String(prefix)) = args.first() {
                    return Ok(Some(Value::Bool(s.starts_with(prefix.as_str()))));
                }
                Err(TenthError::RuntimeError { line: None, col: None,
                    message: "starts_with() 需要一个字符串参数".into(),
                })
            }
            "ends_with" => {
                if let Some(Value::String(suffix)) = args.first() {
                    return Ok(Some(Value::Bool(s.ends_with(suffix.as_str()))));
                }
                Err(TenthError::RuntimeError { line: None, col: None,
                    message: "ends_with() 需要一个字符串参数".into(),
                })
            }
            "parse_int" => {
                return Ok(Some(Value::Int(s.trim().parse::<i64>().unwrap_or(0), BaseType::I32)));
            }
            "parse_float" => {
                return Ok(Some(Value::Float(s.trim().parse::<f64>().unwrap_or(0.0))));
            }
            "is_empty" => {
                return Ok(Some(Value::Bool(s.is_empty())));
            }
            "repeat" => {
                if let Some(arg) = args.first() {
                    let n = arg.as_int().unwrap_or(0).max(0) as usize;
                    return Ok(Some(Value::String(s.repeat(n))));
                }
                Err(TenthError::RuntimeError { line: None, col: None,
                    message: "repeat() 需要一个整数参数".into(),
                })
            }
            "chars" => {
                let chars: Vec<Value> = s.chars().map(|c| Value::String(c.to_string())).collect();
                Ok(Some(Value::Vec(Rc::new(RefCell::new(chars)))))
            }
            "bytes" => {
                let bytes: Vec<Value> = s.bytes().map(|b| Value::Int(b as i64, BaseType::I32)).collect();
                Ok(Some(Value::Vec(Rc::new(RefCell::new(bytes)))))
            }
            "trim_start" => Ok(Some(Value::String(s.trim_start().to_string()))),
            "trim_end" => Ok(Some(Value::String(s.trim_end().to_string()))),
            "strip_prefix" => {
                if let Some(Value::String(prefix)) = args.first() {
                    return Ok(Some(match s.strip_prefix(prefix.as_str()) {
                        Some(rest) => Value::String(rest.to_string()),
                        None => Value::String(s.to_string()),
                    }));
                }
                Err(TenthError::RuntimeError { line: None, col: None,
                    message: "strip_prefix() 需要一个字符串参数".into(),
                })
            }
            "strip_suffix" => {
                if let Some(Value::String(suffix)) = args.first() {
                    return Ok(Some(match s.strip_suffix(suffix.as_str()) {
                        Some(rest) => Value::String(rest.to_string()),
                        None => Value::String(s.to_string()),
                    }));
                }
                Err(TenthError::RuntimeError { line: None, col: None,
                    message: "strip_suffix() 需要一个字符串参数".into(),
                })
            }
            _ => Err(TenthError::RuntimeError { line: None, col: None,
                message: format!("String 没有方法 '{}'", method),
            }),
        }
    }

    pub(super) fn eval_vec_method(&mut self, items: &Rc<RefCell<Vec<Value>>>, method: &str, args: &[Value]) -> TenthResult<Option<Value>> {
        match method {
            "len" => Ok(Some(Value::Int(items.borrow().len() as i64, BaseType::I32))),
            "push" => {
                if args.len() != 1 {
                    return Err(TenthError::RuntimeError { line: None, col: None,
                        message: "push() 需要 1 个参数".into(),
                    });
                }
                // Wrap in Shared so elements can be mutated via indexed assignment.
                // If the value is already Shared, use it directly to avoid double-wrapping.
                let elem = match &args[0] {
                    Value::Shared(rc) => Value::Shared(rc.clone()),
                    other => Value::Shared(Rc::new(RefCell::new(other.clone()))),
                };
                items.borrow_mut().push(elem);
                Ok(Some(Value::Unit))
            }
            "get" => {
                if args.len() != 1 {
                    return Err(TenthError::RuntimeError { line: None, col: None,
                        message: "get() 需要 1 个参数".into(),
                    });
                }
                let idx = args[0].as_int().unwrap_or(0) as usize;
                let vec = items.borrow();
                match vec.get(idx) {
                    Some(v) => Ok(Some(v.clone())),
                    None => Err(TenthError::RuntimeError { line: None, col: None,
                        message: format!("Vec 索引 {} 越界", idx),
                    }),
                }
            }
            "pop" => {
                let mut vec = items.borrow_mut();
                match vec.pop() {
                    Some(v) => Ok(Some(v)),
                    None => Err(TenthError::RuntimeError { line: None, col: None,
                            message: "对空 Vec 调用 pop()".into(),
                    }),
                }
            }
            "set" => {
                if args.len() != 2 {
                    return Err(TenthError::RuntimeError { line: None, col: None,
                        message: "set() 需要 2 个参数 (索引, 值)".into(),
                    });
                }
                let idx = args[0].as_int().unwrap_or(0) as usize;
                let mut vec = items.borrow_mut();
                if idx < vec.len() {
                    vec[idx] = match &args[1] {
                        Value::Shared(rc) => Value::Shared(rc.clone()),
                        other => Value::Shared(Rc::new(RefCell::new(other.clone()))),
                    };
                    Ok(Some(Value::Unit))
                } else {
                    Err(TenthError::RuntimeError { line: None, col: None,
                        message: format!("Vec 索引 {} 越界", idx),
                    })
                }
            }
            "clear" => {
                items.borrow_mut().clear();
                Ok(Some(Value::Unit))
            }
            "contains" => {
                if args.len() != 1 {
                    return Err(TenthError::RuntimeError { line: None, col: None,
                        message: "contains() 需要 1 个参数".into(),
                    });
                }
                let vec = items.borrow();
                let found = vec.iter().any(|v| {
                    let unwrapped = match v {
                        Value::Shared(rc) => rc.borrow().clone(),
                        other => other.clone(),
                    };
                    self.values_eq(&unwrapped, &args[0])
                });
                Ok(Some(Value::Bool(found)))
            }
            "index_of" => {
                if args.len() != 1 {
                    return Err(TenthError::RuntimeError { line: None, col: None,
                        message: "index_of() 需要 1 个参数".into(),
                    });
                }
                let vec = items.borrow();
                for (i, v) in vec.iter().enumerate() {
                    let unwrapped = match v {
                        Value::Shared(rc) => rc.borrow().clone(),
                        other => other.clone(),
                    };
                    if self.values_eq(&unwrapped, &args[0]) {
                        return Ok(Some(Value::Int(i as i64, BaseType::I32)));
                    }
                }
                Ok(Some(Value::Int(-1, BaseType::I32)))
            }
            "remove" => {
                if args.len() != 1 {
                    return Err(TenthError::RuntimeError { line: None, col: None,
                        message: "remove() 需要 1 个参数 (索引)".into(),
                    });
                }
                let idx = args[0].as_int().ok_or_else(|| TenthError::RuntimeError { line: None, col: None,
                    message: "remove() 索引必须是整数".into(),
                })? as usize;
                let mut vec = items.borrow_mut();
                if idx < vec.len() {
                    Ok(Some(vec.remove(idx)))
                } else {
                    Err(TenthError::RuntimeError { line: None, col: None,
                        message: format!("Vec remove 索引 {} 越界", idx),
                    })
                }
            }
            "join" => {
                if args.len() != 1 {
                    return Err(TenthError::RuntimeError { line: None, col: None,
                        message: "join() 需要 1 个参数 (分隔符)".into(),
                    });
                }
                let delim = match &args[0] {
                    Value::String(s) => s.clone(),
                    _ => return Err(TenthError::RuntimeError { line: None, col: None,
                            message: "join() 分隔符必须是字符串".into(),
                    }),
                };
                let vec = items.borrow();
                let parts: Vec<String> = vec.iter().map(|v| {
                    match v {
                        Value::Shared(rc) => format!("{}", rc.borrow()),
                        other => format!("{}", other),
                    }
                }).collect();
                Ok(Some(Value::String(parts.join(&delim))))
            }
            "is_empty" => {
                Ok(Some(Value::Bool(items.borrow().is_empty())))
            }
            "reverse" => {
                let vec = items.borrow();
                let reversed: Vec<Value> = vec.iter().rev().cloned().collect();
                Ok(Some(Value::Vec(Rc::new(RefCell::new(reversed)))))
            }
            "slice" => {
                if args.len() != 2 {
                    return Err(TenthError::RuntimeError { line: None, col: None,
                        message: "slice() 需要 2 个参数 (起始, 结束)".into(),
                    });
                }
                let start = args[0].as_int().unwrap_or(0).max(0) as usize;
                let end = args[1].as_int().unwrap_or(0).max(0) as usize;
                let vec = items.borrow();
                let end = end.min(vec.len());
                if start > end {
                    return Ok(Some(Value::Vec(Rc::new(RefCell::new(Vec::new())))));
                }
                let sliced: Vec<Value> = vec[start..end].to_vec();
                Ok(Some(Value::Vec(Rc::new(RefCell::new(sliced)))))
            }
            "extend" => {
                if args.len() != 1 {
                    return Err(TenthError::RuntimeError { line: None, col: None,
                        message: "extend() 需要 1 个参数 (Vec)".into(),
                    });
                }
                if let Value::Vec(other) = &args[0] {
                    let other_vals = other.borrow().clone();
                    let mut vec = items.borrow_mut();
                    for v in other_vals {
                        let elem = match v {
                            Value::Shared(rc) => Value::Shared(rc),
                            other => Value::Shared(Rc::new(RefCell::new(other))),
                        };
                        vec.push(elem);
                    }
                    return Ok(Some(Value::Unit));
                }
                Err(TenthError::RuntimeError { line: None, col: None,
                    message: "extend() 参数必须是 Vec".into(),
                })
            }
            "sort" => {
                let mut vec = items.borrow_mut();
                vec.sort_by(|a, b| {
                    let av = match a { Value::Shared(rc) => rc.borrow().clone(), o => o.clone() };
                    let bv = match b { Value::Shared(rc) => rc.borrow().clone(), o => o.clone() };
                    match (&av, &bv) {
                        (Value::Int(x, _), Value::Int(y, _)) => x.cmp(y),
                        (Value::Float(x), Value::Float(y)) => x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal),
                        (Value::String(x), Value::String(y)) => x.cmp(y),
                        _ => std::cmp::Ordering::Equal,
                    }
                });
                Ok(Some(Value::Unit))
            }
            "dedup" => {
                let mut vec = items.borrow_mut();
                vec.dedup_by(|a, b| {
                    let av = match a { Value::Shared(rc) => rc.borrow().clone(), o => o.clone() };
                    let bv = match b { Value::Shared(rc) => rc.borrow().clone(), o => o.clone() };
                    self.values_eq(&av, &bv)
                });
                Ok(Some(Value::Unit))
            }
            "first" => {
                let vec = items.borrow();
                match vec.first() {
                    Some(v) => Ok(Some(v.clone())),
                    None => Ok(None),
                }
            }
            "last" => {
                let vec = items.borrow();
                match vec.last() {
                    Some(v) => Ok(Some(v.clone())),
                    None => Ok(None),
                }
            }
            "flatten" => {
                let vec = items.borrow();
                let mut result = Vec::new();
                for v in vec.iter() {
                    let unwrapped = match v {
                        Value::Shared(rc) => rc.borrow().clone(),
                        other => other.clone(),
                    };
                    if let Value::Vec(inner) = unwrapped {
                        for item in inner.borrow().iter() {
                            let elem = match item {
                                Value::Shared(rc) => Value::Shared(rc.clone()),
                                o => Value::Shared(Rc::new(RefCell::new(o.clone()))),
                            };
                            result.push(elem);
                        }
                    } else {
                        let elem = match v {
                            Value::Shared(rc) => Value::Shared(rc.clone()),
                            o => Value::Shared(Rc::new(RefCell::new(o.clone()))),
                        };
                        result.push(elem);
                    }
                }
                Ok(Some(Value::Vec(Rc::new(RefCell::new(result)))))
            }
            "chunks" => {
                if args.len() != 1 {
                    return Err(TenthError::RuntimeError { line: None, col: None,
                        message: "chunks() 需要 1 个参数 (大小)".into(),
                    });
                }
                let size = args[0].as_int().unwrap_or(1).max(1) as usize;
                let vec = items.borrow();
                let mut result = Vec::new();
                for chunk in vec.chunks(size) {
                    let c: Vec<Value> = chunk.to_vec();
                    result.push(Value::Vec(Rc::new(RefCell::new(c))));
                }
                Ok(Some(Value::Vec(Rc::new(RefCell::new(result)))))
            }
            // Lazy iterator methods
            "iter" => {
                Ok(Some(Value::Iterator(LazyIterator::from_vec(items))))
            }
            "map" => {
                if args.len() != 1 {
                    return Err(TenthError::RuntimeError { line: None, col: None,
                        message: "map() 需要 1 个参数 (闭包)".into(),
                    });
                }
                let iter = LazyIterator::from_vec(items);
                let iter = iter.with_transform(IteratorTransform::Map { closure: args[0].clone() });
                Ok(Some(Value::Iterator(iter)))
            }
            "filter" => {
                if args.len() != 1 {
                    return Err(TenthError::RuntimeError { line: None, col: None,
                        message: "filter() 需要 1 个参数 (闭包)".into(),
                    });
                }
                let iter = LazyIterator::from_vec(items);
                let iter = iter.with_transform(IteratorTransform::Filter { closure: args[0].clone() });
                Ok(Some(Value::Iterator(iter)))
            }
            "collect" => {
                // Vec.collect() is a no-op — already a Vec
                Ok(Some(Value::Vec(items.clone())))
            }
            _ => Err(TenthError::RuntimeError { line: None, col: None,
                message: format!("Vec 没有方法 '{}'", method),
            }),
        }
    }

    pub(super) fn eval_range_method(&self, start: i64, end: i64, inclusive: bool, method: &str, _args: &[Value]) -> TenthResult<Option<Value>> {
        match method {
            "iter" => Ok(Some(Value::Iterator(LazyIterator::from_range(start, end, inclusive)))),
            "len" => {
                let len = if inclusive { (end - start + 1).max(0) as i64 } else { (end - start).max(0) as i64 };
                Ok(Some(Value::Int(len, BaseType::I32)))
            }
            _ => Err(TenthError::RuntimeError { line: None, col: None,
                message: format!("Range 没有方法 '{}'", method),
            }),
        }
    }

    pub(super) fn eval_iterator_method(&mut self, iter: &LazyIterator, method: &str, args: &[Value]) -> TenthResult<Option<Value>> {
        match method {
            "map" => {
                if args.len() != 1 {
                    return Err(TenthError::RuntimeError { line: None, col: None,
                        message: "map() 需要 1 个参数 (闭包)".into(),
                    });
                }
                let new_iter = iter.with_transform(IteratorTransform::Map { closure: args[0].clone() });
                Ok(Some(Value::Iterator(new_iter)))
            }
            "filter" => {
                if args.len() != 1 {
                    return Err(TenthError::RuntimeError { line: None, col: None,
                        message: "filter() 需要 1 个参数 (闭包)".into(),
                    });
                }
                let new_iter = iter.with_transform(IteratorTransform::Filter { closure: args[0].clone() });
                Ok(Some(Value::Iterator(new_iter)))
            }
            "take" => {
                if args.len() != 1 {
                    return Err(TenthError::RuntimeError { line: None, col: None,
                        message: "take() 需要 1 个参数 (n)".into(),
                    });
                }
                let n = args[0].as_int().unwrap_or(0).max(0) as usize;
                let new_iter = iter.with_transform(IteratorTransform::Take { n });
                Ok(Some(Value::Iterator(new_iter)))
            }
            "skip" => {
                if args.len() != 1 {
                    return Err(TenthError::RuntimeError { line: None, col: None,
                        message: "skip() 需要 1 个参数 (n)".into(),
                    });
                }
                let n = args[0].as_int().unwrap_or(0).max(0) as usize;
                let new_iter = iter.with_transform(IteratorTransform::Skip { n });
                Ok(Some(Value::Iterator(new_iter)))
            }
            "collect" => {
                // Materialize the iterator: apply all transforms to each element
                let source = iter.source.borrow();
                let transforms = iter.transforms.borrow();
                let mut result = Vec::new();
                let mut seen_count: usize = 0;
                for item in source.iter() {
                    let mut val = match item {
                        Value::Shared(rc) => rc.borrow().clone(),
                        other => other.clone(),
                    };
                    let mut included = true;
                    for transform in transforms.iter() {
                        match transform {
                            IteratorTransform::Map { closure } => {
                                val = self.apply_closure(closure, &[val])?;
                            }
                            IteratorTransform::Filter { closure } => {
                                let pred_result = self.apply_closure(closure, &[val.clone()])?;
                                match pred_result {
                                    Value::Bool(true) => {},
                                    Value::Bool(false) => { included = false; break; }
                                    _ => { included = false; break; }
                                }
                            }
                            IteratorTransform::Take { n } => {
                                if result.len() >= *n {
                                    included = false;
                                    break;
                                }
                            }
                            IteratorTransform::Skip { n } => {
                                if seen_count < *n {
                                    included = false;
                                    break;
                                }
                            }
                        }
                    }
                    seen_count += 1;
                    if included {
                        result.push(Value::Shared(Rc::new(RefCell::new(val))));
                    }
                }
                Ok(Some(Value::Vec(Rc::new(RefCell::new(result)))))
            }
            "len" => {
                // Materialize to count — not ideal but correct
                let collected = self.eval_iterator_method(iter, "collect", &[])?;
                match collected {
                    Some(Value::Vec(v)) => Ok(Some(Value::Int(v.borrow().len() as i64, BaseType::I32))),
                    _ => Ok(Some(Value::Int(0, BaseType::I32))),
                }
            }
            _ => Err(TenthError::RuntimeError { line: None, col: None,
                message: format!("Iterator 没有方法 '{}'", method),
            }),
        }
    }

    /// Apply a closure value to arguments, returning the result.
    pub(super) fn apply_closure(&mut self, closure: &Value, args: &[Value]) -> TenthResult<Value> {
        match closure {
            Value::Closure { params, body, captures } => {
                // AUDIT-11.4.3: push_scope 后逐个 insert captures
                self.push_scope();
                for (cap_name, cap_val) in captures.clone() {
                    self.insert_var(cap_name, cap_val);
                }
                for (i, (name, _)) in params.iter().enumerate() {
                    let val = args.get(i).cloned().unwrap_or(Value::Unit);
                    self.insert_var(name.clone(), val);
                }
                let result = self.eval_expr(body);
                self.pop_scope();
                match result {
                    Ok(Some(v)) => Ok(v),
                    Ok(None) => Ok(Value::Unit),
                    Err(TenthError::ReturnValue(v)) => Ok(v),
                    Err(e) => Err(e),
                }
            }
            Value::FnRef { name, .. } => {
                // Call named function
                let span = crate::lexer::token::Span { line: 0, col: 0 };
                let result = self.call_named_fn(name, args, &span)?;
                Ok(result.unwrap_or(Value::Unit))
            }
            _ => Err(TenthError::RuntimeError { line: None, col: None,
                message: format!("期望可调用值，得到 {:?}", closure),
            }),
        }
    }

    pub(super) fn eval_map_method(&mut self, m: &Rc<RefCell<HashMap<String, Value>>>, method: &str, args: &[Value]) -> TenthResult<Option<Value>> {
        match method {
            "len" => Ok(Some(Value::Int(m.borrow().len() as i64, BaseType::I32))),
            "get" => {
                if args.len() != 1 {
                    return Err(TenthError::RuntimeError { line: None, col: None,
                        message: "get() 需要 1 个参数".into(),
                    });
                }
                // 问题2：支持字符串、整数、布尔键（内部统一转为 String 存储）
                let key = map_key_to_string(&args[0])?;
                Ok(m.borrow().get(&key).cloned().or(Some(Value::Unit)))
            }
            "insert" => {
                if args.len() != 2 {
                    return Err(TenthError::RuntimeError { line: None, col: None,
                        message: "insert() 需要 2 个参数".into(),
                    });
                }
                let key = map_key_to_string(&args[0])?;
                m.borrow_mut().insert(key, args[1].clone());
                Ok(Some(Value::Unit))
            }
            "contains_key" => {
                if args.len() != 1 {
                    return Err(TenthError::RuntimeError { line: None, col: None,
                        message: "contains_key() 需要 1 个参数".into(),
                    });
                }
                let key = map_key_to_string(&args[0])?;
                Ok(Some(Value::Bool(m.borrow().contains_key(&key))))
            }
            "remove" => {
                if args.len() != 1 {
                    return Err(TenthError::RuntimeError { line: None, col: None,
                        message: "remove() 需要 1 个参数".into(),
                    });
                }
                let key = map_key_to_string(&args[0])?;
                Ok(m.borrow_mut().remove(&key))
            }
            "keys" => {
                let map = m.borrow();
                let keys: Vec<Value> = map.keys().map(|k| Value::String(k.clone())).collect();
                Ok(Some(Value::Vec(Rc::new(RefCell::new(keys)))))
            }
            "values" => {
                let map = m.borrow();
                let vals: Vec<Value> = map.values().cloned().collect();
                Ok(Some(Value::Vec(Rc::new(RefCell::new(vals)))))
            }
            "is_empty" => {
                Ok(Some(Value::Bool(m.borrow().is_empty())))
            }
            "entries" => {
                let map = m.borrow();
                let entries: Vec<Value> = map.iter().map(|(k, v)| {
                    Value::Vec(Rc::new(RefCell::new(vec![
                        Value::String(k.clone()),
                        v.clone(),
                    ])))
                }).collect();
                Ok(Some(Value::Vec(Rc::new(RefCell::new(entries)))))
            }
            "merge" => {
                if args.len() != 1 {
                    return Err(TenthError::RuntimeError { line: None, col: None,
                        message: "merge() 需要 1 个参数 (HashMap)".into(),
                    });
                }
                if let Value::Map(other) = &args[0] {
                    let other_map = other.borrow().clone();
                    let mut map = m.borrow_mut();
                    for (k, v) in other_map {
                        map.insert(k, v);
                    }
                    return Ok(Some(Value::Unit));
                }
                Err(TenthError::RuntimeError { line: None, col: None,
                    message: "merge() 参数必须是 HashMap".into(),
                })
            }
            _ => Err(TenthError::RuntimeError { line: None, col: None,
                message: format!("HashMap 没有方法 '{}'", method),
            }),
        }
    }

    pub(super) fn find_methods_for_type(&self, type_name: &str) -> Option<HashMap<String, HirFnDef>> {
        for (impl_type, methods) in &self.methods {
            if impl_type == type_name {
                return Some(methods.clone());
            }
        }
        for module in self.modules.values() {
            for func in &module.functions {
                if func.name == type_name {
                    return None;
                }
            }
        }
        None
    }

    pub(super) fn call_method_impl(&mut self, receiver: &Value, method_fn: &HirFnDef, args: &[Value]) -> TenthResult<Option<Value>> {
        self.push_scope();
        self.insert_var("self".to_string(), receiver.clone());

        for ((pname, _), arg) in method_fn.params.iter().skip(1).zip(args.iter()) {
            self.insert_var(pname.clone(), arg.clone());
        }

        let result = self.eval_expr(&method_fn.body);

        self.pop_scope();

        result
    }

    pub(super) fn eval_tensor_method(&mut self, recv: &Value, method: &str, args: &[Value]) -> TenthResult<Value> {
        match recv {
            Value::Tensor(t) => {
                let tensor = t.borrow();
                match method {
                    "sum" => {
                        if args.is_empty() {
                            if self.recording {
                                // Return a 1-element tensor so it can be recorded
                                let scalar = tensor.sum();
                                let result = Rc::new(RefCell::new(
                                    Tensor::from_vec(vec![scalar], vec![1])
                                ));
                                self.record_unary(TapeOp::Sum, t, &result);
                                Ok(Value::Tensor(result))
                            } else {
                                Ok(Value::Float(tensor.sum()))
                            }
                        } else {
                            let axis = args[0].as_int().unwrap_or(0) as usize;
                            match tensor.sum_axis(axis) {
                                Ok(result) => Ok(Value::Tensor(Rc::new(RefCell::new(result)))),
                                Err(msg) => Err(TenthError::RuntimeError { line: None, col: None, message: msg }),
                            }
                        }
                    }
                    "mean" => {
                        if self.recording {
                            let scalar = tensor.mean();
                            let result = Rc::new(RefCell::new(
                                Tensor::from_vec(vec![scalar], vec![1])
                            ));
                            self.record_unary(TapeOp::Mean, t, &result);
                            Ok(Value::Tensor(result))
                        } else {
                            Ok(Value::Float(tensor.mean()))
                        }
                    }
                    "abs" => {
                        let result_tensor = tensor.abs();
                        let result = Rc::new(RefCell::new(result_tensor));
                        if self.recording {
                            self.record_unary(TapeOp::Abs, t, &result);
                        }
                        Ok(Value::Tensor(result))
                    }
                    "sqrt" => {
                        let result_tensor = tensor.sqrt();
                        let result = Rc::new(RefCell::new(result_tensor));
                        Ok(Value::Tensor(result))
                    }
                    "exp" => {
                        let result_tensor = tensor.exp();
                        let result = Rc::new(RefCell::new(result_tensor));
                        if self.recording {
                            self.record_unary(TapeOp::Exp, t, &result);
                        }
                        Ok(Value::Tensor(result))
                    }
                    "log" => {
                        let result_tensor = tensor.log();
                        let result = Rc::new(RefCell::new(result_tensor));
                        if self.recording {
                            self.record_unary(TapeOp::Log, t, &result);
                        }
                        Ok(Value::Tensor(result))
                    }
                    "relu" => {
                        let result_tensor = tensor.relu();
                        let result = Rc::new(RefCell::new(result_tensor));
                        if self.recording {
                            self.record_unary(TapeOp::ReLU, t, &result);
                        }
                        Ok(Value::Tensor(result))
                    }
                    "sigmoid" => {
                        let result_tensor = tensor.sigmoid();
                        let result = Rc::new(RefCell::new(result_tensor));
                        if self.recording {
                            self.record_unary(TapeOp::Sigmoid, t, &result);
                        }
                        Ok(Value::Tensor(result))
                    }
                    "tanh" => {
                        let result_tensor = tensor.tanh();
                        let result = Rc::new(RefCell::new(result_tensor));
                        Ok(Value::Tensor(result))
                    }
                    "matmul" => {
                        if args.len() != 1 {
                            return Err(TenthError::RuntimeError { line: None, col: None,
                                message: "matmul() 需要 1 个参数".into(),
                            });
                        }
                        if let Value::Tensor(other) = &args[0] {
                            let result_tensor = tensor.matmul(&other.borrow())
                                .map_err(|msg| TenthError::RuntimeError { line: None, col: None, message: msg })?;
                            let result = Rc::new(RefCell::new(result_tensor));
                            if self.recording {
                                self.record_binary(TapeOp::MatMul, t, other, &result);
                            }
                            Ok(Value::Tensor(result))
                        } else {
                            Err(TenthError::RuntimeError { line: None, col: None,
                                message: "matmul() 参数必须是张量".into(),
                            })
                        }
                    }
                    "bmm" => {
                        // batched matmul: (B, M, K) @ (B, K, N) -> (B, M, N)
                        if args.len() != 1 {
                            return Err(TenthError::RuntimeError { line: None, col: None,
                                message: "bmm() 需要 1 个参数".into(),
                            });
                        }
                        if let Value::Tensor(other) = &args[0] {
                            let result_tensor = tensor.bmm(&other.borrow())
                                .map_err(|msg| TenthError::RuntimeError { line: None, col: None, message: msg })?;
                            let result = Rc::new(RefCell::new(result_tensor));
                            if self.recording {
                                self.record_binary(TapeOp::BatchedMatMul, t, other, &result);
                            }
                            Ok(Value::Tensor(result))
                        } else {
                            Err(TenthError::RuntimeError { line: None, col: None,
                                message: "bmm() 参数必须是张量".into(),
                            })
                        }
                    }
                    "transpose" => {
                        if !args.is_empty() {
                            return Err(TenthError::RuntimeError { line: None, col: None,
                                message: "transpose() 不需要参数".into(),
                            });
                        }
                        let result_tensor = tensor.transpose().ok_or_else(|| {
                            TenthError::RuntimeError { line: None, col: None,
                                message: "转置至少需要 2 个维度".into(),
                            }
                        })?;
                        let result = Rc::new(RefCell::new(result_tensor));
                        if self.recording {
                            self.record_unary(TapeOp::Transpose, t, &result);
                        }
                        Ok(Value::Tensor(result))
                    }
                    "reshape" | "view" => {
                        let shape: Vec<usize> = args.iter()
                            .map(|a| a.as_int().unwrap_or(1) as usize)
                            .collect();
                        let result_tensor = tensor.reshape(&shape).ok_or_else(|| {
                            TenthError::RuntimeError { line: None, col: None,
                                message: format!("无法重塑形状为 {:?}", shape),
                            }
                        })?;
                        let result = Rc::new(RefCell::new(result_tensor));
                        if self.recording {
                            self.record_unary(TapeOp::Reshape, t, &result);
                        }
                        Ok(Value::Tensor(result))
                    }
                    "flatten" => {
                        let result = tensor.flatten();
                        Ok(Value::Tensor(Rc::new(RefCell::new(result))))
                    }
                    "conv2d" => {
                        // x.conv2d(w, kernel_h, kernel_w, stride, pad)
                        if args.len() < 5 {
                            return Err(TenthError::RuntimeError { line: None, col: None,
                                message: "conv2d() 需要 5 个参数: w, kH, kW, stride, pad".into(),
                            });
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
                                return Err(TenthError::RuntimeError { line: None, col: None,
                                    message: format!(
                                        "conv2d: 权重必须是 4D (C_out, C_in, kH, kW)，得到 {:?}D",
                                        w_shape.len()
                                    ),
                                });
                            }
                            if w_shape[2] != k_h || w_shape[3] != k_w {
                                return Err(TenthError::RuntimeError { line: None, col: None,
                                    message: format!(
                                        "conv2d: 权重 kernel 尺寸 {:?} 与参数 kH={}, kW={} 不匹配",
                                        &w_shape[2..4], k_h, k_w
                                    ),
                                });
                            }
                            // im2col: (N,C,H,W) → (N*H_out*W_out, C*kH*kW)
                            let (cols, h_out, w_out) = tensor.im2col(k_h, k_w, stride, pad)
                                .ok_or_else(|| TenthError::RuntimeError { line: None, col: None,
                                    message: "im2col 失败 (输入必须是 4D)".into(),
                                })?;
                            // Reshape weight: (C_out, C_in, kH, kW) → (C_out, C_in*kH*kW)
                            let c_out = w_shape[0];
                            // matmul: cols @ w_flat^T → (N*H_out*W_out, C_out)
                            let w_flat = w_data.reshape(&[c_out, w_shape[1] * w_shape[2] * w_shape[3]])
                                .ok_or_else(|| TenthError::RuntimeError { line: None, col: None,
                                    message: "权重重塑失败".into(),
                                })?;
                            let output_2d = cols.matmul(&w_flat.transpose()
                                .ok_or_else(|| TenthError::RuntimeError { line: None, col: None,
                                    message: "权重转置失败".into(),
                                })?).map_err(|msg| TenthError::RuntimeError { line: None, col: None, message: msg })?;
                            // Reshape output to (N, C_out, H_out, W_out)
                            let n = tensor.shape()[0];
                            let result = output_2d.reshape(&[n, c_out, h_out, w_out])
                                .ok_or_else(|| TenthError::RuntimeError { line: None, col: None,
                                    message: "输出重塑失败".into(),
                                })?;
                            let result_rc = Rc::new(RefCell::new(result));
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
                                        cols_rc, result_rc.clone(),
                                    );
                                    result_rc.borrow_mut().tape_id = Some(node_id);
                                }
                            }
                            return Ok(Value::Tensor(result_rc));
                        }
                        return Err(TenthError::RuntimeError { line: None, col: None,
                            message: "conv2d: 权重必须是张量".into(),
                        });
                    }
                    "batchnorm" => {
                        // x.batchnorm(gamma, beta, eps)
                        // gamma, beta: 1D tensors of shape (C,)
                        // x: (N, C, H, W) — computes mean/var over N, H, W per channel
                        if args.len() < 3 {
                            return Err(TenthError::RuntimeError { line: None, col: None,
                                message: "batchnorm() 需要 gamma, beta, eps".into(),
                            });
                        }
                        let eps = args[2].as_float().unwrap_or(1e-5);
                        if let (Value::Tensor(gamma_rc), Value::Tensor(beta_rc)) = (&args[0], &args[1]) {
                            let x_shape = tensor.shape();
                            if x_shape.len() < 2 {
                                return Err(TenthError::RuntimeError { line: None, col: None,
                                    message: "batchnorm 需要至少 2D 输入".into(),
                                });
                            }
                            let c = x_shape[1]; // channel dim
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
                                // Mean over N, H, W for this channel
                                let mut sum = 0.0;
                                let mut count = 0;
                                for ni in 0..n {
                                    for si in 0..spatial {
                                        let idx = ((ni * c + ci) * spatial) + si;
                                        if idx < x_slice.len() {
                                            sum += x_slice[idx];
                                            count += 1;
                                        }
                                    }
                                }
                                let mean = if count > 0 { sum / count as f64 } else { 0.0 };

                                // Variance
                                let mut var_sum = 0.0;
                                for ni in 0..n {
                                    for si in 0..spatial {
                                        let idx = ((ni * c + ci) * spatial) + si;
                                        if idx < x_slice.len() {
                                            let diff = x_slice[idx] - mean;
                                            var_sum += diff * diff;
                                        }
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
                                            let x_norm = (x_slice[idx] - mean) * std_inv;
                                            x_hat_data.push(x_norm);
                                            result_data.push(g * x_norm + b);
                                        }
                                    }
                                }
                            }

                            let result = Rc::new(RefCell::new(Tensor::from_vec(result_data, x_shape.clone())));
                            if self.recording {
                                let x_hat = Rc::new(RefCell::new(Tensor::from_vec(x_hat_data, x_shape)));
                                let std_inv_tensor = Rc::new(RefCell::new(
                                    Tensor::from_vec(std_inv_data, vec![c])
                                ));
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
                            return Ok(Value::Tensor(result));
                        }
                        return Err(TenthError::RuntimeError { line: None, col: None,
                            message: "batchnorm: gamma 和 beta 必须是张量".into(),
                        });
                    }
                    "dropout" => {
                        if args.is_empty() {
                            return Err(TenthError::RuntimeError { line: None, col: None,
                                message: "dropout() 需要 1 个参数 (比率)".into(),
                            });
                        }
                        let rate = args[0].as_float().unwrap_or(0.5);
                        // Generate inverted dropout mask
                        use rand::Rng;
                        let mut rng = rand::thread_rng();
                        let scale = 1.0 / (1.0 - rate);
                        // Phase 5.4：按 tensor dtype 分支保持精度（f32 tensor → f32 mask + f32 result）
                        let (mask_tensor, result_tensor) = if tensor.is_f32() {
                            let scale_f32 = scale as f32;
                            let rate_f32 = rate as f32;
                            let mask_arr = tensor.data.as_f32().expect("is_f32 checked").mapv(|_| {
                                if rng.r#gen::<f32>() < rate_f32 { 0.0f32 } else { scale_f32 }
                            });
                            let mask_t = Tensor::from_data_f32(mask_arr);
                            let t_arr = tensor.data.as_f32().expect("is_f32 checked");
                            let res = Tensor::from_data_f32(t_arr * mask_t.data.as_f32().expect("f32 mask"));
                            (mask_t, res)
                        } else {
                            let mask_data = tensor.data.mapv(|_| {
                                if rng.r#gen::<f64>() < rate { 0.0 } else { scale }
                            });
                            let mask_t = Tensor::from_data(mask_data);
                            let res = Tensor::from_data(&tensor.data * &mask_t.data);
                            (mask_t, res)
                        };
                        let mask = Rc::new(RefCell::new(mask_tensor));
                        let result = Rc::new(RefCell::new(result_tensor));
                        if self.recording {
                            let node_id = if let Some(ref mut tape) = self.tape {
                                let input_id = t.borrow().tape_id
                                    .unwrap_or_else(|| tape.input(t.clone()));
                                let _mask_id = tape.input(mask.clone());
                                tape.dropout(input_id, t.clone(), mask.clone(), result.clone())
                            } else {
                                0
                            };
                            result.borrow_mut().tape_id = Some(node_id);
                        }
                        Ok(Value::Tensor(result))
                    }
                    "layer_norm" => {
                        // x.layer_norm(gamma, beta, eps)
                        if args.len() < 2 {
                            return Err(TenthError::RuntimeError { line: None, col: None,
                                message: "layer_norm() 需要 gamma, beta, [eps]".into(),
                            });
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
                                return Err(TenthError::RuntimeError { line: None, col: None,
                                    message: format!(
                                        "layer_norm: gamma shape {:?} does not match last axis length {}",
                                        g_shape, axis_len
                                    ),
                                });
                            }
                            if b_shape.len() != 1 || b_shape[0] != axis_len {
                                return Err(TenthError::RuntimeError { line: None, col: None,
                                    message: format!(
                                        "layer_norm: beta shape {:?} does not match last axis length {}",
                                        b_shape, axis_len
                                    ),
                                });
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
                                let x_hat = Rc::new(RefCell::new(Tensor::from_vec(x_hat_data, x_shape)));
                                let std_inv_tensor = Rc::new(RefCell::new(
                                    Tensor::from_vec(std_inv_data, vec![outer_len])
                                ));
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
                            return Ok(Value::Tensor(result));
                        }
                        return Err(TenthError::RuntimeError { line: None, col: None,
                            message: "layer_norm: gamma 和 beta 必须是张量".into(),
                        });
                    }
                    "gelu" => {
                        let result_tensor = tensor.gelu();
                        let result = Rc::new(RefCell::new(result_tensor));
                        if self.recording {
                            self.record_unary(TapeOp::Gelu, t, &result);
                        }
                        Ok(Value::Tensor(result))
                    }
                    "cat" => {
                        // x.cat(other, dim)
                        if args.is_empty() {
                            return Err(TenthError::RuntimeError { line: None, col: None,
                                message: "cat() 至少需要 1 个参数 (other, [dim])".into(),
                            });
                        }
                        let dim = args.get(1).and_then(|a| a.as_int()).unwrap_or(0) as usize;
                        if let Value::Tensor(other) = &args[0] {
                            let result_tensor = tensor.cat(&other.borrow(), dim)
                                .map_err(|msg| TenthError::RuntimeError { line: None, col: None, message: msg })?;
                            Ok(Value::Tensor(Rc::new(RefCell::new(result_tensor))))
                        } else {
                            Err(TenthError::RuntimeError { line: None, col: None,
                                message: "cat() 第一个参数必须是张量".into(),
                            })
                        }
                    }
                    "masked_fill" => {
                        // x.masked_fill(mask, value)
                        if args.len() < 2 {
                            return Err(TenthError::RuntimeError { line: None, col: None,
                                message: "masked_fill() 需要 mask 和 value".into(),
                            });
                        }
                        let value = args[1].as_float().unwrap_or(0.0);
                        if let Value::Tensor(mask_rc) = &args[0] {
                            let result_tensor = tensor.masked_fill(&mask_rc.borrow(), value)
                                .map_err(|msg| TenthError::RuntimeError { line: None, col: None, message: msg })?;
                            let result = Rc::new(RefCell::new(result_tensor));
                            if self.recording {
                                let node_id = if let Some(ref mut tape) = self.tape {
                                    let input_id = t.borrow().tape_id;
                                    tape.masked_fill(input_id, t.clone(), mask_rc.clone(), result.clone())
                                } else {
                                    0
                                };
                                result.borrow_mut().tape_id = Some(node_id);
                            }
                            Ok(Value::Tensor(result))
                        } else {
                            Err(TenthError::RuntimeError { line: None, col: None,
                                message: "masked_fill() mask 必须是张量".into(),
                            })
                        }
                    }
                    "permute" => {
                        // x.permute(dims...)
                        let dims: Vec<usize> = args.iter()
                            .map(|a| a.as_int().unwrap_or(0) as usize)
                            .collect();
                        let result_tensor = tensor.permute(&dims)
                            .map_err(|msg| TenthError::RuntimeError { line: None, col: None, message: msg })?;
                        Ok(Value::Tensor(Rc::new(RefCell::new(result_tensor))))
                    }
                    "softmax" => {
                        let result_tensor = tensor.softmax().ok_or_else(|| TenthError::RuntimeError { line: None, col: None,
                            message: "softmax 失败".into(),
                        })?;
                        let result = Rc::new(RefCell::new(result_tensor));
                        if self.recording {
                            self.record_unary(TapeOp::Softmax, t, &result);
                        }
                        Ok(Value::Tensor(result))
                    }
                    "argmax" => Ok(Value::Int(tensor.argmax(), BaseType::I32)),
                    // 梯度裁剪辅助：元素级裁剪到 [min_val, max_val]
                    // 用于 std::optim::clip 模块，避免依赖 tensor mask
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
                    // 张量属性查询（配合护城河 D 内存预估）
                    "numel" => Ok(Value::Int(tensor.data.len() as i64, BaseType::I32)),
                    "nbytes" | "bytes" => {
                        // 字节数 = 元素数 * dtype 字节数
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
                        Ok(Value::Tensor(Rc::new(RefCell::new(
                            Tensor::from_data(ArrayD::from_shape_vec(IxDyn(&[shape.len()]), shape).unwrap())
                        ))))
                    }
                    _ => Err(TenthError::RuntimeError { line: None, col: None,
                        message: format!("未知的张量方法: {}", method),
                    }),
                }
            }
            Value::Struct { .. } => Err(TenthError::RuntimeError { line: None, col: None,
                message: format!("未知的方法 '{}'", method),
            }),
            _ => Err(TenthError::RuntimeError { line: None, col: None,
                message: format!("此类型不支持方法 '{}'", method),
            }),
        }
    }

    pub(super) fn eval_scalar_method(&self, val: f64, method: &str, _args: &[Value]) -> TenthResult<Value> {
        match method {
            "sqrt" => Ok(Value::Float(val.sqrt())),
            "abs" => Ok(Value::Float(val.abs())),
            "exp" => Ok(Value::Float(val.exp())),
            "log" => Ok(Value::Float(val.ln())),
            _ => Err(TenthError::RuntimeError { line: None, col: None,
                message: format!("标量上未知的方法 '{}'", method),
            }),
        }
    }
}

