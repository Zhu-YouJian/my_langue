//! 字段访问与模式匹配。
//!
//! 从 `interpreter.rs` 第 1142-1330 行迁移而来。包含：
//! - `eval_field`：结构体/枚举字段访问（自动解引用 Ref/MutRef/Shared）
//! - `pattern_matches` / `bind_pattern` / `unbind_pattern`：match 表达式的
//!   模式测试、变量绑定与解绑

use crate::error::{TenthError, TenthResult};
use crate::hir::hir::*;
use crate::runtime::value::Value;

impl super::Interpreter {
    pub(super) fn eval_field(&self, val: &Value, field: &str) -> TenthResult<Option<Value>> {
        // Auto-dereference Ref/MutRef/Shared to reach the struct/enum
        let v = match val {
            Value::Ref(rc) => {
                let inner = rc.borrow();
                return self.eval_field(&inner, field);
            }
            Value::MutRef(weak) => {
                if let Some(rc) = weak.upgrade() {
                    let inner = rc.borrow();
                    return self.eval_field(&inner, field);
                }
                return Err(TenthError::RuntimeError {
                    message: format!("无法访问悬垂 &mut 引用上的字段 '{}'", field),
                });
            }
            Value::Shared(rc) => {
                let inner = rc.borrow();
                return self.eval_field(&inner, field);
            }
            v => v,
        };

        match v {
            Value::Struct { fields, .. } => {
                for (fname, fval) in fields.borrow().iter() {
                    if fname == field {
                        return Ok(Some(fval.clone()));
                    }
                }
                Err(TenthError::RuntimeError {
                    message: format!("结构体没有字段 '{}'", field),
                })
            }
            Value::Enum { fields, .. } => {
                for (fname, fval) in fields.borrow().iter() {
                    if fname == field {
                        return Ok(Some(fval.clone()));
                    }
                }
                Err(TenthError::RuntimeError {
                    message: format!("枚举变体没有字段 '{}'", field),
                })
            }
            Value::Vec(items) => {
                // Allow .len() on Vec — handled in MethodCall, but also allow field-style access
                if field == "len" {
                    return Ok(Some(Value::Int(items.borrow().len() as i64)));
                }
                Err(TenthError::RuntimeError {
                    message: format!("Vec 没有字段 '{}'", field),
                })
            }
            _ => Err(TenthError::RuntimeError {
                message: format!("无法访问 {:?} 上的字段 '{}'", v, field),
            }),
        }
    }

    pub(super) fn pattern_matches(&self, pattern: &HirPattern, val: &Value) -> bool {
        match pattern {
            HirPattern::Wildcard => true,
            HirPattern::Binding(_) => true,
            HirPattern::Literal(lit) => {
                match (lit, val) {
                    (Literal::Int(a), Value::Int(b)) => a == b,
                    (Literal::Float(a, _), Value::Float(b)) => (a - b).abs() < 1e-10,
                    (Literal::Float(a, _), Value::Float32(b)) => ((a - *b as f64).abs() as f64) < 1e-6,
                    (Literal::Bool(a), Value::Bool(b)) => a == b,
                    _ => false,
                }
            }
            HirPattern::EnumVariant { enum_name, variant, .. } => {
                match val {
                    Value::Enum { enum_name: e, variant: v, .. } => {
                        enum_name == e && variant == v
                    }
                    _ => false,
                }
            }
            HirPattern::Tuple(patterns) => {
                match val {
                    Value::Tuple(items) if items.len() == patterns.len() => {
                        patterns.iter().zip(items.iter())
                            .all(|(p, v)| self.pattern_matches(p, v))
                    }
                    Value::Vec(items) => {
                        let items_ref = items.borrow();
                        items_ref.len() == patterns.len()
                            && patterns.iter().zip(items_ref.iter())
                                .all(|(p, v)| self.pattern_matches(p, v))
                    }
                    _ => false,
                }
            }
            HirPattern::Range { start, end, inclusive } => {
                match val {
                    Value::Int(n) => {
                        if *inclusive {
                            *n >= *start && *n <= *end
                        } else {
                            *n >= *start && *n < *end
                        }
                    }
                    _ => false,
                }
            }
            HirPattern::Struct { name, .. } => {
                match val {
                    Value::Struct { name: struct_name, .. } => struct_name == name,
                    _ => false,
                }
            }
        }
    }

    /// Bind variables from a matched pattern into the current scope.
    pub(super) fn bind_pattern(&mut self, pattern: &HirPattern, val: &Value) {
        match pattern {
            HirPattern::Binding(name) => {
                self.insert_var(name.clone(), val.clone());
            }
            HirPattern::EnumVariant { field_bind, tuple_binds, .. } => {
                if let Value::Enum { fields, .. } = val {
                    let fields_ref = fields.borrow();
                    if let Some((_fname, bname)) = field_bind {
                        if let Some((_, v)) = fields_ref.first() {
                            self.insert_var(bname.clone(), v.clone());
                        }
                    }
                    for (field_name, bind_name) in tuple_binds {
                        if let Some((_, v)) = fields_ref.iter().find(|(n, _)| n == field_name) {
                            self.insert_var(bind_name.clone(), v.clone());
                        }
                    }
                }
            }
            HirPattern::Tuple(patterns) => {
                match val {
                    Value::Tuple(items) => {
                        for (p, v) in patterns.iter().zip(items.iter()) {
                            self.bind_pattern(p, v);
                        }
                    }
                    Value::Vec(items) => {
                        let items_ref = items.borrow();
                        for (p, v) in patterns.iter().zip(items_ref.iter()) {
                            self.bind_pattern(p, v);
                        }
                    }
                    _ => {}
                }
            }
            HirPattern::Struct { fields, .. } => {
                if let Value::Struct { fields: val_fields, .. } = val {
                    let val_fields_ref = val_fields.borrow();
                    for (field_name, bind_name) in fields {
                        if let Some((_, v)) = val_fields_ref.iter().find(|(n, _)| n == field_name) {
                            self.insert_var(bind_name.clone(), v.clone());
                        }
                    }
                }
            }
            HirPattern::Wildcard | HirPattern::Literal(_) | HirPattern::Range { .. } => {}
        }
    }

    /// Remove variables bound by a pattern from the current scope.
    pub(super) fn unbind_pattern(&mut self, pattern: &HirPattern) {
        match pattern {
            HirPattern::Binding(name) => {
                self.remove_var(name);
            }
            HirPattern::EnumVariant { field_bind, tuple_binds, .. } => {
                if let Some((_, bname)) = field_bind {
                    self.remove_var(bname);
                }
                for (_, bind_name) in tuple_binds {
                    self.remove_var(bind_name);
                }
            }
            HirPattern::Tuple(patterns) => {
                for p in patterns {
                    self.unbind_pattern(p);
                }
            }
            HirPattern::Struct { fields, .. } => {
                for (_, bind_name) in fields {
                    self.remove_var(bind_name);
                }
            }
            HirPattern::Wildcard | HirPattern::Literal(_) | HirPattern::Range { .. } => {}
        }
    }
}
