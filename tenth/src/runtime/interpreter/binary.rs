//! 二元/一元运算与值转换。
//!
//! 从 `interpreter.rs` 第 1365-1778 行迁移而来。包含：
//! - `eval_binary`：算术/比较/逻辑运算（含 f32 标量与张量分支）
//! - `eval_unary`：取负/逻辑非/`?` 传播
//! - `values_eq` / `value_to_string`：值比较与字符串化

use std::rc::Rc;
use std::cell::RefCell;
use crate::error::{TenthError, TenthResult};
use crate::hir::hir::*;
use crate::runtime::value::{Value, check_int_overflow, int_overflow_err, promote_int_dtype, value_to_display_string, deref_wrapped, is_wrapped};
use crate::runtime::tensor::Tensor;
use crate::runtime::autodiff::TapeOp;

impl super::Interpreter {
    pub(super) fn eval_binary(&mut self, op: &BinOp, l: &Value, r: &Value) -> TenthResult<Value> {
        // L2.3a-a2：解释器路径 Vec.push 会用 Shared 包裹元素，导致 `acc + Vec.get(i)`
        // （std::collections::iter::sum / collections::product 等）在解释器路径报
        // "加法类型不匹配"，而 VM 路径正常（VM 的 Vec 元素不包裹）。
        // 运算前统一解壳 Shared/Ref/MutRef/SharedBox，使解释器与 VM 行为一致。
        // deref_wrapped 递归处理多层包裹；非包裹值直接返回 clone（无性能影响路径仅多一次 clone）。
        if is_wrapped(l) || is_wrapped(r) {
            let l = deref_wrapped(l);
            let r = deref_wrapped(r);
            return self.eval_binary(op, &l, &r);
        }
        match op {
            BinOp::Add => match (l, r) {
                (Value::Int(a, dt), Value::Int(b, rt)) => {
                    // AUDIT-11.4.53 R4：整数混合提升（可交换）——两边 dtype 的公共 dtype，
                    // 与 VM `add_priv` 同规则，消除 `x_i64 + 1` / `1 + x_i64` 的 dtype 分叉。
                    let dt = &promote_int_dtype(*dt, *rt);
                    // AUDIT-11.4.17：checked_add 拦截 i64 层溢出
                    let s = a.checked_add(*b).ok_or_else(|| int_overflow_err(*dt))?;
                    check_int_overflow(s, *dt)?;
                    Ok(Value::Int(s, *dt))
                }
                (Value::Float(a), Value::Float(b)) => Ok(Value::Float(a + b)),
                (Value::Int(a, _), Value::Float(b)) => Ok(Value::Float(*a as f64 + b)),
                (Value::Float(a), Value::Int(b, _)) => Ok(Value::Float(a + *b as f64)),
                // Phase 5.4：f32 标量算术（按 promote_float_dtype 规则：F32 op F32/Int → F32，F32 op F64 → F64）
                (Value::Float32(a), Value::Float32(b)) => Ok(Value::Float32(a + b)),
                (Value::Int(a, _), Value::Float32(b)) => Ok(Value::Float32(*a as f32 + b)),
                (Value::Float32(a), Value::Int(b, _)) => Ok(Value::Float32(a + *b as f32)),
                (Value::Float(a), Value::Float32(b)) => Ok(Value::Float(a + *b as f64)),
                (Value::Float32(a), Value::Float(b)) => Ok(Value::Float(*a as f64 + b)),
                (Value::String(a), Value::String(b)) => Ok(Value::String(format!("{}{}", a, b))),
                (Value::Tensor(t1), Value::Tensor(t2)) => {
                    let result_tensor = t1.borrow().add_tensor(&t2.borrow())
                        .map_err(|msg| TenthError::RuntimeError { line: None, col: None, message: msg })?;
                    let result = Rc::new(RefCell::new(result_tensor));
                    if self.recording {
                        self.record_binary(TapeOp::Add, t1, t2, &result);
                    }
                    Ok(Value::Tensor(result))
                }
                (Value::Tensor(t), Value::Float(s)) => {
                    let result = Rc::new(RefCell::new(t.borrow().add_scalar(*s)));
                    if self.recording {
                        let scalar_tensor = Rc::new(RefCell::new(Tensor::full(&t.borrow().shape(), *s)));
                        self.record_binary(TapeOp::Add, t, &scalar_tensor, &result);
                    }
                    Ok(Value::Tensor(result))
                }
                (Value::Float(s), Value::Tensor(t)) => {
                    let result = Rc::new(RefCell::new(t.borrow().add_scalar(*s)));
                    if self.recording {
                        let scalar_tensor = Rc::new(RefCell::new(Tensor::full(&t.borrow().shape(), *s)));
                        self.record_binary(TapeOp::Add, &scalar_tensor, t, &result);
                    }
                    Ok(Value::Tensor(result))
                }
                // Phase 5.4：f32 标量 + Tensor（add_scalar 按 tensor dtype 分支保持精度）
                (Value::Tensor(t), Value::Float32(s)) => {
                    let s_f64 = *s as f64;
                    let result = Rc::new(RefCell::new(t.borrow().add_scalar(s_f64)));
                    if self.recording {
                        let scalar_tensor = Rc::new(RefCell::new(Tensor::full(&t.borrow().shape(), s_f64)));
                        self.record_binary(TapeOp::Add, t, &scalar_tensor, &result);
                    }
                    Ok(Value::Tensor(result))
                }
                (Value::Float32(s), Value::Tensor(t)) => {
                    let s_f64 = *s as f64;
                    let result = Rc::new(RefCell::new(t.borrow().add_scalar(s_f64)));
                    if self.recording {
                        let scalar_tensor = Rc::new(RefCell::new(Tensor::full(&t.borrow().shape(), s_f64)));
                        self.record_binary(TapeOp::Add, &scalar_tensor, t, &result);
                    }
                    Ok(Value::Tensor(result))
                }
                _ => Err(TenthError::RuntimeError { line: None, col: None,
                    message: "加法类型不匹配".into(),
                }),
            },
            BinOp::Sub => match (l, r) {
                // AUDIT-11.4.17：与 VM sub_priv 对齐——公共 dtype + 范围检查
                (Value::Int(a, dt), Value::Int(b, rt)) => {
                    let dt = &promote_int_dtype(*dt, *rt);
                    let s = a.checked_sub(*b).ok_or_else(|| int_overflow_err(*dt))?;
                    check_int_overflow(s, *dt)?;
                    Ok(Value::Int(s, *dt))
                }
                (Value::Float(a), Value::Float(b)) => Ok(Value::Float(a - b)),
                (Value::Int(a, _), Value::Float(b)) => Ok(Value::Float(*a as f64 - b)),
                (Value::Float(a), Value::Int(b, _)) => Ok(Value::Float(a - *b as f64)),
                // Phase 5.4：f32 标量算术
                (Value::Float32(a), Value::Float32(b)) => Ok(Value::Float32(a - b)),
                (Value::Int(a, _), Value::Float32(b)) => Ok(Value::Float32(*a as f32 - b)),
                (Value::Float32(a), Value::Int(b, _)) => Ok(Value::Float32(a - *b as f32)),
                (Value::Float(a), Value::Float32(b)) => Ok(Value::Float(a - *b as f64)),
                (Value::Float32(a), Value::Float(b)) => Ok(Value::Float(*a as f64 - b)),
                (Value::Tensor(t1), Value::Tensor(t2)) => {
                    let result_tensor = t1.borrow().sub_tensor(&t2.borrow())
                        .map_err(|msg| TenthError::RuntimeError { line: None, col: None, message: msg })?;
                    let result = Rc::new(RefCell::new(result_tensor));
                    if self.recording {
                        self.record_binary(TapeOp::Sub, t1, t2, &result);
                    }
                    Ok(Value::Tensor(result))
                }
                (Value::Tensor(t), Value::Float(s)) => {
                    let result = Rc::new(RefCell::new(t.borrow().sub_scalar(*s)));
                    if self.recording {
                        let scalar_tensor = Rc::new(RefCell::new(Tensor::full(&t.borrow().shape(), *s)));
                        self.record_binary(TapeOp::Sub, t, &scalar_tensor, &result);
                    }
                    Ok(Value::Tensor(result))
                }
                (Value::Float(s), Value::Tensor(t)) => {
                    // s - t: compute as -(t - s) but record as Sub(scalar, t)
                    // because d(s-t)/dt = -1 = d(scalar - t)/dt
                    let result = Rc::new(RefCell::new(t.borrow().sub_scalar(*s).neg()));
                    if self.recording {
                        let scalar_tensor = Rc::new(RefCell::new(Tensor::full(&t.borrow().shape(), *s)));
                        self.record_binary(TapeOp::Sub, &scalar_tensor, t, &result);
                    }
                    Ok(Value::Tensor(result))
                }
                // Phase 5.4：f32 标量 - Tensor / Tensor - f32 标量
                (Value::Tensor(t), Value::Float32(s)) => {
                    let s_f64 = *s as f64;
                    let result = Rc::new(RefCell::new(t.borrow().sub_scalar(s_f64)));
                    if self.recording {
                        let scalar_tensor = Rc::new(RefCell::new(Tensor::full(&t.borrow().shape(), s_f64)));
                        self.record_binary(TapeOp::Sub, t, &scalar_tensor, &result);
                    }
                    Ok(Value::Tensor(result))
                }
                (Value::Float32(s), Value::Tensor(t)) => {
                    let s_f64 = *s as f64;
                    let result = Rc::new(RefCell::new(t.borrow().sub_scalar(s_f64).neg()));
                    if self.recording {
                        let scalar_tensor = Rc::new(RefCell::new(Tensor::full(&t.borrow().shape(), s_f64)));
                        self.record_binary(TapeOp::Sub, &scalar_tensor, t, &result);
                    }
                    Ok(Value::Tensor(result))
                }
                _ => Err(TenthError::RuntimeError { line: None, col: None,
                    message: "减法类型不匹配".into(),
                }),
            },
            BinOp::Mul => match (l, r) {
                (Value::Int(a, dt), Value::Int(b, rt)) => {
                    // AUDIT-11.4.53 R4：公共 dtype（可交换）
                    let dt = &promote_int_dtype(*dt, *rt);
                    // AUDIT-11.4.17：checked_mul 拦截 i64 层溢出
                    let s = a.checked_mul(*b).ok_or_else(|| int_overflow_err(*dt))?;
                    check_int_overflow(s, *dt)?;
                    Ok(Value::Int(s, *dt))
                }
                (Value::Float(a), Value::Float(b)) => Ok(Value::Float(a * b)),
                (Value::Int(a, _), Value::Float(b)) => Ok(Value::Float(*a as f64 * b)),
                (Value::Float(a), Value::Int(b, _)) => Ok(Value::Float(a * *b as f64)),
                // Phase 5.4：f32 标量算术
                (Value::Float32(a), Value::Float32(b)) => Ok(Value::Float32(a * b)),
                (Value::Int(a, _), Value::Float32(b)) => Ok(Value::Float32(*a as f32 * b)),
                (Value::Float32(a), Value::Int(b, _)) => Ok(Value::Float32(a * *b as f32)),
                (Value::Float(a), Value::Float32(b)) => Ok(Value::Float(a * *b as f64)),
                (Value::Float32(a), Value::Float(b)) => Ok(Value::Float(*a as f64 * b)),
                (Value::Tensor(t1), Value::Tensor(t2)) => {
                    let result_tensor = t1.borrow().mul_tensor(&t2.borrow())
                        .map_err(|msg| TenthError::RuntimeError { line: None, col: None, message: msg })?;
                    let result = Rc::new(RefCell::new(result_tensor));
                    if self.recording {
                        self.record_binary(TapeOp::Mul, t1, t2, &result);
                    }
                    Ok(Value::Tensor(result))
                }
                (Value::Tensor(t), Value::Float(s)) => {
                    let result = Rc::new(RefCell::new(t.borrow().mul_scalar(*s)));
                    if self.recording {
                        let scalar_tensor = Rc::new(RefCell::new(Tensor::full(&t.borrow().shape(), *s)));
                        self.record_binary(TapeOp::Mul, t, &scalar_tensor, &result);
                    }
                    Ok(Value::Tensor(result))
                }
                (Value::Float(s), Value::Tensor(t)) => {
                    let result = Rc::new(RefCell::new(t.borrow().mul_scalar(*s)));
                    if self.recording {
                        let scalar_tensor = Rc::new(RefCell::new(Tensor::full(&t.borrow().shape(), *s)));
                        self.record_binary(TapeOp::Mul, &scalar_tensor, t, &result);
                    }
                    Ok(Value::Tensor(result))
                }
                // Phase 5.4：f32 标量 × Tensor（mul_scalar 按 tensor dtype 分支保持精度）
                (Value::Tensor(t), Value::Float32(s)) => {
                    let s_f64 = *s as f64;
                    let result = Rc::new(RefCell::new(t.borrow().mul_scalar(s_f64)));
                    if self.recording {
                        let scalar_tensor = Rc::new(RefCell::new(Tensor::full(&t.borrow().shape(), s_f64)));
                        self.record_binary(TapeOp::Mul, t, &scalar_tensor, &result);
                    }
                    Ok(Value::Tensor(result))
                }
                (Value::Float32(s), Value::Tensor(t)) => {
                    let s_f64 = *s as f64;
                    let result = Rc::new(RefCell::new(t.borrow().mul_scalar(s_f64)));
                    if self.recording {
                        let scalar_tensor = Rc::new(RefCell::new(Tensor::full(&t.borrow().shape(), s_f64)));
                        self.record_binary(TapeOp::Mul, &scalar_tensor, t, &result);
                    }
                    Ok(Value::Tensor(result))
                }
                _ => Err(TenthError::RuntimeError { line: None, col: None,
                    message: "乘法类型不匹配".into(),
                }),
            },
            BinOp::Div => match (l, r) {
                (Value::Int(a, dt), Value::Int(b, rt)) => {
                    // AUDIT-11.4.53 R4：公共 dtype（可交换）
                    let dt = &promote_int_dtype(*dt, *rt);
                    if *b == 0 {
                        return Err(TenthError::RuntimeError { line: None, col: None,
                            message: "整数除零".into(),
                        });
                    }
                    // AUDIT-11.4.17：checked_div 拦截 i64::MIN / -1 等溢出
                    let s = a.checked_div(*b).ok_or_else(|| int_overflow_err(*dt))?;
                    check_int_overflow(s, *dt)?;
                    Ok(Value::Int(s, *dt))
                }
                (Value::Float(a), Value::Float(b)) => Ok(Value::Float(a / b)),
                (Value::Int(a, _), Value::Float(b)) => Ok(Value::Float(*a as f64 / b)),
                (Value::Float(a), Value::Int(b, _)) => Ok(Value::Float(a / *b as f64)),
                // Phase 5.4：f32 标量算术
                (Value::Float32(a), Value::Float32(b)) => Ok(Value::Float32(a / b)),
                (Value::Int(a, _), Value::Float32(b)) => Ok(Value::Float32(*a as f32 / b)),
                (Value::Float32(a), Value::Int(b, _)) => Ok(Value::Float32(a / *b as f32)),
                (Value::Float(a), Value::Float32(b)) => Ok(Value::Float(a / *b as f64)),
                (Value::Float32(a), Value::Float(b)) => Ok(Value::Float(*a as f64 / b)),
                (Value::Tensor(t1), Value::Tensor(t2)) => {
                    let result_tensor = t1.borrow().div_tensor(&t2.borrow())
                        .map_err(|msg| TenthError::RuntimeError { line: None, col: None, message: msg })?;
                    let result = Rc::new(RefCell::new(result_tensor));
                    if self.recording {
                        self.record_binary(TapeOp::Div, t1, t2, &result);
                    }
                    Ok(Value::Tensor(result))
                }
                (Value::Tensor(t), Value::Float(s)) => {
                    let result = Rc::new(RefCell::new(t.borrow().div_scalar(*s)));
                    if self.recording {
                        let scalar_tensor = Rc::new(RefCell::new(Tensor::full(&t.borrow().shape(), *s)));
                        self.record_binary(TapeOp::Div, t, &scalar_tensor, &result);
                    }
                    Ok(Value::Tensor(result))
                }
                // Phase 5.4：f32 标量 / Tensor / Tensor / f32 标量
                (Value::Tensor(t), Value::Float32(s)) => {
                    let s_f64 = *s as f64;
                    let result = Rc::new(RefCell::new(t.borrow().div_scalar(s_f64)));
                    if self.recording {
                        let scalar_tensor = Rc::new(RefCell::new(Tensor::full(&t.borrow().shape(), s_f64)));
                        self.record_binary(TapeOp::Div, t, &scalar_tensor, &result);
                    }
                    Ok(Value::Tensor(result))
                }
                (Value::Float32(s), Value::Tensor(t)) => {
                    let s_f64 = *s as f64;
                    let result = Rc::new(RefCell::new(t.borrow().div_scalar_inv(s_f64)));
                    if self.recording {
                        let scalar_tensor = Rc::new(RefCell::new(Tensor::full(&t.borrow().shape(), s_f64)));
                        self.record_binary(TapeOp::Div, &scalar_tensor, t, &result);
                    }
                    Ok(Value::Tensor(result))
                }
                _ => Err(TenthError::RuntimeError { line: None, col: None,
                    message: "除法类型不匹配".into(),
                }),
            },
            BinOp::Mod => match (l, r) {
                (Value::Int(a, dt), Value::Int(b, rt)) => {
                    // AUDIT-11.4.53 R4：公共 dtype（可交换）
                    let dt = &promote_int_dtype(*dt, *rt);
                    if *b == 0 {
                        return Err(TenthError::RuntimeError { line: None, col: None,
                            message: "整数取模除零".into(),
                        });
                    }
                    // AUDIT-11.4.17：checked_rem 拦截 i64::MIN % -1 等溢出
                    let s = a.checked_rem(*b).ok_or_else(|| int_overflow_err(*dt))?;
                    check_int_overflow(s, *dt)?;
                    Ok(Value::Int(s, *dt))
                }
                _ => Err(TenthError::RuntimeError { line: None, col: None,
                    message: "取模仅支持整数".into(),
                }),
            },
            BinOp::Eq => Ok(Value::Bool(self.values_eq(l, r))),
            BinOp::NotEq => Ok(Value::Bool(!self.values_eq(l, r))),
            BinOp::Lt => match (l, r) {
                (Value::Int(a, _), Value::Int(b, _)) => Ok(Value::Bool(a < b)),
                (Value::Float(a), Value::Float(b)) => Ok(Value::Bool(a < b)),
                (Value::Int(a, _), Value::Float(b)) => Ok(Value::Bool((*a as f64) < *b)),
                (Value::Float(a), Value::Int(b, _)) => Ok(Value::Bool(*a < *b as f64)),
                (Value::String(a), Value::String(b)) => Ok(Value::Bool(a < b)),
                _ => Err(TenthError::RuntimeError { line: None, col: None,
                    message: "比较需要数值类型".into(),
                }),
            },
            BinOp::Gt => match (l, r) {
                (Value::Int(a, _), Value::Int(b, _)) => Ok(Value::Bool(a > b)),
                (Value::Float(a), Value::Float(b)) => Ok(Value::Bool(a > b)),
                (Value::Int(a, _), Value::Float(b)) => Ok(Value::Bool((*a as f64) > *b)),
                (Value::Float(a), Value::Int(b, _)) => Ok(Value::Bool(*a > *b as f64)),
                (Value::String(a), Value::String(b)) => Ok(Value::Bool(a > b)),
                _ => Err(TenthError::RuntimeError { line: None, col: None,
                    message: "比较需要数值类型".into(),
                }),
            },
            BinOp::LtEq => match (l, r) {
                (Value::Int(a, _), Value::Int(b, _)) => Ok(Value::Bool(a <= b)),
                (Value::Float(a), Value::Float(b)) => Ok(Value::Bool(a <= b)),
                (Value::Int(a, _), Value::Float(b)) => Ok(Value::Bool((*a as f64) <= *b)),
                (Value::Float(a), Value::Int(b, _)) => Ok(Value::Bool(*a <= *b as f64)),
                (Value::String(a), Value::String(b)) => Ok(Value::Bool(a <= b)),
                _ => Err(TenthError::RuntimeError { line: None, col: None,
                    message: "比较需要数值类型".into(),
                }),
            },
            BinOp::GtEq => match (l, r) {
                (Value::Int(a, _), Value::Int(b, _)) => Ok(Value::Bool(a >= b)),
                (Value::Float(a), Value::Float(b)) => Ok(Value::Bool(a >= b)),
                (Value::Int(a, _), Value::Float(b)) => Ok(Value::Bool((*a as f64) >= *b)),
                (Value::Float(a), Value::Int(b, _)) => Ok(Value::Bool(*a >= *b as f64)),
                (Value::String(a), Value::String(b)) => Ok(Value::Bool(a >= b)),
                _ => Err(TenthError::RuntimeError { line: None, col: None,
                    message: "比较需要数值类型".into(),
                }),
            },
            BinOp::And => Ok(Value::Bool(l.is_truthy() && r.is_truthy())),
            BinOp::Or => Ok(Value::Bool(l.is_truthy() || r.is_truthy())),
        }
    }

    pub(super) fn values_eq(&self, l: &Value, r: &Value) -> bool {
        match (l, r) {
            (Value::Int(a, _), Value::Int(b, _)) => a == b,
            (Value::Float(a), Value::Float(b)) => (a - b).abs() < 1e-10,
            (Value::Float32(a), Value::Float32(b)) => (a - b).abs() < 1e-6,
            // 跨类型数值比较：与 <、>、<=、>= 保持一致（问题7）
            (Value::Int(a, _), Value::Float(b)) => ((*a as f64) - b).abs() < 1e-10,
            (Value::Float(a), Value::Int(b, _)) => (a - (*b as f64)).abs() < 1e-10,
            // Float32 与其他数值类型跨类型比较（问题8 扩展，保持一致性）
            (Value::Int(a, _), Value::Float32(b)) => ((*a as f32) - b).abs() < 1e-6,
            (Value::Float32(a), Value::Int(b, _)) => (a - (*b as f32)).abs() < 1e-6,
            (Value::Float(a), Value::Float32(b)) => ((*a as f32) - b).abs() < 1e-6,
            (Value::Float32(a), Value::Float(b)) => (a - (*b as f32)).abs() < 1e-6,
            (Value::Bool(a), Value::Bool(b)) => a == b,
            (Value::Char(a), Value::Char(b)) => a == b,
            (Value::String(a), Value::String(b)) => a == b,
            (Value::Unit, Value::Unit) => true,
            (Value::Tuple(a), Value::Tuple(b)) => {
                a.len() == b.len()
                    && a.iter().zip(b.iter()).all(|(x, y)| self.values_eq(x, y))
            }
            // AUDIT-11.4.24：Vec/Array 相等比较——元素逐一相等（解包 Shared 包裹）。
            // 此前落入 `_ => false`，元素相同也返回 false（base64/hex 断言静默失败）。
            // 解释器数组字面量元素是 Value::Shared，需 deref_wrapped 再比较。
            (Value::Vec(a), Value::Vec(b))
            | (Value::Array(a), Value::Array(b))
            | (Value::Vec(a), Value::Array(b))
            | (Value::Array(a), Value::Vec(b)) => {
                let a = a.borrow();
                let b = b.borrow();
                a.len() == b.len()
                    && a.iter().zip(b.iter()).all(|(x, y)| {
                        self.values_eq(&deref_wrapped(x), &deref_wrapped(y))
                    })
            }
            _ => false,
        }
    }

    /// 值 → 字符串。**仅转发**到单一权威实现 `value_to_display_string`
    /// （P2-B6：消除值字符串化双实现）。解释器的 to_string native 与字符串插值
    /// 都调用这里，从而与 VM 的 print/println/to_string（也走 Display）输出一致。
    pub(super) fn value_to_string(&self, val: &Value) -> String {
        value_to_display_string(val)
    }

    pub(super) fn eval_unary(&self, op: &UnaryOp, val: &Value) -> TenthResult<Value> {
        match op {
            UnaryOp::Neg => match val {
                Value::Int(n, dt) => {
                    // AUDIT-11.4.17：checked_neg 拦截 i64::MIN 取负溢出
                    let s = n.checked_neg().ok_or_else(|| int_overflow_err(*dt))?;
                    check_int_overflow(s, *dt)?;
                    Ok(Value::Int(s, *dt))
                }
                Value::Float(n) => Ok(Value::Float(-n)),
                Value::Tensor(t) => {
                    let result = t.borrow().neg();
                    Ok(Value::Tensor(Rc::new(RefCell::new(result))))
                }
                _ => Err(TenthError::RuntimeError { line: None, col: None,
                    message: "无法对此值取负".into(),
                }),
            },
            UnaryOp::Not => Ok(Value::Bool(!val.is_truthy())),
            UnaryOp::Try => {
                // `expr?` — if Result::Err, propagate early return; if Result::Ok, unwrap
                match val {
                    Value::Enum { enum_name, variant, fields } => {
                        if enum_name == "Result" {
                            if variant == "Ok" {
                                // Unwrap: return the inner value
                                let borrowed = fields.borrow();
                                if let Some((_, v)) = borrowed.first() {
                                    return Ok(v.clone());
                                }
                                return Ok(Value::Unit);
                            } else if variant == "Err" {
                                // Propagate: return the Err as a special early-return signal
                                return Err(TenthError::TryPropagate(val.clone()));
                            }
                        }
                        // Non-Result: just pass through
                        Ok(val.clone())
                    }
                    _ => Ok(val.clone()),
                }
            }
        }
    }
}
