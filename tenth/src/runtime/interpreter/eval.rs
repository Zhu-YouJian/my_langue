//! 解释器执行逻辑：`eval_expr` / `eval_call` / `eval_stmt`。
//!
//! 从 `mod.rs` 拆出（架构重构 T3e），包含：
//! - `eval_expr`：HIR 表达式求值主分派（字面量、变量、二元/一元、调用、方法调用、
//!   索引、字段、数组/张量字面量、if、block、赋值、闭包、range、引用、解引用、
//!   字段赋值、move、try 块、await/spawn、插值字符串、元组、结构体/枚举字面量、match）
//! - `eval_call`：函数/闭包/枚举构造器调用分派
//! - `eval_stmt`：语句执行（let/return/break/continue/loop/while/for）

use std::collections::HashMap;
use crate::hir::types::BaseType;
use std::rc::Rc;
use std::cell::RefCell;
use crate::error::{TenthError, TenthResult};
use crate::hir::hir::*;
use crate::hir::types::Type;
use crate::runtime::value::Value;
use crate::runtime::tensor::Tensor;

impl super::Interpreter {
    pub(super) fn eval_expr(&mut self, expr: &HirExpr) -> TenthResult<Option<Value>> {
        use HirExprKind;

        self.tick()?;
        match &expr.kind {
            HirExprKind::Literal(lit) => {
                Ok(Some(match lit {
                    Literal::Int(n, _) => Value::Int(*n, BaseType::I32),
                    Literal::Float(n, dt) => match dt {
                        crate::hir::types::BaseType::F32 => Value::Float32(*n as f32),
                        _ => Value::Float(*n),
                    },
                    Literal::Bool(b) => Value::Bool(*b),
                    Literal::Char(c) => Value::Char(*c),
                    Literal::String(s) => Value::String(s.clone()),
                }))
            }

            HirExprKind::Var(name) => {
                if name.contains("::") {
                    let parts: Vec<&str> = name.splitn(2, "::").collect();
                    if parts.len() == 2 {
                        let mod_name = parts[0];
                        let item_name = parts[1];
                        if let Some(module) = self.modules.get(mod_name) {
                            if let Some(fn_def) = module.functions.iter().find(|f| f.name == item_name) {
                                return Ok(Some(Value::FnRef {
                                    name: name.clone(),
                                    params: fn_def.params.clone(),
                                    return_type: fn_def.return_type.clone(),
                                }));
                            }
                        }
                    }
                    return Ok(Some(Value::FnRef {
                        name: name.clone(),
                        params: Vec::new(),
                        return_type: Type::Unknown,
                    }));
                }
                if self.vars.get(name).map_or(false, |s| s.iter().any(|(_, v)| matches!(v, Value::Moved))) {
                    return Err(TenthError::RuntimeError { line: Some(expr.span.line), col: Some(expr.span.col),
                        message: format!("使用了已移动的值 '{}'", name),
                    });
                }
                self.resolve_var(name)
                    .or_else(|| {
                        match name.as_str() {
                            "println" | "print" | "eprintln" | "eprint" | "tensor" | "rand" | "randn" | "randn_f32" | "rand_f32" | "zeros_f32" | "ones_f32"
                            | "read_file" | "write_file" | "write_bytes" | "read_bytes" | "compile_host"
                            | "compile_program"
                            | "read_line" | "env_get" | "env_set" | "exit"
                            // 阶段1-静默失败：Result/Option 显式解包原语（自由函数 native）
                            | "or_die" | "assume_ok"
                            // M1.3：dyn Trait 升级 native（into_dyn(value, trait_name)）
                            | "into_dyn"
                            | "Vec::new" | "HashMap::new"
                            | "start_grad" | "new_grad" | "stop_grad"
                            | "param" | "backward" | "grad" | "zero_grad"
                            | "cross_entropy"
                            | "select"
                            | "scatter"
                            | "gather"
                            // PROJ-006：自定义可微算子调用 native（Rust 端 register_custom_op + .th wrapper）
                            | "__call_custom_op"
                            // Wave 2 第 4 项：张量比较 native（返回 F64 0.0/1.0 张量）
                            | "tensor_gt" | "tensor_lt" | "tensor_ge" | "tensor_le" | "tensor_eq" | "tensor_ne"
                            | "abs" | "sqrt" | "sin" | "cos" | "ln" | "pow" | "to_float" | "to_f32" | "to_f64" | "tensor_from_vec"
                            | "f64_bits" | "f64_from_bits"
                            | "zeros" | "ones"
                            | "save_weights" | "load_weights"
                            | "format" | "parse_int" | "parse_float"
                            | "to_string" | "type_name"
                            | "assert" | "assert_eq"
                            | "with_step_limit" | "with_timeout_ms" | "is_timeout"
                            | "path_join" | "path_exists" | "path_is_file" | "path_is_dir"
                            | "mkdir" | "list_dir" | "file_size" | "remove_file" | "copy_file" | "rename_file"
                            | "time_now" | "time_now_ms" | "time_date" | "time_time" | "time_datetime" | "time_sleep_ms"
                            | "random_int" | "random_float"
                            | "math_tan" | "math_asin" | "math_acos" | "math_atan" | "math_atan2"
                            | "math_sinh" | "math_cosh" | "math_tanh" | "math_log10" | "math_log2" | "math_exp" | "math_pow"
                            | "math_floor" | "math_ceil" | "math_round"
                            | "cli_args_count" | "cli_arg"
                            | "json_encode" | "json_encode_pretty" | "json_decode"
                            // Stage 3+4 TCP/HTTP 原语
                            | "tcp_connect" | "tcp_read" | "tcp_write" | "tcp_close" | "tcp_set_timeout"
                            | "tcp_listen" | "tcp_accept" | "tcp_listener_close"
                            // UDP 原语（基本功核查第 69 项；handle table 模式，与 tcp_* 同构）
                            | "udp_bind" | "udp_recv_from" | "udp_send_to" | "udp_close" | "udp_set_timeout"
                            | "command_new" | "command_arg" | "command_run" | "command_output"
                            | "http_get" | "http_post"
                            | "regex_compile" | "regex_match" | "regex_find" | "regex_find_all" | "regex_replace" | "regex_split"
                            // Wave 3 第 8 项：Date native（路径 B，复用 struct 机制，返回 i64 或 Tuple）
                            | "date_to_unix_days" | "date_from_unix_days" | "date_i64_add_days"
                            | "date_diff_days" | "date_day_of_week"
                            // M3.4：Weak 弱引用 native（Weak::new 含 :: 走上方 FnRef 路径，自动覆盖）
                            | "weak_upgrade" | "weak_strong_count" | "weak_weak_count"
                            // B批：字符串/文本处理 native
                            | "unicode_nfc" | "unicode_nfd"
                            | "str_to_utf16" | "utf16_to_str"
                            | "str_to_bytes" | "bytes_to_str"
                            | "to_utf8" | "to_utf16" | "from_utf16"
                            | "to_gbk" | "from_gbk"
                            | "base64_encode" | "base64_decode"
                            | "hex_encode" | "hex_decode"
                            | "url_encode" | "url_decode"
                            // 哈希函数（SHA-256/SHA-512/MD5）
                            | "sha256" | "sha512" | "md5"
                            | "sha256_str" | "sha512_str" | "md5_str" => {
                                Some(Value::FnRef {
                                    name: name.clone(),
                                    params: Vec::new(),
                                    return_type: Type::Unknown,
                                })
                            }
                            _ => None,
                        }
                    })
                    .ok_or_else(|| TenthError::RuntimeError { line: Some(expr.span.line), col: Some(expr.span.col),
                        message: format!("未定义变量 '{}'", name),
                    })
                    .map(Some)
            }

            HirExprKind::Binary { op, left, right, .. } => {
                // Short-circuit evaluation for && and ||
                match op {
                    BinOp::And => {
                        let l = self.eval_expr(left)?.ok_or_else(|| TenthError::RuntimeError { line: None, col: None,
                            message: "左操作数为空值".into(),
                        })?;
                        if !l.is_truthy() {
                            return Ok(Some(Value::Bool(false)));
                        }
                        let r = self.eval_expr(right)?.ok_or_else(|| TenthError::RuntimeError { line: None, col: None,
                            message: "右操作数为空值".into(),
                        })?;
                        Ok(Some(Value::Bool(r.is_truthy())))
                    }
                    BinOp::Or => {
                        let l = self.eval_expr(left)?.ok_or_else(|| TenthError::RuntimeError { line: None, col: None,
                            message: "左操作数为空值".into(),
                        })?;
                        if l.is_truthy() {
                            return Ok(Some(Value::Bool(true)));
                        }
                        let r = self.eval_expr(right)?.ok_or_else(|| TenthError::RuntimeError { line: None, col: None,
                            message: "右操作数为空值".into(),
                        })?;
                        Ok(Some(Value::Bool(r.is_truthy())))
                    }
                    _ => {
                        let l = self.eval_expr(left)?.ok_or_else(|| TenthError::RuntimeError { line: None, col: None,
                            message: "左操作数为空值".into(),
                        })?;
                        let r = self.eval_expr(right)?.ok_or_else(|| TenthError::RuntimeError { line: None, col: None,
                            message: "右操作数为空值".into(),
                        })?;
                        self.eval_binary(op, &l, &r).map(Some)
                    }
                }
            }

            HirExprKind::Unary { op, expr: inner, .. } => {
                let val = self.eval_expr(inner)?.ok_or_else(|| TenthError::RuntimeError { line: None, col: None,
                    message: "一元操作数为空值".into(),
                })?;
                self.eval_unary(op, &val).map(Some)
            }

            HirExprKind::Call { func, args, .. } => {
                let f = self.eval_expr(func)?.ok_or_else(|| TenthError::RuntimeError { line: None, col: None,
                    message: "函数值为空值".into(),
                })?;

                let mut arg_values = Vec::new();
                for a in args {
                    arg_values.push(self.eval_expr(a)?.ok_or_else(|| TenthError::RuntimeError { line: None, col: None,
                        message: "参数为空值".into(),
                    })?);
                }

                self.eval_call(&f, &arg_values, &expr.span)
            }

            HirExprKind::GenericCall { func, generics, args, .. } => {
                let func_name = match &func.kind {
                    HirExprKind::Var(name) => name.clone(),
                    _ => {
                        return Err(TenthError::RuntimeError { line: Some(expr.span.line), col: Some(expr.span.col),
                            message: "泛型调用的目标必须是具名函数".into(),
                        });
                    }
                };

                let template = self.generic_funcs.get(&func_name)
                    .ok_or_else(|| TenthError::RuntimeError { line: None, col: None,
                        message: format!("未定义的泛型函数 '{}'", func_name),
                    })?
                    .clone();

                let mut type_map: HashMap<String, Type> = HashMap::new();
                for (i, gen_name) in template.generics.iter().enumerate() {
                    type_map.insert(gen_name.clone(), generics.get(i).cloned().unwrap_or(Type::Unknown));
                }

                let mut arg_values = Vec::new();
                for a in args {
                    arg_values.push(self.eval_expr(a)?.ok_or_else(|| TenthError::RuntimeError { line: None, col: None,
                        message: "参数为空值".into(),
                    })?);
                }

                self.push_scope();
                for ((pname, _), arg) in template.params.iter().zip(arg_values.iter()) {
                    self.insert_var(pname.clone(), arg.clone());
                }

                let result = self.eval_expr(&template.body);

                self.pop_scope();
                result
            }

            HirExprKind::MethodCall { receiver, method, args, .. } => {
                let recv = self.eval_expr(receiver)?.ok_or_else(|| TenthError::RuntimeError { line: None, col: None,
                    message: "接收者为空值".into(),
                })?;

                let mut arg_values = Vec::new();
                for a in args {
                    arg_values.push(self.eval_expr(a)?.ok_or_else(|| TenthError::RuntimeError { line: None, col: None,
                        message: "方法参数为空值".into(),
                    })?);
                }

                self.eval_method_call(&recv, method, &arg_values)
            }

            HirExprKind::Index { target, indices } => {
                let t = self.eval_expr(target)?.ok_or_else(|| TenthError::RuntimeError { line: None, col: None,
                    message: "索引目标为空值".into(),
                })?;
                self.eval_index(&t, indices).map(Some)
            }

            HirExprKind::Field { target, field } => {
                let t = self.eval_expr(target)?.ok_or_else(|| TenthError::RuntimeError { line: None, col: None,
                    message: "字段访问目标为空值".into(),
                })?;
                self.eval_field(&t, field)
            }

            HirExprKind::ArrayLiteral { elements, .. } => {
                let mut vals = Vec::new();
                for elem in elements {
                    let v = self.eval_expr(elem)?.ok_or_else(|| TenthError::RuntimeError { line: None, col: None,
                        message: "数组元素为空值".into(),
                    })?;
                    // Wrap in Shared so elements can be mutated via indexed assignment
                    vals.push(Value::Shared(Rc::new(RefCell::new(v))));
                }
                Ok(Some(Value::Vec(Rc::new(RefCell::new(vals)))))
            }

            HirExprKind::TensorLiteral { data, ty, .. } => {
                // 按 HIR 类型注解的 dtype 选择构造路径
                let is_f32 = ty.tensor_dtype() == Some(crate::hir::types::BaseType::F32);
                let mut rows_f32: Vec<Vec<f32>> = Vec::new();
                let mut rows_f64: Vec<Vec<f64>> = Vec::new();
                for row in data {
                    let mut row_f32: Vec<f32> = Vec::new();
                    let mut row_f64: Vec<f64> = Vec::new();
                    for elem in row {
                        let v = self.eval_expr(elem)?.ok_or_else(|| TenthError::RuntimeError { line: None, col: None,
                            message: "张量元素为空值".into(),
                        })?;
                        if is_f32 {
                            row_f32.push(v.as_f32().unwrap_or(0.0));
                        } else {
                            row_f64.push(v.as_float().unwrap_or(0.0));
                        }
                    }
                    if is_f32 { rows_f32.push(row_f32); } else { rows_f64.push(row_f64); }
                }
                let nrows = if is_f32 { rows_f32.len() } else { rows_f64.len() };
                let ncols = if is_f32 {
                    rows_f32.first().map(|r| r.len()).unwrap_or(0)
                } else {
                    rows_f64.first().map(|r| r.len()).unwrap_or(0)
                };
                if is_f32 {
                    let flat: Vec<f32> = rows_f32.into_iter().flatten().collect();
                    let tensor = Tensor::from_vec_f32(flat, vec![nrows, ncols]);
                    Ok(Some(Value::Tensor(Rc::new(RefCell::new(tensor)))))
                } else {
                    let flat: Vec<f64> = rows_f64.into_iter().flatten().collect();
                    let tensor = self.make_tensor(flat, vec![nrows, ncols])?;
                    Ok(Some(Value::Tensor(Rc::new(RefCell::new(tensor)))))
                }
            }

            HirExprKind::If { cond, then_branch, else_branch, .. } => {
                let c = self.eval_expr(cond)?.ok_or_else(|| TenthError::RuntimeError { line: None, col: None,
                    message: "if 条件为空值".into(),
                })?;
                if c.is_truthy() {
                    self.eval_expr(then_branch)
                } else if let Some(eb) = else_branch {
                    self.eval_expr(eb)
                } else {
                    Ok(Some(Value::Unit))
                }
            }

            HirExprKind::Block { stmts, final_expr } => {
                for stmt in stmts {
                    self.eval_stmt(stmt)?;
                }
                match final_expr {
                    Some(e) => self.eval_expr(e),
                    None => Ok(Some(Value::Unit)),
                }
            }

            HirExprKind::Assign { target, value } => {
                let v = self.eval_expr(value)?.ok_or_else(|| TenthError::RuntimeError { line: None, col: None,
                    message: "赋值值为空值".into(),
                })?;
                self.set_var(target.clone(), v);
                Ok(Some(Value::Unit))
            }

            HirExprKind::DerefAssign { target, value } => {
                let target_val = self.eval_expr(target)?.ok_or_else(|| TenthError::RuntimeError { line: None, col: None,
                    message: "解引用赋值目标为空值".into(),
                })?;
                let rhs = self.eval_expr(value)?.ok_or_else(|| TenthError::RuntimeError { line: None, col: None,
                    message: "解引用赋值值为空值".into(),
                })?;
                match &target_val {
                    Value::MutRef(weak) => {
                        let rc = weak.upgrade().ok_or_else(|| TenthError::RuntimeError { line: None, col: None,
                            message: "无法通过悬垂的 &mut 引用赋值".into(),
                        })?;
                        *rc.borrow_mut() = rhs;
                        Ok(Some(Value::Unit))
                    }
                    _ => Err(TenthError::RuntimeError { line: None, col: None,
                        message: "只能通过可变引用赋值".into(),
                    }),
                }
            }

            HirExprKind::AssignOp { target, op, value } => {
                let current = self.resolve_var(target).ok_or_else(|| {
                    TenthError::RuntimeError { line: None, col: None,
                        message: format!("undefined variable '{}'", target),
                    }
                })?;
                let rhs = self.eval_expr(value)?.ok_or_else(|| TenthError::RuntimeError { line: None, col: None,
                    message: "复合赋值值为空值".into(),
                })?;
                let result = self.eval_binary(op, &current, &rhs)?;
                self.set_var(target.clone(), result);
                Ok(Some(Value::Unit))
            }

            HirExprKind::DerefAssignOp { target, op, value } => {
                let target_val = self.eval_expr(target)?.ok_or_else(|| TenthError::RuntimeError { line: None, col: None,
                    message: "解引用复合赋值目标为空值".into(),
                })?;
                let rhs = self.eval_expr(value)?.ok_or_else(|| TenthError::RuntimeError { line: None, col: None,
                    message: "解引用复合赋值值为空值".into(),
                })?;
                match &target_val {
                    Value::MutRef(weak) => {
                        let rc = weak.upgrade().ok_or_else(|| TenthError::RuntimeError { line: None, col: None,
                            message: "无法通过悬垂的 &mut 引用赋值".into(),
                        })?;
                        let current = rc.borrow().clone();
                        let result = self.eval_binary(op, &current, &rhs)?;
                        *rc.borrow_mut() = result;
                        Ok(Some(Value::Unit))
                    }
                    _ => Err(TenthError::RuntimeError { line: None, col: None,
                        message: "只能通过可变引用赋值".into(),
                    }),
                }
            }

            HirExprKind::Closure { params, body, captures } => {
                // Capture the values of free variables from the current scope
                let captured_values: Vec<(String, Value)> = captures.iter()
                    .filter_map(|name| {
                        self.resolve_var(name).map(|v| (name.clone(), v))
                    })
                    .collect();
                Ok(Some(Value::Closure {
                    params: params.clone(),
                    body: Rc::new((**body).clone()),
                    captures: captured_values,
                }))
            }

            HirExprKind::Range { start, end, inclusive } => {
                let s = match start {
                    Some(e) => self.eval_expr(e)?.and_then(|v| v.as_int()).unwrap_or(0),
                    None => 0,
                };
                let e = match end {
                    Some(e) => self.eval_expr(e)?.and_then(|v| v.as_int()).unwrap_or(0),
                    None => 0,
                };
                Ok(Some(Value::Range { start: s, end: e, inclusive: *inclusive }))
            }

            HirExprKind::Ref(inner) => {
                let val = self.eval_expr(inner)?.ok_or_else(|| TenthError::RuntimeError { line: None, col: None,
                    message: "引用操作数为空值".into(),
                })?;
                Ok(Some(Value::Ref(Rc::new(RefCell::new(val)))))
            }

            HirExprKind::MutRef(inner) => {
                let val = self.eval_expr(inner)?.ok_or_else(|| TenthError::RuntimeError { line: None, col: None,
                    message: "可变引用操作数为空值".into(),
                })?;
                if let HirExprKind::Var(var_name) = &inner.kind {
                    // Reuse existing Shared wrapper, or create new one in current scope
                    let rc = if let Some(Value::Shared(existing)) = self.resolve_var(var_name) {
                        existing
                    } else {
                        let cell = Rc::new(RefCell::new(val));
                        self.insert_var(var_name.clone(), Value::Shared(cell.clone()));
                        cell
                    };
                    // Weak reference: does NOT contribute to Rc strong count.
                    // This prevents cycles when structs hold &mut references.
                    Ok(Some(Value::MutRef(Rc::downgrade(&rc))))
                } else {
                    // Non-variable &mut expr: fall back to immutable Ref
                    // (Weak would dangle immediately without an owner)
                    Ok(Some(Value::Ref(Rc::new(RefCell::new(val)))))
                }
            }

            HirExprKind::Deref(inner) => {
                let val = self.eval_expr(inner)?.ok_or_else(|| TenthError::RuntimeError { line: None, col: None,
                    message: "解引用操作数为空值".into(),
                })?;
                match &val {
                    Value::Ref(rc) => Ok(Some(rc.borrow().clone())),
                    Value::MutRef(weak) => {
                        let rc = weak.upgrade().ok_or_else(|| TenthError::RuntimeError { line: None, col: None,
                            message: "无法解引用悬垂的 &mut 引用".into(),
                        })?;
                        Ok(Some(rc.borrow().clone()))
                    }
                    _ => Err(TenthError::RuntimeError { line: None, col: None,
                        message: "无法解引用非引用值".into(),
                    }),
                }
            }

            HirExprKind::FieldAssign { target, field, value } => {
                let target_val = self.eval_expr(target)?.ok_or_else(|| TenthError::RuntimeError { line: None, col: None,
                    message: "字段赋值目标为空值".into(),
                })?;
                let rhs = self.eval_expr(value)?.ok_or_else(|| TenthError::RuntimeError { line: None, col: None,
                    message: "字段赋值值为空值".into(),
                })?;
                let target_var_name = if let HirExprKind::Var(vn) = &target.kind {
                    Some(vn.clone())
                } else {
                    None
                };
                match &target_val {
                    // Direct struct — mutate fields and update scope
                    Value::Struct { name: _, fields } => {
                        let mut found = false;
                        for (fname, fval) in fields.borrow_mut().iter_mut() {
                            if fname == field {
                                *fval = rhs.clone();
                                found = true;
                                break;
                            }
                        }
                        if !found {
                            return Err(TenthError::RuntimeError { line: Some(expr.span.line), col: Some(expr.span.col),
                                message: format!("结构体没有字段 '{}'", field),
                            });
                        }
                        // Update scope so subsequent reads see the change
                        if let Some(vn) = target_var_name {
                            // Re-read the struct from fields to get updated value
                            let updated = Value::Struct {
                                name: String::new(),
                                fields: fields.clone(),
                            };
                            self.insert_var(vn, updated);
                        }
                        return Ok(Some(Value::Unit));
                    }
                    // M1.2：Union 字段修改——只允许修改当前 active 字段（tagged union 语义）
                    Value::Union { name, active_field, .. } => {
                        if active_field != field {
                            return Err(TenthError::RuntimeError { line: Some(expr.span.line), col: Some(expr.span.col),
                                message: format!(
                                    "union '{}' 当前活跃字段是 '{}'，不能修改非活跃字段 '{}'",
                                    name, active_field, field
                                ),
                            });
                        }
                        let updated = Value::Union {
                            name: name.clone(),
                            active_field: active_field.clone(),
                            value: Box::new(rhs),
                        };
                        if let Some(vn) = target_var_name {
                            self.insert_var(vn, updated);
                        }
                        return Ok(Some(Value::Unit));
                    }
                    Value::MutRef(weak) => {
                        let rc = weak.upgrade().ok_or_else(|| TenthError::RuntimeError { line: None, col: None,
                            message: "无法通过悬垂的 &mut 引用赋值字段".into(),
                        })?;
                        let mut inner = rc.borrow_mut();
                        match &mut *inner {
                            Value::Struct { fields, .. } => {
                                for (fname, fval) in fields.borrow_mut().iter_mut() {
                                    if fname == field {
                                        *fval = rhs;
                                        return Ok(Some(Value::Unit));
                                    }
                                }
                                Err(TenthError::RuntimeError { line: None, col: None,
                                    message: format!("结构体没有字段 '{}'", field),
                                })
                            }
                            _ => Err(TenthError::RuntimeError { line: None, col: None,
                                message: "字段赋值仅支持结构体".into(),
                            }),
                        }
                    }
                    // Shared reference from Vec indexing: mutate through the RefCell
                    Value::Shared(rc) => {
                        let mut inner = rc.borrow_mut();
                        match &mut *inner {
                            Value::Struct { fields, .. } => {
                                for (fname, fval) in fields.borrow_mut().iter_mut() {
                                    if fname == field {
                                        *fval = rhs;
                                        return Ok(Some(Value::Unit));
                                    }
                                }
                                Err(TenthError::RuntimeError { line: None, col: None,
                                    message: format!("结构体没有字段 '{}'", field),
                                })
                            }
                            _ => Err(TenthError::RuntimeError { line: None, col: None,
                                message: "字段赋值仅支持结构体".into(),
                            }),
                        }
                    }
                    _ => Err(TenthError::RuntimeError { line: None, col: None,
                        message: "只能通过可变引用赋值字段".into(),
                    }),
                }
            }

            HirExprKind::Move(inner) => {
                let val = self.eval_expr(inner)?.ok_or_else(|| TenthError::RuntimeError { line: None, col: None,
                    message: "move 操作数为空值".into(),
                })?;
                if let HirExprKind::Var(var_name) = &inner.kind {
                    self.insert_var(var_name.clone(), Value::Moved);
                }
                Ok(Some(val))
            }

            HirExprKind::Lossy(inner) => {
                // `lossy` 是纯编译期构造（污点归零），运行时 no-op：直接求值 inner
                self.eval_expr(inner)
            }

            HirExprKind::TryBlock(inner) => {
                // `try { block }` — catch TryPropagate and wrap as Result::Err
                match self.eval_expr(inner) {
                    Ok(val) => {
                        // Success: wrap in Result::Ok
                        let inner_val = val.unwrap_or(Value::Unit);
                        Ok(Some(Value::Enum {
                            enum_name: "Result".to_string(),
                            variant: "Ok".to_string(),
                            fields: Rc::new(RefCell::new(vec![("_0".to_string(), inner_val)])),
                        }))
                    }
                    Err(TenthError::TryPropagate(err_val)) => {
                        // Caught a ? propagation: wrap in Result::Err
                        Ok(Some(Value::Enum {
                            enum_name: "Result".to_string(),
                            variant: "Err".to_string(),
                            fields: Rc::new(RefCell::new(vec![("_0".to_string(), err_val)])),
                        }))
                    }
                    Err(other) => Err(other),
                }
            }

            HirExprKind::Await(_) | HirExprKind::Spawn(_) => {
                return Err(TenthError::RuntimeError { line: Some(expr.span.line), col: Some(expr.span.col),
                    message: "async/await/spawn 不支持解释器路径，请使用 VM".into(),
                });
            }
            HirExprKind::Yield(_) => {
                return Err(TenthError::RuntimeError { line: Some(expr.span.line), col: Some(expr.span.col),
                    message: "yield 不支持解释器路径，请使用 VM".into(),
                });
            }
            HirExprKind::InterpolatedString { parts } => {
                let mut result = String::new();
                for p in parts {
                    match p {
                        crate::hir::hir::InterpPart::Literal(s) => {
                            result.push_str(s);
                        }
                        crate::hir::hir::InterpPart::Expr(name) => {
                            if let Some(val) = self.resolve_var(name) {
                                result.push_str(&self.value_to_string(&val));
                            } else {
                                result.push_str(name);
                            }
                        }
                    }
                }
                Ok(Some(Value::String(result)))
            }

            HirExprKind::Tuple(elems) => {
                if elems.is_empty() {
                    return Ok(Some(Value::Unit));
                }
                let mut values = Vec::new();
                for e in elems {
                    if let Some(v) = self.eval_expr(e)? {
                        values.push(v);
                    } else {
                        values.push(Value::Unit);
                    }
                }
                Ok(Some(Value::Tuple(values)))
            }

            HirExprKind::StructLiteral { name, fields, has_default: _ } => {
                let mut field_vals = Vec::new();
                for (fname, fexpr) in fields {
                    let v = self.eval_expr(fexpr)?.ok_or_else(|| TenthError::RuntimeError { line: None, col: None,
                        message: format!("结构体字段 '{}' 为空值", fname),
                    })?;
                    field_vals.push((fname.clone(), v));
                }
                // Note: when has_default is true, the lowerer has already filled in
                // default values for any missing fields in the HIR. The interpreter
                // just evaluates the complete field list as-is.
                Ok(Some(Value::Struct { name: name.clone(), fields: Rc::new(RefCell::new(field_vals)) }))
            }

            HirExprKind::UnionLiteral { name, active_field, value } => {
                let v = self.eval_expr(value)?.ok_or_else(|| TenthError::RuntimeError { line: None, col: None,
                    message: format!("union '{}' 活跃字段 '{}' 的值为空", name, active_field),
                })?;
                Ok(Some(Value::Union {
                    name: name.clone(),
                    active_field: active_field.clone(),
                    value: Box::new(v),
                }))
            }

            HirExprKind::EnumLiteral { enum_name, variant, fields } => {
                let mut field_vals = Vec::new();
                for (fname, fexpr) in fields {
                    let v = self.eval_expr(fexpr)?.ok_or_else(|| TenthError::RuntimeError { line: None, col: None,
                        message: format!("枚举字段 '{}' 为空值", fname),
                    })?;
                    field_vals.push((fname.clone(), v));
                }
                Ok(Some(Value::Enum {
                    enum_name: enum_name.clone(),
                    variant: variant.clone(),
                    fields: Rc::new(RefCell::new(field_vals)),
                }))
            }

            HirExprKind::Match { scrutinee, arms } => {
                let val = self.eval_expr(scrutinee)?.ok_or_else(|| TenthError::RuntimeError { line: None, col: None,
                    message: "match 表达式为空值".into(),
                })?;

                for arm in arms {
                    if self.pattern_matches(&arm.pattern, &val) {
                        // Bind variables from the matched pattern (needed for guard evaluation)
                        self.bind_pattern(&arm.pattern, &val);
                        // Check guard if present
                        if let Some(guard) = &arm.guard {
                            let guard_val = self.eval_expr(guard)?;
                            let guard_bool = match guard_val {
                                Some(Value::Bool(true)) => true,
                                Some(Value::Bool(false)) => false,
                                _ => false,
                            };
                            if !guard_bool {
                                // Clean up bindings and try next arm
                                self.unbind_pattern(&arm.pattern);
                                continue;
                            }
                        }
                        let result = self.eval_expr(&arm.body);
                        // Clean up bound variables
                        self.unbind_pattern(&arm.pattern);
                        return result;
                    }
                }
                Ok(Some(Value::Unit))
            }
        }.map_err(|e| Self::fill_span(e, &expr.span))
    }

    pub(super) fn eval_call(
        &mut self, func: &Value, args: &[Value], span: &crate::lexer::token::Span,
    ) -> TenthResult<Option<Value>> {
        match func {
            Value::FnRef { name, .. } => {
                self.call_named_fn(name, args, span)
            }
            Value::Enum { enum_name, variant, fields: _ } => {
                // Enum variant used as constructor: Result::Ok(42) → Enum with fields
                let field_vals: Vec<(String, Value)> = args.iter().enumerate()
                    .map(|(i, v)| (format!("_{}", i), v.clone()))
                    .collect();
                Ok(Some(Value::Enum {
                    enum_name: enum_name.clone(),
                    variant: variant.clone(),
                    fields: Rc::new(RefCell::new(field_vals)),
                }))
            }
            Value::Closure { params, body, captures } => {
                self.push_scope();

                for ((pname, _), arg) in params.iter().zip(args.iter()) {
                    self.insert_var(pname.clone(), arg.clone());
                }

                for (cap_name, cap_val) in captures {
                    self.insert_var(cap_name.clone(), cap_val.clone());
                }

                let result = self.eval_expr(body);

                self.pop_scope();

                result
            }
            _ => Err(TenthError::RuntimeError { line: None, col: None,
                message: "不是可调用值".into(),
            }),
        }
    }

    pub(super) fn eval_stmt(&mut self, stmt: &HirStmt) -> TenthResult<()> {
        self.tick()?;
        match &stmt.kind {
            HirStmtKind::Expr(e) => {
                self.eval_expr(e)?;
                Ok(())
            }
            HirStmtKind::Let { names, init, .. } => {
                let val = match init {
                    Some(e) => self.eval_expr(e)?.unwrap_or(Value::Unit),
                    None => Value::Unit,
                };
                if names.len() == 1 {
                    self.insert_var(names[0].clone(), val);
                } else {
                    // Tuple destructuring: val should be a Vec / Array / Tuple
                    match &val {
                        Value::Vec(v) => {
                            let borrowed = v.borrow();
                            for (i, name) in names.iter().enumerate() {
                                let v = borrowed.get(i).cloned().unwrap_or(Value::Unit);
                                self.insert_var(name.clone(), v);
                            }
                        }
                        Value::Array(a) => {
                            let borrowed = a.borrow();
                            for (i, name) in names.iter().enumerate() {
                                let v = borrowed.get(i).cloned().unwrap_or(Value::Unit);
                                self.insert_var(name.clone(), v);
                            }
                        }
                        Value::Tuple(items) => {
                            for (i, name) in names.iter().enumerate() {
                                let v = items.get(i).cloned().unwrap_or(Value::Unit);
                                self.insert_var(name.clone(), v);
                            }
                        }
                        _ => {
                            // Fallback: assign entire value to first name
                            self.insert_var(names[0].clone(), val);
                        }
                    }
                }
                Ok(())
            }
            HirStmtKind::Return(expr) => {
                let val = match expr {
                    Some(e) => self.eval_expr(e)?.unwrap_or(Value::Unit),
                    None => Value::Unit,
                };
                Err(TenthError::ReturnValue(val))
            }
            HirStmtKind::Break { label, value } => {
                // If break has a value, evaluate it first (result is discarded since
                // loops in interpreter don't produce values).
                if let Some(e) = value {
                    self.eval_expr(e)?;
                }
                // M2.3：带标签的 break 用 LabeledBreakSignal 携带目标标签向上传播
                match label {
                    Some(l) => Err(TenthError::LabeledBreakSignal(l.clone())),
                    None => Err(TenthError::BreakSignal),
                }
            }
            HirStmtKind::Continue { label } => {
                // M2.3：带标签的 continue 用 LabeledContinueSignal 携带目标标签向上传播
                match label {
                    Some(l) => Err(TenthError::LabeledContinueSignal(l.clone())),
                    None => Err(TenthError::ContinueSignal),
                }
            }
            HirStmtKind::Loop { label, body } => {
                loop {
                    let mut should_break = false;
                    for s in body {
                        match self.eval_stmt(s) {
                            Ok(()) => {}
                            Err(e) => match self.classify_loop_signal(&label, e) {
                                LoopSignal::Break => { should_break = true; break; }
                                LoopSignal::Continue => continue,
                                LoopSignal::Propagate(e) => return Err(e),
                            },
                        }
                    }
                    if should_break { break; }
                }
                Ok(())
            }
            HirStmtKind::DoWhile { label, body, cond } => {
                // do-while: execute body once, then check condition
                loop {
                    match self.eval_stmt(body) {
                        Ok(()) => {}
                        Err(e) => match self.classify_loop_signal(&label, e) {
                            LoopSignal::Break => break,
                            LoopSignal::Continue => continue,
                            LoopSignal::Propagate(e) => return Err(e),
                        },
                    }
                    let c = self.eval_expr(cond)?.ok_or_else(|| TenthError::RuntimeError { line: Some(stmt.span.line), col: Some(stmt.span.col),
                        message: "do-while 条件为空值".into(),
                    })?;
                    if !c.is_truthy() {
                        break;
                    }
                }
                Ok(())
            }
            HirStmtKind::While { label, cond, body } => {
                loop {
                    let c = self.eval_expr(cond)?.ok_or_else(|| TenthError::RuntimeError { line: Some(stmt.span.line), col: Some(stmt.span.col),
                        message: "while 条件为空值".into(),
                    })?;
                    if !c.is_truthy() {
                        break;
                    }
                    match self.eval_stmt(body) {
                        Ok(()) => {}
                        Err(e) => match self.classify_loop_signal(&label, e) {
                            LoopSignal::Break => break,
                            LoopSignal::Continue => continue,
                            LoopSignal::Propagate(e) => return Err(e),
                        },
                    }
                }
                Ok(())
            }
            HirStmtKind::For { label, var, iter, body } => {
                let iter_val = self.eval_expr(iter)?.ok_or_else(|| TenthError::RuntimeError { line: Some(stmt.span.line), col: Some(stmt.span.col),
                    message: "for 迭代对象为空值".into(),
                })?;
                match iter_val {
                    Value::Range { start, end, inclusive } => {
                        let e = if inclusive { end + 1 } else { end };
                        for i in start..e {
                            self.insert_var(var.clone(), Value::Int(i, BaseType::I32));
                            match self.eval_stmt(body) {
                                Ok(()) => {}
                                Err(e) => match self.classify_loop_signal(&label, e) {
                                    LoopSignal::Break => break,
                                    LoopSignal::Continue => continue,
                                    LoopSignal::Propagate(e) => return Err(e),
                                },
                            }
                        }
                    }
                    Value::Vec(items) => {
                        let vec = items.borrow();
                        for item in vec.iter() {
                            let val = match item {
                                Value::Shared(rc) => rc.borrow().clone(),
                                other => other.clone(),
                            };
                            self.insert_var(var.clone(), val);
                            match self.eval_stmt(body) {
                                Ok(()) => {}
                                Err(e) => match self.classify_loop_signal(&label, e) {
                                    LoopSignal::Break => break,
                                    LoopSignal::Continue => continue,
                                    LoopSignal::Propagate(e) => return Err(e),
                                },
                            }
                        }
                    }
                    Value::Tensor(t) => {
                        let tensor = t.borrow();
                        let shape = tensor.shape();
                        let n = shape.first().copied().unwrap_or(0);
                        let row_size: usize = shape[1..].iter().product();
                        let flat = tensor.data.as_standard_layout().to_owned();
                        let slice = flat.as_slice().unwrap_or(&[]);
                        for i in 0..n {
                            let start = i * row_size;
                            let end = (start + row_size).min(slice.len());
                            let row_data = slice[start..end].to_vec();
                            let row_shape = if shape.len() > 1 {
                                shape[1..].to_vec()
                            } else {
                                vec![1]
                            };
                            let row_tensor = Tensor::from_vec(row_data, row_shape);
                            let val = Value::Tensor(Rc::new(RefCell::new(row_tensor)));
                            self.insert_var(var.clone(), val);
                            match self.eval_stmt(body) {
                                Ok(()) => {}
                                Err(e) => match self.classify_loop_signal(&label, e) {
                                    LoopSignal::Break => break,
                                    LoopSignal::Continue => continue,
                                    LoopSignal::Propagate(e) => return Err(e),
                                },
                            }
                        }
                    }
                    Value::Iterator(lazy_iter) => {
                        // Collect the iterator and iterate over the result
                        let collected = self.eval_iterator_method(&lazy_iter, "collect", &[])?;
                        if let Some(Value::Vec(items)) = collected {
                            let vec = items.borrow();
                            for item in vec.iter() {
                                let val = match item {
                                    Value::Shared(rc) => rc.borrow().clone(),
                                    other => other.clone(),
                                };
                                self.insert_var(var.clone(), val);
                                match self.eval_stmt(body) {
                                    Ok(()) => {}
                                    Err(e) => match self.classify_loop_signal(&label, e) {
                                        LoopSignal::Break => break,
                                        LoopSignal::Continue => continue,
                                        LoopSignal::Propagate(e) => return Err(e),
                                    },
                                }
                            }
                        }
                    }
                    _ => {
                        return Err(TenthError::RuntimeError { line: Some(stmt.span.line), col: Some(stmt.span.col),
                            message: "for 循环支持 range、Vec、张量和迭代器".into(),
                        });
                    }
                }
                Ok(())
            }
        }.map_err(|e| Self::fill_span(e, &stmt.span))
    }
}

/// M2.3：循环体执行结果的信号分类——break / continue 是否为作用于当前循环的信号。
enum LoopSignal {
    /// break 信号（无标签，或标签匹配当前循环）——退出当前循环
    Break,
    /// continue 信号（无标签，或标签匹配当前循环）——跳到当前循环继续点
    Continue,
    /// 其他（普通完成 / 非当前循环的标签信号 / 其他错误）——向上传播
    Propagate(TenthError),
}

impl super::Interpreter {
    /// M2.3：分类循环体执行结果。
    /// - 无标签 BreakSignal/ContinueSignal → 作用于最近循环（当前循环）
    /// - LabeledBreakSignal(l)/LabeledContinueSignal(l) 且 l == 当前循环标签 → 作用于当前循环
    /// - 标签不匹配 → 向上传播（逐层退出直到标签匹配的循环）
    fn classify_loop_signal(&self, label: &Option<String>, e: TenthError) -> LoopSignal {
        match e {
            TenthError::BreakSignal => LoopSignal::Break,
            TenthError::ContinueSignal => LoopSignal::Continue,
            TenthError::LabeledBreakSignal(l) if label.as_ref() == Some(&l) => LoopSignal::Break,
            TenthError::LabeledContinueSignal(l) if label.as_ref() == Some(&l) => LoopSignal::Continue,
            other => LoopSignal::Propagate(other),
        }
    }
}

impl super::Interpreter {
    /// 问题12：补全 RuntimeError 的源码位置。
    /// 若 RuntimeError 已有 line/col（手动填充的），则保留；
    /// 否则用当前表达式/语句的 span 填充。
    fn fill_span(e: TenthError, span: &crate::lexer::token::Span) -> TenthError {
        if let TenthError::RuntimeError { line, col, message } = e {
            TenthError::RuntimeError {
                line: line.or(Some(span.line)),
                col: col.or(Some(span.col)),
                message,
            }
        } else {
            e
        }
    }
}
