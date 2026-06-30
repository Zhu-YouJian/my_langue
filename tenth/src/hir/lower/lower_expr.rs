use std::collections::HashMap;
use crate::error::{TenthError, TenthResult};
use crate::parser::ast as ast;
use crate::hir::hir::*;
use crate::hir::types::*;
use super::Scope;
use super::Ownership;
use super::{substitute_type, lower_binop};
use super::Lowerer;

impl Lowerer {
    pub(super) fn lower_expr(&mut self, expr: &ast::Expr) -> TenthResult<HirExpr> {
        use ast::ExprKind;

        let span = expr.span.clone();

        let (kind, ty) = match &expr.kind {
            ExprKind::Literal(lit) => {
                let (hir_lit, ty) = match lit {
                    ast::Literal::Int(n) => (Literal::Int(*n), Type::i32()),
                    ast::Literal::Float(n, dt) => (Literal::Float(*n, *dt), Type::Base(*dt)),
                    ast::Literal::Bool(b) => (Literal::Bool(*b), Type::bool_()),
                    ast::Literal::String(s) => (Literal::String(s.clone()), Type::str_()),
                };
                (HirExprKind::Literal(hir_lit), ty)
            }

            ExprKind::Ident(ident) => {
                if ident.name.contains("::") {
                    let parts: Vec<&str> = ident.name.splitn(2, "::").collect();
                    if parts.len() == 2 {
                        let enum_name = parts[0];
                        let variant = parts[1];
                        if let Some(variants) = self.enums.get(enum_name) {
                            if variants.iter().any(|(v, _)| v == variant) {
                                return Ok(HirExpr {
                                    kind: HirExprKind::EnumLiteral {
                                        enum_name: enum_name.to_string(),
                                        variant: variant.to_string(),
                                        fields: Vec::new(),
                                    },
                                    ty: Type::Enum(enum_name.to_string()),
                                    span,
                                });
                            }
                        }
                    }
                    (HirExprKind::Var(ident.name.clone()), Type::Unknown)
                } else {
                    self.scope.check_use(&ident.name, &ident.span)?;
                    let var_info = self.scope.lookup_var(&ident.name);
                    let fn_info = self.scope.lookup_fn(&ident.name);
                    if var_info.is_none() && fn_info.is_none() {
                        match ident.name.as_str() {
                            "println" | "eprintln" | "tensor" | "rand" | "randn" | "randn_f32" | "rand_f32" | "zeros_f32" | "ones_f32"
                            | "read_file" | "write_file" | "write_bytes" | "read_bytes"
                            | "str_at" | "str_len" | "str_cmp" | "str_slice" | "str_add" | "str_eq" | "str_int"
                            | "Vec::new" | "HashMap::new"
                            | "compile_host" | "compile_program"
                            | "start_grad" | "new_grad" | "stop_grad"
                            | "param" | "backward" | "grad" | "zero_grad"
                            | "cross_entropy"
                            | "abs" | "sqrt" | "sin" | "cos" | "ln" | "pow" | "to_float" | "to_f32" | "to_f64" | "tensor_from_vec"
                            | "f64_bits" | "f64_from_bits"
                            | "zeros" | "ones"
                            | "save_weights" | "load_weights"
                            | "format" | "parse_int" | "parse_float"
                            | "to_string" | "type_name"
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
                            | "lexer_new" | "lexer_tokenize" | "parse_program"
                            | "lower_program" | "compile_to_wasm" => {
                                (HirExprKind::Var(ident.name.clone()), Type::Unknown)
                            }
                            _ => {
                                return Err(TenthError::TypeError {
                                    line: span.line,
                                    col: span.col,
                                    message: format!("undefined variable '{}'", ident.name),
                                });
                            }
                        }
                    } else {
                        let ty = var_info.map(|v| v.0).or_else(|| {
                            fn_info.map(|f| Type::FnType {
                                params: f.0.iter().map(|(_, t)| t.clone()).collect(),
                                ret: Box::new(f.1),
                            })
                        }).unwrap_or(Type::Unknown);
                        (HirExprKind::Var(ident.name.clone()), ty)
                    }
                }
            }

            ExprKind::Binary { op, left, right } => {
                let l = self.lower_expr(left)?;
                let r = self.lower_expr(right)?;
                let ty = self.infer_binary_type(op, &l.ty, &r.ty);
                let hir_op = lower_binop(op);
                (HirExprKind::Binary { op: hir_op, left: Box::new(l), right: Box::new(r), ty: ty.clone() }, ty)
            }

            ExprKind::Unary { op, expr: inner } => {
                let e = self.lower_expr(inner)?;
                let ty = e.ty.clone();
                let hir_op = match op {
                    ast::UnaryOp::Neg => UnaryOp::Neg,
                    ast::UnaryOp::Not => UnaryOp::Not,
                    ast::UnaryOp::Try => UnaryOp::Try,
                };
                (HirExprKind::Unary { op: hir_op, expr: Box::new(e), ty: ty.clone() }, ty)
            }

            ExprKind::Call { func, args } => {
                let f = self.lower_expr(func)?;
                let lowered_args: Vec<_> = args.iter()
                    .map(|a| self.lower_expr(a))
                    .collect::<TenthResult<_>>()?;

                // If the func is an EnumLiteral, merge args as tuple fields
                if let HirExprKind::EnumLiteral { enum_name, variant, fields } = &f.kind {
                    if fields.is_empty() && !lowered_args.is_empty() {
                        let tuple_fields: Vec<(String, HirExpr)> = lowered_args.into_iter().enumerate()
                            .map(|(i, a)| (format!("_{}", i), a))
                            .collect();
                        return Ok(HirExpr {
                            kind: HirExprKind::EnumLiteral {
                                enum_name: enum_name.clone(),
                                variant: variant.clone(),
                                fields: tuple_fields,
                            },
                            ty: Type::Unknown,
                            span,
                        });
                    }
                }

                let ret_ty = self.resolve_call_type(&f, &lowered_args, &span)?;

                (HirExprKind::Call {
                    func: Box::new(f),
                    args: lowered_args,
                    ret_ty: ret_ty.clone(),
                }, ret_ty)
            }

            ExprKind::GenericCall { func, generics, args } => {
                let func_name = match &func.kind {
                    ExprKind::Ident(ident) => ident.name.clone(),
                    _ => {
                        return Err(TenthError::TypeError {
                            line: span.line,
                            col: span.col,
                            message: "泛型调用的目标必须是具名函数".into(),
                        });
                    }
                };

                let template = self.generic_funcs.get(&func_name)
                    .ok_or_else(|| TenthError::TypeError {
                        line: span.line,
                        col: span.col,
                        message: format!("未定义的泛型函数 '{}'", func_name),
                    })?
                    .clone();

                let type_args: Vec<Type> = generics.iter()
                    .map(|ta| Type::from_annotation(ta))
                    .collect();

                let mut type_map: HashMap<String, Type> = HashMap::new();
                for (i, gen_name) in template.generics.iter().enumerate() {
                    type_map.insert(gen_name.clone(), type_args.get(i).cloned().unwrap_or(Type::Unknown));
                }

                let inst_ret_ty = substitute_type(&template.return_type, &type_map);

                let lowered_args: Vec<HirExpr> = args.iter()
                    .map(|a| self.lower_expr(a))
                    .collect::<TenthResult<_>>()?;

                // Generate mangled name and instantiate if not already done
                let mangled_name: String = type_args.iter()
                    .fold(func_name.clone(), |acc, ty| format!("{}_{}", acc, ty));
                let already_instantiated = self.functions.iter().any(|f| f.name == mangled_name);
                if !already_instantiated {
                    let inst_params: Vec<(String, Type)> = template.params.iter()
                        .map(|(n, t)| (n.clone(), substitute_type(t, &type_map)))
                        .collect();
                    let inst_fn = HirFnDef {
                        name: mangled_name.clone(),
                        params: inst_params,
                        return_type: inst_ret_ty.clone(),
                        body: template.body.clone(),
                        generics: vec![],
                        generics_bounds: std::collections::HashMap::new(),
                        span: template.span.clone(),
                    };
                    self.functions.push(inst_fn);
                }

                // Generate a regular Call to the mangled function name
                let call_func = HirExpr {
                    kind: HirExprKind::Var(mangled_name),
                    ty: Type::Unknown,
                    span: span.clone(),
                };
                (HirExprKind::Call {
                    func: Box::new(call_func),
                    args: lowered_args,
                    ret_ty: inst_ret_ty.clone(),
                }, inst_ret_ty)
            }

            ExprKind::MethodCall { receiver, method, args } => {
                let recv = self.lower_expr(receiver)?;
                let lowered_args: Vec<_> = args.iter()
                    .map(|a| self.lower_expr(a))
                    .collect::<TenthResult<_>>()?;

                // Try user-defined method resolution (inherent impl).
                // If the receiver is a struct type and a mangled function
                // __<Type>_<method> exists, rewrite to a regular Call so the
                // WASM backend can compile it without special method support.
                let recv_type_name = match &recv.ty {
                    Type::Struct(name) | Type::TypeParam { name } => Some(name.clone()),
                    _ => None,
                };
                if let Some(type_name) = recv_type_name {
                    let mangled = format!("__{}_{}", type_name, method.name);
                    if self.functions.iter().any(|f| f.name == mangled) {
                        let mut all_args = vec![recv.clone()];
                        all_args.extend(lowered_args.clone());
                        let ret_ty = self.resolve_method_type(&recv.ty, &method.name, &all_args);
                        let func = HirExpr {
                            kind: HirExprKind::Var(mangled),
                            ty: Type::Unknown,
                            span: expr.span.clone(),
                        };
                        return Ok(HirExpr {
                            kind: HirExprKind::Call {
                                func: Box::new(func),
                                args: all_args,
                                ret_ty: ret_ty.clone(),
                            },
                            ty: ret_ty,
                            span: expr.span.clone(),
                        });
                    }
                }

                let ret_ty = self.resolve_method_type(&recv.ty, &method.name, &lowered_args);

                (HirExprKind::MethodCall {
                    receiver: Box::new(recv),
                    method: method.name.clone(),
                    args: lowered_args,
                    ret_ty: ret_ty.clone(),
                }, ret_ty)
            }

            ExprKind::Index { target, indices } => {
                let t = self.lower_expr(target)?;
                let lowered_indices: Vec<_> = indices.iter()
                    .map(|idx| self.lower_index(idx))
                    .collect::<TenthResult<_>>()?;

                let ty = self.index_type(&t.ty, &lowered_indices);
                (HirExprKind::Index { target: Box::new(t), indices: lowered_indices }, ty)
            }

            ExprKind::Field { target, field } => {
                let t = self.lower_expr(target)?;
                // Unwrap reference types to get the inner struct type
                let inner_ty = match &t.ty {
                    Type::Ref(inner) | Type::MutRef(inner) => inner.as_ref(),
                    other => other,
                };
                let field_ty = match inner_ty {
                    Type::Struct(name) | Type::TypeParam { name } => {
                        self.structs.get(name)
                            .and_then(|fields| fields.iter().find(|(n, _)| n == &field.name))
                            .map(|(_, ty)| ty.clone())
                            .unwrap_or(Type::Unknown)
                    }
                    _ => Type::Unknown,
                };
                (HirExprKind::Field { target: Box::new(t), field: field.name.clone() }, field_ty)
            }

            ExprKind::TensorLiteral(data) => {
                let lowered: Vec<Vec<HirExpr>> = data.iter()
                    .map(|row| row.iter().map(|e| self.lower_expr(e)).collect())
                    .collect::<TenthResult<_>>()?;
                let rows = lowered.len() as i64;
                let cols = lowered.first().map_or(0, |r| r.len() as i64);
                // 按元素字面量 dtype 推断 Tensor dtype：任一元素为 F32 → F32，否则 F64
                let dtype = lowered.iter().flatten().find_map(|e| {
                    if matches!(e.ty, Type::Base(BaseType::F32)) { Some(BaseType::F32) } else { None }
                }).unwrap_or(BaseType::F64);
                let ty = Type::tensor(dtype, vec![Dim::Known(rows), Dim::Known(cols)]);
                (HirExprKind::TensorLiteral { data: lowered, ty: ty.clone() }, ty)
            }

            ExprKind::ArrayLiteral(elements) => {
                let lowered: Vec<HirExpr> = elements.iter()
                    .map(|e| self.lower_expr(e))
                    .collect::<TenthResult<_>>()?;
                let elem_ty = lowered.first()
                    .map(|e| e.ty.clone())
                    .unwrap_or(Type::Unknown);
                let ty = Type::Array(Box::new(elem_ty));
                (HirExprKind::ArrayLiteral { elements: lowered, ty: ty.clone() }, ty)
            }

            ExprKind::Range { start, end, inclusive } => {
                let s = start.as_ref().map(|e| self.lower_expr(e)).transpose()?;
                let e = end.as_ref().map(|e| self.lower_expr(e)).transpose()?;
                let ty = s.as_ref()
                    .or(e.as_ref())
                    .map(|expr| expr.ty.clone())
                    .unwrap_or(Type::i32());
                (HirExprKind::Range { start: s.map(Box::new), end: e.map(Box::new), inclusive: *inclusive }, ty)
            }

            ExprKind::If { cond, then_branch, else_branch } => {
                let c = self.lower_expr(cond)?;
                // Release borrows from the condition so the body can reborrow.
                // Without this, `if peek(&p).disc == 54 { advance(&mut p); }` fails
                // because the condition's shared borrow of `p` persists into the body.
                self.scope.release_borrows();
                let t = self.lower_expr(then_branch)?;
                self.scope.release_borrows();
                let e = else_branch.as_ref().map(|eb| self.lower_expr(eb)).transpose()?;
                self.scope.release_borrows();
                let ty = if let Some(ref eb) = e {
                    eb.ty.clone()
                } else {
                    Type::unit()
                };
                (HirExprKind::If { cond: Box::new(c), then_branch: Box::new(t), else_branch: e.map(Box::new), ty: ty.clone() }, ty)
            }

            ExprKind::Block(stmts) => {
                let inner_scope = Scope::with_parent(std::mem::replace(&mut self.scope, Scope::new()));
                self.scope = inner_scope;

                let mut lowered_stmts: Vec<HirStmt> = Vec::new();
                for s in stmts {
                    let lowered = self.lower_stmt(s)?;
                    // Release borrows after each statement, unless the statement
                    // creates a persistent borrow (e.g., `let r = &x;`).
                    if !Self::creates_persistent_borrow(s) {
                        self.scope.release_borrows();
                    }
                    lowered_stmts.push(lowered);
                }

                let final_expr = lowered_stmts.last().and_then(|s| match &s.kind {
                    HirStmtKind::Expr(e) => Some(e.clone()),
                    _ => None,
                });

                let ty = final_expr.as_ref().map(|e| e.ty.clone()).unwrap_or(Type::unit());

                let stmts_without_final: Vec<HirStmt> = if final_expr.is_some() {
                    lowered_stmts[..lowered_stmts.len().saturating_sub(1)].to_vec()
                } else {
                    lowered_stmts
                };

                let outer_scope = std::mem::replace(&mut self.scope, Scope::new());
                self.scope = outer_scope;

                (HirExprKind::Block { stmts: stmts_without_final, final_expr: final_expr.map(Box::new) }, ty)
            }

            ExprKind::Closure { params, body } => {
                let lowered_params: Vec<_> = params.iter()
                    .map(|(name, ann)| {
                        let ty = ann.as_ref()
                            .map(|a| Type::from_annotation(a))
                            .unwrap_or(Type::Unknown);
                        (name.name.clone(), ty)
                    })
                    .collect();

                // Create closure scope with params bound
                let closure_scope = Scope::with_parent(std::mem::replace(&mut self.scope, Scope::new()));
                self.scope = closure_scope;
                for (name, ty) in &lowered_params {
                    self.scope.define_var(name.clone(), ty.clone(), false);
                }

                let b = self.lower_expr(body)?;

                let outer_scope = std::mem::replace(&mut self.scope, Scope::new());
                self.scope = outer_scope;

                // Analyze free variables in the closure body (excluding params)
                let captures = Self::free_vars_in(&b);

                let closure_ty = Type::FnType {
                    params: lowered_params.iter().map(|(_, t)| t.clone()).collect(),
                    ret: Box::new(b.ty.clone()),
                };
                (HirExprKind::Closure { params: lowered_params, body: Box::new(b), captures }, closure_ty)
            }

            ExprKind::Assign { target, value } => {
                let v = self.lower_expr(value)?;
                match &target.kind {
                    ExprKind::Ident(id) => {
                        let name = id.name.clone();
                        self.scope.define_var(name.clone(), v.ty.clone(), true);
                        (HirExprKind::Assign { target: name, value: Box::new(v) }, Type::unit())
                    }
                    ExprKind::Deref(inner) => {
                        let inner_hir = self.lower_expr(inner)?;
                        (HirExprKind::DerefAssign { target: Box::new(inner_hir), value: Box::new(v) }, Type::unit())
                    }
                    ExprKind::Field { target: field_target, field } => {
                        let inner_hir = self.lower_expr(field_target)?;
                        (HirExprKind::FieldAssign {
                            target: Box::new(inner_hir),
                            field: field.name.clone(),
                            value: Box::new(v),
                        }, Type::unit())
                    }
                    _ => {
                        return Err(TenthError::ParseError {
                            line: span.line,
                            col: span.col,
                            message: "invalid assignment target".into(),
                        });
                    }
                }
            }

            ExprKind::AssignOp { target, op, value } => {
                let v = self.lower_expr(value)?;
                match &target.kind {
                    ExprKind::Ident(id) => {
                        let name = id.name.clone();
                        let hir_op = lower_binop(op);
                        (HirExprKind::AssignOp { target: name, op: hir_op, value: Box::new(v) }, Type::unit())
                    }
                    ExprKind::Deref(inner) => {
                        let inner_hir = self.lower_expr(inner)?;
                        let hir_op = lower_binop(op);
                        (HirExprKind::DerefAssignOp { target: Box::new(inner_hir), op: hir_op, value: Box::new(v) }, Type::unit())
                    }
                    _ => return Err(TenthError::ParseError {
                        line: span.line,
                        col: span.col,
                        message: "invalid assignment target".into(),
                    }),
                }
            }

            ExprKind::StructLiteral { name, generics: _, fields, use_defaults } => {
                let mut lowered_fields: Vec<(String, HirExpr)> = fields.iter()
                    .map(|(id, e)| {
                        let lowered = self.lower_expr(e)?;
                        Ok((id.name.clone(), lowered))
                    })
                    .collect::<TenthResult<_>>()?;

                if *use_defaults {
                    // Fill missing fields with default values based on type
                    let field_names: Vec<String> = lowered_fields.iter().map(|(n, _)| n.clone()).collect();
                    if let Some(struct_fields) = self.structs.get(&name.name) {
                        for (fname, fty) in struct_fields {
                            if !field_names.contains(fname) {
                                let default_val = match fty {
                                    Type::Base(b) => match b {
                                        BaseType::I32 | BaseType::I64 | BaseType::I8 | BaseType::I16
                                        | BaseType::U8 | BaseType::U16 | BaseType::U32 | BaseType::U64 => HirExpr {
                                            kind: HirExprKind::Literal(Literal::Int(0)),
                                            ty: fty.clone(),
                                            span: name.span.clone(),
                                        },
                                        BaseType::F64 | BaseType::F32 | BaseType::F16 | BaseType::BF16 => HirExpr {
                                            kind: HirExprKind::Literal(Literal::Float(0.0, BaseType::F64)),
                                            ty: fty.clone(),
                                            span: name.span.clone(),
                                        },
                                        BaseType::Bool => HirExpr {
                                            kind: HirExprKind::Literal(Literal::Bool(false)),
                                            ty: fty.clone(),
                                            span: name.span.clone(),
                                        },
                                        BaseType::Str => HirExpr {
                                            kind: HirExprKind::Literal(Literal::String(String::new())),
                                            ty: fty.clone(),
                                            span: name.span.clone(),
                                        },
                                        BaseType::Char => HirExpr {
                                            kind: HirExprKind::Literal(Literal::Int(0)),
                                            ty: fty.clone(),
                                            span: name.span.clone(),
                                        },
                                        BaseType::Unit => HirExpr {
                                            kind: HirExprKind::Literal(Literal::Int(0)),
                                            ty: Type::unit(),
                                            span: name.span.clone(),
                                        },
                                    },
                                    _ => HirExpr {
                                        kind: HirExprKind::Literal(Literal::Int(0)),
                                        ty: Type::Unknown,
                                        span: name.span.clone(),
                                    },
                                };
                                lowered_fields.push((fname.clone(), default_val));
                            }
                        }
                    }
                }

                let struct_ty = Type::from_annotation(&ast::TypeAnnotation::Named(ast::Ident { name: name.name.clone(), span: name.span.clone() }));
                (HirExprKind::StructLiteral {
                    name: name.name.clone(),
                    fields: lowered_fields,
                    has_default: *use_defaults,
                }, struct_ty)
            }

            ExprKind::EnumLiteral { enum_name, variant, fields } => {
                let lowered_fields: Vec<(String, HirExpr)> = fields.iter()
                    .map(|(id, e)| {
                        let lowered = self.lower_expr(e)?;
                        Ok((id.name.clone(), lowered))
                    })
                    .collect::<TenthResult<_>>()?;
                (HirExprKind::EnumLiteral {
                    enum_name: enum_name.name.clone(),
                    variant: variant.name.clone(),
                    fields: lowered_fields,
                }, Type::Enum(enum_name.name.clone()))
            }

            ExprKind::Match { scrutinee, arms } => {
                let lowered_scrutinee = self.lower_expr(scrutinee)?;
                // Release borrows from the scrutinee so arms can reborrow.
                self.scope.release_borrows();
                let lowered_arms: Vec<HirMatchArm> = arms.iter()
                    .map(|arm| {
                        let hir_pattern = self.lower_pattern(&arm.pattern)?;

                        let arm_scope = Scope::with_parent(std::mem::replace(&mut self.scope, Scope::new()));
                        self.scope = arm_scope;

                        // Bind variables from pattern
                        self.bind_pattern_vars(&hir_pattern, &lowered_scrutinee.ty);

                        // Lower guard if present
                        let guard = arm.guard.as_ref()
                            .map(|g| self.lower_expr(g))
                            .transpose()?;

                        let body = self.lower_expr(&arm.body)?;

                        let outer_scope = std::mem::replace(&mut self.scope, Scope::new());
                        self.scope = outer_scope;

                        Ok(HirMatchArm { pattern: hir_pattern, guard, body })
                    })
                    .collect::<TenthResult<_>>()?;
                // Infer match type from first non-Unknown arm, falling back to first arm
                let match_ty = lowered_arms.iter()
                    .map(|arm| arm.body.ty.clone())
                    .find(|ty| !matches!(ty, Type::Unknown))
                    .or_else(|| lowered_arms.first().map(|arm| arm.body.ty.clone()))
                    .unwrap_or(Type::Unknown);
                (HirExprKind::Match {
                    scrutinee: Box::new(lowered_scrutinee),
                    arms: lowered_arms,
                }, match_ty)
            }

            ExprKind::Ref(inner) => {
                if let ExprKind::Ident(ident) = &inner.kind {
                    self.scope.check_borrow_shared(&ident.name, &ident.span)?;
                }
                let e = self.lower_expr(inner)?;
                let ty = Type::Ref(Box::new(e.ty.clone()));
                if let ExprKind::Ident(ident) = &inner.kind {
                    let count = match self.scope.get_ownership(&ident.name) {
                        Some(Ownership::SharedRef(n)) => n + 1,
                        _ => 1,
                    };
                    self.scope.set_ownership(&ident.name, Ownership::SharedRef(count));
                }
                (HirExprKind::Ref(Box::new(e)), ty)
            }

            ExprKind::MutRef(inner) => {
                if let ExprKind::Ident(ident) = &inner.kind {
                    self.scope.check_borrow_mut(&ident.name, &ident.span)?;
                }
                let e = self.lower_expr(inner)?;
                let ty = Type::MutRef(Box::new(e.ty.clone()));
                if let ExprKind::Ident(ident) = &inner.kind {
                    self.scope.set_ownership(&ident.name, Ownership::ExclusiveRef);
                }
                (HirExprKind::MutRef(Box::new(e)), ty)
            }

            ExprKind::Deref(inner) => {
                let e = self.lower_expr(inner)?;
                let inner_ty = match &e.ty {
                    Type::Ref(t) | Type::MutRef(t) => (**t).clone(),
                    _ => Type::Unknown,
                };
                (HirExprKind::Deref(Box::new(e)), inner_ty)
            }

            ExprKind::Move(inner) => {
                let e = self.lower_expr(inner)?;
                let ty = e.ty.clone();
                if let ExprKind::Ident(ident) = &inner.kind {
                    self.scope.set_ownership(&ident.name, Ownership::Moved);
                }
                (HirExprKind::Move(Box::new(e)), ty)
            }

            ExprKind::TryBlock(inner) => {
                let e = self.lower_expr(inner)?;
                let result_ty = Type::Generic {
                    base: Box::new(Type::Enum("Result".to_string())),
                    args: vec![e.ty.clone(), Type::str_()],
                };
                (HirExprKind::TryBlock(Box::new(e)), result_ty)
            }

            ExprKind::InterpolatedString(parts) => {
                let hir_parts: Vec<crate::hir::hir::InterpPart> = parts.iter().map(|p| match p {
                    ast::InterpPart::Literal(s) => crate::hir::hir::InterpPart::Literal(s.clone()),
                    ast::InterpPart::Expr(e) => crate::hir::hir::InterpPart::Expr(e.clone()),
                }).collect();
                (HirExprKind::InterpolatedString { parts: hir_parts }, Type::str_())
            }

            ExprKind::Tuple(elems) => {
                let hir_elems: Vec<HirExpr> = elems.iter().map(|e| self.lower_expr(e)).collect::<Result<_, _>>()?;
                let elem_types: Vec<Type> = hir_elems.iter().map(|e| e.ty.clone()).collect();
                (HirExprKind::Tuple(hir_elems), Type::Tuple(elem_types))
            }
        };

        Ok(HirExpr { kind, ty, span })
    }

    pub(super) fn lower_index(&mut self, idx: &ast::IndexExpr) -> TenthResult<Index> {
        match idx {
            ast::IndexExpr::Single(e) => Ok(Index::Single(self.lower_expr(e)?)),
            ast::IndexExpr::Range { start, end } => {
                let s = start.as_ref().map(|e| self.lower_expr(e)).transpose()?;
                let e = end.as_ref().map(|e| self.lower_expr(e)).transpose()?;
                Ok(Index::Range { start: s.map(Box::new), end: e.map(Box::new) })
            }
            ast::IndexExpr::Colon => Ok(Index::Colon),
        }
    }

    pub(super) fn lower_pattern(&mut self, pattern: &ast::Pattern) -> TenthResult<HirPattern> {
        match pattern {
            ast::Pattern::EnumVariant { enum_name, variant, field_bind, tuple_fields } => {
                Ok(HirPattern::EnumVariant {
                    enum_name: enum_name.clone(),
                    variant: variant.clone(),
                    field_bind: field_bind.clone(),
                    tuple_binds: tuple_fields.iter().enumerate()
                        .map(|(i, bind_name)| (format!("_{}", i), bind_name.clone()))
                        .collect(),
                })
            }
            ast::Pattern::Wildcard => Ok(HirPattern::Wildcard),
            ast::Pattern::Literal(lit) => {
                let hir_lit = match lit {
                    ast::Literal::Int(n) => Literal::Int(*n),
                    ast::Literal::Float(n, dt) => Literal::Float(*n, *dt),
                    ast::Literal::Bool(b) => Literal::Bool(*b),
                    ast::Literal::String(s) => Literal::String(s.clone()),
                };
                Ok(HirPattern::Literal(hir_lit))
            }
            ast::Pattern::Tuple(patterns) => {
                let hir_patterns: Vec<HirPattern> = patterns.iter()
                    .map(|p| self.lower_pattern(p))
                    .collect::<TenthResult<_>>()?;
                Ok(HirPattern::Tuple(hir_patterns))
            }
            ast::Pattern::Range { start, end, inclusive } => {
                Ok(HirPattern::Range {
                    start: *start,
                    end: *end,
                    inclusive: *inclusive,
                })
            }
            ast::Pattern::Binding(name) => {
                Ok(HirPattern::Binding(name.clone()))
            }
        }
    }

    /// Define variables in scope from a matched pattern.
    pub(super) fn bind_pattern_vars(&mut self, pattern: &HirPattern, scrutinee_ty: &Type) {
        match pattern {
            HirPattern::EnumVariant { enum_name, variant, field_bind, tuple_binds } => {
                let variant_fields = self.enums.get(enum_name)
                    .and_then(|variants| variants.iter().find(|(v, _)| v == variant))
                    .map(|(_, fields)| fields.clone());

                if let Some((_fname, bname)) = field_bind {
                    let bind_ty = variant_fields.as_ref()
                        .and_then(|f| f.first())
                        .map(|(_, t)| t.clone())
                        .unwrap_or(Type::Unknown);
                    self.scope.define_var(bname.clone(), bind_ty, false);
                }
                for (i, (_, bind_name)) in tuple_binds.iter().enumerate() {
                    let bind_ty = variant_fields.as_ref()
                        .and_then(|f| f.get(i))
                        .map(|(_, t)| t.clone())
                        .unwrap_or(Type::Unknown);
                    self.scope.define_var(bind_name.clone(), bind_ty, false);
                }
            }
            HirPattern::Tuple(patterns) => {
                // Bind each sub-pattern with type from tuple element
                if let Type::Tuple(elem_types) = scrutinee_ty {
                    for (i, sub_pat) in patterns.iter().enumerate() {
                        let elem_ty = elem_types.get(i).cloned().unwrap_or(Type::Unknown);
                        self.bind_pattern_vars(sub_pat, &elem_ty);
                    }
                } else {
                    for sub_pat in patterns {
                        self.bind_pattern_vars(sub_pat, &Type::Unknown);
                    }
                }
            }
            HirPattern::Binding(name) => {
                self.scope.define_var(name.clone(), scrutinee_ty.clone(), false);
            }
            HirPattern::Wildcard | HirPattern::Literal(_) | HirPattern::Range { .. } => {}
        }
    }
}
