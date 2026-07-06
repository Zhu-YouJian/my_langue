use std::collections::HashMap;
use crate::error::{TenthError, TenthResult, TenthWarning};
use crate::parser::ast as ast;
use crate::hir::hir::*;
use crate::hir::types::*;
use super::Scope;
use super::Ownership;
use super::{substitute_type, substitute_expr, check_generic_instantiation_soundness, lower_binop};
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
                            "println" | "print" | "eprintln" | "eprint" | "tensor" | "rand" | "randn" | "randn_f32" | "rand_f32" | "zeros_f32" | "ones_f32"
                            | "read_file" | "write_file" | "write_bytes" | "read_bytes"
                            | "read_line" | "env_get" | "env_set" | "exit"
                            | "str_at" | "str_len" | "str_cmp" | "str_slice" | "str_add" | "str_eq" | "str_int"
                            | "Vec::new" | "HashMap::new"
                            | "compile_host" | "compile_program"
                            | "start_grad" | "new_grad" | "stop_grad"
                            | "param" | "backward" | "grad" | "zero_grad"
                            | "explain_error"
                            | "cross_entropy"
                            | "select"
                            | "scatter"
                            | "gather"
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
                            | "lower_program" | "compile_to_wasm"
                            // Stage 3+4 TCP/HTTP 原语
                            | "tcp_connect" | "tcp_read" | "tcp_write" | "tcp_close" | "tcp_set_timeout"
                            | "http_get" | "http_post" => {
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
                // 编译期 shape 检查：两侧 Tensor shape 不兼容时报错
                Self::check_binary_shape_compat(op, &l.ty, &r.ty, &span)?;
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
                // 编译期内存预估：构造函数返回大 tensor 时发 warning
                self.emit_memory_estimate(&ret_ty, &span, "函数调用");
                // 方向 A：对 param(t) 调用，提示梯度 shape 应与 t 一致（让用户意识到梯度 shape 约束）
                if let HirExprKind::Var(name) = &f.kind {
                    if name == "param" {
                        if let Some(arg) = lowered_args.first() {
                            if let Some(bytes) = arg.ty.static_bytes() {
                                const GB: u64 = 1024 * 1024 * 1024;
                                if bytes >= GB {
                                    let msg = format!(
                                        "param() 注册约 {:.2} GB 的可训练参数（反向传播将分配同等大小的梯度，可能触发 OOM）",
                                        bytes as f64 / GB as f64
                                    );
                                    self.warnings.push(TenthWarning::new(span.line, span.col, msg));
                                }
                            }
                        }
                    }
                }

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

                // 支持泛型实例化的 native 构造函数（无 HIR 函数体，仅做类型推断）
                // randn<T>(d) / zeros<T>(d) / ones<T>(d) / rand<T>(d) / tensor<T>(...) / tensor_from_vec<T>(...)
                const NATIVE_GENERIC_CTORS: &[&str] = &[
                    "randn", "zeros", "ones", "rand", "tensor", "tensor_from_vec",
                ];
                if NATIVE_GENERIC_CTORS.contains(&func_name.as_str()) {
                    // 类型参数必须是具体 BaseType（native 不支持 TypeParam 嵌套）
                    let type_args: Vec<Type> = generics.iter()
                        .map(|ta| Type::from_annotation(ta))
                        .collect();
                    if type_args.len() != 1 {
                        return Err(TenthError::TypeError {
                            line: span.line,
                            col: span.col,
                            message: format!(
                                "native 构造函数 '{}' 期望 1 个类型参数，得到 {}",
                                func_name, type_args.len()
                            ),
                        });
                    }
                    let dtype = match &type_args[0] {
                        Type::Base(b) => *b,
                        _ => {
                            return Err(TenthError::TypeError {
                                line: span.line,
                                col: span.col,
                                message: format!(
                                    "native 构造函数 '{}' 的类型参数必须是具体 BaseType，得到 {:?}",
                                    func_name, type_args[0]
                                ),
                            });
                        }
                    };
                    let lowered_args: Vec<HirExpr> = args.iter()
                        .map(|a| self.lower_expr(a))
                        .collect::<TenthResult<_>>()?;
                    // shape 推断：字面量参数（如 randn<f32>(3, 4)）→ [Known(3), Known(4)]
                    let ret_ty = Type::tensor(dtype, Self::shape_from_int_args(&lowered_args));
                    // 运行时按名字分发：f32 dtype 映射到 randn_f32/zeros_f32/ones_f32/rand_f32
                    // （tensor/tensor_from_vec 不需要后缀，运行时按参数 dtype 构造）
                    let runtime_name = match (func_name.as_str(), dtype) {
                        ("randn", BaseType::F32) => "randn_f32".to_string(),
                        ("zeros", BaseType::F32) => "zeros_f32".to_string(),
                        ("ones", BaseType::F32) => "ones_f32".to_string(),
                        ("rand", BaseType::F32) => "rand_f32".to_string(),
                        _ => func_name.clone(),
                    };
                    // 编译期内存预估：泛型构造函数返回大 tensor 时发 warning
                    self.emit_memory_estimate(&ret_ty, &span, &format!("泛型构造函数 {}", func_name));
                    return Ok(HirExpr {
                        kind: HirExprKind::Call {
                            func: Box::new(HirExpr {
                                kind: HirExprKind::Var(runtime_name),
                                ty: Type::Unknown,
                                span: span.clone(),
                            }),
                            args: lowered_args,
                            ret_ty: ret_ty.clone(),
                        },
                        ty: ret_ty,
                        span: span.clone(),
                    });
                }

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

                // AUDIT-11.1.5 / T18 修复：泛型实例化健全性检查。
                // type_map 必须覆盖所有声明的泛型参数，且每个替换值必须是具体类型
                // （不能是 Unknown 或 TypeParam），否则 body 内仍残留类型变量。
                if let Err(msg) = check_generic_instantiation_soundness(&template.generics, &type_map) {
                    return Err(TenthError::TypeError {
                        line: span.line,
                        col: span.col,
                        message: format!("泛型函数 '{}' 实例化失败：{}", func_name, msg),
                    });
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
                    // AUDIT-11.1.5 / T18 修复：body 不能直接 clone，
                    // 必须递归替换 body 中所有 TypeParam，确保实例化后无残留类型变量。
                    let inst_body = substitute_expr(&template.body, &type_map);
                    let inst_fn = HirFnDef {
                        name: mangled_name.clone(),
                        params: inst_params,
                        return_type: inst_ret_ty.clone(),
                        body: inst_body,
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
                // 编译期内存预估：泛型函数实例化返回大 tensor 时发 warning
                self.emit_memory_estimate(&inst_ret_ty, &span, &format!("泛型函数 {}", func_name));
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
                        // 编译期 shape 检查（如 matmul 的内侧维度）
                        Self::check_method_shape(&recv.ty, &method.name, &all_args, &span)?;
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
                // 编译期 shape 检查（如 matmul 的内侧维度）
                Self::check_method_shape(&recv.ty, &method.name, &lowered_args, &span)?;
                // 编译期算力/内存预估：matmul/bmm FLOPs + 结果 tensor bytes
                if method.name == "matmul" {
                    if let Some(arg) = lowered_args.first() {
                        self.emit_matmul_flop_estimate(&recv.ty, &arg.ty, &span);
                    }
                }
                if method.name == "bmm" {
                    if let Some(arg) = lowered_args.first() {
                        self.emit_bmm_flop_estimate(&recv.ty, &arg.ty, &span);
                    }
                }
                self.emit_memory_estimate(&ret_ty, &span, &format!("方法 {}", method.name));

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
                // 跨分支 shape 检查：then/else 都有静态 shape 时，必须可广播
                if let Some(ref eb) = e {
                    Self::check_branch_shape_compat(&t.ty, &eb.ty, &span, "if/else")?;
                }
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
                // 跨 arm shape 检查：所有 arm 的 body shape 必须兼容
                // 取第一个 arm 作为基准，与其他 arm 两两检查
                if let Some(first_arm) = lowered_arms.first() {
                    for arm in lowered_arms.iter().skip(1) {
                        Self::check_branch_shape_compat(&first_arm.body.ty, &arm.body.ty, &span, "match arm")?;
                    }
                }
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

            ExprKind::Await(inner) => {
                let e = self.lower_expr(inner)?;
                // await 的类型：若 inner 是 Future<T>，取 T；否则 Unknown（运行时处理）
                let await_ty = match &e.ty {
                    Type::Future(inner_t) => (**inner_t).clone(),
                    _ => Type::Unknown,
                };
                (HirExprKind::Await(Box::new(e)), await_ty)
            }

            ExprKind::Spawn(inner) => {
                let e = self.lower_expr(inner)?;
                // spawn 返回 Future<inner.ty>
                let spawn_ty = Type::Future(Box::new(e.ty.clone()));
                (HirExprKind::Spawn(Box::new(e)), spawn_ty)
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
            ast::Pattern::Struct { name, fields } => {
                Ok(HirPattern::Struct {
                    name: name.clone(),
                    fields: fields.clone(),
                })
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
            HirPattern::Struct { name, fields } => {
                // Look up struct field types for proper type inference on binds.
                let field_types = self.structs.get(name).cloned();
                for (field_name, bind_name) in fields {
                    let bind_ty = field_types.as_ref()
                        .and_then(|fts| fts.iter().find(|(n, _)| n == field_name))
                        .map(|(_, t)| t.clone())
                        .unwrap_or(Type::Unknown);
                    self.scope.define_var(bind_name.clone(), bind_ty, false);
                }
            }
            HirPattern::Wildcard | HirPattern::Literal(_) | HirPattern::Range { .. } => {}
        }
    }
}
