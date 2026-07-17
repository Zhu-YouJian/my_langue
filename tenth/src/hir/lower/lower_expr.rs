use std::collections::HashMap;
use crate::error::{TenthError, TenthResult, TenthWarning};
use crate::lexer::token::Span;
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
            ast::Literal::Int(n, dt) => (Literal::Int(*n, *dt), Type::Base(*dt)),
                    ast::Literal::Float(n, dt) => (Literal::Float(*n, *dt), Type::Base(*dt)),
                    ast::Literal::Bool(b) => (Literal::Bool(*b), Type::bool_()),
                    ast::Literal::String(s) => (Literal::String(s.clone()), Type::str_()),
                    ast::Literal::Char(c) => (Literal::Char(*c), Type::Base(BaseType::Char)),
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
                                // 问题1：Option/Result 作为泛型枚举——返回 Type::Generic 携带类型参数
                                // Option<T> → Generic { base: Enum("Option"), args: [Unknown] }
                                // Result<T, E> → Generic { base: Enum("Result"), args: [Unknown, str] }
                                // 具体类型参数在 Call 表达式处理时从实参推断（见下方 Call 分支）
                                let ty = match enum_name {
                                    "Option" => Type::Generic {
                                        base: Box::new(Type::Enum("Option".to_string())),
                                        args: vec![Type::Unknown],
                                    },
                                    "Result" => Type::Generic {
                                        base: Box::new(Type::Enum("Result".to_string())),
                                        args: vec![Type::Unknown, Type::str_()],
                                    },
                                    _ => Type::Enum(enum_name.to_string()),
                                };
                                return Ok(HirExpr {
                                    kind: HirExprKind::EnumLiteral {
                                        enum_name: enum_name.to_string(),
                                        variant: variant.to_string(),
                                        fields: Vec::new(),
                                    },
                                    ty,
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
                            | "zeros_f16" | "ones_f16" | "zeros_bf16" | "ones_bf16"
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
                            | "tcp_listen" | "tcp_accept" | "tcp_listener_close"
                            // UDP 原语（基本功核查第 69 项；handle table 模式，与 tcp_* 同构）
                            | "udp_bind" | "udp_recv_from" | "udp_send_to" | "udp_close" | "udp_set_timeout"
                            | "command_new" | "command_arg" | "command_run" | "command_output"
                            | "http_get" | "http_post"
                            // Phase 2 Step 5：异步 I/O 原语（返回 Future）
                            | "async_sleep_ms" | "async_tcp_read" | "async_tcp_write"
                            // 正则表达式原语（句柄表方案，与 tcp_streams 对齐）
                            | "regex_compile" | "regex_match" | "regex_find" | "regex_find_all"
                            | "regex_replace" | "regex_split"
                            // Wave 3 第 8 项：Date native（路径 B，复用 struct 机制，返回 i64 或 Tuple）
                            | "date_to_unix_days" | "date_from_unix_days" | "date_i64_add_days"
                            | "date_diff_days" | "date_day_of_week"
                            // B批：字符串/文本处理 native
                            | "unicode_nfc" | "unicode_nfd"
                            | "str_to_utf16" | "utf16_to_str"
                            | "str_to_bytes" | "bytes_to_str"
                            | "to_utf8" | "to_utf16" | "from_utf16"
                            | "to_gbk" | "from_gbk"
                            | "base64_encode" | "base64_decode"
                            | "hex_encode" | "hex_decode"
                            | "url_encode" | "url_decode" => {
                                (HirExprKind::Var(ident.name.clone()), Type::Unknown)
                            }
                            _ => {
                                return Err(TenthError::TypeError {
                                    line: span.line,
                                    col: span.col,
                                    message: format!("未定义变量 '{}'", ident.name),
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
                // 运算符重载检查：若左侧类型实现了对应运算符 trait，则转为方法调用
                if let Some((trait_name, method_name)) = self.try_binary_op_overload(op, &l.ty) {
                    if self.has_trait_impl_for_type(&trait_name, &l.ty) {
                        let method = method_name.to_string();
                        let ret_ty = self.resolve_method_type(&l.ty, &method, &[r.clone()]);
                        let ret_ty2 = ret_ty.clone();
                        (HirExprKind::MethodCall {
                            receiver: Box::new(l),
                            method,
                            args: vec![r],
                            ret_ty,
                        }, ret_ty2)
                    } else {
                        // 无特质实现，回退到默认行为
                        Self::check_binary_shape_compat(op, &l.ty, &r.ty, &span)?;
                        let ty = self.infer_binary_type(op, &l.ty, &r.ty);
                        let hir_op = lower_binop(op);
                        (HirExprKind::Binary { op: hir_op, left: Box::new(l), right: Box::new(r), ty: ty.clone() }, ty)
                    }
                } else {
                    // 编译期 shape 检查：两侧 Tensor shape 不兼容时报错
                    Self::check_binary_shape_compat(op, &l.ty, &r.ty, &span)?;
                    let ty = self.infer_binary_type(op, &l.ty, &r.ty);
                    let hir_op = lower_binop(op);
                    (HirExprKind::Binary { op: hir_op, left: Box::new(l), right: Box::new(r), ty: ty.clone() }, ty)
                }
            }

            ExprKind::Unary { op, expr: inner } => {
                let e = self.lower_expr(inner)?;
                // 一元运算符重载检查：若类型实现了对应 trait，转为方法调用
                let (kind, ty) = if let Some((trait_name, method_name)) = self.try_unary_op_overload(op) {
                    if self.has_trait_impl_for_type(&trait_name, &e.ty) {
                        let method = method_name.to_string();
                        let ret_ty = self.resolve_method_type(&e.ty, &method, &[]);
                        let ret_ty2 = ret_ty.clone();
                        (HirExprKind::MethodCall {
                            receiver: Box::new(e),
                            method,
                            args: vec![],
                            ret_ty,
                        }, ret_ty2)
                    } else {
                        let hir_op = match op {
                            ast::UnaryOp::Neg => UnaryOp::Neg,
                            ast::UnaryOp::Not => UnaryOp::Not,
                            _ => UnaryOp::Try,
                        };
                        let ty = e.ty.clone();
                        (HirExprKind::Unary { op: hir_op, expr: Box::new(e), ty: ty.clone() }, ty)
                    }
                } else {
                    let hir_op = match op {
                        ast::UnaryOp::Neg => UnaryOp::Neg,
                        ast::UnaryOp::Not => UnaryOp::Not,
                        ast::UnaryOp::Try => UnaryOp::Try,
                    };
                    // Try 操作符类型推断：Result<T> → T；其他类型保持不变（运行时处理）
                    let ty = match (&hir_op, &e.ty) {
                        (UnaryOp::Try, Type::Generic { base, args }) => {
                            if let Type::Enum(name) = base.as_ref() {
                                if name == "Result" {
                                    args.first().cloned().unwrap_or(Type::Unknown)
                                } else {
                                    e.ty.clone()
                                }
                            } else {
                                e.ty.clone()
                            }
                        }
                        (UnaryOp::Try, _) => e.ty.clone(),
                        _ => e.ty.clone(),
                    };
                    (HirExprKind::Unary { op: hir_op, expr: Box::new(e), ty: ty.clone() }, ty)
                };
                (kind, ty)
            }

            ExprKind::Call { func, args } => {
                let f = self.lower_expr(func)?;

                // Process call arguments: resolve named args, fill defaults, collect variadic
                let processed_args = self.process_call_args(func, args, &span)?;
                let lowered_args = processed_args;

                // If the func is an EnumLiteral, merge args as tuple fields
                if let HirExprKind::EnumLiteral { enum_name, variant, fields } = &f.kind {
                    if fields.is_empty() && !lowered_args.is_empty() {
                        // 问题1：Option/Result 从实参推断类型参数
                        // Option::Some(value) → Generic { base: Enum("Option"), args: [value.ty] }
                        // Result::Ok(value) → Generic { base: Enum("Result"), args: [value.ty, str] }
                        // Result::Err(msg) → Generic { base: Enum("Result"), args: [Unknown, msg.ty] }
                        let ty = match enum_name.as_str() {
                            "Option" => {
                                let inner_ty = lowered_args.first().map(|a| a.ty.clone()).unwrap_or(Type::Unknown);
                                Type::Generic {
                                    base: Box::new(Type::Enum("Option".to_string())),
                                    args: vec![inner_ty],
                                }
                            }
                            "Result" => {
                                let (ok_ty, err_ty) = match variant.as_str() {
                                    "Ok" => (lowered_args.first().map(|a| a.ty.clone()).unwrap_or(Type::Unknown), Type::str_()),
                                    "Err" => (Type::Unknown, lowered_args.first().map(|a| a.ty.clone()).unwrap_or(Type::str_())),
                                    _ => (Type::Unknown, Type::str_()),
                                };
                                Type::Generic {
                                    base: Box::new(Type::Enum("Result".to_string())),
                                    args: vec![ok_ty, err_ty],
                                }
                            }
                            _ => Type::Unknown,
                        };
                        let tuple_fields: Vec<(String, HirExpr)> = lowered_args.into_iter().enumerate()
                            .map(|(i, a)| (format!("_{}", i), a))
                            .collect();
                        return Ok(HirExpr {
                            kind: HirExprKind::EnumLiteral {
                                enum_name: enum_name.clone(),
                                variant: variant.clone(),
                                fields: tuple_fields,
                            },
                            ty,
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
                    // 护城河 A Phase 1：cross_entropy native 函数的 logits/target shape 检查
                    // 把运行时 cross_entropy 反向 shape 错误提升到编译期 TypeError
                    if name == "cross_entropy" {
                        Self::check_cross_entropy_shape(&lowered_args, &span)?;
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
                    // Wave 2: f16/bf16 dtype 映射到 zeros_f16/ones_f16/zeros_bf16/ones_bf16
                    // （tensor/tensor_from_vec 不需要后缀，运行时按参数 dtype 构造）
                    let runtime_name = match (func_name.as_str(), dtype) {
                        ("randn", BaseType::F32) => "randn_f32".to_string(),
                        ("zeros", BaseType::F32) => "zeros_f32".to_string(),
                        ("ones", BaseType::F32) => "ones_f32".to_string(),
                        ("rand", BaseType::F32) => "rand_f32".to_string(),
                        ("zeros", BaseType::F16) => "zeros_f16".to_string(),
                        ("ones", BaseType::F16) => "ones_f16".to_string(),
                        ("zeros", BaseType::BF16) => "zeros_bf16".to_string(),
                        ("ones", BaseType::BF16) => "ones_bf16".to_string(),
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
                        param_defaults: template.param_defaults.clone(),
                        param_variadic: template.param_variadic.clone(),
                        return_type: inst_ret_ty.clone(),
                        body: inst_body,
                        generics: vec![],
                        generics_bounds: std::collections::HashMap::new(),
                        span: template.span.clone(),
                        is_test: false,
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
                    Type::Ref(inner, _) | Type::MutRef(inner, _) => inner.as_ref(),
                    other => other,
                };
                let field_ty = match inner_ty {
                    Type::Struct(name) | Type::TypeParam { name } => {
                        self.structs.get(name)
                            .and_then(|fields| fields.iter().find(|(n, _)| n == &field.name))
                            .map(|(_, ty)| ty.clone())
                            .or_else(|| {
                                self.unions.get(name)
                                    .and_then(|fields| fields.iter().find(|(n, _)| n == &field.name))
                                    .map(|(_, ty)| ty.clone())
                            })
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
                let ty = Type::Array { inner: Box::new(elem_ty), size: None };
                (HirExprKind::ArrayLiteral { elements: lowered, ty: ty.clone() }, ty)
            }

            ExprKind::Range { start, end, inclusive } => {
                let s = start.as_ref().map(|e| self.lower_expr(e)).transpose()?;
                let e = end.as_ref().map(|e| self.lower_expr(e)).transpose()?;
                // 问题9修复：Range 作为一等类型，返回 Type::Range 而非元素类型
                let inner_ty = s.as_ref()
                    .or(e.as_ref())
                    .map(|expr| expr.ty.clone())
                    .unwrap_or(Type::i32());
                let ty = Type::Range { inner: Box::new(inner_ty), inclusive: *inclusive };
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
                // 类型推断规则（含 Never 类型）：
                // - 无 else 分支：返回 Unit（旧语义）
                // - else 存在：
                //   * then=Never → 用 else 类型（then 永不返回）
                //   * else=Never → 用 then 类型（else 永不返回）
                //   * 两者都=Never → Never（整段永不返回）
                //   * 否则 → 用 else 类型（保持原行为）
                let ty = if let Some(ref eb) = e {
                    match (&t.ty, &eb.ty) {
                        (Type::Never, Type::Never) => Type::Never,
                        (Type::Never, _) => eb.ty.clone(),
                        (_, Type::Never) => t.ty.clone(),
                        _ => eb.ty.clone(),
                    }
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

                // Block 类型推断：
                // - 末尾是 Expr → 用其类型
                // - 末尾是 Return → Never（块永不正常返回）
                // - 否则 → Unit
                let ty = if let Some(e) = final_expr.as_ref() {
                    e.ty.clone()
                } else if matches!(lowered_stmts.last().map(|s| &s.kind), Some(HirStmtKind::Return(_))) {
                    Type::Never
                } else {
                    Type::unit()
                };

                // RAII：收集当前作用域中实现了 Drop trait 的变量，在作用域退出时调用 drop()
                let drop_vars: Vec<String> = self.collect_drop_vars();

                let stmts_without_final: Vec<HirStmt> = if final_expr.is_some() {
                    lowered_stmts[..lowered_stmts.len().saturating_sub(1)].to_vec()
                } else {
                    lowered_stmts
                };

                // 构造包含 drop 调用的块
                let outer_scope = std::mem::replace(&mut self.scope, Scope::new());
                self.scope = outer_scope;

                if drop_vars.is_empty() {
                    (HirExprKind::Block { stmts: stmts_without_final, final_expr: final_expr.map(Box::new) }, ty)
                } else {
                    // 有需要 drop 的变量：将 final_expr 保存到临时变量，插入 drop 调用，再返回临时变量
                    let drop_stmt = Self::make_drop_stmt(&drop_vars, span.clone());
                    let mut new_stmts = stmts_without_final;
                    if let Some(fe) = final_expr {
                        // 保存 final_expr 到临时变量
                        let save_temp = HirStmt {
                            kind: HirStmtKind::Let {
                                names: vec!["__drop_tmp__".to_string()],
                                type_ann: None,
                                mutable: false,
                                init: Some(fe),
                            },
                            span: span.clone(),
                        };
                        new_stmts.push(save_temp);
                        // 插入 drop 调用
                        new_stmts.push(drop_stmt);
                        // 返回临时变量
                        (HirExprKind::Block {
                            stmts: new_stmts,
                            final_expr: Some(Box::new(HirExpr {
                                kind: HirExprKind::Var("__drop_tmp__".to_string()),
                                ty: ty.clone(),
                                span: span.clone(),
                            })),
                        }, ty)
                    } else {
                        new_stmts.push(drop_stmt);
                        (HirExprKind::Block { stmts: new_stmts, final_expr: None }, ty)
                    }
                }
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
                            message: "无效的赋值目标".into(),
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
                        message: "无效的赋值目标".into(),
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
                                            kind: HirExprKind::Literal(Literal::Int(0, *b)),
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
                                            kind: HirExprKind::Literal(Literal::Int(0, BaseType::Char)),
                                            ty: fty.clone(),
                                            span: name.span.clone(),
                                        },
                                        BaseType::Unit => HirExpr {
                                            kind: HirExprKind::Literal(Literal::Int(0, BaseType::I32)),
                                            ty: Type::unit(),
                                            span: name.span.clone(),
                                        },
                                        BaseType::BigInt => HirExpr {
                                            kind: HirExprKind::Literal(Literal::String("0".to_string())),
                                            ty: fty.clone(),
                                            span: name.span.clone(),
                                        },
                                        BaseType::C64 | BaseType::C128 => HirExpr {
                                            kind: HirExprKind::Literal(Literal::Float(0.0, BaseType::F64)),
                                            ty: fty.clone(),
                                            span: name.span.clone(),
                                        },
                                        BaseType::Decimal => HirExpr {
                                            kind: HirExprKind::Literal(Literal::String("0".to_string())),
                                            ty: fty.clone(),
                                            span: name.span.clone(),
                                        },
                                    },
                                    _ => HirExpr {
                                        kind: HirExprKind::Literal(Literal::Int(0, BaseType::I32)),
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
                // 问题1：Option/Result 作为泛型枚举——从字段推断类型参数
                let ty = match enum_name.name.as_str() {
                    "Option" => {
                        let inner_ty = lowered_fields.first()
                            .map(|(_, e)| e.ty.clone())
                            .unwrap_or(Type::Unknown);
                        Type::Generic {
                            base: Box::new(Type::Enum("Option".to_string())),
                            args: vec![inner_ty],
                        }
                    }
                    "Result" => {
                        let (ok_ty, err_ty) = match variant.name.as_str() {
                            "Ok" => (
                                lowered_fields.first().map(|(_, e)| e.ty.clone()).unwrap_or(Type::Unknown),
                                Type::str_(),
                            ),
                            "Err" => (
                                Type::Unknown,
                                lowered_fields.first().map(|(_, e)| e.ty.clone()).unwrap_or(Type::str_()),
                            ),
                            _ => (Type::Unknown, Type::str_()),
                        };
                        Type::Generic {
                            base: Box::new(Type::Enum("Result".to_string())),
                            args: vec![ok_ty, err_ty],
                        }
                    }
                    _ => Type::Enum(enum_name.name.clone()),
                };
                (HirExprKind::EnumLiteral {
                    enum_name: enum_name.name.clone(),
                    variant: variant.name.clone(),
                    fields: lowered_fields,
                }, ty)
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
                // 问题10修复：match 穷尽性检查——对枚举 match，检查所有变体是否覆盖
                // 若存在 Wildcard arm（`_`），视为覆盖所有，跳过检查
                // 否则收集所有 EnumVariant arm 的 variant 名，与枚举定义对比
                // 问题1：Option/Result 现在是 Type::Generic { base: Enum, args }，
                // 需要从 Generic.base 提取枚举名
                let scrutinee_enum_name: Option<&str> = match &lowered_scrutinee.ty {
                    Type::Enum(name) => Some(name.as_str()),
                    Type::Generic { base, .. } => {
                        if let Type::Enum(name) = base.as_ref() { Some(name.as_str()) } else { None }
                    }
                    _ => None,
                };
                if let Some(enum_name) = scrutinee_enum_name {
                    let has_wildcard = lowered_arms.iter().any(|arm| matches!(arm.pattern, HirPattern::Wildcard));
                    if !has_wildcard {
                        if let Some(variants) = self.enums.get(enum_name) {
                            let covered: std::collections::HashSet<&str> = lowered_arms.iter()
                                .filter_map(|arm| {
                                    if let HirPattern::EnumVariant { variant, .. } = &arm.pattern {
                                        Some(variant.as_str())
                                    } else {
                                        None
                                    }
                                })
                                .collect();
                            let all_variants: std::collections::HashSet<&str> = variants.iter()
                                .map(|(name, _)| name.as_str())
                                .collect();
                            let missing: Vec<&str> = all_variants.difference(&covered).copied().collect();
                            if !missing.is_empty() {
                                return Err(TenthError::TypeError {
                                    line: span.line,
                                    col: span.col,
                                    message: format!(
                                        "match 不穷尽：枚举 {} 缺少变体 {}（请添加对应 arm 或通配符 _）",
                                        enum_name,
                                        missing.iter().map(|s| format!("'{}'", s)).collect::<Vec<_>>().join(", ")
                                    ),
                                });
                            }
                        }
                    }
                }
                // Infer match type: prefer first non-Unknown/non-Never arm（普通类型），
                // 其次 Never（所有 arm 都 return 时），最后 Unknown 兜底。
                // 这样含 return 的 arm 不会污染整个 match 的类型。
                let match_ty = lowered_arms.iter()
                    .map(|arm| arm.body.ty.clone())
                    .find(|ty| !matches!(ty, Type::Unknown | Type::Never))
                    .or_else(|| lowered_arms.iter()
                        .map(|arm| arm.body.ty.clone())
                        .find(|ty| matches!(ty, Type::Never)))
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
                let ty = Type::Ref(Box::new(e.ty.clone()), None);
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
                let ty = Type::MutRef(Box::new(e.ty.clone()), None);
                if let ExprKind::Ident(ident) = &inner.kind {
                    self.scope.set_ownership(&ident.name, Ownership::ExclusiveRef);
                }
                (HirExprKind::MutRef(Box::new(e)), ty)
            }

            ExprKind::Deref(inner) => {
                let e = self.lower_expr(inner)?;
                let inner_ty = match &e.ty {
                    Type::Ref(t, _) | Type::MutRef(t, _) => (**t).clone(),
                    _ => Type::Unknown,
                };
                (HirExprKind::Deref(Box::new(e)), inner_ty)
            }

            ExprKind::Move(inner) => {
                let e = self.lower_expr(inner)?;
                let ty = e.ty.clone();
                if let ExprKind::Ident(ident) = &inner.kind {
                    // Copy 类型的值在 move 时不标记为 Moved（值被复制，原变量仍可用）
                    if !super::is_copy_type(&ty, &self.structs, &self.trait_impls) {
                        self.scope.set_ownership(&ident.name, Ownership::Moved);
                    }
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

            ExprKind::Yield(inner) => {
                // yield [expr]：让出控制权给 VM 调度器。
                // 设计决策（与 Op::Yield 语义对齐）：
                // - inner 若存在会被求值（lower 之，副作用保留），但其值被丢弃
                // - yield 表达式本身返回 Unit（恢复时调度器不向栈上 push 任何值）
                let hir_inner = match inner {
                    Some(e) => Some(Box::new(self.lower_expr(e)?)),
                    None => None,
                };
                (HirExprKind::Yield(hir_inner), Type::unit())
            }

            ExprKind::InterpolatedString(parts) => {
                let hir_parts: Vec<crate::hir::hir::InterpPart> = parts.iter().map(|p| match p {
                    ast::InterpPart::Literal(s) => crate::hir::hir::InterpPart::Literal(s.clone()),
                    ast::InterpPart::Expr(e) => crate::hir::hir::InterpPart::Expr(e.clone()),
                }).collect();
                (HirExprKind::InterpolatedString { parts: hir_parts }, Type::str_())
            }

            // f"..." 模板字符串 → 编译为 format("template", arg1, arg2, ...)
            ExprKind::FString(parts) => {
                let mut template = String::new();
                let mut args: Vec<HirExpr> = Vec::new();
                for p in parts {
                    match p {
                        ast::InterpPart::Literal(s) => template.push_str(s),
                        ast::InterpPart::Expr(var_name) => {
                            template.push_str("{}");
                            // Resolve variable by name
                            let var_expr = HirExpr {
                                kind: HirExprKind::Var(var_name.clone()),
                                ty: Type::Unknown,
                                span: span.clone(),
                            };
                            args.push(var_expr);
                        }
                    }
                }
                // Build template literal
                let template_lit = HirExpr {
                    kind: HirExprKind::Literal(Literal::String(template)),
                    ty: Type::str_(),
                    span: span.clone(),
                };
                // Build Call: format(template, args...)
                let func_expr = HirExpr {
                    kind: HirExprKind::Var("format".to_string()),
                    ty: Type::Unknown,
                    span: span.clone(),
                };
                let mut call_args = vec![template_lit];
                call_args.extend(args);
                (HirExprKind::Call {
                    func: Box::new(func_expr),
                    args: call_args,
                    ret_ty: Type::str_(),
                }, Type::str_())
            }

            ExprKind::Tuple(elems) => {
                let hir_elems: Vec<HirExpr> = elems.iter().map(|e| self.lower_expr(e)).collect::<Result<_, _>>()?;
                let elem_types: Vec<Type> = hir_elems.iter().map(|e| e.ty.clone()).collect();
                (HirExprKind::Tuple(hir_elems), Type::Tuple(elem_types))
            }

            // NamedArg should only appear as a direct child of Call/MethodCall args,
            // where it is handled by process_call_args. If we encounter one here,
            // it means it somehow reached general expression lowering — error out.
            ExprKind::NamedArg { name, .. } => {
                return Err(TenthError::TypeError {
                    line: span.line,
                    col: span.col,
                    message: format!(
                        "命名参数 '{}' 只能在函数调用参数列表中使用",
                        name.name
                    ),
                });
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
        ast::Literal::Int(n, dt) => Literal::Int(*n, *dt),
                    ast::Literal::Float(n, dt) => Literal::Float(*n, *dt),
                    ast::Literal::Bool(b) => Literal::Bool(*b),
                    ast::Literal::Char(c) => Literal::Char(*c),
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

    /// Process function call arguments: resolve named args, fill defaults, collect variadic.
    /// Returns lowered HIR expressions ready for call emission.
    fn process_call_args(&mut self, func: &ast::Expr, args: &[ast::Expr], span: &Span) -> TenthResult<Vec<HirExpr>> {
        use ast::ExprKind;

        // Get function name if it's a simple identifier
        let fn_name = match &func.kind {
            ExprKind::Ident(ident) => Some(ident.name.clone()),
            _ => None,
        };

        // Look up function definition info (clone to avoid borrow conflicts)
        let fn_param_names: Vec<String> = fn_name.as_ref()
            .and_then(|name| self.functions.iter().find(|f| f.name == *name)
                .or_else(|| self.generic_funcs.get(name)))
            .map(|def| def.params.iter().map(|(n, _)| n.clone()).collect())
            .unwrap_or_default();
        let fn_param_defaults: Vec<Option<HirExpr>> = fn_name.as_ref()
            .and_then(|name| self.functions.iter().find(|f| f.name == *name)
                .or_else(|| self.generic_funcs.get(name)))
            .map(|def| def.param_defaults.clone())
            .unwrap_or_default();
        let fn_param_variadic: Vec<bool> = fn_name.as_ref()
            .and_then(|name| self.functions.iter().find(|f| f.name == *name)
                .or_else(|| self.generic_funcs.get(name)))
            .map(|def| def.param_variadic.clone())
            .unwrap_or_default();
        let has_def = fn_name.as_ref()
            .map(|name| {
                self.functions.iter().any(|f| f.name == *name)
                    || self.generic_funcs.contains_key(name)
            })
            .unwrap_or(false);

        // Separate positional args (non-NamedArg) and named args
        let mut positional_ast: Vec<&ast::Expr> = Vec::new();
        let mut named_ast: Vec<(String, &ast::Expr)> = Vec::new();

        for arg in args {
            match &arg.kind {
                ExprKind::NamedArg { name, value } => {
                    named_ast.push((name.name.clone(), value.as_ref()));
                }
                _ => {
                    positional_ast.push(arg);
                }
            }
        }

        // If there are named args, we need to resolve them to positions
        if !named_ast.is_empty() || has_def {
            // Build the list of AST expressions in parameter order
            let param_count = if has_def { fn_param_names.len() } else {
                positional_ast.len() + named_ast.len()
            };

            let mut resolved: Vec<Option<&ast::Expr>> = vec![None; param_count];
            let mut used_positional: Vec<bool> = vec![false; param_count];

            // Place positional arguments
            for (i, arg) in positional_ast.iter().enumerate() {
                if i < param_count {
                    resolved[i] = Some(arg);
                    used_positional[i] = true;
                }
            }

            // Place named arguments, ensuring no duplicate
            for (name, expr) in &named_ast {
                if has_def {
                    if let Some(pos) = fn_param_names.iter().position(|n| n == name) {
                        if used_positional[pos] {
                            // Position already filled by positional arg
                            return Err(TenthError::TypeError {
                                line: span.line,
                                col: span.col,
                                message: format!(
                                    "参数 '{}' 在函数 '{}' 调用中被同时指定为位置参数和命名参数",
                                    name, fn_name.as_deref().unwrap_or("?")
                                ),
                            });
                        }
                        resolved[pos] = Some(expr);
                        used_positional[pos] = true;
                    } else {
                        return Err(TenthError::TypeError {
                            line: span.line,
                            col: span.col,
                            message: format!(
                                "函数 '{}' 没有名为 '{}' 的参数",
                                fn_name.as_deref().unwrap_or("?"), name
                            ),
                        });
                    }
                } else {
                    // Unknown function with named args: just skip position resolution
                    // Fall through to positional-only
                }
            }

            // Collect lowered HIR args, handling defaults and variadic.
            // We produce HIR expressions directly for each parameter position.
            let mut hir_result: Vec<HirExpr> = Vec::new();
            let mut extra_hir_args: Vec<HirExpr> = Vec::new();
            let mut has_variadic = false;

            if has_def {
                for (i, slot) in resolved.iter().enumerate() {
                    if let Some(expr) = slot {
                        // Normal argument: lower the AST expression
                        hir_result.push(self.lower_expr(expr)?);
                    } else if i < fn_param_defaults.len() && fn_param_defaults[i].is_some() {
                        // Has default value: clone the already-lowered default expression
                        hir_result.push(fn_param_defaults[i].clone().unwrap());
                    } else if i < fn_param_variadic.len() && fn_param_variadic[i] {
                        // Variadic param slot: will be filled from extra args if any
                        has_variadic = true;
                        // Temporarily push a placeholder (will be replaced if extra args exist)
                        let empty_array = HirExpr {
                            kind: HirExprKind::ArrayLiteral {
                                elements: Vec::new(),
                                ty: Type::Array { inner: Box::new(Type::Unknown), size: None },
                            },
                            ty: Type::Array { inner: Box::new(Type::Unknown), size: None },
                            span: span.clone(),
                        };
                        hir_result.push(empty_array);
                    } else {
                        // Missing required parameter
                        return Err(TenthError::TypeError {
                            line: span.line,
                            col: span.col,
                            message: format!(
                                "函数 '{}' 调用缺少必需参数 '{}'",
                                fn_name.as_deref().unwrap_or("?"),
                                fn_param_names.get(i).map(|s| s.as_str()).unwrap_or("?")
                            ),
                        });
                    }
                }

                // Collect extra positional args for variadic parameter
                if has_variadic && positional_ast.len() > resolved.len() {
                    for arg in positional_ast.iter().skip(resolved.len()) {
                        extra_hir_args.push(self.lower_expr(arg)?);
                    }
                }
            } else {
                // No function definition found (e.g., native function or unknown).
                // Lower positional args, then append named args as key-value pairs.
                for arg in positional_ast.iter() {
                    hir_result.push(self.lower_expr(arg)?);
                }
                // Append named args as interleaved key-value pairs (key_str, value_expr, ...)
                // so native functions like format() can receive them.
                for (name, expr) in &named_ast {
                    // Key: lowered string literal
                    hir_result.push(HirExpr {
                        kind: HirExprKind::Literal(Literal::String(name.clone())),
                        ty: Type::str_(),
                        span: expr.span.clone(),
                    });
                    // Value: lowered expression
                    hir_result.push(self.lower_expr(expr)?);
                }
            }

            // If we have extra variadic args, build the array and replace the placeholder
            if has_variadic && !extra_hir_args.is_empty() {
                let elem_ty = extra_hir_args.first()
                    .map(|e| e.ty.clone())
                    .unwrap_or(Type::Unknown);
                let elem_ty_clone = elem_ty.clone();
                let array_lit = HirExpr {
                    kind: HirExprKind::ArrayLiteral {
                        elements: extra_hir_args,
                        ty: Type::Array { inner: Box::new(elem_ty), size: None },
                    },
                    ty: Type::Array { inner: Box::new(elem_ty_clone), size: None },
                    span: span.clone(),
                };
                // Replace the variadic slot (last element added for it)
                if let Some(last) = hir_result.last_mut() {
                    *last = array_lit;
                } else {
                    hir_result.push(array_lit);
                }
            }

            return Ok(hir_result);
        } else {
            // No named args, no function def lookup needed
            let mut lowered: Vec<HirExpr> = Vec::new();
            for arg in positional_ast {
                lowered.push(self.lower_expr(arg)?);
            }
            return Ok(lowered);
        }
    }

    /// 运算符重载：二元运算符 → trait 方法名称映射。
    /// 返回 (trait_name, method_name)，如 ("Add", "add")。
    pub(super) fn try_binary_op_overload(&self, op: &ast::BinOp, _ty: &Type) -> Option<(&'static str, &'static str)> {
        use ast::BinOp;
        match op {
            BinOp::Add => Some(("Add", "add")),
            BinOp::Sub => Some(("Sub", "sub")),
            BinOp::Mul => Some(("Mul", "mul")),
            BinOp::Div => Some(("Div", "div")),
            BinOp::Mod => Some(("Rem", "rem")),
            BinOp::Eq | BinOp::NotEq => Some(("Eq", "eq")),
            BinOp::Lt => Some(("Ord", "lt")),
            BinOp::Gt => Some(("Ord", "gt")),
            BinOp::LtEq => Some(("Ord", "le")),
            BinOp::GtEq => Some(("Ord", "ge")),
            BinOp::And | BinOp::Or => None, // 短路逻辑运算符不重载
        }
    }

    /// 运算符重载：一元运算符 → trait 方法名称映射。
    pub(super) fn try_unary_op_overload(&self, op: &ast::UnaryOp) -> Option<(&'static str, &'static str)> {
        use ast::UnaryOp;
        match op {
            UnaryOp::Neg => Some(("Neg", "neg")),
            UnaryOp::Not => Some(("Not", "not")),
            UnaryOp::Try => None,
        }
    }

    /// 检查指定类型是否实现了指定 trait。
    /// 通过搜索 trait_impls[trait_name][type_name] 来判断。
    pub(super) fn has_trait_impl_for_type(&self, trait_name: &str, ty: &Type) -> bool {
        let type_name = match ty {
            Type::Base(_b) => return false, // 基础类型的内置运算不在 trait 系统中重载
            Type::Struct(name) | Type::Enum(name) => name,
            Type::Generic { base, .. } => match base.as_ref() {
                Type::Enum(name) | Type::Struct(name) => name,
                _ => return false,
            },
            _ => return false,
        };
        self.trait_impls.get(trait_name)
            .and_then(|impls| impls.get(type_name))
            .is_some()
    }

    /// 收集当前作用域中所有实现了 Drop trait 的变量名。
    /// 按定义顺序的逆序返回（后定义的先 drop）。
    fn collect_drop_vars(&self) -> Vec<String> {
        // 从 scope 中获取所有变量及其类型
        let mut drop_vars: Vec<String> = Vec::new();
        self.scope.for_each_var(|name, ty| {
            if self.type_impls_drop(ty) {
                drop_vars.push(name.to_string());
            }
        });
        // 逆序：后定义的先 drop
        drop_vars.reverse();
        drop_vars
    }

    /// 检查类型是否实现了 Drop trait。
    fn type_impls_drop(&self, ty: &Type) -> bool {
        match ty {
            Type::Base(_) | Type::Never => false,
            Type::Ref(_, _) | Type::MutRef(_, _) => false, // 引用不 drop
            Type::Struct(name) => {
                self.trait_impls.get("Drop")
                    .and_then(|impls| impls.get(name))
                    .is_some()
            }
            Type::Enum(name) => {
                self.trait_impls.get("Drop")
                    .and_then(|impls| impls.get(name))
                    .is_some()
            }
            Type::Dyn(_) => false, // trait 对象不自动 drop（需运行时决定）
            Type::Array { inner, .. } => self.type_impls_drop(inner),
            Type::Tuple(types) => types.iter().any(|t| self.type_impls_drop(t)),
            // HeapBox/Pin/SharedBox/AtomicBox：容器类型，drop 行为由运行时管理，不在编译期自动生成 drop 调用
            Type::HeapBox(inner) | Type::Pin(inner) => self.type_impls_drop(inner),
            Type::SharedBox(_) | Type::AtomicBox(_) => false, // RC/Arc 管理自身生命周期
            _ => false,
        }
    }

    /// 为指定的变量列表生成 drop 调用语句。
    /// 每个变量生成一个 `var.drop()` 方法调用。
    fn make_drop_stmt(drop_vars: &[String], span: crate::lexer::token::Span) -> HirStmt {
        // 将所有 drop 调用组合到一个 Expr 语句中
        // 使用 Block 按顺序执行每个 drop 调用
        let drop_calls: Vec<HirExpr> = drop_vars.iter().map(|var_name| {
            HirExpr {
                kind: HirExprKind::MethodCall {
                    receiver: Box::new(HirExpr {
                        kind: HirExprKind::Var(var_name.clone()),
                        ty: Type::Unknown,
                        span: span.clone(),
                    }),
                    method: "drop".to_string(),
                    args: vec![],
                    ret_ty: Type::unit(),
                },
                ty: Type::unit(),
                span: span.clone(),
            }
        }).collect();

        // 如果只有一个 drop 调用，直接返回表达式语句
        if drop_calls.len() == 1 {
            return HirStmt {
                kind: HirStmtKind::Expr(drop_calls.into_iter().next().unwrap()),
                span,
            };
        }

        // 多个 drop 调用：用 Block 包装，每个 drop 作为表达式语句
        let drop_stmts: Vec<HirStmt> = drop_calls.into_iter().map(|expr| {
            HirStmt {
                kind: HirStmtKind::Expr(expr),
                span: span.clone(),
            }
        }).collect();

        HirStmt {
            kind: HirStmtKind::Expr(HirExpr {
                kind: HirExprKind::Block { stmts: drop_stmts, final_expr: None },
                ty: Type::unit(),
                span: span.clone(),
            }),
            span,
        }
    }
}
