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
                        if self.enum_variant_fields(enum_name, variant).is_some() {
                            // 问题1：Option/Result 作为泛型枚举——返回 Type::Generic 携带类型参数
                            // Option<T> → Generic { base: Enum("Option"), args: [Unknown] }
                            // Result<T, E> → Generic { base: Enum("Result"), args: [Unknown, str] }
                            // 具体类型参数在 Call 表达式处理时从实参推断（见下方 Call 分支）
                            // M2.1：泛型枚举（`enum X<T> { .. }`）的单元变体也返回 Generic，
                            // 实参先用 Unknown 占位，构造点从实参推断。用户泛型枚举优先于
                            // 内置 Option/Result 按名特判（shadow 语义与泛型 struct 一致）。
                            let ty = if self.generic_enums.contains_key(enum_name) {
                                let n = self.generic_enum_param_names(enum_name).len();
                                Type::Generic {
                                    base: Box::new(Type::Enum(enum_name.to_string())),
                                    args: vec![Type::Unknown; n],
                                }
                            } else {
                                match enum_name {
                                    "Option" => Type::Generic {
                                        base: Box::new(Type::Enum("Option".to_string())),
                                        args: vec![Type::Unknown],
                                    },
                                    "Result" => Type::Generic {
                                        base: Box::new(Type::Enum("Result".to_string())),
                                        args: vec![Type::Unknown, Type::str_()],
                                    },
                                    _ => Type::Enum(enum_name.to_string()),
                                }
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
                            // 阶段1-静默失败：Result/Option 显式解包原语（自由函数 native，非枚举方法）
                            | "or_die" | "assume_ok"
                            | "str_at" | "str_len" | "str_cmp" | "str_slice" | "str_add" | "str_eq" | "str_int"
                            | "Vec::new" | "HashMap::new"
                            // 问题29：智能指针构造 native（Box/Rc/Arc/Pin，返回类型见 resolve_builtin）
                            // M3.4：Weak 弱引用 native（Weak::new 构造 / weak_upgrade 升级 / 计数辅助）
                            | "Box::new" | "Rc::new" | "Arc::new" | "Pin::new"
                            | "Weak::new" | "weak_upgrade" | "weak_strong_count" | "weak_weak_count"
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
                            | "url_encode" | "url_decode"
                            // 哈希函数（SHA-256/SHA-512/MD5）
                            | "sha256" | "sha512" | "md5"
                            | "sha256_str" | "sha512_str" | "md5_str"
                            // M1.3：dyn Trait 升级 native（into_dyn(value, trait_name)）
                            // 由类型注解驱动的隐式升级在 Let 分支改写为对它的调用
                            | "into_dyn" => {
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
                        // 批次2 C：运算符重载的 trait 方法在 VM 的具体值分派。
                        // `a + b` 降级为 `add` 方法调用，但直接构造 MethodCall 会绕过
                        // MethodCall 分支的 trait 改写（VM call_method_priv 对
                        // Value::Struct 只做字段访问、不查 trait 表）。运算符→trait
                        // 映射固定（Add→add），此处按已知 trait 直接改写为对
                        // `__dyn_{trait}_{type}_{method}` 的普通 Call（与 MethodCall
                        // 分支同模式，VM/JIT/WASM/解释器四路径同通）。
                        if let Some((kind, ty)) = self.try_rewrite_trait_method(
                            &l, &method, std::slice::from_ref(&r), &span, Some(trait_name),
                        ) {
                            return Ok(HirExpr {
                                kind,
                                ty: ty.clone(),
                                span: span.clone(),
                            });
                        }
                        // AUDIT #19：trait 方法（`impl Add for Point` 的 `add`）不在
                        // inherent 方法表（methods）中，resolve_method_type 查不到会
                        // 回退 Unknown —— 单层 `a + b` 可用，但链式 `(a + b) + c` 的
                        // 复合 receiver 类型丢失为 Unknown，外层降级检查
                        // has_trait_impl_for_type("Add", Unknown) 为 false → 断链，
                        // 运行时"加法类型不匹配"。改为优先从 trait impl 方法定义
                        // 取真实返回类型（链式 receiver 保持为 Point）。
                        let ret_ty = self.trait_impl_method_ret_type(&trait_name, &l.ty, &method)
                            .unwrap_or_else(|| self.resolve_method_type(&l.ty, &method, &[r.clone()]));
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
                        // lossy lattice M1 spike：字面量/静态可判定零除数 → 编译期报错
                        Self::check_binary_static_divzero(op, &r, &span)?;
                        let ty = self.infer_binary_type(op, &l.ty, &r.ty);
                        let hir_op = lower_binop(op);
                        (HirExprKind::Binary { op: hir_op, left: Box::new(l), right: Box::new(r), ty: ty.clone() }, ty)
                    }
                } else {
                    // 编译期 shape 检查：两侧 Tensor shape 不兼容时报错
                    Self::check_binary_shape_compat(op, &l.ty, &r.ty, &span)?;
                    // lossy lattice M1 spike：字面量/静态可判定零除数 → 编译期报错
                    Self::check_binary_static_divzero(op, &r, &span)?;
                    let ty = self.infer_binary_type(op, &l.ty, &r.ty);
                    let hir_op = lower_binop(op);
                    (HirExprKind::Binary { op: hir_op, left: Box::new(l), right: Box::new(r), ty: ty.clone() }, ty)
                }
            }

            ExprKind::CustomBinary { op, left, right } => {
                // M3.1：自定义运算符 `a <op> b` → 对绑定函数的普通调用。
                // 绑定在 lower_program 第一遍注册到 custom_ops（声明先于使用），
                // 未声明即使用在此报编译期 TypeError。
                let l = self.lower_expr(left)?;
                let r = self.lower_expr(right)?;
                let fn_name = self.custom_ops.get(op).cloned().ok_or_else(|| {
                    TenthError::TypeError {
                        line: span.line,
                        col: span.col,
                        message: format!(
                            "未声明的运算符 '{}'（需先用 `operator {} = fn(...)` 声明）",
                            op, op
                        ),
                    }
                })?;
                let func = HirExpr {
                    kind: HirExprKind::Var(fn_name),
                    ty: Type::Unknown,
                    span: span.clone(),
                };
                let ret_ty = self.resolve_call_type(&func, &[l.clone(), r.clone()], &span)?;
                let ret_ty2 = ret_ty.clone();
                (HirExprKind::Call {
                    func: Box::new(func),
                    args: vec![l, r],
                    ret_ty,
                }, ret_ty2)
            }

            ExprKind::Unary { op, expr: inner } => {
                let e = self.lower_expr(inner)?;
                // 一元运算符重载检查：若类型实现了对应 trait，转为方法调用
                let (kind, ty) = if let Some((trait_name, method_name)) = self.try_unary_op_overload(op) {
                    if self.has_trait_impl_for_type(&trait_name, &e.ty) {
                        let method = method_name.to_string();
                        // 批次2 C：与二元重载同理——一元运算符（`-a` → `a.neg()`）直接
                        // 构造 MethodCall 绕过 trait 改写，此处按已知 trait 改写为
                        // `__dyn_*` 调用（四路径同通）。
                        if let Some((kind, ty)) = self.try_rewrite_trait_method(
                            &e, &method, &[], &span, Some(trait_name),
                        ) {
                            return Ok(HirExpr {
                                kind,
                                ty: ty.clone(),
                                span: span.clone(),
                            });
                        }
                        // AUDIT #19：与二元重载同因——trait 方法返回类型从 trait
                        // impl 定义取，避免回退 Unknown 使外层链式断链。
                        let ret_ty = self.trait_impl_method_ret_type(&trait_name, &e.ty, &method)
                            .unwrap_or_else(|| self.resolve_method_type(&e.ty, &method, &[]));
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
                // M2.2：Newtype（tuple struct）构造 `Meters(3.5)` / `Pair(1, "a")`。
                // parser 把 `Name(args)` 解析为 Call { func: Ident("Name"), args }，
                // 与普通函数调用在 AST 层无法区分；当 Name 是已声明的非泛型
                // tuple struct（且无同名函数遮蔽）时，改写为 StructLiteral
                // （字段名 `_0, _1, ...`，与 lower_program 的 Tuple 注册一致），
                // 与 named struct 共用运行时 Struct 表示（解释器/VM/WASM 均支持）。
                if let ExprKind::Ident(ident) = &func.kind {
                    if self.tuple_structs.contains(&ident.name)
                        && self.structs.contains_key(&ident.name)
                        && self.scope.lookup_fn(&ident.name).is_none()
                    {
                        let lowered_args = self.process_call_args(func, args, &span)?;
                        let lowered_fields: Vec<(String, HirExpr)> = lowered_args.into_iter().enumerate()
                            .map(|(i, a)| (format!("_{}", i), a))
                            .collect();
                        return Ok(HirExpr {
                            kind: HirExprKind::StructLiteral {
                                name: ident.name.clone(),
                                fields: lowered_fields,
                                has_default: false,
                            },
                            ty: Type::TypeParam { name: ident.name.clone() },
                            span,
                        });
                    }
                }

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
                        // M2.1：用户泛型枚举优先（shadow 内置 Option/Result 时走泛型推断）。
                        let ty = if self.generic_enums.contains_key(enum_name.as_str()) {
                            let arg_tys: Vec<Type> = lowered_args.iter().map(|a| a.ty.clone()).collect();
                            Type::Generic {
                                base: Box::new(Type::Enum(enum_name.clone())),
                                args: self.infer_generic_enum_args(enum_name, variant, &arg_tys),
                            }
                        } else {
                            match enum_name.as_str() {
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
                            }
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
                    // 类型参数可以是具体 BaseType（直接确定 runtime_name），
                    // 也可以是 TypeParam（出现在泛型函数体内，如 `randn<T>(...)`）。
                    // TypeParam 场景：保留原始 func_name，ret_ty 的 dtype 保留为 TypeParam，
                    // 等实例化时 substitute_expr 替换 T 为具体 BaseType 后，
                    // 由 substitute_kind_in_place 中的 native dtype 修正逻辑改写 func_name
                    // （如 F32 → randn_f32）。
                    let type_args: Vec<Type> = generics.iter()
                        .map(|ta| self.annotation_type(ta))
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
                    let lowered_args: Vec<HirExpr> = args.iter()
                        .map(|a| self.lower_expr(a))
                        .collect::<TenthResult<_>>()?;
                    let shape = Self::shape_from_int_args(&lowered_args);
                    let (runtime_name, ret_ty) = match &type_args[0] {
                        Type::Base(dtype) => {
                            // 运行时按名字分发：f32 dtype 映射到 randn_f32/zeros_f32/ones_f32/rand_f32
                            // Wave 2: f16/bf16 dtype 映射到 zeros_f16/ones_f16/zeros_bf16/ones_bf16
                            // （tensor/tensor_from_vec 不需要后缀，运行时按参数 dtype 构造）
                            let name = match (func_name.as_str(), *dtype) {
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
                            (name, Type::tensor(*dtype, shape))
                        }
                        Type::TypeParam { name } => {
                            // 泛型函数体内的 native 构造：T 未实例化，
                            // 保留原始 func_name，ret_ty.dtype 保留 TypeParam。
                            // 实例化时 substitute_expr 会替换 T，并修正 func_name。
                            let ty = Type::Tensor {
                                dtype: Box::new(Type::TypeParam { name: name.clone() }),
                                dims: shape,
                            };
                            (func_name.clone(), ty)
                        }
                        _ => {
                            return Err(TenthError::TypeError {
                                line: span.line,
                                col: span.col,
                                message: format!(
                                    "native 构造函数 '{}' 的类型参数必须是具体 BaseType 或 TypeParam，得到 {:?}",
                                    func_name, type_args[0]
                                ),
                            });
                        }
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
                    .map(|ta| self.annotation_type(ta))
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

                // 断点 4.1（符号维度 unify）：泛型实例化调用点实参代换——
                // 把实例化返回 shape 中的 `Dim::Symbol(形参名)` 代换为调用点
                // 实参推导的维度（与 resolve_call_type 的普通调用路径一致）。
                // 只代换形参名对应的 Symbol，保持保守（防误报）。
                let inst_ret_ty = {
                    let mut dim_map: HashMap<String, Dim> = HashMap::new();
                    for ((pname, _pty), arg) in template.params.iter().zip(lowered_args.iter()) {
                        dim_map.insert(pname.clone(), Self::dim_from_expr(arg));
                    }
                    Self::substitute_dims_in_type(&inst_ret_ty, &dim_map)
                };

                // Generate mangled name and instantiate if not already done
                let mangled_name: String = type_args.iter()
                    .fold(func_name.clone(), |acc, ty| format!("{}_{}", acc, ty));
                // 记录泛型实例化函数名：层 3 lossy 污点分析跳过其 body（防误报，
                // 见 Lowerer.generic_instantiations 注释）。
                self.generic_instantiations.insert(mangled_name.clone());

                // G6：泛型实例化后的形参（type_map 已代入具体类型），供实例化注册
                // 与调用点实参检查共用（必须在 already_instantiated 分支之前计算，
                // 否则已实例化的函数调用点无法拿到形参类型）。
                let inst_params: Vec<(String, Type)> = template.params.iter()
                    .map(|(n, t)| (n.clone(), substitute_type(t, &type_map)))
                    .collect();
                // G6：泛型实例化调用点参数类型检查（如 `foo<i64>("x")` 的 "x" 不匹配）
                self.check_call_arg_types(&mangled_name, &inst_params, &lowered_args, &span)?;

                let already_instantiated = self.functions.iter().any(|f| f.name == mangled_name);
                if !already_instantiated {
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
                    // G5 修复（阶段2a M2）：Generic receiver（如 `make() -> File<Open>`
                    // 的返回类型 `Type::Generic { base: TypeParam("File"), .. }`）上的
                    // 方法调用也应触发编译期改写。此前该分支缺失导致 Generic receiver
                    // 仅解释器（按运行时值名查方法表）可用，VM 路径不改写、又无运行时
                    // 查表兜底，报「没有方法」——两条路径不一致。改写条件
                    // （`__Type_method` 存在）保证只对 inherent impl 方法生效，与普通
                    // struct 行为一致；trait 方法不注册 `__` 前缀函数，天然不会命中。
                    Type::Generic { base, .. } => match base.as_ref() {
                        Type::Struct(name) | Type::TypeParam { name } => Some(name.clone()),
                        _ => None,
                    },
                    _ => None,
                };
                if let Some(type_name) = recv_type_name {
                    // 阶段2a M2（G3）：候选 mangled 名按 receiver 状态解析——
                    // 特化 impl（`impl File<Open>` → `__File_Open_read`）优先；
                    // 裸 impl（`impl File` → `__File_read`）回退，对所有状态可用
                    // （保持既有 Generic receiver 行为：G5 测试等）。
                    let mut candidates: Vec<String> = Vec::new();
                    if let Some(prefix) = super::type_mangle_prefix(&recv.ty) {
                        candidates.push(format!("__{}_{}", prefix, method.name));
                    }
                    if let Some(key) = super::type_method_key(&recv.ty) {
                        if key != type_name {
                            candidates.push(format!("__{}_{}", type_name, method.name));
                        }
                    }
                    if let Some(mangled) = candidates.iter().find(|m| {
                        self.functions.iter().any(|f| f.name == **m)
                    }) {
                        let mut all_args = vec![recv.clone()];
                        all_args.extend(lowered_args.clone());
                        // G6：用户方法调用点参数类型检查（含 receiver 自身）。
                        // self 形参类型为 TypeParam("Self")（未声明）→ 自动放行；
                        // 其余实参按形参类型逐项校验。仅 inherent 用户方法走此路径，
                        // tensor/原生方法不经过（无 __ 改写），不引入新误报面。
                        if let Some(def) = self.find_inherent_method(&recv.ty, &method.name) {
                            self.check_call_arg_types(mangled, &def.params, &all_args, &span)?;
                        }
                        let ret_ty = self.resolve_method_type(&recv.ty, &method.name, &all_args);
                        // 编译期 shape 检查（如 matmul 的内侧维度）
                        Self::check_method_shape(&recv.ty, &method.name, &all_args, &span)?;
                        // 阶段2a M2（G4）：状态转换消费式 self——若方法返回的泛型状态
                        // 与 receiver 状态不同（如 `close(self) -> File<Closed>` 之于
                        // `File<Open>`），receiver 变量按值消费，标记为 Moved，
                        // 后续使用报「使用了已移动的值」。仅当 receiver 是变量、
                        // 方法是 inherent 用户方法、且确实发生状态转换时触发——
                        // `read(self) -> str`（状态不变）不消费，保证 `f.read(); f.close()`
                        // 模式可用。不依赖运行时移动语义，属「状态参数检查」层面。
                        if let HirExprKind::Var(recv_var) = &recv.kind {
                            if let Some(def) = self.find_inherent_method(&recv.ty, &method.name) {
                                let self_by_value = def.params.first()
                                    .map(|(n, t)| n == "self" && !matches!(t, Type::Ref(..) | Type::MutRef(..)))
                                    .unwrap_or(false);
                                if self_by_value && Self::is_state_transition(&recv.ty, &def.return_type) {
                                    self.scope.try_move(recv_var);
                                }
                            }
                        }
                        let func = HirExpr {
                            kind: HirExprKind::Var(mangled.clone()),
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
                    // 批次2 C：具体值 trait 方法编译期改写（VM/JIT/WASM/解释器四路径打通）。
                    //
                    // 背景：inherent 方法有 `__{Type}_{method}` 编译期改写（上方），但 trait
                    // 方法（`impl Shape for Circle` 的 `area`）此前直接 fall-through 为带 plain
                    // 方法名的 MethodCall——VM 的 `call_method_priv` 对 Value::Struct 只做字段
                    // 访问、不查 trait 表，报「没有方法」。而 `__dyn_{Trait}_{Type}_{method}`
                    // 函数已由 lower_stmt 无条件注册进 self.functions 并被编译进 VM
                    // （dyn 路径可用即实证），缺的只是「具体值方法调用 → __dyn_*」这条路由。
                    //
                    // 改写规则（严格防静默错值）：
                    // - 按 receiver 静态类型（type_name，与 inherent 块同源）查 trait_impls，
                    //   收集所有在该类型上实现了此方法的 trait；
                    // - **恰好 1 个** trait 命中 → 改写为对 `__dyn_{trait}_{type}_{method}`
                    //   的普通 Call（与 inherent 改写同模式），四后端同通；
                    // - 0 个（无实现）或 ≥2 个（歧义）命中 → **不改写**：无匹配保持既有
                    //   fall-through 响亮报错；歧义不静默选一个，保持现状（未来增强）。
                    //
                    // 边界：
                    // - 枚举不改写（recv_type_name 不含 Enum，与解释器 Value::Enum 不分派一致）；
                    // - trait 默认方法体不注册进 trait_impls（lower_stmt 只注册 impl 显式方法），
                    //   此处查表不命中 → 不改写，保持双路径现状（能力全梳理已记录）；
                    // - 泛型 `<T>` 内具体值 trait 方法：TypeParam("T") 查表通常不命中，
                    //   不改写（VM 响亮报错，与 P3 可选运行时兜底衔接，非静默）。
                    // 具体值 trait 方法编译期改写（恰一 trait 命中 → __dyn_* Call；
                    // 0 无匹配 / ≥2 歧义 → 不改写保持现状响亮报错）。
                    if let Some((kind, ty)) = self.try_rewrite_trait_method(
                        &recv, &method.name, &lowered_args, &span, None,
                    ) {
                        return Ok(HirExpr {
                            kind,
                            ty: ty.clone(),
                            span: span.clone(),
                        });
                    }
                    // 阶段2a M2（G3）：泛型 struct 实例（如 `File<Closed>`）上调用
                    // 该方法不存在（特化与裸 impl 都没有）→ 编译期报错——
                    // 「非法状态表达不出来」（typestate 核心）。非泛型类型保持
                    // 既有行为（落到 MethodCall，运行时报错），防误报。
                    if super::is_generic_struct_instance(&recv.ty, &self.generic_structs) {
                        return Err(TenthError::TypeError {
                            line: span.line,
                            col: span.col,
                            message: format!(
                                "类型 '{}' 没有方法 '{}'（该状态不支持此操作）",
                                recv.ty, method.name
                            ),
                        });
                    }
                }

                // AUDIT-11.4.12：张量 `.shape()` 是类型系统误标——运行时无该
                // native（`x.shape()` 类型检查曾能通过但运行时崩溃），正确路径是
                // `.shape_tensor()`（返回 `Tensor[f64, ndim]`）。编译期直接报错
                // 引导用户，避免"类型检查通过、运行时崩溃"。仅对 Tensor receiver
                // 生效；用户 struct/trait 自定义 `shape` 方法不受影响。
                if matches!(&recv.ty, Type::Tensor { .. }) && method.name == "shape" {
                    return Err(TenthError::TypeError {
                        line: span.line,
                        col: span.col,
                        message: format!(
                            "张量没有方法 'shape()'——取形状请用 'shape_tensor()'（返回 Tensor[f64, ndim]）"
                        ),
                    });
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
                    // M1.2：Union 字段访问。tagged union 的字段类型从声明表解析，
                    // 运行时会检查该字段是否为当前 active 字段。
                    Type::Union(name) => {
                        self.unions.get(name)
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

                // 阶段1-静默失败（层1）：除 final_expr（块返回值，处于"使用"中）外，
                // 所有被丢弃的表达式语句若产出 Result/Option 值则 emit TenthWarning。
                // 例：`read_line();` 作为语句被丢弃 → warning（应写 or_die(值,"消息") 或 ?）。
                let discard_check_len = if final_expr.is_some() {
                    lowered_stmts.len().saturating_sub(1)
                } else {
                    lowered_stmts.len()
                };
                for s in lowered_stmts.iter().take(discard_check_len) {
                    if let HirStmtKind::Expr(e) = &s.kind {
                        self.check_silent_failure_discard(e, &s.span);
                    }
                }

                // RAII：收集当前作用域中实现了 Drop trait 的变量，在作用域退出时调用 drop()
                let drop_vars: Vec<(String, Option<String>)> = self.collect_drop_vars();

                let stmts_without_final: Vec<HirStmt> = if final_expr.is_some() {
                    lowered_stmts[..lowered_stmts.len().saturating_sub(1)].to_vec()
                } else {
                    lowered_stmts
                };

                // 构造包含 drop 调用的块
                // 块结束：弹回父作用域（块内变量不可见、不再参与外层 drop 收集）。
                // 既有实现错误地恢复到块自身作用域，导致块内变量泄漏到块外——
                // Drop 特性激活后表现为嵌套块变量在块退出后于外层作用域被二次 drop。
                self.scope = *self.scope.parent.take().unwrap();

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
                            .map(|a| self.annotation_type(a))
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

                // M2.3：闭包是独立函数体，不能跳出外层循环——清空循环标签栈，
                // 使闭包内的 `break 'x` / `continue 'x` 不会误匹配外层函数循环的标签。
                let saved_loop_labels = std::mem::take(&mut self.loop_labels);
                let b = self.lower_expr(body)?;
                self.loop_labels = saved_loop_labels;

                // 闭包体作用域结束：弹回父作用域（闭包参数/局部变量不外泄）。
                self.scope = *self.scope.parent.take().unwrap();

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

            ExprKind::StructLiteral { name, generics, fields, use_defaults } => {
                // M1.2：Union 构造。Tenth 的 union 是带 active_field 的 tagged union
                // （非 C 风格内存重叠），`MyUnion { field: value }` 恰好激活一个字段
                // → 生成 UnionLiteral，运行时构造 Value::Union { active_field, value }。
                // 声明先于表达式全部注册（lower_program 两遍），字段名校验无前向引用问题。
                if self.unions.contains_key(&name.name) {
                    if *use_defaults {
                        return Err(TenthError::TypeError {
                            line: span.line,
                            col: span.col,
                            message: format!("union '{}' 不支持默认字段填充（..）", name.name),
                        });
                    }
                    if fields.len() != 1 {
                        return Err(TenthError::TypeError {
                            line: span.line,
                            col: span.col,
                            message: format!(
                                "union '{}' 构造必须恰好激活一个字段（tagged union，实际给了 {} 个字段）",
                                name.name, fields.len()
                            ),
                        });
                    }
                    let (fname, fexpr) = &fields[0];
                    let declared = self.unions.get(&name.name)
                        .map(|fs| fs.iter().any(|(n, _)| n == &fname.name))
                        .unwrap_or(false);
                    if !declared {
                        return Err(TenthError::TypeError {
                            line: span.line,
                            col: span.col,
                            message: format!("union '{}' 没有字段 '{}'", name.name, fname.name),
                        });
                    }
                    let lowered = self.lower_expr(fexpr)?;
                    return Ok(HirExpr {
                        kind: HirExprKind::UnionLiteral {
                            name: name.name.clone(),
                            active_field: fname.name.clone(),
                            value: Box::new(lowered),
                        },
                        ty: Type::Union(name.name.clone()),
                        span: span.clone(),
                    });
                }
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

                // 阶段2a M2（G1）：泛型 struct 字面量保留类型实参（状态）。
                // 此前 `File<Open> { ... }` 的 `generics` 被直接丢弃，字面量类型
                // 固定为 TypeParam("File")，状态无法传播。现在构造
                // `Type::Generic { base: TypeParam("File"), args: [TypeParam("Open")] }`，
                // 使方法解析（G3）与返回类型传播（close 的 File<Closed>）可用。
                let struct_ty = if generics.is_empty() {
                    Type::from_annotation(&ast::TypeAnnotation::Named(ast::Ident { name: name.name.clone(), span: name.span.clone() }))
                } else {
                    let arg_tys: Vec<Type> = generics.iter()
                        .map(|g| self.annotation_type(g))
                        .collect();
                    Type::Generic {
                        base: Box::new(Type::TypeParam { name: name.name.clone() }),
                        args: arg_tys,
                    }
                };
                (HirExprKind::StructLiteral {
                    name: name.name.clone(),
                    fields: lowered_fields,
                    has_default: *use_defaults,
                }, struct_ty)
            }

            ExprKind::EnumLiteral { enum_name, variant, fields, generics } => {
                let lowered_fields: Vec<(String, HirExpr)> = fields.iter()
                    .map(|(id, e)| {
                        let lowered = self.lower_expr(e)?;
                        Ok((id.name.clone(), lowered))
                    })
                    .collect::<TenthResult<_>>()?;
                // 问题1：Option/Result 作为泛型枚举——从字段推断类型参数
                // M2.1：泛型枚举 `enum X<T> { .. }`：
                //   - 显式实参（`MyEnum<i64>::Some(5)`）优先
                //   - 否则从变体字段类型推断（`MyEnum::Some(5)`，与 Option/Result 同款）
                // 用户泛型枚举优先于内置 Option/Result 按名特判（shadow 语义）。
                let ty = if !generics.is_empty() {
                    // 显式类型实参
                    let arg_tys: Vec<Type> = generics.iter().map(|g| self.annotation_type(g)).collect();
                    if self.generic_enums.contains_key(&enum_name.name) {
                        Type::Generic {
                            base: Box::new(Type::Enum(enum_name.name.clone())),
                            args: arg_tys,
                        }
                    } else {
                        // 非泛型枚举给出显式实参：保守回退为普通枚举类型
                        Type::Enum(enum_name.name.clone())
                    }
                } else if self.generic_enums.contains_key(&enum_name.name) {
                    // M2.1：泛型枚举无显式实参 → 从变体字段类型推断
                    let arg_tys: Vec<Type> = lowered_fields.iter().map(|(_, e)| e.ty.clone()).collect();
                    Type::Generic {
                        base: Box::new(Type::Enum(enum_name.name.clone())),
                        args: self.infer_generic_enum_args(&enum_name.name, &variant.name, &arg_tys),
                    }
                } else {
                    match enum_name.name.as_str() {
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
                    }
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

                        // match arm 作用域结束：弹回父作用域（模式绑定不外泄到 match 之后）。
                        self.scope = *self.scope.parent.take().unwrap();

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
                        // M2.1：穷尽性检查同时覆盖非泛型枚举（enums）与泛型枚举（generic_enums）。
                        // 泛型枚举的变体定义（含 TypeParam 字段类型）在 generic_enums 中。
                        let variants = self.enums.get(enum_name)
                            .or_else(|| self.generic_enums.get(enum_name).map(|ge| &ge.variants));
                        if let Some(variants) = variants {
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
                let is_copy = super::is_copy_type(&ty, &self.structs, &self.trait_impls);
                if let ExprKind::Ident(ident) = &inner.kind {
                    // Copy 类型的值在 move 时不标记为 Moved（值被复制，原变量仍可用）
                    if !is_copy {
                        self.scope.set_ownership(&ident.name, Ownership::Moved);
                    }
                }
                // Copy 类型：`move x` 是值拷贝（无所有权转移），直接解包为
                // 普通表达式——解释器运行时 Move 分支会无条件把变量标为
                // Value::Moved（不感知 Copy），不解包会导致 Copy 变量在
                // 解释器路径被误判为「已移动」。VM 路径 Move 编译为值拷贝，
                // 解包前后行为一致（no-op）。
                if is_copy {
                    (e.kind, ty)
                } else {
                    (HirExprKind::Move(Box::new(e)), ty)
                }
            }

            ExprKind::Lossy(inner) => {
                // `lossy expr`：编译期构造，运行时 no-op——lower 为 HirExprKind::Lossy(inner)，
                // 各后端（bytecode/wasm）将其编译为 inner 本身。污点归零由 taint.rs 旁路分析
                // 在 lowering 完成后处理（方案 C，Type/HIR 数据结构零侵入）。
                let e = self.lower_expr(inner)?;
                let ty = e.ty.clone();
                (HirExprKind::Lossy(Box::new(e)), ty)
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

    // ── M2.1：泛型枚举辅助 ─────────────────────────────────────────────────
    // 构造/匹配路径统一从这里查询枚举定义：非泛型枚举在 `enums`，泛型枚举在
    // `generic_enums`（变体字段类型含 TypeParam，实例化时替换）。

    /// 枚举变体字段定义（含泛型枚举），找不到返回 None。
    /// 泛型枚举返回带 `TypeParam("T")` 的字段类型（未实例化）。
    pub(super) fn enum_variant_fields(&self, enum_name: &str, variant: &str) -> Option<Vec<(String, Type)>> {
        if let Some(variants) = self.enums.get(enum_name) {
            return variants.iter().find(|(v, _)| v == variant).map(|(_, f)| f.clone());
        }
        self.generic_enums.get(enum_name)
            .and_then(|ge| ge.variants.iter().find(|(v, _)| v == variant))
            .map(|(_, f)| f.clone())
    }

    /// 泛型枚举声明的类型参数名列表（非泛型枚举返回空 Vec）。
    pub(super) fn generic_enum_param_names(&self, name: &str) -> Vec<String> {
        self.generic_enums.get(name)
            .map(|ge| ge.generics.clone())
            .unwrap_or_default()
    }

    /// 从构造实参推断泛型枚举的类型实参。
    /// 对每个声明类型参数，找变体字段类型为 `TypeParam(param)`（或嵌套其中，如
    /// `Vec<T>`/`Option<T>`）的字段，取对应位置实参类型；找不到的保持 Unknown。
    fn infer_generic_enum_args(&self, enum_name: &str, variant: &str, arg_tys: &[Type]) -> Vec<Type> {
        let param_names = self.generic_enum_param_names(enum_name);
        let mut args: Vec<Type> = vec![Type::Unknown; param_names.len()];
        if let Some(vfields) = self.enum_variant_fields(enum_name, variant) {
            for (i, (_, fty)) in vfields.iter().enumerate() {
                let arg_ty = arg_tys.get(i).cloned().unwrap_or(Type::Unknown);
                for (pi, pname) in param_names.iter().enumerate() {
                    if let Some(t) = extract_param_type_from_field(fty, pname, &arg_ty) {
                        args[pi] = t;
                    }
                }
            }
        }
        args
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
                // M2.1：泛型枚举——变体字段类型中的 TypeParam 需替换为 scrutinee 的实参，
                // 否则 `match MyEnum::Some(x) { Some(v) => v + 1 }` 中 v 的类型是未实例化的 T。
                let variant_fields = self.enum_variant_fields(enum_name, variant);
                let type_map: HashMap<String, Type> = if self.generic_enums.contains_key(enum_name) {
                    let params = self.generic_enum_param_names(enum_name);
                    let args = match scrutinee_ty {
                        Type::Generic { base, args } if matches!(base.as_ref(), Type::Enum(n) if n == enum_name) => args.clone(),
                        _ => Vec::new(),
                    };
                    params.into_iter().zip(args.into_iter()).collect()
                } else {
                    HashMap::new()
                };
                let subst = |t: &Type| substitute_type(t, &type_map);

                if let Some((_fname, bname)) = field_bind {
                    let bind_ty = variant_fields.as_ref()
                        .and_then(|f| f.first())
                        .map(|(_, t)| subst(t))
                        .unwrap_or(Type::Unknown);
                    self.scope.define_var(bname.clone(), bind_ty, false);
                }
                for (i, (_, bind_name)) in tuple_binds.iter().enumerate() {
                    let bind_ty = variant_fields.as_ref()
                        .and_then(|f| f.get(i))
                        .map(|(_, t)| subst(t))
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
            // TypeParam：用户命名类型（struct/enum 字面量、类型注解）在 Type 系统
            // 中表示为 TypeParam（见 types.rs::from_annotation），与 Struct/Enum
            // 一视同仁，否则 `a + b`（a/b 为用户 struct）永远不会触发运算符重载。
            Type::TypeParam { name } => name,
            Type::Generic { base, .. } => match base.as_ref() {
                Type::Enum(name) | Type::Struct(name) | Type::TypeParam { name } => name,
                _ => return false,
            },
            _ => return false,
        };
        self.trait_impls.get(trait_name)
            .and_then(|impls| impls.get(type_name))
            .is_some()
    }

    /// 从 trait impl 方法定义取返回类型（运算符重载降级用）。
    /// trait 方法（`impl Add for Point` 的 `add`）不在 inherent 方法表
    /// （methods）中，`resolve_method_type` 查不到会回退 `Unknown`；
    /// 从 `trait_impls[trait][type][method].return_type` 取真实签名，
    /// 使链式重载 `(a + b) + c` / `-a + b` 的复合 receiver 类型保持为
    /// 具体类型，外层降级检查不断链（AUDIT #19）。
    pub(super) fn trait_impl_method_ret_type(&self, trait_name: &str, ty: &Type, method: &str) -> Option<Type> {
        let type_name = match ty {
            Type::Struct(name) | Type::Enum(name) => name,
            Type::TypeParam { name } => name,
            Type::Generic { base, .. } => match base.as_ref() {
                Type::Enum(name) | Type::Struct(name) | Type::TypeParam { name } => name,
                _ => return None,
            },
            _ => return None,
        };
        self.trait_impls.get(trait_name)
            .and_then(|impls| impls.get(type_name))
            .and_then(|methods| methods.get(method))
            .map(|def| def.return_type.clone())
    }

    /// 批次2 C：具体值 trait 方法编译期改写（与 inherent `__Type_method` 改写同模式）。
    ///
    /// 在 `__dyn_{trait}_{type}_{method}` 函数已由 lower_stmt 无条件注册的前提下，
    /// 把「具体值上的 trait 方法调用」改写为对它的普通 Call，使 VM/JIT/WASM/解释器
    /// 四路径同通（VM `call_method_priv` 对 Value::Struct 只做字段访问、不查 trait 表）。
    ///
    /// - `known_trait: Some(t)`：调用方已确定 trait（运算符重载 Add→add 固定映射），
    ///   直接按该 trait 改写（无 `__dyn_*` 函数则回退 None 不改写）；
    /// - `known_trait: None`：按 receiver 静态类型收集所有实现该方法的 trait，
    ///   **恰好 1 个**才改写；0 或 ≥2 返回 None 不改写——无匹配保持既有 fall-through
    ///   响亮报错，歧义不静默选一个（规避静默错值，歧义报错为未来增强）。
    ///
    /// 返回 `Some((HirExprKind::Call, ret_ty))`（改写成功）或 `None`（不改写）。
    fn try_rewrite_trait_method(
        &self,
        recv: &HirExpr,
        method: &str,
        args: &[HirExpr],
        span: &Span,
        known_trait: Option<&str>,
    ) -> Option<(HirExprKind, Type)> {
        // 与 inherent 块同源的 receiver 类型名（Struct/TypeParam/Generic base）。
        // 枚举不改写（recv_type_name 不含 Enum，与解释器 Value::Enum 不分派一致）。
        let type_name = match &recv.ty {
            Type::Struct(name) | Type::TypeParam { name } => name.clone(),
            Type::Generic { base, .. } => match base.as_ref() {
                Type::Struct(name) | Type::TypeParam { name } => name.clone(),
                _ => return None,
            },
            _ => return None,
        };
        // 收集命中 trait：known_trait 模式只查该 trait；否则遍历全部 trait_impls。
        let mut matching_traits: Vec<String> = Vec::new();
        match known_trait {
            Some(t) => {
                if self.trait_impls.get(t)
                    .and_then(|impls| impls.get(&type_name))
                    .and_then(|methods| methods.get(method))
                    .is_some() {
                    matching_traits.push(t.to_string());
                }
            }
            None => {
                for (trait_name, type_impls) in &self.trait_impls {
                    if type_impls.get(&type_name)
                        .and_then(|methods| methods.get(method))
                        .is_some() {
                        matching_traits.push(trait_name.clone());
                    }
                }
            }
        }
        if matching_traits.len() != 1 {
            return None;
        }
        let trait_name = &matching_traits[0];
        let mangled = format!("__dyn_{}_{}_{}", trait_name, type_name, method);
        // 防御性校验：__dyn_* 注册与 trait_impls 同源（lower_stmt），理论上必含；
        // 不含则不改写（防用户病态手写同名函数干扰路由）。
        if !self.functions.iter().any(|f| f.name == mangled) {
            return None;
        }
        let mut all_args = vec![recv.clone()];
        all_args.extend(args.iter().cloned());
        let ret_ty = self.trait_impl_method_ret_type(trait_name, &recv.ty, method)
            .unwrap_or_else(|| self.resolve_method_type(&recv.ty, method, &all_args));
        let func = HirExpr {
            kind: HirExprKind::Var(mangled),
            ty: Type::Unknown,
            span: span.clone(),
        };
        Some((
            HirExprKind::Call {
                func: Box::new(func),
                args: all_args,
                ret_ty: ret_ty.clone(),
            },
            ret_ty,
        ))
    }

    /// 收集当前作用域中所有实现了 Drop trait 的变量。
    /// 返回 (变量名, 类型名) 列表：类型名为普通命名类型（struct/enum）时
    /// 用于生成对 `__dyn_Drop_{Type}_drop` 的直接调用（VM 可执行）；
    /// 容器类型（元组/数组/Box 内嵌 Drop 类型）无独立 mangled 函数，
    /// 类型名为 None，回退为 `var.drop()` 方法调用（仅解释器可执行）。
    /// 按定义顺序的逆序返回（后定义的先 drop）。
    fn collect_drop_vars(&self) -> Vec<(String, Option<String>)> {
        // 从 scope 中获取所有变量及其类型
        let mut drop_vars: Vec<(String, Option<String>)> = Vec::new();
        self.scope.for_each_var(|name, ty| {
            if self.type_impls_drop(ty) {
                // 已移动的变量不再拥有值（所有权已转移），跳过——
                // 否则 `let b = move a;` 后 a 与 b 都会被 drop，双重释放。
                if matches!(self.scope.get_ownership(name), Some(Ownership::Moved)) {
                    return;
                }
                let type_name = match ty {
                    Type::Struct(n) | Type::Enum(n) => Some(n.clone()),
                    Type::TypeParam { name } => Some(name.clone()),
                    _ => None,
                };
                drop_vars.push((name.to_string(), type_name));
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
            // TypeParam：struct/enum 字面量与注解经 from_annotation 映射为
            // TypeParam("Name")（非 Type::Struct/Enum），必须按同名类型查
            // trait_impls["Drop"]，否则 RAII 永不触发（M2.5 缺口）。
            Type::TypeParam { name } => {
                self.trait_impls.get("Drop")
                    .and_then(|impls| impls.get(name))
                    .is_some()
            }
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
            Type::Weak(_) => false, // Weak 不持有所有权，无需 drop
            _ => false,
        }
    }

    /// 为指定的变量列表生成 drop 调用语句。
    /// 普通命名类型生成对 `__dyn_Drop_{Type}_drop` 的直接函数调用
    /// （该函数由 lower_stmt 的 trait impl 注册，解释器/VM 均可执行）；
    /// 容器类型回退为 `var.drop()` 方法调用。
    fn make_drop_stmt(drop_vars: &[(String, Option<String>)], span: crate::lexer::token::Span) -> HirStmt {
        // 将所有 drop 调用组合到一个 Expr 语句中
        // 使用 Block 按顺序执行每个 drop 调用
        let drop_calls: Vec<HirExpr> = drop_vars.iter().map(|(var_name, type_name)| {
            match type_name {
                Some(tn) => HirExpr {
                    kind: HirExprKind::Call {
                        func: Box::new(HirExpr {
                            kind: HirExprKind::Var(format!("__dyn_Drop_{}_drop", tn)),
                            ty: Type::Unknown,
                            span: span.clone(),
                        }),
                        args: vec![HirExpr {
                            kind: HirExprKind::Var(var_name.clone()),
                            ty: Type::Unknown,
                            span: span.clone(),
                        }],
                        ret_ty: Type::unit(),
                    },
                    ty: Type::unit(),
                    span: span.clone(),
                },
                None => HirExpr {
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
                },
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

    /// 阶段2a M2（G3）：按 receiver 类型查找 inherent 用户方法定义。
    ///
    /// 查找顺序（与编译期改写候选一致）：
    /// 1. 特化键（`File<Open>` → `impl File<Open>` 的方法）
    /// 2. 裸名回退（`File` → `impl File` 的方法，对所有状态可用）
    /// 3. 模块/import 场景兜底：methods 表不随 use 合并进父 lowerer，
    ///    改从 `self.functions` 中找 `__File_Open_method` mangled 函数取签名。
    ///
    /// 返回方法的 HirFnDef，使 `resolve_method_type` 能取到真实返回类型
    /// （状态传播的关键：`close(self) -> File<Closed>` 的返回类型在此取出）。
    pub(super) fn find_inherent_method(&self, receiver: &Type, method: &str) -> Option<HirFnDef> {
        if let Some(key) = super::type_method_key(receiver) {
            if let Some(def) = self.methods.get(&key).and_then(|m| m.get(method)) {
                return Some(def.clone());
            }
            if let Some(base) = super::type_base_name(receiver) {
                if base != key {
                    if let Some(def) = self.methods.get(&base).and_then(|m| m.get(method)) {
                        return Some(def.clone());
                    }
                }
            }
            if let Some(prefix) = super::type_mangle_prefix(receiver) {
                let mangled = format!("__{}_{}", prefix, method);
                if let Some(f) = self.functions.iter().find(|f| f.name == mangled) {
                    return Some(f.clone());
                }
            }
        }
        None
    }

    /// 阶段2a M2（G4）：判断方法是否「状态转换」——receiver 与返回类型都是
    /// 同一泛型 struct 的不同状态实参（如 `File<Open>` → `File<Closed>`）。
    /// 仅此类方法消费 receiver（标记 Moved）；状态不变的方法（如 `read -> str`、
    /// `touch -> File<Open>`）不消费。
    pub(super) fn is_state_transition(receiver: &Type, ret_ty: &Type) -> bool {
        let recv_key = match super::type_method_key(receiver) {
            Some(k) => k,
            None => return false,
        };
        let ret_key = match super::type_method_key(ret_ty) {
            Some(k) => k,
            None => return false,
        };
        // 只有 Generic（带实参）才可能发生状态转换；键不同即转换
        matches!(receiver, Type::Generic { .. }) && recv_key != ret_key
    }
}

/// M2.1：若 `field_ty` 中参数 `pname` 的位置与 `arg_ty` 结构对应，
/// 返回 `arg_ty` 中对应位置的类型（即泛型枚举类型实参）。
/// 覆盖：字段类型恰为 `T` → 取整个实参类型；`Vec<T>`/`Option<T>`/`[T]`/`&T` 等
/// 嵌套结构 → 递归内层（实参结构需匹配，否则返回 None 保守保持 Unknown）。
fn extract_param_type_from_field(field_ty: &Type, pname: &str, arg_ty: &Type) -> Option<Type> {
    match field_ty {
        Type::TypeParam { name } if name == pname => Some(arg_ty.clone()),
        // 容器/引用：实参需为同形结构才提取内层
        Type::Array { inner, .. } | Type::Ref(inner, _) | Type::MutRef(inner, _) => {
            match arg_ty {
                Type::Array { inner: ai, .. } | Type::Ref(ai, _) | Type::MutRef(ai, _) => {
                    extract_param_type_from_field(inner, pname, ai)
                }
                _ => None,
            }
        }
        Type::Generic { args, .. } => {
            // 实参结构：Generic(...) 取 args；Array（如 [i64] 表示 Vec<i64>）视为单元素
            let arg_args: Vec<Type> = match arg_ty {
                Type::Generic { args, .. } => args.clone(),
                Type::Array { inner, .. } => vec![(**inner).clone()],
                _ => return None,
            };
            if args.len() != arg_args.len() {
                return None;
            }
            for (ft, at) in args.iter().zip(arg_args.iter()) {
                if let Some(t) = extract_param_type_from_field(ft, pname, at) {
                    return Some(t);
                }
            }
            None
        }
        _ => None,
    }
}
