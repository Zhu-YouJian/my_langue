use std::collections::HashMap;
use crate::error::{TenthError, TenthResult};
use crate::lexer::token::Span;
use crate::parser::ast as ast;
use super::hir::*;
use super::types::*;

#[derive(Debug, Clone, Copy, PartialEq)]
enum Ownership {
    Owned,
    SharedRef(usize),
    ExclusiveRef,
    Moved,
}

struct Scope {
    variables: HashMap<String, (Type, bool)>,
    functions: HashMap<String, (Vec<(String, Type)>, Type)>,
    ownership: HashMap<String, Ownership>,
    parent: Option<Box<Scope>>,
}

impl Scope {
    fn new() -> Self {
        Scope {
            variables: HashMap::new(),
            functions: HashMap::new(),
            ownership: HashMap::new(),
            parent: None,
        }
    }

    fn with_parent(parent: Scope) -> Self {
        Scope {
            variables: HashMap::new(),
            functions: HashMap::new(),
            ownership: HashMap::new(),
            parent: Some(Box::new(parent)),
        }
    }

    fn lookup_var(&self, name: &str) -> Option<(Type, bool)> {
        if let Some(v) = self.variables.get(name) {
            return Some(v.clone());
        }
        self.parent.as_ref().and_then(|p| p.lookup_var(name))
    }

    fn get_ownership(&self, name: &str) -> Option<Ownership> {
        if let Some(o) = self.ownership.get(name) {
            return Some(*o);
        }
        self.parent.as_ref().and_then(|p| p.get_ownership(name))
    }

    fn set_ownership(&mut self, name: &str, state: Ownership) {
        self.ownership.insert(name.to_string(), state);
    }

    fn check_use(&self, name: &str) -> TenthResult<()> {
        if let Some(Ownership::Moved) = self.get_ownership(name) {
            return Err(TenthError::TypeError {
                line: 0, col: 0,
                message: format!("use of moved value '{}'", name),
            });
        }
        Ok(())
    }

    fn check_borrow_shared(&self, name: &str) -> TenthResult<()> {
        match self.get_ownership(name) {
            Some(Ownership::ExclusiveRef) => {
                // Relaxed: allow shared borrow through mutable borrow (for self-hosting)
                Ok(())
            },
            Some(Ownership::Moved) => Err(TenthError::TypeError {
                line: 0, col: 0,
                message: format!("cannot borrow moved value '{}'", name),
            }),
            _ => Ok(()),
        }
    }

    fn check_borrow_mut(&self, name: &str) -> TenthResult<()> {
        match self.get_ownership(name) {
            Some(Ownership::SharedRef(n)) if n > 0 => {
                // Relaxed: allow mutable borrow after shared (for self-hosting)
                Ok(())
            },
            Some(Ownership::ExclusiveRef) => {
                // Allow sequential mutable borrows (relaxed for self-hosting)
                Ok(())
            },
            Some(Ownership::Moved) => Err(TenthError::TypeError {
                line: 0, col: 0,
                message: format!("cannot borrow moved value '{}'", name),
            }),
            _ => Ok(()),
        }
    }

    fn define_var(&mut self, name: String, ty: Type, mutable: bool) {
        self.variables.insert(name.clone(), (ty, mutable));
        self.ownership.insert(name, Ownership::Owned);
    }

    fn define_fn(&mut self, name: String, params: Vec<(String, Type)>, ret: Type) {
        self.functions.insert(name, (params, ret));
    }

    fn lookup_fn(&self, name: &str) -> Option<(Vec<(String, Type)>, Type)> {
        if let Some(f) = self.functions.get(name) {
            return Some(f.clone());
        }
        self.parent.as_ref().and_then(|p| p.lookup_fn(name))
    }
}

pub struct Lowerer {
    scope: Scope,
    functions: Vec<HirFnDef>,
    generic_funcs: HashMap<String, HirFnDef>,
    structs: HashMap<String, Vec<(String, Type)>>,
    generic_structs: HashMap<String, HirGenericStruct>,
    enums: HashMap<String, Vec<(String, Vec<(String, Type)>)>>,
    methods: HashMap<String, HashMap<String, HirFnDef>>,
    modules: HashMap<String, HirProgram>,
    uses: Vec<(Vec<String>, String)>,
    trait_defs: HashMap<String, HirTraitDef>,
    trait_impls: HashMap<String, HashMap<String, HashMap<String, HirFnDef>>>,
}

impl Lowerer {
    pub fn new() -> Self {
        let mut scope = Scope::new();
        scope.define_fn(
            "tensor".to_string(),
            vec![("data".to_string(), Type::Unknown)],
            Type::Tensor {
                dtype: BaseType::F64,
                dims: vec![Dim::Any],
            },
        );
        let mut lowerer = Lowerer {
            scope,
            functions: Vec::new(),
            generic_funcs: HashMap::new(),
            structs: HashMap::new(),
            generic_structs: HashMap::new(),
            enums: HashMap::new(),
            methods: HashMap::new(),
            modules: HashMap::new(),
            uses: Vec::new(),
            trait_defs: HashMap::new(),
            trait_impls: HashMap::new(),
        };

        lowerer.trait_defs.insert("Display".to_string(), HirTraitDef {
            name: "Display".to_string(),
            generics: vec![],
            methods: vec![("display".to_string(), vec![("self".to_string(), Type::Unknown)], Type::str_())],
        });
        lowerer.trait_defs.insert("Eq".to_string(), HirTraitDef {
            name: "Eq".to_string(),
            generics: vec![],
            methods: vec![("eq".to_string(), vec![("self".to_string(), Type::Unknown), ("other".to_string(), Type::Unknown)], Type::bool_())],
        });
        lowerer.trait_defs.insert("Clone".to_string(), HirTraitDef {
            name: "Clone".to_string(),
            generics: vec![],
            methods: vec![("clone".to_string(), vec![("self".to_string(), Type::Unknown)], Type::Unknown)],
        });

        // Preload Option enum
        lowerer.enums.insert("Option".to_string(), vec![
            ("Some".to_string(), vec![("value".to_string(), Type::Unknown)]),
            ("None".to_string(), vec![]),
        ]);

        // Preload Result enum
        lowerer.enums.insert("Result".to_string(), vec![
            ("Ok".to_string(), vec![("value".to_string(), Type::Unknown)]),
            ("Err".to_string(), vec![("error".to_string(), Type::str_())]),
        ]);

        lowerer
    }

    fn lower_expr(&mut self, expr: &ast::Expr) -> TenthResult<HirExpr> {
        use ast::ExprKind;

        let span = expr.span.clone();

        let (kind, ty) = match &expr.kind {
            ExprKind::Literal(lit) => {
                let (hir_lit, ty) = match lit {
                    ast::Literal::Int(n) => (Literal::Int(*n), Type::i32()),
                    ast::Literal::Float(n) => (Literal::Float(*n), Type::f64()),
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
                                    ty: Type::Unknown,
                                    span,
                                });
                            }
                        }
                    }
                    (HirExprKind::Var(ident.name.clone()), Type::Unknown)
                } else {
                    self.scope.check_use(&ident.name)?;
                    let var_info = self.scope.lookup_var(&ident.name);
                    let fn_info = self.scope.lookup_fn(&ident.name);
                    if var_info.is_none() && fn_info.is_none() {
                        match ident.name.as_str() {
                            "println" | "eprintln" | "tensor" | "rand" | "randn"
                            | "read_file" | "write_file" | "str_at" | "Vec::new" | "HashMap::new"
                            | "compile_host" | "compile_program" | "write_bytes"
                            | "start_grad" | "new_grad" | "stop_grad"
                            | "param" | "backward" | "grad" | "zero_grad"
                            | "cross_entropy"
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
                };
                (HirExprKind::Unary { op: hir_op, expr: Box::new(e), ty: ty.clone() }, ty)
            }

            ExprKind::Call { func, args } => {
                let f = self.lower_expr(func)?;
                let lowered_args: Vec<_> = args.iter()
                    .map(|a| self.lower_expr(a))
                    .collect::<TenthResult<_>>()?;

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
                            message: "generic call target must be a named function".into(),
                        });
                    }
                };

                let template = self.generic_funcs.get(&func_name)
                    .ok_or_else(|| TenthError::TypeError {
                        line: span.line,
                        col: span.col,
                        message: format!("undefined generic function '{}'", func_name),
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

                (HirExprKind::GenericCall {
                    func: Box::new(HirExpr {
                        kind: HirExprKind::Var(func_name),
                        ty: Type::Unknown,
                        span: span.clone(),
                    }),
                    generics: type_args,
                    args: lowered_args,
                    ret_ty: inst_ret_ty.clone(),
                }, inst_ret_ty)
            }

            ExprKind::MethodCall { receiver, method, args } => {
                let recv = self.lower_expr(receiver)?;
                let lowered_args: Vec<_> = args.iter()
                    .map(|a| self.lower_expr(a))
                    .collect::<TenthResult<_>>()?;

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
                let ty = Type::Tensor { dtype: BaseType::F64, dims: vec![Dim::Known(rows), Dim::Known(cols)] };
                (HirExprKind::TensorLiteral { data: lowered, ty: ty.clone() }, ty)
            }

            ExprKind::ArrayLiteral(elements) => {
                let lowered: Vec<HirExpr> = elements.iter()
                    .map(|e| self.lower_expr(e))
                    .collect::<TenthResult<_>>()?;
                let ty = Type::Unknown;
                (HirExprKind::ArrayLiteral { elements: lowered, ty: ty.clone() }, ty)
            }

            ExprKind::Range { start, end, inclusive } => {
                let s = start.as_ref().map(|e| self.lower_expr(e)).transpose()?;
                let e = end.as_ref().map(|e| self.lower_expr(e)).transpose()?;
                (HirExprKind::Range { start: s.map(Box::new), end: e.map(Box::new), inclusive: *inclusive }, Type::Unknown)
            }

            ExprKind::If { cond, then_branch, else_branch } => {
                let c = self.lower_expr(cond)?;
                let t = self.lower_expr(then_branch)?;
                let e = else_branch.as_ref().map(|eb| self.lower_expr(eb)).transpose()?;
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

                let lowered_stmts: Vec<HirStmt> = stmts.iter()
                    .map(|s| self.lower_stmt(s))
                    .collect::<TenthResult<_>>()?;

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

                (HirExprKind::Closure { params: lowered_params, body: Box::new(b) }, Type::Unknown)
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
                                        BaseType::I32 | BaseType::I64 | BaseType::I8 | BaseType::I16 => HirExpr {
                                            kind: HirExprKind::Literal(Literal::Int(0)),
                                            ty: fty.clone(),
                                            span: name.span.clone(),
                                        },
                                        BaseType::F64 | BaseType::F32 | BaseType::F16 | BaseType::BF16 => HirExpr {
                                            kind: HirExprKind::Literal(Literal::Float(0.0)),
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
                                        _ => HirExpr {
                                            kind: HirExprKind::Literal(Literal::Int(0)),
                                            ty: Type::Unknown,
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
                }, Type::Unknown)
            }

            ExprKind::Match { scrutinee, arms } => {
                let lowered_scrutinee = self.lower_expr(scrutinee)?;
                let lowered_arms: Vec<HirMatchArm> = arms.iter()
                    .map(|arm| {
                        let hir_pattern = self.lower_pattern(&arm.pattern)?;

                        let arm_scope = Scope::with_parent(std::mem::replace(&mut self.scope, Scope::new()));
                        self.scope = arm_scope;

                        if let ast::Pattern::EnumVariant { field_bind, .. } = &arm.pattern {
                            if let Some((_fname, bname)) = field_bind {
                                self.scope.define_var(bname.clone(), Type::Unknown, false);
                            }
                        }

                        let body = self.lower_expr(&arm.body)?;

                        let outer_scope = std::mem::replace(&mut self.scope, Scope::new());
                        self.scope = outer_scope;

                        Ok(HirMatchArm { pattern: hir_pattern, body })
                    })
                    .collect::<TenthResult<_>>()?;
                (HirExprKind::Match {
                    scrutinee: Box::new(lowered_scrutinee),
                    arms: lowered_arms,
                }, Type::Unknown)
            }

            ExprKind::Ref(inner) => {
                if let ExprKind::Ident(ident) = &inner.kind {
                    self.scope.check_borrow_shared(&ident.name)?;
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
                    self.scope.check_borrow_mut(&ident.name)?;
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
        };

        Ok(HirExpr { kind, ty, span })
    }

    fn lower_index(&mut self, idx: &ast::IndexExpr) -> TenthResult<Index> {
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

    fn lower_pattern(&mut self, pattern: &ast::Pattern) -> TenthResult<HirPattern> {
        match pattern {
            ast::Pattern::EnumVariant { enum_name, variant, field_bind } => {
                Ok(HirPattern::EnumVariant {
                    enum_name: enum_name.clone(),
                    variant: variant.clone(),
                    field_bind: field_bind.clone(),
                })
            }
            ast::Pattern::Wildcard => Ok(HirPattern::Wildcard),
            ast::Pattern::Literal(lit) => {
                let hir_lit = match lit {
                    ast::Literal::Int(n) => Literal::Int(*n),
                    ast::Literal::Float(n) => Literal::Float(*n),
                    ast::Literal::Bool(b) => Literal::Bool(*b),
                    ast::Literal::String(s) => Literal::String(s.clone()),
                };
                Ok(HirPattern::Literal(hir_lit))
            }
        }
    }

    fn index_type(&self, base: &Type, indices: &[Index]) -> Type {
        match base {
            Type::Tensor { dtype, dims } => {
                let num_removed = indices.len();
                let remaining: Vec<Dim> = dims.iter().skip(num_removed).cloned().collect();
                if remaining.is_empty() {
                    Type::Base(dtype.clone())
                } else {
                    Type::Tensor { dtype: dtype.clone(), dims: remaining }
                }
            }
            // For non-tensor types (Vec, etc.), we don't track element types
            _ => Type::Unknown,
        }
    }

    fn infer_binary_type(&self, op: &ast::BinOp, l: &Type, r: &Type) -> Type {
        use ast::BinOp;
        match op {
            BinOp::Eq | BinOp::NotEq | BinOp::Lt | BinOp::Gt | BinOp::LtEq | BinOp::GtEq | BinOp::And | BinOp::Or => {
                Type::bool_()
            }
            BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div | BinOp::Mod => {
                match (l, r) {
                    (Type::Tensor { dtype, .. }, _) | (_, Type::Tensor { dtype, .. }) => {
                        Type::Tensor { dtype: dtype.clone(), dims: vec![Dim::Any] }
                    }
                    // Mixed int/float: promote to float
                    (Type::Base(_), Type::Base(rb)) if matches!(rb, BaseType::F16 | BaseType::F32 | BaseType::F64 | BaseType::BF16) => r.clone(),
                    (Type::Base(lb), Type::Base(_)) if matches!(lb, BaseType::F16 | BaseType::F32 | BaseType::F64 | BaseType::BF16) => l.clone(),
                    _ => l.clone(),
                }
            }
        }
    }

    /// Resolve TypeParam to Struct/Enum if the name matches a known definition.
    fn resolve_struct_type(&self, ty: Type) -> Type {
        match &ty {
            Type::TypeParam { name } => {
                if self.structs.contains_key(name) {
                    Type::Struct(name.clone())
                } else if self.enums.contains_key(name) {
                    Type::Enum(name.clone())
                } else {
                    ty
                }
            }
            _ => ty,
        }
    }

    fn resolve_call_type(&self, func: &HirExpr, args: &[HirExpr], span: &Span) -> TenthResult<Type> {
        match &func.kind {
            HirExprKind::Var(name) => {
                if let Some((params, ret)) = self.scope.lookup_fn(name) {
                    if params.len() != args.len() {
                        return Err(TenthError::TypeError {
                            line: span.line,
                            col: span.col,
                            message: format!(
                                "function '{}' expects {} arguments, got {}",
                                name, params.len(), args.len()
                            ),
                        });
                    }
                    return Ok(self.resolve_struct_type(ret));
                }
                self.resolve_builtin(name, args, span)
            }
            _ => Ok(Type::Unknown),
        }
    }

    fn resolve_method_type(&self, receiver: &Type, method: &str, _args: &[HirExpr]) -> Type {
        match receiver {
            Type::Tensor { dtype, dims } => {
                match method {
                    "sum" => {
                        if _args.iter().any(|a| matches!(&a.kind, HirExprKind::Var(_))) {
                            Type::Tensor { dtype: dtype.clone(), dims: dims.clone() }
                        } else {
                            Type::Base(dtype.clone())
                        }
                    }
                    "mean" | "max" | "min" => Type::Base(dtype.clone()),
                    "reshape" | "view" => Type::Tensor { dtype: dtype.clone(), dims: vec![Dim::Any] },
                    "flatten" => Type::Tensor { dtype: dtype.clone(), dims: vec![Dim::Any] },
                    "abs" | "sqrt" | "exp" | "log" | "relu" |
                    "sigmoid" | "tanh" | "softmax" => {
                        Type::Tensor { dtype: dtype.clone(), dims: dims.clone() }
                    }
                    _ => Type::Unknown,
                }
            }
            Type::Base(BaseType::Str) => match method {
                "len" => Type::Base(BaseType::I64),
                _ => Type::Unknown,
            },
            _ => match method {
                "len" => Type::Base(BaseType::I64),
                "push" => Type::unit(),
                "get" => Type::Unknown,
                _ => Type::Unknown,
            },
        }
    }

    fn resolve_builtin(&self, name: &str, _args: &[HirExpr], _span: &Span) -> TenthResult<Type> {
        match name {
            "println" | "eprintln" => Ok(Type::unit()),
            "tensor" => Ok(Type::Tensor { dtype: BaseType::F64, dims: vec![Dim::Any] }),
            "rand" | "randn" => Ok(Type::Tensor { dtype: BaseType::F64, dims: vec![Dim::Any] }),
            "read_file" => Ok(Type::str_()),
            "str_at" => Ok(Type::str_()),
            "write_file" => Ok(Type::unit()),
            "Vec::new" | "HashMap::new" => Ok(Type::Unknown),
            "compile_host" => Ok(Type::Base(BaseType::I32)),
            _ => Ok(Type::Unknown),
        }
    }

    fn lower_stmt(&mut self, stmt: &ast::Stmt) -> TenthResult<HirStmt> {
        use ast::StmtKind;

        let span = stmt.span.clone();

        let kind = match &stmt.kind {
            StmtKind::Let { name, type_ann, mutable, init } => {
                let lowered_init = init.as_ref().map(|i| self.lower_expr(i)).transpose()?;
                let ty = type_ann.as_ref()
                    .map(|a| Type::from_annotation(a))
                    .or_else(|| lowered_init.as_ref().map(|e| e.ty.clone()))
                    .unwrap_or(Type::Unknown);

                self.scope.define_var(name.name.clone(), ty.clone(), *mutable);

                HirStmtKind::Let {
                    name: name.name.clone(),
                    type_ann: type_ann.as_ref().map(|a| Type::from_annotation(a)),
                    mutable: *mutable,
                    init: lowered_init,
                }
            }
            StmtKind::Expr(e) => {
                HirStmtKind::Expr(self.lower_expr(e)?)
            }
            StmtKind::Return(e) => {
                HirStmtKind::Return(e.as_ref().map(|e| self.lower_expr(e)).transpose()?)
            }
            StmtKind::While { cond, body } => {
                let c = self.lower_expr(cond)?;
                let b = self.lower_stmt(body)?;
                HirStmtKind::While { cond: c, body: Box::new(b) }
            }
            StmtKind::For { var, iter, body } => {
                let it = self.lower_expr(iter)?;
                let b = self.lower_stmt(body)?;
                HirStmtKind::For { var: var.name.clone(), iter: it, body: Box::new(b) }
            }
            StmtKind::Break => HirStmtKind::Break,
            StmtKind::Continue => HirStmtKind::Continue,
            StmtKind::Loop { body } => {
                let lowered_body: Vec<HirStmt> = body.iter()
                    .map(|s| self.lower_stmt(s))
                    .collect::<TenthResult<_>>()?;
                HirStmtKind::Loop { body: lowered_body }
            }

        };

        Ok(HirStmt { kind, span })
    }

    pub fn lower_program(&mut self, program: &ast::Program) -> TenthResult<HirProgram> {
        for item in &program.items {
            match &item.kind {
                ast::ItemKind::StructDef { name, generics, fields } => {
                    let field_types: Vec<(String, Type)> = fields.iter()
                        .map(|f| (f.name.name.clone(), Type::from_annotation(&f.type_ann)))
                        .collect();
                    if generics.is_empty() {
                        self.structs.insert(name.name.clone(), field_types);
                    } else {
                        let gen_names: Vec<String> = generics.iter().map(|g| g.name.name.clone()).collect();
                        self.generic_structs.insert(name.name.clone(), HirGenericStruct {
                            name: name.name.clone(),
                            generics: gen_names,
                            fields: field_types,
                        });
                    }
                }
                ast::ItemKind::EnumDef { name, variants } => {
                    let variant_list: Vec<(String, Vec<(String, Type)>)> = variants.iter()
                        .map(|v| {
                            let fields: Vec<(String, Type)> = v.fields.iter()
                                .map(|f| (f.name.name.clone(), Type::from_annotation(&f.type_ann)))
                                .collect();
                            (v.name.name.clone(), fields)
                        })
                        .collect();
                    self.enums.insert(name.name.clone(), variant_list);
                }
                ast::ItemKind::Trait { name, generics, methods } => {
                    let gen_names: Vec<String> = generics.iter().map(|g| g.name.name.clone()).collect();
                    let method_sigs: Vec<(String, Vec<(String, Type)>, Type)> = methods.iter()
                        .map(|m| {
                            let param_types: Vec<(String, Type)> = m.params.iter()
                                .map(|p| (p.name.name.clone(), Type::from_annotation(&p.type_ann)))
                                .collect();
                            let ret_ty = m.return_type.as_ref()
                                .map(|rt| Type::from_annotation(rt))
                                .unwrap_or(Type::unit());
                            (m.name.name.clone(), param_types, ret_ty)
                        })
                        .collect();
                    self.trait_defs.insert(name.name.clone(), HirTraitDef {
                        name: name.name.clone(),
                        generics: gen_names,
                        methods: method_sigs,
                    });
                }
                ast::ItemKind::Function { name, generics, params, return_type, .. } => {
                    if name.name == "<expr>" || !generics.is_empty() {
                        continue;
                    }
                    let param_types: Vec<(String, Type)> = params.iter()
                        .map(|p| (p.name.name.clone(), Type::from_annotation(&p.type_ann)))
                        .collect();
                    let ret_ty = return_type.as_ref()
                        .map(|rt| Type::from_annotation(rt))
                        .unwrap_or(Type::unit());
                    let ret_ty = self.resolve_struct_type(ret_ty);
                    self.scope.define_fn(name.name.clone(), param_types, ret_ty);
                }
                _ => {}
            }
        }

        for item in &program.items {
            match &item.kind {
                ast::ItemKind::Impl { type_name, trait_name, functions, .. } => {
                    if let Some(trait_name) = trait_name {
                        let trait_name_str = trait_name.name.clone();
                        let type_name_str = type_name.name.clone();
                        let mut method_map = HashMap::new();
                        for fn_item in functions {
                            if let ast::ItemKind::Function { name, generics, params, return_type, body } = &fn_item.kind {
                                let gen_names: Vec<String> = generics.iter().map(|g| g.name.name.clone()).collect();
                                let param_types: Vec<(String, Type)> = params.iter()
                                    .map(|p| (p.name.name.clone(), Type::from_annotation(&p.type_ann)))
                                    .collect();
                                let ret_ty = return_type.as_ref()
                                    .map(|rt| Type::from_annotation(rt))
                                    .unwrap_or(Type::unit());

                                let body_scope = Scope::with_parent(std::mem::replace(&mut self.scope, Scope::new()));
                                self.scope = body_scope;

                                for (n, t) in &param_types {
                                    self.scope.define_var(n.clone(), t.clone(), false);
                                }

                                let lowered_body = self.lower_expr(body)?;

                                let outer_scope = std::mem::replace(&mut self.scope, Scope::new());
                                self.scope = outer_scope;

                                let fn_def = HirFnDef {
                                    name: name.name.clone(),
                                    generics: gen_names,
                                    generics_bounds: build_generics_bounds(generics),
                                    params: param_types,
                                    return_type: ret_ty,
                                    body: lowered_body,
                                    span: fn_item.span.clone(),
                                };
                                method_map.insert(fn_def.name.clone(), fn_def);
                            }
                        }
                        self.trait_impls.entry(trait_name_str)
                            .or_insert_with(HashMap::new)
                            .insert(type_name_str, method_map);
                    } else {
                        let mut method_map = HashMap::new();
                        for fn_item in functions {
                            if let ast::ItemKind::Function { name, generics, params, return_type, body } = &fn_item.kind {
                                let gen_names: Vec<String> = generics.iter().map(|g| g.name.name.clone()).collect();
                                let param_types: Vec<(String, Type)> = params.iter()
                                    .map(|p| (p.name.name.clone(), Type::from_annotation(&p.type_ann)))
                                    .collect();
                                let ret_ty = return_type.as_ref()
                                    .map(|rt| Type::from_annotation(rt))
                                    .unwrap_or(Type::unit());

                                let body_scope = Scope::with_parent(std::mem::replace(&mut self.scope, Scope::new()));
                                self.scope = body_scope;

                                for (n, t) in &param_types {
                                    self.scope.define_var(n.clone(), t.clone(), false);
                                }

                                let lowered_body = self.lower_expr(body)?;

                                let outer_scope = std::mem::replace(&mut self.scope, Scope::new());
                                self.scope = outer_scope;

                                let fn_def = HirFnDef {
                                    name: name.name.clone(),
                                    generics: gen_names,
                                    generics_bounds: build_generics_bounds(generics),
                                    params: param_types,
                                    return_type: ret_ty,
                                    body: lowered_body,
                                    span: fn_item.span.clone(),
                                };
                                method_map.insert(fn_def.name.clone(), fn_def);
                            }
                        }
                        self.methods.insert(type_name.name.clone(), method_map);
                    }
                }
                ast::ItemKind::Mod { name, items } => {
                    let mut lowerer = Lowerer::new();
                    lowerer.structs = self.structs.clone();
                    lowerer.enums = self.enums.clone();
                    lowerer.methods = self.methods.clone();
                    lowerer.generic_funcs = self.generic_funcs.clone();
                    lowerer.generic_structs = self.generic_structs.clone();
                    lowerer.trait_defs = self.trait_defs.clone();
                    lowerer.trait_impls = self.trait_impls.clone();
                    let mod_program = ast::Program { items: items.clone() };
                    let hir_mod = lowerer.lower_program(&mod_program)?;
                    self.modules.insert(name.name.clone(), hir_mod);
                }
                ast::ItemKind::Use { path } => {
                    let path_strs: Vec<String> = path.iter().map(|p| p.name.clone()).collect();
                    if path_strs.len() >= 2 {
                        let alias = path_strs.last().cloned().unwrap_or_default();
                        self.uses.push((path_strs.clone(), alias.clone()));
                        let mod_name = &path_strs[0];
                        let fn_name = &path_strs[1];
                        if let Some(module) = self.modules.get(mod_name) {
                            if let Some(fn_def) = module.functions.iter().find(|f| &f.name == fn_name) {
                                let param_types = fn_def.params.clone();
                                let ret_ty = fn_def.return_type.clone();
                                self.scope.define_fn(alias, param_types, ret_ty);
                            }
                        }
                    }
                }
                ast::ItemKind::Function { name, generics, params, return_type, body } => {
                    if name.name == "<expr>" {
                        continue;
                    }
                    let gen_names: Vec<String> = generics.iter().map(|g| g.name.name.clone()).collect();
                    let param_types: Vec<(String, Type)> = params.iter()
                        .map(|p| (p.name.name.clone(), Type::from_annotation(&p.type_ann)))
                        .collect();
                    let ret_ty = return_type.as_ref()
                        .map(|rt| Type::from_annotation(rt))
                        .unwrap_or(Type::unit());

                    let body_scope = Scope::with_parent(std::mem::replace(&mut self.scope, Scope::new()));
                    self.scope = body_scope;

                    for (n, t) in &param_types {
                        self.scope.define_var(n.clone(), t.clone(), false);
                    }

                    let lowered_body = self.lower_expr(body)?;

                    let outer_scope = std::mem::replace(&mut self.scope, Scope::new());
                    self.scope = outer_scope;

                    let fn_def = HirFnDef {
                        name: name.name.clone(),
                        generics: gen_names,
                        generics_bounds: build_generics_bounds(generics),
                        params: param_types,
                        return_type: ret_ty,
                        body: lowered_body,
                        span: item.span.clone(),
                    };

                    if fn_def.generics.is_empty() {
                        self.functions.push(fn_def);
                    } else {
                        self.generic_funcs.insert(fn_def.name.clone(), fn_def);
                    }
                }
                _ => {}
            }
        }

        let mut main_expr = None;
        for item in &program.items {
            if let ast::ItemKind::Function { name, body, .. } = &item.kind {
                if name.name == "<expr>" {
                    let body_scope = Scope::with_parent(std::mem::replace(&mut self.scope, Scope::new()));
                    self.scope = body_scope;

                    let lowered_body = self.lower_expr(body)?;

                    let outer_scope = std::mem::replace(&mut self.scope, Scope::new());
                    self.scope = outer_scope;

                    main_expr = Some(lowered_body);
                    break;
                }
            }
        }

        Ok(HirProgram {
            functions: self.functions.clone(),
            generic_funcs: self.generic_funcs.values().cloned().collect(),
            main_expr,
            modules: self.modules.clone(),
            uses: self.uses.clone(),
            methods: self.methods.clone(),
            structs: self.structs.clone(),
            generic_structs: self.generic_structs.clone(),
            enums: self.enums.clone(),
            trait_defs: self.trait_defs.clone(),
            trait_impls: self.trait_impls.clone(),
        })
    }
}

fn substitute_type(ty: &Type, map: &HashMap<String, Type>) -> Type {
    match ty {
        Type::TypeParam { name } => {
            map.get(name).cloned().unwrap_or_else(|| ty.clone())
        }
        Type::Ref(inner) => Type::Ref(Box::new(substitute_type(inner, map))),
        Type::MutRef(inner) => Type::MutRef(Box::new(substitute_type(inner, map))),
        _ => ty.clone(),
    }
}

fn build_generics_bounds(generics: &[ast::GenericParam]) -> HashMap<String, Vec<String>> {
    let mut bounds_map = HashMap::new();
    for gp in generics {
        if !gp.bounds.is_empty() {
            bounds_map.insert(gp.name.name.clone(), gp.bounds.iter().map(|b| b.name.clone()).collect());
        }
    }
    bounds_map
}

fn lower_binop(op: &ast::BinOp) -> BinOp {
    match op {
        ast::BinOp::Add => BinOp::Add, ast::BinOp::Sub => BinOp::Sub,
        ast::BinOp::Mul => BinOp::Mul, ast::BinOp::Div => BinOp::Div,
        ast::BinOp::Mod => BinOp::Mod, ast::BinOp::Eq => BinOp::Eq,
        ast::BinOp::NotEq => BinOp::NotEq, ast::BinOp::Lt => BinOp::Lt,
        ast::BinOp::Gt => BinOp::Gt, ast::BinOp::LtEq => BinOp::LtEq,
        ast::BinOp::GtEq => BinOp::GtEq, ast::BinOp::And => BinOp::And,
        ast::BinOp::Or => BinOp::Or,
    }
}