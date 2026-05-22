use std::collections::HashMap;
use crate::error::{TenthError, TenthResult};
use crate::lexer::token::Span;
use crate::parser::ast as ast;
use super::hir::*;
use super::types::*;

struct Scope {
    variables: HashMap<String, (Type, bool)>,
    functions: HashMap<String, (Vec<(String, Type)>, Type)>,
    parent: Option<Box<Scope>>,
}

impl Scope {
    fn new() -> Self {
        Scope {
            variables: HashMap::new(),
            functions: HashMap::new(),
            parent: None,
        }
    }

    fn with_parent(parent: Scope) -> Self {
        Scope {
            variables: HashMap::new(),
            functions: HashMap::new(),
            parent: Some(Box::new(parent)),
        }
    }

    fn lookup_var(&self, name: &str) -> Option<(Type, bool)> {
        if let Some(v) = self.variables.get(name) {
            return Some(v.clone());
        }
        self.parent.as_ref().and_then(|p| p.lookup_var(name))
    }

    fn define_var(&mut self, name: String, ty: Type, mutable: bool) {
        self.variables.insert(name, (ty, mutable));
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
        Lowerer {
            scope,
            functions: Vec::new(),
        }
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
                let var_info = self.scope.lookup_var(&ident.name);
                let fn_info = self.scope.lookup_fn(&ident.name);
                if var_info.is_none() && fn_info.is_none() {
                    return Err(TenthError::TypeError {
                        line: span.line,
                        col: span.col,
                        message: format!("undefined variable '{}'", ident.name),
                    });
                }
                let ty = var_info.map(|v| v.0).or_else(|| {
                    fn_info.map(|f| Type::FnType {
                        params: f.0.iter().map(|(_, t)| t.clone()).collect(),
                        ret: Box::new(f.1),
                    })
                }).unwrap_or(Type::Unknown);
                (HirExprKind::Var(ident.name.clone()), ty)
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
                (HirExprKind::Field { target: Box::new(t), field: field.name.clone() }, Type::Unknown)
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
                let b = self.lower_expr(body)?;
                (HirExprKind::Closure { params: lowered_params, body: Box::new(b) }, Type::Unknown)
            }

            ExprKind::Assign { target, value } => {
                let v = self.lower_expr(value)?;
                let name = match &target.kind {
                    ExprKind::Ident(id) => id.name.clone(),
                    _ => {
                        // allow shadowing for simplicity
                        return Err(TenthError::ParseError {
                            line: span.line,
                            col: span.col,
                            message: "invalid assignment target".into(),
                        });
                    }
                };
                // define/update variable
                self.scope.define_var(name.clone(), v.ty.clone(), true);
                (HirExprKind::Assign { target: name, value: Box::new(v) }, Type::unit())
            }

            ExprKind::AssignOp { target, op, value } => {
                let v = self.lower_expr(value)?;
                let name = match &target.kind {
                    ExprKind::Ident(id) => id.name.clone(),
                    _ => return Err(TenthError::ParseError {
                        line: span.line,
                        col: span.col,
                        message: "invalid assignment target".into(),
                    }),
                };
                let hir_op = lower_binop(op);
                (HirExprKind::AssignOp { target: name, op: hir_op, value: Box::new(v) }, Type::unit())
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
            _ => base.clone(),
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
                    _ => l.clone(),
                }
            }
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
                    return Ok(ret);
                }
                self.resolve_builtin(name, args, span)
            }
            _ => Ok(Type::Unknown),
        }
    }

    fn resolve_method_type(&self, receiver: &Type, method: &str, args: &[HirExpr]) -> Type {
        match receiver {
            Type::Tensor { dtype, dims } => {
                match method {
                    "sum" => {
                        if args.iter().any(|a| matches!(&a.kind, HirExprKind::Var(_))) {
                            Type::Tensor { dtype: dtype.clone(), dims: dims.clone() }
                        } else {
                            Type::Base(dtype.clone())
                        }
                    }
                    "mean" | "max" | "min" => Type::Base(dtype.clone()),
                    "reshape" | "view" => Type::Tensor { dtype: dtype.clone(), dims: vec![Dim::Any] },
                    "flatten" => Type::Tensor { dtype: dtype.clone(), dims: vec![Dim::Any] },
                    "abs" | "sqrt" | "exp" | "log" | "relu" | "sigmoid" | "tanh" => {
                        Type::Tensor { dtype: dtype.clone(), dims: dims.clone() }
                    }
                    _ => Type::Unknown,
                }
            }
            _ => Type::Unknown,
        }
    }

    fn resolve_builtin(&self, name: &str, _args: &[HirExpr], _span: &Span) -> TenthResult<Type> {
        match name {
            "println" | "eprintln" => Ok(Type::unit()),
            "tensor" => Ok(Type::Tensor { dtype: BaseType::F64, dims: vec![Dim::Any] }),
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
            _ => {
                return Err(TenthError::ParseError {
                    line: span.line,
                    col: span.col,
                    message: "unsupported statement in lowering".into(),
                });
            }
        };

        Ok(HirStmt { kind, span })
    }

    pub fn lower_program(&mut self, program: &ast::Program) -> TenthResult<HirProgram> {
        for item in &program.items {
            if let ast::ItemKind::Function { name, params, return_type, .. } = &item.kind {
                if name.name == "<expr>" {
                    continue;
                }
                let param_types: Vec<(String, Type)> = params.iter()
                    .map(|p| (p.name.name.clone(), Type::from_annotation(&p.type_ann)))
                    .collect();
                let ret_ty = return_type.as_ref()
                    .map(|rt| Type::from_annotation(rt))
                    .unwrap_or(Type::unit());
                self.scope.define_fn(name.name.clone(), param_types, ret_ty);
            }
        }

        let mut main_expr = None;

        for item in &program.items {
            if let ast::ItemKind::Function { name, params, return_type, body } = &item.kind {
                if name.name == "<expr>" {
                    let body_scope = Scope::with_parent(std::mem::replace(&mut self.scope, Scope::new()));
                    self.scope = body_scope;

                    let lowered_body = self.lower_expr(body)?;

                    let outer_scope = std::mem::replace(&mut self.scope, Scope::new());
                    self.scope = outer_scope;

                    main_expr = Some(lowered_body);
                    continue;
                }

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

                self.functions.push(HirFnDef {
                    name: name.name.clone(),
                    params: param_types,
                    return_type: ret_ty,
                    body: lowered_body,
                    span: item.span.clone(),
                });
            } else {
                let body = match &item.kind {
                    ast::ItemKind::Function { body, .. } => body.clone(),
                    _ => continue,
                };
                let main_expr_val = self.lower_expr(&body)?;
                return Ok(HirProgram {
                    functions: self.functions.clone(),
                    main_expr: Some(main_expr_val),
                });
            }
        }

        Ok(HirProgram {
            functions: self.functions.clone(),
            main_expr,
        })
    }
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