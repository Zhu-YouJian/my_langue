use crate::hir::hir::*;
use super::Lowerer;

impl Lowerer {
    /// Collect free variables referenced in an HIR expression.
    /// A variable is "free" if it is referenced via `Var(name)` but not
    /// bound by an enclosing `Let`, `For`, `Closure`, or `Assign` in the
    /// given expression subtree.  We also exclude built-in names.
    pub(super) fn free_vars_in(expr: &HirExpr) -> Vec<String> {
        let mut vars = Vec::new();
        Self::collect_free_vars(expr, &mut vars);
        vars.sort();
        vars.dedup();
        vars
    }

    pub(super) fn collect_free_vars(expr: &HirExpr, vars: &mut Vec<String>) {
        match &expr.kind {
            HirExprKind::Var(name) => {
                // Skip built-in names and qualified paths (e.g. "mod::fn")
                if name.contains("::") { return; }
                match name.as_str() {
                    "println" | "eprintln" | "tensor" | "rand" | "randn" | "randn_f32" | "rand_f32" | "zeros_f32" | "ones_f32"
                    | "read_file" | "write_file" | "str_at" | "Vec::new" | "HashMap::new"
                    | "compile_host" | "compile_program" | "write_bytes"
                    | "start_grad" | "new_grad" | "stop_grad"
                    | "param" | "backward" | "grad" | "zero_grad"
                    | "explain_error"
                    | "cross_entropy"
                    | "select"
                    | "scatter"
                    | "abs" | "sqrt" | "sin" | "cos" | "ln" | "pow"
                    | "zeros" | "ones"
                    | "save_weights" | "load_weights"
                    | "lexer_new" | "lexer_tokenize" | "parse_program"
                    | "lower_program" | "compile_to_wasm" | "self" => {}
                    _ => { vars.push(name.clone()); }
                }
            }
            HirExprKind::Literal(_) => {}
            HirExprKind::Binary { left, right, .. } => {
                Self::collect_free_vars(left, vars);
                Self::collect_free_vars(right, vars);
            }
            HirExprKind::Unary { expr, .. } => {
                Self::collect_free_vars(expr, vars);
            }
            HirExprKind::Call { func, args, .. } => {
                Self::collect_free_vars(func, vars);
                for a in args { Self::collect_free_vars(a, vars); }
            }
            HirExprKind::GenericCall { func, args, .. } => {
                Self::collect_free_vars(func, vars);
                for a in args { Self::collect_free_vars(a, vars); }
            }
            HirExprKind::MethodCall { receiver, args, .. } => {
                Self::collect_free_vars(receiver, vars);
                for a in args { Self::collect_free_vars(a, vars); }
            }
            HirExprKind::Index { target, indices } => {
                Self::collect_free_vars(target, vars);
                for idx in indices {
                    match idx {
                        crate::hir::hir::Index::Single(e) => {
                            Self::collect_free_vars(e, vars);
                        }
                        crate::hir::hir::Index::Range { start, end } => {
                            if let Some(s) = start { Self::collect_free_vars(s, vars); }
                            if let Some(e) = end { Self::collect_free_vars(e, vars); }
                        }
                        crate::hir::hir::Index::Colon => {}
                    }
                }
            }
            HirExprKind::Field { target, .. } => {
                Self::collect_free_vars(target, vars);
            }
            HirExprKind::TensorLiteral { data, .. } => {
                for row in data {
                    for e in row { Self::collect_free_vars(e, vars); }
                }
            }
            HirExprKind::ArrayLiteral { elements, .. } => {
                for e in elements { Self::collect_free_vars(e, vars); }
            }
            HirExprKind::Range { start, end, .. } => {
                if let Some(s) = start { Self::collect_free_vars(s, vars); }
                if let Some(e) = end { Self::collect_free_vars(e, vars); }
            }
            HirExprKind::If { cond, then_branch, else_branch, .. } => {
                Self::collect_free_vars(cond, vars);
                Self::collect_free_vars(then_branch, vars);
                if let Some(eb) = else_branch { Self::collect_free_vars(eb, vars); }
            }
            HirExprKind::Block { stmts, final_expr } => {
                // Track variables bound within the block
                let mut bound = Vec::new();
                for s in stmts {
                    if let HirStmtKind::Let { names, .. } = &s.kind {
                        for name in names {
                            bound.push(name.clone());
                        }
                    }
                    Self::collect_free_vars_stmt(s, vars);
                }
                if let Some(e) = final_expr { Self::collect_free_vars(e, vars); }
                // Remove variables that were bound in this block
                vars.retain(|v| !bound.contains(v));
            }
            HirExprKind::Closure { params, body, .. } => {
                // Collect all free vars in the body, then remove params
                let mut inner_vars = Vec::new();
                Self::collect_free_vars(body, &mut inner_vars);
                let param_names: Vec<String> = params.iter().map(|(n, _)| n.clone()).collect();
                inner_vars.retain(|v| !param_names.contains(v));
                vars.extend(inner_vars);
            }
            HirExprKind::Assign { target, value } => {
                // target is a variable name that is being written to — it may be
                // a free variable if it comes from an outer scope
                vars.push(target.clone());
                Self::collect_free_vars(value, vars);
            }
            HirExprKind::AssignOp { target, op: _, value } => {
                vars.push(target.clone());
                Self::collect_free_vars(value, vars);
            }
            HirExprKind::StructLiteral { fields, .. } => {
                for (_, e) in fields { Self::collect_free_vars(e, vars); }
            }
            HirExprKind::EnumLiteral { fields, .. } => {
                for (_, e) in fields { Self::collect_free_vars(e, vars); }
            }
            HirExprKind::Match { scrutinee, arms } => {
                Self::collect_free_vars(scrutinee, vars);
                for arm in arms { Self::collect_free_vars(&arm.body, vars); }
            }
            HirExprKind::Ref(inner) | HirExprKind::MutRef(inner) | HirExprKind::Deref(inner) => {
                Self::collect_free_vars(inner, vars);
            }
            HirExprKind::Move(inner) | HirExprKind::TryBlock(inner) => {
                Self::collect_free_vars(inner, vars);
            }
            HirExprKind::InterpolatedString { parts } => {
                for p in parts {
                    if let crate::hir::hir::InterpPart::Expr(name) = p {
                        vars.push(name.clone());
                    }
                }
            }
            HirExprKind::Tuple(elems) => {
                for e in elems {
                    Self::collect_free_vars(e, vars);
                }
            }
            HirExprKind::DerefAssign { target, value } | HirExprKind::DerefAssignOp { target, value, .. } => {
                Self::collect_free_vars(target, vars);
                Self::collect_free_vars(value, vars);
            }
            HirExprKind::FieldAssign { target, value, .. } => {
                Self::collect_free_vars(target, vars);
                Self::collect_free_vars(value, vars);
            }
        }
    }

    pub(super) fn collect_free_vars_stmt(stmt: &HirStmt, vars: &mut Vec<String>) {
        match &stmt.kind {
            HirStmtKind::Let { init, .. } => {
                if let Some(e) = init { Self::collect_free_vars(e, vars); }
            }
            HirStmtKind::Expr(e) => { Self::collect_free_vars(e, vars); }
            HirStmtKind::Return(e) => {
                if let Some(e) = e { Self::collect_free_vars(e, vars); }
            }
            HirStmtKind::While { cond, body } => {
                Self::collect_free_vars(cond, vars);
                Self::collect_free_vars_stmt(body, vars);
            }
            HirStmtKind::For { var, iter, body } => {
                Self::collect_free_vars(iter, vars);
                let mut inner = Vec::new();
                Self::collect_free_vars_stmt(body, &mut inner);
                inner.retain(|v| v != var);
                vars.extend(inner);
            }
            HirStmtKind::Break | HirStmtKind::Continue => {}
            HirStmtKind::Loop { body } => {
                for s in body { Self::collect_free_vars_stmt(s, vars); }
            }
        }
    }
}
