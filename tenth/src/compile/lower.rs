use std::collections::HashMap;
use crate::hir::hir::*;
use crate::hir::types::Type;
use crate::error::{TenthError, TenthResult};
use super::mir::*;

pub struct MirLowerer {
    local_counter: usize,
    block_counter: usize,
    locals: HashMap<String, MirLocal>,
}

impl MirLowerer {
    pub fn new() -> Self {
        MirLowerer { local_counter: 0, block_counter: 0, locals: HashMap::new() }
    }

    pub fn lower_program(&mut self, program: &HirProgram) -> TenthResult<MirProgram> {
        let mut functions = Vec::new();
        for func in &program.functions {
            functions.push(self.lower_function(func)?);
        }
        let main_expr = program.main_expr.as_ref().map(|e| self.lower_top_level_expr(e)).transpose()?;
        let mut struct_defs = Vec::new();
        for (name, fields) in &program.structs {
            struct_defs.push((name.clone(), fields.clone()));
        }
        Ok(MirProgram { functions, main_expr, struct_defs })
    }

    fn lower_function(&mut self, func: &HirFnDef) -> TenthResult<MirFunction> {
        self.locals.clear(); self.local_counter = 0; self.block_counter = 0;
        for (name, ty) in &func.params {
            self.locals.insert(name.clone(), MirLocal { name: name.clone(), ty: ty.clone(), mutable: false });
        }
        let body_block = self.new_block();
        let (stmts, ret_val) = self.lower_expr_to_block(&func.body)?;
        let terminator = match ret_val { Some(v) => MirTerminator::Return(Some(v)), None => MirTerminator::Return(None) };
        Ok(MirFunction {
            name: func.name.clone(), params: func.params.clone(), return_type: func.return_type.clone(),
            locals: self.locals.values().cloned().collect(), blocks: vec![BasicBlock { id: body_block, stmts, terminator }],
        })
    }

    fn lower_top_level_expr(&mut self, expr: &HirExpr) -> TenthResult<MirFunction> {
        self.locals.clear(); self.local_counter = 0; self.block_counter = 0;
        let (stmts, ret_val) = self.lower_expr_to_block(expr)?;
        Ok(MirFunction {
            name: "main".to_string(), params: Vec::new(), return_type: Type::i32(),
            locals: self.locals.values().cloned().collect(),
            blocks: vec![BasicBlock { id: 0, stmts, terminator: match ret_val {
                Some(v) => MirTerminator::Return(Some(v)),
                None => MirTerminator::Return(Some(rv(Type::i32(), MirRvalueKind::Literal(LiteralValue::Int(0))))),
            }}],
        })
    }

    fn lower_expr_to_block(&mut self, expr: &HirExpr) -> TenthResult<(Vec<MirStmt>, Option<MirRvalue>)> {
        let ty = expr.ty.clone();
        match &expr.kind {
            HirExprKind::Literal(lit) => Ok((vec![], Some(rv(ty, MirRvalueKind::Literal(lower_literal_val(lit)))))),

            HirExprKind::Var(name) => {
                // If the variable's type is a reference, generate Deref for field access
                if matches!(&ty, Type::Ref(_) | Type::MutRef(_)) {
                    Ok((vec![], Some(rv(ty.clone(), MirRvalueKind::Deref(name.clone())))))
                } else {
                    Ok((vec![], Some(rv(ty, MirRvalueKind::Use(name.clone())))))
                }
            }

            HirExprKind::Binary { op, left, right, .. } => {
                let l = self.lower_expr_rvalue(left)?;
                let r = self.lower_expr_rvalue(right)?;
                let mut stmts = Vec::new(); stmts.extend(l.0); stmts.extend(r.0);
                Ok((stmts, Some(rv(ty, MirRvalueKind::BinaryOp(op.clone(), Box::new(l.1), Box::new(r.1))))))
            }

            HirExprKind::Unary { op, expr: inner, .. } => {
                let (s, val) = self.lower_expr_rvalue(inner)?;
                Ok((s, Some(rv(ty, MirRvalueKind::UnaryOp(op.clone(), Box::new(val))))))
            }

            HirExprKind::Call { func, args, .. } => {
                let func_name = match &func.kind { HirExprKind::Var(n) => n.clone(), _ => return Err(TenthError::RuntimeError { message: "can only call named functions".into() }) };
                let mut stmts = Vec::new(); let mut arg_vals = Vec::new();
                for a in args { let (s, v) = self.lower_expr_rvalue(a)?; stmts.extend(s); arg_vals.push(v); }
                Ok((stmts, Some(rv(ty, MirRvalueKind::Call { func: func_name, args: arg_vals }))))
            }

            HirExprKind::Block { stmts: block_stmts, final_expr } => {
                let mut stmts = Vec::new();
                for s in block_stmts {
                    match &s.kind {
                        HirStmtKind::Let { name, init, .. } => {
                            if let Some(init) = init {
                                let init_ty = init.ty.clone();
                                let (s2, val) = self.lower_expr_rvalue(init)?;
                                stmts.extend(s2);
                                stmts.push(MirStmt::Let { name: name.clone(), ty: init_ty, value: val });
                            }
                        }
                        HirStmtKind::Expr(e) => {
                            let (mut s2, val) = self.lower_expr_to_block(e)?;
                            stmts.append(&mut s2);
                            if let Some(v) = val { stmts.push(MirStmt::Expr(v)); }
                        }
                        _ => {}
                    }
                }
                let ret = final_expr.as_ref().map(|e| self.lower_expr_rvalue(e)).transpose()?;
                if let Some((s2, val)) = ret { stmts.extend(s2); Ok((stmts, Some(val))) }
                else { Ok((stmts, None)) }
            }

            HirExprKind::If { cond, then_branch, else_branch, .. } => {
                // Lower condition and branches
                let (cs, cv) = self.lower_expr_rvalue(cond)?;
                let (ts, tv) = self.lower_expr_to_block(then_branch)?;
                let (es, ev) = else_branch.as_ref()
                    .map(|e| self.lower_expr_to_block(e))
                    .transpose()?
                    .unwrap_or((vec![], None));
                let mut stmts = cs;
                // If branches have side-effect statements, use IfElse
                if !ts.is_empty() || !es.is_empty() {
                    stmts.push(MirStmt::IfElse {
                        cond: cv,
                        then_body: ts,
                        else_body: es,
                    });
                    // The result is the then-value or else-value (if any), else unit
                    Ok((stmts, tv.or(ev)))
                } else {
                    // Pure expression branches — use ternary IfExpr
                    Ok(match (tv, ev) {
                        (Some(tv), Some(ev)) => {
                            (stmts, Some(rv(ty.clone(), MirRvalueKind::IfExpr { cond: Box::new(cv), then_val: Box::new(tv), else_val: Box::new(ev) })))
                        }
                        (Some(tv), None) => (stmts, Some(tv)),
                        _ => (stmts, None),
                    })
                }
            }

            HirExprKind::Assign { target, value } => {
                let (s, val) = self.lower_expr_rvalue(value)?;
                let mut stmts = s;
                stmts.push(MirStmt::Assign { name: target.clone(), value: val });
                Ok((stmts, None))
            }

            HirExprKind::Field { target, field } => {
                let (mut s, t) = self.lower_expr_rvalue(target)?;
                Ok((s, Some(rv(ty, MirRvalueKind::Field { target: Box::new(t), field: field.clone() }))))
            }

            HirExprKind::StructLiteral { name, fields } => {
                let mut stmts = Vec::new(); let mut mf = Vec::new();
                for (fnm, fe) in fields { let (s, v) = self.lower_expr_rvalue(fe)?; stmts.extend(s); mf.push((fnm.clone(), v)); }
                Ok((stmts, Some(rv(ty, MirRvalueKind::StructLiteral { name: name.clone(), fields: mf }))))
            }

            _ => Ok((vec![], Some(rv(ty, MirRvalueKind::Literal(LiteralValue::Int(0)))))),
        }
    }

    fn lower_expr_rvalue(&mut self, expr: &HirExpr) -> TenthResult<(Vec<MirStmt>, MirRvalue)> {
        let ty = expr.ty.clone();
        match &expr.kind {
            HirExprKind::Block { .. } | HirExprKind::If { .. } => {
                let tmp = self.new_local("tmp", ty.clone());
                let (stmts, val) = self.lower_expr_to_block(expr)?;
                if let Some(v) = val {
                    let mut s = stmts; s.push(MirStmt::Let { name: tmp.clone(), ty, value: v });
                    Ok((s, rv(Type::Unknown, MirRvalueKind::Use(tmp))))
                } else {
                    Ok((stmts, rv(ty, MirRvalueKind::Literal(LiteralValue::Int(0)))))
                }
            }
            _ => {
                let (stmts, val) = self.lower_expr_to_block(expr)?;
                Ok((stmts, val.unwrap_or(rv(ty, MirRvalueKind::Literal(LiteralValue::Int(0))))))
            }
        }
    }

    fn new_block(&mut self) -> usize { let id = self.block_counter; self.block_counter += 1; id }
    fn new_local(&mut self, prefix: &str, ty: Type) -> String {
        let name = format!("{}_{}", prefix, self.local_counter); self.local_counter += 1;
        self.locals.insert(name.clone(), MirLocal { name: name.clone(), ty, mutable: false }); name
    }
}

fn rv(ty: Type, kind: MirRvalueKind) -> MirRvalue { MirRvalue { kind, ty } }

fn lower_literal_val(lit: &Literal) -> LiteralValue {
    match lit {
        Literal::Int(n) => LiteralValue::Int(*n),
        Literal::Float(n) => LiteralValue::Float(*n),
        Literal::Bool(b) => LiteralValue::Bool(*b),
        Literal::String(s) => LiteralValue::Str(s.clone()),
    }
}
