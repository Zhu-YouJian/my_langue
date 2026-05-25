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
        MirLowerer {
            local_counter: 0,
            block_counter: 0,
            locals: HashMap::new(),
        }
    }

    pub fn lower_program(&mut self, program: &HirProgram) -> TenthResult<MirProgram> {
        let mut functions = Vec::new();

        for func in &program.functions {
            let mir_func = self.lower_function(func)?;
            functions.push(mir_func);
        }

        let main_expr = program.main_expr.as_ref().map(|expr| {
            self.lower_top_level_expr(expr)
        }).transpose()?;

        Ok(MirProgram { functions, main_expr })
    }

    fn lower_function(&mut self, func: &HirFnDef) -> TenthResult<MirFunction> {
        self.locals.clear();
        self.local_counter = 0;
        self.block_counter = 0;

        for (name, ty) in &func.params {
            self.locals.insert(name.clone(), MirLocal {
                name: name.clone(),
                ty: ty.clone(),
                mutable: false,
            });
        }

        let body_block = self.new_block();
        let (stmts, ret_val) = self.lower_expr_to_block(&func.body)?;

        let terminator = match ret_val {
            Some(val) => MirTerminator::Return(Some(val)),
            None => MirTerminator::Return(None),
        };

        let blocks = vec![BasicBlock { id: body_block, stmts, terminator }];

        Ok(MirFunction {
            name: func.name.clone(),
            params: func.params.clone(),
            return_type: func.return_type.clone(),
            locals: self.locals.values().cloned().collect(),
            blocks,
        })
    }

    fn lower_top_level_expr(&mut self, expr: &HirExpr) -> TenthResult<MirFunction> {
        self.locals.clear();
        self.local_counter = 0;
        self.block_counter = 0;

        let (stmts, ret_val) = self.lower_expr_to_block(expr)?;

        Ok(MirFunction {
            name: "main".to_string(),
            params: Vec::new(),
            return_type: Type::i32(),
            locals: self.locals.values().cloned().collect(),
            blocks: vec![BasicBlock {
                id: 0,
                stmts,
                terminator: match ret_val {
                    Some(v) => MirTerminator::Return(Some(v)),
                    None => MirTerminator::Return(Some(MirRvalue::Literal(LiteralValue::Int(0)))),
                },
            }],
        })
    }

    fn lower_expr_to_block(&mut self, expr: &HirExpr) -> TenthResult<(Vec<MirStmt>, Option<MirRvalue>)> {
        match &expr.kind {
            HirExprKind::Literal(lit) => Ok((vec![], Some(self.lower_literal(lit)))),

            HirExprKind::Var(name) => Ok((vec![], Some(MirRvalue::Use(name.clone())))),

            HirExprKind::Binary { op, left, right, .. } => {
                let l = self.lower_expr_rvalue(left)?;
                let r = self.lower_expr_rvalue(right)?;
                let mut stmts = Vec::new();
                stmts.extend(l.0);
                stmts.extend(r.0);
                Ok((stmts, Some(MirRvalue::BinaryOp(op.clone(), Box::new(l.1), Box::new(r.1)))))
            }

            HirExprKind::Unary { op, expr: inner, .. } => {
                let (stmts, val) = self.lower_expr_rvalue(inner)?;
                Ok((stmts, Some(MirRvalue::UnaryOp(op.clone(), Box::new(val)))))
            }

            HirExprKind::Call { func, args, .. } => {
                let func_name = match &func.kind {
                    HirExprKind::Var(name) => name.clone(),
                    _ => return Err(TenthError::RuntimeError {
                        message: "can only call named functions".into(),
                    }),
                };
                let mut stmts = Vec::new();
                let mut arg_vals = Vec::new();
                for a in args {
                    let (s, v) = self.lower_expr_rvalue(a)?;
                    stmts.extend(s);
                    arg_vals.push(v);
                }
                Ok((stmts, Some(MirRvalue::Call { func: func_name, args: arg_vals })))
            }

            HirExprKind::Block { stmts: block_stmts, final_expr } => {
                let mut stmts = Vec::new();
                for s in block_stmts {
                    match &s.kind {
                        HirStmtKind::Let { name, init, .. } => {
                            if let Some(init) = init {
                                let (s2, val) = self.lower_expr_rvalue(init)?;
                                stmts.extend(s2);
                                stmts.push(MirStmt::Let {
                                    name: name.clone(),
                                    ty: Type::Unknown,
                                    value: val,
                                });
                            }
                        }
                        HirStmtKind::Expr(e) => {
                            let (s2, _) = self.lower_expr_to_block(e)?;
                            stmts.extend(s2);
                        }
                        _ => {}
                    }
                }
                let ret = final_expr.as_ref().map(|e| self.lower_expr_rvalue(e)).transpose()?;
                if let Some((s2, val)) = ret {
                    stmts.extend(s2);
                    Ok((stmts, Some(val)))
                } else {
                    Ok((stmts, None))
                }
            }

            HirExprKind::If { cond, then_branch, else_branch, .. } => {
                let (cond_stmts, cond_val) = self.lower_expr_rvalue(cond)?;
                let (then_stmts, _then_val) = self.lower_expr_to_block(then_branch)?;
                let (else_stmts, _else_val) = else_branch.as_ref()
                    .map(|e| self.lower_expr_to_block(e))
                    .transpose()?
                    .unwrap_or((vec![], None));

                let then_block = self.new_block();
                let else_block = if else_branch.is_some() { Some(self.new_block()) } else { None };
                let after_block = self.new_block();

                let mut stmts = cond_stmts;

                if let Some(eb) = else_block {
                    stmts.push(MirStmt::Expr(MirRvalue::If {
                        cond: Box::new(cond_val),
                        then_block,
                        else_block: Some(eb),
                    }));
                } else {
                    stmts.push(MirStmt::Expr(MirRvalue::If {
                        cond: Box::new(cond_val),
                        then_block,
                        else_block: None,
                    }));
                }

                let mut blocks = vec![];

                blocks.push(BasicBlock {
                    id: then_block,
                    stmts: then_stmts,
                    terminator: MirTerminator::Goto(after_block),
                });

                if let Some(eb) = else_block {
                    blocks.push(BasicBlock {
                        id: eb,
                        stmts: else_stmts,
                        terminator: MirTerminator::Goto(after_block),
                    });
                }

                Ok((stmts, None))
            }

            HirExprKind::Assign { target, value } => {
                let (s, val) = self.lower_expr_rvalue(value)?;
                let mut stmts = s;
                stmts.push(MirStmt::Assign { name: target.clone(), value: val });
                Ok((stmts, None))
            }

            HirExprKind::StructLiteral { name, fields } => {
                let mut stmts = Vec::new();
                let mut mir_fields = Vec::new();
                for (fname, fexpr) in fields {
                    let (s, v) = self.lower_expr_rvalue(fexpr)?;
                    stmts.extend(s);
                    mir_fields.push((fname.clone(), v));
                }
                Ok((stmts, Some(MirRvalue::StructLiteral {
                    name: name.clone(),
                    fields: mir_fields,
                })))
            }

            _ => Ok((vec![], Some(MirRvalue::Literal(LiteralValue::Int(0))))),
        }
    }

    fn lower_expr_rvalue(&mut self, expr: &HirExpr) -> TenthResult<(Vec<MirStmt>, MirRvalue)> {
        match &expr.kind {
            HirExprKind::Block { .. } | HirExprKind::If { .. } => {
                let tmp = self.new_local("tmp", Type::Unknown);
                let (stmts, val) = self.lower_expr_to_block(expr)?;
                if let Some(v) = val {
                    let mut s = stmts;
                    s.push(MirStmt::Let { name: tmp.clone(), ty: Type::Unknown, value: v });
                    Ok((s, MirRvalue::Use(tmp)))
                } else {
                    Ok((stmts, MirRvalue::Literal(LiteralValue::Int(0))))
                }
            }
            _ => {
                let (stmts, val) = self.lower_expr_to_block(expr)?;
                Ok((stmts, val.unwrap_or(MirRvalue::Literal(LiteralValue::Int(0)))))
            }
        }
    }

    fn lower_literal(&self, lit: &Literal) -> MirRvalue {
        match lit {
            Literal::Int(n) => MirRvalue::Literal(LiteralValue::Int(*n)),
            Literal::Float(n) => MirRvalue::Literal(LiteralValue::Float(*n)),
            Literal::Bool(b) => MirRvalue::Literal(LiteralValue::Bool(*b)),
            Literal::String(s) => MirRvalue::Literal(LiteralValue::Str(s.clone())),
        }
    }

    fn new_block(&mut self) -> usize {
        let id = self.block_counter;
        self.block_counter += 1;
        id
    }

    fn new_local(&mut self, prefix: &str, ty: Type) -> String {
        let name = format!("{}_{}", prefix, self.local_counter);
        self.local_counter += 1;
        self.locals.insert(name.clone(), MirLocal { name: name.clone(), ty, mutable: false });
        name
    }
}
