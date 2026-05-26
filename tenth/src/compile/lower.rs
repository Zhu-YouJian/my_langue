use std::collections::HashMap;
use crate::hir::hir::*;
use crate::hir::types::{Type, BaseType};
use crate::error::{TenthError, TenthResult};
use super::mir::*;

pub struct MirLowerer {
    local_counter: usize,
    block_counter: usize,
    locals: HashMap<String, MirLocal>,
    enum_discr: HashMap<String, HashMap<String, i64>>,
}

impl MirLowerer {
    pub fn new() -> Self {
        MirLowerer { local_counter: 0, block_counter: 0, locals: HashMap::new(), enum_discr: HashMap::new() }
    }

    pub fn lower_program(&mut self, program: &HirProgram) -> TenthResult<MirProgram> {
        // Build enum discriminant maps
        for (ename, variants) in &program.enums {
            let mut disc_map = HashMap::new();
            for (i, (vname, _)) in variants.iter().enumerate() {
                disc_map.insert(vname.clone(), i as i64);
            }
            self.enum_discr.insert(ename.clone(), disc_map);
        }
        let mut functions = Vec::new();
        for func in &program.functions {
            functions.push(self.lower_function(func)?);
        }
        let main_expr = program.main_expr.as_ref().map(|e| self.lower_top_level_expr(e)).transpose()?;
        let mut struct_defs = Vec::new();
        for (name, fields) in &program.structs {
            struct_defs.push((name.clone(), fields.clone()));
        }
        let mut enum_defs = Vec::new();
        for (name, variants) in &program.enums {
            enum_defs.push((name.clone(), variants.iter().map(|(v, _)| v.clone()).collect()));
        }
        Ok(MirProgram { functions, main_expr, struct_defs, enum_defs })
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
                        HirStmtKind::Return(val) => {
                            if let Some(e) = val {
                                let (s2, v) = self.lower_expr_rvalue(e)?;
                                stmts.extend(s2);
                                stmts.push(MirStmt::Return(Some(v)));
                            } else {
                                stmts.push(MirStmt::Return(None));
                            }
                        }
                        HirStmtKind::While { cond, body } => {
                            let (cs, cv) = self.lower_expr_rvalue(cond)?;
                            stmts.extend(cs);
                            let (bs, _) = self.lower_stmt_to_block(body)?;
                            stmts.push(MirStmt::While { cond: cv, body: bs });
                        }
                        HirStmtKind::Loop { body } => {
                            let (bs, _) = self.lower_stmts_to_block(body)?;
                            stmts.push(MirStmt::Loop { body: bs });
                        }
                        HirStmtKind::Break => {
                            stmts.push(MirStmt::Break);
                        }
                        HirStmtKind::Continue => {
                            stmts.push(MirStmt::Continue);
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
                // If branches have side-effect statements, use IfElse with outer temp
                if !ts.is_empty() || !es.is_empty() {
                    // Only create temp if there's a non-unit value to capture
                    let is_not_unit = |v: &MirRvalue| !matches!(&v.ty, Type::Base(crate::hir::types::BaseType::Unit));
                    let result_tmp = if tv.as_ref().map_or(false, is_not_unit) ||
                                         ev.as_ref().map_or(false, is_not_unit) {
                        Some(self.new_local("ifv", ty.clone()))
                    } else {
                        None
                    };
                    // Declare outer temp before IfElse
                    if let Some(tmp_name) = &result_tmp {
                        // Use the best type among then-value, else-value, and expression type
                        let init_ty = {
                            let t_ty = tv.as_ref().map(|v| v.ty.clone());
                            let e_ty = ev.as_ref().map(|v| v.ty.clone());
                            let all: Vec<Option<Type>> = vec![t_ty, e_ty, Some(ty.clone())];
                            let best = all.into_iter()
                                .find(|t| t.as_ref().map_or(false, |t| !matches!(t, Type::Unknown) && !matches!(t, Type::Base(crate::hir::types::BaseType::Unit))));
                            best.flatten().unwrap_or(Type::Unknown)
                        };
                        if !matches!(&init_ty, Type::Base(crate::hir::types::BaseType::Unit)) {
                            let val = rv(init_ty.clone(), MirRvalueKind::Literal(LiteralValue::Int(0)));
                            stmts.push(MirStmt::Let {
                                name: tmp_name.clone(),
                                ty: init_ty,
                                value: val,
                            });
                        }
                    }
                    // Append the values to the branch bodies as assignments to temp
                    let mut then_body = ts;
                    if let (Some(tmp_name), Some(tv)) = (&result_tmp, tv) {
                        then_body.push(MirStmt::Assign { name: tmp_name.clone(), value: tv });
                    }
                    let mut else_body = es;
                    if let (Some(tmp_name), Some(ev)) = (&result_tmp, ev) {
                        else_body.push(MirStmt::Assign { name: tmp_name.clone(), value: ev });
                    }
                    stmts.push(MirStmt::IfElse {
                        cond: cv,
                        then_body,
                        else_body,
                    });
                    // Return Use of the outer temp variable
                    Ok((stmts, result_tmp.map(|n| rv(ty.clone(), MirRvalueKind::Use(n)))))
                } else {
                    // Pure expression branches — use ternary IfExpr, or IfElse for if-without-else
                    Ok(match (tv, ev) {
                        (Some(tv), Some(ev)) => {
                            (stmts, Some(rv(ty.clone(), MirRvalueKind::IfExpr { cond: Box::new(cv), then_val: Box::new(tv), else_val: Box::new(ev) })))
                        }
                        (Some(tv), None) => {
                            // If without else — must preserve condition, emit as IfElse with empty else
                            stmts.push(MirStmt::IfElse {
                                cond: cv,
                                then_body: vec![MirStmt::Expr(tv)],
                                else_body: vec![],
                            });
                            (stmts, None)
                        }
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

            HirExprKind::FieldAssign { target, field, value } => {
                let (s, val) = self.lower_expr_rvalue(value)?;
                let (ts, tv) = self.lower_expr_rvalue(target)?;
                let mut stmts = s;
                stmts.extend(ts);
                stmts.push(MirStmt::FieldAssign { target: tv, field: field.clone(), value: val });
                Ok((stmts, None))
            }

            HirExprKind::Field { target, field } => {
                let (s, t) = self.lower_expr_rvalue(target)?;
                Ok((s, Some(rv(ty, MirRvalueKind::Field { target: Box::new(t), field: field.clone() }))))
            }

            HirExprKind::StructLiteral { name, fields } => {
                let mut stmts = Vec::new(); let mut mf = Vec::new();
                for (fnm, fe) in fields { let (s, v) = self.lower_expr_rvalue(fe)?; stmts.extend(s); mf.push((fnm.clone(), v)); }
                Ok((stmts, Some(rv(ty, MirRvalueKind::StructLiteral { name: name.clone(), fields: mf }))))
            }

            HirExprKind::Ref(inner) => {
                let (mut s, v) = self.lower_expr_rvalue(inner)?;
                let name = self.ensure_var("ref_tmp", v, &mut s)?;
                Ok((s, Some(rv(ty, MirRvalueKind::Ref(name)))))
            }
            HirExprKind::MutRef(inner) => {
                let (mut s, v) = self.lower_expr_rvalue(inner)?;
                let name = self.ensure_var("mutref_tmp", v, &mut s)?;
                Ok((s, Some(rv(ty, MirRvalueKind::MutRef(name)))))
            }
            HirExprKind::Deref(inner) => {
                let (s, v) = self.lower_expr_rvalue(inner)?;
                Ok((s, Some(rv(ty, MirRvalueKind::Deref(v.extract_name()?)))))
            }
            HirExprKind::Move(inner) => {
                let (s, v) = self.lower_expr_rvalue(inner)?;
                Ok((s, Some(rv(ty, MirRvalueKind::Move(v.extract_name()?)))))
            }
            HirExprKind::MethodCall { receiver, method, args, .. } => {
                let (rs, rv_val) = self.lower_expr_rvalue(receiver)?;
                let mut stmts = rs;
                let mut arg_vals = Vec::new();
                for a in args { let (s, v) = self.lower_expr_rvalue(a)?; stmts.extend(s); arg_vals.push(v); }
                Ok((stmts, Some(rv(ty, MirRvalueKind::MethodCall { receiver: Box::new(rv_val), method: method.clone(), args: arg_vals }))))
            }

            HirExprKind::Index { target, indices } => {
                let (ts, tv) = self.lower_expr_rvalue(target)?;
                let mut stmts = ts;
                if indices.len() == 1 {
                    if let Index::Single(idx_expr) = &indices[0] {
                        let (is, iv) = self.lower_expr_rvalue(idx_expr)?;
                        stmts.extend(is);
                        // For string indexing, use str_at; for Vec, use Vec_get
                        let is_str_target = matches!(&tv.ty, Type::Base(BaseType::Str));
                        let func_name = if is_str_target { "str_at" } else { "Vec_get" };
                        return Ok((stmts, Some(rv(ty, MirRvalueKind::Call {
                            func: func_name.to_string(),
                            args: vec![tv, iv],
                        }))));
                    }
                }
                // Multi-index or range — not yet supported
                Ok((stmts, Some(rv(ty, MirRvalueKind::Literal(LiteralValue::Int(0))))))
            }

            HirExprKind::EnumLiteral { enum_name, variant, .. } => {
                let disc = self.enum_discr.get(enum_name)
                    .and_then(|m| m.get(variant))
                    .copied()
                    .unwrap_or(0);
                Ok((vec![], Some(rv(Type::Base(BaseType::I64), MirRvalueKind::Literal(LiteralValue::Int(disc))))))
            }

            HirExprKind::Match { scrutinee, arms } => {
                let (mut stmts, sv) = self.lower_expr_rvalue(scrutinee)?;
                let disc_name = self.new_local("match_disc", Type::Base(BaseType::I64));
                stmts.push(MirStmt::Let { name: disc_name.clone(), ty: Type::Base(BaseType::I64), value: sv });

                // Infer result type from first arm's body type (or default to I64)
                let result_ty = arms.first()
                    .map(|arm| arm.body.ty.clone())
                    .unwrap_or(Type::Base(BaseType::I64));
                let result_ty = if matches!(result_ty, Type::Unknown | Type::Base(BaseType::Unit)) { Type::Base(BaseType::I64) } else { result_ty };

                // Build if-else chain for the arms
                let result_tmp = self.new_local("match_res", result_ty.clone());
                // Initialize with 0
                stmts.push(MirStmt::Let { name: result_tmp.clone(), ty: result_ty.clone(), value: rv(result_ty.clone(), MirRvalueKind::Literal(LiteralValue::Int(0))) });

                // Process arms in reverse to build nested if-else
                let mut current_else: Vec<MirStmt> = vec![]; // final else (wildcard or empty)
                let mut have_wildcard = false;

                for arm in arms.iter().rev() {
                    match &arm.pattern {
                        HirPattern::EnumVariant { enum_name: _, variant, .. } => {
                            let disc_val = self.enum_discr.values()
                                .find_map(|m| m.get(variant))
                                .copied()
                                .unwrap_or(-1);
                            let (body_stmts, body_val) = self.lower_expr_to_block(&arm.body)?;
                            let mut then_body = body_stmts;
                            if let Some(v) = body_val {
                                // Only assign if value is a scalar type compatible with int64_t
                                let v_is_scalar = matches!(&v.ty, Type::Base(BaseType::I64 | BaseType::I32 | BaseType::I8 | BaseType::I16 | BaseType::Bool));
                                if v_is_scalar || matches!(&v.ty, Type::Unknown) {
                                    then_body.push(MirStmt::Assign { name: result_tmp.clone(), value: v });
                                } else {
                                    // Non-scalar value (struct/call) — emit as expression to preserve side effects
                                    then_body.push(MirStmt::Expr(v));
                                }
                            }
                            let cond = rv(Type::bool_(), MirRvalueKind::BinaryOp(
                                BinOp::Eq,
                                Box::new(rv(Type::Base(BaseType::I64), MirRvalueKind::Use(disc_name.clone()))),
                                Box::new(rv(Type::Base(BaseType::I64), MirRvalueKind::Literal(LiteralValue::Int(disc_val)))),
                            ));
                            current_else = vec![MirStmt::IfElse {
                                cond,
                                then_body,
                                else_body: std::mem::take(&mut current_else),
                            }];
                        }
                        HirPattern::Wildcard => {
                            have_wildcard = true;
                            let (body_stmts, body_val) = self.lower_expr_to_block(&arm.body)?;
                            let mut else_body = body_stmts;
                            if let Some(v) = body_val {
                                let v_is_scalar = matches!(&v.ty, Type::Base(BaseType::I64 | BaseType::I32 | BaseType::I8 | BaseType::I16 | BaseType::Bool));
                                if v_is_scalar || matches!(&v.ty, Type::Unknown) {
                                    else_body.push(MirStmt::Assign { name: result_tmp.clone(), value: v });
                                } else {
                                    else_body.push(MirStmt::Expr(v));
                                }
                            }
                            current_else = else_body;
                        }
                        _ => {} // literal patterns ignored for now
                    }
                }

                if !have_wildcard {
                    // No wildcard → add empty else
                    current_else = vec![MirStmt::Expr(rv(Type::unit(), MirRvalueKind::Literal(LiteralValue::Int(0))))];
                }

                stmts.extend(current_else);
                Ok((stmts, Some(rv(result_ty, MirRvalueKind::Use(result_tmp)))))
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
                    let mut s = stmts; s.push(MirStmt::Let { name: tmp.clone(), ty: ty.clone(), value: v });
                    Ok((s, rv(ty.clone(), MirRvalueKind::Use(tmp))))
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

    fn lower_stmt_to_block(&mut self, stmt: &HirStmt) -> TenthResult<(Vec<MirStmt>, Option<MirRvalue>)> {
        // Wrap a single HirStmt in a pseudo-block and lower it
        let pseudo_block = HirExpr {
            kind: HirExprKind::Block { stmts: vec![stmt.clone()], final_expr: None },
            ty: Type::unit(),
            span: stmt.span.clone(),
        };
        self.lower_expr_to_block(&pseudo_block)
    }

    fn lower_stmts_to_block(&mut self, stmts: &[HirStmt]) -> TenthResult<(Vec<MirStmt>, Option<MirRvalue>)> {
        let pseudo_block = HirExpr {
            kind: HirExprKind::Block { stmts: stmts.to_vec(), final_expr: None },
            ty: Type::unit(),
            span: crate::lexer::token::Span { line: 0, col: 0 },
        };
        self.lower_expr_to_block(&pseudo_block)
    }

    fn new_block(&mut self) -> usize { let id = self.block_counter; self.block_counter += 1; id }
    fn new_local(&mut self, prefix: &str, ty: Type) -> String {
        let name = format!("{}_{}", prefix, self.local_counter); self.local_counter += 1;
        self.locals.insert(name.clone(), MirLocal { name: name.clone(), ty, mutable: false }); name
    }
}

fn rv(ty: Type, kind: MirRvalueKind) -> MirRvalue { MirRvalue { kind, ty } }

impl MirLowerer {
    /// If value is a simple variable, return its name. Otherwise, create a temp and push a Let.
    fn ensure_var(&mut self, prefix: &str, value: MirRvalue, extra_stmts: &mut Vec<MirStmt>) -> TenthResult<String> {
        match &value.kind {
            MirRvalueKind::Use(name) | MirRvalueKind::Deref(name) | MirRvalueKind::MutRef(name) | MirRvalueKind::Ref(name) | MirRvalueKind::Move(name) => Ok(name.clone()),
            _ => {
                let name = self.new_local(prefix, value.ty.clone());
                extra_stmts.push(MirStmt::Let { name: name.clone(), ty: value.ty.clone(), value });
                Ok(name)
            }
        }
    }
}

impl MirRvalue {
    fn extract_name(&self) -> TenthResult<String> {
        match &self.kind {
            MirRvalueKind::Use(name) | MirRvalueKind::Deref(name) | MirRvalueKind::MutRef(name) | MirRvalueKind::Ref(name) | MirRvalueKind::Move(name) => Ok(name.clone()),
            _ => Err(TenthError::RuntimeError { message: format!("expected variable name, got {:?}", self.kind) }),
        }
    }
}

fn lower_literal_val(lit: &Literal) -> LiteralValue {
    match lit {
        Literal::Int(n) => LiteralValue::Int(*n),
        Literal::Float(n) => LiteralValue::Float(*n),
        Literal::Bool(b) => LiteralValue::Bool(*b),
        Literal::String(s) => LiteralValue::Str(s.clone()),
    }
}
