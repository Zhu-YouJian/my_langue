//! Closure body compilation, string collection, and closure collection.

use wasm_encoder::{CodeSection, Function, Instruction, ValType};
use crate::error::TenthResult;
use crate::hir::hir::*;
use crate::hir::types::{BaseType, Type};
use super::{IMPORT_COUNT, WasmCompiler};

impl WasmCompiler {
    // ── Closure body compilation (D5) ───────────────────────────────────

    /// Traverse HIR and compile each closure body in order.
    /// Order must match collect_closures and emit_function_section.
    pub(super) fn compile_closure_bodies(&mut self, codes: &mut CodeSection, program: &HirProgram) -> TenthResult<()> {
        for func in &program.functions {
            self.ccb_expr(codes, &func.body)?;
        }
        if let Some(ref e) = program.main_expr {
            self.ccb_expr(codes, e)?;
        }
        Ok(())
    }

    pub(super) fn ccb_expr(&mut self, codes: &mut CodeSection, e: &HirExpr) -> TenthResult<()> {
        match &e.kind {
            HirExprKind::Closure { params, body, captures } => {
                let func = self.compile_closure_body(params, body, captures)?;
                codes.function(&func);
                // Recurse for nested closures
                self.ccb_expr(codes, body)?;
            }
            HirExprKind::Binary { left, right, .. } => {
                self.ccb_expr(codes, left)?;
                self.ccb_expr(codes, right)?;
            }
            HirExprKind::Unary { expr: inner, .. } => { self.ccb_expr(codes, inner)?; }
            HirExprKind::Call { func, args, .. } => {
                self.ccb_expr(codes, func)?;
                for a in args { self.ccb_expr(codes, a)?; }
            }
            HirExprKind::GenericCall { func, args, .. } => {
                self.ccb_expr(codes, func)?;
                for a in args { self.ccb_expr(codes, a)?; }
            }
            HirExprKind::MethodCall { receiver, args, .. } => {
                self.ccb_expr(codes, receiver)?;
                for a in args { self.ccb_expr(codes, a)?; }
            }
            HirExprKind::Block { stmts, final_expr } => {
                for s in stmts { self.ccb_stmt(codes, s)?; }
                if let Some(e) = final_expr { self.ccb_expr(codes, e)?; }
            }
            HirExprKind::If { cond, then_branch, else_branch, .. } => {
                self.ccb_expr(codes, cond)?;
                self.ccb_expr(codes, then_branch)?;
                if let Some(e) = else_branch { self.ccb_expr(codes, e)?; }
            }
            HirExprKind::Assign { value, .. } => { self.ccb_expr(codes, value)?; }
            HirExprKind::AssignOp { value, .. } => { self.ccb_expr(codes, value)?; }
            HirExprKind::StructLiteral { fields, .. } => {
                for (_, e) in fields { self.ccb_expr(codes, e)?; }
            }
            HirExprKind::EnumLiteral { fields, .. } => {
                for (_, e) in fields { self.ccb_expr(codes, e)?; }
            }
            HirExprKind::Field { target, .. } => { self.ccb_expr(codes, target)?; }
            HirExprKind::FieldAssign { target, value, .. } => {
                self.ccb_expr(codes, target)?;
                self.ccb_expr(codes, value)?;
            }
            HirExprKind::Index { target, indices } => {
                self.ccb_expr(codes, target)?;
                for idx in indices {
                    match idx {
                        Index::Single(e) => { self.ccb_expr(codes, e)?; }
                        Index::Range { start, end } => {
                            if let Some(s) = start { self.ccb_expr(codes, s)?; }
                            if let Some(e) = end { self.ccb_expr(codes, e)?; }
                        }
                        _ => {}
                    }
                }
            }
            HirExprKind::Ref(inner) | HirExprKind::MutRef(inner)
            | HirExprKind::Deref(inner) | HirExprKind::TryBlock(inner)
            | HirExprKind::Lossy(inner) => {
                self.ccb_expr(codes, inner)?;
            }
            HirExprKind::TensorLiteral { data, .. } => {
                for row in data { for e in row { self.ccb_expr(codes, e)?; } }
            }
            HirExprKind::ArrayLiteral { elements, .. } => {
                for e in elements { self.ccb_expr(codes, e)?; }
            }
            HirExprKind::Range { start, end, .. } => {
                if let Some(s) = start { self.ccb_expr(codes, s)?; }
                if let Some(e) = end { self.ccb_expr(codes, e)?; }
            }
            HirExprKind::Match { scrutinee, arms, .. } => {
                self.ccb_expr(codes, scrutinee)?;
                for arm in arms { self.ccb_expr(codes, &arm.body)?; }
            }
            _ => {}
        }
        Ok(())
    }

    pub(super) fn ccb_stmt(&mut self, codes: &mut CodeSection, s: &HirStmt) -> TenthResult<()> {
        use crate::hir::hir::HirStmtKind;
        match &s.kind {
            HirStmtKind::Expr(e) => { self.ccb_expr(codes, e)?; }
            HirStmtKind::Let { init, .. } => { if let Some(e) = init { self.ccb_expr(codes, e)?; } }
            HirStmtKind::While { cond, body } => {
                self.ccb_expr(codes, cond)?;
                self.ccb_stmt(codes, body)?;
            }
            HirStmtKind::Loop { body } => { for s in body { self.ccb_stmt(codes, s)?; } }
            HirStmtKind::For { body, .. } => { self.ccb_stmt(codes, body)?; }
            HirStmtKind::Return(expr) => { if let Some(e) = expr { self.ccb_expr(codes, e)?; } }
            _ => {}
        }
        Ok(())
    }

    /// Compile a closure body into a WASM function.
    /// Param 0 = env_ptr (i64), params 1..N = closure params (i64).
    pub(super) fn compile_closure_body(
        &mut self,
        params: &[(String, Type)],
        body: &HirExpr,
        captures: &[String],
    ) -> TenthResult<Function> {
        // Reset local state for closure body
        self.local_map.clear();
        self.local_count = 0;
        self.param_count = 0;
        self.if_depths.clear();
        // Param 0 = env_ptr (unnamed)
        self.param_count = 1;
        self.local_count = 1;
        // Register closure params (param 1..N)
        for (name, _) in params {
            self.local_map.insert(name.clone(), self.local_count);
            self.local_count += 1;
            self.param_count += 1;
        }
        // Set closure compilation state
        self.compiling_closure = true;
        self.current_captures = captures.to_vec();
        // All extra locals are i64
        let locals: Vec<ValType> = (0..512).map(|_| ValType::I64).collect();
        let mut func = Function::new_with_locals_types(locals);
        self.compile_expr(&mut func, body)?;
        if matches!(&body.ty, Type::Base(BaseType::Unit)) {
            func.instruction(&Instruction::Return);
        }
        func.instruction(&Instruction::End);
        // Reset closure compilation state
        self.compiling_closure = false;
        self.current_captures.clear();
        Ok(func)
    }

    // ── String collection ───────────────────────────────────────────────

    pub(super) fn collect_strings(&mut self, p: &HirProgram) {
        // Pre-intern all single ASCII characters so str_at can return
        // pointers to pre-allocated strings without heap allocation.
        for byte in 1u8..128u8 {
            if let Some(c) = char::from_u32(byte as u32) {
                let s = c.to_string();
                self.intern_string(&s);
            }
        }
        for f in &p.functions { self.cs_expr(&f.body); }
        if let Some(ref e) = p.main_expr { self.cs_expr(e); }
    }

    pub(super) fn cs_expr(&mut self, e: &HirExpr) {
        use HirExprKind;
        match &e.kind {
            HirExprKind::Literal(Literal::String(s)) => { self.intern_string(s); }
            HirExprKind::Binary { left, right, .. } => { self.cs_expr(left); self.cs_expr(right); }
            HirExprKind::Unary { expr: inner, .. } => self.cs_expr(inner),
            HirExprKind::Call { args, .. } => { for a in args { self.cs_expr(a); } }
            HirExprKind::Block { stmts, final_expr } => {
                for s in stmts { self.cs_stmt(s); }
                if let Some(e) = final_expr { self.cs_expr(e); }
            }
            HirExprKind::If { cond, then_branch, else_branch, .. } => {
                self.cs_expr(cond); self.cs_expr(then_branch);
                if let Some(e) = else_branch { self.cs_expr(e); }
            }
            HirExprKind::Assign { value, .. } => self.cs_expr(value),
            HirExprKind::AssignOp { value, .. } => self.cs_expr(value),
            HirExprKind::StructLiteral { fields, .. } => {
                for (_, e) in fields { self.cs_expr(e); }
            }
            HirExprKind::EnumLiteral { fields, .. } => {
                for (_, e) in fields { self.cs_expr(e); }
            }
            HirExprKind::MethodCall { receiver, args, .. } => {
                self.cs_expr(receiver);
                for a in args { self.cs_expr(a); }
            }
            HirExprKind::Index { target, indices } => {
                self.cs_expr(target);
                for idx in indices {
                    match idx {
                        Index::Single(e) => self.cs_expr(e),
                        Index::Range { start, end } => {
                            if let Some(s) = start { self.cs_expr(s); }
                            if let Some(e) = end { self.cs_expr(e); }
                        }
                        _ => {}
                    }
                }
            }
            _ => {}
        }
    }

    pub(super) fn cs_stmt(&mut self, s: &HirStmt) {
        use crate::hir::hir::HirStmtKind;
        match &s.kind {
            HirStmtKind::Expr(e) => self.cs_expr(e),
            HirStmtKind::Let { init, .. } => { if let Some(e) = init { self.cs_expr(e); } }
            HirStmtKind::While { cond, body } => { self.cs_expr(cond); self.cs_stmt(body); }
            HirStmtKind::Loop { body } => { for s in body { self.cs_stmt(s); } }
            HirStmtKind::For { body, .. } => self.cs_stmt(body),
            HirStmtKind::Return(expr) => { if let Some(e) = expr { self.cs_expr(e); } }
            _ => {}
        }
    }

    // ── Closure collection (D5) ─────────────────────────────────────────

    /// Traverse HIR and register all Closure nodes. Assigns func_idx and
    /// stores captures. type_idx is filled later during emit_type_section.
    pub(super) fn collect_closures(&mut self, program: &HirProgram) {
        let num_user_funcs = program.functions.len() as u32;
        for func in &program.functions {
            self.cc_expr(&func.body, num_user_funcs);
        }
        if let Some(ref e) = program.main_expr {
            self.cc_expr(e, num_user_funcs);
        }
    }

    pub(super) fn cc_expr(&mut self, e: &HirExpr, num_user_funcs: u32) {
        match &e.kind {
            HirExprKind::Closure { params, body, captures } => {
                let cidx = self.closure_info.len() as u32;
                // func_idx = IMPORT_COUNT + num_user_funcs + 1 (main) + cidx
                let func_idx = IMPORT_COUNT + num_user_funcs + 1 + cidx;
                let param_count = params.len() as u32;
                self.closure_info.push((func_idx, 0, param_count));
                self.closure_captures.push(captures.clone());
                let ptr = e as *const HirExpr as usize;
                self.closure_expr_map.insert(ptr, cidx as usize);
                // Recurse for nested closures
                self.cc_expr(body, num_user_funcs);
            }
            HirExprKind::Binary { left, right, .. } => {
                self.cc_expr(left, num_user_funcs);
                self.cc_expr(right, num_user_funcs);
            }
            HirExprKind::Unary { expr: inner, .. } => {
                self.cc_expr(inner, num_user_funcs);
            }
            HirExprKind::Call { func, args, .. } => {
                self.cc_expr(func, num_user_funcs);
                for a in args { self.cc_expr(a, num_user_funcs); }
            }
            HirExprKind::GenericCall { func, args, .. } => {
                self.cc_expr(func, num_user_funcs);
                for a in args { self.cc_expr(a, num_user_funcs); }
            }
            HirExprKind::MethodCall { receiver, args, .. } => {
                self.cc_expr(receiver, num_user_funcs);
                for a in args { self.cc_expr(a, num_user_funcs); }
            }
            HirExprKind::Block { stmts, final_expr } => {
                for s in stmts { self.cc_stmt(s, num_user_funcs); }
                if let Some(e) = final_expr { self.cc_expr(e, num_user_funcs); }
            }
            HirExprKind::If { cond, then_branch, else_branch, .. } => {
                self.cc_expr(cond, num_user_funcs);
                self.cc_expr(then_branch, num_user_funcs);
                if let Some(e) = else_branch { self.cc_expr(e, num_user_funcs); }
            }
            HirExprKind::Assign { value, .. } => { self.cc_expr(value, num_user_funcs); }
            HirExprKind::AssignOp { value, .. } => { self.cc_expr(value, num_user_funcs); }
            HirExprKind::StructLiteral { fields, .. } => {
                for (_, e) in fields { self.cc_expr(e, num_user_funcs); }
            }
            HirExprKind::EnumLiteral { fields, .. } => {
                for (_, e) in fields { self.cc_expr(e, num_user_funcs); }
            }
            HirExprKind::Field { target, .. } => { self.cc_expr(target, num_user_funcs); }
            HirExprKind::FieldAssign { target, value, .. } => {
                self.cc_expr(target, num_user_funcs);
                self.cc_expr(value, num_user_funcs);
            }
            HirExprKind::Index { target, indices } => {
                self.cc_expr(target, num_user_funcs);
                for idx in indices {
                    match idx {
                        Index::Single(e) => self.cc_expr(e, num_user_funcs),
                        Index::Range { start, end } => {
                            if let Some(s) = start { self.cc_expr(s, num_user_funcs); }
                            if let Some(e) = end { self.cc_expr(e, num_user_funcs); }
                        }
                        _ => {}
                    }
                }
            }
            HirExprKind::Ref(inner) | HirExprKind::MutRef(inner)
            | HirExprKind::Deref(inner) | HirExprKind::TryBlock(inner)
            | HirExprKind::Lossy(inner) => {
                self.cc_expr(inner, num_user_funcs);
            }
            HirExprKind::TensorLiteral { data, .. } => {
                for row in data { for e in row { self.cc_expr(e, num_user_funcs); } }
            }
            HirExprKind::ArrayLiteral { elements, .. } => {
                for e in elements { self.cc_expr(e, num_user_funcs); }
            }
            HirExprKind::Range { start, end, .. } => {
                if let Some(s) = start { self.cc_expr(s, num_user_funcs); }
                if let Some(e) = end { self.cc_expr(e, num_user_funcs); }
            }
            HirExprKind::Match { scrutinee, arms, .. } => {
                self.cc_expr(scrutinee, num_user_funcs);
                for arm in arms { self.cc_expr(&arm.body, num_user_funcs); }
            }
            _ => {}
        }
    }

    pub(super) fn cc_stmt(&mut self, s: &HirStmt, num_user_funcs: u32) {
        use crate::hir::hir::HirStmtKind;
        match &s.kind {
            HirStmtKind::Expr(e) => self.cc_expr(e, num_user_funcs),
            HirStmtKind::Let { init, .. } => { if let Some(e) = init { self.cc_expr(e, num_user_funcs); } }
            HirStmtKind::While { cond, body } => {
                self.cc_expr(cond, num_user_funcs);
                self.cc_stmt(body, num_user_funcs);
            }
            HirStmtKind::Loop { body } => { for s in body { self.cc_stmt(s, num_user_funcs); } }
            HirStmtKind::For { body, .. } => self.cc_stmt(body, num_user_funcs),
            HirStmtKind::Return(expr) => { if let Some(e) = expr { self.cc_expr(e, num_user_funcs); } }
            _ => {}
        }
    }
}
