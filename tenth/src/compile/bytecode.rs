//! HIR → Bytecode compiler.
//!
//! Walks HIR and emits bytecode for the stack VM.

use std::collections::HashMap;
use crate::error::TenthResult;
use crate::hir::hir::*;
use super::super::runtime::vm::{Chunk, Op};

pub struct BytecodeCompiler {
    chunk: Chunk,
    /// Local variable name → slot index
    locals: Vec<String>,
    /// Pending backpatches: (patch_offset, label_id)
    patches: Vec<(usize, usize)>,
    /// Labels: (label_id, code_offset)
    labels: HashMap<usize, usize>,
    next_label: usize,
    /// Loop context stack: (loop_start_code_offset, break_label, continue_label)
    loop_stack: Vec<(usize, usize, usize)>,
}

impl BytecodeCompiler {
    pub fn new() -> Self {
        BytecodeCompiler {
            chunk: Chunk::new(),
            locals: Vec::new(),
            patches: Vec::new(),
            labels: HashMap::new(),
            next_label: 0,
            loop_stack: Vec::new(),
        }
    }

    pub fn compile(mut self, func: &HirFnDef) -> TenthResult<Chunk> {
        self.chunk.num_args = func.params.len();
        for (name, _) in &func.params {
            self.locals.push(name.clone());
        }
        self.compile_expr(&func.body)?;
        // Only push Unit for void functions
        if matches!(func.return_type, crate::hir::types::Type::Base(crate::hir::types::BaseType::Unit)) {
            self.chunk.emit(Op::PushUnit);
        }
        self.chunk.emit(Op::Ret);
        self.resolve_patches();
        Ok(self.chunk)
    }

    pub fn compile_main(mut self, expr: &HirExpr) -> TenthResult<Chunk> {
        self.compile_expr(expr)?;
        self.chunk.emit(Op::Ret);
        self.resolve_patches();
        Ok(self.chunk)
    }

    fn compile_expr(&mut self, expr: &HirExpr) -> TenthResult<()> {
        use HirExprKind::*;
        match &expr.kind {
            Literal(lit) => match lit {
                crate::hir::hir::Literal::Int(n) => self.chunk.emit(Op::PushInt(*n)),
                crate::hir::hir::Literal::Float(f) => self.chunk.emit(Op::PushFloat(*f)),
                crate::hir::hir::Literal::Bool(b) => self.chunk.emit(Op::PushBool(*b)),
                crate::hir::hir::Literal::String(s) => {
                    let i = self.chunk.add_string(s);
                    self.chunk.emit(Op::PushStr(i));
                }
            },

            Var(name) => {
                // Check locals first
                if let Some(pos) = self.locals.iter().position(|n| n == name) {
                    self.chunk.emit(Op::Load(pos));
                } else {
                    let i = self.chunk.add_string(name);
                    self.chunk.emit(Op::LoadGlobal(i));
                }
            }

            Binary { op, left, right, .. } => {
                self.compile_expr(left)?;
                self.compile_expr(right)?;
                use crate::hir::hir::BinOp::*;
                self.chunk.emit(match op {
                    Add => Op::Add, Sub => Op::Sub, Mul => Op::Mul, Div => Op::Div,
                    Mod => Op::Mod,
                    Eq => Op::Eq, NotEq => Op::Neq,
                    Lt => Op::Lt, Gt => Op::Gt, LtEq => Op::Lte, GtEq => Op::Gte,
                    And => {
                        // Short-circuit: left && right
                        // If left is false, jump to end with false; else eval right
                        let end_label = self.new_label();
                        self.chunk.emit(Op::Dup);
                        self.chunk.emit(Op::JmpFalse(0)); // patch later
                        self.patch_jump(end_label);
                        self.chunk.emit(Op::Pop);
                        self.compile_expr(right)?;
                        self.label(end_label);
                        return Ok(());
                    }
                    Or => {
                        let end_label = self.new_label();
                        self.chunk.emit(Op::Dup);
                        self.chunk.emit(Op::JmpTrue(0));
                        self.patch_jump(end_label);
                        self.chunk.emit(Op::Pop);
                        self.compile_expr(right)?;
                        self.label(end_label);
                        return Ok(());
                    }
                });
            }

            Unary { op, expr: inner, .. } => {
                self.compile_expr(inner)?;
                use crate::hir::hir::UnaryOp::*;
                self.chunk.emit(match op { Neg => Op::Neg, Not => Op::Not });
            }

            Call { func, args, .. } => {
                // Push args left-to-right; VM pops in reverse into locals
                for a in args.iter() {
                    self.compile_expr(a)?;
                }
                match &func.kind {
                    Var(name) => {
                        let i = self.chunk.add_string(name);
                        self.chunk.emit(Op::CallN(i, args.len()));
                    }
                    _ => {
                        // Indirect call — compile func as expression, then we'd need CallIndirect
                        // For now, fallback: evaluate func and push it as a value
                        self.compile_expr(func)?;
                        for _ in args { let _ = self.chunk; } // dummy
                    }
                }
            }

            If { cond, then_branch, else_branch, .. } => {
                self.compile_expr(cond)?;
                let else_label = self.new_label();
                let end_label = self.new_label();
                self.chunk.emit(Op::JmpFalse(0));
                self.patch_jump(else_label);
                self.compile_expr(then_branch)?;
                self.chunk.emit(Op::Jump(0));
                self.patch_jump(end_label);
                self.label(else_label);
                if let Some(eb) = else_branch {
                    self.compile_expr(eb)?;
                } else {
                    self.chunk.emit(Op::PushUnit);
                }
                self.label(end_label);
            }

            Block { stmts, final_expr } => {
                for s in stmts {
                    self.compile_stmt(s)?;
                }
                if let Some(e) = final_expr {
                    self.compile_expr(e)?;
                } else {
                    self.chunk.emit(Op::PushUnit);
                }
            }

            Assign { target, value } => {
                self.compile_expr(value)?;
                self.chunk.emit(Op::Dup);
                if let Some(pos) = self.locals.iter().position(|n| n == target) {
                    self.chunk.emit(Op::Store(pos));
                } else {
                    // New local
                    let pos = self.locals.len();
                    self.locals.push(target.clone());
                    self.chunk.emit(Op::Store(pos));
                }
            }

            AssignOp { target, op, value } => {
                // target = target op value
                if let Some(pos) = self.locals.iter().position(|n| n == target) {
                    self.chunk.emit(Op::Load(pos));
                } else {
                    let i = self.chunk.add_string(target);
                    self.chunk.emit(Op::LoadGlobal(i));
                }
                self.compile_expr(value)?;
                use crate::hir::hir::BinOp::*;
                self.chunk.emit(match op { Add=>Op::Add, Sub=>Op::Sub, Mul=>Op::Mul, Div=>Op::Div, _=>Op::Add });
                self.chunk.emit(Op::Dup);
                if let Some(pos) = self.locals.iter().position(|n| n == target) {
                    self.chunk.emit(Op::Store(pos));
                } else {
                    let i = self.chunk.add_string(target);
                    self.chunk.emit(Op::StoreGlobal(i));
                }
            }

            StructLiteral { name, fields } => {
                let ni = self.chunk.add_string(name);
                for (fname, fexpr) in fields.iter().rev() {
                    self.compile_expr(fexpr)?;
                    let fi = self.chunk.add_string(fname);
                    self.chunk.emit(Op::PushStr(fi));
                }
                self.chunk.emit(Op::NewStruct(ni, fields.len()));
            }

            Field { target, field } => {
                self.compile_expr(target)?;
                let fi = self.chunk.add_string(field);
                self.chunk.emit(Op::LoadField(fi));
            }

            FieldAssign { target, field, value } => {
                self.compile_expr(target)?;
                self.compile_expr(value)?;
                let fi = self.chunk.add_string(field);
                self.chunk.emit(Op::StoreField(fi));
            }

            DerefAssign { target, value } => {
                self.compile_expr(target)?;
                self.compile_expr(value)?;
                self.chunk.emit(Op::Store(0));
            }
            DerefAssignOp { target, value, .. } => {
                self.compile_expr(target)?;
                self.compile_expr(value)?;
                self.chunk.emit(Op::Store(0));
            }

            Index { target, indices } => {
                self.compile_expr(target)?;
                for idx in indices {
                    match idx {
                        crate::hir::hir::Index::Single(e) => {
                            self.compile_expr(e)?;
                            self.chunk.emit(Op::IndexGet);
                        }
                        crate::hir::hir::Index::Range { start, end } => {
                            if let Some(s) = start {
                                self.compile_expr(s)?;
                            } else {
                                self.chunk.emit(Op::PushInt(0));
                            }
                            if let Some(e) = end {
                                self.compile_expr(e)?;
                            } else {
                                self.chunk.emit(Op::PushInt(i64::MAX));
                            }
                            self.chunk.emit(Op::SliceStr);
                        }
                        _ => {}
                    }
                }
            }

            MethodCall { receiver, method, args, .. } => {
                self.compile_expr(receiver)?;
                for a in args.iter() {
                    self.compile_expr(a)?;
                }
                let mi = self.chunk.add_string(method);
                self.chunk.emit(Op::MethodCall(mi, args.len()));
            }

            ArrayLiteral { elements, .. } => {
                for e in elements.iter().rev() {
                    self.compile_expr(e)?;
                }
                self.chunk.emit(Op::MakeVec(elements.len()));
            }

            Ref(inner) | MutRef(inner) | Deref(inner) => {
                // VM doesn't track ownership; treat as pass-through
                self.compile_expr(inner)?;
            }

            EnumLiteral { enum_name, variant, fields } => {
                let name_i = self.chunk.add_string(enum_name);
                let variant_i = self.chunk.add_string(variant);
                for (fname, fexpr) in fields.iter().rev() {
                    self.compile_expr(fexpr)?;
                    let fi = self.chunk.add_string(fname);
                    self.chunk.emit(Op::PushStr(fi));
                }
                self.chunk.emit(Op::MakeEnum(name_i, variant_i, fields.len()));
            }

            Match { scrutinee, arms } => {
                self.compile_expr(scrutinee)?;
                let end_label = self.new_label();
                for arm in arms {
                    match &arm.pattern {
                        HirPattern::EnumVariant { enum_name: _, variant, field_bind } => {
                            self.chunk.emit(Op::Dup); // dup for IsEnumVariant check
                            let variant_i = self.chunk.add_string(variant);
                            self.chunk.emit(Op::IsEnumVariant(variant_i));
                            let next_label = self.new_label();
                            self.chunk.emit(Op::JmpFalse(0));
                            self.patch_jump(next_label);
                            // IsEnumVariant consumed the dup; scrutinee remains
                            if let Some((bind_name, field_name)) = field_bind {
                                let fi = self.chunk.add_string(field_name);
                                self.chunk.emit(Op::EnumGetField(fi)); // pops scrutinee, pushes field
                                let pos = self.locals.len();
                                self.locals.push(bind_name.clone());
                                self.chunk.emit(Op::Store(pos)); // pops field
                            } else {
                                self.chunk.emit(Op::Pop); // drop scrutinee (no field needed)
                            }
                            self.compile_expr(&arm.body)?;
                            self.chunk.emit(Op::Jump(0));
                            self.patch_jump(end_label);
                            self.label(next_label);
                        }
                        HirPattern::Wildcard => {
                            self.chunk.emit(Op::Pop); // drop scrutinee (wildcard ignores value)
                            self.compile_expr(&arm.body)?;
                            self.chunk.emit(Op::Jump(0));
                            self.patch_jump(end_label);
                        }
                        _ => {
                            // Other patterns: ignore for now
                        }
                    }
                }
                self.chunk.emit(Op::Pop); // drop scrutinee if no arm matched
                self.chunk.emit(Op::PushUnit); // fallback result
                self.label(end_label);
            }

            // Fallback to tree-walk for complex constructs
            _ => {
                self.chunk.emit(Op::PushUnit);
            }
        }
        Ok(())
    }

    fn compile_stmt(&mut self, stmt: &HirStmt) -> TenthResult<()> {
        match &stmt.kind {
            HirStmtKind::Let { name, init, .. } => {
                let pos = self.locals.len();
                self.locals.push(name.clone());
                if let Some(e) = init {
                    self.compile_expr(e)?;
                } else {
                    self.chunk.emit(Op::PushUnit);
                }
                self.chunk.emit(Op::Store(pos));
            }
            HirStmtKind::Expr(e) => {
                self.compile_expr(e)?;
                self.chunk.emit(Op::Pop);
            }
            HirStmtKind::Return(expr) => {
                if let Some(e) = expr {
                    self.compile_expr(e)?;
                } else {
                    self.chunk.emit(Op::PushUnit);
                }
                self.chunk.emit(Op::Ret);
            }
            HirStmtKind::While { cond, body } => {
                let loop_start = self.chunk.code.len();
                let break_label = self.new_label();
                let continue_label = self.new_label();
                self.label(continue_label); // continue: re-evaluate cond
                self.loop_stack.push((loop_start, break_label, continue_label));
                self.compile_expr(cond)?;
                self.chunk.emit(Op::JmpFalse(0));
                self.patch_jump(break_label);
                self.compile_stmt(body)?;
                // Jump back to condition (continue_label)
                let offset = loop_start as i32 - self.chunk.code.len() as i32 - 5;
                self.chunk.emit(Op::Jump(offset));
                self.label(break_label);
                self.loop_stack.pop();
            }
            HirStmtKind::Loop { body } => {
                let loop_start = self.chunk.code.len();
                let break_label = self.new_label();
                let continue_label = self.new_label();
                self.label(continue_label); // continue jumps here
                self.loop_stack.push((loop_start, break_label, continue_label));
                for s in body {
                    self.compile_stmt(s)?;
                }
                // Unconditional jump back to loop start
                let offset = loop_start as i32 - self.chunk.code.len() as i32 - 5;
                self.chunk.emit(Op::Jump(offset));
                self.label(break_label); // break jumps here
                self.loop_stack.pop();
            }
            HirStmtKind::Break => {
                if let Some(&(_, break_label, _)) = self.loop_stack.last() {
                    self.chunk.emit(Op::Jump(0));
                    self.patches.push((self.chunk.code.len() - 4, break_label));
                }
            }
            HirStmtKind::Continue => {
                if let Some(&(_, _, continue_label)) = self.loop_stack.last() {
                    self.chunk.emit(Op::Jump(0));
                    self.patches.push((self.chunk.code.len() - 4, continue_label));
                }
            }
            _ => {}
        }
        Ok(())
    }

    fn new_label(&mut self) -> usize {
        let id = self.next_label;
        self.next_label += 1;
        id
    }

    fn label(&mut self, id: usize) {
        self.labels.insert(id, self.chunk.code.len());
    }

    fn patch_jump(&mut self, label_id: usize) {
        // Patch the most recent Jump/JmpFalse/JmpTrue with 0 offset
        // The offset to patch is 4 bytes before the end (the i32 operand)
        self.patches.push((self.chunk.code.len() - 4, label_id));
    }

    fn resolve_patches(&mut self) {
        for &(patch_pos, label_id) in &self.patches.clone() {
            if let Some(&label_pos) = self.labels.get(&label_id) {
                let offset = label_pos as i32 - patch_pos as i32 - 4;
                let bytes = offset.to_le_bytes();
                for (i, b) in bytes.iter().enumerate() {
                    if patch_pos + i < self.chunk.code.len() {
                        self.chunk.code[patch_pos + i] = *b;
                    }
                }
            }
        }
    }
}