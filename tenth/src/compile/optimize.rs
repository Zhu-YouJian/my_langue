use super::mir::*;
use crate::hir::hir::{BinOp, UnaryOp};
use crate::hir::types::Type;
use std::collections::HashSet;

pub trait OptimizationPass {
    fn name(&self) -> &str;
    fn run(&self, program: &mut MirProgram);
}

pub fn optimize(program: &mut MirProgram, passes: &[Box<dyn OptimizationPass>]) {
    for pass in passes { pass.run(program); }
}

// Helper to wrap kind with type
fn rv(ty: Type, kind: MirRvalueKind) -> MirRvalue { MirRvalue { kind, ty } }

pub struct ConstantFolding;
impl OptimizationPass for ConstantFolding {
    fn name(&self) -> &str { "constant_folding" }
    fn run(&self, program: &mut MirProgram) {
        for func in &mut program.functions {
            for block in &mut func.blocks {
                for stmt in &mut block.stmts { fold_stmt(stmt); }
                fold_terminator(&mut block.terminator);
            }
        }
        if let Some(ref mut m) = program.main_expr {
            for block in &mut m.blocks {
                for stmt in &mut block.stmts { fold_stmt(stmt); }
                fold_terminator(&mut block.terminator);
            }
        }
    }
}

fn fold_stmt(stmt: &mut MirStmt) {
    match stmt {
        MirStmt::Let { value, .. } => *value = fold_rvalue_ref(value),
        MirStmt::Assign { value, .. } => *value = fold_rvalue_ref(value),
        MirStmt::Expr(val) => *val = fold_rvalue_ref(val),
        MirStmt::Return(Some(val)) => *val = fold_rvalue_ref(val),
        _ => {}
    }
}

fn fold_terminator(term: &mut MirTerminator) {
    match term {
        MirTerminator::Return(Some(val)) => *val = fold_rvalue_ref(val),
        MirTerminator::If { cond, .. } => *cond = fold_rvalue_ref(cond),
        _ => {}
    }
}

fn fold_rvalue_ref(val: &MirRvalue) -> MirRvalue { fold_rvalue(val) }

fn fold_rvalue(val: &MirRvalue) -> MirRvalue {
    let ty = val.ty.clone();
    match &val.kind {
        MirRvalueKind::BinaryOp(op, left, right) => {
            let op_clone = op.clone();
            let l = fold_rvalue(left); let r = fold_rvalue(right);
            match (&l.kind, &r.kind) {
                (MirRvalueKind::Literal(a), MirRvalueKind::Literal(b)) => {
                    eval_const_binop(&op_clone, a, b).unwrap_or(rv(ty, MirRvalueKind::BinaryOp(op_clone, Box::new(l), Box::new(r))))
                }
                _ => rv(ty, MirRvalueKind::BinaryOp(op_clone, Box::new(l), Box::new(r))),
            }
        }
        MirRvalueKind::UnaryOp(op, expr) => {
            let op_clone = op.clone(); let e = fold_rvalue(expr);
            match (&e.kind, &op_clone) {
                (MirRvalueKind::Literal(lit), UnaryOp::Neg) => match lit {
                    LiteralValue::Int(n) => rv(ty, MirRvalueKind::Literal(LiteralValue::Int(-n))),
                    LiteralValue::Float(n) => rv(ty, MirRvalueKind::Literal(LiteralValue::Float(-n))),
                    _ => rv(ty, MirRvalueKind::UnaryOp(op_clone, Box::new(e))),
                },
                (MirRvalueKind::Literal(LiteralValue::Bool(b)), UnaryOp::Not) => {
                    rv(ty, MirRvalueKind::Literal(LiteralValue::Bool(!b)))
                }
                _ => rv(ty, MirRvalueKind::UnaryOp(op_clone, Box::new(e))),
            }
        }
        MirRvalueKind::If { cond, then_block, else_block } => {
            let c = fold_rvalue(cond);
            if let MirRvalueKind::Literal(LiteralValue::Bool(true)) = &c.kind {
                rv(ty, MirRvalueKind::Literal(LiteralValue::Int(*then_block as i64)))
            } else if let MirRvalueKind::Literal(LiteralValue::Bool(false)) = &c.kind {
                if let Some(eb) = else_block { rv(ty, MirRvalueKind::Literal(LiteralValue::Int(*eb as i64))) }
                else { rv(ty, MirRvalueKind::Literal(LiteralValue::Int(0))) }
            } else {
                rv(ty, MirRvalueKind::If { cond: Box::new(c), then_block: *then_block, else_block: *else_block })
            }
        }
        _ => val.clone(),
    }
}

fn eval_const_binop(op: &BinOp, a: &LiteralValue, b: &LiteralValue) -> Option<MirRvalue> {
    let ty = Type::Unknown;
    match (a, b) {
        (LiteralValue::Int(x), LiteralValue::Int(y)) => Some(rv(ty, match op {
            BinOp::Add => MirRvalueKind::Literal(LiteralValue::Int(x + y)),
            BinOp::Sub => MirRvalueKind::Literal(LiteralValue::Int(x - y)),
            BinOp::Mul => MirRvalueKind::Literal(LiteralValue::Int(x * y)),
            BinOp::Div => MirRvalueKind::Literal(LiteralValue::Int(x / y)),
            BinOp::Mod => MirRvalueKind::Literal(LiteralValue::Int(x % y)),
            BinOp::Eq => MirRvalueKind::Literal(LiteralValue::Bool(x == y)),
            BinOp::NotEq => MirRvalueKind::Literal(LiteralValue::Bool(x != y)),
            BinOp::Lt => MirRvalueKind::Literal(LiteralValue::Bool(x < y)),
            BinOp::Gt => MirRvalueKind::Literal(LiteralValue::Bool(x > y)),
            BinOp::LtEq => MirRvalueKind::Literal(LiteralValue::Bool(x <= y)),
            BinOp::GtEq => MirRvalueKind::Literal(LiteralValue::Bool(x >= y)),
            _ => return None,
        })),
        (LiteralValue::Float(x), LiteralValue::Float(y)) => Some(rv(ty, match op {
            BinOp::Add => MirRvalueKind::Literal(LiteralValue::Float(x + y)),
            BinOp::Sub => MirRvalueKind::Literal(LiteralValue::Float(x - y)),
            BinOp::Mul => MirRvalueKind::Literal(LiteralValue::Float(x * y)),
            BinOp::Div => MirRvalueKind::Literal(LiteralValue::Float(x / y)),
            BinOp::Eq => MirRvalueKind::Literal(LiteralValue::Bool((x - y).abs() < 1e-10)),
            BinOp::NotEq => MirRvalueKind::Literal(LiteralValue::Bool((x - y).abs() >= 1e-10)),
            BinOp::Lt => MirRvalueKind::Literal(LiteralValue::Bool(x < y)),
            BinOp::Gt => MirRvalueKind::Literal(LiteralValue::Bool(x > y)),
            BinOp::LtEq => MirRvalueKind::Literal(LiteralValue::Bool(x <= y)),
            BinOp::GtEq => MirRvalueKind::Literal(LiteralValue::Bool(x >= y)),
            _ => return None,
        })),
        (LiteralValue::Int(x), LiteralValue::Float(y)) => eval_const_binop(op, &LiteralValue::Float(*x as f64), &LiteralValue::Float(*y)),
        (LiteralValue::Float(x), LiteralValue::Int(y)) => eval_const_binop(op, &LiteralValue::Float(*x), &LiteralValue::Float(*y as f64)),
        (LiteralValue::Bool(x), LiteralValue::Bool(y)) => Some(rv(ty, match op {
            BinOp::And => MirRvalueKind::Literal(LiteralValue::Bool(*x && *y)),
            BinOp::Or => MirRvalueKind::Literal(LiteralValue::Bool(*x || *y)),
            BinOp::Eq => MirRvalueKind::Literal(LiteralValue::Bool(x == y)),
            BinOp::NotEq => MirRvalueKind::Literal(LiteralValue::Bool(x != y)),
            _ => return None,
        })),
        _ => None,
    }
}

pub struct DeadCodeElimination;
impl OptimizationPass for DeadCodeElimination {
    fn name(&self) -> &str { "dead_code_elimination" }
    fn run(&self, program: &mut MirProgram) {
        for func in &mut program.functions { eliminate_dead(func); }
        if let Some(ref mut m) = program.main_expr { eliminate_dead(m); }
    }
}

fn eliminate_dead(func: &mut MirFunction) {
    let mut used = HashSet::new();
    for block in &func.blocks {
        for stmt in &block.stmts { collect_uses_stmt(stmt, &mut used); }
        collect_uses_term(&block.terminator, &mut used);
    }
    for block in &mut func.blocks {
        block.stmts.retain(|s| match s { MirStmt::Let { name, .. } => used.contains(name), _ => true });
    }
}

fn collect_uses_stmt(stmt: &MirStmt, used: &mut HashSet<String>) {
    match stmt {
        MirStmt::Assign { value, .. } => collect_uses_rvalue(value, used),
        MirStmt::Expr(val) => collect_uses_rvalue(val, used),
        MirStmt::Return(Some(val)) => collect_uses_rvalue(val, used),
        _ => {}
    }
}

fn collect_uses_term(term: &MirTerminator, used: &mut HashSet<String>) {
    match term {
        MirTerminator::Return(Some(val)) => collect_uses_rvalue(val, used),
        MirTerminator::If { cond, .. } => collect_uses_rvalue(cond, used),
        _ => {}
    }
}

fn collect_uses_rvalue(val: &MirRvalue, used: &mut HashSet<String>) {
    match &val.kind {
        MirRvalueKind::Use(name) | MirRvalueKind::Ref(name) | MirRvalueKind::MutRef(name) | MirRvalueKind::Deref(name) | MirRvalueKind::Move(name) => { used.insert(name.clone()); }
        MirRvalueKind::BinaryOp(_, l, r) => { collect_uses_rvalue(l, used); collect_uses_rvalue(r, used); }
        MirRvalueKind::UnaryOp(_, e) | MirRvalueKind::Field { target: e, .. } => collect_uses_rvalue(e, used),
        MirRvalueKind::Call { args, .. } | MirRvalueKind::MethodCall { args, .. } => { for a in args { collect_uses_rvalue(a, used); } }
        MirRvalueKind::If { cond, .. } => collect_uses_rvalue(cond, used),
        MirRvalueKind::StructLiteral { fields, .. } => { for (_, f) in fields { collect_uses_rvalue(f, used); } }
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rv_test(kind: MirRvalueKind) -> MirRvalue { rv(Type::Unknown, kind) }

    #[test]
    fn test_const_fold_int_arithmetic() {
        let val = rv_test(MirRvalueKind::BinaryOp(BinOp::Add,
            Box::new(rv_test(MirRvalueKind::Literal(LiteralValue::Int(2)))),
            Box::new(rv_test(MirRvalueKind::BinaryOp(BinOp::Mul,
                Box::new(rv_test(MirRvalueKind::Literal(LiteralValue::Int(3)))),
                Box::new(rv_test(MirRvalueKind::Literal(LiteralValue::Int(4)))))))));
        let result = fold_rvalue(&val);
        match &result.kind {
            MirRvalueKind::Literal(LiteralValue::Int(14)) => {}
            v => panic!("expected Int(14), got {:?}", v),
        }
    }

    #[test]
    fn test_const_fold_bool_and() {
        let val = rv_test(MirRvalueKind::BinaryOp(BinOp::And,
            Box::new(rv_test(MirRvalueKind::Literal(LiteralValue::Bool(true)))),
            Box::new(rv_test(MirRvalueKind::Literal(LiteralValue::Bool(false))))));
        let result = fold_rvalue(&val);
        match &result.kind {
            MirRvalueKind::Literal(LiteralValue::Bool(false)) => {}
            v => panic!("expected Bool(false), got {:?}", v),
        }
    }
}
