use super::mir::*;
use crate::hir::hir::{BinOp, UnaryOp};
use std::collections::HashSet;

/// Optimization pass trait.
pub trait OptimizationPass {
    fn name(&self) -> &str;
    fn run(&self, program: &mut MirProgram);
}

/// Run a sequence of optimization passes on the MIR program.
pub fn optimize(program: &mut MirProgram, passes: &[Box<dyn OptimizationPass>]) {
    for pass in passes {
        pass.run(program);
    }
}

// ── Constant Folding ──────────────────────────────────────────────

pub struct ConstantFolding;

impl OptimizationPass for ConstantFolding {
    fn name(&self) -> &str { "constant_folding" }

    fn run(&self, program: &mut MirProgram) {
        for func in &mut program.functions {
            for block in &mut func.blocks {
                for stmt in &mut block.stmts {
                    fold_stmt(stmt);
                }
                fold_terminator(&mut block.terminator);
            }
        }
        if let Some(ref mut main_fn) = program.main_expr {
            for block in &mut main_fn.blocks {
                for stmt in &mut block.stmts {
                    fold_stmt(stmt);
                }
                fold_terminator(&mut block.terminator);
            }
        }
    }
}

fn fold_stmt(stmt: &mut MirStmt) {
    match stmt {
        MirStmt::Let { value, .. } => *value = fold_rvalue(value),
        MirStmt::Assign { value, .. } => *value = fold_rvalue(value),
        MirStmt::Expr(val) => *val = fold_rvalue(val),
        MirStmt::Return(Some(val)) => *val = fold_rvalue(val),
        _ => {}
    }
}

fn fold_terminator(term: &mut MirTerminator) {
    match term {
        MirTerminator::Return(Some(val)) => *val = fold_rvalue(val),
        MirTerminator::If { cond, .. } => *cond = fold_rvalue(cond),
        _ => {}
    }
}

fn fold_rvalue(val: &mut MirRvalue) -> MirRvalue {
    match val {
        MirRvalue::BinaryOp(op, left, right) => {
            let op_clone = op.clone();
            let l = fold_rvalue(left);
            let r = fold_rvalue(right);
            match (&l, &r) {
                (MirRvalue::Literal(a), MirRvalue::Literal(b)) => {
                    eval_const_binop(&op_clone, a, b).unwrap_or(MirRvalue::BinaryOp(op_clone, Box::new(l), Box::new(r)))
                }
                _ => MirRvalue::BinaryOp(op_clone, Box::new(l), Box::new(r)),
            }
        }
        MirRvalue::UnaryOp(op, expr) => {
            let op_clone = op.clone();
            let e = fold_rvalue(expr);
            match (&e, &op_clone) {
                (MirRvalue::Literal(lit), UnaryOp::Neg) => {
                    match lit {
                        LiteralValue::Int(n) => MirRvalue::Literal(LiteralValue::Int(-n)),
                        LiteralValue::Float(n) => MirRvalue::Literal(LiteralValue::Float(-n)),
                        _ => MirRvalue::UnaryOp(op_clone, Box::new(e)),
                    }
                }
                (MirRvalue::Literal(LiteralValue::Bool(b)), UnaryOp::Not) => {
                    MirRvalue::Literal(LiteralValue::Bool(!b))
                }
                _ => MirRvalue::UnaryOp(op_clone, Box::new(e)),
            }
        }
        MirRvalue::If { cond, then_block, else_block } => {
            let c = fold_rvalue(cond);
            if let MirRvalue::Literal(LiteralValue::Bool(true)) = &c {
                MirRvalue::Literal(LiteralValue::Int(*then_block as i64))
            } else if let MirRvalue::Literal(LiteralValue::Bool(false)) = &c {
                if let Some(eb) = else_block {
                    MirRvalue::Literal(LiteralValue::Int(*eb as i64))
                } else {
                    MirRvalue::Literal(LiteralValue::Int(0))
                }
            } else {
                MirRvalue::If { cond: Box::new(c), then_block: *then_block, else_block: *else_block }
            }
        }
        other => other.clone(),
    }
}

fn eval_const_binop(op: &BinOp, a: &LiteralValue, b: &LiteralValue) -> Option<MirRvalue> {
    match (a, b) {
        (LiteralValue::Int(x), LiteralValue::Int(y)) => match op {
            BinOp::Add => Some(MirRvalue::Literal(LiteralValue::Int(x + y))),
            BinOp::Sub => Some(MirRvalue::Literal(LiteralValue::Int(x - y))),
            BinOp::Mul => Some(MirRvalue::Literal(LiteralValue::Int(x * y))),
            BinOp::Div => Some(MirRvalue::Literal(LiteralValue::Int(x / y))),
            BinOp::Mod => Some(MirRvalue::Literal(LiteralValue::Int(x % y))),
            BinOp::Eq => Some(MirRvalue::Literal(LiteralValue::Bool(x == y))),
            BinOp::NotEq => Some(MirRvalue::Literal(LiteralValue::Bool(x != y))),
            BinOp::Lt => Some(MirRvalue::Literal(LiteralValue::Bool(x < y))),
            BinOp::Gt => Some(MirRvalue::Literal(LiteralValue::Bool(x > y))),
            BinOp::LtEq => Some(MirRvalue::Literal(LiteralValue::Bool(x <= y))),
            BinOp::GtEq => Some(MirRvalue::Literal(LiteralValue::Bool(x >= y))),
            _ => None,
        },
        (LiteralValue::Float(x), LiteralValue::Float(y)) => match op {
            BinOp::Add => Some(MirRvalue::Literal(LiteralValue::Float(x + y))),
            BinOp::Sub => Some(MirRvalue::Literal(LiteralValue::Float(x - y))),
            BinOp::Mul => Some(MirRvalue::Literal(LiteralValue::Float(x * y))),
            BinOp::Div => Some(MirRvalue::Literal(LiteralValue::Float(x / y))),
            BinOp::Eq => Some(MirRvalue::Literal(LiteralValue::Bool((x - y).abs() < 1e-10))),
            BinOp::NotEq => Some(MirRvalue::Literal(LiteralValue::Bool((x - y).abs() >= 1e-10))),
            BinOp::Lt => Some(MirRvalue::Literal(LiteralValue::Bool(x < y))),
            BinOp::Gt => Some(MirRvalue::Literal(LiteralValue::Bool(x > y))),
            BinOp::LtEq => Some(MirRvalue::Literal(LiteralValue::Bool(x <= y))),
            BinOp::GtEq => Some(MirRvalue::Literal(LiteralValue::Bool(x >= y))),
            _ => None,
        },
        (LiteralValue::Int(x), LiteralValue::Float(y)) => {
            let xf = *x as f64;
            eval_const_binop(op, &LiteralValue::Float(xf), &LiteralValue::Float(*y))
        }
        (LiteralValue::Float(x), LiteralValue::Int(y)) => {
            let yf = *y as f64;
            eval_const_binop(op, &LiteralValue::Float(*x), &LiteralValue::Float(yf))
        }
        (LiteralValue::Bool(x), LiteralValue::Bool(y)) => match op {
            BinOp::And => Some(MirRvalue::Literal(LiteralValue::Bool(*x && *y))),
            BinOp::Or => Some(MirRvalue::Literal(LiteralValue::Bool(*x || *y))),
            BinOp::Eq => Some(MirRvalue::Literal(LiteralValue::Bool(x == y))),
            BinOp::NotEq => Some(MirRvalue::Literal(LiteralValue::Bool(x != y))),
            _ => None,
        },
        _ => None,
    }
}

// ── Dead Code Elimination ─────────────────────────────────────────

pub struct DeadCodeElimination;

impl OptimizationPass for DeadCodeElimination {
    fn name(&self) -> &str { "dead_code_elimination" }

    fn run(&self, program: &mut MirProgram) {
        for func in &mut program.functions {
            eliminate_dead_code_in_function(func);
        }
        if let Some(ref mut main_fn) = program.main_expr {
            eliminate_dead_code_in_function(main_fn);
        }
    }
}

fn eliminate_dead_code_in_function(func: &mut MirFunction) {
    // Find all variables that are actually read
    let mut used: HashSet<String> = HashSet::new();
    for block in &func.blocks {
        for stmt in &block.stmts {
            collect_uses_stmt(stmt, &mut used);
        }
        collect_uses_terminator(&block.terminator, &mut used);
    }

    // Remove Let statements for unused variables
    for block in &mut func.blocks {
        block.stmts.retain(|stmt| {
            match stmt {
                MirStmt::Let { name, .. } => used.contains(name),
                _ => true,
            }
        });
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

fn collect_uses_terminator(term: &MirTerminator, used: &mut HashSet<String>) {
    match term {
        MirTerminator::Return(Some(val)) => collect_uses_rvalue(val, used),
        MirTerminator::If { cond, .. } => collect_uses_rvalue(cond, used),
        _ => {}
    }
}

fn collect_uses_rvalue(val: &MirRvalue, used: &mut HashSet<String>) {
    match val {
        MirRvalue::Use(name) | MirRvalue::Ref(name) | MirRvalue::MutRef(name) | MirRvalue::Deref(name) | MirRvalue::Move(name) => {
            used.insert(name.clone());
        }
        MirRvalue::BinaryOp(_, l, r) => {
            collect_uses_rvalue(l, used);
            collect_uses_rvalue(r, used);
        }
        MirRvalue::UnaryOp(_, e) => collect_uses_rvalue(e, used),
        MirRvalue::Call { args, .. } | MirRvalue::MethodCall { args, .. } => {
            for a in args { collect_uses_rvalue(a, used); }
        }
        MirRvalue::If { cond, .. } => collect_uses_rvalue(cond, used),
        MirRvalue::StructLiteral { fields, .. } => {
            for (_, v) in fields { collect_uses_rvalue(v, used); }
        }
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_const_fold_int_arithmetic() {
        // 2 + 3 * 4 should fold to 14
        let mut val = MirRvalue::BinaryOp(
            BinOp::Add,
            Box::new(MirRvalue::Literal(LiteralValue::Int(2))),
            Box::new(MirRvalue::BinaryOp(
                BinOp::Mul,
                Box::new(MirRvalue::Literal(LiteralValue::Int(3))),
                Box::new(MirRvalue::Literal(LiteralValue::Int(4))),
            )),
        );
        let result = fold_rvalue(&mut val);
        match result {
            MirRvalue::Literal(LiteralValue::Int(14)) => {}
            v => panic!("expected Int(14), got {:?}", v),
        }
    }

    #[test]
    fn test_const_fold_bool_and() {
        let mut val = MirRvalue::BinaryOp(
            BinOp::And,
            Box::new(MirRvalue::Literal(LiteralValue::Bool(true))),
            Box::new(MirRvalue::Literal(LiteralValue::Bool(false))),
        );
        let result = fold_rvalue(&mut val);
        match result {
            MirRvalue::Literal(LiteralValue::Bool(false)) => {}
            v => panic!("expected Bool(false), got {:?}", v),
        }
    }
}
