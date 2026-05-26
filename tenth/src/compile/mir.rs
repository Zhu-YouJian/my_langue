use crate::hir::hir::*;
use crate::hir::types::Type;

/// Mid-level Intermediate Representation
#[derive(Debug, Clone)]
pub struct MirFunction {
    pub name: String,
    pub params: Vec<(String, Type)>,
    pub return_type: Type,
    pub locals: Vec<MirLocal>,
    pub blocks: Vec<BasicBlock>,
}

#[derive(Debug, Clone)]
pub struct MirLocal {
    pub name: String,
    pub ty: Type,
    pub mutable: bool,
}

#[derive(Debug, Clone)]
pub struct BasicBlock {
    pub id: usize,
    pub stmts: Vec<MirStmt>,
    pub terminator: MirTerminator,
}

#[derive(Debug, Clone)]
pub enum MirStmt {
    Let { name: String, ty: Type, value: MirRvalue },
    Assign { name: String, value: MirRvalue },
    FieldAssign { target: MirRvalue, field: String, value: MirRvalue },
    Expr(MirRvalue),
    Return(Option<MirRvalue>),
    IfElse { cond: MirRvalue, then_body: Vec<MirStmt>, else_body: Vec<MirStmt> },
    While { cond: MirRvalue, body: Vec<MirStmt> },
    Loop { body: Vec<MirStmt> },
    Break,
    Continue,
}

/// MirRvalue wraps a kind with its type
#[derive(Debug, Clone)]
pub struct MirRvalue {
    pub kind: MirRvalueKind,
    pub ty: Type,
}

#[derive(Debug, Clone)]
pub enum MirRvalueKind {
    Literal(LiteralValue),
    Use(String),
    BinaryOp(BinOp, Box<MirRvalue>, Box<MirRvalue>),
    UnaryOp(UnaryOp, Box<MirRvalue>),
    Call { func: String, args: Vec<MirRvalue> },
    MethodCall { receiver: Box<MirRvalue>, method: String, args: Vec<MirRvalue> },
    StructLiteral { name: String, fields: Vec<(String, MirRvalue)> },
    Field { target: Box<MirRvalue>, field: String },
    Ref(String), MutRef(String), Deref(String), Move(String),
    If { cond: Box<MirRvalue>, then_block: usize, else_block: Option<usize> },
    IfExpr { cond: Box<MirRvalue>, then_val: Box<MirRvalue>, else_val: Box<MirRvalue> },
}

#[derive(Debug, Clone)]
pub enum LiteralValue {
    Int(i64), Float(f64), Bool(bool), Str(String),
}

#[derive(Debug, Clone)]
pub enum MirTerminator {
    Return(Option<MirRvalue>),
    Goto(usize),
    If { cond: MirRvalue, then_block: usize, else_block: usize },
    Unreachable,
}

#[derive(Debug, Clone)]
pub struct MirProgram {
    pub functions: Vec<MirFunction>,
    pub main_expr: Option<MirFunction>,
    pub struct_defs: Vec<(String, Vec<(String, Type)>)>,
    pub enum_defs: Vec<(String, Vec<String>)>,
}
