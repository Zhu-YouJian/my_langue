use std::collections::HashMap;
use crate::lexer::token::Span;
use super::types::Type;

#[derive(Debug, Clone, PartialEq)]
pub enum HirExprKind {
    Literal(Literal),
    Var(String),
    Binary {
        op: BinOp,
        left: Box<HirExpr>,
        right: Box<HirExpr>,
        ty: Type,
    },
    Unary {
        op: UnaryOp,
        expr: Box<HirExpr>,
        ty: Type,
    },
    Call {
        func: Box<HirExpr>,
        args: Vec<HirExpr>,
        ret_ty: Type,
    },
    GenericCall {
        func: Box<HirExpr>,
        generics: Vec<Type>,
        args: Vec<HirExpr>,
        ret_ty: Type,
    },
    MethodCall {
        receiver: Box<HirExpr>,
        method: String,
        args: Vec<HirExpr>,
        ret_ty: Type,
    },
    Index {
        target: Box<HirExpr>,
        indices: Vec<Index>,
    },
    Field {
        target: Box<HirExpr>,
        field: String,
    },
    TensorLiteral {
        data: Vec<Vec<HirExpr>>,
        ty: Type,
    },
    ArrayLiteral {
        elements: Vec<HirExpr>,
        ty: Type,
    },
    Range {
        start: Option<Box<HirExpr>>,
        end: Option<Box<HirExpr>>,
        inclusive: bool,
    },
    If {
        cond: Box<HirExpr>,
        then_branch: Box<HirExpr>,
        else_branch: Option<Box<HirExpr>>,
        ty: Type,
    },
    Block {
        stmts: Vec<HirStmt>,
        final_expr: Option<Box<HirExpr>>,
    },
    Closure {
        params: Vec<(String, Type)>,
        body: Box<HirExpr>,
        captures: Vec<String>,
    },
    Assign {
        target: String,
        value: Box<HirExpr>,
    },
    AssignOp {
        target: String,
        op: BinOp,
        value: Box<HirExpr>,
    },
    StructLiteral {
        name: String,
        fields: Vec<(String, HirExpr)>,
        has_default: bool,
    },
    EnumLiteral {
        enum_name: String,
        variant: String,
        fields: Vec<(String, HirExpr)>,
    },
    Match {
        scrutinee: Box<HirExpr>,
        arms: Vec<HirMatchArm>,
    },
    Ref(Box<HirExpr>),
    MutRef(Box<HirExpr>),
    Deref(Box<HirExpr>),
    DerefAssign {
        target: Box<HirExpr>,
        value: Box<HirExpr>,
    },
    DerefAssignOp {
        target: Box<HirExpr>,
        op: BinOp,
        value: Box<HirExpr>,
    },
    Move(Box<HirExpr>),
    FieldAssign {
        target: Box<HirExpr>,
        field: String,
        value: Box<HirExpr>,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub struct HirMatchArm {
    pub pattern: HirPattern,
    pub body: HirExpr,
}

#[derive(Debug, Clone, PartialEq)]
pub enum HirPattern {
    EnumVariant {
        enum_name: String,
        variant: String,
        field_bind: Option<(String, String)>,
        /// For tuple variants: list of (field_name, bind_name) pairs.
        /// e.g. `Some(x)` → [("_0", "x")], `Pair(a, b)` → [("_0", "a"), ("_1", "b")]
        tuple_binds: Vec<(String, String)>,
    },
    Wildcard,
    Literal(Literal),
}

#[derive(Debug, Clone, PartialEq)]
pub struct HirExpr {
    pub kind: HirExprKind,
    pub ty: Type,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Literal {
    Int(i64),
    Float(f64),
    Bool(bool),
    String(String),
}

#[derive(Debug, Clone, PartialEq)]
pub enum BinOp {
    Add, Sub, Mul, Div, Mod,
    Eq, NotEq, Lt, Gt, LtEq, GtEq,
    And, Or,
}

#[derive(Debug, Clone, PartialEq)]
pub enum UnaryOp {
    Neg, Not,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Index {
    Single(HirExpr),
    Range {
        start: Option<Box<HirExpr>>,
        end: Option<Box<HirExpr>>,
    },
    Colon,
}

#[derive(Debug, Clone, PartialEq)]
pub enum HirStmtKind {
    Let {
        name: String,
        type_ann: Option<Type>,
        mutable: bool,
        init: Option<HirExpr>,
    },
    Expr(HirExpr),
    Return(Option<HirExpr>),
    While {
        cond: HirExpr,
        body: Box<HirStmt>,
    },
    For {
        var: String,
        iter: HirExpr,
        body: Box<HirStmt>,
    },
    Break,
    Continue,
    Loop {
        body: Vec<HirStmt>,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub struct HirStmt {
    pub kind: HirStmtKind,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct HirFnDef {
    pub name: String,
    pub generics: Vec<String>,
    pub generics_bounds: HashMap<String, Vec<String>>,
    pub params: Vec<(String, Type)>,
    pub return_type: Type,
    pub body: HirExpr,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct HirGenericStruct {
    pub name: String,
    pub generics: Vec<String>,
    pub fields: Vec<(String, Type)>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct HirProgram {
    pub functions: Vec<HirFnDef>,
    pub generic_funcs: Vec<HirFnDef>,
    pub main_expr: Option<HirExpr>,
    pub modules: HashMap<String, HirProgram>,
    pub uses: Vec<(Vec<String>, String)>,
    pub methods: HashMap<String, HashMap<String, HirFnDef>>,
    pub structs: HashMap<String, Vec<(String, Type)>>,
    pub generic_structs: HashMap<String, HirGenericStruct>,
    pub enums: HashMap<String, Vec<(String, Vec<(String, Type)>)>>,
    pub trait_defs: HashMap<String, HirTraitDef>,
    pub trait_impls: HashMap<String, HashMap<String, HashMap<String, HirFnDef>>>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct HirTraitDef {
    pub name: String,
    pub generics: Vec<String>,
    pub methods: Vec<(String, Vec<(String, Type)>, Type)>,
}