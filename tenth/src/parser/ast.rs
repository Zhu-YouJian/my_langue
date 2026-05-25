use crate::lexer::token::Span;

#[derive(Debug, Clone, PartialEq)]
pub enum Literal {
    Int(i64),
    Float(f64),
    Bool(bool),
    String(String),
}

#[derive(Debug, Clone, PartialEq)]
pub struct Ident {
    pub name: String,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub enum TypeAnnotation {
    Named(Ident),
    Tensor {
        dtype: Box<TypeAnnotation>,
        dims: Vec<DimSpec>,
    },
    Array(Box<TypeAnnotation>),
    FnType {
        params: Vec<TypeAnnotation>,
        ret: Box<TypeAnnotation>,
    },
    Unit,
}

#[derive(Debug, Clone, PartialEq)]
pub enum DimSpec {
    Literal(i64),
    Symbol(String),
    Wildcard,
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
pub enum ExprKind {
    Literal(Literal),
    Ident(Ident),
    Binary {
        op: BinOp,
        left: Box<Expr>,
        right: Box<Expr>,
    },
    Unary {
        op: UnaryOp,
        expr: Box<Expr>,
    },
    Call {
        func: Box<Expr>,
        args: Vec<Expr>,
    },
    GenericCall {
        func: Box<Expr>,
        generics: Vec<TypeAnnotation>,
        args: Vec<Expr>,
    },
    MethodCall {
        receiver: Box<Expr>,
        method: Ident,
        args: Vec<Expr>,
    },
    Index {
        target: Box<Expr>,
        indices: Vec<IndexExpr>,
    },
    Field {
        target: Box<Expr>,
        field: Ident,
    },
    TensorLiteral(Vec<Vec<Expr>>),
    ArrayLiteral(Vec<Expr>),
    Range {
        start: Option<Box<Expr>>,
        end: Option<Box<Expr>>,
        inclusive: bool,
    },
    If {
        cond: Box<Expr>,
        then_branch: Box<Expr>,
        else_branch: Option<Box<Expr>>,
    },
    Block(Vec<Stmt>),
    Closure {
        params: Vec<(Ident, Option<TypeAnnotation>)>,
        body: Box<Expr>,
    },
    Assign {
        target: Box<Expr>,
        value: Box<Expr>,
    },
    AssignOp {
        target: Box<Expr>,
        op: BinOp,
        value: Box<Expr>,
    },
    StructLiteral {
        name: Ident,
        generics: Vec<TypeAnnotation>,
        fields: Vec<(Ident, Expr)>,
    },
    EnumLiteral {
        enum_name: Ident,
        variant: Ident,
        fields: Vec<(Ident, Expr)>,
    },
    Match {
        scrutinee: Box<Expr>,
        arms: Vec<MatchArm>,
    },
    Ref(Box<Expr>),
    MutRef(Box<Expr>),
    Deref(Box<Expr>),
    Move(Box<Expr>),
}

#[derive(Debug, Clone, PartialEq)]
pub struct MatchArm {
    pub pattern: Pattern,
    pub body: Expr,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Pattern {
    EnumVariant {
        enum_name: String,
        variant: String,
        field_bind: Option<(String, String)>,
    },
    Wildcard,
    Literal(Literal),
}

#[derive(Debug, Clone, PartialEq)]
pub struct Expr {
    pub kind: ExprKind,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub enum IndexExpr {
    Single(Expr),
    Range {
        start: Option<Box<Expr>>,
        end: Option<Box<Expr>>,
    },
    Colon,
}

#[derive(Debug, Clone, PartialEq)]
pub enum StmtKind {
    Let {
        name: Ident,
        type_ann: Option<TypeAnnotation>,
        mutable: bool,
        init: Option<Expr>,
    },
    Expr(Expr),
    Return(Option<Expr>),
    Break,
    Continue,
    While {
        cond: Expr,
        body: Box<Stmt>,
    },
    For {
        var: Ident,
        iter: Expr,
        body: Box<Stmt>,
    },
    Loop {
        body: Vec<Stmt>,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub struct Stmt {
    pub kind: StmtKind,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Param {
    pub name: Ident,
    pub type_ann: TypeAnnotation,
}

#[derive(Debug, Clone, PartialEq)]
pub struct GenericParam {
    pub name: Ident,
    pub bounds: Vec<Ident>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ItemKind {
    Function {
        name: Ident,
        generics: Vec<GenericParam>,
        params: Vec<Param>,
        return_type: Option<TypeAnnotation>,
        body: Expr,
    },
    Const {
        name: Ident,
        type_ann: TypeAnnotation,
        value: Expr,
    },
    Use {
        path: Vec<Ident>,
    },
    StructDef {
        name: Ident,
        generics: Vec<GenericParam>,
        fields: Vec<StructField>,
    },
    EnumDef {
        name: Ident,
        variants: Vec<EnumVariant>,
    },
    Impl {
        type_name: Ident,
        trait_name: Option<Ident>,
        generics: Vec<GenericParam>,
        functions: Vec<Item>,
    },
    Mod {
        name: Ident,
        items: Vec<Item>,
    },
    Trait {
        name: Ident,
        generics: Vec<GenericParam>,
        methods: Vec<TraitMethod>,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub struct TraitMethod {
    pub name: Ident,
    pub params: Vec<Param>,
    pub return_type: Option<TypeAnnotation>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Item {
    pub kind: ItemKind,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct StructField {
    pub name: Ident,
    pub type_ann: TypeAnnotation,
}

#[derive(Debug, Clone, PartialEq)]
pub struct EnumVariant {
    pub name: Ident,
    pub fields: Vec<StructField>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Program {
    pub items: Vec<Item>,
}