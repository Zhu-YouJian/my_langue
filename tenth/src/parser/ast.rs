use crate::hir::types::BaseType;
use crate::lexer::token::Span;

#[derive(Debug, Clone, PartialEq)]
pub enum Literal {
    /// 整数字面量。第二字段为 dtype（I8/I16/I32/I64/U8/U16/U32/U64），由 lexer 后缀决定，默认 I32。
    Int(i64, BaseType),
    /// 浮点字面量。第二字段为 dtype（F32 或 F64），由 lexer 的 `f32`/`f64` 后缀决定。
    Float(f64, BaseType),
    Bool(bool),
    String(String),
    Char(char),
}

#[derive(Debug, Clone, PartialEq)]
pub struct Ident {
    pub name: String,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub enum TypeAnnotation {
    Named(Ident),
    Generic {
        base: Ident,
        args: Vec<TypeAnnotation>,
    },
    Tensor {
        dtype: Box<TypeAnnotation>,
        dims: Vec<DimSpec>,
    },
    Array { inner: Box<TypeAnnotation>, size: Option<usize> },
    FnType {
        params: Vec<TypeAnnotation>,
        ret: Box<TypeAnnotation>,
    },
    Unit,
    /// Reference type: `&T`, `&'a T`, `&mut T`, `&'a mut T`
    Ref { inner: Box<TypeAnnotation>, mutable: bool, lifetime: Option<String> },
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
    Neg, Not, Try,
}

#[derive(Debug, Clone, PartialEq)]
pub enum InterpPart {
    Literal(String),
    Expr(String),
}

#[derive(Debug, Clone, PartialEq)]
pub enum ExprKind {
    Literal(Literal),
    InterpolatedString(Vec<InterpPart>),
    /// f"..." 模板字符串：与 InterpolatedString 结构相同但语义不同，
    /// 在 HIR 层会转换为 format() 调用。
    FString(Vec<InterpPart>),
    Tuple(Vec<Expr>),
    Ident(Ident),
    Binary {
        op: BinOp,
        left: Box<Expr>,
        right: Box<Expr>,
    },
    /// M3.1：自定义运算符中缀表达式 `a <op> b`。
    /// `op` 为运算符名（如 `@@`），lower 时降级为对绑定函数的调用。
    CustomBinary {
        op: String,
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
        use_defaults: bool,
    },
    EnumLiteral {
        enum_name: Ident,
        variant: Ident,
        fields: Vec<(Ident, Expr)>,
        /// 显式泛型实参：`MyEnum<i64>::Some(5)` 的 `<i64>`（无则为空 Vec，靠推断）
        generics: Vec<TypeAnnotation>,
    },
    Match {
        scrutinee: Box<Expr>,
        arms: Vec<MatchArm>,
    },
    Ref(Box<Expr>),
    MutRef(Box<Expr>),
    Deref(Box<Expr>),
    Move(Box<Expr>),
    /// `lossy expr`：编译期显式接受可能算错的值（污点归零）；运行时求值 inner（no-op）。
    /// 对应 Rust 的 `unsafe`。
    Lossy(Box<Expr>),
    TryBlock(Box<Expr>),
    Await(Box<Expr>),
    Spawn(Box<Expr>),
    /// `yield`（无值，返回 Unit）或 `yield expr`（带值，expr 求值后丢弃，整体返回 Unit）。
    /// 让出控制权给 VM 调度器；解释器路径不支持。
    Yield(Option<Box<Expr>>),
    /// Named argument in a function call: `f(name = value)`.
    /// Only valid inside `Call` / `GenericCall` / `MethodCall` args.
    NamedArg {
        name: Ident,
        value: Box<Expr>,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub struct MatchArm {
    pub pattern: Pattern,
    pub guard: Option<Expr>,
    pub body: Expr,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Pattern {
    EnumVariant {
        enum_name: String,
        variant: String,
        field_bind: Option<(String, String)>,
        /// For tuple variant patterns: list of bind names for positional fields.
        /// e.g. `Some(x)` → ["x"], `Pair(a, b)` → ["a", "b"]
        tuple_fields: Vec<String>,
    },
    Wildcard,
    Literal(Literal),
    /// Tuple destructuring: (a, b, c)
    Tuple(Vec<Pattern>),
    /// Range pattern: start..end or start..=end
    Range { start: i64, end: i64, inclusive: bool },
    /// Variable binding: `x` (catch-all that binds the value)
    Binding(String),
    /// Struct destructuring: `Point { x, y }` or `Point { x: a, y: b }`.
    /// Each entry is (field_name, bind_name).
    Struct { name: String, fields: Vec<(String, String)> },
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
        names: Vec<Ident>,
        type_ann: Option<TypeAnnotation>,
        mutable: bool,
        init: Option<Expr>,
    },
    Expr(Expr),
    Return(Option<Expr>),
    /// M2.3：break。`label` 为 `break 'outer` 的循环标签（复用 Lifetime token），
    /// `value` 为 `break val` 的返回值（可为 None）。
    Break {
        label: Option<String>,
        value: Option<Expr>,
    },
    /// M2.3：continue。`label` 为 `continue 'outer` 的循环标签。
    Continue {
        label: Option<String>,
    },
    While {
        /// M2.3：循环标签 `'outer: while ...`（None 表示无标签）
        label: Option<String>,
        cond: Expr,
        body: Box<Stmt>,
    },
    DoWhile {
        label: Option<String>,
        body: Box<Stmt>,
        condition: Expr,
    },
    For {
        label: Option<String>,
        var: Ident,
        iter: Expr,
        body: Box<Stmt>,
    },
    Loop {
        label: Option<String>,
        body: Vec<Stmt>,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub struct Stmt {
    pub kind: StmtKind,
    pub span: Span,
}

/// A call argument: either positional (name is None) or named (name is Some).
#[derive(Debug, Clone, PartialEq)]
pub struct CallArg {
    pub name: Option<Ident>,
    pub value: Expr,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Param {
    pub name: Ident,
    pub type_ann: TypeAnnotation,
    pub default_value: Option<Expr>,
    pub variadic: bool,
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
        is_pub: bool,
        is_async: bool,
        is_test: bool,
    },
    Const {
        name: Ident,
        type_ann: TypeAnnotation,
        value: Expr,
    },
    Use {
        path: Vec<Ident>,
        glob: bool,  // true for `use path::*`
    },
    StructDef {
        name: Ident,
        generics: Vec<GenericParam>,
        kind: StructKind,
        is_pub: bool,
    },
    EnumDef {
        name: Ident,
        generics: Vec<GenericParam>,
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
        associated_types: Vec<Ident>,
    },
    Union {
        name: Ident,
        fields: Vec<StructField>,
    },
    /// M3.1：自定义运算符声明 `operator <op> = fn(...)`。
    /// `op` 为运算符名（如 `@@`），`func` 为绑定函数（合成名 `__custom_op_<op>`）。
    Operator {
        op: String,
        func: Box<Item>,
    },
    /// M3.3：声明式宏定义 `macro name(param1, param2) { body_expr }`。
    /// body 是表达式模板，参数位置在展开时替换为实参 AST。宏是编译期构造：
    /// `parse_program` 末尾的展开 pass 收集后，调用点 AST 替换为 body（参数代入），
    /// 宏定义本身从 AST 移除，不进 HIR。
    MacroDef {
        name: Ident,
        params: Vec<Ident>,
        body: Expr,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub struct TraitMethod {
    pub name: Ident,
    pub params: Vec<Param>,
    pub return_type: Option<TypeAnnotation>,
    pub body: Option<Expr>,
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

/// 结构体种类：命名结构体 `struct Foo { x: i32 }` 或元组结构体（Newtype）`struct Meters(f64)`
#[derive(Debug, Clone, PartialEq)]
pub enum StructKind {
    Named(Vec<StructField>),
    Tuple(Vec<TypeAnnotation>),
}

#[derive(Debug, Clone, PartialEq)]
pub enum EnumVariantKind {
    Unit,
    Named(Vec<StructField>),
    Tuple(Vec<TypeAnnotation>),
}

#[derive(Debug, Clone, PartialEq)]
pub struct EnumVariant {
    pub name: Ident,
    pub kind: EnumVariantKind,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Program {
    pub items: Vec<Item>,
}