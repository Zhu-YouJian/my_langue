use std::collections::HashMap;
use crate::lexer::token::Span;
use super::types::{BaseType, Type};

#[derive(Debug, Clone, PartialEq)]
pub enum InterpPart {
    Literal(String),
    Expr(String),
}

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
    /// Union 构造：带 active_field 的 tagged union。
    /// `MyUnion { field: value }` 只激活一个字段 → 运行时 Value::Union。
    UnionLiteral {
        name: String,
        active_field: String,
        value: Box<HirExpr>,
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
    /// `lossy expr`：编译期显式接受可能算错的值（污点归零）；运行时求值 inner（no-op）。
    /// bytecode/wasm 编译为 inner 表达式本身，无附加指令。
    Lossy(Box<HirExpr>),
    TryBlock(Box<HirExpr>),
    Await(Box<HirExpr>),
    Spawn(Box<HirExpr>),
    /// `yield` / `yield expr`：让出控制权给 VM 调度器；恢复后返回 Unit。
    /// inner 若存在会被求值但结果被丢弃（与 VM Op::Yield 语义一致：不消费栈）。
    /// 解释器与 WASM 路径不支持。
    Yield(Option<Box<HirExpr>>),
    InterpolatedString { parts: Vec<InterpPart> },
    Tuple(Vec<HirExpr>),
    FieldAssign {
        target: Box<HirExpr>,
        field: String,
        value: Box<HirExpr>,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub struct HirMatchArm {
    pub pattern: HirPattern,
    pub guard: Option<HirExpr>,
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
    /// Tuple destructuring: (a, b, c)
    Tuple(Vec<HirPattern>),
    /// Range pattern: start..end or start..=end
    Range { start: i64, end: i64, inclusive: bool },
    /// Variable binding: `x` (catch-all that binds the value)
    Binding(String),
    /// Struct destructuring: `Point { x, y }` or `Point { x: a, y: b }`.
    /// Each entry is (field_name, bind_name).
    Struct { name: String, fields: Vec<(String, String)> },
}

#[derive(Debug, Clone, PartialEq)]
pub struct HirExpr {
    pub kind: HirExprKind,
    pub ty: Type,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Literal {
    /// 整数字面量。第二字段为 dtype（I8/I16/I32/I64/U8/U16/U32/U64），保留到运行时。
    Int(i64, BaseType),
    /// 浮点字面量。第二字段为 dtype（F32 或 F64），保留到字节码与运行时。
    Float(f64, BaseType),
    Bool(bool),
    String(String),
    Char(char),
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
        names: Vec<String>,
        type_ann: Option<Type>,
        mutable: bool,
        init: Option<HirExpr>,
    },
    Expr(HirExpr),
    Return(Option<HirExpr>),
    While {
        /// M2.3：循环标签（`'outer: while ...`）
        label: Option<String>,
        cond: HirExpr,
        body: Box<HirStmt>,
    },
    DoWhile {
        label: Option<String>,
        body: Box<HirStmt>,
        cond: HirExpr,
    },
    For {
        label: Option<String>,
        var: String,
        iter: HirExpr,
        body: Box<HirStmt>,
    },
    /// M2.3：break。`label` 为 `break 'outer` 的循环标签，`value` 为 `break val` 的返回值。
    Break {
        label: Option<String>,
        value: Option<Box<HirExpr>>,
    },
    /// M2.3：continue。`label` 为 `continue 'outer` 的循环标签。
    Continue {
        label: Option<String>,
    },
    Loop {
        label: Option<String>,
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
    /// Default values for parameters that have them. `param_defaults[i]` corresponds to `params[i]`.
    pub param_defaults: Vec<Option<HirExpr>>,
    /// Whether each parameter is variadic (`...args`). `param_variadic[i]` corresponds to `params[i]`.
    pub param_variadic: Vec<bool>,
    pub return_type: Type,
    pub body: HirExpr,
    pub span: Span,
    /// 是否为 #[test] 测试函数
    pub is_test: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct HirGenericStruct {
    pub name: String,
    pub generics: Vec<String>,
    pub fields: Vec<(String, Type)>,
}

/// M2.1：泛型枚举定义（`enum Option<T> { Some(T), None }`）。
/// 变体字段类型中的 `TypeParam("T")` 在实例化时替换为具体类型实参。
#[derive(Debug, Clone, PartialEq)]
pub struct HirGenericEnum {
    pub name: String,
    pub generics: Vec<String>,
    /// 变体列表： (variant_name, Vec<(field_name, Type)>) —— 与 HirProgram.enums 同形
    pub variants: Vec<(String, Vec<(String, Type)>)>,
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
    pub unions: HashMap<String, Vec<(String, Type)>>,
    pub enums: HashMap<String, Vec<(String, Vec<(String, Type)>)>>,
    /// M2.1：泛型枚举（`enum X<T> { .. }`），非泛型枚举仍在 `enums` 中。
    pub generic_enums: HashMap<String, HirGenericEnum>,
    pub trait_defs: HashMap<String, HirTraitDef>,
    pub trait_impls: HashMap<String, HashMap<String, HashMap<String, HirFnDef>>>,
    /// 编译期警告（内存/算力预估等，非致命）
    pub warnings: Vec<crate::error::TenthWarning>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct HirTraitDef {
    pub name: String,
    pub generics: Vec<String>,
    pub methods: Vec<HirTraitMethod>,
    pub associated_types: Vec<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct HirTraitMethod {
    pub name: String,
    pub params: Vec<(String, Type)>,
    pub return_type: Type,
    /// Default method body (if provided in trait definition)
    pub default_body: Option<HirExpr>,
}