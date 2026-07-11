use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BaseType {
    I8, I16, I32, I64,
    U8, U16, U32, U64,
    F16, F32, F64, BF16,
    Bool, Char, Str,
    Unit,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Dim {
    Known(i64),
    Symbol(String),
    Any,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Type {
    Base(BaseType),
    Tensor {
        dtype: Box<Type>,
        dims: Vec<Dim>,
    },
    Array(Box<Type>),
    FnType {
        params: Vec<Type>,
        ret: Box<Type>,
    },
    TypeParam { name: String },
    Generic {
        base: Box<Type>,
        args: Vec<Type>,
    },
    Ref(Box<Type>),
    MutRef(Box<Type>),
    Struct(String),
    Enum(String),
    Tuple(Vec<Type>),
    Future(Box<Type>),
    /// Never 类型（发散类型 `!`）：标记永不返回的表达式/函数（如 `exit()`、无限循环）。
    /// 语义：Never 可以统一到任何类型 T（unify 结果为 T）。
    /// 若函数体所有分支都是 Never，则函数返回类型为 Never。
    Never,
    Unknown,
}

impl fmt::Display for Type {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Type::Base(b) => write!(f, "{:?}", b),
            Type::Tensor { dtype, dims } => {
                write!(f, "Tensor[{}", dtype)?;
                for dim in dims {
                    match dim {
                        Dim::Known(n) => write!(f, ", {}", n)?,
                        Dim::Symbol(s) => write!(f, ", {}", s)?,
                        Dim::Any => write!(f, ", ..")?,
                    }
                }
                write!(f, "]")
            }
            Type::Array(t) => write!(f, "[{}]", t),
            Type::FnType { params, ret } => {
                write!(f, "fn(")?;
                for (i, p) in params.iter().enumerate() {
                    if i > 0 { write!(f, ", ")?; }
                    write!(f, "{}", p)?;
                }
                write!(f, ") -> {}", ret)
            }
            Type::Future(t) => {
                write!(f, "Future<")?;
                t.fmt(f)?;
                write!(f, ">")
            },
            Type::Unknown => write!(f, "<unknown>"),
            Type::Never => write!(f, "!"),
            Type::TypeParam { name } => write!(f, "{}", name),
            Type::Generic { base, args } => {
                write!(f, "{}<", base)?;
                for (i, a) in args.iter().enumerate() {
                    if i > 0 { write!(f, ", ")?; }
                    write!(f, "{}", a)?;
                }
                write!(f, ">")
            }
            Type::Ref(inner) => write!(f, "&{}", inner),
            Type::MutRef(inner) => write!(f, "&mut {}", inner),
            Type::Struct(name) => write!(f, "{}", name),
            Type::Enum(name) => write!(f, "{}", name),
            Type::Tuple(types) => {
                write!(f, "(")?;
                for (i, t) in types.iter().enumerate() {
                    if i > 0 { write!(f, ", ")?; }
                    write!(f, "{}", t)?;
                }
                write!(f, ")")
            }
        }
    }
}

impl Type {
    pub fn i32() -> Self { Type::Base(BaseType::I32) }
    pub fn f64() -> Self { Type::Base(BaseType::F64) }
    pub fn f32() -> Self { Type::Base(BaseType::F32) }
    pub fn bool_() -> Self { Type::Base(BaseType::Bool) }
    pub fn str_() -> Self { Type::Base(BaseType::Str) }
    pub fn unit() -> Self { Type::Base(BaseType::Unit) }

    pub fn tensor(dtype: BaseType, dims: Vec<Dim>) -> Self {
        Type::Tensor { dtype: Box::new(Type::Base(dtype)), dims }
    }

    /// 对于 Tensor 类型，返回其 dtype 作为 BaseType。
    /// 若 dtype 是 TypeParam（泛型未实例化场景），返回 None。
    /// 运行时值已实例化，dtype 必为 Base，可安全 unwrap。
    pub fn tensor_dtype(&self) -> Option<BaseType> {
        match self {
            Type::Tensor { dtype, .. } => match dtype.as_ref() {
                Type::Base(b) => Some(*b),
                _ => None,
            },
            _ => None,
        }
    }

    /// 编译期静态元素总数（所有维均为 Known 时返回 Some(product)）。
    /// 任一维为 Symbol/Any、或非 Tensor 类型、或乘法溢出时返回 None。
    /// 用于内存/算力预估。
    pub fn static_numel(&self) -> Option<u64> {
        if let Type::Tensor { dims, .. } = self {
            let mut product: u64 = 1;
            for d in dims {
                match d {
                    Dim::Known(n) => {
                        if *n < 0 { return None; }
                        product = product.checked_mul(*n as u64)?;
                    }
                    _ => return None,
                }
            }
            Some(product)
        } else {
            None
        }
    }

    /// 编译期静态字节大小（dtype 字节数 × numel）。
    /// 用于内存预估（如 zeros(1024,1024,1024) → 8GB）。
    pub fn static_bytes(&self) -> Option<u64> {
        let numel = self.static_numel()?;
        let dtype_size = match self.tensor_dtype()? {
            BaseType::F64 | BaseType::I64 | BaseType::U64 => 8u64,
            BaseType::F32 | BaseType::I32 | BaseType::U32 => 4u64,
            BaseType::F16 | BaseType::I16 | BaseType::U16 | BaseType::BF16 => 2u64,
            BaseType::I8 | BaseType::U8 | BaseType::Bool | BaseType::Char => 1u64,
            _ => 8u64, // 默认按 f64 估算
        };
        numel.checked_mul(dtype_size)
    }

    pub fn from_annotation(ann: &super::super::parser::ast::TypeAnnotation) -> Self {
        use super::super::parser::ast::TypeAnnotation as TA;
        match ann {
            TA::Named(ident) => {
                let name = ident.name.as_str();
                // Handle reference types: &T, &mut T
                if let Some(inner) = name.strip_prefix("&mut ") {
                    return Type::MutRef(Box::new(Type::TypeParam { name: inner.to_string() }));
                }
                if let Some(inner) = name.strip_prefix("&") {
                    return Type::Ref(Box::new(Type::TypeParam { name: inner.to_string() }));
                }
                // 元组类型注解：parser 把 `(A, B, C)` 折叠成 Named("(A, B, C)")。
                // 这里反向解析为 Type::Tuple，使函数返回类型 `-> (i64, i64)` 等正确推断。
                if name.starts_with('(') && name.ends_with(')') && name.len() >= 2 {
                    let inner = &name[1..name.len() - 1];
                    // 至少含一个逗号才视为元组；单个类型名括起来（如 `(i64)`）应解包为该类型
                    if inner.contains(',') {
                        let parts: Vec<Type> = inner.split(',')
                            .map(|s| {
                                let s = s.trim();
                                Self::from_annotation(&TA::Named(super::super::parser::ast::Ident {
                                    name: s.to_string(),
                                    span: ident.span.clone(),
                                }))
                            })
                            .collect();
                        return Type::Tuple(parts);
                    }
                }
                match name {
                    "i8" => Type::Base(BaseType::I8),
                    "i16" => Type::Base(BaseType::I16),
                    "i32" => Type::Base(BaseType::I32),
                    "i64" => Type::Base(BaseType::I64),
                    "u8" => Type::Base(BaseType::U8),
                    "u16" => Type::Base(BaseType::U16),
                    "u32" => Type::Base(BaseType::U32),
                    "u64" => Type::Base(BaseType::U64),
                    "f16" => Type::Base(BaseType::F16),
                    "f32" => Type::Base(BaseType::F32),
                    "f64" => Type::Base(BaseType::F64),
                    "bf16" => Type::Base(BaseType::BF16),
                    "bool" => Type::Base(BaseType::Bool),
                    "char" => Type::Base(BaseType::Char),
                    "str" => Type::Base(BaseType::Str),
                    "!" => Type::Never,
                    _ => Type::TypeParam { name: ident.name.clone() },
                }
            }
            TA::Tensor { dtype, dims } => {
                let dt = Self::from_annotation(dtype);
                let resolved_dims: Vec<Dim> = dims.iter().map(|d| match d {
                    super::super::parser::ast::DimSpec::Literal(n) => Dim::Known(*n),
                    super::super::parser::ast::DimSpec::Symbol(s) => Dim::Symbol(s.clone()),
                    super::super::parser::ast::DimSpec::Wildcard => Dim::Any,
                }).collect();
                Type::Tensor { dtype: Box::new(dt), dims: resolved_dims }
            }
            TA::Generic { base, args } => {
                let base_ty = Self::from_annotation(&TA::Named(base.clone()));
                let arg_tys: Vec<Type> = args.iter().map(Self::from_annotation).collect();
                Type::Generic {
                    base: Box::new(base_ty),
                    args: arg_tys,
                }
            }
            TA::Array(inner) => Type::Array(Box::new(Self::from_annotation(inner))),
            TA::FnType { params, ret } => Type::FnType {
                params: params.iter().map(Self::from_annotation).collect(),
                ret: Box::new(Self::from_annotation(ret)),
            },
            TA::Unit => Type::unit(),
        }
    }
}