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
            Type::Unknown => write!(f, "<unknown>"),
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