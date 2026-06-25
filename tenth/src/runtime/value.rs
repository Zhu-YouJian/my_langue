use std::rc::{Rc, Weak};
use std::cell::RefCell;
use std::collections::HashMap;
use std::fmt;
use super::tensor::Tensor;
use crate::hir::types::{Type, BaseType, Dim};

/// A lazy iterator that yields values on demand.
/// Stores the source data and a chain of transformations (map/filter).
#[derive(Debug, Clone)]
pub struct LazyIterator {
    /// Source items (cloned from Vec/Range)
    pub source: Rc<RefCell<Vec<Value>>>,
    /// Current position in the source
    pub cursor: Rc<RefCell<usize>>,
    /// Chain of transformations to apply lazily
    pub transforms: Rc<RefCell<Vec<IteratorTransform>>>,
}

/// A single transformation in an iterator chain.
#[derive(Debug, Clone)]
pub enum IteratorTransform {
    /// Map each element through a closure
    Map { closure: Value },
    /// Filter elements through a predicate closure
    Filter { closure: Value },
    /// Take only the first N elements
    Take { n: usize },
    /// Skip the first N elements
    Skip { n: usize },
}

impl LazyIterator {
    pub fn from_vec(vec: &Rc<RefCell<Vec<Value>>>) -> Self {
        let items = vec.borrow().clone();
        LazyIterator {
            source: Rc::new(RefCell::new(items)),
            cursor: Rc::new(RefCell::new(0)),
            transforms: Rc::new(RefCell::new(Vec::new())),
        }
    }

    pub fn from_range(start: i64, end: i64, inclusive: bool) -> Self {
        let items: Vec<Value> = if inclusive {
            (start..=end).map(Value::Int).collect()
        } else {
            (start..end).map(Value::Int).collect()
        };
        LazyIterator {
            source: Rc::new(RefCell::new(items)),
            cursor: Rc::new(RefCell::new(0)),
            transforms: Rc::new(RefCell::new(Vec::new())),
        }
    }

    pub fn with_transform(&self, transform: IteratorTransform) -> Self {
        let mut new_transforms = self.transforms.borrow().clone();
        new_transforms.push(transform);
        LazyIterator {
            source: self.source.clone(),
            cursor: self.cursor.clone(),
            transforms: Rc::new(RefCell::new(new_transforms)),
        }
    }
}

#[derive(Debug, Clone)]
pub enum Value {
    Int(i64),
    Float(f64),
    /// f32 标量值。与 Float(f64) 区分以保留 dtype 信息到运行时。
    Float32(f32),
    Bool(bool),
    String(String),
    Tensor(Rc<RefCell<Tensor>>),
    Unit,
    Array(Rc<RefCell<Vec<Value>>>),
    FnRef {
        name: String,
        params: Vec<(String, Type)>,
        return_type: Type,
    },
    Closure {
        params: Vec<(String, Type)>,
        body: Rc<crate::hir::hir::HirExpr>,
        captures: Vec<(String, Value)>,
    },
    Struct {
        name: String,
        fields: Rc<RefCell<Vec<(String, Value)>>>,
    },
    Enum {
        enum_name: String,
        variant: String,
        fields: Rc<RefCell<Vec<(String, Value)>>>,
    },
    Ref(Rc<RefCell<Value>>),
    MutRef(Weak<RefCell<Value>>),
    Shared(Rc<RefCell<Value>>),
    Moved,
    Vec(Rc<RefCell<Vec<Value>>>),
    Map(Rc<RefCell<HashMap<String, Value>>>),
    Range { start: i64, end: i64, inclusive: bool },
    Iterator(LazyIterator),
    Tuple(Vec<Value>),
}

impl Value {
    pub fn type_of(&self) -> Type {
        match self {
            Value::Int(_) => Type::Base(BaseType::I32),
            Value::Float(_) => Type::Base(BaseType::F64),
            Value::Float32(_) => Type::Base(BaseType::F32),
            Value::Bool(_) => Type::Base(BaseType::Bool),
            Value::String(_) => Type::Base(BaseType::Str),
            Value::Tensor(t) => {
                let t = t.borrow();
                let dims: Vec<Dim> = t.shape().iter().map(|&d| Dim::Known(d as i64)).collect();
                Type::Tensor { dtype: t.dtype(), dims }
            }
            Value::Unit => Type::unit(),
            Value::Array(_) => Type::Unknown,
            Value::FnRef { params, return_type, .. } => {
                Type::FnType {
                    params: params.iter().map(|(_, t)| t.clone()).collect(),
                    ret: Box::new(return_type.clone()),
                }
            }
            Value::Closure { params, .. } => {
                Type::FnType {
                    params: params.iter().map(|(_, t)| t.clone()).collect(),
                    ret: Box::new(Type::Unknown),
                }
            }
            Value::Struct { name, .. } => Type::Struct(name.clone()),
            Value::Enum { enum_name, .. } => Type::Enum(enum_name.clone()),
            Value::Ref(v) => Type::Ref(Box::new(v.borrow().type_of())),
            Value::MutRef(v) => {
                match v.upgrade() {
                    Some(rc) => Type::MutRef(Box::new(rc.borrow().type_of())),
                    None => Type::Unknown,
                }
            }
            Value::Shared(v) => v.borrow().type_of(),
            Value::Moved => Type::unit(),
            Value::Vec(_) => Type::Unknown,
            Value::Map(_) => Type::Unknown,
            Value::Range { .. } => Type::Unknown,
            Value::Iterator(_) => Type::Unknown,
            Value::Tuple(items) => Type::Tuple(items.iter().map(|v| v.type_of()).collect()),
        }
    }

    pub fn as_float(&self) -> Option<f64> {
        match self {
            Value::Float(f) => Some(*f),
            Value::Float32(f) => Some(*f as f64),
            Value::Int(i) => Some(*i as f64),
            _ => None,
        }
    }

    /// 以 f32 访问（用于 f32 路径）。
    /// Float32 直接返回；Float/Int 提升为 f32；其他返回 None。
    pub fn as_f32(&self) -> Option<f32> {
        match self {
            Value::Float32(f) => Some(*f),
            Value::Float(f) => Some(*f as f32),
            Value::Int(i) => Some(*i as f32),
            _ => None,
        }
    }

    pub fn as_int(&self) -> Option<i64> {
        match self {
            Value::Int(i) => Some(*i),
            Value::Float(f) => Some(*f as i64),
            Value::Float32(f) => Some(*f as i64),
            _ => None,
        }
    }

    pub fn is_truthy(&self) -> bool {
        match self {
            Value::Bool(b) => *b,
            Value::Int(n) => *n != 0,
            Value::Float(f) => *f != 0.0,
            Value::Float32(f) => *f != 0.0,
            Value::Ref(v) => v.borrow().is_truthy(),
            Value::MutRef(v) => v.upgrade().map_or(false, |rc| rc.borrow().is_truthy()),
            Value::Shared(v) => v.borrow().is_truthy(),
            Value::Moved => false,
            Value::Vec(v) => !v.borrow().is_empty(),
            Value::Map(m) => !m.borrow().is_empty(),
            Value::String(s) => !s.is_empty(),
            _ => true,
        }
    }
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Value::Int(n) => write!(f, "{}", n),
            Value::Float(n) => write!(f, "{}", n),
            Value::Float32(n) => write!(f, "{}f32", n),
            Value::Bool(b) => write!(f, "{}", b),
            Value::String(s) => write!(f, "{}", s),
            Value::Tensor(t) => write!(f, "{}", t.borrow()),
            Value::Unit => write!(f, "()"),
            Value::Array(items) => {
                let items = items.borrow();
                write!(f, "[")?;
                for (i, item) in items.iter().enumerate() {
                    if i > 0 { write!(f, ", ")?; }
                    write!(f, "{}", item)?;
                }
                write!(f, "]")
            }
            Value::FnRef { name, .. } => write!(f, "<fn {}>", name),
            Value::Closure { .. } => write!(f, "<closure>"),
            Value::Struct { name, fields } => {
                let fields = fields.borrow();
                write!(f, "{} {{ ", name)?;
                for (i, (fname, fval)) in fields.iter().enumerate() {
                    if i > 0 { write!(f, ", ")?; }
                    write!(f, "{}: {}", fname, fval)?;
                }
                write!(f, " }}")
            }
            Value::Ref(v) => write!(f, "&{}", v.borrow()),
            Value::MutRef(v) => {
                match v.upgrade() {
                    Some(rc) => write!(f, "&mut {}", rc.borrow()),
                    None => write!(f, "&mut <dangling>"),
                }
            }
            Value::Shared(v) => write!(f, "{}", v.borrow()),
            Value::Moved => write!(f, "<moved>"),
            Value::Vec(items) => {
                let items = items.borrow();
                write!(f, "[")?;
                for (i, item) in items.iter().enumerate() {
                    if i > 0 { write!(f, ", ")?; }
                    write!(f, "{}", item)?;
                }
                write!(f, "]")
            }
            Value::Range { start, end, inclusive } => {
                let op = if *inclusive { "..=" } else { ".." };
                write!(f, "{}{}{}", start, op, end)
            }
            Value::Iterator(_) => write!(f, "<iterator>"),
            Value::Tuple(items) => {
                write!(f, "(")?;
                for (i, item) in items.iter().enumerate() {
                    if i > 0 { write!(f, ", ")?; }
                    write!(f, "{}", item)?;
                }
                write!(f, ")")
            }
            Value::Map(entries) => {
                let entries = entries.borrow();
                write!(f, "{{")?;
                for (i, (k, v)) in entries.iter().enumerate() {
                    if i > 0 { write!(f, ", ")?; }
                    write!(f, "{}: {}", k, v)?;
                }
                write!(f, "}}")
            }
            Value::Enum { enum_name, variant, fields } => {
                let fields = fields.borrow();
                if fields.is_empty() {
                    write!(f, "{}::{}", enum_name, variant)
                } else {
                    write!(f, "{}::{}(", enum_name, variant)?;
                    for (i, (fname, fval)) in fields.iter().enumerate() {
                        if i > 0 { write!(f, ", ")?; }
                        write!(f, "{}: {}", fname, fval)?;
                    }
                    write!(f, ")")
                }
            }
        }
    }
}