use std::rc::{Rc, Weak};
use std::cell::RefCell;
use std::collections::HashMap;
use std::fmt;
use super::tensor::Tensor;
use crate::hir::types::{Type, BaseType, Dim};
use crate::error::{TenthResult, TenthError};

/// 格式化 f64，确保整数值显示 `.0` 后缀（如 `2.0` 而非 `2`）。
/// NaN/Inf 保持原样；已有小数点或科学记数法的值不变。
fn format_f64(n: f64) -> String {
    let s = format!("{}", n);
    if n.is_finite() && !s.contains('.') && !s.contains('e') {
        format!("{}.0", s)
    } else {
        s
    }
}

/// 格式化 f32，确保整数值显示 `.0` 后缀。
fn format_f32(n: f32) -> String {
    let s = format!("{}", n);
    if n.is_finite() && !s.contains('.') && !s.contains('e') {
        format!("{}.0", s)
    } else {
        s
    }
}

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
            (start..=end).map(|n| Value::Int(n, BaseType::I32)).collect()
        } else {
            (start..end).map(|n| Value::Int(n, BaseType::I32)).collect()
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

/// Future 的运行时状态。
/// - `Pending`：未完成，记录等待该 Future 完成的 task_id 列表（Phase 2 调度器使用）
/// - `Ready`：已完成，包含最终值
#[derive(Debug, Clone)]
pub enum FutureState {
    /// 等待者 task_id 列表。Phase 1 不使用（spawn 立即完成），Phase 2 调度器使用。
    Pending(Vec<u64>),
    /// 已完成，包含最终值。
    Ready(Value),
}

#[derive(Debug, Clone)]
pub enum Value {
    /// 整数值。第二字段为 dtype（I8/I16/I32/I64/U8/U16/U32/U64），保留到运行时。
    Int(i64, BaseType),
    Float(f64),
    /// f32 标量值。与 Float(f64) 区分以保留 dtype 信息到运行时。
    Float32(f32),
    Bool(bool),
    Char(char),
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
    Union {
        name: String,
        /// 当前活跃字段名和值
        active_field: String,
        value: Box<Value>,
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
    /// Future 值。共享语义：多个引用者看到同一 Future。
    /// Phase 1：spawn 立即包装为 `Ready`，await 立即解包 `Ready`（同步语义）。
    /// Phase 2：将支持 `Pending` 状态与协程调度。
    Future(Rc<RefCell<FutureState>>),

    // ── 问题29：智能指针 ──
    /// Box<T>：堆分配的所有权指针。
    HeapBox(Box<Value>),
    /// Rc<T> / Arc<T>：引用计数共享指针（Arc 暂用 Rc 等价实现）。
    SharedBox(Rc<RefCell<Value>>),
    /// Pin<T>：固定不可移动包装（问题31）。
    Pin(Box<Value>),

    // ── M1.3：dyn Trait 动态分发 ──
    /// dyn Trait 动态分发对象：
    /// - trait_name：dyn 指向的 trait 名（如 "Draw"）
    /// - type_name：具体类型名（如 "Circle"）
    /// - value：具体值（Box 包装，避免递归大小）
    Dyn {
        trait_name: String,
        type_name: String,
        value: Box<Value>,
    },

    // ── 问题35：BigInt ──
    BigInt(String),

    // ── 问题36：Complex ──
    /// Complex(f64, f64)：复数的笛卡尔坐标 (re, im)。
    Complex(f64, f64),

    // ── 问题37：Decimal ──
    Decimal(String),
}

impl Value {
    /// 构造一个已完成的 Future（Phase 1：spawn 立即包装为 Ready）。
    pub fn future_ready(v: Value) -> Value {
        Value::Future(Rc::new(RefCell::new(FutureState::Ready(v))))
    }

    /// 构造一个未完成的 Future（Phase 2：调度器创建协程任务时使用）。
    pub fn future_pending() -> Value {
        Value::Future(Rc::new(RefCell::new(FutureState::Pending(vec![]))))
    }

    pub fn type_of(&self) -> Type {
        match self {
            Value::Int(_, dt) => Type::Base(*dt),
            Value::Float(_) => Type::Base(BaseType::F64),
            Value::Float32(_) => Type::Base(BaseType::F32),
            Value::Bool(_) => Type::Base(BaseType::Bool),
            Value::Char(_) => Type::Base(BaseType::Char),
            Value::String(_) => Type::Base(BaseType::Str),
            Value::Tensor(t) => {
                let t = t.borrow();
                let dims: Vec<Dim> = t.shape().iter().map(|&d| Dim::Known(d as i64)).collect();
                Type::tensor(t.dtype(), dims)
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
            Value::Union { name, .. } => Type::Union(name.clone()),
            Value::Enum { enum_name, .. } => Type::Enum(enum_name.clone()),
            Value::Ref(v) => Type::Ref(Box::new(v.borrow().type_of()), None),
            Value::MutRef(v) => {
                match v.upgrade() {
                    Some(rc) => Type::MutRef(Box::new(rc.borrow().type_of()), None),
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
            Value::Future(state) => {
                match &*state.borrow() {
                    FutureState::Ready(v) => v.type_of(),
                    FutureState::Pending(_) => Type::Unknown,
                }
            }
            Value::HeapBox(v) => Type::HeapBox(Box::new(v.type_of())),
            Value::SharedBox(v) => Type::SharedBox(Box::new(v.borrow().type_of())),
            Value::Pin(v) => Type::Pin(Box::new(v.type_of())),
            Value::Dyn { trait_name, .. } => Type::Dyn(trait_name.clone()),
            Value::BigInt(_) => Type::Base(BaseType::BigInt),
            Value::Complex(_, _) => Type::Base(BaseType::C128),
            Value::Decimal(_) => Type::Base(BaseType::Decimal),
        }
    }

    pub fn as_float(&self) -> Option<f64> {
        match self {
            Value::Float(f) => Some(*f),
            Value::Float32(f) => Some(*f as f64),
            Value::Int(i, _) => Some(*i as f64),
            Value::Complex(re, _) => Some(*re),
            Value::BigInt(s) => s.parse::<f64>().ok(),
            Value::Decimal(s) => s.parse::<f64>().ok(),
            Value::HeapBox(v) => v.as_float(),
            Value::SharedBox(v) => v.borrow().as_float(),
            Value::Pin(v) => v.as_float(),
            Value::Dyn { value, .. } => value.as_float(),
            _ => None,
        }
    }

    /// 以 f32 访问（用于 f32 路径）。
    /// Float32 直接返回；Float/Int 提升为 f32；其他返回 None。
    pub fn as_f32(&self) -> Option<f32> {
        match self {
            Value::Float32(f) => Some(*f),
            Value::Float(f) => Some(*f as f32),
            Value::Int(i, _) => Some(*i as f32),
            Value::Complex(re, _) => Some(*re as f32),
            Value::BigInt(s) => s.parse::<f32>().ok(),
            Value::Decimal(s) => s.parse::<f32>().ok(),
            Value::HeapBox(v) => v.as_f32(),
            Value::SharedBox(v) => v.borrow().as_f32(),
            Value::Pin(v) => v.as_f32(),
            Value::Dyn { value, .. } => value.as_f32(),
            _ => None,
        }
    }

    pub fn as_int(&self) -> Option<i64> {
        match self {
            Value::Int(i, _) => Some(*i),
            Value::Float(f) => Some(*f as i64),
            Value::Float32(f) => Some(*f as i64),
            Value::Complex(re, _) => Some(*re as i64),
            Value::BigInt(s) => s.parse::<i64>().ok(),
            Value::Decimal(s) => s.parse::<i64>().ok(),
            Value::HeapBox(v) => v.as_int(),
            Value::SharedBox(v) => v.borrow().as_int(),
            Value::Pin(v) => v.as_int(),
            Value::Dyn { value, .. } => value.as_int(),
            _ => None,
        }
    }

    pub fn is_truthy(&self) -> bool {
        match self {
            Value::Bool(b) => *b,
            Value::Int(n, _) => *n != 0,
            Value::Float(f) => *f != 0.0,
            Value::Float32(f) => *f != 0.0,
            Value::Ref(v) => v.borrow().is_truthy(),
            Value::MutRef(v) => v.upgrade().map_or(false, |rc| rc.borrow().is_truthy()),
            Value::Shared(v) => v.borrow().is_truthy(),
            Value::Moved => false,
            Value::Vec(v) => !v.borrow().is_empty(),
            Value::Map(m) => !m.borrow().is_empty(),
            Value::String(s) => !s.is_empty(),
            Value::BigInt(s) => s != "0",
            Value::Complex(re, im) => *re != 0.0 || *im != 0.0,
            Value::Decimal(s) => s != "0",
            Value::HeapBox(v) => v.is_truthy(),
            Value::SharedBox(v) => v.borrow().is_truthy(),
            Value::Pin(v) => v.is_truthy(),
            Value::Dyn { value, .. } => value.is_truthy(),
            _ => true,
        }
    }
}

/// 检查整数运算结果是否在 dtype 范围内。溢出时返回 Err。
pub fn check_int_overflow(result: i64, dtype: BaseType) -> TenthResult<()> {
    use crate::error::TenthError;
    let ok = match dtype {
        BaseType::I8 => result >= -128 && result <= 127,
        BaseType::I16 => result >= -32768 && result <= 32767,
        BaseType::I32 => result >= -2147483648 && result <= 2147483647,
        BaseType::I64 => true,
        BaseType::U8 => result >= 0 && result <= 255,
        BaseType::U16 => result >= 0 && result <= 65535,
        BaseType::U32 => result >= 0 && result <= 4294967295,
        BaseType::U64 => result >= 0,
        _ => true,
    };
    if !ok {
        let name = match dtype {
            BaseType::I8 => "i8", BaseType::I16 => "i16", BaseType::I32 => "i32", BaseType::I64 => "i64",
            BaseType::U8 => "u8", BaseType::U16 => "u16", BaseType::U32 => "u32", BaseType::U64 => "u64",
            _ => "unknown",
        };
        Err(TenthError::RuntimeError {
            line: None, col: None,
            message: format!("整数运算结果 {} 溢出 {} 范围", result, name),
        })
    } else {
        Ok(())
    }
}

/// 将 Value::Array 递归转换为 Value::Tensor。
/// 用于 `tensor<f64>([1.0, 2.0, 3.0])` 等构造函数——当 HIR 把
/// `tensor<>()` 编译成 `Call("tensor", [ArrayLiteral])` 时，
/// native 函数需要将 Value::Array 转为 Value::Tensor 才能参与张量运算。
/// 支持嵌套数组（如 `[[1.0, 2.0], [3.0, 4.0]]` → 2D tensor）。
pub fn array_to_tensor(val: &Value) -> TenthResult<Value> {
    match val {
        Value::Tensor(_) => Ok(val.clone()),
        Value::Array(arr) => {
            let borrowed = arr.borrow();
            let (shape, data) = flatten_values(&borrowed)?;
            if data.is_empty() {
                return Err(TenthError::RuntimeError {
                    line: None, col: None,
                    message: "tensor() 构造函数收到空数组".into(),
                });
            }
            let tensor = Tensor::from_vec(data, shape);
            Ok(Value::Tensor(Rc::new(RefCell::new(tensor))))
        }
        Value::Vec(arr) => {
            let borrowed = arr.borrow();
            let (shape, data) = flatten_values(&borrowed)?;
            if data.is_empty() {
                return Err(TenthError::RuntimeError {
                    line: None, col: None,
                    message: "tensor() 构造函数收到空数组".into(),
                });
            }
            let tensor = Tensor::from_vec(data, shape);
            Ok(Value::Tensor(Rc::new(RefCell::new(tensor))))
        }
        _ => Ok(val.clone()),
    }
}

/// 递归展平 Value 切片，返回 (shape, flat_data)。
/// 支持 Value::Shared 包装（ArrayLiteral 元素被 Shared 包裹）。
fn flatten_values(arr: &[Value]) -> TenthResult<(Vec<usize>, Vec<f64>)> {
    if arr.is_empty() {
        return Ok((vec![0], vec![]));
    }
    // 解包 Shared 获取第一个元素的实际类型
    let first = unpack_shared(arr.first().unwrap());
    match first {
        Value::Array(_) | Value::Vec(_) => {
            // 嵌套数组：递归展平
            let mut shape = vec![arr.len()];
            let mut data = Vec::new();
            let mut sub_shape: Option<Vec<usize>> = None;
            for v in arr {
                let unwrapped = unpack_shared(v);
                let (ss, mut sd) = match &unwrapped {
                    Value::Array(sub_arr) => {
                        let borrowed = sub_arr.borrow();
                        flatten_values(&borrowed)?
                    }
                    Value::Vec(sub_arr) => {
                        let borrowed = sub_arr.borrow();
                        flatten_values(&borrowed)?
                    }
                    _ => return Err(TenthError::RuntimeError {
                        line: None, col: None,
                        message: "张量构造：嵌套数组中混合了非数组元素".into(),
                    }),
                };
                if let Some(ref expected) = sub_shape {
                    if ss != *expected {
                        return Err(TenthError::RuntimeError {
                            line: None, col: None,
                            message: format!("张量形状不一致：{:?} vs {:?}", ss, expected),
                        });
                    }
                } else {
                    sub_shape = Some(ss);
                }
                data.append(&mut sd);
            }
            if let Some(ss) = sub_shape {
                shape.extend(ss);
            }
            Ok((shape, data))
        }
        _ => {
            // 叶子层：提取数值
            let data: Vec<f64> = arr.iter()
                .map(|v| {
                    let unwrapped = unpack_shared(v);
                    match unwrapped {
                        Value::Float(f) => f,
                        Value::Int(i, _) => i as f64,
                        Value::Float32(f) => f as f64,
                        Value::Bool(b) => if b { 1.0 } else { 0.0 },
                        _ => 0.0,
                    }
                })
                .collect();
            Ok((vec![arr.len()], data))
        }
    }
}

/// 解包 Value::Shared / Value::SharedBox，返回内部值的 owned 副本。
/// 返回 owned Value 是因为 `RefCell::borrow()` 返回 `Ref<'_, Value>`，
/// 无法直接转为 `&Value`；调用方拿到 owned Value 后可按值 match（Copy 字段
/// 如 f64/i64/bool 直接 by-value 绑定，无需额外解引用）。
fn unpack_shared(v: &Value) -> Value {
    match v {
        Value::Shared(inner) => inner.borrow().clone(),
        Value::SharedBox(inner) => inner.borrow().clone(),
        _ => v.clone(),
    }
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Value::Int(n, _) => write!(f, "{}", n),
            Value::Float(n) => write!(f, "{}", format_f64(*n)),
            Value::Float32(n) => write!(f, "{}f32", format_f32(*n)),
            Value::Bool(b) => write!(f, "{}", b),
            Value::Char(c) => write!(f, "'{}'", c),
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
            Value::Union { name, active_field, value } => {
                write!(f, "union {} {{ {}: {} }}", name, active_field, value)
            }
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
            Value::Future(state) => {
                match &*state.borrow() {
                    FutureState::Ready(v) => write!(f, "Future<{}>", v),
                    FutureState::Pending(waiters) => {
                        write!(f, "Future<Pending({})>", waiters.len())
                    }
                }
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
            Value::HeapBox(v) => write!(f, "Box({})", v),
            Value::SharedBox(v) => write!(f, "Rc({})", v.borrow()),
            Value::Pin(v) => write!(f, "Pin({})", v),
            Value::Dyn { trait_name, type_name, value } => {
                write!(f, "dyn {}<{}>({})", trait_name, type_name, value)
            }
            Value::BigInt(s) => write!(f, "{}bi", s),
            Value::Complex(re, im) => {
                if *im < 0.0 {
                    write!(f, "({}{}i)", re, im)
                } else {
                    write!(f, "({}+{}i)", re, im)
                }
            }
            Value::Decimal(s) => write!(f, "{}dec", s),
        }
    }
}