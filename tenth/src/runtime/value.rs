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

#[derive(Clone)]
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
        /// a1 P3：捕获值内联（与解释器 `Value::Closure.captures` 对齐）。
        /// 闭包创建时（Op::MakeClosure）从父栈弹出 captures_count 个捕获值装入；
        /// 调用时按闭包 chunk 的捕获槽位（`params..params+captures`）追加为额外实参。
        /// 默认空——非闭包/无捕获的 FnRef（顶层函数/native 别名）不受影响。
        captures: Vec<Value>,
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
    /// Weak<T>：弱引用（M3.4）。不增加引用计数，可 upgrade() 尝试取强引用。
    /// 存 `std::rc::Weak<RefCell<Value>>` 才能 upgrade（返回 Option<Rc<RefCell<Value>>>）。
    Weak(Weak<RefCell<Value>>),

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

/// M1-S2（true letrec）：手动 Debug——转发 Display（天然有界）。
///
/// 派生 Debug 在 true letrec 自引用 cell 上会**无限递归**：cell 是
/// `Value::Shared(Rc<RefCell<Value>>)`，闭包创建后 cell 内含闭包值
/// （`Value::FnRef`/`Value::Closure`），其 captures 又含该 cell → `{:?}` 递归
/// 直到栈溢出（可达路径：`期望可调用值，得到 {:?}` 打印含 cell 的闭包值）。
/// Display 对 `FnRef`/`Closure` 只打印 `<fn {name}>`/`<closure>`（不递归 captures），
/// 因此 Debug→Display 转发对一切可达环均有界。
impl fmt::Debug for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
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
            // Weak 已悬垂（原 Rc 被释放）→ 内部类型不可知，保守返回 Unknown。
            Value::Weak(w) => match w.upgrade() {
                Some(rc) => Type::Weak(Box::new(rc.borrow().type_of())),
                None => Type::Unknown,
            },
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
            Value::Weak(w) => w.upgrade().and_then(|rc| rc.borrow().as_float()),
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
            Value::Weak(w) => w.upgrade().and_then(|rc| rc.borrow().as_f32()),
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
            Value::Weak(w) => w.upgrade().and_then(|rc| rc.borrow().as_int()),
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
            Value::Weak(w) => w.upgrade().map_or(false, |rc| rc.borrow().is_truthy()),
            Value::Dyn { value, .. } => value.is_truthy(),
            _ => true,
        }
    }
}

/// 整数类型名称（用于溢出错误消息）。
pub fn int_dtype_name(dtype: BaseType) -> &'static str {
    match dtype {
        BaseType::I8 => "i8", BaseType::I16 => "i16", BaseType::I32 => "i32", BaseType::I64 => "i64",
        BaseType::U8 => "u8", BaseType::U16 => "u16", BaseType::U32 => "u32", BaseType::U64 => "u64",
        _ => "unknown",
    }
}

/// AUDIT-11.4.53：`Op::PushInt` 载荷 dtype 的 1 字节编码。
///
/// 只覆盖整型（PushInt 的 dtype 语义域）；非整型回退 I32（= 历史行为，
/// 保证编码/解码对称且不 panic）。与 `int_dtype_from_tag` 严格互逆。
pub fn int_dtype_tag(dtype: BaseType) -> u8 {
    match dtype {
        BaseType::I8 => 0, BaseType::I16 => 1, BaseType::I32 => 2, BaseType::I64 => 3,
        BaseType::U8 => 4, BaseType::U16 => 5, BaseType::U32 => 6, BaseType::U64 => 7,
        _ => 2,
    }
}

/// `int_dtype_tag` 的逆映射。未知字节回退 I32（解码宽容，不 panic）。
pub fn int_dtype_from_tag(tag: u8) -> BaseType {
    match tag {
        0 => BaseType::I8, 1 => BaseType::I16, 2 => BaseType::I32, 3 => BaseType::I64,
        4 => BaseType::U8, 5 => BaseType::U16, 6 => BaseType::U32, 7 => BaseType::U64,
        _ => BaseType::I32,
    }
}

/// 混合整数运算的公共 dtype（AUDIT-11.4.53 R4：**可交换**提升规则）。
///
/// 规则（用户 2026-09-17 裁定「宽度优先」）：
/// - rank：`i8/u8 < i16/u16 < i32/u32 < i64/u64`
/// - 取两操作数中 **rank 更大**者
/// - rank 相同且同型 → 该型
/// - rank 相同但异号 → 提升到**下一个更宽的有符号类型**
///   （`u8+i8→i16`、`u16+i16→i32`、`u32+i32→i64`、`u64+i64→i64`）
/// - 非整型参与时保持 `l`（浮点混算由调用方各自的浮点分支处理）
///
/// 交换性由构造保证：`rank(l)` 与 `rank(r)` 的比较、以及「同 rank 异号」
/// 分支都只依赖两操作数的**对称**属性（rank / 符号），与左右次序无关。
pub fn promote_int_dtype(l: BaseType, r: BaseType) -> BaseType {
    use BaseType::*;
    fn rank(t: BaseType) -> Option<u8> {
        match t {
            I8 | U8 => Some(0), I16 | U16 => Some(1), I32 | U32 => Some(2), I64 | U64 => Some(3),
            _ => None,
        }
    }
    let (rl, rr) = match (rank(l), rank(r)) {
        (Some(a), Some(b)) => (a, b),
        // 任一操作数非整型：保持既有语义（返回左操作数）
        _ => return l,
    };
    if rl != rr {
        return if rl > rr { l } else { r };
    }
    if l == r { return l; }
    // 同 rank 异号 → 下一个更宽的**有符号**类型
    match rl {
        0 => I16,
        1 => I32,
        2 => I64,
        // u64 + i64 → i64（已是最宽有符号；u64 超出 i64 范围的值由运行期范围检查兜底）
        _ => I64,
    }
}

/// 整数算术在 i64 层溢出（如 `i64::MAX + 1`、`i64::MIN / -1`）的错误。
/// 与 `check_int_overflow` 的窄 dtype 范围检查互补：checked_* 先拦截 i64 层溢出，
/// 再交给 `check_int_overflow` 做窄 dtype 范围检查。AUDIT-11.4.17。
pub fn int_overflow_err(dtype: BaseType) -> TenthError {
    use crate::error::TenthError;
    TenthError::RuntimeError {
        line: None, col: None,
        message: format!("整数运算结果溢出 {} 范围", int_dtype_name(dtype)),
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
        Err(TenthError::RuntimeError {
            line: None, col: None,
            message: format!("整数运算结果 {} 溢出 {} 范围", result, int_dtype_name(dtype)),
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

/// 判断 `v` 是否为需要自动解包的包裹值。
///
/// P2-B5：`deref_wrapped` 的配套守卫。解释器 `eval_binary` 与 VM 的
/// `add_priv`/`sub_priv`/`vm_eq` 等都在运算前用此判断是否需要先解壳。
/// 统一到共享层，避免解释器/VM 各自维护一份（VMs 含 SharedBox 而解释器仅
/// Shared/Ref/MutRef 曾是不一致来源）。
pub fn is_wrapped(v: &Value) -> bool {
    matches!(v, Value::Shared(_) | Value::Ref(_) | Value::MutRef(_) | Value::SharedBox(_))
}

/// 自动解包 Shared/Ref/MutRef/SharedBox 包裹值，返回内部值的 owned 副本。
///
/// **单一权威实现**（P2-B5 收敛，解释器/VM 双后端一致）。此前解释器
/// `interpreter/natives.rs` 与 VM `vm/execute.rs` 各有一份且语义分叉：
/// - 解释器对 Shared/Ref **递归一层**（防 `Shared<Shared<T>>`），`MutRef` 悬垂
///   返回 `Value::Unit`，且**无 SharedBox 分支**；
/// - VM **不递归**（单层），有 `SharedBox` 分支，`MutRef` 悬垂返回 `Value::Moved`。
///
/// 收敛后的统一语义：
/// - **递归剥壳**：解包后若仍是包裹值（如 `Shared<Shared<T>>` 双重包裹）继续解包，
///   直到非包裹类型——解释器的鲁棒行为，VM 一并获得（对常见单层无行为变化）。
/// - **`MutRef` 悬垂 → `Value::Moved`**：`&mut` 引用失效（原强引用已 drop）时返回
///   `Value::Moved`，对齐 VM 的"失效引用=移动"哨兵。解释器此前用 `Unit` 属演化
///   分叉——解释器**别处的显式解引用/赋值对悬垂 `&mut` 均报错或返回 `Moved`**，
///   仅 `deref_wrapped` 用 `Unit`，因此 `Unit` 不是有意的解释器语义，统一到 `Moved`
///   无行为回归（`Unit`/`Moved` 在数值/比较上下文中都落入同样的兜底 false/错误分支）。
/// - **`SharedBox` 解包**：`Rc<T>`/`Arc<T>` 与 VM 一致地自动解包。
///
/// 返回 owned `Value` 是因为 `RefCell::borrow()` 返回 `Ref<'_, Value>`，无法直接
/// 转为 `&Value`；调用方拿到 owned Value 后可按值 match（Copy 字段如 f64/i64/bool
/// 直接 by-value 绑定，无需额外解引用）。
pub fn deref_wrapped(v: &Value) -> Value {
    match v {
        Value::Shared(rc) => deref_wrapped(&rc.borrow().clone()),
        Value::Ref(rc) => deref_wrapped(&rc.borrow().clone()),
        Value::MutRef(weak) => match weak.upgrade() {
            Some(rc) => deref_wrapped(&rc.borrow().clone()),
            None => Value::Moved,
        },
        Value::SharedBox(rc) => deref_wrapped(&rc.borrow().clone()),
        other => other.clone(),
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

/// 值 → 用户可读字符串的**单一权威**实现。
///
/// `fmt::Display for Value` 与解释器的 `value_to_string` 都转发到这里，
/// 确保 VM 与解释器两条执行路径对同一值在 print / println / to_string /
/// 字符串插值下的**输出一致**（P2-B6：消除值字符串化双实现）。
///
/// 显示语义（用户可观察，刻意保留类型信息，便于 round-trip 与调试）：
/// - Float32 输出 `1.5f32`（带 `f32` 后缀，区别于 f64）；
/// - Float 走 `format_f64`，保证整数值显示 `.0` 后缀（`2.0` 而非 `2`）；
/// - Char 单引号包裹 `'a'`；引用带 `&` / `&mut` 前缀；
/// - 包装类型带 `Box(..)` / `Rc(..)` / `Pin(..)` / `Weak<..>` 前缀；
/// - `dyn Trait<Type>(..)`；`Future<v>`（保留 Future 类型信息）。
pub fn value_to_display_string(v: &Value) -> String {
    match v {
        Value::Int(n, _) => format!("{}", n),
        Value::Float(n) => format!("{}", format_f64(*n)),
        Value::Float32(n) => format!("{}f32", format_f32(*n)),
        Value::Bool(b) => format!("{}", b),
        Value::Char(c) => format!("'{}'", c),
        Value::String(s) => s.clone(),
        Value::Tensor(t) => format!("{}", t.borrow()),
        Value::Unit => "()".to_string(),
        Value::Array(items) => {
            let items = items.borrow();
            let inner: Vec<String> = items.iter().map(|it| value_to_display_string(it)).collect();
            format!("[{}]", inner.join(", "))
        }
        Value::FnRef { name, .. } => format!("<fn {}>", name),
        Value::Closure { .. } => "<closure>".to_string(),
        Value::Union { name, active_field, value } => {
            format!("union {} {{ {}: {} }}", name, active_field, value_to_display_string(value))
        }
        Value::Struct { name, fields } => {
            let fields = fields.borrow();
            let inner: Vec<String> = fields.iter()
                .map(|(fname, fval)| format!("{}: {}", fname, value_to_display_string(fval)))
                .collect();
            format!("{} {{ {} }}", name, inner.join(", "))
        }
        Value::Ref(v) => format!("&{}", value_to_display_string(&v.borrow())),
        Value::MutRef(v) => {
            match v.upgrade() {
                Some(rc) => format!("&mut {}", value_to_display_string(&rc.borrow())),
                None => "&mut <dangling>".to_string(),
            }
        }
        Value::Shared(v) => value_to_display_string(&v.borrow()),
        Value::Moved => "<moved>".to_string(),
        Value::Vec(items) => {
            let items = items.borrow();
            let inner: Vec<String> = items.iter().map(|it| value_to_display_string(it)).collect();
            format!("[{}]", inner.join(", "))
        }
        Value::Range { start, end, inclusive } => {
            let op = if *inclusive { "..=" } else { ".." };
            format!("{}{}{}", start, op, end)
        }
        Value::Iterator(_) => "<iterator>".to_string(),
        Value::Tuple(items) => {
            let inner: Vec<String> = items.iter().map(|it| value_to_display_string(it)).collect();
            format!("({})", inner.join(", "))
        }
        Value::Future(state) => {
            match &*state.borrow() {
                FutureState::Ready(v) => format!("Future<{}>", value_to_display_string(v)),
                FutureState::Pending(waiters) => format!("Future<Pending({})>", waiters.len()),
            }
        }
        Value::Map(entries) => {
            let entries = entries.borrow();
            let inner: Vec<String> = entries.iter()
                .map(|(k, v)| format!("{}: {}", k, value_to_display_string(v)))
                .collect();
            format!("{{{}}}", inner.join(", "))
        }
        Value::Enum { enum_name, variant, fields } => {
            let fields = fields.borrow();
            if fields.is_empty() {
                format!("{}::{}", enum_name, variant)
            } else {
                let inner: Vec<String> = fields.iter()
                    .map(|(fname, fval)| format!("{}: {}", fname, value_to_display_string(fval)))
                    .collect();
                format!("{}::{}({})", enum_name, variant, inner.join(", "))
            }
        }
        Value::HeapBox(v) => format!("Box({})", value_to_display_string(v)),
        Value::SharedBox(v) => format!("Rc({})", value_to_display_string(&v.borrow())),
        Value::Pin(v) => format!("Pin({})", value_to_display_string(v)),
        Value::Weak(w) => {
            match w.upgrade() {
                Some(rc) => format!("Weak<{}>", value_to_display_string(&rc.borrow())),
                None => "Weak<dangling>".to_string(),
            }
        }
        Value::Dyn { trait_name, type_name, value } => {
            format!("dyn {}<{}>({})", trait_name, type_name, value_to_display_string(value))
        }
        Value::BigInt(s) => format!("{}bi", s),
        Value::Complex(re, im) => {
            if *im < 0.0 {
                format!("({}{}i)", re, im)
            } else {
                format!("({}+{}i)", re, im)
            }
        }
        Value::Decimal(s) => format!("{}dec", s),
    }
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // 单一权威：所有字符串化经由 value_to_display_string，
        // 与解释器 value_to_string 保持输出一致。
        write!(f, "{}", value_to_display_string(self))
    }
}