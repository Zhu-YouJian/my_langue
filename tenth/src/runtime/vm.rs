//! Bytecode VM for Tenth — stack-based virtual machine.
//!
//! Architecture: HIR → compile → Chunk (bytecode) → Vm::run()

use std::collections::HashMap;
use std::rc::Rc;
use std::cell::RefCell;
use crate::error::{TenthError, TenthResult};
use super::value::Value;
use super::autodiff::{Tape, TapeOp};
use super::tensor::Tensor;

/// Native Rust function callable from VM bytecode.
pub type NativeFn = fn(&mut Vm, &[Value]) -> TenthResult<Value>;

// ── Opcode ─────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Op {
    PushInt(i64), PushFloat(f64), PushFloat32(f32), PushBool(bool), PushStr(usize), PushUnit,
    Pop, Dup,
    Load(usize), Store(usize),
    LoadGlobal(usize), StoreGlobal(usize),
    Add, Sub, Mul, Div, Mod, Neg, Not,
    Eq, Neq, Lt, Gt, Lte, Gte,
    Jump(i32), JmpFalse(i32), JmpTrue(i32),
    Call(usize), CallN(usize, usize), MethodCall(usize, usize), Ret,
    MakeVec(usize), MakeMap(usize),
    NewStruct(usize, usize), LoadField(usize), StoreField(usize),
    IndexGet,
    SliceStr,
    MakeEnum(usize, usize, usize),
    IsEnumVariant(usize),
    EnumGetField(usize),
    IsStruct(usize),
    PushRange(i64, i64, bool),  // start, end, inclusive
    MoveOp,                     // no-op marker for move semantics
    MakeTensor(usize, usize, u8), // rows, cols, dtype (0=F64, 1=F32) — pops rows*cols values
    MakeClosure(usize, usize),  // params_count, chunk_idx — creates a closure value
}

// ── Chunk ──────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct Chunk {
    pub code: Vec<u8>,
    pub strings: Vec<String>,
    pub num_locals: usize,
    pub num_args: usize,
}

impl Chunk {
    pub fn new() -> Self { Chunk { code: vec![], strings: vec![], num_locals: 0, num_args: 0 } }

    pub fn add_string(&mut self, s: &str) -> usize {
        if let Some(i) = self.strings.iter().position(|x| x == s) { return i; }
        let i = self.strings.len(); self.strings.push(s.to_string()); i
    }

    pub fn emit(&mut self, op: Op) {
        use Op::*;
        self.code.push(match &op {
            PushInt(_) => 0, PushFloat(_) => 1, PushBool(_) => 2, PushStr(_) => 3,
            PushUnit => 4, Pop => 5, Dup => 6,
            Load(_) => 7, Store(_) => 8, LoadGlobal(_) => 9, StoreGlobal(_) => 10,
            Add => 11, Sub => 12, Mul => 13, Div => 14, Mod => 15,
            Neg => 16, Not => 17,
            Eq => 18, Neq => 19, Lt => 20, Gt => 21, Lte => 22, Gte => 23,
            Jump(_) => 24, JmpFalse(_) => 25, JmpTrue(_) => 26,
            Call(_) => 27, CallN(..) => 28, MethodCall(..) => 29, Ret => 30,
            MakeVec(_) => 31, MakeMap(_) => 32,
            NewStruct(..) => 33, LoadField(_) => 34, StoreField(_) => 35,
            IndexGet => 36, SliceStr => 37,
            MakeEnum(..) => 38, IsEnumVariant(_) => 39, EnumGetField(_) => 40,
            PushRange(..) => 41, MoveOp => 42,
            MakeTensor(..) => 43, MakeClosure(..) => 44,
            PushFloat32(_) => 45,
            IsStruct(_) => 46,
        });

        // Emit operands
        macro_rules! w { ($n:expr, $t:ty) => { self.code.extend_from_slice(&($n as $t).to_le_bytes()) } }
        match &op {
            PushInt(n) => w!(*n, i64), PushFloat(f) => w!(*f, f64),
            PushFloat32(f) => w!(*f, f32),
            PushBool(b) => self.code.push(if *b {1} else {0}),
            PushStr(i) | LoadGlobal(i) | StoreGlobal(i) | Call(i) | LoadField(i) | StoreField(i) => w!(*i, u64),
            CallN(i, n) => { w!(*i, u64); w!(*n, u64); }
            MethodCall(i, n) => { w!(*i, u64); w!(*n, u64); }
            Load(i) | Store(i) => w!(*i, u64),
            Jump(o) | JmpFalse(o) | JmpTrue(o) => w!(*o, i32),
            MakeVec(n) | MakeMap(n) => w!(*n, u64),
            NewStruct(n, f) => { w!(*n, u64); w!(*f, u64); }
            MakeEnum(n, v, f) => { w!(*n, u64); w!(*v, u64); w!(*f, u64); }
            IsEnumVariant(v) => w!(*v, u64),
            EnumGetField(f) => w!(*f, u64),
            IsStruct(n) => w!(*n, u64),
            PushRange(s, e, inc) => { w!(*s, i64); w!(*e, i64); self.code.push(if *inc {1} else {0}); }
            MakeTensor(r, c, d) => { w!(*r, u64); w!(*c, u64); self.code.push(*d); }
            MakeClosure(p, c) => { w!(*p, u64); w!(*c, u64); }
            MoveOp => {}
            _ => {}
        }
    }

    pub fn read_op(&self, ip: &mut usize) -> Op {
        use Op::*;
        if *ip >= self.code.len() {
            return Ret; // graceful exit on overrun
        }
        let b = self.code[*ip]; *ip += 1;
        macro_rules! r { ($t:ty) => {{ let n = std::mem::size_of::<$t>(); if *ip + n > self.code.len() { return Ret; } let mut buf = [0u8; std::mem::size_of::<$t>()]; buf.copy_from_slice(&self.code[*ip..*ip+n]); *ip += n; <$t>::from_le_bytes(buf) }}; }
        match b {
            0 => PushInt(r!(i64)), 1 => PushFloat(r!(f64)),
            2 => { let v = self.code[*ip] != 0; *ip += 1; PushBool(v) },
            3 => PushStr(r!(u64) as usize),
            4 => PushUnit, 5 => Pop, 6 => Dup,
            7 => Load(r!(u64) as usize), 8 => Store(r!(u64) as usize),
            9 => LoadGlobal(r!(u64) as usize), 10 => StoreGlobal(r!(u64) as usize),
            11 => Add, 12 => Sub, 13 => Mul, 14 => Div, 15 => Mod,
            16 => Neg, 17 => Not,
            18 => Eq, 19 => Neq, 20 => Lt, 21 => Gt, 22 => Lte, 23 => Gte,
            24 => Jump(r!(i32)), 25 => JmpFalse(r!(i32)), 26 => JmpTrue(r!(i32)),
            27 => Call(r!(u64) as usize), 28 => CallN(r!(u64) as usize, r!(u64) as usize),
            29 => MethodCall(r!(u64) as usize, r!(u64) as usize), 30 => Ret,
            31 => MakeVec(r!(u64) as usize), 32 => MakeMap(r!(u64) as usize),
            33 => NewStruct(r!(u64) as usize, r!(u64) as usize),
            34 => LoadField(r!(u64) as usize),
            35 => StoreField(r!(u64) as usize),
            36 => IndexGet,
            37 => SliceStr,
            38 => MakeEnum(r!(u64) as usize, r!(u64) as usize, r!(u64) as usize),
            39 => IsEnumVariant(r!(u64) as usize),
            40 => EnumGetField(r!(u64) as usize),
            41 => PushRange(r!(i64), r!(i64), { let b = self.code[*ip]; *ip += 1; b != 0 }),
            42 => MoveOp,
            43 => MakeTensor(r!(u64) as usize, r!(u64) as usize, { let d = self.code[*ip]; *ip += 1; d }),
            44 => MakeClosure(r!(u64) as usize, r!(u64) as usize),
            45 => PushFloat32(r!(f32)),
            46 => IsStruct(r!(u64) as usize),
            _ => panic!("bad opcode {b}"),
        }
    }
}

// ── Vm ─────────────────────────────────────────────────────────────────────

struct Frame {
    ip: usize,
    chunk_idx: usize,
    locals: Vec<Value>,
    stack_base: usize,
}

pub struct Vm {
    pub functions: HashMap<String, usize>,
    chunks: Vec<Chunk>,
    chunk_names: Vec<String>,
    pub natives: HashMap<String, NativeFn>,
    globals: HashMap<String, Value>,
    stack: Vec<Value>,
    frames: Vec<Frame>,
    /// Autodiff computation tape (active when `recording` is true).
    pub tape: Option<Tape>,
    /// Whether tensor operations should be recorded on the tape.
    pub recording: bool,
    /// Execution step budget. When `Some(n)`, each dispatched opcode
    /// decrements the counter; reaching zero raises `TenthError::Timeout`.
    /// `None` means unlimited (default).
    pub step_budget: Option<u64>,
    /// Optional wall-clock deadline (Unix ms). Checked periodically.
    pub deadline_ms: Option<u128>,
    /// 文件系统沙箱。`Some` 时所有文件 I/O 原生函数必须经过校验。
    /// `None` 表示无沙箱（默认，向后兼容）。
    pub fs_sandbox: Option<crate::runtime::limits::FsSandbox>,
    /// Lazily-initialised Cranelift JIT context. `None` until first JIT use.
    pub jit_ctx: Option<crate::compile::jit::context::JitContext>,
    /// Last error message set by a JIT hostcall trampoline.
    last_error: Option<String>,
    /// Index of the chunk currently being executed by JIT (for string lookup).
    pub current_chunk_idx: usize,
    /// 护城河 F：上一次 backward 失败时的根因说明列表（由 formal_explain 生成）。
    /// 由 `explain_error()` native 读取并清空。
    pub last_explanation: Vec<String>,
}

impl Vm {
    pub fn new() -> Self {
        Vm { functions: HashMap::new(), chunks: Vec::new(), chunk_names: Vec::new(), natives: HashMap::new(), globals: HashMap::new(), stack: Vec::new(), frames: Vec::new(), tape: None, recording: false, step_budget: None, deadline_ms: None, fs_sandbox: None, jit_ctx: None, last_error: None, current_chunk_idx: 0, last_explanation: Vec::new() }
    }

    // ── JIT accessors ──────────────────────────────────────────────────────

    pub fn is_recording(&self) -> bool { self.recording }
    pub fn stack_len(&self) -> usize { self.stack.len() }
    pub fn stack_push(&mut self, v: Value) { self.stack.push(v); }
    pub fn stack_pop(&mut self) -> Value { self.stack.pop().unwrap_or(Value::Unit) }
    pub fn get_global(&self, name: &str) -> Option<Value> { self.globals.get(name).cloned() }
    pub fn set_last_error(&mut self, msg: String) { self.last_error = Some(msg); }
    pub fn take_last_error(&mut self) -> Option<String> { self.last_error.take() }

    pub fn chunk_index_of(&self, name: &str) -> Option<usize> {
        self.functions.get(name).copied()
    }
    pub fn chunk_at(&self, idx: usize) -> &Chunk { &self.chunks[idx] }
    pub fn string_at(&self, idx: usize) -> Option<String> {
        self.chunks.get(self.current_chunk_idx)
            .and_then(|c| c.strings.get(idx).cloned())
    }
    pub fn chunk_name_at(&self, idx: usize) -> Option<String> {
        self.chunk_names.get(idx).cloned()
    }

    /// Call a function by name with explicit args (used by JIT hostcalls).
    pub fn call_with_args(&mut self, name: &str, args: &[Value]) -> TenthResult<Value> {
        // Try native first
        if let Some(native_fn) = self.natives.get(name).copied() {
            return native_fn(self, args);
        }
        // Push args and call user function
        for a in args { self.stack.push(a.clone()); }
        self.call(name)
    }

    // ── Public arithmetic/method wrappers (for JIT hostcalls) ──────────────

    pub fn add(&mut self, a: &Value, b: &Value) -> TenthResult<Value> { self.add_priv(a, b) }
    pub fn sub(&mut self, a: &Value, b: &Value) -> TenthResult<Value> { self.sub_priv(a, b) }
    pub fn mul(&mut self, a: &Value, b: &Value) -> TenthResult<Value> { self.mul_priv(a, b) }
    pub fn div(&mut self, a: &Value, b: &Value) -> TenthResult<Value> { self.div_priv(a, b) }
    pub fn rem(&mut self, a: &Value, b: &Value) -> TenthResult<Value> {
        match (a, b) {
            (Value::Int(x), Value::Int(y)) => {
                if *y == 0 { return Err(TenthError::RuntimeError { message: "整数取模除零".into() }); }
                Ok(Value::Int(x % y))
            }
            _ => Err(TenthError::RuntimeError { message: "% 需要整数".into() }),
        }
    }
    pub fn neg(&mut self, a: &Value) -> TenthResult<Value> {
        match a {
            Value::Int(n) => Ok(Value::Int(-n)),
            Value::Float(n) => Ok(Value::Float(-n)),
            Value::Float32(n) => Ok(Value::Float32(-n)),
            Value::Tensor(t) => Ok(Value::Tensor(Rc::new(RefCell::new(t.borrow().neg())))),
            _ => Err(TenthError::RuntimeError { message: "无法取负".into() }),
        }
    }
    pub fn not(&mut self, a: &Value) -> TenthResult<Value> {
        Ok(Value::Bool(!a.is_truthy()))
    }
    pub fn eq(&mut self, a: &Value, b: &Value) -> TenthResult<Value> { Ok(Value::Bool(self.vm_eq(a, b))) }
    pub fn neq(&mut self, a: &Value, b: &Value) -> TenthResult<Value> { Ok(Value::Bool(!self.vm_eq(a, b))) }
    pub fn lt(&mut self, a: &Value, b: &Value) -> TenthResult<Value> {
        Ok(Value::Bool(self.compare(a, b, |x, y| x < y, |x, y| x < y)?))
    }
    pub fn gt(&mut self, a: &Value, b: &Value) -> TenthResult<Value> {
        Ok(Value::Bool(self.compare(a, b, |x, y| x > y, |x, y| x > y)?))
    }
    pub fn lte(&mut self, a: &Value, b: &Value) -> TenthResult<Value> {
        Ok(Value::Bool(self.compare(a, b, |x, y| x <= y, |x, y| x <= y)?))
    }
    pub fn gte(&mut self, a: &Value, b: &Value) -> TenthResult<Value> {
        Ok(Value::Bool(self.compare(a, b, |x, y| x >= y, |x, y| x >= y)?))
    }
    pub fn index_get(&mut self, target: &Value, idx: &Value) -> TenthResult<Value> {
        match target {
            Value::Vec(items) => {
                let i = idx.as_int().unwrap_or(0) as usize;
                Ok(items.borrow().get(i).cloned().unwrap_or(Value::Unit))
            }
            Value::String(s) => {
                let i = idx.as_int().unwrap_or(0) as usize;
                Ok(Value::String(s.chars().nth(i).map(|c| c.to_string()).unwrap_or_default()))
            }
            _ => Err(TenthError::RuntimeError { message: "无法索引".into() }),
        }
    }
    pub fn slice_str(&mut self, target: &Value, start: &Value, end: &Value) -> TenthResult<Value> {
        let start_idx = start.as_int().unwrap_or(0) as usize;
        let end_idx = end.as_int().unwrap_or(0) as usize;
        match target {
            Value::String(s) => {
                let chars: Vec<char> = s.chars().collect();
                let len = chars.len();
                let si = start_idx.min(len);
                let ei = end_idx.min(len);
                if si > ei {
                    return Err(TenthError::RuntimeError { message: "字符串切片起始位置大于结束位置".into() });
                }
                Ok(Value::String(chars[si..ei].iter().collect()))
            }
            _ => Err(TenthError::RuntimeError { message: "SliceStr 需要字符串目标".into() }),
        }
    }
    pub fn call_method(&mut self, receiver: &Value, method: &str, args: &[Value]) -> TenthResult<Value> {
        self.call_method_priv(receiver, method, args)
    }

    pub fn add_fn(&mut self, name: String, chunk: Chunk) {
        let idx = self.chunks.len();
        self.chunks.push(chunk);
        self.chunk_names.push(name.clone());
        self.functions.insert(name, idx);
    }

    pub fn has_fn(&self, name: &str) -> bool {
        self.functions.contains_key(name)
    }

    pub fn add_native(&mut self, name: String, f: NativeFn) {
        self.natives.insert(name, f);
    }

    pub fn set_global(&mut self, name: String, val: Value) {
        self.globals.insert(name, val);
    }

    /// Push arguments and call a native function.
    pub fn call_native(&mut self, name: &str, args: &[Value]) -> TenthResult<Value> {
        if let Some(f) = self.natives.get(name).copied() {
            f(self, args)
        } else {
            Err(TenthError::RuntimeError { message: format!("未定义的原生函数 '{}'", name) })
        }
    }

    pub fn call(&mut self, name: &str) -> TenthResult<Value> {
        let idx = self.functions.get(name).copied()
            .ok_or_else(|| TenthError::RuntimeError { message: format!("未定义的函数 '{}'", name) })?;
        self.run(idx)
    }

    fn run(&mut self, mut chunk_idx: usize) -> TenthResult<Value> {
        let mut ip: usize = 0;
        let num_args = self.chunks[chunk_idx].num_args;
        // `base` is the stack position BEFORE args. When called via
        // `call_with_args`, args are already pushed on the stack, so we
        // subtract num_args. When called directly (e.g. `vm.call("main")`
        // with num_args=0), this is a no-op.
        let base = self.stack.len().saturating_sub(num_args);
        let num_locals = self.chunks[chunk_idx].num_locals;
        let mut locals = vec![Value::Unit; num_locals.max(num_args)];

        // Pop args into locals (args were pushed right-to-left, so pop in reverse)
        for i in (0..num_args).rev() {
            if self.stack.len() > base {
                locals[i] = self.stack.pop().unwrap();
            }
        }

        // Load initial chunk data
        let mut code = self.chunks[chunk_idx].code.clone();
        let mut strings = self.chunks[chunk_idx].strings.clone();

        // H-4: 独立的循环计数器，用于触发周期性 deadline 检查。
        // 不依赖 step_budget（用户可能只设 --timeout 而不设步数预算）。
        let mut loop_counter: u64 = 0;

        loop {
            // 安全 H-4：step_budget 和 deadline_ms 独立检查。
            // 历史实现把 deadline 检查嵌套在 step_budget 内，导致只设
            // `--timeout` 而不设 step_budget 时 deadline 永远不触发。
            if let Some(ref mut budget) = self.step_budget {
                if *budget == 0 {
                    return Err(TenthError::Timeout {
                        message: "VM 步数预算耗尽".into(),
                    });
                }
                *budget -= 1;
            }
            // 每隔 4096 次循环检查一次墙钟 deadline，开销可忽略。
            // 用独立计数器避免依赖 step_budget（step_budget 可能未设）。
            loop_counter = loop_counter.wrapping_add(1);
            if (loop_counter & 0xFFF) == 0 {
                if let Some(deadline) = self.deadline_ms {
                    let now = std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .map(|d| d.as_millis())
                        .unwrap_or(0);
                    if now >= deadline {
                        return Err(TenthError::Timeout {
                            message: "VM 时间预算耗尽".into(),
                        });
                    }
                }
            }
            // Inline opcode read (no closure, so code/strings can be reassigned)
            let op: Op = {
                use Op::*;
                if ip >= code.len() { return Ok(Value::Unit); }
                let b = code[ip]; ip += 1;
                macro_rules! r { ($t:ty) => {{ let n = std::mem::size_of::<$t>(); if ip + n > code.len() { return Ok(Value::Unit); } let mut buf = [0u8; std::mem::size_of::<$t>()]; buf.copy_from_slice(&code[ip..ip+n]); ip += n; <$t>::from_le_bytes(buf) }}; }
                match b {
                    0 => PushInt(r!(i64)), 1 => PushFloat(r!(f64)),
                    2 => PushBool({ let v = code[ip] != 0; ip += 1; v }),
                    3 => PushStr(r!(u64) as usize),
                    4 => PushUnit, 5 => Pop, 6 => Dup,
                    7 => Load(r!(u64) as usize), 8 => Store(r!(u64) as usize),
                    9 => LoadGlobal(r!(u64) as usize), 10 => StoreGlobal(r!(u64) as usize),
                    11 => Add, 12 => Sub, 13 => Mul, 14 => Div, 15 => Mod,
                    16 => Neg, 17 => Not,
                    18 => Eq, 19 => Neq, 20 => Lt, 21 => Gt, 22 => Lte, 23 => Gte,
                    24 => Jump(r!(i32)), 25 => JmpFalse(r!(i32)), 26 => JmpTrue(r!(i32)),
                    27 => Call(r!(u64) as usize), 28 => CallN(r!(u64) as usize, r!(u64) as usize),
                    29 => MethodCall(r!(u64) as usize, r!(u64) as usize), 30 => Ret,
                    31 => MakeVec(r!(u64) as usize), 32 => MakeMap(r!(u64) as usize),
                    33 => NewStruct(r!(u64) as usize, r!(u64) as usize),
                    34 => LoadField(r!(u64) as usize),
                    35 => StoreField(r!(u64) as usize),
                    36 => IndexGet,
                    37 => SliceStr,
                    38 => MakeEnum(r!(u64) as usize, r!(u64) as usize, r!(u64) as usize),
                    39 => IsEnumVariant(r!(u64) as usize),
                    40 => EnumGetField(r!(u64) as usize),
                    41 => PushRange(r!(i64), r!(i64), { let b = code[ip]; ip += 1; b != 0 }),
                    42 => MoveOp,
                    43 => MakeTensor(r!(u64) as usize, r!(u64) as usize, { let d = code[ip]; ip += 1; d }),
                    44 => MakeClosure(r!(u64) as usize, r!(u64) as usize),
                    45 => PushFloat32(r!(f32)),
                    46 => IsStruct(r!(u64) as usize),
                    _ => Ret,
                }
            };
            match op {
                Op::PushInt(n) => self.stack.push(Value::Int(n)),
                Op::PushFloat(f) => self.stack.push(Value::Float(f)),
                Op::PushFloat32(f) => self.stack.push(Value::Float32(f)),
                Op::PushBool(b) => self.stack.push(Value::Bool(b)),
                Op::PushStr(i) => {
                    let s = strings.get(i).cloned().unwrap_or_default();
                    self.stack.push(Value::String(s));
                }
                Op::PushUnit => self.stack.push(Value::Unit),
                Op::Pop => { self.stack.pop(); }
                Op::Dup => {
                    let v = self.stack.last().cloned().unwrap_or(Value::Unit);
                    self.stack.push(v);
                }

                Op::Load(i) => {
                    let v = locals.get(i).cloned().unwrap_or(Value::Unit);
                    self.stack.push(v);
                }
                Op::Store(i) => {
                    let v = self.stack.pop().unwrap_or(Value::Unit);
                    if i >= locals.len() { locals.resize(i+1, Value::Unit); }
                    locals[i] = v;
                }
                Op::LoadGlobal(i) => {
                    let name = strings.get(i).cloned().unwrap_or_default();
                    let v = self.globals.get(&name).cloned().unwrap_or(Value::Unit);
                    self.stack.push(v);
                }
                Op::StoreGlobal(i) => {
                    let name = strings.get(i).cloned().unwrap_or_default();
                    let v = self.stack.pop().unwrap_or(Value::Unit);
                    self.globals.insert(name, v);
                }

                Op::Add => { let (a,b)=self.pop2(); let r=self.add_priv(&a,&b)?; self.stack.push(r); }
                Op::Sub => { let (a,b)=self.pop2(); let r=self.sub_priv(&a,&b)?; self.stack.push(r); }
                Op::Mul => { let (a,b)=self.pop2(); let r=self.mul_priv(&a,&b)?; self.stack.push(r); }
                Op::Div => { let (a,b)=self.pop2(); let r=self.div_priv(&a,&b)?; self.stack.push(r); }
                Op::Mod => {
                    let b = self.pop_int()?; let a = self.pop_int()?;
                    if b == 0 {
                        return err("整数取模除零");
                    }
                    self.stack.push(Value::Int(a % b));
                }
                Op::Neg => {
                    let v = self.stack.pop().unwrap_or(Value::Unit);
                    self.stack.push(match v {
                        Value::Int(n) => Value::Int(-n),
                        Value::Float(n) => Value::Float(-n),
                        Value::Float32(n) => Value::Float32(-n),
                        Value::Tensor(t) => Value::Tensor(Rc::new(RefCell::new(t.borrow().neg()))),
                        _ => return err("无法取负"),
                    });
                }
                Op::Not => {
                    let v = self.stack.pop().unwrap_or(Value::Unit);
                    self.stack.push(Value::Bool(!v.is_truthy()));
                }

                Op::Eq => { let (a,b)=self.pop2(); self.stack.push(Value::Bool(self.vm_eq(&a,&b))); }
                Op::Neq => { let (a,b)=self.pop2(); self.stack.push(Value::Bool(!self.vm_eq(&a,&b))); }
                Op::Lt => { let (a,b)=self.pop2(); self.stack.push(Value::Bool(self.compare(&a,&b,|x,y|x<y,|x,y|x<y)?)); }
                Op::Gt => { let (a,b)=self.pop2(); self.stack.push(Value::Bool(self.compare(&a,&b,|x,y|x>y,|x,y|x>y)?)); }
                Op::Lte => { let (a,b)=self.pop2(); self.stack.push(Value::Bool(self.compare(&a,&b,|x,y|x<=y,|x,y|x<=y)?)); }
                Op::Gte => { let (a,b)=self.pop2(); self.stack.push(Value::Bool(self.compare(&a,&b,|x,y|x>=y,|x,y|x>=y)?)); }

                Op::Jump(o) => { ip = (ip as i32 + o) as usize; }
                Op::JmpFalse(o) => {
                    if !self.stack.pop().unwrap_or(Value::Unit).is_truthy() {
                        ip = (ip as i32 + o) as usize;
                    }
                }
                Op::JmpTrue(o) => {
                    if self.stack.pop().unwrap_or(Value::Unit).is_truthy() {
                        ip = (ip as i32 + o) as usize;
                    }
                }

                Op::Call(i) => {
                    let name = strings[i].clone();
                    // Try native first (legacy: uses stack depth as arg count)
                    if let Some(native_fn) = self.natives.get(&name).copied() {
                        let n = self.stack.len() - base;
                        let mut args = vec![Value::Unit; n];
                        for i in (0..n).rev() { args[i] = self.stack.pop().unwrap_or(Value::Unit); }
                        let result = native_fn(self, &args)?;
                        self.stack.push(result);
                    } else if let Some(&callee_idx) = self.functions.get(&name) {
                        let callee_args = self.chunks[callee_idx].num_args;
                        let callee_locals = self.chunks[callee_idx].num_locals;
                        self.frames.push(Frame { ip, chunk_idx, locals: locals.clone(), stack_base: base });
                        chunk_idx = callee_idx;
                        code = self.chunks[chunk_idx].code.clone();
                        strings = self.chunks[chunk_idx].strings.clone();
                        ip = 0;
                        locals = vec![Value::Unit; callee_locals.max(callee_args)];
                        for i in (0..callee_args).rev() {
                            if self.stack.len() > base { locals[i] = self.stack.pop().unwrap(); }
                        }
                    } else {
                        return Err(TenthError::RuntimeError { message: format!("未定义的函数 '{}'", name) });
                    }
                }
                Op::CallN(i, num_args) => {
                    let name = strings.get(i).cloned().unwrap_or_default();
                    let n = num_args;
                    let mut args = vec![Value::Unit; n];
                    for i in (0..n).rev() { args[i] = self.stack.pop().unwrap_or(Value::Unit); }

                    // Try to find the function by name, checking globals for FnRef closures
                    let callee_name = if let Some(Value::FnRef { name: fname, .. }) = self.globals.get(&name) {
                        fname.clone()
                    } else {
                        name.clone()
                    };

                    if let Some(native_fn) = self.natives.get(&callee_name).copied() {
                        let result = native_fn(self, &args)?;
                        self.stack.push(result);
                    } else if let Some(&callee_idx) = self.functions.get(&callee_name) {
                        self.frames.push(Frame { ip, chunk_idx, locals: locals.clone(), stack_base: self.stack.len() });
                        chunk_idx = callee_idx;
                        code = self.chunks[chunk_idx].code.clone();
                        strings = self.chunks[chunk_idx].strings.clone();
                        ip = 0;
                        locals = args;
                        locals.resize(self.chunks[chunk_idx].num_locals.max(locals.len()), Value::Unit);
                    } else if let Some(native_fn) = self.natives.get(&name).copied() {
                        let result = native_fn(self, &args)?;
                        self.stack.push(result);
                    } else if let Some(&callee_idx) = self.functions.get(&name) {
                        self.frames.push(Frame { ip, chunk_idx, locals: locals.clone(), stack_base: self.stack.len() });
                        chunk_idx = callee_idx;
                        code = self.chunks[chunk_idx].code.clone();
                        strings = self.chunks[chunk_idx].strings.clone();
                        ip = 0;
                        locals = args;
                        locals.resize(self.chunks[chunk_idx].num_locals.max(locals.len()), Value::Unit);
                    } else {
                        return Err(TenthError::RuntimeError { message: format!("未定义的函数 '{}'", name) });
                    }
                }
                Op::MethodCall(i, num_args) => {
                    let name = strings.get(i).cloned().unwrap_or_default();
                    let n = num_args;
                    let mut args = vec![Value::Unit; n];
                    for i in (0..n).rev() { args[i] = self.stack.pop().unwrap_or(Value::Unit); }
                    let receiver = self.stack.pop().unwrap_or(Value::Unit);
                    let result = self.call_method_priv(&receiver, &name, &args)?;
                    self.stack.push(result);
                }

                Op::Ret => {
                    let result = self.stack.pop().unwrap_or(Value::Unit);
                    if let Some(f) = self.frames.pop() {
                        self.stack.truncate(f.stack_base);
                        self.stack.push(result);
                        ip = f.ip;
                        chunk_idx = f.chunk_idx;
                        code = self.chunks[chunk_idx].code.clone();
                        strings = self.chunks[chunk_idx].strings.clone();
                        locals = f.locals;
                    } else {
                        return Ok(result);
                    }
                }

                Op::MakeVec(n) => {
                    let mut v = Vec::new();
                    for _ in 0..n { v.push(self.stack.pop().unwrap_or(Value::Unit)); }
                    v.reverse();
                    self.stack.push(Value::Vec(Rc::new(RefCell::new(v))));
                }
                Op::MakeMap(n) => {
                    let mut m = HashMap::new();
                    for _ in 0..n {
                        let val = self.stack.pop().unwrap_or(Value::Unit);
                        let key = match self.stack.pop().unwrap_or(Value::Unit) {
                            Value::String(s) => s,
                            _ => String::new(),
                        };
                        m.insert(key, val);
                    }
                    self.stack.push(Value::Map(Rc::new(RefCell::new(m))));
                }

                Op::NewStruct(name_i, n) => {
                    let name = strings.get(name_i).cloned().unwrap_or_default();
                    let mut fields = Vec::new();
                    for _ in 0..n {
                        // Compiler pushes value then name (name on top); pop name first
                        let fname = match self.stack.pop().unwrap_or(Value::Unit) {
                            Value::String(s) => s,
                            _ => String::new(),
                        };
                        let val = self.stack.pop().unwrap_or(Value::Unit);
                        fields.push((fname, val));
                    }
                    fields.reverse();
                    self.stack.push(Value::Struct { name, fields: Rc::new(RefCell::new(fields)) });
                }

                Op::LoadField(i) => {
                    let fname = strings.get(i).cloned().unwrap_or_default();
                    let val = self.stack.pop().unwrap_or(Value::Unit);
                    let v = self.get_field(&val, &fname)?;
                    self.stack.push(v);
                }

                Op::StoreField(i) => {
                    let fname = strings.get(i).cloned().unwrap_or_default();
                    let new_val = self.stack.pop().unwrap_or(Value::Unit);
                    let target = self.stack.pop().unwrap_or(Value::Unit);
                    self.set_field(&target, &fname, new_val)?;
                }

                Op::IndexGet => {
                    let idx = self.stack.pop().unwrap_or(Value::Unit);
                    let target = self.stack.pop().unwrap_or(Value::Unit);
                    match target {
                        Value::Vec(items) => {
                            let i = idx.as_int().unwrap_or(0) as usize;
                            let v = items.borrow().get(i).cloned().unwrap_or(Value::Unit);
                            self.stack.push(v);
                        }
                        Value::String(s) => {
                            let i = idx.as_int().unwrap_or(0) as usize;
                            let c = s.chars().nth(i).map(|c| c.to_string()).unwrap_or_default();
                            self.stack.push(Value::String(c));
                        }
                        _ => return err("无法索引"),
                    }
                }

                Op::SliceStr => {
                    let end_idx = self.pop_int()? as usize;
                    let start_idx = self.pop_int()? as usize;
                    let target = self.stack.pop().unwrap_or(Value::Unit);
                    match target {
                        Value::String(s) => {
                            let chars: Vec<char> = s.chars().collect();
                            let len = chars.len();
                            let si = start_idx.min(len);
                            let ei = end_idx.min(len);
                            if si > ei {
                                return err("字符串切片起始位置大于结束位置");
                            }
                            let slice: String = chars[si..ei].iter().collect();
                            self.stack.push(Value::String(slice));
                        }
                        _ => return err("SliceStr 需要字符串目标"),
                    }
                }

                Op::MakeEnum(name_i, variant_i, n) => {
                    let enum_name = strings.get(name_i).cloned().unwrap_or_default();
                    let variant = strings.get(variant_i).cloned().unwrap_or_default();
                    let mut fields = Vec::new();
                    for _ in 0..n {
                        let fname = match self.stack.pop().unwrap_or(Value::Unit) {
                            Value::String(s) => s,
                            _ => String::new(),
                        };
                        let val = self.stack.pop().unwrap_or(Value::Unit);
                        fields.push((fname, val));
                    }
                    fields.reverse();
                    self.stack.push(Value::Enum {
                        enum_name,
                        variant,
                        fields: Rc::new(RefCell::new(fields)),
                    });
                }

                Op::IsEnumVariant(variant_i) => {
                    let variant_name = strings.get(variant_i).cloned().unwrap_or_default();
                    let val = self.stack.pop().unwrap_or(Value::Unit);
                    let matches = match &val {
                        Value::Enum { variant, .. } => variant == &variant_name,
                        _ => false,
                    };
                    self.stack.push(Value::Bool(matches));
                }

                Op::IsStruct(name_i) => {
                    let struct_name = strings.get(name_i).cloned().unwrap_or_default();
                    let val = self.stack.pop().unwrap_or(Value::Unit);
                    let matches = match &val {
                        Value::Struct { name, .. } => name == &struct_name,
                        _ => false,
                    };
                    self.stack.push(Value::Bool(matches));
                }

                Op::EnumGetField(field_i) => {
                    let field_name = strings.get(field_i).cloned().unwrap_or_default();
                    let val = self.stack.pop().unwrap_or(Value::Unit);
                    let found = match val {
                        Value::Enum { fields, .. } => {
                            let mut result = None;
                            for (n, v) in fields.borrow().iter() {
                                if n == &field_name {
                                    result = Some(v.clone());
                                    break;
                                }
                            }
                            result
                        }
                        _ => None,
                    };
                    match found {
                        Some(v) => self.stack.push(v),
                        None => self.stack.push(Value::Unit),
                    }
                }

                Op::PushRange(start, end, inclusive) => {
                    self.stack.push(Value::Range { start, end, inclusive });
                }

                Op::MoveOp => {
                    // no-op: move semantics are checked at HIR level
                }

                Op::MakeTensor(rows, cols, dtype) => {
                    use super::tensor::Tensor;
                    use crate::hir::types::BaseType;
                    let total = rows * cols;
                    let dt = match dtype {
                        1 => BaseType::F32,
                        _ => BaseType::F64,
                    };
                    if dt == BaseType::F32 {
                        let mut data: Vec<f32> = Vec::with_capacity(total);
                        for _ in 0..total {
                            let v = self.stack.pop().unwrap_or(Value::Float32(0.0));
                            data.push(match v {
                                Value::Float32(f) => f,
                                Value::Float(f) => f as f32,
                                Value::Int(n) => n as f32,
                                _ => 0.0,
                            });
                        }
                        data.reverse();
                        let tensor = Tensor::from_vec_f32(data, vec![rows, cols]);
                        self.stack.push(Value::Tensor(Rc::new(RefCell::new(tensor))));
                    } else {
                        let mut data = Vec::with_capacity(total);
                        for _ in 0..total {
                            let v = self.stack.pop().unwrap_or(Value::Float(0.0));
                            data.push(match v {
                                Value::Float(f) => f,
                                Value::Float32(f) => f as f64,
                                Value::Int(n) => n as f64,
                                _ => 0.0,
                            });
                        }
                        data.reverse();
                        let tensor = Tensor::from_vec(data, vec![rows, cols]);
                        self.stack.push(Value::Tensor(Rc::new(RefCell::new(tensor))));
                    }
                }

                Op::MakeClosure(params_count, name_idx) => {
                    // Create a FnRef value pointing to the closure function
                    let name = strings.get(name_idx).cloned().unwrap_or_default();
                    let param_names: Vec<(String, crate::hir::types::Type)> = (0..params_count)
                        .map(|i| (format!("__param_{i}"), crate::hir::types::Type::Unknown))
                        .collect();
                    self.stack.push(Value::FnRef {
                        name,
                        params: param_names,
                        return_type: crate::hir::types::Type::Unknown,
                    });
                }
            }
        }
    }

    fn pop2(&mut self) -> (Value, Value) {
        let b = self.stack.pop().unwrap_or(Value::Unit);
        let a = self.stack.pop().unwrap_or(Value::Unit);
        (a, b)
    }

    fn pop_int(&mut self) -> TenthResult<i64> {
        match self.stack.pop() {
            Some(Value::Int(n)) => Ok(n),
            _ => err("期望整数"),
        }
    }

    fn add_priv(&mut self, a: &Value, b: &Value) -> TenthResult<Value> {
        Ok(match (a, b) {
            (Value::Int(x), Value::Int(y)) => Value::Int(x + y),
            (Value::Float(x), Value::Float(y)) => Value::Float(x + y),
            (Value::Int(x), Value::Float(y)) => Value::Float(*x as f64 + y),
            (Value::Float(x), Value::Int(y)) => Value::Float(x + *y as f64),
            // f32 路径：相同 dtype 保持 f32，混合提升为 f64
            (Value::Float32(x), Value::Float32(y)) => Value::Float32(x + y),
            (Value::Int(x), Value::Float32(y)) => Value::Float32(*x as f32 + y),
            (Value::Float32(x), Value::Int(y)) => Value::Float32(x + *y as f32),
            (Value::Float32(x), Value::Float(y)) => Value::Float(*x as f64 + y),
            (Value::Float(x), Value::Float32(y)) => Value::Float(x + *y as f64),
            (Value::String(x), Value::String(y)) => Value::String(format!("{x}{y}")),
            (Value::Tensor(t), Value::Float(s)) => {
                let result = Rc::new(RefCell::new(t.borrow().add_scalar(*s)));
                if self.recording {
                    let s_tensor = Rc::new(RefCell::new(Tensor::from_vec(vec![*s], vec![1])));
                    self.record_binary(TapeOp::Add, &t, &s_tensor, &result);
                }
                Value::Tensor(result)
            }
            (Value::Float(s), Value::Tensor(t)) => {
                let result = Rc::new(RefCell::new(t.borrow().add_scalar(*s)));
                if self.recording {
                    let s_tensor = Rc::new(RefCell::new(Tensor::from_vec(vec![*s], vec![1])));
                    self.record_binary(TapeOp::Add, &s_tensor, &t, &result);
                }
                Value::Tensor(result)
            }
            // f32 标量 × Tensor：转为 f64 调用 scalar 方法（scalar 方法按 Tensor dtype 分支保持精度）
            (Value::Tensor(t), Value::Float32(s)) => {
                let result = Rc::new(RefCell::new(t.borrow().add_scalar(*s as f64)));
                if self.recording {
                    let s_tensor = Rc::new(RefCell::new(Tensor::from_vec(vec![*s as f64], vec![1])));
                    self.record_binary(TapeOp::Add, &t, &s_tensor, &result);
                }
                Value::Tensor(result)
            }
            (Value::Float32(s), Value::Tensor(t)) => {
                let result = Rc::new(RefCell::new(t.borrow().add_scalar(*s as f64)));
                if self.recording {
                    let s_tensor = Rc::new(RefCell::new(Tensor::from_vec(vec![*s as f64], vec![1])));
                    self.record_binary(TapeOp::Add, &s_tensor, &t, &result);
                }
                Value::Tensor(result)
            }
            (Value::Tensor(t1), Value::Tensor(t2)) => {
                let result_tensor = t1.borrow().add_tensor(&t2.borrow())
                    .map_err(|msg| TenthError::RuntimeError { message: msg })?;
                let result = Rc::new(RefCell::new(result_tensor));
                if self.recording { self.record_binary(TapeOp::Add, &t1, &t2, &result); }
                Value::Tensor(result)
            }
            _ => return err("+ 类型不匹配"),
        })
    }

    fn sub_priv(&mut self, a: &Value, b: &Value) -> TenthResult<Value> {
        Ok(match (a, b) {
            (Value::Int(x), Value::Int(y)) => Value::Int(x - y),
            (Value::Float(x), Value::Float(y)) => Value::Float(x - y),
            (Value::Int(x), Value::Float(y)) => Value::Float(*x as f64 - y),
            (Value::Float(x), Value::Int(y)) => Value::Float(x - *y as f64),
            // f32 路径
            (Value::Float32(x), Value::Float32(y)) => Value::Float32(x - y),
            (Value::Int(x), Value::Float32(y)) => Value::Float32(*x as f32 - y),
            (Value::Float32(x), Value::Int(y)) => Value::Float32(x - *y as f32),
            (Value::Float32(x), Value::Float(y)) => Value::Float(*x as f64 - y),
            (Value::Float(x), Value::Float32(y)) => Value::Float(x - *y as f64),
            (Value::Tensor(t), Value::Float(s)) => {
                let result = Rc::new(RefCell::new(t.borrow().add_scalar(-*s)));
                if self.recording {
                    let s_tensor = Rc::new(RefCell::new(Tensor::from_vec(vec![*s], vec![1])));
                    self.record_binary(TapeOp::Sub, &t, &s_tensor, &result);
                }
                Value::Tensor(result)
            }
            (Value::Float(s), Value::Tensor(t)) => {
                let result = Rc::new(RefCell::new(t.borrow().add_scalar(-*s)));
                if self.recording {
                    let s_tensor = Rc::new(RefCell::new(Tensor::from_vec(vec![*s], vec![1])));
                    self.record_binary(TapeOp::Sub, &s_tensor, &t, &result);
                }
                Value::Tensor(result)
            }
            // f32 标量 × Tensor
            (Value::Tensor(t), Value::Float32(s)) => {
                let s_f64 = *s as f64;
                let result = Rc::new(RefCell::new(t.borrow().add_scalar(-s_f64)));
                if self.recording {
                    let s_tensor = Rc::new(RefCell::new(Tensor::from_vec(vec![s_f64], vec![1])));
                    self.record_binary(TapeOp::Sub, &t, &s_tensor, &result);
                }
                Value::Tensor(result)
            }
            (Value::Float32(s), Value::Tensor(t)) => {
                let s_f64 = *s as f64;
                let result = Rc::new(RefCell::new(t.borrow().add_scalar(-s_f64)));
                if self.recording {
                    let s_tensor = Rc::new(RefCell::new(Tensor::from_vec(vec![s_f64], vec![1])));
                    self.record_binary(TapeOp::Sub, &s_tensor, &t, &result);
                }
                Value::Tensor(result)
            }
            (Value::Tensor(t1), Value::Tensor(t2)) => {
                let result_tensor = t1.borrow().sub_tensor(&t2.borrow())
                    .map_err(|msg| TenthError::RuntimeError { message: msg })?;
                let result = Rc::new(RefCell::new(result_tensor));
                if self.recording { self.record_binary(TapeOp::Sub, &t1, &t2, &result); }
                Value::Tensor(result)
            }
            _ => return err("- 类型不匹配"),
        })
    }

    fn mul_priv(&mut self, a: &Value, b: &Value) -> TenthResult<Value> {
        Ok(match (a, b) {
            (Value::Int(x), Value::Int(y)) => Value::Int(x * y),
            (Value::Float(x), Value::Float(y)) => Value::Float(x * y),
            (Value::Int(x), Value::Float(y)) => Value::Float(*x as f64 * y),
            (Value::Float(x), Value::Int(y)) => Value::Float(x * *y as f64),
            // f32 路径
            (Value::Float32(x), Value::Float32(y)) => Value::Float32(x * y),
            (Value::Int(x), Value::Float32(y)) => Value::Float32(*x as f32 * y),
            (Value::Float32(x), Value::Int(y)) => Value::Float32(x * *y as f32),
            (Value::Float32(x), Value::Float(y)) => Value::Float(*x as f64 * y),
            (Value::Float(x), Value::Float32(y)) => Value::Float(x * *y as f64),
            (Value::Tensor(t), Value::Float(s)) => {
                let result = Rc::new(RefCell::new(t.borrow().mul_scalar(*s)));
                if self.recording {
                    let s_tensor = Rc::new(RefCell::new(Tensor::from_vec(vec![*s], vec![1])));
                    self.record_binary(TapeOp::Mul, &t, &s_tensor, &result);
                }
                Value::Tensor(result)
            }
            (Value::Float(s), Value::Tensor(t)) => {
                let result = Rc::new(RefCell::new(t.borrow().mul_scalar(*s)));
                if self.recording {
                    let s_tensor = Rc::new(RefCell::new(Tensor::from_vec(vec![*s], vec![1])));
                    self.record_binary(TapeOp::Mul, &s_tensor, &t, &result);
                }
                Value::Tensor(result)
            }
            // f32 标量 × Tensor
            (Value::Tensor(t), Value::Float32(s)) => {
                let s_f64 = *s as f64;
                let result = Rc::new(RefCell::new(t.borrow().mul_scalar(s_f64)));
                if self.recording {
                    let s_tensor = Rc::new(RefCell::new(Tensor::from_vec(vec![s_f64], vec![1])));
                    self.record_binary(TapeOp::Mul, &t, &s_tensor, &result);
                }
                Value::Tensor(result)
            }
            (Value::Float32(s), Value::Tensor(t)) => {
                let s_f64 = *s as f64;
                let result = Rc::new(RefCell::new(t.borrow().mul_scalar(s_f64)));
                if self.recording {
                    let s_tensor = Rc::new(RefCell::new(Tensor::from_vec(vec![s_f64], vec![1])));
                    self.record_binary(TapeOp::Mul, &s_tensor, &t, &result);
                }
                Value::Tensor(result)
            }
            (Value::Tensor(t1), Value::Tensor(t2)) => {
                let result_tensor = t1.borrow().mul_tensor(&t2.borrow())
                    .map_err(|msg| TenthError::RuntimeError { message: msg })?;
                let result = Rc::new(RefCell::new(result_tensor));
                if self.recording { self.record_binary(TapeOp::Mul, &t1, &t2, &result); }
                Value::Tensor(result)
            }
            _ => return err("* 类型不匹配"),
        })
    }

    fn div_priv(&mut self, a: &Value, b: &Value) -> TenthResult<Value> {
        Ok(match (a, b) {
            (Value::Int(x), Value::Int(y)) => {
                if *y == 0 {
                    return err("整数除零");
                }
                Value::Int(x / y)
            }
            (Value::Float(x), Value::Float(y)) => Value::Float(x / y),
            (Value::Int(x), Value::Float(y)) => Value::Float(*x as f64 / y),
            (Value::Float(x), Value::Int(y)) => Value::Float(x / *y as f64),
            // f32 路径
            (Value::Float32(x), Value::Float32(y)) => Value::Float32(x / y),
            (Value::Int(x), Value::Float32(y)) => Value::Float32(*x as f32 / y),
            (Value::Float32(x), Value::Int(y)) => Value::Float32(x / *y as f32),
            (Value::Float32(x), Value::Float(y)) => Value::Float(*x as f64 / y),
            (Value::Float(x), Value::Float32(y)) => Value::Float(x / *y as f64),
            (Value::Tensor(t), Value::Float(s)) => {
                let result = Rc::new(RefCell::new(t.borrow().div_scalar(*s)));
                if self.recording {
                    let s_tensor = Rc::new(RefCell::new(Tensor::from_vec(vec![*s], vec![1])));
                    self.record_binary(TapeOp::Div, &t, &s_tensor, &result);
                }
                Value::Tensor(result)
            }
            (Value::Float(s), Value::Tensor(t)) => {
                // s / t: scalar divided by tensor element-wise
                let result = Rc::new(RefCell::new(t.borrow().div_scalar_inv(*s)));
                if self.recording {
                    let s_tensor = Rc::new(RefCell::new(Tensor::from_vec(vec![*s], vec![1])));
                    self.record_binary(TapeOp::Div, &s_tensor, &t, &result);
                }
                Value::Tensor(result)
            }
            // f32 标量 × Tensor
            (Value::Tensor(t), Value::Float32(s)) => {
                let s_f64 = *s as f64;
                let result = Rc::new(RefCell::new(t.borrow().div_scalar(s_f64)));
                if self.recording {
                    let s_tensor = Rc::new(RefCell::new(Tensor::from_vec(vec![s_f64], vec![1])));
                    self.record_binary(TapeOp::Div, &t, &s_tensor, &result);
                }
                Value::Tensor(result)
            }
            (Value::Float32(s), Value::Tensor(t)) => {
                // s / t: scalar divided by tensor element-wise
                let s_f64 = *s as f64;
                let result = Rc::new(RefCell::new(t.borrow().div_scalar_inv(s_f64)));
                if self.recording {
                    let s_tensor = Rc::new(RefCell::new(Tensor::from_vec(vec![s_f64], vec![1])));
                    self.record_binary(TapeOp::Div, &s_tensor, &t, &result);
                }
                Value::Tensor(result)
            }
            (Value::Tensor(t1), Value::Tensor(t2)) => {
                let result_tensor = t1.borrow().div_tensor(&t2.borrow())
                    .map_err(|msg| TenthError::RuntimeError { message: msg })?;
                let result = Rc::new(RefCell::new(result_tensor));
                if self.recording { self.record_binary(TapeOp::Div, &t1, &t2, &result); }
                Value::Tensor(result)
            }
            _ => return err("/ 类型不匹配"),
        })
    }

    fn compare(&self, a: &Value, b: &Value, nf: fn(f64, f64) -> bool, sf: fn(&str, &str) -> bool) -> TenthResult<bool> {
        Ok(match (a, b) {
            (Value::Int(x), Value::Int(y)) => nf(*x as f64, *y as f64),
            (Value::Float(x), Value::Float(y)) => nf(*x, *y),
            (Value::Int(x), Value::Float(y)) => nf(*x as f64, *y),
            (Value::Float(x), Value::Int(y)) => nf(*x, *y as f64),
            // f32 路径：提升为 f64 比较
            (Value::Float32(x), Value::Float32(y)) => nf(*x as f64, *y as f64),
            (Value::Int(x), Value::Float32(y)) => nf(*x as f64, *y as f64),
            (Value::Float32(x), Value::Int(y)) => nf(*x as f64, *y as f64),
            (Value::Float32(x), Value::Float(y)) => nf(*x as f64, *y),
            (Value::Float(x), Value::Float32(y)) => nf(*x, *y as f64),
            (Value::String(x), Value::String(y)) => sf(x, y),
            _ => return err("无法比较"),
        })
    }

    fn call_method_priv(&mut self, receiver: &Value, method: &str, args: &[Value]) -> TenthResult<Value> {
        // Auto-deref via cloning (avoids borrow issues)
        let recv = match receiver {
            Value::Ref(rc) => rc.borrow().clone(),
            Value::MutRef(w) => w.upgrade().map(|rc| rc.borrow().clone()).unwrap_or(Value::Moved),
            Value::Shared(rc) => rc.borrow().clone(),
            v => v.clone(),
        };
        match recv {
            Value::String(s) => match method {
                "len" => Ok(Value::Int(s.chars().count() as i64)),
                "trim" => Ok(Value::String(s.trim().to_string())),
                "to_upper" => Ok(Value::String(s.to_uppercase())),
                "to_lower" => Ok(Value::String(s.to_lowercase())),
                "replace" => {
                    if args.len() >= 2 {
                        if let (Value::String(from), Value::String(to)) = (&args[0], &args[1]) {
                            Ok(Value::String(s.replace(from.as_str(), to.as_str())))
                        } else { err("replace() 需要 2 个字符串参数") }
                    } else { err("replace() 需要 2 个字符串参数") }
                }
                "split" => {
                    if let Some(Value::String(delim)) = args.first() {
                        let parts: Vec<Value> = s.split(delim.as_str()).map(|p| Value::String(p.to_string())).collect();
                        Ok(Value::Vec(Rc::new(RefCell::new(parts))))
                    } else { err("split() 需要一个字符串分隔符") }
                }
                "substring" => {
                    if args.len() >= 2 {
                        let start = args[0].as_int().unwrap_or(0).max(0) as usize;
                        let len = args[1].as_int().unwrap_or(0).max(0) as usize;
                        let chars: Vec<char> = s.chars().collect();
                        let end = (start + len).min(chars.len());
                        let sub: String = chars[start..end].iter().collect();
                        Ok(Value::String(sub))
                    } else { err("substring() 需要起始位置和长度") }
                }
                "contains" => {
                    if let Some(Value::String(sub)) = args.first() {
                        Ok(Value::Bool(s.contains(sub.as_str())))
                    } else { err("contains() 需要一个字符串参数") }
                }
                "find" => {
                    if let Some(Value::String(sub)) = args.first() {
                        Ok(Value::Int(s.find(sub.as_str()).map(|i| i as i64).unwrap_or(-1)))
                    } else { err("find() 需要一个字符串参数") }
                }
                "starts_with" => {
                    if let Some(Value::String(prefix)) = args.first() {
                        Ok(Value::Bool(s.starts_with(prefix.as_str())))
                    } else { err("starts_with() 需要一个字符串参数") }
                }
                "ends_with" => {
                    if let Some(Value::String(suffix)) = args.first() {
                        Ok(Value::Bool(s.ends_with(suffix.as_str())))
                    } else { err("ends_with() 需要一个字符串参数") }
                }
                "parse_int" => Ok(Value::Int(s.trim().parse::<i64>().unwrap_or(0))),
                "parse_float" => Ok(Value::Float(s.trim().parse::<f64>().unwrap_or(0.0))),
                "is_empty" => Ok(Value::Bool(s.is_empty())),
                "repeat" => {
                    if let Some(arg) = args.first() {
                        let n = arg.as_int().unwrap_or(0).max(0) as usize;
                        Ok(Value::String(s.repeat(n)))
                    } else { err("repeat() 需要一个整数参数") }
                }
                "chars" => {
                    let chars: Vec<Value> = s.chars().map(|c| Value::String(c.to_string())).collect();
                    Ok(Value::Vec(Rc::new(RefCell::new(chars))))
                }
                "bytes" => {
                    let bytes: Vec<Value> = s.bytes().map(|b| Value::Int(b as i64)).collect();
                    Ok(Value::Vec(Rc::new(RefCell::new(bytes))))
                }
                "trim_start" => Ok(Value::String(s.trim_start().to_string())),
                "trim_end" => Ok(Value::String(s.trim_end().to_string())),
                "strip_prefix" => {
                    if let Some(Value::String(prefix)) = args.first() {
                        Ok(match s.strip_prefix(prefix.as_str()) {
                            Some(rest) => Value::String(rest.to_string()),
                            None => Value::String(s.to_string()),
                        })
                    } else { err("strip_prefix() 需要一个字符串参数") }
                }
                "strip_suffix" => {
                    if let Some(Value::String(suffix)) = args.first() {
                        Ok(match s.strip_suffix(suffix.as_str()) {
                            Some(rest) => Value::String(rest.to_string()),
                            None => Value::String(s.to_string()),
                        })
                    } else { err("strip_suffix() 需要一个字符串参数") }
                }
                _ => err(&format!("字符串没有方法 '{}'", method)),
            },
            Value::Vec(items) => match method {
                "len" => Ok(Value::Int(items.borrow().len() as i64)),
                "push" => {
                    if args.len() == 1 {
                        items.borrow_mut().push(args[0].clone());
                        Ok(Value::Unit)
                    } else { err("push 需要 1 个参数") }
                }
                "get" => {
                    if args.len() == 1 {
                        let idx = args[0].as_int().unwrap_or(0) as usize;
                        Ok(items.borrow().get(idx).cloned().unwrap_or(Value::Unit))
                    } else { err("get 需要 1 个参数") }
                }
                "pop" => {
                    let mut vec = items.borrow_mut();
                    match vec.pop() {
                        Some(v) => Ok(v),
                        None => err("对空 Vec 调用 pop()"),
                    }
                }
                "set" => {
                    if args.len() != 2 { return err("set() 需要 2 个参数 (索引, 值)"); }
                    let idx = args[0].as_int().unwrap_or(0) as usize;
                    let mut vec = items.borrow_mut();
                    if idx < vec.len() {
                        vec[idx] = args[1].clone();
                        Ok(Value::Unit)
                    } else { err(&format!("Vec 索引 {} 越界", idx)) }
                }
                "clear" => {
                    items.borrow_mut().clear();
                    Ok(Value::Unit)
                }
                "contains" => {
                    if args.len() != 1 { return err("contains() 需要 1 个参数"); }
                    let vec = items.borrow();
                    let found = vec.iter().any(|v| self.vm_eq(v, &args[0]));
                    Ok(Value::Bool(found))
                }
                "insert" => {
                    if args.len() != 2 { return err("insert() 需要 2 个参数 (索引, 值)"); }
                    let idx = args[0].as_int().unwrap_or(0) as usize;
                    items.borrow_mut().insert(idx, args[1].clone());
                    Ok(Value::Unit)
                }
                "remove" => {
                    if args.len() != 1 { return err("remove() 需要 1 个参数 (索引)"); }
                    let idx = args[0].as_int().unwrap_or(0) as usize;
                    let vec_len = items.borrow().len();
                    if idx < vec_len {
                        Ok(items.borrow_mut().remove(idx))
                    } else { err(&format!("Vec 索引 {} 越界", idx)) }
                }
                "join" => {
                    if args.len() != 1 { return err("join() 需要 1 个参数 (分隔符)"); }
                    if let Value::String(delim) = &args[0] {
                        let vec = items.borrow();
                        let parts: Vec<String> = vec.iter().map(|v| match v {
                            Value::String(s) => s.clone(),
                            other => format!("{:?}", other),
                        }).collect();
                        Ok(Value::String(parts.join(delim)))
                    } else { err("join() 分隔符必须是字符串") }
                }
                "is_empty" => Ok(Value::Bool(items.borrow().is_empty())),
                _ => err(&format!("Vec 没有方法 '{}'", method)),
            },
            Value::Map(m) => match method {
                "len" => Ok(Value::Int(m.borrow().len() as i64)),
                "insert" => {
                    if args.len() != 2 { return err("insert() 需要 2 个参数 (键, 值)"); }
                    if let Value::String(key) = &args[0] {
                        m.borrow_mut().insert(key.clone(), args[1].clone());
                        Ok(Value::Unit)
                    } else { err("Map 的键必须是字符串") }
                }
                "get" => {
                    if args.len() != 1 { return err("get() 需要 1 个参数 (键)"); }
                    if let Value::String(key) = &args[0] {
                        Ok(m.borrow().get(key).cloned().unwrap_or(Value::Unit))
                    } else { err("Map 的键必须是字符串") }
                }
                "contains_key" => {
                    if args.len() != 1 { return err("contains_key() 需要 1 个参数 (键)"); }
                    if let Value::String(key) = &args[0] {
                        Ok(Value::Bool(m.borrow().contains_key(key)))
                    } else { err("Map 的键必须是字符串") }
                }
                "remove" => {
                    if args.len() != 1 { return err("remove() 需要 1 个参数 (键)"); }
                    if let Value::String(key) = &args[0] {
                        Ok(m.borrow_mut().remove(key).unwrap_or(Value::Unit))
                    } else { err("Map 的键必须是字符串") }
                }
                "keys" => {
                    let keys: Vec<Value> = m.borrow().keys().map(|k| Value::String(k.clone())).collect();
                    Ok(Value::Vec(Rc::new(RefCell::new(keys))))
                }
                "values" => {
                    let values: Vec<Value> = m.borrow().values().cloned().collect();
                    Ok(Value::Vec(Rc::new(RefCell::new(values))))
                }
                "is_empty" => Ok(Value::Bool(m.borrow().is_empty())),
                _ => err(&format!("Map 没有方法 '{}'", method)),
            },
            Value::Tensor(t) => {
                let tensor = t.borrow();
                match method {
                    // ── Reductions ──
                    "sum" => {
                        if args.is_empty() {
                            if self.recording {
                                let scalar = tensor.sum();
                                let result = Rc::new(RefCell::new(Tensor::from_vec(vec![scalar], vec![1])));
                                self.record_unary(TapeOp::Sum, &t, &result);
                                Ok(Value::Tensor(result))
                            } else if tensor.is_f32() {
                                Ok(Value::Float32(tensor.sum() as f32))
                            } else {
                                Ok(Value::Float(tensor.sum()))
                            }
                        } else {
                            let axis = args[0].as_int().unwrap_or(0) as usize;
                            match tensor.sum_axis(axis) {
                                Ok(t) => Ok(Value::Tensor(Rc::new(RefCell::new(t)))),
                                Err(msg) => err(&msg),
                            }
                        }
                    }
                    "mean" => {
                        if self.recording {
                            let scalar = tensor.mean();
                            let result = Rc::new(RefCell::new(Tensor::from_vec(vec![scalar], vec![1])));
                            self.record_unary(TapeOp::Mean, &t, &result);
                            Ok(Value::Tensor(result))
                        } else if tensor.is_f32() {
                            Ok(Value::Float32(tensor.mean() as f32))
                        } else {
                            Ok(Value::Float(tensor.mean()))
                        }
                    }
                    "max_val" => {
                        if tensor.is_f32() {
                            Ok(Value::Float32(tensor.max_val() as f32))
                        } else {
                            Ok(Value::Float(tensor.max_val()))
                        }
                    }

                    // ── Elementwise unary ──
                    "abs" => {
                        let result = Rc::new(RefCell::new(tensor.abs()));
                        if self.recording { self.record_unary(TapeOp::Abs, &t, &result); }
                        Ok(Value::Tensor(result))
                    }
                    "sqrt" => {
                        let result = Rc::new(RefCell::new(tensor.sqrt()));
                        Ok(Value::Tensor(result))
                    }
                    "exp" => {
                        let result = Rc::new(RefCell::new(tensor.exp()));
                        if self.recording { self.record_unary(TapeOp::Exp, &t, &result); }
                        Ok(Value::Tensor(result))
                    }
                    "log" => {
                        let result = Rc::new(RefCell::new(tensor.log()));
                        if self.recording { self.record_unary(TapeOp::Log, &t, &result); }
                        Ok(Value::Tensor(result))
                    }
                    "relu" => {
                        let result = Rc::new(RefCell::new(tensor.relu()));
                        if self.recording { self.record_unary(TapeOp::ReLU, &t, &result); }
                        Ok(Value::Tensor(result))
                    }
                    "sigmoid" => {
                        let result = Rc::new(RefCell::new(tensor.sigmoid()));
                        if self.recording { self.record_unary(TapeOp::Sigmoid, &t, &result); }
                        Ok(Value::Tensor(result))
                    }
                    "tanh" => {
                        let result = Rc::new(RefCell::new(tensor.tanh()));
                        Ok(Value::Tensor(result))
                    }
                    "gelu" => {
                        let result = Rc::new(RefCell::new(tensor.gelu()));
                        if self.recording { self.record_unary(TapeOp::Gelu, &t, &result); }
                        Ok(Value::Tensor(result))
                    }
                    "softmax" => {
                        let result_tensor = tensor.softmax().ok_or_else(|| {
                            TenthError::RuntimeError { message: "softmax 计算失败".into() }
                        })?;
                        let result = Rc::new(RefCell::new(result_tensor));
                        if self.recording { self.record_unary(TapeOp::Softmax, &t, &result); }
                        Ok(Value::Tensor(result))
                    }
                    "argmax" => Ok(Value::Int(tensor.argmax())),
                    // 梯度裁剪辅助：元素级裁剪到 [min_val, max_val]（与 interpreter 同步）
                    "clip_scalar" => {
                        if args.len() < 2 {
                            return Err(TenthError::RuntimeError {
                                message: "clip_scalar() 需要 min_val 和 max_val".into(),
                            });
                        }
                        let min_val = args[0].as_float().unwrap_or(f64::NEG_INFINITY);
                        let max_val = args[1].as_float().unwrap_or(f64::INFINITY);
                        let clipped = tensor.clip_scalar(min_val, max_val);
                        Ok(Value::Tensor(Rc::new(RefCell::new(clipped))))
                    }
                    // 张量属性查询（配合护城河 D 内存预估，与 interpreter 同步）
                    "numel" => Ok(Value::Int(tensor.data.len() as i64)),
                    "nbytes" | "bytes" => {
                        let n = tensor.data.len() as i64;
                        let bytes_per_elem: i64 = match &tensor.data {
                            super::tensor::TensorData::F64(_) => 8,
                            super::tensor::TensorData::F32(_) => 4,
                        };
                        Ok(Value::Int(n * bytes_per_elem))
                    }
                    "ndim" | "rank" => Ok(Value::Int(tensor.data.ndim() as i64)),
                    "shape_tensor" => {
                        // 返回 shape 作为 f64 tensor（便于运行时查询）
                        let shape: Vec<f64> = tensor.data.shape().iter().map(|&d| d as f64).collect();
                        let len = shape.len();
                        Ok(Value::Tensor(Rc::new(RefCell::new(
                            Tensor::from_vec(shape, vec![len])
                        ))))
                    }

                    // ── Shape operations ──
                    "reshape" | "view" => {
                        let shape: Vec<usize> = args.iter()
                            .map(|a| a.as_int().unwrap_or(1) as usize)
                            .collect();
                        let result_tensor = tensor.reshape(&shape).ok_or_else(|| {
                            TenthError::RuntimeError { message: format!("无法重塑形状为 {:?}", shape) }
                        })?;
                        let result = Rc::new(RefCell::new(result_tensor));
                        if self.recording { self.record_unary(TapeOp::Reshape, &t, &result); }
                        Ok(Value::Tensor(result))
                    }
                    "flatten" => Ok(Value::Tensor(Rc::new(RefCell::new(tensor.flatten())))),
                    "transpose" => {
                        let result_tensor = tensor.transpose().ok_or_else(|| {
                            TenthError::RuntimeError { message: "转置至少需要 2 个维度".into() }
                        })?;
                        let result = Rc::new(RefCell::new(result_tensor));
                        if self.recording { self.record_unary(TapeOp::Transpose, &t, &result); }
                        Ok(Value::Tensor(result))
                    }
                    "permute" => {
                        let dims: Vec<usize> = args.iter()
                            .map(|a| a.as_int().unwrap_or(0) as usize)
                            .collect();
                        let result = tensor.permute(&dims)
                            .map_err(|msg| TenthError::RuntimeError { message: msg })?;
                        Ok(Value::Tensor(Rc::new(RefCell::new(result))))
                    }
                    "broadcast_to" => {
                        let target_shape: Vec<usize> = args.iter()
                            .map(|a| a.as_int().unwrap_or(1) as usize)
                            .collect();
                        let result = tensor.broadcast_to(&target_shape).ok_or_else(|| {
                            TenthError::RuntimeError { message: format!("无法广播到 {:?}", target_shape) }
                        })?;
                        Ok(Value::Tensor(Rc::new(RefCell::new(result))))
                    }
                    "cat" => {
                        if args.is_empty() {
                            return err("cat() 至少需要 1 个参数 (other, [dim])");
                        }
                        let dim = args.get(1).and_then(|a| a.as_int()).unwrap_or(0) as usize;
                        if let Value::Tensor(other) = &args[0] {
                            let result = tensor.cat(&other.borrow(), dim)
                                .map_err(|msg| TenthError::RuntimeError { message: msg })?;
                            Ok(Value::Tensor(Rc::new(RefCell::new(result))))
                        } else {
                            err("cat() 第一个参数必须是张量")
                        }
                    }
                    "masked_fill" => {
                        if args.len() < 2 {
                            return err("masked_fill() 需要 mask 和 value 参数");
                        }
                        let value = args[1].as_float().unwrap_or(0.0);
                        if let Value::Tensor(mask_rc) = &args[0] {
                            let result_tensor = tensor.masked_fill(&mask_rc.borrow(), value)
                                .map_err(|msg| TenthError::RuntimeError { message: msg })?;
                            let result = Rc::new(RefCell::new(result_tensor));
                            if self.recording {
                                if let Some(ref mut tape) = self.tape {
                                    let input_id = t.borrow().tape_id;
                                    let node_id = tape.masked_fill(input_id, t.clone(), mask_rc.clone(), result.clone());
                                    result.borrow_mut().tape_id = Some(node_id);
                                }
                            }
                            Ok(Value::Tensor(result))
                        } else {
                            err("masked_fill() 的 mask 必须是张量")
                        }
                    }

                    // ── Matrix / NN operations ──
                    "matmul" => {
                        if args.len() != 1 {
                            return err("matmul() 需要 1 个参数");
                        }
                        if let Value::Tensor(other) = &args[0] {
                            let result_tensor = tensor.matmul(&other.borrow())
                                .map_err(|msg| TenthError::RuntimeError { message: msg })?;
                            let result = Rc::new(RefCell::new(result_tensor));
                            if self.recording { self.record_binary(TapeOp::MatMul, &t, &other, &result); }
                            Ok(Value::Tensor(result))
                        } else {
                            err("matmul() 参数必须是张量")
                        }
                    }
                    "bmm" => {
                        // batched matmul: (B, M, K) @ (B, K, N) -> (B, M, N)
                        if args.len() != 1 {
                            return err("bmm() 需要 1 个参数");
                        }
                        if let Value::Tensor(other) = &args[0] {
                            let result_tensor = tensor.bmm(&other.borrow())
                                .map_err(|msg| TenthError::RuntimeError { message: msg })?;
                            let result = Rc::new(RefCell::new(result_tensor));
                            if self.recording { self.record_binary(TapeOp::BatchedMatMul, &t, &other, &result); }
                            Ok(Value::Tensor(result))
                        } else {
                            err("bmm() 参数必须是张量")
                        }
                    }
                    "conv2d" => {
                        // x.conv2d(w, kernel_h, kernel_w, stride, pad)
                        if args.len() < 5 {
                            return err("conv2d() 需要 5 个参数: w, kH, kW, stride, pad");
                        }
                        let k_h = args[1].as_int().unwrap_or(3) as usize;
                        let k_w = args[2].as_int().unwrap_or(3) as usize;
                        let stride = args[3].as_int().unwrap_or(1) as usize;
                        let pad = args[4].as_int().unwrap_or(0) as usize;
                        if let Value::Tensor(w_rc) = &args[0] {
                            let w_data = w_rc.borrow();
                            let w_shape = w_data.shape();
                            // Validate weight shape: must be 4D (C_out, C_in, kH, kW)
                            if w_shape.len() != 4 {
                                return err(&format!(
                                    "conv2d: 权重必须是 4D (C_out, C_in, kH, kW)，得到 {:?}D",
                                    w_shape.len()
                                ));
                            }
                            if w_shape[2] != k_h || w_shape[3] != k_w {
                                return err(&format!(
                                    "conv2d: 权重 kernel 尺寸 {:?} 与参数 kH={}, kW={} 不匹配",
                                    &w_shape[2..4], k_h, k_w
                                ));
                            }
                            let (cols, h_out, w_out) = tensor.im2col(k_h, k_w, stride, pad)
                                .ok_or_else(|| TenthError::RuntimeError {
                                    message: "im2col 失败（输入必须是 4D）".into(),
                                })?;
                            let c_out = w_shape[0];
                            let w_flat = w_data.reshape(&[c_out, w_shape[1] * w_shape[2] * w_shape[3]])
                                .ok_or_else(|| TenthError::RuntimeError {
                                    message: "权重重塑失败".into(),
                                })?;
                            let output_2d = cols.matmul(&w_flat.transpose()
                                .ok_or_else(|| TenthError::RuntimeError {
                                    message: "权重转置失败".into(),
                                })?).map_err(|msg| TenthError::RuntimeError { message: msg })?;
                            let n = tensor.shape()[0];
                            let result_tensor = output_2d.reshape(&[n, c_out, h_out, w_out])
                                .ok_or_else(|| TenthError::RuntimeError {
                                    message: "输出重塑失败".into(),
                                })?;
                            let result = Rc::new(RefCell::new(result_tensor));
                            if self.recording {
                                let cols_rc = Rc::new(RefCell::new(cols));
                                if let Some(ref mut tape) = self.tape {
                                    let x_id = t.borrow().tape_id
                                        .unwrap_or_else(|| tape.input(t.clone()));
                                    let w_id = w_rc.borrow().tape_id
                                        .unwrap_or_else(|| tape.input(w_rc.clone()));
                                    let node_id = tape.conv2d(
                                        x_id, t.clone(),
                                        w_id, w_rc.clone(),
                                        cols_rc, result.clone(),
                                    );
                                    result.borrow_mut().tape_id = Some(node_id);
                                }
                            }
                            Ok(Value::Tensor(result))
                        } else {
                            err("conv2d: 权重必须是张量")
                        }
                    }
                    "batchnorm" => {
                        // x.batchnorm(gamma, beta, eps)
                        if args.len() < 3 {
                            return err("batchnorm() 需要 gamma, beta, eps 参数");
                        }
                        let eps = args[2].as_float().unwrap_or(1e-5);
                        if let (Value::Tensor(gamma_rc), Value::Tensor(beta_rc)) = (&args[0], &args[1]) {
                            use super::tensor::Tensor;
                            let x_shape = tensor.shape();
                            if x_shape.len() < 2 {
                                return err("batchnorm 至少需要 2D 输入");
                            }
                            let c = x_shape[1];
                            let n = x_shape[0];
                            let spatial: usize = x_shape[2..].iter().product();
                            let x_flat = tensor.data.as_standard_layout().to_owned();
                            let x_slice = x_flat.as_slice().unwrap_or(&[]);
                            let gamma_ref = gamma_rc.borrow();
                            let beta_ref = beta_rc.borrow();
                            let g_flat = gamma_ref.data.as_standard_layout().to_owned();
                            let b_flat = beta_ref.data.as_standard_layout().to_owned();
                            let g_slice = g_flat.as_slice().unwrap_or(&[]);
                            let b_slice = b_flat.as_slice().unwrap_or(&[]);
                            let mut result_data = Vec::with_capacity(x_slice.len());
                            let mut x_hat_data = Vec::with_capacity(x_slice.len());
                            let mut std_inv_data = Vec::with_capacity(c);
                            for ci in 0..c {
                                let mut sum = 0.0;
                                let mut count = 0;
                                for ni in 0..n {
                                    for si in 0..spatial {
                                        let idx = ((ni * c + ci) * spatial) + si;
                                        if idx < x_slice.len() { sum += x_slice[idx]; count += 1; }
                                    }
                                }
                                let mean = if count > 0 { sum / count as f64 } else { 0.0 };
                                let mut var_sum = 0.0;
                                for ni in 0..n {
                                    for si in 0..spatial {
                                        let idx = ((ni * c + ci) * spatial) + si;
                                        if idx < x_slice.len() { let d = x_slice[idx] - mean; var_sum += d * d; }
                                    }
                                }
                                let var = if count > 0 { var_sum / count as f64 } else { 1.0 };
                                let std_inv = 1.0 / (var + eps).sqrt();
                                std_inv_data.push(std_inv);
                                let g = g_slice.get(ci).copied().unwrap_or(1.0);
                                let b = b_slice.get(ci).copied().unwrap_or(0.0);
                                for ni in 0..n {
                                    for si in 0..spatial {
                                        let idx = ((ni * c + ci) * spatial) + si;
                                        if idx < x_slice.len() {
                                            let x_hat = (x_slice[idx] - mean) * std_inv;
                                            x_hat_data.push(x_hat);
                                            result_data.push(g * x_hat + b);
                                        }
                                    }
                                }
                            }
                            let result = Rc::new(RefCell::new(Tensor::from_vec(result_data, x_shape.clone())));
                            if self.recording {
                                let x_hat = Rc::new(RefCell::new(Tensor::from_vec(x_hat_data, x_shape.clone())));
                                let std_inv_tensor = Rc::new(RefCell::new(Tensor::from_vec(std_inv_data, vec![c])));
                                if let Some(ref mut tape) = self.tape {
                                    let x_id = t.borrow().tape_id
                                        .unwrap_or_else(|| tape.input(t.clone()));
                                    let node_id = tape.batchnorm(
                                        x_id, t.clone(),
                                        gamma_rc.clone(), beta_rc.clone(),
                                        x_hat, std_inv_tensor, result.clone(),
                                    );
                                    result.borrow_mut().tape_id = Some(node_id);
                                }
                            }
                            Ok(Value::Tensor(result))
                        } else {
                            err("batchnorm: gamma 和 beta 必须是张量")
                        }
                    }
                    "layer_norm" => {
                        // x.layer_norm(gamma, beta, [eps])
                        if args.len() < 2 {
                            return err("layer_norm() 需要 gamma, beta, [eps] 参数");
                        }
                        let eps = args.get(2).and_then(|a| a.as_float()).unwrap_or(1e-5);
                        if let (Value::Tensor(gamma_rc), Value::Tensor(beta_rc)) = (&args[0], &args[1]) {
                            use super::tensor::Tensor;
                            let x_shape = tensor.shape();
                            let ndim = x_shape.len();
                            if ndim == 0 || x_shape[ndim - 1] == 0 {
                                return Ok(Value::Tensor(Rc::new(RefCell::new(tensor.clone()))));
                            }
                            let axis_len = x_shape[ndim - 1];
                            // Validate gamma/beta shapes
                            let g_shape = gamma_rc.borrow().shape();
                            let b_shape = beta_rc.borrow().shape();
                            if g_shape.len() != 1 || g_shape[0] != axis_len {
                                return err(&format!(
                                    "layer_norm: gamma shape {:?} does not match last axis length {}",
                                    g_shape, axis_len
                                ));
                            }
                            if b_shape.len() != 1 || b_shape[0] != axis_len {
                                return err(&format!(
                                    "layer_norm: beta shape {:?} does not match last axis length {}",
                                    b_shape, axis_len
                                ));
                            }
                            let outer_len: usize = x_shape[..ndim - 1].iter().product();
                            let contiguous = tensor.data.as_standard_layout().to_owned();
                            let flat = match contiguous.as_slice() {
                                Some(s) => s.to_vec(),
                                None => tensor.data.iter().collect(),
                            };
                            let gamma_ref = gamma_rc.borrow();
                            let beta_ref = beta_rc.borrow();
                            let g_flat = gamma_ref.data.as_standard_layout().to_owned();
                            let b_flat = beta_ref.data.as_standard_layout().to_owned();
                            let g_slice = g_flat.as_slice().unwrap_or(&[]);
                            let b_slice = b_flat.as_slice().unwrap_or(&[]);
                            let mut result_data = Vec::with_capacity(flat.len());
                            let mut x_hat_data = Vec::with_capacity(flat.len());
                            let mut std_inv_data = Vec::with_capacity(outer_len);
                            for i in 0..outer_len {
                                let start = i * axis_len;
                                let slice = &flat[start..start + axis_len];
                                let mean: f64 = slice.iter().sum::<f64>() / axis_len as f64;
                                let var: f64 = slice.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / axis_len as f64;
                                let std_inv = 1.0 / (var + eps).sqrt();
                                std_inv_data.push(std_inv);
                                for j in 0..axis_len {
                                    let x_hat = (slice[j] - mean) * std_inv;
                                    x_hat_data.push(x_hat);
                                    let g = g_slice[j];
                                    let b = b_slice[j];
                                    result_data.push(g * x_hat + b);
                                }
                            }
                            let result = Rc::new(RefCell::new(Tensor::from_vec(result_data, x_shape.clone())));
                            if self.recording {
                                let x_hat = Rc::new(RefCell::new(Tensor::from_vec(x_hat_data, x_shape.clone())));
                                let std_inv_tensor = Rc::new(RefCell::new(Tensor::from_vec(std_inv_data, vec![outer_len])));
                                if let Some(ref mut tape) = self.tape {
                                    let x_id = t.borrow().tape_id
                                        .unwrap_or_else(|| tape.input(t.clone()));
                                    let node_id = tape.layernorm(
                                        x_id, t.clone(),
                                        gamma_rc.clone(), beta_rc.clone(),
                                        x_hat, std_inv_tensor, result.clone(),
                                    );
                                    result.borrow_mut().tape_id = Some(node_id);
                                }
                            }
                            Ok(Value::Tensor(result))
                        } else {
                            err("layer_norm: gamma 和 beta 必须是张量")
                        }
                    }
                    "dropout" => {
                        if args.is_empty() {
                            return err("dropout() 需要 1 个参数 (rate)");
                        }
                        let rate = args[0].as_float().unwrap_or(0.5);
                        use rand::Rng;
                        let mut rng = rand::thread_rng();
                        let scale = 1.0 / (1.0 - rate);
                        let mask_data = tensor.data.mapv(|_| {
                            if rng.r#gen::<f64>() < rate { 0.0 } else { scale }
                        });
                        let mask = Rc::new(RefCell::new(Tensor::from_data(mask_data)));
                        let result_tensor = Tensor::from_data(&tensor.data * &mask.borrow().data);
                        let result = Rc::new(RefCell::new(result_tensor));
                        if self.recording {
                            if let Some(ref mut tape) = self.tape {
                                let input_id = t.borrow().tape_id
                                    .unwrap_or_else(|| tape.input(t.clone()));
                                let _mask_id = tape.input(mask.clone());
                                let node_id = tape.dropout(input_id, t.clone(), mask.clone(), result.clone());
                                result.borrow_mut().tape_id = Some(node_id);
                            }
                        }
                        Ok(Value::Tensor(result))
                    }

                    _ => err(&format!("张量没有方法 '{}'", method)),
                }
            }
            Value::Struct { name: _, fields } => {
                // Try field-like access (e.g. .len on Vec field)
                for (fname, fval) in fields.borrow().iter() {
                    if fname == method { return Ok(fval.clone()); }
                }
                err(&format!("没有方法 '{}'", method))
            }
            _ => err(&format!("没有方法 '{}'", method)),
        }
    }

    fn set_field(&self, val: &Value, field: &str, new_val: Value) -> TenthResult<()> {
        match val {
            Value::Struct { fields, .. } => {
                for (n, v) in fields.borrow_mut().iter_mut() {
                    if n == field { *v = new_val; return Ok(()); }
                }
                err(&format!("没有字段 '{}'", field))
            }
            Value::Shared(rc) => self.set_field(&rc.borrow(), field, new_val),
            Value::Ref(rc) => self.set_field(&rc.borrow(), field, new_val),
            _ => err("无法设置字段"),
        }
    }

    fn get_field(&self, val: &Value, field: &str) -> TenthResult<Value> {
        let v = match val {
            Value::Ref(rc) => return self.get_field(&rc.borrow(), field),
            Value::MutRef(w) => {
                if let Some(rc) = w.upgrade() { return self.get_field(&rc.borrow(), field); }
                return err("悬垂的 &mut 引用");
            }
            Value::Shared(rc) => return self.get_field(&rc.borrow(), field),
            v => v,
        };
        match v {
            Value::Struct { fields, .. } => {
                for (n, v) in fields.borrow().iter() {
                    if n == field { return Ok(v.clone()); }
                }
                err(&format!("没有字段 '{}'", field))
            }
            Value::Enum { fields, .. } => {
                for (n, v) in fields.borrow().iter() {
                    if n == field { return Ok(v.clone()); }
                }
                err(&format!("没有字段 '{}'", field))
            }
            _ => err(&format!("没有字段 '{}'", field)),
        }
    }

    fn vm_eq(&self, a: &Value, b: &Value) -> bool {
        match (a, b) {
            (Value::Int(x), Value::Int(y)) => x == y,
            (Value::Float(x), Value::Float(y)) => (x - y).abs() < 1e-10,
            (Value::Float32(x), Value::Float32(y)) => (x - y).abs() < 1e-6,
            // f32 与 f64 比较：按 f64 精度判等（f32 提升为 f64 无损）
            (Value::Float32(x), Value::Float(y)) => ((*x as f64) - y).abs() < 1e-10,
            (Value::Float(x), Value::Float32(y)) => (x - (*y as f64)).abs() < 1e-10,
            (Value::Int(x), Value::Float32(y)) => (*x as f32 - y).abs() < 1e-6,
            (Value::Float32(x), Value::Int(y)) => (x - *y as f32).abs() < 1e-6,
            (Value::Int(x), Value::Float(y)) => ((*x as f64) - y).abs() < 1e-10,
            (Value::Float(x), Value::Int(y)) => (x - (*y as f64)).abs() < 1e-10,
            (Value::Bool(x), Value::Bool(y)) => x == y,
            (Value::String(x), Value::String(y)) => x == y,
            (Value::Unit, Value::Unit) => true,
            _ => false,
        }
    }

    // ── Autodiff recording helpers ─────────────────────────────────────

    fn record_unary(&mut self, op: TapeOp, input: &Rc<RefCell<Tensor>>, result: &Rc<RefCell<Tensor>>) {
        if let Some(ref mut tape) = self.tape {
            let node_id = match input.borrow().tape_id {
                Some(input_id) => tape.unary(op, input_id, input.clone(), result.clone()),
                None => {
                    let dummy = tape.input(input.clone());
                    tape.unary(op, dummy, input.clone(), result.clone())
                }
            };
            result.borrow_mut().tape_id = Some(node_id);
        }
    }

    fn record_binary(&mut self, op: TapeOp, t1: &Rc<RefCell<Tensor>>, t2: &Rc<RefCell<Tensor>>, result: &Rc<RefCell<Tensor>>) {
        if let Some(ref mut tape) = self.tape {
            let id1 = t1.borrow().tape_id;
            let id2 = t2.borrow().tape_id;
            let node_id = match (id1, id2) {
                (Some(a), Some(b)) => tape.binary(op, a, b, t1.clone(), t2.clone(), result.clone()),
                (Some(a), None) => {
                    let dummy = tape.input(t2.clone());
                    tape.binary(op, a, dummy, t1.clone(), t2.clone(), result.clone())
                }
                (None, Some(b)) => {
                    let dummy = tape.input(t1.clone());
                    tape.binary(op, dummy, b, t1.clone(), t2.clone(), result.clone())
                }
                (None, None) => tape.binary_direct(op, t1.clone(), t2.clone(), result.clone()),
            };
            result.borrow_mut().tape_id = Some(node_id);
        }
    }
}

fn err<T>(msg: &str) -> TenthResult<T> {
    Err(TenthError::RuntimeError { message: msg.into() })
}