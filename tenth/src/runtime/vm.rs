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
    PushInt(i64), PushFloat(f64), PushBool(bool), PushStr(usize), PushUnit,
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
    PushRange(i64, i64, bool),  // start, end, inclusive
    MoveOp,                     // no-op marker for move semantics
    MakeTensor(usize, usize),   // rows, cols — pops rows*cols f64 values from stack
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
        });

        // Emit operands
        macro_rules! w { ($n:expr, $t:ty) => { self.code.extend_from_slice(&($n as $t).to_le_bytes()) } }
        match &op {
            PushInt(n) => w!(*n, i64), PushFloat(f) => w!(*f, f64),
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
            PushRange(s, e, inc) => { w!(*s, i64); w!(*e, i64); self.code.push(if *inc {1} else {0}); }
            MakeTensor(r, c) => { w!(*r, u64); w!(*c, u64); }
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
            2 => PushBool(self.code[*ip] != 0),
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
            43 => MakeTensor(r!(u64) as usize, r!(u64) as usize),
            44 => MakeClosure(r!(u64) as usize, r!(u64) as usize),
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
}

impl Vm {
    pub fn new() -> Self {
        Vm { functions: HashMap::new(), chunks: Vec::new(), chunk_names: Vec::new(), natives: HashMap::new(), globals: HashMap::new(), stack: Vec::new(), frames: Vec::new(), tape: None, recording: false }
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
        let base = self.stack.len();
        let num_args = self.chunks[chunk_idx].num_args;
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

        loop {
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
                    43 => MakeTensor(r!(u64) as usize, r!(u64) as usize),
                    44 => MakeClosure(r!(u64) as usize, r!(u64) as usize),
                    _ => Ret,
                }
            };
            match op {
                Op::PushInt(n) => self.stack.push(Value::Int(n)),
                Op::PushFloat(f) => self.stack.push(Value::Float(f)),
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

                Op::Add => { let (a,b)=self.pop2(); let r=self.add(&a,&b)?; self.stack.push(r); }
                Op::Sub => { let (a,b)=self.pop2(); let r=self.sub(&a,&b)?; self.stack.push(r); }
                Op::Mul => { let (a,b)=self.pop2(); let r=self.mul(&a,&b)?; self.stack.push(r); }
                Op::Div => { let (a,b)=self.pop2(); let r=self.div(&a,&b)?; self.stack.push(r); }
                Op::Mod => {
                    let b = self.pop_int()?; let a = self.pop_int()?;
                    self.stack.push(Value::Int(a % b));
                }
                Op::Neg => {
                    let v = self.stack.pop().unwrap_or(Value::Unit);
                    self.stack.push(match v {
                        Value::Int(n) => Value::Int(-n),
                        Value::Float(n) => Value::Float(-n),
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
                    let result = self.call_method(&receiver, &name, &args)?;
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

                Op::MakeTensor(rows, cols) => {
                    use super::tensor::Tensor;
                    let total = rows * cols;
                    let mut data = Vec::with_capacity(total);
                    for _ in 0..total {
                        let v = self.stack.pop().unwrap_or(Value::Float(0.0));
                        data.push(match v {
                            Value::Float(f) => f,
                            Value::Int(n) => n as f64,
                            _ => 0.0,
                        });
                    }
                    data.reverse();
                    let tensor = Tensor::from_vec(data, vec![rows, cols]);
                    self.stack.push(Value::Tensor(Rc::new(RefCell::new(tensor))));
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

    fn add(&mut self, a: &Value, b: &Value) -> TenthResult<Value> {
        Ok(match (a, b) {
            (Value::Int(x), Value::Int(y)) => Value::Int(x + y),
            (Value::Float(x), Value::Float(y)) => Value::Float(x + y),
            (Value::Int(x), Value::Float(y)) => Value::Float(*x as f64 + y),
            (Value::Float(x), Value::Int(y)) => Value::Float(x + *y as f64),
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

    fn sub(&mut self, a: &Value, b: &Value) -> TenthResult<Value> {
        Ok(match (a, b) {
            (Value::Int(x), Value::Int(y)) => Value::Int(x - y),
            (Value::Float(x), Value::Float(y)) => Value::Float(x - y),
            (Value::Int(x), Value::Float(y)) => Value::Float(*x as f64 - y),
            (Value::Float(x), Value::Int(y)) => Value::Float(x - *y as f64),
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

    fn mul(&mut self, a: &Value, b: &Value) -> TenthResult<Value> {
        Ok(match (a, b) {
            (Value::Int(x), Value::Int(y)) => Value::Int(x * y),
            (Value::Float(x), Value::Float(y)) => Value::Float(x * y),
            (Value::Int(x), Value::Float(y)) => Value::Float(*x as f64 * y),
            (Value::Float(x), Value::Int(y)) => Value::Float(x * *y as f64),
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

    fn div(&mut self, a: &Value, b: &Value) -> TenthResult<Value> {
        Ok(match (a, b) {
            (Value::Int(x), Value::Int(y)) => Value::Int(x / y),
            (Value::Float(x), Value::Float(y)) => Value::Float(x / y),
            (Value::Int(x), Value::Float(y)) => Value::Float(*x as f64 / y),
            (Value::Float(x), Value::Int(y)) => Value::Float(x / *y as f64),
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
            (Value::String(x), Value::String(y)) => sf(x, y),
            _ => return err("无法比较"),
        })
    }

    fn call_method(&mut self, receiver: &Value, method: &str, args: &[Value]) -> TenthResult<Value> {
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
                _ => err(&format!("Vec 没有方法 '{}'", method)),
            },
            Value::Map(m) => match method {
                "len" => Ok(Value::Int(m.borrow().len() as i64)),
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
                            } else {
                                Ok(Value::Float(tensor.sum()))
                            }
                        } else {
                            let axis = args[0].as_int().unwrap_or(0) as usize;
                            Ok(Value::Tensor(Rc::new(RefCell::new(tensor.sum_axis(axis)))))
                        }
                    }
                    "mean" => {
                        if self.recording {
                            let scalar = tensor.mean();
                            let result = Rc::new(RefCell::new(Tensor::from_vec(vec![scalar], vec![1])));
                            self.record_unary(TapeOp::Mean, &t, &result);
                            Ok(Value::Tensor(result))
                        } else {
                            Ok(Value::Float(tensor.mean()))
                        }
                    }
                    "max_val" => Ok(Value::Float(tensor.max_val())),

                    // ── Elementwise unary ──
                    "abs" => {
                        let result = Rc::new(RefCell::new(tensor.abs()));
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

                    // ── Shape operations ──
                    "reshape" | "view" => {
                        let shape: Vec<usize> = args.iter()
                            .map(|a| a.as_int().unwrap_or(1) as usize)
                            .collect();
                        let result = tensor.reshape(&shape).ok_or_else(|| {
                            TenthError::RuntimeError { message: format!("无法重塑形状为 {:?}", shape) }
                        })?;
                        Ok(Value::Tensor(Rc::new(RefCell::new(result))))
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
                            let result = tensor.masked_fill(&mask_rc.borrow(), value)
                                .map_err(|msg| TenthError::RuntimeError { message: msg })?;
                            Ok(Value::Tensor(Rc::new(RefCell::new(result))))
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
                            let (cols, h_out, w_out) = tensor.im2col(k_h, k_w, stride, pad)
                                .ok_or_else(|| TenthError::RuntimeError {
                                    message: "im2col 失败（输入必须是 4D）".into(),
                                })?;
                            let w_shape = w_data.shape();
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
                            let outer_len: usize = x_shape[..ndim - 1].iter().product();
                            let contiguous = tensor.data.as_standard_layout().to_owned();
                            let flat = match contiguous.as_slice() {
                                Some(s) => s.to_vec(),
                                None => tensor.data.iter().cloned().collect(),
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
                                    let g = g_slice.get(j).copied().unwrap_or(1.0);
                                    let b = b_slice.get(j).copied().unwrap_or(0.0);
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