//! Bytecode VM for Tenth — stack-based virtual machine.
//!
//! Architecture: HIR → compile → Chunk (bytecode) → Vm::run()

use std::collections::HashMap;
use std::rc::Rc;
use std::cell::RefCell;
use crate::error::{TenthError, TenthResult};
use super::value::Value;

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
}

impl Vm {
    pub fn new() -> Self {
        Vm { functions: HashMap::new(), chunks: Vec::new(), chunk_names: Vec::new(), natives: HashMap::new(), globals: HashMap::new(), stack: Vec::new(), frames: Vec::new() }
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
            Err(TenthError::RuntimeError { message: format!("VM: undefined native '{name}'") })
        }
    }

    pub fn call(&mut self, name: &str) -> TenthResult<Value> {
        let idx = self.functions.get(name).copied()
            .ok_or_else(|| TenthError::RuntimeError { message: format!("VM: undefined '{name}'") })?;
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

                Op::Add => { let (a,b)=self.pop2(); self.stack.push(self.add(&a,&b)?); }
                Op::Sub => { let (a,b)=self.pop2(); self.stack.push(self.sub(&a,&b)?); }
                Op::Mul => { let (a,b)=self.pop2(); self.stack.push(self.mul(&a,&b)?); }
                Op::Div => { let (a,b)=self.pop2(); self.stack.push(self.div(&a,&b)?); }
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
                        _ => return err("cannot negate"),
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
                        return Err(TenthError::RuntimeError { message: format!("VM: undefined '{name}'") });
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
                        return Err(TenthError::RuntimeError { message: format!("VM: undefined '{name}'") });
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
                        _ => return err("cannot index"),
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
                                return err("string slice start > end");
                            }
                            let slice: String = chars[si..ei].iter().collect();
                            self.stack.push(Value::String(slice));
                        }
                        _ => return err("SliceStr requires string target"),
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
                    let tensor = if rows == 1 || cols == 1 {
                        Tensor::from_vec(data, vec![total])
                    } else {
                        Tensor::from_vec(data, vec![rows, cols])
                    };
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
            _ => err("expected int"),
        }
    }

    fn add(&self, a: &Value, b: &Value) -> TenthResult<Value> {
        Ok(match (a, b) {
            (Value::Int(x), Value::Int(y)) => Value::Int(x + y),
            (Value::Float(x), Value::Float(y)) => Value::Float(x + y),
            (Value::Int(x), Value::Float(y)) => Value::Float(*x as f64 + y),
            (Value::Float(x), Value::Int(y)) => Value::Float(x + *y as f64),
            (Value::String(x), Value::String(y)) => Value::String(format!("{x}{y}")),
            (Value::Tensor(t), Value::Float(s)) => Value::Tensor(Rc::new(RefCell::new(t.borrow().add_scalar(*s)))),
            (Value::Float(s), Value::Tensor(t)) => Value::Tensor(Rc::new(RefCell::new(t.borrow().add_scalar(*s)))),
            _ => return err("type mismatch in +"),
        })
    }

    fn sub(&self, a: &Value, b: &Value) -> TenthResult<Value> {
        Ok(match (a, b) {
            (Value::Int(x), Value::Int(y)) => Value::Int(x - y),
            (Value::Float(x), Value::Float(y)) => Value::Float(x - y),
            (Value::Int(x), Value::Float(y)) => Value::Float(*x as f64 - y),
            (Value::Float(x), Value::Int(y)) => Value::Float(x - *y as f64),
            _ => return err("type mismatch in -"),
        })
    }

    fn mul(&self, a: &Value, b: &Value) -> TenthResult<Value> {
        Ok(match (a, b) {
            (Value::Int(x), Value::Int(y)) => Value::Int(x * y),
            (Value::Float(x), Value::Float(y)) => Value::Float(x * y),
            (Value::Int(x), Value::Float(y)) => Value::Float(*x as f64 * y),
            (Value::Float(x), Value::Int(y)) => Value::Float(x * *y as f64),
            (Value::Tensor(t), Value::Float(s)) => Value::Tensor(Rc::new(RefCell::new(t.borrow().mul_scalar(*s)))),
            (Value::Float(s), Value::Tensor(t)) => Value::Tensor(Rc::new(RefCell::new(t.borrow().mul_scalar(*s)))),
            _ => return err("type mismatch in *"),
        })
    }

    fn div(&self, a: &Value, b: &Value) -> TenthResult<Value> {
        Ok(match (a, b) {
            (Value::Int(x), Value::Int(y)) => Value::Int(x / y),
            (Value::Float(x), Value::Float(y)) => Value::Float(x / y),
            (Value::Int(x), Value::Float(y)) => Value::Float(*x as f64 / y),
            (Value::Float(x), Value::Int(y)) => Value::Float(x / *y as f64),
            _ => return err("type mismatch in /"),
        })
    }

    fn compare(&self, a: &Value, b: &Value, nf: fn(f64, f64) -> bool, sf: fn(&str, &str) -> bool) -> TenthResult<bool> {
        Ok(match (a, b) {
            (Value::Int(x), Value::Int(y)) => nf(*x as f64, *y as f64),
            (Value::Float(x), Value::Float(y)) => nf(*x, *y),
            (Value::Int(x), Value::Float(y)) => nf(*x as f64, *y),
            (Value::Float(x), Value::Int(y)) => nf(*x, *y as f64),
            (Value::String(x), Value::String(y)) => sf(x, y),
            _ => return err("cannot compare"),
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
                _ => err(&format!("String has no method '{method}'")),
            },
            Value::Vec(items) => match method {
                "len" => Ok(Value::Int(items.borrow().len() as i64)),
                "push" => {
                    if args.len() == 1 {
                        items.borrow_mut().push(args[0].clone());
                        Ok(Value::Unit)
                    } else { err("push takes 1 arg") }
                }
                "get" => {
                    if args.len() == 1 {
                        let idx = args[0].as_int().unwrap_or(0) as usize;
                        Ok(items.borrow().get(idx).cloned().unwrap_or(Value::Unit))
                    } else { err("get takes 1 arg") }
                }
                _ => err(&format!("Vec has no method '{method}'")),
            },
            Value::Map(m) => match method {
                "len" => Ok(Value::Int(m.borrow().len() as i64)),
                _ => err(&format!("Map has no method '{method}'")),
            },
            Value::Tensor(t) => {
                let tensor = t.borrow();
                match method {
                    "sum" => Ok(Value::Float(tensor.sum())),
                    "mean" => Ok(Value::Float(tensor.mean())),
                    "relu" => Ok(Value::Tensor(Rc::new(RefCell::new(tensor.relu())))),
                    "flatten" => Ok(Value::Tensor(Rc::new(RefCell::new(tensor.flatten())))),
                    _ => err(&format!("Tensor has no method '{method}'")),
                }
            }
            Value::Struct { name: _, fields } => {
                // Try field-like access (e.g. .len on Vec field)
                for (fname, fval) in fields.borrow().iter() {
                    if fname == method { return Ok(fval.clone()); }
                }
                err(&format!("no method '{method}'"))
            }
            _ => err(&format!("no method '{method}'")),
        }
    }

    fn set_field(&self, val: &Value, field: &str, new_val: Value) -> TenthResult<()> {
        match val {
            Value::Struct { fields, .. } => {
                for (n, v) in fields.borrow_mut().iter_mut() {
                    if n == field { *v = new_val; return Ok(()); }
                }
                err(&format!("no field '{field}'"))
            }
            Value::Shared(rc) => self.set_field(&rc.borrow(), field, new_val),
            Value::Ref(rc) => self.set_field(&rc.borrow(), field, new_val),
            _ => err("cannot set field"),
        }
    }

    fn get_field(&self, val: &Value, field: &str) -> TenthResult<Value> {
        let v = match val {
            Value::Ref(rc) => return self.get_field(&rc.borrow(), field),
            Value::MutRef(w) => {
                if let Some(rc) = w.upgrade() { return self.get_field(&rc.borrow(), field); }
                return err("dangling &mut");
            }
            Value::Shared(rc) => return self.get_field(&rc.borrow(), field),
            v => v,
        };
        match v {
            Value::Struct { fields, .. } => {
                for (n, v) in fields.borrow().iter() {
                    if n == field { return Ok(v.clone()); }
                }
                err(&format!("no field '{field}'"))
            }
            Value::Enum { fields, .. } => {
                for (n, v) in fields.borrow().iter() {
                    if n == field { return Ok(v.clone()); }
                }
                err(&format!("no field '{field}'"))
            }
            _ => err(&format!("no field '{field}'")),
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
}

fn err<T>(msg: &str) -> TenthResult<T> {
    Err(TenthError::RuntimeError { message: msg.into() })
}