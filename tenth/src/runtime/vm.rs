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
    Call(usize), CallN(usize, usize), Ret,
    MakeVec(usize), MakeMap(usize),
    NewStruct(usize, usize), LoadField(usize),
    IndexGet,
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
            Call(_) => 27, CallN(..) => 28, Ret => 29,
            MakeVec(_) => 30, MakeMap(_) => 31,
            NewStruct(..) => 32, LoadField(_) => 33,
            IndexGet => 34,
        });

        // Emit operands
        macro_rules! w { ($n:expr, $t:ty) => { self.code.extend_from_slice(&($n as $t).to_le_bytes()) } }
        match &op {
            PushInt(n) => w!(*n, i64), PushFloat(f) => w!(*f, f64),
            PushBool(b) => self.code.push(if *b {1} else {0}),
            PushStr(i) | LoadGlobal(i) | StoreGlobal(i) | Call(i) | LoadField(i) => w!(*i, u64),
            CallN(i, n) => { w!(*i, u64); w!(*n, u64); }
            Load(i) | Store(i) => w!(*i, u64),
            Jump(o) | JmpFalse(o) | JmpTrue(o) => w!(*o, i32),
            MakeVec(n) | MakeMap(n) => w!(*n, u64),
            NewStruct(n, f) => { w!(*n, u64); w!(*f, u64); }
            _ => {}
        }
    }

    pub fn read_op(&self, ip: &mut usize) -> Op {
        use Op::*;
        let b = self.code[*ip]; *ip += 1;
        macro_rules! r { ($t:ty) => {{ let mut buf = [0u8; std::mem::size_of::<$t>()]; let n = std::mem::size_of::<$t>(); buf.copy_from_slice(&self.code[*ip..*ip+n]); *ip += n; <$t>::from_le_bytes(buf) }}; }
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
            27 => Call(r!(u64) as usize), 28 => CallN(r!(u64) as usize, r!(u64) as usize), 29 => Ret,
            30 => MakeVec(r!(u64) as usize), 31 => MakeMap(r!(u64) as usize),
            32 => NewStruct(r!(u64) as usize, r!(u64) as usize),
            33 => LoadField(r!(u64) as usize),
            34 => IndexGet,
            _ => panic!("bad opcode {b}"),
        }
    }
}

// ── Vm ─────────────────────────────────────────────────────────────────────

struct Frame {
    ip: usize,
    chunk: Chunk,
    locals: Vec<Value>,
    stack_base: usize,
}

pub struct Vm {
    pub functions: HashMap<String, Chunk>,
    pub natives: HashMap<String, NativeFn>,
    globals: HashMap<String, Value>,
    stack: Vec<Value>,
    frames: Vec<Frame>,
}

impl Vm {
    pub fn new() -> Self {
        Vm { functions: HashMap::new(), natives: HashMap::new(), globals: HashMap::new(), stack: Vec::new(), frames: Vec::new() }
    }

    pub fn add_fn(&mut self, name: String, chunk: Chunk) {
        self.functions.insert(name, chunk);
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
        let chunk = self.functions.get(name).cloned()
            .ok_or_else(|| TenthError::RuntimeError { message: format!("VM: undefined '{name}'") })?;
        self.run(chunk)
    }

    fn run(&mut self, mut chunk: Chunk) -> TenthResult<Value> {
        let mut ip: usize = 0;
        let base = self.stack.len();
        let mut locals = vec![Value::Unit; chunk.num_locals.max(chunk.num_args)];

        // Pop args into locals (args were pushed right-to-left, so pop in reverse)
        for i in (0..chunk.num_args).rev() {
            if self.stack.len() > base {
                locals[i] = self.stack.pop().unwrap();
            }
        }

        loop {
            let op = chunk.read_op(&mut ip);
            match op {
                Op::PushInt(n) => self.stack.push(Value::Int(n)),
                Op::PushFloat(f) => self.stack.push(Value::Float(f)),
                Op::PushBool(b) => self.stack.push(Value::Bool(b)),
                Op::PushStr(i) => {
                    let s = chunk.strings.get(i).cloned().unwrap_or_default();
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
                    let name = chunk.strings.get(i).cloned().unwrap_or_default();
                    let v = self.globals.get(&name).cloned().unwrap_or(Value::Unit);
                    self.stack.push(v);
                }
                Op::StoreGlobal(i) => {
                    let name = chunk.strings.get(i).cloned().unwrap_or_default();
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
                Op::Lt => { let (a,b)=self.pop2(); self.stack.push(Value::Bool(self.cmp(&a,&b,|x,y|x<y)?)); }
                Op::Gt => { let (a,b)=self.pop2(); self.stack.push(Value::Bool(self.cmp(&a,&b,|x,y|x>y)?)); }
                Op::Lte => { let (a,b)=self.pop2(); self.stack.push(Value::Bool(self.cmp(&a,&b,|x,y|x<=y)?)); }
                Op::Gte => { let (a,b)=self.pop2(); self.stack.push(Value::Bool(self.cmp(&a,&b,|x,y|x>=y)?)); }

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
                    let name = chunk.strings.get(i).cloned().unwrap_or_default();
                    // Try native first (legacy: uses stack depth as arg count)
                    if let Some(native_fn) = self.natives.get(&name).copied() {
                        let n = self.stack.len() - base;
                        let mut args = vec![Value::Unit; n];
                        for i in (0..n).rev() { args[i] = self.stack.pop().unwrap_or(Value::Unit); }
                        let result = native_fn(self, &args)?;
                        self.stack.push(result);
                    } else if let Some(callee) = self.functions.get(&name).cloned() {
                        // Save current frame ... (existing bytecode logic)
                        self.frames.push(Frame { ip, chunk: chunk.clone(), locals: locals.clone(), stack_base: base });
                        chunk = callee;
                        ip = 0;
                        locals = vec![Value::Unit; chunk.num_locals.max(chunk.num_args)];
                        for i in (0..chunk.num_args).rev() {
                            if self.stack.len() > base { locals[i] = self.stack.pop().unwrap(); }
                        }
                    } else {
                        return Err(TenthError::RuntimeError { message: format!("VM: undefined '{name}'") });
                    }
                }
                Op::CallN(i, num_args) => {
                    let name = chunk.strings.get(i).cloned().unwrap_or_default();
                    // Native call with explicit arg count
                    let n = num_args;
                    let mut args = vec![Value::Unit; n];
                    for i in (0..n).rev() { args[i] = self.stack.pop().unwrap_or(Value::Unit); }
                    if let Some(native_fn) = self.natives.get(&name).copied() {
                        let result = native_fn(self, &args)?;
                        self.stack.push(result);
                    } else if let Some(callee) = self.functions.get(&name).cloned() {
                        // Save current frame
                        self.frames.push(Frame { ip, chunk: chunk.clone(), locals: locals.clone(), stack_base: base });
                        // Switch
                        chunk = callee;
                        ip = 0;
                        locals = vec![Value::Unit; chunk.num_locals.max(chunk.num_args)];
                        for i in (0..chunk.num_args).rev() {
                            if self.stack.len() > base {
                                locals[i] = self.stack.pop().unwrap();
                            }
                        }
                    } else {
                        return Err(TenthError::RuntimeError { message: format!("VM: undefined '{name}'") });
                    }
                }

                Op::Ret => {
                    let result = self.stack.pop().unwrap_or(Value::Unit);
                    self.stack.truncate(base);
                    if let Some(f) = self.frames.pop() {
                        self.stack.push(result);
                        ip = f.ip;
                        chunk = f.chunk;
                        locals = f.locals;
                        // base stays as the original caller's base
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
                    let name = chunk.strings.get(name_i).cloned().unwrap_or_default();
                    let mut fields = Vec::new();
                    for _ in 0..n {
                        let val = self.stack.pop().unwrap_or(Value::Unit);
                        let fname = match self.stack.pop().unwrap_or(Value::Unit) {
                            Value::String(s) => s,
                            _ => String::new(),
                        };
                        fields.push((fname, val));
                    }
                    fields.reverse();
                    self.stack.push(Value::Struct { name, fields: Rc::new(RefCell::new(fields)) });
                }

                Op::LoadField(i) => {
                    let fname = chunk.strings.get(i).cloned().unwrap_or_default();
                    let val = self.stack.pop().unwrap_or(Value::Unit);
                    let v = self.get_field(&val, &fname)?;
                    self.stack.push(v);
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

    fn cmp(&self, a: &Value, b: &Value, f: fn(f64, f64) -> bool) -> TenthResult<bool> {
        Ok(match (a, b) {
            (Value::Int(x), Value::Int(y)) => f(*x as f64, *y as f64),
            (Value::Float(x), Value::Float(y)) => f(*x, *y),
            (Value::Int(x), Value::Float(y)) => f(*x as f64, *y),
            (Value::Float(x), Value::Int(y)) => f(*x, *y as f64),
            _ => return err("cannot compare"),
        })
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