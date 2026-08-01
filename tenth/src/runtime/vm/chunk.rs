//! VM 字节码块：code + strings + 元信息。
//!
//! 从 runtime/vm.rs 拆分而来（T3b 架构重构）。

use super::op::Op;

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
            NewUnion(..) => 56,
            IndexGet => 36, SliceStr => 37,
            MakeEnum(..) => 38, IsEnumVariant(_) => 39, EnumGetField(_) => 40,
            PushRange(..) => 41, MoveOp => 42,
            MakeTensor(..) => 43, MakeClosure(..) => 44,
            PushFloat32(_) => 45,
            IsStruct(_) => 46,
            Await => 47,
            Spawn => 48,
            MakeTuple(_) => 49,
            IsTuple(_) => 50,
            TupleGet(_) => 51,
            Try => 52,
            Yield => 53,
            PushChar(_) => 54,
            TailCall(..) => 55,
        });

        // Emit operands
        macro_rules! w { ($n:expr, $t:ty) => { self.code.extend_from_slice(&($n as $t).to_le_bytes()) } }
        match &op {
            PushInt(n) => w!(*n, i64), PushFloat(f) => w!(*f, f64),
            PushFloat32(f) => w!(*f, f32),
            PushBool(b) => self.code.push(if *b {1} else {0}),
            PushChar(c) => w!(*c, u32),
            PushStr(i) | LoadGlobal(i) | StoreGlobal(i) | Call(i) | LoadField(i) | StoreField(i) => w!(*i, u64),
            CallN(i, n) => { w!(*i, u64); w!(*n, u64); }
            MethodCall(i, n) => { w!(*i, u64); w!(*n, u64); }
            Load(i) | Store(i) => w!(*i, u64),
            Jump(o) | JmpFalse(o) | JmpTrue(o) => w!(*o, i32),
            MakeVec(n) | MakeMap(n) => w!(*n, u64),
            NewStruct(n, f) => { w!(*n, u64); w!(*f, u64); }
            NewUnion(n, f) => { w!(*n, u64); w!(*f, u64); }
            MakeEnum(n, v, f) => { w!(*n, u64); w!(*v, u64); w!(*f, u64); }
            IsEnumVariant(v) => w!(*v, u64),
            EnumGetField(f) => w!(*f, u64),
            IsStruct(n) => w!(*n, u64),
            PushRange(s, e, inc) => { w!(*s, i64); w!(*e, i64); self.code.push(if *inc {1} else {0}); }
            MakeTensor(r, c, d) => { w!(*r, u64); w!(*c, u64); self.code.push(*d); }
            MakeClosure(p, c) => { w!(*p, u64); w!(*c, u64); }
            MakeTuple(n) | IsTuple(n) | TupleGet(n) => w!(*n, u64),
            MoveOp => {}
            TailCall(i, n) => { w!(*i, u64); w!(*n, u64); }
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
            56 => NewUnion(r!(u64) as usize, r!(u64) as usize),
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
            47 => Await,
            48 => Spawn,
            49 => MakeTuple(r!(u64) as usize),
            50 => IsTuple(r!(u64) as usize),
            51 => TupleGet(r!(u64) as usize),
            52 => Try,
            53 => Yield,
            54 => PushChar(r!(u32)),
            55 => TailCall(r!(u64) as usize, r!(u64) as usize),
            _ => panic!("bad opcode {b}"),
        }
    }
}
