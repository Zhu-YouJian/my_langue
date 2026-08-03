//! VM 字节码块：code + strings + 行号表 + 元信息。
//!
//! 从 runtime/vm.rs 拆分而来（T3b 架构重构）。
//!
//! 2026-08-02（性能优化 R1）：`code`/`strings` 改为 `Rc` 共享——
//! 消除 Op::Call/CallN/TailCall 每次函数调用对整段字节码与字符串表的深拷贝。
//! 编译期通过 `Rc::make_mut` 原地写入（refcount==1 时零拷贝）；运行期只读。
//!
//! 2026-08-02（B 批：VM 报错行号补全）：新增 `lines` 行号表（指令偏移 → 源码行号，
//! 记录于语句/表达式边界）。VM 运行时错误按当前 `ip` 查最近前驱条目定位源码行，
//! 使 `RuntimeError` 不再是无行号的 `line: None`。仅行号（无列号）；运行期只读，
//! 与 code/strings 同模式用 `Rc` 共享。

use std::rc::Rc;
use super::op::Op;

// ── Chunk ──────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct Chunk {
    /// 字节码。Rc 共享：运行期 Call/Ret 只做引用计数 +1，不深拷贝。
    pub code: Rc<Vec<u8>>,
    /// 字符串表。Rc 共享：与 code 同理，消除调用路径深拷贝。
    pub strings: Rc<Vec<String>>,
    /// 行号表：(指令偏移, 源码行号)，按偏移升序。编译期在语句/表达式边界记录；
    /// 运行期报错时按 `ip` 查最近前驱条目得到行号。Rc 共享，与 code/strings 同理。
    pub lines: Rc<Vec<(u32, u32)>>,
    pub num_locals: usize,
    pub num_args: usize,
    /// M2.5-A6：函数标量 ABI 签名（JIT 特化调用用；None = 非特化）。
    /// 由 BytecodeCompiler 编译时从 HIR 签名推导（纯 i64 标量函数）；闭包/
    /// main_expr/globals chunk 保持 None。运行期只读。
    pub scalar_sig: Option<crate::compile::jit::context::ChunkSig>,
}

impl Chunk {
    pub fn new() -> Self {
        Chunk {
            code: Rc::new(vec![]),
            strings: Rc::new(vec![]),
            lines: Rc::new(vec![]),
            num_locals: 0,
            num_args: 0,
            scalar_sig: None,
        }
    }

    /// 记录当前指令偏移对应的源码行号（供运行时错误定位）。
    /// 去重：与最后一条行号相同则跳过（同一行内的连续语句/表达式只记一条）。
    /// 行号为 0 的合成 span 不记录。
    pub fn note_line(&mut self, line: usize) {
        if line == 0 { return; }
        let offset = Rc::make_mut(&mut self.code).len() as u32;
        let lines = Rc::make_mut(&mut self.lines);
        if let Some(&(_, last)) = lines.last() {
            if last == line as u32 { return; }
        }
        lines.push((offset, line as u32));
    }

    /// 按指令偏移查最近前驱行号条目。条目按偏移升序（编译期顺序记录），
    /// 线性扫描即可；仅在运行时错误路径调用，条目数 ≈ 源码行数，开销可忽略。
    pub fn line_at(&self, ip: usize) -> Option<usize> {
        let ip = ip as u32;
        let mut result = None;
        for &(off, line) in self.lines.iter() {
            if off <= ip { result = Some(line as usize); } else { break; }
        }
        result
    }

    pub fn add_string(&mut self, s: &str) -> usize {
        // 编译期唯一持有者（refcount==1），make_mut 原地写零拷贝
        let strings = Rc::make_mut(&mut self.strings);
        if let Some(i) = strings.iter().position(|x| x == s) { return i; }
        let i = strings.len(); strings.push(s.to_string()); i
    }

    pub fn emit(&mut self, op: Op) {
        use Op::*;
        // 编译期唯一持有者（refcount==1），make_mut 原地写零拷贝
        let code = Rc::make_mut(&mut self.code);
        code.push(match &op {
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
            // a1 P1：57/58 追加在尾部（56 = NewUnion），不动既有指令编码
            CallClosure(_) => 57,
            TailCallClosure(_) => 58,
            // AUDIT-11.4.21：引用语义 opcodes 59-62
            MakeRef => 59,
            MakeMutRef(_) => 60,
            Deref => 61,
            DerefStore => 62,
            // M1-S2（true letrec）：自引用 cell opcodes 63/64
            MakeCell => 63,
            BindSelfCapture(_) => 64,
        });

        // Emit operands
        macro_rules! w { ($n:expr, $t:ty) => { code.extend_from_slice(&($n as $t).to_le_bytes()) } }
        match &op {
            PushInt(n) => w!(*n, i64), PushFloat(f) => w!(*f, f64),
            PushFloat32(f) => w!(*f, f32),
            PushBool(b) => code.push(if *b {1} else {0}),
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
            PushRange(s, e, inc) => { w!(*s, i64); w!(*e, i64); code.push(if *inc {1} else {0}); }
            MakeTensor(r, c, d) => { w!(*r, u64); w!(*c, u64); code.push(*d); }
            MakeClosure(p, c, n) => { w!(*p, u64); w!(*c, u64); w!(*n, u64); }
            MakeTuple(n) | IsTuple(n) | TupleGet(n) => w!(*n, u64),
            MoveOp => {}
            TailCall(i, n) => { w!(*i, u64); w!(*n, u64); }
            // a1 P1：单操作数（参数数量）
            CallClosure(n) | TailCallClosure(n) => w!(*n, u64),
            // AUDIT-11.4.21：MakeMutRef 带槽位操作数
            MakeMutRef(i) => w!(*i, u64),
            // M1-S2（true letrec）：BindSelfCapture 带捕获槽位索引
            BindSelfCapture(i) => w!(*i, u64),
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
            44 => MakeClosure(r!(u64) as usize, r!(u64) as usize, r!(u64) as usize),
            45 => PushFloat32(r!(f32)),
            46 => IsStruct(r!(u64) as usize),
            47 => Await,
            48 => Spawn,
            49 => MakeTuple(r!(u64) as usize),
            50 => IsTuple(r!(u64) as usize),
            51 => TupleGet(r!(u64) as usize),
            52 => Try,
            53 => Yield,
            // a1 P1：57/58 追加在尾部（56 = NewUnion）
            57 => CallClosure(r!(u64) as usize),
            58 => TailCallClosure(r!(u64) as usize),
            59 => MakeRef,
            60 => MakeMutRef(r!(u64) as usize),
            // M1-S2（true letrec）：自引用 cell opcodes 63/64
            63 => MakeCell,
            64 => BindSelfCapture(r!(u64) as usize),
            61 => Deref,
            62 => DerefStore,
            54 => PushChar(r!(u32)),
            55 => TailCall(r!(u64) as usize, r!(u64) as usize),
            _ => panic!("bad opcode {b}"),
        }
    }
}
