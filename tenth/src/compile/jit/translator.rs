//! Bytecode → Cranelift IR translator (stack-area design).
//!
//! The virtual stack is a single large `StackSlot` (an area of
//! `MAX_STACK_DEPTH * VALUE_SIZE` bytes). A compile-time stack pointer
//! `sp` tracks the current top. Push increments `sp`; pop decrements it.
//! Both branches of an if/else write to the same memory offsets, so
//! control-flow merges need no phi nodes — the correct value is already
//! in memory at runtime.
//!
//! Locals are individual `StackSlot`s (they don't have merge issues).
//!
//! Generated function signature:
//! `extern "C" fn(vm: *mut u8, args: *const u8, n: usize, out: *mut u8) -> bool`

use cranelift::prelude::*;
use crate::hir::types::BaseType;
use cranelift_module::{Linkage, Module};
use std::collections::{HashMap, HashSet, VecDeque};
use std::mem::size_of;

use crate::runtime::vm::{Chunk, Op, Vm};
use crate::runtime::value::Value;

// Cranelift's `StackSlot` and `SigRef` entity types live in `codegen::ir`
// and are NOT re-exported from the prelude. Bring them in explicitly.
use cranelift::codegen::ir::{StackSlot, SigRef};

/// Size of one `Value` on the stack.
const VALUE_SIZE: u32 = size_of::<Value>() as u32;

/// A2b：标量专用化的种类。`Value` 是普通枚举（无稳定布局），原生代码无法直接
/// 读写 `Value::Int` 的载荷——用**伴随标量槽**（原生 i64/f64/i8 栈槽）绕开：
/// 已知为标量的值同时维护「Value 槽」（延迟物化）与「标量槽」（原生快路径）。
///
/// 安全性（静默错值红线）：
/// - 局部变量的 Value 槽**始终有效**（Store 双写：标量槽 + host_make_* 写 Value 槽），
///   任何通用消费者读到正确 Value；标量槽是附加快路径，读它仅当分析证明该局部
///   恒为标量（must 分析，跨块 GFP）。
/// - 栈值**延迟物化**：专用化算子只写标量槽；Value 槽在通用消费者/块边界前物化。
///
/// I32 = `Value::Int(i64, I32)`——JIT 可达的 Int 链均为 I32（PushInt→host_make_int
/// 固定 I32；算术结果 dtype = 第一操作数 dtype = I32；native 返回值为 Unknown 不参与）。
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum ScalarKind {
    Unknown,
    I32,
    F64,
    Bool,
    /// 分析用的乐观顶元素（meet(Top, x) = x），仅分析期存在，发射期视作 Unknown。
    Top,
}

/// I32 范围检查边界（`check_int_overflow` 对 I32 的语义）。
const I32_MIN: i64 = -2147483648;
const I32_MAX: i64 = 2147483647;

/// A2b：原生 I32 运算的错误种类（错误消息与 VM 逐字一致）。
#[derive(Clone, Copy)]
enum NativeErr {
    /// 整数运算结果溢出 i32 范围（i64 层 checked_* 失败）
    Overflow,
    /// 整数运算结果 {r} 溢出 i32 范围（窄 dtype 范围检查失败）
    Range,
    /// 整数除零
    DivZero,
    /// 整数取模除零
    ModZero,
}

/// Maximum virtual-stack depth (number of Values). Functions exceeding
/// this will need a larger area.
///
/// A1 调为 64（原 256）：每 JIT 帧的虚拟栈槽 = VALUE_SIZE × 该值（VALUE_SIZE
/// = size_of::<Value>() = 112 字节，256 时每帧 28KB）。JIT-to-JIT 直接调用后
/// 递归在原生栈上展开（不再逃逸解释器的堆栈），28KB/帧 + Cranelift 无 stack
/// probe → 跨 guard page 直接 0xC0000005（原生栈溢出）。64 = 7KB/帧，fib(28)
/// 约 196KB，普通 1MB 栈即可承载；需要更深的虚拟栈（罕见：大量参数/深层嵌套
/// 表达式/大张量字面量）的函数经 bump_sp 报错优雅回退解释器（既有机制）。
const MAX_STACK_DEPTH: u32 = 64;

/// A2：内联资格——被调函数最大指令数（保守）。超过则不内联（静默回退 A1 直接调用）。
/// 该值同时作为内联栈槽数的上界（每个 push 类指令至多使栈深 +1）。
const INLINE_MAX_INSTR: usize = 16;

pub fn translate<M: Module>(
    module: &mut M,
    chunk_idx: usize,
    chunk: &Chunk,
    name_to_chunk: &[Option<usize>],
    all_chunks: &[Chunk],
) -> Result<cranelift_module::FuncId, String> {
    let mut ctx = module.make_context();
    let mut fn_ctx = FunctionBuilderContext::new();

    let ptr = module.target_config().pointer_type();
    ctx.func.signature.params.push(AbiParam::new(ptr)); // vm
    ctx.func.signature.params.push(AbiParam::new(ptr)); // args
    ctx.func.signature.params.push(AbiParam::new(ptr)); // n (usize, pointer-sized)
    ctx.func.signature.params.push(AbiParam::new(ptr)); // out
    ctx.func.signature.returns.push(AbiParam::new(types::I8)); // bool

    let func_id = module.declare_function(
        &format!("__tenth_jit_chunk_{}", chunk_idx),
        Linkage::Local,
        &ctx.func.signature,
    ).map_err(|e| format!("declare_function: {e}"))?;

    {
        let mut builder = FunctionBuilder::new(&mut ctx.func, &mut fn_ctx);
        let entry = builder.create_block();
        builder.append_block_params_for_function_params(entry);
        builder.switch_to_block(entry);
        builder.seal_block(entry);

        let vm_param = builder.block_params(entry)[0];
        let args_ptr = builder.block_params(entry)[1];
        let _args_n = builder.block_params(entry)[2];
        let out_ptr = builder.block_params(entry)[3];

        let stack_slot = builder.create_sized_stack_slot(StackSlotData::new(
            StackSlotKind::ExplicitSlot,
            VALUE_SIZE * MAX_STACK_DEPTH,
            8,
        ));

        let mut t = Translator {
            module,
            builder,
            vm: vm_param,
            chunk,
            name_to_chunk,
            all_chunks,
            sp: 0,
            stack_slot,
            locals: HashMap::new(),
            blocks: HashMap::new(),
            block_sp: HashMap::new(),
            visited: HashSet::new(),
            args_ptr,
            out_ptr,
            ptr,
            terminated: false,
            cur_line: 0,
            inline_out: None,
            inline_cont: None,
            scalar_enabled: false,
            stack_scalars: HashMap::new(),
            local_scalars: HashMap::new(),
            block_entry_kinds: HashMap::new(),
            cur_local_kinds: Vec::new(),
        };
        t.translate_body()?;
        // translate_body calls builder.finalize() internally (consuming it).
    }

    module.define_function(func_id, &mut ctx).map_err(|e| format!("define_function: {e}"))?;
    ctx.clear();
    Ok(func_id)
}

struct Translator<'a, M: Module> {
    module: &'a mut M,
    builder: FunctionBuilder<'a>,
    vm: Value_,
    chunk: &'a Chunk,
    /// A1：字符串表索引 → 函数 chunk 索引（本 chunk 内的直接调用目标；None = 不可直接调用）。
    name_to_chunk: &'a [Option<usize>],
    /// A2：全部 chunk（调用点内联时读取被调函数字节码）。chunk 索引与
    /// `name_to_chunk` 解析出的索引对齐（即 `Vm.chunks` 索引，`functions` 映射一致）。
    all_chunks: &'a [Chunk],
    /// Compile-time stack pointer (byte offset into `stack_slot`).
    sp: i32,
    /// Virtual stack area — one large slot holding up to MAX_STACK_DEPTH Values.
    stack_slot: StackSlot,
    /// Local variables — individual slots (no merge issues).
    locals: HashMap<usize, StackSlot>,
    /// Bytecode IP → Cranelift Block.
    blocks: HashMap<usize, Block>,
    /// Block → sp at entry (recorded at every jump/branch).
    block_sp: HashMap<Block, i32>,
    /// Blocks that have been emitted into (to detect unfilled merge blocks).
    visited: HashSet<Block>,
    args_ptr: Value_,
    out_ptr: Value_,
    ptr: types::Type,
    /// Whether the current block already has a terminator.
    terminated: bool,
    /// 9c：当前 opcode 的源码行号（0 = 无）。在 `emit_op` 入口从 chunk 行号表
    /// `line_at(op_start)` 查得；每个 hostcall 前由 `emit_line_hint` 写入
    /// `vm.current_line`，供 hostcall 报错时携带行号（对齐 VM err_here/with_line）。
    cur_line: usize,
    /// A2：内联模式（Some 时）——`Ret` 写 `inline_out` 槽并跳 `inline_cont` 汇合块，
    /// 而非返回整个 JIT 函数。仅在调用点内联被调函数主体期间非 None。
    inline_out: Option<Value_>,
    inline_cont: Option<Block>,
    /// A2b：是否启用标量专用化（分析全部成功时 true）。false → 全函数走既有通用路径。
    scalar_enabled: bool,
    /// A2b：栈偏移 → (标量种类, 伴随槽)。伴随槽存裸 i64/f64/i8（8 字节）。
    /// 通用 hostcall 后全部清除（保守）；专用化算子按需重建。
    stack_scalars: HashMap<i32, (ScalarKind, StackSlot)>,
    /// A2b：局部变量索引 → (标量种类, 伴随槽)。仅当分析证明该局部恒为标量。
    local_scalars: HashMap<usize, (ScalarKind, StackSlot)>,
    /// A2b：分析结果——leader IP → 块入口处各局部种类（按局部索引）。
    block_entry_kinds: HashMap<usize, Vec<ScalarKind>>,
    /// A2b：当前块内的局部种类（块入口设为该块 entry kinds，块内 Store 演化）。
    cur_local_kinds: Vec<ScalarKind>,
}

// Cranelift re-exports — `Value` clashes with our runtime `Value`, so alias.
type Value_ = cranelift::prelude::Value;

impl<'a, M: Module> Translator<'a, M> {
    fn translate_body(mut self) -> Result<(), String> {
        // ── Create blocks for all leaders ──────────────────────────────────
        let leaders = self.find_leaders();
        for &ip in &leaders {
            let blk = self.builder.create_block();
            self.blocks.insert(ip, blk);
        }

        // ── Initialise locals: copy args from args_ptr ─────────────────────
        let num_args = self.chunk.num_args;
        let num_locals = self.chunk.num_locals.max(num_args);
        for i in 0..num_locals {
            let slot = self.builder.create_sized_stack_slot(StackSlotData::new(
                StackSlotKind::ExplicitSlot,
                VALUE_SIZE,
                8,
            ));
            self.locals.insert(i, slot);
        }
        for i in 0..num_args {
            let dst = self.locals[&i];
            let src_off = (i as i32) * (VALUE_SIZE as i32);
            self.copy_ptr_to_slot(self.args_ptr, src_off, dst, 0);
        }

        // ── A2b：局部变量标量种类分析（跨块 must 分析）──────────────
        // 分析成功 → 启用标量专用化（Load/Store/算术/比较走原生路径）；
        // 失败（含分析未知 opcode）→ 全函数保持既有通用路径（零行为变化）。
        if let Some(kinds) = self.analyze_scalar_kinds() {
            self.scalar_enabled = true;
            self.block_entry_kinds = kinds;
            let n = self.chunk.num_locals.max(self.chunk.num_args);
            self.cur_local_kinds = vec![ScalarKind::Unknown; n];
        }

        // ── Emit code for each instruction ─────────────────────────────────
        let mut ip = 0usize;
        let code_len = self.chunk.code.len();

        while ip < code_len {
            if let Some(&blk) = self.blocks.get(&ip) {
                if !self.terminated {
                    // Fall-through: record sp and jump.
                    // A2b：顺序落入也是块边界——先物化栈标量（否则标量 Value 槽
                    // 陈旧，目标块入口清跟踪后读陈旧 Value → 静默错值）。
                    self.materialize_all_stack();
                    self.block_sp.insert(blk, self.sp);
                    self.builder.ins().jump(blk, &[]);
                }
                self.builder.switch_to_block(blk);
                self.builder.seal_block(blk);
                self.visited.insert(blk);
                if let Some(&sp) = self.block_sp.get(&blk) {
                    self.sp = sp;
                }
                // A2b：块入口——清栈标量跟踪（入口栈值视为 Unknown，Value 槽由
                // 前驱终止器物化保证有效）；局部标量槽保留（分析证明跨块成立），
                // 但种类重设为该块入口分析值。
                self.stack_scalars.clear();
                if let Some(entry) = self.block_entry_kinds.get(&ip) {
                    let n = self.chunk.num_locals.max(self.chunk.num_args);
                    self.cur_local_kinds = vec![ScalarKind::Unknown; n];
                    for (i, &k) in entry.iter().enumerate() {
                        if let Some(slot) = self.cur_local_kinds.get_mut(i) {
                            *slot = if k == ScalarKind::Top { ScalarKind::Unknown } else { k };
                        }
                    }
                } else {
                    let n = self.chunk.num_locals.max(self.chunk.num_args);
                    self.cur_local_kinds = vec![ScalarKind::Unknown; n];
                }
                self.terminated = false;
            }

            let op_start = ip;
            let op = self.chunk.read_op(&mut ip);
            self.emit_op(op, op_start, ip)?;
        }

        // ── End of function: return vstack top (or Unit) ───────────────────
        if !self.terminated {
            self.emit_return();
        }

        // ── Fill any unfilled merge blocks (e.g. end-of-if/else labels) ────
        let all_blocks: Vec<(usize, Block)> = self.blocks.iter().map(|(&k, &v)| (k, v)).collect();
        for (_ip, blk) in all_blocks {
            if !self.visited.contains(&blk) {
                self.sp = self.block_sp.get(&blk).copied().unwrap_or(0);
                self.builder.switch_to_block(blk);
                self.builder.seal_block(blk);
                self.visited.insert(blk);
                self.emit_return();
            }
        }

        self.builder.finalize();
        Ok(())
    }

    fn find_leaders(&self) -> Vec<usize> {
        let mut leaders = vec![0usize];
        let mut ip = 0usize;
        let len = self.chunk.code.len();
        while ip < len {
            let _start = ip;
            let op = self.chunk.read_op(&mut ip);
            if let Op::Jump(o) | Op::JmpFalse(o) | Op::JmpTrue(o) = op {
                let target = ((ip as i64) + (o as i64)) as usize;
                leaders.push(target);
                leaders.push(ip); // instruction after the jump is a leader
            }
            if matches!(op, Op::Ret) {
                leaders.push(ip);
            }
        }
        leaders.sort();
        leaders.dedup();
        leaders
    }

    // ── A2b：标量专用化——跨块 must 分析 ──────────────────────────────────

    /// A2b：跨块 must 分析——每个块入口处各局部变量的标量种类。
    /// 返回 leader IP → 入口种类（按局部索引）；`None` = 函数含分析未知 opcode
    /// （或块图不完整）→ 禁用专用化（全函数走既有通用路径，零行为变化）。
    ///
    /// 正确性（静默错值红线）：种类 I32/F64/Bool 意味着「该局部在所有到达该点的
    /// 路径上恒为该标量」——块 0（函数入口，含参数）强制 Unknown；其余块从乐观
    /// 顶 Top 出发，经 meet（跨前驱求交）收敛到最大不动点（GFP）。栈值在块内按
    /// 确定性模拟；通用 opcode 清空栈标量信息（与发射端 `call_hostcall_*` 清栈
    /// 一致）；块入口的栈值不可知（视 Unknown，安全保守）。
    fn analyze_scalar_kinds(&self) -> Option<HashMap<usize, Vec<ScalarKind>>> {
        use ScalarKind::*;
        use Op::*;
        let chunk = self.chunk;
        let num_locals = chunk.num_locals.max(chunk.num_args).max(1);

        // 解码全部指令：(ip, op, next_ip)
        let mut insns: Vec<(usize, Op, usize)> = Vec::new();
        let mut ip = 0usize;
        while ip < chunk.code.len() {
            let start = ip;
            let op = chunk.read_op(&mut ip);
            insns.push((start, op, ip));
        }
        if insns.is_empty() {
            return None;
        }

        // leaders（与 find_leaders 一致）
        let mut leaders = vec![0usize];
        for &(ip, ref op, next) in &insns {
            if let Jump(o) | JmpFalse(o) | JmpTrue(o) = op {
                let target = (next as i64 + *o as i64) as usize;
                if target >= chunk.code.len() {
                    return None;
                }
                leaders.push(target);
                leaders.push(next);
            }
            if matches!(op, Ret) {
                leaders.push(next);
            }
        }
        // 仅保留「指令起点」的 leader：Ret 后的 `next` 可能为 code.len()（函数末尾），
        // 非指令起点会形成空块导致越界；跳转目标在良构字节码中必为指令起点。
        let insn_starts: HashSet<usize> = insns.iter().map(|(s, _, _)| *s).collect();
        leaders.retain(|l| *l < chunk.code.len() && insn_starts.contains(l));
        leaders.sort();
        leaders.dedup();
        let mut leader_index: HashMap<usize, usize> = HashMap::new();
        for (i, &l) in leaders.iter().enumerate() {
            leader_index.insert(l, i);
        }
        let nblocks = leaders.len();

        // 块内指令首索引
        let mut block_first: Vec<usize> = vec![usize::MAX; nblocks];
        for (idx, &(start, _, _)) in insns.iter().enumerate() {
            if let Some(&bi) = leader_index.get(&start) {
                if block_first[bi] == usize::MAX {
                    block_first[bi] = idx;
                }
            }
        }
        // 后继（end = 下一非空块的起始，防御空块）
        let mut succs: Vec<Vec<usize>> = vec![Vec::new(); nblocks];
        for bi in 0..nblocks {
            if block_first[bi] == usize::MAX {
                continue;
            }
            let mut end = insns.len();
            let mut j = bi + 1;
            while j < nblocks && block_first[j] == usize::MAX {
                j += 1;
            }
            if j < nblocks {
                end = block_first[j];
            }
            let last_idx = end - 1;
            let (_, ref op, next) = insns[last_idx];
            match op {
                Jump(o) => {
                    let t = (next as i64 + *o as i64) as usize;
                    if let Some(&tbi) = leader_index.get(&t) {
                        succs[bi].push(tbi);
                    }
                }
                JmpFalse(o) | JmpTrue(o) => {
                    let t = (next as i64 + *o as i64) as usize;
                    if let Some(&tbi) = leader_index.get(&t) {
                        succs[bi].push(tbi);
                    }
                    if let Some(&nbi) = leader_index.get(&next) {
                        succs[bi].push(nbi);
                    }
                }
                Ret => {}
                _ => {
                    if let Some(&nbi) = leader_index.get(&next) {
                        succs[bi].push(nbi);
                    }
                }
            }
        }
        for s in succs.iter_mut() {
            s.sort();
            s.dedup();
        }

        // 转移函数：模拟块 bi（入口种类 → 出口种类）。None = 分析未知 opcode。
        fn transfer(
            insns: &[(usize, Op, usize)],
            block_first: &[usize],
            nblocks: usize,
            bi: usize,
            entry: &[ScalarKind],
        ) -> Option<Vec<ScalarKind>> {
            use ScalarKind::*;
            use Op::*;
            let mut locals = entry.to_vec();
            let mut stack: Vec<ScalarKind> = Vec::new();
            let end = if bi + 1 < nblocks { block_first[bi + 1] } else { insns.len() };
            let start = block_first[bi];
            for idx in start..end {
                let (_, ref op, _) = insns[idx];
                match op {
                    PushInt(_) => stack.push(I32),
                    PushFloat(_) => stack.push(F64),
                    PushBool(_) => stack.push(Bool),
                    PushChar(_) => stack.push(I32),
                    // PushStr/PushUnit/PushFloat32 走通用 hostcall（发射端失效栈标量）
                    PushUnit | PushStr(_) | PushFloat32(_) => {
                        stack.push(Unknown);
                        clear_stack(&mut stack);
                    }
                    Pop => {
                        stack.pop();
                    }
                    Dup => {
                        let k = stack.last().copied().unwrap_or(Unknown);
                        stack.push(k);
                    }
                    Load(i) => {
                        let k = locals.get(*i).copied().unwrap_or(Unknown);
                        stack.push(k);
                    }
                    Store(i) => {
                        let k = stack.pop().unwrap_or(Unknown);
                        if *i >= locals.len() {
                            locals.resize(*i + 1, Unknown);
                        }
                        locals[*i] = k;
                    }
                    Add | Sub | Mul | Div | Mod => {
                        let b = stack.pop().unwrap_or(Unknown);
                        let a = stack.pop().unwrap_or(Unknown);
                        stack.push(match (a, b) {
                            (I32, I32) => I32,
                            (F64, F64) => F64,
                            _ => Unknown,
                        });
                    }
                    Neg => {
                        let a = stack.pop().unwrap_or(Unknown);
                        stack.push(match a {
                            I32 => I32,
                            F64 => F64,
                            _ => Unknown,
                        });
                    }
                    Not => {
                        let a = stack.pop().unwrap_or(Unknown);
                        stack.push(match a {
                            Bool => Bool,
                            _ => Unknown,
                        });
                    }
                    Eq | Neq | Lt | Gt | Lte | Gte => {
                        stack.pop();
                        stack.pop();
                        stack.push(Bool);
                    }
                    Jump(_) => {}
                    JmpFalse(_) | JmpTrue(_) => {
                        stack.pop();
                    }
                    MoveOp => {}
                    // 通用 opcode：清空栈标量信息（与发射端 call_hostcall_* 清栈一致）
                    LoadGlobal(_) => {
                        stack.push(Unknown);
                        clear_stack(&mut stack);
                    }
                    StoreGlobal(_) => {
                        // M2-A3：与发射端对齐——消费栈值（VM opcode 10 语义，不推回）。
                        stack.pop();
                        clear_stack(&mut stack);
                    }
                    Call(_) => {
                        stack.push(Unknown);
                        clear_stack(&mut stack);
                    }
                    CallN(_, n) => {
                        for _ in 0..*n {
                            stack.pop();
                        }
                        stack.push(Unknown);
                        clear_stack(&mut stack);
                    }
                    MethodCall(_, n) => {
                        for _ in 0..(*n + 1) {
                            stack.pop();
                        }
                        stack.push(Unknown);
                        clear_stack(&mut stack);
                    }
                    Ret => {
                        stack.pop();
                    }
                    MakeVec(n) => {
                        for _ in 0..*n {
                            stack.pop();
                        }
                        stack.push(Unknown);
                        clear_stack(&mut stack);
                    }
                    MakeMap(n) => {
                        for _ in 0..(*n * 2) {
                            stack.pop();
                        }
                        stack.push(Unknown);
                        clear_stack(&mut stack);
                    }
                    NewStruct(_, f) => {
                        for _ in 0..(*f * 2) {
                            stack.pop();
                        }
                        stack.push(Unknown);
                        clear_stack(&mut stack);
                    }
                    NewUnion(_, _) => {
                        stack.pop();
                        stack.push(Unknown);
                        clear_stack(&mut stack);
                    }
                    LoadField(_) => {
                        stack.pop();
                        stack.push(Unknown);
                        clear_stack(&mut stack);
                    }
                    StoreField(_) => {
                        stack.pop();
                        stack.pop();
                        stack.push(Unknown);
                        clear_stack(&mut stack);
                    }
                    IndexGet => {
                        stack.pop();
                        stack.pop();
                        stack.push(Unknown);
                        clear_stack(&mut stack);
                    }
                    SliceStr => {
                        stack.pop();
                        stack.pop();
                        stack.pop();
                        stack.push(Unknown);
                        clear_stack(&mut stack);
                    }
                    MakeEnum(_, _, f) => {
                        for _ in 0..(*f * 2) {
                            stack.pop();
                        }
                        stack.push(Unknown);
                        clear_stack(&mut stack);
                    }
                    IsEnumVariant(_) => {
                        stack.pop();
                        stack.push(Unknown);
                        clear_stack(&mut stack);
                    }
                    EnumGetField(_) => {
                        stack.pop();
                        stack.push(Unknown);
                        clear_stack(&mut stack);
                    }
                    PushRange(_, _, _) => {
                        stack.push(Unknown);
                        clear_stack(&mut stack);
                    }
                    MakeTensor(r, c, _) => {
                        for _ in 0..(*r * *c) {
                            stack.pop();
                        }
                        stack.push(Unknown);
                        clear_stack(&mut stack);
                    }
                    MakeClosure(_, c, _) => {
                        for _ in 0..*c {
                            stack.pop();
                        }
                        stack.push(Unknown);
                        clear_stack(&mut stack);
                    }
                    // M2-A3：新覆盖 opcode——pop 对应槽 + push Unknown + 清栈
                    // （与发射端 hostcall 的 invalidate_stack_scalars 清栈一致）。
                    MakeTuple(n) => {
                        for _ in 0..*n {
                            stack.pop();
                        }
                        stack.push(Unknown);
                        clear_stack(&mut stack);
                    }
                    IsTuple(_) => {
                        stack.pop();
                        stack.push(Unknown);
                        clear_stack(&mut stack);
                    }
                    TupleGet(_) => {
                        stack.pop();
                        stack.push(Unknown);
                        clear_stack(&mut stack);
                    }
                    IsStruct(_) => {
                        stack.pop();
                        stack.push(Unknown);
                        clear_stack(&mut stack);
                    }
                    Try => {
                        // 弹值；Err 早退（终止）或 Ok 解包/非 Result 透传——结果种类不可知。
                        stack.pop();
                        stack.push(Unknown);
                        clear_stack(&mut stack);
                    }
                    Spawn => {
                        stack.pop();
                        stack.push(Unknown);
                        clear_stack(&mut stack);
                    }
                    TailCall(_, n) => {
                        for _ in 0..*n {
                            stack.pop();
                        }
                        stack.push(Unknown);
                        clear_stack(&mut stack);
                    }
                    TailCallClosure(n) => {
                        for _ in 0..(*n + 1) {
                            stack.pop();
                        }
                        stack.push(Unknown);
                        clear_stack(&mut stack);
                    }
                    // 不可 JIT / 分析未知 → 禁用专用化
                    Await | Yield | CallClosure(_)
                    | MakeRef | MakeMutRef(_) | Deref | DerefStore | MakeCell
                    | BindSelfCapture(_) => {
                        return None;
                    }
                }
            }
            Some(locals)
        }

        fn clear_stack(stack: &mut Vec<ScalarKind>) {
            for k in stack.iter_mut() {
                *k = ScalarKind::Unknown;
            }
        }

        fn meet(a: ScalarKind, b: ScalarKind) -> ScalarKind {
            use ScalarKind::*;
            if a == b {
                return a;
            }
            if a == Top {
                return b;
            }
            if b == Top {
                return a;
            }
            Unknown
        }

        // GFP：block0 = Unknown（参数/未初始化）；其余 = Top（乐观顶）
        let mut entry_kinds: Vec<Vec<ScalarKind>> = vec![vec![Top; num_locals]; nblocks];
        entry_kinds[0] = vec![Unknown; num_locals];
        let mut worklist: VecDeque<usize> = (0..nblocks).collect();
        let mut in_work = vec![true; nblocks];
        while let Some(bi) = worklist.pop_front() {
            in_work[bi] = false;
            let entry = entry_kinds[bi].clone();
            let exit = match transfer(&insns, &block_first, nblocks, bi, &entry) {
                Some(e) => e,
                None => return None,
            };
            for &si in &succs[bi] {
                let old = &entry_kinds[si];
                let mut changed = false;
                let mut merged = old.clone();
                for i in 0..num_locals {
                    let m = meet(old[i], exit[i]);
                    if m != old[i] {
                        changed = true;
                        merged[i] = m;
                    }
                }
                if changed {
                    entry_kinds[si] = merged;
                    if !in_work[si] {
                        worklist.push_back(si);
                        in_work[si] = true;
                    }
                }
            }
        }

        // 组装结果：leader IP → entry kinds
        let mut result = HashMap::new();
        for (bi, &lip) in leaders.iter().enumerate() {
            result.insert(lip, entry_kinds[bi].clone());
        }
        Some(result)
    }

    // ── A2b：标量辅助（槽 / 清空 / 物化 / 原生运算）────────────────────────

    /// 获取栈偏移 `off` 的标量伴随槽（按需创建）。槽存 8 字节裸标量。
    fn stack_scalar_slot(&mut self, off: i32, kind: ScalarKind) -> StackSlot {
        if let Some(&(k, slot)) = self.stack_scalars.get(&off) {
            if k == kind {
                return slot;
            }
        }
        let slot = self.builder.create_sized_stack_slot(StackSlotData::new(
            StackSlotKind::ExplicitSlot,
            8,
            8,
        ));
        self.stack_scalars.insert(off, (kind, slot));
        slot
    }

    /// 获取局部 `i` 的标量伴随槽（按需创建）。
    fn local_scalar_slot(&mut self, i: usize, kind: ScalarKind) -> StackSlot {
        if let Some(&(k, slot)) = self.local_scalars.get(&i) {
            if k == kind {
                return slot;
            }
        }
        let slot = self.builder.create_sized_stack_slot(StackSlotData::new(
            StackSlotKind::ExplicitSlot,
            8,
            8,
        ));
        self.local_scalars.insert(i, (kind, slot));
        slot
    }

    /// 清空单个栈偏移的标量跟踪。
    fn clear_stack_scalar_at(&mut self, off: i32) {
        self.stack_scalars.remove(&off);
    }

    /// 低层物化单个栈偏移（写 Value 槽，若为标量）。用**非失效**的原始调用发射
    /// （不触发 `invalidate_stack_scalars`，避免递归）。**物化后移除该偏移的标量
    /// 跟踪**——物化的值必然被通用消费者/调用消费（弹栈），此后 Value 槽权威；
    /// 保留跟踪会导致调用/Dup 后误读过期标量槽（静默错值红线，A2 调试发现）。
    fn materialize_stack_at(&mut self, off: i32) {
        if !self.scalar_enabled {
            return;
        }
        if let Some(&(kind, slot)) = self.stack_scalars.get(&off) {
            let addr = self.builder.ins().stack_addr(self.ptr, self.stack_slot, off);
            self.materialize_scalar_slot_to(kind, slot, addr);
            self.stack_scalars.remove(&off);
        }
    }

    /// 物化 [off, off + count*VS) 内的标量栈值（写 Value 槽，供通用消费者读取）。
    fn materialize_stack_range(&mut self, off: i32, count: usize) {
        if !self.scalar_enabled {
            return;
        }
        for i in 0..count {
            self.materialize_stack_at(off + (i as i32) * (VALUE_SIZE as i32));
        }
    }

    /// 物化全部栈上标量（块终止符前调用，使合并点内存权威）。
    fn materialize_all_stack(&mut self) {
        if !self.scalar_enabled {
            return;
        }
        let offsets: Vec<i32> = self
            .stack_scalars
            .keys()
            .copied()
            .filter(|&o| o >= 0 && o < self.sp)
            .collect();
        for o in offsets {
            self.materialize_stack_at(o);
        }
    }

    /// 失效全部栈标量跟踪：先物化（写 Value 槽）再清跟踪——使清栈后任何通用
    /// 消费者读到有效 Value。在通用 hostcall 入口调用（与分析的「通用 opcode
    /// 清空栈」一致）。物化用非失效的 `emit_scalar_to_value`，无递归。
    fn invalidate_stack_scalars(&mut self) {
        if !self.scalar_enabled {
            return;
        }
        self.materialize_all_stack();
        self.stack_scalars.clear();
    }

    /// 低层：把 SSA 标量物化为 Value（写 out 槽）。非失效（供物化/Store/错误路径用）。
    fn emit_scalar_to_value(&mut self, name: &str, ty: types::Type, v: Value_, out: Value_) {
        self.emit_line_hint();
        let callee = self.hostcall_addr(name).unwrap();
        let sig = self.import_sig(&[ty, self.ptr], None);
        self.builder.ins().call_indirect(sig, callee, &[self.vm, v, out]);
    }

    /// 拷贝标量槽 src → dst（按种类的原生类型，8 字节）。
    fn copy_scalar_slot(&mut self, src: StackSlot, dst: StackSlot, kind: ScalarKind) {
        let (ty, v) = match kind {
            ScalarKind::I32 => (types::I64, self.builder.ins().stack_load(types::I64, src, 0)),
            ScalarKind::F64 => (types::F64, self.builder.ins().stack_load(types::F64, src, 0)),
            ScalarKind::Bool => (types::I8, self.builder.ins().stack_load(types::I8, src, 0)),
            _ => (types::I64, self.builder.ins().stack_load(types::I64, src, 0)),
        };
        let daddr = self.builder.ins().stack_addr(self.ptr, dst, 0);
        self.builder.ins().store(MemFlags::new(), v, daddr, 0);
        let _ = ty;
    }

    /// 把标量槽的值物化为 Value（写 addr 槽）。非失效。
    fn materialize_scalar_slot_to(&mut self, kind: ScalarKind, slot: StackSlot, addr: Value_) {
        match kind {
            ScalarKind::I32 => {
                let v = self.builder.ins().stack_load(types::I64, slot, 0);
                self.emit_scalar_to_value("host_make_int", types::I64, v, addr);
            }
            ScalarKind::F64 => {
                let v = self.builder.ins().stack_load(types::F64, slot, 0);
                self.emit_scalar_to_value("host_make_float", types::F64, v, addr);
            }
            ScalarKind::Bool => {
                let v = self.builder.ins().stack_load(types::I8, slot, 0);
                self.emit_scalar_to_value("host_make_bool", types::I8, v, addr);
            }
            _ => {}
        }
    }

    /// A2b：读取当前 sp 处条件值的真值（i8）。已知标量 → 原生测试；否则 host_truthy
    /// （Unknown 值的 Value 槽始终有效）。调用前 sp 已弹到条件位置。
    fn emit_cond_truthiness(&mut self) -> Value_ {
        if self.scalar_enabled {
            if let Some(&(ck, cslot)) = self.stack_scalars.get(&self.sp) {
                match ck {
                    ScalarKind::Bool => {
                        return self.builder.ins().stack_load(types::I8, cslot, 0);
                    }
                    ScalarKind::I32 => {
                        let v = self.builder.ins().stack_load(types::I64, cslot, 0);
                        let nz = self.builder.ins().icmp_imm(IntCC::NotEqual, v, 0);
                        let one = self.builder.ins().iconst(types::I8, 1);
                        let zero = self.builder.ins().iconst(types::I8, 0);
                        return self.builder.ins().select(nz, one, zero);
                    }
                    ScalarKind::F64 => {
                        let v = self.builder.ins().stack_load(types::F64, cslot, 0);
                        let fz = self.builder.ins().f64const(0.0);
                        let nz = self.builder.ins().fcmp(FloatCC::NotEqual, v, fz);
                        let one = self.builder.ins().iconst(types::I8, 1);
                        let zero = self.builder.ins().iconst(types::I8, 0);
                        return self.builder.ins().select(nz, one, zero);
                    }
                    _ => {}
                }
            }
        }
        let cond_addr = self.builder.ins().stack_addr(self.ptr, self.stack_slot, self.sp);
        self.call_hostcall_val_ret_u8("host_truthy", cond_addr)
    }

    /// 当前块内局部的标量种类（块入口设为分析值；块内 Store 演化）。
    fn local_kind(&self, i: usize) -> ScalarKind {
        self.cur_local_kinds.get(i).copied().unwrap_or(ScalarKind::Unknown)
    }

    fn set_local_kind(&mut self, i: usize, kind: ScalarKind) {
        if i >= self.cur_local_kinds.len() {
            self.cur_local_kinds.resize(i + 1, ScalarKind::Unknown);
        }
        self.cur_local_kinds[i] = kind;
    }

    /// I32 范围检查：`r < I32_MIN || r > I32_MAX`（b1）。
    /// I32 范围检查：`r < I32_MIN || r > I32_MAX`（b1）。
    fn emit_i32_range_err(&mut self, r: Value_) -> Value_ {
        let lo = self.builder.ins().icmp_imm(IntCC::SignedLessThan, r, I32_MIN);
        let hi = self.builder.ins().icmp_imm(IntCC::SignedGreaterThan, r, I32_MAX);
        self.builder.ins().bor(lo, hi)
    }

    /// A2b：原生标量二元运算（两操作数均为已知标量；sp 已弹出 a、b）。
    /// 结果写 a_off（out 位置）标量槽；Value 槽延迟物化。
    /// I32：i64 溢出检查 + I32 范围检查 + 除零/MIN/-1 检查（错误 hostcall 冷路径，
    /// 消息与 VM 逐字一致）；F64：原生浮点（VM 浮点算术无错误路径）。
    /// Mul 不专用化（checked_mul 校验复杂，保 hostcall 路径正确性优先）。
    fn emit_native_binop(&mut self, op: Op, a_off: i32, b_off: i32, kind: ScalarKind) -> Result<(), String> {
        use ScalarKind::*;
        let (akind, aslot) = self.stack_scalars[&a_off];
        let (bkind, bslot) = self.stack_scalars[&b_off];
        debug_assert!(akind == kind && bkind == kind);
        match kind {
            I32 => {
                let a = self.builder.ins().stack_load(types::I64, aslot, 0);
                let b = self.builder.ins().stack_load(types::I64, bslot, 0);
                let min = self.builder.ins().iconst(types::I64, i64::MIN);
                // (r, 错误条件列表)
                let (r, checks): (Value_, Vec<(Value_, NativeErr)>) = match op {
                    Op::Add => {
                        let r = self.builder.ins().iadd(a, b);
                        // 有符号溢出：(a^r) & (b^r) 符号位为 1
                        let x = self.builder.ins().bxor(a, r);
                        let y = self.builder.ins().bxor(b, r);
                        let m = self.builder.ins().band(x, y);
                        let ovf = self.builder.ins().icmp_imm(IntCC::SignedLessThan, m, 0);
                        let rng = self.emit_i32_range_err(r);
                        (
                            r,
                            vec![(ovf, NativeErr::Overflow), (rng, NativeErr::Range)],
                        )
                    }
                    Op::Sub => {
                        let r = self.builder.ins().isub(a, b);
                        // 有符号溢出：(a^b) & (a^r) 符号位为 1
                        let x = self.builder.ins().bxor(a, b);
                        let y = self.builder.ins().bxor(a, r);
                        let m = self.builder.ins().band(x, y);
                        let ovf = self.builder.ins().icmp_imm(IntCC::SignedLessThan, m, 0);
                        let rng = self.emit_i32_range_err(r);
                        (
                            r,
                            vec![(ovf, NativeErr::Overflow), (rng, NativeErr::Range)],
                        )
                    }
                    Op::Div => {
                        let b_eq0 = self.builder.ins().icmp_imm(IntCC::Equal, b, 0);
                        let a_eq_min = self.builder.ins().icmp(IntCC::Equal, a, min);
                        let b_eq_m1 = self.builder.ins().icmp_imm(IntCC::Equal, b, -1);
                        let minm1 = self.builder.ins().band(a_eq_min, b_eq_m1);
                        // 安全除数：仅除零或 i64::MIN/-1（真陷阱）时替换为 1——
                        // 有效除法（如 I32::MIN / -1 = 2^31）必须保留真实结果供范围检查。
                        let one = self.builder.ins().iconst(types::I64, 1);
                        let b_bad = self.builder.ins().bor(b_eq0, minm1);
                        let safe = self.builder.ins().select(b_bad, one, b);
                        let r = self.builder.ins().sdiv(a, safe);
                        let rng = self.emit_i32_range_err(r);
                        (
                            r,
                            vec![
                                (b_eq0, NativeErr::DivZero),
                                (minm1, NativeErr::Overflow),
                                (rng, NativeErr::Range),
                            ],
                        )
                    }
                    Op::Mod => {
                        let b_eq0 = self.builder.ins().icmp_imm(IntCC::Equal, b, 0);
                        let a_eq_min = self.builder.ins().icmp(IntCC::Equal, a, min);
                        let b_eq_m1 = self.builder.ins().icmp_imm(IntCC::Equal, b, -1);
                        let minm1 = self.builder.ins().band(a_eq_min, b_eq_m1);
                        // 安全除数：仅除零或 i64::MIN/-1（真陷阱）时替换为 1。
                        let one = self.builder.ins().iconst(types::I64, 1);
                        let b_bad = self.builder.ins().bor(b_eq0, minm1);
                        let safe = self.builder.ins().select(b_bad, one, b);
                        let r = self.builder.ins().srem(a, safe);
                        let rng = self.emit_i32_range_err(r);
                        (
                            r,
                            vec![
                                (b_eq0, NativeErr::ModZero),
                                (minm1, NativeErr::Overflow),
                                (rng, NativeErr::Range),
                            ],
                        )
                    }
                    _ => return Err("native i32 binop: unsupported".into()),
                };
                // 写结果标量槽（a_off 处；先清旧跟踪）
                self.clear_stack_scalar_at(a_off);
                let rslot = self.stack_scalar_slot(a_off, I32);
                let raddr = self.builder.ins().stack_addr(self.ptr, rslot, 0);
                self.builder.ins().store(MemFlags::new(), r, raddr, 0);
                // 错误链（冷路径）
                self.emit_native_err_chain(r, &checks);
                self.bump_sp()?;
                Ok(())
            }
            F64 => {
                let a = self.builder.ins().stack_load(types::F64, aslot, 0);
                let b = self.builder.ins().stack_load(types::F64, bslot, 0);
                let r = match op {
                    Op::Add => self.builder.ins().fadd(a, b),
                    Op::Sub => self.builder.ins().fsub(a, b),
                    Op::Mul => self.builder.ins().fmul(a, b),
                    Op::Div => self.builder.ins().fdiv(a, b),
                    _ => return Err("native f64 binop: unsupported".into()),
                };
                self.clear_stack_scalar_at(a_off);
                let rslot = self.stack_scalar_slot(a_off, F64);
                let raddr = self.builder.ins().stack_addr(self.ptr, rslot, 0);
                self.builder.ins().store(MemFlags::new(), r, raddr, 0);
                self.bump_sp()?;
                Ok(())
            }
            _ => Err("native binop: bad kind".into()),
        }
    }

    /// A2b：在错误分支中按序检查各条件并设置对应错误，然后返回 ok=0。
    /// 冷路径（正确性优先）：链式 brif，逐层设错误消息后终止。
    fn emit_native_err_chain(&mut self, r: Value_, checks: &[(Value_, NativeErr)]) {
        for (cond, kind) in checks {
            let set_blk = self.builder.create_block();
            let cont_blk = self.builder.create_block();
            self.builder.ins().brif(*cond, set_blk, &[], cont_blk, &[]);
            self.builder.switch_to_block(set_blk);
            self.builder.seal_block(set_blk);
            match kind {
                NativeErr::Overflow => self.call_hostcall_err_set("host_set_int_overflow"),
                NativeErr::Range => self.call_hostcall_err_set_i64("host_set_int_range_error", r),
                NativeErr::DivZero => self.call_hostcall_err_set("host_set_div_zero"),
                NativeErr::ModZero => self.call_hostcall_err_set("host_set_mod_zero"),
            }
            let ok_false = self.builder.ins().iconst(types::I8, 0);
            self.builder.ins().return_(&[ok_false]);
            self.builder.switch_to_block(cont_blk);
            self.builder.seal_block(cont_blk);
        }
        self.terminated = false;
    }

    /// `fn(vm)` 无参错误设置 hostcall。
    fn call_hostcall_err_set(&mut self, name: &str) {
        self.emit_line_hint();
        let callee = self.hostcall_addr(name).unwrap();
        let sig = self.import_sig(&[], None);
        self.builder.ins().call_indirect(sig, callee, &[self.vm]);
    }

    /// `fn(vm, i64)` 错误设置 hostcall（范围错误带结果值）。
    fn call_hostcall_err_set_i64(&mut self, name: &str, arg: Value_) {
        self.emit_line_hint();
        let callee = self.hostcall_addr(name).unwrap();
        let sig = self.import_sig(&[types::I64], None);
        self.builder.ins().call_indirect(sig, callee, &[self.vm, arg]);
    }

    /// A2b：原生标量一元运算（Neg/Not；sp 已弹操作数）。
    fn emit_native_unop(&mut self, op: Op, a_off: i32, kind: ScalarKind) -> Result<(), String> {
        use ScalarKind::*;
        let (akind, aslot) = self.stack_scalars[&a_off];
        debug_assert_eq!(akind, kind);
        match (op, kind) {
            (Op::Neg, I32) => {
                let a = self.builder.ins().stack_load(types::I64, aslot, 0);
                let r = self.builder.ins().ineg(a);
                // i64::MIN 取负溢出（checked_neg）+ I32 范围检查
                let ovf = self.builder.ins().icmp_imm(IntCC::Equal, a, i64::MIN);
                let rng = self.emit_i32_range_err(r);
                let checks = vec![(ovf, NativeErr::Overflow), (rng, NativeErr::Range)];
                self.clear_stack_scalar_at(a_off);
                let rslot = self.stack_scalar_slot(a_off, I32);
                let raddr = self.builder.ins().stack_addr(self.ptr, rslot, 0);
                self.builder.ins().store(MemFlags::new(), r, raddr, 0);
                self.emit_native_err_chain(r, &checks);
                self.bump_sp()?;
                Ok(())
            }
            (Op::Neg, F64) => {
                let a = self.builder.ins().stack_load(types::F64, aslot, 0);
                let r = self.builder.ins().fneg(a);
                self.clear_stack_scalar_at(a_off);
                let rslot = self.stack_scalar_slot(a_off, F64);
                let raddr = self.builder.ins().stack_addr(self.ptr, rslot, 0);
                self.builder.ins().store(MemFlags::new(), r, raddr, 0);
                self.bump_sp()?;
                Ok(())
            }
            (Op::Not, Bool) => {
                let a = self.builder.ins().stack_load(types::I8, aslot, 0);
                let one = self.builder.ins().iconst(types::I8, 1);
                let r = self.builder.ins().bxor(a, one);
                self.clear_stack_scalar_at(a_off);
                let rslot = self.stack_scalar_slot(a_off, Bool);
                let raddr = self.builder.ins().stack_addr(self.ptr, rslot, 0);
                self.builder.ins().store(MemFlags::new(), r, raddr, 0);
                self.bump_sp()?;
                Ok(())
            }
            _ => Err("native unop: unsupported".into()),
        }
    }

    /// A2b：原生标量比较（Eq/Neq/Lt/Gt/Lte/Gte；sp 已弹 a、b）。
    /// 结果 Bool（i8）写 a_off 标量槽；Value 槽延迟物化。无错误路径。
    fn emit_native_cmp(&mut self, op: Op, a_off: i32, b_off: i32, kind: ScalarKind) -> Result<(), String> {
        use ScalarKind::*;
        let (akind, aslot) = self.stack_scalars[&a_off];
        let (bkind, bslot) = self.stack_scalars[&b_off];
        debug_assert!(akind == kind && bkind == kind);
        let cond = match kind {
            I32 => {
                let a = self.builder.ins().stack_load(types::I64, aslot, 0);
                let b = self.builder.ins().stack_load(types::I64, bslot, 0);
                let cc = match op {
                    Op::Eq => IntCC::Equal,
                    Op::Neq => IntCC::NotEqual,
                    Op::Lt => IntCC::SignedLessThan,
                    Op::Gt => IntCC::SignedGreaterThan,
                    Op::Lte => IntCC::SignedLessThanOrEqual,
                    Op::Gte => IntCC::SignedGreaterThanOrEqual,
                    _ => return Err("native i32 cmp: unsupported".into()),
                };
                self.builder.ins().icmp(cc, a, b)
            }
            F64 => {
                let a = self.builder.ins().stack_load(types::F64, aslot, 0);
                let b = self.builder.ins().stack_load(types::F64, bslot, 0);
                let cc = match op {
                    Op::Eq => FloatCC::Equal,
                    Op::Neq => FloatCC::NotEqual,
                    Op::Lt => FloatCC::LessThan,
                    Op::Gt => FloatCC::GreaterThan,
                    Op::Lte => FloatCC::LessThanOrEqual,
                    Op::Gte => FloatCC::GreaterThanOrEqual,
                    _ => return Err("native f64 cmp: unsupported".into()),
                };
                self.builder.ins().fcmp(cc, a, b)
            }
            _ => return Err("native cmp: bad kind".into()),
        };
        // b1 → i8（0/1）
        let one = self.builder.ins().iconst(types::I8, 1);
        let zero = self.builder.ins().iconst(types::I8, 0);
        let r = self.builder.ins().select(cond, one, zero);
        self.clear_stack_scalar_at(a_off);
        let rslot = self.stack_scalar_slot(a_off, Bool);
        let raddr = self.builder.ins().stack_addr(self.ptr, rslot, 0);
        self.builder.ins().store(MemFlags::new(), r, raddr, 0);
        self.bump_sp()?;
        Ok(())
    }

    fn emit_op(&mut self, op: Op, op_start: usize, ip_after: usize) -> Result<(), String> {
        use Op::*;
        // 9c：记录当前 opcode 的源码行号（0 = 无），供 hostcall 报错时携带。
        self.cur_line = self.chunk.line_at(op_start).unwrap_or(0);
        match op {
            PushInt(n) => {
                if self.scalar_enabled {
                    // A2b：专用化——写标量槽（原生常量），Value 槽延迟物化（通用消费者前）。
                    let off = self.sp;
                    let slot = self.stack_scalar_slot(off, ScalarKind::I32);
                    let addr = self.builder.ins().stack_addr(self.ptr, slot, 0);
                    let v = self.builder.ins().iconst(types::I64, n);
                    self.builder.ins().store(MemFlags::new(), v, addr, 0);
                } else {
                    let out = self.stack_addr_at_sp();
                    self.call_hostcall_i64("host_make_int", n, out);
                }
                self.bump_sp()?;
            }
            PushFloat(f) => {
                if self.scalar_enabled {
                    let off = self.sp;
                    let slot = self.stack_scalar_slot(off, ScalarKind::F64);
                    let addr = self.builder.ins().stack_addr(self.ptr, slot, 0);
                    let v = self.builder.ins().f64const(f);
                    self.builder.ins().store(MemFlags::new(), v, addr, 0);
                } else {
                    let out = self.stack_addr_at_sp();
                    self.call_hostcall_f64("host_make_float", f, out);
                }
                self.bump_sp()?;
            }
            PushFloat32(f) => {
                // 阶段 6：真正的 f32 hostcall，保留 dtype 信息（不再降级为 f64）。
                // A2b：f32 不专用化（保持 hostcall 语义），通用 hostcall 会失效栈标量。
                let out = self.stack_addr_at_sp();
                self.call_hostcall_f32("host_make_float32", f, out);
                self.bump_sp()?;
            }
            PushBool(b) => {
                if self.scalar_enabled {
                    let off = self.sp;
                    let slot = self.stack_scalar_slot(off, ScalarKind::Bool);
                    let addr = self.builder.ins().stack_addr(self.ptr, slot, 0);
                    let v = self.builder.ins().iconst(types::I8, if b { 1 } else { 0 });
                    self.builder.ins().store(MemFlags::new(), v, addr, 0);
                } else {
                    let out = self.stack_addr_at_sp();
                    self.call_hostcall_u8("host_make_bool", if b { 1 } else { 0 }, out);
                }
                self.bump_sp()?;
            }
            PushChar(c) => {
                if self.scalar_enabled {
                    // A2b：Char 走 I32 标量（host_make_int 语义 = Value::Int(c, I32)）。
                    let off = self.sp;
                    let slot = self.stack_scalar_slot(off, ScalarKind::I32);
                    let addr = self.builder.ins().stack_addr(self.ptr, slot, 0);
                    let v = self.builder.ins().iconst(types::I64, c as i64);
                    self.builder.ins().store(MemFlags::new(), v, addr, 0);
                } else {
                    // JIT不支持Char直接作为标量值，使用Int作为fallback
                    let out = self.stack_addr_at_sp();
                    self.call_hostcall_i64("host_make_int", c as i64, out);
                }
                self.bump_sp()?;
            }
            PushStr(i) => {
                // 写 Value 前清除该位置残留标量跟踪（防止过期标量被后续误读）。
                self.clear_stack_scalar_at(self.sp);
                let out = self.stack_addr_at_sp();
                self.call_hostcall_u64("host_make_str", i as u64, out);
                self.bump_sp()?;
            }
            PushUnit => {
                self.clear_stack_scalar_at(self.sp);
                let out = self.stack_addr_at_sp();
                self.call_hostcall_unit("host_make_unit", out);
                self.bump_sp()?;
            }
            Pop => {
                self.sp -= VALUE_SIZE as i32;
                self.clear_stack_scalar_at(self.sp);
            }
            Dup => {
                let src_off = self.sp - VALUE_SIZE as i32;
                let dst_off = self.sp;
                let specialized = if self.scalar_enabled {
                    if let Some(&(kind, sslot)) = self.stack_scalars.get(&src_off) {
                        if kind != ScalarKind::Unknown {
                            // 源为标量 → 拷贝标量槽（原生）；两者 Value 槽皆延迟物化。
                            let dslot = self.stack_scalar_slot(dst_off, kind);
                            self.copy_scalar_slot(sslot, dslot, kind);
                            true
                        } else {
                            false
                        }
                    } else {
                        false
                    }
                } else {
                    false
                };
                if specialized {
                    self.bump_sp()?;
                } else {
                    self.copy_within_stack(src_off, dst_off);
                    self.bump_sp()?;
                }
            }
            Load(i) => {
                // A2b：局部分析为标量 → 读局部标量槽（原生）；Value 槽延迟物化。
                let lk = self.local_kind(i);
                let specialized = if self.scalar_enabled
                    && lk != ScalarKind::Unknown
                    && lk != ScalarKind::Top
                {
                    if let Some(&(_, lslot)) = self.local_scalars.get(&i) {
                        let dst_off = self.sp;
                        let dslot = self.stack_scalar_slot(dst_off, lk);
                        self.copy_scalar_slot(lslot, dslot, lk);
                        true
                    } else {
                        false
                    }
                } else {
                    false
                };
                if specialized {
                    self.bump_sp()?;
                } else {
                    let src = self.locals.get(&i).copied().ok_or("Load: bad local")?;
                    let dst_off = self.sp;
                    self.copy_slot_to_stack(src, 0, dst_off);
                    self.bump_sp()?;
                }
            }
            Store(i) => {
                self.sp -= VALUE_SIZE as i32;
                let src_off = self.sp;
                let specialized = if self.scalar_enabled {
                    if let Some(&(kind, sslot)) = self.stack_scalars.get(&src_off) {
                        if kind != ScalarKind::Unknown && kind != ScalarKind::Top {
                            // 值已知标量 → 写局部标量槽（原生）+ 局部 Value 槽（双写，跨块安全）。
                            let lslot = self.local_scalar_slot(i, kind);
                            self.copy_scalar_slot(sslot, lslot, kind);
                            if !self.locals.contains_key(&i) {
                                let new_s = self.builder.create_sized_stack_slot(StackSlotData::new(
                                    StackSlotKind::ExplicitSlot, VALUE_SIZE, 8,
                                ));
                                self.locals.insert(i, new_s);
                            }
                            let lval = self.locals[&i];
                            let lval_addr = self.builder.ins().stack_addr(self.ptr, lval, 0);
                            self.materialize_scalar_slot_to(kind, sslot, lval_addr);
                            self.set_local_kind(i, kind);
                            // 值已弹栈消费（存入局部）——清除该栈偏移的标量跟踪，
                            // 否则旧跟踪残留会导致后续 Store/Load 误读过期标量（静默错值）。
                            self.clear_stack_scalar_at(src_off);
                            true
                        } else {
                            false
                        }
                    } else {
                        false
                    }
                } else {
                    false
                };
                if !specialized {
                    // 通用路径：写 Value 槽；局部取消标量跟踪。
                    if !self.locals.contains_key(&i) {
                        let new_s = self.builder.create_sized_stack_slot(StackSlotData::new(
                            StackSlotKind::ExplicitSlot, VALUE_SIZE, 8,
                        ));
                        self.locals.insert(i, new_s);
                    }
                    let dst = self.locals[&i];
                    self.copy_stack_to_slot(src_off, dst, 0);
                    self.set_local_kind(i, ScalarKind::Unknown);
                    // 同标量路径：弹栈消费后清除残留标量跟踪。
                    self.clear_stack_scalar_at(src_off);
                }
            }
            LoadGlobal(i) => {
                self.clear_stack_scalar_at(self.sp);
                let out = self.stack_addr_at_sp();
                self.call_hostcall_u64("host_load_global", i as u64, out);
                self.bump_sp()?;
            }
            StoreGlobal(i) => {
                // M2-A3 修复：与 VM opcode 10 对齐——StoreGlobal **消费**栈值（pop 不推回）。
                // 此前 JIT 推回结果，与 VM 不一致：字节码在需要值保留的上下文（Assign 表达式）
                // 用 `Dup` 前置，let/解构上下文期待消费。推回导致 tuple 解构
                // （`let (a,b)=t`）后续 TupleGet 读到残留错位值（nm/nv 静默变 Unit）。
                // 该 bug 在 A3 前被「含 TupleGet 的函数整体 fallback」掩盖。
                self.sp -= VALUE_SIZE as i32;
                let val_off = self.sp;
                self.materialize_stack_at(val_off);
                self.clear_stack_scalar_at(val_off);
                let val_addr = self.builder.ins().stack_addr(self.ptr, self.stack_slot, val_off);
                let out_addr = val_addr; // 结果写回原槽（已弹栈消费，无害；签名需 out）
                self.call_hostcall_u64_val("host_store_global", i as u64, val_addr, out_addr);
            }
            Add => self.emit_binop(Op::Add, "host_add")?,
            Sub => self.emit_binop(Op::Sub, "host_sub")?,
            Mul => self.emit_binop(Op::Mul, "host_mul")?,
            Div => self.emit_binop(Op::Div, "host_div")?,
            Mod => self.emit_binop(Op::Mod, "host_mod")?,
            Neg => self.emit_unop(Op::Neg, "host_neg")?,
            Not => self.emit_unop(Op::Not, "host_not")?,
            Eq => self.emit_binop(Op::Eq, "host_eq")?,
            Neq => self.emit_binop(Op::Neq, "host_neq")?,
            Lt => self.emit_binop(Op::Lt, "host_lt")?,
            Gt => self.emit_binop(Op::Gt, "host_gt")?,
            Lte => self.emit_binop(Op::Lte, "host_lte")?,
            Gte => self.emit_binop(Op::Gte, "host_gte")?,
            Jump(o) => {
                // A2b：终止符——物化栈标量（合并点内存权威）。
                self.materialize_all_stack();
                let target = ((op_start as i64) + 5 + (o as i64)) as usize;
                let blk = self.blocks.get(&target).copied()
                    .ok_or_else(|| format!("Jump target {} not a leader", target))?;
                self.block_sp.insert(blk, self.sp);
                self.builder.ins().jump(blk, &[]);
                self.terminated = true;
            }
            JmpFalse(o) => {
                let target = ((op_start as i64) + 5 + (o as i64)) as usize;
                self.sp -= VALUE_SIZE as i32;
                // A2b：终止符——物化其余栈标量（合并点内存权威）。
                self.materialize_all_stack();
                // 条件：已知标量 → 原生真值；否则 host_truthy（Value 槽有效）。
                let truth = self.emit_cond_truthiness();
                let is_false = self.builder.ins().icmp_imm(IntCC::Equal, truth, 0);
                let jmp_blk = self.blocks.get(&target).copied()
                    .ok_or_else(|| format!("JmpFalse target {} not a leader", target))?;
                let next_blk = self.blocks.get(&ip_after).copied()
                    .ok_or_else(|| format!("JmpFalse fall-through {} not a leader", ip_after))?;
                self.block_sp.insert(jmp_blk, self.sp);
                self.block_sp.insert(next_blk, self.sp);
                self.builder.ins().brif(is_false, jmp_blk, &[], next_blk, &[]);
                self.terminated = true;
            }
            JmpTrue(o) => {
                let target = ((op_start as i64) + 5 + (o as i64)) as usize;
                self.sp -= VALUE_SIZE as i32;
                // A2b：终止符——物化其余栈标量。
                self.materialize_all_stack();
                let truth = self.emit_cond_truthiness();
                let is_true = self.builder.ins().icmp_imm(IntCC::NotEqual, truth, 0);
                let jmp_blk = self.blocks.get(&target).copied()
                    .ok_or_else(|| format!("JmpTrue target {} not a leader", target))?;
                let next_blk = self.blocks.get(&ip_after).copied()
                    .ok_or_else(|| format!("JmpTrue fall-through {} not a leader", ip_after))?;
                self.block_sp.insert(jmp_blk, self.sp);
                self.block_sp.insert(next_blk, self.sp);
                self.builder.ins().brif(is_true, jmp_blk, &[], next_blk, &[]);
                self.terminated = true;
            }
            Call(i) => {
                let out = self.stack_addr_at_sp();
                let null_ptr = self.builder.ins().iconst(self.ptr, 0);
                // A1：目标为已注册用户函数 → JIT-to-JIT 直接调用（不再逃逸解释器）。
                if let Some(callee_idx) = self.name_to_chunk.get(i).copied().flatten() {
                    self.emit_direct_call(callee_idx, i, 0, null_ptr, out)?;
                } else {
                    self.call_hostcall_call("host_call", i as u64, 0, null_ptr, out);
                }
                // A2b：调用结果写 `out`（Value）——清除该偏移的标量跟踪（参数物化
                // 保留了旧标量跟踪，调用后指向过期值 → 静默错值红线）。
                self.clear_stack_scalar_at(self.sp);
                self.bump_sp()?;
            }
            CallN(i, n) => {
                // Args are at [sp - n*VS, sp). Pop them, then out is at sp.
                self.sp -= (n as i32) * (VALUE_SIZE as i32);
                // A2b：物化参数（Value 槽有效供被调函数读取）。
                self.materialize_stack_range(self.sp, n as usize);
                let args_addr = self.builder.ins().stack_addr(self.ptr, self.stack_slot, self.sp);
                let out = self.stack_addr_at_sp();
                // A1：目标为已注册用户函数 → JIT-to-JIT 直接调用（不再逃逸解释器）。
                if let Some(callee_idx) = self.name_to_chunk.get(i).copied().flatten() {
                    self.emit_direct_call(callee_idx, i, n as u64, args_addr, out)?;
                } else {
                    self.call_hostcall_call("host_call", i as u64, n as u64, args_addr, out);
                }
                // A2b：调用结果写 `out`（Value）——清除该偏移的标量跟踪。
                self.clear_stack_scalar_at(self.sp);
                self.bump_sp()?;
            }
            CallClosure(n) => {
                // a1 P1：间接调用闭包/函数值。栈上 [arg1..argN, callee]（N+1 个值），
                // host_call_indirect 取最后一个为 callee，其余为参数（走 Vm::call_value）。
                self.sp -= ((n + 1) as i32) * (VALUE_SIZE as i32);
                self.materialize_stack_range(self.sp, (n + 1) as usize);
                let args_addr = self.builder.ins().stack_addr(self.ptr, self.stack_slot, self.sp);
                let out = self.stack_addr_at_sp();
                self.call_hostcall_make_n("host_call_indirect", (n + 1) as u64, args_addr, out);
                self.bump_sp()?;
                // B2: 检查 host_call_indirect 是否设置了错误（如「期望可调用值」/未定义函数）。
                // 若有错误，立即返回 ok=false，让 run_jit 读取 last_error 并触发 fallback/报错，
                // 避免错误延迟到 run_jit 末尾才浮出（复用 MethodCall 的 B2 模式）。
                let err_flag = self.call_hostcall_vm_ret_u8("host_check_error");
                let has_err = self.builder.ins().icmp_imm(IntCC::NotEqual, err_flag, 0);
                let err_blk = self.builder.create_block();
                let cont_blk = self.builder.create_block();
                self.builder.ins().brif(has_err, err_blk, &[], cont_blk, &[]);
                // 错误路径：返回 ok=0（run_jit 会 take_last_error 并返回 Err）
                self.builder.switch_to_block(err_blk);
                self.builder.seal_block(err_blk);
                let ok_false = self.builder.ins().iconst(types::I8, 0);
                self.builder.ins().return_(&[ok_false]);
                // 继续路径
                self.builder.switch_to_block(cont_blk);
                self.builder.seal_block(cont_blk);
                self.terminated = false;
            }
            MethodCall(i, n) => {
                // host_method_call 期望 receiver + n 个 args = n+1 个值
                // sp 下移 (n+1)*VS，让 args_addr 指向 receiver
                self.sp -= ((n + 1) as i32) * (VALUE_SIZE as i32);
                self.materialize_stack_range(self.sp, (n + 1) as usize);
                let args_addr = self.builder.ins().stack_addr(self.ptr, self.stack_slot, self.sp);
                let out = self.stack_addr_at_sp();
                self.call_hostcall_call("host_method_call", i as u64, (n + 1) as u64, args_addr, out);
                self.bump_sp()?;
                // B2: 检查 host_method_call 是否设置了错误（如 matmul shape mismatch）。
                // 若有错误，立即返回 ok=false，让 run_jit 读取 last_error 并触发 fallback。
                // 这避免了"静默 push Unit + 后续 println 输出 ()"的问题。
                let err_flag = self.call_hostcall_vm_ret_u8("host_check_error");
                let has_err = self.builder.ins().icmp_imm(IntCC::NotEqual, err_flag, 0);
                let err_blk = self.builder.create_block();
                let cont_blk = self.builder.create_block();
                self.builder.ins().brif(has_err, err_blk, &[], cont_blk, &[]);
                // 错误路径：返回 ok=0（run_jit 会 take_last_error 并返回 Err）
                self.builder.switch_to_block(err_blk);
                self.builder.seal_block(err_blk);
                let ok_false = self.builder.ins().iconst(types::I8, 0);
                self.builder.ins().return_(&[ok_false]);
                // 继续路径
                self.builder.switch_to_block(cont_blk);
                self.builder.seal_block(cont_blk);
                self.terminated = false;
            }
            Ret => {
                // A2b：终止符——物化栈标量（含返回值，Value 槽有效供拷贝）。
                self.materialize_all_stack();
                self.sp -= VALUE_SIZE as i32;
                if let (Some(out), Some(cont)) = (self.inline_out, self.inline_cont) {
                    // A2：内联模式——结果写调用方 out 槽，跳汇合块（不返回整个函数）。
                    self.copy_stack_to_ptr(self.sp, out, 0);
                    self.builder.ins().jump(cont, &[]);
                    self.terminated = true;
                } else {
                    self.copy_stack_to_ptr(self.sp, self.out_ptr, 0);
                    let ok = self.builder.ins().iconst(types::I8, 1);
                    self.builder.ins().return_(&[ok]);
                    self.terminated = true;
                }
            }
            MakeVec(n) => {
                self.sp -= (n as i32) * (VALUE_SIZE as i32);
                self.materialize_stack_range(self.sp, n as usize);
                let args_addr = self.builder.ins().stack_addr(self.ptr, self.stack_slot, self.sp);
                let out = self.stack_addr_at_sp();
                self.call_hostcall_make_n("host_make_vec", n as u64, args_addr, out);
                self.bump_sp()?;
            }
            MakeMap(n) => {
                let total = n * 2;
                self.sp -= (total as i32) * (VALUE_SIZE as i32);
                self.materialize_stack_range(self.sp, total as usize);
                let args_addr = self.builder.ins().stack_addr(self.ptr, self.stack_slot, self.sp);
                let out = self.stack_addr_at_sp();
                self.call_hostcall_make_n("host_make_map", n as u64, args_addr, out);
                self.bump_sp()?;
            }
            NewStruct(name_i, field_count) => {
                let total = field_count * 2;
                self.sp -= (total as i32) * (VALUE_SIZE as i32);
                self.materialize_stack_range(self.sp, total as usize);
                let args_addr = self.builder.ins().stack_addr(self.ptr, self.stack_slot, self.sp);
                let out = self.stack_addr_at_sp();
                self.call_hostcall_new_struct("host_new_struct", name_i as u64, field_count as u64, args_addr, out);
                self.bump_sp()?;
            }
            NewUnion(name_i, field_i) => {
                // M1.2：union 构造 — 栈顶单个 value 弹出，构造 Value::Union
                self.sp -= VALUE_SIZE as i32;
                self.materialize_stack_at(self.sp);
                let val_addr = self.builder.ins().stack_addr(self.ptr, self.stack_slot, self.sp);
                let out = self.stack_addr_at_sp();
                self.call_hostcall_new_union("host_new_union", name_i as u64, field_i as u64, val_addr, out);
                self.bump_sp()?;
            }
            LoadField(i) => {
                self.sp -= VALUE_SIZE as i32;
                self.materialize_stack_at(self.sp);
                let recv_addr = self.builder.ins().stack_addr(self.ptr, self.stack_slot, self.sp);
                let out = self.stack_addr_at_sp();
                self.call_hostcall_u64_val("host_load_field", i as u64, recv_addr, out);
                self.bump_sp()?;
            }
            StoreField(i) => {
                // Stack: [..., recv, val]. Pop val, then recv. Out = modified recv.
                self.sp -= VALUE_SIZE as i32;
                let val_off = self.sp;
                self.sp -= VALUE_SIZE as i32;
                let recv_off = self.sp;
                self.materialize_stack_range(recv_off, 2);
                let recv_addr = self.builder.ins().stack_addr(self.ptr, self.stack_slot, recv_off);
                let val_addr = self.builder.ins().stack_addr(self.ptr, self.stack_slot, val_off);
                let out_addr = recv_addr; // result overwrites recv slot
                self.call_hostcall_store_field("host_store_field", i as u64, recv_addr, val_addr, out_addr);
                self.bump_sp()?; // push result back
            }
            IndexGet => {
                self.sp -= VALUE_SIZE as i32;
                let idx_off = self.sp;
                self.sp -= VALUE_SIZE as i32;
                let target_off = self.sp;
                self.materialize_stack_range(target_off, 2);
                let target_addr = self.builder.ins().stack_addr(self.ptr, self.stack_slot, target_off);
                let idx_addr = self.builder.ins().stack_addr(self.ptr, self.stack_slot, idx_off);
                let out = self.stack_addr_at_sp();
                self.call_hostcall_2_val("host_index_get", target_addr, idx_addr, out);
                self.bump_sp()?;
            }
            SliceStr => {
                self.sp -= VALUE_SIZE as i32;
                let end_off = self.sp;
                self.sp -= VALUE_SIZE as i32;
                let start_off = self.sp;
                self.sp -= VALUE_SIZE as i32;
                let target_off = self.sp;
                self.materialize_stack_range(target_off, 3);
                let target_addr = self.builder.ins().stack_addr(self.ptr, self.stack_slot, target_off);
                let start_addr = self.builder.ins().stack_addr(self.ptr, self.stack_slot, start_off);
                let end_addr = self.builder.ins().stack_addr(self.ptr, self.stack_slot, end_off);
                let out = self.stack_addr_at_sp();
                self.call_hostcall_3_val("host_slice_str", target_addr, start_addr, end_addr, out);
                self.bump_sp()?;
            }
            MakeEnum(name_i, variant_i, field_count) => {
                // 字节码 EnumLiteral 对每个字段压 [value, name] 两值（name 在顶），
                // 共 2×field_count 个栈槽——与 NewStruct 的 field_count*2 一致。
                // 修复前只减 field_count，导致 args_addr 指向 name 槽，
                // host_make_enum 把字段名当值（or_die(Result::Ok(42)) → "_0"）。
                let total = field_count * 2;
                self.sp -= (total as i32) * (VALUE_SIZE as i32);
                self.materialize_stack_range(self.sp, total as usize);
                let args_addr = self.builder.ins().stack_addr(self.ptr, self.stack_slot, self.sp);
                let out = self.stack_addr_at_sp();
                self.call_hostcall_make_enum("host_make_enum", name_i as u64, variant_i as u64, field_count as u64, args_addr, out);
                self.bump_sp()?;
            }
            IsEnumVariant(variant_i) => {
                self.sp -= VALUE_SIZE as i32;
                self.materialize_stack_at(self.sp);
                let recv_addr = self.builder.ins().stack_addr(self.ptr, self.stack_slot, self.sp);
                let out = self.stack_addr_at_sp();
                self.call_hostcall_u64_val("host_is_enum_variant", variant_i as u64, recv_addr, out);
                self.bump_sp()?;
            }
            EnumGetField(field_i) => {
                self.sp -= VALUE_SIZE as i32;
                self.materialize_stack_at(self.sp);
                let recv_addr = self.builder.ins().stack_addr(self.ptr, self.stack_slot, self.sp);
                let out = self.stack_addr_at_sp();
                self.call_hostcall_u64_val("host_enum_get_field", field_i as u64, recv_addr, out);
                self.bump_sp()?;
            }
            PushRange(s, e, inc) => {
                let out = self.stack_addr_at_sp();
                self.call_hostcall_push_range("host_push_range", s, e, if inc { 1 } else { 0 }, out);
                self.bump_sp()?;
            }
            MoveOp => { /* no-op */ }
            MakeTensor(rows, cols, dtype) => {
                let count = rows * cols;
                self.sp -= (count as i32) * (VALUE_SIZE as i32);
                self.materialize_stack_range(self.sp, count as usize);
                let args_addr = self.builder.ins().stack_addr(self.ptr, self.stack_slot, self.sp);
                let out = self.stack_addr_at_sp();
                // 阶段 6：根据 dtype 分发到对应 hostcall，保留 dtype 信息。
                // dtype 编码与 bytecode 一致：0 = F64，1 = F32，2 = F16，3 = BF16。
                match dtype {
                    1 => self.call_hostcall_make_tensor("host_make_tensor_f32", rows as u64, cols as u64, args_addr, out),
                    2 => self.call_hostcall_make_tensor("host_make_tensor_f16", rows as u64, cols as u64, args_addr, out),
                    3 => self.call_hostcall_make_tensor("host_make_tensor_bf16", rows as u64, cols as u64, args_addr, out),
                    _ => self.call_hostcall_make_tensor("host_make_tensor", rows as u64, cols as u64, args_addr, out),
                }
                self.bump_sp()?;
            }
            MakeClosure(params, captures, chunk_idx) => {
                // a1 P3：捕获值已由 bytecode 压到 JIT 栈（[cap0..capN]），MakeClosure 弹出
                // 装入 FnRef.captures（值内联，与 VM opcode 44 对齐）。复用 make_enum 的
                // hostcall 签名（vm, u64, u64, u64, *const Value, *mut Value）。
                self.sp -= (captures as i32) * (VALUE_SIZE as i32);
                self.materialize_stack_range(self.sp, captures as usize);
                let args_addr = self.builder.ins().stack_addr(self.ptr, self.stack_slot, self.sp);
                let out = self.stack_addr_at_sp();
                self.call_hostcall_make_enum(
                    "host_make_closure", params as u64, captures as u64, chunk_idx as u64, args_addr, out,
                );
                self.bump_sp()?;
            }
            IsStruct(name_i) => {
                // M2-A3：IsStruct → host_is_struct（VM opcode 46 语义：Struct 名匹配 → Bool）。
                // 结构体构造/字段读写（NewStruct/LoadField/StoreField）此前已 JIT。
                self.sp -= VALUE_SIZE as i32;
                self.materialize_stack_at(self.sp);
                let recv_addr = self.builder.ins().stack_addr(self.ptr, self.stack_slot, self.sp);
                let out = self.stack_addr_at_sp();
                self.call_hostcall_u64_val("host_is_struct", name_i as u64, recv_addr, out);
                self.bump_sp()?;
            }
            Spawn => {
                // M2-A3：Spawn → host_spawn（eager future_ready，VM opcode 48 语义；
                // 纯构造不涉及调度器挂起，JIT 安全）。
                self.sp -= VALUE_SIZE as i32;
                self.materialize_stack_at(self.sp);
                let val_addr = self.builder.ins().stack_addr(self.ptr, self.stack_slot, self.sp);
                let out = self.stack_addr_at_sp();
                self.call_hostcall_1_val("host_spawn", val_addr, out);
                self.bump_sp()?;
            }
            MakeTuple(n) => {
                // M2-A3：MakeTuple → host_make_tuple（VM opcode 49 语义：弹 n 个值装 Tuple）。
                self.sp -= (n as i32) * (VALUE_SIZE as i32);
                self.materialize_stack_range(self.sp, n as usize);
                let args_addr = self.builder.ins().stack_addr(self.ptr, self.stack_slot, self.sp);
                let out = self.stack_addr_at_sp();
                self.call_hostcall_make_n("host_make_tuple", n as u64, args_addr, out);
                self.bump_sp()?;
            }
            IsTuple(expected_len) => {
                // M2-A3：IsTuple → host_is_tuple（VM opcode 50 语义：长度匹配 → Bool）。
                self.sp -= VALUE_SIZE as i32;
                self.materialize_stack_at(self.sp);
                let recv_addr = self.builder.ins().stack_addr(self.ptr, self.stack_slot, self.sp);
                let out = self.stack_addr_at_sp();
                self.call_hostcall_u64_val("host_is_tuple", expected_len as u64, recv_addr, out);
                self.bump_sp()?;
            }
            TupleGet(i) => {
                // M2-A3：TupleGet → host_tuple_get（VM opcode 51 语义：取 index 元素，越界 Unit）。
                self.sp -= VALUE_SIZE as i32;
                self.materialize_stack_at(self.sp);
                let recv_addr = self.builder.ins().stack_addr(self.ptr, self.stack_slot, self.sp);
                let out = self.stack_addr_at_sp();
                self.call_hostcall_u64_val("host_tuple_get", i as u64, recv_addr, out);
                self.bump_sp()?;
            }
            Try => {
                // M2-A3：Try → host_try_pop（VM opcode 52 语义）。Err → 把完整 Result::Err
                // 作为函数返回值 early return（对齐 VM 的 frame 早退 / 解释器 unwrap_return
                // 单层透传）；Ok → 解包 push；非 Result → 透传 push。
                // 注意：`?` 在 try 块内编译为 IsEnumVariant+JmpFalse+EnumGetField（非本指令），
                // 因此 Try 的 early return 语义在单 chunk 内自洽，无需跨帧传播。
                self.sp -= VALUE_SIZE as i32;
                self.materialize_stack_at(self.sp);
                let val_addr = self.builder.ins().stack_addr(self.ptr, self.stack_slot, self.sp);
                let out = self.stack_addr_at_sp();
                let is_err = self.call_hostcall_val_ret_u8_out("host_try_pop", val_addr, out);
                let is_err_b = self.builder.ins().icmp_imm(IntCC::NotEqual, is_err, 0);
                let err_blk = self.builder.create_block();
                let cont_blk = self.builder.create_block();
                self.builder.ins().brif(is_err_b, err_blk, &[], cont_blk, &[]);
                // Err 路径：out 槽已是完整 Result::Err——复制到函数 out_ptr 并返回 ok=1
                // （调用方收到该 Err 作为函数结果，与 VM 早退一致）。
                self.builder.switch_to_block(err_blk);
                self.builder.seal_block(err_blk);
                self.copy_stack_to_ptr(self.sp, self.out_ptr, 0);
                let ok = self.builder.ins().iconst(types::I8, 1);
                self.builder.ins().return_(&[ok]);
                // 继续路径：值已写 out 槽 → bump_sp 继续。
                self.builder.switch_to_block(cont_blk);
                self.builder.seal_block(cont_blk);
                self.bump_sp()?;
                self.terminated = false;
            }
            TailCall(i, n) => {
                // M2-A3 + D2（保守）：TailCall 不激进帧复用——语义等价实现 = 弹出 n 个参数 →
                // host_call（VM call_with_args）→ 立即把结果作为函数返回值返回。
                // 注意：**不内联、不走 A1 直接调用**——TailCall 是函数终止指令（其后无继续
                // 代码），内联会错误地继续执行调用方（TailCall 后即 chunk 末尾）。
                // 深尾递归会随普通调用增长 VM 帧栈（优雅报错而非静默错值；与 CallN 行为一致）。
                self.sp -= (n as i32) * (VALUE_SIZE as i32);
                self.materialize_stack_range(self.sp, n as usize);
                let args_addr = self.builder.ins().stack_addr(self.ptr, self.stack_slot, self.sp);
                let out = self.stack_addr_at_sp();
                self.call_hostcall_call("host_call", i as u64, n as u64, args_addr, out);
                // 结果已在 out 槽 → 返回（host_call 失败时 last_error 由 run_jit surface）。
                self.copy_stack_to_ptr(self.sp, self.out_ptr, 0);
                let ok = self.builder.ins().iconst(types::I8, 1);
                self.builder.ins().return_(&[ok]);
                self.terminated = true;
            }
            TailCallClosure(n) => {
                // M2-A3 + D2（保守）：闭包尾调用同样走 host_call_indirect + 立即返回结果
                // （语义等价：结果 = 被调闭包结果）。B2 错误立即中断（复用 CallClosure 模式）。
                self.sp -= ((n + 1) as i32) * (VALUE_SIZE as i32);
                self.materialize_stack_range(self.sp, (n + 1) as usize);
                let args_addr = self.builder.ins().stack_addr(self.ptr, self.stack_slot, self.sp);
                let out = self.stack_addr_at_sp();
                self.call_hostcall_make_n("host_call_indirect", (n + 1) as u64, args_addr, out);
                // B2：host_call_indirect 设置错误 → 立即返回 ok=0（run_jit surface）。
                let err_flag = self.call_hostcall_vm_ret_u8("host_check_error");
                let has_err = self.builder.ins().icmp_imm(IntCC::NotEqual, err_flag, 0);
                let err_blk = self.builder.create_block();
                let cont_blk = self.builder.create_block();
                self.builder.ins().brif(has_err, err_blk, &[], cont_blk, &[]);
                self.builder.switch_to_block(err_blk);
                self.builder.seal_block(err_blk);
                let ok_false = self.builder.ins().iconst(types::I8, 0);
                self.builder.ins().return_(&[ok_false]);
                // 无错误：返回被调闭包结果。
                self.builder.switch_to_block(cont_blk);
                self.builder.seal_block(cont_blk);
                self.copy_stack_to_ptr(self.sp, self.out_ptr, 0);
                let ok = self.builder.ins().iconst(types::I8, 1);
                self.builder.ins().return_(&[ok]);
                self.terminated = true;
            }
            Await | Yield => {
                // M2-A3：async 挂起语义（Await 对 Pending Future 保存整个 VM 调度器栈并挂起；
                // Yield 保存调用栈让出控制权）——JIT 机器码帧无法序列化/恢复，**保守保持整函数
                // fallback**（语义正确优先；不做运行期 hostcall 混合——那会把合法挂起变成错误）。
                return Err(format!("JIT: async suspend opcode (await/yield) not supported, fallback to VM"));
            }
            MakeRef | MakeMutRef(_) | Deref | DerefStore => {
                // AUDIT-11.4.21：引用语义 opcodes 不 JIT 编译；整体 fallback VM。
                // M2-A3 评估：MakeMutRef 需写回 JIT 局部槽（跨边界），DerefStore 含写穿+行号报错；
                // 非本次优先级（Tuple/Try/Struct），保守保持 fallback 与 VM 语义一致。
                return Err(format!("JIT: ref/deref opcode not supported, fallback to VM"));
            }
            MakeCell | BindSelfCapture(_) => {
                // M1-S2（true letrec）：自引用 cell opcodes 不 JIT 编译；整体 fallback VM。
                // 闭包体本身（Load + CallClosure）仍可 JIT——host_call_indirect 走
                // Vm::call_value（已支持 Shared cell 解包），letrec 语义保持正确。
                // M2-A3 评估：仅闭包创建点（顶层 let 递归闭包）含此二指令，非本次优先级。
                return Err(format!("JIT: letrec cell opcode not supported, fallback to VM"));
            }
        }
        Ok(())
    }

    // ── Return helper ──────────────────────────────────────────────────────

    fn emit_return(&mut self) {
        // A2b：终止符——物化栈标量（含返回值，Value 槽有效供拷贝）。
        self.materialize_all_stack();
        if self.sp >= VALUE_SIZE as i32 {
            self.copy_stack_to_ptr(self.sp - VALUE_SIZE as i32, self.out_ptr, 0);
        } else {
            // No value on stack — return Unit via a temp slot.
            let tmp = self.builder.create_sized_stack_slot(StackSlotData::new(
                StackSlotKind::ExplicitSlot,
                VALUE_SIZE,
                8,
            ));
            let tmp_addr = self.builder.ins().stack_addr(self.ptr, tmp, 0);
            self.call_hostcall_unit("host_make_unit", tmp_addr);
            self.copy_slot_to_ptr(tmp, 0, self.out_ptr, 0);
        }
        let ok = self.builder.ins().iconst(types::I8, 1);
        self.builder.ins().return_(&[ok]);
        self.terminated = true;
    }

    // ── Stack management ───────────────────────────────────────────────────

    /// Increment `sp` by `VALUE_SIZE`, returning an error if this would
    /// exceed `MAX_STACK_DEPTH * VALUE_SIZE` bytes. Translators must call
    /// this *after* writing the pushed value into the stack slot.
    ///
    /// Returning `Err` here causes `translate_body` → `translate` to fail,
    /// which `JitContext::get_or_compile` propagates up; `run_jit` then
    /// catches it and falls back to `Vm::call` (interpreter) — see
    /// `compile/jit/mod.rs:62-65`. So a stack-overflow at translate time
    /// is a graceful degradation, not an abort.
    fn bump_sp(&mut self) -> Result<(), String> {
        let max_bytes = (VALUE_SIZE * MAX_STACK_DEPTH) as i32;
        if self.sp + VALUE_SIZE as i32 > max_bytes {
            return Err(format!(
                "JIT stack overflow: sp={} (slot {}) would exceed MAX_STACK_DEPTH={} ({} bytes)",
                self.sp, self.sp / VALUE_SIZE as i32, MAX_STACK_DEPTH, max_bytes
            ));
        }
        self.sp += VALUE_SIZE as i32;
        Ok(())
    }

    // ── Stack address helpers ──────────────────────────────────────────────

    /// Address of `stack_slot[sp]` (the next free slot, where a push would write).
    fn stack_addr_at_sp(&mut self) -> Value_ {
        self.builder.ins().stack_addr(self.ptr, self.stack_slot, self.sp)
    }

    // ── Copy helpers ───────────────────────────────────────────────────────

    /// Copy VALUE_SIZE bytes within `stack_slot` (src_off → dst_off).
    fn copy_within_stack(&mut self, src_off: i32, dst_off: i32) {
        let mut off = 0i32;
        while off < VALUE_SIZE as i32 {
            let v = self.builder.ins().stack_load(self.ptr, self.stack_slot, src_off + off);
            self.builder.ins().stack_store(v, self.stack_slot, dst_off + off);
            off += self.ptr.bytes() as i32;
        }
    }

    /// Copy from `stack_slot[src_off]` to a local `slot[dst_off]`.
    fn copy_stack_to_slot(&mut self, src_off: i32, dst: StackSlot, dst_off: i32) {
        let mut off = 0i32;
        while off < VALUE_SIZE as i32 {
            let v = self.builder.ins().stack_load(self.ptr, self.stack_slot, src_off + off);
            self.builder.ins().stack_store(v, dst, dst_off + off);
            off += self.ptr.bytes() as i32;
        }
    }

    /// Copy from a local `slot[src_off]` to `stack_slot[dst_off]`.
    fn copy_slot_to_stack(&mut self, src: StackSlot, src_off: i32, dst_off: i32) {
        let mut off = 0i32;
        while off < VALUE_SIZE as i32 {
            let v = self.builder.ins().stack_load(self.ptr, src, src_off + off);
            self.builder.ins().stack_store(v, self.stack_slot, dst_off + off);
            off += self.ptr.bytes() as i32;
        }
    }

    /// Copy from `stack_slot[src_off]` to a raw pointer `dst[dst_off]`.
    fn copy_stack_to_ptr(&mut self, src_off: i32, dst: Value_, dst_off: i32) {
        let mut off = 0i32;
        while off < VALUE_SIZE as i32 {
            let v = self.builder.ins().stack_load(self.ptr, self.stack_slot, src_off + off);
            self.builder.ins().store(MemFlags::trusted(), v, dst, dst_off + off);
            off += self.ptr.bytes() as i32;
        }
    }

    /// Copy from a raw pointer `src[src_off]` to a local `slot[dst_off]`.
    fn copy_ptr_to_slot(&mut self, src: Value_, src_off: i32, dst: StackSlot, dst_off: i32) {
        let mut off = 0i32;
        while off < VALUE_SIZE as i32 {
            let v = self.builder.ins().load(self.ptr, MemFlags::trusted(), src, src_off + off);
            self.builder.ins().stack_store(v, dst, dst_off + off);
            off += self.ptr.bytes() as i32;
        }
    }

    /// Copy from a local `slot[src_off]` to a raw pointer `dst[dst_off]`.
    fn copy_slot_to_ptr(&mut self, src: StackSlot, src_off: i32, dst: Value_, dst_off: i32) {
        let mut off = 0i32;
        while off < VALUE_SIZE as i32 {
            let v = self.builder.ins().stack_load(self.ptr, src, src_off + off);
            self.builder.ins().store(MemFlags::trusted(), v, dst, dst_off + off);
            off += self.ptr.bytes() as i32;
        }
    }

    // ── Hostcall signature & address helpers ───────────────────────────────

    fn hostcall_addr(&mut self, name: &str) -> Result<Value_, String> {
        let addr = super::hostcalls::hostcall_addr(name)
            .ok_or_else(|| format!("unknown hostcall: {name}"))?;
        Ok(self.builder.ins().iconst(self.ptr, addr as i64))
    }

    fn import_sig(&mut self, params: &[types::Type], ret: Option<types::Type>) -> SigRef {
        let mut sig = Signature::new(self.module.target_config().default_call_conv);
        sig.params.push(AbiParam::new(self.ptr)); // vm
        for p in params {
            sig.params.push(AbiParam::new(*p));
        }
        if let Some(r) = ret {
            sig.returns.push(AbiParam::new(r));
        }
        self.builder.import_signature(sig)
    }

    /// 9c：在 hostcall 前把当前指令源码行号写入 `vm.current_line`（JIT 报错行号）。
    /// `current_line` 是 `usize`（0 = 无行号），按字段偏移直接 store 指针宽整数。
    /// hostcall 的 `set_last_error`/`set_jit_error` 读取补行号——对齐 VM 的
    /// err_here/with_line 行为。行号每 opcode 恒定，Cranelift 会 CSE 常量。
    fn emit_line_hint(&mut self) {
        let off = std::mem::offset_of!(Vm, current_line) as i64;
        let off_v = self.builder.ins().iconst(self.ptr, off);
        let addr = self.builder.ins().iadd(self.vm, off_v);
        let line_v = self.builder.ins().iconst(self.ptr, self.cur_line as i64);
        self.builder.ins().store(MemFlags::new(), line_v, addr, 0);
    }

    // ── Hostcall emitters (all take raw `Value_` addresses for Value params) ─

    fn call_hostcall_unit(&mut self, name: &str, out: Value_) {
        self.invalidate_stack_scalars();
        self.emit_line_hint();
        let callee = self.hostcall_addr(name).unwrap();
        let sig = self.import_sig(&[self.ptr], None);
        self.builder.ins().call_indirect(sig, callee, &[self.vm, out]);
    }

    fn call_hostcall_i64(&mut self, name: &str, arg: i64, out: Value_) {
        self.invalidate_stack_scalars();
        self.emit_line_hint();
        let callee = self.hostcall_addr(name).unwrap();
        let sig = self.import_sig(&[types::I64, self.ptr], None);
        let a = self.builder.ins().iconst(types::I64, arg);
        self.builder.ins().call_indirect(sig, callee, &[self.vm, a, out]);
    }

    fn call_hostcall_f64(&mut self, name: &str, arg: f64, out: Value_) {
        self.invalidate_stack_scalars();
        self.emit_line_hint();
        let callee = self.hostcall_addr(name).unwrap();
        let sig = self.import_sig(&[types::F64, self.ptr], None);
        let a = self.builder.ins().f64const(arg);
        self.builder.ins().call_indirect(sig, callee, &[self.vm, a, out]);
    }

    /// f32 hostcall 调用：参数以 f32 ABI 传递（4 字节寄存器），与 f64 路径
    /// 区分以保留 dtype 信息到运行时。栈布局不变——out 仍为 *mut Value。
    fn call_hostcall_f32(&mut self, name: &str, arg: f32, out: Value_) {
        self.invalidate_stack_scalars();
        self.emit_line_hint();
        let callee = self.hostcall_addr(name).unwrap();
        let sig = self.import_sig(&[types::F32, self.ptr], None);
        let a = self.builder.ins().f32const(arg);
        self.builder.ins().call_indirect(sig, callee, &[self.vm, a, out]);
    }

    fn call_hostcall_u8(&mut self, name: &str, arg: u8, out: Value_) {
        self.invalidate_stack_scalars();
        self.emit_line_hint();
        let callee = self.hostcall_addr(name).unwrap();
        let sig = self.import_sig(&[types::I8, self.ptr], None);
        let a = self.builder.ins().iconst(types::I8, arg as i64);
        self.builder.ins().call_indirect(sig, callee, &[self.vm, a, out]);
    }

    fn call_hostcall_u64(&mut self, name: &str, arg: u64, out: Value_) {
        self.call_hostcall_i64(name, arg as i64, out);
    }

    fn call_hostcall_val_ret_u8(&mut self, name: &str, arg: Value_) -> Value_ {
        self.invalidate_stack_scalars();
        self.emit_line_hint();
        let callee = self.hostcall_addr(name).unwrap();
        let sig = self.import_sig(&[self.ptr], Some(types::I8));
        let call = self.builder.ins().call_indirect(sig, callee, &[self.vm, arg]);
        self.builder.inst_results(call)[0]
    }

    /// `fn(vm) -> u8` — e.g. host_check_error.
    fn call_hostcall_vm_ret_u8(&mut self, name: &str) -> Value_ {
        self.invalidate_stack_scalars();
        self.emit_line_hint();
        let callee = self.hostcall_addr(name).unwrap();
        let sig = self.import_sig(&[], Some(types::I8));
        let call = self.builder.ins().call_indirect(sig, callee, &[self.vm]);
        self.builder.inst_results(call)[0]
    }

    /// `fn(vm, *const Value, *mut Value)` — e.g. host_spawn（单值消费 + 写结果）。
    fn call_hostcall_1_val(&mut self, name: &str, arg: Value_, out: Value_) {
        self.invalidate_stack_scalars();
        self.emit_line_hint();
        let callee = self.hostcall_addr(name).unwrap();
        let sig = self.import_sig(&[self.ptr, self.ptr], None);
        self.builder.ins().call_indirect(sig, callee, &[self.vm, arg, out]);
    }

    /// `fn(vm, *const Value, *mut Value) -> u8` — hostcall 写 out 同时返回 u8 标志
    /// （如 host_try_pop：写继续/早退用的值到 out，返回 1 = 应 early return）。
    fn call_hostcall_val_ret_u8_out(&mut self, name: &str, arg: Value_, out: Value_) -> Value_ {
        self.invalidate_stack_scalars();
        self.emit_line_hint();
        let callee = self.hostcall_addr(name).unwrap();
        let sig = self.import_sig(&[self.ptr, self.ptr], Some(types::I8));
        let call = self.builder.ins().call_indirect(sig, callee, &[self.vm, arg, out]);
        self.builder.inst_results(call)[0]
    }

    /// `fn(vm, u64, *const Value, *mut Value)` — e.g. host_load_field.
    fn call_hostcall_u64_val(&mut self, name: &str, a: u64, val: Value_, out: Value_) {
        self.invalidate_stack_scalars();
        self.emit_line_hint();
        let callee = self.hostcall_addr(name).unwrap();
        let sig = self.import_sig(&[types::I64, self.ptr, self.ptr], None);
        let a_val = self.builder.ins().iconst(types::I64, a as i64);
        self.builder.ins().call_indirect(sig, callee, &[self.vm, a_val, val, out]);
    }

    /// `fn(vm, *const Value, *const Value, *mut Value)` — e.g. host_add.
    fn call_hostcall_2_val(&mut self, name: &str, a: Value_, b: Value_, out: Value_) {
        self.invalidate_stack_scalars();
        self.emit_line_hint();
        let callee = self.hostcall_addr(name).unwrap();
        let sig = self.import_sig(&[self.ptr, self.ptr, self.ptr], None);
        self.builder.ins().call_indirect(sig, callee, &[self.vm, a, b, out]);
    }

    /// `fn(vm, *const Value, *const Value, *const Value, *mut Value)` — e.g. host_slice_str.
    fn call_hostcall_3_val(&mut self, name: &str, a: Value_, b: Value_, c: Value_, out: Value_) {
        self.invalidate_stack_scalars();
        self.emit_line_hint();
        let callee = self.hostcall_addr(name).unwrap();
        let sig = self.import_sig(&[self.ptr, self.ptr, self.ptr, self.ptr], None);
        self.builder.ins().call_indirect(sig, callee, &[self.vm, a, b, c, out]);
    }

    /// `fn(vm, u64, u64, *const Value, *mut Value)` — e.g. host_call.
    fn call_hostcall_call(&mut self, name: &str, name_idx: u64, arg_count: u64, args: Value_, out: Value_) {
        self.invalidate_stack_scalars();
        self.emit_line_hint();
        let callee = self.hostcall_addr(name).unwrap();
        let sig = self.import_sig(&[types::I64, types::I64, self.ptr, self.ptr], None);
        let a1 = self.builder.ins().iconst(types::I64, name_idx as i64);
        let a2 = self.builder.ins().iconst(types::I64, arg_count as i64);
        self.builder.ins().call_indirect(sig, callee, &[self.vm, a1, a2, args, out]);
    }

    /// A1：JIT-to-JIT 直接调用（`Call`/`CallN` 的用户函数目标）。
    ///
    /// 目标函数 `callee_idx` 若已在函数指针表（`vm.jit_table_ptr[callee_idx]`）
    /// 注册（非 0）→ Cranelift `call_indirect` **直接调用**已编译机器码（参数从
    /// JIT 栈传、结果写 `out`，不逃逸 VM/解释器）；否则 → `host_jit_call`
    /// trampoline（运行时编译目标 chunk → 注册进表 → 直接调用；编译失败回退
    /// 解释器）。未编译函数保持既有 `host_call` 语义（正确性优先）。
    ///
    /// 正确性关键（两处）：
    /// 1. 被调函数内 hostcall（如 `PushStr`→`host_make_str`）依赖
    ///    `vm.current_chunk_idx` 定位字符串表——快路径在调用前保存调用方
    ///    chunk_idx、写入被调 chunk_idx，调用后恢复；递归/多级嵌套安全。
    /// 2. 慢路径 trampoline 需用调用方 chunk 的字符串表解析函数名（name_idx
    ///    是调用方字符串表索引），因此调用前**不得**切换 current_chunk_idx；
    ///    trampoline 内部自行保存/切换/恢复后再调被调函数。
    fn emit_direct_call(
        &mut self,
        callee_idx: usize,
        name_idx: usize,
        n: u64,
        args_addr: Value_,
        out: Value_,
    ) -> Result<(), String> {
        self.emit_line_hint();
        // A2：内联快路径——callee 满足内联条件时把 body 直插调用点（不 emit call）。
        // Ok(true) = 已内联；Ok(false) = 不满足条件 → 静默回退下方 A1 直接调用
        // （不报错、不改变行为）；Err = 内联中途失败（静态过滤后不应发生）→
        // 整个函数回退解释器（正确性优先）。
        match self.try_inline_call(callee_idx, n, args_addr, out)? {
            true => return Ok(()),
            false => { /* 静默回退 A1 */ }
        }
        let vm = self.vm;

        // ── 从函数指针表加载 callee 指针（0 = 未编译 → 慢路径）──
        let table_off = std::mem::offset_of!(Vm, jit_table_ptr) as i64;
        let table_off_v = self.builder.ins().iconst(self.ptr, table_off);
        let table_addr = self.builder.ins().iadd(vm, table_off_v);
        let table_base = self.builder.ins().load(self.ptr, MemFlags::new(), table_addr, 0);
        let entry_off_v = self.builder.ins().iconst(
            self.ptr,
            (callee_idx * std::mem::size_of::<usize>()) as i64,
        );
        let entry_addr = self.builder.ins().iadd(table_base, entry_off_v);
        let callee_ptr = self.builder.ins().load(self.ptr, MemFlags::new(), entry_addr, 0);

        let is_zero = self.builder.ins().icmp_imm(IntCC::Equal, callee_ptr, 0);
        let slow_blk = self.builder.create_block();
        let fast_blk = self.builder.create_block();
        let merge_blk = self.builder.create_block();
        self.builder.ins().brif(is_zero, slow_blk, &[], fast_blk, &[]);

        // 快路径：保存 chunk_idx → 切换为被调 chunk → 直接调用（JitFn 签名
        // (vm, args, n, out) -> bool）→ 恢复调用方 chunk_idx。
        self.builder.switch_to_block(fast_blk);
        self.builder.seal_block(fast_blk);
        let chunk_off = std::mem::offset_of!(Vm, current_chunk_idx) as i64;
        let chunk_off_v = self.builder.ins().iconst(self.ptr, chunk_off);
        let chunk_addr = self.builder.ins().iadd(vm, chunk_off_v);
        let saved_chunk = self.builder.ins().load(self.ptr, MemFlags::new(), chunk_addr, 0);
        let callee_idx_v = self.builder.ins().iconst(self.ptr, callee_idx as i64);
        self.builder.ins().store(MemFlags::new(), callee_idx_v, chunk_addr, 0);
        let sig = self.import_sig(&[self.ptr, self.ptr, self.ptr], Some(types::I8));
        let n_v = self.builder.ins().iconst(self.ptr, n as i64);
        self.builder.ins().call_indirect(sig, callee_ptr, &[vm, args_addr, n_v, out]);
        self.builder.ins().store(MemFlags::new(), saved_chunk, chunk_addr, 0);
        self.builder.ins().jump(merge_blk, &[]);

        // 慢路径：trampoline（current_chunk_idx 仍为调用方 → name_idx 可正确
        // 解析函数名；编译 + 注册 + 直接调用；失败 → 解释器回退）。
        self.builder.switch_to_block(slow_blk);
        self.builder.seal_block(slow_blk);
        self.call_hostcall_call("host_jit_call", name_idx as u64, n, args_addr, out);
        self.builder.ins().jump(merge_blk, &[]);

        // 汇合：继续执行。
        self.builder.switch_to_block(merge_blk);
        self.builder.seal_block(merge_blk);
        self.terminated = false;
        Ok(())
    }

    // ── A2：调用点内联（小函数）─────────────────────────────────────────────

    /// A2：尝试把 `callee_idx` 内联到当前调用点。
    ///
    /// - `Ok(true)`：已内联——被调函数 body 直插调用方机器码（不 emit call）。
    /// - `Ok(false)`：不满足内联条件（大小/opcode 白名单/参数个数/递归）→
    ///   静默回退，调用方继续走 A1 直接调用（不报错、不改变行为）。
    /// - `Err`：内联中途失败（静态过滤后不应发生）→ 整个函数回退解释器（保正确）。
    ///
    /// 被调 chunk 仍照常编译（供非内联调用点/间接引用使用）——内联不改变
    /// name_to_chunk/函数指针表/current_chunk_idx 机制。
    fn try_inline_call(
        &mut self,
        callee_idx: usize,
        n: u64,
        args_addr: Value_,
        out: Value_,
    ) -> Result<bool, String> {
        let callee = match self.all_chunks.get(callee_idx) {
            Some(c) => c,
            None => return Ok(false),
        };
        // 参数个数必须匹配（防止参数错位 → 静默错值）。
        if callee.num_args as u64 != n {
            return Ok(false);
        }
        let slot_count = match self.inline_analyze(callee) {
            Some(s) => s,
            None => return Ok(false),
        };
        self.translate_inline_body(callee, slot_count, args_addr, out)?;
        Ok(true)
    }

    /// A2：内联资格静态判定。`Some(内联栈槽数)` = 可内联；`None` = 不可内联。
    ///
    /// 保守条件（v1）：
    /// - 指令数 ≤ `INLINE_MAX_INSTR`（16）
    /// - 只含「平凡」opcode：纯标量构造/局部变量/算术/比较/控制流/Ret/MoveOp
    ///   （**无任何调用** → 无递归/自引用；无 PushStr/LoadGlobal/StoreGlobal →
    ///   不依赖 `current_chunk_idx` 字符串表，内联到调用方 chunk 上下文仍正确；
    ///   无 Tuple/Struct/Try/async/闭包/引用/cell → 无复杂 hostcall 语义）
    /// - `num_locals` 合理（≤ 64，避免每内联点创建过多栈槽）
    ///
    /// 内联栈槽数 = 指令数（每个 push 类指令至多使栈深 +1 的保守上界；已被
    /// `INLINE_MAX_INSTR` 限为 ≤16，远小于 `MAX_STACK_DEPTH`=64，bump_sp 不会
    /// 误触发，槽也足够容纳实际深度）。
    fn inline_analyze(&self, callee: &Chunk) -> Option<usize> {
        use Op::*;
        let mut ip = 0usize;
        let mut count = 0usize;
        let code_len = callee.code.len();
        while ip < code_len {
            let op = callee.read_op(&mut ip);
            count += 1;
            if count > INLINE_MAX_INSTR {
                return None;
            }
            let is_simple = matches!(op,
                PushInt(_) | PushFloat(_) | PushFloat32(_) | PushBool(_) | PushChar(_) | PushUnit
                | Pop | Dup | Load(_) | Store(_)
                | Add | Sub | Mul | Div | Mod | Neg | Not
                | Eq | Neq | Lt | Gt | Lte | Gte
                | Jump(_) | JmpFalse(_) | JmpTrue(_) | Ret | MoveOp
            );
            if !is_simple {
                return None;
            }
        }
        if callee.num_locals > 64 {
            return None;
        }
        Some(count.max(1))
    }

    /// A2：把被调函数 body 直插到调用方当前 block 之后（内联翻译）。
    ///
    /// 语义保持关键（逐指令保持 VM 语义）：
    /// - 被调函数用自己的 `StackSlot`（内联栈槽）与自己的 locals——绝不覆盖
    ///   调用方栈上活值（调用方 sp 处的参数/结果区域在调用方栈槽内，互不干扰）。
    /// - 参数从 `args_addr`（调用方栈区）复制到被调函数 locals 前 num_args 个。
    /// - `Ret` 写调用方 `out` 槽并跳 `inline_cont` 汇合块（不返回整个函数）。
    /// - `self.chunk` 切到被调 chunk：`cur_line` 取自被调 chunk 行号表，
    ///   内联函数内 hostcall 报错（溢出/除零/取模）携带正确行号（set_jit_error）。
    /// - 内联体只含平凡 opcode（白名单已保证），其 hostcall（算术/比较/构造）
    ///   不依赖 `current_chunk_idx` 字符串表 → 无需切换 chunk 上下文。
    /// - 内联体中的 `emit_err_check_abort`（binop 溢出检查）返回 ok=0 直接终止
    ///   调用方函数——与 VM 语义一致（callee 内错误传播到调用方）。
    fn translate_inline_body(
        &mut self,
        callee: &'a Chunk,
        slot_count: usize,
        args_addr: Value_,
        out: Value_,
    ) -> Result<(), String> {
        // ── 保存调用方状态 ──
        let saved_chunk = self.chunk;
        let saved_sp = self.sp;
        let saved_stack_slot = self.stack_slot;
        let saved_locals = std::mem::take(&mut self.locals);
        let saved_blocks = std::mem::take(&mut self.blocks);
        let saved_block_sp = std::mem::take(&mut self.block_sp);
        let saved_visited = std::mem::take(&mut self.visited);
        let saved_terminated = self.terminated;
        let saved_inline = (self.inline_out, self.inline_cont);
        let saved_cur_line = self.cur_line;
        // A2b：内联体禁用标量专用化（被调函数未做种类分析；调用方标量状态
        // 若泄漏到内联体，Load/Store 可能读错槽位 → 静默错值红线）。
        let saved_scalar_enabled = self.scalar_enabled;
        let saved_stack_scalars = std::mem::take(&mut self.stack_scalars);
        let saved_local_scalars = std::mem::take(&mut self.local_scalars);
        let saved_block_entry_kinds = std::mem::take(&mut self.block_entry_kinds);
        let saved_cur_local_kinds = std::mem::take(&mut self.cur_local_kinds);

        // ── 设置内联上下文 ──
        let cont = self.builder.create_block();
        self.chunk = callee;
        self.sp = 0;
        self.scalar_enabled = false;
        self.stack_slot = self.builder.create_sized_stack_slot(StackSlotData::new(
            StackSlotKind::ExplicitSlot,
            (VALUE_SIZE * slot_count as u32) as u32,
            8,
        ));
        self.inline_out = Some(out);
        self.inline_cont = Some(cont);
        self.terminated = false;

        // ── 初始化被调函数 locals（前 num_args 从 args_addr 复制）──
        let num_args = callee.num_args;
        let num_locals = callee.num_locals.max(num_args);
        for i in 0..num_locals {
            let slot = self.builder.create_sized_stack_slot(StackSlotData::new(
                StackSlotKind::ExplicitSlot,
                VALUE_SIZE,
                8,
            ));
            self.locals.insert(i, slot);
        }
        for i in 0..num_args {
            let dst = self.locals[&i];
            let src_off = (i as i32) * (VALUE_SIZE as i32);
            self.copy_ptr_to_slot(args_addr, src_off, dst, 0);
        }

        // ── 为被调函数创建 leader blocks ──
        let leaders = self.find_leaders();
        for &ip in &leaders {
            let blk = self.builder.create_block();
            self.blocks.insert(ip, blk);
        }

        // ── 线性翻译被调函数主体（与 translate_body 同构；Ret → 跳 cont）──
        let mut ip = 0usize;
        let code_len = callee.code.len();
        while ip < code_len {
            if let Some(&blk) = self.blocks.get(&ip) {
                if !self.terminated {
                    // A2b：顺序落入块边界——先物化栈标量（与 translate_body 一致）。
                    self.materialize_all_stack();
                    self.block_sp.insert(blk, self.sp);
                    self.builder.ins().jump(blk, &[]);
                }
                self.builder.switch_to_block(blk);
                self.builder.seal_block(blk);
                self.visited.insert(blk);
                if let Some(&sp) = self.block_sp.get(&blk) {
                    self.sp = sp;
                }
                self.terminated = false;
            }
            let op_start = ip;
            let op = self.chunk.read_op(&mut ip);
            self.emit_op(op, op_start, ip)?;
        }

        // ── 结尾未 terminated（函数自然走完无 Ret）：写 Unit 到 out 并跳 cont ──
        if !self.terminated {
            self.emit_inline_fallthrough(out);
        }

        // ── 填充未访问 merge blocks（与 translate_body 一致；未访问块 → Unit + 跳 cont）──
        let all_blocks: Vec<(usize, Block)> = self.blocks.iter().map(|(&k, &v)| (k, v)).collect();
        for (_ip, blk) in all_blocks {
            if !self.visited.contains(&blk) {
                self.sp = self.block_sp.get(&blk).copied().unwrap_or(0);
                self.builder.switch_to_block(blk);
                self.builder.seal_block(blk);
                self.visited.insert(blk);
                self.emit_inline_fallthrough(out);
            }
        }

        // ── 切到汇合块（内联 Ret 的跳转目标），恢复调用方状态 ──
        self.builder.switch_to_block(cont);
        self.builder.seal_block(cont);

        self.chunk = saved_chunk;
        self.sp = saved_sp;
        self.stack_slot = saved_stack_slot;
        self.locals = saved_locals;
        self.blocks = saved_blocks;
        self.block_sp = saved_block_sp;
        self.visited = saved_visited;
        self.terminated = false;
        self.inline_out = saved_inline.0;
        self.inline_cont = saved_inline.1;
        self.cur_line = saved_cur_line;
        self.scalar_enabled = saved_scalar_enabled;
        self.stack_scalars = saved_stack_scalars;
        self.local_scalars = saved_local_scalars;
        self.block_entry_kinds = saved_block_entry_kinds;
        self.cur_local_kinds = saved_cur_local_kinds;
        Ok(())
    }

    /// A2：内联函数自然走完（无 Ret）时——写 Unit 到 out 槽，跳 inline_cont。
    fn emit_inline_fallthrough(&mut self, out: Value_) {
        let tmp = self.builder.create_sized_stack_slot(StackSlotData::new(
            StackSlotKind::ExplicitSlot,
            VALUE_SIZE,
            8,
        ));
        let tmp_addr = self.builder.ins().stack_addr(self.ptr, tmp, 0);
        self.call_hostcall_unit("host_make_unit", tmp_addr);
        self.copy_slot_to_ptr(tmp, 0, out, 0);
        let cont = self.inline_cont.expect("inline fallthrough without cont");
        self.builder.ins().jump(cont, &[]);
        self.terminated = true;
    }

    /// `fn(vm, u64, *const Value, *mut Value)` — e.g. host_make_vec.
    fn call_hostcall_make_n(&mut self, name: &str, count: u64, args: Value_, out: Value_) {
        self.invalidate_stack_scalars();
        self.emit_line_hint();
        let callee = self.hostcall_addr(name).unwrap();
        let sig = self.import_sig(&[types::I64, self.ptr, self.ptr], None);
        let c = self.builder.ins().iconst(types::I64, count as i64);
        self.builder.ins().call_indirect(sig, callee, &[self.vm, c, args, out]);
    }

    /// `fn(vm, u64, u64, *const Value, *mut Value)` — e.g. host_new_struct.
    fn call_hostcall_new_struct(&mut self, name: &str, name_idx: u64, field_count: u64, args: Value_, out: Value_) {
        self.invalidate_stack_scalars();
        self.emit_line_hint();
        let callee = self.hostcall_addr(name).unwrap();
        let sig = self.import_sig(&[types::I64, types::I64, self.ptr, self.ptr], None);
        let a1 = self.builder.ins().iconst(types::I64, name_idx as i64);
        let a2 = self.builder.ins().iconst(types::I64, field_count as i64);
        self.builder.ins().call_indirect(sig, callee, &[self.vm, a1, a2, args, out]);
    }

    /// `fn(vm, u64, u64, *const Value, *mut Value)` — e.g. host_new_union.
    fn call_hostcall_new_union(&mut self, name: &str, name_idx: u64, field_idx: u64, val: Value_, out: Value_) {
        self.invalidate_stack_scalars();
        self.emit_line_hint();
        let callee = self.hostcall_addr(name).unwrap();
        let sig = self.import_sig(&[types::I64, types::I64, self.ptr, self.ptr], None);
        let a1 = self.builder.ins().iconst(types::I64, name_idx as i64);
        let a2 = self.builder.ins().iconst(types::I64, field_idx as i64);
        self.builder.ins().call_indirect(sig, callee, &[self.vm, a1, a2, val, out]);
    }

    /// `fn(vm, u64, u64, u64, *const Value, *mut Value)` — e.g. host_make_enum.
    fn call_hostcall_make_enum(&mut self, name: &str, name_idx: u64, variant_idx: u64, field_count: u64, args: Value_, out: Value_) {
        self.invalidate_stack_scalars();
        self.emit_line_hint();
        let callee = self.hostcall_addr(name).unwrap();
        let sig = self.import_sig(&[types::I64, types::I64, types::I64, self.ptr, self.ptr], None);
        let a1 = self.builder.ins().iconst(types::I64, name_idx as i64);
        let a2 = self.builder.ins().iconst(types::I64, variant_idx as i64);
        let a3 = self.builder.ins().iconst(types::I64, field_count as i64);
        self.builder.ins().call_indirect(sig, callee, &[self.vm, a1, a2, a3, args, out]);
    }

    /// `fn(vm, i64, i64, u8, *mut Value)` — e.g. host_push_range.
    fn call_hostcall_push_range(&mut self, name: &str, start: i64, end: i64, inc: u8, out: Value_) {
        self.invalidate_stack_scalars();
        self.emit_line_hint();
        let callee = self.hostcall_addr(name).unwrap();
        let sig = self.import_sig(&[types::I64, types::I64, types::I8, self.ptr], None);
        let a1 = self.builder.ins().iconst(types::I64, start);
        let a2 = self.builder.ins().iconst(types::I64, end);
        let a3 = self.builder.ins().iconst(types::I8, inc as i64);
        self.builder.ins().call_indirect(sig, callee, &[self.vm, a1, a2, a3, out]);
    }

    /// `fn(vm, u64, u64, *const Value, *mut Value)` — e.g. host_make_tensor.
    fn call_hostcall_make_tensor(&mut self, name: &str, rows: u64, cols: u64, args: Value_, out: Value_) {
        self.invalidate_stack_scalars();
        self.emit_line_hint();
        let callee = self.hostcall_addr(name).unwrap();
        let sig = self.import_sig(&[types::I64, types::I64, self.ptr, self.ptr], None);
        let a1 = self.builder.ins().iconst(types::I64, rows as i64);
        let a2 = self.builder.ins().iconst(types::I64, cols as i64);
        self.builder.ins().call_indirect(sig, callee, &[self.vm, a1, a2, args, out]);
    }

    /// `fn(vm, u64, *const Value, *const Value, *mut Value)` — e.g. host_store_field.
    fn call_hostcall_store_field(&mut self, name: &str, field_idx: u64, recv: Value_, val: Value_, out: Value_) {
        self.invalidate_stack_scalars();
        self.emit_line_hint();
        let callee = self.hostcall_addr(name).unwrap();
        let sig = self.import_sig(&[types::I64, self.ptr, self.ptr, self.ptr], None);
        let a1 = self.builder.ins().iconst(types::I64, field_idx as i64);
        self.builder.ins().call_indirect(sig, callee, &[self.vm, a1, recv, val, out]);
    }

    // ── Binop / unop ───────────────────────────────────────────────────────

    /// hostcall 报错检查：若 `host_check_error` 非零，立即返回 ok=0（run_jit 读取
    /// last_error 触发 fallback）。与 MethodCall 分支的 B2 模式一致——避免 binop
    /// 报错（如整数溢出）后继续执行、错误被后续操作覆盖。AUDIT-11.4.17。
    fn emit_err_check_abort(&mut self) {
        let err_flag = self.call_hostcall_vm_ret_u8("host_check_error");
        let has_err = self.builder.ins().icmp_imm(IntCC::NotEqual, err_flag, 0);
        let err_blk = self.builder.create_block();
        let cont_blk = self.builder.create_block();
        self.builder.ins().brif(has_err, err_blk, &[], cont_blk, &[]);
        // 错误路径：返回 ok=0（run_jit 会 take_last_error 并返回 Err）
        self.builder.switch_to_block(err_blk);
        self.builder.seal_block(err_blk);
        let ok_false = self.builder.ins().iconst(types::I8, 0);
        self.builder.ins().return_(&[ok_false]);
        // 继续路径
        self.builder.switch_to_block(cont_blk);
        self.builder.seal_block(cont_blk);
        self.terminated = false;
    }

    fn emit_binop(&mut self, op: Op, name: &str) -> Result<(), String> {
        // Stack: [..., a, b]. Pop b, then a. Out at a's position.
        self.sp -= VALUE_SIZE as i32;
        let b_off = self.sp;
        self.sp -= VALUE_SIZE as i32;
        let a_off = self.sp;
        // A2b：两操作数均为已知同类标量 → 原生路径（含 I32 溢出/范围/除零检查）。
        let specialized = if self.scalar_enabled {
            if let (Some(&(ak, _)), Some(&(bk, _))) = (
                self.stack_scalars.get(&a_off),
                self.stack_scalars.get(&b_off),
            ) {
                if ak == bk && ak != ScalarKind::Unknown && ak != ScalarKind::Top {
                    let native_ok = matches!(
                        (op, ak),
                        // I32：加/减/除/模；F64：加/减/乘/除（F64 模 VM 会报错，不专用化）
                        (Op::Add | Op::Sub | Op::Div | Op::Mod, ScalarKind::I32)
                            | (Op::Add | Op::Sub | Op::Mul | Op::Div, ScalarKind::F64)
                    );
                    let is_cmp = matches!(op, Op::Eq | Op::Neq | Op::Lt | Op::Gt | Op::Lte | Op::Gte);
                    if native_ok {
                        true
                    } else if is_cmp && (ak == ScalarKind::I32 || ak == ScalarKind::F64) {
                        true
                    } else {
                        false
                    }
                } else {
                    false
                }
            } else {
                false
            }
        } else {
            false
        };
        if specialized {
            let (ak, _) = self.stack_scalars[&a_off];
            if matches!(op, Op::Eq | Op::Neq | Op::Lt | Op::Gt | Op::Lte | Op::Gte) {
                return self.emit_native_cmp(op, a_off, b_off, ak);
            }
            return self.emit_native_binop(op, a_off, b_off, ak);
        }
        // 通用路径：先物化两操作数（Value 槽有效），再 hostcall。
        self.materialize_stack_range(a_off, 2);
        let a_addr = self.builder.ins().stack_addr(self.ptr, self.stack_slot, a_off);
        let b_addr = self.builder.ins().stack_addr(self.ptr, self.stack_slot, b_off);
        let out = a_addr; // result overwrites a's slot
        self.call_hostcall_2_val(name, a_addr, b_addr, out);
        self.bump_sp()?;
        self.emit_err_check_abort();
        Ok(())
    }

    fn emit_unop(&mut self, op: Op, name: &str) -> Result<(), String> {
        self.sp -= VALUE_SIZE as i32;
        let a_off = self.sp;
        // A2b：操作数为已知标量 → 原生路径（Neg: I32/F64；Not: Bool）。
        let specialized = if self.scalar_enabled {
            if let Some(&(ak, _)) = self.stack_scalars.get(&a_off) {
                let native_ok = matches!(
                    (op, ak),
                    (Op::Neg, ScalarKind::I32)
                        | (Op::Neg, ScalarKind::F64)
                        | (Op::Not, ScalarKind::Bool)
                );
                ak != ScalarKind::Unknown && ak != ScalarKind::Top && native_ok
            } else {
                false
            }
        } else {
            false
        };
        if specialized {
            let (ak, _) = self.stack_scalars[&a_off];
            return self.emit_native_unop(op, a_off, ak);
        }
        // 通用路径：先物化操作数。
        self.materialize_stack_at(a_off);
        let a_addr = self.builder.ins().stack_addr(self.ptr, self.stack_slot, a_off);
        let out = a_addr; // result overwrites operand slot
        self.emit_line_hint();
        let callee = self.hostcall_addr(name)?;
        let sig = self.import_sig(&[self.ptr, self.ptr], None);
        self.builder.ins().call_indirect(sig, callee, &[self.vm, a_addr, out]);
        self.bump_sp()?;
        self.emit_err_check_abort();
        Ok(())
    }
}
