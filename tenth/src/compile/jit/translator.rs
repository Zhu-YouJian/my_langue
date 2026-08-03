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

use crate::compile::jit::context::{ChunkSig, ScalarAbiKind, MAX_SPEC_ARGS};
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

// ── M2.5-A6：特化 ABI（scalar-specialized call）────────────────────────────

/// CallN/Call 目标是否可走特化 ABI（**分析期与发射期共用**，防静默错值漂移）。
///
/// 条件（保守）：
/// - 目标 chunk 有特化签名且参数个数 = 实参个数 = chunk 声明参数个数
///   （防默认参数/可变参数错位）
/// - 实参个数 ≤ MAX_SPEC_ARGS
/// - 签名种类为 I64/F64 混合（v1 仅 I64；P3 扩展 F64，按参数顺序声明）
///
/// 注意：此谓词只判「目标侧资格」；调用侧「实参是否全为对应标量槽」由
/// 分析期/发射期各自的栈模型判定（二者对应同一运行时状态）。
fn spec_target_sig<'a>(
    name_to_chunk: &[Option<usize>],
    chunk_sigs: &'a [Option<ChunkSig>],
    all_chunks: &[Chunk],
    name_i: usize,
    n: usize,
) -> Option<&'a ChunkSig> {
    if n > MAX_SPEC_ARGS {
        return None;
    }
    let callee_idx = match name_to_chunk.get(name_i).copied().flatten() {
        Some(i) => i,
        None => return None,
    };
    let sig = match chunk_sigs.get(callee_idx) {
        Some(Some(s)) => s,
        _ => return None,
    };
    if sig.param_kinds.len() != n {
        return None;
    }
    // 参数个数与 chunk 声明一致（防默认参数调用点实参 < 声明参数数）
    match all_chunks.get(callee_idx) {
        Some(c) if c.num_args == n => Some(sig),
        _ => None,
    }
}

fn spec_target_qualifies(
    name_to_chunk: &[Option<usize>],
    chunk_sigs: &[Option<ChunkSig>],
    all_chunks: &[Chunk],
    name_i: usize,
    n: usize,
) -> bool {
    spec_target_sig(name_to_chunk, chunk_sigs, all_chunks, name_i, n).is_some()
}

/// A2：内联资格静态判定（**分析期与发射期共用**，防静默错值漂移）。
///
/// 条件（与发射端 `try_inline_call` 的 `inline_analyze` 逐一对应）：
/// - 参数个数匹配（`num_args == n`；发射端在 `try_inline_call` 单独检查同一条件）
/// - `num_locals` ≤ 64（避免每内联点创建过多栈槽）
/// - 指令数 ≤ `INLINE_MAX_INSTR`（16），且只含「平凡」opcode：纯标量构造/局部变量/
///   算术/比较/控制流/Ret/MoveOp（**无任何调用** → 无递归/自引用；无 PushStr/
///   LoadGlobal/StoreGlobal → 不依赖 `current_chunk_idx` 字符串表；无 Tuple/Struct/
///   Try/async/闭包/引用/cell → 无复杂 hostcall 语义）
///
/// 内联体只含白名单 opcode → 其 hostcall（算术/比较/构造/`set_*_error`）均不依赖
/// `current_chunk_idx` 字符串表（行号由 `vm.current_line` 携带）→ 内联到调用方
/// chunk 上下文仍正确。此性质同时是 P1「跳过 chunk 切换」判定的基石。
fn inline_eligible(all_chunks: &[Chunk], callee_idx: usize, n: usize) -> bool {
    use Op::*;
    let callee = match all_chunks.get(callee_idx) {
        Some(c) => c,
        None => return false,
    };
    if callee.num_args != n {
        return false;
    }
    if callee.num_locals > 64 {
        return false;
    }
    let mut ip = 0usize;
    let mut count = 0usize;
    while ip < callee.code.len() {
        let op = callee.read_op(&mut ip);
        count += 1;
        if count > INLINE_MAX_INSTR {
            return false;
        }
        let is_simple = matches!(op,
            PushInt(_) | PushFloat(_) | PushFloat32(_) | PushBool(_) | PushChar(_) | PushUnit
            | Pop | Dup | Load(_) | Store(_)
            | Add | Sub | Mul | Div | Mod | Neg | Not
            | Eq | Neq | Lt | Gt | Lte | Gte
            | Jump(_) | JmpFalse(_) | JmpTrue(_) | Ret | MoveOp
        );
        if !is_simple {
            return false;
        }
    }
    true
}

/// P1：判定特化函数 body 是否「纯标量、绝不读取 `current_chunk_idx` 字符串表」。
///
/// 判定（静默错值红线，AUDIT-11.4.35 分析/发射共用精神）：
/// - 全部 op 属安全白名单（= 内联白名单：纯标量构造/局部/算术/比较/控制流；
///   `set_*_error` 错误链 hostcall 不依赖字符串表——行号由 `vm.current_line`
///   携带，A2 已做）
/// - 每个 Call/CallN 站点：目标可内联（内联体白名单保证不读 chunk ctx）**或**
///   分析期预测走特化（`call_spec[ip] == true`）。特化调用**总是安全**：
///   * 目标也为纯标量 → 嵌套站点也跳过切换 → 被调方不读 chunk ctx（归纳）
///   * 目标非纯标量 → 嵌套站点自带 save/switch/restore（自洽，无论入口 chunk）
///   * 慢路径 `host_jit_call_spec` → `Vm::jit_call_chunk_spec` 自带 save/switch/restore
/// - 其余 op（PushStr/LoadGlobal/StoreGlobal/MethodCall/张量/闭包/引用/async 等）
///   → 不跳过（保持现状，零变化）
///
/// 满足 → 特化调用点可跳过 current_chunk_idx 保存/切换/恢复（快路径零 chunk 读写）。
fn spec_body_pure_scalar(
    chunk: &Chunk,
    name_to_chunk: &[Option<usize>],
    all_chunks: &[Chunk],
    call_spec: &HashMap<usize, bool>,
) -> bool {
    use Op::*;
    let mut ip = 0usize;
    let len = chunk.code.len();
    while ip < len {
        let start = ip;
        let op = chunk.read_op(&mut ip);
        let safe = matches!(op,
            PushInt(_) | PushFloat(_) | PushFloat32(_) | PushBool(_) | PushChar(_) | PushUnit
            | Pop | Dup | Load(_) | Store(_)
            | Add | Sub | Mul | Div | Mod | Neg | Not
            | Eq | Neq | Lt | Gt | Lte | Gte
            | Jump(_) | JmpFalse(_) | JmpTrue(_) | Ret | MoveOp
        );
        if safe {
            continue;
        }
        // Call/CallN：仅当（内联 OR 预测特化）时安全，否则不跳过。
        let ok = match op {
            Call(i) => {
                let inline = match name_to_chunk.get(i).copied().flatten() {
                    Some(ci) => inline_eligible(all_chunks, ci, 0),
                    None => false,
                };
                inline || call_spec.get(&start).copied().unwrap_or(false)
            }
            CallN(i, n) => {
                let inline = match name_to_chunk.get(i).copied().flatten() {
                    Some(ci) => inline_eligible(all_chunks, ci, n),
                    None => false,
                };
                inline || call_spec.get(&start).copied().unwrap_or(false)
            }
            _ => false,
        };
        if !ok {
            return false;
        }
    }
    true
}

/// P1：计算「纯标量、可跳过 current_chunk_idx 切换」的 chunk 集合。
///
/// 对每个带特化签名（`scalar_sig`）的 chunk，用与发射期**完全相同**的分析
/// （`analyze_scalar_kinds`，含 spec seed）获取 `call_spec_results`，再按
/// `spec_body_pure_scalar` 判定。判定是「chunk 自身字节码 + 静态信息」的纯函数
/// （不依赖调用点/编译顺序/编译缓存）→ 分析期与发射期共用同一份数据，无漂移。
///
/// 返回 `skip_ctx[chunk_idx]`：true = 该 chunk 的特化调用点可跳过
/// save/switch/restore（被调方体内不读字符串表）。
pub fn compute_skip_chunk_ctx(
    all_chunks: &[Chunk],
    name_to_chunk: &HashMap<String, usize>,
    chunk_sigs: &[Option<ChunkSig>],
) -> Vec<bool> {
    let n = all_chunks.len();
    let mut skip = vec![false; n];
    for i in 0..n {
        let chunk = &all_chunks[i];
        let sig = match &chunk.scalar_sig {
            Some(s) => s.clone(),
            None => continue,
        };
        // 与发射端 get_or_compile* 一致的 per-chunk 字符串表 → 目标 chunk 映射。
        let name_to_chunk: Vec<Option<usize>> = chunk.strings.iter()
            .map(|s| name_to_chunk.get(s).copied())
            .collect();
        // 与发射期 translate_body 完全相同的 spec seed（参数种类预置）。
        let seed: Vec<ScalarKind> = sig.param_kinds.iter().map(|k| match k {
            ScalarAbiKind::I64 => ScalarKind::I32,
            ScalarAbiKind::F64 => ScalarKind::F64,
        }).collect();
        let analysis =
            analyze_scalar_kinds(chunk, &name_to_chunk, chunk_sigs, all_chunks, Some(&seed));
        let pure = match &analysis {
            Some(a) => spec_body_pure_scalar(chunk, &name_to_chunk, all_chunks, &a.call_spec_results),
            None => false,
        };
        if pure {
            skip[i] = true;
        }
    }
    skip
}

/// A2b 标量分析结果（块入口局部种类 + 各 CallN 是否预测走特化）。
struct ScalarAnalysis {
    /// leader IP → 块入口处各局部种类（按局部索引）。
    entry_kinds: HashMap<usize, Vec<ScalarKind>>,
    /// CallN/Call 指令偏移 → 分析期是否预测特化（发射期防漂移护栏）。
    call_spec_results: HashMap<usize, bool>,
}

pub fn translate<M: Module>(
    module: &mut M,
    chunk_idx: usize,
    chunk: &Chunk,
    name_to_chunk: &[Option<usize>],
    all_chunks: &[Chunk],
    chunk_sigs: &[Option<ChunkSig>],
    spec: Option<&ChunkSig>,
    skip_chunk_ctx: &[bool],
) -> Result<cranelift_module::FuncId, String> {
    let is_spec = spec.is_some();
    let mut ctx = module.make_context();
    let mut fn_ctx = FunctionBuilderContext::new();

    let ptr = module.target_config().pointer_type();
    ctx.func.signature.params.push(AbiParam::new(ptr)); // vm
    if is_spec {
        // M2.5-A6：特化入口签名 `(vm, i64 x MAX_SPEC_ARGS) -> i64`——参数/返回走寄存器。
        for _ in 0..MAX_SPEC_ARGS {
            ctx.func.signature.params.push(AbiParam::new(types::I64));
        }
        ctx.func.signature.returns.push(AbiParam::new(types::I64)); // 标量返回
    } else {
        ctx.func.signature.params.push(AbiParam::new(ptr)); // args
        ctx.func.signature.params.push(AbiParam::new(ptr)); // n (usize, pointer-sized)
        ctx.func.signature.params.push(AbiParam::new(ptr)); // out
        ctx.func.signature.returns.push(AbiParam::new(types::I8)); // bool
    }

    let func_name = if is_spec {
        format!("__tenth_jit_spec_{}", chunk_idx)
    } else {
        format!("__tenth_jit_chunk_{}", chunk_idx)
    };
    let func_id = module.declare_function(
        &func_name,
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
        let (args_ptr, out_ptr, spec_args) = if is_spec {
            // 特化入口：参数走寄存器；args/out 指针未用（占位 0）。
            let zero = builder.ins().iconst(ptr, 0);
            let spec_args: Vec<Value_> =
                (0..MAX_SPEC_ARGS).map(|i| builder.block_params(entry)[1 + i]).collect();
            (zero, zero, spec_args)
        } else {
            let args_ptr = builder.block_params(entry)[1];
            let _args_n = builder.block_params(entry)[2];
            let out_ptr = builder.block_params(entry)[3];
            (args_ptr, out_ptr, Vec::new())
        };

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
            chunk_sigs,
            skip_chunk_ctx,
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
            spec_mode: is_spec,
            spec_args,
            spec_sig: spec.cloned(),
            call_spec_results: HashMap::new(),
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
    /// M2.5-A6：chunk_idx → 特化签名（None = 非特化）。调用点判定特化 ABI 资格。
    chunk_sigs: &'a [Option<ChunkSig>],
    /// P1：chunk_idx → 是否「纯标量、可跳过 current_chunk_idx 切换」（由
    /// `compute_skip_chunk_ctx` 在 run_jit 设置，发射期只读）。特化调用点据此
    /// 决定快路径是否省去 save/switch/restore 3 次内存操作。
    skip_chunk_ctx: &'a [bool],
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
    /// M2.5-A6：本函数是否为特化入口（签名 `(vm, i64 x MAX_SPEC_ARGS) -> i64`）。
    /// 特化模式：入口参数走寄存器、Ret/错误返回哨兵 i64。
    spec_mode: bool,
    /// M2.5-A6：特化入口的寄存器参数（长度 = MAX_SPEC_ARGS，未用参数为 0）。
    spec_args: Vec<Value_>,
    /// M2.5-A6：本函数特化签名（spec_mode 时 Some）。
    spec_sig: Option<ChunkSig>,
    /// M2.5-A6：分析期预测各 CallN/Call 是否走特化（发射期防漂移护栏：预测特化
    /// 而发射期不可特化 → Err → 整函数回退解释器，杜绝静默错值）。
    call_spec_results: HashMap<usize, bool>,
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

        // ── A2b/A6：局部变量标量种类分析（跨块 must 分析）先行 ──────
        // 分析成功 → 启用标量专用化（Load/Store/算术/比较走原生路径）；
        // 失败（含分析未知 opcode）→ 全函数保持既有通用路径（零行为变化）。
        // A6 特化入口：参数种类**预置**（seed）——i64 参数视为 I32 标量 →
        // 函数体参数 Load/算术走原生路径（fib 递归全链路标量的关键）。
        let seed: Option<Vec<ScalarKind>> = if self.spec_mode {
            let sig = self.spec_sig.as_ref().unwrap();
            Some(sig.param_kinds.iter().map(|k| match k {
                ScalarAbiKind::I64 => ScalarKind::I32,
                ScalarAbiKind::F64 => ScalarKind::F64,
            }).collect())
        } else {
            None
        };
        if let Some(analysis) = analyze_scalar_kinds(
            self.chunk, self.name_to_chunk, self.chunk_sigs, self.all_chunks, seed.as_deref(),
        ) {
            self.scalar_enabled = true;
            self.block_entry_kinds = analysis.entry_kinds;
            self.call_spec_results = analysis.call_spec_results;
            let n = self.chunk.num_locals.max(self.chunk.num_args);
            self.cur_local_kinds = vec![ScalarKind::Unknown; n];
        }

        // ── Initialise locals ─────────────────────────────────────────────
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
        if self.spec_mode {
            // A6 特化入口：寄存器参数 → 局部标量槽（8B）+ 按需物化 Value 槽。
            self.init_spec_args();
        } else {
            for i in 0..num_args {
                let dst = self.locals[&i];
                let src_off = (i as i32) * (VALUE_SIZE as i32);
                self.copy_ptr_to_slot(self.args_ptr, src_off, dst, 0);
            }
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
            let start = ip;
            let op = self.chunk.read_op(&mut ip);
            if let Op::Jump(o) | Op::JmpFalse(o) | Op::JmpTrue(o) = op {
                let target = ((ip as i64) + (o as i64)) as usize;
                leaders.push(target);
                leaders.push(ip); // instruction after the jump is a leader
            }
            // A6 修复：Ret 的 leader 应为其**起始偏移**（此前记录结束偏移 → emit 循环
            // 在 Ret 起始处查不到块 → TailCall 后的尾 Ret 被发射进已终止块 → verifier
            // 双重 return 错误；通用路径被整函数 fallback 掩盖）。
            if matches!(op, Op::Ret) {
                leaders.push(start);
            }
        }
        leaders.sort();
        leaders.dedup();
        leaders
    }
}

// ── A2b：标量专用化——跨块 must 分析 ─────────────────────────────────────

    /// A2b：跨块 must 分析——每个块入口处各局部变量的标量种类。
    /// 返回 leader IP → 入口种类（按局部索引）；`None` = 函数含分析未知 opcode
    /// （或块图不完整）→ 禁用专用化（全函数走既有通用路径，零行为变化）。
    ///
    /// 正确性（静默错值红线）：种类 I32/F64/Bool 意味着「该局部在所有到达该点的
    /// 路径上恒为该标量」——块 0（函数入口，含参数）强制 Unknown；其余块从乐观
    /// 顶 Top 出发，经 meet（跨前驱求交）收敛到最大不动点（GFP）。栈值在块内按
    /// 确定性模拟；通用 opcode 清空栈标量信息（与发射端 `call_hostcall_*` 清栈
    /// 一致）；块入口的栈值不可知（视 Unknown，安全保守）。
    ///
    /// M2.5-A6：`seed` 为特化入口的参数种类预置（块 0 前 num_args 个局部据此
    /// 初始化，其余 Unknown）。同时记录各 CallN/Call 是否预测走特化 ABI
    /// （`call_spec_results`，发射期防漂移护栏）。
    ///
    /// P1：已提升为自由函数（入参 = chunk + 静态上下文），供发射期（translate_body）
    /// 与跳过判定（`compute_skip_chunk_ctx`）共用同一份代码——防分析/发射漂移。
    fn analyze_scalar_kinds(
        chunk: &Chunk,
        name_to_chunk: &[Option<usize>],
        chunk_sigs: &[Option<ChunkSig>],
        all_chunks: &[Chunk],
        seed: Option<&[ScalarKind]>,
    ) -> Option<ScalarAnalysis> {
        use ScalarKind::*;
        use Op::*;
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

        // leaders（与 find_leaders 一致；A6 修复：Ret 的 leader 为起始偏移）
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
                leaders.push(ip);
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
        #[allow(clippy::too_many_arguments)]
        fn transfer(
            insns: &[(usize, Op, usize)],
            block_first: &[usize],
            nblocks: usize,
            bi: usize,
            entry: &[ScalarKind],
            call_spec: &mut HashMap<usize, bool>,
            name_to_chunk: &[Option<usize>],
            chunk_sigs: &[Option<ChunkSig>],
            all_chunks: &[Chunk],
        ) -> Option<Vec<ScalarKind>> {
            use ScalarKind::*;
            use Op::*;

            // A2-AUDIT-11.4.35：内联资格静态判定（与发射端 `inline_analyze`/
            // `try_inline_call` 条件逐一对应，防分析/发射漂移）。可内联调用在发射端
            // **优先于** A6 特化（`emit_direct_call` 先 try_inline_call 再 try_spec_call），
            // 其结果是 Value（非标量寄存器）——因此分析期必须把可内联调用预测为
            // Unknown。若仍预测 I32/spec，循环回边处分析把局部重置为 I32 而局部标量槽
            // 残留过期值（一般 Store 只写 Value 槽）→ Load 重专用化读过期标量 → 静默错值。
            // P1：谓词已提升为模块级 `inline_eligible`（分析/发射/跳过判定三方共用，
            // 单一来源防漂移）。

            let mut locals = entry.to_vec();
            let mut stack: Vec<ScalarKind> = Vec::new();
            let end = if bi + 1 < nblocks { block_first[bi + 1] } else { insns.len() };
            let start = block_first[bi];
            for idx in start..end {
                let (ip, ref op, _) = insns[idx];
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
                    // P2-AUDIT-11.4.36：分析/发射专用化集合**逐 opcode 对齐**
                    // （防 spec 预测漂移 → 发射端缺标量槽 → 整函数回退/潜在静默错值）。
                    // 发射端 emit_binop 的 native_ok 集合：
                    //   I32：Add/Sub/Div/Mod（Mul 不专用化——checked_mul 复杂，保 hostcall）
                    //   F64：Add/Sub/Mul/Div（Mod 不专用化——VM 会报错）
                    // 故分析只对「发射端会专用化」的组合预测标量，其余 Unknown。
                    Add | Sub | Div => {
                        let b = stack.pop().unwrap_or(Unknown);
                        let a = stack.pop().unwrap_or(Unknown);
                        stack.push(match (a, b) {
                            (I32, I32) => I32,
                            (F64, F64) => F64,
                            _ => Unknown,
                        });
                    }
                    Mul => {
                        let b = stack.pop().unwrap_or(Unknown);
                        let a = stack.pop().unwrap_or(Unknown);
                        stack.push(match (a, b) {
                            // 发射端 F64 Mul 专用化 → 预测 F64
                            (F64, F64) => F64,
                            // 发射端 I32 Mul **不**专用化（hostcall 路径）→ Unknown
                            _ => Unknown,
                        });
                    }
                    Mod => {
                        let b = stack.pop().unwrap_or(Unknown);
                        let a = stack.pop().unwrap_or(Unknown);
                        stack.push(match (a, b) {
                            // 发射端 I32 Mod 专用化 → 预测 I32
                            (I32, I32) => I32,
                            // 发射端 F64 Mod **不**专用化（VM 报错）→ Unknown
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
                    Call(i) => {
                        // M2.5-A6/P3：目标可特化（0 参 i64/f64 返回）→ 结果预测按
                        // ret_kind（I64→I32 / F64→F64）；否则 Unknown + 清栈
                        // （与发射端 host_call 失效一致）。
                        // A2-11.4.35：可内联调用（内联优先于特化）→ 预测 Unknown。
                        let inline = match name_to_chunk.get(*i).copied().flatten() {
                            Some(ci) => inline_eligible(all_chunks, ci, 0),
                            None => false,
                        };
                        let spec_sig = if inline {
                            None
                        } else {
                            spec_target_sig(name_to_chunk, chunk_sigs, all_chunks, *i, 0)
                        };
                        let spec = spec_sig.is_some();
                        call_spec.insert(ip, spec);
                        if spec {
                            let rk = spec_sig.unwrap().ret_kind.unwrap();
                            stack.push(match rk {
                                ScalarAbiKind::I64 => I32,
                                ScalarAbiKind::F64 => F64,
                            });
                        } else {
                            stack.push(Unknown);
                            clear_stack(&mut stack);
                        }
                    }
                    CallN(i, n) => {
                        // M2.5-A6/P3：实参种类逐个匹配目标签名 param_kinds
                        // （I64→实参须 I32、F64→实参须 F64）且目标可特化 → 结果预测
                        // 按 ret_kind（I64→I32 / F64→F64，特化调用后接原生 fadd 的
                        // 关键）；否则 Unknown + 清栈。
                        // A2-11.4.35：可内联调用（内联优先于特化）→ 预测 Unknown，
                        // 与发射端内联路径一致（否则循环回边 Load 读过期标量 → 静默错值）。
                        let mut arg_kinds: Vec<ScalarKind> = Vec::with_capacity(*n);
                        for _ in 0..*n {
                            arg_kinds.push(stack.pop().unwrap_or(Unknown));
                        }
                        arg_kinds.reverse(); // 栈弹出逆序 → 还原为 [arg0..argN-1]
                        let inline = match name_to_chunk.get(*i).copied().flatten() {
                            Some(ci) => inline_eligible(all_chunks, ci, *n as usize),
                            None => false,
                        };
                        let spec_sig = if inline {
                            None
                        } else {
                            spec_target_sig(name_to_chunk, chunk_sigs, all_chunks, *i, *n as usize)
                        };
                        let mut args_match = true;
                        if let Some(s) = spec_sig {
                            for (pos, want) in s.param_kinds.iter().enumerate() {
                                let need = match want {
                                    ScalarAbiKind::I64 => I32,
                                    ScalarAbiKind::F64 => F64,
                                };
                                if arg_kinds.get(pos).copied().unwrap_or(Unknown) != need {
                                    args_match = false;
                                    break;
                                }
                            }
                        }
                        let spec = spec_sig.is_some() && args_match;
                        call_spec.insert(ip, spec);
                        if spec {
                            let rk = spec_sig.unwrap().ret_kind.unwrap();
                            stack.push(match rk {
                                ScalarAbiKind::I64 => I32,
                                ScalarAbiKind::F64 => F64,
                            });
                        } else {
                            stack.push(Unknown);
                            clear_stack(&mut stack);
                        }
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

        // GFP：block0 = seed（特化入口预置参数种类）或 Unknown（参数/未初始化）；
        // 其余 = Top（乐观顶）
        let mut entry_kinds: Vec<Vec<ScalarKind>> = vec![vec![Top; num_locals]; nblocks];
        let mut e0 = vec![Unknown; num_locals];
        if let Some(seed) = seed {
            for (i, k) in seed.iter().enumerate() {
                if i < num_locals {
                    e0[i] = *k;
                }
            }
        }
        entry_kinds[0] = e0;
        let mut call_spec: HashMap<usize, bool> = HashMap::new();
        let mut worklist: VecDeque<usize> = (0..nblocks).collect();
        let mut in_work = vec![true; nblocks];
        while let Some(bi) = worklist.pop_front() {
            in_work[bi] = false;
            let entry = entry_kinds[bi].clone();
            let exit = match transfer(
                &insns, &block_first, nblocks, bi, &entry,
                &mut call_spec, name_to_chunk, chunk_sigs, all_chunks,
            ) {
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
        Some(ScalarAnalysis {
            entry_kinds: result,
            call_spec_results: call_spec,
        })
}

impl<'a, M: Module> Translator<'a, M> {
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

    // ── M2.5-A6：特化 ABI 辅助 ──────────────────────────────────────────────

    /// 从标量槽读 8 字节为 i64（I32 种类槽；特化 ABI 以 i64 寄存器传参）。
    fn load_scalar_i64(&mut self, slot: StackSlot) -> Value_ {
        self.builder.ins().stack_load(types::I64, slot, 0)
    }

    /// 写 i64 到标量槽（8 字节；特化返回/参数）。
    fn store_scalar_i64(&mut self, slot: StackSlot, v: Value_) {
        self.builder.ins().stack_store(v, slot, 0);
    }

    /// P3：写 f64 到标量槽（8 字节位模式；特化 F64 返回/参数位打包）。
    fn store_scalar_f64(&mut self, slot: StackSlot, v: Value_) {
        self.builder.ins().stack_store(v, slot, 0);
    }

    /// `fn(vm, *const Value) -> i64` — 从 Value 解包 i64（特化返回/慢路径结果）。
    fn call_hostcall_val_ret_i64(&mut self, name: &str, arg: Value_) -> Value_ {
        self.invalidate_stack_scalars();
        self.emit_line_hint();
        let callee = self.hostcall_addr(name).unwrap();
        let sig = self.import_sig(&[self.ptr], Some(types::I64));
        let call = self.builder.ins().call_indirect(sig, callee, &[self.vm, arg]);
        self.builder.inst_results(call)[0]
    }

    /// A6：特化慢路径 hostcall（`host_jit_call_spec`）。**不失效栈标量**——
    /// 被调方只写 out 槽/VM 内部，不触碰调用方标量槽。失效会清除**其他活标量**
    /// 的跟踪 → 后续通用消费者读陈旧 Value 槽 → 静默错值（A6 递归调试发现：
    /// 递归第 1 个 fib 结果在第二次调用后跟踪被清 → Add 读错值）。
    fn call_hostcall_spec_call(&mut self, name: &str, name_idx: u64, arg_count: u64, args: Value_, out: Value_) {
        self.emit_line_hint();
        let callee = self.hostcall_addr(name).unwrap();
        let sig = self.import_sig(&[types::I64, types::I64, self.ptr, self.ptr], None);
        let a1 = self.builder.ins().iconst(types::I64, name_idx as i64);
        let a2 = self.builder.ins().iconst(types::I64, arg_count as i64);
        self.builder.ins().call_indirect(sig, callee, &[self.vm, a1, a2, args, out]);
    }

    /// A6：特化返回值解包（`host_value_to_i64`）。**不失效栈标量**（同 `call_hostcall_spec_call`）。
    fn call_hostcall_val_ret_i64_noinv(&mut self, name: &str, arg: Value_) -> Value_ {
        self.emit_line_hint();
        let callee = self.hostcall_addr(name).unwrap();
        let sig = self.import_sig(&[self.ptr], Some(types::I64));
        let call = self.builder.ins().call_indirect(sig, callee, &[self.vm, arg]);
        self.builder.inst_results(call)[0]
    }

    /// P3：特化 F64 返回值解包（`host_value_to_f64`）。**失效栈标量**（同
    /// `call_hostcall_val_ret_i64`）——用于特化入口 Ret 的 fallback 路径。
    fn call_hostcall_val_ret_f64(&mut self, name: &str, arg: Value_) -> Value_ {
        self.invalidate_stack_scalars();
        self.emit_line_hint();
        let callee = self.hostcall_addr(name).unwrap();
        let sig = self.import_sig(&[self.ptr], Some(types::F64));
        let call = self.builder.ins().call_indirect(sig, callee, &[self.vm, arg]);
        self.builder.inst_results(call)[0]
    }

    /// P3：特化 F64 返回值解包（`host_value_to_f64`）。**不失效栈标量**（同
    /// `call_hostcall_val_ret_i64_noinv`）——特化调用慢路径用。
    fn call_hostcall_val_ret_f64_noinv(&mut self, name: &str, arg: Value_) -> Value_ {
        self.emit_line_hint();
        let callee = self.hostcall_addr(name).unwrap();
        let sig = self.import_sig(&[self.ptr], Some(types::F64));
        let call = self.builder.ins().call_indirect(sig, callee, &[self.vm, arg]);
        self.builder.inst_results(call)[0]
    }

    /// P3：`fn(vm, f64, *mut Value)` — 写 f64 Value 到任意地址（特化入口 F64 参数物化）。
    fn call_hostcall_f64_ptr(&mut self, name: &str, arg: Value_, out: Value_) {
        self.invalidate_stack_scalars();
        self.emit_line_hint();
        let callee = self.hostcall_addr(name).unwrap();
        let sig = self.import_sig(&[types::F64, self.ptr], None);
        self.builder.ins().call_indirect(sig, callee, &[self.vm, arg, out]);
    }

    /// A6：特化调用后的错误检查（`host_check_error`）。**不失效栈标量**（同上）。
    fn call_hostcall_check_error(&mut self) -> Value_ {
        self.emit_line_hint();
        let callee = self.hostcall_addr("host_check_error").unwrap();
        let sig = self.import_sig(&[], Some(types::I8));
        let call = self.builder.ins().call_indirect(sig, callee, &[self.vm]);
        self.builder.inst_results(call)[0]
    }

    /// P1：`host_check_error` 内联化——直接 load `vm.jit_error_flag`（I8），
    /// 返回错误标志（0 = 无错误）。与 `host_check_error`（= `has_last_error()`）
    /// **语义恒等**：`jit_error_flag` 由 `set_last_error`/`set_jit_error` 置 1、
    /// `take_last_error` 清 0，是 `last_error.is_some()` 的布尔镜像（last_error
    /// 为私有字段，全部写入仅经这三方法）。省一次 call_indirect 函数调用；
    /// 不做 emit_line_hint（行号仅 hostcall 报错时需要，检查本身不需要）。
    fn inline_check_error_flag(&mut self) -> Value_ {
        let off = std::mem::offset_of!(Vm, jit_error_flag) as i64;
        let off_v = self.builder.ins().iconst(self.ptr, off);
        let addr = self.builder.ins().iadd(self.vm, off_v);
        self.builder.ins().load(types::I8, MemFlags::new(), addr, 0)
    }

    /// `fn(vm, i64, *mut Value)` — 写标量 Value 到任意地址（特化入口参数物化）。
    fn call_hostcall_i64_ptr(&mut self, name: &str, arg: Value_, out: Value_) {
        self.invalidate_stack_scalars();
        self.emit_line_hint();
        let callee = self.hostcall_addr(name).unwrap();
        let sig = self.import_sig(&[types::I64, self.ptr], None);
        self.builder.ins().call_indirect(sig, callee, &[self.vm, arg, out]);
    }

    /// A6：特化入口——寄存器参数写入局部标量槽（8B）+ 按需物化 Value 槽。
    ///
    /// Value 槽物化条件（静默错值红线）：标量分析失败（全通用）或任一块入口
    /// 该局部为 Unknown（存在经通用路径读取的可能）→ 必须物化；全部块入口
    /// 均为标量 → 只写标量槽（Value 槽永不读取，零 hostcall）。
    fn init_spec_args(&mut self) {
        // 先克隆参数种类（避免与后续 &mut self 调用冲突）
        let kinds: Vec<ScalarKind> = {
            let sig = self.spec_sig.as_ref().unwrap();
            sig.param_kinds.iter().map(|k| match k {
                ScalarAbiKind::I64 => ScalarKind::I32,
                ScalarAbiKind::F64 => ScalarKind::F64,
            }).collect()
        };
        let num_args = self.chunk.num_args.min(self.spec_args.len());
        for i in 0..num_args {
            let reg = self.spec_args[i];
            let kind = match kinds.get(i) {
                Some(k) => *k,
                None => continue, // 防御：签名参数数异常
            };
            if self.scalar_enabled {
                let lslot = self.local_scalar_slot(i, kind);
                self.store_scalar_i64(lslot, reg);
            }
            if self.spec_param_needs_value(i) {
                let lval = self.locals[&i];
                let lval_addr = self.builder.ins().stack_addr(self.ptr, lval, 0);
                match kind {
                    ScalarKind::I32 => {
                        self.call_hostcall_i64_ptr("host_make_int", reg, lval_addr);
                    }
                    ScalarKind::F64 => {
                        // P3：f64 参数——i64 寄存器位模式 bitcast 回 f64，经
                        // host_make_float 物化 Value（特化 ABI 位打包）。
                        let f = self.builder.ins().bitcast(types::F64, MemFlags::new(), reg);
                        self.call_hostcall_f64_ptr("host_make_float", f, lval_addr);
                    }
                    _ => {}
                }
            }
        }
    }

    /// A6：特化入口参数 i 的 Value 槽是否需要物化。
    fn spec_param_needs_value(&self, i: usize) -> bool {
        if !self.scalar_enabled {
            return true; // 分析失败 → 全通用 → 所有参数 Value 槽必须有效
        }
        for (_, entry) in self.block_entry_kinds.iter() {
            match entry.get(i) {
                Some(k) if matches!(k, ScalarKind::I32 | ScalarKind::F64 | ScalarKind::Bool) => {}
                _ => return true, // Unknown/Top/缺失 → 保守物化
            }
        }
        false
    }

    /// 按本函数返回约定返回「错误标志」：通用 → ok=0（I8）；特化 → 哨兵 0（I64）。
    /// 用于所有错误提前中止路径（B2 / 溢出 / 除零等），last_error 由 run_jit surface。
    fn emit_fn_return_error(&mut self) {
        if self.spec_mode {
            let v = self.builder.ins().iconst(types::I64, 0);
            self.builder.ins().return_(&[v]);
        } else {
            let ok = self.builder.ins().iconst(types::I8, 0);
            self.builder.ins().return_(&[ok]);
        }
        self.terminated = true;
    }

    /// 把当前栈顶（`self.sp` 处）值作为函数返回值返回：通用 → 写 out_ptr + ok=1；
    /// 特化 → 解包按 ret_kind（I64→host_value_to_i64 / F64→host_value_to_f64 +
    /// bitcast）i64 寄存器返回。用于 TailCall/TailCallClosure/Try 提前返回路径。
    fn emit_return_top(&mut self) {
        if self.spec_mode {
            let addr = self.builder.ins().stack_addr(self.ptr, self.stack_slot, self.sp);
            let ret_kind = self.spec_sig.as_ref().and_then(|s| s.ret_kind);
            let v = match ret_kind {
                Some(ScalarAbiKind::F64) => {
                    let f = self.call_hostcall_val_ret_f64("host_value_to_f64", addr);
                    self.builder.ins().bitcast(types::I64, MemFlags::new(), f)
                }
                _ => self.call_hostcall_val_ret_i64("host_value_to_i64", addr),
            };
            self.builder.ins().return_(&[v]);
        } else {
            self.copy_stack_to_ptr(self.sp, self.out_ptr, 0);
            let ok = self.builder.ins().iconst(types::I8, 1);
            self.builder.ins().return_(&[ok]);
        }
        self.terminated = true;
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
            // 按本函数返回约定中止（通用 ok=0 / 特化哨兵 0）；last_error 由 run_jit surface
            self.emit_fn_return_error();
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
        // (cond, invert)：invert=true 时结果 = !cond（F64 Neq 的 epsilon 翻转）。
        let (cond, invert) = match kind {
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
                (self.builder.ins().icmp(cc, a, b), false)
            }
            F64 => {
                let a = self.builder.ins().stack_load(types::F64, aslot, 0);
                let b = self.builder.ins().stack_load(types::F64, bslot, 0);
                // P3（静默错值修复）：VM/解释器浮点 `Eq`/`Neq` 为 **epsilon 比较**
                // `(x-y).abs() < 1e-10`（runtime/vm/execute.rs 同型快路径、interpreter/
                // binary.rs 一致）——JIT 原生此前用精确 FloatCC::Equal/NotEqual 会与
                // 参考路径分歧（如 0.1+0.2 == 0.3）。现对齐：Eq = |a-b| < 1e-10；
                // Neq = !(|a-b| < 1e-10)（select 翻转，NaN 语义与 VM 一致：
                // NaN 下 |a-b| 比较为 false → Eq=false、Neq=true）。
                // 注意 Neq 不能用 FloatCC::NotEqual（|a-b| == 1e-10 恰好时与
                // !(<) 分歧；NaN 语义也不同）。Lt/Gt/Lte/Gte 保持精确有序比较（VM 同）。
                match op {
                    Op::Eq | Op::Neq => {
                        let diff = self.builder.ins().fsub(a, b);
                        let ad = self.builder.ins().fabs(diff);
                        let eps = self.builder.ins().f64const(1e-10);
                        let lt = self.builder.ins().fcmp(FloatCC::LessThan, ad, eps);
                        (lt, op == Op::Neq)
                    }
                    Op::Lt => (self.builder.ins().fcmp(FloatCC::LessThan, a, b), false),
                    Op::Gt => (self.builder.ins().fcmp(FloatCC::GreaterThan, a, b), false),
                    Op::Lte => (self.builder.ins().fcmp(FloatCC::LessThanOrEqual, a, b), false),
                    Op::Gte => (self.builder.ins().fcmp(FloatCC::GreaterThanOrEqual, a, b), false),
                    _ => return Err("native f64 cmp: unsupported".into()),
                }
            }
            _ => return Err("native cmp: bad kind".into()),
        };
        // b1 → i8（0/1）；Neq（invert）翻转选择分支
        let one = self.builder.ins().iconst(types::I8, 1);
        let zero = self.builder.ins().iconst(types::I8, 0);
        let r = if invert {
            self.builder.ins().select(cond, zero, one)
        } else {
            self.builder.ins().select(cond, one, zero)
        };
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
                    // 静默错值红线（A2-AUDIT-11.4.35）：一般路径 Store 只写 Value 槽，
                    // **必须同时失效 local_scalars[i]**——否则循环回边处分析把该局部重置为
                    // I32（分析预测 spec 而发射走了内联→Value 结果），后续 Load(i) 重新
                    // 专用化会读到上一轮残留的过期标量槽（静默 0，溢出检查失效）。
                    self.local_scalars.remove(&i);
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
                let spec = if let Some(callee_idx) = self.name_to_chunk.get(i).copied().flatten() {
                    self.emit_direct_call(callee_idx, i, 0, null_ptr, out, self.sp, op_start)?
                } else {
                    self.call_hostcall_call("host_call", i as u64, 0, null_ptr, out);
                    false
                };
                // A2b：调用结果写 `out`（Value）——清除该偏移的标量跟踪（参数物化
                // 保留了旧标量跟踪，调用后指向过期值 → 静默错值红线）。
                // A6：特化路径结果在标量槽（8B），跟踪已建立——不清除。
                if !spec {
                    self.clear_stack_scalar_at(self.sp);
                }
                self.bump_sp()?;
            }
            CallN(i, n) => {
                // Args are at [sp - n*VS, sp). Pop them, then out is at sp.
                self.sp -= (n as i32) * (VALUE_SIZE as i32);
                let args_addr = self.builder.ins().stack_addr(self.ptr, self.stack_slot, self.sp);
                let out = self.stack_addr_at_sp();
                // A1：目标为已注册用户函数 → JIT-to-JIT 直接调用（不再逃逸解释器）。
                let spec = if let Some(callee_idx) = self.name_to_chunk.get(i).copied().flatten() {
                    self.emit_direct_call(callee_idx, i, n as u64, args_addr, out, self.sp, op_start)?
                } else {
                    // A2b：物化参数（Value 槽有效供被调函数读取）。
                    self.materialize_stack_range(self.sp, n as usize);
                    self.call_hostcall_call("host_call", i as u64, n as u64, args_addr, out);
                    false
                };
                // A2b：调用结果写 `out`（Value）——清除该偏移的标量跟踪。
                // A6：特化路径结果在标量槽（8B），跟踪已建立——不清除。
                if !spec {
                    self.clear_stack_scalar_at(self.sp);
                }
                self.bump_sp()?;
            }
            CallClosure(n) => {
                // a1 P1：间接调用闭包/函数值。栈上 [arg1..argN, callee]（N+1 个值），
                // 取最后一个为 callee，其余为参数。
                // M2.6-P4：`host_jit_call_indirect`——闭包路径的 JIT-to-JIT（A1 慢路径
                // 等价物）：FnRef 按名解析 + 捕获追加 → `jit_call_chunk` → 闭包体直接
                // 执行 JIT 机器码（不再逃逸解释器）。CallClosure opcode 不带名字
                // （callee 是运行期 Value），无法编译期静态解析 → 无 A1 call_indirect
                // 机器码快路径，保持 hostcall trampoline（保守正确性优先）。
                // 失败/编译失败 → `jit_call_chunk` 完整回退 `call_value` 语义（零变化）。
                self.sp -= ((n + 1) as i32) * (VALUE_SIZE as i32);
                self.materialize_stack_range(self.sp, (n + 1) as usize);
                let args_addr = self.builder.ins().stack_addr(self.ptr, self.stack_slot, self.sp);
                let out = self.stack_addr_at_sp();
                self.call_hostcall_make_n("host_jit_call_indirect", (n + 1) as u64, args_addr, out);
                self.bump_sp()?;
                // B2: 检查 host_jit_call_indirect 是否设置了错误（如「期望可调用值」/未定义函数）。
                // 若有错误，立即返回按约定中止（run_jit 读取 last_error 并触发 fallback/报错），
                // 避免错误延迟到 run_jit 末尾才浮出（复用 MethodCall 的 B2 模式）。
                let err_flag = self.call_hostcall_vm_ret_u8("host_check_error");
                let has_err = self.builder.ins().icmp_imm(IntCC::NotEqual, err_flag, 0);
                let err_blk = self.builder.create_block();
                let cont_blk = self.builder.create_block();
                self.builder.ins().brif(has_err, err_blk, &[], cont_blk, &[]);
                // 错误路径：按约定中止（通用 ok=0 / 特化哨兵 0）
                self.builder.switch_to_block(err_blk);
                self.builder.seal_block(err_blk);
                self.emit_fn_return_error();
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
                // 若有错误，立即按约定中止（通用 ok=0 / 特化哨兵 0），让 run_jit 读取
                // last_error 并触发 fallback。这避免了"静默 push Unit + 后续 println 输出 ()"的问题。
                let err_flag = self.call_hostcall_vm_ret_u8("host_check_error");
                let has_err = self.builder.ins().icmp_imm(IntCC::NotEqual, err_flag, 0);
                let err_blk = self.builder.create_block();
                let cont_blk = self.builder.create_block();
                self.builder.ins().brif(has_err, err_blk, &[], cont_blk, &[]);
                // 错误路径：按约定中止
                self.builder.switch_to_block(err_blk);
                self.builder.seal_block(err_blk);
                self.emit_fn_return_error();
                // 继续路径
                self.builder.switch_to_block(cont_blk);
                self.builder.seal_block(cont_blk);
                self.terminated = false;
            }
            Ret => {
                self.sp -= VALUE_SIZE as i32;
                if let (Some(out), Some(cont)) = (self.inline_out, self.inline_cont) {
                    // A2：内联模式——结果写调用方 out 槽，跳汇合块（不返回整个函数）。
                    self.materialize_all_stack();
                    self.copy_stack_to_ptr(self.sp, out, 0);
                    self.builder.ins().jump(cont, &[]);
                    self.terminated = true;
                } else if self.spec_mode {
                    // A6/P3：特化入口——从栈顶读标量返回值（i64 寄存器返回，零装箱）。
                    // I64 返回：I32 标量槽直接读 8B；F64 返回：F64 标量槽直接读再
                    // bitcast；无标量槽（通用 hostcall/内联结果/分析失败）→ 物化 +
                    // 按 ret_kind 解包（host_value_to_i64 / host_value_to_f64）。
                    let ret_kind = self.spec_sig.as_ref().and_then(|s| s.ret_kind);
                    let ret_off = self.sp;
                    let ret: Value_ = match ret_kind {
                        Some(ScalarAbiKind::F64) => {
                            let scalar_f64 = self.scalar_enabled.then(|| {
                                match self.stack_scalars.get(&ret_off) {
                                    Some(&(ScalarKind::F64, slot)) => Some(self.builder.ins().stack_load(types::F64, slot, 0)),
                                    _ => None,
                                }
                            }).flatten();
                            match scalar_f64 {
                                Some(f) => self.builder.ins().bitcast(types::I64, MemFlags::new(), f),
                                None => {
                                    self.materialize_stack_at(ret_off);
                                    let addr = self.builder.ins().stack_addr(self.ptr, self.stack_slot, ret_off);
                                    let f = self.call_hostcall_val_ret_f64("host_value_to_f64", addr);
                                    self.builder.ins().bitcast(types::I64, MemFlags::new(), f)
                                }
                            }
                        }
                        _ => {
                            // 既有 I64 返回逻辑（I32 标量槽直接读；否则物化 + 解包）
                            if self.scalar_enabled {
                                if let Some(&(kind, slot)) = self.stack_scalars.get(&ret_off) {
                                    if kind == ScalarKind::I32 {
                                        self.load_scalar_i64(slot)
                                    } else {
                                        // 非 I32（如 F64 标量 / 通用结果）→ 物化 + 解包
                                        self.materialize_stack_at(ret_off);
                                        let addr = self.builder.ins().stack_addr(self.ptr, self.stack_slot, ret_off);
                                        self.call_hostcall_val_ret_i64("host_value_to_i64", addr)
                                    }
                                } else {
                                    // 无标量槽（通用 hostcall/内联结果）→ 物化 + 解包
                                    self.materialize_stack_at(ret_off);
                                    let addr = self.builder.ins().stack_addr(self.ptr, self.stack_slot, ret_off);
                                    self.call_hostcall_val_ret_i64("host_value_to_i64", addr)
                                }
                            } else {
                                // 分析失败（全通用）→ 解包 Value 槽
                                self.materialize_stack_at(ret_off);
                                let addr = self.builder.ins().stack_addr(self.ptr, self.stack_slot, ret_off);
                                self.call_hostcall_val_ret_i64("host_value_to_i64", addr)
                            }
                        }
                    };
                    self.builder.ins().return_(&[ret]);
                    self.terminated = true;
                } else {
                    // 通用路径：物化栈标量（含返回值，Value 槽有效供拷贝）。
                    self.materialize_all_stack();
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
                // Err 路径：out 槽已是完整 Result::Err——按约定作为函数返回值返回
                // （通用：复制到 out_ptr + ok=1；特化：解包 i64；与 VM 早退一致）。
                self.builder.switch_to_block(err_blk);
                self.builder.seal_block(err_blk);
                self.emit_return_top();
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
                // 结果已在 out 槽 → 按约定作为函数返回值返回（通用：out_ptr + ok=1；
                // 特化：解包 i64；host_call 失败时 last_error 由 run_jit surface）。
                self.emit_return_top();
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
                // B2：host_call_indirect 设置错误 → 立即按约定中止（run_jit surface）。
                let err_flag = self.call_hostcall_vm_ret_u8("host_check_error");
                let has_err = self.builder.ins().icmp_imm(IntCC::NotEqual, err_flag, 0);
                let err_blk = self.builder.create_block();
                let cont_blk = self.builder.create_block();
                self.builder.ins().brif(has_err, err_blk, &[], cont_blk, &[]);
                self.builder.switch_to_block(err_blk);
                self.builder.seal_block(err_blk);
                self.emit_fn_return_error();
                // 无错误：按约定返回被调闭包结果（通用：out_ptr + ok=1；特化：解包 i64）。
                self.builder.switch_to_block(cont_blk);
                self.builder.seal_block(cont_blk);
                self.emit_return_top();
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
        if self.spec_mode {
            // A6：特化入口——fall-through/未填充块的防御性返回（哨兵 0）。
            let v = self.builder.ins().iconst(types::I64, 0);
            self.builder.ins().return_(&[v]);
            self.terminated = true;
            return;
        }
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
    ///
    /// M2.5-A6：内联（A2）优先 → 特化 ABI（标量寄存器传递）→ 通用 A1。
    /// 返回 `Ok(true)` = 走了特化路径（结果在标量槽，调用方不清除跟踪）；
    /// `Ok(false)` = 内联或通用路径（结果在 `out` Value 槽，调用方清除跟踪）。
    fn emit_direct_call(
        &mut self,
        callee_idx: usize,
        name_idx: usize,
        n: u64,
        args_addr: Value_,
        out: Value_,
        args_base_off: i32,
        op_start: usize,
    ) -> Result<bool, String> {
        self.emit_line_hint();
        let n_usize = n as usize;
        // A6/P3：预读参数标量值（SSA 寄存器）。按目标特化签名的参数种类逐个匹配
        // 槽：I64 → I32 槽读 i64；F64 → F64 槽读 f64 再 bitcast 为 i64（特化 ABI
        // 位打包）。槽种类与签名不符/无槽 → None（发射期 try_spec_call 据此回退/
        // 报错，防漂移）。物化（下方）会移除栈标量跟踪，但 SSA 值已捕获——特化
        // 路径直接以寄存器传参（零装箱）。
        let callee_sig_kinds: Vec<ScalarAbiKind> = match self.chunk_sigs.get(callee_idx) {
            Some(Some(s)) => s.param_kinds.clone(),
            _ => Vec::new(),
        };
        let mut arg_scalars: Vec<Option<Value_>> = Vec::with_capacity(n_usize);
        for i in 0..n_usize {
            let off = args_base_off + (i as i32) * (VALUE_SIZE as i32);
            let want = callee_sig_kinds.get(i).copied();
            match (want, self.stack_scalars.get(&off)) {
                (Some(ScalarAbiKind::I64), Some(&(ScalarKind::I32, slot))) => {
                    arg_scalars.push(Some(self.load_scalar_i64(slot)));
                }
                (Some(ScalarAbiKind::F64), Some(&(ScalarKind::F64, slot))) => {
                    let f = self.builder.ins().stack_load(types::F64, slot, 0);
                    arg_scalars.push(Some(self.builder.ins().bitcast(types::I64, MemFlags::new(), f)));
                }
                _ => arg_scalars.push(None),
            }
        }
        // A2：内联 + A1 通用路径需要物化参数（Value 槽有效供被调方读取）。
        self.materialize_stack_range(args_base_off, n_usize);

        // A2：内联快路径——callee 满足内联条件时把 body 直插调用点（不 emit call）。
        // Ok(true) = 已内联；Ok(false) = 不满足条件 → 静默回退下方 A6/A1
        // （不报错、不改变行为）；Err = 内联中途失败（静态过滤后不应发生）→
        // 整个函数回退解释器（正确性优先）。
        match self.try_inline_call(callee_idx, n, args_addr, out)? {
            true => return Ok(false),
            false => { /* 静默回退 */ }
        }

        // M2.5-A6：特化 ABI（标量寄存器传递）。
        if self.try_spec_call(callee_idx, name_idx, n, args_addr, out, args_base_off, op_start, &arg_scalars)? {
            return Ok(true);
        }

        // ── A1：通用直接调用（参数已物化）──
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
        Ok(false)
    }

    // ── M2.5-A6：特化 ABI 调用（标量寄存器传递）──────────────────────────

    /// A6/P3：特化 ABI 调用。返回 `Ok(true)` = 已特化（结果在标量槽，跟踪已建立）；
    /// `Ok(false)` = 回退通用；`Err` = 编译失败（整函数回退解释器）。
    ///
    /// 前提：`args` 已物化为 Value（慢路径 `host_jit_call_spec` 需要）；`arg_scalars`
    /// 为物化前预读的标量值（快路径寄存器传参，零装箱；I64 参= i64 SSA、F64 参=
    /// f64 位打包后的 i64 SSA，特化 ABI 位打包统一为 i64 寄存器）。
    ///
    /// 资格（与分析期共享 `spec_target_sig`，防静默错值漂移）：
    /// - 目标有特化签名且参数个数匹配、种类为 I64/F64 混合、实参个数匹配
    /// - 调用侧实参全部有对应标量槽（I64→I32 槽、F64→F64 槽；否则回退通用）
    /// - 分析期预测特化而发射期不可特化 → Err（整函数回退解释器，杜绝错值）
    ///
    /// 结果合并槽按 `ret_kind` 建立（I64→I32 槽、F64→F64 槽）——与分析期预测的
    /// 结果种类一致（AUDIT-11.4.36 教训：分析/发射结果种类必须逐一对齐）。
    ///
    /// 错误路径（静默错值红线）：特化入口返回 i64 无法带错误标志——调用后必须
    /// `host_check_error`（B2 模式）：有错 → 立即按本函数返回约定中止（通用 ok=0 /
    /// 特化哨兵 0），last_error 由 run_jit surface。
    fn try_spec_call(
        &mut self,
        callee_idx: usize,
        name_idx: usize,
        n: u64,
        args_addr: Value_,
        _out: Value_,
        args_base_off: i32,
        op_start: usize,
        arg_scalars: &[Option<Value_>],
    ) -> Result<bool, String> {
        let n_usize = n as usize;
        let sig = match spec_target_sig(self.name_to_chunk, self.chunk_sigs, self.all_chunks, name_idx, n_usize) {
            Some(s) => s,
            None => {
                // 防御：分析期预测特化而发射期不可 → 整函数回退解释器（静默错值红线）。
                if self.call_spec_results.get(&op_start) == Some(&true) {
                    return Err("JIT A6: 特化资格预测不一致（发射期目标不可特化）".into());
                }
                return Ok(false);
            }
        };
        // P3：发射端 arg_scalars 已按签名种类预读（emit_direct_call 按 param_kinds
        // 逐个匹配槽），任一缺失/不符 → None → 回退/报错（防漂移）。
        if arg_scalars.iter().any(|a| a.is_none()) {
            // 防御：分析期预测特化而发射期缺标量槽 → 整函数回退解释器。
            if self.call_spec_results.get(&op_start) == Some(&true) {
                return Err("JIT A6: 特化参数预测不一致（发射期缺标量槽）".into());
            }
            return Ok(false);
        }
        let ret_kind = sig.ret_kind.unwrap_or(ScalarAbiKind::I64);
        let vm = self.vm;
        // 慢路径地址（out 覆盖首个参数槽；参数已物化为 Value）
        let out_addr = self.builder.ins().stack_addr(self.ptr, self.stack_slot, args_base_off);
        // 结果合并槽：快/慢路径都写 8B 结果，汇合块读取（避免块参数 phi——
        // 多次特化调用同块共存时块参数 phi 触发 Cranelift verifier 问题）。
        // P3：种类按 ret_kind（I64→I32 槽 / F64→F64 槽）——与分析期预测一致。
        let merge_kind = match ret_kind {
            ScalarAbiKind::F64 => ScalarKind::F64,
            _ => ScalarKind::I32,
        };
        let merge_slot = self.stack_scalar_slot(args_base_off, merge_kind);
        // 特化指针表：vm.jit_spec_table_ptr[callee_idx]（0 = 未编译 → 慢路径）
        let table_off = std::mem::offset_of!(Vm, jit_spec_table_ptr) as i64;
        let table_off_v = self.builder.ins().iconst(self.ptr, table_off);
        let table_addr = self.builder.ins().iadd(vm, table_off_v);
        let table_base = self.builder.ins().load(self.ptr, MemFlags::new(), table_addr, 0);
        let entry_off_v = self.builder.ins().iconst(
            self.ptr,
            (callee_idx * std::mem::size_of::<usize>()) as i64,
        );
        let entry_addr = self.builder.ins().iadd(table_base, entry_off_v);
        let spec_ptr = self.builder.ins().load(self.ptr, MemFlags::new(), entry_addr, 0);
        let is_zero = self.builder.ins().icmp_imm(IntCC::Equal, spec_ptr, 0);
        let slow_blk = self.builder.create_block();
        let fast_blk = self.builder.create_block();
        let merge_blk = self.builder.create_block();
        self.builder.ins().brif(is_zero, slow_blk, &[], fast_blk, &[]);

        // 快路径：特化直接调用（签名 (vm, i64 x8) -> i64）→ 结果写合并槽。
        // P1：`skip_chunk_ctx[callee_idx]`（被调方为纯标量、体内不读字符串表）→
        // 省去 current_chunk_idx 保存/切换/恢复 3 次内存操作（save/store/restore）。
        // 否则保持既有 save/switch/restore（被调方 hostcall 需按被调 chunk 解析
        // 字符串表；慢路径 host_jit_call_spec 自带切换，不受本优化影响）。
        self.builder.switch_to_block(fast_blk);
        self.builder.seal_block(fast_blk);
        let skip_chunk = self.skip_chunk_ctx.get(callee_idx).copied().unwrap_or(false);
        let spec_sig = self.import_sig(&[types::I64; MAX_SPEC_ARGS], Some(types::I64));
        // 参数补足到 MAX_SPEC_ARGS（未用参数传 0，被调方忽略）
        let mut args8: Vec<Value_> = arg_scalars.iter().map(|a| a.unwrap()).collect();
        while args8.len() < MAX_SPEC_ARGS {
            let z = self.builder.ins().iconst(types::I64, 0);
            args8.push(z);
        }
        let saved_chunk = if skip_chunk {
            None
        } else {
            let chunk_off = std::mem::offset_of!(Vm, current_chunk_idx) as i64;
            let chunk_off_v = self.builder.ins().iconst(self.ptr, chunk_off);
            let chunk_addr = self.builder.ins().iadd(vm, chunk_off_v);
            let saved_chunk = self.builder.ins().load(self.ptr, MemFlags::new(), chunk_addr, 0);
            let callee_idx_v = self.builder.ins().iconst(self.ptr, callee_idx as i64);
            self.builder.ins().store(MemFlags::new(), callee_idx_v, chunk_addr, 0);
            Some((saved_chunk, chunk_addr))
        };
        let call = self.builder.ins().call_indirect(spec_sig, spec_ptr, &[vm, args8[0], args8[1], args8[2], args8[3], args8[4], args8[5], args8[6], args8[7]]);
        let ret_i64 = self.builder.inst_results(call)[0];
        if let Some((saved_chunk, chunk_addr)) = saved_chunk {
            self.builder.ins().store(MemFlags::new(), saved_chunk, chunk_addr, 0);
        }
        // P3：返回按 ret_kind 解包——I64 直接写 i64；F64 从 i64 位模式 bitcast 回
        // f64 写 F64 槽（位打包 ABI）。
        match ret_kind {
            ScalarAbiKind::F64 => {
                let ret_f = self.builder.ins().bitcast(types::F64, MemFlags::new(), ret_i64);
                self.store_scalar_f64(merge_slot, ret_f);
            }
            _ => self.store_scalar_i64(merge_slot, ret_i64),
        }
        self.builder.ins().jump(merge_blk, &[]);

        // 慢路径：trampoline（current_chunk_idx 仍为调用方 → name_idx 可正确解析；
        // 编译特化入口 + 注册 + 调用；失败 → 通用回退）。结果 Value → 按 ret_kind
        // 解包（I64→host_value_to_i64；F64→host_value_to_f64 再 bitcast）。
        // 注意：两条 hostcall 均**不失效栈标量**（被调方不触碰调用方标量槽；
        // 失效会清除其他活标量跟踪 → 后续通用消费者读陈旧 Value 槽 → 静默错值）。
        self.builder.switch_to_block(slow_blk);
        self.builder.seal_block(slow_blk);
        self.call_hostcall_spec_call("host_jit_call_spec", name_idx as u64, n, args_addr, out_addr);
        let v_addr = self.builder.ins().stack_addr(self.ptr, self.stack_slot, args_base_off);
        match ret_kind {
            ScalarAbiKind::F64 => {
                let f = self.call_hostcall_val_ret_f64_noinv("host_value_to_f64", v_addr);
                let bits = self.builder.ins().bitcast(types::I64, MemFlags::new(), f);
                self.store_scalar_i64(merge_slot, bits);
            }
            _ => {
                let slow_ret = self.call_hostcall_val_ret_i64_noinv("host_value_to_i64", v_addr);
                self.store_scalar_i64(merge_slot, slow_ret);
            }
        }
        self.builder.ins().jump(merge_blk, &[]);

        // 汇合：错误检查（B2，静默错值红线）→ 结果已在合并槽（快/慢路径各写）。
        // P1：host_check_error 内联化——直接 load `vm.jit_error_flag`（与
        // `has_last_error()` 恒等：set_last_error/set_jit_error 置 1、take_last_error
        // 清 0），省一次函数调用 + 一次内存读，语义完全不变（B2 红线保留）。
        self.builder.switch_to_block(merge_blk);
        self.builder.seal_block(merge_blk);
        let err_flag = self.inline_check_error_flag();
        let has_err = self.builder.ins().icmp_imm(IntCC::NotEqual, err_flag, 0);
        let err_blk = self.builder.create_block();
        let cont_blk = self.builder.create_block();
        self.builder.ins().brif(has_err, err_blk, &[], cont_blk, &[]);
        // 错误路径：立即按本函数返回约定中止（last_error 由 run_jit surface）
        self.builder.switch_to_block(err_blk);
        self.builder.seal_block(err_blk);
        self.emit_fn_return_error();
        // 继续路径：结果在合并槽（args_base_off 的 I32 标量槽），跟踪已由
        // stack_scalar_slot 注册——后续原生/通用消费者均正确（通用消费者经
        // materialize_stack_at 从标量槽物化 Value）。
        self.builder.switch_to_block(cont_blk);
        self.builder.seal_block(cont_blk);
        self.terminated = false;
        Ok(true)
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

    /// hostcall 报错检查：若 `host_check_error` 非零，立即按约定中止（通用返回 ok=0，
    /// 特化返回哨兵 0——run_jit 读取 last_error 触发 fallback）。与 MethodCall 分支的
    /// B2 模式一致——避免 binop 报错（如整数溢出）后继续执行、错误被后续操作覆盖。
    /// AUDIT-11.4.17。
    fn emit_err_check_abort(&mut self) {
        let err_flag = self.call_hostcall_vm_ret_u8("host_check_error");
        let has_err = self.builder.ins().icmp_imm(IntCC::NotEqual, err_flag, 0);
        let err_blk = self.builder.create_block();
        let cont_blk = self.builder.create_block();
        self.builder.ins().brif(has_err, err_blk, &[], cont_blk, &[]);
        // 错误路径：按约定中止（通用 ok=0 / 特化哨兵 0）
        self.builder.switch_to_block(err_blk);
        self.builder.seal_block(err_blk);
        self.emit_fn_return_error();
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
