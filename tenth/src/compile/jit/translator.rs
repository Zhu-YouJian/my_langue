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
use std::collections::{HashMap, HashSet};
use std::mem::size_of;

use crate::runtime::vm::{Chunk, Op, Vm};
use crate::runtime::value::Value;

// Cranelift's `StackSlot` and `SigRef` entity types live in `codegen::ir`
// and are NOT re-exported from the prelude. Bring them in explicitly.
use cranelift::codegen::ir::{StackSlot, SigRef};

/// Size of one `Value` on the stack.
const VALUE_SIZE: u32 = size_of::<Value>() as u32;

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

pub fn translate<M: Module>(
    module: &mut M,
    chunk_idx: usize,
    chunk: &Chunk,
    name_to_chunk: &[Option<usize>],
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

        // ── Emit code for each instruction ─────────────────────────────────
        let mut ip = 0usize;
        let code_len = self.chunk.code.len();

        while ip < code_len {
            if let Some(&blk) = self.blocks.get(&ip) {
                if !self.terminated {
                    // Fall-through: record sp and jump.
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

    fn emit_op(&mut self, op: Op, op_start: usize, ip_after: usize) -> Result<(), String> {
        use Op::*;
        // 9c：记录当前 opcode 的源码行号（0 = 无），供 hostcall 报错时携带。
        self.cur_line = self.chunk.line_at(op_start).unwrap_or(0);
        match op {
            PushInt(n) => {
                let out = self.stack_addr_at_sp();
                self.call_hostcall_i64("host_make_int", n, out);
                self.bump_sp()?;
            }
            PushFloat(f) => {
                let out = self.stack_addr_at_sp();
                self.call_hostcall_f64("host_make_float", f, out);
                self.bump_sp()?;
            }
            PushFloat32(f) => {
                // 阶段 6：真正的 f32 hostcall，保留 dtype 信息（不再降级为 f64）。
                let out = self.stack_addr_at_sp();
                self.call_hostcall_f32("host_make_float32", f, out);
                self.bump_sp()?;
            }
            PushBool(b) => {
                let out = self.stack_addr_at_sp();
                self.call_hostcall_u8("host_make_bool", if b { 1 } else { 0 }, out);
                self.bump_sp()?;
            }
            PushChar(c) => {
                // JIT不支持Char直接作为标量值，使用Int作为fallback
                let out = self.stack_addr_at_sp();
                self.call_hostcall_i64("host_make_int", c as i64, out);
                self.bump_sp()?;
            }
            PushStr(i) => {
                let out = self.stack_addr_at_sp();
                self.call_hostcall_u64("host_make_str", i as u64, out);
                self.bump_sp()?;
            }
            PushUnit => {
                let out = self.stack_addr_at_sp();
                self.call_hostcall_unit("host_make_unit", out);
                self.bump_sp()?;
            }
            Pop => { self.sp -= VALUE_SIZE as i32; }
            Dup => {
                let src_off = self.sp - VALUE_SIZE as i32;
                let dst_off = self.sp;
                self.copy_within_stack(src_off, dst_off);
                self.bump_sp()?;
            }
            Load(i) => {
                let src = self.locals.get(&i).copied().ok_or("Load: bad local")?;
                let dst_off = self.sp;
                self.copy_slot_to_stack(src, 0, dst_off);
                self.bump_sp()?;
            }
            Store(i) => {
                self.sp -= VALUE_SIZE as i32;
                let src_off = self.sp;
                if !self.locals.contains_key(&i) {
                    let new_s = self.builder.create_sized_stack_slot(StackSlotData::new(
                        StackSlotKind::ExplicitSlot,
                        VALUE_SIZE,
                        8,
                    ));
                    self.locals.insert(i, new_s);
                }
                let dst = self.locals[&i];
                self.copy_stack_to_slot(src_off, dst, 0);
            }
            LoadGlobal(i) => {
                let out = self.stack_addr_at_sp();
                self.call_hostcall_u64("host_load_global", i as u64, out);
                self.bump_sp()?;
            }
            StoreGlobal(i) => {
                self.sp -= VALUE_SIZE as i32;
                let val_off = self.sp;
                let val_addr = self.builder.ins().stack_addr(self.ptr, self.stack_slot, val_off);
                let out_addr = val_addr; // result = stored value, same slot
                self.call_hostcall_u64_val("host_store_global", i as u64, val_addr, out_addr);
                self.bump_sp()?; // push result back
            }
            Add => self.emit_binop("host_add")?,
            Sub => self.emit_binop("host_sub")?,
            Mul => self.emit_binop("host_mul")?,
            Div => self.emit_binop("host_div")?,
            Mod => self.emit_binop("host_mod")?,
            Neg => self.emit_unop("host_neg")?,
            Not => self.emit_unop("host_not")?,
            Eq => self.emit_binop("host_eq")?,
            Neq => self.emit_binop("host_neq")?,
            Lt => self.emit_binop("host_lt")?,
            Gt => self.emit_binop("host_gt")?,
            Lte => self.emit_binop("host_lte")?,
            Gte => self.emit_binop("host_gte")?,
            Jump(o) => {
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
                let cond_addr = self.builder.ins().stack_addr(self.ptr, self.stack_slot, self.sp);
                let truth = self.call_hostcall_val_ret_u8("host_truthy", cond_addr);
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
                let cond_addr = self.builder.ins().stack_addr(self.ptr, self.stack_slot, self.sp);
                let truth = self.call_hostcall_val_ret_u8("host_truthy", cond_addr);
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
                self.bump_sp()?;
            }
            CallN(i, n) => {
                // Args are at [sp - n*VS, sp). Pop them, then out is at sp.
                self.sp -= (n as i32) * (VALUE_SIZE as i32);
                let args_addr = self.builder.ins().stack_addr(self.ptr, self.stack_slot, self.sp);
                let out = self.stack_addr_at_sp();
                // A1：目标为已注册用户函数 → JIT-to-JIT 直接调用（不再逃逸解释器）。
                if let Some(callee_idx) = self.name_to_chunk.get(i).copied().flatten() {
                    self.emit_direct_call(callee_idx, i, n as u64, args_addr, out)?;
                } else {
                    self.call_hostcall_call("host_call", i as u64, n as u64, args_addr, out);
                }
                self.bump_sp()?;
            }
            CallClosure(n) => {
                // a1 P1：间接调用闭包/函数值。栈上 [arg1..argN, callee]（N+1 个值），
                // host_call_indirect 取最后一个为 callee，其余为参数（走 Vm::call_value）。
                self.sp -= ((n + 1) as i32) * (VALUE_SIZE as i32);
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
                self.sp -= VALUE_SIZE as i32;
                self.copy_stack_to_ptr(self.sp, self.out_ptr, 0);
                let ok = self.builder.ins().iconst(types::I8, 1);
                self.builder.ins().return_(&[ok]);
                self.terminated = true;
            }
            MakeVec(n) => {
                self.sp -= (n as i32) * (VALUE_SIZE as i32);
                let args_addr = self.builder.ins().stack_addr(self.ptr, self.stack_slot, self.sp);
                let out = self.stack_addr_at_sp();
                self.call_hostcall_make_n("host_make_vec", n as u64, args_addr, out);
                self.bump_sp()?;
            }
            MakeMap(n) => {
                let total = n * 2;
                self.sp -= (total as i32) * (VALUE_SIZE as i32);
                let args_addr = self.builder.ins().stack_addr(self.ptr, self.stack_slot, self.sp);
                let out = self.stack_addr_at_sp();
                self.call_hostcall_make_n("host_make_map", n as u64, args_addr, out);
                self.bump_sp()?;
            }
            NewStruct(name_i, field_count) => {
                let total = field_count * 2;
                self.sp -= (total as i32) * (VALUE_SIZE as i32);
                let args_addr = self.builder.ins().stack_addr(self.ptr, self.stack_slot, self.sp);
                let out = self.stack_addr_at_sp();
                self.call_hostcall_new_struct("host_new_struct", name_i as u64, field_count as u64, args_addr, out);
                self.bump_sp()?;
            }
            NewUnion(name_i, field_i) => {
                // M1.2：union 构造 — 栈顶单个 value 弹出，构造 Value::Union
                self.sp -= VALUE_SIZE as i32;
                let val_addr = self.builder.ins().stack_addr(self.ptr, self.stack_slot, self.sp);
                let out = self.stack_addr_at_sp();
                self.call_hostcall_new_union("host_new_union", name_i as u64, field_i as u64, val_addr, out);
                self.bump_sp()?;
            }
            LoadField(i) => {
                self.sp -= VALUE_SIZE as i32;
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
                let args_addr = self.builder.ins().stack_addr(self.ptr, self.stack_slot, self.sp);
                let out = self.stack_addr_at_sp();
                self.call_hostcall_make_enum("host_make_enum", name_i as u64, variant_i as u64, field_count as u64, args_addr, out);
                self.bump_sp()?;
            }
            IsEnumVariant(variant_i) => {
                self.sp -= VALUE_SIZE as i32;
                let recv_addr = self.builder.ins().stack_addr(self.ptr, self.stack_slot, self.sp);
                let out = self.stack_addr_at_sp();
                self.call_hostcall_u64_val("host_is_enum_variant", variant_i as u64, recv_addr, out);
                self.bump_sp()?;
            }
            EnumGetField(field_i) => {
                self.sp -= VALUE_SIZE as i32;
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
                let args_addr = self.builder.ins().stack_addr(self.ptr, self.stack_slot, self.sp);
                let out = self.stack_addr_at_sp();
                self.call_hostcall_make_enum(
                    "host_make_closure", params as u64, captures as u64, chunk_idx as u64, args_addr, out,
                );
                self.bump_sp()?;
            }
            IsStruct(_) => {
                // Struct pattern matching not JIT-compiled; fallback to VM.
                return Err(format!("JIT: IsStruct not supported, fallback to VM"));
            }
            Await | Spawn | Yield => {
                // Async opcodes not JIT-compiled; fallback to VM.
                // Yield（Phase 2 Step 3-4 协作式调度）同样需要调度器支持，JIT 不支持。
                return Err(format!("JIT: async opcode not supported, fallback to VM"));
            }
            MakeTuple(_) | IsTuple(_) | TupleGet(_) | Try | TailCall(..) | TailCallClosure(..) => {
                // Tuple / Try / TailCall opcodes not JIT-compiled; fallback to VM.
                // a1 P1：TailCallClosure（闭包尾调用）与 TailCall 同策略——不 JIT 编译，整体 fallback VM。
                return Err(format!("JIT: tuple/try/tailcall opcode not supported, fallback to VM"));
            }
            MakeRef | MakeMutRef(_) | Deref | DerefStore => {
                // AUDIT-11.4.21：引用语义 opcodes 不 JIT 编译；整体 fallback VM。
                return Err(format!("JIT: ref/deref opcode not supported, fallback to VM"));
            }
            MakeCell | BindSelfCapture(_) => {
                // M1-S2（true letrec）：自引用 cell opcodes 不 JIT 编译；整体 fallback VM。
                // 闭包体本身（Load + CallClosure）仍可 JIT——host_call_indirect 走
                // Vm::call_value（已支持 Shared cell 解包），letrec 语义保持正确。
                return Err(format!("JIT: letrec cell opcode not supported, fallback to VM"));
            }
        }
        Ok(())
    }

    // ── Return helper ──────────────────────────────────────────────────────

    fn emit_return(&mut self) {
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
        self.emit_line_hint();
        let callee = self.hostcall_addr(name).unwrap();
        let sig = self.import_sig(&[self.ptr], None);
        self.builder.ins().call_indirect(sig, callee, &[self.vm, out]);
    }

    fn call_hostcall_i64(&mut self, name: &str, arg: i64, out: Value_) {
        self.emit_line_hint();
        let callee = self.hostcall_addr(name).unwrap();
        let sig = self.import_sig(&[types::I64, self.ptr], None);
        let a = self.builder.ins().iconst(types::I64, arg);
        self.builder.ins().call_indirect(sig, callee, &[self.vm, a, out]);
    }

    fn call_hostcall_f64(&mut self, name: &str, arg: f64, out: Value_) {
        self.emit_line_hint();
        let callee = self.hostcall_addr(name).unwrap();
        let sig = self.import_sig(&[types::F64, self.ptr], None);
        let a = self.builder.ins().f64const(arg);
        self.builder.ins().call_indirect(sig, callee, &[self.vm, a, out]);
    }

    /// f32 hostcall 调用：参数以 f32 ABI 传递（4 字节寄存器），与 f64 路径
    /// 区分以保留 dtype 信息到运行时。栈布局不变——out 仍为 *mut Value。
    fn call_hostcall_f32(&mut self, name: &str, arg: f32, out: Value_) {
        self.emit_line_hint();
        let callee = self.hostcall_addr(name).unwrap();
        let sig = self.import_sig(&[types::F32, self.ptr], None);
        let a = self.builder.ins().f32const(arg);
        self.builder.ins().call_indirect(sig, callee, &[self.vm, a, out]);
    }

    fn call_hostcall_u8(&mut self, name: &str, arg: u8, out: Value_) {
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
        self.emit_line_hint();
        let callee = self.hostcall_addr(name).unwrap();
        let sig = self.import_sig(&[self.ptr], Some(types::I8));
        let call = self.builder.ins().call_indirect(sig, callee, &[self.vm, arg]);
        self.builder.inst_results(call)[0]
    }

    /// `fn(vm) -> u8` — e.g. host_check_error.
    fn call_hostcall_vm_ret_u8(&mut self, name: &str) -> Value_ {
        self.emit_line_hint();
        let callee = self.hostcall_addr(name).unwrap();
        let sig = self.import_sig(&[], Some(types::I8));
        let call = self.builder.ins().call_indirect(sig, callee, &[self.vm]);
        self.builder.inst_results(call)[0]
    }

    /// `fn(vm, u64, *const Value, *mut Value)` — e.g. host_load_field.
    fn call_hostcall_u64_val(&mut self, name: &str, a: u64, val: Value_, out: Value_) {
        self.emit_line_hint();
        let callee = self.hostcall_addr(name).unwrap();
        let sig = self.import_sig(&[types::I64, self.ptr, self.ptr], None);
        let a_val = self.builder.ins().iconst(types::I64, a as i64);
        self.builder.ins().call_indirect(sig, callee, &[self.vm, a_val, val, out]);
    }

    /// `fn(vm, *const Value, *const Value, *mut Value)` — e.g. host_add.
    fn call_hostcall_2_val(&mut self, name: &str, a: Value_, b: Value_, out: Value_) {
        self.emit_line_hint();
        let callee = self.hostcall_addr(name).unwrap();
        let sig = self.import_sig(&[self.ptr, self.ptr, self.ptr], None);
        self.builder.ins().call_indirect(sig, callee, &[self.vm, a, b, out]);
    }

    /// `fn(vm, *const Value, *const Value, *const Value, *mut Value)` — e.g. host_slice_str.
    fn call_hostcall_3_val(&mut self, name: &str, a: Value_, b: Value_, c: Value_, out: Value_) {
        self.emit_line_hint();
        let callee = self.hostcall_addr(name).unwrap();
        let sig = self.import_sig(&[self.ptr, self.ptr, self.ptr, self.ptr], None);
        self.builder.ins().call_indirect(sig, callee, &[self.vm, a, b, c, out]);
    }

    /// `fn(vm, u64, u64, *const Value, *mut Value)` — e.g. host_call.
    fn call_hostcall_call(&mut self, name: &str, name_idx: u64, arg_count: u64, args: Value_, out: Value_) {
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

    /// `fn(vm, u64, *const Value, *mut Value)` — e.g. host_make_vec.
    fn call_hostcall_make_n(&mut self, name: &str, count: u64, args: Value_, out: Value_) {
        self.emit_line_hint();
        let callee = self.hostcall_addr(name).unwrap();
        let sig = self.import_sig(&[types::I64, self.ptr, self.ptr], None);
        let c = self.builder.ins().iconst(types::I64, count as i64);
        self.builder.ins().call_indirect(sig, callee, &[self.vm, c, args, out]);
    }

    /// `fn(vm, u64, u64, *const Value, *mut Value)` — e.g. host_new_struct.
    fn call_hostcall_new_struct(&mut self, name: &str, name_idx: u64, field_count: u64, args: Value_, out: Value_) {
        self.emit_line_hint();
        let callee = self.hostcall_addr(name).unwrap();
        let sig = self.import_sig(&[types::I64, types::I64, self.ptr, self.ptr], None);
        let a1 = self.builder.ins().iconst(types::I64, name_idx as i64);
        let a2 = self.builder.ins().iconst(types::I64, field_count as i64);
        self.builder.ins().call_indirect(sig, callee, &[self.vm, a1, a2, args, out]);
    }

    /// `fn(vm, u64, u64, *const Value, *mut Value)` — e.g. host_new_union.
    fn call_hostcall_new_union(&mut self, name: &str, name_idx: u64, field_idx: u64, val: Value_, out: Value_) {
        self.emit_line_hint();
        let callee = self.hostcall_addr(name).unwrap();
        let sig = self.import_sig(&[types::I64, types::I64, self.ptr, self.ptr], None);
        let a1 = self.builder.ins().iconst(types::I64, name_idx as i64);
        let a2 = self.builder.ins().iconst(types::I64, field_idx as i64);
        self.builder.ins().call_indirect(sig, callee, &[self.vm, a1, a2, val, out]);
    }

    /// `fn(vm, u64, u64, u64, *const Value, *mut Value)` — e.g. host_make_enum.
    fn call_hostcall_make_enum(&mut self, name: &str, name_idx: u64, variant_idx: u64, field_count: u64, args: Value_, out: Value_) {
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
        self.emit_line_hint();
        let callee = self.hostcall_addr(name).unwrap();
        let sig = self.import_sig(&[types::I64, types::I64, self.ptr, self.ptr], None);
        let a1 = self.builder.ins().iconst(types::I64, rows as i64);
        let a2 = self.builder.ins().iconst(types::I64, cols as i64);
        self.builder.ins().call_indirect(sig, callee, &[self.vm, a1, a2, args, out]);
    }

    /// `fn(vm, u64, *const Value, *const Value, *mut Value)` — e.g. host_store_field.
    fn call_hostcall_store_field(&mut self, name: &str, field_idx: u64, recv: Value_, val: Value_, out: Value_) {
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

    fn emit_binop(&mut self, name: &str) -> Result<(), String> {
        // Stack: [..., a, b]. Pop b, then a. Out at a's position.
        self.sp -= VALUE_SIZE as i32;
        let b_off = self.sp;
        self.sp -= VALUE_SIZE as i32;
        let a_off = self.sp;
        let a_addr = self.builder.ins().stack_addr(self.ptr, self.stack_slot, a_off);
        let b_addr = self.builder.ins().stack_addr(self.ptr, self.stack_slot, b_off);
        let out = a_addr; // result overwrites a's slot
        self.call_hostcall_2_val(name, a_addr, b_addr, out);
        self.bump_sp()?;
        self.emit_err_check_abort();
        Ok(())
    }

    fn emit_unop(&mut self, name: &str) -> Result<(), String> {
        self.sp -= VALUE_SIZE as i32;
        let a_addr = self.builder.ins().stack_addr(self.ptr, self.stack_slot, self.sp);
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
