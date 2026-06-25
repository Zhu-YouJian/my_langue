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
use cranelift_module::{Linkage, Module};
use std::collections::{HashMap, HashSet};
use std::mem::size_of;

use crate::runtime::vm::{Chunk, Op};
use crate::runtime::value::Value;

// Cranelift's `StackSlot` and `SigRef` entity types live in `codegen::ir`
// and are NOT re-exported from the prelude. Bring them in explicitly.
use cranelift::codegen::ir::{StackSlot, SigRef};

/// Size of one `Value` on the stack.
const VALUE_SIZE: u32 = size_of::<Value>() as u32;

/// Maximum virtual-stack depth (number of Values). Functions exceeding
/// this will need a larger area.
const MAX_STACK_DEPTH: u32 = 256;

pub fn translate<M: Module>(
    module: &mut M,
    chunk_idx: usize,
    chunk: &Chunk,
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
        match op {
            PushInt(n) => {
                let out = self.stack_addr_at_sp();
                self.call_hostcall_i64("host_make_int", n, out);
                self.sp += VALUE_SIZE as i32;
            }
            PushFloat(f) => {
                let out = self.stack_addr_at_sp();
                self.call_hostcall_f64("host_make_float", f, out);
                self.sp += VALUE_SIZE as i32;
            }
            PushFloat32(f) => {
                // JIT 路径暂降级为 f64（Phase 5 补齐真正的 f32 JIT）
                let out = self.stack_addr_at_sp();
                self.call_hostcall_f64("host_make_float", f as f64, out);
                self.sp += VALUE_SIZE as i32;
            }
            PushBool(b) => {
                let out = self.stack_addr_at_sp();
                self.call_hostcall_u8("host_make_bool", if b { 1 } else { 0 }, out);
                self.sp += VALUE_SIZE as i32;
            }
            PushStr(i) => {
                let out = self.stack_addr_at_sp();
                self.call_hostcall_u64("host_make_str", i as u64, out);
                self.sp += VALUE_SIZE as i32;
            }
            PushUnit => {
                let out = self.stack_addr_at_sp();
                self.call_hostcall_unit("host_make_unit", out);
                self.sp += VALUE_SIZE as i32;
            }
            Pop => { self.sp -= VALUE_SIZE as i32; }
            Dup => {
                let src_off = self.sp - VALUE_SIZE as i32;
                let dst_off = self.sp;
                self.copy_within_stack(src_off, dst_off);
                self.sp += VALUE_SIZE as i32;
            }
            Load(i) => {
                let src = self.locals.get(&i).copied().ok_or("Load: bad local")?;
                let dst_off = self.sp;
                self.copy_slot_to_stack(src, 0, dst_off);
                self.sp += VALUE_SIZE as i32;
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
                self.sp += VALUE_SIZE as i32;
            }
            StoreGlobal(i) => {
                self.sp -= VALUE_SIZE as i32;
                let val_off = self.sp;
                let val_addr = self.builder.ins().stack_addr(self.ptr, self.stack_slot, val_off);
                let out_addr = val_addr; // result = stored value, same slot
                self.call_hostcall_u64_val("host_store_global", i as u64, val_addr, out_addr);
                self.sp += VALUE_SIZE as i32; // push result back
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
                self.call_hostcall_call("host_call", i as u64, 0, null_ptr, out);
                self.sp += VALUE_SIZE as i32;
            }
            CallN(i, n) => {
                // Args are at [sp - n*VS, sp). Pop them, then out is at sp.
                self.sp -= (n as i32) * (VALUE_SIZE as i32);
                let args_addr = self.builder.ins().stack_addr(self.ptr, self.stack_slot, self.sp);
                let out = self.stack_addr_at_sp();
                self.call_hostcall_call("host_call", i as u64, n as u64, args_addr, out);
                self.sp += VALUE_SIZE as i32;
            }
            MethodCall(i, n) => {
                self.sp -= (n as i32) * (VALUE_SIZE as i32);
                let args_addr = self.builder.ins().stack_addr(self.ptr, self.stack_slot, self.sp);
                let out = self.stack_addr_at_sp();
                self.call_hostcall_call("host_method_call", i as u64, n as u64, args_addr, out);
                self.sp += VALUE_SIZE as i32;
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
                self.sp += VALUE_SIZE as i32;
            }
            MakeMap(n) => {
                let total = n * 2;
                self.sp -= (total as i32) * (VALUE_SIZE as i32);
                let args_addr = self.builder.ins().stack_addr(self.ptr, self.stack_slot, self.sp);
                let out = self.stack_addr_at_sp();
                self.call_hostcall_make_n("host_make_map", n as u64, args_addr, out);
                self.sp += VALUE_SIZE as i32;
            }
            NewStruct(name_i, field_count) => {
                let total = field_count * 2;
                self.sp -= (total as i32) * (VALUE_SIZE as i32);
                let args_addr = self.builder.ins().stack_addr(self.ptr, self.stack_slot, self.sp);
                let out = self.stack_addr_at_sp();
                self.call_hostcall_new_struct("host_new_struct", name_i as u64, field_count as u64, args_addr, out);
                self.sp += VALUE_SIZE as i32;
            }
            LoadField(i) => {
                self.sp -= VALUE_SIZE as i32;
                let recv_addr = self.builder.ins().stack_addr(self.ptr, self.stack_slot, self.sp);
                let out = self.stack_addr_at_sp();
                self.call_hostcall_u64_val("host_load_field", i as u64, recv_addr, out);
                self.sp += VALUE_SIZE as i32;
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
                self.sp += VALUE_SIZE as i32; // push result back
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
                self.sp += VALUE_SIZE as i32;
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
                self.sp += VALUE_SIZE as i32;
            }
            MakeEnum(name_i, variant_i, field_count) => {
                self.sp -= (field_count as i32) * (VALUE_SIZE as i32);
                let args_addr = self.builder.ins().stack_addr(self.ptr, self.stack_slot, self.sp);
                let out = self.stack_addr_at_sp();
                self.call_hostcall_make_enum("host_make_enum", name_i as u64, variant_i as u64, field_count as u64, args_addr, out);
                self.sp += VALUE_SIZE as i32;
            }
            IsEnumVariant(variant_i) => {
                self.sp -= VALUE_SIZE as i32;
                let recv_addr = self.builder.ins().stack_addr(self.ptr, self.stack_slot, self.sp);
                let out = self.stack_addr_at_sp();
                self.call_hostcall_u64_val("host_is_enum_variant", variant_i as u64, recv_addr, out);
                self.sp += VALUE_SIZE as i32;
            }
            EnumGetField(field_i) => {
                self.sp -= VALUE_SIZE as i32;
                let recv_addr = self.builder.ins().stack_addr(self.ptr, self.stack_slot, self.sp);
                let out = self.stack_addr_at_sp();
                self.call_hostcall_u64_val("host_enum_get_field", field_i as u64, recv_addr, out);
                self.sp += VALUE_SIZE as i32;
            }
            PushRange(s, e, inc) => {
                let out = self.stack_addr_at_sp();
                self.call_hostcall_push_range("host_push_range", s, e, if inc { 1 } else { 0 }, out);
                self.sp += VALUE_SIZE as i32;
            }
            MoveOp => { /* no-op */ }
            MakeTensor(rows, cols, dtype) => {
                let count = rows * cols;
                self.sp -= (count as i32) * (VALUE_SIZE as i32);
                let args_addr = self.builder.ins().stack_addr(self.ptr, self.stack_slot, self.sp);
                let out = self.stack_addr_at_sp();
                self.call_hostcall_make_tensor("host_make_tensor", rows as u64, cols as u64, args_addr, out);
                // 注：dtype 当前在 JIT 路径降级为 F64；后续 phase 5 中补齐 f32 路径。
                let _ = dtype;
                self.sp += VALUE_SIZE as i32;
            }
            MakeClosure(params, chunk_idx) => {
                let out = self.stack_addr_at_sp();
                self.call_hostcall_2_u64("host_make_closure", params as u64, chunk_idx as u64, out);
                self.sp += VALUE_SIZE as i32;
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

    // ── Hostcall emitters (all take raw `Value_` addresses for Value params) ─

    fn call_hostcall_unit(&mut self, name: &str, out: Value_) {
        let callee = self.hostcall_addr(name).unwrap();
        let sig = self.import_sig(&[self.ptr], None);
        self.builder.ins().call_indirect(sig, callee, &[self.vm, out]);
    }

    fn call_hostcall_i64(&mut self, name: &str, arg: i64, out: Value_) {
        let callee = self.hostcall_addr(name).unwrap();
        let sig = self.import_sig(&[types::I64, self.ptr], None);
        let a = self.builder.ins().iconst(types::I64, arg);
        self.builder.ins().call_indirect(sig, callee, &[self.vm, a, out]);
    }

    fn call_hostcall_f64(&mut self, name: &str, arg: f64, out: Value_) {
        let callee = self.hostcall_addr(name).unwrap();
        let sig = self.import_sig(&[types::F64, self.ptr], None);
        let a = self.builder.ins().f64const(arg);
        self.builder.ins().call_indirect(sig, callee, &[self.vm, a, out]);
    }

    fn call_hostcall_u8(&mut self, name: &str, arg: u8, out: Value_) {
        let callee = self.hostcall_addr(name).unwrap();
        let sig = self.import_sig(&[types::I8, self.ptr], None);
        let a = self.builder.ins().iconst(types::I8, arg as i64);
        self.builder.ins().call_indirect(sig, callee, &[self.vm, a, out]);
    }

    fn call_hostcall_u64(&mut self, name: &str, arg: u64, out: Value_) {
        self.call_hostcall_i64(name, arg as i64, out);
    }

    fn call_hostcall_val_ret_u8(&mut self, name: &str, arg: Value_) -> Value_ {
        let callee = self.hostcall_addr(name).unwrap();
        let sig = self.import_sig(&[self.ptr], Some(types::I8));
        let call = self.builder.ins().call_indirect(sig, callee, &[self.vm, arg]);
        self.builder.inst_results(call)[0]
    }

    /// `fn(vm, u64, *const Value, *mut Value)` — e.g. host_load_field.
    fn call_hostcall_u64_val(&mut self, name: &str, a: u64, val: Value_, out: Value_) {
        let callee = self.hostcall_addr(name).unwrap();
        let sig = self.import_sig(&[types::I64, self.ptr, self.ptr], None);
        let a_val = self.builder.ins().iconst(types::I64, a as i64);
        self.builder.ins().call_indirect(sig, callee, &[self.vm, a_val, val, out]);
    }

    /// `fn(vm, *const Value, *const Value, *mut Value)` — e.g. host_add.
    fn call_hostcall_2_val(&mut self, name: &str, a: Value_, b: Value_, out: Value_) {
        let callee = self.hostcall_addr(name).unwrap();
        let sig = self.import_sig(&[self.ptr, self.ptr, self.ptr], None);
        self.builder.ins().call_indirect(sig, callee, &[self.vm, a, b, out]);
    }

    /// `fn(vm, *const Value, *const Value, *const Value, *mut Value)` — e.g. host_slice_str.
    fn call_hostcall_3_val(&mut self, name: &str, a: Value_, b: Value_, c: Value_, out: Value_) {
        let callee = self.hostcall_addr(name).unwrap();
        let sig = self.import_sig(&[self.ptr, self.ptr, self.ptr, self.ptr], None);
        self.builder.ins().call_indirect(sig, callee, &[self.vm, a, b, c, out]);
    }

    /// `fn(vm, u64, u64, *mut Value)` — e.g. host_make_closure.
    fn call_hostcall_2_u64(&mut self, name: &str, a: u64, b: u64, out: Value_) {
        let callee = self.hostcall_addr(name).unwrap();
        let sig = self.import_sig(&[types::I64, types::I64, self.ptr], None);
        let a_val = self.builder.ins().iconst(types::I64, a as i64);
        let b_val = self.builder.ins().iconst(types::I64, b as i64);
        self.builder.ins().call_indirect(sig, callee, &[self.vm, a_val, b_val, out]);
    }

    /// `fn(vm, u64, u64, *const Value, *mut Value)` — e.g. host_call.
    fn call_hostcall_call(&mut self, name: &str, name_idx: u64, arg_count: u64, args: Value_, out: Value_) {
        let callee = self.hostcall_addr(name).unwrap();
        let sig = self.import_sig(&[types::I64, types::I64, self.ptr, self.ptr], None);
        let a1 = self.builder.ins().iconst(types::I64, name_idx as i64);
        let a2 = self.builder.ins().iconst(types::I64, arg_count as i64);
        self.builder.ins().call_indirect(sig, callee, &[self.vm, a1, a2, args, out]);
    }

    /// `fn(vm, u64, *const Value, *mut Value)` — e.g. host_make_vec.
    fn call_hostcall_make_n(&mut self, name: &str, count: u64, args: Value_, out: Value_) {
        let callee = self.hostcall_addr(name).unwrap();
        let sig = self.import_sig(&[types::I64, self.ptr, self.ptr], None);
        let c = self.builder.ins().iconst(types::I64, count as i64);
        self.builder.ins().call_indirect(sig, callee, &[self.vm, c, args, out]);
    }

    /// `fn(vm, u64, u64, *const Value, *mut Value)` — e.g. host_new_struct.
    fn call_hostcall_new_struct(&mut self, name: &str, name_idx: u64, field_count: u64, args: Value_, out: Value_) {
        let callee = self.hostcall_addr(name).unwrap();
        let sig = self.import_sig(&[types::I64, types::I64, self.ptr, self.ptr], None);
        let a1 = self.builder.ins().iconst(types::I64, name_idx as i64);
        let a2 = self.builder.ins().iconst(types::I64, field_count as i64);
        self.builder.ins().call_indirect(sig, callee, &[self.vm, a1, a2, args, out]);
    }

    /// `fn(vm, u64, u64, u64, *const Value, *mut Value)` — e.g. host_make_enum.
    fn call_hostcall_make_enum(&mut self, name: &str, name_idx: u64, variant_idx: u64, field_count: u64, args: Value_, out: Value_) {
        let callee = self.hostcall_addr(name).unwrap();
        let sig = self.import_sig(&[types::I64, types::I64, types::I64, self.ptr, self.ptr], None);
        let a1 = self.builder.ins().iconst(types::I64, name_idx as i64);
        let a2 = self.builder.ins().iconst(types::I64, variant_idx as i64);
        let a3 = self.builder.ins().iconst(types::I64, field_count as i64);
        self.builder.ins().call_indirect(sig, callee, &[self.vm, a1, a2, a3, args, out]);
    }

    /// `fn(vm, i64, i64, u8, *mut Value)` — e.g. host_push_range.
    fn call_hostcall_push_range(&mut self, name: &str, start: i64, end: i64, inc: u8, out: Value_) {
        let callee = self.hostcall_addr(name).unwrap();
        let sig = self.import_sig(&[types::I64, types::I64, types::I8, self.ptr], None);
        let a1 = self.builder.ins().iconst(types::I64, start);
        let a2 = self.builder.ins().iconst(types::I64, end);
        let a3 = self.builder.ins().iconst(types::I8, inc as i64);
        self.builder.ins().call_indirect(sig, callee, &[self.vm, a1, a2, a3, out]);
    }

    /// `fn(vm, u64, u64, *const Value, *mut Value)` — e.g. host_make_tensor.
    fn call_hostcall_make_tensor(&mut self, name: &str, rows: u64, cols: u64, args: Value_, out: Value_) {
        let callee = self.hostcall_addr(name).unwrap();
        let sig = self.import_sig(&[types::I64, types::I64, self.ptr, self.ptr], None);
        let a1 = self.builder.ins().iconst(types::I64, rows as i64);
        let a2 = self.builder.ins().iconst(types::I64, cols as i64);
        self.builder.ins().call_indirect(sig, callee, &[self.vm, a1, a2, args, out]);
    }

    /// `fn(vm, u64, *const Value, *const Value, *mut Value)` — e.g. host_store_field.
    fn call_hostcall_store_field(&mut self, name: &str, field_idx: u64, recv: Value_, val: Value_, out: Value_) {
        let callee = self.hostcall_addr(name).unwrap();
        let sig = self.import_sig(&[types::I64, self.ptr, self.ptr, self.ptr], None);
        let a1 = self.builder.ins().iconst(types::I64, field_idx as i64);
        self.builder.ins().call_indirect(sig, callee, &[self.vm, a1, recv, val, out]);
    }

    // ── Binop / unop ───────────────────────────────────────────────────────

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
        self.sp += VALUE_SIZE as i32;
        Ok(())
    }

    fn emit_unop(&mut self, name: &str) -> Result<(), String> {
        self.sp -= VALUE_SIZE as i32;
        let a_addr = self.builder.ins().stack_addr(self.ptr, self.stack_slot, self.sp);
        let out = a_addr; // result overwrites operand slot
        let callee = self.hostcall_addr(name)?;
        let sig = self.import_sig(&[self.ptr, self.ptr], None);
        self.builder.ins().call_indirect(sig, callee, &[self.vm, a_addr, out]);
        self.sp += VALUE_SIZE as i32;
        Ok(())
    }
}
