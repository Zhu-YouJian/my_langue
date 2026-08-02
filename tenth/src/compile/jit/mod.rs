//! Cranelift-based JIT compiler for Tenth bytecode.
//!
//! Strategy (conservative JIT):
//! - Compiles a single `Chunk` into a Cranelift function with signature
//!   `extern "C" fn(vm: *mut Vm, args: *const Value, n: usize, out: *mut Value) -> bool`.
//! - Uses a compile-time virtual stack of SSA values; never touches the
//!   runtime `Vm.stack` for pure-scalar operations.
//! - All complex operations (calls, heap allocations, field access, tensor
//!   ops, autodiff recording) are routed through host trampolines defined in
//!   [`hostcalls`]. The JIT simply emits a call to the trampoline and pushes
//!   the returned `Value` onto the virtual stack.
//! - Autodiff safety: if `vm.recording` is true at function entry, the JIT
//!   function immediately delegates to `Vm::run` (interpreter) to preserve
//!   tape semantics. This keeps the hot path fast while staying correct.
//!
//! Module layout:
//! - [`context`]  — `JitContext` owns the `JITModule` and caches compiled
//!                   functions by chunk index.
//! - [`translator`]— core bytecode → Cranelift IR translator.
//! - [`hostcalls`] — `extern "C"` trampolines bridging JIT code back to `Vm`.

pub mod context;
pub mod translator;
pub mod hostcalls;

use crate::error::{TenthError, TenthResult};
use crate::runtime::value::Value;
use crate::runtime::vm::Vm;
use context::JitContext;

/// Entry point: run the function `name` on `vm` via JIT, falling back to
/// the interpreter when JIT compilation is not possible (e.g. recording
/// mode, unsupported opcode, or compilation failure).
///
/// On success, returns the function's result value. On failure, returns an
/// error that callers can retry via `Vm::call`.
pub fn run_jit(vm: &mut Vm, name: &str) -> TenthResult<Value> {
    // Safety gate: never JIT when autodiff is recording — tape writes happen
    // inside the interpreter's Add/Sub/Mul/Div and tensor method handlers,
    // and JIT-compiled scalar arithmetic would silently skip them.
    if vm.is_recording() {
        return vm.call(name);
    }

    let chunk_idx = match vm.chunk_index_of(name) {
        Some(i) => i,
        None => return vm.call(name), // unknown function → let VM produce the error
    };

    // Lazily initialise the JIT context on the Vm.
    if vm.jit_ctx.is_none() {
        vm.jit_ctx = Some(JitContext::new());
    }

    // Snapshot the chunk's data BEFORE borrowing `ctx` mutably, so the
    // borrow checker sees `vm` as borrowed only once at a time. Chunks are
    // small (bytecode + string pool), so cloning is cheap relative to
    // compilation cost.
    let chunk_view = vm.chunk_at(chunk_idx).clone();

    let ctx = vm.jit_ctx.as_mut().unwrap();
    let fn_ptr = match ctx.get_or_compile(chunk_idx, &chunk_view) {
        Ok(p) => p,
        Err(_) => return vm.call(name), // compilation failed → fallback
    };

    // Marshal arguments: the VM convention is that args are already pushed
    // onto `vm.stack` right-to-left. Pop them into a flat slice, run the JIT
    // function, then push the result back.
    let num_args = chunk_view.num_args;
    let base = vm.stack_len().saturating_sub(num_args);
    let args: Vec<Value> = (0..num_args)
        .map(|_| vm.stack_pop())
        .collect();
    debug_assert_eq!(vm.stack_len(), base);

    // Set current_chunk_idx so hostcall trampolines can resolve string indices.
    vm.current_chunk_idx = chunk_idx;

    let mut out = Value::Unit;
    let ok = unsafe { hostcalls::invoke_jit(fn_ptr, vm as *mut Vm, &args, &mut out) };
    if ok {
        // B2: 安全网——即使 JIT 返回 ok=true，也检查是否有 hostcall 设置了
        // last_error（如 host_call / host_index_get 等非 MethodCall hostcall 报错）。
        // 若有错误，surface 之并触发 fallback，而非静默返回可能的 Unit。
        // 9c：last_error 携带行号（`(line, message)`），构造 RuntimeError 时带上，
        // 使 JIT 报错与 VM 的 err_here/with_line 行为一致（对齐报错文案）。
        if let Some((line, msg)) = vm.take_last_error() {
            return Err(TenthError::RuntimeError { line, col: None, message: msg });
        }
        vm.stack_push(out.clone());
        Ok(out)
    } else {
        // The trampoline sets the VM's last-error field; surface it here.
        let (line, msg) = vm.take_last_error().unwrap_or((None, "JIT 执行失败".into()));
        Err(TenthError::RuntimeError { line, col: None, message: msg })
    }
}
