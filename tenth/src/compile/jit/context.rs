//! Owns the Cranelift `JITModule` and caches compiled function pointers.

use cranelift::prelude::*;
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{Linkage, Module};
use std::collections::HashMap;

use crate::runtime::vm::Chunk;

/// Function pointer type for JIT-compiled Tenth functions.
/// Signature: `extern "C" fn(vm: *mut Vm, args: *const Value, n: usize, out: *mut Value) -> bool`.
pub type JitFn = unsafe extern "C" fn(*mut crate::runtime::vm::Vm, *const crate::runtime::value::Value, usize, *mut crate::runtime::value::Value) -> bool;

pub struct JitContext {
    module: JITModule,
    /// Cached compiled function pointers, keyed by chunk index.
    cache: HashMap<usize, JitFn>,
}

impl JitContext {
    pub fn new() -> Self {
        let mut flag_builder = settings::builder();
        // Enable basic optimisations; disable unwinding across FFI.
        flag_builder.set("use_colocated_libcalls", "false").unwrap();
        // PIC must be disabled for call_indirect to work correctly with
        // absolute hostcall addresses on Windows x64.
        flag_builder.set("is_pic", "false").unwrap();
        let isa_builder = cranelift_native::builder().expect("host machine not supported");
        let isa = isa_builder.finish(settings::Flags::new(flag_builder)).unwrap();
        let builder = JITBuilder::with_isa(isa, cranelift_module::default_libcall_names());
        let module = JITModule::new(builder);
        JitContext { module, cache: HashMap::new() }
    }

    /// Compile (or fetch from cache) the JIT function for `chunk_idx`.
    pub fn get_or_compile(&mut self, chunk_idx: usize, chunk: &Chunk) -> Result<JitFn, String> {
        if let Some(f) = self.cache.get(&chunk_idx) {
            return Ok(*f);
        }

        let fn_id = super::translator::translate(&mut self.module, chunk_idx, chunk)?;
        self.module.finalize_definitions();
        let raw_ptr = self.module.get_finalized_function(fn_id);
        // `get_finalized_function` returns `*const u8`; transmute to the
        // typed function pointer. This is safe because we declared the
        // signature with the same calling convention and parameter types.
        let ptr: JitFn = unsafe { std::mem::transmute(raw_ptr) };
        self.cache.insert(chunk_idx, ptr);
        Ok(ptr)
    }
}

impl Drop for JitContext {
    fn drop(&mut self) {
        // Free compiled code by invalidating everything. The JITModule
        // itself cleans up its mappings on drop.
        // (JITModule doesn't expose a public `finish`; rely on Drop.)
    }
}
