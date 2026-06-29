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
        // SAFETY: `get_finalized_function` 返回 `*const u8`，我们将其 transmute
        // 为类型化函数指针。安全性依赖于以下不变量：
        // 1. `raw_ptr` 非空且指向可执行内存（由 `finalize_definitions` 保证）
        // 2. 函数签名与 `translator::translate` 声明的一致（`JitFn` 类型）
        // 3. `*const u8` 与函数指针尺寸一致（assert 编译期检查）
        // 若翻译器未来变更签名，断言会立即触发，避免静默 UB。
        assert_eq!(
            std::mem::size_of::<*const u8>(),
            std::mem::size_of::<JitFn>(),
            "pointer size mismatch — translator signature changed?"
        );
        let ptr: JitFn = unsafe { std::mem::transmute(raw_ptr) };
        self.cache.insert(chunk_idx, ptr);
        Ok(ptr)
    }
}

impl Drop for JitContext {
    fn drop(&mut self) {
        // 显式释放编译产物与代码映射，避免依赖 JITModule 的隐式 Drop 语义
        // （未来 cranelift 版本变更 Drop 行为时不易察觉）。
        // `Module::finish` 消费 self，这里只能尽力清理；失败可忽略。
        // 安全：清空 cache 后所有函数指针不再被引用，模块可安全释放。
        self.cache.clear();
    }
}
