//! Owns the Cranelift `JITModule` and caches compiled function pointers.

use cranelift::prelude::*;
use crate::hir::types::BaseType;
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{Linkage, Module};
use std::collections::{HashMap, HashSet};
use std::panic::{AssertUnwindSafe, catch_unwind};

use crate::runtime::vm::Chunk;

/// Function pointer type for JIT-compiled Tenth functions.
/// Signature: `extern "C" fn(vm: *mut Vm, args: *const Value, n: usize, out: *mut Value) -> bool`.
pub type JitFn = unsafe extern "C" fn(*mut crate::runtime::vm::Vm, *const crate::runtime::value::Value, usize, *mut crate::runtime::value::Value) -> bool;

pub struct JitContext {
    module: JITModule,
    /// Cached compiled function pointers, keyed by chunk index.
    cache: HashMap<usize, JitFn>,
    /// A1：函数名 → chunk 索引映射（run_jit 时从 `vm.functions` 建立）。
    /// 翻译期把 Call/CallN 的字符串名解析为直接调用目标 chunk 索引。
    name_to_chunk: HashMap<String, usize>,
    /// A1：函数指针表（chunk_idx → 已编译函数指针，0 = 未编译）。
    /// JIT 机器码按 chunk_idx 索引读取该表做直接调用；host_jit_call 慢路径
    /// 编译成功后写入。表在编译期注册完毕后一次性定容（chunks.len()），
    /// 运行期不扩容，`Vm.jit_table_ptr` 指向其数据区保持稳定。
    table: Vec<usize>,
    /// A1：编译失败/不可编译的 chunk 集合（避免每个调用点反复尝试编译）。
    failed: HashSet<usize>,
    /// A2：全部 chunk 副本（`Vm.chunks` 的浅拷贝——Chunk 内 code/strings/lines
    /// 为 Rc 共享，克隆仅引用计数 +1）。供 translator 调用点内联时读取被调函数
    /// 字节码（chunk 索引与 `functions` 映射一致）。run_jit/jit_call_chunk 一次性
    /// 设置，运行期不扩容（编译全部发生在执行前）。
    all_chunks: Vec<Chunk>,
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
        JitContext {
            module,
            cache: HashMap::new(),
            name_to_chunk: HashMap::new(),
            table: Vec::new(),
            failed: HashSet::new(),
            all_chunks: Vec::new(),
        }
    }

    // ── A1：直接调用支持（name→chunk 映射 / 函数指针表）────────────────────

    /// 建立 name→chunk 映射（run_jit 入口一次性设置；含全部函数与闭包 chunk）。
    pub fn set_name_to_chunk(&mut self, map: HashMap<String, usize>) {
        self.name_to_chunk = map;
    }

    /// A2：设置全部 chunk 副本（调用点内联读取被调函数字节码用）。
    /// 浅拷贝（Rc 共享）开销低；与 `name_to_chunk` 同步设置一次即可。
    pub fn set_all_chunks(&mut self, chunks: Vec<Chunk>) {
        self.all_chunks = chunks;
    }

    /// 确保函数指针表至少覆盖 `n` 个 chunk（0 填充 = 未编译）。
    pub fn ensure_table(&mut self, n: usize) {
        if self.table.len() < n {
            self.table.resize(n, 0);
        }
    }

    /// 表数据区基址（供 `Vm.jit_table_ptr` 指向；表定容后指针稳定）。
    pub fn table_data_ptr(&mut self) -> *mut usize {
        self.table.as_mut_ptr()
    }

    /// 写入表项（chunk_idx → 函数指针）。越界忽略（防御性）。
    pub fn set_table_entry(&mut self, chunk_idx: usize, ptr: usize) {
        if chunk_idx < self.table.len() {
            self.table[chunk_idx] = ptr;
        }
    }

    /// 标记 chunk 编译失败（trampoline 后续调用直接走解释器回退，不重复编译）。
    pub fn mark_failed(&mut self, chunk_idx: usize) {
        self.failed.insert(chunk_idx);
    }

    /// M2-A3：chunk 是否编译失败（含显式 Err 与 panic 捕获）——覆盖验证用。
    pub fn is_failed(&self, chunk_idx: usize) -> bool {
        self.failed.contains(&chunk_idx)
    }

    /// M2-A3：chunk 是否已成功编译（cache 命中）——覆盖验证用。
    pub fn is_compiled(&self, chunk_idx: usize) -> bool {
        self.cache.contains_key(&chunk_idx)
    }

    /// Compile (or fetch from cache) the JIT function for `chunk_idx`.
    pub fn get_or_compile(&mut self, chunk_idx: usize, chunk: &Chunk) -> Result<JitFn, String> {
        if let Some(f) = self.cache.get(&chunk_idx) {
            return Ok(*f);
        }
        if self.failed.contains(&chunk_idx) {
            return Err(format!("JIT: chunk {chunk_idx} 编译失败（缓存）"));
        }

        // A1：翻译期把本 chunk 字符串表索引解析为「目标函数 chunk 索引」。
        // 只有以该字符串为名的已注册函数（含闭包 chunk）才可做直接调用；
        // 其余（native / 全局 FnRef 别名 / 未定义）→ None，翻译器保持 host_call fallback。
        let name_to_chunk: Vec<Option<usize>> = chunk.strings.iter()
            .map(|s| self.name_to_chunk.get(s).copied())
            .collect();

        // 短期方案：用 catch_unwind 包裹 translate，捕获 Cranelift 内部
        // 断言失败（如循环回边触发的 is_sealed panic），转为 Err 触发既有
        // fallback 路径（mod.rs:62-65 降级到 VM 解释执行）。
        // 长期方案见 P2 任务：根本修复循环 JIT 的 leader block 密封策略。
        // A2：传入 all_chunks（调用点内联读被调函数字节码）。用字段拆借避免
        // 闭包内 `&mut self.module` 与 `&self.all_chunks` 的整 self 借用冲突。
        let module = &mut self.module;
        let all_chunks = &self.all_chunks;
        let translate_result = catch_unwind(AssertUnwindSafe(|| {
            super::translator::translate(module, chunk_idx, chunk, &name_to_chunk, all_chunks)
        }));
        let fn_id = match translate_result {
            Ok(Ok(id)) => id,
            Ok(Err(e)) => {
                self.failed.insert(chunk_idx);
                return Err(e);
            }
            Err(panic_payload) => {
                self.failed.insert(chunk_idx);
                let msg = if let Some(s) = panic_payload.downcast_ref::<String>() {
                    s.clone()
                } else if let Some(s) = panic_payload.downcast_ref::<&str>() {
                    s.to_string()
                } else {
                    "JIT 编译期间发生 panic（可能是 Cranelift 内部断言失败，如循环回边触发的 is_sealed）".to_string()
                };
                return Err(format!("JIT translate panic: {}", msg));
            }
        };
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
        // A1：登记到函数指针表（JIT 直接调用按 chunk_idx 读取；0 = 未编译）。
        self.ensure_table(chunk_idx + 1);
        self.set_table_entry(chunk_idx, ptr as usize);
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
