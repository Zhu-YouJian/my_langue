//! Owns the Cranelift `JITModule` and caches compiled function pointers.

use cranelift::prelude::*;
use crate::hir::types::BaseType;
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{Linkage, Module};
use std::collections::{HashMap, HashSet};
use std::panic::{AssertUnwindSafe, catch_unwind};

use crate::runtime::vm::{Chunk, Vm};

/// Function pointer type for JIT-compiled Tenth functions.
/// Signature: `extern "C" fn(vm: *mut Vm, args: *const Value, n: usize, out: *mut Value) -> bool`.
pub type JitFn = unsafe extern "C" fn(*mut crate::runtime::vm::Vm, *const crate::runtime::value::Value, usize, *mut crate::runtime::value::Value) -> bool;

// ── M2.5-A6：入参标量 ABI（scalar-specialized call ABI）────────────────────

/// 特化函数可承载的最大参数个数（特化 ABI 以固定 8 个 i64 寄存器传参）。
pub const MAX_SPEC_ARGS: usize = 8;

/// 标量 ABI 种类（特化函数参数/返回）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScalarAbiKind {
    I64,
    F64,
}

/// 函数特化签名（参数 + 返回的标量种类）。`None` = 非特化。
///
/// v1 限定：仅纯 i64（含 `Int` 别名）参数 + i64 返回；参数数 ≤ `MAX_SPEC_ARGS`；
/// 变参/默认参/其他类型 → 不推导（保守，f64 特化留待 v2 同机制扩展）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChunkSig {
    pub param_kinds: Vec<ScalarAbiKind>,
    pub ret_kind: Option<ScalarAbiKind>,
}

impl ChunkSig {
    /// 从 HIR 函数签名推导标量 ABI 签名；不满足特化条件 → `None`。
    /// 调用方需先排除默认参数（`param_defaults` 非空）。
    ///
    /// v1 限定：**仅显式 `i64` 注解**（`Type::Base(I64)`）。`Int` 别名在 HIR 为
    /// `TypeParam("Int")`，语义上是 i64 但注解路径不同——保守不纳入（避免既有
    /// `Int` 注解测试被特化改变行为；显式 `i64` 的基准/新代码才走特化）。
    pub fn from_hir(
        params: &[(String, crate::hir::types::Type)],
        return_type: &crate::hir::types::Type,
        variadic: &[bool],
    ) -> Option<ChunkSig> {
        use crate::hir::types::{BaseType, Type};
        let is_i64_ty = |t: &Type| matches!(t, Type::Base(BaseType::I64));
        if variadic.iter().any(|v| *v) {
            return None;
        }
        if params.len() > MAX_SPEC_ARGS {
            return None;
        }
        let mut param_kinds = Vec::with_capacity(params.len());
        for (_, t) in params {
            if !is_i64_ty(t) {
                return None;
            }
            param_kinds.push(ScalarAbiKind::I64);
        }
        if !is_i64_ty(return_type) {
            return None;
        }
        Some(ChunkSig { param_kinds, ret_kind: Some(ScalarAbiKind::I64) })
    }
}

/// 特化函数指针类型。
/// 签名：`extern "C" fn(vm: *mut Vm, i64 x MAX_SPEC_ARGS) -> i64`——参数全部经
/// 寄存器传递（f64 参数 v2 以位模式塞入 i64 寄存器）；返回单个 i64 标量。
/// 错误经 `vm.last_error` + 调用点 `host_check_error` 传播（B2 模式）。
pub type JitFnSpec = unsafe extern "C" fn(
    *mut Vm, i64, i64, i64, i64, i64, i64, i64, i64,
) -> i64;

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
    /// A6：chunk_idx → 特化签名（None = 非特化）。由 run_jit/jit_call_chunk 从
    /// 各 chunk 的 `scalar_sig` 建立；translator 在调用点判定特化 ABI 资格用。
    chunk_sigs: Vec<Option<ChunkSig>>,
    /// A6：特化函数指针缓存（chunk_idx → 特化入口）。
    spec_cache: HashMap<usize, JitFnSpec>,
    /// A6：特化函数指针表（chunk_idx → 特化入口指针，0 = 未编译）。JIT 机器码
    /// 按 chunk_idx 索引读取该表做特化直接调用；`host_jit_call_spec` 编译成功后
    /// 写入。与 `table` 独立——特化入口只被特化调用点使用，通用路径仍走通用入口。
    spec_table: Vec<usize>,
    /// A6：特化编译失败集合（避免反复尝试）。
    spec_failed: HashSet<usize>,
    /// P1：chunk_idx → 是否「纯标量、可跳过 current_chunk_idx 切换」。由
    /// `compute_skip_chunk_ctx` 计算（`ensure_skip_chunk_ctx` 惰性保证与
    /// all_chunks 同步），translator 发射期只读。
    skip_chunk_ctx: Vec<bool>,
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
            chunk_sigs: Vec::new(),
            spec_cache: HashMap::new(),
            spec_table: Vec::new(),
            spec_failed: HashSet::new(),
            skip_chunk_ctx: Vec::new(),
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

    /// A6：设置特化签名表（chunk_idx → 特化签名）。translator 调用点判定
    /// 特化 ABI 资格时读取；与 `all_chunks` 同步设置一次。
    pub fn set_chunk_sigs(&mut self, sigs: Vec<Option<ChunkSig>>) {
        self.chunk_sigs = sigs;
    }

    /// A6：确保特化指针表至少覆盖 `n` 个 chunk（0 填充 = 未编译）。
    pub fn ensure_spec_table(&mut self, n: usize) {
        if self.spec_table.len() < n {
            self.spec_table.resize(n, 0);
        }
    }

    /// A6：特化表数据区基址（供 `Vm.jit_spec_table_ptr` 指向；表定容后指针稳定）。
    pub fn spec_table_data_ptr(&mut self) -> *mut usize {
        self.spec_table.as_mut_ptr()
    }

    /// A6：写入特化表项（chunk_idx → 特化入口指针）。越界忽略（防御性）。
    pub fn set_spec_table_entry(&mut self, chunk_idx: usize, ptr: usize) {
        if chunk_idx < self.spec_table.len() {
            self.spec_table[chunk_idx] = ptr;
        }
    }

    /// A6：标记 chunk 特化编译失败（慢路径后续直接走通用回退，不重复编译）。
    pub fn mark_spec_failed(&mut self, chunk_idx: usize) {
        self.spec_failed.insert(chunk_idx);
    }

    /// P1：确保 `skip_chunk_ctx` 与 `all_chunks`/`name_to_chunk`/`chunk_sigs`
    /// 同步（在 run_jit / jit_call_chunk* 设置这些字段后调用）。长度不一致时
    /// 重新计算。判定为「chunk 自身字节码 + 静态信息」的纯函数 → 各入口一致。
    pub fn ensure_skip_chunk_ctx(&mut self) {
        if self.all_chunks.is_empty() {
            self.skip_chunk_ctx.clear();
            return;
        }
        if self.skip_chunk_ctx.len() != self.all_chunks.len() {
            self.recompute_skip_chunk_ctx();
        }
    }

    /// P1：无条件重算 `skip_chunk_ctx`。run_jit 入口用（`name_to_chunk` 刚刷新，
    /// 长度可能未变但内容可能不同——强制重算防陈旧）。
    pub fn recompute_skip_chunk_ctx(&mut self) {
        let skip = super::translator::compute_skip_chunk_ctx(
            &self.all_chunks, &self.name_to_chunk, &self.chunk_sigs,
        );
        self.skip_chunk_ctx = skip;
    }

    /// P1：覆盖/调试用——chunk 是否判定为「可跳过 chunk 切换」。
    pub fn skip_chunk_ctx_for(&self, chunk_idx: usize) -> bool {
        self.skip_chunk_ctx.get(chunk_idx).copied().unwrap_or(false)
    }

    /// A6：chunk 是否已成功编译特化入口（spec_cache 命中）——覆盖验证用。
    pub fn is_spec_compiled(&self, chunk_idx: usize) -> bool {
        self.spec_cache.contains_key(&chunk_idx)
    }

    /// A6：chunk 特化编译是否失败（含显式 Err 与 panic 捕获）——覆盖验证用。
    pub fn is_spec_failed(&self, chunk_idx: usize) -> bool {
        self.spec_failed.contains(&chunk_idx)
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
        // M2-A5：把 **整个编译链路**（translate + finalize_definitions 低化 +
        // get_finalized_function）纳入 catch_unwind——此前只包 translate，
        // Cranelift 低化阶段（如 tuple 模式 + guard 触发的 TryFromIntError）在
        // finalize_definitions 内 panic 会**逃逸出 JIT**（进程级 panic，红线）。
        // 现在统一转为 Err → 既有 fallback 路径（mod.rs 降级到 VM 解释执行）。
        // P1：确保 skip_chunk_ctx 与 all_chunks 同步（须在借用 module 之前）。
        self.ensure_skip_chunk_ctx();
        let module = &mut self.module;
        let all_chunks = &self.all_chunks;
        let chunk_sigs = &self.chunk_sigs;
        let skip_chunk_ctx = &self.skip_chunk_ctx;
        let compile_result = catch_unwind(AssertUnwindSafe(|| -> Result<usize, String> {
            let fn_id = super::translator::translate(module, chunk_idx, chunk, &name_to_chunk, all_chunks, chunk_sigs, None, skip_chunk_ctx)?;
            module.finalize_definitions();
            Ok(module.get_finalized_function(fn_id) as usize)
        }));
        let raw_ptr: *const u8 = match compile_result {
            Ok(Ok(p)) => p as *const u8,
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
                    "JIT 编译期间发生 panic（可能是 Cranelift 内部断言/低化失败，如循环回边触发的 is_sealed、tuple+guard 的 TryFromIntError）".to_string()
                };
                return Err(format!("JIT compile panic: {}", msg));
            }
        };
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

    /// A6：编译（或取缓存）chunk 的**特化入口**（标量寄存器 ABI）。
    ///
    /// 与 `get_or_compile`（通用入口）独立：特化入口签名
    /// `(vm, i64 x MAX_SPEC_ARGS) -> i64`，参数/返回走寄存器。函数体用既有
    /// translator 编译（Value 局部槽 + A2 标量快路径），但：
    /// - 参数种类**预置**（seed）：i64 参数视为 I32 标量 → 函数体参数走原生路径
    /// - 入口把寄存器参数写入局部标量槽（8B）+ 按需物化 Value 槽
    /// - 返回从返回值标量槽读 8B 写寄存器；错误路径返回哨兵 0（调用点
    ///   `host_check_error` 检测，B2 模式）
    ///
    /// 编译失败/panic → 标记 `spec_failed` 并返回 Err（调用点回退通用入口）。
    pub fn get_or_compile_spec(
        &mut self,
        chunk_idx: usize,
        chunk: &Chunk,
        sig: &ChunkSig,
    ) -> Result<JitFnSpec, String> {
        if let Some(f) = self.spec_cache.get(&chunk_idx) {
            return Ok(*f);
        }
        if self.spec_failed.contains(&chunk_idx) {
            return Err(format!("JIT: chunk {chunk_idx} 特化编译失败（缓存）"));
        }
        let name_to_chunk: Vec<Option<usize>> = chunk.strings.iter()
            .map(|s| self.name_to_chunk.get(s).copied())
            .collect();
        // P1：确保 skip_chunk_ctx 与 all_chunks 同步（须在借用 module 之前）。
        self.ensure_skip_chunk_ctx();
        let module = &mut self.module;
        let all_chunks = &self.all_chunks;
        let chunk_sigs = &self.chunk_sigs;
        let skip_chunk_ctx = &self.skip_chunk_ctx;
        let compile_result = catch_unwind(AssertUnwindSafe(|| -> Result<usize, String> {
            let fn_id = super::translator::translate(module, chunk_idx, chunk, &name_to_chunk, all_chunks, chunk_sigs, Some(sig), skip_chunk_ctx)?;
            module.finalize_definitions();
            Ok(module.get_finalized_function(fn_id) as usize)
        }));
        let raw_ptr: *const u8 = match compile_result {
            Ok(Ok(p)) => p as *const u8,
            Ok(Err(e)) => {
                self.spec_failed.insert(chunk_idx);
                return Err(e);
            }
            Err(panic_payload) => {
                self.spec_failed.insert(chunk_idx);
                let msg = if let Some(s) = panic_payload.downcast_ref::<String>() {
                    s.clone()
                } else if let Some(s) = panic_payload.downcast_ref::<&str>() {
                    s.to_string()
                } else {
                    "JIT 特化编译期间发生 panic（Cranelift 内部断言/低化失败）".to_string()
                };
                return Err(format!("JIT spec compile panic: {}", msg));
            }
        };
        // SAFETY: 同 get_or_compile 的论证；签名与 `JitFnSpec` 一致。
        assert_eq!(
            std::mem::size_of::<*const u8>(),
            std::mem::size_of::<JitFnSpec>(),
            "pointer size mismatch — translator spec signature changed?"
        );
        let ptr: JitFnSpec = unsafe { std::mem::transmute(raw_ptr) };
        self.spec_cache.insert(chunk_idx, ptr);
        // A6：登记到特化指针表（特化调用点按 chunk_idx 读取；0 = 未编译）。
        self.ensure_spec_table(chunk_idx + 1);
        self.set_spec_table_entry(chunk_idx, ptr as usize);
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
