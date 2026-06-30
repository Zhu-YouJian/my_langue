//! HIR -> WebAssembly bytecode compiler.
//!
//! Generates a WASM module from a Tenth HIR program using `wasm-encoder`.
//! The module is designed to be executed by `wasmi` (embedded in the Rust host).

mod types;
mod sections;
mod compile;
mod closures;
mod host;

pub use self::host::{register_host_functions, run_wasm_module};

use std::collections::HashMap;
use wasm_encoder::{Module, ValType};
use crate::error::{TenthError, TenthResult};
use crate::hir::hir::*;
use crate::hir::types::{BaseType, Type};

// ── Type mapping ───────────────────────────────────────────────────────────

pub(super) fn to_val_type(ty: &Type) -> Option<ValType> {
    match ty {
        Type::Base(b) => match b {
            BaseType::I8 | BaseType::I16 | BaseType::I32 | BaseType::I64 => Some(ValType::I64),
            BaseType::F32 | BaseType::F64 => Some(ValType::F64),
            BaseType::Bool => Some(ValType::I32),
            BaseType::Str => Some(ValType::I64), // stored as i64 internally, converted at host boundary
            BaseType::Unit => None,
            _ => None,
        },
        Type::Ref(_) | Type::MutRef(_) => Some(ValType::I64),
        Type::Struct(_) => Some(ValType::I64),
        Type::TypeParam { .. } => Some(ValType::I64), // unresolved generic/struct → i64
        Type::Generic { .. } => Some(ValType::I64),   // Vec<T>, etc. → i64 pointer
        Type::Unknown => Some(ValType::I64),
        _ => None,
    }
}

pub(super) fn to_val_type_required(ty: &Type) -> TenthResult<ValType> {
    to_val_type(ty).ok_or_else(|| TenthError::RuntimeError {
        message: format!("无法将类型 {:?} 映射到 WASM 值类型", ty),
    })
}

/// Compute byte size and WASM type for a struct field. All fields are 8 bytes.
pub(super) fn field_size_and_type(ty: &Type) -> (u32, ValType) {
    match ty {
        Type::Base(b) => match b {
            BaseType::I8 | BaseType::I16 | BaseType::I32 | BaseType::I64 => (8, ValType::I64),
            BaseType::F32 | BaseType::F64 => (8, ValType::F64),
            BaseType::Bool => (8, ValType::I32),
            BaseType::Str => (8, ValType::I64),
            _ => (8, ValType::I64),
        },
        _ => (8, ValType::I64),
    }
}

// ── Compiler state ─────────────────────────────────────────────────────────

/// First user-function index (after all host imports).
///
/// Host import indices are named constants so that `Call(N)` sites in
/// `compile.rs` reference symbolic names instead of magic numbers. Adding a
/// new host import only requires appending a `HOST_*` constant below and
/// registering the matching type+import in `sections.rs` (and the
/// implementation in `host.rs`). `IMPORT_COUNT` is derived from the last
/// index, so it stays in sync automatically.
pub(super) const HOST_PRINTLN: u32 = 0;
pub(super) const HOST_WRITE_FILE: u32 = 1;
pub(super) const HOST_READ_FILE: u32 = 2;
pub(super) const HOST_STR_ADD: u32 = 3;
pub(super) const HOST_STR_EQ: u32 = 4;
pub(super) const HOST_STR_INT: u32 = 5;
pub(super) const HOST_TENTH_ALLOC: u32 = 6;
pub(super) const HOST_VEC_NEW: u32 = 7;
pub(super) const HOST_VEC_PUSH: u32 = 8;
pub(super) const HOST_VEC_LEN: u32 = 9;
pub(super) const HOST_VEC_GET: u32 = 10;
pub(super) const HOST_COMPILE_HOST: u32 = 11;
pub(super) const HOST_STR_LEN: u32 = 12;
pub(super) const HOST_STR_AT: u32 = 13;
pub(super) const HOST_STR_CMP: u32 = 14;
pub(super) const HOST_F64_BITS: u32 = 15;
pub(super) const HOST_STR_SLICE: u32 = 16;
pub(super) const HOST_TENSOR_FROM_VEC: u32 = 17;
pub(super) const IMPORT_COUNT: u32 = HOST_TENSOR_FROM_VEC + 1;

pub struct WasmCompiler {
    type_cache: HashMap<(Vec<ValType>, Vec<ValType>), u32>,
    func_map: HashMap<String, u32>,
    hir_funcs: Vec<HirFnDef>,
    string_data: Vec<u8>,
    string_offsets: HashMap<String, u32>,
    /// Struct name -> field name -> (byte offset, field size, WASM type)
    struct_layouts: HashMap<String, HashMap<String, (u32, u32, ValType)>>,
    /// Variable name -> local index (all i64 typed)
    local_map: HashMap<String, u32>,
    local_count: u32,
    /// Number of function parameters (params use correct WASM types, not i64)
    param_count: u32,
    /// Stack of (If-block depth, break_offset) per enclosing loop.
    /// Break emits Br(1 + break_offset + if_depth), Continue emits Br(if_depth).
    /// break_offset=0 for while/loop, 1 for for (inner body block).
    if_depths: Vec<(u32, u32)>,
    // ── Closure tracking (D5) ──
    /// Closure idx -> (func_idx, type_idx, param_count)
    closure_info: Vec<(u32, u32, u32)>,
    /// Closure expr pointer -> closure idx (for lookup during compile_expr)
    closure_expr_map: HashMap<usize, usize>,
    /// Captures for each closure (parallel to closure_info)
    closure_captures: Vec<Vec<String>>,
    /// Variable name -> type_idx for closure variables (for call_indirect)
    closure_vars: HashMap<String, u32>,
    /// Current closure captures (when compiling a closure body)
    current_captures: Vec<String>,
    /// Whether we're currently compiling a closure body
    compiling_closure: bool,
}

impl WasmCompiler {
    pub fn new() -> Self {
        WasmCompiler {
            type_cache: HashMap::new(),
            func_map: HashMap::new(),
            hir_funcs: Vec::new(),
            string_data: Vec::new(),
            string_offsets: HashMap::new(),
            struct_layouts: HashMap::new(),
            local_map: HashMap::new(),
            local_count: 0,
            param_count: 0,
            if_depths: Vec::new(),
            closure_info: Vec::new(),
            closure_expr_map: HashMap::new(),
            closure_captures: Vec::new(),
            closure_vars: HashMap::new(),
            current_captures: Vec::new(),
            compiling_closure: false,
        }
    }

    pub fn compile(&mut self, program: &HirProgram) -> TenthResult<Vec<u8>> {
        self.hir_funcs = program.functions.clone();
        self.build_struct_layouts(program);
        self.collect_strings(program);
        self.collect_closures(program);

        let mut module = Module::new();

        self.emit_type_section(&mut module, program)?;
        self.emit_import_section(&mut module);
        let _fc = self.emit_function_section(&mut module, program);
        self.emit_table_section(&mut module);
        self.emit_memory_section(&mut module);
        self.emit_global_section(&mut module);
        self.emit_export_section(&mut module, program);
        self.emit_elem_section(&mut module);
        self.emit_code_section(&mut module, program)?;
        if !self.string_data.is_empty() {
            self.emit_data_section(&mut module);
        }

        Ok(module.finish())
    }
}
