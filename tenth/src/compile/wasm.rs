//! HIR -> WebAssembly bytecode compiler.
//!
//! Generates a WASM module from a Tenth HIR program using `wasm-encoder`.
//! The module is designed to be executed by `wasmi` (embedded in the Rust host).

use std::collections::HashMap;
use wasm_encoder::{
    BlockType, CodeSection, ConstExpr, DataSection, EntityType, ExportKind,
    ExportSection, Elements, ElementSection, Function, FunctionSection, ImportSection, Instruction,
    MemorySection, MemoryType, Module, RefType, TableSection, TableType, TypeSection, ValType,
};
use crate::error::{TenthError, TenthResult};
use crate::hir::hir::*;
use crate::hir::types::{BaseType, Type};

// ── Type mapping ───────────────────────────────────────────────────────────

fn to_val_type(ty: &Type) -> Option<ValType> {
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

fn to_val_type_required(ty: &Type) -> TenthResult<ValType> {
    to_val_type(ty).ok_or_else(|| TenthError::RuntimeError {
        message: format!("无法将类型 {:?} 映射到 WASM 值类型", ty),
    })
}

/// Compute byte size and WASM type for a struct field. All fields are 8 bytes.
fn field_size_and_type(ty: &Type) -> (u32, ValType) {
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
const IMPORT_COUNT: u32 = 18; // 0-11 original + str_len(12) + str_at(13) + str_cmp(14) + f64_bits(15) + str_slice(16) + tensor_from_vec(17)

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

    // ── Struct layout ───────────────────────────────────────────────────

    fn build_struct_layouts(&mut self, program: &HirProgram) {
        for (sname, fields) in &program.structs {
            let mut offset = 0u32;
            let mut layout = HashMap::new();
            for (fname, fty) in fields {
                let (size, vt) = field_size_and_type(fty);
                layout.insert(fname.clone(), (offset, size, vt));
                offset += size;
            }
            self.struct_layouts.insert(sname.clone(), layout);
        }
        // Also build layouts for enum variants (keyed as "EnumName::VariantName")
        for (ename, variants) in &program.enums {
            for (vname, vfields) in variants {
                let mut offset = 0u32;
                let mut layout = HashMap::new();
                for (fname, fty) in vfields {
                    let (size, vt) = field_size_and_type(fty);
                    layout.insert(fname.clone(), (offset, size, vt));
                    offset += size;
                }
                let key = format!("{}::{}", ename, vname);
                self.struct_layouts.insert(key, layout);
            }
        }
    }

    fn struct_size(&self, name: &str) -> u32 {
        self.struct_layouts.get(name).map_or(0, |layout| {
            layout.values().map(|(off, sz, _)| off + sz).max().unwrap_or(0)
        })
    }

    fn infer_struct_name(&self, ty: &Type) -> String {
        match ty {
            Type::Struct(name) => name.clone(),
            Type::TypeParam { name } if self.struct_layouts.contains_key(name) => name.clone(),
            Type::Ref(inner) | Type::MutRef(inner) => self.infer_struct_name(inner),
            _ => String::new(),
        }
    }

    /// Find a struct that contains the given field. Returns (struct_name, offset, size, vt).
    fn resolve_field(&self, struct_hint: &str, field: &str) -> TenthResult<(String, u32, u32, ValType)> {
        // First try the hinted struct
        if !struct_hint.is_empty() {
            if let Some(layout) = self.struct_layouts.get(struct_hint) {
                if let Some(&info) = layout.get(field) {
                    return Ok((struct_hint.to_string(), info.0, info.1, info.2));
                }
            }
        }
        // Search all structs
        for (sname, layout) in &self.struct_layouts {
            if let Some(&info) = layout.get(field) {
                return Ok((sname.clone(), info.0, info.1, info.2));
            }
        }
        Err(TenthError::RuntimeError {
            message: format!("WASM: 没有结构体包含字段 '{}'", field),
        })
    }

    // ── Section builders ────────────────────────────────────────────────

    fn emit_type_section(&mut self, module: &mut Module, program: &HirProgram) -> TenthResult<()> {
        let mut types = TypeSection::new();
        let mut next = 0u32;
        let mut reg = |p: Vec<ValType>, r: Vec<ValType>| -> u32 {
            let k = (p.clone(), r.clone());
            *self.type_cache.entry(k).or_insert_with(|| {
                types.function(p, r);
                let n = next; next += 1; n
            })
        };
        // Host imports
        reg(vec![ValType::I32], vec![]);                                    // 0: println
        reg(vec![ValType::I32, ValType::I32], vec![]);                      // 1: write_file
        reg(vec![ValType::I32], vec![ValType::I32]);                        // 2: read_file
        reg(vec![ValType::I32, ValType::I32], vec![ValType::I32]);          // 3: str_add
        reg(vec![ValType::I32, ValType::I32], vec![ValType::I32]);          // 4: str_eq
        reg(vec![ValType::I64], vec![ValType::I32]);                        // 5: str_int
        reg(vec![ValType::I32], vec![ValType::I32]);                        // 6: tenth_alloc
        reg(vec![], vec![ValType::I64]);                                    // 7: Vec_new
        reg(vec![ValType::I64, ValType::I64], vec![ValType::I64]);          // 8: Vec_push
        reg(vec![ValType::I64], vec![ValType::I64]);                        // 9: Vec_len
        reg(vec![ValType::I64, ValType::I64], vec![ValType::I64]);          // 10: Vec_get
        reg(vec![ValType::I32, ValType::I32], vec![ValType::I32]);          // 11: compile_host
        reg(vec![ValType::I32], vec![ValType::I32]);                        // 12: str_len
        reg(vec![ValType::I32, ValType::I64], vec![ValType::I32]);          // 13: str_at
        reg(vec![ValType::I32, ValType::I32, ValType::I32], vec![ValType::I32]); // 14: str_cmp(op, a, b) -> bool
        reg(vec![ValType::F64], vec![ValType::I64]);                            // 15: f64_bits(f64) -> i64
        reg(vec![ValType::I32, ValType::I64, ValType::I64], vec![ValType::I32]); // 16: str_slice(ptr, start, end) -> ptr
        reg(vec![ValType::I32, ValType::I32, ValType::I32], vec![ValType::I64]); // 17: tensor_from_vec(data_ptr, len, rank) -> tensor_handle
        for func in &program.functions {
            let p: Vec<ValType> = func.params.iter().filter_map(|(_, t)| to_val_type(t)).collect();
            let r: Vec<ValType> = to_val_type(&func.return_type).into_iter().collect();
            reg(p, r);
        }
        // main
        reg(vec![], vec![ValType::I32]);
        // Closure types (D5): (i64 env_ptr, i64 param1, ..., i64 paramN) -> i64
        let param_counts: Vec<u32> = self.closure_info.iter().map(|&(_, _, pc)| pc).collect();
        let mut closure_type_idxs: Vec<u32> = Vec::new();
        for &param_count in &param_counts {
            let mut params: Vec<ValType> = vec![ValType::I64]; // env_ptr
            for _ in 0..param_count { params.push(ValType::I64); }
            let ti = reg(params, vec![ValType::I64]);
            closure_type_idxs.push(ti);
        }
        for (cidx, ti) in closure_type_idxs.into_iter().enumerate() {
            self.closure_info[cidx].1 = ti;
        }
        module.section(&types);
        Ok(())
    }

    fn emit_import_section(&mut self, module: &mut Module) {
        let mut imports = ImportSection::new();
        let ti = |p: Vec<ValType>, r: Vec<ValType>| -> u32 {
            *self.type_cache.get(&(p, r)).unwrap_or(&0)
        };
        imports.import("host", "println", EntityType::Function(ti(vec![ValType::I32], vec![])));
        imports.import("host", "write_file", EntityType::Function(ti(vec![ValType::I32, ValType::I32], vec![])));
        imports.import("host", "read_file", EntityType::Function(ti(vec![ValType::I32], vec![ValType::I32])));
        imports.import("host", "str_add", EntityType::Function(ti(vec![ValType::I32, ValType::I32], vec![ValType::I32])));
        imports.import("host", "str_eq", EntityType::Function(ti(vec![ValType::I32, ValType::I32], vec![ValType::I32])));
        imports.import("host", "str_int", EntityType::Function(ti(vec![ValType::I64], vec![ValType::I32])));
        imports.import("host", "tenth_alloc", EntityType::Function(ti(vec![ValType::I32], vec![ValType::I32])));
        imports.import("host", "Vec_new", EntityType::Function(ti(vec![], vec![ValType::I64])));
        imports.import("host", "Vec_push", EntityType::Function(ti(vec![ValType::I64, ValType::I64], vec![ValType::I64])));
        imports.import("host", "Vec_len", EntityType::Function(ti(vec![ValType::I64], vec![ValType::I64])));
        imports.import("host", "Vec_get", EntityType::Function(ti(vec![ValType::I64, ValType::I64], vec![ValType::I64])));
        imports.import("host", "compile_host", EntityType::Function(ti(vec![ValType::I32, ValType::I32], vec![ValType::I32])));
        imports.import("host", "str_len", EntityType::Function(ti(vec![ValType::I32], vec![ValType::I32])));
        imports.import("host", "str_at", EntityType::Function(ti(vec![ValType::I32, ValType::I64], vec![ValType::I32])));
        imports.import("host", "str_cmp", EntityType::Function(ti(vec![ValType::I32, ValType::I32, ValType::I32], vec![ValType::I32])));
        imports.import("host", "f64_bits", EntityType::Function(ti(vec![ValType::F64], vec![ValType::I64])));
        imports.import("host", "str_slice", EntityType::Function(ti(vec![ValType::I32, ValType::I64, ValType::I64], vec![ValType::I32])));
        imports.import("host", "tensor_from_vec", EntityType::Function(ti(vec![ValType::I32, ValType::I32, ValType::I32], vec![ValType::I64])));
        module.section(&imports);
    }

    fn emit_function_section(&mut self, module: &mut Module, program: &HirProgram) -> u32 {
        let mut funcs = FunctionSection::new();
        let mut idx = IMPORT_COUNT;
        for func in &program.functions {
            self.func_map.insert(func.name.clone(), idx);
            let p: Vec<ValType> = func.params.iter().filter_map(|(_, t)| to_val_type(t)).collect();
            let r: Vec<ValType> = to_val_type(&func.return_type).into_iter().collect();
            let ti = *self.type_cache.get(&(p, r)).unwrap_or(&0);
            funcs.function(ti);
            idx += 1;
        }
        // main
        let mti = *self.type_cache.get(&(vec![], vec![ValType::I32])).unwrap_or(&0);
        funcs.function(mti);
        idx += 1;
        // Closure functions (D5)
        for &(_, type_idx, _) in &self.closure_info {
            funcs.function(type_idx);
            idx += 1;
        }
        module.section(&funcs);
        idx
    }

    fn emit_memory_section(&self, module: &mut Module) {
        let mut mem = MemorySection::new();
        mem.memory(MemoryType { minimum: 16, maximum: Some(256), memory64: false, shared: false, page_size_log2: None });
        module.section(&mem);
    }

    /// D5.2: Emit table section (funcref table for call_indirect).
    /// Only emitted when there are closures.
    fn emit_table_section(&self, module: &mut Module) {
        let num_closures = self.closure_info.len() as u64;
        if num_closures == 0 { return; }
        let mut tables = TableSection::new();
        tables.table(TableType {
            element_type: RefType::FUNCREF,
            table64: false,
            minimum: num_closures,
            maximum: None,
            shared: false,
        });
        module.section(&tables);
    }

    /// Global section: no-op for Rust backend (bump pointer is host-managed).
    fn emit_global_section(&self, _module: &mut Module) {
        // The Rust host manages the bump allocator offset via store state (u32),
        // so no WASM global is needed.
    }

    /// D5.2: Emit element section to fill the table with closure function indices.
    /// Only emitted when there are closures.
    fn emit_elem_section(&self, module: &mut Module) {
        if self.closure_info.is_empty() { return; }
        let func_idxs: Vec<u32> = self.closure_info.iter().map(|&(fi, _, _)| fi).collect();
        let mut elements = ElementSection::new();
        elements.active(
            Some(0),
            &ConstExpr::i32_const(0),
            Elements::Functions(&func_idxs),
        );
        module.section(&elements);
    }

    fn emit_export_section(&mut self, module: &mut Module, program: &HirProgram) {
        let mut exports = ExportSection::new();
        for func in &program.functions {
            if let Some(&fi) = self.func_map.get(&func.name) {
                // Don't export user-defined "main" — our wrapper handles it
                if func.name != "main" {
                    exports.export(&func.name, ExportKind::Func, fi);
                }
            }
        }
        let mi = self.func_map.len() as u32 + IMPORT_COUNT;
        exports.export("main", ExportKind::Func, mi);
        exports.export("memory", ExportKind::Memory, 0);
        module.section(&exports);
    }

    fn emit_code_section(&mut self, module: &mut Module, program: &HirProgram) -> TenthResult<()> {
        let mut codes = CodeSection::new();
        for func in &program.functions {
            codes.function(&self.compile_function(func)?);
        }
        codes.function(&self.compile_main(program)?);
        // D5: Compile closure bodies (traverse HIR to find Closure nodes)
        self.compile_closure_bodies(&mut codes, program)?;
        module.section(&codes);
        Ok(())
    }

    fn emit_data_section(&self, module: &mut Module) {
        let mut data = DataSection::new();
        data.active(0, &ConstExpr::i32_const(0), self.string_data.clone());
        module.section(&data);
    }

    // ── Function compilation ─────────────────────────────────────────────

    fn compile_function(&mut self, func: &HirFnDef) -> TenthResult<Function> {
        eprintln!("[WASM] compile {}", func.name);
        self.local_map.clear();
        self.local_count = 0;
        for (name, _) in &func.params {
            self.local_map.insert(name.clone(), self.local_count);
            self.local_count += 1;
        }
        self.param_count = self.local_count; // parameters use correct types
        let locals: Vec<ValType> = (0..256).map(|_| ValType::I64).collect();
        let mut body = Function::new_with_locals_types(locals);
        self.compile_expr(&mut body, &func.body)?;
        if matches!(&func.return_type, Type::Base(BaseType::Unit)) {
            body.instruction(&Instruction::Return);
        }
        body.instruction(&Instruction::End);
        Ok(body)
    }

    fn compile_main(&mut self, program: &HirProgram) -> TenthResult<Function> {
        self.local_map.clear();
        self.local_count = 0;
        let locals: Vec<ValType> = (0..256).map(|_| ValType::I64).collect();
        let mut body = Function::new_with_locals_types(locals);
        if let Some(ref expr) = program.main_expr {
            self.compile_expr(&mut body, expr)?;
            self.wrap_to_i32(&mut body, &expr.ty);
        } else if let Some(mf) = program.functions.iter().find(|f| f.name == "main") {
            let fi = self.resolve_func("main")?;
            body.instruction(&Instruction::Call(fi));
            if matches!(mf.return_type, Type::Base(BaseType::Unit)) {
                // main() returns void — nothing to drop, just push exit code
                body.instruction(&Instruction::I32Const(0));
            } else {
                self.wrap_to_i32(&mut body, &mf.return_type);
            }
        } else {
            body.instruction(&Instruction::I32Const(0));
        }
        body.instruction(&Instruction::End);
        Ok(body)
    }

    // ── Closure body compilation (D5) ───────────────────────────────────

    /// Traverse HIR and compile each closure body in order.
    /// Order must match collect_closures and emit_function_section.
    fn compile_closure_bodies(&mut self, codes: &mut CodeSection, program: &HirProgram) -> TenthResult<()> {
        for func in &program.functions {
            self.ccb_expr(codes, &func.body)?;
        }
        if let Some(ref e) = program.main_expr {
            self.ccb_expr(codes, e)?;
        }
        Ok(())
    }

    fn ccb_expr(&mut self, codes: &mut CodeSection, e: &HirExpr) -> TenthResult<()> {
        match &e.kind {
            HirExprKind::Closure { params, body, captures } => {
                let func = self.compile_closure_body(params, body, captures)?;
                codes.function(&func);
                // Recurse for nested closures
                self.ccb_expr(codes, body)?;
            }
            HirExprKind::Binary { left, right, .. } => {
                self.ccb_expr(codes, left)?;
                self.ccb_expr(codes, right)?;
            }
            HirExprKind::Unary { expr: inner, .. } => { self.ccb_expr(codes, inner)?; }
            HirExprKind::Call { func, args, .. } => {
                self.ccb_expr(codes, func)?;
                for a in args { self.ccb_expr(codes, a)?; }
            }
            HirExprKind::GenericCall { func, args, .. } => {
                self.ccb_expr(codes, func)?;
                for a in args { self.ccb_expr(codes, a)?; }
            }
            HirExprKind::MethodCall { receiver, args, .. } => {
                self.ccb_expr(codes, receiver)?;
                for a in args { self.ccb_expr(codes, a)?; }
            }
            HirExprKind::Block { stmts, final_expr } => {
                for s in stmts { self.ccb_stmt(codes, s)?; }
                if let Some(e) = final_expr { self.ccb_expr(codes, e)?; }
            }
            HirExprKind::If { cond, then_branch, else_branch, .. } => {
                self.ccb_expr(codes, cond)?;
                self.ccb_expr(codes, then_branch)?;
                if let Some(e) = else_branch { self.ccb_expr(codes, e)?; }
            }
            HirExprKind::Assign { value, .. } => { self.ccb_expr(codes, value)?; }
            HirExprKind::AssignOp { value, .. } => { self.ccb_expr(codes, value)?; }
            HirExprKind::StructLiteral { fields, .. } => {
                for (_, e) in fields { self.ccb_expr(codes, e)?; }
            }
            HirExprKind::EnumLiteral { fields, .. } => {
                for (_, e) in fields { self.ccb_expr(codes, e)?; }
            }
            HirExprKind::Field { target, .. } => { self.ccb_expr(codes, target)?; }
            HirExprKind::FieldAssign { target, value, .. } => {
                self.ccb_expr(codes, target)?;
                self.ccb_expr(codes, value)?;
            }
            HirExprKind::Index { target, indices } => {
                self.ccb_expr(codes, target)?;
                for idx in indices {
                    match idx {
                        Index::Single(e) => { self.ccb_expr(codes, e)?; }
                        Index::Range { start, end } => {
                            if let Some(s) = start { self.ccb_expr(codes, s)?; }
                            if let Some(e) = end { self.ccb_expr(codes, e)?; }
                        }
                        _ => {}
                    }
                }
            }
            HirExprKind::Ref(inner) | HirExprKind::MutRef(inner)
            | HirExprKind::Deref(inner) | HirExprKind::TryBlock(inner) => {
                self.ccb_expr(codes, inner)?;
            }
            HirExprKind::TensorLiteral { data, .. } => {
                for row in data { for e in row { self.ccb_expr(codes, e)?; } }
            }
            HirExprKind::ArrayLiteral { elements, .. } => {
                for e in elements { self.ccb_expr(codes, e)?; }
            }
            HirExprKind::Range { start, end, .. } => {
                if let Some(s) = start { self.ccb_expr(codes, s)?; }
                if let Some(e) = end { self.ccb_expr(codes, e)?; }
            }
            HirExprKind::Match { scrutinee, arms, .. } => {
                self.ccb_expr(codes, scrutinee)?;
                for arm in arms { self.ccb_expr(codes, &arm.body)?; }
            }
            _ => {}
        }
        Ok(())
    }

    fn ccb_stmt(&mut self, codes: &mut CodeSection, s: &HirStmt) -> TenthResult<()> {
        use crate::hir::hir::HirStmtKind;
        match &s.kind {
            HirStmtKind::Expr(e) => { self.ccb_expr(codes, e)?; }
            HirStmtKind::Let { init, .. } => { if let Some(e) = init { self.ccb_expr(codes, e)?; } }
            HirStmtKind::While { cond, body } => {
                self.ccb_expr(codes, cond)?;
                self.ccb_stmt(codes, body)?;
            }
            HirStmtKind::Loop { body } => { for s in body { self.ccb_stmt(codes, s)?; } }
            HirStmtKind::For { body, .. } => { self.ccb_stmt(codes, body)?; }
            HirStmtKind::Return(expr) => { if let Some(e) = expr { self.ccb_expr(codes, e)?; } }
            _ => {}
        }
        Ok(())
    }

    /// Compile a closure body into a WASM function.
    /// Param 0 = env_ptr (i64), params 1..N = closure params (i64).
    fn compile_closure_body(
        &mut self,
        params: &[(String, Type)],
        body: &HirExpr,
        captures: &[String],
    ) -> TenthResult<Function> {
        // Reset local state for closure body
        self.local_map.clear();
        self.local_count = 0;
        self.param_count = 0;
        self.if_depths.clear();
        // Param 0 = env_ptr (unnamed)
        self.param_count = 1;
        self.local_count = 1;
        // Register closure params (param 1..N)
        for (name, _) in params {
            self.local_map.insert(name.clone(), self.local_count);
            self.local_count += 1;
            self.param_count += 1;
        }
        // Set closure compilation state
        self.compiling_closure = true;
        self.current_captures = captures.to_vec();
        // All extra locals are i64
        let locals: Vec<ValType> = (0..256).map(|_| ValType::I64).collect();
        let mut func = Function::new_with_locals_types(locals);
        self.compile_expr(&mut func, body)?;
        if matches!(&body.ty, Type::Base(BaseType::Unit)) {
            func.instruction(&Instruction::Return);
        }
        func.instruction(&Instruction::End);
        // Reset closure compilation state
        self.compiling_closure = false;
        self.current_captures.clear();
        Ok(func)
    }

    /// Emit conversion from a value type to i32 (for main's exit code).
    fn wrap_to_i32(&self, body: &mut Function, ty: &Type) {
        match ty {
            Type::Base(b) => match b {
                BaseType::I8 | BaseType::I16 | BaseType::I32 | BaseType::I64 => {
                    body.instruction(&Instruction::I32WrapI64);
                }
                BaseType::Bool => {
                    // Already i32, no conversion needed
                }
                BaseType::F32 | BaseType::F64 => {
                    body.instruction(&Instruction::I32TruncF64S);
                }
                _ => {
                    body.instruction(&Instruction::Drop);
                    body.instruction(&Instruction::I32Const(0));
                }
            },
            _ => {
                // Struct pointers, etc. → i64 → i32
                body.instruction(&Instruction::I32WrapI64);
            }
        }
    }

    fn resolve_func(&self, name: &str) -> TenthResult<u32> {
        // User-defined functions take priority over host functions
        if let Some(&idx) = self.func_map.get(name) {
            return Ok(idx);
        }
        match name {
            "println" | "eprintln" => Ok(0),
            "write_file" => Ok(1),
            "read_file" => Ok(2),
            "str_add" => Ok(3),
            "str_eq" => Ok(4),
            "str_int" => Ok(5),
            "tenth_alloc" => Ok(6),
            "Vec::new" | "Vec_new" => Ok(7),
            "Vec::push" | "Vec_push" => Ok(8),
            "Vec::len" | "Vec_len" => Ok(9),
            "Vec::get" | "Vec_get" => Ok(10),
            "compile_host" => Ok(11),
            _ => Err(TenthError::RuntimeError {
                    message: format!("WASM: 未定义函数 '{}'", name),
                }),
        }
    }

    // ── Expression compilation ───────────────────────────────────────────

    fn compile_expr(&mut self, body: &mut Function, expr: &HirExpr) -> TenthResult<()> {
        use HirExprKind;
        match &expr.kind {
            HirExprKind::Literal(lit) => self.compile_literal(body, lit)?,

            HirExprKind::Var(name) => {
                if let Some(&idx) = self.local_map.get(name) {
                    body.instruction(&Instruction::LocalGet(idx));
                    // Extra locals (index >= param_count) are stored as i64.
                    // Convert back to the expression's actual type.
                    if idx >= self.param_count {
                        if matches!(&expr.ty, Type::Base(BaseType::F64 | BaseType::F32)) {
                            body.instruction(&Instruction::F64ReinterpretI64);
                        } else if matches!(&expr.ty, Type::Base(BaseType::Bool)) {
                            body.instruction(&Instruction::I32WrapI64);
                        }
                    }
                } else if self.compiling_closure {
                    // D5.5: Check if this is a captured variable
                    if let Some(ci) = self.current_captures.iter().position(|c| c == name) {
                        // Load from env_ptr (local 0) + ci * 8
                        body.instruction(&Instruction::LocalGet(0));
                        body.instruction(&Instruction::I32WrapI64);
                        body.instruction(&Instruction::I32Const(ci as i32 * 8));
                        body.instruction(&Instruction::I32Add);
                        let arg = wasm_encoder::MemArg { offset: 0, align: 3, memory_index: 0 };
                        body.instruction(&Instruction::I64Load(arg));
                        // Convert to actual type if needed
                        if matches!(&expr.ty, Type::Base(BaseType::F64 | BaseType::F32)) {
                            body.instruction(&Instruction::F64ReinterpretI64);
                        } else if matches!(&expr.ty, Type::Base(BaseType::Bool)) {
                            body.instruction(&Instruction::I32WrapI64);
                        }
                    } else if !["println","eprintln","write_file","read_file"].contains(&name.as_str()) {
                        return Err(TenthError::RuntimeError {
                            message: format!("WASM: 未定义变量 '{}'", name),
                        });
                    }
                } else if !["println","eprintln","write_file","read_file"].contains(&name.as_str()) {
                    return Err(TenthError::RuntimeError {
                        message: format!("WASM: 未定义变量 '{}'", name),
                    });
                }
            }

            HirExprKind::Binary { op, left, right, .. } => {
                // String operations: emit str_add, str_eq, or str_cmp host call
                let is_str_op = matches!(&left.ty, Type::Base(BaseType::Str));
                let is_str_add = is_str_op && matches!(op, BinOp::Add);
                let is_str_eq = is_str_op && matches!(op, BinOp::Eq | BinOp::NotEq);
                let is_str_cmp = is_str_op && matches!(op, BinOp::Lt | BinOp::Gt | BinOp::LtEq | BinOp::GtEq);
                if is_str_add {
                    self.compile_string_arg(body, left)?;
                    self.compile_string_arg(body, right)?;
                    body.instruction(&Instruction::Call(3)); // str_add(a, b) -> i32
                    body.instruction(&Instruction::I64ExtendI32U); // i32 -> i64 for local storage
                } else if is_str_eq {
                    self.compile_string_arg(body, left)?;
                    self.compile_string_arg(body, right)?;
                    body.instruction(&Instruction::Call(4)); // str_eq(a, b) -> i32
                    if matches!(op, BinOp::NotEq) {
                        body.instruction(&Instruction::I32Eqz); // negate
                    }
                } else if is_str_cmp {
                    let op_code: i32 = match op {
                        BinOp::Lt => 0,
                        BinOp::Gt => 1,
                        BinOp::LtEq => 2,
                        BinOp::GtEq => 3,
                        _ => 0,
                    };
                    body.instruction(&Instruction::I32Const(op_code));
                    self.compile_string_arg(body, left)?;
                    self.compile_string_arg(body, right)?;
                    body.instruction(&Instruction::Call(14)); // str_cmp(op, a, b) -> i32
                } else {
                    self.compile_expr(body, left)?;
                    // Convert i64 to f64 if the operation involves float
                    let left_is_int = matches!(&left.ty, Type::Base(BaseType::I8 | BaseType::I16 | BaseType::I32 | BaseType::I64));
                    let right_is_float = matches!(&right.ty, Type::Base(BaseType::F32 | BaseType::F64));
                    if left_is_int && right_is_float {
                        body.instruction(&Instruction::F64ConvertI64S);
                    }
                    self.compile_expr(body, right)?;
                    let left_is_float = matches!(&left.ty, Type::Base(BaseType::F32 | BaseType::F64));
                    let right_is_int = matches!(&right.ty, Type::Base(BaseType::I8 | BaseType::I16 | BaseType::I32 | BaseType::I64));
                    if right_is_int && left_is_float {
                        body.instruction(&Instruction::F64ConvertI64S);
                    }
                    self.compile_binop(body, op, &left.ty, &right.ty)?;
                }
            }

            HirExprKind::Unary { op, expr: inner, .. } => {
                self.compile_expr(body, inner)?;
                self.compile_unary(body, op, &inner.ty)?;
            }

            HirExprKind::Call { func, args, .. } => {
                let fname = match &func.kind {
                    HirExprKind::Var(n) => n.clone(),
                    _ => return Err(TenthError::RuntimeError {
                        message: "WASM: 不支持间接调用".into(),
                    }),
                };
                match fname.as_str() {
                    "println" | "eprintln" => {
                        if let Some(a) = args.first() {
                            self.compile_string_arg(body, a)?;
                            body.instruction(&Instruction::Call(0));
                        }
                    }
                    "write_file" => {
                        if args.len() >= 2 {
                            self.compile_string_arg(body, &args[0])?;
                            self.compile_string_arg(body, &args[1])?;
                            body.instruction(&Instruction::Call(1));
                        }
                    }
                    "read_file" => {
                        if let Some(a) = args.first() {
                            self.compile_string_arg(body, a)?;
                            body.instruction(&Instruction::Call(2));
                            body.instruction(&Instruction::I64ExtendI32U); // i32 ptr -> i64
                        }
                    }
                    "compile_host" => {
                        if args.len() >= 2 {
                            self.compile_string_arg(body, &args[0])?;
                            self.compile_string_arg(body, &args[1])?;
                            body.instruction(&Instruction::Call(11));
                            body.instruction(&Instruction::I64ExtendI32U); // i32 -> i64
                        }
                    }
                    "Vec::new" | "Vec_new" => {
                        body.instruction(&Instruction::Call(7)); // Vec_new() -> i64
                    }
                    "f64_bits" => {
                        // f64_bits(f64) -> i64: reinterpret f64 bit pattern as i64
                        if let Some(a) = args.first() {
                            self.compile_expr(body, a)?;
                        }
                        body.instruction(&Instruction::Call(15));
                    }
                    "str_len" => {
                        // str_len(i32 ptr) -> i32: string length
                        if let Some(a) = args.first() {
                            self.compile_expr(body, a)?;
                            body.instruction(&Instruction::I32WrapI64); // i64 ptr -> i32
                        }
                        body.instruction(&Instruction::Call(12));
                        body.instruction(&Instruction::I64ExtendI32U); // i32 -> i64
                    }
                    "str_at" => {
                        // str_at(i32 ptr, i64 idx) -> i32: char at index
                        if args.len() >= 2 {
                            self.compile_expr(body, &args[0])?;
                            body.instruction(&Instruction::I32WrapI64); // i64 ptr -> i32
                            self.compile_expr(body, &args[1])?;
                        }
                        body.instruction(&Instruction::Call(13));
                        body.instruction(&Instruction::I64ExtendI32U); // i32 -> i64
                    }
                    "str_cmp" => {
                        // str_cmp(i32 op, i32 a, i32 b) -> i32: compare strings
                        if args.len() >= 3 {
                            self.compile_expr(body, &args[0])?;
                            body.instruction(&Instruction::I32WrapI64);
                            self.compile_expr(body, &args[1])?;
                            body.instruction(&Instruction::I32WrapI64);
                            self.compile_expr(body, &args[2])?;
                            body.instruction(&Instruction::I32WrapI64);
                        }
                        body.instruction(&Instruction::Call(14));
                        body.instruction(&Instruction::I64ExtendI32U); // i32 -> i64
                    }
                    "str_slice" => {
                        // str_slice(i32 ptr, i64 start, i64 end) -> i32: substring
                        if args.len() >= 3 {
                            self.compile_expr(body, &args[0])?;
                            body.instruction(&Instruction::I32WrapI64); // i64 ptr -> i32
                            self.compile_expr(body, &args[1])?;
                            self.compile_expr(body, &args[2])?;
                        }
                        body.instruction(&Instruction::Call(16));
                        body.instruction(&Instruction::I64ExtendI32U); // i32 -> i64
                    }
                    _ => {
                        // D5.4: Closure call detection — if fname is a closure variable,
                        // use call_indirect instead of regular call
                        if let Some(&type_idx) = self.closure_vars.get(&fname) {
                            if let Some(&cv_local) = self.local_map.get(&fname) {
                                // Unpack closure value (i64): high 32 = fn_ptr, low 32 = env_ptr
                                // 1. fn_ptr = cv >> 32, store in temp
                                body.instruction(&Instruction::LocalGet(cv_local));
                                body.instruction(&Instruction::I64Const(32));
                                body.instruction(&Instruction::I64ShrU);
                                let tmp = self.local_count;
                                self.local_count += 1;
                                body.instruction(&Instruction::LocalSet(tmp));
                                // 2. Push env_ptr = cv & 0xFFFFFFFF
                                body.instruction(&Instruction::LocalGet(cv_local));
                                body.instruction(&Instruction::I64Const(0xFFFFFFFF));
                                body.instruction(&Instruction::I64And);
                                // 3. Push args
                                for a in args { self.compile_expr(body, a)?; }
                                // 4. Push fn_ptr (from temp), wrap to i32, call_indirect
                                body.instruction(&Instruction::LocalGet(tmp));
                                body.instruction(&Instruction::I32WrapI64);
                                body.instruction(&Instruction::CallIndirect {
                                    type_index: type_idx,
                                    table_index: 0,
                                });
                                return Ok(());
                            }
                        }
                        for a in args { self.compile_expr(body, a)?; }
                        body.instruction(&Instruction::Call(self.resolve_func(&fname)?));
                    }
                }
            }

            HirExprKind::Block { stmts, final_expr } => {
                for s in stmts { self.compile_stmt(body, s)?; }
                if let Some(e) = final_expr { self.compile_expr(body, e)?; }
            }

            HirExprKind::If { cond, then_branch, else_branch, .. } => {
                self.compile_expr(body, cond)?;
                // WASM requires BOTH branches for Result type.
                // If-without-else must use Empty (value handled by caller).
                let has_value = else_branch.is_some()
                    && !matches!(&then_branch.ty, Type::Base(BaseType::Unit))
                    && !matches!(&else_branch.as_ref().unwrap().ty, Type::Base(BaseType::Unit));
                if has_value {
                    body.instruction(&Instruction::If(BlockType::Result(to_val_type_required(&then_branch.ty)?)));
                } else {
                    body.instruction(&Instruction::If(BlockType::Empty));
                }
                // Track If depth inside loops so Break/Continue emit correct Br depth.
                let in_loop = !self.if_depths.is_empty();
                if in_loop { self.if_depths.last_mut().unwrap().0 += 1; }
                self.compile_expr(body, then_branch)?;
                // If the If block is Empty but then_branch produces a value, drop it.
                if !has_value && !matches!(&then_branch.ty, Type::Base(BaseType::Unit)) {
                    body.instruction(&Instruction::Drop);
                }
                if let Some(eb) = else_branch {
                    body.instruction(&Instruction::Else);
                    self.compile_expr(body, eb)?;
                    if !has_value && !matches!(&eb.ty, Type::Base(BaseType::Unit)) {
                        body.instruction(&Instruction::Drop);
                    }
                }
                body.instruction(&Instruction::End);
                if in_loop { self.if_depths.last_mut().unwrap().0 -= 1; }
            }

            HirExprKind::Assign { target, value } => {
                self.compile_expr(body, value)?;
                // f64/bool values must be stored as i64 (all locals are i64)
                if matches!(&value.ty, Type::Base(BaseType::F64 | BaseType::F32)) {
                    body.instruction(&Instruction::I64ReinterpretF64);
                } else if matches!(&value.ty, Type::Base(BaseType::Bool)) {
                    body.instruction(&Instruction::I64ExtendI32U);
                }
                if let Some(&idx) = self.local_map.get(target) {
                    body.instruction(&Instruction::LocalSet(idx));
                } else {
                    self.local_map.insert(target.clone(), self.local_count);
                    body.instruction(&Instruction::LocalSet(self.local_count));
                    self.local_count += 1;
                }
            }

            HirExprKind::AssignOp { target, op, value } => {
                let idx = if let Some(&idx) = self.local_map.get(target) {
                    idx
                } else {
                    self.local_map.insert(target.clone(), self.local_count);
                    let idx = self.local_count;
                    self.local_count += 1;
                    idx
                };
                let is_float = matches!(&value.ty, Type::Base(BaseType::F32 | BaseType::F64));
                // Load current value, convert to f64 if needed
                body.instruction(&Instruction::LocalGet(idx));
                if is_float {
                    body.instruction(&Instruction::F64ConvertI64S);
                }
                // Compile RHS
                self.compile_expr(body, value)?;
                // Apply binary op
                self.compile_binop(body, op, &value.ty, &value.ty)?;
                // Convert result back to i64 for local storage
                if is_float {
                    body.instruction(&Instruction::I64ReinterpretF64);
                } else if matches!(&value.ty, Type::Base(BaseType::Bool)) {
                    body.instruction(&Instruction::I64ExtendI32U);
                }
                body.instruction(&Instruction::LocalSet(idx));
            }

            HirExprKind::EnumLiteral { enum_name, variant, fields } => {
                // Enum variants are stored like structs — allocate and write fields.
                // Layout keyed as "EnumName::VariantName".
                let layout_key = format!("{}::{}", enum_name, variant);
                let sz = self.struct_size(&layout_key);
                body.instruction(&Instruction::I32Const(sz as i32));
                body.instruction(&Instruction::Call(6)); // tenth_alloc -> i32
                body.instruction(&Instruction::I64ExtendI32U);
                // Save in a freshly-allocated temp local so nested exprs don't clobber it
                let tmp = self.local_count;
                self.local_count += 1;
                body.instruction(&Instruction::LocalSet(tmp));
                let layout = self.struct_layouts.get(&layout_key).cloned()
                    .ok_or_else(|| TenthError::RuntimeError {
                        message: format!("WASM: 未知的枚举变体 '{}/{}'", enum_name, variant),
                    })?;
                for (fname, fexpr) in fields {
                    if let Some(&(offset, _size, vt)) = layout.get(fname) {
                        body.instruction(&Instruction::LocalGet(tmp));
                        body.instruction(&Instruction::I32WrapI64);
                        self.compile_expr(body, fexpr)?;
                        let arg = wasm_encoder::MemArg { offset: offset as u64, align: 0, memory_index: 0 };
                        match vt {
                            ValType::I64 => { body.instruction(&Instruction::I64Store(arg)); }
                            ValType::I32 => { body.instruction(&Instruction::I32Store(arg)); }
                            ValType::F64 => { body.instruction(&Instruction::F64Store(arg)); }
                            _ => {}
                        }
                    }
                }
                body.instruction(&Instruction::LocalGet(tmp));
            }

            HirExprKind::StructLiteral { name, fields, has_default: _ } => {
                let sz = self.struct_size(name);
                body.instruction(&Instruction::I32Const(sz as i32));
                body.instruction(&Instruction::Call(6)); // tenth_alloc -> i32
                body.instruction(&Instruction::I64ExtendI32U); // i32 -> i64
                // Save in a freshly-allocated temp local so nested exprs don't clobber it
                let tmp = self.local_count;
                self.local_count += 1;
                body.instruction(&Instruction::LocalSet(tmp));
                let layout = self.struct_layouts.get(name).cloned()
                    .ok_or_else(|| TenthError::RuntimeError {
                        message: format!("WASM: 未知结构体 '{}'", name),
                    })?;
                for (fname, fexpr) in fields {
                    if let Some(&(offset, _size, vt)) = layout.get(fname) {
                        body.instruction(&Instruction::LocalGet(tmp));
                        body.instruction(&Instruction::I32WrapI64);
                        self.compile_expr(body, fexpr)?;
                        let arg = wasm_encoder::MemArg { offset: offset as u64, align: 0, memory_index: 0 };
                        match vt {
                            ValType::I64 => { body.instruction(&Instruction::I64Store(arg)); }
                            ValType::I32 => { body.instruction(&Instruction::I32Store(arg)); }
                            ValType::F64 => { body.instruction(&Instruction::F64Store(arg)); }
                            _ => {}
                        }
                    }
                }
                // Push i64 pointer as result
                body.instruction(&Instruction::LocalGet(tmp));
            }

            HirExprKind::Field { target, field } => {
                self.compile_expr(body, target)?;
                body.instruction(&Instruction::I32WrapI64); // pointer i64 -> i32
                let hint = self.infer_struct_name(&target.ty);
                let (_, offset, _, vt) = self.resolve_field(&hint, field)?;
                let arg = wasm_encoder::MemArg { offset: offset as u64, align: 0, memory_index: 0 };
                match vt {
                    ValType::I64 => { body.instruction(&Instruction::I64Load(arg)); }
                    ValType::I32 => { body.instruction(&Instruction::I32Load(arg)); }
                    ValType::F64 => { body.instruction(&Instruction::F64Load(arg)); }
                    _ => {}
                }
            }

            HirExprKind::FieldAssign { target, field, value } => {
                self.compile_expr(body, target)?;
                body.instruction(&Instruction::I32WrapI64);
                let hint = self.infer_struct_name(&target.ty);
                let (_, offset, _, vt) = self.resolve_field(&hint, field)?;
                self.compile_expr(body, value)?;
                // Convert i64→f64 if assigning to f64 field
                if matches!(vt, ValType::F64) && matches!(&value.ty, Type::Base(BaseType::I8 | BaseType::I16 | BaseType::I32 | BaseType::I64)) {
                    body.instruction(&Instruction::F64ConvertI64S);
                }
                let arg = wasm_encoder::MemArg { offset: offset as u64, align: 0, memory_index: 0 };
                match vt {
                    ValType::I64 => { body.instruction(&Instruction::I64Store(arg)); }
                    ValType::I32 => { body.instruction(&Instruction::I32Store(arg)); }
                    ValType::F64 => { body.instruction(&Instruction::F64Store(arg)); }
                    _ => {}
                }
            }

            HirExprKind::MethodCall { receiver, method, args, .. } => {
                self.compile_expr(body, receiver)?;
                // Receiver is i64; for string methods we need to convert to i32 (ptr)
                match method.as_str() {
                    "len" => {
                        // Check if receiver is String type
                        let is_string = matches!(&receiver.ty, Type::Base(BaseType::Str));
                        if is_string {
                            body.instruction(&Instruction::I32WrapI64); // i64 -> i32 pointer
                            body.instruction(&Instruction::Call(12));    // str_len(i32) -> i32
                            body.instruction(&Instruction::I64ExtendI32U); // i32 -> i64
                        } else {
                            body.instruction(&Instruction::Call(9)); // Vec_len(i64) -> i64
                        }
                    }
                    "push" => {
                        if let Some(a) = args.first() {
                            self.compile_expr(body, a)?;
                        } else {
                            body.instruction(&Instruction::I64Const(0));
                        }
                        body.instruction(&Instruction::Call(8)); // Vec_push -> i64
                        body.instruction(&Instruction::Drop);     // push returns Unit
                    }
                    "get" => {
                        if let Some(a) = args.first() {
                            self.compile_expr(body, a)?;
                        } else {
                            body.instruction(&Instruction::I64Const(0));
                        }
                        body.instruction(&Instruction::Call(10)); // Vec_get(i64, i64) -> i64
                    }
                    _ => return Err(TenthError::RuntimeError {
                        message: format!("WASM: 不支持的方法 '{}'", method),
                    }),
                }
            }

            // Ref/MutRef/Deref are identity ops for struct pointers (stored as i64)
            HirExprKind::Ref(inner)
            | HirExprKind::MutRef(inner)
            | HirExprKind::Deref(inner)
            | HirExprKind::TryBlock(inner) => {
                self.compile_expr(body, inner)?;
            }

            HirExprKind::InterpolatedString { parts } => {
                // Evaluate by concatenating all parts as strings via str_add host call
                let mut first = true;
                for p in parts {
                    match p {
                        crate::hir::hir::InterpPart::Literal(s) => {
                            body.instruction(&Instruction::I32Const(self.intern_string(s) as i32));
                        }
                        crate::hir::hir::InterpPart::Expr(name) => {
                            // Look up variable and convert to string
                            if let Some(&idx) = self.local_map.get(name) {
                                body.instruction(&Instruction::LocalGet(idx));
                                // Convert to i32 string pointer via str_int
                                body.instruction(&Instruction::Call(5)); // str_int(i64) -> i32
                            }
                        }
                    }
                    if first {
                        first = false;
                    } else {
                        // str_add: pop two i32 string ptrs, push concatenated i32 ptr
                        body.instruction(&Instruction::Call(3)); // str_add
                    }
                }
                // Result is i32 string pointer; extend to i64 for local storage
                body.instruction(&Instruction::I64ExtendI32U);
            }
            HirExprKind::Tuple(_elems) => {
                // Tuple not yet supported in WASM backend
                body.instruction(&Instruction::I64Const(0));
            }

            HirExprKind::Index { target, indices } => {
                // Distinguish String indexing (str_at) from Vec indexing (vec_get)
                // based on the target's type. Unknown types default to vec_get
                // since Vec indexing is far more common in tenthc.
                let is_string = matches!(&target.ty, Type::Base(BaseType::Str));
                self.compile_expr(body, target)?;
                if is_string {
                    // String pointer is stored as i64; str_at expects i32
                    body.instruction(&Instruction::I32WrapI64);
                }
                // For Vec: target stays as i64 (vec ptr) for vec_get
                match indices.first() {
                    Some(Index::Single(idx)) => {
                        self.compile_expr(body, idx)?; // compile index expression (i64)
                        if is_string {
                            body.instruction(&Instruction::Call(13)); // str_at(i32, i64) -> i32
                            body.instruction(&Instruction::I64ExtendI32U); // i32 -> i64
                        } else {
                            body.instruction(&Instruction::Call(10)); // vec_get(i64, i64) -> i64
                        }
                    }
                    Some(Index::Range { start, end }) => {
                        // String slice: s[start..end] -> str_slice(ptr, start, end) -> ptr
                        if !is_string {
                            return Err(TenthError::RuntimeError {
                                message: "WASM: Vec range slicing not yet supported".to_string(),
                            });
                        }
                        if let Some(s) = start {
                            self.compile_expr(body, s)?;
                        } else {
                            body.instruction(&Instruction::I64Const(0));
                        }
                        if let Some(e) = end {
                            self.compile_expr(body, e)?;
                        } else {
                            // No end: use string length — push a sentinel and let host handle it
                            // For now, push a large value; the host will clamp to strlen
                            body.instruction(&Instruction::I64Const(i64::MAX));
                        }
                        body.instruction(&Instruction::Call(16)); // str_slice(i32, i64, i64) -> i32
                        body.instruction(&Instruction::I64ExtendI32U); // i32 -> i64
                    }
                    _ => {
                        body.instruction(&Instruction::I64Const(0));
                        if is_string {
                            body.instruction(&Instruction::Call(13)); // str_at fallback
                            body.instruction(&Instruction::I64ExtendI32U);
                        } else {
                            body.instruction(&Instruction::Call(10)); // vec_get fallback
                        }
                    }
                }
            }

            HirExprKind::TensorLiteral { data, .. } => {
                // Flatten 2D data into elements, allocate memory, write f64 values,
                // then call tensor_from_vec(data_ptr, len, rank) host import.
                let rows = data.len() as i32;
                let total: i32 = data.iter().map(|r| r.len() as i32).sum();
                let size = total * 8; // each f64 is 8 bytes

                // Allocate memory: tenth_alloc(size) -> i32 ptr -> i64
                body.instruction(&Instruction::I32Const(size));
                body.instruction(&Instruction::Call(6)); // tenth_alloc
                body.instruction(&Instruction::I64ExtendI32U); // i32 -> i64
                let tmp = self.local_count;
                self.local_count += 1;
                body.instruction(&Instruction::LocalSet(tmp));

                // Write each element as f64 at offset idx*8
                let mut idx: i32 = 0;
                for row in data {
                    for elem in row {
                        body.instruction(&Instruction::LocalGet(tmp));
                        body.instruction(&Instruction::I32WrapI64); // ptr i64 -> i32
                        self.compile_expr(body, elem)?;
                        // Convert i64 to f64 if element is integer-typed
                        if matches!(&elem.ty, Type::Base(BaseType::I8 | BaseType::I16 | BaseType::I32 | BaseType::I64)) {
                            body.instruction(&Instruction::F64ConvertI64S);
                        }
                        let arg = wasm_encoder::MemArg { offset: (idx as u64) * 8, align: 0, memory_index: 0 };
                        body.instruction(&Instruction::F64Store(arg));
                        idx += 1;
                    }
                }

                // Call tensor_from_vec(data_ptr, len, rank) — import 17
                body.instruction(&Instruction::LocalGet(tmp));
                body.instruction(&Instruction::I32WrapI64); // data_ptr
                body.instruction(&Instruction::I32Const(total)); // len
                body.instruction(&Instruction::I32Const(rows)); // rank (rows)
                body.instruction(&Instruction::Call(17)); // tensor_from_vec -> i64
            }

            // D5.3/D5.6: Closure — compile as packed i64 (table_idx << 32 | env_ptr)
            HirExprKind::Closure { captures, .. } => {
                let ptr = expr as *const HirExpr as usize;
                let cidx = *self.closure_expr_map.get(&ptr).ok_or_else(|| TenthError::RuntimeError {
                    message: "WASM: 闭包未注册".into(),
                })?;
                let (_func_idx, _type_idx, _pc) = self.closure_info[cidx];
                // Use closure index (table position) as fn_ptr, NOT func_idx
                let table_idx = cidx as i64;
                if captures.is_empty() {
                    // No captures: env_ptr = 0, packed = table_idx << 32
                    body.instruction(&Instruction::I64Const(table_idx << 32));
                } else {
                    // Allocate env struct via tenth_alloc (import 6)
                    // size = captures_count * 8
                    let env_size = captures.len() as i32 * 8;
                    body.instruction(&Instruction::I32Const(env_size));
                    body.instruction(&Instruction::Call(6)); // tenth_alloc -> i32
                    body.instruction(&Instruction::I64ExtendI32U); // i32 -> i64 env_ptr
                    // Store env_ptr in temp local
                    let tmp = self.local_count;
                    self.local_count += 1;
                    body.instruction(&Instruction::LocalSet(tmp));
                    // Write each captured variable to env struct
                    for (ci, cap_name) in captures.iter().enumerate() {
                        body.instruction(&Instruction::LocalGet(tmp));
                        body.instruction(&Instruction::I32WrapI64);
                        // Load the captured variable's value
                        if let Some(&idx) = self.local_map.get(cap_name) {
                            body.instruction(&Instruction::LocalGet(idx));
                            // Convert f64/bool to i64 for storage
                            // (locals already store as i64, so no conversion needed)
                        } else {
                            // Captured variable not found — push 0
                            body.instruction(&Instruction::I64Const(0));
                        }
                        let arg = wasm_encoder::MemArg { offset: (ci as u64) * 8, align: 0, memory_index: 0 };
                        body.instruction(&Instruction::I64Store(arg));
                    }
                    // Push packed value: (table_idx << 32) | env_ptr
                    body.instruction(&Instruction::I64Const(table_idx << 32));
                    body.instruction(&Instruction::LocalGet(tmp));
                    body.instruction(&Instruction::I64Or);
                }
            }

            _ => return Err(TenthError::RuntimeError {
                message: format!("WASM: 不支持的表达式 {:?}", expr.kind),
            }),
        }
        Ok(())
    }

    fn compile_stmt(&mut self, body: &mut Function, stmt: &HirStmt) -> TenthResult<()> {
        use crate::hir::hir::HirStmtKind;
        match &stmt.kind {
            HirStmtKind::Expr(e) => {
                self.compile_expr(body, e)?;
                if !matches!(&e.ty, Type::Base(BaseType::Unit)) {
                    body.instruction(&Instruction::Drop);
                }
            }
            HirStmtKind::Let { names, type_ann, init, .. } => {
                if let Some(e) = init {
                    self.compile_expr(body, e)?;
                    let target_f = matches!(type_ann, Some(Type::Base(BaseType::F64 | BaseType::F32)));
                    let expr_f = matches!(&e.ty, Type::Base(BaseType::F64 | BaseType::F32));
                    let expr_bool = matches!(&e.ty, Type::Base(BaseType::Bool));
                    if target_f && !expr_f {
                        body.instruction(&Instruction::F64ConvertI64S);
                        body.instruction(&Instruction::I64ReinterpretF64);
                    } else if expr_f {
                        body.instruction(&Instruction::I64ReinterpretF64);
                    } else if expr_bool {
                        body.instruction(&Instruction::I64ExtendI32U);
                    }
                    // D5: If init is a Closure, register variable as closure var
                    if let HirExprKind::Closure { .. } = &e.kind {
                        let ptr = e as *const HirExpr as usize;
                        if let Some(&cidx) = self.closure_expr_map.get(&ptr) {
                            let (_, type_idx, _) = self.closure_info[cidx];
                            for name in names {
                                self.closure_vars.insert(name.clone(), type_idx);
                            }
                        }
                    }
                } else {
                    body.instruction(&Instruction::I64Const(0));
                }
                for name in names {
                    self.local_map.insert(name.clone(), self.local_count);
                    body.instruction(&Instruction::LocalSet(self.local_count));
                    self.local_count += 1;
                }
            }
            HirStmtKind::Loop { body: lb } => {
                self.if_depths.push((0, 0));
                body.instruction(&Instruction::Block(BlockType::Empty));
                body.instruction(&Instruction::Loop(BlockType::Empty));
                for s in lb { self.compile_stmt(body, s)?; }
                body.instruction(&Instruction::Br(0));
                body.instruction(&Instruction::End);
                body.instruction(&Instruction::End);
                self.if_depths.pop();
            }
            HirStmtKind::While { cond, body: lb } => {
                self.if_depths.push((0, 0));
                body.instruction(&Instruction::Block(BlockType::Empty));
                body.instruction(&Instruction::Loop(BlockType::Empty));
                self.compile_expr(body, cond)?;
                body.instruction(&Instruction::I32Eqz);
                body.instruction(&Instruction::BrIf(1));
                self.compile_stmt(body, lb.as_ref())?;
                body.instruction(&Instruction::Br(0));
                body.instruction(&Instruction::End);
                body.instruction(&Instruction::End);
                self.if_depths.pop();
            }
            HirStmtKind::Return(expr) => {
                if let Some(e) = expr {
                    self.compile_expr(body, e)?;
                }
                body.instruction(&Instruction::Return);
            }
            HirStmtKind::Break => {
                let &(if_depth, break_offset) = self.if_depths.last().unwrap_or(&(0, 0));
                let depth = 1 + break_offset + if_depth;
                body.instruction(&Instruction::Br(depth));
            }
            HirStmtKind::Continue => {
                let &(if_depth, _) = self.if_depths.last().unwrap_or(&(0, 0));
                let depth = if_depth;
                body.instruction(&Instruction::Br(depth));
            }
            HirStmtKind::For { var, iter, body: lb } => {
                // Only Range iterators are supported (for x in start..end { ... })
                match &iter.kind {
                    HirExprKind::Range { start, end, inclusive } => {
                        // Allocate local for the loop variable
                        let var_local = self.local_count;
                        self.local_map.insert(var.clone(), var_local);
                        self.local_count += 1;

                        // var = start (default 0)
                        if let Some(s) = start {
                            self.compile_expr(body, s)?;
                        } else {
                            body.instruction(&Instruction::I64Const(0));
                        }
                        body.instruction(&Instruction::LocalSet(var_local));

                        self.if_depths.push((0, 1)); // break_offset=1 for for's inner body block
                        // block (break target) + loop (continue/back-edge target)
                        body.instruction(&Instruction::Block(BlockType::Empty));
                        body.instruction(&Instruction::Loop(BlockType::Empty));

                        // condition: var < end (or var <= end if inclusive)
                        body.instruction(&Instruction::LocalGet(var_local));
                        if let Some(e) = end {
                            self.compile_expr(body, e)?;
                        } else {
                            body.instruction(&Instruction::I64Const(i64::MAX));
                        }
                        if *inclusive {
                            body.instruction(&Instruction::I64LeS);
                        } else {
                            body.instruction(&Instruction::I64LtS);
                        }
                        body.instruction(&Instruction::I32Eqz);
                        body.instruction(&Instruction::BrIf(1)); // exit block when condition false

                        // Wrap body in inner block so `continue` (br if_depth=0) breaks
                        // the inner body block and falls through to var += 1, avoiding
                        // infinite loop where continue skips the increment.
                        body.instruction(&Instruction::Block(BlockType::Empty));

                        // loop body
                        self.compile_stmt(body, lb.as_ref())?;

                        body.instruction(&Instruction::End); // end inner body block

                        // var += 1
                        body.instruction(&Instruction::LocalGet(var_local));
                        body.instruction(&Instruction::I64Const(1));
                        body.instruction(&Instruction::I64Add);
                        body.instruction(&Instruction::LocalSet(var_local));

                        body.instruction(&Instruction::Br(0)); // back-edge to loop header
                        body.instruction(&Instruction::End); // loop
                        body.instruction(&Instruction::End); // block
                        self.if_depths.pop();
                    }
                    _ => return Err(TenthError::RuntimeError {
                        message: format!("WASM: For 循环仅支持 Range 迭代器, got {:?}", iter.kind),
                    }),
                }
            }
            _ => return Err(TenthError::RuntimeError {
                message: format!("WASM: 不支持的语句 {:?}", stmt.kind),
            }),
        }
        Ok(())
    }

    // ── Literals & ops ────────────────────────────────────────────────────

    fn compile_literal(&mut self, body: &mut Function, lit: &Literal) -> TenthResult<()> {
        match lit {
            Literal::Int(n) => { body.instruction(&Instruction::I64Const(*n)); }
            // 策略 A：f32→f64 提升。F32 字面量在 WASM 后端统一当 f64 处理，
            // 与下游所有 F64 处理（compile_binop/compile_unary/Var/Assign 等）保持一致，
            // 避免 F32Const 压入 f32 栈后下游期望 f64 导致 WASM 验证失败。
            // dtype 信息在 HIR 层已保留，仅 WASM 执行路径做精度提升。
            Literal::Float(n, _dt) => { body.instruction(&Instruction::F64Const(*n)); }
            Literal::Bool(b) => { body.instruction(&Instruction::I32Const(if *b { 1 } else { 0 })); }
            Literal::String(s) => {
                let off = self.intern_string(s);
                body.instruction(&Instruction::I32Const(off as i32));
                body.instruction(&Instruction::I64ExtendI32U); // strings are stored as i64 in locals
            }
        }
        Ok(())
    }

    fn compile_binop(&self, body: &mut Function, op: &BinOp, lty: &Type, rty: &Type) -> TenthResult<()> {
        let is_f = matches!(lty, Type::Base(BaseType::F32 | BaseType::F64))
            || matches!(rty, Type::Base(BaseType::F32 | BaseType::F64));
        match op {
            BinOp::Add => self.emit_if(body, is_f, Instruction::F64Add, Instruction::I64Add),
            BinOp::Sub => self.emit_if(body, is_f, Instruction::F64Sub, Instruction::I64Sub),
            BinOp::Mul => self.emit_if(body, is_f, Instruction::F64Mul, Instruction::I64Mul),
            BinOp::Div => {
                self.emit_if(body, is_f, Instruction::F64Div, Instruction::I64DivS);
            }
            BinOp::Mod => { body.instruction(&Instruction::I64RemS); }
            BinOp::Eq => self.emit_if(body, is_f, Instruction::F64Eq, Instruction::I64Eq),
            BinOp::NotEq => self.emit_if(body, is_f, Instruction::F64Ne, Instruction::I64Ne),
            BinOp::Lt => self.emit_if(body, is_f, Instruction::F64Lt, Instruction::I64LtS),
            BinOp::Gt => self.emit_if(body, is_f, Instruction::F64Gt, Instruction::I64GtS),
            BinOp::LtEq => self.emit_if(body, is_f, Instruction::F64Le, Instruction::I64LeS),
            BinOp::GtEq => self.emit_if(body, is_f, Instruction::F64Ge, Instruction::I64GeS),
            BinOp::And => { body.instruction(&Instruction::I32And); }
            BinOp::Or => { body.instruction(&Instruction::I32Or); }
        }
        Ok(())
    }

    fn emit_if(&self, body: &mut Function, cond: bool, then_i: Instruction<'_>, else_i: Instruction<'_>) {
        if cond { body.instruction(&then_i); } else { body.instruction(&else_i); }
    }

    fn compile_unary(&self, body: &mut Function, op: &UnaryOp, ty: &Type) -> TenthResult<()> {
        let is_f = matches!(ty, Type::Base(BaseType::F32 | BaseType::F64));
        match op {
            UnaryOp::Neg => if is_f {
                body.instruction(&Instruction::F64Neg);
            } else {
                body.instruction(&Instruction::LocalSet(self.local_count));
                body.instruction(&Instruction::I64Const(0));
                body.instruction(&Instruction::LocalGet(self.local_count));
                body.instruction(&Instruction::I64Sub);
            },
            UnaryOp::Not => { body.instruction(&Instruction::I32Eqz); }
            UnaryOp::Try => { /* Try not supported in WASM backend */ }
        }
        Ok(())
    }

    fn compile_string_arg(&mut self, body: &mut Function, expr: &HirExpr) -> TenthResult<()> {
        match &expr.kind {
            HirExprKind::Literal(Literal::String(s)) => {
                body.instruction(&Instruction::I32Const(self.intern_string(s) as i32));
            }
            HirExprKind::Literal(Literal::Int(n)) => {
                body.instruction(&Instruction::I64Const(*n));
                body.instruction(&Instruction::Call(5));
            }
            HirExprKind::Literal(Literal::Float(n, _)) => {
                body.instruction(&Instruction::F64Const(*n));
                body.instruction(&Instruction::I64TruncF64S);
                body.instruction(&Instruction::Call(5));
            }
            HirExprKind::Binary { op: BinOp::Add, left, right, .. } => {
                self.compile_string_arg(body, left)?;
                self.compile_string_arg(body, right)?;
                body.instruction(&Instruction::Call(3));
            }
            _ => {
                self.compile_expr(body, expr)?;
                // If value is a string (stored as i64 in locals), wrap to i32
                if matches!(&expr.ty, Type::Base(BaseType::Str)) {
                    body.instruction(&Instruction::I32WrapI64);
                } else if matches!(&expr.ty, Type::Base(BaseType::I64)) {
                    body.instruction(&Instruction::Call(5)); // str_int
                }
            }
        }
        Ok(())
    }

    // ── String interning ─────────────────────────────────────────────────

    fn intern_string(&mut self, s: &str) -> u32 {
        if let Some(&off) = self.string_offsets.get(s) { return off; }
        let off = self.string_data.len() as u32;
        self.string_data.extend_from_slice(s.as_bytes());
        self.string_data.push(0);
        self.string_offsets.insert(s.to_string(), off);
        off
    }

    fn collect_strings(&mut self, p: &HirProgram) {
        // Pre-intern all single ASCII characters so str_at can return
        // pointers to pre-allocated strings without heap allocation.
        for byte in 1u8..128u8 {
            if let Some(c) = char::from_u32(byte as u32) {
                let s = c.to_string();
                self.intern_string(&s);
            }
        }
        for f in &p.functions { self.cs_expr(&f.body); }
        if let Some(ref e) = p.main_expr { self.cs_expr(e); }
    }

    fn cs_expr(&mut self, e: &HirExpr) {
        use HirExprKind;
        match &e.kind {
            HirExprKind::Literal(Literal::String(s)) => { self.intern_string(s); }
            HirExprKind::Binary { left, right, .. } => { self.cs_expr(left); self.cs_expr(right); }
            HirExprKind::Unary { expr: inner, .. } => self.cs_expr(inner),
            HirExprKind::Call { args, .. } => { for a in args { self.cs_expr(a); } }
            HirExprKind::Block { stmts, final_expr } => {
                for s in stmts { self.cs_stmt(s); }
                if let Some(e) = final_expr { self.cs_expr(e); }
            }
            HirExprKind::If { cond, then_branch, else_branch, .. } => {
                self.cs_expr(cond); self.cs_expr(then_branch);
                if let Some(e) = else_branch { self.cs_expr(e); }
            }
            HirExprKind::Assign { value, .. } => self.cs_expr(value),
            HirExprKind::AssignOp { value, .. } => self.cs_expr(value),
            HirExprKind::StructLiteral { fields, .. } => {
                for (_, e) in fields { self.cs_expr(e); }
            }
            HirExprKind::EnumLiteral { fields, .. } => {
                for (_, e) in fields { self.cs_expr(e); }
            }
            HirExprKind::MethodCall { receiver, args, .. } => {
                self.cs_expr(receiver);
                for a in args { self.cs_expr(a); }
            }
            HirExprKind::Index { target, indices } => {
                self.cs_expr(target);
                for idx in indices {
                    match idx {
                        Index::Single(e) => self.cs_expr(e),
                        Index::Range { start, end } => {
                            if let Some(s) = start { self.cs_expr(s); }
                            if let Some(e) = end { self.cs_expr(e); }
                        }
                        _ => {}
                    }
                }
            }
            _ => {}
        }
    }

    fn cs_stmt(&mut self, s: &HirStmt) {
        use crate::hir::hir::HirStmtKind;
        match &s.kind {
            HirStmtKind::Expr(e) => self.cs_expr(e),
            HirStmtKind::Let { init, .. } => { if let Some(e) = init { self.cs_expr(e); } }
            HirStmtKind::While { cond, body } => { self.cs_expr(cond); self.cs_stmt(body); }
            HirStmtKind::Loop { body } => { for s in body { self.cs_stmt(s); } }
            HirStmtKind::For { body, .. } => { self.cs_stmt(body); }
            HirStmtKind::Return(expr) => { if let Some(e) = expr { self.cs_expr(e); } }
            _ => {}
        }
    }

    // ── Closure collection (D5) ─────────────────────────────────────────

    /// Traverse HIR and register all Closure nodes. Assigns func_idx and
    /// stores captures. type_idx is filled later during emit_type_section.
    fn collect_closures(&mut self, program: &HirProgram) {
        let num_user_funcs = program.functions.len() as u32;
        for func in &program.functions {
            self.cc_expr(&func.body, num_user_funcs);
        }
        if let Some(ref e) = program.main_expr {
            self.cc_expr(e, num_user_funcs);
        }
    }

    fn cc_expr(&mut self, e: &HirExpr, num_user_funcs: u32) {
        match &e.kind {
            HirExprKind::Closure { params, body, captures } => {
                let cidx = self.closure_info.len() as u32;
                // func_idx = IMPORT_COUNT + num_user_funcs + 1 (main) + cidx
                let func_idx = IMPORT_COUNT + num_user_funcs + 1 + cidx;
                let param_count = params.len() as u32;
                self.closure_info.push((func_idx, 0, param_count));
                self.closure_captures.push(captures.clone());
                let ptr = e as *const HirExpr as usize;
                self.closure_expr_map.insert(ptr, cidx as usize);
                // Recurse for nested closures
                self.cc_expr(body, num_user_funcs);
            }
            HirExprKind::Binary { left, right, .. } => {
                self.cc_expr(left, num_user_funcs);
                self.cc_expr(right, num_user_funcs);
            }
            HirExprKind::Unary { expr: inner, .. } => {
                self.cc_expr(inner, num_user_funcs);
            }
            HirExprKind::Call { func, args, .. } => {
                self.cc_expr(func, num_user_funcs);
                for a in args { self.cc_expr(a, num_user_funcs); }
            }
            HirExprKind::GenericCall { func, args, .. } => {
                self.cc_expr(func, num_user_funcs);
                for a in args { self.cc_expr(a, num_user_funcs); }
            }
            HirExprKind::MethodCall { receiver, args, .. } => {
                self.cc_expr(receiver, num_user_funcs);
                for a in args { self.cc_expr(a, num_user_funcs); }
            }
            HirExprKind::Block { stmts, final_expr } => {
                for s in stmts { self.cc_stmt(s, num_user_funcs); }
                if let Some(e) = final_expr { self.cc_expr(e, num_user_funcs); }
            }
            HirExprKind::If { cond, then_branch, else_branch, .. } => {
                self.cc_expr(cond, num_user_funcs);
                self.cc_expr(then_branch, num_user_funcs);
                if let Some(e) = else_branch { self.cc_expr(e, num_user_funcs); }
            }
            HirExprKind::Assign { value, .. } => { self.cc_expr(value, num_user_funcs); }
            HirExprKind::AssignOp { value, .. } => { self.cc_expr(value, num_user_funcs); }
            HirExprKind::StructLiteral { fields, .. } => {
                for (_, e) in fields { self.cc_expr(e, num_user_funcs); }
            }
            HirExprKind::EnumLiteral { fields, .. } => {
                for (_, e) in fields { self.cc_expr(e, num_user_funcs); }
            }
            HirExprKind::Field { target, .. } => { self.cc_expr(target, num_user_funcs); }
            HirExprKind::FieldAssign { target, value, .. } => {
                self.cc_expr(target, num_user_funcs);
                self.cc_expr(value, num_user_funcs);
            }
            HirExprKind::Index { target, indices } => {
                self.cc_expr(target, num_user_funcs);
                for idx in indices {
                    match idx {
                        Index::Single(e) => self.cc_expr(e, num_user_funcs),
                        Index::Range { start, end } => {
                            if let Some(s) = start { self.cc_expr(s, num_user_funcs); }
                            if let Some(e) = end { self.cc_expr(e, num_user_funcs); }
                        }
                        _ => {}
                    }
                }
            }
            HirExprKind::Ref(inner) | HirExprKind::MutRef(inner)
            | HirExprKind::Deref(inner) | HirExprKind::TryBlock(inner) => {
                self.cc_expr(inner, num_user_funcs);
            }
            HirExprKind::TensorLiteral { data, .. } => {
                for row in data { for e in row { self.cc_expr(e, num_user_funcs); } }
            }
            HirExprKind::ArrayLiteral { elements, .. } => {
                for e in elements { self.cc_expr(e, num_user_funcs); }
            }
            HirExprKind::Range { start, end, .. } => {
                if let Some(s) = start { self.cc_expr(s, num_user_funcs); }
                if let Some(e) = end { self.cc_expr(e, num_user_funcs); }
            }
            HirExprKind::Match { scrutinee, arms, .. } => {
                self.cc_expr(scrutinee, num_user_funcs);
                for arm in arms { self.cc_expr(&arm.body, num_user_funcs); }
            }
            _ => {}
        }
    }

    fn cc_stmt(&mut self, s: &HirStmt, num_user_funcs: u32) {
        use crate::hir::hir::HirStmtKind;
        match &s.kind {
            HirStmtKind::Expr(e) => self.cc_expr(e, num_user_funcs),
            HirStmtKind::Let { init, .. } => { if let Some(e) = init { self.cc_expr(e, num_user_funcs); } }
            HirStmtKind::While { cond, body } => {
                self.cc_expr(cond, num_user_funcs);
                self.cc_stmt(body, num_user_funcs);
            }
            HirStmtKind::Loop { body } => { for s in body { self.cc_stmt(s, num_user_funcs); } }
            HirStmtKind::For { body, .. } => { self.cc_stmt(body, num_user_funcs); }
            HirStmtKind::Return(expr) => { if let Some(e) = expr { self.cc_expr(e, num_user_funcs); } }
            _ => {}
        }
    }
}

// ── wasmi runtime ──────────────────────────────────────────────────────────

use wasmi::{Engine, Store, Linker, Caller};

/// Register all host imports (module "host") on the given linker.
/// The store state must be a `u32` representing the bump-allocator offset.
pub fn register_host_functions(linker: &mut Linker<u32>) -> TenthResult<()> {
    linker.func_wrap("host", "println", |caller: Caller<'_, u32>, ptr: i32| {
        let mem = caller.get_export("memory").and_then(|e| e.into_memory()).unwrap();
        let data = mem.data(&caller);
        let end = data[ptr as usize..].iter().position(|&b| b == 0).unwrap_or(0);
        println!("{}", std::str::from_utf8(&data[ptr as usize..ptr as usize + end]).unwrap_or(""));
    }).map_err(|e| TenthError::RuntimeError { message: format!("链接器：{}", e) })?;

    linker.func_wrap("host", "write_file",
        |caller: Caller<'_, u32>, path_ptr: i32, content_ptr: i32| {
            let mem = caller.get_export("memory").and_then(|e| e.into_memory()).unwrap();
            let data = mem.data(&caller);
            let rs = |p: i32| -> &str {
                let end = data[p as usize..].iter().position(|&b| b == 0).unwrap_or(0);
                std::str::from_utf8(&data[p as usize..p as usize + end]).unwrap_or("")
            };
            let _ = std::fs::write(rs(path_ptr), rs(content_ptr));
    }).map_err(|e| TenthError::RuntimeError { message: format!("链接器：{}", e) })?;

    // read_file(path: i32) -> i32
    linker.func_wrap("host", "read_file",
        |mut caller: Caller<'_, u32>, path_ptr: i32| -> i32 {
            let mem = caller.get_export("memory").and_then(|e| e.into_memory()).unwrap();
            let data = mem.data(&caller);
            let end = data[path_ptr as usize..].iter().position(|&b| b == 0).unwrap_or(0);
            let path = std::str::from_utf8(&data[path_ptr as usize..path_ptr as usize + end]).unwrap_or("");
            match std::fs::read_to_string(path) {
                Ok(content) => {
                    let bump = *caller.data();
                    let bytes = content.as_bytes();
                    let needed = bytes.len() + 1;
                    *caller.data_mut() = bump + needed as u32;
                    let dest = mem.data_mut(&mut caller);
                    let off = bump as usize;
                    if off + needed <= dest.len() {
                        dest[off..off + bytes.len()].copy_from_slice(bytes);
                        dest[off + bytes.len()] = 0;
                        bump as i32
                    } else { 0i32 }
                }
                Err(_) => 0i32,
            }
    }).map_err(|e| TenthError::RuntimeError { message: format!("链接器：{}", e) })?;

    linker.func_wrap("host", "str_add",
        |mut caller: Caller<'_, u32>, a_ptr: i32, b_ptr: i32| -> i32 {
            let mem = caller.get_export("memory").and_then(|e| e.into_memory()).unwrap();
            let data = mem.data(&caller);
            let rs = |p: i32| -> &str {
                let end = data[p as usize..].iter().position(|&b| b == 0).unwrap_or(0);
                std::str::from_utf8(&data[p as usize..p as usize + end]).unwrap_or("")
            };
            let result = format!("{}{}", rs(a_ptr), rs(b_ptr));
            let bytes = result.as_bytes();
            let np = *caller.data();
            let needed = np as usize + bytes.len() + 1;
            let current_len = mem.data(&caller).len();
            if needed > current_len {
                let pages = ((needed - current_len + 65535) / 65536) as u32;
                mem.grow(&mut caller, pages).ok();
            }
            *caller.data_mut() = np + bytes.len() as u32 + 1;
            let d = mem.data_mut(&mut caller);
            d[np as usize..np as usize + bytes.len()].copy_from_slice(bytes);
            d[np as usize + bytes.len()] = 0;
            np as i32
    }).map_err(|e| TenthError::RuntimeError { message: format!("链接器：{}", e) })?;

    linker.func_wrap("host", "str_eq",
        |caller: Caller<'_, u32>, a_ptr: i32, b_ptr: i32| -> i32 {
            let mem = caller.get_export("memory").and_then(|e| e.into_memory()).unwrap();
            let data = mem.data(&caller);
            let rs = |p: i32| -> &str {
                let end = data[p as usize..].iter().position(|&b| b == 0).unwrap_or(0);
                std::str::from_utf8(&data[p as usize..p as usize + end]).unwrap_or("")
            };
            if rs(a_ptr) == rs(b_ptr) { 1 } else { 0 }
    }).map_err(|e| TenthError::RuntimeError { message: format!("链接器：{}", e) })?;

    linker.func_wrap("host", "str_int",
        |mut caller: Caller<'_, u32>, n: i64| -> i32 {
            let s = n.to_string();
            let mem = caller.get_export("memory").and_then(|e| e.into_memory()).unwrap();
            let data = mem.data_mut(&mut caller);
            let off = 4096i32;
            let b = s.as_bytes();
            if off as usize + b.len() + 1 <= data.len() {
                data[off as usize..off as usize + b.len()].copy_from_slice(b);
                data[off as usize + b.len()] = 0;
                off
            } else { 0 }
    }).map_err(|e| TenthError::RuntimeError { message: format!("链接器：{}", e) })?;

    // tenth_alloc(size: i32) -> i32
    linker.func_wrap("host", "tenth_alloc",
        |mut caller: Caller<'_, u32>, size: i32| -> i32 {
            let ptr = *caller.data();
            let needed = ptr as usize + size as usize;
            let mem = caller.get_export("memory").and_then(|e| e.into_memory()).unwrap();
            let current_len = mem.data(&caller).len();
            // Grow memory if needed (each page is 64KiB)
            while needed > current_len {
                let pages_needed = (needed - current_len + 65535) / 65536;
                mem.grow(&mut caller, pages_needed as u32).ok();
                let new_len = mem.data(&caller).len();
                if new_len == current_len { break; } // couldn't grow
            }
            *caller.data_mut() = ptr + size as u32;
            ptr as i32
    }).map_err(|e| TenthError::RuntimeError { message: format!("链接器：{}", e) })?;

    // Vec_new() -> i64 (pointer extended to i64)
    linker.func_wrap("host", "Vec_new",
        |mut caller: Caller<'_, u32>| -> i64 {
            let ptr = *caller.data();
            // Zero-initialize the Vec header (cap=0, len=0, dp=0) so that
            // Vec_push correctly triggers the first allocation.
            let mem = caller.get_export("memory").and_then(|e| e.into_memory()).unwrap();
            let data = mem.data_mut(&mut caller);
            let p = ptr as usize;
            data[p..p+8].copy_from_slice(&0i64.to_le_bytes());       // cap
            data[p+8..p+16].copy_from_slice(&0i64.to_le_bytes());    // len
            data[p+16..p+20].copy_from_slice(&0i32.to_le_bytes());   // dp
            *caller.data_mut() = ptr + 24;
            ptr as i64
    }).map_err(|e| TenthError::RuntimeError { message: format!("链接器：{}", e) })?;

    // Vec_len(vec: i64) -> i64
    linker.func_wrap("host", "Vec_len",
        |caller: Caller<'_, u32>, vec: i64| -> i64 {
            let vec_ptr = vec as i32 as usize;
            let mem = caller.get_export("memory").and_then(|e| e.into_memory()).unwrap();
            let data = mem.data(&caller);
            if vec_ptr + 16 <= data.len() {
                i64::from_le_bytes(data[vec_ptr+8..vec_ptr+16].try_into().unwrap())
            } else { 0 }
    }).map_err(|e| TenthError::RuntimeError { message: format!("链接器：{}", e) })?;

    // Vec_get(vec: i64, idx: i64) -> i64
    linker.func_wrap("host", "Vec_get",
        |caller: Caller<'_, u32>, vec: i64, idx: i64| -> i64 {
            let vec_ptr = vec as i32 as usize;
            let mem = caller.get_export("memory").and_then(|e| e.into_memory()).unwrap();
            let data = mem.data(&caller);
            if vec_ptr + 20 > data.len() { return 0; }
            let dp = i32::from_le_bytes(data[vec_ptr+16..vec_ptr+20].try_into().unwrap()) as usize;
            let pos = dp + idx as usize * 8;
            if pos + 8 <= data.len() {
                i64::from_le_bytes(data[pos..pos+8].try_into().unwrap())
            } else { 0 }
    }).map_err(|e| TenthError::RuntimeError { message: format!("链接器：{}", e) })?;

    // Vec_push(vec: i64, item: i64) -> i64
    linker.func_wrap("host", "Vec_push",
        |mut caller: Caller<'_, u32>, vec: i64, item: i64| -> i64 {
            let vec_ptr = vec as i32 as usize;
            // Phase 1: read header
            let (cap, len, dp) = {
                let mem = caller.get_export("memory").and_then(|e| e.into_memory()).unwrap();
                let data = mem.data(&caller);
                let vp = vec_ptr;
                let cap = if vp+8 <= data.len() { i64::from_le_bytes(data[vp..vp+8].try_into().unwrap()) } else { 0 };
                let len = if vp+16 <= data.len() { i64::from_le_bytes(data[vp+8..vp+16].try_into().unwrap()) } else { 0 };
                let dp = if vp+20 <= data.len() { i32::from_le_bytes(data[vp+16..vp+20].try_into().unwrap()) } else { 0 };
                (cap, len, dp)
            };
            // Phase 2: allocate if needed
            let (new_cap, new_dp) = if len >= cap || dp == 0 {
                let nc = if cap == 0 { 4 } else { cap * 2 };
                let new_sz = nc as usize * 8;
                let np = *caller.data();
                *caller.data_mut() = np + new_sz as u32;
                // Copy old data from dp to new allocation (if any)
                if dp != 0 && len > 0 {
                    let mem = caller.get_export("memory").and_then(|e| e.into_memory()).unwrap();
                    let data = mem.data_mut(&mut caller);
                    let old_sz = len as usize * 8;
                    data.copy_within(dp as usize..dp as usize + old_sz, np as usize);
                }
                (nc, np as i32)
            } else {
                (cap, dp)
            };
            // Phase 3: write
            {
                let mem = caller.get_export("memory").and_then(|e| e.into_memory()).unwrap();
                let data = mem.data_mut(&mut caller);
                let vp = vec_ptr;
                data[vp..vp+8].copy_from_slice(&new_cap.to_le_bytes());
                data[vp+8..vp+16].copy_from_slice(&(len + 1).to_le_bytes());
                data[vp+16..vp+20].copy_from_slice(&new_dp.to_le_bytes());
                let pos = new_dp as usize + len as usize * 8;
                data[pos..pos+8].copy_from_slice(&item.to_le_bytes());
            }
            vec
    }).map_err(|e| TenthError::RuntimeError { message: format!("链接器：{}", e) })?;

    // compile_host(src: i64, out_path: i64) -> i32
    // Reads source from WASM memory, compiles it via Rust pipeline, writes .wasm.
    linker.func_wrap("host", "compile_host",
        |caller: Caller<'_, u32>, src_ptr: i32, out_ptr: i32| -> i32 {
            let mem = caller.get_export("memory").and_then(|e| e.into_memory()).unwrap();
            let data = mem.data(&caller);
            let read_str = |p: i32| -> String {
                let off = p as usize;
                let end = data[off..].iter().position(|&b| b == 0).unwrap_or(0);
                std::str::from_utf8(&data[off..off+end]).unwrap_or("").to_string()
            };
            let src = read_str(src_ptr);
            let out = read_str(out_ptr);
            // Compile via Rust pipeline
            match crate::lexer::lexer::Lexer::new(&src).tokenize()
                .and_then(|tokens| crate::parser::parser::Parser::new(tokens).parse_program())
                .and_then(|prog| crate::hir::lower::Lowerer::new().lower_program(&prog))
                .and_then(|hir| crate::compile::compile_to_wasm(&hir))
            {
                Ok(wasm_bytes) => {
                    let _ = std::fs::write(&out, &wasm_bytes);
                    0i32
                }
                Err(_) => 1i32,
            }
    }).map_err(|e| TenthError::RuntimeError { message: format!("链接器：{}", e) })?;

    // str_len(s: i32) -> i32  — returns length of null-terminated string
    linker.func_wrap("host", "str_len",
        |caller: Caller<'_, u32>, ptr: i32| -> i32 {
            let mem = caller.get_export("memory").and_then(|e| e.into_memory()).unwrap();
            let data = mem.data(&caller);
            let off = ptr as usize;
            data[off..].iter().position(|&b| b == 0).unwrap_or(0) as i32
    }).map_err(|e| TenthError::RuntimeError { message: format!("链接器：{}", e) })?;

    // str_at(s: i32, idx: i64) -> i32  — returns single-char string at index
    // Characters are pre-interned as "X\0" in the data section, so we
    // can return a direct pointer without heap allocation.
    linker.func_wrap("host", "str_at",
        |mut caller: Caller<'_, u32>, ptr: i32, idx: i64| -> i32 {
            let mem = caller.get_export("memory").and_then(|e| e.into_memory()).unwrap();
            let data = mem.data(&caller);
            let off = ptr as usize;
            let s = {
                let end = data[off..].iter().position(|&b| b == 0).unwrap_or(0);
                std::str::from_utf8(&data[off..off+end]).unwrap_or("")
            };
            let ch = s.chars().nth(idx as usize).unwrap_or('\0');
            // Pre-interned ASCII: characters 1..127 are stored as "X\0" 
            // at offset (ch-1)*2 in the data section.
            let cu = ch as u32;
            if cu >= 1 && cu < 128 {
                return ((cu - 1) * 2) as i32;
            }
            // Non-ASCII fallback: allocate from bump allocator (rare)
            let ch_str = ch.to_string();
            let ch_bytes = ch_str.as_bytes();
            let np = *caller.data();
            let needed = np as usize + ch_bytes.len() + 1;
            let current_len = mem.data(&caller).len();
            if needed > current_len {
                let pages = ((needed - current_len + 65535) / 65536) as u32;
                mem.grow(&mut caller, pages).ok();
            }
            *caller.data_mut() = np + ch_bytes.len() as u32 + 1;
            let d = mem.data_mut(&mut caller);
            d[np as usize..np as usize + ch_bytes.len()].copy_from_slice(ch_bytes);
            d[np as usize + ch_bytes.len()] = 0;
            np as i32
    }).map_err(|e| TenthError::RuntimeError { message: format!("链接器：{}", e) })?;

    // str_cmp(op: i32, a: i32, b: i32) -> i32  — op: 0=LT,1=GT,2=LE,3=GE; returns 0 or 1
    linker.func_wrap("host", "str_cmp",
        |caller: Caller<'_, u32>, op: i32, a: i32, b: i32| -> i32 {
            let mem = caller.get_export("memory").and_then(|e| e.into_memory()).unwrap();
            let data = mem.data(&caller);
            let read = |p: i32| -> String {
                let off = p as usize;
                let end = data[off..].iter().position(|&b| b == 0).unwrap_or(0);
                std::str::from_utf8(&data[off..off+end]).unwrap_or("").to_string()
            };
            let sa = read(a);
            let sb = read(b);
            let result = match op {
                0 => sa < sb,
                1 => sa > sb,
                2 => sa <= sb,
                3 => sa >= sb,
                _ => false,
            };
            result as i32
    }).map_err(|e| TenthError::RuntimeError { message: format!("链接器：{}", e) })?;

    // f64_bits(f64) -> i64: convert f64 to its IEEE 754 bit representation
    linker.func_wrap("host", "f64_bits",
        |val: f64| -> i64 {
            val.to_bits() as i64
    }).map_err(|e| TenthError::RuntimeError { message: format!("链接器：{}", e) })?;

    // str_slice(ptr: i32, start: i64, end: i64) -> i32: allocate new string s[start..end]
    linker.func_wrap("host", "str_slice",
        |mut caller: Caller<'_, u32>, ptr: i32, start: i64, end: i64| -> i32 {
            let mem = caller.get_export("memory").and_then(|e| e.into_memory()).unwrap();
            // Phase 1: read source slice into an owned Vec so the immutable
            // borrow of `caller` ends before any mutable operation below.
            let slice_bytes: Vec<u8> = {
                let data = mem.data(&caller);
                let off = ptr as usize;
                let slen = data[off..].iter().position(|&b| b == 0).unwrap_or(0);
                let s = start.max(0) as usize;
                let e = if end >= i64::MAX { slen } else { end.max(0) as usize };
                let s = s.min(slen);
                let e = e.min(slen).max(s);
                data[off + s..off + e].to_vec()
            };
            let slice_len = slice_bytes.len();
            // Phase 2: bump-allocate and write the slice.
            let np = *caller.data();
            let needed = np as usize + slice_len + 1;
            let current_len = mem.data(&caller).len();
            if needed > current_len {
                let pages = ((needed - current_len + 65535) / 65536) as u32;
                mem.grow(&mut caller, pages).ok();
            }
            *caller.data_mut() = np + slice_len as u32 + 1;
            let d = mem.data_mut(&mut caller);
            d[np as usize..np as usize + slice_len].copy_from_slice(&slice_bytes);
            d[np as usize + slice_len] = 0;
            np as i32
    }).map_err(|e| TenthError::RuntimeError { message: format!("链接器：{}", e) })?;

    // tensor_from_vec(data_ptr: i32, len: i32, rank: i32) -> i64
    // Simplified: return total element count (len) as the tensor handle.
    // This provides a deterministic value for parity testing.
    linker.func_wrap("host", "tensor_from_vec",
        |_caller: Caller<'_, u32>, _data_ptr: i32, len: i32, _rank: i32| -> i64 {
            len as i64
    }).map_err(|e| TenthError::RuntimeError { message: format!("链接器：{}", e) })?;

    Ok(())
}

/// Execute a WASM bytecode module in-process using wasmi.
pub fn run_wasm_module(wasm_bytes: &[u8]) -> TenthResult<()> {
    let engine = Engine::default();
    let module = wasmi::Module::new(&engine, wasm_bytes).map_err(|e| {
        TenthError::RuntimeError { message: format!("WASM 模块解析错误：{}", e) }
    })?;

    let mut store = Store::new(&engine, 8192u32);
    let mut linker = Linker::new(&engine);
    register_host_functions(&mut linker)?;

    let instance = linker.instantiate(&mut store, &module)
        .and_then(|pre| pre.start(&mut store))
        .map_err(|e| TenthError::RuntimeError {
            message: format!("WASM 实例化错误：{}", e),
        })?;

    let main_fn = instance.get_typed_func::<(), i32>(&store, "main")
        .map_err(|_| TenthError::RuntimeError {
            message: "WASM 模块没有导出的 'main' 函数".into(),
        })?;

    let exit_code = main_fn.call(&mut store, ())
        .map_err(|e| TenthError::RuntimeError {
            message: format!("WASM main() 错误：{}", e),
        })?;

    if exit_code != 0 {
        return Err(TenthError::RuntimeError {
            message: format!("WASM main() 以代码 {} 退出", exit_code),
        });
    }

    Ok(())
}
