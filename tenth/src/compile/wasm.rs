//! HIR -> WebAssembly bytecode compiler.
//!
//! Generates a WASM module from a Tenth HIR program using `wasm-encoder`.
//! The module is designed to be executed by `wasmi` (embedded in the Rust host).

use std::collections::HashMap;
use wasm_encoder::{
    BlockType, CodeSection, ConstExpr, DataSection, EntityType, ExportKind,
    ExportSection, Function, FunctionSection, ImportSection, Instruction,
    MemorySection, MemoryType, Module, TypeSection, ValType,
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
        Type::Unknown => Some(ValType::I64),
        _ => None,
    }
}

fn to_val_type_required(ty: &Type) -> TenthResult<ValType> {
    to_val_type(ty).ok_or_else(|| TenthError::RuntimeError {
        message: format!("cannot map type {:?} to WASM value type", ty),
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
const IMPORT_COUNT: u32 = 15; // 0-11 original + str_len(12) + str_at(13) + str_cmp(14)

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
        }
    }

    pub fn compile(&mut self, program: &HirProgram) -> TenthResult<Vec<u8>> {
        self.hir_funcs = program.functions.clone();
        self.build_struct_layouts(program);
        self.collect_strings(program);

        let mut module = Module::new();

        self.emit_type_section(&mut module, program)?;
        self.emit_import_section(&mut module);
        let _fc = self.emit_function_section(&mut module, program);
        self.emit_memory_section(&mut module);
        self.emit_export_section(&mut module, program);
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
            message: format!("WASM: no struct has field '{}'", field),
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
        for func in &program.functions {
            let p: Vec<ValType> = func.params.iter().filter_map(|(_, t)| to_val_type(t)).collect();
            let r: Vec<ValType> = to_val_type(&func.return_type).into_iter().collect();
            reg(p, r);
        }
        // main
        reg(vec![], vec![ValType::I32]);
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
        let mti = *self.type_cache.get(&(vec![], vec![ValType::I32])).unwrap_or(&0);
        funcs.function(mti);
        module.section(&funcs);
        idx
    }

    fn emit_memory_section(&self, module: &mut Module) {
        let mut mem = MemorySection::new();
        mem.memory(MemoryType { minimum: 16, maximum: Some(256), memory64: false, shared: false, page_size_log2: None });
        module.section(&mem);
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
        self.local_map.clear();
        self.local_count = 0;
        for (name, _) in &func.params {
            self.local_map.insert(name.clone(), self.local_count);
            self.local_count += 1;
        }
        self.param_count = self.local_count; // parameters use correct types
        let locals: Vec<ValType> = (0..64).map(|_| ValType::I64).collect();
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
        let locals: Vec<ValType> = (0..16).map(|_| ValType::I64).collect();
        let mut body = Function::new_with_locals_types(locals);
        if let Some(ref expr) = program.main_expr {
            self.compile_expr(&mut body, expr)?;
            self.wrap_to_i32(&mut body, &expr.ty);
        } else if let Some(mf) = program.functions.iter().find(|f| f.name == "main") {
            let fi = self.resolve_func("main")?;
            body.instruction(&Instruction::Call(fi));
            if matches!(mf.return_type, Type::Base(BaseType::Unit)) {
                body.instruction(&Instruction::Drop);
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
            _ => self.func_map.get(name).copied()
                .ok_or_else(|| TenthError::RuntimeError {
                    message: format!("WASM: undefined function '{}'", name),
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
                } else if !["println","eprintln","write_file","read_file"].contains(&name.as_str()) {
                    return Err(TenthError::RuntimeError {
                        message: format!("WASM: undefined variable '{}'", name),
                    });
                }
            }

            HirExprKind::Binary { op, left, right, .. } => {
                // String comparisons: emit str_cmp or str_eq host call
                let is_str_op = matches!(&left.ty, Type::Base(BaseType::Str));
                let is_str_eq = is_str_op && matches!(op, BinOp::Eq | BinOp::NotEq);
                let is_str_cmp = is_str_op && matches!(op, BinOp::Lt | BinOp::Gt | BinOp::LtEq | BinOp::GtEq);
                if is_str_eq {
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
                        message: "WASM: indirect calls not supported".into(),
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
                    _ => {
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
                // Detect if this if-expression produces a value.
                // If both branches end with a value (not just return), use Result type.
                let has_value = !matches!(&then_branch.ty, Type::Base(BaseType::Unit))
                    && (else_branch.is_none() || !matches!(&else_branch.as_ref().unwrap().ty, Type::Base(BaseType::Unit)));
                if has_value {
                    body.instruction(&Instruction::If(BlockType::Result(to_val_type_required(&then_branch.ty)?)));
                } else {
                    body.instruction(&Instruction::If(BlockType::Empty));
                }
                self.compile_expr(body, then_branch)?;
                if let Some(eb) = else_branch {
                    body.instruction(&Instruction::Else);
                    self.compile_expr(body, eb)?;
                }
                body.instruction(&Instruction::End);
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

            HirExprKind::EnumLiteral { enum_name, variant, fields } => {
                // Enum variants are stored like structs — allocate and write fields.
                // Layout keyed as "EnumName::VariantName".
                let layout_key = format!("{}::{}", enum_name, variant);
                let sz = self.struct_size(&layout_key);
                body.instruction(&Instruction::I32Const(sz as i32));
                body.instruction(&Instruction::Call(6)); // tenth_alloc -> i32
                body.instruction(&Instruction::I64ExtendI32U);
                let tmp = if self.local_count > 0 { self.local_count } else { 1 };
                body.instruction(&Instruction::LocalSet(tmp));
                let layout = self.struct_layouts.get(&layout_key).cloned()
                    .ok_or_else(|| TenthError::RuntimeError {
                        message: format!("WASM: unknown enum variant '{}/{}'", enum_name, variant),
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

            HirExprKind::StructLiteral { name, fields } => {
                let sz = self.struct_size(name);
                body.instruction(&Instruction::I32Const(sz as i32));
                body.instruction(&Instruction::Call(6)); // tenth_alloc -> i32
                body.instruction(&Instruction::I64ExtendI32U); // i32 -> i64
                // Save in temp, then push copy for result
                let tmp = if self.local_count > 0 { self.local_count } else { 1 };
                body.instruction(&Instruction::LocalSet(tmp));
                let layout = self.struct_layouts.get(name).cloned()
                    .ok_or_else(|| TenthError::RuntimeError {
                        message: format!("WASM: unknown struct '{}'", name),
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
                        message: format!("WASM: unsupported method '{}'", method),
                    }),
                }
            }

            // Ref/MutRef/Deref are identity ops for struct pointers (stored as i64)
            HirExprKind::Ref(inner)
            | HirExprKind::MutRef(inner)
            | HirExprKind::Deref(inner) => {
                self.compile_expr(body, inner)?;
            }

            HirExprKind::Index { target, indices } => {
                // String indexing: target is a string pointer (i32 after conversion)
                self.compile_expr(body, target)?;
                body.instruction(&Instruction::I32WrapI64); // pointer i64 -> i32
                if let Some(Index::Single(idx)) = indices.first() {
                    self.compile_expr(body, idx)?; // compile index expression (i64)
                } else {
                    body.instruction(&Instruction::I64Const(0));
                }
                body.instruction(&Instruction::Call(13)); // str_at(i32, i64) -> i32
                body.instruction(&Instruction::I64ExtendI32U); // i32 -> i64
            }

            _ => return Err(TenthError::RuntimeError {
                message: format!("WASM: unsupported expr {:?}", expr.kind),
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
            HirStmtKind::Let { name, type_ann, init, .. } => {
                if let Some(e) = init {
                    self.compile_expr(body, e)?;
                    // Convert/Reinterpret based on declared type vs expression type
                    let target_f = matches!(type_ann, Some(Type::Base(BaseType::F64 | BaseType::F32)));
                    let expr_f = matches!(&e.ty, Type::Base(BaseType::F64 | BaseType::F32));
                    let expr_bool = matches!(&e.ty, Type::Base(BaseType::Bool));
                    if target_f && !expr_f {
                        // i64 → f64 conversion (not reinterpret)
                        body.instruction(&Instruction::F64ConvertI64S);
                        body.instruction(&Instruction::I64ReinterpretF64);
                    } else if expr_f {
                        // f64 → store as i64 bits
                        body.instruction(&Instruction::I64ReinterpretF64);
                    } else if expr_bool {
                        // bool (i32) → i64 for local storage
                        body.instruction(&Instruction::I64ExtendI32U);
                    }
                } else {
                    body.instruction(&Instruction::I64Const(0));
                }
                self.local_map.insert(name.clone(), self.local_count);
                body.instruction(&Instruction::LocalSet(self.local_count));
                self.local_count += 1;
            }
            HirStmtKind::Loop { body: lb } => {
                body.instruction(&Instruction::Block(BlockType::Empty));
                body.instruction(&Instruction::Loop(BlockType::Empty));
                for s in lb { self.compile_stmt(body, s)?; }
                body.instruction(&Instruction::Br(0));
                body.instruction(&Instruction::End);
                body.instruction(&Instruction::End);
            }
            HirStmtKind::While { cond, body: lb } => {
                body.instruction(&Instruction::Block(BlockType::Empty));
                body.instruction(&Instruction::Loop(BlockType::Empty));
                self.compile_expr(body, cond)?;
                body.instruction(&Instruction::I32Eqz);
                body.instruction(&Instruction::BrIf(1));
                self.compile_stmt(body, lb.as_ref())?;
                body.instruction(&Instruction::Br(0));
                body.instruction(&Instruction::End);
                body.instruction(&Instruction::End);
            }
            HirStmtKind::Return(expr) => {
                if let Some(e) = expr {
                    self.compile_expr(body, e)?;
                }
                body.instruction(&Instruction::Return);
            }
            HirStmtKind::Break => { body.instruction(&Instruction::Br(1)); }
            HirStmtKind::Continue => { body.instruction(&Instruction::Br(0)); }
            _ => return Err(TenthError::RuntimeError {
                message: format!("WASM: unsupported stmt {:?}", stmt.kind),
            }),
        }
        Ok(())
    }

    // ── Literals & ops ────────────────────────────────────────────────────

    fn compile_literal(&mut self, body: &mut Function, lit: &Literal) -> TenthResult<()> {
        match lit {
            Literal::Int(n) => { body.instruction(&Instruction::I64Const(*n)); }
            Literal::Float(n) => { body.instruction(&Instruction::F64Const(*n)); }
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
                if !is_f {
                    body.instruction(&Instruction::F64ConvertI64S);
                    body.instruction(&Instruction::LocalSet(self.local_count));
                    body.instruction(&Instruction::F64ConvertI64S);
                    body.instruction(&Instruction::LocalGet(self.local_count));
                }
                body.instruction(&Instruction::F64Div);
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
                body.instruction(&Instruction::LocalTee(self.local_count));
                body.instruction(&Instruction::I64Const(0));
                body.instruction(&Instruction::LocalGet(self.local_count));
                body.instruction(&Instruction::I64Sub);
            },
            UnaryOp::Not => { body.instruction(&Instruction::I32Eqz); }
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
            HirExprKind::Literal(Literal::Float(n)) => {
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
}

// ── wasmi runtime ──────────────────────────────────────────────────────────

use wasmi::{Engine, Store, Linker, Caller};

/// Execute a WASM bytecode module in-process using wasmi.
pub fn run_wasm_module(wasm_bytes: &[u8]) -> TenthResult<()> {
    let engine = Engine::default();
    let module = wasmi::Module::new(&engine, wasm_bytes).map_err(|e| {
        TenthError::RuntimeError { message: format!("WASM module parse error: {}", e) }
    })?;

    let mut store = Store::new(&engine, 8192u32);
    let mut linker = Linker::new(&engine);

    linker.func_wrap("host", "println", |caller: Caller<'_, u32>, ptr: i32| {
        let mem = caller.get_export("memory").and_then(|e| e.into_memory()).unwrap();
        let data = mem.data(&caller);
        let end = data[ptr as usize..].iter().position(|&b| b == 0).unwrap_or(0);
        println!("{}", std::str::from_utf8(&data[ptr as usize..ptr as usize + end]).unwrap_or(""));
    }).map_err(|e| TenthError::RuntimeError { message: format!("linker: {}", e) })?;

    linker.func_wrap("host", "write_file",
        |caller: Caller<'_, u32>, path_ptr: i32, content_ptr: i32| {
            let mem = caller.get_export("memory").and_then(|e| e.into_memory()).unwrap();
            let data = mem.data(&caller);
            let rs = |p: i32| -> &str {
                let end = data[p as usize..].iter().position(|&b| b == 0).unwrap_or(0);
                std::str::from_utf8(&data[p as usize..p as usize + end]).unwrap_or("")
            };
            let _ = std::fs::write(rs(path_ptr), rs(content_ptr));
    }).map_err(|e| TenthError::RuntimeError { message: format!("linker: {}", e) })?;

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
    }).map_err(|e| TenthError::RuntimeError { message: format!("linker: {}", e) })?;

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
    }).map_err(|e| TenthError::RuntimeError { message: format!("linker: {}", e) })?;

    linker.func_wrap("host", "str_eq",
        |caller: Caller<'_, u32>, a_ptr: i32, b_ptr: i32| -> i32 {
            let mem = caller.get_export("memory").and_then(|e| e.into_memory()).unwrap();
            let data = mem.data(&caller);
            let rs = |p: i32| -> &str {
                let end = data[p as usize..].iter().position(|&b| b == 0).unwrap_or(0);
                std::str::from_utf8(&data[p as usize..p as usize + end]).unwrap_or("")
            };
            if rs(a_ptr) == rs(b_ptr) { 1 } else { 0 }
    }).map_err(|e| TenthError::RuntimeError { message: format!("linker: {}", e) })?;

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
    }).map_err(|e| TenthError::RuntimeError { message: format!("linker: {}", e) })?;

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
    }).map_err(|e| TenthError::RuntimeError { message: format!("linker: {}", e) })?;

    // Vec_new() -> i64 (pointer extended to i64)
    linker.func_wrap("host", "Vec_new",
        |mut caller: Caller<'_, u32>| -> i64 {
            let ptr = *caller.data();
            *caller.data_mut() = ptr + 24;
            ptr as i64
    }).map_err(|e| TenthError::RuntimeError { message: format!("linker: {}", e) })?;

    // Vec_len(vec: i64) -> i64
    linker.func_wrap("host", "Vec_len",
        |caller: Caller<'_, u32>, vec: i64| -> i64 {
            let vec_ptr = vec as i32 as usize;
            let mem = caller.get_export("memory").and_then(|e| e.into_memory()).unwrap();
            let data = mem.data(&caller);
            if vec_ptr + 16 <= data.len() {
                i64::from_le_bytes(data[vec_ptr+8..vec_ptr+16].try_into().unwrap())
            } else { 0 }
    }).map_err(|e| TenthError::RuntimeError { message: format!("linker: {}", e) })?;

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
    }).map_err(|e| TenthError::RuntimeError { message: format!("linker: {}", e) })?;

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
    }).map_err(|e| TenthError::RuntimeError { message: format!("linker: {}", e) })?;

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
    }).map_err(|e| TenthError::RuntimeError { message: format!("linker: {}", e) })?;

    // str_len(s: i32) -> i32  — returns length of null-terminated string
    linker.func_wrap("host", "str_len",
        |caller: Caller<'_, u32>, ptr: i32| -> i32 {
            let mem = caller.get_export("memory").and_then(|e| e.into_memory()).unwrap();
            let data = mem.data(&caller);
            let off = ptr as usize;
            data[off..].iter().position(|&b| b == 0).unwrap_or(0) as i32
    }).map_err(|e| TenthError::RuntimeError { message: format!("linker: {}", e) })?;

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
    }).map_err(|e| TenthError::RuntimeError { message: format!("linker: {}", e) })?;

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
    }).map_err(|e| TenthError::RuntimeError { message: format!("linker: {}", e) })?;

    let instance = linker.instantiate(&mut store, &module)
        .and_then(|pre| pre.start(&mut store))
        .map_err(|e| TenthError::RuntimeError {
            message: format!("WASM instantiation error: {}", e),
        })?;

    let main_fn = instance.get_typed_func::<(), i32>(&store, "main")
        .map_err(|_| TenthError::RuntimeError {
            message: "WASM module has no exported 'main' function".into(),
        })?;

    let exit_code = main_fn.call(&mut store, ())
        .map_err(|e| TenthError::RuntimeError {
            message: format!("WASM main() error: {}", e),
        })?;

    if exit_code != 0 {
        return Err(TenthError::RuntimeError {
            message: format!("WASM main() exited with code {}", exit_code),
        });
    }

    Ok(())
}
