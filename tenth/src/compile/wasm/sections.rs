//! WASM section emission (type, import, function, memory, table, global,
//! element, export, code, data).

use wasm_encoder::{
    CodeSection, ConstExpr, DataSection, ElementSection, Elements, EntityType, ExportKind,
    ExportSection, FunctionSection, GlobalSection, ImportSection, MemorySection, MemoryType, Module,
    RefType, TableSection, TableType, TypeSection, ValType,
};
use crate::error::TenthResult;
use crate::hir::hir::*;
use super::{IMPORT_COUNT, WasmCompiler, to_val_type};

impl WasmCompiler {
    // ── Section builders ────────────────────────────────────────────────

    pub(super) fn emit_type_section(&mut self, module: &mut Module, program: &HirProgram) -> TenthResult<()> {
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
        // Phase 5.2 F1：host_make_tensor_f16/bf16 复用相同签名 (i32,i32,i32)->i64，
        // type_cache 会去重，所以 type 索引仍为 12（与 tensor_from_vec 共享）
        // M1-S1（P4）：标量 math host 函数 sin/cos/ln（f64->f64）与 pow（f64,f64->f64）
        reg(vec![ValType::F64], vec![ValType::F64]);                                // 18: sin/cos/ln
        reg(vec![ValType::F64, ValType::F64], vec![ValType::F64]);                  // 19: pow
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

    pub(super) fn emit_import_section(&mut self, module: &mut Module) {
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
        // Phase 5.2 F1：F16/BF16 张量专用 hostcall（与 tensor_from_vec 同签名）
        imports.import("host", "host_make_tensor_f16", EntityType::Function(ti(vec![ValType::I32, ValType::I32, ValType::I32], vec![ValType::I64])));
        imports.import("host", "host_make_tensor_bf16", EntityType::Function(ti(vec![ValType::I32, ValType::I32, ValType::I32], vec![ValType::I64])));
        // M1-S1（P4）：标量 math host 函数
        imports.import("host", "host_sin", EntityType::Function(ti(vec![ValType::F64], vec![ValType::F64])));
        imports.import("host", "host_cos", EntityType::Function(ti(vec![ValType::F64], vec![ValType::F64])));
        imports.import("host", "host_ln", EntityType::Function(ti(vec![ValType::F64], vec![ValType::F64])));
        imports.import("host", "host_pow", EntityType::Function(ti(vec![ValType::F64, ValType::F64], vec![ValType::F64])));
        module.section(&imports);
    }

    pub(super) fn emit_function_section(&mut self, module: &mut Module, program: &HirProgram) -> u32 {
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

    pub(super) fn emit_memory_section(&self, module: &mut Module) {
        let mut mem = MemorySection::new();
        mem.memory(MemoryType { minimum: 16, maximum: Some(256), memory64: false, shared: false, page_size_log2: None });
        module.section(&mem);
    }

    /// D5.2: Emit table section (funcref table for call_indirect).
    /// Only emitted when there are closures.
    pub(super) fn emit_table_section(&self, module: &mut Module) {
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

    /// Global section: 程序级顶层 let 全局（M1-S1 P3）。
    /// 每个全局声明一个 mut i64 WASM 全局（初始 0），函数体经 GlobalGet/GlobalSet
    /// 访问；main 开头求值 init 后 GlobalSet 初始化。bump pointer 仍由 host 管理
    /// （无需 WASM 全局）。
    pub(super) fn emit_global_section(&mut self, module: &mut Module, program: &HirProgram) {
        let mut globals = GlobalSection::new();
        let mut gi = 0u32;
        for g in &program.globals {
            if g.name.is_empty() {
                continue;
            }
            self.global_map.insert(g.name.clone(), gi);
            globals.global(
                wasm_encoder::GlobalType { val_type: ValType::I64, mutable: true, shared: false },
                &ConstExpr::i64_const(0),
            );
            gi += 1;
        }
        module.section(&globals);
    }

    /// D5.2: Emit element section to fill the table with closure function indices.
    /// Only emitted when there are closures.
    pub(super) fn emit_elem_section(&self, module: &mut Module) {
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

    pub(super) fn emit_export_section(&mut self, module: &mut Module, program: &HirProgram) {
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

    pub(super) fn emit_code_section(&mut self, module: &mut Module, program: &HirProgram) -> TenthResult<()> {
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

    pub(super) fn emit_data_section(&self, module: &mut Module) {
        let mut data = DataSection::new();
        data.active(0, &ConstExpr::i32_const(0), self.string_data.clone());
        module.section(&data);
    }
}
