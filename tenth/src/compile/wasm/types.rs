//! Struct layout computation and field resolution.

use std::collections::HashMap;
use wasm_encoder::ValType;
use crate::error::{TenthError, TenthResult};
use crate::hir::hir::*;
use crate::hir::types::Type;
use super::{field_size_and_type, WasmCompiler};

impl WasmCompiler {
    // ── Struct layout ───────────────────────────────────────────────────

    pub(super) fn build_struct_layouts(&mut self, program: &HirProgram) {
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

    pub(super) fn struct_size(&self, name: &str) -> u32 {
        self.struct_layouts.get(name).map_or(0, |layout| {
            layout.values().map(|(off, sz, _)| off + sz).max().unwrap_or(0)
        })
    }

    pub(super) fn infer_struct_name(&self, ty: &Type) -> String {
        match ty {
            Type::Struct(name) => name.clone(),
            Type::TypeParam { name } if self.struct_layouts.contains_key(name) => name.clone(),
            Type::Ref(inner) | Type::MutRef(inner) => self.infer_struct_name(inner),
            _ => String::new(),
        }
    }

    /// Find a struct that contains the given field. Returns (struct_name, offset, size, vt).
    pub(super) fn resolve_field(&self, struct_hint: &str, field: &str) -> TenthResult<(String, u32, u32, ValType)> {
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
        Err(TenthError::RuntimeError { line: None, col: None,
            message: format!("WASM: 没有结构体包含字段 '{}'", field),
        })
    }
}
