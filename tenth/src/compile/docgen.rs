use crate::hir::hir::*;

/// Extract documentation from HIR function definitions.
/// Doc comments in Tenth source start with `//!`.
pub struct DocGen;

impl DocGen {
    pub fn new() -> Self { DocGen }

    /// Generate Markdown API documentation from a HIR program.
    pub fn generate(&self, program: &HirProgram) -> String {
        let mut md = String::new();
        md.push_str("# API Documentation\n\n");

        for func in &program.functions {
            md.push_str(&self.document_function(func));
        }

        for func in &program.generic_funcs {
            md.push_str(&self.document_function(func));
        }

        if md == "# API Documentation\n\n" {
            md.push_str("*No documented functions found.*\n");
        }

        md
    }

    fn document_function(&self, func: &HirFnDef) -> String {
        let mut md = String::new();

        md.push_str(&format!("## `{}`\n\n", func.name));

        // Parameters
        if !func.params.is_empty() {
            md.push_str("**Parameters:**\n\n");
            for (name, ty) in &func.params {
                md.push_str(&format!("- `{}`: {}\n", name, ty));
            }
            md.push('\n');
        }

        // Return type
        md.push_str(&format!("**Returns:** `{}`\n\n", func.return_type));

        // Generics
        if !func.generics.is_empty() {
            md.push_str(&format!("**Generics:** {}\n\n",
                func.generics.iter().map(|g| format!("`{}`", g)).collect::<Vec<_>>().join(", ")));
        }

        md.push_str("---\n\n");
        md
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_docgen_empty() {
        let program = HirProgram {
            functions: vec![],
            generic_funcs: vec![],
            main_expr: None,
            modules: std::collections::HashMap::new(),
            uses: vec![],
            methods: std::collections::HashMap::new(),
            generic_structs: std::collections::HashMap::new(),
            trait_defs: std::collections::HashMap::new(),
            trait_impls: std::collections::HashMap::new(),
        };
        let generator = DocGen::new();
        let md = generator.generate(&program);
        assert!(md.contains("API Documentation"));
    }

    #[test]
    fn test_docgen_function() {
        let program = HirProgram {
            functions: vec![HirFnDef {
                name: "add".to_string(),
                generics: vec![],
                generics_bounds: std::collections::HashMap::new(),
                params: vec![
                    ("a".to_string(), crate::hir::types::Type::i32()),
                    ("b".to_string(), crate::hir::types::Type::i32()),
                ],
                return_type: crate::hir::types::Type::i32(),
                body: HirExpr {
                    kind: HirExprKind::Literal(Literal::Int(0)),
                    ty: crate::hir::types::Type::i32(),
                    span: crate::lexer::token::Span { line: 1, col: 1 },
                },
                span: crate::lexer::token::Span { line: 1, col: 1 },
            }],
            generic_funcs: vec![],
            main_expr: None,
            modules: std::collections::HashMap::new(),
            uses: vec![],
            methods: std::collections::HashMap::new(),
            generic_structs: std::collections::HashMap::new(),
            trait_defs: std::collections::HashMap::new(),
            trait_impls: std::collections::HashMap::new(),
        };
        let generator = DocGen::new();
        let md = generator.generate(&program);
        assert!(md.contains("## `add`"));
        assert!(md.contains("`a`: I32"));
        assert!(md.contains("`I32`"));
    }
}
