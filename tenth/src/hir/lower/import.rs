use crate::error::TenthResult;
use crate::hir::hir::HirProgram;
use super::Lowerer;

impl Lowerer {
    /// Create a Lowerer with additional search paths for file imports.
    /// The `std_path` should point to the `std/` directory of the Tenth installation.
    pub fn with_search_paths(search_paths: Vec<String>) -> Self {
        let mut lowerer = Self::new();
        lowerer.search_paths = search_paths;
        lowerer
    }

    /// Try to resolve a module path to a .th file and load it.
    /// Path resolution order:
    ///   1. <search_path>/<mod_path>.th
    ///   2. <search_path>/<mod_path>/mod.th
    ///   3. <search_path>/<mod_path>/<last_segment>.th
    pub(super) fn try_import_file(&mut self, mod_path: &[String]) -> TenthResult<Option<HirProgram>> {
        // Build the relative path: "std::nn::linear" -> "std/nn/linear"
        let rel_path = mod_path.join(std::path::MAIN_SEPARATOR_STR);
        let canonical_key = rel_path.replace(std::path::MAIN_SEPARATOR, "::");

        // Prevent circular imports
        if self.imported_files.contains(&canonical_key) {
            return Ok(None);
        }

        for search_dir in &self.search_paths {
            // Try <search_dir>/<rel_path>.th
            let direct = std::path::Path::new(search_dir).join(format!("{}.th", rel_path));
            if direct.exists() {
                return self.load_and_compile_file(&direct, &canonical_key);
            }

            // Try <search_dir>/<rel_path>/mod.th
            let mod_file = std::path::Path::new(search_dir).join(&rel_path).join("mod.th");
            if mod_file.exists() {
                return self.load_and_compile_file(&mod_file, &canonical_key);
            }

            // Try <search_dir>/<rel_path>/<last_segment>.th
            // （目录型模块：`use std::collections;` → std/collections/collections.th，
            //   与头部注释第 3 条一致；此前未实现导致目录模块裸引用成为静默 no-op。
            //   AUDIT-11.4.23 模块别名导入的前置。）
            if let Some(last) = mod_path.last() {
                let dir_mod = std::path::Path::new(search_dir)
                    .join(&rel_path)
                    .join(format!("{}.th", last));
                if dir_mod.exists() {
                    return self.load_and_compile_file(&dir_mod, &canonical_key);
                }
            }
        }

        Ok(None)
    }

    pub(super) fn load_and_compile_file(&mut self, path: &std::path::Path, canonical_key: &str) -> TenthResult<Option<HirProgram>> {
        let source = crate::error::read_source(path)?;

        self.imported_files.insert(canonical_key.to_string());

        let mut lexer = crate::lexer::lexer::Lexer::new(&source);
        let tokens = lexer.tokenize()?;
        let mut parser = crate::parser::parser::Parser::new(tokens);
        let program = parser.parse_program()?;

        // Create a sub-lowerer with the same search paths but fresh scope
        let mut sub_lowerer = Lowerer::with_search_paths(self.search_paths.clone());
        sub_lowerer.imported_files = self.imported_files.clone();
        // M3.5：模块模式——提取全部顶层 let 为全局（模块 main_expr 导入时
        // 不执行，顺序无关；导入方需要模块全部顶层 let 可解析）。
        sub_lowerer.is_module = true;
        let hir = sub_lowerer.lower_program(&program)?;
        self.imported_files = sub_lowerer.imported_files;

        Ok(Some(hir))
    }
}
