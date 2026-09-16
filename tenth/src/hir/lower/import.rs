use crate::error::TenthResult;
use crate::hir::hir::HirProgram;
use super::Lowerer;

/// `try_import_file` 的解析结果——**三态**。
///
/// AUDIT-11.4.41 真根因：旧实现把「已导入（去重 / 循环导入守卫短路）」与
/// 「搜索路径里没有这个文件」**复用同一个 `Ok(None)`**，调用点只能按「文件不存在」
/// 处理 ⇒ 已导入模块的内部 `use` 拿不到缓存又不去加载，落到 inline-mod fallback
/// **静默空转**（无诊断、无绑定），表现为「顺序相关：后导入者的内部 use 失效」。
/// 拆成三态后，调用点必须分别处理，不再可能静默吞掉「已导入」。
pub(super) enum ImportResolution {
    /// 命中：本次从磁盘加载，或复用**此前加载的 HIR 缓存**（AUDIT-11.4.41 修复点）。
    Resolved(HirProgram),
    /// 此前已导入（`imported_files` 命中）但缓存里没有 HIR。
    /// 正常路径不可达（加载时必同时写入 `modules` 缓存）；出现即为缺陷，调用点**响亮报告**。
    AlreadyImported,
    /// 所有搜索路径都没有该模块文件（真·不存在）。
    NotFound,
}

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
    ///
    /// 返回三态（`ImportResolution`）而不是 `Option`：**「已导入」≠「文件不存在」**
    /// （AUDIT-11.4.41）。
    pub(super) fn try_import_file(&mut self, mod_path: &[String]) -> TenthResult<ImportResolution> {
        // Build the relative path: "std::nn::linear" -> "std/nn/linear"
        let rel_path = mod_path.join(std::path::MAIN_SEPARATOR_STR);
        let canonical_key = rel_path.replace(std::path::MAIN_SEPARATOR, "::");

        // 已导入（含循环导入守卫）：优先**命中缓存**复用 HIR——这正是修复点：
        // 兄弟模块此前已把该模块 lowering 过，其 HIR 就在 `modules` 里，
        // 若这里返回「找不到」，后导入者的内部 use 就整段空转。
        if self.imported_files.contains(&canonical_key) {
            return Ok(match self.modules.get(&canonical_key) {
                Some(m) => ImportResolution::Resolved(m.clone()),
                None => ImportResolution::AlreadyImported,
            });
        }

        for search_dir in &self.search_paths {
            // Try <search_dir>/<rel_path>.th
            let direct = std::path::Path::new(search_dir).join(format!("{}.th", rel_path));
            if direct.exists() {
                return self.load_and_compile_file(&direct, &canonical_key)
                    .map(ImportResolution::Resolved);
            }

            // Try <search_dir>/<rel_path>/mod.th
            let mod_file = std::path::Path::new(search_dir).join(&rel_path).join("mod.th");
            if mod_file.exists() {
                return self.load_and_compile_file(&mod_file, &canonical_key)
                    .map(ImportResolution::Resolved);
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
                    return self.load_and_compile_file(&dir_mod, &canonical_key)
                        .map(ImportResolution::Resolved);
                }
            }
        }

        Ok(ImportResolution::NotFound)
    }

    pub(super) fn load_and_compile_file(
        &mut self,
        path: &std::path::Path,
        canonical_key: &str,
    ) -> TenthResult<HirProgram> {
        let source = crate::error::read_source(path)?;

        self.imported_files.insert(canonical_key.to_string());

        let mut lexer = crate::lexer::lexer::Lexer::new(&source);
        let tokens = lexer.tokenize()?;
        let mut parser = crate::parser::parser::Parser::new(tokens);
        let program = parser.parse_program()?;

        // Create a sub-lowerer with the same search paths but fresh scope
        let mut sub_lowerer = Lowerer::with_search_paths(self.search_paths.clone());
        sub_lowerer.imported_files = self.imported_files.clone();
        // AUDIT-11.4.41：**向下**传播文件模块缓存（键 = `imported_files` 中的 canonical key）。
        //
        // 为什么必须传播：被导入模块的内部 `use` 会先命中 `imported_files`（去重守卫），
        // 此时若子 lowerer 的 `modules` 是空的，缓存取不到 → 「已导入」被打回 → 空转。
        // 只传播 `imported_files` 的键（= 文件模块），**不**把父级内联 `mod` 的命名空间
        // 漏进子模块（内联 mod 只应在声明它的编译单元内可见）。
        for key in &self.imported_files {
            if let Some(m) = self.modules.get(key) {
                sub_lowerer.modules.insert(key.clone(), m.clone());
            }
        }
        // M3.5：模块模式——提取全部顶层 let 为全局（模块 main_expr 导入时
        // 不执行，顺序无关；导入方需要模块全部顶层 let 可解析）。
        sub_lowerer.is_module = true;
        let hir = sub_lowerer.lower_program(&program)?;

        // AUDIT-11.4.41：**向上**回流子 lowerer 的模块缓存（此前只回流 `imported_files`，
        // `modules` 被直接丢弃 ⇒ 子模块内部 `use` 加载的模块级 HIR 在父级不可见）。
        // 同样只收 `imported_files` 的键，避免把子模块的内联 mod 泄漏到导入方。
        for key in &sub_lowerer.imported_files {
            if let Some(m) = sub_lowerer.modules.get(key) {
                self.modules.entry(key.clone()).or_insert_with(|| m.clone());
            }
        }
        self.imported_files = sub_lowerer.imported_files;
        // 子模块 lowering 时产生的诊断（含「use 解析不到模块」的响亮化警告）此前被
        // 整体丢弃 ⇒ 嵌套的静默空转不会有任何输出。上浮到父级，由 main.rs 统一打印
        // （`for w in &hir.warnings`）。注意 `lower_program` 用 `mem::take` 把
        // warnings 搬进了返回的 HirProgram，故此处从 `hir` 取。
        self.warnings.extend(hir.warnings.iter().cloned());

        Ok(hir)
    }
}
