use crate::error::TenthResult;
use crate::hir::hir::HirProgram;
use super::Lowerer;

/// AUDIT-11.4.60(a′)/(3)：`try_import_file` 的三种相对路径形态。
/// `DirNamed` 的末段取自**相对路径**（= `mod_path` 的最后一段）。
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum CandidateForm {
    /// `<搜索目录>/<路径>.th`
    Direct,
    /// `<搜索目录>/<路径>/mod.th`
    DirMod,
    /// `<搜索目录>/<路径>/<末段>.th`（目录型模块）
    DirNamed,
}

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

    /// AUDIT-11.4.60(a′)：把「被加载模块文件所在目录」**前置**进子 lowerer 的搜索路径。
    ///
    /// 模块 A 导入同目录的模块 B（`a.th` 里写 `use b`）时，此前只按**顶层**
    /// 搜索目录找 `b.th` ⇒ 同级模块不可见。以「导入方文件所在目录」为最高
    /// 优先目录后，同级/子目录模块可解析。
    ///
    /// 三态（`ImportResolution`）与「已导入」守卫都不受影响：这里只改
    /// **去哪找**，不改**找到后怎么办**。
    fn child_search_paths(&self, module_path: &std::path::Path) -> Vec<String> {
        match module_path.parent() {
            Some(dir) => {
                let dir = dir.to_string_lossy().to_string();
                let mut paths = self.search_paths.clone();
                if let Some(pos) = paths.iter().position(|p| *p == dir) {
                    // 双保险：同一目录只留一份，且提到最前（不影响语义，只影响顺序）。
                    paths.remove(pos);
                }
                paths.insert(0, dir);
                paths
            }
            None => self.search_paths.clone(),
        }
    }

    /// 收集某个「相对路径形态」在全部搜索目录里的**不同**候选文件。
    ///
    /// 用途：AUDIT-11.4.60 裁定「多处命中 → 响亮警告」。返回的候选顺序 =
    /// 搜索目录顺序（首个即**最终采用者**，语义仍是「先命中者胜」）。
    ///
    /// 去重按规范化后的绝对路径——避免「脚本目录 == cwd」这类**同一文件
    /// 被列两次**的假歧义（实测：`tenth.exe run ./x/main.th` 时二者相同）。
    fn collect_candidates(&self, rel_path: &str, form: CandidateForm) -> Vec<std::path::PathBuf> {
        let mut cands: Vec<std::path::PathBuf> = Vec::new();
        let mut keys: Vec<std::path::PathBuf> = Vec::new();
        for search_dir in &self.search_paths {
            let p = match form {
                CandidateForm::Direct => {
                    std::path::Path::new(search_dir).join(format!("{}.th", rel_path))
                }
                CandidateForm::DirMod => {
                    std::path::Path::new(search_dir).join(rel_path).join("mod.th")
                }
                CandidateForm::DirNamed => match rel_path.rsplit(['/', '\\']).next() {
                    Some(last) if !last.is_empty() => std::path::Path::new(search_dir)
                        .join(rel_path)
                        .join(format!("{}.th", last)),
                    _ => continue,
                },
            };
            if !p.exists() {
                continue;
            }
            let key = std::fs::canonicalize(&p).unwrap_or_else(|_| p.clone());
            if keys.iter().any(|k| *k == key) {
                continue;
            }
            keys.push(key);
            cands.push(p);
        }
        cands
    }

    /// 多处命中：响亮警告（列出全部候选与最终采用者）。
    ///
    /// 只命中一处 ⇒ 不产生任何输出（零误报，`import_order_test` 的
    /// `resolvable_use_produces_no_warning` 守护这一点）。
    fn warn_ambiguous_module(
        &mut self,
        canonical_key: &str,
        form: CandidateForm,
        cands: &[std::path::PathBuf],
    ) {
        debug_assert!(cands.len() >= 2);
        let desc = match form {
            CandidateForm::Direct => "<搜索目录>/<路径>.th",
            CandidateForm::DirMod => "<搜索目录>/<路径>/mod.th",
            CandidateForm::DirNamed => "<搜索目录>/<路径>/<末段>.th",
        };
        let mut list = String::new();
        for c in cands {
            list.push_str(&format!("\n      - {}", c.display()));
        }
        let message = format!(
            "模块 '{}' 在多个搜索目录中都能解析（形态 {}）——共 {} 个候选：{}\n      最终采用（搜索顺序第一个）：{}（其余候选被忽略）",
            canonical_key,
            desc,
            cands.len(),
            list,
            cands[0].display()
        );
        self.warnings
            .push(crate::error::TenthWarning::new(0, 0, message));
    }

    /// Try to resolve a module path to a .th file and load it.
    /// Path resolution order:
    ///   1. <search_path>/<mod_path>.th
    ///   2. <search_path>/<mod_path>/mod.th
    ///   3. <search_path>/<mod_path>/<last_segment>.th
    ///
    /// 三条形态**依次**尝试（形态 1 全部目录都没命中才试形态 2），与既有
    /// 单目录实现的选择结果**逐字节一致**——见 `collect_candidates` 的注释。
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

        for form in [CandidateForm::Direct, CandidateForm::DirMod, CandidateForm::DirNamed] {
            let cands = self.collect_candidates(&rel_path, form);
            let Some(first) = cands.first() else { continue };
            // 同一路径在多个搜索目录都能解析 ⇒ 响亮警告（候选全文 + 采用者）。
            if cands.len() > 1 {
                self.warn_ambiguous_module(&canonical_key, form, &cands);
            }
            return self
                .load_and_compile_file(first, &canonical_key)
                .map(ImportResolution::Resolved);
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

        // Create a sub-lowerer with the same search paths but fresh scope.
        // AUDIT-11.4.60(a′)：**被加载模块文件所在目录**前置为最高优先搜索目录 ⇒
        // 模块 A 可以 `use b` 导入与自己同目录的模块 B（此前只按顶层搜索目录找）。
        let mut sub_lowerer = Lowerer::with_search_paths(self.child_search_paths(path));
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
        // AUDIT-11.4.60(a′)：回流**嵌套模块**（内联 `mod`）。先在
        // `imported_files` 被移走**之前**取出「非文件键」的候选，避免顺序陷阱。
        //
        // last-segment 键在 `try_import_file` 里**从不**被当作文件解析
        // （文件形态只吃 `mod_path` 的子路径），故保留该键不会产生
        // 「按错误键取到错误模块」的风险；它只让后续 `use p::sub::f` 的
        // inline-mod 导航能命中（此前必然 NotFound）。
        let sub_nested: Vec<(String, HirProgram)> = sub_lowerer
            .modules
            .iter()
            .filter(|(k, _)| !sub_lowerer.imported_files.contains(*k))
            .map(|(k, m)| (k.clone(), m.clone()))
            .collect();
        self.imported_files = sub_lowerer.imported_files;
        for (name, m) in sub_nested {
            self.modules.entry(name).or_insert(m);
        }
        // 子模块 lowering 时产生的诊断（含「use 解析不到模块」的响亮化警告）此前被
        // 整体丢弃 ⇒ 嵌套的静默空转不会有任何输出。上浮到父级，由 main.rs 统一打印
        // （`for w in &hir.warnings`）。注意 `lower_program` 用 `mem::take` 把
        // warnings 搬进了返回的 HirProgram，故此处从 `hir` 取。
        self.warnings.extend(hir.warnings.iter().cloned());

        Ok(hir)
    }
}
