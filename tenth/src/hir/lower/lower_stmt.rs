use std::collections::HashMap;
use crate::error::{TenthError, TenthResult};
use crate::parser::ast as ast;
use crate::hir::hir::*;
use crate::hir::types::*;
use super::Scope;
use super::build_generics_bounds;
use super::Lowerer;

impl Lowerer {
    pub(super) fn lower_stmt(&mut self, stmt: &ast::Stmt) -> TenthResult<HirStmt> {
        use ast::StmtKind;

        let span = stmt.span.clone();

        let kind = match &stmt.kind {
            StmtKind::Let { names, type_ann, mutable, init } => {
                let lowered_init = init.as_ref().map(|i| self.lower_expr(i)).transpose()?;
                // 类型注解强制化：若 type_ann 和 init 都存在，检查 shape 兼容性并合并
                let ty = match (type_ann.as_ref(), lowered_init.as_ref()) {
                    (Some(ann), Some(init_expr)) => {
                        let ann_ty = Type::from_annotation(ann);
                        Self::check_and_merge_tensor_shape(&ann_ty, &init_expr.ty, &span, "let 注解")?
                    }
                    (Some(ann), None) => Type::from_annotation(ann),
                    (None, Some(init_expr)) => init_expr.ty.clone(),
                    (None, None) => Type::Unknown,
                };

                for name in names {
                    self.scope.define_var(name.name.clone(), ty.clone(), *mutable);
                }

                HirStmtKind::Let {
                    names: names.iter().map(|n| n.name.clone()).collect(),
                    type_ann: type_ann.as_ref().map(|a| Type::from_annotation(a)),
                    mutable: *mutable,
                    init: lowered_init,
                }
            }
            StmtKind::Expr(e) => {
                HirStmtKind::Expr(self.lower_expr(e)?)
            }
            StmtKind::Return(e) => {
                HirStmtKind::Return(e.as_ref().map(|e| self.lower_expr(e)).transpose()?)
            }
            StmtKind::While { cond, body } => {
                let c = self.lower_expr(cond)?;
                // Release borrows from the condition so the body can reborrow.
                self.scope.release_borrows();
                let b = self.lower_stmt(body)?;
                HirStmtKind::While { cond: c, body: Box::new(b) }
            }
            StmtKind::For { var, iter, body } => {
                let it = self.lower_expr(iter)?;
                // Release borrows from the iterator expression so the body can reborrow.
                self.scope.release_borrows();
                // Push loop variable into scope so body can reference it
                let body_scope = Scope::with_parent(std::mem::replace(&mut self.scope, Scope::new()));
                self.scope = body_scope;
                self.scope.define_var(var.name.clone(), Type::Unknown, false);
                let b = self.lower_stmt(body)?;
                self.scope = *self.scope.parent.take().unwrap();
                HirStmtKind::For { var: var.name.clone(), iter: it, body: Box::new(b) }
            }
            StmtKind::Break => HirStmtKind::Break,
            StmtKind::Continue => HirStmtKind::Continue,
            StmtKind::Loop { body } => {
                let mut lowered_body: Vec<HirStmt> = Vec::new();
                for s in body {
                    let lowered = self.lower_stmt(s)?;
                    if !Self::creates_persistent_borrow(s) {
                        self.scope.release_borrows();
                    }
                    lowered_body.push(lowered);
                }
                HirStmtKind::Loop { body: lowered_body }
            }

        };

        Ok(HirStmt { kind, span })
    }

    pub fn lower_program(&mut self, program: &ast::Program) -> TenthResult<HirProgram> {
        for item in &program.items {
            match &item.kind {
                ast::ItemKind::StructDef { name, generics, fields, .. } => {
                    let field_types: Vec<(String, Type)> = fields.iter()
                        .map(|f| (f.name.name.clone(), Type::from_annotation(&f.type_ann)))
                        .collect();
                    if generics.is_empty() {
                        self.structs.insert(name.name.clone(), field_types);
                    } else {
                        let gen_names: Vec<String> = generics.iter().map(|g| g.name.name.clone()).collect();
                        self.generic_structs.insert(name.name.clone(), HirGenericStruct {
                            name: name.name.clone(),
                            generics: gen_names,
                            fields: field_types,
                        });
                    }
                }
                ast::ItemKind::EnumDef { name, variants } => {
                    let variant_list: Vec<(String, Vec<(String, Type)>)> = variants.iter()
                        .map(|v| {
                            let fields: Vec<(String, Type)> = match &v.kind {
                                ast::EnumVariantKind::Unit => Vec::new(),
                                ast::EnumVariantKind::Named(named_fields) => {
                                    named_fields.iter()
                                        .map(|f| (f.name.name.clone(), Type::from_annotation(&f.type_ann)))
                                        .collect()
                                }
                                ast::EnumVariantKind::Tuple(types) => {
                                    types.iter().enumerate()
                                        .map(|(i, ty)| (format!("_{}", i), Type::from_annotation(ty)))
                                        .collect()
                                }
                            };
                            (v.name.name.clone(), fields)
                        })
                        .collect();
                    self.enums.insert(name.name.clone(), variant_list);
                }
                ast::ItemKind::Trait { name, generics, methods, associated_types } => {
                    let gen_names: Vec<String> = generics.iter().map(|g| g.name.name.clone()).collect();
                    let assoc_type_names: Vec<String> = associated_types.iter().map(|t| t.name.clone()).collect();
                    let method_sigs: Vec<HirTraitMethod> = methods.iter()
                        .map(|m| {
                            let param_types: Vec<(String, Type)> = m.params.iter()
                                .map(|p| (p.name.name.clone(), Type::from_annotation(&p.type_ann)))
                                .collect();
                            let ret_ty = m.return_type.as_ref()
                                .map(|rt| Type::from_annotation(rt))
                                .unwrap_or(Type::unit());
                            // Lower default body if present
                            let default_body = if let Some(body) = &m.body {
                                let body_scope = Scope::with_parent(std::mem::replace(&mut self.scope, Scope::new()));
                                self.scope = body_scope;
                                for (n, t) in &param_types {
                                    self.scope.define_var(n.clone(), t.clone(), false);
                                }
                                let lowered = self.lower_expr(body).ok();
                                let outer_scope = std::mem::replace(&mut self.scope, Scope::new());
                                self.scope = outer_scope;
                                lowered
                            } else {
                                None
                            };
                            HirTraitMethod {
                                name: m.name.name.clone(),
                                params: param_types,
                                return_type: ret_ty,
                                default_body,
                            }
                        })
                        .collect();
                    self.trait_defs.insert(name.name.clone(), HirTraitDef {
                        name: name.name.clone(),
                        generics: gen_names,
                        methods: method_sigs,
                        associated_types: assoc_type_names,
                    });
                }
                ast::ItemKind::Function { name, generics, params, return_type, .. } => {
                    if name.name == "<expr>" || !generics.is_empty() {
                        continue;
                    }
                    let param_types: Vec<(String, Type)> = params.iter()
                        .map(|p| (p.name.name.clone(), Type::from_annotation(&p.type_ann)))
                        .collect();
                    let ret_ty = return_type.as_ref()
                        .map(|rt| Type::from_annotation(rt))
                        .unwrap_or(Type::unit());
                    let ret_ty = self.resolve_struct_type(ret_ty);
                    self.scope.define_fn(name.name.clone(), param_types, ret_ty);
                }
                _ => {}
            }
        }

        for item in &program.items {
            match &item.kind {
                ast::ItemKind::Impl { type_name, trait_name, functions, .. } => {
                    if let Some(trait_name) = trait_name {
                        let trait_name_str = trait_name.name.clone();
                        let type_name_str = type_name.name.clone();
                        let mut method_map = HashMap::new();
                        for fn_item in functions {
                            if let ast::ItemKind::Function { name, generics, params, return_type, body, .. } = &fn_item.kind {
                                let gen_names: Vec<String> = generics.iter().map(|g| g.name.name.clone()).collect();
                                let param_types: Vec<(String, Type)> = params.iter()
                                    .map(|p| (p.name.name.clone(), Type::from_annotation(&p.type_ann)))
                                    .collect();
                                let ret_ty = return_type.as_ref()
                                    .map(|rt| Type::from_annotation(rt))
                                    .unwrap_or(Type::unit());

                                let body_scope = Scope::with_parent(std::mem::replace(&mut self.scope, Scope::new()));
                                self.scope = body_scope;

                                for (n, t) in &param_types {
                                    self.scope.define_var(n.clone(), t.clone(), false);
                                }

                                let lowered_body = self.lower_expr(body)?;

                                let outer_scope = std::mem::replace(&mut self.scope, Scope::new());
                                self.scope = outer_scope;

                                let fn_def = HirFnDef {
                                    name: name.name.clone(),
                                    generics: gen_names,
                                    generics_bounds: build_generics_bounds(generics),
                                    params: param_types,
                                    return_type: ret_ty,
                                    body: lowered_body,
                                    span: fn_item.span.clone(),
                                };
                                method_map.insert(fn_def.name.clone(), fn_def);
                            }
                        }
                        self.trait_impls.entry(trait_name_str.clone())
                            .or_insert_with(HashMap::new)
                            .insert(type_name_str.clone(), method_map);

                        // Trait bound check: verify all required methods are implemented
                        if let Some(trait_def) = self.trait_defs.get(&trait_name_str) {
                            let impl_methods: Vec<String> = self.trait_impls
                                .get(&trait_name_str)
                                .and_then(|m| m.get(&type_name_str))
                                .map(|m| m.keys().cloned().collect())
                                .unwrap_or_default();
                            for method in &trait_def.methods {
                                if method.default_body.is_none() && !impl_methods.contains(&method.name) {
                                    return Err(TenthError::TypeError {
                                        line: type_name.span.line,
                                        col: type_name.span.col,
                                        message: format!(
                                            "类型 '{}' 实现 trait '{}' 时缺少方法 '{}'",
                                            type_name_str, trait_name_str, method.name
                                        ),
                                    });
                                }
                            }
                        }
                    } else {
                        let mut method_map = HashMap::new();
                        for fn_item in functions {
                            if let ast::ItemKind::Function { name, generics, params, return_type, body, .. } = &fn_item.kind {
                                let gen_names: Vec<String> = generics.iter().map(|g| g.name.name.clone()).collect();
                                let param_types: Vec<(String, Type)> = params.iter()
                                    .map(|p| (p.name.name.clone(), Type::from_annotation(&p.type_ann)))
                                    .collect();
                                let ret_ty = return_type.as_ref()
                                    .map(|rt| Type::from_annotation(rt))
                                    .unwrap_or(Type::unit());

                                let body_scope = Scope::with_parent(std::mem::replace(&mut self.scope, Scope::new()));
                                self.scope = body_scope;

                                for (n, t) in &param_types {
                                    self.scope.define_var(n.clone(), t.clone(), false);
                                }

                                let lowered_body = self.lower_expr(body)?;

                                let outer_scope = std::mem::replace(&mut self.scope, Scope::new());
                                self.scope = outer_scope;

                                let fn_def = HirFnDef {
                                    name: name.name.clone(),
                                    generics: gen_names,
                                    generics_bounds: build_generics_bounds(generics),
                                    params: param_types,
                                    return_type: ret_ty,
                                    body: lowered_body,
                                    span: fn_item.span.clone(),
                                };
                                // Also register with mangled name for WASM backend method dispatch
                                let mangled_name = format!("__{}_{}", type_name.name, fn_def.name);
                                let mut mangled_fn = fn_def.clone();
                                mangled_fn.name = mangled_name;
                                self.functions.push(mangled_fn);
                                method_map.insert(fn_def.name.clone(), fn_def);
                            }
                        }
                        self.methods.insert(type_name.name.clone(), method_map);
                    }
                }
                ast::ItemKind::Mod { name, items } => {
                    let mut lowerer = Lowerer::new();
                    lowerer.structs = self.structs.clone();
                    lowerer.enums = self.enums.clone();
                    lowerer.methods = self.methods.clone();
                    lowerer.generic_funcs = self.generic_funcs.clone();
                    lowerer.generic_structs = self.generic_structs.clone();
                    lowerer.trait_defs = self.trait_defs.clone();
                    lowerer.trait_impls = self.trait_impls.clone();
                    let mod_program = ast::Program { items: items.clone() };
                    let hir_mod = lowerer.lower_program(&mod_program)?;
                    self.modules.insert(name.name.clone(), hir_mod);
                }
                ast::ItemKind::Use { path, glob } => {
                    let path_strs: Vec<String> = path.iter().map(|p| p.name.clone()).collect();
                    if path_strs.is_empty() { continue; }

                    if *glob {
                        // `use path::*` — import all public functions from the module
                        // First try to load the file at <path>.th
                        let mut loaded_module: Option<&HirProgram> = None;
                        let mod_key = path_strs.join("::");
                        if !self.modules.contains_key(&mod_key) {
                            match self.try_import_file(&path_strs) {
                                Ok(Some(imported_hir)) => {
                                    self.modules.insert(mod_key.clone(), imported_hir);
                                }
                                Ok(None) => {
                                    // File not found or circular import — fall back to inline mod navigation below
                                }
                                Err(e) => {
                                    // Propagate the import error so the user sees the real cause
                                    // (e.g. a syntax error in the imported module) instead of a
                                    // confusing "undefined variable" later.
                                    return Err(e);
                                }
                            }
                        }
                        if let Some(m) = self.modules.get(&mod_key) {
                            loaded_module = Some(m);
                        }

                        if let Some(module) = loaded_module {
                            for fn_def in &module.functions {
                                let param_types = fn_def.params.clone();
                                let ret_ty = fn_def.return_type.clone();
                                self.scope.define_fn(fn_def.name.clone(), param_types, ret_ty);
                                if !self.functions.iter().any(|f| f.name == fn_def.name) {
                                    self.functions.push(fn_def.clone());
                                }
                            }
                        } else {
                            // Fall back to nested module navigation (for inline mod blocks)
                            let mod_name = &path_strs[0];
                            let module = if path_strs.len() == 1 {
                                self.modules.get(mod_name)
                            } else {
                                let mut current = self.modules.get(&path_strs[0]);
                                for seg in &path_strs[1..] {
                                    if let Some(m) = current {
                                        current = m.modules.get(seg);
                                    } else {
                                        break;
                                    }
                                }
                                current
                            };

                            if let Some(module) = module {
                                for fn_def in &module.functions {
                                    let param_types = fn_def.params.clone();
                                    let ret_ty = fn_def.return_type.clone();
                                    self.scope.define_fn(fn_def.name.clone(), param_types, ret_ty);
                                    if !self.functions.iter().any(|f| f.name == fn_def.name) {
                                        self.functions.push(fn_def.clone());
                                    }
                                }
                            }
                        }
                    } else if path_strs.len() >= 2 {
                        // `use path::name` — import a single function
                        let alias = path_strs.last().cloned().unwrap_or_default();
                        self.uses.push((path_strs.clone(), alias.clone()));
                        let fn_name = path_strs.last().unwrap();

                        // First try to load the file at <path[..len-1]>.th
                        let parent_path = &path_strs[..path_strs.len()-1];
                        let mod_key = parent_path.join("::");
                        let mut loaded_module: Option<&HirProgram> = None;
                        if !self.modules.contains_key(&mod_key) {
                            match self.try_import_file(parent_path) {
                                Ok(Some(imported_hir)) => {
                                    self.modules.insert(mod_key.clone(), imported_hir);
                                }
                                Ok(None) => {
                                    // File not found or circular import — fall back to inline mod navigation below
                                }
                                Err(e) => {
                                    // Propagate the import error so the user sees the real cause
                                    return Err(e);
                                }
                            }
                        }
                        if let Some(m) = self.modules.get(&mod_key) {
                            loaded_module = Some(m);
                        }

                        if let Some(module) = loaded_module {
                            // Add ALL functions from the module so that the
                            // imported function can call its helpers
                            // (e.g., parse calls _parse_value)
                            for fn_def in &module.functions {
                                if !self.functions.iter().any(|f| f.name == fn_def.name) {
                                    self.functions.push(fn_def.clone());
                                }
                            }
                            // Define the specifically requested function in scope
                            if let Some(fn_def) = module.functions.iter().find(|f| &f.name == fn_name) {
                                let param_types = fn_def.params.clone();
                                let ret_ty = fn_def.return_type.clone();
                                self.scope.define_fn(alias.clone(), param_types, ret_ty);
                            }
                        } else {
                            // Fall back to nested module navigation (for inline mod blocks)
                            let mod_name = &path_strs[0];
                            let (module, fn_name) = if path_strs.len() == 2 {
                                (self.modules.get(mod_name), &path_strs[1])
                            } else {
                                let mut current = self.modules.get(&path_strs[0]);
                                for seg in &path_strs[1..path_strs.len()-1] {
                                    if let Some(m) = current {
                                        current = m.modules.get(seg);
                                    } else {
                                        break;
                                    }
                                }
                                (current, path_strs.last().unwrap())
                            };

                            if let Some(module) = module {
                                if let Some(fn_def) = module.functions.iter().find(|f| &f.name == fn_name) {
                                    let param_types = fn_def.params.clone();
                                    let ret_ty = fn_def.return_type.clone();
                                    self.scope.define_fn(alias.clone(), param_types, ret_ty);
                                    if !self.functions.iter().any(|f| f.name == fn_def.name) {
                                        self.functions.push(fn_def.clone());
                                    }
                                }
                            }
                        }
                    }
                }
                ast::ItemKind::Function { name, generics, params, return_type, body, .. } => {
                    if name.name == "<expr>" {
                        continue;
                    }
                    let gen_names: Vec<String> = generics.iter().map(|g| g.name.name.clone()).collect();
                    let param_types: Vec<(String, Type)> = params.iter()
                        .map(|p| (p.name.name.clone(), Type::from_annotation(&p.type_ann)))
                        .collect();
                    let ret_ty = return_type.as_ref()
                        .map(|rt| Type::from_annotation(rt))
                        .unwrap_or(Type::unit());

                    let body_scope = Scope::with_parent(std::mem::replace(&mut self.scope, Scope::new()));
                    self.scope = body_scope;

                    for (n, t) in &param_types {
                        self.scope.define_var(n.clone(), t.clone(), false);
                    }

                    let lowered_body = self.lower_expr(body)?;

                    let outer_scope = std::mem::replace(&mut self.scope, Scope::new());
                    self.scope = outer_scope;

                    // 跨函数 shape 求解：合并 body 推断的 shape 到 return_type
                    // 让调用方能拿到更精确的返回 shape（如 `fn make() -> Tensor[f64, ..] { zeros(3,4) }` → [3,4]）
                    // 若 body 推断 shape 与注解 shape 静态冲突（如注解 [3,4] 但 body 推断 [2,3]），报错。
                    let merged_ret_ty = Self::check_and_merge_tensor_shape(
                        &ret_ty, &lowered_body.ty, &item.span, "函数返回值"
                    )?;

                    let fn_def = HirFnDef {
                        name: name.name.clone(),
                        generics: gen_names,
                        generics_bounds: build_generics_bounds(generics),
                        params: param_types,
                        return_type: merged_ret_ty,
                        body: lowered_body,
                        span: item.span.clone(),
                    };

                    if fn_def.generics.is_empty() {
                        self.functions.push(fn_def);
                    } else {
                        self.generic_funcs.insert(fn_def.name.clone(), fn_def);
                    }
                }
                _ => {}
            }
        }

        let mut main_expr = None;
        for item in &program.items {
            if let ast::ItemKind::Function { name, body, .. } = &item.kind {
                if name.name == "<expr>" {
                    let body_scope = Scope::with_parent(std::mem::replace(&mut self.scope, Scope::new()));
                    self.scope = body_scope;

                    let lowered_body = self.lower_expr(body)?;

                    let outer_scope = std::mem::replace(&mut self.scope, Scope::new());
                    self.scope = outer_scope;

                    main_expr = Some(lowered_body);
                    break;
                }
            }
        }

        Ok(HirProgram {
            functions: self.functions.clone(),
            generic_funcs: self.generic_funcs.values().cloned().collect(),
            main_expr,
            modules: self.modules.clone(),
            uses: self.uses.clone(),
            methods: self.methods.clone(),
            structs: self.structs.clone(),
            generic_structs: self.generic_structs.clone(),
            enums: self.enums.clone(),
            trait_defs: self.trait_defs.clone(),
            trait_impls: self.trait_impls.clone(),
            warnings: std::mem::take(&mut self.warnings),
        })
    }
}
