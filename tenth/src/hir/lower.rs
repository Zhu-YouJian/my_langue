use std::collections::HashMap;
use std::collections::HashSet;
use crate::error::{TenthError, TenthResult};
use crate::lexer::token::Span;
use crate::parser::ast as ast;
use super::hir::*;
use super::types::*;

#[derive(Debug, Clone, Copy, PartialEq)]
enum Ownership {
    Owned,
    SharedRef(usize),
    ExclusiveRef,
    Moved,
}

struct Scope {
    variables: HashMap<String, (Type, bool)>,
    functions: HashMap<String, (Vec<(String, Type)>, Type)>,
    ownership: HashMap<String, Ownership>,
    parent: Option<Box<Scope>>,
}

impl Scope {
    fn new() -> Self {
        Scope {
            variables: HashMap::new(),
            functions: HashMap::new(),
            ownership: HashMap::new(),
            parent: None,
        }
    }

    fn with_parent(parent: Scope) -> Self {
        Scope {
            variables: HashMap::new(),
            functions: HashMap::new(),
            ownership: HashMap::new(),
            parent: Some(Box::new(parent)),
        }
    }

    fn lookup_var(&self, name: &str) -> Option<(Type, bool)> {
        if let Some(v) = self.variables.get(name) {
            return Some(v.clone());
        }
        self.parent.as_ref().and_then(|p| p.lookup_var(name))
    }

    fn get_ownership(&self, name: &str) -> Option<Ownership> {
        if let Some(o) = self.ownership.get(name) {
            return Some(*o);
        }
        self.parent.as_ref().and_then(|p| p.get_ownership(name))
    }

    fn set_ownership(&mut self, name: &str, state: Ownership) {
        self.ownership.insert(name.to_string(), state);
    }

    fn check_use(&self, name: &str, span: &crate::lexer::token::Span) -> TenthResult<()> {
        if let Some(Ownership::Moved) = self.get_ownership(name) {
            return Err(TenthError::TypeError {
                line: span.line, col: span.col,
                message: format!("use of moved value '{}'", name),
            });
        }
        Ok(())
    }

    fn check_borrow_shared(&self, name: &str, span: &crate::lexer::token::Span) -> TenthResult<()> {
        match self.get_ownership(name) {
            Some(Ownership::ExclusiveRef) => Err(TenthError::TypeError {
                line: span.line, col: span.col,
                message: format!("cannot borrow '{}' as shared because it is also borrowed as mutable", name),
            }),
            Some(Ownership::Moved) => Err(TenthError::TypeError {
                line: span.line, col: span.col,
                message: format!("cannot borrow moved value '{}'", name),
            }),
            _ => Ok(()),
        }
    }

    fn check_borrow_mut(&self, name: &str, span: &crate::lexer::token::Span) -> TenthResult<()> {
        match self.get_ownership(name) {
            Some(Ownership::SharedRef(n)) if n > 0 => Err(TenthError::TypeError {
                line: span.line, col: span.col,
                message: format!("cannot borrow '{}' as mutable because it is also borrowed as shared", name),
            }),
            Some(Ownership::ExclusiveRef) => Err(TenthError::TypeError {
                line: span.line, col: span.col,
                message: format!("cannot borrow '{}' as mutable more than once at a time", name),
            }),
            Some(Ownership::Moved) => Err(TenthError::TypeError {
                line: span.line, col: span.col,
                message: format!("cannot borrow moved value '{}'", name),
            }),
            _ => Ok(()),
        }
    }

    fn define_var(&mut self, name: String, ty: Type, mutable: bool) {
        self.variables.insert(name.clone(), (ty, mutable));
        self.ownership.insert(name, Ownership::Owned);
    }

    fn define_fn(&mut self, name: String, params: Vec<(String, Type)>, ret: Type) {
        self.functions.insert(name, (params, ret));
    }

    fn lookup_fn(&self, name: &str) -> Option<(Vec<(String, Type)>, Type)> {
        if let Some(f) = self.functions.get(name) {
            return Some(f.clone());
        }
        self.parent.as_ref().and_then(|p| p.lookup_fn(name))
    }
}

pub struct Lowerer {
    scope: Scope,
    functions: Vec<HirFnDef>,
    generic_funcs: HashMap<String, HirFnDef>,
    structs: HashMap<String, Vec<(String, Type)>>,
    generic_structs: HashMap<String, HirGenericStruct>,
    enums: HashMap<String, Vec<(String, Vec<(String, Type)>)>>,
    methods: HashMap<String, HashMap<String, HirFnDef>>,
    modules: HashMap<String, HirProgram>,
    uses: Vec<(Vec<String>, String)>,
    trait_defs: HashMap<String, HirTraitDef>,
    trait_impls: HashMap<String, HashMap<String, HashMap<String, HirFnDef>>>,
    /// Directories to search for imported .th files
    search_paths: Vec<String>,
    /// Set of files already imported (to prevent circular imports)
    imported_files: HashSet<String>,
}

impl Lowerer {
    pub fn new() -> Self {
        let mut scope = Scope::new();
        scope.define_fn(
            "tensor".to_string(),
            vec![("data".to_string(), Type::Unknown)],
            Type::Tensor {
                dtype: BaseType::F64,
                dims: vec![Dim::Any],
            },
        );
        let mut lowerer = Lowerer {
            scope,
            functions: Vec::new(),
            generic_funcs: HashMap::new(),
            structs: HashMap::new(),
            generic_structs: HashMap::new(),
            enums: HashMap::new(),
            methods: HashMap::new(),
            modules: HashMap::new(),
            uses: Vec::new(),
            trait_defs: HashMap::new(),
            trait_impls: HashMap::new(),
            search_paths: Vec::new(),
            imported_files: HashSet::new(),
        };

        lowerer.trait_defs.insert("Display".to_string(), HirTraitDef {
            name: "Display".to_string(),
            generics: vec![],
            methods: vec![HirTraitMethod {
                name: "display".to_string(),
                params: vec![("self".to_string(), Type::Unknown)],
                return_type: Type::str_(),
                default_body: None,
            }],
            associated_types: vec![],
        });
        lowerer.trait_defs.insert("Eq".to_string(), HirTraitDef {
            name: "Eq".to_string(),
            generics: vec![],
            methods: vec![HirTraitMethod {
                name: "eq".to_string(),
                params: vec![("self".to_string(), Type::Unknown), ("other".to_string(), Type::Unknown)],
                return_type: Type::bool_(),
                default_body: None,
            }],
            associated_types: vec![],
        });
        lowerer.trait_defs.insert("Clone".to_string(), HirTraitDef {
            name: "Clone".to_string(),
            generics: vec![],
            methods: vec![HirTraitMethod {
                name: "clone".to_string(),
                params: vec![("self".to_string(), Type::Unknown)],
                return_type: Type::Unknown,
                default_body: None,
            }],
            associated_types: vec![],
        });

        // Preload Option enum
        lowerer.enums.insert("Option".to_string(), vec![
            ("Some".to_string(), vec![("value".to_string(), Type::Unknown)]),
            ("None".to_string(), vec![]),
        ]);

        // Preload Result enum
        lowerer.enums.insert("Result".to_string(), vec![
            ("Ok".to_string(), vec![("value".to_string(), Type::Unknown)]),
            ("Err".to_string(), vec![("error".to_string(), Type::str_())]),
        ]);

        lowerer
    }

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
    fn try_import_file(&mut self, mod_path: &[String]) -> TenthResult<Option<HirProgram>> {
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
        }

        Ok(None)
    }

    fn load_and_compile_file(&mut self, path: &std::path::Path, canonical_key: &str) -> TenthResult<Option<HirProgram>> {
        let source = std::fs::read_to_string(path)
            .map_err(|e| TenthError::RuntimeError {
                message: format!("cannot read import '{}': {}", path.display(), e),
            })?;

        self.imported_files.insert(canonical_key.to_string());

        let mut lexer = crate::lexer::lexer::Lexer::new(&source);
        let tokens = lexer.tokenize()?;
        let mut parser = crate::parser::parser::Parser::new(tokens);
        let program = parser.parse_program()?;

        // Create a sub-lowerer with the same search paths but fresh scope
        let mut sub_lowerer = Lowerer::with_search_paths(self.search_paths.clone());
        sub_lowerer.imported_files = self.imported_files.clone();
        let hir = sub_lowerer.lower_program(&program)?;
        self.imported_files = sub_lowerer.imported_files;

        Ok(Some(hir))
    }

    fn lower_expr(&mut self, expr: &ast::Expr) -> TenthResult<HirExpr> {
        use ast::ExprKind;

        let span = expr.span.clone();

        let (kind, ty) = match &expr.kind {
            ExprKind::Literal(lit) => {
                let (hir_lit, ty) = match lit {
                    ast::Literal::Int(n) => (Literal::Int(*n), Type::i32()),
                    ast::Literal::Float(n) => (Literal::Float(*n), Type::f64()),
                    ast::Literal::Bool(b) => (Literal::Bool(*b), Type::bool_()),
                    ast::Literal::String(s) => (Literal::String(s.clone()), Type::str_()),
                };
                (HirExprKind::Literal(hir_lit), ty)
            }

            ExprKind::Ident(ident) => {
                if ident.name.contains("::") {
                    let parts: Vec<&str> = ident.name.splitn(2, "::").collect();
                    if parts.len() == 2 {
                        let enum_name = parts[0];
                        let variant = parts[1];
                        if let Some(variants) = self.enums.get(enum_name) {
                            if variants.iter().any(|(v, _)| v == variant) {
                                return Ok(HirExpr {
                                    kind: HirExprKind::EnumLiteral {
                                        enum_name: enum_name.to_string(),
                                        variant: variant.to_string(),
                                        fields: Vec::new(),
                                    },
                                    ty: Type::Enum(enum_name.to_string()),
                                    span,
                                });
                            }
                        }
                    }
                    (HirExprKind::Var(ident.name.clone()), Type::Unknown)
                } else {
                    self.scope.check_use(&ident.name, &ident.span)?;
                    let var_info = self.scope.lookup_var(&ident.name);
                    let fn_info = self.scope.lookup_fn(&ident.name);
                    if var_info.is_none() && fn_info.is_none() {
                        match ident.name.as_str() {
                            "println" | "eprintln" | "tensor" | "rand" | "randn"
                            | "read_file" | "write_file" | "write_bytes" | "str_at" | "Vec::new" | "HashMap::new"
                            | "compile_host" | "compile_program"
                            | "start_grad" | "new_grad" | "stop_grad"
                            | "param" | "backward" | "grad" | "zero_grad"
                            | "cross_entropy"
                            | "abs" | "sqrt" | "sin" | "cos" | "ln" | "pow"
                            | "zeros" | "ones"
                            | "save_weights" | "load_weights"
                            | "format" | "parse_int" | "parse_float"
                            | "path_join" | "path_exists" | "path_is_file" | "path_is_dir"
                            | "mkdir" | "list_dir" | "file_size" | "remove_file" | "copy_file"
                            | "lexer_new" | "lexer_tokenize" | "parse_program"
                            | "lower_program" | "compile_to_wasm" => {
                                (HirExprKind::Var(ident.name.clone()), Type::Unknown)
                            }
                            _ => {
                                return Err(TenthError::TypeError {
                                    line: span.line,
                                    col: span.col,
                                    message: format!("undefined variable '{}'", ident.name),
                                });
                            }
                        }
                    } else {
                        let ty = var_info.map(|v| v.0).or_else(|| {
                            fn_info.map(|f| Type::FnType {
                                params: f.0.iter().map(|(_, t)| t.clone()).collect(),
                                ret: Box::new(f.1),
                            })
                        }).unwrap_or(Type::Unknown);
                        (HirExprKind::Var(ident.name.clone()), ty)
                    }
                }
            }

            ExprKind::Binary { op, left, right } => {
                let l = self.lower_expr(left)?;
                let r = self.lower_expr(right)?;
                let ty = self.infer_binary_type(op, &l.ty, &r.ty);
                let hir_op = lower_binop(op);
                (HirExprKind::Binary { op: hir_op, left: Box::new(l), right: Box::new(r), ty: ty.clone() }, ty)
            }

            ExprKind::Unary { op, expr: inner } => {
                let e = self.lower_expr(inner)?;
                let ty = e.ty.clone();
                let hir_op = match op {
                    ast::UnaryOp::Neg => UnaryOp::Neg,
                    ast::UnaryOp::Not => UnaryOp::Not,
                    ast::UnaryOp::Try => UnaryOp::Try,
                };
                (HirExprKind::Unary { op: hir_op, expr: Box::new(e), ty: ty.clone() }, ty)
            }

            ExprKind::Call { func, args } => {
                let f = self.lower_expr(func)?;
                let lowered_args: Vec<_> = args.iter()
                    .map(|a| self.lower_expr(a))
                    .collect::<TenthResult<_>>()?;

                // If the func is an EnumLiteral, merge args as tuple fields
                if let HirExprKind::EnumLiteral { enum_name, variant, fields } = &f.kind {
                    if fields.is_empty() && !lowered_args.is_empty() {
                        let tuple_fields: Vec<(String, HirExpr)> = lowered_args.into_iter().enumerate()
                            .map(|(i, a)| (format!("_{}", i), a))
                            .collect();
                        return Ok(HirExpr {
                            kind: HirExprKind::EnumLiteral {
                                enum_name: enum_name.clone(),
                                variant: variant.clone(),
                                fields: tuple_fields,
                            },
                            ty: Type::Unknown,
                            span,
                        });
                    }
                }

                let ret_ty = self.resolve_call_type(&f, &lowered_args, &span)?;

                (HirExprKind::Call {
                    func: Box::new(f),
                    args: lowered_args,
                    ret_ty: ret_ty.clone(),
                }, ret_ty)
            }

            ExprKind::GenericCall { func, generics, args } => {
                let func_name = match &func.kind {
                    ExprKind::Ident(ident) => ident.name.clone(),
                    _ => {
                        return Err(TenthError::TypeError {
                            line: span.line,
                            col: span.col,
                            message: "generic call target must be a named function".into(),
                        });
                    }
                };

                let template = self.generic_funcs.get(&func_name)
                    .ok_or_else(|| TenthError::TypeError {
                        line: span.line,
                        col: span.col,
                        message: format!("undefined generic function '{}'", func_name),
                    })?
                    .clone();

                let type_args: Vec<Type> = generics.iter()
                    .map(|ta| Type::from_annotation(ta))
                    .collect();

                let mut type_map: HashMap<String, Type> = HashMap::new();
                for (i, gen_name) in template.generics.iter().enumerate() {
                    type_map.insert(gen_name.clone(), type_args.get(i).cloned().unwrap_or(Type::Unknown));
                }

                let inst_ret_ty = substitute_type(&template.return_type, &type_map);

                let lowered_args: Vec<HirExpr> = args.iter()
                    .map(|a| self.lower_expr(a))
                    .collect::<TenthResult<_>>()?;

                (HirExprKind::GenericCall {
                    func: Box::new(HirExpr {
                        kind: HirExprKind::Var(func_name),
                        ty: Type::Unknown,
                        span: span.clone(),
                    }),
                    generics: type_args,
                    args: lowered_args,
                    ret_ty: inst_ret_ty.clone(),
                }, inst_ret_ty)
            }

            ExprKind::MethodCall { receiver, method, args } => {
                let recv = self.lower_expr(receiver)?;
                let lowered_args: Vec<_> = args.iter()
                    .map(|a| self.lower_expr(a))
                    .collect::<TenthResult<_>>()?;

                let ret_ty = self.resolve_method_type(&recv.ty, &method.name, &lowered_args);

                (HirExprKind::MethodCall {
                    receiver: Box::new(recv),
                    method: method.name.clone(),
                    args: lowered_args,
                    ret_ty: ret_ty.clone(),
                }, ret_ty)
            }

            ExprKind::Index { target, indices } => {
                let t = self.lower_expr(target)?;
                let lowered_indices: Vec<_> = indices.iter()
                    .map(|idx| self.lower_index(idx))
                    .collect::<TenthResult<_>>()?;

                let ty = self.index_type(&t.ty, &lowered_indices);
                (HirExprKind::Index { target: Box::new(t), indices: lowered_indices }, ty)
            }

            ExprKind::Field { target, field } => {
                let t = self.lower_expr(target)?;
                // Unwrap reference types to get the inner struct type
                let inner_ty = match &t.ty {
                    Type::Ref(inner) | Type::MutRef(inner) => inner.as_ref(),
                    other => other,
                };
                let field_ty = match inner_ty {
                    Type::Struct(name) | Type::TypeParam { name } => {
                        self.structs.get(name)
                            .and_then(|fields| fields.iter().find(|(n, _)| n == &field.name))
                            .map(|(_, ty)| ty.clone())
                            .unwrap_or(Type::Unknown)
                    }
                    _ => Type::Unknown,
                };
                (HirExprKind::Field { target: Box::new(t), field: field.name.clone() }, field_ty)
            }

            ExprKind::TensorLiteral(data) => {
                let lowered: Vec<Vec<HirExpr>> = data.iter()
                    .map(|row| row.iter().map(|e| self.lower_expr(e)).collect())
                    .collect::<TenthResult<_>>()?;
                let rows = lowered.len() as i64;
                let cols = lowered.first().map_or(0, |r| r.len() as i64);
                let ty = Type::Tensor { dtype: BaseType::F64, dims: vec![Dim::Known(rows), Dim::Known(cols)] };
                (HirExprKind::TensorLiteral { data: lowered, ty: ty.clone() }, ty)
            }

            ExprKind::ArrayLiteral(elements) => {
                let lowered: Vec<HirExpr> = elements.iter()
                    .map(|e| self.lower_expr(e))
                    .collect::<TenthResult<_>>()?;
                let elem_ty = lowered.first()
                    .map(|e| e.ty.clone())
                    .unwrap_or(Type::Unknown);
                let ty = Type::Array(Box::new(elem_ty));
                (HirExprKind::ArrayLiteral { elements: lowered, ty: ty.clone() }, ty)
            }

            ExprKind::Range { start, end, inclusive } => {
                let s = start.as_ref().map(|e| self.lower_expr(e)).transpose()?;
                let e = end.as_ref().map(|e| self.lower_expr(e)).transpose()?;
                let ty = s.as_ref()
                    .or(e.as_ref())
                    .map(|expr| expr.ty.clone())
                    .unwrap_or(Type::i32());
                (HirExprKind::Range { start: s.map(Box::new), end: e.map(Box::new), inclusive: *inclusive }, ty)
            }

            ExprKind::If { cond, then_branch, else_branch } => {
                let c = self.lower_expr(cond)?;
                let t = self.lower_expr(then_branch)?;
                let e = else_branch.as_ref().map(|eb| self.lower_expr(eb)).transpose()?;
                let ty = if let Some(ref eb) = e {
                    eb.ty.clone()
                } else {
                    Type::unit()
                };
                (HirExprKind::If { cond: Box::new(c), then_branch: Box::new(t), else_branch: e.map(Box::new), ty: ty.clone() }, ty)
            }

            ExprKind::Block(stmts) => {
                let inner_scope = Scope::with_parent(std::mem::replace(&mut self.scope, Scope::new()));
                self.scope = inner_scope;

                let lowered_stmts: Vec<HirStmt> = stmts.iter()
                    .map(|s| self.lower_stmt(s))
                    .collect::<TenthResult<_>>()?;

                let final_expr = lowered_stmts.last().and_then(|s| match &s.kind {
                    HirStmtKind::Expr(e) => Some(e.clone()),
                    _ => None,
                });

                let ty = final_expr.as_ref().map(|e| e.ty.clone()).unwrap_or(Type::unit());

                let stmts_without_final: Vec<HirStmt> = if final_expr.is_some() {
                    lowered_stmts[..lowered_stmts.len().saturating_sub(1)].to_vec()
                } else {
                    lowered_stmts
                };

                let outer_scope = std::mem::replace(&mut self.scope, Scope::new());
                self.scope = outer_scope;

                (HirExprKind::Block { stmts: stmts_without_final, final_expr: final_expr.map(Box::new) }, ty)
            }

            ExprKind::Closure { params, body } => {
                let lowered_params: Vec<_> = params.iter()
                    .map(|(name, ann)| {
                        let ty = ann.as_ref()
                            .map(|a| Type::from_annotation(a))
                            .unwrap_or(Type::Unknown);
                        (name.name.clone(), ty)
                    })
                    .collect();

                // Create closure scope with params bound
                let closure_scope = Scope::with_parent(std::mem::replace(&mut self.scope, Scope::new()));
                self.scope = closure_scope;
                for (name, ty) in &lowered_params {
                    self.scope.define_var(name.clone(), ty.clone(), false);
                }

                let b = self.lower_expr(body)?;

                let outer_scope = std::mem::replace(&mut self.scope, Scope::new());
                self.scope = outer_scope;

                // Analyze free variables in the closure body (excluding params)
                let captures = Self::free_vars_in(&b);

                let closure_ty = Type::FnType {
                    params: lowered_params.iter().map(|(_, t)| t.clone()).collect(),
                    ret: Box::new(b.ty.clone()),
                };
                (HirExprKind::Closure { params: lowered_params, body: Box::new(b), captures }, closure_ty)
            }

            ExprKind::Assign { target, value } => {
                let v = self.lower_expr(value)?;
                match &target.kind {
                    ExprKind::Ident(id) => {
                        let name = id.name.clone();
                        self.scope.define_var(name.clone(), v.ty.clone(), true);
                        (HirExprKind::Assign { target: name, value: Box::new(v) }, Type::unit())
                    }
                    ExprKind::Deref(inner) => {
                        let inner_hir = self.lower_expr(inner)?;
                        (HirExprKind::DerefAssign { target: Box::new(inner_hir), value: Box::new(v) }, Type::unit())
                    }
                    ExprKind::Field { target: field_target, field } => {
                        let inner_hir = self.lower_expr(field_target)?;
                        (HirExprKind::FieldAssign {
                            target: Box::new(inner_hir),
                            field: field.name.clone(),
                            value: Box::new(v),
                        }, Type::unit())
                    }
                    _ => {
                        return Err(TenthError::ParseError {
                            line: span.line,
                            col: span.col,
                            message: "invalid assignment target".into(),
                        });
                    }
                }
            }

            ExprKind::AssignOp { target, op, value } => {
                let v = self.lower_expr(value)?;
                match &target.kind {
                    ExprKind::Ident(id) => {
                        let name = id.name.clone();
                        let hir_op = lower_binop(op);
                        (HirExprKind::AssignOp { target: name, op: hir_op, value: Box::new(v) }, Type::unit())
                    }
                    ExprKind::Deref(inner) => {
                        let inner_hir = self.lower_expr(inner)?;
                        let hir_op = lower_binop(op);
                        (HirExprKind::DerefAssignOp { target: Box::new(inner_hir), op: hir_op, value: Box::new(v) }, Type::unit())
                    }
                    _ => return Err(TenthError::ParseError {
                        line: span.line,
                        col: span.col,
                        message: "invalid assignment target".into(),
                    }),
                }
            }

            ExprKind::StructLiteral { name, generics: _, fields, use_defaults } => {
                let mut lowered_fields: Vec<(String, HirExpr)> = fields.iter()
                    .map(|(id, e)| {
                        let lowered = self.lower_expr(e)?;
                        Ok((id.name.clone(), lowered))
                    })
                    .collect::<TenthResult<_>>()?;

                if *use_defaults {
                    // Fill missing fields with default values based on type
                    let field_names: Vec<String> = lowered_fields.iter().map(|(n, _)| n.clone()).collect();
                    if let Some(struct_fields) = self.structs.get(&name.name) {
                        for (fname, fty) in struct_fields {
                            if !field_names.contains(fname) {
                                let default_val = match fty {
                                    Type::Base(b) => match b {
                                        BaseType::I32 | BaseType::I64 | BaseType::I8 | BaseType::I16
                                        | BaseType::U8 | BaseType::U16 | BaseType::U32 | BaseType::U64 => HirExpr {
                                            kind: HirExprKind::Literal(Literal::Int(0)),
                                            ty: fty.clone(),
                                            span: name.span.clone(),
                                        },
                                        BaseType::F64 | BaseType::F32 | BaseType::F16 | BaseType::BF16 => HirExpr {
                                            kind: HirExprKind::Literal(Literal::Float(0.0)),
                                            ty: fty.clone(),
                                            span: name.span.clone(),
                                        },
                                        BaseType::Bool => HirExpr {
                                            kind: HirExprKind::Literal(Literal::Bool(false)),
                                            ty: fty.clone(),
                                            span: name.span.clone(),
                                        },
                                        BaseType::Str => HirExpr {
                                            kind: HirExprKind::Literal(Literal::String(String::new())),
                                            ty: fty.clone(),
                                            span: name.span.clone(),
                                        },
                                        BaseType::Char => HirExpr {
                                            kind: HirExprKind::Literal(Literal::Int(0)),
                                            ty: fty.clone(),
                                            span: name.span.clone(),
                                        },
                                        BaseType::Unit => HirExpr {
                                            kind: HirExprKind::Literal(Literal::Int(0)),
                                            ty: Type::unit(),
                                            span: name.span.clone(),
                                        },
                                    },
                                    _ => HirExpr {
                                        kind: HirExprKind::Literal(Literal::Int(0)),
                                        ty: Type::Unknown,
                                        span: name.span.clone(),
                                    },
                                };
                                lowered_fields.push((fname.clone(), default_val));
                            }
                        }
                    }
                }

                let struct_ty = Type::from_annotation(&ast::TypeAnnotation::Named(ast::Ident { name: name.name.clone(), span: name.span.clone() }));
                (HirExprKind::StructLiteral {
                    name: name.name.clone(),
                    fields: lowered_fields,
                    has_default: *use_defaults,
                }, struct_ty)
            }

            ExprKind::EnumLiteral { enum_name, variant, fields } => {
                let lowered_fields: Vec<(String, HirExpr)> = fields.iter()
                    .map(|(id, e)| {
                        let lowered = self.lower_expr(e)?;
                        Ok((id.name.clone(), lowered))
                    })
                    .collect::<TenthResult<_>>()?;
                (HirExprKind::EnumLiteral {
                    enum_name: enum_name.name.clone(),
                    variant: variant.name.clone(),
                    fields: lowered_fields,
                }, Type::Enum(enum_name.name.clone()))
            }

            ExprKind::Match { scrutinee, arms } => {
                let lowered_scrutinee = self.lower_expr(scrutinee)?;
                let lowered_arms: Vec<HirMatchArm> = arms.iter()
                    .map(|arm| {
                        let hir_pattern = self.lower_pattern(&arm.pattern)?;

                        let arm_scope = Scope::with_parent(std::mem::replace(&mut self.scope, Scope::new()));
                        self.scope = arm_scope;

                        // Bind variables from pattern
                        self.bind_pattern_vars(&hir_pattern, &lowered_scrutinee.ty);

                        // Lower guard if present
                        let guard = arm.guard.as_ref()
                            .map(|g| self.lower_expr(g))
                            .transpose()?;

                        let body = self.lower_expr(&arm.body)?;

                        let outer_scope = std::mem::replace(&mut self.scope, Scope::new());
                        self.scope = outer_scope;

                        Ok(HirMatchArm { pattern: hir_pattern, guard, body })
                    })
                    .collect::<TenthResult<_>>()?;
                // Infer match type from first non-Unknown arm, falling back to first arm
                let match_ty = lowered_arms.iter()
                    .map(|arm| arm.body.ty.clone())
                    .find(|ty| !matches!(ty, Type::Unknown))
                    .or_else(|| lowered_arms.first().map(|arm| arm.body.ty.clone()))
                    .unwrap_or(Type::Unknown);
                (HirExprKind::Match {
                    scrutinee: Box::new(lowered_scrutinee),
                    arms: lowered_arms,
                }, match_ty)
            }

            ExprKind::Ref(inner) => {
                if let ExprKind::Ident(ident) = &inner.kind {
                    self.scope.check_borrow_shared(&ident.name, &ident.span)?;
                }
                let e = self.lower_expr(inner)?;
                let ty = Type::Ref(Box::new(e.ty.clone()));
                if let ExprKind::Ident(ident) = &inner.kind {
                    let count = match self.scope.get_ownership(&ident.name) {
                        Some(Ownership::SharedRef(n)) => n + 1,
                        _ => 1,
                    };
                    self.scope.set_ownership(&ident.name, Ownership::SharedRef(count));
                }
                (HirExprKind::Ref(Box::new(e)), ty)
            }

            ExprKind::MutRef(inner) => {
                if let ExprKind::Ident(ident) = &inner.kind {
                    self.scope.check_borrow_mut(&ident.name, &ident.span)?;
                }
                let e = self.lower_expr(inner)?;
                let ty = Type::MutRef(Box::new(e.ty.clone()));
                if let ExprKind::Ident(ident) = &inner.kind {
                    self.scope.set_ownership(&ident.name, Ownership::ExclusiveRef);
                }
                (HirExprKind::MutRef(Box::new(e)), ty)
            }

            ExprKind::Deref(inner) => {
                let e = self.lower_expr(inner)?;
                let inner_ty = match &e.ty {
                    Type::Ref(t) | Type::MutRef(t) => (**t).clone(),
                    _ => Type::Unknown,
                };
                (HirExprKind::Deref(Box::new(e)), inner_ty)
            }

            ExprKind::Move(inner) => {
                let e = self.lower_expr(inner)?;
                let ty = e.ty.clone();
                if let ExprKind::Ident(ident) = &inner.kind {
                    self.scope.set_ownership(&ident.name, Ownership::Moved);
                }
                (HirExprKind::Move(Box::new(e)), ty)
            }

            ExprKind::TryBlock(inner) => {
                let e = self.lower_expr(inner)?;
                let result_ty = Type::Generic {
                    base: Box::new(Type::Enum("Result".to_string())),
                    args: vec![e.ty.clone(), Type::str_()],
                };
                (HirExprKind::TryBlock(Box::new(e)), result_ty)
            }

            ExprKind::InterpolatedString(parts) => {
                let hir_parts: Vec<crate::hir::hir::InterpPart> = parts.iter().map(|p| match p {
                    ast::InterpPart::Literal(s) => crate::hir::hir::InterpPart::Literal(s.clone()),
                    ast::InterpPart::Expr(e) => crate::hir::hir::InterpPart::Expr(e.clone()),
                }).collect();
                (HirExprKind::InterpolatedString { parts: hir_parts }, Type::str_())
            }

            ExprKind::Tuple(elems) => {
                let hir_elems: Vec<HirExpr> = elems.iter().map(|e| self.lower_expr(e)).collect::<Result<_, _>>()?;
                let elem_types: Vec<Type> = hir_elems.iter().map(|e| e.ty.clone()).collect();
                (HirExprKind::Tuple(hir_elems), Type::Tuple(elem_types))
            }
        };

        Ok(HirExpr { kind, ty, span })
    }

    fn lower_index(&mut self, idx: &ast::IndexExpr) -> TenthResult<Index> {
        match idx {
            ast::IndexExpr::Single(e) => Ok(Index::Single(self.lower_expr(e)?)),
            ast::IndexExpr::Range { start, end } => {
                let s = start.as_ref().map(|e| self.lower_expr(e)).transpose()?;
                let e = end.as_ref().map(|e| self.lower_expr(e)).transpose()?;
                Ok(Index::Range { start: s.map(Box::new), end: e.map(Box::new) })
            }
            ast::IndexExpr::Colon => Ok(Index::Colon),
        }
    }

    fn lower_pattern(&mut self, pattern: &ast::Pattern) -> TenthResult<HirPattern> {
        match pattern {
            ast::Pattern::EnumVariant { enum_name, variant, field_bind, tuple_fields } => {
                Ok(HirPattern::EnumVariant {
                    enum_name: enum_name.clone(),
                    variant: variant.clone(),
                    field_bind: field_bind.clone(),
                    tuple_binds: tuple_fields.iter().enumerate()
                        .map(|(i, bind_name)| (format!("_{}", i), bind_name.clone()))
                        .collect(),
                })
            }
            ast::Pattern::Wildcard => Ok(HirPattern::Wildcard),
            ast::Pattern::Literal(lit) => {
                let hir_lit = match lit {
                    ast::Literal::Int(n) => Literal::Int(*n),
                    ast::Literal::Float(n) => Literal::Float(*n),
                    ast::Literal::Bool(b) => Literal::Bool(*b),
                    ast::Literal::String(s) => Literal::String(s.clone()),
                };
                Ok(HirPattern::Literal(hir_lit))
            }
            ast::Pattern::Tuple(patterns) => {
                let hir_patterns: Vec<HirPattern> = patterns.iter()
                    .map(|p| self.lower_pattern(p))
                    .collect::<TenthResult<_>>()?;
                Ok(HirPattern::Tuple(hir_patterns))
            }
            ast::Pattern::Range { start, end, inclusive } => {
                Ok(HirPattern::Range {
                    start: *start,
                    end: *end,
                    inclusive: *inclusive,
                })
            }
            ast::Pattern::Binding(name) => {
                Ok(HirPattern::Binding(name.clone()))
            }
        }
    }

    /// Define variables in scope from a matched pattern.
    fn bind_pattern_vars(&mut self, pattern: &HirPattern, scrutinee_ty: &Type) {
        match pattern {
            HirPattern::EnumVariant { enum_name, variant, field_bind, tuple_binds } => {
                let variant_fields = self.enums.get(enum_name)
                    .and_then(|variants| variants.iter().find(|(v, _)| v == variant))
                    .map(|(_, fields)| fields.clone());

                if let Some((_fname, bname)) = field_bind {
                    let bind_ty = variant_fields.as_ref()
                        .and_then(|f| f.first())
                        .map(|(_, t)| t.clone())
                        .unwrap_or(Type::Unknown);
                    self.scope.define_var(bname.clone(), bind_ty, false);
                }
                for (i, (_, bind_name)) in tuple_binds.iter().enumerate() {
                    let bind_ty = variant_fields.as_ref()
                        .and_then(|f| f.get(i))
                        .map(|(_, t)| t.clone())
                        .unwrap_or(Type::Unknown);
                    self.scope.define_var(bind_name.clone(), bind_ty, false);
                }
            }
            HirPattern::Tuple(patterns) => {
                // Bind each sub-pattern with type from tuple element
                if let Type::Tuple(elem_types) = scrutinee_ty {
                    for (i, sub_pat) in patterns.iter().enumerate() {
                        let elem_ty = elem_types.get(i).cloned().unwrap_or(Type::Unknown);
                        self.bind_pattern_vars(sub_pat, &elem_ty);
                    }
                } else {
                    for sub_pat in patterns {
                        self.bind_pattern_vars(sub_pat, &Type::Unknown);
                    }
                }
            }
            HirPattern::Binding(name) => {
                self.scope.define_var(name.clone(), scrutinee_ty.clone(), false);
            }
            HirPattern::Wildcard | HirPattern::Literal(_) | HirPattern::Range { .. } => {}
        }
    }

    fn index_type(&self, base: &Type, indices: &[Index]) -> Type {
        match base {
            Type::Tensor { dtype, dims } => {
                let num_removed = indices.len();
                let remaining: Vec<Dim> = dims.iter().skip(num_removed).cloned().collect();
                if remaining.is_empty() {
                    Type::Base(dtype.clone())
                } else {
                    Type::Tensor { dtype: dtype.clone(), dims: remaining }
                }
            }
            // For non-tensor types (Vec, etc.), we don't track element types
            _ => Type::Unknown,
        }
    }

    fn infer_binary_type(&self, op: &ast::BinOp, l: &Type, r: &Type) -> Type {
        use ast::BinOp;
        match op {
            BinOp::Eq | BinOp::NotEq | BinOp::Lt | BinOp::Gt | BinOp::LtEq | BinOp::GtEq | BinOp::And | BinOp::Or => {
                Type::bool_()
            }
            BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div | BinOp::Mod => {
                match (l, r) {
                    (Type::Tensor { dtype, .. }, _) | (_, Type::Tensor { dtype, .. }) => {
                        Type::Tensor { dtype: dtype.clone(), dims: vec![Dim::Any] }
                    }
                    // Mixed int/float: promote to float
                    (Type::Base(_), Type::Base(rb)) if matches!(rb, BaseType::F16 | BaseType::F32 | BaseType::F64 | BaseType::BF16) => r.clone(),
                    (Type::Base(lb), Type::Base(_)) if matches!(lb, BaseType::F16 | BaseType::F32 | BaseType::F64 | BaseType::BF16) => l.clone(),
                    _ => l.clone(),
                }
            }
        }
    }

    /// Resolve TypeParam to Struct/Enum if the name matches a known definition.
    fn resolve_struct_type(&self, ty: Type) -> Type {
        match &ty {
            Type::TypeParam { name } => {
                if self.structs.contains_key(name) {
                    Type::Struct(name.clone())
                } else if self.enums.contains_key(name) {
                    Type::Enum(name.clone())
                } else {
                    ty
                }
            }
            _ => ty,
        }
    }

    fn resolve_call_type(&self, func: &HirExpr, args: &[HirExpr], span: &Span) -> TenthResult<Type> {
        match &func.kind {
            HirExprKind::Var(name) => {
                if let Some((params, ret)) = self.scope.lookup_fn(name) {
                    if params.len() != args.len() {
                        let expected: Vec<String> = params.iter()
                            .map(|(n, t)| format!("{}: {}", n, t))
                            .collect();
                        let got: Vec<String> = args.iter()
                            .map(|a| format!("{}", a.ty))
                            .collect();
                        return Err(TenthError::TypeError {
                            line: span.line,
                            col: span.col,
                            message: format!(
                                "function '{}' expects {} argument(s) [{}], got {} [{}]",
                                name, params.len(), expected.join(", "), args.len(), got.join(", ")
                            ),
                        });
                    }
                    return Ok(self.resolve_struct_type(ret));
                }
                self.resolve_builtin(name, args, span)
            }
            _ => Ok(Type::Unknown),
        }
    }

    fn resolve_method_type(&self, receiver: &Type, method: &str, _args: &[HirExpr]) -> Type {
        match receiver {
            Type::Tensor { dtype, dims } => {
                match method {
                    "sum" => {
                        if _args.iter().any(|a| matches!(&a.kind, HirExprKind::Var(_))) {
                            Type::Tensor { dtype: dtype.clone(), dims: dims.clone() }
                        } else {
                            Type::Base(dtype.clone())
                        }
                    }
                    "mean" | "max" | "min" => Type::Base(dtype.clone()),
                    "reshape" | "view" => Type::Tensor { dtype: dtype.clone(), dims: vec![Dim::Any] },
                    "flatten" => Type::Tensor { dtype: dtype.clone(), dims: vec![Dim::Any] },
                    "abs" | "sqrt" | "exp" | "log" | "relu" |
                    "sigmoid" | "tanh" | "softmax" |
                    "transpose" | "permute" => {
                        Type::Tensor { dtype: dtype.clone(), dims: dims.clone() }
                    }
                    "to_vec" => Type::Array(Box::new(Type::Base(dtype.clone()))),
                    "len" | "size" | "dim" => Type::Base(BaseType::I64),
                    "shape" => Type::Array(Box::new(Type::Base(BaseType::I64))),
                    _ => Type::Unknown,
                }
            }
            Type::Base(BaseType::Str) => match method {
                "len" => Type::Base(BaseType::I64),
                "contains" | "starts_with" | "ends_with" => Type::bool_(),
                "trim" | "to_lowercase" | "to_uppercase" => Type::str_(),
                "split" | "lines" => Type::Array(Box::new(Type::str_())),
                "replace" => Type::str_(),
                "parse_int" | "parse_float" => Type::Enum("Option".to_string()),
                "chars" => Type::Array(Box::new(Type::Base(BaseType::Char))),
                _ => Type::Unknown,
            },
            Type::Array(inner) => match method {
                "len" => Type::Base(BaseType::I64),
                "push" => Type::unit(),
                "pop" => Type::Enum("Option".to_string()),
                "get" => Type::Enum("Option".to_string()),
                "map" | "filter" => Type::Array(inner.clone()),
                "is_empty" => Type::bool_(),
                "iter" => Type::Unknown,
                _ => Type::Unknown,
            },
            _ => match method {
                "len" => Type::Base(BaseType::I64),
                "push" => Type::unit(),
                "get" => Type::Unknown,
                _ => Type::Unknown,
            },
        }
    }

    fn resolve_builtin(&self, name: &str, _args: &[HirExpr], _span: &Span) -> TenthResult<Type> {
        match name {
            "println" | "eprintln" => Ok(Type::unit()),
            "tensor" => Ok(Type::Tensor { dtype: BaseType::F64, dims: vec![Dim::Any] }),
            "rand" | "randn" => Ok(Type::Tensor { dtype: BaseType::F64, dims: vec![Dim::Any] }),
            "read_file" => Ok(Type::str_()),
            "str_at" => Ok(Type::str_()),
            "write_file" | "write_bytes" => Ok(Type::unit()),
            "Vec::new" => Ok(Type::Array(Box::new(Type::Unknown))),
            "HashMap::new" => Ok(Type::Unknown),
            "compile_host" => Ok(Type::Base(BaseType::I32)),
            "format" => Ok(Type::str_()),
            "parse_int" => Ok(Type::Enum("Option".to_string())),
            "parse_float" => Ok(Type::Enum("Option".to_string())),
            "abs" | "sqrt" | "sin" | "cos" | "ln" | "pow" => Ok(Type::f64()),
            "zeros" | "ones" => Ok(Type::Tensor { dtype: BaseType::F64, dims: vec![Dim::Any] }),
            "save_weights" | "load_weights" => Ok(Type::unit()),
            "cross_entropy" => Ok(Type::Tensor { dtype: BaseType::F64, dims: vec![Dim::Any] }),
            "start_grad" | "new_grad" | "stop_grad" | "param" => Ok(Type::Tensor { dtype: BaseType::F64, dims: vec![Dim::Any] }),
            "backward" => Ok(Type::unit()),
            "grad" | "zero_grad" => Ok(Type::Unknown),
            "path_join" => Ok(Type::str_()),
            "path_exists" | "path_is_file" | "path_is_dir" => Ok(Type::bool_()),
            "mkdir" => Ok(Type::unit()),
            "list_dir" => Ok(Type::Array(Box::new(Type::str_()))),
            "file_size" => Ok(Type::Base(BaseType::I64)),
            "remove_file" | "copy_file" => Ok(Type::unit()),
            "lexer_new" | "lexer_tokenize" | "parse_program" | "lower_program" | "compile_to_wasm" | "compile_program" => Ok(Type::Unknown),
            _ => Ok(Type::Unknown),
        }
    }

    fn lower_stmt(&mut self, stmt: &ast::Stmt) -> TenthResult<HirStmt> {
        use ast::StmtKind;

        let span = stmt.span.clone();

        let kind = match &stmt.kind {
            StmtKind::Let { names, type_ann, mutable, init } => {
                let lowered_init = init.as_ref().map(|i| self.lower_expr(i)).transpose()?;
                let ty = type_ann.as_ref()
                    .map(|a| Type::from_annotation(a))
                    .or_else(|| lowered_init.as_ref().map(|e| e.ty.clone()))
                    .unwrap_or(Type::Unknown);

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
                let b = self.lower_stmt(body)?;
                HirStmtKind::While { cond: c, body: Box::new(b) }
            }
            StmtKind::For { var, iter, body } => {
                let it = self.lower_expr(iter)?;
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
                let lowered_body: Vec<HirStmt> = body.iter()
                    .map(|s| self.lower_stmt(s))
                    .collect::<TenthResult<_>>()?;
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
                                            "missing implementation of '{}' for trait '{}' on type '{}'",
                                            method.name, trait_name_str, type_name_str
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
                        let mod_name = &path_strs[0];

                        // Try to load the module from file if not already loaded
                        if !self.modules.contains_key(mod_name) && path_strs.len() > 1 {
                            if let Ok(Some(imported_hir)) = self.try_import_file(&path_strs[..path_strs.len()-1]) {
                                self.modules.insert(mod_name.clone(), imported_hir);
                            }
                        }

                        // Also check nested modules
                        let module = if path_strs.len() == 1 {
                            self.modules.get(mod_name)
                        } else {
                            // Navigate nested modules: a::b::c
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
                                // Also add the function definition to the top-level function list
                                // so the interpreter can find and execute it
                                if !self.functions.iter().any(|f| f.name == fn_def.name) {
                                    self.functions.push(fn_def.clone());
                                }
                            }
                        }
                    } else if path_strs.len() >= 2 {
                        // `use path::name` — import a single function
                        let alias = path_strs.last().cloned().unwrap_or_default();
                        self.uses.push((path_strs.clone(), alias.clone()));
                        let mod_name = &path_strs[0];

                        // If the module is not already loaded, try to import it from file
                        if !self.modules.contains_key(mod_name) && !path_strs.is_empty() {
                            if let Ok(Some(imported_hir)) = self.try_import_file(&path_strs[..path_strs.len()-1]) {
                                self.modules.insert(mod_name.clone(), imported_hir);
                            }
                        }

                        // Navigate nested modules for the target function
                        let (module, fn_name) = if path_strs.len() == 2 {
                            (self.modules.get(mod_name), &path_strs[1])
                        } else {
                            // Navigate to the parent module of the target
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
                                // Also add the function definition to the top-level function list
                                if !self.functions.iter().any(|f| f.name == fn_def.name) {
                                    self.functions.push(fn_def.clone());
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

                    let fn_def = HirFnDef {
                        name: name.name.clone(),
                        generics: gen_names,
                        generics_bounds: build_generics_bounds(generics),
                        params: param_types,
                        return_type: ret_ty,
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
        })
    }

    /// Collect free variables referenced in an HIR expression.
    /// A variable is "free" if it is referenced via `Var(name)` but not
    /// bound by an enclosing `Let`, `For`, `Closure`, or `Assign` in the
    /// given expression subtree.  We also exclude built-in names.
    fn free_vars_in(expr: &HirExpr) -> Vec<String> {
        let mut vars = Vec::new();
        Self::collect_free_vars(expr, &mut vars);
        vars.sort();
        vars.dedup();
        vars
    }

    fn collect_free_vars(expr: &HirExpr, vars: &mut Vec<String>) {
        match &expr.kind {
            HirExprKind::Var(name) => {
                // Skip built-in names and qualified paths (e.g. "mod::fn")
                if name.contains("::") { return; }
                match name.as_str() {
                    "println" | "eprintln" | "tensor" | "rand" | "randn"
                    | "read_file" | "write_file" | "str_at" | "Vec::new" | "HashMap::new"
                    | "compile_host" | "compile_program" | "write_bytes"
                    | "start_grad" | "new_grad" | "stop_grad"
                    | "param" | "backward" | "grad" | "zero_grad"
                    | "cross_entropy"
                    | "abs" | "sqrt" | "sin" | "cos" | "ln" | "pow"
                    | "zeros" | "ones"
                    | "save_weights" | "load_weights"
                    | "lexer_new" | "lexer_tokenize" | "parse_program"
                    | "lower_program" | "compile_to_wasm" | "self" => {}
                    _ => { vars.push(name.clone()); }
                }
            }
            HirExprKind::Literal(_) => {}
            HirExprKind::Binary { left, right, .. } => {
                Self::collect_free_vars(left, vars);
                Self::collect_free_vars(right, vars);
            }
            HirExprKind::Unary { expr, .. } => {
                Self::collect_free_vars(expr, vars);
            }
            HirExprKind::Call { func, args, .. } => {
                Self::collect_free_vars(func, vars);
                for a in args { Self::collect_free_vars(a, vars); }
            }
            HirExprKind::GenericCall { func, args, .. } => {
                Self::collect_free_vars(func, vars);
                for a in args { Self::collect_free_vars(a, vars); }
            }
            HirExprKind::MethodCall { receiver, args, .. } => {
                Self::collect_free_vars(receiver, vars);
                for a in args { Self::collect_free_vars(a, vars); }
            }
            HirExprKind::Index { target, indices } => {
                Self::collect_free_vars(target, vars);
                for idx in indices {
                    match idx {
                        crate::hir::hir::Index::Single(e) => {
                            Self::collect_free_vars(e, vars);
                        }
                        crate::hir::hir::Index::Range { start, end } => {
                            if let Some(s) = start { Self::collect_free_vars(s, vars); }
                            if let Some(e) = end { Self::collect_free_vars(e, vars); }
                        }
                        crate::hir::hir::Index::Colon => {}
                    }
                }
            }
            HirExprKind::Field { target, .. } => {
                Self::collect_free_vars(target, vars);
            }
            HirExprKind::TensorLiteral { data, .. } => {
                for row in data {
                    for e in row { Self::collect_free_vars(e, vars); }
                }
            }
            HirExprKind::ArrayLiteral { elements, .. } => {
                for e in elements { Self::collect_free_vars(e, vars); }
            }
            HirExprKind::Range { start, end, .. } => {
                if let Some(s) = start { Self::collect_free_vars(s, vars); }
                if let Some(e) = end { Self::collect_free_vars(e, vars); }
            }
            HirExprKind::If { cond, then_branch, else_branch, .. } => {
                Self::collect_free_vars(cond, vars);
                Self::collect_free_vars(then_branch, vars);
                if let Some(eb) = else_branch { Self::collect_free_vars(eb, vars); }
            }
            HirExprKind::Block { stmts, final_expr } => {
                // Track variables bound within the block
                let mut bound = Vec::new();
                for s in stmts {
                    if let HirStmtKind::Let { names, .. } = &s.kind {
                        for name in names {
                            bound.push(name.clone());
                        }
                    }
                    Self::collect_free_vars_stmt(s, vars);
                }
                if let Some(e) = final_expr { Self::collect_free_vars(e, vars); }
                // Remove variables that were bound in this block
                vars.retain(|v| !bound.contains(v));
            }
            HirExprKind::Closure { params, body, .. } => {
                // Collect all free vars in the body, then remove params
                let mut inner_vars = Vec::new();
                Self::collect_free_vars(body, &mut inner_vars);
                let param_names: Vec<String> = params.iter().map(|(n, _)| n.clone()).collect();
                inner_vars.retain(|v| !param_names.contains(v));
                vars.extend(inner_vars);
            }
            HirExprKind::Assign { target, value } => {
                // target is a variable name that is being written to — it may be
                // a free variable if it comes from an outer scope
                vars.push(target.clone());
                Self::collect_free_vars(value, vars);
            }
            HirExprKind::AssignOp { target, op: _, value } => {
                vars.push(target.clone());
                Self::collect_free_vars(value, vars);
            }
            HirExprKind::StructLiteral { fields, .. } => {
                for (_, e) in fields { Self::collect_free_vars(e, vars); }
            }
            HirExprKind::EnumLiteral { fields, .. } => {
                for (_, e) in fields { Self::collect_free_vars(e, vars); }
            }
            HirExprKind::Match { scrutinee, arms } => {
                Self::collect_free_vars(scrutinee, vars);
                for arm in arms { Self::collect_free_vars(&arm.body, vars); }
            }
            HirExprKind::Ref(inner) | HirExprKind::MutRef(inner) | HirExprKind::Deref(inner) => {
                Self::collect_free_vars(inner, vars);
            }
            HirExprKind::Move(inner) | HirExprKind::TryBlock(inner) => {
                Self::collect_free_vars(inner, vars);
            }
            HirExprKind::InterpolatedString { parts } => {
                for p in parts {
                    if let crate::hir::hir::InterpPart::Expr(name) = p {
                        vars.push(name.clone());
                    }
                }
            }
            HirExprKind::Tuple(elems) => {
                for e in elems {
                    Self::collect_free_vars(e, vars);
                }
            }
            HirExprKind::DerefAssign { target, value } | HirExprKind::DerefAssignOp { target, value, .. } => {
                Self::collect_free_vars(target, vars);
                Self::collect_free_vars(value, vars);
            }
            HirExprKind::FieldAssign { target, value, .. } => {
                Self::collect_free_vars(target, vars);
                Self::collect_free_vars(value, vars);
            }
        }
    }

    fn collect_free_vars_stmt(stmt: &HirStmt, vars: &mut Vec<String>) {
        match &stmt.kind {
            HirStmtKind::Let { init, .. } => {
                if let Some(e) = init { Self::collect_free_vars(e, vars); }
            }
            HirStmtKind::Expr(e) => { Self::collect_free_vars(e, vars); }
            HirStmtKind::Return(e) => {
                if let Some(e) = e { Self::collect_free_vars(e, vars); }
            }
            HirStmtKind::While { cond, body } => {
                Self::collect_free_vars(cond, vars);
                Self::collect_free_vars_stmt(body, vars);
            }
            HirStmtKind::For { var, iter, body } => {
                Self::collect_free_vars(iter, vars);
                let mut inner = Vec::new();
                Self::collect_free_vars_stmt(body, &mut inner);
                inner.retain(|v| v != var);
                vars.extend(inner);
            }
            HirStmtKind::Break | HirStmtKind::Continue => {}
            HirStmtKind::Loop { body } => {
                for s in body { Self::collect_free_vars_stmt(s, vars); }
            }
        }
    }
}

fn substitute_type(ty: &Type, map: &HashMap<String, Type>) -> Type {
    match ty {
        Type::TypeParam { name } => {
            map.get(name).cloned().unwrap_or_else(|| ty.clone())
        }
        Type::Ref(inner) => Type::Ref(Box::new(substitute_type(inner, map))),
        Type::MutRef(inner) => Type::MutRef(Box::new(substitute_type(inner, map))),
        _ => ty.clone(),
    }
}

fn build_generics_bounds(generics: &[ast::GenericParam]) -> HashMap<String, Vec<String>> {
    let mut bounds_map = HashMap::new();
    for gp in generics {
        if !gp.bounds.is_empty() {
            bounds_map.insert(gp.name.name.clone(), gp.bounds.iter().map(|b| b.name.clone()).collect());
        }
    }
    bounds_map
}

fn lower_binop(op: &ast::BinOp) -> BinOp {
    match op {
        ast::BinOp::Add => BinOp::Add, ast::BinOp::Sub => BinOp::Sub,
        ast::BinOp::Mul => BinOp::Mul, ast::BinOp::Div => BinOp::Div,
        ast::BinOp::Mod => BinOp::Mod, ast::BinOp::Eq => BinOp::Eq,
        ast::BinOp::NotEq => BinOp::NotEq, ast::BinOp::Lt => BinOp::Lt,
        ast::BinOp::Gt => BinOp::Gt, ast::BinOp::LtEq => BinOp::LtEq,
        ast::BinOp::GtEq => BinOp::GtEq, ast::BinOp::And => BinOp::And,
        ast::BinOp::Or => BinOp::Or,
    }
}