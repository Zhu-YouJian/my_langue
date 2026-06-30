use std::collections::HashMap;
use std::collections::HashSet;
use crate::parser::ast as ast;
use crate::parser::ast::{ExprKind, StmtKind};
use crate::error::TenthWarning;
use super::hir::*;
use super::types::*;

mod scope;
mod import;
mod lower_expr;
mod lower_stmt;
mod types;
mod closures;

use self::scope::Scope;
use self::scope::Ownership;

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
    /// 编译期收集的警告（内存/算力预估等，非致命）
    pub(super) warnings: Vec<TenthWarning>,
}

impl Lowerer {
    /// Returns true if the statement is a `let` with a direct reference initializer
    /// (e.g., `let r = &x;` or `let m = &mut x;`), which creates a persistent
    /// borrow that should NOT be released after the statement.
    pub(super) fn creates_persistent_borrow(stmt: &ast::Stmt) -> bool {
        match &stmt.kind {
            StmtKind::Let { init: Some(init), .. } => {
                matches!(init.kind, ExprKind::Ref(_) | ExprKind::MutRef(_))
            }
            _ => false
        }
    }

    pub fn new() -> Self {
        let mut scope = Scope::new();
        scope.define_fn(
            "tensor".to_string(),
            vec![("data".to_string(), Type::Unknown)],
            Type::tensor(BaseType::F64, vec![Dim::Any]),
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
            warnings: Vec::new(),
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
}

pub(super) fn substitute_type(ty: &Type, map: &HashMap<String, Type>) -> Type {
    match ty {
        Type::TypeParam { name } => {
            map.get(name).cloned().unwrap_or_else(|| ty.clone())
        }
        Type::Ref(inner) => Type::Ref(Box::new(substitute_type(inner, map))),
        Type::MutRef(inner) => Type::MutRef(Box::new(substitute_type(inner, map))),
        Type::Tensor { dtype, dims } => Type::Tensor {
            dtype: Box::new(substitute_type(dtype, map)),
            dims: dims.clone(),
        },
        Type::Array(inner) => Type::Array(Box::new(substitute_type(inner, map))),
        Type::Generic { base, args } => Type::Generic {
            base: Box::new(substitute_type(base, map)),
            args: args.iter().map(|t| substitute_type(t, map)).collect(),
        },
        _ => ty.clone(),
    }
}

pub(super) fn build_generics_bounds(generics: &[ast::GenericParam]) -> HashMap<String, Vec<String>> {
    let mut bounds_map = HashMap::new();
    for gp in generics {
        if !gp.bounds.is_empty() {
            bounds_map.insert(gp.name.name.clone(), gp.bounds.iter().map(|b| b.name.clone()).collect());
        }
    }
    bounds_map
}

pub(super) fn lower_binop(op: &ast::BinOp) -> BinOp {
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
