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
mod backward_shapes;
mod backward_shape_pass;

use self::scope::Scope;
use self::scope::Ownership;

pub struct Lowerer {
    scope: Scope,
    functions: Vec<HirFnDef>,
    generic_funcs: HashMap<String, HirFnDef>,
    structs: HashMap<String, Vec<(String, Type)>>,
    generic_structs: HashMap<String, HirGenericStruct>,
    unions: HashMap<String, Vec<(String, Type)>>,
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
    // 注意：跨函数 shape 求解已函子化（阶段 0）——不再使用全局可变收集器
    // `current_fn_return_shapes`。函数返回 shape 由 `types.rs` 的
    // `collect_return_tensor_dims` 对已 lower 的 HIR 做纯递归推导
    // （Φ 的构造性定义），组合性由 IR 结构自动保证。
}

impl Lowerer {
    /// Returns true if the statement is a `let` whose initializer may produce
    /// a persistent borrow that should NOT be released after the statement.
    ///
    /// 覆盖直接 `let r = &x;` / `let m = &mut x;`，以及通过控制流产生的
    /// Ref/MutRef 值（AUDIT-11.1.2 / T20 PB2 修复）：
    /// - `let r = if c { &x } else { &y };`
    /// - `let r = { &x };`（Block final stmt 为 Ref）
    /// - `let r = match c { _ => &x };`
    /// Call 节点不在此列——`some_call(&x)` 中的 &x 是临时参数，调用结束后
    /// 借用关系即结束，不应阻止后续语句的 release_borrows。
    pub(super) fn creates_persistent_borrow(stmt: &ast::Stmt) -> bool {
        match &stmt.kind {
            StmtKind::Let { init: Some(init), .. } => {
                Self::expr_may_produce_ref(init)
            }
            _ => false
        }
    }

    /// 递归判断表达式是否可能产生 Ref/MutRef 类型的最终值。
    /// 用于 creates_persistent_borrow 识别 If/Block/Match 中的 Ref 借用。
    fn expr_may_produce_ref(expr: &ast::Expr) -> bool {
        match &expr.kind {
            ExprKind::Ref(_) | ExprKind::MutRef(_) => true,
            ExprKind::If { then_branch, else_branch, .. } => {
                Self::expr_may_produce_ref(then_branch)
                    || else_branch.as_ref().map(|e| Self::expr_may_produce_ref(e)).unwrap_or(false)
            }
            ExprKind::Block(stmts) => {
                // Block 的最终值由最后一个 stmt 决定（若为 Expr stmt）
                stmts.last().map(|s| match &s.kind {
                    StmtKind::Expr(e) => Self::expr_may_produce_ref(e),
                    _ => false,
                }).unwrap_or(false)
            }
            ExprKind::Match { arms, .. } => {
                arms.iter().any(|arm| Self::expr_may_produce_ref(&arm.body))
            }
            // 直接 Ref/MutRef 在外层包装（如 Move(&x)）也视为持久借用源
            ExprKind::Move(inner) => Self::expr_may_produce_ref(inner),
            _ => false,
        }
    }

    /// 收集表达式中所有"作为最终值产出"的 Ref/MutRef 所引用的变量名。
    /// 用于 `let r = if c { &x } else { &y };` 类构造，让 r 同时被记录为
    /// x 和 y 的 borrow holder——任一分支选中都需保留对应借用状态。
    pub(super) fn collect_persistent_borrowed_idents(expr: &ast::Expr) -> Vec<String> {
        let mut out = Vec::new();
        Self::collect_persistent_borrowed_idents_into(expr, &mut out);
        out
    }

    fn collect_persistent_borrowed_idents_into(expr: &ast::Expr, out: &mut Vec<String>) {
        match &expr.kind {
            ExprKind::Ref(inner) | ExprKind::MutRef(inner) => {
                if let ExprKind::Ident(ident) = &inner.kind {
                    out.push(ident.name.clone());
                }
            }
            ExprKind::If { then_branch, else_branch, .. } => {
                Self::collect_persistent_borrowed_idents_into(then_branch, out);
                if let Some(eb) = else_branch {
                    Self::collect_persistent_borrowed_idents_into(eb, out);
                }
            }
            ExprKind::Block(stmts) => {
                if let Some(last) = stmts.last() {
                    if let StmtKind::Expr(e) = &last.kind {
                        Self::collect_persistent_borrowed_idents_into(e, out);
                    }
                }
            }
            ExprKind::Match { arms, .. } => {
                for arm in arms {
                    Self::collect_persistent_borrowed_idents_into(&arm.body, out);
                }
            }
            ExprKind::Move(inner) => {
                Self::collect_persistent_borrowed_idents_into(inner, out);
            }
            _ => {}
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
            unions: HashMap::new(),
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
        // Copy trait: 空 trait，编译器内部识别，自动派生
        lowerer.trait_defs.insert("Copy".to_string(), HirTraitDef {
            name: "Copy".to_string(),
            generics: vec![],
            methods: vec![],
            associated_types: vec![],
        });
        // Drop trait: 析构函数 `fn drop(self)`
        lowerer.trait_defs.insert("Drop".to_string(), HirTraitDef {
            name: "Drop".to_string(),
            generics: vec![],
            methods: vec![HirTraitMethod {
                name: "drop".to_string(),
                params: vec![("self".to_string(), Type::Unknown)],
                return_type: Type::unit(),
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

/// 检查类型是否实现了 Copy trait。
/// 基本类型都是 Copy；结构体如果所有字段都是 Copy 则自动 Copy。
pub(super) fn is_copy_type(ty: &Type, structs: &HashMap<String, Vec<(String, Type)>>,
                           trait_impls: &HashMap<String, HashMap<String, HashMap<String, HirFnDef>>>) -> bool {
    match ty {
        // 基础类型中，BigInt/C64/C128/Decimal 是堆分配字符串，不可 Copy
        Type::Base(b) => !matches!(b, BaseType::BigInt | BaseType::C64 | BaseType::C128 | BaseType::Decimal),
        Type::Never => true,
        Type::Struct(name) => {
            // 检查是否有显式的 impl Copy for StructName
            if let Some(impls) = trait_impls.get("Copy") {
                if impls.contains_key(name) {
                    return true;
                }
            }
            // 自动派生：检查所有字段是否都是 Copy
            if let Some(fields) = structs.get(name) {
                fields.iter().all(|(_, ft)| is_copy_type(ft, structs, trait_impls))
            } else {
                false
            }
        }
        Type::Enum(name) => {
            if let Some(impls) = trait_impls.get("Copy") {
                if impls.contains_key(name) {
                    return true;
                }
            }
            false
        }
        Type::Ref(_, _) | Type::MutRef(_, _) => true, // 引用总是 Copy
        Type::Array { inner, .. } => is_copy_type(inner, structs, trait_impls),
        Type::Tuple(types) => types.iter().all(|t| is_copy_type(t, structs, trait_impls)),
        Type::Dyn(_) => false, // trait 对象不可 Copy
        // HeapBox/Pin: 所有权指针，不可 Copy
        Type::HeapBox(_) | Type::Pin(_) => false,
        // SharedBox/AtomicBox: 共享指针，不可 Copy
        Type::SharedBox(_) | Type::AtomicBox(_) => false,
        _ => false, // Function types, Tensor types, etc. are not Copy by default
    }
}

pub(super) fn substitute_type(ty: &Type, map: &HashMap<String, Type>) -> Type {
    match ty {
        Type::TypeParam { name } => {
            map.get(name).cloned().unwrap_or_else(|| ty.clone())
        }
        Type::Ref(inner, lt) => Type::Ref(Box::new(substitute_type(inner, map)), lt.clone()),
        Type::MutRef(inner, lt) => Type::MutRef(Box::new(substitute_type(inner, map)), lt.clone()),
        Type::Tensor { dtype, dims } => Type::Tensor {
            dtype: Box::new(substitute_type(dtype, map)),
            dims: dims.clone(),
        },
        Type::Array { inner, size } => Type::Array {
            inner: Box::new(substitute_type(inner, map)),
            size: *size,
        },
        Type::Generic { base, args } => Type::Generic {
            base: Box::new(substitute_type(base, map)),
            args: args.iter().map(|t| substitute_type(t, map)).collect(),
        },
        _ => ty.clone(),
    }
}

/// 检查泛型实例化健全性：type_map 必须覆盖所有声明的泛型参数，且每个替换值
/// 不能是 Type::Unknown（类型参数数量不足时 type_map 会用 Unknown 兜底）。
/// AUDIT-11.1.5 / T18 修复。
///
/// 注：Tenth 中用户定义类型（如 `Point`、`Vec`）在 Type 系统中也表示为
/// `Type::TypeParam { name }`（见 types.rs::from_annotation），因此不能
/// 把 TypeParam 视为"未解析类型变量"——只有 Unknown 才是真正的未解析。
pub(super) fn check_generic_instantiation_soundness(
    template_generics: &[String],
    type_map: &HashMap<String, Type>,
) -> Result<(), String> {
    for gen_name in template_generics {
        match type_map.get(gen_name) {
            None => {
                return Err(format!(
                    "泛型参数 '{}' 未被替换（type_map 缺失）",
                    gen_name
                ));
            }
            Some(Type::Unknown) => {
                return Err(format!(
                    "泛型参数 '{}' 被替换为 Unknown（类型参数数量不足）",
                    gen_name
                ));
            }
            Some(_) => {}
        }
    }
    Ok(())
}

/// 递归替换 HirExpr 中所有 Type 字段（包括子表达式的 ty、ret_ty、params 等）。
/// AUDIT-11.1.5 / T18 修复：泛型函数实例化时 body 不能直接 clone，
/// 必须将 body 中所有 TypeParam 替换为 type_map 中的具体类型，
/// 否则实例化后 body 内仍残留类型变量，导致后续类型推断/字节码生成语义偏移。
pub(super) fn substitute_expr(expr: &HirExpr, map: &HashMap<String, Type>) -> HirExpr {
    let mut new_expr = expr.clone();
    substitute_expr_in_place(&mut new_expr, map);
    new_expr
}

fn substitute_expr_in_place(expr: &mut HirExpr, map: &HashMap<String, Type>) {
    expr.ty = substitute_type(&expr.ty, map);
    substitute_kind_in_place(&mut expr.kind, map);
}

fn substitute_kind_in_place(kind: &mut HirExprKind, map: &HashMap<String, Type>) {
    use crate::hir::hir::Index as HirIndex;
    match kind {
        HirExprKind::Literal(_) | HirExprKind::Var(_) | HirExprKind::InterpolatedString { .. } => {}
        HirExprKind::Binary { left, right, ty, .. } => {
            *ty = substitute_type(ty, map);
            substitute_expr_in_place(left, map);
            substitute_expr_in_place(right, map);
        }
        HirExprKind::Unary { expr, ty, .. } => {
            *ty = substitute_type(ty, map);
            substitute_expr_in_place(expr, map);
        }
        HirExprKind::Call { func, args, ret_ty } => {
            *ret_ty = substitute_type(ret_ty, map);
            substitute_expr_in_place(func, map);
            for a in args.iter_mut() {
                substitute_expr_in_place(a, map);
            }
            // AUDIT-11.1.5 / 泛型函数运行时不可用修复：
            // 泛型函数体内 `randn<T>(...)` 等 native 构造在 lower 阶段保留了
            // 原始 func_name（如 "randn"），ret_ty.dtype 保留为 TypeParam。
            // substitute_expr 将 TypeParam 替换为具体 BaseType 后，需根据 dtype
            // 修正 func_name（F32 → randn_f32 / zeros_f32 / ones_f32 / rand_f32，
            // F16 → zeros_f16 / ones_f16，BF16 → zeros_bf16 / ones_bf16），
            // 否则运行时会调用 F64 版本，导致 dtype 语义偏移。
            if let HirExprKind::Var(name) = &func.kind {
                const NATIVE_GENERIC_CTORS: &[&str] = &[
                    "randn", "zeros", "ones", "rand",
                ];
                if NATIVE_GENERIC_CTORS.contains(&name.as_str()) {
                    if let Type::Tensor { dtype, .. } = &ret_ty {
                        if let Type::Base(b) = dtype.as_ref() {
                            let new_name = match (name.as_str(), *b) {
                                ("randn", crate::hir::types::BaseType::F32) => "randn_f32",
                                ("zeros", crate::hir::types::BaseType::F32) => "zeros_f32",
                                ("ones", crate::hir::types::BaseType::F32) => "ones_f32",
                                ("rand", crate::hir::types::BaseType::F32) => "rand_f32",
                                ("zeros", crate::hir::types::BaseType::F16) => "zeros_f16",
                                ("ones", crate::hir::types::BaseType::F16) => "ones_f16",
                                ("zeros", crate::hir::types::BaseType::BF16) => "zeros_bf16",
                                ("ones", crate::hir::types::BaseType::BF16) => "ones_bf16",
                                _ => "",
                            };
                            if !new_name.is_empty() && new_name != *name {
                                func.kind = HirExprKind::Var(new_name.to_string());
                            }
                        }
                    }
                }
            }
        }
        HirExprKind::GenericCall { func, generics, args, ret_ty } => {
            *ret_ty = substitute_type(ret_ty, map);
            for g in generics.iter_mut() {
                *g = substitute_type(g, map);
            }
            substitute_expr_in_place(func, map);
            for a in args.iter_mut() {
                substitute_expr_in_place(a, map);
            }
        }
        HirExprKind::MethodCall { receiver, args, ret_ty, .. } => {
            *ret_ty = substitute_type(ret_ty, map);
            substitute_expr_in_place(receiver, map);
            for a in args.iter_mut() {
                substitute_expr_in_place(a, map);
            }
        }
        HirExprKind::Index { target, indices } => {
            substitute_expr_in_place(target, map);
            for idx in indices.iter_mut() {
                match idx {
                    HirIndex::Single(e) => substitute_expr_in_place(e, map),
                    HirIndex::Range { start, end } => {
                        if let Some(s) = start { substitute_expr_in_place(s, map); }
                        if let Some(e) = end { substitute_expr_in_place(e, map); }
                    }
                    HirIndex::Colon => {}
                }
            }
        }
        HirExprKind::Field { target, .. } => {
            substitute_expr_in_place(target, map);
        }
        HirExprKind::TensorLiteral { data, ty } => {
            *ty = substitute_type(ty, map);
            for row in data.iter_mut() {
                for e in row.iter_mut() {
                    substitute_expr_in_place(e, map);
                }
            }
        }
        HirExprKind::ArrayLiteral { elements, ty } => {
            *ty = substitute_type(ty, map);
            for e in elements.iter_mut() {
                substitute_expr_in_place(e, map);
            }
        }
        HirExprKind::Range { start, end, .. } => {
            if let Some(s) = start { substitute_expr_in_place(s, map); }
            if let Some(e) = end { substitute_expr_in_place(e, map); }
        }
        HirExprKind::If { cond, then_branch, else_branch, ty } => {
            *ty = substitute_type(ty, map);
            substitute_expr_in_place(cond, map);
            substitute_expr_in_place(then_branch, map);
            if let Some(eb) = else_branch { substitute_expr_in_place(eb, map); }
        }
        HirExprKind::Block { stmts, final_expr } => {
            for s in stmts.iter_mut() {
                substitute_stmt_in_place(s, map);
            }
            if let Some(fe) = final_expr { substitute_expr_in_place(fe, map); }
        }
        HirExprKind::Closure { params, body, .. } => {
            for (_, t) in params.iter_mut() {
                *t = substitute_type(t, map);
            }
            substitute_expr_in_place(body, map);
        }
        HirExprKind::Assign { value, .. } => {
            substitute_expr_in_place(value, map);
        }
        HirExprKind::AssignOp { value, .. } => {
            substitute_expr_in_place(value, map);
        }
        HirExprKind::StructLiteral { fields, .. } => {
            for (_, e) in fields.iter_mut() {
                substitute_expr_in_place(e, map);
            }
        }
        HirExprKind::EnumLiteral { fields, .. } => {
            for (_, e) in fields.iter_mut() {
                substitute_expr_in_place(e, map);
            }
        }
        HirExprKind::Match { scrutinee, arms } => {
            substitute_expr_in_place(scrutinee, map);
            for arm in arms.iter_mut() {
                if let Some(g) = arm.guard.as_mut() { substitute_expr_in_place(g, map); }
                substitute_expr_in_place(&mut arm.body, map);
            }
        }
        HirExprKind::Ref(e) | HirExprKind::MutRef(e) | HirExprKind::Deref(e)
        | HirExprKind::Move(e) | HirExprKind::TryBlock(e) | HirExprKind::Await(e)
        | HirExprKind::Spawn(e) => {
            substitute_expr_in_place(e, map);
        }
        HirExprKind::Yield(inner) => {
            if let Some(e) = inner.as_mut() {
                substitute_expr_in_place(e, map);
            }
        }
        HirExprKind::DerefAssign { target, value } => {
            substitute_expr_in_place(target, map);
            substitute_expr_in_place(value, map);
        }
        HirExprKind::DerefAssignOp { target, value, .. } => {
            substitute_expr_in_place(target, map);
            substitute_expr_in_place(value, map);
        }
        HirExprKind::Tuple(elements) => {
            for e in elements.iter_mut() {
                substitute_expr_in_place(e, map);
            }
        }
        HirExprKind::FieldAssign { target, value, .. } => {
            substitute_expr_in_place(target, map);
            substitute_expr_in_place(value, map);
        }
    }
}

fn substitute_stmt_in_place(stmt: &mut HirStmt, map: &HashMap<String, Type>) {
    match &mut stmt.kind {
        HirStmtKind::Let { type_ann, init, .. } => {
            if let Some(t) = type_ann { *t = substitute_type(t, map); }
            if let Some(e) = init { substitute_expr_in_place(e, map); }
        }
        HirStmtKind::Expr(e) => substitute_expr_in_place(e, map),
        HirStmtKind::Return(e) => {
            if let Some(e) = e.as_mut() { substitute_expr_in_place(e, map); }
        }
        HirStmtKind::While { cond, body } => {
            substitute_expr_in_place(cond, map);
            substitute_stmt_in_place(body, map);
        }
        HirStmtKind::DoWhile { body, cond } => {
            substitute_stmt_in_place(body, map);
            substitute_expr_in_place(cond, map);
        }
        HirStmtKind::For { iter, body, .. } => {
            substitute_expr_in_place(iter, map);
            substitute_stmt_in_place(body, map);
        }
        HirStmtKind::Break(_) | HirStmtKind::Continue => {}
        HirStmtKind::Loop { body } => {
            for s in body.iter_mut() {
                substitute_stmt_in_place(s, map);
            }
        }
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
