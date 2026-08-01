use std::collections::HashMap;
use std::collections::HashSet;
use crate::parser::ast as ast;
use crate::parser::ast::{ExprKind, StmtKind};
use crate::error::{TenthError, TenthResult, TenthWarning};
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
/// 层 3 lossy lattice 污点旁路分析（方案 C，M2）：`lossy` 关键字 + 污点传播 + 使用点检查。
mod taint;

use self::scope::Scope;
use self::scope::Ownership;

pub struct Lowerer {
    scope: Scope,
    functions: Vec<HirFnDef>,
    generic_funcs: HashMap<String, HirFnDef>,
    structs: HashMap<String, Vec<(String, Type)>>,
    /// M2.2：Newtype（tuple struct）名集合——`struct Name(Type)` 声明的
    /// 非泛型 tuple struct。构造 `Name(args)` 在 Call 分支按此集合改写为
    /// StructLiteral（字段名 `_0, _1, ...`）。
    tuple_structs: HashSet<String>,
    generic_structs: HashMap<String, HirGenericStruct>,
    unions: HashMap<String, Vec<(String, Type)>>,
    enums: HashMap<String, Vec<(String, Vec<(String, Type)>)>>,
    /// M2.1：泛型枚举（`enum X<T> { .. }`）。
    generic_enums: HashMap<String, HirGenericEnum>,
    methods: HashMap<String, HashMap<String, HirFnDef>>,
    modules: HashMap<String, HirProgram>,
    uses: Vec<(Vec<String>, String)>,
    /// M3.5：程序级顶层 `let` 全局（常量与可变状态）。
    globals: Vec<HirGlobal>,
    trait_defs: HashMap<String, HirTraitDef>,
    trait_impls: HashMap<String, HashMap<String, HashMap<String, HirFnDef>>>,
    /// Directories to search for imported .th files
    search_paths: Vec<String>,
    /// Set of files already imported (to prevent circular imports)
    imported_files: HashSet<String>,
    /// 泛型实例化产生的 mangled 函数名（如 `scale_Tensor[f16, ..]`）。
    /// 层 3 lossy 污点分析跳过这些函数体——泛型实例化是模板的机械展开，
    /// 按「类型不确定（泛型）时不报」的防误报原则，不参与污点判定
    /// （实例化体类型虽已具体化，但标准库大量泛型化，参与会导致大面积新报错）。
    pub(super) generic_instantiations: HashSet<String>,
    /// M2.3：循环标签栈（lower 语句粒度）——`'outer: while/for/loop/do` 压入标签，
    /// 退出循环弹出；用于在 lower 期校验 `break 'x` / `continue 'x` 的标签
    /// 是否指向某个外层循环（未定义标签 / 标签在循环外 → 编译期 TypeError）。
    /// 注意：闭包体 lower 时会清空（闭包是独立函数体，不能跳出外层循环）。
    loop_labels: Vec<Option<String>>,
    /// M3.1：自定义运算符 → 绑定函数名映射（`operator <op> = fn(...)`）。
    /// lower_program 第一遍注册，表达式降级为对绑定函数（合成名
    /// `__custom_op_<op>`）的普通调用。
    pub(super) custom_ops: HashMap<String, String>,
    /// M3.5：模块模式——提取**全部**顶层单名 `let` 为全局。
    /// 模块（use 导入的文件）与 REPL 行使用：模块 main_expr 在导入时不执行，
    /// 顺序无关；导入方需要模块的全部顶层 let 可解析（含未被模块自身函数
    /// 引用的）。顶层文件保持"仅提取被函数引用者"（保护执行顺序）。
    is_module: bool,
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
            // lossy(&x)：同样视为持久借用源（lossy 是编译期包装，借用语义不变）
            ExprKind::Lossy(inner) => Self::expr_may_produce_ref(inner),
            _ => false,
        }
    }

    /// 类型注解 → Type（M1.1 智能指针入口）。
    ///
    /// 在 `Type::from_annotation` 基础上叠加「has_struct 防护」：`Box<T>`/
    /// `Rc<T>`/`Arc<T>`/`Pin<T>`/`Weak<T>` 默认映射为内置容器类型，但若用户
    /// 恰好声明了同名 struct/enum/union/泛型 struct（如 `struct Box<T> { .. }`），
    /// 则回退为 `Type::Generic`（用户类型优先，防误报）。所有处理用户类型注解的
    /// 调用点应使用本方法而非裸 `Type::from_annotation`。
    pub(super) fn annotation_type(&self, ann: &ast::TypeAnnotation) -> Type {
        let ty = Type::from_annotation(ann);
        match ty {
            Type::HeapBox(inner) if self.user_type_declared("Box") => Type::Generic {
                base: Box::new(Type::TypeParam { name: "Box".to_string() }),
                args: vec![*inner],
            },
            Type::SharedBox(inner) if self.user_type_declared("Rc") => Type::Generic {
                base: Box::new(Type::TypeParam { name: "Rc".to_string() }),
                args: vec![*inner],
            },
            Type::AtomicBox(inner) if self.user_type_declared("Arc") => Type::Generic {
                base: Box::new(Type::TypeParam { name: "Arc".to_string() }),
                args: vec![*inner],
            },
            Type::Pin(inner) if self.user_type_declared("Pin") => Type::Generic {
                base: Box::new(Type::TypeParam { name: "Pin".to_string() }),
                args: vec![*inner],
            },
            Type::Weak(inner) if self.user_type_declared("Weak") => Type::Generic {
                base: Box::new(Type::TypeParam { name: "Weak".to_string() }),
                args: vec![*inner],
            },
            _ => ty,
        }
    }

    /// 用户是否声明了名为 name 的 struct/enum/union/泛型 struct。
    /// 用于 has_struct 防护：内置智能指针名与用户类型名冲突时用户类型优先。
    fn user_type_declared(&self, name: &str) -> bool {
        self.structs.contains_key(name)
            || self.enums.contains_key(name)
            || self.unions.contains_key(name)
            || self.generic_structs.contains_key(name)
            || self.generic_enums.contains_key(name)
    }

    /// M1.3：把具体值表达式改写为 `into_dyn(value, "TraitName")` 调用。
    /// 生成的 HirExpr 类型为 Type::Dyn(trait_name)，运行时由 into_dyn native
    /// 包装为 Value::Dyn。
    pub(super) fn make_into_dyn_call(&self, value: HirExpr, trait_name: &str, span: &crate::lexer::token::Span) -> HirExpr {
        let trait_name_owned = trait_name.to_string();
        let dyn_ty = Type::Dyn(trait_name_owned.clone());
        HirExpr {
            kind: HirExprKind::Call {
                func: Box::new(HirExpr {
                    kind: HirExprKind::Var("into_dyn".to_string()),
                    ty: Type::Unknown,
                    span: span.clone(),
                }),
                args: vec![
                    value,
                    HirExpr {
                        kind: HirExprKind::Literal(Literal::String(trait_name_owned.clone())),
                        ty: Type::str_(),
                        span: span.clone(),
                    },
                ],
                ret_ty: dyn_ty.clone(),
            },
            ty: dyn_ty,
            span: span.clone(),
        }
    }

    /// M1.3：dyn 升级编译期类型检查（防误报底线）。
    /// 仅在「明确知道具体类型 + 明确 trait 名」时检查 trait_impls；
    /// Unknown / 未声明的 TypeParam（真泛型参数）/ 其他 一律保守放行。
    pub(super) fn check_dyn_upgrade(&self, trait_name: &str, ty: &Type, span: &crate::lexer::token::Span) -> TenthResult<()> {
        let type_name = match ty {
            Type::Struct(name) | Type::Enum(name) | Type::Union(name) => name.clone(),
            Type::Generic { base, .. } => match base.as_ref() {
                Type::Struct(name) | Type::Enum(name) => name.clone(),
                // 泛型 struct 实例（`File<Open>`）：trait impl 以 base 名注册
                Type::TypeParam { name } => name.clone(),
                _ => return Ok(()),
            },
            // struct/enum/union 字面量表达式推断为 TypeParam(name)——
            // 仅当 name 是已声明的用户类型时才做 impl 检查；
            // 未声明的 TypeParam（真泛型参数 T）保守放行（防误报）。
            Type::TypeParam { name } => {
                if self.structs.contains_key(name) || self.enums.contains_key(name) || self.unions.contains_key(name)
                    || self.generic_structs.contains_key(name) || self.generic_enums.contains_key(name) {
                    name.clone()
                } else {
                    return Ok(());
                }
            }
            // Unknown / 其他：保守放行（防误报）
            _ => return Ok(()),
        };
        let has_impl = self.trait_impls
            .get(trait_name)
            .map_or(false, |impls| impls.contains_key(&type_name));
        if !has_impl {
            return Err(TenthError::TypeError {
                line: span.line,
                col: span.col,
                message: format!(
                    "类型 '{}' 未实现 trait '{}'，无法升级为 dyn {}",
                    type_name, trait_name, trait_name
                ),
            });
        }
        Ok(())
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

    /// M3.5：把已累积的程序级全局注册进 lower 作用域（REPL 跨行可见）。
    /// 只做符号注册（scope + globals 列表），不负责运行时初始化——
    /// 由解释器/VM 在 main 之前统一初始化 `HirProgram.globals`。
    pub fn seed_globals(&mut self, globals: &[HirGlobal]) {
        for g in globals {
            if self.globals.iter().any(|x| x.name == g.name) {
                continue;
            }
            self.scope.define_var(g.name.clone(), g.ty.clone(), g.mutable);
            self.globals.push(g.clone());
        }
    }

    /// M3.5：模块模式——提取全部顶层单名 `let` 为全局。
    /// 用于 use 导入的模块文件与 REPL 行（跨行/跨模块需要全部顶层 let 可解析）。
    pub fn set_module_mode(&mut self) {
        self.is_module = true;
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
            tuple_structs: HashSet::new(),
            generic_structs: HashMap::new(),
            unions: HashMap::new(),
            enums: HashMap::new(),
            generic_enums: HashMap::new(),
            methods: HashMap::new(),
            modules: HashMap::new(),
            uses: Vec::new(),
            globals: Vec::new(),
            trait_defs: HashMap::new(),
            trait_impls: HashMap::new(),
            search_paths: Vec::new(),
            imported_files: HashSet::new(),
            generic_instantiations: HashSet::new(),
            warnings: Vec::new(),
            loop_labels: Vec::new(),
            custom_ops: HashMap::new(),
            is_module: false,
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
            // Copy 与 Drop 互斥（Rust 语义）：实现 Drop 的类型不可 Copy——
            // 否则 `move`/赋值后原变量仍可用，同一资源会被 drop 多次。
            if let Some(impls) = trait_impls.get("Drop") {
                if impls.contains_key(name) {
                    return false;
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
            // Copy 与 Drop 互斥（Rust 语义）
            if let Some(impls) = trait_impls.get("Drop") {
                if impls.contains_key(name) {
                    return false;
                }
            }
            false
        }
        // TypeParam：struct/enum 字面量与注解经 from_annotation 映射为
        // TypeParam("Name")（非 Type::Struct/Enum）。若 Name 是已声明的
        // struct/enum，按同名类型判定 Copy（显式 impl Copy → 是；实现
        // Drop → 否；否则 struct 字段全 Copy → 自动派生）。未声明的真
        // 泛型变量保守视为非 Copy（move 后不可再用，防误放行）。
        Type::TypeParam { name } => {
            if let Some(impls) = trait_impls.get("Copy") {
                if impls.contains_key(name) {
                    return true;
                }
            }
            // Copy 与 Drop 互斥（Rust 语义）
            if let Some(impls) = trait_impls.get("Drop") {
                if impls.contains_key(name) {
                    return false;
                }
            }
            if let Some(fields) = structs.get(name) {
                return fields.iter().all(|(_, ft)| is_copy_type(ft, structs, trait_impls));
            }
            false
        }
        Type::Ref(_, _) | Type::MutRef(_, _) => true, // 引用总是 Copy
        Type::Array { inner, .. } => is_copy_type(inner, structs, trait_impls),
        Type::Tuple(types) => types.iter().all(|t| is_copy_type(t, structs, trait_impls)),
        Type::Dyn(_) => false, // trait 对象不可 Copy
        // HeapBox/Pin: 所有权指针，不可 Copy
        Type::HeapBox(_) | Type::Pin(_) => false,
        // SharedBox/AtomicBox/Weak: 共享/弱引用指针，不可 Copy
        Type::SharedBox(_) | Type::AtomicBox(_) | Type::Weak(_) => false,
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

// ── 阶段2a M2（G2/G3）：用户类型方法表键与 mangled 命名 ──────────────────────
//
// 方法表 `Lowerer.methods` 从「裸类型名」键升级为「类型名 + 状态实参」键：
// - 裸 impl（`impl File`）→ 键 `File`，对所有状态可用（既有行为）
// - 特化 impl（`impl File<Open>`）→ 键 `File<Open>`，仅对应状态可用
// 键由 Type 的叶子名构造，注册端（从 impl generics）与调用端（从 receiver 类型）
// 使用同一套归一化，保证两侧一致（TypeParam("Open") 与 Struct("Open") 都归一为 "Open"）。

/// 用户类型的 base 名：`File` / `File<Open>` → "File"。非用户类型返回 None。
pub(super) fn type_base_name(ty: &Type) -> Option<String> {
    match ty {
        Type::Struct(name) | Type::TypeParam { name } => Some(name.clone()),
        Type::Generic { base, .. } => match base.as_ref() {
            Type::Struct(name) | Type::TypeParam { name } => Some(name.clone()),
            _ => None,
        },
        _ => None,
    }
}

/// 类型实参的叶子名（归一化：`TypeParam("Open")` / `Struct("Open")` / `Enum("Open")` → "Open"）。
pub(super) fn type_leaf_name(t: &Type) -> String {
    match t {
        Type::Struct(name) | Type::TypeParam { name } | Type::Enum(name) | Type::Union(name) => name.clone(),
        Type::Base(b) => format!("{:?}", b),
        other => mangle_type_ident(other),
    }
}

/// 方法表键：`File`（裸名）或 `File<Open>`（带状态实参）。非用户类型返回 None。
pub(super) fn type_method_key(ty: &Type) -> Option<String> {
    match ty {
        Type::Struct(name) | Type::TypeParam { name } => Some(name.clone()),
        Type::Generic { base, args } => {
            let base_name = match base.as_ref() {
                Type::Struct(name) | Type::TypeParam { name } => name.clone(),
                _ => return None,
            };
            let args_str: Vec<String> = args.iter().map(type_leaf_name).collect();
            Some(format!("{}<{}>", base_name, args_str.join(", ")))
        }
        _ => None,
    }
}

/// mangled 前缀：`File` → "File"；`File<Open>` → "File_Open"（非字母数字清洗为 `_`）。
pub(super) fn type_mangle_prefix(ty: &Type) -> Option<String> {
    match ty {
        Type::Struct(name) | Type::TypeParam { name } => Some(name.clone()),
        Type::Generic { base, args } => {
            let base_name = match base.as_ref() {
                Type::Struct(name) | Type::TypeParam { name } => name.clone(),
                _ => return None,
            };
            let args_str: Vec<String> = args.iter().map(type_leaf_name).collect();
            Some(format!("{}_{}", base_name, args_str.join("_")))
        }
        _ => None,
    }
}

/// 类型显示形式清洗为字母数字/下划线（用于 mangled 函数名）。
fn mangle_type_ident(t: &Type) -> String {
    format!("{}", t)
        .chars()
        .map(|c| if c.is_alphanumeric() { c } else { '_' })
        .collect()
}

/// 该类型是否为泛型 struct 实例（base 在 generic_structs 中，即带状态实参的
/// `File<Open>` 这类）。用于 typestate 的「非法状态方法调用 → 编译期报错」判定。
/// Vec/Option/Result 等内置泛型不在 generic_structs 中，不会被误判。
pub(super) fn is_generic_struct_instance(
    ty: &Type,
    generic_structs: &HashMap<String, HirGenericStruct>,
) -> bool {
    match ty {
        Type::Generic { base, .. } => match base.as_ref() {
            Type::Struct(name) | Type::TypeParam { name } => generic_structs.contains_key(name),
            _ => false,
        },
        _ => false,
    }
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
        HirExprKind::Literal(_) | HirExprKind::Var(_) | HirExprKind::InterpolatedString { .. } => {}        HirExprKind::Binary { left, right, ty, .. } => {
            *ty = substitute_type(ty, map);
            substitute_expr_in_place(left, map);
            substitute_expr_in_place(right, map);
        }
        HirExprKind::Unary { expr, ty, .. } => {
            *ty = substitute_type(ty, map);
            substitute_expr_in_place(expr, map);
        }
        HirExprKind::Lossy(inner) => {
            // lossy 是编译期包装：内层表达式的类型仍需替换
            substitute_expr_in_place(inner, map);
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
        HirExprKind::UnionLiteral { value, .. } => {
            substitute_expr_in_place(value, map);
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
        HirStmtKind::While { cond, body, .. } => {
            substitute_expr_in_place(cond, map);
            substitute_stmt_in_place(body, map);
        }
        HirStmtKind::DoWhile { body, cond, .. } => {
            substitute_stmt_in_place(body, map);
            substitute_expr_in_place(cond, map);
        }
        HirStmtKind::For { iter, body, .. } => {
            substitute_expr_in_place(iter, map);
            substitute_stmt_in_place(body, map);
        }
        HirStmtKind::Break { value, .. } => {
            if let Some(e) = value.as_mut() { substitute_expr_in_place(e, map); }
        }
        HirStmtKind::Continue { .. } => {}
        HirStmtKind::Loop { body, .. } => {
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
