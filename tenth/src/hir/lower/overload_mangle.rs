use std::collections::{HashMap, HashSet};
use crate::hir::hir::*;
use crate::hir::types::Type;

/// AUDIT-11.4.39：重载函数运行时分派三路径一致性（编译期改写）。
///
/// 背景：`fn g(x: i64, y: i64) -> i64` + `fn g(x: str) -> str` 这类同文件重载，
/// 编译期 `resolve_fn_overload` 已按实参类型选中唯一签名，但运行时分派只按函数名：
/// - VM：`functions: HashMap<String, usize>`（vm/mod.rs `add_fn` 按名 insert），
///   后注册覆盖先注册 → `g(1,2)` 可能命中 `g(str)` chunk 静默返回错误值；
/// - 解释器：`self.functions.iter().find(|f| f.name == name)` 取第一条同名；
/// - JIT 经 VM 分派，同 VM。
/// 即「编译期选中的签名与运行时实际调用签名可能不一致」（静默错值红线）。
///
/// 本 pass 在 lowering 全部完成后对顶层 `HirProgram` 做**确定性编译期改写**：
/// 1. 名字存在 ≥2 个不同签名（去重后）的函数 → 每个签名分配唯一 mangled 名
///    `__ovl_<name>_<idx>`（idx = 去重签名首现序，确定性；`__ovl_` 沿用项目
///    `__` 前缀内部名保留约定，与 `__dyn_*`/`__{Type}_{method}` 一致）；
/// 2. 定义改名：`program.functions` 中对应 `HirFnDef.name` 改为 mangled 名；
/// 3. 调用点改写：`Call`/`GenericCall` 目标为 `Var(name)` 且 name 重载时，按
///    实参类型（复用 `resolve_fn_overload` 的精确匹配→参数数量匹配规则）选中
///    签名 → 改为 `Var(mangled)`；
/// 4. 函数值引用改写：裸 `Var(name)`（`expr.ty` 为 `FnType`，即函数值引用，
///    与局部变量遮蔽/递归闭包自引用区分）且 name 重载时 → 首个签名的 mangled
///    （与类型检查 `lookup_fn` 对多重载取首个签名的语义一致）。
///
/// 三后端（bytecode/VM/JIT/解释器/WASM）都按 `HirFnDef.name` 与调用点函数名
/// 解析，改写后天然一致：运行时零歧义、零额外开销、无运行时类型分发。
///
/// 范围边界（保守，防破坏既有依赖）：
/// - 仅处理顶层 `program.functions` 与其函数体、`program.main_expr`；
/// - `program.modules` 不处理：跨文件模块函数合并时按名 first-wins 去重
///   （lower_stmt.rs），模块内重载不会完整存活到运行时分派集合；且模块限定调用
///   `mod::g(...)` 按**原名**在 `module.functions` 查找
///   （lower_expr.rs `try_resolve_module_qualified`），改写模块内部会破坏该查找；
/// - `generic_funcs` 不处理：泛型函数不注册进 scope.functions，调用走显式类型
///   实参（`g<T>(...)` → 实例化 mangled 名），与普通重载无运行时歧义；
/// - 重复签名（同名同参多定义，scope `define_fn` 去重但 `self.functions` 都保留）
///   的两个定义改同一 mangled 名 → 与既有「同名注册覆盖/取第一条」行为一致，不回归。
pub(super) fn mangle_overloads(program: &mut HirProgram) {
    // ── 1. 收集每个名字的去重签名（参数类型序列） ──
    let mut sigs_by_name: HashMap<String, Vec<Vec<Type>>> = HashMap::new();
    for f in &program.functions {
        let sig: Vec<Type> = f.params.iter().map(|(_, t)| t.clone()).collect();
        let entry = sigs_by_name.entry(f.name.clone()).or_default();
        if !entry.contains(&sig) {
            entry.push(sig);
        }
    }

    // ── 2. 重载名字集合（≥2 个不同签名） ──
    let overloaded: HashSet<String> = sigs_by_name
        .iter()
        .filter(|(_, sigs)| sigs.len() >= 2)
        .map(|(n, _)| n.clone())
        .collect();
    if overloaded.is_empty() {
        return;
    }

    // ── 3. 为每个 (name, 去重签名) 分配唯一 mangled 名 ──
    // 收集既有名字集合防碰撞（函数/泛型函数/方法名）。
    let mut used: HashSet<String> = program.functions.iter().map(|f| f.name.clone()).collect();
    for f in &program.generic_funcs {
        used.insert(f.name.clone());
    }
    for methods in program.methods.values() {
        for def in methods.values() {
            used.insert(def.name.clone());
        }
    }
    // name → Vec<(签名, mangled)>，顺序 = 去重签名首现序
    let mut mangle_table: HashMap<String, Vec<(Vec<Type>, String)>> = HashMap::new();
    for (name, sigs) in &sigs_by_name {
        if !overloaded.contains(name) {
            continue;
        }
        let mut table = Vec::new();
        for (idx, sig) in sigs.iter().enumerate() {
            let base = format!("__ovl_{}_{}", name, idx);
            let mut candidate = base.clone();
            while used.contains(&candidate) {
                candidate.push('_');
            }
            used.insert(candidate.clone());
            table.push((sig.clone(), candidate));
        }
        mangle_table.insert(name.clone(), table);
    }

    // ── 4. 定义改名 ──
    for f in &mut program.functions {
        if let Some(table) = mangle_table.get(&f.name) {
            let sig: Vec<Type> = f.params.iter().map(|(_, t)| t.clone()).collect();
            if let Some((_, m)) = table.iter().find(|(s, _)| *s == sig) {
                f.name = m.clone();
            }
        }
    }

    // ── 5. 函数体 + main_expr 调用点/函数值引用改写 ──
    for f in &mut program.functions {
        mangle_expr_in_place(&mut f.body, &mangle_table);
    }
    if let Some(me) = &mut program.main_expr {
        mangle_expr_in_place(me, &mangle_table);
    }
}

/// 按实参类型在重载签名表中选中唯一签名（与 scope.rs `resolve_fn_overload`
/// 的匹配规则一致：精确匹配 → 参数数量匹配）。无法确定时返回 None（保持原名，
/// 对合法程序不应发生——编译期已解析并校验过）。
fn resolve_call(table: &[(Vec<Type>, String)], args: &[HirExpr]) -> Option<String> {
    let arg_types: Vec<Type> = args.iter().map(|a| a.ty.clone()).collect();
    let n = arg_types.len();
    // 精确匹配：参数数量 + 类型全等
    let exact: Vec<&(Vec<Type>, String)> = table
        .iter()
        .filter(|(s, _)| {
            s.len() == n && s.iter().zip(arg_types.iter()).all(|(a, b)| a == b)
        })
        .collect();
    if exact.len() == 1 {
        return Some(exact[0].1.clone());
    }
    // 兼容回退：仅参数数量匹配
    let compat: Vec<&(Vec<Type>, String)> = table.iter().filter(|(s, _)| s.len() == n).collect();
    if compat.len() == 1 {
        return Some(compat[0].1.clone());
    }
    None
}

/// 递归改写表达式中的重载调用点与函数值引用。
fn mangle_expr_in_place(expr: &mut HirExpr, table: &HashMap<String, Vec<(Vec<Type>, String)>>) {
    match &mut expr.kind {
        HirExprKind::Call { func, args, .. } => {
            if let HirExprKind::Var(name) = &func.kind {
                if let Some(t) = table.get(name) {
                    if let Some(m) = resolve_call(t, args) {
                        func.kind = HirExprKind::Var(m);
                    }
                }
            }
            mangle_expr_in_place(func, table);
            for a in args.iter_mut() {
                mangle_expr_in_place(a, table);
            }
        }
        HirExprKind::GenericCall { func, args, .. } => {
            if let HirExprKind::Var(name) = &func.kind {
                if let Some(t) = table.get(name) {
                    if let Some(m) = resolve_call(t, args) {
                        func.kind = HirExprKind::Var(m);
                    }
                }
            }
            mangle_expr_in_place(func, table);
            for a in args.iter_mut() {
                mangle_expr_in_place(a, table);
            }
        }
        HirExprKind::Var(name) => {
            // 仅函数值引用（ty 为 FnType）改写——区分局部变量遮蔽 / 递归闭包
            // 自引用（ty 为 Unknown 或变量类型）。多重载的裸函数值引用与类型
            // 检查 `lookup_fn` 一致取首个签名。
            if matches!(expr.ty, Type::FnType { .. }) {
                if let Some(t) = table.get(name) {
                    if let Some((_, m)) = t.first() {
                        *name = m.clone();
                    }
                }
            }
        }
        HirExprKind::Binary { left, right, .. } => {
            mangle_expr_in_place(left, table);
            mangle_expr_in_place(right, table);
        }
        HirExprKind::Unary { expr: inner, .. } => {
            mangle_expr_in_place(inner, table);
        }
        HirExprKind::MethodCall { receiver, args, .. } => {
            mangle_expr_in_place(receiver, table);
            for a in args.iter_mut() {
                mangle_expr_in_place(a, table);
            }
        }
        HirExprKind::Index { target, indices } => {
            mangle_expr_in_place(target, table);
            for idx in indices.iter_mut() {
                match idx {
                    Index::Single(e) => mangle_expr_in_place(e, table),
                    Index::Range { start, end } => {
                        if let Some(s) = start {
                            mangle_expr_in_place(s, table);
                        }
                        if let Some(e) = end {
                            mangle_expr_in_place(e, table);
                        }
                    }
                    Index::Colon => {}
                }
            }
        }
        HirExprKind::Field { target, .. } => {
            mangle_expr_in_place(target, table);
        }
        HirExprKind::TensorLiteral { data, .. } => {
            for row in data.iter_mut() {
                for e in row.iter_mut() {
                    mangle_expr_in_place(e, table);
                }
            }
        }
        HirExprKind::ArrayLiteral { elements, .. } => {
            for e in elements.iter_mut() {
                mangle_expr_in_place(e, table);
            }
        }
        HirExprKind::Range { start, end, .. } => {
            if let Some(s) = start {
                mangle_expr_in_place(s, table);
            }
            if let Some(e) = end {
                mangle_expr_in_place(e, table);
            }
        }
        HirExprKind::If { cond, then_branch, else_branch, .. } => {
            mangle_expr_in_place(cond, table);
            mangle_expr_in_place(then_branch, table);
            if let Some(eb) = else_branch {
                mangle_expr_in_place(eb, table);
            }
        }
        HirExprKind::Block { stmts, final_expr } => {
            for s in stmts.iter_mut() {
                mangle_stmt_in_place(s, table);
            }
            if let Some(fe) = final_expr {
                mangle_expr_in_place(fe, table);
            }
        }
        HirExprKind::Closure { body, .. } => {
            mangle_expr_in_place(body, table);
        }
        HirExprKind::Assign { value, .. } => {
            mangle_expr_in_place(value, table);
        }
        HirExprKind::AssignOp { value, .. } => {
            mangle_expr_in_place(value, table);
        }
        HirExprKind::StructLiteral { fields, .. } => {
            for (_, e) in fields.iter_mut() {
                mangle_expr_in_place(e, table);
            }
        }
        HirExprKind::UnionLiteral { value, .. } => {
            mangle_expr_in_place(value, table);
        }
        HirExprKind::EnumLiteral { fields, .. } => {
            for (_, e) in fields.iter_mut() {
                mangle_expr_in_place(e, table);
            }
        }
        HirExprKind::Match { scrutinee, arms } => {
            mangle_expr_in_place(scrutinee, table);
            for arm in arms.iter_mut() {
                if let Some(g) = arm.guard.as_mut() {
                    mangle_expr_in_place(g, table);
                }
                mangle_expr_in_place(&mut arm.body, table);
            }
        }
        HirExprKind::Ref(e)
        | HirExprKind::MutRef(e)
        | HirExprKind::Deref(e)
        | HirExprKind::Move(e)
        | HirExprKind::Lossy(e)
        | HirExprKind::TryBlock(e)
        | HirExprKind::Await(e)
        | HirExprKind::Spawn(e) => {
            mangle_expr_in_place(e, table);
        }
        HirExprKind::Yield(inner) => {
            if let Some(e) = inner.as_mut() {
                mangle_expr_in_place(e, table);
            }
        }
        HirExprKind::InterpolatedString { .. } | HirExprKind::Literal(_) => {}
        HirExprKind::Tuple(elements) => {
            for e in elements.iter_mut() {
                mangle_expr_in_place(e, table);
            }
        }
        HirExprKind::FieldAssign { target, value, .. } => {
            mangle_expr_in_place(target, table);
            mangle_expr_in_place(value, table);
        }
        HirExprKind::DerefAssign { target, value } => {
            mangle_expr_in_place(target, table);
            mangle_expr_in_place(value, table);
        }
        HirExprKind::DerefAssignOp { target, value, .. } => {
            mangle_expr_in_place(target, table);
            mangle_expr_in_place(value, table);
        }
    }
}

fn mangle_stmt_in_place(stmt: &mut HirStmt, table: &HashMap<String, Vec<(Vec<Type>, String)>>) {
    match &mut stmt.kind {
        HirStmtKind::Let { init, .. } => {
            if let Some(i) = init {
                mangle_expr_in_place(i, table);
            }
        }
        HirStmtKind::Expr(e) => {
            mangle_expr_in_place(e, table);
        }
        HirStmtKind::Return(Some(e)) => {
            mangle_expr_in_place(e, table);
        }
        HirStmtKind::Return(None) => {}
        HirStmtKind::While { cond, body, .. } => {
            mangle_expr_in_place(cond, table);
            mangle_stmt_in_place(body, table);
        }
        HirStmtKind::DoWhile { body, cond, .. } => {
            mangle_stmt_in_place(body, table);
            mangle_expr_in_place(cond, table);
        }
        HirStmtKind::For { iter, body, .. } => {
            mangle_expr_in_place(iter, table);
            mangle_stmt_in_place(body, table);
        }
        HirStmtKind::Break { value, .. } => {
            if let Some(v) = value {
                mangle_expr_in_place(v, table);
            }
        }
        HirStmtKind::Continue { .. } => {}
        HirStmtKind::Loop { body, .. } => {
            for s in body.iter_mut() {
                mangle_stmt_in_place(s, table);
            }
        }
    }
}
