//! M3.3：声明式宏（最小可行版本）展开 pass。
//!
//! 设计：
//! - 声明：`macro name(param1, param2) { body_expr }` → `ItemKind::MacroDef`
//! - 调用：`name(args)`（与函数调用同形）。`parse_program` 末尾对整棵 AST
//!   做一遍展开 pass：先收集宏定义（并从 AST 移除），再把调用点 AST 替换为
//!   body（参数按名代入），然后继续正常编译——展开后与手写代码等价。
//! - 展开时机：parse 完成后、进入 lower 前。挂在 `parse_program` 末尾，
//!   所有调用方（main / repl / wasm host / import / 测试）零改动获得宏能力。
//! - 嵌套：宏体内可以调用其他宏（含自身），展开时递归进行，深度上限
//!   `MAX_EXPAND_DEPTH` 防止无限递归。
//! - 边界：
//!   - 参数个数不匹配 → 编译期 ParseError
//!   - 重复定义宏 → 编译期 ParseError
//!   - 递归宏（展开后无限增长）→ 深度上限 ParseError
//!   - 未定义名调用照常走函数调用路径（由 lower 报"未定义函数"）
//!   - 宏定义从 AST 移除（编译期构造，不进 HIR）
//! - 不做（文档标注遗留）：hygiene（参数/局部变量同名时按名整体替换，存在
//!   标识符捕获问题）、模式匹配宏（`match` 式规则）、过程宏、
//!   `name!(args)` 调用语法、宏展开进 tenthc（tenthc 语法层不解析宏）。

use super::ast::*;
use crate::error::{TenthError, TenthResult};
use std::collections::HashMap;

/// 最大宏展开深度，防止递归宏（展开后无限增长）死循环。
const MAX_EXPAND_DEPTH: usize = 64;

/// 宏定义表：名字 → (参数名列表, body 模板)
type MacroTable = HashMap<String, (Vec<String>, Expr)>;

/// 收集全部宏定义并从 AST 中移除，再对剩余项做展开。
pub fn expand_program_macros(program: &mut Program) -> TenthResult<()> {
    let mut macros: MacroTable = HashMap::new();
    let mut kept: Vec<Item> = Vec::with_capacity(program.items.len());

    // Pass 1：收集宏定义（宏是编译期构造，不进 HIR，直接从 AST 移除）
    for item in program.items.drain(..) {
        let span = item.span.clone();
        match item.kind {
            ItemKind::MacroDef { name, params, body } => {
                let key = name.name.clone();
                if macros.contains_key(&key) {
                    return Err(TenthError::ParseError {
                        line: span.line,
                        col: span.col,
                        message: format!("宏 '{}' 重复定义", key),
                    });
                }
                let mut seen: Vec<String> = Vec::new();
                for p in &params {
                    if seen.contains(&p.name) {
                        return Err(TenthError::ParseError {
                            line: p.span.line,
                            col: p.span.col,
                            message: format!("宏 '{}' 参数 '{}' 重复", key, p.name),
                        });
                    }
                    seen.push(p.name.clone());
                }
                let param_names: Vec<String> = params.iter().map(|p| p.name.clone()).collect();
                macros.insert(key, (param_names, body));
            }
            other => kept.push(Item { kind: other, span }),
        }
    }

    // Pass 2：对剩余项做展开
    for item in &mut kept {
        expand_item(item, &macros)?;
    }

    program.items = kept;
    Ok(())
}

/// 展开 item 内所有表达式中的宏调用。
fn expand_item(item: &mut Item, macros: &MacroTable) -> TenthResult<()> {
    match &mut item.kind {
        ItemKind::Function { body, .. } => expand_expr(body, macros, 0),
        ItemKind::Const { value, .. } => expand_expr(value, macros, 0),
        ItemKind::Impl { functions, .. } => {
            for f in functions.iter_mut() {
                expand_item(f, macros)?;
            }
            Ok(())
        }
        ItemKind::Mod { items, .. } => {
            for it in items.iter_mut() {
                expand_item(it, macros)?;
            }
            Ok(())
        }
        ItemKind::Trait { methods, .. } => {
            for m in methods.iter_mut() {
                if let Some(b) = &mut m.body {
                    expand_expr(b, macros, 0)?;
                }
            }
            Ok(())
        }
        ItemKind::Operator { func, .. } => expand_item(func.as_mut(), macros),
        _ => Ok(()),
    }
}

/// 递归展开一个表达式中的宏调用（后序：先展开子节点）。
fn expand_expr(expr: &mut Expr, macros: &MacroTable, depth: usize) -> TenthResult<()> {
    if depth > MAX_EXPAND_DEPTH {
        return Err(TenthError::ParseError {
            line: expr.span.line,
            col: expr.span.col,
            message: format!(
                "宏展开超过最大深度 {}（疑似递归宏或展开后无限增长）",
                MAX_EXPAND_DEPTH
            ),
        });
    }

    // 后序：先展开子表达式（嵌套宏调用先展开）
    let mut f = |child: &mut Expr| expand_expr(child, macros, depth);
    for_each_child_expr(expr, &mut f)?;

    // 当前节点若是宏调用则替换为 body（参数代入），并递归展开展开结果
    let macro_name: Option<String> = match &expr.kind {
        ExprKind::Call { func, .. } => match &func.kind {
            ExprKind::Ident(id) => macros.get(&id.name).map(|_| id.name.clone()),
            _ => None,
        },
        _ => None,
    };

    if let Some(name) = macro_name {
        let (params, body) = &macros[&name];
        let (arg_count, arg_exprs) = match &expr.kind {
            ExprKind::Call { args, .. } => (args.len(), args.clone()),
            _ => unreachable!(),
        };
        if arg_count != params.len() {
            return Err(TenthError::ParseError {
                line: expr.span.line,
                col: expr.span.col,
                message: format!(
                    "宏 '{}' 参数个数不匹配：期望 {} 个，实际 {} 个",
                    name,
                    params.len(),
                    arg_count
                ),
            });
        }
        let mut new_body = body.clone();
        subst_params(&mut new_body, params, &arg_exprs);
        *expr = new_body;
        // 展开后的 body 里可能还有宏调用（嵌套），深度递增后递归展开
        return expand_expr(expr, macros, depth + 1);
    }

    Ok(())
}

/// 参数代入：把 body 模板中所有与参数同名的标识符替换为实参 AST。
/// 不做 hygiene（捕获问题文档标注）：body 中与参数同名的局部绑定也会被替换，
/// 这是最小版本的设计取舍（与 C 宏类似）。
fn subst_params(expr: &mut Expr, params: &[String], args: &[Expr]) {
    if let ExprKind::Ident(id) = &expr.kind {
        if let Some(pos) = params.iter().position(|p| *p == id.name) {
            *expr = args[pos].clone();
            return; // 实参 AST 原样保留，不再深入（防止被再次替换）
        }
    }
    let mut f = |child: &mut Expr| {
        subst_params(child, params, args);
        Ok(())
    };
    let _ = for_each_child_expr(expr, &mut f);
}

/// 对表达式的所有直接子表达式调用 `f`（含块内语句的子表达式）。
fn for_each_child_expr(
    expr: &mut Expr,
    f: &mut dyn FnMut(&mut Expr) -> TenthResult<()>,
) -> TenthResult<()> {
    match &mut expr.kind {
        ExprKind::Literal(_)
        | ExprKind::Ident(_)
        | ExprKind::InterpolatedString(_)
        | ExprKind::FString(_) => {}
        ExprKind::Tuple(items) => {
            for it in items.iter_mut() {
                f(it)?;
            }
        }
        ExprKind::Binary { left, right, .. } => {
            f(left)?;
            f(right)?;
        }
        ExprKind::CustomBinary { left, right, .. } => {
            f(left)?;
            f(right)?;
        }
        ExprKind::Unary { expr: e, .. } => {
            f(e)?;
        }
        ExprKind::Call { func, args } => {
            f(func)?;
            for a in args.iter_mut() {
                f(a)?;
            }
        }
        ExprKind::GenericCall { func, args, .. } => {
            f(func)?;
            for a in args.iter_mut() {
                f(a)?;
            }
        }
        ExprKind::MethodCall { receiver, args, .. } => {
            f(receiver)?;
            for a in args.iter_mut() {
                f(a)?;
            }
        }
        ExprKind::Index { target, indices } => {
            f(target)?;
            for idx in indices.iter_mut() {
                match idx {
                    IndexExpr::Single(e) => f(e)?,
                    IndexExpr::Range { start, end } => {
                        if let Some(s) = start {
                            f(s)?;
                        }
                        if let Some(e) = end {
                            f(e)?;
                        }
                    }
                    IndexExpr::Colon => {}
                }
            }
        }
        ExprKind::Field { target, .. } => {
            f(target)?;
        }
        ExprKind::TensorLiteral(rows) => {
            for row in rows.iter_mut() {
                for e in row.iter_mut() {
                    f(e)?;
                }
            }
        }
        ExprKind::ArrayLiteral(items) => {
            for it in items.iter_mut() {
                f(it)?;
            }
        }
        ExprKind::Range { start, end, .. } => {
            if let Some(s) = start {
                f(s)?;
            }
            if let Some(e) = end {
                f(e)?;
            }
        }
        ExprKind::If {
            cond,
            then_branch,
            else_branch,
        } => {
            f(cond)?;
            f(then_branch)?;
            if let Some(eb) = else_branch {
                f(eb)?;
            }
        }
        ExprKind::Block(stmts) => {
            for s in stmts.iter_mut() {
                for_each_child_stmt(s, f)?;
            }
        }
        ExprKind::Closure { body, .. } => {
            f(body)?;
        }
        ExprKind::Assign { target, value } => {
            f(target)?;
            f(value)?;
        }
        ExprKind::AssignOp { target, value, .. } => {
            f(target)?;
            f(value)?;
        }
        ExprKind::StructLiteral { fields, .. } => {
            for (_, e) in fields.iter_mut() {
                f(e)?;
            }
        }
        ExprKind::EnumLiteral { fields, .. } => {
            for (_, e) in fields.iter_mut() {
                f(e)?;
            }
        }
        ExprKind::Match { scrutinee, arms } => {
            f(scrutinee)?;
            for arm in arms.iter_mut() {
                if let Some(g) = &mut arm.guard {
                    f(g)?;
                }
                f(&mut arm.body)?;
            }
        }
        ExprKind::Ref(e)
        | ExprKind::MutRef(e)
        | ExprKind::Deref(e)
        | ExprKind::Move(e)
        | ExprKind::Lossy(e)
        | ExprKind::TryBlock(e)
        | ExprKind::Await(e)
        | ExprKind::Spawn(e) => {
            f(e)?;
        }
        ExprKind::Yield(e) => {
            if let Some(e) = e {
                f(e)?;
            }
        }
        ExprKind::NamedArg { value, .. } => {
            f(value)?;
        }
    }
    Ok(())
}

/// 对语句的所有直接子表达式调用 `f`（含嵌套语句）。
fn for_each_child_stmt(
    stmt: &mut Stmt,
    f: &mut dyn FnMut(&mut Expr) -> TenthResult<()>,
) -> TenthResult<()> {
    match &mut stmt.kind {
        StmtKind::Let { init, .. } => {
            if let Some(e) = init {
                f(e)?;
            }
        }
        StmtKind::Expr(e) => {
            f(e)?;
        }
        StmtKind::Return(e) => {
            if let Some(e) = e {
                f(e)?;
            }
        }
        StmtKind::Break { value, .. } => {
            if let Some(e) = value {
                f(e)?;
            }
        }
        StmtKind::Continue { .. } => {}
        StmtKind::While { cond, body, .. } => {
            f(cond)?;
            for_each_child_stmt(body, f)?;
        }
        StmtKind::DoWhile { body, condition, .. } => {
            for_each_child_stmt(body, f)?;
            f(condition)?;
        }
        StmtKind::For { iter, body, .. } => {
            f(iter)?;
            for_each_child_stmt(body, f)?;
        }
        StmtKind::Loop { body, .. } => {
            for s in body.iter_mut() {
                for_each_child_stmt(s, f)?;
            }
        }
    }
    Ok(())
}
