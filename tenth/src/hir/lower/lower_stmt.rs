use std::collections::HashMap;
use std::collections::HashSet;
use crate::error::{TenthError, TenthResult, TenthWarning};
use crate::lexer::token::Span;
use crate::parser::ast as ast;
use crate::hir::hir::*;
use crate::hir::types::*;
use super::Scope;
use super::build_generics_bounds;
use super::Lowerer;

impl Lowerer {
    /// 阶段1-静默失败（层1）：检查表达式是否"静默丢弃"了 Result/Option 值。
    ///
    /// 触发条件：表达式类型是 Result/Option（`Type::Generic` base 名为
    /// `Result`/`Option`——注意注解 `Result<i64, str>` 解析出的 base 是
    /// `TypeParam("Result")` 而非 `Type::Enum`，两者都要匹配；或 `Type::Enum`
    /// 同名）且其值被丢弃。
    ///
    /// 不算丢弃（不触发）：
    /// - 被 `?` 消费（`HirExprKind::Unary { op: UnaryOp::Try }`）——即使静态类型
    ///   因 read_line 等注册为 `Type::Enum("Result")` 而非 Generic 而仍是 Result，
    ///   也显式排除（`?` 是显式传播）
    /// - 被 `or_die` / `assume_ok` 消费（这些 native 返回内部值类型，非 Result，
    ///   由类型检查自然排除）
    /// - 被 match 消费（match 的类型是 arm 类型，非 Result，由类型检查自然排除）
    ///
    /// 注意：函数最后一个表达式作为返回值不算丢弃（那是"使用"），由调用方
    /// （Block lowering）保证不把 final_expr 传入本函数。
    pub(super) fn check_silent_failure_discard(&mut self, expr: &HirExpr, span: &Span) {
        // 类型检查：Result/Option（Generic base 名 或 Enum 同名）
        let type_name = match &expr.ty {
            Type::Generic { base, .. } => {
                let name = match base.as_ref() {
                    Type::Enum(name) | Type::TypeParam { name } => name,
                    _ => return,
                };
                if name != "Result" && name != "Option" {
                    return;
                }
                name
            }
            Type::Enum(name) if name == "Result" || name == "Option" => name,
            _ => return,
        };
        // `x?` 显式传播：不算丢弃
        if let HirExprKind::Unary { op: UnaryOp::Try, .. } = &expr.kind {
            return;
        }
        self.warnings.push(TenthWarning::new(
            span.line,
            span.col,
            format!(
                "{} 被忽略，可能静默失败——用 or_die(值, \"消息\") 或 ? 显式处理",
                type_name
            ),
        ));
    }

    /// M2.3：校验 `break 'x` / `continue 'x` 的标签。
    /// - 无标签（普通 break/continue）→ 不校验（沿用既有行为，循环外静默忽略）。
    /// - 带标签但完全不在任何循环内 → 报"标签形式必须位于循环体内"。
    /// - 带标签但没有任何外层循环匹配该标签 → 报"未定义循环标签"。
    fn check_loop_label(&self, label: &Option<String>, span: &Span, kind: &str) -> TenthResult<()> {
        let Some(l) = label else { return Ok(()) };
        if self.loop_labels.is_empty() {
            return Err(TenthError::TypeError {
                line: span.line,
                col: span.col,
                message: format!("{} '{}' 不在任何循环内（标签形式必须位于循环体内）", kind, l),
            });
        }
        if !self.loop_labels.iter().any(|lb| lb.as_ref() == Some(l)) {
            return Err(TenthError::TypeError {
                line: span.line,
                col: span.col,
                message: format!("未定义循环标签 '{}'", l),
            });
        }
        Ok(())
    }

    pub(super) fn lower_stmt(&mut self, stmt: &ast::Stmt) -> TenthResult<HirStmt> {
        use ast::StmtKind;

        let span = stmt.span.clone();

        let kind = match &stmt.kind {
            StmtKind::Let { names, type_ann, mutable, init } => {
                let mut lowered_init = init.as_ref().map(|i| self.lower_expr(i)).transpose()?;
                // 类型注解强制化：若 type_ann 和 init 都存在，检查 shape 兼容性并合并
                let ty = match (type_ann.as_ref(), lowered_init.as_ref()) {
                    (Some(ann), Some(init_expr)) => {
                        let ann_ty = self.annotation_type(ann);
                        Self::check_and_merge_tensor_shape(&ann_ty, &init_expr.ty, &span, "let 注解")?
                    }
                    (Some(ann), None) => self.annotation_type(ann),
                    (None, Some(init_expr)) => init_expr.ty.clone(),
                    (None, None) => Type::Unknown,
                };

                // M1.3：dyn 类型注解驱动的隐式升级——`let d: dyn Draw = rect;`
                // ① 编译期检查：具体类型必须实现该 trait（未实现 → TypeError，防误报）；
                // ② init 改写为 `into_dyn(rect, "Draw")`，运行时包装为 Value::Dyn。
                if let Type::Dyn(trait_name) = &ty {
                    if let Some(init_expr) = lowered_init {
                        self.check_dyn_upgrade(trait_name, &init_expr.ty, &span)?;
                        let call = self.make_into_dyn_call(init_expr, trait_name, &span);
                        lowered_init = Some(call);
                    }
                }

                for name in names {
                    self.scope.define_var(name.name.clone(), ty.clone(), *mutable);
                }

                // 跨语句借用跟踪（AUDIT-11.1.1 / T19 B6 + AUDIT-11.1.2 / T20 PB2 修复）：
                // 若 init 是 `&ident` / `&mut ident` 或包含 Ref/MutRef 的 If/Block/Match
                // （如 `if c { &x } else { &y }`），则本 let 创建的 holder 变量持有
                // 这些 ident 的持久借用。记录到 scope，使后续 release_borrows
                // 在 holder 仍活跃时不释放被借变量的借用状态。
                if let Some(init_expr) = init {
                    let borrowed_idents = Self::collect_persistent_borrowed_idents(init_expr);
                    if !borrowed_idents.is_empty() {
                        for name in names {
                            for borrowed in &borrowed_idents {
                                self.scope.record_borrow_holder(&name.name, borrowed);
                            }
                        }
                    }
                }

                HirStmtKind::Let {
                    names: names.iter().map(|n| n.name.clone()).collect(),
                    type_ann: type_ann.as_ref().map(|a| self.annotation_type(a)),
                    mutable: *mutable,
                    init: lowered_init,
                }
            }
            StmtKind::Expr(e) => {
                HirStmtKind::Expr(self.lower_expr(e)?)
            }
            StmtKind::Return(e) => {
                let lowered = e.as_ref().map(|e| self.lower_expr(e)).transpose()?;
                // 跨函数 shape 求解已函子化（阶段 0）：return 路径 shape 不再在此手工收集，
                // 而是在函数体 lower 完后由 `collect_return_tensor_dims` 对已 lower 的 HIR
                // 做纯递归推导（Φ 的构造性定义，见 types.rs）。
                HirStmtKind::Return(lowered)
            }
            StmtKind::While { label, cond, body } => {
                self.loop_labels.push(label.clone());
                let c = self.lower_expr(cond)?;
                // Release borrows from the condition so the body can reborrow.
                self.scope.release_borrows();
                let b = self.lower_stmt(body)?;
                self.loop_labels.pop();
                HirStmtKind::While { label: label.clone(), cond: c, body: Box::new(b) }
            }
            StmtKind::DoWhile { label, body, condition } => {
                // Lower do-while to loop { body; if !condition { break; } }
                self.loop_labels.push(label.clone());
                let b = self.lower_stmt(body)?;
                let c = self.lower_expr(condition)?;
                let neg_cond = HirExpr {
                    kind: HirExprKind::Unary { op: UnaryOp::Not, expr: Box::new(c), ty: Type::bool_() },
                    ty: Type::bool_(),
                    span: span.clone(),
                };
                let break_stmt = HirStmt {
                    kind: HirStmtKind::Break { label: None, value: None },
                    span: span.clone(),
                };
                let if_break = HirExpr {
                    kind: HirExprKind::If {
                        cond: Box::new(neg_cond),
                        then_branch: Box::new(HirExpr {
                            kind: HirExprKind::Block { stmts: vec![break_stmt], final_expr: None },
                            ty: Type::unit(),
                            span: span.clone(),
                        }),
                        else_branch: None,
                        ty: Type::unit(),
                    },
                    ty: Type::unit(),
                    span: span.clone(),
                };
                self.loop_labels.pop();
                HirStmtKind::Loop {
                    label: label.clone(),
                    body: vec![
                        b,
                        HirStmt {
                            kind: HirStmtKind::Expr(if_break),
                            span: span.clone(),
                        },
                    ],
                }
            }
            StmtKind::For { label, var, iter, body } => {
                self.loop_labels.push(label.clone());
                let it = self.lower_expr(iter)?;
                // Release borrows from the iterator expression so the body can reborrow.
                self.scope.release_borrows();
                // Push loop variable into scope so body can reference it
                let body_scope = Scope::with_parent(std::mem::replace(&mut self.scope, Scope::new()));
                self.scope = body_scope;
                self.scope.define_var(var.name.clone(), Type::Unknown, false);
                let b = self.lower_stmt(body)?;
                self.scope = *self.scope.parent.take().unwrap();
                self.loop_labels.pop();
                HirStmtKind::For { label: label.clone(), var: var.name.clone(), iter: it, body: Box::new(b) }
            }
            StmtKind::Break { label, value } => {
                self.check_loop_label(&label, &span, "break")?;
                HirStmtKind::Break {
                    label: label.clone(),
                    value: value.as_ref().map(|e| Box::new(self.lower_expr(e).unwrap_or_else(|_| {
                        // Fallback: if lowering fails, create a unit expression
                        HirExpr { kind: HirExprKind::Block { stmts: vec![], final_expr: None }, ty: Type::unit(), span: span.clone() }
                    }))),
                }
            }
            StmtKind::Continue { label } => {
                self.check_loop_label(&label, &span, "continue")?;
                HirStmtKind::Continue { label: label.clone() }
            }
            StmtKind::Loop { label, body } => {
                self.loop_labels.push(label.clone());
                let mut lowered_body: Vec<HirStmt> = Vec::new();
                for s in body {
                    let lowered = self.lower_stmt(s)?;
                    // 阶段1-静默失败：loop 体内所有表达式语句的值都会被丢弃
                    // （loop 体没有"末表达式作为返回值"的语义），若产出
                    // Result/Option 则 emit warning。
                    if let HirStmtKind::Expr(e) = &lowered.kind {
                        self.check_silent_failure_discard(e, &lowered.span);
                    }
                    if !Self::creates_persistent_borrow(s) {
                        self.scope.release_borrows();
                    }
                    lowered_body.push(lowered);
                }
                self.loop_labels.pop();
                HirStmtKind::Loop { label: label.clone(), body: lowered_body }
            }

        };

        Ok(HirStmt { kind, span })
    }

    /// M3.5：把被函数引用的顶层 `let` 提升为程序级全局。
    ///
    /// 语义（v2，2026-08-02）：
    /// - **仅**被函数（含方法/trait 默认体）引用的顶层单名 `let` 提升为全局
    ///   （`HirProgram.globals`，带 init），并在顶层作用域注册符号——函数体
    ///   作用域的父链包含顶层作用域，故同文件函数可解析全局名。
    /// - 未被函数引用的顶层 `let`（如 autodiff 测试中与 `new_grad()`/`backward()`
    ///   交错的一次性计算）**保留在 main_expr 原位**，保持执行顺序，不提升。
    /// - 传递引用：若某全局的 init 引用了另一个顶层 let，后者也一并提升
    ///   （直到不动点），保证全局初始化顺序依赖完整。
    /// - 被提升的 let 从 `<expr>` 体移除；运行时由各后端在 main 之前统一初始化。
    /// - 多名绑定（元组解构）始终保持在 main_expr，不提升。
    fn collect_toplevel_globals(&mut self, items: &mut Vec<ast::Item>) -> TenthResult<()> {
        // ① 收集所有函数体中的**自由引用**标识符（排除函数参数与局部绑定名）。
        //    只有自由引用的名字才可能是对顶层 let 全局的引用——否则函数内
        //    同名参数/局部变量（如 hashset 助手函数的参数 `a`/`b`）会造成
        //    误提取：顶层 `let a = ...` 被提升为全局，而 init 依赖 main_expr
        //    局部变量 → 顺序错乱。
        let mut fn_refs: HashSet<String> = HashSet::new();
        for item in items.iter() {
            let mut params: Vec<String> = Vec::new();
            let mut body_expr: Option<&ast::Expr> = None;
            match &item.kind {
                ast::ItemKind::Function { name, params: ps, body, .. } => {
                    if name.name != "<expr>" {
                        params = ps.iter().map(|p| p.name.name.clone()).collect();
                        body_expr = Some(body);
                    }
                }
                ast::ItemKind::Impl { functions, .. } => {
                    for f in functions {
                        if let ast::ItemKind::Function { name, params: ps, body, .. } = &f.kind {
                            let _ = name;
                            let mut r = HashSet::new();
                            let mut b: HashSet<String> = ps.iter().map(|p| p.name.name.clone()).collect();
                            Self::collect_refs_and_bindings(body, &mut r, &mut b);
                            for n in r {
                                if !b.contains(&n) {
                                    fn_refs.insert(n);
                                }
                            }
                        }
                    }
                }
                ast::ItemKind::Trait { methods, .. } => {
                    for m in methods {
                        if let Some(body) = &m.body {
                            let mut r = HashSet::new();
                            let mut b: HashSet<String> =
                                m.params.iter().map(|p| p.name.name.clone()).collect();
                            Self::collect_refs_and_bindings(body, &mut r, &mut b);
                            for n in r {
                                if !b.contains(&n) {
                                    fn_refs.insert(n);
                                }
                            }
                        }
                    }
                }
                _ => {}
            }
            if let Some(body) = body_expr {
                let mut r = HashSet::new();
                let mut b: HashSet<String> = params.into_iter().collect();
                Self::collect_refs_and_bindings(body, &mut r, &mut b);
                for n in r {
                    if !b.contains(&n) {
                        fn_refs.insert(n);
                    }
                }
            }
        }

        // ② 定位 <expr> 体，两阶段标记（含传递引用）待提升的顶层 let
        let mut extract: HashSet<String> = HashSet::new();
        let toplevel_lets: Vec<(String, bool)> = {
            let mut v = Vec::new();
            for item in items.iter() {
                if let ast::ItemKind::Function { name, body, .. } = &item.kind {
                    if name.name != "<expr>" {
                        continue;
                    }
                    if let ast::ExprKind::Block(stmts) = &body.kind {
                        for stmt in stmts {
                            if let ast::StmtKind::Let { names, init, .. } = &stmt.kind {
                                if names.len() == 1 {
                                    v.push((names[0].name.clone(), init.is_some()));
                                }
                            }
                        }
                    }
                    break;
                }
            }
            v
        };
        // 直接引用
        for (name, _) in &toplevel_lets {
            if fn_refs.contains(name) {
                extract.insert(name.clone());
            }
        }
        // M3.5 模块模式：提取全部顶层单名 let（模块 main_expr 导入时不执行，
        // 顺序无关；导入方需要模块的全部顶层 let 可解析）。
        if self.is_module {
            for (name, _) in &toplevel_lets {
                extract.insert(name.clone());
            }
        }
        // 传递引用直到不动点：已提取全局的 init 引用的顶层 let 一并提取。
        // （TOTAL 被函数引用 → 提取；TOTAL 的 init 引用 BASE → BASE 也提取）
        let all_toplevel: HashSet<String> = toplevel_lets.iter().map(|(n, _)| n.clone()).collect();
        loop {
            let mut changed = false;
            for item in items.iter() {
                if let ast::ItemKind::Function { name, body, .. } = &item.kind {
                    if name.name != "<expr>" {
                        continue;
                    }
                    if let ast::ExprKind::Block(stmts) = &body.kind {
                        for stmt in stmts {
                            if let ast::StmtKind::Let { names, init, .. } = &stmt.kind {
                                if names.len() != 1 || !extract.contains(&names[0].name) {
                                    continue;
                                }
                                if let Some(init_expr) = init {
                                    let mut refs = HashSet::new();
                                    Self::collect_ident_names_expr(init_expr, &mut refs);
                                    for r in refs {
                                        if all_toplevel.contains(&r) && !extract.contains(&r) {
                                            extract.insert(r);
                                            changed = true;
                                        }
                                    }
                                }
                            }
                        }
                    }
                    break;
                }
            }
            if !changed {
                break;
            }
        }

        // ③ 遍历 <expr> 体：被标记的 let 提升为全局，其余保留原位
        for item in items.iter_mut() {
            match &mut item.kind {
                ast::ItemKind::Function { name, body, .. } if name.name == "<expr>" => {
                    if let ast::ExprKind::Block(stmts) = &mut body.kind {
                        let mut keep: Vec<ast::Stmt> = Vec::new();
                        let mut gs: Vec<HirGlobal> = Vec::new();
                        for stmt in stmts.drain(..) {
                            let span = stmt.span.clone();
                            match stmt.kind {
                                ast::StmtKind::Let { names, type_ann, mutable, init } => {
                                    if names.len() != 1 || !extract.contains(&names[0].name) {
                                        keep.push(ast::Stmt {
                                            kind: ast::StmtKind::Let { names, type_ann, mutable, init },
                                            span,
                                        });
                                        continue;
                                    }
                                    let name = names[0].name.clone();
                                    let mut lowered_init =
                                        init.as_ref().map(|i| self.lower_expr(i)).transpose()?;
                                    // 类型注解强制化（shape 检查，与 lower_stmt 的 let 一致）
                                    let ty = match (type_ann.as_ref(), lowered_init.as_ref()) {
                                        (Some(ann), Some(init_expr)) => {
                                            let ann_ty = self.annotation_type(ann);
                                            Self::check_and_merge_tensor_shape(
                                                &ann_ty, &init_expr.ty, &span, "顶层 let 注解",
                                            )?
                                        }
                                        (Some(ann), None) => self.annotation_type(ann),
                                        (None, Some(init_expr)) => init_expr.ty.clone(),
                                        (None, None) => Type::Unknown,
                                    };
                                    // M1.3：dyn 类型注解驱动的隐式升级（与 lower_stmt 的 let 一致）
                                    if let Type::Dyn(trait_name) = &ty {
                                        if let Some(init_expr) = lowered_init {
                                            self.check_dyn_upgrade(trait_name, &init_expr.ty, &span)?;
                                            let call = self.make_into_dyn_call(init_expr, trait_name, &span);
                                            lowered_init = Some(call);
                                        }
                                    }
                                    self.scope.define_var(name.clone(), ty.clone(), mutable);
                                    gs.push(HirGlobal { name, ty, mutable, init: lowered_init });
                                }
                                _ => {
                                    keep.push(stmt);
                                }
                            }
                        }
                        *stmts = keep;
                        self.globals.extend(gs);
                    }
                    break;
                }
                _ => {}
            }
        }
        Ok(())
    }

    /// AST 递归收集表达式中的标识符：`refs` = 出现的引用名（含绑定名），
    /// `bindings` = 绑定名（let 名 / 闭包参数 / for 变量 / 模式绑定等）。
    /// 调用方用「refs − bindings」得到自由引用（对顶层 let 全局的潜在引用）。
    fn collect_ident_names_expr(expr: &ast::Expr, refs: &mut HashSet<String>) {
        let mut _b = HashSet::new();
        Self::collect_refs_and_bindings(expr, refs, &mut _b);
    }

    fn collect_refs_and_bindings(expr: &ast::Expr, refs: &mut HashSet<String>, bindings: &mut HashSet<String>) {
        use ast::ExprKind;
        match &expr.kind {
            ExprKind::Literal(_) | ExprKind::InterpolatedString(_) | ExprKind::FString(_) => {}
            ExprKind::Tuple(es) => {
                for e in es {
                    Self::collect_refs_and_bindings(e, refs, bindings);
                }
            }
            ExprKind::Ident(id) => {
                refs.insert(id.name.clone());
            }
            ExprKind::Binary { left, right, .. } => {
                Self::collect_refs_and_bindings(left, refs, bindings);
                Self::collect_refs_and_bindings(right, refs, bindings);
            }
            ExprKind::CustomBinary { left, right, .. } => {
                Self::collect_refs_and_bindings(left, refs, bindings);
                Self::collect_refs_and_bindings(right, refs, bindings);
            }
            ExprKind::Unary { expr: e, .. } => Self::collect_refs_and_bindings(e, refs, bindings),
            ExprKind::Call { func, args } => {
                Self::collect_refs_and_bindings(func, refs, bindings);
                for a in args {
                    Self::collect_refs_and_bindings(a, refs, bindings);
                }
            }
            ExprKind::GenericCall { func, args, .. } => {
                Self::collect_refs_and_bindings(func, refs, bindings);
                for a in args {
                    Self::collect_refs_and_bindings(a, refs, bindings);
                }
            }
            ExprKind::MethodCall { receiver, args, .. } => {
                Self::collect_refs_and_bindings(receiver, refs, bindings);
                for a in args {
                    Self::collect_refs_and_bindings(a, refs, bindings);
                }
            }
            ExprKind::Index { target, indices } => {
                Self::collect_refs_and_bindings(target, refs, bindings);
                for i in indices {
                    if let ast::IndexExpr::Single(e) = i {
                        Self::collect_refs_and_bindings(e, refs, bindings);
                    }
                }
            }
            ExprKind::Field { target, .. } => Self::collect_refs_and_bindings(target, refs, bindings),
            ExprKind::TensorLiteral(rows) => {
                for row in rows {
                    for e in row {
                        Self::collect_refs_and_bindings(e, refs, bindings);
                    }
                }
            }
            ExprKind::ArrayLiteral(es) => {
                for e in es {
                    Self::collect_refs_and_bindings(e, refs, bindings);
                }
            }
            ExprKind::Range { start, end, .. } => {
                if let Some(s) = start {
                    Self::collect_refs_and_bindings(s, refs, bindings);
                }
                if let Some(e) = end {
                    Self::collect_refs_and_bindings(e, refs, bindings);
                }
            }
            ExprKind::If { cond, then_branch, else_branch } => {
                Self::collect_refs_and_bindings(cond, refs, bindings);
                Self::collect_refs_and_bindings(then_branch, refs, bindings);
                if let Some(e) = else_branch {
                    Self::collect_refs_and_bindings(e, refs, bindings);
                }
            }
            ExprKind::Block(stmts) => {
                for s in stmts {
                    Self::collect_refs_and_bindings_stmt(s, refs, bindings);
                }
            }
            ExprKind::Closure { params, body } => {
                for (p, _) in params {
                    bindings.insert(p.name.clone());
                }
                Self::collect_refs_and_bindings(body, refs, bindings);
            }
            ExprKind::Assign { target, value } => {
                Self::collect_refs_and_bindings(target, refs, bindings);
                Self::collect_refs_and_bindings(value, refs, bindings);
            }
            ExprKind::AssignOp { target, value, .. } => {
                Self::collect_refs_and_bindings(target, refs, bindings);
                Self::collect_refs_and_bindings(value, refs, bindings);
            }
            ExprKind::StructLiteral { fields, .. } => {
                for (_, e) in fields {
                    Self::collect_refs_and_bindings(e, refs, bindings);
                }
            }
            ExprKind::EnumLiteral { fields, .. } => {
                for (_, e) in fields {
                    Self::collect_refs_and_bindings(e, refs, bindings);
                }
            }
            ExprKind::Match { scrutinee, arms } => {
                Self::collect_refs_and_bindings(scrutinee, refs, bindings);
                for arm in arms {
                    if let Some(g) = &arm.guard {
                        Self::collect_refs_and_bindings(g, refs, bindings);
                    }
                    Self::collect_pattern_bindings(&arm.pattern, bindings);
                    Self::collect_refs_and_bindings(&arm.body, refs, bindings);
                }
            }
            ExprKind::Ref(e)
            | ExprKind::MutRef(e)
            | ExprKind::Deref(e)
            | ExprKind::Move(e)
            | ExprKind::Lossy(e)
            | ExprKind::TryBlock(e)
            | ExprKind::Await(e)
            | ExprKind::Spawn(e) => Self::collect_refs_and_bindings(e, refs, bindings),
            ExprKind::Yield(Some(e)) => Self::collect_refs_and_bindings(e, refs, bindings),
            ExprKind::Yield(None) => {}
            ExprKind::NamedArg { value, .. } => Self::collect_refs_and_bindings(value, refs, bindings),
        }
    }

    fn collect_refs_and_bindings_stmt(stmt: &ast::Stmt, refs: &mut HashSet<String>, bindings: &mut HashSet<String>) {
        use ast::StmtKind;
        match &stmt.kind {
            StmtKind::Let { names, init, .. } => {
                for n in names {
                    bindings.insert(n.name.clone());
                }
                if let Some(i) = init {
                    Self::collect_refs_and_bindings(i, refs, bindings);
                }
            }
            StmtKind::Expr(e) => Self::collect_refs_and_bindings(e, refs, bindings),
            StmtKind::Return(Some(e)) => Self::collect_refs_and_bindings(e, refs, bindings),
            StmtKind::Return(None) => {}
            StmtKind::Break { value: Some(e), .. } => Self::collect_refs_and_bindings(e, refs, bindings),
            StmtKind::Break { .. } => {}
            StmtKind::Continue { .. } => {}
            StmtKind::While { cond, body, .. } => {
                Self::collect_refs_and_bindings(cond, refs, bindings);
                Self::collect_refs_and_bindings_stmt(body, refs, bindings);
            }
            StmtKind::DoWhile { body, condition, .. } => {
                Self::collect_refs_and_bindings_stmt(body, refs, bindings);
                Self::collect_refs_and_bindings(condition, refs, bindings);
            }
            StmtKind::For { var, iter, body, .. } => {
                bindings.insert(var.name.clone());
                Self::collect_refs_and_bindings(iter, refs, bindings);
                Self::collect_refs_and_bindings_stmt(body, refs, bindings);
            }
            StmtKind::Loop { body, .. } => {
                for s in body {
                    Self::collect_refs_and_bindings_stmt(s, refs, bindings);
                }
            }
        }
    }

    /// 收集 match 模式中的绑定名（Binding / Struct 字段绑定 / EnumVariant 绑定 / Tuple 子模式）。
    fn collect_pattern_bindings(pattern: &ast::Pattern, bindings: &mut HashSet<String>) {
        match pattern {
            ast::Pattern::Binding(name) => {
                bindings.insert(name.clone());
            }
            ast::Pattern::Struct { fields, .. } => {
                for (_, bind_name) in fields {
                    bindings.insert(bind_name.clone());
                }
            }
            ast::Pattern::EnumVariant { field_bind, tuple_fields, .. } => {
                if let Some((_, bind_name)) = field_bind {
                    bindings.insert(bind_name.clone());
                }
                for t in tuple_fields {
                    bindings.insert(t.clone());
                }
            }
            ast::Pattern::Tuple(sub) => {
                for p in sub {
                    Self::collect_pattern_bindings(p, bindings);
                }
            }
            _ => {}
        }
    }

    /// M3.5：把被导入模块的顶层 let 全局合并进当前程序（符号注册 + 全局列表）。
    ///
    /// 常量与可变状态均随 `use` 导入带入；同名冲突时先注册者胜（本地定义优先）。
    /// 运行时由 harness 在 main 之前统一初始化 `self.globals`（含导入的全局）。
    ///
    /// 入参为独立 clone 的全局切片（不借用 self），调用点负责先 clone 模块
    /// 的 globals 再传入，避免与 `self.modules` 的不可变借用冲突。
    fn merge_module_globals(&mut self, globals: &[HirGlobal]) {
        for g in globals {
            if self.globals.iter().any(|x| x.name == g.name) {
                continue;
            }
            self.scope.define_var(g.name.clone(), g.ty.clone(), g.mutable);
            self.globals.push(g.clone());
        }
    }

    pub fn lower_program(&mut self, program: &ast::Program) -> TenthResult<HirProgram> {
        // M3.1：自定义运算符展开。`operator <op> = fn(...)` 在 AST 中表示为
        // ItemKind::Operator，其绑定函数（合成名 `__custom_op_<op>`）在此
        // 展开为普通 Function 项，使后续两遍（注册 + body lowering）统一处理；
        // 同时注册 `op → 函数名` 映射供表达式降级使用。
        let mut items: Vec<ast::Item> = Vec::with_capacity(program.items.len());
        for item in &program.items {
            match &item.kind {
                ast::ItemKind::Operator { op, func } => {
                    let fn_name = match &func.kind {
                        ast::ItemKind::Function { name, .. } => name.name.clone(),
                        _ => {
                            return Err(TenthError::TypeError {
                                line: item.span.line,
                                col: item.span.col,
                                message: format!("运算符 '{}' 绑定必须是函数", op),
                            });
                        }
                    };
                    // 重复声明同一运算符 → 编译期报错（防静默覆盖）。
                    if self.custom_ops.contains_key(op) {
                        return Err(TenthError::TypeError {
                            line: item.span.line,
                            col: item.span.col,
                            message: format!("运算符 '{}' 重复声明", op),
                        });
                    }
                    self.custom_ops.insert(op.clone(), fn_name);
                    items.push(func.as_ref().clone());
                }
                _ => items.push(item.clone()),
            }
        }

        for item in &items {
            match &item.kind {
                ast::ItemKind::StructDef { name, generics, kind, .. } => {
                    let field_types: Vec<(String, Type)> = match kind {
                        ast::StructKind::Named(fields) => fields.iter()
                            .map(|f| (f.name.name.clone(), self.annotation_type(&f.type_ann)))
                            .collect(),
                        ast::StructKind::Tuple(types) => types.iter().enumerate()
                            .map(|(i, ty)| (format!("_{}", i), self.annotation_type(ty)))
                            .collect(),
                    };
                    if generics.is_empty() {
                        self.structs.insert(name.name.clone(), field_types);
                        // M2.2：记录 tuple struct（Newtype）名，供 Call 分支改写构造。
                        if matches!(kind, ast::StructKind::Tuple(_)) {
                            self.tuple_structs.insert(name.name.clone());
                        }
                    } else {
                        let gen_names: Vec<String> = generics.iter().map(|g| g.name.name.clone()).collect();
                        self.generic_structs.insert(name.name.clone(), HirGenericStruct {
                            name: name.name.clone(),
                            generics: gen_names,
                            fields: field_types,
                        });
                    }
                }
                ast::ItemKind::EnumDef { name, generics, variants } => {
                    let variant_list: Vec<(String, Vec<(String, Type)>)> = variants.iter()
                        .map(|v| {
                            let fields: Vec<(String, Type)> = match &v.kind {
                                ast::EnumVariantKind::Unit => Vec::new(),
                                ast::EnumVariantKind::Named(named_fields) => {
                                    named_fields.iter()
                                        .map(|f| (f.name.name.clone(), self.annotation_type(&f.type_ann)))
                                        .collect()
                                }
                                ast::EnumVariantKind::Tuple(types) => {
                                    types.iter().enumerate()
                                        .map(|(i, ty)| (format!("_{}", i), self.annotation_type(ty)))
                                        .collect()
                                }
                            };
                            (v.name.name.clone(), fields)
                        })
                        .collect();
                    if generics.is_empty() {
                        // 无泛型参数：注册为非泛型枚举（向后兼容）
                        // 清除同名泛型枚举条目，保证"后声明覆盖"（shadow 一致）。
                        self.generic_enums.remove(&name.name);
                        self.enums.insert(name.name.clone(), variant_list);
                    } else {
                        // M2.1：泛型枚举 `enum X<T> { .. }` → generic_enums，
                        // 变体字段类型中的 TypeParam("T") 在实例化时替换。
                        // 清除同名内置/非泛型条目（如用户 `enum Option<T>` shadow 内置 Option）。
                        self.enums.remove(&name.name);
                        let gen_names: Vec<String> = generics.iter().map(|g| g.name.name.clone()).collect();
                        self.generic_enums.insert(name.name.clone(), HirGenericEnum {
                            name: name.name.clone(),
                            generics: gen_names,
                            variants: variant_list,
                        });
                    }
                }
                ast::ItemKind::Union { name, fields } => {
                    let field_types: Vec<(String, Type)> = fields.iter()
                        .map(|f| (f.name.name.clone(), self.annotation_type(&f.type_ann)))
                        .collect();
                    self.unions.insert(name.name.clone(), field_types);
                }
                ast::ItemKind::Trait { name, generics, methods, associated_types } => {
                    let gen_names: Vec<String> = generics.iter().map(|g| g.name.name.clone()).collect();
                    let assoc_type_names: Vec<String> = associated_types.iter().map(|t| t.name.clone()).collect();
                    let method_sigs: Vec<HirTraitMethod> = methods.iter()
                        .map(|m| {
                            let param_types: Vec<(String, Type)> = m.params.iter()
                                .map(|p| (p.name.name.clone(), self.annotation_type(&p.type_ann)))
                                .collect();
                            let ret_ty = m.return_type.as_ref()
                                .map(|rt| self.annotation_type(rt))
                                .unwrap_or(Type::unit());
                            // Lower default body if present
                            let default_body = if let Some(body) = &m.body {
                                let body_scope = Scope::with_parent(std::mem::replace(&mut self.scope, Scope::new()));
                                self.scope = body_scope;
                                for (n, t) in &param_types {
                                    self.scope.define_var(n.clone(), t.clone(), false);
                                }
                                let lowered = self.lower_expr(body).ok();
                                // trait 默认方法体作用域结束：弹回父作用域。
                                self.scope = *self.scope.parent.take().unwrap();
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
                        .map(|p| (p.name.name.clone(), self.annotation_type(&p.type_ann)))
                        .collect();
                    let ret_ty = return_type.as_ref()
                        .map(|rt| self.annotation_type(rt))
                        .unwrap_or(Type::unit());
                    let ret_ty = self.resolve_struct_type(ret_ty);
                    // 断点 4.2：记录原始注解返回类型，供前向引用 shape 不动点
                    // pass（run_forward_ref_shape_fixpoint）做注解合并/冲突检查。
                    self.fn_annotations.insert(name.name.clone(), ret_ty.clone());
                    self.scope.define_fn(name.name.clone(), param_types, ret_ty);
                }
                _ => {}
            }
        }

        // M3.5：提取顶层 `let` 为程序级全局（在函数体 lower 之前）。
        // 把 `<expr>` 体中的单名顶层 let 提升为 `self.globals` 并在顶层作用域
        // 注册符号——函数体作用域的父链包含顶层作用域，故同文件函数可解析全局名。
        // 被提取的 let 从 `<expr>` 体移除，运行时由各后端在 main 之前统一初始化。
        self.collect_toplevel_globals(&mut items)?;

        for item in &items {
            match &item.kind {
                ast::ItemKind::Impl { type_name, trait_name, generics, functions } => {
                    if let Some(trait_name) = trait_name {
                        let trait_name_str = trait_name.name.clone();
                        let type_name_str = type_name.name.clone();
                        let mut method_map = HashMap::new();
                        for fn_item in functions {
                            if let ast::ItemKind::Function { name, generics, params, return_type, body, .. } = &fn_item.kind {
                                let gen_names: Vec<String> = generics.iter().map(|g| g.name.name.clone()).collect();
                                let param_types: Vec<(String, Type)> = params.iter()
                                    .map(|p| (p.name.name.clone(), self.annotation_type(&p.type_ann)))
                                    .collect();
                                let ret_ty = return_type.as_ref()
                                    .map(|rt| self.annotation_type(rt))
                                    .unwrap_or(Type::unit());

                                let body_scope = Scope::with_parent(std::mem::replace(&mut self.scope, Scope::new()));
                                self.scope = body_scope;

                                for (n, t) in &param_types {
                                    self.scope.define_var(n.clone(), t.clone(), false);
                                }

                                let lowered_body = self.lower_expr(body)?;

                                // trait impl 方法体作用域结束：弹回父作用域。
                                self.scope = *self.scope.parent.take().unwrap();

                                let fn_def = HirFnDef {
                                    name: name.name.clone(),
                                    generics: gen_names,
                                    generics_bounds: build_generics_bounds(generics),
                                    params: param_types,
                                    param_defaults: Vec::new(),
                                    param_variadic: Vec::new(),
                                    return_type: ret_ty,
                                    body: lowered_body,
                                    span: fn_item.span.clone(),
                                    is_test: false,
                                };
                                // M1.3：dyn 分派函数注册——VM 路径（call_method_priv 的
                                // Value::Dyn 分支）通过 `__dyn_{trait}_{type}_{method}`
                                // 名字调用字节码函数（VM 不能直接执行 HirFnDef）；
                                // 解释器路径仍直接用 trait_impls 表。
                                let mut dyn_fn = fn_def.clone();
                                dyn_fn.name = format!("__dyn_{}_{}_{}", trait_name_str, type_name_str, fn_def.name);
                                self.functions.push(dyn_fn);
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
                        // 阶段2a M2（G2）：impl 按类型实参特化注册。
                        //
                        // `impl File<Open>` 的 `<Open>` 被 parse_generic_params 解析进
                        // `generics`。对 inherent impl，这些是目标类型实参（状态类型），
                        // 不是 impl 泛型参数——Tenth 目前不解析 `impl<T> ...`（`<` 不能
                        // 紧跟 impl 关键字），因此无歧义。用它们构造方法表键 `File<Open>`
                        // 与 mangled 前缀 `File_Open`，使 `impl File<Open>` 与
                        // `impl File<Closed>` 的方法互不覆盖；裸 `impl File` 键为 `File`，
                        // 作为所有状态的通用回退（保持既有行为）。
                        let type_args: Vec<Type> = generics.iter()
                            .map(|g| Type::from_annotation(&ast::TypeAnnotation::Named(g.name.clone())))
                            .collect();
                        let method_key = if type_args.is_empty() {
                            type_name.name.clone()
                        } else {
                            let args_str: Vec<String> = type_args.iter().map(super::type_leaf_name).collect();
                            format!("{}<{}>", type_name.name, args_str.join(", "))
                        };
                        let mangled_prefix = if type_args.is_empty() {
                            type_name.name.clone()
                        } else {
                            let args_str: Vec<String> = type_args.iter().map(super::type_leaf_name).collect();
                            format!("{}_{}", type_name.name, args_str.join("_"))
                        };
                        let mut method_map = HashMap::new();
                        for fn_item in functions {
                            if let ast::ItemKind::Function { name, generics, params, return_type, body, .. } = &fn_item.kind {
                                let gen_names: Vec<String> = generics.iter().map(|g| g.name.name.clone()).collect();
                                let param_types: Vec<(String, Type)> = params.iter()
                                    .map(|p| (p.name.name.clone(), self.annotation_type(&p.type_ann)))
                                    .collect();
                                let ret_ty = return_type.as_ref()
                                    .map(|rt| self.annotation_type(rt))
                                    .unwrap_or(Type::unit());

                                let body_scope = Scope::with_parent(std::mem::replace(&mut self.scope, Scope::new()));
                                self.scope = body_scope;

                                for (n, t) in &param_types {
                                    self.scope.define_var(n.clone(), t.clone(), false);
                                }

                                let lowered_body = self.lower_expr(body)?;

                                // inherent impl 方法体作用域结束：弹回父作用域。
                                self.scope = *self.scope.parent.take().unwrap();

                                let fn_def = HirFnDef {
                                    name: name.name.clone(),
                                    generics: gen_names,
                                    generics_bounds: build_generics_bounds(generics),
                                    params: param_types,
                                    param_defaults: Vec::new(),
                                    param_variadic: Vec::new(),
                                    return_type: ret_ty,
                                    body: lowered_body,
                                    span: fn_item.span.clone(),
                                    is_test: false,
                                };
                                // Also register with mangled name for WASM backend method dispatch
                                let mangled_name = format!("__{}_{}", mangled_prefix, fn_def.name);
                                let mut mangled_fn = fn_def.clone();
                                mangled_fn.name = mangled_name;
                                self.functions.push(mangled_fn);
                                method_map.insert(fn_def.name.clone(), fn_def);
                            }
                        }
                        self.methods.insert(method_key, method_map);
                    }
                }
                ast::ItemKind::Mod { name, items } => {
                    let mut lowerer = Lowerer::new();
                    lowerer.structs = self.structs.clone();
                    lowerer.enums = self.enums.clone();
                    lowerer.generic_enums = self.generic_enums.clone();
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
                            // 泛型函数存于 module.generic_funcs（非 module.functions）
                            for fn_def in &module.generic_funcs {
                                let param_types = fn_def.params.clone();
                                let ret_ty = fn_def.return_type.clone();
                                self.scope.define_fn(fn_def.name.clone(), param_types, ret_ty);
                                self.generic_funcs.insert(fn_def.name.clone(), fn_def.clone());
                            }
                            // M3.5：模块顶层 let 全局随 use 导入合并（常量/状态）
                            let module_globals = module.globals.clone();
                            self.merge_module_globals(&module_globals);
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
                                for fn_def in &module.generic_funcs {
                                    let param_types = fn_def.params.clone();
                                    let ret_ty = fn_def.return_type.clone();
                                    self.scope.define_fn(fn_def.name.clone(), param_types, ret_ty);
                                    self.generic_funcs.insert(fn_def.name.clone(), fn_def.clone());
                                }
                                // M3.5：模块顶层 let 全局随 use 导入合并（常量/状态）
                                let module_globals = module.globals.clone();
                                self.merge_module_globals(&module_globals);
                            }
                        }
                    } else if path_strs.len() >= 2 {
                        // `use path::name` — import a single function
                        let alias = path_strs.last().cloned().unwrap_or_default();
                        self.uses.push((path_strs.clone(), alias.clone()));
                        let fn_name = path_strs.last().unwrap();

                        // AUDIT #11 修复：先尝试完整 path 作为文件
                        // （如 `use std::nn::gelu` → 查找 std/nn/gelu.th），
                        // 失败再回退到 parent_path 作为模块
                        // （如 `use std::nn::activations::gelu` → 查找 std/nn/activations.th）。
                        // 此前只用 parent_path，对目录型模块（nn/ 是目录非 nn.th 文件）
                        // 必然失败，导致 3 段路径 use 报 undefined variable。
                        let parent_path = &path_strs[..path_strs.len()-1];
                        let full_key = path_strs.join("::");
                        let parent_key = parent_path.join("::");
                        let mut loaded_module: Option<&HirProgram> = None;
                        // 先尝试完整 path（如 std/nn/gelu.th）
                        if !self.modules.contains_key(&full_key) {
                            match self.try_import_file(&path_strs) {
                                Ok(Some(imported_hir)) => {
                                    self.modules.insert(full_key.clone(), imported_hir);
                                }
                                Ok(None) => {
                                    // 完整 path 找不到文件，回退到 parent_path
                                    // （如 std/nn/activations.th）
                                    if !self.modules.contains_key(&parent_key) {
                                        match self.try_import_file(parent_path) {
                                            Ok(Some(imported_hir)) => {
                                                self.modules.insert(parent_key.clone(), imported_hir);
                                            }
                                            Ok(None) => {
                                                // 两者都失败 — fall back to inline mod navigation below
                                            }
                                            Err(e) => return Err(e),
                                        }
                                    }
                                }
                                Err(e) => return Err(e),
                            }
                        }
                        // 查找加载的模块：先查 full_key，再查 parent_key
                        if let Some(m) = self.modules.get(&full_key) {
                            loaded_module = Some(m);
                        } else if let Some(m) = self.modules.get(&parent_key) {
                            loaded_module = Some(m);
                        }

                        if let Some(module) = loaded_module {
                            // 先 clone 为拥有值：避免 module（借自 self.modules）
                            // 的不可变借用与下方 self.merge_module_globals 的可变借用冲突。
                            let module = module.clone();
                            // Add ALL functions from the module so that the
                            // imported function can call its helpers
                            // (e.g., parse calls _parse_value)
                            for fn_def in &module.functions {
                                if !self.functions.iter().any(|f| f.name == fn_def.name) {
                                    self.functions.push(fn_def.clone());
                                }
                            }
                            // 泛型函数存于 module.generic_funcs（非 module.functions）
                            for fn_def in &module.generic_funcs {
                                self.generic_funcs.insert(fn_def.name.clone(), fn_def.clone());
                            }
                            // M3.5：模块顶层 let 全局随 use 导入合并（常量/状态）
                            self.merge_module_globals(&module.globals);
                            // Define the specifically requested function in scope.
                            // 目标函数可能在 functions（非泛型）或 generic_funcs（泛型）中。
                            let found = module.functions.iter().find(|f| &f.name == fn_name)
                                .or_else(|| module.generic_funcs.iter().find(|f| &f.name == fn_name));
                            if let Some(fn_def) = found {
                                let param_types = fn_def.params.clone();
                                let ret_ty = fn_def.return_type.clone();
                                self.scope.define_fn(alias.clone(), param_types, ret_ty);
                                if !fn_def.generics.is_empty() {
                                    // 用 alias 作为键，使调用端写 `alias<T>(...)` 能解析到
                                    self.generic_funcs.insert(alias.clone(), fn_def.clone());
                                }
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
                                // 先 clone 为拥有值：避免 module（借自 self.modules）
                                // 的不可变借用与下方 self.merge_module_globals 的可变借用冲突。
                                let module = module.clone();
                                // M3.5：模块顶层 let 全局随 use 导入合并（常量/状态）
                                self.merge_module_globals(&module.globals);
                                // 目标函数可能在 functions 或 generic_funcs 中
                                let found = module.functions.iter().find(|f| &f.name == fn_name)
                                    .or_else(|| module.generic_funcs.iter().find(|f| &f.name == fn_name));
                                if let Some(fn_def) = found {
                                    let param_types = fn_def.params.clone();
                                    let ret_ty = fn_def.return_type.clone();
                                    self.scope.define_fn(alias.clone(), param_types, ret_ty);
                                    if !fn_def.generics.is_empty() {
                                        self.generic_funcs.insert(alias.clone(), fn_def.clone());
                                    } else if !self.functions.iter().any(|f| f.name == fn_def.name) {
                                        self.functions.push(fn_def.clone());
                                    }
                                }
                            }
                        }
                    }
                }
                ast::ItemKind::Function { name, generics, params, return_type, body, is_test, .. } => {
                    if name.name == "<expr>" {
                        continue;
                    }
                    let gen_names: Vec<String> = generics.iter().map(|g| g.name.name.clone()).collect();
                    let param_types: Vec<(String, Type)> = params.iter()
                        .map(|p| (p.name.name.clone(), self.annotation_type(&p.type_ann)))
                        .collect();
                    let ret_ty = return_type.as_ref()
                        .map(|rt| self.annotation_type(rt))
                        .unwrap_or(Type::unit());

                    let body_scope = Scope::with_parent(std::mem::replace(&mut self.scope, Scope::new()));
                    self.scope = body_scope;

                    for (n, t) in &param_types {
                        self.scope.define_var(n.clone(), t.clone(), false);
                    }

                    let lowered_body = self.lower_expr(body)?;

                    // 函数体作用域结束：弹回父作用域（函数局部变量不外泄到后续函数）。
                    self.scope = *self.scope.parent.take().unwrap();

                    // 跨函数 shape 求解（函子化，阶段 0）：合并 body 推断的 shape 到 return_type
                    // 让调用方能拿到更精确的返回 shape（如 `fn make() -> Tensor[f64, ..] { zeros(3,4) }` → [3,4]）
                    //
                    // 多 return 路径 join：由 `collect_return_tensor_dims` 对已 lower 的 HIR 做
                    // 纯递归推导——收集所有 return 语句的 shape + Block 末表达式的 shape
                    // （若 body 末尾是表达式而非 return）。不再依赖全局可变收集器。
                    // join 规则：相同 Known 保留、不同 Known 降为 Any、Symbol 相同保留、不同 Symbol 降为 Any。
                    // 维度数不同则报错（明确的逻辑错误）。
                    let mut all_shapes: Vec<Vec<Dim>> = Self::collect_return_tensor_dims(&lowered_body);
                    if let Type::Tensor { dims, .. } = &lowered_body.ty {
                        if !matches!(lowered_body.ty, Type::Never) && dims.iter().any(|d| !matches!(d, Dim::Any)) {
                            all_shapes.push(dims.clone());
                        }
                    }
                    let actual_ty = if all_shapes.is_empty() {
                        lowered_body.ty.clone()
                    } else {
                        match Self::join_return_dims(&all_shapes) {
                            Ok(joined_dims) => {
                                let dtype = match &lowered_body.ty {
                                    Type::Tensor { dtype, .. } => dtype.clone(),
                                    _ => match &ret_ty {
                                        Type::Tensor { dtype, .. } => dtype.clone(),
                                        _ => Box::new(Type::Unknown),
                                    },
                                };
                                Type::Tensor { dtype, dims: joined_dims }
                            }
                            Err(msg) => {
                                return Err(TenthError::TypeError {
                                    line: item.span.line,
                                    col: item.span.col,
                                    message: format!("函数 '{}' 多 return 路径 shape {}", name.name, msg),
                                });
                            }
                        }
                    };

                    // 若 body 推断 shape 与注解 shape 静态冲突（如注解 [3,4] 但 body 推断 [2,3]），报错。
                    let merged_ret_ty = Self::check_and_merge_tensor_shape(
                        &ret_ty, &actual_ty, &item.span, "函数返回值"
                    )?;

                    // Lower default parameter values
                    let mut param_defaults: Vec<Option<HirExpr>> = Vec::new();
                    for p in params.iter() {
                        if let Some(dv) = &p.default_value {
                            let lowered = self.lower_expr(dv)?;
                            param_defaults.push(Some(lowered));
                        } else {
                            param_defaults.push(None);
                        }
                    }

                    let fn_def = HirFnDef {
                        name: name.name.clone(),
                        generics: gen_names,
                        generics_bounds: build_generics_bounds(generics),
                        params: param_types,
                        param_defaults,
                        param_variadic: params.iter().map(|p| p.variadic).collect(),
                        return_type: merged_ret_ty,
                        body: lowered_body,
                        span: item.span.clone(),
                        is_test: *is_test,
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

        // 断点 4.2：前向引用 shape 不动点 pass。
        // 全部函数 lower 完成后，对同文件常规函数做「无环全量不动点 + 轮数上限兜底」：
        // 每轮用上一轮的精化返回类型重算调用点 ret_ty（纯 shape pass，复用
        // resolve_method_type / infer_binary_type / collect_return_tensor_dims /
        // join_return_dims / check_and_merge_tensor_shape），直到稳定或达 2n+1 轮；
        // 递归（SCC）场景超限后回退首轮状态（保留既有行为）。
        // 收敛性见 docs/shape-check-roadmap/前向引用断点收敛性论证.md（数理部 2026-08-02）。
        // 主表达式（<expr>）在本 pass 之后 lower——其调用点天然拿到精化后的返回类型。
        self.run_forward_ref_shape_fixpoint()?;

        // 自动派生 Copy trait：对于所有字段都是 Copy 类型的结构体，自动注册 impl Copy。
        // 与 Rust 语义一致：实现 Drop 的类型不可派生 Copy（drop 会随拷贝倍增）。
        let copy_auto_types: Vec<String> = self.structs.iter()
            .filter(|(name, fields)| {
                !self.trait_impls.get("Copy").map_or(false, |impls| impls.contains_key(*name))
                    && !self.trait_impls.get("Drop").map_or(false, |impls| impls.contains_key(*name))
                    && fields.iter().all(|(_, ft)| super::is_copy_type(ft, &self.structs, &self.trait_impls))
            })
            .map(|(name, _)| name.clone())
            .collect();
        for type_name in copy_auto_types {
            self.trait_impls.entry("Copy".to_string())
                .or_insert_with(HashMap::new)
                .insert(type_name, HashMap::new());
        }

        let mut main_expr = None;
        for item in &items {
            if let ast::ItemKind::Function { name, body, .. } = &item.kind {
                if name.name == "<expr>" {
                    let body_scope = Scope::with_parent(std::mem::replace(&mut self.scope, Scope::new()));
                    self.scope = body_scope;

                    let lowered_body = self.lower_expr(body)?;

                    // <expr> 顶层表达式作用域结束：弹回父作用域。
                    self.scope = *self.scope.parent.take().unwrap();

                    main_expr = Some(lowered_body);
                    break;
                }
            }
        }

        // 护城河 A 深化 Phase 2：跨算子反向 shape 传播 pass
        // 在所有 fn_defs lowering 完成后，对每个函数体做后向分析，
        // 把方向 A 的运行时 acc_grad shape 不匹配错误提升到编译期 TypeError。
        // 详见 `backward_shape_pass::backward_shape_pass`。
        let phase2_errors = super::backward_shape_pass::backward_shape_pass(&self.functions);
        if let Some(first_err) = phase2_errors.into_iter().next() {
            return Err(first_err);
        }

        // 层 3 lossy lattice M2：污点旁路分析（方案 C）
        // 对已 lower 的完整程序做结构递归污点传播（含跨函数，函子组合性）与使用点检查。
        // `lossy` 为纯编译期构造（bytecode/wasm 编译为 inner，运行时 no-op），
        // 本 pass 在 lowering 完成后统一执行，Type/HIR 数据结构零侵入。
        let taint_errors = super::taint::analyze_program(
            &self.functions,
            &self.generic_funcs,
            &self.methods,
            &self.generic_instantiations,
            &main_expr,
        );
        if let Some(first_err) = taint_errors.into_iter().next() {
            return Err(first_err);
        }

        Ok(HirProgram {
            functions: self.functions.clone(),
            generic_funcs: self.generic_funcs.values().cloned().collect(),
            globals: self.globals.clone(),
            main_expr,
            modules: self.modules.clone(),
            uses: self.uses.clone(),
            methods: self.methods.clone(),
            structs: self.structs.clone(),
            generic_structs: self.generic_structs.clone(),
            unions: self.unions.clone(),
            enums: self.enums.clone(),
            generic_enums: self.generic_enums.clone(),
            trait_defs: self.trait_defs.clone(),
            trait_impls: self.trait_impls.clone(),
            warnings: std::mem::take(&mut self.warnings),
        })
    }
}
