use std::collections::{HashMap, HashSet};
use crate::error::{TenthError, TenthResult};
use crate::hir::types::Type;

#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) enum Ownership {
    Owned,
    SharedRef(usize),
    ExclusiveRef,
    Moved,
}

pub(super) struct Scope {
    variables: HashMap<String, (Type, bool)>,
    /// 函数重载支持：同一个函数名可以对应多个签名。
    /// key = 函数名，value = Vec<(param_types, return_type)>
    functions: HashMap<String, Vec<(Vec<(String, Type)>, Type)>>,
    ownership: HashMap<String, Ownership>,
    /// 跨语句借用跟踪：holder_var -> Vec<borrowed_var>。
    /// 当 `let r = &x;` 创建持久借用时，记录 r → [x]，使 release_borrows
    /// 在 r 仍活跃时不释放 x 的借用状态（修复 AUDIT-11.1.1 T19 B6）。
    /// 支持 `let r = if c { &x } else { &y };` 多变量场景（修复 AUDIT-11.1.2 T20 PB2）。
    borrow_holders: HashMap<String, Vec<String>>,
    pub(super) parent: Option<Box<Scope>>,
}

impl Scope {
    pub(super) fn new() -> Self {
        Scope {
            variables: HashMap::new(),
            functions: HashMap::new(),
            ownership: HashMap::new(),
            borrow_holders: HashMap::new(),
            parent: None,
        }
    }

    pub(super) fn with_parent(parent: Scope) -> Self {
        Scope {
            variables: HashMap::new(),
            functions: HashMap::new(),
            ownership: HashMap::new(),
            borrow_holders: HashMap::new(),
            parent: Some(Box::new(parent)),
        }
    }

    pub(super) fn lookup_var(&self, name: &str) -> Option<(Type, bool)> {
        if let Some(v) = self.variables.get(name) {
            return Some(v.clone());
        }
        self.parent.as_ref().and_then(|p| p.lookup_var(name))
    }

    pub(super) fn get_ownership(&self, name: &str) -> Option<Ownership> {
        if let Some(o) = self.ownership.get(name) {
            return Some(*o);
        }
        self.parent.as_ref().and_then(|p| p.get_ownership(name))
    }

    pub(super) fn set_ownership(&mut self, name: &str, state: Ownership) {
        self.ownership.insert(name.to_string(), state);
    }

    /// 记录持久借用关系：`holder` 持有 `borrowed` 的引用。
    /// 由 `let r = &x;` / `let m = &mut x;` 触发，让 release_borrows
    /// 在 holder 仍活跃时保留 borrowed 的借用状态。
    /// 同一 holder 可多次调用以记录多变量借用（如 `if c { &x } else { &y }`）。
    pub(super) fn record_borrow_holder(&mut self, holder: &str, borrowed: &str) {
        self.borrow_holders
            .entry(holder.to_string())
            .or_insert_with(Vec::new)
            .push(borrowed.to_string());
    }

    pub(super) fn check_use(&self, name: &str, span: &crate::lexer::token::Span) -> TenthResult<()> {
        if let Some(Ownership::Moved) = self.get_ownership(name) {
            return Err(TenthError::TypeError {
                line: span.line, col: span.col,
                message: format!("使用了已移动的值 '{}'", name),
            });
        }
        Ok(())
    }

    pub(super) fn check_borrow_shared(&self, name: &str, span: &crate::lexer::token::Span) -> TenthResult<()> {
        match self.get_ownership(name) {
            Some(Ownership::ExclusiveRef) => Err(TenthError::TypeError {
                line: span.line, col: span.col,
                message: format!("不可将 '{}' 共享借用，因为它已被可变借用", name),
            }),
            Some(Ownership::Moved) => Err(TenthError::TypeError {
                line: span.line, col: span.col,
                message: format!("不可借用已移动的值 '{}'", name),
            }),
            _ => Ok(()),
        }
    }

    pub(super) fn check_borrow_mut(&self, name: &str, span: &crate::lexer::token::Span) -> TenthResult<()> {
        match self.get_ownership(name) {
            Some(Ownership::SharedRef(n)) if n > 0 => Err(TenthError::TypeError {
                line: span.line, col: span.col,
                message: format!("不可将 '{}' 可变借用，因为它已被共享借用", name),
            }),
            Some(Ownership::ExclusiveRef) => Err(TenthError::TypeError {
                line: span.line, col: span.col,
                message: format!("不可同时多次可变借用 '{}'", name),
            }),
            Some(Ownership::Moved) => Err(TenthError::TypeError {
                line: span.line, col: span.col,
                message: format!("不可借用已移动的值 '{}'", name),
            }),
            _ => Ok(()),
        }
    }

    /// 遍历当前作用域（不含父作用域）的所有变量及其类型。
    pub(super) fn for_each_var<F: FnMut(&str, &Type)>(&self, mut f: F) {
        for (name, (ty, _)) in &self.variables {
            f(name, ty);
        }
    }

    pub(super) fn define_var(&mut self, name: String, ty: Type, mutable: bool) {
        self.variables.insert(name.clone(), (ty, mutable));
        self.ownership.insert(name, Ownership::Owned);
    }

    /// 尝试 move 一个变量：如果类型是 Copy，不变更所有权状态；否则标记为 Moved。
    /// 返回 true 表示 move 成功（Copy 总是成功，non-Copy 只能 move 一次）。
    pub(super) fn try_move(&mut self, name: &str) -> bool {
        match self.get_ownership(name) {
            Some(Ownership::Moved) => false,
            Some(Ownership::Owned) => {
                // 检查变量类型是否为 Copy
                let is_copy = self.variables.get(name)
                    .map(|(ty, _)| {
                        // 简化的 Copy 检查：基础类型和引用是 Copy
                        matches!(ty, Type::Base(_) | Type::Ref(_) | Type::MutRef(_) | Type::Never)
                    })
                    .unwrap_or(false);
                if !is_copy {
                    self.set_ownership(name, Ownership::Moved);
                }
                true
            }
            _ => true, // References can be used any number of times
        }
    }

    /// Release all shared and exclusive borrows in the current scope.
    ///
    /// The borrow checker does not implement NLL or two-phase borrows, so without
    /// this, any `&x` or `&mut x` permanently marks `x` as borrowed for the rest
    /// of the enclosing scope. Releasing borrows after each statement and after
    /// sub-expressions like `if` conditions is a pragmatic approximation that
    /// allows common patterns like `if peek(&p).disc == 54 { advance(&mut p); }`
    /// while still catching double-mut-borrow within a single expression.
    /// `Moved` state is preserved so use-after-move is still detected.
    ///
    /// 跨语句借用保护（AUDIT-11.1.1 / T19 B6 修复）：
    /// 若 scope 中存在仍活跃的 borrow holder（如 `let r = &x;` 后 r 仍未离开 scope
    /// 且未被 move），则 r 所引用的变量的借用状态必须保留——否则下一条语句就能
    /// 通过 `&mut x` 偷走 r 持有的借用，造成别名违规。
    pub(super) fn release_borrows(&mut self) {
        // 收集仍活跃 holder 引用的 borrowed 变量名集合
        let mut protected: HashSet<String> = HashSet::new();
        for (holder, borroweds) in &self.borrow_holders {
            // holder 必须仍在本 scope 的 variables 表中
            if !self.variables.contains_key(holder) {
                continue;
            }
            // holder 已被 move，借用关系随之失效
            if matches!(self.ownership.get(holder), Some(Ownership::Moved)) {
                continue;
            }
            for borrowed in borroweds {
                protected.insert(borrowed.clone());
            }
        }
        for (name, state) in self.ownership.iter_mut() {
            if protected.contains(name) {
                continue;
            }
            match state {
                Ownership::SharedRef(_) | Ownership::ExclusiveRef => {
                    *state = Ownership::Owned;
                }
                _ => {}
            }
        }
    }

    pub(super) fn define_fn(&mut self, name: String, params: Vec<(String, Type)>, ret: Type) {
        let entry = self.functions.entry(name).or_insert_with(Vec::new);
        // 若已存在完全相同的签名（参数类型完全一致），跳过重复定义
        if !entry.iter().any(|(existing_params, _)| {
            existing_params.len() == params.len()
                && existing_params.iter().zip(params.iter()).all(|(a, b)| a.1 == b.1)
        }) {
            entry.push((params, ret));
        }
    }

    /// 按函数名查找函数签名列表。返回所有同名的重载签名。
    pub(super) fn lookup_fn_all(&self, name: &str) -> Option<&Vec<(Vec<(String, Type)>, Type)>> {
        if let Some(fns) = self.functions.get(name) {
            return Some(fns);
        }
        self.parent.as_ref().and_then(|p| p.lookup_fn_all(name))
    }

    /// 查询函数签名。若有重载，按参数类型匹配最佳签名；若无重载，返回唯一签名。
    pub(super) fn lookup_fn(&self, name: &str) -> Option<(Vec<(String, Type)>, Type)> {
        let all = self.lookup_fn_all(name)?;
        if all.len() == 1 {
            return Some(all[0].clone());
        }
        // 多签名场景：返回第一个（实际调用时由 resolve_call_type 按参数类型匹配合适签名）
        Some(all[0].clone())
    }

    /// 按参数类型匹配合适的函数重载签名。
    /// 返回匹配的 (param_types, return_type)。若无精确匹配，报告歧义。
    pub(super) fn resolve_fn_overload(
        &self,
        name: &str,
        arg_types: &[Type],
        span: &crate::lexer::token::Span,
    ) -> TenthResult<(Vec<(String, Type)>, Type)> {
        let all = self.lookup_fn_all(name).ok_or_else(|| {
            TenthError::TypeError {
                line: span.line, col: span.col,
                message: format!("未定义的函数 '{}'", name),
            }
        })?;

        if all.is_empty() {
            return Err(TenthError::TypeError {
                line: span.line, col: span.col,
                message: format!("未定义的函数 '{}'", name),
            });
        }

        if all.len() == 1 {
            return Ok(all[0].clone());
        }

        // 多重载：找参数数量匹配且参数类型精确匹配的签名
        let mut matches: Vec<&(Vec<(String, Type)>, Type)> = all.iter()
            .filter(|(params, _)| params.len() == arg_types.len())
            .filter(|(params, _)| {
                params.iter().zip(arg_types.iter()).all(|((_, pty), aty)| {
                    pty == aty
                })
            })
            .collect();

        if matches.is_empty() {
            // 没有精确匹配，尝试兼容匹配（参数类型可统一）
            let mut compatible: Vec<&(Vec<(String, Type)>, Type)> = all.iter()
                .filter(|(params, _)| params.len() == arg_types.len())
                .collect();
            if compatible.is_empty() {
                return Err(TenthError::TypeError {
                    line: span.line, col: span.col,
                    message: format!(
                        "函数 '{}' 有 {} 个重载，但参数数量 {} 与任一重载不匹配",
                        name, all.len(), arg_types.len()
                    ),
                });
            }
            if compatible.len() == 1 {
                return Ok(compatible[0].clone());
            }
            matches = compatible;
        }

        if matches.len() == 1 {
            return Ok(matches[0].clone());
        }

        // 多个匹配：报告歧义
        let sigs: Vec<String> = matches.iter().map(|(params, ret)| {
            let param_str: Vec<String> = params.iter().map(|(_, t)| format!("{}", t)).collect();
            format!("({}) -> {}", param_str.join(", "), ret)
        }).collect();
        Err(TenthError::TypeError {
            line: span.line, col: span.col,
            message: format!(
                "函数 '{}' 调用歧义：匹配到多个重载签名 [{}]",
                name, sigs.join("; ")
            ),
        })
    }
}
