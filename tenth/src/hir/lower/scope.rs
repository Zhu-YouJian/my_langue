use std::collections::HashMap;
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
    functions: HashMap<String, (Vec<(String, Type)>, Type)>,
    ownership: HashMap<String, Ownership>,
    pub(super) parent: Option<Box<Scope>>,
}

impl Scope {
    pub(super) fn new() -> Self {
        Scope {
            variables: HashMap::new(),
            functions: HashMap::new(),
            ownership: HashMap::new(),
            parent: None,
        }
    }

    pub(super) fn with_parent(parent: Scope) -> Self {
        Scope {
            variables: HashMap::new(),
            functions: HashMap::new(),
            ownership: HashMap::new(),
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

    pub(super) fn check_use(&self, name: &str, span: &crate::lexer::token::Span) -> TenthResult<()> {
        if let Some(Ownership::Moved) = self.get_ownership(name) {
            return Err(TenthError::TypeError {
                line: span.line, col: span.col,
                message: format!("use of moved value '{}'", name),
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
                message: format!("cannot borrow moved value '{}'", name),
            }),
            _ => Ok(()),
        }
    }

    pub(super) fn check_borrow_mut(&self, name: &str, span: &crate::lexer::token::Span) -> TenthResult<()> {
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

    pub(super) fn define_var(&mut self, name: String, ty: Type, mutable: bool) {
        self.variables.insert(name.clone(), (ty, mutable));
        self.ownership.insert(name, Ownership::Owned);
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
    pub(super) fn release_borrows(&mut self) {
        for (_, state) in self.ownership.iter_mut() {
            match state {
                Ownership::SharedRef(_) | Ownership::ExclusiveRef => {
                    *state = Ownership::Owned;
                }
                _ => {}
            }
        }
    }

    pub(super) fn define_fn(&mut self, name: String, params: Vec<(String, Type)>, ret: Type) {
        self.functions.insert(name, (params, ret));
    }

    pub(super) fn lookup_fn(&self, name: &str) -> Option<(Vec<(String, Type)>, Type)> {
        if let Some(f) = self.functions.get(name) {
            return Some(f.clone());
        }
        self.parent.as_ref().and_then(|p| p.lookup_fn(name))
    }
}
