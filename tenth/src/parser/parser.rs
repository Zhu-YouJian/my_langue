//! 递归下降解析器核心。
//!
//! 本模块仅保留 `Parser` 结构体定义、核心辅助方法（peek/advance/expect 等）
//! 和程序入口（`parse_program` / `parse_program_with_recovery`）。
//!
//! 解析逻辑已按职责拆分到子模块（均通过 cross-module `impl Parser` 扩展）：
//! - `expr.rs`：表达式解析（parse_expr / parse_primary / parse_binary 等）
//! - `stmt.rs`：语句解析（parse_stmt / parse_block_stmts / parse_param / 模式匹配）
//! - `decl.rs`：声明解析（parse_item：fn / struct / enum / impl / mod / use / trait）
//! - `type_parser.rs`：类型注解解析（parse_type / parse_generic_params）

use crate::error::{TenthError, TenthResult};
use crate::lexer::token::{Span, Token, TokenKind};
use super::ast::*;
use std::collections::HashSet;

pub struct Parser {
    pub(super) tokens: Vec<Token>,
    pub(super) pos: usize,
    pub(super) known_enums: HashSet<String>,
    /// Collected errors during error-recovery parsing
    pub(super) errors: Vec<TenthError>,
}

/// Synchronization tokens — we skip to these after a parse error to resume parsing.
const SYNC_TOKENS: &[TokenKind] = &[
    TokenKind::Fn,
    TokenKind::Struct,
    TokenKind::Enum,
    TokenKind::Trait,
    TokenKind::Impl,
    TokenKind::Mod,
    TokenKind::Use,
    TokenKind::Macro,
    TokenKind::RBrace,
    TokenKind::Eof,
];

static EOF_TOKEN: Token = Token {
    kind: TokenKind::Eof,
    span: Span { line: 0, col: 0 },
};

impl Parser {
    pub fn new(tokens: Vec<Token>) -> Self {
        let mut known_enums = HashSet::new();
        known_enums.insert("Option".to_string());
        known_enums.insert("Result".to_string());
        Parser { tokens, pos: 0, known_enums, errors: Vec::new() }
    }

    pub(super) fn peek(&self) -> &Token {
        self.tokens.get(self.pos).unwrap_or(&EOF_TOKEN)
    }

    pub(super) fn peek_kind(&self) -> &TokenKind {
        &self.peek().kind
    }

    pub(super) fn advance(&mut self) -> &Token {
        let idx = self.pos;
        self.pos += 1;
        self.tokens.get(idx).unwrap_or(&EOF_TOKEN)
    }

    /// Expect a `>` token to close a generic type parameter list.
    /// Handles the case where `>>` is tokenized as `Shr` by splitting it:
    /// consumes the `Shr` and inserts a synthetic `Gt` token at the current
    /// position so the outer generic context sees its closing `>`.
    pub(super) fn expect_gt(&mut self) -> TenthResult<&Token> {
        let token = self.peek();
        match &token.kind {
            TokenKind::Gt => {
                self.pos += 1;
                Ok(self.tokens.get(self.pos - 1).unwrap())
            }
            TokenKind::Shr => {
                // >> encountered where > expected: split into two > tokens.
                // Consume the Shr and insert a synthetic Gt at current position.
                let span = token.span.clone();
                self.pos += 1;
                // Insert a synthetic Gt token so the outer parse_type sees it
                let synthetic_gt = Token {
                    kind: TokenKind::Gt,
                    span: Span { line: span.line, col: span.col + 1 },
                };
                self.tokens.insert(self.pos, synthetic_gt);
                Ok(self.tokens.get(self.pos - 1).unwrap())
            }
            _ => {
                Err(TenthError::ParseError {
                    line: token.span.line,
                    col: token.span.col,
                    message: format!("期望 >，但遇到了 {}", token.kind),
                })
            }
        }
    }

    pub(super) fn expect(&mut self, kind: TokenKind) -> TenthResult<&Token> {
        let token = self.peek();
        if std::mem::discriminant(&token.kind) == std::mem::discriminant(&kind) {
            self.pos += 1;
            Ok(self.tokens.get(self.pos - 1).unwrap())
        } else {
            Err(TenthError::ParseError {
                line: token.span.line,
                col: token.span.col,
                message: format!("期望 {}，但遇到了 {}", kind, token.kind),
            })
        }
    }

    pub(super) fn span(&self) -> Span {
        self.peek().span.clone()
    }

    pub(super) fn at_eof(&self) -> bool {
        matches!(self.peek_kind(), TokenKind::Eof)
    }

    pub(super) fn match_token(&mut self, kind: TokenKind) -> bool {
        if std::mem::discriminant(self.peek_kind()) == std::mem::discriminant(&kind) {
            self.advance();
            true
        } else {
            false
        }
    }

    /// Skip tokens until we reach a synchronization point, allowing parsing to resume.
    fn synchronize(&mut self) {
        loop {
            let kind = self.peek_kind();
            if SYNC_TOKENS.iter().any(|t| std::mem::discriminant(kind) == std::mem::discriminant(t)) {
                break;
            }
            if self.at_eof() { break; }
            self.pos += 1;
        }
    }

    /// Record a parse error and attempt to recover by synchronizing.
    /// Returns a placeholder item so parsing can continue.
    fn record_error_and_recover(&mut self, err: TenthError) -> Option<Item> {
        self.errors.push(err);
        self.synchronize();
        None
    }

    /// Parse the program with error recovery, collecting multiple errors.
    /// Returns the successfully parsed items and any errors encountered.
    pub fn parse_program_with_recovery(&mut self) -> (Program, Vec<TenthError>) {
        let mut items = Vec::new();
        let mut main_expr_stmts = Vec::new();
        let mut has_main_fn = false;

        while !self.at_eof() {
            match self.parse_item() {
                Ok(item) => {
                    if let ItemKind::Function { name, .. } = &item.kind {
                        if name.name == "<expr>" {
                            if let ItemKind::Function { body, .. } = item.kind {
                                if let ExprKind::Block(stmts) = body.kind {
                                    main_expr_stmts.extend(stmts);
                                } else {
                                    main_expr_stmts.push(Stmt { kind: StmtKind::Expr(body), span: item.span });
                                }
                            }
                            continue;
                        }
                        if name.name == "main" { has_main_fn = true; }
                    }
                    items.push(item);
                }
                Err(err) => {
                    self.synchronize();
                    self.errors.push(err);
                }
            }
        }

        if !main_expr_stmts.is_empty() && !has_main_fn {
            let span = if let Some(s) = main_expr_stmts.first() { s.span.clone() } else { Span { line: 1, col: 0 } };
            items.push(Item {
                kind: ItemKind::Function {
                    name: Ident { name: "<expr>".into(), span: span.clone() },
                    generics: Vec::new(),
                    params: Vec::new(),
                    return_type: None,
                    body: Expr { kind: ExprKind::Block(main_expr_stmts), span: span.clone() },
                    is_pub: false,
                    is_async: false,
                    is_test: false,
                },
                span,
            });
        }

        let errors = std::mem::take(&mut self.errors);
        (Program { items }, errors)
    }

    pub fn parse_program(&mut self) -> TenthResult<Program> {
        let mut items = Vec::new();
        let mut stmts = Vec::new();
        while !self.at_eof() {
            match self.peek_kind() {
                TokenKind::Async | TokenKind::Fn | TokenKind::Struct | TokenKind::Enum | TokenKind::Union
                | TokenKind::Impl | TokenKind::Mod | TokenKind::Use | TokenKind::Trait
                | TokenKind::Operator | TokenKind::Macro => {
                    items.push(self.parse_item()?);
                }
                _ => {
                    let span = self.span();
                    match self.peek_kind() {
                        // M2.3：Lifetime(_) —— `'label: while/for/loop/do` 循环标签前缀
                        // （复用 Lifetime token），需走 parse_stmt 解析标签。
                        // 顺带补全顶层语句分发缺口：for/loop/do/return/break/continue
                        // 也是 StmtKind 语句，此前在顶层会落入 parse_expr 报错。
                        TokenKind::Let | TokenKind::While | TokenKind::For
                        | TokenKind::Loop | TokenKind::Do | TokenKind::Return
                        | TokenKind::Break | TokenKind::Continue | TokenKind::Lifetime(_) => {
                            stmts.push(self.parse_stmt()?);
                        }
                        _ => {
                            let expr = self.parse_expr()?;
                            self.match_token(TokenKind::Semicolon);
                            stmts.push(Stmt {
                                kind: StmtKind::Expr(expr),
                                span,
                            });
                        }
                    }
                }
            }
        }
        if !stmts.is_empty() {
            let span = stmts
                .first()
                .map(|s| s.span.clone())
                .unwrap_or_else(|| self.span());
            items.push(Item {
                kind: ItemKind::Function {
                    name: Ident {
                        name: "<expr>".into(),
                        span: span.clone(),
                    },
                    generics: Vec::new(),
                    params: Vec::new(),
                    return_type: None,
                    body: Expr {
                        kind: ExprKind::Block(stmts),
                        span: span.clone(),
                    },
                    is_pub: false,
                    is_async: false,
                    is_test: false,
                },
                span,
            });
        }
        let mut program = Program { items };
        // M3.3：声明式宏展开 pass（parse 完成后、lower 前）。收集宏定义并从 AST
        // 移除，再把调用点 AST 替换为 body（参数代入）。挂在 parse_program 末尾
        // 使所有调用方（main/repl/wasm host/import/测试）零改动获得宏能力。
        super::macro_expand::expand_program_macros(&mut program)?;
        Ok(program)
    }
}
