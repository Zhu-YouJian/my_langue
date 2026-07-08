//! 语句解析：`parse_stmt` / `parse_block_stmts` / `parse_param` / 模式匹配。
//!
//! 从 `parser.rs` 拆出（架构重构 T3d）。包含：
//! - `parse_stmt`：let / return / break / continue / loop / while / for / expr
//! - `parse_block_stmts`：块语句序列
//! - `parse_param`：函数参数（含 `self` 特殊处理）
//! - `parse_match_pattern`：match 模式（通配符 / 字面量 / 元组 / 枚举变体 / 结构体 / 绑定）
//! - `parse_pattern_fields` / `parse_pattern_fields_inner`：枚举变体字段绑定

use crate::error::{TenthError, TenthResult};
use crate::lexer::token::{Span, Token, TokenKind};
use super::ast::*;
use super::parser::Parser;

impl Parser {
    pub(super) fn parse_param(&mut self) -> TenthResult<Param> {
        let name = match &self.peek().kind {
            TokenKind::Identifier(name) => Ident {
                name: name.clone(),
                span: self.peek().span.clone(),
            },
            TokenKind::Self_ => Ident {
                name: "self".to_string(),
                span: self.peek().span.clone(),
            },
            _ => {
                return Err(TenthError::ParseError {
                    line: self.peek().span.line,
                    col: self.peek().span.col,
                    message: "期望参数名".into(),
                });
            }
        };
        self.advance();
        if name.name == "self" && !matches!(self.peek_kind(), TokenKind::Colon) {
            let type_ann = TypeAnnotation::Named(Ident {
                name: "Self".to_string(),
                span: name.span.clone(),
            });
            return Ok(Param { name, type_ann });
        }
        let type_ann = if matches!(self.peek_kind(), TokenKind::Colon) {
            self.advance();
            self.parse_type()?
        } else {
            // Untyped parameter — infer as Unknown
            TypeAnnotation::Named(Ident {
                name: "Unknown".to_string(),
                span: name.span.clone(),
            })
        };
        Ok(Param { name, type_ann })
    }

    pub(super) fn parse_block_stmts(&mut self) -> TenthResult<Vec<Stmt>> {
        let mut stmts = Vec::new();
        while !matches!(self.peek_kind(), TokenKind::RBrace) && !self.at_eof() {
            stmts.push(self.parse_stmt()?);
        }
        self.expect(TokenKind::RBrace)?;
        Ok(stmts)
    }

    pub(super) fn parse_stmt(&mut self) -> TenthResult<Stmt> {
        let span = self.span();

        match self.peek_kind() {
            TokenKind::Let => {
                self.advance();
                let mutable = self.match_token(TokenKind::Mut);

                // Check for tuple destructuring: let (a, b, c) = expr
                let names: Vec<Ident> = if matches!(self.peek_kind(), TokenKind::LParen) {
                    self.advance();
                    let mut idents = Vec::new();
                    if !matches!(self.peek_kind(), TokenKind::RParen) {
                        loop {
                            if let TokenKind::Identifier(name) = &self.peek().kind {
                                idents.push(Ident {
                                    name: name.clone(),
                                    span: self.peek().span.clone(),
                                });
                                self.advance();
                            } else {
                                return Err(TenthError::ParseError {
                                    line: self.peek().span.line,
                                    col: self.peek().span.col,
                                    message: "expected variable name in destructuring".into(),
                                });
                            }
                            if !matches!(self.peek_kind(), TokenKind::Comma) {
                                break;
                            }
                            self.advance();
                            if matches!(self.peek_kind(), TokenKind::RParen) {
                                break;
                            }
                        }
                    }
                    self.expect(TokenKind::RParen)?;
                    idents
                } else if let TokenKind::Identifier(name) = &self.peek().kind {
                    let idents = vec![Ident {
                        name: name.clone(),
                        span: self.peek().span.clone(),
                    }];
                    self.advance();
                    idents
                } else {
                    return Err(TenthError::ParseError {
                        line: self.peek().span.line,
                        col: self.peek().span.col,
                        message: "expected variable name".into(),
                    });
                };
                // Note: advance() is now per-branch — LParen path already consumed `)`
                // via expect(RParen), so it must NOT advance again (would eat `=`).

                let type_ann = if matches!(self.peek_kind(), TokenKind::Colon) {
                    self.advance();
                    Some(self.parse_type()?)
                } else {
                    None
                };

                let init = if matches!(self.peek_kind(), TokenKind::Assign) {
                    self.advance();
                    Some(self.parse_expr()?)
                } else {
                    None
                };

                self.match_token(TokenKind::Semicolon);

                Ok(Stmt {
                    kind: StmtKind::Let {
                        names,
                        type_ann,
                        mutable,
                        init,
                    },
                    span,
                })
            }
            TokenKind::Return => {
                self.advance();
                let value = if !matches!(self.peek_kind(), TokenKind::Semicolon)
                    && !matches!(self.peek_kind(), TokenKind::RBrace)
                {
                    Some(self.parse_expr()?)
                } else {
                    None
                };
                self.match_token(TokenKind::Semicolon);
                Ok(Stmt {
                    kind: StmtKind::Return(value),
                    span,
                })
            }
            TokenKind::Break => {
                self.advance();
                self.match_token(TokenKind::Semicolon);
                Ok(Stmt {
                    kind: StmtKind::Break,
                    span,
                })
            }
            TokenKind::Continue => {
                self.advance();
                self.match_token(TokenKind::Semicolon);
                Ok(Stmt {
                    kind: StmtKind::Continue,
                    span,
                })
            }
            TokenKind::Loop => {
                self.advance();
                self.expect(TokenKind::LBrace)?;
                let stmts = self.parse_block_stmts()?;
                self.match_token(TokenKind::Semicolon);
                Ok(Stmt {
                    kind: StmtKind::Loop { body: stmts },
                    span,
                })
            }
            TokenKind::While => {
                self.advance();
                let cond = self.parse_expr()?;
                let body = if matches!(self.peek_kind(), TokenKind::LBrace) {
                    self.advance();
                    let stmts = self.parse_block_stmts()?;
                    Stmt {
                        kind: StmtKind::Expr(Expr {
                            kind: ExprKind::Block(stmts),
                            span: self.span(),
                        }),
                        span: self.span(),
                    }
                } else {
                    self.parse_stmt()?
                };
                self.match_token(TokenKind::Semicolon);
                Ok(Stmt {
                    kind: StmtKind::While {
                        cond,
                        body: Box::new(body),
                    },
                    span,
                })
            }
            TokenKind::For => {
                self.advance();
                let var = self.expect_ident()?;
                self.expect(TokenKind::In)?;
                let iter = self.parse_expr()?;
                let body = if matches!(self.peek_kind(), TokenKind::LBrace) {
                    self.advance();
                    let stmts = self.parse_block_stmts()?;
                    Stmt {
                        kind: StmtKind::Expr(Expr {
                            kind: ExprKind::Block(stmts),
                            span: self.span(),
                        }),
                        span: self.span(),
                    }
                } else {
                    self.parse_stmt()?
                };
                self.match_token(TokenKind::Semicolon);
                Ok(Stmt {
                    kind: StmtKind::For {
                        var,
                        iter,
                        body: Box::new(body),
                    },
                    span,
                })
            }
            _ => {
                let expr = self.parse_expr()?;
                self.match_token(TokenKind::Semicolon);
                Ok(Stmt {
                    kind: StmtKind::Expr(expr),
                    span,
                })
            }
        }
    }

    pub(super) fn parse_match_pattern(&mut self) -> TenthResult<Pattern> {
        match &self.peek().kind {
            TokenKind::Identifier(s) if s == "_" => {
                self.advance();
                Ok(Pattern::Wildcard)
            }
            TokenKind::IntLiteral(n) => {
                let n = *n;
                self.advance();
                // Check for range pattern: 1..10 or 1..=10
                if matches!(self.peek_kind(), TokenKind::DotDot) {
                    self.advance();
                    let inclusive = self.match_token(TokenKind::Assign);
                    if let TokenKind::IntLiteral(end) = &self.peek().kind {
                        let end = *end;
                        self.advance();
                        return Ok(Pattern::Range { start: n, end, inclusive });
                    }
                    return Err(TenthError::ParseError {
                        line: self.peek().span.line,
                        col: self.peek().span.col,
                        message: "范围模式缺少结束值".into(),
                    });
                }
                if matches!(self.peek_kind(), TokenKind::DotDotEq) {
                    self.advance();
                    if let TokenKind::IntLiteral(end) = &self.peek().kind {
                        let end = *end;
                        self.advance();
                        return Ok(Pattern::Range { start: n, end, inclusive: true });
                    }
                    return Err(TenthError::ParseError {
                        line: self.peek().span.line,
                        col: self.peek().span.col,
                        message: "范围模式缺少结束值".into(),
                    });
                }
                Ok(Pattern::Literal(Literal::Int(n)))
            }
            TokenKind::FloatLiteral(n, dt) => {
                let n = *n;
                let dt = *dt;
                self.advance();
                Ok(Pattern::Literal(Literal::Float(n, dt)))
            }
            TokenKind::True => {
                self.advance();
                Ok(Pattern::Literal(Literal::Bool(true)))
            }
            TokenKind::False => {
                self.advance();
                Ok(Pattern::Literal(Literal::Bool(false)))
            }
            TokenKind::LParen => {
                // Tuple pattern: (a, b, c)
                self.advance();
                let mut patterns = Vec::new();
                if !matches!(self.peek_kind(), TokenKind::RParen) {
                    patterns.push(self.parse_match_pattern()?);
                    while matches!(self.peek_kind(), TokenKind::Comma) {
                        self.advance();
                        if matches!(self.peek_kind(), TokenKind::RParen) { break; }
                        patterns.push(self.parse_match_pattern()?);
                    }
                }
                self.expect(TokenKind::RParen)?;
                Ok(Pattern::Tuple(patterns))
            }
            TokenKind::Identifier(_) => {
                let path_str = self.parse_qualified_path()?;
                let parts: Vec<&str> = path_str.splitn(2, "::").collect();
                if parts.len() == 2 {
                    let enum_name = parts[0].to_string();
                    let variant = parts[1].to_string();
                    let (field_bind, tuple_fields) = self.parse_pattern_fields()?;
                    Ok(Pattern::EnumVariant { enum_name, variant, field_bind, tuple_fields })
                } else {
                    let name = path_str;
                    if matches!(self.peek_kind(), TokenKind::LParen) {
                        self.advance();
                        let (field_bind, tuple_fields) = self.parse_pattern_fields_inner()?;
                        Ok(Pattern::EnumVariant {
                            enum_name: String::new(),
                            variant: name,
                            field_bind,
                            tuple_fields,
                        })
                    } else if matches!(self.peek_kind(), TokenKind::LBrace) {
                        // Struct destructuring pattern: `Point { x, y }` or `Point { x: a, y: b }`
                        self.advance();
                        let mut fields: Vec<(String, String)> = Vec::new();
                        if !matches!(self.peek_kind(), TokenKind::RBrace) {
                            loop {
                                let field = self.expect_ident()?;
                                let bind_name = if matches!(self.peek_kind(), TokenKind::Colon) {
                                    self.advance();
                                    let b = self.expect_ident()?;
                                    b.name
                                } else {
                                    // Shorthand: `x` binds to `x`
                                    field.name.clone()
                                };
                                fields.push((field.name, bind_name));
                                if matches!(self.peek_kind(), TokenKind::Comma) {
                                    self.advance();
                                    if matches!(self.peek_kind(), TokenKind::RBrace) { break; }
                                } else {
                                    break;
                                }
                            }
                        }
                        self.expect(TokenKind::RBrace)?;
                        Ok(Pattern::Struct { name, fields })
                    } else {
                        // Variable binding pattern
                        Ok(Pattern::Binding(name))
                    }
                }
            }
            _ => Err(TenthError::ParseError {
                line: self.peek().span.line,
                col: self.peek().span.col,
                message: "expected match pattern".into(),
            }),
        }
    }

    /// Parse the field bindings after an enum variant name in a match pattern.
    /// Returns (field_bind, tuple_fields).
    /// - Named field: `Some(value: v)` → field_bind = Some(("value", "v"))
    /// - Tuple field: `Some(x)` → tuple_fields = ["x"]
    /// - Multiple tuple fields: `Pair(a, b)` → tuple_fields = ["a", "b"]
    pub(super) fn parse_pattern_fields(&mut self) -> TenthResult<(Option<(String, String)>, Vec<String>)> {
        if !matches!(self.peek_kind(), TokenKind::LParen) {
            return Ok((None, Vec::new()));
        }
        self.advance();
        self.parse_pattern_fields_inner()
    }

    /// Parse field bindings inside parentheses (LParen already consumed).
    pub(super) fn parse_pattern_fields_inner(&mut self) -> TenthResult<(Option<(String, String)>, Vec<String>)> {
        if matches!(self.peek_kind(), TokenKind::RParen) {
            self.advance();
            return Ok((None, Vec::new()));
        }

        // Check if this is a named-field pattern: `name: bind_name`
        let first = self.expect_ident()?;
        if matches!(self.peek_kind(), TokenKind::Colon) {
            // Named field binding: `value: v`
            self.advance();
            let bname = self.expect_ident()?;
            self.expect(TokenKind::RParen)?;
            Ok((Some((first.name, bname.name)), Vec::new()))
        } else {
            // Tuple-style binding: positional identifiers
            let mut tuple_fields = vec![first.name];
            while matches!(self.peek_kind(), TokenKind::Comma) {
                self.advance();
                let bind = self.expect_ident()?;
                tuple_fields.push(bind.name);
            }
            self.expect(TokenKind::RParen)?;
            Ok((None, tuple_fields))
        }
    }
}
