//! 表达式解析：`parse_expr` / `parse_primary` / `parse_postfix` / `parse_unary` / `parse_binary` 等。
//!
//! 从 `parser.rs` 拆出（架构重构 T3d）。包含：
//! - `parse_expr`：表达式入口（公开 API）
//! - `parse_primary`：字面量 / 标识符 / 结构体字面量 / 枚举字面量 / 元组 / 块 / if / match / 闭包
//! - `parse_postfix`：后缀操作（泛型调用 / 函数调用 / 方法调用 / 字段 / 索引 / Try）
//! - `parse_unary`：一元操作（Neg / Not / Ref / MutRef / Await / Spawn / Move / Deref）
//! - `parse_binary`：二元操作（含范围 / 赋值 / 复合赋值）
//! - `parse_tensor_or_array_literal`：张量/数组字面量
//! - 辅助：`looks_like_generic_call` / `looks_like_generic_struct_literal` / `looks_like_named_enum_fields`
//! - 辅助：`binop_precedence` / `token_to_binop` / `parse_closure_after_pipe` / `parse_arg_list`
//! - 辅助：`parse_block_or_expr` / `parse_qualified_path` / `expect_ident`

use crate::error::{TenthError, TenthResult};
use crate::hir::types::BaseType;
use crate::lexer::token::{Span, Token, TokenKind};
use super::ast::*;
use super::parser::Parser;

impl Parser {
    pub(super) fn parse_primary(&mut self) -> TenthResult<Expr> {
        let token = self.advance();
        let span = token.span.clone();
        let expr_span = token.span.clone();

        let kind = match &token.kind {
            TokenKind::IntLiteral(n, dt) => ExprKind::Literal(Literal::Int(*n, *dt)),
            TokenKind::FloatLiteral(n, dt) => ExprKind::Literal(Literal::Float(*n, *dt)),
            TokenKind::True => ExprKind::Literal(Literal::Bool(true)),
            TokenKind::False => ExprKind::Literal(Literal::Bool(false)),
            TokenKind::StringLiteral(s) => ExprKind::Literal(Literal::String(s.clone())),
            TokenKind::RawString(s) => ExprKind::Literal(Literal::String(s.clone())),
            TokenKind::MultiLineString(s) => ExprKind::Literal(Literal::String(s.clone())),
            TokenKind::CharLiteral(c) => ExprKind::Literal(Literal::Char(*c)),
            TokenKind::InterpolatedString(parts) => {
                let interp_parts: Vec<InterpPart> = parts.iter().map(|p| match p {
                    crate::lexer::token::StringPart::Literal(s) => InterpPart::Literal(s.clone()),
                    crate::lexer::token::StringPart::Expr(e) => InterpPart::Expr(e.clone()),
                }).collect();
                ExprKind::InterpolatedString(interp_parts)
            }
            TokenKind::FString(parts) => {
                let interp_parts: Vec<InterpPart> = parts.iter().map(|p| match p {
                    crate::lexer::token::StringPart::Literal(s) => InterpPart::Literal(s.clone()),
                    crate::lexer::token::StringPart::Expr(e) => InterpPart::Expr(e.clone()),
                }).collect();
                ExprKind::FString(interp_parts)
            }
            TokenKind::Self_ => ExprKind::Ident(Ident { name: "self".to_string(), span }),
            TokenKind::Identifier(name) => {
                let name = name.clone();
                if matches!(self.peek_kind(), TokenKind::Lt) && self.looks_like_generic_struct_literal() {
                    self.advance();
                    let mut generic_args = Vec::new();
                    while !matches!(self.peek_kind(), TokenKind::Gt | TokenKind::Shr) {
                        generic_args.push(self.parse_type()?);
                        if !matches!(self.peek_kind(), TokenKind::Comma) {
                            break;
                        }
                        self.advance();
                    }
                    self.expect_gt()?;
                    self.expect(TokenKind::LBrace)?;
                    let mut fields = Vec::new();
                    let mut use_defaults = false;
                    while !matches!(self.peek_kind(), TokenKind::RBrace) {
                        if matches!(self.peek_kind(), TokenKind::DotDot) {
                            self.advance();
                            use_defaults = true;
                            break;
                        }
                        let field_name = self.expect_ident()?;
                        self.expect(TokenKind::Colon)?;
                        let value = self.parse_expr()?;
                        fields.push((field_name, value));
                        if !matches!(self.peek_kind(), TokenKind::Comma) {
                            break;
                        }
                        self.advance();
                    }
                    self.expect(TokenKind::RBrace)?;
                    ExprKind::StructLiteral { name: Ident { name, span }, generics: generic_args, fields, use_defaults }
                } else if matches!(self.peek_kind(), TokenKind::LBrace) {
                    let is_struct = self.tokens.get(self.pos + 1)
                        .map(|t| matches!(t.kind, TokenKind::Identifier(_) | TokenKind::RBrace | TokenKind::DotDot))
                        .unwrap_or(false);
                    let has_colon = if is_struct && matches!(self.tokens.get(self.pos + 1).map(|t| &t.kind), Some(TokenKind::RBrace) | Some(TokenKind::DotDot)) {
                        true
                    } else if is_struct {
                        self.tokens.get(self.pos + 2)
                            .map(|t| matches!(t.kind, TokenKind::Colon))
                            .unwrap_or(false)
                    } else {
                        false
                    };

                    if has_colon || (is_struct && matches!(self.tokens.get(self.pos + 1).map(|t| &t.kind), Some(TokenKind::RBrace) | Some(TokenKind::DotDot))) {
                        let ident = Ident { name, span };
                        self.advance();
                        let mut fields = Vec::new();
                        let mut use_defaults = false;
                        while !matches!(self.peek_kind(), TokenKind::RBrace) {
                            if matches!(self.peek_kind(), TokenKind::DotDot) {
                                self.advance();
                                use_defaults = true;
                                break;
                            }
                            let field_name = self.expect_ident()?;
                            self.expect(TokenKind::Colon)?;
                            let value = self.parse_expr()?;
                            fields.push((field_name, value));
                            if !matches!(self.peek_kind(), TokenKind::Comma) {
                                break;
                            }
                            self.advance();
                        }
                        self.expect(TokenKind::RBrace)?;
                        ExprKind::StructLiteral { name: ident, generics: Vec::new(), fields, use_defaults }
                    } else {
                        ExprKind::Ident(Ident { name, span })
                    }
                } else if matches!(self.peek_kind(), TokenKind::Lt) && self.looks_like_generic_enum_construction() {
                    // M2.1：泛型枚举构造 `MyEnum<i64>::Some(5)` — 显式类型实参
                    self.advance();
                    let mut generic_args = Vec::new();
                    while !matches!(self.peek_kind(), TokenKind::Gt | TokenKind::Shr) {
                        generic_args.push(self.parse_type()?);
                        if !matches!(self.peek_kind(), TokenKind::Comma) {
                            break;
                        }
                        self.advance();
                    }
                    self.expect_gt()?;
                    self.expect(TokenKind::ColonColon)?;
                    let enum_name = Ident { name: name.clone(), span: span.clone() };
                    let variant_name = self.expect_ident()?;
                    self.parse_enum_variant_expr(&enum_name, &variant_name, generic_args)?
                } else if matches!(self.peek_kind(), TokenKind::ColonColon) {
                    let enum_name = Ident { name: name.clone(), span: span.clone() };
                    self.advance();
                    let variant_name = self.expect_ident()?;
                    self.parse_enum_variant_expr(&enum_name, &variant_name, Vec::new())?
                } else {
                    ExprKind::Ident(Ident { name, span })
                }
            }
            TokenKind::LParen => {
                // Handle () as unit literal (empty tuple)
                if matches!(self.peek_kind(), TokenKind::RParen) {
                    self.advance();
                    return Ok(Expr {
                        kind: ExprKind::Tuple(vec![]),
                        span: self.span(),
                    });
                }
                let expr = self.parse_expr()?;
                if matches!(self.peek_kind(), TokenKind::Comma) {
                    // Tuple expression: (expr, expr, ...)
                    let mut elems = vec![expr];
                    while matches!(self.peek_kind(), TokenKind::Comma) {
                        self.advance();
                        if matches!(self.peek_kind(), TokenKind::RParen) {
                            break;
                        }
                        elems.push(self.parse_expr()?);
                    }
                    self.expect(TokenKind::RParen)?;
                    return Ok(Expr {
                        kind: ExprKind::Tuple(elems),
                        span: self.span(),
                    });
                }
                self.expect(TokenKind::RParen)?;
                return Ok(expr);
            }
            TokenKind::LBrace => {
                let stmts = self.parse_block_stmts()?;
                ExprKind::Block(stmts)
            }
            TokenKind::If => {
                let cond = self.parse_expr()?;
                let then_branch = self.parse_expr()?;
                let has_else = self.match_token(TokenKind::Else);
                let else_branch = if has_else {
                    Some(Box::new(self.parse_expr()?))
                } else {
                    None
                };
                ExprKind::If {
                    cond: Box::new(cond),
                    then_branch: Box::new(then_branch),
                    else_branch,
                }
            }
            TokenKind::Try => {
                let block = self.parse_expr()?;
                ExprKind::TryBlock(Box::new(block))
            }
            TokenKind::Pipe => {
                return self.parse_closure_after_pipe(span);
            }
            TokenKind::Match => {
                let scrutinee = self.parse_expr()?;
                self.expect(TokenKind::LBrace)?;
                let mut arms = Vec::new();
                while !matches!(self.peek_kind(), TokenKind::RBrace) {
                    let pattern = self.parse_match_pattern()?;
                    // Parse optional guard: `if condition`
                    let guard = if matches!(self.peek_kind(), TokenKind::If) {
                        self.advance();
                        Some(self.parse_expr()?)
                    } else {
                        None
                    };
                    self.expect(TokenKind::FatArrow)?;
                    let body = self.parse_expr()?;
                    arms.push(MatchArm { pattern, guard, body });
                    if matches!(self.peek_kind(), TokenKind::Comma) {
                        self.advance();
                    }
                }
                self.expect(TokenKind::RBrace)?;
                ExprKind::Match {
                    scrutinee: Box::new(scrutinee),
                    arms,
                }
            }
            TokenKind::LBracket => {
                return self.parse_tensor_or_array_literal(span);
            }
            _ => {
                return Err(TenthError::ParseError {
                    line: span.line,
                    col: span.col,
                    message: format!("意外的标记：{}", token.kind),
                });
            }
        };

        Ok(Expr { kind, span: expr_span })
    }

    pub(super) fn parse_tensor_or_array_literal(&mut self, span: Span) -> TenthResult<Expr> {
        if matches!(self.peek_kind(), TokenKind::LBracket) {
            let mut rows: Vec<Vec<Expr>> = Vec::new();
            loop {
                if matches!(self.peek_kind(), TokenKind::RBracket) {
                    break;
                }
                self.expect(TokenKind::LBracket)?;
                let mut row = Vec::new();
                loop {
                    if matches!(self.peek_kind(), TokenKind::RBracket) {
                        break;
                    }
                    row.push(self.parse_expr()?);
                    if !matches!(self.peek_kind(), TokenKind::Comma) {
                        break;
                    }
                    self.advance();
                }
                self.expect(TokenKind::RBracket)?;
                rows.push(row);
                if !matches!(self.peek_kind(), TokenKind::Comma) {
                    break;
                }
                self.advance();
            }
            self.expect(TokenKind::RBracket)?;
            Ok(Expr {
                kind: ExprKind::TensorLiteral(rows),
                span,
            })
        } else {
            let mut elements: Vec<Expr> = Vec::new();
            loop {
                if matches!(self.peek_kind(), TokenKind::RBracket) {
                    break;
                }
                elements.push(self.parse_expr()?);
                if !matches!(self.peek_kind(), TokenKind::Comma) {
                    break;
                }
                self.advance();
            }
            self.expect(TokenKind::RBracket)?;
            Ok(Expr {
                kind: ExprKind::ArrayLiteral(elements),
                span,
            })
        }
    }

    pub(super) fn parse_postfix(&mut self) -> TenthResult<Expr> {
        let mut expr = self.parse_primary()?;

        loop {
            let span = self.span();
            match self.peek_kind() {
                TokenKind::Lt => {
                    if !self.looks_like_generic_call() {
                        break;
                    }
                    self.advance();
                    let mut generics = Vec::new();
                    while !matches!(self.peek_kind(), TokenKind::Gt | TokenKind::Shr) {
                        generics.push(self.parse_type()?);
                        if !matches!(self.peek_kind(), TokenKind::Comma) {
                            break;
                        }
                        self.advance();
                    }
                    self.expect_gt()?;

                    if matches!(self.peek_kind(), TokenKind::LParen) {
                        self.advance();
                        let mut args = Vec::new();
                        if !matches!(self.peek_kind(), TokenKind::RParen) {
                            args = self.parse_arg_list()?;
                        }
                        self.expect(TokenKind::RParen)?;
                        let expr_span = expr.span.clone();
                        expr = Expr {
                            kind: ExprKind::GenericCall {
                                func: Box::new(expr),
                                generics,
                                args,
                            },
                            span: expr_span,
                        };
                        continue;
                    }
                }
                TokenKind::LParen => {
                    self.advance();
                    let mut args = Vec::new();
                    if !matches!(self.peek_kind(), TokenKind::RParen) {
                        args = self.parse_arg_list()?;
                    }
                    self.expect(TokenKind::RParen)?;
                    let expr_span = expr.span.clone();
                    expr = Expr {
                        kind: ExprKind::Call {
                            func: Box::new(expr),
                            args,
                        },
                        span: expr_span,
                    };
                }
                TokenKind::Dot => {
                    self.advance();
                    if let TokenKind::Identifier(name) = &self.peek().kind {
                        let name = name.clone();
                        let method_span = self.peek().span.clone();
                        self.advance();

                        if matches!(self.peek_kind(), TokenKind::LParen) {
                            self.advance();
                            let mut args = Vec::new();
                            if !matches!(self.peek_kind(), TokenKind::RParen) {
                                args = self.parse_arg_list()?;
                            }
                            self.expect(TokenKind::RParen)?;
                            let expr_span = expr.span.clone();
                            expr = Expr {
                                kind: ExprKind::MethodCall {
                                    receiver: Box::new(expr),
                                    method: Ident { name, span: method_span },
                                    args,
                                },
                                span: expr_span,
                            };
                        } else {
                            let expr_span = expr.span.clone();
                            expr = Expr {
                                kind: ExprKind::Field {
                                    target: Box::new(expr),
                                    field: Ident { name, span: method_span },
                                },
                                span: expr_span,
                            };
                        }
                    } else {
                        return Err(TenthError::ParseError {
                            line: span.line,
                            col: span.col,
                            message: "'.' 后面期望标识符".into(),
                        });
                    }
                }
                TokenKind::LBracket => {
                    self.advance();
                    if matches!(self.peek_kind(), TokenKind::LBracket) {
                        let tensor_lit = self.parse_tensor_or_array_literal(self.span())?;
                        let expr_span = expr.span.clone();
                        expr = Expr {
                            kind: ExprKind::Call {
                                func: Box::new(expr),
                                args: vec![tensor_lit],
                            },
                            span: expr_span,
                        };
                    } else {
                        let mut indices = Vec::new();
                        loop {
                            if matches!(self.peek_kind(), TokenKind::RBracket) {
                                break;
                            }
                            if matches!(self.peek_kind(), TokenKind::Colon) {
                                self.advance();
                                indices.push(IndexExpr::Colon);
                            } else if matches!(self.peek_kind(), TokenKind::DotDot) {
                                self.advance();
                                let end = if !matches!(self.peek_kind(), TokenKind::Comma)
                                    && !matches!(self.peek_kind(), TokenKind::RBracket)
                                {
                                    Some(Box::new(self.parse_expr()?))
                                } else {
                                    None
                                };
                                indices.push(IndexExpr::Range { start: None, end });
                            } else {
                                let start = self.parse_expr()?;
                                // Check if parse_expr consumed a Range (e.g. 0..5)
                                if let ExprKind::Range { start: rs, end: re, inclusive: _ } = &start.kind {
                                    indices.push(IndexExpr::Range {
                                        start: rs.as_ref().map(|b| Box::new(*b.clone())),
                                        end: re.as_ref().map(|b| Box::new(*b.clone())),
                                    });
                                } else if matches!(self.peek_kind(), TokenKind::DotDot) {
                                    self.advance();
                                    let end = if !matches!(self.peek_kind(), TokenKind::Comma)
                                        && !matches!(self.peek_kind(), TokenKind::RBracket)
                                    {
                                        Some(Box::new(self.parse_expr()?))
                                    } else {
                                        None
                                    };
                                    indices.push(IndexExpr::Range {
                                        start: Some(Box::new(start)),
                                        end,
                                    });
                                } else {
                                    indices.push(IndexExpr::Single(start));
                                }
                            }
                            if !matches!(self.peek_kind(), TokenKind::Comma) {
                                break;
                            }
                            self.advance();
                        }
                        self.expect(TokenKind::RBracket)?;
                        let expr_span = expr.span.clone();
                        expr = Expr {
                            kind: ExprKind::Index {
                                target: Box::new(expr),
                                indices,
                            },
                            span: expr_span,
                        };
                    }
                }
                TokenKind::QuestionMark => {
                    self.advance();
                    let expr_span = expr.span.clone();
                    expr = Expr {
                        kind: ExprKind::Unary { op: UnaryOp::Try, expr: Box::new(expr) },
                        span: expr_span,
                    };
                }
                _ => break,
            }
        }

        Ok(expr)
    }

    pub(super) fn looks_like_generic_call(&self) -> bool {
        let mut i = self.pos + 1;
        if i >= self.tokens.len() {
            return false;
        }
        if !matches!(&self.tokens[i].kind, TokenKind::Identifier(_)) {
            return false;
        }
        i += 1;
        while i < self.tokens.len() {
            if matches!(&self.tokens[i].kind, TokenKind::Comma) {
                i += 1;
                if i >= self.tokens.len() || !matches!(&self.tokens[i].kind, TokenKind::Identifier(_)) {
                    return false;
                }
                i += 1;
            } else {
                break;
            }
        }
        if i >= self.tokens.len() || !matches!(&self.tokens[i].kind, TokenKind::Gt) {
            return false;
        }
        i += 1;
        if i >= self.tokens.len() {
            return false;
        }
        matches!(&self.tokens[i].kind, TokenKind::LParen)
    }

    pub(super) fn looks_like_generic_struct_literal(&self) -> bool {
        let mut i = self.pos + 1;
        if i >= self.tokens.len() {
            return false;
        }
        if !matches!(&self.tokens[i].kind, TokenKind::Identifier(_)) {
            return false;
        }
        i += 1;
        while i < self.tokens.len() {
            if matches!(&self.tokens[i].kind, TokenKind::Comma) {
                i += 1;
                if i >= self.tokens.len() || !matches!(&self.tokens[i].kind, TokenKind::Identifier(_)) {
                    return false;
                }
                i += 1;
            } else {
                break;
            }
        }
        if i >= self.tokens.len() || !matches!(&self.tokens[i].kind, TokenKind::Gt) {
            return false;
        }
        i += 1;
        if i >= self.tokens.len() {
            return false;
        }
        matches!(&self.tokens[i].kind, TokenKind::LBrace)
    }

    /// 泛型枚举构造检测：`Name<TypeArgs>::Variant`（匹配的 `>` 后跟 `::`）。
    /// 与 looks_like_generic_struct_literal（`>` 后跟 `{`）和 looks_like_generic_call
    /// （`>` 后跟 `(`）区分。用深度扫描支持嵌套泛型实参（`Wrap<Vec<i64>>::Item`，
    /// 内层 `>>` 被词法化为 Shr）。
    pub(super) fn looks_like_generic_enum_construction(&self) -> bool {
        let mut i = self.pos + 1;
        let mut depth: i32 = 0;
        let mut closed = false;
        while i < self.tokens.len() {
            match &self.tokens[i].kind {
                TokenKind::Lt => depth += 1,
                TokenKind::Gt => {
                    depth -= 1;
                    if depth <= 0 { closed = true; }
                }
                TokenKind::Shr => {
                    // >> 视为两个 >；depth==1 时正好闭合最外层
                    depth -= 2;
                    if depth <= 0 { closed = true; }
                }
                TokenKind::Comma | TokenKind::Identifier(_) => {}
                _ => return false,
            }
            i += 1;
            if closed { break; }
        }
        if !closed {
            return false;
        }
        if i >= self.tokens.len() {
            return false;
        }
        matches!(&self.tokens[i].kind, TokenKind::ColonColon)
    }

    /// 解析 `EnumName::Variant(...)` 的变体构造（M2.1：含泛型枚举 `EnumName<T>::Variant`）。
    /// `generics` 为显式类型实参（非泛型枚举传空 Vec；为空时 lower 端按字段推断）。
    /// 从 parse_primary 的 `::` 分支提取，供裸 `::` 与 `Name<...>::` 两条路径共用。
    pub(super) fn parse_enum_variant_expr(
        &mut self,
        enum_name: &Ident,
        variant_name: &Ident,
        generics: Vec<TypeAnnotation>,
    ) -> TenthResult<ExprKind> {
        let path_name = format!("{}::{}", enum_name.name, variant_name.name);
        if matches!(self.peek_kind(), TokenKind::LParen) {
            // Check if next token is RParen (empty parens → function call, not enum)
            let next_is_rparen = self.tokens.get(self.pos + 1)
                .map_or(false, |t| matches!(t.kind, TokenKind::RParen));
            if next_is_rparen {
                // Empty parens: treat as function call, e.g. HashMap::new()
                Ok(ExprKind::Ident(Ident {
                    name: path_name,
                    span: enum_name.span.clone(),
                }))
            } else {
                let next_is_ident = self.tokens.get(self.pos + 1)
                    .map_or(false, |t| matches!(t.kind, TokenKind::Identifier(_)));
                // Check if identifier is followed by `:` → named-field enum construction
                let is_named_field = next_is_ident && self.tokens.get(self.pos + 2)
                    .map_or(false, |t| matches!(t.kind, TokenKind::Colon));
                if is_named_field {
                    self.advance();
                    let mut fields = Vec::new();
                    while !matches!(self.peek_kind(), TokenKind::RParen) {
                        let fname = self.expect_ident()?;
                        self.expect(TokenKind::Colon)?;
                        let val = self.parse_expr()?;
                        fields.push((fname, val));
                        if !matches!(self.peek_kind(), TokenKind::Comma) {
                            break;
                        }
                        self.advance();
                    }
                    self.expect(TokenKind::RParen)?;
                    Ok(ExprKind::EnumLiteral {
                        enum_name: enum_name.clone(),
                        variant: variant_name.clone(),
                        fields,
                        generics,
                    })
                } else if next_is_ident {
                    // Identifier without colon → function call, e.g. math::add(x, y)
                    Ok(ExprKind::Ident(Ident {
                        name: path_name,
                        span: enum_name.span.clone(),
                    }))
                } else if self.known_enums.contains(&enum_name.name) {
                    // Known enum constructor with positional arg: Some(42)
                    self.advance();
                    let mut fields = Vec::new();
                    let mut field_idx = 0;
                    while !matches!(self.peek_kind(), TokenKind::RParen) {
                        let val = self.parse_expr()?;
                        let fname = Ident {
                            name: format!("_{}", field_idx),
                            span: val.span.clone(),
                        };
                        fields.push((fname, val));
                        field_idx += 1;
                        if !matches!(self.peek_kind(), TokenKind::Comma) {
                            break;
                        }
                        self.advance();
                    }
                    self.expect(TokenKind::RParen)?;
                    Ok(ExprKind::EnumLiteral {
                        enum_name: enum_name.clone(),
                        variant: variant_name.clone(),
                        fields,
                        generics,
                    })
                } else {
                    // Unknown name with positional args → function call
                    Ok(ExprKind::Ident(Ident {
                        name: path_name,
                        span: enum_name.span.clone(),
                    }))
                }
            }
        } else if matches!(self.peek_kind(), TokenKind::LBrace) {
            self.advance();
            let mut fields = Vec::new();
            while !matches!(self.peek_kind(), TokenKind::RBrace) {
                let fname = self.expect_ident()?;
                self.expect(TokenKind::Colon)?;
                let val = self.parse_expr()?;
                fields.push((fname, val));
                if !matches!(self.peek_kind(), TokenKind::Comma) {
                    break;
                }
                self.advance();
            }
            self.expect(TokenKind::RBrace)?;
            Ok(ExprKind::EnumLiteral {
                enum_name: enum_name.clone(),
                variant: variant_name.clone(),
                fields,
                generics,
            })
        } else {
            // Unit variant: TokenKind::Eof, Option::None, etc.
            Ok(ExprKind::EnumLiteral {
                enum_name: enum_name.clone(),
                variant: variant_name.clone(),
                fields: Vec::new(),
                generics,
            })
        }
    }

    /// Look ahead inside enum variant parentheses to determine if fields are named.
    /// Named: `name: Type, ...` — first token is Identifier, second is Colon.
    /// Tuple: `Type, ...` — first token is Identifier but second is NOT Colon,
    ///        or first token is a keyword type like `i64`, `str`, etc.
    pub(super) fn looks_like_named_enum_fields(&self) -> bool {
        let i = self.pos;
        // Skip past the LParen (already consumed), so we're at the first content token
        // Check: is the first token an Identifier followed by Colon?
        if i >= self.tokens.len() {
            return false;
        }
        // Empty parens: neither
        if matches!(&self.tokens[i].kind, TokenKind::RParen) {
            return false;
        }
        // If first token is Identifier and second is Colon → named fields
        if let TokenKind::Identifier(_) = &self.tokens[i].kind {
            if i + 1 < self.tokens.len() && matches!(self.tokens[i + 1].kind, TokenKind::Colon) {
                return true;
            }
        }
        false
    }

    pub(super) fn parse_unary(&mut self) -> TenthResult<Expr> {
        let span = self.span();
        match self.peek_kind() {
            TokenKind::Minus => {
                self.advance();
                let expr = self.parse_unary()?;
                Ok(Expr {
                    kind: ExprKind::Unary { op: UnaryOp::Neg, expr: Box::new(expr) },
                    span,
                })
            }
            TokenKind::Not => {
                self.advance();
                let expr = self.parse_unary()?;
                Ok(Expr {
                    kind: ExprKind::Unary { op: UnaryOp::Not, expr: Box::new(expr) },
                    span,
                })
            }
            TokenKind::Ampersand => {
                self.advance();
                if matches!(self.peek_kind(), TokenKind::Mut) {
                    self.advance();
                    let expr = self.parse_unary()?;
                    Ok(Expr {
                        kind: ExprKind::MutRef(Box::new(expr)),
                        span,
                    })
                } else {
                    let expr = self.parse_unary()?;
                    Ok(Expr {
                        kind: ExprKind::Ref(Box::new(expr)),
                        span,
                    })
                }
            }
            TokenKind::Await => {
                self.advance();
                let expr = self.parse_unary()?;
                Ok(Expr {
                    kind: ExprKind::Await(Box::new(expr)),
                    span,
                })
            }
            TokenKind::Spawn => {
                self.advance();
                let expr = self.parse_unary()?;
                Ok(Expr {
                    kind: ExprKind::Spawn(Box::new(expr)),
                    span,
                })
            }
            TokenKind::Yield => {
                // yield [expr]
                // 无值形式：`yield;` / `yield)` / `yield}` / `yield,` / `yield <EOF>`
                // 带值形式：`yield expr`（expr 由 parse_unary 解析）
                self.advance();
                let inner = match self.peek_kind() {
                    TokenKind::Semicolon | TokenKind::RParen | TokenKind::RBrace
                    | TokenKind::Comma | TokenKind::Eof => None,
                    _ => Some(Box::new(self.parse_unary()?)),
                };
                Ok(Expr {
                    kind: ExprKind::Yield(inner),
                    span,
                })
            }
            TokenKind::Move => {
                self.advance();
                let expr = self.parse_unary()?;
                Ok(Expr {
                    kind: ExprKind::Move(Box::new(expr)),
                    span,
                })
            }
            TokenKind::Lossy => {
                // `lossy expr`：编译期标记"我确认这里可能算错，接受该污点"；
                // 运行时求值 inner（no-op）。`lossy(expr)` 同样成立（括号是普通分组）。
                self.advance();
                let expr = self.parse_unary()?;
                Ok(Expr {
                    kind: ExprKind::Lossy(Box::new(expr)),
                    span,
                })
            }
            TokenKind::Star => {
                self.advance();
                let expr = self.parse_unary()?;
                Ok(Expr {
                    kind: ExprKind::Deref(Box::new(expr)),
                    span,
                })
            }
            _ => self.parse_postfix(),
        }
    }

    pub(super) fn binop_precedence(kind: &TokenKind) -> u8 {
        match kind {
            TokenKind::Star | TokenKind::Slash | TokenKind::Percent => 5,
            TokenKind::Plus | TokenKind::Minus => 4,
            TokenKind::Lt | TokenKind::Gt | TokenKind::LtEq | TokenKind::GtEq => 3,
            TokenKind::EqEq | TokenKind::NotEq => 2,
            TokenKind::AndAnd => 1,
            TokenKind::OrOr => 0,
            TokenKind::DotDot | TokenKind::DotDotEq => 0,  // range operator: lowest precedence
            // M3.1：自定义运算符统一默认优先级 4（与 `+`/`-` 同级，左结合）。
            // 最小版本不提供 per-operator 优先级声明，文档标注。
            TokenKind::CustomOperator(_) => 4,
            TokenKind::Assign
            | TokenKind::PlusAssign
            | TokenKind::MinusAssign
            | TokenKind::StarAssign
            | TokenKind::SlashAssign => 1,
            _ => 255,
        }
    }

    pub(super) fn token_to_binop(kind: &TokenKind) -> Option<BinOp> {
        match kind {
            TokenKind::Plus => Some(BinOp::Add),
            TokenKind::Minus => Some(BinOp::Sub),
            TokenKind::Star => Some(BinOp::Mul),
            TokenKind::Slash => Some(BinOp::Div),
            TokenKind::Percent => Some(BinOp::Mod),
            TokenKind::EqEq => Some(BinOp::Eq),
            TokenKind::NotEq => Some(BinOp::NotEq),
            TokenKind::Lt => Some(BinOp::Lt),
            TokenKind::Gt => Some(BinOp::Gt),
            TokenKind::LtEq => Some(BinOp::LtEq),
            TokenKind::GtEq => Some(BinOp::GtEq),
            TokenKind::AndAnd => Some(BinOp::And),
            TokenKind::OrOr => Some(BinOp::Or),
            _ => None,
        }
    }

    pub(super) fn parse_binary(&mut self, min_prec: u8) -> TenthResult<Expr> {
        let mut left = self.parse_unary()?;

        loop {
            let prec = Self::binop_precedence(self.peek_kind());
            if prec < min_prec || prec == 255 {
                break;
            }

            // Range expressions: start..end, start..=end
            if matches!(self.peek_kind(), TokenKind::DotDot) || matches!(self.peek_kind(), TokenKind::DotDotEq) {
                let inclusive = if matches!(self.peek_kind(), TokenKind::DotDotEq) {
                    self.advance();
                    true
                } else {
                    self.advance();
                    self.match_token(TokenKind::Assign) // ..=  (when lexer splits)
                };
                let end = if !matches!(self.peek_kind(), TokenKind::Semicolon)
                    && !matches!(self.peek_kind(), TokenKind::RBrace)
                    && !matches!(self.peek_kind(), TokenKind::RParen)
                    && !matches!(self.peek_kind(), TokenKind::Comma)
                    && !matches!(self.peek_kind(), TokenKind::RBracket)
                {
                    Some(Box::new(self.parse_expr()?))
                } else {
                    None
                };
                let left_span = left.span.clone();
                left = Expr {
                    kind: ExprKind::Range {
                        start: Some(Box::new(left)),
                        end,
                        inclusive,
                    },
                    span: left_span,
                };
                continue;
            }

            if matches!(self.peek_kind(), TokenKind::Assign) {
                self.advance();
                let right = self.parse_expr()?;
                let left_span = left.span.clone();
                left = Expr {
                    kind: ExprKind::Assign {
                        target: Box::new(left),
                        value: Box::new(right),
                    },
                    span: left_span,
                };
                continue;
            }

            if matches!(self.peek_kind(), TokenKind::PlusAssign)
                || matches!(self.peek_kind(), TokenKind::MinusAssign)
                || matches!(self.peek_kind(), TokenKind::StarAssign)
                || matches!(self.peek_kind(), TokenKind::SlashAssign)
            {
                let op = match self.peek_kind() {
                    TokenKind::PlusAssign => BinOp::Add,
                    TokenKind::MinusAssign => BinOp::Sub,
                    TokenKind::StarAssign => BinOp::Mul,
                    TokenKind::SlashAssign => BinOp::Div,
                    _ => unreachable!(),
                };
                self.advance();
                let right = self.parse_expr()?;
                let left_span = left.span.clone();
                left = Expr {
                    kind: ExprKind::AssignOp {
                        target: Box::new(left),
                        op,
                        value: Box::new(right),
                    },
                    span: left_span,
                };
                continue;
            }

            let op_kind = self.peek_kind().clone();
            // M3.1：自定义运算符中缀表达式。
            // 声明见 `operator <op> = fn(...)`；此处把 `a <op> b` 解析为
            // ExprKind::CustomBinary，lower 阶段降级为对绑定函数的调用。
            // 未声明的运算符在 lower 时报错（"未声明的运算符"）。
            if let TokenKind::CustomOperator(op) = &op_kind {
                let op = op.clone();
                self.advance();
                let right = self.parse_binary(prec + 1)?;
                let left_span = left.span.clone();
                left = Expr {
                    kind: ExprKind::CustomBinary {
                        op,
                        left: Box::new(left),
                        right: Box::new(right),
                    },
                    span: left_span,
                };
                continue;
            }
            self.advance();
            let op = Self::token_to_binop(&op_kind).unwrap();
            let right = self.parse_binary(prec + 1)?;
            let left_span = left.span.clone();

            left = Expr {
                kind: ExprKind::Binary {
                    op,
                    left: Box::new(left),
                    right: Box::new(right),
                },
                span: left_span,
            };
        }

        Ok(left)
    }

    pub fn parse_expr(&mut self) -> TenthResult<Expr> {
        self.parse_binary(0)
    }

    pub(super) fn parse_closure_after_pipe(&mut self, span: Span) -> TenthResult<Expr> {
        let mut params: Vec<(Ident, Option<TypeAnnotation>)> = Vec::new();

        loop {
            if matches!(self.peek_kind(), TokenKind::Pipe) {
                self.advance();
                break;
            }

            let name = if let TokenKind::Identifier(name) = &self.peek().kind {
                Ident {
                    name: name.clone(),
                    span: self.peek().span.clone(),
                }
            } else {
                return Err(TenthError::ParseError {
                    line: self.peek().span.line,
                    col: self.peek().span.col,
                    message: "期望参数名".into(),
                });
            };
            self.advance();

            let type_ann = if matches!(self.peek_kind(), TokenKind::Colon) {
                self.advance();
                Some(self.parse_type()?)
            } else {
                None
            };

            params.push((name, type_ann));

            if matches!(self.peek_kind(), TokenKind::Comma) {
                self.advance();
            }
        }

        let body = self.parse_expr()?;

        Ok(Expr {
            kind: ExprKind::Closure {
                params,
                body: Box::new(body),
            },
            span,
        })
    }

    pub(super) fn parse_arg_list(&mut self) -> TenthResult<Vec<Expr>> {
        let mut args = Vec::new();
        loop {
            // Check for named argument: `name = expr`
            // We look ahead: if the current token is an Identifier and the
            // next token is `=` (but not `==`), treat it as a named arg.
            let is_named = matches!(self.peek_kind(), TokenKind::Identifier(_))
                && self.tokens.get(self.pos + 1)
                    .map(|t| matches!(t.kind, TokenKind::Assign))
                    .unwrap_or(false);

            if is_named {
                let name = self.expect_ident()?;
                self.advance(); // consume `=`
                let value = self.parse_expr()?;
                args.push(Expr {
                    kind: ExprKind::NamedArg {
                        name,
                        value: Box::new(value),
                    },
                    span: self.span(),
                });
            } else {
                args.push(self.parse_expr()?);
            }

            if !matches!(self.peek_kind(), TokenKind::Comma) {
                break;
            }
            self.advance();
        }
        Ok(args)
    }

    pub(super) fn parse_block_or_expr(&mut self) -> TenthResult<Expr> {
        if matches!(self.peek_kind(), TokenKind::LBrace) {
            self.advance();
            let stmts = self.parse_block_stmts()?;
            Ok(Expr {
                kind: ExprKind::Block(stmts),
                span: self.span(),
            })
        } else {
            self.parse_expr()
        }
    }

    pub(super) fn parse_qualified_path(&mut self) -> TenthResult<String> {
        let first = self.expect_ident()?;
        if matches!(self.peek_kind(), TokenKind::ColonColon) {
            self.advance();
            let second = self.expect_ident()?;
            return Ok(format!("{}::{}", first.name, second.name));
        }
        Ok(first.name)
    }

    pub(super) fn expect_ident(&mut self) -> TenthResult<Ident> {
        let span = self.span();
        if let TokenKind::Identifier(name) = &self.peek().kind {
            let ident = Ident {
                name: name.clone(),
                span: span,
            };
            self.advance();
            Ok(ident)
        } else {
            Err(TenthError::ParseError {
                line: span.line,
                col: span.col,
                message: "期望标识符".into(),
            })
        }
    }
}
