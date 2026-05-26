use crate::error::{TenthError, TenthResult};
use crate::lexer::token::{Span, Token, TokenKind};
use super::ast::*;
use std::collections::HashSet;

pub struct Parser {
    tokens: Vec<Token>,
    pos: usize,
    known_enums: HashSet<String>,
}

static EOF_TOKEN: Token = Token {
    kind: TokenKind::Eof,
    span: Span { line: 0, col: 0 },
};

impl Parser {
    pub fn new(tokens: Vec<Token>) -> Self {
        let mut known_enums = HashSet::new();
        known_enums.insert("Option".to_string());
        known_enums.insert("Result".to_string());
        Parser { tokens, pos: 0, known_enums }
    }

    fn peek(&self) -> &Token {
        self.tokens.get(self.pos).unwrap_or(&EOF_TOKEN)
    }

    fn peek_kind(&self) -> &TokenKind {
        &self.peek().kind
    }

    fn advance(&mut self) -> &Token {
        let idx = self.pos;
        self.pos += 1;
        self.tokens.get(idx).unwrap_or(&EOF_TOKEN)
    }

    fn expect(&mut self, kind: TokenKind) -> TenthResult<&Token> {
        let token = self.peek();
        if std::mem::discriminant(&token.kind) == std::mem::discriminant(&kind) {
            self.pos += 1;
            Ok(self.tokens.get(self.pos - 1).unwrap())
        } else {
            Err(TenthError::ParseError {
                line: token.span.line,
                col: token.span.col,
                message: format!("expected {}, got {}", kind, token.kind),
            })
        }
    }

    fn span(&self) -> Span {
        self.peek().span.clone()
    }

    fn at_eof(&self) -> bool {
        matches!(self.peek_kind(), TokenKind::Eof)
    }

    fn match_token(&mut self, kind: TokenKind) -> bool {
        if std::mem::discriminant(self.peek_kind()) == std::mem::discriminant(&kind) {
            self.advance();
            true
        } else {
            false
        }
    }

    fn parse_primary(&mut self) -> TenthResult<Expr> {
        let token = self.advance();
        let span = token.span.clone();
        let expr_span = token.span.clone();

        let kind = match &token.kind {
            TokenKind::IntLiteral(n) => ExprKind::Literal(Literal::Int(*n)),
            TokenKind::FloatLiteral(n) => ExprKind::Literal(Literal::Float(*n)),
            TokenKind::True => ExprKind::Literal(Literal::Bool(true)),
            TokenKind::False => ExprKind::Literal(Literal::Bool(false)),
            TokenKind::StringLiteral(s) => ExprKind::Literal(Literal::String(s.clone())),
            TokenKind::Self_ => ExprKind::Ident(Ident { name: "self".to_string(), span }),
            TokenKind::Identifier(name) => {
                let name = name.clone();
                if matches!(self.peek_kind(), TokenKind::Lt) && self.looks_like_generic_struct_literal() {
                    self.advance();
                    let mut generic_args = Vec::new();
                    while !matches!(self.peek_kind(), TokenKind::Gt) {
                        generic_args.push(self.parse_type()?);
                        if !matches!(self.peek_kind(), TokenKind::Comma) {
                            break;
                        }
                        self.advance();
                    }
                    self.expect(TokenKind::Gt)?;
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
                        .map(|t| matches!(t.kind, TokenKind::Identifier(_) | TokenKind::RBrace))
                        .unwrap_or(false);
                    let has_colon = if is_struct && matches!(self.tokens.get(self.pos + 1).map(|t| &t.kind), Some(TokenKind::RBrace)) {
                        true
                    } else if is_struct {
                        self.tokens.get(self.pos + 2)
                            .map(|t| matches!(t.kind, TokenKind::Colon))
                            .unwrap_or(false)
                    } else {
                        false
                    };

                    if has_colon || (is_struct && matches!(self.tokens.get(self.pos + 1).map(|t| &t.kind), Some(TokenKind::RBrace))) {
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
                } else if matches!(self.peek_kind(), TokenKind::ColonColon) {
                    let enum_name = Ident { name: name.clone(), span: span.clone() };
                    self.advance();
                    let variant_name = self.expect_ident()?;
                    let path_name = format!("{}::{}", enum_name.name, variant_name.name);

                    if matches!(self.peek_kind(), TokenKind::LParen) {
                        // Check if next token is RParen (empty parens → function call, not enum)
                        let next_is_rparen = self.tokens.get(self.pos + 1)
                            .map_or(false, |t| matches!(t.kind, TokenKind::RParen));
                        if next_is_rparen {
                            // Empty parens: treat as function call, e.g. HashMap::new()
                            ExprKind::Ident(Ident {
                                name: path_name,
                                span: enum_name.span,
                            })
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
                            ExprKind::EnumLiteral {
                                enum_name,
                                variant: variant_name,
                                fields,
                            }
                        } else if next_is_ident {
                            // Identifier without colon → function call, e.g. math::add(x, y)
                            ExprKind::Ident(Ident {
                                name: path_name,
                                span: enum_name.span,
                            })
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
                            ExprKind::EnumLiteral {
                                enum_name,
                                variant: variant_name,
                                fields,
                            }
                        } else {
                            // Unknown name with positional args → function call
                            ExprKind::Ident(Ident {
                                name: path_name,
                                span: enum_name.span,
                            })
                        }
                        } // close if next_is_rparen else
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
                        ExprKind::EnumLiteral {
                            enum_name,
                            variant: variant_name,
                            fields,
                        }
                    } else {
                        // Unit variant: TokenKind::Eof, Option::None, etc.
                        ExprKind::EnumLiteral {
                            enum_name,
                            variant: variant_name,
                            fields: Vec::new(),
                        }
                    }
                } else {
                    ExprKind::Ident(Ident { name, span })
                }
            }
            TokenKind::LParen => {
                let expr = self.parse_expr()?;
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
            TokenKind::Pipe => {
                return self.parse_closure_after_pipe(span);
            }
            TokenKind::Match => {
                let scrutinee = self.parse_expr()?;
                self.expect(TokenKind::LBrace)?;
                let mut arms = Vec::new();
                while !matches!(self.peek_kind(), TokenKind::RBrace) {
                    let pattern = self.parse_match_pattern()?;
                    self.expect(TokenKind::FatArrow)?;
                    let body = self.parse_expr()?;
                    arms.push(MatchArm { pattern, body });
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
                    message: format!("unexpected token: {}", token.kind),
                });
            }
        };

        Ok(Expr { kind, span: expr_span })
    }

    fn parse_tensor_or_array_literal(&mut self, span: Span) -> TenthResult<Expr> {
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

    fn parse_postfix(&mut self) -> TenthResult<Expr> {
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
                    while !matches!(self.peek_kind(), TokenKind::Gt) {
                        generics.push(self.parse_type()?);
                        if !matches!(self.peek_kind(), TokenKind::Comma) {
                            break;
                        }
                        self.advance();
                    }
                    self.expect(TokenKind::Gt)?;

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
                            message: "expected identifier after '.'".into(),
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
                                if matches!(self.peek_kind(), TokenKind::DotDot) {
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
                _ => break,
            }
        }

        Ok(expr)
    }

    fn looks_like_generic_call(&self) -> bool {
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

    fn looks_like_generic_struct_literal(&self) -> bool {
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

    fn parse_unary(&mut self) -> TenthResult<Expr> {
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
            TokenKind::Move => {
                self.advance();
                let expr = self.parse_unary()?;
                Ok(Expr {
                    kind: ExprKind::Move(Box::new(expr)),
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

    fn binop_precedence(kind: &TokenKind) -> u8 {
        match kind {
            TokenKind::Star | TokenKind::Slash | TokenKind::Percent => 5,
            TokenKind::Plus | TokenKind::Minus => 4,
            TokenKind::Lt | TokenKind::Gt | TokenKind::LtEq | TokenKind::GtEq => 3,
            TokenKind::EqEq | TokenKind::NotEq => 2,
            TokenKind::AndAnd => 1,
            TokenKind::OrOr => 0,
            TokenKind::Assign
            | TokenKind::PlusAssign
            | TokenKind::MinusAssign
            | TokenKind::StarAssign
            | TokenKind::SlashAssign => 1,
            _ => 255,
        }
    }

    fn token_to_binop(kind: &TokenKind) -> Option<BinOp> {
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

    fn parse_binary(&mut self, min_prec: u8) -> TenthResult<Expr> {
        let mut left = self.parse_unary()?;

        loop {
            let prec = Self::binop_precedence(self.peek_kind());
            if prec < min_prec || prec == 255 {
                break;
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

    fn parse_closure_after_pipe(&mut self, span: Span) -> TenthResult<Expr> {
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
                    message: "expected parameter name".into(),
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

    fn parse_arg_list(&mut self) -> TenthResult<Vec<Expr>> {
        let mut args = Vec::new();
        loop {
            args.push(self.parse_expr()?);
            if !matches!(self.peek_kind(), TokenKind::Comma) {
                break;
            }
            self.advance();
        }
        Ok(args)
    }

    fn parse_type(&mut self) -> TenthResult<TypeAnnotation> {
        let span = self.span();
        match self.peek_kind() {
            TokenKind::Ampersand => {
                self.advance();
                let is_mut = matches!(self.peek_kind(), TokenKind::Mut);
                if is_mut { self.advance(); }
                let inner = self.parse_type()?;
                let name_str = match &inner {
                    TypeAnnotation::Named(id) => {
                        if is_mut { format!("&mut {}", id.name) }
                        else { format!("&{}", id.name) }
                    }
                    _ => if is_mut { "&mut _".to_string() } else { "&_".to_string() },
                };
                Ok(TypeAnnotation::Named(Ident { name: name_str, span }))
            }
            TokenKind::Identifier(name) => {
                let name = name.clone();
                self.advance();
                if name == "Tensor" && matches!(self.peek_kind(), TokenKind::LBracket) {
                    self.advance();
                    let dtype = self.parse_type()?;
                    let mut dims = Vec::new();
                    loop {
                        if matches!(self.peek_kind(), TokenKind::RBracket) {
                            break;
                        }
                        if matches!(self.peek_kind(), TokenKind::Comma) {
                            self.advance();
                            continue;
                        }
                        match self.peek_kind() {
                            TokenKind::IntLiteral(n) => {
                                let n = *n;
                                self.advance();
                                dims.push(DimSpec::Literal(n));
                            }
                            TokenKind::Identifier(s) => {
                                let s = s.clone();
                                self.advance();
                                dims.push(DimSpec::Symbol(s));
                            }
                            TokenKind::DotDot => {
                                self.advance();
                                dims.push(DimSpec::Wildcard);
                            }
                            _ => break,
                        }
                    }
                    self.expect(TokenKind::RBracket)?;
                    Ok(TypeAnnotation::Tensor {
                        dtype: Box::new(dtype),
                        dims,
                    })
                } else if matches!(self.peek_kind(), TokenKind::Lt) {
                    // Generic type: Vec<Token>, HashMap<K,V>, etc.
                    self.advance();
                    let mut args = Vec::new();
                    while !matches!(self.peek_kind(), TokenKind::Gt) {
                        args.push(self.parse_type()?);
                        if !matches!(self.peek_kind(), TokenKind::Comma) {
                            break;
                        }
                        self.advance();
                    }
                    self.expect(TokenKind::Gt)?;
                    Ok(TypeAnnotation::Generic {
                        base: Ident { name, span },
                        args,
                    })
                } else {
                    Ok(TypeAnnotation::Named(Ident { name, span }))
                }
            }
            TokenKind::LBracket => {
                self.advance();
                let inner = self.parse_type()?;
                self.expect(TokenKind::RBracket)?;
                Ok(TypeAnnotation::Array(Box::new(inner)))
            }
            TokenKind::Fn => {
                self.advance();
                self.expect(TokenKind::LParen)?;
                let mut params = Vec::new();
                if !matches!(self.peek_kind(), TokenKind::RParen) {
                    loop {
                        params.push(self.parse_type()?);
                        if !matches!(self.peek_kind(), TokenKind::Comma) {
                            break;
                        }
                        self.advance();
                    }
                }
                self.expect(TokenKind::RParen)?;
                let ret = if matches!(self.peek_kind(), TokenKind::Arrow) {
                    self.advance();
                    Box::new(self.parse_type()?)
                } else {
                    Box::new(TypeAnnotation::Unit)
                };
                Ok(TypeAnnotation::FnType { params, ret })
            }
            TokenKind::LParen => {
                self.advance();
                let inner = self.parse_type()?;
                self.expect(TokenKind::RParen)?;
                Ok(inner)
            }
            _ => Err(TenthError::ParseError {
                line: span.line,
                col: span.col,
                message: "expected type".into(),
            }),
        }
    }

    fn parse_param(&mut self) -> TenthResult<Param> {
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
                    message: "expected parameter name".into(),
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
        self.expect(TokenKind::Colon)?;
        let type_ann = self.parse_type()?;
        Ok(Param { name, type_ann })
    }

    fn parse_block_stmts(&mut self) -> TenthResult<Vec<Stmt>> {
        let mut stmts = Vec::new();
        while !matches!(self.peek_kind(), TokenKind::RBrace) && !self.at_eof() {
            stmts.push(self.parse_stmt()?);
        }
        self.expect(TokenKind::RBrace)?;
        Ok(stmts)
    }

    fn parse_stmt(&mut self) -> TenthResult<Stmt> {
        let span = self.span();

        match self.peek_kind() {
            TokenKind::Let => {
                self.advance();
                let mutable = self.match_token(TokenKind::Mut);
                let name = if let TokenKind::Identifier(name) = &self.peek().kind {
                    Ident {
                        name: name.clone(),
                        span: self.peek().span.clone(),
                    }
                } else {
                    return Err(TenthError::ParseError {
                        line: self.peek().span.line,
                        col: self.peek().span.col,
                        message: "expected variable name".into(),
                    });
                };
                self.advance();

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
                        name,
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

    pub fn parse_program(&mut self) -> TenthResult<Program> {
        let mut items = Vec::new();
        let mut stmts = Vec::new();
        while !self.at_eof() {
            match self.peek_kind() {
                TokenKind::Fn | TokenKind::Struct | TokenKind::Enum | TokenKind::Impl
                | TokenKind::Mod | TokenKind::Use | TokenKind::Trait => {
                    items.push(self.parse_item()?);
                }
                _ => {
                    let span = self.span();
                    match self.peek_kind() {
                        TokenKind::Let | TokenKind::While => {
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
                },
                span,
            });
        }
        Ok(Program { items })
    }

    fn parse_item(&mut self) -> TenthResult<Item> {
        let span = self.span();
        match self.peek_kind() {
            TokenKind::Fn => {
                self.advance();
                let name = if let TokenKind::Identifier(name) = &self.peek().kind {
                    Ident {
                        name: name.clone(),
                        span: self.peek().span.clone(),
                    }
                } else {
                    return Err(TenthError::ParseError {
                        line: self.peek().span.line,
                        col: self.peek().span.col,
                        message: "expected function name".into(),
                    });
                };
                self.advance();
                let generics = self.parse_generic_params()?;
                self.expect(TokenKind::LParen)?;
                let mut params = Vec::new();
                if !matches!(self.peek_kind(), TokenKind::RParen) {
                    loop {
                        params.push(self.parse_param()?);
                        if !matches!(self.peek_kind(), TokenKind::Comma) {
                            break;
                        }
                        self.advance();
                    }
                }
                self.expect(TokenKind::RParen)?;

                let return_type = if matches!(self.peek_kind(), TokenKind::Arrow) {
                    self.advance();
                    Some(self.parse_type()?)
                } else {
                    None
                };

                let body = self.parse_block_or_expr()?;

                self.match_token(TokenKind::Semicolon);

                Ok(Item {
                    kind: ItemKind::Function {
                        name,
                        generics,
                        params,
                        return_type,
                        body,
                    },
                    span,
                })
            }
            TokenKind::Struct => {
                self.advance();
                let name = self.expect_ident()?;
                let generics = self.parse_generic_params()?;
                self.expect(TokenKind::LBrace)?;
                let mut fields = Vec::new();
                while !matches!(self.peek_kind(), TokenKind::RBrace) {
                    let field_name = self.expect_ident()?;
                    self.expect(TokenKind::Colon)?;
                    let type_ann = self.parse_type()?;
                    fields.push(StructField { name: field_name, type_ann });
                    if !matches!(self.peek_kind(), TokenKind::Comma) { break; }
                    self.advance();
                }
                self.expect(TokenKind::RBrace)?;
                self.match_token(TokenKind::Semicolon);
                Ok(Item { kind: ItemKind::StructDef { name, generics, fields }, span })
            }
            TokenKind::Enum => {
                self.advance();
                let name = self.expect_ident()?;
                self.known_enums.insert(name.name.clone());
                self.expect(TokenKind::LBrace)?;
                let mut variants = Vec::new();
                while !matches!(self.peek_kind(), TokenKind::RBrace) {
                    let variant_name = self.expect_ident()?;
                    let mut fields = Vec::new();
                    if matches!(self.peek_kind(), TokenKind::LParen) {
                        self.advance();
                        while !matches!(self.peek_kind(), TokenKind::RParen) {
                            let fname = self.expect_ident()?;
                            self.expect(TokenKind::Colon)?;
                            let ftype = self.parse_type()?;
                            fields.push(StructField { name: fname, type_ann: ftype });
                            if !matches!(self.peek_kind(), TokenKind::Comma) { break; }
                            self.advance();
                        }
                        self.expect(TokenKind::RParen)?;
                    }
                    variants.push(EnumVariant { name: variant_name, fields });
                    if !matches!(self.peek_kind(), TokenKind::Comma) { break; }
                    self.advance();
                }
                self.expect(TokenKind::RBrace)?;
                self.match_token(TokenKind::Semicolon);
                Ok(Item { kind: ItemKind::EnumDef { name, variants }, span })
            }
            TokenKind::Impl => {
                self.advance();
                let first_ident = self.expect_ident()?;
                let generics = self.parse_generic_params()?;
                let (trait_name, type_name) = if matches!(self.peek_kind(), TokenKind::For) {
                    self.advance();
                    let tn = self.expect_ident()?;
                    self.expect(TokenKind::LBrace)?;
                    (Some(first_ident), tn)
                } else {
                    self.expect(TokenKind::LBrace)?;
                    (None, first_ident)
                };
                let mut functions = Vec::new();
                while !matches!(self.peek_kind(), TokenKind::RBrace) {
                    functions.push(self.parse_item()?);
                }
                self.expect(TokenKind::RBrace)?;
                Ok(Item { kind: ItemKind::Impl { type_name, trait_name, generics, functions }, span })
            }
            TokenKind::Mod => {
                self.advance();
                let name = self.expect_ident()?;
                self.expect(TokenKind::LBrace)?;
                let mut items = Vec::new();
                while !matches!(self.peek_kind(), TokenKind::RBrace) {
                    items.push(self.parse_item()?);
                }
                self.expect(TokenKind::RBrace)?;
                Ok(Item { kind: ItemKind::Mod { name, items }, span })
            }
            TokenKind::Use => {
                self.advance();
                let mut path = vec![self.expect_ident()?];
                while matches!(self.peek_kind(), TokenKind::ColonColon) {
                    self.advance();
                    path.push(self.expect_ident()?);
                }
                self.match_token(TokenKind::Semicolon);
                Ok(Item { kind: ItemKind::Use { path }, span })
            }
            TokenKind::Trait => {
                self.advance();
                let name = self.expect_ident()?;
                let generics = self.parse_generic_params()?;
                self.expect(TokenKind::LBrace)?;
                let mut methods = Vec::new();
                while !matches!(self.peek_kind(), TokenKind::RBrace) {
                    self.expect(TokenKind::Fn)?;
                    let method_name = self.expect_ident()?;
                    self.expect(TokenKind::LParen)?;
                    let mut params = Vec::new();
                    if !matches!(self.peek_kind(), TokenKind::RParen) {
                        loop {
                            params.push(self.parse_param()?);
                            if !matches!(self.peek_kind(), TokenKind::Comma) {
                                break;
                            }
                            self.advance();
                        }
                    }
                    self.expect(TokenKind::RParen)?;
                    let return_type = if matches!(self.peek_kind(), TokenKind::Arrow) {
                        self.advance();
                        Some(self.parse_type()?)
                    } else {
                        None
                    };
                    self.expect(TokenKind::Semicolon)?;
                    methods.push(TraitMethod { name: method_name, params, return_type });
                }
                self.expect(TokenKind::RBrace)?;
                Ok(Item { kind: ItemKind::Trait { name, generics, methods }, span })
            }
            _ => {
                let expr = self.parse_expr()?;
                self.match_token(TokenKind::Semicolon);
                Ok(Item {
                    kind: ItemKind::Function {
                        name: Ident {
                            name: "<expr>".into(),
                            span: span.clone(),
                        },
                        generics: Vec::new(),
                        params: Vec::new(),
                        return_type: None,
                        body: expr,
                    },
                    span,
                })
            }
        }
    }

    fn parse_block_or_expr(&mut self) -> TenthResult<Expr> {
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

    fn parse_generic_params(&mut self) -> TenthResult<Vec<GenericParam>> {
        if !matches!(self.peek_kind(), TokenKind::Lt) {
            return Ok(Vec::new());
        }
        self.advance();
        let mut params = Vec::new();
        while !matches!(self.peek_kind(), TokenKind::Gt) {
            let name = self.expect_ident()?;
            let mut bounds = Vec::new();
            if matches!(self.peek_kind(), TokenKind::Colon) {
                self.advance();
                bounds.push(self.expect_ident()?);
                while matches!(self.peek_kind(), TokenKind::Plus) {
                    self.advance();
                    bounds.push(self.expect_ident()?);
                }
            }
            params.push(GenericParam { name, bounds });
            if !matches!(self.peek_kind(), TokenKind::Comma) {
                break;
            }
            self.advance();
        }
        self.expect(TokenKind::Gt)?;
        Ok(params)
    }

    fn parse_match_pattern(&mut self) -> TenthResult<Pattern> {
        match &self.peek().kind {
            TokenKind::Identifier(s) if s == "_" => {
                self.advance();
                Ok(Pattern::Wildcard)
            }
            TokenKind::IntLiteral(n) => {
                let n = n.clone();
                self.advance();
                Ok(Pattern::Literal(Literal::Int(n)))
            }
            TokenKind::FloatLiteral(n) => {
                let n = n.clone();
                self.advance();
                Ok(Pattern::Literal(Literal::Float(n)))
            }
            TokenKind::True => {
                self.advance();
                Ok(Pattern::Literal(Literal::Bool(true)))
            }
            TokenKind::False => {
                self.advance();
                Ok(Pattern::Literal(Literal::Bool(false)))
            }
            TokenKind::Identifier(_) => {
                let path_str = self.parse_qualified_path()?;
                let parts: Vec<&str> = path_str.splitn(2, "::").collect();
                if parts.len() == 2 {
                    let enum_name = parts[0].to_string();
                    let variant = parts[1].to_string();
                    let field_bind = self.parse_pattern_field_bind()?;
                    Ok(Pattern::EnumVariant { enum_name, variant, field_bind })
                } else {
                    let name = path_str;
                    if matches!(self.peek_kind(), TokenKind::LParen) {
                        self.advance();
                        if matches!(self.peek_kind(), TokenKind::RParen) {
                            self.advance();
                            Ok(Pattern::EnumVariant {
                                enum_name: String::new(),
                                variant: name,
                                field_bind: None,
                            })
                        } else {
                            let bind = self.expect_ident()?;
                            if matches!(self.peek_kind(), TokenKind::Colon) {
                                self.advance();
                                let bname = self.expect_ident()?;
                                self.expect(TokenKind::RParen)?;
                                Ok(Pattern::EnumVariant {
                                    enum_name: String::new(),
                                    variant: name,
                                    field_bind: Some((bind.name, bname.name)),
                                })
                            } else {
                                self.expect(TokenKind::RParen)?;
                                Ok(Pattern::EnumVariant {
                                    enum_name: String::new(),
                                    variant: name,
                                    field_bind: Some((String::new(), bind.name)),
                                })
                            }
                        }
                    } else {
                        Ok(Pattern::Wildcard)
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

    fn parse_pattern_field_bind(&mut self) -> TenthResult<Option<(String, String)>> {
        if !matches!(self.peek_kind(), TokenKind::LParen) {
            return Ok(None);
        }
        self.advance();
        if matches!(self.peek_kind(), TokenKind::RParen) {
            self.advance();
            return Ok(None);
        }
        let fname = self.expect_ident()?;
        if matches!(self.peek_kind(), TokenKind::Colon) {
            self.advance();
            let bname = self.expect_ident()?;
            self.expect(TokenKind::RParen)?;
            Ok(Some((fname.name, bname.name)))
        } else {
            self.expect(TokenKind::RParen)?;
            Ok(Some((String::new(), fname.name)))
        }
    }

    fn parse_qualified_path(&mut self) -> TenthResult<String> {
        let first = self.expect_ident()?;
        if matches!(self.peek_kind(), TokenKind::ColonColon) {
            self.advance();
            let second = self.expect_ident()?;
            return Ok(format!("{}::{}", first.name, second.name));
        }
        Ok(first.name)
    }

    fn expect_ident(&mut self) -> TenthResult<Ident> {
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
                message: "expected identifier".into(),
            })
        }
    }
}
