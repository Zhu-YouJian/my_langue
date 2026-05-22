use crate::error::{TenthError, TenthResult};
use crate::lexer::token::{Span, Token, TokenKind};
use super::ast::*;

pub struct Parser {
    tokens: Vec<Token>,
    pos: usize,
}

static EOF_TOKEN: Token = Token {
    kind: TokenKind::Eof,
    span: Span { line: 0, col: 0 },
};

impl Parser {
    pub fn new(tokens: Vec<Token>) -> Self {
        Parser { tokens, pos: 0 }
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

        let kind = match &token.kind {
            TokenKind::IntLiteral(n) => ExprKind::Literal(Literal::Int(*n)),
            TokenKind::FloatLiteral(n) => ExprKind::Literal(Literal::Float(*n)),
            TokenKind::True => ExprKind::Literal(Literal::Bool(true)),
            TokenKind::False => ExprKind::Literal(Literal::Bool(false)),
            TokenKind::StringLiteral(s) => ExprKind::Literal(Literal::String(s.clone())),
            TokenKind::Identifier(name) => ExprKind::Ident(Ident {
                name: name.clone(),
                span: token.span.clone(),
            }),
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
                let else_branch = if self.match_token(TokenKind::Else) {
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

        Ok(Expr { kind, span })
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
        while !self.at_eof() {
            if matches!(self.peek_kind(), TokenKind::Fn) {
                items.push(self.parse_item()?);
            } else {
                let mut stmts = Vec::new();
                while !self.at_eof() && !matches!(self.peek_kind(), TokenKind::Fn) {
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

                Ok(Item {
                    kind: ItemKind::Function {
                        name,
                        params,
                        return_type,
                        body,
                    },
                    span,
                })
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
}