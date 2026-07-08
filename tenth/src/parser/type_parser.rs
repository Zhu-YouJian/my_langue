//! 类型注解解析：`parse_type` / `parse_generic_params`。
//!
//! 从 `parser.rs` 拆出（架构重构 T3d）。包含：
//! - `parse_type`：类型注解解析（Named / Generic / Tensor / Array / FnType / Ref / Tuple）
//! - `parse_generic_params`：泛型参数列表 `<T: Bound, U>`

use crate::error::{TenthError, TenthResult};
use crate::lexer::token::{Span, Token, TokenKind};
use super::ast::*;
use super::parser::Parser;

impl Parser {
    pub(super) fn parse_type(&mut self) -> TenthResult<TypeAnnotation> {
        let span = self.span();
        match self.peek_kind() {
            TokenKind::LParen => {
                // Tuple type: (A, B, C)
                self.advance();
                let mut types = Vec::new();
                if !matches!(self.peek_kind(), TokenKind::RParen) {
                    loop {
                        types.push(self.parse_type()?);
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
                if types.is_empty() {
                    Ok(TypeAnnotation::Named(Ident { name: "()".to_string(), span }))
                } else if types.len() == 1 {
                    Ok(types.into_iter().next().unwrap())
                } else {
                    // Flatten tuple type into a named type like "(A, B, C)"
                    let names: Vec<String> = types.iter().map(|t| {
                        match t {
                            TypeAnnotation::Named(id) => id.name.clone(),
                            TypeAnnotation::Tensor { dtype, dims } => {
                                let dim_strs: Vec<String> = dims.iter().map(|d| match d {
                                    DimSpec::Literal(n) => n.to_string(),
                                    DimSpec::Symbol(s) => s.clone(),
                                    DimSpec::Wildcard => "..".to_string(),
                                }).collect();
                                let dtype_name = match dtype.as_ref() {
                                    TypeAnnotation::Named(id) => id.name.clone(),
                                    _ => "_".to_string(),
                                };
                                format!("Tensor[{}, {}]", dtype_name, dim_strs.join(", "))
                            }
                            _ => "_".to_string(),
                        }
                    }).collect();
                    Ok(TypeAnnotation::Named(Ident { name: format!("({})", names.join(", ")), span }))
                }
            }
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
                    while !matches!(self.peek_kind(), TokenKind::Gt | TokenKind::Shr) {
                        args.push(self.parse_type()?);
                        if !matches!(self.peek_kind(), TokenKind::Comma) {
                            break;
                        }
                        self.advance();
                    }
                    self.expect_gt()?;
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
            _ => Err(TenthError::ParseError {
                line: span.line,
                col: span.col,
                message: "期望类型".into(),
            }),
        }
    }

    pub(super) fn parse_generic_params(&mut self) -> TenthResult<Vec<GenericParam>> {
        if !matches!(self.peek_kind(), TokenKind::Lt) {
            return Ok(Vec::new());
        }
        self.advance();
        let mut params = Vec::new();
        while !matches!(self.peek_kind(), TokenKind::Gt | TokenKind::Shr) {
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
        self.expect_gt()?;
        Ok(params)
    }
}
