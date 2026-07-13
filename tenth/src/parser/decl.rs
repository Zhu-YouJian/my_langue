//! 声明解析：`parse_item`。
//!
//! 从 `parser.rs` 拆出（架构重构 T3d）。包含顶层声明项解析：
//! - 函数：`fn name<T>(params) -> Type { body }`
//! - 结构体：`struct Name<T> { field: Type, ... }`
//! - 枚举：`enum Name { Variant, Variant(T), Variant { field: Type } }`
//! - 实现：`impl<T> Type { fn ... }` / `impl Trait for Type { ... }`
//! - 模块：`mod name { items }`
//! - 导入：`use path::to::item` / `use path::*`
//! - 特质：`trait Name<T> { fn ...; type T; }`

use crate::error::{TenthError, TenthResult};
use crate::hir::types::BaseType;
use crate::lexer::token::{Span, Token, TokenKind};
use super::ast::*;
use super::parser::Parser;

impl Parser {
    pub(super) fn parse_item(&mut self) -> TenthResult<Item> {
        let span = self.span();
        // Parse attributes like `#[test]`
        let mut is_test = false;
        if matches!(self.peek_kind(), TokenKind::Hash) {
            let saved_pos = self.pos;
            self.advance(); // consume `#`
            if matches!(self.peek_kind(), TokenKind::LBracket) {
                self.advance(); // consume `[`
                if let TokenKind::Identifier(name) = self.peek_kind() {
                    if name == "test" {
                        self.advance(); // consume "test"
                        if matches!(self.peek_kind(), TokenKind::RBracket) {
                            self.advance(); // consume `]`
                            is_test = true;
                        } else {
                            self.pos = saved_pos; // restore
                        }
                    } else {
                        self.pos = saved_pos; // restore
                    }
                } else {
                    self.pos = saved_pos; // restore
                }
            } else {
                self.pos = saved_pos; // restore
            }
        }
        // Handle `pub` prefix
        let is_pub = if matches!(self.peek_kind(), TokenKind::Pub) {
            self.advance();
            true
        } else {
            false
        };
        let is_async = if matches!(self.peek_kind(), TokenKind::Async) {
            self.advance();
            true
        } else {
            false
        };
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
                        message: "期望函数名".into(),
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
                        // Trailing comma: if RParen follows the comma, stop
                        if matches!(self.peek_kind(), TokenKind::RParen) {
                            break;
                        }
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
                        is_pub,
                        is_async,
                        is_test,
                    },
                    span,
                })
            }
            TokenKind::Struct => {
                self.advance();
                let name = self.expect_ident()?;
                let generics = self.parse_generic_params()?;
                let kind = if matches!(self.peek_kind(), TokenKind::LParen) {
                    // Tuple struct: struct Name(Type) — Newtype pattern
                    self.advance();
                    let mut types = Vec::new();
                    while !matches!(self.peek_kind(), TokenKind::RParen) {
                        let ty = self.parse_type()?;
                        types.push(ty);
                        if !matches!(self.peek_kind(), TokenKind::Comma) { break; }
                        self.advance();
                    }
                    self.expect(TokenKind::RParen)?;
                    StructKind::Tuple(types)
                } else {
                    // Named struct: struct Name { field: Type }
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
                    StructKind::Named(fields)
                };
                self.match_token(TokenKind::Semicolon);
                Ok(Item { kind: ItemKind::StructDef { name, generics, kind, is_pub }, span })
            }
            TokenKind::Enum => {
                self.advance();
                let name = self.expect_ident()?;
                self.known_enums.insert(name.name.clone());
                self.expect(TokenKind::LBrace)?;
                let mut variants = Vec::new();
                while !matches!(self.peek_kind(), TokenKind::RBrace) {
                    let variant_name = self.expect_ident()?;
                    let kind = if matches!(self.peek_kind(), TokenKind::LParen) {
                        self.advance();
                        // Determine if this is a named-field or tuple variant.
                        // Look ahead: if we see `Identifier :` it's named; otherwise tuple.
                        let is_named = self.looks_like_named_enum_fields();
                        if is_named {
                            let mut fields = Vec::new();
                            while !matches!(self.peek_kind(), TokenKind::RParen) {
                                let fname = self.expect_ident()?;
                                self.expect(TokenKind::Colon)?;
                                let ftype = self.parse_type()?;
                                fields.push(StructField { name: fname, type_ann: ftype });
                                if !matches!(self.peek_kind(), TokenKind::Comma) { break; }
                                self.advance();
                            }
                            self.expect(TokenKind::RParen)?;
                            EnumVariantKind::Named(fields)
                        } else {
                            let mut types = Vec::new();
                            while !matches!(self.peek_kind(), TokenKind::RParen) {
                                let ty = self.parse_type()?;
                                types.push(ty);
                                if !matches!(self.peek_kind(), TokenKind::Comma) { break; }
                                self.advance();
                            }
                            self.expect(TokenKind::RParen)?;
                            EnumVariantKind::Tuple(types)
                        }
                    } else {
                        EnumVariantKind::Unit
                    };
                    variants.push(EnumVariant { name: variant_name, kind });
                    if !matches!(self.peek_kind(), TokenKind::Comma) { break; }
                    self.advance();
                }
                self.expect(TokenKind::RBrace)?;
                self.match_token(TokenKind::Semicolon);
                Ok(Item { kind: ItemKind::EnumDef { name, variants }, span })
            }
            TokenKind::Union => {
                self.advance();
                let name = self.expect_ident()?;
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
                Ok(Item { kind: ItemKind::Union { name, fields }, span })
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
                    // Check for glob: `use path::*`
                    if matches!(self.peek_kind(), TokenKind::Star) {
                        self.advance();
                        self.match_token(TokenKind::Semicolon);
                        return Ok(Item { kind: ItemKind::Use { path, glob: true }, span });
                    }
                    path.push(self.expect_ident()?);
                }
                self.match_token(TokenKind::Semicolon);
                Ok(Item { kind: ItemKind::Use { path, glob: false }, span })
            }
            TokenKind::Trait => {
                self.advance();
                let name = self.expect_ident()?;
                let generics = self.parse_generic_params()?;
                self.expect(TokenKind::LBrace)?;
                let mut methods = Vec::new();
                let mut associated_types = Vec::new();
                while !matches!(self.peek_kind(), TokenKind::RBrace) {
                    // Parse associated type: `type Name;`
                    if matches!(self.peek_kind(), TokenKind::Type) {
                        self.advance();
                        let type_name = self.expect_ident()?;
                        self.expect(TokenKind::Semicolon)?;
                        associated_types.push(type_name);
                        continue;
                    }
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
                    // Default method body: `fn foo() -> T { ... }` instead of `fn foo() -> T;`
                    let body = if matches!(self.peek_kind(), TokenKind::LBrace) {
                        Some(self.parse_expr()?)
                    } else {
                        self.expect(TokenKind::Semicolon)?;
                        None
                    };
                    methods.push(TraitMethod { name: method_name, params, return_type, body });
                }
                self.expect(TokenKind::RBrace)?;
                Ok(Item { kind: ItemKind::Trait { name, generics, methods, associated_types }, span })
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
                        is_pub: false,
                        is_async: false,
                        is_test: false,
                    },
                    span,
                })
            }
        }
    }
}
