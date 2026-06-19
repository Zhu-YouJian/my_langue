use tenth::lexer::lexer::Lexer;
use tenth::lexer::token::{Token, TokenKind};

use super::Handler;
use crate::lsp_types::*;

pub struct SemanticTokensHandler;

impl Handler for SemanticTokensHandler {
    fn handle(&self, params: Option<&serde_json::Value>) -> serde_json::Value {
        let uri = params
            .and_then(|p| p.get("textDocument"))
            .and_then(|td| td.get("uri"))
            .and_then(|u| u.as_str())
            .unwrap_or("");

        let source = crate::document_store::get_content_or_disk_global(uri)
            .unwrap_or_default();

        let tokens = compute_semantic_tokens(&source);
        let semantic_tokens = SemanticTokens { data: tokens };
        serde_json::to_value(semantic_tokens).unwrap()
    }
}

// Token type indices (must match the legend in initialize.rs)
const TYPE_KEYWORD: u32 = 0;
const TYPE_FUNCTION: u32 = 1;
const TYPE_VARIABLE: u32 = 2;
const TYPE_TYPE: u32 = 3;
const TYPE_STRING: u32 = 4;
const TYPE_NUMBER: u32 = 5;
const TYPE_OPERATOR: u32 = 6;
const TYPE_COMMENT: u32 = 7;
const TYPE_ENUM_MEMBER: u32 = 8;
const TYPE_STRUCT: u32 = 9;

fn compute_semantic_tokens(source: &str) -> Vec<u32> {
    let mut lexer = Lexer::new(source);
    let tokens = match lexer.tokenize() {
        Ok(t) => t,
        Err(_) => return Vec::new(),
    };

    // Build line offsets for computing character positions
    let _lines: Vec<&str> = source.lines().collect();

    let mut data = Vec::new();
    let mut prev_line: u32 = 0;
    let mut prev_char: u32 = 0;

    for token in &tokens {
        if token.kind == TokenKind::Eof {
            continue;
        }

        let (token_type, text_len) = match classify_token(token) {
            Some((tt, len)) => (tt, len),
            None => continue,
        };

        // LSP positions are 0-based, lexer spans are 1-based
        let line = token.span.line.saturating_sub(1) as u32;
        let char_pos = token.span.col.saturating_sub(1) as u32;

        // Delta encoding: relative to previous token
        let delta_line = line - prev_line;
        let delta_start = if delta_line == 0 {
            char_pos - prev_char
        } else {
            char_pos
        };

        data.push(delta_line);
        data.push(delta_start);
        data.push(text_len);
        data.push(token_type);
        data.push(0); // no modifiers

        prev_line = line;
        prev_char = char_pos;
    }

    data
}

fn classify_token(token: &Token) -> Option<(u32, u32)> {
    let text_len = match &token.kind {
        TokenKind::Identifier(s) => s.len() as u32,
        TokenKind::IntLiteral(n) => n.to_string().len() as u32,
        TokenKind::FloatLiteral(n) => n.to_string().len() as u32,
        TokenKind::StringLiteral(s) => (s.len() + 2) as u32, // +2 for quotes
        TokenKind::CharLiteral(c) => (c.len_utf8() + 3) as u32, // 'c'
        TokenKind::Fn => 2,
        TokenKind::Let => 3,
        TokenKind::Mut => 3,
        TokenKind::If => 2,
        TokenKind::Else => 4,
        TokenKind::Match => 5,
        TokenKind::For => 3,
        TokenKind::While => 5,
        TokenKind::Loop => 4,
        TokenKind::Break => 5,
        TokenKind::Continue => 8,
        TokenKind::Return => 6,
        TokenKind::Try => 3,
        TokenKind::Use => 3,
        TokenKind::Mod => 3,
        TokenKind::Pub => 3,
        TokenKind::Trait => 5,
        TokenKind::Impl => 4,
        TokenKind::Enum => 4,
        TokenKind::Struct => 6,
        TokenKind::Type => 4,
        TokenKind::Self_ => 4,
        TokenKind::Spawn => 5,
        TokenKind::Task => 4,
        TokenKind::Shard => 5,
        TokenKind::Node => 4,
        TokenKind::Macro => 5,
        TokenKind::Where => 5,
        TokenKind::As => 2,
        TokenKind::In => 2,
        TokenKind::True => 4,
        TokenKind::False => 5,
        TokenKind::Move => 4,
        TokenKind::Plus => 1,
        TokenKind::Minus => 1,
        TokenKind::Star => 1,
        TokenKind::Slash => 1,
        TokenKind::Percent => 1,
        TokenKind::EqEq => 2,
        TokenKind::NotEq => 2,
        TokenKind::Lt => 1,
        TokenKind::Gt => 1,
        TokenKind::LtEq => 2,
        TokenKind::GtEq => 2,
        TokenKind::AndAnd => 2,
        TokenKind::OrOr => 2,
        TokenKind::Not => 1,
        TokenKind::Assign => 1,
        TokenKind::Arrow => 2,
        TokenKind::FatArrow => 2,
        _ => return None,
    };

    let token_type = match &token.kind {
        // Keywords
        TokenKind::Fn | TokenKind::Let | TokenKind::Mut | TokenKind::If | TokenKind::Else
        | TokenKind::Match | TokenKind::For | TokenKind::While | TokenKind::Loop
        | TokenKind::Break | TokenKind::Continue | TokenKind::Return | TokenKind::Try
        | TokenKind::Use | TokenKind::Mod | TokenKind::Pub | TokenKind::Trait
        | TokenKind::Impl | TokenKind::Enum | TokenKind::Struct | TokenKind::Type
        | TokenKind::Self_ | TokenKind::Spawn | TokenKind::Task | TokenKind::Shard
        | TokenKind::Node | TokenKind::Macro | TokenKind::Where | TokenKind::As
        | TokenKind::In | TokenKind::True | TokenKind::False | TokenKind::Move => TYPE_KEYWORD,

        // Types (heuristically: identifiers that start with uppercase)
        TokenKind::Identifier(s) => {
            if s.chars().next().map(|c| c.is_uppercase()).unwrap_or(false) {
                // Could be a type or enum variant
                TYPE_TYPE
            } else {
                // Function or variable — we can't easily distinguish without context
                // Default to variable; the client can refine
                TYPE_VARIABLE
            }
        }

        // Literals
        TokenKind::IntLiteral(_) | TokenKind::FloatLiteral(_) => TYPE_NUMBER,
        TokenKind::StringLiteral(_) => TYPE_STRING,
        TokenKind::CharLiteral(_) => TYPE_STRING,

        // Operators
        TokenKind::Plus | TokenKind::Minus | TokenKind::Star | TokenKind::Slash
        | TokenKind::Percent | TokenKind::EqEq | TokenKind::NotEq | TokenKind::Lt
        | TokenKind::Gt | TokenKind::LtEq | TokenKind::GtEq | TokenKind::AndAnd
        | TokenKind::OrOr | TokenKind::Not | TokenKind::Assign | TokenKind::Arrow
        | TokenKind::FatArrow => TYPE_OPERATOR,

        _ => return None,
    };

    Some((token_type, text_len))
}
