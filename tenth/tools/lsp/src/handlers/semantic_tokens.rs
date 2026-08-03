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

        // LSP positions are 0-based; lexer span cols are normalized via span.rs
        let line = token.span.line.saturating_sub(1) as u32;
        let char_pos = crate::span::token_start_col0(token) as u32;

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
        TokenKind::IntLiteral(n, _) => n.to_string().len() as u32,
        TokenKind::FloatLiteral(n, _) => n.to_string().len() as u32,
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
        TokenKind::IntLiteral(..) | TokenKind::FloatLiteral(_, _) => TYPE_NUMBER,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_semantic_tokens_empty_file() {
        // 空文件不应产生任何 token
        let data = compute_semantic_tokens("");
        assert!(data.is_empty(), "expected empty token data, got {:?}", data);
    }

    #[test]
    fn test_semantic_tokens_simple_program_has_tokens() {
        // 简单程序：至少应识别出 fn 关键字、标识符、字面量等
        // 每个识别出的 token 输出 5 个 u32 (delta_line, delta_start, length, type, modifiers)
        let src = "fn add() -> i32 { 42 }";
        let data = compute_semantic_tokens(src);
        assert!(
            !data.is_empty(),
            "expected non-empty token data for simple program"
        );
        // 数据长度应为 5 的倍数（每个 token 5 个 u32）
        assert_eq!(
            data.len() % 5,
            0,
            "token data length must be multiple of 5, got {}",
            data.len()
        );
        // 至少应有几个 token：fn、add、i32、42、{、} 中的若干
        // 保守断言至少 3 个 token（fn + add + 42）
        let token_count = data.len() / 5;
        assert!(
            token_count >= 3,
            "expected at least 3 tokens, got {} (data: {:?})",
            token_count,
            data
        );
        // 第一个 token 应在第一行（delta_line == 0）
        assert_eq!(data[0], 0, "first token delta_line should be 0, got {}", data[0]);
    }

    #[test]
    fn test_semantic_tokens_lex_error_returns_empty() {
        // lexer 错误（如未闭合字符串）应返回空 Vec，不 panic
        let src = "fn bad() -> i32 { \"unclosed }";
        let data = compute_semantic_tokens(src);
        // 出错时返回空 Vec
        assert!(data.is_empty(), "expected empty Vec on lex error, got {:?}", data);
    }

    #[test]
    fn test_semantic_tokens_keyword_type_index() {
        // fn 关键字应映射到 TYPE_KEYWORD (0)
        let src = "fn x() -> i32 { 0 }";
        let data = compute_semantic_tokens(src);
        assert!(!data.is_empty());
        // 第一个 token 是 fn：type 索引在第 4 个 u32（index 3）
        // data = [delta_line, delta_start, length, token_type, modifiers, ...]
        assert_eq!(
            data[3], TYPE_KEYWORD,
            "first token 'fn' should be keyword type ({}), got {}",
            TYPE_KEYWORD, data[3]
        );
        // fn 长度应为 2
        assert_eq!(data[2], 2, "fn length should be 2, got {}", data[2]);
    }
}
