//! Token span 归一化（LSP 侧）。
//!
//! Tenth lexer 记录的 `Token::span.col` 存在不一致（见 `tenth/src/lexer/lexer.rs`）：
//! - 由 `read_identifier` / `read_number` 产出的 token（标识符、关键字、整数/浮点
//!   字面量）在**首字符已被消费之后**才记录 span → 真实 0-based 起始列 = `col - 2`；
//! - 其余 token（标点、字符串、字符、Lifetime、CustomOperator）在消费前记录 span
//!   → 真实 0-based 起始列 = `col - 1`。
//!
//! 本模块集中做该转换，使 LSP handler 无论 token 种类都能得到准确的 0-based 位置。
//! 注：这是 lexer 的预存行为，LSP 只在边界归一化，不改动主编译器（护城河红线）。

use tenth::lexer::token::{Token, TokenKind};

/// Token 在行内的真实 0-based 起始列。
pub fn token_start_col0(token: &Token) -> usize {
    if is_identifier_like(&token.kind) {
        token.span.col.saturating_sub(2)
    } else {
        token.span.col.saturating_sub(1)
    }
}

/// 该 token 种类是否由 lexer 的 `read_identifier`/`read_number` 产出
/// （其在首字符已消费后记录 span.col）。
fn is_identifier_like(kind: &TokenKind) -> bool {
    matches!(
        kind,
        TokenKind::Identifier(_)
            | TokenKind::IntLiteral(..)
            | TokenKind::FloatLiteral(..)
            | TokenKind::Fn
            | TokenKind::Let
            | TokenKind::Mut
            | TokenKind::If
            | TokenKind::Else
            | TokenKind::Match
            | TokenKind::For
            | TokenKind::While
            | TokenKind::Do
            | TokenKind::Yield
            | TokenKind::Loop
            | TokenKind::Break
            | TokenKind::Continue
            | TokenKind::Return
            | TokenKind::Try
            | TokenKind::Use
            | TokenKind::Mod
            | TokenKind::Pub
            | TokenKind::Trait
            | TokenKind::Impl
            | TokenKind::Enum
            | TokenKind::Struct
            | TokenKind::Union
            | TokenKind::Type
            | TokenKind::Self_
            | TokenKind::Async
            | TokenKind::Await
            | TokenKind::Spawn
            | TokenKind::Task
            | TokenKind::Shard
            | TokenKind::Node
            | TokenKind::Macro
            | TokenKind::Where
            | TokenKind::As
            | TokenKind::In
            | TokenKind::True
            | TokenKind::False
            | TokenKind::Move
            | TokenKind::Dyn
            | TokenKind::Lossy
            | TokenKind::Operator
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use tenth::lexer::lexer::Lexer;

    fn tokens(src: &str) -> Vec<Token> {
        let mut lexer = Lexer::new(src);
        lexer.tokenize().expect("tokenize failed")
    }

    #[test]
    fn test_identifier_col_is_2based() {
        // 首个 helper 在 `fn helper(`：0-based 起始列 3，lexer span.col = 5 → start0 = 3
        let toks = tokens("fn helper(x: i32) -> i32 { helper(1) }");
        let helper = toks
            .iter()
            .find(|t| matches!(&t.kind, TokenKind::Identifier(s) if s == "helper"))
            .expect("helper token");
        assert_eq!(helper.span.col, 5, "lexer 应记录 2-based 列");
        assert_eq!(token_start_col0(helper), 3, "应还原为 0-based 3");
    }

    #[test]
    fn test_keyword_col_is_2based() {
        let toks = tokens("fn main() -> i32 { x }");
        let fntok = toks.iter().find(|t| t.kind == TokenKind::Fn).expect("fn token");
        assert_eq!(fntok.span.col, 2, "fn 在 0-based 0，lexer col=2");
        assert_eq!(token_start_col0(fntok), 0);
    }

    #[test]
    fn test_punct_col_is_1based() {
        let toks = tokens("fn main() -> i32 { x }");
        let lb = toks
            .iter()
            .find(|t| t.kind == TokenKind::LBrace)
            .expect("{ token");
        // { 在 0-based 17，lexer col=18（1-based）
        assert_eq!(lb.span.col, 18);
        assert_eq!(token_start_col0(lb), 17);
    }
}
