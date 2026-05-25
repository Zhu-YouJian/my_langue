use crate::error::{TenthError, TenthResult};
use super::token::{Span, Token, TokenKind};

pub struct Lexer {
    source: Vec<char>,
    pos: usize,
    line: usize,
    col: usize,
}

impl Lexer {
    pub fn new(source: &str) -> Self {
        Lexer {
            source: source.chars().collect(),
            pos: 0,
            line: 1,
            col: 1,
        }
    }

    fn peek(&self) -> Option<char> {
        self.source.get(self.pos).copied()
    }

    fn peek_next(&self) -> Option<char> {
        self.source.get(self.pos + 1).copied()
    }

    fn advance(&mut self) -> Option<char> {
        let ch = self.source.get(self.pos).copied();
        if let Some(c) = ch {
            self.pos += 1;
            if c == '\n' {
                self.line += 1;
                self.col = 1;
            } else {
                self.col += 1;
            }
        }
        ch
    }

    fn span(&self) -> Span {
        Span {
            line: self.line,
            col: self.col,
        }
    }

    fn skip_whitespace_and_comments(&mut self) {
        while let Some(ch) = self.peek() {
            if ch.is_whitespace() {
                self.advance();
            } else if ch == '/' && self.peek_next() == Some('/') {
                self.advance();
                self.advance();
                while let Some(c) = self.peek() {
                    if c == '\n' {
                        break;
                    }
                    self.advance();
                }
            } else {
                break;
            }
        }
    }

    fn read_number(&mut self, first: char) -> TenthResult<Token> {
        let span = self.span();
        let mut s = String::new();
        s.push(first);

        while let Some(ch) = self.peek() {
            if ch.is_ascii_digit() || ch == '_' {
                s.push(ch);
                self.advance();
            } else {
                break;
            }
        }

        if self.peek() == Some('.') && self.peek_next().map_or(false, |c| c.is_ascii_digit()) {
            s.push('.');
            self.advance();
            while let Some(ch) = self.peek() {
                if ch.is_ascii_digit() || ch == '_' {
                    s.push(ch);
                    self.advance();
                } else {
                    break;
                }
            }
            let n: f64 = s.parse().map_err(|_| TenthError::LexerError {
                line: span.line,
                col: span.col,
                message: format!("invalid float: {}", s),
            })?;
            return Ok(Token {
                kind: TokenKind::FloatLiteral(n),
                span,
            });
        }

        let n: i64 = s.parse().map_err(|_| TenthError::LexerError {
            line: span.line,
            col: span.col,
            message: format!("invalid integer: {}", s),
        })?;
        Ok(Token {
            kind: TokenKind::IntLiteral(n),
            span,
        })
    }

    fn read_identifier(&mut self, first: char) -> Token {
        let span = self.span();
        let mut s = String::new();
        s.push(first);

        while let Some(ch) = self.peek() {
            if ch.is_alphanumeric() || ch == '_' {
                s.push(ch);
                self.advance();
            } else {
                break;
            }
        }

        let kind = match s.as_str() {
            "fn" => TokenKind::Fn,
            "let" => TokenKind::Let,
            "mut" => TokenKind::Mut,
            "if" => TokenKind::If,
            "else" => TokenKind::Else,
            "match" => TokenKind::Match,
            "for" => TokenKind::For,
            "while" => TokenKind::While,
            "loop" => TokenKind::Loop,
            "break" => TokenKind::Break,
            "continue" => TokenKind::Continue,
            "return" => TokenKind::Return,
            "try" => TokenKind::Try,
            "use" => TokenKind::Use,
            "mod" => TokenKind::Mod,
            "pub" => TokenKind::Pub,
            "trait" => TokenKind::Trait,
            "impl" => TokenKind::Impl,
            "enum" => TokenKind::Enum,
            "struct" => TokenKind::Struct,
            "type" => TokenKind::Type,
            "self" => TokenKind::Self_,
            "spawn" => TokenKind::Spawn,
            "task" => TokenKind::Task,
            "shard" => TokenKind::Shard,
            "node" => TokenKind::Node,
            "macro" => TokenKind::Macro,
            "where" => TokenKind::Where,
            "as" => TokenKind::As,
            "in" => TokenKind::In,
            "move" => TokenKind::Move,
            "true" => TokenKind::True,
            "false" => TokenKind::False,
            _ => TokenKind::Identifier(s),
        };

        Token { kind, span }
    }

    fn read_string(&mut self) -> TenthResult<Token> {
        let span = self.span();
        self.advance();
        let mut s = String::new();
        while let Some(ch) = self.peek() {
            if ch == '"' {
                self.advance();
                return Ok(Token {
                    kind: TokenKind::StringLiteral(s),
                    span,
                });
            }
            if ch == '\\' {
                self.advance();
                match self.peek() {
                    Some('n') => { self.advance(); s.push('\n'); }
                    Some('t') => { self.advance(); s.push('\t'); }
                    Some('\\') => { self.advance(); s.push('\\'); }
                    Some('"') => { self.advance(); s.push('"'); }
                    Some(c) => { self.advance(); s.push(c); }
                    None => break,
                }
            } else {
                s.push(ch);
                self.advance();
            }
        }
        Err(TenthError::LexerError {
            line: span.line,
            col: span.col,
            message: "unterminated string literal".into(),
        })
    }

    fn single_char_token(&self, ch: char) -> Option<TokenKind> {
        match ch {
            '(' => Some(TokenKind::LParen),
            ')' => Some(TokenKind::RParen),
            '[' => Some(TokenKind::LBracket),
            ']' => Some(TokenKind::RBracket),
            '{' => Some(TokenKind::LBrace),
            '}' => Some(TokenKind::RBrace),
            ',' => Some(TokenKind::Comma),
            ';' => Some(TokenKind::Semicolon),
            ':' => Some(TokenKind::Colon),
            '.' => Some(TokenKind::Dot),
            '%' => Some(TokenKind::Percent),
            '^' => Some(TokenKind::Caret),
            _ => None,
        }
    }

    pub fn next_token(&mut self) -> TenthResult<Token> {
        self.skip_whitespace_and_comments();

        let ch = match self.peek() {
            Some(c) => c,
            None => {
                return Ok(Token {
                    kind: TokenKind::Eof,
                    span: self.span(),
                });
            }
        };

        let span = self.span();

        if ch.is_ascii_digit() {
            self.advance();
            return self.read_number(ch);
        }

        if ch.is_alphabetic() || ch == '_' {
            self.advance();
            return Ok(self.read_identifier(ch));
        }

        if ch == '"' {
            return self.read_string();
        }

        self.advance();

        if ch == '=' {
            if self.peek() == Some('=') {
                self.advance();
                return Ok(Token { kind: TokenKind::EqEq, span });
            }
            if self.peek() == Some('>') {
                self.advance();
                return Ok(Token { kind: TokenKind::FatArrow, span });
            }
            return Ok(Token { kind: TokenKind::Assign, span });
        }
        if ch == '!' {
            if self.peek() == Some('=') {
                self.advance();
                return Ok(Token { kind: TokenKind::NotEq, span });
            }
            return Ok(Token { kind: TokenKind::Not, span });
        }
        if ch == '<' {
            if self.peek() == Some('=') {
                self.advance();
                return Ok(Token { kind: TokenKind::LtEq, span });
            }
            if self.peek() == Some('<') {
                self.advance();
                return Ok(Token { kind: TokenKind::Shl, span });
            }
            return Ok(Token { kind: TokenKind::Lt, span });
        }
        if ch == '>' {
            if self.peek() == Some('=') {
                self.advance();
                return Ok(Token { kind: TokenKind::GtEq, span });
            }
            if self.peek() == Some('>') {
                self.advance();
                return Ok(Token { kind: TokenKind::Shr, span });
            }
            return Ok(Token { kind: TokenKind::Gt, span });
        }
        if ch == '&' {
            if self.peek() == Some('&') {
                self.advance();
                return Ok(Token { kind: TokenKind::AndAnd, span });
            }
            return Ok(Token { kind: TokenKind::Ampersand, span });
        }
        if ch == '|' {
            if self.peek() == Some('|') {
                self.advance();
                return Ok(Token { kind: TokenKind::OrOr, span });
            }
            return Ok(Token { kind: TokenKind::Pipe, span });
        }
        if ch == '+' {
            if self.peek() == Some('=') {
                self.advance();
                return Ok(Token { kind: TokenKind::PlusAssign, span });
            }
            return Ok(Token { kind: TokenKind::Plus, span });
        }
        if ch == '-' {
            if self.peek() == Some('=') {
                self.advance();
                return Ok(Token { kind: TokenKind::MinusAssign, span });
            }
            if self.peek() == Some('>') {
                self.advance();
                return Ok(Token { kind: TokenKind::Arrow, span });
            }
            return Ok(Token { kind: TokenKind::Minus, span });
        }
        if ch == '*' {
            if self.peek() == Some('=') {
                self.advance();
                return Ok(Token { kind: TokenKind::StarAssign, span });
            }
            return Ok(Token { kind: TokenKind::Star, span });
        }
        if ch == '/' {
            if self.peek() == Some('=') {
                self.advance();
                return Ok(Token { kind: TokenKind::SlashAssign, span });
            }
            return Ok(Token { kind: TokenKind::Slash, span });
        }
        if ch == '.' {
            if self.peek() == Some('.') {
                self.advance();
                if self.peek() == Some('=') {
                    self.advance();
                    return Ok(Token { kind: TokenKind::DotDotEq, span });
                }
                return Ok(Token { kind: TokenKind::DotDot, span });
            }
            return Ok(Token { kind: TokenKind::Dot, span });
        }
        if ch == ':' {
            if self.peek() == Some(':') {
                self.advance();
                return Ok(Token { kind: TokenKind::ColonColon, span });
            }
            return Ok(Token { kind: TokenKind::Colon, span });
        }

        if let Some(kind) = self.single_char_token(ch) {
            return Ok(Token { kind, span });
        }

        Err(TenthError::LexerError {
            line: span.line,
            col: span.col,
            message: format!("unexpected character: '{}'", ch),
        })
    }

    pub fn tokenize(&mut self) -> TenthResult<Vec<Token>> {
        let mut tokens = Vec::new();
        loop {
            let token = self.next_token()?;
            if token.kind == TokenKind::Eof {
                tokens.push(token);
                break;
            }
            tokens.push(token);
        }
        Ok(tokens)
    }
}