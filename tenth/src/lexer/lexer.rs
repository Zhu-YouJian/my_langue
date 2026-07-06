use crate::error::{TenthError, TenthResult};
use crate::hir::types::BaseType;
use super::token::{Span, Token, TokenKind, StringPart};

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
                // Line comment: skip to end of line
                self.advance();
                self.advance();
                while let Some(c) = self.peek() {
                    if c == '\n' {
                        break;
                    }
                    self.advance();
                }
            } else if ch == '/' && self.peek_next() == Some('*') {
                // Block comment: skip until */
                self.advance(); // skip /
                self.advance(); // skip *
                let mut depth: i32 = 1;
                while depth > 0 {
                    match (self.peek(), self.peek_next()) {
                        (Some('/'), Some('*')) => {
                            self.advance();
                            self.advance();
                            depth += 1;
                        }
                        (Some('*'), Some('/')) => {
                            self.advance();
                            self.advance();
                            depth -= 1;
                        }
                        (Some(_), _) => {
                            self.advance();
                        }
                        (None, _) => {
                            // Reached EOF inside block comment — let parser handle the error
                            break;
                        }
                    }
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

        let mut is_float = false;

        if self.peek() == Some('.') && self.peek_next().map_or(false, |c| c.is_ascii_digit()) {
            is_float = true;
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
        }

        // Scientific notation: 1e9, 1.5e-3, 1e+4
        if let Some(ch) = self.peek() {
            if ch == 'e' || ch == 'E' {
                is_float = true;
                s.push(ch);
                self.advance();
                if let Some(sign) = self.peek() {
                    if sign == '+' || sign == '-' {
                        s.push(sign);
                        self.advance();
                    }
                }
                while let Some(ch) = self.peek() {
                    if ch.is_ascii_digit() || ch == '_' {
                        s.push(ch);
                        self.advance();
                    } else {
                        break;
                    }
                }
            }
        }

        // f32/f64 后缀检测：`3.14f32` / `3f64` / `1e5f32` 等。
        // 后缀使整数字面量也变为浮点（与 Rust 语义一致：`3f32` == `3.0f32`）。
        let mut suffix_dtype: Option<BaseType> = None;
        if self.peek() == Some('f') {
            let c1 = self.source.get(self.pos + 1).copied();
            let c2 = self.source.get(self.pos + 2).copied();
            let c3 = self.source.get(self.pos + 3).copied();
            let boundary_ok = c3.map_or(true, |c| !c.is_alphanumeric() && c != '_');
            match (c1, c2, boundary_ok) {
                (Some('3'), Some('2'), true) => {
                    self.advance(); self.advance(); self.advance();
                    suffix_dtype = Some(BaseType::F32);
                }
                (Some('6'), Some('4'), true) => {
                    self.advance(); self.advance(); self.advance();
                    suffix_dtype = Some(BaseType::F64);
                }
                _ => {}
            }
        }

        if let Some(dt) = suffix_dtype {
            // 有 f32/f64 后缀 → 浮点字面量（即使数字部分无小数点，如 `3f32`）
            let n: f64 = s.parse().map_err(|_| TenthError::LexerError {
                line: span.line,
                col: span.col,
                message: format!("无效的浮点数：{}", s),
            })?;
            Ok(Token {
                kind: TokenKind::FloatLiteral(n, dt),
                span,
            })
        } else if is_float {
            let n: f64 = s.parse().map_err(|_| TenthError::LexerError {
                line: span.line,
                col: span.col,
                message: format!("无效的浮点数：{}", s),
            })?;
            Ok(Token {
                kind: TokenKind::FloatLiteral(n, BaseType::F64),
                span,
            })
        } else {
            let n: i64 = s.parse().map_err(|_| TenthError::LexerError {
                line: span.line,
                col: span.col,
                message: format!("无效的整数：{}", s),
            })?;
            Ok(Token {
                kind: TokenKind::IntLiteral(n),
                span,
            })
        }
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
            "async" => TokenKind::Async,
            "await" => TokenKind::Await,
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
        self.advance(); // consume opening "
        let mut parts: Vec<StringPart> = Vec::new();
        let mut current_literal = String::new();
        let mut has_interpolation = false;

        while let Some(ch) = self.peek() {
            if ch == '"' {
                self.advance();
                if !current_literal.is_empty() {
                    parts.push(StringPart::Literal(current_literal));
                }
                if has_interpolation {
                    return Ok(Token {
                        kind: TokenKind::InterpolatedString(parts),
                        span,
                    });
                } else if parts.len() == 1 {
                    if let Some(StringPart::Literal(s)) = parts.into_iter().next() {
                        return Ok(Token {
                            kind: TokenKind::StringLiteral(s),
                            span,
                        });
                    }
                    unreachable!()
                } else {
                    return Ok(Token {
                        kind: TokenKind::StringLiteral(String::new()),
                        span,
                    });
                }
            }
            if ch == '\\' {
                self.advance();
                match self.peek() {
                    Some('n') => { self.advance(); current_literal.push('\n'); }
                    Some('r') => { self.advance(); current_literal.push('\r'); }
                    Some('t') => { self.advance(); current_literal.push('\t'); }
                    Some('\\') => { self.advance(); current_literal.push('\\'); }
                    Some('"') => { self.advance(); current_literal.push('"'); }
                    Some('{') => { self.advance(); current_literal.push('{'); }
                    Some('}') => { self.advance(); current_literal.push('}'); }
                    Some(c) => { self.advance(); current_literal.push(c); }
                    None => break,
                }
            } else if ch == '{' {
                // String interpolation: {expr}
                // Only treat as interpolation if the next character looks like
                // the start of a valid identifier (alphabetic or _). Otherwise
                // treat { as a literal character (e.g., the string "{").
                let next = self.peek_next();
                if next.map(|c| c.is_alphabetic() || c == '_').unwrap_or(false) {
                    self.advance(); // consume {
                    let mut expr = String::new();
                    let mut found_close = false;
                    while let Some(c) = self.peek() {
                        if c == '}' {
                            self.advance(); // consume }
                            found_close = true;
                            break;
                        }
                        if c == '"' {
                            // Reached end of string literal without finding }
                            // — treat { as literal and stop
                            break;
                        }
                        expr.push(c);
                        self.advance();
                    }
                    let trimmed = expr.trim();
                    // Check if the expression is a valid identifier (alphanumeric + _)
                    let is_valid_ident = found_close
                        && !trimmed.is_empty()
                        && trimmed.chars().all(|c| c.is_alphanumeric() || c == '_' || c == '.')
                        && trimmed.contains(|c: char| c.is_alphanumeric());
                    if is_valid_ident {
                        has_interpolation = true;
                        if !current_literal.is_empty() {
                            parts.push(StringPart::Literal(current_literal));
                            current_literal = String::new();
                        }
                        parts.push(StringPart::Expr(trimmed.to_string()));
                    } else {
                        // Not a valid identifier — treat { } as literal text
                        current_literal.push('{');
                        current_literal.push_str(&expr);
                        if found_close {
                            current_literal.push('}');
                        }
                    }
                } else {
                    // { followed by non-identifier — treat as literal
                    current_literal.push(ch);
                    self.advance();
                }
            } else {
                current_literal.push(ch);
                self.advance();
            }
        }
        Err(TenthError::LexerError {
            line: span.line,
            col: span.col,
            message: "字符串未闭合".into(),
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
            '?' => Some(TokenKind::QuestionMark),
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
            message: format!("意外字符：'{}'", ch),
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