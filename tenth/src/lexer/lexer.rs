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
        // 跳过 UTF-8 BOM（PowerShell 等编辑器默认添加）
        let source = source.strip_prefix('\u{FEFF}').unwrap_or(source);
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

        // 进制字面量：0x..（十六进制）/ 0b..（二进制）/ 0o..（八进制）
        // 仅当 first=='0' 且下一个字符是进制前缀时进入此分支；
        // 否则按原十进制路径处理（含 0、0.5、0e1 等）。
        if first == '0' {
            if let Some(prefix) = self.peek() {
                let radix: u32 = match prefix {
                    'x' | 'X' => 16,
                    'b' | 'B' => 2,
                    'o' | 'O' => 8,
                    _ => 0,
                };
                if radix != 0 {
                    self.advance(); // consume prefix (x/X/b/B/o/O)
                    let mut digits = String::new();
                    while let Some(ch) = self.peek() {
                        let is_valid = match radix {
                            16 => ch.is_ascii_hexdigit(),
                            2 => ch == '0' || ch == '1',
                            8 => ('0'..='7').contains(&ch),
                            _ => false,
                        };
                        if is_valid {
                            digits.push(ch);
                            self.advance();
                        } else if ch == '_' {
                            // 支持下划线分隔（如 0xFF_FF）——跳过不加入 digits
                            self.advance();
                        } else {
                            break;
                        }
                    }
                    if digits.is_empty() {
                        return Err(TenthError::LexerError {
                            line: span.line,
                            col: span.col,
                            message: format!("无效的进制字面量：0{}（前缀后无数字）", prefix),
                        });
                    }
                    let n: i64 = i64::from_str_radix(&digits, radix).map_err(|_| TenthError::LexerError {
                        line: span.line,
                        col: span.col,
                        message: format!("无效的进制字面量：0{}{}", prefix, digits),
                    })?;
                    return Ok(Token {
                        kind: TokenKind::IntLiteral(n, BaseType::I32),
                        span,
                    });
                }
            }
        }

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
            // 整数后缀检测：i8/i16/i32/i64/u8/u16/u32/u64
            let mut int_dtype = BaseType::I32;
            if matches!(self.peek(), Some('i') | Some('u')) {
                let c0 = self.peek().unwrap();
                let c1 = self.source.get(self.pos + 1).copied();
                let c2 = self.source.get(self.pos + 2).copied();
                let c3 = self.source.get(self.pos + 3).copied();
                let b2 = c2.map_or(true, |c| !c.is_alphanumeric() && c != '_');
                let b3 = c3.map_or(true, |c| !c.is_alphanumeric() && c != '_');
                let matched = match (c0, c1, c2, b2, b3) {
                    ('i', Some('8'), _, true, _) => { self.advance(); self.advance(); Some(BaseType::I8) }
                    ('u', Some('8'), _, true, _) => { self.advance(); self.advance(); Some(BaseType::U8) }
                    ('i', Some('1'), Some('6'), _, true) => { self.advance(); self.advance(); self.advance(); Some(BaseType::I16) }
                    ('i', Some('3'), Some('2'), _, true) => { self.advance(); self.advance(); self.advance(); Some(BaseType::I32) }
                    ('i', Some('6'), Some('4'), _, true) => { self.advance(); self.advance(); self.advance(); Some(BaseType::I64) }
                    ('u', Some('1'), Some('6'), _, true) => { self.advance(); self.advance(); self.advance(); Some(BaseType::U16) }
                    ('u', Some('3'), Some('2'), _, true) => { self.advance(); self.advance(); self.advance(); Some(BaseType::U32) }
                    ('u', Some('6'), Some('4'), _, true) => { self.advance(); self.advance(); self.advance(); Some(BaseType::U64) }
                    _ => None,
                };
                if let Some(dt) = matched {
                    int_dtype = dt;
                }
            }

            let n: i64 = s.parse().map_err(|_| TenthError::LexerError {
                line: span.line,
                col: span.col,
                message: format!("无效的整数：{}", s),
            })?;

            // 编译期字面量范围检查
            // 无后缀（默认 I32）时：超出 i32 范围自动提升为 I64，不报错
            if int_dtype == BaseType::I32 && (n < -2147483648 || n > 2147483647) { int_dtype = BaseType::I64; }
            let range_ok = match int_dtype {
                BaseType::I8 => (n >= -128 && n <= 127),
                BaseType::I16 => (n >= -32768 && n <= 32767),
                BaseType::I32 => (n >= -2147483648 && n <= 2147483647),
                BaseType::I64 => true,
                BaseType::U8 => (n >= 0 && n <= 255),
                BaseType::U16 => (n >= 0 && n <= 65535),
                BaseType::U32 => (n >= 0 && n <= 4294967295),
                BaseType::U64 => (n >= 0),
                _ => true,
            };
            if !range_ok {
                let dtype_name = match int_dtype {
                    BaseType::I8 => "i8",
                    BaseType::I16 => "i16",
                    BaseType::I32 => "i32",
                    BaseType::I64 => "i64",
                    BaseType::U8 => "u8",
                    BaseType::U16 => "u16",
                    BaseType::U32 => "u32",
                    BaseType::U64 => "u64",
                    _ => "?",
                };
                return Err(TenthError::LexerError {
                    line: span.line,
                    col: span.col,
                    message: format!("整数字面量 {} 超出 {} 范围", n, dtype_name),
                });
            }

            Ok(Token {
                kind: TokenKind::IntLiteral(n, int_dtype),
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
            "do" => TokenKind::Do,
            "yield" => TokenKind::Yield,
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
            "union" => TokenKind::Union,
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
            "dyn" => TokenKind::Dyn,
            "lossy" => TokenKind::Lossy,
            "operator" => TokenKind::Operator,
            "true" => TokenKind::True,
            "false" => TokenKind::False,
            _ => TokenKind::Identifier(s),
        };

        Token { kind, span }
    }

    /// 解析字符字面量：`'a'`、`'\n'`、`'\t'`、`'\r'`、`'\\'`、`'\''`、`'\0'` 等。
    /// 调用前 `self.peek()` 必须是开头的 `'`。返回 `TokenKind::CharLiteral(char)`。
    /// 风格参考 `read_string` 的转义处理。
    fn read_char(&mut self) -> TenthResult<Token> {
        let span = self.span();
        self.advance(); // consume opening '

        let c = match self.peek() {
            None => {
                return Err(TenthError::LexerError {
                    line: span.line,
                    col: span.col,
                    message: "字符字面量未闭合".into(),
                });
            }
            Some('\\') => {
                self.advance(); // consume backslash
                let escaped = match self.peek() {
                    Some('n') => { self.advance(); '\n' }
                    Some('t') => { self.advance(); '\t' }
                    Some('r') => { self.advance(); '\r' }
                    Some('\\') => { self.advance(); '\\' }
                    Some('\'') => { self.advance(); '\'' }
                    Some('"') => { self.advance(); '"' }
                    Some('0') => { self.advance(); '\0' }
                    Some(c) => { self.advance(); c }
                    None => {
                        return Err(TenthError::LexerError {
                            line: span.line,
                            col: span.col,
                            message: "字符字面量转义序列不完整".into(),
                        });
                    }
                };
                escaped
            }
            Some(c) => {
                self.advance();
                c
            }
        };

        // 期望闭合的 `'`
        if self.peek() != Some('\'') {
            return Err(TenthError::LexerError {
                line: span.line,
                col: span.col,
                message: "字符字面量未闭合（缺少结尾 '）".into(),
            });
        }
        self.advance(); // consume closing '

        Ok(Token {
            kind: TokenKind::CharLiteral(c),
            span,
        })
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

    /// 解析字节串字面量：`b"..."`。
    /// 仅允许 ASCII 字符 + `\xNN` 转义序列。调用前已消费 `b` 和 `"`。
    fn read_byte_string(&mut self) -> TenthResult<Token> {
        let span = self.span();
        // 已消费了 b 和 " — 直接读取内容
        let mut bytes: Vec<u8> = Vec::new();
        while let Some(ch) = self.peek() {
            if ch == '"' {
                self.advance(); // consume closing "
                return Ok(Token {
                    kind: TokenKind::ByteString(bytes),
                    span,
                });
            }
            if ch == '\\' {
                self.advance();
                match self.peek() {
                    Some('x') => {
                        // \xNN hex escape
                        self.advance();
                        let hi = self.peek().and_then(|c| c.to_digit(16));
                        self.advance();
                        let lo = self.peek().and_then(|c| c.to_digit(16));
                        match (hi, lo) {
                            (Some(h), Some(l)) => {
                                self.advance();
                                bytes.push((h as u8) * 16 + l as u8);
                            }
                            _ => {
                                return Err(TenthError::LexerError {
                                    line: span.line,
                                    col: span.col,
                                    message: "无效的字节串转义序列：\\x 后需要两个十六进制数字".into(),
                                });
                            }
                        }
                    }
                    Some('n') => { self.advance(); bytes.push(b'\n'); }
                    Some('r') => { self.advance(); bytes.push(b'\r'); }
                    Some('t') => { self.advance(); bytes.push(b'\t'); }
                    Some('\\') => { self.advance(); bytes.push(b'\\'); }
                    Some('"') => { self.advance(); bytes.push(b'"'); }
                    Some(c) => {
                        self.advance();
                        return Err(TenthError::LexerError {
                            line: span.line,
                            col: span.col,
                            message: format!("字节串中不支持的转义序列：\\{}", c),
                        });
                    }
                    None => {
                        return Err(TenthError::LexerError {
                            line: span.line,
                            col: span.col,
                            message: "字节串中不完整的转义序列".into(),
                        });
                    }
                }
            } else if ch.is_ascii() {
                let byte = ch as u8;
                if byte > 0x7F {
                    return Err(TenthError::LexerError {
                        line: span.line,
                        col: span.col,
                        message: format!("字节串中不允许非 ASCII 字符：'{}'", ch),
                    });
                }
                bytes.push(byte);
                self.advance();
            } else {
                return Err(TenthError::LexerError {
                    line: span.line,
                    col: span.col,
                    message: format!("字节串中不允许非 ASCII 字符：'{}'", ch),
                });
            }
        }
        Err(TenthError::LexerError {
            line: span.line,
            col: span.col,
            message: "字节串未闭合".into(),
        })
    }

    /// 解析原始字符串字面量：`r"..."`。
    /// 不处理任何转义序列，直接读取到下一个 `"`。
    /// 调用前已消费 `r` 和 `"`。
    fn read_raw_string(&mut self) -> TenthResult<Token> {
        let span = self.span();
        let mut s = String::new();
        while let Some(ch) = self.peek() {
            if ch == '"' {
                self.advance(); // consume closing "
                return Ok(Token {
                    kind: TokenKind::RawString(s),
                    span,
                });
            }
            s.push(ch);
            self.advance();
        }
        Err(TenthError::LexerError {
            line: span.line,
            col: span.col,
            message: "原始字符串未闭合".into(),
        })
    }

    /// 解析多行字符串字面量：`"""..."""`。
    /// 调用前已消费开头的 `"""`。读取直到遇关闭 `"""`。
    fn read_multiline_string(&mut self) -> TenthResult<Token> {
        let span = self.span();
        let mut s = String::new();
        // 已消费三个 "，直接读取内容
        loop {
            // Check for closing """
            if self.peek() == Some('"') && self.peek_next() == Some('"')
                && self.source.get(self.pos + 2).copied() == Some('"')
            {
                self.advance(); // consume 1st "
                self.advance(); // consume 2nd "
                self.advance(); // consume 3rd "
                return Ok(Token {
                    kind: TokenKind::MultiLineString(s),
                    span,
                });
            }
            match self.peek() {
                None => {
                    return Err(TenthError::LexerError {
                        line: span.line,
                        col: span.col,
                        message: "多行字符串未闭合".into(),
                    });
                }
                Some(ch) => {
                    s.push(ch);
                    self.advance();
                }
            }
        }
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
            '#' => Some(TokenKind::Hash),
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

        // 字节串字面量 `b"..."`
        if ch == 'b' && self.peek_next() == Some('"') {
            self.advance(); // consume 'b'
            self.advance(); // consume '"'
            return self.read_byte_string();
        }

        // 原始字符串字面量 `r"..."`
        if ch == 'r' && self.peek_next() == Some('"') {
            self.advance(); // consume 'r'
            self.advance(); // consume '"'
            return self.read_raw_string();
        }

        // f-字符串字面量 `f"..."`（模板字符串 → 编译为 format() 调用）
        if ch == 'f' && self.peek_next() == Some('"') {
            self.advance(); // consume 'f'
            // read_string 已经支持 {expr} 插值，生成 InterpolatedString token
            // 我们将其包装为 FString token 以便 HIR 层区分
            let mut token = self.read_string()?;
            if let TokenKind::InterpolatedString(parts) = token.kind {
                token.kind = TokenKind::FString(parts);
            }
            return Ok(token);
        }

        if ch.is_alphabetic() || ch == '_' {
            self.advance();
            return Ok(self.read_identifier(ch));
        }

        if ch == '"' {
            // 多行字符串 `"""..."""`
            if self.peek_next() == Some('"') && self.source.get(self.pos + 2).copied() == Some('"') {
                self.advance(); // consume 1st "
                self.advance(); // consume 2nd "
                self.advance(); // consume 3rd "
                return self.read_multiline_string();
            }
            return self.read_string();
        }

        if ch == '\'' {
            // Lifetime: 'ident  — 开引号后跟标识符，且标识符后不是闭引号
            let next = self.peek_next();
            if next.map_or(false, |c| c.is_alphabetic() || c == '_') {
                // 先预读标识符，然后检查后面是不是闭引号
                let saved_pos = self.pos;
                self.advance(); // consume '
                let mut name = String::new();
                while let Some(c) = self.peek() {
                    if c.is_alphanumeric() || c == '_' {
                        name.push(c);
                        self.advance();
                    } else { break; }
                }
                if self.peek() == Some('\'') {
                    // 后面紧跟 ' — 这是字符字面量，回退走 read_char 路径
                    self.pos = saved_pos;
                    return self.read_char();
                }
                // 否则是生命周期标注
                return Ok(Token {
                    kind: TokenKind::Lifetime(name),
                    span,
                });
            }
            return self.read_char();
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
                if self.peek() == Some('.') {
                    self.advance();
                    return Ok(Token { kind: TokenKind::DotDotDot, span });
                }
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

        // M3.1：自定义运算符。`@`/`$`/`~` 三个字符的连续组合
        // （如 `@@`、`@~`、`$@$`）。这三个字符此前在 lexer 中无任何
        // 用途（会报"意外字符"），因此与全部内置 token 零冲突。
        if matches!(ch, '@' | '$' | '~') {
            let mut s = String::new();
            s.push(ch);
            while let Some(c) = self.peek() {
                if matches!(c, '@' | '$' | '~') {
                    s.push(c);
                    self.advance();
                } else {
                    break;
                }
            }
            return Ok(Token {
                kind: TokenKind::CustomOperator(s),
                span,
            });
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
