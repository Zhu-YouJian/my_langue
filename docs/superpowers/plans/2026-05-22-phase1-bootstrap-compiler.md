# Phase 1: Tenth Bootstrap 编译器实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 用 Rust 实现 Tenth 语言的最小原型编译器，支持词法分析、语法解析、HIR 生成、基础类型检查、张量操作解释执行，并提供一个可交互的 REPL。

**Architecture:** 经典编译器前端管线：Lexer → Parser → HIR Lowering + Type Check → Interpreter。全 CPU 解释执行，不涉及 MLIR/LLVM。张量操作用 Rust 的 ndarray 库在解释器中直接实现。

**Tech Stack:** Rust 2024 edition, `ndarray` 用于张量计算, `rustyline` 用于 REPL 交互。

**自举说明:** 此编译器为 bootstrap 版本（v0），用 Rust 编写。目标是产出足够编译后续 Tenth 自举编译器（v1）的最小功能集。不追求完美错误信息或完整标准库。

---

## 文件结构

```
tenth/
├── Cargo.toml
└── src/
    ├── main.rs                    // REPL 入口
    ├── repl.rs                    // REPL 循环
    ├── lexer/
    │   ├── mod.rs
    │   ├── token.rs               // Token 类型定义
    │   └── lexer.rs               // 词法分析器
    ├── parser/
    │   ├── mod.rs
    │   ├── ast.rs                 // AST 节点定义
    │   └── parser.rs              // 递归下降解析器
    ├── hir/
    │   ├── mod.rs
    │   ├── types.rs               // 类型系统定义
    │   ├── hir.rs                 // HIR 节点定义
    │   └── lower.rs               // AST → HIR 降级 + 类型检查
    ├── runtime/
    │   ├── mod.rs
    │   ├── value.rs               // 运行时值（带类型标签）
    │   ├── tensor.rs              // 张量运行时实现
    │   └── interpreter.rs         // 树遍历解释器
    └── error.rs                   // 错误类型定义

tests/
├── lexer_test.rs
├── parser_test.rs
├── hir_test.rs
├── interpreter_test.rs
└── integration_test.rs
```

---

### Task 1: 项目初始化与 Cargo 配置

**Files:**
- Create: `tenth/Cargo.toml`
- Create: `tenth/src/main.rs`
- Create: `tenth/src/error.rs`

- [ ] **Step 1: 创建 Rust 项目**

```bash
cd /workspace && cargo init tenth --name tenth
```

- [ ] **Step 2: 配置 Cargo.toml**

编辑 `tenth/Cargo.toml`：

```toml
[package]
name = "tenth"
version = "0.1.0"
edition = "2024"
description = "Tenth language bootstrap compiler"

[dependencies]
ndarray = "0.16"
rustyline = "15"
thiserror = "2"
```

- [ ] **Step 3: 创建错误类型模块**

写入 `tenth/src/error.rs`：

```rust
use thiserror::Error;

#[derive(Error, Debug, Clone, PartialEq)]
pub enum TenthError {
    #[error("Lexer error at line {line}, col {col}: {message}")]
    LexerError {
        line: usize,
        col: usize,
        message: String,
    },

    #[error("Parser error at line {line}, col {col}: {message}")]
    ParseError {
        line: usize,
        col: usize,
        message: String,
    },

    #[error("Type error at line {line}, col {col}: {message}")]
    TypeError {
        line: usize,
        col: usize,
        message: String,
    },

    #[error("Runtime error: {message}")]
    RuntimeError { message: String },

    #[error("Unexpected end of input")]
    UnexpectedEof,
}

pub type TenthResult<T> = Result<T, TenthError>;
```

- [ ] **Step 4: 写入 main.rs 骨架**

```rust
mod error;

use error::TenthResult;

fn main() -> TenthResult<()> {
    println!("Tenth v0.1.0 — bootstrap compiler");
    println!("Type ':q' to quit, ':h' for help");
    Ok(())
}
```

- [ ] **Step 5: 验证编译**

```bash
cd /workspace/tenth && cargo build
```

期望：编译成功，输出 "Tenth v0.1.0 — bootstrap compiler"

---

### Task 2: 词法分析器 — Token 类型定义

**Files:**
- Create: `tenth/src/lexer/mod.rs`
- Create: `tenth/src/lexer/token.rs`

- [ ] **Step 1: 写入 token.rs**

```rust
use std::fmt;

#[derive(Debug, Clone, PartialEq)]
pub enum TokenKind {
    // 字面量
    IntLiteral(i64),
    FloatLiteral(f64),
    StringLiteral(String),
    CharLiteral(char),

    // 标识符与关键字
    Identifier(String),
    Fn,
    Let,
    Mut,
    If,
    Else,
    Match,
    For,
    While,
    Loop,
    Break,
    Continue,
    Return,
    Try,
    Use,
    Mod,
    Pub,
    Trait,
    Impl,
    Enum,
    Struct,
    Type,
    Self_,
    Spawn,
    Task,
    Shard,
    Node,
    Macro,
    Where,
    As,
    In,
    True,
    False,

    // 运算符
    Plus,
    Minus,
    Star,
    Slash,
    Percent,
    EqEq,
    NotEq,
    Lt,
    Gt,
    LtEq,
    GtEq,
    AndAnd,
    OrOr,
    Not,
    Ampersand,
    Pipe,
    Caret,
    Shl,
    Shr,
    Assign,
    PlusAssign,
    MinusAssign,
    StarAssign,
    SlashAssign,

    // 分隔符
    LParen,
    RParen,
    LBracket,
    RBracket,
    LBrace,
    RBrace,
    Comma,
    Semicolon,
    Colon,
    Dot,
    DotDot,
    DotDotEq,
    Arrow,
    FatArrow,
    ColonColon,

    // 其他
    Eof,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Span {
    pub line: usize,
    pub col: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Token {
    pub kind: TokenKind,
    pub span: Span,
}

impl fmt::Display for TokenKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TokenKind::IntLiteral(n) => write!(f, "{}", n),
            TokenKind::FloatLiteral(n) => write!(f, "{}", n),
            TokenKind::StringLiteral(s) => write!(f, "\"{}\"", s),
            TokenKind::CharLiteral(c) => write!(f, "'{}'", c),
            TokenKind::Identifier(s) => write!(f, "{}", s),
            TokenKind::Fn => write!(f, "fn"),
            TokenKind::Let => write!(f, "let"),
            TokenKind::Mut => write!(f, "mut"),
            TokenKind::If => write!(f, "if"),
            TokenKind::Else => write!(f, "else"),
            TokenKind::Match => write!(f, "match"),
            TokenKind::For => write!(f, "for"),
            TokenKind::While => write!(f, "while"),
            TokenKind::Loop => write!(f, "loop"),
            TokenKind::Break => write!(f, "break"),
            TokenKind::Continue => write!(f, "continue"),
            TokenKind::Return => write!(f, "return"),
            TokenKind::Try => write!(f, "try"),
            TokenKind::Use => write!(f, "use"),
            TokenKind::Mod => write!(f, "mod"),
            TokenKind::Pub => write!(f, "pub"),
            TokenKind::Trait => write!(f, "trait"),
            TokenKind::Impl => write!(f, "impl"),
            TokenKind::Enum => write!(f, "enum"),
            TokenKind::Struct => write!(f, "struct"),
            TokenKind::Type => write!(f, "type"),
            TokenKind::Self_ => write!(f, "self"),
            TokenKind::Spawn => write!(f, "spawn"),
            TokenKind::Task => write!(f, "task"),
            TokenKind::Node => write!(f, "node"),
            TokenKind::Macro => write!(f, "macro"),
            TokenKind::Where => write!(f, "where"),
            TokenKind::As => write!(f, "as"),
            TokenKind::In => write!(f, "in"),
            TokenKind::True => write!(f, "true"),
            TokenKind::False => write!(f, "false"),
            TokenKind::Plus => write!(f, "+"),
            TokenKind::Minus => write!(f, "-"),
            TokenKind::Star => write!(f, "*"),
            TokenKind::Slash => write!(f, "/"),
            TokenKind::Percent => write!(f, "%"),
            TokenKind::EqEq => write!(f, "=="),
            TokenKind::NotEq => write!(f, "!="),
            TokenKind::Lt => write!(f, "<"),
            TokenKind::Gt => write!(f, ">"),
            TokenKind::LtEq => write!(f, "<="),
            TokenKind::GtEq => write!(f, ">="),
            TokenKind::AndAnd => write!(f, "&&"),
            TokenKind::OrOr => write!(f, "||"),
            TokenKind::Not => write!(f, "!"),
            TokenKind::Ampersand => write!(f, "&"),
            TokenKind::Pipe => write!(f, "|"),
            TokenKind::Caret => write!(f, "^"),
            TokenKind::Shl => write!(f, "<<"),
            TokenKind::Shr => write!(f, ">>"),
            TokenKind::Assign => write!(f, "="),
            TokenKind::PlusAssign => write!(f, "+="),
            TokenKind::MinusAssign => write!(f, "-="),
            TokenKind::StarAssign => write!(f, "*="),
            TokenKind::SlashAssign => write!(f, "/="),
            TokenKind::LParen => write!(f, "("),
            TokenKind::RParen => write!(f, ")"),
            TokenKind::LBracket => write!(f, "["),
            TokenKind::RBracket => write!(f, "]"),
            TokenKind::LBrace => write!(f, "{{"),
            TokenKind::RBrace => write!(f, "}}"),
            TokenKind::Comma => write!(f, ","),
            TokenKind::Semicolon => write!(f, ";"),
            TokenKind::Colon => write!(f, ":"),
            TokenKind::Dot => write!(f, "."),
            TokenKind::DotDot => write!(f, ".."),
            TokenKind::DotDotEq => write!(f, "..="),
            TokenKind::Arrow => write!(f, "->"),
            TokenKind::FatArrow => write!(f, "=>"),
            TokenKind::ColonColon => write!(f, "::"),
            TokenKind::Eof => write!(f, "<EOF>"),
        }
    }
}
```

- [ ] **Step 2: 写入 lexer/mod.rs**

```rust
pub mod token;
pub mod lexer;
```

- [ ] **Step 3: 验证编译**

```bash
cd /workspace/tenth && cargo build
```

---

### Task 3: 词法分析器 — Lexer 实现

**Files:**
- Create: `tenth/src/lexer/lexer.rs`

- [ ] **Step 1: 写入 lexer.rs 核心结构**

```rust
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
                // 行注释：跳过直到换行
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
```

- [ ] **Step 2: 添加数字字面量解析**

在 `impl Lexer` 块中追加：

```rust
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

        // 浮点数
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
```

- [ ] **Step 3: 添加标识符/关键字解析**

```rust
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
            "node" => TokenKind::Node,
            "macro" => TokenKind::Macro,
            "where" => TokenKind::Where,
            "as" => TokenKind::As,
            "in" => TokenKind::In,
            "true" => TokenKind::True,
            "false" => TokenKind::False,
            _ => TokenKind::Identifier(s),
        };

        Token { kind, span }
    }
```

- [ ] **Step 4: 添加字符串字面量解析**

```rust
    fn read_string(&mut self) -> TenthResult<Token> {
        let span = self.span();
        self.advance(); // 跳过开头的 "
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
```

- [ ] **Step 5: 添加单字符 token 与双字符 token 识别**

```rust
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
            '~' => Some(TokenKind::Not), // ~ 作为按位取反
            _ => None,
        }
    }

    fn two_char_operator(
        &self,
        first: char,
        second: char,
        one: TokenKind,
        two: TokenKind,
    ) -> Option<TokenKind> {
        if self.peek_next() == Some(second) {
            self.advance();
            Some(two)
        } else {
            Some(one)
        }
    }
```

- [ ] **Step 6: 实现 next_token 主方法**

```rust
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

        // 数字
        if ch.is_ascii_digit() {
            self.advance();
            return self.read_number(ch);
        }

        // 标识符
        if ch.is_alphabetic() || ch == '_' {
            self.advance();
            return Ok(self.read_identifier(ch));
        }

        // 字符串
        if ch == '"' {
            return self.read_string();
        }

        self.advance();

        // 双字符运算符
        if ch == '=' {
            if self.peek() == Some('=') {
                self.advance();
                return Ok(Token { kind: TokenKind::EqEq, span });
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
        if ch == '=' {
            if self.peek() == Some('>') {
                self.advance();
                return Ok(Token { kind: TokenKind::FatArrow, span });
            }
            return Ok(Token { kind: TokenKind::Assign, span });
        }

        // 单字符
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
```

- [ ] **Step 7: 添加测试**

创建 `tenth/tests/lexer_test.rs`：

```rust
use tenth::lexer::lexer::Lexer;
use tenth::lexer::token::TokenKind;

fn tokenize(src: &str) -> Vec<TokenKind> {
    let mut lexer = Lexer::new(src);
    lexer.tokenize().unwrap().into_iter().map(|t| t.kind).collect()
}

#[test]
fn test_integers() {
    let tokens = tokenize("42 0 100");
    assert_eq!(tokens[0], TokenKind::IntLiteral(42));
    assert_eq!(tokens[1], TokenKind::IntLiteral(0));
    assert_eq!(tokens[2], TokenKind::IntLiteral(100));
}

#[test]
fn test_keywords() {
    let tokens = tokenize("fn let mut if else match for while return");
    assert_eq!(tokens[0], TokenKind::Fn);
    assert_eq!(tokens[1], TokenKind::Let);
    assert_eq!(tokens[2], TokenKind::Mut);
    assert_eq!(tokens[3], TokenKind::If);
    assert_eq!(tokens[4], TokenKind::Else);
    assert_eq!(tokens[5], TokenKind::Match);
    assert_eq!(tokens[6], TokenKind::For);
    assert_eq!(tokens[7], TokenKind::While);
    assert_eq!(tokens[8], TokenKind::Return);
}

#[test]
fn test_operators() {
    let tokens = tokenize("+ - * / == != < > <= >= && || !");
    assert_eq!(tokens[0], TokenKind::Plus);
    assert_eq!(tokens[1], TokenKind::Minus);
    assert_eq!(tokens[2], TokenKind::Star);
    assert_eq!(tokens[3], TokenKind::Slash);
    assert_eq!(tokens[4], TokenKind::EqEq);
    assert_eq!(tokens[5], TokenKind::NotEq);
    assert_eq!(tokens[6], TokenKind::Lt);
    assert_eq!(tokens[7], TokenKind::Gt);
    assert_eq!(tokens[8], TokenKind::LtEq);
    assert_eq!(tokens[9], TokenKind::GtEq);
    assert_eq!(tokens[10], TokenKind::AndAnd);
    assert_eq!(tokens[11], TokenKind::OrOr);
    assert_eq!(tokens[12], TokenKind::Not);
}

#[test]
fn test_string_literal() {
    let tokens = tokenize("\"hello world\"");
    assert_eq!(tokens[0], TokenKind::StringLiteral("hello world".into()));
}

#[test]
fn test_comment_skip() {
    let tokens = tokenize("// this is a comment\n42");
    assert_eq!(tokens[0], TokenKind::IntLiteral(42));
}

#[test]
fn test_identifier() {
    let tokens = tokenize("my_var tensor randn");
    assert_eq!(tokens[0], TokenKind::Identifier("my_var".into()));
    assert_eq!(tokens[1], TokenKind::Identifier("tensor".into()));
    assert_eq!(tokens[2], TokenKind::Identifier("randn".into()));
}
```

更新 `tenth/src/main.rs` 顶部添加模块声明：

```rust
pub mod error;
pub mod lexer;
```

- [ ] **Step 8: 运行测试**

```bash
cd /workspace/tenth && cargo test
```

期望：所有 lexer 测试通过。

---

### Task 4: 语法解析器 — AST 定义

**Files:**
- Create: `tenth/src/parser/mod.rs`
- Create: `tenth/src/parser/ast.rs`

- [ ] **Step 1: 写入 ast.rs**

```rust
use crate::lexer::token::Span;

#[derive(Debug, Clone, PartialEq)]
pub enum Literal {
    Int(i64),
    Float(f64),
    Bool(bool),
    String(String),
}

#[derive(Debug, Clone, PartialEq)]
pub struct Ident {
    pub name: String,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub enum TypeAnnotation {
    Named(Ident),
    Tensor {
        dtype: Box<TypeAnnotation>,
        dims: Vec<DimSpec>,
    },
    Array(Box<TypeAnnotation>), // [T]
    FnType {
        params: Vec<TypeAnnotation>,
        ret: Box<TypeAnnotation>,
    },
    Unit,
}

#[derive(Debug, Clone, PartialEq)]
pub enum DimSpec {
    Literal(i64),
    Symbol(String),
    Wildcard,
}

#[derive(Debug, Clone, PartialEq)]
pub enum BinOp {
    Add, Sub, Mul, Div, Mod,
    Eq, NotEq, Lt, Gt, LtEq, GtEq,
    And, Or,
}

#[derive(Debug, Clone, PartialEq)]
pub enum UnaryOp {
    Neg, Not,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ExprKind {
    Literal(Literal),
    Ident(Ident),
    Binary {
        op: BinOp,
        left: Box<Expr>,
        right: Box<Expr>,
    },
    Unary {
        op: UnaryOp,
        expr: Box<Expr>,
    },
    Call {
        func: Box<Expr>,
        args: Vec<Expr>,
    },
    MethodCall {
        receiver: Box<Expr>,
        method: Ident,
        args: Vec<Expr>,
    },
    Index {
        target: Box<Expr>,
        indices: Vec<IndexExpr>,
    },
    Field {
        target: Box<Expr>,
        field: Ident,
    },
    TensorLiteral(Vec<Vec<Expr>>),
    Range {
        start: Option<Box<Expr>>,
        end: Option<Box<Expr>>,
        inclusive: bool,
    },
    If {
        cond: Box<Expr>,
        then_branch: Box<Expr>,
        else_branch: Option<Box<Expr>>,
    },
    Block(Vec<Stmt>),
    Closure {
        params: Vec<(Ident, Option<TypeAnnotation>)>,
        body: Box<Expr>,
    },
    Assign {
        target: Box<Expr>,
        value: Box<Expr>,
    },
    AssignOp {
        target: Box<Expr>,
        op: BinOp,
        value: Box<Expr>,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub struct Expr {
    pub kind: ExprKind,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub enum IndexExpr {
    Single(Expr),
    Range {
        start: Option<Box<Expr>>,
        end: Option<Box<Expr>>,
    },
    Colon,
}

#[derive(Debug, Clone, PartialEq)]
pub enum StmtKind {
    Let {
        name: Ident,
        type_ann: Option<TypeAnnotation>,
        mutable: bool,
        init: Option<Expr>,
    },
    Expr(Expr),
    Return(Option<Expr>),
    Break,
    Continue,
    While {
        cond: Expr,
        body: Box<Stmt>,
    },
    For {
        var: Ident,
        iter: Expr,
        body: Box<Stmt>,
    },
    Loop {
        body: Vec<Stmt>,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub struct Stmt {
    pub kind: StmtKind,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Param {
    pub name: Ident,
    pub type_ann: TypeAnnotation,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ItemKind {
    Function {
        name: Ident,
        params: Vec<Param>,
        return_type: Option<TypeAnnotation>,
        body: Expr,
    },
    Const {
        name: Ident,
        type_ann: TypeAnnotation,
        value: Expr,
    },
    Use {
        path: Vec<Ident>,
    },
    Mod {
        name: Ident,
        items: Vec<Item>,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub struct Item {
    pub kind: ItemKind,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Program {
    pub items: Vec<Item>,
}
```

- [ ] **Step 2: 写入 parser/mod.rs**

```rust
pub mod ast;
pub mod parser;
```

- [ ] **Step 3: 验证编译**

```bash
cd /workspace/tenth && cargo build
```

---

### Task 5: 语法解析器 — Parser 实现

**Files:**
- Create: `tenth/src/parser/parser.rs`

由于 parser 代码量较大，分多个 step 写入。

- [ ] **Step 1: 写入 parser 骨架与 token 游标**

```rust
use crate::error::{TenthError, TenthResult};
use crate::lexer::token::{Span, Token, TokenKind};
use super::ast::*;

pub struct Parser {
    tokens: Vec<Token>,
    pos: usize,
}

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
        let token = self.peek();
        self.pos += 1;
        unsafe { std::mem::transmute::<&Token, &Token>(token) }
    }

    fn expect(&mut self, kind: TokenKind) -> TenthResult<&Token> {
        let token = self.peek();
        if std::mem::discriminant(&token.kind) == std::mem::discriminant(&kind) {
            self.pos += 1;
            Ok(token)
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
}

static EOF_TOKEN: Token = Token {
    kind: TokenKind::Eof,
    span: Span { line: 0, col: 0 },
};
```

- [ ] **Step 2: 添加初级表达式解析**

```rust
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
            TokenKind::While => {
                let cond = self.parse_expr()?;
                let body = self.parse_expr()?;
                return Ok(Expr {
                    kind: ExprKind::Block(vec![Stmt {
                        kind: StmtKind::While {
                            cond,
                            body: Box::new(Stmt {
                                kind: StmtKind::Expr(body),
                                span: span.clone(),
                            }),
                        },
                        span: span.clone(),
                    }]),
                    span,
                });
            }
            TokenKind::Pipe => {
                return self.parse_closure_after_pipe(span);
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
```

- [ ] **Step 3: 添加后缀表达式（调用、索引、字段访问）**

```rust
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
                    expr = Expr {
                        kind: ExprKind::Call {
                            func: Box::new(expr),
                            args,
                        },
                        span: expr.span.clone(),
                    };
                }
                TokenKind::Dot => {
                    self.advance();
                    if let TokenKind::Identifier(name) = &self.peek().kind {
                        let name = name.clone();
                        let method_span = self.peek().span.clone();
                        self.advance();

                        if matches!(self.peek_kind(), TokenKind::LParen) {
                            // 方法调用
                            self.advance();
                            let mut args = Vec::new();
                            if !matches!(self.peek_kind(), TokenKind::RParen) {
                                args = self.parse_arg_list()?;
                            }
                            self.expect(TokenKind::RParen)?;
                            expr = Expr {
                                kind: ExprKind::MethodCall {
                                    receiver: Box::new(expr),
                                    method: Ident { name, span: method_span },
                                    args,
                                },
                                span: expr.span.clone(),
                            };
                        } else {
                            // 字段访问
                            expr = Expr {
                                kind: ExprKind::Field {
                                    target: Box::new(expr),
                                    field: Ident { name, span: method_span },
                                },
                                span: expr.span.clone(),
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
                    expr = Expr {
                        kind: ExprKind::Index {
                            target: Box::new(expr),
                            indices,
                        },
                        span: expr.span.clone(),
                    };
                }
                _ => break,
            }
        }

        Ok(expr)
    }
```

- [ ] **Step 4: 添加一元表达式与二元表达式（Pratt 解析）**

```rust
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
                left = Expr {
                    kind: ExprKind::Assign {
                        target: Box::new(left),
                        value: Box::new(right),
                    },
                    span: left.span.clone(),
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
                left = Expr {
                    kind: ExprKind::AssignOp {
                        target: Box::new(left),
                        op,
                        value: Box::new(right),
                    },
                    span: left.span.clone(),
                };
                continue;
            }

            let op_kind = self.peek_kind().clone();
            self.advance();
            let op = Self::token_to_binop(&op_kind).unwrap();
            let right = self.parse_binary(prec + 1)?;

            left = Expr {
                kind: ExprKind::Binary {
                    op,
                    left: Box::new(left),
                    right: Box::new(right),
                },
                span: left.span.clone(),
            };
        }

        Ok(left)
    }

    fn parse_expr(&mut self) -> TenthResult<Expr> {
        self.parse_binary(0)
    }
```

- [ ] **Step 5: 添加闭包、参数列表、类型标注解析**

```rust
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
                // 检查是否是 Tensor
                if name == "Tensor" && matches!(self.peek_kind(), TokenKind::LBracket) {
                    self.advance(); // [
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
                                if s == ".." {
                                    dims.push(DimSpec::Wildcard);
                                } else {
                                    dims.push(DimSpec::Symbol(s));
                                }
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
                    Ok(TypeAnnotation::Named(Ident {
                        name,
                        span,
                    }))
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
```

- [ ] **Step 6: 添加语句解析与顶层项解析**

```rust
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
            TokenKind::For => {
                self.advance();
                let var = if let TokenKind::Identifier(name) = &self.peek().kind {
                    Ident {
                        name: name.clone(),
                        span: self.peek().span.clone(),
                    }
                } else {
                    return Err(TenthError::ParseError {
                        line: self.peek().span.line,
                        col: self.peek().span.col,
                        message: "expected loop variable".into(),
                    });
                };
                self.advance();
                self.expect(TokenKind::In)?;
                let iter = self.parse_expr()?;
                let body = self.parse_expr()?;
                Ok(Stmt {
                    kind: StmtKind::For {
                        var,
                        iter,
                        body: Box::new(Stmt {
                            kind: StmtKind::Expr(body),
                            span: span.clone(),
                        }),
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
            items.push(self.parse_item()?);
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

                let body = self.parse_expr()?;

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
                        name: Ident { name: "<expr>".into(), span: span.clone() },
                        params: Vec::new(),
                        return_type: None,
                        body: expr,
                    },
                    span,
                })
            }
        }
    }
}
```

- [ ] **Step 7: 更新 main.rs 添加 parser 模块, 验证编译**

更新 `tenth/src/main.rs`：

```rust
pub mod error;
pub mod lexer;
pub mod parser;
```

```bash
cd /workspace/tenth && cargo build
```

---

### Task 6: 类型系统定义

**Files:**
- Create: `tenth/src/hir/mod.rs`
- Create: `tenth/src/hir/types.rs`
- Create: `tenth/src/hir/hir.rs`

- [ ] **Step 1: 写入 types.rs**

```rust
use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum BaseType {
    I8, I16, I32, I64,
    U8, U16, U32, U64,
    F16, F32, F64, BF16,
    Bool, Char, Str,
    Unit,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Dim {
    Known(i64),
    Symbol(String),
    Any,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Type {
    Base(BaseType),
    Tensor {
        dtype: BaseType,
        dims: Vec<Dim>,
    },
    Array(Box<Type>),
    FnType {
        params: Vec<Type>,
        ret: Box<Type>,
    },
    Unknown,
}

impl fmt::Display for Type {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Type::Base(b) => write!(f, "{:?}", b),
            Type::Tensor { dtype, dims } => {
                write!(f, "Tensor[{:?}", dtype)?;
                for dim in dims {
                    match dim {
                        Dim::Known(n) => write!(f, ", {}", n)?,
                        Dim::Symbol(s) => write!(f, ", {}", s)?,
                        Dim::Any => write!(f, ", ..")?,
                    }
                }
                write!(f, "]")
            }
            Type::Array(t) => write!(f, "[{}]", t),
            Type::FnType { params, ret } => {
                write!(f, "fn(")?;
                for (i, p) in params.iter().enumerate() {
                    if i > 0 { write!(f, ", ")?; }
                    write!(f, "{}", p)?;
                }
                write!(f, ") -> {}", ret)
            }
            Type::Unknown => write!(f, "<unknown>"),
        }
    }
}

impl Type {
    pub fn i32() -> Self { Type::Base(BaseType::I32) }
    pub fn f64() -> Self { Type::Base(BaseType::F64) }
    pub fn f32() -> Self { Type::Base(BaseType::F32) }
    pub fn bool_() -> Self { Type::Base(BaseType::Bool) }
    pub fn str_() -> Self { Type::Base(BaseType::Str) }
    pub fn unit() -> Self { Type::Base(BaseType::Unit) }

    pub fn tensor(dtype: BaseType, dims: Vec<Dim>) -> Self {
        Type::Tensor { dtype, dims }
    }

    pub fn from_annotation(ann: &super::super::parser::ast::TypeAnnotation) -> Self {
        use super::super::parser::ast::TypeAnnotation as TA;
        match ann {
            TA::Named(ident) => {
                match ident.name.as_str() {
                    "i8" => Type::Base(BaseType::I8),
                    "i16" => Type::Base(BaseType::I16),
                    "i32" => Type::Base(BaseType::I32),
                    "i64" => Type::Base(BaseType::I64),
                    "u8" => Type::Base(BaseType::U8),
                    "u16" => Type::Base(BaseType::U16),
                    "u32" => Type::Base(BaseType::U32),
                    "u64" => Type::Base(BaseType::U64),
                    "f16" => Type::Base(BaseType::F16),
                    "f32" => Type::Base(BaseType::F32),
                    "f64" => Type::Base(BaseType::F64),
                    "bf16" => Type::Base(BaseType::BF16),
                    "bool" => Type::Base(BaseType::Bool),
                    "char" => Type::Base(BaseType::Char),
                    "str" => Type::Base(BaseType::Str),
                    _ => Type::Unknown,
                }
            }
            TA::Tensor { dtype, dims } => {
                let dt = Self::from_annotation(dtype);
                let base = match dt {
                    Type::Base(b) => b,
                    _ => BaseType::F32,
                };
                let resolved_dims: Vec<Dim> = dims.iter().map(|d| match d {
                    super::super::parser::ast::DimSpec::Literal(n) => Dim::Known(*n),
                    super::super::parser::ast::DimSpec::Symbol(s) => Dim::Symbol(s.clone()),
                    super::super::parser::ast::DimSpec::Wildcard => Dim::Any,
                }).collect();
                Type::Tensor { dtype: base, dims: resolved_dims }
            }
            TA::Array(inner) => Type::Array(Box::new(Self::from_annotation(inner))),
            TA::FnType { params, ret } => Type::FnType {
                params: params.iter().map(Self::from_annotation).collect(),
                ret: Box::new(Self::from_annotation(ret)),
            },
            TA::Unit => Type::Unit,
        }
    }
}
```

- [ ] **Step 2: 写入 hir.rs**

```rust
use crate::lexer::token::Span;
use super::types::Type;

#[derive(Debug, Clone, PartialEq)]
pub enum HirExprKind {
    Literal(Literal),
    Var(String),
    Binary {
        op: BinOp,
        left: Box<HirExpr>,
        right: Box<HirExpr>,
        ty: Type,
    },
    Unary {
        op: UnaryOp,
        expr: Box<HirExpr>,
        ty: Type,
    },
    Call {
        func: Box<HirExpr>,
        args: Vec<HirExpr>,
        ret_ty: Type,
    },
    MethodCall {
        receiver: Box<HirExpr>,
        method: String,
        args: Vec<HirExpr>,
        ret_ty: Type,
    },
    Index {
        target: Box<HirExpr>,
        indices: Vec<Index>,
    },
    Field {
        target: Box<HirExpr>,
        field: String,
    },
    TensorLiteral {
        data: Vec<Vec<HirExpr>>,
        ty: Type,
    },
    Range {
        start: Option<Box<HirExpr>>,
        end: Option<Box<HirExpr>>,
        inclusive: bool,
    },
    If {
        cond: Box<HirExpr>,
        then_branch: Box<HirExpr>,
        else_branch: Option<Box<HirExpr>>,
        ty: Type,
    },
    Block {
        stmts: Vec<HirStmt>,
        final_expr: Option<Box<HirExpr>>,
    },
    Closure {
        params: Vec<(String, Type)>,
        body: Box<HirExpr>,
    },
    Assign {
        target: String,
        value: Box<HirExpr>,
    },
    AssignOp {
        target: String,
        op: BinOp,
        value: Box<HirExpr>,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub struct HirExpr {
    pub kind: HirExprKind,
    pub ty: Type,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Literal {
    Int(i64),
    Float(f64),
    Bool(bool),
    String(String),
}

#[derive(Debug, Clone, PartialEq)]
pub enum BinOp {
    Add, Sub, Mul, Div, Mod,
    Eq, NotEq, Lt, Gt, LtEq, GtEq,
    And, Or,
}

#[derive(Debug, Clone, PartialEq)]
pub enum UnaryOp {
    Neg, Not,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Index {
    Single(HirExpr),
    Range {
        start: Option<Box<HirExpr>>,
        end: Option<Box<HirExpr>>,
    },
    Colon,
}

#[derive(Debug, Clone, PartialEq)]
pub enum HirStmtKind {
    Let {
        name: String,
        type_ann: Option<Type>,
        mutable: bool,
        init: Option<HirExpr>,
    },
    Expr(HirExpr),
    Return(Option<HirExpr>),
    While {
        cond: HirExpr,
        body: Box<HirStmt>,
    },
    For {
        var: String,
        iter: HirExpr,
        body: Box<HirStmt>,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub struct HirStmt {
    pub kind: HirStmtKind,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct HirFnDef {
    pub name: String,
    pub params: Vec<(String, Type)>,
    pub return_type: Type,
    pub body: HirExpr,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct HirProgram {
    pub functions: Vec<HirFnDef>,
    pub main_expr: Option<HirExpr>,
}
```

- [ ] **Step 3: 写入 hir/mod.rs**

```rust
pub mod types;
pub mod hir;
pub mod lower;
```

更新 `tenth/src/main.rs`：

```rust
pub mod error;
pub mod lexer;
pub mod parser;
pub mod hir;
```

- [ ] **Step 4: 验证编译**

```bash
cd /workspace/tenth && cargo build
```

---

### Task 7: AST → HIR 降级与类型检查

**Files:**
- Create: `tenth/src/hir/lower.rs`

- [ ] **Step 1: 写入 lower.rs 框架与符号表**

```rust
use std::collections::HashMap;
use crate::error::{TenthError, TenthResult};
use crate::lexer::token::Span;
use crate::parser::ast as ast;
use super::hir::*;
use super::types::*;

struct Scope {
    variables: HashMap<String, (Type, bool)>, // (type, mutable)
    functions: HashMap<String, (Vec<(String, Type)>, Type)>, // (params, return_type)
    parent: Option<Box<Scope>>,
}

impl Scope {
    fn new() -> Self {
        Scope {
            variables: HashMap::new(),
            functions: HashMap::new(),
            parent: None,
        }
    }

    fn with_parent(parent: Scope) -> Self {
        Scope {
            variables: HashMap::new(),
            functions: HashMap::new(),
            parent: Some(Box::new(parent)),
        }
    }

    fn lookup_var(&self, name: &str) -> Option<(Type, bool)> {
        if let Some(v) = self.variables.get(name) {
            return Some(v.clone());
        }
        self.parent.as_ref().and_then(|p| p.lookup_var(name))
    }

    fn define_var(&mut self, name: String, ty: Type, mutable: bool) {
        self.variables.insert(name, (ty, mutable));
    }

    fn assign_var(&mut self, name: &str, _span: &Span) -> TenthResult<()> {
        if let Some((_, mutable)) = self.variables.get(name) {
            if !mutable {
                // 简化处理，暂不报错
            }
            return Ok(());
        }
        if let Some(parent) = &self.parent {
            return parent.assign_var(name, _span);
        }
        Err(TenthError::TypeError {
            line: _span.line,
            col: _span.col,
            message: format!("undefined variable '{}'", name),
        })
    }

    fn define_fn(&mut self, name: String, params: Vec<(String, Type)>, ret: Type) {
        self.functions.insert(name, (params, ret));
    }

    fn lookup_fn(&self, name: &str) -> Option<(Vec<(String, Type)>, Type)> {
        if let Some(f) = self.functions.get(name) {
            return Some(f.clone());
        }
        self.parent.as_ref().and_then(|p| p.lookup_fn(name))
    }
}

pub struct Lowerer {
    scope: Scope,
    functions: Vec<HirFnDef>,
}

impl Lowerer {
    pub fn new() -> Self {
        Lowerer {
            scope: Scope::new(),
            functions: Vec::new(),
        }
    }
```

- [ ] **Step 2: 添加表达式降级方法**

```rust
    fn lower_expr(&mut self, expr: &ast::Expr) -> TenthResult<HirExpr> {
        use ast::ExprKind;

        let span = expr.span.clone();

        let (kind, ty) = match &expr.kind {
            ExprKind::Literal(lit) => {
                let (hir_lit, ty) = match lit {
                    ast::Literal::Int(n) => (Literal::Int(*n), Type::i32()),
                    ast::Literal::Float(n) => (Literal::Float(*n), Type::f64()),
                    ast::Literal::Bool(b) => (Literal::Bool(*b), Type::bool_()),
                    ast::Literal::String(s) => (Literal::String(s.clone()), Type::str_()),
                };
                (HirExprKind::Literal(hir_lit), ty)
            }

            ExprKind::Ident(ident) => {
                let var_info = self.scope.lookup_var(&ident.name).ok_or_else(|| {
                    TenthError::TypeError {
                        line: span.line,
                        col: span.col,
                        message: format!("undefined variable '{}'", ident.name),
                    }
                })?;
                (HirExprKind::Var(ident.name.clone()), var_info.0)
            }

            ExprKind::Binary { op, left, right } => {
                let l = self.lower_expr(left)?;
                let r = self.lower_expr(right)?;
                let ty = self.infer_binary_type(op, &l.ty, &r.ty, &span)?;
                let hir_op = lower_binop(op);
                (HirExprKind::Binary { op: hir_op, left: Box::new(l), right: Box::new(r), ty: ty.clone() }, ty)
            }

            ExprKind::Unary { op, expr } => {
                let e = self.lower_expr(expr)?;
                let ty = e.ty.clone();
                let hir_op = match op {
                    ast::UnaryOp::Neg => UnaryOp::Neg,
                    ast::UnaryOp::Not => UnaryOp::Not,
                };
                (HirExprKind::Unary { op: hir_op, expr: Box::new(e), ty: ty.clone() }, ty)
            }

            ExprKind::Call { func, args } => {
                let f = self.lower_expr(func)?;
                let lowered_args: Vec<_> = args.iter()
                    .map(|a| self.lower_expr(a))
                    .collect::<TenthResult<_>>()?;

                // 基本内置函数处理
                let ret_ty = self.resolve_call_type(&f, &lowered_args, &span)?;

                (HirExprKind::Call {
                    func: Box::new(f),
                    args: lowered_args,
                    ret_ty: ret_ty.clone(),
                }, ret_ty)
            }

            ExprKind::MethodCall { receiver, method, args } => {
                let recv = self.lower_expr(receiver)?;
                let lowered_args: Vec<_> = args.iter()
                    .map(|a| self.lower_expr(a))
                    .collect::<TenthResult<_>>()?;

                let ret_ty = self.resolve_method_type(&recv.ty, &method.name, &lowered_args, &span)?;

                (HirExprKind::MethodCall {
                    receiver: Box::new(recv),
                    method: method.name.clone(),
                    args: lowered_args,
                    ret_ty: ret_ty.clone(),
                }, ret_ty)
            }

            ExprKind::Index { target, indices } => {
                let t = self.lower_expr(target)?;
                let lowered_indices: Vec<_> = indices.iter()
                    .map(|idx| self.lower_index(idx))
                    .collect::<TenthResult<_>>()?;

                // 索引操作的返回类型：去掉被索引的维度
                let ty = self.index_type(&t.ty, &lowered_indices);
                (HirExprKind::Index { target: Box::new(t), indices: lowered_indices }, ty)
            }

            ExprKind::Field { target, field } => {
                let t = self.lower_expr(target)?;
                (HirExprKind::Field { target: Box::new(t), field: field.name.clone() }, Type::Unknown)
            }

            ExprKind::TensorLiteral(data) => {
                let lowered: Vec<Vec<HirExpr>> = data.iter()
                    .map(|row| row.iter().map(|e| self.lower_expr(e)).collect())
                    .collect::<TenthResult<_>>()?;
                let rows = lowered.len() as i64;
                let cols = lowered.first().map_or(0, |r| r.len() as i64);
                let ty = Type::Tensor { dtype: BaseType::F64, dims: vec![Dim::Known(rows), Dim::Known(cols)] };
                (HirExprKind::TensorLiteral { data: lowered, ty: ty.clone() }, ty)
            }

            ExprKind::Range { start, end, inclusive } => {
                let s = start.as_ref().map(|e| self.lower_expr(e)).transpose()?;
                let e = end.as_ref().map(|e| self.lower_expr(e)).transpose()?;
                // range 类型暂时为 Unknown，由使用上下文决定
                (HirExprKind::Range { start: s.map(Box::new), end: e.map(Box::new), inclusive: *inclusive }, Type::Unknown)
            }

            ExprKind::If { cond, then_branch, else_branch } => {
                let c = self.lower_expr(cond)?;
                let t = self.lower_expr(then_branch)?;
                let e = else_branch.as_ref().map(|eb| self.lower_expr(eb)).transpose()?;
                let ty = if let Some(ref eb) = e {
                    eb.ty.clone()
                } else {
                    Type::unit()
                };
                (HirExprKind::If { cond: Box::new(c), then_branch: Box::new(t), else_branch: e.map(Box::new), ty: ty.clone() }, ty)
            }

            ExprKind::Block(stmts) => {
                let mut inner_scope = Scope::with_parent(std::mem::replace(&mut self.scope, Scope::new()));
                self.scope = inner_scope;

                let lowered_stmts: Vec<HirStmt> = stmts.iter()
                    .map(|s| self.lower_stmt(s))
                    .collect::<TenthResult<_>>()?;

                // 检查最后一个语句是否是表达式
                let final_expr = lowered_stmts.last().and_then(|s| match &s.kind {
                    HirStmtKind::Expr(e) => Some(e.clone()),
                    _ => None,
                });

                let ty = final_expr.as_ref().map(|e| e.ty.clone()).unwrap_or(Type::unit());

                // 如果最后一条是表达式语句，去掉它（移到 final_expr）
                let stmts_without_final: Vec<HirStmt> = if final_expr.is_some() {
                    lowered_stmts[..lowered_stmts.len()-1].to_vec()
                } else {
                    lowered_stmts
                };

                let outer_scope = std::mem::replace(&mut self.scope, Scope::new());
                self.scope = outer_scope;

                (HirExprKind::Block { stmts: stmts_without_final, final_expr: final_expr.map(Box::new) }, ty)
            }

            ExprKind::Closure { params, body } => {
                let lowered_params: Vec<_> = params.iter()
                    .map(|(name, ann)| {
                        let ty = ann.as_ref()
                            .map(|a| Type::from_annotation(a))
                            .unwrap_or(Type::Unknown);
                        (name.name.clone(), ty)
                    })
                    .collect();
                let b = self.lower_expr(body)?;
                (HirExprKind::Closure { params: lowered_params, body: Box::new(b) }, Type::Unknown)
            }

            ExprKind::Assign { target, value } => {
                let v = self.lower_expr(value)?;
                let name = match &target.kind {
                    ExprKind::Ident(id) => id.name.clone(),
                    _ => return Err(TenthError::ParseError {
                        line: span.line,
                        col: span.col,
                        message: "invalid assignment target".into(),
                    }),
                };
                self.scope.assign_var(&name, &span)?;
                (HirExprKind::Assign { target: name, value: Box::new(v) }, Type::unit())
            }

            ExprKind::AssignOp { target, op, value } => {
                let v = self.lower_expr(value)?;
                let name = match &target.kind {
                    ExprKind::Ident(id) => id.name.clone(),
                    _ => return Err(TenthError::ParseError {
                        line: span.line,
                        col: span.col,
                        message: "invalid assignment target".into(),
                    }),
                };
                let hir_op = lower_binop(op);
                (HirExprKind::AssignOp { target: name, op: hir_op, value: Box::new(v) }, Type::unit())
            }
        };

        Ok(HirExpr { kind, ty, span })
    }
```

- [ ] **Step 3: 添加辅助方法与语句降级**

```rust
    fn lower_index(&mut self, idx: &ast::IndexExpr) -> TenthResult<Index> {
        match idx {
            ast::IndexExpr::Single(e) => Ok(Index::Single(self.lower_expr(e)?)),
            ast::IndexExpr::Range { start, end } => {
                let s = start.as_ref().map(|e| self.lower_expr(e)).transpose()?;
                let e = end.as_ref().map(|e| self.lower_expr(e)).transpose()?;
                Ok(Index::Range { start: s.map(Box::new), end: e.map(Box::new) })
            }
            ast::IndexExpr::Colon => Ok(Index::Colon),
        }
    }

    fn index_type(&self, base: &Type, indices: &[Index]) -> Type {
        match base {
            Type::Tensor { dtype, dims } => {
                let num_removed = indices.len();
                let remaining: Vec<Dim> = dims.iter().skip(num_removed).cloned().collect();
                if remaining.is_empty() {
                    Type::Base(*dtype)
                } else {
                    Type::Tensor { dtype: *dtype, dims: remaining }
                }
            }
            _ => base.clone(),
        }
    }

    fn infer_binary_type(&self, op: &ast::BinOp, l: &Type, r: &Type, span: &Span) -> TenthResult<Type> {
        use ast::BinOp;
        match op {
            BinOp::Eq | BinOp::NotEq | BinOp::Lt | BinOp::Gt | BinOp::LtEq | BinOp::GtEq | BinOp::And | BinOp::Or => {
                Ok(Type::bool_())
            }
            BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div | BinOp::Mod => {
                match (l, r) {
                    (Type::Tensor { dtype, .. }, _) | (_, Type::Tensor { dtype, .. }) => {
                        Ok(Type::Tensor { dtype: *dtype, dims: vec![Dim::Any] })
                    }
                    _ => Ok(l.clone()),
                }
            }
        }
    }

    fn resolve_call_type(&self, func: &HirExpr, args: &[HirExpr], span: &Span) -> TenthResult<Type> {
        match &func.kind {
            HirExprKind::Var(name) => {
                if let Some((params, ret)) = self.scope.lookup_fn(name) {
                    if params.len() != args.len() {
                        return Err(TenthError::TypeError {
                            line: span.line,
                            col: span.col,
                            message: format!(
                                "function '{}' expects {} arguments, got {}",
                                name, params.len(), args.len()
                            ),
                        });
                    }
                    return Ok(ret);
                }
                // 内置函数
                self.resolve_builtin(name, args, span)
            }
            _ => Ok(Type::Unknown),
        }
    }

    fn resolve_method_type(
        &self, receiver: &Type, method: &str, args: &[HirExpr], span: &Span,
    ) -> TenthResult<Type> {
        match receiver {
            Type::Tensor { dtype, dims } => {
                match method {
                    "sum" => {
                        if args.iter().any(|a| matches!(&a.kind, HirExprKind::Var(_))) {
                            Ok(Type::Tensor { dtype: *dtype, dims: dims.clone() })
                        } else {
                            Ok(Type::Base(*dtype))
                        }
                    }
                    "mean" => Ok(Type::Base(*dtype)),
                    "max" | "min" => Ok(Type::Base(*dtype)),
                    "reshape" | "view" => Ok(Type::Tensor { dtype: *dtype, dims: vec![Dim::Any] }),
                    "flatten" => Ok(Type::Tensor { dtype: *dtype, dims: vec![Dim::Any] }),
                    "abs" | "sqrt" | "exp" | "log" | "relu" | "sigmoid" | "tanh" => {
                        Ok(Type::Tensor { dtype: *dtype, dims: dims.clone() })
                    }
                    _ => Ok(Type::Unknown),
                }
            }
            _ => Ok(Type::Unknown),
        }
    }

    fn resolve_builtin(&self, name: &str, args: &[HirExpr], span: &Span) -> TenthResult<Type> {
        match name {
            "println" | "eprintln" => Ok(Type::unit()),
            "tensor" => Ok(Type::Tensor { dtype: BaseType::F64, dims: vec![Dim::Any] }),
            _ => Err(TenthError::TypeError {
                line: span.line,
                col: span.col,
                message: format!("undefined function '{}'", name),
            }),
        }
    }

    fn lower_stmt(&mut self, stmt: &ast::Stmt) -> TenthResult<HirStmt> {
        use ast::StmtKind;

        let span = stmt.span.clone();

        let kind = match &stmt.kind {
            StmtKind::Let { name, type_ann, mutable, init } => {
                let lowered_init = init.as_ref().map(|i| self.lower_expr(i)).transpose()?;
                let ty = type_ann.as_ref()
                    .map(|a| Type::from_annotation(a))
                    .or_else(|| lowered_init.as_ref().map(|e| e.ty.clone()))
                    .unwrap_or(Type::Unknown);

                self.scope.define_var(name.name.clone(), ty.clone(), *mutable);

                HirStmtKind::Let {
                    name: name.name.clone(),
                    type_ann: type_ann.as_ref().map(|a| Type::from_annotation(a)),
                    mutable: *mutable,
                    init: lowered_init,
                }
            }
            StmtKind::Expr(e) => {
                HirStmtKind::Expr(self.lower_expr(e)?)
            }
            StmtKind::Return(e) => {
                HirStmtKind::Return(e.as_ref().map(|e| self.lower_expr(e)).transpose()?)
            }
            StmtKind::Break => return Err(TenthError::ParseError {
                line: span.line,
                col: span.col,
                message: "break not yet supported in HIR".into(),
            }),
            StmtKind::Continue => return Err(TenthError::ParseError {
                line: span.line,
                col: span.col,
                message: "continue not yet supported in HIR".into(),
            }),
            StmtKind::While { cond, body } => {
                let c = self.lower_expr(cond)?;
                let b = self.lower_stmt(body)?;
                HirStmtKind::While { cond: c, body: Box::new(b) }
            }
            StmtKind::For { var, iter, body } => {
                let it = self.lower_expr(iter)?;
                let b = self.lower_stmt(body)?;
                HirStmtKind::For { var: var.name.clone(), iter: it, body: Box::new(b) }
            }
            StmtKind::Loop { .. } => return Err(TenthError::ParseError {
                line: span.line,
                col: span.col,
                message: "loop not yet supported in HIR".into(),
            }),
        };

        Ok(HirStmt { kind, span })
    }
```

- [ ] **Step 4: 添加程序降级方法**

```rust
    pub fn lower_program(&mut self, program: &ast::Program) -> TenthResult<HirProgram> {
        // 第一遍：注册所有函数
        for item in &program.items {
            if let ast::ItemKind::Function { name, params, return_type, .. } = &item.kind {
                let param_types: Vec<(String, Type)> = params.iter()
                    .map(|p| (p.name.name.clone(), Type::from_annotation(&p.type_ann)))
                    .collect();
                let ret_ty = return_type.as_ref()
                    .map(|rt| Type::from_annotation(rt))
                    .unwrap_or(Type::unit());
                self.scope.define_fn(name.name.clone(), param_types, ret_ty);
            }
        }

        // 第二遍：降级函数体
        for item in &program.items {
            if let ast::ItemKind::Function { name, params, return_type, body } = &item.kind {
                let param_types: Vec<(String, Type)> = params.iter()
                    .map(|p| (p.name.name.clone(), Type::from_annotation(&p.type_ann)))
                    .collect();
                let ret_ty = return_type.as_ref()
                    .map(|rt| Type::from_annotation(rt))
                    .unwrap_or(Type::unit());

                let mut body_scope = Scope::with_parent(std::mem::replace(&mut self.scope, Scope::new()));
                self.scope = body_scope;

                // 注册参数
                for (n, t) in &param_types {
                    self.scope.define_var(n.clone(), t.clone(), false);
                }

                let lowered_body = self.lower_expr(body)?;

                let outer_scope = std::mem::replace(&mut self.scope, Scope::new());
                self.scope = outer_scope;

                self.functions.push(HirFnDef {
                    name: name.name.clone(),
                    params: param_types,
                    return_type: ret_ty,
                    body: lowered_body,
                    span: item.span.clone(),
                });
            } else {
                // 顶层表达式
                let main_expr = self.lower_expr(&body_from_item(item)?)?;
                return Ok(HirProgram {
                    functions: self.functions.clone(),
                    main_expr: Some(main_expr),
                });
            }
        }

        Ok(HirProgram {
            functions: self.functions.clone(),
            main_expr: None,
        })
    }
}

fn lower_binop(op: &ast::BinOp) -> BinOp {
    use ast::BinOp;
    match op {
        BinOp::Add => BinOp::Add, BinOp::Sub => BinOp::Sub,
        BinOp::Mul => BinOp::Mul, BinOp::Div => BinOp::Div,
        BinOp::Mod => BinOp::Mod, BinOp::Eq => BinOp::Eq,
        BinOp::NotEq => BinOp::NotEq, BinOp::Lt => BinOp::Lt,
        BinOp::Gt => BinOp::Gt, BinOp::LtEq => BinOp::LtEq,
        BinOp::GtEq => BinOp::GtEq, BinOp::And => BinOp::And,
        BinOp::Or => BinOp::Or,
    }
}

fn body_from_item(item: &ast::Item) -> TenthResult<&ast::Expr> {
    match &item.kind {
        ast::ItemKind::Function { body, .. } => Ok(body),
        _ => Err(TenthError::ParseError {
            line: item.span.line,
            col: item.span.col,
            message: "expected expression item".into(),
        }),
    }
}
```

- [ ] **Step 5: 验证编译**

```bash
cd /workspace/tenth && cargo build
```

---

### Task 8: 运行时值系统与张量实现

**Files:**
- Create: `tenth/src/runtime/mod.rs`
- Create: `tenth/src/runtime/value.rs`
- Create: `tenth/src/runtime/tensor.rs`

- [ ] **Step 1: 写入 value.rs**

```rust
use std::rc::Rc;
use std::cell::RefCell;
use std::fmt;
use super::tensor::Tensor;
use crate::hir::types::{Type, BaseType, Dim};

#[derive(Clone)]
pub enum Value {
    Int(i64),
    Float(f64),
    Bool(bool),
    String(String),
    Tensor(Rc<RefCell<Tensor>>),
    Unit,
    Array(Vec<Value>),
    FnRef {
        name: String,
        params: Vec<(String, Type)>,
        return_type: Type,
    },
    Closure {
        params: Vec<(String, Type)>,
        body: Rc<crate::hir::hir::HirExpr>,
        captures: Vec<(String, Value)>,
    },
}

impl Value {
    pub fn type_of(&self) -> Type {
        match self {
            Value::Int(_) => Type::Base(BaseType::I32),
            Value::Float(_) => Type::Base(BaseType::F64),
            Value::Bool(_) => Type::Base(BaseType::Bool),
            Value::String(_) => Type::Base(BaseType::Str),
            Value::Tensor(t) => {
                let t = t.borrow();
                let dims: Vec<Dim> = t.shape().iter().map(|&d| Dim::Known(d as i64)).collect();
                Type::Tensor { dtype: BaseType::F64, dims }
            }
            Value::Unit => Type::Unit,
            Value::Array(_) => Type::Unknown,
            Value::FnRef { params, return_type, .. } => {
                Type::FnType {
                    params: params.iter().map(|(_, t)| t.clone()).collect(),
                    ret: Box::new(return_type.clone()),
                }
            }
            Value::Closure { params, .. } => {
                Type::FnType {
                    params: params.iter().map(|(_, t)| t.clone()).collect(),
                    ret: Box::new(Type::Unknown),
                }
            }
        }
    }

    pub fn as_float(&self) -> Option<f64> {
        match self {
            Value::Float(f) => Some(*f),
            Value::Int(i) => Some(*i as f64),
            _ => None,
        }
    }

    pub fn as_int(&self) -> Option<i64> {
        match self {
            Value::Int(i) => Some(*i),
            Value::Float(f) => Some(*f as i64),
            _ => None,
        }
    }

    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Value::Bool(b) => Some(*b),
            _ => None,
        }
    }

    pub fn is_truthy(&self) -> bool {
        match self {
            Value::Bool(b) => *b,
            Value::Int(n) => *n != 0,
            Value::Float(f) => *f != 0.0,
            Value::String(s) => !s.is_empty(),
            _ => true,
        }
    }
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Value::Int(n) => write!(f, "{}", n),
            Value::Float(n) => write!(f, "{}", n),
            Value::Bool(b) => write!(f, "{}", b),
            Value::String(s) => write!(f, "{}", s),
            Value::Tensor(t) => write!(f, "{}", t.borrow()),
            Value::Unit => write!(f, "()"),
            Value::Array(items) => {
                write!(f, "[")?;
                for (i, item) in items.iter().enumerate() {
                    if i > 0 { write!(f, ", ")?; }
                    write!(f, "{}", item)?;
                }
                write!(f, "]")
            }
            Value::FnRef { name, .. } => write!(f, "<fn {}>", name),
            Value::Closure { .. } => write!(f, "<closure>"),
        }
    }
}
```

- [ ] **Step 2: 写入 tensor.rs**

使用 ndarray 实现张量：

```rust
use ndarray::{ArrayD, IxDyn};
use std::fmt;

#[derive(Clone)]
pub struct Tensor {
    data: ArrayD<f64>,
}

impl Tensor {
    pub fn from_vec(data: Vec<f64>, shape: Vec<usize>) -> Self {
        let array = ArrayD::from_shape_vec(IxDyn(&shape), data)
            .expect("invalid tensor shape");
        Tensor { data: array }
    }

    pub fn zeros(shape: &[usize]) -> Self {
        let array = ArrayD::zeros(IxDyn(shape));
        Tensor { data: array }
    }

    pub fn ones(shape: &[usize]) -> Self {
        let array = ArrayD::ones(IxDyn(shape));
        Tensor { data: array }
    }

    pub fn rand(shape: &[usize]) -> Self {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let size: usize = shape.iter().product();
        let data: Vec<f64> = (0..size).map(|_| rng.gen::<f64>()).collect();
        Tensor::from_vec(data, shape.to_vec())
    }

    pub fn randn(shape: &[usize]) -> Self {
        use rand::Rng;
        use rand_distr::{Normal, Distribution};
        let mut rng = rand::thread_rng();
        let normal = Normal::new(0.0, 1.0).unwrap();
        let size: usize = shape.iter().product();
        let data: Vec<f64> = (0..size).map(|_| normal.sample(&mut rng)).collect();
        Tensor::from_vec(data, shape.to_vec())
    }

    pub fn shape(&self) -> Vec<usize> {
        self.data.shape().to_vec()
    }

    pub fn ndim(&self) -> usize {
        self.data.ndim()
    }

    pub fn get(&self, index: &[usize]) -> Option<f64> {
        self.data.get(IxDyn(index)).copied()
    }

    pub fn sum(&self) -> f64 {
        self.data.sum()
    }

    pub fn sum_axis(&self, axis: usize) -> Tensor {
        let summed = self.data.sum_axis(ndarray::Axis(axis));
        Tensor { data: summed }
    }

    pub fn mean(&self) -> f64 {
        self.data.mean().unwrap_or(0.0)
    }

    pub fn add_scalar(&self, scalar: f64) -> Tensor {
        Tensor { data: &self.data + scalar }
    }

    pub fn sub_scalar(&self, scalar: f64) -> Tensor {
        Tensor { data: &self.data - scalar }
    }

    pub fn mul_scalar(&self, scalar: f64) -> Tensor {
        Tensor { data: &self.data * scalar }
    }

    pub fn div_scalar(&self, scalar: f64) -> Tensor {
        Tensor { data: &self.data / scalar }
    }

    pub fn neg(&self) -> Tensor {
        Tensor { data: -&self.data }
    }

    pub fn abs(&self) -> Tensor {
        Tensor { data: self.data.mapv(|x| x.abs()) }
    }

    pub fn sqrt(&self) -> Tensor {
        Tensor { data: self.data.mapv(|x| x.sqrt()) }
    }

    pub fn exp(&self) -> Tensor {
        Tensor { data: self.data.mapv(|x| x.exp()) }
    }

    pub fn log(&self) -> Tensor {
        Tensor { data: self.data.mapv(|x| x.ln()) }
    }

    pub fn relu(&self) -> Tensor {
        Tensor { data: self.data.mapv(|x| if x > 0.0 { x } else { 0.0 }) }
    }

    pub fn sigmoid(&self) -> Tensor {
        Tensor { data: self.data.mapv(|x| 1.0 / (1.0 + (-x).exp())) }
    }

    pub fn tanh(&self) -> Tensor {
        Tensor { data: self.data.mapv(|x| x.tanh()) }
    }

    pub fn reshape(&self, shape: &[usize]) -> Option<Tensor> {
        let array = self.data.clone().into_shape(IxDyn(shape)).ok()?;
        Some(Tensor { data: array })
    }

    pub fn flatten(&self) -> Tensor {
        let size = self.data.len();
        let array = self.data.clone().into_shape(IxDyn(&[size])).unwrap();
        Tensor { data: array }
    }
}

impl fmt::Display for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.data)
    }
}
```

需要在 Cargo.toml 中添加 rand 依赖：

更新 `tenth/Cargo.toml` 的 `[dependencies]`：

```toml
[dependencies]
ndarray = "0.16"
rustyline = "15"
thiserror = "2"
rand = "0.8"
rand_distr = "0.4"
```

- [ ] **Step 3: 写入 runtime/mod.rs**

```rust
pub mod value;
pub mod tensor;
pub mod interpreter;
```

更新 `tenth/src/main.rs`：

```rust
pub mod error;
pub mod lexer;
pub mod parser;
pub mod hir;
pub mod runtime;
```

- [ ] **Step 4: 验证编译**

```bash
cd /workspace/tenth && cargo build
```

---

### Task 9: 解释器实现

**Files:**
- Create: `tenth/src/runtime/interpreter.rs`

- [ ] **Step 1: 写入解释器核心结构**

```rust
use std::collections::HashMap;
use std::rc::Rc;
use std::cell::RefCell;
use crate::error::{TenthError, TenthResult};
use crate::hir::hir::*;
use super::value::Value;
use super::tensor::Tensor;

pub struct Interpreter {
    variables: HashMap<String, Value>,
    functions: Vec<HirFnDef>,
}

impl Interpreter {
    pub fn new(functions: Vec<HirFnDef>) -> Self {
        Interpreter {
            variables: HashMap::new(),
            functions,
        }
    }

    pub fn execute_program(&mut self, program: &HirProgram) -> TenthResult<Option<Value>> {
        // 注册所有函数
        for func in &program.functions {
            let params = func.params.clone();
            let ret = func.return_type.clone();
            let body = func.body.clone();
            self.variables.insert(
                func.name.clone(),
                Value::FnRef {
                    name: func.name.clone(),
                    params: params.clone(),
                    return_type: ret.clone(),
                },
            );
        }

        // 执行 main
        if let Some(ref expr) = program.main_expr {
            self.eval_expr(expr)
        } else if let Some(main_fn) = self.functions.iter().find(|f| f.name == "main") {
            let body = main_fn.body.clone();
            self.eval_expr(&body)
        } else {
            Ok(None)
        }
    }

    fn eval_expr(&mut self, expr: &HirExpr) -> TenthResult<Option<Value>> {
        use HirExprKind;

        match &expr.kind {
            HirExprKind::Literal(lit) => {
                Ok(Some(match lit {
                    Literal::Int(n) => Value::Int(*n),
                    Literal::Float(n) => Value::Float(*n),
                    Literal::Bool(b) => Value::Bool(*b),
                    Literal::String(s) => Value::String(s.clone()),
                }))
            }

            HirExprKind::Var(name) => {
                self.variables.get(name)
                    .cloned()
                    .ok_or_else(|| TenthError::RuntimeError {
                        message: format!("undefined variable '{}'", name),
                    })
                    .map(Some)
            }

            HirExprKind::Binary { op, left, right, .. } => {
                let l = self.eval_expr(left)?.ok_or_else(|| TenthError::RuntimeError {
                    message: "left operand is void".into(),
                })?;
                let r = self.eval_expr(right)?.ok_or_else(|| TenthError::RuntimeError {
                    message: "right operand is void".into(),
                })?;
                self.eval_binary(op, &l, &r).map(Some)
            }

            HirExprKind::Unary { op, expr, .. } => {
                let val = self.eval_expr(expr)?.ok_or_else(|| TenthError::RuntimeError {
                    message: "unary operand is void".into(),
                })?;
                self.eval_unary(op, &val).map(Some)
            }

            HirExprKind::Call { func, args, .. } => {
                let f = self.eval_expr(func)?.ok_or_else(|| TenthError::RuntimeError {
                    message: "function value is void".into(),
                })?;

                let mut arg_values = Vec::new();
                for a in args {
                    arg_values.push(self.eval_expr(a)?.ok_or_else(|| TenthError::RuntimeError {
                        message: "argument is void".into(),
                    })?);
                }

                self.eval_call(&f, &arg_values, &expr.span)
            }

            HirExprKind::MethodCall { receiver, method, args, .. } => {
                let recv = self.eval_expr(receiver)?.ok_or_else(|| TenthError::RuntimeError {
                    message: "receiver is void".into(),
                })?;

                let mut arg_values = Vec::new();
                for a in args {
                    arg_values.push(self.eval_expr(a)?.ok_or_else(|| TenthError::RuntimeError {
                        message: "method argument is void".into(),
                    })?);
                }

                self.eval_method(&recv, method, &arg_values, &expr.span).map(Some)
            }

            HirExprKind::Index { target, indices } => {
                let t = self.eval_expr(target)?.ok_or_else(|| TenthError::RuntimeError {
                    message: "index target is void".into(),
                })?;
                self.eval_index(&t, indices).map(Some)
            }

            HirExprKind::TensorLiteral { data, .. } => {
                let mut rows: Vec<Vec<f64>> = Vec::new();
                for row in data {
                    let mut row_vals = Vec::new();
                    for elem in row {
                        let v = self.eval_expr(elem)?.ok_or_else(|| TenthError::RuntimeError {
                            message: "tensor element is void".into(),
                        })?;
                        row_vals.push(v.as_float().unwrap_or(0.0));
                    }
                    rows.push(row_vals);
                }
                let nrows = rows.len();
                let ncols = rows.first().map(|r| r.len()).unwrap_or(0);
                let flat: Vec<f64> = rows.into_iter().flatten().collect();
                let tensor = Tensor::from_vec(flat, vec![nrows, ncols]);
                Ok(Some(Value::Tensor(Rc::new(RefCell::new(tensor)))))
            }

            HirExprKind::If { cond, then_branch, else_branch, .. } => {
                let c = self.eval_expr(cond)?.ok_or_else(|| TenthError::RuntimeError {
                    message: "if condition is void".into(),
                })?;
                if c.is_truthy() {
                    self.eval_expr(then_branch)
                } else if let Some(ref eb) = else_branch {
                    self.eval_expr(eb)
                } else {
                    Ok(Some(Value::Unit))
                }
            }

            HirExprKind::Block { stmts, final_expr } => {
                for stmt in stmts {
                    self.eval_stmt(stmt)?;
                }
                match final_expr {
                    Some(e) => self.eval_expr(e),
                    None => Ok(Some(Value::Unit)),
                }
            }

            HirExprKind::Assign { target, value } => {
                let v = self.eval_expr(value)?.ok_or_else(|| TenthError::RuntimeError {
                    message: "assign value is void".into(),
                })?;
                self.variables.insert(target.clone(), v);
                Ok(Some(Value::Unit))
            }

            HirExprKind::AssignOp { target, op, value } => {
                let current = self.variables.get(target).cloned().ok_or_else(|| {
                    TenthError::RuntimeError {
                        message: format!("undefined variable '{}'", target),
                    }
                })?;
                let rhs = self.eval_expr(value)?.ok_or_else(|| TenthError::RuntimeError {
                    message: "assign-op value is void".into(),
                })?;
                let result = self.eval_binary(op, &current, &rhs)?;
                self.variables.insert(target.clone(), result);
                Ok(Some(Value::Unit))
            }

            HirExprKind::Closure { params, body } => {
                Ok(Some(Value::Closure {
                    params: params.clone(),
                    body: Rc::new((**body).clone()),
                    captures: Vec::new(),
                }))
            }

            _ => {
                Err(TenthError::RuntimeError {
                    message: format!("unimplemented expression: {:?}", expr.kind),
                })
            }
        }
    }
```

- [ ] **Step 2: 添加二元/一元运算、方法调用、内置函数**

```rust
    fn eval_binary(&self, op: &BinOp, l: &Value, r: &Value) -> TenthResult<Value> {
        match op {
            BinOp::Add => match (l, r) {
                (Value::Int(a), Value::Int(b)) => Ok(Value::Int(a + b)),
                (Value::Float(a), Value::Float(b)) => Ok(Value::Float(a + b)),
                (Value::Int(a), Value::Float(b)) => Ok(Value::Float(*a as f64 + b)),
                (Value::Float(a), Value::Int(b)) => Ok(Value::Float(a + *b as f64)),
                (Value::Tensor(t), Value::Float(s)) => {
                    let result = t.borrow().add_scalar(*s);
                    Ok(Value::Tensor(Rc::new(RefCell::new(result))))
                }
                (Value::Float(s), Value::Tensor(t)) => {
                    let result = t.borrow().add_scalar(*s);
                    Ok(Value::Tensor(Rc::new(RefCell::new(result))))
                }
                _ => Err(TenthError::RuntimeError {
                    message: format!("cannot add {:?} and {:?}", l.type_of(), r.type_of()),
                }),
            },
            BinOp::Sub => match (l, r) {
                (Value::Int(a), Value::Int(b)) => Ok(Value::Int(a - b)),
                (Value::Float(a), Value::Float(b)) => Ok(Value::Float(a - b)),
                (Value::Tensor(t), Value::Float(s)) => {
                    let result = t.borrow().sub_scalar(*s);
                    Ok(Value::Tensor(Rc::new(RefCell::new(result))))
                }
                _ => Err(TenthError::RuntimeError {
                    message: format!("cannot subtract {:?} and {:?}", l.type_of(), r.type_of()),
                }),
            },
            BinOp::Mul => match (l, r) {
                (Value::Int(a), Value::Int(b)) => Ok(Value::Int(a * b)),
                (Value::Float(a), Value::Float(b)) => Ok(Value::Float(a * b)),
                (Value::Tensor(t), Value::Float(s)) => {
                    let result = t.borrow().mul_scalar(*s);
                    Ok(Value::Tensor(Rc::new(RefCell::new(result))))
                }
                _ => Err(TenthError::RuntimeError {
                    message: format!("cannot multiply {:?} and {:?}", l.type_of(), r.type_of()),
                }),
            },
            BinOp::Div => match (l, r) {
                (Value::Int(a), Value::Int(b)) => Ok(Value::Float(*a as f64 / *b as f64)),
                (Value::Float(a), Value::Float(b)) => Ok(Value::Float(a / b)),
                (Value::Tensor(t), Value::Float(s)) => {
                    let result = t.borrow().div_scalar(*s);
                    Ok(Value::Tensor(Rc::new(RefCell::new(result))))
                }
                _ => Err(TenthError::RuntimeError {
                    message: format!("cannot divide {:?} and {:?}", l.type_of(), r.type_of()),
                }),
            },
            BinOp::Eq => Ok(Value::Bool(self.values_eq(l, r))),
            BinOp::NotEq => Ok(Value::Bool(!self.values_eq(l, r))),
            BinOp::Lt => match (l, r) {
                (Value::Int(a), Value::Int(b)) => Ok(Value::Bool(a < b)),
                (Value::Float(a), Value::Float(b)) => Ok(Value::Bool(a < b)),
                _ => Ok(Value::Bool(false)),
            },
            BinOp::Gt => match (l, r) {
                (Value::Int(a), Value::Int(b)) => Ok(Value::Bool(a > b)),
                (Value::Float(a), Value::Float(b)) => Ok(Value::Bool(a > b)),
                _ => Ok(Value::Bool(false)),
            },
            BinOp::LtEq => match (l, r) {
                (Value::Int(a), Value::Int(b)) => Ok(Value::Bool(a <= b)),
                (Value::Float(a), Value::Float(b)) => Ok(Value::Bool(a <= b)),
                _ => Ok(Value::Bool(false)),
            },
            BinOp::GtEq => match (l, r) {
                (Value::Int(a), Value::Int(b)) => Ok(Value::Bool(a >= b)),
                (Value::Float(a), Value::Float(b)) => Ok(Value::Bool(a >= b)),
                _ => Ok(Value::Bool(false)),
            },
            BinOp::And => Ok(Value::Bool(l.is_truthy() && r.is_truthy())),
            BinOp::Or => Ok(Value::Bool(l.is_truthy() || r.is_truthy())),
            _ => Err(TenthError::RuntimeError {
                message: format!("unsupported binary operator: {:?}", op),
            }),
        }
    }

    fn values_eq(&self, l: &Value, r: &Value) -> bool {
        match (l, r) {
            (Value::Int(a), Value::Int(b)) => a == b,
            (Value::Float(a), Value::Float(b)) => (a - b).abs() < 1e-10,
            (Value::Bool(a), Value::Bool(b)) => a == b,
            (Value::String(a), Value::String(b)) => a == b,
            (Value::Unit, Value::Unit) => true,
            _ => false,
        }
    }

    fn eval_unary(&self, op: &UnaryOp, val: &Value) -> TenthResult<Value> {
        match op {
            UnaryOp::Neg => match val {
                Value::Int(n) => Ok(Value::Int(-n)),
                Value::Float(n) => Ok(Value::Float(-n)),
                Value::Tensor(t) => {
                    let result = t.borrow().neg();
                    Ok(Value::Tensor(Rc::new(RefCell::new(result))))
                }
                _ => Err(TenthError::RuntimeError {
                    message: "cannot negate this value".into(),
                }),
            },
            UnaryOp::Not => Ok(Value::Bool(!val.is_truthy())),
        }
    }

    fn eval_method(
        &self, recv: &Value, method: &str, args: &[Value], span: &crate::lexer::token::Span,
    ) -> TenthResult<Value> {
        match recv {
            Value::Tensor(t) => {
                let tensor = t.borrow();
                match method {
                    "sum" => {
                        if args.is_empty() {
                            Ok(Value::Float(tensor.sum()))
                        } else {
                            let axis = args[0].as_int().unwrap_or(0) as usize;
                            let result = tensor.sum_axis(axis);
                            Ok(Value::Tensor(Rc::new(RefCell::new(result))))
                        }
                    }
                    "mean" => Ok(Value::Float(tensor.mean())),
                    "abs" => {
                        let result = tensor.abs();
                        Ok(Value::Tensor(Rc::new(RefCell::new(result))))
                    }
                    "sqrt" => {
                        let result = tensor.sqrt();
                        Ok(Value::Tensor(Rc::new(RefCell::new(result))))
                    }
                    "exp" => {
                        let result = tensor.exp();
                        Ok(Value::Tensor(Rc::new(RefCell::new(result))))
                    }
                    "log" => {
                        let result = tensor.log();
                        Ok(Value::Tensor(Rc::new(RefCell::new(result))))
                    }
                    "relu" => {
                        let result = tensor.relu();
                        Ok(Value::Tensor(Rc::new(RefCell::new(result))))
                    }
                    "sigmoid" => {
                        let result = tensor.sigmoid();
                        Ok(Value::Tensor(Rc::new(RefCell::new(result))))
                    }
                    "tanh" => {
                        let result = tensor.tanh();
                        Ok(Value::Tensor(Rc::new(RefCell::new(result))))
                    }
                    "reshape" | "view" => {
                        let shape: Vec<usize> = args.iter()
                            .map(|a| a.as_int().unwrap_or(1) as usize)
                            .collect();
                        let result = tensor.reshape(&shape).ok_or_else(|| {
                            TenthError::RuntimeError {
                                message: format!("cannot reshape to {:?}", shape),
                            }
                        })?;
                        Ok(Value::Tensor(Rc::new(RefCell::new(result))))
                    }
                    "flatten" => {
                        let result = tensor.flatten();
                        Ok(Value::Tensor(Rc::new(RefCell::new(result))))
                    }
                    _ => Err(TenthError::RuntimeError {
                        message: format!("unknown tensor method: {}", method),
                    }),
                }
            }
            _ => Err(TenthError::RuntimeError {
                message: format!("method '{}' not supported on this type", method),
            }),
        }
    }

    fn eval_index(&self, target: &Value, indices: &[Index]) -> TenthResult<Value> {
        match target {
            Value::Tensor(t) => {
                let tensor = t.borrow();
                let shape = tensor.shape();
                let mut idx: Vec<usize> = Vec::new();
                for (i, index_expr) in indices.iter().enumerate() {
                    match index_expr {
                        Index::Single(e) => {
                            let v = self.eval_expr(e)?.ok_or_else(|| TenthError::RuntimeError {
                                message: "index is void".into(),
                            })?;
                            idx.push(v.as_int().unwrap_or(0) as usize);
                        }
                        _ => {
                            // Range/Colon not fully implemented
                            if i < shape.len() {
                                idx.push(0);
                            }
                        }
                    }
                }
                match tensor.get(&idx) {
                    Some(val) => Ok(Value::Float(val)),
                    None => Err(TenthError::RuntimeError {
                        message: format!("index {:?} out of bounds for shape {:?}", idx, shape),
                    }),
                }
            }
            _ => Err(TenthError::RuntimeError {
                message: "indexing only supported on tensors".into(),
            }),
        }
    }

    fn eval_call(
        &mut self, func: &Value, args: &[Value], span: &crate::lexer::token::Span,
    ) -> TenthResult<Option<Value>> {
        match func {
            Value::FnRef { name, .. } => {
                self.call_named_fn(name, args, span)
            }
            Value::Closure { params, body, captures } => {
                // 保存旧变量
                let saved: HashMap<String, Value> = params.iter()
                    .filter_map(|(n, _)| self.variables.get(n).cloned().map(|v| (n.clone(), v)))
                    .collect();

                for ((pname, _), arg) in params.iter().zip(args.iter()) {
                    self.variables.insert(pname.clone(), arg.clone());
                }

                for (name, val) in captures {
                    self.variables.insert(name.clone(), val.clone());
                }

                let result = self.eval_expr(body);

                // 恢复
                for (n, v) in saved {
                    self.variables.insert(n, v);
                }

                result
            }
            _ => Err(TenthError::RuntimeError {
                message: "not a callable value".into(),
            }),
        }
    }

    fn call_named_fn(
        &mut self, name: &str, args: &[Value], span: &crate::lexer::token::Span,
    ) -> TenthResult<Option<Value>> {
        // 内置函数
        match name {
            "println" => {
                for arg in args {
                    print!("{}", arg);
                }
                println!();
                return Ok(Some(Value::Unit));
            }
            "eprintln" => {
                for arg in args {
                    eprint!("{}", arg);
                }
                eprintln!();
                return Ok(Some(Value::Unit));
            }
            _ => {}
        }

        // 查找用户定义函数
        let func_def = self.functions.iter().find(|f| f.name == name).cloned();
        if let Some(fd) = func_def {
            let saved: HashMap<String, Value> = fd.params.iter()
                .filter_map(|(n, _)| self.variables.get(n).cloned().map(|v| (n.clone(), v)))
                .collect();

            for ((pname, _), arg) in fd.params.iter().zip(args.iter()) {
                self.variables.insert(pname.clone(), arg.clone());
            }

            let result = self.eval_expr(&fd.body);

            for (n, v) in saved {
                self.variables.insert(n, v);
            }

            return result;
        }

        Err(TenthError::RuntimeError {
            message: format!("undefined function '{}'", name),
        })
    }

    fn eval_stmt(&mut self, stmt: &HirStmt) -> TenthResult<()> {
        match &stmt.kind {
            HirStmtKind::Expr(e) => {
                self.eval_expr(e)?;
                Ok(())
            }
            HirStmtKind::Let { name, init, .. } => {
                let val = match init {
                    Some(e) => self.eval_expr(e)?.unwrap_or(Value::Unit),
                    None => Value::Unit,
                };
                self.variables.insert(name.clone(), val);
                Ok(())
            }
            HirStmtKind::Return(_) => {
                // Return is handled at function call level
                Ok(())
            }
            HirStmtKind::While { cond, body } => {
                loop {
                    let c = self.eval_expr(cond)?.ok_or_else(|| TenthError::RuntimeError {
                        message: "while condition is void".into(),
                    })?;
                    if !c.is_truthy() {
                        break;
                    }
                    self.eval_stmt(body)?;
                }
                Ok(())
            }
            HirStmtKind::For { var, iter, body } => {
                let iter_val = self.eval_expr(iter)?.ok_or_else(|| TenthError::RuntimeError {
                    message: "for iterable is void".into(),
                })?;
                match iter_val {
                    Value::Tensor(t) => {
                        let tensor = t.borrow();
                        let shape = tensor.shape();
                        let n = shape.first().copied().unwrap_or(0);
                        let item_shape: Vec<usize> = if shape.len() <= 1 {
                            vec![1]
                        } else {
                            shape[1..].to_vec()
                        };
                        let total_items: usize = item_shape.iter().product();
                        let idx_size = if shape.is_empty() { 0 } else { shape[0] };

                        for i in 0..idx_size {
                            let flat: Vec<f64> = (0..total_items).map(|j| {
                                tensor.get(&[i, j]).unwrap_or(0.0)
                            }).collect();
                            let item = Tensor::from_vec(flat, item_shape.clone());
                            self.variables.insert(
                                var.clone(),
                                Value::Tensor(Rc::new(RefCell::new(item))),
                            );
                            self.eval_stmt(body)?;
                        }
                    }
                    _ => {
                        // 对于范围表达式 (Range → 我们暂时不支持解析range为value)
                        return Err(TenthError::RuntimeError {
                            message: "for loop only supports tensor iteration for now".into(),
                        });
                    }
                }
                Ok(())
            }
        }
    }
}
```

- [ ] **Step 5: 验证编译**

```bash
cd /workspace/tenth && cargo build
```

---

### Task 10: REPL 与主入口

**Files:**
- Create: `tenth/src/repl.rs`
- Modify: `tenth/src/main.rs`

- [ ] **Step 1: 写入 repl.rs**

```rust
use rustyline::error::ReadlineError;
use rustyline::DefaultEditor;
use crate::error::TenthResult;
use crate::lexer::lexer::Lexer;
use crate::parser::parser::Parser;
use crate::hir::lower::Lowerer;
use crate::runtime::interpreter::Interpreter;
use crate::runtime::value::Value;

pub fn run_repl() -> TenthResult<()> {
    let mut rl = DefaultEditor::new().unwrap();
    println!("Tenth v0.1.0 REPL");
    println!("Type expressions, ':q' to quit, ':h' for help");
    println!();

    let mut functions = Vec::new();
    let mut variables: std::collections::HashMap<String, Value> = std::collections::HashMap::new();

    let mut line_num = 0;

    loop {
        line_num += 1;
        let prompt = format!("tenth> ");
        let readline = rl.readline(&prompt);

        match readline {
            Ok(line) => {
                let trimmed = line.trim();

                if trimmed.is_empty() {
                    continue;
                }

                // 特殊命令
                if trimmed == ":q" {
                    println!("Goodbye!");
                    break;
                }
                if trimmed == ":h" {
                    println!("Tenth REPL commands:");
                    println!("  :q         quit");
                    println!("  :h         help");
                    println!("  :vars      show variables");
                    println!();
                    println!("Examples:");
                    println!("  let x = 42");
                    println!("  x + 10");
                    println!("  tensor.rand([3, 224, 224]).sum()");
                    continue;
                }
                if trimmed == ":vars" {
                    for (name, val) in &variables {
                        println!("  {} = {}", name, val);
                    }
                    continue;
                }

                rl.add_history_entry(trimmed).ok();

                // 尝试解析为函数定义或表达式
                match execute_line(trimmed, &mut functions, &mut variables) {
                    Ok(Some(val)) => {
                        match val {
                            Value::Unit => {}
                            _ => println!("= {}", val),
                        }
                    }
                    Ok(None) => {}
                    Err(e) => {
                        eprintln!("Error: {}", e);
                    }
                }
            }
            Err(ReadlineError::Interrupted) => {
                println!("^C");
                continue;
            }
            Err(ReadlineError::Eof) => {
                println!("Goodbye!");
                break;
            }
            Err(err) => {
                eprintln!("REPL error: {:?}", err);
                break;
            }
        }
    }

    Ok(())
}

fn execute_line(
    line: &str,
    functions: &mut Vec<crate::hir::hir::HirFnDef>,
    variables: &mut std::collections::HashMap<String, Value>,
) -> TenthResult<Option<Value>> {
    // 词法分析
    let mut lexer = Lexer::new(line);
    let tokens = lexer.tokenize()?;

    // 语法解析
    let mut parser = Parser::new(tokens);
    let program = parser.parse_program()?;

    // 降级到 HIR
    let mut lowerer = Lowerer::new();
    let hir_program = lowerer.lower_program(&program)?;

    // 合并函数定义
    functions.extend(hir_program.functions.clone());

    // 执行
    let mut interpreter = Interpreter::new(functions.clone());
    interpreter.variables.extend(variables.clone());
    let result = interpreter.execute_program(&hir_program)?;

    // 保存变量（简化：把解释器变量拷贝回来）
    *variables = interpreter.variables.clone();

    Ok(result)
}
```

- [ ] **Step 2: 更新 main.rs**

```rust
pub mod error;
pub mod lexer;
pub mod parser;
pub mod hir;
pub mod runtime;
pub mod repl;

use error::TenthResult;

fn main() -> TenthResult<()> {
    repl::run_repl()
}
```

- [ ] **Step 3: 验证编译并运行测试**

```bash
cd /workspace/tenth && cargo build
```

---

### Task 11: 集成测试

**Files:**
- Create: `tenth/tests/integration_test.rs`

- [ ] **Step 1: 写入集成测试**

```rust
use tenth::lexer::lexer::Lexer;
use tenth::parser::parser::Parser;
use tenth::hir::lower::Lowerer;
use tenth::runtime::interpreter::Interpreter;
use tenth::runtime::value::Value;

fn run_code(src: &str) -> Result<Option<Value>, String> {
    let mut lexer = Lexer::new(src);
    let tokens = lexer.tokenize().map_err(|e| e.to_string())?;
    let mut parser = Parser::new(tokens);
    let program = parser.parse_program().map_err(|e| e.to_string())?;
    let mut lowerer = Lowerer::new();
    let hir = lowerer.lower_program(&program).map_err(|e| e.to_string())?;
    let mut interpreter = Interpreter::new(hir.functions);
    interpreter.execute_program(&hir).map_err(|e| e.to_string())
}

#[test]
fn test_simple_arithmetic() {
    let result = run_code("1 + 2");
    assert!(result.is_ok());
    match result.unwrap() {
        Some(Value::Int(3)) => {}
        v => panic!("expected Some(Int(3)), got {:?}", v),
    }
}

#[test]
fn test_variable_and_use() {
    let result = run_code("let x = 42; x + 10");
    assert!(result.is_ok());
    match result.unwrap() {
        Some(Value::Int(52)) => {}
        v => panic!("expected Some(Int(52)), got {:?}", v),
    }
}

#[test]
fn test_float_arithmetic() {
    let result = run_code("3.14 * 2.0");
    assert!(result.is_ok());
    match result.unwrap() {
        Some(Value::Float(n)) => assert!((n - 6.28).abs() < 0.01),
        v => panic!("expected Some(Float(6.28)), got {:?}", v),
    }
}

#[test]
fn test_boolean_ops() {
    let result = run_code("true && false");
    match result.unwrap() {
        Some(Value::Bool(false)) => {}
        v => panic!("expected Some(Bool(false)), got {:?}", v),
    }
}

#[test]
fn test_comparison() {
    let result = run_code("5 > 3");
    match result.unwrap() {
        Some(Value::Bool(true)) => {}
        v => panic!("expected Some(Bool(true)), got {:?}", v),
    }
}

#[test]
fn test_if_expression() {
    let src = "if true { 1 } else { 2 }";
    let result = run_code(src);
    match result.unwrap() {
        Some(Value::Int(1)) => {}
        v => panic!("expected Some(Int(1)), got {:?}", v),
    }
}

#[test]
fn test_tensor_creation() {
    let result = run_code("tensor.rand([3, 224, 224]).sum()");
    assert!(result.is_ok());
}

#[test]
fn test_tensor_methods() {
    let src = "let x = tensor[[1.0, 2.0], [3.0, 4.0]]; x.sum()";
    let result = run_code(src);
    match result.unwrap() {
        Some(Value::Float(n)) => assert!((n - 10.0).abs() < 0.01),
        v => panic!("expected Some(Float(10.0)), got {:?}", v),
    }
}

#[test]
fn test_function_definition_and_call() {
    let src = "fn add(a: f64, b: f64) -> f64 { a + b }";
    // 函数定义在 REPL 模式下需要特殊处理，这里只验证不报错
    let result = run_code(src);
    assert!(result.is_ok());
}
```

- [ ] **Step 2: 运行测试**

```bash
cd /workspace/tenth && cargo test
```

期望：所有 10 个测试通过。

---

### Task 12: 验收 —— 手动运行 REPL

- [ ] **Step 1: 验证张量核心用例**

启动 REPL：
```bash
cd /workspace/tenth && cargo run --release
```

在 REPL 中依次输入：

```
let x = tensor.rand([3, 224, 224])
x.sum()
x.mean()
x + 1.0
```

期望：每行都有合理的输出（`tensor.rand` 输出随机张量，`sum()` 输出一个标量浮点数，`mean()` 类似，`x + 1.0` 输出加了 1 的张量）。

- [ ] **Step 2: 验证函数定义与调用**

```
fn double(x: f64) -> f64 { x * 2.0 }
double(21.0)
```

期望：输出 `42.0`。

- [ ] **Step 3: 验证控制流**

```
if 3 > 2 { "yes" } else { "no" }
```

期望：输出 `yes`。

- [ ] **Step 4: 验证错误处理**

```
let a = tensor.rand([10, 10])
let b = tensor.rand([10, 5])
```

期望：两个张量创建都成功。

---

## 验收标准总结

Phase 1 完成的标志：
- [ ] `cargo build` 编译成功
- [ ] `cargo test` 全部通过
- [ ] REPL 可交互运行
- [ ] 张量创建、运算、规约均可用
- [ ] 变量定义与使用可用
- [ ] 函数定义与调用可用
- [ ] if/else 控制流可用
- [ ] 能够执行 `tensor.rand([3, 224, 224]).sum()` 并输出合理结果