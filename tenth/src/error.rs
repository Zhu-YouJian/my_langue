use thiserror::Error;
use crate::runtime::value::Value;

#[derive(Error, Debug, Clone)]
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

    /// Non-error signal: a return statement was executed with this value.
    /// Propagated up through blocks/statements to the enclosing function call.
    #[error("return")]
    ReturnValue(Value),

    /// Non-error signal: a break statement was executed.
    /// Propagated up through blocks/statements to the enclosing loop.
    #[error("break")]
    BreakSignal,

    /// Non-error signal: a continue statement was executed.
    /// Propagated up through blocks/statements to the enclosing loop.
    #[error("continue")]
    ContinueSignal,

    /// Non-error signal: a `?` operator encountered Result::Err and propagates it.
    /// Caught by `try { }` blocks and function boundaries.
    #[error("try propagate")]
    TryPropagate(Value),
}

pub type TenthResult<T> = Result<T, TenthError>;

impl TenthError {
    /// Pretty-print the error with source context (if source is provided).
    pub fn display_with_source(&self, source: Option<&str>) -> String {
        let base = self.to_string();
        let (line_num, _col) = self.location();
        if let (Some(src), Some(ln)) = (source, line_num) {
            let source_line = src.lines().nth(ln.saturating_sub(1)).unwrap_or("");
            format!("Error: {}\n  |\n{:>3} | {}\n  |", base, ln, source_line)
        } else {
            format!("Error: {}", base)
        }
    }

    /// Return (line, col) if the error carries position info.
    pub fn location(&self) -> (Option<usize>, Option<usize>) {
        match self {
            TenthError::LexerError { line, col, .. }
            | TenthError::ParseError { line, col, .. }
            | TenthError::TypeError { line, col, .. } => (Some(*line), Some(*col)),
            _ => (None, None),
        }
    }
}