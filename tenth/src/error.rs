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
}

pub type TenthResult<T> = Result<T, TenthError>;