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