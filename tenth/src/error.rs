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
    /// Shows the error line with a caret pointing to the column position.
    pub fn display_with_source(&self, source: Option<&str>) -> String {
        let base = self.to_string();
        let (line_num, col_num) = self.location();
        if let (Some(src), Some(ln)) = (source, line_num) {
            let source_line = src.lines().nth(ln.saturating_sub(1)).unwrap_or("");
            let col = col_num.unwrap_or(1);
            let caret = build_caret(col, source_line);
            format!("Error: {}\n  |\n{:>3} | {}\n  | {}{}", base, ln, source_line, caret, self.suggestion())
        } else {
            let suggestion = self.suggestion();
            if suggestion.is_empty() {
                format!("Error: {}", base)
            } else {
                format!("Error: {}{}", base, suggestion)
            }
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

    /// Provide a suggestion for common errors.
    fn suggestion(&self) -> String {
        match self {
            TenthError::ParseError { message, .. } => {
                let msg = message.to_lowercase();
                if msg.contains("expected") && msg.contains("=>") {
                    "\n  help: match arms use `pattern => body` syntax".to_string()
                } else if msg.contains("expected") && msg.contains("}") {
                    "\n  help: check for missing closing brace `}`".to_string()
                } else if msg.contains("expected") && msg.contains(")") {
                    "\n  help: check for missing closing parenthesis `)`".to_string()
                } else if msg.contains("unexpected token") && msg.contains("=") {
                    "\n  help: did you mean `==` for equality comparison?".to_string()
                } else if msg.contains("undefined variable") {
                    extract_var_hint(message)
                } else {
                    String::new()
                }
            }
            TenthError::TypeError { message, .. } => {
                if message.contains("undefined variable") {
                    extract_var_hint(message)
                } else if message.contains("mismatched type") || message.contains("type mismatch") {
                    "\n  help: check that the types on both sides of the expression match".to_string()
                } else if message.contains("cannot borrow") {
                    "\n  help: Tenth uses ownership and borrowing rules similar to Rust".to_string()
                } else if message.contains("missing implementation") {
                    "\n  help: all required trait methods must be implemented (default methods are optional)".to_string()
                } else {
                    String::new()
                }
            }
            TenthError::RuntimeError { message } => {
                if message.contains("undefined variable") {
                    extract_var_hint(message)
                } else if message.contains("use of moved value") {
                    "\n  help: the value has been moved and can no longer be used; consider cloning it first".to_string()
                } else if message.contains("index out of bounds") {
                    "\n  help: the index exceeds the length of the collection".to_string()
                } else {
                    String::new()
                }
            }
            _ => String::new(),
        }
    }
}

/// Build a caret string pointing to the error column.
/// Accounts for tab characters by expanding them to visual width.
fn build_caret(col: usize, source_line: &str) -> String {
    let mut visual_pos = 0;
    for ch in source_line.chars() {
        if visual_pos + 1 >= col {
            break;
        }
        visual_pos += if ch == '\t' { 4 } else { 1 };
    }
    let padding = visual_pos;
    let mut caret = " ".repeat(padding);
    caret.push('^');
    caret
}

/// Try to extract a "did you mean" hint for undefined variable errors.
fn extract_var_hint(message: &str) -> String {
    // Try to extract the variable name from "undefined variable 'x'"
    if let Some(start) = message.find('\'') {
        if let Some(end) = message[start + 1..].find('\'') {
            let var = &message[start + 1..start + 1 + end];
            // Common typo suggestions
            let suggestions = match var {
                "pritnln" | "printn" => Some("println"),
                "print" => Some("println"),
                "strng" => Some("string"),
                "flase" => Some("false"),
                "ture" | "fales" => Some("true"),
                "fn" => None, // keyword, not a variable
                _ => None,
            };
            if let Some(s) = suggestions {
                return format!("\n  help: did you mean `{}`?", s);
            }
        }
    }
    String::new()
}

/// Format multiple errors collected during parsing with recovery.
pub fn format_multiple_errors(errors: &[TenthError], source: Option<&str>) -> String {
    if errors.is_empty() {
        return String::new();
    }
    if errors.len() == 1 {
        return errors[0].display_with_source(source);
    }
    let mut output = format!("found {} errors:\n\n", errors.len());
    for (i, err) in errors.iter().enumerate() {
        output.push_str(&format!("Error {} of {}:\n", i + 1, errors.len()));
        output.push_str(&err.display_with_source(source));
        output.push_str("\n\n");
    }
    output
}
