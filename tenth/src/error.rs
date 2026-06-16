use thiserror::Error;
use crate::runtime::value::Value;

#[derive(Error, Debug, Clone)]
pub enum TenthError {
    #[error("第 {line} 行第 {col} 列：词法错误 — {message}")]
    LexerError {
        line: usize,
        col: usize,
        message: String,
    },

    #[error("第 {line} 行第 {col} 列：语法错误 — {message}")]
    ParseError {
        line: usize,
        col: usize,
        message: String,
    },

    #[error("第 {line} 行第 {col} 列：类型错误 — {message}")]
    TypeError {
        line: usize,
        col: usize,
        message: String,
    },

    #[error("运行时错误 — {message}")]
    RuntimeError { message: String },

    #[error("输入意外结束")]
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
            format!("错误：{}\n  |\n{:>3} | {}\n  | {}{}", base, ln, source_line, caret, self.suggestion())
        } else {
            let suggestion = self.suggestion();
            if suggestion.is_empty() {
                format!("错误：{}", base)
            } else {
                format!("错误：{}{}", base, suggestion)
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
                if msg.contains("期望") && msg.contains("=>") {
                    "\n  提示：match 分支使用 `模式 => 代码体` 语法".to_string()
                } else if msg.contains("期望") && msg.contains("}") {
                    "\n  提示：检查是否缺少右花括号 `}`".to_string()
                } else if msg.contains("期望") && msg.contains(")") {
                    "\n  提示：检查是否缺少右圆括号 `)`".to_string()
                } else if msg.contains("意外") && msg.contains("=") {
                    "\n  提示：你是否想用 `==` 进行相等比较？".to_string()
                } else if msg.contains("未定义") {
                    extract_var_hint(message)
                } else {
                    String::new()
                }
            }
            TenthError::TypeError { message, .. } => {
                if message.contains("未定义") {
                    extract_var_hint(message)
                } else if message.contains("已移动") || message.contains("移动后") {
                    "\n  提示：该值已被移动，无法再使用；考虑先克隆一份".to_string()
                } else if message.contains("不可同时") || message.contains("可变借用") || message.contains("共享借用") {
                    "\n  提示：Tenth 采用类似 Rust 的所有权和借用规则".to_string()
                } else if message.contains("缺少实现") {
                    "\n  提示：trait 中所有非默认方法都必须实现".to_string()
                } else if message.contains("参数") && message.contains("不匹配") {
                    "\n  提示：检查函数调用时传入的参数数量和类型是否正确".to_string()
                } else {
                    String::new()
                }
            }
            TenthError::RuntimeError { message } => {
                if message.contains("未定义") {
                    extract_var_hint(message)
                } else if message.contains("已移动") || message.contains("移动后") {
                    "\n  提示：该值已被移动，无法再使用；考虑先克隆一份".to_string()
                } else if message.contains("越界") || message.contains("索引") {
                    "\n  提示：索引超出了集合的长度范围".to_string()
                } else if message.contains("无法读取") || message.contains("找不到") {
                    "\n  提示：检查文件路径是否正确".to_string()
                } else {
                    String::new()
                }
            }
            TenthError::LexerError { message, .. } => {
                if message.contains("未终止") || message.contains("未闭合") {
                    "\n  提示：检查字符串是否缺少右引号".to_string()
                } else if message.contains("意外字符") {
                    "\n  提示：该字符在此位置不合法，检查是否拼写错误".to_string()
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
    // Try to extract the variable name from "未定义变量 'x'" or "undefined variable 'x'"
    if let Some(start) = message.find('\'') {
        if let Some(end) = message[start + 1..].find('\'') {
            let var = &message[start + 1..start + 1 + end];
            let suggestions = match var {
                "pritnln" | "printn" => Some("println"),
                "print" => Some("println"),
                "strng" => Some("string"),
                "flase" => Some("false"),
                "ture" | "fales" => Some("true"),
                _ => None,
            };
            if let Some(s) = suggestions {
                return format!("\n  提示：你是否想用 `{}`？", s);
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
    let mut output = format!("发现 {} 个错误：\n\n", errors.len());
    for (i, err) in errors.iter().enumerate() {
        output.push_str(&format!("第 {} 个错误（共 {} 个）：\n", i + 1, errors.len()));
        output.push_str(&err.display_with_source(source));
        output.push_str("\n\n");
    }
    output
}
