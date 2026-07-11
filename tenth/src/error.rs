use thiserror::Error;
use crate::runtime::value::Value;

/// 护城河 F：关系调试器的 5 类错误分类（T2 §4.3 FormalExplain 扩展）。
///
/// 用于 `TenthError::RelationError` 与 `RootCause::error_type`，对 backward
/// 失败的原因进行细粒度归类，供关系调试器输出结构化诊断。
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ErrorType {
    /// shape 不匹配（MatMul 的 K≠K'、Conv2D 通道数不符等）。
    ShapeMismatch,
    /// 试图静默 squeeze（护城河 A 已在编译期拦截，F 附加关系上下文）。
    SilentSqueeze,
    /// 广播失败（Add/Sub/Mul/Div 两操作数 shape 不可广播）。
    BroadcastFail,
    /// 梯度 shape 漂移（前向 shape 流 vs 反向 grad shape 流不一致）。
    GradDrift,
    /// dtype 冲突（f32 vs f64 / F16 vs BF16 混合运算未提升）。
    DtypeConflict,
}

/// 形状错误上下文（护城河 F：T2 FormalExplain 动态层）。
/// 由 autodiff backward 在抛出 ShapeMismatch 时填充，
/// 携带报错节点的 id、算子名与期望/实际 shape。
#[derive(Debug, Clone)]
pub struct TapeErrorContext {
    /// 报错节点 v_err 的 tape node id。
    pub tape_node_id: usize,
    /// 算子名（如 "MatMul" / "Add" / "Conv2D"）。
    pub op: String,
    /// 期望的 shape（若未知则为空）。
    pub expected_shape: Vec<usize>,
    /// 实际的 shape（若未知则为空）。
    pub actual_shape: Vec<usize>,
}

impl std::fmt::Display for TapeErrorContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "节点 #{} {}", self.tape_node_id, self.op)?;
        if !self.expected_shape.is_empty() || !self.actual_shape.is_empty() {
            write!(f, "（期望 {:?} / 实际 {:?}）", self.expected_shape, self.actual_shape)?;
        }
        Ok(())
    }
}

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

    #[error("{formatted}", formatted = runtime_error_format(*line, *col, message))]
    RuntimeError {
        line: Option<usize>,
        col: Option<usize>,
        message: String,
    },

    /// VM 编译期失败：VM 不支持某些结构（如特定语法/算子），
    /// 在执行前就失败，**没有副作用**，可安全回退到解释器。
    /// 与 `RuntimeError` 区分：`RuntimeError` 是运行时失败（可能已部分
    /// 执行并产生副作用如 println），不应回退。
    #[error("VM 编译期不支持 — {message}")]
    VmCompileFailed { message: String },

    /// 形状错误（护城河 F：T2 FormalExplain 动态层）。
    /// 携带 tape 上下文与根因分析说明，由 autodiff backward 抛出。
    #[error("形状错误（{context}）— {message}")]
    ShapeMismatch {
        context: TapeErrorContext,
        message: String,
    },

    /// 关系调试器错误（护城河 F：5 类错误分类）。
    /// 由 autodiff backward 在检测到关系级错误时抛出，携带错误类型分类
    ///（`ErrorType`）与报错节点 id。与 `ShapeMismatch` 区分：
    /// - `ShapeMismatch`：前向 shape 不匹配（携带完整 `TapeErrorContext`）
    /// - `RelationError`：关系调试器归类后的错误（携带 5 类 `ErrorType`）
    #[error("关系错误（{error_type:?}，节点 #{tape_node_id}）— {message}")]
    RelationError {
        error_type: ErrorType,
        tape_node_id: usize,
        message: String,
    },

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

    /// Execution budget exhausted: step limit or timeout reached.
    /// Raised by the step counter in the interpreter/VM when the configured
    /// budget hits zero. Users enable this explicitly via
    /// `std::runtime::with_step_limit` / `with_timeout_ms`.
    #[error("执行超时：{message}")]
    Timeout { message: String },
}

pub type TenthResult<T> = Result<T, TenthError>;

/// 格式化 RuntimeError 的显示文本（问题12）。
/// 若 line/col 存在则显示 "第 L 行第 C 列：运行时错误 — message"，
/// 否则保持原格式 "运行时错误 — message"。
fn runtime_error_format(line: Option<usize>, col: Option<usize>, message: &str) -> String {
    match (line, col) {
        (Some(l), Some(c)) => format!("第 {} 行第 {} 列：运行时错误 — {}", l, c, message),
        _ => format!("运行时错误 — {}", message),
    }
}

/// 编译期警告（非致命）。用于内存/算力预估等不阻断编译的提示。
#[derive(Debug, Clone, PartialEq)]
pub struct TenthWarning {
    pub line: usize,
    pub col: usize,
    pub message: String,
}

impl TenthWarning {
    pub fn new(line: usize, col: usize, message: String) -> Self {
        Self { line, col, message }
    }

    /// Pretty-print the warning with source context (复用 TenthError 的源码定位逻辑)。
    pub fn display_with_source(&self, source: Option<&str>) -> String {
        if let Some(src) = source {
            let source_line = src.lines().nth(self.line.saturating_sub(1)).unwrap_or("");
            let caret = build_caret(self.col, source_line);
            format!(
                "警告：第 {} 行第 {} 列：{}\n  |\n{:>3} | {}\n  | {}",
                self.line, self.col, self.message, self.line, source_line, caret
            )
        } else {
            format!("警告：第 {} 行第 {} 列：{}", self.line, self.col, self.message)
        }
    }
}

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
            TenthError::RuntimeError { line, col, .. } => (*line, *col),
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
            TenthError::RuntimeError { message, .. } => {
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
