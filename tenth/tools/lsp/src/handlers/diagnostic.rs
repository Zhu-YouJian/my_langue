use tenth::error::TenthError;
use tenth::hir::lower::Lowerer;
use tenth::lexer::lexer::Lexer;
use tenth::parser::parser::Parser;

use super::Handler;
use crate::lsp_types::*;

pub struct DiagnosticHandler;

impl Handler for DiagnosticHandler {
    fn handle(&self, params: Option<&serde_json::Value>) -> serde_json::Value {
        let uri = params
            .and_then(|p| p.get("textDocument"))
            .and_then(|td| td.get("uri"))
            .and_then(|u| u.as_str())
            .unwrap_or("");

        let diagnostics = diagnose_uri(uri);
        serde_json::to_value(diagnostics).unwrap()
    }
}

/// Diagnose a file by URI. Reads from disk (for pull diagnostics).
pub fn diagnose_uri(uri: &str) -> Vec<Diagnostic> {
    let path = crate::document_store::uri_to_path(uri);
    let content = match std::fs::read_to_string(&path) {
        Ok(c) => c,
        Err(_) => return Vec::new(),
    };
    diagnose_source(&content)
}

/// Diagnose source text: lex + parse + lower.
pub fn diagnose_source(content: &str) -> Vec<Diagnostic> {
    let mut diagnostics = Vec::new();

    // Lex
    let mut lexer = Lexer::new(content);
    let tokens = match lexer.tokenize() {
        Ok(t) => t,
        Err(e) => {
            diagnostics.push(error_to_diagnostic(&e));
            return diagnostics;
        }
    };

    // Parse with recovery
    let mut parser = Parser::new(tokens);
    let (program, errors) = parser.parse_program_with_recovery();
    for e in &errors {
        diagnostics.push(error_to_diagnostic(e));
    }

    // Lower (type checking / semantic analysis)
    if errors.is_empty() {
        let mut lowerer = Lowerer::new();
        if let Err(e) = lowerer.lower_program(&program) {
            diagnostics.push(error_to_diagnostic(&e));
        }
    }

    diagnostics
}

fn error_to_diagnostic(err: &TenthError) -> Diagnostic {
    let (line, col) = err.location();
    let line = line.map(|l| l.saturating_sub(1) as u32).unwrap_or(0);
    let col = col.map(|c| c.saturating_sub(1) as u32).unwrap_or(0);

    Diagnostic {
        range: Range {
            start: Position { line, character: col },
            end: Position { line, character: col + 1 },
        },
        severity: DiagnosticSeverity::Error,
        message: err.to_string(),
        source: Some("tenth".to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_diagnose_valid_program_no_diagnostics() {
        // 合法的 Tenth 程序：函数定义 + 简单返回值
        let src = "fn add(a: i32, b: i32) -> i32 { a + b }";
        let diags = diagnose_source(src);
        assert!(
            diags.is_empty(),
            "expected no diagnostics for valid program, got: {:?}",
            diags.iter().map(|d| &d.message).collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_diagnose_empty_file_no_diagnostics() {
        // 空文件不应产生诊断（lexer 不会报错，parser 也接受空程序）
        let diags = diagnose_source("");
        assert!(diags.is_empty(), "expected no diagnostics for empty file");
    }

    #[test]
    fn test_diagnose_undefined_variable() {
        // 引用未声明的变量应触发 TypeError
        let src = "fn buggy() -> i32 { undefined_name }";
        let diags = diagnose_source(src);
        assert!(
            !diags.is_empty(),
            "expected at least one diagnostic for undefined variable"
        );
        // 至少有一条诊断包含 "undefined variable"
        let has_undef = diags.iter().any(|d| d.message.contains("undefined variable"));
        assert!(
            has_undef,
            "expected 'undefined variable' in diagnostics, got: {:?}",
            diags.iter().map(|d| &d.message).collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_diagnose_type_mismatch() {
        // 类型不匹配：let 注解 shape 与实际 shape 不符
        // 参考 tenth/tests/shape_check_compile_test.rs 中的同类用例
        let src = r#"
fn bad() -> Tensor[f64, ..] {
    let x: Tensor[f64, 3, 4] = zeros(2, 3);
    x
}
"#;
        let diags = diagnose_source(src);
        assert!(
            !diags.is_empty(),
            "expected at least one diagnostic for type/shape mismatch"
        );
    }

    #[test]
    fn test_diagnose_unclosed_string_literal() {
        // 未闭合的字符串字面量：lexer 会返回 LexerError
        let src = "fn bad() -> i32 { \"unclosed }";
        let diags = diagnose_source(src);
        assert!(
            !diags.is_empty(),
            "expected at least one diagnostic for unclosed string"
        );
        // 第一条诊断应由 lexer 错误触发（字符串未闭合）
        let msg = &diags[0].message;
        assert!(
            msg.contains("未闭合") || msg.contains("unclosed") || msg.contains("string"),
            "expected unclosed-string error, got: {}",
            msg
        );
    }

    #[test]
    fn test_diagnose_severity_is_error() {
        // 所有诊断的严重级别都应该是 Error
        let src = "fn bad() -> i32 { undefined_name }";
        let diags = diagnose_source(src);
        assert!(!diags.is_empty());
        for d in &diags {
            assert!(
                matches!(d.severity, DiagnosticSeverity::Error),
                "expected Error severity, got {:?} for: {}",
                d.severity,
                d.message
            );
        }
    }

    #[test]
    fn test_diagnose_source_field_set() {
        // source 字段应该被设置为 "tenth"
        let src = "fn bad() -> i32 { undefined_name }";
        let diags = diagnose_source(src);
        assert!(!diags.is_empty());
        for d in &diags {
            assert_eq!(d.source.as_deref(), Some("tenth"));
        }
    }
}
