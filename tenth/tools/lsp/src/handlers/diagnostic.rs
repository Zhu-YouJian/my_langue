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
