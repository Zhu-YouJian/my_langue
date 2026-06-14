use std::fs;
use std::path::Path;

use tenth::lexer::lexer::Lexer;
use tenth::parser::parser::Parser;
use tenth::error::TenthError;

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

        let diagnostics = diagnose_file(uri);
        serde_json::to_value(diagnostics).unwrap()
    }
}

fn diagnose_file(uri: &str) -> Vec<Diagnostic> {
    let path = uri_to_path(uri);
    let content = match fs::read_to_string(Path::new(&path)) {
        Ok(c) => c,
        Err(_) => return Vec::new(),
    };

    let mut diagnostics = Vec::new();

    // Lex
    let mut lexer = Lexer::new(&content);
    let tokens = match lexer.tokenize() {
        Ok(t) => t,
        Err(e) => {
            diagnostics.push(error_to_diagnostic(&e));
            return diagnostics;
        }
    };

    // Parse
    let mut parser = Parser::new(tokens);
    if let Err(e) = parser.parse_program() {
        diagnostics.push(error_to_diagnostic(&e));
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

fn uri_to_path(uri: &str) -> String {
    if let Some(stripped) = uri.strip_prefix("file:///") {
        stripped.to_string()
    } else if let Some(stripped) = uri.strip_prefix("file://") {
        stripped.to_string()
    } else {
        uri.to_string()
    }
}
