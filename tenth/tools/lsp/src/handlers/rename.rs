use std::collections::HashMap;

use tenth::lexer::lexer::Lexer;
use tenth::lexer::token::{Token, TokenKind};

use super::Handler;
use crate::lsp_types::*;

pub struct RenameHandler;

impl Handler for RenameHandler {
    fn handle(&self, params: Option<&serde_json::Value>) -> serde_json::Value {
        let uri = params
            .and_then(|p| p.get("textDocument"))
            .and_then(|td| td.get("uri"))
            .and_then(|u| u.as_str())
            .unwrap_or("");

        let position = params
            .and_then(|p| p.get("position"))
            .and_then(|pos| {
                let line = pos.get("line")?.as_u64()? as u32;
                let character = pos.get("character")?.as_u64()? as u32;
                Some(Position { line, character })
            });

        let new_name = params
            .and_then(|p| p.get("newName"))
            .and_then(|n| n.as_str())
            .unwrap_or("");

        let (line, character) = match position {
            Some(pos) => (pos.line, pos.character),
            None => return serde_json::Value::Null,
        };

        if new_name.is_empty() {
            return serde_json::Value::Null;
        }

        let workspace_edit = rename_symbol(uri, line, character, new_name);
        serde_json::to_value(workspace_edit).unwrap_or(serde_json::Value::Null)
    }
}

fn rename_symbol(uri: &str, line: u32, character: u32, new_name: &str) -> WorkspaceEdit {
    let content = match crate::document_store::get_content_or_disk_global(uri) {
        Some(c) => c,
        None => return WorkspaceEdit { changes: HashMap::new() },
    };

    let mut lexer = Lexer::new(&content);
    let tokens = match lexer.tokenize() {
        Ok(t) => t,
        Err(_) => return WorkspaceEdit { changes: HashMap::new() },
    };

    // Find the identifier at the cursor position
    let target_line = (line + 1) as usize;
    let target_col = character as usize;

    let identifier = find_token_at_position(&tokens, target_line, target_col);
    let identifier = match identifier {
        Some(name) => name,
        None => return WorkspaceEdit { changes: HashMap::new() },
    };

    // Create TextEdits for all occurrences
    let mut edits = Vec::new();
    for token in &tokens {
        if let TokenKind::Identifier(ref name) = token.kind {
            if name == &identifier {
                let tok_line = token.span.line.saturating_sub(1) as u32;
                let tok_col = crate::span::token_start_col0(token) as u32;
                edits.push(TextEdit {
                    range: Range {
                        start: Position { line: tok_line, character: tok_col },
                        end: Position { line: tok_line, character: tok_col + name.len() as u32 },
                    },
                    new_text: new_name.to_string(),
                });
            }
        }
    }

    let mut changes = HashMap::new();
    changes.insert(uri.to_string(), edits);
    WorkspaceEdit { changes }
}

fn find_token_at_position(tokens: &[Token], target_line: usize, target_col0: usize) -> Option<String> {
    for token in tokens {
        if token.kind == TokenKind::Eof {
            continue;
        }
        let token_line = token.span.line;

        if let TokenKind::Identifier(ref name) = token.kind {
            if token_line == target_line {
                let start0 = crate::span::token_start_col0(token);
                if target_col0 >= start0 && target_col0 < start0 + name.len() {
                    return Some(name.clone());
                }
            }
        }
    }
    None
}
