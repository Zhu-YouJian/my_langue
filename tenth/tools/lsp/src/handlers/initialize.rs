use serde_json::json;

use super::Handler;
use crate::lsp_types::*;

pub struct InitializeHandler;

impl Handler for InitializeHandler {
    fn handle(&self, _params: Option<&serde_json::Value>) -> serde_json::Value {
        let result = InitializeResult {
            capabilities: ServerCapabilities {
                text_document_sync: Some(TextDocumentSyncOptions {
                    open_close: true,
                    change: 2, // Incremental sync
                }),
                diagnostic_provider: Some(DiagnosticProvider {
                    identifier: Some("tenth".to_string()),
                }),
                hover_provider: Some(true),
                completion_provider: Some(CompletionOptions {
                    trigger_characters: vec![".".to_string(), ":".to_string()],
                }),
                definition_provider: Some(true),
                document_formatting_provider: Some(true),
                document_symbol_provider: Some(true),
                references_provider: Some(true),
                rename_provider: Some(true),
                signature_help_provider: Some(SignatureHelpOptions {
                    trigger_characters: vec!["(".to_string(), ",".to_string()],
                }),
                folding_range_provider: Some(true),
                semantic_tokens_provider: Some(SemanticTokensOptions {
                    legend: SemanticTokensLegend {
                        token_types: vec![
                            "keyword".to_string(),
                            "function".to_string(),
                            "variable".to_string(),
                            "type".to_string(),
                            "string".to_string(),
                            "number".to_string(),
                            "operator".to_string(),
                            "comment".to_string(),
                            "enumMember".to_string(),
                            "struct".to_string(),
                        ],
                        token_modifiers: vec![
                            "declaration".to_string(),
                            "readonly".to_string(),
                            "static".to_string(),
                        ],
                    },
                    full: true,
                    range: Some(true),
                }),
            },
        };

        let mut value = serde_json::to_value(&result).unwrap();
        if let Some(obj) = value.as_object_mut() {
            obj.insert(
                "serverInfo".to_string(),
                json!({
                    "name": "tenth-lsp",
                    "version": "0.2.0"
                }),
            );
        }
        value
    }
}
