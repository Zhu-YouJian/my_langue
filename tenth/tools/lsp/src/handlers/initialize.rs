use serde_json::json;

use super::Handler;
use crate::lsp_types::*;

pub struct InitializeHandler;

impl Handler for InitializeHandler {
    fn handle(&self, _params: Option<&serde_json::Value>) -> serde_json::Value {
        let result = InitializeResult {
            capabilities: ServerCapabilities {
                diagnostic_provider: Some(DiagnosticProvider {
                    identifier: Some("tenth".to_string()),
                }),
                hover_provider: Some(true),
                completion_provider: Some(CompletionOptions {
                    trigger_characters: vec![".".to_string(), ":".to_string()],
                }),
                definition_provider: Some(true),
                document_formatting_provider: Some(true),
            },
        };

        let mut value = serde_json::to_value(&result).unwrap();
        if let Some(obj) = value.as_object_mut() {
            obj.insert(
                "serverInfo".to_string(),
                json!({
                    "name": "tenth-lsp",
                    "version": "0.1.0"
                }),
            );
        }
        value
    }
}
