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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_initialize_returns_server_capabilities() {
        // InitializeHandler.handle() 应返回包含完整 ServerCapabilities 的 JSON
        // 注意：lsp_types.rs 中未启用 #[serde(rename_all = "camelCase")]
        // 所以字段名是 Rust 默认的 snake_case
        let handler = InitializeHandler;
        let value = handler.handle(None);
        let obj = value.as_object().expect("result should be a JSON object");

        // 顶层应有 "capabilities" 字段
        let caps = obj.get("capabilities")
            .and_then(|c| c.as_object())
            .expect("result should contain 'capabilities' object");

        // 检查各 provider 是否存在（snake_case 字段名）
        let required = [
            "hoverProvider",
            "completionProvider",
            "definitionProvider",
            "documentFormattingProvider",
            "documentSymbolProvider",
            "referencesProvider",
            "renameProvider",
            "signatureHelpProvider",
            "foldingRangeProvider",
            "semanticTokensProvider",
            "diagnosticProvider",
            "textDocumentSync",
        ];
        for field in &required {
            // 同时允许 snake_case 和 camelCase 以兼容未来可能的修改
            let camel = *field;
            let snake = camel_to_snake(camel);
            assert!(
                caps.get(camel).is_some() || caps.get(snake.as_str()).is_some(),
                "capabilities should declare {} (or {}), got: {:?}",
                camel,
                snake,
                caps.keys().collect::<Vec<_>>()
            );
        }
    }

    #[test]
    fn test_initialize_includes_server_info() {
        // 应包含 serverInfo 字段（name + version）
        let handler = InitializeHandler;
        let value = handler.handle(None);
        let obj = value.as_object().expect("result should be a JSON object");
        let info = obj.get("serverInfo")
            .and_then(|i| i.as_object())
            .expect("result should contain 'serverInfo'");
        assert_eq!(
            info.get("name").and_then(|n| n.as_str()),
            Some("tenth-lsp"),
            "serverInfo.name should be 'tenth-lsp'"
        );
        assert!(
            info.get("version").and_then(|v| v.as_str()).is_some(),
            "serverInfo.version should be present"
        );
    }

    #[test]
    fn test_initialize_completion_provider_has_trigger_characters() {
        // completionProvider 应有 trigger_characters 包含 "."
        let handler = InitializeHandler;
        let value = handler.handle(None);
        let caps = value.get("capabilities")
            .and_then(|c| c.as_object())
            .expect("missing capabilities");
        // 兼容 snake_case 与 camelCase
        let comp = caps.get("completionProvider")
            .or_else(|| caps.get("completion_provider"))
            .and_then(|c| c.as_object())
            .expect("missing completionProvider");
        let triggers = comp.get("triggerCharacters")
            .or_else(|| comp.get("trigger_characters"))
            .and_then(|t| t.as_array())
            .expect("missing triggerCharacters");
        let trigger_strs: Vec<&str> = triggers.iter()
            .filter_map(|v| v.as_str())
            .collect();
        assert!(
            trigger_strs.contains(&"."),
            "completion triggerCharacters should contain '.', got: {:?}",
            trigger_strs
        );
    }

    #[test]
    fn test_initialize_semantic_tokens_legend_has_token_types() {
        // semanticTokensProvider.legend.tokenTypes 应有 10 种类型
        let handler = InitializeHandler;
        let value = handler.handle(None);
        let caps = value.get("capabilities")
            .and_then(|c| c.as_object())
            .expect("missing capabilities");
        let st = caps.get("semanticTokensProvider")
            .or_else(|| caps.get("semantic_tokens_provider"))
            .and_then(|s| s.as_object())
            .expect("missing semanticTokensProvider");
        let legend = st.get("legend")
            .and_then(|l| l.as_object())
            .expect("missing legend");
        let token_types = legend.get("tokenTypes")
            .or_else(|| legend.get("token_types"))
            .and_then(|t| t.as_array())
            .expect("missing tokenTypes");
        assert!(
            token_types.len() >= 10,
            "expected at least 10 token types, got {}",
            token_types.len()
        );
        // 应包含 "keyword"
        let type_strs: Vec<&str> = token_types.iter()
            .filter_map(|v| v.as_str())
            .collect();
        assert!(
            type_strs.contains(&"keyword"),
            "token types should contain 'keyword', got: {:?}",
            type_strs
        );
    }

    /// 简单的 camelCase → snake_case 转换（用于字段名兼容性检查）
    fn camel_to_snake(s: &str) -> String {
        let mut out = String::new();
        for c in s.chars() {
            if c.is_uppercase() {
                out.push('_');
                out.push(c.to_lowercase().next().unwrap_or(c));
            } else {
                out.push(c);
            }
        }
        out
    }
}
