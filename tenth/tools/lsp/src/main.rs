mod document_store;
mod handlers;
mod io;
mod lsp_types;

use handlers::{
    completion::CompletionHandler,
    definition::DefinitionHandler,
    diagnostic::DiagnosticHandler,
    document_symbol::DocumentSymbolHandler,
    folding_range::FoldingRangeHandler,
    formatting::FormattingHandler,
    hover::HoverHandler,
    initialize::InitializeHandler,
    references::ReferencesHandler,
    rename::RenameHandler,
    semantic_tokens::SemanticTokensHandler,
    signature_help::SignatureHelpHandler,
    Handler,
};
use lsp_types::{LspNotification, LspResponse};

fn main() {
    let initialize = InitializeHandler;
    let diagnostic = DiagnosticHandler;
    let hover = HoverHandler;
    let completion = CompletionHandler;
    let definition = DefinitionHandler;
    let formatting = FormattingHandler;
    let document_symbol = DocumentSymbolHandler;
    let references = ReferencesHandler;
    let rename = RenameHandler;
    let signature_help = SignatureHelpHandler;
    let folding_range = FoldingRangeHandler;
    let semantic_tokens = SemanticTokensHandler;

    while let Some(request) = io::read_message() {
        // Check if this is a notification (no id) or a request (has id)
        let is_notification = request.id.is_none();

        // Handle notifications
        if is_notification {
            handle_notification(&request.method, request.params.as_ref());
            // Check if we should exit
            if request.method == "exit" {
                break;
            }
            continue;
        }

        // Handle requests
        let response = match request.method.as_str() {
            "initialize" => {
                let result = initialize.handle(request.params.as_ref());
                LspResponse {
                    jsonrpc: "2.0".to_string(),
                    id: request.id,
                    result: Some(result),
                    error: None,
                }
            }
            "textDocument/diagnostic" => {
                let result = diagnostic.handle(request.params.as_ref());
                LspResponse {
                    jsonrpc: "2.0".to_string(),
                    id: request.id,
                    result: Some(result),
                    error: None,
                }
            }
            "textDocument/hover" => {
                let result = hover.handle(request.params.as_ref());
                LspResponse {
                    jsonrpc: "2.0".to_string(),
                    id: request.id,
                    result: Some(result),
                    error: None,
                }
            }
            "textDocument/completion" => {
                let result = completion.handle(request.params.as_ref());
                LspResponse {
                    jsonrpc: "2.0".to_string(),
                    id: request.id,
                    result: Some(result),
                    error: None,
                }
            }
            "textDocument/definition" => {
                let result = definition.handle(request.params.as_ref());
                LspResponse {
                    jsonrpc: "2.0".to_string(),
                    id: request.id,
                    result: Some(result),
                    error: None,
                }
            }
            "textDocument/formatting" => {
                let result = formatting.handle(request.params.as_ref());
                LspResponse {
                    jsonrpc: "2.0".to_string(),
                    id: request.id,
                    result: Some(result),
                    error: None,
                }
            }
            "textDocument/documentSymbol" => {
                let result = document_symbol.handle(request.params.as_ref());
                LspResponse {
                    jsonrpc: "2.0".to_string(),
                    id: request.id,
                    result: Some(result),
                    error: None,
                }
            }
            "textDocument/references" => {
                let result = references.handle(request.params.as_ref());
                LspResponse {
                    jsonrpc: "2.0".to_string(),
                    id: request.id,
                    result: Some(result),
                    error: None,
                }
            }
            "textDocument/rename" => {
                let result = rename.handle(request.params.as_ref());
                LspResponse {
                    jsonrpc: "2.0".to_string(),
                    id: request.id,
                    result: Some(result),
                    error: None,
                }
            }
            "textDocument/signatureHelp" => {
                let result = signature_help.handle(request.params.as_ref());
                LspResponse {
                    jsonrpc: "2.0".to_string(),
                    id: request.id,
                    result: Some(result),
                    error: None,
                }
            }
            "textDocument/foldingRange" => {
                let result = folding_range.handle(request.params.as_ref());
                LspResponse {
                    jsonrpc: "2.0".to_string(),
                    id: request.id,
                    result: Some(result),
                    error: None,
                }
            }
            "textDocument/semanticTokens/full" => {
                let result = semantic_tokens.handle(request.params.as_ref());
                LspResponse {
                    jsonrpc: "2.0".to_string(),
                    id: request.id,
                    result: Some(result),
                    error: None,
                }
            }
            "shutdown" => {
                LspResponse {
                    jsonrpc: "2.0".to_string(),
                    id: request.id,
                    result: Some(serde_json::Value::Null),
                    error: None,
                }
            }
            "exit" => {
                break;
            }
            _ => {
                LspResponse {
                    jsonrpc: "2.0".to_string(),
                    id: request.id,
                    result: None,
                    error: Some(lsp_types::LspError {
                        code: -32601,
                        message: format!("Method not found: {}", request.method),
                    }),
                }
            }
        };

        io::write_response(response);
    }
}

/// Handle LSP notifications (messages without an id).
fn handle_notification(method: &str, params: Option<&serde_json::Value>) {
    match method {
        "initialized" => {
            // Client has received our initialize response. Nothing to do.
        }
        "textDocument/didOpen" => {
            handle_did_open(params);
        }
        "textDocument/didChange" => {
            handle_did_change(params);
        }
        "textDocument/didClose" => {
            handle_did_close(params);
        }
        "textDocument/didSave" => {
            // Re-publish diagnostics on save
            if let Some(uri) = params
                .and_then(|p| p.get("textDocument"))
                .and_then(|td| td.get("uri"))
                .and_then(|u| u.as_str())
            {
                publish_diagnostics(uri);
            }
        }
        "$/cancelRequest" => {
            // Request cancellation — we don't support it, just ignore
        }
        _ => {
            // Unknown notification — ignore
        }
    }
}

fn handle_did_open(params: Option<&serde_json::Value>) {
    let uri = params
        .and_then(|p| p.get("textDocument"))
        .and_then(|td| td.get("uri"))
        .and_then(|u| u.as_str())
        .unwrap_or("");
    let version = params
        .and_then(|p| p.get("textDocument"))
        .and_then(|td| td.get("version"))
        .and_then(|v| v.as_i64())
        .unwrap_or(0);
    let text = params
        .and_then(|p| p.get("textDocument"))
        .and_then(|td| td.get("text"))
        .and_then(|t| t.as_str())
        .unwrap_or("");

    if !uri.is_empty() {
        document_store::global().open(uri, version, text);
        publish_diagnostics(uri);
    }
}

fn handle_did_change(params: Option<&serde_json::Value>) {
    let uri = params
        .and_then(|p| p.get("textDocument"))
        .and_then(|td| td.get("uri"))
        .and_then(|u| u.as_str())
        .unwrap_or("");
    let version = params
        .and_then(|p| p.get("textDocument"))
        .and_then(|td| td.get("version"))
        .and_then(|v| v.as_i64())
        .unwrap_or(0);

    // Get content changes
    let changes = params
        .and_then(|p| p.get("contentChanges"))
        .and_then(|c| c.as_array());

    if uri.is_empty() {
        return;
    }

    if let Some(changes) = changes {
        for change in changes {
            // Check if this is a full sync (no range) or incremental (has range)
            if let Some(range) = change.get("range") {
                // Incremental sync
                let start_line = range
                    .get("start")
                    .and_then(|s| s.get("line"))
                    .and_then(|l| l.as_u64())
                    .unwrap_or(0) as u32;
                let start_char = range
                    .get("start")
                    .and_then(|s| s.get("character"))
                    .and_then(|c| c.as_u64())
                    .unwrap_or(0) as u32;
                let end_line = range
                    .get("end")
                    .and_then(|e| e.get("line"))
                    .and_then(|l| l.as_u64())
                    .unwrap_or(0) as u32;
                let end_char = range
                    .get("end")
                    .and_then(|e| e.get("character"))
                    .and_then(|c| c.as_u64())
                    .unwrap_or(0) as u32;
                let text = change
                    .get("text")
                    .and_then(|t| t.as_str())
                    .unwrap_or("");

                document_store::global().update_incremental(
                    uri,
                    version,
                    &lsp_types::Range {
                        start: lsp_types::Position {
                            line: start_line,
                            character: start_char,
                        },
                        end: lsp_types::Position {
                            line: end_line,
                            character: end_char,
                        },
                    },
                    text,
                );
            } else if let Some(text) = change.get("text").and_then(|t| t.as_str()) {
                // Full sync
                document_store::global().update_full(uri, version, text);
            }
        }
    }

    publish_diagnostics(uri);
}

fn handle_did_close(params: Option<&serde_json::Value>) {
    let uri = params
        .and_then(|p| p.get("textDocument"))
        .and_then(|td| td.get("uri"))
        .and_then(|u| u.as_str())
        .unwrap_or("");

    if !uri.is_empty() {
        document_store::global().close(uri);
        // Clear diagnostics for closed document
        let notification = LspNotification {
            jsonrpc: "2.0".to_string(),
            method: "textDocument/publishDiagnostics".to_string(),
            params: Some(
                serde_json::to_value(lsp_types::PublishDiagnosticsParams {
                    uri: uri.to_string(),
                    diagnostics: Vec::new(),
                })
                .unwrap(),
            ),
        };
        io::write_notification(notification);
    }
}

/// Compute diagnostics for a document and publish them via notification.
fn publish_diagnostics(uri: &str) {
    let content = document_store::global().get_content_or_disk(uri);
    let diagnostics = match content {
        Some(ref c) => handlers::diagnostic::diagnose_source(c),
        None => Vec::new(),
    };

    let notification = LspNotification {
        jsonrpc: "2.0".to_string(),
        method: "textDocument/publishDiagnostics".to_string(),
        params: Some(
            serde_json::to_value(lsp_types::PublishDiagnosticsParams {
                uri: uri.to_string(),
                diagnostics,
            })
            .unwrap(),
        ),
    };
    io::write_notification(notification);
}
