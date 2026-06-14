mod handlers;
mod io;
mod lsp_types;

use handlers::{
    initialize::InitializeHandler,
    diagnostic::DiagnosticHandler,
    hover::HoverHandler,
    completion::CompletionHandler,
    definition::DefinitionHandler,
    formatting::FormattingHandler,
    Handler,
};
use lsp_types::LspResponse;

fn main() {
    let initialize = InitializeHandler;
    let diagnostic = DiagnosticHandler;
    let hover = HoverHandler;
    let completion = CompletionHandler;
    let definition = DefinitionHandler;
    let formatting = FormattingHandler;

    while let Some(request) = io::read_message() {
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

        io::write_message(response);
    }
}
