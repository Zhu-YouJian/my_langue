use super::Handler;
use crate::lsp_types::*;

pub struct HoverHandler;

impl Handler for HoverHandler {
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

        // Simplified: return hover info for keywords and builtins
        // A full implementation would use a symbol table from the AST
        let _ = (uri, position);

        // For now, return null — a complete implementation would look up
        // the symbol at the given position and return type info + docs
        serde_json::Value::Null
    }
}

/// Lookup documentation for a known symbol name.
/// Used by a future full implementation.
#[allow(dead_code)]
fn lookup_hover(name: &str) -> Option<Hover> {
    let docs: &[(&str, &str)] = &[
        ("fn", "Keyword: Define a function"),
        ("let", "Keyword: Bind a variable"),
        ("mut", "Keyword: Mutable binding"),
        ("if", "Keyword: Conditional expression"),
        ("else", "Keyword: Else branch"),
        ("while", "Keyword: While loop"),
        ("for", "Keyword: For-in loop"),
        ("in", "Keyword: Iterator binding in for loop"),
        ("return", "Keyword: Return from function"),
        ("struct", "Keyword: Define a struct"),
        ("enum", "Keyword: Define an enum"),
        ("impl", "Keyword: Implement methods on a type"),
        ("trait", "Keyword: Define a trait"),
        ("import", "Keyword: Import a module"),
        ("tensor", "Keyword: Tensor type annotation"),
        ("true", "Literal: Boolean true"),
        ("false", "Literal: Boolean false"),
        ("print", "Builtin: Print a value to stdout"),
        ("len", "Builtin: Return the length of a collection"),
        ("range", "Builtin: Create an iterator range"),
        ("shape", "Builtin: Return the shape of a tensor"),
        ("reshape", "Builtin: Reshape a tensor"),
        ("zeros", "Builtin: Create a tensor of zeros"),
        ("ones", "Builtin: Create a tensor of ones"),
        ("randn", "Builtin: Create a tensor with random normal values"),
        ("i64", "Type: 64-bit signed integer"),
        ("f64", "Type: 64-bit floating point"),
        ("bool", "Type: Boolean"),
        ("String", "Type: UTF-8 string"),
        ("Tensor", "Type: N-dimensional tensor"),
        ("Vec", "Type: Dynamic array"),
    ];

    docs.iter()
        .find(|(k, _)| *k == name)
        .map(|(_, doc)| Hover {
            contents: MarkupContent {
                kind: "plaintext".to_string(),
                value: doc.to_string(),
            },
        })
}
