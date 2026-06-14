use super::Handler;
use crate::lsp_types::*;

pub struct CompletionHandler;

impl Handler for CompletionHandler {
    fn handle(&self, _params: Option<&serde_json::Value>) -> serde_json::Value {
        let items = completion_items();
        serde_json::to_value(items).unwrap()
    }
}

fn completion_items() -> Vec<CompletionItem> {
    let mut items = Vec::new();

    // Keywords
    let keywords = [
        "fn", "let", "mut", "if", "else", "while", "for", "in", "return",
        "struct", "enum", "impl", "trait", "import", "tensor", "true", "false",
    ];
    for kw in &keywords {
        items.push(CompletionItem {
            label: kw.to_string(),
            kind: CompletionItemKind::Keyword,
            detail: Some(format!("keyword {}", kw)),
        });
    }

    // Built-in functions
    let builtins = [
        ("print", "Print a value to stdout"),
        ("len", "Return the length of a collection"),
        ("range", "Create an iterator range"),
        ("shape", "Return the shape of a tensor"),
        ("reshape", "Reshape a tensor"),
        ("zeros", "Create a tensor of zeros"),
        ("ones", "Create a tensor of ones"),
        ("randn", "Create a tensor with random normal values"),
    ];
    for (name, doc) in &builtins {
        items.push(CompletionItem {
            label: name.to_string(),
            kind: CompletionItemKind::Function,
            detail: Some(doc.to_string()),
        });
    }

    // Types
    let types = [
        ("i64", "64-bit signed integer"),
        ("f64", "64-bit floating point"),
        ("bool", "Boolean type"),
        ("String", "UTF-8 string type"),
        ("Tensor", "N-dimensional tensor"),
        ("Vec", "Dynamic array"),
    ];
    for (name, doc) in &types {
        items.push(CompletionItem {
            label: name.to_string(),
            kind: CompletionItemKind::Class,
            detail: Some(doc.to_string()),
        });
    }

    items
}
