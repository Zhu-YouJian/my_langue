use std::collections::HashSet;

use super::Handler;
use crate::lsp_types::*;
use tenth::lexer::lexer::Lexer;
use tenth::parser::ast::ItemKind;
use tenth::parser::parser::Parser;

pub struct CompletionHandler;

impl Handler for CompletionHandler {
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

        let source = crate::document_store::get_content_or_disk_global(uri)
            .unwrap_or_default();
        let is_method_context = position
            .map(|pos| is_after_dot(&source, pos))
            .unwrap_or(false);

        let items = if is_method_context {
            method_completion_items(&source)
        } else {
            full_completion_items(&source)
        };

        serde_json::to_value(items).unwrap()
    }
}

/// Read the source file content from a file URI.
/// Returns empty string if the file cannot be read.
fn read_source(uri: &str) -> String {
    // Handle file:// URIs
    let path = if let Some(rest) = uri.strip_prefix("file:///") {
        // On Windows, the path after file:/// is like /C:/...
        // strip the leading slash if it looks like a drive letter
        if rest.len() > 2 && rest.chars().nth(1) == Some(':') {
            &rest[1..]
        } else {
            rest
        }
    } else if let Some(rest) = uri.strip_prefix("file://") {
        rest
    } else {
        uri
    };

    std::fs::read_to_string(path).unwrap_or_default()
}

/// Check if the cursor position is immediately after a `.` character,
/// indicating a method call or field access context.
fn is_after_dot(source: &str, position: Position) -> bool {
    let lines: Vec<&str> = source.lines().collect();
    let line_idx = position.line as usize;
    if line_idx >= lines.len() {
        return false;
    }
    let line = lines[line_idx];
    let char_idx = position.character as usize;
    if char_idx == 0 || char_idx > line.len() {
        return false;
    }
    // Check the character just before the cursor
    let before = &line[..char_idx];
    before.ends_with('.')
}

/// Build completion items for method/field context (after a `.`).
/// Includes methods from impl blocks and struct fields.
fn method_completion_items(source: &str) -> Vec<CompletionItem> {
    let mut items = Vec::new();

    if let Some(program) = parse_source(source) {
        let mut method_names: HashSet<String> = HashSet::new();

        for item in &program.items {
            if let ItemKind::Impl { functions, .. } = &item.kind {
                for func in functions {
                    if let ItemKind::Function { name, .. } = &func.kind {
                        if method_names.insert(name.name.clone()) {
                            items.push(CompletionItem {
                                label: name.name.clone(),
                                kind: CompletionItemKind::Method,
                                detail: Some(format!("method {}", name.name)),
                            documentation: None,
                            insert_text: None,
                        });
                        }
                    }
                }
            }
        }
    }

    // Fallback: if no methods found, still return empty — the client
    // will show no completions which is correct for unknown types
    items
}

/// Build the full set of completion items: static keywords/builtins/types
/// plus user-defined symbols from the parsed program.
fn full_completion_items(source: &str) -> Vec<CompletionItem> {
    let mut items = Vec::new();

    // Add user-defined symbols from the parsed program
    if let Some(program) = parse_source(source) {
        add_user_defined_items(&mut items, &program);
    }

    // Add static keywords, builtins, and types as fallback
    add_keywords(&mut items);
    add_builtins(&mut items);
    add_types(&mut items);

    items
}

/// Try to parse the source file. Returns Some(Program) on success,
/// None on failure (caller falls back to static list).
fn parse_source(source: &str) -> Option<tenth::parser::ast::Program> {
    let mut lexer = Lexer::new(source);
    let tokens = lexer.tokenize().ok()?;
    let mut parser = Parser::new(tokens);
    // Use parse_program_with_recovery for fault tolerance in LSP context
    let (program, _errors) = parser.parse_program_with_recovery();
    Some(program)
}

/// Extract user-defined functions, structs, and enums from the AST
/// and add them as completion items.
fn add_user_defined_items(items: &mut Vec<CompletionItem>, program: &tenth::parser::ast::Program) {
    let mut seen_funcs: HashSet<String> = HashSet::new();
    let mut seen_structs: HashSet<String> = HashSet::new();
    let mut seen_enums: HashSet<String> = HashSet::new();

    for item in &program.items {
        match &item.kind {
            ItemKind::Function { name, .. } => {
                if seen_funcs.insert(name.name.clone()) {
                    items.push(CompletionItem {
                        label: name.name.clone(),
                        kind: CompletionItemKind::Function,
                        detail: Some(format!("fn {}", name.name)),
                        documentation: None,
                        insert_text: None,
                    });
                }
            }
            ItemKind::StructDef { name, .. } => {
                if seen_structs.insert(name.name.clone()) {
                    items.push(CompletionItem {
                        label: name.name.clone(),
                        kind: CompletionItemKind::Class,
                        detail: Some(format!("struct {}", name.name)),
                        documentation: None,
                        insert_text: None,
                    });
                }
            }
            ItemKind::EnumDef { name, .. } => {
                if seen_enums.insert(name.name.clone()) {
                    items.push(CompletionItem {
                        label: name.name.clone(),
                        kind: CompletionItemKind::Class,
                        detail: Some(format!("enum {}", name.name)),
                        documentation: None,
                        insert_text: None,
                    });
                }
            }
            ItemKind::Impl { functions, .. } => {
                for func in functions {
                    if let ItemKind::Function { name, .. } = &func.kind {
                        // Impl methods are added as Function (not Method) in
                        // top-level completions; they appear as Method only in
                        // dot-completion context.
                        if seen_funcs.insert(name.name.clone()) {
                            items.push(CompletionItem {
                                label: name.name.clone(),
                                kind: CompletionItemKind::Function,
                                detail: Some(format!("fn {}", name.name)),
                                documentation: None,
                                insert_text: None,
                            });
                        }
                    }
                }
            }
            _ => {}
        }
    }
}

fn add_keywords(items: &mut Vec<CompletionItem>) {
    let keywords = [
        "fn", "let", "mut", "if", "else", "while", "for", "in", "return",
        "struct", "enum", "impl", "trait", "import", "tensor", "true", "false",
        "match", "mod", "use", "pub", "const", "ref", "move", "loop",
        "break", "continue",
    ];
    for kw in &keywords {
        items.push(CompletionItem {
            label: kw.to_string(),
            kind: CompletionItemKind::Keyword,
            detail: Some(format!("keyword {}", kw)),
            documentation: None,
            insert_text: None,
        });
    }
}

fn add_builtins(items: &mut Vec<CompletionItem>) {
    let builtins = [
        ("print", "Print a value to stdout"),
        ("println", "Print a value to stdout with newline"),
        ("param", "Declare a learnable parameter"),
        ("grad", "Compute gradient of a tensor"),
        ("stop_grad", "Stop gradient propagation"),
        ("new_grad", "Create a new gradient context"),
        ("backward", "Run backward pass for autodiff"),
        ("tensor", "Create a tensor from data"),
        ("zeros", "Create a tensor of zeros"),
        ("ones", "Create a tensor of ones"),
        ("randn", "Create a tensor with random normal values"),
        ("range", "Create an iterator range"),
        ("len", "Return the length of a collection"),
        ("shape", "Return the shape of a tensor"),
        ("reshape", "Reshape a tensor"),
        ("read_file", "Read a file as a string"),
        ("write_file", "Write a string to a file"),
        ("read_bytes", "Read a file as raw bytes"),
        ("time_now", "Get current timestamp"),
        ("random_int", "Generate a random integer"),
        ("random_float", "Generate a random float"),
        ("json_encode", "Encode a value as JSON"),
        ("json_decode", "Decode a JSON string"),
        ("abs", "Absolute value"),
        ("sqrt", "Square root"),
        ("sin", "Sine function"),
        ("cos", "Cosine function"),
        ("tan", "Tangent function"),
        ("exp", "Exponential function"),
        ("ln", "Natural logarithm"),
        ("ceil", "Round up to nearest integer"),
        ("floor", "Round down to nearest integer"),
        ("pow", "Power function"),
        ("max", "Maximum value"),
        ("min", "Minimum value"),
        ("argmax", "Index of maximum value"),
    ];
    for (name, doc) in &builtins {
        items.push(CompletionItem {
            label: name.to_string(),
            kind: CompletionItemKind::Function,
            detail: Some(doc.to_string()),
            documentation: None,
            insert_text: None,
        });
    }
}

fn add_types(items: &mut Vec<CompletionItem>) {
    let types = [
        ("i64", "64-bit signed integer"),
        ("f64", "64-bit floating point"),
        ("bool", "Boolean type"),
        ("String", "UTF-8 string type"),
        ("Tensor", "N-dimensional tensor"),
        ("Vec", "Dynamic array"),
        ("Option", "Optional value"),
        ("Result", "Result type"),
    ];
    for (name, doc) in &types {
        items.push(CompletionItem {
            label: name.to_string(),
            kind: CompletionItemKind::Class,
            detail: Some(doc.to_string()),
            documentation: None,
            insert_text: None,
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_after_dot_true() {
        // 光标在 "x." 之后（character=2，正好在 '.' 之后的位置）
        let src = "x.";
        let pos = Position { line: 0, character: 2 };
        assert!(
            is_after_dot(src, pos),
            "expected is_after_dot=true for cursor right after '.'"
        );
    }

    #[test]
    fn test_is_after_dot_false_in_identifier() {
        // 光标在标识符中间，前面没有 '.'
        let src = "let variable = 0;";
        // 光标在 character 5（"varia|ble" 中）
        let pos = Position { line: 0, character: 5 };
        assert!(
            !is_after_dot(src, pos),
            "expected is_after_dot=false when cursor is mid-identifier"
        );
    }

    #[test]
    fn test_is_after_dot_false_at_line_start() {
        // 光标在行首（character=0），前面没有 '.'
        let src = "fn main() -> i32 { 0 }";
        let pos = Position { line: 0, character: 0 };
        assert!(
            !is_after_dot(src, pos),
            "expected is_after_dot=false at line start"
        );
    }

    #[test]
    fn test_is_after_dot_false_for_line_out_of_range() {
        // 行号超出范围应返回 false
        let src = "let x = 0;";
        let pos = Position { line: 100, character: 0 };
        assert!(
            !is_after_dot(src, pos),
            "expected is_after_dot=false when line out of range"
        );
    }

    #[test]
    fn test_is_after_dot_true_with_more_text() {
        // "obj.method" 中光标正好在 '.' 之后（character=4，即 "obj.|method"）
        let src = "obj.method";
        let pos = Position { line: 0, character: 4 };
        assert!(
            is_after_dot(src, pos),
            "expected is_after_dot=true when cursor is right after '.'"
        );
    }

    #[test]
    fn test_full_completion_items_contains_keywords() {
        // 完整补全列表应包含关键字
        let items = full_completion_items("");
        let labels: Vec<&str> = items.iter().map(|i| i.label.as_str()).collect();
        // 至少应包含 "fn" 这个关键字
        assert!(
            labels.contains(&"fn"),
            "expected completions to contain 'fn' keyword, got: {:?}",
            labels.iter().take(10).collect::<Vec<_>>()
        );
        // 应包含 "let" 关键字
        assert!(
            labels.contains(&"let"),
            "expected completions to contain 'let' keyword"
        );
    }
}
