use super::Handler;
use crate::lsp_types::*;
use tenth::lexer::lexer::Lexer;
use tenth::lexer::token::{Token, TokenKind};
use tenth::hir::lower::Lowerer;
use tenth::parser::parser::Parser;

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

        let pos = match position {
            Some(p) => p,
            None => return serde_json::Value::Null,
        };

        // Read the source file (from document store or disk)
        let source = match crate::document_store::get_content_or_disk_global(uri) {
            Some(s) => s,
            None => return serde_json::Value::Null,
        };

        // Tokenize
        let tokens = match Lexer::new(&source).tokenize() {
            Ok(t) => t,
            Err(_) => return serde_json::Value::Null,
        };

        // Find the token at the cursor position
        let target_line = (pos.line + 1) as usize; // LSP lines are 0-based, lexer is 1-based
        let target_col = pos.character as usize; // LSP chars are 0-based

        let token = find_token_at_position(&tokens, target_line, target_col);

        // Get the text of the token for lookup
        let token_text = match &token {
            Some(t) => token_to_text(t),
            None => return serde_json::Value::Null,
        };

        if token_text.is_empty() {
            return serde_json::Value::Null;
        }

        // First, try the keyword/builtin/type lookup table
        if let Some(hover) = lookup_hover(&token_text) {
            return serde_json::to_value(&hover).unwrap_or(serde_json::Value::Null);
        }

        // For identifiers, try HIR-based type lookup
        if let Some(tok) = &token {
            if matches!(tok.kind, TokenKind::Identifier(_)) {
                if let Some(hover) = lookup_hir_type(&source, &token_text) {
                    return serde_json::to_value(&hover).unwrap_or(serde_json::Value::Null);
                }
            }
        }

        serde_json::Value::Null
    }
}

/// Convert a URI string to a file system path.
/// Strips "file://" prefix and handles URL-encoded characters.
fn uri_to_path(uri: &str) -> String {
    let path = if let Some(stripped) = uri.strip_prefix("file:///") {
        stripped.to_string()
    } else if let Some(stripped) = uri.strip_prefix("file://") {
        stripped.to_string()
    } else {
        uri.to_string()
    };

    // On Windows, forward slashes in file URIs need to be converted
    let path = path.replace('/', "\\");

    // Decode percent-encoded characters
    percent_decode(&path)
}

/// Simple percent-decode for common URL-encoded characters.
fn percent_decode(s: &str) -> String {
    let mut result = String::with_capacity(s.len());
    let mut chars = s.chars();
    while let Some(c) = chars.next() {
        if c == '%' {
            let hex: String = chars.by_ref().take(2).collect();
            if let Ok(byte) = u8::from_str_radix(&hex, 16) {
                result.push(byte as char);
            } else {
                result.push('%');
                result.push_str(&hex);
            }
        } else {
            result.push(c);
        }
    }
    result
}

/// Find the token at the given line and column position.
/// `line` is 1-based (lexer), `col0` is 0-based (LSP).
fn find_token_at_position(tokens: &[Token], line: usize, col0: usize) -> Option<Token> {
    for token in tokens {
        if token.kind == TokenKind::Eof {
            continue;
        }

        if token.span.line != line {
            continue;
        }

        let token_text = token_to_text(token);
        // Lexer span.col 对标识符/关键字/数字 token 是 2-based；归一化为 0-based。
        let start0 = crate::span::token_start_col0(token);
        let end0 = start0 + token_text.len();

        // 光标落在 [start, end] 内（end 含，允许光标紧贴 token 末尾）
        if col0 >= start0 && col0 <= end0 {
            return Some(token.clone());
        }
    }
    None
}

/// Get the display text of a token.
fn token_to_text(token: &Token) -> String {
    match &token.kind {
        TokenKind::Identifier(s) => s.clone(),
        TokenKind::IntLiteral(n, _) => n.to_string(),
        TokenKind::FloatLiteral(n, _) => n.to_string(),
        TokenKind::StringLiteral(s) => s.clone(),
        TokenKind::CharLiteral(c) => c.to_string(),
        TokenKind::InterpolatedString(parts) => {
            let mut s = String::new();
            for p in parts {
                match p {
                    tenth::lexer::token::StringPart::Literal(l) => s.push_str(l),
                    tenth::lexer::token::StringPart::Expr(e) => {
                        s.push('{');
                        s.push_str(e);
                        s.push('}');
                    }
                }
            }
            s
        }
        _ => token.kind.to_string(),
    }
}

/// Try to look up type information from the HIR for a given identifier.
fn lookup_hir_type(source: &str, name: &str) -> Option<Hover> {
    // Parse the source
    let tokens = Lexer::new(source).tokenize().ok()?;
    let mut parser = Parser::new(tokens);
    let program = parser.parse_program().ok()?;

    // Lower to HIR
    let mut lowerer = Lowerer::new();
    let hir_program = lowerer.lower_program(&program).ok()?;

    // Search through functions
    for fn_def in &hir_program.functions {
        if fn_def.name == name {
            let params: Vec<String> = fn_def.params.iter()
                .map(|(n, t)| format!("{}: {}", n, t))
                .collect();
            let sig = format!(
                "fn {}({}) -> {}",
                fn_def.name,
                params.join(", "),
                fn_def.return_type
            );
            return Some(Hover {
                contents: MarkupContent {
                    kind: "plaintext".to_string(),
                    value: sig,
                },
                range: None,
            });
        }
    }

    // Search through generic functions
    for fn_def in &hir_program.generic_funcs {
        if fn_def.name == name {
            let generics = if fn_def.generics.is_empty() {
                String::new()
            } else {
                format!("<{}>", fn_def.generics.join(", "))
            };
            let params: Vec<String> = fn_def.params.iter()
                .map(|(n, t)| format!("{}: {}", n, t))
                .collect();
            let sig = format!(
                "fn {}{}({}) -> {}",
                fn_def.name,
                generics,
                params.join(", "),
                fn_def.return_type
            );
            return Some(Hover {
                contents: MarkupContent {
                    kind: "plaintext".to_string(),
                    value: sig,
                },
                range: None,
            });
        }
    }

    // Search through structs
    if let Some(fields) = hir_program.structs.get(name) {
        let field_strs: Vec<String> = fields.iter()
            .map(|(n, t)| format!("  {}: {}", n, t))
            .collect();
        let sig = format!("struct {} {{\n{}\n}}", name, field_strs.join(",\n"));
        return Some(Hover {
            contents: MarkupContent {
                kind: "plaintext".to_string(),
                value: sig,
            },
            range: None,
        });
    }

    // Search through generic structs
    if let Some(gs) = hir_program.generic_structs.get(name) {
        let generics = if gs.generics.is_empty() {
            String::new()
        } else {
            format!("<{}>", gs.generics.join(", "))
        };
        let field_strs: Vec<String> = gs.fields.iter()
            .map(|(n, t)| format!("  {}: {}", n, t))
            .collect();
        let sig = format!("struct {}{} {{\n{}\n}}", name, generics, field_strs.join(",\n"));
        return Some(Hover {
            contents: MarkupContent {
                kind: "plaintext".to_string(),
                value: sig,
            },
            range: None,
        });
    }

    // Search through enums
    if let Some(variants) = hir_program.enums.get(name) {
        let variant_strs: Vec<String> = variants.iter()
            .map(|(v, fields)| {
                if fields.is_empty() {
                    format!("  {}", v)
                } else {
                    let field_strs: Vec<String> = fields.iter()
                        .map(|(n, t)| format!("{}: {}", n, t))
                        .collect();
                    format!("  {}({})", v, field_strs.join(", "))
                }
            })
            .collect();
        let sig = format!("enum {} {{\n{}\n}}", name, variant_strs.join(",\n"));
        return Some(Hover {
            contents: MarkupContent {
                kind: "plaintext".to_string(),
                value: sig,
            },
            range: None,
        });
    }

    // Search through trait definitions
    if let Some(trait_def) = hir_program.trait_defs.get(name) {
        let method_strs: Vec<String> = trait_def.methods.iter()
            .map(|m| {
                let params: Vec<String> = m.params.iter()
                    .map(|(n, t)| format!("{}: {}", n, t))
                    .collect();
                format!("  fn {}({}) -> {}", m.name, params.join(", "), m.return_type)
            })
            .collect();
        let sig = format!("trait {} {{\n{}\n}}", name, method_strs.join(",\n"));
        return Some(Hover {
            contents: MarkupContent {
                kind: "plaintext".to_string(),
                value: sig,
            },
            range: None,
        });
    }

    // Search through methods on types (impl blocks)
    for (type_name, method_map) in &hir_program.methods {
        if let Some(fn_def) = method_map.get(name) {
            let params: Vec<String> = fn_def.params.iter()
                .map(|(n, t)| format!("{}: {}", n, t))
                .collect();
            let sig = format!(
                "impl {} {{ fn {}({}) -> {} }}",
                type_name,
                fn_def.name,
                params.join(", "),
                fn_def.return_type
            );
            return Some(Hover {
                contents: MarkupContent {
                    kind: "plaintext".to_string(),
                    value: sig,
                },
                range: None,
            });
        }
    }

    None
}

/// Lookup documentation for a known symbol name.
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
        // Additional keywords from the lexer
        ("match", "Keyword: Pattern matching expression"),
        ("loop", "Keyword: Infinite loop"),
        ("break", "Keyword: Break out of a loop"),
        ("continue", "Keyword: Skip to next loop iteration"),
        ("try", "Keyword: Try block for error handling"),
        ("use", "Keyword: Import module items"),
        ("mod", "Keyword: Define a module"),
        ("pub", "Keyword: Public visibility modifier"),
        ("type", "Keyword: Type alias definition"),
        ("self", "Keyword: Self reference in method"),
        ("spawn", "Keyword: Spawn a concurrent task"),
        ("task", "Keyword: Define an async task"),
        ("shard", "Keyword: Define a distributed shard"),
        ("node", "Keyword: Define a compute node"),
        ("macro", "Keyword: Define a macro"),
        ("where", "Keyword: Constraint clause"),
        ("as", "Keyword: Type cast or alias"),
        ("move", "Keyword: Move ownership"),
        // Additional builtins
        ("println", "Builtin: Print a value with newline"),
        ("eprintln", "Builtin: Print to stderr with newline"),
        ("rand", "Builtin: Create a tensor with random uniform values"),
        ("abs", "Builtin: Absolute value"),
        ("sqrt", "Builtin: Square root"),
        ("sin", "Builtin: Sine function"),
        ("cos", "Builtin: Cosine function"),
        ("ln", "Builtin: Natural logarithm"),
        ("pow", "Builtin: Power function"),
        // Additional types
        ("i8", "Type: 8-bit signed integer"),
        ("i16", "Type: 16-bit signed integer"),
        ("i32", "Type: 32-bit signed integer"),
        ("u8", "Type: 8-bit unsigned integer"),
        ("u16", "Type: 16-bit unsigned integer"),
        ("u32", "Type: 32-bit unsigned integer"),
        ("u64", "Type: 64-bit unsigned integer"),
        ("f16", "Type: 16-bit floating point"),
        ("f32", "Type: 32-bit floating point"),
        ("bf16", "Type: BFloat16"),
        ("char", "Type: Unicode character"),
        ("Option", "Type: Optional value (Some | None)"),
        ("Result", "Type: Result value (Ok | Err)"),
    ];

    docs.iter()
        .find(|(k, _)| *k == name)
        .map(|(_, doc)| Hover {
            contents: MarkupContent {
                kind: "plaintext".to_string(),
                value: doc.to_string(),
            },
            range: None,
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lookup_hover_known_keyword_fn() {
        // 查 "fn" 关键字应返回 Some(Hover)，且内容包含描述
        let hover = lookup_hover("fn");
        assert!(hover.is_some(), "expected Some(hover) for 'fn' keyword");
        let h = hover.unwrap();
        assert_eq!(h.contents.kind, "plaintext");
        assert!(
            !h.contents.value.is_empty(),
            "hover doc for 'fn' should not be empty"
        );
        // 描述应包含 "function" 或 "函数" 字样
        assert!(
            h.contents.value.to_lowercase().contains("function")
                || h.contents.value.contains("函数"),
            "hover for 'fn' should mention function, got: {}",
            h.contents.value
        );
    }

    #[test]
    fn test_lookup_hover_unknown_keyword_returns_none() {
        // 查不存在的符号应返回 None
        let hover = lookup_hover("definitely_not_a_keyword_xyz123");
        assert!(hover.is_none(), "expected None for unknown keyword");
    }

    #[test]
    fn test_lookup_hover_struct_keyword() {
        // "struct" 关键字应返回 Some
        let hover = lookup_hover("struct");
        assert!(hover.is_some(), "expected Some(hover) for 'struct' keyword");
        let h = hover.unwrap();
        assert!(
            h.contents.value.to_lowercase().contains("struct")
                || h.contents.value.contains("结构"),
            "hover for 'struct' should mention struct, got: {}",
            h.contents.value
        );
    }

    #[test]
    fn test_lookup_hover_builtin_print() {
        // 内建函数 "print" 应返回 Some
        let hover = lookup_hover("print");
        assert!(hover.is_some(), "expected Some(hover) for 'print' builtin");
    }

    #[test]
    fn test_lookup_hover_type_i32() {
        // 类型 "i32" 应返回 Some
        let hover = lookup_hover("i32");
        assert!(hover.is_some(), "expected Some(hover) for 'i32' type");
    }
}
