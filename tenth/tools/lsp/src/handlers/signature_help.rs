use tenth::lexer::lexer::Lexer;
use tenth::parser::ast::ItemKind;
use tenth::parser::parser::Parser;

use super::Handler;
use crate::lsp_types::*;

pub struct SignatureHelpHandler;

impl Handler for SignatureHelpHandler {
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

        let source = crate::document_store::get_content_or_disk_global(uri)
            .unwrap_or_default();

        let help = compute_signature_help(&source, pos);
        serde_json::to_value(help).unwrap_or(serde_json::Value::Null)
    }
}

fn compute_signature_help(source: &str, position: Position) -> Option<SignatureHelp> {
    // Find the function name being called by scanning backwards for `name(`
    let lines: Vec<&str> = source.lines().collect();
    let line_idx = position.line as usize;
    if line_idx >= lines.len() {
        return None;
    }

    // Collect text up to the cursor position
    let mut text_up_to_cursor = String::new();
    for (i, line) in lines.iter().enumerate() {
        if i < line_idx {
            text_up_to_cursor.push_str(line);
            text_up_to_cursor.push('\n');
        } else if i == line_idx {
            let char_idx = (position.character as usize).min(line.len());
            text_up_to_cursor.push_str(&line[..char_idx]);
        }
    }

    // Find the last unmatched `(` and the function name before it
    let (func_name, active_param) = find_call_context(&text_up_to_cursor)?;
    if func_name.is_empty() {
        return None;
    }

    // Parse the source to find the function definition
    let mut lexer = Lexer::new(source);
    let tokens = lexer.tokenize().ok()?;
    let mut parser = Parser::new(tokens);
    let (program, _errors) = parser.parse_program_with_recovery();

    // Search for the function definition
    for item in &program.items {
        if let ItemKind::Function { name, params, return_type, .. } = &item.kind {
            if name.name == func_name {
                let param_labels: Vec<ParameterInformation> = params
                    .iter()
                    .map(|p| ParameterInformation {
                        label: format!("{}: {}", p.name.name, type_ann_str(&p.type_ann)),
                    })
                    .collect();

                let ret_str = return_type
                    .as_ref()
                    .map(|t| type_ann_str(t))
                    .unwrap_or_else(|| "()".to_string());

                let sig_label = format!(
                    "{}({}) -> {}",
                    func_name,
                    param_labels.iter().map(|p| p.label.clone()).collect::<Vec<_>>().join(", "),
                    ret_str
                );

                return Some(SignatureHelp {
                    signatures: vec![SignatureInformation {
                        label: sig_label,
                        parameters: param_labels,
                    }],
                    active_signature: Some(0),
                    active_parameter: Some(active_param as u32),
                });
            }
        }
    }

    // Also search in impl blocks
    for item in &program.items {
        if let ItemKind::Impl { functions, .. } = &item.kind {
            for func in functions {
                if let ItemKind::Function { name, params, return_type, .. } = &func.kind {
                    if name.name == func_name {
                        let param_labels: Vec<ParameterInformation> = params
                            .iter()
                            .map(|p| ParameterInformation {
                                label: format!("{}: {}", p.name.name, type_ann_str(&p.type_ann)),
                            })
                            .collect();

                        let ret_str = return_type
                            .as_ref()
                            .map(|t| type_ann_str(t))
                            .unwrap_or_else(|| "()".to_string());

                        let sig_label = format!(
                            "{}({}) -> {}",
                            func_name,
                            param_labels.iter().map(|p| p.label.clone()).collect::<Vec<_>>().join(", "),
                            ret_str
                        );

                        return Some(SignatureHelp {
                            signatures: vec![SignatureInformation {
                                label: sig_label,
                                parameters: param_labels,
                            }],
                            active_signature: Some(0),
                            active_parameter: Some(active_param as u32),
                        });
                    }
                }
            }
        }
    }

    None
}

/// Find the function call context: the function name and the active parameter index.
/// Scans backwards from the cursor to find the last unmatched `(`.
fn find_call_context(text: &str) -> Option<(String, usize)> {
    let chars: Vec<char> = text.chars().collect();
    let mut i = chars.len();

    // Skip trailing whitespace
    while i > 0 && chars[i - 1].is_whitespace() {
        i -= 1;
    }

    // Scan backwards to find the matching `(`, tracking nesting
    let mut depth = 0;
    let mut param_count = 0;

    while i > 0 {
        i -= 1;
        let c = chars[i];

        if c == ')' {
            depth += 1;
        } else if c == '(' {
            if depth == 0 {
                // Found the opening paren — now extract the function name before it
                let mut j = i;
                while j > 0 && chars[j - 1].is_whitespace() {
                    j -= 1;
                }

                // Scan backwards for the identifier
                let end = j;
                let mut start = j;
                while start > 0 {
                    let prev = chars[start - 1];
                    if prev.is_alphanumeric() || prev == '_' {
                        start -= 1;
                    } else {
                        break;
                    }
                }

                if start < end {
                    let name: String = chars[start..end].iter().collect();
                    return Some((name, param_count));
                }
                return None;
            }
            depth -= 1;
        } else if depth == 0 && c == ',' {
            param_count += 1;
        }
    }

    None
}

fn type_ann_str(t: &tenth::parser::ast::TypeAnnotation) -> String {
    use tenth::parser::ast::TypeAnnotation;
    match t {
        TypeAnnotation::Named(ident) => ident.name.clone(),
        TypeAnnotation::Generic { base, args } => {
            let args_str = args.iter().map(type_ann_str).collect::<Vec<_>>().join(", ");
            format!("{}<{}>", base.name, args_str)
        }
        TypeAnnotation::Tensor { dtype, dims } => {
            let dtype_str = type_ann_str(dtype);
            let dims_str = dims.iter().map(|d| match d {
                tenth::parser::ast::DimSpec::Literal(n) => n.to_string(),
                tenth::parser::ast::DimSpec::Symbol(s) => s.clone(),
                tenth::parser::ast::DimSpec::Wildcard => "_".to_string(),
            }).collect::<Vec<_>>().join(", ");
            format!("Tensor[{}, {}]", dtype_str, dims_str)
        }
        TypeAnnotation::Array(inner) => format!("[{}]", type_ann_str(inner)),
        TypeAnnotation::FnType { params, ret } => {
            let params_str = params.iter().map(type_ann_str).collect::<Vec<_>>().join(", ");
            format!("fn({}) -> {}", params_str, type_ann_str(ret))
        }
        TypeAnnotation::Unit => "()".to_string(),
    }
}
