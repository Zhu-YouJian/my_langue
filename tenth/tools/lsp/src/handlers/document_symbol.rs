use tenth::lexer::lexer::Lexer;
use tenth::parser::ast::{ItemKind, StructKind};
use tenth::parser::parser::Parser;

use super::Handler;
use crate::lsp_types::*;

pub struct DocumentSymbolHandler;

impl Handler for DocumentSymbolHandler {
    fn handle(&self, params: Option<&serde_json::Value>) -> serde_json::Value {
        let uri = params
            .and_then(|p| p.get("textDocument"))
            .and_then(|td| td.get("uri"))
            .and_then(|u| u.as_str())
            .unwrap_or("");

        let source = crate::document_store::get_content_or_disk_global(uri)
            .unwrap_or_default();

        let symbols = extract_symbols(&source);
        serde_json::to_value(symbols).unwrap()
    }
}

fn extract_symbols(source: &str) -> Vec<DocumentSymbol> {
    let mut lexer = Lexer::new(source);
    let tokens = match lexer.tokenize() {
        Ok(t) => t,
        Err(_) => return Vec::new(),
    };

    let mut parser = Parser::new(tokens);
    let (program, _errors) = parser.parse_program_with_recovery();

    let mut symbols = Vec::new();

    for item in &program.items {
        if let Some(sym) = item_to_symbol(item) {
            symbols.push(sym);
        }
    }

    symbols
}

fn item_to_symbol(item: &tenth::parser::ast::Item) -> Option<DocumentSymbol> {
    let span = &item.span;
    let line = span.line.saturating_sub(1) as u32;
    // item.span 是 item 起始关键字（identifier-like，lexer col 2-based）
    let col = span.col.saturating_sub(2) as u32;

    let range = Range {
        start: Position { line, character: col },
        end: Position { line, character: col + 200 }, // Approximate end
    };
    let selection_range = range.clone();

    match &item.kind {
        ItemKind::Function { name, params, return_type, .. } => {
            let detail = format!(
                "({}) -> {}",
                params.iter()
                    .map(|p| format!("{}: {}", p.name.name, type_ann_str(&p.type_ann)))
                    .collect::<Vec<_>>()
                    .join(", "),
                return_type.as_ref().map(|t| type_ann_str(t)).unwrap_or_else(|| "()".to_string())
            );
            Some(DocumentSymbol {
                name: name.name.clone(),
                kind: SymbolKind::Function,
                range,
                selection_range,
                detail: Some(detail),
                children: Vec::new(),
            })
        }
        ItemKind::StructDef { name, kind, .. } => {
            let children: Vec<DocumentSymbol> = match kind {
                StructKind::Named(fields) => fields.iter().map(|f| {
                    let fline = f.name.span.line.saturating_sub(1) as u32;
                    let fcol = f.name.span.col.saturating_sub(2) as u32;
                    DocumentSymbol {
                        name: f.name.name.clone(),
                        kind: SymbolKind::Field,
                        range: Range {
                            start: Position { line: fline, character: fcol },
                            end: Position { line: fline, character: fcol + f.name.name.len() as u32 },
                        },
                        selection_range: Range {
                            start: Position { line: fline, character: fcol },
                            end: Position { line: fline, character: fcol + f.name.name.len() as u32 },
                        },
                        detail: Some(type_ann_str(&f.type_ann)),
                        children: Vec::new(),
                    }
                }).collect(),
                StructKind::Tuple(types) => types.iter().enumerate().map(|(i, ty)| {
                    DocumentSymbol {
                        name: format!("_{}", i),
                        kind: SymbolKind::Field,
                        range: range.clone(),
                        selection_range: selection_range.clone(),
                        detail: Some(type_ann_str(ty)),
                        children: Vec::new(),
                    }
                }).collect(),
            };
            let field_count = match kind {
                StructKind::Named(fields) => fields.len(),
                StructKind::Tuple(types) => types.len(),
            };
            Some(DocumentSymbol {
                name: name.name.clone(),
                kind: SymbolKind::Struct,
                range,
                selection_range,
                detail: Some(format!("{} fields", field_count)),
                children,
            })
        }
        ItemKind::EnumDef { name, variants, .. } => {
            let children: Vec<DocumentSymbol> = variants.iter().map(|v| {
                let vline = v.name.span.line.saturating_sub(1) as u32;
                let vcol = v.name.span.col.saturating_sub(2) as u32;
                DocumentSymbol {
                    name: v.name.name.clone(),
                    kind: SymbolKind::EnumMember,
                    range: Range {
                        start: Position { line: vline, character: vcol },
                        end: Position { line: vline, character: vcol + v.name.name.len() as u32 },
                    },
                    selection_range: Range {
                        start: Position { line: vline, character: vcol },
                        end: Position { line: vline, character: vcol + v.name.name.len() as u32 },
                    },
                    detail: None,
                    children: Vec::new(),
                }
            }).collect();
            Some(DocumentSymbol {
                name: name.name.clone(),
                kind: SymbolKind::Enum,
                range,
                selection_range,
                detail: Some(format!("{} variants", variants.len())),
                children,
            })
        }
        ItemKind::Trait { name, methods, .. } => {
            let children: Vec<DocumentSymbol> = methods.iter().map(|m| {
                let mline = m.name.span.line.saturating_sub(1) as u32;
                let mcol = m.name.span.col.saturating_sub(2) as u32;
                DocumentSymbol {
                    name: m.name.name.clone(),
                    kind: SymbolKind::Method,
                    range: Range {
                        start: Position { line: mline, character: mcol },
                        end: Position { line: mline, character: mcol + m.name.name.len() as u32 },
                    },
                    selection_range: Range {
                        start: Position { line: mline, character: mcol },
                        end: Position { line: mline, character: mcol + m.name.name.len() as u32 },
                    },
                    detail: Some(format!(
                        "({}) -> {}",
                        m.params.iter()
                            .map(|p| p.name.name.clone())
                            .collect::<Vec<_>>()
                            .join(", "),
                        m.return_type.as_ref().map(|t| type_ann_str(t)).unwrap_or_else(|| "()".to_string())
                    )),
                    children: Vec::new(),
                }
            }).collect();
            Some(DocumentSymbol {
                name: name.name.clone(),
                kind: SymbolKind::Interface,
                range,
                selection_range,
                detail: Some(format!("{} methods", methods.len())),
                children,
            })
        }
        ItemKind::Impl { type_name, functions, .. } => {
            let children: Vec<DocumentSymbol> = functions.iter().filter_map(|f| {
                if let ItemKind::Function { name, params, return_type, .. } = &f.kind {
                    let fline = name.span.line.saturating_sub(1) as u32;
                    let fcol = name.span.col.saturating_sub(2) as u32;
                    Some(DocumentSymbol {
                        name: name.name.clone(),
                        kind: SymbolKind::Method,
                        range: Range {
                            start: Position { line: fline, character: fcol },
                            end: Position { line: fline, character: fcol + name.name.len() as u32 },
                        },
                        selection_range: Range {
                            start: Position { line: fline, character: fcol },
                            end: Position { line: fline, character: fcol + name.name.len() as u32 },
                        },
                        detail: Some(format!(
                            "({}) -> {}",
                            params.iter()
                                .map(|p| p.name.name.clone())
                                .collect::<Vec<_>>()
                                .join(", "),
                            return_type.as_ref().map(|t| type_ann_str(t)).unwrap_or_else(|| "()".to_string())
                        )),
                        children: Vec::new(),
                    })
                } else {
                    None
                }
            }).collect();
            Some(DocumentSymbol {
                name: format!("impl {}", type_name.name),
                kind: SymbolKind::Namespace,
                range,
                selection_range,
                detail: Some(format!("{} methods", functions.len())),
                children,
            })
        }
        ItemKind::Const { name, type_ann, .. } => {
            Some(DocumentSymbol {
                name: name.name.clone(),
                kind: SymbolKind::Constant,
                range,
                selection_range,
                detail: Some(type_ann_str(type_ann)),
                children: Vec::new(),
            })
        }
        ItemKind::Mod { name, items, .. } => {
            let children: Vec<DocumentSymbol> = items.iter().filter_map(item_to_symbol).collect();
            Some(DocumentSymbol {
                name: format!("mod {}", name.name),
                kind: SymbolKind::Module,
                range,
                selection_range,
                detail: Some(format!("{} items", items.len())),
                children,
            })
        }
        ItemKind::Union { name, fields, .. } => {
            Some(DocumentSymbol {
                name: name.name.clone(),
                kind: SymbolKind::Struct,
                range,
                selection_range,
                detail: Some(format!("union {} fields", fields.len())),
                children: Vec::new(),
            })
        }
        ItemKind::Operator { op, .. } => {
            Some(DocumentSymbol {
                name: format!("operator {}", op),
                kind: SymbolKind::Function,
                range,
                selection_range,
                detail: Some("custom operator".to_string()),
                children: Vec::new(),
            })
        }
        ItemKind::MacroDef { name, .. } => {
            Some(DocumentSymbol {
                name: name.name.clone(),
                kind: SymbolKind::Function,
                range,
                selection_range,
                detail: Some("macro".to_string()),
                children: Vec::new(),
            })
        }
        ItemKind::Use { .. } => None,
    }
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
        TypeAnnotation::Array { inner, .. } => format!("[{}]", type_ann_str(inner)),
        TypeAnnotation::Ref { inner, mutable, .. } => {
            let prefix = if *mutable { "&mut " } else { "&" };
            format!("{}{}", prefix, type_ann_str(inner))
        }
        TypeAnnotation::FnType { params, ret } => {
            let params_str = params.iter().map(type_ann_str).collect::<Vec<_>>().join(", ");
            format!("fn({}) -> {}", params_str, type_ann_str(ret))
        }
        TypeAnnotation::Unit => "()".to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_symbols_function_definition() {
        // 含 fn 定义的代码应提取出 Function 类型的符号
        let src = "fn add(a: i32, b: i32) -> i32 { a + b }";
        let symbols = extract_symbols(src);
        assert_eq!(symbols.len(), 1, "expected 1 symbol, got {}", symbols.len());
        let s = &symbols[0];
        assert_eq!(s.name, "add", "expected name 'add', got {}", s.name);
        assert!(
            matches!(s.kind, SymbolKind::Function),
            "expected Function kind, got {:?}",
            s.kind
        );
        // detail 应包含参数与返回类型信息
        let detail = s.detail.as_deref().unwrap_or("");
        assert!(
            detail.contains("a") && detail.contains("b"),
            "expected detail to contain param names, got: {}",
            detail
        );
        assert!(
            detail.contains("i32"),
            "expected detail to contain return type i32, got: {}",
            detail
        );
    }

    #[test]
    fn test_extract_symbols_struct_definition() {
        // 含 struct 定义的代码应提取出 Struct 类型的符号，且子字段为 children
        let src = "struct Point { x: f64, y: f64 }";
        let symbols = extract_symbols(src);
        assert_eq!(symbols.len(), 1, "expected 1 symbol, got {}", symbols.len());
        let s = &symbols[0];
        assert_eq!(s.name, "Point");
        assert!(
            matches!(s.kind, SymbolKind::Struct),
            "expected Struct kind, got {:?}",
            s.kind
        );
        // 应有 2 个字段作为 children
        assert_eq!(
            s.children.len(),
            2,
            "expected 2 child fields, got {}",
            s.children.len()
        );
        // 子字段名应为 x 和 y
        let child_names: Vec<&str> = s.children.iter().map(|c| c.name.as_str()).collect();
        assert!(child_names.contains(&"x"), "expected field 'x', got {:?}", child_names);
        assert!(child_names.contains(&"y"), "expected field 'y', got {:?}", child_names);
    }

    #[test]
    fn test_extract_symbols_empty_input() {
        // 空输入不应产生任何符号
        let symbols = extract_symbols("");
        assert!(symbols.is_empty(), "expected no symbols for empty input");
    }

    #[test]
    fn test_extract_symbols_multiple_items() {
        // 多个顶层 item 应都被提取
        let src = "fn a() -> i32 { 1 }\nfn b() -> i32 { 2 }";
        let symbols = extract_symbols(src);
        assert_eq!(symbols.len(), 2, "expected 2 symbols, got {}", symbols.len());
        let names: Vec<&str> = symbols.iter().map(|s| s.name.as_str()).collect();
        assert!(names.contains(&"a"), "expected function 'a', got {:?}", names);
        assert!(names.contains(&"b"), "expected function 'b', got {:?}", names);
    }

    #[test]
    fn test_extract_symbols_lex_error_returns_empty() {
        // lexer 错误时应返回空 Vec，不 panic
        let src = "fn bad() -> i32 { \"unclosed }";
        let symbols = extract_symbols(src);
        assert!(symbols.is_empty(), "expected empty symbols on lex error");
    }
}
