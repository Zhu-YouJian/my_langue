use tenth::lexer::lexer::Lexer;
use tenth::parser::ast::ItemKind;
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
    let col = span.col.saturating_sub(1) as u32;

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
        ItemKind::StructDef { name, fields, .. } => {
            let children: Vec<DocumentSymbol> = fields.iter().map(|f| {
                let fline = f.name.span.line.saturating_sub(1) as u32;
                let fcol = f.name.span.col.saturating_sub(1) as u32;
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
            }).collect();
            Some(DocumentSymbol {
                name: name.name.clone(),
                kind: SymbolKind::Struct,
                range,
                selection_range,
                detail: Some(format!("{} fields", fields.len())),
                children,
            })
        }
        ItemKind::EnumDef { name, variants, .. } => {
            let children: Vec<DocumentSymbol> = variants.iter().map(|v| {
                let vline = v.name.span.line.saturating_sub(1) as u32;
                let vcol = v.name.span.col.saturating_sub(1) as u32;
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
                let mcol = m.name.span.col.saturating_sub(1) as u32;
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
                    let fcol = name.span.col.saturating_sub(1) as u32;
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
        TypeAnnotation::Array(inner) => format!("[{}]", type_ann_str(inner)),
        TypeAnnotation::FnType { params, ret } => {
            let params_str = params.iter().map(type_ann_str).collect::<Vec<_>>().join(", ");
            format!("fn({}) -> {}", params_str, type_ann_str(ret))
        }
        TypeAnnotation::Unit => "()".to_string(),
    }
}
