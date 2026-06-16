use std::fs;
use std::path::Path;

use tenth::lexer::lexer::Lexer;
use tenth::lexer::token::{Token, TokenKind};
use tenth::parser::parser::Parser;
use tenth::hir::lower::Lowerer;

use super::Handler;
use crate::lsp_types::*;

pub struct DefinitionHandler;

impl Handler for DefinitionHandler {
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

        let (line, character) = match position {
            Some(pos) => (pos.line, pos.character),
            None => return serde_json::to_value(Vec::<Location>::new()).unwrap(),
        };

        let locations = find_definition(uri, line, character);
        serde_json::to_value(locations).unwrap()
    }
}

fn uri_to_path(uri: &str) -> String {
    if let Some(stripped) = uri.strip_prefix("file:///") {
        stripped.to_string()
    } else if let Some(stripped) = uri.strip_prefix("file://") {
        stripped.to_string()
    } else {
        uri.to_string()
    }
}

fn find_definition(uri: &str, line: u32, character: u32) -> Vec<Location> {
    let path = uri_to_path(uri);
    let content = match fs::read_to_string(Path::new(&path)) {
        Ok(c) => c,
        Err(_) => return Vec::new(),
    };

    // Tokenize to find the identifier at the cursor position
    let mut lexer = Lexer::new(&content);
    let tokens = match lexer.tokenize() {
        Ok(t) => t,
        Err(_) => return Vec::new(),
    };

    // LSP positions are 0-based; lexer spans are 1-based
    let target_line = (line + 1) as usize;
    let target_col = (character + 1) as usize;

    let identifier = find_token_at_position(&tokens, target_line, target_col);
    let identifier = match identifier {
        Some(name) => name,
        None => return Vec::new(),
    };

    // Try to resolve via HIR first, then fall back to source scanning
    if let Some(loc) = resolve_via_hir(&content, uri, &identifier) {
        return vec![loc];
    }

    // Fall back: scan source lines for definition patterns
    resolve_via_source_scan(&content, uri, &identifier)
}

/// Find the identifier token at the given (1-based) line and column.
fn find_token_at_position(tokens: &[Token], target_line: usize, target_col: usize) -> Option<String> {
    for token in tokens {
        if token.kind == TokenKind::Eof {
            continue;
        }
        let token_line = token.span.line;
        let token_col = token.span.col;

        // Check if cursor is within this token's span
        if token_line == target_line {
            let name = match &token.kind {
                TokenKind::Identifier(s) => s.clone(),
                TokenKind::Fn => "fn".to_string(),
                TokenKind::Struct => "struct".to_string(),
                TokenKind::Enum => "enum".to_string(),
                TokenKind::Impl => "impl".to_string(),
                TokenKind::Trait => "trait".to_string(),
                _ => continue,
            };

            // Token spans store the start column; the token extends for name.len() chars
            if target_col >= token_col && target_col < token_col + name.len() {
                return Some(name);
            }
        }
    }
    None
}

/// Try to resolve the identifier using the HIR (parse + lower).
fn resolve_via_hir(content: &str, uri: &str, identifier: &str) -> Option<Location> {
    let mut lexer = Lexer::new(content);
    let tokens = lexer.tokenize().ok()?;
    let mut parser = Parser::new(tokens);
    let program = parser.parse_program().ok()?;
    let mut lowerer = Lowerer::new();
    let hir_program = lowerer.lower_program(&program).ok()?;

    // Search in functions
    for fn_def in &hir_program.functions {
        if fn_def.name == identifier {
            let (line, col) = (fn_def.span.line, fn_def.span.col);
            return Some(Location {
                uri: uri.to_string(),
                range: Range {
                    start: Position {
                        line: line.saturating_sub(1) as u32,
                        character: col.saturating_sub(1) as u32,
                    },
                    end: Position {
                        line: line.saturating_sub(1) as u32,
                        character: col.saturating_sub(1) as u32 + identifier.len() as u32,
                    },
                },
            });
        }
    }

    // Search in generic functions
    for fn_def in &hir_program.generic_funcs {
        if fn_def.name == identifier {
            let (line, col) = (fn_def.span.line, fn_def.span.col);
            return Some(Location {
                uri: uri.to_string(),
                range: Range {
                    start: Position {
                        line: line.saturating_sub(1) as u32,
                        character: col.saturating_sub(1) as u32,
                    },
                    end: Position {
                        line: line.saturating_sub(1) as u32,
                        character: col.saturating_sub(1) as u32 + identifier.len() as u32,
                    },
                },
            });
        }
    }

    // Search in structs
    if hir_program.structs.contains_key(identifier) {
        // HIR structs don't store span; fall back to source scan
        return None;
    }

    // Search in generic structs
    if hir_program.generic_structs.contains_key(identifier) {
        return None;
    }

    // Search in enums
    if hir_program.enums.contains_key(identifier) {
        return None;
    }

    // Search in trait definitions
    if hir_program.trait_defs.contains_key(identifier) {
        return None;
    }

    // Search in methods (impl blocks)
    for (_type_name, method_map) in &hir_program.methods {
        if let Some(fn_def) = method_map.get(identifier) {
            let (line, col) = (fn_def.span.line, fn_def.span.col);
            return Some(Location {
                uri: uri.to_string(),
                range: Range {
                    start: Position {
                        line: line.saturating_sub(1) as u32,
                        character: col.saturating_sub(1) as u32,
                    },
                    end: Position {
                        line: line.saturating_sub(1) as u32,
                        character: col.saturating_sub(1) as u32 + identifier.len() as u32,
                    },
                },
            });
        }
    }

    // Search in trait impls
    for (_trait_name, type_map) in &hir_program.trait_impls {
        for (_type_name, method_map) in type_map {
            if let Some(fn_def) = method_map.get(identifier) {
                let (line, col) = (fn_def.span.line, fn_def.span.col);
                return Some(Location {
                    uri: uri.to_string(),
                    range: Range {
                        start: Position {
                            line: line.saturating_sub(1) as u32,
                            character: col.saturating_sub(1) as u32,
                        },
                        end: Position {
                            line: line.saturating_sub(1) as u32,
                            character: col.saturating_sub(1) as u32 + identifier.len() as u32,
                        },
                    },
                });
            }
        }
    }

    // Search in modules recursively
    for (_mod_name, mod_program) in &hir_program.modules {
        if let Some(loc) = resolve_in_hir_program(mod_program, uri, identifier) {
            return Some(loc);
        }
    }

    None
}

/// Recursively search for a definition within an HIR program (for modules).
fn resolve_in_hir_program(hir_program: &tenth::hir::hir::HirProgram, uri: &str, identifier: &str) -> Option<Location> {
    for fn_def in &hir_program.functions {
        if fn_def.name == identifier {
            let (line, col) = (fn_def.span.line, fn_def.span.col);
            return Some(Location {
                uri: uri.to_string(),
                range: Range {
                    start: Position {
                        line: line.saturating_sub(1) as u32,
                        character: col.saturating_sub(1) as u32,
                    },
                    end: Position {
                        line: line.saturating_sub(1) as u32,
                        character: col.saturating_sub(1) as u32 + identifier.len() as u32,
                    },
                },
            });
        }
    }

    for fn_def in &hir_program.generic_funcs {
        if fn_def.name == identifier {
            let (line, col) = (fn_def.span.line, fn_def.span.col);
            return Some(Location {
                uri: uri.to_string(),
                range: Range {
                    start: Position {
                        line: line.saturating_sub(1) as u32,
                        character: col.saturating_sub(1) as u32,
                    },
                    end: Position {
                        line: line.saturating_sub(1) as u32,
                        character: col.saturating_sub(1) as u32 + identifier.len() as u32,
                    },
                },
            });
        }
    }

    for (_type_name, method_map) in &hir_program.methods {
        if let Some(fn_def) = method_map.get(identifier) {
            let (line, col) = (fn_def.span.line, fn_def.span.col);
            return Some(Location {
                uri: uri.to_string(),
                range: Range {
                    start: Position {
                        line: line.saturating_sub(1) as u32,
                        character: col.saturating_sub(1) as u32,
                    },
                    end: Position {
                        line: line.saturating_sub(1) as u32,
                        character: col.saturating_sub(1) as u32 + identifier.len() as u32,
                    },
                },
            });
        }
    }

    for (_trait_name, type_map) in &hir_program.trait_impls {
        for (_type_name, method_map) in type_map {
            if let Some(fn_def) = method_map.get(identifier) {
                let (line, col) = (fn_def.span.line, fn_def.span.col);
                return Some(Location {
                    uri: uri.to_string(),
                    range: Range {
                        start: Position {
                            line: line.saturating_sub(1) as u32,
                            character: col.saturating_sub(1) as u32,
                        },
                        end: Position {
                            line: line.saturating_sub(1) as u32,
                            character: col.saturating_sub(1) as u32 + identifier.len() as u32,
                        },
                    },
                });
            }
        }
    }

    for (_mod_name, mod_program) in &hir_program.modules {
        if let Some(loc) = resolve_in_hir_program(mod_program, uri, identifier) {
            return Some(loc);
        }
    }

    None
}

/// Fall back: scan source lines for definition patterns like `fn NAME`, `struct NAME`, etc.
fn resolve_via_source_scan(content: &str, uri: &str, identifier: &str) -> Vec<Location> {
    let lines: Vec<&str> = content.lines().collect();
    let mut locations = Vec::new();

    // Patterns to search for: (keyword_prefix, whether name follows keyword directly)
    let patterns: &[(&str, fn(&str, &str) -> Option<usize>)] = &[
        ("fn ", |line, name| find_name_after_keyword(line, "fn", name)),
        ("struct ", |line, name| find_name_after_keyword(line, "struct", name)),
        ("enum ", |line, name| find_name_after_keyword(line, "enum", name)),
        ("impl ", |line, name| find_name_after_keyword(line, "impl", name)),
        ("trait ", |line, name| find_name_after_keyword(line, "trait", name)),
        ("let ", |line, name| find_let_name(line, name)),
    ];

    for (line_idx, line_text) in lines.iter().enumerate() {
        let trimmed = line_text.trim();

        for (keyword, finder) in patterns {
            if trimmed.starts_with(keyword) {
                if let Some(col_offset) = finder(trimmed, identifier) {
                    // Account for leading whitespace in the original line
                    let leading_ws = line_text.len() - line_text.trim_start().len();
                    let char_offset = leading_ws + col_offset;

                    locations.push(Location {
                        uri: uri.to_string(),
                        range: Range {
                            start: Position {
                                line: line_idx as u32,
                                character: char_offset as u32,
                            },
                            end: Position {
                                line: line_idx as u32,
                                character: char_offset as u32 + identifier.len() as u32,
                            },
                        },
                    });
                }
            }
        }
    }

    locations
}

/// Find the column offset where `name` appears after `keyword` in a line.
/// e.g., for "fn my_func(" with keyword="fn" and name="my_func", returns 3.
fn find_name_after_keyword(line: &str, keyword: &str, name: &str) -> Option<usize> {
    let after_keyword = line.strip_prefix(keyword)?;
    let after_keyword = after_keyword.trim_start();
    if after_keyword.starts_with(name) {
        // Check that the name is followed by a word boundary
        let rest = &after_keyword[name.len()..];
        if rest.is_empty() || !rest.chars().next().map_or(false, |c| c.is_alphanumeric() || c == '_') {
            let keyword_end = keyword.len();
            let spaces = line.len() - keyword.len() - after_keyword.len();
            return Some(keyword_end + spaces);
        }
    }
    None
}

/// Find the column offset where `name` appears in a `let` binding.
/// Handles patterns like `let name`, `let mut name`, `let name:`, `let name =`.
fn find_let_name(line: &str, name: &str) -> Option<usize> {
    let after_let = line.strip_prefix("let ")?;
    let after_let = after_let.trim_start();

    // Handle `let mut name`
    if after_let.starts_with("mut ") {
        let after_mut = after_let.strip_prefix("mut ")?;
        let after_mut = after_mut.trim_start();
        if after_mut.starts_with(name) {
            let rest = &after_mut[name.len()..];
            if rest.is_empty() || !rest.chars().next().map_or(false, |c| c.is_alphanumeric() || c == '_') {
                let offset = line.len() - after_mut.len();
                return Some(offset);
            }
        }
    }

    // Handle `let name`
    if after_let.starts_with(name) {
        let rest = &after_let[name.len()..];
        if rest.is_empty() || !rest.chars().next().map_or(false, |c| c.is_alphanumeric() || c == '_') {
            let offset = line.len() - after_let.len();
            return Some(offset);
        }
    }

    None
}
