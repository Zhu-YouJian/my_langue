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
    let content = match crate::document_store::get_content_or_disk_global(uri) {
        Some(c) => c,
        None => return Vec::new(),
    };

    // Tokenize to find the identifier at the cursor position
    let mut lexer = Lexer::new(&content);
    let tokens = match lexer.tokenize() {
        Ok(t) => t,
        Err(_) => return Vec::new(),
    };

    // LSP positions are 0-based; lexer span lines are 1-based.
    let target_line = (line + 1) as usize;
    let target_col = character as usize;

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

/// Find the identifier token at the given position.
/// `target_line` is 1-based (lexer), `target_col0` is 0-based (LSP).
fn find_token_at_position(tokens: &[Token], target_line: usize, target_col0: usize) -> Option<String> {
    for token in tokens {
        if token.kind == TokenKind::Eof {
            continue;
        }
        let token_line = token.span.line;

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

            // Lexer span.col 对标识符/关键字 token 是 2-based；用归一化 helper
            // 还原 0-based 起始列，再判断光标是否落在 [start, start+len) 内。
            let start0 = crate::span::token_start_col0(token);
            let end0 = start0 + name.len();
            if target_col0 >= start0 && target_col0 < end0 {
                return Some(name);
            }
        }
    }
    None
}

/// Build a `Location` from an HIR function-definition span.
/// HIR fn spans come from identifier spans (lexer col is 2-based) and
/// lines are 1-based; convert to LSP 0-based positions.
fn hir_fn_location(uri: &str, line: usize, col: usize, name_len: usize) -> Location {
    Location {
        uri: uri.to_string(),
        range: Range {
            start: Position {
                line: line.saturating_sub(1) as u32,
                character: col.saturating_sub(2) as u32,
            },
            end: Position {
                line: line.saturating_sub(1) as u32,
                character: col.saturating_sub(2) as u32 + name_len as u32,
            },
        },
    }
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
            return Some(hir_fn_location(uri, line, col, identifier.len()));
        }
    }

    // Search in generic functions
    for fn_def in &hir_program.generic_funcs {
        if fn_def.name == identifier {
            let (line, col) = (fn_def.span.line, fn_def.span.col);
            return Some(hir_fn_location(uri, line, col, identifier.len()));
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
            return Some(hir_fn_location(uri, line, col, identifier.len()));
        }
    }

    // Search in trait impls
    for (_trait_name, type_map) in &hir_program.trait_impls {
        for (_type_name, method_map) in type_map {
            if let Some(fn_def) = method_map.get(identifier) {
                let (line, col) = (fn_def.span.line, fn_def.span.col);
                return Some(hir_fn_location(uri, line, col, identifier.len()));
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
            return Some(hir_fn_location(uri, line, col, identifier.len()));
        }
    }

    for fn_def in &hir_program.generic_funcs {
        if fn_def.name == identifier {
            let (line, col) = (fn_def.span.line, fn_def.span.col);
            return Some(hir_fn_location(uri, line, col, identifier.len()));
        }
    }

    for (_type_name, method_map) in &hir_program.methods {
        if let Some(fn_def) = method_map.get(identifier) {
            let (line, col) = (fn_def.span.line, fn_def.span.col);
            return Some(hir_fn_location(uri, line, col, identifier.len()));
        }
    }

    for (_trait_name, type_map) in &hir_program.trait_impls {
        for (_type_name, method_map) in type_map {
            if let Some(fn_def) = method_map.get(identifier) {
                let (line, col) = (fn_def.span.line, fn_def.span.col);
                return Some(hir_fn_location(uri, line, col, identifier.len()));
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

#[cfg(test)]
mod tests {
    use super::*;

    const URI: &str = "file:///C:/tmp/def_test.th";

    #[test]
    fn test_find_definition_function_jump() {
        // 调用点 → 函数定义跳转（HIR 解析）
        let src = "fn helper(x: i32) -> i32 { x }\nfn main() -> i32 { helper(1) }";
        let uri = "file:///C:/tmp/def_fn_test.th";
        crate::document_store::global().open(uri, 1, src);
        // 光标在 main 中的 `helper` 调用处（第 1 行）
        let locs = find_definition(uri, 1, 19);
        assert_eq!(locs.len(), 1, "应解析到唯一函数定义，实际: {locs:?}");
        assert_eq!(
            locs[0].range.start.line,
            0,
            "fn helper 定义应在第 0 行，实际: {:?}",
            locs[0]
        );
        assert_eq!(locs[0].uri, uri);
    }

    #[test]
    fn test_find_definition_local_variable_jump() {
        // let 局部变量 → 定义跳转（源码扫描 fallback）
        let src = "fn main() -> i32 {\n    let value = 42;\n    value\n}";
        let uri = "file:///C:/tmp/def_let_test.th";
        crate::document_store::global().open(uri, 2, src);
        // 光标在 return 处的 `value`（第 2 行第 8 列：4 空格 + "let "）
        let locs = find_definition(uri, 2, 8);
        assert!(
            !locs.is_empty(),
            "value 应能跳转到定义，实际: {locs:?}"
        );
        assert_eq!(locs[0].range.start.line, 1, "let value 定义应在第 1 行");
        assert_eq!(locs[0].range.start.character, 8);
    }

    #[test]
    fn test_find_definition_unknown_identifier_empty() {
        // 光标落在字面量（非标识符）上 → 无跳转结果（不静默返回错误位置）
        let src = "fn main() -> i32 { 0 }";
        let uri = "file:///C:/tmp/def_unknown_test.th";
        crate::document_store::global().open(uri, 3, src);
        // `0` 字面量在 0-based 19
        let locs = find_definition(uri, 0, 19);
        assert!(locs.is_empty(), "字面量不应有定义，实际: {locs:?}");
    }

    #[test]
    fn test_find_definition_struct_jump_via_source_scan() {
        // struct 定义跳转（源码扫描；HIR struct 无 span 走 fallback）
        let src = "struct Point { x: i32, y: i32 }\nfn make() -> Point { Point { x: 1, y: 2 } }";
        let uri = "file:///C:/tmp/def_struct_test.th";
        crate::document_store::global().open(uri, 4, src);
        // 光标在第二行 `Point {` 类型处
        let locs = find_definition(uri, 1, 15);
        assert!(
            !locs.is_empty(),
            "Point 应能跳转到 struct 定义，实际: {locs:?}"
        );
        assert_eq!(locs[0].range.start.line, 0, "struct Point 定义应在第 0 行");
    }
}
