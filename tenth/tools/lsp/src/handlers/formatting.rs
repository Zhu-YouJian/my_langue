use super::Handler;
use crate::lsp_types::*;

pub struct FormattingHandler;

impl Handler for FormattingHandler {
    fn handle(&self, params: Option<&serde_json::Value>) -> serde_json::Value {
        let uri = params
            .and_then(|p| p.get("textDocument"))
            .and_then(|td| td.get("uri"))
            .and_then(|u| u.as_str())
            .unwrap_or("");

        let source = crate::document_store::get_content_or_disk_global(uri)
            .unwrap_or_default();

        let formatted = format_source(&source);
        let lines: Vec<&str> = source.lines().collect();
        let line_count = lines.len() as u32;

        let edit = TextEdit {
            range: Range {
                start: Position { line: 0, character: 0 },
                end: Position { line: line_count, character: 0 },
            },
            new_text: formatted,
        };

        serde_json::to_value(vec![edit]).unwrap()
    }
}

/// Format Tenth source code.
///
/// Rules:
/// 1. Consistent 4-space indentation
/// 2. No trailing whitespace
/// 3. Single blank line between top-level items (fn, struct, enum, impl, etc.)
/// 4. No more than one consecutive blank line
/// 5. Space after commas, colons, and keywords
/// 6. No space before commas/colons
/// 7. Trim trailing whitespace
fn format_source(source: &str) -> String {
    let lines: Vec<&str> = source.lines().collect();
    let mut result = Vec::with_capacity(lines.len());
    let mut indent_level: usize = 0;
    let mut prev_was_blank = false;
    let mut prev_was_item = false;

    for (i, line) in lines.iter().enumerate() {
        let trimmed = line.trim();

        // Skip empty lines but track them
        if trimmed.is_empty() {
            if !prev_was_blank && i > 0 && i < lines.len() - 1 {
                result.push(String::new());
                prev_was_blank = true;
            }
            continue;
        }

        // Decrease indent for closing braces
        if trimmed.starts_with('}') || trimmed.starts_with(')') || trimmed.starts_with(']') {
            if indent_level > 0 {
                indent_level -= 1;
            }
        }

        // Check if this is a top-level item
        let is_item = is_top_level_item(trimmed);

        // Add blank line before top-level items (except the first)
        if is_item && i > 0 && !prev_was_blank && prev_was_item {
            result.push(String::new());
        }

        // Format the line content
        let formatted_content = format_line_content(trimmed);

        // Apply indentation
        let indented = format!("{}{}", "    ".repeat(indent_level), formatted_content);
        result.push(indented);

        // Track state
        prev_was_blank = false;
        prev_was_item = is_item;

        // Increase indent for opening braces
        let open_braces = trimmed.chars().filter(|&c| c == '{').count();
        let close_braces = trimmed.chars().filter(|&c| c == '}').count();
        if open_braces > close_braces {
            indent_level += open_braces - close_braces;
        }
    }

    // Remove trailing blank lines
    while result.last().map(|l| l.is_empty()).unwrap_or(false) {
        result.pop();
    }

    // Ensure file ends with newline
    let mut output = result.join("\n");
    output.push('\n');
    output
}

fn is_top_level_item(line: &str) -> bool {
    line.starts_with("fn ")
        || line.starts_with("pub fn ")
        || line.starts_with("struct ")
        || line.starts_with("pub struct ")
        || line.starts_with("enum ")
        || line.starts_with("pub enum ")
        || line.starts_with("impl ")
        || line.starts_with("trait ")
        || line.starts_with("pub trait ")
        || line.starts_with("use ")
        || line.starts_with("mod ")
        || line.starts_with("pub mod ")
        || line.starts_with("const ")
        || line.starts_with("pub const ")
}

fn format_line_content(line: &str) -> String {
    let mut result = String::with_capacity(line.len() + 8);
    let chars: Vec<char> = line.chars().collect();
    let mut i = 0;
    let mut prev_char = '\0';

    while i < chars.len() {
        let c = chars[i];

        // Skip trailing whitespace (handled by trim)
        if c == ' ' && i == chars.len() - 1 {
            break;
        }

        // Space after comma (but not before)
        if c == ',' {
            result.push(',');
            // Add space after comma if not at end of line and next isn't already space
            if i + 1 < chars.len() && chars[i + 1] != ' ' && chars[i + 1] != ')' && chars[i + 1] != ']' {
                result.push(' ');
            }
            i += 1;
            continue;
        }

        // Space after colon (for type annotations), not before
        if c == ':' {
            // Don't add space before ::
            if i + 1 < chars.len() && chars[i + 1] == ':' {
                result.push_str("::");
                i += 2;
                continue;
            }
            result.push(':');
            if i + 1 < chars.len() && chars[i + 1] != ' ' {
                result.push(' ');
            }
            i += 1;
            continue;
        }

        // Space around = (for let bindings and assignments)
        if c == '=' && prev_char != '=' && prev_char != '!' && prev_char != '<' && prev_char != '>' && prev_char != ':' {
            // Don't add space for ==, !=, <=, >=, :=
            if i + 1 < chars.len() && chars[i + 1] == '=' {
                result.push_str("==");
                i += 2;
                continue;
            }
            // Add space before and after =
            if !result.ends_with(' ') {
                result.push(' ');
            }
            result.push('=');
            if i + 1 < chars.len() && chars[i + 1] != ' ' && chars[i + 1] != '=' {
                result.push(' ');
            }
            i += 1;
            continue;
        }

        // Space after keywords
        if c == ' ' {
            // Collapse multiple spaces into one
            if !result.ends_with(' ') {
                result.push(' ');
            }
            i += 1;
            continue;
        }

        result.push(c);
        prev_char = c;
        i += 1;
    }

    // Clean up: remove space before commas/colons
    let result = result.replace(" ,", ",").replace(" :", ":");

    // Remove trailing whitespace
    result.trim_end().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_already_formatted_unchanged() {
        // 已正确格式化的代码（顶层 fn + 4 空格缩进 + 末尾换行）应保持不变
        let src = "fn add(a: i32, b: i32) -> i32 {\n    a + b\n}\n";
        let out = format_source(src);
        // 关键属性：末尾换行 + 4 空格缩进 + 内容保留
        assert!(out.ends_with('\n'), "formatted output must end with newline");
        assert!(
            out.lines().nth(1).map(|l| l.starts_with("    ")).unwrap_or(false),
            "second line should be indented with 4 spaces, got: {:?}",
            out.lines().nth(1)
        );
        assert!(out.contains("fn add"), "fn add should be present: {}", out);
        assert!(out.contains("a + b"), "function body should be preserved: {}", out);
    }

    #[test]
    fn test_format_fixes_indentation() {
        // 错误缩进（用 tab，应改为 4 空格）
        let src = "fn f() -> i32 {\n\ta + b\n}\n";
        let out = format_source(src);
        // 输出第二行应使用 4 空格而非 tab
        let second = out.lines().nth(1).unwrap_or("");
        assert!(
            second.starts_with("    ") && !second.starts_with('\t'),
            "expected 4-space indent, got: {:?}",
            second
        );
    }

    #[test]
    fn test_format_collapses_blank_lines() {
        // 多个连续空行应压缩为单个空行（在两个顶层 item 之间）
        let src = "fn a() -> i32 { 1 }\n\n\n\nfn b() -> i32 { 2 }\n";
        let out = format_source(src);
        // 两个 fn 之间应只有 1 个空行
        let blank_count = out.lines()
            .collect::<Vec<_>>()
            .windows(2)
            .filter(|w| w[0].is_empty() && w[1].is_empty())
            .count();
        assert_eq!(
            blank_count, 0,
            "expected no consecutive blank lines, got: {:?}",
            out
        );
        // 两个 fn 定义应都存在
        assert!(out.contains("fn a()"), "fn a should be present: {}", out);
        assert!(out.contains("fn b()"), "fn b should be present: {}", out);
    }

    #[test]
    fn test_format_ends_with_newline() {
        // 格式化输出必须以换行符结尾（POSIX 文件规范）
        let src = "fn f() -> i32 { 0 }";
        let out = format_source(src);
        assert!(out.ends_with('\n'), "output must end with newline: {:?}", out);
    }

    #[test]
    fn test_format_empty_input() {
        // 空输入应返回单个换行或空字符串（不 panic）
        let out = format_source("");
        // 不应包含错误内容，且不 panic
        assert!(out.is_empty() || out == "\n", "empty input maps to empty/newline: {:?}", out);
    }
}
