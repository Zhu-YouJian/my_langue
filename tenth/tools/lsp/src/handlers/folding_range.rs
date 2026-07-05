use super::Handler;
use crate::lsp_types::*;

pub struct FoldingRangeHandler;

impl Handler for FoldingRangeHandler {
    fn handle(&self, params: Option<&serde_json::Value>) -> serde_json::Value {
        let uri = params
            .and_then(|p| p.get("textDocument"))
            .and_then(|td| td.get("uri"))
            .and_then(|u| u.as_str())
            .unwrap_or("");

        let source = crate::document_store::get_content_or_disk_global(uri)
            .unwrap_or_default();

        let ranges = compute_folding_ranges(&source);
        serde_json::to_value(ranges).unwrap()
    }
}

fn compute_folding_ranges(source: &str) -> Vec<FoldingRange> {
    let lines: Vec<&str> = source.lines().collect();
    let mut ranges = Vec::new();

    // Track brace depth and the start line of each `{`
    let mut brace_stack: Vec<u32> = Vec::new();

    for (line_idx, line) in lines.iter().enumerate() {
        for ch in line.chars() {
            match ch {
                '{' => {
                    brace_stack.push(line_idx as u32);
                }
                '}' => {
                    if let Some(start_line) = brace_stack.pop() {
                        let end_line = line_idx as u32;
                        // Only fold if there's at least one line between start and end
                        if end_line > start_line {
                            ranges.push(FoldingRange {
                                start_line,
                                end_line,
                                start_character: None,
                                end_character: None,
                                kind: Some("region".to_string()),
                            });
                        }
                    }
                }
                _ => {}
            }
        }
    }

    // Also fold multi-line comments (lines starting with //)
    let mut comment_start: Option<u32> = None;
    for (line_idx, line) in lines.iter().enumerate() {
        let trimmed = line.trim_start();
        if trimmed.starts_with("//") {
            if comment_start.is_none() {
                comment_start = Some(line_idx as u32);
            }
        } else {
            if let Some(start) = comment_start {
                let end = (line_idx - 1) as u32;
                if end > start {
                    ranges.push(FoldingRange {
                        start_line: start,
                        end_line: end,
                        start_character: None,
                        end_character: None,
                        kind: Some("comment".to_string()),
                    });
                }
                comment_start = None;
            }
        }
    }
    // Handle comment block at end of file
    if let Some(start) = comment_start {
        let end = (lines.len() - 1) as u32;
        if end > start {
            ranges.push(FoldingRange {
                start_line: start,
                end_line: end,
                start_character: None,
                end_character: None,
                kind: Some("comment".to_string()),
            });
        }
    }

    ranges
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_folding_ranges_with_braces() {
        // 含多行 {} 块的代码应产生至少一个 region 折叠范围
        let src = "fn add() -> i32 {\n    let a = 1;\n    let b = 2;\n    a + b\n}\n";
        let ranges = compute_folding_ranges(src);
        assert!(
            !ranges.is_empty(),
            "expected at least one folding range for brace block"
        );
        // 应有一个 region 类型的折叠，从 line 0 到 line 4
        let region = ranges.iter().find(|r| r.kind.as_deref() == Some("region"));
        assert!(region.is_some(), "expected a 'region' kind folding, got: {:?}", ranges);
        let r = region.unwrap();
        assert_eq!(r.start_line, 0, "region should start at line 0");
        assert_eq!(r.end_line, 4, "region should end at line 4 (the closing brace line)");
    }

    #[test]
    fn test_folding_ranges_no_braces() {
        // 无 {} 块的代码不应产生任何 region 折叠
        let src = "fn empty() -> i32 { 0 }\n";
        let ranges = compute_folding_ranges(src);
        // 单行 { 0 } 的开闭在同一行，end_line == start_line，不会产生范围
        let region_count = ranges.iter()
            .filter(|r| r.kind.as_deref() == Some("region"))
            .count();
        assert_eq!(
            region_count, 0,
            "expected no region folding for single-line block, got: {:?}",
            ranges
        );
    }

    #[test]
    fn test_folding_ranges_nested_blocks() {
        // 嵌套块应产生多个折叠范围
        let src = "fn outer() -> i32 {\n    if true {\n        1\n    } else {\n        2\n    }\n}\n";
        let ranges = compute_folding_ranges(src);
        let region_count = ranges.iter()
            .filter(|r| r.kind.as_deref() == Some("region"))
            .count();
        // outer { ... } 包含两个内部 if/else 块，应至少 3 个 region
        assert!(
            region_count >= 3,
            "expected at least 3 region foldings (outer + 2 inner), got {}: {:?}",
            region_count, ranges
        );
    }

    #[test]
    fn test_folding_ranges_empty_input() {
        // 空输入不应产生任何折叠范围
        let ranges = compute_folding_ranges("");
        assert!(ranges.is_empty(), "expected no folding ranges for empty input");
    }
}
