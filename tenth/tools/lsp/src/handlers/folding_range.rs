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
