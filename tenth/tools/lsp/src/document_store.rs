use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// In-memory store for open text documents.
///
/// Tracks the current content of each open file so that LSP handlers
/// (diagnostics, hover, completion, etc.) can work with live edits
/// instead of reading stale content from disk.
#[derive(Debug, Clone)]
pub struct DocumentStore {
    documents: Arc<Mutex<HashMap<String, Document>>>,
}

#[derive(Debug, Clone)]
pub struct Document {
    pub uri: String,
    pub version: i64,
    pub content: String,
}

/// Global document store instance.
static GLOBAL_STORE: std::sync::OnceLock<DocumentStore> = std::sync::OnceLock::new();

/// Get the global document store.
pub fn global() -> &'static DocumentStore {
    GLOBAL_STORE.get_or_init(DocumentStore::new)
}

/// Convenience: get content for a URI from the global store, falling back to disk.
pub fn get_content_or_disk_global(uri: &str) -> Option<String> {
    global().get_content_or_disk(uri)
}

impl DocumentStore {
    pub fn new() -> Self {
        DocumentStore {
            documents: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Open a document with the given content.
    pub fn open(&self, uri: &str, version: i64, content: &str) {
        let mut docs = self.documents.lock().unwrap();
        docs.insert(
            uri.to_string(),
            Document {
                uri: uri.to_string(),
                version,
                content: content.to_string(),
            },
        );
    }

    /// Update a document with full content sync (change type 1).
    pub fn update_full(&self, uri: &str, version: i64, content: &str) {
        let mut docs = self.documents.lock().unwrap();
        if let Some(doc) = docs.get_mut(uri) {
            doc.version = version;
            doc.content = content.to_string();
        } else {
            docs.insert(
                uri.to_string(),
                Document {
                    uri: uri.to_string(),
                    version,
                    content: content.to_string(),
                },
            );
        }
    }

    /// Update a document with incremental sync (change type 2).
    /// Applies a range edit to the current content.
    pub fn update_incremental(
        &self,
        uri: &str,
        version: i64,
        range: &crate::lsp_types::Range,
        text: &str,
    ) {
        let mut docs = self.documents.lock().unwrap();
        if let Some(doc) = docs.get_mut(uri) {
            doc.version = version;
            doc.content = apply_range_edit(&doc.content, range, text);
        }
    }

    /// Close a document.
    pub fn close(&self, uri: &str) {
        let mut docs = self.documents.lock().unwrap();
        docs.remove(uri);
    }

    /// Get the content of a document. Returns None if not open.
    pub fn get_content(&self, uri: &str) -> Option<String> {
        let docs = self.documents.lock().unwrap();
        docs.get(uri).map(|d| d.content.clone())
    }

    /// Get a document's metadata.
    pub fn get_document(&self, uri: &str) -> Option<Document> {
        let docs = self.documents.lock().unwrap();
        docs.get(uri).cloned()
    }

    /// Check if a document is open.
    pub fn is_open(&self, uri: &str) -> bool {
        let docs = self.documents.lock().unwrap();
        docs.contains_key(uri)
    }

    /// Get the content for a URI, falling back to disk if not open.
    pub fn get_content_or_disk(&self, uri: &str) -> Option<String> {
        if let Some(content) = self.get_content(uri) {
            return Some(content);
        }
        // Fall back to disk
        let path = uri_to_path(uri);
        std::fs::read_to_string(&path).ok()
    }
}

/// Apply a range edit to source text.
/// LSP positions are 0-based (line, character).
fn apply_range_edit(
    content: &str,
    range: &crate::lsp_types::Range,
    new_text: &str,
) -> String {
    let lines: Vec<&str> = content.lines().collect();
    let start_line = range.start.line as usize;
    let start_col = range.start.character as usize;
    let end_line = range.end.line as usize;
    let end_col = range.end.character as usize;

    if start_line >= lines.len() {
        // Append at end
        let mut result = content.to_string();
        result.push_str(new_text);
        return result;
    }

    let mut result = String::new();

    // Lines before the edit
    for (i, line) in lines.iter().enumerate() {
        if i < start_line {
            result.push_str(line);
            result.push('\n');
        } else if i == start_line {
            // Prefix of the start line
            let prefix = if start_col <= line.len() {
                &line[..start_col]
            } else {
                line
            };
            result.push_str(prefix);
            result.push_str(new_text);

            // If the edit is on a single line, append the suffix
            if end_line == start_line {
                let suffix = if end_col <= line.len() {
                    &line[end_col..]
                } else {
                    ""
                };
                result.push_str(suffix);
            }
        } else if i > start_line && i < end_line {
            // Skip lines within the deleted range
            continue;
        } else if i == end_line && end_line != start_line {
            // Suffix of the end line
            let suffix = if end_col <= line.len() {
                &line[end_col..]
            } else {
                ""
            };
            result.push_str(suffix);
        } else if i > end_line {
            result.push('\n');
            result.push_str(line);
        }
    }

    result
}

/// Convert a file:// URI to a filesystem path.
pub fn uri_to_path(uri: &str) -> String {
    let path = if let Some(stripped) = uri.strip_prefix("file:///") {
        // On Windows, the path after file:/// is like /C:/...
        // Strip the leading slash if it looks like a drive letter
        if stripped.len() > 2 && stripped.chars().nth(1) == Some(':') {
            &stripped[1..]
        } else {
            stripped
        }
    } else if let Some(stripped) = uri.strip_prefix("file://") {
        stripped
    } else {
        uri
    };

    // On Windows, convert forward slashes to backslashes
    #[cfg(windows)]
    {
        path.replace('/', "\\")
    }
    #[cfg(not(windows))]
    {
        path.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lsp_types::{Position, Range};

    #[test]
    fn test_document_store_open_and_get_content() {
        // open + get_content 应返回正确内容
        let store = DocumentStore::new();
        let uri = "file:///tmp/test_open.th";
        let content = "fn main() -> i32 { 0 }";
        store.open(uri, 1, content);
        let retrieved = store.get_content(uri);
        assert_eq!(
            retrieved.as_deref(),
            Some(content),
            "open+get_content should return the original content"
        );
        // is_open 应为 true
        assert!(store.is_open(uri), "is_open should return true after open");
    }

    #[test]
    fn test_document_store_update_full_replaces_content() {
        // update_full 应整体替换内容
        let store = DocumentStore::new();
        let uri = "file:///tmp/test_full.th";
        store.open(uri, 1, "fn old() -> i32 { 1 }");
        store.update_full(uri, 2, "fn new() -> i32 { 2 }");
        let retrieved = store.get_content(uri);
        assert_eq!(
            retrieved.as_deref(),
            Some("fn new() -> i32 { 2 }"),
            "update_full should replace content entirely"
        );
        // 版本号应更新
        let doc = store.get_document(uri).unwrap();
        assert_eq!(doc.version, 2, "version should be updated to 2");
    }

    #[test]
    fn test_document_store_update_incremental_replaces_range() {
        // update_incremental 应在指定 range 内替换文本
        let store = DocumentStore::new();
        let uri = "file:///tmp/test_inc.th";
        // 初始：第一行 "fn add(a, b) -> i32 { a + b }"
        let initial = "fn add(a, b) -> i32 { a + b }";
        store.open(uri, 1, initial);
        // 把第 0 行 character 7-8 (即 'a') 替换为 'x'
        // "fn add(a, b)" → "fn add(x, b)"
        store.update_incremental(
            uri,
            2,
            &Range {
                start: Position { line: 0, character: 7 },
                end: Position { line: 0, character: 8 },
            },
            "x",
        );
        let retrieved = store.get_content(uri);
        assert_eq!(
            retrieved.as_deref(),
            Some("fn add(x, b) -> i32 { a + b }"),
            "incremental update should replace only the range, got: {:?}",
            retrieved
        );
    }

    #[test]
    fn test_document_store_close_removes_document() {
        // close 应移除文档
        let store = DocumentStore::new();
        let uri = "file:///tmp/test_close.th";
        store.open(uri, 1, "fn x() -> i32 { 0 }");
        assert!(store.is_open(uri));
        store.close(uri);
        assert!(
            !store.is_open(uri),
            "is_open should be false after close"
        );
        assert!(
            store.get_content(uri).is_none(),
            "get_content should return None after close"
        );
    }

    #[test]
    fn test_document_store_update_incremental_insert_at_end() {
        // update_incremental 在文件末尾追加文本（start_line 超出范围）
        let store = DocumentStore::new();
        let uri = "file:///tmp/test_append.th";
        store.open(uri, 1, "fn x() -> i32 { 0 }");
        store.update_incremental(
            uri,
            2,
            &Range {
                start: Position { line: 100, character: 0 },
                end: Position { line: 100, character: 0 },
            },
            "\n// appended",
        );
        let retrieved = store.get_content(uri);
        assert!(
            retrieved.as_deref().unwrap_or("").ends_with("// appended"),
            "incremental update at end should append text, got: {:?}",
            retrieved
        );
    }

    #[test]
    fn test_uri_to_path_strips_file_prefix() {
        // file:// URI 应被转换为路径
        let uri = "file:///tmp/test.th";
        let path = uri_to_path(uri);
        // 不应再包含 "file://" 前缀
        assert!(
            !path.starts_with("file:"),
            "uri_to_path should strip file:// prefix, got: {}",
            path
        );
    }
}
