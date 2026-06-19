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
