use std::io::{self, BufRead, Read, Write};

use crate::lsp_types::{LspNotification, LspRequest, LspResponse};

/// Read an LSP message from stdin (Content-Length header + JSON body).
pub fn read_message() -> Option<LspRequest> {
    let stdin = io::stdin();
    let mut stdin_lock = stdin.lock();

    // Read Content-Length header
    let mut content_length: usize = 0;
    loop {
        let mut line = String::new();
        match stdin_lock.read_line(&mut line) {
            Ok(0) => return None, // EOF
            Ok(_) => {}
            Err(_) => return None,
        }

        let line = line.trim();
        if line.is_empty() {
            // Empty line signals end of headers
            break;
        }

        if let Some(len_str) = line.strip_prefix("Content-Length:") {
            content_length = len_str.trim().parse().ok()?;
        }
    }

    if content_length == 0 {
        return None;
    }

    // Read JSON body
    let mut body = vec![0u8; content_length];
    if let Err(_) = stdin_lock.read_exact(&mut body) {
        return None;
    }

    let body_str = String::from_utf8(body).ok()?;
    serde_json::from_str(&body_str).ok()
}

/// Write an LSP response to stdout (Content-Length header + JSON body).
pub fn write_response(response: LspResponse) {
    let json = match serde_json::to_string(&response) {
        Ok(j) => j,
        Err(_) => return,
    };

    write_json(&json);
}

/// Write an LSP notification to stdout (Content-Length header + JSON body).
pub fn write_notification(notification: LspNotification) {
    let json = match serde_json::to_string(&notification) {
        Ok(j) => j,
        Err(_) => return,
    };

    write_json(&json);
}

fn write_json(json: &str) {
    let stdout = io::stdout();
    let mut stdout_lock = stdout.lock();

    if write!(stdout_lock, "Content-Length: {}\r\n\r\n", json.len()).is_err() {
        return;
    }
    if stdout_lock.write_all(json.as_bytes()).is_err() {
        return;
    }
    let _ = stdout_lock.flush();
}
