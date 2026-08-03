//! LSP 协议层冒烟测试：真实启动 `tenth-lsp` 二进制，通过 stdio 做 JSON-RPC 交互。
//!
//! 覆盖：initialize（能力协商）→ initialized → didOpen（推送诊断）→
//! shutdown → exit 完整生命周期。诊断推送与拉取（textDocument/diagnostic）均有覆盖。
//!
//! 说明：`CARGO_BIN_EXE_tenth-lsp` 由 cargo 在集成测试中注入，指向构建出的二进制。

use std::io::{BufRead, BufReader, Read, Write};
use std::process::{Child, Command, Stdio};
use std::sync::mpsc;
use std::thread;
use std::time::Duration;

/// 将 JSON 消息封装为 LSP 帧（Content-Length 头 + 空行 + body）。
fn frame(json: &str) -> Vec<u8> {
    let header = format!("Content-Length: {}\r\n\r\n", json.len());
    let mut buf = header.into_bytes();
    buf.extend_from_slice(json.as_bytes());
    buf
}

/// 从流中读取一条 LSP 消息（解析 Content-Length 帧）。
fn read_message(reader: &mut impl BufRead) -> Option<serde_json::Value> {
    let mut content_length: Option<usize> = None;
    loop {
        let mut line = String::new();
        if reader.read_line(&mut line).ok()? == 0 {
            return None;
        }
        let line = line.trim_end();
        if line.is_empty() {
            break;
        }
        if let Some(v) = line.strip_prefix("Content-Length:") {
            content_length = v.trim().parse().ok();
        }
    }
    let len = content_length?;
    let mut body = vec![0u8; len];
    reader.read_exact(&mut body).ok()?;
    serde_json::from_slice(&body).ok()
}

/// 一个已启动的 tenth-lsp 子进程，stdout 由后台线程持续读入 mpsc 通道。
struct LspServer {
    child: Child,
    rx: mpsc::Receiver<serde_json::Value>,
    next_id: u64,
}

impl LspServer {
    fn start() -> Self {
        let bin = env!("CARGO_BIN_EXE_tenth-lsp");
        let mut child = Command::new(bin)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .expect("failed to spawn tenth-lsp");
        let stdout = child.stdout.take().unwrap();
        let (tx, rx) = mpsc::channel();
        thread::spawn(move || {
            let mut reader = BufReader::new(stdout);
            while let Some(msg) = read_message(&mut reader) {
                if tx.send(msg).is_err() {
                    break;
                }
            }
        });
        LspServer {
            child,
            rx,
            next_id: 1,
        }
    }

    fn send_raw(&mut self, json: &str) {
        let stdin = self.child.stdin.as_mut().expect("stdin closed");
        stdin.write_all(&frame(json)).expect("write to lsp stdin");
        stdin.flush().expect("flush lsp stdin");
    }

    /// 发送请求并等待同 id 的响应（10s 超时）。
    fn request(&mut self, method: &str, params: serde_json::Value) -> serde_json::Value {
        let id = self.next_id;
        self.next_id += 1;
        let msg = serde_json::json!({
            "jsonrpc": "2.0",
            "id": id,
            "method": method,
            "params": params,
        });
        self.send_raw(&serde_json::to_string(&msg).unwrap());
        loop {
            let msg = self
                .rx
                .recv_timeout(Duration::from_secs(10))
                .expect("timeout waiting for response");
            if msg.get("id").and_then(|i| i.as_u64()) == Some(id) {
                return msg;
            }
        }
    }

    fn notification(&mut self, method: &str, params: serde_json::Value) {
        let msg = serde_json::json!({
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
        });
        self.send_raw(&serde_json::to_string(&msg).unwrap());
    }

    /// 等待下一条消息（通知或响应），超时返回 None。
    fn recv(&self, timeout: Duration) -> Option<serde_json::Value> {
        self.rx.recv_timeout(timeout).ok()
    }

    /// 关闭 stdin 并等待进程退出。
    fn shutdown_and_wait(mut self) -> std::process::ExitStatus {
        drop(self.child.stdin.take());
        self.child.wait().expect("failed to wait for lsp exit")
    }
}

const TEST_URI: &str = "file:///C:/tmp/lsp_smoke_test.th";

/// 完整的握手流程：initialize → initialized。
fn handshake(server: &mut LspServer) {
    let resp = server.request(
        "initialize",
        serde_json::json!({
            "processId": null,
            "rootUri": null,
            "capabilities": {}
        }),
    );
    assert!(resp.get("error").is_none(), "initialize 不应报错: {resp}");
    assert!(
        resp.get("result")
            .and_then(|r| r.get("capabilities"))
            .is_some(),
        "initialize 应返回 capabilities: {resp}"
    );
    server.notification("initialized", serde_json::json!({}));
}

#[test]
fn test_initialize_returns_capabilities() {
    let mut server = LspServer::start();
    handshake(&mut server);
    let status = server.shutdown_and_wait();
    assert!(status.success(), "exit 后进程应以 0 退出，实际: {status}");
}

#[test]
fn test_did_open_valid_source_no_diagnostics() {
    let mut server = LspServer::start();
    handshake(&mut server);

    server.notification(
        "textDocument/didOpen",
        serde_json::json!({
            "textDocument": {
                "uri": TEST_URI,
                "languageId": "tenth",
                "version": 1,
                "text": "fn add(a: i32, b: i32) -> i32 { a + b }"
            }
        }),
    );

    // 合法源码 → publishDiagnostics 应推送空诊断列表
    let msg = server
        .recv(Duration::from_secs(10))
        .expect("应收到 publishDiagnostics 通知");
    assert_eq!(msg["method"], "textDocument/publishDiagnostics");
    let params = &msg["params"];
    assert_eq!(params["uri"], TEST_URI);
    let diags = params["diagnostics"].as_array().unwrap();
    assert!(
        diags.is_empty(),
        "合法源码不应有诊断，实际: {diags:?}"
    );

    let status = server.shutdown_and_wait();
    assert!(status.success());
}

#[test]
fn test_did_open_invalid_source_pushes_error_diagnostic() {
    let mut server = LspServer::start();
    handshake(&mut server);

    server.notification(
        "textDocument/didOpen",
        serde_json::json!({
            "textDocument": {
                "uri": TEST_URI,
                "languageId": "tenth",
                "version": 1,
                "text": "fn buggy() -> i32 { undefined_name }"
            }
        }),
    );

    let msg = server
        .recv(Duration::from_secs(10))
        .expect("应收到 publishDiagnostics 通知");
    assert_eq!(msg["method"], "textDocument/publishDiagnostics");
    let diags = msg["params"]["diagnostics"].as_array().unwrap();
    assert!(
        !diags.is_empty(),
        "未定义变量应产生诊断"
    );
    // 诊断应包含行/列与消息
    let d = &diags[0];
    assert_eq!(d["severity"].as_u64(), Some(1), "错误诊断 severity 应为 Error=1");
    assert!(d["range"]["start"]["line"].is_u64(), "诊断应包含行号");
    assert!(
        d["message"].as_str().unwrap().contains("未定义变量"),
        "诊断消息应包含未定义变量，实际: {}",
        d["message"]
    );
    assert_eq!(d["source"], "tenth");

    let status = server.shutdown_and_wait();
    assert!(status.success());
}

#[test]
fn test_pull_diagnostics_request() {
    let mut server = LspServer::start();
    handshake(&mut server);

    // didOpen 先推送，再发拉取请求 textDocument/diagnostic
    server.notification(
        "textDocument/didOpen",
        serde_json::json!({
            "textDocument": {
                "uri": TEST_URI,
                "languageId": "tenth",
                "version": 1,
                "text": "fn bad() -> i32 { \"unclosed }"
            }
        }),
    );
    // 消费推送的通知
    let pushed = server
        .recv(Duration::from_secs(10))
        .expect("应收到 publishDiagnostics 通知");
    assert_eq!(pushed["method"], "textDocument/publishDiagnostics");

    let resp = server.request(
        "textDocument/diagnostic",
        serde_json::json!({
            "textDocument": { "uri": TEST_URI }
        }),
    );
    assert!(resp.get("error").is_none(), "pull 诊断不应报错: {resp}");
    let result = resp["result"].as_array().unwrap();
    assert!(
        !result.is_empty(),
        "未闭合字符串应产生诊断"
    );

    let status = server.shutdown_and_wait();
    assert!(status.success());
}

#[test]
fn test_shutdown_then_exit_clean() {
    let mut server = LspServer::start();
    handshake(&mut server);

    let resp = server.request("shutdown", serde_json::Value::Null);
    assert!(resp.get("error").is_none(), "shutdown 不应报错: {resp}");
    assert_eq!(resp["result"], serde_json::Value::Null);

    server.notification("exit", serde_json::json!({}));
    let status = server.shutdown_and_wait();
    assert!(status.success(), "exit 后应以 0 退出，实际: {status}");
}
