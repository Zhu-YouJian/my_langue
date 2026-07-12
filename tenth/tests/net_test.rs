//! Stage 3+4 TCP/HTTP 原语集成测试。
//!
//! 覆盖 5 个 TCP native + 2 个 HTTP native：
//! - `tcp_connect(host: String, port: i64)` → `Result<i64>`（返回 1-based handle）
//! - `tcp_read(handle: i64, n: i64)` → `Result<Vec<i64>>`（读最多 n 字节）
//! - `tcp_write(handle: i64, data: Vec<i64>)` → `Result<i64>`（返回写入字节数）
//! - `tcp_close(handle: i64)` → `Unit`
//! - `tcp_set_timeout(handle: i64, ms: i64)` → `Unit`
//! - `http_get(url: String)` → `Result<String>`
//! - `http_post(url: String, body: String)` → `Result<String>`
//!
//! 使用解释器（Interpreter）执行——其 `call_named_fn`（natives.rs）已内置全部 TCP/HTTP native。
//! TCP 测试通过本地 echo 服务器验证；HTTP 测试通过本地 HTTP 服务器验证（避免联网依赖）。

use std::io::{Read, Write};
use std::net::TcpListener;

use tenth::hir::lower::Lowerer;
use tenth::lexer::lexer::Lexer;
use tenth::parser::parser::Parser;
use tenth::runtime::interpreter::Interpreter;
use tenth::runtime::value::Value;

/// Run source through lexer → parser → HIR → interpreter.
fn run_code(src: &str) -> Result<Option<Value>, String> {
    let mut lexer = Lexer::new(src);
    let tokens = lexer.tokenize().map_err(|e| e.to_string())?;
    let mut parser = Parser::new(tokens);
    let program = parser.parse_program().map_err(|e| e.to_string())?;
    let mut lowerer = Lowerer::new();
    let hir = lowerer.lower_program(&program).map_err(|e| e.to_string())?;
    let mut interpreter = Interpreter::new(&hir);
    interpreter.execute_program(&hir).map_err(|e| e.to_string())
}

/// 启动一个本地 TCP echo 服务器，返回端口号。
/// 服务器在单独线程中运行，循环读取并回写数据，直到客户端关闭连接。
fn start_echo_server() -> u16 {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    std::thread::spawn(move || {
        if let Ok((mut stream, _)) = listener.accept() {
            let mut buf = [0u8; 1024];
            loop {
                match stream.read(&mut buf) {
                    Ok(0) => break,
                    Ok(n) => {
                        if stream.write_all(&buf[..n]).is_err() {
                            break;
                        }
                    }
                    Err(_) => break,
                }
            }
        }
    });
    port
}

/// 启动一个本地 HTTP 服务器，返回端口号。
/// 服务器只处理一个请求，返回 HTTP 200 + 指定 body，然后关闭连接。
fn start_http_server(body: &str) -> u16 {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    let body = body.to_string();
    std::thread::spawn(move || {
        if let Ok((mut stream, _)) = listener.accept() {
            // 读取请求（不解析内容，只消费掉）
            let mut buf = [0u8; 4096];
            let _ = stream.read(&mut buf);
            // 写入 HTTP 响应；Connection: close 让客户端 read_to_end 在收完 body 后返回
            let response = format!(
                "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                body.len(),
                body
            );
            let _ = stream.write_all(response.as_bytes());
            // drop stream 关闭连接
        }
    });
    port
}

// ─── Test 1: tcp_connect 连接失败返回 Err ────────────────────────────────────

#[test]
fn test_tcp_connect_fail() {
    // 连接一个几乎肯定没有监听的端口（1 号端口，特权端口，通常无服务且无权限监听）
    let src = r#"
        let r = tcp_connect("127.0.0.1", 1);
        match r {
            Result::Ok(_) => 0,
            Result::Err(_) => 1,
        }
    "#;
    let result = run_code(src).unwrap();
    match result {
        Some(Value::Int(1, _)) => {}
        v => panic!("期望 Some(Int(1)) 表示连接失败返回 Err，got {:?}", v),
    }
}

// ─── Test 2: tcp_echo 本地回显服务器 ────────────────────────────────────────

#[test]
fn test_tcp_echo() {
    let port = start_echo_server();
    // 连接 → 设置超时 → 写 "Hi"([72,105]) → 读回 → 返回读到的字节数
    let src = format!(
        r#"
        let r = tcp_connect("127.0.0.1", {port});
        match r {{
            Result::Ok(handle) => {{
                tcp_set_timeout(handle, 5000);
                tcp_write(handle, [72, 105]);
                let data = tcp_read(handle, 100);
                match data {{
                    Result::Ok(v) => v.len(),
                    Result::Err(_) => -1,
                }}
            }},
            Result::Err(_) => -2,
        }}
        "#
    );
    let result = run_code(&src).unwrap();
    match result {
        Some(Value::Int(2, _)) => {}
        v => panic!("期望 Some(Int(2)) 表示读到 2 字节，got {:?}", v),
    }
}

// ─── Test 3: tcp_close 后再 read 返回 Err ───────────────────────────────────

#[test]
fn test_tcp_close() {
    let port = start_echo_server();
    // 连接 → 关闭 → 再 read 应返回 Err（"连接已关闭"）
    let src = format!(
        r#"
        let r = tcp_connect("127.0.0.1", {port});
        match r {{
            Result::Ok(handle) => {{
                tcp_close(handle);
                let data = tcp_read(handle, 100);
                match data {{
                    Result::Ok(_) => 0,
                    Result::Err(_) => 1,
                }}
            }},
            Result::Err(_) => -2,
        }}
        "#
    );
    let result = run_code(&src).unwrap();
    match result {
        Some(Value::Int(1, _)) => {}
        v => panic!("期望 Some(Int(1)) 表示 close 后 read 返回 Err，got {:?}", v),
    }
}

// ─── Test 4: http_get 本地 HTTP 服务器 ──────────────────────────────────────

#[test]
fn test_http_get_local() {
    let port = start_http_server("Hello");
    let src = format!(
        r#"
        let url = "http://127.0.0.1:{port}/";
        let r = http_get(url);
        match r {{
            Result::Ok(body) => body,
            Result::Err(_) => "FAIL",
        }}
        "#
    );
    let result = run_code(&src).unwrap();
    match result {
        Some(Value::String(s)) => assert_eq!(s, "Hello"),
        v => panic!("期望 Some(String(\"Hello\"))，got {:?}", v),
    }
}

// ─── Test 5: http_get 无效 URL 返回 Err ─────────────────────────────────────

#[test]
fn test_http_get_invalid_url() {
    // 不以 http:// 开头的 URL，parse_http_url 会返回 Err
    let src = r#"
        let r = http_get("not-a-valid-url");
        match r {
            Result::Ok(_) => 0,
            Result::Err(_) => 1,
        }
    "#;
    let result = run_code(src).unwrap();
    match result {
        Some(Value::Int(1, _)) => {}
        v => panic!("期望 Some(Int(1)) 表示无效 URL 返回 Err，got {:?}", v),
    }
}

// ─── Test 6: http_get https:// URL 返回 Err（不支持 HTTPS）──────────────────

#[test]
fn test_http_get_https_rejected() {
    // parse_http_url 对 https:// 开头的 URL 显式返回 Err
    let src = r#"
        let r = http_get("https://example.com/");
        match r {
            Result::Ok(_) => 0,
            Result::Err(_) => 1,
        }
    "#;
    let result = run_code(src).unwrap();
    match result {
        Some(Value::Int(1, _)) => {}
        v => panic!("期望 Some(Int(1)) 表示 HTTPS 被拒绝，got {:?}", v),
    }
}
