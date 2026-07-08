//! HTTP 客户端辅助函数（手写 HTTP/1.1，不引入 reqwest/hyper 依赖）。
//!
//! 从 `main.rs` 迁移而来（T1.2）。原 `runtime/interpreter/natives.rs` 中的
//! 重复副本已删除，改用 `crate::http::*`。两侧语义必须保持一致。

/// 解析 URL，返回 (host, port, path)
///
/// 支持 `http://host:port/path` 与 `http://host/path`（默认端口 80）。
/// HTTPS 不支持：`https://` 开头返回 Err。
pub fn parse_http_url(url: &str) -> Result<(String, u16, String), String> {
    if url.starts_with("https://") {
        return Err("不支持 HTTPS，请使用 http://".to_string());
    }
    let rest = url
        .strip_prefix("http://")
        .ok_or_else(|| "URL 必须以 http:// 开头".to_string())?;
    let (host_port, path) = match rest.find('/') {
        Some(i) => (&rest[..i], &rest[i..]),
        None => (rest, "/"),
    };
    let (host, port) = match host_port.find(':') {
        Some(i) => {
            let h = &host_port[..i];
            let p: u16 = host_port[i + 1..]
                .parse()
                .map_err(|_| "端口号无效".to_string())?;
            (h.to_string(), p)
        }
        None => (host_port.to_string(), 80),
    };
    Ok((host, port, path.to_string()))
}

/// 解码 HTTP chunked 传输编码
///
/// 格式：`<size_hex>\r\n<data>\r\n` 重复，以 `0\r\n\r\n` 结束。
pub fn decode_chunked(data: &[u8]) -> Result<Vec<u8>, String> {
    let mut result = Vec::new();
    let mut pos = 0;
    while pos < data.len() {
        // 找 chunk size 行尾的 \r\n
        let mut line_end = pos;
        while line_end + 1 < data.len() && &data[line_end..line_end + 2] != b"\r\n" {
            line_end += 1;
        }
        if line_end + 1 >= data.len() {
            return Err("chunked 格式错误：未找到 size 行结束".to_string());
        }
        let size_str = std::str::from_utf8(&data[pos..line_end])
            .map_err(|e| format!("chunk size 非 UTF-8: {e}"))?;
        // size 可能带 ;extension，取分号前部分
        let size_hex = size_str.split(';').next().unwrap_or("").trim();
        let size = usize::from_str_radix(size_hex, 16)
            .map_err(|_| format!("chunk size 非十六进制: {size_hex}"))?;
        pos = line_end + 2; // 跳过 \r\n
        if size == 0 {
            break; // 末尾 chunk
        }
        if pos + size > data.len() {
            return Err("chunked 格式错误：chunk 数据不足".to_string());
        }
        result.extend_from_slice(&data[pos..pos + size]);
        pos += size;
        // 跳过 chunk 后的 \r\n
        if pos + 1 < data.len() && &data[pos..pos + 2] == b"\r\n" {
            pos += 2;
        }
    }
    Ok(result)
}

/// 读取并解析 HTTP 响应，返回 body 字符串
///
/// 一次性读到 EOF（依赖 Connection: close），分离响应头与 body。
/// 如果响应头含 `Transfer-Encoding: chunked`，按 chunked 解码 body。
pub fn http_read_response(stream: &mut std::net::TcpStream) -> Result<String, String> {
    use std::io::Read;
    let mut buf: Vec<u8> = Vec::new();
    stream
        .read_to_end(&mut buf)
        .map_err(|e| format!("读取响应失败: {e}"))?;
    let sep = b"\r\n\r\n";
    let mut header_end = None;
    let mut i = 0;
    while i + sep.len() <= buf.len() {
        if &buf[i..i + sep.len()] == sep {
            header_end = Some(i + sep.len());
            break;
        }
        i += 1;
    }
    let header_end = header_end.ok_or_else(|| "响应头格式错误：未找到 \r\n\r\n".to_string())?;
    let headers = std::str::from_utf8(&buf[..header_end - sep.len()])
        .map_err(|e| format!("响应头非 UTF-8: {e}"))?;
    let body = &buf[header_end..];
    let chunked = headers.lines().any(|line| {
        let lower = line.to_ascii_lowercase();
        lower.starts_with("transfer-encoding:") && lower.contains("chunked")
    });
    if chunked {
        let decoded = decode_chunked(body)?;
        String::from_utf8(decoded).map_err(|e| format!("响应体非 UTF-8: {e}"))
    } else {
        String::from_utf8(body.to_vec()).map_err(|e| format!("响应体非 UTF-8: {e}"))
    }
}

/// 发送 HTTP GET 请求，返回响应 body
pub fn http_get_impl(url: &str) -> Result<String, String> {
    let (host, port, path) = parse_http_url(url)?;
    let addr = format!("{host}:{port}");
    let mut stream = std::net::TcpStream::connect(&addr)
        .map_err(|e| format!("连接失败: {e}"))?;
    let timeout = std::time::Duration::from_secs(10);
    stream.set_read_timeout(Some(timeout)).ok();
    stream.set_write_timeout(Some(timeout)).ok();
    let request = format!(
        "GET {path} HTTP/1.1\r\nHost: {host}\r\nConnection: close\r\nUser-Agent: Tenth/0.3.3\r\n\r\n"
    );
    use std::io::Write;
    stream
        .write_all(request.as_bytes())
        .map_err(|e| format!("写入请求失败: {e}"))?;
    http_read_response(&mut stream)
}

/// 发送 HTTP POST 请求（Content-Type: text/plain），返回响应 body
pub fn http_post_impl(url: &str, body: &str) -> Result<String, String> {
    let (host, port, path) = parse_http_url(url)?;
    let addr = format!("{host}:{port}");
    let mut stream = std::net::TcpStream::connect(&addr)
        .map_err(|e| format!("连接失败: {e}"))?;
    let timeout = std::time::Duration::from_secs(10);
    stream.set_read_timeout(Some(timeout)).ok();
    stream.set_write_timeout(Some(timeout)).ok();
    let body_bytes = body.as_bytes();
    let request = format!(
        "POST {path} HTTP/1.1\r\nHost: {host}\r\nConnection: close\r\nContent-Type: text/plain\r\nContent-Length: {len}\r\nUser-Agent: Tenth/0.3.3\r\n\r\n",
        len = body_bytes.len()
    );
    use std::io::Write;
    stream
        .write_all(request.as_bytes())
        .map_err(|e| format!("写入请求头失败: {e}"))?;
    stream
        .write_all(body_bytes)
        .map_err(|e| format!("写入请求体失败: {e}"))?;
    http_read_response(&mut stream)
}
