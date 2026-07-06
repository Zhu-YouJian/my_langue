//! 原生函数注册：`call_named_fn`。
//!
//! 从 `interpreter.rs` 第 3255-4489 行迁移而来。包含所有内置原生函数的分派：
//! - I/O：println / to_string / type_name / format / parse_int / parse_float
//! - 执行控制：with_step_limit / with_timeout_ms / is_timeout
//! - 张量与自动微分：tensor / start_grad / new_grad / stop_grad / zero_grad /
//!   param / backward / grad / cross_entropy / abs / sqrt / sin / cos / ln / pow /
//!   rand / randn / randn_f32 / rand_f32 / zeros_f32 / ones_f32 / zeros / ones /
//!   to_float / f64_bits / f64_from_bits / tensor_from_vec
//! - 序列化：save_weights / load_weights / json_encode / json_encode_pretty / json_decode
//! - 文件系统（H-2 沙箱校验）：read_file / write_file / write_bytes / read_bytes /
//!   path_join / path_exists / path_is_file / path_is_dir / mkdir / list_dir /
//!   file_size / remove_file / copy_file / rename_file
//! - 容器构造：Vec::new / HashMap::new
//! - 编译：compile_host / compile_program
//! - 时间：time_now / time_now_ms / time_date / time_time / time_datetime / time_sleep_ms
//! - 随机：random_int / random_float
//! - 数学：math_tan / math_asin / math_acos / math_atan / math_atan2 /
//!   math_sinh / math_cosh / math_tanh / math_log10 / math_log2 / math_exp /
//!   math_pow / math_floor / math_ceil / math_round
//! - CLI：cli_args_count / cli_arg
//!
//! JSON 与日期函数调用 `super::json` / `super::datetime` 子模块中的安全版本
//! （带 H-6 深度闸门与转义状态机修复）。

use std::collections::HashMap;
use std::rc::Rc;
use std::cell::RefCell;
use crate::error::{TenthError, TenthResult};
use crate::runtime::value::Value;
use crate::runtime::tensor::Tensor;
use crate::runtime::autodiff::Tape;
use super::json::{json_encode_value, json_encode_value_pretty, json_decode_string};
use super::datetime::days_to_date;

/// 构造 Result::Ok(value)
fn ok_result(value: Value) -> Value {
    Value::Enum {
        enum_name: "Result".to_string(),
        variant: "Ok".to_string(),
        fields: Rc::new(RefCell::new(vec![("_0".to_string(), value)])),
    }
}

/// 构造 Result::Err(message)
fn err_result(msg: impl Into<String>) -> Value {
    Value::Enum {
        enum_name: "Result".to_string(),
        variant: "Err".to_string(),
        fields: Rc::new(RefCell::new(vec![("_0".to_string(), Value::String(msg.into()))])),
    }
}

// —— HTTP 客户端辅助函数（手写 HTTP/1.1，与 main.rs 双侧对齐）——

/// 解析 URL，返回 (host, port, path)
///
/// 支持 `http://host:port/path` 与 `http://host/path`（默认端口 80）。
/// HTTPS 不支持：`https://` 开头返回 Err。
fn parse_http_url(url: &str) -> Result<(String, u16, String), String> {
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
fn decode_chunked(data: &[u8]) -> Result<Vec<u8>, String> {
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
fn http_read_response(stream: &mut std::net::TcpStream) -> Result<String, String> {
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
fn http_get_impl(url: &str) -> Result<String, String> {
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
fn http_post_impl(url: &str, body: &str) -> Result<String, String> {
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

impl super::Interpreter {
    pub(super) fn call_named_fn(
        &mut self, name: &str, args: &[Value], _span: &crate::lexer::token::Span,
    ) -> TenthResult<Option<Value>> {
        match name {
            "println" => {
                for arg in args {
                    print!("{}", arg);
                }
                println!();
                return Ok(Some(Value::Unit));
            }
            // —— I/O 原语：stderr + stdin ——
            "eprint" => {
                use std::io::Write;
                let stderr = std::io::stderr();
                let mut handle = stderr.lock();
                for arg in args { write!(handle, "{}", arg).ok(); }
                return Ok(Some(Value::Unit));
            }
            "eprintln" => {
                use std::io::Write;
                let stderr = std::io::stderr();
                let mut handle = stderr.lock();
                for arg in args { write!(handle, "{}", arg).ok(); }
                writeln!(handle).ok();
                return Ok(Some(Value::Unit));
            }
            "read_line" => {
                use std::io::BufRead;
                let stdin = std::io::stdin();
                let mut line = String::new();
                match stdin.lock().read_line(&mut line) {
                    Ok(0) => return Ok(Some(err_result("EOF"))),
                    Ok(_) => {
                        if line.ends_with('\n') { line.pop(); if line.ends_with('\r') { line.pop(); } }
                        return Ok(Some(ok_result(Value::String(line))));
                    }
                    Err(e) => return Ok(Some(err_result(format!("读取输入失败: {e}")))),
                }
            }
            // —— 环境变量 + 进程控制 ——
            "env_get" => {
                if let Some(Value::String(name)) = args.first() {
                    match std::env::var(name) {
                        Ok(val) => return Ok(Some(ok_result(Value::String(val)))),
                        Err(_) => return Ok(Some(err_result("环境变量不存在"))),
                    }
                }
                return Ok(Some(err_result("env_get 需要 1 个 String 参数")));
            }
            "env_set" => {
                if args.len() >= 2 {
                    if let (Value::String(name), Value::String(val)) = (&args[0], &args[1]) {
                        // Rust 2024 edition: set_var is unsafe
                        unsafe { std::env::set_var(name, val); }
                    }
                }
                return Ok(Some(Value::Unit));
            }
            "exit" => {
                let code = if let Some(Value::Int(c)) = args.first() { *c } else { 0 };
                std::process::exit(code as i32);
            }
            // —— TCP 网络原语（句柄表方案，handle 1-based，0 表示无效）——
            "tcp_connect" => {
                if args.len() < 2 {
                    return Ok(Some(err_result("tcp_connect 需要 (String, i64) 参数")));
                }
                if let (Value::String(host), Value::Int(port)) = (&args[0], &args[1]) {
                    let addr = format!("{}:{}", host, port);
                    match std::net::TcpStream::connect(&addr) {
                        Ok(stream) => {
                            self.tcp_streams.push(Some(stream));
                            let handle = self.tcp_streams.len() as i64; // 1-based
                            return Ok(Some(ok_result(Value::Int(handle))));
                        }
                        Err(e) => return Ok(Some(err_result(format!("连接失败: {e}")))),
                    }
                }
                return Ok(Some(err_result("tcp_connect 需要 (String, i64) 参数")));
            }
            "tcp_read" => {
                if args.len() < 2 {
                    return Ok(Some(err_result("tcp_read 需要 (i64, i64) 参数")));
                }
                if let (Value::Int(handle), Value::Int(n)) = (&args[0], &args[1]) {
                    let idx = *handle as usize;
                    if idx == 0 || idx > self.tcp_streams.len() {
                        return Ok(Some(err_result("无效的句柄")));
                    }
                    let max = (*n).max(0).min(65536) as usize;
                    if let Some(ref mut stream) = self.tcp_streams[idx - 1] {
                        use std::io::Read;
                        let mut buf = vec![0u8; max];
                        match stream.read(&mut buf) {
                            Ok(0) => {
                                // EOF：返回空 Vec
                                return Ok(Some(ok_result(Value::Vec(Rc::new(RefCell::new(Vec::new()))))));
                            }
                            Ok(read_n) => {
                                let bytes: Vec<Value> = buf[..read_n]
                                    .iter()
                                    .map(|b| Value::Int(*b as i64))
                                    .collect();
                                return Ok(Some(ok_result(Value::Vec(Rc::new(RefCell::new(bytes))))));
                            }
                            Err(e) => return Ok(Some(err_result(format!("读取失败: {e}")))),
                        }
                    } else {
                        return Ok(Some(err_result("连接已关闭")));
                    }
                }
                return Ok(Some(err_result("tcp_read 需要 (i64, i64) 参数")));
            }
            "tcp_write" => {
                if args.len() < 2 {
                    return Ok(Some(err_result("tcp_write 需要 (i64, Vec<i64>) 参数")));
                }
                if let (Value::Int(handle), Value::Vec(data)) = (&args[0], &args[1]) {
                    let idx = *handle as usize;
                    if idx == 0 || idx > self.tcp_streams.len() {
                        return Ok(Some(err_result("无效的句柄")));
                    }
                    let bytes: Vec<u8> = data
                        .borrow()
                        .iter()
                        .map(|x| match x {
                            Value::Int(b) => *b as u8,
                            _ => 0,
                        })
                        .collect();
                    if let Some(ref mut stream) = self.tcp_streams[idx - 1] {
                        use std::io::Write;
                        match stream.write_all(&bytes) {
                            Ok(_) => return Ok(Some(ok_result(Value::Int(bytes.len() as i64)))),
                            Err(e) => return Ok(Some(err_result(format!("写入失败: {e}")))),
                        }
                    } else {
                        return Ok(Some(err_result("连接已关闭")));
                    }
                }
                return Ok(Some(err_result("tcp_write 需要 (i64, Vec<i64>) 参数")));
            }
            "tcp_close" => {
                if let Some(Value::Int(handle)) = args.first() {
                    let idx = *handle as usize;
                    if idx > 0 && idx <= self.tcp_streams.len() {
                        self.tcp_streams[idx - 1] = None; // drop 自动关闭
                    }
                }
                return Ok(Some(Value::Unit));
            }
            "tcp_set_timeout" => {
                if args.len() >= 2 {
                    if let (Value::Int(handle), Value::Int(ms)) = (&args[0], &args[1]) {
                        let idx = *handle as usize;
                        if idx > 0 && idx <= self.tcp_streams.len() {
                            if let Some(ref mut stream) = self.tcp_streams[idx - 1] {
                                let dur = std::time::Duration::from_millis(*ms as u64);
                                stream.set_read_timeout(Some(dur)).ok();
                                stream.set_write_timeout(Some(dur)).ok();
                            }
                        }
                    }
                }
                return Ok(Some(Value::Unit));
            }
            // —— HTTP 客户端原语（手写 HTTP/1.1，10 秒默认超时）——
            "http_get" => {
                if let Some(Value::String(url)) = args.first() {
                    match http_get_impl(url) {
                        Ok(body) => return Ok(Some(ok_result(Value::String(body)))),
                        Err(e) => return Ok(Some(err_result(e))),
                    }
                }
                return Ok(Some(err_result("http_get 需要 1 个 String 参数")));
            }
            "http_post" => {
                if args.len() >= 2 {
                    if let (Value::String(url), Value::String(body)) = (&args[0], &args[1]) {
                        match http_post_impl(url, body) {
                            Ok(resp) => return Ok(Some(ok_result(Value::String(resp)))),
                            Err(e) => return Ok(Some(err_result(e))),
                        }
                    }
                    return Ok(Some(err_result("http_post 需要 (String, String) 参数")));
                }
                return Ok(Some(err_result("http_post 需要 (String, String) 参数")));
            }
            // 论文 T37 修复第二批：补齐解释器缺失的 print/to_f64/to_f32（与 VM main.rs 对齐）
            "print" => {
                for arg in args {
                    print!("{}", arg);
                }
                return Ok(Some(Value::Unit));
            }
            "to_string" => {
                if let Some(arg) = args.first() {
                    return Ok(Some(Value::String(self.value_to_string(arg))));
                }
                return Ok(Some(Value::String(String::new())));
            }
            "type_name" => {
                if let Some(arg) = args.first() {
                    let tn = match arg {
                        Value::Int(_) => "int",
                        Value::Float(_) => "float",
                        Value::Bool(_) => "bool",
                        Value::String(_) => "string",
                        Value::Unit => "unit",
                        Value::Vec(_) => "vec",
                        Value::Array(_) => "array",
                        Value::Map(_) => "map",
                        Value::Tuple(_) => "tuple",
                        Value::Closure { .. } => "closure",
                        Value::FnRef { .. } => "fn",
                        _ => "unknown",
                    };
                    return Ok(Some(Value::String(tn.to_string())));
                }
                return Ok(Some(Value::String("unknown".to_string())));
            }
            "with_step_limit" => {
                // with_step_limit(limit: Int, fn) -> runs fn with a step budget.
                // Returns whatever fn returns, or () on timeout.
                if args.len() < 2 {
                    return Err(TenthError::RuntimeError {
                        message: "with_step_limit(limit, fn) 需要 2 个参数".into(),
                    });
                }
                let limit = args[0].as_int().ok_or_else(|| TenthError::RuntimeError {
                    message: "with_step_limit 的第一个参数必须是整数步数".into(),
                })?;
                let closure = args[1].clone();
                // Save and set the budget.
                let saved_budget = self.step_budget;
                let saved_deadline = self.deadline_ms;
                self.step_budget = Some(limit.max(0) as u64);
                self.deadline_ms = None;
                let result = self.eval_call(&closure, &[], &crate::lexer::token::Span { line: 0, col: 0 });
                // Restore previous budget so the limit is scoped to this call.
                self.step_budget = saved_budget;
                self.deadline_ms = saved_deadline;
                return match result {
                    Ok(v) => Ok(v),
                    Err(TenthError::Timeout { .. }) => Ok(Some(Value::Unit)),
                    Err(e) => Err(e),
                };
            }
            "with_timeout_ms" => {
                // with_timeout_ms(ms: Int, fn) -> runs fn with a wall-clock deadline.
                if args.len() < 2 {
                    return Err(TenthError::RuntimeError {
                        message: "with_timeout_ms(ms, fn) 需要 2 个参数".into(),
                    });
                }
                let ms = args[0].as_int().ok_or_else(|| TenthError::RuntimeError {
                    message: "with_timeout_ms 的第一个参数必须是整数毫秒".into(),
                })?;
                let closure = args[1].clone();
                let now = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_millis())
                    .unwrap_or(0);
                let saved_budget = self.step_budget;
                let saved_deadline = self.deadline_ms;
                // Use a large step budget as the tick vehicle; the deadline
                // check inside tick() does the actual time comparison.
                self.step_budget = Some(u64::MAX);
                self.deadline_ms = Some(now + (ms.max(0) as u128));
                let result = self.eval_call(&closure, &[], &crate::lexer::token::Span { line: 0, col: 0 });
                self.step_budget = saved_budget;
                self.deadline_ms = saved_deadline;
                return match result {
                    Ok(v) => Ok(v),
                    Err(TenthError::Timeout { .. }) => Ok(Some(Value::Unit)),
                    Err(e) => Err(e),
                };
            }
            "is_timeout" => {
                // is_timeout(result) -> true if the result is the unit value
                // returned by a timed-out with_step_limit/with_timeout_ms call.
                // This is a best-effort sentinel check; for precise control
                // prefer matching on the returned value directly.
                if let Some(arg) = args.first() {
                    return Ok(Some(Value::Bool(matches!(arg, Value::Unit))));
                }
                return Ok(Some(Value::Bool(false)));
            }
            "tensor" => {
                if let Some(arg) = args.first() {
                    return Ok(Some(arg.clone()));
                }
                return Ok(Some(Value::Unit));
            }
            "start_grad" => {
                self.tape = Some(Tape::new());
                self.recording = true;
                return Ok(Some(Value::Unit));
            }
            "param" => {
                if let Some(Value::Tensor(t)) = args.first() {
                    // Register this tensor as a leaf parameter on the tape
                    if let Some(ref mut tape) = self.tape {
                        let node_id = tape.input(t.clone());
                        t.borrow_mut().tape_id = Some(node_id);
                    }
                    return Ok(Some(Value::Tensor(t.clone())));
                }
                return Err(TenthError::RuntimeError {
                    message: "param() 期望一个张量参数".into(),
                });
            }
            "backward" => {
                if let Some(Value::Tensor(loss)) = args.first() {
                    if let (Some(tape), Some(loss_id)) = (&self.tape, loss.borrow().tape_id) {
                        // 护城河 F：包裹 backward 错误，附加 formal_explain 根因分析
                        match tape.backward(loss_id) {
                            Ok(()) => return Ok(Some(Value::Unit)),
                            Err(e) => {
                                let causes = tape.formal_explain(loss_id, &[], &[]);
                                let explanations: Vec<String> = causes.iter().map(|c| c.explanation.clone()).collect();
                                self.last_explanation = explanations.clone();
                                let context = crate::error::TapeErrorContext {
                                    tape_node_id: loss_id,
                                    op: "backward".to_string(),
                                    expected_shape: Vec::new(),
                                    actual_shape: Vec::new(),
                                };
                                let root_cause_msg = if explanations.is_empty() {
                                    format!("{}", e)
                                } else {
                                    format!(
                                        "{}\n根因分析（formal_explain）：\n{}",
                                        e,
                                        explanations.iter()
                                            .map(|s| format!("  - {}", s))
                                            .collect::<Vec<_>>()
                                            .join("\n")
                                    )
                                };
                                return Err(TenthError::ShapeMismatch {
                                    context,
                                    message: root_cause_msg,
                                });
                            }
                        }
                    }
                    return Ok(Some(Value::Unit));
                }
                return Err(TenthError::RuntimeError {
                    message: "backward() 期望一个张量参数".into(),
                });
            }
            // 护城河 F：explain_error() — 返回上一次 backward 失败的根因说明列表
            "explain_error" => {
                let explanations = std::mem::take(&mut self.last_explanation);
                let values: Vec<Value> = explanations.into_iter().map(Value::String).collect();
                return Ok(Some(Value::Vec(Rc::new(RefCell::new(values)))));
            }
            "stop_grad" => {
                self.recording = false;
                return Ok(Some(Value::Unit));
            }
            "new_grad" => {
                self.tape = Some(Tape::new());
                self.recording = true;
                return Ok(Some(Value::Unit));
            }
            "zero_grad" => {
                if let Some(ref tape) = self.tape {
                    tape.zero_grad();
                }
                return Ok(Some(Value::Unit));
            }
            "abs" => {
                if let Some(arg) = args.first() {
                    return Ok(Some(match arg {
                        Value::Int(n) => Value::Int(n.abs()),
                        Value::Float(n) => Value::Float(n.abs()),
                        _ => return Err(TenthError::RuntimeError {
                            message: "abs() 期望一个数值参数".into(),
                        }),
                    }));
                }
                return Err(TenthError::RuntimeError {
                    message: "abs() 期望 1 个参数".into(),
                });
            }
            "sqrt" => {
                if let Some(arg) = args.first() {
                    let n = arg.as_float().ok_or_else(|| TenthError::RuntimeError {
                        message: "sqrt() 期望一个数值参数".into(),
                    })?;
                    return Ok(Some(Value::Float(n.sqrt())));
                }
                return Err(TenthError::RuntimeError {
                    message: "sqrt() 期望 1 个参数".into(),
                });
            }
            "to_float" => {
                if let Some(arg) = args.first() {
                    return Ok(Some(match arg {
                        Value::Int(n) => Value::Float(*n as f64),
                        Value::Float(f) => Value::Float(*f),
                        _ => return Err(TenthError::RuntimeError {
                            message: "to_float() 期望一个数值参数".into(),
                        }),
                    }));
                }
                return Err(TenthError::RuntimeError {
                    message: "to_float() 期望 1 个参数".into(),
                });
            }
            // to_f64 — 与 to_float 同语义（VM 侧 main.rs:1112 已注册，这里补齐解释器对齐）
            "to_f64" => {
                if let Some(arg) = args.first() {
                    return Ok(Some(match arg {
                        Value::Int(n) => Value::Float(*n as f64),
                        Value::Float(f) => Value::Float(*f),
                        Value::Float32(f) => Value::Float(*f as f64),
                        _ => return Err(TenthError::RuntimeError {
                            message: "to_f64() 期望一个数值参数".into(),
                        }),
                    }));
                }
                return Err(TenthError::RuntimeError {
                    message: "to_f64() 期望 1 个参数".into(),
                });
            }
            // to_f32 — 转 f32（VM 侧 main.rs:1120 已注册）
            "to_f32" => {
                if let Some(arg) = args.first() {
                    return Ok(Some(match arg {
                        Value::Int(n) => Value::Float32(*n as f32),
                        Value::Float(f) => Value::Float32(*f as f32),
                        Value::Float32(f) => Value::Float32(*f),
                        _ => return Err(TenthError::RuntimeError {
                            message: "to_f32() 期望一个数值参数".into(),
                        }),
                    }));
                }
                return Err(TenthError::RuntimeError {
                    message: "to_f32() 期望 1 个参数".into(),
                });
            }
            "f64_bits" => {
                if let Some(arg) = args.first() {
                    let f = arg.as_float().ok_or_else(|| TenthError::RuntimeError {
                        message: "f64_bits() 期望一个 f64 参数".into(),
                    })?;
                    return Ok(Some(Value::Int(f.to_bits() as i64)));
                }
                return Err(TenthError::RuntimeError {
                    message: "f64_bits() 期望 1 个参数".into(),
                });
            }
            "f64_from_bits" => {
                if let Some(arg) = args.first() {
                    let n = arg.as_int().ok_or_else(|| TenthError::RuntimeError {
                        message: "f64_from_bits() 期望一个 i64 参数".into(),
                    })?;
                    return Ok(Some(Value::Float(f64::from_bits(n as u64))));
                }
                return Err(TenthError::RuntimeError {
                    message: "f64_from_bits() 期望 1 个参数".into(),
                });
            }
            "tensor_from_vec" => {
                if args.len() >= 3 {
                    if let (Value::Vec(items), Value::Int(rows), Value::Int(cols)) = (&args[0], &args[1], &args[2]) {
                        // 按 Vec 内元素 dtype 判断：含 Float32 → f32 Tensor
                        let has_f32 = items.borrow().iter().any(|v| matches!(v, Value::Float32(_)));
                        if has_f32 {
                            let data: Vec<f32> = items.borrow().iter().map(|v| v.as_f32().unwrap_or(0.0)).collect();
                            let tensor = Tensor::from_vec_f32(data, vec![*rows as usize, *cols as usize]);
                            return Ok(Some(Value::Tensor(Rc::new(RefCell::new(tensor)))));
                        }
                        let data: Vec<f64> = items.borrow().iter().map(|v| v.as_float().unwrap_or(0.0)).collect();
                        let tensor = Tensor::from_vec(data, vec![*rows as usize, *cols as usize]);
                        return Ok(Some(Value::Tensor(Rc::new(RefCell::new(tensor)))));
                    }
                }
                return Err(TenthError::RuntimeError {
                    message: "tensor_from_vec(vec, rows, cols) 期望一个 Vec 和两个整数".into(),
                });
            }
            "sin" => {
                if let Some(arg) = args.first() {
                    let n = arg.as_float().ok_or_else(|| TenthError::RuntimeError {
                        message: "sin() 期望一个数值参数".into(),
                    })?;
                    return Ok(Some(Value::Float(n.sin())));
                }
                return Err(TenthError::RuntimeError {
                    message: "sin() 期望 1 个参数".into(),
                });
            }
            "cos" => {
                if let Some(arg) = args.first() {
                    let n = arg.as_float().ok_or_else(|| TenthError::RuntimeError {
                        message: "cos() 期望一个数值参数".into(),
                    })?;
                    return Ok(Some(Value::Float(n.cos())));
                }
                return Err(TenthError::RuntimeError {
                    message: "cos() 期望 1 个参数".into(),
                });
            }
            "ln" => {
                if let Some(arg) = args.first() {
                    let n = arg.as_float().ok_or_else(|| TenthError::RuntimeError {
                        message: "ln() 期望一个数值参数".into(),
                    })?;
                    if n <= 0.0 {
                        return Err(TenthError::RuntimeError {
                            message: "ln() 参数必须 > 0".into(),
                        });
                    }
                    return Ok(Some(Value::Float(n.ln())));
                }
                return Err(TenthError::RuntimeError {
                    message: "ln() 期望 1 个参数".into(),
                });
            }
            "pow" => {
                if args.len() >= 2 {
                    let base = args[0].as_float().ok_or_else(|| TenthError::RuntimeError {
                        message: "pow() 期望数值参数".into(),
                    })?;
                    let exp = args[1].as_float().ok_or_else(|| TenthError::RuntimeError {
                        message: "pow() 期望数值参数".into(),
                    })?;
                    return Ok(Some(Value::Float(base.powf(exp))));
                }
                return Err(TenthError::RuntimeError {
                    message: "pow() 期望 2 个参数".into(),
                });
            }
            "cross_entropy" => {
                if args.len() >= 2 {
                    if let (Value::Tensor(logits), Value::Tensor(target)) = (&args[0], &args[1]) {
                        let logits_data = logits.borrow();
                        let target_data = target.borrow();

                        // Compute softmax along last axis
                        let sm = logits_data.softmax().ok_or_else(|| {
                            TenthError::RuntimeError { message: "cross_entropy 中 softmax 失败".into() }
                        })?;

                        // CE loss: -mean(sum(target * log(softmax + ε)))
                        let eps = 1e-10;
                        let sm_data = sm.data.as_standard_layout().to_owned();
                        let tgt_flat = target_data.data.as_standard_layout().to_owned();
                        let sm_slice = sm_data.as_slice().unwrap_or(&[]);
                        let tgt_slice = tgt_flat.as_slice().unwrap_or(&[]);

                        let mut loss_val = 0.0f64;
                        let n = sm_slice.len() as f64;
                        for i in 0..sm_slice.len().min(tgt_slice.len()) {
                            let p = sm_slice[i].max(eps);
                            loss_val -= tgt_slice[i] * p.ln();
                        }
                        loss_val /= n.max(1.0); // mean over all elements

                        let loss_tensor = Tensor::from_vec(vec![loss_val], vec![1]);
                        let result = Rc::new(RefCell::new(loss_tensor));

                        if self.recording {
                            // Record: input_tensors = [logits, softmax, target]
                            let sm_rc = Rc::new(RefCell::new(sm));
                            if let Some(ref mut tape) = self.tape {
                                let logits_id = logits.borrow().tape_id
                                    .unwrap_or_else(|| tape.input(logits.clone()));
                                let _sm_id = tape.input(sm_rc.clone());
                                // Create CrossEntropy node manually
                                let node_id = tape.cross_entropy(
                                    logits_id, logits.clone(),
                                    sm_rc,
                                    target.clone(),
                                    result.clone(),
                                );
                                result.borrow_mut().tape_id = Some(node_id);
                            }
                        }

                        return Ok(Some(Value::Tensor(result)));
                    }
                }
                return Err(TenthError::RuntimeError {
                    message: "cross_entropy(logits, target) 期望两个张量".into(),
                });
            }
            "select" => {
                // select(cond, then, else) — 逐元素条件选择原语（论文 T47/T48/T50）
                // 支持广播；cond 非 0 视为 true。可微：d_then = grad*mask, d_else = grad*(1-mask)
                if args.len() < 3 {
                    return Err(TenthError::RuntimeError {
                        message: "select(cond, then, else) 期望三个参数".into(),
                    });
                }
                let (cond, then, else_) = match (&args[0], &args[1], &args[2]) {
                    (Value::Tensor(c), Value::Tensor(t), Value::Tensor(e)) => (c.clone(), t.clone(), e.clone()),
                    _ => return Err(TenthError::RuntimeError {
                        message: "select(cond, then, else) 期望三个张量参数".into(),
                    }),
                };
                let result_tensor = Tensor::select(&cond.borrow(), &then.borrow(), &else_.borrow())
                    .map_err(|msg| TenthError::RuntimeError { message: msg })?;
                let result = Rc::new(RefCell::new(result_tensor));
                if self.recording {
                    if let Some(ref mut tape) = self.tape {
                        let then_id = then.borrow().tape_id;
                        let else_id = else_.borrow().tape_id;
                        let node_id = tape.select(then_id, else_id, cond.clone(), then.clone(), else_.clone(), result.clone());
                        result.borrow_mut().tape_id = Some(node_id);
                    }
                }
                return Ok(Some(Value::Tensor(result)));
            }
            "scatter" => {
                // scatter(base, dim, index, src) — 不可变散布原语
                // 支持任意 dim + 多维 index/src（PyTorch 对齐）。
                // 可微：d_src=gather(grad,index,dim), d_base=grad 但 index 指向位置置 0
                if args.len() < 4 {
                    return Err(TenthError::RuntimeError {
                        message: "scatter(base, dim, index, src) 期望四个参数".into(),
                    });
                }
                let dim = args[1].as_int().unwrap_or(0) as usize;
                let (base, index, src) = match (&args[0], &args[2], &args[3]) {
                    (Value::Tensor(b), Value::Tensor(i), Value::Tensor(s)) => (b.clone(), i.clone(), s.clone()),
                    _ => return Err(TenthError::RuntimeError {
                        message: "scatter(base, dim, index, src) 期望 base/index/src 为张量".into(),
                    }),
                };
                let result_tensor = Tensor::scatter(&base.borrow(), dim, &index.borrow(), &src.borrow())
                    .map_err(|msg| TenthError::RuntimeError { message: msg })?;
                let result = Rc::new(RefCell::new(result_tensor));
                if self.recording {
                    if let Some(ref mut tape) = self.tape {
                        let base_id = base.borrow().tape_id;
                        let src_id = src.borrow().tape_id;
                        let node_id = tape.scatter(base_id, src_id, base.clone(), src.clone(), index.clone(), result.clone(), dim);
                        result.borrow_mut().tape_id = Some(node_id);
                    }
                }
                return Ok(Some(Value::Tensor(result)));
            }
            "gather" => {
                // gather(base, dim, index) — 沿 dim 维按 index 取值，与 PyTorch gather 对齐
                // out[i,j,...] = base[index[i,j,...], j, ...]  (dim=0)
                // 可微：d_base = zeros_like(base); d_base[actual] += grad[idx] (scatter-add 语义)
                // index 不可微
                if args.len() < 3 {
                    return Err(TenthError::RuntimeError {
                        message: "gather(base, dim, index) 期望三个参数".into(),
                    });
                }
                let dim = args[1].as_int().unwrap_or(0) as usize;
                let (base, index) = match (&args[0], &args[2]) {
                    (Value::Tensor(b), Value::Tensor(i)) => (b.clone(), i.clone()),
                    _ => return Err(TenthError::RuntimeError {
                        message: "gather(base, dim, index) 期望 base/index 为张量".into(),
                    }),
                };
                let result_tensor = Tensor::gather(&base.borrow(), dim, &index.borrow())
                    .map_err(|msg| TenthError::RuntimeError { message: msg })?;
                let result = Rc::new(RefCell::new(result_tensor));
                if self.recording {
                    if let Some(ref mut tape) = self.tape {
                        let base_id = base.borrow().tape_id;
                        let node_id = tape.gather(base_id, base.clone(), index.clone(), result.clone(), dim);
                        result.borrow_mut().tape_id = Some(node_id);
                    }
                }
                return Ok(Some(Value::Tensor(result)));
            }
            "grad" => {
                if let Some(Value::Tensor(param)) = args.first() {
                    let p = param.borrow();
                    if let Some(ref grad) = p.grad {
                        let grad_tensor = Tensor::from_tensor_data(grad.clone());
                        return Ok(Some(Value::Tensor(Rc::new(RefCell::new(grad_tensor)))));
                    }
                    // No gradient → return zeros matching param dtype (Phase 5.4：f32 张量返回 f32 zeros)
                    let shape = p.shape();
                    let zeros = Tensor::zeros_with_dtype(&shape, p.dtype());
                    return Ok(Some(Value::Tensor(Rc::new(RefCell::new(zeros)))));
                }
                return Err(TenthError::RuntimeError {
                    message: "grad() 期望一个张量参数".into(),
                });
            }
            "rand" => {
                let shape: Vec<usize> = args.iter()
                    .map(|a| a.as_int().unwrap_or(1) as usize)
                    .collect();
                let t = Tensor::rand(&shape);
                return Ok(Some(Value::Tensor(Rc::new(RefCell::new(t)))));
            }
            "randn" => {
                let shape: Vec<usize> = args.iter()
                    .map(|a| a.as_int().unwrap_or(1) as usize)
                    .collect();
                let t = Tensor::randn(&shape);
                return Ok(Some(Value::Tensor(Rc::new(RefCell::new(t)))));
            }
            "randn_f32" => {
                let shape: Vec<usize> = args.iter()
                    .map(|a| a.as_int().unwrap_or(1) as usize)
                    .collect();
                let t = Tensor::randn_f32(&shape);
                return Ok(Some(Value::Tensor(Rc::new(RefCell::new(t)))));
            }
            // Phase 5.5：补全 f32 构造函数
            "rand_f32" => {
                let shape: Vec<usize> = args.iter()
                    .map(|a| a.as_int().unwrap_or(1) as usize)
                    .collect();
                let t = Tensor::rand_f32(&shape);
                return Ok(Some(Value::Tensor(Rc::new(RefCell::new(t)))));
            }
            "zeros_f32" => {
                let shape: Vec<usize> = args.iter()
                    .map(|a| a.as_int().unwrap_or(1) as usize)
                    .collect();
                let t = Tensor::zeros_f32(&shape);
                return Ok(Some(Value::Tensor(Rc::new(RefCell::new(t)))));
            }
            "ones_f32" => {
                let shape: Vec<usize> = args.iter()
                    .map(|a| a.as_int().unwrap_or(1) as usize)
                    .collect();
                let t = Tensor::ones_f32(&shape);
                return Ok(Some(Value::Tensor(Rc::new(RefCell::new(t)))));
            }
            "zeros" => {
                let shape: Vec<usize> = args.iter()
                    .map(|a| a.as_int().unwrap_or(1) as usize)
                    .collect();
                let t = Tensor::zeros(&shape);
                return Ok(Some(Value::Tensor(Rc::new(RefCell::new(t)))));
            }
            "ones" => {
                let shape: Vec<usize> = args.iter()
                    .map(|a| a.as_int().unwrap_or(1) as usize)
                    .collect();
                let t = Tensor::ones(&shape);
                return Ok(Some(Value::Tensor(Rc::new(RefCell::new(t)))));
            }
            "save_weights" => {
                if args.len() >= 2 {
                    if let Value::String(path) = &args[0] {
                        // H-2: 沙箱校验
                        let resolved = if let Some(ref sb) = self.fs_sandbox {
                            match sb.check_write(path) {
                                Ok(p) => p,
                                Err(e) => return Err(TenthError::RuntimeError { message: e }),
                            }
                        } else {
                            std::path::PathBuf::from(path)
                        };
                        // args[1] can be Array or Vec of tensors
                        let tensors: &Rc<RefCell<Vec<Value>>> = match &args[1] {
                            Value::Vec(v) => v,
                            Value::Array(a) => a,
                            _ => {
                                return Err(TenthError::RuntimeError {
                                    message: "save_weights 期望一个张量列表".into(),
                                });
                            }
                        };
                            let tensors_ref = tensors.borrow();
                            let mut bytes: Vec<u8> = Vec::new();
                            // Header: number of tensors (i32)
                            bytes.extend(&(tensors_ref.len() as i32).to_le_bytes());
                            for val in tensors_ref.iter() {
                                // Unwrap Shared wrapper (Vec::push wraps elements in Shared)
                                let tensor_rc = match val {
                                    Value::Tensor(t) => Some(t.clone()),
                                    Value::Shared(rc) => {
                                        if let Value::Tensor(t) = &*rc.borrow() {
                                            Some(t.clone())
                                        } else { None }
                                    }
                                    _ => None,
                                };
                                if let Some(t) = tensor_rc {
                                    let t_ref = t.borrow();
                                    let shape = t_ref.shape();
                                    let ndim = shape.len() as i32;
                                    bytes.extend(&ndim.to_le_bytes());
                                    for &d in &shape {
                                        bytes.extend(&(d as i32).to_le_bytes());
                                    }
                                    let flat = t_ref.data.as_standard_layout().to_owned();
                                    if let Some(slice) = flat.as_slice() {
                                        for &x in slice {
                                            bytes.extend(&x.to_le_bytes());
                                        }
                                    }
                                }
                            }
                            let _ = std::fs::write(&resolved, &bytes);
                            return Ok(Some(Value::Unit));
                    }
                }
                return Err(TenthError::RuntimeError {
                    message: "save_weights(路径, 张量列表)".into(),
                });
            }
            "load_weights" => {
                if let Some(Value::String(path)) = args.first() {
                    // H-2: 沙箱校验
                    let resolved = if let Some(ref sb) = self.fs_sandbox {
                        match sb.check_read(path) {
                            Ok(p) => p,
                            Err(e) => return Err(TenthError::RuntimeError { message: e }),
                        }
                    } else {
                        std::path::PathBuf::from(path)
                    };
                    match std::fs::read(&resolved) {
                        Ok(bytes) => {
                            if bytes.len() < 4 {
                                return Err(TenthError::RuntimeError {
                                    message: "load_weights: 文件过短".into(),
                                });
                            }
                            let num = i32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as usize;
                            let mut offset: usize = 4;
                            let mut result: Vec<Value> = Vec::new();
                            for _ in 0..num {
                                if offset + 4 > bytes.len() { break; }
                                let ndim = i32::from_le_bytes([
                                    bytes[offset], bytes[offset+1], bytes[offset+2], bytes[offset+3]
                                ]) as usize;
                                offset += 4;
                                let mut shape = Vec::new();
                                for _ in 0..ndim {
                                    if offset + 4 > bytes.len() { break; }
                                    let d = i32::from_le_bytes([
                                        bytes[offset], bytes[offset+1], bytes[offset+2], bytes[offset+3]
                                    ]) as usize;
                                    shape.push(d);
                                    offset += 4;
                                }
                                let nel: usize = shape.iter().product();
                                let data_len = nel * 8; // f64 = 8 bytes
                                if offset + data_len > bytes.len() { break; }
                                let mut data = Vec::with_capacity(nel);
                                for i in 0..nel {
                                    let start = offset + i * 8;
                                    let val = f64::from_le_bytes([
                                        bytes[start], bytes[start+1], bytes[start+2], bytes[start+3],
                                        bytes[start+4], bytes[start+5], bytes[start+6], bytes[start+7],
                                    ]);
                                    data.push(val);
                                }
                                offset += data_len;
                                result.push(Value::Tensor(Rc::new(RefCell::new(
                                    Tensor::from_vec(data, shape)
                                ))));
                            }
                            return Ok(Some(Value::Vec(Rc::new(RefCell::new(result)))));
                        }
                        Err(e) => return Err(TenthError::RuntimeError {
                            message: format!("load_weights: {}", e),
                        }),
                    }
                }
                return Err(TenthError::RuntimeError {
                    message: "load_weights(路径)".into(),
                });
            }
            "read_file" => {
                if let Some(Value::String(path)) = args.first() {
                    // H-2: 沙箱校验
                    let resolved = if let Some(ref sb) = self.fs_sandbox {
                        match sb.check_read(path) {
                            Ok(p) => p,
                            Err(e) => return Err(TenthError::RuntimeError { message: e }),
                        }
                    } else {
                        std::path::PathBuf::from(path)
                    };
                    match std::fs::read_to_string(&resolved) {
                        Ok(content) => return Ok(Some(Value::String(content))),
                        Err(e) => return Err(TenthError::RuntimeError {
                            message: format!("读取文件失败: {}", e),
                        }),
                    }
                }
                return Err(TenthError::RuntimeError {
                    message: "read_file(路径) 期望一个字符串路径".into(),
                });
            }
            "write_file" => {
                if args.len() >= 2 {
                    if let (Value::String(path), Value::String(content)) = (&args[0], &args[1]) {
                        // H-2: 沙箱校验
                        let resolved = if let Some(ref sb) = self.fs_sandbox {
                            match sb.check_write(path) {
                                Ok(p) => p,
                                Err(e) => return Err(TenthError::RuntimeError { message: e }),
                            }
                        } else {
                            std::path::PathBuf::from(path)
                        };
                        match std::fs::write(&resolved, content) {
                            Ok(()) => return Ok(Some(Value::Unit)),
                            Err(e) => return Err(TenthError::RuntimeError {
                                message: format!("写入文件失败: {}", e),
                            }),
                        }
                    }
                }
                return Err(TenthError::RuntimeError {
                    message: "write_file(路径, 内容) 期望两个字符串参数".into(),
                });
            }
            "write_bytes" => {
                if args.len() >= 2 {
                    if let (Value::String(path), Value::Vec(bytes)) = (&args[0], &args[1]) {
                        // H-2: 沙箱校验
                        let resolved = if let Some(ref sb) = self.fs_sandbox {
                            match sb.check_write(path) {
                                Ok(p) => p,
                                Err(e) => return Err(TenthError::RuntimeError { message: e }),
                            }
                        } else {
                            std::path::PathBuf::from(path)
                        };
                        let data: Vec<u8> = bytes.borrow().iter().filter_map(|v| {
                            if let Value::Int(n) = v { Some(*n as u8) } else { None }
                        }).collect();
                        match std::fs::write(&resolved, &data) {
                            Ok(()) => return Ok(Some(Value::Unit)),
                            Err(e) => return Err(TenthError::RuntimeError {
                                message: format!("写入字节失败: {}", e),
                            }),
                        }
                    }
                }
                return Err(TenthError::RuntimeError {
                    message: "write_bytes(路径, 字节数组) 期望一个字符串和一个字节 Vec".into(),
                });
            }
            "read_bytes" => {
                if let Some(Value::String(path)) = args.first() {
                    // H-2: 沙箱校验
                    let resolved = if let Some(ref sb) = self.fs_sandbox {
                        match sb.check_read(path) {
                            Ok(p) => p,
                            Err(e) => return Err(TenthError::RuntimeError { message: e }),
                        }
                    } else {
                        std::path::PathBuf::from(path)
                    };
                    match std::fs::read(&resolved) {
                        Ok(data) => {
                            let bytes: Vec<Value> = data.iter()
                                .map(|b| Value::Int(*b as i64))
                                .collect();
                            return Ok(Some(Value::Vec(Rc::new(RefCell::new(bytes)))));
                        }
                        Err(e) => return Err(TenthError::RuntimeError {
                            message: format!("读取字节失败: {}", e),
                        }),
                    }
                }
                return Err(TenthError::RuntimeError {
                    message: "read_bytes(路径) 期望一个字符串路径".into(),
                });
            }
            "path_join" => {
                if args.len() >= 2 {
                    if let (Value::String(base), Value::String(rest)) = (&args[0], &args[1]) {
                        let joined = std::path::Path::new(base).join(rest);
                        return Ok(Some(Value::String(joined.to_string_lossy().to_string())));
                    }
                }
                return Err(TenthError::RuntimeError {
                    message: "path_join(基础路径, 子路径) 期望两个字符串参数".into(),
                });
            }
            "path_exists" => {
                if let Some(Value::String(path)) = args.first() {
                    // H-2: 沙箱校验（只读检查）
                    if let Some(ref sb) = self.fs_sandbox {
                        if let Err(e) = sb.check_read(path) {
                            return Err(TenthError::RuntimeError { message: e });
                        }
                    }
                    return Ok(Some(Value::Bool(std::path::Path::new(path).exists())));
                }
                return Err(TenthError::RuntimeError {
                    message: "path_exists(路径) 期望一个字符串路径".into(),
                });
            }
            "path_is_file" => {
                if let Some(Value::String(path)) = args.first() {
                    if let Some(ref sb) = self.fs_sandbox {
                        if let Err(e) = sb.check_read(path) {
                            return Err(TenthError::RuntimeError { message: e });
                        }
                    }
                    return Ok(Some(Value::Bool(std::path::Path::new(path).is_file())));
                }
                return Err(TenthError::RuntimeError {
                    message: "path_is_file(路径) 期望一个字符串路径".into(),
                });
            }
            "path_is_dir" => {
                if let Some(Value::String(path)) = args.first() {
                    if let Some(ref sb) = self.fs_sandbox {
                        if let Err(e) = sb.check_read(path) {
                            return Err(TenthError::RuntimeError { message: e });
                        }
                    }
                    return Ok(Some(Value::Bool(std::path::Path::new(path).is_dir())));
                }
                return Err(TenthError::RuntimeError {
                    message: "path_is_dir(路径) 期望一个字符串路径".into(),
                });
            }
            "mkdir" => {
                if let Some(Value::String(path)) = args.first() {
                    // H-2: 沙箱校验（写操作）
                    let resolved = if let Some(ref sb) = self.fs_sandbox {
                        match sb.check_write(path) {
                            Ok(p) => p,
                            Err(e) => return Err(TenthError::RuntimeError { message: e }),
                        }
                    } else {
                        std::path::PathBuf::from(path)
                    };
                    match std::fs::create_dir_all(&resolved) {
                        Ok(()) => return Ok(Some(Value::Unit)),
                        Err(e) => return Err(TenthError::RuntimeError {
                            message: format!("创建目录失败: {}", e),
                        }),
                    }
                }
                return Err(TenthError::RuntimeError {
                    message: "mkdir(路径) 期望一个字符串路径".into(),
                });
            }
            "list_dir" => {
                if let Some(Value::String(path)) = args.first() {
                    // H-2: 沙箱校验
                    let resolved = if let Some(ref sb) = self.fs_sandbox {
                        match sb.check_read(path) {
                            Ok(p) => p,
                            Err(e) => return Err(TenthError::RuntimeError { message: e }),
                        }
                    } else {
                        std::path::PathBuf::from(path)
                    };
                    match std::fs::read_dir(&resolved) {
                        Ok(entries) => {
                            let names: Vec<Value> = entries.filter_map(|e| {
                                e.ok().map(|entry| Value::String(entry.file_name().to_string_lossy().to_string()))
                            }).collect();
                            return Ok(Some(Value::Vec(Rc::new(RefCell::new(names)))));
                        }
                        Err(e) => return Err(TenthError::RuntimeError {
                            message: format!("列出目录失败: {}", e),
                        }),
                    }
                }
                return Err(TenthError::RuntimeError {
                    message: "list_dir(路径) 期望一个字符串路径".into(),
                });
            }
            "file_size" => {
                if let Some(Value::String(path)) = args.first() {
                    // H-2: 沙箱校验
                    let resolved = if let Some(ref sb) = self.fs_sandbox {
                        match sb.check_read(path) {
                            Ok(p) => p,
                            Err(e) => return Err(TenthError::RuntimeError { message: e }),
                        }
                    } else {
                        std::path::PathBuf::from(path)
                    };
                    match std::fs::metadata(&resolved) {
                        Ok(meta) => return Ok(Some(Value::Int(meta.len() as i64))),
                        Err(e) => return Err(TenthError::RuntimeError {
                            message: format!("获取文件大小失败: {}", e),
                        }),
                    }
                }
                return Err(TenthError::RuntimeError {
                    message: "file_size(路径) 期望一个字符串路径".into(),
                });
            }
            "remove_file" => {
                if let Some(Value::String(path)) = args.first() {
                    // H-2: 沙箱校验（写操作）
                    let resolved = if let Some(ref sb) = self.fs_sandbox {
                        match sb.check_write(path) {
                            Ok(p) => p,
                            Err(e) => return Err(TenthError::RuntimeError { message: e }),
                        }
                    } else {
                        std::path::PathBuf::from(path)
                    };
                    match std::fs::remove_file(&resolved) {
                        Ok(()) => return Ok(Some(Value::Unit)),
                        Err(e) => return Err(TenthError::RuntimeError {
                            message: format!("删除文件失败: {}", e),
                        }),
                    }
                }
                return Err(TenthError::RuntimeError {
                    message: "remove_file(路径) 期望一个字符串路径".into(),
                });
            }
            "copy_file" => {
                if args.len() >= 2 {
                    if let (Value::String(src), Value::String(dst)) = (&args[0], &args[1]) {
                        // H-2: 沙箱校验（读源 + 写目标）
                        let src_resolved = if let Some(ref sb) = self.fs_sandbox {
                            match sb.check_read(src) {
                                Ok(p) => p,
                                Err(e) => return Err(TenthError::RuntimeError { message: e }),
                            }
                        } else {
                            std::path::PathBuf::from(src)
                        };
                        let dst_resolved = if let Some(ref sb) = self.fs_sandbox {
                            match sb.check_write(dst) {
                                Ok(p) => p,
                                Err(e) => return Err(TenthError::RuntimeError { message: e }),
                            }
                        } else {
                            std::path::PathBuf::from(dst)
                        };
                        match std::fs::copy(&src_resolved, &dst_resolved) {
                            Ok(_) => return Ok(Some(Value::Unit)),
                            Err(e) => return Err(TenthError::RuntimeError {
                                message: format!("复制文件失败: {}", e),
                            }),
                        }
                    }
                }
                return Err(TenthError::RuntimeError {
                    message: "copy_file(源路径, 目标路径) 期望两个字符串参数".into(),
                });
            }
            "rename_file" => {
                if args.len() >= 2 {
                    if let (Value::String(src), Value::String(dst)) = (&args[0], &args[1]) {
                        // H-2: 沙箱校验（读源 + 写目标）
                        let src_resolved = if let Some(ref sb) = self.fs_sandbox {
                            match sb.check_read(src) {
                                Ok(p) => p,
                                Err(e) => return Err(TenthError::RuntimeError { message: e }),
                            }
                        } else {
                            std::path::PathBuf::from(src)
                        };
                        let dst_resolved = if let Some(ref sb) = self.fs_sandbox {
                            match sb.check_write(dst) {
                                Ok(p) => p,
                                Err(e) => return Err(TenthError::RuntimeError { message: e }),
                            }
                        } else {
                            std::path::PathBuf::from(dst)
                        };
                        match std::fs::rename(&src_resolved, &dst_resolved) {
                            Ok(()) => return Ok(Some(Value::Unit)),
                            Err(e) => return Err(TenthError::RuntimeError {
                                message: format!("重命名文件失败: {}", e),
                            }),
                        }
                    }
                }
                return Err(TenthError::RuntimeError {
                    message: "rename_file(源路径, 目标路径) 期望两个字符串参数".into(),
                });
            }
            "Vec::new" => return Ok(Some(Value::Vec(Rc::new(RefCell::new(Vec::new()))))),
            "HashMap::new" => return Ok(Some(Value::Map(Rc::new(RefCell::new(HashMap::new()))))),
            "compile_host" => {
                if args.len() >= 2 {
                    if let (Value::String(src), Value::String(out)) = (&args[0], &args[1]) {
                        // H-2/L-7: 沙箱校验写路径
                        let out_resolved = if let Some(ref sb) = self.fs_sandbox {
                            match sb.check_write(out) {
                                Ok(p) => p,
                                Err(e) => return Err(TenthError::RuntimeError { message: e }),
                            }
                        } else {
                            std::path::PathBuf::from(out)
                        };
                        match crate::lexer::lexer::Lexer::new(src).tokenize()
                            .and_then(|tokens| crate::parser::parser::Parser::new(tokens).parse_program())
                            .and_then(|prog| crate::hir::lower::Lowerer::new().lower_program(&prog))
                            .and_then(|hir| crate::compile::compile_to_wasm(&hir))
                        {
                            Ok(wasm_bytes) => {
                                let _ = std::fs::write(&out_resolved, &wasm_bytes);
                                return Ok(Some(Value::Int(0)));
                            }
                            Err(_) => return Ok(Some(Value::Int(1))),
                        }
                    }
                }
                return Err(TenthError::RuntimeError {
                    message: "compile_host(源码, 输出路径) 期望两个字符串参数".into(),
                });
            }
            "compile_program" => {
                // Takes (program: Program, out_path: str) -> i64
                // Program is the struct produced by the self-hosting parser.
                if args.len() >= 2 {
                    if let Value::String(out) = &args[1] {
                        // H-2/L-7: 沙箱校验写路径
                        let out_resolved = if let Some(ref sb) = self.fs_sandbox {
                            match sb.check_write(out) {
                                Ok(p) => p,
                                Err(e) => return Err(TenthError::RuntimeError { message: e }),
                            }
                        } else {
                            std::path::PathBuf::from(out)
                        };
                        match crate::compile::compile_program_to_wasm(&args[0]) {
                            Ok(wasm_bytes) => {
                                let _ = std::fs::write(&out_resolved, &wasm_bytes);
                                return Ok(Some(Value::Int(0)));
                            }
                            Err(e) => {
                                eprintln!("[compile_program] error: {}", e);
                                return Ok(Some(Value::Int(1)));
                            }
                        }
                    }
                }
                return Err(TenthError::RuntimeError {
                    message: "compile_program(程序, 输出路径) 期望 Program 结构体和字符串路径".into(),
                });
            }
            "format" => {
                if args.is_empty() {
                    return Err(TenthError::RuntimeError {
                        message: "format() 至少需要一个模板字符串".into(),
                    });
                }
                if let Value::String(template) = &args[0] {
                    let mut result = String::new();
                    let mut arg_idx = 1;
                    let mut chars = template.chars().peekable();
                    while let Some(c) = chars.next() {
                        if c == '{' {
                            if chars.peek() == Some(&'{') {
                                chars.next();
                                result.push('{');
                            } else {
                                // Find closing }
                                let mut placeholder = String::new();
                                while let Some(pc) = chars.next() {
                                    if pc == '}' {
                                        break;
                                    }
                                    placeholder.push(pc);
                                }
                                if arg_idx < args.len() {
                                    result.push_str(&format!("{}", args[arg_idx]));
                                    arg_idx += 1;
                                } else {
                                    result.push('{');
                                    result.push_str(&placeholder);
                                    result.push('}');
                                }
                            }
                        } else if c == '}' {
                            if chars.peek() == Some(&'}') {
                                chars.next();
                                result.push('}');
                            } else {
                                result.push('}');
                            }
                        } else {
                            result.push(c);
                        }
                    }
                    return Ok(Some(Value::String(result)));
                }
                return Err(TenthError::RuntimeError {
                    message: "format() 第一个参数必须是字符串模板".into(),
                })
            }
            "parse_int" => {
                if let Some(arg) = args.first() {
                    if let Value::String(s) = arg {
                        return Ok(Some(Value::Int(s.trim().parse::<i64>().unwrap_or(0))));
                    }
                }
                return Err(TenthError::RuntimeError {
                    message: "parse_int() 期望一个字符串参数".into(),
                })
            }
            "parse_float" => {
                if let Some(arg) = args.first() {
                    if let Value::String(s) = arg {
                        return Ok(Some(Value::Float(s.trim().parse::<f64>().unwrap_or(0.0))));
                    }
                }
                return Err(TenthError::RuntimeError {
                    message: "parse_float() 期望一个字符串参数".into(),
                })
            }
            // Time functions
            "time_now" => {
                let now = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs_f64();
                return Ok(Some(Value::Float(now)));
            }
            "time_now_ms" => {
                let now = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_millis() as f64;
                return Ok(Some(Value::Float(now)));
            }
            "time_date" => {
                let secs = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs();
                let days_since_epoch = secs / 86400;
                let (year, month, day) = days_to_date(days_since_epoch);
                return Ok(Some(Value::String(format!("{}-{:02}-{:02}", year, month, day))));
            }
            "time_time" => {
                let secs = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs() % 86400;
                let h = secs / 3600;
                let m = (secs % 3600) / 60;
                let s = secs % 60;
                return Ok(Some(Value::String(format!("{}:{:02}:{:02}", h, m, s))));
            }
            "time_datetime" => {
                let secs = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs();
                let days_since_epoch = secs / 86400;
                let (year, month, day) = days_to_date(days_since_epoch);
                let day_secs = secs % 86400;
                let h = day_secs / 3600;
                let m = (day_secs % 3600) / 60;
                let s = day_secs % 60;
                return Ok(Some(Value::String(format!("{}-{:02}-{:02} {}:{:02}:{:02}", year, month, day, h, m, s))));
            }
            "time_sleep_ms" => {
                if let Some(Value::Int(ms)) = args.first() {
                    std::thread::sleep(std::time::Duration::from_millis(*ms as u64));
                    return Ok(Some(Value::Unit));
                }
                return Err(TenthError::RuntimeError {
                    message: "time_sleep_ms(ms) 期望一个整数".into(),
                });
            }
            // Random functions
            "random_int" => {
                if let (Some(Value::Int(lo)), Some(Value::Int(hi))) = (args.first(), args.get(1)) {
                    use std::collections::hash_map::DefaultHasher;
                    use std::hash::{Hash, Hasher};
                    let now = std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap_or_default()
                        .as_nanos();
                    let mut hasher = DefaultHasher::new();
                    now.hash(&mut hasher);
                    let rand_val = hasher.finish();
                    let range = (*hi - *lo + 1).max(1);
                    let result = *lo + ((rand_val % (range as u64)) as i64);
                    return Ok(Some(Value::Int(result)));
                }
                return Ok(Some(Value::Int(0)));
            }
            "random_float" => {
                use std::collections::hash_map::DefaultHasher;
                use std::hash::{Hash, Hasher};
                let now = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_nanos();
                let mut hasher = DefaultHasher::new();
                now.hash(&mut hasher);
                let rand_val = hasher.finish();
                let result = (rand_val as f64) / (u64::MAX as f64);
                return Ok(Some(Value::Float(result)));
            }
            // Math functions
            "math_tan" => {
                if let Some(Value::Float(x)) = args.first() {
                    return Ok(Some(Value::Float(x.tan())));
                }
                return Ok(Some(Value::Float(0.0)));
            }
            "math_asin" => {
                if let Some(Value::Float(x)) = args.first() {
                    return Ok(Some(Value::Float(x.asin())));
                }
                return Ok(Some(Value::Float(0.0)));
            }
            "math_acos" => {
                if let Some(Value::Float(x)) = args.first() {
                    return Ok(Some(Value::Float(x.acos())));
                }
                return Ok(Some(Value::Float(0.0)));
            }
            "math_atan" => {
                if let Some(Value::Float(x)) = args.first() {
                    return Ok(Some(Value::Float(x.atan())));
                }
                return Ok(Some(Value::Float(0.0)));
            }
            "math_atan2" => {
                if let (Some(Value::Float(y)), Some(Value::Float(x))) = (args.first(), args.get(1)) {
                    return Ok(Some(Value::Float(y.atan2(*x))));
                }
                return Ok(Some(Value::Float(0.0)));
            }
            "math_sinh" => {
                if let Some(Value::Float(x)) = args.first() {
                    return Ok(Some(Value::Float(x.sinh())));
                }
                return Ok(Some(Value::Float(0.0)));
            }
            "math_cosh" => {
                if let Some(Value::Float(x)) = args.first() {
                    return Ok(Some(Value::Float(x.cosh())));
                }
                return Ok(Some(Value::Float(0.0)));
            }
            "math_tanh" => {
                if let Some(Value::Float(x)) = args.first() {
                    return Ok(Some(Value::Float(x.tanh())));
                }
                return Ok(Some(Value::Float(0.0)));
            }
            "math_log10" => {
                if let Some(Value::Float(x)) = args.first() {
                    return Ok(Some(Value::Float(x.log10())));
                }
                return Ok(Some(Value::Float(0.0)));
            }
            "math_log2" => {
                if let Some(Value::Float(x)) = args.first() {
                    return Ok(Some(Value::Float(x.log2())));
                }
                return Ok(Some(Value::Float(0.0)));
            }
            "math_exp" => {
                if let Some(Value::Float(x)) = args.first() {
                    return Ok(Some(Value::Float(x.exp())));
                }
                return Ok(Some(Value::Float(0.0)));
            }
            "math_pow" => {
                if let (Some(Value::Float(base)), Some(Value::Float(exp))) = (args.first(), args.get(1)) {
                    return Ok(Some(Value::Float(base.powf(*exp))));
                }
                return Ok(Some(Value::Float(0.0)));
            }
            "math_floor" => {
                if let Some(Value::Float(x)) = args.first() {
                    return Ok(Some(Value::Float(x.floor())));
                }
                return Ok(Some(Value::Float(0.0)));
            }
            "math_ceil" => {
                if let Some(Value::Float(x)) = args.first() {
                    return Ok(Some(Value::Float(x.ceil())));
                }
                return Ok(Some(Value::Float(0.0)));
            }
            "math_round" => {
                if let Some(Value::Float(x)) = args.first() {
                    return Ok(Some(Value::Float(x.round())));
                }
                return Ok(Some(Value::Float(0.0)));
            }
            // CLI functions
            "cli_args_count" => {
                return Ok(Some(Value::Int(1))); // Default: just program name
            }
            "cli_arg" => {
                if let Some(Value::Int(_idx)) = args.first() {
                    return Ok(Some(Value::String(String::new())));
                }
                return Ok(Some(Value::String(String::new())));
            }
            // JSON functions
            "json_encode" => {
                if let Some(val) = args.first() {
                    return Ok(Some(Value::String(json_encode_value(val))));
                }
                return Ok(Some(Value::String("null".into())));
            }
            "json_encode_pretty" => {
                if let Some(val) = args.first() {
                    return Ok(Some(Value::String(json_encode_value_pretty(val, 0))));
                }
                return Ok(Some(Value::String("null".into())));
            }
            "json_decode" => {
                if let Some(Value::String(s)) = args.first() {
                    return Ok(Some(json_decode_string(s)));
                }
                return Ok(Some(Value::Unit));
            }
            _ => {}
        }

        if name.contains("::") {
            let parts: Vec<&str> = name.splitn(2, "::").collect();
            if parts.len() == 2 {
                let mod_name = parts[0];
                let fn_name = parts[1];
                if let Some(module) = self.modules.get(mod_name) {
                    if let Some(fn_def) = module.functions.iter().find(|f| f.name == fn_name) {
                        let fn_def = fn_def.clone();
                        self.scopes.push(HashMap::new());

                        for ((pname, _), arg) in fn_def.params.iter().zip(args.iter()) {
                            self.current_scope().insert(pname.clone(), arg.clone());
                        }

                        let result = self.eval_expr(&fn_def.body);

                        self.scopes.pop();

                        return Self::unwrap_return(result);
                    }
                }
                return Err(TenthError::RuntimeError {
                    message: format!("未定义函数 '{}'", name),
                });
            }
        }

        let func_def = self.functions.iter().find(|f| f.name == name).cloned();
        if let Some(fd) = func_def {
            // Push a new scope for function-local variables.
            // Parameters and locals are isolated; globals remain visible via scope chain.
            self.scopes.push(HashMap::new());

            for ((pname, _), arg) in fd.params.iter().zip(args.iter()) {
                self.current_scope().insert(pname.clone(), arg.clone());
            }

            let result = self.eval_expr(&fd.body);

            self.scopes.pop();

            return Self::unwrap_return(result);
        }

        Err(TenthError::RuntimeError {
            message: format!("undefined function '{}'", name),
        })
    }
}
