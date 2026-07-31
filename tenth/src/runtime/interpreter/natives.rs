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
use crate::hir::types::BaseType;
use std::rc::Rc;
use std::cell::RefCell;
use crate::error::{TenthError, TenthResult};
use crate::runtime::value::Value;
use crate::runtime::tensor::{Tensor, TensorData};
use crate::runtime::autodiff::Tape;
use super::json::{json_encode_value, json_encode_value_pretty, json_decode_string};
use super::datetime::days_to_date;
use super::datetime::{date_to_days, days_to_date_i64};

// B批：编码工具
use unicode_normalization::UnicodeNormalization;
use base64::Engine as _;

/// 构造 Result::Ok(value)
fn ok_result(value: Value) -> Value {
    Value::Enum {
        enum_name: "Result".to_string(),
        variant: "Ok".to_string(),
        fields: Rc::new(RefCell::new(vec![("_0".to_string(), value)])),
    }
}

/// 自动解引用 Shared/Ref/MutRef 包裹的值，返回一个 clone 后的内部 Value。
/// 用于原生数值函数（to_f64/to_float/to_f32 等）接收 Vec.get() 返回值时
/// 自动剥壳——解释器路径下 Vec.push 会用 Shared 包裹元素（便于索引赋值变更），
/// 而 to_f64 等仅识别 Int/Float/Float32/Tensor，需要此处解壳以对齐 VM 行为。
/// 非包裹类型原样 clone 返回。
fn deref_wrapped(v: &Value) -> Value {
    match v {
        Value::Shared(rc) => {
            let inner = rc.borrow();
            let inner_val = inner.clone();
            // 递归一层，防止 Shared<Shared<T>> 之类的双重包裹
            match &inner_val {
                Value::Shared(_) | Value::Ref(_) | Value::MutRef(_) => deref_wrapped(&inner_val),
                _ => inner_val,
            }
        }
        Value::Ref(rc) => {
            let inner = rc.borrow();
            let inner_val = inner.clone();
            match &inner_val {
                Value::Shared(_) | Value::Ref(_) | Value::MutRef(_) => deref_wrapped(&inner_val),
                _ => inner_val,
            }
        }
        Value::MutRef(weak) => {
            if let Some(rc) = weak.upgrade() {
                let inner = rc.borrow();
                inner.clone()
            } else {
                Value::Unit
            }
        }
        other => other.clone(),
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

/// 将 Tenth 格式说明符应用到值上，返回格式化后的字符串。
/// 支持：`>5`（右对齐）、`<5`（左对齐）、`^5`（居中）、`.2f`（小数点精度）
fn apply_format_spec(value: &str, spec: &str) -> String {
    let spec = spec.trim();
    if spec.is_empty() {
        return value.to_string();
    }
    if let Some(width_str) = spec.strip_prefix('>') {
        let width = width_str.parse::<usize>().unwrap_or(0);
        format!("{:>width$}", value, width = width)
    } else if let Some(width_str) = spec.strip_prefix('<') {
        let width = width_str.parse::<usize>().unwrap_or(0);
        format!("{:<width$}", value, width = width)
    } else if let Some(width_str) = spec.strip_prefix('^') {
        let width = width_str.parse::<usize>().unwrap_or(0);
        format!("{:^width$}", value, width = width)
    } else if spec.starts_with('.') {
        let trimmed = spec.trim_end_matches('f');
        let decimals = trimmed[1..].parse::<usize>().unwrap_or(2);
        let val: f64 = value.parse().unwrap_or(0.0);
        format!("{:.decimals$}", val, decimals = decimals)
    } else {
        // Unknown specifier - pass through as-is
        value.to_string()
    }
}

/// 扫描模板字符串，统计位置占位符数量并判断是否包含命名占位符。
fn count_placeholders(template: &str) -> (usize, bool) {
    let mut pos_count = 0;
    let mut has_named = false;
    let mut chars = template.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '{' {
            if chars.peek() == Some(&'{') {
                chars.next();
                continue;
            }
            let mut content = String::new();
            let mut found_colon = false;
            while let Some(pc) = chars.next() {
                if pc == '}' { break; }
                if pc == ':' { found_colon = true; }
                else if !found_colon { content.push(pc); }
            }
            if content.is_empty() || content.chars().all(|c| c.is_ascii_digit()) {
                pos_count += 1;
            } else {
                has_named = true;
            }
        }
    }
    (pos_count, has_named)
}

/// 格式化单个占位符：解析位置/命名参数并应用格式说明符。
fn format_placeholder(
    placeholder: &str,
    fmt_spec: &str,
    args: &[Value],
    named_args: &HashMap<String, Value>,
    arg_idx: usize,
    num_positional: usize,
    _has_named_args: bool,
) -> TenthResult<String> {
    let arg_val = if placeholder.is_empty() {
        // Positional placeholder: take next positional arg (arg_idx)
        let pos = 1 + arg_idx; // +1 for template at args[0]
        if pos < args.len() && pos < 1 + num_positional {
            let val = &args[pos];
            format!("{}", val)
        } else {
            return Err(TenthError::RuntimeError { line: None, col: None,
                message: format!("format() 位置参数 #{} 越界（共 {} 个位置参数）",
                    arg_idx, num_positional),
            });
        }
    } else if placeholder.chars().all(|c| c.is_ascii_digit()) {
        // Explicit numeric index: {0}, {1}
        let idx: usize = placeholder.parse::<usize>().unwrap_or(0);
        if idx < num_positional {
            let val = &args[1 + idx];
            format!("{}", val)
        } else {
            return Err(TenthError::RuntimeError { line: None, col: None,
                message: format!("format() 索引 #{} 越界（共 {} 个位置参数）", idx, num_positional),
            });
        }
    } else {
        // Named parameter: {name}
        match named_args.get(placeholder) {
            Some(val) => format!("{}", val),
            None => {
                return Err(TenthError::RuntimeError { line: None, col: None,
                    message: format!("format() 未找到命名参数 '{}'", placeholder),
                });
            }
        }
    };

    // Apply format specifier if present
    if fmt_spec.is_empty() {
        Ok(arg_val)
    } else {
        // Parse format specifier and apply with Rust's format!
        let formatted = apply_format_spec(&arg_val, fmt_spec);
        Ok(formatted)
    }
}

// HTTP 客户端辅助函数（parse_http_url / decode_chunked / http_read_response /
// http_get_impl / http_post_impl）已迁移到 `crate::http`（T1.2）。
// 下方 call_named_fn 中的 http_get/http_post 分支通过 `crate::http::*` 调用。

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
                let code = if let Some(Value::Int(c, _)) = args.first() { *c } else { 0 };
                std::process::exit(code as i32);
            }
            // —— 阶段1-静默失败：Result/Option 显式解包原语 ——
            // or_die(x, msg)：x 是 Result/Option。Ok/Some → 取出内部值；Err/None → panic（RuntimeError，消息含 msg）。
            // 与 VM 侧 `register_all_natives` 的 or_die native 语义保持一致（双重注册硬要求）。
            "or_die" => {
                if args.is_empty() {
                    return Err(TenthError::RuntimeError { line: None, col: None,
                        message: "or_die 需要至少 1 个参数（值, 可选消息）".into(),
                    });
                }
                match &args[0] {
                    Value::Enum { enum_name, variant, fields } => {
                        let is_ok = (enum_name == "Result" && variant == "Ok")
                            || (enum_name == "Option" && variant == "Some");
                        if is_ok {
                            // Ok/Some：取出内部值（_0 字段）
                            let borrowed = fields.borrow();
                            let inner = borrowed.first().map(|(_, v)| v.clone()).unwrap_or(Value::Unit);
                            return Ok(Some(inner));
                        } else {
                            // Err/None：panic（Tenth 运行时错误机制）
                            let msg = if args.len() >= 2 {
                                match &args[1] {
                                    Value::String(s) => s.clone(),
                                    v => format!("{}", v),
                                }
                            } else {
                                "值失败".to_string()
                            };
                            return Err(TenthError::RuntimeError { line: None, col: None,
                                message: format!("or_die: {}", msg),
                            });
                        }
                    }
                    // 非 Result/Option：原样透传（不 panic，保持宽容）
                    v => return Ok(Some(v.clone())),
                }
            }
            // assume_ok(x)：不做检查直接取内部值（声明"我保证成功"，用户负责）。
            // 对 Ok/Some 取 _0；对 Err/None 也取 _0（Err 的 _0 是错误消息）或 Unit。
            "assume_ok" => {
                match args.first() {
                    Some(Value::Enum { fields, .. }) => {
                        let borrowed = fields.borrow();
                        let inner = borrowed.first().map(|(_, v)| v.clone()).unwrap_or(Value::Unit);
                        return Ok(Some(inner));
                    }
                    // 非 Result/Option：原样透传
                    Some(v) => return Ok(Some(v.clone())),
                    None => return Ok(Some(Value::Unit)),
                }
            }
            // —— TCP 网络原语（句柄表方案，handle 1-based，0 表示无效）——
            "tcp_connect" => {
                if args.len() < 2 {
                    return Ok(Some(err_result("tcp_connect 需要 (String, i64) 参数")));
                }
                if let (Value::String(host), Value::Int(port, _)) = (&args[0], &args[1]) {
                    let addr = format!("{}:{}", host, port);
                    match std::net::TcpStream::connect(&addr) {
                        Ok(stream) => {
                            self.tcp_streams.push(Some(stream));
                            let handle = self.tcp_streams.len() as i64; // 1-based
                            return Ok(Some(ok_result(Value::Int(handle, BaseType::I32))));
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
                if let (Value::Int(handle, _), Value::Int(n, _)) = (&args[0], &args[1]) {
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
                                    .map(|b| Value::Int(*b as i64, BaseType::I32))
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
                if let (Value::Int(handle, _), Value::Vec(data)) = (&args[0], &args[1]) {
                    let idx = *handle as usize;
                    if idx == 0 || idx > self.tcp_streams.len() {
                        return Ok(Some(err_result("无效的句柄")));
                    }
                    let bytes: Vec<u8> = data
                        .borrow()
                        .iter()
                        .map(|x| match x {
                            Value::Int(b, _) => *b as u8,
                            _ => 0,
                        })
                        .collect();
                    if let Some(ref mut stream) = self.tcp_streams[idx - 1] {
                        use std::io::Write;
                        match stream.write_all(&bytes) {
                            Ok(_) => return Ok(Some(ok_result(Value::Int(bytes.len() as i64, BaseType::I32)))),
                            Err(e) => return Ok(Some(err_result(format!("写入失败: {e}")))),
                        }
                    } else {
                        return Ok(Some(err_result("连接已关闭")));
                    }
                }
                return Ok(Some(err_result("tcp_write 需要 (i64, Vec<i64>) 参数")));
            }
            "tcp_close" => {
                if let Some(Value::Int(handle, _)) = args.first() {
                    let idx = *handle as usize;
                    if idx > 0 && idx <= self.tcp_streams.len() {
                        self.tcp_streams[idx - 1] = None; // drop 自动关闭
                    }
                }
                return Ok(Some(Value::Unit));
            }
            "tcp_set_timeout" => {
                if args.len() >= 2 {
                    if let (Value::Int(handle, _), Value::Int(ms, _)) = (&args[0], &args[1]) {
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
            // —— TCP 服务端原语（句柄表方案，handle 1-based，0 表示无效）——
            // 与 std/net.th 的 listen/accept/listener_close wrapper 对齐。
            "tcp_listen" => {
                if let Some(Value::String(addr)) = args.first() {
                    match std::net::TcpListener::bind(addr) {
                        Ok(listener) => {
                            self.tcp_listeners.push(Some(listener));
                            let handle = self.tcp_listeners.len() as i64; // 1-based
                            return Ok(Some(ok_result(Value::Int(handle, BaseType::I32))));
                        }
                        Err(e) => return Ok(Some(err_result(format!("监听失败: {e}")))),
                    }
                }
                return Ok(Some(err_result("tcp_listen 需要 1 个 String 参数")));
            }
            "tcp_accept" => {
                if let Some(Value::Int(handle, _)) = args.first() {
                    let idx = *handle as usize;
                    if idx == 0 || idx > self.tcp_listeners.len() {
                        return Ok(Some(err_result("无效的监听器句柄")));
                    }
                    if let Some(ref listener) = self.tcp_listeners[idx - 1] {
                        match listener.accept() {
                            Ok((stream, _)) => {
                                self.tcp_streams.push(Some(stream));
                                let stream_handle = self.tcp_streams.len() as i64; // 1-based
                                return Ok(Some(ok_result(Value::Int(stream_handle, BaseType::I32))));
                            }
                            Err(e) => return Ok(Some(err_result(format!("接受连接失败: {e}")))),
                        }
                    } else {
                        return Ok(Some(err_result("监听器已关闭")));
                    }
                }
                return Ok(Some(err_result("tcp_accept 需要 1 个 i64 参数")));
            }
            "tcp_listener_close" => {
                if let Some(Value::Int(handle, _)) = args.first() {
                    let idx = *handle as usize;
                    if idx > 0 && idx <= self.tcp_listeners.len() {
                        self.tcp_listeners[idx - 1] = None; // drop 自动关闭
                    }
                }
                return Ok(Some(Value::Unit));
            }
            // —— UDP 网络原语（基本功核查第 69 项；句柄表方案，handle 1-based，0 表示无效）——
            // 与 std/net.th 的 udp_bind/udp_recv_from/udp_send_to/udp_close/udp_set_timeout wrapper 对齐。
            // 与 runtime::natives 中的实现语义对齐（双侧注册）。
            // UDP 无连接：bind 后用 send_to/recv_from 携带对端地址；handle 表与 TCP 独立避免类型混淆。
            "udp_bind" => {
                if let Some(Value::String(addr)) = args.first() {
                    match std::net::UdpSocket::bind(addr) {
                        Ok(sock) => {
                            self.udp_sockets.push(Some(sock));
                            let handle = self.udp_sockets.len() as i64; // 1-based
                            return Ok(Some(ok_result(Value::Int(handle, BaseType::I32))));
                        }
                        Err(e) => return Ok(Some(err_result(format!("绑定失败: {e}")))),
                    }
                }
                return Ok(Some(err_result("udp_bind 需要 1 个 String 参数")));
            }
            "udp_recv_from" => {
                if args.len() < 2 {
                    return Ok(Some(err_result("udp_recv_from 需要 (i64, i64) 参数")));
                }
                if let (Value::Int(handle, _), Value::Int(n, _)) = (&args[0], &args[1]) {
                    let idx = *handle as usize;
                    if idx == 0 || idx > self.udp_sockets.len() {
                        return Ok(Some(err_result("无效的句柄")));
                    }
                    let max = (*n).max(0).min(65536) as usize;
                    if let Some(ref mut sock) = self.udp_sockets[idx - 1] {
                        let mut buf = vec![0u8; max];
                        match sock.recv_from(&mut buf) {
                            Ok((read_n, peer)) => {
                                let bytes: Vec<Value> = buf[..read_n]
                                    .iter()
                                    .map(|b| Value::Int(*b as i64, BaseType::I32))
                                    .collect();
                                let peer_str = peer.to_string();
                                // 返回 Tuple(Vec<i64>, String)：字节数组 + 来源地址 "ip:port"
                                return Ok(Some(ok_result(Value::Tuple(vec![
                                    Value::Vec(Rc::new(RefCell::new(bytes))),
                                    Value::String(peer_str),
                                ]))));
                            }
                            Err(e) => return Ok(Some(err_result(format!("接收失败: {e}")))),
                        }
                    } else {
                        return Ok(Some(err_result("socket 已关闭")));
                    }
                }
                return Ok(Some(err_result("udp_recv_from 需要 (i64, i64) 参数")));
            }
            "udp_send_to" => {
                if args.len() < 3 {
                    return Ok(Some(err_result("udp_send_to 需要 (i64, Vec<i64>, String) 参数")));
                }
                if let (Value::Int(handle, _), Value::Vec(data), Value::String(addr)) =
                    (&args[0], &args[1], &args[2])
                {
                    let idx = *handle as usize;
                    if idx == 0 || idx > self.udp_sockets.len() {
                        return Ok(Some(err_result("无效的句柄")));
                    }
                    let bytes: Vec<u8> = data
                        .borrow()
                        .iter()
                        .map(|x| match x {
                            Value::Int(b, _) => *b as u8,
                            _ => 0,
                        })
                        .collect();
                    if let Some(ref mut sock) = self.udp_sockets[idx - 1] {
                        match sock.send_to(&bytes, addr) {
                            Ok(n) => return Ok(Some(ok_result(Value::Int(n as i64, BaseType::I32)))),
                            Err(e) => return Ok(Some(err_result(format!("发送失败: {e}")))),
                        }
                    } else {
                        return Ok(Some(err_result("socket 已关闭")));
                    }
                }
                return Ok(Some(err_result("udp_send_to 需要 (i64, Vec<i64>, String) 参数")));
            }
            "udp_close" => {
                if let Some(Value::Int(handle, _)) = args.first() {
                    let idx = *handle as usize;
                    if idx > 0 && idx <= self.udp_sockets.len() {
                        self.udp_sockets[idx - 1] = None; // drop 自动关闭
                    }
                }
                return Ok(Some(Value::Unit));
            }
            "udp_set_timeout" => {
                if args.len() >= 2 {
                    if let (Value::Int(handle, _), Value::Int(ms, _)) = (&args[0], &args[1]) {
                        let idx = *handle as usize;
                        if idx > 0 && idx <= self.udp_sockets.len() {
                            if let Some(ref mut sock) = self.udp_sockets[idx - 1] {
                                let dur = std::time::Duration::from_millis(*ms as u64);
                                sock.set_read_timeout(Some(dur)).ok();
                                sock.set_write_timeout(Some(dur)).ok();
                            }
                        }
                    }
                }
                return Ok(Some(Value::Unit));
            }
            // —— 子进程原语（句柄表方案，handle 1-based，0 表示无效）——
            // 与 std/process.th 的 new/arg/run/output wrapper 对齐。
            // command_output 消费 Command（mem::take 取出所有权），再次调用返回 Err。
            "command_new" => {
                if let Some(Value::String(program)) = args.first() {
                    let cmd = std::process::Command::new(program);
                    self.commands.push(Some(cmd));
                    let handle = self.commands.len() as i64; // 1-based
                    return Ok(Some(ok_result(Value::Int(handle, BaseType::I32))));
                }
                return Ok(Some(err_result("command_new 需要 1 个 String 参数")));
            }
            "command_arg" => {
                if args.len() >= 2 {
                    if let (Value::Int(handle, _), Value::String(arg)) = (&args[0], &args[1]) {
                        let idx = *handle as usize;
                        if idx > 0 && idx <= self.commands.len() {
                            if let Some(ref mut cmd) = self.commands[idx - 1] {
                                cmd.arg(arg);
                            }
                        }
                    }
                }
                return Ok(Some(Value::Unit));
            }
            "command_run" => {
                if let Some(Value::Int(handle, _)) = args.first() {
                    let idx = *handle as usize;
                    if idx == 0 || idx > self.commands.len() {
                        return Ok(Some(err_result("无效的命令句柄")));
                    }
                    if let Some(ref mut cmd) = self.commands[idx - 1] {
                        match cmd.status() {
                            Ok(status) => {
                                let code = status.code().unwrap_or(-1) as i64;
                                return Ok(Some(ok_result(Value::Int(code, BaseType::I32))));
                            }
                            Err(e) => return Ok(Some(err_result(format!("执行失败: {e}")))),
                        }
                    } else {
                        return Ok(Some(err_result("命令已释放")));
                    }
                }
                return Ok(Some(err_result("command_run 需要 1 个 i64 参数")));
            }
            "command_output" => {
                if let Some(Value::Int(handle, _)) = args.first() {
                    let idx = *handle as usize;
                    if idx == 0 || idx > self.commands.len() {
                        return Ok(Some(err_result("无效的命令句柄")));
                    }
                    // output() 消费 Command 语义：用 mem::take 取出所有权，槽位变 None
                    let cmd_opt = std::mem::take(&mut self.commands[idx - 1]);
                    if let Some(mut cmd) = cmd_opt {
                        match cmd.output() {
                            Ok(output) => {
                                let stdout = String::from_utf8_lossy(&output.stdout).to_string();
                                return Ok(Some(ok_result(Value::String(stdout))));
                            }
                            Err(e) => return Ok(Some(err_result(format!("执行失败: {e}")))),
                        }
                    } else {
                        return Ok(Some(err_result("命令已释放")));
                    }
                }
                return Ok(Some(err_result("command_output 需要 1 个 i64 参数")));
            }
            // —— 正则表达式原语（句柄表方案，handle 1-based，0 表示无效）——
            // 与 std/regex.th 对齐：Tenth 层不暴露 Regex 类型，仅用 i64 handle。
            "regex_compile" => {
                if let Some(Value::String(pattern)) = args.first() {
                    match regex::Regex::new(pattern) {
                        Ok(re) => {
                            self.regexes.push(Some(re));
                            let handle = self.regexes.len() as i64; // 1-based
                            return Ok(Some(ok_result(Value::Int(handle, BaseType::I32))));
                        }
                        Err(e) => return Ok(Some(err_result(format!("正则编译失败: {e}")))),
                    }
                }
                return Ok(Some(err_result("regex_compile 需要 1 个 String 参数")));
            }
            "regex_match" => {
                if args.len() < 2 {
                    return Ok(Some(Value::Bool(false)));
                }
                if let (Value::Int(handle, _), Value::String(input)) = (&args[0], &args[1]) {
                    let idx = *handle as usize;
                    if idx == 0 || idx > self.regexes.len() {
                        return Ok(Some(Value::Bool(false)));
                    }
                    if let Some(ref re) = self.regexes[idx - 1] {
                        return Ok(Some(Value::Bool(re.is_match(input))));
                    }
                    return Ok(Some(Value::Bool(false)));
                }
                return Ok(Some(Value::Bool(false)));
            }
            "regex_find" => {
                // 与 std/regex.th 契约对齐：返回 String，无匹配返回空字符串 ""
                if args.len() < 2 {
                    return Ok(Some(Value::String(String::new())));
                }
                if let (Value::Int(handle, _), Value::String(input)) = (&args[0], &args[1]) {
                    let idx = *handle as usize;
                    if idx == 0 || idx > self.regexes.len() {
                        return Ok(Some(Value::String(String::new())));
                    }
                    if let Some(ref re) = self.regexes[idx - 1] {
                        if let Some(m) = re.find(input) {
                            return Ok(Some(Value::String(m.as_str().to_string())));
                        }
                    }
                    return Ok(Some(Value::String(String::new())));
                }
                return Ok(Some(Value::String(String::new())));
            }
            "regex_find_all" => {
                // 与 std/regex.th 契约对齐：返回 Vec<String>
                if args.len() < 2 {
                    return Ok(Some(Value::Vec(Rc::new(RefCell::new(Vec::new())))));
                }
                if let (Value::Int(handle, _), Value::String(input)) = (&args[0], &args[1]) {
                    let idx = *handle as usize;
                    if idx == 0 || idx > self.regexes.len() {
                        return Ok(Some(Value::Vec(Rc::new(RefCell::new(Vec::new())))));
                    }
                    if let Some(ref re) = self.regexes[idx - 1] {
                        let collected: Vec<Value> = re
                            .find_iter(input)
                            .map(|m| Value::String(m.as_str().to_string()))
                            .collect();
                        return Ok(Some(Value::Vec(Rc::new(RefCell::new(collected)))));
                    }
                    return Ok(Some(Value::Vec(Rc::new(RefCell::new(Vec::new())))));
                }
                return Ok(Some(Value::Vec(Rc::new(RefCell::new(Vec::new())))));
            }
            "regex_replace" => {
                if args.len() < 3 {
                    return Ok(Some(Value::String(String::new())));
                }
                if let (Value::Int(handle, _), Value::String(input), Value::String(replacement)) =
                    (&args[0], &args[1], &args[2])
                {
                    let idx = *handle as usize;
                    if idx == 0 || idx > self.regexes.len() {
                        return Ok(Some(Value::String(input.clone())));
                    }
                    if let Some(ref re) = self.regexes[idx - 1] {
                        let result = re.replace_all(input, replacement.as_str()).into_owned();
                        return Ok(Some(Value::String(result)));
                    }
                    return Ok(Some(Value::String(input.clone())));
                }
                return Ok(Some(Value::String(String::new())));
            }
            "regex_split" => {
                // 与 std/regex.th 契约对齐：返回 Vec<String>
                if args.len() < 2 {
                    return Ok(Some(Value::Vec(Rc::new(RefCell::new(Vec::new())))));
                }
                if let (Value::Int(handle, _), Value::String(input)) = (&args[0], &args[1]) {
                    let idx = *handle as usize;
                    if idx == 0 || idx > self.regexes.len() {
                        return Ok(Some(Value::Vec(Rc::new(RefCell::new(Vec::new())))));
                    }
                    if let Some(ref re) = self.regexes[idx - 1] {
                        let collected: Vec<Value> = re
                            .split(input)
                            .map(|s| Value::String(s.to_string()))
                            .collect();
                        return Ok(Some(Value::Vec(Rc::new(RefCell::new(collected)))));
                    }
                    return Ok(Some(Value::Vec(Rc::new(RefCell::new(Vec::new())))));
                }
                return Ok(Some(Value::Vec(Rc::new(RefCell::new(Vec::new())))));
            }
            // —— HTTP 客户端原语（手写 HTTP/1.1，10 秒默认超时）——
            "http_get" => {
                if let Some(Value::String(url)) = args.first() {
                    match crate::http::http_get_impl(url) {
                        Ok(body) => return Ok(Some(ok_result(Value::String(body)))),
                        Err(e) => return Ok(Some(err_result(e))),
                    }
                }
                return Ok(Some(err_result("http_get 需要 1 个 String 参数")));
            }
            "http_post" => {
                if args.len() >= 2 {
                    if let (Value::String(url), Value::String(body)) = (&args[0], &args[1]) {
                        match crate::http::http_post_impl(url, body) {
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
                        Value::Int(_, _) => "int",
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
                    return Err(TenthError::RuntimeError { line: None, col: None,
                        message: "with_step_limit(limit, fn) 需要 2 个参数".into(),
                    });
                }
                let limit = args[0].as_int().ok_or_else(|| TenthError::RuntimeError { line: None, col: None,
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
                    return Err(TenthError::RuntimeError { line: None, col: None,
                        message: "with_timeout_ms(ms, fn) 需要 2 个参数".into(),
                    });
                }
                let ms = args[0].as_int().ok_or_else(|| TenthError::RuntimeError { line: None, col: None,
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
                    return Ok(Some(crate::runtime::value::array_to_tensor(arg)?));
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
                return Err(TenthError::RuntimeError { line: None, col: None,
                    message: "param() 期望一个张量参数".into(),
                });
            }
            "backward" => {
                if let Some(Value::Tensor(loss)) = args.first() {
                    let loss_id_opt = loss.borrow().tape_id;
                    // PROJ-006：先把 custom_ops Rc 副本交给 tape，使 Custom 节点的
                    // backward 能通过 registry 查到用户实现的 CustomBackward。
                    let custom_ops = self.custom_ops.clone();
                    if let (Some(tape), Some(loss_id)) = (&mut self.tape, loss_id_opt) {
                        tape.set_custom_ops(custom_ops);
                        // 护城河 F：包裹 backward 错误，附加 formal_explain 根因分析
                        // Phase 1：从 backward 抛出的 ShapeMismatch 错误中提取真实 v_err/expected/actual，
                        // 传给 formal_explain 提升根因分析精度（替代 Phase 0 的占位值 loss_id/&[]/&[]）。
                        match tape.backward(loss_id) {
                            Ok(()) => return Ok(Some(Value::Unit)),
                            Err(e) => {
                                // 从错误中提取结构化上下文（若错误是 ShapeMismatch）
                                let (v_err, expected, actual, error_msg) = match &e {
                                    TenthError::ShapeMismatch { context, message } => (
                                        context.tape_node_id,
                                        context.expected_shape.as_slice(),
                                        context.actual_shape.as_slice(),
                                        message.as_str(),
                                    ),
                                    _ => (loss_id, &[][..], &[][..], ""),
                                };
                                // 护城河 F Phase 2：error_msg 用于 5 类错误分类
                                let causes = tape.formal_explain(v_err, expected, actual, error_msg);
                                let explanations: Vec<String> = causes.iter().map(|c| c.explanation.clone()).collect();
                                self.last_explanation = explanations.clone();
                                // 若 backward 已返回 ShapeMismatch，复用其 context（保留真实 v_err/expected/actual）；
                                // 否则构造一个以 loss_id 为 v_err 的兜底 context。
                                let context = match &e {
                                    TenthError::ShapeMismatch { context, .. } => context.clone(),
                                    _ => crate::error::TapeErrorContext {
                                        tape_node_id: loss_id,
                                        op: "backward".to_string(),
                                        expected_shape: Vec::new(),
                                        actual_shape: Vec::new(),
                                    },
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
                return Err(TenthError::RuntimeError { line: None, col: None,
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
                    let arg = deref_wrapped(arg);
                    return Ok(Some(match &arg {
                        Value::Int(n, _) => Value::Int(n.abs(), BaseType::I32),
                        Value::Float(n) => Value::Float(n.abs()),
                        _ => return Err(TenthError::RuntimeError { line: None, col: None,
                            message: "abs() 期望一个数值参数".into(),
                        }),
                    }));
                }
                return Err(TenthError::RuntimeError { line: None, col: None,
                    message: "abs() 期望 1 个参数".into(),
                });
            }
            "sqrt" => {
                if let Some(arg) = args.first() {
                    let n = arg.as_float().ok_or_else(|| TenthError::RuntimeError { line: None, col: None,
                        message: "sqrt() 期望一个数值参数".into(),
                    })?;
                    return Ok(Some(Value::Float(n.sqrt())));
                }
                return Err(TenthError::RuntimeError { line: None, col: None,
                    message: "sqrt() 期望 1 个参数".into(),
                });
            }
            "to_float" => {
                if let Some(arg) = args.first() {
                    let arg = deref_wrapped(arg);
                    return Ok(Some(match &arg {
                        Value::Int(n, _) => Value::Float(*n as f64),
                        Value::Float(f) => Value::Float(*f),
                        Value::Float32(f) => Value::Float(*f as f64),
                        Value::Tensor(t) => {
                            let tensor = t.borrow();
                            let shape = tensor.shape();
                            let scalar = if shape.is_empty() {
                                tensor.get(&[])
                            } else if tensor.size() == 1 {
                                tensor.get(&vec![0usize; shape.len()])
                            } else {
                                return Err(TenthError::RuntimeError { line: None, col: None,
                                    message: format!("to_float() 不接受多元素 Tensor (shape={:?})", shape),
                                });
                            };
                            scalar.map(Value::Float).ok_or_else(|| TenthError::RuntimeError { line: None, col: None,
                                message: "to_float() Tensor 标量提取失败".into(),
                            })?
                        }
                        _ => return Err(TenthError::RuntimeError { line: None, col: None,
                            message: "to_float() 期望一个数值参数".into(),
                        }),
                    }));
                }
                return Err(TenthError::RuntimeError { line: None, col: None,
                    message: "to_float() 期望 1 个参数".into(),
                });
            }
            // to_f64 — 与 to_float 同语义（VM 侧 main.rs:1112 已注册，这里补齐解释器对齐）
            "to_f64" => {
                if let Some(arg) = args.first() {
                    let arg = deref_wrapped(arg);
                    return Ok(Some(match &arg {
                        Value::Int(n, _) => Value::Float(*n as f64),
                        Value::Float(f) => Value::Float(*f),
                        Value::Float32(f) => Value::Float(*f as f64),
                        Value::Tensor(t) => {
                            let tensor = t.borrow();
                            let shape = tensor.shape();
                            let scalar = if shape.is_empty() {
                                tensor.get(&[])
                            } else if tensor.size() == 1 {
                                tensor.get(&vec![0usize; shape.len()])
                            } else {
                                return Err(TenthError::RuntimeError { line: None, col: None,
                                    message: format!("to_f64() 不接受多元素 Tensor (shape={:?})", shape),
                                });
                            };
                            scalar.map(Value::Float).ok_or_else(|| TenthError::RuntimeError { line: None, col: None,
                                message: "to_f64() Tensor 标量提取失败".into(),
                            })?
                        }
                        _ => return Err(TenthError::RuntimeError { line: None, col: None,
                            message: "to_f64() 期望一个数值参数".into(),
                        }),
                    }));
                }
                return Err(TenthError::RuntimeError { line: None, col: None,
                    message: "to_f64() 期望 1 个参数".into(),
                });
            }
            // to_f32 — 转 f32（VM 侧 main.rs:1120 已注册）
            "to_f32" => {
                if let Some(arg) = args.first() {
                    let arg = deref_wrapped(arg);
                    return Ok(Some(match &arg {
                        Value::Int(n, _) => Value::Float32(*n as f32),
                        Value::Float(f) => Value::Float32(*f as f32),
                        Value::Float32(f) => Value::Float32(*f),
                        Value::Tensor(t) => {
                            let tensor = t.borrow();
                            let shape = tensor.shape();
                            let scalar = if shape.is_empty() {
                                tensor.get(&[])
                            } else if tensor.size() == 1 {
                                tensor.get(&vec![0usize; shape.len()])
                            } else {
                                return Err(TenthError::RuntimeError { line: None, col: None,
                                    message: format!("to_f32() 不接受多元素 Tensor (shape={:?})", shape),
                                });
                            };
                            scalar.map(|v| Value::Float32(v as f32)).ok_or_else(|| TenthError::RuntimeError { line: None, col: None,
                                message: "to_f32() Tensor 标量提取失败".into(),
                            })?
                        }
                        _ => return Err(TenthError::RuntimeError { line: None, col: None,
                            message: "to_f32() 期望一个数值参数".into(),
                        }),
                    }));
                }
                return Err(TenthError::RuntimeError { line: None, col: None,
                    message: "to_f32() 期望 1 个参数".into(),
                });
            }
            "f64_bits" => {
                if let Some(arg) = args.first() {
                    let f = arg.as_float().ok_or_else(|| TenthError::RuntimeError { line: None, col: None,
                        message: "f64_bits() 期望一个 f64 参数".into(),
                    })?;
                    return Ok(Some(Value::Int(f.to_bits() as i64, BaseType::I32)));
                }
                return Err(TenthError::RuntimeError { line: None, col: None,
                    message: "f64_bits() 期望 1 个参数".into(),
                });
            }
            "f64_from_bits" => {
                if let Some(arg) = args.first() {
                    let n = arg.as_int().ok_or_else(|| TenthError::RuntimeError { line: None, col: None,
                        message: "f64_from_bits() 期望一个 i64 参数".into(),
                    })?;
                    return Ok(Some(Value::Float(f64::from_bits(n as u64))));
                }
                return Err(TenthError::RuntimeError { line: None, col: None,
                    message: "f64_from_bits() 期望 1 个参数".into(),
                });
            }
            "tensor_from_vec" => {
                if args.len() >= 3 {
                    if let (Value::Vec(items), Value::Int(rows, _), Value::Int(cols, _)) = (&args[0], &args[1], &args[2]) {
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
                return Err(TenthError::RuntimeError { line: None, col: None,
                    message: "tensor_from_vec(vec, rows, cols) 期望一个 Vec 和两个整数".into(),
                });
            }
            "sin" => {
                if let Some(arg) = args.first() {
                    let n = arg.as_float().ok_or_else(|| TenthError::RuntimeError { line: None, col: None,
                        message: "sin() 期望一个数值参数".into(),
                    })?;
                    return Ok(Some(Value::Float(n.sin())));
                }
                return Err(TenthError::RuntimeError { line: None, col: None,
                    message: "sin() 期望 1 个参数".into(),
                });
            }
            "cos" => {
                if let Some(arg) = args.first() {
                    let n = arg.as_float().ok_or_else(|| TenthError::RuntimeError { line: None, col: None,
                        message: "cos() 期望一个数值参数".into(),
                    })?;
                    return Ok(Some(Value::Float(n.cos())));
                }
                return Err(TenthError::RuntimeError { line: None, col: None,
                    message: "cos() 期望 1 个参数".into(),
                });
            }
            "ln" => {
                if let Some(arg) = args.first() {
                    let n = arg.as_float().ok_or_else(|| TenthError::RuntimeError { line: None, col: None,
                        message: "ln() 期望一个数值参数".into(),
                    })?;
                    if n <= 0.0 {
                        return Err(TenthError::RuntimeError { line: None, col: None,
                            message: "ln() 参数必须 > 0".into(),
                        });
                    }
                    return Ok(Some(Value::Float(n.ln())));
                }
                return Err(TenthError::RuntimeError { line: None, col: None,
                    message: "ln() 期望 1 个参数".into(),
                });
            }
            "pow" => {
                if args.len() >= 2 {
                    let base = args[0].as_float().ok_or_else(|| TenthError::RuntimeError { line: None, col: None,
                        message: "pow() 期望数值参数".into(),
                    })?;
                    let exp = args[1].as_float().ok_or_else(|| TenthError::RuntimeError { line: None, col: None,
                        message: "pow() 期望数值参数".into(),
                    })?;
                    return Ok(Some(Value::Float(base.powf(exp))));
                }
                return Err(TenthError::RuntimeError { line: None, col: None,
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
                            TenthError::RuntimeError { line: None, col: None, message: "cross_entropy 中 softmax 失败".into() }
                        })?;

                        // CE loss: -mean(sum(target * log(softmax + ε)))
                        // 标准 CE 的 mean reduction 应除以 B（batch size），而非 B*V（元素总数）。
                        // B = softmax 张量除最后一维外所有前导维度的乘积（softmax 与 logits 同 shape）。
                        // 1D [V] 视为单样本 B=1；2D [B,V] 取 B；更高维取所有前导维度乘积。
                        let eps = 1e-10;
                        let sm_data = sm.data.as_standard_layout().to_owned();
                        let tgt_flat = target_data.data.as_standard_layout().to_owned();
                        let sm_slice = sm_data.as_slice().unwrap_or(&[]);
                        let tgt_slice = tgt_flat.as_slice().unwrap_or(&[]);

                        let sm_shape = sm.shape();
                        let b_size = if sm_shape.is_empty() {
                            1.0f64
                        } else {
                            sm_shape[..sm_shape.len() - 1].iter().product::<usize>() as f64
                        };
                        let mut loss_val = 0.0f64;
                        for i in 0..sm_slice.len().min(tgt_slice.len()) {
                            let p = sm_slice[i].max(eps);
                            loss_val -= tgt_slice[i] * p.ln();
                        }
                        loss_val /= b_size.max(1.0); // mean over batch

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
                return Err(TenthError::RuntimeError { line: None, col: None,
                    message: "cross_entropy(logits, target) 期望两个张量".into(),
                });
            }
            "select" => {
                // select(cond, then, else) — 逐元素条件选择原语（论文 T47/T48/T50）
                // 支持广播；cond 非 0 视为 true。可微：d_then = grad*mask, d_else = grad*(1-mask)
                if args.len() < 3 {
                    return Err(TenthError::RuntimeError { line: None, col: None,
                        message: "select(cond, then, else) 期望三个参数".into(),
                    });
                }
                let (cond, then, else_) = match (&args[0], &args[1], &args[2]) {
                    (Value::Tensor(c), Value::Tensor(t), Value::Tensor(e)) => (c.clone(), t.clone(), e.clone()),
                    _ => return Err(TenthError::RuntimeError { line: None, col: None,
                        message: "select(cond, then, else) 期望三个张量参数".into(),
                    }),
                };
                let result_tensor = Tensor::select(&cond.borrow(), &then.borrow(), &else_.borrow())
                    .map_err(|msg| TenthError::RuntimeError { line: None, col: None, message: msg })?;
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
                    return Err(TenthError::RuntimeError { line: None, col: None,
                        message: "scatter(base, dim, index, src) 期望四个参数".into(),
                    });
                }
                let dim = args[1].as_int().unwrap_or(0) as usize;
                let (base, index, src) = match (&args[0], &args[2], &args[3]) {
                    (Value::Tensor(b), Value::Tensor(i), Value::Tensor(s)) => (b.clone(), i.clone(), s.clone()),
                    _ => return Err(TenthError::RuntimeError { line: None, col: None,
                        message: "scatter(base, dim, index, src) 期望 base/index/src 为张量".into(),
                    }),
                };
                let result_tensor = Tensor::scatter(&base.borrow(), dim, &index.borrow(), &src.borrow())
                    .map_err(|msg| TenthError::RuntimeError { line: None, col: None, message: msg })?;
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
                    return Err(TenthError::RuntimeError { line: None, col: None,
                        message: "gather(base, dim, index) 期望三个参数".into(),
                    });
                }
                let dim = args[1].as_int().unwrap_or(0) as usize;
                let (base, index) = match (&args[0], &args[2]) {
                    (Value::Tensor(b), Value::Tensor(i)) => (b.clone(), i.clone()),
                    _ => return Err(TenthError::RuntimeError { line: None, col: None,
                        message: "gather(base, dim, index) 期望 base/index 为张量".into(),
                    }),
                };
                let result_tensor = Tensor::gather(&base.borrow(), dim, &index.borrow())
                    .map_err(|msg| TenthError::RuntimeError { line: None, col: None, message: msg })?;
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
            // ── PROJ-006：自定义可微算子调用 native ──────────────────────────
            // __call_custom_op(op_id, ...inputs) — 查 registry 找到 CustomBackward 实现，
            // 调用其 forward 计算输出；若 recording 则记录 TapeOp::Custom(op_id) 到 tape。
            // 与 VM 的 __call_custom_op native 语义对齐（双重注册一致性）。
            "__call_custom_op" => {
                if args.is_empty() {
                    return Err(TenthError::RuntimeError { line: None, col: None,
                        message: "__call_custom_op(op_id, ...inputs) 至少需要 op_id 参数".into() });
                }
                let op_id = match &args[0] {
                    Value::Int(n, _) => *n as usize,
                    _ => return Err(TenthError::RuntimeError { line: None, col: None,
                        message: "__call_custom_op 第一个参数必须是 op_id (Int)".into() }),
                };
                // 收集输入张量（克隆 Rc 副本，避免后续借用冲突）
                let input_tensors: Vec<Rc<RefCell<Tensor>>> = args[1..]
                    .iter()
                    .map(|v| match v {
                        Value::Tensor(t) => Ok(t.clone()),
                        _ => Err(TenthError::RuntimeError { line: None, col: None,
                            message: "__call_custom_op 输入参数必须是张量".into() }),
                    })
                    .collect::<TenthResult<_>>()?;
                // 查 registry 调用 forward（Ref 借用在 forward 返回后释放）
                let result_tensor = {
                    let registry = self.custom_ops.borrow();
                    let custom_op = match registry.get(op_id) {
                        Some(op) => op,
                        None => return Err(TenthError::RuntimeError { line: None, col: None,
                            message: format!("__call_custom_op: op_id={} 未注册", op_id) }),
                    };
                    let borrowed: Vec<std::cell::Ref<Tensor>> = input_tensors.iter().map(|t| t.borrow()).collect();
                    let input_refs: Vec<&Tensor> = borrowed.iter().map(|r| &**r).collect();
                    custom_op.forward(&input_refs).map_err(|e| TenthError::RuntimeError {
                        line: None, col: None,
                        message: format!("Custom#{} ({}) forward 失败：{}", op_id, custom_op.name(), e),
                    })?
                };
                let result = Rc::new(RefCell::new(result_tensor));
                // 若 recording，记录到 tape（与 select/scatter/gather 一致的模式）
                if self.recording {
                    if let Some(tape) = &mut self.tape {
                        let input_ids: Vec<Option<usize>> = input_tensors.iter().map(|t| t.borrow().tape_id).collect();
                        let node_id = tape.custom_op(op_id, input_ids, input_tensors, result.clone());
                        result.borrow_mut().tape_id = Some(node_id);
                    }
                }
                return Ok(Some(Value::Tensor(result)));
            }
            // ── 张量比较运算（Wave 2 第 4 项）──────────────────────────
            // 6 个比较 native：gt/lt/ge/le/eq/ne。返回 F64 张量（0.0/1.0 编码 bool）。
            // 不可微：比较结果是布尔掩码，不进入 tape（与 select 耦合可微，见标准库 where_）。
            "tensor_gt" | "tensor_lt" | "tensor_ge" | "tensor_le" | "tensor_eq" | "tensor_ne" => {
                if args.len() < 2 {
                    return Err(TenthError::RuntimeError { line: None, col: None,
                        message: format!("{}(a, b) 期望两个张量参数", name),
                    });
                }
                let (a, b) = match (&args[0], &args[1]) {
                    (Value::Tensor(x), Value::Tensor(y)) => (x.clone(), y.clone()),
                    _ => return Err(TenthError::RuntimeError { line: None, col: None,
                        message: format!("{}(a, b) 期望两个张量参数", name),
                    }),
                };
                let r = match name {
                    "tensor_gt" => a.borrow().gt(&b.borrow()),
                    "tensor_lt" => a.borrow().lt(&b.borrow()),
                    "tensor_ge" => a.borrow().ge(&b.borrow()),
                    "tensor_le" => a.borrow().le(&b.borrow()),
                    "tensor_eq" => a.borrow().eq(&b.borrow()),
                    "tensor_ne" => a.borrow().ne(&b.borrow()),
                    _ => unreachable!(),
                }.map_err(|m| TenthError::RuntimeError { line: None, col: None, message: m })?;
                return Ok(Some(Value::Tensor(Rc::new(RefCell::new(r)))));
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
                return Err(TenthError::RuntimeError { line: None, col: None,
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
            // Wave 2: f16/bf16 构造函数
            "zeros_f16" => {
                let shape: Vec<usize> = args.iter()
                    .map(|a| a.as_int().unwrap_or(1) as usize)
                    .collect();
                let t = Tensor::zeros_f16(&shape);
                return Ok(Some(Value::Tensor(Rc::new(RefCell::new(t)))));
            }
            "ones_f16" => {
                let shape: Vec<usize> = args.iter()
                    .map(|a| a.as_int().unwrap_or(1) as usize)
                    .collect();
                let t = Tensor::ones_f16(&shape);
                return Ok(Some(Value::Tensor(Rc::new(RefCell::new(t)))));
            }
            "zeros_bf16" => {
                let shape: Vec<usize> = args.iter()
                    .map(|a| a.as_int().unwrap_or(1) as usize)
                    .collect();
                let t = Tensor::zeros_bf16(&shape);
                return Ok(Some(Value::Tensor(Rc::new(RefCell::new(t)))));
            }
            "ones_bf16" => {
                let shape: Vec<usize> = args.iter()
                    .map(|a| a.as_int().unwrap_or(1) as usize)
                    .collect();
                let t = Tensor::ones_bf16(&shape);
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
                                Err(e) => return Err(TenthError::RuntimeError { line: None, col: None, message: e }),
                            }
                        } else {
                            std::path::PathBuf::from(path)
                        };
                        // args[1] can be Array or Vec of tensors
                        let tensors: &Rc<RefCell<Vec<Value>>> = match &args[1] {
                            Value::Vec(v) => v,
                            Value::Array(a) => a,
                            _ => {
                                return Err(TenthError::RuntimeError { line: None, col: None,
                                    message: "save_weights 期望一个张量列表".into(),
                                });
                            }
                        };
                            let tensors_ref = tensors.borrow();
                        let mut bytes: Vec<u8> = Vec::new();
                        // Header: magic "THW1" + version=2 + num_tensors (v2 格式)
                        bytes.extend(b"THW1");
                        bytes.extend(&2i32.to_le_bytes());
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
                                // dtype 字段: F32=0, F64=1, F16=2, BF16=3 (Wave 2)
                                let dtype_val: i32 = match &t_ref.data {
                                    TensorData::F32(_) => 0,
                                    TensorData::F64(_) => 1,
                                    TensorData::F16(_) => 2,
                                    TensorData::BF16(_) => 3,
                                };
                                bytes.extend(&dtype_val.to_le_bytes());
                                // 按 dtype 分发写数据（避免 F32→F64 cast 损失精度）
                                match &t_ref.data {
                                    TensorData::F64(arr) => {
                                        let flat = arr.as_standard_layout();
                                        if let Some(slice) = flat.as_slice() {
                                            for &x in slice { bytes.extend(&x.to_le_bytes()); }
                                        } else {
                                            for &x in flat.iter() { bytes.extend(&x.to_le_bytes()); }
                                        }
                                    }
                                    TensorData::F32(arr) => {
                                        let flat = arr.as_standard_layout();
                                        if let Some(slice) = flat.as_slice() {
                                            for &x in slice { bytes.extend(&x.to_le_bytes()); }
                                        } else {
                                            for &x in flat.iter() { bytes.extend(&x.to_le_bytes()); }
                                        }
                                    }
                                    // Wave 2: F16/BF16 各 2 字节/元素
                                    TensorData::F16(arr) => {
                                        let flat = arr.as_standard_layout();
                                        if let Some(slice) = flat.as_slice() {
                                            for &x in slice { bytes.extend(&x.to_le_bytes()); }
                                        } else {
                                            for &x in flat.iter() { bytes.extend(&x.to_le_bytes()); }
                                        }
                                    }
                                    TensorData::BF16(arr) => {
                                        let flat = arr.as_standard_layout();
                                        if let Some(slice) = flat.as_slice() {
                                            for &x in slice { bytes.extend(&x.to_le_bytes()); }
                                        } else {
                                            for &x in flat.iter() { bytes.extend(&x.to_le_bytes()); }
                                        }
                                    }
                                }
                            }
                        }
                        let _ = std::fs::write(&resolved, &bytes);
                        return Ok(Some(Value::Unit));
                    }
                }
                return Err(TenthError::RuntimeError { line: None, col: None,
                    message: "save_weights(路径, 张量列表)".into(),
                });
            }
            "load_weights" => {
                if let Some(Value::String(path)) = args.first() {
                    // H-2: 沙箱校验
                    let resolved = if let Some(ref sb) = self.fs_sandbox {
                        match sb.check_read(path) {
                            Ok(p) => p,
                            Err(e) => return Err(TenthError::RuntimeError { line: None, col: None, message: e }),
                        }
                    } else {
                        std::path::PathBuf::from(path)
                    };
                    match std::fs::read(&resolved) {
                        Ok(bytes) => {
                            if bytes.len() < 4 {
                                return Err(TenthError::RuntimeError { line: None, col: None,
                                    message: "load_weights: 文件过短".into(),
                                });
                            }
                            let mut result: Vec<Value> = Vec::new();
                            // 检测 v2 格式: magic "THW1" + version + num_tensors
                            let is_v2 = bytes.len() >= 12 && &bytes[0..4] == b"THW1";
                            let num: usize;
                            let mut offset: usize;
                            if is_v2 {
                                // v2: [magic][version:i32=2][num_tensors:i32] × [ndim][shape][dtype][data]
                                let _version = i32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
                                num = i32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize;
                                offset = 12;
                            } else {
                                // v1 旧格式: [num_tensors:i32] × [ndim][shape][data:f64×nel]
                                num = i32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as usize;
                                offset = 4;
                            }
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
                                if is_v2 {
                                    // 读 dtype 字段
                                    if offset + 4 > bytes.len() { break; }
                                    let dtype = i32::from_le_bytes([
                                        bytes[offset], bytes[offset+1], bytes[offset+2], bytes[offset+3]
                                    ]);
                                    offset += 4;
                                    match dtype {
                                        0 => {
                                            // F32: 4 字节/元素
                                            let data_len = nel * 4;
                                            if offset + data_len > bytes.len() { break; }
                                            let mut data = Vec::with_capacity(nel);
                                            for i in 0..nel {
                                                let start = offset + i * 4;
                                                let val = f32::from_le_bytes([
                                                    bytes[start], bytes[start+1], bytes[start+2], bytes[start+3],
                                                ]);
                                                data.push(val);
                                            }
                                            offset += data_len;
                                            result.push(Value::Tensor(Rc::new(RefCell::new(
                                                Tensor::from_vec_f32(data, shape)
                                            ))));
                                        }
                                        1 => {
                                            // F64: 8 字节/元素
                                            let data_len = nel * 8;
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
                                        2 => {
                                            // F16: 2 字节/元素
                                            let data_len = nel * 2;
                                            if offset + data_len > bytes.len() { break; }
                                            let mut data = Vec::with_capacity(nel);
                                            for i in 0..nel {
                                                let start = offset + i * 2;
                                                let val = half::f16::from_le_bytes([
                                                    bytes[start], bytes[start+1],
                                                ]);
                                                data.push(val);
                                            }
                                            offset += data_len;
                                            result.push(Value::Tensor(Rc::new(RefCell::new(
                                                Tensor::from_vec_f16(data, shape)
                                            ))));
                                        }
                                        3 => {
                                            // BF16: 2 字节/元素
                                            let data_len = nel * 2;
                                            if offset + data_len > bytes.len() { break; }
                                            let mut data = Vec::with_capacity(nel);
                                            for i in 0..nel {
                                                let start = offset + i * 2;
                                                let val = half::bf16::from_le_bytes([
                                                    bytes[start], bytes[start+1],
                                                ]);
                                                data.push(val);
                                            }
                                            offset += data_len;
                                            result.push(Value::Tensor(Rc::new(RefCell::new(
                                                Tensor::from_vec_bf16(data, shape)
                                            ))));
                                        }
                                        other => {
                                            return Err(TenthError::RuntimeError { line: None, col: None,
                                                message: format!("load_weights: 未知 dtype={}", other),
                                            });
                                        }
                                    }
                                } else {
                                    // 旧格式: 纯 f64 (8 字节/元素)
                                    let data_len = nel * 8;
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
                            }
                            return Ok(Some(Value::Vec(Rc::new(RefCell::new(result)))));
                        }
                        Err(e) => return Err(TenthError::RuntimeError { line: None, col: None,
                            message: format!("load_weights: {}", e),
                        }),
                    }
                }
                return Err(TenthError::RuntimeError { line: None, col: None,
                    message: "load_weights(路径)".into(),
                });
            }
            "read_file" => {
                if let Some(Value::String(path)) = args.first() {
                    // H-2: 沙箱校验
                    let resolved = if let Some(ref sb) = self.fs_sandbox {
                        match sb.check_read(path) {
                            Ok(p) => p,
                            Err(e) => return Err(TenthError::RuntimeError { line: None, col: None, message: e }),
                        }
                    } else {
                        std::path::PathBuf::from(path)
                    };
                    match std::fs::read_to_string(&resolved) {
                        Ok(content) => return Ok(Some(Value::String(content))),
                        Err(e) => return Err(TenthError::RuntimeError { line: None, col: None,
                            message: format!("读取文件失败: {}", e),
                        }),
                    }
                }
                return Err(TenthError::RuntimeError { line: None, col: None,
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
                                Err(e) => return Err(TenthError::RuntimeError { line: None, col: None, message: e }),
                            }
                        } else {
                            std::path::PathBuf::from(path)
                        };
                        match std::fs::write(&resolved, content) {
                            Ok(()) => return Ok(Some(Value::Unit)),
                            Err(e) => return Err(TenthError::RuntimeError { line: None, col: None,
                                message: format!("写入文件失败: {}", e),
                            }),
                        }
                    }
                }
                return Err(TenthError::RuntimeError { line: None, col: None,
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
                                Err(e) => return Err(TenthError::RuntimeError { line: None, col: None, message: e }),
                            }
                        } else {
                            std::path::PathBuf::from(path)
                        };
                        let data: Vec<u8> = bytes.borrow().iter().filter_map(|v| {
                            if let Value::Int(n, _) = v { Some(*n as u8) } else { None }
                        }).collect();
                        match std::fs::write(&resolved, &data) {
                            Ok(()) => return Ok(Some(Value::Unit)),
                            Err(e) => return Err(TenthError::RuntimeError { line: None, col: None,
                                message: format!("写入字节失败: {}", e),
                            }),
                        }
                    }
                }
                return Err(TenthError::RuntimeError { line: None, col: None,
                    message: "write_bytes(路径, 字节数组) 期望一个字符串和一个字节 Vec".into(),
                });
            }
            "read_bytes" => {
                if let Some(Value::String(path)) = args.first() {
                    // H-2: 沙箱校验
                    let resolved = if let Some(ref sb) = self.fs_sandbox {
                        match sb.check_read(path) {
                            Ok(p) => p,
                            Err(e) => return Err(TenthError::RuntimeError { line: None, col: None, message: e }),
                        }
                    } else {
                        std::path::PathBuf::from(path)
                    };
                    match std::fs::read(&resolved) {
                        Ok(data) => {
                            let bytes: Vec<Value> = data.iter()
                                .map(|b| Value::Int(*b as i64, BaseType::I32))
                                .collect();
                            return Ok(Some(Value::Vec(Rc::new(RefCell::new(bytes)))));
                        }
                        Err(e) => return Err(TenthError::RuntimeError { line: None, col: None,
                            message: format!("读取字节失败: {}", e),
                        }),
                    }
                }
                return Err(TenthError::RuntimeError { line: None, col: None,
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
                return Err(TenthError::RuntimeError { line: None, col: None,
                    message: "path_join(基础路径, 子路径) 期望两个字符串参数".into(),
                });
            }
            "path_exists" => {
                if let Some(Value::String(path)) = args.first() {
                    // H-2: 沙箱校验（只读检查）
                    if let Some(ref sb) = self.fs_sandbox {
                        if let Err(e) = sb.check_read(path) {
                            return Err(TenthError::RuntimeError { line: None, col: None, message: e });
                        }
                    }
                    return Ok(Some(Value::Bool(std::path::Path::new(path).exists())));
                }
                return Err(TenthError::RuntimeError { line: None, col: None,
                    message: "path_exists(路径) 期望一个字符串路径".into(),
                });
            }
            "path_is_file" => {
                if let Some(Value::String(path)) = args.first() {
                    if let Some(ref sb) = self.fs_sandbox {
                        if let Err(e) = sb.check_read(path) {
                            return Err(TenthError::RuntimeError { line: None, col: None, message: e });
                        }
                    }
                    return Ok(Some(Value::Bool(std::path::Path::new(path).is_file())));
                }
                return Err(TenthError::RuntimeError { line: None, col: None,
                    message: "path_is_file(路径) 期望一个字符串路径".into(),
                });
            }
            "path_is_dir" => {
                if let Some(Value::String(path)) = args.first() {
                    if let Some(ref sb) = self.fs_sandbox {
                        if let Err(e) = sb.check_read(path) {
                            return Err(TenthError::RuntimeError { line: None, col: None, message: e });
                        }
                    }
                    return Ok(Some(Value::Bool(std::path::Path::new(path).is_dir())));
                }
                return Err(TenthError::RuntimeError { line: None, col: None,
                    message: "path_is_dir(路径) 期望一个字符串路径".into(),
                });
            }
            "mkdir" => {
                if let Some(Value::String(path)) = args.first() {
                    // H-2: 沙箱校验（写操作）
                    let resolved = if let Some(ref sb) = self.fs_sandbox {
                        match sb.check_write(path) {
                            Ok(p) => p,
                            Err(e) => return Err(TenthError::RuntimeError { line: None, col: None, message: e }),
                        }
                    } else {
                        std::path::PathBuf::from(path)
                    };
                    match std::fs::create_dir_all(&resolved) {
                        Ok(()) => return Ok(Some(Value::Unit)),
                        Err(e) => return Err(TenthError::RuntimeError { line: None, col: None,
                            message: format!("创建目录失败: {}", e),
                        }),
                    }
                }
                return Err(TenthError::RuntimeError { line: None, col: None,
                    message: "mkdir(路径) 期望一个字符串路径".into(),
                });
            }
            "list_dir" => {
                if let Some(Value::String(path)) = args.first() {
                    // H-2: 沙箱校验
                    let resolved = if let Some(ref sb) = self.fs_sandbox {
                        match sb.check_read(path) {
                            Ok(p) => p,
                            Err(e) => return Err(TenthError::RuntimeError { line: None, col: None, message: e }),
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
                        Err(e) => return Err(TenthError::RuntimeError { line: None, col: None,
                            message: format!("列出目录失败: {}", e),
                        }),
                    }
                }
                return Err(TenthError::RuntimeError { line: None, col: None,
                    message: "list_dir(路径) 期望一个字符串路径".into(),
                });
            }
            "file_size" => {
                if let Some(Value::String(path)) = args.first() {
                    // H-2: 沙箱校验
                    let resolved = if let Some(ref sb) = self.fs_sandbox {
                        match sb.check_read(path) {
                            Ok(p) => p,
                            Err(e) => return Err(TenthError::RuntimeError { line: None, col: None, message: e }),
                        }
                    } else {
                        std::path::PathBuf::from(path)
                    };
                    match std::fs::metadata(&resolved) {
                        Ok(meta) => return Ok(Some(Value::Int(meta.len() as i64, BaseType::I32))),
                        Err(e) => return Err(TenthError::RuntimeError { line: None, col: None,
                            message: format!("获取文件大小失败: {}", e),
                        }),
                    }
                }
                return Err(TenthError::RuntimeError { line: None, col: None,
                    message: "file_size(路径) 期望一个字符串路径".into(),
                });
            }
            "remove_file" => {
                if let Some(Value::String(path)) = args.first() {
                    // H-2: 沙箱校验（写操作）
                    let resolved = if let Some(ref sb) = self.fs_sandbox {
                        match sb.check_write(path) {
                            Ok(p) => p,
                            Err(e) => return Err(TenthError::RuntimeError { line: None, col: None, message: e }),
                        }
                    } else {
                        std::path::PathBuf::from(path)
                    };
                    match std::fs::remove_file(&resolved) {
                        Ok(()) => return Ok(Some(Value::Unit)),
                        Err(e) => return Err(TenthError::RuntimeError { line: None, col: None,
                            message: format!("删除文件失败: {}", e),
                        }),
                    }
                }
                return Err(TenthError::RuntimeError { line: None, col: None,
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
                                Err(e) => return Err(TenthError::RuntimeError { line: None, col: None, message: e }),
                            }
                        } else {
                            std::path::PathBuf::from(src)
                        };
                        let dst_resolved = if let Some(ref sb) = self.fs_sandbox {
                            match sb.check_write(dst) {
                                Ok(p) => p,
                                Err(e) => return Err(TenthError::RuntimeError { line: None, col: None, message: e }),
                            }
                        } else {
                            std::path::PathBuf::from(dst)
                        };
                        match std::fs::copy(&src_resolved, &dst_resolved) {
                            Ok(_) => return Ok(Some(Value::Unit)),
                            Err(e) => return Err(TenthError::RuntimeError { line: None, col: None,
                                message: format!("复制文件失败: {}", e),
                            }),
                        }
                    }
                }
                return Err(TenthError::RuntimeError { line: None, col: None,
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
                                Err(e) => return Err(TenthError::RuntimeError { line: None, col: None, message: e }),
                            }
                        } else {
                            std::path::PathBuf::from(src)
                        };
                        let dst_resolved = if let Some(ref sb) = self.fs_sandbox {
                            match sb.check_write(dst) {
                                Ok(p) => p,
                                Err(e) => return Err(TenthError::RuntimeError { line: None, col: None, message: e }),
                            }
                        } else {
                            std::path::PathBuf::from(dst)
                        };
                        match std::fs::rename(&src_resolved, &dst_resolved) {
                            Ok(()) => return Ok(Some(Value::Unit)),
                            Err(e) => return Err(TenthError::RuntimeError { line: None, col: None,
                                message: format!("重命名文件失败: {}", e),
                            }),
                        }
                    }
                }
                return Err(TenthError::RuntimeError { line: None, col: None,
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
                                Err(e) => return Err(TenthError::RuntimeError { line: None, col: None, message: e }),
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
                                return Ok(Some(Value::Int(0, BaseType::I32)));
                            }
                            Err(_) => return Ok(Some(Value::Int(1, BaseType::I32))),
                        }
                    }
                }
                return Err(TenthError::RuntimeError { line: None, col: None,
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
                                Err(e) => return Err(TenthError::RuntimeError { line: None, col: None, message: e }),
                            }
                        } else {
                            std::path::PathBuf::from(out)
                        };
                        match crate::compile::compile_program_to_wasm(&args[0]) {
                            Ok(wasm_bytes) => {
                                let _ = std::fs::write(&out_resolved, &wasm_bytes);
                                return Ok(Some(Value::Int(0, BaseType::I32)));
                            }
                            Err(e) => {
                                eprintln!("[compile_program] error: {}", e);
                                return Ok(Some(Value::Int(1, BaseType::I32)));
                            }
                        }
                    }
                }
                return Err(TenthError::RuntimeError { line: None, col: None,
                    message: "compile_program(程序, 输出路径) 期望 Program 结构体和字符串路径".into(),
                });
            }
            "format" => {
                if args.is_empty() {
                    return Err(TenthError::RuntimeError { line: None, col: None,
                        message: "format() 至少需要一个模板字符串".into(),
                    });
                }
                if let Value::String(template) = &args[0] {
                    // Scan template to count positional placeholders and detect named placeholders.
                    let (pos_count, has_named_placeholders) = count_placeholders(template);
                    let raw_args = &args[1..];
                    let total_available = raw_args.len();
                    let positional_used = pos_count.min(total_available);
                    let excess = total_available - positional_used;
                    let named_pair_count = if has_named_placeholders { excess / 2 } else { 0 };
                    let actual_positional_count = positional_used;

                    // Build named args HashMap from key-value pairs
                    let mut named_args: HashMap<String, Value> = HashMap::new();
                    if named_pair_count > 0 {
                        let pair_start = actual_positional_count;
                        for i in 0..named_pair_count {
                            let key_idx = pair_start + i * 2;
                            let val_idx = pair_start + i * 2 + 1;
                            if key_idx < raw_args.len() && val_idx < raw_args.len() {
                                if let Value::String(key) = &raw_args[key_idx] {
                                    named_args.insert(key.clone(), raw_args[val_idx].clone());
                                }
                            }
                        }
                    }

                    let mut result = String::new();
                    let mut arg_idx: usize = 0;
                    let mut chars = template.chars().peekable();
                    while let Some(c) = chars.next() {
                        if c == '{' {
                            if chars.peek() == Some(&'{') {
                                chars.next();
                                result.push('{');
                            } else {
                                // Parse placeholder: read everything up to }
                                let mut placeholder = String::new();
                                let mut fmt_spec = String::new();
                                let mut in_spec = false;
                                while let Some(pc) = chars.next() {
                                    if pc == '}' {
                                        break;
                                    }
                                    if pc == ':' && !in_spec {
                                        in_spec = true;
                                    } else if in_spec {
                                        fmt_spec.push(pc);
                                    } else {
                                        placeholder.push(pc);
                                    }
                                }
                                let formatted = format_placeholder(
                                    &placeholder, &fmt_spec, &args,
                                    &named_args, arg_idx, actual_positional_count, named_pair_count > 0
                                )?;
                                result.push_str(&formatted);
                                if placeholder.is_empty() {
                                    arg_idx += 1;
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
                return Err(TenthError::RuntimeError { line: None, col: None,
                    message: "format() 第一个参数必须是字符串模板".into(),
                })
            }
            "parse_int" => {
                if let Some(arg) = args.first() {
                    if let Value::String(s) = arg {
                        return Ok(Some(Value::Int(s.trim().parse::<i64>().unwrap_or(0), BaseType::I32)));
                    }
                }
                return Err(TenthError::RuntimeError { line: None, col: None,
                    message: "parse_int() 期望一个字符串参数".into(),
                })
            }
            "parse_float" => {
                if let Some(arg) = args.first() {
                    if let Value::String(s) = arg {
                        return Ok(Some(Value::Float(s.trim().parse::<f64>().unwrap_or(0.0))));
                    }
                }
                return Err(TenthError::RuntimeError { line: None, col: None,
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
                if let Some(Value::Int(ms, _)) = args.first() {
                    std::thread::sleep(std::time::Duration::from_millis(*ms as u64));
                    return Ok(Some(Value::Unit));
                }
                return Err(TenthError::RuntimeError { line: None, col: None,
                    message: "time_sleep_ms(ms) 期望一个整数".into(),
                });
            }
            // —— Date native（Wave 3 第 8 项：Date 类型，路径 B 复用 struct 机制）——
            // 算法：Howard Hinnant date_algorithms（与 days_to_date 同源）。
            // 不引入新 HIR 类型——返回 i64 或 Tuple<i64,i64,i64>，date.th 用 struct 包装。
            "date_to_unix_days" => {
                match (args.first(), args.get(1), args.get(2)) {
                    (Some(Value::Int(y, _)), Some(Value::Int(m, _)), Some(Value::Int(d, _))) => {
                        return Ok(Some(Value::Int(date_to_days(*y, *m, *d), BaseType::I64)));
                    }
                    _ => return Err(TenthError::RuntimeError { line: None, col: None,
                        message: "date_to_unix_days(year, month, day) 期望三个整数".into(),
                    }),
                }
            }
            "date_from_unix_days" => {
                if let Some(Value::Int(days, _)) = args.first() {
                    let (y, m, d) = days_to_date_i64(*days);
                    return Ok(Some(Value::Tuple(vec![
                        Value::Int(y, BaseType::I64),
                        Value::Int(m, BaseType::I64),
                        Value::Int(d, BaseType::I64),
                    ])));
                }
                return Err(TenthError::RuntimeError { line: None, col: None,
                    message: "date_from_unix_days(days) 期望一个整数".into(),
                });
            }
            "date_i64_add_days" => {
                match (args.first(), args.get(1)) {
                    (Some(Value::Int(days, _)), Some(Value::Int(delta, _))) => {
                        return Ok(Some(Value::Int(days.wrapping_add(*delta), BaseType::I64)));
                    }
                    _ => return Err(TenthError::RuntimeError { line: None, col: None,
                        message: "date_i64_add_days(days, delta) 期望两个整数".into(),
                    }),
                }
            }
            "date_diff_days" => {
                match (args.first(), args.get(1)) {
                    (Some(Value::Int(d1, _)), Some(Value::Int(d2, _))) => {
                        return Ok(Some(Value::Int(d1.wrapping_sub(*d2), BaseType::I64)));
                    }
                    _ => return Err(TenthError::RuntimeError { line: None, col: None,
                        message: "date_diff_days(days1, days2) 期望两个整数".into(),
                    }),
                }
            }
            "date_day_of_week" => {
                if let Some(Value::Int(days, _)) = args.first() {
                    let w = ((*days + 4) % 7 + 7) % 7;
                    return Ok(Some(Value::Int(w, BaseType::I64)));
                }
                return Err(TenthError::RuntimeError { line: None, col: None,
                    message: "date_day_of_week(days) 期望一个整数".into(),
                });
            }
            // Random functions — 使用 rand crate 的 CSPRNG（thread_rng），
            // 与 VM 路径（runtime/natives.rs 第 941-963 行）对齐。
            // 历史 `DefaultHasher` + SystemTime 方案可被攻击者枚举纳秒时刻预测输出。
            "random_int" => {
                let lo = match args.first() {
                    Some(Value::Int(n, _)) => *n,
                    _ => 0,
                };
                let hi = match args.get(1) {
                    Some(Value::Int(n, _)) => *n,
                    _ => lo,
                };
                use rand::Rng;
                // 处理 lo > hi 的边界：交换而不是 (hi - lo + 1) 为负时回绕
                let (low, high) = if lo <= hi { (lo, hi) } else { (hi, lo) };
                // 用 u64 全域取模，避免 i64 范围回绕到负数
                let range = (high as u64).saturating_sub(low as u64).saturating_add(1).max(1);
                let r: u64 = rand::thread_rng().r#gen();
                return Ok(Some(Value::Int(low + (r % range) as i64, BaseType::I32)));
            }
            "random_float" => {
                use rand::Rng;
                // [0, 1) 半开区间，标准做法
                let r: f64 = rand::thread_rng().r#gen();
                return Ok(Some(Value::Float(r)));
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
                return Ok(Some(Value::Int(1, BaseType::I32))); // Default: just program name
            }
            "cli_arg" => {
                if let Some(Value::Int(_idx, _)) = args.first() {
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
            // —— 断言原语 ——
            "assert" => {
                if args.is_empty() {
                    return Err(TenthError::RuntimeError { line: None, col: None,
                        message: "assert() 需要至少一个参数（布尔条件）".into(),
                    });
                }
                match &args[0] {
                    Value::Bool(true) => return Ok(Some(Value::Unit)),
                    Value::Bool(false) => {
                        let msg = if args.len() >= 2 {
                            if let Value::String(s) = &args[1] { s.clone() } else { format!("{}", args[1]) }
                        } else {
                            "assertion failed".to_string()
                        };
                        return Err(TenthError::RuntimeError { line: None, col: None, message: msg });
                    }
                    _ => return Err(TenthError::RuntimeError { line: None, col: None,
                        message: "assert() 需要一个布尔值作为第一个参数".into(),
                    }),
                }
            }
            "assert_eq" => {
                if args.len() < 2 {
                    return Err(TenthError::RuntimeError { line: None, col: None,
                        message: "assert_eq() 需要至少两个参数（左值，右值）".into(),
                    });
                }
                let left_str = format!("{}", args[0]);
                let right_str = format!("{}", args[1]);
                if left_str == right_str {
                    return Ok(Some(Value::Unit));
                } else {
                    let extra = if args.len() >= 3 {
                        if let Value::String(s) = &args[2] {
                            if !s.is_empty() { format!(" — {}", s) } else { String::new() }
                        } else { String::new() }
                    } else { String::new() };
                    return Err(TenthError::RuntimeError { line: None, col: None,
                        message: format!("assertion failed: {} != {}{}", left_str, right_str, extra),
                    });
                }
            }
            // ── 问题29：智能指针 Box/Rc/Arc/Pin ──
            "Box::new" => {
                if args.is_empty() {
                    return Err(TenthError::RuntimeError { line: None, col: None, message: "Box::new() 需要 1 个参数".into() });
                }
                return Ok(Some(Value::HeapBox(Box::new(args[0].clone()))));
            }
            "Rc::new" => {
                if args.is_empty() {
                    return Err(TenthError::RuntimeError { line: None, col: None, message: "Rc::new() 需要 1 个参数".into() });
                }
                return Ok(Some(Value::SharedBox(Rc::new(RefCell::new(args[0].clone())))));
            }
            "Arc::new" => {
                if args.is_empty() {
                    return Err(TenthError::RuntimeError { line: None, col: None, message: "Arc::new() 需要 1 个参数".into() });
                }
                return Ok(Some(Value::SharedBox(Rc::new(RefCell::new(args[0].clone())))));
            }
            "Pin::new" => {
                if args.is_empty() {
                    return Err(TenthError::RuntimeError { line: None, col: None, message: "Pin::new() 需要 1 个参数".into() });
                }
                return Ok(Some(Value::Pin(Box::new(args[0].clone()))));
            }
            // ── 问题35：BigInt 运算 ──
            "bigint_add" => {
                if args.len() < 2 { return Err(TenthError::RuntimeError { line: None, col: None, message: "bigint_add() 需要 2 个 bigint 参数".into() }); }
                let a = match &args[0] { Value::BigInt(s) => s.clone(), _ => return Err(TenthError::RuntimeError { line: None, col: None, message: "参数必须是 BigInt".into() }) };
                let b = match &args[1] { Value::BigInt(s) => s.clone(), _ => return Err(TenthError::RuntimeError { line: None, col: None, message: "参数必须是 BigInt".into() }) };
                return Ok(Some(Value::BigInt(crate::runtime::natives::bigint_add_str(&a, &b))));
            }
            "bigint_sub" => {
                if args.len() < 2 { return Err(TenthError::RuntimeError { line: None, col: None, message: "bigint_sub() 需要 2 个 bigint 参数".into() }); }
                let a = match &args[0] { Value::BigInt(s) => s.clone(), _ => return Err(TenthError::RuntimeError { line: None, col: None, message: "参数必须是 BigInt".into() }) };
                let b = match &args[1] { Value::BigInt(s) => s.clone(), _ => return Err(TenthError::RuntimeError { line: None, col: None, message: "参数必须是 BigInt".into() }) };
                return Ok(Some(Value::BigInt(crate::runtime::natives::bigint_sub_str(&a, &b))));
            }
            "bigint_mul" => {
                if args.len() < 2 { return Err(TenthError::RuntimeError { line: None, col: None, message: "bigint_mul() 需要 2 个 bigint 参数".into() }); }
                let a = match &args[0] { Value::BigInt(s) => s.clone(), _ => return Err(TenthError::RuntimeError { line: None, col: None, message: "参数必须是 BigInt".into() }) };
                let b = match &args[1] { Value::BigInt(s) => s.clone(), _ => return Err(TenthError::RuntimeError { line: None, col: None, message: "参数必须是 BigInt".into() }) };
                return Ok(Some(Value::BigInt(crate::runtime::natives::bigint_mul_str(&a, &b))));
            }
            // ── 问题36：Complex 运算 ──
            "complex_add" => {
                if args.len() < 2 { return Err(TenthError::RuntimeError { line: None, col: None, message: "complex_add() 需要 2 个 complex 参数".into() }); }
                let (re1, im1) = match &args[0] { Value::Complex(r, i) => (*r, *i), _ => return Err(TenthError::RuntimeError { line: None, col: None, message: "参数必须是 Complex".into() }) };
                let (re2, im2) = match &args[1] { Value::Complex(r, i) => (*r, *i), _ => return Err(TenthError::RuntimeError { line: None, col: None, message: "参数必须是 Complex".into() }) };
                return Ok(Some(Value::Complex(re1 + re2, im1 + im2)));
            }
            "complex_sub" => {
                if args.len() < 2 { return Err(TenthError::RuntimeError { line: None, col: None, message: "complex_sub() 需要 2 个 complex 参数".into() }); }
                let (re1, im1) = match &args[0] { Value::Complex(r, i) => (*r, *i), _ => return Err(TenthError::RuntimeError { line: None, col: None, message: "参数必须是 Complex".into() }) };
                let (re2, im2) = match &args[1] { Value::Complex(r, i) => (*r, *i), _ => return Err(TenthError::RuntimeError { line: None, col: None, message: "参数必须是 Complex".into() }) };
                return Ok(Some(Value::Complex(re1 - re2, im1 - im2)));
            }
            "complex_mul" => {
                if args.len() < 2 { return Err(TenthError::RuntimeError { line: None, col: None, message: "complex_mul() 需要 2 个 complex 参数".into() }); }
                let (re1, im1) = match &args[0] { Value::Complex(r, i) => (*r, *i), _ => return Err(TenthError::RuntimeError { line: None, col: None, message: "参数必须是 Complex".into() }) };
                let (re2, im2) = match &args[1] { Value::Complex(r, i) => (*r, *i), _ => return Err(TenthError::RuntimeError { line: None, col: None, message: "参数必须是 Complex".into() }) };
                return Ok(Some(Value::Complex(re1 * re2 - im1 * im2, re1 * im2 + im1 * re2)));
            }
            "complex_div" => {
                if args.len() < 2 { return Err(TenthError::RuntimeError { line: None, col: None, message: "complex_div() 需要 2 个 complex 参数".into() }); }
                let (re1, im1) = match &args[0] { Value::Complex(r, i) => (*r, *i), _ => return Err(TenthError::RuntimeError { line: None, col: None, message: "参数必须是 Complex".into() }) };
                let (re2, im2) = match &args[1] { Value::Complex(r, i) => (*r, *i), _ => return Err(TenthError::RuntimeError { line: None, col: None, message: "参数必须是 Complex".into() }) };
                let denom = re2 * re2 + im2 * im2;
                if denom == 0.0 {
                    return Err(TenthError::RuntimeError { line: None, col: None, message: "Complex 除法：分母为零".into() });
                }
                return Ok(Some(Value::Complex((re1 * re2 + im1 * im2) / denom, (im1 * re2 - re1 * im2) / denom)));
            }
            // ── 问题37：Decimal 运算 ──
            "decimal_add" => {
                if args.len() < 2 { return Err(TenthError::RuntimeError { line: None, col: None, message: "decimal_add() 需要 2 个 decimal 参数".into() }); }
                let a = match &args[0] { Value::Decimal(s) => s.clone(), _ => return Err(TenthError::RuntimeError { line: None, col: None, message: "参数必须是 Decimal".into() }) };
                let b = match &args[1] { Value::Decimal(s) => s.clone(), _ => return Err(TenthError::RuntimeError { line: None, col: None, message: "参数必须是 Decimal".into() }) };
                return Ok(Some(Value::Decimal(crate::runtime::natives::decimal_add_str(&a, &b))));
            }
            "decimal_sub" => {
                if args.len() < 2 { return Err(TenthError::RuntimeError { line: None, col: None, message: "decimal_sub() 需要 2 个 decimal 参数".into() }); }
                let a = match &args[0] { Value::Decimal(s) => s.clone(), _ => return Err(TenthError::RuntimeError { line: None, col: None, message: "参数必须是 Decimal".into() }) };
                let b = match &args[1] { Value::Decimal(s) => s.clone(), _ => return Err(TenthError::RuntimeError { line: None, col: None, message: "参数必须是 Decimal".into() }) };
                return Ok(Some(Value::Decimal(crate::runtime::natives::decimal_sub_str(&a, &b))));
            }
            "decimal_mul" => {
                if args.len() < 2 { return Err(TenthError::RuntimeError { line: None, col: None, message: "decimal_mul() 需要 2 个 decimal 参数".into() }); }
                let a = match &args[0] { Value::Decimal(s) => s.clone(), _ => return Err(TenthError::RuntimeError { line: None, col: None, message: "参数必须是 Decimal".into() }) };
                let b = match &args[1] { Value::Decimal(s) => s.clone(), _ => return Err(TenthError::RuntimeError { line: None, col: None, message: "参数必须是 Decimal".into() }) };
                return Ok(Some(Value::Decimal(crate::runtime::natives::decimal_mul_str(&a, &b))));
            }
            "decimal_div" => {
                if args.len() < 2 { return Err(TenthError::RuntimeError { line: None, col: None, message: "decimal_div() 需要 2 个 decimal 参数".into() }); }
                let a = match &args[0] { Value::Decimal(s) => s.clone(), _ => return Err(TenthError::RuntimeError { line: None, col: None, message: "参数必须是 Decimal".into() }) };
                let b = match &args[1] { Value::Decimal(s) => s.clone(), _ => return Err(TenthError::RuntimeError { line: None, col: None, message: "参数必须是 Decimal".into() }) };
                return Ok(Some(Value::Decimal(crate::runtime::natives::decimal_div_str(&a, &b))));
            }
            // ── B批：Unicode 规范化 ──
            "unicode_nfc" => {
                if let Some(Value::String(s)) = args.first() {
                    return Ok(Some(Value::String(s.chars().nfc().collect::<String>())));
                }
                return Err(TenthError::RuntimeError { line: None, col: None, message: "_unicode_nfc 需要 1 个 String 参数".into() });
            }
            "unicode_nfd" => {
                if let Some(Value::String(s)) = args.first() {
                    return Ok(Some(Value::String(s.chars().nfd().collect::<String>())));
                }
                return Err(TenthError::RuntimeError { line: None, col: None, message: "_unicode_nfd 需要 1 个 String 参数".into() });
            }
            // ── B批：UTF-8 ↔ UTF-16 ──
            "str_to_utf16" => {
                if let Some(Value::String(s)) = args.first() {
                    let encoded: Vec<u16> = s.encode_utf16().collect();
                    let result: Vec<Value> = encoded.into_iter()
                        .map(|c| Value::Int(c as i64, BaseType::I32))
                        .collect();
                    return Ok(Some(Value::Vec(Rc::new(RefCell::new(result)))));
                }
                return Err(TenthError::RuntimeError { line: None, col: None, message: "_str_to_utf16 需要 1 个 String 参数".into() });
            }
            "_utf16_to_str" => {
                if let Some(Value::Vec(arr)) = args.first() {
                    let code_units: Vec<u16> = arr.borrow().iter()
                        .map(|v| match v {
                            Value::Int(n, _) => *n as u16,
                            _ => 0,
                        })
                        .collect();
                    let result = String::from_utf16(&code_units)
                        .unwrap_or_else(|_| {
                            let mut s = String::new();
                            let mut i = 0;
                            while i < code_units.len() {
                                let c = code_units[i];
                                if c >= 0xD800 && c <= 0xDBFF && i + 1 < code_units.len() {
                                    let c2 = code_units[i + 1];
                                    if c2 >= 0xDC00 && c2 <= 0xDFFF {
                                        let cp = 0x10000 + ((c as u32 - 0xD800) << 10) + (c2 as u32 - 0xDC00);
                                        if let Some(ch) = char::from_u32(cp) {
                                            s.push(ch);
                                        } else { s.push('\u{FFFD}'); }
                                        i += 2;
                                        continue;
                                    }
                                } else if c >= 0xDC00 && c <= 0xDFFF {
                                    s.push('\u{FFFD}');
                                    i += 1;
                                    continue;
                                }
                                if let Some(ch) = char::from_u32(c as u32) { s.push(ch); } else { s.push('\u{FFFD}'); }
                                i += 1;
                            }
                            s
                        });
                    return Ok(Some(Value::String(result)));
                }
                return Err(TenthError::RuntimeError { line: None, col: None, message: "_utf16_to_str 需要 1 个 Vec 参数".into() });
            }
            // ── B批：UTF-8 ↔ 字节数组 ──
            "str_to_bytes" => {
                if let Some(Value::String(s)) = args.first() {
                    let bytes: Vec<Value> = s.bytes()
                        .map(|b| Value::Int(b as i64, BaseType::I32))
                        .collect();
                    return Ok(Some(Value::Vec(Rc::new(RefCell::new(bytes)))));
                }
                return Err(TenthError::RuntimeError { line: None, col: None, message: "_str_to_bytes 需要 1 个 String 参数".into() });
            }
            "_bytes_to_str" => {
                if let Some(Value::Vec(arr)) = args.first() {
                    let bytes: Vec<u8> = arr.borrow().iter()
                        .map(|v| match v {
                            Value::Int(n, _) => *n as u8,
                            _ => 0,
                        })
                        .collect();
                    return Ok(Some(Value::String(String::from_utf8_lossy(&bytes).to_string())));
                }
                return Err(TenthError::RuntimeError { line: None, col: None, message: "_bytes_to_str 需要 1 个 Vec 参数".into() });
            }
            // ── B批：Base64 ──
            "base64_encode" => {
                if let Some(Value::Vec(arr)) = args.first() {
                    let bytes: Vec<u8> = arr.borrow().iter()
                        .map(|v| match v {
                            Value::Int(n, _) => *n as u8,
                            _ => 0,
                        })
                        .collect();
                    use base64::engine::general_purpose;
                    return Ok(Some(Value::String(general_purpose::STANDARD.encode(&bytes))));
                }
                return Err(TenthError::RuntimeError { line: None, col: None, message: "base64_encode 需要 1 个 Vec 参数".into() });
            }
            "base64_decode" => {
                if let Some(Value::String(s)) = args.first() {
                    use base64::engine::general_purpose;
                    match general_purpose::STANDARD.decode(s) {
                        Ok(bytes) => {
                            let result: Vec<Value> = bytes.into_iter()
                                .map(|b| Value::Int(b as i64, BaseType::I32))
                                .collect();
                            return Ok(Some(ok_result(Value::Vec(Rc::new(RefCell::new(result))))));
                        }
                        Err(e) => return Ok(Some(err_result(format!("Base64 解码失败: {e}")))),
                    }
                }
                return Ok(Some(err_result("_base64_decode 需要 1 个 String 参数")));
            }
            // ── B批：十六进制 ──
            "hex_encode" => {
                if let Some(Value::Vec(arr)) = args.first() {
                    let bytes: Vec<u8> = arr.borrow().iter()
                        .map(|v| match v {
                            Value::Int(n, _) => *n as u8,
                            _ => 0,
                        })
                        .collect();
                    return Ok(Some(Value::String(bytes.iter().map(|b| format!("{:02x}", b)).collect::<String>())));
                }
                return Err(TenthError::RuntimeError { line: None, col: None, message: "hex_encode 需要 1 个 Vec 参数".into() });
            }
            "hex_decode" => {
                if let Some(Value::String(s)) = args.first() {
                    let trimmed = s.trim();
                    if trimmed.len() % 2 != 0 {
                        return Ok(Some(err_result("十六进制字符串长度必须为偶数")));
                    }
                    let bytes: Result<Vec<u8>, _> = (0..trimmed.len())
                        .step_by(2)
                        .map(|i| u8::from_str_radix(&trimmed[i..i+2], 16))
                        .collect();
                    match bytes {
                        Ok(data) => {
                            let result: Vec<Value> = data.into_iter()
                                .map(|b| Value::Int(b as i64, BaseType::I32))
                                .collect();
                            return Ok(Some(ok_result(Value::Vec(Rc::new(RefCell::new(result))))));
                        }
                        Err(e) => return Ok(Some(err_result(format!("十六进制解码失败: {e}")))),
                    }
                }
                return Ok(Some(err_result("_hex_decode 需要 1 个 String 参数")));
            }
            // ── B批：URL 编解码 ──
            "url_encode" => {
                if let Some(Value::String(s)) = args.first() {
                    use percent_encoding::{utf8_percent_encode, NON_ALPHANUMERIC};
                    return Ok(Some(Value::String(utf8_percent_encode(s, NON_ALPHANUMERIC).to_string())));
                }
                return Err(TenthError::RuntimeError { line: None, col: None, message: "_url_encode 需要 1 个 String 参数".into() });
            }
            "url_decode" => {
                if let Some(Value::String(s)) = args.first() {
                    use percent_encoding::percent_decode;
                    match percent_decode(s.as_bytes()).decode_utf8() {
                        Ok(decoded) => return Ok(Some(ok_result(Value::String(decoded.to_string())))),
                        Err(_) => return Ok(Some(err_result("URL 解码失败：无效的百分号编码序列"))),
                    }
                }
                return Ok(Some(err_result("_url_decode 需要 1 个 String 参数")));
            }
            // ── 哈希函数（SHA-256/SHA-512/MD5） ──
            // 接受 Vec<u8>（Vec<i64>，每个元素 0-255），返回小写 hex 字符串
            // 注：数组字面量元素在解释器中被包裹为 Value::Shared，需先 deref_wrapped
            "sha256" => {
                if let Some(Value::Vec(arr)) = args.first() {
                    let bytes: Vec<u8> = arr.borrow().iter()
                        .map(|v| match deref_wrapped(v) {
                            Value::Int(n, _) => n as u8,
                            _ => 0,
                        })
                        .collect();
                    use sha2::{Sha256, Digest};
                    let mut hasher = Sha256::new();
                    hasher.update(&bytes);
                    let result = hasher.finalize();
                    return Ok(Some(Value::String(result.iter().map(|b| format!("{:02x}", b)).collect())));
                }
                return Err(TenthError::RuntimeError { line: None, col: None, message: "sha256 需要 1 个 Vec 参数".into() });
            }
            "sha512" => {
                if let Some(Value::Vec(arr)) = args.first() {
                    let bytes: Vec<u8> = arr.borrow().iter()
                        .map(|v| match deref_wrapped(v) {
                            Value::Int(n, _) => n as u8,
                            _ => 0,
                        })
                        .collect();
                    use sha2::{Sha512, Digest};
                    let mut hasher = Sha512::new();
                    hasher.update(&bytes);
                    let result = hasher.finalize();
                    return Ok(Some(Value::String(result.iter().map(|b| format!("{:02x}", b)).collect())));
                }
                return Err(TenthError::RuntimeError { line: None, col: None, message: "sha512 需要 1 个 Vec 参数".into() });
            }
            "md5" => {
                if let Some(Value::Vec(arr)) = args.first() {
                    let bytes: Vec<u8> = arr.borrow().iter()
                        .map(|v| match deref_wrapped(v) {
                            Value::Int(n, _) => n as u8,
                            _ => 0,
                        })
                        .collect();
                    use md5::{Md5, Digest};
                    let mut hasher = Md5::new();
                    hasher.update(&bytes);
                    let result = hasher.finalize();
                    return Ok(Some(Value::String(result.iter().map(|b| format!("{:02x}", b)).collect())));
                }
                return Err(TenthError::RuntimeError { line: None, col: None, message: "md5 需要 1 个 Vec 参数".into() });
            }
            // 便捷版：接受 String（对 UTF-8 字节哈希），返回 hex 字符串
            "sha256_str" => {
                if let Some(Value::String(s)) = args.first() {
                    let bytes = s.as_bytes();
                    use sha2::{Sha256, Digest};
                    let mut hasher = Sha256::new();
                    hasher.update(bytes);
                    let result = hasher.finalize();
                    return Ok(Some(Value::String(result.iter().map(|b| format!("{:02x}", b)).collect())));
                }
                return Err(TenthError::RuntimeError { line: None, col: None, message: "sha256_str 需要 1 个 String 参数".into() });
            }
            "sha512_str" => {
                if let Some(Value::String(s)) = args.first() {
                    let bytes = s.as_bytes();
                    use sha2::{Sha512, Digest};
                    let mut hasher = Sha512::new();
                    hasher.update(bytes);
                    let result = hasher.finalize();
                    return Ok(Some(Value::String(result.iter().map(|b| format!("{:02x}", b)).collect())));
                }
                return Err(TenthError::RuntimeError { line: None, col: None, message: "sha512_str 需要 1 个 String 参数".into() });
            }
            "md5_str" => {
                if let Some(Value::String(s)) = args.first() {
                    let bytes = s.as_bytes();
                    use md5::{Md5, Digest};
                    let mut hasher = Md5::new();
                    hasher.update(bytes);
                    let result = hasher.finalize();
                    return Ok(Some(Value::String(result.iter().map(|b| format!("{:02x}", b)).collect())));
                }
                return Err(TenthError::RuntimeError { line: None, col: None, message: "md5_str 需要 1 个 String 参数".into() });
            }
            // ── B批：编码转换新 API 别名 ──
            "_to_utf8" => {
                if let Some(Value::String(s)) = args.first() {
                    let bytes: Vec<Value> = s.bytes()
                        .map(|b| Value::Int(b as i64, BaseType::I32))
                        .collect();
                    return Ok(Some(Value::Vec(Rc::new(RefCell::new(bytes)))));
                }
                return Err(TenthError::RuntimeError { line: None, col: None, message: "_to_utf8 需要 1 个 String 参数".into() });
            }
            "_to_utf16" => {
                if let Some(Value::String(s)) = args.first() {
                    let encoded: Vec<u16> = s.encode_utf16().collect();
                    let result: Vec<Value> = encoded.into_iter()
                        .map(|c| Value::Int(c as i64, BaseType::I32))
                        .collect();
                    return Ok(Some(Value::Vec(Rc::new(RefCell::new(result)))));
                }
                return Err(TenthError::RuntimeError { line: None, col: None, message: "_to_utf16 需要 1 个 String 参数".into() });
            }
            "from_utf16" => {
                if let Some(Value::Vec(arr)) = args.first() {
                    let code_units: Vec<u16> = arr.borrow().iter()
                        .map(|v| match v {
                            Value::Int(n, _) => *n as u16,
                            _ => 0,
                        })
                        .collect();
                    let result = String::from_utf16(&code_units)
                        .unwrap_or_else(|_| {
                            let mut s = String::new();
                            let mut i = 0;
                            while i < code_units.len() {
                                let c = code_units[i];
                                if c >= 0xD800 && c <= 0xDBFF && i + 1 < code_units.len() {
                                    let c2 = code_units[i + 1];
                                    if c2 >= 0xDC00 && c2 <= 0xDFFF {
                                        let cp = 0x10000 + ((c as u32 - 0xD800) << 10) + (c2 as u32 - 0xDC00);
                                        if let Some(ch) = char::from_u32(cp) {
                                            s.push(ch);
                                        } else { s.push('\u{FFFD}'); }
                                        i += 2;
                                        continue;
                                    }
                                } else if c >= 0xDC00 && c <= 0xDFFF {
                                    s.push('\u{FFFD}');
                                    i += 1;
                                    continue;
                                }
                                if let Some(ch) = char::from_u32(c as u32) { s.push(ch); } else { s.push('\u{FFFD}'); }
                                i += 1;
                            }
                            s
                        });
                    return Ok(Some(Value::String(result)));
                }
                return Err(TenthError::RuntimeError { line: None, col: None, message: "_from_utf16 需要 1 个 Vec 参数".into() });
            }
            // ── B批：GBK 编码 ──
            "to_gbk" => {
                if let Some(Value::String(s)) = args.first() {
                    let (bytes, _, _) = encoding_rs::GBK.encode(s);
                    let result: Vec<Value> = bytes.iter()
                        .map(|b| Value::Int(*b as i64, BaseType::I32))
                        .collect();
                    return Ok(Some(Value::Vec(Rc::new(RefCell::new(result)))));
                }
                return Err(TenthError::RuntimeError { line: None, col: None, message: "_to_gbk 需要 1 个 String 参数".into() });
            }
            "from_gbk" => {
                if let Some(Value::Vec(arr)) = args.first() {
                    let bytes: Vec<u8> = arr.borrow().iter()
                        .map(|v| match v {
                            Value::Int(n, _) => *n as u8,
                            _ => 0,
                        })
                        .collect();
                    let (result, _, _) = encoding_rs::GBK.decode(&bytes);
                    return Ok(Some(Value::String(result.to_string())));
                }
                return Err(TenthError::RuntimeError { line: None, col: None, message: "_from_gbk 需要 1 个 Vec 参数".into() });
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
                        self.push_scope();

                        for ((pname, _), arg) in fn_def.params.iter().zip(args.iter()) {
                            self.insert_var(pname.clone(), arg.clone());
                        }

                        let result = self.eval_expr(&fn_def.body);

                        self.pop_scope();

                        return Self::unwrap_return(result);
                    }
                }
                return Err(TenthError::RuntimeError { line: None, col: None,
                    message: format!("未定义函数 '{}'", name),
                });
            }
        }

        let func_def = self.functions.iter().find(|f| f.name == name).cloned();
        if let Some(fd) = func_def {
            // Push a new scope for function-local variables.
            // Parameters and locals are isolated; globals remain visible via scope chain.
            self.push_scope();

            for ((pname, _), arg) in fd.params.iter().zip(args.iter()) {
                self.insert_var(pname.clone(), arg.clone());
            }

            let result = self.eval_expr(&fd.body);

            self.pop_scope();

            return Self::unwrap_return(result);
        }

        Err(TenthError::RuntimeError { line: None, col: None,
            message: format!("undefined function '{}'", name),
        })
    }
}

