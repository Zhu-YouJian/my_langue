//! Native 函数集中注册：`register_all_natives(vm)`。
//!
//! 从 `main.rs` 迁移而来（T1.3）。原 `register_natives` 改名为
//! `register_all_natives`，作为后续 T2 NativeRegistry 统一的入口点。
//!
//! 包含所有 VM native 函数注册：I/O、TCP 网络、正则表达式、HTTP 客户端、
//! 文件系统（沙箱校验）、时间、异步 I/O、随机数、数学、CLI、JSON、编译、
//! 自动微分（new_grad/param/backward/grad/select/scatter/gather/cross_entropy）、
//! 张量构造（zeros/ones/rand/zeros_f32/...）、序列化（save/load_weights）、
//! 类型转换（to_float/to_f64/to_f32/to_string/type_name）、字符串格式化等。

use std::cell::RefCell;
use std::collections::HashMap;
use crate::hir::types::BaseType;
use std::rc::Rc;

use crate::error::{TenthError, TenthResult};
use crate::runtime::async_io::{ASYNC_IO, IoResult};
use crate::runtime::autodiff::Tape;
use crate::runtime::interpreter::datetime;
use crate::runtime::interpreter::json;
use crate::runtime::tensor::{Tensor, TensorData};
use crate::runtime::value::Value;
use crate::runtime::vm::Vm;
use crate::http::{http_get_impl, http_post_impl};

// B批：编码工具
use unicode_normalization::UnicodeNormalization;
use base64::Engine as _;

/// 构造 Result::Ok(value)
pub(crate) fn ok_result(value: Value) -> Value {
    Value::Enum {
        enum_name: "Result".to_string(),
        variant: "Ok".to_string(),
        fields: Rc::new(RefCell::new(vec![("_0".to_string(), value)])),
    }
}

/// 构造 Result::Err(message)
pub(crate) fn err_result(msg: impl Into<String>) -> Value {
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
/// 位置占位符：`{}`、`{:spec}`、`{0}`、`{0:spec}`
/// 命名占位符：`{name}`、`{name:spec}`
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

/// 注册所有 native 函数到 VM。
///
/// 从 `main.rs::register_natives` 迁移并改名。函数体保持不变，
/// 仅调整 imports 路径（`tenth::` → `crate::`，http 函数改用 `crate::http`）。
pub fn register_all_natives(vm: &mut Vm) {
    vm.add_native("println".into(), |_vm, args| {
        for a in args { print!("{a}"); }
        println!();
        Ok(Value::Unit)
    });

    // —— I/O 原语：stderr + stdin ——
    vm.add_native("eprint".into(), |_vm, args| {
        use std::io::Write;
        let stderr = std::io::stderr();
        let mut handle = stderr.lock();
        for a in args { write!(handle, "{a}").ok(); }
        Ok(Value::Unit)
    });
    vm.add_native("eprintln".into(), |_vm, args| {
        use std::io::Write;
        let stderr = std::io::stderr();
        let mut handle = stderr.lock();
        for a in args { write!(handle, "{a}").ok(); }
        writeln!(handle).ok();
        Ok(Value::Unit)
    });
    vm.add_native("read_line".into(), |_vm, _args| {
        use std::io::BufRead;
        let stdin = std::io::stdin();
        let mut line = String::new();
        match stdin.lock().read_line(&mut line) {
            Ok(0) => Ok(err_result("EOF")),
            Ok(_) => {
                if line.ends_with('\n') { line.pop(); if line.ends_with('\r') { line.pop(); } }
                Ok(ok_result(Value::String(line)))
            }
            Err(e) => Ok(err_result(format!("读取输入失败: {e}"))),
        }
    });

    // —— 环境变量 + 进程控制 ——
    vm.add_native("env_get".into(), |_vm, args| {
        if let Some(Value::String(name)) = args.first() {
            match std::env::var(name) {
                Ok(val) => Ok(ok_result(Value::String(val))),
                Err(_) => Ok(err_result("环境变量不存在")),
            }
        } else {
            Ok(err_result("env_get 需要 1 个 String 参数"))
        }
    });
    vm.add_native("env_set".into(), |_vm, args| {
        if args.len() >= 2 {
            if let (Value::String(name), Value::String(val)) = (&args[0], &args[1]) {
                // Rust 2024 edition: set_var is unsafe
                unsafe { std::env::set_var(name, val); }
            }
        }
        Ok(Value::Unit)
    });
    vm.add_native("exit".into(), |_vm, args| {
        let code = if let Some(Value::Int(c, _)) = args.first() { *c } else { 0 };
        std::process::exit(code as i32);
    });

    // —— TCP 网络原语（句柄表方案，handle 1-based，0 表示无效）——
    vm.add_native("tcp_connect".into(), |vm, args| {
        if args.len() < 2 {
            return Ok(err_result("tcp_connect 需要 (String, i64) 参数"));
        }
        if let (Value::String(host), Value::Int(port, _)) = (&args[0], &args[1]) {
            let addr = format!("{}:{}", host, port);
            match std::net::TcpStream::connect(&addr) {
                Ok(stream) => {
                    vm.tcp_streams.push(Some(stream));
                    let handle = vm.tcp_streams.len() as i64; // 1-based
                    Ok(ok_result(Value::Int(handle, BaseType::I32)))
                }
                Err(e) => Ok(err_result(format!("连接失败: {e}"))),
            }
        } else {
            Ok(err_result("tcp_connect 需要 (String, i64) 参数"))
        }
    });
    vm.add_native("tcp_read".into(), |vm, args| {
        if args.len() < 2 {
            return Ok(err_result("tcp_read 需要 (i64, i64) 参数"));
        }
        if let (Value::Int(handle, _), Value::Int(n, _)) = (&args[0], &args[1]) {
            let idx = *handle as usize;
            if idx == 0 || idx > vm.tcp_streams.len() {
                return Ok(err_result("无效的句柄"));
            }
            let max = (*n).max(0).min(65536) as usize;
            if let Some(ref mut stream) = vm.tcp_streams[idx - 1] {
                use std::io::Read;
                let mut buf = vec![0u8; max];
                match stream.read(&mut buf) {
                    Ok(0) => {
                        // EOF：返回空 Vec
                        Ok(ok_result(Value::Vec(Rc::new(RefCell::new(Vec::new())))))
                    }
                    Ok(read_n) => {
                        let bytes: Vec<Value> = buf[..read_n]
                            .iter()
                            .map(|b| Value::Int(*b as i64, BaseType::I32))
                            .collect();
                        Ok(ok_result(Value::Vec(Rc::new(RefCell::new(bytes)))))
                    }
                    Err(e) => Ok(err_result(format!("读取失败: {e}"))),
                }
            } else {
                Ok(err_result("连接已关闭"))
            }
        } else {
            Ok(err_result("tcp_read 需要 (i64, i64) 参数"))
        }
    });
    vm.add_native("tcp_write".into(), |vm, args| {
        if args.len() < 2 {
            return Ok(err_result("tcp_write 需要 (i64, Vec<i64>) 参数"));
        }
        if let (Value::Int(handle, _), Value::Vec(data)) = (&args[0], &args[1]) {
            let idx = *handle as usize;
            if idx == 0 || idx > vm.tcp_streams.len() {
                return Ok(err_result("无效的句柄"));
            }
            let bytes: Vec<u8> = data
                .borrow()
                .iter()
                .map(|x| match x {
                    Value::Int(b, _) => *b as u8,
                    _ => 0,
                })
                .collect();
            if let Some(ref mut stream) = vm.tcp_streams[idx - 1] {
                use std::io::Write;
                match stream.write_all(&bytes) {
                    Ok(_) => Ok(ok_result(Value::Int(bytes.len() as i64, BaseType::I32))),
                    Err(e) => Ok(err_result(format!("写入失败: {e}"))),
                }
            } else {
                Ok(err_result("连接已关闭"))
            }
        } else {
            Ok(err_result("tcp_write 需要 (i64, Vec<i64>) 参数"))
        }
    });
    vm.add_native("tcp_close".into(), |vm, args| {
        if let Some(Value::Int(handle, _)) = args.first() {
            let idx = *handle as usize;
            if idx > 0 && idx <= vm.tcp_streams.len() {
                vm.tcp_streams[idx - 1] = None; // drop 自动关闭
            }
        }
        Ok(Value::Unit)
    });
    vm.add_native("tcp_set_timeout".into(), |vm, args| {
        if args.len() >= 2 {
            if let (Value::Int(handle, _), Value::Int(ms, _)) = (&args[0], &args[1]) {
                let idx = *handle as usize;
                if idx > 0 && idx <= vm.tcp_streams.len() {
                    if let Some(ref mut stream) = vm.tcp_streams[idx - 1] {
                        let dur = std::time::Duration::from_millis(*ms as u64);
                        stream.set_read_timeout(Some(dur)).ok();
                        stream.set_write_timeout(Some(dur)).ok();
                    }
                }
            }
        }
        Ok(Value::Unit)
    });

    // —— TCP 服务端原语（句柄表方案，handle 1-based，0 表示无效）——
    // 与 std/net.th 的 listen/accept/listener_close wrapper 对齐。
    // 与 interpreter::natives::call_named_fn 中的实现语义对齐（双侧注册）。
    vm.add_native("tcp_listen".into(), |vm, args| {
        if let Some(Value::String(addr)) = args.first() {
            match std::net::TcpListener::bind(addr) {
                Ok(listener) => {
                    vm.tcp_listeners.push(Some(listener));
                    let handle = vm.tcp_listeners.len() as i64; // 1-based
                    Ok(ok_result(Value::Int(handle, BaseType::I32)))
                }
                Err(e) => Ok(err_result(format!("监听失败: {e}"))),
            }
        } else {
            Ok(err_result("tcp_listen 需要 1 个 String 参数"))
        }
    });
    vm.add_native("tcp_accept".into(), |vm, args| {
        if let Some(Value::Int(handle, _)) = args.first() {
            let idx = *handle as usize;
            if idx == 0 || idx > vm.tcp_listeners.len() {
                return Ok(err_result("无效的监听器句柄"));
            }
            if let Some(ref listener) = vm.tcp_listeners[idx - 1] {
                match listener.accept() {
                    Ok((stream, _)) => {
                        vm.tcp_streams.push(Some(stream));
                        let stream_handle = vm.tcp_streams.len() as i64; // 1-based
                        Ok(ok_result(Value::Int(stream_handle, BaseType::I32)))
                    }
                    Err(e) => Ok(err_result(format!("接受连接失败: {e}"))),
                }
            } else {
                Ok(err_result("监听器已关闭"))
            }
        } else {
            Ok(err_result("tcp_accept 需要 1 个 i64 参数"))
        }
    });
    vm.add_native("tcp_listener_close".into(), |vm, args| {
        if let Some(Value::Int(handle, _)) = args.first() {
            let idx = *handle as usize;
            if idx > 0 && idx <= vm.tcp_listeners.len() {
                vm.tcp_listeners[idx - 1] = None; // drop 自动关闭
            }
        }
        Ok(Value::Unit)
    });

    // —— UDP 网络原语（基本功核查第 69 项；句柄表方案，handle 1-based，0 表示无效）——
    // 与 std/net.th 的 udp_bind/udp_recv_from/udp_send_to/udp_close/udp_set_timeout wrapper 对齐。
    // 与 interpreter::natives::call_named_fn 中的实现语义对齐（双侧注册）。
    // UDP 无连接：bind 后用 send_to/recv_from 携带对端地址；handle 表与 TCP 独立避免类型混淆。
    vm.add_native("udp_bind".into(), |vm, args| {
        if let Some(Value::String(addr)) = args.first() {
            match std::net::UdpSocket::bind(addr) {
                Ok(sock) => {
                    vm.udp_sockets.push(Some(sock));
                    let handle = vm.udp_sockets.len() as i64; // 1-based
                    Ok(ok_result(Value::Int(handle, BaseType::I32)))
                }
                Err(e) => Ok(err_result(format!("绑定失败: {e}"))),
            }
        } else {
            Ok(err_result("udp_bind 需要 1 个 String 参数"))
        }
    });
    vm.add_native("udp_recv_from".into(), |vm, args| {
        if args.len() < 2 {
            return Ok(err_result("udp_recv_from 需要 (i64, i64) 参数"));
        }
        if let (Value::Int(handle, _), Value::Int(n, _)) = (&args[0], &args[1]) {
            let idx = *handle as usize;
            if idx == 0 || idx > vm.udp_sockets.len() {
                return Ok(err_result("无效的句柄"));
            }
            let max = (*n).max(0).min(65536) as usize;
            if let Some(ref mut sock) = vm.udp_sockets[idx - 1] {
                let mut buf = vec![0u8; max];
                match sock.recv_from(&mut buf) {
                    Ok((read_n, peer)) => {
                        let bytes: Vec<Value> = buf[..read_n]
                            .iter()
                            .map(|b| Value::Int(*b as i64, BaseType::I32))
                            .collect();
                        let peer_str = peer.to_string();
                        // 返回 Tuple(Vec<i64>, String)：字节数组 + 来源地址 "ip:port"
                        Ok(ok_result(Value::Tuple(vec![
                            Value::Vec(Rc::new(RefCell::new(bytes))),
                            Value::String(peer_str),
                        ])))
                    }
                    Err(e) => Ok(err_result(format!("接收失败: {e}"))),
                }
            } else {
                Ok(err_result("socket 已关闭"))
            }
        } else {
            Ok(err_result("udp_recv_from 需要 (i64, i64) 参数"))
        }
    });
    vm.add_native("udp_send_to".into(), |vm, args| {
        if args.len() < 3 {
            return Ok(err_result("udp_send_to 需要 (i64, Vec<i64>, String) 参数"));
        }
        if let (Value::Int(handle, _), Value::Vec(data), Value::String(addr)) =
            (&args[0], &args[1], &args[2])
        {
            let idx = *handle as usize;
            if idx == 0 || idx > vm.udp_sockets.len() {
                return Ok(err_result("无效的句柄"));
            }
            let bytes: Vec<u8> = data
                .borrow()
                .iter()
                .map(|x| match x {
                    Value::Int(b, _) => *b as u8,
                    _ => 0,
                })
                .collect();
            if let Some(ref mut sock) = vm.udp_sockets[idx - 1] {
                match sock.send_to(&bytes, addr) {
                    Ok(n) => Ok(ok_result(Value::Int(n as i64, BaseType::I32))),
                    Err(e) => Ok(err_result(format!("发送失败: {e}"))),
                }
            } else {
                Ok(err_result("socket 已关闭"))
            }
        } else {
            Ok(err_result("udp_send_to 需要 (i64, Vec<i64>, String) 参数"))
        }
    });
    vm.add_native("udp_close".into(), |vm, args| {
        if let Some(Value::Int(handle, _)) = args.first() {
            let idx = *handle as usize;
            if idx > 0 && idx <= vm.udp_sockets.len() {
                vm.udp_sockets[idx - 1] = None; // drop 自动关闭
            }
        }
        Ok(Value::Unit)
    });
    vm.add_native("udp_set_timeout".into(), |vm, args| {
        if args.len() >= 2 {
            if let (Value::Int(handle, _), Value::Int(ms, _)) = (&args[0], &args[1]) {
                let idx = *handle as usize;
                if idx > 0 && idx <= vm.udp_sockets.len() {
                    if let Some(ref mut sock) = vm.udp_sockets[idx - 1] {
                        let dur = std::time::Duration::from_millis(*ms as u64);
                        sock.set_read_timeout(Some(dur)).ok();
                        sock.set_write_timeout(Some(dur)).ok();
                    }
                }
            }
        }
        Ok(Value::Unit)
    });

    // —— 子进程原语（句柄表方案，handle 1-based，0 表示无效）——
    // 与 std/process.th 的 new/arg/run/output wrapper 对齐。
    // command_output 消费 Command（mem::take 取出所有权），再次调用返回 Err。
    vm.add_native("command_new".into(), |vm, args| {
        if let Some(Value::String(program)) = args.first() {
            let cmd = std::process::Command::new(program);
            vm.commands.push(Some(cmd));
            let handle = vm.commands.len() as i64; // 1-based
            Ok(ok_result(Value::Int(handle, BaseType::I32)))
        } else {
            Ok(err_result("command_new 需要 1 个 String 参数"))
        }
    });
    vm.add_native("command_arg".into(), |vm, args| {
        if args.len() >= 2 {
            if let (Value::Int(handle, _), Value::String(arg)) = (&args[0], &args[1]) {
                let idx = *handle as usize;
                if idx > 0 && idx <= vm.commands.len() {
                    if let Some(ref mut cmd) = vm.commands[idx - 1] {
                        cmd.arg(arg);
                    }
                }
            }
        }
        Ok(Value::Unit)
    });
    vm.add_native("command_run".into(), |vm, args| {
        if let Some(Value::Int(handle, _)) = args.first() {
            let idx = *handle as usize;
            if idx == 0 || idx > vm.commands.len() {
                return Ok(err_result("无效的命令句柄"));
            }
            if let Some(ref mut cmd) = vm.commands[idx - 1] {
                match cmd.status() {
                    Ok(status) => {
                        let code = status.code().unwrap_or(-1) as i64;
                        Ok(ok_result(Value::Int(code, BaseType::I32)))
                    }
                    Err(e) => Ok(err_result(format!("执行失败: {e}"))),
                }
            } else {
                Ok(err_result("命令已释放"))
            }
        } else {
            Ok(err_result("command_run 需要 1 个 i64 参数"))
        }
    });
    vm.add_native("command_output".into(), |vm, args| {
        if let Some(Value::Int(handle, _)) = args.first() {
            let idx = *handle as usize;
            if idx == 0 || idx > vm.commands.len() {
                return Ok(err_result("无效的命令句柄"));
            }
            // output() 消费 Command 语义：用 mem::take 取出所有权，槽位变 None
            let cmd_opt = std::mem::take(&mut vm.commands[idx - 1]);
            if let Some(mut cmd) = cmd_opt {
                match cmd.output() {
                    Ok(output) => {
                        let stdout = String::from_utf8_lossy(&output.stdout).to_string();
                        Ok(ok_result(Value::String(stdout)))
                    }
                    Err(e) => Ok(err_result(format!("执行失败: {e}"))),
                }
            } else {
                Ok(err_result("命令已释放"))
            }
        } else {
            Ok(err_result("command_output 需要 1 个 i64 参数"))
        }
    });

    // —— 正则表达式原语（句柄表方案，handle 1-based，0 表示无效）——
    // 与 std/regex.th 对齐：Tenth 层不暴露 Regex 类型，仅用 i64 handle。
    // 与 interpreter::natives::call_named_fn 中的实现语义对齐（双侧注册）。
    vm.add_native("regex_compile".into(), |vm, args| {
        if let Some(Value::String(pattern)) = args.first() {
            match regex::Regex::new(pattern) {
                Ok(re) => {
                    vm.regexes.push(Some(re));
                    let handle = vm.regexes.len() as i64; // 1-based
                    Ok(ok_result(Value::Int(handle, BaseType::I32)))
                }
                Err(e) => Ok(err_result(format!("正则编译失败: {e}"))),
            }
        } else {
            Ok(err_result("regex_compile 需要 1 个 String 参数"))
        }
    });
    vm.add_native("regex_match".into(), |vm, args| {
        if args.len() < 2 {
            return Ok(Value::Bool(false));
        }
        if let (Value::Int(handle, _), Value::String(input)) = (&args[0], &args[1]) {
            let idx = *handle as usize;
            if idx == 0 || idx > vm.regexes.len() {
                return Ok(Value::Bool(false));
            }
            if let Some(ref re) = vm.regexes[idx - 1] {
                return Ok(Value::Bool(re.is_match(input)));
            }
            Ok(Value::Bool(false))
        } else {
            Ok(Value::Bool(false))
        }
    });
    vm.add_native("regex_find".into(), |vm, args| {
        // 与 std/regex.th 契约对齐：返回 String，无匹配返回空字符串 ""
        if args.len() < 2 {
            return Ok(Value::String(String::new()));
        }
        if let (Value::Int(handle, _), Value::String(input)) = (&args[0], &args[1]) {
            let idx = *handle as usize;
            if idx == 0 || idx > vm.regexes.len() {
                return Ok(Value::String(String::new()));
            }
            if let Some(ref re) = vm.regexes[idx - 1] {
                if let Some(m) = re.find(input) {
                    return Ok(Value::String(m.as_str().to_string()));
                }
            }
            Ok(Value::String(String::new()))
        } else {
            Ok(Value::String(String::new()))
        }
    });
    vm.add_native("regex_find_all".into(), |vm, args| {
        // 与 std/regex.th 契约对齐：返回 Vec<String>
        if args.len() < 2 {
            return Ok(Value::Vec(Rc::new(RefCell::new(Vec::new()))));
        }
        if let (Value::Int(handle, _), Value::String(input)) = (&args[0], &args[1]) {
            let idx = *handle as usize;
            if idx == 0 || idx > vm.regexes.len() {
                return Ok(Value::Vec(Rc::new(RefCell::new(Vec::new()))));
            }
            if let Some(ref re) = vm.regexes[idx - 1] {
                let collected: Vec<Value> = re
                    .find_iter(input)
                    .map(|m| Value::String(m.as_str().to_string()))
                    .collect();
                return Ok(Value::Vec(Rc::new(RefCell::new(collected))));
            }
            Ok(Value::Vec(Rc::new(RefCell::new(Vec::new()))))
        } else {
            Ok(Value::Vec(Rc::new(RefCell::new(Vec::new()))))
        }
    });
    vm.add_native("regex_replace".into(), |vm, args| {
        if args.len() < 3 {
            return Ok(Value::String(String::new()));
        }
        if let (Value::Int(handle, _), Value::String(input), Value::String(replacement)) =
            (&args[0], &args[1], &args[2])
        {
            let idx = *handle as usize;
            if idx == 0 || idx > vm.regexes.len() {
                return Ok(Value::String(input.clone()));
            }
            if let Some(ref re) = vm.regexes[idx - 1] {
                let result = re.replace_all(input, replacement.as_str()).into_owned();
                return Ok(Value::String(result));
            }
            Ok(Value::String(input.clone()))
        } else {
            Ok(Value::String(String::new()))
        }
    });
    vm.add_native("regex_split".into(), |vm, args| {
        // 与 std/regex.th 契约对齐：返回 Vec<String>
        if args.len() < 2 {
            return Ok(Value::Vec(Rc::new(RefCell::new(Vec::new()))));
        }
        if let (Value::Int(handle, _), Value::String(input)) = (&args[0], &args[1]) {
            let idx = *handle as usize;
            if idx == 0 || idx > vm.regexes.len() {
                return Ok(Value::Vec(Rc::new(RefCell::new(Vec::new()))));
            }
            if let Some(ref re) = vm.regexes[idx - 1] {
                let collected: Vec<Value> = re
                    .split(input)
                    .map(|s| Value::String(s.to_string()))
                    .collect();
                return Ok(Value::Vec(Rc::new(RefCell::new(collected))));
            }
            Ok(Value::Vec(Rc::new(RefCell::new(Vec::new()))))
        } else {
            Ok(Value::Vec(Rc::new(RefCell::new(Vec::new()))))
        }
    });

    // —— HTTP 客户端原语（手写 HTTP/1.1，10 秒默认超时）——
    vm.add_native("http_get".into(), |_vm, args| {
        if let Some(Value::String(url)) = args.first() {
            match http_get_impl(url) {
                Ok(body) => Ok(ok_result(Value::String(body))),
                Err(e) => Ok(err_result(e)),
            }
        } else {
            Ok(err_result("http_get 需要 1 个 String 参数"))
        }
    });
    vm.add_native("http_post".into(), |_vm, args| {
        if args.len() >= 2 {
            if let (Value::String(url), Value::String(body)) = (&args[0], &args[1]) {
                match http_post_impl(url, body) {
                    Ok(resp) => Ok(ok_result(Value::String(resp))),
                    Err(e) => Ok(err_result(e)),
                }
            } else {
                Ok(err_result("http_post 需要 (String, String) 参数"))
            }
        } else {
            Ok(err_result("http_post 需要 (String, String) 参数"))
        }
    });

    vm.add_native("read_file".into(), |vm, args| {
        if let Some(Value::String(path)) = args.first() {
            // H-2: 沙箱校验
            let resolved = if let Some(ref sb) = vm.fs_sandbox {
                match sb.check_read(path) {
                    Ok(p) => p,
                    Err(e) => return Err(TenthError::RuntimeError { line: None, col: None, message: e }),
                }
            } else {
                std::path::PathBuf::from(path)
            };
            match std::fs::read_to_string(&resolved) {
                Ok(s) => Ok(Value::String(s)),
                Err(e) => Err(TenthError::RuntimeError { line: None, col: None, message: format!("读取文件: {e}") }),
            }
        } else {
            Ok(Value::String(String::new()))
        }
    });
    vm.add_native("Vec::new".into(), |_vm, _args| {
        Ok(Value::Vec(Rc::new(RefCell::new(Vec::new()))))
    });
    vm.add_native("tensor".into(), |_vm, args| {
        // tensor() constructor: when called as tensor[[...]], the bytecode
        // compiler handles TensorLiteral via Op::MakeTensor directly.
        // This native handles the rare case where tensor() is called as a function.
        if args.len() == 1 {
            Ok(args[0].clone())
        } else {
            Err(TenthError::RuntimeError { line: None, col: None, message: "tensor() 参数异常".into() })
        }
    });
    vm.add_native("write_bytes".into(), |vm, args| {
        if args.len() >= 2 {
            if let Value::String(path) = &args[1] {
                if let Value::Vec(items) = &args[0] {
                    // H-2: 沙箱校验
                    let resolved = if let Some(ref sb) = vm.fs_sandbox {
                        match sb.check_write(path) {
                            Ok(p) => p,
                            Err(e) => return Err(TenthError::RuntimeError { line: None, col: None, message: e }),
                        }
                    } else {
                        std::path::PathBuf::from(path)
                    };
                    let bytes: Vec<u8> = items.borrow().iter().map(|v| v.as_int().unwrap_or(0) as u8).collect();
                    let _ = std::fs::write(&resolved, &bytes);
                    return Ok(Value::Int(0, BaseType::I32));
                }
            }
        }
        Ok(Value::Int(1, BaseType::I32))
    });
    vm.add_native("read_bytes".into(), |vm, args| {
        if let Some(Value::String(path)) = args.first() {
            // H-2: 沙箱校验
            let resolved = if let Some(ref sb) = vm.fs_sandbox {
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
                    Ok(Value::Vec(Rc::new(RefCell::new(bytes))))
                }
                Err(e) => Err(TenthError::RuntimeError { line: None, col: None,
                    message: format!("读取字节失败: {}", e),
                }),
            }
        } else {
            Err(TenthError::RuntimeError { line: None, col: None,
                message: "read_bytes(路径) 期望一个字符串路径".into(),
            })
        }
    });
    // Time functions
    vm.add_native("time_now".into(), |_vm, _args| {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs_f64();
        Ok(Value::Float(now))
    });
    vm.add_native("time_now_ms".into(), |_vm, _args| {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as f64;
        Ok(Value::Float(now))
    });
    vm.add_native("time_date".into(), |_vm, _args| {
        let secs = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        let days_since_epoch = secs / 86400;
        let (year, month, day) = datetime::days_to_date(days_since_epoch);
        Ok(Value::String(format!("{}-{:02}-{:02}", year, month, day)))
    });
    vm.add_native("time_time".into(), |_vm, _args| {
        let secs = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs() % 86400;
        let h = secs / 3600;
        let m = (secs % 3600) / 60;
        let s = secs % 60;
        Ok(Value::String(format!("{}:{:02}:{:02}", h, m, s)))
    });
    vm.add_native("time_datetime".into(), |_vm, _args| {
        let secs = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        let days_since_epoch = secs / 86400;
        let (year, month, day) = datetime::days_to_date(days_since_epoch);
        let day_secs = secs % 86400;
        let h = day_secs / 3600;
        let m = (day_secs % 3600) / 60;
        let s = day_secs % 60;
        Ok(Value::String(format!("{}-{:02}-{:02} {}:{:02}:{:02}", year, month, day, h, m, s)))
    });
    vm.add_native("time_sleep_ms".into(), |_vm, args| {
        if let Some(Value::Int(ms, _)) = args.first() {
            // 安全：拒绝负数（`as u64` 会符号扩展为巨大值，导致近乎永久的 DoS）
            // 上限 24 小时，防止 `.th` 程序意外将进程睡眠数年
            const MAX_SLEEP_MS: i64 = 24 * 60 * 60 * 1000;
            if *ms < 0 {
                return Err(TenthError::RuntimeError { line: None, col: None,
                    message: format!("time_sleep_ms: 不接受负数（{}）", ms),
                });
            }
            if *ms > MAX_SLEEP_MS {
                return Err(TenthError::RuntimeError { line: None, col: None,
                    message: format!("time_sleep_ms: 超过 24 小时上限（{}ms）", ms),
                });
            }
            std::thread::sleep(std::time::Duration::from_millis(*ms as u64));
            Ok(Value::Unit)
        } else {
            Err(TenthError::RuntimeError { line: None, col: None, message: "time_sleep_ms(ms) 期望一个整数".into() })
        }
    });

    // —— Date native（Wave 3 第 8 项：Date 类型，路径 B 复用 struct 机制）——
    // 算法：Howard Hinnant date_algorithms（与 datetime::days_to_date 同源）。
    // 不引入新的 HIR 类型——返回 i64 或 Tuple<i64,i64,i64>，标准库 date.th 用 struct 包装。
    vm.add_native("date_to_unix_days".into(), |_vm, args| {
        match (args.first(), args.get(1), args.get(2)) {
            (Some(Value::Int(y, _)), Some(Value::Int(m, _)), Some(Value::Int(d, _))) => {
                Ok(Value::Int(datetime::date_to_days(*y, *m, *d), BaseType::I64))
            }
            _ => Err(TenthError::RuntimeError { line: None, col: None,
                message: "date_to_unix_days(year, month, day) 期望三个整数".into(),
            }),
        }
    });
    vm.add_native("date_from_unix_days".into(), |_vm, args| {
        if let Some(Value::Int(days, _)) = args.first() {
            let (y, m, d) = datetime::days_to_date_i64(*days);
            // 返回 Tuple(year, month, day)；标准库 date.th 用 `let (y,m,d) = ...` 解构。
            Ok(Value::Tuple(vec![
                Value::Int(y, BaseType::I64),
                Value::Int(m, BaseType::I64),
                Value::Int(d, BaseType::I64),
            ]))
        } else {
            Err(TenthError::RuntimeError { line: None, col: None,
                message: "date_from_unix_days(days) 期望一个整数".into(),
            })
        }
    });
    vm.add_native("date_i64_add_days".into(), |_vm, args| {
        match (args.first(), args.get(1)) {
            (Some(Value::Int(days, _)), Some(Value::Int(delta, _))) => {
                // 直接 i64 加法：Unix days 是线性天数序列，无闰秒/时区问题。
                // 溢出由 i64 自然回绕（与 i64 加法语义一致）；调用方需自行约束范围。
                // 注：native 名带 _i64_ 前缀以避免与 std/date.th 中的 helper
                // `date_add_days(Date, i64) -> Date` 同名冲突（native 优先匹配会
                // 阻止 user function 被调用）。
                Ok(Value::Int(days.wrapping_add(*delta), BaseType::I64))
            }
            _ => Err(TenthError::RuntimeError { line: None, col: None,
                message: "date_i64_add_days(days, delta) 期望两个整数".into(),
            }),
        }
    });
    vm.add_native("date_diff_days".into(), |_vm, args| {
        match (args.first(), args.get(1)) {
            (Some(Value::Int(d1, _)), Some(Value::Int(d2, _))) => {
                // days1 - days2：正数表示 d1 在 d2 之后。
                Ok(Value::Int(d1.wrapping_sub(*d2), BaseType::I64))
            }
            _ => Err(TenthError::RuntimeError { line: None, col: None,
                message: "date_diff_days(days1, days2) 期望两个整数".into(),
            }),
        }
    });
    vm.add_native("date_day_of_week".into(), |_vm, args| {
        if let Some(Value::Int(days, _)) = args.first() {
            // Unix epoch 1970-01-01 是周四（=4）。(days + 4) % 7 给出 0=周日..6=周六。
            // 用 ((days + 4) % 7 + 7) % 7 处理负数（1970 年前日期）。
            let w = ((*days + 4) % 7 + 7) % 7;
            Ok(Value::Int(w, BaseType::I64))
        } else {
            Err(TenthError::RuntimeError { line: None, col: None,
                message: "date_day_of_week(days) 期望一个整数".into(),
            })
        }
    });

    // —— Phase 2 Step 5：异步 I/O native ——
    // 设计：std::thread + mpsc + thread_local。native 创建 Pending Future，
    // 注册到 ASYNC_IO，VM 调度器在 run_scheduler 循环中 poll。
    // await Pending Future 时 Op::Await 把当前 task 加入 waiters 并挂起，
    // I/O 就绪后 poll 把 Future 设为 Ready 并唤醒 waiters。
    vm.add_native("async_sleep_ms".into(), |_vm, args| {
        let ms = match args.first() {
            Some(Value::Int(n, _)) => *n,
            _ => return Err(TenthError::RuntimeError { line: None, col: None,
                message: "async_sleep_ms(ms) 期望一个整数".into(),
            }),
        };
        if ms < 0 {
            return Err(TenthError::RuntimeError { line: None, col: None,
                message: format!("async_sleep_ms: 不接受负数（{}）", ms),
            });
        }
        const MAX_SLEEP_MS: i64 = 24 * 60 * 60 * 1000;
        if ms > MAX_SLEEP_MS {
            return Err(TenthError::RuntimeError { line: None, col: None,
                message: format!("async_sleep_ms: 超过 24 小时上限（{}ms）", ms),
            });
        }
        // 创建 Pending Future，注册定时器
        let future = Value::future_pending();
        if let Value::Future(rc) = &future {
            let rc_clone = rc.clone();
            ASYNC_IO.with(|io| io.borrow_mut().add_timer(ms as u64, rc_clone));
        }
        Ok(future)
    });

    vm.add_native("async_tcp_read".into(), |vm, args| {
        // 参数校验：参数错误时返回 Ready Future 包含 Result::Err
        if args.len() < 2 {
            return Ok(Value::future_ready(err_result("async_tcp_read 需要 (i64, i64) 参数")));
        }
        let (handle, max_bytes) = match (&args[0], &args[1]) {
            (Value::Int(h, _), Value::Int(n, _)) => (*h, *n),
            _ => return Ok(Value::future_ready(err_result("async_tcp_read 需要 (i64, i64) 参数"))),
        };
        let idx = handle as usize;
        if idx == 0 || idx > vm.tcp_streams.len() {
            return Ok(Value::future_ready(err_result("无效的句柄")));
        }
        // try_clone 让 worker 线程持有 stream 副本，原 stream 留在 VM 中
        let stream_clone = match vm.tcp_streams[idx - 1].as_ref() {
            Some(s) => match s.try_clone() {
                Ok(c) => c,
                Err(e) => return Ok(Value::future_ready(err_result(format!("句柄克隆失败: {e}")))),
            },
            None => return Ok(Value::future_ready(err_result("连接已关闭"))),
        };
        // 重置 stream_clone 的 read_timeout 为 None（阻塞读）
        // （原 stream 可能被 tcp_set_timeout 设了短超时，对 worker 不利）
        stream_clone.set_read_timeout(None).ok();
        let max = max_bytes.max(0).min(65536) as usize;

        let future = Value::future_pending();
        let future_rc = match &future {
            Value::Future(rc) => rc.clone(),
            _ => unreachable!(),
        };
        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            use std::io::Read;
            let mut buf = vec![0u8; max];
            let mut s = stream_clone;
            let result = match s.read(&mut buf) {
                Ok(0) => IoResult::Bytes(Vec::new()), // EOF
                Ok(n) => IoResult::Bytes(buf[..n].to_vec()),
                Err(e) => IoResult::Err(format!("读取失败: {e}")),
            };
            let _ = tx.send(result);
        });
        ASYNC_IO.with(|io| io.borrow_mut().add_io(rx, future_rc));
        Ok(future)
    });

    vm.add_native("async_tcp_write".into(), |vm, args| {
        if args.len() < 2 {
            return Ok(Value::future_ready(err_result("async_tcp_write 需要 (i64, Vec<i64>) 参数")));
        }
        let (handle, data) = match (&args[0], &args[1]) {
            (Value::Int(h, _), Value::Vec(v)) => (*h, v.clone()),
            _ => return Ok(Value::future_ready(err_result("async_tcp_write 需要 (i64, Vec<i64>) 参数"))),
        };
        let idx = handle as usize;
        if idx == 0 || idx > vm.tcp_streams.len() {
            return Ok(Value::future_ready(err_result("无效的句柄")));
        }
        let stream_clone = match vm.tcp_streams[idx - 1].as_ref() {
            Some(s) => match s.try_clone() {
                Ok(c) => c,
                Err(e) => return Ok(Value::future_ready(err_result(format!("句柄克隆失败: {e}")))),
            },
            None => return Ok(Value::future_ready(err_result("连接已关闭"))),
        };
        stream_clone.set_write_timeout(None).ok();
        let bytes: Vec<u8> = data.borrow().iter().map(|x| match x {
            Value::Int(b, _) => *b as u8,
            _ => 0,
        }).collect();

        let future = Value::future_pending();
        let future_rc = match &future {
            Value::Future(rc) => rc.clone(),
            _ => unreachable!(),
        };
        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            use std::io::Write;
            let mut s = stream_clone;
            let result = match s.write_all(&bytes) {
                Ok(_) => IoResult::Count(bytes.len()),
                Err(e) => IoResult::Err(format!("写入失败: {e}")),
            };
            let _ = tx.send(result);
        });
        ASYNC_IO.with(|io| io.borrow_mut().add_io(rx, future_rc));
        Ok(future)
    });
    // Random functions — 使用 rand crate 的 CSPRNG（thread_rng），避免可预测种子。
    // 历史 `DefaultHasher` + SystemTime 方案可被攻击者枚举纳秒时刻预测输出。
    vm.add_native("random_int".into(), |_vm, args| {
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
        Ok(Value::Int(low + (r % range) as i64, BaseType::I32))
    });
    vm.add_native("random_float".into(), |_vm, _args| {
        use rand::Rng;
        // [0, 1) 半开区间，标准做法
        let r: f64 = rand::thread_rng().r#gen();
        Ok(Value::Float(r))
    });
    // Math functions（输入为 Float32 时返回 Float32，否则 Float）
    vm.add_native("math_tan".into(), |_vm, args| {
        match args.first() {
            Some(Value::Float(x)) => Ok(Value::Float(x.tan())),
            Some(Value::Float32(x)) => Ok(Value::Float32(x.tan())),
            _ => Ok(Value::Float(0.0))
        }
    });
    vm.add_native("math_asin".into(), |_vm, args| {
        match args.first() {
            Some(Value::Float(x)) => Ok(Value::Float(x.asin())),
            Some(Value::Float32(x)) => Ok(Value::Float32(x.asin())),
            _ => Ok(Value::Float(0.0))
        }
    });
    vm.add_native("math_acos".into(), |_vm, args| {
        match args.first() {
            Some(Value::Float(x)) => Ok(Value::Float(x.acos())),
            Some(Value::Float32(x)) => Ok(Value::Float32(x.acos())),
            _ => Ok(Value::Float(0.0))
        }
    });
    vm.add_native("math_atan".into(), |_vm, args| {
        match args.first() {
            Some(Value::Float(x)) => Ok(Value::Float(x.atan())),
            Some(Value::Float32(x)) => Ok(Value::Float32(x.atan())),
            _ => Ok(Value::Float(0.0))
        }
    });
    vm.add_native("math_atan2".into(), |_vm, args| {
        match (args.first(), args.get(1)) {
            (Some(Value::Float(y)), Some(Value::Float(x))) => Ok(Value::Float(y.atan2(*x))),
            (Some(Value::Float32(y)), Some(Value::Float32(x))) => Ok(Value::Float32(y.atan2(*x))),
            _ => Ok(Value::Float(0.0))
        }
    });
    vm.add_native("math_sinh".into(), |_vm, args| {
        match args.first() {
            Some(Value::Float(x)) => Ok(Value::Float(x.sinh())),
            Some(Value::Float32(x)) => Ok(Value::Float32(x.sinh())),
            _ => Ok(Value::Float(0.0))
        }
    });
    vm.add_native("math_cosh".into(), |_vm, args| {
        match args.first() {
            Some(Value::Float(x)) => Ok(Value::Float(x.cosh())),
            Some(Value::Float32(x)) => Ok(Value::Float32(x.cosh())),
            _ => Ok(Value::Float(0.0))
        }
    });
    vm.add_native("math_tanh".into(), |_vm, args| {
        match args.first() {
            Some(Value::Float(x)) => Ok(Value::Float(x.tanh())),
            Some(Value::Float32(x)) => Ok(Value::Float32(x.tanh())),
            _ => Ok(Value::Float(0.0))
        }
    });
    vm.add_native("math_log10".into(), |_vm, args| {
        match args.first() {
            Some(Value::Float(x)) => Ok(Value::Float(x.log10())),
            Some(Value::Float32(x)) => Ok(Value::Float32(x.log10())),
            _ => Ok(Value::Float(0.0))
        }
    });
    vm.add_native("math_log2".into(), |_vm, args| {
        match args.first() {
            Some(Value::Float(x)) => Ok(Value::Float(x.log2())),
            Some(Value::Float32(x)) => Ok(Value::Float32(x.log2())),
            _ => Ok(Value::Float(0.0))
        }
    });
    vm.add_native("math_exp".into(), |_vm, args| {
        match args.first() {
            Some(Value::Float(x)) => Ok(Value::Float(x.exp())),
            Some(Value::Float32(x)) => Ok(Value::Float32(x.exp())),
            _ => Ok(Value::Float(0.0))
        }
    });
    vm.add_native("math_pow".into(), |_vm, args| {
        match (args.first(), args.get(1)) {
            (Some(Value::Float(base)), Some(Value::Float(exp))) => Ok(Value::Float(base.powf(*exp))),
            (Some(Value::Float32(base)), Some(Value::Float32(exp))) => Ok(Value::Float32(base.powf(*exp))),
            _ => Ok(Value::Float(0.0))
        }
    });
    vm.add_native("math_floor".into(), |_vm, args| {
        match args.first() {
            Some(Value::Float(x)) => Ok(Value::Float(x.floor())),
            Some(Value::Float32(x)) => Ok(Value::Float32(x.floor())),
            _ => Ok(Value::Float(0.0))
        }
    });
    vm.add_native("math_ceil".into(), |_vm, args| {
        match args.first() {
            Some(Value::Float(x)) => Ok(Value::Float(x.ceil())),
            Some(Value::Float32(x)) => Ok(Value::Float32(x.ceil())),
            _ => Ok(Value::Float(0.0))
        }
    });
    vm.add_native("math_round".into(), |_vm, args| {
        match args.first() {
            Some(Value::Float(x)) => Ok(Value::Float(x.round())),
            Some(Value::Float32(x)) => Ok(Value::Float32(x.round())),
            _ => Ok(Value::Float(0.0))
        }
    });
    // CLI functions
    vm.add_native("cli_args_count".into(), |_vm, _args| {
        Ok(Value::Int(1, BaseType::I32))
    });
    vm.add_native("cli_arg".into(), |_vm, _args| {
        Ok(Value::String(String::new()))
    });
    // JSON functions
    vm.add_native("json_encode".into(), |_vm, args| {
        if let Some(val) = args.first() {
            Ok(Value::String(json::json_encode_value(val)))
        } else {
            Ok(Value::String("null".into()))
        }
    });
    vm.add_native("json_encode_pretty".into(), |_vm, args| {
        if let Some(val) = args.first() {
            Ok(Value::String(json::json_encode_value_pretty(val, 0)))
        } else {
            Ok(Value::String("null".into()))
        }
    });
    vm.add_native("json_decode".into(), |_vm, args| {
        if let Some(Value::String(s)) = args.first() {
            Ok(json::json_decode_string(s))
        } else {
            Ok(Value::Unit)
        }
    });
    vm.add_native("compile_host".into(), |vm, args| {
        if args.len() >= 2 {
            if let (Value::String(src), Value::String(out)) = (&args[0], &args[1]) {
                // H-2/L-7: 沙箱校验写路径
                let out_resolved = if let Some(ref sb) = vm.fs_sandbox {
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
                    Ok(bytes) => { let _ = std::fs::write(&out_resolved, &bytes); return Ok(Value::Int(0, BaseType::I32)); }
                    Err(_) => return Ok(Value::Int(1, BaseType::I32)),
                }
            }
        }
        Ok(Value::Int(1, BaseType::I32))
    });
    vm.add_native("compile_program".into(), |vm, args| {
        if args.len() >= 2 {
            if let Value::String(out) = &args[1] {
                // H-2/L-7: 沙箱校验写路径
                let out_resolved = if let Some(ref sb) = vm.fs_sandbox {
                    match sb.check_write(out) {
                        Ok(p) => p,
                        Err(e) => return Err(TenthError::RuntimeError { line: None, col: None, message: e }),
                    }
                } else {
                    std::path::PathBuf::from(out)
                };
                match crate::compile::compile_program_to_wasm(&args[0]) {
                    Ok(bytes) => { let _ = std::fs::write(&out_resolved, &bytes); return Ok(Value::Int(0, BaseType::I32)); }
                    Err(_) => return Ok(Value::Int(1, BaseType::I32)),
                }
            }
        }
        Ok(Value::Int(1, BaseType::I32))
    });

    // ── Autodiff native functions ──
    vm.add_native("new_grad".into(), |vm, _args| {
        vm.tape = Some(Tape::new());
        vm.recording = true;
        Ok(Value::Unit)
    });
    vm.add_native("param".into(), |vm, args| {
        if let Some(Value::Tensor(t)) = args.first() {
            if let Some(ref mut tape) = vm.tape {
                let node_id = tape.input(t.clone());
                t.borrow_mut().tape_id = Some(node_id);
            }
            Ok(Value::Tensor(t.clone()))
        } else {
            Err(TenthError::RuntimeError { line: None, col: None, message: "param() 需要一个张量参数".into() })
        }
    });
    vm.add_native("backward".into(), |vm, args| {
        if let Some(Value::Tensor(t)) = args.first() {
            // PROJ-006：先把 custom_ops Rc 副本交给 tape，使 Custom 节点的
            // backward 能通过 registry 查到用户实现的 CustomBackward。
            let custom_ops = vm.custom_ops.clone();
            if let Some(tape) = &mut vm.tape {
                tape.set_custom_ops(custom_ops);
                let loss_id = t.borrow().tape_id
                    .ok_or_else(|| TenthError::RuntimeError { line: None, col: None, message: "backward(): 张量没有 tape_id".into() })?;
                // 护城河 F：包裹 backward 错误，附加 formal_explain 根因分析
                // Phase 1：从 backward 抛出的 ShapeMismatch 错误中提取真实 v_err/expected/actual，
                // 传给 formal_explain 提升根因分析精度（替代 Phase 0 的占位值 loss_id/&[]/&[]）。
                match tape.backward(loss_id) {
                    Ok(()) => Ok(Value::Unit),
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
                        // 计算 formal_explain 根因候选（传入真实 v_err/expected/actual/error_msg）
                        // 护城河 F Phase 2：error_msg 用于 5 类错误分类
                        let causes = tape.formal_explain(v_err, expected, actual, error_msg);
                        let explanations: Vec<String> = causes.iter().map(|c| c.explanation.clone()).collect();
                        // 存到 vm.last_explanation，供 explain_error() native 读取
                        vm.last_explanation = explanations.clone();
                        // 构造最终错误：若 backward 已返回 ShapeMismatch，复用其 context（保留真实 v_err/expected/actual）；
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
                        Err(TenthError::ShapeMismatch {
                            context,
                            message: root_cause_msg,
                        })
                    }
                }
            } else {
                Err(TenthError::RuntimeError { line: None, col: None, message: "未调用 new_grad()".into() })
            }
        } else {
            Err(TenthError::RuntimeError { line: None, col: None, message: "backward() 需要一个张量参数".into() })
        }
    });
    // 护城河 F：explain_error() — 返回上一次 backward 失败的根因说明列表
    // 用户在 try-catch backward 错误后调用此 native 获取详细分析。
    vm.add_native("explain_error".into(), |vm, _args| {
        let explanations = std::mem::take(&mut vm.last_explanation);
        let values: Vec<Value> = explanations.into_iter().map(Value::String).collect();
        Ok(Value::Vec(Rc::new(RefCell::new(values))))
    });
    vm.add_native("grad".into(), |_vm, args| {
        if let Some(Value::Tensor(t)) = args.first() {
            let p = t.borrow();
            if let Some(ref grad) = p.grad {
                let grad_tensor = Tensor::from_tensor_data(grad.clone());
                Ok(Value::Tensor(Rc::new(RefCell::new(grad_tensor))))
            } else {
                // 按参数 dtype 返回零张量
                let zeros = if p.is_f32() {
                    Tensor::zeros_f32(&p.shape())
                } else {
                    Tensor::zeros(&p.shape())
                };
                Ok(Value::Tensor(Rc::new(RefCell::new(zeros))))
            }
        } else {
            Err(TenthError::RuntimeError { line: None, col: None, message: "grad() 需要一个张量参数".into() })
        }
    });
    vm.add_native("stop_grad".into(), |vm, args| {
        if let Some(Value::Tensor(t)) = args.first() {
            let mut detached = t.borrow().clone();
            detached.tape_id = None;
            Ok(Value::Tensor(Rc::new(RefCell::new(detached))))
        } else {
            // No-arg form: stop gradient recording
            vm.recording = false;
            Ok(Value::Unit)
        }
    });
    vm.add_native("zero_grad".into(), |vm, _args| {
        if let Some(ref tape) = vm.tape {
            tape.zero_grad();
        }
        Ok(Value::Unit)
    });
    vm.add_native("select".into(), |vm, args| {
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
        if vm.recording {
            if let Some(ref mut tape) = vm.tape {
                let then_id = then.borrow().tape_id;
                let else_id = else_.borrow().tape_id;
                let node_id = tape.select(then_id, else_id, cond.clone(), then.clone(), else_.clone(), result.clone());
                result.borrow_mut().tape_id = Some(node_id);
            }
        }
        Ok(Value::Tensor(result))
    });
    vm.add_native("scatter".into(), |vm, args| {
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
        if vm.recording {
            if let Some(ref mut tape) = vm.tape {
                let base_id = base.borrow().tape_id;
                let src_id = src.borrow().tape_id;
                let node_id = tape.scatter(base_id, src_id, base.clone(), src.clone(), index.clone(), result.clone(), dim);
                result.borrow_mut().tape_id = Some(node_id);
            }
        }
        Ok(Value::Tensor(result))
    });
    vm.add_native("gather".into(), |vm, args| {
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
        if vm.recording {
            if let Some(ref mut tape) = vm.tape {
                let base_id = base.borrow().tape_id;
                let node_id = tape.gather(base_id, base.clone(), index.clone(), result.clone(), dim);
                result.borrow_mut().tape_id = Some(node_id);
            }
        }
        Ok(Value::Tensor(result))
    });
    // ── PROJ-006：自定义可微算子调用 native ──────────────────────────────
    // __call_custom_op(op_id, ...inputs) — 查 registry 找到 CustomBackward 实现，
    // 调用其 forward 计算输出；若 recording 则记录 TapeOp::Custom(op_id) 到 tape。
    // op_id 由 Rust 端 `Vm::register_custom_op` 返回，标准库 wrapper 据此分发。
    // 兼容 vm.recording 与否：不录制时仅执行 forward，不写 tape 节点（与 select/scatter 模式一致）。
    vm.add_native("__call_custom_op".into(), |vm, args| {
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
            let registry = vm.custom_ops.borrow();
            let custom_op = match registry.get(op_id) {
                Some(op) => op,
                None => return Err(TenthError::RuntimeError { line: None, col: None,
                    message: format!("__call_custom_op: op_id={} 未注册", op_id) }),
            };
            // 借用所有输入张量
            let borrowed: Vec<std::cell::Ref<Tensor>> = input_tensors.iter().map(|t| t.borrow()).collect();
            let input_refs: Vec<&Tensor> = borrowed.iter().map(|r| &**r).collect();
            custom_op.forward(&input_refs).map_err(|e| TenthError::RuntimeError {
                line: None, col: None,
                message: format!("Custom#{} ({}) forward 失败：{}", op_id, custom_op.name(), e),
            })?
        };
        let result = Rc::new(RefCell::new(result_tensor));
        // 若 recording，记录到 tape（与 select/scatter/gather 一致的模式）
        if vm.recording {
            if let Some(tape) = &mut vm.tape {
                let input_ids: Vec<Option<usize>> = input_tensors.iter().map(|t| t.borrow().tape_id).collect();
                let node_id = tape.custom_op(op_id, input_ids, input_tensors, result.clone());
                result.borrow_mut().tape_id = Some(node_id);
            }
        }
        Ok(Value::Tensor(result))
    });
    // ── 张量比较运算（Wave 2 第 4 项）──────────────────────────────────
    // 6 个比较 native：gt/lt/ge/le/eq/ne。返回 F64 张量（0.0/1.0 编码 bool）。
    // 输入 dtype 任意（F32/F64/F16/BF16），先 cast f64 视图再广播比较。
    // 不可微：比较结果是布尔掩码，不进入 tape（与 select 耦合可微，见标准库 where_）。
    vm.add_native("tensor_gt".into(), |_vm, args| {
        if args.len() < 2 {
            return Err(TenthError::RuntimeError { line: None, col: None,
                message: "tensor_gt(a, b) 期望两个张量参数".into() });
        }
        let (a, b) = match (&args[0], &args[1]) {
            (Value::Tensor(x), Value::Tensor(y)) => (x.clone(), y.clone()),
            _ => return Err(TenthError::RuntimeError { line: None, col: None,
                message: "tensor_gt(a, b) 期望两个张量参数".into() }),
        };
        let r = a.borrow().gt(&b.borrow())
            .map_err(|m| TenthError::RuntimeError { line: None, col: None, message: m })?;
        Ok(Value::Tensor(Rc::new(RefCell::new(r))))
    });
    vm.add_native("tensor_lt".into(), |_vm, args| {
        if args.len() < 2 {
            return Err(TenthError::RuntimeError { line: None, col: None,
                message: "tensor_lt(a, b) 期望两个张量参数".into() });
        }
        let (a, b) = match (&args[0], &args[1]) {
            (Value::Tensor(x), Value::Tensor(y)) => (x.clone(), y.clone()),
            _ => return Err(TenthError::RuntimeError { line: None, col: None,
                message: "tensor_lt(a, b) 期望两个张量参数".into() }),
        };
        let r = a.borrow().lt(&b.borrow())
            .map_err(|m| TenthError::RuntimeError { line: None, col: None, message: m })?;
        Ok(Value::Tensor(Rc::new(RefCell::new(r))))
    });
    vm.add_native("tensor_ge".into(), |_vm, args| {
        if args.len() < 2 {
            return Err(TenthError::RuntimeError { line: None, col: None,
                message: "tensor_ge(a, b) 期望两个张量参数".into() });
        }
        let (a, b) = match (&args[0], &args[1]) {
            (Value::Tensor(x), Value::Tensor(y)) => (x.clone(), y.clone()),
            _ => return Err(TenthError::RuntimeError { line: None, col: None,
                message: "tensor_ge(a, b) 期望两个张量参数".into() }),
        };
        let r = a.borrow().ge(&b.borrow())
            .map_err(|m| TenthError::RuntimeError { line: None, col: None, message: m })?;
        Ok(Value::Tensor(Rc::new(RefCell::new(r))))
    });
    vm.add_native("tensor_le".into(), |_vm, args| {
        if args.len() < 2 {
            return Err(TenthError::RuntimeError { line: None, col: None,
                message: "tensor_le(a, b) 期望两个张量参数".into() });
        }
        let (a, b) = match (&args[0], &args[1]) {
            (Value::Tensor(x), Value::Tensor(y)) => (x.clone(), y.clone()),
            _ => return Err(TenthError::RuntimeError { line: None, col: None,
                message: "tensor_le(a, b) 期望两个张量参数".into() }),
        };
        let r = a.borrow().le(&b.borrow())
            .map_err(|m| TenthError::RuntimeError { line: None, col: None, message: m })?;
        Ok(Value::Tensor(Rc::new(RefCell::new(r))))
    });
    vm.add_native("tensor_eq".into(), |_vm, args| {
        if args.len() < 2 {
            return Err(TenthError::RuntimeError { line: None, col: None,
                message: "tensor_eq(a, b) 期望两个张量参数".into() });
        }
        let (a, b) = match (&args[0], &args[1]) {
            (Value::Tensor(x), Value::Tensor(y)) => (x.clone(), y.clone()),
            _ => return Err(TenthError::RuntimeError { line: None, col: None,
                message: "tensor_eq(a, b) 期望两个张量参数".into() }),
        };
        let r = a.borrow().eq(&b.borrow())
            .map_err(|m| TenthError::RuntimeError { line: None, col: None, message: m })?;
        Ok(Value::Tensor(Rc::new(RefCell::new(r))))
    });
    vm.add_native("tensor_ne".into(), |_vm, args| {
        if args.len() < 2 {
            return Err(TenthError::RuntimeError { line: None, col: None,
                message: "tensor_ne(a, b) 期望两个张量参数".into() });
        }
        let (a, b) = match (&args[0], &args[1]) {
            (Value::Tensor(x), Value::Tensor(y)) => (x.clone(), y.clone()),
            _ => return Err(TenthError::RuntimeError { line: None, col: None,
                message: "tensor_ne(a, b) 期望两个张量参数".into() }),
        };
        let r = a.borrow().ne(&b.borrow())
            .map_err(|m| TenthError::RuntimeError { line: None, col: None, message: m })?;
        Ok(Value::Tensor(Rc::new(RefCell::new(r))))
    });
    vm.add_native("cross_entropy".into(), |vm, args| {
        if args.len() < 2 {
            return Err(TenthError::RuntimeError { line: None, col: None, message: "cross_entropy(logits, target) 期望两个张量".into() });
        }
        if let (Value::Tensor(logits), Value::Tensor(target)) = (&args[0], &args[1]) {
            let logits_data = logits.borrow();
            let target_data = target.borrow();
            let is_f32 = logits_data.is_f32();
            let sm = logits_data.softmax().ok_or_else(|| {
                TenthError::RuntimeError { line: None, col: None, message: "cross_entropy 中 softmax 失败".into() }
            })?;
            let eps = 1e-10;
            let sm_data = sm.data.as_standard_layout().to_owned();
            let tgt_flat = target_data.data.as_standard_layout().to_owned();
            let sm_slice = sm_data.as_slice().unwrap_or(&[]);
            let tgt_slice = tgt_flat.as_slice().unwrap_or(&[]);
            let n = sm_slice.len() as f64;
            let mut loss_val = 0.0f64;
            for i in 0..sm_slice.len().min(tgt_slice.len()) {
                let p = sm_slice[i].max(eps);
                loss_val -= tgt_slice[i] * p.ln();
            }
            loss_val /= n.max(1.0);
            // 按 logits dtype 构造对应 loss tensor
            let loss_tensor = if is_f32 {
                Tensor::from_vec_f32(vec![loss_val as f32], vec![1])
            } else {
                Tensor::from_vec(vec![loss_val], vec![1])
            };
            let result = Rc::new(RefCell::new(loss_tensor));
            if vm.recording {
                let sm_rc = Rc::new(RefCell::new(sm));
                if let Some(ref mut tape) = vm.tape {
                    let logits_id = logits.borrow().tape_id
                        .unwrap_or_else(|| tape.input(logits.clone()));
                    let _sm_id = tape.input(sm_rc.clone());
                    let node_id = tape.cross_entropy(
                        logits_id, logits.clone(),
                        sm_rc,
                        target.clone(),
                        result.clone(),
                    );
                    result.borrow_mut().tape_id = Some(node_id);
                }
            }
            Ok(Value::Tensor(result))
        } else {
            Err(TenthError::RuntimeError { line: None, col: None, message: "cross_entropy(logits, target) 期望两个张量".into() })
        }
    });
    // File system functions
    vm.add_native("write_file".into(), |vm, args| {
        if args.len() >= 2 {
            if let (Value::String(path), Value::String(content)) = (&args[0], &args[1]) {
                // H-2: 沙箱校验
                let resolved = if let Some(ref sb) = vm.fs_sandbox {
                    match sb.check_write(path) {
                        Ok(p) => p,
                        Err(e) => return Err(TenthError::RuntimeError { line: None, col: None, message: e }),
                    }
                } else {
                    std::path::PathBuf::from(path)
                };
                match std::fs::write(&resolved, content) {
                    Ok(()) => Ok(Value::Unit),
                    Err(e) => Err(TenthError::RuntimeError { line: None, col: None, message: format!("写入文件失败: {}", e) }),
                }
            } else {
                Err(TenthError::RuntimeError { line: None, col: None, message: "write_file(路径, 内容) 期望两个字符串参数".into() })
            }
        } else {
            Err(TenthError::RuntimeError { line: None, col: None, message: "write_file(路径, 内容) 期望两个字符串参数".into() })
        }
    });
    vm.add_native("path_join".into(), |_vm, args| {
        if args.len() >= 2 {
            if let (Value::String(base), Value::String(rest)) = (&args[0], &args[1]) {
                let joined = std::path::Path::new(base).join(rest);
                Ok(Value::String(joined.to_string_lossy().to_string()))
            } else {
                Err(TenthError::RuntimeError { line: None, col: None, message: "path_join(基础路径, 子路径) 期望两个字符串参数".into() })
            }
        } else {
            Err(TenthError::RuntimeError { line: None, col: None, message: "path_join(基础路径, 子路径) 期望两个字符串参数".into() })
        }
    });
    vm.add_native("path_exists".into(), |vm, args| {
        if let Some(Value::String(path)) = args.first() {
            // H-2: 沙箱校验
            if let Some(ref sb) = vm.fs_sandbox {
                if let Err(e) = sb.check_read(path) {
                    return Err(TenthError::RuntimeError { line: None, col: None, message: e });
                }
            }
            Ok(Value::Bool(std::path::Path::new(path).exists()))
        } else {
            Err(TenthError::RuntimeError { line: None, col: None, message: "path_exists(路径) 期望一个字符串路径".into() })
        }
    });
    vm.add_native("path_is_file".into(), |vm, args| {
        if let Some(Value::String(path)) = args.first() {
            if let Some(ref sb) = vm.fs_sandbox {
                if let Err(e) = sb.check_read(path) {
                    return Err(TenthError::RuntimeError { line: None, col: None, message: e });
                }
            }
            Ok(Value::Bool(std::path::Path::new(path).is_file()))
        } else {
            Err(TenthError::RuntimeError { line: None, col: None, message: "path_is_file(路径) 期望一个字符串路径".into() })
        }
    });
    vm.add_native("path_is_dir".into(), |vm, args| {
        if let Some(Value::String(path)) = args.first() {
            if let Some(ref sb) = vm.fs_sandbox {
                if let Err(e) = sb.check_read(path) {
                    return Err(TenthError::RuntimeError { line: None, col: None, message: e });
                }
            }
            Ok(Value::Bool(std::path::Path::new(path).is_dir()))
        } else {
            Err(TenthError::RuntimeError { line: None, col: None, message: "path_is_dir(路径) 期望一个字符串路径".into() })
        }
    });
    vm.add_native("mkdir".into(), |vm, args| {
        if let Some(Value::String(path)) = args.first() {
            // H-2: 沙箱校验
            let resolved = if let Some(ref sb) = vm.fs_sandbox {
                match sb.check_write(path) {
                    Ok(p) => p,
                    Err(e) => return Err(TenthError::RuntimeError { line: None, col: None, message: e }),
                }
            } else {
                std::path::PathBuf::from(path)
            };
            match std::fs::create_dir_all(&resolved) {
                Ok(()) => Ok(Value::Unit),
                Err(e) => Err(TenthError::RuntimeError { line: None, col: None, message: format!("创建目录失败: {}", e) }),
            }
        } else {
            Err(TenthError::RuntimeError { line: None, col: None, message: "mkdir(路径) 期望一个字符串路径".into() })
        }
    });
    vm.add_native("list_dir".into(), |vm, args| {
        if let Some(Value::String(path)) = args.first() {
            // H-2: 沙箱校验
            let resolved = if let Some(ref sb) = vm.fs_sandbox {
                match sb.check_read(path) {
                    Ok(p) => p,
                    Err(e) => return Err(TenthError::RuntimeError { line: None, col: None, message: e }),
                }
            } else {
                std::path::PathBuf::from(path)
            };
            match std::fs::read_dir(&resolved) {
                Ok(entries) => {
                    let items: Vec<Value> = entries
                        .filter_map(|e| e.ok())
                        .map(|e| Value::String(e.file_name().to_string_lossy().to_string()))
                        .collect();
                    Ok(Value::Vec(Rc::new(RefCell::new(items))))
                }
                Err(e) => Err(TenthError::RuntimeError { line: None, col: None, message: format!("列出目录失败: {}", e) }),
            }
        } else {
            Err(TenthError::RuntimeError { line: None, col: None, message: "list_dir(路径) 期望一个字符串路径".into() })
        }
    });
    vm.add_native("file_size".into(), |vm, args| {
        if let Some(Value::String(path)) = args.first() {
            // H-2: 沙箱校验
            let resolved = if let Some(ref sb) = vm.fs_sandbox {
                match sb.check_read(path) {
                    Ok(p) => p,
                    Err(e) => return Err(TenthError::RuntimeError { line: None, col: None, message: e }),
                }
            } else {
                std::path::PathBuf::from(path)
            };
            match std::fs::metadata(&resolved) {
                Ok(meta) => Ok(Value::Int(meta.len() as i64, BaseType::I32)),
                Err(e) => Err(TenthError::RuntimeError { line: None, col: None, message: format!("获取文件大小失败: {}", e) }),
            }
        } else {
            Err(TenthError::RuntimeError { line: None, col: None, message: "file_size(路径) 期望一个字符串路径".into() })
        }
    });
    vm.add_native("remove_file".into(), |vm, args| {
        if let Some(Value::String(path)) = args.first() {
            // H-2: 沙箱校验
            let resolved = if let Some(ref sb) = vm.fs_sandbox {
                match sb.check_write(path) {
                    Ok(p) => p,
                    Err(e) => return Err(TenthError::RuntimeError { line: None, col: None, message: e }),
                }
            } else {
                std::path::PathBuf::from(path)
            };
            match std::fs::remove_file(&resolved) {
                Ok(()) => Ok(Value::Unit),
                Err(e) => Err(TenthError::RuntimeError { line: None, col: None, message: format!("删除文件失败: {}", e) }),
            }
        } else {
            Err(TenthError::RuntimeError { line: None, col: None, message: "remove_file(路径) 期望一个字符串路径".into() })
        }
    });
    vm.add_native("copy_file".into(), |vm, args| {
        if args.len() >= 2 {
            if let (Value::String(src), Value::String(dst)) = (&args[0], &args[1]) {
                // H-2: 沙箱校验
                let src_resolved = if let Some(ref sb) = vm.fs_sandbox {
                    match sb.check_read(src) {
                        Ok(p) => p,
                        Err(e) => return Err(TenthError::RuntimeError { line: None, col: None, message: e }),
                    }
                } else {
                    std::path::PathBuf::from(src)
                };
                let dst_resolved = if let Some(ref sb) = vm.fs_sandbox {
                    match sb.check_write(dst) {
                        Ok(p) => p,
                        Err(e) => return Err(TenthError::RuntimeError { line: None, col: None, message: e }),
                    }
                } else {
                    std::path::PathBuf::from(dst)
                };
                match std::fs::copy(&src_resolved, &dst_resolved) {
                    Ok(_) => Ok(Value::Unit),
                    Err(e) => Err(TenthError::RuntimeError { line: None, col: None, message: format!("复制文件失败: {}", e) }),
                }
            } else {
                Err(TenthError::RuntimeError { line: None, col: None, message: "copy_file(源路径, 目标路径) 期望两个字符串参数".into() })
            }
        } else {
            Err(TenthError::RuntimeError { line: None, col: None, message: "copy_file(源路径, 目标路径) 期望两个字符串参数".into() })
        }
    });
    vm.add_native("rename_file".into(), |vm, args| {
        if args.len() >= 2 {
            if let (Value::String(src), Value::String(dst)) = (&args[0], &args[1]) {
                // H-2: 沙箱校验
                let src_resolved = if let Some(ref sb) = vm.fs_sandbox {
                    match sb.check_read(src) {
                        Ok(p) => p,
                        Err(e) => return Err(TenthError::RuntimeError { line: None, col: None, message: e }),
                    }
                } else {
                    std::path::PathBuf::from(src)
                };
                let dst_resolved = if let Some(ref sb) = vm.fs_sandbox {
                    match sb.check_write(dst) {
                        Ok(p) => p,
                        Err(e) => return Err(TenthError::RuntimeError { line: None, col: None, message: e }),
                    }
                } else {
                    std::path::PathBuf::from(dst)
                };
                match std::fs::rename(&src_resolved, &dst_resolved) {
                    Ok(()) => Ok(Value::Unit),
                    Err(e) => Err(TenthError::RuntimeError { line: None, col: None, message: format!("重命名文件失败: {}", e) }),
                }
            } else {
                Err(TenthError::RuntimeError { line: None, col: None, message: "rename_file(源路径, 目标路径) 期望两个字符串参数".into() })
            }
        } else {
            Err(TenthError::RuntimeError { line: None, col: None, message: "rename_file(源路径, 目标路径) 期望两个字符串参数".into() })
        }
    });
    vm.add_native("randn".into(), |_vm, args| {
        let rows = match args.first() { Some(Value::Int(n, _)) => *n as usize, _ => 1 };
        let cols = match args.get(1) { Some(Value::Int(n, _)) => *n as usize, _ => 1 };
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let data: Vec<f64> = (0..rows * cols).map(|_| {
            // Box-Muller transform for normal distribution
            let u1: f64 = rng.r#gen::<f64>().max(1e-10);
            let u2: f64 = rng.r#gen::<f64>();
            (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
        }).collect();
        Ok(Value::Tensor(Rc::new(RefCell::new(Tensor::from_vec(data, vec![rows, cols])))))
    });
    vm.add_native("randn_f32".into(), |_vm, args| {
        let rows = match args.first() { Some(Value::Int(n, _)) => *n as usize, _ => 1 };
        let cols = match args.get(1) { Some(Value::Int(n, _)) => *n as usize, _ => 1 };
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let data: Vec<f32> = (0..rows * cols).map(|_| {
            // Box-Muller transform for normal distribution (f32 版本)
            let u1: f32 = rng.r#gen::<f32>().max(1e-10);
            let u2: f32 = rng.r#gen::<f32>();
            (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
        }).collect();
        Ok(Value::Tensor(Rc::new(RefCell::new(Tensor::from_vec_f32(data, vec![rows, cols])))))
    });
    // ── Tensor 构造函数（与 interpreter::natives 对齐，支持任意 shape）──
    // 历史：这些函数仅在 interpreter 实现，JIT/VM 路径下返回 Unit。
    // 补齐后 zeros(256,256,256).numel() 等才能在默认 tenth run 路径下正常工作。
    vm.add_native("zeros".into(), |_vm, args| {
        let shape: Vec<usize> = args.iter()
            .map(|a| a.as_int().unwrap_or(1) as usize)
            .collect();
        Ok(Value::Tensor(Rc::new(RefCell::new(Tensor::zeros(&shape)))))
    });
    vm.add_native("ones".into(), |_vm, args| {
        let shape: Vec<usize> = args.iter()
            .map(|a| a.as_int().unwrap_or(1) as usize)
            .collect();
        Ok(Value::Tensor(Rc::new(RefCell::new(Tensor::ones(&shape)))))
    });
    vm.add_native("rand".into(), |_vm, args| {
        let shape: Vec<usize> = args.iter()
            .map(|a| a.as_int().unwrap_or(1) as usize)
            .collect();
        Ok(Value::Tensor(Rc::new(RefCell::new(Tensor::rand(&shape)))))
    });
    vm.add_native("zeros_f32".into(), |_vm, args| {
        let shape: Vec<usize> = args.iter()
            .map(|a| a.as_int().unwrap_or(1) as usize)
            .collect();
        Ok(Value::Tensor(Rc::new(RefCell::new(Tensor::zeros_f32(&shape)))))
    });
    vm.add_native("ones_f32".into(), |_vm, args| {
        let shape: Vec<usize> = args.iter()
            .map(|a| a.as_int().unwrap_or(1) as usize)
            .collect();
        Ok(Value::Tensor(Rc::new(RefCell::new(Tensor::ones_f32(&shape)))))
    });
    // Wave 2: f16/bf16 构造函数（与 interpreter::natives 双侧注册对齐）
    vm.add_native("zeros_f16".into(), |_vm, args| {
        let shape: Vec<usize> = args.iter()
            .map(|a| a.as_int().unwrap_or(1) as usize)
            .collect();
        Ok(Value::Tensor(Rc::new(RefCell::new(Tensor::zeros_f16(&shape)))))
    });
    vm.add_native("ones_f16".into(), |_vm, args| {
        let shape: Vec<usize> = args.iter()
            .map(|a| a.as_int().unwrap_or(1) as usize)
            .collect();
        Ok(Value::Tensor(Rc::new(RefCell::new(Tensor::ones_f16(&shape)))))
    });
    vm.add_native("zeros_bf16".into(), |_vm, args| {
        let shape: Vec<usize> = args.iter()
            .map(|a| a.as_int().unwrap_or(1) as usize)
            .collect();
        Ok(Value::Tensor(Rc::new(RefCell::new(Tensor::zeros_bf16(&shape)))))
    });
    vm.add_native("ones_bf16".into(), |_vm, args| {
        let shape: Vec<usize> = args.iter()
            .map(|a| a.as_int().unwrap_or(1) as usize)
            .collect();
        Ok(Value::Tensor(Rc::new(RefCell::new(Tensor::ones_bf16(&shape)))))
    });
    vm.add_native("rand_f32".into(), |_vm, args| {
        let shape: Vec<usize> = args.iter()
            .map(|a| a.as_int().unwrap_or(1) as usize)
            .collect();
        Ok(Value::Tensor(Rc::new(RefCell::new(Tensor::rand_f32(&shape)))))
    });
    vm.add_native("HashMap::new".into(), |_vm, _args| {
        Ok(Value::Map(Rc::new(RefCell::new(std::collections::HashMap::new()))))
    });
    vm.add_native("print".into(), |_vm, args| {
        for a in args { print!("{a}"); }
        Ok(Value::Unit)
    });
    vm.add_native("abs".into(), |_vm, args| {
        match args.first() {
            Some(Value::Int(n, _)) => Ok(Value::Int(n.abs(), BaseType::I32)),
            Some(Value::Float(f)) => Ok(Value::Float(f.abs())),
            Some(Value::Float32(f)) => Ok(Value::Float32(f.abs())),
            _ => Err(TenthError::RuntimeError { line: None, col: None, message: "abs() 需要一个数值参数".into() }),
        }
    });
    vm.add_native("sqrt".into(), |_vm, args| {
        match args.first() {
            Some(Value::Float(f)) => Ok(Value::Float(f.sqrt())),
            Some(Value::Float32(f)) => Ok(Value::Float32(f.sqrt())),
            Some(Value::Int(n, _)) => Ok(Value::Float((*n as f64).sqrt())),
            _ => Err(TenthError::RuntimeError { line: None, col: None, message: "sqrt() 需要一个数值参数".into() }),
        }
    });
    vm.add_native("to_float".into(), |_vm, args| {
        match args.first() {
            Some(Value::Int(n, _)) => Ok(Value::Float(*n as f64)),
            Some(Value::Float(f)) => Ok(Value::Float(*f)),
            Some(Value::Float32(f)) => Ok(Value::Float(*f as f64)),
            Some(Value::Tensor(t)) => {
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
                })
            }
            _ => Err(TenthError::RuntimeError { line: None, col: None, message: "to_float() 需要一个数值参数".into() }),
        }
    });
    vm.add_native("to_f64".into(), |_vm, args| {
        match args.first() {
            Some(Value::Int(n, _)) => Ok(Value::Float(*n as f64)),
            Some(Value::Float(f)) => Ok(Value::Float(*f)),
            Some(Value::Float32(f)) => Ok(Value::Float(*f as f64)),
            Some(Value::Tensor(t)) => {
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
                })
            }
            _ => Err(TenthError::RuntimeError { line: None, col: None, message: "to_f64() 需要一个数值参数".into() }),
        }
    });
    vm.add_native("to_f32".into(), |_vm, args| {
        match args.first() {
            Some(Value::Int(n, _)) => Ok(Value::Float32(*n as f32)),
            Some(Value::Float(f)) => Ok(Value::Float32(*f as f32)),
            Some(Value::Float32(f)) => Ok(Value::Float32(*f)),
            Some(Value::Tensor(t)) => {
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
                })
            }
            _ => Err(TenthError::RuntimeError { line: None, col: None, message: "to_f32() 需要一个数值参数".into() }),
        }
    });
    vm.add_native("tensor_from_vec".into(), |_vm, args| {
        if args.len() >= 3 {
            if let (Value::Vec(items), Value::Int(rows, _), Value::Int(cols, _)) = (&args[0], &args[1], &args[2]) {
                // 按 Vec 内元素 dtype 判断：含 Float32 → f32 Tensor
                let has_f32 = items.borrow().iter().any(|v| matches!(v, Value::Float32(_)));
                if has_f32 {
                    let data: Vec<f32> = items.borrow().iter().map(|v| v.as_f32().unwrap_or(0.0)).collect();
                    let tensor = Tensor::from_vec_f32(data, vec![*rows as usize, *cols as usize]);
                    Ok(Value::Tensor(Rc::new(RefCell::new(tensor))))
                } else {
                    let data: Vec<f64> = items.borrow().iter().map(|v| v.as_float().unwrap_or(0.0)).collect();
                    let tensor = Tensor::from_vec(data, vec![*rows as usize, *cols as usize]);
                    Ok(Value::Tensor(Rc::new(RefCell::new(tensor))))
                }
            } else {
                Err(TenthError::RuntimeError { line: None, col: None, message: "tensor_from_vec(vec, rows, cols) 期望一个 Vec 和两个整数".into() })
            }
        } else {
            Err(TenthError::RuntimeError { line: None, col: None, message: "tensor_from_vec(vec, rows, cols) 期望 3 个参数".into() })
        }
    });

    // ── 论文 T37 修复第二批：补齐 VM 缺失的 17 项 native（与 interpreter::natives 对齐）──
    // 历史：这些 native 仅在解释器实现，VM 路径下返回 Unit（DX/ML 训练关键路径断裂）。

    // 1. to_string — 值转字符串（与解释器 value_to_string 对齐到 Display 实现）
    vm.add_native("to_string".into(), |_vm, args| {
        if let Some(arg) = args.first() {
            Ok(Value::String(format!("{}", arg)))
        } else {
            Ok(Value::String(String::new()))
        }
    });
    // 2. type_name — 值类型名
    vm.add_native("type_name".into(), |_vm, args| {
        if let Some(arg) = args.first() {
            let tn = match arg {
                Value::Int(_, _) => "int",
                Value::Float(_) => "float",
                Value::Float32(_) => "float",
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
            Ok(Value::String(tn.to_string()))
        } else {
            Ok(Value::String("unknown".to_string()))
        }
    });
    // 3. with_step_limit(limit, fn) — 步数预算内执行闭包
    //    VM 中闭包以 Value::FnRef 表示（Op::MakeClosure 创建），可通过 call_with_args 调用。
    vm.add_native("with_step_limit".into(), |vm, args| {
        if args.len() < 2 {
            return Err(TenthError::RuntimeError { line: None, col: None,
                message: "with_step_limit(limit, fn) 需要 2 个参数".into(),
            });
        }
        let limit = args[0].as_int().ok_or_else(|| TenthError::RuntimeError { line: None, col: None,
            message: "with_step_limit 的第一个参数必须是整数步数".into(),
        })?;
        let saved_budget = vm.step_budget;
        let saved_deadline = vm.deadline_ms;
        vm.step_budget = Some(limit.max(0) as u64);
        vm.deadline_ms = None;
        let result = match &args[1] {
            Value::FnRef { name, .. } => vm.call_with_args(name, &[]),
            // VM 无法在 native 内执行 tree-walk 闭包；与解释器 Timeout 语义一致返回 Unit。
            _ => {
                vm.step_budget = saved_budget;
                vm.deadline_ms = saved_deadline;
                return Ok(Value::Unit);
            }
        };
        vm.step_budget = saved_budget;
        vm.deadline_ms = saved_deadline;
        match result {
            Ok(v) => Ok(v),
            Err(TenthError::Timeout { .. }) => Ok(Value::Unit),
            Err(e) => Err(e),
        }
    });
    // 4. with_timeout_ms(ms, fn) — 毫秒截止期内执行闭包
    vm.add_native("with_timeout_ms".into(), |vm, args| {
        if args.len() < 2 {
            return Err(TenthError::RuntimeError { line: None, col: None,
                message: "with_timeout_ms(ms, fn) 需要 2 个参数".into(),
            });
        }
        let ms = args[0].as_int().ok_or_else(|| TenthError::RuntimeError { line: None, col: None,
            message: "with_timeout_ms 的第一个参数必须是整数毫秒".into(),
        })?;
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis())
            .unwrap_or(0);
        let saved_budget = vm.step_budget;
        let saved_deadline = vm.deadline_ms;
        // 与解释器一致：用大步数预算作为 tick 载体，deadline 做实际时间比较。
        vm.step_budget = Some(u64::MAX);
        vm.deadline_ms = Some(now + (ms.max(0) as u128));
        let result = match &args[1] {
            Value::FnRef { name, .. } => vm.call_with_args(name, &[]),
            _ => {
                vm.step_budget = saved_budget;
                vm.deadline_ms = saved_deadline;
                return Ok(Value::Unit);
            }
        };
        vm.step_budget = saved_budget;
        vm.deadline_ms = saved_deadline;
        match result {
            Ok(v) => Ok(v),
            Err(TenthError::Timeout { .. }) => Ok(Value::Unit),
            Err(e) => Err(e),
        }
    });
    // 5. is_timeout(result) — 判断是否超时哨兵（Unit）
    vm.add_native("is_timeout".into(), |_vm, args| {
        if let Some(arg) = args.first() {
            Ok(Value::Bool(matches!(arg, Value::Unit)))
        } else {
            Ok(Value::Bool(false))
        }
    });
    // 6. start_grad — 新建 Tape（与 new_grad 同义）
    vm.add_native("start_grad".into(), |vm, _args| {
        vm.tape = Some(Tape::new());
        vm.recording = true;
        Ok(Value::Unit)
    });
    // 7. f64_bits — f64 → i64 位表示
    vm.add_native("f64_bits".into(), |_vm, args| {
        if let Some(arg) = args.first() {
            let f = arg.as_float().ok_or_else(|| TenthError::RuntimeError { line: None, col: None,
                message: "f64_bits() 期望一个 f64 参数".into(),
            })?;
            Ok(Value::Int(f.to_bits() as i64, BaseType::I32))
        } else {
            Err(TenthError::RuntimeError { line: None, col: None, message: "f64_bits() 期望 1 个参数".into() })
        }
    });
    // 8. f64_from_bits — i64 → f64
    vm.add_native("f64_from_bits".into(), |_vm, args| {
        if let Some(arg) = args.first() {
            let n = arg.as_int().ok_or_else(|| TenthError::RuntimeError { line: None, col: None,
                message: "f64_from_bits() 期望一个 i64 参数".into(),
            })?;
            Ok(Value::Float(f64::from_bits(n as u64)))
        } else {
            Err(TenthError::RuntimeError { line: None, col: None, message: "f64_from_bits() 期望 1 个参数".into() })
        }
    });
    // 9-12. 标量数学（sin/cos/ln/pow）— 与解释器一致，仅操作 Float（as_float 自动提升 Int/Float32）
    vm.add_native("sin".into(), |_vm, args| {
        if let Some(arg) = args.first() {
            let n = arg.as_float().ok_or_else(|| TenthError::RuntimeError { line: None, col: None,
                message: "sin() 期望一个数值参数".into(),
            })?;
            Ok(Value::Float(n.sin()))
        } else {
            Err(TenthError::RuntimeError { line: None, col: None, message: "sin() 期望 1 个参数".into() })
        }
    });
    vm.add_native("cos".into(), |_vm, args| {
        if let Some(arg) = args.first() {
            let n = arg.as_float().ok_or_else(|| TenthError::RuntimeError { line: None, col: None,
                message: "cos() 期望一个数值参数".into(),
            })?;
            Ok(Value::Float(n.cos()))
        } else {
            Err(TenthError::RuntimeError { line: None, col: None, message: "cos() 期望 1 个参数".into() })
        }
    });
    vm.add_native("ln".into(), |_vm, args| {
        if let Some(arg) = args.first() {
            let n = arg.as_float().ok_or_else(|| TenthError::RuntimeError { line: None, col: None,
                message: "ln() 期望一个数值参数".into(),
            })?;
            if n <= 0.0 {
                return Err(TenthError::RuntimeError { line: None, col: None,
                    message: "ln() 参数必须 > 0".into(),
                });
            }
            Ok(Value::Float(n.ln()))
        } else {
            Err(TenthError::RuntimeError { line: None, col: None, message: "ln() 期望 1 个参数".into() })
        }
    });
    vm.add_native("pow".into(), |_vm, args| {
        if args.len() >= 2 {
            let base = args[0].as_float().ok_or_else(|| TenthError::RuntimeError { line: None, col: None,
                message: "pow() 期望数值参数".into(),
            })?;
            let exp = args[1].as_float().ok_or_else(|| TenthError::RuntimeError { line: None, col: None,
                message: "pow() 期望数值参数".into(),
            })?;
            Ok(Value::Float(base.powf(exp)))
        } else {
            Err(TenthError::RuntimeError { line: None, col: None, message: "pow() 期望 2 个参数".into() })
        }
    });
    // 13. save_weights(path, tensors) — 张量列表序列化到二进制文件（ML 训练关键路径）
    //     二进制格式与解释器完全一致：i32 num_tensors, [i32 ndim, i32×ndim shape, f64×nel data]（LE）
    vm.add_native("save_weights".into(), |vm, args| {
        if args.len() >= 2 {
            if let Value::String(path) = &args[0] {
                // H-2: 沙箱校验
                let resolved = if let Some(ref sb) = vm.fs_sandbox {
                    match sb.check_write(path) {
                        Ok(p) => p,
                        Err(e) => return Err(TenthError::RuntimeError { line: None, col: None, message: e }),
                    }
                } else {
                    std::path::PathBuf::from(path)
                };
                let tensors: &Rc<RefCell<Vec<Value>>> = match &args[1] {
                    Value::Vec(v) => v,
                    Value::Array(a) => a,
                    _ => return Err(TenthError::RuntimeError { line: None, col: None,
                        message: "save_weights 期望一个张量列表".into(),
                    }),
                };
                let tensors_ref = tensors.borrow();
                let mut bytes: Vec<u8> = Vec::new();
                // Header: magic "THW1" + version=2 + num_tensors (v2 格式)
                bytes.extend(b"THW1");
                bytes.extend(&2i32.to_le_bytes());
                bytes.extend(&(tensors_ref.len() as i32).to_le_bytes());
                for val in tensors_ref.iter() {
                    // 解包 Shared 包装（Vec::push 会将元素包装在 Shared 中）
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
                return Ok(Value::Unit);
            }
        }
        Err(TenthError::RuntimeError { line: None, col: None,
            message: "save_weights(路径, 张量列表)".into(),
        })
    });
    // 14. load_weights(path) — 从二进制文件反序列化张量列表（ML 训练关键路径）
    vm.add_native("load_weights".into(), |vm, args| {
        if let Some(Value::String(path)) = args.first() {
            // H-2: 沙箱校验
            let resolved = if let Some(ref sb) = vm.fs_sandbox {
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
                    Ok(Value::Vec(Rc::new(RefCell::new(result))))
                }
                Err(e) => Err(TenthError::RuntimeError { line: None, col: None,
                    message: format!("load_weights: {}", e),
                }),
            }
        } else {
            Err(TenthError::RuntimeError { line: None, col: None,
                message: "load_weights(路径)".into(),
            })
        }
    });
    // 15. format(template, args...) — 模板字符串格式化（{}/{{/}}）
    //     支持命名参数（通过最后一个 Map 参数传递）、格式说明符（{:>5} / {:.2f}）、
    //     越界返回错误（而非原样输出占位符原文）
    vm.add_native("format".into(), |_vm, args| {
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
            Ok(Value::String(result))
        } else {
            Err(TenthError::RuntimeError { line: None, col: None,
                message: "format() 第一个参数必须是字符串模板".into(),
            })
        }
    });
    // 16. parse_int(s) — 字符串→整数（解析失败返回 0，与解释器一致）
    vm.add_native("parse_int".into(), |_vm, args| {
        if let Some(Value::String(s)) = args.first() {
            Ok(Value::Int(s.trim().parse::<i64>().unwrap_or(0), BaseType::I32))
        } else {
            Err(TenthError::RuntimeError { line: None, col: None,
                message: "parse_int() 期望一个字符串参数".into(),
            })
        }
    });
    // 17. parse_float(s) — 字符串→浮点（解析失败返回 0.0，与解释器一致）
    vm.add_native("parse_float".into(), |_vm, args| {
        if let Some(Value::String(s)) = args.first() {
            Ok(Value::Float(s.trim().parse::<f64>().unwrap_or(0.0)))
        } else {
            Err(TenthError::RuntimeError { line: None, col: None,
                message: "parse_float() 期望一个字符串参数".into(),
            })
        }
    });

    // 18. assert(condition, message?) — 断言条件为真，否则 panic
    vm.add_native("assert".into(), |_vm, args| {
        if args.is_empty() {
            return Err(TenthError::RuntimeError { line: None, col: None,
                message: "assert() 需要至少一个参数（布尔条件）".into(),
            });
        }
        match &args[0] {
            Value::Bool(true) => Ok(Value::Unit),
            Value::Bool(false) => {
                let msg = if args.len() >= 2 {
                    if let Value::String(s) = &args[1] { s.clone() } else { format!("{}", args[1]) }
                } else {
                    "assertion failed".to_string()
                };
                Err(TenthError::RuntimeError { line: None, col: None, message: msg })
            }
            _ => Err(TenthError::RuntimeError { line: None, col: None,
                message: "assert() 需要一个布尔值作为第一个参数".into(),
            }),
        }
    });

    // 19. assert_eq(left, right, message?) — 断言两值相等，否则 panic
    vm.add_native("assert_eq".into(), |_vm, args| {
        if args.len() < 2 {
            return Err(TenthError::RuntimeError { line: None, col: None,
                message: "assert_eq() 需要至少两个参数（左值，右值）".into(),
            });
        }
        let left_str = format!("{}", args[0]);
        let right_str = format!("{}", args[1]);
        if left_str == right_str {
            Ok(Value::Unit)
        } else {
            let extra = if args.len() >= 3 {
                if let Value::String(s) = &args[2] {
                    if !s.is_empty() { format!(" — {}", s) } else { String::new() }
                } else { String::new() }
            } else { String::new() };
            Err(TenthError::RuntimeError { line: None, col: None,
                message: format!("assertion failed: {} != {}{}", left_str, right_str, extra),
            })
        }
    });

    // ── 问题29：智能指针 Box/Rc/Arc/Pin ──
    vm.add_native("Box::new".into(), |_vm, args| {
        if args.is_empty() {
            return Err(TenthError::RuntimeError { line: None, col: None, message: "Box::new() 需要 1 个参数".into() });
        }
        Ok(Value::HeapBox(Box::new(args[0].clone())))
    });
    vm.add_native("Rc::new".into(), |_vm, args| {
        if args.is_empty() {
            return Err(TenthError::RuntimeError { line: None, col: None, message: "Rc::new() 需要 1 个参数".into() });
        }
        Ok(Value::SharedBox(Rc::new(RefCell::new(args[0].clone()))))
    });
    vm.add_native("Arc::new".into(), |_vm, args| {
        // Arc 暂用 Rc 等价实现
        if args.is_empty() {
            return Err(TenthError::RuntimeError { line: None, col: None, message: "Arc::new() 需要 1 个参数".into() });
        }
        Ok(Value::SharedBox(Rc::new(RefCell::new(args[0].clone()))))
    });
    vm.add_native("Pin::new".into(), |_vm, args| {
        if args.is_empty() {
            return Err(TenthError::RuntimeError { line: None, col: None, message: "Pin::new() 需要 1 个参数".into() });
        }
        Ok(Value::Pin(Box::new(args[0].clone())))
    });

    // ── 问题35：BigInt 运算 ──
    vm.add_native("bigint_add".into(), |_vm, args| {
        if args.len() < 2 { return Err(TenthError::RuntimeError { line: None, col: None, message: "bigint_add() 需要 2 个 bigint 参数".into() }); }
        let a = match &args[0] { Value::BigInt(s) => s.clone(), _ => return Err(TenthError::RuntimeError { line: None, col: None, message: "参数必须是 BigInt".into() }) };
        let b = match &args[1] { Value::BigInt(s) => s.clone(), _ => return Err(TenthError::RuntimeError { line: None, col: None, message: "参数必须是 BigInt".into() }) };
        // 简单十进制字符串加法
        let result = bigint_add_str(&a, &b);
        Ok(Value::BigInt(result))
    });
    vm.add_native("bigint_sub".into(), |_vm, args| {
        if args.len() < 2 { return Err(TenthError::RuntimeError { line: None, col: None, message: "bigint_sub() 需要 2 个 bigint 参数".into() }); }
        let a = match &args[0] { Value::BigInt(s) => s.clone(), _ => return Err(TenthError::RuntimeError { line: None, col: None, message: "参数必须是 BigInt".into() }) };
        let b = match &args[1] { Value::BigInt(s) => s.clone(), _ => return Err(TenthError::RuntimeError { line: None, col: None, message: "参数必须是 BigInt".into() }) };
        let result = bigint_sub_str(&a, &b);
        Ok(Value::BigInt(result))
    });
    vm.add_native("bigint_mul".into(), |_vm, args| {
        if args.len() < 2 { return Err(TenthError::RuntimeError { line: None, col: None, message: "bigint_mul() 需要 2 个 bigint 参数".into() }); }
        let a = match &args[0] { Value::BigInt(s) => s.clone(), _ => return Err(TenthError::RuntimeError { line: None, col: None, message: "参数必须是 BigInt".into() }) };
        let b = match &args[1] { Value::BigInt(s) => s.clone(), _ => return Err(TenthError::RuntimeError { line: None, col: None, message: "参数必须是 BigInt".into() }) };
        let result = bigint_mul_str(&a, &b);
        Ok(Value::BigInt(result))
    });

    // ── 问题36：Complex 运算 ──
    vm.add_native("complex_add".into(), |_vm, args| {
        if args.len() < 2 { return Err(TenthError::RuntimeError { line: None, col: None, message: "complex_add() 需要 2 个 complex 参数".into() }); }
        let (re1, im1) = match &args[0] { Value::Complex(r, i) => (*r, *i), _ => return Err(TenthError::RuntimeError { line: None, col: None, message: "参数必须是 Complex".into() }) };
        let (re2, im2) = match &args[1] { Value::Complex(r, i) => (*r, *i), _ => return Err(TenthError::RuntimeError { line: None, col: None, message: "参数必须是 Complex".into() }) };
        Ok(Value::Complex(re1 + re2, im1 + im2))
    });
    vm.add_native("complex_sub".into(), |_vm, args| {
        if args.len() < 2 { return Err(TenthError::RuntimeError { line: None, col: None, message: "complex_sub() 需要 2 个 complex 参数".into() }); }
        let (re1, im1) = match &args[0] { Value::Complex(r, i) => (*r, *i), _ => return Err(TenthError::RuntimeError { line: None, col: None, message: "参数必须是 Complex".into() }) };
        let (re2, im2) = match &args[1] { Value::Complex(r, i) => (*r, *i), _ => return Err(TenthError::RuntimeError { line: None, col: None, message: "参数必须是 Complex".into() }) };
        Ok(Value::Complex(re1 - re2, im1 - im2))
    });
    vm.add_native("complex_mul".into(), |_vm, args| {
        if args.len() < 2 { return Err(TenthError::RuntimeError { line: None, col: None, message: "complex_mul() 需要 2 个 complex 参数".into() }); }
        let (re1, im1) = match &args[0] { Value::Complex(r, i) => (*r, *i), _ => return Err(TenthError::RuntimeError { line: None, col: None, message: "参数必须是 Complex".into() }) };
        let (re2, im2) = match &args[1] { Value::Complex(r, i) => (*r, *i), _ => return Err(TenthError::RuntimeError { line: None, col: None, message: "参数必须是 Complex".into() }) };
        Ok(Value::Complex(re1 * re2 - im1 * im2, re1 * im2 + im1 * re2))
    });
    vm.add_native("complex_div".into(), |_vm, args| {
        if args.len() < 2 { return Err(TenthError::RuntimeError { line: None, col: None, message: "complex_div() 需要 2 个 complex 参数".into() }); }
        let (re1, im1) = match &args[0] { Value::Complex(r, i) => (*r, *i), _ => return Err(TenthError::RuntimeError { line: None, col: None, message: "参数必须是 Complex".into() }) };
        let (re2, im2) = match &args[1] { Value::Complex(r, i) => (*r, *i), _ => return Err(TenthError::RuntimeError { line: None, col: None, message: "参数必须是 Complex".into() }) };
        let denom = re2 * re2 + im2 * im2;
        if denom == 0.0 {
            return Err(TenthError::RuntimeError { line: None, col: None, message: "Complex 除法：分母为零".into() });
        }
        Ok(Value::Complex((re1 * re2 + im1 * im2) / denom, (im1 * re2 - re1 * im2) / denom))
    });

    // ── 问题37：Decimal 运算 ──
    vm.add_native("decimal_add".into(), |_vm, args| {
        if args.len() < 2 { return Err(TenthError::RuntimeError { line: None, col: None, message: "decimal_add() 需要 2 个 decimal 参数".into() }); }
        let a = match &args[0] { Value::Decimal(s) => s.clone(), _ => return Err(TenthError::RuntimeError { line: None, col: None, message: "参数必须是 Decimal".into() }) };
        let b = match &args[1] { Value::Decimal(s) => s.clone(), _ => return Err(TenthError::RuntimeError { line: None, col: None, message: "参数必须是 Decimal".into() }) };
        Ok(Value::Decimal(decimal_add_str(&a, &b)))
    });
    vm.add_native("decimal_sub".into(), |_vm, args| {
        if args.len() < 2 { return Err(TenthError::RuntimeError { line: None, col: None, message: "decimal_sub() 需要 2 个 decimal 参数".into() }); }
        let a = match &args[0] { Value::Decimal(s) => s.clone(), _ => return Err(TenthError::RuntimeError { line: None, col: None, message: "参数必须是 Decimal".into() }) };
        let b = match &args[1] { Value::Decimal(s) => s.clone(), _ => return Err(TenthError::RuntimeError { line: None, col: None, message: "参数必须是 Decimal".into() }) };
        Ok(Value::Decimal(decimal_sub_str(&a, &b)))
    });
    vm.add_native("decimal_mul".into(), |_vm, args| {
        if args.len() < 2 { return Err(TenthError::RuntimeError { line: None, col: None, message: "decimal_mul() 需要 2 个 decimal 参数".into() }); }
        let a = match &args[0] { Value::Decimal(s) => s.clone(), _ => return Err(TenthError::RuntimeError { line: None, col: None, message: "参数必须是 Decimal".into() }) };
        let b = match &args[1] { Value::Decimal(s) => s.clone(), _ => return Err(TenthError::RuntimeError { line: None, col: None, message: "参数必须是 Decimal".into() }) };
        Ok(Value::Decimal(decimal_mul_str(&a, &b)))
    });
    vm.add_native("decimal_div".into(), |_vm, args| {
        if args.len() < 2 { return Err(TenthError::RuntimeError { line: None, col: None, message: "decimal_div() 需要 2 个 decimal 参数".into() }); }
        let a = match &args[0] { Value::Decimal(s) => s.clone(), _ => return Err(TenthError::RuntimeError { line: None, col: None, message: "参数必须是 Decimal".into() }) };
        let b = match &args[1] { Value::Decimal(s) => s.clone(), _ => return Err(TenthError::RuntimeError { line: None, col: None, message: "参数必须是 Decimal".into() }) };
        Ok(Value::Decimal(decimal_div_str(&a, &b)))
    });

    // ── B批：Unicode 规范化 ──
    vm.add_native("unicode_nfc".into(), |_vm, args| {
        if let Some(Value::String(s)) = args.first() {
            Ok(Value::String(s.chars().nfc().collect::<String>()))
        } else {
            Err(TenthError::RuntimeError { line: None, col: None, message: "_unicode_nfc 需要 1 个 String 参数".into() })
        }
    });
    vm.add_native("unicode_nfd".into(), |_vm, args| {
        if let Some(Value::String(s)) = args.first() {
            Ok(Value::String(s.chars().nfd().collect::<String>()))
        } else {
            Err(TenthError::RuntimeError { line: None, col: None, message: "_unicode_nfd 需要 1 个 String 参数".into() })
        }
    });

    // ── B批：UTF-8 ↔ UTF-16 ──
    vm.add_native("str_to_utf16".into(), |_vm, args| {
        if let Some(Value::String(s)) = args.first() {
            let encoded: Vec<u16> = s.encode_utf16().collect();
            let result: Vec<Value> = encoded.into_iter()
                .map(|c| Value::Int(c as i64, BaseType::I32))
                .collect();
            Ok(Value::Vec(Rc::new(RefCell::new(result))))
        } else {
            Err(TenthError::RuntimeError { line: None, col: None, message: "_str_to_utf16 需要 1 个 String 参数".into() })
        }
    });
    vm.add_native("utf16_to_str".into(), |_vm, args| {
        if let Some(Value::Vec(arr)) = args.first() {
            let code_units: Vec<u16> = arr.borrow().iter()
                .map(|v| match v {
                    Value::Int(n, _) => *n as u16,
                    _ => 0,
                })
                .collect();
            let result = String::from_utf16(&code_units)
                .unwrap_or_else(|_| {
                    // 替换无效序列
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
                                } else {
                                    s.push('\u{FFFD}');
                                }
                                i += 2;
                                continue;
                            }
                        } else if c >= 0xDC00 && c <= 0xDFFF {
                            s.push('\u{FFFD}');
                            i += 1;
                            continue;
                        }
                        if let Some(ch) = char::from_u32(c as u32) {
                            s.push(ch);
                        } else {
                            s.push('\u{FFFD}');
                        }
                        i += 1;
                    }
                    s
                });
            Ok(Value::String(result))
        } else {
            Err(TenthError::RuntimeError { line: None, col: None, message: "_utf16_to_str 需要 1 个 Vec 参数".into() })
        }
    });

    // ── B批：UTF-8 ↔ 字节数组 ──
    vm.add_native("str_to_bytes".into(), |_vm, args| {
        if let Some(Value::String(s)) = args.first() {
            let bytes: Vec<Value> = s.bytes()
                .map(|b| Value::Int(b as i64, BaseType::I32))
                .collect();
            Ok(Value::Vec(Rc::new(RefCell::new(bytes))))
        } else {
            Err(TenthError::RuntimeError { line: None, col: None, message: "_str_to_bytes 需要 1 个 String 参数".into() })
        }
    });
    vm.add_native("bytes_to_str".into(), |_vm, args| {
        if let Some(Value::Vec(arr)) = args.first() {
            let bytes: Vec<u8> = arr.borrow().iter()
                .map(|v| match v {
                    Value::Int(n, _) => *n as u8,
                    _ => 0,
                })
                .collect();
            let result = String::from_utf8_lossy(&bytes).to_string();
            Ok(Value::String(result))
        } else {
            Err(TenthError::RuntimeError { line: None, col: None, message: "_bytes_to_str 需要 1 个 Vec 参数".into() })
        }
    });

    // ── B批：Base64 ──
    vm.add_native("base64_encode".into(), |_vm, args| {
        if let Some(Value::Vec(arr)) = args.first() {
            let bytes: Vec<u8> = arr.borrow().iter()
                .map(|v| match v {
                    Value::Int(n, _) => *n as u8,
                    _ => 0,
                })
                .collect();
            use base64::engine::general_purpose;
            Ok(Value::String(general_purpose::STANDARD.encode(&bytes)))
        } else {
            Err(TenthError::RuntimeError { line: None, col: None, message: "base64_encode 需要 1 个 Vec 参数".into() })
        }
    });
    vm.add_native("base64_decode".into(), |_vm, args| {
        if let Some(Value::String(s)) = args.first() {
            use base64::engine::general_purpose;
            match general_purpose::STANDARD.decode(s) {
                Ok(bytes) => {
                    let result: Vec<Value> = bytes.into_iter()
                        .map(|b| Value::Int(b as i64, BaseType::I32))
                        .collect();
                    Ok(ok_result(Value::Vec(Rc::new(RefCell::new(result)))))
                }
                Err(e) => Ok(err_result(format!("Base64 解码失败: {e}"))),
            }
        } else {
            Ok(err_result("_base64_decode 需要 1 个 String 参数"))
        }
    });

    // ── B批：十六进制 ──
    vm.add_native("hex_encode".into(), |_vm, args| {
        if let Some(Value::Vec(arr)) = args.first() {
            let bytes: Vec<u8> = arr.borrow().iter()
                .map(|v| match v {
                    Value::Int(n, _) => *n as u8,
                    _ => 0,
                })
                .collect();
            Ok(Value::String(bytes.iter().map(|b| format!("{:02x}", b)).collect::<String>()))
        } else {
            Err(TenthError::RuntimeError { line: None, col: None, message: "hex_encode 需要 1 个 Vec 参数".into() })
        }
    });
    vm.add_native("hex_decode".into(), |_vm, args| {
        if let Some(Value::String(s)) = args.first() {
            let trimmed = s.trim();
            if trimmed.len() % 2 != 0 {
                return Ok(err_result("十六进制字符串长度必须为偶数"));
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
                    Ok(ok_result(Value::Vec(Rc::new(RefCell::new(result)))))
                }
                Err(e) => Ok(err_result(format!("十六进制解码失败: {e}"))),
            }
        } else {
            Ok(err_result("_hex_decode 需要 1 个 String 参数"))
        }
    });

    // ── B批：URL 编解码 ──
    vm.add_native("url_encode".into(), |_vm, args| {
        if let Some(Value::String(s)) = args.first() {
            use percent_encoding::{utf8_percent_encode, NON_ALPHANUMERIC};
            Ok(Value::String(utf8_percent_encode(s, NON_ALPHANUMERIC).to_string()))
        } else {
            Err(TenthError::RuntimeError { line: None, col: None, message: "url_encode 需要 1 个 String 参数".into() })
        }
    });
    vm.add_native("url_decode".into(), |_vm, args| {
        if let Some(Value::String(s)) = args.first() {
            use percent_encoding::percent_decode;
            match percent_decode(s.as_bytes()).decode_utf8() {
                Ok(decoded) => Ok(ok_result(Value::String(decoded.to_string()))),
                Err(_) => Ok(err_result("URL 解码失败：无效的百分号编码序列")),
            }
        } else {
            Ok(err_result("_url_decode 需要 1 个 String 参数"))
        }
    });

    // ── B批：编码转换新 API 别名 ──
    vm.add_native("to_utf8".into(), |_vm, args| {
        if let Some(Value::String(s)) = args.first() {
            let bytes: Vec<Value> = s.bytes()
                .map(|b| Value::Int(b as i64, BaseType::I32))
                .collect();
            Ok(Value::Vec(Rc::new(RefCell::new(bytes))))
        } else {
            Err(TenthError::RuntimeError { line: None, col: None, message: "_to_utf8 需要 1 个 String 参数".into() })
        }
    });
    vm.add_native("to_utf16".into(), |_vm, args| {
        if let Some(Value::String(s)) = args.first() {
            let encoded: Vec<u16> = s.encode_utf16().collect();
            let result: Vec<Value> = encoded.into_iter()
                .map(|c| Value::Int(c as i64, BaseType::I32))
                .collect();
            Ok(Value::Vec(Rc::new(RefCell::new(result))))
        } else {
            Err(TenthError::RuntimeError { line: None, col: None, message: "_to_utf16 需要 1 个 String 参数".into() })
        }
    });
    vm.add_native("from_utf16".into(), |_vm, args| {
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
                                } else {
                                    s.push('\u{FFFD}');
                                }
                                i += 2;
                                continue;
                            }
                        } else if c >= 0xDC00 && c <= 0xDFFF {
                            s.push('\u{FFFD}');
                            i += 1;
                            continue;
                        }
                        if let Some(ch) = char::from_u32(c as u32) {
                            s.push(ch);
                        } else {
                            s.push('\u{FFFD}');
                        }
                        i += 1;
                    }
                    s
                });
            Ok(Value::String(result))
        } else {
            Err(TenthError::RuntimeError { line: None, col: None, message: "_from_utf16 需要 1 个 Vec 参数".into() })
        }
    });

    // ── B批：GBK 编码 ──
    vm.add_native("to_gbk".into(), |_vm, args| {
        if let Some(Value::String(s)) = args.first() {
            let (bytes, _, _) = encoding_rs::GBK.encode(s);
            let result: Vec<Value> = bytes.iter()
                .map(|b| Value::Int(*b as i64, BaseType::I32))
                .collect();
            Ok(Value::Vec(Rc::new(RefCell::new(result))))
        } else {
            Err(TenthError::RuntimeError { line: None, col: None, message: "_to_gbk 需要 1 个 String 参数".into() })
        }
    });
    vm.add_native("from_gbk".into(), |_vm, args| {
        if let Some(Value::Vec(arr)) = args.first() {
            let bytes: Vec<u8> = arr.borrow().iter()
                .map(|v| match v {
                    Value::Int(n, _) => *n as u8,
                    _ => 0,
                })
                .collect();
            let (result, _, _) = encoding_rs::GBK.decode(&bytes);
            Ok(Value::String(result.to_string()))
        } else {
            Err(TenthError::RuntimeError { line: None, col: None, message: "_from_gbk 需要 1 个 Vec 参数".into() })
        }
    });
}

// ── 问题35：BigInt 字符串算术辅助函数 ──
/// 大整数加法（字符串十进制，支持负数）
pub(crate) fn bigint_add_str(a: &str, b: &str) -> String {
    if a.starts_with('-') && b.starts_with('-') {
        return format!("-{}", bigint_add_str_unsigned(&a[1..], &b[1..]));
    }
    if a.starts_with('-') {
        return bigint_sub_str_unsigned(&b[1..], a);
    }
    if b.starts_with('-') {
        return bigint_sub_str_unsigned(&a[1..], b);
    }
    bigint_add_str_unsigned(a, b)
}

/// 大整数减法（字符串十进制，支持负数）
pub(crate) fn bigint_sub_str(a: &str, b: &str) -> String {
    if a.starts_with('-') && b.starts_with('-') {
        return bigint_sub_str_unsigned(&b[1..], &a[1..]);
    }
    if a.starts_with('-') {
        return format!("-{}", bigint_add_str_unsigned(&a[1..], b));
    }
    if b.starts_with('-') {
        return bigint_add_str_unsigned(a, &b[1..]);
    }
    bigint_sub_str_unsigned(a, b)
}

/// 大整数乘法（字符串十进制，支持负数）
pub(crate) fn bigint_mul_str(a: &str, b: &str) -> String {
    let neg = (a.starts_with('-') && !b.starts_with('-')) || (!a.starts_with('-') && b.starts_with('-'));
    let a_abs = a.trim_start_matches('-');
    let b_abs = b.trim_start_matches('-');
    let result = bigint_mul_str_unsigned(a_abs, b_abs);
    if neg && result != "0" { format!("-{}", result) } else { result }
}

/// 无符号大整数加法
fn bigint_add_str_unsigned(a: &str, b: &str) -> String {
    let mut result = String::new();
    let mut carry = 0u8;
    let a_chars: Vec<char> = a.chars().collect();
    let b_chars: Vec<char> = b.chars().collect();
    let mut i = a_chars.len() as isize - 1;
    let mut j = b_chars.len() as isize - 1;
    while i >= 0 || j >= 0 || carry > 0 {
        let digit_a = if i >= 0 { a_chars[i as usize].to_digit(10).unwrap_or(0) as u8 } else { 0 };
        let digit_b = if j >= 0 { b_chars[j as usize].to_digit(10).unwrap_or(0) as u8 } else { 0 };
        let sum = digit_a + digit_b + carry;
        result.push(char::from(b'0' + (sum % 10)));
        carry = sum / 10;
        i -= 1;
        j -= 1;
    }
    result.chars().rev().collect()
}

/// 无符号大整数减法（假设 a >= b）
fn bigint_sub_str_unsigned(a: &str, b: &str) -> String {
    // 先处理 a < b 的情况（结果为负）
    if a.len() < b.len() || (a.len() == b.len() && a < b) {
        let result = bigint_sub_str_unsigned(b, a);
        if result == "0" { return result; }
        return format!("-{}", result);
    }
    let mut result = String::new();
    let mut borrow = 0i8;
    let a_chars: Vec<char> = a.chars().collect();
    let b_chars: Vec<char> = b.chars().collect();
    let mut i = a_chars.len() as isize - 1;
    let mut j = b_chars.len() as isize - 1;
    while i >= 0 {
        let digit_a = a_chars[i as usize].to_digit(10).unwrap_or(0) as i8;
        let digit_b = if j >= 0 { b_chars[j as usize].to_digit(10).unwrap_or(0) as i8 } else { 0 };
        let mut diff = digit_a - digit_b - borrow;
        if diff < 0 {
            diff += 10;
            borrow = 1;
        } else {
            borrow = 0;
        }
        result.push(char::from(b'0' + diff as u8));
        i -= 1;
        j -= 1;
    }
    // 去除前导零
    let trimmed: String = result.chars().rev().skip_while(|&c| c == '0').collect();
    if trimmed.is_empty() { "0".to_string() } else { trimmed }
}

/// 无符号大整数乘法（小学竖式法）
fn bigint_mul_str_unsigned(a: &str, b: &str) -> String {
    if a == "0" || b == "0" { return "0".to_string(); }
    let a_chars: Vec<u8> = a.chars().map(|c| c.to_digit(10).unwrap_or(0) as u8).collect();
    let b_chars: Vec<u8> = b.chars().map(|c| c.to_digit(10).unwrap_or(0) as u8).collect();
    let mut result = vec![0u8; a_chars.len() + b_chars.len()];
    for (i, &digit_a) in a_chars.iter().enumerate().rev() {
        for (j, &digit_b) in b_chars.iter().enumerate().rev() {
            let prod = digit_a as u16 * digit_b as u16 + result[i + j + 1] as u16;
            result[i + j + 1] = (prod % 10) as u8;
            result[i + j] += (prod / 10) as u8;
        }
    }
    let s: String = result.into_iter().map(|d| char::from(b'0' + d)).collect();
    s.trim_start_matches('0').to_string()
}

// ── 问题37：Decimal 字符串算术辅助函数 ──
/// 十进制加法（精确字符串运算）
pub(crate) fn decimal_add_str(a: &str, b: &str) -> String {
    let (a_int, a_frac) = split_decimal(a);
    let (b_int, b_frac) = split_decimal(b);
    let max_frac_len = a_frac.len().max(b_frac.len());
    let a_frac_padded = format!("{:0<width$}", a_frac, width = max_frac_len);
    let b_frac_padded = format!("{:0<width$}", b_frac, width = max_frac_len);
    // 将小数部分当作整数相加
    let frac_sum = bigint_add_str_unsigned(&a_frac_padded, &b_frac_padded);
    let (frac_result, carry) = if frac_sum.len() > max_frac_len {
        (frac_sum[frac_sum.len() - max_frac_len..].to_string(), &frac_sum[..frac_sum.len() - max_frac_len])
    } else {
        (format!("{:0>width$}", frac_sum, width = max_frac_len), "0")
    };
    let int_sum = bigint_add_str_unsigned(a_int, b_int);
    let int_result = bigint_add_str_unsigned(&int_sum, carry);
    if frac_result.chars().all(|c| c == '0') {
        int_result
    } else {
        format!("{}.{}", int_result, frac_result.trim_end_matches('0'))
    }
}

/// 十进制减法（精确字符串运算）
pub(crate) fn decimal_sub_str(a: &str, b: &str) -> String {
    let (a_int, a_frac) = split_decimal(a);
    let (b_int, b_frac) = split_decimal(b);
    let max_frac_len = a_frac.len().max(b_frac.len());
    let a_frac_padded = format!("{:0<width$}", a_frac, width = max_frac_len);
    let b_frac_padded = format!("{:0<width$}", b_frac, width = max_frac_len);
    let frac_sub = bigint_sub_str_unsigned(&a_frac_padded, &b_frac_padded);
    let (frac_result, borrow) = if frac_sub.starts_with('-') {
        // 需要从整数部分借位
        let frac_val = bigint_sub_str_unsigned(&b_frac_padded, &a_frac_padded);
        let power = format!("1{:0<width$}", "", width = max_frac_len);
        let borrowed = bigint_sub_str_unsigned(&power, &frac_val);
        let formatted = format!("{:0>width$}", borrowed, width = max_frac_len);
        (formatted, true)
    } else {
        (format!("{:0>width$}", frac_sub, width = max_frac_len), false)
    };
    let int_sub = if borrow {
        bigint_sub_str_unsigned(a_int, &bigint_add_str_unsigned(b_int, "1"))
    } else {
        bigint_sub_str_unsigned(a_int, b_int)
    };
    if frac_result.chars().all(|c| c == '0') {
        int_sub
    } else {
        format!("{}.{}", int_sub, frac_result.trim_end_matches('0'))
    }
}

/// 十进制乘法（精确字符串运算）
pub(crate) fn decimal_mul_str(a: &str, b: &str) -> String {
    let neg = (a.starts_with('-') && !b.starts_with('-')) || (!a.starts_with('-') && b.starts_with('-'));
    let a_abs = a.trim_start_matches('-');
    let b_abs = b.trim_start_matches('-');
    let (a_int, a_frac) = split_decimal(a_abs);
    let (b_int, b_frac) = split_decimal(b_abs);
    let total_frac_digits = a_frac.len() + b_frac.len();
    let a_str = format!("{}{}", a_int, a_frac);
    let b_str = format!("{}{}", b_int, b_frac);
    let prod = bigint_mul_str_unsigned(&a_str, &b_str);
    let result = if prod.len() > total_frac_digits {
        format!("{}.{}", &prod[..prod.len() - total_frac_digits], &prod[prod.len() - total_frac_digits..])
    } else {
        format!("0.{:0>width$}", prod, width = total_frac_digits)
    };
    let trimmed = result.trim_end_matches('0').trim_end_matches('.');
    if trimmed.is_empty() { return "0".to_string(); }
    if neg && trimmed != "0" { format!("-{}", trimmed) } else { trimmed.to_string() }
}

/// 十进制除法（近似到 10 位小数）
pub(crate) fn decimal_div_str(a: &str, b: &str) -> String {
    if b == "0" || b == "0.0" {
        return "NaN".to_string();
    }
    let neg = (a.starts_with('-') && !b.starts_with('-')) || (!a.starts_with('-') && b.starts_with('-'));
    let a_abs = a.trim_start_matches('-');
    let b_abs = b.trim_start_matches('-');
    let (a_int, a_frac) = split_decimal(a_abs);
    let (b_int, b_frac) = split_decimal(b_abs);
    // 将除数、被除数转为整数
    let max_frac = a_frac.len().max(b_frac.len());
    let a_padded = format!("{}{:0<width$}", a_int, a_frac, width = max_frac);
    let b_padded = format!("{}{:0<width$}", b_int, b_frac, width = max_frac);
    // 长除法：求商到 10 位小数
    let precision = 10;
    let mut dividend = bigint_sub_str_unsigned(&a_padded, &"0"); // just clone
    let divisor = bigint_sub_str_unsigned(&b_padded, &"0");
    // 简单长除法
    let dividend_val: String = a_padded.clone();
    let divisor_val: String = b_padded.clone();
    let mut quotient = String::new();
    let mut remainder = String::new();
    for (i, c) in dividend_val.chars().enumerate() {
        remainder.push(c);
        let rem_trimmed = remainder.trim_start_matches('0');
        if rem_trimmed.is_empty() { remainder = "0".to_string(); }
        let (q, r) = divide_one_digit(&remainder, &divisor_val);
        quotient.push(q);
        remainder = r;
    }
    // 小数部分
    if !quotient.is_empty() {
        // 已有整数部分
    } else {
        quotient = "0".to_string();
    }
    let mut frac = String::new();
    for _ in 0..precision {
        remainder.push('0');
        let (q, r) = divide_one_digit(&remainder, &divisor_val);
        frac.push(q);
        remainder = r;
        if remainder == "0" { break; }
    }
    let result = if frac.chars().all(|c| c == '0') {
        quotient.trim_start_matches('0').to_string()
    } else {
        format!("{}.{}", quotient.trim_start_matches('0'), frac)
    };
    if result.is_empty() || result == "." { return "0".to_string(); }
    if neg && result != "0" { format!("-{}", result) } else { result }
}

/// 一次除法迭代：返回商的一位数字和余数
fn divide_one_digit(dividend: &str, divisor: &str) -> (char, String) {
    let dividend_trimmed = dividend.trim_start_matches('0');
    if dividend_trimmed.is_empty() { return ('0', "0".to_string()); }
    let d = dividend_trimmed.to_string();
    let dv = divisor.trim_start_matches('0');
    if dv.is_empty() { return ('9', "0".to_string()); }
    let d_val: u128 = d.parse().unwrap_or(0);
    let dv_val: u128 = dv.parse().unwrap_or(1);
    if dv_val == 0 { return ('0', d); }
    let q = d_val / dv_val;
    let r = d_val % dv_val;
    let q_char = char::from(b'0' + (q.min(9) as u8));
    (q_char, r.to_string())
}

/// 将十进制字符串拆分为 (整数部分, 小数部分)
fn split_decimal(s: &str) -> (&str, &str) {
    if let Some(dot) = s.find('.') {
        let int_part = &s[..dot];
        let frac_part = &s[dot + 1..];
        (if int_part.is_empty() { "0" } else { int_part }, frac_part)
    } else {
        (s, "")
    }
}

