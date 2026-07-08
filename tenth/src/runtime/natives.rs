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
use std::rc::Rc;

use crate::error::TenthError;
use crate::runtime::async_io::{ASYNC_IO, IoResult};
use crate::runtime::autodiff::Tape;
use crate::runtime::interpreter::datetime;
use crate::runtime::interpreter::json;
use crate::runtime::tensor::{Tensor, TensorData};
use crate::runtime::value::Value;
use crate::runtime::vm::Vm;
use crate::http::{http_get_impl, http_post_impl};

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
        let code = if let Some(Value::Int(c)) = args.first() { *c } else { 0 };
        std::process::exit(code as i32);
    });

    // —— TCP 网络原语（句柄表方案，handle 1-based，0 表示无效）——
    vm.add_native("tcp_connect".into(), |vm, args| {
        if args.len() < 2 {
            return Ok(err_result("tcp_connect 需要 (String, i64) 参数"));
        }
        if let (Value::String(host), Value::Int(port)) = (&args[0], &args[1]) {
            let addr = format!("{}:{}", host, port);
            match std::net::TcpStream::connect(&addr) {
                Ok(stream) => {
                    vm.tcp_streams.push(Some(stream));
                    let handle = vm.tcp_streams.len() as i64; // 1-based
                    Ok(ok_result(Value::Int(handle)))
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
        if let (Value::Int(handle), Value::Int(n)) = (&args[0], &args[1]) {
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
                            .map(|b| Value::Int(*b as i64))
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
        if let (Value::Int(handle), Value::Vec(data)) = (&args[0], &args[1]) {
            let idx = *handle as usize;
            if idx == 0 || idx > vm.tcp_streams.len() {
                return Ok(err_result("无效的句柄"));
            }
            let bytes: Vec<u8> = data
                .borrow()
                .iter()
                .map(|x| match x {
                    Value::Int(b) => *b as u8,
                    _ => 0,
                })
                .collect();
            if let Some(ref mut stream) = vm.tcp_streams[idx - 1] {
                use std::io::Write;
                match stream.write_all(&bytes) {
                    Ok(_) => Ok(ok_result(Value::Int(bytes.len() as i64))),
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
        if let Some(Value::Int(handle)) = args.first() {
            let idx = *handle as usize;
            if idx > 0 && idx <= vm.tcp_streams.len() {
                vm.tcp_streams[idx - 1] = None; // drop 自动关闭
            }
        }
        Ok(Value::Unit)
    });
    vm.add_native("tcp_set_timeout".into(), |vm, args| {
        if args.len() >= 2 {
            if let (Value::Int(handle), Value::Int(ms)) = (&args[0], &args[1]) {
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

    // —— 正则表达式原语（句柄表方案，handle 1-based，0 表示无效）——
    // 与 std/regex.th 对齐：Tenth 层不暴露 Regex 类型，仅用 i64 handle。
    // 与 interpreter::natives::call_named_fn 中的实现语义对齐（双侧注册）。
    vm.add_native("regex_compile".into(), |vm, args| {
        if let Some(Value::String(pattern)) = args.first() {
            match regex::Regex::new(pattern) {
                Ok(re) => {
                    vm.regexes.push(Some(re));
                    let handle = vm.regexes.len() as i64; // 1-based
                    Ok(ok_result(Value::Int(handle)))
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
        if let (Value::Int(handle), Value::String(input)) = (&args[0], &args[1]) {
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
        if let (Value::Int(handle), Value::String(input)) = (&args[0], &args[1]) {
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
        if let (Value::Int(handle), Value::String(input)) = (&args[0], &args[1]) {
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
        if let (Value::Int(handle), Value::String(input), Value::String(replacement)) =
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
        if let (Value::Int(handle), Value::String(input)) = (&args[0], &args[1]) {
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
                    Err(e) => return Err(TenthError::RuntimeError { message: e }),
                }
            } else {
                std::path::PathBuf::from(path)
            };
            match std::fs::read_to_string(&resolved) {
                Ok(s) => Ok(Value::String(s)),
                Err(e) => Err(TenthError::RuntimeError { message: format!("读取文件: {e}") }),
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
            Err(TenthError::RuntimeError { message: "tensor() 参数异常".into() })
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
                            Err(e) => return Err(TenthError::RuntimeError { message: e }),
                        }
                    } else {
                        std::path::PathBuf::from(path)
                    };
                    let bytes: Vec<u8> = items.borrow().iter().map(|v| v.as_int().unwrap_or(0) as u8).collect();
                    let _ = std::fs::write(&resolved, &bytes);
                    return Ok(Value::Int(0));
                }
            }
        }
        Ok(Value::Int(1))
    });
    vm.add_native("read_bytes".into(), |vm, args| {
        if let Some(Value::String(path)) = args.first() {
            // H-2: 沙箱校验
            let resolved = if let Some(ref sb) = vm.fs_sandbox {
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
                    Ok(Value::Vec(Rc::new(RefCell::new(bytes))))
                }
                Err(e) => Err(TenthError::RuntimeError {
                    message: format!("读取字节失败: {}", e),
                }),
            }
        } else {
            Err(TenthError::RuntimeError {
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
        if let Some(Value::Int(ms)) = args.first() {
            // 安全：拒绝负数（`as u64` 会符号扩展为巨大值，导致近乎永久的 DoS）
            // 上限 24 小时，防止 `.th` 程序意外将进程睡眠数年
            const MAX_SLEEP_MS: i64 = 24 * 60 * 60 * 1000;
            if *ms < 0 {
                return Err(TenthError::RuntimeError {
                    message: format!("time_sleep_ms: 不接受负数（{}）", ms),
                });
            }
            if *ms > MAX_SLEEP_MS {
                return Err(TenthError::RuntimeError {
                    message: format!("time_sleep_ms: 超过 24 小时上限（{}ms）", ms),
                });
            }
            std::thread::sleep(std::time::Duration::from_millis(*ms as u64));
            Ok(Value::Unit)
        } else {
            Err(TenthError::RuntimeError { message: "time_sleep_ms(ms) 期望一个整数".into() })
        }
    });

    // —— Phase 2 Step 5：异步 I/O native ——
    // 设计：std::thread + mpsc + thread_local。native 创建 Pending Future，
    // 注册到 ASYNC_IO，VM 调度器在 run_scheduler 循环中 poll。
    // await Pending Future 时 Op::Await 把当前 task 加入 waiters 并挂起，
    // I/O 就绪后 poll 把 Future 设为 Ready 并唤醒 waiters。
    vm.add_native("async_sleep_ms".into(), |_vm, args| {
        let ms = match args.first() {
            Some(Value::Int(n)) => *n,
            _ => return Err(TenthError::RuntimeError {
                message: "async_sleep_ms(ms) 期望一个整数".into(),
            }),
        };
        if ms < 0 {
            return Err(TenthError::RuntimeError {
                message: format!("async_sleep_ms: 不接受负数（{}）", ms),
            });
        }
        const MAX_SLEEP_MS: i64 = 24 * 60 * 60 * 1000;
        if ms > MAX_SLEEP_MS {
            return Err(TenthError::RuntimeError {
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
            (Value::Int(h), Value::Int(n)) => (*h, *n),
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
            (Value::Int(h), Value::Vec(v)) => (*h, v.clone()),
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
            Value::Int(b) => *b as u8,
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
            Some(Value::Int(n)) => *n,
            _ => 0,
        };
        let hi = match args.get(1) {
            Some(Value::Int(n)) => *n,
            _ => lo,
        };
        use rand::Rng;
        // 处理 lo > hi 的边界：交换而不是 (hi - lo + 1) 为负时回绕
        let (low, high) = if lo <= hi { (lo, hi) } else { (hi, lo) };
        // 用 u64 全域取模，避免 i64 范围回绕到负数
        let range = (high as u64).saturating_sub(low as u64).saturating_add(1).max(1);
        let r: u64 = rand::thread_rng().r#gen();
        Ok(Value::Int(low + ((r % range) as i64)))
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
        Ok(Value::Int(1))
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
                    Ok(bytes) => { let _ = std::fs::write(&out_resolved, &bytes); return Ok(Value::Int(0)); }
                    Err(_) => return Ok(Value::Int(1)),
                }
            }
        }
        Ok(Value::Int(1))
    });
    vm.add_native("compile_program".into(), |vm, args| {
        if args.len() >= 2 {
            if let Value::String(out) = &args[1] {
                // H-2/L-7: 沙箱校验写路径
                let out_resolved = if let Some(ref sb) = vm.fs_sandbox {
                    match sb.check_write(out) {
                        Ok(p) => p,
                        Err(e) => return Err(TenthError::RuntimeError { message: e }),
                    }
                } else {
                    std::path::PathBuf::from(out)
                };
                match crate::compile::compile_program_to_wasm(&args[0]) {
                    Ok(bytes) => { let _ = std::fs::write(&out_resolved, &bytes); return Ok(Value::Int(0)); }
                    Err(_) => return Ok(Value::Int(1)),
                }
            }
        }
        Ok(Value::Int(1))
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
            Err(TenthError::RuntimeError { message: "param() 需要一个张量参数".into() })
        }
    });
    vm.add_native("backward".into(), |vm, args| {
        if let Some(Value::Tensor(t)) = args.first() {
            if let Some(ref tape) = vm.tape {
                let loss_id = t.borrow().tape_id
                    .ok_or_else(|| TenthError::RuntimeError { message: "backward(): 张量没有 tape_id".into() })?;
                // 护城河 F：包裹 backward 错误，附加 formal_explain 根因分析
                match tape.backward(loss_id) {
                    Ok(()) => Ok(Value::Unit),
                    Err(e) => {
                        // 计算 formal_explain 根因候选
                        let causes = tape.formal_explain(loss_id, &[], &[]);
                        let explanations: Vec<String> = causes.iter().map(|c| c.explanation.clone()).collect();
                        // 存到 vm.last_explanation，供 explain_error() native 读取
                        vm.last_explanation = explanations.clone();
                        // 构造 ShapeMismatch 错误（携带 tape 上下文 + 根因消息）
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
                        Err(TenthError::ShapeMismatch {
                            context,
                            message: root_cause_msg,
                        })
                    }
                }
            } else {
                Err(TenthError::RuntimeError { message: "未调用 new_grad()".into() })
            }
        } else {
            Err(TenthError::RuntimeError { message: "backward() 需要一个张量参数".into() })
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
            Err(TenthError::RuntimeError { message: "grad() 需要一个张量参数".into() })
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
        if vm.recording {
            if let Some(ref mut tape) = vm.tape {
                let base_id = base.borrow().tape_id;
                let node_id = tape.gather(base_id, base.clone(), index.clone(), result.clone(), dim);
                result.borrow_mut().tape_id = Some(node_id);
            }
        }
        Ok(Value::Tensor(result))
    });
    vm.add_native("cross_entropy".into(), |vm, args| {
        if args.len() < 2 {
            return Err(TenthError::RuntimeError { message: "cross_entropy(logits, target) 期望两个张量".into() });
        }
        if let (Value::Tensor(logits), Value::Tensor(target)) = (&args[0], &args[1]) {
            let logits_data = logits.borrow();
            let target_data = target.borrow();
            let is_f32 = logits_data.is_f32();
            let sm = logits_data.softmax().ok_or_else(|| {
                TenthError::RuntimeError { message: "cross_entropy 中 softmax 失败".into() }
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
            Err(TenthError::RuntimeError { message: "cross_entropy(logits, target) 期望两个张量".into() })
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
                        Err(e) => return Err(TenthError::RuntimeError { message: e }),
                    }
                } else {
                    std::path::PathBuf::from(path)
                };
                match std::fs::write(&resolved, content) {
                    Ok(()) => Ok(Value::Unit),
                    Err(e) => Err(TenthError::RuntimeError { message: format!("写入文件失败: {}", e) }),
                }
            } else {
                Err(TenthError::RuntimeError { message: "write_file(路径, 内容) 期望两个字符串参数".into() })
            }
        } else {
            Err(TenthError::RuntimeError { message: "write_file(路径, 内容) 期望两个字符串参数".into() })
        }
    });
    vm.add_native("path_join".into(), |_vm, args| {
        if args.len() >= 2 {
            if let (Value::String(base), Value::String(rest)) = (&args[0], &args[1]) {
                let joined = std::path::Path::new(base).join(rest);
                Ok(Value::String(joined.to_string_lossy().to_string()))
            } else {
                Err(TenthError::RuntimeError { message: "path_join(基础路径, 子路径) 期望两个字符串参数".into() })
            }
        } else {
            Err(TenthError::RuntimeError { message: "path_join(基础路径, 子路径) 期望两个字符串参数".into() })
        }
    });
    vm.add_native("path_exists".into(), |vm, args| {
        if let Some(Value::String(path)) = args.first() {
            // H-2: 沙箱校验
            if let Some(ref sb) = vm.fs_sandbox {
                if let Err(e) = sb.check_read(path) {
                    return Err(TenthError::RuntimeError { message: e });
                }
            }
            Ok(Value::Bool(std::path::Path::new(path).exists()))
        } else {
            Err(TenthError::RuntimeError { message: "path_exists(路径) 期望一个字符串路径".into() })
        }
    });
    vm.add_native("path_is_file".into(), |vm, args| {
        if let Some(Value::String(path)) = args.first() {
            if let Some(ref sb) = vm.fs_sandbox {
                if let Err(e) = sb.check_read(path) {
                    return Err(TenthError::RuntimeError { message: e });
                }
            }
            Ok(Value::Bool(std::path::Path::new(path).is_file()))
        } else {
            Err(TenthError::RuntimeError { message: "path_is_file(路径) 期望一个字符串路径".into() })
        }
    });
    vm.add_native("path_is_dir".into(), |vm, args| {
        if let Some(Value::String(path)) = args.first() {
            if let Some(ref sb) = vm.fs_sandbox {
                if let Err(e) = sb.check_read(path) {
                    return Err(TenthError::RuntimeError { message: e });
                }
            }
            Ok(Value::Bool(std::path::Path::new(path).is_dir()))
        } else {
            Err(TenthError::RuntimeError { message: "path_is_dir(路径) 期望一个字符串路径".into() })
        }
    });
    vm.add_native("mkdir".into(), |vm, args| {
        if let Some(Value::String(path)) = args.first() {
            // H-2: 沙箱校验
            let resolved = if let Some(ref sb) = vm.fs_sandbox {
                match sb.check_write(path) {
                    Ok(p) => p,
                    Err(e) => return Err(TenthError::RuntimeError { message: e }),
                }
            } else {
                std::path::PathBuf::from(path)
            };
            match std::fs::create_dir_all(&resolved) {
                Ok(()) => Ok(Value::Unit),
                Err(e) => Err(TenthError::RuntimeError { message: format!("创建目录失败: {}", e) }),
            }
        } else {
            Err(TenthError::RuntimeError { message: "mkdir(路径) 期望一个字符串路径".into() })
        }
    });
    vm.add_native("list_dir".into(), |vm, args| {
        if let Some(Value::String(path)) = args.first() {
            // H-2: 沙箱校验
            let resolved = if let Some(ref sb) = vm.fs_sandbox {
                match sb.check_read(path) {
                    Ok(p) => p,
                    Err(e) => return Err(TenthError::RuntimeError { message: e }),
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
                Err(e) => Err(TenthError::RuntimeError { message: format!("列出目录失败: {}", e) }),
            }
        } else {
            Err(TenthError::RuntimeError { message: "list_dir(路径) 期望一个字符串路径".into() })
        }
    });
    vm.add_native("file_size".into(), |vm, args| {
        if let Some(Value::String(path)) = args.first() {
            // H-2: 沙箱校验
            let resolved = if let Some(ref sb) = vm.fs_sandbox {
                match sb.check_read(path) {
                    Ok(p) => p,
                    Err(e) => return Err(TenthError::RuntimeError { message: e }),
                }
            } else {
                std::path::PathBuf::from(path)
            };
            match std::fs::metadata(&resolved) {
                Ok(meta) => Ok(Value::Int(meta.len() as i64)),
                Err(e) => Err(TenthError::RuntimeError { message: format!("获取文件大小失败: {}", e) }),
            }
        } else {
            Err(TenthError::RuntimeError { message: "file_size(路径) 期望一个字符串路径".into() })
        }
    });
    vm.add_native("remove_file".into(), |vm, args| {
        if let Some(Value::String(path)) = args.first() {
            // H-2: 沙箱校验
            let resolved = if let Some(ref sb) = vm.fs_sandbox {
                match sb.check_write(path) {
                    Ok(p) => p,
                    Err(e) => return Err(TenthError::RuntimeError { message: e }),
                }
            } else {
                std::path::PathBuf::from(path)
            };
            match std::fs::remove_file(&resolved) {
                Ok(()) => Ok(Value::Unit),
                Err(e) => Err(TenthError::RuntimeError { message: format!("删除文件失败: {}", e) }),
            }
        } else {
            Err(TenthError::RuntimeError { message: "remove_file(路径) 期望一个字符串路径".into() })
        }
    });
    vm.add_native("copy_file".into(), |vm, args| {
        if args.len() >= 2 {
            if let (Value::String(src), Value::String(dst)) = (&args[0], &args[1]) {
                // H-2: 沙箱校验
                let src_resolved = if let Some(ref sb) = vm.fs_sandbox {
                    match sb.check_read(src) {
                        Ok(p) => p,
                        Err(e) => return Err(TenthError::RuntimeError { message: e }),
                    }
                } else {
                    std::path::PathBuf::from(src)
                };
                let dst_resolved = if let Some(ref sb) = vm.fs_sandbox {
                    match sb.check_write(dst) {
                        Ok(p) => p,
                        Err(e) => return Err(TenthError::RuntimeError { message: e }),
                    }
                } else {
                    std::path::PathBuf::from(dst)
                };
                match std::fs::copy(&src_resolved, &dst_resolved) {
                    Ok(_) => Ok(Value::Unit),
                    Err(e) => Err(TenthError::RuntimeError { message: format!("复制文件失败: {}", e) }),
                }
            } else {
                Err(TenthError::RuntimeError { message: "copy_file(源路径, 目标路径) 期望两个字符串参数".into() })
            }
        } else {
            Err(TenthError::RuntimeError { message: "copy_file(源路径, 目标路径) 期望两个字符串参数".into() })
        }
    });
    vm.add_native("rename_file".into(), |vm, args| {
        if args.len() >= 2 {
            if let (Value::String(src), Value::String(dst)) = (&args[0], &args[1]) {
                // H-2: 沙箱校验
                let src_resolved = if let Some(ref sb) = vm.fs_sandbox {
                    match sb.check_read(src) {
                        Ok(p) => p,
                        Err(e) => return Err(TenthError::RuntimeError { message: e }),
                    }
                } else {
                    std::path::PathBuf::from(src)
                };
                let dst_resolved = if let Some(ref sb) = vm.fs_sandbox {
                    match sb.check_write(dst) {
                        Ok(p) => p,
                        Err(e) => return Err(TenthError::RuntimeError { message: e }),
                    }
                } else {
                    std::path::PathBuf::from(dst)
                };
                match std::fs::rename(&src_resolved, &dst_resolved) {
                    Ok(()) => Ok(Value::Unit),
                    Err(e) => Err(TenthError::RuntimeError { message: format!("重命名文件失败: {}", e) }),
                }
            } else {
                Err(TenthError::RuntimeError { message: "rename_file(源路径, 目标路径) 期望两个字符串参数".into() })
            }
        } else {
            Err(TenthError::RuntimeError { message: "rename_file(源路径, 目标路径) 期望两个字符串参数".into() })
        }
    });
    vm.add_native("randn".into(), |_vm, args| {
        let rows = match args.first() { Some(Value::Int(n)) => *n as usize, _ => 1 };
        let cols = match args.get(1) { Some(Value::Int(n)) => *n as usize, _ => 1 };
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
        let rows = match args.first() { Some(Value::Int(n)) => *n as usize, _ => 1 };
        let cols = match args.get(1) { Some(Value::Int(n)) => *n as usize, _ => 1 };
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
            Some(Value::Int(n)) => Ok(Value::Int(n.abs())),
            Some(Value::Float(f)) => Ok(Value::Float(f.abs())),
            Some(Value::Float32(f)) => Ok(Value::Float32(f.abs())),
            _ => Err(TenthError::RuntimeError { message: "abs() 需要一个数值参数".into() }),
        }
    });
    vm.add_native("sqrt".into(), |_vm, args| {
        match args.first() {
            Some(Value::Float(f)) => Ok(Value::Float(f.sqrt())),
            Some(Value::Float32(f)) => Ok(Value::Float32(f.sqrt())),
            Some(Value::Int(n)) => Ok(Value::Float((*n as f64).sqrt())),
            _ => Err(TenthError::RuntimeError { message: "sqrt() 需要一个数值参数".into() }),
        }
    });
    vm.add_native("to_float".into(), |_vm, args| {
        match args.first() {
            Some(Value::Int(n)) => Ok(Value::Float(*n as f64)),
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
                    return Err(TenthError::RuntimeError {
                        message: format!("to_float() 不接受多元素 Tensor (shape={:?})", shape),
                    });
                };
                scalar.map(Value::Float).ok_or_else(|| TenthError::RuntimeError {
                    message: "to_float() Tensor 标量提取失败".into(),
                })
            }
            _ => Err(TenthError::RuntimeError { message: "to_float() 需要一个数值参数".into() }),
        }
    });
    vm.add_native("to_f64".into(), |_vm, args| {
        match args.first() {
            Some(Value::Int(n)) => Ok(Value::Float(*n as f64)),
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
                    return Err(TenthError::RuntimeError {
                        message: format!("to_f64() 不接受多元素 Tensor (shape={:?})", shape),
                    });
                };
                scalar.map(Value::Float).ok_or_else(|| TenthError::RuntimeError {
                    message: "to_f64() Tensor 标量提取失败".into(),
                })
            }
            _ => Err(TenthError::RuntimeError { message: "to_f64() 需要一个数值参数".into() }),
        }
    });
    vm.add_native("to_f32".into(), |_vm, args| {
        match args.first() {
            Some(Value::Int(n)) => Ok(Value::Float32(*n as f32)),
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
                    return Err(TenthError::RuntimeError {
                        message: format!("to_f32() 不接受多元素 Tensor (shape={:?})", shape),
                    });
                };
                scalar.map(|v| Value::Float32(v as f32)).ok_or_else(|| TenthError::RuntimeError {
                    message: "to_f32() Tensor 标量提取失败".into(),
                })
            }
            _ => Err(TenthError::RuntimeError { message: "to_f32() 需要一个数值参数".into() }),
        }
    });
    vm.add_native("tensor_from_vec".into(), |_vm, args| {
        if args.len() >= 3 {
            if let (Value::Vec(items), Value::Int(rows), Value::Int(cols)) = (&args[0], &args[1], &args[2]) {
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
                Err(TenthError::RuntimeError { message: "tensor_from_vec(vec, rows, cols) 期望一个 Vec 和两个整数".into() })
            }
        } else {
            Err(TenthError::RuntimeError { message: "tensor_from_vec(vec, rows, cols) 期望 3 个参数".into() })
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
                Value::Int(_) => "int",
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
            return Err(TenthError::RuntimeError {
                message: "with_step_limit(limit, fn) 需要 2 个参数".into(),
            });
        }
        let limit = args[0].as_int().ok_or_else(|| TenthError::RuntimeError {
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
            return Err(TenthError::RuntimeError {
                message: "with_timeout_ms(ms, fn) 需要 2 个参数".into(),
            });
        }
        let ms = args[0].as_int().ok_or_else(|| TenthError::RuntimeError {
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
            let f = arg.as_float().ok_or_else(|| TenthError::RuntimeError {
                message: "f64_bits() 期望一个 f64 参数".into(),
            })?;
            Ok(Value::Int(f.to_bits() as i64))
        } else {
            Err(TenthError::RuntimeError { message: "f64_bits() 期望 1 个参数".into() })
        }
    });
    // 8. f64_from_bits — i64 → f64
    vm.add_native("f64_from_bits".into(), |_vm, args| {
        if let Some(arg) = args.first() {
            let n = arg.as_int().ok_or_else(|| TenthError::RuntimeError {
                message: "f64_from_bits() 期望一个 i64 参数".into(),
            })?;
            Ok(Value::Float(f64::from_bits(n as u64)))
        } else {
            Err(TenthError::RuntimeError { message: "f64_from_bits() 期望 1 个参数".into() })
        }
    });
    // 9-12. 标量数学（sin/cos/ln/pow）— 与解释器一致，仅操作 Float（as_float 自动提升 Int/Float32）
    vm.add_native("sin".into(), |_vm, args| {
        if let Some(arg) = args.first() {
            let n = arg.as_float().ok_or_else(|| TenthError::RuntimeError {
                message: "sin() 期望一个数值参数".into(),
            })?;
            Ok(Value::Float(n.sin()))
        } else {
            Err(TenthError::RuntimeError { message: "sin() 期望 1 个参数".into() })
        }
    });
    vm.add_native("cos".into(), |_vm, args| {
        if let Some(arg) = args.first() {
            let n = arg.as_float().ok_or_else(|| TenthError::RuntimeError {
                message: "cos() 期望一个数值参数".into(),
            })?;
            Ok(Value::Float(n.cos()))
        } else {
            Err(TenthError::RuntimeError { message: "cos() 期望 1 个参数".into() })
        }
    });
    vm.add_native("ln".into(), |_vm, args| {
        if let Some(arg) = args.first() {
            let n = arg.as_float().ok_or_else(|| TenthError::RuntimeError {
                message: "ln() 期望一个数值参数".into(),
            })?;
            if n <= 0.0 {
                return Err(TenthError::RuntimeError {
                    message: "ln() 参数必须 > 0".into(),
                });
            }
            Ok(Value::Float(n.ln()))
        } else {
            Err(TenthError::RuntimeError { message: "ln() 期望 1 个参数".into() })
        }
    });
    vm.add_native("pow".into(), |_vm, args| {
        if args.len() >= 2 {
            let base = args[0].as_float().ok_or_else(|| TenthError::RuntimeError {
                message: "pow() 期望数值参数".into(),
            })?;
            let exp = args[1].as_float().ok_or_else(|| TenthError::RuntimeError {
                message: "pow() 期望数值参数".into(),
            })?;
            Ok(Value::Float(base.powf(exp)))
        } else {
            Err(TenthError::RuntimeError { message: "pow() 期望 2 个参数".into() })
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
                        Err(e) => return Err(TenthError::RuntimeError { message: e }),
                    }
                } else {
                    std::path::PathBuf::from(path)
                };
                let tensors: &Rc<RefCell<Vec<Value>>> = match &args[1] {
                    Value::Vec(v) => v,
                    Value::Array(a) => a,
                    _ => return Err(TenthError::RuntimeError {
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
        Err(TenthError::RuntimeError {
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
                                    return Err(TenthError::RuntimeError {
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
                Err(e) => Err(TenthError::RuntimeError {
                    message: format!("load_weights: {}", e),
                }),
            }
        } else {
            Err(TenthError::RuntimeError {
                message: "load_weights(路径)".into(),
            })
        }
    });
    // 15. format(template, args...) — 模板字符串格式化（{}/{{/}}）
    vm.add_native("format".into(), |_vm, args| {
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
            Ok(Value::String(result))
        } else {
            Err(TenthError::RuntimeError {
                message: "format() 第一个参数必须是字符串模板".into(),
            })
        }
    });
    // 16. parse_int(s) — 字符串→整数（解析失败返回 0，与解释器一致）
    vm.add_native("parse_int".into(), |_vm, args| {
        if let Some(Value::String(s)) = args.first() {
            Ok(Value::Int(s.trim().parse::<i64>().unwrap_or(0)))
        } else {
            Err(TenthError::RuntimeError {
                message: "parse_int() 期望一个字符串参数".into(),
            })
        }
    });
    // 17. parse_float(s) — 字符串→浮点（解析失败返回 0.0，与解释器一致）
    vm.add_native("parse_float".into(), |_vm, args| {
        if let Some(Value::String(s)) = args.first() {
            Ok(Value::Float(s.trim().parse::<f64>().unwrap_or(0.0)))
        } else {
            Err(TenthError::RuntimeError {
                message: "parse_float() 期望一个字符串参数".into(),
            })
        }
    });
}
