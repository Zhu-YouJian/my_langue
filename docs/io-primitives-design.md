# 同步 I/O 原语完整设计方案

> 版本：v1.0 | 2026-07-06 | 状态：设计阶段
> 目标：补齐 Tenth 语言的 I/O 能力缺口，采用同步阻塞模型，为未来异步 I/O 铺路

---

## 一、背景与动机

### 1.1 现状

Tenth 已有较完整的**文件系统 I/O**（17 个 native：read_file/write_file/read_bytes/write_bytes/path_*/mkdir/list_dir/file_size/remove_file/copy_file/rename_file），但存在以下缺口：

| 缺口 | 影响 |
|------|------|
| stderr（eprint/eprintln） | 无法输出诊断信息到 stderr；`prelude.th:10` 误标为可用 |
| stdin（read_line） | 无法读取用户输入，无 REPL 交互能力 |
| 环境变量 | 无法读取配置（如 HOME、PATH） |
| 进程控制（exit） | 无法主动退出，只能靠 main 返回 |
| 网络（TCP/HTTP） | 无法做任何网络通信 |

### 1.2 设计目标

1. **同步阻塞模型**：所有 I/O 操作阻塞调用线程，简单可靠
2. **零新依赖**：仅用 Rust 标准库（`std::io`/`std::net`），不引入 tokio/reqwest
3. **错误可捕获**：I/O 函数返回 `Result<T>` 而非 panic，让 Tenth 程序能 `try { }` 捕获
4. **双侧对齐**：VM（`main.rs::register_natives`）+ 解释器（`interpreter/natives.rs`）语义一致
5. **沙箱安全**：文件 I/O 经 `FsSandbox` 校验；网络 I/O 不受沙箱限制（但有超时保护）
6. **标准库封装**：新增 `std::io`/`std::net`/`std::http` 模块，提供高层 API

### 1.3 非目标

- **异步 I/O**：不涉及（需 Phase 2 协程调度 + Phase 3 状态机）
- **UDP/原始套接字**：不涉及（TCP 足够覆盖绝大多数场景）
- **TLS/HTTPS**：不涉及（需要加密库依赖，留作后续）
- **并发服务器**：不涉及（需 Phase 2 协程调度）

---

## 二、架构总览

```
┌─────────────────────────────────────────────────────────────┐
│ Tenth 程序                                                   │
│   use std::io::io     →  read_line / eprint / eprintln      │
│   use std::env::env   →  getenv / setenv / exit             │
│   use std::net::net   →  TcpStream / tcp_connect / ...      │
│   use std::http::http →  http_get / http_post               │
└─────────┬───────────────────────────────────────────────────┘
          │ 调用
          ▼
┌─────────────────────────────────────────────────────────────┐
│ Native 函数层（main.rs::register_natives + natives.rs）      │
│   read_line / eprint / eprintln                             │
│   env_get / env_set / exit                                  │
│   tcp_connect / tcp_read / tcp_write / tcp_close            │
│   http_get / http_post                                      │
│   （所有 I/O 返回 Result<T>）                                │
└─────────┬───────────────────────────────────────────────────┘
          │ 调用
          ▼
┌─────────────────────────────────────────────────────────────┐
│ Rust 标准库                                                   │
│   std::io::stdin / std::io::stderr                          │
│   std::env::var / std::env::set_var / std::process::exit   │
│   std::net::TcpStream / std::io::Read/Write                 │
└─────────────────────────────────────────────────────────────┘
```

---

## 三、错误处理策略

### 3.1 当前问题

现有 I/O native 返回 `Err(TenthError::RuntimeError)`，会**冒泡到顶层导致程序终止**，Tenth 程序无法 `try { }` 捕获。

### 3.2 新策略：返回 Result<T>

**所有新增 I/O native 返回 `Value::Enum { enum_name: "Result", ... }`**，让 Tenth 程序能用 `?` 操作符或 `try { }` 块捕获错误。

```tenth
// 用法示例
try {
    let content = read_file("config.th")?;
    print(content)
} match {
    Result::Ok(_) => print("成功"),
    Result::Err(e) => print("失败: " + e),
}
```

### 3.3 Result 构造辅助函数

在 native 实现中用辅助函数构造 Result：

```rust
fn ok_result(val: Value) -> Value {
    Value::Enum {
        enum_name: "Result".to_string(),
        variant: "Ok".to_string(),
        fields: Rc::new(RefCell::new(vec![("_0".to_string(), val)])),
    }
}

fn err_result(msg: &str) -> Value {
    Value::Enum {
        enum_name: "Result".to_string(),
        variant: "Err".to_string(),
        fields: Rc::new(RefCell::new(vec![("_0".to_string(), Value::String(msg.to_string()))])),
    }
}
```

### 3.4 现有 I/O 函数的处理

**不改动现有 17 个文件 I/O native**（避免破坏向后兼容），仅新增的 I/O 函数采用 Result 返回。未来可逐步迁移。

---

## 四、模块设计

### 4.1 模块 1：stderr + stdin（std::io）

#### 4.1.1 Native 函数

| 函数名 | 签名 | 说明 |
|--------|------|------|
| `eprint` | `(value) -> ()` | 输出到 stderr，不换行 |
| `eprintln` | `(value) -> ()` | 输出到 stderr，换行 |
| `read_line` | `() -> Result<String>` | 从 stdin 读一行，返回 `Result<String>`（EOF 时返回 Err） |

#### 4.1.2 实现要点

```rust
// eprint
vm.add_native("eprint".into(), |_vm, args| {
    use std::io::Write;
    let stderr = std::io::stderr();
    let mut handle = stderr.lock();
    for a in args { write!(handle, "{a}").ok(); }
    Ok(Value::Unit)
});

// eprintln
vm.add_native("eprintln".into(), |_vm, args| {
    use std::io::Write;
    let stderr = std::io::stderr();
    let mut handle = stderr.lock();
    for a in args { write!(handle, "{a}").ok(); }
    writeln!(handle).ok();
    Ok(Value::Unit)
});

// read_line
vm.add_native("read_line".into(), |_vm, _args| {
    use std::io::BufRead;
    let stdin = std::io::stdin();
    let mut line = String::new();
    match stdin.lock().read_line(&mut line) {
        Ok(0) => Ok(ok_result(Value::String(String::new()))),  // EOF
        Ok(_) => {
            // 去掉末尾换行符
            if line.ends_with('\n') { line.pop(); if line.ends_with('\r') { line.pop(); } }
            Ok(ok_result(Value::String(line)))
        }
        Err(e) => Ok(err_result(&format!("读取输入失败: {e}"))),
    }
});
```

#### 4.1.3 标准库封装 `std/io.th`

```tenth
// Standard I/O utilities
//
// Usage:
//   use std::io::io

/// 输出到 stderr（不换行）
fn eprint(value: String) {
    eprint(value)
}

/// 输出到 stderr（换行）
fn eprintln(value: String) {
    eprintln(value)
}

/// 从 stdin 读取一行，返回 Result<String>
fn read_line() -> String {
    // 标准库封装层简化错误处理：失败时返回空字符串
    try {
        read_line()?
    } match {
        Result::Ok(s) => s,
        Result::Err(_) => "",
    }
}
```

---

### 4.2 模块 2：环境变量 + 进程控制（std::env）

#### 4.2.1 Native 函数

| 函数名 | 签名 | 说明 |
|--------|------|------|
| `env_get` | `(name: String) -> Result<String>` | 读取环境变量，不存在时返回 Err |
| `env_set` | `(name: String, value: String) -> ()` | 设置环境变量 |
| `exit` | `(code: i64) -> ()` | 退出进程（code=0 正常，非 0 异常） |

#### 4.2.2 实现要点

```rust
// env_get
vm.add_native("env_get".into(), |_vm, args| {
    if let Some(Value::String(name)) = args.first() {
        match std::env::var(name) {
            Ok(val) => Ok(ok_result(Value::String(val))),
            Err(_) => Ok(err_result(&format!("环境变量不存在: {name}"))),
        }
    } else {
        Ok(err_result("env_get 需要 1 个字符串参数"))
    }
});

// env_set
vm.add_native("env_set".into(), |_vm, args| {
    if args.len() >= 2 {
        if let (Value::String(name), Value::String(val)) = (&args[0], &args[1]) {
            std::env::set_var(name, val);
        }
    }
    Ok(Value::Unit)
});

// exit
vm.add_native("exit".into(), |_vm, args| {
    let code = if let Some(Value::Int(c)) = args.first() { *c } else { 0 };
    std::process::exit(code as i32);
});
```

#### 4.2.3 标准库封装 `std/env.th`

```tenth
// Environment and process control
//
// Usage:
//   use std::env::env

/// 读取环境变量，不存在时返回默认值
fn get(name: String, default: String) -> String {
    try {
        env_get(name)?
    } match {
        Result::Ok(v) => v,
        Result::Err(_) => default,
    }
}

/// 设置环境变量
fn set(name: String, value: String) {
    env_set(name, value)
}

/// 退出进程
fn exit(code: i64) {
    exit(code)
}
```

---

### 4.3 模块 3：TCP 网络原语（std::net）

#### 4.3.1 设计挑战

TCP socket 是**有状态资源**，需要在 VM 中存储 `TcpStream` 对象。当前 `Value` 枚举没有 socket 类型。

#### 4.3.2 方案：句柄表（Handle Table）

不引入新的 `Value` 变体（避免 HIR 数据结构变更），而是用 **句柄表**：

- VM 新增字段：`tcp_streams: Vec<Option<TcpStream>>`
- `tcp_connect` 返回 `Value::Int(handle)`（句柄 = 索引）
- `tcp_read`/`tcp_write`/`tcp_close` 接收句柄，查表操作

#### 4.3.3 Native 函数

| 函数名 | 签名 | 说明 |
|--------|------|------|
| `tcp_connect` | `(host: String, port: i64, timeout_ms: i64) -> Result<i64>` | 连接 TCP，返回句柄 |
| `tcp_read` | `(handle: i64, max_bytes: i64) -> Result<Vec<i64>>` | 读取数据，返回字节列表 |
| `tcp_write` | `(handle: i64, data: Vec<i64>) -> Result<i64>` | 写入数据，返回写入字节数 |
| `tcp_close` | `(handle: i64) -> ()` | 关闭连接 |
| `tcp_set_timeout` | `(handle: i64, read_ms: i64, write_ms: i64) -> ()` | 设置读写超时 |

#### 4.3.4 VM 结构变更

```rust
// vm.rs - Vm struct 新增字段
pub struct Vm {
    // ... 现有字段 ...
    /// TCP 流句柄表。索引即句柄。None 表示已关闭或未使用。
    pub tcp_streams: Vec<Option<std::net::TcpStream>>,
}
```

#### 4.3.5 实现要点

```rust
// tcp_connect
vm.add_native("tcp_connect".into(), |vm, args| {
    if args.len() < 3 {
        return Ok(err_result("tcp_connect 需要 3 个参数: host, port, timeout_ms"));
    }
    let (host, port, timeout_ms) = match (&args[0], &args[1], &args[2]) {
        (Value::String(h), Value::Int(p), Value::Int(t)) => (h, *p, *t),
        _ => return Ok(err_result("参数类型错误")),
    };
    let addr = format!("{host}:{port}");
    let timeout = std::time::Duration::from_millis(timeout_ms as u64);
    match std::net::TcpStream::connect_timeout(&addr.parse().unwrap_or("0.0.0.0:0".parse().unwrap()), timeout) {
        Ok(stream) => {
            let handle = vm.tcp_streams.len() as i64;
            vm.tcp_streams.push(Some(stream));
            Ok(ok_result(Value::Int(handle)))
        }
        Err(e) => Ok(err_result(&format!("连接失败: {e}"))),
    }
});

// tcp_read
vm.add_native("tcp_read".into(), |vm, args| {
    if args.len() < 2 {
        return Ok(err_result("tcp_read 需要 2 个参数: handle, max_bytes"));
    }
    let (handle, max_bytes) = match (&args[0], &args[1]) {
        (Value::Int(h), Value::Int(m)) => (*h, *m),
        _ => return Ok(err_result("参数类型错误")),
    };
    let idx = handle as usize;
    if idx >= vm.tcp_streams.len() {
        return Ok(err_result("无效的句柄"));
    }
    if let Some(ref mut stream) = vm.tcp_streams[idx] {
        use std::io::Read;
        let mut buf = vec![0u8; max_bytes.min(65536) as usize];
        match stream.read(&mut buf) {
            Ok(n) => {
                let bytes: Vec<Value> = buf[..n].iter().map(|b| Value::Int(*b as i64)).collect();
                Ok(ok_result(Value::Vec(Rc::new(RefCell::new(bytes)))))
            }
            Err(e) => Ok(err_result(&format!("读取失败: {e}"))),
        }
    } else {
        Ok(err_result("连接已关闭"))
    }
});

// tcp_write
vm.add_native("tcp_write".into(), |vm, args| {
    if args.len() < 2 {
        return Ok(err_result("tcp_write 需要 2 个参数: handle, data"));
    }
    let handle = match &args[0] { Value::Int(h) => *h, _ => return Ok(err_result("参数类型错误")) };
    let data = match &args[1] {
        Value::Vec(v) => v.borrow().iter().map(|x| match x { Value::Int(b) => *b as u8, _ => 0 }).collect::<Vec<u8>>(),
        Value::String(s) => s.as_bytes().to_vec(),
        _ => return Ok(err_result("data 参数必须是 Vec 或 String")),
    };
    let idx = handle as usize;
    if idx >= vm.tcp_streams.len() {
        return Ok(err_result("无效的句柄"));
    }
    if let Some(ref mut stream) = vm.tcp_streams[idx] {
        use std::io::Write;
        match stream.write_all(&data) {
            Ok(_) => Ok(ok_result(Value::Int(data.len() as i64))),
            Err(e) => Ok(err_result(&format!("写入失败: {e}"))),
        }
    } else {
        Ok(err_result("连接已关闭"))
    }
});

// tcp_close
vm.add_native("tcp_close".into(), |vm, args| {
    if let Some(Value::Int(handle)) = args.first() {
        let idx = *handle as usize;
        if idx < vm.tcp_streams.len() {
            vm.tcp_streams[idx] = None;  // drop 自动关闭
        }
    }
    Ok(Value::Unit)
});
```

#### 4.3.6 标准库封装 `std/net.th`

```tenth
// TCP network utilities
//
// Usage:
//   use std::net::net

/// 连接到 TCP 服务器，返回句柄
fn connect(host: String, port: i64) -> i64 {
    try {
        tcp_connect(host, port, 5000)?
    } match {
        Result::Ok(h) => h,
        Result::Err(e) => { eprintln("连接失败: " + e); -1 },
    }
}

/// 发送数据
fn send(handle: i64, data: String) -> i64 {
    try {
        // 将 String 转为 Vec<i64> 后发送
        let bytes = string_to_bytes(data);
        tcp_write(handle, bytes)?
    } match {
        Result::Ok(n) => n,
        Result::Err(_) => -1,
    }
}

/// 接收数据
fn recv(handle: i64, max_bytes: i64) -> String {
    try {
        let bytes = tcp_read(handle, max_bytes)?;
        bytes_to_string(bytes)
    } match {
        Result::Ok(s) => s,
        Result::Err(_) => "",
    }
}

/// 关闭连接
fn close(handle: i64) {
    tcp_close(handle)
}
```

---

### 4.4 模块 4：HTTP 客户端（std::http）

#### 4.4.1 设计思路

基于 TCP 原语**手写 HTTP/1.1**，不引入 reqwest/hyper 依赖。支持 GET/POST 方法，足够覆盖大多数 API 调用场景。

#### 4.4.2 Native 函数

| 函数名 | 签名 | 说明 |
|--------|------|------|
| `http_get` | `(url: String, timeout_ms: i64) -> Result<String>` | GET 请求，返回响应体 |
| `http_post` | `(url: String, body: String, content_type: String, timeout_ms: i64) -> Result<String>` | POST 请求 |

#### 4.4.3 实现要点

```rust
// http_get
vm.add_native("http_get".into(), |_vm, args| {
    if args.len() < 2 {
        return Ok(err_result("http_get 需要 2 个参数: url, timeout_ms"));
    }
    let (url, timeout_ms) = match (&args[0], &args[1]) {
        (Value::String(u), Value::Int(t)) => (u, *t),
        _ => return Ok(err_result("参数类型错误")),
    };
    match http_request("GET", url, "", "text/plain", timeout_ms) {
        Ok(body) => Ok(ok_result(Value::String(body))),
        Err(e) => Ok(err_result(&e)),
    }
});

// http_post
vm.add_native("http_post".into(), |_vm, args| {
    if args.len() < 4 {
        return Ok(err_result("http_post 需要 4 个参数: url, body, content_type, timeout_ms"));
    }
    let (url, body, content_type, timeout_ms) = match (&args[0], &args[1], &args[2], &args[3]) {
        (Value::String(u), Value::String(b), Value::String(c), Value::Int(t)) => (u, b, c, *t),
        _ => return Ok(err_result("参数类型错误")),
    };
    match http_request("POST", url, body, content_type, timeout_ms) {
        Ok(response_body) => Ok(ok_result(Value::String(response_body))),
        Err(e) => Ok(err_result(&e)),
    }
});

/// 纯 Rust HTTP/1.1 请求实现
fn http_request(method: &str, url: &str, body: &str, content_type: &str, timeout_ms: i64) -> Result<String, String> {
    // 解析 URL: http://host:port/path
    let (host, port, path) = parse_url(url)?;
    let timeout = std::time::Duration::from_millis(timeout_ms as u64);
    let addr = format!("{host}:{port}");
    let socket_addr: std::net::SocketAddr = addr.parse().map_err(|e: std::net::AddrParseError| format!("地址解析失败: {e}"))?;
    let mut stream = std::net::TcpStream::connect_timeout(&socket_addr, timeout)
        .map_err(|e| format!("连接失败: {e}"))?;
    stream.set_read_timeout(Some(timeout)).ok();
    stream.set_write_timeout(Some(timeout)).ok();

    // 构造 HTTP 请求
    let request = format!(
        "{method} {path} HTTP/1.1\r\nHost: {host}\r\nContent-Type: {content_type}\r\nContent-Length: {len}\r\nConnection: close\r\n\r\n{body}",
        len = body.len()
    );
    use std::io::Write;
    stream.write_all(request.as_bytes()).map_err(|e| format!("写入失败: {e}"))?;

    // 读取响应
    use std::io::Read;
    let mut response = String::new();
    stream.read_to_string(&mut response).map_err(|e| format!("读取失败: {e}"))?;

    // 分离响应头和响应体
    let body_start = response.find("\r\n\r\n")
        .map(|i| i + 4)
        .unwrap_or(response.len());
    Ok(response[body_start..].to_string())
}

fn parse_url(url: &str) -> Result<(String, u16, String), String> {
    let url = url.strip_prefix("http://").ok_or("仅支持 http:// URL")?;
    let (host_port, path) = match url.find('/') {
        Some(i) => (&url[..i], &url[i..]),
        None => (url, "/"),
    };
    let (host, port) = match host_port.find(':') {
        Some(i) => {
            let h = &host_port[..i];
            let p: u16 = host_port[i+1..].parse().map_err(|_| "端口号无效")?;
            (h.to_string(), p)
        }
        None => (host_port.to_string(), 80),
    };
    Ok((host, port, path.to_string()))
}
```

#### 4.4.4 标准库封装 `std/http.th`

```tenth
// HTTP client utilities
//
// Usage:
//   use std::http::http

/// 发送 GET 请求
fn get(url: String) -> String {
    try {
        http_get(url, 10000)?
    } match {
        Result::Ok(body) => body,
        Result::Err(e) => { eprintln("HTTP GET 失败: " + e); "" },
    }
}

/// 发送 POST 请求
fn post(url: String, body: String) -> String {
    try {
        http_post(url, body, "application/json", 10000)?
    } match {
        Result::Ok(body) => body,
        Result::Err(e) => { eprintln("HTTP POST 失败: " + e); "" },
    }
}
```

---

## 五、文件变更清单

### 5.1 Rust 源码（5 个文件）

| 文件 | 改动 |
|------|------|
| `tenth/src/runtime/vm.rs` | Vm 结构新增 `tcp_streams: Vec<Option<TcpStream>>` 字段；`Vm::new()` 初始化 |
| `tenth/src/main.rs` | `register_natives` 新增 10 个 native：eprint/eprintln/read_line/env_get/env_set/exit/tcp_connect/tcp_read/tcp_write/tcp_close/http_get/http_post（12 个） |
| `tenth/src/runtime/interpreter/natives.rs` | 对应新增 12 个 native 的解释器实现（双侧对齐） |
| `tenth/src/runtime/value.rs` | 无变更（不引入新 Value 变体） |
| `tenth/src/runtime/limits.rs` | 无变更（网络 I/O 不受沙箱限制） |

### 5.2 标准库（4 个新文件）

| 文件 | 说明 |
|------|------|
| `tenth/std/io.th` | stderr/stdin 封装 |
| `tenth/std/env.th` | 环境变量/进程控制封装 |
| `tenth/std/net.th` | TCP 网络封装 |
| `tenth/std/http.th` | HTTP 客户端封装 |

### 5.3 文档（3 个文件）

| 文件 | 改动 |
|------|------|
| `tenth/std/prelude.th` | 修正 eprintln 标记；新增 I/O 函数索引 |
| `docs/语言参考手册.md` | §12 标准库章节新增 I/O/API 详表 |
| `MEMO.md` | 变更记录 |

### 5.4 测试（2 个新文件）

| 文件 | 说明 |
|------|------|
| `tenth/tests/io_test.rs` | Rust 集成测试：eprint/eprintln/read_line/env_get/env_set |
| `tenth/tests/net_test.rs` | Rust 集成测试：tcp_connect/tcp_read/tcp_write/http_get（用 echo TCP 服务器 + httpbin.org 测试） |

---

## 六、实现顺序

### 阶段 1：stderr + stdin（低风险，立即可用）

1. `main.rs` 新增 eprint/eprintln/read_line native
2. `natives.rs` 对应实现
3. `std/io.th` 封装
4. `prelude.th` 修正
5. `io_test.rs` 测试
6. 验证：cargo test 全绿

### 阶段 2：环境变量 + 进程控制

1. `main.rs` 新增 env_get/env_set/exit native
2. `natives.rs` 对应实现
3. `std/env.th` 封装
4. `io_test.rs` 补充测试
5. 验证：cargo test 全绿

### 阶段 3：TCP 网络原语（中风险）

1. `vm.rs` Vm 结构新增 tcp_streams 字段
2. `main.rs` 新增 tcp_connect/tcp_read/tcp_write/tcp_close/tcp_set_timeout native
3. `natives.rs` 对应实现
4. `std/net.th` 封装
5. `net_test.rs` 测试（本地 echo 服务器）
6. 验证：cargo test 全绿

### 阶段 4：HTTP 客户端（基于阶段 3）

1. `main.rs` 新增 http_get/http_post native（+ http_request/parse_url 辅助函数）
2. `natives.rs` 对应实现
3. `std/http.th` 封装
4. `net_test.rs` 补充 HTTP 测试（本地 HTTP 服务器或 httpbin.org）
5. 验证：cargo test 全绿

### 阶段 5：文档同步 + 提交

1. `prelude.th` 更新索引
2. `docs/语言参考手册.md` §12 新增 I/O 章节
3. `MEMO.md` 变更记录
4. 全量回归测试
5. 自举验证
6. 提交

---

## 七、风险分析

### 7.1 高风险点

| 风险 | 影响 | 缓解 |
|------|------|------|
| `exit()` 绕过 VM 清理 | 资源泄漏 | 文档明确：exit 立即终止，不执行 defer/finalizer |
| `tcp_streams` 句柄表无限增长 | 内存泄漏 | 句柄可重用槽位（用 `Option<TcpStream>`，None 槽可回收） |
| HTTP 解析不完整 | 仅支持简单响应 | 文档明确：不支持 chunked transfer、重定向、HTTPS |
| `read_line` 阻塞 | VM 卡死 | 文档明确：read_line 是阻塞调用；未来异步 I/O 解决 |

### 7.2 低风险点

| 点 | 说明 |
|----|------|
| eprint/eprintln | 纯输出，无副作用 |
| env_get/env_set | 标准库调用，安全 |
| 不引入新 Value 变体 | 避免全链路 HIR 变更 |
| 不改现有 I/O native | 向后兼容 |

---

## 八、测试计划

### 8.1 单元测试（io_test.rs）

```rust
#[test]
fn test_eprint_eprintln() {
    // 测试 stderr 输出（捕获 stderr）
    let src = r#"
        fn main() {
            eprint("错误信息")
            eprintln("换行错误")
        }
    "#;
    // 运行并验证 stderr 输出
}

#[test]
fn test_env_get_set() {
    let src = r#"
        fn main() {
            env_set("TENTH_TEST_VAR", "hello")
            try {
                let v = env_get("TENTH_TEST_VAR")?
                print(v)
            } match {
                Result::Ok(_) => print("ok"),
                Result::Err(_) => print("err"),
            }
        }
    "#;
    // 验证输出 "hello"
}
```

### 8.2 网络测试（net_test.rs）

```rust
#[test]
fn test_tcp_echo() {
    // 启动本地 echo TCP 服务器
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    std::thread::spawn(move || {
        let (mut stream, _) = listener.accept().unwrap();
        use std::io::{Read, Write};
        let mut buf = [0u8; 1024];
        let n = stream.read(&mut buf).unwrap();
        stream.write_all(&buf[..n]).unwrap();
    });

    let src = format!(r#"
        fn main() {{
            try {{
                let h = tcp_connect("127.0.0.1", {port}, 5000)?
                let data = vec(1, 2, 3, 4, 5)
                tcp_write(h, data)?
                let resp = tcp_read(h, 1024)?
                tcp_close(h)
                print("received " + int_to_str(resp.len()))
            }} match {{
                Result::Ok(_) => print("ok"),
                Result::Err(e) => print("err: " + e),
            }}
        }}
    "#);
    // 验证输出 "received 5"
}

#[test]
fn test_http_get() {
    // 启动本地 HTTP 服务器
    // ...
    let src = format!(r#"
        fn main() {{
            try {{
                let body = http_get("http://127.0.0.1:{port}/test", 5000)?
                print(body)
            }} match {{
                Result::Ok(_) => print("ok"),
                Result::Err(e) => print("err: " + e),
            }}
        }}
    "#);
    // 验证输出预期响应体
}
```

---

## 九、API 完整清单

### 9.1 新增 Native 函数（12 个）

| # | 函数名 | 参数 | 返回值 | 模块 |
|---|--------|------|--------|------|
| 1 | `eprint` | `value` | `()` | io |
| 2 | `eprintln` | `value` | `()` | io |
| 3 | `read_line` | `()` | `Result<String>` | io |
| 4 | `env_get` | `name: String` | `Result<String>` | env |
| 5 | `env_set` | `name: String, value: String` | `()` | env |
| 6 | `exit` | `code: i64` | `()` | env |
| 7 | `tcp_connect` | `host: String, port: i64, timeout_ms: i64` | `Result<i64>` | net |
| 8 | `tcp_read` | `handle: i64, max_bytes: i64` | `Result<Vec<i64>>` | net |
| 9 | `tcp_write` | `handle: i64, data: Vec/i64>` | `Result<i64>` | net |
| 10 | `tcp_close` | `handle: i64` | `()` | net |
| 11 | `tcp_set_timeout` | `handle: i64, read_ms: i64, write_ms: i64` | `()` | net |
| 12 | `http_get` | `url: String, timeout_ms: i64` | `Result<String>` | http |
| 13 | `http_post` | `url: String, body: String, content_type: String, timeout_ms: i64` | `Result<String>` | http |

### 9.2 新增标准库模块（4 个）

```
tenth/std/
├── io.th        (io 模块: eprint/eprintln/read_line)
├── env.th       (env 模块: get/set/exit)
├── net.th       (net 模块: connect/send/recv/close)
└── http.th      (http 模块: get/post)
```

---

## 十、与现有系统的关系

### 10.1 与 Phase 1 async 的关系

- **当前**：同步 I/O 与 Phase 1 的 `Future<T>` 互不干扰
- **未来 Phase 2/3**：异步 I/O 可基于同步 I/O 原语包装：`async_read_file(path) = spawn read_file(path)`，协程调度器在 I/O 阻塞时挂起

### 10.2 与沙箱的关系

- **文件 I/O**：经 `FsSandbox` 校验（现有机制，不变）
- **网络 I/O**：**不受沙箱限制**（无路径逃逸风险），但有超时保护
- **环境变量**：**不受沙箱限制**（环境变量本就是进程级配置）

### 10.3 与 tenthc 自举的关系

- **不涉及 tenthc**：本次改动仅在 Rust 侧 + 标准库 `.th` 文件，不触及 tenthc 编译器源码
- **自举不受影响**：tenthc 不使用 I/O 原语，自举路径不变

---

## 十一、后续演进路径

```
当前：同步 I/O 原语（本方案）
  ↓
Phase 2：协程调度（Ready Queue + Frame 挂起/恢复）
  ↓
Phase 3：异步 I/O（sync I/O 包装为 async I/O，spawn + await）
  ↓
未来：TLS/HTTPS、并发服务器、文件事件监听
```

---

## 十二、验证标准

| 验证项 | 通过标准 |
|--------|---------|
| `cargo check` | 0 error |
| `cargo test --lib` | 全部通过 |
| `cargo test --test io_test` | 全部通过 |
| `cargo test --test net_test` | 全部通过 |
| 现有测试 | 0 回归（64 个测试全绿） |
| 自举 Path B | `[OK] Full compiler compiled to tenthc_full.wasm` |
| 临时功能测试 | eprint/read_line/env_get/tcp_echo/http_get 均输出预期结果 |

---

## 十三、工作量估算

| 阶段 | 文件数 | 新增代码行 | 风险 |
|------|--------|-----------|------|
| 阶段 1（stderr+stdin） | 3 | ~80 | 低 |
| 阶段 2（env+exit） | 3 | ~60 | 低 |
| 阶段 3（TCP） | 4 | ~250 | 中 |
| 阶段 4（HTTP） | 3 | ~150 | 中 |
| 阶段 5（文档+测试） | 5 | ~300 | 低 |
| **合计** | **~18** | **~840** | **中** |
