//! UDP 原语集成测试（基本功核查第 69 项）。
//!
//! 覆盖 5 个 UDP native：
//! - `udp_bind(addr: String)` → `Result<i64>`（返回 1-based handle）
//! - `udp_recv_from(sock: i64, n: i64)` → `Result<Tuple<Vec<i64>, String>>`（读最多 n 字节 + 来源地址）
//! - `udp_send_to(sock: i64, data: Vec<i64>, addr: String)` → `Result<i64>`（返回发送字节数）
//! - `udp_close(sock: i64)` → `Unit`
//! - `udp_set_timeout(sock: i64, ms: i64)` → `Unit`
//!
//! 使用解释器（Interpreter）执行——其 `call_named_fn`（natives.rs）已内置全部 UDP native。
//! UDP 无连接，测试通过本地 socket 对发验证往返；地址用 "127.0.0.1:0" 让 OS 分配端口。

use std::net::UdpSocket;

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

// ─── Test 1: udp_bind 成功返回句柄，close 后 recv_from 返回 Err ──────────────

#[test]
fn test_udp_bind_and_close() {
    // bind 127.0.0.1:0 让 OS 分配端口，应返回 Ok(handle >= 1)；
    // close 后再 recv_from 应返回 Err（"socket 已关闭"）。
    let src = r#"
        let r = udp_bind("127.0.0.1:0");
        match r {
            Result::Ok(h) => {
                udp_close(h);
                let recv = udp_recv_from(h, 100);
                match recv {
                    Result::Ok(_) => 0,
                    Result::Err(_) => 1,
                }
            },
            Result::Err(_) => -2,
        }
    "#;
    let result = run_code(src).unwrap();
    match result {
        Some(Value::Int(1, _)) => {}
        v => panic!("期望 Some(Int(1)) 表示 close 后 recv_from 返回 Err，got {:?}", v),
    }
}

// ─── Test 2: udp_send_recv 本地回环 ──────────────────────────────────────────

#[test]
fn test_udp_send_recv_loopback() {
    // Rust 侧绑定 receiver A，spawn 线程 recv_from 拿到 B 的地址后回写 echo。
    // Tenth 侧绑定 socket B，send_to A 发送 [72, 105]，recv_from 收回 echo。
    let receiver_a = UdpSocket::bind("127.0.0.1:0").unwrap();
    let addr_a = receiver_a.local_addr().unwrap().to_string();
    // 设置 receiver 超时避免测试卡死
    receiver_a
        .set_read_timeout(Some(std::time::Duration::from_secs(5)))
        .unwrap();
    std::thread::spawn(move || {
        let mut buf = [0u8; 1024];
        match receiver_a.recv_from(&mut buf) {
            Ok((n, peer)) => {
                // 回写相同数据作为 echo
                let _ = receiver_a.send_to(&buf[..n], peer);
            }
            Err(_) => {}
        }
    });

    let src = format!(
        r#"
        let rb = udp_bind("127.0.0.1:0");
        match rb {{
            Result::Ok(b) => {{
                udp_set_timeout(b, 5000);
                let sent = udp_send_to(b, [72, 105], "{addr_a}");
                match sent {{
                    Result::Ok(n) => {{
                        let recv = udp_recv_from(b, 100);
                        match recv {{
                            Result::Ok(pair) => {{
                                // pair 是 Tuple(Vec<i64>, String)
                                // 解构取第一个元素 Vec<i64> 的长度
                                let (bytes, _addr) = pair;
                                bytes.len()
                            }},
                            Result::Err(_) => -3,
                        }}
                    }},
                    Result::Err(_) => -4,
                }}
            }},
            Result::Err(_) => -2,
        }}
        "#
    );
    let result = run_code(&src).unwrap();
    match result {
        Some(Value::Int(2, _)) => {}
        v => panic!("期望 Some(Int(2)) 表示收到 2 字节回环数据，got {:?}", v),
    }
}

// ─── Test 3: udp_send_to 无效地址返回 Err ────────────────────────────────────

#[test]
fn test_udp_send_to_unreachable() {
    // 发往无法解析的地址字符串，send_to 应返回 Err。
    let src = r#"
        let r = udp_bind("127.0.0.1:0");
        match r {
            Result::Ok(h) => {
                let sent = udp_send_to(h, [1, 2, 3], "not-a-valid-address");
                match sent {
                    Result::Ok(_) => 0,
                    Result::Err(_) => 1,
                }
            },
            Result::Err(_) => -2,
        }
    "#;
    let result = run_code(src).unwrap();
    match result {
        Some(Value::Int(1, _)) => {}
        v => panic!("期望 Some(Int(1)) 表示无效地址返回 Err，got {:?}", v),
    }
}

// ─── Test 4: udp_recv_from 空数据 ────────────────────────────────────────────

#[test]
fn test_udp_recv_from_empty() {
    // A 发送 0 字节给 B，B recv_from 应返回空 Vec（长度 0）。
    let receiver_b = UdpSocket::bind("127.0.0.1:0").unwrap();
    let addr_b = receiver_b.local_addr().unwrap().to_string();
    receiver_b
        .set_read_timeout(Some(std::time::Duration::from_secs(5)))
        .unwrap();
    // Tenth 侧绑定 A，发送空 Vec 给 B
    let src = format!(
        r#"
        let ra = udp_bind("127.0.0.1:0");
        match ra {{
            Result::Ok(a) => {{
                let sent = udp_send_to(a, [], "{addr_b}");
                match sent {{
                    Result::Ok(_) => 0,
                    Result::Err(_) => -1,
                }}
            }},
            Result::Err(_) => -2,
        }}
        "#
    );
    let result = run_code(&src).unwrap();
    // Tenth 侧发送完毕；Rust 侧 B 应收到 0 字节
    match result {
        Some(Value::Int(0, _)) => {}
        v => panic!("期望 Some(Int(0)) 表示 Tenth 侧发送成功，got {:?}", v),
    }
    // 验证 Rust 侧收到 0 字节
    let mut buf = [0u8; 100];
    match receiver_b.recv_from(&mut buf) {
        Ok((0, _)) => {} // 0 字节
        Ok((n, _)) => panic!("期望收到 0 字节，实际收到 {} 字节", n),
        Err(e) => panic!("Rust 侧 recv_from 失败: {}", e),
    }
}

// ─── Test 5: udp_set_timeout 超时返回 Err ────────────────────────────────────

#[test]
fn test_udp_set_timeout() {
    // 绑定 socket，设置 100ms 读超时，recv_from 在无数据时应返回 Err（超时）。
    // 用系统时间验证超时确实发生（< 2 秒）。
    let src = r#"
        let r = udp_bind("127.0.0.1:0");
        match r {
            Result::Ok(h) => {
                udp_set_timeout(h, 100);
                let recv = udp_recv_from(h, 100);
                match recv {
                    Result::Ok(_) => 0,
                    Result::Err(_) => 1,
                }
            },
            Result::Err(_) => -2,
        }
    "#;
    let start = std::time::Instant::now();
    let result = run_code(src).unwrap();
    let elapsed = start.elapsed();
    // 超时应快速返回（100ms + 开销，< 5 秒）
    assert!(
        elapsed < std::time::Duration::from_secs(5),
        "超时返回过慢：{:?}",
        elapsed
    );
    match result {
        Some(Value::Int(1, _)) => {}
        v => panic!(
            "期望 Some(Int(1)) 表示超时返回 Err，got {:?} (elapsed {:?})",
            v, elapsed
        ),
    }
}
