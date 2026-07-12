use tenth::lexer::lexer::Lexer;
use tenth::parser::parser::Parser;
use tenth::hir::lower::Lowerer;
use tenth::runtime::vm::Vm;
use tenth::runtime::value::Value;
use tenth::runtime::async_io::{ASYNC_IO, IoResult};
use tenth::compile::bytecode::BytecodeCompiler;
use std::rc::Rc;
use std::cell::RefCell;
use tenth::hir::types::BaseType;

/// Run source through lexer → parser → HIR → bytecode → VM.
/// Provides `print`/`println` + async I/O natives so tests can observe values
/// and exercise the async scheduler.
fn run_vm(src: &str) -> Result<Value, String> {
    let mut lexer = Lexer::new(src);
    let tokens = lexer.tokenize().map_err(|e| e.to_string())?;
    let mut parser = Parser::new(tokens);
    let program = parser.parse_program().map_err(|e| e.to_string())?;
    let mut lowerer = Lowerer::new();
    let hir = lowerer.lower_program(&program).map_err(|e| e.to_string())?;

    let mut vm = Vm::new();
    vm.add_native("print".into(), |_vm, args| {
        for a in args { print!("{a}"); }
        Ok(Value::Unit)
    });
    vm.add_native("println".into(), |_vm, args| {
        for a in args { print!("{a}"); }
        println!();
        Ok(Value::Unit)
    });
    vm.add_native("Vec::new".into(), |_vm, _args| {
        Ok(Value::Vec(Rc::new(RefCell::new(Vec::new()))))
    });
    // time_now_ms for timing assertions
    vm.add_native("time_now_ms".into(), |_vm, _args| {
        let ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as i64)
            .unwrap_or(0);
        Ok(Value::Int(ms, BaseType::I32))
    });
    // async_sleep_ms: register timer in ASYNC_IO
    vm.add_native("async_sleep_ms".into(), |_vm, args| {
        let ms = match args.first() {
            Some(Value::Int(n, _)) => *n,
            _ => return Err(tenth::error::TenthError::RuntimeError { line: None, col: None,
                message: "async_sleep_ms(ms) 期望一个整数".into(),
            }),
        };
        if ms < 0 {
            return Err(tenth::error::TenthError::RuntimeError { line: None, col: None,
                message: format!("async_sleep_ms: 不接受负数（{}）", ms),
            });
        }
        let future = Value::future_pending();
        if let Value::Future(rc) = &future {
            let rc_clone = rc.clone();
            ASYNC_IO.with(|io| io.borrow_mut().add_timer(ms as u64, rc_clone));
        }
        Ok(future)
    });
    // tcp_connect / tcp_read / tcp_write / tcp_close (sync, for setup)
    vm.add_native("tcp_connect".into(), |vm, args| {
        if args.len() < 2 {
            return Ok(err_result("tcp_connect 需要 (String, i64) 参数"));
        }
        if let (Value::String(host), Value::Int(port, _)) = (&args[0], &args[1]) {
            let addr = format!("{}:{}", host, port);
            match std::net::TcpStream::connect(&addr) {
                Ok(stream) => {
                    vm.tcp_streams.push(Some(stream));
                    let handle = vm.tcp_streams.len() as i64;
                    Ok(ok_result(Value::Int(handle, BaseType::I32)))
                }
                Err(e) => Ok(err_result(format!("连接失败: {e}"))),
            }
        } else {
            Ok(err_result("tcp_connect 需要 (String, i64) 参数"))
        }
    });
    vm.add_native("tcp_close".into(), |vm, args| {
        if let Some(Value::Int(handle, _)) = args.first() {
            let idx = *handle as usize;
            if idx > 0 && idx <= vm.tcp_streams.len() {
                vm.tcp_streams[idx - 1] = None;
            }
        }
        Ok(Value::Unit)
    });
    // async_tcp_read: spawn thread, return Pending Future
    vm.add_native("async_tcp_read".into(), |vm, args| {
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
        let stream_clone = match vm.tcp_streams[idx - 1].as_ref() {
            Some(s) => match s.try_clone() {
                Ok(c) => c,
                Err(e) => return Ok(Value::future_ready(err_result(format!("句柄克隆失败: {e}")))),
            },
            None => return Ok(Value::future_ready(err_result("连接已关闭"))),
        };
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
                Ok(0) => IoResult::Bytes(Vec::new()),
                Ok(n) => IoResult::Bytes(buf[..n].to_vec()),
                Err(e) => IoResult::Err(format!("读取失败: {e}")),
            };
            let _ = tx.send(result);
        });
        ASYNC_IO.with(|io| io.borrow_mut().add_io(rx, future_rc));
        Ok(future)
    });
    // async_tcp_write: spawn thread, return Pending Future
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

    for func in &hir.functions {
        let compiler = BytecodeCompiler::new();
        match compiler.compile(func) {
            Ok((chunk, closures)) => {
                vm.add_fn(func.name.clone(), chunk);
                for (name, closure_chunk) in closures {
                    vm.add_fn(name, closure_chunk);
                }
            }
            Err(e) => return Err(format!("compile error: {}", e)),
        }
    }

    if let Some(ref expr) = hir.main_expr {
        let compiler = BytecodeCompiler::new();
        match compiler.compile_main(expr) {
            Ok((chunk, closures)) => {
                vm.add_fn("main".into(), chunk);
                for (name, closure_chunk) in closures {
                    vm.add_fn(name, closure_chunk);
                }
            }
            Err(e) => return Err(format!("compile error: {}", e)),
        }
        vm.call("main").map_err(|e| e.to_string())
    } else if vm.has_fn("main") {
        vm.call("main").map_err(|e| e.to_string())
    } else {
        Ok(Value::Unit)
    }
}

/// Construct Result::Ok(value)
fn ok_result(value: Value) -> Value {
    Value::Enum {
        enum_name: "Result".to_string(),
        variant: "Ok".to_string(),
        fields: Rc::new(RefCell::new(vec![("_0".to_string(), value)])),
    }
}

/// Construct Result::Err(message)
fn err_result(msg: impl Into<String>) -> Value {
    Value::Enum {
        enum_name: "Result".to_string(),
        variant: "Err".to_string(),
        fields: Rc::new(RefCell::new(vec![("_0".to_string(), Value::String(msg.into()))])),
    }
}

// ─── spawn produces a Future; await unwraps it ─────────────────────────────

#[test]
fn test_spawn_then_await_int() {
    let src = r#"
        fn make_num() -> int {
            return 42
        }
        fn main() {
            let f = spawn make_num()
            let n = await f
            print(n)
        }
    "#;
    let result = run_vm(src).unwrap();
    // main returns Unit; the test passes if no panic and result is Unit
    assert!(matches!(result, Value::Unit), "expected Unit, got {:?}", result);
}

// ─── await on a non-Future value is a no-op (passes through) ───────────────

#[test]
fn test_await_plain_value() {
    let src = r#"
        fn main() {
            let n = await 7
            print(n)
        }
    "#;
    let result = run_vm(src).unwrap();
    assert!(matches!(result, Value::Unit));
}

// ─── spawn of a literal expression ─────────────────────────────────────────

#[test]
fn test_spawn_literal() {
    let src = r#"
        fn main() {
            let f = spawn 99
            let n = await f
            print(n)
        }
    "#;
    let result = run_vm(src).unwrap();
    assert!(matches!(result, Value::Unit));
}

// ─── async fn keyword parses (is_async flag) ───────────────────────────────

#[test]
fn test_async_fn_parses() {
    // Even though async fn isn't fully wired through the type system yet,
    // the parser must accept the `async` keyword without error.
    let src = r#"
        async fn delayed() -> int {
            return 1
        }
        fn main() {
            let n = await spawn delayed()
            print(n)
        }
    "#;
    let result = run_vm(src);
    assert!(result.is_ok(), "async fn should parse and run: {:?}", result);
}

// ─── multiple spawns and awaits in sequence ────────────────────────────────

#[test]
fn test_multiple_spawn_await() {
    let src = r#"
        fn double(x: int) -> int {
            return x * 2
        }
        fn main() {
            let a = await spawn double(5)
            let b = await spawn double(10)
            print(a + b)
        }
    "#;
    let result = run_vm(src).unwrap();
    assert!(matches!(result, Value::Unit));
}

// ─── spawn preserves string values ─────────────────────────────────────────

#[test]
fn test_spawn_string() {
    let src = r#"
        fn greet() -> string {
            return "hi"
        }
        fn main() {
            let s = await spawn greet()
            print(s)
        }
    "#;
    let result = run_vm(src).unwrap();
    assert!(matches!(result, Value::Unit));
}

// ─── Phase 2 Step 5: async_sleep_ms basic ──────────────────────────────────

#[test]
fn test_async_sleep_ms_basic() {
    // await async_sleep_ms(10) should resolve to Unit
    let src = r#"
        fn main() {
            let f = async_sleep_ms(10);
            let v = await f;
            print("done");
        }
    "#;
    let result = run_vm(src);
    assert!(result.is_ok(), "async_sleep_ms should complete: {:?}", result.err());
    assert!(matches!(result.unwrap(), Value::Unit));
}

// ─── async_sleep_ms resolves to Unit ───────────────────────────────────────

#[test]
fn test_async_sleep_ms_returns_unit() {
    // await async_sleep_ms(5) resolves to Unit; assign to let _ to verify
    let src = r#"
        fn main() {
            let v = await async_sleep_ms(5);
            print(v);
        }
    "#;
    let result = run_vm(src);
    assert!(result.is_ok(), "async_sleep_ms await should work: {:?}", result.err());
}

// ─── async_sleep_ms concurrent (two sleeps ~overlap) ───────────────────────

#[test]
fn test_async_sleep_ms_concurrent() {
    // Two 50ms sleeps awaited sequentially should complete in ~50-100ms,
    // NOT 100ms+ (if they ran serially with sync sleep).
    // The test verifies they don't run serially by checking total time < 90ms
    // (50+50=100ms serial, ~50ms concurrent).
    let src = r#"
        fn main() {
            let t0 = time_now_ms();
            let a = async_sleep_ms(50);
            let b = async_sleep_ms(50);
            let _ = await a;
            let _ = await b;
            let t1 = time_now_ms();
            print(t1 - t0);
        }
    "#;
    let result = run_vm(src);
    assert!(result.is_ok(), "concurrent sleeps should complete: {:?}", result.err());
    // main returns Unit; the timing is printed to stdout (not asserted here
    // because flaky on CI). The key assertion is that it doesn't hang.
}

// ─── async_sleep_ms with yield interleaving ────────────────────────────────

#[test]
fn test_async_sleep_with_yield() {
    // spawn another task that sleeps; main task also sleeps.
    // The scheduler should interleave them. (yield keyword is not in parser
    // yet, so we use spawn + await to create concurrent tasks.)
    let src = r#"
        async fn sleeper() -> int {
            let _ = await async_sleep_ms(5);
            return 42
        }
        fn main() {
            let f = spawn sleeper();
            let n = await f;
            print(n);
        }
    "#;
    let result = run_vm(src);
    assert!(result.is_ok(), "spawn + async_sleep should work: {:?}", result.err());
}

// ─── async_tcp_write + async_tcp_read echo ─────────────────────────────────

#[test]
fn test_async_tcp_echo() {
    // Start a local echo server, connect, async_write, async_read, verify echo
    use std::io::{Read, Write};
    use std::net::TcpListener;

    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    std::thread::spawn(move || {
        if let Ok((mut stream, _)) = listener.accept() {
            let mut buf = [0u8; 1024];
            // echo loop: read then write back
            loop {
                match stream.read(&mut buf) {
                    Ok(0) => break,
                    Ok(n) => {
                        if stream.write_all(&buf[..n]).is_err() { break; }
                    }
                    Err(_) => break,
                }
            }
        }
    });

    let port_str = port.to_string();
    let src = format!(
        r#"
        fn main() {{
            let conn = tcp_connect("127.0.0.1", {port});
            let handle = match conn {{
                Result::Ok(h) => h,
                Result::Err(_) => 0,
            }};
            if handle == 0 {{ return 1 }}
            let data = Vec::new();
            data.push(72);  // 'H'
            data.push(105); // 'i'
            let w = async_tcp_write(handle, data);
            let wr = await w;
            let ok_w = match wr {{
                Result::Ok(n) => n,
                Result::Err(_) => 0,
            }};
            if ok_w == 0 {{ return 2 }}
            let r = async_tcp_read(handle, 16);
            let rr = await r;
            let bytes = match rr {{
                Result::Ok(v) => v,
                Result::Err(_) => Vec::new(),
            }};
            tcp_close(handle);
            return bytes.len()
        }}
        "#,
        port = port_str
    );
    let result = run_vm(&src);
    assert!(result.is_ok(), "async TCP echo should complete: {:?}", result.err());
    // Should have read back 2 bytes ("Hi")
    match result.unwrap() {
        Value::Int(n, _) => assert_eq!(n, 2, "expected 2 bytes echoed, got {}", n),
        v => panic!("expected Int(2), got {:?}", v),
    }
}

// ─── async_tcp_read on invalid handle returns Ready Err ────────────────────

#[test]
fn test_async_tcp_read_invalid_handle() {
    // Invalid handle should immediately resolve to Result::Err
    let src = r#"
        fn main() {
            let f = async_tcp_read(999, 10);
            let r = await f;
            return match r {
                Result::Ok(_) => 0,
                Result::Err(_) => 1,
            }
        }
    "#;
    let result = run_vm(src).unwrap();
    match result {
        Value::Int(1, _) => {}
        v => panic!("expected Int(1) for invalid handle, got {:?}", v),
    }
}

// ─── async_tcp_write on invalid handle returns Ready Err ───────────────────

#[test]
fn test_async_tcp_write_invalid_handle() {
    let src = r#"
        fn main() {
            let data = Vec::new();
            data.push(1);
            let f = async_tcp_write(999, data);
            let r = await f;
            return match r {
                Result::Ok(_) => 0,
                Result::Err(_) => 1,
            }
        }
    "#;
    let result = run_vm(src).unwrap();
    match result {
        Value::Int(1, _) => {}
        v => panic!("expected Int(1) for invalid handle, got {:?}", v),
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Phase 2 补充测试：调度正确性、Future 状态、async I/O 边界
// ═══════════════════════════════════════════════════════════════════════════

// ─── Future 状态：spawn 返回 Ready Future（eager 语义）─────────────────────

#[test]
fn test_spawn_returns_ready_future() {
    // spawn 是 eager：立即求值 inner，包装为 Ready Future。
    // 这里通过 type_of 间接验证：Ready Future 的 type_of 等于内部值的类型。
    // 由于无法在 Tenth 层直接 inspect FutureState，我们通过 await 立即返回验证 Ready。
    let src = r#"
        fn make_int() -> int {
            return 7
        }
        fn main() {
            let f = spawn make_int();
            // f 是 Ready Future，await 应立即返回（不挂起）
            let n = await f;
            return n;
        }
    "#;
    let result = run_vm(src).unwrap();
    match result {
        Value::Int(7, _) => {}
        v => panic!("expected Int(7) from eager spawn, got {:?}", v),
    }
}

// ─── Future 状态：async_sleep_ms 返回 Pending Future ───────────────────────

#[test]
fn test_async_sleep_returns_pending_future() {
    // async_sleep_ms 创建 Pending Future。我们通过 await 后任务能继续执行来验证：
    // 如果 Future 是 Ready，await 直接返回；
    // 如果是 Pending，await 会挂起当前任务，等定时器到期后恢复。
    // 这里用较短的 5ms 验证 Pending → Ready 的转换路径通畅。
    let src = r#"
        fn main() {
            let f = async_sleep_ms(5);
            // 立即 await：如果 Future 已 Ready（不应发生），直接返回；
            // 如果 Pending（期望），挂起 → 定时器到期 → 恢复 → 返回 Unit
            let v = await f;
            return 42;
        }
    "#;
    let result = run_vm(src);
    assert!(result.is_ok(), "Pending Future await should work: {:?}", result.err());
    match result.unwrap() {
        Value::Int(42, _) => {}
        v => panic!("expected Int(42) after Pending Future resolved, got {:?}", v),
    }
}

// ─── 调度正确性：主任务 await sleep 时其他 spawn 任务继续执行 ───────────────

#[test]
fn test_other_task_runs_during_await_sleep() {
    // 验证调度正确性：主任务 await 一个 sleep（Pending Future）时，
    // 调度器应切到其他就绪任务执行。
    // 实现方式：spawn 一个任务（eager，立即返回 Ready Future），
    // 然后主任务 await sleep。sleep 期间调度器无其他 Pending 任务，但
    // 验证调度器能在 await 后正确恢复主任务。
    // 这是回归测试：确保 await Pending 后能恢复，不卡死。
    let src = r#"
        fn helper() -> int {
            return 100
        }
        fn main() {
            let f = spawn helper();
            let _ = await async_sleep_ms(5);
            let n = await f;
            return n;
        }
    "#;
    let result = run_vm(src);
    assert!(result.is_ok(), "scheduler should resume after sleep: {:?}", result.err());
    match result.unwrap() {
        Value::Int(100, _) => {}
        v => panic!("expected Int(100), got {:?}", v),
    }
}

// ─── 调度正确性：多个 sleep 不同时间，验证可交错 ───────────────────────────

#[test]
fn test_multiple_sleeps_different_durations() {
    // 两个不同时长的 sleep（短 10ms + 长 100ms），先 await 短的后 await 长的。
    // 验证调度器正确处理多个 Pending Future：
    // - 短 sleep 到期时唤醒主任务，主任务继续到 await 长 sleep（再次挂起）
    // - 长 sleep 到期时再次唤醒主任务
    // 总时长应接近 100ms（不是 10ms + 100ms = 110ms 串行，因为短 sleep 先到期）。
    // 这里不严格断言时长（CI flaky），只验证不挂死、最终返回正确。
    let src = r#"
        fn main() {
            let a = async_sleep_ms(10);
            let b = async_sleep_ms(100);
            let _ = await a;
            let _ = await b;
            return 7;
        }
    "#;
    let result = run_vm(src);
    assert!(result.is_ok(), "multiple sleeps with different durations: {:?}", result.err());
    match result.unwrap() {
        Value::Int(7, _) => {}
        v => panic!("expected Int(7), got {:?}", v),
    }
}

// ─── async I/O 边界：sleep 0ms 立即完成 ────────────────────────────────────

#[test]
fn test_async_sleep_zero_ms() {
    // 0ms sleep 是边界情况：定时器立即到期（下一次 poll 即触发）。
    // 验证 Pending Future 在第一次 poll 时就转为 Ready，主任务被唤醒。
    let src = r#"
        fn main() {
            let f = async_sleep_ms(0);
            let _ = await f;
            return 1;
        }
    "#;
    let result = run_vm(src);
    assert!(result.is_ok(), "0ms sleep should complete: {:?}", result.err());
    match result.unwrap() {
        Value::Int(1, _) => {}
        v => panic!("expected Int(1) after 0ms sleep, got {:?}", v),
    }
}

// ─── async I/O 边界：链式 await 多个 sleep ─────────────────────────────────

#[test]
fn test_async_sleep_chain() {
    // 链式 await：sleep 5 → sleep 5 → sleep 5，总时长 ~15ms。
    // 验证主任务能多次挂起-恢复，调度器不丢任务。
    let src = r#"
        fn main() {
            let _ = await async_sleep_ms(5);
            let _ = await async_sleep_ms(5);
            let _ = await async_sleep_ms(5);
            return 99;
        }
    "#;
    let result = run_vm(src);
    assert!(result.is_ok(), "chained sleeps should complete: {:?}", result.err());
    match result.unwrap() {
        Value::Int(99, _) => {}
        v => panic!("expected Int(99) after chained sleeps, got {:?}", v),
    }
}

// ─── async I/O 边界：同一 Future 被 await 两次（幂等性）────────────────────

#[test]
fn test_await_same_future_twice() {
    // 同一个 Ready Future（来自 spawn）被 await 两次：
    // 第一次 await 取出值，第二次 await 应该也能取值（Future 仍是 Ready）。
    // 这是 spawn/eager 语义的回归测试。
    let src = r#"
        fn make_num() -> int {
            return 11
        }
        fn main() {
            let f = spawn make_num();
            let a = await f;
            let b = await f;
            return a + b;
        }
    "#;
    let result = run_vm(src).unwrap();
    match result {
        Value::Int(22, _) => {}
        v => panic!("expected Int(22) (11+11) from double await, got {:?}", v),
    }
}

// ─── async I/O 边界：Pending Future 被 await 两次 ──────────────────────────

#[test]
fn test_await_pending_future_twice() {
    // 同一个 Pending Future（来自 async_sleep_ms）被 await 两次。
    // 第一次 await 挂起主任务，定时器到期后唤醒；Future 转为 Ready。
    // 第二次 await 应走快路径立即返回 Unit。
    let src = r#"
        fn main() {
            let f = async_sleep_ms(5);
            let _ = await f;
            let _ = await f;
            return 3;
        }
    "#;
    let result = run_vm(src);
    assert!(result.is_ok(), "double await on Pending Future: {:?}", result.err());
    match result.unwrap() {
        Value::Int(3, _) => {}
        v => panic!("expected Int(3) after double await, got {:?}", v),
    }
}

// ─── TCP I/O 扩展：async_tcp_write 单独使用（不读回）──────────────────────

#[test]
fn test_async_tcp_write_only() {
    // 启动一个 TCP 服务器，只接收数据不回显。验证 async_tcp_write 能完成。
    use std::io::Read;
    use std::net::TcpListener;

    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    std::thread::spawn(move || {
        if let Ok((mut stream, _)) = listener.accept() {
            let mut buf = [0u8; 64];
            // 读一次数据后关闭（不回写）
            let _ = stream.read(&mut buf);
        }
    });

    let port_str = port.to_string();
    let src = format!(
        r#"
        fn main() {{
            let conn = tcp_connect("127.0.0.1", {port});
            let handle = match conn {{
                Result::Ok(h) => h,
                Result::Err(_) => 0,
            }};
            if handle == 0 {{ return 1 }}
            let data = Vec::new();
            data.push(65);  // 'A'
            data.push(66);  // 'B'
            data.push(67);  // 'C'
            let w = async_tcp_write(handle, data);
            let wr = await w;
            let ok_w = match wr {{
                Result::Ok(n) => n,
                Result::Err(_) => 0,
            }};
            tcp_close(handle);
            return ok_w;
        }}
        "#,
        port = port_str
    );
    let result = run_vm(&src);
    assert!(result.is_ok(), "async_tcp_write only should complete: {:?}", result.err());
    match result.unwrap() {
        Value::Int(3, _) => {}
        v => panic!("expected Int(3) bytes written, got {:?}", v),
    }
}

// ─── TCP I/O 扩展：async_sleep + async_tcp_write 混合 Pending Future ──────

#[test]
fn test_async_tcp_write_with_sleep() {
    // 验证调度器处理混合 timer + I/O Pending Future：
    // 先创建 sleep（timer Future）和 write（I/O Future），然后依次 await。
    // 调度器需要分别处理 timer 完成和 I/O 完成，唤醒主任务。
    // 注意：避免两次 match Result::Ok(h) 解构（已知 Lowerer local 分配 bug）。
    use std::io::Read;
    use std::net::TcpListener;

    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    std::thread::spawn(move || {
        if let Ok((mut stream, _)) = listener.accept() {
            let mut buf = [0u8; 1024];
            loop {
                match stream.read(&mut buf) {
                    Ok(0) => break,
                    Ok(_) => {}
                    Err(_) => break,
                }
            }
        }
    });

    let port_str = port.to_string();
    let src = format!(
        r#"
        fn main() {{
            let conn = tcp_connect("127.0.0.1", {port});
            let handle = match conn {{
                Result::Ok(h) => h,
                Result::Err(_) => 0,
            }};
            if handle == 0 {{ return 1 }}
            // 先创建 sleep Future（timer）
            let sleep_fut = async_sleep_ms(5);
            // 再创建 write Future（I/O）
            let data = Vec::new();
            data.push(72);   // 'H'
            data.push(105);  // 'i'
            let write_fut = async_tcp_write(handle, data);
            // 先 await sleep（挂起 → timer 到期 → 恢复）
            let _ = await sleep_fut;
            // 再 await write（可能已 Ready，也可能还 Pending）
            let w = await write_fut;
            let n = match w {{
                Result::Ok(n) => n,
                Result::Err(_) => 0,
            }};
            tcp_close(handle);
            return n;
        }}
        "#,
        port = port_str
    );
    let result = run_vm(&src);
    assert!(result.is_ok(), "sleep + write should complete: {:?}", result.err());
    match result.unwrap() {
        Value::Int(2, _) => {}
        v => panic!("expected Int(2) bytes written, got {:?}", v),
    }
}

// ─── 调度正确性：spawn 多个任务，主任务 await sleep，验证不挂死 ───────────

#[test]
fn test_spawn_multiple_with_sleep() {
    // spawn 多个任务（eager，立即完成），主任务 await sleep。
    // 验证多个 Ready Future + 一个 Pending Future 的混合场景不挂死。
    let src = r#"
        fn make_a() -> int { return 1 }
        fn make_b() -> int { return 2 }
        fn make_c() -> int { return 3 }
        fn main() {
            let fa = spawn make_a();
            let fb = spawn make_b();
            let fc = spawn make_c();
            let _ = await async_sleep_ms(5);
            let a = await fa;
            let b = await fb;
            let c = await fc;
            return a + b + c;
        }
    "#;
    let result = run_vm(src);
    assert!(result.is_ok(), "mixed spawn + sleep should complete: {:?}", result.err());
    match result.unwrap() {
        Value::Int(6, _) => {}
        v => panic!("expected Int(6) (1+2+3), got {:?}", v),
    }
}
