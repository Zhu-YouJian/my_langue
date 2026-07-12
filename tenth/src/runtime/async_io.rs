//! 异步 I/O 状态管理（Phase 2 Step 5）。
//!
//! 设计方案：`std::thread` + `std::sync::mpsc` + `thread_local`，零新依赖。
//!
//! 工作流程：
//! 1. async native 被调用时，创建 Pending Future，注册到 ASYNC_IO
//!    - `async_sleep_ms`：注册定时器（deadline + Future）
//!    - `async_tcp_read`/`async_tcp_write`：spawn OS 线程做阻塞 I/O，
//!      通过 mpsc channel 回传结果，注册 (Receiver, Future) 到 ASYNC_IO
//! 2. VM 调度器在 `run_scheduler` 循环中调用 `AsyncIoState::poll()`
//! 3. poll 检查定时器到期 / channel 有数据 → 设置 Future 为 Ready，提取等待者列表
//! 4. 等待者列表返回给调度器，调度器把它们推入 ready_queue
//!
//! 关键设计点：
//! - `Value`/`Rc` 非 `Send`，不能跨线程传输。worker 线程只发送 `IoResult`
//!   （`Vec<u8>`/`usize`/`String`，全部 `Send`），主线程收到后转换为 `Value`。
//! - `TcpStream::try_clone()` 让 worker 线程持有 stream 副本，原 stream 留在 VM 中。
//! - `task_id` 不在注册时存储——等待者列表由 `Op::Await` 写入 FutureState::Pending，
//!   poll 时从 FutureState 提取，这样解耦了 Future 创建者与等待者。

use std::cell::RefCell;
use crate::hir::types::BaseType;
use std::rc::Rc;
use std::sync::mpsc::{Receiver, TryRecvError};
use std::time::{Duration, Instant};

use super::value::{FutureState, Value};

/// 跨线程传输的 I/O 结果（全部 `Send`）。
#[derive(Debug)]
pub enum IoResult {
    /// 读取到的字节（`async_tcp_read` 成功）
    Bytes(Vec<u8>),
    /// 写入的字节数（`async_tcp_write` 成功）
    Count(usize),
    /// I/O 错误消息
    Err(String),
}

/// 异步 I/O 状态（thread_local）。
#[derive(Default)]
pub struct AsyncIoState {
    /// 定时器列表：(deadline, future)
    pub timers: Vec<(Instant, Rc<RefCell<FutureState>>)>,
    /// I/O 完成通道列表：(receiver, future)
    pub io_receivers: Vec<(Receiver<IoResult>, Rc<RefCell<FutureState>>)>,
}

thread_local! {
    pub static ASYNC_IO: RefCell<AsyncIoState> = RefCell::new(AsyncIoState::default());
}

impl AsyncIoState {
    /// 清空所有状态（每次 `run_scheduler` 入口调用，避免上次残留）。
    pub fn clear(&mut self) {
        self.timers.clear();
        self.io_receivers.clear();
    }

    /// 注册一个定时器：`ms` 毫秒后把 `future` 设为 `Ready(Unit)`。
    pub fn add_timer(&mut self, ms: u64, future: Rc<RefCell<FutureState>>) {
        let deadline = Instant::now() + Duration::from_millis(ms);
        self.timers.push((deadline, future));
    }

    /// 注册一个 I/O 完成通道：worker 线程通过 `rx` 发送结果，收到后设置 `future`。
    pub fn add_io(&mut self, rx: Receiver<IoResult>, future: Rc<RefCell<FutureState>>) {
        self.io_receivers.push((rx, future));
    }

    /// 是否有待完成的 I/O 或定时器。
    pub fn has_pending(&self) -> bool {
        !self.timers.is_empty() || !self.io_receivers.is_empty()
    }

    /// 轮询所有定时器与 I/O 通道，把就绪的 Future 设为 Ready，返回被唤醒的 task_id 列表。
    ///
    /// 返回的 task_id 来自 FutureState::Pending 中的 waiters 列表（由 `Op::Await` 写入）。
    /// 调度器收到后推入 ready_queue。
    pub fn poll(&mut self) -> Vec<u64> {
        let mut woken: Vec<u64> = Vec::new();
        let now = Instant::now();

        // ── 检查定时器 ──
        let mut i = 0;
        while i < self.timers.len() {
            if now >= self.timers[i].0 {
                let (_, future) = self.timers.swap_remove(i);
                wake_future(future, Value::Unit, &mut woken);
            } else {
                i += 1;
            }
        }

        // ── 检查 I/O 通道 ──
        let mut i = 0;
        while i < self.io_receivers.len() {
            match self.io_receivers[i].0.try_recv() {
                Ok(result) => {
                    let (_, future) = self.io_receivers.swap_remove(i);
                    let val = match result {
                        IoResult::Bytes(bytes) => {
                            let v: Vec<Value> = bytes.iter().map(|b| Value::Int(*b as i64, BaseType::I32)).collect();
                            ok_result(Value::Vec(Rc::new(RefCell::new(v))))
                        }
                        IoResult::Count(n) => ok_result(Value::Int(n as i64, BaseType::I32)),
                        IoResult::Err(e) => err_result(e),
                    };
                    wake_future(future, val, &mut woken);
                }
                Err(TryRecvError::Empty) => i += 1,
                Err(TryRecvError::Disconnected) => {
                    // worker 线程 panic / 退出未发送结果
                    let (_, future) = self.io_receivers.swap_remove(i);
                    wake_future(future, err_result("I/O 线程意外终止".to_string()), &mut woken);
                }
            }
        }

        woken
    }
}

/// 把 `future` 设为 `Ready(val)`，提取 Pending 状态中的 waiters 列表追加到 `woken`。
fn wake_future(future: Rc<RefCell<FutureState>>, val: Value, woken: &mut Vec<u64>) {
    let waiters = {
        let mut state = future.borrow_mut();
        let ws = match &*state {
            FutureState::Pending(ws) => ws.clone(),
            _ => vec![],
        };
        *state = FutureState::Ready(val);
        ws
    };
    woken.extend(waiters);
}

/// 构造 `Result::Ok(value)`。
fn ok_result(value: Value) -> Value {
    Value::Enum {
        enum_name: "Result".to_string(),
        variant: "Ok".to_string(),
        fields: Rc::new(RefCell::new(vec![("_0".to_string(), value)])),
    }
}

/// 构造 `Result::Err(message)`。
fn err_result(msg: impl Into<String>) -> Value {
    Value::Enum {
        enum_name: "Result".to_string(),
        variant: "Err".to_string(),
        fields: Rc::new(RefCell::new(vec![("_0".to_string(), Value::String(msg.into()))])),
    }
}
