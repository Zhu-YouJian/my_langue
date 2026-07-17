//! Bytecode VM for Tenth — stack-based virtual machine.
//!
//! Architecture: HIR → compile → Chunk (bytecode) → Vm::run()
//!
//! 从 runtime/vm.rs 拆分而来（T3b 架构重构）：
//! - op.rs：Op 枚举 + YieldReason + Frame + TaskId
//! - chunk.rs：Chunk 结构体 + impl Chunk
//! - execute.rs：调度器 + 执行循环 + 私有算术 + 字段访问 + autodiff 记录
//! - natives.rs：call_method_priv 方法分派

use std::collections::{HashMap, VecDeque};
use crate::hir::types::BaseType;
use std::rc::Rc;
use std::cell::RefCell;
use crate::error::{TenthError, TenthResult};
use super::value::{Value, FutureState};
use super::autodiff::Tape;
use super::async_io::ASYNC_IO;

mod chunk;
mod execute;
mod natives;
mod op;

pub use chunk::Chunk;
pub use op::Op;
// Frame/YieldReason/TaskId 是 pub(super)，仅 vm 模块内部使用，不重新导出

/// Native Rust function callable from VM bytecode.
pub type NativeFn = fn(&mut Vm, &[Value]) -> TenthResult<Value>;

use op::{Frame, TaskId};

// ── Vm ─────────────────────────────────────────────────────────────────────

pub struct Vm {
    pub functions: HashMap<String, usize>,
    chunks: Vec<Chunk>,
    chunk_names: Vec<String>,
    pub natives: HashMap<String, NativeFn>,
    globals: HashMap<String, Value>,
    stack: Vec<Value>,
    frames: Vec<Frame>,
    /// Autodiff computation tape (active when `recording` is true).
    pub tape: Option<Tape>,
    /// Whether tensor operations should be recorded on the tape.
    pub recording: bool,
    /// Execution step budget. When `Some(n)`, each dispatched opcode
    /// decrements the counter; reaching zero raises `TenthError::Timeout`.
    /// `None` means unlimited (default).
    pub step_budget: Option<u64>,
    /// Optional wall-clock deadline (Unix ms). Checked periodically.
    pub deadline_ms: Option<u128>,
    /// 文件系统沙箱。`Some` 时所有文件 I/O 原生函数必须经过校验。
    /// `None` 表示无沙箱（默认，向后兼容）。
    pub fs_sandbox: Option<crate::runtime::limits::FsSandbox>,
    /// Lazily-initialised Cranelift JIT context. `None` until first JIT use.
    pub jit_ctx: Option<crate::compile::jit::context::JitContext>,
    /// Last error message set by a JIT hostcall trampoline.
    last_error: Option<String>,
    /// Index of the chunk currently being executed by JIT (for string lookup).
    pub current_chunk_idx: usize,
    /// 护城河 F：上一次 backward 失败时的根因说明列表（由 formal_explain 生成）。
    /// 由 `explain_error()` native 读取并清空。
    pub last_explanation: Vec<String>,
    /// TCP 流句柄表。索引+1 即句柄（1-based，0 表示无效）。
    /// `None` 表示已关闭的槽位（可被复用或保留）。
    pub tcp_streams: Vec<Option<std::net::TcpStream>>,
    /// TCP 监听器句柄表。索引+1 即句柄（1-based，0 表示无效）。
    /// `None` 表示已关闭的槽位（可被复用或保留）。
    pub tcp_listeners: Vec<Option<std::net::TcpListener>>,
    /// 正则表达式句柄表。索引+1 即句柄（1-based，0 表示无效）。
    /// `None` 表示已释放的槽位（可被复用或保留）。
    pub regexes: Vec<Option<regex::Regex>>,
    /// 子进程 Command 句柄表。索引+1 即句柄（1-based，0 表示无效）。
    /// `None` 表示已释放的槽位（command_output 消费后变 None）。
    pub commands: Vec<Option<std::process::Command>>,
    /// 下一个协程任务 ID 生成器。0 保留给主任务；从此字段递增分配。
    /// Phase 2 Step 1-2 仅初始化，不使用（spawn 仍走同步路径）。
    next_task_id: TaskId,
    /// Phase 2 调度器：就绪任务队列。`run_scheduler` 循环 `pop_front` 取任务执行。
    ready_queue: VecDeque<TaskId>,
    /// Phase 2 调度器：挂起任务的完整调用栈。
    /// key = task_id，value = 该任务挂起时的 `self.frames` 快照（含当前帧）。
    /// `await` 遇到 Pending Future 时写入；`run_scheduler` 恢复时取出。
    suspended: HashMap<TaskId, Vec<Frame>>,
    /// Phase 2 调度器：已完成任务的结果。
    /// 任务顶层 Ret 时写入；主任务（task_id=0）的结果即 `run_scheduler` 返回值。
    task_results: HashMap<TaskId, Value>,
    /// Phase 2 调度器：task_id → Future 句柄映射。
    /// 真正的异步任务（Step 5 的 async I/O）创建 Pending Future 时注册；
    /// 任务完成时调度器据此把 Future 设为 Ready 并唤醒等待者。
    /// Phase 2 Step 3-4 中 spawn 仍为 eager（不注册此表），此表为空。
    task_futures: HashMap<TaskId, Rc<RefCell<FutureState>>>,
    /// Phase 2 调度器：当前正在执行的任务标识。
    /// 由 `run_until_yield` 在入口/Frame 切换时同步更新；native 函数可读取此值
    /// 以便 Step 5 的 async I/O 能正确注册到当前 task。
    current_task: TaskId,
}

impl Vm {
    pub fn new() -> Self {
        Vm {
            functions: HashMap::new(), chunks: Vec::new(), chunk_names: Vec::new(),
            natives: HashMap::new(), globals: HashMap::new(),
            stack: Vec::new(), frames: Vec::new(),
            tape: None, recording: false,
            step_budget: None, deadline_ms: None, fs_sandbox: None,
            jit_ctx: None, last_error: None, current_chunk_idx: 0,
            last_explanation: Vec::new(), tcp_streams: Vec::new(),
            tcp_listeners: Vec::new(),
            regexes: Vec::new(),
            commands: Vec::new(),
            next_task_id: 1,
            ready_queue: VecDeque::new(),
            suspended: HashMap::new(),
            task_results: HashMap::new(),
            task_futures: HashMap::new(),
            current_task: 0,
        }
    }

    // ── JIT accessors ──────────────────────────────────────────────────────

    pub fn is_recording(&self) -> bool { self.recording }
    pub fn stack_len(&self) -> usize { self.stack.len() }
    pub fn stack_push(&mut self, v: Value) { self.stack.push(v); }
    pub fn stack_pop(&mut self) -> Value { self.stack.pop().unwrap_or(Value::Unit) }
    pub fn get_global(&self, name: &str) -> Option<Value> { self.globals.get(name).cloned() }
    pub fn set_last_error(&mut self, msg: String) { self.last_error = Some(msg); }
    pub fn take_last_error(&mut self) -> Option<String> { self.last_error.take() }
    /// 检查是否有未处理的错误（不清除）。
    /// JIT translator 在 MethodCall 后调用此方法，若发现错误则提前中止。
    pub fn has_last_error(&self) -> bool { self.last_error.is_some() }

    pub fn chunk_index_of(&self, name: &str) -> Option<usize> {
        self.functions.get(name).copied()
    }
    pub fn chunk_at(&self, idx: usize) -> &Chunk { &self.chunks[idx] }
    pub fn string_at(&self, idx: usize) -> Option<String> {
        self.chunks.get(self.current_chunk_idx)
            .and_then(|c| c.strings.get(idx).cloned())
    }
    pub fn chunk_name_at(&self, idx: usize) -> Option<String> {
        self.chunk_names.get(idx).cloned()
    }

    /// Call a function by name with explicit args (used by JIT hostcalls).
    pub fn call_with_args(&mut self, name: &str, args: &[Value]) -> TenthResult<Value> {
        // Try native first
        if let Some(native_fn) = self.natives.get(name).copied() {
            return native_fn(self, args);
        }
        // Push args and call user function
        for a in args { self.stack.push(a.clone()); }
        self.call(name)
    }

    // ── Public arithmetic/method wrappers (for JIT hostcalls) ──────────────

    pub fn add(&mut self, a: &Value, b: &Value) -> TenthResult<Value> { self.add_priv(a, b) }
    pub fn sub(&mut self, a: &Value, b: &Value) -> TenthResult<Value> { self.sub_priv(a, b) }
    pub fn mul(&mut self, a: &Value, b: &Value) -> TenthResult<Value> { self.mul_priv(a, b) }
    pub fn div(&mut self, a: &Value, b: &Value) -> TenthResult<Value> { self.div_priv(a, b) }
    pub fn rem(&mut self, a: &Value, b: &Value) -> TenthResult<Value> {
        match (a, b) {
            (Value::Int(x, _), Value::Int(y, _)) => {
                if *y == 0 { return Err(TenthError::RuntimeError { line: None, col: None, message: "整数取模除零".into() }); }
                Ok(Value::Int(x % y, BaseType::I32))
            }
            _ => Err(TenthError::RuntimeError { line: None, col: None, message: "% 需要整数".into() }),
        }
    }
    pub fn neg(&mut self, a: &Value) -> TenthResult<Value> {
        match a {
            Value::Int(n, _) => Ok(Value::Int(-n, BaseType::I32)),
            Value::Float(n) => Ok(Value::Float(-n)),
            Value::Float32(n) => Ok(Value::Float32(-n)),
            Value::Tensor(t) => Ok(Value::Tensor(Rc::new(RefCell::new(t.borrow().neg())))),
            _ => Err(TenthError::RuntimeError { line: None, col: None, message: "无法取负".into() }),
        }
    }
    pub fn not(&mut self, a: &Value) -> TenthResult<Value> {
        Ok(Value::Bool(!a.is_truthy()))
    }
    pub fn eq(&mut self, a: &Value, b: &Value) -> TenthResult<Value> { Ok(Value::Bool(self.vm_eq(a, b))) }
    pub fn neq(&mut self, a: &Value, b: &Value) -> TenthResult<Value> { Ok(Value::Bool(!self.vm_eq(a, b))) }
    pub fn lt(&mut self, a: &Value, b: &Value) -> TenthResult<Value> {
        Ok(Value::Bool(self.compare(a, b, |x, y| x < y, |x, y| x < y)?))
    }
    pub fn gt(&mut self, a: &Value, b: &Value) -> TenthResult<Value> {
        Ok(Value::Bool(self.compare(a, b, |x, y| x > y, |x, y| x > y)?))
    }
    pub fn lte(&mut self, a: &Value, b: &Value) -> TenthResult<Value> {
        Ok(Value::Bool(self.compare(a, b, |x, y| x <= y, |x, y| x <= y)?))
    }
    pub fn gte(&mut self, a: &Value, b: &Value) -> TenthResult<Value> {
        Ok(Value::Bool(self.compare(a, b, |x, y| x >= y, |x, y| x >= y)?))
    }
    pub fn index_get(&mut self, target: &Value, idx: &Value) -> TenthResult<Value> {
        match target {
            Value::Vec(items) => {
                let i = idx.as_int().unwrap_or(0) as usize;
                Ok(items.borrow().get(i).cloned().unwrap_or(Value::Unit))
            }
            Value::String(s) => {
                let i = idx.as_int().unwrap_or(0) as usize;
                Ok(Value::String(s.chars().nth(i).map(|c| c.to_string()).unwrap_or_default()))
            }
            Value::Tensor(t) => {
                // NumPy 语义：单索引沿第 0 维降维。
                // - 1D 张量 t[i] → 标量 Value::Float
                // - N-D 张量 t[i] → (N-1)-D 子张量 Value::Tensor
                let i = idx.as_int().unwrap_or(0) as usize;
                let tensor = t.borrow();
                if tensor.ndim() <= 1 {
                    // 1D 或 0D：返回标量
                    match tensor.get(&[i]) {
                        Some(val) => Ok(Value::Float(val)),
                        None => Err(TenthError::RuntimeError { line: None, col: None,
                            message: format!("索引 {} 越界，形状为 {:?}", i, tensor.shape()),
                        }),
                    }
                } else {
                    // N-D：返回降维子张量
                    match tensor.index_dim(i) {
                        Ok(sub) => Ok(Value::Tensor(Rc::new(RefCell::new(sub)))),
                        Err(msg) => Err(TenthError::RuntimeError { line: None, col: None, message: msg }),
                    }
                }
            }
            _ => Err(TenthError::RuntimeError { line: None, col: None, message: "无法索引".into() }),
        }
    }
    pub fn slice_str(&mut self, target: &Value, start: &Value, end: &Value) -> TenthResult<Value> {
        let start_idx = start.as_int().unwrap_or(0) as usize;
        let end_idx = end.as_int().unwrap_or(0) as usize;
        match target {
            Value::String(s) => {
                let chars: Vec<char> = s.chars().collect();
                let len = chars.len();
                let si = start_idx.min(len);
                let ei = end_idx.min(len);
                if si > ei {
                    return Err(TenthError::RuntimeError { line: None, col: None, message: "字符串切片起始位置大于结束位置".into() });
                }
                Ok(Value::String(chars[si..ei].iter().collect()))
            }
            _ => Err(TenthError::RuntimeError { line: None, col: None, message: "SliceStr 需要字符串目标".into() }),
        }
    }
    pub fn call_method(&mut self, receiver: &Value, method: &str, args: &[Value]) -> TenthResult<Value> {
        self.call_method_priv(receiver, method, args)
    }

    pub fn add_fn(&mut self, name: String, chunk: Chunk) {
        let idx = self.chunks.len();
        self.chunks.push(chunk);
        self.chunk_names.push(name.clone());
        self.functions.insert(name, idx);
    }

    pub fn has_fn(&self, name: &str) -> bool {
        self.functions.contains_key(name)
    }

    pub fn add_native(&mut self, name: String, f: NativeFn) {
        self.natives.insert(name, f);
    }

    pub fn set_global(&mut self, name: String, val: Value) {
        self.globals.insert(name, val);
    }

    /// Push arguments and call a native function.
    pub fn call_native(&mut self, name: &str, args: &[Value]) -> TenthResult<Value> {
        if let Some(f) = self.natives.get(name).copied() {
            f(self, args)
        } else {
            Err(TenthError::RuntimeError { line: None, col: None, message: format!("未定义的原生函数 '{}'", name) })
        }
    }

    pub fn call(&mut self, name: &str) -> TenthResult<Value> {
        let idx = self.functions.get(name).copied()
            .ok_or_else(|| TenthError::RuntimeError { line: None, col: None, message: format!("未定义的函数 '{}'", name) })?;
        self.run_scheduler(idx)
    }
}

fn err<T>(msg: &str) -> TenthResult<T> {
    Err(TenthError::RuntimeError { line: None, col: None, message: msg.into() })
}
