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
use super::value::{Value, FutureState, check_int_overflow};
use super::autodiff::{Tape, CustomOpRegistry, CustomBackward};
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
    /// R4 操作数栈复用池：存放已清空但保留容量的空闲操作数栈。
    /// 函数返回（Ret/Try 提前返回）时 callee 栈入池，下次 Call 时复用，
    /// 消除每次函数调用的栈 buffer alloc/free（fib(28) 约 83 万次调用的大头）。
    /// 池内栈均为空（len=0）且不再被任何 Frame 引用，跨任务安全。
    stack_pool: Vec<Vec<Value>>,
    /// R5 locals 复用池：存放已清空但保留容量的空闲 locals 向量。
    /// 函数返回（Ret/Try 提前返回/未知 opcode 回退）时 callee locals 入池，
    /// 下次 Call/CallN/TailCall 时复用（resize 填 Unit），消除每次调用
    /// `vec![Value::Unit; n]` 的 buffer alloc/free（fib(28) 约 83 万次调用的大头之一）。
    /// 池内 locals 均 len=0（元素已 drop）且不再被任何 Frame/闭包引用——跨任务安全。
    locals_pool: Vec<Vec<Value>>,
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
    /// UDP socket 句柄表（基本功核查第 69 项）。索引+1 即句柄（1-based，0 表示无效）。
    /// `None` 表示已关闭的槽位（可被复用或保留）。与 TCP 表独立，避免类型混淆。
    pub udp_sockets: Vec<Option<std::net::UdpSocket>>,
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
    /// 自定义算子注册表（PROJ-006）。
    ///
    /// 用 `Rc<RefCell<...>>` 共享——Tape 在 backward 前通过 `set_custom_ops`
    /// 拿到 Rc 副本，使 backward 能访问用户的 `CustomBackward` 实现。
    /// register_custom_op 通过 `borrow_mut()` 修改；查询通过 `borrow()`。
    pub custom_ops: Rc<RefCell<CustomOpRegistry>>,
}

impl Vm {
    pub fn new() -> Self {
        Vm {
            functions: HashMap::new(), chunks: Vec::new(), chunk_names: Vec::new(),
            natives: HashMap::new(), globals: HashMap::new(),
            stack: Vec::new(), frames: Vec::new(), stack_pool: Vec::new(), locals_pool: Vec::new(),
            tape: None, recording: false,
            step_budget: None, deadline_ms: None, fs_sandbox: None,
            jit_ctx: None, last_error: None, current_chunk_idx: 0,
            last_explanation: Vec::new(), tcp_streams: Vec::new(),
            tcp_listeners: Vec::new(),
            udp_sockets: Vec::new(),
            regexes: Vec::new(),
            commands: Vec::new(),
            next_task_id: 1,
            ready_queue: VecDeque::new(),
            suspended: HashMap::new(),
            task_results: HashMap::new(),
            task_futures: HashMap::new(),
            current_task: 0,
            custom_ops: Rc::new(RefCell::new(CustomOpRegistry::new())),
        }
    }

    /// 注册自定义可微算子（PROJ-006）。
    ///
    /// 返回 `op_id`（用于 `TapeOp::Custom(op_id)`）。
    /// 若同名算子已注册，返回 `Err`。
    pub fn register_custom_op(&mut self, op: Box<dyn CustomBackward>) -> Result<usize, String> {
        self.custom_ops.borrow_mut().register(op)
    }

    /// 自定义算子注册表访问器（PROJ-006）。
    ///
    /// 返回 `Rc` 副本，供 Tape 在 backward 前通过 `set_custom_ops` 共享。
    pub fn custom_ops(&self) -> Rc<RefCell<CustomOpRegistry>> {
        Rc::clone(&self.custom_ops)
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
    ///
    /// a1 P1：补 globals-FnRef 解析——闭包值以 `StoreGlobal` 存为全局名（bytecode
    /// 闭包捕获 hack）时，按 `FnRef.name`（即闭包 chunk 名 `__closure_N`）解析；
    /// 与 execute.rs opcode 28 CallN 的既有逻辑对齐。仅 FnRef 分支条件性 clone。
    ///
    /// a1 P3：全局闭包 FnRef 携带捕获值 → 按名调用追加捕获实参（槽位
    /// params..params+captures），否则捕获缺失 → 静默错值。无捕获的常见路径
    /// 保持零额外分配（extra_captures 为空时直接借用 args）。
    pub fn call_with_args(&mut self, name: &str, args: &[Value]) -> TenthResult<Value> {
        // Try native first (直名)
        if let Some(native_fn) = self.natives.get(name).copied() {
            return native_fn(self, args);
        }
        // globals-FnRef 解析：`let f = |x| x+1; f(5)` 在 JIT/解释器宿主调用时，
        // `f` 是全局 FnRef（name = 闭包 chunk 名）。natives 别名同样可解析
        // （`let p = println; p("x")` → FnRef.name = "println"）。
        let mut extra_captures: Vec<Value> = Vec::new();
        let callee_name: String = match self.globals.get(name) {
            Some(Value::FnRef { name: fname, captures, .. }) => {
                extra_captures = captures.clone();
                fname.clone()
            }
            _ => name.to_string(),
        };
        if let Some(native_fn) = self.natives.get(&callee_name).copied() {
            if !extra_captures.is_empty() {
                let mut all = args.to_vec();
                all.extend(extra_captures.iter().cloned());
                return native_fn(self, &all);
            }
            return native_fn(self, args);
        }
        // Push args and call user function
        for a in args { self.stack.push(a.clone()); }
        for c in &extra_captures { self.stack.push(c.clone()); }
        self.call(&callee_name)
    }

    /// a1 P1：调用一个「可调用值」（解释器 `apply_closure` 的 VM 等价物）。
    ///
    /// 语义对齐解释器 `apply_closure`（methods.rs:673-705）：
    /// - `Value::FnRef { name, .. }` → 按名调用（natives → globals-FnRef → functions）
    /// - 其余值 → 「期望可调用值，得到 {:?}」（带类型）
    ///
    /// 供 JIT `host_call_indirect` 与 VM `CallClosure` 共用。
    /// a1 P3：捕获值追加为额外实参（槽位 params..params+captures）。name 是闭包
    /// chunk 名（非全局变量名），call_with_args 不会从 globals 二次追加，不会重复。
    pub fn call_value(&mut self, callee: &Value, args: &[Value]) -> TenthResult<Value> {
        match callee {
            Value::FnRef { name, captures, .. } => {
                if captures.is_empty() {
                    self.call_with_args(name, args)
                } else {
                    let mut all = args.to_vec();
                    all.extend(captures.iter().cloned());
                    self.call_with_args(name, &all)
                }
            }
            _ => Err(TenthError::RuntimeError {
                line: None,
                col: None,
                message: format!("期望可调用值，得到 {:?}", callee),
            }),
        }
    }

    // ── Public arithmetic/method wrappers (for JIT hostcalls) ──────────────

    pub fn add(&mut self, a: &Value, b: &Value) -> TenthResult<Value> { self.add_priv(a, b) }
    pub fn sub(&mut self, a: &Value, b: &Value) -> TenthResult<Value> { self.sub_priv(a, b) }
    pub fn mul(&mut self, a: &Value, b: &Value) -> TenthResult<Value> { self.mul_priv(a, b) }
    pub fn div(&mut self, a: &Value, b: &Value) -> TenthResult<Value> { self.div_priv(a, b) }
    pub fn rem(&mut self, a: &Value, b: &Value) -> TenthResult<Value> {
        match (a, b) {
            (Value::Int(x, dt), Value::Int(y, _)) => {
                if *y == 0 { return Err(TenthError::RuntimeError { line: None, col: None, message: "整数取模除零".into() }); }
                // AUDIT-11.4.17：checked_rem 拦截 i64::MIN % -1 等溢出（overflow-checks=true 下直接 % 会 panic）
                let r = x.checked_rem(*y).ok_or_else(|| super::value::int_overflow_err(*dt))?;
                check_int_overflow(r, *dt)?;
                Ok(Value::Int(r, BaseType::I32))
            }
            _ => Err(TenthError::RuntimeError { line: None, col: None, message: "% 需要整数".into() }),
        }
    }
    pub fn neg(&mut self, a: &Value) -> TenthResult<Value> {
        match a {
            // AUDIT-11.4.17：checked_neg 拦截 i64::MIN 取负溢出；check_int_overflow 与 VM Op::Neg 一致做 dtype 范围检查
            Value::Int(n, dt) => {
                let r = n.checked_neg().ok_or_else(|| super::value::int_overflow_err(*dt))?;
                check_int_overflow(r, *dt)?;
                Ok(Value::Int(r, BaseType::I32))
            }
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

impl Vm {
    /// B 批（VM 报错行号）：给运行时错误补充当前指令位置对应的源码行号。
    /// 仅当错误尚未携带行号时补齐（已带行号/非 RuntimeError 原样透传）。
    /// 用于 dispatch 循环内以 `?` 传播的错误（native / 方法调用 / 张量算术 / 字段访问等）。
    fn with_line(&self, chunk_idx: usize, ip: usize, err: TenthError) -> TenthError {
        match err {
            TenthError::RuntimeError { line: None, col, message } => {
                TenthError::RuntimeError { line: self.chunks[chunk_idx].line_at(ip), col, message }
            }
            other => other,
        }
    }

    /// B 批（VM 报错行号）：构造带当前指令位置行号的运行时错误。
    /// 用于 dispatch 循环内直接构造 RuntimeError 的场景（未定义函数 / 除零 / 无法索引等）。
    fn err_here(&self, chunk_idx: usize, ip: usize, message: String) -> TenthError {
        TenthError::RuntimeError { line: self.chunks[chunk_idx].line_at(ip), col: None, message }
    }
}
