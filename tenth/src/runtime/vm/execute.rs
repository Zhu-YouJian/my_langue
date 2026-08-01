//! VM 调度器、执行循环、私有算术、字段访问、autodiff 记录。
//!
//! 从 runtime/vm.rs 拆分而来（T3b 架构重构）。

use std::collections::HashMap;
use crate::hir::types::BaseType;
use std::rc::Rc;
use std::cell::RefCell;
use crate::error::{TenthError, TenthResult};
use crate::runtime::value::{Value, FutureState, check_int_overflow};
use crate::runtime::autodiff::TapeOp;
use crate::runtime::tensor::Tensor;
use crate::runtime::async_io::ASYNC_IO;

use super::Vm;
use super::err;
use super::op::{Op, Frame, YieldReason};

impl Vm {
    /// Phase 2 调度器主循环（Step 3-4）。
    /// 主任务（task_id=0）入队 → 循环取任务 → 恢复 → run_until_yield → 处理结果。
    /// 主任务完成时返回结果；其他任务完成时把结果写入 task_results 并唤醒等待者。
    /// Step 3-4 中 spawn 仍为 eager（不创建 task_futures 条目），真正的并发来自
    /// Step 5 的 async I/O 创建 Pending Future 触发挂起。
    pub(super) fn run_scheduler(&mut self, entry_chunk: usize) -> TenthResult<Value> {
        // 清理上次 run 残留的 async I/O 状态（thread_local 跨调用持久，需显式清理）
        ASYNC_IO.with(|io| io.borrow_mut().clear());

        // 创建主任务的初始 Frame 并推入 suspended（统一恢复路径）
        let num_args = self.chunks[entry_chunk].num_args;
        let num_locals = self.chunks[entry_chunk].num_locals;
        let mut locals = vec![Value::Unit; num_locals.max(num_args)];
        let base = self.stack.len().saturating_sub(num_args);
        for i in (0..num_args).rev() {
            if self.stack.len() > base {
                locals[i] = self.stack.pop().unwrap();
            }
        }
        let initial_frame = Frame {
            ip: 0,
            chunk_idx: entry_chunk,
            locals,
            stack_base: 0,
            operand_stack: std::mem::take(&mut self.stack),
            task_id: 0,
        };
        self.suspended.insert(0, vec![initial_frame]);
        self.ready_queue.push_back(0);

        loop {
            // Phase 2 Step 5：轮询 async I/O，把就绪的 Future 设为 Ready，
            // 唤醒等待者（waiters）推入 ready_queue。
            let woken = ASYNC_IO.with(|io| io.borrow_mut().poll());
            for tid in woken {
                self.ready_queue.push_back(tid);
            }

            let task_id = match self.ready_queue.pop_front() {
                Some(id) => id,
                None => {
                    // 无就绪任务。若仍有 pending I/O，短暂休眠后继续轮询
                    // （避免忙等；1ms 粒度对当前应用足够）。
                    // 若无 pending I/O，所有任务已完成或死锁，退出调度器。
                    let has_pending = ASYNC_IO.with(|io| io.borrow().has_pending());
                    if has_pending {
                        std::thread::sleep(std::time::Duration::from_millis(1));
                        continue;
                    }
                    break;
                }
            };

            // 恢复任务上下文：从 suspended 取出完整调用栈
            if let Some(frames) = self.suspended.remove(&task_id) {
                self.frames = frames;
            } else {
                // 任务不在 suspended 中（可能已完成或无效），跳过
                continue;
            }

            self.current_task = task_id;

            match self.run_until_yield()? {
                YieldReason::Completed => {
                    let result = self.stack.pop().unwrap_or(Value::Unit);
                    self.task_results.insert(task_id, result.clone());

                    // 如果有关联 Future，设为 Ready 并唤醒等待者
                    //（Step 3-4 中 task_futures 为空，此分支不触发；
                    //  Step 5 的 async I/O 任务完成时走此路径唤醒 await 者）
                    if let Some(fut) = self.task_futures.remove(&task_id) {
                        let waiters = {
                            let mut state = fut.borrow_mut();
                            let ws = match &*state {
                                FutureState::Pending(ws) => ws.clone(),
                                _ => vec![],
                            };
                            *state = FutureState::Ready(result.clone());
                            ws
                        };
                        for w in waiters {
                            self.ready_queue.push_back(w);
                        }
                    }

                    if task_id == 0 {
                        return Ok(result);
                    }
                }
                YieldReason::Suspended(_) => {
                    // frames 已在 run_until_yield 中保存到 suspended
                    // task_id 不自动重新入队——待 Future 就绪时由调度器唤醒
                }
                YieldReason::Yield(_) => {
                    // frames 已在 run_until_yield 中保存到 suspended
                    // task_id 已在 run_until_yield 中推回 ready_queue
                }
            }
        }

        Ok(self.task_results.get(&0).cloned().unwrap_or(Value::Unit))
    }

    /// 执行当前任务直到 yield（Step 3-4）。
    /// 入口：从 `self.frames.pop()` 取出当前帧，恢复局部状态。
    /// 出口：
    /// - `Completed`：顶层 Ret，结果已 push 到 `self.stack`，调度器负责 pop
    /// - `Suspended(task_id)`：await 遇到 Pending Future，调用栈已保存到 `suspended[task_id]`
    /// - `Yield(task_id)`：主动让出，调用栈已保存到 `suspended[task_id]`，task_id 已推回 ready_queue
    fn run_until_yield(&mut self) -> TenthResult<YieldReason> {
        // 从 self.frames 恢复当前帧（调度器已把完整调用栈放入 self.frames）
        let frame = match self.frames.pop() {
            Some(f) => f,
            None => return Ok(YieldReason::Completed),
        };
        let mut ip = frame.ip;
        let mut chunk_idx = frame.chunk_idx;
        let mut locals = frame.locals;
        let mut base = frame.stack_base;
        let mut current_task_id = frame.task_id;
        self.stack = frame.operand_stack;
        self.current_task = current_task_id;

        let mut code = self.chunks[chunk_idx].code.clone();
        let mut strings = self.chunks[chunk_idx].strings.clone();

        // H-4: 独立的循环计数器，用于触发周期性 deadline 检查。
        // 不依赖 step_budget（用户可能只设 --timeout 而不设步数预算）。
        let mut loop_counter: u64 = 0;

        loop {
            // 安全 H-4：step_budget 和 deadline_ms 独立检查。
            // 历史实现把 deadline 检查嵌套在 step_budget 内，导致只设
            // `--timeout` 而不设 step_budget 时 deadline 永远不触发。
            if let Some(ref mut budget) = self.step_budget {
                if *budget == 0 {
                    return Err(TenthError::Timeout {
                        message: "VM 步数预算耗尽".into(),
                    });
                }
                *budget -= 1;
            }
            // 每隔 4096 次循环检查一次墙钟 deadline，开销可忽略。
            // 用独立计数器避免依赖 step_budget（step_budget 可能未设）。
            loop_counter = loop_counter.wrapping_add(1);
            if (loop_counter & 0xFFF) == 0 {
                if let Some(deadline) = self.deadline_ms {
                    let now = std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .map(|d| d.as_millis())
                        .unwrap_or(0);
                    if now >= deadline {
                        return Err(TenthError::Timeout {
                            message: "VM 时间预算耗尽".into(),
                        });
                    }
                }
            }
            // Inline opcode read (no closure, so code/strings can be reassigned)
            let op: Op = {
                use Op::*;
                if ip >= code.len() {
                    // 代码执行完毕（隐式 Ret）：任务完成，结果为 Unit
                    self.stack.push(Value::Unit);
                    return Ok(YieldReason::Completed);
                }
                let b = code[ip]; ip += 1;
                macro_rules! r { ($t:ty) => {{ let n = std::mem::size_of::<$t>(); if ip + n > code.len() { self.stack.push(Value::Unit); return Ok(YieldReason::Completed); } let mut buf = [0u8; std::mem::size_of::<$t>()]; buf.copy_from_slice(&code[ip..ip+n]); ip += n; <$t>::from_le_bytes(buf) }}; }
                match b {
                    0 => PushInt(r!(i64)), 1 => PushFloat(r!(f64)),
                    2 => PushBool({ let v = code[ip] != 0; ip += 1; v }),
                    3 => PushStr(r!(u64) as usize),
                    4 => PushUnit, 5 => Pop, 6 => Dup,
                    7 => Load(r!(u64) as usize), 8 => Store(r!(u64) as usize),
                    9 => LoadGlobal(r!(u64) as usize), 10 => StoreGlobal(r!(u64) as usize),
                    11 => Add, 12 => Sub, 13 => Mul, 14 => Div, 15 => Mod,
                    16 => Neg, 17 => Not,
                    18 => Eq, 19 => Neq, 20 => Lt, 21 => Gt, 22 => Lte, 23 => Gte,
                    24 => Jump(r!(i32)), 25 => JmpFalse(r!(i32)), 26 => JmpTrue(r!(i32)),
                    27 => Call(r!(u64) as usize), 28 => CallN(r!(u64) as usize, r!(u64) as usize),
                    29 => MethodCall(r!(u64) as usize, r!(u64) as usize), 30 => Ret,
                    31 => MakeVec(r!(u64) as usize), 32 => MakeMap(r!(u64) as usize),
                    33 => NewStruct(r!(u64) as usize, r!(u64) as usize),
                    34 => LoadField(r!(u64) as usize),
                    35 => StoreField(r!(u64) as usize),
                    56 => NewUnion(r!(u64) as usize, r!(u64) as usize),
                    36 => IndexGet,
                    37 => SliceStr,
                    38 => MakeEnum(r!(u64) as usize, r!(u64) as usize, r!(u64) as usize),
                    39 => IsEnumVariant(r!(u64) as usize),
                    40 => EnumGetField(r!(u64) as usize),
                    41 => PushRange(r!(i64), r!(i64), { let b = code[ip]; ip += 1; b != 0 }),
                    42 => MoveOp,
                    43 => MakeTensor(r!(u64) as usize, r!(u64) as usize, { let d = code[ip]; ip += 1; d }),
                    44 => MakeClosure(r!(u64) as usize, r!(u64) as usize),
                    45 => PushFloat32(r!(f32)),
                    46 => IsStruct(r!(u64) as usize),
                    47 => Await,
                    48 => Spawn,
                    49 => MakeTuple(r!(u64) as usize),
                    50 => IsTuple(r!(u64) as usize),
                    51 => TupleGet(r!(u64) as usize),
                    52 => Try,
                    53 => Yield,
                    54 => PushChar(r!(u32)),
                    55 => TailCall(r!(u64) as usize, r!(u64) as usize),
                    _ => Ret,
                }
            };
            match op {
                Op::PushInt(n) => self.stack.push(Value::Int(n, BaseType::I32)),
                Op::PushFloat(f) => self.stack.push(Value::Float(f)),
                Op::PushFloat32(f) => self.stack.push(Value::Float32(f)),
                Op::PushBool(b) => self.stack.push(Value::Bool(b)),
                Op::PushChar(c) => {
                    let ch = char::from_u32(c).unwrap_or('\0');
                    self.stack.push(Value::Char(ch));
                }
                Op::PushStr(i) => {
                    let s = strings.get(i).cloned().unwrap_or_default();
                    self.stack.push(Value::String(s));
                }
                Op::PushUnit => self.stack.push(Value::Unit),
                Op::Pop => { self.stack.pop(); }
                Op::Dup => {
                    let v = self.stack.last().cloned().unwrap_or(Value::Unit);
                    self.stack.push(v);
                }

                Op::Load(i) => {
                    let v = locals.get(i).cloned().unwrap_or(Value::Unit);
                    self.stack.push(v);
                }
                Op::Store(i) => {
                    let v = self.stack.pop().unwrap_or(Value::Unit);
                    if i >= locals.len() { locals.resize(i+1, Value::Unit); }
                    locals[i] = v;
                }
                Op::LoadGlobal(i) => {
                    let name = strings.get(i).cloned().unwrap_or_default();
                    let v = self.globals.get(&name).cloned().unwrap_or(Value::Unit);
                    self.stack.push(v);
                }
                Op::StoreGlobal(i) => {
                    let name = strings.get(i).cloned().unwrap_or_default();
                    let v = self.stack.pop().unwrap_or(Value::Unit);
                    self.globals.insert(name, v);
                }

                Op::Add => { let (a,b)=self.pop2(); let r=self.add_priv(&a,&b)?; self.stack.push(r); }
                Op::Sub => { let (a,b)=self.pop2(); let r=self.sub_priv(&a,&b)?; self.stack.push(r); }
                Op::Mul => { let (a,b)=self.pop2(); let r=self.mul_priv(&a,&b)?; self.stack.push(r); }
                Op::Div => { let (a,b)=self.pop2(); let r=self.div_priv(&a,&b)?; self.stack.push(r); }
                Op::Mod => {
                    let b = self.pop_int()?; let a = self.pop_int()?;
                    if b == 0 {
                        return err("整数取模除零");
                    }
                    self.stack.push(Value::Int(a % b, BaseType::I32));
                    // 注：Mod 指令通过 pop_int 获取值，丢失 dtype，默认 I32
                }
                Op::Neg => {
                    let v = self.stack.pop().unwrap_or(Value::Unit);
                    match v {
                        Value::Int(n, dt) => {
                            check_int_overflow(-n, dt)?;
                            self.stack.push(Value::Int(-n, dt));
                        }
                        Value::Float(n) => self.stack.push(Value::Float(-n)),
                        Value::Float32(n) => self.stack.push(Value::Float32(-n)),
                        Value::Tensor(t) => self.stack.push(Value::Tensor(Rc::new(RefCell::new(t.borrow().neg())))),
                        _ => return err("无法取负"),
                    }
                }
                Op::Not => {
                    let v = self.stack.pop().unwrap_or(Value::Unit);
                    self.stack.push(Value::Bool(!v.is_truthy()));
                }

                Op::Eq => { let (a,b)=self.pop2(); self.stack.push(Value::Bool(self.vm_eq(&a,&b))); }
                Op::Neq => { let (a,b)=self.pop2(); self.stack.push(Value::Bool(!self.vm_eq(&a,&b))); }
                Op::Lt => { let (a,b)=self.pop2(); self.stack.push(Value::Bool(self.compare(&a,&b,|x,y|x<y,|x,y|x<y)?)); }
                Op::Gt => { let (a,b)=self.pop2(); self.stack.push(Value::Bool(self.compare(&a,&b,|x,y|x>y,|x,y|x>y)?)); }
                Op::Lte => { let (a,b)=self.pop2(); self.stack.push(Value::Bool(self.compare(&a,&b,|x,y|x<=y,|x,y|x<=y)?)); }
                Op::Gte => { let (a,b)=self.pop2(); self.stack.push(Value::Bool(self.compare(&a,&b,|x,y|x>=y,|x,y|x>=y)?)); }

                Op::Jump(o) => { ip = (ip as i32 + o) as usize; }
                Op::JmpFalse(o) => {
                    if !self.stack.pop().unwrap_or(Value::Unit).is_truthy() {
                        ip = (ip as i32 + o) as usize;
                    }
                }
                Op::JmpTrue(o) => {
                    if self.stack.pop().unwrap_or(Value::Unit).is_truthy() {
                        ip = (ip as i32 + o) as usize;
                    }
                }

                Op::Call(i) => {
                    let name = strings[i].clone();
                    // Try native first (legacy: uses stack depth as arg count)
                    if let Some(native_fn) = self.natives.get(&name).copied() {
                        let n = self.stack.len() - base;
                        let mut args = vec![Value::Unit; n];
                        for i in (0..n).rev() { args[i] = self.stack.pop().unwrap_or(Value::Unit); }
                        let result = native_fn(self, &args)?;
                        self.stack.push(result);
                    } else if let Some(&callee_idx) = self.functions.get(&name) {
                        let callee_args = self.chunks[callee_idx].num_args;
                        let callee_locals = self.chunks[callee_idx].num_locals;
                        // Phase 2 栈迁移：先从 caller 栈 pop args 到 callee locals
                        let mut new_locals = vec![Value::Unit; callee_locals.max(callee_args)];
                        for i in (0..callee_args).rev() {
                            if self.stack.len() > base { new_locals[i] = self.stack.pop().unwrap(); }
                        }
                        // 保存 caller 栈到 Frame，给 callee 一个空栈
                        let caller_stack = std::mem::take(&mut self.stack);
                        self.frames.push(Frame {
                            ip,
                            chunk_idx,
                            locals: locals.clone(),
                            stack_base: base,
                            operand_stack: caller_stack,
                            task_id: current_task_id,
                        });
                        chunk_idx = callee_idx;
                        code = self.chunks[chunk_idx].code.clone();
                        strings = self.chunks[chunk_idx].strings.clone();
                        ip = 0;
                        locals = new_locals;
                        base = 0;  // callee 新栈从 0 开始
                    } else {
                        return Err(TenthError::RuntimeError { line: None, col: None, message: format!("未定义的函数 '{}'", name) });
                    }
                }
                Op::CallN(i, num_args) => {
                    let name = strings.get(i).cloned().unwrap_or_default();
                    let n = num_args;
                    let mut args = vec![Value::Unit; n];
                    for i in (0..n).rev() { args[i] = self.stack.pop().unwrap_or(Value::Unit); }

                    // Try to find the function by name, checking globals for FnRef closures
                    let callee_name = if let Some(Value::FnRef { name: fname, .. }) = self.globals.get(&name) {
                        fname.clone()
                    } else {
                        name.clone()
                    };

                    if let Some(native_fn) = self.natives.get(&callee_name).copied() {
                        let result = native_fn(self, &args)?;
                        self.stack.push(result);
                    } else if let Some(&callee_idx) = self.functions.get(&callee_name) {
                        // Phase 2 栈迁移：保存 caller 栈到 Frame，给 callee 一个空栈
                        let caller_stack = std::mem::take(&mut self.stack);
                        self.frames.push(Frame {
                            ip,
                            chunk_idx,
                            locals: locals.clone(),
                            stack_base: base,
                            operand_stack: caller_stack,
                            task_id: current_task_id,
                        });
                        chunk_idx = callee_idx;
                        code = self.chunks[chunk_idx].code.clone();
                        strings = self.chunks[chunk_idx].strings.clone();
                        ip = 0;
                        locals = args;
                        locals.resize(self.chunks[chunk_idx].num_locals.max(locals.len()), Value::Unit);
                        base = 0;  // callee 新栈从 0 开始
                    } else if let Some(native_fn) = self.natives.get(&name).copied() {
                        let result = native_fn(self, &args)?;
                        self.stack.push(result);
                    } else if let Some(&callee_idx) = self.functions.get(&name) {
                        // Phase 2 栈迁移：保存 caller 栈到 Frame，给 callee 一个空栈
                        let caller_stack = std::mem::take(&mut self.stack);
                        self.frames.push(Frame {
                            ip,
                            chunk_idx,
                            locals: locals.clone(),
                            stack_base: base,
                            operand_stack: caller_stack,
                            task_id: current_task_id,
                        });
                        chunk_idx = callee_idx;
                        code = self.chunks[chunk_idx].code.clone();
                        strings = self.chunks[chunk_idx].strings.clone();
                        ip = 0;
                        locals = args;
                        locals.resize(self.chunks[chunk_idx].num_locals.max(locals.len()), Value::Unit);
                        base = 0;  // callee 新栈从 0 开始
                    } else {
                        return Err(TenthError::RuntimeError { line: None, col: None, message: format!("未定义的函数 '{}'", name) });
                    }
                }
                Op::TailCall(i, num_args) => {
                    // TCO：复用当前帧，不压新帧
                    let name = strings.get(i).cloned().unwrap_or_default();
                    let n = num_args;
                    let mut args = vec![Value::Unit; n];
                    for i in (0..n).rev() { args[i] = self.stack.pop().unwrap_or(Value::Unit); }

                    // 查找函数（同 CallN）
                    let callee_name = if let Some(Value::FnRef { name: fname, .. }) = self.globals.get(&name) {
                        fname.clone()
                    } else {
                        name.clone()
                    };

                    if let Some(native_fn) = self.natives.get(&callee_name).copied() {
                        let result = native_fn(self, &args)?;
                        self.stack.push(result);
                    } else if let Some(&callee_idx) = self.functions.get(&callee_name) {
                        // TCO：不压帧，直接替换当前帧状态
                        chunk_idx = callee_idx;
                        code = self.chunks[chunk_idx].code.clone();
                        strings = self.chunks[chunk_idx].strings.clone();
                        ip = 0;
                        locals = args;
                        locals.resize(self.chunks[chunk_idx].num_locals.max(locals.len()), Value::Unit);
                        // base 不重置 — 当前栈帧继续使用
                    } else if let Some(native_fn) = self.natives.get(&name).copied() {
                        let result = native_fn(self, &args)?;
                        self.stack.push(result);
                    } else if let Some(&callee_idx) = self.functions.get(&name) {
                        // TCO：不压帧，直接替换当前帧状态
                        chunk_idx = callee_idx;
                        code = self.chunks[chunk_idx].code.clone();
                        strings = self.chunks[chunk_idx].strings.clone();
                        ip = 0;
                        locals = args;
                        locals.resize(self.chunks[chunk_idx].num_locals.max(locals.len()), Value::Unit);
                    } else {
                        return Err(TenthError::RuntimeError { line: None, col: None, message: format!("未定义的函数 '{}'", name) });
                    }
                }
                Op::MethodCall(i, num_args) => {
                    let name = strings.get(i).cloned().unwrap_or_default();
                    let n = num_args;
                    let mut args = vec![Value::Unit; n];
                    for i in (0..n).rev() { args[i] = self.stack.pop().unwrap_or(Value::Unit); }
                    let receiver = self.stack.pop().unwrap_or(Value::Unit);
                    let result = self.call_method_priv(&receiver, &name, &args)?;
                    self.stack.push(result);
                }

                Op::Ret => {
                    let result = self.stack.pop().unwrap_or(Value::Unit);
                    if let Some(f) = self.frames.pop() {
                        // Phase 2 栈迁移：恢复 caller 栈（丢弃 callee 栈），push 结果
                        self.stack = f.operand_stack;
                        self.stack.push(result);
                        ip = f.ip;
                        chunk_idx = f.chunk_idx;
                        code = self.chunks[chunk_idx].code.clone();
                        strings = self.chunks[chunk_idx].strings.clone();
                        locals = f.locals;
                        base = f.stack_base;
                        current_task_id = f.task_id;
                        self.current_task = current_task_id;
                    } else {
                        // 顶层 Ret：任务完成。把结果推回栈供调度器 pop。
                        self.stack.push(result);
                        return Ok(YieldReason::Completed);
                    }
                }

                Op::MakeVec(n) => {
                    let mut v = Vec::new();
                    for _ in 0..n { v.push(self.stack.pop().unwrap_or(Value::Unit)); }
                    v.reverse();
                    self.stack.push(Value::Vec(Rc::new(RefCell::new(v))));
                }
                Op::MakeMap(n) => {
                    let mut m = HashMap::new();
                    for _ in 0..n {
                        let val = self.stack.pop().unwrap_or(Value::Unit);
                        let key = match self.stack.pop().unwrap_or(Value::Unit) {
                            Value::String(s) => s,
                            _ => String::new(),
                        };
                        m.insert(key, val);
                    }
                    self.stack.push(Value::Map(Rc::new(RefCell::new(m))));
                }

                Op::NewStruct(name_i, n) => {
                    let name = strings.get(name_i).cloned().unwrap_or_default();
                    let mut fields = Vec::new();
                    for _ in 0..n {
                        // Compiler pushes value then name (name on top); pop name first
                        let fname = match self.stack.pop().unwrap_or(Value::Unit) {
                            Value::String(s) => s,
                            _ => String::new(),
                        };
                        let val = self.stack.pop().unwrap_or(Value::Unit);
                        fields.push((fname, val));
                    }
                    fields.reverse();
                    self.stack.push(Value::Struct { name, fields: Rc::new(RefCell::new(fields)) });
                }

                // M1.2：union 构造 — 弹出栈顶 value，构造带 active_field 的 tagged union
                Op::NewUnion(name_i, field_i) => {
                    let name = strings.get(name_i).cloned().unwrap_or_default();
                    let active_field = strings.get(field_i).cloned().unwrap_or_default();
                    let value = self.stack.pop().unwrap_or(Value::Unit);
                    self.stack.push(Value::Union { name, active_field, value: Box::new(value) });
                }

                Op::LoadField(i) => {
                    let fname = strings.get(i).cloned().unwrap_or_default();
                    let val = self.stack.pop().unwrap_or(Value::Unit);
                    let v = self.get_field(&val, &fname)?;
                    self.stack.push(v);
                }

                Op::StoreField(i) => {
                    let fname = strings.get(i).cloned().unwrap_or_default();
                    let new_val = self.stack.pop().unwrap_or(Value::Unit);
                    let target = self.stack.pop().unwrap_or(Value::Unit);
                    // M1.2：Union 字段修改（tagged union）——只允许修改 active 字段，
                    // 构造新 Value::Union 推回栈（bytecode 对 Union 目标随后 Store 写回变量槽）。
                    match &target {
                        Value::Union { name, active_field, .. } => {
                            if active_field == &fname {
                                self.stack.push(Value::Union {
                                    name: name.clone(),
                                    active_field: active_field.clone(),
                                    value: Box::new(new_val),
                                });
                            } else {
                                return Err(TenthError::RuntimeError { line: None, col: None,
                                    message: format!(
                                        "union '{}' 当前活跃字段是 '{}'，不能修改非活跃字段 '{}'",
                                        name, active_field, fname
                                    ),
                                });
                            }
                        }
                        _ => {
                            self.set_field(&target, &fname, new_val)?;
                        }
                    }
                }

                Op::IndexGet => {
                    let idx = self.stack.pop().unwrap_or(Value::Unit);
                    let target = self.stack.pop().unwrap_or(Value::Unit);
                    match target {
                        Value::Vec(items) => {
                            let i = idx.as_int().unwrap_or(0) as usize;
                            let v = items.borrow().get(i).cloned().unwrap_or(Value::Unit);
                            self.stack.push(v);
                        }
                        Value::String(s) => {
                            let i = idx.as_int().unwrap_or(0) as usize;
                            let c = s.chars().nth(i).map(|c| c.to_string()).unwrap_or_default();
                            self.stack.push(Value::String(c));
                        }
                        Value::Tensor(t) => {
                            // NumPy 语义：单索引沿第 0 维降维。
                            // - 1D 张量 t[i] → 标量 Value::Float
                            // - N-D 张量 t[i] → (N-1)-D 子张量 Value::Tensor
                            let i = idx.as_int().unwrap_or(0) as usize;
                            let tensor = t.borrow();
                            if tensor.ndim() <= 1 {
                                match tensor.get(&[i]) {
                                    Some(val) => {
                                        drop(tensor);
                                        self.stack.push(Value::Float(val));
                                    }
                                    None => {
                                        return err(&format!(
                                            "索引 {} 越界，形状为 {:?}",
                                            i, tensor.shape()
                                        ));
                                    }
                                }
                            } else {
                                match tensor.index_dim(i) {
                                    Ok(sub) => {
                                        drop(tensor);
                                        self.stack.push(Value::Tensor(Rc::new(RefCell::new(sub))));
                                    }
                                    Err(msg) => return err(&msg),
                                }
                            }
                        }
                        _ => return err("无法索引"),
                    }
                }

                Op::SliceStr => {
                    let end_idx = self.pop_int()? as usize;
                    let start_idx = self.pop_int()? as usize;
                    let target = self.stack.pop().unwrap_or(Value::Unit);
                    match target {
                        Value::String(s) => {
                            let chars: Vec<char> = s.chars().collect();
                            let len = chars.len();
                            let si = start_idx.min(len);
                            let ei = end_idx.min(len);
                            if si > ei {
                                return err("字符串切片起始位置大于结束位置");
                            }
                            let slice: String = chars[si..ei].iter().collect();
                            self.stack.push(Value::String(slice));
                        }
                        _ => return err("SliceStr 需要字符串目标"),
                    }
                }

                Op::MakeEnum(name_i, variant_i, n) => {
                    let enum_name = strings.get(name_i).cloned().unwrap_or_default();
                    let variant = strings.get(variant_i).cloned().unwrap_or_default();
                    let mut fields = Vec::new();
                    for _ in 0..n {
                        let fname = match self.stack.pop().unwrap_or(Value::Unit) {
                            Value::String(s) => s,
                            _ => String::new(),
                        };
                        let val = self.stack.pop().unwrap_or(Value::Unit);
                        fields.push((fname, val));
                    }
                    // 不 reverse：bytecode 按逆序压 [value,name] 对，循环 pop 后
                    // fields 已按源码声明顺序排列（f0 在首）。reverse 会使字段序
                    // 变为反源码序，与解释器/JIT（host_make_enum）不一致，导致
                    // or_die/assume_ok 的 .first() 与 Display 顺序错位
                    // （2026-08-01 JIT enum 字段颠倒 bug 修复时统一）。
                    self.stack.push(Value::Enum {
                        enum_name,
                        variant,
                        fields: Rc::new(RefCell::new(fields)),
                    });
                }

                Op::IsEnumVariant(variant_i) => {
                    let variant_name = strings.get(variant_i).cloned().unwrap_or_default();
                    let val = self.stack.pop().unwrap_or(Value::Unit);
                    let matches = match &val {
                        Value::Enum { variant, .. } => variant == &variant_name,
                        _ => false,
                    };
                    self.stack.push(Value::Bool(matches));
                }

                Op::IsStruct(name_i) => {
                    let struct_name = strings.get(name_i).cloned().unwrap_or_default();
                    let val = self.stack.pop().unwrap_or(Value::Unit);
                    let matches = match &val {
                        Value::Struct { name, .. } => name == &struct_name,
                        _ => false,
                    };
                    self.stack.push(Value::Bool(matches));
                }

                Op::EnumGetField(field_i) => {
                    let field_name = strings.get(field_i).cloned().unwrap_or_default();
                    let val = self.stack.pop().unwrap_or(Value::Unit);
                    let found = match val {
                        Value::Enum { fields, .. } => {
                            let mut result = None;
                            for (n, v) in fields.borrow().iter() {
                                if n == &field_name {
                                    result = Some(v.clone());
                                    break;
                                }
                            }
                            result
                        }
                        _ => None,
                    };
                    match found {
                        Some(v) => self.stack.push(v),
                        None => self.stack.push(Value::Unit),
                    }
                }

                Op::PushRange(start, end, inclusive) => {
                    self.stack.push(Value::Range { start, end, inclusive });
                }

                Op::MoveOp => {
                    // no-op: move semantics are checked at HIR level
                }

                Op::MakeTensor(rows, cols, dtype) => {
                    use crate::hir::types::BaseType;
                    use half::{bf16, f16};
                    let total = rows * cols;
                    let dt = match dtype {
                        1 => BaseType::F32,
                        2 => BaseType::F16,
                        3 => BaseType::BF16,
                        _ => BaseType::F64,
                    };
                    if dt == BaseType::F32 {
                        let mut data: Vec<f32> = Vec::with_capacity(total);
                        for _ in 0..total {
                            let v = self.stack.pop().unwrap_or(Value::Float32(0.0));
                            data.push(match v {
                                Value::Float32(f) => f,
                                Value::Float(f) => f as f32,
                                Value::Int(n, _) => n as f32,
                                _ => 0.0,
                            });
                        }
                        data.reverse();
                        let tensor = Tensor::from_vec_f32(data, vec![rows, cols]);
                        self.stack.push(Value::Tensor(Rc::new(RefCell::new(tensor))));
                    } else if dt == BaseType::F16 {
                        let mut data: Vec<f16> = Vec::with_capacity(total);
                        for _ in 0..total {
                            let v = self.stack.pop().unwrap_or(Value::Float(0.0));
                            data.push(match v {
                                Value::Float(f) => f16::from_f64(f),
                                Value::Float32(f) => f16::from_f32(f),
                                Value::Int(n, _) => f16::from_f64(n as f64),
                                _ => f16::from_f32(0.0),
                            });
                        }
                        data.reverse();
                        let tensor = Tensor::from_vec_f16(data, vec![rows, cols]);
                        self.stack.push(Value::Tensor(Rc::new(RefCell::new(tensor))));
                    } else if dt == BaseType::BF16 {
                        let mut data: Vec<bf16> = Vec::with_capacity(total);
                        for _ in 0..total {
                            let v = self.stack.pop().unwrap_or(Value::Float(0.0));
                            data.push(match v {
                                Value::Float(f) => bf16::from_f64(f),
                                Value::Float32(f) => bf16::from_f32(f),
                                Value::Int(n, _) => bf16::from_f64(n as f64),
                                _ => bf16::from_f32(0.0),
                            });
                        }
                        data.reverse();
                        let tensor = Tensor::from_vec_bf16(data, vec![rows, cols]);
                        self.stack.push(Value::Tensor(Rc::new(RefCell::new(tensor))));
                    } else {
                        let mut data = Vec::with_capacity(total);
                        for _ in 0..total {
                            let v = self.stack.pop().unwrap_or(Value::Float(0.0));
                            data.push(match v {
                                Value::Float(f) => f,
                                Value::Float32(f) => f as f64,
                                Value::Int(n, _) => n as f64,
                                _ => 0.0,
                            });
                        }
                        data.reverse();
                        let tensor = Tensor::from_vec(data, vec![rows, cols]);
                        self.stack.push(Value::Tensor(Rc::new(RefCell::new(tensor))));
                    }
                }

                Op::MakeClosure(params_count, name_idx) => {
                    // Create a FnRef value pointing to the closure function
                    let name = strings.get(name_idx).cloned().unwrap_or_default();
                    let param_names: Vec<(String, crate::hir::types::Type)> = (0..params_count)
                        .map(|i| (format!("__param_{i}"), crate::hir::types::Type::Unknown))
                        .collect();
                    self.stack.push(Value::FnRef {
                        name,
                        params: param_names,
                        return_type: crate::hir::types::Type::Unknown,
                    });
                }

                Op::Spawn => {
                    let v = self.stack.pop().unwrap_or(Value::Unit);
                    // Phase 2 Step 3-4：spawn 保持 eager 语义（立即求值，包装为 Ready Future）。
                    // 设计决策：真正的并发不来自 spawn，而来自 Step 5 的 async I/O 创建 Pending Future。
                    // spawn 的 inner 在 bytecode 层面已被编译为"先求值 inner 再 Op::Spawn"，
                    // 改为延迟执行需要重构 bytecode 编译逻辑，超出 Step 3-4 范围。
                    // eager spawn 仍然有用：await 时如果遇到 Pending Future（由 async I/O 创建），
                    // 调度器会切换到其他就绪任务，包括其他 spawn 产生的 eager Future 的 await 者。
                    self.stack.push(Value::future_ready(v));
                }

                Op::Await => {
                    let v = self.stack.pop().unwrap_or(Value::Unit);
                    // Phase 2 Step 3-4：检查 FutureState
                    // - Ready：直接取值（快路径，Phase 1 兼容）
                    // - Pending：把当前 task_id 加入 Future 等待者列表，
                    //   保存当前调用栈到 suspended，返回 Suspended 信号给调度器
                    match &v {
                        Value::Future(rc) => {
                            // clone Rc 以便后续 borrow_mut（避免在 match 借用中修改）
                            let rc_clone = rc.clone();
                            let is_ready = matches!(&*rc_clone.borrow(), FutureState::Ready(_));
                            if is_ready {
                                let inner = match &*rc_clone.borrow() {
                                    FutureState::Ready(v) => v.clone(),
                                    _ => unreachable!(),
                                };
                                self.stack.push(inner);
                            } else {
                                // Pending：把当前 task_id 加入 Future 的等待者列表
                                if let FutureState::Pending(waiters) = &mut *rc_clone.borrow_mut() {
                                    waiters.push(current_task_id);
                                }
                                // 把 Future 推回栈顶，以便恢复时重新执行 Await 取值。
                                // Await 无操作数（1 字节 opcode），ip-1 指向 Await opcode，
                                // 恢复后重读该字节并重新执行：此时 Future 已 Ready，走快路径取值。
                                self.stack.push(v.clone());
                                self.frames.push(Frame {
                                    ip: ip - 1,
                                    chunk_idx,
                                    locals: std::mem::take(&mut locals),
                                    stack_base: base,
                                    operand_stack: std::mem::take(&mut self.stack),
                                    task_id: current_task_id,
                                });
                                let frames = std::mem::take(&mut self.frames);
                                self.suspended.insert(current_task_id, frames);
                                return Ok(YieldReason::Suspended(current_task_id));
                            }
                        }
                        other => {
                            // 非 Future 值：直接传值（await 7 → 7）
                            self.stack.push(other.clone());
                        }
                    }
                }

                Op::MakeTuple(n) => {
                    // Pop n values (last pushed = highest index) and assemble Tuple
                    let mut items = Vec::with_capacity(n);
                    for _ in 0..n {
                        items.push(self.stack.pop().unwrap_or(Value::Unit));
                    }
                    items.reverse();
                    self.stack.push(Value::Tuple(items));
                }

                Op::IsTuple(expected_len) => {
                    // Pop value, push Bool(val is Tuple with expected_len).
                    // Mirrors IsEnumVariant: pop + push bool.
                    let val = self.stack.pop().unwrap_or(Value::Unit);
                    let matches = match &val {
                        Value::Tuple(items) => items.len() == expected_len,
                        _ => false,
                    };
                    self.stack.push(Value::Bool(matches));
                }

                Op::TupleGet(i) => {
                    let val = self.stack.pop().unwrap_or(Value::Unit);
                    let elem = match &val {
                        Value::Tuple(items) => items.get(i).cloned().unwrap_or(Value::Unit),
                        _ => Value::Unit,
                    };
                    self.stack.push(elem);
                }

                Op::Try => {
                    // `expr?` — Result::Ok(v) → push v; Result::Err(e) → early return
                    // 语义：若 Err，构造 Result::Err 并执行函数 early return（frame 恢复）。
                    // 这与 interpreter 的 TryPropagate 信号等价，但 VM 直接在内部完成 frame 切换。
                    let val = self.stack.pop().unwrap_or(Value::Unit);
                    let is_err = matches!(
                        &val,
                        Value::Enum { enum_name, variant, .. } if enum_name == "Result" && variant == "Err"
                    );
                    if is_err {
                        // 构造 Result::Err 作为函数返回值，执行 early return（复用 Ret 的 frame 恢复逻辑）
                        let result = val.clone();
                        if let Some(f) = self.frames.pop() {
                            // Phase 2 栈迁移：恢复 caller 栈（同 Op::Ret）
                            self.stack = f.operand_stack;
                            self.stack.push(result);
                            ip = f.ip;
                            chunk_idx = f.chunk_idx;
                            code = self.chunks[chunk_idx].code.clone();
                            strings = self.chunks[chunk_idx].strings.clone();
                            locals = f.locals;
                            base = f.stack_base;
                            current_task_id = f.task_id;
                            self.current_task = current_task_id;
                        } else {
                            // 最外层函数：任务完成（early return Err）。把结果推回栈供调度器 pop。
                            self.stack.push(result);
                            return Ok(YieldReason::Completed);
                        }
                    } else {
                        // Ok(v) → 解包 push v；非 Result 类型直接传值
                        let inner = match &val {
                            Value::Enum { enum_name, variant, fields } if enum_name == "Result" && variant == "Ok" => {
                                fields.borrow().first()
                                    .map(|(_, v)| v.clone())
                                    .unwrap_or(Value::Unit)
                            }
                            _ => val,
                        };
                        self.stack.push(inner);
                    }
                }

                Op::Yield => {
                    // Phase 2 Step 3-4：协作式调度，主动让出控制权。
                    // 保存当前帧到 self.frames，take 整个 frames 到 suspended，
                    // task_id 推回 ready_queue 尾部，返回 Yield 信号给调度器。
                    // 调度器下次轮到该 task 时会从 suspended 恢复并继续执行（ip 不变）。
                    self.frames.push(Frame {
                        ip,
                        chunk_idx,
                        locals: std::mem::take(&mut locals),
                        stack_base: base,
                        operand_stack: std::mem::take(&mut self.stack),
                        task_id: current_task_id,
                    });
                    let frames = std::mem::take(&mut self.frames);
                    self.suspended.insert(current_task_id, frames);
                    self.ready_queue.push_back(current_task_id);
                    return Ok(YieldReason::Yield(current_task_id));
                }
            }
        }
    }

    fn pop2(&mut self) -> (Value, Value) {
        let b = self.stack.pop().unwrap_or(Value::Unit);
        let a = self.stack.pop().unwrap_or(Value::Unit);
        (a, b)
    }

    fn pop_int(&mut self) -> TenthResult<i64> {
        match self.stack.pop() {
            Some(Value::Int(n, _)) => Ok(n),
            _ => err("期望整数"),
        }
    }

    pub(super) fn add_priv(&mut self, a: &Value, b: &Value) -> TenthResult<Value> {
        Ok(match (a, b) {
            (Value::Int(x, dt), Value::Int(y, _)) => { check_int_overflow(x + y, *dt)?; Value::Int(x + y, *dt) },
            (Value::Float(x), Value::Float(y)) => Value::Float(x + y),
            (Value::Int(x, _), Value::Float(y)) => Value::Float(*x as f64 + y),
            (Value::Float(x), Value::Int(y, _)) => Value::Float(x + *y as f64),
            // f32 路径：相同 dtype 保持 f32，混合提升为 f64
            (Value::Float32(x), Value::Float32(y)) => Value::Float32(x + y),
            (Value::Int(x, _), Value::Float32(y)) => Value::Float32(*x as f32 + y),
            (Value::Float32(x), Value::Int(y, _)) => Value::Float32(x + *y as f32),
            (Value::Float32(x), Value::Float(y)) => Value::Float(*x as f64 + y),
            (Value::Float(x), Value::Float32(y)) => Value::Float(x + *y as f64),
            (Value::String(x), Value::String(y)) => Value::String(format!("{x}{y}")),
            (Value::Tensor(t), Value::Float(s)) => {
                let result = Rc::new(RefCell::new(t.borrow().add_scalar(*s)));
                if self.recording {
                    let s_tensor = Rc::new(RefCell::new(Tensor::from_vec(vec![*s], vec![1])));
                    self.record_binary(TapeOp::Add, &t, &s_tensor, &result);
                }
                Value::Tensor(result)
            }
            (Value::Float(s), Value::Tensor(t)) => {
                let result = Rc::new(RefCell::new(t.borrow().add_scalar(*s)));
                if self.recording {
                    let s_tensor = Rc::new(RefCell::new(Tensor::from_vec(vec![*s], vec![1])));
                    self.record_binary(TapeOp::Add, &s_tensor, &t, &result);
                }
                Value::Tensor(result)
            }
            // f32 标量 × Tensor：转为 f64 调用 scalar 方法（scalar 方法按 Tensor dtype 分支保持精度）
            (Value::Tensor(t), Value::Float32(s)) => {
                let result = Rc::new(RefCell::new(t.borrow().add_scalar(*s as f64)));
                if self.recording {
                    let s_tensor = Rc::new(RefCell::new(Tensor::from_vec(vec![*s as f64], vec![1])));
                    self.record_binary(TapeOp::Add, &t, &s_tensor, &result);
                }
                Value::Tensor(result)
            }
            (Value::Float32(s), Value::Tensor(t)) => {
                let result = Rc::new(RefCell::new(t.borrow().add_scalar(*s as f64)));
                if self.recording {
                    let s_tensor = Rc::new(RefCell::new(Tensor::from_vec(vec![*s as f64], vec![1])));
                    self.record_binary(TapeOp::Add, &s_tensor, &t, &result);
                }
                Value::Tensor(result)
            }
            (Value::Tensor(t1), Value::Tensor(t2)) => {
                let result_tensor = t1.borrow().add_tensor(&t2.borrow())
                    .map_err(|msg| TenthError::RuntimeError { line: None, col: None, message: msg })?;
                let result = Rc::new(RefCell::new(result_tensor));
                if self.recording { self.record_binary(TapeOp::Add, &t1, &t2, &result); }
                Value::Tensor(result)
            }
            _ => return err("+ 类型不匹配"),
        })
    }

    pub(super) fn sub_priv(&mut self, a: &Value, b: &Value) -> TenthResult<Value> {
        Ok(match (a, b) {
            (Value::Int(x, dt), Value::Int(y, _)) => { check_int_overflow(x - y, *dt)?; Value::Int(x - y, *dt) },
            (Value::Float(x), Value::Float(y)) => Value::Float(x - y),
            (Value::Int(x, _), Value::Float(y)) => Value::Float(*x as f64 - y),
            (Value::Float(x), Value::Int(y, _)) => Value::Float(x - *y as f64),
            // f32 路径
            (Value::Float32(x), Value::Float32(y)) => Value::Float32(x - y),
            (Value::Int(x, _), Value::Float32(y)) => Value::Float32(*x as f32 - y),
            (Value::Float32(x), Value::Int(y, _)) => Value::Float32(x - *y as f32),
            (Value::Float32(x), Value::Float(y)) => Value::Float(*x as f64 - y),
            (Value::Float(x), Value::Float32(y)) => Value::Float(x - *y as f64),
            (Value::Tensor(t), Value::Float(s)) => {
                let result = Rc::new(RefCell::new(t.borrow().add_scalar(-*s)));
                if self.recording {
                    let s_tensor = Rc::new(RefCell::new(Tensor::from_vec(vec![*s], vec![1])));
                    self.record_binary(TapeOp::Sub, &t, &s_tensor, &result);
                }
                Value::Tensor(result)
            }
            (Value::Float(s), Value::Tensor(t)) => {
                let result = Rc::new(RefCell::new(t.borrow().add_scalar(-*s)));
                if self.recording {
                    let s_tensor = Rc::new(RefCell::new(Tensor::from_vec(vec![*s], vec![1])));
                    self.record_binary(TapeOp::Sub, &s_tensor, &t, &result);
                }
                Value::Tensor(result)
            }
            // f32 标量 × Tensor
            (Value::Tensor(t), Value::Float32(s)) => {
                let s_f64 = *s as f64;
                let result = Rc::new(RefCell::new(t.borrow().add_scalar(-s_f64)));
                if self.recording {
                    let s_tensor = Rc::new(RefCell::new(Tensor::from_vec(vec![s_f64], vec![1])));
                    self.record_binary(TapeOp::Sub, &t, &s_tensor, &result);
                }
                Value::Tensor(result)
            }
            (Value::Float32(s), Value::Tensor(t)) => {
                let s_f64 = *s as f64;
                let result = Rc::new(RefCell::new(t.borrow().add_scalar(-s_f64)));
                if self.recording {
                    let s_tensor = Rc::new(RefCell::new(Tensor::from_vec(vec![s_f64], vec![1])));
                    self.record_binary(TapeOp::Sub, &s_tensor, &t, &result);
                }
                Value::Tensor(result)
            }
            (Value::Tensor(t1), Value::Tensor(t2)) => {
                let result_tensor = t1.borrow().sub_tensor(&t2.borrow())
                    .map_err(|msg| TenthError::RuntimeError { line: None, col: None, message: msg })?;
                let result = Rc::new(RefCell::new(result_tensor));
                if self.recording { self.record_binary(TapeOp::Sub, &t1, &t2, &result); }
                Value::Tensor(result)
            }
            _ => return err("- 类型不匹配"),
        })
    }

    pub(super) fn mul_priv(&mut self, a: &Value, b: &Value) -> TenthResult<Value> {
        Ok(match (a, b) {
            (Value::Int(x, dt), Value::Int(y, _)) => { check_int_overflow(x * y, *dt)?; Value::Int(x * y, *dt) },
            (Value::Float(x), Value::Float(y)) => Value::Float(x * y),
            (Value::Int(x, _), Value::Float(y)) => Value::Float(*x as f64 * y),
            (Value::Float(x), Value::Int(y, _)) => Value::Float(x * *y as f64),
            // f32 路径
            (Value::Float32(x), Value::Float32(y)) => Value::Float32(x * y),
            (Value::Int(x, _), Value::Float32(y)) => Value::Float32(*x as f32 * y),
            (Value::Float32(x), Value::Int(y, _)) => Value::Float32(x * *y as f32),
            (Value::Float32(x), Value::Float(y)) => Value::Float(*x as f64 * y),
            (Value::Float(x), Value::Float32(y)) => Value::Float(x * *y as f64),
            (Value::Tensor(t), Value::Float(s)) => {
                let result = Rc::new(RefCell::new(t.borrow().mul_scalar(*s)));
                if self.recording {
                    let s_tensor = Rc::new(RefCell::new(Tensor::from_vec(vec![*s], vec![1])));
                    self.record_binary(TapeOp::Mul, &t, &s_tensor, &result);
                }
                Value::Tensor(result)
            }
            (Value::Float(s), Value::Tensor(t)) => {
                let result = Rc::new(RefCell::new(t.borrow().mul_scalar(*s)));
                if self.recording {
                    let s_tensor = Rc::new(RefCell::new(Tensor::from_vec(vec![*s], vec![1])));
                    self.record_binary(TapeOp::Mul, &s_tensor, &t, &result);
                }
                Value::Tensor(result)
            }
            // f32 标量 × Tensor
            (Value::Tensor(t), Value::Float32(s)) => {
                let s_f64 = *s as f64;
                let result = Rc::new(RefCell::new(t.borrow().mul_scalar(s_f64)));
                if self.recording {
                    let s_tensor = Rc::new(RefCell::new(Tensor::from_vec(vec![s_f64], vec![1])));
                    self.record_binary(TapeOp::Mul, &t, &s_tensor, &result);
                }
                Value::Tensor(result)
            }
            (Value::Float32(s), Value::Tensor(t)) => {
                let s_f64 = *s as f64;
                let result = Rc::new(RefCell::new(t.borrow().mul_scalar(s_f64)));
                if self.recording {
                    let s_tensor = Rc::new(RefCell::new(Tensor::from_vec(vec![s_f64], vec![1])));
                    self.record_binary(TapeOp::Mul, &s_tensor, &t, &result);
                }
                Value::Tensor(result)
            }
            (Value::Tensor(t1), Value::Tensor(t2)) => {
                let result_tensor = t1.borrow().mul_tensor(&t2.borrow())
                    .map_err(|msg| TenthError::RuntimeError { line: None, col: None, message: msg })?;
                let result = Rc::new(RefCell::new(result_tensor));
                if self.recording { self.record_binary(TapeOp::Mul, &t1, &t2, &result); }
                Value::Tensor(result)
            }
            _ => return err("* 类型不匹配"),
        })
    }

    pub(super) fn div_priv(&mut self, a: &Value, b: &Value) -> TenthResult<Value> {
        Ok(match (a, b) {
            (Value::Int(x, dt), Value::Int(y, _)) => {
                if *y == 0 {
                    return err("整数除零");
                }
                { check_int_overflow(x / y, *dt)?; Value::Int(x / y, *dt) }
            }
            (Value::Float(x), Value::Float(y)) => Value::Float(x / y),
            (Value::Int(x, _), Value::Float(y)) => Value::Float(*x as f64 / y),
            (Value::Float(x), Value::Int(y, _)) => Value::Float(x / *y as f64),
            // f32 路径
            (Value::Float32(x), Value::Float32(y)) => Value::Float32(x / y),
            (Value::Int(x, _), Value::Float32(y)) => Value::Float32(*x as f32 / y),
            (Value::Float32(x), Value::Int(y, _)) => Value::Float32(x / *y as f32),
            (Value::Float32(x), Value::Float(y)) => Value::Float(*x as f64 / y),
            (Value::Float(x), Value::Float32(y)) => Value::Float(x / *y as f64),
            (Value::Tensor(t), Value::Float(s)) => {
                let result = Rc::new(RefCell::new(t.borrow().div_scalar(*s)));
                if self.recording {
                    let s_tensor = Rc::new(RefCell::new(Tensor::from_vec(vec![*s], vec![1])));
                    self.record_binary(TapeOp::Div, &t, &s_tensor, &result);
                }
                Value::Tensor(result)
            }
            (Value::Float(s), Value::Tensor(t)) => {
                // s / t: scalar divided by tensor element-wise
                let result = Rc::new(RefCell::new(t.borrow().div_scalar_inv(*s)));
                if self.recording {
                    let s_tensor = Rc::new(RefCell::new(Tensor::from_vec(vec![*s], vec![1])));
                    self.record_binary(TapeOp::Div, &s_tensor, &t, &result);
                }
                Value::Tensor(result)
            }
            // f32 标量 × Tensor
            (Value::Tensor(t), Value::Float32(s)) => {
                let s_f64 = *s as f64;
                let result = Rc::new(RefCell::new(t.borrow().div_scalar(s_f64)));
                if self.recording {
                    let s_tensor = Rc::new(RefCell::new(Tensor::from_vec(vec![s_f64], vec![1])));
                    self.record_binary(TapeOp::Div, &t, &s_tensor, &result);
                }
                Value::Tensor(result)
            }
            (Value::Float32(s), Value::Tensor(t)) => {
                // s / t: scalar divided by tensor element-wise
                let s_f64 = *s as f64;
                let result = Rc::new(RefCell::new(t.borrow().div_scalar_inv(s_f64)));
                if self.recording {
                    let s_tensor = Rc::new(RefCell::new(Tensor::from_vec(vec![s_f64], vec![1])));
                    self.record_binary(TapeOp::Div, &s_tensor, &t, &result);
                }
                Value::Tensor(result)
            }
            (Value::Tensor(t1), Value::Tensor(t2)) => {
                let result_tensor = t1.borrow().div_tensor(&t2.borrow())
                    .map_err(|msg| TenthError::RuntimeError { line: None, col: None, message: msg })?;
                let result = Rc::new(RefCell::new(result_tensor));
                if self.recording { self.record_binary(TapeOp::Div, &t1, &t2, &result); }
                Value::Tensor(result)
            }
            _ => return err("/ 类型不匹配"),
        })
    }

    pub(super) fn compare(&self, a: &Value, b: &Value, nf: fn(f64, f64) -> bool, sf: fn(&str, &str) -> bool) -> TenthResult<bool> {
        Ok(match (a, b) {
            (Value::Int(x, _), Value::Int(y, _)) => nf(*x as f64, *y as f64),
            (Value::Float(x), Value::Float(y)) => nf(*x, *y),
            (Value::Int(x, _), Value::Float(y)) => nf(*x as f64, *y),
            (Value::Float(x), Value::Int(y, _)) => nf(*x, *y as f64),
            // f32 路径：提升为 f64 比较
            (Value::Float32(x), Value::Float32(y)) => nf(*x as f64, *y as f64),
            (Value::Int(x, _), Value::Float32(y)) => nf(*x as f64, *y as f64),
            (Value::Float32(x), Value::Int(y, _)) => nf(*x as f64, *y as f64),
            (Value::Float32(x), Value::Float(y)) => nf(*x as f64, *y),
            (Value::Float(x), Value::Float32(y)) => nf(*x, *y as f64),
            (Value::String(x), Value::String(y)) => sf(x, y),
            _ => return err("无法比较"),
        })
    }

    fn set_field(&self, val: &Value, field: &str, new_val: Value) -> TenthResult<()> {
        match val {
            Value::Struct { fields, .. } => {
                for (n, v) in fields.borrow_mut().iter_mut() {
                    if n == field { *v = new_val; return Ok(()); }
                }
                err(&format!("没有字段 '{}'", field))
            }
            Value::Shared(rc) => self.set_field(&rc.borrow(), field, new_val),
            Value::Ref(rc) => self.set_field(&rc.borrow(), field, new_val),
            _ => err("无法设置字段"),
        }
    }

    fn get_field(&self, val: &Value, field: &str) -> TenthResult<Value> {
        let v = match val {
            Value::Ref(rc) => return self.get_field(&rc.borrow(), field),
            Value::MutRef(w) => {
                if let Some(rc) = w.upgrade() { return self.get_field(&rc.borrow(), field); }
                return err("悬垂的 &mut 引用");
            }
            Value::Shared(rc) => return self.get_field(&rc.borrow(), field),
            v => v,
        };
        match v {
            Value::Struct { fields, .. } => {
                for (n, v) in fields.borrow().iter() {
                    if n == field { return Ok(v.clone()); }
                }
                err(&format!("没有字段 '{}'", field))
            }
            Value::Enum { fields, .. } => {
                for (n, v) in fields.borrow().iter() {
                    if n == field { return Ok(v.clone()); }
                }
                err(&format!("没有字段 '{}'", field))
            }
            // M1.2：Union 字段访问（tagged union）——只允许读取当前 active 字段
            Value::Union { name, active_field, value } => {
                if active_field == field {
                    Ok((**value).clone())
                } else {
                    err(&format!(
                        "union '{}' 当前活跃字段是 '{}'，不能访问非活跃字段 '{}'",
                        name, active_field, field
                    ))
                }
            }
            _ => err(&format!("没有字段 '{}'", field)),
        }
    }

    pub(super) fn vm_eq(&self, a: &Value, b: &Value) -> bool {
        match (a, b) {
            (Value::Int(x, _), Value::Int(y, _)) => x == y,
            (Value::Float(x), Value::Float(y)) => (x - y).abs() < 1e-10,
            (Value::Float32(x), Value::Float32(y)) => (x - y).abs() < 1e-6,
            // f32 与 f64 比较：按 f64 精度判等（f32 提升为 f64 无损）
            (Value::Float32(x), Value::Float(y)) => ((*x as f64) - y).abs() < 1e-10,
            (Value::Float(x), Value::Float32(y)) => (x - (*y as f64)).abs() < 1e-10,
            (Value::Int(x, _), Value::Float32(y)) => (*x as f32 - y).abs() < 1e-6,
            (Value::Float32(x), Value::Int(y, _)) => (x - *y as f32).abs() < 1e-6,
            (Value::Int(x, _), Value::Float(y)) => ((*x as f64) - y).abs() < 1e-10,
            (Value::Float(x), Value::Int(y, _)) => (x - (*y as f64)).abs() < 1e-10,
            (Value::Bool(x), Value::Bool(y)) => x == y,
            (Value::Char(x), Value::Char(y)) => x == y,
            (Value::String(x), Value::String(y)) => x == y,
            (Value::Unit, Value::Unit) => true,
            _ => false,
        }
    }

    // ── Autodiff recording helpers ─────────────────────────────────────

    pub(super) fn record_unary(&mut self, op: TapeOp, input: &Rc<RefCell<Tensor>>, result: &Rc<RefCell<Tensor>>) {
        if let Some(ref mut tape) = self.tape {
            let node_id = match input.borrow().tape_id {
                Some(input_id) => tape.unary(op, input_id, input.clone(), result.clone()),
                None => {
                    let dummy = tape.input(input.clone());
                    tape.unary(op, dummy, input.clone(), result.clone())
                }
            };
            result.borrow_mut().tape_id = Some(node_id);
        }
    }

    pub(super) fn record_binary(&mut self, op: TapeOp, t1: &Rc<RefCell<Tensor>>, t2: &Rc<RefCell<Tensor>>, result: &Rc<RefCell<Tensor>>) {
        if let Some(ref mut tape) = self.tape {
            let id1 = t1.borrow().tape_id;
            let id2 = t2.borrow().tape_id;
            let node_id = match (id1, id2) {
                (Some(a), Some(b)) => tape.binary(op, a, b, t1.clone(), t2.clone(), result.clone()),
                (Some(a), None) => {
                    let dummy = tape.input(t2.clone());
                    tape.binary(op, a, dummy, t1.clone(), t2.clone(), result.clone())
                }
                (None, Some(b)) => {
                    let dummy = tape.input(t1.clone());
                    tape.binary(op, dummy, b, t1.clone(), t2.clone(), result.clone())
                }
                (None, None) => tape.binary_direct(op, t1.clone(), t2.clone(), result.clone()),
            };
            result.borrow_mut().tape_id = Some(node_id);
        }
    }
}
