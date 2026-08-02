//! VM 调度器、执行循环、私有算术、字段访问、autodiff 记录。
//!
//! 从 runtime/vm.rs 拆分而来（T3b 架构重构）。

use std::collections::HashMap;
use crate::hir::types::BaseType;
use std::rc::Rc;
use std::cell::RefCell;
use crate::error::{TenthError, TenthResult};
use crate::runtime::value::{Value, FutureState, check_int_overflow, int_overflow_err};
use crate::runtime::autodiff::TapeOp;
use crate::runtime::tensor::Tensor;
use crate::runtime::async_io::ASYNC_IO;

use super::Vm;
use super::err;
use super::op::{Frame, YieldReason};

/// R4 操作数栈复用池上限：防止极端深递归/深调用链时池内空闲栈无限累积。
/// 池内每个栈初始容量 64（Value 约 32-40 字节，单栈约 2.5KB），
/// 1024 个上限 ≈ 2.5MB 峰值，足够覆盖常规递归深度。
const STACK_POOL_MAX: usize = 1024;
/// R5 locals 复用池上限：防止极端深递归/深调用链时池内空闲 locals 无限累积。
/// 每个 locals 向量容量 = 局部变量数 × Value 大小（数十字节），
/// 256 个上限足够覆盖常规调用模式，峰值内存可控。
const LOCALS_POOL_MAX: usize = 256;

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
        // R4 操作数栈预分配：初始帧栈 reserve 后 take，主任务从预分配容量开始
        self.stack.reserve(64);
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

        // R1 性能优化：Rc::clone 只做引用计数 +1，消除调用路径对整段
        // 字节码与字符串表的深拷贝（fib(28) 约 83 万次调用的主要瓶颈）。
        let mut code = Rc::clone(&self.chunks[chunk_idx].code);
        let mut strings = Rc::clone(&self.chunks[chunk_idx].strings);

        // H-4: 独立的循环计数器，用于触发周期性 deadline 检查。
        // 不依赖 step_budget（用户可能只设 --timeout 而不设步数预算）。
        let mut loop_counter: u64 = 0;

        // R1 每指令预算检查轻量化：循环外判定一次，循环内用寄存器内计数器，
        // 避免每条指令的 Option match。语义与旧实现完全一致：
        // - 步数预算耗尽仍报 Timeout("VM 步数预算耗尽")
        // - 只设 --timeout 不设步数预算时 deadline 仍触发（H-4，逻辑不动）
        // - 退出 run_until_yield 时把剩余预算写回 self.step_budget，
        //   保证 with_step_limit/with_timeout_ms 的 save/restore 与
        //   跨任务恢复（Suspended/Yield 后重新进入）语义不变。
        let has_budget = self.step_budget.is_some();
        let mut budget_left = self.step_budget.unwrap_or(0);

        loop {
            if has_budget {
                if budget_left == 0 {
                    return Err(TenthError::Timeout {
                        message: "VM 步数预算耗尽".into(),
                    });
                }
                budget_left -= 1;
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
            // R3 性能优化：合并 decode（byte→Op）与 dispatch（Op→动作）双重分派为
            // 单一 match（byte→动作），消除 Op 枚举构造与第二次分派跳转的开销。
            // 语义零改动：操作数读取顺序、边界检查（ip 越界或操作数越界时
            // push Unit + 返回 Completed）、每指令动作均与原实现逐字一致。
            if ip >= code.len() {
                // 代码执行完毕（隐式 Ret）：任务完成，结果为 Unit
                if has_budget { self.step_budget = Some(budget_left); } // R1：预算回写
                self.stack.push(Value::Unit);
                return Ok(YieldReason::Completed);
            }
            let b = code[ip]; ip += 1;
            // 内联操作数读取宏（原 decode 的 r! 原样保留）：边界检查行为一致，
            // ip + n > code.len() 时 push Unit + 返回 Completed。
            macro_rules! r { ($t:ty) => {{ let n = std::mem::size_of::<$t>(); if ip + n > code.len() { if has_budget { self.step_budget = Some(budget_left); } self.stack.push(Value::Unit); return Ok(YieldReason::Completed); } let mut buf = [0u8; std::mem::size_of::<$t>()]; buf.copy_from_slice(&code[ip..ip+n]); ip += n; <$t>::from_le_bytes(buf) }}; }
            match b {
                // 0 PushInt
                0 => {
                    let n = r!(i64);
                    self.stack.push(Value::Int(n, BaseType::I32));
                }
                // 1 PushFloat
                1 => {
                    let f = r!(f64);
                    self.stack.push(Value::Float(f));
                }
                // 2 PushBool
                2 => {
                    let v = code[ip] != 0; ip += 1;
                    self.stack.push(Value::Bool(v));
                }
                // 3 PushStr
                3 => {
                    let i = r!(u64) as usize;
                    let s = strings.get(i).cloned().unwrap_or_default();
                    self.stack.push(Value::String(s));
                }
                // 4 PushUnit / 5 Pop / 6 Dup
                4 => self.stack.push(Value::Unit),
                5 => { self.stack.pop(); }
                6 => {
                    let v = self.stack.last().cloned().unwrap_or(Value::Unit);
                    self.stack.push(v);
                }
                // 7 Load / 8 Store
                7 => {
                    let i = r!(u64) as usize;
                    let v = locals.get(i).cloned().unwrap_or(Value::Unit);
                    self.stack.push(v);
                }
                8 => {
                    let i = r!(u64) as usize;
                    let v = self.stack.pop().unwrap_or(Value::Unit);
                    if i >= locals.len() { locals.resize(i+1, Value::Unit); }
                    locals[i] = v;
                }
                // 9 LoadGlobal / 10 StoreGlobal
                9 => {
                    let i = r!(u64) as usize;
                    let name = strings.get(i).cloned().unwrap_or_default();
                    let v = self.globals.get(&name).cloned().unwrap_or(Value::Unit);
                    self.stack.push(v);
                }
                10 => {
                    let i = r!(u64) as usize;
                    let name = strings.get(i).cloned().unwrap_or_default();
                    let v = self.stack.pop().unwrap_or(Value::Unit);
                    self.globals.insert(name, v);
                }
                // 11 Add / 12 Sub / 13 Mul / 14 Div（R2 标量快路径，逐字保留）
                11 => {
                    // R2 快路径：栈顶两元素同为同型标量时 peek 计算（不弹栈），命中才弹栈；
                    // 混合类型 / String / Tensor 一律走 add_priv 慢路径（语义零改动）。
                    let n = self.stack.len();
                    if n >= 2 {
                        let fast = match (&self.stack[n - 2], &self.stack[n - 1]) {
                            (Value::Int(x, dt), Value::Int(y, _)) => {
                                // AUDIT-11.4.17：与 add_priv 完全一致（checked_add + 窄 dtype 检查，dtype 取左操作数）
                                let r = x.checked_add(*y).ok_or_else(|| int_overflow_err(*dt))?;
                                check_int_overflow(r, *dt)?;
                                Some(Value::Int(r, *dt))
                            }
                            (Value::Float(x), Value::Float(y)) => Some(Value::Float(x + y)),
                            (Value::Float32(x), Value::Float32(y)) => Some(Value::Float32(x + y)),
                            _ => None,
                        };
                        if let Some(r) = fast {
                            self.stack.pop();
                            self.stack.pop();
                            self.stack.push(r);
                        } else {
                            let (a, b) = self.pop2();
                            let r = self.add_priv(&a, &b).map_err(|e| self.with_line(chunk_idx, ip, e))?;
                            self.stack.push(r);
                        }
                    } else {
                        let (a, b) = self.pop2();
                        let r = self.add_priv(&a, &b).map_err(|e| self.with_line(chunk_idx, ip, e))?;
                        self.stack.push(r);
                    }
                }
                12 => {
                    // R2 快路径：同 add_priv 模式，慢路径 sub_priv 兜底。
                    let n = self.stack.len();
                    if n >= 2 {
                        let fast = match (&self.stack[n - 2], &self.stack[n - 1]) {
                            (Value::Int(x, dt), Value::Int(y, _)) => {
                                let r = x.checked_sub(*y).ok_or_else(|| int_overflow_err(*dt))?;
                                check_int_overflow(r, *dt)?;
                                Some(Value::Int(r, *dt))
                            }
                            (Value::Float(x), Value::Float(y)) => Some(Value::Float(x - y)),
                            (Value::Float32(x), Value::Float32(y)) => Some(Value::Float32(x - y)),
                            _ => None,
                        };
                        if let Some(r) = fast {
                            self.stack.pop();
                            self.stack.pop();
                            self.stack.push(r);
                        } else {
                            let (a, b) = self.pop2();
                            let r = self.sub_priv(&a, &b).map_err(|e| self.with_line(chunk_idx, ip, e))?;
                            self.stack.push(r);
                        }
                    } else {
                        let (a, b) = self.pop2();
                        let r = self.sub_priv(&a, &b).map_err(|e| self.with_line(chunk_idx, ip, e))?;
                        self.stack.push(r);
                    }
                }
                13 => {
                    // R2 快路径：同 add_priv 模式，慢路径 mul_priv 兜底。
                    let n = self.stack.len();
                    if n >= 2 {
                        let fast = match (&self.stack[n - 2], &self.stack[n - 1]) {
                            (Value::Int(x, dt), Value::Int(y, _)) => {
                                let r = x.checked_mul(*y).ok_or_else(|| int_overflow_err(*dt))?;
                                check_int_overflow(r, *dt)?;
                                Some(Value::Int(r, *dt))
                            }
                            (Value::Float(x), Value::Float(y)) => Some(Value::Float(x * y)),
                            (Value::Float32(x), Value::Float32(y)) => Some(Value::Float32(x * y)),
                            _ => None,
                        };
                        if let Some(r) = fast {
                            self.stack.pop();
                            self.stack.pop();
                            self.stack.push(r);
                        } else {
                            let (a, b) = self.pop2();
                            let r = self.mul_priv(&a, &b).map_err(|e| self.with_line(chunk_idx, ip, e))?;
                            self.stack.push(r);
                        }
                    } else {
                        let (a, b) = self.pop2();
                        let r = self.mul_priv(&a, &b).map_err(|e| self.with_line(chunk_idx, ip, e))?;
                        self.stack.push(r);
                    }
                }
                14 => {
                    // R2 快路径：Int-Int 分支与 div_priv 完全一致（除零检查 + checked_div 拦 i64::MIN/-1 + 窄 dtype 检查）。
                    let n = self.stack.len();
                    if n >= 2 {
                        let fast = match (&self.stack[n - 2], &self.stack[n - 1]) {
                            (Value::Int(x, dt), Value::Int(y, _)) => {
                                if *y == 0 {
                                    return Err(self.err_here(chunk_idx, ip, "整数除零".into()));
                                }
                                let r = x.checked_div(*y).ok_or_else(|| int_overflow_err(*dt))?;
                                check_int_overflow(r, *dt)?;
                                Some(Value::Int(r, *dt))
                            }
                            (Value::Float(x), Value::Float(y)) => Some(Value::Float(x / y)),
                            (Value::Float32(x), Value::Float32(y)) => Some(Value::Float32(x / y)),
                            _ => None,
                        };
                        if let Some(r) = fast {
                            self.stack.pop();
                            self.stack.pop();
                            self.stack.push(r);
                        } else {
                            let (a, b) = self.pop2();
                            let r = self.div_priv(&a, &b).map_err(|e| self.with_line(chunk_idx, ip, e))?;
                            self.stack.push(r);
                        }
                    } else {
                        let (a, b) = self.pop2();
                        let r = self.div_priv(&a, &b).map_err(|e| self.with_line(chunk_idx, ip, e))?;
                        self.stack.push(r);
                    }
                }
                // 15 Mod / 16 Neg / 17 Not
                15 => {
                    let b = self.pop_int()?; let a = self.pop_int()?;
                    if b == 0 {
                        return Err(self.err_here(chunk_idx, ip, "整数取模除零".into()));
                    }
                    // AUDIT-11.4.17：checked_rem 拦截 i64::MIN % -1 等溢出（overflow-checks=true 下直接 % 会 panic）
                    let r = a.checked_rem(b).ok_or_else(|| int_overflow_err(BaseType::I32))?;
                    self.stack.push(Value::Int(r, BaseType::I32));
                    // 注：Mod 指令通过 pop_int 获取值，丢失 dtype，默认 I32
                }
                16 => {
                    let v = self.stack.pop().unwrap_or(Value::Unit);
                    match v {
                        Value::Int(n, dt) => {
                            // AUDIT-11.4.17：checked_neg 拦截 i64::MIN 取负溢出
                            let r = n.checked_neg().ok_or_else(|| int_overflow_err(dt))?;
                            check_int_overflow(r, dt)?;
                            self.stack.push(Value::Int(r, dt));
                        }
                        Value::Float(n) => self.stack.push(Value::Float(-n)),
                        Value::Float32(n) => self.stack.push(Value::Float32(-n)),
                        Value::Tensor(t) => self.stack.push(Value::Tensor(Rc::new(RefCell::new(t.borrow().neg())))),
                        _ => return Err(self.err_here(chunk_idx, ip, "无法取负".into())),
                    }
                }
                17 => {
                    let v = self.stack.pop().unwrap_or(Value::Unit);
                    self.stack.push(Value::Bool(!v.is_truthy()));
                }

                // 18 Eq / 19 Neq / 20 Lt / 21 Gt / 22 Lte / 23 Gte
                18 => {
                    // R2 快路径：同型标量比较（语义与 vm_eq 完全一致：Int 精确相等，Float 1e-10，Float32 1e-6）
                    let n = self.stack.len();
                    if n >= 2 {
                        let fast = match (&self.stack[n - 2], &self.stack[n - 1]) {
                            (Value::Int(x, _), Value::Int(y, _)) => Some(x == y),
                            (Value::Float(x), Value::Float(y)) => Some((x - y).abs() < 1e-10),
                            (Value::Float32(x), Value::Float32(y)) => Some((x - y).abs() < 1e-6),
                            _ => None,
                        };
                        if let Some(b) = fast {
                            self.stack.pop();
                            self.stack.pop();
                            self.stack.push(Value::Bool(b));
                        } else {
                            let (a, b) = self.pop2();
                            self.stack.push(Value::Bool(self.vm_eq(&a, &b)));
                        }
                    } else {
                        let (a, b) = self.pop2();
                        self.stack.push(Value::Bool(self.vm_eq(&a, &b)));
                    }
                }
                19 => {
                    // R2 快路径：vm_eq 取反（保持 NaN 语义：用 !(...) 而非 >= 反向，与 vm_eq 一致）
                    let n = self.stack.len();
                    if n >= 2 {
                        let fast = match (&self.stack[n - 2], &self.stack[n - 1]) {
                            (Value::Int(x, _), Value::Int(y, _)) => Some(x != y),
                            (Value::Float(x), Value::Float(y)) => Some(!((x - y).abs() < 1e-10)),
                            (Value::Float32(x), Value::Float32(y)) => Some(!((x - y).abs() < 1e-6)),
                            _ => None,
                        };
                        if let Some(b) = fast {
                            self.stack.pop();
                            self.stack.pop();
                            self.stack.push(Value::Bool(b));
                        } else {
                            let (a, b) = self.pop2();
                            self.stack.push(Value::Bool(!self.vm_eq(&a, &b)));
                        }
                    } else {
                        let (a, b) = self.pop2();
                        self.stack.push(Value::Bool(!self.vm_eq(&a, &b)));
                    }
                }
                20 => {
                    // R2 快路径：同型标量比较（语义与 compare 完全一致：Int-Int / Float32-Float32 提升为 f64 比较）
                    let n = self.stack.len();
                    if n >= 2 {
                        let fast = match (&self.stack[n - 2], &self.stack[n - 1]) {
                            (Value::Int(x, _), Value::Int(y, _)) => Some((*x as f64) < (*y as f64)),
                            (Value::Float(x), Value::Float(y)) => Some(x < y),
                            (Value::Float32(x), Value::Float32(y)) => Some((*x as f64) < (*y as f64)),
                            _ => None,
                        };
                        if let Some(b) = fast {
                            self.stack.pop();
                            self.stack.pop();
                            self.stack.push(Value::Bool(b));
                        } else {
                            let (a, b) = self.pop2();
                            self.stack.push(Value::Bool(self.compare(&a,&b,|x,y|x<y,|x,y|x<y)?));
                        }
                    } else {
                        let (a, b) = self.pop2();
                        self.stack.push(Value::Bool(self.compare(&a,&b,|x,y|x<y,|x,y|x<y)?));
                    }
                }
                21 => {
                    let n = self.stack.len();
                    if n >= 2 {
                        let fast = match (&self.stack[n - 2], &self.stack[n - 1]) {
                            (Value::Int(x, _), Value::Int(y, _)) => Some((*x as f64) > (*y as f64)),
                            (Value::Float(x), Value::Float(y)) => Some(x > y),
                            (Value::Float32(x), Value::Float32(y)) => Some((*x as f64) > (*y as f64)),
                            _ => None,
                        };
                        if let Some(b) = fast {
                            self.stack.pop();
                            self.stack.pop();
                            self.stack.push(Value::Bool(b));
                        } else {
                            let (a, b) = self.pop2();
                            self.stack.push(Value::Bool(self.compare(&a,&b,|x,y|x>y,|x,y|x>y)?));
                        }
                    } else {
                        let (a, b) = self.pop2();
                        self.stack.push(Value::Bool(self.compare(&a,&b,|x,y|x>y,|x,y|x>y)?));
                    }
                }
                22 => {
                    let n = self.stack.len();
                    if n >= 2 {
                        let fast = match (&self.stack[n - 2], &self.stack[n - 1]) {
                            (Value::Int(x, _), Value::Int(y, _)) => Some((*x as f64) <= (*y as f64)),
                            (Value::Float(x), Value::Float(y)) => Some(x <= y),
                            (Value::Float32(x), Value::Float32(y)) => Some((*x as f64) <= (*y as f64)),
                            _ => None,
                        };
                        if let Some(b) = fast {
                            self.stack.pop();
                            self.stack.pop();
                            self.stack.push(Value::Bool(b));
                        } else {
                            let (a, b) = self.pop2();
                            self.stack.push(Value::Bool(self.compare(&a,&b,|x,y|x<=y,|x,y|x<=y)?));
                        }
                    } else {
                        let (a, b) = self.pop2();
                        self.stack.push(Value::Bool(self.compare(&a,&b,|x,y|x<=y,|x,y|x<=y)?));
                    }
                }
                23 => {
                    let n = self.stack.len();
                    if n >= 2 {
                        let fast = match (&self.stack[n - 2], &self.stack[n - 1]) {
                            (Value::Int(x, _), Value::Int(y, _)) => Some((*x as f64) >= (*y as f64)),
                            (Value::Float(x), Value::Float(y)) => Some(x >= y),
                            (Value::Float32(x), Value::Float32(y)) => Some((*x as f64) >= (*y as f64)),
                            _ => None,
                        };
                        if let Some(b) = fast {
                            self.stack.pop();
                            self.stack.pop();
                            self.stack.push(Value::Bool(b));
                        } else {
                            let (a, b) = self.pop2();
                            self.stack.push(Value::Bool(self.compare(&a,&b,|x,y|x>=y,|x,y|x>=y)?));
                        }
                    } else {
                        let (a, b) = self.pop2();
                        self.stack.push(Value::Bool(self.compare(&a,&b,|x,y|x>=y,|x,y|x>=y)?));
                    }
                }
                // 24 Jump / 25 JmpFalse / 26 JmpTrue
                24 => {
                    let o = r!(i32);
                    ip = (ip as i32 + o) as usize;
                }
                25 => {
                    let o = r!(i32);
                    if !self.stack.pop().unwrap_or(Value::Unit).is_truthy() {
                        ip = (ip as i32 + o) as usize;
                    }
                }
                26 => {
                    let o = r!(i32);
                    if self.stack.pop().unwrap_or(Value::Unit).is_truthy() {
                        ip = (ip as i32 + o) as usize;
                    }
                }

                // 27 Call / 28 CallN / 29 MethodCall（TailCall=55 在末尾）
                27 => {
                    let i = r!(u64) as usize;
                    // R5：借用 strings 表，消灭每次调用的无条件 String clone
                    let name = strings.get(i).map(|s| s.as_str()).unwrap_or("");
                    // Try native first (legacy: uses stack depth as arg count)
                    if let Some(native_fn) = self.natives.get(name).copied() {
                        let n = self.stack.len() - base;
                        let mut args = vec![Value::Unit; n];
                        for i in (0..n).rev() { args[i] = self.stack.pop().unwrap_or(Value::Unit); }
                        let result = native_fn(self, &args).map_err(|e| self.with_line(chunk_idx, ip, e))?;
                        self.stack.push(result);
                    } else if let Some(&callee_idx) = self.functions.get(name) {
                        let callee_args = self.chunks[callee_idx].num_args;
                        let callee_locals = self.chunks[callee_idx].num_locals;
                        // Phase 2 栈迁移：先从 caller 栈 pop args 到 callee locals
                        // R5 locals 复用池：取空闲 locals（保留容量），池空则新建，resize 填 Unit
                        let mut new_locals = self.locals_pool.pop().unwrap_or_default();
                        new_locals.resize(callee_locals.max(callee_args), Value::Unit);
                        for i in (0..callee_args).rev() {
                            if self.stack.len() > base { new_locals[i] = self.stack.pop().unwrap(); }
                        }
                        // 保存 caller 栈到 Frame，给 callee 一个空栈
                        let caller_stack = std::mem::take(&mut self.stack);
                        // R4 操作数栈复用：从池取空闲栈（保留容量），池空则新建预分配栈，
                        // 消除每次调用 callee 栈的 buffer alloc/free
                        self.stack = self.stack_pool.pop().unwrap_or_else(|| Vec::with_capacity(64));
                        self.frames.push(Frame {
                            ip,
                            chunk_idx,
                            // R4：caller locals 零拷贝移入 frame（Ret 时恢复），消除每次调用的深拷贝
                            locals: std::mem::take(&mut locals),
                            stack_base: base,
                            operand_stack: caller_stack,
                            task_id: current_task_id,
                        });
                        chunk_idx = callee_idx;
                        code = Rc::clone(&self.chunks[chunk_idx].code);
                        strings = Rc::clone(&self.chunks[chunk_idx].strings);
                        ip = 0;
                        locals = new_locals;
                        base = 0;  // callee 新栈从 0 开始
                    } else {
                        return Err(self.err_here(chunk_idx, ip, format!("未定义的函数 '{}'", name)));
                    }
                }
                28 => {
                    let i = r!(u64) as usize;
                    let num_args = r!(u64) as usize;
                    // R5：借用 strings 表，消灭每次调用的无条件 String clone；
                    // args 从 locals 池取（CallN 的 args 即 callee locals）
                    let name = strings.get(i).map(|s| s.as_str()).unwrap_or("");
                    let n = num_args;
                    let mut args = self.locals_pool.pop().unwrap_or_default();
                    args.resize(n, Value::Unit);
                    for i in (0..n).rev() { args[i] = self.stack.pop().unwrap_or(Value::Unit); }

                    // Try to find the function by name, checking globals for FnRef closures
                    // a1 P3：全局闭包 FnRef 携带捕获值 → 按名调用须追加捕获实参
                    // （槽位 params..params+captures），否则捕获缺失 → 静默错值。
                    let (callee_name, extra_captures): (String, Vec<Value>) =
                        if let Some(Value::FnRef { name: fname, captures, .. }) = self.globals.get(name) {
                            (fname.clone(), captures.clone())
                        } else {
                            (name.to_string(), Vec::new())
                        };
                    for cap in extra_captures {
                        args.push(cap);
                    }

                    if let Some(native_fn) = self.natives.get(&callee_name).copied() {
                        let result = native_fn(self, &args).map_err(|e| self.with_line(chunk_idx, ip, e))?;
                        self.stack.push(result);
                    } else if let Some(&callee_idx) = self.functions.get(&callee_name) {
                        // Phase 2 栈迁移：保存 caller 栈到 Frame，给 callee 一个空栈
                        let caller_stack = std::mem::take(&mut self.stack);
                        // R4 操作数栈复用：从池取空闲栈（保留容量），池空则新建预分配栈，
                        // 消除每次调用 callee 栈的 buffer alloc/free
                        self.stack = self.stack_pool.pop().unwrap_or_else(|| Vec::with_capacity(64));
                        self.frames.push(Frame {
                            ip,
                            chunk_idx,
                            // R4：caller locals 零拷贝移入 frame（Ret 时恢复），消除每次调用的深拷贝
                            locals: std::mem::take(&mut locals),
                            stack_base: base,
                            operand_stack: caller_stack,
                            task_id: current_task_id,
                        });
                        chunk_idx = callee_idx;
                        code = Rc::clone(&self.chunks[chunk_idx].code);
                        strings = Rc::clone(&self.chunks[chunk_idx].strings);
                        ip = 0;
                        // R5：args（来自池）成为 callee locals
                        locals = args;
                        locals.resize(self.chunks[chunk_idx].num_locals.max(locals.len()), Value::Unit);
                        base = 0;  // callee 新栈从 0 开始
                    } else if let Some(native_fn) = self.natives.get(&callee_name).copied() {
                        let result = native_fn(self, &args).map_err(|e| self.with_line(chunk_idx, ip, e))?;
                        self.stack.push(result);
                    } else if let Some(&callee_idx) = self.functions.get(&callee_name) {
                        // Phase 2 栈迁移：保存 caller 栈到 Frame，给 callee 一个空栈
                        let caller_stack = std::mem::take(&mut self.stack);
                        // R4 操作数栈复用：从池取空闲栈（保留容量），池空则新建预分配栈，
                        // 消除每次调用 callee 栈的 buffer alloc/free
                        self.stack = self.stack_pool.pop().unwrap_or_else(|| Vec::with_capacity(64));
                        self.frames.push(Frame {
                            ip,
                            chunk_idx,
                            // R4：caller locals 零拷贝移入 frame（Ret 时恢复），消除每次调用的深拷贝
                            locals: std::mem::take(&mut locals),
                            stack_base: base,
                            operand_stack: caller_stack,
                            task_id: current_task_id,
                        });
                        chunk_idx = callee_idx;
                        code = Rc::clone(&self.chunks[chunk_idx].code);
                        strings = Rc::clone(&self.chunks[chunk_idx].strings);
                        ip = 0;
                        // R5：args（来自池）成为 callee locals
                        locals = args;
                        locals.resize(self.chunks[chunk_idx].num_locals.max(locals.len()), Value::Unit);
                        base = 0;  // callee 新栈从 0 开始
                    } else {
                        return Err(self.err_here(chunk_idx, ip, format!("未定义的函数 '{}'", name)));
                    }
                }
                29 => {
                    let i = r!(u64) as usize;
                    let num_args = r!(u64) as usize;
                    let name = strings.get(i).cloned().unwrap_or_default();
                    let n = num_args;
                    let mut args = vec![Value::Unit; n];
                    for i in (0..n).rev() { args[i] = self.stack.pop().unwrap_or(Value::Unit); }
                    let receiver = self.stack.pop().unwrap_or(Value::Unit);
                    let result = self.call_method_priv(&receiver, &name, &args).map_err(|e| self.with_line(chunk_idx, ip, e))?;
                    self.stack.push(result);
                }

                // 30 Ret
                30 => {
                    let result = self.stack.pop().unwrap_or(Value::Unit);
                    if let Some(f) = self.frames.pop() {
                        // R5 locals 复用池：callee locals 清空入池（函数返回后 locals 不再被任何 Frame/闭包引用）
                        let mut callee_locals = std::mem::take(&mut locals);
                        callee_locals.clear();
                        if self.locals_pool.len() < LOCALS_POOL_MAX {
                            self.locals_pool.push(callee_locals);
                        }
                        // Phase 2 栈迁移：恢复 caller 栈；callee 栈清空入池复用（R4：消除每调用一次 alloc/free）
                        let mut callee_stack = std::mem::replace(&mut self.stack, f.operand_stack);
                        callee_stack.clear();
                        if self.stack_pool.len() < STACK_POOL_MAX {
                            self.stack_pool.push(callee_stack);
                        }
                        self.stack.push(result);
                        ip = f.ip;
                        chunk_idx = f.chunk_idx;
                        code = Rc::clone(&self.chunks[chunk_idx].code);
                        strings = Rc::clone(&self.chunks[chunk_idx].strings);
                        locals = f.locals;
                        base = f.stack_base;
                        current_task_id = f.task_id;
                        self.current_task = current_task_id;
                    } else {
                        // 顶层 Ret：任务完成。把结果推回栈供调度器 pop。
                        if has_budget { self.step_budget = Some(budget_left); } // R1：预算回写
                        self.stack.push(result);
                        return Ok(YieldReason::Completed);
                    }
                }
                // 31 MakeVec / 32 MakeMap
                31 => {
                    let n = r!(u64) as usize;
                    let mut v = Vec::new();
                    for _ in 0..n { v.push(self.stack.pop().unwrap_or(Value::Unit)); }
                    v.reverse();
                    self.stack.push(Value::Vec(Rc::new(RefCell::new(v))));
                }
                32 => {
                    let n = r!(u64) as usize;
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

                // 33 NewStruct / 34 LoadField / 35 StoreField / 56 NewUnion
                33 => {
                    let name_i = r!(u64) as usize;
                    let n = r!(u64) as usize;
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
                34 => {
                    let i = r!(u64) as usize;
                    let fname = strings.get(i).cloned().unwrap_or_default();
                    let val = self.stack.pop().unwrap_or(Value::Unit);
                    let v = self.get_field(&val, &fname).map_err(|e| self.with_line(chunk_idx, ip, e))?;
                    self.stack.push(v);
                }
                35 => {
                    let i = r!(u64) as usize;
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
                                return Err(self.err_here(chunk_idx, ip, format!(
                                    "union '{}' 当前活跃字段是 '{}'，不能修改非活跃字段 '{}'",
                                    name, active_field, fname
                                )));
                            }
                        }
                        _ => {
                            self.set_field(&target, &fname, new_val).map_err(|e| self.with_line(chunk_idx, ip, e))?;
                        }
                    }
                }
                // M1.2：union 构造 — 弹出栈顶 value，构造带 active_field 的 tagged union
                56 => {
                    let name_i = r!(u64) as usize;
                    let field_i = r!(u64) as usize;
                    let name = strings.get(name_i).cloned().unwrap_or_default();
                    let active_field = strings.get(field_i).cloned().unwrap_or_default();
                    let value = self.stack.pop().unwrap_or(Value::Unit);
                    self.stack.push(Value::Union { name, active_field, value: Box::new(value) });
                }

                // 36 IndexGet / 37 SliceStr
                36 => {
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
                                        return Err(self.err_here(chunk_idx, ip, format!(
                                            "索引 {} 越界，形状为 {:?}",
                                            i, tensor.shape()
                                        )));
                                    }
                                }
                            } else {
                                match tensor.index_dim(i) {
                                    Ok(sub) => {
                                        drop(tensor);
                                        self.stack.push(Value::Tensor(Rc::new(RefCell::new(sub))));
                                    }
                                    Err(msg) => return Err(self.err_here(chunk_idx, ip, msg)),
                                }
                            }
                        }
                        _ => return Err(self.err_here(chunk_idx, ip, "无法索引".into())),
                    }
                }
                37 => {
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
                                return Err(self.err_here(chunk_idx, ip, "字符串切片起始位置大于结束位置".into()));
                            }
                            let slice: String = chars[si..ei].iter().collect();
                            self.stack.push(Value::String(slice));
                        }
                        _ => return Err(self.err_here(chunk_idx, ip, "SliceStr 需要字符串目标".into())),
                    }
                }

                // 38 MakeEnum / 39 IsEnumVariant / 40 EnumGetField（IsStruct=46 在下方）
                38 => {
                    let name_i = r!(u64) as usize;
                    let variant_i = r!(u64) as usize;
                    let n = r!(u64) as usize;
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
                39 => {
                    let variant_i = r!(u64) as usize;
                    let variant_name = strings.get(variant_i).cloned().unwrap_or_default();
                    let val = self.stack.pop().unwrap_or(Value::Unit);
                    let matches = match &val {
                        Value::Enum { variant, .. } => variant == &variant_name,
                        _ => false,
                    };
                    self.stack.push(Value::Bool(matches));
                }
                40 => {
                    let field_i = r!(u64) as usize;
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

                // 41 PushRange / 42 MoveOp / 43 MakeTensor / 44 MakeClosure
                41 => {
                    let start = r!(i64);
                    let end = r!(i64);
                    let inclusive = { let b = code[ip]; ip += 1; b != 0 };
                    self.stack.push(Value::Range { start, end, inclusive });
                }
                42 => {
                    // no-op: move semantics are checked at HIR level
                }
                43 => {
                    let rows = r!(u64) as usize;
                    let cols = r!(u64) as usize;
                    let dtype = { let d = code[ip]; ip += 1; d };
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
                44 => {
                    let params_count = r!(u64) as usize;
                    let captures_count = r!(u64) as usize;
                    let name_idx = r!(u64) as usize;
                    // Create a FnRef value pointing to the closure function
                    let name = strings.get(name_idx).cloned().unwrap_or_default();
                    // a1 P3：从父栈弹出 captures_count 个捕获值装入 FnRef.captures（值内联）。
                    // 弹出顺序与闭包 chunk 捕获槽（params..params+captures）一致。
                    let mut captures = Vec::with_capacity(captures_count);
                    for _ in 0..captures_count {
                        captures.push(self.stack.pop().unwrap_or(Value::Unit));
                    }
                    captures.reverse();
                    let param_names: Vec<(String, crate::hir::types::Type)> = (0..params_count)
                        .map(|i| (format!("__param_{i}"), crate::hir::types::Type::Unknown))
                        .collect();
                    self.stack.push(Value::FnRef {
                        name,
                        params: param_names,
                        return_type: crate::hir::types::Type::Unknown,
                        captures,
                    });
                }

                // 45 PushFloat32 / 46 IsStruct / 47 Await / 48 Spawn / 49 MakeTuple / 50 IsTuple / 51 TupleGet
                45 => {
                    let f = r!(f32);
                    self.stack.push(Value::Float32(f));
                }
                46 => {
                    let name_i = r!(u64) as usize;
                    let struct_name = strings.get(name_i).cloned().unwrap_or_default();
                    let val = self.stack.pop().unwrap_or(Value::Unit);
                    let matches = match &val {
                        Value::Struct { name, .. } => name == &struct_name,
                        _ => false,
                    };
                    self.stack.push(Value::Bool(matches));
                }
                47 => {
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
                                if has_budget { self.step_budget = Some(budget_left); } // R1：预算回写
                                return Ok(YieldReason::Suspended(current_task_id));
                            }
                        }
                        other => {
                            // 非 Future 值：直接传值（await 7 → 7）
                            self.stack.push(other.clone());
                        }
                    }
                }
                48 => {
                    let v = self.stack.pop().unwrap_or(Value::Unit);
                    // Phase 2 Step 3-4：spawn 保持 eager 语义（立即求值，包装为 Ready Future）。
                    // 设计决策：真正的并发不来自 spawn，而来自 Step 5 的 async I/O 创建 Pending Future。
                    // spawn 的 inner 在 bytecode 层面已被编译为"先求值 inner 再 Op::Spawn"，
                    // 改为延迟执行需要重构 bytecode 编译逻辑，超出 Step 3-4 范围。
                    // eager spawn 仍然有用：await 时如果遇到 Pending Future（由 async I/O 创建），
                    // 调度器会切换到其他就绪任务，包括其他 spawn 产生的 eager Future 的 await 者。
                    self.stack.push(Value::future_ready(v));
                }
                49 => {
                    let n = r!(u64) as usize;
                    // Pop n values (last pushed = highest index) and assemble Tuple
                    let mut items = Vec::with_capacity(n);
                    for _ in 0..n {
                        items.push(self.stack.pop().unwrap_or(Value::Unit));
                    }
                    items.reverse();
                    self.stack.push(Value::Tuple(items));
                }
                50 => {
                    let expected_len = r!(u64) as usize;
                    // Pop value, push Bool(val is Tuple with expected_len).
                    // Mirrors IsEnumVariant: pop + push bool.
                    let val = self.stack.pop().unwrap_or(Value::Unit);
                    let matches = match &val {
                        Value::Tuple(items) => items.len() == expected_len,
                        _ => false,
                    };
                    self.stack.push(Value::Bool(matches));
                }
                51 => {
                    let i = r!(u64) as usize;
                    let val = self.stack.pop().unwrap_or(Value::Unit);
                    let elem = match &val {
                        Value::Tuple(items) => items.get(i).cloned().unwrap_or(Value::Unit),
                        _ => Value::Unit,
                    };
                    self.stack.push(elem);
                }
                // 52 Try / 53 Yield / 54 PushChar / 55 TailCall / 未知 opcode → Ret
                52 => {
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
                            // R5 locals 复用池：callee locals 清空入池（early return 后不再被引用）
                            let mut callee_locals = std::mem::take(&mut locals);
                            callee_locals.clear();
                            if self.locals_pool.len() < LOCALS_POOL_MAX {
                                self.locals_pool.push(callee_locals);
                            }
                            // Phase 2 栈迁移：恢复 caller 栈（同 Op::Ret）；callee 栈清空入池复用（R4）
                            let mut callee_stack = std::mem::replace(&mut self.stack, f.operand_stack);
                            callee_stack.clear();
                            if self.stack_pool.len() < STACK_POOL_MAX {
                                self.stack_pool.push(callee_stack);
                            }
                            self.stack.push(result);
                            ip = f.ip;
                            chunk_idx = f.chunk_idx;
                            code = Rc::clone(&self.chunks[chunk_idx].code);
                            strings = Rc::clone(&self.chunks[chunk_idx].strings);
                            locals = f.locals;
                            base = f.stack_base;
                            current_task_id = f.task_id;
                            self.current_task = current_task_id;
                        } else {
                            // 最外层函数：任务完成（early return Err）。把结果推回栈供调度器 pop。
                            if has_budget { self.step_budget = Some(budget_left); } // R1：预算回写
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
                53 => {
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
                    if has_budget { self.step_budget = Some(budget_left); } // R1：预算回写
                    return Ok(YieldReason::Yield(current_task_id));
                }
                54 => {
                    let c = r!(u32);
                    let ch = char::from_u32(c).unwrap_or('\0');
                    self.stack.push(Value::Char(ch));
                }
                55 => {
                    let i = r!(u64) as usize;
                    let num_args = r!(u64) as usize;
                    // TCO：复用当前帧，不压新帧
                    // R5：借用 strings 表；args 从 locals 池取（TCO 的 args 即新 callee locals）
                    let name = strings.get(i).map(|s| s.as_str()).unwrap_or("");
                    let n = num_args;
                    let mut args = self.locals_pool.pop().unwrap_or_default();
                    args.resize(n, Value::Unit);
                    for i in (0..n).rev() { args[i] = self.stack.pop().unwrap_or(Value::Unit); }

                    // 查找函数（同 CallN）；a1 P3：全局闭包 FnRef 携带捕获值 → 追加捕获实参
                    // （槽位 params..params+captures），否则捕获缺失 → 静默错值。
                    let (callee_name, extra_captures): (String, Vec<Value>) =
                        if let Some(Value::FnRef { name: fname, captures, .. }) = self.globals.get(name) {
                            (fname.clone(), captures.clone())
                        } else {
                            (name.to_string(), Vec::new())
                        };
                    for cap in extra_captures {
                        args.push(cap);
                    }

                    if let Some(native_fn) = self.natives.get(&callee_name).copied() {
                        let result = native_fn(self, &args).map_err(|e| self.with_line(chunk_idx, ip, e))?;
                        self.stack.push(result);
                    } else if let Some(&callee_idx) = self.functions.get(&callee_name) {
                        // TCO：不压帧，直接替换当前帧状态
                        // R5：旧 locals 清空入池，args 成为新 locals（buffer 循环复用）
                        let mut old_locals = std::mem::replace(&mut locals, args);
                        old_locals.clear();
                        if self.locals_pool.len() < LOCALS_POOL_MAX {
                            self.locals_pool.push(old_locals);
                        }
                        chunk_idx = callee_idx;
                        code = Rc::clone(&self.chunks[chunk_idx].code);
                        strings = Rc::clone(&self.chunks[chunk_idx].strings);
                        ip = 0;
                        locals.resize(self.chunks[chunk_idx].num_locals.max(locals.len()), Value::Unit);
                        // base 不重置 — 当前栈帧继续使用
                    } else if let Some(native_fn) = self.natives.get(&callee_name).copied() {
                        let result = native_fn(self, &args).map_err(|e| self.with_line(chunk_idx, ip, e))?;
                        self.stack.push(result);
                    } else if let Some(&callee_idx) = self.functions.get(&callee_name) {
                        // TCO：不压帧，直接替换当前帧状态
                        // R5：旧 locals 清空入池，args 成为新 locals（buffer 循环复用）
                        let mut old_locals = std::mem::replace(&mut locals, args);
                        old_locals.clear();
                        if self.locals_pool.len() < LOCALS_POOL_MAX {
                            self.locals_pool.push(old_locals);
                        }
                        chunk_idx = callee_idx;
                        code = Rc::clone(&self.chunks[chunk_idx].code);
                        strings = Rc::clone(&self.chunks[chunk_idx].strings);
                        ip = 0;
                        locals.resize(self.chunks[chunk_idx].num_locals.max(locals.len()), Value::Unit);
                    } else {
                        return Err(self.err_here(chunk_idx, ip, format!("未定义的函数 '{}'", name)));
                    }
                }
                // a1 P1：57 CallClosure / 58 TailCallClosure（VM 闭包值间接调用）
                // 栈布局 [arg1..argN, callee]：先弹 callee 值，再弹 N 个参数（locals 池复用，同 CallN）。
                57 => {
                    let n = r!(u64) as usize;
                    let callee = self.stack.pop().unwrap_or(Value::Unit);
                    let mut args = self.locals_pool.pop().unwrap_or_default();
                    args.resize(n, Value::Unit);
                    for i in (0..n).rev() { args[i] = self.stack.pop().unwrap_or(Value::Unit); }

                    match &callee {
                        Value::FnRef { name, captures, .. } => {
                            // a1 P3：追加捕获值作为额外实参（槽位 params..params+captures），
                            // 闭包 chunk 以 params+captures 个槽位接收。
                            for cap in captures {
                                args.push(cap.clone());
                            }
                            // R5：callee_name 借用 globals/strings，仅 FnRef 分支条件性 clone（同 CallN）
                            let callee_name: &str = if let Some(Value::FnRef { name: fname, .. }) = self.globals.get(name) {
                                fname.as_str()
                            } else {
                                name
                            };
                            // FnRef 指向 native 函数名也支持（如 `let p = println; p("x")`）
                            if let Some(native_fn) = self.natives.get(callee_name).copied() {
                                let result = native_fn(self, &args).map_err(|e| self.with_line(chunk_idx, ip, e))?;
                                self.stack.push(result);
                            } else if let Some(&callee_idx) = self.functions.get(callee_name) {
                                // 压新帧调用（同 CallN）
                                let caller_stack = std::mem::take(&mut self.stack);
                                self.stack = self.stack_pool.pop().unwrap_or_else(|| Vec::with_capacity(64));
                                self.frames.push(Frame {
                                    ip,
                                    chunk_idx,
                                    locals: std::mem::take(&mut locals),
                                    stack_base: base,
                                    operand_stack: caller_stack,
                                    task_id: current_task_id,
                                });
                                chunk_idx = callee_idx;
                                code = Rc::clone(&self.chunks[chunk_idx].code);
                                strings = Rc::clone(&self.chunks[chunk_idx].strings);
                                ip = 0;
                                locals = args;
                                locals.resize(self.chunks[chunk_idx].num_locals.max(locals.len()), Value::Unit);
                                base = 0;
                            } else {
                                return Err(self.err_here(chunk_idx, ip, format!("未定义的函数 '{}'", name)));
                            }
                        }
                        _ => return Err(self.err_here(chunk_idx, ip, format!("期望可调用值，得到 {:?}", callee))),
                    }
                }
                58 => {
                    let n = r!(u64) as usize;
                    let callee = self.stack.pop().unwrap_or(Value::Unit);
                    let mut args = self.locals_pool.pop().unwrap_or_default();
                    args.resize(n, Value::Unit);
                    for i in (0..n).rev() { args[i] = self.stack.pop().unwrap_or(Value::Unit); }

                    match &callee {
                        Value::FnRef { name, captures, .. } => {
                            // a1 P3：追加捕获值作为额外实参（槽位 params..params+captures），
                            // 闭包 chunk 以 params+captures 个槽位接收。
                            for cap in captures {
                                args.push(cap.clone());
                            }
                            let callee_name: &str = if let Some(Value::FnRef { name: fname, .. }) = self.globals.get(name) {
                                fname.as_str()
                            } else {
                                name
                            };
                            if let Some(native_fn) = self.natives.get(callee_name).copied() {
                                let result = native_fn(self, &args).map_err(|e| self.with_line(chunk_idx, ip, e))?;
                                self.stack.push(result);
                            } else if let Some(&callee_idx) = self.functions.get(callee_name) {
                                // TCO：复用当前帧（同 TailCall），不压新帧
                                let mut old_locals = std::mem::replace(&mut locals, args);
                                old_locals.clear();
                                if self.locals_pool.len() < LOCALS_POOL_MAX {
                                    self.locals_pool.push(old_locals);
                                }
                                chunk_idx = callee_idx;
                                code = Rc::clone(&self.chunks[chunk_idx].code);
                                strings = Rc::clone(&self.chunks[chunk_idx].strings);
                                ip = 0;
                                locals.resize(self.chunks[chunk_idx].num_locals.max(locals.len()), Value::Unit);
                            } else {
                                return Err(self.err_here(chunk_idx, ip, format!("未定义的函数 '{}'", name)));
                            }
                        }
                        _ => return Err(self.err_here(chunk_idx, ip, format!("期望可调用值，得到 {:?}", callee))),
                    }
                }
                // AUDIT-11.4.21：59 MakeRef / 60 MakeMutRef / 61 Deref / 62 DerefStore
                // （引用语义，与解释器 eval.rs 对齐——不再 pass-through + 硬编码 Store(0)）
                59 => {
                    // &x → Value::Ref(独立 RefCell)
                    let v = self.stack.pop().unwrap_or(Value::Unit);
                    self.stack.push(Value::Ref(Rc::new(RefCell::new(v))));
                }
                60 => {
                    // &mut 变量 → 包装/复用 Shared 回写槽位 + Value::MutRef(Weak)
                    let i = r!(u64) as usize;
                    if i < locals.len() {
                        let v = locals[i].clone();
                        let rc = match v {
                            Value::Shared(rc) => rc,
                            other => {
                                let rc = Rc::new(RefCell::new(other));
                                locals[i] = Value::Shared(rc.clone());
                                rc
                            }
                        };
                        self.stack.push(Value::MutRef(Rc::downgrade(&rc)));
                    } else {
                        self.stack.push(Value::Moved);
                    }
                }
                61 => {
                    // *r → Ref/MutRef 读穿；非引用透传（VM 宽松，兼容旧 pass-through）
                    let v = self.stack.pop().unwrap_or(Value::Unit);
                    let out = match &v {
                        Value::Ref(rc) => rc.borrow().clone(),
                        Value::MutRef(w) => w.upgrade().map(|rc| rc.borrow().clone()).unwrap_or(Value::Moved),
                        other => other.clone(),
                    };
                    self.stack.push(out);
                }
                62 => {
                    // *m = v → 栈 [value, target]：target 为 MutRef 写穿 Weak（解释器一致）；
                    // Ref 也写穿（仅用于全局 &mut 退化路径，净效果与解释器局部影子一致）；
                    // 其他报错（与解释器「只能通过可变引用赋值」一致，避免静默错值）。
                    let value = self.stack.pop().unwrap_or(Value::Unit);
                    let target = self.stack.pop().unwrap_or(Value::Unit);
                    match &target {
                        Value::MutRef(w) => {
                            let rc = w.upgrade().ok_or_else(|| self.err_here(chunk_idx, ip, "无法通过悬垂的 &mut 引用赋值".into()))?;
                            *rc.borrow_mut() = value;
                        }
                        Value::Ref(rc) => {
                            *rc.borrow_mut() = value;
                        }
                        _ => return Err(self.err_here(chunk_idx, ip, "只能通过可变引用赋值".into())),
                    }
                    self.stack.push(Value::Unit);
                }
                // 未知 opcode：与旧 decode 的 `_ => Ret` 一致，执行 Ret 动作
                _ => {
                    let result = self.stack.pop().unwrap_or(Value::Unit);
                    if let Some(f) = self.frames.pop() {
                        // R5 locals 复用池：callee locals 清空入池（回退 Ret 后不再被引用）
                        let mut callee_locals = std::mem::take(&mut locals);
                        callee_locals.clear();
                        if self.locals_pool.len() < LOCALS_POOL_MAX {
                            self.locals_pool.push(callee_locals);
                        }
                        // Phase 2 栈迁移：恢复 caller 栈；callee 栈清空入池复用（R4：消除每调用一次 alloc/free）
                        let mut callee_stack = std::mem::replace(&mut self.stack, f.operand_stack);
                        callee_stack.clear();
                        if self.stack_pool.len() < STACK_POOL_MAX {
                            self.stack_pool.push(callee_stack);
                        }
                        self.stack.push(result);
                        ip = f.ip;
                        chunk_idx = f.chunk_idx;
                        code = Rc::clone(&self.chunks[chunk_idx].code);
                        strings = Rc::clone(&self.chunks[chunk_idx].strings);
                        locals = f.locals;
                        base = f.stack_base;
                        current_task_id = f.task_id;
                        self.current_task = current_task_id;
                    } else {
                        // 顶层：任务完成。把结果推回栈供调度器 pop。
                        if has_budget { self.step_budget = Some(budget_left); } // R1：预算回写
                        self.stack.push(result);
                        return Ok(YieldReason::Completed);
                    }
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
            Some(v) => match Self::deref_wrapped(&v) {
                Value::Int(n, _) => Ok(n),
                _ => err("期望整数"),
            },
            None => err("期望整数"),
        }
    }

    /// AUDIT-11.4.21：解包 Shared/Ref/MutRef/SharedBox（对齐解释器 natives::deref_wrapped）。
    /// `&mut` 后变量槽位变为 Value::Shared，算术/比较/取整前需解包再运算——
    /// 否则 VM 在 &mut 之后对变量做算术会报类型不匹配（解释器 eval_binary 已前置解包）。
    fn deref_wrapped(v: &Value) -> Value {
        match v {
            Value::Shared(rc) => rc.borrow().clone(),
            Value::Ref(rc) => rc.borrow().clone(),
            Value::MutRef(w) => w.upgrade().map(|rc| rc.borrow().clone()).unwrap_or(Value::Moved),
            Value::SharedBox(rc) => rc.borrow().clone(),
            other => other.clone(),
        }
    }

    fn is_wrapped(v: &Value) -> bool {
        matches!(v, Value::Shared(_) | Value::Ref(_) | Value::MutRef(_) | Value::SharedBox(_))
    }

    pub(super) fn add_priv(&mut self, a: &Value, b: &Value) -> TenthResult<Value> {
        // AUDIT-11.4.21：运算前解包包裹值（对齐解释器 eval_binary 前置 deref）
        if Self::is_wrapped(a) || Self::is_wrapped(b) {
            let a = Self::deref_wrapped(a);
            let b = Self::deref_wrapped(b);
            return self.add_priv(&a, &b);
        }
        Ok(match (a, b) {
            // AUDIT-11.4.17：checked_add 拦截 i64 层溢出（overflow-checks=true 下直接 + 会 panic）
            (Value::Int(x, dt), Value::Int(y, _)) => {
                let r = x.checked_add(*y).ok_or_else(|| int_overflow_err(*dt))?;
                check_int_overflow(r, *dt)?;
                Value::Int(r, *dt)
            },
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
        // AUDIT-11.4.21：运算前解包包裹值
        if Self::is_wrapped(a) || Self::is_wrapped(b) {
            let a = Self::deref_wrapped(a);
            let b = Self::deref_wrapped(b);
            return self.sub_priv(&a, &b);
        }
        Ok(match (a, b) {
            // AUDIT-11.4.17：checked_sub 拦截 i64 层溢出
            (Value::Int(x, dt), Value::Int(y, _)) => {
                let r = x.checked_sub(*y).ok_or_else(|| int_overflow_err(*dt))?;
                check_int_overflow(r, *dt)?;
                Value::Int(r, *dt)
            },
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
        // AUDIT-11.4.21：运算前解包包裹值
        if Self::is_wrapped(a) || Self::is_wrapped(b) {
            let a = Self::deref_wrapped(a);
            let b = Self::deref_wrapped(b);
            return self.mul_priv(&a, &b);
        }
        Ok(match (a, b) {
            // AUDIT-11.4.17：checked_mul 拦截 i64 层溢出
            (Value::Int(x, dt), Value::Int(y, _)) => {
                let r = x.checked_mul(*y).ok_or_else(|| int_overflow_err(*dt))?;
                check_int_overflow(r, *dt)?;
                Value::Int(r, *dt)
            },
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
        // AUDIT-11.4.21：运算前解包包裹值
        if Self::is_wrapped(a) || Self::is_wrapped(b) {
            let a = Self::deref_wrapped(a);
            let b = Self::deref_wrapped(b);
            return self.div_priv(&a, &b);
        }
        Ok(match (a, b) {
            (Value::Int(x, dt), Value::Int(y, _)) => {
                if *y == 0 {
                    return err("整数除零");
                }
                // AUDIT-11.4.17：checked_div 拦截 i64::MIN / -1 等溢出
                let r = x.checked_div(*y).ok_or_else(|| int_overflow_err(*dt))?;
                check_int_overflow(r, *dt)?;
                Value::Int(r, *dt)
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
        // AUDIT-11.4.21：比较前解包包裹值（&mut 后变量为 Shared）
        if Self::is_wrapped(a) || Self::is_wrapped(b) {
            let a = Self::deref_wrapped(a);
            let b = Self::deref_wrapped(b);
            return self.compare(&a, &b, nf, sf);
        }
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
        // AUDIT-11.4.21/24：比较前解包包裹值（&mut 后变量为 Shared；对齐解释器 values_eq 前置 deref）
        if Self::is_wrapped(a) || Self::is_wrapped(b) {
            return self.vm_eq(&Self::deref_wrapped(a), &Self::deref_wrapped(b));
        }
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
            // AUDIT-11.4.24：Vec 相等比较——元素逐一相等（VM 元素为普通值）。
            // 此前落入 `_ => false`，元素相同也返回 false（base64/hex 断言静默失败）。
            (Value::Vec(a), Value::Vec(b)) => {
                let a = a.borrow();
                let b = b.borrow();
                a.len() == b.len()
                    && a.iter().zip(b.iter()).all(|(x, y)| self.vm_eq(x, y))
            }
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
