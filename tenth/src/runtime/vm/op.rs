//! VM opcode 枚举、调度器信号、Frame 结构。
//!
//! 从 runtime/vm.rs 拆分而来（T3b 架构重构）。

// ── Opcode ─────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Op {
    PushInt(i64), PushFloat(f64), PushFloat32(f32), PushBool(bool), PushChar(u32), PushStr(usize), PushUnit,
    Pop, Dup,
    Load(usize), Store(usize),
    LoadGlobal(usize), StoreGlobal(usize),
    Add, Sub, Mul, Div, Mod, Neg, Not,
    Eq, Neq, Lt, Gt, Lte, Gte,
    Jump(i32), JmpFalse(i32), JmpTrue(i32),
    Call(usize), CallN(usize, usize), MethodCall(usize, usize), Ret,
    MakeVec(usize), MakeMap(usize),
    NewStruct(usize, usize), LoadField(usize), StoreField(usize),
    NewUnion(usize, usize),    // M1.2：union 构造 — name_idx, active_field_idx；弹出栈顶 value → Value::Union
    IndexGet,
    SliceStr,
    MakeEnum(usize, usize, usize),
    IsEnumVariant(usize),
    EnumGetField(usize),
    IsStruct(usize),
    PushRange(i64, i64, bool),  // start, end, inclusive
    MoveOp,                     // no-op marker for move semantics
    MakeTensor(usize, usize, u8), // rows, cols, dtype (0=F64, 1=F32) — pops rows*cols values
    MakeClosure(usize, usize, usize),  // params_count, captures_count, chunk_idx — creates a closure value
    Await,
    Spawn,
    MakeTuple(usize),           // n — pops n values, pushes Value::Tuple
    IsTuple(usize),             // expected_len — pops value, pushes Bool(val is Tuple with len)
    TupleGet(usize),            // index — pops Tuple, pushes element at index
    Try,                        // pops Result; Ok(v) → push v; Err(e) → early return TryPropagate(e)
    Yield,                      // 协作式调度：让出控制权，当前 task 回到 ready_queue 尾部
    TailCall(usize, usize),     // TCO：函数名索引 + 参数数量 — 不压新帧，复用当前帧替换 PC 和 slot
    // a1（VM 闭包值调用 / CallIndirect）P1：间接调用栈上闭包/函数值。
    // 编码 57/58 追加在尾部，不动既有指令编码（最大 56 = NewUnion）。
    CallClosure(usize),         // 参数数量 n — 栈上 [arg1..argN, callee] → 弹 callee + N 参数 → 压新帧调用
    TailCallClosure(usize),     // TCO：同上但复用当前帧（尾调用）；JIT 不支持 → fallback VM
    // AUDIT-11.4.21（&mut 写回顺序失效）：引用语义 opcodes（编码 59-62 追加在尾部）。
    // 与解释器 eval.rs 的引用处理对齐：&x → Value::Ref；&mut 变量 → 复用/创建
    // Value::Shared 槽位 + Value::MutRef(Weak)；*m = v → DerefStore 写穿 Weak。
    MakeRef,                    // 弹值 → Value::Ref(Rc(RefCell(v)))
    MakeMutRef(usize),          // slot — locals[slot] 包装/复用 Shared 回写槽位，压 Value::MutRef(Weak)
    Deref,                      // 弹值；Ref/MutRef 读穿，其他透传（VM 宽松，兼容旧 pass-through）
    DerefStore,                 // 栈 [value, target] — target 为 MutRef/Ref 写穿，其他报错
}

// ── Scheduler internals ─────────────────────────────────────────────────────

/// 协程任务标识。0 保留给主任务；其余由 `next_task_id` 递增分配。
pub(super) type TaskId = u64;

/// `run_until_yield` 的退出原因。Phase 2 调度器据此决定下一步调度动作。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum YieldReason {
    /// 任务正常完成（顶层 Ret）。结果已 push 到 `Vm.stack`，调度器负责 pop。
    Completed,
    /// 任务在 `await` 处挂起，等待某个 Pending Future。
    /// 当前调用栈已保存到 `suspended[task_id]`，不自动重新入队——
    /// 待 Future 就绪时由调度器（或 Step 5 的 I/O 事件）唤醒。
    Suspended(TaskId),
    /// 任务主动 `yield` 让出控制权。当前调用栈已保存到 `suspended[task_id]`，
    /// 且 `task_id` 已被推回 `ready_queue` 尾部，等待下次调度。
    Yield(TaskId),
}

pub(super) struct Frame {
    pub(super) ip: usize,
    pub(super) chunk_idx: usize,
    pub(super) locals: Vec<crate::runtime::value::Value>,
    /// 兼容字段：保留以减小本次改动的扩散面。新栈模型下不再依赖此值做 truncate。
    pub(super) stack_base: usize,
    /// 该 Frame 私有的操作数栈。Phase 2：每任务独立栈，支持协程挂起/恢复。
    /// 切换 Frame 时与 `Vm.stack` 做 swap：caller 栈保存在此，callee 使用 `Vm.stack`。
    pub(super) operand_stack: Vec<crate::runtime::value::Value>,
    /// 该 Frame 所属的协程任务标识。普通函数调用继承 caller 的 task_id；
    /// spawn 创建新协程时分配新 task_id（Phase 2 调度器使用）。
    pub(super) task_id: TaskId,
}
