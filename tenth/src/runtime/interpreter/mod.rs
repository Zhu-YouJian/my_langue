//! 树遍解释器（拆分自原 `interpreter.rs` 上帝文件）。
//!
//! 模块结构：
//! - `mod.rs`（本文件）：`Interpreter` 结构体、构造、作用域管理、`eval_expr`、
//!   `eval_call`、`eval_stmt`、自动微分记录辅助、`unwrap_return`
//! - `json.rs`：JSON 编解码（带 H-6 安全修复）
//! - `datetime.rs`：Unix 天数 → (年, 月, 日) 转换
//! - `binary.rs`：二元/一元运算、值比较与字符串化
//! - `pattern.rs`：字段访问与模式匹配
//! - `methods.rs`：方法分派（String/Vec/Map/Range/Iterator/Tensor/Scalar）
//! - `index.rs`：索引与切片
//! - `natives.rs`：原生函数注册（`call_named_fn`）

pub mod json;
pub mod datetime;
mod binary;
mod pattern;
mod methods;
mod natives;
mod index;

use std::collections::HashMap;
use std::rc::Rc;
use std::cell::RefCell;
use crate::error::{TenthError, TenthResult};
use crate::hir::hir::*;
use crate::hir::types::Type;
use super::value::Value;
use super::tensor::Tensor;
use super::limits::RuntimeLimits;
use super::arena::Arena;
use super::autodiff::{Tape, TapeOp};

/// Default arena capacity when no explicit limit is configured.
const DEFAULT_ARENA_CAPACITY: usize = 64 * 1024; // 64K f64 slots = 512 KB

pub struct Interpreter {
    /// AUDIT-11.4.3: 扁平化 `name → Vec<(scope_depth, Value)>` 索引。
    /// 每个 name 关联一个栈，支持 shadowing；`resolve_var` 为 O(1)。
    /// 取代旧的 `Vec<HashMap<String, Value>>` 双重存储（lookup 在重嵌套场景下 O(n*m)）。
    pub vars: HashMap<String, Vec<(usize, Value)>>,
    /// 当前 scope 深度（0 = 全局）。`push_scope` 递增，`pop_scope` 递减。
    scope_depth: usize,
    /// 每层 scope 中插入的变量名列表，`pop_scope` 时只清理这些变量（O(m)，m 为本层变量数）。
    scope_vars: Vec<Vec<String>>,
    functions: Vec<HirFnDef>,
    generic_funcs: HashMap<String, HirFnDef>,
    methods: HashMap<String, HashMap<String, HirFnDef>>,
    modules: HashMap<String, HirProgram>,
    trait_impls: HashMap<String, HashMap<String, HashMap<String, HirFnDef>>>,
    /// Resource limits — when set, allocations are checked against caps.
    pub limits: Option<RuntimeLimits>,
    /// Arena for temporary tensor/computation data.
    /// Reset via scope around each top-level evaluation.
    pub arena: Arena,
    /// Autodiff computation tape (active when `recording` is true).
    pub tape: Option<Tape>,
    /// Whether tensor operations should be recorded on the tape.
    pub recording: bool,
    /// Execution step budget. When `Some(n)`, each `eval_expr`/`eval_stmt`
    /// decrements the counter; reaching zero raises `TenthError::Timeout`.
    /// `None` means unlimited (default). Set via `with_step_limit`.
    pub step_budget: Option<u64>,
    /// Optional wall-clock deadline (Unix ms). When set, the step counter
    /// periodically checks `now >= deadline` and raises `Timeout`.
    /// Set via `with_timeout_ms`.
    pub deadline_ms: Option<u128>,
    /// H-4: 独立的 tick 计数器，用于触发周期性 deadline 检查。
    /// 不依赖 step_budget（用户可能只设 --timeout 而不设步数预算）。
    tick_counter: u64,
    /// H-2: 文件系统沙箱。`Some` 时所有文件 I/O 原生函数必须经过校验。
    /// `None` 表示无沙箱（默认，向后兼容）。
    pub fs_sandbox: Option<crate::runtime::limits::FsSandbox>,
    /// 护城河 F：上一次 backward 失败时的根因说明列表（由 formal_explain 生成）。
    /// 由 `explain_error()` native 读取并清空。
    pub last_explanation: Vec<String>,
    /// TCP 流句柄表（与 vm.rs 的 Vm::tcp_streams 对齐）。
    /// 索引+1 即句柄（1-based，0 表示无效）。`None` 表示已关闭的槽位。
    pub tcp_streams: Vec<Option<std::net::TcpStream>>,
}

impl Interpreter {
    pub fn new(program: &HirProgram) -> Self {
        Interpreter {
            vars: HashMap::new(),
            scope_depth: 0,
            scope_vars: vec![Vec::new()],
            functions: program.functions.clone(),
            generic_funcs: HashMap::new(),
            methods: program.methods.clone(),
            modules: program.modules.clone(),
            trait_impls: program.trait_impls.clone(),
            limits: None,
            arena: Arena::new(DEFAULT_ARENA_CAPACITY),
            tape: None,
            recording: false,
            step_budget: None,
            deadline_ms: None,
            tick_counter: 0,
            fs_sandbox: None,
            last_explanation: Vec::new(),
            tcp_streams: Vec::new(),
        }
    }

    /// AUDIT-11.4.3: 进入新 scope（depth +1）。
    pub(super) fn push_scope(&mut self) {
        self.scope_depth += 1;
        self.scope_vars.push(Vec::new());
    }

    /// AUDIT-11.4.3: 退出当前 scope，清理本层变量（O(m)，m 为本层变量数）。
    /// 全局 scope（depth 0）不可 pop。
    pub(super) fn pop_scope(&mut self) {
        if self.scope_depth == 0 {
            return;
        }
        let names = self.scope_vars.pop().unwrap_or_default();
        for name in &names {
            if let Some(stack) = self.vars.get_mut(name) {
                while stack.last().map_or(false, |(d, _)| *d == self.scope_depth) {
                    stack.pop();
                }
                if stack.is_empty() {
                    self.vars.remove(name);
                }
            }
        }
        self.scope_depth -= 1;
    }

    /// AUDIT-11.4.3: 在当前 scope 插入/覆盖变量。
    /// 同一 scope 内同名变量为覆盖语义（与原 HashMap::insert 一致）；
    /// 不同 scope 的同名变量为 shadowing（栈追加）。
    pub(super) fn insert_var(&mut self, name: String, val: Value) {
        let stack = self.vars.entry(name.clone()).or_default();
        if stack.last().map_or(false, |(d, _)| *d == self.scope_depth) {
            stack.last_mut().unwrap().1 = val;
        } else {
            stack.push((self.scope_depth, val));
            if let Some(scope) = self.scope_vars.get_mut(self.scope_depth) {
                scope.push(name);
            }
        }
    }

    /// AUDIT-11.4.3: 从当前 scope 移除变量（用于模式绑定清理）。
    pub(super) fn remove_var(&mut self, name: &str) -> Option<Value> {
        let mut removed = None;
        if let Some(stack) = self.vars.get_mut(name) {
            if stack.last().map_or(false, |(d, _)| *d == self.scope_depth) {
                removed = stack.pop().map(|(_, v)| v);
                if stack.is_empty() {
                    self.vars.remove(name);
                }
            }
        }
        if removed.is_some() {
            if let Some(scope) = self.scope_vars.get_mut(self.scope_depth) {
                scope.retain(|n| n != name);
            }
        }
        removed
    }

    /// AUDIT-11.4.3: REPL 注入全局变量（在 scope_depth==0 时调用）。
    pub fn extend_globals(&mut self, vars: HashMap<String, Value>) {
        for (name, val) in vars {
            self.insert_var(name, val);
        }
    }

    /// AUDIT-11.4.3: REPL 提取全局变量（depth==0 的条目）。
    pub fn globals_clone(&self) -> HashMap<String, Value> {
        let mut result = HashMap::new();
        for (name, stack) in &self.vars {
            if let Some((depth, val)) = stack.first() {
                if *depth == 0 {
                    result.insert(name.clone(), val.clone());
                }
            }
        }
        result
    }

    /// Create an interpreter with resource limits enforced.
    /// The arena capacity is derived from max_arena_bytes.
    pub fn with_limits(program: &HirProgram, limits: RuntimeLimits) -> Self {
        let arena_elems = limits.config.max_arena_bytes / std::mem::size_of::<f64>();
        let arena_cap = arena_elems.min(usize::MAX / 2).max(1024);
        let mut interp = Interpreter::new(program);
        interp.limits = Some(limits);
        interp.arena = Arena::new(arena_cap);
        interp
    }

    pub fn execute_program(&mut self, program: &HirProgram) -> TenthResult<Option<Value>> {
        self.insert_var(
            "tensor".to_string(),
            Value::FnRef {
                name: "tensor".to_string(),
                params: vec![("data".to_string(), Type::Unknown)],
                return_type: Type::Unknown,
            },
        );

        // Autodiff builtins
        self.insert_var(
            "start_grad".to_string(),
            Value::FnRef {
                name: "start_grad".to_string(),
                params: vec![],
                return_type: Type::unit(),
            },
        );
        self.insert_var(
            "new_grad".to_string(),
            Value::FnRef {
                name: "new_grad".to_string(),
                params: vec![],
                return_type: Type::unit(),
            },
        );
        self.insert_var(
            "stop_grad".to_string(),
            Value::FnRef {
                name: "stop_grad".to_string(),
                params: vec![],
                return_type: Type::unit(),
            },
        );
        self.insert_var(
            "zero_grad".to_string(),
            Value::FnRef {
                name: "zero_grad".to_string(),
                params: vec![],
                return_type: Type::unit(),
            },
        );
        self.insert_var(
            "cross_entropy".to_string(),
            Value::FnRef {
                name: "cross_entropy".to_string(),
                params: vec![
                    ("logits".to_string(), Type::Unknown),
                    ("target".to_string(), Type::Unknown),
                ],
                return_type: Type::Unknown,
            },
        );
        // select 原语（论文 T47/T48/T50）：逐元素条件选择，支持广播与可微
        self.insert_var(
            "select".to_string(),
            Value::FnRef {
                name: "select".to_string(),
                params: vec![
                    ("cond".to_string(), Type::Unknown),
                    ("then".to_string(), Type::Unknown),
                    ("else".to_string(), Type::Unknown),
                ],
                return_type: Type::Unknown,
            },
        );
        // scatter 原语：不可变散布，按 index 沿 dim 覆盖 base 的对应位置
        self.insert_var(
            "scatter".to_string(),
            Value::FnRef {
                name: "scatter".to_string(),
                params: vec![
                    ("base".to_string(), Type::Unknown),
                    ("dim".to_string(), Type::Unknown),
                    ("index".to_string(), Type::Unknown),
                    ("src".to_string(), Type::Unknown),
                ],
                return_type: Type::Unknown,
            },
        );
        // gather 原语：沿 dim 维按 index 取值（与 PyTorch gather 对齐）
        self.insert_var(
            "gather".to_string(),
            Value::FnRef {
                name: "gather".to_string(),
                params: vec![
                    ("base".to_string(), Type::Unknown),
                    ("dim".to_string(), Type::Unknown),
                    ("index".to_string(), Type::Unknown),
                ],
                return_type: Type::Unknown,
            },
        );
        // Scalar math
        for name in &["abs", "sqrt", "sin", "cos", "ln", "pow"] {
            self.insert_var(
                name.to_string(),
                Value::FnRef {
                    name: name.to_string(),
                    params: vec![("x".to_string(), Type::Unknown)],
                    return_type: Type::Unknown,
                },
            );
        }
        // Tensor creation
        for name in &["zeros", "ones"] {
            self.insert_var(
                name.to_string(),
                Value::FnRef {
                    name: name.to_string(),
                    params: vec![("dims".to_string(), Type::Unknown)],
                    return_type: Type::Unknown,
                },
            );
        }
        // Serialization
        for name in &["save_weights", "load_weights"] {
            self.insert_var(
                name.to_string(),
                Value::FnRef {
                    name: name.to_string(),
                    params: vec![("path".to_string(), Type::Unknown)],
                    return_type: Type::Unknown,
                },
            );
        }
        self.insert_var(
            "param".to_string(),
            Value::FnRef {
                name: "param".to_string(),
                params: vec![("t".to_string(), Type::Unknown)],
                return_type: Type::Unknown,
            },
        );
        self.insert_var(
            "backward".to_string(),
            Value::FnRef {
                name: "backward".to_string(),
                params: vec![("loss".to_string(), Type::Unknown)],
                return_type: Type::unit(),
            },
        );
        self.insert_var(
            "grad".to_string(),
            Value::FnRef {
                name: "grad".to_string(),
                params: vec![("param".to_string(), Type::Unknown)],
                return_type: Type::Unknown,
            },
        );
        // 护城河 F：explain_error() — 返回上一次 backward 失败的根因说明列表
        self.insert_var(
            "explain_error".to_string(),
            Value::FnRef {
                name: "explain_error".to_string(),
                params: vec![],
                return_type: Type::Unknown,
            },
        );

        for func in &program.functions {
            let params = func.params.clone();
            let ret = func.return_type.clone();
            self.insert_var(
                func.name.clone(),
                Value::FnRef {
                    name: func.name.clone(),
                    params: params.clone(),
                    return_type: ret.clone(),
                },
            );
        }

        for func in &program.generic_funcs {
            self.generic_funcs.insert(func.name.clone(), func.clone());
        }

        for (use_path, alias) in &program.uses {
            if use_path.len() >= 2 {
                let mod_name = &use_path[0];
                let fn_name = &use_path[1];
                if let Some(module) = self.modules.get(mod_name) {
                    if let Some(fn_def) = module.functions.iter().find(|f| &f.name == fn_name) {
                        let params = fn_def.params.clone();
                        let ret = fn_def.return_type.clone();
                        self.functions.push(fn_def.clone());
                        self.insert_var(
                            alias.clone(),
                            Value::FnRef {
                                name: alias.clone(),
                                params,
                                return_type: ret,
                            },
                        );
                    }
                }
            }
        }

        // Reset arena at the start of each top-level evaluation.
        // Any temporary allocations from previous evaluations are freed.
        self.arena.reset();

        if let Some(ref expr) = program.main_expr {
            Self::unwrap_return(self.eval_expr(expr))
        } else if let Some(main_fn) = self.functions.iter().find(|f| f.name == "main") {
            let body = main_fn.body.clone();
            Self::unwrap_return(self.eval_expr(&body))
        } else {
            Ok(None)
        }
    }

    pub(super) fn resolve_var(&self, name: &str) -> Option<Value> {
        // AUDIT-11.4.3: 扁平化索引，O(1) lookup。
        match self.vars.get(name).and_then(|s| s.last()) {
            Some((_, Value::Shared(rc))) => Some(rc.borrow().clone()),
            Some((_, Value::Moved)) => None,
            Some((_, other)) => Some(other.clone()),
            None => None,
        }
    }

    pub(super) fn set_var(&mut self, name: String, val: Value) {
        // Guard: check variable count before inserting a new one.
        // AUDIT-11.4.3: 用 vars.len() 替代 scopes 遍历求和。
        // 新计数为不同变量名数（shadowing 不重复计数），更合理。
        let total_vars: usize = self.vars.len();
        if let Some(ref limits) = self.limits {
            if !self.vars.contains_key(&name) {
                if let Err(msg) = limits.guard_vars(total_vars) {
                    eprintln!("[limits] variable limit: {}", msg);
                    if cfg!(feature = "mem-strict") {
                        panic!("variable limit exceeded: {}", msg);
                    }
                }
            }
        }
        // AUDIT-11.4.3: 用扁平化索引定位变量，O(1)。
        // 栈顶即最内层 scope 的条目（含 shadowing）。
        if let Some(stack) = self.vars.get_mut(&name) {
            if let Some((_, existing)) = stack.last_mut() {
                match existing {
                    Value::Shared(rc) => {
                        *rc.borrow_mut() = val;
                        return;
                    }
                    Value::Moved => {
                        // Variable was moved; re-insert the new value in this scope.
                        *existing = val;
                        return;
                    }
                    _ => {
                        // Plain value: overwrite in the same scope where it was found.
                        *existing = val;
                        return;
                    }
                }
            }
        }
        // Variable not found in any scope: insert into current scope (new definition).
        self.insert_var(name, val);
    }

    /// Execution step counter. Called once per `eval_expr`/`eval_stmt`.
    /// Raises `Timeout` when the budget is exhausted or the wall-clock
    /// deadline passes. No-op when neither limit is set (default).
    #[inline]
    fn tick(&mut self) -> TenthResult<()> {
        // 安全 H-4：step_budget 和 deadline_ms 独立检查。
        // 历史实现把 deadline 检查嵌套在 step_budget 内，导致只设
        // `--timeout` 而不设 step_budget 时 deadline 永远不触发。
        if let Some(ref mut budget) = self.step_budget {
            if *budget == 0 {
                return Err(TenthError::Timeout {
                    message: "步数预算耗尽".into(),
                });
            }
            *budget -= 1;
        }
        // 用独立的 tick 计数器触发周期性 deadline 检查，
        // 避免依赖 step_budget（用户可能只设 --timeout）。
        self.tick_counter = self.tick_counter.wrapping_add(1);
        if (self.tick_counter & 0xFFF) == 0 {
            if let Some(deadline) = self.deadline_ms {
                let now = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_millis())
                    .unwrap_or(0);
                if now >= deadline {
                    return Err(TenthError::Timeout {
                        message: "时间预算耗尽".into(),
                    });
                }
            }
        }
        Ok(())
    }

    /// Guarded tensor creation: checks element count against limits.
    pub fn make_tensor(&self, data: Vec<f64>, shape: Vec<usize>) -> TenthResult<Tensor> {
        let elements = data.len();
        if let Some(ref limits) = self.limits {
            if let Err(msg) = limits.guard_tensor(elements) {
                return Err(TenthError::RuntimeError { message: msg });
            }
        }
        Ok(Tensor::from_vec(data, shape))
    }

    pub(super) fn eval_expr(&mut self, expr: &HirExpr) -> TenthResult<Option<Value>> {
        use HirExprKind;

        self.tick()?;
        match &expr.kind {
            HirExprKind::Literal(lit) => {
                Ok(Some(match lit {
                    Literal::Int(n) => Value::Int(*n),
                    Literal::Float(n, dt) => match dt {
                        crate::hir::types::BaseType::F32 => Value::Float32(*n as f32),
                        _ => Value::Float(*n),
                    },
                    Literal::Bool(b) => Value::Bool(*b),
                    Literal::String(s) => Value::String(s.clone()),
                }))
            }

            HirExprKind::Var(name) => {
                if name.contains("::") {
                    let parts: Vec<&str> = name.splitn(2, "::").collect();
                    if parts.len() == 2 {
                        let mod_name = parts[0];
                        let item_name = parts[1];
                        if let Some(module) = self.modules.get(mod_name) {
                            if let Some(fn_def) = module.functions.iter().find(|f| f.name == item_name) {
                                return Ok(Some(Value::FnRef {
                                    name: name.clone(),
                                    params: fn_def.params.clone(),
                                    return_type: fn_def.return_type.clone(),
                                }));
                            }
                        }
                    }
                    return Ok(Some(Value::FnRef {
                        name: name.clone(),
                        params: Vec::new(),
                        return_type: Type::Unknown,
                    }));
                }
                if self.vars.get(name).map_or(false, |s| s.iter().any(|(_, v)| matches!(v, Value::Moved))) {
                    return Err(TenthError::RuntimeError {
                        message: format!("使用了已移动的值 '{}'", name),
                    });
                }
                self.resolve_var(name)
                    .or_else(|| {
                        match name.as_str() {
                            "println" | "print" | "eprintln" | "eprint" | "tensor" | "rand" | "randn" | "randn_f32" | "rand_f32" | "zeros_f32" | "ones_f32"
                            | "read_file" | "write_file" | "write_bytes" | "read_bytes" | "compile_host"
                            | "compile_program"
                            | "read_line" | "env_get" | "env_set" | "exit"
                            | "Vec::new" | "HashMap::new"
                            | "start_grad" | "new_grad" | "stop_grad"
                            | "param" | "backward" | "grad" | "zero_grad"
                            | "cross_entropy"
                            | "select"
                            | "scatter"
                            | "gather"
                            | "abs" | "sqrt" | "sin" | "cos" | "ln" | "pow" | "to_float" | "to_f32" | "to_f64" | "tensor_from_vec"
                            | "f64_bits" | "f64_from_bits"
                            | "zeros" | "ones"
                            | "save_weights" | "load_weights"
                            | "format" | "parse_int" | "parse_float"
                            | "to_string" | "type_name"
                            | "with_step_limit" | "with_timeout_ms" | "is_timeout"
                            | "path_join" | "path_exists" | "path_is_file" | "path_is_dir"
                            | "mkdir" | "list_dir" | "file_size" | "remove_file" | "copy_file" | "rename_file"
                            | "time_now" | "time_now_ms" | "time_date" | "time_time" | "time_datetime" | "time_sleep_ms"
                            | "random_int" | "random_float"
                            | "math_tan" | "math_asin" | "math_acos" | "math_atan" | "math_atan2"
                            | "math_sinh" | "math_cosh" | "math_tanh" | "math_log10" | "math_log2" | "math_exp" | "math_pow"
                            | "math_floor" | "math_ceil" | "math_round"
                            | "cli_args_count" | "cli_arg"
                            | "json_encode" | "json_encode_pretty" | "json_decode"
                            // Stage 3+4 TCP/HTTP 原语
                            | "tcp_connect" | "tcp_read" | "tcp_write" | "tcp_close" | "tcp_set_timeout"
                            | "http_get" | "http_post" => {
                                Some(Value::FnRef {
                                    name: name.clone(),
                                    params: Vec::new(),
                                    return_type: Type::Unknown,
                                })
                            }
                            _ => None,
                        }
                    })
                    .ok_or_else(|| TenthError::RuntimeError {
                        message: format!("未定义变量 '{}'", name),
                    })
                    .map(Some)
            }

            HirExprKind::Binary { op, left, right, .. } => {
                // Short-circuit evaluation for && and ||
                match op {
                    BinOp::And => {
                        let l = self.eval_expr(left)?.ok_or_else(|| TenthError::RuntimeError {
                            message: "左操作数为空值".into(),
                        })?;
                        if !l.is_truthy() {
                            return Ok(Some(Value::Bool(false)));
                        }
                        let r = self.eval_expr(right)?.ok_or_else(|| TenthError::RuntimeError {
                            message: "右操作数为空值".into(),
                        })?;
                        Ok(Some(Value::Bool(r.is_truthy())))
                    }
                    BinOp::Or => {
                        let l = self.eval_expr(left)?.ok_or_else(|| TenthError::RuntimeError {
                            message: "左操作数为空值".into(),
                        })?;
                        if l.is_truthy() {
                            return Ok(Some(Value::Bool(true)));
                        }
                        let r = self.eval_expr(right)?.ok_or_else(|| TenthError::RuntimeError {
                            message: "右操作数为空值".into(),
                        })?;
                        Ok(Some(Value::Bool(r.is_truthy())))
                    }
                    _ => {
                        let l = self.eval_expr(left)?.ok_or_else(|| TenthError::RuntimeError {
                            message: "左操作数为空值".into(),
                        })?;
                        let r = self.eval_expr(right)?.ok_or_else(|| TenthError::RuntimeError {
                            message: "右操作数为空值".into(),
                        })?;
                        self.eval_binary(op, &l, &r).map(Some)
                    }
                }
            }

            HirExprKind::Unary { op, expr: inner, .. } => {
                let val = self.eval_expr(inner)?.ok_or_else(|| TenthError::RuntimeError {
                    message: "一元操作数为空值".into(),
                })?;
                self.eval_unary(op, &val).map(Some)
            }

            HirExprKind::Call { func, args, .. } => {
                let f = self.eval_expr(func)?.ok_or_else(|| TenthError::RuntimeError {
                    message: "函数值为空值".into(),
                })?;

                let mut arg_values = Vec::new();
                for a in args {
                    arg_values.push(self.eval_expr(a)?.ok_or_else(|| TenthError::RuntimeError {
                        message: "参数为空值".into(),
                    })?);
                }

                self.eval_call(&f, &arg_values, &expr.span)
            }

            HirExprKind::GenericCall { func, generics, args, .. } => {
                let func_name = match &func.kind {
                    HirExprKind::Var(name) => name.clone(),
                    _ => {
                        return Err(TenthError::RuntimeError {
                            message: "泛型调用的目标必须是具名函数".into(),
                        });
                    }
                };

                let template = self.generic_funcs.get(&func_name)
                    .ok_or_else(|| TenthError::RuntimeError {
                        message: format!("未定义的泛型函数 '{}'", func_name),
                    })?
                    .clone();

                let mut type_map: HashMap<String, Type> = HashMap::new();
                for (i, gen_name) in template.generics.iter().enumerate() {
                    type_map.insert(gen_name.clone(), generics.get(i).cloned().unwrap_or(Type::Unknown));
                }

                let mut arg_values = Vec::new();
                for a in args {
                    arg_values.push(self.eval_expr(a)?.ok_or_else(|| TenthError::RuntimeError {
                        message: "参数为空值".into(),
                    })?);
                }

                self.push_scope();
                for ((pname, _), arg) in template.params.iter().zip(arg_values.iter()) {
                    self.insert_var(pname.clone(), arg.clone());
                }

                let result = self.eval_expr(&template.body);

                self.pop_scope();
                result
            }

            HirExprKind::MethodCall { receiver, method, args, .. } => {
                let recv = self.eval_expr(receiver)?.ok_or_else(|| TenthError::RuntimeError {
                    message: "接收者为空值".into(),
                })?;

                let mut arg_values = Vec::new();
                for a in args {
                    arg_values.push(self.eval_expr(a)?.ok_or_else(|| TenthError::RuntimeError {
                        message: "方法参数为空值".into(),
                    })?);
                }

                self.eval_method_call(&recv, method, &arg_values)
            }

            HirExprKind::Index { target, indices } => {
                let t = self.eval_expr(target)?.ok_or_else(|| TenthError::RuntimeError {
                    message: "索引目标为空值".into(),
                })?;
                self.eval_index(&t, indices).map(Some)
            }

            HirExprKind::Field { target, field } => {
                let t = self.eval_expr(target)?.ok_or_else(|| TenthError::RuntimeError {
                    message: "字段访问目标为空值".into(),
                })?;
                self.eval_field(&t, field)
            }

            HirExprKind::ArrayLiteral { elements, .. } => {
                let mut vals = Vec::new();
                for elem in elements {
                    let v = self.eval_expr(elem)?.ok_or_else(|| TenthError::RuntimeError {
                        message: "数组元素为空值".into(),
                    })?;
                    // Wrap in Shared so elements can be mutated via indexed assignment
                    vals.push(Value::Shared(Rc::new(RefCell::new(v))));
                }
                Ok(Some(Value::Vec(Rc::new(RefCell::new(vals)))))
            }

            HirExprKind::TensorLiteral { data, ty, .. } => {
                // 按 HIR 类型注解的 dtype 选择构造路径
                let is_f32 = ty.tensor_dtype() == Some(crate::hir::types::BaseType::F32);
                let mut rows_f32: Vec<Vec<f32>> = Vec::new();
                let mut rows_f64: Vec<Vec<f64>> = Vec::new();
                for row in data {
                    let mut row_f32: Vec<f32> = Vec::new();
                    let mut row_f64: Vec<f64> = Vec::new();
                    for elem in row {
                        let v = self.eval_expr(elem)?.ok_or_else(|| TenthError::RuntimeError {
                            message: "张量元素为空值".into(),
                        })?;
                        if is_f32 {
                            row_f32.push(v.as_f32().unwrap_or(0.0));
                        } else {
                            row_f64.push(v.as_float().unwrap_or(0.0));
                        }
                    }
                    if is_f32 { rows_f32.push(row_f32); } else { rows_f64.push(row_f64); }
                }
                let nrows = if is_f32 { rows_f32.len() } else { rows_f64.len() };
                let ncols = if is_f32 {
                    rows_f32.first().map(|r| r.len()).unwrap_or(0)
                } else {
                    rows_f64.first().map(|r| r.len()).unwrap_or(0)
                };
                if is_f32 {
                    let flat: Vec<f32> = rows_f32.into_iter().flatten().collect();
                    let tensor = Tensor::from_vec_f32(flat, vec![nrows, ncols]);
                    Ok(Some(Value::Tensor(Rc::new(RefCell::new(tensor)))))
                } else {
                    let flat: Vec<f64> = rows_f64.into_iter().flatten().collect();
                    let tensor = self.make_tensor(flat, vec![nrows, ncols])?;
                    Ok(Some(Value::Tensor(Rc::new(RefCell::new(tensor)))))
                }
            }

            HirExprKind::If { cond, then_branch, else_branch, .. } => {
                let c = self.eval_expr(cond)?.ok_or_else(|| TenthError::RuntimeError {
                    message: "if 条件为空值".into(),
                })?;
                if c.is_truthy() {
                    self.eval_expr(then_branch)
                } else if let Some(eb) = else_branch {
                    self.eval_expr(eb)
                } else {
                    Ok(Some(Value::Unit))
                }
            }

            HirExprKind::Block { stmts, final_expr } => {
                for stmt in stmts {
                    self.eval_stmt(stmt)?;
                }
                match final_expr {
                    Some(e) => self.eval_expr(e),
                    None => Ok(Some(Value::Unit)),
                }
            }

            HirExprKind::Assign { target, value } => {
                let v = self.eval_expr(value)?.ok_or_else(|| TenthError::RuntimeError {
                    message: "赋值值为空值".into(),
                })?;
                self.set_var(target.clone(), v);
                Ok(Some(Value::Unit))
            }

            HirExprKind::DerefAssign { target, value } => {
                let target_val = self.eval_expr(target)?.ok_or_else(|| TenthError::RuntimeError {
                    message: "解引用赋值目标为空值".into(),
                })?;
                let rhs = self.eval_expr(value)?.ok_or_else(|| TenthError::RuntimeError {
                    message: "解引用赋值值为空值".into(),
                })?;
                match &target_val {
                    Value::MutRef(weak) => {
                        let rc = weak.upgrade().ok_or_else(|| TenthError::RuntimeError {
                            message: "无法通过悬垂的 &mut 引用赋值".into(),
                        })?;
                        *rc.borrow_mut() = rhs;
                        Ok(Some(Value::Unit))
                    }
                    _ => Err(TenthError::RuntimeError {
                        message: "只能通过可变引用赋值".into(),
                    }),
                }
            }

            HirExprKind::AssignOp { target, op, value } => {
                let current = self.resolve_var(target).ok_or_else(|| {
                    TenthError::RuntimeError {
                        message: format!("undefined variable '{}'", target),
                    }
                })?;
                let rhs = self.eval_expr(value)?.ok_or_else(|| TenthError::RuntimeError {
                    message: "复合赋值值为空值".into(),
                })?;
                let result = self.eval_binary(op, &current, &rhs)?;
                self.set_var(target.clone(), result);
                Ok(Some(Value::Unit))
            }

            HirExprKind::DerefAssignOp { target, op, value } => {
                let target_val = self.eval_expr(target)?.ok_or_else(|| TenthError::RuntimeError {
                    message: "解引用复合赋值目标为空值".into(),
                })?;
                let rhs = self.eval_expr(value)?.ok_or_else(|| TenthError::RuntimeError {
                    message: "解引用复合赋值值为空值".into(),
                })?;
                match &target_val {
                    Value::MutRef(weak) => {
                        let rc = weak.upgrade().ok_or_else(|| TenthError::RuntimeError {
                            message: "无法通过悬垂的 &mut 引用赋值".into(),
                        })?;
                        let current = rc.borrow().clone();
                        let result = self.eval_binary(op, &current, &rhs)?;
                        *rc.borrow_mut() = result;
                        Ok(Some(Value::Unit))
                    }
                    _ => Err(TenthError::RuntimeError {
                        message: "只能通过可变引用赋值".into(),
                    }),
                }
            }

            HirExprKind::Closure { params, body, captures } => {
                // Capture the values of free variables from the current scope
                let captured_values: Vec<(String, Value)> = captures.iter()
                    .filter_map(|name| {
                        self.resolve_var(name).map(|v| (name.clone(), v))
                    })
                    .collect();
                Ok(Some(Value::Closure {
                    params: params.clone(),
                    body: Rc::new((**body).clone()),
                    captures: captured_values,
                }))
            }

            HirExprKind::Range { start, end, inclusive } => {
                let s = match start {
                    Some(e) => self.eval_expr(e)?.and_then(|v| v.as_int()).unwrap_or(0),
                    None => 0,
                };
                let e = match end {
                    Some(e) => self.eval_expr(e)?.and_then(|v| v.as_int()).unwrap_or(0),
                    None => 0,
                };
                Ok(Some(Value::Range { start: s, end: e, inclusive: *inclusive }))
            }

            HirExprKind::Ref(inner) => {
                let val = self.eval_expr(inner)?.ok_or_else(|| TenthError::RuntimeError {
                    message: "引用操作数为空值".into(),
                })?;
                Ok(Some(Value::Ref(Rc::new(RefCell::new(val)))))
            }

            HirExprKind::MutRef(inner) => {
                let val = self.eval_expr(inner)?.ok_or_else(|| TenthError::RuntimeError {
                    message: "可变引用操作数为空值".into(),
                })?;
                if let HirExprKind::Var(var_name) = &inner.kind {
                    // Reuse existing Shared wrapper, or create new one in current scope
                    let rc = if let Some(Value::Shared(existing)) = self.resolve_var(var_name) {
                        existing
                    } else {
                        let cell = Rc::new(RefCell::new(val));
                        self.insert_var(var_name.clone(), Value::Shared(cell.clone()));
                        cell
                    };
                    // Weak reference: does NOT contribute to Rc strong count.
                    // This prevents cycles when structs hold &mut references.
                    Ok(Some(Value::MutRef(Rc::downgrade(&rc))))
                } else {
                    // Non-variable &mut expr: fall back to immutable Ref
                    // (Weak would dangle immediately without an owner)
                    Ok(Some(Value::Ref(Rc::new(RefCell::new(val)))))
                }
            }

            HirExprKind::Deref(inner) => {
                let val = self.eval_expr(inner)?.ok_or_else(|| TenthError::RuntimeError {
                    message: "解引用操作数为空值".into(),
                })?;
                match &val {
                    Value::Ref(rc) => Ok(Some(rc.borrow().clone())),
                    Value::MutRef(weak) => {
                        let rc = weak.upgrade().ok_or_else(|| TenthError::RuntimeError {
                            message: "无法解引用悬垂的 &mut 引用".into(),
                        })?;
                        Ok(Some(rc.borrow().clone()))
                    }
                    _ => Err(TenthError::RuntimeError {
                        message: "无法解引用非引用值".into(),
                    }),
                }
            }

            HirExprKind::FieldAssign { target, field, value } => {
                let target_val = self.eval_expr(target)?.ok_or_else(|| TenthError::RuntimeError {
                    message: "字段赋值目标为空值".into(),
                })?;
                let rhs = self.eval_expr(value)?.ok_or_else(|| TenthError::RuntimeError {
                    message: "字段赋值值为空值".into(),
                })?;
                let target_var_name = if let HirExprKind::Var(vn) = &target.kind {
                    Some(vn.clone())
                } else {
                    None
                };
                match &target_val {
                    // Direct struct — mutate fields and update scope
                    Value::Struct { name: _, fields } => {
                        let mut found = false;
                        for (fname, fval) in fields.borrow_mut().iter_mut() {
                            if fname == field {
                                *fval = rhs.clone();
                                found = true;
                                break;
                            }
                        }
                        if !found {
                            return Err(TenthError::RuntimeError {
                                message: format!("结构体没有字段 '{}'", field),
                            });
                        }
                        // Update scope so subsequent reads see the change
                        if let Some(vn) = target_var_name {
                            // Re-read the struct from fields to get updated value
                            let updated = Value::Struct {
                                name: String::new(),
                                fields: fields.clone(),
                            };
                            self.insert_var(vn, updated);
                        }
                        return Ok(Some(Value::Unit));
                    }
                    Value::MutRef(weak) => {
                        let rc = weak.upgrade().ok_or_else(|| TenthError::RuntimeError {
                            message: "无法通过悬垂的 &mut 引用赋值字段".into(),
                        })?;
                        let mut inner = rc.borrow_mut();
                        match &mut *inner {
                            Value::Struct { fields, .. } => {
                                for (fname, fval) in fields.borrow_mut().iter_mut() {
                                    if fname == field {
                                        *fval = rhs;
                                        return Ok(Some(Value::Unit));
                                    }
                                }
                                Err(TenthError::RuntimeError {
                                    message: format!("结构体没有字段 '{}'", field),
                                })
                            }
                            _ => Err(TenthError::RuntimeError {
                                message: "字段赋值仅支持结构体".into(),
                            }),
                        }
                    }
                    // Shared reference from Vec indexing: mutate through the RefCell
                    Value::Shared(rc) => {
                        let mut inner = rc.borrow_mut();
                        match &mut *inner {
                            Value::Struct { fields, .. } => {
                                for (fname, fval) in fields.borrow_mut().iter_mut() {
                                    if fname == field {
                                        *fval = rhs;
                                        return Ok(Some(Value::Unit));
                                    }
                                }
                                Err(TenthError::RuntimeError {
                                    message: format!("结构体没有字段 '{}'", field),
                                })
                            }
                            _ => Err(TenthError::RuntimeError {
                                message: "字段赋值仅支持结构体".into(),
                            }),
                        }
                    }
                    _ => Err(TenthError::RuntimeError {
                        message: "只能通过可变引用赋值字段".into(),
                    }),
                }
            }

            HirExprKind::Move(inner) => {
                let val = self.eval_expr(inner)?.ok_or_else(|| TenthError::RuntimeError {
                    message: "move 操作数为空值".into(),
                })?;
                if let HirExprKind::Var(var_name) = &inner.kind {
                    self.insert_var(var_name.clone(), Value::Moved);
                }
                Ok(Some(val))
            }

            HirExprKind::TryBlock(inner) => {
                // `try { block }` — catch TryPropagate and wrap as Result::Err
                match self.eval_expr(inner) {
                    Ok(val) => {
                        // Success: wrap in Result::Ok
                        let inner_val = val.unwrap_or(Value::Unit);
                        Ok(Some(Value::Enum {
                            enum_name: "Result".to_string(),
                            variant: "Ok".to_string(),
                            fields: Rc::new(RefCell::new(vec![("_0".to_string(), inner_val)])),
                        }))
                    }
                    Err(TenthError::TryPropagate(err_val)) => {
                        // Caught a ? propagation: wrap in Result::Err
                        Ok(Some(Value::Enum {
                            enum_name: "Result".to_string(),
                            variant: "Err".to_string(),
                            fields: Rc::new(RefCell::new(vec![("_0".to_string(), err_val)])),
                        }))
                    }
                    Err(other) => Err(other),
                }
            }

            HirExprKind::Await(_) | HirExprKind::Spawn(_) => {
                return Err(TenthError::RuntimeError {
                    message: "async/await/spawn 不支持解释器路径，请使用 VM".into(),
                });
            }
            HirExprKind::InterpolatedString { parts } => {
                let mut result = String::new();
                for p in parts {
                    match p {
                        crate::hir::hir::InterpPart::Literal(s) => {
                            result.push_str(s);
                        }
                        crate::hir::hir::InterpPart::Expr(name) => {
                            if let Some(val) = self.resolve_var(name) {
                                result.push_str(&self.value_to_string(&val));
                            } else {
                                result.push_str(name);
                            }
                        }
                    }
                }
                Ok(Some(Value::String(result)))
            }

            HirExprKind::Tuple(elems) => {
                if elems.is_empty() {
                    return Ok(Some(Value::Unit));
                }
                let mut values = Vec::new();
                for e in elems {
                    if let Some(v) = self.eval_expr(e)? {
                        values.push(v);
                    } else {
                        values.push(Value::Unit);
                    }
                }
                Ok(Some(Value::Tuple(values)))
            }

            HirExprKind::StructLiteral { name, fields, has_default: _ } => {
                let mut field_vals = Vec::new();
                for (fname, fexpr) in fields {
                    let v = self.eval_expr(fexpr)?.ok_or_else(|| TenthError::RuntimeError {
                        message: format!("结构体字段 '{}' 为空值", fname),
                    })?;
                    field_vals.push((fname.clone(), v));
                }
                // Note: when has_default is true, the lowerer has already filled in
                // default values for any missing fields in the HIR. The interpreter
                // just evaluates the complete field list as-is.
                Ok(Some(Value::Struct { name: name.clone(), fields: Rc::new(RefCell::new(field_vals)) }))
            }

            HirExprKind::EnumLiteral { enum_name, variant, fields } => {
                let mut field_vals = Vec::new();
                for (fname, fexpr) in fields {
                    let v = self.eval_expr(fexpr)?.ok_or_else(|| TenthError::RuntimeError {
                        message: format!("枚举字段 '{}' 为空值", fname),
                    })?;
                    field_vals.push((fname.clone(), v));
                }
                Ok(Some(Value::Enum {
                    enum_name: enum_name.clone(),
                    variant: variant.clone(),
                    fields: Rc::new(RefCell::new(field_vals)),
                }))
            }

            HirExprKind::Match { scrutinee, arms } => {
                let val = self.eval_expr(scrutinee)?.ok_or_else(|| TenthError::RuntimeError {
                    message: "match 表达式为空值".into(),
                })?;

                for arm in arms {
                    if self.pattern_matches(&arm.pattern, &val) {
                        // Bind variables from the matched pattern (needed for guard evaluation)
                        self.bind_pattern(&arm.pattern, &val);
                        // Check guard if present
                        if let Some(guard) = &arm.guard {
                            let guard_val = self.eval_expr(guard)?;
                            let guard_bool = match guard_val {
                                Some(Value::Bool(true)) => true,
                                Some(Value::Bool(false)) => false,
                                _ => false,
                            };
                            if !guard_bool {
                                // Clean up bindings and try next arm
                                self.unbind_pattern(&arm.pattern);
                                continue;
                            }
                        }
                        let result = self.eval_expr(&arm.body);
                        // Clean up bound variables
                        self.unbind_pattern(&arm.pattern);
                        return result;
                    }
                }
                Ok(Some(Value::Unit))
            }
        }
    }

    /// Convert a ReturnValue error back into a normal Ok result.
    pub(super) fn unwrap_return(result: TenthResult<Option<Value>>) -> TenthResult<Option<Value>> {
        match result {
            Err(TenthError::ReturnValue(v)) => Ok(Some(v)),
            Err(TenthError::TryPropagate(err_val)) => {
                // `?` on Result::Err already carries the whole Result::Err(e) value
                // (see binary.rs UnaryOp::Try). Return it directly to match VM semantics;
                // wrapping again would produce Result::Err(Result::Err(e)) (double-wrap).
                match &err_val {
                    Value::Enum { enum_name, variant, .. }
                        if enum_name == "Result" && variant == "Err" =>
                    {
                        Ok(Some(err_val))
                    }
                    _ => Ok(Some(Value::Enum {
                        enum_name: "Result".to_string(),
                        variant: "Err".to_string(),
                        fields: Rc::new(RefCell::new(vec![("_0".to_string(), err_val)])),
                    })),
                }
            }
            other => other,
        }
    }

    // ── autodiff recording helpers ───────────────────────────────────

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

    pub(super) fn record_unary(&mut self, op: TapeOp, input: &Rc<RefCell<Tensor>>, result: &Rc<RefCell<Tensor>>) {
        if let Some(ref mut tape) = self.tape {
            let node_id = match input.borrow().tape_id {
                Some(input_id) => tape.unary(op, input_id, input.clone(), result.clone()),
                None => {
                    // Create dummy input so the DAG stays connected
                    let dummy = tape.input(input.clone());
                    tape.unary(op, dummy, input.clone(), result.clone())
                }
            };
            result.borrow_mut().tape_id = Some(node_id);
        }
    }

    pub(super) fn eval_call(
        &mut self, func: &Value, args: &[Value], span: &crate::lexer::token::Span,
    ) -> TenthResult<Option<Value>> {
        match func {
            Value::FnRef { name, .. } => {
                self.call_named_fn(name, args, span)
            }
            Value::Enum { enum_name, variant, fields: _ } => {
                // Enum variant used as constructor: Result::Ok(42) → Enum with fields
                let field_vals: Vec<(String, Value)> = args.iter().enumerate()
                    .map(|(i, v)| (format!("_{}", i), v.clone()))
                    .collect();
                Ok(Some(Value::Enum {
                    enum_name: enum_name.clone(),
                    variant: variant.clone(),
                    fields: Rc::new(RefCell::new(field_vals)),
                }))
            }
            Value::Closure { params, body, captures } => {
                self.push_scope();

                for ((pname, _), arg) in params.iter().zip(args.iter()) {
                    self.insert_var(pname.clone(), arg.clone());
                }

                for (cap_name, cap_val) in captures {
                    self.insert_var(cap_name.clone(), cap_val.clone());
                }

                let result = self.eval_expr(body);

                self.pop_scope();

                result
            }
            _ => Err(TenthError::RuntimeError {
                message: "不是可调用值".into(),
            }),
        }
    }

    pub(super) fn eval_stmt(&mut self, stmt: &HirStmt) -> TenthResult<()> {
        self.tick()?;
        match &stmt.kind {
            HirStmtKind::Expr(e) => {
                self.eval_expr(e)?;
                Ok(())
            }
            HirStmtKind::Let { names, init, .. } => {
                let val = match init {
                    Some(e) => self.eval_expr(e)?.unwrap_or(Value::Unit),
                    None => Value::Unit,
                };
                if names.len() == 1 {
                    self.insert_var(names[0].clone(), val);
                } else {
                    // Tuple destructuring: val should be a Vec / Array / Tuple
                    match &val {
                        Value::Vec(v) => {
                            let borrowed = v.borrow();
                            for (i, name) in names.iter().enumerate() {
                                let v = borrowed.get(i).cloned().unwrap_or(Value::Unit);
                                self.insert_var(name.clone(), v);
                            }
                        }
                        Value::Array(a) => {
                            let borrowed = a.borrow();
                            for (i, name) in names.iter().enumerate() {
                                let v = borrowed.get(i).cloned().unwrap_or(Value::Unit);
                                self.insert_var(name.clone(), v);
                            }
                        }
                        Value::Tuple(items) => {
                            for (i, name) in names.iter().enumerate() {
                                let v = items.get(i).cloned().unwrap_or(Value::Unit);
                                self.insert_var(name.clone(), v);
                            }
                        }
                        _ => {
                            // Fallback: assign entire value to first name
                            self.insert_var(names[0].clone(), val);
                        }
                    }
                }
                Ok(())
            }
            HirStmtKind::Return(expr) => {
                let val = match expr {
                    Some(e) => self.eval_expr(e)?.unwrap_or(Value::Unit),
                    None => Value::Unit,
                };
                Err(TenthError::ReturnValue(val))
            }
            HirStmtKind::Break => Err(TenthError::BreakSignal),
            HirStmtKind::Continue => Err(TenthError::ContinueSignal),
            HirStmtKind::Loop { body } => {
                loop {
                    let mut should_break = false;
                    for s in body {
                        match self.eval_stmt(s) {
                            Err(TenthError::BreakSignal) => { should_break = true; break; }
                            Err(TenthError::ContinueSignal) => continue,
                            other => { other?; }
                        }
                    }
                    if should_break { break; }
                }
                Ok(())
            }
            HirStmtKind::While { cond, body } => {
                loop {
                    let c = self.eval_expr(cond)?.ok_or_else(|| TenthError::RuntimeError {
                        message: "while 条件为空值".into(),
                    })?;
                    if !c.is_truthy() {
                        break;
                    }
                    match self.eval_stmt(body) {
                        Err(TenthError::BreakSignal) => break,
                        Err(TenthError::ContinueSignal) => continue,
                        other => { other?; }
                    }
                }
                Ok(())
            }
            HirStmtKind::For { var, iter, body } => {
                let iter_val = self.eval_expr(iter)?.ok_or_else(|| TenthError::RuntimeError {
                    message: "for 迭代对象为空值".into(),
                })?;
                match iter_val {
                    Value::Range { start, end, inclusive } => {
                        let e = if inclusive { end + 1 } else { end };
                        for i in start..e {
                            self.insert_var(var.clone(), Value::Int(i));
                            match self.eval_stmt(body) {
                                Err(TenthError::BreakSignal) => break,
                                Err(TenthError::ContinueSignal) => continue,
                                other => { other?; }
                            }
                        }
                    }
                    Value::Vec(items) => {
                        let vec = items.borrow();
                        for item in vec.iter() {
                            let val = match item {
                                Value::Shared(rc) => rc.borrow().clone(),
                                other => other.clone(),
                            };
                            self.insert_var(var.clone(), val);
                            match self.eval_stmt(body) {
                                Err(TenthError::BreakSignal) => break,
                                Err(TenthError::ContinueSignal) => continue,
                                other => { other?; }
                            }
                        }
                    }
                    Value::Tensor(t) => {
                        let tensor = t.borrow();
                        let shape = tensor.shape();
                        let n = shape.first().copied().unwrap_or(0);
                        let row_size: usize = shape[1..].iter().product();
                        let flat = tensor.data.as_standard_layout().to_owned();
                        let slice = flat.as_slice().unwrap_or(&[]);
                        for i in 0..n {
                            let start = i * row_size;
                            let end = (start + row_size).min(slice.len());
                            let row_data = slice[start..end].to_vec();
                            let row_shape = if shape.len() > 1 {
                                shape[1..].to_vec()
                            } else {
                                vec![1]
                            };
                            let row_tensor = Tensor::from_vec(row_data, row_shape);
                            let val = Value::Tensor(Rc::new(RefCell::new(row_tensor)));
                            self.insert_var(var.clone(), val);
                            match self.eval_stmt(body) {
                                Err(TenthError::BreakSignal) => break,
                                Err(TenthError::ContinueSignal) => continue,
                                other => { other?; }
                            }
                        }
                    }
                    Value::Iterator(lazy_iter) => {
                        // Collect the iterator and iterate over the result
                        let collected = self.eval_iterator_method(&lazy_iter, "collect", &[])?;
                        if let Some(Value::Vec(items)) = collected {
                            let vec = items.borrow();
                            for item in vec.iter() {
                                let val = match item {
                                    Value::Shared(rc) => rc.borrow().clone(),
                                    other => other.clone(),
                                };
                                self.insert_var(var.clone(), val);
                                match self.eval_stmt(body) {
                                    Err(TenthError::BreakSignal) => break,
                                    Err(TenthError::ContinueSignal) => continue,
                                    other => { other?; }
                                }
                            }
                        }
                    }
                    _ => {
                        return Err(TenthError::RuntimeError {
                            message: "for 循环支持 range、Vec、张量和迭代器".into(),
                        });
                    }
                }
                Ok(())
            }
        }
    }
}
