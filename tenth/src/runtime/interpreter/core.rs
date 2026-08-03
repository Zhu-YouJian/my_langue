//! 解释器核心：`Interpreter` 结构体、构造函数、作用域管理、执行入口。
//!
//! 从 `mod.rs` 拆出（架构重构 T3e），包含：
//! - `Interpreter` 结构体定义
//! - 构造函数 `new` / `with_limits`
//! - 作用域管理 `push_scope` / `pop_scope` / `insert_var` / `remove_var` /
//!   `extend_globals` / `globals_clone` / `resolve_var` / `set_var`
//! - 执行入口 `execute_program`（含 native FnRef 注入）
//! - 资源/时间预算 `tick`
//! - `make_tensor`（带 limit 守卫的张量构造）
//! - `unwrap_return`（ReturnValue/TryPropagate 转换）
//!
//! 所有方法仍为 `impl Interpreter`，跨文件 impl；`eval_expr` / `eval_call` /
//! `eval_stmt` 见 `eval.rs`，自动微分记录见 `autodiff_helpers.rs`。

use std::collections::HashMap;
use std::rc::Rc;
use std::cell::RefCell;
use crate::error::{TenthError, TenthResult};
use crate::hir::hir::*;
use crate::hir::types::Type;
use crate::runtime::value::Value;
use crate::runtime::tensor::Tensor;
use crate::runtime::limits::RuntimeLimits;
use crate::runtime::arena::Arena;
use crate::runtime::autodiff::{Tape, CustomOpRegistry, CustomBackward};

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
    // 以下字段在跨文件 impl 中被 eval.rs / methods.rs / natives.rs 等子模块访问，
    // 因此标 `pub(super)`（对 interpreter 模块及其子模块可见）。
    pub(super) functions: Vec<HirFnDef>,
    pub(super) generic_funcs: HashMap<String, HirFnDef>,
    /// M3.5：程序级顶层 `let` 全局（常量与可变状态）。在 main 之前于 depth 0 初始化。
    pub(super) globals: Vec<HirGlobal>,
    pub(super) methods: HashMap<String, HashMap<String, HirFnDef>>,
    pub(super) modules: HashMap<String, HirProgram>,
    pub(super) trait_impls: HashMap<String, HashMap<String, HashMap<String, HirFnDef>>>,
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
    /// TCP 监听器句柄表（与 vm.rs 的 Vm::tcp_listeners 对齐）。
    /// 索引+1 即句柄（1-based，0 表示无效）。`None` 表示已关闭的槽位。
    pub tcp_listeners: Vec<Option<std::net::TcpListener>>,
    /// UDP socket 句柄表（与 vm.rs 的 Vm::udp_sockets 对齐，基本功核查第 69 项）。
    /// 索引+1 即句柄（1-based，0 表示无效）。`None` 表示已关闭的槽位。
    pub udp_sockets: Vec<Option<std::net::UdpSocket>>,
    /// 正则表达式句柄表（与 vm.rs 的 Vm::regexes 对齐）。
    /// 索引+1 即句柄（1-based，0 表示无效）。`None` 表示已释放的槽位。
    pub regexes: Vec<Option<regex::Regex>>,
    /// 子进程 Command 句柄表（与 vm.rs 的 Vm::commands 对齐）。
    /// 索引+1 即句柄（1-based，0 表示无效）。`None` 表示已释放的槽位。
    pub commands: Vec<Option<std::process::Command>>,
    /// 自定义算子注册表（PROJ-006）。
    ///
    /// 用 `Rc<RefCell<...>>` 共享——Tape 在 backward 前通过 `set_custom_ops`
    /// 拿到 Rc 副本，使 backward 能访问用户的 `CustomBackward` 实现。
    /// register_custom_op 通过 `borrow_mut()` 修改；查询通过 `borrow()`。
    /// 与 `vm::Vm::custom_ops` 字段对齐（双重注册一致性）。
    pub custom_ops: Rc<RefCell<CustomOpRegistry>>,
    /// M4.4 调试器：调试钩子。每个语句执行前调用（tree-walk 逐步的基础）。
    ///
    /// - 钩子可阻塞等待用户命令（CLI 调试器同步交互），读 `self.vars` 查看变量；
    /// - `None`（默认）时零行为变化、零开销——仅调试器工具设置；
    /// - 钩子出错时返回 `Err` 立即中止执行（错误响亮）。
    pub debug_hook: Option<Box<dyn FnMut(&mut Interpreter, &HirStmt) -> TenthResult<()>>>,
}

impl Interpreter {
    pub fn new(program: &HirProgram) -> Self {
        Interpreter {
            vars: HashMap::new(),
            scope_depth: 0,
            scope_vars: vec![Vec::new()],
            functions: program.functions.clone(),
            generic_funcs: HashMap::new(),
            globals: program.globals.clone(),
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
            tcp_listeners: Vec::new(),
            udp_sockets: Vec::new(),
            regexes: Vec::new(),
            commands: Vec::new(),
            custom_ops: Rc::new(RefCell::new(CustomOpRegistry::new())),
            debug_hook: None,
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

    /// M3.5：初始化程序级顶层 `let` 全局（在 depth 0，main 之前）。
    ///
    /// - 仅初始化带 init 的全局（`init == None` 的全局由 main_expr 原位初始化，
    ///   保持执行顺序，如 autodiff 测试中的交错计算）。
    /// - 若变量已存在于 vars（REPL 经 extend_globals 持久化的值），跳过——
    ///   避免每条 REPL 行重置可变全局状态。
    /// - 按声明顺序求值（可能引用更早声明的全局）。
    pub(super) fn init_program_globals(&mut self) -> TenthResult<()> {
        let globals = self.globals.clone();
        for g in &globals {
            if g.name.is_empty() {
                continue;
            }
            let Some(e) = &g.init else { continue };
            if self.vars.contains_key(&g.name) {
                continue;
            }
            let val = self.eval_expr(e)?.unwrap_or(Value::Unit);
            self.insert_var(g.name.clone(), val);
        }
        Ok(())
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

    /// M4.4 调试器：设置调试钩子（每个语句执行前调用）。
    ///
    /// 钩子闭包可阻塞等待用户命令、读取 `self.vars` 查看变量值、检查
    /// `stmt.span.line` 决定断点/单步。`None` 时解释器行为完全不变。
    pub fn set_debug_hook(
        &mut self,
        hook: Option<Box<dyn FnMut(&mut Interpreter, &HirStmt) -> TenthResult<()>>>,
    ) {
        self.debug_hook = hook;
    }

    /// 注册自定义可微算子（PROJ-006）。
    ///
    /// 返回 `op_id`（用于 `TapeOp::Custom(op_id)`）。
    /// 若同名算子已注册，返回 `Err`。与 `vm::Vm::register_custom_op` 对齐。
    pub fn register_custom_op(&mut self, op: Box<dyn CustomBackward>) -> Result<usize, String> {
        self.custom_ops.borrow_mut().register(op)
    }

    /// 自定义算子注册表访问器（PROJ-006）。
    ///
    /// 返回 `Rc` 副本，供 Tape 在 backward 前通过 `set_custom_ops` 共享。
    pub fn custom_ops(&self) -> Rc<RefCell<CustomOpRegistry>> {
        Rc::clone(&self.custom_ops)
    }

    pub fn execute_program(&mut self, program: &HirProgram) -> TenthResult<Option<Value>> {
        self.insert_var(
            "tensor".to_string(),
            Value::FnRef {
                name: "tensor".to_string(),
                params: vec![("data".to_string(), Type::Unknown)],
                return_type: Type::Unknown,
                captures: vec![],
            },
        );

        // Autodiff builtins
        self.insert_var(
            "start_grad".to_string(),
            Value::FnRef {
                name: "start_grad".to_string(),
                params: vec![],
                return_type: Type::unit(),
                captures: vec![],
            },
        );
        self.insert_var(
            "new_grad".to_string(),
            Value::FnRef {
                name: "new_grad".to_string(),
                params: vec![],
                return_type: Type::unit(),
                captures: vec![],
            },
        );
        self.insert_var(
            "stop_grad".to_string(),
            Value::FnRef {
                name: "stop_grad".to_string(),
                params: vec![],
                return_type: Type::unit(),
                captures: vec![],
            },
        );
        self.insert_var(
            "zero_grad".to_string(),
            Value::FnRef {
                name: "zero_grad".to_string(),
                params: vec![],
                return_type: Type::unit(),
                captures: vec![],
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
                captures: vec![],
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
                captures: vec![],
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
                captures: vec![],
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
                captures: vec![],
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
                    captures: vec![],
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
                    captures: vec![],
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
                    captures: vec![],
                },
            );
        }
        self.insert_var(
            "param".to_string(),
            Value::FnRef {
                name: "param".to_string(),
                params: vec![("t".to_string(), Type::Unknown)],
                return_type: Type::Unknown,
                captures: vec![],
            },
        );
        self.insert_var(
            "backward".to_string(),
            Value::FnRef {
                name: "backward".to_string(),
                params: vec![("loss".to_string(), Type::Unknown)],
                return_type: Type::unit(),
                captures: vec![],
            },
        );
        self.insert_var(
            "grad".to_string(),
            Value::FnRef {
                name: "grad".to_string(),
                params: vec![("param".to_string(), Type::Unknown)],
                return_type: Type::Unknown,
                captures: vec![],
            },
        );
        // 护城河 F：explain_error() — 返回上一次 backward 失败的根因说明列表
        self.insert_var(
            "explain_error".to_string(),
            Value::FnRef {
                name: "explain_error".to_string(),
                params: vec![],
                return_type: Type::Unknown,
                captures: vec![],
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
                    captures: vec![],
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
                                captures: vec![],
                            },
                        );
                    }
                }
            }
        }

        // Reset arena at the start of each top-level evaluation.
        // Any temporary allocations from previous evaluations are freed.
        self.arena.reset();

        // M3.5：初始化程序级顶层 let 全局（depth 0，main 之前）。
        // 同文件与 use 导入的全局均在此求值；test 路径经 execute_program_inner。
        self.init_program_globals()?;

        if let Some(ref expr) = program.main_expr {
            // M3.5：顶层全部是 let（已提升为全局）时，`<expr>` 为空块。
            // 此时若存在 `fn main` 应执行 main（与 VM 路径一致），而非空块。
            let is_empty_block = matches!(
                &expr.kind,
                HirExprKind::Block { stmts, final_expr }
                    if stmts.is_empty() && final_expr.is_none()
            );
            if !is_empty_block {
                return Self::unwrap_return(self.eval_expr(expr));
            }
        }
        if let Some(main_fn) = self.functions.iter().find(|f| f.name == "main") {
            let body = main_fn.body.clone();
            Self::unwrap_return(self.eval_expr(&body))
        } else {
            Ok(None)
        }
    }

    /// Execute a single #[test] function by name.
    /// First runs execute_program to initialize state, then calls the test function.
    pub fn execute_fn_test(&mut self, fn_name: &str) -> TenthResult<Option<Value>> {
        // Run full program initialization (registers all fn refs, builtins, etc.)
        // This also runs main/main_expr if present, but for test files they typically don't have one.
        // For safety, we ignore the program result since we're only interested in the test fn.
        let _ = self.execute_program_inner()?;

        // Find and evaluate the test function
        self.call_named_function(fn_name)
    }

    /// Internal: run initialization without executing main.
    fn execute_program_inner(&mut self) -> TenthResult<()> {
        // Register builtins from execute_program
        self.insert_var("tensor".to_string(), Value::FnRef {
            name: "tensor".to_string(),
            params: vec![("data".to_string(), Type::Unknown)],
            return_type: Type::Unknown,
            captures: vec![],
        });
        for name in &["start_grad", "new_grad", "stop_grad", "zero_grad"] {
            self.insert_var(name.to_string(), Value::FnRef {
                name: name.to_string(), params: vec![], return_type: Type::unit(), captures: vec![],
            });
        }
        self.insert_var("cross_entropy".to_string(), Value::FnRef {
            name: "cross_entropy".to_string(),
            params: vec![("logits".to_string(), Type::Unknown), ("target".to_string(), Type::Unknown)],
            return_type: Type::Unknown,
            captures: vec![],
        });
        self.insert_var("select".to_string(), Value::FnRef {
            name: "select".to_string(),
            params: vec![("cond".to_string(), Type::Unknown), ("then".to_string(), Type::Unknown), ("else".to_string(), Type::Unknown)],
            return_type: Type::Unknown,
            captures: vec![],
        });
        self.insert_var("scatter".to_string(), Value::FnRef {
            name: "scatter".to_string(),
            params: vec![("base".to_string(), Type::Unknown), ("dim".to_string(), Type::Unknown), ("index".to_string(), Type::Unknown), ("src".to_string(), Type::Unknown)],
            return_type: Type::Unknown,
            captures: vec![],
        });
        self.insert_var("gather".to_string(), Value::FnRef {
            name: "gather".to_string(),
            params: vec![("base".to_string(), Type::Unknown), ("dim".to_string(), Type::Unknown), ("index".to_string(), Type::Unknown)],
            return_type: Type::Unknown,
            captures: vec![],
        });
        for name in &["abs", "sqrt", "sin", "cos", "ln", "pow"] {
            self.insert_var(name.to_string(), Value::FnRef {
                name: name.to_string(), params: vec![("x".to_string(), Type::Unknown)], return_type: Type::Unknown, captures: vec![],
            });
        }
        for name in &["zeros", "ones"] {
            self.insert_var(name.to_string(), Value::FnRef {
                name: name.to_string(), params: vec![("dims".to_string(), Type::Unknown)], return_type: Type::Unknown, captures: vec![],
            });
        }
        // Register all program functions as FnRefs
        for func in self.functions.clone() {
            let params = func.params.clone();
            let ret = func.return_type.clone();
            self.insert_var(func.name.clone(), Value::FnRef { name: func.name.clone(), params, return_type: ret, captures: vec![] });
        }
        // Register module functions
        for module in self.modules.clone().values() {
            for func in &module.functions {
                let params = func.params.clone();
                let ret = func.return_type.clone();
                self.insert_var(func.name.clone(), Value::FnRef { name: func.name.clone(), params, return_type: ret, captures: vec![] });
            }
        }
        // M3.5：初始化程序级顶层 let 全局（test 路径也能读取全局常量/状态）
        self.init_program_globals()?;
        // Reset arena
        self.arena.reset();
        Ok(())
    }

    /// Find a function by name and evaluate its body in a new scope.
    fn call_named_function(&mut self, fn_name: &str) -> TenthResult<Option<Value>> {
        let test_fn = self.functions.iter()
            .find(|f| f.name == fn_name)
            .cloned()
            .ok_or_else(|| crate::error::TenthError::RuntimeError {
                line: None, col: None,
                message: format!("函数 '{}' 未找到", fn_name),
            })?;
        self.push_scope();
        for (param_name, _) in &test_fn.params {
            self.insert_var(param_name.clone(), Value::Unit);
        }
        let result = self.eval_expr(&test_fn.body);
        self.pop_scope();
        match result { Ok(val) => Ok(val), Err(e) => Err(e) }
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
    pub(super) fn tick(&mut self) -> TenthResult<()> {
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
                return Err(TenthError::RuntimeError { line: None, col: None, message: msg });
            }
        }
        Ok(Tensor::from_vec(data, shape))
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
                        fields: std::rc::Rc::new(std::cell::RefCell::new(vec![("_0".to_string(), err_val)])),
                    })),
                }
            }
            other => other,
        }
    }
}
