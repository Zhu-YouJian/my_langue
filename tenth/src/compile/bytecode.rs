//! HIR → Bytecode compiler.
//!
//! Walks HIR and emits bytecode for the stack VM.

use std::collections::HashMap;
use std::rc::Rc;
use crate::error::{TenthError, TenthResult};
use crate::hir::hir::*;
use crate::hir::types::Type;
use super::super::runtime::vm::{Chunk, Op};

pub struct BytecodeCompiler {
    chunk: Chunk,
    /// Local variable name → slot index
    locals: Vec<String>,
    /// Pending backpatches: (patch_offset, label_id)
    patches: Vec<(usize, usize)>,
    /// Labels: (label_id, code_offset)
    labels: HashMap<usize, usize>,
    next_label: usize,
    /// Loop context stack: (label, loop_start_code_offset, break_label, continue_label)
    /// M2.3：`label` 为该循环的 `'outer:` 标签（None 表示无标签），
    /// `break 'outer` / `continue 'outer` 在栈中从顶向下搜索匹配标签取跳转目标。
    loop_stack: Vec<(Option<String>, usize, usize, usize)>,
    /// 编译期作用域栈：记录每层作用域入口处的 `locals.len()`。
    /// `pop_scope` 时 `locals.truncate(start)` —— 纯编译期槽位表操作（不发射字节码），
    /// 使作用域内绑定的变量（如 match 臂绑定）在作用域结束后不再被 rposition 找到。
    /// 修复 AUDIT #18 族：match 臂绑定 x 遮蔽外层 x 后，臂结束应恢复外层绑定；
    /// 此前扁平 locals 无作用域跟踪，臂绑定残留为"最近绑定"污染外层同名变量。
    scope_stack: Vec<usize>,
    /// Additional chunks compiled from closures (name, chunk)
    closure_chunks: Vec<(String, Chunk)>,
    /// Counter for generating unique closure names
    closure_counter: usize,
    /// Stack of try-block catch labels (for intercepting `?` inside try blocks)
    try_block_stack: Vec<usize>,
    /// Number of params in current function (for TCO arg-count check)
    current_fn_args: usize,
    /// True if the current expression is in tail-call position
    tail_call_ok: bool,
    /// M3.5：是否编译顶层（main/globals）chunk。
    /// 顶层 `let` 写全局表（成为程序全局）；函数/闭包内 `let` 若名字是程序
    /// 全局（global_names 命中）则为 shadow，不写全局（避免覆盖），
    /// 非全局名保持旧行为（StoreGlobal 供 VM 按名调用闭包）。
    is_main: bool,
    /// M3.5：程序级全局名集合（顶层 let / use 导入）。
    global_names: std::collections::HashSet<String>,
    /// a1 P1：当前编译函数名（闭包 chunk 命名前缀）。
    /// 每个函数用独立 BytecodeCompiler 编译，闭包名 `__closure_{N}` 会在多个函数
    /// 间冲突（后注册者覆盖先注册者 → FnRef 解析到错误 chunk → 静默错值）。
    /// 闭包名改为 `__closure_{fn_name}_{N}` 保证跨函数唯一。
    fn_name: String,
}

impl BytecodeCompiler {
    pub fn new() -> Self {
        BytecodeCompiler {
            chunk: Chunk::new(),
            locals: Vec::new(),
            patches: Vec::new(),
            labels: HashMap::new(),
            next_label: 0,
            loop_stack: Vec::new(),
            scope_stack: Vec::new(),
            closure_chunks: Vec::new(),
            closure_counter: 0,
            try_block_stack: Vec::new(),
            current_fn_args: 0,
            tail_call_ok: false,
            is_main: false,
            global_names: std::collections::HashSet::new(),
            fn_name: String::new(),
        }
    }

    /// M3.5：构造带程序全局名集合的编译器（顶层 let / use 导入的全局名）。
    pub fn new_with_globals(global_names: std::collections::HashSet<String>) -> Self {
        let mut c = BytecodeCompiler::new();
        c.global_names = global_names;
        c
    }

    pub fn compile(mut self, func: &HirFnDef) -> TenthResult<(Chunk, Vec<(String, Chunk)>)> {
        self.is_main = false;
        // a1 P1：闭包命名前缀（跨函数唯一，防闭包 chunk 名冲突）
        self.fn_name = func.name.clone();
        self.chunk.num_args = func.params.len();
        self.current_fn_args = func.params.len();
        for (name, _) in &func.params {
            self.locals.push(name.clone());
        }
        self.tail_call_ok = true;
        self.compile_expr(&func.body)?;
        self.tail_call_ok = false;
        // Only push Unit for void functions whose body doesn't already yield a value.
        // 若函数无返回类型注解（默认 Unit）但 body 推断为非 Unit 类型
        // （如 `fn main() { f"..." }`），body 已经把值压栈，不能再压 Unit 覆盖。
        if matches!(func.return_type, crate::hir::types::Type::Base(crate::hir::types::BaseType::Unit))
            && matches!(&func.body.ty, crate::hir::types::Type::Base(crate::hir::types::BaseType::Unit) | crate::hir::types::Type::Never)
        {
            self.chunk.emit(Op::PushUnit);
        }
        self.chunk.emit(Op::Ret);
        self.resolve_patches();
        self.chunk.num_locals = self.locals.len();
        Ok((self.chunk, self.closure_chunks))
    }

    pub fn compile_main(mut self, expr: &HirExpr) -> TenthResult<(Chunk, Vec<(String, Chunk)>)> {
        self.is_main = true;
        self.fn_name = "main".to_string();
        self.compile_expr(expr)?;
        self.chunk.emit(Op::Ret);
        self.resolve_patches();
        self.chunk.num_locals = self.locals.len();
        Ok((self.chunk, self.closure_chunks))
    }

    /// M3.5：把程序级顶层 `let` 全局编译为一段初始化 chunk。
    /// 每个全局按声明顺序求值 init 后 StoreGlobal（复用 HirStmtKind::Let 编译，
    /// 与普通 let 一致：同时写 local 槽 + global 表）。main 之前执行。
    pub fn compile_globals(mut self, globals: &[HirGlobal]) -> TenthResult<(Chunk, Vec<(String, Chunk)>)> {
        use crate::hir::hir::HirStmtKind;
        use crate::lexer::token::Span;
        self.is_main = true;
        self.fn_name = "__globals".to_string();
        // 全局名集合（供 StoreGlobal 判定；全局本身必然命中）
        for g in globals {
            self.global_names.insert(g.name.clone());
        }
        let mut stmts = Vec::new();
        for g in globals {
            // 仅编译带 init 的全局（init == None 的全局由 main_expr 原位初始化）
            if g.name.is_empty() || g.init.is_none() {
                continue;
            }
            stmts.push(HirStmt {
                kind: HirStmtKind::Let {
                    names: vec![g.name.clone()],
                    type_ann: Some(g.ty.clone()),
                    mutable: g.mutable,
                    init: g.init.clone(),
                },
                span: Span { line: 0, col: 0 },
            });
        }
        let body = HirExpr {
            kind: HirExprKind::Block { stmts, final_expr: None },
            ty: Type::unit(),
            span: Span { line: 0, col: 0 },
        };
        self.compile_expr(&body)?;
        self.chunk.emit(Op::Ret);
        self.resolve_patches();
        self.chunk.num_locals = self.locals.len();
        Ok((self.chunk, self.closure_chunks))
    }

    fn compile_expr(&mut self, expr: &HirExpr) -> TenthResult<()> {
        use HirExprKind::*;
        // B 批（VM 报错行号）：在表达式边界记录源码行号到 chunk 行号表，
        // 供 VM 运行时错误按指令偏移定位行号。去重由 Chunk::note_line 处理。
        self.chunk.note_line(expr.span.line);
        match &expr.kind {
            Literal(lit) => match lit {
                crate::hir::hir::Literal::Int(n, _) => self.chunk.emit(Op::PushInt(*n)),
                crate::hir::hir::Literal::Float(f, dt) => match dt {
                    crate::hir::types::BaseType::F32 => self.chunk.emit(Op::PushFloat32(*f as f32)),
                    _ => self.chunk.emit(Op::PushFloat(*f)),
                },
                crate::hir::hir::Literal::Bool(b) => self.chunk.emit(Op::PushBool(*b)),
                crate::hir::hir::Literal::Char(c) => self.chunk.emit(Op::PushChar(*c as u32)),
                crate::hir::hir::Literal::String(s) => {
                    let i = self.chunk.add_string(s);
                    self.chunk.emit(Op::PushStr(i));
                }
            },

            Var(name) => {
                // Check locals first. 用 rposition（最近绑定）：同名 let/循环变量
                // 重绑定会追加新槽位，position（首个匹配）会读旧槽位导致错误值
                // （如 `let x = n; let x = x + 10;` 读回首个 x）。
                if let Some(pos) = self.locals.iter().rposition(|n| n == name) {
                    self.chunk.emit(Op::Load(pos));
                } else {
                    let i = self.chunk.add_string(name);
                    self.chunk.emit(Op::LoadGlobal(i));
                }
            }

            Binary { op, left, right, .. } => {
                // Children of a binary op are not in tail position
                let saved_tail = self.tail_call_ok;
                self.tail_call_ok = false;
                self.compile_expr(left)?;
                // 短路逻辑运算符（And/Or）的右操作数**不能**在此急切编译：
                // 其短路分支内部会再次 compile_expr(right)，此前无条件编译右操作数
                // 导致双重求值 + 栈污染 —— `true || false` 错误返回 false、
                // `false && true` 错误返回 true（右操作数覆盖栈顶）。
                if !matches!(op, BinOp::And | BinOp::Or) {
                    self.compile_expr(right)?;
                }
                self.tail_call_ok = saved_tail;
                use crate::hir::hir::BinOp::*;
                self.chunk.emit(match op {
                    Add => Op::Add, Sub => Op::Sub, Mul => Op::Mul, Div => Op::Div,
                    Mod => Op::Mod,
                    Eq => Op::Eq, NotEq => Op::Neq,
                    Lt => Op::Lt, Gt => Op::Gt, LtEq => Op::Lte, GtEq => Op::Gte,
                    And => {
                        // Short-circuit: left && right
                        // If left is false, jump to end with false; else eval right
                        let end_label = self.new_label();
                        self.chunk.emit(Op::Dup);
                        self.chunk.emit(Op::JmpFalse(0)); // patch later
                        self.patch_jump(end_label);
                        self.chunk.emit(Op::Pop);
                        self.compile_expr(right)?;
                        self.label(end_label);
                        return Ok(());
                    }
                    Or => {
                        let end_label = self.new_label();
                        self.chunk.emit(Op::Dup);
                        self.chunk.emit(Op::JmpTrue(0));
                        self.patch_jump(end_label);
                        self.chunk.emit(Op::Pop);
                        self.compile_expr(right)?;
                        self.label(end_label);
                        return Ok(());
                    }
                });
            }

            Unary { op, expr: inner, .. } => {
                let saved_tail = self.tail_call_ok;
                self.tail_call_ok = false;
                self.compile_expr(inner)?;
                self.tail_call_ok = saved_tail;
                use crate::hir::hir::UnaryOp::*;
                match op {
                    Neg => self.chunk.emit(Op::Neg),
                    Not => self.chunk.emit(Op::Not),
                    Try => {
                        if let Some(&catch_label) = self.try_block_stack.last() {
                            // Inside a try block: check for Result::Err, jump to catch
                            // Stack: [..., val]
                            let ok_label = self.new_label();
                            self.chunk.emit(Op::Dup);
                            let err_i = self.chunk.add_string("Err");
                            self.chunk.emit(Op::IsEnumVariant(err_i));
                            // If not Err → jump to ok unwrap path
                            self.chunk.emit(Op::JmpFalse(0));
                            self.patch_jump(ok_label);
                            // It IS Err: jump to try block catch handler (value stays on stack)
                            self.chunk.emit(Op::Jump(0));
                            self.patch_jump(catch_label);
                            // Ok path: extract inner value from Result::Ok
                            self.label(ok_label);
                            let field_i = self.chunk.add_string("_0");
                            self.chunk.emit(Op::EnumGetField(field_i));
                        } else {
                            self.chunk.emit(Op::Try);
                        }
                    }
                }
            }

            Call { func, args, .. } => {
                // Push args left-to-right; VM pops in reverse into locals
                // Arguments are not in tail position — only the call itself might be
                let saved_tail = self.tail_call_ok;
                self.tail_call_ok = false;
                for a in args.iter() {
                    self.compile_expr(a)?;
                }
                self.tail_call_ok = saved_tail;
                match &func.kind {
                    Var(name) => {
                        let i = self.chunk.add_string(name);
                        // a1 P1：Var 命中 local 槽（函数参数/局部变量里装的闭包值）→
                        // 压值 + CallClosure/TailCallClosure（CallN 只查 functions/globals，
                        // 无法按名解析 local 闭包）。未命中 local → 保持按名直调（顶层函数/全局闭包）。
                        if let Some(pos) = self.locals.iter().rposition(|n| n == name) {
                            self.chunk.emit(Op::Load(pos));
                            if self.tail_call_ok && args.len() == self.current_fn_args {
                                self.chunk.emit(Op::TailCallClosure(args.len()));
                            } else {
                                self.chunk.emit(Op::CallClosure(args.len()));
                            }
                        } else if self.tail_call_ok && args.len() == self.current_fn_args {
                            // Tail call optimization: reuse current frame when in tail position
                            // and arg count matches the current function's param count.
                            self.chunk.emit(Op::TailCall(i, args.len()));
                        } else {
                            self.chunk.emit(Op::CallN(i, args.len()));
                        }
                    }
                    _ => {
                        // a1 P1：非 Var 表达式（间接调用，如 `(cond ? f : g)(x)`、`get_fn()(x)`）→
                        // 编译 func 压入函数值，再发 CallClosure/TailCallClosure。
                        // 此前是「无调用占位」（值悬栈、后续栈错乱），此处从占位改为真调用。
                        self.compile_expr(func)?;
                        if self.tail_call_ok && args.len() == self.current_fn_args {
                            self.chunk.emit(Op::TailCallClosure(args.len()));
                        } else {
                            self.chunk.emit(Op::CallClosure(args.len()));
                        }
                    }
                }
            }

            If { cond, then_branch, else_branch, .. } => {
                // Condition is not in tail position
                let saved_tail = self.tail_call_ok;
                self.tail_call_ok = false;
                self.compile_expr(cond)?;
                let else_label = self.new_label();
                let end_label = self.new_label();
                self.chunk.emit(Op::JmpFalse(0));
                self.patch_jump(else_label);
                // Then branch inherits tail position
                self.tail_call_ok = saved_tail;
                self.compile_expr(then_branch)?;
                self.chunk.emit(Op::Jump(0));
                self.patch_jump(end_label);
                self.label(else_label);
                // Else branch inherits tail position
                self.tail_call_ok = saved_tail;
                if let Some(eb) = else_branch {
                    self.compile_expr(eb)?;
                } else {
                    self.chunk.emit(Op::PushUnit);
                }
                self.tail_call_ok = saved_tail;
                self.label(end_label);
            }

            Block { stmts, final_expr } => {
                let saved_tail = self.tail_call_ok;
                // Statements are not in tail position
                self.tail_call_ok = false;
                for s in stmts {
                    self.compile_stmt(s)?;
                }
                // Only final_expr is in tail position
                self.tail_call_ok = saved_tail;
                if let Some(e) = final_expr {
                    self.compile_expr(e)?;
                } else {
                    self.chunk.emit(Op::PushUnit);
                }
                self.tail_call_ok = saved_tail;
            }

            Assign { target, value } => {
                self.compile_expr(value)?;
                self.chunk.emit(Op::Dup);
                // rposition：写最近绑定的槽位（同名重绑定/循环变量场景）
                let is_local = if let Some(pos) = self.locals.iter().rposition(|n| n == target) {
                    self.chunk.emit(Op::Store(pos));
                    true
                } else {
                    // New local
                    let pos = self.locals.len();
                    self.locals.push(target.clone());
                    self.chunk.emit(Op::Store(pos));
                    false
                };
                // M3.5：若目标是「函数内 shadow 全局的局部」则不再写全局表
                // （否则 `fn f() { let x = 1; x = 2; }` 会覆盖全局 x）。
                // 其余情况保持旧行为（写全局供 closure FnRef 按名调用，
                // 以及顶层/真实全局写入）。
                // Also store as global so closure FnRef values can be called by name.
                // Dup again so the assigned value remains on the stack (Assign is an
                // expression and must produce a value — needed for block expressions
                // like `{ x = 5; x }` and `let y = { x = 5; x }`).
                let shadow_of_global = !self.is_main && is_local && self.global_names.contains(target);
                self.chunk.emit(Op::Dup);
                if shadow_of_global {
                    self.chunk.emit(Op::Pop);
                } else {
                    let gi = self.chunk.add_string(target);
                    self.chunk.emit(Op::StoreGlobal(gi));
                }
            }

            AssignOp { target, op, value } => {
                // target = target op value
                // rposition：读/写最近绑定的槽位
                if let Some(pos) = self.locals.iter().rposition(|n| n == target) {
                    self.chunk.emit(Op::Load(pos));
                } else {
                    let i = self.chunk.add_string(target);
                    self.chunk.emit(Op::LoadGlobal(i));
                }
                self.compile_expr(value)?;
                use crate::hir::hir::BinOp::*;
                self.chunk.emit(match op { Add=>Op::Add, Sub=>Op::Sub, Mul=>Op::Mul, Div=>Op::Div, _=>Op::Add });
                self.chunk.emit(Op::Dup);
                if let Some(pos) = self.locals.iter().rposition(|n| n == target) {
                    self.chunk.emit(Op::Store(pos));
                } else {
                    let i = self.chunk.add_string(target);
                    self.chunk.emit(Op::StoreGlobal(i));
                }
            }

            StructLiteral { name, fields, has_default: _ } => {
                let ni = self.chunk.add_string(name);
                for (fname, fexpr) in fields.iter().rev() {
                    self.compile_expr(fexpr)?;
                    let fi = self.chunk.add_string(fname);
                    self.chunk.emit(Op::PushStr(fi));
                }
                self.chunk.emit(Op::NewStruct(ni, fields.len()));
            }

            UnionLiteral { name, active_field, value } => {
                // M1.2：union 构造 — 压 value 后 NewUnion(name, active_field)
                let name_i = self.chunk.add_string(name);
                let field_i = self.chunk.add_string(active_field);
                self.compile_expr(value)?;
                self.chunk.emit(Op::NewUnion(name_i, field_i));
            }

            Field { target, field } => {
                self.compile_expr(target)?;
                let fi = self.chunk.add_string(field);
                self.chunk.emit(Op::LoadField(fi));
            }

            FieldAssign { target, field, value } => {
                // M1.2：Union 字段修改——StoreField 对 Union 会把修改后的值推回栈
                // （tagged union 非共享），若目标是变量则随后 Store 写回变量槽。
                let is_union_var = matches!(target.ty, Type::Union(_))
                    && matches!(target.kind, HirExprKind::Var(_));
                self.compile_expr(target)?;
                self.compile_expr(value)?;
                let fi = self.chunk.add_string(field);
                self.chunk.emit(Op::StoreField(fi));
                if is_union_var {
                    if let HirExprKind::Var(vn) = &target.kind {
                        // rposition：写最近绑定槽位
                        if let Some(pos) = self.locals.iter().rposition(|n| n == vn) {
                            self.chunk.emit(Op::Store(pos));
                        } else {
                            let i = self.chunk.add_string(vn);
                            self.chunk.emit(Op::StoreGlobal(i));
                        }
                    }
                }
            }

            DerefAssign { target, value } => {
                self.compile_expr(target)?;
                self.compile_expr(value)?;
                self.chunk.emit(Op::Store(0));
            }
            DerefAssignOp { target, value, .. } => {
                self.compile_expr(target)?;
                self.compile_expr(value)?;
                self.chunk.emit(Op::Store(0));
            }

            Index { target, indices } => {
                self.compile_expr(target)?;
                for idx in indices {
                    match idx {
                        crate::hir::hir::Index::Single(e) => {
                            self.compile_expr(e)?;
                            self.chunk.emit(Op::IndexGet);
                        }
                        crate::hir::hir::Index::Range { start, end } => {
                            if let Some(s) = start {
                                self.compile_expr(s)?;
                            } else {
                                self.chunk.emit(Op::PushInt(0));
                            }
                            if let Some(e) = end {
                                self.compile_expr(e)?;
                            } else {
                                self.chunk.emit(Op::PushInt(i64::MAX));
                            }
                            self.chunk.emit(Op::SliceStr);
                        }
                        _ => {}
                    }
                }
            }

            MethodCall { receiver, method, args, .. } => {
                self.compile_expr(receiver)?;
                for a in args.iter() {
                    self.compile_expr(a)?;
                }
                let mi = self.chunk.add_string(method);
                self.chunk.emit(Op::MethodCall(mi, args.len()));
            }

            ArrayLiteral { elements, .. } => {
                // 正向压栈：elements[0] 先压（栈底），elements[n-1] 后压（栈顶）。
                // MakeVec 会 pop 所有元素后 reverse 一次，恢复原始顺序。
                // （此前用 iter().rev() 导致双重反转，输出顺序与源码相反。）
                for e in elements.iter() {
                    self.compile_expr(e)?;
                }
                self.chunk.emit(Op::MakeVec(elements.len()));
            }

            Ref(inner) | MutRef(inner) | Deref(inner) => {
                // VM doesn't track ownership; treat as pass-through
                self.compile_expr(inner)?;
            }

            EnumLiteral { enum_name, variant, fields } => {
                let name_i = self.chunk.add_string(enum_name);
                let variant_i = self.chunk.add_string(variant);
                for (fname, fexpr) in fields.iter().rev() {
                    self.compile_expr(fexpr)?;
                    let fi = self.chunk.add_string(fname);
                    self.chunk.emit(Op::PushStr(fi));
                }
                self.chunk.emit(Op::MakeEnum(name_i, variant_i, fields.len()));
            }

            Match { scrutinee, arms } => {
                self.compile_expr(scrutinee)?;
                let end_label = self.new_label();
                for arm in arms {
                    let next_label = self.new_label();
                    let has_guard = arm.guard.is_some();
                    match &arm.pattern {
                        HirPattern::EnumVariant { enum_name: _, variant, field_bind, tuple_binds } => {
                            self.chunk.emit(Op::Dup); // dup for IsEnumVariant check
                            let variant_i = self.chunk.add_string(variant);
                            self.chunk.emit(Op::IsEnumVariant(variant_i));
                            self.chunk.emit(Op::JmpFalse(0));
                            self.patch_jump(next_label);
                            // IsEnumVariant consumed the dup; scrutinee remains
                            // 臂绑定进入子作用域：臂结束后 pop_scope 移除绑定，
                            // 避免遮蔽/污染外层同名变量（AUDIT #18 族）。
                            self.push_scope();
                            // If there's a guard, keep an extra copy so the next arm
                            // can retry if the guard fails.
                            if has_guard {
                                self.chunk.emit(Op::Dup);
                            }
                            if let Some((bind_name, field_name)) = field_bind {
                                let fi = self.chunk.add_string(field_name);
                                self.chunk.emit(Op::EnumGetField(fi)); // pops scrutinee, pushes field
                                let pos = self.locals.len();
                                self.locals.push(bind_name.clone());
                                self.chunk.emit(Op::Store(pos)); // pops field
                            } else if !tuple_binds.is_empty() {
                                // Tuple variant: bind each positional field
                                for (field_name, bind_name) in tuple_binds {
                                    self.chunk.emit(Op::Dup); // dup scrutinee for each field
                                    let fi = self.chunk.add_string(field_name);
                                    self.chunk.emit(Op::EnumGetField(fi));
                                    let pos = self.locals.len();
                                    self.locals.push(bind_name.clone());
                                    self.chunk.emit(Op::Store(pos));
                                }
                                self.chunk.emit(Op::Pop); // drop scrutinee
                            } else {
                                self.chunk.emit(Op::Pop); // drop scrutinee (no field needed)
                            }
                            // Guard
                            if let Some(guard) = &arm.guard {
                                self.compile_expr(guard)?;
                                self.chunk.emit(Op::JmpFalse(0));
                                self.patch_jump(next_label);
                                self.chunk.emit(Op::Pop); // drop the extra scrutinee copy
                            }
                            self.compile_expr(&arm.body)?;
                            self.chunk.emit(Op::Jump(0));
                            self.patch_jump(end_label);
                            // 臂结束：移除本臂绑定（守卫失败路径也在 next_label 前汇聚）
                            self.pop_scope();
                            self.label(next_label);
                        }
                        HirPattern::Wildcard => {
                            // Wildcard matches everything. If there's a guard, evaluate it
                            // before popping the scrutinee (guard may reference outer vars).
                            if let Some(guard) = &arm.guard {
                                self.compile_expr(guard)?;
                                self.chunk.emit(Op::JmpFalse(0));
                                self.patch_jump(next_label);
                            }
                            self.chunk.emit(Op::Pop); // drop scrutinee
                            self.compile_expr(&arm.body)?;
                            self.chunk.emit(Op::Jump(0));
                            self.patch_jump(end_label);
                            self.label(next_label);
                        }
                        HirPattern::Binding(name) => {
                            // `x` or `x if guard => body`: bind scrutinee to x, then
                            // optionally check guard.
                            self.push_scope(); // 臂绑定子作用域（AUDIT #18 族）
                            self.chunk.emit(Op::Dup); // [scrut, scrut]
                            let pos = self.locals.len();
                            self.locals.push(name.clone());
                            self.chunk.emit(Op::Store(pos)); // [scrut], x bound
                            if let Some(guard) = &arm.guard {
                                self.compile_expr(guard)?; // [scrut, bool]
                                self.chunk.emit(Op::JmpFalse(0));
                                self.patch_jump(next_label); // [scrut]
                            }
                            self.chunk.emit(Op::Pop); // drop scrutinee
                            self.compile_expr(&arm.body)?;
                            self.chunk.emit(Op::Jump(0));
                            self.patch_jump(end_label);
                            self.pop_scope(); // 臂结束：移除绑定
                            self.label(next_label);
                        }
                        HirPattern::Literal(lit) => {
                            // `0 => body` or `0 if guard => body`
                            self.chunk.emit(Op::Dup); // [scrut, scrut]
                            match lit {
                                crate::hir::hir::Literal::Int(n, _) => self.chunk.emit(Op::PushInt(*n)),
                                crate::hir::hir::Literal::Float(f, dt) => match dt {
                                    crate::hir::types::BaseType::F32 => self.chunk.emit(Op::PushFloat32(*f as f32)),
                                    _ => self.chunk.emit(Op::PushFloat(*f)),
                                },
                                crate::hir::hir::Literal::Bool(b) => self.chunk.emit(Op::PushBool(*b)),
                                crate::hir::hir::Literal::Char(c) => self.chunk.emit(Op::PushChar(*c as u32)),
                                crate::hir::hir::Literal::String(s) => {
                                    let i = self.chunk.add_string(s);
                                    self.chunk.emit(Op::PushStr(i));
                                }
                            }
                            self.chunk.emit(Op::Eq); // [scrut, bool]
                            self.chunk.emit(Op::JmpFalse(0));
                            self.patch_jump(next_label); // [scrut]
                            if let Some(guard) = &arm.guard {
                                self.compile_expr(guard)?; // [scrut, bool]
                                self.chunk.emit(Op::JmpFalse(0));
                                self.patch_jump(next_label); // [scrut]
                            }
                            self.chunk.emit(Op::Pop); // drop scrutinee
                            self.compile_expr(&arm.body)?;
                            self.chunk.emit(Op::Jump(0));
                            self.patch_jump(end_label);
                            self.label(next_label);
                        }
                        HirPattern::Struct { name, fields } => {
                            // Struct destructuring: check struct name, then bind fields.
                            self.chunk.emit(Op::Dup); // dup for IsStruct check
                            let name_i = self.chunk.add_string(name);
                            self.chunk.emit(Op::IsStruct(name_i));
                            self.chunk.emit(Op::JmpFalse(0));
                            self.patch_jump(next_label);
                            // IsStruct consumed the dup; scrutinee remains.
                            // 臂绑定子作用域（AUDIT #18 族）
                            self.push_scope();
                            // If there's a guard, keep an extra copy for the next arm retry.
                            if has_guard {
                                self.chunk.emit(Op::Dup);
                            }
                            // Bind each field via LoadField.
                            for (field_name, bind_name) in fields {
                                self.chunk.emit(Op::Dup); // dup scrutinee for each field
                                let fi = self.chunk.add_string(field_name);
                                self.chunk.emit(Op::LoadField(fi));
                                let pos = self.locals.len();
                                self.locals.push(bind_name.clone());
                                self.chunk.emit(Op::Store(pos));
                            }
                            self.chunk.emit(Op::Pop); // drop scrutinee
                            // Guard
                            if let Some(guard) = &arm.guard {
                                self.compile_expr(guard)?;
                                self.chunk.emit(Op::JmpFalse(0));
                                self.patch_jump(next_label);
                                self.chunk.emit(Op::Pop); // drop the extra scrutinee copy
                            }
                            self.compile_expr(&arm.body)?;
                            self.chunk.emit(Op::Jump(0));
                            self.patch_jump(end_label);
                            self.pop_scope(); // 臂结束：移除字段绑定
                            self.label(next_label);
                        }
                        HirPattern::Tuple(patterns) => {
                            // Tuple pattern: (a, b, c) — only support Binding/Wildcard
                            // sub-patterns in VM bytecode (most common case).
                            // Nested patterns fall through to next arm.
                            let all_simple = patterns.iter().all(|p| matches!(
                                p,
                                HirPattern::Binding(_) | HirPattern::Wildcard
                            ));
                            if !all_simple {
                                // Nested tuple patterns not supported in VM bytecode;
                                // fall through to next arm (interpreter handles them).
                            } else {
                                self.chunk.emit(Op::Dup); // [scrut, scrut]
                                self.chunk.emit(Op::IsTuple(patterns.len())); // [scrut, bool]
                                self.chunk.emit(Op::JmpFalse(0));
                                self.patch_jump(next_label); // [scrut]
                                self.push_scope(); // 臂绑定子作用域（AUDIT #18 族）
                                // Bind each sub-pattern
                                for (i, sub_pat) in patterns.iter().enumerate() {
                                    if let HirPattern::Binding(name) = sub_pat {
                                        self.chunk.emit(Op::Dup); // [scrut, scrut]
                                        self.chunk.emit(Op::TupleGet(i)); // [scrut, elem]
                                        let pos = self.locals.len();
                                        self.locals.push(name.clone());
                                        self.chunk.emit(Op::Store(pos)); // [scrut]
                                    }
                                    // Wildcard: no binding, skip
                                }
                                self.chunk.emit(Op::Pop); // drop scrutinee
                                // Guard
                                if let Some(guard) = &arm.guard {
                                    self.compile_expr(guard)?;
                                    self.chunk.emit(Op::JmpFalse(0));
                                    self.patch_jump(next_label);
                                }
                                self.compile_expr(&arm.body)?;
                                self.chunk.emit(Op::Jump(0));
                                self.patch_jump(end_label);
                                self.pop_scope(); // 臂结束：移除绑定
                                self.label(next_label);
                            }
                        }
                        _ => {
                            // Other patterns (Range): not yet supported in VM.
                            // Fall through to next arm.
                        }
                    }
                }
                self.chunk.emit(Op::Pop); // drop scrutinee if no arm matched
                self.chunk.emit(Op::PushUnit); // fallback result
                self.label(end_label);
            }

            // Handle generic calls (monomorphized = regular call)
            HirExprKind::GenericCall { func, args, .. } => {
                if let HirExprKind::Var(name) = &func.kind {
                    let i = self.chunk.add_string(name);
                    for a in args {
                        self.compile_expr(a)?;
                    }
                    self.chunk.emit(Op::CallN(i, args.len()));
                } else {
                    return Err(crate::error::TenthError::RuntimeError { line: None, col: None,
                        message: "字节码：间接 GenericCall（回退）".into(),
                    });
                }
            }
            HirExprKind::Move(inner) => {
                // 修复：Move 在 VM 中应求值 inner（把被移动的值压栈），而非仅发射
                // no-op 的 MoveOp。原先 `let y = move x`（非 Copy 类型）只发射
                // MoveOp（不压入 x 的值），y 拿到栈上垃圾值——移动语义的接收端
                // 在纯 VM 路径损坏（CLI 靠 VmCompileFailed 静默回退解释器掩盖）。
                // 编译期所有权标记（scope Moved）仍负责拦截 move 后的使用。
                self.compile_expr(inner)?;
            }
            HirExprKind::Lossy(inner) => {
                // `lossy` 是纯编译期构造：编译为 inner 本身（运行时 no-op，无附加指令）
                self.compile_expr(inner)?;
            }
            HirExprKind::TryBlock(inner) => {
                // `try { body }` — catch ? propagation, wrap as Result::Ok/Err
                // See interpreter eval.rs:552-574 for reference semantics.
                let catch_label = self.new_label();
                let end_label = self.new_label();
                self.try_block_stack.push(catch_label);
                self.compile_expr(inner)?;
                self.try_block_stack.pop();

                // Normal path: wrap value in Result::Ok(val)
                let ok_field_i = self.chunk.add_string("_0");
                let result_i = self.chunk.add_string("Result");
                let ok_i = self.chunk.add_string("Ok");
                self.chunk.emit(Op::PushStr(ok_field_i));
                self.chunk.emit(Op::MakeEnum(result_i, ok_i, 1));
                self.chunk.emit(Op::Jump(0));
                self.patch_jump(end_label);

                // Catch path: ? encountered Err — Result::Err(val) is on stack, pass through
                self.label(catch_label);
                self.label(end_label);
            }
            HirExprKind::Await(inner) => {
                // await expr: lower inner, emit Await bytecode
                self.compile_expr(inner)?;
                self.chunk.emit(Op::Await);
            }
            HirExprKind::Spawn(inner) => {
                // spawn expr: lower inner, emit Spawn bytecode
                self.compile_expr(inner)?;
                self.chunk.emit(Op::Spawn);
            }
            HirExprKind::Yield(inner) => {
                // yield [expr]：让出控制权给 VM 调度器。
                // 与 Op::Yield 语义对齐：Op::Yield 不消费栈顶，仅保存整个调用栈后让出。
                // 因此 inner 的值（若存在）必须先 emit Pop 丢弃，避免恢复后栈污染；
                // 恢复后由 PushUnit 提供 yield 表达式的返回值（Unit）。
                if let Some(e) = inner {
                    self.compile_expr(e)?;
                    self.chunk.emit(Op::Pop);
                }
                self.chunk.emit(Op::Yield);
                self.chunk.emit(Op::PushUnit);
            }
            HirExprKind::InterpolatedString { parts } => {
                // Evaluate by concatenating all parts as strings
                // First, push all parts as strings, then concatenate
                let mut count = 0;
                for p in parts {
                    match p {
                        crate::hir::hir::InterpPart::Literal(s) => {
                            let i = self.chunk.add_string(s);
                            self.chunk.emit(Op::PushStr(i));
                            count += 1;
                        }
                        crate::hir::hir::InterpPart::Expr(name) => {
                            // Look up variable and convert to string
                            // rposition：最近绑定槽位
                            if let Some(pos) = self.locals.iter().rposition(|n| n == name) {
                                self.chunk.emit(Op::Load(pos));
                            } else {
                                let i = self.chunk.add_string(name);
                                self.chunk.emit(Op::LoadGlobal(i));
                            }
                            // Convert to string via format builtin
                            let fi = self.chunk.add_string("format");
                            self.chunk.emit(Op::CallN(fi, 1));
                            count += 1;
                        }
                    }
                }
                // Concatenate all parts: result = parts[0] + parts[1] + ...
                for _ in 1..count {
                    // str_add is host function index 3
                    let fi = self.chunk.add_string("str_add");
                    self.chunk.emit(Op::CallN(fi, 2));
                }
            }
            HirExprKind::Tuple(elems) => {
                if elems.is_empty() {
                    // `()` is Unit, not an empty tuple
                    self.chunk.emit(Op::PushUnit);
                } else {
                    // Push elements left-to-right, then assemble via Op::MakeTuple
                    for e in elems {
                        self.compile_expr(e)?;
                    }
                    self.chunk.emit(Op::MakeTuple(elems.len()));
                }
            }
            HirExprKind::Range { start, end, inclusive } => {
                // Compile range expression. Extract integer literals when possible
                // so `let r = 1..5` produces a correct Range value. Non-literal
                // start/end fall back to 0 (the for-in loop path compiles them
                // directly and does not use PushRange).
                let s = start.as_ref().and_then(|e| match &e.kind {
                    HirExprKind::Literal(lit) => match lit {
                        crate::hir::hir::Literal::Int(n, _) => Some(*n),
                        _ => None,
                    },
                    _ => None,
                }).unwrap_or(0);
                let e = end.as_ref().and_then(|e| match &e.kind {
                    HirExprKind::Literal(lit) => match lit {
                        crate::hir::hir::Literal::Int(n, _) => Some(*n),
                        _ => None,
                    },
                    _ => None,
                }).unwrap_or(0);
                self.chunk.emit(Op::PushRange(s, e, *inclusive));
            }

            // Tensor literal: [[1.0, 2.0], [3.0, 4.0]]
            TensorLiteral { data, ty } => {
                let rows = data.len();
                let cols = if rows > 0 { data[0].len() } else { 0 };
                // 根据类型注解选择 dtype（统一编码：F64=0, F32=1, F16=2, BF16=3）
                let dtype_code: u8 = match ty.tensor_dtype() {
                    Some(crate::hir::types::BaseType::F32) => 1,
                    Some(crate::hir::types::BaseType::F16) => 2,
                    Some(crate::hir::types::BaseType::BF16) => 3,
                    _ => 0,  // F64 或无注解
                };
                // Push all elements in row-major order
                for row in data.iter() {
                    for elem in row.iter() {
                        self.compile_expr(elem)?;
                    }
                }
                self.chunk.emit(Op::MakeTensor(rows, cols, dtype_code));
            }

            // Closure: |params| body
            Closure { params, body, captures } => {
                // Compile the closure body as a separate chunk
                // a1 P1：闭包名带函数名前缀（跨函数唯一）——多个函数各自含闭包时，
                // `__closure_{N}` 会在 functions 表中互相覆盖（FnRef 名解析到错误 chunk → 静默错值）。
                let closure_name = format!("__closure_{}_{}", self.fn_name, self.closure_counter);
                self.closure_counter += 1;

                // a1 P3：捕获正确性——捕获值内联（替代旧的「全局名 hack」）。
                // 捕获分两类：
                //   1) 父局部变量（self.locals 命中）→ 值内联进 FnRef.captures；闭包 chunk
                //      以 params.len()+i 槽位接收（不再 LoadGlobal——旧方案多实例捕获同名变量
                //      会互相覆盖 → 静默错值，如 make_adder(5)/make_adder(10) add5(3) 返回 13）。
                //   2) 程序全局名（self.locals 未命中）→ 闭包体直接 LoadGlobal 解析（不内联）。
                let mut captured_locals: Vec<(String, usize)> = Vec::new();
                for cap_name in captures {
                    // rposition：捕获最近绑定值
                    if let Some(pos) = self.locals.iter().rposition(|n| n == cap_name) {
                        captured_locals.push((cap_name.clone(), pos));
                    }
                }

                let mut closure_compiler = BytecodeCompiler::new();
                closure_compiler.closure_counter = self.closure_counter;
                // M3.5：闭包编译器共享全局名集合（shadow 判定一致）
                closure_compiler.global_names = self.global_names.clone();
                // a1 P1：闭包编译器继承函数名前缀（嵌套闭包命名一致）
                closure_compiler.fn_name = self.fn_name.clone();

                // a1 P3：闭包 chunk 槽位 = params + 捕获局部（捕获在 params.len()+i）。
                // num_args 含捕获（调用时 CallClosure/call_value 把捕获值追加为额外实参）。
                let total_args = params.len() + captured_locals.len();
                closure_compiler.chunk.num_args = total_args;
                closure_compiler.current_fn_args = total_args;
                for (name, _) in params {
                    closure_compiler.locals.push(name.clone());
                }
                for (cap_name, _) in &captured_locals {
                    closure_compiler.locals.push(cap_name.clone());
                }

                // Compile the closure body (in tail position for TCO)
                closure_compiler.tail_call_ok = true;
                closure_compiler.compile_expr(body)?;
                closure_compiler.tail_call_ok = false;
                closure_compiler.chunk.emit(Op::Ret);
                closure_compiler.resolve_patches();
                closure_compiler.chunk.num_locals = closure_compiler.locals.len();

                // a1 P3：嵌套闭包 chunk 注册——闭包体内的嵌套闭包（如 `|n| |x| x+n`）的
                // chunk 收在 closure_compiler.closure_chunks，必须并入父级注册表，
                // 否则内层 FnRef 名（如 __closure_main_1）在 functions 表查不到 → 「未定义的函数」。
                // 同时回传计数器，避免外层后续闭包与内层编号冲突（命名唯一性）。
                for (n, c) in std::mem::take(&mut closure_compiler.closure_chunks) {
                    self.closure_chunks.push((n, c));
                }
                if closure_compiler.closure_counter > self.closure_counter {
                    self.closure_counter = closure_compiler.closure_counter;
                }

                // a1 P3：父栈压捕获值（值内联），MakeClosure 弹出装入 FnRef.captures。
                // 压栈顺序与闭包 chunk 捕获槽（params..params+captures）一致。
                for (_, pos) in &captured_locals {
                    self.chunk.emit(Op::Load(*pos));
                }

                // Register the closure chunk
                self.closure_chunks.push((closure_name.clone(), closure_compiler.chunk));

                // Emit MakeClosure with the closure name index
                let name_i = self.chunk.add_string(&closure_name);
                self.chunk.emit(Op::MakeClosure(params.len(), captured_locals.len(), name_i));
            }

        }
        Ok(())
    }

    fn compile_stmt(&mut self, stmt: &HirStmt) -> TenthResult<()> {
        // B 批（VM 报错行号）：语句边界记录行号——覆盖无 init 的 let / 裸 return /
        // break / continue 等不经过 compile_expr 的语句，保证语句级粒度完整。
        self.chunk.note_line(stmt.span.line);
        match &stmt.kind {
            HirStmtKind::Let { names, init, .. } => {
                if let Some(e) = init {
                    self.compile_expr(e)?;
                } else {
                    self.chunk.emit(Op::PushUnit);
                }
                if names.len() == 1 {
                    // Single binding: store entire value
                    let pos = self.locals.len();
                    self.locals.push(names[0].clone());
                    self.chunk.emit(Op::Dup);
                    self.chunk.emit(Op::Store(pos));
                    // M3.5：写全局表的判定：
                    // - 顶层（main/globals chunk）：总是写全局（顶层 let 成为全局）
                    // - 函数/闭包内：仅当名字是程序全局时为 shadow，不写全局
                    //   （避免函数内 `let PI = 99` 覆盖全局 PI）；
                    //   非全局名保持旧行为（StoreGlobal 供 VM 按名调用闭包）。
                    if self.is_main || !self.global_names.contains(&names[0]) {
                        let gi = self.chunk.add_string(&names[0]);
                        self.chunk.emit(Op::StoreGlobal(gi));
                    } else {
                        self.chunk.emit(Op::Pop);
                    }
                } else {
                    // Tuple destructuring: extract each element via TupleGet.
                    // Stack discipline: init pushes the tuple; each iteration dups
                    // the tuple, extracts element i, then dups the element so
                    // Store (local) 与 StoreGlobal/Pop 各消费一个，不消耗 tuple。
                    // Final Pop drops the tuple.
                    for (i, name) in names.iter().enumerate() {
                        self.chunk.emit(Op::Dup);          // [Tuple, Tuple]
                        self.chunk.emit(Op::TupleGet(i));  // [Tuple, elem]
                        self.chunk.emit(Op::Dup);          // [Tuple, elem, elem]
                        let pos = self.locals.len();
                        self.locals.push(name.clone());
                        self.chunk.emit(Op::Store(pos));   // [Tuple, elem]
                        // M3.5：与单名一致——顶层写全局；函数内 shadow 全局不写
                        if self.is_main || !self.global_names.contains(name) {
                            let gi = self.chunk.add_string(name);
                            self.chunk.emit(Op::StoreGlobal(gi)); // [Tuple]
                        } else {
                            self.chunk.emit(Op::Pop);      // [Tuple]
                        }
                    }
                    self.chunk.emit(Op::Pop); // [] drop tuple
                }
            }
            HirStmtKind::Expr(e) => {
                self.compile_expr(e)?;
                self.chunk.emit(Op::Pop);
            }
            HirStmtKind::Return(expr) => {
                if let Some(e) = expr {
                    self.compile_expr(e)?;
                } else {
                    self.chunk.emit(Op::PushUnit);
                }
                self.chunk.emit(Op::Ret);
            }
            HirStmtKind::DoWhile { label, body, cond } => {
                // do-while: execute body first, then check condition
                let loop_start = self.chunk.code.len();
                let break_label = self.new_label();
                let continue_label = self.new_label();
                self.label(continue_label); // continue jumps to condition check
                self.loop_stack.push((label.clone(), loop_start, break_label, continue_label));
                self.compile_stmt(body)?;
                // Check condition
                self.compile_expr(cond)?;
                // If condition is true, jump back to loop_start; else exit
                self.chunk.emit(Op::JmpFalse(0));
                let exit_label = self.new_label();
                self.patch_jump(exit_label);
                let offset = loop_start as i32 - self.chunk.code.len() as i32 - 5;
                self.chunk.emit(Op::Jump(offset));
                self.label(exit_label);
                self.label(break_label); // break jumps here
                self.loop_stack.pop();
            }
            HirStmtKind::While { label, cond, body } => {
                let loop_start = self.chunk.code.len();
                let break_label = self.new_label();
                let continue_label = self.new_label();
                self.label(continue_label); // continue: re-evaluate cond
                self.loop_stack.push((label.clone(), loop_start, break_label, continue_label));
                self.compile_expr(cond)?;
                self.chunk.emit(Op::JmpFalse(0));
                self.patch_jump(break_label);
                self.compile_stmt(body)?;
                // Jump back to condition (continue_label)
                let offset = loop_start as i32 - self.chunk.code.len() as i32 - 5;
                self.chunk.emit(Op::Jump(offset));
                self.label(break_label);
                self.loop_stack.pop();
            }
            HirStmtKind::Loop { label, body } => {
                let loop_start = self.chunk.code.len();
                let break_label = self.new_label();
                let continue_label = self.new_label();
                self.label(continue_label); // continue jumps here
                self.loop_stack.push((label.clone(), loop_start, break_label, continue_label));
                for s in body {
                    self.compile_stmt(s)?;
                }
                // Unconditional jump back to loop start
                let offset = loop_start as i32 - self.chunk.code.len() as i32 - 5;
                self.chunk.emit(Op::Jump(offset));
                self.label(break_label); // break jumps here
                self.loop_stack.pop();
            }
            HirStmtKind::For { label, var, iter, body } => {
                // Compile for-in loop
                match &iter.kind {
                    HirExprKind::Range { start, end, inclusive } => {
                        // for var in start..end { body }
                        // Compile as: var = start; while var < end { body; var += 1 }
                        let var_slot = self.locals.len();
                        self.locals.push(var.clone());

                        // var = start
                        if let Some(s) = start {
                            self.compile_expr(s)?;
                        } else {
                            self.chunk.emit(Op::PushInt(0));
                        }
                        self.chunk.emit(Op::Store(var_slot));

                        let loop_start = self.chunk.code.len();
                        let break_label = self.new_label();
                        let continue_label = self.new_label();
                        // M2.3 fix：continue_label 不在 loop_start 处绑定——
                        // 对 for 循环，continue 应跳过 body 剩余并执行 var += 1，
                        // 目标在下方 `var += 1` 之前（见 compile_stmt body 之后）。
                        self.loop_stack.push((label.clone(), loop_start, break_label, continue_label));

                        // condition: var < end (or var <= end for inclusive)
                        self.chunk.emit(Op::Load(var_slot));
                        if let Some(e) = end {
                            self.compile_expr(e)?;
                        } else {
                            self.chunk.emit(Op::PushInt(i64::MAX));
                        }
                        if *inclusive {
                            self.chunk.emit(Op::Lte);
                        } else {
                            self.chunk.emit(Op::Lt);
                        }
                        self.chunk.emit(Op::JmpFalse(0));
                        self.patch_jump(break_label);

                        // body
                        self.compile_stmt(body)?;

                        // var += 1 — continue 跳到这里（跳过 body 剩余、执行增量）
                        self.label(continue_label);
                        self.chunk.emit(Op::Load(var_slot));
                        self.chunk.emit(Op::PushInt(1));
                        self.chunk.emit(Op::Add);
                        self.chunk.emit(Op::Store(var_slot));

                        // Jump back to condition
                        let offset = loop_start as i32 - self.chunk.code.len() as i32 - 5;
                        self.chunk.emit(Op::Jump(offset));
                        self.label(break_label);
                        self.loop_stack.pop();
                    }
                    _ => {
                        // Generic iteration: for var in iterable { body }
                        // Compile as index-based iteration:
                        //   let __iter = iterable;
                        //   let __idx = 0;
                        //   while __idx < __iter.len() {
                        //       let var = __iter[__idx];
                        //       body;
                        //       __idx += 1;
                        //   }
                        let iter_slot = self.locals.len();
                        self.locals.push("__iter".to_string());
                        let idx_slot = iter_slot + 1;
                        self.locals.push("__idx".to_string());
                        let var_slot = idx_slot + 1;
                        self.locals.push(var.clone());

                        // __iter = iterable
                        self.compile_expr(iter)?;
                        self.chunk.emit(Op::Store(iter_slot));

                        // __idx = 0
                        self.chunk.emit(Op::PushInt(0));
                        self.chunk.emit(Op::Store(idx_slot));

                        let loop_start = self.chunk.code.len();
                        let break_label = self.new_label();
                        let continue_label = self.new_label();
                        // M2.3 fix：同 Range 分支——continue 跳过 body 剩余并执行 __idx += 1
                        self.loop_stack.push((label.clone(), loop_start, break_label, continue_label));

                        // condition: __idx < __iter.len()
                        self.chunk.emit(Op::Load(idx_slot));
                        self.chunk.emit(Op::Load(iter_slot));
                        let len_i = self.chunk.add_string("len");
                        self.chunk.emit(Op::MethodCall(len_i, 0));
                        self.chunk.emit(Op::Lt);
                        self.chunk.emit(Op::JmpFalse(0));
                        self.patch_jump(break_label);

                        // var = __iter[__idx]
                        self.chunk.emit(Op::Load(iter_slot));
                        self.chunk.emit(Op::Load(idx_slot));
                        self.chunk.emit(Op::IndexGet);
                        self.chunk.emit(Op::Store(var_slot));

                        // body
                        self.compile_stmt(body)?;

                        // __idx += 1 — continue 跳到这里（跳过 body 剩余、执行增量）
                        self.label(continue_label);
                        self.chunk.emit(Op::Load(idx_slot));
                        self.chunk.emit(Op::PushInt(1));
                        self.chunk.emit(Op::Add);
                        self.chunk.emit(Op::Store(idx_slot));

                        // Jump back to condition
                        let offset = loop_start as i32 - self.chunk.code.len() as i32 - 5;
                        self.chunk.emit(Op::Jump(offset));
                        self.label(break_label);
                        self.loop_stack.pop();
                    }
                }
            }
            HirStmtKind::Break { label, value } => {
                // M2.3：带标签时从 loop_stack 顶向下搜索匹配标签的循环；无标签取最近循环。
                let target = if let Some(l) = &label {
                    self.loop_stack.iter().rev().find(|(lb, _, _, _)| lb.as_ref() == Some(l))
                } else {
                    self.loop_stack.last()
                };
                if let Some(&(_, _, break_label, _)) = target {
                    // If break has a value, compile it first (pushes value on stack)
                    if let Some(e) = value {
                        self.compile_expr(e)?;
                    }
                    self.chunk.emit(Op::Jump(0));
                    self.patches.push((self.chunk.code.len() - 4, break_label));
                } else if label.is_some() {
                    // lower 期已校验标签，此处理论不可达（防御性报错）
                    return Err(TenthError::TypeError {
                        line: stmt.span.line,
                        col: stmt.span.col,
                        message: format!("未定义循环标签 '{}'", label.as_ref().unwrap()),
                    });
                }
            }
            HirStmtKind::Continue { label } => {
                // M2.3：带标签时从 loop_stack 顶向下搜索匹配标签的循环；无标签取最近循环。
                let target = if let Some(l) = &label {
                    self.loop_stack.iter().rev().find(|(lb, _, _, _)| lb.as_ref() == Some(l))
                } else {
                    self.loop_stack.last()
                };
                if let Some(&(_, _, _, continue_label)) = target {
                    self.chunk.emit(Op::Jump(0));
                    self.patches.push((self.chunk.code.len() - 4, continue_label));
                } else if label.is_some() {
                    // lower 期已校验标签，此处理论不可达（防御性报错）
                    return Err(TenthError::TypeError {
                        line: stmt.span.line,
                        col: stmt.span.col,
                        message: format!("未定义循环标签 '{}'", label.as_deref().unwrap()),
                    });
                }
            }
        }
        Ok(())
    }

    fn new_label(&mut self) -> usize {
        let id = self.next_label;
        self.next_label += 1;
        id
    }

    /// 进入编译期作用域：记录 locals.len()，此后绑定的变量（match 臂绑定等）
    /// 在 pop_scope 时被从槽位表移除，不再污染外层同名变量。
    fn push_scope(&mut self) {
        self.scope_stack.push(self.locals.len());
    }

    /// 退出编译期作用域：截断 locals 到作用域入口长度。
    /// 纯编译期操作（不发射任何字节码）——运行时栈不受影响，仅影响后续
    /// 代码的 rposition 变量解析。
    fn pop_scope(&mut self) {
        if let Some(start) = self.scope_stack.pop() {
            self.locals.truncate(start);
        }
    }

    fn label(&mut self, id: usize) {
        self.labels.insert(id, self.chunk.code.len());
    }

    fn patch_jump(&mut self, label_id: usize) {
        // Patch the most recent Jump/JmpFalse/JmpTrue with 0 offset
        // The offset to patch is 4 bytes before the end (the i32 operand)
        self.patches.push((self.chunk.code.len() - 4, label_id));
    }

    fn resolve_patches(&mut self) {
        for &(patch_pos, label_id) in &self.patches.clone() {
            if let Some(&label_pos) = self.labels.get(&label_id) {
                let offset = label_pos as i32 - patch_pos as i32 - 4;
                let bytes = offset.to_le_bytes();
                for (i, b) in bytes.iter().enumerate() {
                    if patch_pos + i < self.chunk.code.len() {
                        // code 为 Rc<Vec<u8>>：编译期唯一持有者，make_mut 原地写
                        Rc::make_mut(&mut self.chunk.code)[patch_pos + i] = *b;
                    }
                }
            }
        }
    }
}