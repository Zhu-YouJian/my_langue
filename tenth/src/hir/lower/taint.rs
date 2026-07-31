//! 层 3 lossy lattice —— 污点旁路分析（方案 C，M2 里程碑）。
//!
//! 核心命题：「可能算错」的值（NaN、溢出、精度降级）不能当确定正确的值用——
//! 除非显式 `lossy`（对应 Rust 的 `unsafe`）。
//!
//! ## 设计（`.vscode/细分规划/阶段2b-输出物-lattice设计.md` §1.2 方案 C）
//!
//! - **表示**：不嵌入 `Type`；本模块在 lowering 完成后对已 lower 的完整 HIR 程序
//!   做**纯结构递归**的旁路分析（`Type`/bytecode/wasm 零侵入）。
//! - **格**：`Exact ≺ PossibleOverflow ≺ PossibleNaN ≺ Lossy`（链式格，join = max）。
//! - **传播**：结果污点 = 左 ⊔ 右 ⊔ 算子静态效应；`lossy expr` 处显式归零（返回 Exact）。
//! - **跨函数（函子组合性的落地点）**：函数返回污点从 body 推导（memo 化递归，
//!   参照 `collect_return_tensor_dims` 模式）；调用点结果 = 被调函数返回污点 ⊔ 实参污点。
//! - **使用点检查**：只对**静态确定的 Lossy**（隐式标量→张量 dtype 收缩）在使用点
//!   （打印/序列化/写盘 sink）报错；`PossibleOverflow`/`PossibleNaN` 只传播不做使用点报错
//!   （防误报：科学计算全是除法，不做 speculative）。
//!
//! ## 静态可判定来源（防误报底线：只报编译期可判定者）
//!
//! 1. **Lossy**：`标量 F32/F64 ×/±/÷ Tensor[F16/BF16/F32]`——标量被静默 cast 到
//!    张量 dtype（唯一现实存在的语言级静默降级路径，审计见设计文档 §4）。
//!    类型静态已知、误报风险为零。泛型/未知类型一律不报。
//! 2. **PossibleOverflow**：浮点字面量组合溢出（`1e308 + 1e308` → inf，当前静默）。
//!    整数组合溢出由既有 lexer 字面量范围检查 + 运行时 `check_int_overflow` 兜底，
//!    不重复实现（设计文档 §3）。
//! 3. **PossibleNaN**：字面量零除数已在 lowering 报硬错误（M1 spike），本分析不再出现。

use std::collections::{HashMap, HashSet};
use crate::error::TenthError;
use crate::hir::hir::*;
use crate::hir::types::{BaseType, Type};

/// 算错可能性格（lossy lattice）——链式格（全序）：
/// `Exact ≺ PossibleOverflow ≺ PossibleNaN ≺ Lossy`。
/// join = max（取最损者）；传播单调且幂等。
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Lossiness {
    Exact,
    PossibleOverflow,
    /// 静态可判定的 PossibleNaN 来源（字面量零除数）已在 lowering 报硬错误
    /// （M1 spike），因此本分析不会构造该层——保留为格的一层，供机制完整性
    /// 与未来工具链（如告警）使用。
    #[allow(dead_code)]
    PossibleNaN,
    Lossy,
}

impl Lossiness {
    fn join(self, other: Lossiness) -> Lossiness {
        if self >= other { self } else { other }
    }
    fn is_lossy(self) -> bool {
        self == Lossiness::Lossy
    }
}

/// 变量污点表（方案 C 旁路分析）。携带**定义作用域深度**用于分支合并时
/// 区分「分支内 let 绑定」（不外泄，防误报）与「对外部变量的赋值」（需合并）。
#[derive(Default, Clone)]
struct VarTaint {
    taints: HashMap<String, Lossiness>,
    scopes: HashMap<String, usize>,
}

impl VarTaint {
    fn get(&self, name: &str) -> Lossiness {
        self.taints.get(name).copied().unwrap_or(Lossiness::Exact)
    }
    fn let_bind(&mut self, name: &str, t: Lossiness, depth: usize) {
        self.taints.insert(name.to_string(), t);
        self.scopes.insert(name.to_string(), depth);
    }
    /// 赋值保持原定义作用域深度（赋值不改变绑定所在作用域）。
    fn assign(&mut self, name: &str, t: Lossiness) {
        self.taints.insert(name.to_string(), t);
    }
}

/// 使用点合并：仅合并「定义在分支之外」的变量（外部 let / 赋值）；
/// 分支内 `let` 绑定的变量是块作用域，不外泄（否则会造成误报）。
fn merge_vt(vt: &mut VarTaint, branch: &VarTaint, branch_depth: usize) {
    for (k, v) in &branch.taints {
        if let Some(&sd) = branch.scopes.get(k) {
            if sd < branch_depth {
                let old = vt.taints.get(k).copied().unwrap_or(Lossiness::Exact);
                vt.taints.insert(k.clone(), old.join(*v));
            }
        }
    }
}

/// 需要 Exact 值的使用点（sink）：打印 / 序列化 / 写盘——把可能算错的值当确定值输出。
fn is_exact_sink(name: &str) -> bool {
    matches!(
        name,
        "println" | "eprintln" | "eprint" | "to_string" | "format"
            | "write_file" | "write_bytes" | "save_weights"
    )
}

fn sink_error(expr: &HirExpr, sink: &str) -> TenthError {
    TenthError::TypeError {
        line: expr.span.line,
        col: expr.span.col,
        message: format!(
            "检测到可能算错的值（lossy 污点，来源：标量被静默转换为更低精度的张量 dtype）被用于需要精确值的上下文：作为 {} 的输出。若确认此值可以近似正确，请用 lossy(...) 显式接受（污点归零）。",
            sink
        ),
    }
}

/// 算子静态效应：`结果污点 = 左 ⊔ 右 ⊔ op_effect`。
fn op_effect(op: &BinOp, left: &HirExpr, right: &HirExpr) -> Lossiness {
    let mut e = Lossiness::Exact;
    // Lossy：隐式标量 → 张量 dtype 收缩（唯一现实的语言级静默降级路径）
    if matches!(op, BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div)
        && scalar_tensor_contraction(&left.ty, &right.ty)
    {
        e = e.join(Lossiness::Lossy);
    }
    // PossibleOverflow：浮点字面量组合溢出（如 1e308 + 1e308 → inf，当前静默）
    if float_literal_comb_overflow(op, left, right) {
        e = e.join(Lossiness::PossibleOverflow);
    }
    e
}

/// 标量 F32/F64 与 Tensor[F16/BF16/F32] 参与算术时，标量被静默 cast 到张量 dtype
/// （精度降级）→ Lossy。类型静态已知才判定（泛型/Unknown 不报，防误报）。
fn scalar_tensor_contraction(l: &Type, r: &Type) -> bool {
    fn tensor_scalar_lossy(t: &Type, scalar: BaseType) -> bool {
        match t {
            Type::Tensor { dtype, .. } => match (dtype.as_ref(), scalar) {
                (Type::Base(BaseType::F16), BaseType::F32 | BaseType::F64) => true,
                (Type::Base(BaseType::BF16), BaseType::F32 | BaseType::F64) => true,
                (Type::Base(BaseType::F32), BaseType::F64) => true,
                _ => false,
            },
            _ => false,
        }
    }
    fn scalar_float(t: &Type) -> Option<BaseType> {
        match t {
            Type::Base(b @ (BaseType::F32 | BaseType::F64)) => Some(*b),
            _ => None,
        }
    }
    match (l, r) {
        (Type::Tensor { .. }, r) => {
            if let Some(s) = scalar_float(r) { tensor_scalar_lossy(l, s) } else { false }
        }
        (l, Type::Tensor { .. }) => {
            if let Some(s) = scalar_float(l) { tensor_scalar_lossy(r, s) } else { false }
        }
        _ => false,
    }
}

/// 浮点字面量组合溢出：两侧均为字面量且结果溢出到 ±inf（如 `1e308 + 1e308`，
/// 当前运行时静默产生 inf）。整数组合溢出由既有 lexer 范围检查 + 运行时
/// `check_int_overflow` 兜底，不在此重复实现（设计文档 §3）。
fn float_literal_comb_overflow(op: &BinOp, left: &HirExpr, right: &HirExpr) -> bool {
    let (HirExprKind::Literal(Literal::Float(a, _)), HirExprKind::Literal(Literal::Float(b, _))) =
        (&left.kind, &right.kind)
    else {
        return false;
    };
    let (a, b) = (*a, *b);
    let r = match op {
        BinOp::Add => a + b,
        BinOp::Sub => a - b,
        BinOp::Mul => a * b,
        BinOp::Div => {
            // 字面量零除数已在 lowering 报硬错误（M1 spike），这里不会到达
            if b == 0.0 { return true; }
            a / b
        }
        _ => return false,
    };
    r.is_infinite()
}

/// 从 match 模式收集变量绑定名（用于把 scrutinee 污点绑定到模式变量）。
fn pattern_bind_names(p: &HirPattern, out: &mut Vec<String>) {
    match p {
        HirPattern::EnumVariant { field_bind, tuple_binds, .. } => {
            if let Some((_, b)) = field_bind { out.push(b.clone()); }
            for (_, b) in tuple_binds { out.push(b.clone()); }
        }
        HirPattern::Binding(name) => out.push(name.clone()),
        HirPattern::Tuple(ps) => {
            for p in ps { pattern_bind_names(p, out); }
        }
        HirPattern::Struct { fields, .. } => {
            for (_, b) in fields { out.push(b.clone()); }
        }
        _ => {}
    }
}

/// 污点旁路分析器。对完整程序（函数 + 方法 + main 表达式）做结构递归污点传播。
pub struct TaintAnalyzer<'a> {
    fns: HashMap<&'a str, &'a HirFnDef>,
    /// 跨函数调用结果缓存：键 = (函数名, 实参污点向量)。
    /// 实参敏感——调用点结果 = 以实参污点绑定形参后分析的被调函数返回污点。
    memo_call: HashMap<(String, Vec<Lossiness>), Lossiness>,
    /// 调用分析递归保护（环 → Exact）。
    visiting_call: HashSet<(String, Vec<Lossiness>)>,
    errors: Vec<TenthError>,
}

/// 入口：分析整个程序，返回所有使用点错误（调用方取第一个）。
/// `generic_instantiations`：泛型实例化 mangled 函数名，其 body 不参与分析
/// （按「类型不确定（泛型）时不报」的防误报原则，见 Lowerer 字段注释）。
pub fn analyze_program(
    functions: &[HirFnDef],
    generic_funcs: &HashMap<String, HirFnDef>,
    methods: &HashMap<String, HashMap<String, HirFnDef>>,
    generic_instantiations: &HashSet<String>,
    main_expr: &Option<HirExpr>,
) -> Vec<TenthError> {
    let mut fns: HashMap<&str, &HirFnDef> = HashMap::new();
    for f in functions {
        if !generic_instantiations.contains(&f.name) {
            fns.insert(f.name.as_str(), f);
        }
    }
    for f in generic_funcs.values() { fns.insert(f.name.as_str(), f); }
    for impls in methods.values() {
        for def in impls.values() { fns.insert(def.name.as_str(), def); }
    }
    let mut a = TaintAnalyzer {
        fns,
        memo_call: HashMap::new(),
        visiting_call: HashSet::new(),
        errors: Vec::new(),
    };
    let names: Vec<String> = a.fns.keys().map(|s| s.to_string()).collect();
    for name in names {
        a.ensure_analyzed(&name);
    }
    if let Some(me) = main_expr {
        let mut vt = VarTaint::default();
        let mut ret = Lossiness::Exact;
        a.expr_taint(me, &mut vt, &mut ret, 0);
    }
    a.errors
}

impl<'a> TaintAnalyzer<'a> {
    /// 顶层分析函数（实参全部 Exact）：返回该函数的返回污点，并报告其内部使用点错误。
    fn ensure_analyzed(&mut self, name: &str) -> Lossiness {
        let count = self.fns.get(name).map(|d| d.params.len()).unwrap_or(0);
        self.fn_call_taint(name, &vec![Lossiness::Exact; count])
    }

    /// 调用点分析（实参敏感）：以实参污点绑定形参后分析被调函数 body，
    /// 返回污点 = 所有 return 路径 ⊔ 隐式末表达式。这是函子组合性的落地点——
    /// 调用点结果 = 被调函数返回污点 ⊔ 实参污点，跨函数自动涌现。
    fn fn_call_taint(&mut self, name: &str, arg_taints: &[Lossiness]) -> Lossiness {
        let key = (name.to_string(), arg_taints.to_vec());
        if let Some(&t) = self.memo_call.get(&key) { return t; }
        if !self.visiting_call.insert(key.clone()) {
            // 递归/互递归：防无限循环，保守返回 Exact（诚实记录为局限）。
            return Lossiness::Exact;
        }
        let t = match self.fns.get(name) {
            Some(def) => {
                let mut vt = VarTaint::default();
                // 实参污点绑定到形参（参数初始为 Exact 的情况即退化为普通调用）
                for (i, (pname, _)) in def.params.iter().enumerate() {
                    let p = arg_taints.get(i).copied().unwrap_or(Lossiness::Exact);
                    vt.let_bind(pname, p, 0);
                }
                let mut ret = Lossiness::Exact;
                let val = self.expr_taint(&def.body, &mut vt, &mut ret, 0);
                ret.join(val)
            }
            None => Lossiness::Exact,
        };
        self.visiting_call.remove(&key);
        self.memo_call.insert(key, t);
        t
    }

    /// 表达式污点（结构递归）。`vt` 为当前作用域变量污点表，`ret` 累计
    /// `return` 语句路径的污点（函数级），`depth` 为作用域深度。
    fn expr_taint(
        &mut self,
        e: &HirExpr,
        vt: &mut VarTaint,
        ret: &mut Lossiness,
        depth: usize,
    ) -> Lossiness {
        match &e.kind {
            HirExprKind::Literal(_) => Lossiness::Exact,
            HirExprKind::Var(name) => vt.get(name),
            HirExprKind::Lossy(inner) => {
                // 显式接受：inner 仍被分析（嵌套错误/副作用保留），但污点归零返回 Exact。
                self.expr_taint(inner, vt, ret, depth);
                Lossiness::Exact
            }
            HirExprKind::Binary { op, left, right, .. } => {
                let tl = self.expr_taint(left, vt, ret, depth);
                let tr = self.expr_taint(right, vt, ret, depth);
                tl.join(tr).join(op_effect(op, left, right))
            }
            HirExprKind::Unary { expr: inner, .. } => self.expr_taint(inner, vt, ret, depth),
            HirExprKind::Call { func, args, .. } | HirExprKind::GenericCall { func, args, .. } => {
                self.call_taint(func, args, vt, ret, depth)
            }
            HirExprKind::MethodCall { receiver, method, args, .. } => {
                let tr = self.expr_taint(receiver, vt, ret, depth);
                // `to_string` 方法 = 使用点 sink（把可能算错的值当确定值序列化）
                if method == "to_string" && tr.is_lossy() {
                    self.errors.push(sink_error(receiver, method));
                    return Lossiness::Exact;
                }
                let mut acc = tr;
                for a in args { acc = acc.join(self.expr_taint(a, vt, ret, depth)); }
                acc
            }
            HirExprKind::Index { target, .. } => self.expr_taint(target, vt, ret, depth),
            HirExprKind::Field { target, .. } => self.expr_taint(target, vt, ret, depth),
            HirExprKind::TensorLiteral { data, .. } => {
                let mut acc = Lossiness::Exact;
                for row in data {
                    for el in row { acc = acc.join(self.expr_taint(el, vt, ret, depth)); }
                }
                acc
            }
            HirExprKind::ArrayLiteral { elements, .. } => {
                let mut acc = Lossiness::Exact;
                for el in elements { acc = acc.join(self.expr_taint(el, vt, ret, depth)); }
                acc
            }
            HirExprKind::Range { start, end, .. } => {
                if let Some(s) = start { self.expr_taint(s, vt, ret, depth); }
                if let Some(en) = end { self.expr_taint(en, vt, ret, depth); }
                Lossiness::Exact
            }
            HirExprKind::If { cond, then_branch, else_branch, .. } => {
                self.expr_taint(cond, vt, ret, depth);
                let mut vt_then = vt.clone();
                let tt = self.expr_taint(then_branch, &mut vt_then, ret, depth + 1);
                let mut vt_else = vt.clone();
                let te = match else_branch {
                    Some(eb) => self.expr_taint(eb, &mut vt_else, ret, depth + 1),
                    None => Lossiness::Exact,
                };
                merge_vt(vt, &vt_then, depth + 1);
                merge_vt(vt, &vt_else, depth + 1);
                tt.join(te)
            }
            HirExprKind::Block { stmts, final_expr } => {
                for s in stmts { self.stmt_taint(s, vt, ret, depth + 1); }
                match final_expr {
                    Some(fe) => self.expr_taint(fe, vt, ret, depth + 1),
                    None => Lossiness::Exact,
                }
            }
            // 闭包体是独立函数作用域，创建时即 Exact；闭包调用（间接）只传播实参。
            HirExprKind::Closure { .. } => Lossiness::Exact,
            HirExprKind::Assign { target, value } => {
                let t = self.expr_taint(value, vt, ret, depth);
                vt.assign(target, t);
                Lossiness::Exact
            }
            HirExprKind::AssignOp { target, op, value } => {
                let tv = vt.get(target);
                let t = self.expr_taint(value, vt, ret, depth);
                // op_effect 需两侧操作数表达式；赋值左侧是变量，保守只取右值污点
                // （右值若为标量×张量收缩，Lossy 已体现在 t 中）。
                let _ = op;
                vt.assign(target, tv.join(t));
                Lossiness::Exact
            }
            HirExprKind::StructLiteral { fields, .. } | HirExprKind::EnumLiteral { fields, .. } => {
                let mut acc = Lossiness::Exact;
                for (_, f) in fields { acc = acc.join(self.expr_taint(f, vt, ret, depth)); }
                acc
            }
            HirExprKind::Match { scrutinee, arms, .. } => {
                let st = self.expr_taint(scrutinee, vt, ret, depth);
                let mut acc = Lossiness::Exact;
                for arm in arms {
                    let mut vt_arm = vt.clone();
                    let mut binds = Vec::new();
                    pattern_bind_names(&arm.pattern, &mut binds);
                    for b in binds { vt_arm.let_bind(&b, st, depth + 1); }
                    if let Some(g) = &arm.guard {
                        self.expr_taint(g, &mut vt_arm, ret, depth + 1);
                    }
                    let tb = self.expr_taint(&arm.body, &mut vt_arm, ret, depth + 1);
                    acc = acc.join(tb);
                    merge_vt(vt, &vt_arm, depth + 1);
                }
                acc
            }
            HirExprKind::Ref(inner) | HirExprKind::MutRef(inner) | HirExprKind::Deref(inner)
            | HirExprKind::Move(inner) | HirExprKind::TryBlock(inner)
            | HirExprKind::Await(inner) => self.expr_taint(inner, vt, ret, depth),
            HirExprKind::Spawn(inner) => self.expr_taint(inner, vt, ret, depth),
            HirExprKind::DerefAssign { target, value } => {
                self.expr_taint(target, vt, ret, depth);
                self.expr_taint(value, vt, ret, depth);
                Lossiness::Exact
            }
            HirExprKind::DerefAssignOp { target, value, .. } => {
                self.expr_taint(target, vt, ret, depth);
                self.expr_taint(value, vt, ret, depth);
                Lossiness::Exact
            }
            HirExprKind::Yield(inner) => {
                if let Some(i) = inner { self.expr_taint(i, vt, ret, depth); }
                Lossiness::Exact
            }
            HirExprKind::InterpolatedString { .. } => Lossiness::Exact,
            HirExprKind::Tuple(elems) => {
                let mut acc = Lossiness::Exact;
                for el in elems { acc = acc.join(self.expr_taint(el, vt, ret, depth)); }
                acc
            }
            HirExprKind::FieldAssign { target, value, .. } => {
                self.expr_taint(target, vt, ret, depth);
                let t = self.expr_taint(value, vt, ret, depth);
                let _ = t;
                Lossiness::Exact
            }
        }
    }

    /// 调用点污点：sink（打印/序列化/写盘）→ 使用点检查；用户函数 → 实参敏感的
    /// 跨函数分析（返回污点 ⊔ 实参污点）；内置/间接调用 → 只传播实参。
    fn call_taint(
        &mut self,
        func: &HirExpr,
        args: &[HirExpr],
        vt: &mut VarTaint,
        ret: &mut Lossiness,
        depth: usize,
    ) -> Lossiness {
        if let HirExprKind::Var(name) = &func.kind {
            if is_exact_sink(name) {
                // 使用点：Lossy 值被当确定值输出 → 报错，要求 lossy(...)
                for a in args {
                    let t = self.expr_taint(a, vt, ret, depth);
                    if t.is_lossy() {
                        self.errors.push(sink_error(a, name));
                    }
                }
                return Lossiness::Exact;
            }
            if self.fns.contains_key(name.as_str()) {
                // 跨函数污点（函子组合性）：先求实参污点，再以实参绑定形参分析被调函数
                let arg_taints: Vec<Lossiness> =
                    args.iter().map(|a| self.expr_taint(a, vt, ret, depth)).collect();
                return self.fn_call_taint(name, &arg_taints);
            }
            // 内置函数：实参污点传播（构造函数字面量参数自然 Exact）
            let mut acc = Lossiness::Exact;
            for a in args { acc = acc.join(self.expr_taint(a, vt, ret, depth)); }
            return acc;
        }
        // 间接调用（闭包变量等）：静态不可解析 callee → 只传播实参
        self.expr_taint(func, vt, ret, depth);
        let mut acc = Lossiness::Exact;
        for a in args { acc = acc.join(self.expr_taint(a, vt, ret, depth)); }
        acc
    }

    fn stmt_taint(
        &mut self,
        s: &HirStmt,
        vt: &mut VarTaint,
        ret: &mut Lossiness,
        depth: usize,
    ) {
        match &s.kind {
            HirStmtKind::Let { names, init, .. } => {
                if let Some(init) = init {
                    let t = self.expr_taint(init, vt, ret, depth);
                    for n in names { vt.let_bind(n, t, depth); }
                }
            }
            HirStmtKind::Expr(e) => {
                self.expr_taint(e, vt, ret, depth);
            }
            HirStmtKind::Return(Some(e)) => {
                let t = self.expr_taint(e, vt, ret, depth);
                *ret = ret.join(t);
            }
            HirStmtKind::Return(None) => {}
            HirStmtKind::While { cond, body } => {
                self.expr_taint(cond, vt, ret, depth);
                let mut vt_body = vt.clone();
                self.stmt_taint(body, &mut vt_body, ret, depth + 1);
                merge_vt(vt, &vt_body, depth + 1);
            }
            HirStmtKind::DoWhile { body, cond } => {
                let mut vt_body = vt.clone();
                self.stmt_taint(body, &mut vt_body, ret, depth + 1);
                merge_vt(vt, &vt_body, depth + 1);
                self.expr_taint(cond, vt, ret, depth);
            }
            HirStmtKind::For { var, iter, body } => {
                self.expr_taint(iter, vt, ret, depth);
                let mut vt_body = vt.clone();
                vt_body.let_bind(var, Lossiness::Exact, depth + 1);
                self.stmt_taint(body, &mut vt_body, ret, depth + 1);
                merge_vt(vt, &vt_body, depth + 1);
            }
            HirStmtKind::Break(_) | HirStmtKind::Continue => {}
            HirStmtKind::Loop { body } => {
                let mut vt_body = vt.clone();
                for s in body { self.stmt_taint(s, &mut vt_body, ret, depth + 1); }
                merge_vt(vt, &vt_body, depth + 1);
            }
        }
    }
}
