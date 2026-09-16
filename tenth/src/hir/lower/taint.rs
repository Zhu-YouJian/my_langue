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
//! 3. **PossibleNaN**：除数的静态判定已完整（M3 与 shape/常量信息协同）：
//!    静态零除数（字面量零 / 张量字面量全零 / `zeros*` 构造）已在 lowering 报硬错误
//!    （M1 spike），不会到达此处；静态非零除数（非零字面量 / 张量字面量全非零 /
//!    `ones*` 构造）→ 精确豁免不标（M3 正向）；值未知（变量除数 / shape 已知但值未知）
//!    → 不 speculate（防误报）。故默认严格度下 PossibleNaN 无触发源，层保留为
//!    机制完整性与未来 speculative 告警的豁免接口。

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
    /// 除数的静态判定已完整（M3 与 shape/常量信息协同）：静态零除数已在 lowering
    /// 报硬错误（M1 spike），静态非零除数精确豁免（M3），值未知不 speculate
    /// （防误报）——因此本分析不会构造该层，保留为格的一层，供机制完整性
    /// 与未来 speculative 告警（静态豁免接口）使用。
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
    /// G1（AUDIT-11.4.48）：变量的**静态类型**（`let` / 形参绑定处记录，赋值时跟随右值）。
    ///
    /// 复合赋值 `t += x` 的左侧在 HIR 里只是变量名（`HirExprKind::AssignOp.target: String`），
    /// **没有 `HirExpr.ty` 可读**；要判定「f16 张量 += f64 标量 → 逐元素写回低精度 dtype」
    /// 这条与二元算子完全同源的隐式收缩，必须知道左侧变量的类型。
    /// 取不到类型时一律判 Exact（宁可漏报，不可误报）。
    types: HashMap<String, Type>,
}

impl VarTaint {
    fn get(&self, name: &str) -> Lossiness {
        self.taints.get(name).copied().unwrap_or(Lossiness::Exact)
    }
    fn let_bind(&mut self, name: &str, t: Lossiness, depth: usize) {
        self.taints.insert(name.to_string(), t);
        self.scopes.insert(name.to_string(), depth);
    }
    /// 带静态类型的绑定（类型可用于算子效应判定，G1）。
    fn let_bind_typed(&mut self, name: &str, t: Lossiness, depth: usize, ty: &Type) {
        self.let_bind(name, t, depth);
        self.types.insert(name.to_string(), ty.clone());
    }
    /// 变量静态类型（未记录 → None → 不做收缩判定）。
    fn type_of(&self, name: &str) -> Option<&Type> {
        self.types.get(name)
    }
    /// 赋值保持原定义作用域深度（赋值不改变绑定所在作用域）。
    fn assign(&mut self, name: &str, t: Lossiness) {
        self.taints.insert(name.to_string(), t);
    }
    /// 赋值：污点与静态类型都跟随右值。
    /// 与 `lower_expr` 的 `scope.define_var(name, v.ty, true)`（lower_expr.rs:1219）同语义
    /// ——否则 `t = <f64 张量>` 之后 `t += f64标量` 会按旧的低精度类型误判（G1 误报面）。
    fn assign_typed(&mut self, name: &str, t: Lossiness, ty: &Type) {
        self.assign(name, t);
        self.types.insert(name.to_string(), ty.clone());
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
                // G1：外部变量的静态类型也随分支赋值合并（`t` 在分支内被赋成另一种
                // dtype 后，其后的 `t += 标量` 必须按新类型判定；否则按 `let` 的旧类型
                // 判定会 ① 误报（f16 → f64）或 ② 漏报（f64 → f16））。
                if let Some(ty) = branch.types.get(k) {
                    vt.types.insert(k.clone(), ty.clone());
                }
            }
        }
    }
}

/// 需要 Exact 值的使用点（sink）：打印 / 序列化 / 写盘——把可能算错的值当确定值输出。
///
/// G5 补全（AUDIT-11.4.48）：`print`（native 注册见 `runtime/natives.rs:2123`）、
/// `json_encode`（`runtime/natives.rs:1349`）、`json_encode_pretty`
/// （`runtime/natives.rs:1356`）——三者与既有 `println`/`to_string` 同为逃逸点：
/// 前一个是「打印」，后两者把张量**序列化成字符串**（与 `to_string` 同语义）。
fn is_exact_sink(name: &str) -> bool {
    matches!(
        name,
        "println" | "eprintln" | "eprint" | "to_string" | "format" | "print"
            | "json_encode" | "json_encode_pretty"
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

/// 普通字符串插值（`"{x}"`）使用点错误（G4）。
///
/// `"{x}"` 的运行时语义就是 `to_string(x)` 的字符串拼接（VM：`compile/bytecode.rs`
/// 的 `InterpolatedString` 分支为每个 `Expr` 部件 emit 一次 `to_string` 转换；
/// 解释器：`runtime/interpreter/eval.rs` 对 `InterpPart::Expr` 调 `value_to_string`），
/// 故与 `to_string` 同属 sink，报错信息单独定制的原因是：
/// 普通串插值语法只接受 `{identifier}`（`lexer.rs` 对 `is_fstring=false` 分支的
/// 标识符校验），**无法内联写 `{lossy(x)}`**——那会被当作字面文本。
/// 因此这里提示「先 let 绑定再插值」这一可行写法。
fn interp_sink_error(expr: &HirExpr, name: &str) -> TenthError {
    TenthError::TypeError {
        line: expr.span.line,
        col: expr.span.col,
        message: format!(
            "检测到可能算错的值（lossy 污点，来源：标量被静默转换为更低精度的张量 dtype）被用于需要精确值的上下文：字符串插值（变量 {}）。若确认此值可以近似正确，请先 `let y = lossy({});` 显式接受（污点归零）再插值 `{{y}}`（普通字符串插值只接受标识符，无法内联写 lossy(...)）。",
            name, name
        ),
    }
}

/// 算子静态效应的**类型版**：`左类型 × 右类型 → 效应`。
///
/// G1（AUDIT-11.4.48）与 `op_effect` 共用本函数——复合赋值 `t += x` 只有
/// 「左侧变量的静态类型 + 右侧表达式类型」可用，**不能**再手写一份判定
/// （两份手写判定必然漂移，这是本项目的高危模式）。
fn op_effect_ty(op: &BinOp, left: &Type, right: &Type) -> Lossiness {
    // Lossy：隐式标量 → 张量 dtype 收缩（唯一现实的语言级静默降级路径）
    if matches!(op, BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div)
        && scalar_tensor_contraction(left, right)
    {
        Lossiness::Lossy
    } else {
        Lossiness::Exact
    }
}

/// 算子静态效应：`结果污点 = 左 ⊔ 右 ⊔ op_effect`。
fn op_effect(op: &BinOp, left: &HirExpr, right: &HirExpr) -> Lossiness {
    let mut e = op_effect_ty(op, &left.ty, &right.ty);
    // PossibleOverflow：浮点字面量组合溢出（如 1e308 + 1e308 → inf，当前静默）
    if float_literal_comb_overflow(op, left, right) {
        e = e.join(Lossiness::PossibleOverflow);
    }
    // PossibleNaN：除法/取模的除数静态效应（M3：与 shape/常量信息协同）。
    // 除数的静态判定已完整（M1 硬错误 + M3 正向豁免）：
    // - 除数**静态为零**（字面量零 / 张量字面量全零 / `zeros*` 构造）→ 已在
    //   lowering 报硬错误（M1 spike `check_binary_static_divzero`），不会到达此处；
    // - 除数**静态非零**（非零字面量 / 张量字面量全非零 / `ones*` 构造，shape
    //   已知与否不影响）→ 精确：不标 PossibleNaN（M3 正向豁免，防误报）；
    // - 除数值未知（变量除数 / shape 已知但值未知的张量）→ 不 speculate
    //   （保持现状，防误报底线：宁可漏报，不可误报）。
    // 因此 PossibleNaN 在默认严格度下无触发源——保留为格的一层（机制完整性），
    // 并作为未来 speculative 告警的静态豁免接口。
    if matches!(op, BinOp::Div | BinOp::Mod) && super::Lowerer::is_statically_nonzero(right) {
        // 静态非零除数：明确豁免 PossibleNaN（无操作——精确化的落点）。
    }
    e
}

/// G3（AUDIT-11.4.48）：方法实参的隐式「标量 → 张量 dtype」收缩。
///
/// 与二元算子共用 `scalar_tensor_contraction`——两处必须是**同一条判定**
/// （手写第二份必然漂移）。接收者类型静态已知（低精度张量）且实参是更高精度
/// 浮点标量时才判 Lossy；泛型/未知类型一律 Exact（防误报）。
fn method_arg_effect(recv_ty: &Type, arg_ty: &Type) -> Lossiness {
    if scalar_tensor_contraction(recv_ty, arg_ty) {
        Lossiness::Lossy
    } else {
        Lossiness::Exact
    }
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
                for (i, (pname, pty)) in def.params.iter().enumerate() {
                    let p = arg_taints.get(i).copied().unwrap_or(Lossiness::Exact);
                    vt.let_bind_typed(pname, p, 0, pty);
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
                for a in args {
                    let ta = self.expr_taint(a, vt, ret, depth);
                    // G3（AUDIT-11.4.48）：方法实参此前**只传播不检查**——接收者是
                    // 低精度张量而实参是更高精度标量时（如 `t.clamp(0.0, 1.0)`、
                    // `t.masked_fill(mask, 1.234…)`、`t.pow(2.0)`），实参在方法内部被
                    // 静默 cast 到张量 dtype（`scalar_tensor_contraction` 判定的正是
                    // 这条语言级静默降级路径，与二元算子同源），结果张量因此是近似值
                    // 却一路带着 Exact 污点用到 sink。
                    acc = acc.join(ta).join(method_arg_effect(&receiver.ty, &a.ty));
                }
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
            // G2（AUDIT-11.4.48）：闭包体此前**整体不求值**（直接 `Exact`）⇒
            // 闭包体内的降精度使用点（sink）静默漏报。
            //
            // 闭包在 HIR 里是**内联体**（`HirExprKind::Closure { body }`，不生成
            // `HirFnDef`），因此没有别的分析入口会走到它的 body——本分支是唯一机会。
            //
            // 语义（与既有注释一致，不改变闭包值本身的污点）：
            // - 捕获变量继承当前作用域污点（闭包是独立函数体，故用克隆的作用域）；
            // - 形参按 Exact 绑定（调用点未知）；
            // - `return` 只累计到**闭包自己的**返回污点（`ret_body`），不外泄到外层；
            // - 闭包值本身仍返回 `Exact`——函数值不是它将来产出的值，把 body 的
            //   返回污点当闭包值的污点会让 `to_string(closure)` 误报（防误报底线）。
            HirExprKind::Closure { params, body, .. } => {
                let mut vt_body = vt.clone();
                for (pname, pty) in params {
                    vt_body.let_bind_typed(pname, Lossiness::Exact, depth + 1, pty);
                }
                let mut ret_body = Lossiness::Exact;
                self.expr_taint(body, &mut vt_body, &mut ret_body, depth + 1);
                Lossiness::Exact
            }
            HirExprKind::Assign { target, value } => {
                let t = self.expr_taint(value, vt, ret, depth);
                // 类型跟随右值（与 lower_expr.rs:1219 的 `scope.define_var(name, v.ty, true)`
                // 同语义）——否则 `t = <f64张量>` 后 `t += f64标量` 会按旧的低精度类型误判。
                vt.assign_typed(target, t, &value.ty);
                Lossiness::Exact
            }
            HirExprKind::AssignOp { target, op, value } => {
                let tv = vt.get(target);
                let t = self.expr_taint(value, vt, ret, depth);
                // G1（AUDIT-11.4.48）：`t += x` 与 `t = t + x` 同语义。此前只取
                // 「左污点 ⊔ 右值污点」并**丢弃算子效应** ⇒ `f16张量 += f64标量`
                // （标量被静默 cast 到 f16 后逐元素写回）既不报错、变量污点也保持
                // Exact，下游所有 sink 一起漏报（静默丢精度）。
                let effect = match vt.type_of(target) {
                    Some(lty) => op_effect_ty(op, lty, &value.ty),
                    None => Lossiness::Exact,
                };
                vt.assign(target, tv.join(t).join(effect));
                Lossiness::Exact
            }
            HirExprKind::StructLiteral { fields, .. } | HirExprKind::EnumLiteral { fields, .. } => {
                let mut acc = Lossiness::Exact;
                for (_, f) in fields { acc = acc.join(self.expr_taint(f, vt, ret, depth)); }
                acc
            }
            HirExprKind::UnionLiteral { value, .. } => self.expr_taint(value, vt, ret, depth),
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
            HirExprKind::InterpolatedString { parts } => {
                // G4：插值体纳入分析。`"{x}"` 与 `to_string(x)` / f-string 的
                // `format(...)` 同语义——把值转成字符串就是「把可能算错的值当确定值用」
                // 的逃逸点（见 `interp_sink_error` 的语义依据）。此前此处不遍历 parts，
                // 导致 `"{x}"` 静默漏报。
                //
                // 注意：HIR 的 `InterpPart::Expr` 只存**变量名字符串**（词法层普通串
                // 仅接受 `{identifier}`，见 `lexer.rs` 的 is_fstring=false 分支），
                // 故这里按变量名查污点表，而非递归分析子表达式。
                // 含 `.` 的路径（`"{a.b}"`）不是变量表键 → 返回 Exact（不报）：
                // 这是**残留漏报**而非误报——若退化成按根变量取污点，`"{t.shape}"`
                // 这类精确字段访问会被误伤（shape 是精确的整数元组）。
                for p in parts {
                    if let InterpPart::Expr(name) = p {
                        if vt.get(name).is_lossy() {
                            self.errors.push(interp_sink_error(e, name));
                        }
                    }
                }
                // 与 MethodCall 的 `to_string` sink 一致：报错即消耗污点，字符串结果
                // 归 Exact，避免同一逃逸点被外层 sink（`println("{x}")`）重复报错。
                Lossiness::Exact
            }
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
            HirStmtKind::Let { names, type_ann, init, .. } => {
                if let Some(init) = init {
                    let t = self.expr_taint(init, vt, ret, depth);
                    // 记录变量静态类型（G1：复合赋值左侧的类型来源）。
                    // 有显式注解时以注解为准（与 lower_stmt.rs:178-186 的
                    // `scope.define_var(.., ty, ..)` 合并规则一致，避免
                    // `let t: Tensor[f64,..] = ...; t += 标量` 被按旧 dtype 误判）。
                    let ty = type_ann.clone().unwrap_or_else(|| init.ty.clone());
                    for n in names { vt.let_bind_typed(n, t, depth, &ty); }
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
            HirStmtKind::While { cond, body, .. } => {
                self.expr_taint(cond, vt, ret, depth);
                let mut vt_body = vt.clone();
                self.stmt_taint(body, &mut vt_body, ret, depth + 1);
                merge_vt(vt, &vt_body, depth + 1);
            }
            HirStmtKind::DoWhile { body, cond, .. } => {
                let mut vt_body = vt.clone();
                self.stmt_taint(body, &mut vt_body, ret, depth + 1);
                merge_vt(vt, &vt_body, depth + 1);
                self.expr_taint(cond, vt, ret, depth);
            }
            HirStmtKind::For { var, iter, body, .. } => {
                self.expr_taint(iter, vt, ret, depth);
                let mut vt_body = vt.clone();
                vt_body.let_bind(var, Lossiness::Exact, depth + 1);
                self.stmt_taint(body, &mut vt_body, ret, depth + 1);
                merge_vt(vt, &vt_body, depth + 1);
            }
            HirStmtKind::Break { .. } | HirStmtKind::Continue { .. } => {}
            HirStmtKind::Loop { body, .. } => {
                let mut vt_body = vt.clone();
                for s in body { self.stmt_taint(s, &mut vt_body, ret, depth + 1); }
                merge_vt(vt, &vt_body, depth + 1);
            }
        }
    }
}
