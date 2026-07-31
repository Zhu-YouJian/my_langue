use std::collections::HashMap;
use crate::error::{TenthError, TenthResult, TenthWarning};
use crate::hir::types::BaseType;
use crate::lexer::token::Span;
use crate::parser::ast as ast;
use crate::hir::hir::*;
use crate::hir::types::*;
use super::Lowerer;

/// 推断两个 shape 的 broadcast 结果（NumPy 规则，从右往左对齐）。
/// 返回 `Some(dims)` 如果兼容，`None` 如果不兼容。
///
/// 规则：
/// - 任一侧 `Dim::Any`（未知）→ 该维度结果为 `Any`（无法静态确定）
/// - `Known(1)` 与 `Known(n)` → `Known(n)`（广播）
/// - `Known(a)` 与 `Known(b)`（a == b）→ `Known(a)`
/// - `Symbol(s)` 与 `Symbol(s)`（同名）→ `Symbol(s)`
/// - `Symbol(s)` 与 `Known(n)` → `Symbol(s)`（假设兼容，unify 留待 Phase 2）
/// - 其他 → `None`（不兼容）
pub(super) fn broadcast_shapes(l: &[Dim], r: &[Dim]) -> Option<Vec<Dim>> {
    let mut result: Vec<Dim> = Vec::new();
    let mut l_iter = l.iter().rev().peekable();
    let mut r_iter = r.iter().rev().peekable();
    while let (Some(ld), Some(rd)) = (l_iter.peek(), r_iter.peek()) {
        let combined = match (ld, rd) {
            (Dim::Any, _) | (_, Dim::Any) => Dim::Any,
            (Dim::Known(1), other) | (other, Dim::Known(1)) => (*other).clone(),
            (Dim::Known(a), Dim::Known(b)) if a == b => Dim::Known(*a),
            (Dim::Symbol(s), Dim::Symbol(t)) if s == t => Dim::Symbol(s.clone()),
            // 符号与已知：保守地返回符号维度（假设兼容；真正的 unify 留待 Phase 2）
            (Dim::Symbol(s), Dim::Known(_)) | (Dim::Known(_), Dim::Symbol(s)) => Dim::Symbol(s.clone()),
            _ => return None,
        };
        result.push(combined);
        l_iter.next();
        r_iter.next();
    }
    // 剩余维度直接附加
    for d in l_iter { result.push(d.clone()); }
    for d in r_iter { result.push(d.clone()); }
    result.reverse();
    Some(result)
}

/// 判断 dims 是否包含任何静态信息（Known 或 Symbol）。
/// 全 `Any` 时返回 false（无法检查）。
pub(super) fn has_static_info(dims: &[Dim]) -> bool {
    dims.iter().any(|d| !matches!(d, Dim::Any))
}

/// 人类可读的算符名（用于错误信息）。
fn binop_name(op: &ast::BinOp) -> &'static str {
    use ast::BinOp;
    match op {
        BinOp::Add => "+", BinOp::Sub => "-", BinOp::Mul => "*", BinOp::Div => "/", BinOp::Mod => "%",
        BinOp::Eq => "==", BinOp::NotEq => "!=", BinOp::Lt => "<", BinOp::Gt => ">",
        BinOp::LtEq => "<=", BinOp::GtEq => ">=", BinOp::And => "and", BinOp::Or => "or",
    }
}

/// 格式化 dims 为人类可读字符串（如 `[3, 4]` / `[M, K]` / `[..]`）。
pub(super) fn fmt_dims(dims: &[Dim]) -> String {
    let parts: Vec<String> = dims.iter().map(|d| match d {
        Dim::Known(n) => n.to_string(),
        Dim::Symbol(s) => s.clone(),
        Dim::Any => "..".to_string(),
    }).collect();
    format!("[{}]", parts.join(", "))
}

/// 格式化单个维度。
pub(super) fn fmt_dim(d: &Dim) -> String {
    match d {
        Dim::Known(n) => n.to_string(),
        Dim::Symbol(s) => s.clone(),
        Dim::Any => "..".to_string(),
    }
}

/// 从归约算子的参数中提取字面量 axis（如 `x.sum(0)` 中的 0）。
/// 返回 None 表示无字面量 axis 参数。
fn literal_axis_arg(args: &[HirExpr]) -> Option<i64> {
    for a in args {
        if let HirExprKind::Literal(Literal::Int(n, _)) = &a.kind {
            return Some(*n);
        }
    }
    None
}

/// 从参数中提取所有字面量整数（如 `x.permute(2, 0, 1)` → [2, 0, 1]）。
/// 用于 permute/broadcast_to 等需要整数列表的算子。
/// 任一参数非字面量返回 None。
fn literal_int_args(args: &[HirExpr]) -> Option<Vec<i64>> {
    let mut out: Vec<i64> = Vec::with_capacity(args.len());
    for a in args {
        match &a.kind {
            HirExprKind::Literal(Literal::Int(n, _)) => out.push(*n),
            _ => return None,
        }
    }
    Some(out)
}

impl Lowerer {
    pub(super) fn index_type(&self, base: &Type, indices: &[Index]) -> Type {
        match base {
            Type::Tensor { dtype, dims } => {
                let num_removed = indices.len();
                let remaining: Vec<Dim> = dims.iter().skip(num_removed).cloned().collect();
                if remaining.is_empty() {
                    dtype.as_ref().clone()
                } else {
                    Type::Tensor { dtype: dtype.clone(), dims: remaining }
                }
            }
            // Vec<T> or [T] indexing returns the element type T
            Type::Array { inner, .. } => self.resolve_struct_type((**inner).clone()),
            Type::Generic { base, args } => {
                // Vec<T> -> T
                if let Type::TypeParam { name } = base.as_ref() {
                    if name == "Vec" {
                        return args.first()
                            .map(|t| self.resolve_struct_type(t.clone()))
                            .unwrap_or(Type::Unknown);
                    }
                }
                Type::Unknown
            }
            // String indexing (s[i] or s[a..b]) returns a String (char or slice)
            Type::Base(BaseType::Str) => Type::Base(BaseType::Str),
            // For non-tensor types (Vec, etc.), we don't track element types
            _ => Type::Unknown,
        }
    }

    pub(super) fn infer_binary_type(&self, op: &ast::BinOp, l: &Type, r: &Type) -> Type {
        use ast::BinOp;
        match op {
            BinOp::Eq | BinOp::NotEq | BinOp::Lt | BinOp::Gt | BinOp::LtEq | BinOp::GtEq | BinOp::And | BinOp::Or => {
                Type::bool_()
            }
            BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div | BinOp::Mod => {
                match (l, r) {
                    // Tensor 运算：保留 dtype（若两侧 dtype 不同，按 G4 提升规则取较高精度）
                    (Type::Tensor { dtype: ld, dims: ldims }, Type::Tensor { dtype: rd, dims: rdims }) => {
                        let lb = match ld.as_ref() { Type::Base(b) => *b, _ => BaseType::F64 };
                        let rb = match rd.as_ref() { Type::Base(b) => *b, _ => BaseType::F64 };
                        let promoted = Self::promote_float_dtype(lb, rb);
                        // shape 推断：尝试 broadcast；兼容则返回结果 shape，否则保守 Any
                        // （shape 不匹配的报错由 check_binary_shape_compat 负责）
                        match broadcast_shapes(ldims, rdims) {
                            Some(dims) if !dims.is_empty() => Type::Tensor { dtype: Box::new(Type::Base(promoted)), dims },
                            _ => Type::tensor(promoted, vec![Dim::Any]),
                        }
                    }
                    (Type::Tensor { dtype, .. }, _) | (_, Type::Tensor { dtype, .. }) => {
                        Type::Tensor { dtype: dtype.clone(), dims: vec![Dim::Any] }
                    }
                    // 混合标量：按 G4 规则提升（f64 优先 > f32 > 整数）
                    (Type::Base(lb), Type::Base(rb)) => {
                        Type::Base(Self::promote_float_dtype(*lb, *rb))
                    }
                    _ => l.clone(),
                }
            }
        }
    }

    /// Resolve TypeParam to Struct/Enum if the name matches a known definition.
    pub(super) fn resolve_struct_type(&self, ty: Type) -> Type {
        match &ty {
            Type::TypeParam { name } => {
                if self.structs.contains_key(name) {
                    Type::Struct(name.clone())
                } else if self.enums.contains_key(name) {
                    Type::Enum(name.clone())
                } else if self.unions.contains_key(name) {
                    Type::Union(name.clone())
                } else {
                    ty
                }
            }
            _ => ty,
        }
    }

    pub(super) fn resolve_call_type(&self, func: &HirExpr, args: &[HirExpr], span: &Span) -> TenthResult<Type> {
        match &func.kind {
            HirExprKind::Var(name) => {
                // 使用函数重载解析：根据实参类型匹配合适的签名
                let arg_types: Vec<Type> = args.iter().map(|a| a.ty.clone()).collect();
                match self.scope.resolve_fn_overload(name, &arg_types, span) {
                    Ok((params, ret)) => {
                        // G6（审计缺口）：调用点参数类型检查——单签名函数也校验实参类型。
                        // 仅对单签名执行严格检查：多重载的解析（精确匹配/兼容回退）与
                        // 歧义报告已由 resolve_fn_overload 处理，此处不叠加，避免改变
                        // 既有重载行为。typestate 场景（`File<Closed>` 传 `File<Open>`）
                        // 由此拦截，实现"非法状态表达不出来"的最终保障。
                        if let Some(all) = self.scope.lookup_fn_all(name) {
                            if all.len() == 1 {
                                self.check_call_arg_types(name, &params, args, span)?;
                            }
                        }
                        // 跨函数 shape 求解：若 self.functions 中有更精确的 return_type（body lower 后合并的），用它
                        let ret = if let Some(fn_def) = self.functions.iter().find(|f| f.name == name.as_str()) {
                            Self::merge_return_shape(&ret, &fn_def.return_type)
                        } else {
                            ret
                        };
                        // 断点 4.1（符号维度 unify）：调用点实参代换。
                        // 把被调函数返回 shape 中的 `Dim::Symbol(形参名)` 代换为
                        // 调用点实参推导的维度（字面量→Known、简单变量→Symbol、其他→Any）。
                        // 这是「类型携带参数」的调用点代换（复用 substitute_type 的
                        // 思路，但作用于 Dim::Symbol→Dim）。只代换形参名对应的 Symbol——
                        // Symbol 名不在形参列表（来自局部变量/其他来源）时不代换，
                        // 保持保守（防误报）。见 docs/程序代数架构设计.md §4.1。
                        let ret = {
                            let mut dim_map: HashMap<String, Dim> = HashMap::new();
                            for ((pname, _pty), arg) in params.iter().zip(args.iter()) {
                                dim_map.insert(pname.clone(), Self::dim_from_expr(arg));
                            }
                            Self::substitute_dims_in_type(&ret, &dim_map)
                        };
                        return Ok(self.resolve_struct_type(ret));
                    }
                    Err(e) => {
                        // 若 scope 中无匹配，回退到内置函数检查
                        // 仅当 resolve_fn_overload 报的是"未定义函数"错误时才回退
                        if let TenthError::TypeError { message, .. } = &e {
                            if message.starts_with("未定义的函数") {
                                return self.resolve_builtin(name, args, span);
                            }
                        }
                        return Err(e);
                    }
                }
            }
            _ => Ok(Type::Unknown),
        }
    }

    // ── G6（审计缺口）：调用点参数类型检查 ──────────────────────────────────
    //
    // 设计原则：**只对"确定不兼容"报错**，任何不确定性一律放行（防误报是底线）：
    // - Unknown（未推断）/ Never（发散）→ 放行
    // - 未声明的 TypeParam（泛型变量 `T`、未标注参数、`Self`）→ 放行
    // - 数值类型（整型/浮点/bigint/complex/decimal）互相兼容——与运行时值语义一致：
    //   所有整型都是 Value::Int、所有浮点都是 Value::Float/Float32，且算术对
    //   Int/Float 混合完全多态（Int + Float → Float，见 interpreter/binary.rs），
    //   i32 字面量传 i64 形参、f64 传 f32 形参、int 传 float 形参均能运行
    // - 名义用户类型（struct/enum/union/泛型 struct 名，含 TypeParam 归一）：
    //   同名 → 兼容；异名 → 不兼容（typestate 状态实参拦截的关键）
    // - Generic vs Generic：base 兼容 + 实参逐项兼容（File<Open> vs File<Closed>
    //   的 base 同名但状态实参异名 → 不兼容）
    // - Generic vs 其他：base 是名义用户类型 → 与对方名义名比对；否则
    //   （Vec/Option/Result 等内置泛型）保守放行
    // - Tensor：dtype 兼容 + 维度宽松（Known 精确、Symbol 精确、Any 通配、
    //   Known vs Symbol 放行——沿用 broadcast_shapes 的宽松约定）
    // - Ref/MutRef：inner 兼容（&T 与 &mut T 相互视为可借）
    // - Array/Tuple/FnType：结构逐项
    // - 其余未覆盖类型：放行

    /// 名义用户类型名：`Struct/Enum/Union` 直接取；`TypeParam` 在命中已声明
    /// struct/enum/union/泛型 struct 时视为名义类型（`Open`/`Closed` 声明为 enum
    /// 后即名义）；`Generic` 取其 base 的名义名。未声明的 TypeParam（泛型变量）
    /// 返回 None。
    fn nominal_type_name(&self, t: &Type) -> Option<String> {
        match t {
            Type::Struct(n) | Type::Enum(n) | Type::Union(n) => Some(n.clone()),
            Type::TypeParam { name } => {
                if self.structs.contains_key(name)
                    || self.enums.contains_key(name)
                    || self.unions.contains_key(name)
                    || self.generic_structs.contains_key(name)
                {
                    Some(name.clone())
                } else {
                    None
                }
            }
            Type::Generic { base, .. } => self.nominal_type_name(base),
            _ => None,
        }
    }

    /// 数值基类型：整型/浮点/bigint/complex/decimal 互相兼容（运行时值语义宽松）。
    fn is_numeric_base(b: BaseType) -> bool {
        use BaseType::*;
        matches!(
            b,
            I8 | I16 | I32 | I64 | U8 | U16 | U32 | U64 | F16 | F32 | F64 | BF16 | BigInt | C64 | C128 | Decimal
        )
    }

    /// 维度兼容：Any 通配；Known/Symbol 精确；Known vs Symbol 保守放行
    /// （与 broadcast_shapes 的宽松约定一致，避免符号维度误报）。
    fn dims_compatible(p: &Dim, a: &Dim) -> bool {
        match (p, a) {
            (Dim::Any, _) | (_, Dim::Any) => true,
            (Dim::Known(x), Dim::Known(y)) => x == y,
            (Dim::Symbol(x), Dim::Symbol(y)) => x == y,
            (Dim::Known(_), Dim::Symbol(_)) | (Dim::Symbol(_), Dim::Known(_)) => true,
        }
    }

    /// Tensor 维度列表兼容：**单个 `[Any]`（`Tensor[T, ..]`）是任意秩通配**，
    /// 与任意维度列表兼容；否则要求秩相同且逐维兼容。
    fn tensor_dims_compatible(p: &[Dim], a: &[Dim]) -> bool {
        if p.len() == 1 && matches!(p[0], Dim::Any) {
            return true;
        }
        if a.len() == 1 && matches!(a[0], Dim::Any) {
            return true;
        }
        p.len() == a.len() && p.iter().zip(a.iter()).all(|(x, y)| Self::dims_compatible(x, y))
    }

    /// 判断形参类型能否接受实参类型（保守兼容检查，防误报优先）。
    pub(super) fn types_compatible(&self, param: &Type, arg: &Type) -> bool {
        // 结构等价（最快路径，覆盖绝大多数精确匹配）
        if param == arg {
            return true;
        }
        // Unknown / Never：放行（无法/无需静态检查）
        if matches!(param, Type::Unknown) || matches!(arg, Type::Unknown) {
            return true;
        }
        if matches!(param, Type::Never) || matches!(arg, Type::Never) {
            return true;
        }

        // Generic vs Generic：base 与实参逐项比较。必须在名义捷径之前——
        // `File<Open>` 与 `File<Closed>` 的 base 同名（名义名都是 "File"），
        // 若先走名义捷径会错误短路放行，typestate 状态差异就无法拦截。
        if let (Type::Generic { base: pb, args: pa }, Type::Generic { base: ab, args: aa }) = (param, arg)
        {
            if pa.len() != aa.len() {
                return false;
            }
            return self.types_compatible(pb, ab)
                && pa.iter().zip(aa.iter()).all(|(p, a)| self.types_compatible(p, a));
        }

        // 名义用户类型：两侧都是已声明用户类型时——同名兼容、异名不兼容
        // （typestate 状态实参 `Open` vs `Closed` 在此拦截；同时
        //  `TypeParam("Point")` 形参 vs `Struct("Point")` 实参在此归一放行）
        if let (Some(p), Some(a)) = (self.nominal_type_name(param), self.nominal_type_name(arg)) {
            return p == a;
        }

        // TypeParam 形参：**名义用户类型**（已声明的 struct/enum/union/泛型
        // struct 名）→ 实参必须是同名名义类型（如 `other: Point` 拒收 `42`）；
        // 未声明的 TypeParam（泛型变量 `T` / 未标注参数 / `Self`）→ 放行。
        if let Type::TypeParam { name } = param {
            if self.structs.contains_key(name)
                || self.enums.contains_key(name)
                || self.unions.contains_key(name)
                || self.generic_structs.contains_key(name)
            {
                return self.nominal_type_name(arg).as_deref() == Some(name.as_str());
            }
            return true;
        }
        // TypeParam 实参（泛型变量）：形参是具体类型时无法静态确认 → 放行
        if matches!(arg, Type::TypeParam { .. }) {
            return true;
        }

        match (param, arg) {
            (Type::Base(a), Type::Base(b)) => Self::is_numeric_base(*a) && Self::is_numeric_base(*b),
            // Generic 与具体类型：base 是名义用户类型且对方无同名名义 → 不兼容；
            // 否则（Vec/Option/Result 等内置泛型与数组/标量的交互）保守放行
            (Type::Generic { base, .. }, other) | (other, Type::Generic { base, .. }) => {
                match self.nominal_type_name(base) {
                    Some(b) => self.nominal_type_name(other) == Some(b),
                    None => true,
                }
            }
            (Type::Array { inner: pi, size: ps }, Type::Array { inner: ai, size: as_ }) => {
                let size_ok = match (ps, as_) {
                    (Some(p), Some(a)) => p == a,
                    _ => true,
                };
                size_ok && self.types_compatible(pi, ai)
            }
            (Type::Tuple(ps), Type::Tuple(as_)) => {
                ps.len() == as_.len()
                    && ps.iter().zip(as_.iter()).all(|(p, a)| self.types_compatible(p, a))
            }
            (
                Type::Tensor { dtype: pd, dims: pdims },
                Type::Tensor { dtype: ad, dims: adims },
            ) => {
                self.types_compatible(pd, ad) && Self::tensor_dims_compatible(pdims, adims)
            }
            (Type::Ref(pi, _), Type::Ref(ai, _))
            | (Type::Ref(pi, _), Type::MutRef(ai, _))
            | (Type::MutRef(pi, _), Type::Ref(ai, _))
            | (Type::MutRef(pi, _), Type::MutRef(ai, _)) => self.types_compatible(pi, ai),
            (Type::FnType { params: pp, ret: pr }, Type::FnType { params: ap, ret: ar }) => {
                pp.len() == ap.len()
                    && pp.iter().zip(ap.iter()).all(|(p, a)| self.types_compatible(p, a))
                    && self.types_compatible(pr, ar)
            }
            // 其余未覆盖类型（Range/HeapBox/SharedBox/Pin/Future/Dyn 等）：
            // 保守放行，避免规则不全导致误报
            _ => true,
        }
    }

    /// 校验调用点实参类型与形参类型兼容（只对确定不兼容报错，带行/列）。
    /// 尊重变参/默认参数：只检查两侧都存在的参数位置（min 长度），
    /// 多余实参（变参）与缺失实参（默认值）都不触发。
    pub(super) fn check_call_arg_types(
        &self,
        func_name: &str,
        params: &[(String, Type)],
        args: &[HirExpr],
        span: &Span,
    ) -> TenthResult<()> {
        let n = params.len().min(args.len());
        for i in 0..n {
            let (pname, pty) = &params[i];
            let aty = &args[i].ty;
            if !self.types_compatible(pty, aty) {
                return Err(TenthError::TypeError {
                    line: span.line,
                    col: span.col,
                    message: format!(
                        "函数 '{}' 的第 {} 个实参（{}）类型不兼容：期望 '{}'，实参为 '{}'",
                        func_name,
                        i + 1,
                        pname,
                        pty,
                        aty
                    ),
                });
            }
        }
        Ok(())
    }

    pub(super) fn resolve_method_type(&self, receiver: &Type, method: &str, _args: &[HirExpr]) -> Type {
        match receiver {
            Type::Tensor { dtype, dims } => {
                match method {
                    "matmul" => {
                        // 2D matmul: (M, K) @ (K, N) → (M, N)
                        // 静态 shape 推断：若两侧 dims 都已知且 K 匹配，返回精确 shape；
                        // 否则保守返回 2D Any（不匹配的报错由 check_method_shape 负责）
                        if dims.len() == 2 {
                            if let Some(arg) = _args.first() {
                                if let Type::Tensor { dims: adims, .. } = &arg.ty {
                                    if adims.len() == 2 {
                                        // 两侧 K 都已知且相等时才能静态推断
                                        let k_match = match (&dims[1], &adims[0]) {
                                            (Dim::Known(a), Dim::Known(b)) => a == b,
                                            (Dim::Symbol(a), Dim::Symbol(b)) => a == b,
                                            (Dim::Any, _) | (_, Dim::Any) => true,  // 未知：保守视为兼容
                                            _ => false,
                                        };
                                        if k_match {
                                            return Type::Tensor {
                                                dtype: dtype.clone(),
                                                dims: vec![dims[0].clone(), adims[1].clone()],
                                            };
                                        }
                                        // K 不匹配：返回 Unknown，由 check 报错
                                        return Type::Unknown;
                                    }
                                }
                            }
                            // 参数 shape 未知：保守返回 2D Any
                            return Type::Tensor { dtype: dtype.clone(), dims: vec![Dim::Any, Dim::Any] };
                        }
                        // 非 2D：运行时只支持 2D，返回 Unknown
                        Type::Unknown
                    }
                    "bmm" => {
                        // 3D batched matmul: (B, M, K) @ (B, K, N) → (B, M, N)
                        // 静态 shape 推断：若两侧 dims 都是 3D Known 且 B/K 匹配，返回精确 shape；
                        // 否则保守返回 3D Any（不匹配的报错由 check_method_shape 负责）
                        if dims.len() == 3 {
                            if let Some(arg) = _args.first() {
                                if let Type::Tensor { dims: adims, .. } = &arg.ty {
                                    if adims.len() == 3 {
                                        // batch (B) 和内侧 K 都必须兼容
                                        let b_match = match (&dims[0], &adims[0]) {
                                            (Dim::Known(a), Dim::Known(b)) => a == b,
                                            (Dim::Symbol(a), Dim::Symbol(b)) => a == b,
                                            (Dim::Any, _) | (_, Dim::Any) => true,
                                            _ => false,
                                        };
                                        let k_match = match (&dims[2], &adims[1]) {
                                            (Dim::Known(a), Dim::Known(b)) => a == b,
                                            (Dim::Symbol(a), Dim::Symbol(b)) => a == b,
                                            (Dim::Any, _) | (_, Dim::Any) => true,
                                            _ => false,
                                        };
                                        if b_match && k_match {
                                            return Type::Tensor {
                                                dtype: dtype.clone(),
                                                dims: vec![dims[0].clone(), dims[1].clone(), adims[2].clone()],
                                            };
                                        }
                                        // B 或 K 不匹配：返回 Unknown，由 check 报错
                                        return Type::Unknown;
                                    }
                                }
                            }
                            // 参数 shape 未知：保守返回 3D Any
                            return Type::Tensor { dtype: dtype.clone(), dims: vec![Dim::Any, Dim::Any, Dim::Any] };
                        }
                        // 非 3D：运行时只支持 3D，返回 Unknown
                        Type::Unknown
                    }
                    // 归约算子（sum/mean/max/min）：
                    // - 无参数：全部降维到标量（dtype）
                    // - 字面量 axis 参数（如 x.sum(0)）：移除指定维度
                    // - 变量参数（如 keepdim 标志）：保守保持原 shape（运行时处理）
                    "sum" | "mean" | "max" | "min" => {
                        if let Some(axis) = literal_axis_arg(_args) {
                            if axis >= 0 && (axis as usize) < dims.len() {
                                let mut new_dims: Vec<Dim> = dims.iter().cloned().collect();
                                new_dims.remove(axis as usize);
                                if new_dims.is_empty() {
                                    dtype.as_ref().clone()
                                } else {
                                    Type::Tensor { dtype: dtype.clone(), dims: new_dims }
                                }
                            } else {
                                // axis 越界：保守返回标量
                                dtype.as_ref().clone()
                            }
                        } else if _args.iter().any(|a| matches!(&a.kind, HirExprKind::Var(_))) {
                            Type::Tensor { dtype: dtype.clone(), dims: dims.clone() }
                        } else {
                            dtype.as_ref().clone()
                        }
                    }
                    // reshape/view：从字面量参数推断新 shape（如 x.reshape(3, 4) → [3, 4]）
                    "reshape" | "view" => {
                        Type::Tensor { dtype: dtype.clone(), dims: Self::shape_from_int_args(_args) }
                    }
                    // flatten：展平为 1D（元素总数未知，因可能含动态维度）
                    "flatten" => Type::Tensor { dtype: dtype.clone(), dims: vec![Dim::Any] },
                    // 逐元素激活/数学函数：保持原 shape
                    "abs" | "sqrt" | "exp" | "log" | "relu" |
                    "sigmoid" | "tanh" | "softmax" | "gelu" => {
                        Type::Tensor { dtype: dtype.clone(), dims: dims.clone() }
                    }
                    // masked_fill(mask, value)：保持原 shape
                    "masked_fill" => Type::Tensor { dtype: dtype.clone(), dims: dims.clone() },
                    // permute(dims...)：按字面量索引重排原 dims（如 [3,8,5].permute(2,0,1) → [5,3,8]）
                    // 字面量参数：按索引重排；非字面量：保守返回原秩的 Any
                    "permute" => {
                        match literal_int_args(_args) {
                            Some(idxs) if !idxs.is_empty() => {
                                let mut new_dims: Vec<Dim> = Vec::with_capacity(idxs.len());
                                let mut ok = true;
                                for i in &idxs {
                                    if *i >= 0 && (*i as usize) < dims.len() {
                                        new_dims.push(dims[*i as usize].clone());
                                    } else {
                                        ok = false;
                                        break;
                                    }
                                }
                                if ok {
                                    Type::Tensor { dtype: dtype.clone(), dims: new_dims }
                                } else {
                                    Type::Tensor { dtype: dtype.clone(), dims: vec![Dim::Any; idxs.len()] }
                                }
                            }
                            _ => Type::Tensor { dtype: dtype.clone(), dims: dims.clone() },
                        }
                    }
                    // broadcast_to(shape...)：字面量参数即目标 shape
                    "broadcast_to" => {
                        Type::Tensor { dtype: dtype.clone(), dims: Self::shape_from_int_args(_args) }
                    }
                    // cat(other, dim=0)：沿 dim 拼接，dim 维相加，其余维度必须匹配
                    // 字面量 dim + 两侧 shape 已知 → 精确推断；否则保守返回原秩的 Any
                    "cat" => {
                        let dim = _args.get(1)
                            .and_then(|a| match &a.kind {
                                HirExprKind::Literal(Literal::Int(n, _)) => Some(*n),
                                _ => None,
                            })
                            .unwrap_or(0);
                        if let Some(arg) = _args.first() {
                            if let Type::Tensor { dims: adims, .. } = &arg.ty {
                                if adims.len() == dims.len() && dim >= 0 && (dim as usize) < dims.len() {
                                    let mut new_dims: Vec<Dim> = dims.iter().cloned().collect();
                                    new_dims[dim as usize] = match (&dims[dim as usize], &adims[dim as usize]) {
                                        (Dim::Known(a), Dim::Known(b)) => Dim::Known(a + b),
                                        _ => Dim::Any,
                                    };
                                    return Type::Tensor { dtype: dtype.clone(), dims: new_dims };
                                }
                            }
                        }
                        Type::Tensor { dtype: dtype.clone(), dims: vec![Dim::Any; dims.len().max(1)] }
                    }
                    // argmax/argmin：返回 i64 标量（当前运行时仅支持全张量归约，无 axis 参数）
                    "argmax" | "argmin" => Type::Base(BaseType::I64),
                    // transpose：交换最后两维（与运行时 tensor::methods.rs:1057-1063 对齐）
                    "transpose" => {
                        // transpose() 交换最后两维（与运行时 tensor::methods.rs:1057-1063 对齐）
                        // 2D: (M, K) → (K, M)；3D: (B, M, K) → (B, K, M)；其他维度数 ≥ 2 同理
                        if dims.len() >= 2 {
                            let n = dims.len();
                            let mut new_dims = dims.clone();
                            new_dims[n - 1] = dims[n - 2].clone();
                            new_dims[n - 2] = dims[n - 1].clone();
                            Type::Tensor { dtype: dtype.clone(), dims: new_dims }
                        } else {
                            Type::Tensor { dtype: dtype.clone(), dims: dims.clone() }
                        }
                    }
                    "to_vec" => Type::Array { inner: Box::new(dtype.as_ref().clone()), size: None },
                    "len" | "size" | "dim" => Type::Base(BaseType::I64),
                    "shape" => Type::Array { inner: Box::new(Type::Base(BaseType::I64)), size: None },
                    _ => Type::Unknown,
                }
            }
            Type::Base(BaseType::Str) => match method {
                "len" => Type::Base(BaseType::I64),
                "contains" | "starts_with" | "ends_with" => Type::bool_(),
                "trim" | "to_lowercase" | "to_uppercase" => Type::str_(),
                "split" | "lines" => Type::Array { inner: Box::new(Type::str_()), size: None },
                "replace" => Type::str_(),
                "parse_int" | "parse_float" => Type::Enum("Option".to_string()),
                "chars" => Type::Array { inner: Box::new(Type::Base(BaseType::Char)), size: None },
                _ => Type::Unknown,
            },
            Type::Array { inner, size } => match method {
                "len" => Type::Base(BaseType::I64),
                "push" => Type::unit(),
                "pop" => Type::Enum("Option".to_string()),
                "get" => Type::Enum("Option".to_string()),
                "map" | "filter" => Type::Array { inner: inner.clone(), size: *size },
                "is_empty" => Type::bool_(),
                "iter" => Type::Unknown,
                _ => Type::Unknown,
            },
            _ => {
                // 阶段2a M2（G3）：用户自定义方法——查方法表返回真实返回类型。
                // 状态传播的关键：`close(self) -> File<Closed>` 的返回类型在此取出，
                // 使 `let c = f.close()` 后 c 的类型为 File<Closed>，后续方法解析
                // 按 Closed 状态过滤。Vec/Option 等内置泛型不在方法表中，返回 None
                // 落到下方原生方法兜底，保持既有行为。
                if let Some(def) = self.find_inherent_method(receiver, method) {
                    return def.return_type.clone();
                }
                match method {
                    "len" => Type::Base(BaseType::I64),
                    "push" => Type::unit(),
                    "get" => Type::Unknown,
                    _ => Type::Unknown,
                }
            }
        }
    }

    pub(super) fn resolve_builtin(&self, name: &str, args: &[HirExpr], _span: &Span) -> TenthResult<Type> {
        match name {
            "println" | "eprintln" | "eprint" => Ok(Type::unit()),
            // Stage 1+2 I/O 原语
            "env_set" | "exit" => Ok(Type::unit()),
            "read_line" | "env_get" => Ok(Type::Enum("Result".to_string())),
            // 阶段1-静默失败：or_die(x, msg) / assume_ok(x) 自由函数 native。
            // 返回类型 = Result/Option 的内部值类型（提取泛型参数 args[0]）。
            // - Result<T, E> → T；Option<T> → T
            // - base 名匹配：注解 `Result<i64, str>` 解析出的 base 是
            //   TypeParam("Result")（from_annotation 的 Generic base 走 Named），
            //   EnumLiteral 构造的 Result 是 Type::Enum("Result")——两者都匹配。
            // - 无泛型参数（如 read_line 保守注册为 Type::Enum("Result")）→ Unknown
            //   此时运行时仍正确解包，仅静态类型信息缺失。
            "or_die" | "assume_ok" => {
                let inner = match args.first().map(|a| &a.ty) {
                    Some(Type::Generic { base, args: gen_args }) => {
                        let base_name = match base.as_ref() {
                            Type::Enum(name) | Type::TypeParam { name } => name,
                            _ => "",
                        };
                        if base_name == "Result" || base_name == "Option" {
                            gen_args.first().cloned().unwrap_or(Type::Unknown)
                        } else {
                            Type::Unknown
                        }
                    }
                    _ => Type::Unknown,
                };
                Ok(inner)
            }
            // Stage 3+4 TCP/HTTP 原语：返回 Unit 的 close/set_timeout 与返回 Result 的其余
            "tcp_close" | "tcp_set_timeout" | "tcp_listener_close" | "command_arg" => Ok(Type::unit()),
            "tcp_connect" | "tcp_read" | "tcp_write" | "http_get" | "http_post"
            | "tcp_listen" | "tcp_accept" | "command_new" | "command_run" | "command_output" => {
                Ok(Type::Enum("Result".to_string()))
            }
            // UDP 原语（基本功核查第 69 项）：close/set_timeout 返回 Unit；bind/recv_from/send_to 返回 Result
            // recv_from 内部返回 Tuple<Vec<i64>, String>，但 HIR 类型推断保守返回 Result 枚举
            // （Tuple 内部类型在运行时由 native 填充，与 tcp_read 的 Vec<i64> 同模式）。
            "udp_close" | "udp_set_timeout" => Ok(Type::unit()),
            "udp_bind" | "udp_recv_from" | "udp_send_to" => Ok(Type::Enum("Result".to_string())),
            // Phase 2 Step 5：异步 I/O 原语（返回 Future，类型暂为 Unknown——
            // await 解包后才是 Result/Unit，由 Op::Await 在运行时处理）
            "async_sleep_ms" | "async_tcp_read" | "async_tcp_write" => Ok(Type::Unknown),
            // 正则表达式原语：handle table 模式，与 std/regex.th 契约对齐
            "regex_compile" => Ok(Type::Enum("Result".to_string())),
            "regex_match" => Ok(Type::Base(BaseType::Bool)),
            "regex_find" | "regex_replace" => Ok(Type::str_()),
            "regex_find_all" | "regex_split" => Ok(Type::Array { inner: Box::new(Type::Unknown), size: None }),
            // Tensor 构造函数：dtype 从参数推断（若无 f32 线索则默认 F64）
            "tensor" => Ok(Type::tensor(Self::infer_tensor_dtype(args), Self::shape_from_int_args(args))),
            "rand" | "randn" => Ok(Type::tensor(Self::infer_tensor_dtype(args), Self::shape_from_int_args(args))),
            "randn_f32" => Ok(Type::tensor(BaseType::F32, vec![Dim::Any])),
            // Phase 5.5：补全 f32 构造函数 native 注册
            "rand_f32" | "zeros_f32" | "ones_f32" => Ok(Type::tensor(BaseType::F32, vec![Dim::Any])),
            // Wave 2：f16/bf16 构造函数 native 注册
            "zeros_f16" | "ones_f16" => Ok(Type::tensor(BaseType::F16, vec![Dim::Any])),
            "zeros_bf16" | "ones_bf16" => Ok(Type::tensor(BaseType::BF16, vec![Dim::Any])),
            "read_file" => Ok(Type::str_()),
            "str_at" => Ok(Type::str_()),
            "write_file" | "write_bytes" => Ok(Type::unit()),
            "Vec::new" => Ok(Type::Array { inner: Box::new(Type::Unknown), size: None }),
            "HashMap::new" => Ok(Type::Unknown),
            "compile_host" => Ok(Type::Base(BaseType::I32)),
            "format" => Ok(Type::str_()),
            "to_string" | "type_name" => Ok(Type::str_()),
            "with_step_limit" | "with_timeout_ms" => Ok(Type::Unknown),
            "is_timeout" => Ok(Type::bool_()),
            "parse_int" => Ok(Type::Enum("Option".to_string())),
            "parse_float" => Ok(Type::Enum("Option".to_string())),
            // 标量数学函数：dtype 跟随输入
            "abs" | "sqrt" | "sin" | "cos" | "ln" | "pow" => Ok(Self::infer_scalar_dtype(args, Type::f64())),
            // to_float 保留为 f64 别名（向后兼容）；新增 to_f32 / to_f64
            "to_float" | "to_f64" => Ok(Type::f64()),
            "to_f32" => Ok(Type::f32()),
            "f64_bits" => Ok(Type::Base(BaseType::I64)),
            "f64_from_bits" => Ok(Type::f64()),
            "tensor_from_vec" => Ok(Type::tensor(Self::infer_tensor_dtype(args), Self::shape_from_int_args(args))),
            "zeros" | "ones" => Ok(Type::tensor(Self::infer_tensor_dtype(args), Self::shape_from_int_args(args))),
            "save_weights" | "load_weights" => Ok(Type::unit()),
            "cross_entropy" => Ok(Type::tensor(Self::infer_tensor_dtype(args), vec![Dim::Any])),
            // select 原语（论文 T47/T48/T50）：broadcast 三输入 shape；dtype 由 then/else 决定
            "select" => {
                // 提取三个输入的 dims（若任一非 Tensor，保守返回 Any）
                let mut input_dims: Vec<&[Dim]> = Vec::new();
                for a in args.iter().take(3) {
                    if let Type::Tensor { dims, .. } = &a.ty {
                        input_dims.push(dims.as_slice());
                    } else {
                        return Ok(Type::tensor(Self::infer_tensor_dtype(args), vec![Dim::Any]));
                    }
                }
                if input_dims.len() < 3 {
                    return Ok(Type::tensor(Self::infer_tensor_dtype(args), vec![Dim::Any]));
                }
                // 链式 broadcast：cond ⊗ then ⊗ else
                let mut acc: Vec<Dim> = input_dims[0].to_vec();
                for dims in &input_dims[1..] {
                    match broadcast_shapes(&acc, dims) {
                        Some(b) => acc = b,
                        None => return Ok(Type::Unknown),
                    }
                }
                // dtype：若 then/else 任一为 F32 则 F32，否则 F64
                let dtype = match (&args[1].ty, &args[2].ty) {
                    (Type::Tensor { dtype: dt, .. }, _) if matches!(dt.as_ref(), Type::Base(BaseType::F32)) => BaseType::F32,
                    (_, Type::Tensor { dtype: dt, .. }) if matches!(dt.as_ref(), Type::Base(BaseType::F32)) => BaseType::F32,
                    _ => BaseType::F64,
                };
                Ok(Type::tensor(dtype, acc))
            },
            // gather 原语：out.shape == index.shape；dtype 跟随 base（args[0]）
            "gather" => {
                // args = [base, dim, index]
                let base_dtype = match args.first() {
                    Some(HirExpr { ty: Type::Tensor { dtype, .. }, .. }) if matches!(dtype.as_ref(), Type::Base(BaseType::F32)) => BaseType::F32,
                    _ => BaseType::F64,
                };
                let index_dims = match args.get(2) {
                    Some(HirExpr { ty: Type::Tensor { dims, .. }, .. }) => dims.clone(),
                    _ => vec![Dim::Any],
                };
                Ok(Type::tensor(base_dtype, index_dims))
            },
            // PROJ-006：__call_custom_op(op_id, ...inputs)
            // 编译期无法预知用户算子的 forward_shape（CustomBackward::forward_shape 默认 None），
            // 保守返回 Tensor[Dim::Any]（dtype 跟随输入张量，护城河 A 走运行时兜底）。
            "__call_custom_op" => {
                // 从 args[1..] 推断 dtype（若任一为 F32 则 F32，否则 F64）
                let dtype = args.iter().skip(1).find_map(|a| match &a.ty {
                    Type::Tensor { dtype, .. } if matches!(dtype.as_ref(), Type::Base(BaseType::F32)) => Some(BaseType::F32),
                    _ => None,
                }).unwrap_or(BaseType::F64);
                Ok(Type::tensor(dtype, vec![Dim::Any]))
            },
            // Wave 2 第 4 项：张量比较 native（gt/lt/ge/le/eq/ne）
            // 返回 F64 张量（0.0/1.0 编码 bool）；shape = broadcast(a.shape, b.shape)
            "tensor_gt" | "tensor_lt" | "tensor_ge" | "tensor_le" | "tensor_eq" | "tensor_ne" => {
                // 提取两个输入的 dims（若任一非 Tensor，保守返回 F64 Tensor[..]）
                let mut input_dims: Vec<&[Dim]> = Vec::new();
                for a in args.iter().take(2) {
                    if let Type::Tensor { dims, .. } = &a.ty {
                        input_dims.push(dims.as_slice());
                    } else {
                        return Ok(Type::tensor(BaseType::F64, vec![Dim::Any]));
                    }
                }
                if input_dims.len() < 2 {
                    return Ok(Type::tensor(BaseType::F64, vec![Dim::Any]));
                }
                match broadcast_shapes(input_dims[0], input_dims[1]) {
                    Some(b) => Ok(Type::tensor(BaseType::F64, b)),
                    None => Ok(Type::Unknown),
                }
            },
            "start_grad" | "new_grad" | "stop_grad" | "param" => Ok(Type::tensor(Self::infer_tensor_dtype(args), vec![Dim::Any])),
            "backward" => Ok(Type::unit()),
            "grad" | "zero_grad" => Ok(Type::Unknown),
            "path_join" => Ok(Type::str_()),
            "path_exists" | "path_is_file" | "path_is_dir" => Ok(Type::bool_()),
            "mkdir" => Ok(Type::unit()),
            "list_dir" => Ok(Type::Array { inner: Box::new(Type::str_()), size: None }),
            "file_size" => Ok(Type::Base(BaseType::I64)),
            "remove_file" | "copy_file" => Ok(Type::unit()),
            "lexer_new" | "lexer_tokenize" | "parse_program" | "lower_program" | "compile_to_wasm" | "compile_program" => Ok(Type::Unknown),
            // Wave 3 第 8 项：Date native 类型签名
            // - date_to_unix_days / date_i64_add_days / date_diff_days / date_day_of_week → i64
            // - date_from_unix_days → (i64, i64, i64) Tuple（标准库 date.th 用 `let (y,m,d) = ...` 解构）
            "date_to_unix_days" | "date_i64_add_days" | "date_diff_days" | "date_day_of_week" => {
                Ok(Type::Base(BaseType::I64))
            }
            "date_from_unix_days" => {
                Ok(Type::Tuple(vec![
                    Type::Base(BaseType::I64),
                    Type::Base(BaseType::I64),
                    Type::Base(BaseType::I64),
                ]))
            }
            _ => Ok(Type::Unknown),
        }
    }

    /// 根据参数列表推断 Tensor dtype。
    /// 规则：若任一参数是 F32（字面量或类型注解为 F32），则结果为 F32；否则默认 F64。
    pub(super) fn infer_tensor_dtype(args: &[HirExpr]) -> BaseType {
        for a in args {
            match &a.ty {
                Type::Base(BaseType::F32) => return BaseType::F32,
                Type::Tensor { dtype, .. } if matches!(dtype.as_ref(), Type::Base(BaseType::F32)) => return BaseType::F32,
                _ => {}
            }
        }
        BaseType::F64
    }

    /// 从构造函数的参数推断 shape。
    /// - 整数字面量（如 `zeros(3, 4)`）→ `[Known(3), Known(4)]`
    /// - 简单变量（如 `randn(n)`）→ `[Symbol("n")]`（P1 层级一：变量提升为 Symbol）
    /// - 其他形式（表达式、函数调用等，如 `zeros(n*2)`）→ `[Any]`（运行时才能确定）
    pub(super) fn shape_from_int_args(args: &[HirExpr]) -> Vec<Dim> {
        if args.is_empty() {
            return vec![Dim::Any];
        }
        let mut dims: Vec<Dim> = Vec::with_capacity(args.len());
        for a in args {
            match &a.kind {
                HirExprKind::Literal(Literal::Int(n, _)) => dims.push(Dim::Known(*n)),
                HirExprKind::Var(name) => dims.push(Dim::Symbol(name.clone())),
                _ => return vec![Dim::Any],
            }
        }
        dims
    }

    /// 从单个实参表达式推导维度（断点 4.1 调用点实参代换用）。
    /// 与 `shape_from_int_args` 的逐参数语义一致：
    /// - 整数字面量 → `Known(n)`
    /// - 简单变量 → `Symbol(name)`
    /// - 其他形式（表达式、函数调用等）→ `Any`（运行时才能确定，保守）
    pub(super) fn dim_from_expr(expr: &HirExpr) -> Dim {
        match &expr.kind {
            HirExprKind::Literal(Literal::Int(n, _)) => Dim::Known(*n),
            HirExprKind::Var(name) => Dim::Symbol(name.clone()),
            _ => Dim::Any,
        }
    }

    /// 断点 4.1（符号维度 unify）：把类型中 Tensor 维度的 `Dim::Symbol(name)`
    /// 代换为调用点实参推导的维度（map）。递归下降所有子类型（dtype/Array/
    /// Tuple/Generic/Ref/MutRef），形态与 `substitute_type`（`mod.rs:278`，
    /// TypeParam→Type）一致，但作用对象是 Dim::Symbol→Dim。
    /// map 中不存在的 Symbol 保持原样（保守，不猜测）。
    pub(super) fn substitute_dims_in_type(ty: &Type, map: &HashMap<String, Dim>) -> Type {
        match ty {
            Type::Tensor { dtype, dims } => {
                let new_dims: Vec<Dim> = dims.iter().map(|d| match d {
                    Dim::Symbol(name) => map.get(name).cloned().unwrap_or_else(|| d.clone()),
                    other => other.clone(),
                }).collect();
                Type::Tensor {
                    dtype: Box::new(Self::substitute_dims_in_type(dtype, map)),
                    dims: new_dims,
                }
            }
            Type::Array { inner, size } => Type::Array {
                inner: Box::new(Self::substitute_dims_in_type(inner, map)),
                size: *size,
            },
            Type::Tuple(types) => Type::Tuple(types.iter().map(|t| Self::substitute_dims_in_type(t, map)).collect()),
            Type::Generic { base, args } => Type::Generic {
                base: Box::new(Self::substitute_dims_in_type(base, map)),
                args: args.iter().map(|t| Self::substitute_dims_in_type(t, map)).collect(),
            },
            Type::Ref(inner, lt) => Type::Ref(Box::new(Self::substitute_dims_in_type(inner, map)), lt.clone()),
            Type::MutRef(inner, lt) => Type::MutRef(Box::new(Self::substitute_dims_in_type(inner, map)), lt.clone()),
            _ => ty.clone(),
        }
    }

    /// 标量函数 dtype 推断：若输入为 F32 则返回 F32，否则返回默认（fallback）。
    pub(super) fn infer_scalar_dtype(args: &[HirExpr], fallback: Type) -> Type {
        for a in args {
            if matches!(&a.ty, Type::Base(BaseType::F32)) {
                return Type::f32();
            }
        }
        fallback
    }

    /// 按 spec §4.3 隐式转换规则提升两个 dtype：
    /// - f64 与任意浮点 → f64
    /// - f32 与 f32 → f32
    /// - f32 与整数 → f32
    /// - f64 与整数 → f64
    /// - 整数与整数 → 左侧（保留现有整数运算语义）
    pub(super) fn promote_float_dtype(l: BaseType, r: BaseType) -> BaseType {
        use BaseType::*;
        match (l, r) {
            (F64, _) | (_, F64) => F64,
            (F32, _) | (_, F32) => F32,
            (F16, _) | (_, F16) => F16,
            (BF16, _) | (_, BF16) => BF16,
            _ => l,
        }
    }

    /// 跨函数 shape 求解：合并 scope 中的 return_type 和 fn_def 中的 return_type。
    ///
    /// fn_def.return_type 可能在函数体 lower 后被更新（含更精确的 shape）。
    /// 此函数取两者中更精确的 shape（Known/Symbol 优先于 Any）。
    /// 若两者都是 Tensor，逐维取更精确的；否则取 fn_def 的（可能更精确）。
    fn merge_return_shape(scope_ret: &Type, fn_def_ret: &Type) -> Type {
        match (scope_ret, fn_def_ret) {
            (Type::Tensor { dtype: sd, dims: sdims }, Type::Tensor { dtype: fd, dims: fdims }) => {
                // dtype 取 fn_def 的（可能更精确，如从 body 推断的 F32 vs scope 的 Unknown）
                let dtype = if matches!(sd.as_ref(), Type::Unknown) { fd.clone() } else { sd.clone() };
                // 若维度数不同，取 fn_def 的（可能是 body 推断的精确维度数）
                if sdims.len() != fdims.len() {
                    return Type::Tensor { dtype, dims: fdims.clone() };
                }
                // 逐维取更精确的
                let dims: Vec<Dim> = sdims.iter().zip(fdims.iter()).map(|(s, f)| {
                    match (s, f) {
                        // fn_def 有精确信息，优先用
                        (Dim::Any, f_precise) => (*f_precise).clone(),
                        // scope 有精确信息，fn_def 是 Any，用 scope 的
                        (s_precise, Dim::Any) => (*s_precise).clone(),
                        // 两者都精确，取 fn_def 的（body 推断更新过）
                        (_, f_precise) => (*f_precise).clone(),
                    }
                }).collect();
                Type::Tensor { dtype, dims }
            }
            // 非 Tensor 或不匹配：取 fn_def 的
            _ => fn_def_ret.clone(),
        }
    }

    /// 函子化 shape 分析核心：对已 lower 的 HIR 做结构递归，收集表达式树中
    /// 所有"return 路径"的 Tensor shape（含隐式末表达式路径）。
    ///
    /// 这是映射 Φ: 程序片段 → shape 空间 的**构造性定义**：
    /// 每个 IR 构造的组合规则直接给出，组合性由构造保证，
    /// 不依赖任何全局可变收集器（旧的 `current_fn_return_shapes` 字段已移除）：
    /// - `Block`：所有 stmt 的 return 路径 ∪ `final_expr` 的 shape（隐式返回）
    /// - `If`：then ∪ else 两分支
    /// - `Match`：所有 arm body
    /// - 循环体（While/DoWhile/For/Loop）：体内的 return 路径
    /// - 调用节点（Call/MethodCall/GenericCall）：其 `ret_ty` 已经编码了 Φ(callee)
    ///   在调用点的结果——`Φ(f∘g) = Φ(f)∘Φ(g)` 在此自动成立
    /// - `Closure`：**不下降**（闭包是独立函数体，其 return 属于闭包自身；
    ///   旧收集器因 lowering 时共享 Lowerer 状态会把闭包 return 误算进外围函数，
    ///   纯递归版本天然修复此缺陷）
    ///
    /// 语义等价于旧的"在 Return 语句处 push"的手工收集器，但：
    /// 1) 是纯函数——不依赖 Lowerer 的任何可变状态；
    /// 2) 顺序无关——join 结果与遍历顺序无关（join 是交换的）；
    /// 3) 不下钻闭包体——如上所述，修复潜在缺陷。
    pub(super) fn collect_return_tensor_dims(expr: &HirExpr) -> Vec<Vec<Dim>> {
        let mut out = Vec::new();
        Self::collect_return_dims_expr(expr, &mut out);
        out
    }

    fn collect_return_dims_expr(expr: &HirExpr, out: &mut Vec<Vec<Dim>>) {
        match &expr.kind {
            HirExprKind::Block { stmts, final_expr } => {
                for s in stmts {
                    Self::collect_return_dims_stmt(s, out);
                }
                // 注意：嵌套 Block 的 final_expr 不是 return 路径——其值向上流动，
                // 由顶层 lowered_body.ty 捕获（见 lower_stmt.rs）。这里只递归
                // 查找 final_expr 内部嵌套的 return 语句。
                if let Some(fe) = final_expr {
                    Self::collect_return_dims_expr(fe, out);
                }
            }
            HirExprKind::If { cond, then_branch, else_branch, .. } => {
                Self::collect_return_dims_expr(cond, out);
                Self::collect_return_dims_expr(then_branch, out);
                if let Some(eb) = else_branch {
                    Self::collect_return_dims_expr(eb, out);
                }
            }
            HirExprKind::Match { scrutinee, arms, .. } => {
                Self::collect_return_dims_expr(scrutinee, out);
                for arm in arms {
                    if let Some(g) = &arm.guard {
                        Self::collect_return_dims_expr(g, out);
                    }
                    Self::collect_return_dims_expr(&arm.body, out);
                }
            }
            // Closure：闭包体是独立函数，其 return 属于闭包自身，不下钻。
            HirExprKind::Closure { .. } => {}
            // 其余表达式节点：return 是语句，只出现在 Block/If/Match/循环体的
            // 子位置；但子表达式本身可能是 Block（如调用参数、二元操作数），
            // 因此仍需下降以找到嵌套的 return。逐个枚举所有子表达式。
            HirExprKind::Binary { left, right, .. } => {
                Self::collect_return_dims_expr(left, out);
                Self::collect_return_dims_expr(right, out);
            }
            HirExprKind::Unary { expr: inner, .. } => {
                Self::collect_return_dims_expr(inner, out);
            }
            // lossy 是编译期包装（运行时 no-op）：内层表达式的 return 仍需下钻
            HirExprKind::Lossy(inner) => {
                Self::collect_return_dims_expr(inner, out);
            }
            HirExprKind::Call { func, args, .. } => {
                Self::collect_return_dims_expr(func, out);
                for a in args {
                    Self::collect_return_dims_expr(a, out);
                }
            }
            HirExprKind::GenericCall { func, args, .. } => {
                Self::collect_return_dims_expr(func, out);
                for a in args {
                    Self::collect_return_dims_expr(a, out);
                }
            }
            HirExprKind::MethodCall { receiver, args, .. } => {
                Self::collect_return_dims_expr(receiver, out);
                for a in args {
                    Self::collect_return_dims_expr(a, out);
                }
            }
            HirExprKind::Index { target, indices } => {
                Self::collect_return_dims_expr(target, out);
                for idx in indices {
                    match idx {
                        Index::Single(e) => Self::collect_return_dims_expr(e, out),
                        Index::Range { start, end, .. } => {
                            if let Some(s) = start {
                                Self::collect_return_dims_expr(s, out);
                            }
                            if let Some(e) = end {
                                Self::collect_return_dims_expr(e, out);
                            }
                        }
                        Index::Colon => {}
                    }
                }
            }
            HirExprKind::Field { target, .. } => {
                Self::collect_return_dims_expr(target, out);
            }
            HirExprKind::TensorLiteral { data, .. } => {
                for row in data {
                    for e in row {
                        Self::collect_return_dims_expr(e, out);
                    }
                }
            }
            HirExprKind::ArrayLiteral { elements, .. } => {
                for e in elements {
                    Self::collect_return_dims_expr(e, out);
                }
            }
            HirExprKind::Range { start, end, .. } => {
                if let Some(s) = start {
                    Self::collect_return_dims_expr(s, out);
                }
                if let Some(e) = end {
                    Self::collect_return_dims_expr(e, out);
                }
            }
            HirExprKind::Assign { value, .. } => {
                Self::collect_return_dims_expr(value, out);
            }
            HirExprKind::AssignOp { value, .. } => {
                Self::collect_return_dims_expr(value, out);
            }
            HirExprKind::StructLiteral { fields, .. } => {
                for (_, e) in fields {
                    Self::collect_return_dims_expr(e, out);
                }
            }
            HirExprKind::EnumLiteral { fields, .. } => {
                for (_, e) in fields {
                    Self::collect_return_dims_expr(e, out);
                }
            }
            HirExprKind::Ref(inner)
            | HirExprKind::MutRef(inner)
            | HirExprKind::Deref(inner)
            | HirExprKind::Move(inner)
            | HirExprKind::TryBlock(inner)
            | HirExprKind::Await(inner)
            | HirExprKind::Spawn(inner) => {
                Self::collect_return_dims_expr(inner, out);
            }
            HirExprKind::DerefAssign { target, value } => {
                Self::collect_return_dims_expr(target, out);
                Self::collect_return_dims_expr(value, out);
            }
            HirExprKind::DerefAssignOp { target, value, .. } => {
                Self::collect_return_dims_expr(target, out);
                Self::collect_return_dims_expr(value, out);
            }
            HirExprKind::Yield(inner) => {
                if let Some(e) = inner {
                    Self::collect_return_dims_expr(e, out);
                }
            }
            HirExprKind::Tuple(elements) => {
                for e in elements {
                    Self::collect_return_dims_expr(e, out);
                }
            }
            HirExprKind::FieldAssign { target, value, .. } => {
                Self::collect_return_dims_expr(target, out);
                Self::collect_return_dims_expr(value, out);
            }
            // 叶子节点：无子表达式
            HirExprKind::Literal(_)
            | HirExprKind::Var(_)
            | HirExprKind::InterpolatedString { .. } => {}
        }
    }

    fn collect_return_dims_stmt(stmt: &HirStmt, out: &mut Vec<Vec<Dim>>) {
        match &stmt.kind {
            HirStmtKind::Return(Some(e)) => {
                Self::push_tensor_dims(&e.ty, out);
            }
            HirStmtKind::Return(None) => {}
            HirStmtKind::Expr(e) => Self::collect_return_dims_expr(e, out),
            HirStmtKind::Let { init: Some(e), .. } => Self::collect_return_dims_expr(e, out),
            HirStmtKind::Let { init: None, .. } => {}
            HirStmtKind::While { cond, body } => {
                Self::collect_return_dims_expr(cond, out);
                Self::collect_return_dims_stmt(body, out);
            }
            HirStmtKind::DoWhile { body, cond } => {
                Self::collect_return_dims_stmt(body, out);
                Self::collect_return_dims_expr(cond, out);
            }
            HirStmtKind::For { iter, body, .. } => {
                Self::collect_return_dims_expr(iter, out);
                Self::collect_return_dims_stmt(body, out);
            }
            HirStmtKind::Loop { body } => {
                for s in body {
                    Self::collect_return_dims_stmt(s, out);
                }
            }
            // Break 的值表达式理论上不可能含 return 语句（return 是语句）；
            // 保守起见仍下降以与旧收集器行为对齐。
            HirStmtKind::Break(Some(e)) => Self::collect_return_dims_expr(e, out),
            HirStmtKind::Break(None) | HirStmtKind::Continue => {}
        }
    }

    /// 若类型是含静态信息的 Tensor（任一维度非 Any），push 其 dims。
    /// 过滤条件与旧 `current_fn_return_shapes` 收集器完全一致。
    fn push_tensor_dims(ty: &Type, out: &mut Vec<Vec<Dim>>) {
        if let Type::Tensor { dims, .. } = ty {
            if dims.iter().any(|d| !matches!(d, Dim::Any)) {
                out.push(dims.clone());
            }
        }
    }

    /// 跨函数 shape 求解：多 return 路径的 shape join。
    ///
    /// 收集函数体中所有 return 语句的 shape（以及函数体末尾表达式的 shape），
    /// join 成一个统一的 shape，用于推断函数的精确返回 shape。
    ///
    /// join 规则（逐维）：
    /// - `Any` 与任意 → 取另一侧（Any 不约束）
    /// - `Known(a)` 与 `Known(b)`：a == b → `Known(a)`；a ≠ b → `Any`（不同 Known 降为 Any）
    /// - `Symbol(s)` 与 `Symbol(t)`：s == t → `Symbol(s)`；s ≠ t → `Any`（不同 Symbol 降为 Any）
    /// - `Known` 与 `Symbol` → `Any`（保守降级，不假设 unify）
    ///
    /// 维度数不一致 → 返回 Err（明确的逻辑错误，应报错）。
    pub(super) fn join_return_dims(shapes: &[Vec<Dim>]) -> Result<Vec<Dim>, String> {
        if shapes.is_empty() {
            return Ok(vec![]);
        }
        let first = &shapes[0];
        // 维度数必须一致
        for s in &shapes[1..] {
            if s.len() != first.len() {
                return Err(format!(
                    "维度数不匹配：{} vs {}（无法 join）",
                    first.len(), s.len()
                ));
            }
        }
        // 逐维 join
        let mut result: Vec<Dim> = Vec::with_capacity(first.len());
        for i in 0..first.len() {
            let mut joined: Dim = first[i].clone();
            for s in &shapes[1..] {
                joined = match (&joined, &s[i]) {
                    // Any 与任意 → 取另一侧
                    (Dim::Any, d) | (d, Dim::Any) => d.clone(),
                    // Known 与 Known：相等保留，不等降为 Any
                    (Dim::Known(a), Dim::Known(b)) if a == b => Dim::Known(*a),
                    (Dim::Known(_), Dim::Known(_)) => Dim::Any,
                    // Symbol 与 Symbol：同名保留，不同名降为 Any
                    (Dim::Symbol(a), Dim::Symbol(b)) if a == b => Dim::Symbol(a.clone()),
                    (Dim::Symbol(_), Dim::Symbol(_)) => Dim::Any,
                    // Known 与 Symbol 混合 → 保守 Any（不假设 unify）
                    (Dim::Known(_), Dim::Symbol(_)) | (Dim::Symbol(_), Dim::Known(_)) => Dim::Any,
                };
            }
            result.push(joined);
        }
        Ok(result)
    }

    /// 检查类型注解与实际推断类型是否兼容，并返回合并后的类型。
    ///
    /// 用于：
    /// - `let x: Tensor[f64, 3, 4] = zeros(2, 3)` 报错（注解与实际 shape 不匹配）
    /// - `let x: Tensor[f64, ..] = zeros(3, 4)` 合并为 `Tensor[f64, 3, 4]`（注解 wildcard，用实际 shape）
    /// - 函数返回值：`fn make() -> Tensor[f64, ..] { zeros(3, 4) }` 合并为精确 shape
    ///
    /// 合并规则：
    /// - annotation `Any`（`..`） + actual 精确 → 用 actual（合并）
    /// - annotation 精确 + actual `Any` → 保留 annotation（actual 无法验证）
    /// - annotation 精确 + actual 精确 → 必须相等，否则报错
    /// - annotation `[Any]`（单 Any，视为 wildcard）→ 直接用 actual dims
    /// - actual `[Any]`（单 Any，视为维度未知）→ 保留 annotation，不检查维度数
    ///
    /// `context` 用于错误信息（如 "let 注解" 或 "函数返回值"）。
    pub(super) fn check_and_merge_tensor_shape(
        annot: &Type,
        actual: &Type,
        span: &Span,
        context: &str,
    ) -> TenthResult<Type> {
        // Never 类型：保留 annotation（Never 可统一到任何类型，
        // 但作为返回类型注解时应当被保留——如 `fn exit() -> !`）
        if matches!(annot, Type::Never) {
            return Ok(Type::Never);
        }
        // actual 为 Never（body 永不返回，如纯 return/loop）：
        // 保留 annotation（Never 统一到 annotation 类型）
        if matches!(actual, Type::Never) {
            return Ok(annot.clone());
        }
        // 非 Tensor 类型：检查函数返回值是否匹配（问题11）
        let (annot_dtype, annot_dims) = match annot {
            Type::Tensor { dtype, dims } => (dtype.clone(), dims.clone()),
            _ => {
                // 问题11修复：函数返回值检查——若声明非 Unit 类型但 body 返回 Unit，
                // 报 TypeError（如 `fn f() -> i64 { }` 不会报错的问题）
                if context == "函数返回值"
                    && !matches!(annot, Type::Base(BaseType::Unit) | Type::Unknown)
                    && matches!(actual, Type::Base(BaseType::Unit))
                {
                    return Err(TenthError::TypeError {
                        line: span.line,
                        col: span.col,
                        message: format!(
                            "函数声明返回 {} 但函数体未返回值（body 为空或缺少 return 表达式）",
                            annot
                        ),
                    });
                }
                return Ok(annot.clone());
            }
        };
        let actual_dims: Vec<Dim> = match actual {
            Type::Tensor { dims, .. } => dims.clone(),
            _ => return Ok(annot.clone()),
        };

        // annotation 是单 Any（`[..]` wildcard）：直接用 actual dims
        if annot_dims.len() == 1 && matches!(annot_dims[0], Dim::Any) {
            return Ok(Type::Tensor { dtype: annot_dtype, dims: actual_dims });
        }

        // actual 是单 Any（维度未知）：保留 annotation，不检查维度数
        if actual_dims.len() == 1 && matches!(actual_dims[0], Dim::Any) {
            return Ok(annot.clone());
        }

        // 维度数必须相同
        if annot_dims.len() != actual_dims.len() {
            return Err(TenthError::TypeError {
                line: span.line,
                col: span.col,
                message: format!(
                    "{} shape 维度数不匹配：注解 Tensor{} 与实际 Tensor{}（维度数 {} ≠ {}）",
                    context, fmt_dims(&annot_dims), fmt_dims(&actual_dims),
                    annot_dims.len(), actual_dims.len()
                ),
            });
        }

        // 逐维检查与合并
        let mut merged_dims: Vec<Dim> = Vec::with_capacity(annot_dims.len());
        for (i, (a, b)) in annot_dims.iter().zip(actual_dims.iter()).enumerate() {
            let merged = match (a, b) {
                (Dim::Any, other) | (other, Dim::Any) => (*other).clone(),
                (Dim::Known(x), Dim::Known(y)) if x == y => Dim::Known(*x),
                (Dim::Symbol(s), Dim::Symbol(t)) if s == t => Dim::Symbol(s.clone()),
                // annotation 精确 + actual 精确但不等 → 报错
                (Dim::Known(x), Dim::Known(y)) => {
                    return Err(TenthError::TypeError {
                        line: span.line,
                        col: span.col,
                        message: format!(
                            "{} shape 不匹配：注解 Tensor{} 与实际 Tensor{}（第 {} 维 {} ≠ {}）",
                            context, fmt_dims(&annot_dims), fmt_dims(&actual_dims),
                            i, x, y
                        ),
                    });
                }
                (Dim::Symbol(s), Dim::Symbol(t)) => {
                    return Err(TenthError::TypeError {
                        line: span.line,
                        col: span.col,
                        message: format!(
                            "{} shape 不匹配：注解 Tensor{} 与实际 Tensor{}（第 {} 维符号 {} ≠ {}）",
                            context, fmt_dims(&annot_dims), fmt_dims(&actual_dims),
                            i, s, t
                        ),
                    });
                }
                // annotation Known/Symbol + actual Symbol/Known：保留 annotation（假设兼容）
                (annot_precise, _) => (*annot_precise).clone(),
            };
            merged_dims.push(merged);
        }
        Ok(Type::Tensor { dtype: annot_dtype, dims: merged_dims })
    }

    /// 编译期 shape 检查：跨分支（if/else、match arms）返回 shape 是否兼容。
    ///
    /// 仅在两侧 shape 都含静态信息（Known 或 Symbol，非全 Any）时才检查；
    /// 不兼容时返回 TypeError。任一侧全 Any 则跳过。
    ///
    /// Never 类型（`!`）可以统一到任何类型——若任一分支为 Never，则跳过检查
    /// （如 `if cond { return exit() } else { 42 }` 应当通过）。
    pub(super) fn check_branch_shape_compat(
        then_ty: &Type,
        else_ty: &Type,
        span: &Span,
        context: &str,
    ) -> TenthResult<()> {
        // Never 类型统一到任何类型——不报错
        if matches!(then_ty, Type::Never) || matches!(else_ty, Type::Never) {
            return Ok(());
        }
        if let (Type::Tensor { dims: ldims, .. }, Type::Tensor { dims: rdims, .. }) = (then_ty, else_ty) {
            if has_static_info(ldims) && has_static_info(rdims) {
                if broadcast_shapes(ldims, rdims).is_none() {
                    return Err(TenthError::TypeError {
                        line: span.line,
                        col: span.col,
                        message: format!(
                            "{} 分支 shape 不兼容：then Tensor{} 与 else Tensor{}（无法广播）",
                            context, fmt_dims(ldims), fmt_dims(rdims)
                        ),
                    });
                }
            }
        }
        Ok(())
    }

    /// 判断 HirExpr 是否静态可判定为零（lossy lattice M1 spike + M3 shape 协同）。
    ///
    /// 防误报原则（宁可漏报，不可误报）：只认**编译期可静态确定**的零：
    /// - 直接字面量零与一元负号包裹的零字面量（`-0` / `-0.0`，作为除数同样产生
    ///   ±inf/NaN，且静态可判定）（M1）
    /// - **张量字面量全零**（`[[0.0, 0.0], [0.0, 0.0]]`）：所有元素静态零
    ///   → 张量级判零（M3，shape/值均静态已知）
    /// - **`zeros`/`zeros_f32`/`zeros_f16`/`zeros_bf16` 构造**：内置全零张量构造
    ///   （shape 已知与否不影响——`zeros` 语义确定返回全零）（M3）
    ///
    /// 变量、非 zeros 的函数调用、算术表达式一律不判定——例如 `let y = 0.0; x / y`
    /// 不报（可能被后续赋值改变；即便当前字面量是 0，也只算"运行时值"，不是编译期常量）。
    /// **部分零张量**（如 `[[0.0, 1.0]]`）不判定（逐元素粒度过度，先做张量级粒度，
    /// 漏报接受——设计文档阶段 2b §6 M3）。
    fn is_statically_zero(expr: &HirExpr) -> bool {
        match &expr.kind {
            HirExprKind::Literal(Literal::Int(n, _)) => *n == 0,
            HirExprKind::Literal(Literal::Float(n, _)) => *n == 0.0,
            HirExprKind::Unary { op: UnaryOp::Neg, expr: inner, .. } => Self::is_statically_zero(inner),
            // M3：张量字面量全零 → 张量级静态零
            HirExprKind::TensorLiteral { data, .. } => {
                !data.is_empty()
                    && data.iter().all(|row| row.iter().all(|el| Self::is_statically_zero(el)))
            }
            // M3：内置全零张量构造（zeros/zeros_f32/zeros_f16/zeros_bf16），
            // 以及 `tensor[[...]]` 字面量调用（参数是张量字面量 → 递归判定全零）
            HirExprKind::Call { func, args, .. } | HirExprKind::GenericCall { func, args, .. } => {
                if let HirExprKind::Var(name) = &func.kind {
                    match name.as_str() {
                        "zeros" | "zeros_f32" | "zeros_f16" | "zeros_bf16" => true,
                        // `tensor[[0.0, 0.0], ...]`：parser 解析为 tensor 调用 + 张量字面量参数
                        "tensor" => args.iter().any(|a| Self::is_statically_zero(a)),
                        _ => false,
                    }
                } else {
                    false
                }
            }
            _ => false,
        }
    }

    /// 判断 HirExpr 是否静态可判定为**非零**（lossy lattice M3：除零精确化的正向豁免）。
    ///
    /// 用途：污点分析（`taint.rs` `op_effect`）中，除法/取模的除数**静态非零** →
    /// 精确地**不标 PossibleNaN**。与 M1 的"静态零 → 硬错误"互补后，除数的静态
    /// 判定完整：静态零 → 硬错误；静态非零 → 无污点；值未知 → 不 speculate。
    ///
    /// 只认编译期可静态确定的非零：
    /// - 非零字面量（`2.0` / `1e3` / `-3.0`）与一元负号包裹的非零
    /// - 张量字面量**所有元素**非零（`[[1.0, 2.0], [3.0, 4.0]]`）
    /// - `ones`/`ones_f32`/`ones_f16`/`ones_bf16` 构造（内置全一张量构造）
    ///
    /// 防误报：本判定只用于**豁免** PossibleNaN（不引入任何新报错），即便误判
    /// （把可能为零者当非零）也只漏掉污点标记，不产生误报。变量、shape 已知但
    /// 值未知的张量（如 `Tensor[f64, M, K]` 参数）一律不判定（不 speculate）。
    pub(super) fn is_statically_nonzero(expr: &HirExpr) -> bool {
        match &expr.kind {
            HirExprKind::Literal(Literal::Int(n, _)) => *n != 0,
            HirExprKind::Literal(Literal::Float(n, _)) => *n != 0.0,
            HirExprKind::Unary { op: UnaryOp::Neg, expr: inner, .. } => Self::is_statically_nonzero(inner),
            HirExprKind::TensorLiteral { data, .. } => {
                !data.is_empty()
                    && data.iter().all(|row| row.iter().all(|el| Self::is_statically_nonzero(el)))
            }
            HirExprKind::Call { func, args, .. } | HirExprKind::GenericCall { func, args, .. } => {
                if let HirExprKind::Var(name) = &func.kind {
                    match name.as_str() {
                        "ones" | "ones_f32" | "ones_f16" | "ones_bf16" => true,
                        // `tensor[[1.0, 2.0], ...]`：parser 解析为 tensor 调用 + 张量字面量参数
                        "tensor" => args.iter().any(|a| Self::is_statically_nonzero(a)),
                        _ => false,
                    }
                } else {
                    false
                }
            }
            _ => false,
        }
    }

    /// 编译期零除数检测（lossy lattice M1 spike，最小落地样例）。
    ///
    /// 当 `/` 或 `%` 的右操作数**静态可判定为零**时编译期报错，比现状提前：
    /// - 现状（基线已实测）：浮点 `1.0 / 0.0` → **静默产生 inf**（静默算错）；
    ///   整数 `10 / 0` → 运行时"整数除零"错误。
    /// - 改进后：两类都在 lower 阶段报 `TypeError`，带行列号。
    ///
    /// 触发条件（严格限定，防误报）：op ∈ {Div, Mod} 且右操作数是字面量零
    /// （含 `-0` / `-0.0`）。不覆盖变量除数、函数调用返回值——那些属于
    /// PossibleNaN 传播（M2），而非本 spike 的静态确定范围。
    pub(super) fn check_binary_static_divzero(
        op: &ast::BinOp,
        right: &HirExpr,
        span: &Span,
    ) -> TenthResult<()> {
        use ast::BinOp;
        if matches!(op, BinOp::Div | BinOp::Mod) && Self::is_statically_zero(right) {
            return Err(TenthError::TypeError {
                line: span.line,
                col: span.col,
                message: format!(
                    "编译期检测到除数为零（右操作数为字面量/静态可判定零）：{} 0 的结果是 inf/NaN（浮点）或触发运行时除零错误（整数），无法作为确定正确的值使用。若确需此语义，请显式检查除数（如 if y == 0.0 则分支处理）；lossy 放行属 M2 规划",
                    binop_name(op)
                ),
            });
        }
        Ok(())
    }

    /// 编译期 shape 检查：二元运算（+、-、*、/、%）两侧 Tensor shape 是否兼容。
    ///
    /// 仅在两侧 shape 都含静态信息（Known 或 Symbol，非全 Any）时才检查；
    /// 不兼容时返回 TypeError。任一侧全 Any（运行时构造的默认情况）则跳过。
    pub(super) fn check_binary_shape_compat(
        op: &ast::BinOp,
        l: &Type,
        r: &Type,
        span: &Span,
    ) -> TenthResult<()> {
        if let (Type::Tensor { dims: ldims, .. }, Type::Tensor { dims: rdims, .. }) = (l, r) {
            if has_static_info(ldims) && has_static_info(rdims) {
                if broadcast_shapes(ldims, rdims).is_none() {
                    return Err(TenthError::TypeError {
                        line: span.line,
                        col: span.col,
                        message: format!(
                            "编译期 shape 不兼容：Tensor{} {} Tensor{}（无法广播）",
                            fmt_dims(ldims), binop_name(op), fmt_dims(rdims)
                        ),
                    });
                }
            }
        }
        Ok(())
    }

    /// 编译期 shape 检查：方法调用的 shape 约束。
    ///
    /// 当前覆盖：
    /// - `matmul`：2D (M, K) @ (K, N)，内侧 K 必须相等
    ///   - Known vs Known：数值不等则报错
    ///   - Symbol vs Symbol：名字不等则报错（同名视为同一维度）
    ///   - Symbol vs Known：保守通过（unify 留待 Phase 3）
    pub(super) fn check_method_shape(
        receiver: &Type,
        method: &str,
        args: &[HirExpr],
        span: &Span,
    ) -> TenthResult<()> {
        if let Type::Tensor { dims: ldims, .. } = receiver {
            match method {
                "matmul" => {
                    if ldims.len() == 2 {
                        if let Some(arg) = args.first() {
                            if let Type::Tensor { dims: rdims, .. } = &arg.ty {
                                if rdims.len() == 2 {
                                    let lk = &ldims[1];
                                    let rk = &rdims[0];
                                    let mismatch = match (lk, rk) {
                                        (Dim::Known(a), Dim::Known(b)) => a != b,
                                        (Dim::Symbol(a), Dim::Symbol(b)) => a != b,
                                        // Symbol vs Known 或任一 Any：保守通过
                                        _ => false,
                                    };
                                    if mismatch {
                                        return Err(TenthError::TypeError {
                                            line: span.line,
                                            col: span.col,
                                            message: format!(
                                                "编译期 matmul shape 不兼容：{} @ {}（内侧维度 {} ≠ {} 必须相等）",
                                                fmt_dims(ldims), fmt_dims(rdims), fmt_dim(lk), fmt_dim(rk)
                                            ),
                                        });
                                    }
                                    // 护城河 A Phase 1：反向 shape 验证
                                    // matmul 梯度 shape 与输入 shape 一致（d_a=M,K; d_b=K,N），当前总是通过；
                                    // 调用统一入口以便 Phase 2 跨算子传播扩展
                                    let fwd_in_shapes = vec![ldims.to_vec(), rdims.to_vec()];
                                    let fwd_out_shape = vec![ldims[0].clone(), rdims[1].clone()];
                                    super::backward_shapes::check_backward_shape_compat(
                                        "matmul", &fwd_in_shapes, &fwd_out_shape, span,
                                    )?;
                                } else if rdims.len() != 0 {
                                    // 非 2D 参数：运行时只支持 2D，但这里不报错（让运行时报）
                                }
                            }
                        }
                    }
                }
                "bmm" => {
                    // 3D bmm: (B, M, K) @ (B, K, N) — batch B 和内侧 K 必须相等
                    // 非 3D 不报错（让运行时处理 "requires 3D tensors"）
                    if ldims.len() == 3 {
                        if let Some(arg) = args.first() {
                            if let Type::Tensor { dims: rdims, .. } = &arg.ty {
                                if rdims.len() == 3 {
                                    // batch 维度：ldims[0] vs rdims[0]
                                    let lb = &ldims[0];
                                    let rb = &rdims[0];
                                    let batch_mismatch = match (lb, rb) {
                                        (Dim::Known(a), Dim::Known(b)) => a != b,
                                        (Dim::Symbol(a), Dim::Symbol(b)) => a != b,
                                        _ => false,
                                    };
                                    if batch_mismatch {
                                        return Err(TenthError::TypeError {
                                            line: span.line,
                                            col: span.col,
                                            message: format!(
                                                "编译期 bmm shape 不兼容：{} @ {}（batch 维度 {} ≠ {} 必须相等）",
                                                fmt_dims(ldims), fmt_dims(rdims), fmt_dim(lb), fmt_dim(rb)
                                            ),
                                        });
                                    }
                                    // 内侧 K：ldims[2] vs rdims[1]
                                    let lk = &ldims[2];
                                    let rk = &rdims[1];
                                    let inner_mismatch = match (lk, rk) {
                                        (Dim::Known(a), Dim::Known(b)) => a != b,
                                        (Dim::Symbol(a), Dim::Symbol(b)) => a != b,
                                        _ => false,
                                    };
                                    if inner_mismatch {
                                        return Err(TenthError::TypeError {
                                            line: span.line,
                                            col: span.col,
                                            message: format!(
                                                "编译期 bmm shape 不兼容：{} @ {}（inner 维度 {} ≠ {} 必须相等）",
                                                fmt_dims(ldims), fmt_dims(rdims), fmt_dim(lk), fmt_dim(rk)
                                            ),
                                        });
                                    }
                                    // 护城河 A Phase 1：反向 shape 验证
                                    // bmm 梯度 shape 与输入 shape 一致（d_a=B,M,K; d_b=B,K,N），当前总是通过
                                    let fwd_in_shapes = vec![ldims.to_vec(), rdims.to_vec()];
                                    let fwd_out_shape = vec![
                                        ldims[0].clone(),
                                        ldims[1].clone(),
                                        rdims[2].clone(),
                                    ];
                                    super::backward_shapes::check_backward_shape_compat(
                                        "bmm", &fwd_in_shapes, &fwd_out_shape, span,
                                    )?;
                                }
                                // 非 3D 参数：运行时只支持 3D，但这里不报错（让运行时报 "requires 3D"）
                            }
                        }
                    }
                }
                "reshape" | "view" => {
                    // 护城河 A Phase 1：reshape 反向 shape 验证
                    // d_input = grad.reshape(input_shape)，需 output numel == input numel
                    Self::check_reshape_shape(receiver, args, span)?;
                }
                _ => {}
            }
        }
        Ok(())
    }

    /// 编译期 shape 检查：reshape/view 方法的元素数一致性验证。
    ///
    /// reshape 反向：d_input = grad.reshape(input_shape)，需 output numel == input numel。
    /// 仅在两侧 shape 全 Known 时检查；含 Any/Symbol 时跳过（保守）。
    pub(super) fn check_reshape_shape(
        receiver: &Type,
        args: &[HirExpr],
        span: &Span,
    ) -> TenthResult<()> {
        if let Type::Tensor { dims: ldims, .. } = receiver {
            let out_shape = Self::shape_from_int_args(args);
            let fwd_in_shapes = vec![ldims.to_vec()];
            super::backward_shapes::check_backward_shape_compat(
                "reshape", &fwd_in_shapes, &out_shape, span,
            )?;
        }
        Ok(())
    }

    /// 编译期 shape 检查：cross_entropy native 函数的 logits/target shape 兼容性。
    ///
    /// 规则：
    /// - logits shape 应为 [B, V] 或更高维（至少 2D）
    /// - target shape 应为 [B] 或 [B, V]（与 logits 的 batch 维对齐）
    /// - 不兼容时返回 `TenthError::TypeError`
    ///
    /// 保守策略：任一 shape 全 Any 时跳过。
    /// target shape 的核心兼容性检查委托给
    /// `backward_shapes::check_backward_shape_compat("cross_entropy", ...)`。
    pub(super) fn check_cross_entropy_shape(args: &[HirExpr], span: &Span) -> TenthResult<()> {
        if args.len() < 2 {
            return Ok(());
        }
        let (logits_dims, target_dims) = match (&args[0].ty, &args[1].ty) {
            (Type::Tensor { dims: ld, .. }, Type::Tensor { dims: td, .. }) => {
                (ld.to_vec(), td.to_vec())
            }
            _ => return Ok(()), // 非 Tensor，保守跳过
        };
        // 任一全 Any 时跳过
        if !has_static_info(&logits_dims) || !has_static_info(&target_dims) {
            return Ok(());
        }
        // logits 应为至少 2D [B, V]
        if logits_dims.len() < 2 {
            return Err(TenthError::TypeError {
                line: span.line,
                col: span.col,
                message: format!(
                    "cross_entropy logits 期望至少 2D [B, V]，实际 {}",
                    fmt_dims(&logits_dims)
                ),
            });
        }
        // 委托给 backward_shapes 检查 target shape 兼容性
        // cross_entropy 不依赖 output shape（output 是标量），传空 vec
        super::backward_shapes::check_backward_shape_compat(
            "cross_entropy",
            &[logits_dims, target_dims],
            &[],
            span,
        )
    }

    // ── 编译期内存/算力预估（护城河 D）──────────────────────────────────────

    /// 内存预估：对大 tensor 发 warning。
    /// 阈值：1GB（可调整）。仅在 shape 全 Known 时预估。
    pub(super) fn emit_memory_estimate(&mut self, ty: &Type, span: &Span, context: &str) {
        if let Some(bytes) = ty.static_bytes() {
            const GB: u64 = 1024 * 1024 * 1024;
            if bytes >= GB {
                let msg = format!(
                    "{} 创建约 {:.2} GB 的 tensor（编译期预估，可能触发 OOM）",
                    context,
                    bytes as f64 / GB as f64
                );
                self.warnings.push(TenthWarning::new(span.line, span.col, msg));
            }
        }
    }

    /// matmul FLOPs 预估：(M,K)@(K,N) → M*K*N 乘加。
    /// 阈值：1 GFLOP（10^9 乘加）。仅在两侧 shape 都 2D Known 时预估。
    pub(super) fn emit_matmul_flop_estimate(
        &mut self,
        recv_ty: &Type,
        arg_ty: &Type,
        span: &Span,
    ) {
        if let (Type::Tensor { dims: rdims, .. }, Type::Tensor { dims: adims, .. }) = (recv_ty, arg_ty) {
            if rdims.len() == 2 && adims.len() == 2 {
                if let (Dim::Known(m), Dim::Known(k1), Dim::Known(k2), Dim::Known(n)) =
                    (&rdims[0], &rdims[1], &adims[0], &adims[1])
                {
                    if k1 == k2 {
                        if let Some(flops) = (*m as u64)
                            .checked_mul(*k1 as u64)
                            .and_then(|x| x.checked_mul(*n as u64))
                        {
                            const GFLOP: u64 = 1_000_000_000;
                            if flops >= GFLOP {
                                let msg = format!(
                                    "matmul 约 {:.2} GFLOPs（{}×{} @ {}×{}，编译期预估）",
                                    flops as f64 / GFLOP as f64,
                                    m, k1, k2, n
                                );
                                self.warnings.push(TenthWarning::new(span.line, span.col, msg));
                            }
                        }
                    }
                }
            }
        }
    }

    /// bmm FLOPs 预估：(B,M,K)@(B,K,N) → B*M*K*N 乘加（每乘加算 2 FLOP）。
    /// 阈值：1 GFLOP（10^9）。仅在两侧 shape 都 3D Known 且 B/K 匹配时预估。
    pub(super) fn emit_bmm_flop_estimate(
        &mut self,
        recv_ty: &Type,
        arg_ty: &Type,
        span: &Span,
    ) {
        if let (Type::Tensor { dims: rdims, .. }, Type::Tensor { dims: adims, .. }) = (recv_ty, arg_ty) {
            if rdims.len() == 3 && adims.len() == 3 {
                // batch 必须匹配：rdims[0] == adims[0]
                // 内侧 K 必须匹配：rdims[2] == adims[1]
                if let (Dim::Known(b1), Dim::Known(m), Dim::Known(k1), Dim::Known(b2), Dim::Known(k2), Dim::Known(n)) =
                    (&rdims[0], &rdims[1], &rdims[2], &adims[0], &adims[1], &adims[2])
                {
                    if b1 == b2 && k1 == k2 {
                        // FLOPs = B * M * K * N * 2（乘加各算一次）
                        if let Some(mul_add) = (*b1 as u64)
                            .checked_mul(*m as u64)
                            .and_then(|x| x.checked_mul(*k1 as u64))
                            .and_then(|x| x.checked_mul(*n as u64))
                        {
                            let flops = mul_add.saturating_mul(2);
                            const GFLOP: u64 = 1_000_000_000;
                            if flops >= GFLOP {
                                let msg = format!(
                                    "bmm 约 {:.2} GFLOPs（{}×{}×{} @ {}×{}×{}，编译期预估）",
                                    flops as f64 / GFLOP as f64,
                                    b1, m, k1, b2, k2, n
                                );
                                self.warnings.push(TenthWarning::new(span.line, span.col, msg));
                            }
                        }
                    }
                }
            }
        }
    }
}
