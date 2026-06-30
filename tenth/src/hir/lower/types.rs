use crate::error::{TenthError, TenthResult};
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
fn has_static_info(dims: &[Dim]) -> bool {
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
fn fmt_dims(dims: &[Dim]) -> String {
    let parts: Vec<String> = dims.iter().map(|d| match d {
        Dim::Known(n) => n.to_string(),
        Dim::Symbol(s) => s.clone(),
        Dim::Any => "..".to_string(),
    }).collect();
    format!("[{}]", parts.join(", "))
}

/// 格式化单个维度。
fn fmt_dim(d: &Dim) -> String {
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
        if let HirExprKind::Literal(Literal::Int(n)) = &a.kind {
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
            HirExprKind::Literal(Literal::Int(n)) => out.push(*n),
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
            Type::Array(inner) => self.resolve_struct_type((**inner).clone()),
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
                if let Some((params, ret)) = self.scope.lookup_fn(name) {
                    if params.len() != args.len() {
                        let expected: Vec<String> = params.iter()
                            .map(|(n, t)| format!("{}: {}", n, t))
                            .collect();
                        let got: Vec<String> = args.iter()
                            .map(|a| format!("{}", a.ty))
                            .collect();
                        return Err(TenthError::TypeError {
                            line: span.line,
                            col: span.col,
                            message: format!(
                                "函数 '{}' 期望 {} 个参数 [{}]，但传入了 {} 个 [{}]",
                                name, params.len(), expected.join(", "), args.len(), got.join(", ")
                            ),
                        });
                    }
                    return Ok(self.resolve_struct_type(ret));
                }
                self.resolve_builtin(name, args, span)
            }
            _ => Ok(Type::Unknown),
        }
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
                                HirExprKind::Literal(Literal::Int(n)) => Some(*n),
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
                    // transpose：2D 时反转两维；其他情况保持原 shape（运行时按行列互换）
                    "transpose" => {
                        if dims.len() == 2 {
                            Type::Tensor { dtype: dtype.clone(), dims: vec![dims[1].clone(), dims[0].clone()] }
                        } else {
                            Type::Tensor { dtype: dtype.clone(), dims: dims.clone() }
                        }
                    }
                    "to_vec" => Type::Array(Box::new(dtype.as_ref().clone())),
                    "len" | "size" | "dim" => Type::Base(BaseType::I64),
                    "shape" => Type::Array(Box::new(Type::Base(BaseType::I64))),
                    _ => Type::Unknown,
                }
            }
            Type::Base(BaseType::Str) => match method {
                "len" => Type::Base(BaseType::I64),
                "contains" | "starts_with" | "ends_with" => Type::bool_(),
                "trim" | "to_lowercase" | "to_uppercase" => Type::str_(),
                "split" | "lines" => Type::Array(Box::new(Type::str_())),
                "replace" => Type::str_(),
                "parse_int" | "parse_float" => Type::Enum("Option".to_string()),
                "chars" => Type::Array(Box::new(Type::Base(BaseType::Char))),
                _ => Type::Unknown,
            },
            Type::Array(inner) => match method {
                "len" => Type::Base(BaseType::I64),
                "push" => Type::unit(),
                "pop" => Type::Enum("Option".to_string()),
                "get" => Type::Enum("Option".to_string()),
                "map" | "filter" => Type::Array(inner.clone()),
                "is_empty" => Type::bool_(),
                "iter" => Type::Unknown,
                _ => Type::Unknown,
            },
            _ => match method {
                "len" => Type::Base(BaseType::I64),
                "push" => Type::unit(),
                "get" => Type::Unknown,
                _ => Type::Unknown,
            },
        }
    }

    pub(super) fn resolve_builtin(&self, name: &str, args: &[HirExpr], _span: &Span) -> TenthResult<Type> {
        match name {
            "println" | "eprintln" => Ok(Type::unit()),
            // Tensor 构造函数：dtype 从参数推断（若无 f32 线索则默认 F64）
            "tensor" => Ok(Type::tensor(Self::infer_tensor_dtype(args), Self::shape_from_int_args(args))),
            "rand" | "randn" => Ok(Type::tensor(Self::infer_tensor_dtype(args), Self::shape_from_int_args(args))),
            "randn_f32" => Ok(Type::tensor(BaseType::F32, vec![Dim::Any])),
            // Phase 5.5：补全 f32 构造函数 native 注册
            "rand_f32" | "zeros_f32" | "ones_f32" => Ok(Type::tensor(BaseType::F32, vec![Dim::Any])),
            "read_file" => Ok(Type::str_()),
            "str_at" => Ok(Type::str_()),
            "write_file" | "write_bytes" => Ok(Type::unit()),
            "Vec::new" => Ok(Type::Array(Box::new(Type::Unknown))),
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
            "start_grad" | "new_grad" | "stop_grad" | "param" => Ok(Type::tensor(Self::infer_tensor_dtype(args), vec![Dim::Any])),
            "backward" => Ok(Type::unit()),
            "grad" | "zero_grad" => Ok(Type::Unknown),
            "path_join" => Ok(Type::str_()),
            "path_exists" | "path_is_file" | "path_is_dir" => Ok(Type::bool_()),
            "mkdir" => Ok(Type::unit()),
            "list_dir" => Ok(Type::Array(Box::new(Type::str_()))),
            "file_size" => Ok(Type::Base(BaseType::I64)),
            "remove_file" | "copy_file" => Ok(Type::unit()),
            "lexer_new" | "lexer_tokenize" | "parse_program" | "lower_program" | "compile_to_wasm" | "compile_program" => Ok(Type::Unknown),
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

    /// 从构造函数的字面量参数推断 shape。
    /// 若所有参数都是 IntLiteral（如 `zeros(3, 4)`），返回 `[Known(3), Known(4)]`；
    /// 任一参数非字面量（如 `zeros(n)`），返回 `[Any]`（运行时才能确定）。
    pub(super) fn shape_from_int_args(args: &[HirExpr]) -> Vec<Dim> {
        if args.is_empty() {
            return vec![Dim::Any];
        }
        let mut dims: Vec<Dim> = Vec::with_capacity(args.len());
        for a in args {
            match &a.kind {
                HirExprKind::Literal(Literal::Int(n)) => dims.push(Dim::Known(*n)),
                _ => return vec![Dim::Any],
            }
        }
        dims
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
                                } else if rdims.len() != 0 {
                                    // 非 2D 参数：运行时只支持 2D，但这里不报错（让运行时报）
                                }
                            }
                        }
                    }
                }
                _ => {}
            }
        }
        Ok(())
    }
}
