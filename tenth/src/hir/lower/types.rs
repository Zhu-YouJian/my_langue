use crate::error::{TenthError, TenthResult};
use crate::lexer::token::Span;
use crate::parser::ast as ast;
use crate::hir::hir::*;
use crate::hir::types::*;
use super::Lowerer;

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
                    (Type::Tensor { dtype: ld, .. }, Type::Tensor { dtype: rd, .. }) => {
                        let lb = match ld.as_ref() { Type::Base(b) => *b, _ => BaseType::F64 };
                        let rb = match rd.as_ref() { Type::Base(b) => *b, _ => BaseType::F64 };
                        let promoted = Self::promote_float_dtype(lb, rb);
                        Type::tensor(promoted, vec![Dim::Any])
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
                    "sum" => {
                        if _args.iter().any(|a| matches!(&a.kind, HirExprKind::Var(_))) {
                            Type::Tensor { dtype: dtype.clone(), dims: dims.clone() }
                        } else {
                            dtype.as_ref().clone()
                        }
                    }
                    "mean" | "max" | "min" => dtype.as_ref().clone(),
                    "reshape" | "view" => Type::Tensor { dtype: dtype.clone(), dims: vec![Dim::Any] },
                    "flatten" => Type::Tensor { dtype: dtype.clone(), dims: vec![Dim::Any] },
                    "abs" | "sqrt" | "exp" | "log" | "relu" |
                    "sigmoid" | "tanh" | "softmax" |
                    "transpose" | "permute" => {
                        Type::Tensor { dtype: dtype.clone(), dims: dims.clone() }
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
            "tensor" => Ok(Type::tensor(Self::infer_tensor_dtype(args), vec![Dim::Any])),
            "rand" | "randn" => Ok(Type::tensor(Self::infer_tensor_dtype(args), vec![Dim::Any])),
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
            "tensor_from_vec" => Ok(Type::tensor(Self::infer_tensor_dtype(args), vec![Dim::Any])),
            "zeros" | "ones" => Ok(Type::tensor(Self::infer_tensor_dtype(args), vec![Dim::Any])),
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
}
