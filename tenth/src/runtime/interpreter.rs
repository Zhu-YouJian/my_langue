use std::collections::HashMap;
use std::rc::Rc;
use std::cell::RefCell;
use crate::error::{TenthError, TenthResult};
use crate::hir::hir::*;
use crate::hir::types::{BaseType, Dim, Type};
use super::value::Value;
use super::tensor::Tensor;
use super::limits::RuntimeLimits;

pub struct Interpreter {
    pub variables: HashMap<String, Value>,
    functions: Vec<HirFnDef>,
    generic_funcs: HashMap<String, HirFnDef>,
    methods: HashMap<String, HashMap<String, HirFnDef>>,
    modules: HashMap<String, HirProgram>,
    trait_impls: HashMap<String, HashMap<String, HashMap<String, HirFnDef>>>,
    /// Resource limits — when set, allocations are checked against caps.
    pub limits: Option<RuntimeLimits>,
}

impl Interpreter {
    pub fn new(program: &HirProgram) -> Self {
        Interpreter {
            variables: HashMap::new(),
            functions: program.functions.clone(),
            generic_funcs: HashMap::new(),
            methods: program.methods.clone(),
            modules: program.modules.clone(),
            trait_impls: program.trait_impls.clone(),
            limits: None,
        }
    }

    /// Create an interpreter with resource limits enforced.
    pub fn with_limits(program: &HirProgram, limits: RuntimeLimits) -> Self {
        let mut interp = Interpreter::new(program);
        interp.limits = Some(limits);
        interp
    }

    pub fn execute_program(&mut self, program: &HirProgram) -> TenthResult<Option<Value>> {
        self.variables.insert(
            "tensor".to_string(),
            Value::FnRef {
                name: "tensor".to_string(),
                params: vec![("data".to_string(), Type::Unknown)],
                return_type: Type::Tensor {
                    dtype: BaseType::F64,
                    dims: vec![Dim::Any],
                },
            },
        );

        for func in &program.functions {
            let params = func.params.clone();
            let ret = func.return_type.clone();
            self.variables.insert(
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
                        self.functions.push(fn_def.clone());
                        self.variables.insert(
                            alias.clone(),
                            Value::FnRef {
                                name: alias.clone(),
                                params: fn_def.params.clone(),
                                return_type: fn_def.return_type.clone(),
                            },
                        );
                    }
                }
            }
        }

        if let Some(ref expr) = program.main_expr {
            self.eval_expr(expr)
        } else if let Some(main_fn) = self.functions.iter().find(|f| f.name == "main") {
            let body = main_fn.body.clone();
            self.eval_expr(&body)
        } else {
            Ok(None)
        }
    }

    fn resolve_var(&self, name: &str) -> Option<Value> {
        match self.variables.get(name) {
            Some(Value::Shared(rc)) => Some(rc.borrow().clone()),
            Some(Value::Moved) => None,
            other => other.cloned(),
        }
    }

    fn set_var(&mut self, name: String, val: Value) {
        // Guard: check variable count before inserting a new one
        if let Some(ref limits) = self.limits {
            if !self.variables.contains_key(&name) {
                if let Err(msg) = limits.guard_vars(self.variables.len()) {
                    // In mem-strict mode, this would panic; in default, warn
                    eprintln!("[limits] variable limit: {}", msg);
                    if cfg!(feature = "mem-strict") {
                        panic!("variable limit exceeded: {}", msg);
                    }
                    // Otherwise, proceed but warn
                }
            }
        }
        match self.variables.get(&name) {
            Some(Value::Shared(rc)) => {
                *rc.borrow_mut() = val;
            }
            _ => {
                self.variables.insert(name, val);
            }
        }
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

    fn eval_expr(&mut self, expr: &HirExpr) -> TenthResult<Option<Value>> {
        use HirExprKind;

        match &expr.kind {
            HirExprKind::Literal(lit) => {
                Ok(Some(match lit {
                    Literal::Int(n) => Value::Int(*n),
                    Literal::Float(n) => Value::Float(*n),
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
                if matches!(self.variables.get(name), Some(Value::Moved)) {
                    return Err(TenthError::RuntimeError {
                        message: format!("use of moved value '{}'", name),
                    });
                }
                self.resolve_var(name)
                    .or_else(|| {
                        match name.as_str() {
                            "println" | "eprintln" | "tensor" | "rand" | "randn"
                            | "read_file" | "write_file" | "Vec::new" | "HashMap::new" => {
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
                        message: format!("undefined variable '{}'", name),
                    })
                    .map(Some)
            }

            HirExprKind::Binary { op, left, right, .. } => {
                let l = self.eval_expr(left)?.ok_or_else(|| TenthError::RuntimeError {
                    message: "left operand is void".into(),
                })?;
                let r = self.eval_expr(right)?.ok_or_else(|| TenthError::RuntimeError {
                    message: "right operand is void".into(),
                })?;
                self.eval_binary(op, &l, &r).map(Some)
            }

            HirExprKind::Unary { op, expr: inner, .. } => {
                let val = self.eval_expr(inner)?.ok_or_else(|| TenthError::RuntimeError {
                    message: "unary operand is void".into(),
                })?;
                self.eval_unary(op, &val).map(Some)
            }

            HirExprKind::Call { func, args, .. } => {
                let f = self.eval_expr(func)?.ok_or_else(|| TenthError::RuntimeError {
                    message: "function value is void".into(),
                })?;

                let mut arg_values = Vec::new();
                for a in args {
                    arg_values.push(self.eval_expr(a)?.ok_or_else(|| TenthError::RuntimeError {
                        message: "argument is void".into(),
                    })?);
                }

                self.eval_call(&f, &arg_values, &expr.span)
            }

            HirExprKind::GenericCall { func, generics, args, .. } => {
                let func_name = match &func.kind {
                    HirExprKind::Var(name) => name.clone(),
                    _ => {
                        return Err(TenthError::RuntimeError {
                            message: "generic call target must be a named function".into(),
                        });
                    }
                };

                let template = self.generic_funcs.get(&func_name)
                    .ok_or_else(|| TenthError::RuntimeError {
                        message: format!("undefined generic function '{}'", func_name),
                    })?
                    .clone();

                let mut type_map: HashMap<String, Type> = HashMap::new();
                for (i, gen_name) in template.generics.iter().enumerate() {
                    type_map.insert(gen_name.clone(), generics.get(i).cloned().unwrap_or(Type::Unknown));
                }

                let mut arg_values = Vec::new();
                for a in args {
                    arg_values.push(self.eval_expr(a)?.ok_or_else(|| TenthError::RuntimeError {
                        message: "argument is void".into(),
                    })?);
                }

                let saved: HashMap<String, Value> = template.params.iter()
                    .filter_map(|(n, _)| self.variables.get(n).cloned().map(|v| (n.clone(), v)))
                    .collect();

                for ((pname, _), arg) in template.params.iter().zip(arg_values.iter()) {
                    self.variables.insert(pname.clone(), arg.clone());
                }

                let result = self.eval_expr(&template.body);

                for (n, v) in saved {
                    self.variables.insert(n, v);
                }

                result
            }

            HirExprKind::MethodCall { receiver, method, args, .. } => {
                let recv = self.eval_expr(receiver)?.ok_or_else(|| TenthError::RuntimeError {
                    message: "receiver is void".into(),
                })?;

                let mut arg_values = Vec::new();
                for a in args {
                    arg_values.push(self.eval_expr(a)?.ok_or_else(|| TenthError::RuntimeError {
                        message: "method argument is void".into(),
                    })?);
                }

                self.eval_method_call(&recv, method, &arg_values)
            }

            HirExprKind::Index { target, indices } => {
                let t = self.eval_expr(target)?.ok_or_else(|| TenthError::RuntimeError {
                    message: "index target is void".into(),
                })?;
                self.eval_index(&t, indices).map(Some)
            }

            HirExprKind::Field { target, field } => {
                let t = self.eval_expr(target)?.ok_or_else(|| TenthError::RuntimeError {
                    message: "field access target is void".into(),
                })?;
                match &t {
                    Value::Struct { fields, .. } => {
                        for (fname, fval) in fields {
                            if fname == field {
                                return Ok(Some(fval.clone()));
                            }
                        }
                        Err(TenthError::RuntimeError {
                            message: format!("struct has no field '{}'", field),
                        })
                    }
                    Value::Enum { fields, .. } => {
                        for (fname, fval) in fields {
                            if fname == field {
                                return Ok(Some(fval.clone()));
                            }
                        }
                        Err(TenthError::RuntimeError {
                            message: format!("enum variant has no field '{}'", field),
                        })
                    }
                    _ => Err(TenthError::RuntimeError {
                        message: format!("cannot access field '{}' on {:?}", field, t),
                    }),
                }
            }

            HirExprKind::ArrayLiteral { elements, .. } => {
                let mut vals = Vec::new();
                for elem in elements {
                    let v = self.eval_expr(elem)?.ok_or_else(|| TenthError::RuntimeError {
                        message: "array element is void".into(),
                    })?;
                    vals.push(v);
                }
                Ok(Some(Value::Vec(Rc::new(RefCell::new(vals)))))
            }

            HirExprKind::TensorLiteral { data, .. } => {
                let mut rows: Vec<Vec<f64>> = Vec::new();
                for row in data {
                    let mut row_vals = Vec::new();
                    for elem in row {
                        let v = self.eval_expr(elem)?.ok_or_else(|| TenthError::RuntimeError {
                            message: "tensor element is void".into(),
                        })?;
                        row_vals.push(v.as_float().unwrap_or(0.0));
                    }
                    rows.push(row_vals);
                }
                let nrows = rows.len();
                let ncols = rows.first().map(|r| r.len()).unwrap_or(0);
                let flat: Vec<f64> = rows.into_iter().flatten().collect();
                let tensor = self.make_tensor(flat, vec![nrows, ncols])?;
                Ok(Some(Value::Tensor(Rc::new(RefCell::new(tensor)))))
            }

            HirExprKind::If { cond, then_branch, else_branch, .. } => {
                let c = self.eval_expr(cond)?.ok_or_else(|| TenthError::RuntimeError {
                    message: "if condition is void".into(),
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
                    message: "assign value is void".into(),
                })?;
                self.set_var(target.clone(), v);
                Ok(Some(Value::Unit))
            }

            HirExprKind::DerefAssign { target, value } => {
                let target_val = self.eval_expr(target)?.ok_or_else(|| TenthError::RuntimeError {
                    message: "deref-assign target is void".into(),
                })?;
                let rhs = self.eval_expr(value)?.ok_or_else(|| TenthError::RuntimeError {
                    message: "deref-assign value is void".into(),
                })?;
                match &target_val {
                    Value::MutRef(rc) => {
                        *rc.borrow_mut() = rhs;
                        Ok(Some(Value::Unit))
                    }
                    _ => Err(TenthError::RuntimeError {
                        message: "can only assign through mutable reference".into(),
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
                    message: "assign-op value is void".into(),
                })?;
                let result = self.eval_binary(op, &current, &rhs)?;
                self.set_var(target.clone(), result);
                Ok(Some(Value::Unit))
            }

            HirExprKind::DerefAssignOp { target, op, value } => {
                let target_val = self.eval_expr(target)?.ok_or_else(|| TenthError::RuntimeError {
                    message: "deref-assignop target is void".into(),
                })?;
                let rhs = self.eval_expr(value)?.ok_or_else(|| TenthError::RuntimeError {
                    message: "deref-assignop value is void".into(),
                })?;
                match &target_val {
                    Value::MutRef(rc) => {
                        let current = rc.borrow().clone();
                        let result = self.eval_binary(op, &current, &rhs)?;
                        *rc.borrow_mut() = result;
                        Ok(Some(Value::Unit))
                    }
                    _ => Err(TenthError::RuntimeError {
                        message: "can only assign through mutable reference".into(),
                    }),
                }
            }

            HirExprKind::Closure { params, body } => {
                Ok(Some(Value::Closure {
                    params: params.clone(),
                    body: Rc::new((**body).clone()),
                    captures: Vec::new(),
                }))
            }

            HirExprKind::Range { .. } => Ok(Some(Value::Unit)),

            HirExprKind::Ref(inner) => {
                let val = self.eval_expr(inner)?.ok_or_else(|| TenthError::RuntimeError {
                    message: "ref operand is void".into(),
                })?;
                Ok(Some(Value::Ref(Rc::new(RefCell::new(val)))))
            }

            HirExprKind::MutRef(inner) => {
                let val = self.eval_expr(inner)?.ok_or_else(|| TenthError::RuntimeError {
                    message: "mut ref operand is void".into(),
                })?;
                if let HirExprKind::Var(var_name) = &inner.kind {
                    let rc = match self.variables.get(var_name) {
                        Some(Value::Shared(existing)) => existing.clone(),
                        _ => {
                            let cell = Rc::new(RefCell::new(val));
                            self.variables.insert(var_name.clone(), Value::Shared(cell.clone()));
                            cell
                        }
                    };
                    Ok(Some(Value::MutRef(rc)))
                } else {
                    Ok(Some(Value::MutRef(Rc::new(RefCell::new(val)))))
                }
            }

            HirExprKind::Deref(inner) => {
                let val = self.eval_expr(inner)?.ok_or_else(|| TenthError::RuntimeError {
                    message: "deref operand is void".into(),
                })?;
                match &val {
                    Value::Ref(rc) | Value::MutRef(rc) => Ok(Some(rc.borrow().clone())),
                    _ => Err(TenthError::RuntimeError {
                        message: "cannot dereference non-reference value".into(),
                    }),
                }
            }

            HirExprKind::FieldAssign { target, field, value } => {
                let target_val = self.eval_expr(target)?.ok_or_else(|| TenthError::RuntimeError {
                    message: "field-assign target is void".into(),
                })?;
                let rhs = self.eval_expr(value)?.ok_or_else(|| TenthError::RuntimeError {
                    message: "field-assign value is void".into(),
                })?;
                match &target_val {
                    Value::MutRef(rc) => {
                        let mut inner = rc.borrow_mut();
                        match &mut *inner {
                            Value::Struct { fields, .. } => {
                                for (fname, fval) in &mut *fields {
                                    if fname == field {
                                        *fval = rhs;
                                        return Ok(Some(Value::Unit));
                                    }
                                }
                                Err(TenthError::RuntimeError {
                                    message: format!("struct has no field '{}'", field),
                                })
                            }
                            _ => Err(TenthError::RuntimeError {
                                message: "field assignment only supported on structs".into(),
                            }),
                        }
                    }
                    _ => Err(TenthError::RuntimeError {
                        message: "can only assign fields through mutable reference".into(),
                    }),
                }
            }

            HirExprKind::Move(inner) => {
                let val = self.eval_expr(inner)?.ok_or_else(|| TenthError::RuntimeError {
                    message: "move operand is void".into(),
                })?;
                if let HirExprKind::Var(var_name) = &inner.kind {
                    self.variables.insert(var_name.clone(), Value::Moved);
                }
                Ok(Some(val))
            }

            HirExprKind::StructLiteral { name, fields } => {
                let mut field_vals = Vec::new();
                for (fname, fexpr) in fields {
                    let v = self.eval_expr(fexpr)?.ok_or_else(|| TenthError::RuntimeError {
                        message: format!("struct field '{}' is void", fname),
                    })?;
                    field_vals.push((fname.clone(), v));
                }
                Ok(Some(Value::Struct { name: name.clone(), fields: field_vals }))
            }

            HirExprKind::EnumLiteral { enum_name, variant, fields } => {
                let mut field_vals = Vec::new();
                for (fname, fexpr) in fields {
                    let v = self.eval_expr(fexpr)?.ok_or_else(|| TenthError::RuntimeError {
                        message: format!("enum field '{}' is void", fname),
                    })?;
                    field_vals.push((fname.clone(), v));
                }
                Ok(Some(Value::Enum {
                    enum_name: enum_name.clone(),
                    variant: variant.clone(),
                    fields: field_vals,
                }))
            }

            HirExprKind::Match { scrutinee, arms } => {
                let val = self.eval_expr(scrutinee)?.ok_or_else(|| TenthError::RuntimeError {
                    message: "match scrutinee is void".into(),
                })?;

                for arm in arms {
                    if self.pattern_matches(&arm.pattern, &val) {
                        if let HirPattern::EnumVariant { field_bind, .. } = &arm.pattern {
                            if let Some((_fname, bname)) = field_bind {
                                if let Value::Enum { fields, .. } = &val {
                                    if let Some((_, v)) = fields.first() {
                                        self.variables.insert(bname.clone(), v.clone());
                                    }
                                }
                            }
                        }
                        let result = self.eval_expr(&arm.body);
                        if let HirPattern::EnumVariant { field_bind, .. } = &arm.pattern {
                            if let Some((_, bname)) = field_bind {
                                self.variables.remove(bname);
                            }
                        }
                        return result;
                    }
                }
                Ok(Some(Value::Unit))
            }
        }
    }

    fn pattern_matches(&self, pattern: &HirPattern, val: &Value) -> bool {
        match pattern {
            HirPattern::Wildcard => true,
            HirPattern::Literal(lit) => {
                match (lit, val) {
                    (Literal::Int(a), Value::Int(b)) => a == b,
                    (Literal::Float(a), Value::Float(b)) => (a - b).abs() < 1e-10,
                    (Literal::Bool(a), Value::Bool(b)) => a == b,
                    _ => false,
                }
            }
            HirPattern::EnumVariant { enum_name, variant, .. } => {
                match val {
                    Value::Enum { enum_name: e, variant: v, .. } => {
                        enum_name == e && variant == v
                    }
                    _ => false,
                }
            }
        }
    }

    fn eval_binary(&self, op: &BinOp, l: &Value, r: &Value) -> TenthResult<Value> {
        match op {
            BinOp::Add => match (l, r) {
                (Value::Int(a), Value::Int(b)) => Ok(Value::Int(a + b)),
                (Value::Float(a), Value::Float(b)) => Ok(Value::Float(a + b)),
                (Value::Int(a), Value::Float(b)) => Ok(Value::Float(*a as f64 + b)),
                (Value::Float(a), Value::Int(b)) => Ok(Value::Float(a + *b as f64)),
                (Value::String(a), Value::String(b)) => Ok(Value::String(format!("{}{}", a, b))),
                (Value::Tensor(t), Value::Float(s)) => {
                    let result = t.borrow().add_scalar(*s);
                    Ok(Value::Tensor(Rc::new(RefCell::new(result))))
                }
                (Value::Float(s), Value::Tensor(t)) => {
                    let result = t.borrow().add_scalar(*s);
                    Ok(Value::Tensor(Rc::new(RefCell::new(result))))
                }
                _ => Err(TenthError::RuntimeError {
                    message: "type mismatch in addition".into(),
                }),
            },
            BinOp::Sub => match (l, r) {
                (Value::Int(a), Value::Int(b)) => Ok(Value::Int(a - b)),
                (Value::Float(a), Value::Float(b)) => Ok(Value::Float(a - b)),
                (Value::Int(a), Value::Float(b)) => Ok(Value::Float(*a as f64 - b)),
                (Value::Float(a), Value::Int(b)) => Ok(Value::Float(a - *b as f64)),
                (Value::Tensor(t), Value::Float(s)) => {
                    let result = t.borrow().sub_scalar(*s);
                    Ok(Value::Tensor(Rc::new(RefCell::new(result))))
                }
                _ => Err(TenthError::RuntimeError {
                    message: "type mismatch in subtraction".into(),
                }),
            },
            BinOp::Mul => match (l, r) {
                (Value::Int(a), Value::Int(b)) => Ok(Value::Int(a * b)),
                (Value::Float(a), Value::Float(b)) => Ok(Value::Float(a * b)),
                (Value::Int(a), Value::Float(b)) => Ok(Value::Float(*a as f64 * b)),
                (Value::Float(a), Value::Int(b)) => Ok(Value::Float(a * *b as f64)),
                (Value::Tensor(t), Value::Float(s)) => {
                    let result = t.borrow().mul_scalar(*s);
                    Ok(Value::Tensor(Rc::new(RefCell::new(result))))
                }
                (Value::Float(s), Value::Tensor(t)) => {
                    let result = t.borrow().mul_scalar(*s);
                    Ok(Value::Tensor(Rc::new(RefCell::new(result))))
                }
                _ => Err(TenthError::RuntimeError {
                    message: "type mismatch in multiplication".into(),
                }),
            },
            BinOp::Div => match (l, r) {
                (Value::Int(a), Value::Int(b)) => Ok(Value::Float(*a as f64 / *b as f64)),
                (Value::Float(a), Value::Float(b)) => Ok(Value::Float(a / b)),
                (Value::Int(a), Value::Float(b)) => Ok(Value::Float(*a as f64 / b)),
                (Value::Float(a), Value::Int(b)) => Ok(Value::Float(a / *b as f64)),
                (Value::Tensor(t), Value::Float(s)) => {
                    let result = t.borrow().div_scalar(*s);
                    Ok(Value::Tensor(Rc::new(RefCell::new(result))))
                }
                _ => Err(TenthError::RuntimeError {
                    message: "type mismatch in division".into(),
                }),
            },
            BinOp::Mod => match (l, r) {
                (Value::Int(a), Value::Int(b)) => Ok(Value::Int(a % b)),
                _ => Err(TenthError::RuntimeError {
                    message: "modulo only supports integers".into(),
                }),
            },
            BinOp::Eq => Ok(Value::Bool(self.values_eq(l, r))),
            BinOp::NotEq => Ok(Value::Bool(!self.values_eq(l, r))),
            BinOp::Lt => match (l, r) {
                (Value::Int(a), Value::Int(b)) => Ok(Value::Bool(a < b)),
                (Value::Float(a), Value::Float(b)) => Ok(Value::Bool(a < b)),
                (Value::Int(a), Value::Float(b)) => Ok(Value::Bool((*a as f64) < *b)),
                (Value::Float(a), Value::Int(b)) => Ok(Value::Bool(*a < *b as f64)),
                _ => Err(TenthError::RuntimeError {
                    message: "comparison requires numeric types".into(),
                }),
            },
            BinOp::Gt => match (l, r) {
                (Value::Int(a), Value::Int(b)) => Ok(Value::Bool(a > b)),
                (Value::Float(a), Value::Float(b)) => Ok(Value::Bool(a > b)),
                (Value::Int(a), Value::Float(b)) => Ok(Value::Bool((*a as f64) > *b)),
                (Value::Float(a), Value::Int(b)) => Ok(Value::Bool(*a > *b as f64)),
                _ => Err(TenthError::RuntimeError {
                    message: "comparison requires numeric types".into(),
                }),
            },
            BinOp::LtEq => match (l, r) {
                (Value::Int(a), Value::Int(b)) => Ok(Value::Bool(a <= b)),
                (Value::Float(a), Value::Float(b)) => Ok(Value::Bool(a <= b)),
                (Value::Int(a), Value::Float(b)) => Ok(Value::Bool((*a as f64) <= *b)),
                (Value::Float(a), Value::Int(b)) => Ok(Value::Bool(*a <= *b as f64)),
                _ => Err(TenthError::RuntimeError {
                    message: "comparison requires numeric types".into(),
                }),
            },
            BinOp::GtEq => match (l, r) {
                (Value::Int(a), Value::Int(b)) => Ok(Value::Bool(a >= b)),
                (Value::Float(a), Value::Float(b)) => Ok(Value::Bool(a >= b)),
                (Value::Int(a), Value::Float(b)) => Ok(Value::Bool((*a as f64) >= *b)),
                (Value::Float(a), Value::Int(b)) => Ok(Value::Bool(*a >= *b as f64)),
                _ => Err(TenthError::RuntimeError {
                    message: "comparison requires numeric types".into(),
                }),
            },
            BinOp::And => Ok(Value::Bool(l.is_truthy() && r.is_truthy())),
            BinOp::Or => Ok(Value::Bool(l.is_truthy() || r.is_truthy())),
        }
    }

    fn values_eq(&self, l: &Value, r: &Value) -> bool {
        match (l, r) {
            (Value::Int(a), Value::Int(b)) => a == b,
            (Value::Float(a), Value::Float(b)) => (a - b).abs() < 1e-10,
            (Value::Bool(a), Value::Bool(b)) => a == b,
            (Value::String(a), Value::String(b)) => a == b,
            (Value::Unit, Value::Unit) => true,
            _ => false,
        }
    }

    fn eval_unary(&self, op: &UnaryOp, val: &Value) -> TenthResult<Value> {
        match op {
            UnaryOp::Neg => match val {
                Value::Int(n) => Ok(Value::Int(-n)),
                Value::Float(n) => Ok(Value::Float(-n)),
                Value::Tensor(t) => {
                    let result = t.borrow().neg();
                    Ok(Value::Tensor(Rc::new(RefCell::new(result))))
                }
                _ => Err(TenthError::RuntimeError {
                    message: "cannot negate this value".into(),
                }),
            },
            UnaryOp::Not => Ok(Value::Bool(!val.is_truthy())),
        }
    }

    fn eval_method_call(&mut self, recv: &Value, method: &str, args: &[Value]) -> TenthResult<Option<Value>> {
        match recv {
            Value::Struct { name, .. } => {
                if let Some(type_methods) = self.find_methods_for_type(name) {
                    if let Some(method_fn) = type_methods.get(method) {
                        return self.call_method_impl(recv, method_fn, args);
                    }
                }
                for (_trait_name, type_impls) in &self.trait_impls {
                    if let Some(methods) = type_impls.get(name) {
                        if let Some(method_fn) = methods.get(method) {
                            let fn_def = method_fn.clone();
                            return self.call_method_impl(recv, &fn_def, args);
                        }
                    }
                }
                self.eval_tensor_method(recv, method, args).map(Some)
            }
            Value::Enum { .. } => {
                self.eval_tensor_method(recv, method, args).map(Some)
            }
            Value::Tensor(_) => {
                self.eval_tensor_method(recv, method, args).map(Some)
            }
            Value::Float(f) => {
                self.eval_scalar_method(*f, method, args).map(Some)
            }
            Value::Int(i) => {
                let f = *i as f64;
                self.eval_scalar_method(f, method, args).map(Some)
            }
            Value::String(_) | Value::Vec(_) | Value::Map(_) => {
                self.eval_native_method(recv, method, args)
            }
            _ => Err(TenthError::RuntimeError {
                message: format!("method '{}' not supported on this type", method),
            }),
        }
    }

    fn eval_native_method(&mut self, recv: &Value, method: &str, args: &[Value]) -> TenthResult<Option<Value>> {
        match recv {
            Value::String(s) => self.eval_string_method(s, method, args),
            Value::Vec(items) => self.eval_vec_method(items, method, args),
            Value::Map(m) => self.eval_map_method(m, method, args),
            _ => Err(TenthError::RuntimeError {
                message: format!("native method '{}' not available", method),
            }),
        }
    }

    fn eval_string_method(&self, s: &str, method: &str, _args: &[Value]) -> TenthResult<Option<Value>> {
        match method {
            "len" => Ok(Some(Value::Int(s.chars().count() as i64))),
            _ => Err(TenthError::RuntimeError {
                message: format!("String has no method '{}'", method),
            }),
        }
    }

    fn eval_vec_method(&mut self, items: &Rc<RefCell<Vec<Value>>>, method: &str, args: &[Value]) -> TenthResult<Option<Value>> {
        match method {
            "len" => Ok(Some(Value::Int(items.borrow().len() as i64))),
            "push" => {
                if args.len() != 1 {
                    return Err(TenthError::RuntimeError {
                        message: "push() takes 1 argument".into(),
                    });
                }
                items.borrow_mut().push(args[0].clone());
                Ok(Some(Value::Unit))
            }
            _ => Err(TenthError::RuntimeError {
                message: format!("Vec has no method '{}'", method),
            }),
        }
    }

    fn eval_map_method(&self, m: &Rc<RefCell<HashMap<String, Value>>>, method: &str, args: &[Value]) -> TenthResult<Option<Value>> {
        let map = m.borrow();
        match method {
            "len" => Ok(Some(Value::Int(map.len() as i64))),
            "get" => {
                if args.len() != 1 {
                    return Err(TenthError::RuntimeError {
                        message: "get() takes 1 argument".into(),
                    });
                }
                if let Value::String(key) = &args[0] {
                    Ok(map.get(key).cloned())
                } else {
                    Err(TenthError::RuntimeError {
                        message: "HashMap key must be a string".into(),
                    })
                }
            }
            _ => Err(TenthError::RuntimeError {
                message: format!("HashMap has no method '{}'", method),
            }),
        }
    }

    fn find_methods_for_type(&self, type_name: &str) -> Option<HashMap<String, HirFnDef>> {
        for (impl_type, methods) in &self.methods {
            if impl_type == type_name {
                return Some(methods.clone());
            }
        }
        for module in self.modules.values() {
            for func in &module.functions {
                if func.name == type_name {
                    return None;
                }
            }
        }
        None
    }

    fn call_method_impl(&mut self, receiver: &Value, method_fn: &HirFnDef, args: &[Value]) -> TenthResult<Option<Value>> {
        let saved: HashMap<String, Value> = method_fn.params.iter()
            .filter_map(|(n, _)| self.variables.get(n).cloned().map(|v| (n.clone(), v)))
            .collect();

        let self_saved = self.variables.get("self").cloned();

        self.variables.insert("self".to_string(), receiver.clone());

        for ((pname, _), arg) in method_fn.params.iter().skip(1).zip(args.iter()) {
            self.variables.insert(pname.clone(), arg.clone());
        }

        let result = self.eval_expr(&method_fn.body);

        for (n, v) in saved {
            self.variables.insert(n, v);
        }

        if let Some(v) = self_saved {
            self.variables.insert("self".to_string(), v);
        } else {
            self.variables.remove("self");
        }

        result
    }

    fn eval_tensor_method(&self, recv: &Value, method: &str, args: &[Value]) -> TenthResult<Value> {
        match recv {
            Value::Tensor(t) => {
                let tensor = t.borrow();
                match method {
                    "sum" => {
                        if args.is_empty() {
                            Ok(Value::Float(tensor.sum()))
                        } else {
                            let axis = args[0].as_int().unwrap_or(0) as usize;
                            let result = tensor.sum_axis(axis);
                            Ok(Value::Tensor(Rc::new(RefCell::new(result))))
                        }
                    }
                    "mean" => Ok(Value::Float(tensor.mean())),
                    "abs" => {
                        let result = tensor.abs();
                        Ok(Value::Tensor(Rc::new(RefCell::new(result))))
                    }
                    "sqrt" => {
                        let result = tensor.sqrt();
                        Ok(Value::Tensor(Rc::new(RefCell::new(result))))
                    }
                    "exp" => {
                        let result = tensor.exp();
                        Ok(Value::Tensor(Rc::new(RefCell::new(result))))
                    }
                    "log" => {
                        let result = tensor.log();
                        Ok(Value::Tensor(Rc::new(RefCell::new(result))))
                    }
                    "relu" => {
                        let result = tensor.relu();
                        Ok(Value::Tensor(Rc::new(RefCell::new(result))))
                    }
                    "sigmoid" => {
                        let result = tensor.sigmoid();
                        Ok(Value::Tensor(Rc::new(RefCell::new(result))))
                    }
                    "tanh" => {
                        let result = tensor.tanh();
                        Ok(Value::Tensor(Rc::new(RefCell::new(result))))
                    }
                    "reshape" | "view" => {
                        let shape: Vec<usize> = args.iter()
                            .map(|a| a.as_int().unwrap_or(1) as usize)
                            .collect();
                        let result = tensor.reshape(&shape).ok_or_else(|| {
                            TenthError::RuntimeError {
                                message: format!("cannot reshape to {:?}", shape),
                            }
                        })?;
                        Ok(Value::Tensor(Rc::new(RefCell::new(result))))
                    }
                    "flatten" => {
                        let result = tensor.flatten();
                        Ok(Value::Tensor(Rc::new(RefCell::new(result))))
                    }
                    "softmax" => {
                        let result = tensor.softmax().ok_or_else(|| TenthError::RuntimeError {
                            message: "softmax failed".into(),
                        })?;
                        Ok(Value::Tensor(Rc::new(RefCell::new(result))))
                    }
                    _ => Err(TenthError::RuntimeError {
                        message: format!("unknown tensor method: {}", method),
                    }),
                }
            }
            Value::Struct { .. } => Err(TenthError::RuntimeError {
                message: format!("unknown method '{}'", method),
            }),
            _ => Err(TenthError::RuntimeError {
                message: format!("method '{}' not supported on this type", method),
            }),
        }
    }

    fn eval_scalar_method(&self, val: f64, method: &str, _args: &[Value]) -> TenthResult<Value> {
        match method {
            "sqrt" => Ok(Value::Float(val.sqrt())),
            "abs" => Ok(Value::Float(val.abs())),
            "exp" => Ok(Value::Float(val.exp())),
            "log" => Ok(Value::Float(val.ln())),
            _ => Err(TenthError::RuntimeError {
                message: format!("unknown method '{}' on scalar", method),
            }),
        }
    }

    fn eval_index(&mut self, target: &Value, indices: &[Index]) -> TenthResult<Value> {
        match target {
            Value::String(s) => {
                if indices.len() != 1 {
                    return Err(TenthError::RuntimeError {
                        message: "string indexing takes exactly 1 index".into(),
                    });
                }
                match &indices[0] {
                    Index::Single(e) => {
                        let v = self.eval_expr(e)?.ok_or_else(|| TenthError::RuntimeError {
                            message: "index is void".into(),
                        })?;
                        let idx = v.as_int().unwrap_or(0) as usize;
                        s.chars().nth(idx).map(|c| Value::String(c.to_string())).ok_or_else(|| {
                            TenthError::RuntimeError {
                                message: format!("string index {} out of bounds", idx),
                            }
                        })
                    }
                    Index::Range { start, end } => {
                        let s_val = s.clone();
                        let start_idx = match start {
                            Some(e) => {
                                let v = self.eval_expr(e)?.ok_or_else(|| TenthError::RuntimeError {
                                    message: "range start is void".into(),
                                })?;
                                v.as_int().unwrap_or(0) as usize
                            }
                            None => 0,
                        };
                        let end_idx = match end {
                            Some(e) => {
                                let v = self.eval_expr(e)?.ok_or_else(|| TenthError::RuntimeError {
                                    message: "range end is void".into(),
                                })?;
                                v.as_int().unwrap_or(0) as usize
                            }
                            None => s_val.chars().count(),
                        };
                        let chars: Vec<char> = s_val.chars().collect();
                        if start_idx > chars.len() || end_idx > chars.len() || start_idx > end_idx {
                            return Err(TenthError::RuntimeError {
                                message: format!("string slice {}..{} out of bounds", start_idx, end_idx),
                            });
                        }
                        let slice: String = chars[start_idx..end_idx].iter().collect();
                        Ok(Value::String(slice))
                    }
                    _ => Err(TenthError::RuntimeError {
                        message: "string index must be int or range".into(),
                    }),
                }
            }
            Value::Tensor(t) => {
                let tensor = t.borrow();
                let shape = tensor.shape();
                let mut idx: Vec<usize> = Vec::new();
                for (i, index_expr) in indices.iter().enumerate() {
                    match index_expr {
                        Index::Single(e) => {
                            let v = self.eval_expr(e)?.ok_or_else(|| TenthError::RuntimeError {
                                message: "index is void".into(),
                            })?;
                            idx.push(v.as_int().unwrap_or(0) as usize);
                        }
                        _ => {
                            if i < shape.len() {
                                idx.push(0);
                            }
                        }
                    }
                }
                match tensor.get(&idx) {
                    Some(val) => Ok(Value::Float(val)),
                    None => Err(TenthError::RuntimeError {
                        message: format!("index {:?} out of bounds for shape {:?}", idx, shape),
                    }),
                }
            }
            Value::Vec(items) => {
                if indices.len() != 1 {
                    return Err(TenthError::RuntimeError {
                        message: "Vec indexing takes exactly 1 index".into(),
                    });
                }
                match &indices[0] {
                    Index::Single(e) => {
                        let v = self.eval_expr(e)?.ok_or_else(|| TenthError::RuntimeError {
                            message: "index is void".into(),
                        })?;
                        let idx = v.as_int().unwrap_or(0) as usize;
                        items.borrow().get(idx).cloned().ok_or_else(|| {
                            TenthError::RuntimeError {
                                message: format!("Vec index {} out of bounds", idx),
                            }
                        })
                    }
                    _ => Err(TenthError::RuntimeError {
                        message: "Vec index must be an integer".into(),
                    }),
                }
            }
            _ => Err(TenthError::RuntimeError {
                message: "indexing not supported on this type".into(),
            }),
        }
    }

    fn eval_call(
        &mut self, func: &Value, args: &[Value], span: &crate::lexer::token::Span,
    ) -> TenthResult<Option<Value>> {
        match func {
            Value::FnRef { name, .. } => {
                self.call_named_fn(name, args, span)
            }
            Value::Closure { params, body, captures } => {
                let saved: HashMap<String, Value> = params.iter()
                    .filter_map(|(n, _)| self.variables.get(n).cloned().map(|v| (n.clone(), v)))
                    .collect();

                for ((pname, _), arg) in params.iter().zip(args.iter()) {
                    self.variables.insert(pname.clone(), arg.clone());
                }

                for (cap_name, cap_val) in captures {
                    self.variables.insert(cap_name.clone(), cap_val.clone());
                }

                let result = self.eval_expr(body);

                for (n, v) in saved {
                    self.variables.insert(n, v);
                }

                result
            }
            _ => Err(TenthError::RuntimeError {
                message: "not a callable value".into(),
            }),
        }
    }

    fn call_named_fn(
        &mut self, name: &str, args: &[Value], _span: &crate::lexer::token::Span,
    ) -> TenthResult<Option<Value>> {
        match name {
            "println" => {
                for arg in args {
                    print!("{}", arg);
                }
                println!();
                return Ok(Some(Value::Unit));
            }
            "tensor" => {
                if let Some(arg) = args.first() {
                    return Ok(Some(arg.clone()));
                }
                return Ok(Some(Value::Unit));
            }
            "rand" => {
                let shape: Vec<usize> = args.iter()
                    .map(|a| a.as_int().unwrap_or(1) as usize)
                    .collect();
                let t = Tensor::rand(&shape);
                return Ok(Some(Value::Tensor(Rc::new(RefCell::new(t)))));
            }
            "randn" => {
                let shape: Vec<usize> = args.iter()
                    .map(|a| a.as_int().unwrap_or(1) as usize)
                    .collect();
                let t = Tensor::randn(&shape);
                return Ok(Some(Value::Tensor(Rc::new(RefCell::new(t)))));
            }
            "read_file" => {
                if let Some(Value::String(path)) = args.first() {
                    match std::fs::read_to_string(path) {
                        Ok(content) => return Ok(Some(Value::String(content))),
                        Err(e) => return Err(TenthError::RuntimeError {
                            message: format!("read_file failed: {}", e),
                        }),
                    }
                }
                return Err(TenthError::RuntimeError {
                    message: "read_file(path) expects a string path".into(),
                });
            }
            "write_file" => {
                if args.len() >= 2 {
                    if let (Value::String(path), Value::String(content)) = (&args[0], &args[1]) {
                        match std::fs::write(path, content) {
                            Ok(()) => return Ok(Some(Value::Unit)),
                            Err(e) => return Err(TenthError::RuntimeError {
                                message: format!("write_file failed: {}", e),
                            }),
                        }
                    }
                }
                return Err(TenthError::RuntimeError {
                    message: "write_file(path, content) expects two string args".into(),
                });
            }
            "Vec::new" => return Ok(Some(Value::Vec(Rc::new(RefCell::new(Vec::new()))))),
            "HashMap::new" => return Ok(Some(Value::Map(Rc::new(RefCell::new(HashMap::new()))))),
            _ => {}
        }

        if name.contains("::") {
            let parts: Vec<&str> = name.splitn(2, "::").collect();
            if parts.len() == 2 {
                let mod_name = parts[0];
                let fn_name = parts[1];
                if let Some(module) = self.modules.get(mod_name) {
                    if let Some(fn_def) = module.functions.iter().find(|f| f.name == fn_name) {
                        let fn_def = fn_def.clone();
                        let saved: HashMap<String, Value> = fn_def.params.iter()
                            .filter_map(|(n, _)| self.variables.get(n).cloned().map(|v| (n.clone(), v)))
                            .collect();

                        for ((pname, _), arg) in fn_def.params.iter().zip(args.iter()) {
                            self.variables.insert(pname.clone(), arg.clone());
                        }

                        let result = self.eval_expr(&fn_def.body);

                        for (n, v) in saved {
                            self.variables.insert(n, v);
                        }

                        return result;
                    }
                }
                return Err(TenthError::RuntimeError {
                    message: format!("undefined function '{}'", name),
                });
            }
        }

        let func_def = self.functions.iter().find(|f| f.name == name).cloned();
        if let Some(fd) = func_def {
            let saved: HashMap<String, Value> = fd.params.iter()
                .filter_map(|(n, _)| self.variables.get(n).cloned().map(|v| (n.clone(), v)))
                .collect();

            for ((pname, _), arg) in fd.params.iter().zip(args.iter()) {
                self.variables.insert(pname.clone(), arg.clone());
            }

            let result = self.eval_expr(&fd.body);

            for (n, v) in saved {
                self.variables.insert(n, v);
            }

            return result;
        }

        Err(TenthError::RuntimeError {
            message: format!("undefined function '{}'", name),
        })
    }

    fn eval_stmt(&mut self, stmt: &HirStmt) -> TenthResult<()> {
        match &stmt.kind {
            HirStmtKind::Expr(e) => {
                self.eval_expr(e)?;
                Ok(())
            }
            HirStmtKind::Let { name, init, .. } => {
                let val = match init {
                    Some(e) => self.eval_expr(e)?.unwrap_or(Value::Unit),
                    None => Value::Unit,
                };
                self.variables.insert(name.clone(), val);
                Ok(())
            }
            HirStmtKind::Return(_) => Ok(()),
            HirStmtKind::Break => Ok(()),
            HirStmtKind::Continue => Ok(()),
            HirStmtKind::Loop { body } => {
                loop {
                    for s in body {
                        self.eval_stmt(s)?;
                    }
                }
            }
            HirStmtKind::While { cond, body } => {
                loop {
                    let c = self.eval_expr(cond)?.ok_or_else(|| TenthError::RuntimeError {
                        message: "while condition is void".into(),
                    })?;
                    if !c.is_truthy() {
                        break;
                    }
                    self.eval_stmt(body)?;
                }
                Ok(())
            }
            HirStmtKind::For { var, iter, body } => {
                let iter_val = self.eval_expr(iter)?.ok_or_else(|| TenthError::RuntimeError {
                    message: "for iterable is void".into(),
                })?;
                match iter_val {
                    Value::Tensor(t) => {
                        let tensor = t.borrow();
                        let shape = tensor.shape();
                        let n = shape.first().copied().unwrap_or(0);

                        for i in 0..n {
                            let val = match tensor.get(&[i]) {
                                Some(v) => Value::Float(v),
                                None => Value::Unit,
                            };
                            self.variables.insert(var.clone(), val);
                            self.eval_stmt(body)?;
                        }
                    }
                    _ => {
                        return Err(TenthError::RuntimeError {
                            message: "for loop only supports tensor iteration for now".into(),
                        });
                    }
                }
                Ok(())
            }
        }
    }
}