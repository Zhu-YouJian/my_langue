use std::collections::HashMap;
use std::rc::Rc;
use std::cell::RefCell;
use crate::error::{TenthError, TenthResult};
use crate::hir::hir::*;
use crate::hir::types::{BaseType, Dim, Type};
use super::value::{Value, LazyIterator, IteratorTransform};
use super::tensor::Tensor;
use super::limits::RuntimeLimits;
use super::arena::Arena;
use super::autodiff::{Tape, TapeOp};

/// Default arena capacity when no explicit limit is configured.
const DEFAULT_ARENA_CAPACITY: usize = 64 * 1024; // 64K f64 slots = 512 KB

pub struct Interpreter {
    /// Scope chain: scopes[0] is global, pushed/popped on function entry/exit.
    /// Variable lookup walks from the last scope backward.
    pub scopes: Vec<HashMap<String, Value>>,
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
}

fn days_to_date(days: u64) -> (u64, u64, u64) {
    // Algorithm from http://howardhinnant.github.io/date_algorithms.html
    let z = days + 719468;
    let era = z / 146097;
    let doe = z - era * 146097;
    let yoe = (doe - doe/1460 + doe/36524 - doe/146096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365*yoe + yoe/4 - yoe/100);
    let mp = (5*doy + 2) / 153;
    let d = doy - (153*mp+2)/5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    (y, m, d)
}

fn json_encode_value(val: &Value) -> String {
    match val {
        Value::Int(n) => format!("{}", n),
        Value::Float(f) => format!("{}", f),
        Value::Bool(b) => if *b { "true".into() } else { "false".into() },
        Value::String(s) => format!("\"{}\"", s.replace('\\', "\\\\").replace('"', "\\\"").replace('\n', "\\n").replace('\t', "\\t")),
        Value::Unit => "null".into(),
        Value::Vec(v) => {
            let items: Vec<String> = v.borrow().iter().map(|v| json_encode_value(v)).collect();
            format!("[{}]", items.join(", "))
        }
        Value::Array(a) => {
            let items: Vec<String> = a.borrow().iter().map(|v| json_encode_value(v)).collect();
            format!("[{}]", items.join(", "))
        }
        Value::Map(map) => {
            let entries: Vec<String> = map.borrow().iter().map(|(k, v)| {
                format!("\"{}\": {}", k, json_encode_value(v))
            }).collect();
            format!("{{{}}}", entries.join(", "))
        }
        _ => "null".into(),
    }
}

fn json_encode_value_pretty(val: &Value, indent: usize) -> String {
    let prefix = "  ".repeat(indent);
    let inner_prefix = "  ".repeat(indent + 1);
    match val {
        Value::Int(n) => format!("{}", n),
        Value::Float(f) => format!("{}", f),
        Value::Bool(b) => if *b { "true".into() } else { "false".into() },
        Value::String(s) => format!("\"{}\"", s.replace('\\', "\\\\").replace('"', "\\\"").replace('\n', "\\n").replace('\t', "\\t")),
        Value::Unit => "null".into(),
        Value::Vec(v) => {
            if v.borrow().is_empty() { return "[]".into(); }
            let items: Vec<String> = v.borrow().iter().map(|v| format!("{}{}", inner_prefix, json_encode_value_pretty(v, indent + 1))).collect();
            format!("[\n{}\n{}]", items.join(",\n"), prefix)
        }
        Value::Array(a) => {
            if a.borrow().is_empty() { return "[]".into(); }
            let items: Vec<String> = a.borrow().iter().map(|v| format!("{}{}", inner_prefix, json_encode_value_pretty(v, indent + 1))).collect();
            format!("[\n{}\n{}]", items.join(",\n"), prefix)
        }
        Value::Map(map) => {
            if map.borrow().is_empty() { return "{}".into(); }
            let entries: Vec<String> = map.borrow().iter().map(|(k, v)| {
                format!("{}\"{}\": {}", inner_prefix, k, json_encode_value_pretty(v, indent + 1))
            }).collect();
            format!("{{\n{}\n{}}}", entries.join(",\n"), prefix)
        }
        _ => "null".into(),
    }
}

fn json_decode_string(s: &str) -> Value {
    let s = s.trim();
    if s == "null" { return Value::Unit; }
    if s == "true" { return Value::Bool(true); }
    if s == "false" { return Value::Bool(false); }
    if s.starts_with('"') && s.ends_with('"') {
        let inner = &s[1..s.len()-1];
        return Value::String(inner.replace("\\\"", "\"").replace("\\\\", "\\").replace("\\n", "\n").replace("\\t", "\t"));
    }
    if let Ok(n) = s.parse::<i64>() { return Value::Int(n); }
    if let Ok(f) = s.parse::<f64>() { return Value::Float(f); }
    if s.starts_with('[') && s.ends_with(']') {
        let inner = &s[1..s.len()-1];
        if inner.trim().is_empty() { return Value::Vec(Rc::new(RefCell::new(Vec::new()))); }
        let items: Vec<Value> = simple_json_split(inner, ',')
            .iter()
            .map(|s| json_decode_string(s))
            .collect();
        return Value::Vec(Rc::new(RefCell::new(items)));
    }
    if s.starts_with('{') && s.ends_with('}') {
        let inner = &s[1..s.len()-1];
        if inner.trim().is_empty() {
            return Value::Map(Rc::new(RefCell::new(std::collections::HashMap::new())));
        }
        let mut map = std::collections::HashMap::new();
        let entries = simple_json_split(inner, ',');
        for entry in &entries {
            let parts = simple_json_split(entry, ':');
            if parts.len() >= 2 {
                let key = json_decode_string(parts[0].trim());
                if let Value::String(k) = key {
                    let val = json_decode_string(parts[1].trim());
                    map.insert(k, val);
                }
            }
        }
        return Value::Map(Rc::new(RefCell::new(map)));
    }
    Value::Unit
}

fn simple_json_split(s: &str, delimiter: char) -> Vec<String> {
    let mut result = Vec::new();
    let mut current = String::new();
    let mut depth = 0i32;
    let mut in_string = false;
    let mut chars = s.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '"' && !in_string { in_string = true; current.push(c); continue; }
        if c == '"' && in_string { in_string = false; current.push(c); continue; }
        if in_string { current.push(c); continue; }
        match c {
            '[' | '{' => { depth += 1; current.push(c); }
            ']' | '}' => { depth -= 1; current.push(c); }
            d if d == delimiter && depth == 0 => {
                result.push(current.trim().to_string());
                current = String::new();
            }
            _ => { current.push(c); }
        }
    }
    let trimmed = current.trim().to_string();
    if !trimmed.is_empty() { result.push(trimmed); }
    result
}

impl Interpreter {
    pub fn new(program: &HirProgram) -> Self {
        Interpreter {
            scopes: vec![HashMap::new()],
            functions: program.functions.clone(),
            generic_funcs: HashMap::new(),
            methods: program.methods.clone(),
            modules: program.modules.clone(),
            trait_impls: program.trait_impls.clone(),
            limits: None,
            arena: Arena::new(DEFAULT_ARENA_CAPACITY),
            tape: None,
            recording: false,
        }
    }

    /// Convenience: access the global (bottom) scope.
    pub fn globals(&self) -> &HashMap<String, Value> {
        &self.scopes[0]
    }

    /// Convenience: mutable access to the current (top) scope.
    fn current_scope(&mut self) -> &mut HashMap<String, Value> {
        self.scopes.last_mut().unwrap()
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
        self.current_scope().insert(
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

        // Autodiff builtins
        self.current_scope().insert(
            "start_grad".to_string(),
            Value::FnRef {
                name: "start_grad".to_string(),
                params: vec![],
                return_type: Type::unit(),
            },
        );
        self.current_scope().insert(
            "new_grad".to_string(),
            Value::FnRef {
                name: "new_grad".to_string(),
                params: vec![],
                return_type: Type::unit(),
            },
        );
        self.current_scope().insert(
            "stop_grad".to_string(),
            Value::FnRef {
                name: "stop_grad".to_string(),
                params: vec![],
                return_type: Type::unit(),
            },
        );
        self.current_scope().insert(
            "zero_grad".to_string(),
            Value::FnRef {
                name: "zero_grad".to_string(),
                params: vec![],
                return_type: Type::unit(),
            },
        );
        self.current_scope().insert(
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
        // Scalar math
        for name in &["abs", "sqrt", "sin", "cos", "ln", "pow"] {
            self.current_scope().insert(
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
            self.current_scope().insert(
                name.to_string(),
                Value::FnRef {
                    name: name.to_string(),
                    params: vec![("dims".to_string(), Type::Unknown)],
                    return_type: Type::Tensor { dtype: BaseType::F64, dims: vec![Dim::Any] },
                },
            );
        }
        // Serialization
        for name in &["save_weights", "load_weights"] {
            self.current_scope().insert(
                name.to_string(),
                Value::FnRef {
                    name: name.to_string(),
                    params: vec![("path".to_string(), Type::Unknown)],
                    return_type: Type::Unknown,
                },
            );
        }
        self.current_scope().insert(
            "param".to_string(),
            Value::FnRef {
                name: "param".to_string(),
                params: vec![("t".to_string(), Type::Unknown)],
                return_type: Type::Unknown,
            },
        );
        self.current_scope().insert(
            "backward".to_string(),
            Value::FnRef {
                name: "backward".to_string(),
                params: vec![("loss".to_string(), Type::Unknown)],
                return_type: Type::unit(),
            },
        );
        self.current_scope().insert(
            "grad".to_string(),
            Value::FnRef {
                name: "grad".to_string(),
                params: vec![("param".to_string(), Type::Unknown)],
                return_type: Type::Unknown,
            },
        );

        for func in &program.functions {
            let params = func.params.clone();
            let ret = func.return_type.clone();
            self.current_scope().insert(
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
                        self.current_scope().insert(
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

    fn resolve_var(&self, name: &str) -> Option<Value> {
        for scope in self.scopes.iter().rev() {
            match scope.get(name) {
                Some(Value::Shared(rc)) => return Some(rc.borrow().clone()),
                Some(Value::Moved) => return None,
                Some(other) => return Some(other.clone()),
                None => continue,
            }
        }
        None
    }

    fn set_var(&mut self, name: String, val: Value) {
        // Guard: check variable count before inserting a new one
        let total_vars: usize = self.scopes.iter().map(|s| s.len()).sum();
        if let Some(ref limits) = self.limits {
            if !self.scopes.iter().any(|s| s.contains_key(&name)) {
                if let Err(msg) = limits.guard_vars(total_vars) {
                    eprintln!("[limits] variable limit: {}", msg);
                    if cfg!(feature = "mem-strict") {
                        panic!("variable limit exceeded: {}", msg);
                    }
                }
            }
        }
        // Check all scopes for Shared to update in-place
        for scope in self.scopes.iter_mut().rev() {
            if let Some(Value::Shared(rc)) = scope.get(&name) {
                *rc.borrow_mut() = val;
                return;
            }
        }
        // Otherwise, insert/overwrite in current scope
        self.current_scope().insert(name, val);
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
                if self.scopes.iter().any(|s| matches!(s.get(name), Some(Value::Moved))) {
                    return Err(TenthError::RuntimeError {
                        message: format!("use of moved value '{}'", name),
                    });
                }
                self.resolve_var(name)
                    .or_else(|| {
                        match name.as_str() {
                            "println" | "eprintln" | "tensor" | "rand" | "randn"
                            | "read_file" | "write_file" | "write_bytes" | "read_bytes" | "compile_host"
                            | "compile_program"
                            | "Vec::new" | "HashMap::new"
                            | "start_grad" | "new_grad" | "stop_grad"
                            | "param" | "backward" | "grad" | "zero_grad"
                            | "cross_entropy"
                            | "abs" | "sqrt" | "sin" | "cos" | "ln" | "pow"
                            | "zeros" | "ones"
                            | "save_weights" | "load_weights"
                            | "format" | "parse_int" | "parse_float"
                            | "path_join" | "path_exists" | "path_is_file" | "path_is_dir"
                            | "mkdir" | "list_dir" | "file_size" | "remove_file" | "copy_file"
                            | "time_now" | "time_now_ms" | "time_date" | "time_time" | "time_datetime" | "time_sleep_ms"
                            | "random_int" | "random_float"
                            | "math_tan" | "math_asin" | "math_acos" | "math_atan" | "math_atan2"
                            | "math_sinh" | "math_cosh" | "math_tanh" | "math_log10" | "math_log2" | "math_exp" | "math_pow"
                            | "math_floor" | "math_ceil" | "math_round"
                            | "cli_args_count" | "cli_arg"
                            | "json_encode" | "json_encode_pretty" | "json_decode" => {
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

                self.scopes.push(HashMap::new());
                for ((pname, _), arg) in template.params.iter().zip(arg_values.iter()) {
                    self.current_scope().insert(pname.clone(), arg.clone());
                }

                let result = self.eval_expr(&template.body);

                self.scopes.pop();
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
                self.eval_field(&t, field)
            }

            HirExprKind::ArrayLiteral { elements, .. } => {
                let mut vals = Vec::new();
                for elem in elements {
                    let v = self.eval_expr(elem)?.ok_or_else(|| TenthError::RuntimeError {
                        message: "array element is void".into(),
                    })?;
                    // Wrap in Shared so elements can be mutated via indexed assignment
                    vals.push(Value::Shared(Rc::new(RefCell::new(v))));
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
                    Value::MutRef(weak) => {
                        let rc = weak.upgrade().ok_or_else(|| TenthError::RuntimeError {
                            message: "cannot assign through dangling &mut reference".into(),
                        })?;
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
                    Value::MutRef(weak) => {
                        let rc = weak.upgrade().ok_or_else(|| TenthError::RuntimeError {
                            message: "cannot assign through dangling &mut reference".into(),
                        })?;
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
                    message: "ref operand is void".into(),
                })?;
                Ok(Some(Value::Ref(Rc::new(RefCell::new(val)))))
            }

            HirExprKind::MutRef(inner) => {
                let val = self.eval_expr(inner)?.ok_or_else(|| TenthError::RuntimeError {
                    message: "mut ref operand is void".into(),
                })?;
                if let HirExprKind::Var(var_name) = &inner.kind {
                    // Reuse existing Shared wrapper, or create new one in current scope
                    let rc = if let Some(Value::Shared(existing)) = self.resolve_var(var_name) {
                        existing
                    } else {
                        let cell = Rc::new(RefCell::new(val));
                        self.current_scope().insert(var_name.clone(), Value::Shared(cell.clone()));
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
                    message: "deref operand is void".into(),
                })?;
                match &val {
                    Value::Ref(rc) => Ok(Some(rc.borrow().clone())),
                    Value::MutRef(weak) => {
                        let rc = weak.upgrade().ok_or_else(|| TenthError::RuntimeError {
                            message: "cannot dereference dangling &mut reference".into(),
                        })?;
                        Ok(Some(rc.borrow().clone()))
                    }
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
                                message: format!("struct has no field '{}'", field),
                            });
                        }
                        // Update scope so subsequent reads see the change
                        if let Some(vn) = target_var_name {
                            // Re-read the struct from fields to get updated value
                            let updated = Value::Struct {
                                name: String::new(),
                                fields: fields.clone(),
                            };
                            self.current_scope().insert(vn, updated);
                        }
                        return Ok(Some(Value::Unit));
                    }
                    Value::MutRef(weak) => {
                        let rc = weak.upgrade().ok_or_else(|| TenthError::RuntimeError {
                            message: "cannot assign field through dangling &mut reference".into(),
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
                                    message: format!("struct has no field '{}'", field),
                                })
                            }
                            _ => Err(TenthError::RuntimeError {
                                message: "field assignment only supported on structs".into(),
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
                    self.current_scope().insert(var_name.clone(), Value::Moved);
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
                        message: format!("struct field '{}' is void", fname),
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
                        message: format!("enum field '{}' is void", fname),
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
                    message: "match scrutinee is void".into(),
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

    fn eval_field(&self, val: &Value, field: &str) -> TenthResult<Option<Value>> {
        // Auto-dereference Ref/MutRef/Shared to reach the struct/enum
        let v = match val {
            Value::Ref(rc) => {
                let inner = rc.borrow();
                return self.eval_field(&inner, field);
            }
            Value::MutRef(weak) => {
                if let Some(rc) = weak.upgrade() {
                    let inner = rc.borrow();
                    return self.eval_field(&inner, field);
                }
                return Err(TenthError::RuntimeError {
                    message: format!("cannot access field '{}' on dangling &mut reference", field),
                });
            }
            Value::Shared(rc) => {
                let inner = rc.borrow();
                return self.eval_field(&inner, field);
            }
            v => v,
        };

        match v {
            Value::Struct { fields, .. } => {
                for (fname, fval) in fields.borrow().iter() {
                    if fname == field {
                        return Ok(Some(fval.clone()));
                    }
                }
                Err(TenthError::RuntimeError {
                    message: format!("struct has no field '{}'", field),
                })
            }
            Value::Enum { fields, .. } => {
                for (fname, fval) in fields.borrow().iter() {
                    if fname == field {
                        return Ok(Some(fval.clone()));
                    }
                }
                Err(TenthError::RuntimeError {
                    message: format!("enum variant has no field '{}'", field),
                })
            }
            Value::Vec(items) => {
                // Allow .len() on Vec — handled in MethodCall, but also allow field-style access
                if field == "len" {
                    return Ok(Some(Value::Int(items.borrow().len() as i64)));
                }
                Err(TenthError::RuntimeError {
                    message: format!("Vec has no field '{}'", field),
                })
            }
            _ => Err(TenthError::RuntimeError {
                message: format!("cannot access field '{}' on {:?}", field, v),
            }),
        }
    }

    /// Convert a ReturnValue error back into a normal Ok result.
    fn unwrap_return(result: TenthResult<Option<Value>>) -> TenthResult<Option<Value>> {
        match result {
            Err(TenthError::ReturnValue(v)) => Ok(Some(v)),
            Err(TenthError::TryPropagate(err_val)) => Ok(Some(Value::Enum {
                enum_name: "Result".to_string(),
                variant: "Err".to_string(),
                fields: Rc::new(RefCell::new(vec![("_0".to_string(), err_val)])),
            })),
            other => other,
        }
    }

    fn pattern_matches(&self, pattern: &HirPattern, val: &Value) -> bool {
        match pattern {
            HirPattern::Wildcard => true,
            HirPattern::Binding(_) => true,
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
            HirPattern::Tuple(patterns) => {
                match val {
                    Value::Tuple(items) if items.len() == patterns.len() => {
                        patterns.iter().zip(items.iter())
                            .all(|(p, v)| self.pattern_matches(p, v))
                    }
                    Value::Vec(items) => {
                        let items_ref = items.borrow();
                        items_ref.len() == patterns.len()
                            && patterns.iter().zip(items_ref.iter())
                                .all(|(p, v)| self.pattern_matches(p, v))
                    }
                    _ => false,
                }
            }
            HirPattern::Range { start, end, inclusive } => {
                match val {
                    Value::Int(n) => {
                        if *inclusive {
                            *n >= *start && *n <= *end
                        } else {
                            *n >= *start && *n < *end
                        }
                    }
                    _ => false,
                }
            }
        }
    }

    /// Bind variables from a matched pattern into the current scope.
    fn bind_pattern(&mut self, pattern: &HirPattern, val: &Value) {
        match pattern {
            HirPattern::Binding(name) => {
                self.current_scope().insert(name.clone(), val.clone());
            }
            HirPattern::EnumVariant { field_bind, tuple_binds, .. } => {
                if let Value::Enum { fields, .. } = val {
                    let fields_ref = fields.borrow();
                    if let Some((_fname, bname)) = field_bind {
                        if let Some((_, v)) = fields_ref.first() {
                            self.current_scope().insert(bname.clone(), v.clone());
                        }
                    }
                    for (field_name, bind_name) in tuple_binds {
                        if let Some((_, v)) = fields_ref.iter().find(|(n, _)| n == field_name) {
                            self.current_scope().insert(bind_name.clone(), v.clone());
                        }
                    }
                }
            }
            HirPattern::Tuple(patterns) => {
                match val {
                    Value::Tuple(items) => {
                        for (p, v) in patterns.iter().zip(items.iter()) {
                            self.bind_pattern(p, v);
                        }
                    }
                    Value::Vec(items) => {
                        let items_ref = items.borrow();
                        for (p, v) in patterns.iter().zip(items_ref.iter()) {
                            self.bind_pattern(p, v);
                        }
                    }
                    _ => {}
                }
            }
            HirPattern::Wildcard | HirPattern::Literal(_) | HirPattern::Range { .. } => {}
        }
    }

    /// Remove variables bound by a pattern from the current scope.
    fn unbind_pattern(&mut self, pattern: &HirPattern) {
        match pattern {
            HirPattern::Binding(name) => {
                self.current_scope().remove(name);
            }
            HirPattern::EnumVariant { field_bind, tuple_binds, .. } => {
                if let Some((_, bname)) = field_bind {
                    self.current_scope().remove(bname);
                }
                for (_, bind_name) in tuple_binds {
                    self.current_scope().remove(bind_name);
                }
            }
            HirPattern::Tuple(patterns) => {
                for p in patterns {
                    self.unbind_pattern(p);
                }
            }
            HirPattern::Wildcard | HirPattern::Literal(_) | HirPattern::Range { .. } => {}
        }
    }

    // ── autodiff recording helpers ───────────────────────────────────

    fn record_binary(&mut self, op: TapeOp, t1: &Rc<RefCell<Tensor>>, t2: &Rc<RefCell<Tensor>>, result: &Rc<RefCell<Tensor>>) {
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

    fn record_unary(&mut self, op: TapeOp, input: &Rc<RefCell<Tensor>>, result: &Rc<RefCell<Tensor>>) {
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

    fn eval_binary(&mut self, op: &BinOp, l: &Value, r: &Value) -> TenthResult<Value> {
        match op {
            BinOp::Add => match (l, r) {
                (Value::Int(a), Value::Int(b)) => Ok(Value::Int(a + b)),
                (Value::Float(a), Value::Float(b)) => Ok(Value::Float(a + b)),
                (Value::Int(a), Value::Float(b)) => Ok(Value::Float(*a as f64 + b)),
                (Value::Float(a), Value::Int(b)) => Ok(Value::Float(a + *b as f64)),
                (Value::String(a), Value::String(b)) => Ok(Value::String(format!("{}{}", a, b))),
                (Value::Tensor(t1), Value::Tensor(t2)) => {
                    let result_tensor = t1.borrow().add_tensor(&t2.borrow())
                        .map_err(|msg| TenthError::RuntimeError { message: msg })?;
                    let result = Rc::new(RefCell::new(result_tensor));
                    if self.recording {
                        self.record_binary(TapeOp::Add, t1, t2, &result);
                    }
                    Ok(Value::Tensor(result))
                }
                (Value::Tensor(t), Value::Float(s)) => {
                    let result = Rc::new(RefCell::new(t.borrow().add_scalar(*s)));
                    if self.recording {
                        let scalar_tensor = Rc::new(RefCell::new(Tensor::full(&t.borrow().shape(), *s)));
                        self.record_binary(TapeOp::Add, t, &scalar_tensor, &result);
                    }
                    Ok(Value::Tensor(result))
                }
                (Value::Float(s), Value::Tensor(t)) => {
                    let result = Rc::new(RefCell::new(t.borrow().add_scalar(*s)));
                    if self.recording {
                        let scalar_tensor = Rc::new(RefCell::new(Tensor::full(&t.borrow().shape(), *s)));
                        self.record_binary(TapeOp::Add, &scalar_tensor, t, &result);
                    }
                    Ok(Value::Tensor(result))
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
                (Value::Tensor(t1), Value::Tensor(t2)) => {
                    let result_tensor = t1.borrow().sub_tensor(&t2.borrow())
                        .map_err(|msg| TenthError::RuntimeError { message: msg })?;
                    let result = Rc::new(RefCell::new(result_tensor));
                    if self.recording {
                        self.record_binary(TapeOp::Sub, t1, t2, &result);
                    }
                    Ok(Value::Tensor(result))
                }
                (Value::Tensor(t), Value::Float(s)) => {
                    let result = Rc::new(RefCell::new(t.borrow().sub_scalar(*s)));
                    if self.recording {
                        let scalar_tensor = Rc::new(RefCell::new(Tensor::full(&t.borrow().shape(), *s)));
                        self.record_binary(TapeOp::Sub, t, &scalar_tensor, &result);
                    }
                    Ok(Value::Tensor(result))
                }
                (Value::Float(s), Value::Tensor(t)) => {
                    // s - t: compute as -(t - s) but record as Sub(scalar, t)
                    // because d(s-t)/dt = -1 = d(scalar - t)/dt
                    let result = Rc::new(RefCell::new(t.borrow().sub_scalar(*s).neg()));
                    if self.recording {
                        let scalar_tensor = Rc::new(RefCell::new(Tensor::full(&t.borrow().shape(), *s)));
                        self.record_binary(TapeOp::Sub, &scalar_tensor, t, &result);
                    }
                    Ok(Value::Tensor(result))
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
                (Value::Tensor(t1), Value::Tensor(t2)) => {
                    let result_tensor = t1.borrow().mul_tensor(&t2.borrow())
                        .map_err(|msg| TenthError::RuntimeError { message: msg })?;
                    let result = Rc::new(RefCell::new(result_tensor));
                    if self.recording {
                        self.record_binary(TapeOp::Mul, t1, t2, &result);
                    }
                    Ok(Value::Tensor(result))
                }
                (Value::Tensor(t), Value::Float(s)) => {
                    let result = Rc::new(RefCell::new(t.borrow().mul_scalar(*s)));
                    if self.recording {
                        let scalar_tensor = Rc::new(RefCell::new(Tensor::full(&t.borrow().shape(), *s)));
                        self.record_binary(TapeOp::Mul, t, &scalar_tensor, &result);
                    }
                    Ok(Value::Tensor(result))
                }
                (Value::Float(s), Value::Tensor(t)) => {
                    let result = Rc::new(RefCell::new(t.borrow().mul_scalar(*s)));
                    if self.recording {
                        let scalar_tensor = Rc::new(RefCell::new(Tensor::full(&t.borrow().shape(), *s)));
                        self.record_binary(TapeOp::Mul, &scalar_tensor, t, &result);
                    }
                    Ok(Value::Tensor(result))
                }
                _ => Err(TenthError::RuntimeError {
                    message: "type mismatch in multiplication".into(),
                }),
            },
            BinOp::Div => match (l, r) {
                (Value::Int(a), Value::Int(b)) => Ok(Value::Int(a / b)),
                (Value::Float(a), Value::Float(b)) => Ok(Value::Float(a / b)),
                (Value::Int(a), Value::Float(b)) => Ok(Value::Float(*a as f64 / b)),
                (Value::Float(a), Value::Int(b)) => Ok(Value::Float(a / *b as f64)),
                (Value::Tensor(t1), Value::Tensor(t2)) => {
                    let result_tensor = t1.borrow().div_tensor(&t2.borrow())
                        .map_err(|msg| TenthError::RuntimeError { message: msg })?;
                    let result = Rc::new(RefCell::new(result_tensor));
                    if self.recording {
                        self.record_binary(TapeOp::Div, t1, t2, &result);
                    }
                    Ok(Value::Tensor(result))
                }
                (Value::Tensor(t), Value::Float(s)) => {
                    let result = Rc::new(RefCell::new(t.borrow().div_scalar(*s)));
                    if self.recording {
                        let scalar_tensor = Rc::new(RefCell::new(Tensor::full(&t.borrow().shape(), *s)));
                        self.record_binary(TapeOp::Div, t, &scalar_tensor, &result);
                    }
                    Ok(Value::Tensor(result))
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
                (Value::String(a), Value::String(b)) => Ok(Value::Bool(a < b)),
                _ => Err(TenthError::RuntimeError {
                    message: "comparison requires numeric types".into(),
                }),
            },
            BinOp::Gt => match (l, r) {
                (Value::Int(a), Value::Int(b)) => Ok(Value::Bool(a > b)),
                (Value::Float(a), Value::Float(b)) => Ok(Value::Bool(a > b)),
                (Value::Int(a), Value::Float(b)) => Ok(Value::Bool((*a as f64) > *b)),
                (Value::Float(a), Value::Int(b)) => Ok(Value::Bool(*a > *b as f64)),
                (Value::String(a), Value::String(b)) => Ok(Value::Bool(a > b)),
                _ => Err(TenthError::RuntimeError {
                    message: "comparison requires numeric types".into(),
                }),
            },
            BinOp::LtEq => match (l, r) {
                (Value::Int(a), Value::Int(b)) => Ok(Value::Bool(a <= b)),
                (Value::Float(a), Value::Float(b)) => Ok(Value::Bool(a <= b)),
                (Value::Int(a), Value::Float(b)) => Ok(Value::Bool((*a as f64) <= *b)),
                (Value::Float(a), Value::Int(b)) => Ok(Value::Bool(*a <= *b as f64)),
                (Value::String(a), Value::String(b)) => Ok(Value::Bool(a <= b)),
                _ => Err(TenthError::RuntimeError {
                    message: "comparison requires numeric types".into(),
                }),
            },
            BinOp::GtEq => match (l, r) {
                (Value::Int(a), Value::Int(b)) => Ok(Value::Bool(a >= b)),
                (Value::Float(a), Value::Float(b)) => Ok(Value::Bool(a >= b)),
                (Value::Int(a), Value::Float(b)) => Ok(Value::Bool((*a as f64) >= *b)),
                (Value::Float(a), Value::Int(b)) => Ok(Value::Bool(*a >= *b as f64)),
                (Value::String(a), Value::String(b)) => Ok(Value::Bool(a >= b)),
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

    fn value_to_string(&self, val: &Value) -> String {
        match val {
            Value::Int(n) => n.to_string(),
            Value::Float(f) => f.to_string(),
            Value::Bool(b) => b.to_string(),
            Value::String(s) => s.clone(),
            Value::Unit => "()".to_string(),
            Value::Enum { enum_name, variant, fields } => {
                let borrowed = fields.borrow();
                if borrowed.is_empty() {
                    format!("{}::{}", enum_name, variant)
                } else if borrowed.len() == 1 {
                    format!("{}::{}({})", enum_name, variant, self.value_to_string(&borrowed[0].1))
                } else {
                    let inner: Vec<String> = borrowed.iter().map(|(_, v)| self.value_to_string(v)).collect();
                    format!("{}::{}({})", enum_name, variant, inner.join(", "))
                }
            }
            Value::Vec(v) => {
                let items: Vec<String> = v.borrow().iter().map(|x| self.value_to_string(x)).collect();
                format!("[{}]", items.join(", "))
            }
            Value::Array(v) => {
                let items: Vec<String> = v.borrow().iter().map(|x| self.value_to_string(x)).collect();
                format!("[{}]", items.join(", "))
            }
            Value::Map(m) => {
                let items: Vec<String> = m.borrow().iter()
                    .map(|(k, v)| format!("{}: {}", k, self.value_to_string(v)))
                    .collect();
                format!("{{{}}}", items.join(", "))
            }
            Value::Tensor(t) => format!("{:?}", t.borrow().data),
            Value::Closure { .. } => "<closure>".to_string(),
            Value::FnRef { name, .. } => format!("<fn {}>", name),
            Value::Struct { name, fields } => {
                let borrowed = fields.borrow();
                let items: Vec<String> = borrowed.iter()
                    .map(|(k, v)| format!("{}: {}", k, self.value_to_string(v)))
                    .collect();
                format!("{} {{{}}}", name, items.join(", "))
            }
            Value::Ref(r) => self.value_to_string(&r.borrow()),
            Value::MutRef(r) => {
                if let Some(rc) = r.upgrade() {
                    self.value_to_string(&rc.borrow())
                } else {
                    "<dangling mut ref>".to_string()
                }
            }
            Value::Shared(r) => self.value_to_string(&r.borrow()),
            Value::Moved => "<moved>".to_string(),
            Value::Range { start, end, inclusive } => {
                if *inclusive { format!("{}..={}", start, end) } else { format!("{}..{}", start, end) }
            }
            Value::Iterator(_) => "<iterator>".to_string(),
            Value::Tuple(items) => {
                let strs: Vec<String> = items.iter().map(|x| self.value_to_string(x)).collect();
                format!("({})", strs.join(", "))
            }
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
            UnaryOp::Try => {
                // `expr?` — if Result::Err, propagate early return; if Result::Ok, unwrap
                match val {
                    Value::Enum { enum_name, variant, fields } => {
                        if enum_name == "Result" {
                            if variant == "Ok" {
                                // Unwrap: return the inner value
                                let borrowed = fields.borrow();
                                if let Some((_, v)) = borrowed.first() {
                                    return Ok(v.clone());
                                }
                                return Ok(Value::Unit);
                            } else if variant == "Err" {
                                // Propagate: return the Err as a special early-return signal
                                return Err(TenthError::TryPropagate(val.clone()));
                            }
                        }
                        // Non-Result: just pass through
                        Ok(val.clone())
                    }
                    _ => Ok(val.clone()),
                }
            }
        }
    }

    fn eval_method_call(&mut self, recv: &Value, method: &str, args: &[Value]) -> TenthResult<Option<Value>> {
        // Auto-dereference Ref/MutRef/Shared to reach the inner value
        let recv = match recv {
            Value::Ref(rc) => {
                let inner = rc.borrow();
                return self.eval_method_call(&inner, method, args);
            }
            Value::MutRef(weak) => {
                if let Some(rc) = weak.upgrade() {
                    let inner = rc.borrow();
                    return self.eval_method_call(&inner, method, args);
                }
                return Err(TenthError::RuntimeError {
                    message: format!("method '{}' on dangling &mut reference", method),
                });
            }
            Value::Shared(rc) => {
                let inner = rc.borrow();
                return self.eval_method_call(&inner, method, args);
            }
            other => other,
        };
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
            Value::String(_) | Value::Vec(_) | Value::Map(_) | Value::Range { .. } | Value::Iterator(_) => {
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
            Value::Range { start, end, inclusive } => self.eval_range_method(*start, *end, *inclusive, method, args),
            Value::Iterator(iter) => self.eval_iterator_method(iter, method, args),
            _ => Err(TenthError::RuntimeError {
                message: format!("native method '{}' not available", method),
            }),
        }
    }

    fn eval_string_method(&self, s: &str, method: &str, args: &[Value]) -> TenthResult<Option<Value>> {
        match method {
            "len" => Ok(Some(Value::Int(s.chars().count() as i64))),
            "trim" => Ok(Some(Value::String(s.trim().to_string()))),
            "to_upper" => Ok(Some(Value::String(s.to_uppercase()))),
            "to_lower" => Ok(Some(Value::String(s.to_lowercase()))),
            "replace" => {
                if args.len() >= 2 {
                    if let (Value::String(from), Value::String(to)) = (&args[0], &args[1]) {
                        return Ok(Some(Value::String(s.replace(from.as_str(), to.as_str()))));
                    }
                }
                Err(TenthError::RuntimeError {
                    message: "replace() takes 2 string arguments".into(),
                })
            }
            "split" => {
                if let Some(Value::String(delim)) = args.first() {
                    let parts: Vec<Value> = s.split(delim.as_str())
                        .map(|p| Value::String(p.to_string()))
                        .collect();
                    return Ok(Some(Value::Vec(Rc::new(RefCell::new(parts)))));
                }
                Err(TenthError::RuntimeError {
                    message: "split() takes a string delimiter".into(),
                })
            }
            "substring" => {
                if args.len() >= 2 {
                    let start = args[0].as_int().unwrap_or(0).max(0) as usize;
                    let len = args[1].as_int().unwrap_or(0).max(0) as usize;
                    let chars: Vec<char> = s.chars().collect();
                    let end = (start + len).min(chars.len());
                    let sub: String = chars[start..end].iter().collect();
                    return Ok(Some(Value::String(sub)));
                }
                Err(TenthError::RuntimeError {
                    message: "substring() takes start and length".into(),
                })
            }
            "contains" => {
                if let Some(Value::String(sub)) = args.first() {
                    return Ok(Some(Value::Bool(s.contains(sub.as_str()))));
                }
                Err(TenthError::RuntimeError {
                    message: "contains() takes a string argument".into(),
                })
            }
            "find" => {
                if let Some(Value::String(sub)) = args.first() {
                    return Ok(Some(Value::Int(s.find(sub.as_str()).map(|i| i as i64).unwrap_or(-1))));
                }
                Err(TenthError::RuntimeError {
                    message: "find() takes a string argument".into(),
                })
            }
            "starts_with" => {
                if let Some(Value::String(prefix)) = args.first() {
                    return Ok(Some(Value::Bool(s.starts_with(prefix.as_str()))));
                }
                Err(TenthError::RuntimeError {
                    message: "starts_with() takes a string argument".into(),
                })
            }
            "ends_with" => {
                if let Some(Value::String(suffix)) = args.first() {
                    return Ok(Some(Value::Bool(s.ends_with(suffix.as_str()))));
                }
                Err(TenthError::RuntimeError {
                    message: "ends_with() takes a string argument".into(),
                })
            }
            "parse_int" => {
                return Ok(Some(Value::Int(s.trim().parse::<i64>().unwrap_or(0))));
            }
            "parse_float" => {
                return Ok(Some(Value::Float(s.trim().parse::<f64>().unwrap_or(0.0))));
            }
            "is_empty" => {
                return Ok(Some(Value::Bool(s.is_empty())));
            }
            "repeat" => {
                if let Some(arg) = args.first() {
                    let n = arg.as_int().unwrap_or(0).max(0) as usize;
                    return Ok(Some(Value::String(s.repeat(n))));
                }
                Err(TenthError::RuntimeError {
                    message: "repeat() takes an integer argument".into(),
                })
            }
            "chars" => {
                let chars: Vec<Value> = s.chars().map(|c| Value::String(c.to_string())).collect();
                Ok(Some(Value::Vec(Rc::new(RefCell::new(chars)))))
            }
            "bytes" => {
                let bytes: Vec<Value> = s.bytes().map(|b| Value::Int(b as i64)).collect();
                Ok(Some(Value::Vec(Rc::new(RefCell::new(bytes)))))
            }
            "trim_start" => Ok(Some(Value::String(s.trim_start().to_string()))),
            "trim_end" => Ok(Some(Value::String(s.trim_end().to_string()))),
            "strip_prefix" => {
                if let Some(Value::String(prefix)) = args.first() {
                    return Ok(Some(match s.strip_prefix(prefix.as_str()) {
                        Some(rest) => Value::String(rest.to_string()),
                        None => Value::String(s.to_string()),
                    }));
                }
                Err(TenthError::RuntimeError {
                    message: "strip_prefix() takes a string argument".into(),
                })
            }
            "strip_suffix" => {
                if let Some(Value::String(suffix)) = args.first() {
                    return Ok(Some(match s.strip_suffix(suffix.as_str()) {
                        Some(rest) => Value::String(rest.to_string()),
                        None => Value::String(s.to_string()),
                    }));
                }
                Err(TenthError::RuntimeError {
                    message: "strip_suffix() takes a string argument".into(),
                })
            }
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
                // Wrap in Shared so elements can be mutated via indexed assignment.
                // If the value is already Shared, use it directly to avoid double-wrapping.
                let elem = match &args[0] {
                    Value::Shared(rc) => Value::Shared(rc.clone()),
                    other => Value::Shared(Rc::new(RefCell::new(other.clone()))),
                };
                items.borrow_mut().push(elem);
                Ok(Some(Value::Unit))
            }
            "get" => {
                if args.len() != 1 {
                    return Err(TenthError::RuntimeError {
                        message: "get() takes 1 argument".into(),
                    });
                }
                let idx = args[0].as_int().unwrap_or(0) as usize;
                let vec = items.borrow();
                match vec.get(idx) {
                    Some(v) => Ok(Some(v.clone())),
                    None => Err(TenthError::RuntimeError {
                        message: format!("Vec index {} out of bounds", idx),
                    }),
                }
            }
            "pop" => {
                let mut vec = items.borrow_mut();
                match vec.pop() {
                    Some(v) => Ok(Some(v)),
                    None => Err(TenthError::RuntimeError {
                        message: "pop() on empty Vec".into(),
                    }),
                }
            }
            "set" => {
                if args.len() != 2 {
                    return Err(TenthError::RuntimeError {
                        message: "set() takes 2 arguments (index, value)".into(),
                    });
                }
                let idx = args[0].as_int().unwrap_or(0) as usize;
                let mut vec = items.borrow_mut();
                if idx < vec.len() {
                    vec[idx] = match &args[1] {
                        Value::Shared(rc) => Value::Shared(rc.clone()),
                        other => Value::Shared(Rc::new(RefCell::new(other.clone()))),
                    };
                    Ok(Some(Value::Unit))
                } else {
                    Err(TenthError::RuntimeError {
                        message: format!("Vec index {} out of bounds", idx),
                    })
                }
            }
            "clear" => {
                items.borrow_mut().clear();
                Ok(Some(Value::Unit))
            }
            "contains" => {
                if args.len() != 1 {
                    return Err(TenthError::RuntimeError {
                        message: "contains() takes 1 argument".into(),
                    });
                }
                let vec = items.borrow();
                let found = vec.iter().any(|v| {
                    let unwrapped = match v {
                        Value::Shared(rc) => rc.borrow().clone(),
                        other => other.clone(),
                    };
                    self.values_eq(&unwrapped, &args[0])
                });
                Ok(Some(Value::Bool(found)))
            }
            "index_of" => {
                if args.len() != 1 {
                    return Err(TenthError::RuntimeError {
                        message: "index_of() takes 1 argument".into(),
                    });
                }
                let vec = items.borrow();
                for (i, v) in vec.iter().enumerate() {
                    let unwrapped = match v {
                        Value::Shared(rc) => rc.borrow().clone(),
                        other => other.clone(),
                    };
                    if self.values_eq(&unwrapped, &args[0]) {
                        return Ok(Some(Value::Int(i as i64)));
                    }
                }
                Ok(Some(Value::Int(-1)))
            }
            "remove" => {
                if args.len() != 1 {
                    return Err(TenthError::RuntimeError {
                        message: "remove() takes 1 argument (index)".into(),
                    });
                }
                let idx = args[0].as_int().ok_or_else(|| TenthError::RuntimeError {
                    message: "remove() index must be an integer".into(),
                })? as usize;
                let mut vec = items.borrow_mut();
                if idx < vec.len() {
                    Ok(Some(vec.remove(idx)))
                } else {
                    Err(TenthError::RuntimeError {
                        message: format!("Vec remove index {} out of bounds", idx),
                    })
                }
            }
            "join" => {
                if args.len() != 1 {
                    return Err(TenthError::RuntimeError {
                        message: "join() takes 1 argument (delimiter)".into(),
                    });
                }
                let delim = match &args[0] {
                    Value::String(s) => s.clone(),
                    _ => return Err(TenthError::RuntimeError {
                        message: "join() delimiter must be a string".into(),
                    }),
                };
                let vec = items.borrow();
                let parts: Vec<String> = vec.iter().map(|v| {
                    match v {
                        Value::Shared(rc) => format!("{}", rc.borrow()),
                        other => format!("{}", other),
                    }
                }).collect();
                Ok(Some(Value::String(parts.join(&delim))))
            }
            "is_empty" => {
                Ok(Some(Value::Bool(items.borrow().is_empty())))
            }
            "reverse" => {
                let vec = items.borrow();
                let reversed: Vec<Value> = vec.iter().rev().cloned().collect();
                Ok(Some(Value::Vec(Rc::new(RefCell::new(reversed)))))
            }
            "slice" => {
                if args.len() != 2 {
                    return Err(TenthError::RuntimeError {
                        message: "slice() takes 2 arguments (start, end)".into(),
                    });
                }
                let start = args[0].as_int().unwrap_or(0).max(0) as usize;
                let end = args[1].as_int().unwrap_or(0).max(0) as usize;
                let vec = items.borrow();
                let end = end.min(vec.len());
                if start > end {
                    return Ok(Some(Value::Vec(Rc::new(RefCell::new(Vec::new())))));
                }
                let sliced: Vec<Value> = vec[start..end].to_vec();
                Ok(Some(Value::Vec(Rc::new(RefCell::new(sliced)))))
            }
            "extend" => {
                if args.len() != 1 {
                    return Err(TenthError::RuntimeError {
                        message: "extend() takes 1 argument (Vec)".into(),
                    });
                }
                if let Value::Vec(other) = &args[0] {
                    let other_vals = other.borrow().clone();
                    let mut vec = items.borrow_mut();
                    for v in other_vals {
                        let elem = match v {
                            Value::Shared(rc) => Value::Shared(rc),
                            other => Value::Shared(Rc::new(RefCell::new(other))),
                        };
                        vec.push(elem);
                    }
                    return Ok(Some(Value::Unit));
                }
                Err(TenthError::RuntimeError {
                    message: "extend() argument must be a Vec".into(),
                })
            }
            "sort" => {
                let mut vec = items.borrow_mut();
                vec.sort_by(|a, b| {
                    let av = match a { Value::Shared(rc) => rc.borrow().clone(), o => o.clone() };
                    let bv = match b { Value::Shared(rc) => rc.borrow().clone(), o => o.clone() };
                    match (&av, &bv) {
                        (Value::Int(x), Value::Int(y)) => x.cmp(y),
                        (Value::Float(x), Value::Float(y)) => x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal),
                        (Value::String(x), Value::String(y)) => x.cmp(y),
                        _ => std::cmp::Ordering::Equal,
                    }
                });
                Ok(Some(Value::Unit))
            }
            "dedup" => {
                let mut vec = items.borrow_mut();
                vec.dedup_by(|a, b| {
                    let av = match a { Value::Shared(rc) => rc.borrow().clone(), o => o.clone() };
                    let bv = match b { Value::Shared(rc) => rc.borrow().clone(), o => o.clone() };
                    self.values_eq(&av, &bv)
                });
                Ok(Some(Value::Unit))
            }
            "first" => {
                let vec = items.borrow();
                match vec.first() {
                    Some(v) => Ok(Some(v.clone())),
                    None => Ok(None),
                }
            }
            "last" => {
                let vec = items.borrow();
                match vec.last() {
                    Some(v) => Ok(Some(v.clone())),
                    None => Ok(None),
                }
            }
            "flatten" => {
                let vec = items.borrow();
                let mut result = Vec::new();
                for v in vec.iter() {
                    let unwrapped = match v {
                        Value::Shared(rc) => rc.borrow().clone(),
                        other => other.clone(),
                    };
                    if let Value::Vec(inner) = unwrapped {
                        for item in inner.borrow().iter() {
                            let elem = match item {
                                Value::Shared(rc) => Value::Shared(rc.clone()),
                                o => Value::Shared(Rc::new(RefCell::new(o.clone()))),
                            };
                            result.push(elem);
                        }
                    } else {
                        let elem = match v {
                            Value::Shared(rc) => Value::Shared(rc.clone()),
                            o => Value::Shared(Rc::new(RefCell::new(o.clone()))),
                        };
                        result.push(elem);
                    }
                }
                Ok(Some(Value::Vec(Rc::new(RefCell::new(result)))))
            }
            "chunks" => {
                if args.len() != 1 {
                    return Err(TenthError::RuntimeError {
                        message: "chunks() takes 1 argument (size)".into(),
                    });
                }
                let size = args[0].as_int().unwrap_or(1).max(1) as usize;
                let vec = items.borrow();
                let mut result = Vec::new();
                for chunk in vec.chunks(size) {
                    let c: Vec<Value> = chunk.to_vec();
                    result.push(Value::Vec(Rc::new(RefCell::new(c))));
                }
                Ok(Some(Value::Vec(Rc::new(RefCell::new(result)))))
            }
            // Lazy iterator methods
            "iter" => {
                Ok(Some(Value::Iterator(LazyIterator::from_vec(items))))
            }
            "map" => {
                if args.len() != 1 {
                    return Err(TenthError::RuntimeError {
                        message: "map() takes 1 argument (closure)".into(),
                    });
                }
                let iter = LazyIterator::from_vec(items);
                let iter = iter.with_transform(IteratorTransform::Map { closure: args[0].clone() });
                Ok(Some(Value::Iterator(iter)))
            }
            "filter" => {
                if args.len() != 1 {
                    return Err(TenthError::RuntimeError {
                        message: "filter() takes 1 argument (closure)".into(),
                    });
                }
                let iter = LazyIterator::from_vec(items);
                let iter = iter.with_transform(IteratorTransform::Filter { closure: args[0].clone() });
                Ok(Some(Value::Iterator(iter)))
            }
            "collect" => {
                // Vec.collect() is a no-op — already a Vec
                Ok(Some(Value::Vec(items.clone())))
            }
            _ => Err(TenthError::RuntimeError {
                message: format!("Vec has no method '{}'", method),
            }),
        }
    }

    fn eval_range_method(&self, start: i64, end: i64, inclusive: bool, method: &str, _args: &[Value]) -> TenthResult<Option<Value>> {
        match method {
            "iter" => Ok(Some(Value::Iterator(LazyIterator::from_range(start, end, inclusive)))),
            "len" => {
                let len = if inclusive { (end - start + 1).max(0) as i64 } else { (end - start).max(0) as i64 };
                Ok(Some(Value::Int(len)))
            }
            _ => Err(TenthError::RuntimeError {
                message: format!("Range has no method '{}'", method),
            }),
        }
    }

    fn eval_iterator_method(&mut self, iter: &LazyIterator, method: &str, args: &[Value]) -> TenthResult<Option<Value>> {
        match method {
            "map" => {
                if args.len() != 1 {
                    return Err(TenthError::RuntimeError {
                        message: "map() takes 1 argument (closure)".into(),
                    });
                }
                let new_iter = iter.with_transform(IteratorTransform::Map { closure: args[0].clone() });
                Ok(Some(Value::Iterator(new_iter)))
            }
            "filter" => {
                if args.len() != 1 {
                    return Err(TenthError::RuntimeError {
                        message: "filter() takes 1 argument (closure)".into(),
                    });
                }
                let new_iter = iter.with_transform(IteratorTransform::Filter { closure: args[0].clone() });
                Ok(Some(Value::Iterator(new_iter)))
            }
            "take" => {
                if args.len() != 1 {
                    return Err(TenthError::RuntimeError {
                        message: "take() takes 1 argument (n)".into(),
                    });
                }
                let n = args[0].as_int().unwrap_or(0).max(0) as usize;
                let new_iter = iter.with_transform(IteratorTransform::Take { n });
                Ok(Some(Value::Iterator(new_iter)))
            }
            "skip" => {
                if args.len() != 1 {
                    return Err(TenthError::RuntimeError {
                        message: "skip() takes 1 argument (n)".into(),
                    });
                }
                let n = args[0].as_int().unwrap_or(0).max(0) as usize;
                let new_iter = iter.with_transform(IteratorTransform::Skip { n });
                Ok(Some(Value::Iterator(new_iter)))
            }
            "collect" => {
                // Materialize the iterator: apply all transforms to each element
                let source = iter.source.borrow();
                let transforms = iter.transforms.borrow();
                let mut result = Vec::new();
                let mut seen_count: usize = 0;
                for item in source.iter() {
                    let mut val = match item {
                        Value::Shared(rc) => rc.borrow().clone(),
                        other => other.clone(),
                    };
                    let mut included = true;
                    for transform in transforms.iter() {
                        match transform {
                            IteratorTransform::Map { closure } => {
                                val = self.apply_closure(closure, &[val])?;
                            }
                            IteratorTransform::Filter { closure } => {
                                let pred_result = self.apply_closure(closure, &[val.clone()])?;
                                match pred_result {
                                    Value::Bool(true) => {},
                                    Value::Bool(false) => { included = false; break; }
                                    _ => { included = false; break; }
                                }
                            }
                            IteratorTransform::Take { n } => {
                                if result.len() >= *n {
                                    included = false;
                                    break;
                                }
                            }
                            IteratorTransform::Skip { n } => {
                                if seen_count < *n {
                                    included = false;
                                    break;
                                }
                            }
                        }
                    }
                    seen_count += 1;
                    if included {
                        result.push(Value::Shared(Rc::new(RefCell::new(val))));
                    }
                }
                Ok(Some(Value::Vec(Rc::new(RefCell::new(result)))))
            }
            "len" => {
                // Materialize to count — not ideal but correct
                let collected = self.eval_iterator_method(iter, "collect", &[])?;
                match collected {
                    Some(Value::Vec(v)) => Ok(Some(Value::Int(v.borrow().len() as i64))),
                    _ => Ok(Some(Value::Int(0))),
                }
            }
            _ => Err(TenthError::RuntimeError {
                message: format!("Iterator has no method '{}'", method),
            }),
        }
    }

    /// Apply a closure value to arguments, returning the result.
    fn apply_closure(&mut self, closure: &Value, args: &[Value]) -> TenthResult<Value> {
        match closure {
            Value::Closure { params, body, captures } => {
                self.scopes.push(captures.clone().into_iter().collect());
                for (i, (name, _)) in params.iter().enumerate() {
                    let val = args.get(i).cloned().unwrap_or(Value::Unit);
                    self.current_scope().insert(name.clone(), val);
                }
                let result = self.eval_expr(body);
                self.scopes.pop();
                match result {
                    Ok(Some(v)) => Ok(v),
                    Ok(None) => Ok(Value::Unit),
                    Err(TenthError::ReturnValue(v)) => Ok(v),
                    Err(e) => Err(e),
                }
            }
            Value::FnRef { name, .. } => {
                // Call named function
                let span = crate::lexer::token::Span { line: 0, col: 0 };
                let result = self.call_named_fn(name, args, &span)?;
                Ok(result.unwrap_or(Value::Unit))
            }
            _ => Err(TenthError::RuntimeError {
                message: format!("expected a callable, got {:?}", closure),
            }),
        }
    }

    fn eval_map_method(&mut self, m: &Rc<RefCell<HashMap<String, Value>>>, method: &str, args: &[Value]) -> TenthResult<Option<Value>> {
        match method {
            "len" => Ok(Some(Value::Int(m.borrow().len() as i64))),
            "get" => {
                if args.len() != 1 {
                    return Err(TenthError::RuntimeError {
                        message: "get() takes 1 argument".into(),
                    });
                }
                if let Value::String(key) = &args[0] {
                    Ok(m.borrow().get(key).cloned())
                } else {
                    Err(TenthError::RuntimeError {
                        message: "HashMap key must be a string".into(),
                    })
                }
            }
            "insert" => {
                if args.len() != 2 {
                    return Err(TenthError::RuntimeError {
                        message: "insert() takes 2 arguments".into(),
                    });
                }
                if let Value::String(key) = &args[0] {
                    m.borrow_mut().insert(key.clone(), args[1].clone());
                    Ok(Some(Value::Unit))
                } else {
                    Err(TenthError::RuntimeError {
                        message: "HashMap key must be a string".into(),
                    })
                }
            }
            "contains_key" => {
                if args.len() != 1 {
                    return Err(TenthError::RuntimeError {
                        message: "contains_key() takes 1 argument".into(),
                    });
                }
                if let Value::String(key) = &args[0] {
                    Ok(Some(Value::Bool(m.borrow().contains_key(key))))
                } else {
                    Err(TenthError::RuntimeError {
                        message: "HashMap key must be a string".into(),
                    })
                }
            }
            "remove" => {
                if args.len() != 1 {
                    return Err(TenthError::RuntimeError {
                        message: "remove() takes 1 argument".into(),
                    });
                }
                if let Value::String(key) = &args[0] {
                    Ok(m.borrow_mut().remove(key))
                } else {
                    Err(TenthError::RuntimeError {
                        message: "HashMap key must be a string".into(),
                    })
                }
            }
            "keys" => {
                let map = m.borrow();
                let keys: Vec<Value> = map.keys().map(|k| Value::String(k.clone())).collect();
                Ok(Some(Value::Vec(Rc::new(RefCell::new(keys)))))
            }
            "values" => {
                let map = m.borrow();
                let vals: Vec<Value> = map.values().cloned().collect();
                Ok(Some(Value::Vec(Rc::new(RefCell::new(vals)))))
            }
            "is_empty" => {
                Ok(Some(Value::Bool(m.borrow().is_empty())))
            }
            "entries" => {
                let map = m.borrow();
                let entries: Vec<Value> = map.iter().map(|(k, v)| {
                    Value::Vec(Rc::new(RefCell::new(vec![
                        Value::String(k.clone()),
                        v.clone(),
                    ])))
                }).collect();
                Ok(Some(Value::Vec(Rc::new(RefCell::new(entries)))))
            }
            "merge" => {
                if args.len() != 1 {
                    return Err(TenthError::RuntimeError {
                        message: "merge() takes 1 argument (HashMap)".into(),
                    });
                }
                if let Value::Map(other) = &args[0] {
                    let other_map = other.borrow().clone();
                    let mut map = m.borrow_mut();
                    for (k, v) in other_map {
                        map.insert(k, v);
                    }
                    return Ok(Some(Value::Unit));
                }
                Err(TenthError::RuntimeError {
                    message: "merge() argument must be a HashMap".into(),
                })
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
        self.scopes.push(HashMap::new());
        self.current_scope().insert("self".to_string(), receiver.clone());

        for ((pname, _), arg) in method_fn.params.iter().skip(1).zip(args.iter()) {
            self.current_scope().insert(pname.clone(), arg.clone());
        }

        let result = self.eval_expr(&method_fn.body);

        self.scopes.pop();

        result
    }

    fn eval_tensor_method(&mut self, recv: &Value, method: &str, args: &[Value]) -> TenthResult<Value> {
        match recv {
            Value::Tensor(t) => {
                let tensor = t.borrow();
                match method {
                    "sum" => {
                        if args.is_empty() {
                            if self.recording {
                                // Return a 1-element tensor so it can be recorded
                                let scalar = tensor.sum();
                                let result = Rc::new(RefCell::new(
                                    Tensor::from_vec(vec![scalar], vec![1])
                                ));
                                self.record_unary(TapeOp::Sum, t, &result);
                                Ok(Value::Tensor(result))
                            } else {
                                Ok(Value::Float(tensor.sum()))
                            }
                        } else {
                            let axis = args[0].as_int().unwrap_or(0) as usize;
                            let result = tensor.sum_axis(axis);
                            Ok(Value::Tensor(Rc::new(RefCell::new(result))))
                        }
                    }
                    "mean" => {
                        if self.recording {
                            let scalar = tensor.mean();
                            let result = Rc::new(RefCell::new(
                                Tensor::from_vec(vec![scalar], vec![1])
                            ));
                            self.record_unary(TapeOp::Mean, t, &result);
                            Ok(Value::Tensor(result))
                        } else {
                            Ok(Value::Float(tensor.mean()))
                        }
                    }
                    "abs" => {
                        let result_tensor = tensor.abs();
                        let result = Rc::new(RefCell::new(result_tensor));
                        Ok(Value::Tensor(result))
                    }
                    "sqrt" => {
                        let result_tensor = tensor.sqrt();
                        let result = Rc::new(RefCell::new(result_tensor));
                        Ok(Value::Tensor(result))
                    }
                    "exp" => {
                        let result_tensor = tensor.exp();
                        let result = Rc::new(RefCell::new(result_tensor));
                        if self.recording {
                            self.record_unary(TapeOp::Exp, t, &result);
                        }
                        Ok(Value::Tensor(result))
                    }
                    "log" => {
                        let result_tensor = tensor.log();
                        let result = Rc::new(RefCell::new(result_tensor));
                        if self.recording {
                            self.record_unary(TapeOp::Log, t, &result);
                        }
                        Ok(Value::Tensor(result))
                    }
                    "relu" => {
                        let result_tensor = tensor.relu();
                        let result = Rc::new(RefCell::new(result_tensor));
                        if self.recording {
                            self.record_unary(TapeOp::ReLU, t, &result);
                        }
                        Ok(Value::Tensor(result))
                    }
                    "sigmoid" => {
                        let result_tensor = tensor.sigmoid();
                        let result = Rc::new(RefCell::new(result_tensor));
                        if self.recording {
                            self.record_unary(TapeOp::Sigmoid, t, &result);
                        }
                        Ok(Value::Tensor(result))
                    }
                    "tanh" => {
                        let result_tensor = tensor.tanh();
                        let result = Rc::new(RefCell::new(result_tensor));
                        Ok(Value::Tensor(result))
                    }
                    "matmul" => {
                        if args.len() != 1 {
                            return Err(TenthError::RuntimeError {
                                message: "matmul() takes 1 argument".into(),
                            });
                        }
                        if let Value::Tensor(other) = &args[0] {
                            let result_tensor = tensor.matmul(&other.borrow())
                                .map_err(|msg| TenthError::RuntimeError { message: msg })?;
                            let result = Rc::new(RefCell::new(result_tensor));
                            if self.recording {
                                self.record_binary(TapeOp::MatMul, t, other, &result);
                            }
                            Ok(Value::Tensor(result))
                        } else {
                            Err(TenthError::RuntimeError {
                                message: "matmul() argument must be a tensor".into(),
                            })
                        }
                    }
                    "transpose" => {
                        if !args.is_empty() {
                            return Err(TenthError::RuntimeError {
                                message: "transpose() takes no arguments".into(),
                            });
                        }
                        let result_tensor = tensor.transpose().ok_or_else(|| {
                            TenthError::RuntimeError {
                                message: "transpose requires at least 2 dimensions".into(),
                            }
                        })?;
                        let result = Rc::new(RefCell::new(result_tensor));
                        if self.recording {
                            self.record_unary(TapeOp::Transpose, t, &result);
                        }
                        Ok(Value::Tensor(result))
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
                    "conv2d" => {
                        // x.conv2d(w, kernel_h, kernel_w, stride, pad)
                        if args.len() < 5 {
                            return Err(TenthError::RuntimeError {
                                message: "conv2d() takes 5 args: w, kH, kW, stride, pad".into(),
                            });
                        }
                        let k_h = args[1].as_int().unwrap_or(3) as usize;
                        let k_w = args[2].as_int().unwrap_or(3) as usize;
                        let stride = args[3].as_int().unwrap_or(1) as usize;
                        let pad = args[4].as_int().unwrap_or(0) as usize;
                        if let Value::Tensor(w_rc) = &args[0] {
                            let w_data = w_rc.borrow();
                            // im2col: (N,C,H,W) → (N*H_out*W_out, C*kH*kW)
                            let (cols, h_out, w_out) = tensor.im2col(k_h, k_w, stride, pad)
                                .ok_or_else(|| TenthError::RuntimeError {
                                    message: "im2col failed (input must be 4D)".into(),
                                })?;
                            // Reshape weight: (C_out, C_in, kH, kW) → (C_out, C_in*kH*kW)
                            let w_shape = w_data.shape();
                            let c_out = w_shape[0];
                            // matmul: cols @ w_flat^T → (N*H_out*W_out, C_out)
                            let w_flat = w_data.reshape(&[c_out, w_shape[1] * w_shape[2] * w_shape[3]])
                                .ok_or_else(|| TenthError::RuntimeError {
                                    message: "weight reshape failed".into(),
                                })?;
                            let output_2d = cols.matmul(&w_flat.transpose()
                                .ok_or_else(|| TenthError::RuntimeError {
                                    message: "weight transpose failed".into(),
                                })?).map_err(|msg| TenthError::RuntimeError { message: msg })?;
                            // Reshape output to (N, C_out, H_out, W_out)
                            let n = tensor.shape()[0];
                            let result = output_2d.reshape(&[n, c_out, h_out, w_out])
                                .ok_or_else(|| TenthError::RuntimeError {
                                    message: "output reshape failed".into(),
                                })?;
                            let result_rc = Rc::new(RefCell::new(result));
                            if self.recording {
                                let cols_rc = Rc::new(RefCell::new(cols));
                                if let Some(ref mut tape) = self.tape {
                                    let x_id = t.borrow().tape_id
                                        .unwrap_or_else(|| tape.input(t.clone()));
                                    let w_id = w_rc.borrow().tape_id
                                        .unwrap_or_else(|| tape.input(w_rc.clone()));
                                    let node_id = tape.conv2d(
                                        x_id, t.clone(),
                                        w_id, w_rc.clone(),
                                        cols_rc, result_rc.clone(),
                                    );
                                    result_rc.borrow_mut().tape_id = Some(node_id);
                                }
                            }
                            return Ok(Value::Tensor(result_rc));
                        }
                        return Err(TenthError::RuntimeError {
                            message: "conv2d: weight must be a tensor".into(),
                        });
                    }
                    "batchnorm" => {
                        // x.batchnorm(gamma, beta, eps)
                        // gamma, beta: 1D tensors of shape (C,)
                        // x: (N, C, H, W) — computes mean/var over N, H, W per channel
                        if args.len() < 3 {
                            return Err(TenthError::RuntimeError {
                                message: "batchnorm() takes gamma, beta, eps".into(),
                            });
                        }
                        let eps = args[2].as_float().unwrap_or(1e-5);
                        if let (Value::Tensor(gamma_rc), Value::Tensor(beta_rc)) = (&args[0], &args[1]) {
                            let x_shape = tensor.shape();
                            if x_shape.len() < 2 {
                                return Err(TenthError::RuntimeError {
                                    message: "batchnorm requires at least 2D input".into(),
                                });
                            }
                            let c = x_shape[1]; // channel dim
                            let n = x_shape[0];
                            let spatial: usize = x_shape[2..].iter().product();

                            let x_flat = tensor.data.as_standard_layout().to_owned();
                            let x_slice = x_flat.as_slice().unwrap_or(&[]);
                            let gamma_ref = gamma_rc.borrow();
                            let beta_ref = beta_rc.borrow();
                            let g_flat = gamma_ref.data.as_standard_layout().to_owned();
                            let b_flat = beta_ref.data.as_standard_layout().to_owned();
                            let g_slice = g_flat.as_slice().unwrap_or(&[]);
                            let b_slice = b_flat.as_slice().unwrap_or(&[]);

                            let mut result_data = Vec::with_capacity(x_slice.len());
                            let mut x_hat_data = Vec::with_capacity(x_slice.len());
                            let mut std_inv_data = Vec::with_capacity(c);

                            for ci in 0..c {
                                // Mean over N, H, W for this channel
                                let mut sum = 0.0;
                                let mut count = 0;
                                for ni in 0..n {
                                    for si in 0..spatial {
                                        let idx = ((ni * c + ci) * spatial) + si;
                                        if idx < x_slice.len() {
                                            sum += x_slice[idx];
                                            count += 1;
                                        }
                                    }
                                }
                                let mean = if count > 0 { sum / count as f64 } else { 0.0 };

                                // Variance
                                let mut var_sum = 0.0;
                                for ni in 0..n {
                                    for si in 0..spatial {
                                        let idx = ((ni * c + ci) * spatial) + si;
                                        if idx < x_slice.len() {
                                            let diff = x_slice[idx] - mean;
                                            var_sum += diff * diff;
                                        }
                                    }
                                }
                                let var = if count > 0 { var_sum / count as f64 } else { 1.0 };
                                let std_inv = 1.0 / (var + eps).sqrt();
                                std_inv_data.push(std_inv);

                                let g = g_slice.get(ci).copied().unwrap_or(1.0);
                                let b = b_slice.get(ci).copied().unwrap_or(0.0);

                                for ni in 0..n {
                                    for si in 0..spatial {
                                        let idx = ((ni * c + ci) * spatial) + si;
                                        if idx < x_slice.len() {
                                            let x_norm = (x_slice[idx] - mean) * std_inv;
                                            x_hat_data.push(x_norm);
                                            result_data.push(g * x_norm + b);
                                        }
                                    }
                                }
                            }

                            let result = Rc::new(RefCell::new(Tensor::from_vec(result_data, x_shape.clone())));
                            if self.recording {
                                let x_hat = Rc::new(RefCell::new(Tensor::from_vec(x_hat_data, x_shape)));
                                let std_inv_tensor = Rc::new(RefCell::new(
                                    Tensor::from_vec(std_inv_data, vec![c])
                                ));
                                if let Some(ref mut tape) = self.tape {
                                    let x_id = t.borrow().tape_id
                                        .unwrap_or_else(|| tape.input(t.clone()));
                                    let node_id = tape.batchnorm(
                                        x_id, t.clone(),
                                        gamma_rc.clone(), beta_rc.clone(),
                                        x_hat, std_inv_tensor, result.clone(),
                                    );
                                    result.borrow_mut().tape_id = Some(node_id);
                                }
                            }
                            return Ok(Value::Tensor(result));
                        }
                        return Err(TenthError::RuntimeError {
                            message: "batchnorm: gamma and beta must be tensors".into(),
                        });
                    }
                    "dropout" => {
                        if args.is_empty() {
                            return Err(TenthError::RuntimeError {
                                message: "dropout() takes 1 argument (rate)".into(),
                            });
                        }
                        let rate = args[0].as_float().unwrap_or(0.5);
                        // Generate inverted dropout mask
                        use rand::Rng;
                        let mut rng = rand::thread_rng();
                        let scale = 1.0 / (1.0 - rate);
                        let mask_data = tensor.data.mapv(|_| {
                            if rng.r#gen::<f64>() < rate { 0.0 } else { scale }
                        });
                        let mask = Rc::new(RefCell::new(Tensor::from_data(mask_data)));
                        let result_tensor = Tensor::from_data(&tensor.data * &mask.borrow().data);
                        let result = Rc::new(RefCell::new(result_tensor));
                        if self.recording {
                            let node_id = if let Some(ref mut tape) = self.tape {
                                let input_id = t.borrow().tape_id
                                    .unwrap_or_else(|| tape.input(t.clone()));
                                let _mask_id = tape.input(mask.clone());
                                tape.dropout(input_id, t.clone(), mask.clone(), result.clone())
                            } else {
                                0
                            };
                            result.borrow_mut().tape_id = Some(node_id);
                        }
                        Ok(Value::Tensor(result))
                    }
                    "layer_norm" => {
                        // x.layer_norm(gamma, beta, eps)
                        if args.len() < 2 {
                            return Err(TenthError::RuntimeError {
                                message: "layer_norm() takes gamma, beta, [eps]".into(),
                            });
                        }
                        let eps = args.get(2).and_then(|a| a.as_float()).unwrap_or(1e-5);
                        if let (Value::Tensor(gamma_rc), Value::Tensor(beta_rc)) = (&args[0], &args[1]) {
                            let x_shape = tensor.shape();
                            let ndim = x_shape.len();
                            if ndim == 0 || x_shape[ndim - 1] == 0 {
                                return Ok(Value::Tensor(Rc::new(RefCell::new(tensor.clone()))));
                            }
                            let axis_len = x_shape[ndim - 1];
                            let outer_len: usize = x_shape[..ndim - 1].iter().product();

                            let contiguous = tensor.data.as_standard_layout().to_owned();
                            let flat = match contiguous.as_slice() {
                                Some(s) => s.to_vec(),
                                None => tensor.data.iter().cloned().collect(),
                            };

                            let gamma_ref = gamma_rc.borrow();
                            let beta_ref = beta_rc.borrow();
                            let g_flat = gamma_ref.data.as_standard_layout().to_owned();
                            let b_flat = beta_ref.data.as_standard_layout().to_owned();
                            let g_slice = g_flat.as_slice().unwrap_or(&[]);
                            let b_slice = b_flat.as_slice().unwrap_or(&[]);

                            let mut result_data = Vec::with_capacity(flat.len());
                            let mut x_hat_data = Vec::with_capacity(flat.len());
                            let mut std_inv_data = Vec::with_capacity(outer_len);

                            for i in 0..outer_len {
                                let start = i * axis_len;
                                let slice = &flat[start..start + axis_len];
                                let mean: f64 = slice.iter().sum::<f64>() / axis_len as f64;
                                let var: f64 = slice.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / axis_len as f64;
                                let std_inv = 1.0 / (var + eps).sqrt();
                                std_inv_data.push(std_inv);
                                for j in 0..axis_len {
                                    let x_hat = (slice[j] - mean) * std_inv;
                                    x_hat_data.push(x_hat);
                                    let g = g_slice.get(j).copied().unwrap_or(1.0);
                                    let b = b_slice.get(j).copied().unwrap_or(0.0);
                                    result_data.push(g * x_hat + b);
                                }
                            }

                            let result = Rc::new(RefCell::new(Tensor::from_vec(result_data, x_shape.clone())));
                            if self.recording {
                                let x_hat = Rc::new(RefCell::new(Tensor::from_vec(x_hat_data, x_shape)));
                                let std_inv_tensor = Rc::new(RefCell::new(
                                    Tensor::from_vec(std_inv_data, vec![outer_len])
                                ));
                                if let Some(ref mut tape) = self.tape {
                                    let x_id = t.borrow().tape_id
                                        .unwrap_or_else(|| tape.input(t.clone()));
                                    let node_id = tape.layernorm(
                                        x_id, t.clone(),
                                        gamma_rc.clone(), beta_rc.clone(),
                                        x_hat, std_inv_tensor, result.clone(),
                                    );
                                    result.borrow_mut().tape_id = Some(node_id);
                                }
                            }
                            return Ok(Value::Tensor(result));
                        }
                        return Err(TenthError::RuntimeError {
                            message: "layer_norm: gamma and beta must be tensors".into(),
                        });
                    }
                    "gelu" => {
                        let result_tensor = tensor.gelu();
                        let result = Rc::new(RefCell::new(result_tensor));
                        if self.recording {
                            self.record_unary(TapeOp::Gelu, t, &result);
                        }
                        Ok(Value::Tensor(result))
                    }
                    "cat" => {
                        // x.cat(other, dim)
                        if args.is_empty() {
                            return Err(TenthError::RuntimeError {
                                message: "cat() takes at least 1 argument (other, [dim])".into(),
                            });
                        }
                        let dim = args.get(1).and_then(|a| a.as_int()).unwrap_or(0) as usize;
                        if let Value::Tensor(other) = &args[0] {
                            let result_tensor = tensor.cat(&other.borrow(), dim)
                                .map_err(|msg| TenthError::RuntimeError { message: msg })?;
                            Ok(Value::Tensor(Rc::new(RefCell::new(result_tensor))))
                        } else {
                            Err(TenthError::RuntimeError {
                                message: "cat() first argument must be a tensor".into(),
                            })
                        }
                    }
                    "masked_fill" => {
                        // x.masked_fill(mask, value)
                        if args.len() < 2 {
                            return Err(TenthError::RuntimeError {
                                message: "masked_fill() takes mask and value".into(),
                            });
                        }
                        let value = args[1].as_float().unwrap_or(0.0);
                        if let Value::Tensor(mask_rc) = &args[0] {
                            let result_tensor = tensor.masked_fill(&mask_rc.borrow(), value)
                                .map_err(|msg| TenthError::RuntimeError { message: msg })?;
                            Ok(Value::Tensor(Rc::new(RefCell::new(result_tensor))))
                        } else {
                            Err(TenthError::RuntimeError {
                                message: "masked_fill() mask must be a tensor".into(),
                            })
                        }
                    }
                    "permute" => {
                        // x.permute(dims...)
                        let dims: Vec<usize> = args.iter()
                            .map(|a| a.as_int().unwrap_or(0) as usize)
                            .collect();
                        let result_tensor = tensor.permute(&dims)
                            .map_err(|msg| TenthError::RuntimeError { message: msg })?;
                        Ok(Value::Tensor(Rc::new(RefCell::new(result_tensor))))
                    }
                    "softmax" => {
                        let result_tensor = tensor.softmax().ok_or_else(|| TenthError::RuntimeError {
                            message: "softmax failed".into(),
                        })?;
                        let result = Rc::new(RefCell::new(result_tensor));
                        if self.recording {
                            self.record_unary(TapeOp::Softmax, t, &result);
                        }
                        Ok(Value::Tensor(result))
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
                        // Elements are stored as Shared; return the Shared so
                        // field assignment can mutate through it.
                        match items.borrow().get(idx) {
                            Some(Value::Shared(rc)) => Ok(Value::Shared(rc.clone())),
                            Some(other) => Ok(other.clone()),
                            None => Err(TenthError::RuntimeError {
                                message: format!("Vec index {} out of bounds", idx),
                            }),
                        }
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
                self.scopes.push(HashMap::new());

                for ((pname, _), arg) in params.iter().zip(args.iter()) {
                    self.current_scope().insert(pname.clone(), arg.clone());
                }

                for (cap_name, cap_val) in captures {
                    self.current_scope().insert(cap_name.clone(), cap_val.clone());
                }

                let result = self.eval_expr(body);

                self.scopes.pop();

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
            "start_grad" => {
                self.tape = Some(Tape::new());
                self.recording = true;
                return Ok(Some(Value::Unit));
            }
            "param" => {
                if let Some(Value::Tensor(t)) = args.first() {
                    // Register this tensor as a leaf parameter on the tape
                    if let Some(ref mut tape) = self.tape {
                        let node_id = tape.input(t.clone());
                        t.borrow_mut().tape_id = Some(node_id);
                    }
                    return Ok(Some(Value::Tensor(t.clone())));
                }
                return Err(TenthError::RuntimeError {
                    message: "param() expects a tensor argument".into(),
                });
            }
            "backward" => {
                if let Some(Value::Tensor(loss)) = args.first() {
                    if let (Some(tape), Some(loss_id)) = (&self.tape, loss.borrow().tape_id) {
                        tape.backward(loss_id);
                    }
                    return Ok(Some(Value::Unit));
                }
                return Err(TenthError::RuntimeError {
                    message: "backward() expects a tensor argument".into(),
                });
            }
            "stop_grad" => {
                self.recording = false;
                return Ok(Some(Value::Unit));
            }
            "new_grad" => {
                self.tape = Some(Tape::new());
                self.recording = true;
                return Ok(Some(Value::Unit));
            }
            "zero_grad" => {
                if let Some(ref tape) = self.tape {
                    tape.zero_grad();
                }
                return Ok(Some(Value::Unit));
            }
            "abs" => {
                if let Some(arg) = args.first() {
                    return Ok(Some(match arg {
                        Value::Int(n) => Value::Int(n.abs()),
                        Value::Float(n) => Value::Float(n.abs()),
                        _ => return Err(TenthError::RuntimeError {
                            message: "abs() expects a numeric argument".into(),
                        }),
                    }));
                }
                return Err(TenthError::RuntimeError {
                    message: "abs() expects 1 argument".into(),
                });
            }
            "sqrt" => {
                if let Some(arg) = args.first() {
                    let n = arg.as_float().ok_or_else(|| TenthError::RuntimeError {
                        message: "sqrt() expects a numeric argument".into(),
                    })?;
                    return Ok(Some(Value::Float(n.sqrt())));
                }
                return Err(TenthError::RuntimeError {
                    message: "sqrt() expects 1 argument".into(),
                });
            }
            "sin" => {
                if let Some(arg) = args.first() {
                    let n = arg.as_float().ok_or_else(|| TenthError::RuntimeError {
                        message: "sin() expects a numeric argument".into(),
                    })?;
                    return Ok(Some(Value::Float(n.sin())));
                }
                return Err(TenthError::RuntimeError {
                    message: "sin() expects 1 argument".into(),
                });
            }
            "cos" => {
                if let Some(arg) = args.first() {
                    let n = arg.as_float().ok_or_else(|| TenthError::RuntimeError {
                        message: "cos() expects a numeric argument".into(),
                    })?;
                    return Ok(Some(Value::Float(n.cos())));
                }
                return Err(TenthError::RuntimeError {
                    message: "cos() expects 1 argument".into(),
                });
            }
            "ln" => {
                if let Some(arg) = args.first() {
                    let n = arg.as_float().ok_or_else(|| TenthError::RuntimeError {
                        message: "ln() expects a numeric argument".into(),
                    })?;
                    if n <= 0.0 {
                        return Err(TenthError::RuntimeError {
                            message: "ln() argument must be > 0".into(),
                        });
                    }
                    return Ok(Some(Value::Float(n.ln())));
                }
                return Err(TenthError::RuntimeError {
                    message: "ln() expects 1 argument".into(),
                });
            }
            "pow" => {
                if args.len() >= 2 {
                    let base = args[0].as_float().ok_or_else(|| TenthError::RuntimeError {
                        message: "pow() expects numeric arguments".into(),
                    })?;
                    let exp = args[1].as_float().ok_or_else(|| TenthError::RuntimeError {
                        message: "pow() expects numeric arguments".into(),
                    })?;
                    return Ok(Some(Value::Float(base.powf(exp))));
                }
                return Err(TenthError::RuntimeError {
                    message: "pow() expects 2 arguments".into(),
                });
            }
            "cross_entropy" => {
                if args.len() >= 2 {
                    if let (Value::Tensor(logits), Value::Tensor(target)) = (&args[0], &args[1]) {
                        let logits_data = logits.borrow();
                        let target_data = target.borrow();

                        // Compute softmax along last axis
                        let sm = logits_data.softmax().ok_or_else(|| {
                            TenthError::RuntimeError { message: "softmax failed in cross_entropy".into() }
                        })?;

                        // CE loss: -mean(sum(target * log(softmax + ε)))
                        let eps = 1e-10;
                        let sm_data = sm.data.as_standard_layout().to_owned();
                        let tgt_flat = target_data.data.as_standard_layout().to_owned();
                        let sm_slice = sm_data.as_slice().unwrap_or(&[]);
                        let tgt_slice = tgt_flat.as_slice().unwrap_or(&[]);

                        let mut loss_val = 0.0f64;
                        let n = sm_slice.len() as f64;
                        for i in 0..sm_slice.len().min(tgt_slice.len()) {
                            let p = sm_slice[i].max(eps);
                            loss_val -= tgt_slice[i] * p.ln();
                        }
                        loss_val /= n.max(1.0); // mean over all elements

                        let loss_tensor = Tensor::from_vec(vec![loss_val], vec![1]);
                        let result = Rc::new(RefCell::new(loss_tensor));

                        if self.recording {
                            // Record: input_tensors = [logits, softmax, target]
                            let sm_rc = Rc::new(RefCell::new(sm));
                            if let Some(ref mut tape) = self.tape {
                                let logits_id = logits.borrow().tape_id
                                    .unwrap_or_else(|| tape.input(logits.clone()));
                                let _sm_id = tape.input(sm_rc.clone());
                                // Create CrossEntropy node manually
                                let node_id = tape.cross_entropy(
                                    logits_id, logits.clone(),
                                    sm_rc,
                                    target.clone(),
                                    result.clone(),
                                );
                                result.borrow_mut().tape_id = Some(node_id);
                            }
                        }

                        return Ok(Some(Value::Tensor(result)));
                    }
                }
                return Err(TenthError::RuntimeError {
                    message: "cross_entropy(logits, target) expects two tensors".into(),
                });
            }
            "grad" => {
                if let Some(Value::Tensor(param)) = args.first() {
                    let p = param.borrow();
                    if let Some(ref grad) = p.grad {
                        let grad_tensor = Tensor::from_data(grad.clone());
                        return Ok(Some(Value::Tensor(Rc::new(RefCell::new(grad_tensor)))));
                    }
                    // No gradient → return zeros
                    let shape = p.shape();
                    let zeros = Tensor::zeros(&shape);
                    return Ok(Some(Value::Tensor(Rc::new(RefCell::new(zeros)))));
                }
                return Err(TenthError::RuntimeError {
                    message: "grad() expects a tensor argument".into(),
                });
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
            "zeros" => {
                let shape: Vec<usize> = args.iter()
                    .map(|a| a.as_int().unwrap_or(1) as usize)
                    .collect();
                let t = Tensor::zeros(&shape);
                return Ok(Some(Value::Tensor(Rc::new(RefCell::new(t)))));
            }
            "ones" => {
                let shape: Vec<usize> = args.iter()
                    .map(|a| a.as_int().unwrap_or(1) as usize)
                    .collect();
                let t = Tensor::ones(&shape);
                return Ok(Some(Value::Tensor(Rc::new(RefCell::new(t)))));
            }
            "save_weights" => {
                if args.len() >= 2 {
                    if let Value::String(path) = &args[0] {
                        // args[1] can be Array or Vec of tensors
                        let tensors: &Rc<RefCell<Vec<Value>>> = match &args[1] {
                            Value::Vec(v) => v,
                            Value::Array(a) => a,
                            _ => {
                                return Err(TenthError::RuntimeError {
                                    message: "save_weights expects a list of tensors".into(),
                                });
                            }
                        };
                            let tensors_ref = tensors.borrow();
                            let mut bytes: Vec<u8> = Vec::new();
                            // Header: number of tensors (i32)
                            bytes.extend(&(tensors_ref.len() as i32).to_le_bytes());
                            for val in tensors_ref.iter() {
                                // Unwrap Shared wrapper (Vec::push wraps elements in Shared)
                                let tensor_rc = match val {
                                    Value::Tensor(t) => Some(t.clone()),
                                    Value::Shared(rc) => {
                                        if let Value::Tensor(t) = &*rc.borrow() {
                                            Some(t.clone())
                                        } else { None }
                                    }
                                    _ => None,
                                };
                                if let Some(t) = tensor_rc {
                                    let t_ref = t.borrow();
                                    let shape = t_ref.shape();
                                    let ndim = shape.len() as i32;
                                    bytes.extend(&ndim.to_le_bytes());
                                    for &d in &shape {
                                        bytes.extend(&(d as i32).to_le_bytes());
                                    }
                                    let flat = t_ref.data.as_standard_layout().to_owned();
                                    if let Some(slice) = flat.as_slice() {
                                        for &x in slice {
                                            bytes.extend(&x.to_le_bytes());
                                        }
                                    }
                                }
                            }
                            let _ = std::fs::write(path, &bytes);
                            return Ok(Some(Value::Unit));
                    }
                }
                return Err(TenthError::RuntimeError {
                    message: "save_weights(path, vec_of_tensors)".into(),
                });
            }
            "load_weights" => {
                if let Some(Value::String(path)) = args.first() {
                    match std::fs::read(path) {
                        Ok(bytes) => {
                            if bytes.len() < 4 {
                                return Err(TenthError::RuntimeError {
                                    message: "load_weights: file too short".into(),
                                });
                            }
                            let num = i32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as usize;
                            let mut offset: usize = 4;
                            let mut result: Vec<Value> = Vec::new();
                            for _ in 0..num {
                                if offset + 4 > bytes.len() { break; }
                                let ndim = i32::from_le_bytes([
                                    bytes[offset], bytes[offset+1], bytes[offset+2], bytes[offset+3]
                                ]) as usize;
                                offset += 4;
                                let mut shape = Vec::new();
                                for _ in 0..ndim {
                                    if offset + 4 > bytes.len() { break; }
                                    let d = i32::from_le_bytes([
                                        bytes[offset], bytes[offset+1], bytes[offset+2], bytes[offset+3]
                                    ]) as usize;
                                    shape.push(d);
                                    offset += 4;
                                }
                                let nel: usize = shape.iter().product();
                                let data_len = nel * 8; // f64 = 8 bytes
                                if offset + data_len > bytes.len() { break; }
                                let mut data = Vec::with_capacity(nel);
                                for i in 0..nel {
                                    let start = offset + i * 8;
                                    let val = f64::from_le_bytes([
                                        bytes[start], bytes[start+1], bytes[start+2], bytes[start+3],
                                        bytes[start+4], bytes[start+5], bytes[start+6], bytes[start+7],
                                    ]);
                                    data.push(val);
                                }
                                offset += data_len;
                                result.push(Value::Tensor(Rc::new(RefCell::new(
                                    Tensor::from_vec(data, shape)
                                ))));
                            }
                            return Ok(Some(Value::Vec(Rc::new(RefCell::new(result)))));
                        }
                        Err(e) => return Err(TenthError::RuntimeError {
                            message: format!("load_weights: {}", e),
                        }),
                    }
                }
                return Err(TenthError::RuntimeError {
                    message: "load_weights(path)".into(),
                });
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
            "write_bytes" => {
                if args.len() >= 2 {
                    if let (Value::String(path), Value::Vec(bytes)) = (&args[0], &args[1]) {
                        let data: Vec<u8> = bytes.borrow().iter().filter_map(|v| {
                            if let Value::Int(n) = v { Some(*n as u8) } else { None }
                        }).collect();
                        match std::fs::write(path, &data) {
                            Ok(()) => return Ok(Some(Value::Unit)),
                            Err(e) => return Err(TenthError::RuntimeError {
                                message: format!("write_bytes failed: {}", e),
                            }),
                        }
                    }
                }
                return Err(TenthError::RuntimeError {
                    message: "write_bytes(path, bytes) expects a string and a byte vec".into(),
                });
            }
            "read_bytes" => {
                if let Some(Value::String(path)) = args.first() {
                    match std::fs::read(path) {
                        Ok(data) => {
                            let bytes: Vec<Value> = data.iter()
                                .map(|b| Value::Int(*b as i64))
                                .collect();
                            return Ok(Some(Value::Vec(Rc::new(RefCell::new(bytes)))));
                        }
                        Err(e) => return Err(TenthError::RuntimeError {
                            message: format!("read_bytes failed: {}", e),
                        }),
                    }
                }
                return Err(TenthError::RuntimeError {
                    message: "read_bytes(path) expects a string path".into(),
                });
            }
            "path_join" => {
                if args.len() >= 2 {
                    if let (Value::String(base), Value::String(rest)) = (&args[0], &args[1]) {
                        let joined = std::path::Path::new(base).join(rest);
                        return Ok(Some(Value::String(joined.to_string_lossy().to_string())));
                    }
                }
                return Err(TenthError::RuntimeError {
                    message: "path_join(base, rest) expects two string args".into(),
                });
            }
            "path_exists" => {
                if let Some(Value::String(path)) = args.first() {
                    return Ok(Some(Value::Bool(std::path::Path::new(path).exists())));
                }
                return Err(TenthError::RuntimeError {
                    message: "path_exists(path) expects a string path".into(),
                });
            }
            "path_is_file" => {
                if let Some(Value::String(path)) = args.first() {
                    return Ok(Some(Value::Bool(std::path::Path::new(path).is_file())));
                }
                return Err(TenthError::RuntimeError {
                    message: "path_is_file(path) expects a string path".into(),
                });
            }
            "path_is_dir" => {
                if let Some(Value::String(path)) = args.first() {
                    return Ok(Some(Value::Bool(std::path::Path::new(path).is_dir())));
                }
                return Err(TenthError::RuntimeError {
                    message: "path_is_dir(path) expects a string path".into(),
                });
            }
            "mkdir" => {
                if let Some(Value::String(path)) = args.first() {
                    match std::fs::create_dir_all(path) {
                        Ok(()) => return Ok(Some(Value::Unit)),
                        Err(e) => return Err(TenthError::RuntimeError {
                            message: format!("mkdir failed: {}", e),
                        }),
                    }
                }
                return Err(TenthError::RuntimeError {
                    message: "mkdir(path) expects a string path".into(),
                });
            }
            "list_dir" => {
                if let Some(Value::String(path)) = args.first() {
                    match std::fs::read_dir(path) {
                        Ok(entries) => {
                            let names: Vec<Value> = entries.filter_map(|e| {
                                e.ok().map(|entry| Value::String(entry.file_name().to_string_lossy().to_string()))
                            }).collect();
                            return Ok(Some(Value::Vec(Rc::new(RefCell::new(names)))));
                        }
                        Err(e) => return Err(TenthError::RuntimeError {
                            message: format!("list_dir failed: {}", e),
                        }),
                    }
                }
                return Err(TenthError::RuntimeError {
                    message: "list_dir(path) expects a string path".into(),
                });
            }
            "file_size" => {
                if let Some(Value::String(path)) = args.first() {
                    match std::fs::metadata(path) {
                        Ok(meta) => return Ok(Some(Value::Int(meta.len() as i64))),
                        Err(e) => return Err(TenthError::RuntimeError {
                            message: format!("file_size failed: {}", e),
                        }),
                    }
                }
                return Err(TenthError::RuntimeError {
                    message: "file_size(path) expects a string path".into(),
                });
            }
            "remove_file" => {
                if let Some(Value::String(path)) = args.first() {
                    match std::fs::remove_file(path) {
                        Ok(()) => return Ok(Some(Value::Unit)),
                        Err(e) => return Err(TenthError::RuntimeError {
                            message: format!("remove_file failed: {}", e),
                        }),
                    }
                }
                return Err(TenthError::RuntimeError {
                    message: "remove_file(path) expects a string path".into(),
                });
            }
            "copy_file" => {
                if args.len() >= 2 {
                    if let (Value::String(src), Value::String(dst)) = (&args[0], &args[1]) {
                        match std::fs::copy(src, dst) {
                            Ok(_) => return Ok(Some(Value::Unit)),
                            Err(e) => return Err(TenthError::RuntimeError {
                                message: format!("copy_file failed: {}", e),
                            }),
                        }
                    }
                }
                return Err(TenthError::RuntimeError {
                    message: "copy_file(src, dst) expects two string args".into(),
                });
            }
            "Vec::new" => return Ok(Some(Value::Vec(Rc::new(RefCell::new(Vec::new()))))),
            "HashMap::new" => return Ok(Some(Value::Map(Rc::new(RefCell::new(HashMap::new()))))),
            "compile_host" => {
                if args.len() >= 2 {
                    if let (Value::String(src), Value::String(out)) = (&args[0], &args[1]) {
                        match crate::lexer::lexer::Lexer::new(src).tokenize()
                            .and_then(|tokens| crate::parser::parser::Parser::new(tokens).parse_program())
                            .and_then(|prog| crate::hir::lower::Lowerer::new().lower_program(&prog))
                            .and_then(|hir| crate::compile::compile_to_wasm(&hir))
                        {
                            Ok(wasm_bytes) => {
                                let _ = std::fs::write(out, &wasm_bytes);
                                return Ok(Some(Value::Int(0)));
                            }
                            Err(_) => return Ok(Some(Value::Int(1))),
                        }
                    }
                }
                return Err(TenthError::RuntimeError {
                    message: "compile_host(src, out) expects two string args".into(),
                });
            }
            "compile_program" => {
                // Takes (program: Program, out_path: str) -> i64
                // Program is the struct produced by the self-hosting parser.
                if args.len() >= 2 {
                    if let Value::String(out) = &args[1] {
                        match crate::compile::compile_program_to_wasm(&args[0]) {
                            Ok(wasm_bytes) => {
                                let _ = std::fs::write(out, &wasm_bytes);
                                return Ok(Some(Value::Int(0)));
                            }
                            Err(e) => {
                                eprintln!("[compile_program] error: {}", e);
                                return Ok(Some(Value::Int(1)));
                            }
                        }
                    }
                }
                return Err(TenthError::RuntimeError {
                    message: "compile_program(program, out) expects Program struct and string path".into(),
                });
            }
            "format" => {
                if args.is_empty() {
                    return Err(TenthError::RuntimeError {
                        message: "format() expects at least a template string".into(),
                    });
                }
                if let Value::String(template) = &args[0] {
                    let mut result = String::new();
                    let mut arg_idx = 1;
                    let mut chars = template.chars().peekable();
                    while let Some(c) = chars.next() {
                        if c == '{' {
                            if chars.peek() == Some(&'{') {
                                chars.next();
                                result.push('{');
                            } else {
                                // Find closing }
                                let mut placeholder = String::new();
                                while let Some(pc) = chars.next() {
                                    if pc == '}' {
                                        break;
                                    }
                                    placeholder.push(pc);
                                }
                                if arg_idx < args.len() {
                                    result.push_str(&format!("{}", args[arg_idx]));
                                    arg_idx += 1;
                                } else {
                                    result.push('{');
                                    result.push_str(&placeholder);
                                    result.push('}');
                                }
                            }
                        } else if c == '}' {
                            if chars.peek() == Some(&'}') {
                                chars.next();
                                result.push('}');
                            } else {
                                result.push('}');
                            }
                        } else {
                            result.push(c);
                        }
                    }
                    return Ok(Some(Value::String(result)));
                }
                return Err(TenthError::RuntimeError {
                    message: "format() first argument must be a string template".into(),
                })
            }
            "parse_int" => {
                if let Some(arg) = args.first() {
                    if let Value::String(s) = arg {
                        return Ok(Some(Value::Int(s.trim().parse::<i64>().unwrap_or(0))));
                    }
                }
                return Err(TenthError::RuntimeError {
                    message: "parse_int() expects a string argument".into(),
                })
            }
            "parse_float" => {
                if let Some(arg) = args.first() {
                    if let Value::String(s) = arg {
                        return Ok(Some(Value::Float(s.trim().parse::<f64>().unwrap_or(0.0))));
                    }
                }
                return Err(TenthError::RuntimeError {
                    message: "parse_float() expects a string argument".into(),
                })
            }
            // Time functions
            "time_now" => {
                let now = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs_f64();
                return Ok(Some(Value::Float(now)));
            }
            "time_now_ms" => {
                let now = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_millis() as f64;
                return Ok(Some(Value::Float(now)));
            }
            "time_date" => {
                let secs = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs();
                let days_since_epoch = secs / 86400;
                let (year, month, day) = days_to_date(days_since_epoch);
                return Ok(Some(Value::String(format!("{}-{:02}-{:02}", year, month, day))));
            }
            "time_time" => {
                let secs = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs() % 86400;
                let h = secs / 3600;
                let m = (secs % 3600) / 60;
                let s = secs % 60;
                return Ok(Some(Value::String(format!("{}:{:02}:{:02}", h, m, s))));
            }
            "time_datetime" => {
                let secs = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs();
                let days_since_epoch = secs / 86400;
                let (year, month, day) = days_to_date(days_since_epoch);
                let day_secs = secs % 86400;
                let h = day_secs / 3600;
                let m = (day_secs % 3600) / 60;
                let s = day_secs % 60;
                return Ok(Some(Value::String(format!("{}-{:02}-{:02} {}:{:02}:{:02}", year, month, day, h, m, s))));
            }
            "time_sleep_ms" => {
                if let Some(Value::Int(ms)) = args.first() {
                    std::thread::sleep(std::time::Duration::from_millis(*ms as u64));
                    return Ok(Some(Value::Unit));
                }
                return Err(TenthError::RuntimeError {
                    message: "time_sleep_ms(ms) expects an integer".into(),
                });
            }
            // Random functions
            "random_int" => {
                if let (Some(Value::Int(lo)), Some(Value::Int(hi))) = (args.first(), args.get(1)) {
                    use std::collections::hash_map::DefaultHasher;
                    use std::hash::{Hash, Hasher};
                    let now = std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap_or_default()
                        .as_nanos();
                    let mut hasher = DefaultHasher::new();
                    now.hash(&mut hasher);
                    let rand_val = hasher.finish();
                    let range = (*hi - *lo + 1).max(1);
                    let result = *lo + ((rand_val % (range as u64)) as i64);
                    return Ok(Some(Value::Int(result)));
                }
                return Ok(Some(Value::Int(0)));
            }
            "random_float" => {
                use std::collections::hash_map::DefaultHasher;
                use std::hash::{Hash, Hasher};
                let now = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_nanos();
                let mut hasher = DefaultHasher::new();
                now.hash(&mut hasher);
                let rand_val = hasher.finish();
                let result = (rand_val as f64) / (u64::MAX as f64);
                return Ok(Some(Value::Float(result)));
            }
            // Math functions
            "math_tan" => {
                if let Some(Value::Float(x)) = args.first() {
                    return Ok(Some(Value::Float(x.tan())));
                }
                return Ok(Some(Value::Float(0.0)));
            }
            "math_asin" => {
                if let Some(Value::Float(x)) = args.first() {
                    return Ok(Some(Value::Float(x.asin())));
                }
                return Ok(Some(Value::Float(0.0)));
            }
            "math_acos" => {
                if let Some(Value::Float(x)) = args.first() {
                    return Ok(Some(Value::Float(x.acos())));
                }
                return Ok(Some(Value::Float(0.0)));
            }
            "math_atan" => {
                if let Some(Value::Float(x)) = args.first() {
                    return Ok(Some(Value::Float(x.atan())));
                }
                return Ok(Some(Value::Float(0.0)));
            }
            "math_atan2" => {
                if let (Some(Value::Float(y)), Some(Value::Float(x))) = (args.first(), args.get(1)) {
                    return Ok(Some(Value::Float(y.atan2(*x))));
                }
                return Ok(Some(Value::Float(0.0)));
            }
            "math_sinh" => {
                if let Some(Value::Float(x)) = args.first() {
                    return Ok(Some(Value::Float(x.sinh())));
                }
                return Ok(Some(Value::Float(0.0)));
            }
            "math_cosh" => {
                if let Some(Value::Float(x)) = args.first() {
                    return Ok(Some(Value::Float(x.cosh())));
                }
                return Ok(Some(Value::Float(0.0)));
            }
            "math_tanh" => {
                if let Some(Value::Float(x)) = args.first() {
                    return Ok(Some(Value::Float(x.tanh())));
                }
                return Ok(Some(Value::Float(0.0)));
            }
            "math_log10" => {
                if let Some(Value::Float(x)) = args.first() {
                    return Ok(Some(Value::Float(x.log10())));
                }
                return Ok(Some(Value::Float(0.0)));
            }
            "math_log2" => {
                if let Some(Value::Float(x)) = args.first() {
                    return Ok(Some(Value::Float(x.log2())));
                }
                return Ok(Some(Value::Float(0.0)));
            }
            "math_exp" => {
                if let Some(Value::Float(x)) = args.first() {
                    return Ok(Some(Value::Float(x.exp())));
                }
                return Ok(Some(Value::Float(0.0)));
            }
            "math_pow" => {
                if let (Some(Value::Float(base)), Some(Value::Float(exp))) = (args.first(), args.get(1)) {
                    return Ok(Some(Value::Float(base.powf(*exp))));
                }
                return Ok(Some(Value::Float(0.0)));
            }
            "math_floor" => {
                if let Some(Value::Float(x)) = args.first() {
                    return Ok(Some(Value::Float(x.floor())));
                }
                return Ok(Some(Value::Float(0.0)));
            }
            "math_ceil" => {
                if let Some(Value::Float(x)) = args.first() {
                    return Ok(Some(Value::Float(x.ceil())));
                }
                return Ok(Some(Value::Float(0.0)));
            }
            "math_round" => {
                if let Some(Value::Float(x)) = args.first() {
                    return Ok(Some(Value::Float(x.round())));
                }
                return Ok(Some(Value::Float(0.0)));
            }
            // CLI functions
            "cli_args_count" => {
                return Ok(Some(Value::Int(1))); // Default: just program name
            }
            "cli_arg" => {
                if let Some(Value::Int(_idx)) = args.first() {
                    return Ok(Some(Value::String(String::new())));
                }
                return Ok(Some(Value::String(String::new())));
            }
            // JSON functions
            "json_encode" => {
                if let Some(val) = args.first() {
                    return Ok(Some(Value::String(json_encode_value(val))));
                }
                return Ok(Some(Value::String("null".into())));
            }
            "json_encode_pretty" => {
                if let Some(val) = args.first() {
                    return Ok(Some(Value::String(json_encode_value_pretty(val, 0))));
                }
                return Ok(Some(Value::String("null".into())));
            }
            "json_decode" => {
                if let Some(Value::String(s)) = args.first() {
                    return Ok(Some(json_decode_string(s)));
                }
                return Ok(Some(Value::Unit));
            }
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
                        self.scopes.push(HashMap::new());

                        for ((pname, _), arg) in fn_def.params.iter().zip(args.iter()) {
                            self.current_scope().insert(pname.clone(), arg.clone());
                        }

                        let result = self.eval_expr(&fn_def.body);

                        self.scopes.pop();

                        return Self::unwrap_return(result);
                    }
                }
                return Err(TenthError::RuntimeError {
                    message: format!("undefined function '{}'", name),
                });
            }
        }

        let func_def = self.functions.iter().find(|f| f.name == name).cloned();
        if let Some(fd) = func_def {
            // Push a new scope for function-local variables.
            // Parameters and locals are isolated; globals remain visible via scope chain.
            self.scopes.push(HashMap::new());

            for ((pname, _), arg) in fd.params.iter().zip(args.iter()) {
                self.current_scope().insert(pname.clone(), arg.clone());
            }

            let result = self.eval_expr(&fd.body);

            self.scopes.pop();

            return Self::unwrap_return(result);
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
            HirStmtKind::Let { names, init, .. } => {
                let val = match init {
                    Some(e) => self.eval_expr(e)?.unwrap_or(Value::Unit),
                    None => Value::Unit,
                };
                if names.len() == 1 {
                    self.current_scope().insert(names[0].clone(), val);
                } else {
                    // Tuple destructuring: val should be a Vec
                    match &val {
                        Value::Vec(v) => {
                            let borrowed = v.borrow();
                            for (i, name) in names.iter().enumerate() {
                                let v = borrowed.get(i).cloned().unwrap_or(Value::Unit);
                                self.current_scope().insert(name.clone(), v);
                            }
                        }
                        Value::Array(a) => {
                            let borrowed = a.borrow();
                            for (i, name) in names.iter().enumerate() {
                                let v = borrowed.get(i).cloned().unwrap_or(Value::Unit);
                                self.current_scope().insert(name.clone(), v);
                            }
                        }
                        _ => {
                            // Fallback: assign entire value to first name
                            self.current_scope().insert(names[0].clone(), val);
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
                        message: "while condition is void".into(),
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
                    message: "for iterable is void".into(),
                })?;
                match iter_val {
                    Value::Range { start, end, inclusive } => {
                        let e = if inclusive { end + 1 } else { end };
                        for i in start..e {
                            self.current_scope().insert(var.clone(), Value::Int(i));
                            self.eval_stmt(body)?;
                        }
                    }
                    Value::Vec(items) => {
                        let vec = items.borrow();
                        for item in vec.iter() {
                            let val = match item {
                                Value::Shared(rc) => rc.borrow().clone(),
                                other => other.clone(),
                            };
                            self.current_scope().insert(var.clone(), val);
                            self.eval_stmt(body)?;
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
                            self.current_scope().insert(var.clone(), val);
                            self.eval_stmt(body)?;
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
                                self.current_scope().insert(var.clone(), val);
                                self.eval_stmt(body)?;
                            }
                        }
                    }
                    _ => {
                        return Err(TenthError::RuntimeError {
                            message: "for loop supports range, Vec, tensor, and iterator".into(),
                        });
                    }
                }
                Ok(())
            }
        }
    }
}