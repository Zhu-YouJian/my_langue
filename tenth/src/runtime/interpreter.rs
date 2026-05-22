use std::collections::HashMap;
use std::rc::Rc;
use std::cell::RefCell;
use crate::error::{TenthError, TenthResult};
use crate::hir::hir::*;
use crate::hir::types::{BaseType, Dim, Type};
use super::value::Value;
use super::tensor::Tensor;

pub struct Interpreter {
    pub variables: HashMap<String, Value>,
    functions: Vec<HirFnDef>,
}

impl Interpreter {
    pub fn new(functions: Vec<HirFnDef>) -> Self {
        Interpreter {
            variables: HashMap::new(),
            functions,
        }
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

        if let Some(ref expr) = program.main_expr {
            self.eval_expr(expr)
        } else if let Some(main_fn) = self.functions.iter().find(|f| f.name == "main") {
            let body = main_fn.body.clone();
            self.eval_expr(&body)
        } else {
            Ok(None)
        }
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
                self.variables.get(name)
                    .cloned()
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

                self.eval_method(&recv, method, &arg_values).map(Some)
            }

            HirExprKind::Index { target, indices } => {
                let t = self.eval_expr(target)?.ok_or_else(|| TenthError::RuntimeError {
                    message: "index target is void".into(),
                })?;
                self.eval_index(&t, indices).map(Some)
            }

            HirExprKind::Field { target, field: _field } => {
                let _t = self.eval_expr(target)?;
                Ok(Some(Value::Unit))
            }

            HirExprKind::ArrayLiteral { elements, .. } => {
                let mut vals = Vec::new();
                for elem in elements {
                    let v = self.eval_expr(elem)?.ok_or_else(|| TenthError::RuntimeError {
                        message: "array element is void".into(),
                    })?;
                    vals.push(v);
                }
                Ok(Some(Value::Array(vals)))
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
                let tensor = Tensor::from_vec(flat, vec![nrows, ncols]);
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
                self.variables.insert(target.clone(), v);
                Ok(Some(Value::Unit))
            }

            HirExprKind::AssignOp { target, op, value } => {
                let current = self.variables.get(target).cloned().ok_or_else(|| {
                    TenthError::RuntimeError {
                        message: format!("undefined variable '{}'", target),
                    }
                })?;
                let rhs = self.eval_expr(value)?.ok_or_else(|| TenthError::RuntimeError {
                    message: "assign-op value is void".into(),
                })?;
                let result = self.eval_binary(op, &current, &rhs)?;
                self.variables.insert(target.clone(), result);
                Ok(Some(Value::Unit))
            }

            HirExprKind::Closure { params, body } => {
                Ok(Some(Value::Closure {
                    params: params.clone(),
                    body: Rc::new((**body).clone()),
                    captures: Vec::new(),
                }))
            }

            _ => {
                Err(TenthError::RuntimeError {
                    message: format!("unimplemented expression: {:?}", expr.kind),
                })
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

    fn eval_method(&self, recv: &Value, method: &str, args: &[Value]) -> TenthResult<Value> {
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
                    _ => Err(TenthError::RuntimeError {
                        message: format!("unknown tensor method: {}", method),
                    }),
                }
            }
            _ => Err(TenthError::RuntimeError {
                message: format!("method '{}' not supported on this type", method),
            }),
        }
    }

    fn eval_index(&mut self, target: &Value, indices: &[Index]) -> TenthResult<Value> {
        match target {
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
            _ => Err(TenthError::RuntimeError {
                message: "indexing only supported on tensors".into(),
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
        // Built-in functions
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
            _ => {}
        }

        // User-defined functions
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