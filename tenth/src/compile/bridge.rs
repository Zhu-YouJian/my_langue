//! Bridge: convert the Tenth self-hosting parser's compact Program representation
//! into the Rust AST, then lower + compile to WASM.
//!
//! The compact representation uses flat Vec arrays with 1-based integer indices
//! (0 = nil) to represent the AST, avoiding recursive types in Tenth.
//! All values are cloned from the interpreter's Value tree to avoid borrowing issues.

use crate::error::{TenthError, TenthResult};
use crate::lexer::token::Span;
use crate::parser::ast as ast;
use crate::runtime::value::Value;


/// Convert a Tenth Program Value (from the self-hosting parser) into Rust AST.
pub fn compact_program_to_ast(prog_val: &Value) -> TenthResult<ast::Program> {
    let fields = clone_struct_fields(prog_val, "Program")?;

    let structs_val = extract_vec(&get_field_clone(&fields, "structs"));
    let enums_val = extract_vec(&get_field_clone(&fields, "enums"));
    let fns_val = extract_vec(&get_field_clone(&fields, "fns"));
    let main_stmts_start = get_field_i64(&fields, "main_stmts_start")?;
    let main_stmts_count = get_field_i64(&fields, "main_stmts_count")?;
    let expr_nodes = extract_vec(&get_field_clone(&fields, "expr_nodes"));
    let stmt_nodes = extract_vec(&get_field_clone(&fields, "stmt_nodes"));

    let dummy_span = Span { line: 0, col: 0 };
    let mut items: Vec<ast::Item> = Vec::new();

    // Convert struct definitions
    for s_val in &structs_val {
        items.push(convert_struct_def(s_val, &dummy_span)?);
    }

    // Convert enum definitions
    for e_val in &enums_val {
        items.push(convert_enum_def(e_val, &dummy_span)?);
    }

    // Convert function definitions
    // eprintln!("[bridge] compiling {} fns, {} exprs, {} stmts",
    //     fns_val.len(), expr_nodes.len(), stmt_nodes.len());
    // for (ei, ev) in expr_nodes.iter().enumerate() {
    //     eprintln!("[bridge]   expr[{}]: {:?}", ei+1, clone_struct_fields_opt(ev));
    // }
    for (_fi, f_val) in fns_val.iter().enumerate() {
        // eprintln!("[bridge] fn #{}", _fi);
        items.push(convert_fn_def(f_val, &expr_nodes, &stmt_nodes, &dummy_span)?);
    }

    // Convert main body statements (if any)
    if main_stmts_count > 0 {
        let start = main_stmts_start.max(1) as usize;
        let end = (start as i64 + main_stmts_count - 1) as usize;
        let body_stmts = convert_stmt_range_direct(
            &expr_nodes, &stmt_nodes, &dummy_span, start, end,
        )?;
        if !body_stmts.is_empty() {
            let main_body = ast::Expr {
                kind: ast::ExprKind::Block(body_stmts),
                span: dummy_span.clone(),
            };
            items.push(ast::Item {
                kind: ast::ItemKind::Function {
                    name: ast::Ident { name: "main".to_string(), span: dummy_span.clone() },
                    generics: Vec::new(),
                    params: Vec::new(),
                    return_type: Some(ast::TypeAnnotation::Named(ast::Ident { name: "i64".to_string(), span: dummy_span.clone() })),
                    body: main_body,
                    is_pub: false,
                    is_async: false,
                },
                span: dummy_span.clone(),
            });
        }
    }

    eprintln!("[bridge] compact_program_to_ast done, {} items", items.len());
    Ok(ast::Program { items })
}

// ── Value extraction helpers (all cloning) ─────────────────────────────────

/// Clone the fields Vec of a struct Value.
fn clone_struct_fields(val: &Value, expected_name: &str) -> TenthResult<Vec<(String, Value)>> {
    match val {
        Value::Struct { name, fields } => {
            if name != expected_name {
                return Err(TenthError::RuntimeError {
                    message: format!("期望结构体 '{}'，但得到了 '{}'", expected_name, name),
                });
            }
            Ok(fields.borrow().clone())
        }
        Value::Shared(rc) => {
            let inner = rc.borrow();
            clone_struct_fields(&inner, expected_name)
        }
        _ => Err(TenthError::RuntimeError {
            message: format!("期望结构体 '{}'，但得到了 {:?}", expected_name, val.type_of()),
        }),
    }
}

fn get_field_clone(fields: &[(String, Value)], name: &str) -> Value {
    for (fname, fval) in fields {
        if fname == name {
            return fval.clone();
        }
    }
    Value::Unit
}

fn get_field_i64(fields: &[(String, Value)], name: &str) -> TenthResult<i64> {
    match get_field_clone(fields, name) {
        Value::Int(n) => Ok(n),
        Value::Shared(rc) => {
            let inner = rc.borrow();
            match &*inner {
                Value::Int(n) => Ok(*n),
                v => Err(TenthError::RuntimeError {
                    message: format!("字段 '{}' 期望 i64，但得到了 {:?}", name, v),
                }),
            }
        }
        v => Err(TenthError::RuntimeError {
            message: format!("字段 '{}' 期望 i64，但得到了 {:?}", name, v),
        }),
    }
}

fn get_field_string(fields: &[(String, Value)], name: &str) -> TenthResult<String> {
    match get_field_clone(fields, name) {
        Value::String(s) => Ok(s),
        Value::Shared(rc) => {
            let inner = rc.borrow();
            match &*inner {
                Value::String(s) => Ok(s.clone()),
                v => Err(TenthError::RuntimeError {
                    message: format!("字段 '{}' 期望字符串，但得到了 {:?}", name, v),
                }),
            }
        }
        Value::Int(n) => Ok(n.to_string()),
        v => Err(TenthError::RuntimeError {
            message: format!("字段 '{}' 期望字符串，但得到了 {:?}", name, v),
        }),
    }
}

/// Extract Vec<Value> from a Value, unwrapping Shared wrappers.
fn extract_vec(val: &Value) -> Vec<Value> {
    match val {
        Value::Vec(rc) => {
            rc.borrow().iter().map(|v| {
                match v {
                    Value::Shared(sr) => {
                        sr.borrow().clone()
                    }
                    other => other.clone(),
                }
            }).collect()
        }
        Value::Shared(rc) => {
            extract_vec(&rc.borrow())
        }
        _ => Vec::new(),
    }
}

// ── Struct/Enum conversion ─────────────────────────────────────────────────

fn convert_struct_def(val: &Value, span: &Span) -> TenthResult<ast::Item> {
    let fields = clone_struct_fields(val, "StructDef")?;
    let name = get_field_string(&fields, "name")?;
    let fields_val = extract_vec(&get_field_clone(&fields, "fields"));

    let mut ast_fields = Vec::new();
    for f_val in &fields_val {
        let ff = clone_struct_fields(f_val, "StructField")?;
        let fname = get_field_string(&ff, "name")?;
        let ftype = get_field_string(&ff, "type_ann")?;
        ast_fields.push(ast::StructField {
            name: ast::Ident { name: fname, span: span.clone() },
            type_ann: parse_type_annotation(&ftype, span),
        });
    }

    Ok(ast::Item {
        kind: ast::ItemKind::StructDef {
            name: ast::Ident { name, span: span.clone() },
            generics: Vec::new(),
            fields: ast_fields,
            is_pub: false,
        },
        span: span.clone(),
    })
}

fn convert_enum_def(val: &Value, span: &Span) -> TenthResult<ast::Item> {
    let fields = clone_struct_fields(val, "EnumDef")?;
    let name = get_field_string(&fields, "name")?;
    let variants_val = extract_vec(&get_field_clone(&fields, "variants"));

    let mut ast_variants = Vec::new();
    for v_val in &variants_val {
        let vf = clone_struct_fields(v_val, "EnumVariant")?;
        let vname = get_field_string(&vf, "name")?;
        let vfields_val = extract_vec(&get_field_clone(&vf, "fields"));
        let mut vfields = Vec::new();
        for f_val in &vfields_val {
            let ff = clone_struct_fields(f_val, "StructField")?;
            let ffname = get_field_string(&ff, "name")?;
            let fftype = get_field_string(&ff, "type_ann")?;
            vfields.push(ast::StructField {
                name: ast::Ident { name: ffname, span: span.clone() },
                type_ann: parse_type_annotation(&fftype, span),
            });
        }
        ast_variants.push(ast::EnumVariant {
            name: ast::Ident { name: vname, span: span.clone() },
            kind: if vfields.is_empty() {
                ast::EnumVariantKind::Unit
            } else {
                ast::EnumVariantKind::Named(vfields)
            },
        });
    }

    Ok(ast::Item {
        kind: ast::ItemKind::EnumDef {
            name: ast::Ident { name, span: span.clone() },
            variants: ast_variants,
        },
        span: span.clone(),
    })
}

fn parse_type_annotation(s: &str, span: &Span) -> ast::TypeAnnotation {
    if s.is_empty() {
        return ast::TypeAnnotation::Unit;
    }
    // Stage 7: recognize composite `Tensor[<dtype>, <dims>]` form produced
    // by tenthc parser.th's parse_type_annotation (no spaces inside brackets).
    // Examples: "Tensor[T,..]", "Tensor[f32,..]", "Tensor[T,2,3]".
    if s.starts_with("Tensor[") && s.ends_with(']') && s.len() > "Tensor[]".len() {
        let inner = &s["Tensor[".len()..s.len() - 1];
        // Split at first comma — separates dtype from dims
        if let Some(comma_pos) = inner.find(',') {
            let dtype_str = inner[..comma_pos].trim();
            let dims_str = inner[comma_pos + 1..].trim();
            let dtype = Box::new(parse_type_annotation(dtype_str, span));
            let dims = parse_dim_specs(dims_str);
            return ast::TypeAnnotation::Tensor { dtype, dims };
        }
        // Form like "Tensor[T]" with no comma — treat whole inner as dtype, no dims
        let dtype_str = inner.trim();
        if !dtype_str.is_empty() {
            let dtype = Box::new(parse_type_annotation(dtype_str, span));
            return ast::TypeAnnotation::Tensor { dtype, dims: Vec::new() };
        }
    }
    ast::TypeAnnotation::Named(ast::Ident { name: s.to_string(), span: span.clone() })
}

/// Stage 7: parse dim spec list (e.g., "..", "2,3", "M,N") into Vec<DimSpec>.
/// Each comma-separated token is one of:
///   ".." → Wildcard
///   integer literal → Literal(n)
///   identifier → Symbol(s)
fn parse_dim_specs(s: &str) -> Vec<ast::DimSpec> {
    if s.is_empty() {
        return Vec::new();
    }
    s.split(',')
        .map(|tok| {
            let tok = tok.trim();
            if tok == ".." {
                ast::DimSpec::Wildcard
            } else if let Ok(n) = tok.parse::<i64>() {
                ast::DimSpec::Literal(n)
            } else {
                ast::DimSpec::Symbol(tok.to_string())
            }
        })
        .collect()
}

// ── Function definition conversion ─────────────────────────────────────────

fn convert_fn_def(
    val: &Value,
    expr_nodes: &[Value],
    stmt_nodes: &[Value],
    span: &Span,
) -> TenthResult<ast::Item> {
    let fields = clone_struct_fields(val, "FnDef")?;
    let name = get_field_string(&fields, "name")?;
    let return_type_str = get_field_string(&fields, "return_type")?;
    let body_start = get_field_i64(&fields, "body_start")?;
    let body_count = get_field_i64(&fields, "body_count")?;
    let params_val = extract_vec(&get_field_clone(&fields, "params"));

    // Convert params
    let mut ast_params = Vec::new();
    for p_val in &params_val {
        let pf = clone_struct_fields(p_val, "Param")?;
        let pname = get_field_string(&pf, "name")?;
        let ptype = get_field_string(&pf, "type_ann")?;
        ast_params.push(ast::Param {
            name: ast::Ident { name: pname, span: span.clone() },
            type_ann: parse_type_annotation(&ptype, span),
        });
    }

    // Convert body
    let body_stmts = if body_count > 0 {
        let start = body_start.max(1) as usize;
        let end = start + body_count as usize - 1;
        convert_stmt_range_direct(expr_nodes, stmt_nodes, span, start, end)?
    } else if body_start > 0 {
        let body_expr = convert_expr(body_start as usize, expr_nodes, stmt_nodes, span)?;
        vec![ast::Stmt {
            kind: ast::StmtKind::Return(Some(body_expr)),
            span: span.clone(),
        }]
    } else {
        Vec::new()
    };

    let return_type = if return_type_str.is_empty() || return_type_str == "()" {
        None
    } else {
        Some(parse_type_annotation(&return_type_str, span))
    };

    Ok(ast::Item {
        kind: ast::ItemKind::Function {
            name: ast::Ident { name, span: span.clone() },
            generics: Vec::new(),
            params: ast_params,
            return_type,
            body: ast::Expr {
                kind: ast::ExprKind::Block(body_stmts),
                span: span.clone(),
            },
            is_pub: false,
            is_async: false,
        },
        span: span.clone(),
    })
}

// ── Statement conversion ───────────────────────────────────────────────────

/// Convert a range of statement indices.
fn convert_stmt_range_direct(
    expr_nodes: &[Value],
    stmt_nodes: &[Value],
    span: &Span,
    start: usize,  // 1-based start
    end: usize,    // 1-based inclusive end
) -> TenthResult<Vec<ast::Stmt>> {
    let mut stmts = Vec::new();
    for i in start..=end {
        if i == 0 || i > stmt_nodes.len() { continue; }
        let val = &stmt_nodes[i - 1];
        if let Some(stmt) = convert_stmt(val, expr_nodes, stmt_nodes, span)? {
            stmts.push(stmt);
        }
    }
    Ok(stmts)
}

fn convert_stmt(
    val: &Value,
    expr_nodes: &[Value],
    stmt_nodes: &[Value],
    span: &Span,
) -> TenthResult<Option<ast::Stmt>> {
    let fields = match clone_struct_fields_opt(val) {
        Some(f) => f,
        None => return Ok(None),
    };

    let kind = get_field_string(&fields, "kind")?;
    let expr_idx = get_field_i64(&fields, "expr_idx")?;

    match kind.as_str() {
        "let" => {
            let var_name = get_field_string(&fields, "name")?;
            let init = if expr_idx > 0 {
                Some(convert_expr(expr_idx as usize, expr_nodes, stmt_nodes, span)?)
            } else {
                None
            };
            Ok(Some(ast::Stmt {
                kind: ast::StmtKind::Let {
                    names: vec![ast::Ident { name: var_name, span: span.clone() }],
                    type_ann: None,
                    mutable: false,
                    init,
                },
                span: span.clone(),
            }))
        }
        "expr" => {
            if expr_idx > 0 {
                let e = convert_expr(expr_idx as usize, expr_nodes, stmt_nodes, span)?;
                Ok(Some(ast::Stmt {
                    kind: ast::StmtKind::Expr(e),
                    span: span.clone(),
                }))
            } else {
                Ok(None)
            }
        }
        "return" => {
            let val_expr = if expr_idx > 0 {
                Some(convert_expr(expr_idx as usize, expr_nodes, stmt_nodes, span)?)
            } else {
                None
            };
            Ok(Some(ast::Stmt {
                kind: ast::StmtKind::Return(val_expr),
                span: span.clone(),
            }))
        }
        _ => {
            eprintln!("[bridge] unknown stmt kind: '{}'", kind);
            Ok(None)
        }
    }
}

/// Try to clone struct fields, returning None if not a struct.
fn clone_struct_fields_opt(val: &Value) -> Option<Vec<(String, Value)>> {
    match val {
        Value::Struct { fields, .. } => Some(fields.borrow().clone()),
        Value::Shared(rc) => {
            let inner = rc.borrow();
            match &*inner {
                Value::Struct { fields, .. } => Some(fields.borrow().clone()),
                _ => None,
            }
        }
        _ => None,
    }
}

// ── Expression conversion ──────────────────────────────────────────────────

fn convert_expr(
    idx: usize,  // 1-based index into expr_nodes
    expr_nodes: &[Value],
    stmt_nodes: &[Value],
    span: &Span,
) -> TenthResult<ast::Expr> {
    convert_expr_depth(idx, expr_nodes, stmt_nodes, span, 0)
}

fn convert_expr_depth(
    idx: usize,
    expr_nodes: &[Value],
    stmt_nodes: &[Value],
    span: &Span,
    depth: usize,
) -> TenthResult<ast::Expr> {
    if depth > 50 {
        return Err(TenthError::RuntimeError {
            message: format!("表达式转换递归过深，索引 {}（深度={}）", idx, depth),
        });
    }
    if idx == 0 || idx > expr_nodes.len() {
        return Err(TenthError::RuntimeError {
            message: format!("表达式索引 {} 越界（长度={}）", idx, expr_nodes.len()),
        });
    }

    let val = &expr_nodes[idx - 1];
    // if idx == 8 || depth < 5 {
    //     let k = clone_struct_fields_opt(val).and_then(|f| {
    //         f.iter().find(|(n,_)| n=="kind").map(|(_,v)| format!("{:?}", v))
    //     }).unwrap_or_default();
    //     eprintln!("[bridge] convert_expr idx={} depth={} kind={}", idx, depth, k);
    // }
    let fields = match clone_struct_fields_opt(val) {
        Some(f) => f,
        None => return Err(TenthError::RuntimeError {
            message: "表达式节点不是结构体".into(),
        }),
    };

    let kind = get_field_string(&fields, "kind")?;
    let ival = get_field_i64(&fields, "ival")?;
    let sval = get_field_string(&fields, "sval")?;
    let left = get_field_i64(&fields, "left")?;
    let right = get_field_i64(&fields, "right")?;
    let arg_start = get_field_i64(&fields, "arg_start")?;
    let arg_count = get_field_i64(&fields, "arg_count")?;
    let extra_start = get_field_i64(&fields, "extra_start")?;
    let extra_count = get_field_i64(&fields, "extra_count")?;

    match kind.as_str() {
        "int" => Ok(ast::Expr {
            kind: ast::ExprKind::Literal(ast::Literal::Int(ival)),
            span: span.clone(),
        }),
        "float" => Ok(ast::Expr {
            kind: ast::ExprKind::Literal(ast::Literal::Float(ival as f64, crate::hir::types::BaseType::F64)),
            span: span.clone(),
        }),
        "str" => Ok(ast::Expr {
            kind: ast::ExprKind::Literal(ast::Literal::String(sval)),
            span: span.clone(),
        }),
        "ident" => Ok(ast::Expr {
            kind: ast::ExprKind::Ident(ast::Ident { name: sval, span: span.clone() }),
            span: span.clone(),
        }),
        "bool" => Ok(ast::Expr {
            kind: ast::ExprKind::Literal(ast::Literal::Bool(ival != 0)),
            span: span.clone(),
        }),
        "binary" => {
            let left_expr = convert_expr_depth(left as usize, expr_nodes, stmt_nodes, span, depth + 1)?;
            let right_expr = convert_expr_depth(right as usize, expr_nodes, stmt_nodes, span, depth + 1)?;
            let op = parse_binop(&sval)?;
            Ok(ast::Expr {
                kind: ast::ExprKind::Binary {
                    op,
                    left: Box::new(left_expr),
                    right: Box::new(right_expr),
                },
                span: span.clone(),
            })
        }
        "unary" => {
            let inner = convert_expr_depth(left as usize, expr_nodes, stmt_nodes, span, depth + 1)?;
            let op = match sval.as_str() {
                "-" => ast::UnaryOp::Neg,
                "!" => ast::UnaryOp::Not,
                _ => return Err(TenthError::RuntimeError {
                    message: format!("未知的一元运算符：{}", sval),
                }),
            };
            Ok(ast::Expr {
                kind: ast::ExprKind::Unary { op, expr: Box::new(inner) },
                span: span.clone(),
            })
        }
        "call" => {
            let func_expr = ast::Expr {
                kind: ast::ExprKind::Ident(ast::Ident { name: sval, span: span.clone() }),
                span: span.clone(),
            };
            let mut args = Vec::new();
            if arg_count > 0 {
                let start = arg_start.max(1) as usize;
                let end = start + arg_count as usize;
                for i in start..end {
                    if i > 0 && i <= expr_nodes.len() {
                        args.push(convert_expr_depth(i, expr_nodes, stmt_nodes, span, depth + 1)?);
                    }
                }
            }
            Ok(ast::Expr {
                kind: ast::ExprKind::Call { func: Box::new(func_expr), args },
                span: span.clone(),
            })
        }
        "if" => {
            let cond = convert_expr_depth(left as usize, expr_nodes, stmt_nodes, span, depth + 1)?;
            let then_branch = convert_expr_depth(right as usize, expr_nodes, stmt_nodes, span, depth + 1)?;
            let else_branch = if extra_start > 0 {
                Some(convert_expr_depth(extra_start as usize, expr_nodes, stmt_nodes, span, depth + 1)?)
            } else {
                None
            };
            Ok(ast::Expr {
                kind: ast::ExprKind::If {
                    cond: Box::new(cond),
                    then_branch: Box::new(then_branch),
                    else_branch: else_branch.map(Box::new),
                },
                span: span.clone(),
            })
        }
        "assign" => {
            let target = convert_expr_depth(left as usize, expr_nodes, stmt_nodes, span, depth + 1)?;
            let value = convert_expr_depth(right as usize, expr_nodes, stmt_nodes, span, depth + 1)?;
            Ok(ast::Expr {
                kind: ast::ExprKind::Assign { target: Box::new(target), value: Box::new(value) },
                span: span.clone(),
            })
        }
        "method_call" => {
            let receiver = convert_expr_depth(left as usize, expr_nodes, stmt_nodes, span, depth + 1)?;
            let method = ast::Ident { name: sval, span: span.clone() };
            let mut args = Vec::new();
            if arg_count > 0 {
                let start = arg_start.max(1) as usize;
                let end = start + arg_count as usize;
                for i in start..end {
                    if i > 0 && i <= expr_nodes.len() {
                        args.push(convert_expr_depth(i, expr_nodes, stmt_nodes, span, depth + 1)?);
                    }
                }
            }
            Ok(ast::Expr {
                kind: ast::ExprKind::MethodCall { receiver: Box::new(receiver), method, args },
                span: span.clone(),
            })
        }
        "field" => {
            let target = convert_expr_depth(left as usize, expr_nodes, stmt_nodes, span, depth + 1)?;
            Ok(ast::Expr {
                kind: ast::ExprKind::Field {
                    target: Box::new(target),
                    field: ast::Ident { name: sval, span: span.clone() },
                },
                span: span.clone(),
            })
        }
        "index" => {
            let target = convert_expr_depth(left as usize, expr_nodes, stmt_nodes, span, depth + 1)?;
            let index = convert_expr_depth(right as usize, expr_nodes, stmt_nodes, span, depth + 1)?;
            Ok(ast::Expr {
                kind: ast::ExprKind::Index {
                    target: Box::new(target),
                    indices: vec![ast::IndexExpr::Single(index)],
                },
                span: span.clone(),
            })
        }
        "ref" => {
            let inner = convert_expr_depth(left as usize, expr_nodes, stmt_nodes, span, depth + 1)?;
            Ok(ast::Expr {
                kind: ast::ExprKind::Ref(Box::new(inner)),
                span: span.clone(),
            })
        }
        "deref" => {
            let inner = convert_expr_depth(left as usize, expr_nodes, stmt_nodes, span, depth + 1)?;
            Ok(ast::Expr {
                kind: ast::ExprKind::Deref(Box::new(inner)),
                span: span.clone(),
            })
        }
        "block" => {
            let mut stmts = Vec::new();
            if extra_count > 0 {
                let s_start = extra_start.max(1) as usize;
                let s_end = s_start + extra_count as usize - 1;
                for i in s_start..=s_end {
                    if i == 0 || i > stmt_nodes.len() { continue; }
                    let sval = &stmt_nodes[i - 1];
                    if let Some(stmt) = convert_stmt(sval, expr_nodes, stmt_nodes, span)? {
                        stmts.push(stmt);
                    }
                }
            }
            Ok(ast::Expr {
                kind: ast::ExprKind::Block(stmts),
                span: span.clone(),
            })
        }
        "return" => {
            let val_expr = if left > 0 {
                Some(convert_expr_depth(left as usize, expr_nodes, stmt_nodes, span, depth + 1)?)
            } else {
                None
            };
            Ok(ast::Expr {
                kind: ast::ExprKind::Block(vec![ast::Stmt {
                    kind: ast::StmtKind::Return(val_expr),
                    span: span.clone(),
                }]),
                span: span.clone(),
            })
        }
        _ => {
            eprintln!("[bridge] unknown expr kind: '{}' (ival={}, sval='{}', left={}, right={}, extra_start={}, extra_count={})",
                kind, ival, sval, left, right, extra_start, extra_count);
            Ok(ast::Expr {
                kind: ast::ExprKind::Literal(ast::Literal::Int(0)),
                span: span.clone(),
            })
        }
    }
}

fn parse_binop(s: &str) -> TenthResult<ast::BinOp> {
    match s {
        "+" => Ok(ast::BinOp::Add),
        "-" => Ok(ast::BinOp::Sub),
        "*" => Ok(ast::BinOp::Mul),
        "/" => Ok(ast::BinOp::Div),
        "%" => Ok(ast::BinOp::Mod),
        "==" => Ok(ast::BinOp::Eq),
        "!=" => Ok(ast::BinOp::NotEq),
        "<" => Ok(ast::BinOp::Lt),
        ">" => Ok(ast::BinOp::Gt),
        "<=" => Ok(ast::BinOp::LtEq),
        ">=" => Ok(ast::BinOp::GtEq),
        "&&" => Ok(ast::BinOp::And),
        "||" => Ok(ast::BinOp::Or),
        _ => Err(TenthError::RuntimeError {
            message: format!("未知的二元运算符：{}", s),
        }),
    }
}
