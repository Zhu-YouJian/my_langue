//! Bridge: convert the Tenth self-hosting parser's compact Program representation
//! into the Rust AST, then lower + compile to WASM.
//!
//! The compact representation uses flat Vec arrays with 1-based integer indices
//! (0 = nil) to represent the AST, avoiding recursive types in Tenth.
//! All values are cloned from the interpreter's Value tree to avoid borrowing issues.

use crate::error::{TenthError, TenthResult};
use crate::hir::types::BaseType;
use crate::lexer::token::Span;
use crate::parser::ast as ast;
use crate::runtime::value::Value;


/// Bundle of shared program arrays (all 1-based indexed, 0 = nil).
/// Passed by reference to all conversion functions to avoid deep cloning.
struct ProgArrays<'a> {
    expr_nodes: &'a [Value],
    stmt_nodes: &'a [Value],
    /// Block expression body statement indices (used by `block` expr).
    block_idxs: &'a [Value],
    /// while/for/loop body statement indices.
    loop_idxs: &'a [Value],
    /// Match arms (Value::Struct "MatchArm").
    match_arms: &'a [Value],
}

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
    let block_idxs = extract_vec(&get_field_clone(&fields, "block_idxs"));
    let loop_idxs = extract_vec(&get_field_clone(&fields, "loop_idxs"));
    let match_arms = extract_vec(&get_field_clone(&fields, "match_arms"));

    let dummy_span = Span { line: 0, col: 0 };
    let arrays = ProgArrays {
        expr_nodes: &expr_nodes,
        stmt_nodes: &stmt_nodes,
        block_idxs: &block_idxs,
        loop_idxs: &loop_idxs,
        match_arms: &match_arms,
    };
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
    for (_fi, f_val) in fns_val.iter().enumerate() {
        items.push(convert_fn_def(f_val, &arrays, &dummy_span)?);
    }

    // Convert main body statements (if any)
    if main_stmts_count > 0 {
        let start = main_stmts_start.max(1) as usize;
        let end = (start as i64 + main_stmts_count - 1) as usize;
        let body_stmts = convert_stmt_range_direct(&arrays, &dummy_span, start, end)?;
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
                    is_test: false,
                },
                span: dummy_span.clone(),
            });
        }
    }

    Ok(ast::Program { items })
}

// ── Value extraction helpers (all cloning) ─────────────────────────────────

/// Clone the fields Vec of a struct Value.
fn clone_struct_fields(val: &Value, expected_name: &str) -> TenthResult<Vec<(String, Value)>> {
    match val {
        Value::Struct { name, fields } => {
            if name != expected_name {
                return Err(TenthError::RuntimeError { line: None, col: None,
                    message: format!("期望结构体 '{}'，但得到了 '{}'", expected_name, name),
                });
            }
            Ok(fields.borrow().clone())
        }
        Value::Shared(rc) => {
            let inner = rc.borrow();
            clone_struct_fields(&inner, expected_name)
        }
        _ => Err(TenthError::RuntimeError { line: None, col: None,
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
        Value::Int(n, _) => Ok(n),
        Value::Shared(rc) => {
            let inner = rc.borrow();
            match &*inner {
                Value::Int(n, _) => Ok(*n),
                v => Err(TenthError::RuntimeError { line: None, col: None,
                    message: format!("字段 '{}' 期望 i64，但得到了 {:?}", name, v),
                }),
            }
        }
        v => Err(TenthError::RuntimeError { line: None, col: None,
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
                v => Err(TenthError::RuntimeError { line: None, col: None,
                    message: format!("字段 '{}' 期望字符串，但得到了 {:?}", name, v),
                }),
            }
        }
        Value::Int(n, _) => Ok(n.to_string()),
        v => Err(TenthError::RuntimeError { line: None, col: None,
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
            kind: ast::StructKind::Named(ast_fields),
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
    arrays: &ProgArrays,
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
            default_value: None,
            variadic: false,
        });
    }

    // Convert body
    let body_stmts = if body_count > 0 {
        let start = body_start.max(1) as usize;
        let end = start + body_count as usize - 1;
        convert_stmt_range_direct(arrays, span, start, end)?
    } else if body_start > 0 {
        let body_expr = convert_expr(body_start as usize, arrays, span)?;
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
            is_test: false,
        },
        span: span.clone(),
    })
}

// ── Statement conversion ───────────────────────────────────────────────────

/// Convert a range of statement indices (1-based, inclusive).
fn convert_stmt_range_direct(
    arrays: &ProgArrays,
    span: &Span,
    start: usize,
    end: usize,
) -> TenthResult<Vec<ast::Stmt>> {
    let mut stmts = Vec::new();
    for i in start..=end {
        if i == 0 || i > arrays.stmt_nodes.len() { continue; }
        let val = &arrays.stmt_nodes[i - 1];
        if let Some(stmt) = convert_stmt(val, arrays, span)? {
            stmts.push(stmt);
        }
    }
    Ok(stmts)
}

/// Convert statements from the loop_idxs array (used by while/for/loop bodies).
fn convert_loop_body_range(
    arrays: &ProgArrays,
    span: &Span,
    start: usize,
    end: usize,
) -> TenthResult<Vec<ast::Stmt>> {
    let mut stmts = Vec::new();
    for i in start..=end {
        if i == 0 || i > arrays.loop_idxs.len() { continue; }
        // loop_idxs stores i64 statement indices into stmt_nodes
        let stmt_idx = match &arrays.loop_idxs[i - 1] {
            Value::Int(n, _) => *n as usize,
            Value::Shared(rc) => {
                let inner = rc.borrow();
                match &*inner {
                    Value::Int(n, _) => *n as usize,
                    _ => continue,
                }
            }
            _ => continue,
        };
        if stmt_idx == 0 || stmt_idx > arrays.stmt_nodes.len() { continue; }
        let val = &arrays.stmt_nodes[stmt_idx - 1];
        if let Some(stmt) = convert_stmt(val, arrays, span)? {
            stmts.push(stmt);
        }
    }
    Ok(stmts)
}

fn convert_stmt(
    val: &Value,
    arrays: &ProgArrays,
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
                Some(convert_expr(expr_idx as usize, arrays, span)?)
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
        "let_tuple" => {
            // let (a, b, c) = expr
            // else_start/else_count are reused to store names_start/names_count
            // names are ident expr_nodes at [names_start..names_start+names_count)
            let names_start = get_field_i64(&fields, "else_start")?;
            let names_count = get_field_i64(&fields, "else_count")?;
            let mut names = Vec::new();
            if names_count > 0 {
                let s = names_start.max(1) as usize;
                let e = s + names_count as usize;
                for i in s..e {
                    if i == 0 || i > arrays.expr_nodes.len() { continue; }
                    let nval = &arrays.expr_nodes[i - 1];
                    if let Some(nf) = clone_struct_fields_opt(nval) {
                        let nname = get_field_string(&nf, "sval")?;
                        names.push(ast::Ident { name: nname, span: span.clone() });
                    }
                }
            }
            let init = if expr_idx > 0 {
                Some(convert_expr(expr_idx as usize, arrays, span)?)
            } else {
                None
            };
            Ok(Some(ast::Stmt {
                kind: ast::StmtKind::Let {
                    names,
                    type_ann: None,
                    mutable: false,
                    init,
                },
                span: span.clone(),
            }))
        }
        "expr" => {
            if expr_idx > 0 {
                let e = convert_expr(expr_idx as usize, arrays, span)?;
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
                Some(convert_expr(expr_idx as usize, arrays, span)?)
            } else {
                None
            };
            Ok(Some(ast::Stmt {
                kind: ast::StmtKind::Return(val_expr),
                span: span.clone(),
            }))
        }
        "while" => {
            // expr_idx = cond, body_start/body_count in loop_idxs
            let body_start = get_field_i64(&fields, "body_start")?;
            let body_count = get_field_i64(&fields, "body_count")?;
            let cond = if expr_idx > 0 {
                convert_expr(expr_idx as usize, arrays, span)?
            } else {
                ast::Expr {
                    kind: ast::ExprKind::Literal(ast::Literal::Bool(true)),
                    span: span.clone(),
                }
            };
            let body_stmts = if body_count > 0 {
                let s = body_start.max(1) as usize;
                let e = s + body_count as usize - 1;
                convert_loop_body_range(arrays, span, s, e)?
            } else {
                Vec::new()
            };
            Ok(Some(ast::Stmt {
                kind: ast::StmtKind::While {
                    cond,
                    body: Box::new(ast::Stmt {
                        kind: ast::StmtKind::Expr(ast::Expr {
                            kind: ast::ExprKind::Block(body_stmts),
                            span: span.clone(),
                        }),
                        span: span.clone(),
                    }),
                },
                span: span.clone(),
            }))
        }
        "for" => {
            // name = loop var, expr_idx = iterable, body_start/body_count in loop_idxs
            let var_name = get_field_string(&fields, "name")?;
            let body_start = get_field_i64(&fields, "body_start")?;
            let body_count = get_field_i64(&fields, "body_count")?;
            let iter = if expr_idx > 0 {
                convert_expr(expr_idx as usize, arrays, span)?
            } else {
                ast::Expr {
                    kind: ast::ExprKind::Literal(ast::Literal::Int(0, BaseType::I32)),
                    span: span.clone(),
                }
            };
            let body_stmts = if body_count > 0 {
                let s = body_start.max(1) as usize;
                let e = s + body_count as usize - 1;
                convert_loop_body_range(arrays, span, s, e)?
            } else {
                Vec::new()
            };
            Ok(Some(ast::Stmt {
                kind: ast::StmtKind::For {
                    var: ast::Ident { name: var_name, span: span.clone() },
                    iter,
                    body: Box::new(ast::Stmt {
                        kind: ast::StmtKind::Expr(ast::Expr {
                            kind: ast::ExprKind::Block(body_stmts),
                            span: span.clone(),
                        }),
                        span: span.clone(),
                    }),
                },
                span: span.clone(),
            }))
        }
        "loop" => {
            // body_start/body_count in loop_idxs
            let body_start = get_field_i64(&fields, "body_start")?;
            let body_count = get_field_i64(&fields, "body_count")?;
            let body_stmts = if body_count > 0 {
                let s = body_start.max(1) as usize;
                let e = s + body_count as usize - 1;
                convert_loop_body_range(arrays, span, s, e)?
            } else {
                Vec::new()
            };
            Ok(Some(ast::Stmt {
                kind: ast::StmtKind::Loop { body: body_stmts },
                span: span.clone(),
            }))
        }
        "break" => {
            Ok(Some(ast::Stmt {
                kind: ast::StmtKind::Break(None),
                span: span.clone(),
            }))
        }
        "continue" => {
            Ok(Some(ast::Stmt {
                kind: ast::StmtKind::Continue,
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
    idx: usize,
    arrays: &ProgArrays,
    span: &Span,
) -> TenthResult<ast::Expr> {
    convert_expr_depth(idx, arrays, span, 0)
}

fn convert_expr_depth(
    idx: usize,
    arrays: &ProgArrays,
    span: &Span,
    depth: usize,
) -> TenthResult<ast::Expr> {
    if depth > 50 {
        return Err(TenthError::RuntimeError { line: None, col: None,
            message: format!("表达式转换递归过深，索引 {}（深度={}）", idx, depth),
        });
    }
    if idx == 0 || idx > arrays.expr_nodes.len() {
        return Err(TenthError::RuntimeError { line: None, col: None,
            message: format!("表达式索引 {} 越界（长度={}）", idx, arrays.expr_nodes.len()),
        });
    }

    let val = &arrays.expr_nodes[idx - 1];
    let fields = match clone_struct_fields_opt(val) {
        Some(f) => f,
        None => return Err(TenthError::RuntimeError { line: None, col: None,
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
    let name = get_field_string(&fields, "name")?;
    let variant = get_field_string(&fields, "variant")?;

    match kind.as_str() {
        "int" => Ok(ast::Expr {
            kind: ast::ExprKind::Literal(ast::Literal::Int(ival, BaseType::I32)),
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
        "interp" => {
            // Interpolated string: extra_start..extra_start+extra_count are
            // alternating "str" (literal) and "ident" (expression name) nodes.
            let mut parts = Vec::new();
            if extra_count > 0 {
                let s = extra_start.max(1) as usize;
                let e = s + extra_count as usize;
                for i in s..e {
                    if i == 0 || i > arrays.expr_nodes.len() { continue; }
                    let pval = &arrays.expr_nodes[i - 1];
                    if let Some(pf) = clone_struct_fields_opt(pval) {
                        let pkind = get_field_string(&pf, "kind")?;
                        let psval = get_field_string(&pf, "sval")?;
                        match pkind.as_str() {
                            "str" => parts.push(ast::InterpPart::Literal(psval)),
                            "ident" => parts.push(ast::InterpPart::Expr(psval)),
                            _ => {}
                        }
                    }
                }
            }
            Ok(ast::Expr {
                kind: ast::ExprKind::InterpolatedString(parts),
                span: span.clone(),
            })
        }
        "fstring" => {
            // f"..." 模板字符串：与 interp 结构相同，但产物为 FString
            let mut parts = Vec::new();
            if extra_count > 0 {
                let s = extra_start.max(1) as usize;
                let e = s + extra_count as usize;
                for i in s..e {
                    if i == 0 || i > arrays.expr_nodes.len() { continue; }
                    let pval = &arrays.expr_nodes[i - 1];
                    if let Some(pf) = clone_struct_fields_opt(pval) {
                        let pkind = get_field_string(&pf, "kind")?;
                        let psval = get_field_string(&pf, "sval")?;
                        match pkind.as_str() {
                            "str" => parts.push(ast::InterpPart::Literal(psval)),
                            "ident" => parts.push(ast::InterpPart::Expr(psval)),
                            _ => {}
                        }
                    }
                }
            }
            Ok(ast::Expr {
                kind: ast::ExprKind::FString(parts),
                span: span.clone(),
            })
        }
        "binary" => {
            let left_expr = convert_expr_depth(left as usize, arrays, span, depth + 1)?;
            let right_expr = convert_expr_depth(right as usize, arrays, span, depth + 1)?;
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
            let inner = convert_expr_depth(left as usize, arrays, span, depth + 1)?;
            let op = match sval.as_str() {
                "-" => ast::UnaryOp::Neg,
                "!" => ast::UnaryOp::Not,
                "?" => ast::UnaryOp::Try,
                _ => return Err(TenthError::RuntimeError { line: None, col: None,
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
            let args = convert_arg_range(arg_start, arg_count, arrays, span, depth)?;
            Ok(ast::Expr {
                kind: ast::ExprKind::Call { func: Box::new(func_expr), args },
                span: span.clone(),
            })
        }
        "generic_call" => {
            // sval = function name, arg_start/arg_count = call args,
            // extra_start/extra_count = type arg ident nodes
            let func_expr = ast::Expr {
                kind: ast::ExprKind::Ident(ast::Ident { name: sval, span: span.clone() }),
                span: span.clone(),
            };
            let args = convert_arg_range(arg_start, arg_count, arrays, span, depth)?;
            let mut generics = Vec::new();
            if extra_count > 0 {
                let s = extra_start.max(1) as usize;
                let e = s + extra_count as usize;
                for i in s..e {
                    if i == 0 || i > arrays.expr_nodes.len() { continue; }
                    let tval = &arrays.expr_nodes[i - 1];
                    if let Some(tf) = clone_struct_fields_opt(tval) {
                        let tname = get_field_string(&tf, "sval")?;
                        generics.push(parse_type_annotation(&tname, span));
                    }
                }
            }
            Ok(ast::Expr {
                kind: ast::ExprKind::GenericCall { func: Box::new(func_expr), generics, args },
                span: span.clone(),
            })
        }
        "if" => {
            let cond = convert_expr_depth(left as usize, arrays, span, depth + 1)?;
            let then_branch = convert_expr_depth(right as usize, arrays, span, depth + 1)?;
            let else_branch = if extra_start > 0 {
                Some(convert_expr_depth(extra_start as usize, arrays, span, depth + 1)?)
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
            let target = convert_expr_depth(left as usize, arrays, span, depth + 1)?;
            let value = convert_expr_depth(right as usize, arrays, span, depth + 1)?;
            Ok(ast::Expr {
                kind: ast::ExprKind::Assign { target: Box::new(target), value: Box::new(value) },
                span: span.clone(),
            })
        }
        "assign_op" => {
            // sval is like "+=", "-=", "*=", "/="
            let target = convert_expr_depth(left as usize, arrays, span, depth + 1)?;
            let value = convert_expr_depth(right as usize, arrays, span, depth + 1)?;
            // Strip trailing "=" to get the binary op
            let op_str = if sval.ends_with('=') { &sval[..sval.len() - 1] } else { &sval };
            let op = parse_binop(op_str)?;
            Ok(ast::Expr {
                kind: ast::ExprKind::AssignOp { target: Box::new(target), op, value: Box::new(value) },
                span: span.clone(),
            })
        }
        "method_call" => {
            let receiver = convert_expr_depth(left as usize, arrays, span, depth + 1)?;
            let method = ast::Ident { name: sval, span: span.clone() };
            let args = convert_arg_range(arg_start, arg_count, arrays, span, depth)?;
            Ok(ast::Expr {
                kind: ast::ExprKind::MethodCall { receiver: Box::new(receiver), method, args },
                span: span.clone(),
            })
        }
        "field" => {
            let target = convert_expr_depth(left as usize, arrays, span, depth + 1)?;
            Ok(ast::Expr {
                kind: ast::ExprKind::Field {
                    target: Box::new(target),
                    field: ast::Ident { name: sval, span: span.clone() },
                },
                span: span.clone(),
            })
        }
        "index" => {
            let target = convert_expr_depth(left as usize, arrays, span, depth + 1)?;
            let index = convert_expr_depth(right as usize, arrays, span, depth + 1)?;
            Ok(ast::Expr {
                kind: ast::ExprKind::Index {
                    target: Box::new(target),
                    indices: vec![ast::IndexExpr::Single(index)],
                },
                span: span.clone(),
            })
        }
        "slice" => {
            // left = target, right = start, extra_start = end, ival = inclusive
            let target = convert_expr_depth(left as usize, arrays, span, depth + 1)?;
            let start = if right > 0 {
                Some(Box::new(convert_expr_depth(right as usize, arrays, span, depth + 1)?))
            } else {
                None
            };
            let end = if extra_start > 0 {
                Some(Box::new(convert_expr_depth(extra_start as usize, arrays, span, depth + 1)?))
            } else {
                None
            };
            Ok(ast::Expr {
                kind: ast::ExprKind::Index {
                    target: Box::new(target),
                    indices: vec![ast::IndexExpr::Range { start, end }],
                },
                span: span.clone(),
            })
        }
        "ref" => {
            let inner = convert_expr_depth(left as usize, arrays, span, depth + 1)?;
            Ok(ast::Expr {
                kind: ast::ExprKind::Ref(Box::new(inner)),
                span: span.clone(),
            })
        }
        "mut_ref" => {
            let inner = convert_expr_depth(left as usize, arrays, span, depth + 1)?;
            Ok(ast::Expr {
                kind: ast::ExprKind::MutRef(Box::new(inner)),
                span: span.clone(),
            })
        }
        "deref" => {
            let inner = convert_expr_depth(left as usize, arrays, span, depth + 1)?;
            Ok(ast::Expr {
                kind: ast::ExprKind::Deref(Box::new(inner)),
                span: span.clone(),
            })
        }
        "move" => {
            let inner = convert_expr_depth(left as usize, arrays, span, depth + 1)?;
            Ok(ast::Expr {
                kind: ast::ExprKind::Move(Box::new(inner)),
                span: span.clone(),
            })
        }
        "try_block" => {
            let inner = convert_expr_depth(left as usize, arrays, span, depth + 1)?;
            Ok(ast::Expr {
                kind: ast::ExprKind::TryBlock(Box::new(inner)),
                span: span.clone(),
            })
        }
        "await" => {
            let inner = convert_expr_depth(left as usize, arrays, span, depth + 1)?;
            Ok(ast::Expr {
                kind: ast::ExprKind::Await(Box::new(inner)),
                span: span.clone(),
            })
        }
        "spawn" => {
            let inner = convert_expr_depth(left as usize, arrays, span, depth + 1)?;
            Ok(ast::Expr {
                kind: ast::ExprKind::Spawn(Box::new(inner)),
                span: span.clone(),
            })
        }
        "range" => {
            // left = start (0 if open-start), right = end (0 if open-end),
            // ival = 1 if inclusive
            let start = if left > 0 {
                Some(Box::new(convert_expr_depth(left as usize, arrays, span, depth + 1)?))
            } else {
                None
            };
            let end = if right > 0 {
                Some(Box::new(convert_expr_depth(right as usize, arrays, span, depth + 1)?))
            } else {
                None
            };
            Ok(ast::Expr {
                kind: ast::ExprKind::Range {
                    start,
                    end,
                    inclusive: ival != 0,
                },
                span: span.clone(),
            })
        }
        "tuple" => {
            let elems = convert_arg_range(arg_start, arg_count, arrays, span, depth)?;
            Ok(ast::Expr {
                kind: ast::ExprKind::Tuple(elems),
                span: span.clone(),
            })
        }
        "array" => {
            let elems = convert_arg_range(arg_start, arg_count, arrays, span, depth)?;
            Ok(ast::Expr {
                kind: ast::ExprKind::ArrayLiteral(elems),
                span: span.clone(),
            })
        }
        "tensor" => {
            // ival = rows, arg_count = total element count (flattened)
            // Split elements into rows. Each row has total_count / rows elements.
            let rows = ival.max(1) as usize;
            let total = arg_count as usize;
            let row_len = if rows > 0 { total / rows } else { total };
            let mut all_elems = Vec::new();
            // Collect all element expressions first
            let mut flat: Vec<ast::Expr> = Vec::new();
            if arg_count > 0 {
                let s = arg_start.max(1) as usize;
                let e = s + arg_count as usize;
                for i in s..e {
                    if i == 0 || i > arrays.expr_nodes.len() { continue; }
                    // arg nodes: kind="arg", left=actual expr index
                    let aval = &arrays.expr_nodes[i - 1];
                    if let Some(af) = clone_struct_fields_opt(aval) {
                        let aleft = get_field_i64(&af, "left")?;
                        if aleft > 0 {
                            flat.push(convert_expr_depth(aleft as usize, arrays, span, depth + 1)?);
                        }
                    }
                }
            }
            // Split into rows
            for r in 0..rows {
                let start = r * row_len;
                let end = (start + row_len).min(flat.len());
                if start < end {
                    all_elems.push(flat[start..end].to_vec());
                }
            }
            Ok(ast::Expr {
                kind: ast::ExprKind::TensorLiteral(all_elems),
                span: span.clone(),
            })
        }
        "closure" => {
            // left = body, extra_start/extra_count = param ident nodes
            let body = convert_expr_depth(left as usize, arrays, span, depth + 1)?;
            let mut params = Vec::new();
            if extra_count > 0 {
                let s = extra_start.max(1) as usize;
                let e = s + extra_count as usize;
                for i in s..e {
                    if i == 0 || i > arrays.expr_nodes.len() { continue; }
                    let pval = &arrays.expr_nodes[i - 1];
                    if let Some(pf) = clone_struct_fields_opt(pval) {
                        let pname = get_field_string(&pf, "sval")?;
                        params.push((
                            ast::Ident { name: pname, span: span.clone() },
                            None,
                        ));
                    }
                }
            }
            Ok(ast::Expr {
                kind: ast::ExprKind::Closure {
                    params,
                    body: Box::new(body),
                },
                span: span.clone(),
            })
        }
        "struct_literal" => {
            // name = struct name, arg_start/arg_count = field_init nodes
            // Each field_init node: sval = field name, left = value expr index
            let mut fields = Vec::new();
            if arg_count > 0 {
                let s = arg_start.max(1) as usize;
                let e = s + arg_count as usize;
                for i in s..e {
                    if i == 0 || i > arrays.expr_nodes.len() { continue; }
                    let fval = &arrays.expr_nodes[i - 1];
                    if let Some(ff) = clone_struct_fields_opt(fval) {
                        let fname = get_field_string(&ff, "sval")?;
                        let fleft = get_field_i64(&ff, "left")?;
                        let fexpr = if fleft > 0 {
                            convert_expr_depth(fleft as usize, arrays, span, depth + 1)?
                        } else {
                            ast::Expr {
                                kind: ast::ExprKind::Literal(ast::Literal::Int(0, BaseType::I32)),
                                span: span.clone(),
                            }
                        };
                        fields.push((
                            ast::Ident { name: fname, span: span.clone() },
                            fexpr,
                        ));
                    }
                }
            }
            Ok(ast::Expr {
                kind: ast::ExprKind::StructLiteral {
                    name: ast::Ident { name, span: span.clone() },
                    generics: Vec::new(),
                    fields,
                    use_defaults: false,
                },
                span: span.clone(),
            })
        }
        "enum_literal" => {
            // name = enum name, variant = variant name
            // arg_start/arg_count = tuple-variant positional args
            let mut fields = Vec::new();
            if arg_count > 0 {
                let s = arg_start.max(1) as usize;
                let e = s + arg_count as usize;
                for i in s..e {
                    if i == 0 || i > arrays.expr_nodes.len() { continue; }
                    let aval = &arrays.expr_nodes[i - 1];
                    if let Some(af) = clone_struct_fields_opt(aval) {
                        let aleft = get_field_i64(&af, "left")?;
                        if aleft > 0 {
                            let fexpr = convert_expr_depth(aleft as usize, arrays, span, depth + 1)?;
                            // Use positional index as field name (will be lowered as tuple variant)
                            fields.push((
                                ast::Ident { name: format!("_{}", i - s), span: span.clone() },
                                fexpr,
                            ));
                        }
                    }
                }
            }
            Ok(ast::Expr {
                kind: ast::ExprKind::EnumLiteral {
                    enum_name: ast::Ident { name, span: span.clone() },
                    variant: ast::Ident { name: variant, span: span.clone() },
                    fields,
                },
                span: span.clone(),
            })
        }
        "match" => {
            // left = scrutinee, extra_start/extra_count = arms in match_arms
            let scrutinee = convert_expr_depth(left as usize, arrays, span, depth + 1)?;
            let mut arms = Vec::new();
            if extra_count > 0 {
                let s = extra_start.max(1) as usize;
                let e = s + extra_count as usize;
                for i in s..e {
                    if i == 0 || i > arrays.match_arms.len() { continue; }
                    let amval = &arrays.match_arms[i - 1];
                    if let Some(af) = clone_struct_fields_opt(amval) {
                        let pat_kind = get_field_string(&af, "pat_kind")?;
                        let pat_name = get_field_string(&af, "pat_name")?;
                        let pat_bind = get_field_string(&af, "pat_bind")?;
                        let body_expr = get_field_i64(&af, "body_expr")?;
                        let body = if body_expr > 0 {
                            convert_expr_depth(body_expr as usize, arrays, span, depth + 1)?
                        } else {
                            ast::Expr {
                                kind: ast::ExprKind::Literal(ast::Literal::Int(0, BaseType::I32)),
                                span: span.clone(),
                            }
                        };
                        let pattern = parse_match_pattern(&pat_kind, &pat_name, &pat_bind)?;
                        arms.push(ast::MatchArm {
                            pattern,
                            guard: None,
                            body,
                        });
                    }
                }
            }
            Ok(ast::Expr {
                kind: ast::ExprKind::Match {
                    scrutinee: Box::new(scrutinee),
                    arms,
                },
                span: span.clone(),
            })
        }
        "block" => {
            // extra_start/extra_count point into block_idxs
            let mut stmts = Vec::new();
            if extra_count > 0 {
                let s = extra_start.max(1) as usize;
                let e = s + extra_count as usize;
                for i in s..e {
                    if i == 0 || i > arrays.block_idxs.len() { continue; }
                    let stmt_idx = match &arrays.block_idxs[i - 1] {
                        Value::Int(n, _) => *n as usize,
                        Value::Shared(rc) => {
                            let inner = rc.borrow();
                            match &*inner {
                                Value::Int(n, _) => *n as usize,
                                _ => continue,
                            }
                        }
                        _ => continue,
                    };
                    if stmt_idx == 0 || stmt_idx > arrays.stmt_nodes.len() { continue; }
                    let sval = &arrays.stmt_nodes[stmt_idx - 1];
                    if let Some(stmt) = convert_stmt(sval, arrays, span)? {
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
                Some(convert_expr_depth(left as usize, arrays, span, depth + 1)?)
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
                kind: ast::ExprKind::Literal(ast::Literal::Int(0, BaseType::I32)),
                span: span.clone(),
            })
        }
    }
}

/// Convert a range of "arg" reference nodes into actual expressions.
/// Each arg node has kind="arg" and left=actual expr index.
fn convert_arg_range(
    arg_start: i64,
    arg_count: i64,
    arrays: &ProgArrays,
    span: &Span,
    depth: usize,
) -> TenthResult<Vec<ast::Expr>> {
    let mut args = Vec::new();
    if arg_count > 0 {
        let s = arg_start.max(1) as usize;
        let e = s + arg_count as usize;
        for i in s..e {
            if i == 0 || i > arrays.expr_nodes.len() { continue; }
            let aval = &arrays.expr_nodes[i - 1];
            if let Some(af) = clone_struct_fields_opt(aval) {
                let aleft = get_field_i64(&af, "left")?;
                if aleft > 0 {
                    args.push(convert_expr_depth(aleft as usize, arrays, span, depth + 1)?);
                }
            }
        }
    }
    Ok(args)
}

/// Parse a match pattern from tenthc's pat_kind/pat_name/pat_bind fields.
fn parse_match_pattern(pat_kind: &str, pat_name: &str, pat_bind: &str) -> TenthResult<ast::Pattern> {
    match pat_kind {
        "wildcard" => Ok(ast::Pattern::Wildcard),
        "binding" => Ok(ast::Pattern::Binding(pat_name.to_string())),
        "literal" => {
            // pat_name is the literal value as string
            if pat_name == "true" {
                Ok(ast::Pattern::Literal(ast::Literal::Bool(true)))
            } else if pat_name == "false" {
                Ok(ast::Pattern::Literal(ast::Literal::Bool(false)))
            } else if let Ok(n) = pat_name.parse::<i64>() {
                Ok(ast::Pattern::Literal(ast::Literal::Int(n, BaseType::I32)))
            } else {
                // Fallback: treat as binding
                Ok(ast::Pattern::Binding(pat_name.to_string()))
            }
        }
        "range" => {
            // pat_bind is "start..end" or "start..=end"
            let (start, end, inclusive) = if pat_bind.contains("..=") {
                let parts: Vec<&str> = pat_bind.splitn(2, "..=").collect();
                if parts.len() == 2 {
                    (parts[0].parse::<i64>().unwrap_or(0), parts[1].parse::<i64>().unwrap_or(0), true)
                } else {
                    (0, 0, true)
                }
            } else if pat_bind.contains("..") {
                let parts: Vec<&str> = pat_bind.splitn(2, "..").collect();
                if parts.len() == 2 {
                    (parts[0].parse::<i64>().unwrap_or(0), parts[1].parse::<i64>().unwrap_or(0), false)
                } else {
                    (0, 0, false)
                }
            } else {
                (0, 0, false)
            };
            Ok(ast::Pattern::Range { start, end, inclusive })
        }
        "enum_variant" => {
            // pat_bind stores "EnumName:field1" or "EnumName" or ":field1"
            let (enum_name, tuple_fields) = if pat_bind.is_empty() {
                (String::new(), Vec::new())
            } else if let Some(colon_pos) = pat_bind.find(':') {
                let en = pat_bind[..colon_pos].to_string();
                let rest = &pat_bind[colon_pos + 1..];
                let tf: Vec<String> = if rest.is_empty() {
                    Vec::new()
                } else {
                    rest.split(',').map(|s| s.trim().to_string()).collect()
                };
                (en, tf)
            } else {
                (pat_bind.to_string(), Vec::new())
            };
            Ok(ast::Pattern::EnumVariant {
                enum_name,
                variant: pat_name.to_string(),
                field_bind: None,
                tuple_fields,
            })
        }
        "tuple" => {
            // Problem 21 降级项 5 修复：parser.th:1148-1194 已解析 tuple pattern
            // 的子模式名，存入 pat_bind（逗号分隔，如 "a,b,c"）。下划线 "_" 表示通配。
            // 拆分 pat_bind 构造 Pattern::Tuple，每个子模式为 Binding 或 Wildcard。
            // 注：嵌套 tuple pattern 在 parser.th 中以 "_" 占位（保守）。
            if pat_bind.is_empty() {
                Ok(ast::Pattern::Wildcard)
            } else {
                let mut sub_patterns: Vec<ast::Pattern> = Vec::new();
                for s in pat_bind.split(',') {
                    let trimmed = s.trim();
                    if trimmed.is_empty() || trimmed == "_" {
                        sub_patterns.push(ast::Pattern::Wildcard);
                    } else {
                        sub_patterns.push(ast::Pattern::Binding(trimmed.to_string()));
                    }
                }
                Ok(ast::Pattern::Tuple(sub_patterns))
            }
        }
        "struct" => {
            // We don't have detailed struct pattern info from tenthc parser
            // (it skips the contents). Return a wildcard as fallback.
            Ok(ast::Pattern::Wildcard)
        }
        _ => Ok(ast::Pattern::Wildcard),
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
        _ => Err(TenthError::RuntimeError { line: None, col: None,
            message: format!("未知的二元运算符：{}", s),
        }),
    }
}
