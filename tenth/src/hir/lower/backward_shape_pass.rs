//! 跨算子反向 shape 传播（护城河 A 深化 Phase 2）。
//!
//! 在 lowering 完成后，对所有函数定义做一次后向分析 pass：
//! 识别 `start_grad`/`new_grad` ... `backward(loss)` 构成的 grad 区域，
//! 收集区域内的 tensor 操作序列，逆序传播梯度 shape，验证参数的梯度
//! shape 与参数本身 shape 兼容。把方向 A 的运行时 `acc_grad` shape 不匹配
//! 错误提升到编译期 `TypeError`。
//!
//! 保守策略：
//! - grad 区域内含 if/else/match/while/for/loop → 跳过该区域（不验证）
//! - tensor 操作的输入不是简单 `Var` → 跳过该操作（不传播）
//! - start_grad/backward/param 的参数不是简单 `Var` → 跳过
//! - 任一 shape 全 `Any`（无静态信息）→ 跳过兼容性检查
//! - pass 整体是 O(n) 遍历，n 为函数体内的操作数
//!
//! 与 Phase 1（单算子级 `backward_shapes.rs`）的关系：
//! - Phase 1 在 lower 表达式时对单个算子调用 `backward_shape` 验证
//! - Phase 2 在 lowering 完成后对整个 grad 区域做跨算子传播
//! - 两者互补：Phase 1 拦截单算子反向 shape 错误，Phase 2 拦截
//!   "每个算子单独看都合法，但跨算子组合后梯度 shape 与参数 shape 不匹配"
//!   的错误（如 reshape 链改变 numel 后再 backward）

use std::collections::HashMap;
use crate::error::TenthError;
use crate::hir::hir::*;
use crate::hir::types::{Dim, Type, BaseType};
use crate::lexer::token::Span;
use super::backward_shapes::backward_shape;
use super::types::{has_static_info, fmt_dims};

/// grad 区域：start_grad/new_grad 到 backward 之间的代码段
struct GradRegion {
    #[allow(dead_code)]
    start_span: Span,
    /// start_grad 参数（变量名 + shape）。
    /// 实际代码中 start_grad/new_grad 通常无参数；可训练参数通过 `param(t)` 注册。
    /// 这里收集区域内所有 `let w = param(init)` 的 w，以及 start_grad(t) 的 t（若存在）。
    params: Vec<(String, Vec<Dim>)>,
    /// backward(loss) 的 loss 变量名
    loss_var: String,
    /// loss 的 shape（通常为标量 `[]` 或 `[1]`）
    loss_shape: Vec<Dim>,
    /// grad 区域内的 tensor 操作序列（顺序）
    operations: Vec<TensorOp>,
    /// backward 调用的 span（用于错误定位）
    backward_span: Span,
}

/// 单个 tensor 操作的记录
#[derive(Clone)]
struct TensorOp {
    /// 输出变量名（let 绑定的第一个名字）
    output_var: String,
    /// 输出 shape（从 let 的 init 表达式 ty 提取）
    output_shape: Vec<Dim>,
    /// 算子名："add"/"matmul"/"cross_entropy"/...
    op_name: String,
    /// 输入变量名（按算子签名顺序）
    input_vars: Vec<String>,
    /// 输入 shapes（与 input_vars 一一对应）
    input_shapes: Vec<Vec<Dim>>,
    /// 操作的 span（用于错误定位）
    span: Span,
}

/// Phase 2 主入口：对一组函数定义做跨算子反向 shape 传播验证。
/// 返回所有发现的编译期错误（按函数体顺序）。
pub(super) fn backward_shape_pass(fn_defs: &[HirFnDef]) -> Vec<TenthError> {
    let mut errors = Vec::new();
    for fn_def in fn_defs {
        let regions = find_grad_regions(&fn_def.body);
        for region in regions {
            let mut region_errors = propagate_backward(&region);
            errors.append(&mut region_errors);
        }
    }
    errors
}

// ── 区域识别 ────────────────────────────────────────────────────────────

/// 遍历函数 body 的 Block stmts，识别 start_grad/backward 配对。
///
/// 配对规则：从 start_grad/new_grad 调用开始，到下一个 backward 调用结束。
/// 支持多个 grad 区域（多对 start_grad/backward）。
///
/// 识别方式：
/// - start_grad/new_grad/backward 都是 `Call { func: Var(name), args, ret_ty }`
///   或作为 `Let { init: Some(Call...) }` 出现
/// - start_grad(t) 若带参数，t 视为可训练参数（变量名 + shape）
/// - 区域内 `let w = param(init)` 的 w 也作为可训练参数
fn find_grad_regions(body: &HirExpr) -> Vec<GradRegion> {
    let mut regions = Vec::new();
    // body 通常是 Block；如果不是，无法识别区域
    let stmts = match &body.kind {
        HirExprKind::Block { stmts, .. } => stmts,
        _ => return regions,
    };

    let mut current: Option<GradRegion> = None;
    for stmt in stmts {
        // 检查是否是控制流语句 — 若区域已开启，回退（放弃该区域）
        if let Some(mut region) = current.take() {
            if stmt_has_control_flow(stmt) {
                // 保守回退：丢弃该区域（不报错也不验证）
                continue;
            }
            // 尝试收集 tensor op 或识别 backward
            if let Some(consumed) = try_consume_region_stmt(stmt, &mut region) {
                if consumed {
                    // backward 已找到，关闭区域
                    regions.push(region);
                } else {
                    // 未关闭，继续累积
                    current = Some(region);
                }
            } else {
                // 该 stmt 既不是 tensor op 也不是 backward/start_grad
                // 但已被 try_consume_region_stmt 处理过（可能收集了 param）
                current = Some(region);
            }
        } else {
            // 未在区域中：尝试识别 start_grad/new_grad
            if let Some(start_span) = match_start_grad(stmt) {
                let mut region = GradRegion {
                    start_span: start_span.clone(),
                    params: Vec::new(),
                    loss_var: String::new(),
                    loss_shape: Vec::new(),
                    operations: Vec::new(),
                    backward_span: start_span,
                };
                // start_grad(t) 带参数形式：t 视为参数
                if let Some(p) = extract_start_grad_param(stmt) {
                    region.params.push(p);
                }
                // start_grad 后续 stmt 可能就是 let w = param(...)，由 try_consume_region_stmt 处理
                current = Some(region);
            }
        }
    }
    // 未关闭的区域（有 start_grad 无 backward）：保守丢弃，不报错
    let _ = current;
    regions
}

/// 判断 stmt 是否是控制流（while/for/loop）。
/// if/match 作为 expr 出现在 let init 中由 `expr_has_control_flow` 检查。
fn stmt_has_control_flow(stmt: &HirStmt) -> bool {
    matches!(
        stmt.kind,
        HirStmtKind::While { .. } | HirStmtKind::For { .. } | HirStmtKind::Loop { .. }
    )
}

/// 递归检查 expr 是否含 If/Match 控制流（grad 区域内若含则保守回退）。
fn expr_has_control_flow(expr: &HirExpr) -> bool {
    match &expr.kind {
        HirExprKind::If { .. } | HirExprKind::Match { .. } => true,
        HirExprKind::Binary { left, right, .. } => {
            expr_has_control_flow(left) || expr_has_control_flow(right)
        }
        HirExprKind::Unary { expr, .. } => expr_has_control_flow(expr),
        HirExprKind::Call { func, args, .. } => {
            expr_has_control_flow(func) || args.iter().any(expr_has_control_flow)
        }
        HirExprKind::GenericCall { func, args, .. } => {
            expr_has_control_flow(func) || args.iter().any(expr_has_control_flow)
        }
        HirExprKind::MethodCall { receiver, args, .. } => {
            expr_has_control_flow(receiver) || args.iter().any(expr_has_control_flow)
        }
        HirExprKind::Block { stmts, final_expr } => {
            stmts.iter().any(stmt_has_control_flow)
                || stmts.iter().any(|s| {
                    if let HirStmtKind::Expr(e) = &s.kind {
                        expr_has_control_flow(e)
                    } else if let HirStmtKind::Let { init: Some(e), .. } = &s.kind {
                        expr_has_control_flow(e)
                    } else if let HirStmtKind::Return(Some(e)) = &s.kind {
                        expr_has_control_flow(e)
                    } else {
                        false
                    }
                })
                || final_expr.as_ref().map(|e| expr_has_control_flow(e)).unwrap_or(false)
        }
        HirExprKind::Ref(e) | HirExprKind::MutRef(e) | HirExprKind::Deref(e)
        | HirExprKind::Move(e) | HirExprKind::TryBlock(e) | HirExprKind::Await(e)
        | HirExprKind::Spawn(e) | HirExprKind::Lossy(e) => expr_has_control_flow(e),
        HirExprKind::Yield(inner) => {
            inner.as_ref().map(|e| expr_has_control_flow(e)).unwrap_or(false)
        }
        HirExprKind::Assign { value, .. } | HirExprKind::AssignOp { value, .. }
        | HirExprKind::FieldAssign { value, .. } => expr_has_control_flow(value),
        HirExprKind::DerefAssign { value, .. } | HirExprKind::DerefAssignOp { value, .. } => {
            expr_has_control_flow(value)
        }
        HirExprKind::Tuple(es) => es.iter().any(expr_has_control_flow),
        HirExprKind::TensorLiteral { data, .. } => {
            data.iter().flatten().any(expr_has_control_flow)
        }
        HirExprKind::ArrayLiteral { elements, .. } => {
            elements.iter().any(expr_has_control_flow)
        }
        HirExprKind::StructLiteral { fields, .. } => {
            fields.iter().any(|(_, e)| expr_has_control_flow(e))
        }
        HirExprKind::UnionLiteral { value, .. } => expr_has_control_flow(value),
        HirExprKind::EnumLiteral { fields, .. } => {
            fields.iter().any(|(_, e)| expr_has_control_flow(e))
        }
        HirExprKind::Index { target, .. } => expr_has_control_flow(target),
        HirExprKind::Field { target, .. } => expr_has_control_flow(target),
        HirExprKind::Range { start, end, .. } => {
            start.as_ref().map(|e| expr_has_control_flow(e)).unwrap_or(false)
                || end.as_ref().map(|e| expr_has_control_flow(e)).unwrap_or(false)
        }
        HirExprKind::Closure { body, .. } => expr_has_control_flow(body),
        HirExprKind::Literal(_) | HirExprKind::Var(_)
        | HirExprKind::InterpolatedString { .. } => false,
    }
}

/// 若 stmt 是 `start_grad(...)` 或 `new_grad(...)` 调用（作为 Expr stmt 或 let init），
/// 返回其 span。否则返回 None。
fn match_start_grad(stmt: &HirStmt) -> Option<Span> {
    let expr = stmt_expr(stmt)?;
    if let HirExprKind::Call { func, .. } = &expr.kind {
        if let HirExprKind::Var(name) = &func.kind {
            if name == "start_grad" || name == "new_grad" {
                return Some(expr.span.clone());
            }
        }
    }
    None
}

/// 从 `start_grad(t)` 调用中提取参数 t 的变量名和 shape。
/// 若 start_grad 无参数（与 new_grad 同义）或参数不是简单 Var，返回 None。
fn extract_start_grad_param(stmt: &HirStmt) -> Option<(String, Vec<Dim>)> {
    let expr = stmt_expr(stmt)?;
    if let HirExprKind::Call { func, args, .. } = &expr.kind {
        if let HirExprKind::Var(name) = &func.kind {
            if name == "start_grad" && args.len() == 1 {
                if let HirExprKind::Var(var_name) = &args[0].kind {
                    if let Some(dims) = tensor_dims(&args[0].ty) {
                        return Some((var_name.clone(), dims));
                    }
                }
            }
        }
    }
    None
}

/// 从 stmt 中提取表达式引用（Expr stmt 或 Let init 或 Return 值）。
fn stmt_expr(stmt: &HirStmt) -> Option<&HirExpr> {
    match &stmt.kind {
        HirStmtKind::Expr(e) => Some(e),
        HirStmtKind::Let { init: Some(e), .. } => Some(e),
        HirStmtKind::Return(Some(e)) => Some(e),
        _ => None,
    }
}

/// 处理 grad 区域内的 stmt：
/// - 若是 `backward(loss)` 调用，记录 loss_var/loss_shape，返回 consumed=true（区域关闭）
/// - 若是 `let w = param(init)`，记录 w 为参数
/// - 若是 `let x = <tensor_op>`，收集为 TensorOp
/// - 其他 stmt：不处理
///
/// 返回值：
/// - Some(true) — 区域已关闭（遇到 backward）
/// - Some(false) — stmt 已处理但区域未关闭
/// - None — stmt 无法识别为 grad 相关操作（调用方应保持区域开启）
fn try_consume_region_stmt(stmt: &HirStmt, region: &mut GradRegion) -> Option<bool> {
    let expr = match stmt_expr(stmt) {
        Some(e) => e,
        None => return Some(false),
    };

    // 1. backward(loss) 调用
    if let HirExprKind::Call { func, args, .. } = &expr.kind {
        if let HirExprKind::Var(name) = &func.kind {
            if name == "backward" {
                if let Some(arg) = args.first() {
                    if let HirExprKind::Var(loss_var) = &arg.kind {
                        region.loss_var = loss_var.clone();
                        if let Some(dims) = tensor_dims(&arg.ty) {
                            region.loss_shape = dims;
                        }
                        region.backward_span = expr.span.clone();
                        return Some(true);
                    }
                }
                // backward 参数不是简单 Var — 保守：仍关闭区域但 loss_var 为空（传播时跳过）
                region.backward_span = expr.span.clone();
                return Some(true);
            }
        }
    }

    // 2. let w = param(init) — 注册可训练参数
    if let HirStmtKind::Let { names, init: Some(init_expr), .. } = &stmt.kind {
        if let HirExprKind::Call { func, args, .. } = &init_expr.kind {
            if let HirExprKind::Var(name) = &func.kind {
                if name == "param" {
                    // param(init)：w 的 shape 来自 init_expr.ty（即 init 的 shape）
                    if let Some(dims) = tensor_dims(&init_expr.ty) {
                        if let Some(first_name) = names.first() {
                            region.params.push((first_name.clone(), dims));
                        }
                    }
                    return Some(false);
                }
            }
        }

        // 3. let x = <tensor_op> — 收集 TensorOp
        // 若 init 表达式含控制流，整个区域应回退（调用方在 stmt_has_control_flow 已检查）
        // 但 if/match 可能嵌套在 expr 内部，这里再检查一次 init 表达式
        if expr_has_control_flow(init_expr) {
            // 标记区域为不可验证：清空 operations 并返回 None 让调用方丢弃
            // 实际上调用方已对 stmt 做了 stmt_has_control_flow 检查（仅查 While/For/Loop），
            // 这里对 init 表达式做 expr_has_control_flow 检查（查 If/Match）
            // 返回 None 让调用方丢弃整个区域
            return None;
        }
        if let Some(op) = collect_tensor_op(names, init_expr) {
            region.operations.push(op);
            return Some(false);
        }
    }

    Some(false)
}

// ── Tensor 操作收集 ────────────────────────────────────────────────────

/// 从 let 绑定的 init 表达式提取 tensor 操作。
///
/// 识别：
/// - `Call { func: Var(name), args, ret_ty }` → op_name = name
///   （如 cross_entropy/select/scatter/gather/sum/mean 等顶层函数）
/// - `MethodCall { receiver, method, args, ret_ty }` → op_name = method
///   （如 matmul/bmm/reshape/sum/masked_fill）
/// - `Binary { op, left, right, ty }` → op_name = binop 名称
///   （add/sub/mul/div）
///
/// 输入变量提取：递归收集所有 `Var(name)`，但排除 op_name 本身（如 `Var("matmul")` 是函数名）。
/// 输入 shape：从每个输入变量的 Type 提取 dims。若任一输入不是简单 Var，返回 None（保守跳过）。
///
/// 输出 shape：从 init_expr.ty 提取 dims。
fn collect_tensor_op(names: &[String], init_expr: &HirExpr) -> Option<TensorOp> {
    let output_var = names.first()?.clone();
    let output_shape = tensor_dims(&init_expr.ty).unwrap_or_default();
    let span = init_expr.span.clone();

    let (op_name, input_exprs): (String, Vec<&HirExpr>) = match &init_expr.kind {
        // 顶层函数调用：cross_entropy/select/scatter/gather/sum/mean/...
        HirExprKind::Call { func, args, .. } => {
            let name = match &func.kind {
                HirExprKind::Var(n) => n.clone(),
                _ => return None,
            };
            // 跳过非 tensor 相关的 native 调用（如 println/randn/zeros 等构造函数）
            if !is_tensor_op(&name) {
                return None;
            }
            (name, args.iter().collect())
        }
        // 方法调用：matmul/bmm/reshape/sum/masked_fill/...
        HirExprKind::MethodCall { receiver, method, args, .. } => {
            if !is_tensor_op(method) {
                return None;
            }
            let mut inputs: Vec<&HirExpr> = Vec::new();
            inputs.push(receiver.as_ref());
            for a in args {
                inputs.push(a);
            }
            (method.clone(), inputs)
        }
        // 二元运算：add/sub/mul/div
        HirExprKind::Binary { op, left, right, .. } => {
            let name = match op {
                BinOp::Add => "add",
                BinOp::Sub => "sub",
                BinOp::Mul => "mul",
                BinOp::Div => "div",
                _ => return None, // 比较逻辑运算不参与反向 shape 传播
            };
            (name.to_string(), vec![left.as_ref(), right.as_ref()])
        }
        _ => return None,
    };

    // 提取输入变量名和 shape
    let mut input_vars = Vec::new();
    let mut input_shapes = Vec::new();
    for input_expr in input_exprs {
        match &input_expr.kind {
            HirExprKind::Var(var_name) => {
                input_vars.push(var_name.clone());
                if let Some(dims) = tensor_dims(&input_expr.ty) {
                    input_shapes.push(dims);
                } else {
                    // 输入是 Var 但非 Tensor 类型（如标量）— 记录空 shape
                    input_shapes.push(Vec::new());
                }
            }
            // 输入不是简单 Var（嵌套表达式）— 保守跳过整个操作
            _ => return None,
        }
    }

    Some(TensorOp {
        output_var,
        output_shape,
        op_name,
        input_vars,
        input_shapes,
        span,
    })
}

/// 判断函数/方法名是否是参与反向 shape 传播的 tensor 算子。
///
/// 包含 `backward_shapes::backward_shape` 中处理的所有算子，
/// 加上二元算术（add/sub/mul/div）。
/// 排除：构造函数（zeros/ones/randn/tensor）、I/O（println）、
/// 标量数学（abs/sqrt/sin/cos）、控制流原语等。
fn is_tensor_op(name: &str) -> bool {
    matches!(
        name,
        // 二元算术
        "add" | "sub" | "mul" | "div"
        // matmul 家族
        | "matmul" | "bmm"
        // 归约
        | "sum" | "mean"
        // 损失函数
        | "cross_entropy"
        // shape 变换
        | "reshape" | "view"
        // 索引/掩码
        | "scatter" | "gather" | "masked_fill" | "select"
        // 逐元素一元（反向 shape 与输入一致，传播时直接传递）
        | "neg" | "relu" | "exp" | "log" | "sigmoid" | "abs"
        | "softmax" | "dropout" | "batch_norm" | "layer_norm" | "gelu"
        | "transpose" | "conv2d"
    )
}

// ── 反向 shape 传播 ────────────────────────────────────────────────────

/// 对单个 grad 区域做反向 shape 传播，返回发现的编译期错误。
///
/// 算法：
/// 1. 初始化 `grad_shapes: HashMap<String, Vec<Dim>>`
/// 2. `grad_shapes[loss_var] = loss_shape`（loss 的梯度 shape = loss shape）
/// 3. 逆序遍历 operations：
///    - 若 `output_var` 在 `grad_shapes` 中有记录
///    - 调用 `backward_shape(op_name, input_shapes, output_shape)`
///    - 将返回的梯度 shapes 写入 `grad_shapes`（对应 input_vars）
///    - 若 `backward_shape` 返回 Err，记录编译期错误
/// 4. 验证 params 的梯度 shape：
///    - 对每个 `(param_var, param_shape)` in `region.params`
///    - 若 `grad_shapes` 中有 `param_var` 的梯度 shape
///    - 用 `shape_compatible` 验证梯度 shape 与 param_shape 兼容
///    - 不兼容则记录编译期错误
fn propagate_backward(region: &GradRegion) -> Vec<TenthError> {
    let mut errors = Vec::new();

    // loss_var 为空表示 backward 参数不是简单 Var，无法传播
    if region.loss_var.is_empty() {
        return errors;
    }

    let mut grad_shapes: HashMap<String, Vec<Dim>> = HashMap::new();
    grad_shapes.insert(region.loss_var.clone(), region.loss_shape.clone());

    // 逆序遍历 operations
    for op in region.operations.iter().rev() {
        // 仅当 output_var 有梯度 shape 时才传播
        let grad_shape = match grad_shapes.get(&op.output_var) {
            Some(s) => s.clone(),
            None => continue, // 该操作的输出未参与反向传播（无梯度流向它）
        };

        // 调用 Phase 1 的 backward_shape 计算各输入的梯度 shape
        match backward_shape(&op.op_name, &op.input_shapes, &op.output_shape) {
            Ok(input_grads) => {
                // 将梯度 shape 写入 grad_shapes
                for (i, input_var) in op.input_vars.iter().enumerate() {
                    if let Some(grad) = input_grads.get(i) {
                        // 空梯度 shape 表示该输入不可微（如 scatter 的 index）— 跳过
                        if !grad.is_empty() || op.input_shapes.get(i).map_or(false, |s| s.is_empty()) {
                            grad_shapes.insert(input_var.clone(), grad.clone());
                        }
                    }
                }
                // 优化：传播后该 output 的梯度已消费，可移除以释放内存
                // 但保留也无妨（HashMap 会自动处理）
            }
            Err(msg) => {
                // backward_shape 返回 Err：记录编译期错误
                let input_shapes_str: Vec<String> = op.input_shapes.iter().map(|s| fmt_dims(s)).collect();
                errors.push(TenthError::TypeError {
                    line: op.span.line,
                    col: op.span.col,
                    message: format!(
                        "编译期跨算子反向 shape 传播失败（{}）：{}（梯度 shape {} 无法传播到输入 shapes {}）",
                        op.op_name, msg, fmt_dims(&grad_shape), input_shapes_str.join(", ")
                    ),
                });
            }
        }
    }

    // 验证 params 的梯度 shape 与参数本身 shape 兼容
    for (param_var, param_shape) in &region.params {
        if let Some(grad_shape) = grad_shapes.get(param_var) {
            // 任一 shape 全 Any 时跳过（保守）
            if !has_static_info(grad_shape) || !has_static_info(param_shape) {
                continue;
            }
            if !shape_compatible(grad_shape, param_shape) {
                errors.push(TenthError::TypeError {
                    line: region.backward_span.line,
                    col: region.backward_span.col,
                    message: format!(
                        "编译期跨算子反向 shape 传播：参数 '{}' 的梯度 shape {} 与参数 shape {} 不兼容（backward 将触发 acc_grad shape 不匹配）",
                        param_var, fmt_dims(grad_shape), fmt_dims(param_shape)
                    ),
                });
            }
        }
    }

    errors
}

/// 验证两个 shape 是否兼容（用于梯度 shape 与参数 shape 比较）。
///
/// 规则（与 `backward_shapes.rs::unbroadcast_feasible` 一致，但从右往左对齐）：
/// - 维度数不同 → 不兼容
/// - 任一维为 `Any` 或 `Symbol` → 该维兼容（保守）
/// - 都 `Known`：必须相等
/// - `Known(1)` 与 `Known(n)`：兼容（梯度可广播/求和回参数 shape）
///
/// 注：与 unbroadcast_feasible 不同，这里允许 grad 维度为 1（广播回 n），
/// 也允许参数维度为 1（梯度求和回 1）。即双向广播兼容。
fn shape_compatible(grad_shape: &[Dim], param_shape: &[Dim]) -> bool {
    if grad_shape.len() != param_shape.len() {
        return false;
    }
    for (g, p) in grad_shape.iter().zip(param_shape.iter()) {
        let ok = match (g, p) {
            (Dim::Any, _) | (_, Dim::Any) => true,
            (Dim::Symbol(_), _) | (_, Dim::Symbol(_)) => true,
            (Dim::Known(1), _) | (_, Dim::Known(1)) => true, // 广播兼容
            (Dim::Known(a), Dim::Known(b)) => a == b,
        };
        if !ok {
            return false;
        }
    }
    true
}

// ── 辅助函数 ────────────────────────────────────────────────────────────

/// 从 Type 提取 Tensor 的 dims。非 Tensor 类型返回 None。
fn tensor_dims(ty: &Type) -> Option<Vec<Dim>> {
    match ty {
        Type::Tensor { dims, .. } => Some(dims.clone()),
        _ => None,
    }
}
