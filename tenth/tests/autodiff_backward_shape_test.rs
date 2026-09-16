//! 护城河 A 深化 Phase 1：编译期反向 shape 验证测试。
//!
//! 验证各算子的反向 shape 规则在编译期（lower 阶段）被检查：
//! - CrossEntropy：target shape 应为 [B] 或 [B,V]（与 logits[B,V] 匹配）
//! - MatMul/BMM：前向 shape 检查 + 反向 shape 验证
//! - Add/Sub/Mul/Div：广播 unbroadcast 可行性验证
//! - Reshape：元素数一致性验证
//! - Scatter/Gather/MaskedFill：基本 shape 保持验证
//!
//! 本测试文件为 Phase 1 交付物，依赖编译器部实现：
//! - `hir/lower/backward_shapes.rs`（反向 shape 规则表）
//! - `hir/lower/types.rs::check_method_shape` 扩展（matmul/bmm 反向验证）
//! - `hir/lower/lower_expr.rs` 扩展（cross_entropy 编译期 shape 检查）
//!
//! 参考现有测试：`tenth/tests/shape_check_compile_test.rs`（测试 API 与辅助函数风格）。
//!
//! 注意：编译器部完成实现前，fail case 测试可能因错误消息措辞不同而需要调整
//! `assert_compile_error_any` 的候选子串。pass case 应直接通过（不需匹配错误消息）。

use tenth::error::TenthError;
use tenth::hir::lower::Lowerer;
use tenth::lexer::lexer::Lexer;
use tenth::parser::parser::Parser;

/// 辅助：lower 源码，返回 Result<(), TenthError>。
/// 仿照 shape_check_compile_test.rs 的 lower() 函数。
fn lower(src: &str) -> Result<(), TenthError> {
    let mut lexer = Lexer::new(src);
    let tokens = lexer.tokenize().expect("lex");
    let mut parser = Parser::new(tokens);
    let program = parser.parse_program().expect("parse");
    let mut lowerer = Lowerer::new();
    lowerer.lower_program(&program).map(|_| ())
}

/// 辅助：断言 lower 失败且错误为 TypeError，消息包含指定子串。
fn assert_compile_error(src: &str, expected_msg_part: &str) {
    match lower(src) {
        Err(TenthError::TypeError { message, .. }) => {
            assert!(
                message.contains(expected_msg_part),
                "错误信息不包含预期子串 '{}'\n实际: {}",
                expected_msg_part, message
            );
        }
        Err(other) => panic!("期望 TypeError，实际: {:?}", other),
        Ok(_) => panic!("期望编译失败但成功了；期望错误包含 '{}'", expected_msg_part),
    }
}

/// 辅助：断言 lower 失败且错误为 TypeError，消息包含候选子串中的任意一个。
/// 用于编译器部错误消息措辞尚未最终确定的场景（如新增的 cross_entropy / reshape 检查）。
fn assert_compile_error_any(src: &str, candidates: &[&str]) {
    match lower(src) {
        Err(TenthError::TypeError { message, .. }) => {
            let matched = candidates.iter().any(|c| message.contains(c));
            assert!(
                matched,
                "错误信息不包含任一候选子串 {:?}\n实际: {}",
                candidates, message
            );
        }
        Err(other) => panic!("期望 TypeError，实际: {:?}", other),
        Ok(_) => panic!("期望编译失败但成功了；期望错误包含任一 {:?}", candidates),
    }
}

/// 辅助：断言 lower 成功（编译通过）。
fn assert_compiles(src: &str) {
    lower(src).unwrap_or_else(|e| panic!("期望编译通过但失败: {:?}", e));
}

/// 辅助：端到端跑源码（lower + 解释器），返回程序结果值。
/// 用于"编译通过 **且** 数值正确"的绿用例（W12 广播可还原形态）。
fn run_source(src: &str) -> Result<tenth::runtime::value::Value, TenthError> {
    let mut lexer = Lexer::new(src);
    let tokens = lexer.tokenize()?;
    let mut parser = Parser::new(tokens);
    let program = parser.parse_program()?;
    let mut lowerer = Lowerer::new();
    let hir = lowerer.lower_program(&program)?;
    let mut interp = tenth::runtime::interpreter::Interpreter::new(&hir);
    interp.fs_sandbox = None;
    interp.deadline_ms = None;
    match interp.execute_program(&hir)? {
        Some(v) => Ok(v),
        None => Ok(tenth::runtime::value::Value::Unit),
    }
}

/// 辅助：从 Value 提取 F64 Tensor 的扁平数据（形状用元素数间接断言）。
fn extract_f64_data(v: &tenth::runtime::value::Value) -> Vec<f64> {
    match v {
        tenth::runtime::value::Value::Tensor(t) => {
            let t = t.borrow();
            match &t.data {
                tenth::runtime::tensor::TensorData::F64(arr) => arr.iter().cloned().collect(),
                other => panic!("期望 F64 Tensor，got {:?}", other.dtype()),
            }
        }
        other => panic!("期望 Tensor，got {:?}", other),
    }
}

/// 辅助：断言两组 f64 数据近似相等（逐元素）。
fn assert_f64_approx(actual: &[f64], expected: &[f64], msg: &str) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "{}: 元素数不匹配（等价于 shape 还原失败）actual={:?} expected={:?}",
        msg,
        actual,
        expected
    );
    for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            (a - e).abs() < 1e-9,
            "{}: 第 {} 个元素不匹配 actual={} expected={}",
            msg,
            i,
            a,
            e
        );
    }
}

// ════════════════════════════════════════════════════════════════════════════
// 1. CrossEntropy shape 检查（最高优先级）
//
// cross_entropy(logits, target) 是 native 函数（非方法调用）。
// 规则（来自黑板算子表）：
//   logits shape [B, V]
//   target shape 应为 [B]（class indices 形式）或 [B, V]（概率分布形式）
//   其他 shape 应编译期报 TypeError
// ════════════════════════════════════════════════════════════════════════════

#[test]
fn cross_entropy_target_1d_compiles() {
    // cross_entropy(logits[B,V], targets[B]) → 编译通过
    // target 为 [B] 形式（每个样本一个类索引或标量权重）
    let src = r#"
fn main() {
    let logits = randn(4, 10);
    let targets = randn(4);
    let loss = cross_entropy(logits, targets);
    println(loss);
}
"#;
    assert_compiles(src);
}

#[test]
fn cross_entropy_target_2d_compiles() {
    // cross_entropy(logits[B,V], targets[B,V]) → 编译通过
    // target 为 [B, V] 形式（概率分布，与 logits 同 shape）
    let src = r#"
fn main() {
    let logits = randn(4, 10);
    let targets = randn(4, 10);
    let loss = cross_entropy(logits, targets);
    println(loss);
}
"#;
    assert_compiles(src);
}

#[test]
fn cross_entropy_target_swapped_dims_reports_error() {
    // cross_entropy(logits[B,V], targets[V,B]) → 编译期报错
    // target shape 错误：[V,B] 既非 [B] 也非 [B,V]
    let src = r#"
fn main() {
    let logits = randn(4, 10);
    let targets = randn(10, 4);
    let loss = cross_entropy(logits, targets);
}
"#;
    // 候选子串：编译器部错误消息措辞待确认，覆盖常见措辞
    assert_compile_error_any(src, &["shape", "cross_entropy", "维度", "target"]);
}

#[test]
fn cross_entropy_target_too_many_dims_reports_error() {
    // cross_entropy(logits[B,V], targets[B,V,K]) → 编译期报错
    // target 维度过多（3D，应为 1D 或 2D）
    let src = r#"
fn main() {
    let logits = randn(4, 10);
    let targets = randn(4, 10, 3);
    let loss = cross_entropy(logits, targets);
}
"#;
    assert_compile_error_any(src, &["shape", "cross_entropy", "维度", "target"]);
}

#[test]
fn cross_entropy_logits_2d_target_1d_mismatched_batch_reports_error() {
    // cross_entropy(logits[4,10], targets[5]) → 编译期报错
    // target [5] 与 logits batch=4 不匹配
    let src = r#"
fn main() {
    let logits = randn(4, 10);
    let targets = randn(5);
    let loss = cross_entropy(logits, targets);
}
"#;
    assert_compile_error_any(src, &["shape", "cross_entropy", "batch", "target"]);
}

#[test]
fn cross_entropy_logits_2d_target_2d_mismatched_v_reports_error() {
    // cross_entropy(logits[4,10], targets[4,8]) → 编译期报错
    // target [4,8] 的 V=8 与 logits V=10 不匹配
    let src = r#"
fn main() {
    let logits = randn(4, 10);
    let targets = randn(4, 8);
    let loss = cross_entropy(logits, targets);
}
"#;
    assert_compile_error_any(src, &["shape", "cross_entropy", "target"]);
}

// ════════════════════════════════════════════════════════════════════════════
// 2. MatMul / BMM 反向 shape 验证
//
// matmul 前向：(M,K) @ (K,N) → (M,N)，反向 grad shape 必须为 [M,N]
// bmm 前向：(B,M,K) @ (B,K,N) → (B,M,N)，反向 grad shape 必须为 [B,M,N]
//
// 注：matmul/bmm 反向 shape 天然与前向一致，前向检查已拦截大部分错误。
//     此处主要验证 pass case（编译通过），确保反向 shape 验证不误报。
// ════════════════════════════════════════════════════════════════════════════

#[test]
fn matmul_2d_correct_shape_compiles() {
    // a[M,K] @ b[K,N] → 编译通过
    let src = r#"
fn good() -> Tensor[f64, ..] {
    let a = zeros(3, 4);
    let b = zeros(4, 5);
    a.matmul(b)
}
"#;
    assert_compiles(src);
}

#[test]
fn bmm_3d_correct_shape_compiles() {
    // a[B,M,K] @ b[B,K,N]（bmm）→ 编译通过
    let src = r#"
fn good() -> Tensor[f64, ..] {
    let a = zeros(2, 3, 4);
    let b = zeros(2, 4, 5);
    a.bmm(b)
}
"#;
    assert_compiles(src);
}

#[test]
fn matmul_with_let_propagation_compiles() {
    // let 传播 shape，matmul 反向 shape 验证应通过
    let src = r#"
fn good() -> Tensor[f64, ..] {
    let a = zeros(3, 4);
    let b = zeros(4, 5);
    let c = a.matmul(b);
    c
}
"#;
    assert_compiles(src);
}

#[test]
fn matmul_in_autodiff_context_compiles() {
    // 在 autodiff 上下文中使用 matmul，反向 shape 验证应通过
    let src = r#"
fn main() {
    new_grad();
    let a = param(zeros(3, 4));
    let b = param(zeros(4, 5));
    let c = a.matmul(b);
    let loss = c.sum();
    backward(loss);
}
"#;
    assert_compiles(src);
}

// ════════════════════════════════════════════════════════════════════════════
// 3. Add/Sub/Mul/Div 广播 unbroadcast 验证
//
// 反向 unbroadcast 规则：grad 必须能 unbroadcast 回原参数 shape。
// 前向广播兼容的 shape，反向 unbroadcast 天然可行。
// 前向广播不兼容的 shape，编译期应报错（已有检查拦截）。
// ════════════════════════════════════════════════════════════════════════════

#[test]
fn add_row_broadcast_compiles() {
    // a[3,1] + b[1,4] → 编译通过（grad unbroadcast 可行）
    // 反向：grad[3,4] unbroadcast 到 [3,1]（沿 axis 1 求和）和 [1,4]（沿 axis 0 求和）
    let src = r#"
fn good() -> Tensor[f64, ..] {
    let a = zeros(3, 1);
    let b = zeros(1, 4);
    a + b
}
"#;
    assert_compiles(src);
}

#[test]
fn add_same_shape_compiles() {
    // a[3,4] + b[3,4] → 编译通过
    let src = r#"
fn good() -> Tensor[f64, ..] {
    let a = zeros(3, 4);
    let b = zeros(3, 4);
    a + b
}
"#;
    assert_compiles(src);
}

#[test]
fn add_incompatible_shapes_reports_error() {
    // a[3,4] + b[4,3] → 前向就报错（broadcast 失败），验证已有检查拦截
    let src = r#"
fn bad() -> Tensor[f64, ..] {
    let a = zeros(3, 4);
    let b = zeros(4, 3);
    a + b
}
"#;
    // 已有检查（check_binary_shape_compat）错误消息为 "shape 不兼容"
    assert_compile_error(src, "shape 不兼容");
}

#[test]
fn mul_row_broadcast_compiles() {
    // a[3,1] * b[1,4] → 编译通过（Mul 反向 unbroadcast 可行）
    let src = r#"
fn good() -> Tensor[f64, ..] {
    let a = zeros(3, 1);
    let b = zeros(1, 4);
    a * b
}
"#;
    assert_compiles(src);
}

#[test]
fn sub_col_broadcast_compiles() {
    // a[3,4] - b[3,1] → 编译通过（Sub 反向 unbroadcast 可行）
    let src = r#"
fn good() -> Tensor[f64, ..] {
    let a = zeros(3, 4);
    let b = zeros(3, 1);
    a - b
}
"#;
    assert_compiles(src);
}

#[test]
fn div_scalar_broadcast_compiles() {
    // a[3,4] / b[1] → 编译通过（Div 反向 unbroadcast 可行）
    let src = r#"
fn good() -> Tensor[f64, ..] {
    let a = zeros(3, 4);
    let b = zeros(1);
    a / b
}
"#;
    assert_compiles(src);
}

#[test]
fn add_in_autodiff_context_compiles() {
    // 在 autodiff 上下文中使用广播加法，反向 unbroadcast 验证应通过
    let src = r#"
fn main() {
    new_grad();
    let w = param(zeros(3, 4));
    let b = param(zeros(1, 4));
    let z = w + b;
    let loss = z.sum();
    backward(loss);
}
"#;
    assert_compiles(src);
}

// ════════════════════════════════════════════════════════════════════════════
// 4. Reshape 元素数一致性
//
// 反向规则：grad.reshape(s1)，要求 s1.numel == s2.numel。
// 编译期应检查 reshape 前后元素数一致，不一致报 TypeError。
// ════════════════════════════════════════════════════════════════════════════

#[test]
fn reshape_2d_to_1d_compiles() {
    // a[2,3].reshape(6) → 编译通过（6 == 2*3）
    let src = r#"
fn good() -> Tensor[f64, ..] {
    let a = zeros(2, 3);
    a.reshape(6)
}
"#;
    assert_compiles(src);
}

#[test]
fn reshape_1d_to_2d_compiles() {
    // a[6].reshape(2,3) → 编译通过（2*3 == 6）
    let src = r#"
fn good() -> Tensor[f64, ..] {
    let a = zeros(6);
    a.reshape(2, 3)
}
"#;
    assert_compiles(src);
}

#[test]
fn reshape_2d_to_3d_compiles() {
    // a[2,6].reshape(2,3,2) → 编译通过（2*3*2 == 12 == 2*6）
    let src = r#"
fn good() -> Tensor[f64, ..] {
    let a = zeros(2, 6);
    a.reshape(2, 3, 2)
}
"#;
    assert_compiles(src);
}

#[test]
fn reshape_element_count_mismatch_reports_error() {
    // a[2,3].reshape(7) → 编译期报错（2*3=6 ≠ 7）
    let src = r#"
fn bad() -> Tensor[f64, ..] {
    let a = zeros(2, 3);
    a.reshape(7)
}
"#;
    // 候选子串：编译器部错误消息措辞待确认
    assert_compile_error_any(src, &["reshape", "元素数", "numel", "shape", "不匹配"]);
}

#[test]
fn reshape_element_count_mismatch_2d_reports_error() {
    // a[2,3].reshape(3,4) → 编译期报错（2*3=6 ≠ 3*4=12）
    let src = r#"
fn bad() -> Tensor[f64, ..] {
    let a = zeros(2, 3);
    a.reshape(3, 4)
}
"#;
    assert_compile_error_any(src, &["reshape", "元素数", "numel", "shape", "不匹配"]);
}

#[test]
fn reshape_in_autodiff_context_compiles() {
    // 在 autodiff 上下文中使用 reshape，反向 shape 验证应通过
    let src = r#"
fn main() {
    new_grad();
    let x = param(zeros(2, 3));
    let y = x.reshape(6);
    let loss = y.sum();
    backward(loss);
}
"#;
    assert_compiles(src);
}

// ════════════════════════════════════════════════════════════════════════════
// 5. Scatter / Gather / MaskedFill 基本 shape 验证
//
// 反向规则：
//   - Scatter: grad(base.shape), grad(src.shape)；index 不可微
//   - Gather: grad(base.shape)；index 不可微
//   - MaskedFill: grad * (1-mask)；grad shape == a shape
//
// 此处验证基本 pass case（shape 保持），确保反向 shape 验证不误报。
// ════════════════════════════════════════════════════════════════════════════

#[test]
fn masked_fill_preserves_shape_compiles() {
    // masked_fill(mask, value) 保持原 shape，反向 grad shape == a shape
    let src = r#"
fn good() -> Tensor[f64, ..] {
    let a = zeros(3, 4);
    let mask = zeros(3, 4);
    a.masked_fill(mask, 0.0)
}
"#;
    assert_compiles(src);
}

#[test]
fn masked_fill_in_autodiff_context_compiles() {
    // 在 autodiff 上下文中使用 masked_fill，反向 shape 验证应通过
    let src = r#"
fn main() {
    new_grad();
    let x = param(zeros(3, 4));
    let mask = zeros(3, 4);
    let y = x.masked_fill(mask, -1.0);
    let loss = y.sum();
    backward(loss);
}
"#;
    assert_compiles(src);
}

#[test]
fn scatter_basic_compiles() {
    // scatter 基本调用，编译通过
    let src = r#"
fn good() -> Tensor[f64, ..] {
    let base = zeros(4, 4);
    let src = zeros(2, 4);
    let index = zeros(2, 4);
    base.scatter(0, index, src)
}
"#;
    assert_compiles(src);
}

#[test]
fn gather_basic_compiles() {
    // gather 基本调用，编译通过
    let src = r#"
fn good() -> Tensor[f64, ..] {
    let base = zeros(4, 4);
    let index = zeros(2, 4);
    base.gather(0, index)
}
"#;
    assert_compiles(src);
}

// ════════════════════════════════════════════════════════════════════════════
// 6. 组合场景：多算子链式调用
//
// 验证多算子组合时编译期 shape 检查不误报，autodiff 上下文完整通过。
// ════════════════════════════════════════════════════════════════════════════

#[test]
fn matmul_then_add_bias_compiles() {
    // linear 层：x @ w + b，matmul + broadcast add
    let src = r#"
fn good() -> Tensor[f64, ..] {
    let x = zeros(3, 4);
    let w = zeros(4, 5);
    let b = zeros(1, 5);
    x.matmul(w) + b
}
"#;
    assert_compiles(src);
}

#[test]
fn linear_with_cross_entropy_compiles() {
    // 完整训练场景：linear → cross_entropy
    let src = r#"
fn main() {
    new_grad();
    let x = param(zeros(4, 10));
    let w = param(zeros(10, 5));
    let b = param(zeros(1, 5));
    let logits = x.matmul(w) + b;
    let targets = zeros(4);
    let loss = cross_entropy(logits, targets);
    backward(loss);
}
"#;
    assert_compiles(src);
}

#[test]
fn reshape_then_matmul_compiles() {
    // reshape → matmul 链式：a[6].reshape(2,3) → [2,3], b[3,5]，内侧 K=3 匹配
    let src = r#"
fn good() -> Tensor[f64, ..] {
    let a = zeros(6);
    let b = zeros(3, 5);
    let a_2d = a.reshape(2, 3);
    a_2d.matmul(b)
}
"#;
    assert_compiles(src);
}

#[test]
fn full_autodiff_chain_compiles() {
    // 完整 autodiff 链：matmul → reshape → add → sum → backward
    let src = r#"
fn main() {
    new_grad();
    let x = param(zeros(2, 3));
    let w = param(zeros(3, 4));
    let y = x.matmul(w);
    let z = y.reshape(8);
    let b = param(zeros(1, 8));
    let out = z.reshape(1, 8) + b;
    let loss = out.sum();
    backward(loss);
}
"#;
    assert_compiles(src);
}

// ════════════════════════════════════════════════════════════════════════════
// Phase 2：跨算子反向 shape 传播测试
//
// 验证 HIR 分析 pass 在 start_grad/new_grad → backward 的 grad 区域内：
// - 直线代码：从 loss 反向传播梯度 shape，验证 param 的梯度 shape 与参数 shape 兼容
// - 控制流回退：grad 区域内含 if/for/while 时跳过验证（不报错也不验证）
// - 复杂链路：多算子组合（linear + activation + loss）不误报
//
// 调用范式说明（与 Phase 1 一致）：
//   Tenth 中 start_grad() / new_grad() 均为无参数调用（同义，创建 Tape 并开始记录），
//   param(t) 标记张量为可训练参数。任务描述中的概念性伪代码 `start_grad(t)` 在 Tenth
//   中实际写法为 `new_grad(); let t_param = param(t);`。本节测试统一沿用 Phase 1 的
//   `new_grad()` + `param()` + `backward()` 范式，确保与现有 autodiff 语义一致。
//
// 状态（W11/W12）：Phase 2 pass（backward_shape_pass.rs）已落地；W11 修掉了
// `find_grad_regions` 只遍历 `stmts`、而 `lower_expr` 把函数体最后一条 Expr 提升为
// `final_expr` 的缺陷（此前 `fn main() { …; backward(loss); }` 这种最常见写法下整个
// pass **空跑** ⇒ 既有 pass case 都是空转）。fail case 现有真实构造，本文件**无 `#[ignore]`**。
// ════════════════════════════════════════════════════════════════════════════

// ─── A. 直线代码 pass case（跨算子反向 shape 传播成功） ─────────────────────

#[test]
fn test_phase2_sum_backward_ok() {
    // 场景：sum 的反向是 broadcast，grad(t) shape == t shape
    // t shape [3,4] → loss shape [] (scalar) → grad(t) = [3,4] ✓
    let src = r#"
fn main() {
    new_grad();
    let x = zeros(3, 4);
    let t = param(x);
    let loss = t.sum();
    backward(loss);
}
"#;
    assert_compiles(src);
}

#[test]
fn test_phase2_matmul_sum_ok() {
    // 场景：matmul + sum 链路
    // ta shape [3,4], b shape [4,5] → c shape [3,5] → loss shape []
    // grad(c)=[3,5], grad(ta)=[3,4] ✓（matmul 反向 shape 天然匹配）
    let src = r#"
fn main() {
    new_grad();
    let a = zeros(3, 4);
    let b = zeros(4, 5);
    let ta = param(a);
    let c = ta.matmul(b);
    let loss = c.sum();
    backward(loss);
}
"#;
    assert_compiles(src);
}

#[test]
fn test_phase2_add_broadcast_sum_ok() {
    // 场景：add（广播）+ sum 链路
    // ta shape [3,1], b shape [1,4] → c shape [3,4] → loss shape []
    // grad(c)=[3,4], grad(ta)=unbroadcast([3,4],[3,1])=[3,1] ✓
    let src = r#"
fn main() {
    new_grad();
    let a = zeros(3, 1);
    let b = zeros(1, 4);
    let ta = param(a);
    let c = ta + b;
    let loss = c.sum();
    backward(loss);
}
"#;
    assert_compiles(src);
}

#[test]
fn test_phase2_mul_broadcast_sum_ok() {
    // 场景：mul（广播）+ sum 链路
    // ta shape [3,1], b shape [1,4] → c shape [3,4]
    // grad(c)=[3,4], grad(ta)=unbroadcast(grad*b, [3,1])=[3,1] ✓
    let src = r#"
fn main() {
    new_grad();
    let a = zeros(3, 1);
    let b = zeros(1, 4);
    let ta = param(a);
    let c = ta * b;
    let loss = c.sum();
    backward(loss);
}
"#;
    assert_compiles(src);
}

#[test]
fn test_phase2_reshape_sum_ok() {
    // 场景：reshape + sum 链路
    // t shape [2,3] → y shape [6] → loss shape []
    // grad(y)=[6], grad(t)=y.grad.reshape([2,3])=[2,3] ✓（numel 一致）
    let src = r#"
fn main() {
    new_grad();
    let x = zeros(2, 3);
    let t = param(x);
    let y = t.reshape(6);
    let loss = y.sum();
    backward(loss);
}
"#;
    assert_compiles(src);
}

#[test]
fn test_phase2_linear_chain_ok() {
    // 场景：linear 层 matmul + add(bias) + sum
    // tw shape [4,5], x shape [3,4], b shape [1,5]
    // logits shape [3,5] → loss shape []
    // grad(logits)=[3,5], grad(tw)=[4,5] ✓（matmul 反向 + add unbroadcast）
    let src = r#"
fn main() {
    new_grad();
    let x = zeros(3, 4);
    let w = zeros(4, 5);
    let b = zeros(1, 5);
    let tw = param(w);
    let logits = x.matmul(tw) + b;
    let loss = logits.sum();
    backward(loss);
}
"#;
    assert_compiles(src);
}

#[test]
fn test_phase2_linear_relu_ce_ok() {
    // 场景：linear + relu + cross_entropy 复杂链路
    // x shape [4,8], tw shape [8,10], targets shape [4]
    // h = x.matmul(tw) shape [4,10]
    // logits = h.relu() shape [4,10]
    // loss = cross_entropy(logits, targets) shape []
    // grad(logits)=[4,10], grad(h)=[4,10], grad(tw)=[8,10] ✓
    let src = r#"
fn main() {
    new_grad();
    let x = zeros(4, 8);
    let w = zeros(8, 10);
    let targets = zeros(4);
    let tw = param(w);
    let h = x.matmul(tw);
    let logits = h.relu();
    let loss = cross_entropy(logits, targets);
    backward(loss);
}
"#;
    assert_compiles(src);
}

#[test]
fn test_phase2_multi_param_independent_ok() {
    // 场景：多个 param 各自 grad shape 独立验证
    // tw1 shape [3,4], tw2 shape [4,5]
    // c = tw1.matmul(tw2) shape [3,5] → loss shape []
    // grad(c)=[3,5], grad(tw1)=[3,4], grad(tw2)=[4,5] ✓
    let src = r#"
fn main() {
    new_grad();
    let w1 = zeros(3, 4);
    let w2 = zeros(4, 5);
    let tw1 = param(w1);
    let tw2 = param(w2);
    let c = tw1.matmul(tw2);
    let loss = c.sum();
    backward(loss);
}
"#;
    assert_compiles(src);
}

#[test]
fn test_phase2_start_grad_synonym_ok() {
    // 场景：使用 start_grad()（与 new_grad() 同义）验证 grad 区域识别
    // 确认 Phase 2 pass 同时识别 start_grad 和 new_grad 作为 grad 区域起点
    let src = r#"
fn main() {
    start_grad();
    let x = zeros(3, 4);
    let t = param(x);
    let loss = t.sum();
    backward(loss);
}
"#;
    assert_compiles(src);
}

#[test]
fn test_phase2_chained_matmul_reshape_add_sum_ok() {
    // 场景：多算子链式 matmul → reshape → add → sum
    // x shape [2,3], w shape [3,4] → y shape [2,4]
    // z = y.reshape(8) shape [8]
    // b shape [1,8], out = z.reshape(1,8) + b shape [1,8]
    // loss = out.sum() shape []
    // 反向传播链：grad(out)=[1,8] → grad(z)=[8] → grad(y)=[2,4] → grad(tw)=[3,4] ✓
    let src = r#"
fn main() {
    new_grad();
    let x = zeros(2, 3);
    let w = zeros(3, 4);
    let b = zeros(1, 8);
    let tw = param(w);
    let y = x.matmul(tw);
    let z = y.reshape(8);
    let out = z.reshape(1, 8) + b;
    let loss = out.sum();
    backward(loss);
}
"#;
    assert_compiles(src);
}

// ─── B. 直线代码 fail case（编译期报错） ───────────────────────────────────
//
// 跨算子 fail case：链路上**每个算子单独看都合法**（前向 shape 兼容、单算子反向
// shape 也兼容），只有把**上游算子的输出 shape 传播到下游**之后才暴露出反向 shape
// 约束不满足 ⇒ 编译期报错。这正是不做跨算子反向 shape 传播就抓不到的一类错误。
//
// 与下文 E 节（AUDIT-11.4.66 守卫）的红色用例**不重复**：E 节两条红用例的 base 直接
// 是 `param(zeros(...))`（把字面量 dim 纳入校验即可抓到）；本用例的 base **秩由上游
// `reshape` 产生**（[2,3] → [6]），index 秩 2 ⇒ 必须真的跨算子传播才会报错。
//
// AUDIT-11.4.66 / 11.4.67 落地后 fail 构造已存在，`#[ignore]` 于 W12 移除。

#[test]
fn test_phase2_cross_op_shape_mismatch_fail() {
    // 链路：w[2,3] --reshape(6)--> flat[6] --gather(dim=0, idx[2,3])--> y
    // gather 反向 scatter-add 要求 index 与 base **同秩**：idx 秩 2 ≠ base(flat) 秩 1 ⇒ 报错
    let src = r#"
fn main() {
    new_grad();
    let w = param(zeros(2, 3));
    let flat = w.reshape(6);
    let idx = zeros(2, 3);
    let y = gather(flat, 0, idx);
    let loss = y.sum();
    backward(loss);
}
"#;
    // 错误原文：编译期跨算子反向 shape 传播失败（gather）：gather 反向 shape 不兼容：
    //           index 秩 2 ≠ base 秩 1（反向 scatter-add 需要 index 与 base 同秩…）
    assert_compile_error(src, "反向 shape 不兼容");
}

// ─── C. 控制流回退 pass case（编译通过，不验证） ───────────────────────────
//
// Phase 2 保守策略：grad 区域内含 if/else/while/for/loop 时跳过验证（不报错）。
// 以下测试验证控制流回退不会误报，编译应通过。

#[test]
fn test_phase2_if_in_grad_region_ok() {
    // 场景：grad 区域内含 if/else，Phase 2 应跳过验证（不报错）
    let src = r#"
fn main() {
    new_grad();
    let x = zeros(3, 4);
    let t = param(x);
    let s = t.sum();
    let cond = s > 0.0;
    let c = if cond {
        t.sum()
    } else {
        t.mean()
    };
    backward(c);
}
"#;
    assert_compiles(src);
}

#[test]
fn test_phase2_for_loop_in_grad_region_ok() {
    // 场景：grad 区域内含 for 循环，Phase 2 应跳过验证（不报错）
    let src = r#"
fn main() {
    new_grad();
    let x = zeros(3, 4);
    let t = param(x);
    let mut loss = t.sum();
    for i in 0..3 {
        loss = loss + t.sum();
    }
    backward(loss);
}
"#;
    assert_compiles(src);
}

#[test]
fn test_phase2_while_loop_in_grad_region_ok() {
    // 场景：grad 区域内含 while 循环，Phase 2 应跳过验证（不报错）
    let src = r#"
fn main() {
    new_grad();
    let x = zeros(3, 4);
    let t = param(x);
    let mut loss = t.sum();
    let mut i = 0;
    while i < 3 {
        loss = loss + t.sum();
        i = i + 1;
    }
    backward(loss);
}
"#;
    assert_compiles(src);
}

#[test]
fn test_phase2_nested_control_flow_in_grad_region_ok() {
    // 场景：grad 区域内含嵌套控制流（for + if），Phase 2 应跳过验证（不报错）
    let src = r#"
fn main() {
    new_grad();
    let x = zeros(3, 4);
    let t = param(x);
    let mut loss = t.sum();
    for i in 0..3 {
        if i > 0 {
            loss = loss + t.sum();
        }
    }
    backward(loss);
}
"#;
    assert_compiles(src);
}

// ─── D. 边界场景 ────────────────────────────────────────────────────────────

#[test]
fn test_phase2_no_grad_region_compiles() {
    // 场景：无 start_grad/backward 的普通代码，Phase 2 pass 不应介入（不误报）
    let src = r#"
fn main() {
    let a = zeros(3, 4);
    let b = zeros(4, 5);
    let c = a.matmul(b);
    let s = c.sum();
    println(s);
}
"#;
    assert_compiles(src);
}

#[test]
fn test_phase2_scalar_param_sum_ok() {
    // 场景：标量 param 的反向传播
    // t shape [1] → loss = t.sum() shape []
    // grad(t) = [1] ✓
    let src = r#"
fn main() {
    new_grad();
    let x = zeros(1);
    let t = param(x);
    let loss = t.sum();
    backward(loss);
}
"#;
    assert_compiles(src);
}

#[test]
fn test_phase2_grad_region_with_stop_grad_ok() {
    // 场景：grad 区域内含 stop_grad（停止记录），Phase 2 应能处理或保守回退
    let src = r#"
fn main() {
    new_grad();
    let x = zeros(3, 4);
    let t = param(x);
    let loss = t.sum();
    stop_grad();
    backward(loss);
}
"#;
    assert_compiles(src);
}

// ════════════════════════════════════════════════════════════════════════════
// E. AUDIT-11.4.66 守卫：**字面量实参**的算子必须真的进反向 shape 校验
//
// 病灶：`collect_tensor_op` 对非 `Var` 实参（**字面量**）直接 `return None`
//       ⇒ `gather(w, 0, idx)` / `index_select(w, 0, idx)` 这类**最常见用法**
//       整个算子被排除在校验集合外 = **假覆盖**（守卫长得像守卫，实为装饰）。
// 修法：字面量按**位置占位**纳入（整型字面量的值经 `input_int_consts` 传入，
//       即 `Literal → Known` 静态值），并让这两条 arm 具备**可失败**的校验：
//       - `gather`：反向 scatter-add 需要 index 与 base **同秩**（运行时同判据）；
//       - `index_select`：反向 scatter-add 需定位 base 的 dim 槽 ⇒ dim 必须落在秩内。
//
// 红色用例证明"校验真的跑"（dim 是字面量 ⇒ 修复前必然跳过 ⇒ 编译期不报错）；
// 绿色用例证明"合法用法不误报"。
//
// 注：同文件 `test_phase2_cross_op_shape_mismatch_fail` 曾是标着 "待编译器部提供
//     fail 构造" 的 `#[ignore]` 占位；W12 已按本节红色用例的构造方式与 helper 转正
//     （改为"上游 reshape 改秩 → 下游 gather 秩不匹配"的链式形态，与本节两条红用例
//     不重复），`ignored` 基线随之 23 → 22。
// ════════════════════════════════════════════════════════════════════════════

/// 红色（gather）：字面量 dim + index 与 base **不同秩** ⇒ 编译期报错。
///
/// `dim=0` 是字面量 ⇒ 修复前 `collect_tensor_op` 直接跳过整个 `gather`
/// ⇒ 该错误不但运行期才发现，而且**编译期根本没有任何校验**。
#[test]
fn test_phase2_gather_literal_dim_rank_mismatch_fails() {
    let src = r#"
fn main() {
    new_grad();
    let w = param(zeros(2, 3));
    let idx = zeros(2, 3, 4);
    let y = gather(w, 0, idx);
    let loss = y.sum();
    backward(loss);
}
"#;
    assert_compile_error_any(src, &["反向 shape", "gather"]);
}

/// 绿色（gather）：字面量 dim + index 与 base 同秩 ⇒ 编译通过（不误报）。
#[test]
fn test_phase2_gather_literal_dim_rank_match_ok() {
    let src = r#"
fn main() {
    new_grad();
    let w = param(zeros(4, 4));
    let idx = zeros(2, 4);
    let y = gather(w, 0, idx);
    let loss = y.sum();
    backward(loss);
}
"#;
    assert_compiles(src);
}

/// 红色（index_select）：字面量 dim **越界**（base 秩 2，dim=5） ⇒ 编译期报错。
///
/// 前向类型推断对越界 dim 保守降级为同秩全 `Any`（不报错），故本错误**只能**来自
/// 反向传播 pass ⇒ 报错即证明"字面量真的进了校验集合"（修复前 dim 是字面量 ⇒ 跳过）。
#[test]
fn test_phase2_index_select_literal_dim_out_of_range_fails() {
    let src = r#"
fn main() {
    new_grad();
    let w = param(zeros(2, 3));
    let idx = zeros(2);
    let y = index_select(w, 5, idx);
    let loss = y.sum();
    backward(loss);
}
"#;
    assert_compile_error_any(src, &["反向 shape", "index_select"]);
}

/// 绿色（index_select）：字面量 dim 在秩内 ⇒ 编译通过（不误报）。
#[test]
fn test_phase2_index_select_literal_dim_in_range_ok() {
    let src = r#"
fn main() {
    new_grad();
    let w = param(zeros(2, 4));
    let idx = zeros(3);
    let y = index_select(w, 0, idx);
    let loss = y.sum();
    backward(loss);
}
"#;
    assert_compiles(src);
}

/// 对照（非字面量 dim 不得破坏）：dim 用**变量**（符号）时两条 arm 也必须不误报
/// （变量 dim 无法静态判越界 ⇒ 保守放行；gather 的秩校验仍生效且此处同秩）。
#[test]
fn test_phase2_literal_dim_variable_dim_both_ok() {
    let src = r#"
fn main() {
    new_grad();
    let w = param(zeros(4, 4));
    let idx = zeros(2, 4);
    let d = 0;
    let y = gather(w, d, idx);
    let loss = y.sum();
    backward(loss);
}
"#;
    assert_compiles(src);
}

// ════════════════════════════════════════════════════════════════════════════
// F. W12：**秩扩展广播**的参数梯度 shape（编译期判据 vs 运行期 unbroadcast）
//
// 病灶：`backward_shape_pass.rs::shape_compatible` 旧判据要求**秩相等**，而运行期
// `unbroadcast`（`runtime/autodiff/backward.rs`）是"param 左补 1 → 对 param 维为 1 的
// 轴求和 → 按元素数 reshape 回 param" ⇒ `Add` 的秩扩展广播（param [4] 与 [1,4]）
// 在运行期**本就可还原**，却被编译期 Phase 2 误拒（W12 报的误报）。
//
// 修法：判据改成"运行期可还原"（grad 秩 ≥ param 秩，且右对齐后每维 param==1 或
// param==grad），逐维规则与 `backward_shapes.rs::unbroadcast_feasible` 共用单一权威；
// 不可还原的形态（秩不足 / 两个 >1 维度不等 / param>1 而 grad 为 1）**仍然报错**
// ——见 E 节三条红色用例（gather / index_select / reshape→gather）与
// `backward_shape_pass.rs` 内的 `shape_compatible` 单元测试。
//
// 下列绿用例**同时**断言"编译通过"与"数值正确"：只断言编译通过会漏掉运行期失败
// （元素数断言同时证明真的 unbroadcast 过，而不是把上游 [1,4] 梯度直接透传）。
// ════════════════════════════════════════════════════════════════════════════

/// 绿色（W12 原报构造）：`param [4] + b [1,4]` ⇒ 编译必须通过（旧判据因秩不等误拒）。
#[test]
fn test_phase2_rank_extended_broadcast_add_compiles() {
    let src = r#"
fn main() {
    new_grad();
    let t = param(zeros(4));
    let b = zeros(1, 4);
    let y = t + b;
    let loss = y.sum();
    backward(loss);
}
"#;
    assert_compiles(src);
}

/// 绿色（W12 原报构造）+ **数值**：grad(t) 必须还原成 shape [4]、值 `[1,1,1,1]`。
///
/// 运行期实测（探针 B/`.agents/tmp/w12_probe_b.th`）同构造打印 `[1.0, 1.0, 1.0, 1.0]`。
#[test]
fn test_phase2_rank_extended_broadcast_add_grad_value() {
    let src = r#"
fn run() -> Tensor[f64, ..] {
    new_grad();
    let t = param(tensor([1.0, 2.0, 3.0, 4.0]));
    let b = tensor([[10.0, 20.0, 30.0, 40.0]]);
    let y = t + b;
    let loss = y.sum();
    backward(loss);
    grad(t)
}
run()
"#;
    let v = run_source(src).expect("秩扩展广播 Add 反向应成功");
    // 元素数 4（不是 8）⇒ 梯度确实被 unbroadcast 回 [4] 而非透传 [1,4]
    assert_f64_approx(
        &extract_f64_data(&v),
        &[1.0, 1.0, 1.0, 1.0],
        "rank-extended Add grad(t)",
    );
}

/// 绿色：标量式参数 `param [1] * m [2,3]` ⇒ grad(s) = sum(m) = `[21]`（shape [1]）。
///
/// 覆盖"左补 1 后 param 维仍为 1 ⇒ 全部广播轴都要求和"的还原形态。
#[test]
fn test_phase2_scalarish_param_broadcast_mul_grad_value() {
    let src = r#"
fn run() -> Tensor[f64, ..] {
    new_grad();
    let s = param(tensor([2.0]));
    let m = tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
    let y = s * m;
    let loss = y.sum();
    backward(loss);
    grad(s)
}
run()
"#;
    let v = run_source(src).expect("标量式参数广播 Mul 反向应成功");
    assert_f64_approx(&extract_f64_data(&v), &[21.0], "scalar-ish Mul grad(s)");
}

/// 绿色：列广播 `param [3,1] + b [3,4]` ⇒ grad(t) = `[[4],[4],[4]]`（沿被广播的轴求和）。
#[test]
fn test_phase2_col_broadcast_param_grad_value() {
    let src = r#"
fn run() -> Tensor[f64, ..] {
    new_grad();
    let t = param(tensor([[1.0], [2.0], [3.0]]));
    let b = tensor([[10.0, 20.0, 30.0, 40.0], [50.0, 60.0, 70.0, 80.0], [90.0, 100.0, 110.0, 120.0]]);
    let y = t + b;
    let loss = y.sum();
    backward(loss);
    grad(t)
}
run()
"#;
    let v = run_source(src).expect("列广播 Add 反向应成功");
    assert_f64_approx(&extract_f64_data(&v), &[4.0, 4.0, 4.0], "col-broadcast Add grad(t)");
}

/// 绿色：grad 比 param **多一维且该维为 1**：`param [2,3] * q [1,2,3]` ⇒ grad(p) = q 广播值。
///
/// 这是"秩扩展 + 求和"复合形态：运行期 sum 掉 lead 维并 reshape 回 [2,3]；
/// 值 `[[1,2,3],[4,5,6]]`（行优先）证明维度归约与内存布局都正确。
#[test]
fn test_phase2_leading_dim1_broadcast_param_grad_value() {
    let src = r#"
fn run() -> Tensor[f64, ..] {
    new_grad();
    let p = param(tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]));
    let q = tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).reshape(1, 2, 3);
    let y = p * q;
    let loss = y.sum();
    backward(loss);
    grad(p)
}
run()
"#;
    let v = run_source(src).expect("lead-dim-1 广播 Mul 反向应成功");
    assert_f64_approx(
        &extract_f64_data(&v),
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "lead-dim-1 Mul grad(p)",
    );
}

