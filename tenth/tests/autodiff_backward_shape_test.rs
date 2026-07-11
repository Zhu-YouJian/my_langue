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
// 注意：编译器部 Phase 2 实现（backward_shape_pass.rs）尚在进行中。
//       pass case 应在实现完成后直接通过；fail case 标注 #[ignore] 待实现后确认。
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
// 跨算子 fail case 难以构造：合法的前向算子链路天然产生兼容的反向 shape。
// 前向 shape 不兼容会被 Phase 1 的单算子检查拦截；跨算子传播的额外价值主要在
// "不误报"（不破坏正常代码）。以下 fail case 标注 #[ignore]，待编译器部实现后
// 确认能否触发，或由编译器部提供能触发跨算子检查的构造方式。

#[test]
#[ignore = "Phase 2 跨算子 fail case 待编译器部实现后确认构造方式"]
fn test_phase2_cross_op_shape_mismatch_fail() {
    // 预期场景：构造跨算子传播后 grad shape 与 param shape 不兼容。
    // 当前难点：合法前向算子链路的反向 shape 天然兼容，难以构造 fail case。
    // 待编译器部实现 backward_shape_pass.rs 后，若存在能触发跨算子检查的构造，
    // 在此补充具体源码并将 #[ignore] 移除。
    let src = r#"
fn main() {
    new_grad();
    let x = zeros(3, 4);
    let t = param(x);
    let loss = t.sum();
    backward(loss);
}
"#;
    // 占位：当前为 pass case，待编译器部提供 fail 构造后改为 assert_compile_error_any
    assert_compiles(src);
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
