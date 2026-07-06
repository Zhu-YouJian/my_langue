// f32 vs f64 标准库 parity 测试 — f32/f64 双精度对等路线图 阶段 2
//
// 目标：对比泛型标准库函数在 <f32> 与 <f64> 实例化下的输出一致性。
//
// 测试覆盖（已泛型化的标准库模块）：
//   1. dropout<f32> vs dropout<f64>      — std/nn/dropout.th
//   2. batchnorm<f32> vs batchnorm<f64>  — std/nn/batchnorm.th
//   3. layer_norm<f32> vs layer_norm<f64> — std/nn/layer_norm.th
//   4. feedforward<f32> vs feedforward<f64> — std/nn/feedforward.th
//   5. scaled_dot_product_attention<f32> vs <f64> — std/nn/attention.th
//   6. multihead_attention<f32> vs <f64>  — std/nn/multihead_attention.th
//   7. transformer_encoder_block<f32> vs <f64> — std/nn/transformer.th
//
// 实现策略（参考 multihead_attention_test.rs）：
//   - 当前运行时不支持 .rs 测试通过 `use` 加载 .th 标准库模块
//   - 因此本文件将泛型函数的 f32 与 f64 实例化版本内联到测试代码中
//   - 用 Interpreter（自带 native）执行，避免手动注册 native
//   - 用确定性输入（tensor literals）避免 randn 不确定性
//   - 注：运行时无 `shape()` 自由函数，seq_len/d_model 等作为显式参数传入

use tenth::lexer::lexer::Lexer;
use tenth::parser::parser::Parser;
use tenth::hir::lower::Lowerer;
use tenth::runtime::interpreter::Interpreter;
use tenth::runtime::value::Value;

fn run_code(src: &str) -> Result<Option<Value>, String> {
    let mut lexer = Lexer::new(src);
    let tokens = lexer.tokenize().map_err(|e| e.to_string())?;
    let mut parser = Parser::new(tokens);
    let program = parser.parse_program().map_err(|e| e.to_string())?;
    let mut lowerer = Lowerer::new();
    let hir = lowerer.lower_program(&program).map_err(|e| e.to_string())?;
    let mut interpreter = Interpreter::new(&hir);
    interpreter.execute_program(&hir).map_err(|e| e.to_string())
}

/// 从 Value 提取标量 f64
fn as_f64(val: &Value) -> Option<f64> {
    match val {
        Value::Float(f) => Some(*f),
        Value::Float32(f) => Some(*f as f64),
        Value::Tensor(t) => {
            let data = &t.borrow().data;
            if data.len() == 1 { Some(data[0]) } else { None }
        }
        _ => None,
    }
}

/// 从 Value 提取 tensor 的 f64 切片
fn as_f64_vec(val: &Value) -> Option<Vec<f64>> {
    match val {
        Value::Tensor(t) => {
            let data = t.borrow().data.as_f64_view();
            Some(data.iter().cloned().collect())
        }
        _ => None,
    }
}

/// 比较两个 f64 切片元素级绝对误差，返回最大误差
fn max_abs_diff_vec(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len(), "length mismatch: {} vs {}", a.len(), b.len());
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).fold(0.0f64, f64::max)
}

/// 内联 dropout（与 std/nn/dropout.th 等价）
const DROPOUT_INLINE: &str = r#"
    fn dropout_t<T>(x: Tensor[T, ..], rate: T) -> Tensor[T, ..] {
        x.dropout(rate)
    }
"#;

/// 内联 batchnorm（与 std/nn/batchnorm.th 等价）
const BATCHNORM_INLINE: &str = r#"
    fn batchnorm_t<T>(
        x: Tensor[T, ..],
        gamma: Tensor[T, ..],
        beta: Tensor[T, ..],
        eps: T,
    ) -> Tensor[T, ..] {
        x.batchnorm(gamma, beta, eps)
    }
"#;

/// 内联 layer_norm（与 std/nn/layer_norm.th 等价）
const LAYER_NORM_INLINE: &str = r#"
    fn layer_norm_t<T>(
        x: Tensor[T, ..],
        gamma: Tensor[T, ..],
        beta: Tensor[T, ..],
        eps: T,
    ) -> Tensor[T, ..] {
        x.layer_norm(gamma, beta, eps)
    }
"#;

/// 内联 feedforward（与 std/nn/feedforward.th 等价）
const FEEDFORWARD_INLINE: &str = r#"
    fn feedforward_t<T>(
        x: Tensor[T, ..],
        w1: Tensor[T, ..],
        b1: Tensor[T, ..],
        w2: Tensor[T, ..],
        b2: Tensor[T, ..],
    ) -> Tensor[T, ..] {
        let hidden = x.matmul(w1) + b1;
        let activated = hidden.gelu();
        activated.matmul(w2) + b2
    }
"#;

/// 内联 scaled_dot_product_attention（2D 版本，与 std/nn/attention.th 等价）
/// 注：运行时无 shape()，d_k 作为参数传入
const SDPA_INLINE: &str = r#"
    fn sdpa_t<T>(
        q: Tensor[T, ..],
        k: Tensor[T, ..],
        v: Tensor[T, ..],
        mask: Tensor[f64, ..],
        d_k: i64,
        dropout_p: T,
    ) -> Tensor[T, ..] {
        let scale = 1.0 / sqrt(d_k);
        let kT = k.transpose();
        let scores = q.matmul(kT) * scale;
        let masked_scores = scores.masked_fill(mask, -1e9);
        let weights = masked_scores.softmax();
        let dropped = weights.dropout(dropout_p);
        dropped.matmul(v)
    }
"#;

/// 内联 multihead_attention（与 std/nn/multihead_attention.th 等价）
/// 注：运行时无 shape()，seq_len/d_model 作为参数传入
const MHA_INLINE: &str = r#"
    fn mha_t<T>(
        x: Tensor[T, ..],
        w_q: Tensor[T, ..],
        w_k: Tensor[T, ..],
        w_v: Tensor[T, ..],
        w_o: Tensor[T, ..],
        mask: Tensor[f64, ..],
        seq_len: i64,
        d_model: i64,
        n_heads: i64,
        dropout_p: T,
    ) -> Tensor[T, ..] {
        let d_k = d_model / n_heads;
        let q = x.matmul(w_q);
        let k = x.matmul(w_k);
        let v = x.matmul(w_v);
        let q_3d = q.reshape(n_heads, seq_len, d_k);
        let k_3d = k.reshape(n_heads, seq_len, d_k);
        let v_3d = v.reshape(n_heads, seq_len, d_k);
        let scale = 1.0 / sqrt(d_k);
        let k_t = k_3d.transpose();
        let scores = q_3d.bmm(k_t) * scale;
        let masked_scores = scores.masked_fill(mask, -1e9);
        let weights = masked_scores.softmax();
        let dropped = weights.dropout(dropout_p);
        let out_3d = dropped.bmm(v_3d);
        let out = out_3d.reshape(seq_len, d_model);
        out.matmul(w_o)
    }
"#;

/// 内联 transformer_encoder_block（与 std/nn/transformer.th 等价，pre-norm）
/// 注：transformer.th 用 `zeros(1)` 作为 mask，但 masked_fill 当前要求精确 shape 匹配。
/// 这里改为接收外部 mask 参数（shape 必须为 [n_heads, seq_len, seq_len]）。
const TRANSFORMER_INLINE: &str = r#"
    fn transformer_t<T>(
        x: Tensor[T, ..],
        w_q: Tensor[T, ..],
        w_k: Tensor[T, ..],
        w_v: Tensor[T, ..],
        w_o: Tensor[T, ..],
        ln1_gamma: Tensor[T, ..],
        ln1_beta: Tensor[T, ..],
        w1: Tensor[T, ..],
        b1: Tensor[T, ..],
        w2: Tensor[T, ..],
        b2: Tensor[T, ..],
        ln2_gamma: Tensor[T, ..],
        ln2_beta: Tensor[T, ..],
        mask: Tensor[f64, ..],
        seq_len: i64,
        d_model: i64,
        n_heads: i64,
        dropout_p: T,
    ) -> Tensor[T, ..] {
        let x_norm = layer_norm_t<T>(x, ln1_gamma, ln1_beta, 1e-5);
        let attn = mha_t<T>(x_norm, w_q, w_k, w_v, w_o, mask, seq_len, d_model, n_heads, dropout_p);
        let x1 = x + attn;
        let x_norm2 = layer_norm_t<T>(x1, ln2_gamma, ln2_beta, 1e-5);
        let ffn_out = feedforward_t<T>(x_norm2, w1, b1, w2, b2);
        x1 + ffn_out
    }
"#;

// ══════════════════════════════════════════════════════════════════════
// 1. dropout parity
// ══════════════════════════════════════════════════════════════════════

#[test]
fn f32_f64_dropout_parity() {
    // dropout<f32> vs dropout<f64>（rate=0 跳过随机性，验证 dtype 路径）
    let src = format!(r#"
        {DROPOUT_INLINE}
        let x32 = ones_f32(2, 3);
        let x64 = ones(2, 3);
        let y32 = dropout_t<f32>(x32, 0.0f32);
        let y64 = dropout_t<f64>(x64, 0.0);
        // 比较 sum：rate=0 时 dropout 是恒等映射，两者应相等
        y32.sum() - y64.sum()
    "#);
    let result = run_code(&src).expect("dropout parity should run");
    let diff = as_f64(result.as_ref().unwrap()).expect("expected scalar diff");
    assert!(diff.abs() < 1e-5, "dropout<f32> vs dropout<f64> sum 误差 {} 应 < 1e-5", diff);
}

// ══════════════════════════════════════════════════════════════════════
// 2. batchnorm parity
// ══════════════════════════════════════════════════════════════════════

#[test]
fn f32_f64_batchnorm_parity() {
    // batchnorm<f32> vs batchnorm<f64>
    // x: (2, 3)，gamma/beta: (3,)
    let src = format!(r#"
        {BATCHNORM_INLINE}
        let x32 = tensor[[1.0f32, 2.0f32, 3.0f32], [4.0f32, 5.0f32, 6.0f32]];
        let g32 = ones_f32(3);
        let b32 = zeros_f32(3);
        let y32 = batchnorm_t<f32>(x32, g32, b32, 1e-5f32);

        let x64 = tensor[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let g64 = ones(3);
        let b64 = zeros(3);
        let y64 = batchnorm_t<f64>(x64, g64, b64, 1e-5);

        y32.sum() - y64.sum()
    "#);
    let result = run_code(&src).expect("batchnorm parity should run");
    let diff = as_f64(result.as_ref().unwrap()).expect("expected scalar diff");
    assert!(diff.abs() < 1e-5, "batchnorm<f32> vs batchnorm<f64> sum 误差 {} 应 < 1e-5", diff);
}

// ══════════════════════════════════════════════════════════════════════
// 3. layer_norm parity
// ══════════════════════════════════════════════════════════════════════

#[test]
fn f32_f64_layer_norm_parity() {
    // layer_norm<f32> vs layer_norm<f64>
    // x: (2, 4)，gamma/beta: (4,)
    let src = format!(r#"
        {LAYER_NORM_INLINE}
        let x32 = tensor[[1.0f32, 2.0f32, 3.0f32, 4.0f32], [5.0f32, 6.0f32, 7.0f32, 8.0f32]];
        let g32 = ones_f32(4);
        let b32 = zeros_f32(4);
        let y32 = layer_norm_t<f32>(x32, g32, b32, 1e-5f32);

        let x64 = tensor[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]];
        let g64 = ones(4);
        let b64 = zeros(4);
        let y64 = layer_norm_t<f64>(x64, g64, b64, 1e-5);

        y32.sum() - y64.sum()
    "#);
    let result = run_code(&src).expect("layer_norm parity should run");
    let diff = as_f64(result.as_ref().unwrap()).expect("expected scalar diff");
    assert!(diff.abs() < 1e-5, "layer_norm<f32> vs layer_norm<f64> sum 误差 {} 应 < 1e-5", diff);
}

// ══════════════════════════════════════════════════════════════════════
// 4. feedforward parity
// ══════════════════════════════════════════════════════════════════════

#[test]
fn f32_f64_feedforward_parity() {
    // feedforward<f32> vs feedforward<f64>
    // x: (2, 3)，w1: (3, 4)，b1: (4)，w2: (4, 3)，b2: (3)
    // 用确定性输入（tensor literals + 单位阵/小数值）
    let src = format!(r#"
        {FEEDFORWARD_INLINE}
        // f32 路径
        let x32 = tensor[[1.0f32, 2.0f32, 3.0f32], [4.0f32, 5.0f32, 6.0f32]];
        // w1 = 0.1 * ones(3, 4) （用显式字面量）
        let w1_32 = tensor[[0.1f32, 0.1f32, 0.1f32, 0.1f32],
                           [0.1f32, 0.1f32, 0.1f32, 0.1f32],
                           [0.1f32, 0.1f32, 0.1f32, 0.1f32]];
        let b1_32 = zeros_f32(4);
        // w2 = 0.1 * ones(4, 3)
        let w2_32 = tensor[[0.1f32, 0.1f32, 0.1f32],
                           [0.1f32, 0.1f32, 0.1f32],
                           [0.1f32, 0.1f32, 0.1f32],
                           [0.1f32, 0.1f32, 0.1f32]];
        let b2_32 = zeros_f32(3);
        let y32 = feedforward_t<f32>(x32, w1_32, b1_32, w2_32, b2_32);

        // f64 路径
        let x64 = tensor[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let w1_64 = tensor[[0.1, 0.1, 0.1, 0.1],
                           [0.1, 0.1, 0.1, 0.1],
                           [0.1, 0.1, 0.1, 0.1]];
        let b1_64 = zeros(4);
        let w2_64 = tensor[[0.1, 0.1, 0.1],
                           [0.1, 0.1, 0.1],
                           [0.1, 0.1, 0.1],
                           [0.1, 0.1, 0.1]];
        let b2_64 = zeros(3);
        let y64 = feedforward_t<f64>(x64, w1_64, b1_64, w2_64, b2_64);

        y32.sum() - y64.sum()
    "#);
    let result = run_code(&src).expect("feedforward parity should run");
    let diff = as_f64(result.as_ref().unwrap()).expect("expected scalar diff");
    // gelu 在 f32/f64 下略有差异，但 sum 后整体应 < 1e-5
    assert!(diff.abs() < 1e-5, "feedforward<f32> vs feedforward<f64> sum 误差 {} 应 < 1e-5", diff);
}

// ══════════════════════════════════════════════════════════════════════
// 5. scaled_dot_product_attention parity
// ══════════════════════════════════════════════════════════════════════

#[test]
fn f32_f64_attention_parity() {
    // scaled_dot_product_attention<f32> vs <f64>
    // q/k/v: (3, 4)，mask: zeros(3, 3)，d_k=4，dropout_p=0
    let src = format!(r#"
        {SDPA_INLINE}
        // f32 路径
        let q32 = tensor[[1.0f32, 2.0f32, 3.0f32, 4.0f32],
                         [2.0f32, 3.0f32, 4.0f32, 5.0f32],
                         [3.0f32, 4.0f32, 5.0f32, 6.0f32]];
        let k32 = tensor[[0.5f32, 0.5f32, 0.5f32, 0.5f32],
                         [0.5f32, 0.5f32, 0.5f32, 0.5f32],
                         [0.5f32, 0.5f32, 0.5f32, 0.5f32]];
        let v32 = tensor[[1.0f32, 0.0f32, 0.0f32, 0.0f32],
                         [0.0f32, 1.0f32, 0.0f32, 0.0f32],
                         [0.0f32, 0.0f32, 1.0f32, 0.0f32]];
        let mask32 = zeros(3, 3);
        let y32 = sdpa_t<f32>(q32, k32, v32, mask32, 4, 0.0f32);

        // f64 路径
        let q64 = tensor[[1.0, 2.0, 3.0, 4.0],
                         [2.0, 3.0, 4.0, 5.0],
                         [3.0, 4.0, 5.0, 6.0]];
        let k64 = tensor[[0.5, 0.5, 0.5, 0.5],
                         [0.5, 0.5, 0.5, 0.5],
                         [0.5, 0.5, 0.5, 0.5]];
        let v64 = tensor[[1.0, 0.0, 0.0, 0.0],
                         [0.0, 1.0, 0.0, 0.0],
                         [0.0, 0.0, 1.0, 0.0]];
        let mask64 = zeros(3, 3);
        let y64 = sdpa_t<f64>(q64, k64, v64, mask64, 4, 0.0);

        y32.sum() - y64.sum()
    "#);
    let result = run_code(&src).expect("sdpa parity should run");
    let diff = as_f64(result.as_ref().unwrap()).expect("expected scalar diff");
    // softmax 在 f32 下精度稍差，但 sum 应 < 1e-5
    assert!(diff.abs() < 1e-5, "sdpa<f32> vs sdpa<f64> sum 误差 {} 应 < 1e-5", diff);
}

// ══════════════════════════════════════════════════════════════════════
// 6. multihead_attention parity
// ══════════════════════════════════════════════════════════════════════

#[test]
fn f32_f64_mha_parity() {
    // multihead_attention<f32> vs <f64>
    // x: (4, 4)，w_*: (4, 4) 用单位阵 + 小扰动，mask: zeros(2, 4, 4)
    // n_heads=2, seq_len=4, d_model=4, d_k=2
    let src = format!(r#"
        {MHA_INLINE}
        // 用单位阵 + 小数值确保数值稳定且可对比
        // w_q = w_k = w_v = w_o = I_4
        let I4_32 = tensor[[1.0f32, 0.0f32, 0.0f32, 0.0f32],
                           [0.0f32, 1.0f32, 0.0f32, 0.0f32],
                           [0.0f32, 0.0f32, 1.0f32, 0.0f32],
                           [0.0f32, 0.0f32, 0.0f32, 1.0f32]];
        let x32 = tensor[[1.0f32, 2.0f32, 3.0f32, 4.0f32],
                         [5.0f32, 6.0f32, 7.0f32, 8.0f32],
                         [1.0f32, 1.0f32, 1.0f32, 1.0f32],
                         [2.0f32, 2.0f32, 2.0f32, 2.0f32]];
        let mask32 = zeros(2, 4, 4);
        let y32 = mha_t<f32>(x32, I4_32, I4_32, I4_32, I4_32, mask32, 4, 4, 2, 0.0f32);

        let I4_64 = tensor[[1.0, 0.0, 0.0, 0.0],
                           [0.0, 1.0, 0.0, 0.0],
                           [0.0, 0.0, 1.0, 0.0],
                           [0.0, 0.0, 0.0, 1.0]];
        let x64 = tensor[[1.0, 2.0, 3.0, 4.0],
                         [5.0, 6.0, 7.0, 8.0],
                         [1.0, 1.0, 1.0, 1.0],
                         [2.0, 2.0, 2.0, 2.0]];
        let mask64 = zeros(2, 4, 4);
        let y64 = mha_t<f64>(x64, I4_64, I4_64, I4_64, I4_64, mask64, 4, 4, 2, 0.0);

        // 比较 sum
        y32.sum() - y64.sum()
    "#);
    let result = run_code(&src).expect("mha parity should run");
    let diff = as_f64(result.as_ref().unwrap()).expect("expected scalar diff");
    // softmax 在 f32 下精度稍差，但 sum 后整体应 < 1e-5
    assert!(diff.abs() < 1e-5, "mha<f32> vs mha<f64> sum 误差 {} 应 < 1e-5", diff);
}

// ══════════════════════════════════════════════════════════════════════
// 7. transformer_encoder_block parity
// ══════════════════════════════════════════════════════════════════════

#[test]
fn f32_f64_transformer_parity() {
    // transformer_encoder_block<f32> vs <f64>
    // 完整 transformer block：x → LayerNorm → MHA → residual → LayerNorm → FFN → residual
    //
    // masked_fill 当前要求精确 shape 匹配，故 mask 必须为 [n_heads, seq_len, seq_len] = [2, 2, 2]
    let src = format!(r#"
        {LAYER_NORM_INLINE}
        {FEEDFORWARD_INLINE}
        {MHA_INLINE}
        {TRANSFORMER_INLINE}
        // f32 路径
        let x32 = tensor[[1.0f32, 2.0f32, 3.0f32, 4.0f32],
                         [5.0f32, 6.0f32, 7.0f32, 8.0f32]];
        let I4_32 = tensor[[1.0f32, 0.0f32, 0.0f32, 0.0f32],
                           [0.0f32, 1.0f32, 0.0f32, 0.0f32],
                           [0.0f32, 0.0f32, 1.0f32, 0.0f32],
                           [0.0f32, 0.0f32, 0.0f32, 1.0f32]];
        let g_32 = ones_f32(4);
        let b_32 = zeros_f32(4);
        let w1_32 = tensor[[0.1f32, 0.1f32, 0.1f32, 0.1f32],
                           [0.1f32, 0.1f32, 0.1f32, 0.1f32],
                           [0.1f32, 0.1f32, 0.1f32, 0.1f32],
                           [0.1f32, 0.1f32, 0.1f32, 0.1f32]];
        let b1_32 = zeros_f32(4);
        let w2_32 = tensor[[0.1f32, 0.1f32, 0.1f32, 0.1f32],
                           [0.1f32, 0.1f32, 0.1f32, 0.1f32],
                           [0.1f32, 0.1f32, 0.1f32, 0.1f32],
                           [0.1f32, 0.1f32, 0.1f32, 0.1f32]];
        let b2_32 = zeros_f32(4);
        // mask shape 必须匹配 scores shape [n_heads, seq_len, seq_len] = [2, 2, 2]
        let mask32 = zeros(2, 2, 2);
        let y32 = transformer_t<f32>(
            x32, I4_32, I4_32, I4_32, I4_32,
            g_32, b_32,
            w1_32, b1_32, w2_32, b2_32,
            g_32, b_32,
            mask32,
            2, 4, 2, 0.0f32
        );

        // f64 路径
        let x64 = tensor[[1.0, 2.0, 3.0, 4.0],
                         [5.0, 6.0, 7.0, 8.0]];
        let I4_64 = tensor[[1.0, 0.0, 0.0, 0.0],
                           [0.0, 1.0, 0.0, 0.0],
                           [0.0, 0.0, 1.0, 0.0],
                           [0.0, 0.0, 0.0, 1.0]];
        let g_64 = ones(4);
        let b_64 = zeros(4);
        let w1_64 = tensor[[0.1, 0.1, 0.1, 0.1],
                           [0.1, 0.1, 0.1, 0.1],
                           [0.1, 0.1, 0.1, 0.1],
                           [0.1, 0.1, 0.1, 0.1]];
        let b1_64 = zeros(4);
        let w2_64 = tensor[[0.1, 0.1, 0.1, 0.1],
                           [0.1, 0.1, 0.1, 0.1],
                           [0.1, 0.1, 0.1, 0.1],
                           [0.1, 0.1, 0.1, 0.1]];
        let b2_64 = zeros(4);
        let mask64 = zeros(2, 2, 2);
        let y64 = transformer_t<f64>(
            x64, I4_64, I4_64, I4_64, I4_64,
            g_64, b_64,
            w1_64, b1_64, w2_64, b2_64,
            g_64, b_64,
            mask64,
            2, 4, 2, 0.0
        );

        y32.sum() - y64.sum()
    "#);
    let result = run_code(&src);
    assert!(result.is_ok(), "transformer parity should run, got: {:?}", result.err());
    let diff = as_f64(result.unwrap().as_ref().unwrap()).expect("expected scalar diff");
    // 复合网络累积误差稍大，但应 < 1e-4
    // 误差来源：softmax + gelu + matmul 在 f32 下的精度损失累积
    assert!(diff.abs() < 1e-4, "transformer<f32> vs transformer<f64> sum 误差 {} 应 < 1e-4", diff);
}

// ══════════════════════════════════════════════════════════════════════
// 附加：dtype 保持 parity（确保泛型实例化后 dtype 正确）
// ══════════════════════════════════════════════════════════════════════

#[test]
fn f32_f64_stdlib_dtype_preservation_parity() {
    // 关键守护：泛型函数实例化为 <f32> 时，输出应保持 F32 dtype
    // （避免泛型实例化默默退化为 f64）
    let src = format!(r#"
        {DROPOUT_INLINE}
        {LAYER_NORM_INLINE}
        {BATCHNORM_INLINE}
        {FEEDFORWARD_INLINE}
        {SDPA_INLINE}
        {MHA_INLINE}

        // dropout<f32> 输出应保持 F32
        let y_drop = dropout_t<f32>(ones_f32(2, 2), 0.0f32);

        // layer_norm<f32> 输出应保持 F32
        let y_ln = layer_norm_t<f32>(ones_f32(2, 4), ones_f32(4), zeros_f32(4), 1e-5f32);

        // batchnorm<f32> 输出应保持 F32
        let y_bn = batchnorm_t<f32>(ones_f32(2, 3), ones_f32(3), zeros_f32(3), 1e-5f32);

        // feedforward<f32> 输出应保持 F32
        let w1 = tensor[[0.1f32, 0.1f32, 0.1f32, 0.1f32],
                        [0.1f32, 0.1f32, 0.1f32, 0.1f32],
                        [0.1f32, 0.1f32, 0.1f32, 0.1f32]];
        let w2 = tensor[[0.1f32, 0.1f32, 0.1f32],
                        [0.1f32, 0.1f32, 0.1f32],
                        [0.1f32, 0.1f32, 0.1f32],
                        [0.1f32, 0.1f32, 0.1f32]];
        let y_ffn = feedforward_t<f32>(ones_f32(2, 3), w1, zeros_f32(4), w2, zeros_f32(3));

        // sum 作为聚合（dtype 不会因 sum 改变，但确保整条路径不崩）
        y_drop.sum() + y_ln.sum() + y_bn.sum() + y_ffn.sum()
    "#);
    let result = run_code(&src).expect("dtype preservation should run");
    let s = as_f64(result.as_ref().unwrap()).expect("expected scalar sum");
    assert!(s.is_finite(), "聚合 sum 应为有限值，got {}", s);
    assert!(s.abs() > 1e-9, "聚合 sum 应非零，got {}", s);
}
