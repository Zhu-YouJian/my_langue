//! 标准库 smoke 测试全覆盖（L2.4）
//!
//! 对每个「可用」标准库模块走**真实 use 路径**的最小测试：
//!   `use std::<path>::<item>` + 调用 1-2 个代表性函数/常量，
//!   断言能编译运行且关键值正确（stdout 含 `= true`）。
//!
//! 与既有 `stdlib_test.rs`（直接调 native / 内联实现）不同，本文件通过
//! 真实二进制 `tenth.exe run <tmp.th>` 子进程执行，验证模块 `use` 后
//! 可用的真实路径；任何「模块不可用」回归会立即暴露。
//!
//! 路径选择（见 docs/stdlib-可用性盘点.md §四）：
//! - 默认走 VM（默认路径）
//! - a1（VM 高阶函数/闭包值调用缺口）影响 curry / collections·iter 高阶 /
//!   collections·collections 高阶 / accumulate_loop / runtime 闭包 → 走解释器
//! - autograd 需 Rust 端注册 op 才可调用 → 仅验证模块 use 可编译（import 检查）
//! - http/net 用 127.0.0.1:1（本机拒绝连接，不触网）验证 Result 路径

use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};

/// 被测二进制（同 crate 的 bin target，cargo test 会自动构建）。
const TENTH_EXE: &str = env!("CARGO_BIN_EXE_tenth");
/// 包根目录（= tenth/），作为子进程 cwd 使 `use std::...` 解析到 tenth/std/。
const TENTH_DIR: &str = env!("CARGO_MANIFEST_DIR");

static COUNTER: AtomicU64 = AtomicU64::new(0);

/// 运行一个 .th 程序（走真实 use 路径），返回 (exit_code, stdout, stderr)。
fn run_th(prog: &str, use_vm: bool) -> (i32, String, String) {
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!("tenth_smoke_{}_{}", std::process::id(), n));
    std::fs::create_dir_all(&dir).unwrap();
    let file = dir.join("smoke.th");
    std::fs::write(&file, prog).unwrap();

    let mut cmd = Command::new(TENTH_EXE);
    cmd.arg("run").arg(&file).current_dir(TENTH_DIR);
    if !use_vm {
        cmd.env("TENTH_NO_VM", "1");
    }
    let out = cmd.output().expect("运行 tenth.exe 失败");
    let stdout = String::from_utf8_lossy(&out.stdout).to_string();
    let stderr = String::from_utf8_lossy(&out.stderr).to_string();
    let code = out.status.code().unwrap_or(-1);
    let _ = std::fs::remove_dir_all(&dir);
    (code, stdout, stderr)
}

/// 断言程序 exit 0 且 stdout 含 `= true`（末尾布尔表达式求值为真）。
fn smoke(name: &str, prog: &str, use_vm: bool) {
    let (code, stdout, stderr) = run_th(prog, use_vm);
    assert!(
        code == 0 && stdout.contains("= true"),
        "[{}] 失败: exit={}\n--- stdout ---\n{}\n--- stderr ---\n{}",
        name,
        code,
        stdout,
        stderr
    );
}

fn smoke_vm(name: &str, prog: &str) {
    smoke(name, prog, true)
}

/// a1 缺口模块：走解释器路径（TENTH_NO_VM=1）。
fn smoke_interp(name: &str, prog: &str) {
    smoke(name, prog, false)
}

/// 断言 exit 0 且 stderr 含标记（用于 io 模块 eprint/eprintln）。
fn smoke_stderr(name: &str, prog: &str, marker: &str) {
    let (code, _stdout, stderr) = run_th(prog, true);
    assert!(
        code == 0 && stderr.contains(marker),
        "[{}] 失败: exit={}\n--- stderr ---\n{}",
        name,
        code,
        stderr
    );
}

// ══════════════════════════════════════════════════════════════════
// math
// ══════════════════════════════════════════════════════════════════

const M01_MATH_CONSTANTS: &str = r#"
use std::math::constants::*

PI > 3.14 && PI < 3.15 && E > 2.7 && TAU > 6.28 && PHI > 1.6 &&
deg2rad(180.0) > 3.14 && deg2rad(180.0) < 3.15 && rad2deg(PI) > 179.9 && rad2deg(PI) < 180.1 &&
exp(1.0) > 2.7 && pow_scalar(2.0, 10.0) == 1024.0 && floor(2.7) == 2.0 && ceil(2.1) == 3.0 &&
asin(1.0) > 1.5 && acos(1.0) == 0.0 && atan(1.0) > 0.7 && tan(0.0) == 0.0
"#;

const M02_MATH_STATS: &str = r#"
use std::math::stats::*

mean([1,2,3,4]) == 2.5 && median([3,1,2]) == 2.0 && median([1,2,3,4]) == 2.5 &&
min([3,1,2]) == 1.0 && max([3,1,2]) == 3.0 &&
variance([2,4,4,4,5,5,7,9]) > 3.99 && variance([2,4,4,4,5,5,7,9]) < 4.01 &&
stddev([2,4,4,4,5,5,7,9]) > 1.99 && stddev([2,4,4,4,5,5,7,9]) < 2.01
"#;

// ══════════════════════════════════════════════════════════════════
// nn
// ══════════════════════════════════════════════════════════════════

const M03_NN_ACTIVATIONS: &str = r#"
use std::nn::activations::*

let x = tensor[[-2.0, -1.0, 0.0, 1.0, 2.0]];
let r = relu<f64>(x);
let s = sigmoid<f64>(x);
let sm = softmax<f64>(x);
let lr = leaky_relu<f64>(x, 0.1);
let lrd = leaky_relu_default(x);
let g = gelu<f64>(x);
let e = exp<f64>(x);
// L2.5：leaky_relu 已修符号（负半轴取 slope*x 负值），断言数值：
//   lr  = leaky_relu([-2,-1,0,1,2], 0.1) = [-0.2,-0.1,0,1,2] → sum = 2.7
//   lrd = leaky_relu_default([-2,-1,0,1,2])（slope=0.01）= [-0.02,-0.01,0,1,2] → sum = 2.97
r.sum() == 3.0 && s.mean() > 0.0 && s.mean() < 1.0 && sm.sum() > 0.99 && sm.sum() < 1.01 &&
lr.numel() == 5 && lr.sum() > 2.69 && lr.sum() < 2.71 &&
lrd.numel() == 5 && lrd.sum() > 2.96 && lrd.sum() < 2.98 &&
g.numel() == 5 && e.sum() > 0.0
"#;

const M04_NN_LINEAR: &str = r#"
use std::nn::linear::linear

let x = tensor[[-2.0, -1.0, 0.0, 1.0, 2.0]];
let w = tensor[[1.0, 0.5, 0.0, 1.0, 1.0], [0.5, 0.5, 0.0, 0.0, 1.0]];
let b = tensor[[0.0, 1.0]];
let out = linear<f64>(x, w, b);
out.numel() == 2
"#;

const M05_NN_LOSS: &str = r#"
use std::nn::loss::*

let a = tensor[[1.0, 2.0, 3.0]];
let b = tensor[[1.0, 4.0, 3.0]];
let m = mse<f64>(a, b);
let ml = mse_loss<f64>(a, b);
let l1 = l1_loss<f64>(a, b);
let h = huber_loss<f64>(a, b, 1.0);
let bce = binary_cross_entropy(0.9, 1.0);
m > 1.3 && m < 1.4 && ml.numel() == 3 && l1.numel() == 3 && h > 0.0 && bce > 0.0
"#;

const M06_NN_FEEDFORWARD: &str = r#"
use std::nn::feedforward::feedforward

let x = tensor[[1.0, 0.0]];
let w1 = tensor[[0.5, 0.0, 1.0], [0.0, 0.5, 1.0]];
let b1 = tensor[[0.0, 0.0, 0.0]];
let w2 = tensor[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
let b2 = tensor[[0.0, 0.0]];
let y = feedforward<f64>(x, w1, b1, w2, b2);
y.numel() == 2
"#;

const M07_NN_LAYER_NORM: &str = r#"
use std::nn::layer_norm::layer_norm

let x = tensor[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
let g = ones(3);
let b = zeros(3);
let y = layer_norm<f64>(x, g, b, 1e-5);
y.numel() == 6 && y.mean() > -0.5 && y.mean() < 0.5
"#;

const M08_NN_MHA: &str = r#"
use std::nn::multihead_attention::multihead_attention

let x = tensor[[1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 1.0, 0.0]];
let wq = tensor[[1.0,0.0,0.0,0.0],[0.0,1.0,0.0,0.0],[0.0,0.0,1.0,0.0],[0.0,0.0,0.0,1.0]];
let wk = wq;
let wv = wq;
let wo = wq;
let mask = zeros(2, 2, 2);
let y = multihead_attention<f64>(x, wq, wk, wv, wo, mask, 2, 4, 2, 0.0);
y.numel() == 8 && y.mean() > -1.0 && y.mean() < 1.0
"#;

const M09_NN_POOL: &str = r#"
use std::nn::pool::*

let x = tensor[[1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0,11.0,12.0,13.0,14.0,15.0,16.0]].reshape(1,1,4,4);
let y = max_pool2d<f64>(x, 2, 2, 0);
let a = avg_pool2d<f64>(x, 2, 2, 0);
y.numel() == 4 && y.max_val() == 16.0 && a.numel() == 4
"#;

const M10_NN_OPS: &str = r#"
use std::nn::ops::*

let a = tensor[[1.0, 2.0, 3.0]];
let b = tensor[[3.0, 2.0, 1.0]];
let g = gt<f64>(a, b);
let l = lt<f64>(a, b);
let e = eq<f64>(a, b);
let ge = ge<f64>(a, b);
let w = where_<f64>(gt<f64>(a, b), a, b);
g.sum() == 1.0 && l.sum() == 1.0 && e.sum() == 1.0 && ge.sum() == 2.0 && w.sum() == 8.0
"#;

const M11_NN_BATCHNORM: &str = r#"
use std::nn::batchnorm::batchnorm

let x = tensor[[1.0,2.0,3.0,4.0]].reshape(1,1,2,2);
let g = ones(1);
let b = zeros(1);
let y = batchnorm<f64>(x, g, b, 1e-5);
y.numel() == 4 && y.mean() > -0.5 && y.mean() < 0.5
"#;

const M12_NN_DROPOUT: &str = r#"
use std::nn::dropout::dropout

let x = tensor[[1.0, 2.0, 3.0, 4.0]];
let y = dropout<f64>(x, 0.0);
y.sum() == 10.0
"#;

const M13_NN_POS_ENC: &str = r#"
use std::nn::positional_encoding::positional_encoding

let pe = positional_encoding(4, 8);
pe.numel() == 32
"#;

const M14_NN_ATTENTION: &str = r#"
use std::nn::attention::scaled_dot_product_attention

let q = tensor[[1.0, 0.0], [0.0, 1.0]];
let k = tensor[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
let v = tensor[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
let mask = zeros(2, 3);
let y = scaled_dot_product_attention<f64>(q, k, v, mask, 2, 0.0);
y.numel() == 4
"#;

const M15_NN_CONV: &str = r#"
use std::nn::conv::conv2d

let x = tensor[[1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0,11.0,12.0,13.0,14.0,15.0,16.0]].reshape(1,1,4,4);
let w = tensor[[1.0,0.0],[0.0,0.0]].reshape(1,1,2,2);
let b = zeros(1);
let y = conv2d<f64>(x, w, b, 2, 2, 1, 0);
y.numel() == 9
"#;

const M16_NN_EMBEDDING: &str = r#"
use std::nn::embedding::embedding

let weight = tensor[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
let indices = tensor[[0.0, 2.0]];
let e = embedding<f64>(weight, indices, 2, 2);
e.numel() == 4 && e.sum() == 14.0
"#;

const M17_NN_TRANSFORMER: &str = r#"
use std::nn::transformer::transformer_encoder_block

let x = tensor[[1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 1.0, 0.0]];
let wq = tensor[[1.0,0.0,0.0,0.0],[0.0,1.0,0.0,0.0],[0.0,0.0,1.0,0.0],[0.0,0.0,0.0,1.0]];
let wk = wq;
let wv = wq;
let wo = wq;
let g1 = ones(4);
let b1 = zeros(4);
let fw1 = tensor[[0.5,0.0,0.0,0.0,0.0,0.0,0.0,0.0],[0.0,0.5,0.0,0.0,0.0,0.0,0.0,0.0],[0.0,0.0,0.5,0.0,0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.5,0.0,0.0,0.0,0.0]];
let fb1 = zeros(8);
let fw2 = tensor[[1.0,0.0,0.0,0.0],[0.0,1.0,0.0,0.0],[0.0,0.0,1.0,0.0],[0.0,0.0,0.0,1.0],[0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0]];
let fb2 = zeros(4);
let g2 = ones(4);
let b2 = zeros(4);
let y = transformer_encoder_block<f64>(x, wq, wk, wv, wo, g1, b1, fw1, fb1, fw2, fb2, g2, b2, 2, 4, 2, 0.0);
y.numel() == 8 && y.mean() > -2.0 && y.mean() < 2.0
"#;

// ══════════════════════════════════════════════════════════════════
// init / optim
// ══════════════════════════════════════════════════════════════════

const M18_INIT: &str = r#"
use std::init::initializers::*

xavier_uniform<f64>(3, 4).numel() == 12 && he_normal<f64>(3, 4).numel() == 12 && zeros_init<f64>(2, 3).sum() == 0.0
"#;

const M19_OPTIM_SGD: &str = r#"
use std::optim::sgd::*

new_grad();
let w = param(tensor[[1.0, 2.0]]);
let loss = (w * w).sum();
backward(loss);
let nw = sgd_step<f64>(w, 0.1);
nw.numel() == 2
"#;

const M20_OPTIM_ADAM: &str = r#"
use std::optim::adam::adam_step

new_grad();
let w = param(tensor[[1.0, 2.0]]);
let loss = (w * w).sum();
backward(loss);
let m = zeros(1, 2);
let v = zeros(1, 2);
let (nw, nm, nv) = adam_step<f64>(w, m, v, 0.001, 0.9, 0.999, 1e-8, 0.9, 0.999);
nw.numel() == 2 && nm.numel() == 2 && nv.numel() == 2
"#;

const M21_OPTIM_ADAMW: &str = r#"
use std::optim::adamw::adamw_step_w

new_grad();
let w = param(tensor[[1.0, 2.0]]);
let loss = (w * w).sum();
backward(loss);
let m = zeros(1, 2);
let v = zeros(1, 2);
let nw = adamw_step_w<f64>(w, m, v, 0.001, 0.9, 0.999, 1e-8, 0.01, 0.9, 0.999);
nw.numel() == 2
"#;

const M22_OPTIM_RMSPROP: &str = r#"
use std::optim::rmsprop::rmsprop_step

new_grad();
let w = param(tensor[[1.0, 2.0]]);
let loss = (w * w).sum();
backward(loss);
let g2 = zeros(1, 2);
let (nw, ng2) = rmsprop_step<f64>(w, g2, 0.001, 0.9, 1e-8);
nw.numel() == 2 && ng2.numel() == 2
"#;

const M23_OPTIM_ADAGRAD: &str = r#"
use std::optim::adagrad::adagrad_step

new_grad();
let w = param(tensor[[1.0, 2.0]]);
let loss = (w * w).sum();
backward(loss);
let g2 = zeros(1, 2);
let (nw, ng2) = adagrad_step<f64>(w, g2, 0.01, 1e-8);
nw.numel() == 2 && ng2.numel() == 2
"#;

const M24_OPTIM_CLIP: &str = r#"
use std::optim::clip::*

new_grad();
let w = param(tensor[[1.0, 2.0]]);
let loss = (w * w).sum();
backward(loss);
let gw = grad(w);
let c1 = clip_grad_by_value<f64>(gw, 1.0);
let c2 = clip_grad_by_norm<f64>(gw, 1.0);
c1.max_val() <= 1.0 && c2.numel() == 2
"#;

const M25_OPTIM_LR_SCHEDULE: &str = r#"
use std::optim::lr_schedule::*

cosine_lr(1.0, 0, 10) == 1.0 && step_lr(1.0, 5, 5, 0.5) == 0.5 && exp_lr(1.0, 2, 0.5) == 0.25 &&
warmup_lr(1.0, 5, 10) == 0.5 && LR_PI > 3.14 && LR_EPS < 1e-9
"#;

const M26_OPTIM_ACCUMULATE: &str = r#"
use std::optim::accumulate::accumulate_grad

new_grad();
let w = param(tensor[[1.0, 2.0]]);
let loss = (w * w).sum();
backward(loss);
let g = accumulate_grad<f64>(w, 4);
g.numel() == 2
"#;

// ══════════════════════════════════════════════════════════════════
// collections
// ══════════════════════════════════════════════════════════════════

const M27_COLLECTIONS_HASHSET: &str = r#"
use std::collections::hashset::*

// 注：HashMap 共享可变语义——insert 会改写共享的 inner，故以 s2 为准断言
let s = insert(insert(new(), 1), 2);
let s2 = insert(s, 3);
contains(s2, 1) && contains(s2, 2) && contains(s2, 3) && len(s2) == 3 &&
!is_empty(s2) && to_array(s2).len() == 3 && is_empty(clear(s2))
"#;

const M28_COLLECTIONS_COLLECTIONS: &str = r#"
use std::collections::collections::sum
use std::collections::collections::product
use std::collections::collections::from_entries

// 注：数组字面量作参数会被推断为 Tensor，Vec-of-Vec 需显式构造
let e = Vec::new();
e.push([1, 10]);
e.push([2, 20]);
let m = from_entries(e);
sum([1, 2, 3, 4]) == 10 && product([1, 2, 3, 4]) == 24 && m.len() == 2
"#;

/// collections.th 高阶函数（any/all/find/count_if/partition）——a1 缺口走解释器。
const M29_COLLECTIONS_COLLECTIONS_HOF: &str = r#"
use std::collections::collections::any
use std::collections::collections::all
use std::collections::collections::find
use std::collections::collections::count_if
use std::collections::collections::partition

any([1, 2, 3], |x| { x > 2 }) && !all([1, 2, 3], |x| { x > 2 }) &&
find([1, 2, 3], |x| { x == 2 }, -1) == 2 && find([1, 2, 3], |x| { x == 99 }, -1) == -1 &&
count_if([1, 2, 3, 4], |x| { x % 2 == 0 }) == 2 && partition([1, 2, 3, 4], |x| { x > 2 }).len() == 2
"#;

const M30_COLLECTIONS_ITER: &str = r#"
use std::collections::iter::range
use std::collections::iter::concat
use std::collections::iter::reverse
use std::collections::iter::take
use std::collections::iter::skip
use std::collections::iter::zip
use std::collections::iter::enumerate

range(1, 5).len() == 4 && reverse([1, 2, 3]).get(0) == 3 && concat([1, 2], [3]).len() == 3 &&
take([1,2,3,4], 2).len() == 2 && skip([1,2,3,4], 2).len() == 2 && zip([1,2], [10,20]).len() == 2 &&
enumerate([7,8]).len() == 2
"#;

/// iter.th 高阶函数（map/filter/reduce/any）——a1 缺口走解释器。
const M31_COLLECTIONS_ITER_HOF: &str = r#"
use std::collections::iter::map
use std::collections::iter::filter
use std::collections::iter::reduce
use std::collections::iter::any

map([1, 2, 3], |x| { x * 2 }).get(1) == 4 && filter([1, 2, 3, 4], |x| { x % 2 == 0 }).len() == 2 &&
reduce([1, 2, 3], 0, |a, b| { a + b }) == 6 && any([1, 2], |x| { x == 2 })
"#;

// ══════════════════════════════════════════════════════════════════
// string / json / toml / crypto / random / time / date / duration
// ══════════════════════════════════════════════════════════════════

const M32_STRING_STRING: &str = r#"
use std::string::string::*

join_lines(["a", "b"]) == "a\nb" && join_comma(["a", "b", "c"]) == "a, b, c" && repeat_sep("ab", "-", 3) == "ab-ab-ab" &&
is_blank("   ") && capitalize("hello") == "Hello" && count("abcabc", "a") == 2 && indent("x", 2) == "  x"
"#;

const M33_STRING_BUILDER: &str = r#"
use std::string::string_builder::*

let sb = append(append(new(), "Hello, "), "World!");
build(sb) == "Hello, World!" && len(sb) == 13 && !is_empty(sb)
"#;

const M34_JSON: &str = r#"
use std::json::json::*

let v = parse("{\"a\": 1, \"b\": [1, 2, 3]}");
let s = stringify(v);
let p = stringify_pretty(parse("{\"x\": true}"));
s.len() > 0 && p.len() > 0
"#;

const M35_TOML: &str = r#"
use std::toml::toml::*

let t = parse("name = \"tenth\"\n[sec]\nk = 1\n");
let s = stringify(t);
s.len() > 0 && t.len() > 0
"#;

const M36_CRYPTO_HASH: &str = r#"
use std::crypto::hash::*

sha256_hex([97, 98, 99]) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad" &&
md5_hex([97, 98, 99]) == "900150983cd24fb0d6963f7d28e17f72" && sha512_hex([97, 98, 99]).len() == 128
"#;

const M37_RANDOM: &str = r#"
use std::random::random::*

// L2.5：choice 已修为返回元素（空 Vec 哨兵 -1）；choice_index 返回索引
rand_int(5, 5) == 5 && rand_float() >= 0.0 && rand_float() < 1.0 && rand_range(10.0, 10.0) == 10.0 &&
choice([]) == -1 && choice([7]) == 7 &&
choice([10, 20, 30]) >= 10 && choice([10, 20, 30]) <= 30 &&
choice_index([]) == -1 && choice_index([7]) == 0 &&
choice_index([10, 20, 30]) >= 0 && choice_index([10, 20, 30]) <= 2 &&
sample([1, 2, 3], 2).len() == 2
"#;

const M38_TIME: &str = r#"
use std::time::time::*

now() > 0.0 && now_ms() > 0.0 && date().len() >= 8 && time_of_day().len() >= 5 && datetime().len() >= 15 &&
elapsed(start_timer()) >= 0.0
"#;

const M39_DATE: &str = r#"
use std::date::*

date_to_days(date_from_days(12345)) == 12345 && date_diff(date_new(2026, 1, 1), date_new(2026, 1, 1)) == 0 &&
date_weekday(date_new(2026, 8, 2)) >= 0 && date_weekday(date_new(2026, 8, 2)) <= 6
"#;

const M40_DURATION: &str = r#"
use std::duration::*

duration_as_secs(duration_from_millis(5000)) == 5 && duration_as_millis(duration_from_secs(2)) == 2000 &&
duration_as_nanos(duration_from_micros(1)) == 1000 &&
duration_as_millis(duration_add(duration_from_secs(1), duration_from_millis(500))) == 1500 &&
duration_as_secs(duration_mul_scalar(duration_from_secs(2), 3)) == 6 &&
duration_as_secs(duration_sub(duration_from_secs(5), duration_from_secs(2))) == 3
"#;

// ══════════════════════════════════════════════════════════════════
// fs / cli / process / env / io / http / net / regex
// ══════════════════════════════════════════════════════════════════

const M41_FS: &str = r#"
use std::fs::fs::*

let path = "smoke_tmp_fs.txt";
write_text(path, "hello smoke");
let r = read_text(path) == "hello smoke" && exists(path) && is_file(path) && size(path) == 11 && !is_dir(path);
remove(path);
r && !exists(path)
"#;

const M42_CLI: &str = r#"
use std::cli::cli::*

args_count() >= 1 && arg(0).len() > 0 && args().len() >= 1
"#;

const M43_PROCESS: &str = r#"
use std::process::*

let h = new("definitely_missing_cmd_xyz_12345");
match h {
    Result::Ok(hh) => match run(hh) {
        Result::Err(_) => true,
        Result::Ok(_) => false,
    },
    Result::Err(_) => true,
}
"#;

const M44_ENV: &str = r#"
use std::env::*

set("TENTH_SMOKE_VAR", "abc");
get("TENTH_SMOKE_VAR", "") == "abc" && get("TENTH_SMOKE_MISSING", "dflt") == "dflt" && get_or_empty("TENTH_SMOKE_MISSING") == ""
"#;

const M45_IO: &str = r#"
use std::io::*

eprint("SMOKE_IO_A");
eprintln("SMOKE_IO_B");
true
"#;

/// http：用 127.0.0.1:1（本机拒绝连接）验证 get 返回 Result 且不崩溃、不触网。
const M46_HTTP: &str = r#"
use std::http::*

match get("http://127.0.0.1:1/") {
    Result::Ok(s) => s.len() >= 0,
    Result::Err(_) => true,
}
"#;

/// net：连接本机 1 端口（拒绝）验证 connect 返回 Result::Err 且不触网。
const M47_NET: &str = r#"
use std::net::*

match connect("127.0.0.1", 1) {
    Result::Ok(_) => false,
    Result::Err(_) => true,
}
"#;

// 注：内容含 `"#` 序列，故用 r##"…"## 定界符。
const M48_REGEX: &str = r##"
use std::regex::*

match compile("\\d+") {
    Result::Ok(h) => match_(h, "abc123") && find(h, "abc123") == "123" &&
        replace(h, "a1b2", "#") == "a#b#" && split(h, "a1b22c").len() == 3 && find_all(h, "a1b22c").len() == 2,
    Result::Err(_) => false,
}
"##;

// ══════════════════════════════════════════════════════════════════
// runtime / curry / utils / data / autograd / logging
// ══════════════════════════════════════════════════════════════════

/// runtime：闭包值调用在 VM 失效（a1 类），走解释器路径。
const M49_RUNTIME: &str = r#"
use std::runtime::*

limit_or_default(1000000, |_| { 1 + 1 }, -1) == 2 && timeout_or_default(1000, |_| { 42 }, -1) == 42 &&
run_with_limit(1000000, |_| { 7 }, -1) == 7
"#;

/// curry：全部为闭包构造/调用，a1 缺口走解释器。
const M50_CURRY: &str = r#"
use std::curry::*

fn add(a, b) { a + b }
fn dbl(x) { x * 2 }
fn inc(x) { x + 1 }
let add5 = partial(add, 5);
let c = curry(add, 10);
let f = compose(inc, dbl);
add5(3) == 8 && f(3) == 7 && c(2) == 12
"#;

const M51_UTILS_MATH: &str = r#"
use std::utils::math::*

min(3, 5) == 3 && max(3, 5) == 5 && clamp(10, 0, 5) == 5 && abs(-4) == 4 &&
fmin(1.5, 2.5) == 1.5 && fmax(1.5, 2.5) == 2.5 && fclamp(3.0, 0.0, 1.0) == 1.0 &&
fabs(-2.5) == 2.5 && signum(-3.0) == -1.0
"#;

const M52_SERIALIZATION: &str = r#"
use std::utils::serialization::*

let w = tensor[[1.0, 2.0], [3.0, 4.0]];
let path = "smoke_tmp_ser.thw";
save_model(path, [w]);
let loaded = load_model(path);
let n = loaded.len() == 1;
remove_file(path);
n
"#;

const M53_DATALOADER: &str = r#"
use std::data::dataloader::*

let dl = new([10, 20, 30, 40, 50], 2);
has_next(dl) && next_batch(dl).len() == 2 && next_batch(advance(dl)).len() == 2 &&
next_batch(advance(advance(dl))).len() == 1 && !has_next(advance(advance(advance(dl))))
"#;

const M54_MNIST: &str = r#"
use std::data::mnist::*

read_i32_be([0, 0, 0, 5], 0) == 5 && read_i32_be([1, 2, 3, 4], 0) == 16909060 &&
one_hot(3).len() == 10 && one_hot(3).get(3) == 1.0 && normalize_pixel(255) == 1.0
"#;

/// autograd：call_custom_opN 需 Rust 端注册 op 才可执行，此处仅验证 use 可编译（import 检查）。
const M55_AUTAGRAD: &str = r#"
use std::autograd::call_custom_op1
use std::autograd::call_custom_op2
use std::autograd::call_custom_op3

true
"#;

const M56_LOGGING: &str = r#"
use std::logging::logging::*

let ok1 = format_message(0, "hi") == "[DEBUG] hi" && format_message(1, "hi") == "[INFO] hi" && format_message(3, "hi") == "[ERROR] hi";
set_level(3);
let ok2 = log_level == 3;
set_level(1);
ok1 && ok2 && log_level == 1
"#;

// ══════════════════════════════════════════════════════════════════
// 测试函数（每模块一个，便于定位回归）
// ══════════════════════════════════════════════════════════════════

#[test]
fn smoke_math_constants() { smoke_vm("math/constants", M01_MATH_CONSTANTS); }

#[test]
fn smoke_math_stats() { smoke_vm("math/stats", M02_MATH_STATS); }

#[test]
fn smoke_nn_activations() { smoke_vm("nn/activations", M03_NN_ACTIVATIONS); }

#[test]
fn smoke_nn_linear() { smoke_vm("nn/linear", M04_NN_LINEAR); }

#[test]
fn smoke_nn_loss() { smoke_vm("nn/loss", M05_NN_LOSS); }

#[test]
fn smoke_nn_feedforward() { smoke_vm("nn/feedforward", M06_NN_FEEDFORWARD); }

#[test]
fn smoke_nn_layer_norm() { smoke_vm("nn/layer_norm", M07_NN_LAYER_NORM); }

#[test]
fn smoke_nn_multihead_attention() { smoke_vm("nn/multihead_attention", M08_NN_MHA); }

#[test]
fn smoke_nn_pool() { smoke_vm("nn/pool", M09_NN_POOL); }

#[test]
fn smoke_nn_ops() { smoke_vm("nn/ops", M10_NN_OPS); }

#[test]
fn smoke_nn_batchnorm() { smoke_vm("nn/batchnorm", M11_NN_BATCHNORM); }

#[test]
fn smoke_nn_dropout() { smoke_vm("nn/dropout", M12_NN_DROPOUT); }

#[test]
fn smoke_nn_positional_encoding() { smoke_vm("nn/positional_encoding", M13_NN_POS_ENC); }

#[test]
fn smoke_nn_attention() { smoke_vm("nn/attention", M14_NN_ATTENTION); }

#[test]
fn smoke_nn_conv() { smoke_vm("nn/conv", M15_NN_CONV); }

#[test]
fn smoke_nn_embedding() { smoke_vm("nn/embedding", M16_NN_EMBEDDING); }

#[test]
fn smoke_nn_transformer() { smoke_vm("nn/transformer", M17_NN_TRANSFORMER); }

#[test]
fn smoke_init_initializers() { smoke_vm("init/initializers", M18_INIT); }

#[test]
fn smoke_optim_sgd() { smoke_vm("optim/sgd", M19_OPTIM_SGD); }

#[test]
fn smoke_optim_adam() { smoke_vm("optim/adam", M20_OPTIM_ADAM); }

#[test]
fn smoke_optim_adamw() { smoke_vm("optim/adamw", M21_OPTIM_ADAMW); }

#[test]
fn smoke_optim_rmsprop() { smoke_vm("optim/rmsprop", M22_OPTIM_RMSPROP); }

#[test]
fn smoke_optim_adagrad() { smoke_vm("optim/adagrad", M23_OPTIM_ADAGRAD); }

#[test]
fn smoke_optim_clip() { smoke_vm("optim/clip", M24_OPTIM_CLIP); }

#[test]
fn smoke_optim_lr_schedule() { smoke_vm("optim/lr_schedule", M25_OPTIM_LR_SCHEDULE); }

#[test]
fn smoke_optim_accumulate() { smoke_vm("optim/accumulate", M26_OPTIM_ACCUMULATE); }

#[test]
fn smoke_collections_hashset() { smoke_vm("collections/hashset", M27_COLLECTIONS_HASHSET); }

#[test]
fn smoke_collections_collections() { smoke_vm("collections/collections", M28_COLLECTIONS_COLLECTIONS); }

#[test]
fn smoke_collections_collections_hof() { smoke_interp("collections/collections 高阶(a1 解释器)", M29_COLLECTIONS_COLLECTIONS_HOF); }

#[test]
fn smoke_collections_iter() { smoke_vm("collections/iter", M30_COLLECTIONS_ITER); }

#[test]
fn smoke_collections_iter_hof() { smoke_interp("collections/iter 高阶(a1 解释器)", M31_COLLECTIONS_ITER_HOF); }

#[test]
fn smoke_string_string() { smoke_vm("string/string", M32_STRING_STRING); }

#[test]
fn smoke_string_builder() { smoke_vm("string/string_builder", M33_STRING_BUILDER); }

#[test]
fn smoke_json() { smoke_vm("json/json", M34_JSON); }

#[test]
fn smoke_toml() { smoke_vm("toml/toml", M35_TOML); }

#[test]
fn smoke_crypto_hash() { smoke_vm("crypto/hash", M36_CRYPTO_HASH); }

#[test]
fn smoke_random() { smoke_vm("random/random", M37_RANDOM); }

#[test]
fn smoke_time() { smoke_vm("time/time", M38_TIME); }

#[test]
fn smoke_date() { smoke_vm("date", M39_DATE); }

#[test]
fn smoke_duration() { smoke_vm("duration", M40_DURATION); }

#[test]
fn smoke_fs() { smoke_vm("fs/fs", M41_FS); }

#[test]
fn smoke_cli() { smoke_vm("cli/cli", M42_CLI); }

#[test]
fn smoke_process() { smoke_vm("process", M43_PROCESS); }

#[test]
fn smoke_env() { smoke_vm("env", M44_ENV); }

#[test]
fn smoke_io() { smoke_stderr("io", M45_IO, "SMOKE_IO_ASMOKE_IO_B"); }

#[test]
fn smoke_http() { smoke_vm("http", M46_HTTP); }

#[test]
fn smoke_net() { smoke_vm("net", M47_NET); }

#[test]
fn smoke_regex() { smoke_vm("regex", M48_REGEX); }

#[test]
fn smoke_runtime() { smoke_interp("runtime(a1 闭包解释器)", M49_RUNTIME); }

#[test]
fn smoke_curry() { smoke_interp("curry(a1 解释器)", M50_CURRY); }

#[test]
fn smoke_utils_math() { smoke_vm("utils/math", M51_UTILS_MATH); }

#[test]
fn smoke_serialization() { smoke_vm("utils/serialization", M52_SERIALIZATION); }

#[test]
fn smoke_dataloader() { smoke_vm("data/dataloader", M53_DATALOADER); }

#[test]
fn smoke_mnist() { smoke_vm("data/mnist", M54_MNIST); }

#[test]
fn smoke_autograd() { smoke_interp("autograd(use 编译检查)", M55_AUTAGRAD); }

#[test]
fn smoke_logging() { smoke_vm("logging/logging", M56_LOGGING); }
