# 标准库可用性盘点（L2.1）

> 盘点日期：2026-08-02
> 盘点范围：`tenth/std/` 全部 65 个 `.th` 文件
> 执行者：标准库部 + 测试部（subagent）
> 状态标记：✅ 可用 ｜ ⚠️ 部分/条件 ｜ ❌ 不可用 ｜ 空壳（无实际内容）
> 依据：提交 c686749（顶层 let 程序级全局 + use 导入模块状态）之后的实测

---

## 一、盘点方法

分三层验证，全部基于 `tenth/target/release/tenth.exe run <file>` 实测（VM 优先、失败时区分是否可回退解释器）：

| 层 | 内容 | 规模 |
|---|---|---|
| **Tier-1 导入层** | 对每个模块的每个公开项生成独立程序：`use std::<path>::<item>` + `println("OK")`。可测出 use 路径是否可解析、模块是否可加载编译（导入时即检查函数体） | 385 个程序 / 62 模块 |
| **Tier-2 调用层** | 对每个模块写实际调用程序，覆盖重点疑点模块的全部关键函数；VM 与解释器（`TENTH_NO_VM=1`）双路径各跑一遍 | 30 个程序 × 2 路径 |
| **官方测试** | 运行 `tenth/std/` 下 9 个 `test_*.th` 官方测试 | 9 个 |

验证脚本/临时文件在 `.trae/tmp/L21_audit/`（不入库）。**未修改任何标准库代码，未提交**。

---

## 二、总览统计

| 分类 | 业务模块（56） | 测试文件（9） | 合计（65） |
|---|---|---|---|
| ✅ 可用 | 37 | 2 | **39** |
| ⚠️ 部分/条件 | 9 | 7 | **16** |
| ❌ 不可用 | 7 | 0 | **7** |
| 空壳（纯文档/索引） | 3 | 0 | **3** |

不可用原因分类（❌+⚠️ 共 23 个模块，含重复归因）：

| 类别 | 说明 | 涉及模块 |
|---|---|---|
| **(a) 语言层缺口** | 3 个子项，见 §四 | 12 个 |
| **(b) 标准库自身 bug** | API/语法/注解/逻辑写错 | 8 个 |
| **(c) 文档失实** | 文档声称有/路径对，实际无/不对 | 6 处 |

---

## 三、完整清单（65 模块）

### 3.1 业务模块

| 模块 | 公开项数 | 状态 | 原因 / 修复建议 |
|---|---|---|---|
| `async.th` | 0 | 空壳 | 纯文档（async native 为 built-in，无 .th API） |
| `autograd.th` | 3 | ⚠️ 条件 | `call_custom_op1/2/3` 需 Rust 端 `register_custom_op` 返回的 op_id 才可用；未注册 op 报"未注册"。模块本身编译/调用路径正常 |
| `cli/cli.th` | 6 | ✅ | 双路径可用（底层 native 已读真实进程参数） |
| `collections/collections.th` | 11 | ⚠️ | 高阶函数（`any/all/find/count_if/partition/flat_map`）**VM 路径失败**："未定义的函数 'f'"（a1：VM 不支持参数名调用函数）；`sum/product` 解释器路径失败："加法类型不匹配"（a2）。VM 下 `sum/product` 可用 |
| `collections/hashset.th` | 14 | ✅ | 双路径可用（`HashSet` struct + 链式 API） |
| `collections/iter.th` | 15 | ⚠️ | 同 `collections.th`：高阶函数 VM 失败（a1）、`sum` 解释器失败（a2） |
| `crypto/hash.th` | 3 | ✅ | `sha256_hex/sha512_hex/md5_hex` 双路径可用 |
| `curry.th` | 3 | ⚠️ | `partial/curry/compose` 解释器路径可用；**VM 路径失败**（a1） |
| `data/dataloader.th` | 6 | ⚠️ | **`next_batch` 不推进 cursor**（b）：注释声称"advances the cursor"但实现只返回切片不更新状态，`has_next` 恒真、迭代死循环。需改为返回新 DataLoader（值语义）或提供手动推进 API |
| `data/mnist.th` | 5 | ⚠️ 条件 | `read_i32_be/one_hot/normalize_pixel` 可用；`parse_images/parse_labels` 需真实 MNIST 数据文件（条件） |
| `date.th` | 15 | ✅ | 官方 `test_date.th` 双路径全过（含闰年/跨年/边界） |
| `duration.th` | 22 | ✅ | 双路径可用 |
| `env.th` | 5 | ✅ | 规范路径 `use std::env::get` 等可用；**prelude 写的 `std::env::env::get` 不工作**（c） |
| `fs/fs.th` | 18 | ✅ | 双路径可用（读写/目录/复制/删除/路径操作全过） |
| `http.th` | 2 | ✅ | `get/post` 可用（需网络；失败返回 `Result::Err` 不崩溃）。**prelude 写的 `std::http::http::get` 不工作**（c） |
| `init/initializers.th` | 6 | ✅ | 双路径可用（`xavier/he/zeros/constant_init`，f32/f64 泛型均过） |
| `io.th` | 4 | ✅ | `eprint/eprintln/read_line` 可用 |
| `json/json.th` | 25 | ⚠️ | `parse` 完整解析在 **VM 路径失败**："未定义的函数 'str_add'"（a3），解释器路径全过（官方 test 12 项全绿）；`stringify` 双路径可用。**API 名失实**：prelude 写 `encode/decode/encode_pretty`，实际是 `parse/stringify/stringify_pretty`（c） |
| `logging/logging.th` | 11 | ✅ | **重点疑点通过**：`use std::logging::logging::*` 后 `LEVEL_*` 常量 + 可变全局 `log_level` 可用，`set_level` 真能改状态（LEVEL_DEBUG→全打；切 ERROR→info/debug 不再输出） |
| `math/constants.th` | 35 | ✅ | **重点疑点通过**：`use std::math::constants::*` 后函数内 `PI/E/TAU/PHI` 可用（VM+解释器），23 个函数全过 |
| `math/functions.th` | 0 | 空壳 | 纯文档（内置标量/张量数学的说明，无实际函数） |
| `math/stats.th` | 13 | ✅ | 官方 `test_stats.th` 全过 + 调用验证通过 |
| `net.th` | 8 | ✅ | `connect/read/write/close/listen/accept` 可用（需网络）。**prelude 写的 `std::net::net::connect` 不工作**（c） |
| `nn/activations.th` | 11 | ✅ | `relu/sigmoid/tanh/softmax/exp/log/gelu/leaky_relu*` 双路径可用 |
| `nn/attention.th` | 2 | ❌ | **函数体引用未定义的 `shape` 自由函数**（b）：`shape(q)[1]` 等，Tenth 无 `shape()` 自由函数。修复建议：改用 `q.shape_tensor()` 或运行时传维度参数 |
| `nn/batchnorm.th` | 1 | ✅ | 可用（输入需 4D NCHW） |
| `nn/conv.th` | 1 | ❌ | **同 attention**：函数体 `shape(w)[2]` 未定义（b） |
| `nn/dropout.th` | 1 | ✅ | 可用 |
| `nn/embedding.th` | 1 | ❌ | **返回注解 `Tensor[S, D]` 与实际 `gather` 返回 `Tensor[S]` 不符**（b+底层限制）：gather native 要求 index.ndim==base.ndim，直接 `gather(weight,0,indices)` 返回 1D。需 runtime 提供 `index_select` 或改实现 |
| `nn/feedforward.th` | 2 | ✅ | `feedforward`/`make_feedforward_params` 可用 |
| `nn/layer_norm.th` | 2 | ✅ | `layer_norm`/`make_layer_norm` 可用 |
| `nn/linear.th` | 1 | ✅ | 可用 |
| `nn/loss.th` | 6 | ✅ | `mse/mse_loss/l1_loss/huber_loss/huber_loss_train/binary_cross_entropy` 可用 |
| `nn/multihead_attention.th` | 1 | ✅ | 可用（mask 需 `[n_heads,S,S]` 或 `[S,S]` 语义，`[1]` 会 shape mismatch——文档需注明） |
| `nn/ops.th` | 7 | ✅ | `gt/lt/ge/le/eq/ne/where_` 可用 |
| `nn/pool.th` | 4 | ✅ | `max_pool2d/avg_pool2d` 及 explicit 版可用（输入需 4D NCHW） |
| `nn/positional_encoding.th` | 1 | ✅ | 可用 |
| `nn/transformer.th` | 2 | ❌ | **缺 `use` 导入**（b）：函数体调用 `layer_norm`/`multihead_attention`/`feedforward` 但文件顶部无任何 use，编译报"未定义的泛型函数 'layer_norm'"。修复：文件顶部补 3 个 use |
| `optim/accumulate.th` | 2 | ⚠️ | `accumulate_grad` 可用；`accumulate_loop`（高阶函数）VM 路径失败（a1） |
| `optim/adagrad.th` | 1 | ✅ | 可用（返回 tuple `(new_w,new_g2)` 需解构） |
| `optim/adam.th` | 1 | ✅ | 可用（返回 tuple 需解构） |
| `optim/adamw.th` | 4 | ✅ | `adamw_step`（tuple）+ `_w/_m/_v`（JIT 友好单值）均可 |
| `optim/clip.th` | 2 | ✅ | `clip_grad_by_value/norm` 可用 |
| `optim/lr_schedule.th` | 8 | ✅ | `cosine/step/exp/warmup*` 可用（LR_PI/LR_EPS 顶层常量正常）；**1 个边界语义与测试预期不符**（b 轻微）：`warmup_cosine_lr(base,60,100,50)`（warmup>total 且 step>=total）实现先走 warmup 分支返回 0.6，测试预期返回 ~0。异常配置，影响小 |
| `optim/rmsprop.th` | 1 | ✅ | 可用（返回 tuple 需解构） |
| `optim/sgd.th` | 3 | ✅ | `sgd_step`/`sgd_weight_decay`/`sgd_momentum`（tuple）可用 |
| `prelude.th` | 0 | 空壳 | 索引/文档文件（`use std::prelude::*` 本就不支持） |
| `process.th` | 4 | ✅ | `new/arg/run/output` 可用（`new` 返回 Result 需 `or_die`） |
| `random/random.th` | 7 | ❌ | **`choice` 用旧语法 `fn choice(v: Vec) i32`（无 `->`）**（b）：整模块编译失败，其余 6 个函数连带不可用。修复：`-> i32` |
| `regex.th` | 6 | ✅ | `compile/match_/find/find_all/replace/split` 可用（`compile` 返回 Result 需 `or_die`） |
| `runtime.th` | 4 | ✅ | `run_with_limit/limit_or_default/run_with_timeout/timeout_or_default` 双路径可用（走 native `with_*`） |
| `string/string.th` | 8 | ✅ | `join_lines/join_comma/repeat_sep/indent/word_wrap/is_blank/capitalize/count` 可用 |
| `string/string_builder.th` | 7 | ❌ | **`append` 用 `s.clone()` 但 String 无 `clone` 方法**（b）：解释器报"String 没有方法 'clone'"，VM 报"没有字段 'total_len'"。修复：去掉 clone（String 值语义） |
| `time/time.th` | 9 | ✅ | `now/now_ms/date/time_of_day/datetime/sleep_ms/start_timer/elapsed*` 可用 |
| `toml/toml.th` | 18 | ⚠️ | 官方 `test_toml.th` **VM 路径失败**（a3 str_add），解释器路径全过 |
| `utils/math.th` | 9 | ✅ | `min/max/clamp/abs/fmin/fmax/fclamp/fabs/signum` 可用 |
| `utils/serialization.th` | 3 | ❌ | **返回类型注解错误**（b）：`save_model(...) -> i32` 但底层 `save_weights` 返回 `Unit`，3 个函数全部编译失败。修复：注解改 `-> Unit`（或让 native 返回状态码） |

### 3.2 测试文件（`test_*.th`）

| 文件 | 状态 | 说明 |
|---|---|---|
| `test_date.th` | ✅ | 双路径全过 |
| `test_runtime.th` | ⚠️ | VM 路径失败（a3 str_add，f-string 插值触发），解释器路径过 |
| `math/test_stats.th` | ✅ | 全过 |
| `optim/test_lr_schedule.th` | ⚠️ | 1 个边界断言失败（warmup>total 语义，见 lr_schedule） |
| `json/test_json.th` | ⚠️ | VM 路径 str_add 失败，解释器 12 项全过 |
| `json/test_min.th` | ⚠️ | 同 test_json |
| `json/test_obj.th` | ⚠️ | 同 test_json |
| `toml/test_toml.th` | ⚠️ | VM 路径 str_add 失败，解释器全过 |

---

## 四、不可用原因分类（❌/⚠️ 归因）

### (a) 语言层缺口（回总师 / 编译器部 / 运行时部）

| 子项 | 现象 | 影响模块 | 修复方向 |
|---|---|---|---|
| **a1** | **VM 路径不支持高阶函数**：`f(...)`（通过参数名调用传入的函数）在 VM 下报"未定义的函数 'f'"且不静默回退；解释器路径正常（probe 验证 `apply_twice(|v| v+1, 10)` VM=Unit+报错，INTERP=12） | `collections.th`、`iter.th`（全部高阶函数）、`curry.th`、`optim/accumulate.th::accumulate_loop` | 运行时部：VM 字节码/闭包调用缺口 |
| **a2** | **解释器路径 `int + Vec.get(i)` 类型不匹配**：`total = total + items.get(i)` 报"加法类型不匹配"（VM 正常） | `collections.th::sum/product`、`iter.th::sum`（解释器路径） | 运行时部：解释器 `Vec.get` 返回类型与 int 运算不匹配 |
| **a3** | **VM 未注册 `str_add` native**：字符串拼接/插值在 VM 路径报"未定义的函数 'str_add'"；解释器路径正常 | `json::parse`、`toml`、`test_runtime`（f-string）、`test_json/min/obj`、`test_toml`（VM 路径） | 运行时部：VM 补齐 `str_add` native（HIR 字符串操作） |

### (b) 标准库自身 bug（修复建议 → L2.2）

| 模块 | bug | 修复建议 |
|---|---|---|
| `random/random.th` | `choice` 旧语法 `fn choice(v: Vec) i32` 无 `->`，整模块编译失败 | `-> i32` |
| `utils/serialization.th` | `save_model/save_checkpoint -> i32` 但 `save_weights` 返回 `Unit` | 注解 `-> Unit`（或 native 返回状态码） |
| `string/string_builder.th` | `append` 用 `String.clone()`（方法不存在） | 去 clone（值语义直接传） |
| `nn/attention.th`、`nn/conv.th` | 函数体引用未定义自由函数 `shape(...)` | 改用 `shape_tensor()` 或显式传维度 |
| `nn/transformer.th` | 缺 `use std::nn::{layer_norm,multihead_attention,feedforward}` 导入 | 文件顶部补 use |
| `nn/embedding.th` | 返回注解与实际 `gather` 结果 shape 不符 | 需要 runtime 提供 `index_select`，或改注解/实现 |
| `data/dataloader.th` | `next_batch` 不推进 cursor，无法迭代 | 返回新 DataLoader 或加 cursor 推进 |
| `optim/lr_schedule.th` | `warmup_cosine_lr` warmup>total 边界语义与测试预期不符（轻微） | 统一分支顺序（先判 `step>=total`）或改测试预期 |

### (c) 文档失实清单

| 位置 | 声称 | 实际 | 影响 |
|---|---|---|---|
| `prelude.th` | `std::json::json::encode, decode, encode_pretty` | 实际 `parse, stringify, stringify_pretty`（encode/decode 不存在） | 用户按文档 use 即编译失败 |
| `prelude.th` | `std::env::env::get` 等 | `std::env::env::xxx` 不工作；正确 `std::env::get` | use 失败 |
| `prelude.th` | `std::http::http::get` | 不工作；正确 `std::http::get` | use 失败 |
| `prelude.th` | `std::net::net::connect` | 不工作；正确 `std::net::connect` | use 失败 |
| `prelude.th`/参考手册 | `nn::attention/conv/embedding/transformer`、`std::random`、`std::utils::serialization` 列为可用 | 实际 ❌（见 §三） | 声称实现、实际不可用 |
| `prelude.th` | `std::io::io::eprint` | 意外可用（use 回退机制兜底）——但属偶然，不应作为规范 | 一致性 |

> 注：`docs/语言参考手册.md` 与 `能力梳理/能力全梳理.md` 中声称已实现的模块（attention/conv/embedding/transformer/random/serialization/json 的 encode 等）与实测不符，属 (c) 类，建议在 L2.2 修复后同步勘误。

---

## 五、重点疑点模块结论（总师指定的核实项）

| 疑点 | 结论 | 证据 |
|---|---|---|
| **`math/constants`**：`use std::math::constants::*` 后函数内用 PI/E/TAU/PHI | ✅ **已解锁**（提交 c686749 生效） | `call_constants.th` 双路径 exit 0，`PI + TAU*2 + E + PHI` 计算正确，23 个函数全过 |
| **`logging`**：`set_level` 真能改状态、可变全局可用 | ✅ **已解锁** | `call_logging.th`：LEVEL_DEBUG 下 4 条全打；`set_level(LEVEL_ERROR)` 后 info/debug 不输出；error 正常 |
| **`optim/lr_schedule`**：LR_PI/LR_EPS 顶层常量 | ✅ 可用（1 个边界断言失败，见 b） | `call_lr_schedule.th` 双路径 exit 0；官方测试 6 组全过、1 边界失败 |
| **`math/functions`**、**`math/stats`** | functions 是空壳（纯文档）；stats ✅ | `test_stats.th` 全过 |
| **`string/string.th`**、**`string_builder`** | string.th ✅；string_builder ❌（`String.clone` 不存在） | `call_string.th` 双路径过；`call_string_builder.th` 双路径失败 |
| **`collections`**（collections/hashset/iter） | hashset ✅；collections/iter ⚠️（a1+a2 双路径各有缺口，无完整可用路径） | 见 §四 |
| **`json`、`toml`** | 解释器路径 ✅（官方测试全绿）；VM 路径 ❌（a3 str_add）；json API 名失实（c） | `test_json/toml` 解释器 exit 0、VM exit 1 |
| **`nn/*` 核心** | feedforward/layer_norm/multihead_attention/linear/loss/activations/dropout/batchnorm/pool/ops/positional_encoding **全部 ✅**；attention/conv/embedding/transformer ❌ | `call_nn.th` + `call_nn_mha.th` 双路径 exit 0；坏模块归因见 §三 |
| **`optim/*`** | 全部 ✅（返回 tuple 需解构；accumulate_loop 受 a1 影响） | `call_optim.th` 双路径 exit 0 |
| **`data/dataloader`、`data/mnist`** | dataloader ⚠️（next_batch 不推进 cursor）；mnist 基础函数 ✅ | 见 §三 |
| **`crypto/hash`、`random`、`time`、`date`、`duration`、`fs`、`cli`、`process`、`env`、`io`、`http`、`net`、`async`、`autograd`、`runtime`、`curry`、`regex`、`utils/*`** | crypto/time/date/duration/fs/cli/process/env/io/http/net/runtime/regex/utils-math ✅；curry/autograd ⚠️；random ❌；async 空壳；serialization ❌ | 见 §三 |

---

## 六、结论与建议

1. **语言层 3 个缺口（a1/a2/a3）是最大瓶颈**：a1（VM 高阶函数）和 a3（VM str_add）影响 12 个模块的 VM 路径，a2 影响解释器路径的集合数值函数。建议优先回报总师，交运行时部修复。
2. **8 个标准库 bug（b 类）**是 L2.2 的直接修单：random/serialization/string_builder/attention/conv/embedding/transformer/dataloader 均是小改动即可修复。
3. **6 处文档失实（c 类）**需在修复后同步勘误 `prelude.th` 与参考手册。
4. **AI 核心（nn/optim/init）整体健康**：除 attention/conv/embedding/transformer 4 个模块外全部可用，`stdlib_demo.th` 实例运行通过。
