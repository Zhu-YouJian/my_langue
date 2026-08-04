# Tenth API 冻结清单与版本策略（M5.1）

> **版本**：v1.0.0（已发布）| **日期**：2026-08-04
> **定位**：Tenth 对外 API 的**权威冻结清单**与 **semver 版本承诺**。语言语法与语义的正式定义在 `docs/语言规范.md`（§1.4.1 规范定稿声明）；本清单只负责「**哪些 API 被冻结、冻结到什么程度、版本号如何承诺**」。
> **真理源关系**：语言语法 → `docs/语言规范.md`；用法与 API 详情 → `docs/语言参考手册.md`；native 权威符号清单 → `tenth/std/prelude.th`；能力状态 → `能力梳理/能力全梳理.md`。本清单**不重复搬运**这些内容，只做冻结状态与版本承诺的登记。

---

## 1. semver 版本策略（v1.0.0 起生效）

版本号格式 `MAJOR.MINOR.PATCH`，语义遵循 [SemVer 2.0.0](https://semver.org/)，Tenth 具体映射如下：

| 版本位 | 含义 | 触发条件（Tenth 语境） |
|--------|------|------------------------|
| **PATCH** | bugfix | 修复缺陷且**不改变**任何公开 API 的签名 / 行为 / 错误消息（行为修复到「文档承诺的语义」不视为破坏） |
| **MINOR** | 兼容新增 | 新增语法糖 / 新类型 / 新 std 函数 / 新 native / 新 CLI 子命令 / 新预置宏；既有 API **零改动** |
| **MAJOR** | 破坏性变更 | 移除或改变既有语法 / 语义 / 标准库符号 / native 注册名 / CLI 命令；改变错误模型或静默失败防护边界 |

规则：

1. **破坏性变更必须 bump major**，且在发布说明中显式列出「破坏性变更清单」；
2. **兼容性新增只 bump minor**；新增 API 必须同步 `语言规范.md` + `语言参考手册.md` + `能力全梳理.md`；
3. **0.x 阶段（历史，v0.5.0 及以前）**：曾允许 minor 位包含破坏性修订（SemVer 0.x 规则），破坏性变更均已在 MEMO 与手册/规范中显式登记；**v1.0.0 起该规则失效**，按第 1/2 条执行；
4. **护城河红线**：shape 检查 / 静默失败防护 / lossy 格 / 零除数检测 / 内存预估的**收紧**（拦截更多错误）视为行为改进（minor 内合法），**放宽**（放行更多）视为破坏性变更（须 major）；
5. 编译器/运行时内部结构（HIR、VM 指令集、JIT 内部、tenthc 内部模块）**不属公开 API**，可在 minor/patch 内自由演进，但**不得改变可观察的程序行为**（输出 / 错误 / 行号语义）。

## 2. 冻结范围与状态

### 2.1 语言核心语法与语义 —— ✅ 冻结（v1.0.0 承诺）

| 类别 | 冻结内容 | 权威出处 |
|------|----------|----------|
| 词法 | 字面量（整数/浮点/字符/字符串/原始/多行/字节/插值/模板/张量）、注释（行/块嵌套）、标识符规则、自定义运算符字符集（`@`/`$`/`~` 连续组合） | 规范 §2 |
| 关键字 | 35 个活跃关键字 + 3 个预留（`task`/`shard`/`node`）；**新增关键字 = major 变更**（删除或改义亦然） | 规范 §2.4 |
| 类型系统 | 标量 8 整型 + f16/f32/f64/bf16 + bool/char/str/Unit + BigInt/C64/C128/Decimal 词法存在；张量 `Tensor[dtype, dims]`；容器（Vec/HashMap/String/str/[T;n]/Tuple/Range）；名义类型（struct/Newtype/enum/union）；引用 `&T`/`&mut T`；函数类型；`dyn Trait`；智能指针（Box/Rc/Arc/Pin/Weak）；泛型与实例化规则；typestate 状态参数 | 规范 §3 |
| 控制流 | if/else-if/else、match（含守卫/穷尽性）、while/loop/for/do-while、break/continue（含标签/带值）、return、尾调用 TCO | 规范 §5.6–5.8 |
| 表达式 | 运算符与优先级表（§2.7）、短路、块表达式、闭包与捕获（含 true letrec）、运算符重载、声明式宏、自定义运算符、async/await/spawn、yield、try 块与 `?`、`lossy` | 规范 §2.7–2.8、§5 |
| 所有权 | let 绑定（含无 init 语义）、`move`、Copy 自动派生、借用规则（语句粒度近似）、Drop/RAII、顶层全局 let、已移动值检测 | 规范 §4 |
| 错误模型 | 错误分类（Lexer/Parse/Type/Runtime/ShapeMismatch/Relation/Timeout）、`Result` 优先、or_die/assume_ok、丢弃+误用 warning、lossy 污点格、编译期零除数、行号定位 | 规范 §6 |
| 护城河 | 编译期 shape 检查、静默失败防护、typestate、内存/算力预估、运行时 autodiff shape 校验、FormalExplain | 规范 §1.3、§3.3、§6.7 |

> 冻结含义：v1.0.0 起以上条目的**语法与可观察语义为稳定契约**；`docs/语言规范.md` §1.4.1 为对应声明。远期/未实现特性（HKT、GAT、秩多态等）不属冻结范围，见规范 §7。

### 2.2 标准库 —— ✅ 冻结公开符号（v1.0.0 承诺）

| 类别 | 冻结内容 | 权威出处 |
|------|----------|----------|
| prelude 内置 native | 输出 / 张量构造与比较 / autodiff / 集合与智能指针 / 文件 I-O / 时间日期 / 随机 / 数学 / CLI / JSON / 环境进程 / 正则 / 编码（B批）/ 哈希 / 断言 / 运行时限制 / or_die·assume_ok 等 **150+ 符号** | `tenth/std/prelude.th`（权威清单） |
| 标准库模块（63 个用户模块 + prelude） | 见下方清单；模块名 + 公开函数签名冻结 | `tenth/std/` |

**模块清单（63 个用户模块，2026-08-04 盘点）：**

| 领域 | 模块 |
|------|------|
| 核心 | `prelude`（内置索引）、`async`、`autograd`、`curry`、`date`、`duration`、`env`、`io`、`net`、`http`、`process`、`regex`、`runtime` |
| cli | `cli/cli` |
| collections | `collections/collections`、`collections/hashset`、`collections/iter` |
| crypto | `crypto/hash` |
| data | `data/csv`、`data/dataloader`、`data/mnist`、`data/sampler` |
| distributed | `distributed/distributed` |
| fs | `fs/fs` |
| init | `init/initializers` |
| json | `json/json` |
| logging | `logging/logging` |
| math | `math/constants`、`math/functions`、`math/stats` |
| nn | `nn/activations`、`nn/attention`、`nn/batchnorm`、`nn/conv`、`nn/dropout`、`nn/embedding`、`nn/feedforward`、`nn/layer_norm`、`nn/linear`、`nn/loss`、`nn/multihead_attention`、`nn/ops`、`nn/pool`、`nn/positional_encoding`、`nn/transformer` |
| optim | `optim/accumulate`、`optim/adagrad`、`optim/adam`、`optim/adamw`、`optim/clip`、`optim/lion`、`optim/lr_schedule`、`optim/nadam`、`optim/radam`、`optim/rmsprop`、`optim/sgd` |
| random | `random/random` |
| string | `string/string`、`string/string_builder` |
| time | `time/time` |
| toml | `toml/toml` |
| utils | `utils/math`、`utils/serialization` |

> 冻结含义：模块名、公开函数签名、prelude native 名在 v1.0.0 起冻结；**新增模块/函数走 minor**；移除/改名/改签名 = **major**。模块内私有符号（`_` 前缀或非 prelude 引用）不冻结。

### 2.3 编译器 CLI 与工具链 —— ✅ 冻结命令面（v1.0.0 承诺）

| 工具 | 冻结命令 |
|------|----------|
| `tenth` | `run <file>`（解释执行）、`build <file>`（.wasm）、`wasm <file>`（编译+wasmi 执行）、`--max-memory <n>`（内存上限）、无参进入 REPL |
| `tenth` REPL | `:q` / `:h` / `:vars` / `:clear` / `:mem` / `:print <var>` |
| `tenthc` | `tenthc/main.th`（自举编译器入口，`cargo run ... run tenthc/main.th`） |
| `tenthpm` | `init` / `build` / `test` / `run` / `add` / `remove`(rm) / `list`(ls) / `clean` / `publish`（含 `--registry <dir>`）/ `install`（git/path/`.tenthpkg`/`--registry <dir>`） |
| `tenth-debug`（M4.4） | 调试器 CLI：`--bp N`（预置断点）+ 交互 `b [line]`/`d <line>`/`n`/`p <var>`/`c`/`l`/`q`/`h` |
| `tenth-prof`（M4.4） | 剖析器 CLI：`--top N`（top-N 热点报告） |
| LSP（tenth/tools/lsp） | 13 项能力（文档同步/diagnostics/hover/completion/definition/documentSymbol/references/rename/signatureHelp/foldingRange/semanticTokens/formatting） |

> 冻结含义：命令名、参数、退出码约定在 v1.0.0 起冻结；新增子命令/参数走 minor；移除/改义 = major。

### 2.4 运行时 native 注册名 —— ✅ 冻结（v1.0.0 承诺）

- **权威清单**：`tenth/std/prelude.th` 注释索引（150+ 符号，2026-08-04 盘点）；
- 注册位置：`tenth/src/runtime/natives.rs`（VM）+ `tenth/src/runtime/interpreter/natives.rs`（解释器）**双侧对齐**；
- 冻结含义：native 名与语义（参数/返回/错误行为）在 v1.0.0 起冻结；新增走 minor；改名/删名 = major。

## 3. 冻结状态标注汇总

| API 面 | 状态 | 规模（2026-08-04） |
|--------|------|---------------------|
| 语言核心语法/语义 | ✅ 冻结（1.0 承诺） | 规范 7 章 + 附录 A/B，覆盖语言核心 ✅84 项 |
| prelude 内置 native | ✅ 冻结（1.0 承诺） | 150+ 符号（`prelude.th` 权威清单） |
| 标准库模块 | ✅ 冻结（1.0 承诺） | 63 用户模块 + prelude（71 个 .th 文件） |
| CLI（tenth/REPL/tenthc/tenthpm/tenth-debug/tenth-prof/LSP） | ✅ 冻结（1.0 承诺） | 见 §2.3 |
| 运行时 native 注册名 | ✅ 冻结（1.0 承诺） | prelude 清单 + 双侧注册 |
| 编译器内部（HIR/VM 指令/JIT/tenthc 内部） | 🔒 非公开（可自由演进，不改变可观察行为） | — |
| 远期特性（HKT/GAT/秩多态/GPU 落地等） | ❌ 未冻结（1.0 后按路线图推进） | 规范 §7 |

## 4. 变更流程（冻结后）

```
提议变更
  ├─ 破坏性（语法/语义/符号/命令移除或改义）
  │     → 总师评审 → major bump → 发布说明「破坏性变更清单」
  │     → 同步：语言规范 + 语言参考手册 + 能力全梳理 + MEMO
  └─ 兼容新增（新符号/模块/命令/语法糖）
        → minor bump → 同步：语言规范 + 语言参考手册 + 能力全梳理 + MEMO + prelude（如 native）
```

1. **任何 API 变更先查能力全梳理现状**（改代码前），改完同步 MEMO + 能力全梳理（AGENTS.md 铁律）；
2. 新增 native 必须 **VM + 解释器双侧注册** + prelude.th 索引 + 三处 HIR 白名单同步（编译器部纪律）；
3. 破坏性变更必须**先登记**（AUDIT 或提案），评审通过才实施；
4. 版本号由总师统一 bump（`tenth/Cargo.toml`），文档不自行改版本。

## 5. v1.0.0 门槛检查（M5.1 登记，M5.4 决策）

以下为 M1–M5.1 如实登记的 1.0 门槛遗留项。**M5.4（2026-08-04）完成逐项决策**：`✅ 满足` = 已关闭；`🔴 建议 1.0 前修` = 静默错值/崩溃红线项（已报总师，由总师决定是否加一轮修复）；`⚠️ 已知限制（1.0 后）` = 可 1.0 后处理，已在 `RELEASE_NOTES.md` 已知限制节如实披露并登记 1.0.1+ 排期。

| # | 事项 | M5.4 判定 | 依据（证据） |
|---|------|----------|-------------|
| 1 | 手册/规范版本统一对齐 v1.0.0 | ✅ 满足 | M5.4 全量 bump（5 crate + 8 文档） |
| 2 | `db_query().len()` 误用拦截（M3.4） | ✅ 满足 | `silent_failure_test` 44 项守护 |
| 3 | tenthpm/LSP 完整实现（M4.1/M4.2） | ✅ 满足 | tenthpm 112 项 / LSP 63 项测试 |
| 4 | AUDIT-11.4.34（VM match tuple 模式 + guard 回退错乱） | 🔴 建议 1.0 前修 | **静默错值红线**：guard 失败后不试下一条 tuple 臂直接落 wildcard（VM 返回 `_ =>` 分支而解释器正确）；② JIT panic 逃逸已根治。触发面：tuple 多臂 + guard 组合（AUDIT.md §11.4.34） |
| 5 | AUDIT-11.1（借用检查语句粒度 B6 unsoundness） | ⚠️ 已知限制（1.0 后） | B6 原始反例已堵（borrow_holders）；剩余 B7-1/2/3 为**语义健全性缺口非内存安全**（Tenth 无 unsafe/FFI/并发，B6' 条件健全性已论证）；根治需 NLL（大工程）→ 1.0.1+ 排期（AUDIT.md §11.1） |
| 6 | 远程 registry（tenthpm 中央仓库） | ⚠️ 已知限制（1.0 后） | 纯功能缺口；本地 registry / git / `.tenthpkg` 发布安装闭环可用（M4.1） |
| 7 | AUDIT-11.4.39（重载运行时 VM/解释器分派不一致） | 🔴 建议 1.0 前修 | **静默错值红线**：VM HashMap 后注册覆盖 vs 解释器取第一条同名 → 同一重载调用两路径可能选中不同签名（`g(1,2)` 在 VM 返回 `"ONE_PARAM"` 错误值）；M3.2 编译期检查已拦截「类型确定不兼容」调用，但「类型正确、参数数量/签名不同」重载仍可能选错（AUDIT.md §11.4.39） |
| 8 | 错误消息文本本身不冻结（属可改进项），但**错误类别/行号语义**冻结 | ✅ 满足 | 约定 |
| 9 | AUDIT-11.4.43（JIT Union 字段修改 Cranelift 低化 panic） | ⚠️ 已知限制（1.0 后） | 功能正确（catch_unwind fallback 兜底，exit=0 + 输出正确）；仅 stderr panic 噪音（脚本解析 stderr 会误判失败）→ 1.0.1+ 修复后移除 `KNOWN_JIT_PANIC_STDERR` 分类（AUDIT.md §11.4.43） |

**M5.4 决策**：

- **建议 1.0 前修（2 项）**：AUDIT-11.4.39（重载分派）+ AUDIT-11.4.34（VM match guard）——均为**静默错值红线**（护城河红线），且触发面覆盖语言常用特性（重载 / tuple match + guard）。**已报总师**，由总师决定是否加一轮修复；若不加，两项已在本节与 `RELEASE_NOTES.md` 已知限制中如实披露绕行方式。
- **可 1.0 后（3 项）**：AUDIT-11.1（B6，语义健全性缺口非内存安全）、远程 registry（纯功能缺口）、AUDIT-11.4.43（JIT Union panic，功能正确仅噪音）——登记 1.0.1+ 排期。

**绕行方式（1.0 发布有效期内，供用户参考）**：

- **重载**：避免同一函数名混用不同参数数量的重载签名（编译期 `resolve_fn_overload` 已拦截类型确定不兼容的调用）；跨路径（VM/解释器）行为差异期间以默认 JIT/VM 路径为准，关键重载场景加单测断言。
- **match tuple + guard**：tuple 多臂避免与 guard 混用——用「单 guard 臂 + wildcard」或 if 链 / let 解构替代。
- **JIT Union 字段修改**：功能可用（fallback 保证正确输出），可忽略 stderr panic 噪音；或改用 `let mut` + 重建 Union 值。

---

> 本文档随 M 系列演进持续更新；状态变更同步 MEMO + 能力全梳理。
