# tenth-lens 语言缺口登记册

> **只追加，不回改**；每条缺口由"写这个工具时真实撞上"产生（见 `README.md` §四 工作规则）。
> 格式：编号 / 缺什么 / 影响谁 / 证据 / 状态 / 关联 `AUDIT`。
> 状态：`阻塞`（写不下去）/ `绕行`（能用更笨的写法替代，但记录着）/ `已验证可用`（怕误记，留证）。

---

## GAP-001 · 无法从 CLI 询问编译器内部状态（JIT 是否生效 / 是否回退）

- **缺什么**：一个可查询的**执行记录**——"这个函数被 JIT 编译了吗？回退了吗？为什么？"
- **影响谁**：**P2**（观察层）；以及**任何人手工排查性能/一致性**
- **证据**：2026-09-16 总师做 Python/C 性能对照时，想知道 montecarlo 那段循环走 JIT 还是回退 VM——**从 CLI 问不到**，只能用计时差反推，且**第一次推错**（先判"走 VM"，后以 `TENTH_NO_VM=1` 对照才发现 JIT 生效、比解释器快 23×）。
  内部其实**已有这个自省面**（`jit_consistency_test` 能问 `is_compiled` / `is_failed`），但**只长在测试里**，不是给人看的视图。
- **状态**：`阻塞`（P2 交付前无法做该层）
- **关联**：P2 交付物；`jit_consistency_test`（既有内部自省面）

## GAP-002 · 整型算术被硬限在 i32

- **缺什么**：可用的 i64 整型算术
- **影响谁**：任何需要 >2 147 483 647 的计数 / 偏移 / 哈希 / 时间戳的 Tenth 程序（**本工具若做字节级统计或哈希即会撞上**）
- **证据**：`AUDIT-11.4.53`（2026-09-16 登记，8 个探针定位）：显式 `i64` 后缀、`let x: i64` 标注、`i64` 形参与返回类型**全部无效**；`2000000000i64 * 100i64` 报「溢出 i32 范围」
- **状态**：`阻塞`（已知，待排期；本工具将**主动绕开**——不做大整数运算，并在撞上时登记）
- **关联**：`AUDIT-11.4.53`、`AUDIT-11.4.45` A（同根因）

## GAP-003 · 无各阶段转储（tokens / AST / HIR / 字节码 / JIT IR）

- **缺什么**：`--dump-*` 只读开关
- **影响谁**：诊断类功能全部；日常排查
- **证据**：`tenth/src` 全库 `dump|disasm|--emit|print_bytecode|print_ast` **0 命中**；CLI 仅 `--max-memory`
- **状态**：`阻塞`（P2）；其中 **JIT IR 转储**预计难度最高，优先预留 stub
- **关联**：P2

## GAP-004 · 无 verifier / IR 合法性自检

- **缺什么**：编译器产出物（HIR / 字节码）的自洽校验
- **影响谁**：观察层与守卫层的上游；畸形 IR 只能靠下游症状发现
- **证据**：`tenth/src` 内无 `fn verify*`；`debug_assert` 全仓仅 5 处（4 处在 JIT translator）
- **状态**：`阻塞`（P3 候选）
- **关联**：P3

## GAP-005 · 子进程捕获能力（P0 将第一个撞上）

- **缺什么 / 待验证**：能否 spawn `tenth.exe`、**捕获 stdout/stderr**、拿到**退出码**（差分检查器的地基）
- **影响谁**：**P0**
- **证据**：`tenth/std/process` 存在且 `process_test` 5 项通过（既有测试），但**"捕获输出 + 退出码"这一组合在 .th 侧是否可用，尚未实测**
- **状态**：**待验证**（P0 第一步）
- **关联**：P0；`tenth/std/process`

---

## GAP-005 实测结论（2026-09-17，总师委派的 P0 前置门）

- **能**：① spawn 子进程（`command_new`/`command_arg`）✅；② 捕获 **stdout**（`command_output`）✅；③ 取 **退出码**（`command_run`：`exit(3)` → `code=3`，正常退出 → `code=0`）✅；④ 子进程**继承父进程环境变量**（`env_set` 后 spawn → 子进程读到 `CHILD_ENV=yes`）✅
- **不能**：① 捕获 **stderr**（→ GAP-006）；② 子进程**超时/取消**（→ GAP-007）；③ per-child env / `env_unset`（→ GAP-008）
- **证据**：`tenth-lens/probes/gap005_probe.th`（默认路径与 `TENTH_NO_VM=1` 双侧均跑通；1–8 项双侧结论一致）。解释器路径实跑节选：
  ```
  GAP005|stdout_capture|OK|PROBE_OUT_A\n
  GAP005|stderr_capture|MISSING|返回值只含 stdout=[PROBE_OUT_S\n]，stderr 不可得
  GAP005|exit_code_nonzero|OK|code=3
  GAP005|exit_code_zero|OK|code=0
  GAP005|env_inherit|OK|CHILD_ENV=yes\n
  GAP005|env_per_child|MISSING|NEED: 无 command_env（只能全局 env_set + 继承）
  GAP005|env_unset|MISSING|NEED: 无 env_unset（且 TENTH_NO_VM=空串 仍算已设置）
  GAP005|child_timeout|MISSING|NEED: 无子进程超时/取消能力
  GAP005|diagnostics_via_output|PARTIAL|stdout=[]（错误消息走 stderr，此处不可见）
  GAP005|both_axes_same_spawn|PARTIAL|code=3 stdout=[PROBE_OUT_B\n]，但 run 阶段输出已泄漏到本进程
  ```
  两侧第 9 项取值不同（默认路径 `stdout=[()\n]` / 解释器 `stdout=[]`）——这本身就是 GAP-009 的分歧痕迹；第 10 项两侧同结论。
- **结论**：**GAP-005 由「待验证」结案为「部分可用」**；**P0 门未通过**——P0 要求比对「stdout + stderr + 退出码」三轴，**stderr 轴不可得（GAP-006）**，且**三轴无法取自同一次 spawn（GAP-010）**、**无每用例超时（GAP-007）**、**无法保证子进程走默认路径（GAP-008）**。
- **未采用的绕行**（记录备查，**按章程 §四 未实施**）：可用 `cmd /c "... 1>out.txt 2>err.txt"` + `fs` 读文件迂回拿到两路输出。它会引入临时文件、`cmd` 依赖、改变子进程缓冲行为，等于把「测量工具自身」掺进被测信号，故**不作为 P0 的地基**；是否接受由总师决策。

## GAP-006 · 无法捕获子进程 stderr

- **缺什么**：取子进程 stderr 的能力。`command_output` 只返回 stdout（Rust 侧 `.output()` 已捕获 stderr 后**直接丢弃**）；`command_run` 走 `status()`，stdout/stderr **继承**父进程，更拿不到。
- **影响谁**：**P0**（比对轴之一直接缺失）；以及任何需要看编译器诊断的 Tenth 程序——`tenth run` 的**诊断全部走 stderr**（`main.rs` 13 处 `eprintln!`，含 VM 回退提示 `[info] VM 不支持此程序结构…`，与 GAP-001 直接相关）。
- **证据**：探针项 2 输出 `stderr_capture|MISSING|返回值只含 stdout=[PROBE_OUT_S\n]`——子进程的 `PROBE_ERR_S` 既不在返回值里，也没漏到父进程 stderr（被丢弃）。源码双侧一致：`tenth/src/runtime/natives.rs:677-699` 与 `tenth/src/runtime/interpreter/natives.rs:652-673` 都只取 `output.stdout`，从未读 `output.stderr`。探针项 9 更直观：越界子进程的报错文本在 `command_output` 下**完全不可见**（`stdout=[]`）。
- **状态**：`阻塞`（要在 native 侧补「stderr 版 output」原语＝Rust 改动，与 P0「零 Rust 改动」直接冲突 → 需总师取舍）
- **关联**：P0；GAP-001（VM 回退提示长在 stderr 上）；`tenth/src/runtime/natives.rs:677`

## GAP-007 · 无子进程超时 / 取消能力

- **缺什么**：给子进程设超时，或至少能杀掉/非阻塞查询。`command_*` 无 timeout 参数，句柄只是不透明 `i64`，无 `command_kill`/`command_try_wait`；`command_run`/`command_output` 均阻塞到底。
- **影响谁**：**P0**（章程要求「每个用例必须有超时（例如 60s），超时算不一致/失败，不静默跳过」）；一个卡死的语料会卡死检查器自身。
- **证据**：探针项 8；`tenth/src/runtime/natives.rs:630-699` 四个 native 签名无超时参数；全仓无 `command_kill|command_timeout|command_try_wait`。
- **状态**：`阻塞`（P0 只能退化成「父进程整体超时」，无法按用例计时与定位）
- **关联**：P0

## GAP-008 · 无 per-child 环境变量，且无 `env_unset`（会让 P0 静默跑成「双解释器」）

- **缺什么**：① per-child env（`command_env`）；② 删除环境变量（`env_unset`/`env_remove`）。
- **影响谁**：**P0 的致命项**。差分检查器靠 `TENTH_NO_VM` 区分两条路径，而 `tenth/src/main.rs:194` 判的是 `std::env::var("TENTH_NO_VM").is_ok()`——**任何值都算「已设置」**（实测 `TENTH_NO_VM=0`、`TENTH_NO_VM=空格` 均走解释器）。Tenth 侧**无法 unset**，因此只要检查器进程自己带 `TENTH_NO_VM`，它 spawn 出的「默认路径」其实也是解释器 → 报告「全一致」，实为**自比自的静默假阳性**（正是本项目要压制的失败模式）。
- **证据**：探针项 6/7；`main.rs:194`；实测同一子进程 `gap005_child_bad.th`：不设变量 → VM 语义（stdout `()`，exit 0）；`TENTH_NO_VM=0` / `=空格` → 解释器语义（stderr 报错，exit 1）。
- **状态**：`阻塞`（P0 的可行前提是「父进程 env 干净」，属外部前提；工具内必须**显式校验**——例如让子进程回显 `env_get("TENTH_NO_VM")`——并把「两条路径实为同一后端」判为**硬失败**，绝不允许静默通过）
- **关联**：P0；`tenth/src/main.rs:194`

## GAP-009 · 【跨路径分歧·实测首例】Vec 越界索引：VM 静默返回 Unit，解释器报错

- **缺什么**：VM 路径 `Vec` 索引（`IndexGet` opcode）的**边界检查**。
- **影响谁**：任何用 `v[i]` 且可能越界的 Tenth 程序——**默认路径静默给出 Unit**（打印 `()`）且 **exit 0**；解释器给运行时错误且 exit 1。「跨后端语义不一致 + 静默失败」双料。
- **证据**：`tenth-lens/probes/gap005_child_bad.th`（`let x = v[7]; println(x);`，`v` 为空 Vec）
  - 默认路径：stdout `()`，exit **0**
  - `TENTH_NO_VM=1`：stderr `Error: 第 4 行第 14 列：运行时错误 — Vec 索引 7 越界`，exit **1**
  - 源码定位：`tenth/src/runtime/vm/execute.rs:950` `Value::Vec(items) => { let v = items.borrow().get(i).cloned().unwrap_or(Value::Unit); }`——越界静默变 Unit；对照 `runtime/vm/natives.rs:152`/`:176` 与解释器均报「Vec 索引 N 越界」，同文件 Tensor 分支（`execute.rs:970-975`）也正确报错，**唯 Vec 分支漏检**。
- **状态**：**待转入 `AUDIT.md`**（本会话红线禁改 `docs/`、`AUDIT.md`，故登记于此，编号请总师定）；**未修、未绕行**（P0 是零 Rust 改动阶段）
- **关联**：`AUDIT-11.4.40`（`Vec.get()/pop()` 类型误标，同族「越界语义不统一」）；P0 的首个差分发现

## GAP-010 · 同一次 spawn 拿不到「stdout + 退出码」（三轴必须同源才可信）

- **缺什么**：一次 spawn 同时返回 stdout / stderr / 退出码的原语。现状：`command_output` 返回 `Result<String>` 且**消费句柄**（`mem::take`），**不带退出码**；`command_run` 返回 `Result<i64>` 但走 `status()`，子进程输出**继承父进程**（泄漏到检查器自己的 stdout/stderr）。`command_*` 家族无「句柄 + 状态对象」形态。
- **影响谁**：**P0**——差分检查器必须把三轴取自**同一次执行**，否则「跑两遍」会把副作用（写文件/网络/随机/时间）做两次，且给非确定性程序配上不匹配的输出/退出码对，产生既假阳性又假阴性的报告。
- **证据**：探针项 10（默认路径与解释器路径同结论）
  ```
  GAP005|both_axes_same_spawn|PARTIAL|code=3 stdout=[PROBE_OUT_B\n]，但 run 阶段输出已泄漏到本进程
  ```
  即：先 `command_run` 再 `command_output` 确实能凑出「code + stdout」，**代价是 run 阶段子进程输出已直接漏进检查器自身的 stdout**；而只用 `command_output` 则**永远拿不到退出码**（`stdout_capture|OK` 与 `exit_code_nonzero|OK` 分属两次 spawn）。源码：`tenth/src/runtime/natives.rs:656-699`。
- **状态**：`阻塞`（P0 三轴同源不可得；退路只有「跑两遍」——会破坏副作用与确定性，属需要总师裁定的取舍，**本会话未采用**）
- **关联**：P0；GAP-006（stderr）；`tenth/src/runtime/natives.rs:656`

---

## 索引

| 编号 | 一句话 | 状态 |
|------|--------|------|
| GAP-001 | 无法从 CLI 询问 JIT 是否生效 | 阻塞 |
| GAP-002 | 整型算术被硬限在 i32（`AUDIT-11.4.53`） | 阻塞 |
| GAP-003 | 无各阶段转储开关 | 阻塞 |
| GAP-004 | 无 verifier / IR 自检 | 阻塞 |
| GAP-005 | 子进程捕获输出与退出码（待验证） | 待验证 |
| GAP-006 | 无法捕获子进程 stderr（`command_output` 只回 stdout） | 阻塞 |
| GAP-007 | 无子进程超时 / 取消能力 | 阻塞 |
| GAP-008 | 无 per-child env 与 `env_unset`（P0 会静默跑成「双解释器」） | 阻塞 |
| GAP-009 | Vec 越界：VM 静默 Unit + exit 0，解释器报错（待转 AUDIT） | 待转入 AUDIT |
| GAP-010 | 同一次 spawn 拿不到 stdout + 退出码（三轴无法同源） | 阻塞 |

> **只追加，不回改**：上表 GAP-005 行保留立项原文；其实测结案（**部分可用**）见上方「GAP-005 实测结论（2026-09-17）」节。

---

# P0 差分守卫 v0 实测增量（2026-09-17，总师委派）

> 以下条目**全部由"写这个工具时真实撞上"产生**（章程 §四 工作规则）；旧条目一字未改。

## GAP-011 · `use` 路径段不能含 `-`（`tenth-lens` 目录名无法作为模块前缀 → 项目只能单文件）

- **缺什么**：让 `use` 接受含连字符的路径段；或给"文件级导入"一个与目录名解耦的入口（例如搜索路径包含**脚本自身所在目录**）
- **影响谁**：任何**仓库目录名带连字符**的 Tenth 项目（本项目即是）；以及任何想把自己拆成多模块的脚本——`tenth-lens/src/*.th` 无法被 `tenth-lens/main.th` 导入
- **证据**：`use tenth-lens::src::probe_mod::hello` → `错误：第 1 行第 20 列：语法错误 — 意外的标记：::`。根因在搜索路径：`tenth/src/hir/lower/import.rs:19-57` 只按 `<搜索目录>/<路径>.th` 查找，而 `main.rs:118-152` 给出的搜索目录只有 **cwd** / exe 同级 `std/` / `tenth/` / `tenth/std`——**不含脚本自身目录**；cwd 之下唯一可达的前缀就是带 `-` 的 `tenth-lens`
- **状态**：`绕行`（P0 采用**单文件** `tenth-lens/main.th`，逻辑分层用函数前缀 `rec_*`/`obs_*`/`grd_*` 表达；章程 `README.md` §五 预留的 `src/` 本轮**未使用**）
- **关联**：P0；`README.md` §五；`import.rs:19`；`main.rs:118`

## GAP-012 · 【跨路径分歧·静默失败】VM：全局标量「main 与函数都写」时丢失函数侧写入

- **缺什么**：VM 侧对模块级可变全局的正确读写——同一全局在 `main` 与函数中都被赋值时，两处写入都必须生效
- **影响谁**：**任何用顶层 `let mut` 当累加器 / 计数器 / 状态机的 Tenth 程序**。默认路径**静默算错值**，解释器算对；两路径退出码都是 0、无任何诊断——正是护城河 F（静默失败防护）该拦下的那一类
- **证据**：`tenth-lens/probes/gap012_vm_global_write.th`
  - VM：`N3|A_再被函数写=1`、`N5|A_再被函数写=2`（语义应为 2、4）；`TENTH_NO_VM=1`：2、4；**两路径退出码均 0**
  - 对照组 `B`（只在函数里写）两路径都是 1、2 ⇒ 触发条件是"同一全局**既在 main 写又在函数写**"
  - 本工具第一版症状：`[error] VM 运行时失败：第 467 行：运行时错误 — + 类型不匹配`（语句为 `N_RED = N_RED + 1`），同一脚本解释器可跑完
- **状态**：`绕行`（本工具计数改为 main 的**局部**变量 + `judge_key` 返回状态标签；`main.th` 文件头有同址注释）＋ **建议转入 `AUDIT.md`**（本会话红线禁改 AUDIT）
- **关联**：P0；GAP-013（同族"取值/存储丢信息"）；`probes/gap012_vm_global_write.th`

## GAP-013 · 【跨路径分歧】解释器：容器取出的**临时值**丢运行时类型标签（`type_name` → `unknown`）

- **缺什么**：解释器侧对 `Vec.get()` / `HashMap.get()` 返回值的类型标签保真（与 VM 对齐）
- **影响谁**：任何"从容器取出值**直接**当实参/键"的写法——① 该值作 `HashMap` 键被拒（运行时错误）；② 作 native 实参被**吞**（本工具自检里 `command_arg(h, v.get(i))` 收不到参数 → 子进程退回用法提示，**连退出码都跟着变**）
- **证据**：`tenth-lens/probes/gap013_interp_tag_loss.th`
  - VM：`X2|容器元素标签=string`；解释器：`X2|容器元素标签=unknown`
  - 解释器把该值当 HashMap 键 → `运行时错误 — HashMap 键类型不支持: alpha（仅支持 str/int/bool/float）`；VM 同表达式返回 1
  - `X4`：先 `let local = v.get(0);` 再使用 → 两路径都 `string` 且都能作键（**这就是绕行办法**）
- **状态**：`绕行`（`main.th` 的 `total_notes`/`total_noise` 与 `tests/selftest.th` 的 `spawn_*` 一律"先绑局部"；代码处留注释）
- **关联**：P0；GAP-012；`tests/selftest.th`

## GAP-014 · `parse_float` / `parse_int` 对非法输入静默返回 0（无失败信道）

- **缺什么**：解析失败的显式信道（`Result`/`Option` 版本，或 `parse_float_or` 之类的变体）
- **影响谁**：任何解析**外部文本**（日志 / 配置 / CSV）的程序——错值静默变 0 会污染下游结论；对"守卫/差分"类工具尤其致命：**0 是合法性能值，无法与"解析失败"区分**
- **证据**：`tenth-lens/probes/gap014_parse_float_silent.th`：`parse_float("n/a")=0.000000`、`parse_float("")=0.000000`、`parse_float("abc")=0.000000`、`parse_int("abc")=0`（两路径一致）
- **状态**：`绕行`（`main.th` 的 `rec_make` 显式判 `n/a` / `na` / 空串字面量，才敢信任数值）
- **关联**：P0；`main.th` `rec_make`

## P0 差分守卫 v0 实测结论（2026-09-17）

- **交付**：`tenth-lens/main.th`（记录/观察/守卫三层，单文件——理由见 GAP-011）；`tenth-lens/corpus/` 清单 5 份；`tenth-lens/tests/`（合成夹具 3 份 + `manifest_selftest.txt` + `selftest.th` 22 项断言）
- **验收 1**（污染 vs 干净）：`[RED] G2|matmul_512|vm|min  dev=174.64% (2.75x)  tol=±20.00% … vmref-grouped-polluted=15.397450 ms | perscenario-isolated=5.606350 ms`，退出码 **1**（**默认 ±20%，未放宽**）
- **验收 2**（同会话重复）：`KEYS|14 RED|0 OK|14`，退出码 **0**
- **验收 3**（干净集自比）：`KEYS|61 RED|0 OK|14 ONE|47`，退出码 **0**；工具**自带提示**"47 个 key 只有单一来源、未做比较——不等于已验证一致"（不冒充假绿）。补充**有可比对象**的干净集交叉自比（full-rerun + g2-run1/2，`--metric min`）→ `KEYS|50 RED|0 OK|6`，退出码 0
- **双后端逐字节一致**：三个验收清单 + 自检，默认路径与 `TENTH_NO_VM=1` 输出 `Compare-Object` **差 0 行**、退出码相同
- **全量 6 源诊断**（非验收）：`KEYS|126 RED|70 OK|55 NA|1`；其中 **52 条最严重配对不含已知污染源**（如 9-9 基线 `G2|elementwise_mul_1k` vm `11.722650 ms` vs 9-16 的 `1.591850 ms`，7.36×）⇒ 守卫不是只为"已知那一条"调的
- **观测（非缺口）**：`PERF-NOTE` 明写"跨次/跨机不可直接比"，而 9-9 与 9-16 的绝对量差常超 20%（`elementwise_mul_1k` 7×、`adamw_step_1e6` 5×）。即**"±20% 默认容差"只对同会话/同方法成立**；跨会话绝对量需按 NOTE 口径另行处理。本轮按总师规格**未改默认值**，仅如实记录

## 索引（追加段，2026-09-17）

| 编号 | 一句话 | 状态 |
|------|--------|------|
| GAP-011 | `use` 路径段不能含 `-`（本项目只能单文件） | 绕行 |
| GAP-012 | VM：全局标量被 main 与函数同时写 → 静默丢函数侧写入（**建议转 AUDIT**） | 绕行 |
| GAP-013 | 解释器：容器取出的临时值丢类型标签（作键被拒 / native 实参被吞） | 绕行 |
| GAP-014 | `parse_float`/`parse_int` 非法输入静默返回 0 | 绕行 |

> 原索引表（GAP-001~010）见上方，按"只追加"原则未改动。
