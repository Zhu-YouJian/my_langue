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
