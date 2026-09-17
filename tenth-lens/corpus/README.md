# tenth-lens/corpus —— 语料清单与"去脆"说明

> 本文件回答两件事：**① 每条清单怎么跑、判什么；② 清单引用的输入日志/脚本怎么重新生成**
> （P0 的 5 份清单指向 `.agents/tmp/perf_*.log`——那是 **gitignore 的临时目录，清理即集体失效**，
> 故此处必须写明再生方法）。P0 模式在"清单引用的日志不存在"时会 **`[FATAL] SOURCE_MISSING` + 退出码 2**
> （2026-09-17 W6-lens P1 加硬，之前只是判红），不会静默空转。

---

## 一、清单总览

| 清单 | 模式 | 引用什么 | 期望判决 | 在验收里的角色 |
|------|------|----------|----------|----------------|
| `manifest_vm_diff.txt` | P0 守卫 | 2 份 `perf_*.log` | **RED（退出码 1）** | 验收 1：已知污染 vs 干净 |
| `manifest_g2_repeat.txt` | P0 守卫 | 2 份 `perf_g2_run*.log` | GREEN（退出码 0） | 验收 2：同会话重复 |
| `manifest_clean_set.txt` | P0 守卫 | 3 份 `perf_*.log` | GREEN（含 21 个 ONE 标注） | 验收 3：干净集自比 |
| `manifest_clean_cross.txt` | P0 守卫 | 3 份 `perf_*.log` | GREEN（建议配 `--metric min`） | 干净集**有可比对象**的交叉自比 |
| `manifest_all_sources.txt` | P0 守卫 | 6 份 `perf_*.log` | **默认路径 RED**；**解释器路径会栈溢出**（见 §四） | 非验收：全量红图 |
| `diff_probes.txt` | P1 差分 | `probes/*.th` 全量（10） | **全 PASS + 退出码 0** | 验收 ①（正例） |
| `diff_instances.txt` | P1 差分 | `Tenth实例/` 小批（13，含 2 个 `error`） | 全 PASS + 退出码 0 | 验收 ⑤（探索产出） |
| `diff_divergence.txt` | P1 差分 | 2 个夹具（1 真分歧 + 1 对照） | **RED（退出码 1）** | 验收 ②（判红是活的） |
| `diff_timeout.txt` | P1 差分 | 1 个卡死夹具 | **RED + `[TIMEOUT]`** | 验收 ③（超时明确标注） |
| `diff_nondet.txt` | P1 差分 | `Tenth实例/JSON处理/json_demo.th` | `--repeat 2` ⇒ `[NONDET]`（退出码 1） | 归因诚实性（见 §三） |

跑法（**项目根目录**；P1 模式要求 **env 干净**——`TENTH_NO_VM` 一旦被设置，检查器会**硬失败**）：

```powershell
$e = "tenth\target\release\tenth.exe"     # 已有构建，本轮无需 cargo

# P0
& $e run tenth-lens\main.th --manifest tenth-lens\corpus\manifest_g2_repeat.txt
& $e run tenth-lens\main.th --manifest tenth-lens\corpus\manifest_vm_diff.txt --only G2:matmul_512

# P1
& $e run tenth-lens\main.th --diff tenth-lens\corpus\diff_probes.txt
& $e run tenth-lens\main.th --diff tenth-lens\corpus\diff_instances.txt
& $e run tenth-lens\main.th --diff tenth-lens\corpus\diff_divergence.txt
& $e run tenth-lens\main.th --diff tenth-lens\corpus\diff_timeout.txt --timeout-ms 1200
& $e run tenth-lens\main.th --diff tenth-lens\corpus\diff_nondet.txt --repeat 2
```

**退出码约定**：`0` 全一致 / `1` 有红（DIVERGE、TIMEOUT、NONDET、EXPECT_FAIL）/ `2` 用法或输入错误
（清单不存在、期望值非法、脚本缺失、后端对照失败、父进程 env 脏）。

---

## 二、P0 的 `PERF|` 日志怎么重新生成（**关键：这些日志会随 `.agents/tmp/` 被清掉**）

日志格式：`PERF|<group>|<scenario>|<path>|<metric>|<value>|<unit>`，辅助行 `PERF-NA|`（不可用+原因）、
`PERF-NOTE|`（口径说明，**不是数据**）。方法学 SSOT 是 `docs/性能基线.md` §二/§六（**本文件不复制数字**）。

> 前提：`cargo` 不在 PATH。先 `$env:PATH = "%USERPROFILE%\.cargo\bin;$env:PATH"`。
> **规范跑法必须是"每测试一个独立进程"**（同进程内前一组堆状态会让后续张量算子膨胀 2–3×）。

| 日志（标签） | 生成命令（cwd = 项目根） |
|---|---|
| `perf_full_rerun_20260916.log`<br>（`full-rerun-grouped`） | 命令 ③a：`perf_g1_scalar_controlflow` / `perf_g2_tensor_ops` / `perf_g3_autodiff` / `perf_g4_nn_optim` / `perf_g5_compile_startup` / `perf_g6_three_path_compare` / `perf_interp_reference` **逐个**跑 `cargo test --release --manifest-path tenth/Cargo.toml --test perf_baseline_test -- --ignored --nocapture <目标>`，把 7 次输出合并重定向到本文件 |
| `perf_vm_perScenario_20260916.log`<br>（`perscenario-isolated`） | 命令 ③b：对 §六 列出的 21 个 `PERF_VM_SCENARIO` **逐个** `$env:PERF_VM_SCENARIO=<G>:<name>` 后跑同一目标，合并重定向 |
| `perf_g2_run1_20260916.log`、`perf_g2_run2_20260916.log`<br>（`g2-run1/2`） | 命令 ③a 中的**单个目标**：`cargo test --release --manifest-path tenth/Cargo.toml --test perf_baseline_test -- --ignored --nocapture perf_g2_tensor_ops`，**连跑两次**各存一份（同会话/同机的重复性证据） |
| `perf_vmref_20260916.log`<br>（`vmref-grouped-polluted`） | **污染源样本，历史产物**。它是"VM 列按**组**跑（全组同一进程）"的旧口径产物（`AUDIT-11.4.52` 修复前的做法）。**无法逐字节重建**；若要重现同一**污染签名**，用旧口径跑 VM 列（同一进程连续跑全部组）即可——正是 `manifest_vm_diff.txt` 要判红的那个差（`G2\|matmul_512\|vm\|min` 约 2.75×） |
| `perf_baseline_run_20260909.log`<br>（`baseline-0909-grouped`） | **9/9 历史基线**（`docs/性能基线.md` 文件头声明的权威源），方法同 ③a 的**每测试一进程**口径；跨会话数字按 `PERF-NOTE` 口径**不可直接比**（工具已如实标注） |

**再生后先自检**：`& $e run tenth-lens\main.th --manifest <某清单>` —— 若日志缺失，会看到
`[FATAL] SOURCE_MISSING|<标签>|<路径>` + 退出码 2，而不是"绿"。

---

## 三、P1 差分检查器：`PASS` 到底意味着什么（**别读成"正确"**）

- 每条用例两次 spawn（**各自一次 spawn 同源取三轴**：`stdout` / `stderr` / 退出码 + `timed_out`）：
  ① 默认路径；② `TENTH_NO_VM=1` 的解释器路径。
- 判决：`PASS`（两路径一致）/ `DIVERGE` / `TIMEOUT` / `NONDET` / `EXPECT_FAIL`。
- **`PASS` ＝ 两路径一致，不等于两侧都正确**——两侧一起错（同源同 bug）本工具**看不见**。
  报告第一行 `CLAIM|` 就是这个声明，不许摘掉。
- **`NONDET`（`--repeat n`，n≥2）**：同一侧自己连续 n 次输出就不同 ⇒ **被测程序输出非确定**，
  本工具无法把差异归因于后端。既不许报 `DIVERGE`（那是诬告后端），也不许报 `PASS`。
  默认 `--repeat 1`（只跑一次）无法区分这两种情形——所以默认口径下看到 `DIVERGE` 时，
  **先加 `--repeat 2` 复跑**再下结论。
- 最小复现（非确定输出的真实例子，与后端无关）：

  ```tenth
  // json_encode 按 HashMap 迭代序输出键，而该序**逐进程随机**（RandomState）
  fn main() {
      let mut config = HashMap::new();
      config.insert("name", "Tenth");
      config.insert("version", "0.3");
      config.insert("tensor_native", true);
      println(json_encode(config));
  }
  ```
  连跑多次，键序会变（VM 与解释器**都**变）——完整实例见 `Tenth实例/JSON处理/json_demo.th`。

**防假绿（三层）**：① 父进程 `env_get("TENTH_NO_VM")` 为 `Ok(_)`（**任何值，含 "0"/空串**）⇒ 硬失败
exit 2（`main.rs` 用 `is_ok()` 判定，不带这一层就会"双解释器自比自"）；② 无 per-child env / 无
`env_unset`（GAP-008）⇒ 必须"先跑完全部默认路径，再 `env_set` 跑解释器路径"**两阶段**；
③ **阳性对照**：跑一个已知两路径必不同的夹具（`tests/fixtures/diffctl_backend_split.th`，
分歧由 `AUDIT-11.4.56` 钉住），若它竟然"一致"，说明两次 spawn 落在同一后端 ⇒ 硬失败 exit 2。
`--no-control` 可关掉③，但报告会打印 `CONTROL|DISABLED` 并**如实标注证据强度下降**。

---

## 四、已知限制（诚实登记，勿当缺陷）

1. **P0 的 `manifest_all_sources.txt` 在解释器路径会栈溢出**：`TENTH_NO_VM=1 … --manifest tenth-lens\corpus\manifest_all_sources.txt`
   → `thread 'tenth-main' has overflowed its stack`（退出码 `0xC0000409`，无输出）；默认路径正常出
   `KEYS|126 RED|70 OK|55 NA|1`。**已 A/B 对照确认与 W6-lens 的多文件拆分无关**（`git show HEAD:tenth-lens/main.th`
   的原单文件版同样崩）。登记见 `GAPS.md` GAP-019。
2. **被检查脚本自身 spawn 子进程时**，检查器阶段 B 的 `TENTH_NO_VM=1` 会被**孙进程**继承
   （无 per-child env）⇒ 这类用例若判 `DIVERGE`，可能是**工具自身的测量副作用**。
   `probes/gap005_probe.th` 实测未受影响（10/10 一致），但**判红时必须先排除这一点**。
3. **`stderr` 轴只比"非空性"**（可加 `--stderr-contains <关键字>` 加强）——不比全文，
   因为诊断文本跨路径本来就可能带行号/列号差异（那是既有展示差异，不是分歧）。
4. 清单第 4 段（可选）是**额外参数**（空格分隔），当前语料均未使用。
