# Tenth 1.0.0 发布说明（Release Notes）

> **版本**：v1.0.0（自 v0.5.0）| **日期**：2026-08-04
> **定位**：Tenth 首个 **1.0 正式发布**——语言核心 100%、API 冻结、性能达标、护城河就绪、生态工具链齐全、跨平台产物可发布。
> **配套文档**：API 冻结与 semver 承诺 → `docs/API冻结清单.md`；语言规范（稳定契约）→ `docs/语言规范.md`；用法 → `docs/语言参考手册.md`。

---

## 一、里程碑回顾（M1–M5）

| 里程碑 | 内容 | 成果 |
|--------|------|------|
| **M1 语言核心 100%** | WASM 后端 4 缺口 / true letrec 递归闭包 / 运行时小缺口（broadcast_to·内联 mod·1D 张量迭代）/ 语义杂项（VM 引用语义·顶层 let·命名空间）/ 语言规范 7 章定稿 | 语言核心 **100%**；62 个实例全部双路径（VM=JIT 与解释器）可运行；规范覆盖语言核心 ✅84 项 |
| **M2 性能叙事** | A1 JIT-to-JIT 直接调用 / A2 小函数内联 / A3 opcode 覆盖 / A5 一致性套件 + 基准门槛 / A6 入参标量 ABI | fib(28) **180ms → 8ms（~20×）**、loop 1e7 **488ms → 60-65ms（~7.4×）**、matmul 0-1ms；基准门槛 CI（D4） |
| **M2.6-P 系列** | P1 纯标量固定开销消除 / P2 JIT 静默错值 5 维审计 / P3 f64+Int 特化 / P4 CallClosure JIT-to-JIT | fib **8ms**（每次调用 ~9.6ns）；f64 特化 fp_fib ~3ms / fp_poly ~7ms；closure-heavy **~6×**；修 AUDIT-11.4.36/11.4.37 |
| **M3 护城河** | shape 参数一致化 / typestate（G1-G6）/ 静默失败防护（丢弃 + 误用拦截）/ lossy 格 / 内存·算力预估 | 编译期 + 运行时**双层 shape 防御**；typestate 状态参数 + 调用点检查；`db_query().len()` 等误用编译期 warning |
| **M4 生态工具链** | tenthpm（M4.1）/ LSP（M4.2）/ 标准库 AI 生态（M4.3）/ 调试器·剖析器（M4.4） | 包管理器（传递依赖/锁文件/本地 registry）、语言服务器 13 项能力、3 优化器 + datasets + 分布式本地语义、CLI 调试器 + 热点剖析器 |
| **M5 稳定化与 1.0** | M5.1 API 冻结 + 规范定稿 / M5.2 fuzz + 大规模回归 / M5.3 跨平台产物 / M5.4 1.0 release | **API 冻结**（语法/标准库/CLI/native 四面）；fuzz 6 项 + 实例批量守护；Windows 5 产物 + CI matrix（Win/Linux/macOS）+ WASM 自举；**2377 passed / 0 failed** |

## 二、关键能力

- **语言核心**：张量一等类型（`Tensor[dtype, dims]` 符号维度编译期追踪）、21 个可微算子、控制流 + struct/enum/match + 泛型 + Trait + 闭包捕获（含 true letrec）+ 借用检查 + 智能指针（Box/Rc/Arc/Weak/Pin）+ 宏与自定义运算符 + async/await/spawn + try 块与 `?` + f16/f32/f64/bf16
- **自动微分**：方案 B（前向 f32 + 反向 f64 + 梯度按参数 dtype 写回）；`backward()` 返回 `Result`，shape 错误显式报错
- **护城河**：编译期 shape 检查 / 静默失败防护（丢弃 + 误用）/ typestate / lossy 格 / 内存·算力预估 / 编译期零除数 / 运行时 autodiff shape 校验 / FormalExplain 关系调试器
- **执行管线**：字节码 VM（默认，~0.2s 自举）/ 树遍历解释器（fallback）/ Cranelift JIT（热点编译 + 标量 ABI）/ WASM 后端
- **自举**：Tenth 编译器由 Tenth 自身编写（`tenthc/`，7 个 `.th` 文件，5000+ 行）；三条自举路径验证通过；自举 WASM 产物 `tenthc_full.wasm`（92.5KB）可复现 `[OK]`
- **标准库**：65 模块（63 用户模块 + prelude，71 个 `.th` 文件），覆盖 nn / optim / data / init / collections / string / json / toml / cli / logging / time / random / math / crypto / regex / net / http / fs / process / distributed 等；prelude 150+ native 符号
- **工具链**：`tenthpm`（10 子命令：init/build/test/run/add/remove/list/clean/publish/install，传递依赖解析 + 锁文件 + 本地 registry）、`tenth-lsp`（13 项能力）、`tenth-debug`（断点/单步/变量查看）、`tenth-prof`（top-N 热点剖析）

## 三、性能数据（release + JIT 默认路径，同机 5 次中位数）

| 基准 | 数值 | 对比 |
|------|------|------|
| fib(28) | **8ms** | v0.4.0 基线 180ms → **~20×** |
| loop 1e7 | **60-65ms** | v0.4.0 基线 488ms → **~7.4×** |
| matmul 150×150 | 0-1ms | — |
| f64 特化 fp_fib | ~3ms | P3 成果 |
| 闭包密集（CallClosure JIT-to-JIT） | ~6× 加速 | P4 成果 |
| 自举 | ~0.2s | — |

CI 基准门槛（`bench_gate_test -- --ignored`）：fib <100ms / loop <200ms / matmul <20ms（3× 裕量）。

## 四、API 冻结与语义版本（v1.0.0 起）

- **冻结范围**：语言核心语法/语义（规范 §2-6，稳定契约）、标准库 63 用户模块 + prelude 150+ native 符号、CLI 命令面（tenth/tenthpm/tenth-debug/tenth-prof/tenth-lsp）、运行时 native 注册名
- **semver 承诺**：`PATCH=bugfix / MINOR=兼容新增 / MAJOR=破坏性变更`；护城河红线（shape 检查 / 静默失败防护 / lossy 格 / 零除数 / 内存预估）**收紧属改进、放宽属破坏**（须 major）
- **编译器内部**（HIR / VM 指令 / JIT / tenthc 内部）非公开 API，可自由演进但不改变可观察行为
- 权威清单：`docs/API冻结清单.md`

## 五、已知限制（1.0 门槛遗留，如实披露）

> 1.0 门槛逐项评估见 `docs/API冻结清单.md` §5。以下为 1.0 发布时**已如实登记**的限制与绕行：

| 限制 | 性质 | 绕行 / 排期 |
|------|------|------------|
| **AUDIT-11.4.39 重载分派**：VM/解释器对「类型正确、参数数量/签名不同」的重载调用可能选中不同签名（静默错值风险） | 🔴 静默错值红线（建议 1.0 前修，已报总师） | 避免同一函数名混用不同参数数量的重载签名；编译期已拦截类型确定不兼容的调用；1.0.1+ 根治 |
| **AUDIT-11.4.34 VM match tuple + guard**：guard 失败后不试下一条 tuple 臂直接落 wildcard（静默错值） | 🔴 静默错值红线（建议 1.0 前修，已报总师） | tuple 多臂避免与 guard 混用（单 guard 臂 + wildcard 或 if 链）；1.0.1+ 根治 |
| **AUDIT-11.1 借用 B6 unsoundness**：语句粒度借用检查剩余 3 类变量转义（B7-1/2/3） | ⚠️ 语义健全性缺口（非内存安全；Tenth 无 unsafe/FFI/并发，B6' 条件健全性已论证） | 常规借用用法不受影响；1.0.1+ NLL 根治 |
| **远程 registry**：tenthpm 无中央仓库 | ⚠️ 纯功能缺口 | 本地 registry / git / `.tenthpkg` 发布安装闭环可用 |
| **AUDIT-11.4.43 JIT Union 字段修改**：触发 Cranelift 低化 panic（stderr 噪音），功能由 fallback 兜底正确 | ⚠️ 功能正确仅噪音 | 输出正确可忽略噪音；1.0.1+ 修复 |
| **JIT 边界（既有）**：Await/Yield、递归闭包创建、MakeCell/BindSelfCapture 整函数 fallback VM；>8 参不特化；深递归栈受原生栈限制（~36000 层） | ⚠️ 设计取舍 | 功能完整（fallback 正确执行） |
| **WASM 宿主定位**：`tenth build` 生成 .wasm + wasmi/wasmtime 宿主执行；tenth 主 crate 无法编译到 wasm32 target（cranelift 无 wasm32 ISA） | ⚠️ 平台边界 | 浏览器端运行属远期路线图 |

## 六、升级指引（v0.5.0 → v1.0.0）

**无破坏性变更**：v1.0.0 相对 v0.5.0 仅版本号 bump + 文档同步（语义/API 零改动），既有代码可直接迁移。

```bash
# 1. 从源码构建（Rust ≥ 1.95）
cargo build --release --manifest-path tenth/Cargo.toml

# 2. 验证自举
cargo run --release --manifest-path tenth/Cargo.toml -- run tenthc/main.th   # 期望 [OK]

# 3. 全量测试
cargo test --release --manifest-path tenth/Cargo.toml

# 4. 运行脚本
cargo run --release --manifest-path tenth/Cargo.toml run path/to/file.th
```

- **API 冻结承诺自 v1.0.0 生效**：语法 / 标准库符号 / CLI 命令 / native 名已冻结；后续兼容新增走 minor，破坏性变更须 bump major。
- 使用标准库模块：`use std::<mod>::<fn>;`（详见 `docs/语言参考手册.md`）。

## 七、发布产物与校验

**产物结构**（`dist/tenth-1.0.0-<platform>/`）：`bin/`（5 个可执行产物）+ `std/`（标准库，运行时必需）+ `docs/`（手册/规范/API 冻结清单/README）。

| 平台 | 产物 | 来源 |
|------|------|------|
| Windows | `tenth-1.0.0-windows-x86_64.zip` | `scripts/release/package.ps1`（本地） |
| Linux / macOS | `tenth-1.0.0-<linux|macos>-x86_64.tar.gz` | `scripts/release/package.sh`（CI matrix 原生构建） |

**校验**：

```bash
# Windows
powershell -ExecutionPolicy Bypass -File scripts/release/verify.ps1
# Linux/macOS
cd dist && sha256sum -c SHA256SUMS.txt
```

**产物哈希（SHA-256）**：见 `dist/SHA256SUMS.txt`（本机 Windows 打包 2026-08-04 实测）：

```
4ff0c4d4458b1837da4e609553c601de6efc44585686b53820b04bd478e7b52e  tenth-1.0.0-windows-x86_64.zip
```

**发布流程**：见 `scripts/release/README.md`（构建 → 冒烟 → 全量测试 → 打包 → 校验 → `git tag v1.0.0 && git push origin v1.0.0` 触发 CI 三平台构建）。

---

> 本文档随版本演进更新；状态变更同步 MEMO + 能力全梳理。
