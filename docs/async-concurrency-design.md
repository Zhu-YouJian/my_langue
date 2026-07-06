# Tenth 语言异步/并发特性设计文档

> **版本**：v0.1（设计稿，未实现）
> **日期**：2026-07-06
> **状态**：设计阶段，待用户审批后进入实施
> **作者**：总师角色（架构守门人视角）
> **关联文档**：`MEMO.md`（生存级第一梯队缺口登记）、`CODE_WIKI.md`（模块架构）、`docs/语言参考手册.md`（语法权威定义）、`docs/shape-check-roadmap/战略规划.md`（护城河方向）

---

## 文档约定

- **中文撰写**，代码示例用 Tenth 语法（参考 `tenth/std/*.th` 现有风格）。
- **决策标记**：
  - `[已定]` —— 用户已确认的决策，不再讨论。
  - `[建议]` —— 总师建议的方案，有备选。
  - `[开放]` —— 需要后续讨论的开放问题，列入第 12 章。
- **风险标记**：`[风险:高]` / `[风险:中]` / `[风险:低]`。
- **影响标记**：`[影响:自举A/B/C]` 标注对自举三路径的影响。
- **文件引用**：使用相对路径 + 行号，例如 `tenth/src/lexer/token.rs:42`。

---

## 目录

1. [目标与范围](#1-目标与范围)
2. [语法设计](#2-语法设计)
3. [类型系统设计](#3-类型系统设计)
4. [HIR 数据结构设计](#4-hir-数据结构设计)
5. [运行时架构](#5-运行时架构)
6. [各后端实现策略](#6-各后端实现策略)
7. [WASM 可行性论证](#7-wasm-可行性论证重点章节)
8. [双侧同步策略（tenthc）](#8-双侧同步策略tenthc)
9. [标准库设计](#9-标准库设计)
10. [分阶段实施计划](#10-分阶段实施计划)
11. [风险评估](#11-风险评估)
12. [开放问题](#12-开放问题)

---

## 1. 目标与范围

### 1.1 设计目标

本设计旨在为 Tenth 语言引入**协作式协程 + async/await 异步并发**特性，使 Tenth 能够：

1. **支持单线程并发**：通过事件循环 + 状态机式 Future，让多个任务在单线程内交替执行，避免 OS 线程的开销与数据竞争。
2. **WASM 友好**：异步模型必须在不依赖未标准化 WASM 提案（stack-switching、JSPI）的前提下，在 wasmi 上完整可用——这是路径 C（全 WASM 闭环）的硬约束。
3. **与 AI 原生能力协同**：异步特性需与 Tenth 现有的张量、autodiff、shape 检查等护城河能力正交共存，不破坏现有语义。
4. **自举三路径不破坏**：任何前端改动（lexer/parser/hir）必须能同步到 tenthc，状态机 lower 策略必须在所有后端（VM/解释器/WASM/JIT）一致。
5. **借鉴成熟生态**：吸收 Rust 的 poll 状态机模型（零成本、WASM 友好）、Python 的 Task/Future/Coroutine API 命名、JS 的组合子（all/race/allSettled/any）、Go 的 select 多路复用语法，避免它们各自的痛点（手动 Pin、颜色函数强传染、stack-switching 依赖）。

### 1.2 范围声明 `[已定]`

**本次仅产出设计文档，不写任何实现代码**（不修改 `.rs` / `.th` 源文件）。设计文档经用户审批后，按第 10 章的分阶段计划逐步实施。

设计覆盖的范围：
- 语法（`async fn` / `await` / `spawn` / `channel` / `select`）
- 类型系统（`Future<T>` / `Task<T>` / `Channel<T>` / `Poll<T>` / `Waker`）
- HIR 数据结构与状态机 lower 策略
- 运行时架构（调度器、事件循环、Waker、Task 生命周期）
- 四个后端的实现策略（VM、解释器、WASM、JIT）
- WASM 可行性论证（针对 wasmi 限制）
- tenthc 双侧同步策略
- 标准库 `tenth/std/async/` 模块设计
- 分阶段实施计划与风险评估

### 1.3 非目标 `[已定]`

本设计**明确不做**以下内容：

| 非目标 | 原因 |
|--------|------|
| OS 线程（`std::thread`） | Tenth 的并发定位是协作式协程，OS 线程引入数据竞争、调度复杂度、WASM 不友好，与设计目标冲突。远期如需多线程，可作为独立特性再设计。 |
| 抢占式调度 | 协作式调度足够覆盖 IO 密集场景，抢占式需要栈中断或定时器中断，增加实现复杂度且 wasmi 不友好。 |
| WASM stack-switching 提案 | 该提案尚未标准化，wasmi 不支持。依赖它会破坏路径 C。`[风险:高]` |
| JSPI（JavaScript Promise Integration） | JSPI 仅 V8 浏览器实现，非 WASM 标准，wasmi 不支持。 |
| 真正的并行计算（CPU 多核） | Tenth 的并行计算走 GPU（CUDA）路线，CPU 并行不是本设计目标。异步特性解决的是 IO 并发，不是 CPU 并行。 |
| async generator（`async yield`） | 复杂度高，可远期再议。列入第 12 章开放问题。 |
| 手动 Pin 机制 | Tenth 有 GC（引用计数式内存管理），Future 的自引用状态机可用 GC 根替代 Pin，避免用户接触 Pin 概念。 |

### 1.4 与现有特性的关系

Tenth 当前的相关现状（来自调研）：

- **lexer**：已预留 `spawn` / `task` / `shard` / `node` 4 个关键字（`tenth/src/lexer/token.rs:42-45`），但 parser 完全不处理它们。
- **parser**：无 `async` / `await` 关键字，无对应 AST 节点。
- **HIR**：`HirExprKind` 枚举（`tenth/src/hir/hir.rs:11-138`）无异步相关节点。
- **类型系统**：`Type` 枚举（`tenth/src/hir/types.rs:24-40`）无 `Future` / `Channel` / `Task`。
- **VM 字节码**：`Op` 枚举（`tenth/src/runtime/vm.rs`）无异步指令。
- **运行时**：`runtime/vm.rs` 纯单线程同步执行，`runtime/interpreter/` 同样。
- **Cargo.toml**：无 `tokio` / `async-std` / `mio` 依赖。
- **标准库**：`tenth/std/` 下无 `async/` 目录；`time/sleep_ms` 是阻塞 sleep，`runtime.th` 的 `run_with_timeout` 是基于 step limit 的同步超时。
- **MEMO.md**：登记为"生存级第一梯队缺口"。

本设计将填补这一缺口，**不破坏**上述任何现有能力（所有现有测试必须继续通过）。

### 1.5 设计原则

1. **WASM 优先**：任何设计决策都要先问"wasmi 能不能跑"。如果不能，换方案。
2. **状态机优先**：async fn 编译为状态机 enum + poll 方法，不依赖运行时栈切换。
3. **GC 替代 Pin**：Future 状态机的自引用问题用 GC 根解决，用户不接触 Pin。
4. **颜色函数最小化**：async 函数确实"染色"（返回 Future），但提供 `block_on` bridge 缓解传染（见第 12 章开放问题）。
5. **双侧同步可分层**：tenthc 先支持语法解析，状态机 lower 可延后同步。
6. **不破坏护城河**：autograd shape 检查、shape 代数、JIT 特化等护城河能力不受影响。

---

## 2. 语法设计

### 2.1 关键字表更新 `[已定]`

Tenth 当前关键字（`tenth/src/lexer/token.rs:30-55`）：

```
fn, let, mut, if, else, match, for, while, loop, break, continue, return,
try, use, mod, pub, trait, impl, enum, struct, type, self,
spawn, task, shard, node,    // 预留未启用
macro, where, as, in, true, false, move
```

本设计的关键字变更：

| 关键字 | 状态 | 用途 |
|--------|------|------|
| `async` | **新增** | 标记异步函数 / 异步闭包 |
| `await` | **新增** | 等待 Future 完成，提取值 |
| `spawn` | **启用**（已预留） | 创建 Task，将 Future 提交给调度器 |
| `task` | **启用**（已预留） | 类型名 `Task<T>`（也可作为上下文关键字，见 2.7） |
| `channel` | **新增**（上下文关键字） | 创建通道（也可作为普通函数，见 2.4 的取舍） |
| `select` | **新增**（上下文关键字） | 多路复用 |
| `shard` | 暂不启用 | 远期分布式/并行计算预留 |
| `node` | 暂不启用 | 远期分布式计算预留 |

**设计决策**：`async` / `await` 作为真正的关键字（保留字），`channel` / `select` 作为**上下文关键字**（仅在特定语法位置识别，不破坏作为变量名的兼容性）。`spawn` 已是保留字，直接启用。`task` 已是保留字，但为了未来 `task` 块语法的灵活性，建议改为上下文关键字——不过这会破坏现有的预留语义，`[开放]` 见第 12 章。

**备选方案**：
- 全部用上下文关键字（不新增保留字）：兼容性好，但解析复杂度高，且 `async`/`await` 作为保留字是业界惯例，用户心智模型一致。
- 全部用保留字：解析简单，但破坏性大（`channel` / `select` 作为变量名常见）。

**最终选择**：`async`/`await`/`spawn` 为保留字，`channel`/`select` 为上下文关键字。`task` 暂保留为预留关键字（不启用），`Task<T>` 类型通过普通类型名访问。

### 2.2 `async fn` 与 `async` 闭包

#### 2.2.1 `async fn` 语法

```tenth
// 异步函数：返回 Future<T>，T 是函数体的返回类型
async fn fetch_data(url: String) -> String {
    let response = await http_get(url);
    response.body
}

// 异步函数可以带泛型
async fn load_tensor<T>(path: String) -> Tensor<T, 2> {
    let bytes = await read_file_async(path);
    parse_tensor<T>(bytes)
}

// 异步函数可以是方法
impl Dataset {
    async fn next_batch(&mut self, batch_size: i64) -> Tensor<f64, 2> {
        let indices = self.sample_indices(batch_size);
        await self.load_batch(indices)
    }
}
```

**语义**：
- `async fn foo(args) -> T` 的实际返回类型是 `Future<T>`，不是 `T`。
- 函数体在调用时**不立即执行**，而是构造一个 Future 状态机并返回。
- 函数体内部可以使用 `await` 等待其他 Future。
- `async fn` 不能直接调用——必须 `await` 或 `spawn`。

**类型推断规则**（见第 3 章详解）：
```tenth
let f: Future<String> = fetch_data("http://example.com".to_string());  // f 是 Future<String>
let s: String = await f;  // await 提取 T
```

#### 2.2.2 `async` 闭包

```tenth
// async 闭包：捕获环境，返回 Future
let fetcher = async |url: String| -> String {
    await http_get(url)
};

// 调用 async 闭包：返回 Future，不立即执行
let f: Future<String> = fetcher("http://example.com".to_string());
let s = await f;
```

**与普通闭包的关系**：
- 普通闭包：`|args| -> T { body }`，调用返回 `T`。
- async 闭包：`async |args| -> T { body }`，调用返回 `Future<T>`。

**设计理由**：与 `async fn` 对称，用户心智模型一致。

**备选方案**：让所有闭包都"自动"支持 async（如 JS 的 async 函数）——但这会让类型系统复杂化（闭包类型需区分 sync/async），且违反"显式优于隐式"原则。`[已定]` 采用显式 `async` 标记。

#### 2.2.3 `async` 块

```tenth
// async 块：内联构造 Future，不定义命名函数
let f: Future<i64> = async {
    let x = await compute_a();
    let y = await compute_b();
    x + y
};

// async 块可以捕获环境
let base = 10;
let f2: Future<i64> = async {
    base + await compute_c()
};
```

**语义**：`async { body }` 等价于一个立即调用的 async 闭包 `async || -> T { body }()`，返回 `Future<T>`。

**与 Rust 的 `async move` 块的区别**：Tenth 不需要 `move` 关键字（GC 管理，无需显式移动语义），闭包默认按引用捕获，必要时克隆。

### 2.3 `await` 表达式

#### 2.3.1 基本语法

```tenth
// await 后跟 Future 表达式
let x: i64 = await some_future;
let s: String = await fetch_data(url);

// await 优先级：低于方法调用，高于二元运算符
let y = await fetch() + 1;        // 等价于 (await fetch()) + 1
let z = await obj.method();        // await (obj.method())
let w = (await fetch()).field;     // await 后取字段，需要括号

// await 链式调用
let result = await (await first()).second();
```

**优先级**（从高到低）：
1. 后缀运算符（`.method()` / `[index]` / `!`）
2. `await`（前缀，右结合）
3. 一元运算符（`-` / `!`）
4. 二元运算符（`*` `/` `%` > `+` `-` > `==` `!=` ...）
5. 逻辑运算符（`&&` `||`）
6. 赋值（`=` `+=` ...）

**设计理由**：与 Rust/JS 一致，`await expr` 是一个表达式，可出现在任何表达式位置。

#### 2.3.2 await 的作用

`await future` 做三件事：
1. 调用 `future.poll(waker)`，传入当前 Task 的 Waker。
2. 如果返回 `Poll::Ready(v)`，`await` 表达式的值就是 `v`。
3. 如果返回 `Poll::Pending`，**挂起当前 Task**，将控制权交还调度器；当 Waker 被唤醒时，调度器将当前 Task 重新入就绪队列，最终恢复执行（重新调用 `poll`）。

**关键点**：`await` 是协作式调度的让步点。在 `await` 处，当前 Task 可能被挂起，其他 Task 可能运行。因此 `await` 后的代码不能假设局部变量未变（除非变量是 `let` 绑定的不可变值）。

**与 `?` 错误传播的交互**：`await?` 是常见模式——如果 Future 输出 `Result<T, E>`，`await?` 提取 `T` 或传播 `E`。
```tenth
async fn fetch_or_default(url: String) -> String {
    let body = await fetch_data(url)?;  // fetch_data 返回 Future<Result<String, HttpError>>
    Ok(body)
}
```
`[开放]`：`await?` 的精确语义（是否要求 Future 输出 Result）见第 12 章。

### 2.4 `spawn` 表达式

#### 2.4.1 基本语法

```tenth
// spawn 一个 Future，返回 Task<T>
let task: Task<String> = spawn fetch_data("http://example.com".to_string());

// spawn 一个 async 块
let task2: Task<i64> = spawn {
    let x = await compute_a();
    let y = await compute_b();
    x + y
};

// spawn 后可以继续做其他事，Task 在后台"并发"执行
do_something_sync();
// 等待 Task 完成
let result: String = await task;
```

**语义**：
- `spawn future` 将 `future` 提交给调度器，立即返回一个 `Task<T>` 句柄。
- Task 被加入就绪队列，调度器会在后续事件循环迭代中 poll 它。
- `await task` 等待 Task 完成，提取其输出值（如果 Task 还未完成，当前 Task 会被挂起）。
- Task 可以被 `abort`（取消），见 2.4.3。

**与 `await future` 的区别**：
- `await future`：当前 Task 直接 poll future，不创建新 Task。如果 future 是 Pending，当前 Task 挂起。
- `spawn future`：创建**新** Task，立即返回。当前 Task 不挂起（除非显式 `await` 返回的 Task）。

**典型用法**：
```tenth
// 并发发起 3 个请求
let t1 = spawn fetch("url1");
let t2 = spawn fetch("url2");
let t3 = spawn fetch("url3");
// 等待全部完成
let r1 = await t1;
let r2 = await t2;
let r3 = await t3;
// 或者用组合子
let results = await Future::all([t1, t2, t3]);  // 返回 [String; 3]
```

#### 2.4.2 `spawn` 的返回类型

`spawn future: Future<T>` 返回 `Task<T>`。`Task<T>` 本身也实现了 `Future<T>`，因此可以直接 `await`。

`Task<T>` 额外提供：
- `task.abort()` —— 取消任务
- `task.is_done()` —— 查询是否完成
- `task.id()` —— 获取任务 ID（用于调试）

#### 2.4.3 取消语义

```tenth
let task = spawn long_running_computation();
// 一段时间后取消
await sleep(1000);
if !task.is_done() {
    task.abort();
    println("task was cancelled");
}
```

**取消语义**（参考 Rust）：
- `abort` 将 Task 标记为已取消，丢弃其 Future（Drop）。
- Task 的 `poll` 之后返回 `Poll::Ready(Err(Cancelled))` 或类似机制——`[开放]` 见第 12 章。
- 取消是"协作式"的：Task 在下一个 `await` 点被实际终止，不是立即中断。
- 已取消的 Task 不能再被 poll。

**设计理由**：协作式取消避免资源泄漏（Future 在 await 点有机会释放资源），且实现简单（标记 + Drop）。

### 2.5 `channel<T>()` / `send` / `recv`

#### 2.5.1 通道创建

```tenth
// 创建 MPMC 通道（多生产者多消费者）
let (sender, receiver): (Sender<i64>, Receiver<i64>) = channel<i64>();

// 创建有界通道（容量 16）
let (tx, rx) = channel<i64>(16);

// 创建无界通道（默认）
let (tx2, rx2) = channel<String>();
```

**`channel` 作为上下文关键字**：在 `channel<T>()` 或 `channel<T>(cap)` 的语法位置识别为通道创建，其他位置仍可作为变量名。

**备选方案**：`channel` 作为普通函数（`std::async::channel::channel()`）——但通道创建是高频操作，上下文关键字更符合人体工程学。`[已定]` 采用上下文关键字。

#### 2.5.2 `send` / `recv`

```tenth
// 发送（异步，有界通道满时挂起）
async fn producer(tx: Sender<i64>) {
    for i in 0..100 {
        await tx.send(i);  // 返回 Future<()>
    }
    tx.close();
}

// 接收（异步，通道空时挂起）
async fn consumer(rx: Receiver<i64>) {
    while let Some(val) = await rx.recv() {  // 返回 Future<Option<T>>
        println("got: {}", val);
    }
}
```

**`send` / `recv` 作为方法**：`Sender<T>::send(value)` 和 `Receiver<T>::recv()`，返回 Future。不是关键字。

**关闭语义**：
- `sender.close()` —— 关闭发送端，接收端的 `recv` 在缓冲区耗尽后返回 `None`。
- `receiver.close()` —— 关闭接收端，发送端的 `send` 返回 `Err(Closed)`。
- 所有 Sender 丢弃后，通道自动关闭。

**MPMC 语义**：
- 多个 Sender 可以并发 `send`（在单线程协程模型下，是交替 send，无数据竞争）。
- 多个 Receiver 可以并发 `recv`，每个消息只被一个 Receiver 收到。
- 通道内部用就绪队列管理等待的 send/recv。

#### 2.5.3 通道的 select

```tenth
// 在多个通道上 select
select {
    val = rx1.recv() => {
        println("from rx1: {}", val);
    }
    val = rx2.recv() => {
        println("from rx2: {}", val);
    }
}
```

`select` 语法见 2.6。

### 2.6 `select` 多路复用

#### 2.6.1 基本语法

参考 Go 的 select，但用 Tenth 风格：

```tenth
// 等待多个 Future 中的第一个完成
select {
    result = fetch1 => {           // 等待 Future 完成
        println("first: {}", result);
    }
    result = await fetch2 => {     // 显式 await（等价形式）
        println("second: {}", result);
    }
    _ = timeout(5000) => {         // 超时分支
        println("timeout");
    }
}
```

#### 2.6.2 通道 select

```tenth
select {
    val = rx1.recv() => {          // 等待 rx1 有消息
        process_a(val);
    }
    val = rx2.recv() => {          // 等待 rx2 有消息
        process_b(val);
    }
    _ = tx.send(42) => {           // 等待 tx 可发送
        println("sent");
    }
}
```

#### 2.6.3 select 语义

- `select` 块包含多个 `case`，每个 case 是 `<pattern> = <future_expr> => <body>`。
- `select` 等待**任意一个** case 的 Future 完成，执行对应 body，然后结束。
- 其他未完成的 Future **不会**被取消（除非显式 abort）——它们继续作为独立 Future 存在。
- 如果多个 case 同时就绪，**随机**选一个（避免饥饿）。
- `select` 本身是异步的（在 async 上下文中使用），返回所选 case body 的值。

**备选方案**：
- 选第一个就绪的（顺序优先）：可能饥饿，不采用。
- 选所有就绪的（每个都执行）：那是 `Future::all` 的语义，不是 select。
- 随机选一个：Go 的做法，公平，`[已定]`。

#### 2.6.4 `default` 分支

```tenth
// 非阻塞 select：如果没有 case 就绪，执行 default
select {
    val = rx.recv() => {
        process(val);
    }
    default => {
        println("no message, doing other work");
    }
}
```

**语义**：`default` 分支在所有 case 都未就绪时立即执行，使 select 成为非阻塞操作。

#### 2.6.5 select 的返回值

```tenth
// select 是表达式，返回所选分支的值
let result: String = select {
    a = fetch1 => { format("first: {}", a) }
    b = fetch2 => { format("second: {}", b) }
};
```

每个分支的 body 必须类型一致（或可统一推断）。

### 2.7 `task` 上下文 `[开放]`

**当前设计**：`task` 关键字暂不启用，`Task<T>` 类型通过普通类型名访问。

**远期可能用法**（开放问题，见第 12 章）：
```tenth
// 可能的 task 块语法（远期）
task {
    let result = await compute();
    result * 2
} on_complete(|r| println("done: {}", r));
```

**当前决策**：`[已定]` 不启用 `task` 块语法，`Task<T>` 类型通过 `std::async::task` 模块访问。`task` 保留为预留关键字。

### 2.8 代码示例

#### 2.8.1 并发数据加载

```tenth
use std::async::*;
use std::async::future::*;

// 并发加载多个数据源
async fn load_all_datasets() -> (Tensor<f64, 2>, Tensor<f64, 2>, Tensor<f64, 2>) {
    // spawn 三个并发 Task
    let t1 = spawn load_train_data("train.csv");
    let t2 = spawn load_test_data("test.csv");
    let t3 = spawn load_validation_data("val.csv");

    // 等待全部完成
    let train = await t1;
    let test = await t2;
    let val = await t3;

    (train, test, val)
}

// 主入口：block_on 启动事件循环
fn main() {
    let (train, test, val) = block_on(load_all_datasets());
    println("train shape: {}", train.shape());
}
```

#### 2.8.2 超时控制

```tenth
use std::async::*;
use std::async::time::*;

async fn fetch_with_timeout(url: String, timeout_ms: i64) -> Option<String> {
    select {
        body = fetch(url) => {
            Some(body)
        }
        _ = sleep(timeout_ms) => {
            println("request timed out after {}ms", timeout_ms);
            None
        }
    }
}

// 或者用 with_timeout 组合子
async fn fetch_with_timeout_v2(url: String, timeout_ms: i64) -> Result<String, TimeoutError> {
    fetch(url).with_timeout(timeout_ms).await
}
```

#### 2.8.3 生产者-消费者

```tenth
use std::async::*;

async fn producer(tx: Sender<i64>, count: i64) {
    for i in 0..count {
        await tx.send(i);
        await sleep(10);  // 模拟生产耗时
    }
    tx.close();
}

async fn consumer(rx: Receiver<i64>, id: i64) {
    while let Some(val) = await rx.recv() {
        println("consumer {}: got {}", id, val);
        await sleep(15);  // 模拟处理耗时
    }
}

fn main() {
    block_on(async {
        let (tx, rx) = channel<i64>(8);  // 有界通道，容量 8
        let p = spawn producer(tx, 100);
        let c1 = spawn consumer(rx.clone(), 1);
        let c2 = spawn consumer(rx, 2);  // MPMC：两个消费者

        await p;
        await c1;
        await c2;
    });
}
```

#### 2.8.4 与 Tensor / autodiff 集成

```tenth
use std::async::*;
use std::nn::linear::*;

// 异步加载模型参数并前向传播
async fn async_forward(params_path: String, input: Tensor<f64, 2>) -> Tensor<f64, 2> {
    let params = await load_params_async(params_path);
    linear(input, params.weight, params.bias)
}

// 并发推理多个输入
async fn batch_inference(
    params_path: String,
    inputs: Vec<Tensor<f64, 2>>,
) -> Vec<Tensor<f64, 2>> {
    // 并发发起所有前向传播
    let tasks: Vec<Task<Tensor<f64, 2>>> = inputs
        .iter()
        .map(|x| spawn async_forward(params_path.clone(), x.clone()))
        .collect();

    // 等待全部完成
    let mut results = Vec::new();
    for t in tasks {
        results.push(await t);
    }
    results
}
```

**注意**：autodiff 的 `start_grad` / `backward` 等操作当前是同步的。异步上下文中的 autograd 集成见第 12 章开放问题。

### 2.9 语法设计决策汇总

| 决策点 | 选择 | 理由 | 备选 |
|--------|------|------|------|
| `async`/`await` | 保留字 | 业界惯例，心智模型一致 | 上下文关键字（解析复杂） |
| `spawn` | 启用预留关键字 | 已预留，直接启用 | 改为函数调用 |
| `channel`/`select` | 上下文关键字 | 兼容作为变量名 | 保留字（破坏性大） |
| `task` | 暂不启用 | 远期预留 | 立即启用 task 块语法 |
| `shard`/`node` | 暂不启用 | 远期分布式预留 | —— |
| await 优先级 | 介于后缀和一元之间 | 与 Rust/JS 一致 | —— |
| select 公平性 | 随机选择 | Go 做法，避免饥饿 | 顺序优先 |
| async 闭包 | 显式 `async` 标记 | 显式优于隐式 | 自动推断 |
| async 块 | 支持 | 与 Rust 一致，内联 Future | 只支持 async fn |

---

## 3. 类型系统设计

### 3.1 `Future<T>` 类型

#### 3.1.1 定义

`Future<T>` 表示一个**异步计算**，最终会产出类型为 `T` 的值（或被取消）。

```tenth
// Future 是一个 trait（概念上），实际实现为状态机 enum
trait Future<T> {
    async fn poll(&mut self, waker: Waker) -> Poll<T>;
}
```

**设计决策**：`Future<T>` 在类型系统中表示为 `Type::Future(Box<Type>)`，即 `Future<T>` 是泛型类型，T 是输出类型。

**与 Rust 的区别**：
- Rust 的 `Future` trait 关联类型是 `Output`，poll 返回 `Poll<Self::Output>`。
- Tenth 简化为 `Future<T>`，T 直接作为类型参数，poll 返回 `Poll<T>`。
- Tenth 的 `poll` 方法本身是同步的（不是 async），返回 `Poll<T>`。这与 Rust 一致——poll 不能是 async，否则无限递归。

#### 3.1.2 用户视角

用户**不直接调用 `poll`**。`Future` 通过以下方式消费：
- `await future` —— 在 async 上下文中等待
- `spawn future` —— 提交到调度器
- 组合子（`map` / `then` / `await_all` / `await_race`）—— 见第 9 章

**`Future<T>` 是 trait 还是具体类型？** `[建议]`
- 作为 trait：用户可以自定义 Future 实现（高级用法）。
- 作为具体类型（enum）：实现简单，但用户无法自定义。

**最终选择**：`Future<T>` 是**内置 trait**（编译器认识），用户通常通过 `async fn` / `async` 块构造 Future，不需要手动实现 trait。手动实现 Future 是高级特性，可通过 `impl Future for MyType` 实现（远期）。

### 3.2 `Task<T>` 类型

#### 3.2.1 定义

`Task<T>` 是 `spawn` 返回的句柄，表示一个**被调度器管理的 Future**。

```tenth
// Task<T> 实现了 Future<T>，可以 await
struct Task<T> {
    id: TaskId,
    // 内部：调度器持有的状态机引用
}

impl<T> Future<T> for Task<T> {
    async fn poll(&mut self, waker: Waker) -> Poll<T> {
        // 查询调度器：Task 完成了吗？
        // 如果完成，返回 Ready(结果)
        // 否则注册 waker，返回 Pending
    }
}
```

**Task 的额外方法**：
- `abort(&mut self)` —— 取消任务
- `is_done(&self) -> bool` —— 查询是否完成
- `id(&self) -> TaskId` —— 获取任务 ID

**与 Future 的关系**：`Task<T>` 是 `Future<T>` 的子类型（实现 trait）。所有 `Task` 都是 `Future`，但不是所有 `Future` 都是 `Task`——只有 `spawn` 创建的才是 `Task`。

### 3.3 `Channel<T>` 类型

#### 3.3.1 Sender / Receiver

```tenth
// 通道由两部分组成
struct Sender<T> {
    inner: ChannelHandle<T>,
}

struct Receiver<T> {
    inner: ChannelHandle<T>,
}

// Sender 和 Receiver 的方法
impl<T> Sender<T> {
    async fn send(&mut self, value: T) -> Result<(), Closed>;  // 有界通道满时挂起
    fn close(&mut self);
    fn is_closed(&self) -> bool;
}

impl<T> Receiver<T> {
    async fn recv(&mut self) -> Option<T>;  // 通道空时挂起，关闭后返回 None
    fn close(&mut self);
    fn is_closed(&self) -> bool;
}
```

#### 3.3.2 Channel 类型在类型系统中的表示

`Sender<T>` / `Receiver<T>` 在类型系统中表示为：
- `Type::Generic { base: "Sender", args: [T] }`
- `Type::Generic { base: "Receiver", args: [T] }`

或者更明确地新增类型变体：
- `Type::Sender(Box<Type>)`
- `Type::Receiver(Box<Type>)`

**设计决策**：`[建议]` 用 `Type::Generic`，避免类型枚举膨胀。`Sender` / `Receiver` 作为标准库类型注册，编译器不特殊处理。

**备选**：新增 `Type::Sender` / `Type::Receiver` 变体——但这样每增加一个标准库类型都要改 Type 枚举，不利于扩展。

### 3.4 `Poll<T>` 枚举

`Poll<T>` 是**内部类型**，用户通常不直接接触（除非手动实现 Future）。

```tenth
enum Poll<T> {
    Ready(T),
    Pending,
}
```

**用途**：`Future::poll` 的返回类型。

**用户接触场景**：
- 手动实现 `Future` trait（高级用法）。
- 调试时查看 Future 状态。

**类型系统表示**：`Type::Generic { base: "Poll", args: [T] }`。

### 3.5 `Waker` 类型

`Waker` 是**内部类型**，表示"唤醒机制"。当 Future 处于 Pending 时，它通过 Waker 通知调度器"我准备好了，请重新 poll 我"。

```tenth
struct Waker {
    inner: WakerHandle,
}

impl Waker {
    fn wake(&self);       // 唤醒关联的 Task，重新入就绪队列
    fn wake_by_ref(&self); // 同上，但不消耗 self
}
```

**用户接触场景**：
- 手动实现 `Future` trait 时，需要存储 Waker 以便后续唤醒。
- 调试。

**设计决策**：Waker 是不透明类型（opaque），用户不能构造，只能从 `poll` 的参数获取并存储/调用。

### 3.6 类型推断规则

#### 3.6.1 `async fn` 的返回类型

```tenth
async fn foo() -> T { body }  // 实际返回 Future<T>
```

类型推断规则：
- `async fn foo(args) -> T` 的签名在类型系统中是 `fn(args) -> Future<T>`。
- 调用 `foo(args)` 返回 `Future<T>`，不是 `T`。
- 函数体内，`body` 的类型必须是 `T`（或可转换为 `T`）。

**与普通 fn 的对比**：
```tenth
fn sync_foo() -> i64 { 42 }         // 类型：fn() -> i64
async fn async_foo() -> i64 { 42 }  // 类型：fn() -> Future<i64>

let x: i64 = sync_foo();            // 直接得到 i64
let f: Future<i64> = async_foo();   // 得到 Future<i64>
let y: i64 = await async_foo();     // await 提取 i64
```

#### 3.6.2 `await` 的类型提取

```tenth
let x: T = await future_t;  // future_t: Future<T>
```

类型推断规则：
- `await expr` 要求 `expr: Future<T>`。
- `await expr` 的类型是 `T`。
- 如果 `expr` 不是 `Future<T>` 类型，编译错误。

**特殊处理**：`Task<T>` 实现了 `Future<T>`，因此 `await task_t` 也是合法的，类型为 `T`。

#### 3.6.3 `spawn` 的类型

```tenth
let task: Task<T> = spawn future_t;  // future_t: Future<T>
```

类型推断规则：
- `spawn expr` 要求 `expr: Future<T>`。
- `spawn expr` 的类型是 `Task<T>`。
- `Task<T>` 实现了 `Future<T>`，所以 `await (spawn future_t)` 也是合法的。

#### 3.6.4 `channel` 的类型

```tenth
let (tx, rx): (Sender<T>, Receiver<T>) = channel<T>();
```

类型推断规则：
- `channel<T>()` 返回 `(Sender<T>, Receiver<T>)`。
- `channel<T>(cap)` 返回 `(Sender<T>, Receiver<T>)`，cap 是 `i64`。
- 类型参数 T 可以省略，从后续使用推断（如果上下文明确）。

#### 3.6.5 `select` 的类型

```tenth
let result: U = select {
    a = future_a => { body_a }   // body_a: U
    b = future_b => { body_b }   // body_b: U
};
```

类型推断规则：
- 每个 case 的 `future_expr` 必须是 `Future<T_i>` 类型。
- 每个 case 的 `body` 类型必须一致（或可统一为 `U`）。
- `select` 块的类型是 `U`。
- `default` 分支的 body 也必须是 `U`。

### 3.7 与现有类型系统的集成

#### 3.7.1 Type 枚举扩展

当前 `Type` 枚举（`tenth/src/hir/types.rs:24-40`）：
```rust
pub enum Type {
    Base(BaseType),
    Tensor { dtype: Box<Type>, dims: Vec<Dim> },
    Array(Box<Type>),
    FnType { params: Vec<Type>, ret: Box<Type> },
    TypeParam { name: String },
    Generic { base: Box<Type>, args: Vec<Type> },
    Ref(Box<Type>),
    MutRef(Box<Type>),
    Struct(String),
    Enum(String),
    Tuple(Vec<Type>),
    Unknown,
}
```

**扩展方案** `[建议]`：

新增 `Type::Future(Box<Type>)` 变体，专门表示 `Future<T>`。理由：
- `Future<T>` 是核心类型，频繁使用，专门变体便于模式匹配。
- 与 `Type::Ref` / `Type::Array` 等变体一致（它们也是 Box<Type> 的包装）。

不新增 `Type::Task` / `Type::Sender` / `Type::Receiver` / `Type::Poll` / `Type::Waker`——这些用 `Type::Generic` 或 `Type::Struct` 表示。

**扩展后的 Type 枚举**：
```rust
pub enum Type {
    Base(BaseType),
    Tensor { dtype: Box<Type>, dims: Vec<Dim> },
    Array(Box<Type>),
    Future(Box<Type>),    // 新增：Future<T>
    FnType { params: Vec<Type>, ret: Box<Type> },
    TypeParam { name: String },
    Generic { base: Box<Type>, args: Vec<Type> },
    Ref(Box<Type>),
    MutRef(Box<Type>),
    Struct(String),
    Enum(String),
    Tuple(Vec<Type>),
    Unknown,
}
```

**备选方案**：用 `Type::Generic { base: "Future", args: [T] }` 表示 Future，不新增变体——但这样 Future 与用户自定义泛型类型无异，编译器无法特殊处理（如 async fn 的返回类型推断）。

#### 3.7.2 Tensor + Future 协作

```tenth
// Future<Tensor<T, N>> 是合法类型
async fn load_tensor_async(path: String) -> Future<Tensor<f64, 2>> {
    let bytes = await read_file_async(path);
    Tensor::from_bytes(bytes)
}

// Future<Vec<Tensor<f64, 2>>> 也合法
async fn load_dataset(paths: Vec<String>) -> Vec<Tensor<f64, 2>> {
    let mut result = Vec::new();
    for p in paths {
        result.push(await load_tensor_async(p));
    }
    result
}
```

**autodiff 集成** `[开放]`：
- Future 内部可以包含需要梯度的 Tensor 吗？
- async fn 内部可以调用 `backward()` 吗？
- 这些涉及 autograd Tape 的生命周期与 Future 状态机的交互，见第 12 章开放问题。

**当前设计**：Future 与 Tensor 正交，`Future<Tensor<T, N>>` 是普通 Future。autograd 的 Tape 在 async 上下文中行为与同步一致（但跨 await 点的 Tape 引用需要小心，因为 Task 可能被挂起）。

#### 3.7.3 FnType 与 async fn

```tenth
async fn foo(x: i64) -> String { ... }
// foo 的类型：FnType { params: [i64], ret: Future<String> }
```

async fn 的类型是普通 `FnType`，返回类型是 `Future<T>`。这与普通 fn 类型一致，只是返回类型是 Future。

**这意味着**：
- async fn 可以作为值传递（函数指针 / 闭包）。
- `Vec<async fn(i64) -> String>` 是合法类型（实际上是 `Vec<fn(i64) -> Future<String>>`）。

#### 3.7.4 类型系统设计决策汇总

| 决策点 | 选择 | 理由 |
|--------|------|------|
| `Future<T>` 表示 | 新增 `Type::Future(Box<Type>)` | 核心类型，便于编译器特殊处理 |
| `Task<T>` 表示 | `Type::Generic { base: "Task", args: [T] }` | 标准库类型，不特殊处理 |
| `Sender/Receiver` | `Type::Generic` | 同上 |
| `Poll<T>` / `Waker` | `Type::Generic` / `Type::Struct` | 内部类型，用户少接触 |
| `async fn` 类型 | `FnType { ret: Future<T> }` | 与普通 fn 一致 |
| `await` 类型提取 | `await: Future<T> -> T` | 业界惯例 |
| `spawn` 类型 | `spawn: Future<T> -> Task<T>` | Task 实现 Future |
| `channel` 类型 | `channel<T> -> (Sender<T>, Receiver<T>)` | MPMC |

---

## 4. HIR 数据结构设计

### 4.1 新增 HIR 节点概览

在 `HirExprKind`（`tenth/src/hir/hir.rs:11-138`）中新增以下变体：

| 节点 | 用途 |
|------|------|
| `HirAsyncFn` | 异步函数定义（标记为 async） |
| `HirAsyncBlock` | async 块（内联 Future 构造） |
| `HirAsyncClosure` | async 闭包 |
| `HirAwait` | await 表达式 |
| `HirSpawn` | spawn 表达式 |
| `HirChannel` | channel 创建 |
| `HirSelect` | select 多路复用 |

**设计决策**：将 `HirAsyncFn` / `HirAsyncBlock` / `HirAsyncClosure` 作为独立节点，而不是在现有 `HirFnDef` / `HirExprKind::Closure` / `HirExprKind::Block` 上加 `is_async: bool` 标志。

**理由**：
- async 节点在 lower 阶段需要特殊处理（状态机转换），独立节点便于模式匹配。
- 不污染现有节点的语义（现有 Block/Closure 不需要处理 async 语义）。
- 类型检查时，async 节点的返回类型推断规则不同。

**备选方案**：在现有节点加 `is_async` 标志——更紧凑，但 lower 阶段需要检查标志，容易遗漏。`[已定]` 采用独立节点。

### 4.2 HIR 节点定义

#### 4.2.1 `HirAsyncFn`

```rust
#[derive(Debug, Clone, PartialEq)]
pub struct HirAsyncFn {
    pub name: String,
    pub generics: Vec<String>,
    pub generics_bounds: HashMap<String, Vec<String>>,
    pub params: Vec<(String, Type)>,
    /// 用户声明的返回类型（async fn 的 -> T 中的 T）
    /// 实际函数类型是 fn(params) -> Future<T>
    pub return_type: Type,
    pub body: HirExpr,
    pub span: Span,
}
```

**与 `HirFnDef` 的区别**：
- `HirFnDef` 是顶级函数定义（在 `HirProgram::functions` 中）。
- `HirAsyncFn` 也是顶级函数定义，但在 `HirProgram::async_functions` 中（新增字段）。
- 两者的字段几乎相同，但语义不同：`HirAsyncFn` 的 body 会被 lower 为状态机。

**`HirProgram` 扩展**：
```rust
pub struct HirProgram {
    pub functions: Vec<HirFnDef>,
    pub generic_funcs: Vec<HirFnDef>,
    pub async_functions: Vec<HirAsyncFn>,  // 新增
    // ... 其他字段不变
}
```

#### 4.2.2 `HirAsyncBlock`

```rust
#[derive(Debug, Clone, PartialEq)]
pub struct HirAsyncBlock {
    pub body: HirExpr,
    /// 捕获的环境变量（类似闭包）
    pub captures: Vec<String>,
    pub ty: Type,  // Future<T>，T 是 body 的类型
    pub span: Span,
}
```

#### 4.2.3 `HirAsyncClosure`

```rust
#[derive(Debug, Clone, PartialEq)]
pub struct HirAsyncClosure {
    pub params: Vec<(String, Type)>,
    pub body: HirExpr,
    pub captures: Vec<String>,
    pub ty: Type,  // FnType { ret: Future<T> }
    pub span: Span,
}
```

#### 4.2.4 `HirAwait`

```rust
#[derive(Debug, Clone, PartialEq)]
pub struct HirAwait {
    pub future: Box<HirExpr>,
    /// await 表达式的类型（即 Future 的输出类型 T）
    pub ty: Type,
    pub span: Span,
}
```

#### 4.2.5 `HirSpawn`

```rust
#[derive(Debug, Clone, PartialEq)]
pub struct HirSpawn {
    pub future: Box<HirExpr>,
    /// Task<T>，T 是 Future 的输出类型
    pub ty: Type,
    pub span: Span,
}
```

#### 4.2.6 `HirChannel`

```rust
#[derive(Debug, Clone, PartialEq)]
pub struct HirChannel {
    /// 通道元素类型
    pub elem_ty: Type,
    /// 容量：None = 无界，Some(n) = 有界
    pub capacity: Option<HirExpr>,
    /// 返回类型：(Sender<T>, Receiver<T>)
    pub ty: Type,
    pub span: Span,
}
```

#### 4.2.7 `HirSelect`

```rust
#[derive(Debug, Clone, PartialEq)]
pub struct HirSelect {
    pub arms: Vec<HirSelectArm>,
    /// 默认分支（可选）
    pub default: Option<HirExpr>,
    /// select 块的返回类型
    pub ty: Type,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct HirSelectArm {
    /// 绑定模式：`val = future_expr` 中的 `val`
    /// 可以是变量名，也可以是 `_`（忽略）
    pub binding: HirPattern,
    /// 要等待的 Future 表达式
    pub future: HirExpr,
    /// 分支体
    pub body: HirExpr,
}
```

### 4.3 状态机 lower 策略（核心）

`[风险:高]` 这是本设计最复杂的部分。async fn 的 body 必须被转换为状态机 enum + poll 方法，这是 async/await 能在 wasmi 上运行的根本保证。

#### 4.3.1 状态机 lower 的必要性

**为什么不直接运行 async fn 的 body？**
- async fn 的 body 包含 `await`，await 可能挂起当前 Task。
- 如果直接运行 body，挂起时需要"保存续延"（continuation）——这需要栈切换或 CPS 变换。
- 栈切换在 wasmi 上不可行（不支持 stack-switching）。
- CPS 变换会让所有函数都变成 CPS 风格，侵入性太强。

**状态机 lower 的核心思想**：
- 将 async fn 的 body 按await 点切分为多个"段"。
- 每个段是一个状态。
- 将 body 中的局部变量提升为状态机字段。
- poll 方法根据当前状态执行对应的段，返回 Ready 或 Pending。

**结果**：async fn 变成一个普通 enum + 一个同步的 poll 方法。poll 是普通函数调用，不需要栈切换，wasmi 完全支持。

#### 4.3.2 状态机 lower 示例（源码 → 状态机 enum → poll 实现）

##### 源码

```tenth
async fn fetch_add(url: String, base: i64) -> i64 {
    let body = await fetch(url);        // await 点 1
    let len = body.length() as i64;
    let extra = await compute_extra();   // await 点 2
    base + len + extra
}
```

##### 状态机 enum

```tenth
// 编译器生成的状态机 enum（用户不可见）
enum FetchAddState {
    Start { url: String, base: i64 },
    AwaitingFetch { url: String, base: i64, future_fetch: Future<String> },
    AwaitingComputeExtra { url: String, base: i64, body: String, len: i64, future_compute: Future<i64> },
    Done,
}

struct FetchAddFuture {
    state: FetchAddState,
}

impl Future<i64> for FetchAddFuture {
    fn poll(&mut self, waker: Waker) -> Poll<i64> {
        loop {
            match &mut self.state {
                FetchAddState::Start { url, base } => {
                    // 启动第一个 Future
                    let future_fetch = fetch(url.clone());
                    self.state = FetchAddState::AwaitingFetch {
                        url: url.clone(),
                        base: *base,
                        future_fetch,
                    };
                    // 继续循环，立即 poll 第一个 Future
                }
                FetchAddState::AwaitingFetch { url, base, future_fetch } => {
                    match future_fetch.poll(waker.clone()) {
                        Poll::Ready(body) => {
                            let len = body.length() as i64;
                            let future_compute = compute_extra();
                            self.state = FetchAddState::AwaitingComputeExtra {
                                url: url.clone(),
                                base: *base,
                                body,
                                len,
                                future_compute,
                            };
                            // 继续循环，立即 poll 第二个 Future
                        }
                        Poll::Pending => {
                            return Poll::Pending;
                        }
                    }
                }
                FetchAddState::AwaitingComputeExtra { base, len, future_compute, .. } => {
                    match future_compute.poll(waker.clone()) {
                        Poll::Ready(extra) => {
                            let result = *base + *len + extra;
                            self.state = FetchAddState::Done;
                            return Poll::Ready(result);
                        }
                        Poll::Pending => {
                            return Poll::Pending;
                        }
                    }
                }
                FetchAddState::Done => {
                    panic!("poll after completion");
                }
            }
        }
    }
}
```

##### async fn 调用展开

```tenth
// 源码：fetch_add(url, base)
// 展开后：
fn fetch_add(url: String, base: i64) -> Future<i64> {
    FetchAddFuture {
        state: FetchAddState::Start { url, base }
    }
}
```

**关键点**：
1. `fetch_add` 函数本身是同步的，立即返回一个 `FetchAddFuture`。
2. `FetchAddFuture` 的 `poll` 方法是同步的，按状态分发。
3. 跨 await 的变量（`url`, `base`, `body`, `len`）被提升为状态机字段。
4. await 一个 Future 时，先 poll 一次；如果 Pending，存到状态机字段，返回 Pending；下次 poll 时从字段取出继续。
5. Waker 被传递给内部 Future，内部 Future 完成时会唤醒当前 Task。

#### 4.3.3 状态机 lower 的算法

```
输入：async fn 的 body（HIR 表达式）
输出：状态机 enum 定义 + poll 方法实现

算法：
1. 遍历 body，找出所有 await 点，将 body 切分为 N 个"段"（N = await 点数 + 1）。
2. 分析每段引用的局部变量，确定需要跨段保存的变量集合 V。
3. 生成状态机 enum：
   - 段 0 对应状态 Start（初始状态）
   - 段 i 的最后是 await future_i，对应状态 AwaitingFuture_i
   - 段 N 是最终结果，对应状态 Done
   - 每个状态包含该段及后续段需要的变量（从 V 中选取）
4. 生成 poll 方法：
   - loop { match state { ... } }
   - 每个状态执行对应的段
   - 段末尾的 await 转换为：poll 内部 future，Ready 则切换状态继续循环，Pending 则保存状态返回 Pending
5. 生成 async fn 的调用展开：构造初始状态机，返回 Future
```

**复杂度**：O(await 点数 × 变量数)。对于典型的 async fn（几个 await 点），复杂度可接受。

**实现挑战** `[风险:高]`：
- 控制流分析：if/while/for 中的 await 需要展开控制流图，复杂度高。
- 借用分析：跨 await 的引用需要确保引用有效性（Tenth 有 GC，相对简单，但需验证）。
- Drop 语义：状态机字段在状态切换时可能需要部分 Drop。

**分层实现策略** `[建议]`：
- 阶段 1：只支持"直线型" async fn（无 if/while/for 中的 await）。
- 阶段 2：支持 if 中的 await。
- 阶段 3：支持 loop 中的 await（最复杂，需要循环展开为状态转移）。

#### 4.3.4 状态机 lower 在 HIR 中的表示

lower 后的 HIR 是普通 HIR（无 async 节点），包含：
- 一个 enum 定义（状态机 enum）
- 一个 struct 定义（Future 类型）
- 一个 impl Future for struct（poll 方法）
- 一个普通 fn（原 async fn 的调用展开）

**这些都在 lower 阶段生成，不需要用户书写**。

**HIR 数据流**：
```
源码 → AST → HIR (含 HirAsyncFn) → lower (状态机转换) → HIR (普通 HIR) → bytecode/wasm/jit
```

**关键点**：状态机 lower 是 HIR → HIR 的 pass，输入是含 async 节点的 HIR，输出是纯同步 HIR。后续后端（bytecode/wasm/jit）不需要知道 async 的存在——它们只看到普通的 enum/struct/impl/fn。

#### 4.3.5 控制流中的 await 处理

##### if 中的 await

```tenth
async fn foo(cond: bool) -> i64 {
    let x = if cond {
        await fetch_a()
    } else {
        await fetch_b()
    };
    x + 1
}
```

状态机需要处理分支：
```
状态 Start: 评估 cond
  → cond true: 切换到 AwaitingFetchA
  → cond false: 切换到 AwaitingFetchB
状态 AwaitingFetchA: poll fetch_a
  → Ready(x): 切换到 DoneWithX(x)
  → Pending: 返回 Pending
状态 AwaitingFetchB: poll fetch_b
  → Ready(x): 切换到 DoneWithX(x)
  → Pending: 返回 Pending
状态 DoneWithX(x): 计算 x + 1，返回 Ready(x + 1)
```

##### loop 中的 await

```tenth
async fn counter() -> i64 {
    let mut count = 0;
    loop {
        let val = await recv();
        if val == 0 {
            break;
        }
        count += val;
    }
    count
}
```

状态机需要循环展开：
```
状态 Start: count = 0, 切换到 AwaitingRecv
状态 AwaitingRecv: poll recv
  → Ready(val):
    if val == 0: 切换到 Done, 返回 Ready(count)
    else: count += val, 继续在 AwaitingRecv 状态（重新发起 recv）
  → Pending: 返回 Pending
```

**关键点**：loop 中的 await 状态机可以"自循环"——同一个状态可以多次进入，每次更新字段。这比 if 分支复杂，但可处理。

##### for 循环中的 await

```tenth
async fn sum_all(urls: Vec<String>) -> i64 {
    let mut total = 0;
    for url in urls {
        let body = await fetch(url);
        total += body.length() as i64;
    }
    total
}
```

for 循环需要将迭代器保存为状态机字段：
```
状态 Start: total = 0, iter = urls.into_iter(), 切换到 AwaitingFetch
状态 AwaitingFetch:
  if iter.has_next():
    url = iter.next()
    future = fetch(url)
    切换到 AwaitingFetchBody { total, iter, future }
  else:
    返回 Ready(total)
状态 AwaitingFetchBody { total, iter, future }:
  poll future
  → Ready(body): total += body.length(), 切换到 AwaitingFetch（保留 total, iter）
  → Pending: 返回 Pending
```

#### 4.3.6 借用与 GC 交互

**问题**：async fn 中的局部变量被提升为状态机字段。如果变量是引用（`&T` 或 `&mut T`），引用的目标可能在 Task 挂起期间被释放。

**Tenth 的优势**：Tenth 有 GC（引用计数式），引用是 GC 根。状态机字段持有引用，就是 GC 根，目标不会被释放。

**因此**：Tenth 不需要 Rust 的 Pin 机制。Future 状态机可以自由移动（GC 根会自动更新）。

**验证点** `[风险:中]`：需要确认 Tenth 的 GC 是否支持"状态机字段持有引用时，目标不被提前回收"。如果 GC 是简单的引用计数，循环引用会导致泄漏——但这是 GC 本身的问题，不是 async 引入的。

### 4.4 类型系统 HIR 扩展

如第 3 章所述，`Type` 枚举新增 `Future(Box<Type>)` 变体。

**类型检查规则**：
- `HirAsyncFn` 的返回类型在类型检查时被包装为 `Future<T>`。
- `HirAwait` 的类型检查：要求 future 表达式类型是 `Future<T>`，结果类型是 `T`。
- `HirSpawn` 的类型检查：要求 future 表达式类型是 `Future<T>`，结果类型是 `Task<T>`（`Type::Generic { base: "Task", args: [T] }`）。
- `HirChannel` 的类型检查：结果类型是 `(Sender<T>, Receiver<T>)`。
- `HirSelect` 的类型检查：每个 arm 的 future 必须是 `Future<T_i>`，所有 arm 的 body 类型必须统一。

### 4.5 现有 HIR 节点与异步节点的交互

#### 4.5.1 `Call` / `MethodCall` 与 async fn

```tenth
let f = fetch_data(url);  // fetch_data 是 async fn
// f 的类型是 Future<String>
```

**类型检查**：调用 async fn 时，返回类型是 `Future<T>`，不是 `T`。这与普通 fn 调用的类型推断不同。

**实现**：在 `HirExprKind::Call` 的类型检查中，检查被调用函数是否是 async fn（在符号表中查找）。如果是，返回类型包装为 `Future<T>`。

**备选方案**：在 lower 阶段将 `HirAsyncFn` 转换为 `HirFnDef`（返回类型已是 `Future<T>`），然后类型检查只看到普通 fn。`[建议]` 采用此方案——lower 前做一次"async fn 标记"，类型检查时识别标记。

#### 4.5.2 `Closure` 与 async 闭包

```tenth
let f = async |x: i64| -> i64 { x + 1 };
// f 的类型：fn(i64) -> Future<i64>
```

`HirAsyncClosure` 在 lower 阶段转换为状态机（与 `HirAsyncFn` 类似，但是闭包形式）。

#### 4.5.3 `Block` 与 async 块

```tenth
let f = async { ... };
// f 的类型：Future<T>，T 是块的最后表达式类型
```

`HirAsyncBlock` 在 lower 阶段转换为状态机。async 块本质上是"匿名 async fn 的立即调用"。

#### 4.5.4 `If` / `Match` 中的 await

```tenth
let x = if cond {
    await fetch_a()
} else {
    await fetch_b()
};
```

`HirExprKind::If` 的分支中可以包含 `HirAwait`。状态机 lower 时，if 的两个分支都被纳入状态机（见 4.3.5）。

**类型检查**：`If` 的分支类型必须一致——如果分支是 `await future`，分支类型是 `T`（Future 的输出类型），不是 `Future<T>`。

#### 4.5.5 `TryBlock` 与 async

```tenth
async fn fetch_or_default(url: String) -> String {
    let body = try {
        await fetch(url)
    } catch {
        "default"
    };
    body
}
```

`HirExprKind::TryBlock` 中的 body 可以包含 await。状态机 lower 时，try/catch 被纳入状态机。

**复杂度** `[风险:中]`：try/catch 跨 await 点的错误传播需要小心处理——错误可能在 Task 挂起期间被抛出。需要明确"Future 完成时的错误"和"poll 期间的错误"的语义。

### 4.6 HIR 节点设计决策汇总

| 决策点 | 选择 | 理由 |
|--------|------|------|
| async 节点 | 独立 HirAsyncFn/Block/Closure | 便于 lower 阶段模式匹配 |
| 状态机 lower 位置 | HIR → HIR pass | 后端无需知道 async |
| Type::Future | 新增变体 | 核心类型，便于特殊处理 |
| Task/Sender/Receiver | Type::Generic | 标准库类型，不特殊处理 |
| 控制流中的 await | 分层支持（直线→if→loop） | 降低实现复杂度 |
| Pin 机制 | 不需要（GC 替代） | Tenth 有 GC |
| async fn 调用 | 立即返回 Future | 与 Rust 一致 |

---



## 5. 运行时架构

### 5.1 调度器总体设计

Tenth 的异步运行时是一个**单线程协作式调度器**，核心组件：

```
┌─────────────────────────────────────────────────────────┐
│                     Scheduler                           │
│                                                         │
│  ┌─────────────┐  ┌─────────────┐  ┌───────────────┐  │
│  │ Ready Queue │  │ Timer Heap  │  │ IO Event Source│  │
│  │ (FIFO)      │  │ (min-heap)  │  │ (native/WASM) │  │
│  └──────┬──────┘  └──────┬──────┘  └───────┬───────┘  │
│         │                │                 │           │
│         └────────────────┼─────────────────┘           │
│                          │                             │
│                  ┌───────▼───────┐                     │
│                  │  Event Loop   │                     │
│                  │  (主循环)      │                     │
│                  └───────┬───────┘                     │
│                          │                             │
│  ┌───────────────────────▼───────────────────────┐    │
│  │            Task Table                         │    │
│  │  task_id → Task { state, future, waker, ... } │    │
│  └───────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────┘
```

**核心组件**：
1. **Ready Queue**：就绪任务队列（FIFO），存放可以被 poll 的 Task。
2. **Timer Heap**：定时器堆（min-heap），存放 `sleep` 等 Future，按到期时间排序。
3. **IO Event Source**：IO 事件源，native 后端用 `mio`（跨平台 IO 多路复用），WASM 后端用 host import。
4. **Event Loop**：主循环，从就绪队列取 Task 执行，处理定时器和 IO 事件。
5. **Task Table**：Task 注册表，管理所有活跃 Task 的状态。

### 5.2 事件循环主循环

**事件循环伪代码**：

```
fn run(future: Future<T>) -> T {
    let main_task = spawn(future);  // 创建主 Task
    loop {
        // 1. 从就绪队列取一个 Task
        if let Some(task) = ready_queue.pop() {
            // 2. poll 这个 Task
            let waker = task.waker.clone();
            match task.future.poll(waker) {
                Poll::Ready(value) => {
                    // Task 完成
                    if task.id == main_task.id {
                        return value;  // 主 Task 完成，事件循环退出
                    }
                    task.mark_done(value);
                }
                Poll::Pending => {
                    // Task 仍挂起，等待 Waker 唤醒
                    // 不需要重新入队——Waker 会负责入队
                }
            }
        } else {
            // 3. 就绪队列为空，检查定时器
            if let Some(timer) = timer_heap.peek() {
                let now = current_time();
                if timer.deadline <= now {
                    timer_heap.pop();
                    timer.waker.wake();  // 唤醒等待定时器的 Task
                    continue;
                } else {
                    // 4. 没有就绪 Task，等待 IO 事件或定时器
                    let timeout = timer.deadline - now;
                    io_events.poll(timeout);  // 阻塞等待 IO，最多 timeout
                    // IO 完成后，相关 Waker 会被调用
                }
            } else {
                // 5. 没有定时器，没有就绪 Task
                if task_table.is_empty() {
                    panic!("deadlock: no tasks to run");  // 死锁检测
                }
                // 等待 IO 事件（无限期）
                io_events.poll(None);
            }
        }
    }
}
```

**关键点**：
1. 事件循环是**阻塞式**的——没有就绪 Task 时，会阻塞等待 IO 或定时器。
2. 死锁检测：如果没有任何 Task 可运行（全部 Pending 且没有定时器/IO），报错。
3. 主 Task 完成时，事件循环退出（`block_on` 语义）。
4. Waker 是关键：Task 挂起后，必须通过 Waker 重新入队。

### 5.3 Waker 机制

#### 5.3.1 Waker 的实现

```rust
// Rust 侧实现（概念）
pub struct Waker {
    task_id: TaskId,
    scheduler: Rc<RefCell<Scheduler>>,
}

impl Waker {
    pub fn wake(&self) {
        // 将关联的 Task 重新入就绪队列
        self.scheduler.borrow_mut().ready_queue.push(self.task_id);
    }
}
```

**Waker 的生命周期**：
1. Task 被 poll 时，调度器构造一个 Waker 传给 `future.poll(waker)`。
2. Future 如果返回 Pending，必须**存储** Waker（或其 clone）。
3. Future 完成时（IO 完成、定时器到期等），调用 `waker.wake()`。
4. `wake()` 将 Task 重新入就绪队列。
5. 下次事件循环迭代，Task 被 poll。

**Waker 的 clone**：Waker 必须可 clone（多个 Future 可能等待同一个 Task）。Waker 内部用 `Rc<RefCell<Scheduler>>` 共享调度器引用，clone 开销低。

#### 5.3.2 Waker 与 Future 的交互

```tenth
// 异步 sleep 的实现（概念）
struct SleepFuture {
    deadline: f64,
    waker: Option<Waker>,
}

impl Future<()> for SleepFuture {
    fn poll(&mut self, waker: Waker) -> Poll<()> {
        let now = current_time();
        if now >= self.deadline {
            return Poll::Ready(());
        }
        // 注册定时器，存储 waker
        scheduler.register_timer(self.deadline, waker.clone());
        self.waker = Some(waker);
        Poll::Pending
    }
}
```

**关键点**：Future 在返回 Pending 前，必须确保 Waker 被正确存储/注册。否则 Task 永远不会被唤醒（资源泄漏）。

### 5.4 Task 生命周期

```
┌──────────┐  spawn  ┌──────────┐  poll    ┌──────────┐
│ Created  │ ──────> │ Ready    │ ──────> │ Running  │
└──────────┘         └──────────┘          └──────────┘
                          ^                      │
                          │                      │ Poll::Pending
                          │                      ▼
                          │                 ┌──────────┐
                          │   waker.wake()  │  Pending │
                          └─────────────────│ (挂起)   │
                                            └──────────┘
                                                 │
                                                 │ Poll::Ready
                                                 ▼
                                            ┌──────────┐
                                            │  Done    │
                                            └──────────┘
                                                 │
                                                 │ abort
                                                 ▼
                                            ┌──────────┐
                                            │ Cancelled│
                                            └──────────┘
```

**状态说明**：
- **Created**：Task 刚被 `spawn` 创建，还未入就绪队列。
- **Ready**：在就绪队列中，等待被 poll。
- **Running**：正在被 poll（事件循环正在执行其代码）。
- **Pending**：poll 返回 Pending，等待 Waker 唤醒。
- **Done**：poll 返回 Ready，Task 完成。
- **Cancelled**：被 abort，不再被 poll。

**状态转换**：
1. Created → Ready：spawn 后立即入就绪队列。
2. Ready → Running：事件循环取出 Task 开始 poll。
3. Running → Ready：如果 poll 立即返回 Ready（且 Task 不是主 Task），Task 完成。如果是循环型 Future，可能立即重新入队。
4. Running → Pending：poll 返回 Pending，Task 挂起。
5. Pending → Ready：Waker.wake() 被调用，Task 重新入就绪队列。
6. Pending → Cancelled：abort 被调用。
7. Running → Done：poll 返回 Ready，Task 完成。

### 5.5 取消语义详解

#### 5.5.1 abort 的实现

```rust
pub fn abort(&mut self) {
    let mut scheduler = self.scheduler.borrow_mut();
    if let Some(task) = scheduler.task_table.get_mut(&self.task_id) {
        task.cancelled = true;
        // Drop future，释放资源
        task.future = None;
    }
    // 如果 Task 在就绪队列中，标记为跳过（或移除）
    // 如果 Task 在 Pending，Waker 被调用时检查 cancelled 标志
}
```

**关键点**：
- abort 设置 `cancelled` 标志，并 Drop future。
- Future 的 Drop 会递归 Drop 其内部状态（包括子 Future、Buffer 等）。
- 如果 Task 在就绪队列中，下次 poll 时检查 cancelled 标志，直接跳过。
- 如果 Task 在 Pending，Waker 被调用时检查标志，不重新入队。

#### 5.5.2 await 的取消传播

```tenth
async fn parent() {
    let task = spawn child();
    await sleep(1000);
    task.abort();  // 取消 child
}

async fn child() {
    let data = await fetch(url);  // 如果 parent abort，这里的 await 会怎样？
    process(data);
}
```

**取消传播语义** `[开放]`：
- **选项 A**：abort 立即终止 child，child 内部的 await 抛出 `Cancelled` 异常。
- **选项 B**：abort 标记 child，child 在下次 await 点检查标志并退出。
- **选项 C**：abort 只是 Drop future，child 不会再被 poll（静默消失）。

**建议**：选项 B（协作式取消）——child 在 await 点有机会清理资源（如关闭文件、释放锁），符合 RAII 语义。

**实现**：
- abort 标记 Task 为 cancelled。
- 下次 poll 时，poll 方法检查标志，如果 cancelled，返回 `Poll::Ready(Err(Cancelled))` 或类似机制。
- child 的 await 表达式收到 Cancelled，可以传播（`?`）或捕获（try/catch）。

`[开放]`：Cancelled 的精确语义（是异常还是特殊返回值）见第 12 章。

### 5.6 VM 字节码扩展

当前 VM 指令集（`tenth/src/runtime/vm.rs` 的 `Op` 枚举）有 46 条指令。本设计新增以下指令：

| 指令 | 操作数 | 语义 | 说明 |
|------|--------|------|------|
| `Spawn` | chunk_idx, num_args | 从栈上弹出 num_args 个参数和 Future 值，创建 Task，压入 Task 句柄 | spawn 表达式 |
| `Await` | 无 | 弹出 Future，poll，Ready 则压入值，Pending 则挂起当前 Task | await 表达式 |
| `Yield` | 无 | 让出控制权，当前 Task 重新入就绪队列末尾 | 协作式让步（用于 cooperative yielding） |
| `ScheduleSwitch` | 无 | 切换到调度器（让调度器选择下一个 Task） | 内部使用 |
| `ChannelNew` | elem_ty_idx, capacity_opt | 创建通道，压入 (Sender, Receiver) 元组 | channel 创建 |
| `ChannelSend` | 无 | 弹出 channel 和 value，发送，压入 Future<()> | channel send |
| `ChannelRecv` | 无 | 弹出 channel，压入 Future<Option<T>> | channel recv |
| `ChannelClose` | 无 | 弹出 channel，关闭 | channel close |
| `Select` | num_arms | 弹出 num_arms 个 (future, body_chunk) 对，执行 select | select 多路复用 |
| `TaskAbort` | 无 | 弹出 Task，abort | task.abort() |
| `TaskIsDone` | 无 | 弹出 Task，压入 bool | task.is_done() |
| `PollFuture` | 无 | 弹出 Future 和 Waker，poll，压入 Poll<T> | 内部使用（手动实现 Future） |
| `WakerWake` | 无 | 弹出 Waker，wake | 内部使用 |
| `MakeWaker` | 无 | 为当前 Task 创建 Waker，压入 Waker | 内部使用 |
| `BlockOn` | chunk_idx, num_args | 启动事件循环，阻塞直到 Future 完成 | block_on 入口 |

**字节码编号**：从 47 开始（当前最大是 46）。

**设计决策**：
- `Spawn` / `Await` / `ChannelNew` 等是高级指令，对应 HIR 节点。
- `PollFuture` / `WakerWake` / `MakeWaker` 是低级指令，用于手动实现 Future（高级用法）。
- `BlockOn` 是入口指令，启动事件循环。

**备选方案**：将所有异步操作实现为 native 函数（不新增字节码）——但这样 VM 无法感知"挂起"语义，需要 native 函数能操作 VM 栈，复杂度更高。`[已定]` 新增字节码指令，VM 原生支持异步。

#### 5.6.1 VM 的"挂起"实现

**问题**：`Await` 指令需要"挂起当前 Task"。但 VM 是栈式执行，如何保存当前执行状态？

**方案**：每个 Task 有独立的 VM 栈帧。挂起时，保存当前栈帧；恢复时，恢复栈帧。

```rust
pub struct Task {
    id: TaskId,
    state: TaskState,
    future: Option<Box<dyn Future>>,
    vm_frame: Option<VMFrame>,  // 保存的 VM 栈帧
    waker: Waker,
}

pub struct VMFrame {
    chunk: Rc<Chunk>,
    ip: usize,           // 指令指针
    stack: Vec<Value>,   // 操作数栈
    locals: Vec<Value>,  // 局部变量
}
```

**Await 指令的实现**：
```rust
Op::Await => {
    let future = self.pop();
    let waker = self.make_waker();
    match future.poll(waker) {
        Poll::Ready(value) => {
            self.push(value);  // 继续执行
        }
        Poll::Pending => {
            // 挂起：保存当前栈帧
            let frame = self.save_frame();
            current_task.vm_frame = Some(frame);
            current_task.state = TaskState::Pending;
            // 返回到事件循环
            return ExecutionResult::Pending;
        }
    }
}
```

**恢复执行**：
```rust
fn resume_task(task: &mut Task) {
    if let Some(frame) = task.vm_frame.take() {
        let mut vm = VM::from_frame(frame);
        vm.run();  // 从挂起点继续执行
    }
}
```

**关键点**：每个 Task 有独立的 VM 栈帧，挂起时保存，恢复时还原。这比"续延"实现简单，且 wasmi 友好（栈帧是普通数据结构）。

#### 5.6.2 字节码扩展的影响

`[影响:自举A/B/C]` 新增字节码指令的影响：
- **路径 A（Rust 全栈）**：VM 新增指令处理，无影响。
- **路径 B（Tenth 前端 + Rust 后端）**：bridge.rs 需要传递新指令，但 bridge 只传递 HIR，不直接处理字节码——影响有限。
- **路径 C（全 WASM 闭环）**：WASM 后端需要生成对应的新指令的 WASM 代码（实际上是 host import 调用），见第 6 章。

### 5.7 解释器实现策略

Tenth 有两个执行后端：
- **VM（字节码）**：`runtime/vm.rs`，栈式虚拟机。
- **解释器（tree-walk）**：`runtime/interpreter/`，直接遍历 HIR 执行。

#### 5.7.1 tree-walk 解释器的挑战

tree-walk 解释器直接执行 HIR 表达式。遇到 `await` 时，需要"挂起"当前执行。但 tree-walk 没有"栈帧"的概念——执行状态在 Rust 调用栈上（递归调用 `eval_expr`）。

**问题**：如何在 tree-walk 中实现"挂起"？

**方案 A：CPS 变换**
- 将所有 `eval_expr` 改为 CPS 风格（传入 continuation）。
- await 时，不调用 continuation，而是返回 Pending + 保存 continuation。
- 恢复时，调用保存的 continuation。

**缺点**：侵入性太强，所有 eval 函数都要改签名。`[风险:高]`

**方案 B：状态机 lower 后再解释**
- 即使是解释器后端，也先做状态机 lower（HIR → HIR）。
- lower 后的 HIR 是普通同步 HIR（无 await 节点），包含状态机 enum + poll 方法。
- 解释器执行 poll 方法，poll 内部调用调度器。
- await 被编译为：`poll future, if Pending return Pending`（在 poll 方法内部）。

**优点**：解释器不需要"挂起"机制，只是执行普通同步代码。
**缺点**：解释器也要做状态机 lower，与 VM 后端共享 lower pass。

`[已定]` 采用方案 B：解释器和 VM 共享状态机 lower pass，lower 后的 HIR 是普通同步 HIR，所有后端都执行同步代码。

#### 5.7.2 解释器的调度器集成

```rust
// 解释器执行 poll 方法
fn eval_poll(&mut self, future_expr: &HirExpr, waker: Value) -> Result<Value> {
    // future_expr 是状态机 struct 的引用
    // 调用其 poll 方法（普通方法调用）
    let state = self.eval_field(future_expr, "state")?;
    match state {
        Value::Enum(variant, data) => {
            match variant.as_str() {
                "Start" => { /* 执行段 0，更新状态 */ }
                "AwaitingFetch" => { /* poll 内部 future */ }
                // ...
                "Done" => panic!("poll after completion"),
            }
        }
        _ => panic!("invalid state"),
    }
}
```

**关键点**：解释器执行 lower 后的 HIR，poll 是普通方法调用。await 的"挂起"语义在 lower 阶段已经转换为状态机逻辑。

### 5.8 运行时架构设计决策汇总

| 决策点 | 选择 | 理由 |
|--------|------|------|
| 调度器类型 | 单线程协作式 | WASM 友好，无数据竞争 |
| 就绪队列 | FIFO | 公平，简单 |
| 定时器 | min-heap | 高效的最早到期查询 |
| IO 事件源 | native: mio, WASM: host import | 跨平台 |
| Waker 实现 | Rc<RefCell<Scheduler>> | 单线程，开销低 |
| Task 栈帧 | 每 Task 独立 VMFrame | 挂起/恢复简单 |
| 取消语义 | 协作式（标记 + Drop） | RAII 友好 |
| 解释器策略 | 共享状态机 lower | 不需要 CPS |
| 死锁检测 | 有（无 Task 可运行时报错） | 调试友好 |

---

## 6. 各后端实现策略

### 6.1 后端总览

Tenth 有四个执行后端：
1. **VM（字节码）**：默认后端，`compile/bytecode.rs` 编译，`runtime/vm.rs` 执行。
2. **解释器（tree-walk）**：fallback 后端，`runtime/interpreter/` 直接执行 HIR。
3. **WASM 后端**：`compile/wasm/`，编译到 WASM，wasmi 执行。
4. **JIT 后端**：`compile/jit/`，运行时翻译为 native 代码。

**关键约束**：所有后端必须共享同一套状态机 lower pass，确保语义一致。

### 6.2 VM（字节码）后端

#### 6.2.1 编译策略

`compile/bytecode.rs` 将 lower 后的 HIR 编译为字节码。lower 后的 HIR 是普通同步 HIR（无 async 节点），包含：
- 状态机 enum 定义 → 编译为 `Op::NewStruct` + `Op::MakeEnum`
- Future struct 定义 → 编译为 `Op::NewStruct`
- poll 方法 → 编译为普通函数（使用 `Op::CallN`）
- async fn 调用展开 → 编译为普通函数

**新增编译逻辑**：
- `HirSpawn` → `Op::Spawn`
- `HirAwait` → `Op::Await`
- `HirChannel` → `Op::ChannelNew`
- select / send / recv → 对应指令

**注意**：lower 后的 HIR 中，async 节点已被消除。但 spawn/await/channel/select 等运行时操作仍然存在——它们不是 async fn 内部的东西，而是用户显式的并发操作。

**修正**：spawn/await/channel/select 在 lower 阶段**不**被消除。它们是运行时操作，需要 VM 指令支持。lower 阶段只消除 async fn/async block/async closure（转换为状态机）。

#### 6.2.2 VM 扩展

VM 需要新增：
1. **Task 表**：管理所有活跃 Task。
2. **就绪队列**：FIFO 队列。
3. **定时器堆**：min-heap。
4. **IO 事件源**：mio（native）或 host import（WASM）。
5. **事件循环**：`run` 方法，处理上述组件。

**VM 结构扩展**：
```rust
pub struct VM {
    // 现有字段...
    pub scheduler: Scheduler,  // 新增：调度器
}

pub struct Scheduler {
    pub ready_queue: VecDeque<TaskId>,
    pub timer_heap: BinaryHeap<TimerEntry>,
    pub task_table: HashMap<TaskId, Task>,
    pub io_source: Box<dyn IoEventSource>,
}
```

#### 6.2.3 block_on 的实现

```rust
pub fn block_on(&mut self, future: Value) -> Value {
    let main_task = self.scheduler.spawn(future);
    self.scheduler.run_until_complete(main_task)
}
```

`block_on` 是同步函数，启动事件循环，阻塞直到主 Task 完成，返回结果。

### 6.3 解释器（tree-walk）后端

#### 6.3.1 共享 lower pass

解释器与 VM 共享状态机 lower pass。lower 后的 HIR 是普通同步 HIR。

**实现**：在 `runtime/interpreter/mod.rs` 中，执行前先调用 lower pass（如果 HIR 中有 async 节点）。

```rust
pub fn run(&mut self, program: &HirProgram) -> Result<Value> {
    // 先 lower async 节点
    let lowered_program = lower_async(program)?;
    // 然后执行
    self.execute(&lowered_program)
}
```

#### 6.3.2 解释器的调度器

解释器也需要调度器（与 VM 类似）。调度器逻辑相同，只是"执行 Task"的方式不同：
- VM：恢复 VM 栈帧，执行字节码。
- 解释器：调用 poll 方法的 `eval_expr`。

**调度器抽象**：
```rust
pub trait Executor {
    fn poll_task(&mut self, task_id: TaskId) -> Result<Poll<Value>>;
}

pub struct Scheduler<E: Executor> {
    executor: E,
    // ...
}
```

VM 和解释器都实现 `Executor` trait，调度器逻辑共享。

### 6.4 WASM 后端

#### 6.4.1 状态机式 Future 的 WASM 友好性

`[核心论点]` 状态机式 Future 天然 WASM 友好，原因：

1. **poll 是普通函数**：状态机的 poll 方法是普通 WASM 函数，无栈切换需求。
2. **状态机 enum 是普通内存**：状态字段存储在 WASM 线性内存中，与普通 struct 无异。
3. **调度器是 host 侧**：WASM 模块导出 poll 函数，host 侧（Rust 或 wasmi）驱动事件循环。
4. **无栈切换**：wasmi 不支持 stack-switching，但状态机方案不需要。

**对比**：如果用"绿色线程 + 栈切换"方案，需要 WASM stack-switching 提案（未标准化），wasmi 不支持。状态机方案完全规避此问题。

#### 6.4.2 WASM 编译策略

`compile/wasm/` 将 lower 后的 HIR 编译为 WASM 模块：

1. **状态机 enum** → WASM 结构体内存布局 + tag（enum variant 标记）。
2. **Future struct** → WASM 内存中的结构体。
3. **poll 方法** → WASM 函数，导出为 `__tenth_poll_<future_name>`。
4. **async fn 调用展开** → WASM 函数，构造初始状态机，返回 Future struct 指针。
5. **spawn/await/channel/select** → host import 调用（见 6.4.3）。

#### 6.4.3 host imports 扩展

WASM 模块需要以下 host imports：

| host import | 签名 | 用途 |
|-------------|------|------|
| `__tenth_spawn` | (future_ptr: i32) -> task_id: i32 | 创建 Task |
| `__tenth_await` | (future_ptr: i32) -> poll_result: i32 | poll Future（0=Pending, 1=Ready） |
| `__tenth_channel_new` | (elem_ty: i32, capacity: i32) -> channel_id: i32 | 创建通道 |
| `__tenth_channel_send` | (channel_id: i32, value_ptr: i32) -> future_ptr: i32 | 异步发送 |
| `__tenth_channel_recv` | (channel_id: i32) -> future_ptr: i32 | 异步接收 |
| `__tenth_channel_close` | (channel_id: i32) -> () | 关闭通道 |
| `__tenth_select` | (future_ptrs: i32, count: i32) -> selected_index: i32 | 多路复用 |
| `__tenth_task_abort` | (task_id: i32) -> () | 取消 Task |
| `__tenth_task_is_done` | (task_id: i32) -> i32 | 查询完成状态 |
| `__tenth_waker_wake` | (waker_id: i32) -> () | 唤醒 Task |
| `__tenth_timer_register` | (deadline: f64, waker_id: i32) -> timer_id: i32 | 注册定时器 |
| `__tenth_io_register` | (fd: i32, interest: i32, waker_id: i32) -> io_id: i32 | 注册 IO 事件 |
| `__tenth_block_on` | (future_ptr: i32) -> result_ptr: i32 | 启动事件循环 |

**host 侧实现**：
- Rust 主机：用 `mio` 实现 IO，用 `std::time` 实现定时器。
- wasmi 主机：host 侧驱动事件循环，WASM 模块提供 poll 函数。

#### 6.4.4 WASM 模块导出

WASM 模块导出：
- `main`：主入口（如果有 `fn main`）。
- `__tenth_poll_<future_name>`：每个 async fn 的 poll 函数。
- `__tenth_init_<future_name>`：每个 async fn 的初始化函数（构造初始状态机）。

**host 侧调用流程**：
```
1. host 调用 main()
2. main 调用 __tenth_block_on(future_ptr)
3. host 启动事件循环：
   a. host 调用 __tenth_poll_<future_name>(future_ptr, waker)
   b. WASM 返回 Poll::Pending 或 Poll::Ready
   c. 如果 Pending，host 等待 IO/定时器，然后再次 poll
   d. 如果 Ready，host 取出结果，唤醒等待的 Task
4. 主 Task 完成，事件循环退出
```

### 6.5 JIT 后端

#### 6.5.1 JIT 翻译策略

`compile/jit/translator.rs` 将 lower 后的 HIR 翻译为 native 代码。

**新增翻译**：
- 状态机 enum → native 结构体内存布局 + tag。
- poll 方法 → native 函数。
- spawn/await/channel/select → 调用 native 调度器函数（与 VM 共享）。

**JIT 优化**：
- 状态机 enum 的 match 可以编译为跳转表（jump table）。
- poll 方法的内联：对于小的 async fn，可以内联 poll 方法。
- Waker 的虚调用可以去虚化（如果只有一个 Future 实现）。

#### 6.5.2 JIT 与调度器集成

JIT 后端使用与 VM 相同的调度器（Rust 实现），只是"执行 Task"的方式不同：
- VM：解释字节码。
- JIT：执行 native 代码。

**调度器复用**：调度器逻辑在 Rust 侧，JIT 调用的 spawn/await 等函数是 Rust 函数，通过 FFI 调用。

### 6.6 关键论证：为什么状态机式 Future 在所有后端都可行

#### 6.6.1 论证

**论点**：状态机式 Future 是唯一不依赖未标准化提案、且在所有 Tenth 后端都可行的异步方案。

**论证**：

1. **VM 后端**：
   - 状态机是普通 enum，存储在堆上。
   - poll 是普通函数调用（字节码 CallN）。
   - 挂起通过保存 VM 栈帧实现（普通数据结构）。
   - ✅ 完全可行。

2. **解释器后端**：
   - 共享状态机 lower pass。
   - poll 是普通方法调用（eval_expr）。
   - 挂起通过 poll 返回 Pending 实现（不需要保存续延）。
   - ✅ 完全可行。

3. **WASM 后端**：
   - 状态机是 WASM 线性内存中的结构体。
   - poll 是普通 WASM 函数。
   - 挂起通过 poll 返回 Pending 实现。
   - host 侧驱动事件循环。
   - ✅ 完全可行，wasmi 支持。

4. **JIT 后端**：
   - 状态机是 native 内存中的结构体。
   - poll 是 native 函数。
   - 挂起通过 poll 返回 Pending 实现。
   - 调度器是 Rust 侧函数。
   - ✅ 完全可行。

**对比其他方案**：
- **绿色线程 + 栈切换**：VM 可行，WASM 不可行（wasmi 不支持 stack-switching）。❌
- **CPS 变换**：所有后端可行，但侵入性强，所有函数都要改签名。⚠️
- **OS 线程**：native 后端可行，WASM 不可行。❌

**结论**：状态机式 Future 是唯一全后端通用的方案，且实现侵入性最低。

#### 6.6.2 性能考量

| 后端 | 状态机开销 | 备注 |
|------|-----------|------|
| VM | 状态切换 + 字段访问 | 与普通 enum 操作一致 |
| 解释器 | 状态切换 + eval_expr | 与普通方法调用一致 |
| WASM | 状态切换 + 函数调用 | poll 是普通 WASM 函数调用 |
| JIT | 状态切换 + native 函数调用 | 可优化为跳转表 |

**零成本抽象**：状态机 lower 是零成本抽象——用户写的 async fn，编译后的代码与手写状态机一致。这与 Rust 的 async/await 一致。

### 6.7 各后端实现决策汇总

| 后端 | 策略 | 关键改动 |
|------|------|---------|
| VM | 新增字节码 + 调度器集成 | Op::Spawn/Await/..., Scheduler |
| 解释器 | 共享 lower pass + 调度器抽象 | Executor trait |
| WASM | 状态机编译 + host imports | __tenth_spawn/await/... |
| JIT | 翻译为 native + 调度器复用 | translator 扩展 |

---

## 7. WASM 可行性论证（重点章节）

### 7.1 wasmi 限制调研

wasmi 是 Tenth 路径 C（全 WASM 闭环）使用的 WASM 运行时。其限制：

| 特性 | wasmi 支持情况 | 影响 |
|------|---------------|------|
| WASM 1.0（核心规范） | ✅ 完全支持 | 基础功能可用 |
| WASM 2.0（ SIMD, threads） | ⚠️ 部分支持 | SIMD 可用，threads 有限 |
| stack-switching 提案 | ❌ 不支持 | 不能用绿色线程方案 |
| JSPI（JavaScript Promise Integration） | ❌ 不支持（仅 V8） | 不能用 JSPI 方案 |
| Resumable function calls | ✅ 支持 | host 侧可挂起 WASM，用于 host 异步 IO |
| WASI | ✅ 支持 | 文件/网络 IO |

**关键结论**：
1. 任何依赖 stack-switching 或 JSPI 的方案都不可行。`[风险:高]`
2. 状态机式 Future 完全可行（只用 WASM 1.0 特性）。
3. host 异步 IO 可通过 Resumable function calls 实现（高级优化）。

### 7.2 状态机式 Future 为何可行

#### 7.2.1 核心论点

状态机式 Future 的运行时需求：
1. **存储状态**：状态机 enum 存储在 WASM 线性内存中——普通内存操作，WASM 1.0 支持。
2. **执行 poll**：poll 是普通 WASM 函数——普通函数调用，WASM 1.0 支持。
3. **返回 Poll<T>**：Poll 是普通 enum——普通返回值，WASM 1.0 支持。
4. **调度器**：在 host 侧（Rust）——不依赖 WASM 特性。

**结论**：状态机式 Future 只使用 WASM 1.0 核心特性，wasmi 完全支持。

#### 7.2.2 对比：栈切换方案为何不可行

**绿色线程 + 栈切换**的需求：
1. 每个 Task 有独立的 WASM 栈。
2. 挂起时切换栈（保存当前栈指针，加载新栈指针）。
3. 这需要 WASM stack-switching 提案（或自实现栈切换）。

**wasmi 的问题**：
- wasmi 不支持 stack-switching 提案。
- 自实现栈切换需要操作 WASM 栈指针，但 WASM 规范不允许直接操作栈（只能通过 call/return）。
- 即使勉强实现，性能和安全性都无法保证。

**结论**：栈切换方案在 wasmi 上不可行。`[风险:高]`

#### 7.2.3 对比：CPS 变换方案

**CPS 变换**的需求：
1. 所有函数改为 CPS 风格（传入 continuation）。
2. 挂起时返回 continuation，恢复时调用 continuation。

**问题**：
- CPS 变换是全局变换，所有函数都受影响。
- 生成的 WASM 代码膨胀（每个函数都有 continuation 参数）。
- 与 Tenth 现有 HIR 设计不兼容（现有函数不是 CPS 风格）。

**结论**：CPS 变换技术上可行，但侵入性太强，不采用。

### 7.3 host 异步 IO 集成（Resumable function calls）

wasmi 支持 **Resumable function calls**：host 调用 WASM 函数时，WASM 函数可以"挂起"（通过 `host_panic` 或类似机制），host 侧捕获挂起状态，稍后恢复执行。

#### 7.3.1 Resumable function calls 工作流程

```
1. host 调用 WASM 函数 foo()
2. foo() 内部调用 host import __tenth_io_wait(fd)
3. wasmi 挂起 foo() 的执行，返回到 host
4. host 注册 fd 的 IO 事件，等待
5. IO 完成后，host 恢复 foo() 的执行
6. foo() 继续执行
```

**用途**：实现 host 侧的异步 IO——WASM 模块发起 IO 请求，host 异步执行，完成后恢复 WASM。

#### 7.3.2 与状态机式 Future 的关系

**状态机式 Future 不依赖 Resumable function calls**：
- Future 的 poll 是普通同步函数。
- poll 内部调用 host import（如 `__tenth_io_wait`），host 立即返回结果（如果 IO 就绪）或返回 Pending（如果 IO 未就绪）。
- 如果 Pending，poll 返回 Pending，host 侧调度器选择下一个 Task。

**Resumable function calls 的优化用途**：
- 可以让 WASM 模块内的代码"看起来"是阻塞式（直接调用 IO，等待结果），而 host 侧异步执行。
- 但这会让 WASM 代码的执行模型变复杂（挂起/恢复语义）。
- `[建议]` 第一版不使用 Resumable function calls，只用状态机式 Future + host import。远期可作为优化。

#### 7.3.3 host import 的异步语义

```tenth
// WASM 侧：异步 IO 的实现
async fn read_file_async(path: String) -> Vec<u8> {
    let fd = __tenth_io_open(path);  // host import
    let mut buf = Vec::new();
    loop {
        let result = __tenth_io_read_async(fd);  // host import，返回 Future<Vec<u8>>
        let chunk = await result;
        if chunk.is_empty() {
            break;
        }
        buf.extend(chunk);
    }
    buf
}
```

**host 侧实现**：
- `__tenth_io_read_async` 返回一个 Future，host 侧注册 IO 事件。
- IO 完成时，host 调用 Waker.wake()，Task 被重新 poll。
- poll 时，`__tenth_io_read_async` 返回 Ready(data)。

**关键点**：host import 本身是同步的（立即返回 Future 句柄），异步性通过 Waker 机制实现。

### 7.4 路径 C（全 WASM 闭环）安全性论证

路径 C：全 WASM 闭环——Tenth 源码编译到 WASM，wasmi 执行。

#### 7.4.1 路径 C 的异步执行流程

```
Tenth 源码（含 async fn）
    ↓ 编译
WASM 模块（含状态机 poll 函数 + host imports）
    ↓ wasmi 加载
host 侧（Rust）启动事件循环
    ↓ 调用
WASM poll 函数
    ↓ 返回
Poll::Pending / Poll::Ready
    ↓ host 处理
IO/定时器事件 → Waker.wake() → 重新 poll
```

#### 7.4.2 安全性保证

1. **内存安全**：WASM 线性内存隔离，状态机数据不会越界。
2. **控制流安全**：WASM 函数调用受 WASM 规范保护，无栈溢出（状态机不递归）。
3. **资源安全**：host import 控制资源访问，WASM 模块不能直接访问文件系统。
4. **死锁检测**：host 侧调度器实现死锁检测。

#### 7.4.3 路径 C 不破坏的论证

- **状态机 lower**：在编译时完成，WASM 模块只包含同步代码。✅
- **host imports**：在 WASM 模块导入表声明，host 侧提供实现。✅
- **事件循环**：在 host 侧（Rust）运行，不依赖 WASM 特性。✅
- **poll 函数**：是普通 WASM 函数，wasmi 完全支持。✅

**结论**：状态机式 Future 完全不破坏路径 C。

### 7.5 体积/性能影响评估

#### 7.5.1 体积影响

**新增 WASM 代码**：
- 每个 async fn 生成一个状态机 enum + poll 函数。
- 状态机 enum 体积 = 变体数 × 字段大小。
- poll 函数体积 ≈ 原 async fn body 体积 + 状态分发逻辑。

**估算**：对于典型的 async fn（2-3 个 await 点），poll 函数体积约为原 body 的 1.5-2 倍。

**优化**：
- 状态机 enum 的字段去重（相同字段只存一份）。
- poll 函数的内联优化。
- 死状态消除（不可达的状态变体）。

#### 7.5.2 性能影响

**poll 开销**：
- 状态分发：一次 match（O(1)）。
- 字段访问：内存访问（O(1)）。
- Waker clone：Rc clone（O(1)）。

**对比同步函数**：
- 同步函数：直接执行。
- async fn：每次 await 多一次 poll 开销（状态分发 + 字段访问）。

**估算**：async fn 的开销约为同步函数的 1.1-1.5 倍（取决于 await 点数量）。

**对比其他方案**：
- 栈切换：栈切换开销高（保存/恢复寄存器），但 await 开销低。
- 状态机：无栈切换开销，但每次 await 多一次状态分发。

**结论**：状态机式 Future 的性能开销可接受，且在 WASM 上无替代方案。

### 7.6 WASM 可行性结论

| 维度 | 评估 |
|------|------|
| wasmi 兼容性 | ✅ 完全兼容（只用 WASM 1.0 特性） |
| 路径 C 安全性 | ✅ 不破坏 |
| 体积影响 | ⚠️ 中等（1.5-2x async fn body） |
| 性能影响 | ⚠️ 中等（1.1-1.5x 同步函数） |
| 实现复杂度 | ⚠️ 高（状态机 lower） |
| 替代方案 | ❌ 无（stack-switching/JSPI 不可用） |

**最终结论**：状态机式 Future 是 wasmi 上唯一可行的异步方案，且性能/体积开销可接受。

---

## 8. 双侧同步策略（tenthc）

### 8.1 tenthc 同步的必要性

Tenth 的自举三路径要求 Rust 编译器和 tenthc（Tenth 自举编译器）保持语义一致：
- 路径 A：Rust 全栈编译。
- 路径 B：Tenth 前端 + Rust 后端（bridge.rs）。
- 路径 C：全 WASM 闭环。

任何前端改动（lexer/parser/hir）必须同步到 tenthc，否则路径 B/C 会破坏。

### 8.2 tenthc 需同步的内容

| 模块 | 同步内容 | 优先级 |
|------|---------|--------|
| `tenthc/lexer/token.th` | 新增 `async` / `await` 关键字 | P0（必须） |
| `tenthc/lexer/lexer.th` | 识别 `async` / `await` token | P0 |
| `tenthc/parser/parser.th` | 解析 `async fn` / `await` / `spawn` / `channel` / `select` 语法 | P0（语法解析） |
| `tenthc/parser/ast.th` | 新增对应 AST 节点 | P0 |
| `tenthc/hir/hir.th` | 新增 `HirAsyncFn` / `HirAwait` / `HirSpawn` / `HirChannel` / `HirSelect` 节点 | P1（HIR 节点） |
| `tenthc/hir/lower.th` | 状态机 lower pass | P2（可延后） |
| `tenthc/hir/types.th` | 新增 `Type::Future` 变体 | P1 |

### 8.3 分层同步策略

#### 8.3.1 第 1 层：语法解析（P0）

tenthc 必须能**解析** async/await 语法，即使不能 lower 为状态机。

**目标**：
- tenthc 能识别 `async fn` / `await` / `spawn` / `channel` / `select` 关键字。
- tenthc 能构造对应的 AST 节点。
- tenthc 能构造对应的 HIR 节点（不要求 lower）。

**意义**：
- 路径 B（Tenth 前端 + Rust 后端）：tenthc 解析后，通过 bridge.rs 传递给 Rust 后端，Rust 后端做 lower。✅
- 路径 C（全 WASM）：tenthc 解析 + lower + 编译 WASM。需要 tenthc 也能 lower（P2）。⚠️

**风险**：如果 tenthc 不能 lower，路径 C 会破坏（tenthc 无法编译含 async 的 Tenth 代码到 WASM）。

**缓解**：第 1 层只保证 tenthc 能解析——这样 tenthc 自己的源码（如果不含 async）可以正常自举。路径 C 的 async 支持延后到第 2 层。

#### 8.3.2 第 2 层：类型标注（P1）

tenthc 能进行 async/await 的类型检查：
- `async fn` 的返回类型推断为 `Future<T>`。
- `await` 的类型提取。
- `spawn` / `channel` / `select` 的类型检查。

**意义**：tenthc 能验证 async 代码的类型正确性。

#### 8.3.3 第 3 层：状态机 lower（P2）

tenthc 能将 async fn lower 为状态机。

**意义**：路径 C 完全支持 async。

**风险** `[风险:高]`：状态机 lower 是复杂 pass，tenthc 同步难度高：
- tenthc 用 Tenth 编写，Tenth 的模式匹配、控制流分析与 Rust 不同。
- tenthc 的 HIR 数据结构与 Rust 侧 HIR 必须严格对齐（通过 bridge.rs）。
- 状态机 lower 的算法复杂（控制流分析、变量提升、状态生成）。

**缓解策略**：
- 先让 Rust 侧的 lower 充分测试和稳定。
- 然后逐步移植到 tenthc。
- 可以分阶段：先支持直线型 async fn，再支持 if/loop 中的 await。

### 8.4 自举验证策略

**自举三路径的验证**：

| 路径 | 验证命令 | async 支持要求 |
|------|---------|---------------|
| A | `cargo test --manifest-path tenth/Cargo.toml` | 完整支持 |
| B | `cargo run --release -- run tenthc/main.th`（用 Rust 后端） | tenthc 解析 + Rust lower |
| C | 全 WASM 闭环 | tenthc 解析 + tenthc lower（P2） |

**阶段 1-4 完成后**：
- 路径 A：完整支持 async。✅
- 路径 B：tenthc 解析 async，Rust 后端 lower。✅（只要 tenthc 同步到 P0）
- 路径 C：tenthc 不能 lower async，**路径 C 暂时不支持 async 代码**。⚠️

**风险**：如果 tenthc 自己的源码使用 async，路径 C 会破坏。
**缓解**：tenthc 源码**不使用 async**（tenthc 是编译器，不需要异步 IO）。这样 tenthc 可以自举，即使它不能 lower async。

**关键约束**：tenthc 必须能**解析** async 语法（P0），否则如果 tenthc 源码中包含 async（即使不执行），会报错。但实际上 tenthc 源码不需要 async，所以 P0 的要求是"能解析用户的 async 代码"，而不是"tenthc 自己用 async"。

### 8.5 tenthc 同步的风险评估

| 风险 | 概率 | 影响 | 缓解 |
|------|------|------|------|
| tenthc 解析 async 语法有 bug | 中 | 路径 B/C 破坏 | 充分测试 |
| tenthc 状态机 lower 与 Rust 不一致 | 高 | 路径 C 语义不一致 | 分阶段同步，先直线型 |
| tenthc HIR 数据结构与 Rust 不对齐 | 中 | bridge.rs 失败 | 严格对齐测试 |
| tenthc 自举性能下降 | 低 | 自举超时 | 性能监控 |

### 8.6 tenthc 同步决策汇总

| 决策点 | 选择 | 理由 |
|--------|------|------|
| 同步优先级 | P0（解析）→ P1（类型）→ P2（lower） | 降低风险 |
| tenthc 源码是否用 async | 否 | 避免自举依赖 |
| 路径 C 的 async 支持 | 延后到 P2 | 复杂度高 |
| 状态机 lower 同步 | 分阶段（直线→if→loop） | 降低难度 |

---


---

## 9. 标准库设计

### 9.1 模块组织

新建 tenth/std/async/ 目录，结构如下：

```
tenth/std/async/
├── future.th        # Future trait + 组合子
├── task.th          # spawn / abort / current_task
├── channel.th       # Channel<T> MPMC + send/recv/close
├── select.th        # select 多路复用（语法支持在编译器）
├── time.th          # sleep / timeout（异步版）
├── io.th            # 异步文件读写（native: mio, WASM: host import）
├── sync.th          # 异步锁（Mutex/RwLock/Semaphore/Event）
└── prelude.th       # async 模块的 prelude（重导出常用项）
```

### 9.2 async/future.th

```tenth
// Future trait 定义（编译器内置，此处为文档目的）
// trait Future<T> {
//     fn poll(&mut self, waker: Waker) -> Poll<T>;
// }

// Future 组合子
impl<T> Future<T> {
    // map: Future<T> -> Future<U>，通过 f: fn(T) -> U
    async fn map<U>(self, f: fn(T) -> U) -> Future<U> {
        let value = await self;
        f(value)
    }

    // then: Future<T> -> Future<U>，通过 f: fn(T) -> Future<U>
    async fn then<U>(self, f: fn(T) -> Future<U>) -> Future<U> {
        let value = await self;
        await f(value)
    }

    // catch: Future<T> -> Future<Result<T, E>>，捕获错误
    async fn catch<E>(self) -> Future<Result<T, E>> {
        try {
            Ok(await self)
        } catch (e: E) {
            Err(e)
        }
    }

    // with_timeout: Future<T> -> Future<Result<T, TimeoutError>>
    async fn with_timeout(self, ms: i64) -> Result<T, TimeoutError> {
        select {
            v = self => Ok(v),
            _ = sleep(ms) => Err(TimeoutError)
        }
    }
}

// Future::all: 等待所有 Future 完成，返回结果数组
async fn all<T>(futures: Vec<Future<T>>) -> Vec<T> {
    let mut results = Vec::new();
    for f in futures {
        results.push(await f);
    }
    results
}

// Future::race: 等待任意一个 Future 完成
async fn race<T>(futures: Vec<Future<T>>) -> T {
    let (tx, rx) = channel<T>(1);
    for f in futures {
        spawn async {
            let v = await f;
            await tx.send(v);
        };
    }
    await rx.recv().unwrap()
}

// Future::all_settled: 等待所有 Future 完成（不论成功/失败）
async fn all_settled<T, E>(futures: Vec<Future<Result<T, E>>>) -> Vec<Result<T, E>> {
    let mut results = Vec::new();
    for f in futures {
        results.push(await f);
    }
    results
}

// Future::any: 等待任意一个 Future 成功完成
async fn any<T, E>(futures: Vec<Future<Result<T, E>>>) -> Result<T, Vec<E>> {
    let mut errors = Vec::new();
    for f in futures {
        match await f {
            Ok(v) => return Ok(v),
            Err(e) => errors.push(e),
        }
    }
    Err(errors)
}
```

### 9.3 async/task.th

```tenth
// spawn: 创建 Task
fn spawn<T>(future: Future<T>) -> Task<T> {
    __tenth_spawn(future)
}

// abort: 取消 Task
fn abort<T>(task: &mut Task<T>) {
    __tenth_task_abort(task.id())
}

// is_done: 查询 Task 是否完成
fn is_done<T>(task: &Task<T>) -> bool {
    __tenth_task_is_done(task.id())
}

// current_task: 获取当前 Task 的引用
fn current_task() -> CurrentTaskHandle {
    __tenth_current_task()
}

// yield_now: 让出控制权（协作式让步）
async fn yield_now() {
    __tenth_yield();
}

// block_on: 启动事件循环，阻塞直到 Future 完成
fn block_on<T>(future: Future<T>) -> T {
    __tenth_block_on(future)
}

// Task 结构
struct Task<T> {
    id: TaskId,
}

impl<T> Task<T> {
    fn id(&self) -> TaskId { self.id }
    fn abort(&mut self) { __tenth_task_abort(self.id) }
    fn is_done(&self) -> bool { __tenth_task_is_done(self.id) }
}

impl<T> Future<T> for Task<T> {
    async fn poll(&mut self, waker: Waker) -> Poll<T> {
        __tenth_task_poll(self.id, waker)
    }
}
```
### 9.4 async/channel.th

```tenth
// channel 创建
fn channel<T>(capacity: i64) -> (Sender<T>, Receiver<T>) {
    let cap = if capacity <= 0 { None } else { Some(capacity) };
    __tenth_channel_new<T>(cap)
}

// 无界通道（语法糖）
fn unbounded_channel<T>() -> (Sender<T>, Receiver<T>) {
    __tenth_channel_new<T>(None)
}

// Sender<T>
struct Sender<T> {
    id: ChannelId,
}

impl<T> Sender<T> {
    async fn send(&mut self, value: T) -> Result<(), Closed> {
        await __tenth_channel_send(self.id, value)
    }
    fn close(&mut self) {
        __tenth_channel_close(self.id)
    }
    fn is_closed(&self) -> bool {
        __tenth_channel_is_closed(self.id)
    }
    fn clone(&self) -> Sender<T> {
        Sender { id: self.id }
    }
}

// Receiver<T>
struct Receiver<T> {
    id: ChannelId,
}

impl<T> Receiver<T> {
    async fn recv(&mut self) -> Option<T> {
        await __tenth_channel_recv(self.id)
    }
    fn close(&mut self) {
        __tenth_channel_close(self.id)
    }
    fn is_closed(&self) -> bool {
        __tenth_channel_is_closed(self.id)
    }
    fn clone(&self) -> Receiver<T> {
        Receiver { id: self.id }
    }
}

// try_send / try_recv: 非阻塞版本
impl<T> Sender<T> {
    fn try_send(&mut self, value: T) -> Result<(), TrySendError<T>> {
        __tenth_channel_try_send(self.id, value)
    }
}

impl<T> Receiver<T> {
    fn try_recv(&mut self) -> Result<T, TryRecvError> {
        __tenth_channel_try_recv(self.id)
    }
}
```

### 9.5 async/select.th

select 是上下文关键字，语法在编译器中处理。标准库提供运行时支持：

```tenth
// select 运行时支持（编译器生成代码调用此函数）
// __tenth_select(futures: Vec<*Future>, count: i32) -> i32
// 返回选中的 Future 的索引

// select 的宏形式（可选，如果支持宏）
// macro select! {
//     ($($pat = $fut => $body),* $(,)?) => {
//         ...
//     }
// }
```

### 9.6 async/time.th

```tenth
// sleep: 异步睡眠
async fn sleep(ms: i64) {
    let deadline = time_now_ms() + ms as f64;
    __tenth_timer_register(deadline);
    // 返回 Pending，定时器到期后 Waker 唤醒
}

// timeout: 异步超时包装
async fn timeout<T>(ms: i64, future: Future<T>) -> Result<T, TimeoutError> {
    future.with_timeout(ms)
}

// interval: 定时器间隔（类似 JS 的 setInterval）
struct Interval {
    deadline: f64,
    period: f64,
}

impl Interval {
    fn new(period_ms: i64) -> Interval {
        Interval {
            deadline: time_now_ms() + period_ms as f64,
            period: period_ms as f64,
        }
    }
    async fn tick(&mut self) {
        await sleep((self.deadline - time_now_ms()) as i64);
        self.deadline += self.period;
    }
}

// measure: 测量 async 操作耗时
async fn measure<T>(name: String, future: Future<T>) -> T {
    let start = time_now_ms();
    let result = await future;
    let elapsed = time_now_ms() - start;
    println("{} took {}ms", name, elapsed);
    result
}
```

### 9.7 async/io.th

```tenth
// 异步文件 IO

// 异步读取整个文件
async fn read_file_async(path: String) -> Vec<u8> {
    let fd = __tenth_io_open(path, O_RDONLY);
    let mut buf = Vec::new();
    loop {
        let chunk = await __tenth_io_read_async(fd, 4096);
        if chunk.is_empty() {
            break;
        }
        buf.extend(chunk);
    }
    __tenth_io_close(fd);
    buf
}

// 异步读取文本文件
async fn read_text_async(path: String) -> String {
    let bytes = await read_file_async(path);
    String::from_utf8(bytes)
}

// 异步写入文件
async fn write_file_async(path: String, data: Vec<u8>) {
    let fd = __tenth_io_open(path, O_WRONLY | O_CREAT);
    await __tenth_io_write_async(fd, data);
    __tenth_io_close(fd);
}

// 异步网络 IO（TCP）
async fn tcp_connect(host: String, port: i64) -> TcpStream {
    let fd = await __tenth_tcp_connect_async(host, port);
    TcpStream { fd }
}

struct TcpStream { fd: i64 }

impl TcpStream {
    async fn read(&mut self, buf: &mut Vec<u8>) -> i64 {
        await __tenth_io_read_async(self.fd, 4096)
    }
    async fn write(&mut self, data: Vec<u8>) {
        await __tenth_io_write_async(self.fd, data)
    }
    fn close(&mut self) {
        __tenth_io_close(self.fd)
    }
}
```

**native 后端**：用 mio 实现 IO 多路复用。
**WASM 后端**：用 host import（__tenth_io_*）实现，host 侧处理实际 IO。
### 9.8 async/sync.th

单线程协程模型下，锁用于协调 Task 之间的资源访问（不是防止数据竞争，而是防止逻辑竞争）。

```tenth
// 异步互斥锁
struct Mutex<T> {
    inner: MutexInner<T>,
}

impl<T> Mutex<T> {
    fn new(value: T) -> Mutex<T> {
        Mutex { inner: MutexInner::new(value) }
    }

    // lock: 异步获取锁
    async fn lock(&mut self) -> MutexGuard<T> {
        loop {
            match self.inner.try_lock() {
                Some(guard) => return guard,
                None => {
                    // 锁被占用，注册 Waker 等待
                    await self.inner.wait();
                }
            }
        }
    }

    // try_lock: 非阻塞获取锁
    fn try_lock(&mut self) -> Option<MutexGuard<T>> {
        self.inner.try_lock()
    }
}

// MutexGuard: RAII 锁守卫，Drop 时自动释放
struct MutexGuard<T> {
    inner: &mut MutexInner<T>,
}

impl<T> Drop for MutexGuard<T> {
    fn drop(&mut self) {
        self.inner.unlock();
    }
}

// 读写锁
struct RwLock<T> {
    inner: RwLockInner<T>,
}

impl<T> RwLock<T> {
    fn new(value: T) -> RwLock<T> {
        RwLock { inner: RwLockInner::new(value) }
    }
    async fn read(&mut self) -> RwLockReadGuard<T> {
        loop {
            match self.inner.try_read() {
                Some(guard) => return guard,
                None => await self.inner.wait_reader(),
            }
        }
    }
    async fn write(&mut self) -> RwLockWriteGuard<T> {
        loop {
            match self.inner.try_write() {
                Some(guard) => return guard,
                None => await self.inner.wait_writer(),
            }
        }
    }
}

// 信号量
struct Semaphore {
    permits: i64,
    waiters: Vec<Waker>,
}

impl Semaphore {
    fn new(permits: i64) -> Semaphore {
        Semaphore { permits, waiters: Vec::new() }
    }
    async fn acquire(&mut self) {
        if self.permits > 0 {
            self.permits -= 1;
        } else {
            await self.wait();
        }
    }
    fn release(&mut self) {
        self.permits += 1;
        if let Some(waker) = self.waiters.pop() {
            waker.wake();
        }
    }
}

// 事件（一次性通知）
struct Event {
    signaled: bool,
    waiters: Vec<Waker>,
}

impl Event {
    fn new() -> Event {
        Event { signaled: false, waiters: Vec::new() }
    }
    async fn wait(&mut self) {
        if !self.signaled {
            await self.wait_internal();
        }
    }
    fn signal(&mut self) {
        self.signaled = true;
        for w in self.waiters.drain(..) {
            w.wake();
        }
    }
    fn is_signaled(&self) -> bool {
        self.signaled
    }
    fn reset(&mut self) {
        self.signaled = false;
    }
}
```

**设计理由**：单线程下，这些锁不是防止数据竞争（单线程无数据竞争），而是协调 Task 之间的逻辑顺序（如生产者-消费者、读写互斥等）。

### 9.9 async/prelude.th

```tenth
// async 模块的 prelude，重导出常用项
// 用户可以通过 use std::async::prelude::* 导入

// 核心类型
//   Future<T>, Task<T>, Poll<T>, Waker
//   Sender<T>, Receiver<T>

// 核心函数
//   spawn, block_on, yield_now
//   channel, unbounded_channel
//   sleep, timeout, measure

// 组合子
//   Future::map, then, catch, with_timeout
//   Future::all, race, all_settled, any

// 锁
//   Mutex, RwLock, Semaphore, Event
```

### 9.10 与现有 std/runtime.th 的关系

现有 `tenth/std/runtime.th` 提供：
- `run_with_limit` —— 基于 step limit 的同步执行超时
- `run_with_timeout` —— 基于 step limit 的同步超时
- `with_step_limit` / `with_timeout_ms` —— native 函数

**区别**：
- `runtime.th` 是**同步**超时，通过 step limit 实现（VM 执行 N 步后强制停止）。
- `async/time.th` 是**异步**超时，通过事件循环 + 定时器实现（Task 挂起，定时器到期后唤醒）。

**关系**：
- `runtime.th` 适用于同步代码的超时控制（如防止死循环）。
- `async/time.th` 适用于异步代码的超时控制（如 IO 超时）。
- 两者正交，互不影响。

**不修改 runtime.th**：现有 `runtime.th` 的功能保持不变，async 模块是新功能。

### 9.11 prelude.th 索引同步

需要更新 `tenth/std/prelude.th`，添加 async 模块的索引：

```tenth
// ── Async (std::async::*) ──
//   std::async::future::* — Future trait, map/then/all/race/all_settled/any
//   std::async::task::* — spawn, block_on, yield_now, abort, is_done, Task<T>
//   std::async::channel::* — channel, unbounded_channel, Sender<T>, Receiver<T>
//   std::async::select::* — select 语法（编译器内置）
//   std::async::time::* — sleep, timeout, Interval, measure
//   std::async::io::* — read_file_async, write_file_async, tcp_connect, TcpStream
//   std::async::sync::* — Mutex, RwLock, Semaphore, Event
//   std::async::prelude::* — 重导出所有常用项
```

**注意**：这只是 prelude.th 的注释更新，实际导入需要用户显式 `use std::async::prelude::*`。

### 9.12 标准库设计决策汇总

| 模块 | 内容 | 备注 |
|------|------|------|
| future.th | Future trait + 组合子 | map/then/all/race/... |
| task.th | spawn/block_on/abort | Task<T> 类型 |
| channel.th | MPMC 通道 | Sender/Receiver |
| select.th | select 运行时支持 | 语法在编译器 |
| time.th | sleep/timeout/interval | 异步定时器 |
| io.th | 异步文件/网络 IO | native: mio, WASM: host |
| sync.th | 异步锁 | Mutex/RwLock/Semaphore/Event |
| prelude.th | 重导出 | 便于导入 |
---

## 10. 分阶段实施计划

### 10.1 阶段总览

| 阶段 | 内容 | 优先级 | 风险 | 预计工期 |
|------|------|--------|------|---------|
| 1 | 基础设施（关键字+类型+HIR+lower+VM 调度器+测试） | P0 | 高 | 大 |
| 2 | 标准库 async/ 模块（future/task/channel/select/time） | P0 | 中 | 中 |
| 3 | WASM 后端适配 + host imports | P1 | 中 | 中 |
| 4 | JIT 后端适配 | P2 | 低 | 小 |
| 5 | tenthc 双侧同步（语法解析 + 类型标注） | P0 | 中 | 中 |
| 6 | tenthc 状态机 lower 同步 | P2 | 高 | 大 |
| 7 | 异步 IO 集成（文件/网络，native + WASM） | P1 | 中 | 中 |

### 10.2 阶段 1：基础设施

**目标**：让 Tenth 支持 async/await 的基本语法和执行。

**任务清单**：
1. lexer 新增 `async` / `await` 关键字（启用 `spawn`）。
2. parser 解析 `async fn` / `await` / `spawn` 语法。
3. AST 新增对应节点。
4. HIR 新增 `HirAsyncFn` / `HirAwait` / `HirSpawn` 节点。
5. 类型系统新增 `Type::Future(Box<Type>)`。
6. 状态机 lower pass（直线型 async fn）。
7. VM 新增字节码指令（Spawn/Await/BlockOn 等）。
8. VM 调度器实现（Ready Queue + 简单事件循环）。
9. 基础测试：直线型 async fn + await + spawn。

**验证策略**：
- `cargo test --manifest-path tenth/Cargo.toml` 全绿。
- 新增 async 基础测试通过。
- 现有测试不回归。

**风险控制**：
- 状态机 lower 是核心难点，先只支持直线型（无 if/while/for 中的 await）。
- VM 调度器先实现最简版本（无 IO，只有 Ready Queue + sleep）。
- 每个功能点配测试。

**交付物**：
- `tenth/src/lexer/token.rs` 新增关键字。
- `tenth/src/parser/` 新增解析。
- `tenth/src/hir/` 新增 HIR 节点 + lower pass。
- `tenth/src/runtime/vm.rs` 新增字节码 + 调度器。
- `tenth/tests/async_basic_test.rs` 基础测试。

### 10.3 阶段 2：标准库 async/ 模块

**目标**：实现完整的 async 标准库。

**任务清单**：
1. 新建 `tenth/std/async/` 目录。
2. 实现 `future.th`（Future trait + 组合子）。
3. 实现 `task.th`（spawn/abort/block_on）。
4. 实现 `channel.th`（MPMC 通道）。
5. 实现 `select.th`（select 运行时支持）。
6. 实现 `time.th`（sleep/timeout/interval）。
7. 实现 `sync.th`（Mutex/RwLock/Semaphore/Event）。
8. 实现 `prelude.th`（重导出）。
9. 更新 `tenth/std/prelude.th` 索引。
10. 标准库测试。

**验证策略**：
- `cargo test --manifest-path tenth/Cargo.toml -- stdlib` 全绿。
- 新增 async 标准库测试通过。

**风险控制**：
- select 语法需要编译器支持（上下文关键字）。
- 通道实现需要调度器支持（Waker 注册）。

**交付物**：
- `tenth/std/async/*.th` 标准库模块。
- `tenth/std/async/test_*.th` 标准库测试。

### 10.4 阶段 3：WASM 后端适配

**目标**：让 async/await 在 WASM 后端可用。

**任务清单**：
1. `compile/wasm/` 新增状态机编译逻辑。
2. 定义 host imports 接口。
3. host 侧（Rust）实现调度器 + IO + 定时器。
4. WASM 模块导出 poll 函数。
5. wasmi 集成测试。

**验证策略**：
- 编译含 async 的 Tenth 代码到 WASM。
- wasmi 执行 WASM 模块，async 代码正确运行。
- 路径 C 不破坏。

**风险控制**：
- host imports 接口要稳定（后续阶段不改动）。
- WASM 内存布局要高效（状态机字段紧凑排列）。

**交付物**：
- `tenth/src/compile/wasm/` 新增 async 编译。
- `tenth/src/compile/wasm/host.rs` host imports 实现。
- WASM async 测试。

### 10.5 阶段 4：JIT 后端适配

**目标**：让 async/await 在 JIT 后端可用。

**任务清单**：
1. `compile/jit/translator.rs` 新增 async 翻译。
2. 状态机 enum → native 结构体。
3. poll 方法 → native 函数。
4. spawn/await → 调度器调用。
5. JIT 测试。

**验证策略**：
- JIT 执行 async 代码，结果与 VM 一致。
- JIT 性能不低于 VM。

**风险控制**：
- JIT 复杂度高，可延后。
- 与 VM 共享调度器，降低实现成本。

**交付物**：
- `tenth/src/compile/jit/translator.rs` 新增 async 翻译。
- JIT async 测试。

### 10.6 阶段 5：tenthc 双侧同步（语法解析 + 类型标注）

**目标**：tenthc 能解析和类型检查 async 代码。

**任务清单**：
1. `tenthc/lexer/token.th` 新增 async/await 关键字。
2. `tenthc/lexer/lexer.th` 识别新 token。
3. `tenthc/parser/parser.th` 解析 async 语法。
4. `tenthc/parser/ast.th` 新增 AST 节点。
5. `tenthc/hir/hir.th` 新增 HIR 节点。
6. `tenthc/hir/types.th` 新增 Type::Future。
7. `tenthc/hir/lower.th` 类型检查（不要求状态机 lower）。
8. 自举验证。

**验证策略**：
- `cargo run --release --manifest-path tenth/Cargo.toml -- run tenthc/main.th` 自举成功。
- tenthc 能解析含 async 的 Tenth 代码（不报语法错误）。
- 路径 B 可用（tenthc 解析 + Rust lower）。

**风险控制**：
- tenthc HIR 数据结构与 Rust 严格对齐。
- tenthc 自举性能不下降。

**交付物**：
- `tenthc/lexer/`、`tenthc/parser/`、`tenthc/hir/` 同步更新。
- 自举测试通过。

### 10.7 阶段 6：tenthc 状态机 lower 同步

**目标**：tenthc 能 lower async fn 为状态机。

**任务清单**：
1. 移植状态机 lower pass 到 tenthc。
2. 先支持直线型 async fn。
3. 再支持 if 中的 await。
4. 最后支持 loop 中的 await。
5. 路径 C 全 WASM 闭环验证。

**验证策略**：
- 全 WASM 闭环：tenthc 编译含 async 的 Tenth 代码到 WASM，wasmi 执行。
- 路径 C 不破坏。

**风险控制** [风险:高]：
- 状态机 lower 是复杂 pass，分阶段实现。
- 每阶段充分测试。
- 如果 tenthc lower 不稳定，可延后（路径 C 暂不支持 async）。

**交付物**：
- `tenthc/hir/lower.th` 状态机 lower pass。
- 路径 C async 测试。

### 10.8 阶段 7：异步 IO 集成

**目标**：实现完整的异步 IO（文件 + 网络）。

**任务清单**：
1. `tenth/std/async/io.th` 完整实现。
2. native 后端：用 mio 实现 IO 多路复用。
3. WASM 后端：host imports 实现 IO。
4. 异步文件读写测试。
5. 异步 TCP 网络测试。

**验证策略**：
- 异步文件读写：并发读取多个文件。
- 异步 TCP：echo 服务器 + 客户端。
- 性能：并发 IO 吞吐量。

**风险控制**：
- mio 依赖跨平台兼容性。
- WASM IO 受 host 限制。

**交付物**：
- `tenth/std/async/io.th` 完整实现。
- `tenth/src/runtime/io_native.rs` native IO 实现。
- 异步 IO 测试。

### 10.9 阶段依赖关系

```
阶段 1（基础设施）
  ↓
阶段 2（标准库）  ←─ 阶段 5（tenthc 解析）
  ↓                       ↓
阶段 3（WASM）        阶段 6（tenthc lower）
  ↓                       ↓
阶段 4（JIT）          阶段 7（异步 IO）
```

**关键路径**：阶段 1 → 阶段 2 → 阶段 3 → 阶段 7。

**可并行**：
- 阶段 5 可与阶段 2-3 并行（tenthc 同步独立）。
- 阶段 4 可与阶段 5-6 并行（JIT 独立）。
- 阶段 6 必须在阶段 5 之后。

### 10.10 分阶段实施决策汇总

| 阶段 | 内容 | 关键依赖 | 风险 |
|------|------|---------|------|
| 1 | 基础设施 | 无 | 高（状态机 lower） |
| 2 | 标准库 | 阶段 1 | 中 |
| 3 | WASM | 阶段 1 | 中 |
| 4 | JIT | 阶段 1 | 低 |
| 5 | tenthc 解析 | 阶段 1 | 中 |
| 6 | tenthc lower | 阶段 5 | 高 |
| 7 | 异步 IO | 阶段 2, 3 | 中 |

---

## 11. 风险评估

> 风险不是避免出来的，是识别出来并设计缓解策略后才能承担的。
> 本章识别 async/await 特性引入的各类风险，并为每类风险设计可操作的缓解策略与回退路径。

### 11.1 风险分类总览

按风险来源与影响范围，将本特性涉及的风险分为五大类：

| 类别 | 风险数 | 主要影响 | 总体等级 |
|------|--------|---------|---------|
| 技术风险 | 6 | 实现可行性、性能、兼容性 | 中 |
| 设计风险 | 5 | API 稳定性、语义清晰度 | 中 |
| 自举风险 | 4 | 三路径一致性、tenthc 复杂度 | 高 |
| 工程风险 | 4 | 工期、协作、回归 | 中 |
| 生态风险 | 3 | 与现有 stdlib 冲突、用户迁移成本 | 低 |

总体风险等级为**中-高**：技术路径已被 Rust/Python/JS 充分验证，但 Tenth 的自举约束和 WASM-only 后端约束带来了独特挑战。下文逐项分析。

### 11.2 技术风险

#### R-T1：状态机 lower pass 复杂度爆炸

**风险描述**：将 async fn body 按 await 点切分为状态机，需要正确处理：
- 跨 await 点的局部变量生命周期（必须提升为状态机字段）
- 控制流（if/for/while/loop/match）与 await 的嵌套组合
- 闭包捕获与 await 的交互
- 早期返回（early return）与状态机终止状态的映射

复杂度来源主要是**控制流与 await 的组合爆炸**：一个含 N 个 await 点的 fn body，若有 M 条控制流路径，最坏情况下状态数可达 O(M·N)。

**触发概率**：中。日常代码 await 数量少（1-3 个），复杂度可控；但并发库的 utility 函数可能含 5+ await 点。

**影响**：
- 编译器实现复杂度高（lower pass 可能上千行）
- 编译时间增长（每个 async fn 都要走 lower）
- 调试困难（生成的状态机 enum 难以人工阅读）

**缓解策略**：
1. **限制初始版本的复杂度**：第一阶段只支持"线性 await"（await 不在循环内），后续阶段逐步放开
2. **借用 Rust 的经验**：Rust 的 `async/await` 已经走过这条路，状态机生成算法有公开实现可参考
3. **生成可读的 enum 变体名**：`State0_AfterFirstAwait`、`State1_AfterSecondAwait`，便于调试
4. **添加编译时复杂度检查**：若状态数超过阈值（如 32），发出警告提示用户重构
5. **保留 fallback**：若 lower 失败，回退到"显式 Future trait"模式（用户手写 poll）

**回退路径**：若状态机 lower 始终不稳定，可降级为"只支持显式 Future"——用户用 `async fn` 写代码会失败，但手写 `impl Future` 的 poll 方法可用。这会显著降低易用性，但不破坏语言核心。

#### R-T2：wasmi 异步执行能力不足

**风险描述**：方案依赖 wasmi 的"协作式调度 + host imports + 状态机 lower"组合实现异步。具体依赖：
- wasmi 支持 host imports（已确认）
- wasmi 在 host function 中能存储跨调用的状态（已确认，通过 `Func` 系列 API）
- wasmi 不支持 stack-switching，但状态机方案不需要它（已确认）

风险在于：wasmi 在 host function 内部是否能**主动让出控制权**给其它 task？答案是不能——wasmi 是同步执行的，host function 一旦被调用，必须同步返回。这意味着异步 IO 的"挂起"必须通过"返回 Pending + 注册 waker + 等待下一次 poll"实现，而不是"在 host function 内挂起"。

**触发概率**：低（设计已规避）。但实现时若误用 wasmi API（如试图在 host function 内挂起），会导致死锁或 panic。

**影响**：
- 实现错误会导致 WASM 路径死锁
- 性能可能下降（频繁的 host↔wasm 切换）

**缓解策略**：
1. **明确约束**：在文档和实现中反复强调"wasmi host function 必须同步返回"
2. **设计 HostFuture 抽象**：所有需要挂起的 host 调用都返回 `Poll::Pending` + 注册 waker，由 host 端的 event loop 在事件到达时唤醒
3. **写专门的 WASM 异步测试**：覆盖 sleep/IO/channel/select 在 WASM 后端的正确性
4. **CI 跑 WASM 路径**：每次提交都跑路径 C 的自举验证

**回退路径**：若 wasmi 异步路径不可行，可降级为"VM/解释器后端支持 async，WASM 后端只支持同步"——损失 WASM 异步能力，但不破坏语言语义（async fn 在 WASM 后端可以编译，只是 block_on 会失败）。

#### R-T3：性能回归

**风险描述**：状态机 lower 引入额外开销：
- 每个 async fn 变成 enum + poll 方法，调用开销增加
- 状态机字段的堆分配（Future 通常 boxed）
- Waker 的 Rc<RefCell<>> 引用计数开销
- 调度器轮询的开销

对 IO 密集场景，这些开销可被并发收益抵消；但**对计算密集场景**（如 tensor 运算），如果误用 async 包装同步计算，会带来纯损失。

**触发概率**：高（用户很可能误用）。Tenth 的核心场景是 AI 计算，async 不应该被用在 tensor 运算的热路径上。

**影响**：
- 用户误用导致性能下降 2-10 倍
- 调试困难（性能问题难以定位）

**缓解策略**：
1. **明确文档指导**：async 用于 IO 和并发协调，不用于计算热路径
2. **lint 警告**：检测 `async fn` body 中是否含 tensor 运算，若有则警告
3. **提供 `spawn_blocking` 语义**：对计算密集任务，提供"在独立线程执行"的 escape hatch（但这需要多线程运行时，第六阶段才考虑）
4. **benchmark 对比**：每个版本都跑 async vs sync 的性能对比，监控回归

**回退路径**：若性能问题严重，可在 VM 后端提供"同步 fast path"——当 Future 立即就绪时跳过调度器，直接返回值。这是优化，不需要回退设计。

#### R-T4：调度器死锁/活锁

**风险描述**：协作式调度器依赖任务主动 yield（在 await 点）。若用户写出"无 await 的死循环"，调度器无法切换到其它任务，导致死锁。

此外，channel/select 的实现可能引入活锁：多个任务互相唤醒但不实际推进。

**触发概率**：中（用户误写死循环很常见）。

**影响**：
- 程序卡死，难以调试
- 用户对 async 模型失去信心

**缓解策略**：
1. **文档强调**：async fn 必须包含 await 点，否则失去并发意义
2. **lint 警告**：检测无 await 的 async fn
3. **提供 `yield_now()`**：让用户在长循环中主动让出控制权
4. **调试器支持**：在调试模式下，调度器可记录每个 task 的运行时长，超过阈值时输出警告
5. **超时机制**：网络 IO 默认带超时（参考 std/runtime.th 的 `run_with_timeout`）

**回退路径**：若死锁问题严重，可在第二阶段引入"抢占式调度"——调度器在 task 运行超过 N 条指令后强制切换。但这会破坏"协作式"语义，是最后手段。

#### R-T5：Future 内存泄漏

**风险描述**：Future 是状态机，若被 drop 时正处于 Pending 状态，需要正确释放：
- 状态机字段（局部变量提升的）
- Waker 注册（需要从调度器注销）
- Channel 端的引用（需要通知对端）

Rust 通过 RAII 自动处理，但 Tenth 的 GC 是否能正确处理循环引用（Future ↔ Waker ↔ Scheduler）是风险点。

**触发概率**：中。Tenth 当前使用引用计数 GC，对循环引用处理可能不完整。

**影响**：
- 内存泄漏（长期运行的 server 场景）
- Waker 残留导致错误唤醒

**缓解策略**：
1. **审计 GC**：在实现前确认 Tenth GC 对循环引用的处理（如使用 Weak 引用打破环）
2. **显式 drop**：Future 完成时显式调用 `drop()`，释放资源
3. **Waker 使用 Weak 引用**：Waker 持有 Scheduler 的 Weak 引用，避免强环
4. **添加 leak test**：在测试中反复 spawn/cancel task，监控内存增长

**回退路径**：若 GC 问题无法解决，可限制 async fn 的生命周期——要求所有 task 必须在 `block_on` 作用域内完成，不允许泄漏到外层。这是约束，但能避免泄漏。

#### R-T6：与现有 autograd/tensor 集成冲突

**风险描述**：Tenth 的核心护城河是 autograd shape check 和 tensor 运算。async 引入后可能出现：
- async fn 中调用 tensor 运算，但 TapeOp 的反向传播需要在同步上下文执行
- Future 持有 tensor 引用，但 tensor 在另一个 task 中被修改，导致 shape 不一致
- async fn 的状态机捕获 tensor，但 autograd 期望 tensor 在连续作用域内

**触发概率**：中-高。这是 async 与 AI 计算的固有张力。

**影响**：
- 破坏 autograd 正确性
- shape check 误报或漏报

**缓解策略**：
1. **明确边界**：async 用于 IO 和并发协调，tensor 计算保持同步
2. **lint 警告**：检测 async fn 中是否直接调用 tensor 方法，建议改用 `spawn_blocking`
3. **类型约束**：Future<T> 要求 T: NotTensor（或类似约束），阻止 tensor 直接作为 Future 输出
4. **测试覆盖**：在 autograd 测试套件中加入 async 场景，确保不回归

**回退路径**：若集成问题严重，可禁止 async fn 中使用 tensor——这是强约束，但保护核心护城河。

### 11.3 设计风险

#### R-D1：API 稳定性

**风险描述**：第一版 async API 可能在后续版本中需要调整：
- `Future` trait 的方法签名（poll、map、then）
- `select` 语法（是否支持 default 分支、是否支持 guard）
- `spawn` 的返回类型（Task<T> 还是 JoinHandle<T>）
- Channel 的容量语义（有界 vs 无界）

API 一旦发布，向后兼容性会成为负担。

**触发概率**：高。Rust 的 async API 经历了多次调整（async/await 稳定前用 futures crate 的 0.1/0.3 分裂）。

**影响**：
- 用户代码需要迁移
- 标准库需要维护多个版本

**缓解策略**：
1. **第一阶段标记 experimental**：API 文档明确标注"实验性，可能变更"
2. **最小化初始 API**：只稳定最核心的 `async/await/spawn/block_on`，其它（select、channel）保留实验
3. **提供 deprecation 机制**：在 tenth/std/prelude.th 中维护版本标记
4. **借鉴成熟方案**：API 设计参考 Rust std + tokio + asyncio，减少试错

**回退路径**：通过版本化（`async::v1`、`async::v2`）允许新旧 API 共存，给用户迁移时间。

#### R-D2：select 语义歧义

**风险描述**：select 多路复用的语义需要明确：
- 多个分支同时就绪时，选哪个？（随机 / 优先级 / 第一个）
- 是否支持 guard（带条件的分支）？
- 是否支持 default 分支（无就绪时执行）？
- 分支中的变量绑定（`v = self => ...`）作用域如何？

Go 的 select 是"随机选择就绪分支"，Rust 的 `tokio::select!` 是"按顺序检查，第一个就绪的执行"。

**触发概率**：中。语义选择没有对错，但需要明确。

**影响**：
- 用户对 select 行为的预期不一致
- 调试困难

**缓解策略**：
1. **明确选择 Go 风格**：随机选择就绪分支，避免"顺序依赖"的隐含语义
2. **不支持 guard**：第一阶段不支持 `case x if cond =>`，简化语义
3. **支持 default**：`default => ...` 分支，用于非阻塞场景
4. **文档强调**：select 的随机性，建议不要依赖特定顺序

**回退路径**：若用户反馈强烈，可在未来版本中支持 guard 和优先级——这是扩展，不破坏向后兼容。

#### R-D3：async fn 与同步 fn 的互调

**风险描述**：常见痛点：
- async fn 调用同步 fn：简单，直接调用
- 同步 fn 调用 async fn：需要 `block_on`，但 `block_on` 在 async 上下文中会死锁
- async fn 中的闭包是否自动是 async 闭包？
- 高阶函数（map/filter）是否需要 async 版本？

**触发概率**：高。这是所有 async 语言的共同痛点。

**影响**：
- 用户困惑，写出死锁代码
- 需要双倍 API（sync 版 + async 版）

**缓解策略**：
1. **block_on 检测**：在 async 上下文中调用 block_on 时报错或警告
2. **提供 async 闭包语法**：`async |x| { ... }`
3. **标准库提供 async 版本的高阶函数**：`future_map`、`future_filter` 等
4. **文档指导**：明确"async 边界"概念，建议在边界处统一转换

**回退路径**：若 async 闭包实现复杂，第一阶段可只支持命名 async fn，不支持匿名 async 闭包。

#### R-D4：错误处理与 Future 的交互

**风险描述**：Tenth 的错误处理用 `try` 关键字和 `Result<T, E>` 类型。async fn 中的错误处理需要明确：
- `try` 在 async fn 中的行为（传播错误到 caller Future）
- `Result<Future<T>, E>` vs `Future<Result<T, E>>` 的语义差异
- async fn 中 panic 的行为（应该 cancel 当前 task，但不应崩溃整个运行时）

**触发概率**：中。

**影响**：
- 错误处理不一致导致 bug
- panic 行为不明导致运行时不稳定

**缓解策略**：
1. **明确 Future<Result<T, E>> 为标准模式**：async fn 返回 Result 时，整个 Future 完成后给出 Result
2. **try 在 async fn 中正常工作**：传播错误到 caller，等价于 `return Err(e)`
3. **panic 隔离**：task 内 panic 被 scheduler 捕获，标记 task 为 Failed，不影响其它 task
4. **提供 `catch_unwind` 语义**：让 caller 可以处理 task 的 panic

**回退路径**：若 panic 隔离实现复杂，第一阶段可让 panic 直接崩溃运行时——这是约束，但简化实现。

#### R-D5：与现有 shard/node 关键字的语义冲突

**风险描述**：token.rs 已预留 `shard`（分布式分片）和 `node`（计算节点）关键字。async/await 是单机并发，与分布式语义有重叠：
- 一个 `node` 上的多个 task 是否用 async？
- `shard` 间的通信是否复用 channel？

**触发概率**：低（分布式特性尚未设计）。

**影响**：
- 未来设计分布式时需要明确边界
- 用户可能混淆

**缓解策略**：
1. **明确分工**：async/await 是单机并发，shard/node 是分布式
2. **保留扩展空间**：channel API 设计为"可本地可远程"，未来扩展为分布式 channel 时不破坏 API
3. **文档区分**：在 async 文档中明确"本特性不涉及分布式"

**回退路径**：未来若分布式需要不同的并发原语，可引入新的关键字——这是扩展，不冲突。

### 11.4 自举风险

#### R-S1：tenthc 实现复杂度

**风险描述**：tenthc 是 Tenth 自举编译器，需要同步实现 async/await 的解析、类型、lower。但 tenthc 当前能力有限：
- 类型推断可能不支持 Future 的高级 trait
- lower pass 在 tenthc 中实现可能上千行
- WASM 后端（tenthc/compile/wasm.th）需要扩展异步指令

**触发概率**：高。tenthc 的能力增长跟不上主编译器是常见问题。

**影响**：
- 自举路径 B/C 可能延迟
- 双侧不一致导致语义分歧

**缓解策略**：
1. **分层同步策略**：如第八章所述，P0 解析 → P1 类型 → P2 lower，允许 tenthc 暂时不支持 lower
2. **tenthc 第一阶段只支持解析**：让 tenthc 能解析 `async fn`/`await`，但不实际编译，输出"未实现"错误
3. **主编译器提供 fallback**：若 tenthc 不支持 async，用户可用主编译器编译，不破坏自举路径 A
4. **逐步迁移**：随着 tenthc 能力增强，逐步补齐类型和 lower

**回退路径**：若 tenthc 始终无法支持 async lower，可让 tenthc 只负责"不含 async 的代码"，含 async 的代码由主编译器处理。这破坏自举完整性，但不破坏主路径。

#### R-S2：三路径一致性

**风险描述**：三条自举路径（A: Rust全栈 / B: Tenth前端+Rust后端 / C: 全WASM闭环）必须语义一致。async 引入后：
- 路径 A：Rust 实现的状态机 lower + VM 执行
- 路径 B：tenthc 解析 + Rust lower + VM 执行
- 路径 C：tenthc 解析 + tenthc lower + WASM 执行

三条路径的 lower 算法必须产出**语义等价**的状态机。

**触发概率**：中。Rust 和 tenthc 的 lower 实现独立，容易出现细微差异。

**影响**：
- 同一代码在不同路径下行为不同
- 测试矩阵爆炸

**缓解策略**：
1. **共享 lower 规范**：在文档中明确 lower 算法的每一步，作为两侧实现的"契约"
2. **差分测试**：对同一 async fn，分别用 A/B/C 路径编译执行，对比结果
3. **共享测试套件**：async 测试用例同时跑三条路径
4. **CI 强制三路径验证**：每次提交都跑 selfhost 测试

**回退路径**：若三路径一致性难以保证，可暂时只支持路径 A，路径 B/C 标记"async 实验性，可能不一致"。但这破坏自举约束，是最后手段。

#### R-S3：WASM 后端的 host imports 依赖

**风险描述**：路径 C（全 WASM 闭环）需要 WASM 模块导入 host 函数实现异步 IO。但路径 C 的"host"是 wasmi 引擎本身——这意味着 Tenth 的 Rust 运行时需要在 wasmi 外层提供 host 函数。

风险：host imports 的接口设计可能不稳定，且 wasmi 版本升级可能破坏接口。

**触发概率**：中。wasmi 是活跃项目，API 可能变化。

**影响**：
- 路径 C 维护成本高
- wasmi 升级导致路径 C 失败

**缓解策略**：
1. **最小化 host imports**：只导入最必要的（`__tenth_poll_io`、`__tenth_register_timer`、`__tenth_current_time_ms`）
2. **抽象 host 接口**：在 tenth/std/async/io.th 中抽象 host 调用，便于切换实现
3. **锁定 wasmi 版本**：在 Cargo.toml 中锁定 wasmi 版本，升级前充分测试
4. **提供 mock host**：测试时用 mock host 函数，不依赖真实 wasmi

**回退路径**：若 wasmi 升级破坏路径 C，可暂时禁用路径 C 的 async 支持——同步代码仍走路径 C，async 代码在路径 C 下报错"不支持"。

#### R-S4：自举性能下降

**风险描述**：tenthc 自身是 Tenth 代码，若 tenthc 中使用了 async（如并发解析多个文件），会引入状态机 lower 开销。当前自举性能 ~0.2s，超过 1s 算失败。

**触发概率**：低。tenthc 当前是单线程同步编译，不太可能引入 async。

**影响**：
- 自举时间从 0.2s 增长到 1s+
- 开发体验下降

**缓解策略**：
1. **tenthc 不使用 async**：tenthc 保持同步实现，不引入 async 代码
2. **监控自举性能**：每次提交跑 `cargo run --release -- run tenthc/main.th`，监控时间
3. **若 tenthc 需要 async，单独 benchmark**：确保不回归

**回退路径**：无（这是硬约束，必须保持）。

### 11.5 工程风险

#### R-E1：工期不确定性

**风险描述**：本特性分 7 个阶段，每阶段 2-4 周，总工期 4-6 个月。实际工期可能因：
- 状态机 lower 的复杂度超预期
- wasmi 异步集成的调试
- 测试覆盖的完整性

而大幅延长。

**触发概率**：高。复杂特性工期不准是常态。

**影响**：
- 占用其它特性的开发资源
- 用户期望落空

**缓解策略**：
1. **分阶段交付**：每阶段产出可用的子特性，不依赖完整完成
2. **第一阶段优先**：确保阶段 1（基础语法 + VM 后端）尽快可用，提供基础价值
3. **定期评估**：每阶段结束时评估剩余工期，调整计划
4. **预留缓冲**：每阶段预留 20% 缓冲时间

**回退路径**：若工期严重超期，可缩减范围——只支持 VM 后端，不支持 WASM/JIT 后端；只支持基础语法，不支持 select/channel。

#### R-E2：测试覆盖不足

**风险描述**：async 特性的测试矩阵巨大：
- 三条自举路径 × 多个后端（VM/解释器/WASM/JIT）
- 多种并发模式（spawn/await/select/channel）
- 多种错误场景（panic/cancel/timeout）

测试覆盖不足会导致回归 bug。

**触发概率**：中-高。测试矩阵爆炸是常见问题。

**影响**：
- 回归 bug 频发
- 用户信心下降

**缓解策略**：
1. **分层测试**：单元测试（state machine lower）+ 集成测试（async fn 端到端）+ 差分测试（三路径对比）
2. **property-based testing**：对状态机 lower 用随机生成的 async fn 测试
3. **回归测试套件**：每发现一个 bug，添加测试用例
4. **CI 强制全测试**：每次提交跑完整测试套件

**回退路径**：若测试覆盖不足，可限制 async 的使用场景——只在标准库内部使用，不暴露给用户。这是极端保守，但保护稳定性。

#### R-E3：跨部门协作成本

**风险描述**：本特性涉及编译器部、运行时部、标准库部、测试部、文档部。跨部门协作需要：
- 接口定义清晰
- 黑板协议有效
- 总师协调

若协作不畅，会导致接口不一致、重复工作、遗漏。

**触发概率**：中。

**影响**：
- 开发效率下降
- 接口不一致导致集成困难

**缓解策略**：
1. **总师先行出方案**：本设计文档作为"契约"，各部门按文档实现
2. **黑板协议**：使用 `.trae/tmp/task_board.md` 进行跨部门沟通
3. **接口先行**：先定义 HIR 数据结构、VM 指令、stdlib API，再分头实现
4. **集成测试**：每阶段结束时进行跨部门集成测试

**回退路径**：若协作困难，总师可调整任务分配，减少并行度——串行执行虽然慢，但可控。

#### R-E4：文档与代码不同步

**风险描述**：设计文档（本文件）与实际实现可能脱节：
- 设计假设不成立
- 实现细节与设计不符
- API 变更未同步到语言参考手册

**触发概率**：高。文档与代码同步是永恒的工程难题。

**影响**：
- 用户按文档写代码但运行失败
- 维护困难

**缓解策略**：
1. **设计文档标注版本**：本文件标注 v1.0，实现后更新到 v1.1
2. **语言参考手册同步**：实现完成后，更新 docs/语言参考手册.md 的 async 章节
3. **MEMO.md 记录变更**：每次设计调整记录到 MEMO.md
4. **代码注释引用设计**：实现代码中注释引用设计文档的章节号

**回退路径**：若文档严重脱节，可标记设计文档为"历史参考"，以代码和语言参考手册为准。

### 11.6 生态风险

#### R-C1：与现有 std/runtime.th 的冲突

**风险描述**：`tenth/std/runtime.th` 已提供 `run_with_limit`/`run_with_timeout`（同步超时）。新引入的 async time 模块（sleep/timeout）可能与之重叠，造成用户困惑。

**触发概率**：低。

**影响**：
- 用户不知道用哪个
- 维护两套 API

**缓解策略**：
1. **明确分工**：runtime.th 是同步超时（阻塞当前线程），async/time.th 是异步超时（挂起当前 task）
2. **文档对比**：在两个模块的文档中互相引用，说明区别
3. **长期计划**：未来可能将 runtime.th 的超时标记为 deprecated，统一用 async

**回退路径**：保持两套 API 共存，不强制迁移。

#### R-C2：用户迁移成本

**风险描述**：现有 Tenth 代码都是同步的。引入 async 后，用户可能需要：
- 重写 IO 密集代码为 async
- 学习新的并发模型
- 调试异步代码

**触发概率**：中。

**影响**：
- 用户抵触新特性
- 学习曲线陡峭

**缓解策略**：
1. **async 是可选的**：不强制用户使用，同步代码继续工作
2. **提供迁移指南**：文档中提供"如何把同步代码改成 async"的教程
3. **渐进式采用**：用户可在关键路径局部使用 async，不需要整体重写
4. **示例丰富**：标准库和示例代码中提供大量 async 用例

**回退路径**：无（async 是增量特性，不破坏同步代码）。

#### R-C3：与 AI 计算生态的兼容性

**风险描述**：Tenth 的核心场景是 AI 计算。async 引入后，需要考虑：
- 与 PyTorch/JAX 等框架的互操作（通过 FFI）是否支持 async？
- tensor 计算是否需要 async 接口？
- autograd 反向传播是否需要 async？

**触发概率**：低。当前 Tenth 的 AI 计算是同步的，async 是为 IO 和并发协调引入的。

**影响**：
- 用户可能误用 async 包装 AI 计算
- 与外部框架的互操作可能不支持 async

**缓解策略**：
1. **明确边界**：async 用于 IO 和并发协调，AI 计算保持同步
2. **FFI 不支持 async**：第一阶段 async 不跨越 FFI 边界
3. **未来扩展**：若 AI 框架支持 async（如 JAX 的 async dispatch），再考虑集成

**回退路径**：无（async 是可选的，不影响同步 AI 计算）。

### 11.7 风险矩阵汇总

| 风险 ID | 类别 | 触发概率 | 影响 | 等级 | 缓解策略 |
|---------|------|---------|------|------|---------|
| R-T1 | 技术 | 中 | 高 | 高 | 限制初始复杂度、借鉴 Rust、可读 enum |
| R-T2 | 技术 | 低 | 高 | 中 | 明确约束、HostFuture 抽象、WASM 测试 |
| R-T3 | 技术 | 高 | 中 | 中 | 文档指导、lint 警告、benchmark |
| R-T4 | 技术 | 中 | 中 | 中 | 文档、lint、yield_now、调试器支持 |
| R-T5 | 技术 | 中 | 中 | 中 | 审计 GC、Weak 引用、leak test |
| R-T6 | 技术 | 中-高 | 高 | 高 | 明确边界、lint、类型约束 |
| R-D1 | 设计 | 高 | 中 | 中 | experimental 标记、最小化初始 API |
| R-D2 | 设计 | 中 | 低 | 低 | Go 风格、不支持 guard |
| R-D3 | 设计 | 高 | 中 | 中 | block_on 检测、async 闭包 |
| R-D4 | 设计 | 中 | 中 | 中 | Future<Result>、panic 隔离 |
| R-D5 | 设计 | 低 | 低 | 低 | 明确分工、保留扩展空间 |
| R-S1 | 自举 | 高 | 高 | 高 | 分层同步、tenthc 只解析 |
| R-S2 | 自举 | 中 | 高 | 高 | 共享规范、差分测试 |
| R-S3 | 自举 | 中 | 中 | 中 | 最小化 host imports、锁定 wasmi |
| R-S4 | 自举 | 低 | 高 | 中 | tenthc 不用 async |
| R-E1 | 工程 | 高 | 中 | 中 | 分阶段交付、定期评估 |
| R-E2 | 工程 | 中-高 | 中 | 中 | 分层测试、property testing |
| R-E3 | 工程 | 中 | 中 | 中 | 总师协调、接口先行 |
| R-E4 | 工程 | 高 | 中 | 中 | 文档版本化、MEMO 记录 |
| R-C1 | 生态 | 低 | 低 | 低 | 明确分工、文档对比 |
| R-C2 | 生态 | 中 | 低 | 低 | 可选特性、迁移指南 |
| R-C3 | 生态 | 低 | 低 | 低 | 明确边界、FFI 不支持 |

### 11.8 总体回退策略

若整体特性失败或严重延期，按以下顺序回退：

1. **第一级回退**：只支持 VM 后端，不支持 WASM/JIT 后端
   - 影响：WASM 路径不能用 async，但不破坏同步代码
   - 触发条件：wasmi 异步集成失败

2. **第二级回退**：只支持基础语法（async/await/spawn/block_on），不支持 select/channel
   - 影响：并发能力受限，但有基础 async
   - 触发条件：select/channel 实现复杂度超预期

3. **第三级回退**：只支持显式 Future（手写 poll），不支持 async/await 语法糖
   - 影响：易用性大幅下降，但 Future trait 可用
   - 触发条件：状态机 lower 不稳定

4. **第四级回退**：完全撤回 async 特性，标记为"实验性，未来版本重试"
   - 影响：无 async 能力
   - 触发条件：核心设计缺陷，需要重新设计

每级回退都保留前序阶段的成果（如 HIR 数据结构、VM 指令定义），便于未来重启。

### 11.9 风险监控与告警

在实施过程中，建立以下监控指标：

| 指标 | 阈值 | 告警动作 |
|------|------|---------|
| 状态机 lower 编译时间 | > 100ms 单 fn | 优化或限制复杂度 |
| async fn 测试通过率 | < 95% | 暂停新特性，修复 bug |
| 自举时间 | > 1s | 检查 tenthc 是否引入 async |
| WASM 路径 async 测试 | 任何失败 | 优先修复 |
| 内存泄漏（leak test） | > 1MB/1000 task | 审计 GC 和 Waker |
| 用户反馈（panic/死锁） | 任何 | 立即修复 |

每周 review 这些指标，超过阈值时启动相应的缓解策略。

---

## 12. 开放问题

> 设计不是一次性的决定，而是一系列有待验证的假设。
> 本章列出本设计中尚未敲定的问题，供实现阶段进一步调研和讨论。

### 12.1 同步函数调用 async 的 bridge 机制

**问题陈述**：同步 fn 调用 async fn 时，需要 `block_on` 将 Future 执行到完成。但若同步 fn 本身被 async fn 调用，就形成了"async → sync → async"的嵌套，`block_on` 在 async 上下文中会死锁。

**当前设计**：
- `block_on` 在 async 上下文中报错或警告
- 用户需要在"边界"处统一转换为 async

**开放问题**：
1. 是否提供 `spawn_blocking` 语义，让同步 fn 能在独立线程中执行 async fn？
   - 优点：解决嵌套问题
   - 缺点：需要多线程运行时，第六阶段才考虑
2. 是否在类型系统中区分 `async fn` 和 `sync fn`，禁止 sync fn 直接调用 async fn？
   - 优点：编译时检查
   - 缺点：限制过严，影响互操作
3. 是否提供 `try_block_on`，在无法阻塞时返回错误而非死锁？
   - 优点：显式失败优于隐式死锁
   - 缺点：用户需要处理错误

**建议**：第一阶段采用方案 3（`try_block_on`），第二阶段引入 `spawn_blocking`。

### 12.2 Future 是否需要 Send bound

**问题陈述**：Rust 的 `Future` trait 有 `Send` bound，要求 Future 可以跨线程传递（用于多线程 executor）。Tenth 当前是单线程运行时，不需要 `Send`。但未来若引入多线程：

- 加 `Send` bound：限制 Future 捕获的变量必须 Send（不能是 Rc<RefCell<>>）
- 不加 `Send` bound：单线程优化更好，但多线程时需要重构

**当前设计**：
- 不引入 `Send` bound，Future 默认是 `!Send`
- 单线程运行时，无问题
- 未来多线程时，引入 `Send` variant

**开放问题**：
1. 是否在类型系统中预留 `Send`/`Sync` trait？
   - 优点：未来扩展容易
   - 缺点：增加当前复杂度
2. 是否提供 `LocalFuture`（!Send）和 `SendFuture`（Send）两种类型？
   - 优点：单线程优化 + 多线程支持
   - 缺点：API 复杂
3. 是否在第一阶段就设计为多线程友好（即使运行时是单线程）？
   - 优点：避免未来重构
   - 缺点：当前性能损失

**建议**：第一阶段不引入 `Send`，第三阶段（多线程运行时）再引入。类型系统预留 `Send`/`Sync` trait 关键字但不实现。

### 12.3 async generator（异步生成器）

**问题陈述**：async generator 是 `async fn` 返回 `Stream<T>`（异步迭代器）的特性，可用于异步流式数据处理（如网络流、文件流）。

```tenth
async fn lines(reader: FileReader) -> Stream<String> {
    loop {
        let line = await reader.read_line();
        if line.is_empty() { break; }
        yield line;  // 异步 yield
    }
}

async fn main() {
    let stream = lines(file);
    while let Some(line) = await stream.next() {
        print(line);
    }
}
```

**当前设计**：第一阶段不支持 async generator。

**开放问题**：
1. 语法：用 `yield` 还是 `async yield`？
2. 类型：`Stream<T>` 是 `Future<Option<T>>` 的语法糖，还是独立类型？
3. 状态机 lower：generator 的状态机比 async fn 更复杂（多个 yield 点 + 终止状态）
4. 与同步迭代器的关系：`Iterator<T>` 和 `Stream<T>` 是否统一？

**建议**：第四阶段考虑，参考 Rust 的 `async-stream` crate 和 Python 的 `async generator`。

### 12.4 错误处理的细节

**问题陈述**：async fn 中的错误处理有几个细节未敲定：

1. **panic 的传播**：task 内 panic 是否应该传播到 `block_on` 的 caller？
   - 当前设计：scheduler 捕获 panic，task 标记为 Failed
   - 开放问题：`block_on` 是否应该 re-panic？还是返回 `Result<T, PanicError>`？

2. **task 取消的错误传播**：当 task 被 cancel 时，`Task<T>::await` 返回什么？
   - 选项 A：返回 `None`（Task<T> 变成 Task<Option<T>>）
   - 选项 B：抛出 `CancelledError`
   - 选项 C：返回 `Result<T, CancelledError>`

3. **Future 的 try 语义**：`try` 在 async fn 中是否传播错误到 caller Future？
   - 当前设计：是
   - 开放问题：是否需要 `try_async` 区分同步 try 和异步 try？

**建议**：
- panic：`block_on` re-panic，保持与同步代码一致
- 取消：选项 C（`Result<T, CancelledError>`），显式处理
- try：统一 try 语义，不区分 sync/async

### 12.5 GPU 计算的异步集成

**问题陈述**：Tenth 的 tensor 运算未来可能支持 GPU 后端。GPU 计算是天然异步的（kernel launch → stream → event）。是否需要将 GPU 计算与 async/await 集成？

**场景**：
```tenth
async fn train_step(model: Model, batch: Tensor) -> Tensor {
    let logits = await model.forward(batch);  // GPU kernel async
    let loss = await loss_fn(logits, batch);
    await loss.backward();  // GPU backward async
    await optimizer.step();
}
```

**当前设计**：第一阶段不支持，tensor 计算保持同步。

**开放问题**：
1. 是否将 GPU kernel launch 封装为 Future？
   - 优点：统一异步模型
   - 缺点：GPU stream 的依赖关系难以用 Future 表达
2. 是否引入 `CudaStream` 类型，与 `Task` 并行？
   - 优点：GPU 专用调度
   - 缺点：两套异步模型
3. 是否在 autograd 中集成 async？
   - 当前设计：autograd 保持同步
   - 开放问题：反向传播是否可以 async？

**建议**：第六阶段（GPU 后端）再考虑。当前设计保持 async 和 tensor 分离。

### 12.6 async fn 的递归

**问题陈述**：async fn 递归调用时，状态机会无限嵌套，导致栈溢出。

```tenth
async fn factorial(n: i64) -> i64 {
    if n <= 1 { return 1; }
    let prev = await factorial(n - 1);  // 递归 await
    n * prev
}
```

状态机 lower 后，`factorial` 的状态机会包含 `factorial` 的状态机，无限递归。

**解决方案**：
1. **Boxing**：将递归 await 的 Future boxed，避免编译时大小爆炸
   ```tenth
   let prev = await Box::new(factorial(n - 1));
   ```
2. **限制递归深度**：编译时检测递归 async fn，报错或警告
3. **要求用户使用 `Box`**：递归 async fn 必须显式 box

**开放问题**：
1. 是否自动 boxing（对性能有影响）？
2. 是否禁止递归 async fn（过于严格）？
3. 是否提供 `async_recursion` 宏（自动 box）？

**建议**：第一阶段禁止递归 async fn（编译报错），第二阶段提供 `async_recursion` 宏。

### 12.7 async fn 的 trait 实现

**问题陈述**：async fn 在 trait 中的使用是 Rust 的著名难题（async fn in trait 直到 2024 年才稳定）。Tenth 是否支持？

```tenth
trait AsyncProcessor {
    async fn process(&self, data: Data) -> Result;
}

impl AsyncProcessor for MyProcessor {
    async fn process(&self, data: Data) -> Result {
        // ...
    }
}
```

**问题**：trait 方法返回 Future，但 Future 的大小未知（取决于具体实现）。

**解决方案**：
1. **Boxing**：trait 方法返回 `Box<dyn Future<Output = Result>>`
2. **impl Future**：使用 `impl Future` 返回类型（需要 GAT）
3. **async trait**：引入 `async trait` 语法糖

**开放问题**：
1. Tenth 的 trait 系统是否支持 GAT（generic associated type）？
2. 是否在第一阶段支持 async fn in trait？
3. 性能影响（boxing 开销）是否可接受？

**建议**：第一阶段不支持，第三阶段考虑。参考 Rust 的 `async fn in trait` RFC。

### 12.8 调度器的公平性

**问题陈述**：协作式调度器中，多个 ready task 如何调度？

- **FIFO**：先就绪的先执行
- **优先级**：高优先级 task 先执行
- **轮转**：所有 ready task 轮流执行

**当前设计**：FIFO（Ready Queue 是 VecDeque）。

**开放问题**：
1. 是否支持 task 优先级？
   - 优点：重要 task 先执行
   - 缺点：低优先级 task 可能饥饿
2. 是否支持 task 抢占（运行超过 N ms 后强制切换）？
   - 优点：避免死循环
   - 缺点：破坏协作式语义
3. 是否支持 task 亲和性（绑定到特定"线程"）？
   - 单线程运行时无意义，多线程时考虑

**建议**：第一阶段 FIFO，第二阶段可选优先级，多线程时考虑亲和性。

### 12.9 跨后端的语义一致性

**问题陈述**：async fn 在 VM/解释器/WASM/JIT 四个后端的语义是否完全一致？

- VM：状态机 + 调度器
- 解释器：tree-walk + 协作式 yield
- WASM：状态机 + host event loop
- JIT：编译为原生代码 + 调度器

**开放问题**：
1. 解释器后端如何实现 async？tree-walk + 协程？还是也用状态机？
2. JIT 后端是否将状态机优化为连续代码（消除 enum 分派）？
3. 不同后端的 task 调度顺序是否一致？

**建议**：
- VM/WASM：状态机
- 解释器：tree-walk + 显式 yield（不 lower 为状态机，性能可接受）
- JIT：基于 VM 字节码，优化状态机分派

### 12.10 调试体验

**问题陈述**：async 代码的调试比同步代码困难：
- 调用栈跨越 task 边界
- 状态机的当前状态不直观
- await 点的"挂起"难以观察

**开放问题**：
1. 是否提供 `async backtrace`，显示 task 的"逻辑调用栈"？
2. 是否在调试模式下保留 async fn 的源码映射（而非状态机）？
3. 是否提供 `task inspect` 命令，查看所有 task 的状态？
4. 是否集成到 Tenth 的关系调试器（护城河 F）？

**建议**：第二阶段实现基础调试支持（task 列表 + 状态查看），第三阶段集成到关系调试器。

### 12.11 与现有 `spawn`/`task` 关键字的关系

**问题陈述**：token.rs 已预留 `spawn` 和 `task` 关键字。本设计复用这两个关键字：

- `spawn`：启动新 task
- `task`：task 块语法（可选）

**开放问题**：
1. `spawn` 的语义是否与原有设计一致？
   - 原有设计（推测）：可能用于并行计算（如 data parallel）
   - 本设计：用于异步 task
   - 是否冲突？
2. `task` 关键字是否真的需要？
   - 当前设计：`task { ... }` 等价于 `spawn(async { ... })`
   - 是否可以省略，直接用 `spawn(async { ... })`？
3. `shard`/`node` 关键字是否与 async 冲突？
   - 见风险 R-D5

**建议**：
- `spawn`：复用，语义为"启动异步 task"
- `task`：第一阶段不使用，保留关键字
- `shard`/`node`：明确分工，不冲突

### 12.12 标准库的 async 模块组织

**问题陈述**：async 相关的标准库如何组织？

**选项 A**：`tenth/std/async/` 目录
```
tenth/std/async/
  future.th
  task.th
  channel.th
  select.th
  time.th
  io.th
  sync.th
```

**选项 B**：分散到现有模块
```
tenth/std/
  future.th       (新)
  task.th         (新)
  channel.th      (新)
  time.th         (扩展现有)
  io.th           (扩展现有)
  sync.th         (新)
```

**选项 C**：混合
```
tenth/std/
  async/
    future.th
    task.th
    channel.th
    select.th
  time.th         (扩展，加 async sleep)
  io.th           (扩展，加 async read/write)
  sync.th         (新，同步原语)
```

**开放问题**：
1. 哪种组织最符合 Tenth 的模块习惯？
2. `async` 是否应该是顶层模块，还是子模块？
3. 现有模块（time/io）的 async 部分是否应该分离？

**建议**：选项 C（混合），保持现有模块的同步 API，新增 `async/` 子目录。

### 12.13 与 macro 系统的交互

**问题陈述**：Tenth 的 macro 系统是否支持 async？

```tenth
macro async_retry(retries: i64, body) {
    let mut count = 0;
    loop {
        try {
            return await body();
        } catch {
            count += 1;
            if count >= retries { return Err(...); }
            await sleep(100);
        }
    }
}

async fn fetch() -> Result<String> {
    async_retry!(3, || fetch_data())
}
```

**开放问题**：
1. macro 展开后是否支持 async/await 语法？
2. macro 内的 `return` 是否正确映射到 async fn 的返回？
3. 是否需要 `async_macro` 关键字区分？

**建议**：第一阶段不支持 macro 内的 async，第二阶段考虑。

### 12.14 性能基准与对比

**问题陈述**：如何评估 async 特性的性能？

**需要的 benchmark**：
1. **spawn/await 开销**：对比同步函数调用
2. **channel 吞吐**：对比其它语言的 channel
3. **select 延迟**：多路复用的开销
4. **状态机 lower 编译时间**：async fn 的编译开销
5. **内存占用**：每个 task 的内存开销

**开放问题**：
1. 基准对比对象：Rust（tokio）、Python（asyncio）、Go（goroutine）、JS（Promise）？
2. 基准场景：IO 密集、计算密集、混合？
3. 是否集成到 CI，监控回归？

**建议**：建立 `bench/async/` 目录，包含微基准和宏基准，每次提交跑微基准，每周跑宏基准。

### 12.15 未决问题汇总

| 问题 ID | 问题 | 阶段 | 优先级 |
|---------|------|------|--------|
| OQ-1 | sync→async bridge | 第一阶段 | 高 |
| OQ-2 | Send bound | 第三阶段 | 中 |
| OQ-3 | async generator | 第四阶段 | 低 |
| OQ-4 | 错误处理细节 | 第一阶段 | 高 |
| OQ-5 | GPU 异步集成 | 第六阶段 | 低 |
| OQ-6 | 递归 async fn | 第二阶段 | 中 |
| OQ-7 | async fn in trait | 第三阶段 | 中 |
| OQ-8 | 调度器公平性 | 第二阶段 | 中 |
| OQ-9 | 跨后端一致性 | 持续 | 高 |
| OQ-10 | 调试体验 | 第二阶段 | 中 |
| OQ-11 | spawn/task 关键字 | 第一阶段 | 高 |
| OQ-12 | stdlib 组织 | 第一阶段 | 高 |
| OQ-13 | macro 交互 | 第二阶段 | 低 |
| OQ-14 | 性能基准 | 第一阶段 | 高 |

---

## 结语

本设计文档提出了 Tenth 语言的异步/并发特性方案：**协作式协程 + async/await**，基于**状态机式 Future** 实现，兼容自举三路径和 WASM 后端约束。

核心设计决策：
1. **状态机式 Future**：唯一不依赖未标准化提案、且 wasmi 完全支持的方案
2. **协作式调度**：单线程事件循环 + Ready Queue + Timer Heap + IO Event Source
3. **Waker 机制**：基于 `Rc<RefCell<Scheduler>>`，避免多线程复杂度
4. **分层同步策略**：tenthc 按 P0→P1→P2 逐步支持 async
5. **分阶段实施**：7 个阶段，从基础语法到完整生态

本设计已识别 22 项风险和 15 项开放问题，每项风险都有缓解策略和回退路径。整体风险等级为中-高，但通过分阶段交付和明确的回退策略，可以控制风险在可接受范围内。

下一步：提交用户审批，审批通过后启动第一阶段实施。

---

**文档版本**：v1.0
**创建日期**：2026-07-06
**作者**：Tenth 总师
**状态**：待审批
