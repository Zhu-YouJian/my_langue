# Tenth 未来 API 规划

> 版本：v0.3.3 | 日期：2026-06-26
>
> 范围：能力全梳理.md 中所有 ❌ 未实现 和 ⚠️ 部分/脚手架 状态的能力
>
> 用途：未来开发的参考清单 + 思路梳理
>
> 优先级图例：A=生存级（第一梯队）/ B=通用必需（第二梯队）/ C=生态远期（第三梯队及以下）
>
> 与 能力全梳理.md 的关系：互补——能力全梳理是状态清单（✅/⚠️/❌），本文档是未来规划（目标签名+依赖+实现要点+优先级）
>
> 说明：已排除"LLVM 后端"（MEMO 2026-06-04 已明确不做）；已删除"GC 垃圾回收"（项目策略用 arena+RC 代替）和"goto"（通常不需要）。重复项采用"主项+引用项（见 x.x）"模式去重。各小节项数与能力全梳理.md 详细表保持一致；能力全梳理.md 第九节"总结矩阵"为粗略估算（与逐条统计有出入），本文档以逐条统计为准。

---

## 目录

1. [一、语言核心设计](#一语言核心设计)
2. [二、编译器与工具链](#二编译器与工具链)
3. [三、运行时系统](#三运行时系统)
4. [四、标准库](#四标准库)
5. [五、Tenth 特定需求（AI 原生语言）](#五tenth-特定需求ai-原生语言)
6. [六、通用编程生态](#六通用编程生态tenth-作为通用语言)
7. [七、语言生态与社区](#七语言生态与社区)
8. [八、平台支持](#八平台支持)
9. [优先级统计](#优先级统计)

---

## 一、语言核心设计

### 1.1 词法与语法

| 能力 | 目标签名 | 依赖 | 实现要点 | 优先级 |
|------|---------|------|---------|--------|
| 字符串字面量（原始/多行） | `r"raw\n"` / `"""multi line"""` | 无 | Lexer read_string 增加 `r"` 前缀分支与三引号状态机 | C |
| 数字字面量（0x/0b/下划线） | `0xFF` / `0b1010` / `1_000_000` | 无 | Lexer read_number 增加进制前缀识别 + 下划线过滤 | C |
| 字符字面量 | `'a'` -> `char` | char 类型已存在 | Lexer 增加 `'` 分支，AST Literal 新增 Char 变体 | B |
| 字节串字面量 | `b"bytes"` -> `Bytes` | 字节/Bytes 类型 | Lexer `b"` 前缀 + 新字面量类型 | C |
| 运算符重载 | `impl Add for T { fn add(self, rhs) }` | Trait 系统 | Trait 方法约定 + Lower 二元运算符查 trait 方法 | C |
| 自定义运算符 | `operator <\|> = fn(a, b)` | 运算符重载 | Lexer 运算符表动态化 + Parser 优先级表 | C |
| 宏/元编程 | `macro! { ... }` | 语法扩展 | AST → AST 变换管线 + hygiene | C |
| 语法扩展 | `#[derive(Foo)]` | 宏/元编程 | 编译插件注册 + 过程宏接口 | C |

### 1.2 类型系统

| 能力 | 目标签名 | 依赖 | 实现要点 | 优先级 |
|------|---------|------|---------|--------|
| 泛型枚举（显式 `<T>`） | `enum Option<T> { Some(T), None }` | 无 | Parser 类型参数列表 + Lower 实例化 | B |
| 编译期 Shape 检查 | `Tensor[f64, M, K] @ Tensor[f64, K, N] -> Tensor[f64, M, N]` | Const 泛型、符号维度 | Type::Tensor shape 字段化 + Lower 维度约束求解 | A |
| 符号维度 | `Tensor[f64, M, K]`（M/K 编译期追踪） | Const 泛型 | 维度变量表 + 跨调用 unify | A |
| 秩多态 | `Tensor[f64, ..]` | Const 泛型 | Type 支持 `..` 通配 + 运行时 rank | A |
| 元组类型 | `tuple: (T1, T2, T3)` / `let p = (1, "x")` | 需扩 Type::Tuple + Value::Tuple + Lower 模式匹配 | 新增枚举变体 + 语法解析 + 模式匹配 | B |
| 联合类型 / Union | `union { i32, f32 }` | 内存布局控制 | 重叠内存结构 + unsafe 边界 | C |
| 关联类型 | `trait Iter { type Item; }` | Trait 系统 | Trait 定义新增 type 项 + 实例化时绑定 | C |
| 默认 Trait 方法 | `trait T { fn f() { .. } }` | Trait 系统 | Trait 方法表带默认实现 + impl 未覆盖时回退 | C |
| Trait 对象 / 动态分发 | `dyn Trait` | Trait 系统 | 胖指针（data+vtable）+ Lower 动态调用 | C |
| 高阶类型 (HKT) | `trait Functor<F>` | 关联类型 | Type 支持 `* -> *` kind 系统 | C |
| 生命周期标注 | `fn f<'a>(x: &'a T)` | 引用/借用 | 引入 `'a` 语法 + 区域推断 | C |
| 生命周期省略 | 自动推断 `'a` | 生命周期标注 | 默认规则 elision | C |
| Never 类型 / 发散 | `fn f() -> !` | 无 | Type::Never + 控制流发散推断 | C |
| Newtype 模式 | `struct Meters(f64)` | 元组类型 | 单字段结构体 + 显式转换 | C |
| 协变/逆变 | 泛型子类型关系 | 生命周期 | Variance 推断表 | C |
| Phantom 类型 | `struct T<PhantomData>` | 无 | 未使用类型参数标记 | C |
| GAT | `type Item<'a>;` | 关联类型 | 关联类型 + 生命周期联动 | C |
| Const 泛型 | `Tensor[f64, 3, 4]`（编译期常量参数） | 无 | Type 支持 const 参数 + 编译期求值 | A |

### 1.3 控制流

| 能力 | 目标签名 | 依赖 | 实现要点 | 优先级 |
|------|---------|------|---------|--------|
| break 带值 | `loop { break val; }` | 无 | Break 指令携带 Value | C |
| 标签 break | `break 'outer` | 无 | 循环标签表 + Break 跳转目标 | ✅ 已完成（2026-08-01 M2.3，详见 能力全梳理 §1.3） |
| try / catch / 异常 | `try { } catch(e) { }` | 异常处理运行时 | try 关键字已存在；unwind 机制 + catch 块 | B |
| try 表达式 / `?` 操作符 | `fn f() -> Result<T,E> { let x = g()?; }` | Result 类型 | Lower 把 `?` 展开为 match + 提前 return | B |
| do-while 循环 | `do { } while cond` | 无 | Parser 新增 do 语法 + 条件后测跳转 | C |
| yield / 协程 | `fn gen() { yield v; }` | async runtime | 状态机改写 + 暂停/恢复（含绿色线程） | A |
| 尾调用优化 | TCO | MIR | 编译期尾位置识别 + 跳转替代 Call | C |

### 1.4 函数与闭包

| 能力 | 目标签名 | 依赖 | 实现要点 | 优先级 |
|------|---------|------|---------|--------|
| 柯里化 / 部分应用 | `f(a)(b)` | 闭包 | 多参数函数返回闭包链 | C |
| 默认参数 | `fn f(x: i64 = 0)` | 无 | 函数签名默认值 + 调用点补全 | C |
| 命名参数 | `f(x = 1, y = 2)` | 无 | 调用点参数名匹配 + 重排 | C |
| 可变参数 | `fn f(...args: T)` | 无 | 参数打包为 Vec + 模板生成 | C |
| 函数重载 | `fn f(i64)` / `fn f(str)` | Trait 系统 | 按 signature mangle + 静态分派 | C |
| 构造函数 / 析构函数 | `fn new()` / `fn drop()` | Drop trait | new 约定 + Drop 生命周期钩子 | C |
| async / await | `async fn f() -> T` / `await fut` | async runtime | async 关键字已存在；状态机改写 + Future | A |
| 生成器 / yield | `fn gen() -> Iterator<T>` | yield / 协程 | 状态机改写 + 惰性序列 | C |
| 尾递归 | 递归优化 | MIR | 递归识别 + 跳转替代 | C |

### 1.5 内存与所有权

| 能力 | 目标签名 | 依赖 | 实现要点 | 优先级 |
|------|---------|------|---------|--------|
| 生命周期 | `'a` 显式标注 | 生命周期标注 | 见 1.2 | C |
| 智能指针 | `Box<T>` / `Rc<T>` / `Arc<T>` | 无 | 堆分配 + 引用计数封装 | C |
| 弱引用 | `Weak<T>` | 智能指针 | 弱计数 + upgrade | C |
| Drop trait / RAII | `impl Drop for T { fn drop() }` | 无 | 作用域退出插 Drop 调用 | C |
| Copy trait | `impl Copy for T` | 无 | 按位复制标记 + 隐式复制语义 | C |
| 手动内存管理 | `alloc`/`free` | FFI | 暴露底层分配器（不推荐用户使用） | C |
| 内存对齐 | `#[repr(align)]` | 内存布局控制 | 类型对齐属性 + 分配器对齐 | C |
| Pin | `Pin<T>` | 智能指针 | 自引用安全封装 | C |

---

## 二、编译器与工具链

### 2.1 编译管线

| 能力 | 目标签名 | 依赖 | 实现要点 | 优先级 |
|------|---------|------|---------|--------|
| MIR / IR | HIR -> MIR | HIR | 中层 SSA 表示 + 控制流图 | C |
| LIR | MIR -> LIR | MIR | 低层目标无关 IR | C |
| 常量折叠 | 编译期常量计算 | 编译期求值 | 常量表达式识别 + 求值替换 | C |
| 死代码消除 | DCE | MIR | 可达性分析 + 删除 | C |
| 内联优化 | 函数内联 | MIR | 热点函数展开 | C |
| 循环优化 | 展开向量化 | MIR | 循环不变量外提 + 展开 | C |
| 尾调用优化 | TCO | MIR | 见 1.3 | C |
| 编译期求值 | `const fn` | 无 | 常量上下文解释执行 | C |
| 增量编译 | 只重编译变更 | 编译缓存 | 文件指纹 + 增量图 | C |
| 并行编译 | 多线程编译 | 无 | 模块级并行调度 | C |
| 编译缓存 | 缓存中间结果 | 无 | HIR/字节码磁盘缓存 | C |

### 2.2 执行后端

| 能力 | 目标签名 | 依赖 | 实现要点 | 优先级 |
|------|---------|------|---------|--------|
| JIT 编译 | Cranelift 热点编译 | Cranelift 后端 | 脚手架已存在；热点函数编译为原生码。**P 系列遗留（2026-08-04 M2.6-P5 登记，见 AUDIT-11.4.38）**：Await/Yield JIT 挂起（对应 §1.4 async/await A 级）、深递归栈（stack probe/堆栈切换）、递归闭包 MakeCell/BindSelfCapture JIT、P1 每次调用 ~5ns 开销、f32 特化 | B |
| AOT 编译到原生码 | `tenth compile --target=native` -> `.exe`/ELF | Cranelift 后端 | 全程序编译 + 链接 | C |
| GPU 编译 | `tenth compile --target=cuda` -> CUDA kernel | CUDA 后端 + 算子融合 | 脚手架已存在；MIR->CUDA 算子映射 + kernel 融合 | A |
| 跨编译 | 平台 A 编译平台 B 产物 | 条件编译 | target 三元组配置 | C |
| 链接器 | 静态/动态链接 | AOT 编译 | 符号解析 + 重定位 | C |
| Cranelift 后端 | 编译到原生码 | 无 | 依赖已引入但未使用；接 MIR | C |
| TPU / NPU 后端 | AI 专用硬件 | GPU 编译 | 硬件特定算子生成（远期讨论项） | C |

### 2.3 开发工具

| 能力 | 目标签名 | 依赖 | 实现要点 | 优先级 |
|------|---------|------|---------|--------|
| 调试器 | `fn debugger_start(file: str) -> DebugSession` | 无 | 断点插桩 + 调用栈 + 变量查看 | C |
| 性能分析器 | Profiler / flamegraph | 无 | 采样剖析 + 火焰图渲染 | C |
| Linter | 静态检查/代码风格 | LSP | LSP diagnostics 已部分；扩充规则集 | C |
| 文档生成 | `tenth doc` -> API 文档 | 无 | 从注释提取 + 生成 HTML | C |
| 语法高亮 | 编辑器插件 | 无 | VSCode 扩展 TextMate 语法 | C |
| 代码片段 / Snippets | 编辑器模板 | VSCode 扩展 | snippets.json | C |
| 交互式教程 | Tour / Playground | 在线 Playground | 渐进式课程 + 内嵌运行 | C |
| 在线 Playground | Web 端运行 | WASM 后端 | WASM 编译 + 浏览器运行时 | C |
| 语法树可视化 | AST 图形化 | 无 | AST -> dot/svg 渲染 | C |
| 字节码反汇编 | `tenth disasm` 查看 VM 指令 | 无 | 字节码 -> 文本反汇编 | C |
| WASM 调试 | WASM 模块检查 | WASM 后端 | wasm-tools 集成 | C |

### 2.4 包管理与构建

| 能力 | 目标签名 | 依赖 | 实现要点 | 优先级 |
|------|---------|------|---------|--------|
| 包注册中心 | 中央仓库 | tenthpm | 服务端 + CLI 上传/下载 | C |
| 依赖解析 | semver 版本范围 | tenthpm | semver 求解器（Tenth.lock 已存在） | C |
| 工作空间 | monorepo 多包管理 | tenthpm | 根 manifest + 成员包 | C |
| 构建脚本 | `build.rs` 等价物 | tenthpm | 构建前钩子脚本 | C |
| 条件编译 | `#[cfg(target_os)]` | 无 | cfg 属性解析 + 代码裁剪 | C |
| Feature flags | `--features foo` | 条件编译 | feature 集合 + cfg 联动 | C |
| 增量构建 | 只重编译变更 | 编译缓存 | 文件指纹 + 依赖图 | C |
| 交叉编译支持 | target 配置 | 跨编译 | target 三元组 + 工具链选择 | C |

---

## 三、运行时系统

| 能力 | 目标签名 | 依赖 | 实现要点 | 优先级 |
|------|---------|------|---------|--------|
| 线程 / 并发 | `spawn(fn() -> T) -> JoinHandle<T>` | 无 | spawn 关键字已存在；OS 线程封装 | A |
| async runtime | 事件循环 | 无 | Reactor + Executor + Task 调度 | A |
| 通道 / 消息传递 | `channel<T>() -> (Sender<T>, Receiver<T>)` | 无 | mpsc 队列 + 同步原语 | A |
| 互斥锁 / 读写锁 | `Mutex<T>` / `RwLock<T>` | 原子操作 | OS 互斥量 + RAII guard | A |
| 原子操作 | `AtomicU64` 等 | 无 | 原子指令封装 | B |
| 条件变量 | `Condvar` | 互斥锁 | OS condvar 封装 | C |
| 线程局部存储 | `thread_local!` | 线程/并发 | TLS key 管理 | C |
| 信号处理 | `signal(SIGINT, handler)` | 无 | OS 信号注册 + 回调 | C |
| 异常处理运行时 | unwind | try/catch | 栈展开 + catch 帧链 | C |
| Panic / Abort | `panic("msg")` / `abort()` | 无 | 有错误传播但无 panic 机制；统一致命错误 | C |
| 栈溢出检测 | stack guard | 无 | 栈底 guard page | C |
| FFI 外部函数接口 | `extern "C" fn` / `call_c(lib, sym)` | 无 | libffi 或 dlsym + 调用约定 | B |
| 动态加载 | `dlopen(path) -> Module` | FFI | 平台 dlopen/LoadLibrary | C |
| 内联汇编 | `asm! { }` | 无 | 平台汇编注入（远期） | C |
| SIMD 向量化 | `std::simd` | 无 | 平台 SIMD intrinsics | C |
| 异常安全 | panic 时资源安全 | Panic / Abort | Drop 在 unwind 路径执行 | C |

---

## 四、标准库

### 4.1 核心类型与集合

| 能力 | 目标签名 | 依赖 | 实现要点 | 优先级 |
|------|---------|------|---------|--------|
| 数组 / 定长数组 | `[T; N]` | Const 泛型 | 定长栈数组 + 索引 | B |
| HashSet | `fn hash_set_new() -> HashSet<T>` | 需新增 Hash trait | 基于现有 HashMap 包装去重 | B |
| BTreeMap | `fn btree_map_insert(map: BTreeMap<K,V>, k: K, v: V)` | 需新增 Ord trait | B 树有序映射 | C |
| LinkedList | 双向链表 | 无 | 节点指针链接 | C |
| VecDeque | 双端队列 | 无 | 环形缓冲区 | C |
| BinaryHeap | 优先队列 | 需新增 Ord trait | 二叉堆 | C |
| 元组 | `(a, b, c)` | 元组类型 | 见 1.2 | B |
| Range 类型 | `Range`/`RangeInclusive`（显式类型） | 无 | 隐式支持 `0..10`；提升为显式类型 | B |
| 迭代器 trait | `trait Iterator<T> { fn next() -> Option<T> }` | 需新增 Iterator trait + next 方法约定 | 现有 iter 工具函数；抽象为 trait | B |
| 字符 / Char | `'a'` -> char | 字符字面量 | 有类型无字面量；补字面量语法 | B |
| 字节 / Bytes | `Bytes` 字节序列 | 字节串字面量 | Vec<u8> 封装 + 方法 | C |
| 大整数 | `BigInt` | 无 | 任意精度整数运算 | C |
| 复数 | `Complex<f64>` | 无 | 实部+虚部 + 运算 | C |
| 小数 / Decimal | `Decimal` 精确小数 | 无 | 定点小数运算 | C |

### 4.2 字符串与文本

| 能力 | 目标签名 | 依赖 | 实现要点 | 优先级 |
|------|---------|------|---------|--------|
| split / trim / replace | `s.split(",")` / `s.trim()` / `s.replace(a,b)` | 无 | 部分在 std；补全 API 与边界 | B |
| 正则表达式 | `regex(pattern: str) -> Regex` / `re.match(s)` | 无 | 引入 regex 引擎或自研 NFA | B |
| Unicode 规范化 | `nfc(s)` / `nfd(s)` | 无 | Unicode 规范化算法 | C |
| 字符串格式化 | `format!("{} = {}", k, v)` | 无 | 有 format 但功能有限；补齐占位符 | B |
| 字符串构建器 | `StringBuilder::new() -> StringBuilder` | 无 | 可变缓冲 + append + to_string | C |
| 编码转换 | `to_utf16(s)` / `from_gbk(b)` | 无 | 编码表 + 转换 | C |
| 模板字符串 | `f"hello {name}"` | 字符串格式化 | Lexer f" 前缀 + 插值解析 | C |
| Base64 编解码 | `base64_encode(b: Bytes) -> str` / `base64_decode(s) -> Bytes` | 字节/Bytes | 标准 base64 算法 | C |
| Hex 编解码 | `hex_encode` / `hex_decode` | 字节/Bytes | 十六进制转换 | C |
| URL 编解码 | `url_encode` / `url_decode` | 无 | percent-encoding | C |
| HTML 转义 | `html_escape(s) -> str` | 无 | 实体转义表 | C |
| CSV 解析 | `csv_parse(s: str) -> Vec<Vec<str>>` | 无 | RFC 4180 解析器 | C |
| XML 解析 | `xml_parse(s) -> XmlNode` | 无 | XML DOM/SAX 解析 | C |
| YAML 解析 | `yaml_parse(s) -> Value` | 无 | YAML 解析器 | C |
| Markdown 解析 | `md_to_html(s) -> str` | 无 | Markdown -> HTML 渲染 | C |

### 4.3 数学与数值

| 能力 | 目标签名 | 依赖 | 实现要点 | 优先级 |
|------|---------|------|---------|--------|
| 统计函数 | `mean(v)` / `variance(v)` / `stddev(v)` | 迭代器 trait | 纯算法实现 | C |
| 线性代数 | `svd(m)` / `eig(m)` / `lu(m)` | 张量类型 | 矩阵分解算法 | C |
| FFT | `fft(v: Vec<f64>) -> Vec<Complex>` | 复数 | Cooley-Tukey 算法 | C |
| 复数运算 | `Complex::add` 等 | 复数 | 复数算术 | C |
| 有理数 | `Rational` | 大整数 | 分数运算 + 规约 | C |
| 数值积分 | `integrate(f, a, b)` | 无 | 辛普森/高斯求积 | C |
| 插值 | `interp(xs, ys, x)` | 无 | 线性/样条插值 | C |

### 4.4 文件与 I/O

| 能力 | 目标签名 | 依赖 | 实现要点 | 优先级 |
|------|---------|------|---------|--------|
| 文件流 / 缓冲 I/O | `BufReader::open(path)` / `BufWriter::create(path)` | 无 | Rust std::io 桥接 | B |
| 标准输入 | `stdin().read_line() -> str` | 无 | Rust std::io::stdin 桥接 | B |
| 标准错误输出 | `eprintln(msg)` | 无 | 有 eprintln 但功能有限；补齐格式化 | C |
| 文件锁 | `flock(path, mode)` | 无 | OS 文件锁 | C |
| 文件权限 | `chmod(path, mode)` | 无 | OS 权限调用 | C |
| 文件监听 | `watch(path, cb)` | 无 | inotify/kqueue 封装 | C |
| 内存映射文件 | `mmap(path) -> Mmap` | 无 | memmap2 桥接 | C |
| 临时文件 | `tempfile() -> Path` | 无 | 随机命名 + 自动清理 | C |
| 异步 I/O | `async_read(path) -> Future<Bytes>` | async runtime | 异步文件读写 | C |

### 4.5 网络与通信

| 能力 | 目标签名 | 依赖 | 实现要点 | 优先级 |
|------|---------|------|---------|--------|
| TCP 客户端 | `tcp_connect(addr: str) -> TcpStream` | 无 | native 函数 + Rust std::net::TcpStream 桥接 | B |
| TCP 服务端 | `tcp_bind(addr: str) -> TcpListener` | 无 | Rust std::net::TcpListener 桥接 | B |
| UDP | `UdpSocket::bind(addr)` | 无 | Rust std::net::UdpSocket 桥接 | C |
| HTTP 客户端 | `http_get(url: str) -> HttpResponse` | TCP 客户端 | HTTP/1.1 请求解析 | B |
| HTTP 服务端 | `HttpServer::new().route("/", handler)` | TCP 服务端 | Web 框架脚手架（TCP 服务端的封装层，非通用必需） | C |
| WebSocket | `ws_connect(url) -> WsStream` | HTTP 客户端 | WebSocket 协议握手 + 帧 | C |
| TLS/SSL | `tls_connect(addr) -> TlsStream` | TCP 客户端 | rustls 桥接 | C |
| DNS 解析 | `dns_lookup(host: str) -> Vec<str>` | 无 | getaddrinfo 桥接 | C |
| URL 解析 | `url_parse(s: str) -> Url` | 无 | URL 结构化解析 | C |
| Socket 原始 | `raw_socket() -> Socket` | 无 | 原始套接字 | C |
| Unix Domain Socket | `uds_connect(path) -> UnixStream` | 无 | Unix 域套接字 | C |
| gRPC | `grpc_call(service, method, req)` | HTTP 服务端、序列化协议 | gRPC 框架（远期讨论项） | C |
| MQTT | `mqtt_connect(broker) -> MqttClient` | TCP 客户端 | MQTT 协议客户端 | C |
| 序列化协议 | Protobuf/Thrift 编解码 | 序列化 | 二进制 schema 协议 | C |

### 4.6 时间与日期

| 能力 | 目标签名 | 依赖 | 实现要点 | 优先级 |
|------|---------|------|---------|--------|
| 日期解析 | `date_parse(s: str) -> Date` | 无 | strptime 实现 | B |
| 日期格式化 | `date_format(d: Date, fmt: str) -> str` | 无 | strftime 实现 | B |
| 日期运算 | `date_add_days(d, n)` / `date_add_months(d, n)` | 无 | 日历运算 | C |
| 时区 | `to_timezone(d, tz)` | 无 | 时区数据库 | C |
| 日历 | 农历/回历转换 | 无 | 历法算法 | C |
| Duration 类型 | `Duration::from_secs(n)` | 无 | 时间间隔类型 | C |
| Instant / SystemTime | `Instant::now()` / `SystemTime::now()` | 无 | 精确时间测量 | C |

### 4.7 系统与进程

| 能力 | 目标签名 | 依赖 | 实现要点 | 优先级 |
|------|---------|------|---------|--------|
| 环境变量 | `getenv(key) -> str` / `setenv(key, val)` | 无 | Rust std::env 桥接 | B |
| 执行子进程 | `Command::new(prog).args(a).run() -> ExitStatus` | 无 | Rust std::process 桥接 | B |
| 进程退出 | `exit(code: i32) -> !` | 无 | OS exit 调用（返回类型用发散语义，不需 Never 类型） | C |
| 进程 ID | `getpid() -> i64` | 无 | OS getpid | C |
| 工作目录 | `getcwd() -> str` / `chdir(path)` | 无 | Rust std::env 桥接 | C |
| 信号处理 | 见 三.信号处理 | — | 同 三（去重引用） | C |
| 用户/权限 | `getuid() -> i64` / `setuid(n)` | 无 | OS uid/gid | C |
| 系统信息 | `os_name()` / `cpu_count()` / `mem_total()` | 无 | 系统信息采集 | C |
| 管道 | `pipe() -> (ReadEnd, WriteEnd)` | 无 | OS pipe | C |
| 重定向 | stdin/stdout 重定向 | 管道 | fd 重定向 | C |

### 4.8 序列化

| 能力 | 目标签名 | 依赖 | 实现要点 | 优先级 |
|------|---------|------|---------|--------|
| TOML 解析 | `toml_parse(s) -> Value` / `toml_to_string(v)` | 无 | 有 toml.th 但功能有限；补全 | C |
| CSV | `csv_read(path) -> Table` | 无 | RFC 4180 | C |
| XML | `xml_to_string(node) -> str` | XML 解析 | 序列化输出 | C |
| YAML | `yaml_to_string(v)` | YAML 解析 | 序列化输出 | C |
| Protocol Buffers | `proto_encode(msg) -> Bytes` | 序列化协议 | schema 编译 + 编解码 | C |
| MessagePack | `msgpack_encode(v) -> Bytes` | 无 | 二进制 JSON | C |
| BSON | `bson_encode(v) -> Bytes` | 无 | MongoDB 二进制格式 | C |
| Pickle | `pickle_load(b)` | 无 | Python 兼容协议 | C |
| SafeTensors | `safetensors_save(map)` / `safetensors_load(path)` | 张量类型 | AI 模型格式 | C |
| ONNX | `onnx_export(model)` | 模型保存/加载 | 模型交换格式 | C |
| GGUF / llama.cpp | `gguf_load(path)` | 模型保存/加载 | LLM 模型格式 | C |
| HDF5 | `hdf5_read(path)` | 无 | 科学数据格式 | C |
| Parquet | `parquet_read(path)` | 无 | 列式存储 | C |

### 4.9 加密与安全

| 能力 | 目标签名 | 依赖 | 实现要点 | 优先级 |
|------|---------|------|---------|--------|
| 哈希函数 | `sha256(b: Bytes) -> Bytes` / `md5(b) -> Bytes` | 字节/Bytes | sha2/md5 算法 | C |
| HMAC | `hmac(key, msg) -> Bytes` | 哈希函数 | HMAC 构造 | C |
| 对称加密 | `aes_encrypt(key, data) -> Bytes` | 字节/Bytes | AES 算法 | C |
| 非对称加密 | `rsa_encrypt(pubkey, data) -> Bytes` | 大整数 | RSA/ECC 算法 | C |
| 数字签名 | `sign(privkey, msg) -> Bytes` / `verify(pubkey, msg, sig)` | 非对称加密 | 签名算法 | C |
| 密码哈希 | `bcrypt(pw)` / `argon2(pw)` | 哈希函数 | 慢哈希算法 | C |
| 随机数安全 | `csprandom(n) -> Bytes` | 无 | CSPRNG（当前随机数不安全） | C |
| 证书 | `x509_parse(b)` | 非对称加密 | X.509 证书解析 | C |
| TLS | `tls_server(addr, cert)` | TLS/SSL | 传输层安全服务端 | C |

### 4.10 测试框架

| 能力 | 目标签名 | 依赖 | 实现要点 | 优先级 |
|------|---------|------|---------|--------|
| 单元测试 | `#[test] fn t() { }` | 无 | 测试在 Rust 侧；Tenth 侧 #[test] 属性 + 运行器 | C |
| 断言 | `assert!(cond)` / `assert_eq!(a, b)` | 无 | 宏 + 错误信息 | C |
| 测试框架 | `describe("x") { it("y") { } }` | 单元测试 | BDD 风格 DSL | C |
| 属性测试 | `fuzz(f, strategy)` | 随机数安全 | fuzz 引擎 + 策略 | C |
| 基准测试 | `bench("name") { }` | 无 | 计时 + 统计 | C |
| 测试覆盖率 | `coverage()` | 单元测试 | 行/分支覆盖采集 | C |
| Mock | `mock(Trait) -> Mock` | Trait 对象 | mock 对象生成 | C |
| 集成测试 | `tests/` 目录 | 单元测试 | Tenth 侧集成测试运行 | C |
| 快照测试 | `snapshot!(actual)` | 无 | 黄金文件对比 | C |
| TDD 工具 | watch + auto-run | 文件监听 | 文件变更自动跑测试 | C |

### 4.11 日志与可观测性

| 能力 | 目标签名 | 依赖 | 实现要点 | 优先级 |
|------|---------|------|---------|--------|
| 结构化日志 | `log_json(level, fields)` | 无 | JSON 日志输出 | C |
| 日志轮转 | `rotate(path, max_size)` | 文件流 | 按大小/时间轮转 | C |
| 分布式追踪 | OpenTelemetry 集成 | 网络 | trace/span 上报 | C |
| 指标收集 | `metrics::counter("x")` | 无 | counter/gauge/histogram | C |
| 性能追踪 | `tracing::span("x")` | 无 | 区间追踪 + 导出 | C |

### 4.12 压缩与归档

| 能力 | 目标签名 | 依赖 | 实现要点 | 优先级 |
|------|---------|------|---------|--------|
| Gzip | `gzip_encode(b: Bytes) -> Bytes` | 字节/Bytes | flate2 桥接 | C |
| Zip | `zip_archive(files) -> Bytes` | 字节/Bytes | zip 归档 | C |
| Tar | `tar_archive(files) -> Bytes` | 字节/Bytes | tar 归档 | C |
| Zstd | `zstd_encode(b) -> Bytes` | 字节/Bytes | zstd 桥接 | C |
| Brotli | `brotli_encode(b) -> Bytes` | 字节/Bytes | brotli 桥接 | C |
| LZ4 | `lz4_encode(b) -> Bytes` | 字节/Bytes | lz4 桥接 | C |

---

## 五、Tenth 特定需求（AI 原生语言）

### 5.1 张量系统

| 能力 | 目标签名 | 依赖 | 实现要点 | 优先级 |
|------|---------|------|---------|--------|
| 多精度张量（f16/bf16） | `Tensor[f16, M, N]` / `Tensor[bf16, M, N]` | TensorData enum | f32 已完成；扩展 F16/Bf16 变体 + dtype 分支 | B |
| 混合精度训练 | `autocast(model, ctx) -> AMPModel` | 多精度张量 | AMP 自动精度切换 + loss scaling（主项；5.3 引用此条） | B |
| 稀疏张量 | `SparseTensor::from_coo(indices, values)` | 无 | COO/CSR 表示 + 稀疏算子 | C |
| 量化张量 | `QuantTensor<int8, scale, zero_point>` | 无 | int8/int4 量化存储 | C |
| 张量并行 | `shard(tensor, dim)` 跨设备分片 | GPU 编译、分布式 | 跨设备分片 + 通信 | C |
| 内存高效注意力 | `flash_attention(q, k, v)` | CUDA 后端 | FlashAttention kernel | C |
| 梯度检查点 | 见 5.2.梯度检查点 | — | 同 5.2（去重引用） | C |
| 张量序列化 | `tensor_to_bytes(t) -> Bytes` | 字节/Bytes | 有 save_model 但非标准格式；定义高效二进制格式 | C |
| 张量索引高级 | `tensor[0:3, ::2, 1]` | 无 | 多维切片 + 步长索引 | C |
| 广播规则完善 | NumPy 级广播 | 无 | 部分实现；补全缺失广播场景 | B |
| 张量比较 | `tensor > 0` -> `Tensor[bool]` | 无 | 逐元素比较 + bool 张量 | B |
| Where / Select | `where(cond, x, y) -> Tensor` | 张量比较 | 条件选择算子 | B |
| Scatter / Gather | `scatter(t, idx, src)` / `gather(t, idx)` | 无 | 分散/聚集操作 | C |
| Sort / TopK | `sort(t, dim)` / `topk(t, k)` | 无 | 排序/Top-K 算子 | C |
| Einsum | `einsum("ij,jk->ik", a, b)` | 无 | 爱因斯坦求和解析 + 执行 | C |
| 张量拼接高级 | `stack(ts)` / `hstack(ts)` / `vstack(ts)` | 无 | 新轴拼接 + 水平/垂直拼接 | C |
| Padding | `pad(t, paddings, mode)` | 无 | 零填充/反射填充 | C |
| Pooling | `max_pool(t, k, s)` / `avg_pool(t, k, s)` | 无 | 最大/平均池化算子 | C |
| Upsample | `upsample(t, scale)` | 无 | 上采样算子 | C |

### 5.2 自动微分

| 能力 | 目标签名 | 依赖 | 实现要点 | 优先级 |
|------|---------|------|---------|--------|
| 二阶导数 | `hessian(f, x) -> Tensor` | 高阶导数 | 二次反向或前向+反向 | C |
| 高阶导数 | `n_grad(f, x, n)` | 前向模式 AD | 嵌套求导 | C |
| 前向模式 AD | `jvp(f, x, v)` | 无 | 仅反向模式；补 JVP | C |
| 梯度累积 | `accumulate_grad(steps)` | 自动微分 | 分批累加梯度后更新 | B |
| 梯度裁剪 | `clip_grad_by_norm(params, max_norm)` | 自动微分 | 按范数/值裁剪 | B |
| 自定义算子 | `register_grad(op, fwd, bwd)` | 自动微分 | 用户定义可微算子 | B |
| 梯度检查点 | `checkpoint(f, x)` | 自动微分 | recomputation（主项；5.1/5.3 引用此条） | C |
| 混合精度梯度 | fp16 梯度 | 多精度张量 | f16 反向路径 | C |
| 稀疏梯度 | sparse backward | 稀疏张量 | 稀疏梯度累积 | C |
| JIT 编译训练步 | `@jit train_step` | JIT 编译 | 训练步函数 JIT | C |
| vmap / 批量向量化 | `vmap(f)` | 无 | 自动批处理 | C |
| 函数变换 | `grad(f)` / `vmap(f)` / `pmap(f)` | 自动微分 | 函数式 API | C |

### 5.3 神经网络库

| 能力 | 目标签名 | 依赖 | 实现要点 | 优先级 |
|------|---------|------|---------|--------|
| Conv1D / Conv3D | `conv1d(x, w, stride)` / `conv3d(x, w, stride)` | Conv2D | 复用 Conv2D im2col 变体 | C |
| RNN / LSTM / GRU | `lstm(input, hidden) -> Tensor` | 无 | 循环层 + 门控 | C |
| Transposed Conv | `conv_transpose2d(x, w)` | Conv2D | 反卷积 | C |
| Depthwise Conv | `depthwise_conv2d(x, w)` | Conv2D | 深度可分离卷积 | C |
| GroupNorm / InstanceNorm | `group_norm(x, groups)` / `instance_norm(x)` | 无 | 其他归一化层 | C |
| MaxPool / AvgPool | `max_pool2d(x, k, s)` / `avg_pool2d(x, k, s)` | Pooling | 池化层（封装 5.1 Pooling） | B |
| AdaptivePool | `adaptive_avg_pool(x, out_size)` | Pooling | 自适应输出尺寸 | C |
| Transformer Decoder | `TransformerDecoder::new(layers)` | Transformer | 编码器已完成；补解码器块（AI 库扩展，非通用必需） | C |
| Beam Search | `beam_search(model, start, k)` | Transformer Decoder | 推理解码 | C |
| 模型量化 | `quantize(model, dtype)` | 量化张量 | int8/int4 量化 | C |
| 模型剪枝 | `prune(model, ratio)` | 无 | 结构化/非结构化剪枝 | C |
| 模型蒸馏 | `distill(teacher, student, data)` | 无 | 知识蒸馏 | C |
| 预训练模型加载 | `load_huggingface(name)` | 模型保存/加载 | 加载 HF 等格式 | C |
| 模型推理引擎 | `InferenceModel::load(path)` | 模型保存/加载 | 推理优化 | C |
| ONNX 导出/导入 | `onnx_export(model)` / `onnx_import(path)` | ONNX | 模型交换 | C |
| TensorBoard / 日志 | `summary_writer(logdir)` | 文件流 | 训练可视化 | C |
| 学习率调度器 | `cosine_schedule(opt)` / `step_schedule(opt)` | 优化器 | lr 调度 | B |
| 早停 / 检查点 | `early_stop(monitor)` / `save_best(model)` | 模型保存/加载 | 早停 + 最佳检查点 | B |
| 数据增强 | 见 5.5.数据增强 | — | 同 5.5（去重引用） | C |
| 分布式训练 | DDP / 模型并行 | 分布式 | 见 5.7 | C |
| 梯度累积 | 见 5.2.梯度累积 | — | 同 5.2（去重引用） | C |
| 混合精度训练 | 见 5.1.混合精度训练 | — | 同 5.1（去重引用） | C |

### 5.4 优化器

| 能力 | 目标签名 | 依赖 | 实现要点 | 优先级 |
|------|---------|------|---------|--------|
| AdamW | `adamw_step(params, grads, lr, weight_decay)` | Adam | 解耦权重衰减（单个优化器，非通用必需） | C |
| Lion | `lion_step(params, grads, lr)` | 无 | 新优化器 | C |
| Adafactor | `adafactor_step(params, grads)` | 无 | 因子化二阶矩 | C |
| LAMB | `lamb_step(params, grads, lr)` | Adam | 层自适应 | C |
| NAdam | `nadam_step(params, grads, lr)` | Adam | Nesterov + Adam | C |
| RAdam | `radam_step(params, grads, lr)` | Adam | 矩估计修正 | C |
| Lookahead | `lookahead_step(base_opt, params)` | 优化器 | 前瞻优化器 | C |
| 学习率调度 | `cosine_lr(opt, step)` / `step_lr(opt, step)` / `exp_lr(opt, step)` | 优化器 | cosine/step/exponential | B |
| 预热策略 | `warmup_lr(opt, warmup_steps)` | 学习率调度 | warmup | B |
| 梯度裁剪 | 见 5.2.梯度裁剪 | — | 同 5.2（去重引用） | B |

### 5.5 数据处理

| 能力 | 目标签名 | 依赖 | 实现要点 | 优先级 |
|------|---------|------|---------|--------|
| CIFAR-10 加载 | `load_cifar10(path) -> Dataset` | DataLoader | 二进制解析 + DataLoader 适配 | C |
| ImageNet 加载 | `load_imagenet(path) -> Dataset` | DataLoader | 大规模图像目录扫描 | C |
| CSV 数据加载 | `load_csv(path) -> Dataset` | CSV 解析 | 表格数据集 | C |
| JSON 数据集 | `load_json(path) -> Dataset` | 无 | 结构化数据集 | C |
| 文本数据集 | `load_text(path) -> Dataset` | 无 | NLP 语料加载 | C |
| 数据增强 | `image_augment(img, ops)` | 无 | 图像翻转/裁剪/旋转（主项；5.3 引用此条） | C |
| Tokenizer | `Tokenizer::from_file(path)` | 无 | 分词器 | C |
| BPE / WordPiece | `bpe_encode(tok, s)` / `wordpiece_encode(tok, s)` | Tokenizer | 子词分词 | C |
| 数据缓存 | `cache(dataset) -> CachedDataset` | 文件流 | 预处理缓存 | C |
| 流式加载 | `stream_dataset(path) -> Iterator` | 迭代器 trait | 大数据集流式读取 | C |
| 数据采样器 | `Sampler::random()` / `Sampler::weighted(w)` | 无 | 随机/加权/分层采样 | C |
| 多进程加载 | `multiprocess_loader(dataset, workers)` | 线程/并发 | 并行数据加载 | C |

### 5.6 GPU / 加速计算

| 能力 | 目标签名 | 依赖 | 实现要点 | 优先级 |
|------|---------|------|---------|--------|
| CUDA 后端 + 算子融合 | `cuda_available() -> bool` / 实际 kernel / FusionPass | 无 | 脚手架已存在；实现真实算子 kernel + kernel 自动融合（原 5.6 算子融合合并到此项） | A |
| cuDNN 集成 | `cudnn_conv2d(x, w)` | CUDA 后端 | 深度学习加速库 | C |
| Metal 后端 | `metal_context()` | 无 | Apple GPU（远期讨论项） | C |
| Vulkan 后端 | `vulkan_context()` | 无 | 跨平台 GPU | C |
| OpenCL 后端 | `opencl_context()` | 无 | 跨平台计算 | C |
| ROCm / HIP | `rocm_context()` | 无 | AMD GPU | C |
| TPU 后端 | `tpu_context()` | GPU 编译 | Google TPU（远期讨论项） | C |
| Apple Silicon | `mps_context()` | 无 | MPS / ANE | C |
| 并行分解 | ParallelPass 实际逻辑 | MIR | 脚手架已存在；自动并行化 | C |
| 内存池 | `GpuMemoryPool::new()` | CUDA 后端 | GPU 内存管理 | C |
| 流/异步执行 | `cuda_stream()` / `await_stream(s)` | CUDA 后端 | CUDA streams | C |
| 多 GPU | `data_parallel(model, devices)` | CUDA 后端、分布式 | 数据/模型并行 | C |
| NCCL | `nccl_all_reduce(t)` | 多 GPU | 多 GPU 通信 | C |
| 内核自动调优 | `autotune(kernel, inputs)` | CUDA 后端 | autotuning | C |
| 图优化 | `compile_graph(graph)` | MIR | 计算图编译优化 | C |

### 5.7 分布式

| 能力 | 目标签名 | 依赖 | 实现要点 | 优先级 |
|------|---------|------|---------|--------|
| SPMD | `shard`/`node` 关键字语义 | 多 GPU | 单程序多数据（关键字已存在） | C |
| 数据并行 | `ddp(model, devices)` | 多 GPU | DDP | C |
| 模型并行 | `tensor_parallel(model)` / `pipeline_parallel(model)` | 多 GPU | 张量/流水线并行 | C |
| 参数服务器 | `ParameterServer::new()` | RPC | PS 架构 | C |
| All-Reduce | `all_reduce(t) -> Tensor` | NCCL | 集合通信 | C |
| RPC | `rpc_call(node, method, args)` | 网络 | 远程过程调用 | C |
| 分布式检查点 | `save_distributed(model, path)` | 模型保存/加载 | 分布式检查点 | C |
| 弹性训练 | `elastic_train(model, data)` | 分布式 | 故障恢复 | C |

### 5.8 AI 推理与部署

| 能力 | 目标签名 | 依赖 | 实现要点 | 优先级 |
|------|---------|------|---------|--------|
| 模型导出 | `export_model(model, path)` | 模型保存/加载 | 导出为独立可执行 | C |
| ONNX 导出 | `to_onnx(model, path)` | ONNX | 跨框架交换 | C |
| TorchScript 等价 | `script_model(model) -> ScriptModule` | 模型保存/加载 | 脚本化模型 | C |
| 推理服务器 | `serve_model(model, addr)` | HTTP 服务端 | HTTP/gRPC 推理 API | C |
| 模型量化推理 | `quantize_for_inference(model)` | 量化张量 | int8 推理 | C |
| 批处理推理 | `dynamic_batch(server, max_batch)` | 推理服务器 | dynamic batching | C |
| 模型版本管理 | `ModelRegistry::new()` | 模型保存/加载 | model registry | C |
| A/B 测试 | `ab_test(model_a, model_b, traffic)` | 推理服务器 | 模型对比 | C |
| 边缘部署 | `deploy_edge(model, target)` | 模型导出 | 移动端/嵌入式 | C |
| WebAssembly 推理 | `wasm_infer(model, input)` | WASM 后端 | 有 WASM 但未用于推理；浏览器端推理 | C |

---

## 六、通用编程生态（Tenth 作为通用语言）

### 6.1 Web 与网络服务

| 能力 | 目标签名 | 依赖 | 实现要点 | 优先级 |
|------|---------|------|---------|--------|
| HTTP 服务端框架 | 见 4.5.HTTP 服务端 | — | 同 4.5（去重引用）；上层路由/中间件在此扩展 | B |
| HTTP 客户端 | 见 4.5.HTTP 客户端 | — | 同 4.5（去重引用）；上层请求库 API 在此扩展 | B |
| WebSocket | 见 4.5.WebSocket | — | 同 4.5（去重引用） | C |
| REST API | RESTful 路由约定 | HTTP 服务端框架 | REST 约定 + 路由 | C |
| GraphQL | `graphql_server(schema)` | HTTP 服务端框架 | GraphQL 查询语言 | C |
| 模板引擎 | `render(template, ctx) -> str` | 无 | HTML 模板渲染 | C |
| 静态文件服务 | `serve_static(dir)` | HTTP 服务端框架 | 静态文件中间件 | C |
| Cookie/Session | `session(req) -> Session` | HTTP 服务端框架 | 会话管理 | C |
| CORS | `cors_middleware(server)` | HTTP 服务端框架 | 跨域处理中间件 | C |
| 文件上传 | `multipart_handler(req)` | HTTP 服务端框架 | multipart 解析 | C |

### 6.2 数据库

| 能力 | 目标签名 | 依赖 | 实现要点 | 优先级 |
|------|---------|------|---------|--------|
| SQLite | `sqlite_open(path) -> DbConnection` | FFI | 嵌入式数据库（rusqlite 桥接） | B |
| PostgreSQL 驱动 | `pg_connect(addr) -> PgConn` | TCP 客户端 | Postgres 协议 | C |
| MySQL 驱动 | `mysql_connect(addr) -> MySqlConn` | TCP 客户端 | MySQL 协议 | C |
| Redis 客户端 | `redis_connect(addr) -> RedisClient` | TCP 客户端 | RESP 协议 | C |
| MongoDB 驱动 | `mongo_connect(addr) -> MongoConn` | TCP 客户端 | Mongo 协议 | C |
| ORM | `#[table] struct User {}` / `User::find(id)` | SQLite | 对象关系映射 | C |
| 连接池 | `Pool::new(conn_factory, size)` | 数据库驱动 | 连接池 | C |
| 迁移工具 | `migrate(dir)` | 数据库驱动 | schema 版本管理 | C |
| 向量数据库 | `vector_db::new(dim)` | 无 | AI 向量检索 | C |

### 6.3 GUI 与图形

| 能力 | 目标签名 | 依赖 | 实现要点 | 优先级 |
|------|---------|------|---------|--------|
| 终端 UI | `Tui::new() -> Tui` | 无 | TUI 框架（ratatui 桥接） | C |
| 桌面 GUI | `Window::new(title)` / Qt/GTK 绑定 | FFI | 桌面窗口（远期讨论项） | C |
| 图形渲染 | `gl_context()` | FFI | OpenGL/Vulkan/Metal | C |
| 图表绘制 | `plot(data) -> Image` | 无 | matplotlib 等价 | C |
| 图像处理 | `image_load(path) -> Image` | 无 | PIL/OpenCV 等价 | C |
| 音频处理 | `audio_load(path) -> Audio` | 无 | 音频解码/处理 | C |
| 视频处理 | `video_load(path) -> Video` | 无 | 视频解码/处理 | C |
| 游戏引擎 | `Engine::new()` | 图形渲染 | 游戏引擎 | C |

### 6.4 系统编程

| 能力 | 目标签名 | 依赖 | 实现要点 | 优先级 |
|------|---------|------|---------|--------|
| FFI | 见 三.FFI 外部函数接口 | — | 同 三（去重引用） | B |
| 内联汇编 | 见 三.内联汇编 | — | 同 三（去重引用） | C |
| 原始指针 | `*const T` / `*mut T` | 无 | 裸指针类型 + unsafe | C |
| 内存布局控制 | `#[repr(C)]` | 无 | 类型布局属性 | C |
| 位操作 | `bitflags!` / 位域 | 无 | bitflags 宏 + 位域 | C |
| 系统调用 | `syscall(num, args)` | FFI | syscall 封装 | C |
| 共享内存 | `shm_open(name, size)` | 无 | shm | C |
| 进程间通信 | `ipc_send(ch, msg)` / `ipc_recv(ch)` | 管道 | pipe/shared memory | C |
| 守护进程 | `daemonize()` | 无 | daemon 化 | C |
| systemd 服务 | `systemd_unit(name)` | 无 | systemd 集成 | C |

### 6.5 脚本与自动化

| 能力 | 目标签名 | 依赖 | 实现要点 | 优先级 |
|------|---------|------|---------|--------|
| Shell 调用 | `exec("ls -la") -> str` | 执行子进程 | Shell 调用封装（子进程封装层，非通用必需） | C |
| 管道操作 | `cmd1 \| cmd2` | 执行子进程 | 命令管道 | C |
| 文件 glob | `glob("*.txt") -> Vec<Path>` | 无 | 通配符匹配 | C |
| 正则替换 | `re_replace(pattern, input, replacement)` | 正则表达式 | sed 等价 | C |
| 定时任务 | `cron(expr, fn)` | async runtime | cron 等价 | C |
| 文件监听 | `watch(path, cb)` | 文件监听 | watch | C |
| 环境管理 | `venv_create(path)` | 无 | venv 等价 | C |

---

## 七、语言生态与社区

| 能力 | 目标签名 | 依赖 | 实现要点 | 优先级 |
|------|---------|------|---------|--------|
| API 参考 | 标准库文档生成 | 文档生成 | 有 prelude 但无生成文档；接文档生成工具 | C |
| 教程 | 入门指南 | 无 | 官方教程撰写 | C |
| 包注册中心 | 中央仓库 | 包注册中心 | 服务端 + 索引 | C |
| 官方网站 | tenth-lang.org | 无 | 官网建设 | C |
| 社区论坛 | Discord/Forum | 无 | 社区平台 | C |
| 变更日志 | CHANGELOG | 无 | 版本变更日志 | C |
| VSCode 扩展 | 语法高亮 + LSP 集成 | LSP 服务器、语法高亮 | VSCode 扩展打包 | C |
| JetBrains 插件 | IntelliJ/CLion 插件 | LSP 服务器 | JetBrains 平台插件 | C |
| 在线 Playground | Web 端试用 | 在线 Playground | 浏览器端运行 | C |
| 语言规范 | 形式化规范 | 无 | 仅有手册；形式化规范 | C |
| 一致性测试套件 | conformance tests | 无 | 语言一致性测试 | C |
| 性能基准 | benchmark suite | 无 | 性能基准集 | C |
| 兼容性矩阵 | 平台支持表 | 无 | 平台支持表 | C |

---

## 八、平台支持

| 能力 | 目标签名 | 依赖 | 实现要点 | 优先级 |
|------|---------|------|---------|--------|
| Linux aarch64 | 交叉编译验证 | 无 | aarch64 构建验证（仅构建验证，非交叉编译依赖） | C |
| macOS x86_64 | macOS 构建 | 无 | Intel macOS 验证 | C |
| macOS aarch64 | Apple Silicon 构建 | 无 | Apple Silicon 验证 | C |
| WebAssembly（浏览器运行时） | 浏览器运行时 | WASM 后端 | 可编译到 WASM 但无浏览器运行时；补运行时 | C |
| Android | Android 平台 | 无 | Android 支持（远期讨论项） | C |
| iOS | iOS 平台 | 无 | iOS 支持（远期讨论项） | C |
| 嵌入式 / RTOS | 嵌入式平台 | 无 | 嵌入式 / RTOS（远期讨论项） | C |

---

## 优先级统计

| 优先级 | 含义 | 表内项数 | 实质项数（去重后） |
|--------|------|---------|------------------|
| A | 生存级（第一梯队，不做就无法称为"AI 语言"） | 12 | 12 |
| B | 通用必需（第二梯队，不做就无法称为"通用语言"） | 45 | 43 |
| C | 生态远期（第三梯队及以下） | 339 | 330 |
| 去重引用项 | 标"见 x.x"的副项 | 11 | 0（已计入主项） |
| **合计** | | **396** | **385** |

> 说明：本统计以 能力全梳理.md 各小节详细表的 ❌/⚠️ 条目逐条统计为准（已排除"LLVM 后端"；已删除"GC 垃圾回收"、"goto"——项目策略明确不做）。能力全梳理.md 第九节"总结矩阵"为粗略估算，本文档以逐条统计为权威。文档对 11 处重复项采用"主项+引用项"模式去重，故"表内项数"包含 11 个引用项，"实质项数"为去重后真实能力数。

### A 级清单（生存级，12 项）

1. 编译期 Shape 检查（1.2）
2. 符号维度（1.2）
3. 秩多态（1.2）
4. Const 泛型（1.2）
5. yield / 协程（1.3，含绿色线程，原"协程/绿色线程"已合并）
6. async / await（1.4）
7. GPU 编译（2.2，含算子融合）
8. 线程 / 并发（三）
9. async runtime（三）
10. 通道 / 消息传递（三）
11. 互斥锁 / 读写锁（三）
12. CUDA 后端 + 算子融合（5.6，与 2.2 GPU 编译互为表里）

> 注：原 A 级 14 项经合并后为 13 项——"协程/绿色线程"并入"yield/协程"（同属状态机改写+暂停恢复），"算子融合"并入"CUDA 后端"（kernel 融合是 GPU 编译的子任务）。

### B 级清单（通用必需，43 项，按小节分布）

- 1.2：泛型枚举（显式 `<T>`）、元组类型
- 1.3：try/catch 异常、try 表达式/`?` 操作符
- 2.2：JIT 编译
- 三：原子操作、FFI 外部函数接口
- 4.1：数组、HashSet、Range 类型、迭代器 trait、字符/Char
- 4.2：split/trim/replace、正则表达式、字符串格式化
- 4.4：文件流/缓冲 I/O、标准输入
- 4.5：TCP 客户端、TCP 服务端、HTTP 客户端
- 4.6：日期解析、日期格式化
- 4.7：环境变量、执行子进程
- 5.1：多精度张量（f16/bf16）、混合精度训练、广播规则完善、张量比较、Where/Select
- 5.2：梯度累积、梯度裁剪、自定义算子
- 5.3：MaxPool/AvgPool、学习率调度器、早停/检查点
- 5.4：学习率调度、预热策略（梯度裁剪引用 5.2，不计入）
- 6.1：HTTP 服务端框架、HTTP 客户端（均为 4.5 引用项，不计入）
- 6.2：SQLite
- 6.4：FFI（引用 三，不计入）

> 注：原 B 级中的 AdamW、Transformer Decoder、HTTP 服务端、Shell 调用经评估降为 C 级（单个优化器/AI 库扩展/封装层，非通用必需）。

> 其余 ~330 项为 C 级（生态远期），含 GUI、加密细节、序列化远期格式、平台扩展、生态工具、AI 推理部署、分布式等远期讨论项。
