# Tenth 语言 — Code Wiki

> **Tenth** = Tensor + Zenith，意为「张量之巅」—— 一门为 AI 研究而生的编程语言
>
> 当前版本：**v1.0.0** | 语言实现：Rust | 许可证：MIT

---

## 目录

1. [项目概述](#1-项目概述)
2. [整体架构](#2-整体架构)
3. [编译流水线](#3-编译流水线)
4. [主要模块详解](#4-主要模块详解)
   - 4.1 [词法分析器 (lexer)](#41-词法分析器-lexer)
   - 4.2 [语法分析器 (parser)](#42-语法分析器-parser)
   - 4.3 [高级中间表示 (hir)](#43-高级中间表示-hir)
   - 4.4 [编译后端 (compile)](#44-编译后端-compile)
   - 4.5 [运行时 (runtime)](#45-运行时-runtime)
   - 4.6 [REPL (repl)](#46-repl-repl)
5. [自举编译器 (tenthc)](#5-自举编译器-tenthc)
6. [标准库 (std)](#6-标准库-std)
7. [示例集 (Tenth实例)](#7-示例集-tenth实例)
8. [依赖关系](#8-依赖关系)
9. [项目运行方式](#9-项目运行方式)
10. [路线图与状态](#10-路线图与状态)

---

## 1. 项目概述

Tenth 是一门面向 AI/ML 研究的编程语言，核心特性包括：

- **张量级自动微分**：内置 `TapeOp` 枚举中全部算子的反向传播全链路（清单见 `docs/语言参考手册.md` §11.5，SSOT＝`tenth/src/runtime/autodiff/tape_op.rs`；本文件不硬编码计数），支持 `new_grad()` / `param()` / `backward()` / `grad()` 等控制函数
- **双执行引擎**：字节码 VM（默认；指令集口径＝`tenth/src/runtime/vm/op.rs` 的 `enum Op` 变体数，本文件不硬编码计数）+ 树遍历解释器（fallback）
- **WASM 编译**：通过 `wasm-encoder` 生成 WASM 字节码，`wasmi` 执行验证
- **自举编译器**：用 Tenth 自身编写的编译器（`tenthc/`），经三阶段验证闭环
- **闭包捕获**：闭包自动捕获外层作用域变量，支持单变量和多变量捕获
- **文件级导入**：`use` 语句自动搜索 `std/` 目录下的 `.th` 文件
- **类 Rust 语法**：支持 `struct` / `enum` / `match` / `trait` / `impl` / 泛型 / 引用与移动语义

### 项目目录结构

```
项目根目录/
├── .github/                # CI
│   └── workflows/
│       ├── release.yml    # 三平台构建 + 冒烟 + 发布（无基准步骤）
│       └── perf.yml       # 性能基线巡检（workflow_dispatch + 周跑；仅报告不阻塞）
├── scripts/                # 发布/校验脚本
│   ├── release/            # package.ps1 / package.sh / verify.ps1 / sync-public.ps1
│   └── check_local_path_leaks.py  # 本地路径泄漏检查
├── tenth/                  # 主编译器与运行时（Rust 实现）
│   ├── src/                # 源码
│   │   ├── lexer/          # 词法分析
│   │   ├── parser/         # 语法分析
│   │   ├── hir/            # 高级中间表示（hir.rs / mod.rs / types.rs）
│   │   │   ├── hir.rs
│   │   │   ├── mod.rs
│   │   │   ├── types.rs
│   │   │   └── lower/      # HIR lowering（lower_expr.rs / lower_stmt.rs 等）
│   │   ├── compile/        # 编译后端（字节码 + WASM + JIT + GPU + 优化 Pass）
│   │   │   ├── mod.rs
│   │   │   ├── bytecode.rs # HIR→字节码编译器
│   │   │   ├── wasm/       # HIR→WASM 编译器
│   │   │   ├── jit/        # Cranelift JIT（默认路径）
│   │   │   ├── bridge.rs   # 自举编译器桥接
│   │   │   ├── gpu/        # GPU 后端脚手架
│   │   │   │   ├── mod.rs  # GpuBackend / GpuConfig / GpuCompiler / GpuProgram
│   │   │   │   ├── cuda_kernel.rs  # CudaKernel + elementwise/reduce 模板
│   │   │   │   └── device.rs       # Device trait + CpuDevice / CudaDevice
│   │   │   └── optimizations/      # 编译优化 Pass
│   │   │       ├── mod.rs  # OptimizationPass trait
│   │   │       ├── fusion.rs       # FusionPass 算子融合
│   │   │       └── parallel.rs     # ParallelPass 自动并行
│   │   ├── runtime/        # 运行时（解释器 + VM + 张量 + 自动微分）
│   │   │   ├── vm/         # 字节码 VM
│   │   │   ├── interpreter/  # 树遍历解释器
│   │   │   ├── tensor/     # 张量
│   │   │   ├── autodiff/   # 自动微分
│   │   │   └── native_registry.rs / relation_debugger.rs / async_io.rs
│   │   ├── error.rs        # 统一错误类型
│   │   ├── lib.rs          # 库入口
│   │   ├── main.rs         # CLI 入口
│   │   └── repl.rs         # REPL 交互环境
│   ├── std/                # Tenth 标准库（.th 文件）
│   ├── tests/              # 集成测试
│   ├── tools/              # 生态工具
│   │   ├── tenthpm/        # tenthpm 包管理器
│   │   │   ├── Cargo.toml  # 依赖：serde, serde_json, toml
│   │   │   └── src/
│   │   │       ├── main.rs # CLI 入口
│   │   │       ├── manifest.rs # Tenth.toml 解析
│   │   │       └── commands/   # 子命令（init/build/run/publish）
│   │   ├── lsp/            # LSP 服务器
│   │   │   ├── Cargo.toml  # 依赖：serde, serde_json, tenth (path)
│   │   │   └── src/
│   │   │       ├── main.rs # LSP 入口
│   │   │       ├── lsp_types.rs # LSP 协议类型
│   │   │       ├── io.rs   # stdio 通信
│   │   │       └── handlers/ # 请求处理器
│   │   ├── debugger/       # 调试器
│   │   └── profiler/       # 剖析器
│   ├── Cargo.toml          # Rust 项目配置
│   └── build.rs            # 构建脚本
├── tenthc/                 # 自举编译器（Tenth 编写）
│   ├── main.th             # 入口（拼接各模块源码后调用 compile_host）
│   ├── lexer/              # Tenth 实现的词法分析
│   ├── parser/             # Tenth 实现的语法分析
│   ├── hir/                # Tenth 实现的 HIR
│   └── compile/            # Tenth 实现的 WASM 编译
├── Tenth实例/              # 63 个实例目录（72 个 .th）
├── docs/                   # 文档
│   ├── 语言参考手册.md
│   └── superpowers/plans/  # 开发计划
├── dist/                   # 1.0 发布产物（zip + 解压目录 + SHA256SUMS.txt，gitignored）
├── README.md
├── DEPS.md                 # 依赖说明
├── MEMO.md                 # 开发备忘录
└── AUDIT.md / SECURITY.md / CONTRIBUTING.md / CODE_OF_CONDUCT.md
```

---

## 2. 整体架构

Tenth 采用经典的多阶段编译架构，执行流程如下：

```
源代码 (.th)
    │
    ▼
┌─────────┐    ┌─────────┐    ┌─────────┐
│  Lexer  │───▶│ Parser  │───▶│ Lowerer │
│ 词法分析 │    │ 语法分析 │    │ HIR 生成 │
└─────────┘    └─────────┘    └─────────┘
    │              │               │
    ▼              ▼               ▼
  Token[]        AST           HirProgram
                                  │
                    ┌─────────────┼─────────────┬────────────┐
                    ▼             ▼              ▼            ▼
            ┌──────────┐  ┌──────────┐   ┌──────────┐  ┌───────────┐
            │ Bytecode │  │Interpreter│   │   WASM   │  │    JIT    │
            │ Compiler │  │(树遍历)   │   │ Compiler │  │ (Cranelift)│
            └──────────┘  └──────────┘   └──────────┘  └───────────┘
                    │             │              │            │
                    ▼             ▼              ▼            ▼
              ┌────────┐   ┌──────────┐   ┌──────────┐  ┌───────────┐
              │   VM   │   │ 直接执行  │   │ wasmi    │  │ 原生机器码 │
              │(栈式VM)│   │          │   │ 执行验证  │  │ 热点函数   │
              └────────┘   └──────────┘   └──────────┘  └───────────┘
```

**执行优先级**：VM 优先 → 解释器 fallback。JIT 为热点路径（对满足特化条件的 chunk 由 Cranelift 生成机器码，失败/不支持时回退 VM）。VM 已支持 for-in 循环、闭包调用、字符串切片、张量字面量（MakeTensor）和闭包创建（MakeClosure）。

---

## 3. 编译流水线

### 3.1 完整流水线

```rust
// main.rs 中的核心管线函数
fn source_to_hir(source: &str) -> TenthResult<HirProgram> {
    let tokens = Lexer::new(source).tokenize()?;        // 源码 → Token
    let program = Parser::new(tokens).parse_program()?; // Token → AST
    let hir = Lowerer::with_search_paths(vec!["std".into()])  // AST → HIR（含文件导入）
        .lower_program(&program)?;
    Ok(hir)
}
```

### 3.2 CLI 子命令

| 命令 | 说明 |
|------|------|
| `tenth` (无参数) | 启动 REPL |
| `tenth run <file.th>` | 编译并执行 .th 文件（VM 优先，解释器 fallback） |
| `tenth build <file.th>` | 编译 .th 为 .wasm 文件 |
| `tenth wasm <file.th>` | 编译为 WASM 并通过 wasmi 执行 |
| `--max-memory <MiB>` | 设置内存上限（REPL 模式） |

---

## 4. 主要模块详解

### 4.1 词法分析器 (lexer)

**位置**：`tenth/src/lexer/`

| 文件 | 职责 |
|------|------|
| `token.rs` | Token 类型定义（`TokenKind` 枚举 + `Span` 位置信息） |
| `lexer.rs` | 词法分析器实现 |

#### 关键类型

```rust
// token.rs — Token 类型
pub enum TokenKind {
    IntLiteral(i64), FloatLiteral(f64), StringLiteral(String), CharLiteral(char),
    Identifier(String),
    // 关键字：Fn, Let, Mut, If, Else, Match, For, While, Loop, Break, Continue,
    //         Return, Try, Use, Mod, Pub, Trait, Impl, Enum, Struct, Type, Self_,
    //         Spawn, Task, Shard, Node, Macro, Where, As, In, True, False, Move
    // 运算符：Plus, Minus, Star, Slash, Percent, EqEq, NotEq, Lt, Gt, ...
    // 分隔符：LParen, RParen, LBracket, RBracket, LBrace, RBrace, ...
    Eof,
}

pub struct Span { pub line: usize, pub col: usize }
pub struct Token { pub kind: TokenKind, pub span: Span }
```

#### 关键函数

| 函数 | 签名 | 说明 |
|------|------|------|
| `Lexer::new` | `(source: &str) -> Self` | 创建词法分析器 |
| `tokenize` | `(&mut self) -> TenthResult<Vec<Token>>` | 完整词法分析 |
| `next_token` | `(&mut self) -> TenthResult<Token>` | 获取下一个 Token |

#### 特性

- 支持行注释 `//` 和嵌套块注释 `/* */`
- 支持数字字面量中的 `_` 分隔符
- 字符串支持转义字符：`\n`, `\r`, `\t`, `\\`, `\"`
- 位置信息追踪（行号 + 列号）

---

### 4.2 语法分析器 (parser)

**位置**：`tenth/src/parser/`

| 文件 | 职责 |
|------|------|
| `ast.rs` | AST 节点定义 |
| `parser.rs` | 递归下降语法分析器 |

#### AST 核心类型

```rust
// 表达式
pub enum ExprKind {
    Literal(Literal), Ident(Ident),
    Binary { op, left, right }, Unary { op, expr },
    Call { func, args }, GenericCall { func, generics, args },
    MethodCall { receiver, method, args },
    Index { target, indices }, Field { target, field },
    TensorLiteral(Vec<Vec<Expr>>), ArrayLiteral(Vec<Expr>),
    Range { start, end, inclusive },
    If { cond, then_branch, else_branch },
    Block(Vec<Stmt>), Closure { params, body },
    Assign { target, value }, AssignOp { target, op, value },
    StructLiteral { name, generics, fields, use_defaults },  // use_defaults: .. 语法填充默认值
    EnumLiteral { enum_name, variant, fields },              // fields: Struct / Tuple / Unit
    Match { scrutinee, arms },
    Ref(Box<Expr>), MutRef(Box<Expr>), Deref(Box<Expr>), Move(Box<Expr>),
}

// 语句
pub enum StmtKind {
    Let { name, type_ann, mutable, init },
    Expr(Expr), Return(Option<Expr>), Break, Continue,
    While { cond, body }, For { var, iter, body }, Loop { body },
}

// 顶层项
pub enum ItemKind {
    Function { name, generics, params, return_type, body },
    StructDef { name, generics, fields },
    EnumDef { name, variants },  // variants: EnumVariantKind (Struct / Tuple / Unit)
    Impl { type_name, trait_name, generics, functions },
    Mod { name, items }, Use { path },
    Trait { name, generics, methods },
}
```

#### 解析策略

- **递归下降** + **运算符优先级**（Pratt parser 风格）
- 优先级表：`* / %`(5) > `+ -`(4) > `< > <= >=`(3) > `== !=`(2) > `&&`(1) > `||`(0)
- 自动识别结构体字面量（`Name { field: value }`）和枚举变体（`Enum::Variant`）
- 支持泛型调用语法：`func::<Type>(args)` 或 `func<Type>(args)`

---

### 4.3 高级中间表示 (hir)

**位置**：`tenth/src/hir/`

| 文件 | 职责 |
|------|------|
| `types.rs` | 类型系统定义 |
| `hir.rs` | HIR 节点定义 |
| `lower.rs` | AST → HIR 降低器（含类型推断与借用检查） |

#### 类型系统

```rust
pub enum Type {
    Base(BaseType),                              // 基础类型：i8..i64, u8..u64, f16..f64, bool, char, str, Unit
    Tensor { dtype: BaseType, dims: Vec<Dim> },  // 张量类型：Tensor[f64, N, M]
    Array(Box<Type>),                            // 数组类型
    FnType { params: Vec<Type>, ret: Box<Type> },// 函数类型
    TypeParam { name: String },                  // 类型参数
    Generic { base: Box<Type>, args: Vec<Type> },// 泛型类型
    Ref(Box<Type>),                              // 不可变引用 &T
    MutRef(Box<Type>),                           // 可变引用 &mut T
    Struct(String),                              // 结构体
    Enum(String),                                // 枚举
    Unknown,                                     // 未知/待推断
}

pub enum Dim { Known(i64), Symbol(String), Any }  // 维度规格
```

#### HIR 程序结构

```rust
pub struct HirProgram {
    pub functions: Vec<HirFnDef>,                     // 普通函数
    pub generic_funcs: Vec<HirFnDef>,                 // 泛型函数
    pub main_expr: Option<HirExpr>,                   // 顶层表达式
    pub modules: HashMap<String, HirProgram>,         // 模块
    pub uses: Vec<(Vec<String>, String)>,             // use 导入
    pub methods: HashMap<String, HashMap<String, HirFnDef>>,  // impl 方法
    pub structs: HashMap<String, Vec<(String, Type)>>,        // 结构体定义（字段含 has_default 标记）
    pub generic_structs: HashMap<String, HirGenericStruct>,    // 泛型结构体
    pub enums: HashMap<String, Vec<(String, Vec<(String, Type)>)>>, // 枚举定义（变体含 tuple_binds）
    pub trait_defs: HashMap<String, HirTraitDef>,             // trait 定义
    pub trait_impls: HashMap<String, HashMap<String, HashMap<String, HirFnDef>>>, // trait 实现
}
```

#### Lowerer 核心功能

| 功能 | 说明 |
|------|------|
| **类型推断** | 根据字面量、运算符、函数签名推断表达式类型 |
| **借用检查** | 追踪变量的所有权状态（Owned / SharedRef / ExclusiveRef / Moved） |
| **作用域管理** | 嵌套作用域链，支持变量查找和遮蔽 |
| **闭包捕获分析** | `free_vars_in()` 递归分析闭包自由变量，自动捕获外层作用域 |
| **文件级导入** | `search_paths` + `try_import_file()` 自动发现和加载 .th 文件 |
| **内置函数注册** | 预注册 `println`, `tensor`, `rand`, `randn`, `backward`, `grad` 等 |
| **预加载类型** | 内置 `Option`, `Result` 枚举和 `Display`, `Eq`, `Clone` trait |

---

### 4.4 编译后端 (compile)

**位置**：`tenth/src/compile/`

| 文件 | 职责 |
|------|------|
| `mod.rs` | 编译入口，提供 `compile_to_wasm` / `run_wasm` / `compile_program_to_wasm` |
| `bytecode.rs` | HIR → 字节码编译器 |
| `wasm.rs` | HIR → WASM 字节码编译器 |
| `bridge.rs` | 自举编译器输出 → Rust AST 转换桥 |
| `gpu/mod.rs` | GPU 后端入口：GpuBackend / GpuConfig / GpuCompiler / GpuProgram |
| `gpu/cuda_kernel.rs` | CUDA 内核生成：CudaKernel + elementwise/reduce 模板 |
| `gpu/device.rs` | 设备抽象：Device trait + CpuDevice / CudaDevice |
| `optimizations/mod.rs` | 优化 Pass 入口：OptimizationPass trait |
| `optimizations/fusion.rs` | 算子融合：FusionPass（合并连续算子减少内存带宽） |
| `optimizations/parallel.rs` | 自动并行：ParallelPass（识别独立算子并行执行） |

#### 字节码 VM 指令集（`enum Op` 全量）

> **口径（as-of 2026-09-17）**：下列清单**逐行抄录** `tenth/src/runtime/vm/op.rs` 的 `pub enum Op`（`:8`–`:57`），按源码原有的分组注释排列；变体个数以**该源码文件**为 SSOT，本文件**不硬编码计数**（历史版本曾写“65 条”与“节选 45/65”，均已下线）。注释为摘要，完整注释见源码。

```rust
pub enum Op {
    // 常量压栈（PushInt 携带 dtype tag，见 AUDIT-11.4.53）
    PushInt(i64, crate::hir::types::BaseType), PushFloat(f64), PushFloat32(f32), PushBool(bool), PushChar(u32), PushStr(usize), PushUnit,
    // 栈操作
    Pop, Dup,
    // 局部变量
    Load(usize), Store(usize),
    // 全局变量
    LoadGlobal(usize), StoreGlobal(usize),
    // 算术运算
    Add, Sub, Mul, Div, Mod, Neg, Not,
    // 比较运算
    Eq, Neq, Lt, Gt, Lte, Gte,
    // 控制流
    Jump(i32), JmpFalse(i32), JmpTrue(i32),
    // 函数调用
    Call(usize), CallN(usize, usize), MethodCall(usize, usize), Ret,
    // 数据结构
    MakeVec(usize), MakeMap(usize),
    NewStruct(usize, usize), LoadField(usize), StoreField(usize),
    NewUnion(usize, usize),        // M1.2：union 构造 — name_idx, active_field_idx
    IndexGet,
    SliceStr,
    MakeEnum(usize, usize, usize),
    IsEnumVariant(usize),
    EnumGetField(usize),
    IsStruct(usize),
    PushRange(i64, i64, bool),     // start, end, inclusive
    MoveOp,                        // no-op marker for move semantics
    MakeTensor(usize, usize, u8),  // rows, cols, dtype (0=F64, 1=F32) — pops rows*cols values
    MakeClosure(usize, usize, usize), // params_count, captures_count, chunk_idx
    Await,
    Spawn,
    MakeTuple(usize),              // n — pops n values, pushes Value::Tuple
    IsTuple(usize),                // expected_len — pops value, pushes Bool(is Tuple with len)
    TupleGet(usize),               // index — pops Tuple, pushes element at index
    Try,                           // pops Result; Ok(v) → push v; Err(e) → early return TryPropagate(e)
    Yield,                         // 协作式调度：让出控制权，当前 task 回到 ready_queue 尾部
    TailCall(usize, usize),        // TCO：函数名索引 + 参数数量 — 复用当前帧替换 PC 和 slot
    // a1（VM 闭包值调用 / CallIndirect）P1：编码 57/58 追加在尾部
    CallClosure(usize),            // 参数数量 n — 弹 callee + N 参数 → 压新帧调用
    TailCallClosure(usize),        // TCO：同上但复用当前帧；JIT 不支持 → fallback VM
    // AUDIT-11.4.21（&mut 写回顺序失效）：引用语义 opcodes（编码 59-62 追加在尾部）
    MakeRef,                       // 弹值 → Value::Ref(Rc(RefCell(v)))
    MakeMutRef(usize),             // slot — locals[slot] 包装/复用 Shared 回写槽位，压 Value::MutRef(Weak)
    Deref,                         // 弹值；Ref/MutRef 读穿，其他透传（VM 宽松）
    DerefStore,                    // 栈 [value, target] — target 为 MutRef/Ref 写穿，其他报错
    // M1-S2（true letrec）：递归闭包自引用 cell opcodes（编码 63/64 追加在尾部）
    MakeCell,                      // 压 Value::Shared(Rc(RefCell(Unit)))——自引用占位 cell
    BindSelfCapture(usize),        // 弹 FnRef → 其 captures[k] 为 Shared cell → 写入自身 → 压回
}
```

> **2026-09-17（`AUDIT-11.4.53` 修复）**：`PushInt` 由 `i64` 扩为 **`(i64, BaseType)`**——整型值在**字节码层携带 dtype tag**（编码 **8B 值 + 1B tag**），此前 dtype 在 `compile/bytecode.rs` 被丢弃、`vm/execute.rs` 又补回硬编码 `I32`，导致「整型算术被硬限 i32」。配套：注解驱动 dtype 入口 **`hir/lower/types.rs::coerce_int_dtype`**；整数混合提升（宽度优先、可交换）在 **`runtime/value.rs::promote_int_dtype`** + VM/解释器算术入口；JIT 标量槽只承载 `Int`/f64（`ChunkSig` 特化仅 `Int`/f64，i64 注解走通用 ABI——见 `AUDIT.md` `P-11`）。`tenthc` **无需同步**（无字节码后端）。

#### BytecodeCompiler 关键方法

| 方法 | 说明 |
|------|------|
| `compile(func: &HirFnDef)` | 编译函数为 Chunk |
| `compile_main(expr: &HirExpr)` | 编译顶层表达式为 Chunk |
| `compile_expr(expr: &HirExpr)` | 递归编译表达式 |
| `compile_stmt(stmt: &HirStmt)` | 编译语句 |
| `resolve_patches()` | 解析跳转标签回填 |

#### WASM 编译

- 使用 `wasm-encoder` crate 生成 WASM 二进制
- 通过 `wasmi` 解释器执行和验证
- 支持函数导出和 import

#### GPU 后端（v1.0.0 — CUDA C 源代码生成 + 模拟设备，未接 CUDA Runtime）

GPU 后端为 Phase 4 铺路。**当前状态**：仅生成 CUDA C 源代码字符串 + 模拟设备抽象，**未接 nvcc / CUDA Runtime API / cuLaunchKernel**，不编译、不加载、不执行任何 kernel。`CudaDevice::is_available()` 永远返回 `true`（注释自承 "Simulated"），`total_memory` 硬编码 24GB。详见 `AUDIT.md` §11.4 AUDIT-11.4.6。

```rust
// gpu/mod.rs — GPU 后端核心结构
pub enum GpuBackend { Cpu, Cuda }
pub struct GpuConfig { backend: GpuBackend, device_id: usize, max_threads_per_block: usize }
pub struct GpuCompiler { config: GpuConfig }
pub struct GpuProgram { kernels: Vec<CudaKernel>, config: GpuConfig }
// GpuCompiler::compile_kernel 仅遍历 program.functions 调 CudaKernel::from_hir_function
// 把每个 HIR 函数转成 CudaKernel 字符串，不调用 nvcc、不生成 .ptx

// gpu/cuda_kernel.rs — CUDA 内核模板（生成 CUDA C 源代码字符串）
pub struct CudaKernel { name: String, params: Vec<String>, body: String }
// to_cuda_code() 返回 String，是 CUDA C 源代码文本，非编译产物

// gpu/device.rs — 设备抽象（模拟设备，非真实 CUDA 设备）
pub trait Device {
    fn name(&self) -> &str;
    fn device_type(&self) -> GpuBackend;
    fn memory_limit(&self) -> usize;
    fn is_available(&self) -> bool;   // CudaDevice 永远返回 true（Simulated）
}
pub struct CpuDevice { name: String, memory_limit: usize }   // 16 GB simulated
pub struct CudaDevice { device_id: usize, name: String, total_memory: usize, compute_capability: (u32, u32) }  // 24 GB simulated，is_available() 永远 true
```

#### JIT 后端（Cranelift）：M0 资格守卫与 `TENTH_JIT_POISON`

JIT 不维护 per-native 名表：native 调用按名透传（`host_call` → `vm.natives.contains_key`），因此**新增/改名 native 对 JIT 自动生效**（无需改 JIT 侧名单）。

**M0（`AUDIT-11.4.44` 的第一步）——资格守卫与毒化设施：**

- **D1 `Load` 资格守卫**（已落地）：`Load` 专用化前比对「分析预测的 `ScalarKind`」与「发射端标量槽种类」，两种不一致都**响亮 `Err`** → 走既有「整函数回退 VM」通路（功能正确，仅放弃本轮专用化）：
  - 槽种类 ≠ 分析种类（分析/发射漂移）
  - 分析预测该局部为具体标量，但发射端**没有**标量槽（原为**全 JIT 唯一无护栏点**，UB 本体：后续专用化读未初始化标量槽 → 宿主栈残留 → 随二进制布局翻转的静默错值）
- **D1 是防御性守卫**：`jit_silent_audit_test` 143 个目标全绿、未触发 ⇒ 已如实标注（不是"已修复一个活缺陷"，而是"补上了无护栏的点"）。
- **D2 `Store` 侧守卫：实测过度保守 ⇒ 未纳入**（见 `AUDIT-11.4.74`）：给 `Store` 补同型块入口守卫会**回退正确语料**（`audit_d2_dup_pop_block_boundary`）⇒ 宁缺勿滥，D2 的精确闭合依赖 M1–M5 单表化（归 W10）。
- **`TENTH_JIT_POISON`（调试设施，环境变量开启）**：入口毒化三层——① 虚拟栈区、② 全部局部 `Value` 槽、③ **标量伴随槽**（M0/D10 新补：此前标量槽在发射期**懒创建**，创建点可能落在运行期不执行的分支内，未写先读读到的是宿主栈残留、不可探测；现改为入口按上界预创建并写毒值 `0xAAAA…`，两个取槽函数在毒化模式下复用这些槽）。
  - **范围限制**：`spec_mode`（特化入口）**排除**标量槽池化——实测「池化 + 非零毒值」会使 `spec_f64_recursive_mixed` 以 `0xC0000005` 中止，且与毒值内容相关（`0`→绿、`42`/`2^40`→崩、不池化→绿）⇒ 该入口下的标量槽 UB **仍不可探测**（见 `AUDIT-11.4.75`）。

**剩余工作（M1–M5，排 W10）**：把「分析端资格判定」与「发射端资格判定」两处独立代码路径收敛为**单一资格表**（单一数据结构/函数）。现状一旦两端单独修改（新增 opcode 专用化、调整白名单）就可能再造同族静默错值（历史同族：`11.4.35` / `11.4.36` / `11.4.44`）。

#### 编译优化 Pass（v1.0.0 脚手架）

```rust
// optimizations/mod.rs — 优化 Pass trait
pub trait OptimizationPass {
    fn name(&self) -> &str;
    fn run(&self, program: &mut HirProgram) -> PassResult;
}

// optimizations/fusion.rs — 算子融合
pub struct FusionPass;
// 识别连续算子（如 relu → add）并融合为单一内核，减少内存带宽开销

// optimizations/parallel.rs — 自动并行
pub struct ParallelPass;
// 识别无依赖的独立算子，生成并行执行计划
```

---

### 4.5 运行时 (runtime)

**位置**：`tenth/src/runtime/`

| 文件 | 职责 |
|------|------|
| `value.rs` | 值类型定义 |
| `interpreter.rs` | 树遍历解释器 |
| `vm.rs` | 字节码栈式虚拟机 |
| `tensor.rs` | 张量运算库 |
| `autodiff.rs` | 自动微分引擎 |
| `arena.rs` | 内存池分配器 |
| `limits.rs` | 运行时资源限制 |

#### Value 类型

```rust
pub enum Value {
    Int(i64), Float(f64), Bool(bool), String(String),
    Tensor(Rc<RefCell<Tensor>>), Unit,
    Array(Rc<RefCell<Vec<Value>>>),
    FnRef { name, params, return_type },
    Closure { params, body, captures },
    Struct { name, fields: Rc<RefCell<Vec<(String, Value)>>> },
    Enum { enum_name, variant, fields: Rc<RefCell<Vec<(String, Value)>>> },
    Ref(Rc<RefCell<Value>>),        // 不可变引用
    MutRef(Weak<RefCell<Value>>),   // 可变引用（Weak 防循环）
    Shared(Rc<RefCell<Value>>),     // 共享引用（用于 Vec 元素可变性）
    Moved,                          // 已移动标记
    Vec(Rc<RefCell<Vec<Value>>>),   // 动态数组
    Map(Rc<RefCell<HashMap<String, Value>>>), // 哈希映射
    Range { start: i64, end: i64, inclusive: bool },
    Tuple(Vec<Value>),                  // 元组
}
```

#### Interpreter（树遍历解释器）

核心结构：

```rust
pub struct Interpreter {
    pub scopes: Vec<HashMap<String, Value>>,    // 作用域链
    functions: Vec<HirFnDef>,                   // 函数表
    generic_funcs: HashMap<String, HirFnDef>,   // 泛型函数表
    methods: HashMap<String, HashMap<String, HirFnDef>>, // 方法表
    modules: HashMap<String, HirProgram>,       // 模块表
    trait_impls: ...,                           // trait 实现
    pub limits: Option<RuntimeLimits>,          // 资源限制
    pub arena: Arena,                           // 内存池
    pub tape: Option<Tape>,                     // 自动微分磁带
    pub recording: bool,                        // 是否录制梯度
}
```

关键方法：

| 方法 | 说明 |
|------|------|
| `new(program)` | 从 HIR 程序创建解释器 |
| `with_limits(program, limits)` | 创建带资源限制的解释器 |
| `execute_program(&mut self, program)` | 执行 HIR 程序 |
| `eval_expr(&mut self, expr)` | 求值表达式 |
| `eval_stmt(&mut self, stmt)` | 执行语句 |
| `eval_binary(&mut self, op, l, r)` | 二元运算（含张量自动微分录制） |
| `eval_method_call(&mut self, recv, method, args)` | 方法调用分派 |
| `eval_tensor_method(&mut self, recv, method, args)` | 张量方法（sum/mean/matmul/relu/conv2d/batchnorm/dropout/softmax 等） |
| `call_named_fn(&mut self, name, args, span)` | 内置函数调用（println/tensor/rand/backward/grad/cross_entropy 等） |

#### Vm（字节码虚拟机）

```rust
pub struct Vm {
    pub functions: HashMap<String, usize>,  // 函数名 → Chunk 索引
    chunks: Vec<Chunk>,                     // 字节码块
    chunk_names: Vec<String>,               // Chunk 名称
    pub natives: HashMap<String, NativeFn>, // 原生函数
    globals: HashMap<String, Value>,        // 全局变量
    stack: Vec<Value>,                      // 操作数栈
    frames: Vec<Frame>,                     // 调用帧栈
}
```

关键方法：

| 方法 | 说明 |
|------|------|
| `new()` | 创建空 VM |
| `add_fn(name, chunk)` | 注册字节码函数 |
| `add_native(name, f)` | 注册原生函数 |
| `call(name)` | 调用函数 |
| `run(chunk_idx)` | 执行字节码主循环 |

#### native 注册同步点（16 类）与一致性守卫

新增一个 native **不是改一处**：全仓共有 **16 类同步点**（另有 1 类 `tenthc` 附带项），其中 **A 路径（Rust 全栈）必须改的有 9 类**。漏改后果分「响亮」与「静默」两档：

| 类别 | 同步点 | 形态 | 漏改后果 |
|------|--------|------|---------|
| ① VM 注册 | `runtime/natives.rs` | `vm.add_native("名", …)` 序列 | VM `未定义的原生函数`（响亮） |
| ② 解释器分派 | `runtime/interpreter/natives.rs` | `match name { "名" => … }` 臂 | `undefined function`（响亮） |
| ③ **HIR 返回类型表** | `hir/lower/types.rs::resolve_builtin` | `"名" \| … => Ok(Type::…)` | **静默**：落到 `_ => Ok(Type::Unknown)` ⇒ 静态类型丢失、shape 检查失能 |
| ④ 裸名白名单 | `hir/lower/lower_expr.rs` | Var 白名单长 `\|` 链 | 编译期「未定义变量」（响亮） |
| ⑤ 闭包自由变量排除表 | `hir/lower/closures.rs` | 内建名排除 `match` | 读码判定**良性**（按名回落），建议补齐 |
| ⑥ 解释器 FnRef 白名单 | `runtime/interpreter/eval.rs` | `Var → FnRef` 白名单 | 解释器「未定义变量」（响亮） |
| ⑦ 解释器预注册子集 | `runtime/interpreter/core.rs` | `insert_var` | 多被 ⑤⑥ 兜住 |
| ⑧ **污点逃逸点** | `hir/lower/taint.rs` | 逃逸 sink 名单 | **静默**（输出类 native 漏登 ⇒ 污点漏报） |
| ⑨ 泛型构造 dtype→名映射 | `hir/lower/mod.rs` **＋** `lower_expr.rs`（**同表两份拷贝**） | `NATIVE_GENERIC_CTORS` + 映射臂 | 静默用错 dtype |
| ⑩ `main.rs::vm_run` dead code | `main.rs` | 重复 `add_native` | 隐患：将来误启用会变「三源真相」 |
| ⑪ **方法分派对偶表**（非 native 表） | `runtime/vm/natives.rs` ↔ `runtime/interpreter/methods.rs` | 方法名匹配臂 | 一端漏改 ⇒ 另一端「没有方法 'x'」 |
| ⑫ WASM ABI 5 子点 | `compile/wasm/*`（常量/type 段/import 段/`resolve_func` 白名单/两个宿主 stub） | 序号 + 导入 + stub | 漏 ⇒ 响亮；**序号错位 ⇒ 静默错值**（历史高危） |
| ⑬ LSP 补全表 | `tenth/tools/lsp/.../completion.rs` | `("名", 说明)` | 仅无补全，无正确性影响 |
| ⑭ **文档/权威符号清单** | `docs/语言参考手册.md`、`docs/API冻结清单.md`、`tenth/std/prelude.th` | 手册条目 + prelude 清单 | 对外承诺与实现漂移 |
| ⑮ `std/*.th` 包装 | `tenth/std/**` | wrapper 调用 native | 用户按文档 `use` 后不可用 |
| ⑯ 测试内手抄副本 | `tenth/tests/native_parity_test.rs` | 手抄 native 子集 | 测试自欺（测的是副本） |
| 附 | `tenthc/hir/lower.th::is_builtin_name` | 63 项 `if name == "…"` | 仅影响 tenthc 自由变量分析与路径 C |

**为什么不能用朴素源码扫描做守卫**（已实证）：`runtime/natives.rs` 存在「**注册名是公开名、同一行错误文案却含旧内部名**」的残留（如注册 `to_utf8`、文案写 `_to_utf8`）⇒ 扫 `"_to_utf8"` 会误判；`prelude.th` 有 150+ 名注释清单会被误认成注册点；match 臂跨行续写与 `#[cfg]` 会让正则漏匹配。
**推荐守卫（A2 结构化集合相等 + 差集棘轮）**：VM 侧名集合**无需扫描**——`Vm.natives: HashMap<String, NativeFn>` 是 `pub`，`Vm::new()` + `register_all_natives(&mut vm)` 即可取真实集合；解释器侧因 `call_named_fn` 是 `match`（无数据结构、且**不能用空参探测**：`exit`/`read_line`/`tcp_accept`/`http_*` 空参会阻塞或退出进程）只能**新增 `pub const NATIVE_NAMES`**，用「集合差 == 显式豁免台账」断言（豁免项"变一致"也报红，防台账腐烂）。再配「双路径行为对拍」（同源三轴 + 金标准断言）反守护 const 与语义漂移。

**三处集合的口径与 SSOT（as-of 2026-09-17）**：

- **VM 注册集合**：`runtime/natives.rs` 中 `add_native("名", …)` 的**公开注册名**去重——结构化真表可直接直取（`Vm.natives` 是 `pub`），不扫描源码。
- **解释器分派集合**：`interpreter/natives.rs::call_named_fn` 的 `match name` 臂名（**严格锚点**：同缩进 + 行首引号串或 `|`，且只取 `=>` 之前）。两侧差集现状＝既定设计差异（解释器硬编码拒绝 `async_*`）＋ 1 个 f-string codegen 内部名；**公开名两侧已对齐**（此前的 4 项真缺口已由 W4 修复，见 `AUDIT-11.4.63`）。
- **`resolve_builtin` 返回类型集合**：`hir/lower/types.rs::resolve_builtin` 是 **native 静态返回类型的权威**；未覆盖者落 `_ => Ok(Type::Unknown)`（**静默**：静态类型丢失、编译期 shape 检查失能）。这类漏登是静默的 ⇒ 守卫**额外**断言「`vm.natives` 名集合 ⊆ `resolve_builtin` 可识别名集合」，否则该类漂移仍无守护（这是 A2 方案唯一未闭环处）。

> **本文档不复述覆盖/缺口数字**：历史上该类统计出现过互不一致的口径，且统计量硬编码进多份必然漂移（AGENTS §三）。**SSOT = 常驻守卫 `tenth/tests/native_registry_guard_test.rs`** 的三份显式棘轮台账：`KNOWN_VM_ONLY`（VM 有、解释器无）、`KNOWN_INTERP_ONLY`（解释器独有臂）、`KNOWN_NO_STATIC_RETURN_TYPE`（`resolve_builtin` 未覆盖＝欠债台账，记残差不记真相表）。台账**双向**断言：出现新差集报红，台账项已被修复却仍留在台账里也报红（防台账腐烂）。

#### Tensor（张量运算）

```rust
pub struct Tensor {
    pub data: ArrayD<f64>,     // ndarray 多维数组
    tape_id: Option<usize>,    // 自动微分节点 ID
    pub grad: Option<ArrayD<f64>>, // 梯度
}
```

支持的操作：

| 类别 | 操作 |
|------|------|
| **构造** | `from_vec`, `from_data`, `zeros`, `ones`, `full`, `rand`, `randn` |
| **形状** | `shape`, `reshape`, `flatten`, `transpose` |
| **元素运算** | `add_scalar`, `add_tensor`, `sub_scalar`, `sub_tensor`, `mul_scalar`, `mul_tensor`, `div_scalar`, `div_tensor`, `neg` |
| **归约** | `sum`, `mean`, `sum_axis` |
| **激活函数** | `relu`, `sigmoid`, `tanh`, `exp`, `log`, `abs`, `sqrt` |
| **矩阵运算** | `matmul` |
| **归一化** | `softmax`, `batchnorm` (通过方法调用) |
| **卷积** | `conv2d` (im2col 实现), `dropout` |
| **高级运算** | `gelu` — GELU 激活 (tanh 近似), `layer_norm` — LayerNorm 归一化, `cat` — 沿维度拼接 (2D), `masked_fill` — 掩码填充, `permute` — 维度重排, `broadcast_to` — 广播到目标形状, `max_val` — 最大值 |
| **索引** | `get`, `im2col`, `index_select`（**静态关联函数，非方法**：`Tensor::index_select(base, dim, 1-D index)`，可微；反向为 `scatter_add_along_dim`/重复 index 累加） |

#### Autodiff（自动微分）

```rust
pub struct Tape {
    nodes: Vec<TapeNode>,     // 计算图节点
    pub grads: Vec<ArrayD<f64>>, // 梯度存储
}

pub enum TapeOp {
    Add, Sub, Mul, Div, Neg, ReLU, MatMul, BatchedMatMul, Transpose,
    Sum, Mean, Exp, Log, Sigmoid, Softmax,
    CrossEntropy, Dropout, Conv2D, BatchNorm, LayerNorm, Gelu, Select,
    Abs, Scatter, Gather, IndexSelect, Reshape, MaskedFill,
    MaxPool2D, AvgPool2D, Custom(usize), Input,
}
```

自动微分控制函数：

| 函数 | 作用 |
|------|------|
| `new_grad()` | 创建新计算图，开启录制 |
| `param(tensor)` | 注册可训练参数 |
| `backward(loss)` | 反向传播，梯度写入参数的 `.grad` |
| `grad(param)` | 读取参数梯度 |
| `stop_grad()` | 关闭录制 |
| `zero_grad()` | 清零所有参数梯度 |
| `cross_entropy(logits, target)` | 交叉熵损失（融合 softmax） |

#### Arena（内存池）

```rust
pub struct Arena {
    data: Vec<f64>,       // 预分配的 f64 槽位
    used: usize,          // 已使用槽位数
    capacity: usize,      // 总容量
}
```

- 用于临时张量数据的高效分配
- 支持作用域重置（`reset()`），避免频繁堆分配
- 默认容量：64K f64 槽位（512 KB）

#### Limits（资源限制）

```rust
pub struct MemoryConfig {
    pub max_arena_bytes: usize,       // Arena 最大字节数
    pub max_variables: usize,         // 最大变量数
    pub max_accumulated_defs: usize,  // 最大定义数
    pub max_tensor_elements: usize,   // 最大张量元素数
    pub track_allocations: bool,      // 是否追踪分配
}
```

- 通过 `guard_vars()`, `guard_defs()`, `guard_tensor()` 进行运行时检查
- 支持 `mem-debug` 和 `mem-strict` feature flag

---

### 4.6 REPL (repl)

**位置**：`tenth/src/repl.rs`

REPL 命令：

| 命令 | 说明 |
|------|------|
| `:q` | 退出 |
| `:h` | 帮助 |
| `:vars` | 显示所有变量 |
| `:clear` | 重置所有状态 |
| `:mem` | 显示内存快照 |
| `:print V` | 打印变量值 |

特性：
- 支持多行输入（自动检测括号平衡）
- 累积式程序（函数定义跨行保留）
- 使用 `rustyline` 提供行编辑和历史记录
- 支持资源限制模式

---

## 5. 自举编译器 (tenthc)

**位置**：`tenthc/`

自举编译器完全用 Tenth 语言编写，验证了语言的图灵完备性和自举能力。

| 文件 | 职责 |
|------|------|
| `main.th` | 入口，拼接 6 个模块源码后调用 compile_host |
| `lexer/lexer.th` | Tenth 实现的词法分析器 |
| `lexer/token.th` | Tenth 实现的 Token 定义 |
| `parser/parser.th` | Tenth 实现的语法分析器 |
| `hir/hir.th` | Tenth 实现的 HIR 定义 |
| `hir/lower.th` | Tenth 实现的 HIR 降低器 |
| `compile/wasm.th` | Tenth 实现的 WASM 编译器 |

### 自举验证路径

| 路径 | 词法分析 | 语法分析 | 编译 | 状态 |
|------|---------|---------|------|------|
| A | Rust | Rust | compile_host | 秒级 |
| B | **Tenth** | **Tenth** | compile_program | 已验证 |
| C | WASM | wasmi | compile_host | 闭环 |

### Bridge 模块

`compile/bridge.rs` 负责将自举编译器输出的 compact `Program` 结构体转换为 Rust AST，然后通过标准管线编译为 WASM。这使得 Tenth 编写的编译器可以编译 Tenth 代码。

---

## 5.5 生态工具 (tools)

### tenthpm 包管理器

**位置**：`tenth/tools/tenthpm/`

tenthpm 是 Tenth 语言的包管理器，负责项目初始化、依赖管理、构建和发布。

| 文件 | 职责 |
|------|------|
| `Cargo.toml` | 依赖声明：serde, serde_json, toml |
| `src/main.rs` | CLI 入口，子命令分发 |
| `src/manifest.rs` | Tenth.toml 清单文件解析（项目名、版本、依赖） |
| `src/commands/` | 子命令实现（init / build / test / run / add / remove / list / clean / publish / install） |

#### CLI 子命令

| 命令 | 说明 |
|------|------|
| `tenthpm init <name>` | 创建新项目（生成 Tenth.toml + src/） |
| `tenthpm build` | 编译当前项目 |
| `tenthpm test` | 运行测试 |
| `tenthpm run` | 编译并运行当前项目 |
| `tenthpm add` | 添加依赖 |
| `tenthpm remove` | 移除依赖 |
| `tenthpm list` | 列出依赖 |
| `tenthpm clean` | 清理构建产物 |
| `tenthpm publish` | 发布到包注册表 |
| `tenthpm install` | 安装包 |

#### Tenth.toml 格式

```toml
[package]
name = "my-project"
version = "0.1.0"

[dependencies]
tensor-utils = "0.2"
```

### LSP 服务器

**位置**：`tenth/tools/lsp/`

LSP 服务器为编辑器（VS Code 等）提供语言智能功能，基于 LSP 协议通过 stdio 通信。

| 文件 | 职责 |
|------|------|
| `Cargo.toml` | 依赖声明：serde, serde_json, tenth (path) |
| `src/main.rs` | LSP 入口，启动消息循环 |
| `src/lsp_types.rs` | LSP 协议类型定义（请求/响应/通知） |
| `src/io.rs` | stdio 通信层（JSON-RPC 消息收发） |
| `src/handlers/` | 请求处理器 |

#### 支持的 LSP 功能

| 能力 | 方法 | 状态 |
|------|------|------|
| 文档同步 | didOpen/didChange/didClose/didSave | ✅ |
| 诊断 | publishDiagnostics | ✅ |
| 悬停 | hover | ✅ |
| 补全 | completion（AST 符号+关键字，非语义补全） | ✅ |
| 定义跳转 | definition | ✅ |
| 文档符号 | documentSymbol | ✅ |
| 引用查找 | references | ✅ |
| 重命名 | rename | ✅ |
| 签名帮助 | signatureHelp | ✅ |
| 折叠区域 | foldingRange | ✅ |
| 语义高亮 | semanticTokens | ✅ |
| 格式化 | formatting | ✅ |

---

## 6. 标准库 (std)

**位置**：`tenth/std/`

> **口径（as-of 2026-09-17）**：本树按 `tenth/std/` **磁盘实况** + `tenth/std/prelude.th` 索引（`:154`–`:256`）整理；顶层文件/目录清单与模块计数以这两处为 **SSOT**，本文件**不硬编码计数**。`test_*.th` 为模块内测试，不是用户 API。

```
tenth/std/
├── nn/          ← 神经网络（activations, attention, batchnorm, conv, dropout, embedding, feedforward, layer_norm, linear, loss, multihead_attention, ops, pool, positional_encoding, transformer）
├── optim/       ← 优化器（sgd, adam, adamw, nadam, radam, lion, adagrad, rmsprop, clip, accumulate, lr_schedule）
├── data/        ← 数据加载（dataloader, mnist, csv, sampler）
├── init/        ← 初始化（xavier_uniform/xavier_normal/he_normal/he_uniform/zeros_init/constant_init）
├── collections/ ← 迭代器与集合（iter map/filter/reduce, flat_map/partition, hashset）
├── string/      ← 字符串工具（join_lines/join_comma/repeat_sep/indent/word_wrap/capitalize, string_builder）
├── utils/       ← 工具（serialization save_model/load_model, math min/max/clamp）
├── fs/          ← 文件系统（exists/is_file/is_dir/mkdir/list_dir/remove/copy）
├── json/        ← JSON（parse/stringify/stringify_pretty/load/save）
├── toml/        ← TOML 解析
├── cli/         ← 命令行参数处理
├── logging/     ← 日志（debug/info/warn/error + set_level）
├── time/        ← 时间（now/now_ms/date/datetime/sleep_ms/timer）
├── random/      ← 随机数（rand_int/rand_float/rand_range/choice/shuffle/rand_seed）
├── math/        ← 数学函数与常量（functions / stats / constants）
├── crypto/      ← 哈希 wrapper（hash.th：sha256_hex / sha512_hex / md5_hex）
├── distributed/ ← 分布式本地语义（distributed.th：make_replicas / all_reduce_sum / all_reduce_mean / param_average）
├── io.th        ← 标准流（包装 eprint/eprintln/read_line）
├── env.th       ← 环境变量与进程退出（get/get_result/get_or_empty/set/exit）
├── net.th       ← TCP 客户端 + 服务端包装（connect/read/write/close/set_timeout/listen/accept/listener_close；UDP 直接用 udp_* native，无包装）
├── http.th      ← HTTP/1.1 客户端（get/post）
├── process.th   ← 子进程（new/arg/run/output/output_ex）
├── regex.th     ← 正则表达式（compile/match_/find/find_all/replace/split；句柄表）
├── async.th     ← 异步 I/O（async_sleep_ms/async_tcp_read/async_tcp_write）
├── date.th      ← Date 类型（date_new/date_add_days/date_diff/date_weekday …）
├── duration.th  ← Duration 类型（纳秒精度；duration_from_secs/duration_add/duration_to_string …）
├── autograd.th  ← 自定义可微算子 wrapper（call_custom_op1/2/3）
├── curry.th     ← 柯里化 / 部分应用 / 函数组合（partial/curry/compose）
├── runtime.th   ← 执行约束器（with_step_limit/with_timeout_ms；run_with_limit/run_with_timeout）
├── test_date.th     ← 模块内测试（非用户 API）
├── test_runtime.th  ← 模块内测试（非用户 API）
└── prelude.th   ← 可用项总目录（模块/函数清单 SSOT）
```

### prelude.th 内容

标准库预导入模块，声明了所有标准库模块的路径和常用函数引用。

---

## 7. 示例集 (Tenth实例)

**位置**：`Tenth实例/`（实例数见 `能力梳理/能力全梳理.md` §统计基线），涵盖算法、数据结构和 AI/ML：

### 经典算法

| 示例 | 文件 | 说明 |
|------|------|------|
| 斐波那契数列 | `fibonacci.th` | 递归/迭代实现 |
| 二分查找 | `binary_search.th` | 有序数组查找 |
| 快速排序 | `quicksort.th` | 分治排序 |
| 冒泡排序 | `bubble_sort.th` | 简单排序 |
| 最大公约数 | `gcd.th` | 辗转相除法 |
| 埃拉托色尼筛法 | `sieve.th` | 素数筛选 |
| 汉诺塔 | `hanoi.th` | 递归经典问题 |
| 最长公共子序列 | `lcs.th` | 动态规划 |
| N皇后II | `n_queens.th` | 回溯搜索 |
| 打家劫舍II | `house_robber.th` | 动态规划 |

### 数据结构

| 示例 | 文件 | 说明 |
|------|------|------|
| 栈 | `stack.th` | 基于Vec实现 |
| 队列 | `queue.th` | 基于Vec实现 |
| 链表 | `linked_list.th` | 指针/引用实现 |
| 二叉树遍历 | `bintree.th` | 前序/中序/后序 |
| HashMap使用 | `hashmap.th` | 键值存储 |
| 字符串处理 | `string_utils.th` | 字符串方法演示 |
| 通讯录 | `address_book.th` | 综合结构体使用 |

### 语言特性

| 示例 | 文件 | 说明 |
|------|------|------|
| 泛型示例 | `generics.th` | 泛型函数与结构体 |
| Trait示例 | `trait_demo.th` | Trait 定义与实现 |
| 闭包合集 | `closures.th` | 闭包语法演示（含变量捕获） |
| 凯撒密码 | `caesar.th` | 字符串操作 |

### AI/ML

| 示例 | 文件 | 说明 |
|------|------|------|
| 梯度下降 | `gradient_descent.th` | 标量梯度下降 |
| 自动微分 | `linear_regression.th` | 张量级自动微分 |
| 矩阵乘法 | `matmul.th` | 张量矩阵乘法 |
| 矩阵分解 | `matfact.th` | 矩阵运算 |
| XOR神经网络 | `xor_net.th` | 神经网络训练 |
| 神经网络层 | `neural_net.th` | 多层网络 |
| 多分类器 | `classifier.th` / `classifier_ce.th` | 分类任务 |
| 多项式回归 | `polyfit.th` | 回归拟合 |
| 微型CNN | `cnn.th` | 卷积神经网络 |
| 边缘检测 | `sobel.th` | Sobel 算子 |
| 归并排序 | `merge_sort.th` | 分治排序 |
| 二叉搜索树 | `bst.th` | 枚举递归、函数式不可变更新 |
| 闭包捕获 | `closure_capture.th` | 闭包捕获外层变量、嵌套闭包 |
| Softmax 回归 | `softmax_regression.th` | matmul、cross_entropy、自动微分 |
| Adam 优化器 | `adam.th` | 自适应学习率、一阶/二阶矩 |
| 张量广播 | `tensor_broadcast.th` | 标量广播、matmul、sqrt |
| 词频统计 | `word_count.th` | HashMap、字符串 split |
| 矩阵转置与运算 | `matrix_ops.th` | transpose、matmul、自动微分 |

---

## 8. 依赖关系

> crate 版本及 Feature Flags 详见 `DEPS.md`。

### 模块间依赖图

```
main.rs
  ├── lexer (token.rs, lexer.rs)
  ├── parser (ast.rs, parser.rs) ←── lexer
  ├── hir (types.rs, hir.rs, lower.rs) ←── parser
  ├── compile
  │   ├── bytecode.rs ←── hir, runtime/vm
  │   ├── wasm.rs ←── hir
  │   ├── bridge.rs ←── parser/ast, hir
  │   ├── gpu/ (mod.rs, cuda_kernel.rs, device.rs) ←── hir
  │   └── optimizations/ (mod.rs, fusion.rs, parallel.rs) ←── hir
  ├── runtime
  │   ├── value.rs ←── hir/types, tensor
  │   ├── interpreter.rs ←── hir, value, tensor, autodiff, arena, limits
  │   ├── vm.rs ←── value
  │   ├── tensor.rs ←── ndarray
  │   ├── autodiff.rs ←── tensor
  │   ├── arena.rs
  │   └── limits.rs
  ├── repl.rs ←── lexer, parser, hir, runtime
  └── error.rs
tenth/tools/tenthpm/ ←── serde, serde_json, toml
tenth/tools/lsp/ ←── serde, serde_json, tenth (path)
```

---

## 9. 项目运行方式

Tenth 提供四种运行模式：

| 模式 | 用途 |
|------|------|
| REPL | 交互式执行，适合探索与调试 |
| `run <file.th>` | 编译并执行 .th 文件（VM 优先，解释器 fallback） |
| `build <file.th>` | 编译 .th 为 .wasm 文件 |
| `wasm <file.th>` | 编译为 WASM 并通过 wasmi 执行 |

> 具体构建/运行命令及环境配置见 `DEPS.md`。

> 测试覆盖详见 `AUDIT.md` §三。

### 分发

`dist/` 目录包含：
- `install.bat` — Windows 安装脚本
- `tenth.bat` — Windows 启动脚本
- `README.md` — 分发说明

---

## 10. 路线图与状态

| Phase | 内容 | 状态 |
|-------|------|------|
| Phase 1 | Bootstrap 编译器 | ✅ 完成 |
| Phase 2 | 解释器夯实 | ✅ 完成 |
| Phase 3A | 类型系统深化 | ✅ 完成 |
| ~~Phase 3B~~ | ~~编译后端 (C)~~ | ❌ 已移除 |
| Phase 4 | GPU 与性能 | 🔧 脚手架就绪（gpu/ + optimizations/；随 v1.0.0 落地） |
| Phase 5 | AI 全栈 | 🚧 进行中（核心已实现：张量/自动微分/标准库；随 v1.0.0 落地） |
| Phase 6 | 生态与工具 | ✅ 已落地（tenthpm + lsp + debugger + profiler；随 v1.0.0） |
| Phase 7 | 核心标准库 | ✅ 完成 |
| Phase 8 | 自举编译器 | ✅ 完成 |

> 完整能力清单（594 项逐条状态）见 `能力梳理/能力全梳理.md`；统计量基线见其 §统计基线。

---

> 本文档为人工维护的架构文档，当前版本基线：**v1.0.0**（与文档头部一致，对齐 `RELEASE_NOTES.md` v1.0.0 发布说明，发布日期 2026-08-04）。
