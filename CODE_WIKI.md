# Tenth 语言 — Code Wiki

> **Tenth** = Tensor + Zenith，意为「张量之巅」—— 一门为 AI 研究而生的编程语言
>
> 当前版本：**v0.3.0** | 语言实现：Rust | 许可证：MIT

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

- **张量级自动微分**：内置 19 个算子的反向传播全链路，支持 `new_grad()` / `param()` / `backward()` / `grad()` 等控制函数
- **双执行引擎**：字节码 VM（默认，43 指令）+ 树遍历解释器（fallback）
- **WASM 编译**：通过 `wasm-encoder` 生成 WASM 字节码，`wasmi` 执行验证
- **自举编译器**：用 Tenth 自身编写的编译器（`tenthc/`），经三阶段验证闭环
- **类 Rust 语法**：支持 `struct` / `enum` / `match` / `trait` / `impl` / 泛型 / 引用与移动语义

### 项目目录结构

```
项目根目录/
├── tenth/                  # 主编译器与运行时（Rust 实现）
│   ├── src/                # 源码
│   │   ├── lexer/          # 词法分析
│   │   ├── parser/         # 语法分析
│   │   ├── hir/            # 高级中间表示
│   │   ├── compile/        # 编译后端（字节码 + WASM）
│   │   ├── runtime/        # 运行时（解释器 + VM + 张量 + 自动微分）
│   │   ├── error.rs        # 统一错误类型
│   │   ├── lib.rs          # 库入口
│   │   ├── main.rs         # CLI 入口
│   │   └── repl.rs         # REPL 交互环境
│   ├── std/                # Tenth 标准库（.th 文件）
│   ├── tests/              # 集成测试
│   ├── Cargo.toml          # Rust 项目配置
│   └── build.rs            # 构建脚本
├── tenthc/                 # 自举编译器（Tenth 编写）
│   ├── main.th             # 入口
│   ├── boot.th             # 自举主程序
│   ├── lexer/              # Tenth 实现的词法分析
│   ├── parser/             # Tenth 实现的语法分析
│   ├── hir/                # Tenth 实现的 HIR
│   └── compile/            # Tenth 实现的 WASM 编译
├── Tenth实例/              # 25 个语言示例
├── docs/                   # 文档
│   ├── 语言参考手册.md
│   └── superpowers/plans/  # 开发计划
├── dist/                   # 分发脚本
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
                    ┌─────────────┼─────────────┐
                    ▼             ▼              ▼
            ┌──────────┐  ┌──────────┐   ┌──────────┐
            │ Bytecode │  │Interpreter│   │   WASM   │
            │ Compiler │  │(树遍历)   │   │ Compiler │
            └──────────┘  └──────────┘   └──────────┘
                    │             │              │
                    ▼             ▼              ▼
              ┌────────┐   ┌──────────┐   ┌──────────┐
              │   VM   │   │ 直接执行  │   │ wasmi    │
              │(栈式VM)│   │          │   │ 执行验证  │
              └────────┘   └──────────┘   └──────────┘
```

**执行优先级**：VM 优先 → 解释器 fallback。当 VM 遇到不支持的构造（如闭包、张量字面量）时，自动回退到树遍历解释器。

---

## 3. 编译流水线

### 3.1 完整流水线

```rust
// main.rs 中的核心管线函数
fn source_to_hir(source: &str) -> TenthResult<HirProgram> {
    let tokens = Lexer::new(source).tokenize()?;        // 源码 → Token
    let program = Parser::new(tokens).parse_program()?; // Token → AST
    let hir = Lowerer::new().lower_program(&program)?;  // AST → HIR
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
    StructLiteral { name, generics, fields, use_defaults },
    EnumLiteral { enum_name, variant, fields },
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
    EnumDef { name, variants },
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
    pub structs: HashMap<String, Vec<(String, Type)>>,        // 结构体定义
    pub generic_structs: HashMap<String, HirGenericStruct>,    // 泛型结构体
    pub enums: HashMap<String, Vec<(String, Vec<(String, Type)>)>>, // 枚举定义
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

#### 字节码 VM 指令集（43 条）

```rust
pub enum Op {
    // 常量压栈
    PushInt(i64), PushFloat(f64), PushBool(bool), PushStr(usize), PushUnit,
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
    IndexGet, SliceStr,
    MakeEnum(usize, usize, usize), IsEnumVariant(usize), EnumGetField(usize),
    PushRange(i64, i64, bool), MoveOp,
}
```

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
| **索引** | `get`, `im2col` |

#### Autodiff（自动微分）

```rust
pub struct Tape {
    nodes: Vec<TapeNode>,     // 计算图节点
    pub grads: Vec<ArrayD<f64>>, // 梯度存储
}

pub enum TapeOp {
    Add, Sub, Mul, Div, Neg, ReLU, MatMul, Transpose,
    Sum, Mean, Exp, Log, Sigmoid, Softmax,
    CrossEntropy, Dropout, Conv2D, BatchNorm, Input,
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
| `main.th` | 入口，调用编译管线 |
| `boot.th` | 自举主程序（~810 行），包含完整编译管线 |
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

## 6. 标准库 (std)

**位置**：`tenth/std/`

```
tenth/std/
├── nn/
│   ├── linear.th        # 线性层：fn linear(x, w, b) = x.matmul(w.transpose()) + b
│   ├── loss.th          # 损失函数：MSE, L1, BCE, cross_entropy
│   ├── activations.th   # 激活函数：relu, sigmoid, tanh
│   ├── dropout.th       # Dropout 层
│   ├── batchnorm.th     # BatchNorm 层
│   ├── conv.th          # 卷积层
│   └── embedding.th     # 嵌入层
├── optim/
│   ├── sgd.th           # SGD 优化器（vanilla / momentum / decay）
│   ├── adam.th          # Adam 优化器
│   ├── adagrad.th       # AdaGrad 优化器
│   └── rmsprop.th       # RMSProp 优化器
├── data/
│   └── dataloader.th    # DataLoader（规划中）
├── init/
│   └── initializers.th  # 初始化器
├── math/
│   └── functions.th     # 数学函数参考
├── utils/
│   └── serialization.th # 序列化（规划中）
└── prelude.th           # 可用项总目录
```

### prelude.th 内容

标准库预导入模块，声明了所有标准库模块的路径和常用函数引用。

---

## 7. 示例集 (Tenth实例)

**位置**：`Tenth实例/`，共 25 个示例，涵盖算法、数据结构和 AI/ML：

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
| 闭包合集 | `closures.th` | 闭包语法演示 |
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

---

## 8. 依赖关系

### Rust 依赖 (Cargo.toml)

| 依赖 | 版本 | 用途 |
|------|------|------|
| `ndarray` | 0.16 | 多维数组（张量底层数据结构） |
| `rustyline` | 15 | REPL 行编辑库 |
| `thiserror` | 2 | 错误类型派生宏 |
| `rand` | 0.8 | 随机数生成 |
| `rand_distr` | 0.4 | 随机分布（正态分布等） |
| `wasm-encoder` | 0.215 | WASM 字节码生成 |
| `wasmi` | 0.39 | WASM 解释器 |

### 编译工具链

- Rust ≥ 1.95（edition 2024）
- 所有 crate 依赖通过 Cargo 自动下载

### Feature Flags

| Flag | 说明 |
|------|------|
| `mem-debug` | 启用内存追踪计数器和硬限制检查（~2% 性能开销） |
| `mem-strict` | 严格模式：限制违反时 panic 而非软警告（隐含 `mem-debug`） |

### 模块间依赖图

```
main.rs
  ├── lexer (token.rs, lexer.rs)
  ├── parser (ast.rs, parser.rs) ←── lexer
  ├── hir (types.rs, hir.rs, lower.rs) ←── parser
  ├── compile
  │   ├── bytecode.rs ←── hir, runtime/vm
  │   ├── wasm.rs ←── hir
  │   └── bridge.rs ←── parser/ast, hir
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
```

---

## 9. 项目运行方式

### 编译

```bash
# Release 模式编译（推荐）
cargo build --release --manifest-path tenth/Cargo.toml

# Debug 模式（带内存调试）
cargo build --manifest-path tenth/Cargo.toml --features mem-debug

# 严格内存模式
cargo build --manifest-path tenth/Cargo.toml --features mem-strict
```

### 运行

```bash
# 启动 REPL
cargo run --release --manifest-path tenth/Cargo.toml

# 运行 .th 文件
cargo run --release --manifest-path tenth/Cargo.toml run path/to/file.th

# 编译为 WASM
cargo run --release --manifest-path tenth/Cargo.toml build path/to/file.th

# 运行 WASM
cargo run --release --manifest-path tenth/Cargo.toml wasm path/to/file.th

# 带内存限制的 REPL
cargo run --release --manifest-path tenth/Cargo.toml -- --max-memory 256
```

### 测试

```bash
# 运行所有测试（88 项通过，1 项忽略）
cargo test --manifest-path tenth/Cargo.toml

# 测试文件列表
# tests/lexer_test.rs       — 词法分析测试
# tests/parser_test.rs      — 语法分析测试
# tests/integration_test.rs — 集成测试
# tests/struct_test.rs      — 结构体测试
# tests/enum_test.rs        — 枚举测试
# tests/generic_test.rs     — 泛型测试
# tests/trait_test.rs       — Trait 测试
# tests/ownership_test.rs   — 所有权/借用测试
# tests/memory_test.rs      — 内存限制测试
# tests/module_test.rs      — 模块系统测试
# tests/stdlib_test.rs      — 标准库测试
# tests/selfhost_verify.rs  — 自举验证测试
# tests/three_stage.rs      — 三阶段编译测试
```

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
| Phase 4 | GPU 与性能 | 🚧 进行中 |
| Phase 5 | AI 全栈 | 🚧 进行中 |
| Phase 6 | 生态与工具 | 🚧 进行中 |
| Phase 7 | 核心标准库 | ✅ 完成 |
| Phase 8 | 自举编译器 | ✅ 完成 |

### 当前能力总结

| 能力 | 状态 |
|------|------|
| Lexer / Parser / AST | ✅ |
| HIR + 类型推断 + 借用检查 | ✅ |
| 树遍历解释器 | ✅ (VM fallback) |
| 字节码 VM（栈式，43 指令） | ✅ 默认执行路径 |
| 泛型函数 / 结构体 | ✅ |
| Trait 定义与实现 | ✅ |
| 引用 / 移动语义 | ✅ |
| struct / enum / match | ✅ VM 全支持 |
| REPL 交互环境 | ✅ 多行输入支持 |
| 内存护栏 (arena + limits) | ✅ |
| WASM 编译 (wasm-encoder + wasmi) | ✅ |
| 张量级自动微分 (19 算子) | ✅ backward 全链路 |
| 张量间运算 (matmul/广播/转置) | ✅ |
| Conv2D / Dropout / BatchNorm | ✅ |
| Vec / HashMap / String 标准库 | ✅ pop/split/trim 等 10+ 方法 |
| 自举编译器 (Tenth 编写，全链路) | ✅ ~0.2s |
| WASM import 输出 | ✅ wasmi 验证通过 |
| 块注释 /* */ | ✅ 支持嵌套 |

---

> 本文档基于项目 v0.3.0 版本源码自动生成，最后更新：2026-06-14
