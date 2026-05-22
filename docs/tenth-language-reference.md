# Tenth 语言参考手册

> 版本：0.1.0-draft | 日期：2026-05-22
>
> 本手册随编译器实现逐步完善。标记为「待定」的章节将在对应特性实现后填充。

---

## 目录

1. [引言](#1-引言)
2. [词法结构](#2-词法结构)
3. [类型系统](#3-类型系统)
4. [变量与绑定](#4-变量与绑定)
5. [表达式](#5-表达式)
6. [语句与控制流](#6-语句与控制流)
7. [函数与闭包](#7-函数与闭包)
8. [Trait 与泛型](#8-trait-与泛型)
9. [模块与包](#9-模块与包)
10. [所有权与内存](#10-所有权与内存)
11. [张量操作](#11-张量操作)
12. [并发与并行](#12-并发与并行)
13. [元编程与宏](#13-元编程与宏)
14. [标准库](#14-标准库)
15. [附录](#15-附录)

---

## 1. 引言

### 1.1 Tenth 是什么

Tenth 是一门为 AI 研究而生的通用静态类型编程语言。它将张量运算和自动微分提升为语言的内在概念，同时保持通用高级语言的全部能力。

### 1.2 设计目标

- **张量原生**：`a + b` 天然支持任意维度广播，无需引入外部库
- **编译期安全**：shape 不匹配在保存时即报错，而非训练中途崩溃
- **性能极致**：通过 MLIR 多层编译实现接近手写 CUDA 的性能
- **核心纯净**：autodiff、神经网络层通过宏构建，编译器只理解张量原语

### 1.3 Hello World

```
fn main() {
    println("Hello, Tenth!");
}
```

### 1.4 第一个张量程序

```
fn main() {
    let x = tensor[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
    let y = x + 1.0;
    let s = y.sum();
    println("Sum: {}", s);
}
```

### 1.5 约定

本手册中，`//` 表示行注释。代码块中的语法以 Tenth 0.1 版本为准。标记「待定」的章节表示该特性尚在设计中或未实现。

---

## 2. 词法结构

### 2.1 字符集

Tenth 源文件使用 UTF-8 编码。

### 2.2 注释

```
// 行注释
```

```
/*
   块注释
   可跨多行
*/
```

### 2.3 标识符

标识符由字母、数字和下划线组成，以字母或下划线开头。按约定：

- 变量名、函数名：`snake_case`
- 类型名、Trait 名：`PascalCase`
- 常量：`SCREAMING_SNAKE_CASE`

### 2.4 关键字

```
fn     let    mut    if     else   match
for    while  loop   break  continue
return try    use    mod    pub    trait
impl   enum   struct type   self   Self
spawn  task   shard  node   macro  where
as     in     true   false
```

> 关键字列表随语言演进可能调整。

### 2.5 字面量

```
// 整数
42
0xFF
0b1010

// 浮点数
3.14
1.0e-10

// 布尔
true
false

// 字符
'a'
'\n'

// 字符串
"hello"
"line 1\nline 2"

// 张量
tensor[[1.0, 2.0], [3.0, 4.0]]
tensor.ones([3, 224, 224])
tensor.randn([B, C])
```

### 2.6 运算符与分隔符

```
+  -  *  /  %        // 算术
== != <  > <= >=     // 比较
&& || !              // 逻辑
&  |  ^  << >>       // 位运算
= += -= *= /=        // 赋值
.  :: -> =>          // 成员、命名空间、函数返回、匹配臂
(  )  [  ]  {  }     // 分组、索引、块
,  ;  :              // 分隔符
..                   // 范围
&  &mut              // 引用、可变引用
```

---

## 3. 类型系统

### 3.1 基础类型

| 类型 | 描述 |
|------|------|
| `i8` `i16` `i32` `i64` | 有符号整数 |
| `u8` `u16` `u32` `u64` | 无符号整数 |
| `f16` `f32` `f64` | IEEE 浮点数 |
| `bf16` | Brain floating point（AI 常用） |
| `bool` | 布尔值 |
| `char` | Unicode 字符（4 字节） |
| `str` | 字符串切片 |
| `()` | 单元类型 |

### 3.2 张量类型

张量类型是 Tenth 的核心。Shape 是类型的一部分：

```
Tensor[dtype, dim0, dim1, ...]
```

**具体 shape：**
```
Tensor[f32, 3, 224, 224]       // 3×224×224 的 f32 张量
```

**符号维度：**
```
Tensor[f32, B, C, H, W]         // B/C/H/W 是类型变量
Tensor[f32, B, 3, H, W]         // 混合：部分已知，部分符号
```

符号维度（symbolic dimension）在编译期通过类型推导求解。编译器保证跨调用的维度兼容性：

```
fn matmul(a: Tensor[f32, M, K], b: Tensor[f32, K, N]) -> Tensor[f32, M, N] { ... }
// 编译器保证 a 的列数 == b 的行数（都等于 K）
```

**Rank polymorphism：**

`..` 匹配任意数量的维度：

```
fn sum(x: Tensor[f32, ..]) -> Tensor[f32, []]      // 任意 rank → 标量
fn mean(x: Tensor[f32, ..]) -> Tensor[f32, []]
fn first_dim(x: Tensor[T, N, ..]) -> Tensor[T, ..] // 去掉第一维
```

### 3.3 类型推断

| 位置 | 推断策略 |
|------|----------|
| 函数签名 | 必须显式标注 |
| 局部变量 | 完全推断（`let x = expr` 无需写类型） |
| 泛型参数 | 调用处推断 |
| 闭包参数 | 从上下文推断 |

### 3.4 结构体

```
struct TensorDescriptor {
    dtype: Dtype,
    shape: [u32],
    strides: [u32],
}
```

### 3.5 枚举（代数数据类型）

```
enum Option[T] {
    Some(T),
    None,
}

enum Result[T, E] {
    Ok(T),
    Err(E),
}

enum Activation {
    ReLU,
    GELU,
    SiLU,
    Tanh,
}
```

带数据的变体：

```
enum LayerDef {
    Linear { in_dim: u32, out_dim: u32, bias: bool },
    Conv2D { in_ch: u32, out_ch: u32, kernel: u32, stride: u32 },
    LayerNorm { dim: u32 },
    Sequential([LayerDef]),
}
```

### 3.6 类型别名

```
type Shape = [u32];
type Weights = Tensor[f32, K, D];
type TokenEmbedding = Tensor[f32, B, Seq, D];
```

### 3.7 类型转换

`as` 关键字用于基本类型转换：

```
let x: i32 = 42;
let y: f64 = x as f64;
```

> 张量类型转换语义待定。

---

## 4. 变量与绑定

### 4.1 let 绑定

```
let x = 42;                // 不可变绑定，类型推断为 i32
let y: f64 = 3.14;         // 显式类型标注
let z = x + 10;            // 推断为 i32
```

### 4.2 可变变量

```
let mut counter = 0;
counter = counter + 1;
counter += 1;
```

### 4.3 常量

```
const PI: f64 = 3.141592653589793;
const DEFAULT_DIM: u32 = 512;
```

### 4.4 遮蔽（Shadowing）

```
let x = 5;
let x = x + 1;     // 新的 x 遮蔽旧的 x，值为 6
```

### 4.5 解构

```
let point = (10, 20);
let (x, y) = point;

let opt = Some(42);
match opt {
    Some(v) => println("Value: {}", v),
    None => println("Nothing"),
}
```

---

## 5. 表达式

### 5.1 算术表达式

```
a + b    a - b    a * b    a / b    a % b
-a       a.pow(b) a.sqrt()
```

### 5.2 比较表达式

```
a == b   a != b   a < b    a > b
a <= b   a >= b
```

### 5.3 逻辑表达式

```
a && b   a || b   !a
```

短路求值：`&&` 和 `||` 仅在必要时计算右操作数。

### 5.4 位运算

```
a & b    a | b    a ^ b    a << b    a >> b   !a
```

### 5.5 范围表达式

```
0..10          // 0 到 9（不含 10）
0..=10         // 0 到 10（含 10）
0..B           // 0 到 B（运行时值）
..10           // 从 0 到 9
```

### 5.6 函数调用

```
add(1, 2)
relu(x)
matmul(a, b)
```

### 5.7 方法调用

```
x.mean(axis=0)
tensor.randn([N, D])
vec.push(42)
```

### 5.8 字段访问

```
point.x
model.weights
```

### 5.9 索引

```
a[0]            // 标量索引
a[0..B]         // 范围索引
a[:, :, 0]      // 多维索引（ stride/ellipsis 待定）
```

### 5.10 闭包（Lambda）

```
|x| x + 1
|x, y| x + y
|a, b| {
    let c = a + b;
    c * 2
}
```

### 5.11 块表达式

```
let result = {
    let x = compute_x();
    let y = compute_y();
    x + y       // 最后一个表达式是块的值
};
```

### 5.12 优先级

从高到低：

1. 方法调用、字段访问、索引
2. 一元 `-` `!`
3. `*` `/` `%`
4. `+` `-`
5. `<<` `>>`
6. `&`
7. `^`
8. `|`
9. `==` `!=` `<` `>` `<=` `>=`
10. `&&`
11. `||`
12. 范围 `..` `..=`
13. 赋值 `=` `+=` 等

> 完整优先级表待实现后确认。

---

## 6. 语句与控制流

### 6.1 if 表达式

```
if condition {
    do_something();
}

if x > 0 {
    println("positive");
} else if x < 0 {
    println("negative");
} else {
    println("zero");
}

// if 是表达式
let sign = if x > 0 { 1 } else if x < 0 { -1 } else { 0 };
```

### 6.2 match 表达式

```
match value {
    pattern1 => expr1,
    pattern2 => expr2,
    _ => default_expr,
}
```

模式匹配支持解构：

```
match layer {
    Linear { in_dim, out_dim } => println("Linear: {} → {}", in_dim, out_dim),
    Conv2D { kernel, stride } => println("Conv2D: kernel={}, stride={}", kernel, stride),
    ReLU => println("ReLU"),
}
```

守卫条件：

```
match x {
    n if n > 0 => "positive",
    n if n < 0 => "negative",
    _ => "zero",
}
```

### 6.3 loop 循环

```
loop {
    // 无限循环
    if condition {
        break;
    }
}
```

带返回值：

```
let result = loop {
    counter += 1;
    if counter == 10 {
        break counter * 2;
    }
};
```

### 6.4 while 循环

```
while condition {
    do_something();
}

while loss > threshold {
    loss = train_step(model, batch);
}
```

### 6.5 for 循环

```
for i in 0..10 {
    println(i);
}

for item in items {
    process(item);
}

for (i, item) in items.enumerate() {
    println("{}: {}", i, item);
}
```

### 6.6 break / continue

```
for i in 0..100 {
    if i % 2 == 0 {
        continue;   // 跳过偶数
    }
    if i > 50 {
        break;      // 超过 50 停止
    }
    println(i);
}
```

### 6.7 return

```
fn divide(a: f64, b: f64) -> Option[f64] {
    if b == 0.0 {
        return None;
    }
    Some(a / b)
}
```

---

## 7. 函数与闭包

### 7.1 函数定义

```
fn name(param1: Type1, param2: Type2) -> ReturnType {
    // 函数体
    expression    // 最后一个表达式为返回值
}
```

示例：

```
fn add(a: i32, b: i32) -> i32 {
    a + b
}

fn relu(x: Tensor[f32, ..]) -> Tensor[f32, ..] {
    x.maximum(0.0)
}
```

隐式返回：函数体最后一个表达式即为返回值，不需要显式 `return`。

### 7.2 函数类型

```
fn(i32, i32) -> i32
fn(Tensor[f32, ..]) -> Tensor[f32, ..]
```

函数是一等公民：

```
let f: fn(i32, i32) -> i32 = add;
let result = f(1, 2);
```

### 7.3 闭包

闭包可以捕获环境中的变量：

```
let offset = 10;
let add_offset = |x| x + offset;    // 捕获 offset
let result = add_offset(5);          // 15
```

闭包类型推断：

```
let closure = |x, y| x + y;         // 类型从首次调用推断
```

### 7.4 泛型函数

```
fn identity[T](x: T) -> T {
    x
}

fn first[T](items: [T]) -> Option[T] {
    if items.len() > 0 {
        Some(items[0])
    } else {
        None
    }
}
```

where 子句约束类型参数：

```
fn add_twice[T: Add](a: T, b: T) -> T {
    a + b + b
}
```

### 7.5 参数默认值

> 待定。

### 7.6 命名参数

> 待定。

---

## 8. Trait 与泛型

### 8.1 Trait 定义

Trait 定义了一组类型必须实现的行为：

```
trait Add {
    fn add(self, other: Self) -> Self;
}

trait Diffable {
    fn grad(self) -> Self;
}

trait Module {
    fn forward(self, input: Tensor[f32, B, ..]) -> Tensor[f32, B, ..];
    fn parameters(self) -> [Tensor[f32, ..]];
}
```

### 8.2 Trait 实现

```
impl Add for i32 {
    fn add(self, other: i32) -> i32 {
        // 编译器内在实现
    }
}

struct Point {
    x: f64,
    y: f64,
}

impl Add for Point {
    fn add(self, other: Point) -> Point {
        Point { x: self.x + other.x, y: self.y + other.y }
    }
}
```

### 8.3 Trait 约束

```
fn sum_all[T: Add + Default](items: [T]) -> T {
    let mut total = T::default();
    for item in items {
        total = total.add(item);
    }
    total
}
```

多重约束：

```
fn process[T: Clone + Diffable](x: T) -> (T, T) {
    (x.clone(), x.grad())
}
```

### 8.4 内置 Trait

| Trait | 描述 |
|-------|------|
| `Add` | 加法 |
| `Sub` | 减法 |
| `Mul` | 乘法 |
| `Div` | 除法 |
| `Neg` | 取负 |
| `Eq` | 相等比较 |
| `Ord` | 全序比较 |
| `Clone` | 深拷贝 |
| `Copy` | 按位复制 |
| `Drop` | 析构 |
| `Default` | 默认值 |
| `Display` | 格式化输出 |
| `Iterator` | 迭代器 |
| `Into` / `From` | 类型转换 |
| `Diffable` | 可微分（标准库） |
| `Module` | 神经网络模块（标准库） |

### 8.5 派生宏

```
#[derive(Clone, Eq, Display)]
struct Config {
    lr: f64,
    epochs: u32,
}
```

---

## 9. 模块与包

### 9.1 模块定义

```
mod math {
    pub fn add(a: i32, b: i32) -> i32 {
        a + b
    }

    fn helper() -> i32 {    // 私有
        42
    }
}
```

### 9.2 可见性

- 默认私有
- `pub` 标记公开
- `pub(mod)` 指定可见范围（待定）

### 9.3 use 导入

```
use math::add;
use math::{add, sub};
use layers::conv::Conv2D;
use std::tensor::Tensor;
```

### 9.4 模块文件组织

```
src/
├── main.tenth
├── model.tenth
└── layers/
    ├── mod.tenth
    ├── linear.tenth
    └── conv.tenth
```

> 包管理系统待 Phase 5 实现。

---

## 10. 所有权与内存

### 10.1 所有权规则

每个值有且仅有一个所有者：

```
let x = tensor.rand([1024, 512]);  // x 拥有该张量
let y = x;                           // 所有权转移给 y，x 失效
// println(x);                        // 编译错误：x 已移动
```

### 10.2 引用与借用

```
let x = tensor.rand([1024, 512]);
let r = &x;           // 不可变借用
let r2 = &x;          // 多个不可变借用可以共存
println(r == r2);

let mut y = tensor.rand([512, 256]);
let rm = &mut y;      // 可变借用（独占）
// let r3 = &y;        // 编译错误：已有可变借用
```

### 10.3 默认引用计数

与 Rust 的关键区别：Tenth 默认使用引用计数。同一份权重和中间激活可以被多处同时引用：

```
let weights = tensor.rand([1024, 512]);
let w1 = weights;     // 引用计数 +1（而非移动）
let w2 = weights;     // 引用计数再 +1
// weights, w1, w2 都有效

fn use_weights(w: Tensor[f32, 1024, 512]) {
    // w 进入时引用计数 +1，离开时 -1
}
```

### 10.4 独占所有权模式

性能关键路径可声明独占所有权：

> 语法待定。候选方案：
> - 类型标注：`Owned[Tensor[f32, M, N]]` vs `Shared[Tensor[f32, M, N]]`
> - 关键字：`let own x = ...` vs `let shared x = ...`

### 10.5 Arena 分配器

```
let arena = Arena::new();
for epoch in 0..num_epochs {
    arena.scope(|| {
        let batch = dataloader.next();
        let pred = model.forward(&batch);
        let loss = criterion(pred, batch.labels);
        loss.backward();
        optimizer.step();
    });
    // 该 iteration 的所有临时张量在此批量释放
}
```

### 10.6 GPU 显存管理

```
let gpu_tensor = tensor.rand([4096, 4096]).to_device(GPU(0));
// Drop 时自动释放 GPU 显存
```

设备放置追踪：编译器静态追踪张量所在设备，对不必要的 CPU↔GPU 传输发出警告。

### 10.7 Drop

```
impl Drop for MyResource {
    fn drop(self) {
        // 清理资源
    }
}
```

---

## 11. 张量操作

> 本章是 Tenth 最核心的领域。运算符语义与 NumPy/PyTorch 保持一致以降低学习成本。

### 11.1 张量创建

```
tensor[[1.0, 2.0], [3.0, 4.0]]               // 从字面量
tensor.zeros([B, C, H, W])                     // 全零
tensor.ones([N, D])                             // 全一
tensor.rand([K, M])                             // [0,1) 均匀分布
tensor.randn([K, M])                            // 标准正态分布
tensor.arange(0, 10)                            // 等差序列
tensor.eye(N)                                   // 单位矩阵
tensor.full([H, W], fill_value)                 // 填充常量
tensor.like(other)                              // 形状同 other 的全零张量
```

### 11.2 元素级运算

```
a + b      a - b      a * b      a / b
a + 1.0    // 广播：标量加到每个元素
-a         // 取负
a.abs()    a.sqrt()   a.exp()    a.log()
a.sin()    a.cos()    a.tanh()   a.sigmoid()
a.relu()   a.gelu()   a.silu()
```

### 11.3 广播（Broadcasting）

Tenth 遵循 NumPy 广播规则：从最后一维开始对齐，维度相等或一方为 1 时可广播：

```
Tensor[f32, 3, 4] + Tensor[f32, 4]       → Tensor[f32, 3, 4]   // OK
Tensor[f32, 3, 4] + Tensor[f32, 3, 1]    → Tensor[f32, 3, 4]   // OK
Tensor[f32, 3, 4] + Tensor[f32, 5, 4]    → 编译错误            // 维度不匹配
Tensor[f32, B, 4] + Tensor[f32, 4]       → Tensor[f32, B, 4]   // OK
```

> 广播规则严格度（全面隐式 vs 部分显式）待定。

### 11.4 线性代数

```
matmul(a, b)                          // 矩阵乘法
matmul(a, b, transpose_a=true)        // a 转置后再乘
bmm(a, b)                             // 批量矩阵乘法
a @ b                                 // @ 运算符（待定）
a.transpose()                         // 转置
a.inverse()                           // 逆矩阵
```

### 11.5 规约操作

```
x.sum()                    // 全部求和 → 标量
x.sum(axis=0)              // 沿第 0 轴求和
x.sum(axis=[0, 2])         // 沿多轴求和
x.mean()                   // 均值
x.mean(axis=1)             // 沿轴均值
x.max()                    // 最大值
x.argmax(axis=-1)          // 最大值索引
x.var()                    // 方差
x.std()                    // 标准差
x.norm()                   // L2 范数
x.norm(p=1)                // L1 范数
```

### 11.6 形状操作

```
x.reshape([N, D])           // 改变形状
x.view([N, H, W, C])        // 返回新视图（共享数据）
x.flatten()                  // 展平
x.unsqueeze(0)               // 在第 0 维插入大小为 1 的维度
x.squeeze()                  // 移除大小为 1 的维度
x.permute([1, 0, 2])        // 置换维度顺序
x.expand([B, C, H, W])      // 扩展维度
```

### 11.7 索引与切片

```
x[0]                   // 第 0 行
x[0..B]                // 范围切片
x[0..B, :]             // 多维：取前 B 行，所有列
x[0, ::2]              // 第 0 行，每隔一列取（步长）
x[.., -1]              // 最后一列
```

### 11.8 拼接与堆叠

```
concat([a, b], axis=0)       // 沿轴拼接
stack([a, b], axis=0)        // 沿新轴堆叠
```

### 11.9 设备操作

```
x.to_device(GPU(0))           // 移动到 GPU 0
x.to_device(CPU)              // 移动到 CPU
x.device()                    // 查询当前设备
```

### 11.10 类型转换

```
x.to_dtype(f32)
x.to_dtype(bf16)
```

---

## 12. 并发与并行

### 12.1 轻量级任务（单机多核）

```
spawn task worker(id: u32) -> Result {
    let data = load_data(id);
    let processed = process(data);
    Ok(processed)
}

let handles = (0..4).map(|i| spawn task worker(i));
let results = handles.map(|h| h.await());
```

任务系统基于 M:N 调度，无堆栈协程。async/await 零成本。

### 12.2 SPMD 数据并行（单机多 GPU）

```
shard(batch across [GPU(0), GPU(1)])
fn train_step(model: &Model, batch: Tensor[f32, B, C, H, W]) -> Loss {
    let pred = model.forward(batch);
    let loss = criterion(pred, batch_labels);
    loss.backward();
    return loss;
}
```

编译器自动插入 NCCL all-reduce 通信。

> SPMD 的完整语义（分片策略、通信模式、梯度聚合）待 Phase 4 细化。

### 12.3 分布式消息传递（多机）

```
node(0)
fn coordinator() {
    for i in 1..num_nodes {
        send(node(i), Task { data: prepare_data(i) });
    }
    let results: [Result] = (1..num_nodes).map(|_| recv()).collect();
    aggregate(results);
}

node(1)
fn worker() {
    let task: Task = recv();
    let result = compute(task);
    send(node(0), result);
}
```

> 分布式通信语义（容错、重试、序列化）待 Phase 4 细化。

### 12.4 同步原语

> 待定：Mutex、Channel、Barrier 等。

---

## 13. 元编程与宏

### 13.1 宏概述

Tenth 的宏系统允许在 HIR（高层 IR）层面操作语法树。宏可以生成、变换、删除任意代码。宏本身用 Tenth 编写。

### 13.2 声明宏

> 待定。类似 Rust 的 `macro_rules!`，用于模式匹配和代码替换。

### 13.3 过程宏

过程宏接收 HIR 节点，返回变换后的 HIR 节点：

```
macro derive_serialize(struct_def: StructDef) -> StructDef {
    let name = struct_def.name;
    let fields = struct_def.fields;
    // 生成 serialize 和 deserialize 函数
    let serialize_fn = generate_serialize(name, fields);
    let deserialize_fn = generate_deserialize(name, fields);
    return concat(struct_def, serialize_fn, deserialize_fn);
}
```

### 13.4 自动微分宏

```
use std::autodiff;

fn model(x: Tensor[f32, B, C]) -> Tensor[f32, B, D] {
    ...
}

let (loss, grads) = grad(model)(input);
```

`grad` 宏读取 `model` 的 HIR，构建 Wengert tape，生成前向和反向传播代码。

### 13.5 宏的其他应用

- `#[derive(Debug)]` —— 自动生成调试输出
- `#[derive(Serialize)]` —— 模型权重序列化
- `nn!` —— 神经网络层 DSL
- `pipeline!` —— 数据管道声明

> 宏的详细 API 和安全模型待实现时确定。

---

## 14. 标准库

### 14.1 std::tensor

编译器内在模块。包含张量类型定义和所有基础运算。无需显式导入。

### 14.2 std::autodiff

```
use std::autodiff::{grad, value_and_grad, jvp, vjp};

let f = |x: Tensor[f32, N]| x.sum();
let df = grad(f);
let gradient = df(input);               // 反向模式
let (val, g) = value_and_grad(f)(input);
```

### 14.3 std::arena

```
use std::arena::Arena;

let arena = Arena::new();
arena.scope(|| {
    // 在 scope 内分配的张量在 scope 结束时批量释放
});
```

### 14.4 std::device

```
use std::device::{CPU, GPU, Device};

let dev = GPU(0);
let tensor = tensor.rand([N, D]).to_device(dev);
let current = tensor.device();
```

### 14.5 std::task

参见第十二章并发。

### 14.6 std::spmd / std::dist

参见第十二章并发。

### 14.7 第二方库（官方维护）

- `nn` —— 神经网络层
- `optim` —— 优化器
- `data` —— 数据管道

> 以上库将在 Phase 4 实现，届时补充文档。

---

## 15. 附录

### 15.1 运算符优先级完整表

| 优先级 | 运算符 | 结合性 |
|--------|--------|--------|
| 1 | `.` `::` `[]` `()` | 左 |
| 2 | `-` (一元) `!` | 右 |
| 3 | `*` `/` `%` | 左 |
| 4 | `+` `-` | 左 |
| 5 | `<<` `>>` | 左 |
| 6 | `&` | 左 |
| 7 | `^` | 左 |
| 8 | `\|` | 左 |
| 9 | `==` `!=` `<` `>` `<=` `>=` | 左 |
| 10 | `&&` | 左 |
| 11 | `\|\|` | 左 |
| 12 | `..` `..=` | 左 |
| 13 | `=` `+=` `-=` `*=` `/=` `%=` | 右 |

### 15.2 与 Rust 的关键区别

| 特性 | Rust | Tenth |
|------|------|-------|
| 默认所有权 | 独占（move） | 引用计数（Rc） |
| 借用检查 | 严格编译期检查 | 默认 Rc 降低约束，可选独占 |
| 张量 | 库级别（ndarray） | 语言原生 |
| Shape 类型 | 无 | 符号维度 + rank polymorphism |
| Autodiff | 库级别 | 宏 + 编译器内在 |
| 并行 | Rayon + MPI 库 | 语言内置三层并行 |
| GPU | 库级别（CUDA FFI） | 编译器原生生成 kernel |
| 编译 | LLVM | MLIR → LLVM |

### 15.3 术语表

| 术语 | 英文 | 定义 |
|------|------|------|
| 张量 | Tensor | 多维数组，Tenth 的基础数据结构 |
| 符号维度 | Symbolic Dimension | 编译期通过类型变量推导的维度值 |
| Rank Polymorphism | | 函数不依赖输入张量的具体秩数 |
| 广播 | Broadcasting | 对不同形状的张量应用元素级运算的规则 |
| 自动微分 | Autodiff | 自动计算函数的导数/梯度 |
| SPMD | Single Program Multiple Data | 同一程序在多设备上运行不同数据 |
| HIR | High-level IR | 保留语义信息的高层中间表示 |
| MIR | Mid-level IR | 包含资源分配信息的中层中间表示 |
| LIR | Low-level IR | 接近机器代码的低层中间表示 |
| 引用计数 | Reference Counting | 追踪值被引用次数的内存管理策略 |

### 15.4 参考实现与灵感

- Rust 语言参考：https://doc.rust-lang.org/reference/
- Julia 文档：https://docs.julialang.org/
- JAX 文档：https://jax.readthedocs.io/
- MLIR 文档：https://mlir.llvm.org/docs/

---

> 本手册随 Tenth 编译器实现同步更新。当前版本：0.1.0-draft。