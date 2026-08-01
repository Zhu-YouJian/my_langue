# Tenth 实例目录维护规范

## 目录结构

```
Tenth实例/
├── README.md          ← 本文件
└── <实例名>/
    ├── task.txt       ← 任务描述（可选，复杂实例建议添加）
    └── *.th           ← Tenth 源码实现
```

## 命名约定

- 实例目录名使用中文简短描述，如 `快速排序`、`二分查找`
- Tenth 源码文件使用英文小写蛇形命名，如 `quicksort.th`、`binary_search.th`
- 若实例包含多个文件，可在目录内再建子目录

## 每个实例应包含

1. **源码文件**（`.th`）— 完整的 Tenth 实现，含 `main()` 入口可独立运行
2. **任务描述**（`task.txt`，可选）— 说明算法/数据结构、输入输出示例

## 添加新实例

1. 在 `Tenth实例/` 下创建以实例名命名的子目录
2. 编写 `task.txt`（如需要）
3. 编写 Tenth 源码
4. 通过 `cargo run -- run "Tenth实例/<实例名>/<文件>.th"` 验证可运行
5. 更新本文件末尾的实例索引

## 实例索引

| 实例 | 目录 | 涉及特性 |
|------|------|----------|
| 快速排序 | `快速排序/` | 递归、Vec、while、比较 |
| 二分查找 | `二分查找/` | while、Vec 索引、整数除法 |
| 链表 | `链表/` | 枚举、match、递归、字段访问 |
| 冒泡排序 | `冒泡排序/` | Vec 构建、swap 模式、嵌套 while |
| 斐波那契数列 | `斐波那契数列/` | 递归 vs 迭代、函数对比 |
| 栈 | `栈/` | 结构体、&mut 引用、Vec 封装 |
| 字符串处理 | `字符串处理/` | 字符串索引、拼接、回文判断 |
| 神经网络层 | `神经网络层/` | 张量创建、ReLU、广播运算 |
| HashMap 使用 | `HashMap使用/` | insert、get、键值操作 |
| 最大公约数 | `最大公约数/` | 递归、取模、迭代对比 |
| 埃拉托色尼筛法 | `埃拉托色尼筛法/` | 质数判定、试除法 |
| 二叉树遍历 | `二叉树遍历/` | 枚举递归、中序遍历、&mut Vec |
| 队列 | `队列/` | 结构体、&mut 引用、FIFO |
| 张量运算 | `矩阵乘法/` | 广播、ReLU、exp、softmax |
| 梯度下降 | `梯度下降/` | 线性回归、参数更新循环 |
| 闭包合集 | `闭包合集/` | 匿名函数、高阶用法 |
| Trait 示例 | `Trait示例/` | trait 定义、impl、多态 |
| 泛型示例 | `泛型示例/` | 泛型函数、泛型结构体 |
| 凯撒密码 | `凯撒密码/` | 字符串索引、查表替换 |
| 汉诺塔 | `汉诺塔/` | 纯递归、经典问题 |
| 通讯录 | `通讯录/` | struct+Vec 综合、CRUD |
| N皇后 II | `N皇后II/` | 递归回溯、Vec、剪枝 |
| 最长公共子序列 | `最长公共子序列/` | DP、字符串索引、滚动数组 |
| 打家劫舍 II | `打家劫舍II/` | DP 环形、多分支转移 |
| 归并排序 | `归并排序/` | 分治递归、Vec 合并、while |
| 二叉搜索树 | `二叉搜索树/` | 枚举递归、match、函数式不可变更新 |
| 闭包捕获 | `闭包捕获/` | 闭包捕获外层变量、嵌套闭包、闭包工厂 |
| Softmax 回归 | `Softmax回归/` | matmul、cross_entropy、自动微分、for-in |
| Adam 优化器 | `Adam优化器/` | 自适应学习率、一阶/二阶矩、pow/sqrt |
| 张量广播 | `张量广播/` | 标量广播、matmul、sqrt、transpose、flatten |
| 词频统计 | `词频统计/` | HashMap、字符串 split、条件更新 |
| 矩阵转置与运算 | `矩阵转置与运算/` | transpose、matmul、自动微分 |
| XOR神经网络 | `XOR神经网络/` | 张量创建、反向传播、训练循环 |
| 多项式回归 | `多项式回归/` | 张量运算、拟合 |
| 微型CNN | `微型CNN/` | Conv2D、im2col、自动微分 |
| 边缘检测 | `边缘检测/` | Sobel 算子、张量卷积 |
| 矩阵分解 | `矩阵分解/` | 矩阵运算、LU 分解 |
| 自动微分 | `自动微分/` | 张量级自动微分、线性回归 |
| 计时器 | `计时器/` | time_now、time_date、计时器、sleep |
| JSON 处理 | `JSON处理/` | json_encode、json_decode、序列化 |
| 命令行参数 | `命令行参数/` | cli_args_count、cli_arg、flag解析 |
| 日志系统 | `日志系统/` | debug/info/warn/error、日志级别 |
| 数学函数 | `数学函数/` | PI/E/TAU、三角/双曲/对数、取整、插值 |
| 随机数应用 | `随机数应用/` | random_int、random_float、蒙特卡洛估算π |
| 文件操作 | `文件操作/` | path_join、read/write_file、list_dir、copy |
| MNIST 训练 | `MNIST训练/` | IDX解析、2层MLP、cross_entropy、SGD |
| 智能指针 | `智能指针/` | Box/Rc/Arc/Pin、Weak 弱引用、deref/upgrade |
| Union 类型 | `Union类型/` | tagged union、字段访问/修改（仅 active） |
| Trait 对象 | `Trait对象/` | dyn 动态分发、类型注解驱动升级 |
| 泛型枚举 | `泛型枚举/` | 显式 `<T>` 声明、构造、match 解构 |
| Newtype 模式 | `Newtype模式/` | 元组结构体、`._0` 访问、类型区分 |
| 标签循环 | `标签循环/` | break/continue 'outer、多层循环跳转 |
| Drop 与 Copy | `Drop与Copy/` | Drop/RAII、Copy 自动派生、Phantom 类型 |
| 宏与自定义运算符 | `宏与自定义运算符/` | 声明式宏（嵌套/0参/if体/循环中）、自定义运算符（`@` `$` `~` 组合、优先级、绑定函数）、struct 运算符重载（`impl Add`） |
| Shape 检查演示 | `Shape检查演示/` | shape 检查、matmul 维度 |
| Transformer 示例 | `Transformer示例/` | Self-Attention、GELU、FFN、残差连接 |
| 标准库使用示例 | `标准库使用示例/` | nn::activations、init::initializers、AdamW 公式 |
| 梯度裁剪与累积 | `梯度裁剪与累积/` | clip_grad_by_value、梯度累积概念 |
| 自动微分 Shape 校验 | `自动微分Shape校验/` | autodiff 的 shape 校验、backward 错误 |
| 算力内存预警 | `算力内存预警/` | 编译期算力/内存预估 warning |
| 类型状态 Typestate | `类型状态Typestate/` | 类型状态模式、编译期状态机 |
| 多分类器 | `多分类器/` | 多分类、softmax、交叉熵 |

## 修复与状态标注（2026-08-02 实例目录维护）

- 🔧 **已修复（双路径 🟢 默认 VM + 解释器均通过）**：
  - `Adam优化器/adam.th` — 标量 `pow` 方法改全局函数
  - `Transformer示例/transformer_demo.th`、`标准库使用示例/stdlib_demo.th` — 泛型函数补显式类型参数（`gelu<f64>` 等）
  - `命令行参数/cli_demo.th` — 补 `use std::cli::cli::has_flag/flag_value`（cli 参数 native 已改为读取真实进程参数）
  - `梯度裁剪与累积/grad_clip_accum.th` — `clip_grad_by_value<f64>` 显式类型参数
  - `归并排序/merge_sort.th` — 去掉 while 条件 `&&`（VM 缺陷，已顺带修编译器）、Vec 索引先绑定变量
  - `打家劫舍II/house_robber.th`、`词频统计/word_count.th`、`最长公共子序列/lcs.th` — Vec/HashMap 索引值先绑定变量、`contains_key` 判存在
  - `矩阵乘法/matmul.th` — 避免同名 let 重绑定（VM 缺陷，已顺带修编译器）
  - `微型CNN/cnn.th` — VM `randn` 已支持任意维 shape（原仅 2D 导致 conv2d 权重维度错误）；显示改整张量打印
  - `日志系统/logging_demo.th` — 因顶层 let 对函数不可见的语言级限制，改为无状态显式级别的内联实现
- 🟡 **需解释器路径（VM 结构性缺口，待总师排期，已加头部注释）**：
  - `Trait示例/trait_demo.th` — VM 对具体值 trait 方法分派缺失（call_method_priv 只做字段访问）
  - `闭包合集/closures.th`、`闭包捕获/closure_capture.th` — VM 缺闭包值间接调用（CallIndirect）
  - `矩阵分解/matfact.th`、`边缘检测/sobel.th` — VM 的 `for row in tensor` 迭代缺 tensor.len()

---

*最后更新：2026-08-02*
