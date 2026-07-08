//! 树遍解释器（拆分自原 `interpreter.rs` 上帝文件）。
//!
//! 模块结构（架构重构 T3e 拆分）：
//! - `mod.rs`（本文件）：薄导出层，仅声明子模块并 re-export `Interpreter`
//! - `core.rs`：`Interpreter` 结构体、构造函数、作用域管理、执行入口、
//!   `tick`、`make_tensor`、`unwrap_return`
//! - `eval.rs`：`eval_expr` / `eval_call` / `eval_stmt` 执行逻辑
//! - `autodiff_helpers.rs`：自动微分记录辅助（`record_binary` / `record_unary`）
//! - `json.rs`：JSON 编解码（带 H-6 安全修复）
//! - `datetime.rs`：Unix 天数 → (年, 月, 日) 转换
//! - `binary.rs`：二元/一元运算、值比较与字符串化
//! - `pattern.rs`：字段访问与模式匹配
//! - `methods.rs`：方法分派（String/Vec/Map/Range/Iterator/Tensor/Scalar）
//! - `index.rs`：索引与切片
//! - `natives.rs`：原生函数注册（`call_named_fn`）

pub mod json;
pub mod datetime;
mod core;
mod eval;
mod autodiff_helpers;
mod binary;
mod pattern;
mod methods;
mod natives;
mod index;

pub use core::Interpreter;
