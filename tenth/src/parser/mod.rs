pub mod ast;
pub mod parser;
mod expr;
mod stmt;
mod decl;
mod type_parser;
/// M3.3：声明式宏展开 pass（parse 完成后、lower 前执行）。
mod macro_expand;