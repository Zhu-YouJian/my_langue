//! NativeRegistry：VM/Interpreter 统一的 native 函数注册入口。
//!
//! T2 产物。当前为**最小实现**：`NativeRegistry::register_all(&mut vm)`
//! 委托给 `crate::runtime::natives::register_all_natives(vm)`，保持行为不变。
//!
//! ## 设计意图
//!
//! 历史问题：VM native 通过 `register_all_natives(vm)` 注册（`vm.add_native`），
//! Interpreter native 通过 `Interpreter::call_named_fn` 分派，两套机制独立维护
//! 容易产生"硬缺口"（如 zeros/ones 曾仅在 interpreter 实现，导致 VM 路径返回 Unit）。
//!
//! `NativeRegistry` 作为**统一入口点**，后续可扩展为：
//! 1. 同时注册到 VM 和 Interpreter
//! 2. 按 native 集合分组（I/O / 张量 / autodiff / 文件系统 ...）按需注册
//! 3. 提供 native 函数清单查询（供测试部做 parity 检查）
//!
//! 当前阶段仅做委托，确保不破坏现有行为，为后续统一奠定结构基础。

use crate::runtime::natives::register_all_natives;
use crate::runtime::vm::Vm;

/// Native 函数注册器。
///
/// 统一 VM 与 Interpreter 的 native 注册入口。当前实现委托给
/// `register_all_natives`，后续可扩展为同时注册到两条执行路径。
pub struct NativeRegistry;

impl NativeRegistry {
    /// 注册所有 native 函数到给定 VM。
    ///
    /// 当前委托给 `crate::runtime::natives::register_all_natives`。
    /// 后续扩展时，可在此方法中同时向 Interpreter 注册，
    /// 或根据 `feature_flags` 选择性注册子集。
    pub fn register_all(vm: &mut Vm) {
        register_all_natives(vm);
    }
}
