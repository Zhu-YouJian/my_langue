//! 命令行参数解析：内存配置 / 文件系统沙箱 / 墙钟超时。
//!
//! 从 `main.rs` 迁移而来（T1.1）。保持函数签名和实现不变，仅改为 `pub`。
//! 这些解析器是纯函数，无副作用，便于复用和测试。

use std::path::Path;
use crate::error::TenthResult;
use crate::runtime::limits::{MemoryConfig, FsSandbox};

/// 从命令行参数解析内存配置。
/// - `--no-limits` → 无限制（用户自担风险）
/// - `--max-memory N` → 自定义 N MiB
/// - 默认 → `MemoryConfig::default()`（256 MiB arena / 2 GiB 张量元素上限）
pub fn parse_memory_config(args: &[String]) -> MemoryConfig {
    if args.iter().any(|a| a == "--no-limits") {
        return MemoryConfig::unbounded();
    }
    if let Some(mb) = args.iter()
        .position(|a| a == "--max-memory")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse::<usize>().ok())
    {
        if mb == 0 {
            return MemoryConfig::unbounded();
        }
        return MemoryConfig {
            max_arena_bytes: mb * 1024 * 1024,
            max_variables: 10_000,
            max_accumulated_defs: 5_000,
            max_tensor_elements: mb * 1024 * 128,
            track_allocations: true,
        };
    }
    MemoryConfig::default()
}

/// H-2: 解析文件系统沙箱选项。
/// - `--fs-root <dir>` → 启用沙箱，根目录为 `dir`
/// - `--read-only` → 沙箱只读（必须配合 `--fs-root`）
/// - `--fs-cwd` → 以当前工作目录为沙箱根（等价 `--fs-root .`）
/// - 默认 → `None`（无沙箱，向后兼容）
///
/// 沙箱启用后，所有 `.th` 程序的文件 I/O 原生函数（read_file/write_file/
/// remove_file/mkdir/copy_file/rename_file/compile_host 等）必须经过
/// FsSandbox::check_read/check_write 校验，防止读写沙箱外的文件
/// （如 ~/.ssh/id_rsa、/etc/passwd）。
pub fn parse_fs_sandbox(args: &[String]) -> TenthResult<Option<FsSandbox>> {
    let read_only = args.iter().any(|a| a == "--read-only");
    if let Some(root) = args.iter()
        .position(|a| a == "--fs-root")
        .and_then(|i| args.get(i + 1))
    {
        let sb = FsSandbox::new(Path::new(root), read_only)
            .map_err(|e| crate::error::TenthError::RuntimeError { line: None, col: None, message: e })?;
        return Ok(Some(sb));
    }
    if args.iter().any(|a| a == "--fs-cwd") {
        let sb = FsSandbox::cwd(read_only)
            .map_err(|e| crate::error::TenthError::RuntimeError { line: None, col: None, message: e })?;
        return Ok(Some(sb));
    }
    if read_only {
        return Err(crate::error::TenthError::RuntimeError { line: None, col: None,
            message: "--read-only 必须配合 --fs-root <dir> 或 --fs-cwd 使用".into(),
        });
    }
    Ok(None)
}

/// H-4: 解析墙钟超时（秒）。
/// - `--timeout <secs>` → 设置超时，返回 `Some(now_ms + secs * 1000)`
/// - 默认 → `None`（无超时，向后兼容）
///
/// 防止 `while true {}` 永久挂起宿主进程。VM 和 Interpreter 在主循环中
/// 周期性检查 `now >= deadline`，超时返回 `TenthError::Timeout`。
pub fn parse_timeout_ms(args: &[String]) -> Option<u128> {
    let secs = args.iter()
        .position(|a| a == "--timeout")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse::<u64>().ok())?;
    let now_ms = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0);
    // 用 checked 防止 secs * 1000 溢出（u64::MAX / 1000 ≈ 10^16 秒，远超实际）
    let timeout_ms = secs.checked_mul(1000)? as u128;
    Some(now_ms.checked_add(timeout_ms)?)
}
