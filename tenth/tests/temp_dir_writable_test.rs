//! 常驻守卫（W1 环境可信度）：`std::env::temp_dir()` 必须**可写**——建目录 / 写文件 /
//! 读回校验 / 删干净。
//!
//! WHY 这条守卫存在
//! ----------------
//! 本机沙箱下，位于工作区内的可执行文件以 **Low 完整性**（S-1-16-4096）运行；写入不带
//! 低完整性标签的 `%TEMP%` 被操作系统拒绝（典型 `Os { code: 5, kind: PermissionDenied,
//! message: "拒绝访问。" }`）。历史上这表现为 **12 个测试目标集体假红**——不是代码回归，
//! 却足以让人误判代码。
//!
//! 修复（把 `TMP`/`TEMP` 钉到工作区内的 `tenth/target/tmp`，`force = true`）写在仓库根的
//! `.cargo/config.toml` 的 `[env]` 表里；本文件是它的**守卫**：那条配置一旦被删、被改成
//! 非 force（在 `TEMP` 已存在的 shell 里就是静默 no-op）、或目标目录不再可写，**这里必须
//! 先红**，而不是等别人在别的测试上撞见一片 `PermissionDenied` 再反推是不是回归。
//!
//! WHY 只断言「可写」，绝不断言「等于 tenth/target/tmp」
//! ------------------------------------------------
//! Linux/macOS 的 `std::env::temp_dir()` 读 `TMPDIR`（无 `TMPDIR` 时退回 `/tmp`），
//! **不读** `TMP`/`TEMP` ⇒ 钉死路径的断言会在三平台 CI 上假红。本守卫要抓的是
//! 「`temp_dir()` 不可写」这个真故障，而不是某个具体路径。
//!
//! 本测试**常驻默认套件**（无 `#[ignore]`）：它是环境保证的一部分，跳过它就等于没保证。

use std::path::PathBuf;

/// 探针协议：在 `temp_dir()` 下建唯一子目录 → 写文件 → 读回校验 → 删文件 → 删目录。
///
/// 返回 `Ok(探针所用基目录)` / `Err(可操作原因)`。刻意**不**断言具体路径。
fn probe_temp_dir() -> Result<PathBuf, String> {
    let base = std::env::temp_dir();

    // 0) 基目录必须**已经存在**：有测试直接往 temp_dir() 里写文件而不先建目录
    //    （mnist_loader_test / native_parity_test / stdlib_test / tensor_features_test），
    //    所以「能创建它」不等于「它可用」。这里不兜底创建，缺了就是要红。
    if !base.is_dir() {
        return Err(
            "temp_dir() 不存在或不是目录（直接往 temp_dir() 写文件的测试会以 \
             Os { code: 3, kind: NotFound } 假红）"
                .to_string(),
        );
    }

    let dir = base.join(format!("tenth_temp_probe_{}", std::process::id()));

    // 1) 建子目录（可写性的第一道关：低完整性进程在这里就已被拒）
    std::fs::create_dir_all(&dir).map_err(|e| format!("create_dir_all 失败：{}", e))?;

    // 2) 写文件
    let file = dir.join("probe.txt");
    std::fs::write(&file, b"tenth temp probe").map_err(|e| format!("写入失败：{}", e))?;

    // 3) 读回校验（防「写着成功但内容不落地」的伪可写）
    let back = std::fs::read(&file).map_err(|e| format!("读回失败：{}", e))?;
    if back != b"tenth temp probe" {
        return Err(format!("读回内容与写入不一致（{} 字节）", back.len()));
    }

    // 4) 删干净（临时目录不该留垃圾；删不掉同样是「不可用」）
    std::fs::remove_file(&file).map_err(|e| format!("删除文件失败：{}", e))?;
    std::fs::remove_dir(&dir).map_err(|e| format!("删除目录失败：{}", e))?;

    Ok(base)
}

#[test]
fn temp_dir_is_writable() {
    match probe_temp_dir() {
        Ok(base) => {
            // 成功路径静默（cargo 默认吞掉 stdout/stderr）；`--nocapture` 时可见。
            eprintln!(
                "[temp_dir_writable] OK: std::env::temp_dir() 建/写/读/删探针通过（{}）",
                base.display()
            );
        }
        Err(reason) => {
            let base = std::env::temp_dir();
            panic!(
                "\n\
                 === temp_dir 守卫失败：std::env::temp_dir() 不可用 ===\n\
                 \n\
                 位置  : {}\n\
                 原因  : {}\n\
                 \n\
                 这不是代码回归，是**运行环境**问题：\n\
                 工作区内（如 tenth/target/ 下）的可执行文件在 Windows 上以 **Low 完整性**\n\
                 （S-1-16-4096）运行，而该目录不带低完整性标签 ⇒ 低完整性进程**向上写**被\n\
                 操作系统拒绝（典型 Os {{ code: 5, kind: PermissionDenied, \"拒绝访问。\" }}）。\n\
                 历史上这会让 12 个直接使用 temp_dir() 的测试目标集体假红。\n\
                 \n\
                 怎么办（按顺序）：\n\
                 1. 读 DEPS.md 的「写边界与临时目录（受限沙箱下的机器保证）」一节；\n\
                 2. 确认仓库根 .cargo/config.toml 的 [env] 表里这两条仍在，且**必须**带\n\
                 force = true（cargo [env] 默认不覆盖已存在的变量：shell 里 TEMP 早已存在，\n\
                 非 force 版是静默 no-op，正是本文件与 config 注释批评的那种「假 mitigation」）：\n\
                 \n\
                     TMP  = {{ value = \"tenth/target/tmp\", relative = true, force = true }}\n\
                     TEMP = {{ value = \"tenth/target/tmp\", relative = true, force = true }}\n\
                 \n\
                 3. 确认 <仓库根>/tenth/target/tmp 目录存在（由 tenth/build.rs 在构建期创建）。\n\
                 若缺失，让构建脚本重跑一次——**必须在 tenth/ 下、且必须带 --release**：\n\
                 \n\
                     cd tenth; cargo clean -p tenth --release\n\
                     # 或直接脏化构建脚本：touch tenth/build.rs\n\
                 \n\
                 实测（C1b 复核更正）：不带 --release 的 `cargo clean -p tenth` **无效**——\n\
                 它不会触发 `Compiling`、目录不会回来、直接往 temp_dir() 写文件的测试仍红；\n\
                 且在仓库根直接跑该命令会报找不到 Cargo.toml。手工 mkdir 只是权宜，\n\
                 机器保证靠构建脚本；\n\
                 4. 分别从仓库根 `cargo test --release --manifest-path tenth/Cargo.toml` 与\n\
                 `cd tenth` 后 `cargo test --release` 各跑一次，两种 cwd 都必须绿。\n\
                 \n\
                 注：本守卫**只断言可写**，不断言 temp_dir() 等于 tenth/target/tmp\n\
                 （Linux/macOS 读 TMPDIR，钉路径会在三平台 CI 假红）。\n",
                base.display(),
                reason
            );
        }
    }
}
