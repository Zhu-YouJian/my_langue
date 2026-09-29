fn main() {
    // Increase stack size to 64 MiB to handle deep call chains
    // in the tree-walk interpreter (especially self-hosting parser + bridge).
    //
    // M5.3 跨平台（Win/Linux/macOS）：链接参数按平台区分——MSVC 用 `/STACK:`、
    // GNU ld 用 `-z stack-size`、Apple ld 用 `-stack_size`。原实现硬编码
    // `/STACK:67108864` 在非 Windows 平台会被 GNU/Apple 链接器当作输入文件
    // 处理而报 "cannot find /STACK:67108864"，阻塞 Linux/macOS 构建。
    let os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let env = std::env::var("CARGO_CFG_TARGET_ENV").unwrap_or_default();
    const STACK: &str = "67108864"; // 64 MiB

    match os.as_str() {
        "windows" if env == "msvc" => {
            // MSVC 链接器：/STACK:<bytes>
            println!("cargo:rustc-link-arg=/STACK:{}", STACK);
        }
        "windows" => {
            // GNU/MinGW 链接器：--stack=<bytes>
            println!("cargo:rustc-link-arg=-Wl,--stack,{}", STACK);
        }
        "linux" => {
            // GNU ld（glibc/musl 通用）：-z stack-size=<bytes>
            println!("cargo:rustc-link-arg=-Wl,-z,stack-size={}", STACK);
        }
        "macos" => {
            // Apple ld：-stack_size <bytes>（需页大小倍数；64 MiB 满足）
            println!("cargo:rustc-link-arg=-Wl,-stack_size,{}", STACK);
        }
        _ => {
            // 其他平台（FreeBSD 等）保守不加栈大小参数（链接器语法差异大）
            eprintln!(
                "[build.rs] 未配置平台栈大小链接参数，跳过（target_os={}）",
                os
            );
        }
    }

    create_temp_dir();
}

/// W1（2026-09-17）环境可信度：创建 `.cargo/config.toml` 的 `[env]` 里 `TMP`/`TEMP`
/// 指向的目录（`<package>/target/tmp`）。
///
/// WHY 必须在这里做（而不是靠人 mkdir）
/// -----------------------------------
/// 仓库根的 `.cargo/config.toml` 把 `TMP`/`TEMP` **强制**（`force = true`）钉到
/// `tenth/target/tmp`，因为本机沙箱下工作区内的可执行文件以 Low 完整性运行、写真正的
/// `%TEMP%` 被拒（见 `DEPS.md`「写边界与临时目录（受限沙箱下的机器保证）」）。但 cargo `[env]` **只设变量、不建目录**，
/// 而部分测试是**直接**往 `temp_dir()` 里写文件、不先建目录的
/// （`mnist_loader_test` / `native_parity_test` / `stdlib_test` / `tensor_features_test`）。
/// 目录缺失时它们会以 `Os { code: 3, kind: NotFound }` 假红——把一个环境假红换成另一个
/// 环境假红，等于修了个寂寞。构建期创建 ⇒ `cargo clean` 后重新构建自动恢复，
/// 不依赖任何人的记忆（这正是本轮「把纪律换成机器保证」的诉求）。
///
/// 注：刻意**不**发 `cargo:rerun-if-changed=target/tmp`——那是测试的读写草稿目录，
/// 内容一直在变，盯着它会让构建脚本每次 cargo 调用都重跑（连锁重编译）。
///
/// 残留窗口（C1b 复核更正后的口径）：在 target/ 完好时手工删掉该目录，构建脚本不会自动重跑。
///   * 触发条件收窄：**过滤跑**（`cargo test --test <直接写 temp_dir() 的目标>`）会撞上；
///     **整跑通常自愈**——`audit_11485_regression_test`、`import_order_test`、
///     `s3a_runtime_gap_test`、`module_search_path_test` 等有 `create_dir_all(temp_dir().join(..))`
///     的用例会**连带建出父目录**（C1b 复核：目录缺失时单跑 `audit_11485_regression_test`
///     ⇒ 无 `Compiling`、9 passed、目录自行出现）。
///   * 真修法是让**构建脚本**重跑（否则又是靠人 mkdir）：
///     `cd tenth; cargo clean -p tenth --release`（不带 `--release` 无效：不会触发
///     `Compiling`、目录不回来）或 `touch tenth/build.rs`。
///   * 常驻守卫 `tenth/tests/temp_dir_writable_test.rs` 在残留窗口下会红并给出上面的修法。
fn create_temp_dir() {
    let Ok(manifest_dir) = std::env::var("CARGO_MANIFEST_DIR") else {
        return;
    };
    let tmp = std::path::Path::new(&manifest_dir)
        .join("target")
        .join("tmp");
    if let Err(e) = std::fs::create_dir_all(&tmp) {
        // 不 panic：创建失败时由守卫测试给出可操作的失败文案，构建本身照常继续。
        println!(
            "cargo:warning=无法创建 temp_dir 目标目录（{}）：{}",
            tmp.display(),
            e
        );
    }
}
