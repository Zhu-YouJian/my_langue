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
}
