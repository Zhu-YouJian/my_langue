fn main() {
    // Increase stack size to 8 MiB to avoid overflow in debug builds
    // where the tree-walk interpreter creates deep call chains.
    if std::env::var("PROFILE").unwrap_or_default() == "debug" {
        println!("cargo:rustc-link-arg=/STACK:8388608");
    }
}
