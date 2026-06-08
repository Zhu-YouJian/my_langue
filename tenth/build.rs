fn main() {
    // Increase stack size to 64 MiB to handle deep call chains
    // in the tree-walk interpreter (especially self-hosting parser + bridge).
    println!("cargo:rustc-link-arg=/STACK:67108864");
}
