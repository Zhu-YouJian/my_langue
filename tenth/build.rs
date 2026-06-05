fn main() {
    // Increase stack size to 16 MiB to handle deep call chains
    // in the tree-walk interpreter (especially self-hosting parser).
    println!("cargo:rustc-link-arg=/STACK:16777216");
}
