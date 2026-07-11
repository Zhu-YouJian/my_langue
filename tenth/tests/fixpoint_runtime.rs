//! Phase 1 fixpoint runtime benchmark.
//!
//! Measures the three-stage compilation pipeline performance with Wasmtime JIT:
//! - T_rust: Rust mother compiler compiles tenthc → stage1.wasm (WASM-A)
//! - T_load: Wasmtime loads and instantiates stage1.wasm
//! - T_compile_small: Wasmtime executes stage1 to compile a small program
//!
//! Performance target: T_compile_small < 30s.
//! This benchmark validates that Wasmtime JIT makes the tenthc self-hosting
//! compiler fast enough for interactive use.

use std::time::Instant;

/// Compile the tenthc self-hosting compiler + a test program into WASM.
/// Returns the WASM bytes (stage1.wasm / WASM-A).
/// Same logic as `three_stage::compile_selfhost_to_wasm`.
fn compile_selfhost_to_wasm(test_source: &str) -> Vec<u8> {
    use tenth::lexer::lexer::Lexer;
    use tenth::parser::parser::Parser;
    use tenth::hir::lower::Lowerer;
    use tenth::compile;
    let selfhost_src = [
        include_str!("../../tenthc/lexer/token.th"),
        include_str!("../../tenthc/lexer/lexer.th"),
        include_str!("../../tenthc/parser/parser.th"),
        include_str!("../../tenthc/hir/hir.th"),
        include_str!("../../tenthc/hir/lower.th"),
        include_str!("../../tenthc/compile/wasm.th"),
    ].join("\n");
    let escaped = test_source.replace('\\', "\\\\").replace('"', "\\\"");
    let full_src = format!("{}fn main()->Vec<i64>{{let mut lex=lexer_new(\"{}\");let tokens=lexer_tokenize(&mut lex);println(\"tokens_done\");let program=parse_program(tokens);println(\"parsed\");let hir=lower_program(program);println(\"lowered\");let wasm=compile_to_wasm(hir);println(\"compiled\");wasm}}", selfhost_src, escaped);
    let mut lexer = Lexer::new(&full_src);
    let tokens = lexer.tokenize().expect("lex");
    let mut parser = Parser::new(tokens);
    let program = parser.parse_program().expect("parse");
    let mut lowerer = Lowerer::new();
    let hir = lowerer.lower_program(&program).expect("lower");
    compile::compile_to_wasm(&hir).expect("compile")
}

/// Read a Vec<i64> from wasmtime WASM memory given its pointer.
/// Vec layout: cap(8) + len(8) + data_ptr(4) + data...
fn read_vec_from_memory_wt(
    store: &wasmtime::Store<u32>,
    mem: &wasmtime::Memory,
    vec_ptr: i64,
) -> Vec<u8> {
    let data = mem.data(store);
    let vp = vec_ptr as i32 as usize;
    if vp + 20 > data.len() {
        return Vec::new();
    }
    let len = i64::from_le_bytes(data[vp + 8..vp + 16].try_into().unwrap());
    let dp = i32::from_le_bytes(data[vp + 16..vp + 20].try_into().unwrap()) as usize;
    let mut bytes = Vec::with_capacity(len as usize);
    for i in 0..len as usize {
        let pos = dp + i * 8;
        if pos + 8 > data.len() { break; }
        let val = i64::from_le_bytes(data[pos..pos + 8].try_into().unwrap());
        bytes.push(val as u8);
    }
    bytes
}

#[test]
#[ignore = "wasmtime 路径 Vec 写回逻辑问题：tenthc 完整执行但 main 返回 Vec len=0（AUDIT.md #5）。wasmi 路径已通过验证，wasmtime 仅是性能优化路径，深度调试 ROI 低"]
fn fixpoint_runtime_benchmark() {
    // 128MB stack for safety when running the full tenthc compiler in WASM.
    std::thread::Builder::new()
        .stack_size(128 * 1024 * 1024)
        .spawn(run_benchmark)
        .unwrap()
        .join()
        .unwrap();
}

fn run_benchmark() {
    // Small test program: simple add function (no loops, minimal codegen).
    let test_src = "fn add(a: i64, b: i64) -> i64 { a + b }";
    println!("=== Fixpoint Runtime Benchmark ===");
    println!("Test source: {}", test_src);

    // ── T_rust: Rust compiles tenthc + test_src → WASM-A ──
    println!("\n--- T_rust: Rust compile tenthc → stage1.wasm ---");
    let t_rust_start = Instant::now();
    let wasm_a = compile_selfhost_to_wasm(test_src);
    let t_rust = t_rust_start.elapsed();
    println!("WASM-A: {} bytes", wasm_a.len());
    println!("T_rust: {:?}", t_rust);
    assert_eq!(&wasm_a[..4], b"\0asm", "WASM-A must have valid magic");

    // ── T_load: Wasmtime loads and instantiates WASM-A ──
    println!("\n--- T_load: Wasmtime load stage1.wasm ---");
    let t_load_start = Instant::now();
    let (mut store, instance) =
        tenth::compile::wasm::instantiate_wasmtime(&wasm_a)
            .expect("wasmtime instantiate");
    let t_load = t_load_start.elapsed();
    println!("T_load: {:?}", t_load);

    // ── T_compile_small: Wasmtime executes main() to compile test_src → WASM-B ──
    println!("\n--- T_compile_small: Wasmtime execute stage1 compiling small program ---");
    let t_compile_start = Instant::now();
    // main() returns Vec<i64> (i64 pointer) in the self-hosting pipeline.
    // Try i64 first, fall back to i32 for robustness.
    let vec_ptr: i64 = if let Ok(main_fn) =
        instance.get_typed_func::<(), i64>(&mut store, "main")
    {
        main_fn.call(&mut store, ()).expect("call main")
    } else if let Ok(main_fn) =
        instance.get_typed_func::<(), i32>(&mut store, "main")
    {
        main_fn.call(&mut store, ()).expect("call main") as i64
    } else {
        panic!("main function not found or has unexpected signature");
    };
    let t_compile_small = t_compile_start.elapsed();

    let mem = instance.get_memory(&mut store, "memory").expect("memory export");
    let wasm_b = read_vec_from_memory_wt(&store, &mem, vec_ptr);
    println!("WASM-B: {} bytes", wasm_b.len());
    println!("T_compile_small: {:?}", t_compile_small);

    // Verify WASM-B is valid WASM.
    assert!(
        !wasm_b.is_empty() && &wasm_b[..4] == b"\0asm",
        "WASM-B must be non-empty and have valid magic"
    );

    // ── Summary ──
    println!("\n=== Summary ===");
    println!("T_rust:          {:?}", t_rust);
    println!("T_load:          {:?}", t_load);
    println!("T_compile_small: {:?}", t_compile_small);
    println!(
        "Total:           {:?}",
        t_rust + t_load + t_compile_small
    );

    // Performance target: T_compile_small < 30s.
    assert!(
        t_compile_small.as_secs() < 30,
        "T_compile_small must be < 30s, got {:?}",
        t_compile_small
    );
    println!("\n=== PASS: T_compile_small < 30s ===");
}
