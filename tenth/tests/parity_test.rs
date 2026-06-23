//! Phase D parity tests — verify tenthc (self-hosted) produces WASM that
//! behaves identically to the Rust mother compiler for the same Tenth source.
//!
//! For each test case:
//!   1. Compile `src` with the Rust mother compiler  → WASM-Rust
//!   2. Compile `src` with tenthc (executed via wasmi) → WASM-Tenthc
//!   3. Run both WASMs through wasmi with the same args
//!   4. Assert both return the same value
//!
//! These tests are slow (~1-2s each) because Stage 2 spins up wasmi to run
//! the full tenthc compiler. Skip with `cargo test -- --skip parity`.

#[cfg(test)]
mod parity {
    use wasmi::{Config, Engine, Module, Store, Linker, Caller, StackLimits};
    use tenth::compile::wasm::register_host_functions;
    use std::time::Instant;

    /// Build a wasmi Config with enlarged stack limits for the tenthc compiler's
    /// recursive descent parser and deep call chains. Mirrors three_stage.rs.
    fn selfhost_config() -> Config {
        let mut config = Config::default();
        let limits = StackLimits::new(
            65536,       // initial_value_stack_height
            1_048_576,   // maximum_value_stack_height (1M entries)
            65536,       // maximum_recursion_depth
        ).expect("valid stack limits");
        config.set_stack_limits(limits);
        config.compilation_mode(wasmi::CompilationMode::Eager);
        config
    }

    /// Compile a Tenth source string with the Rust mother compiler → WASM bytes.
    fn compile_via_rust(src: &str) -> Vec<u8> {
        use tenth::lexer::lexer::Lexer;
        use tenth::parser::parser::Parser;
        use tenth::hir::lower::Lowerer;
        use tenth::compile;
        let mut lexer = Lexer::new(src);
        let tokens = lexer.tokenize().expect("lex");
        let mut parser = Parser::new(tokens);
        let program = parser.parse_program().expect("parse");
        let mut lowerer = Lowerer::new();
        let hir = lowerer.lower_program(&program).expect("lower");
        compile::compile_to_wasm(&hir).expect("compile")
    }

    /// Compile a Tenth source string by running tenthc under wasmi → WASM bytes.
    ///
    /// Stage 1: Rust mother compiler compiles `tenthc_src + main(test_src)` → WASM-A
    /// Stage 2: wasmi runs WASM-A; main() invokes tenthc on test_src → WASM-B (Vec<i64>)
    /// Returns WASM-B bytes.
    fn compile_via_tenthc(test_src: &str) -> Vec<u8> {
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

        let escaped = test_src.replace('\\', "\\\\").replace('"', "\\\"");
        let main_src = format!(
            "fn main()->Vec<i64>{{let mut lex=lexer_new(\"{}\");let tokens=lexer_tokenize(&mut lex);let program=parse_program(tokens);let hir=lower_program(program);let wasm=compile_to_wasm(hir);wasm}}",
            escaped
        );
        let full_src = format!("{}\n{}", selfhost_src, main_src);

        // Stage 1: Rust → WASM-A
        let mut lexer = Lexer::new(&full_src);
        let tokens = lexer.tokenize().expect("lex");
        let mut parser = Parser::new(tokens);
        let program = parser.parse_program().expect("parse");
        let mut lowerer = Lowerer::new();
        let hir = lowerer.lower_program(&program).expect("lower");
        let wasm_a = compile::compile_to_wasm(&hir).expect("compile");
        assert_eq!(&wasm_a[..4], b"\0asm", "WASM-A must have valid magic");

        // Stage 2: wasmi runs WASM-A → WASM-B
        let config = selfhost_config();
        let engine = Engine::new(&config);
        let module = Module::new(&engine, &wasm_a).expect("compile wasm-a");
        let mut store = Store::new(&engine, 8192u32);
        let mut linker = Linker::new(&engine);
        register_host_functions(&mut linker).expect("register host functions");

        let inst = linker
            .instantiate(&mut store, &module)
            .expect("inst")
            .start(&mut store)
            .expect("start");
        let main_fn = inst.get_func(&store, "main").expect("main");
        let mut r = [wasmi::Val::I32(0)];
        main_fn.call(&mut store, &[], &mut r).expect("call main");
        let vec_ptr = match r[0] {
            wasmi::Val::I32(v) => v as i64,
            wasmi::Val::I64(v) => v,
            _ => panic!("expected i32/i64 return from main, got {:?}", r[0]),
        };

        // Read Vec<i64> from WASM memory. Layout: cap(8) + len(8) + data_ptr(4) + data...
        let mem = inst.get_memory(&store, "memory").expect("memory");
        let data = mem.data(&store);
        let vp = vec_ptr as i32 as usize;
        assert!(vp + 20 <= data.len(), "vec ptr {} out of range", vp);
        let len = i64::from_le_bytes(data[vp + 8..vp + 16].try_into().unwrap());
        let dp = i32::from_le_bytes(data[vp + 16..vp + 20].try_into().unwrap()) as usize;
        let mut bytes = Vec::with_capacity(len as usize);
        for i in 0..len as usize {
            let pos = dp + i * 8;
            assert!(pos + 8 <= data.len(), "vec data {} out of range", pos);
            let val = i64::from_le_bytes(data[pos..pos + 8].try_into().unwrap());
            bytes.push(val as u8);
        }
        bytes
    }

    /// Set up a wasmi store + linker with host imports for running the *output* WASM.
    /// Registers BOTH `host` (used by Rust mother compiler's wasm.rs) and `env`
    /// (used by tenthc's wasm.th) module namespaces so the same linker can run
    /// WASM from either compiler. Signatures differ between the two compilers
    /// (e.g. tenthc's println takes i64, Rust's takes i32; tenthc's vec_push
    /// returns (), Rust's Vec_push returns i64), so each module is registered
    /// with its own correct signature.
    fn setup_output_store_and_linker(engine: &Engine) -> (Store<()>, Linker<()>) {
        let store = Store::new(engine, ());
        let mut linker = Linker::new(engine);

        // ── `host` module (Rust mother compiler's wasm.rs signatures) ──
        linker.func_wrap("host", "println", |_: Caller<()>, _: i32| {}).unwrap();
        linker.func_wrap("host", "write_file", |_: Caller<()>, _: i32, _: i32| {}).unwrap();
        linker.func_wrap("host", "read_file", |_: Caller<()>, _: i32| -> i32 { 0 }).unwrap();
        linker.func_wrap("host", "str_add", |_: Caller<()>, _: i32, _: i32| -> i32 { 0 }).unwrap();
        linker.func_wrap("host", "str_eq", |_: Caller<()>, _: i32, _: i32| -> i32 { 0 }).unwrap();
        linker.func_wrap("host", "str_int", |_: Caller<()>, _: i64| -> i32 { 0 }).unwrap();
        linker.func_wrap("host", "tenth_alloc", |_: Caller<()>, _: i32| -> i32 { 0 }).unwrap();
        linker.func_wrap("host", "Vec_new", |_: Caller<()>| -> i64 { 0 }).unwrap();
        linker.func_wrap("host", "Vec_push", |_: Caller<()>, _: i64, _: i64| -> i64 { 0 }).unwrap();
        linker.func_wrap("host", "Vec_len", |_: Caller<()>, _: i64| -> i64 { 0 }).unwrap();
        linker.func_wrap("host", "Vec_get", |_: Caller<()>, _: i64, _: i64| -> i64 { 0 }).unwrap();
        linker.func_wrap("host", "compile_host", |_: Caller<()>, _: i32, _: i32| -> i32 { 0 }).unwrap();
        linker.func_wrap("host", "str_len", |_: Caller<()>, _: i32| -> i32 { 0 }).unwrap();
        linker.func_wrap("host", "str_at", |_: Caller<()>, _: i32, _: i64| -> i32 { 0 }).unwrap();
        linker.func_wrap("host", "str_cmp", |_: Caller<()>, _: i32, _: i32, _: i32| -> i32 { 0 }).unwrap();
        linker.func_wrap("host", "f64_bits", |_: Caller<()>, _: f64| -> i64 { 0 }).unwrap();
        linker.func_wrap("host", "str_slice", |_: Caller<()>, _: i32, _: i64, _: i64| -> i32 { 0 }).unwrap();

        // ── `env` module (tenthc wasm.th signatures) ──
        // Type 0: println(i64) -> ()
        linker.func_wrap("env", "println", |_: Caller<()>, _: i64| {}).unwrap();
        // Type 1: vec_new() -> i64
        linker.func_wrap("env", "vec_new", |_: Caller<()>| -> i64 { 0 }).unwrap();
        // Type 2: vec_len(i64) -> i64, read_file(i64) -> i64
        linker.func_wrap("env", "vec_len", |_: Caller<()>, _: i64| -> i64 { 0 }).unwrap();
        linker.func_wrap("env", "read_file", |_: Caller<()>, _: i64| -> i64 { 0 }).unwrap();
        // Type 3: vec_push(i64, i64) -> ()
        linker.func_wrap("env", "vec_push", |_: Caller<()>, _: i64, _: i64| {}).unwrap();
        // Type 4: vec_get(i64, i64) -> i64, write_bytes(i64, i64) -> i64
        linker.func_wrap("env", "vec_get", |_: Caller<()>, _: i64, _: i64| -> i64 { 0 }).unwrap();
        linker.func_wrap("env", "write_bytes", |_: Caller<()>, _: i64, _: i64| -> i64 { 0 }).unwrap();
        // Type 5: str_add(i32, i32) -> i32, str_eq(i32, i32) -> i32, compile_host(i32, i32) -> i32
        linker.func_wrap("env", "str_add", |_: Caller<()>, _: i32, _: i32| -> i32 { 0 }).unwrap();
        linker.func_wrap("env", "str_eq", |_: Caller<()>, _: i32, _: i32| -> i32 { 0 }).unwrap();
        linker.func_wrap("env", "compile_host", |_: Caller<()>, _: i32, _: i32| -> i32 { 0 }).unwrap();
        // Type 6: str_int(i64) -> i32
        linker.func_wrap("env", "str_int", |_: Caller<()>, _: i64| -> i32 { 0 }).unwrap();
        // Type 7: tenth_alloc(i32) -> i32, str_len(i32) -> i32
        linker.func_wrap("env", "tenth_alloc", |_: Caller<()>, _: i32| -> i32 { 0 }).unwrap();
        linker.func_wrap("env", "str_len", |_: Caller<()>, _: i32| -> i32 { 0 }).unwrap();
        // Type 8: str_at(i32, i64) -> i32
        linker.func_wrap("env", "str_at", |_: Caller<()>, _: i32, _: i64| -> i32 { 0 }).unwrap();
        // Type 9: str_cmp(i32, i32, i32) -> i32
        linker.func_wrap("env", "str_cmp", |_: Caller<()>, _: i32, _: i32, _: i32| -> i32 { 0 }).unwrap();

        (store, linker)
    }

    /// Run a WASM module, call `fn_name(args)` and return the i64 result.
    fn run_wasm_i64(wasm: &[u8], fn_name: &str, args: &[i64]) -> i64 {
        assert_eq!(&wasm[..4], b"\0asm", "WASM must have valid magic");
        let engine = Engine::default();
        let module = Module::new(&engine, wasm).expect("module compile");
        let (mut store, linker) = setup_output_store_and_linker(&engine);
        let instance = linker
            .instantiate(&mut store, &module)
            .expect("inst")
            .start(&mut store)
            .expect("start");
        let func = instance.get_func(&store, fn_name).expect("get fn");
        let params: Vec<wasmi::Val> = args.iter().map(|&v| wasmi::Val::I64(v)).collect();
        let mut results = [wasmi::Val::I64(0)];
        func.call(&mut store, &params, &mut results).expect("call");
        match results[0] {
            wasmi::Val::I64(v) => v,
            _ => panic!("unexpected return type"),
        }
    }

    /// Parity assertion: both compilers must produce WASM that returns `expected`
    /// for `fn_name(args)` on `src`.
    fn assert_parity(src: &str, fn_name: &str, args: &[i64], expected: i64) {
        let t0 = Instant::now();
        let wasm_rust = compile_via_rust(src);
        let t1 = Instant::now();
        let wasm_tenthc = compile_via_tenthc(src);
        let t2 = Instant::now();

        let result_rust = run_wasm_i64(&wasm_rust, fn_name, args);
        let t3 = Instant::now();
        let result_tenthc = run_wasm_i64(&wasm_tenthc, fn_name, args);
        let t4 = Instant::now();

        println!(
            "parity[{}]: rust={} (compile {:?}, run {:?}) | tenthc={} (compile {:?}, run {:?}) | expected={}",
            fn_name, result_rust, t1 - t0, t3 - t2, result_tenthc, t2 - t1, t4 - t3, expected
        );

        assert_eq!(result_rust, expected, "Rust compiler path returned wrong value");
        assert_eq!(result_tenthc, expected, "tenthc compiler path returned wrong value");
        assert_eq!(result_rust, result_tenthc, "PARITY BROKEN: Rust and tenthc disagree");
    }

    // ── Basic arithmetic ───────────────────────────────────────────────────

    #[test]
    fn parity_add() {
        let src = "fn add(a: i64, b: i64) -> i64 { a + b }";
        assert_parity(src, "add", &[3, 4], 7);
    }

    #[test]
    fn parity_sub() {
        let src = "fn sub(a: i64, b: i64) -> i64 { a - b }";
        assert_parity(src, "sub", &[10, 3], 7);
    }

    #[test]
    fn parity_mul() {
        let src = "fn mul(a: i64, b: i64) -> i64 { a * b }";
        assert_parity(src, "mul", &[6, 7], 42);
    }

    #[test]
    fn parity_constant() {
        let src = "fn answer() -> i64 { 42 }";
        assert_parity(src, "answer", &[], 42);
    }

    // ── Variables and let bindings ─────────────────────────────────────────

    #[test]
    fn parity_let_reassign() {
        let src = "fn let_test() -> i64 { let x = 1; x = x + 2; x }";
        assert_parity(src, "let_test", &[], 3);
    }

    // ── Control flow ───────────────────────────────────────────────────────

    #[test]
    fn parity_while_count() {
        let src = "fn while_count(n: i64) -> i64 { let i = 0; while i < n { i = i + 1; } i }";
        assert_parity(src, "while_count", &[5], 5);
    }

    #[test]
    fn parity_for_sum() {
        // for-loop sum: tenthc wasm.th supports for-range; Rust mother compiler
        // also supports it (Phase A5).
        let src = "fn for_sum(n: i64) -> i64 { let mut s: i64 = 0; for i in 0..n { s = s + i; }; s }";
        assert_parity(src, "for_sum", &[5], 10);
    }

    #[test]
    fn parity_nested_calls() {
        let src = "fn sq(x: i64) -> i64 { x * x } fn sum_sq(a: i64, b: i64) -> i64 { sq(a) + sq(b) }";
        assert_parity(src, "sum_sq", &[3, 4], 25);
    }

    // ── Comparisons and conditionals ───────────────────────────────────────

    #[test]
    fn parity_div_mod() {
        let src = "fn divmod(a: i64, b: i64) -> i64 { let q = a / b; let r = a % b; q * 100 + r }";
        assert_parity(src, "divmod", &[17, 5], 302);
    }

    #[test]
    fn parity_max_if() {
        // if-expression: both compilers should support this.
        let src = "fn my_max(a: i64, b: i64) -> i64 { if a > b { a } else { b } }";
        assert_parity(src, "my_max", &[3, 7], 7);
    }

    #[test]
    fn parity_abs_if() {
        let src = "fn my_abs(x: i64) -> i64 { if x < 0 { 0 - x } else { x } }";
        assert_parity(src, "my_abs", &[-5], 5);
    }

    #[test]
    fn parity_comparison_chain() {
        // Use comparison results in arithmetic (bool → i64).
        let src = "fn cmp_test(a: i64, b: i64) -> i64 { let lt = if a < b { 1 } else { 0 }; let eq = if a == b { 1 } else { 0 }; lt * 10 + eq }";
        assert_parity(src, "cmp_test", &[3, 5], 10);
    }

    // ── Struct field access ────────────────────────────────────────────────

    #[test]
    fn parity_struct_field() {
        // struct literal + field access. Both compilers allocate struct on
        // heap (tenth_alloc) and access fields by offset.
        let src = "struct Pair { a: i64, b: i64 } fn make_and_sum(x: i64, y: i64) -> i64 { let p = Pair { a: x, b: y }; p.a + p.b }";
        assert_parity(src, "make_and_sum", &[3, 4], 7);
    }

    #[test]
    fn parity_struct_reassign() {
        // Create struct, mutate field, read it back.
        let src = "struct Counter { n: i64 } fn increment(start: i64) -> i64 { let mut c = Counter { n: start }; c.n = c.n + 10; c.n }";
        assert_parity(src, "increment", &[5], 15);
    }

    // ── Advanced if-expression and control flow ───────────────────────────

    #[test]
    fn parity_nested_if() {
        // Nested if-else-if chain
        let src = "fn classify(x: i64) -> i64 { if x < 0 { 0 - 1 } else { if x == 0 { 0 } else { 1 } } }";
        assert_parity(src, "classify", &[-5], 0 - 1);
        // Also test with 0 and positive
        let wasm = compile_via_tenthc(src);
        assert_eq!(run_wasm_i64(&wasm, "classify", &[0]), 0);
        assert_eq!(run_wasm_i64(&wasm, "classify", &[42]), 1);
    }

    #[test]
    fn parity_if_no_else() {
        // if without else as a statement (not producing a value)
        let src = "fn maybe_add(x: i64) -> i64 { let mut r = x; if x < 10 { r = r + 100; }; r }";
        assert_parity(src, "maybe_add", &[5], 105);
        // x >= 10: r stays x
        let wasm = compile_via_tenthc(src);
        assert_eq!(run_wasm_i64(&wasm, "maybe_add", &[20]), 20);
    }

    #[test]
    fn parity_early_return() {
        // Early return inside if
        let src = "fn sign(x: i64) -> i64 { if x < 0 { return 0 - 1; }; if x > 0 { return 1; }; 0 }";
        assert_parity(src, "sign", &[-5], 0 - 1);
        let wasm = compile_via_tenthc(src);
        assert_eq!(run_wasm_i64(&wasm, "sign", &[5]), 1);
        assert_eq!(run_wasm_i64(&wasm, "sign", &[0]), 0);
    }

    #[test]
    fn parity_bool_logic() {
        // Boolean logic: && and ||
        let src = "fn band(a: i64, b: i64) -> i64 { if a > 0 && b > 0 { 1 } else { 0 } }";
        assert_parity(src, "band", &[1, 1], 1);
        let wasm = compile_via_tenthc(src);
        assert_eq!(run_wasm_i64(&wasm, "band", &[1, 0]), 0);
        assert_eq!(run_wasm_i64(&wasm, "band", &[0, 1]), 0);
    }

    #[test]
    fn parity_multi_stmt_fn() {
        // Function with multiple statements before if
        let src = "fn compute(a: i64, b: i64) -> i64 { let s = a + b; let d = a - b; if s > d { s } else { d } }";
        assert_parity(src, "compute", &[3, 4], 7);
    }

    #[test]
    fn parity_if_in_loop() {
        // if inside a while loop — tests loop body + if interaction
        let src = "fn sum_odd(n: i64) -> i64 { let mut i = 0; let mut s = 0; while i < n { if i % 2 == 1 { s = s + i; }; i = i + 1; } s }";
        assert_parity(src, "sum_odd", &[10], 25); // 1+3+5+7+9 = 25
    }

    // ── Debug tests for if-expression investigation ────────────────────────

    #[test]
    fn debug_if_true() {
        // if with true condition: should return then-branch (42)
        let src = "fn test_if() -> i64 { if 1 > 0 { 42 } else { 99 } }";
        let wasm_tenthc = compile_via_tenthc(src);
        let result = run_wasm_i64(&wasm_tenthc, "test_if", &[]);
        println!("debug_if_true: tenthc returned {} (expected 42)", result);
        let wasm_rust = compile_via_rust(src);
        let result_rust = run_wasm_i64(&wasm_rust, "test_if", &[]);
        println!("debug_if_true: rust returned {} (expected 42)", result_rust);
    }

    #[test]
    fn debug_if_false() {
        // if with false condition: should return else-branch (99)
        let src = "fn test_if() -> i64 { if 0 > 1 { 42 } else { 99 } }";
        let wasm_tenthc = compile_via_tenthc(src);
        println!("=== tenthc WASM-B ({} bytes) ===", wasm_tenthc.len());
        for (i, b) in wasm_tenthc.iter().enumerate() {
            print!("{:02x} ", b);
            if (i + 1) % 16 == 0 { println!(); }
        }
        println!();
        let result = run_wasm_i64(&wasm_tenthc, "test_if", &[]);
        println!("debug_if_false: tenthc returned {} (expected 99)", result);
    }

    #[test]
    fn debug_if_no_braces() {
        // Simple if without block braces — tenthc parser may handle this differently
        let src = "fn test_if(x: i64) -> i64 { if x > 0 { 1 } else { 0 } }";
        let wasm_tenthc = compile_via_tenthc(src);
        let result = run_wasm_i64(&wasm_tenthc, "test_if", &[5]);
        println!("debug_if_no_braces(5): tenthc returned {} (expected 1)", result);
        let result0 = run_wasm_i64(&wasm_tenthc, "test_if", &[-5]);
        println!("debug_if_no_braces(-5): tenthc returned {} (expected 0)", result0);
    }
}
