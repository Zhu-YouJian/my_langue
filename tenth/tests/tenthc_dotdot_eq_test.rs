//! Regression tests for AUDIT-11.4.4: tenthc `..=` lexer bug fix.
//!
//! Background: `tenthc/lexer/lexer.th:180-184` previously mis-tokenized
//! `..=` as `..` + `=` (DotDot followed by Assign), because the `.` branch
//! checked `next == "."` and returned `DotDot` immediately without
//! inspecting the third character. This caused `for i in 1..=3 { ... }`
//! to fail parsing in the tenthc self-hosting pipeline.
//!
//! Fix (`tenthc/lexer/lexer.th:180-190`): after matching `..`, peek one
//! more char and check for `=`; if matched, advance and return
//! `DotDotEq` (disc=62). Also handles `.` `=` via the `next == "="`
//! branch (line 188).
//!
//! These tests verify the fix through the full tenthc self-hosting
//! pipeline (Stage 1: Rust mother compiler → WASM-A; Stage 2: wasmi runs
//! WASM-A; tenthc compiles test source → WASM-B; Stage 3: wasmi runs
//! WASM-B and asserts the i64 result).
//!
//! Coverage:
//!   1. tenthc lexes `..=` without parser error (regression for the
//!      mis-tokenization)
//!   2. `for i in 1..=3` iterates 1, 2, 3 (inclusive upper bound)
//!   3. `for i in 2..=4` iterates 2, 3, 4 (different boundary, ensures
//!      `..=` is not accidentally treated as `..` exclusive)
//!
//! The tenthc WASM backend already supports Range inclusive codegen
//! (`tenthc/compile/wasm.th:1442` checks `iter.range_inclusive` and
//! emits `i64.le_s` instead of `i64.lt_s`), so these tests cover the
//! full lexer → parser → HIR → WASM path on the tenthc side.

#[cfg(test)]
mod tenthc_dotdot_eq {
    use wasmi::{Config, Engine, Linker, Module, StackLimits, Store, Caller};
    use tenth::compile::wasm::register_host_functions;

    /// Build a wasmi Config with enlarged stack limits for recursive
    /// functions and the tenthc compiler's recursive descent parser.
    fn selfhost_config() -> Config {
        let mut config = Config::default();
        let limits = StackLimits::new(
            65536,       // initial_value_stack_height
            1_048_576,   // maximum_value_stack_height (1M entries)
            65536,       // maximum_recursion_depth
        ).expect("valid stack limits");
        config.set_stack_limits(limits);
        config
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
    /// Registers BOTH `host` (Rust mother compiler's wasm.rs signatures) and `env`
    /// (tenthc's wasm.th signatures) module namespaces so the same linker can run
    /// WASM from either compiler.
    fn setup_output_store_and_linker(engine: &Engine) -> (Store<()>, Linker<()>) {
        use std::sync::Arc;
        use std::sync::atomic::{AtomicU32, Ordering};
        let store = Store::new(engine, ());
        let mut linker = Linker::new(engine);
        // Bump allocator pointer shared between host/env modules. Starts at 4096
        // to avoid overlapping with string data (stored at low offsets).
        let bump = Arc::new(AtomicU32::new(4096));

        // ── `host` module (Rust mother compiler's wasm.rs signatures) ──
        linker.func_wrap("host", "println", |_: Caller<()>, _: i32| {}).unwrap();
        linker.func_wrap("host", "write_file", |_: Caller<()>, _: i32, _: i32| {}).unwrap();
        linker.func_wrap("host", "read_file", |_: Caller<()>, _: i32| -> i32 { 0 }).unwrap();
        linker.func_wrap("host", "str_add", |_: Caller<()>, _: i32, _: i32| -> i32 { 0 }).unwrap();
        linker.func_wrap("host", "str_eq", |_: Caller<()>, _: i32, _: i32| -> i32 { 0 }).unwrap();
        linker.func_wrap("host", "str_int", |_: Caller<()>, _: i64| -> i32 { 0 }).unwrap();
        let b = bump.clone();
        linker.func_wrap("host", "tenth_alloc", move |mut caller: Caller<()>, size: i32| -> i32 {
            let ptr = b.fetch_add(size as u32, Ordering::SeqCst);
            let needed = ptr as usize + size as usize;
            let mem = caller.get_export("memory").and_then(|e| e.into_memory()).unwrap();
            let mut current_len = mem.data(&caller).len();
            while needed > current_len {
                let pages = ((needed - current_len + 65535) / 65536) as u32;
                mem.grow(&mut caller, pages).ok();
                let new_len = mem.data(&caller).len();
                if new_len == current_len { break; }
                current_len = new_len;
            }
            ptr as i32
        }).unwrap();
        linker.func_wrap("host", "Vec_new", |_: Caller<()>| -> i64 { 0 }).unwrap();
        linker.func_wrap("host", "Vec_push", |_: Caller<()>, _: i64, _: i64| -> i64 { 0 }).unwrap();
        linker.func_wrap("host", "Vec_len", |_: Caller<()>, _: i64| -> i64 { 0 }).unwrap();
        linker.func_wrap("host", "Vec_get", |_: Caller<()>, _: i64, _: i64| -> i64 { 0 }).unwrap();
        linker.func_wrap("host", "compile_host", |_: Caller<()>, _: i32, _: i32| -> i32 { 0 }).unwrap();
        linker.func_wrap("host", "str_len", |caller: Caller<()>, ptr: i32| -> i32 {
            let mem = caller.get_export("memory").and_then(|e| e.into_memory()).unwrap();
            let data = mem.data(&caller);
            let base = ptr as usize;
            let mut len: i32 = 0;
            while base + (len as usize) < data.len() && data[base + (len as usize)] != 0 {
                len += 1;
            }
            len
        }).unwrap();
        linker.func_wrap("host", "str_at", |_: Caller<()>, _: i32, _: i64| -> i32 { 0 }).unwrap();
        linker.func_wrap("host", "str_cmp", |_: Caller<()>, _: i32, _: i32, _: i32| -> i32 { 0 }).unwrap();
        linker.func_wrap("host", "f64_bits", |_: Caller<()>, x: f64| -> i64 {
            x.to_bits() as i64
        }).unwrap();
        let b = bump.clone();
        linker.func_wrap("host", "str_slice", move |mut caller: Caller<()>, ptr: i32, start: i64, end: i64| -> i32 {
            let mem = caller.get_export("memory").and_then(|e| e.into_memory()).unwrap();
            let src_bytes: Vec<u8> = {
                let data = mem.data(&caller);
                let base = ptr as usize;
                let mut src_len: i64 = 0;
                while base + (src_len as usize) < data.len() && data[base + (src_len as usize)] != 0 {
                    src_len += 1;
                }
                let s = if start < 0 { 0 } else if start > src_len { src_len } else { start };
                let e = if end < 0 { 0 } else if end > src_len { src_len } else { end };
                if s >= e {
                    Vec::new()
                } else {
                    let src_start = base + s as usize;
                    let slice_len = (e - s) as usize;
                    data[src_start..src_start + slice_len].to_vec()
                }
            };
            if src_bytes.is_empty() {
                let new_ptr = b.fetch_add(1, Ordering::SeqCst);
                let needed = new_ptr as usize + 1;
                let mut current_len = mem.data(&caller).len();
                while needed > current_len {
                    let pages = ((needed - current_len + 65535) / 65536) as u32;
                    mem.grow(&mut caller, pages).ok();
                    let new_len = mem.data(&caller).len();
                    if new_len == current_len { break; }
                    current_len = new_len;
                }
                let mem = caller.get_export("memory").and_then(|e| e.into_memory()).unwrap();
                mem.data_mut(&mut caller)[new_ptr as usize] = 0;
                return new_ptr as i32;
            }
            let slice_len = src_bytes.len();
            let new_ptr = b.fetch_add((slice_len + 1) as u32, Ordering::SeqCst);
            let needed = new_ptr as usize + slice_len + 1;
            let mut current_len = mem.data(&caller).len();
            while needed > current_len {
                let pages = ((needed - current_len + 65535) / 65536) as u32;
                mem.grow(&mut caller, pages).ok();
                let new_len = mem.data(&caller).len();
                if new_len == current_len { break; }
                current_len = new_len;
            }
            let mem = caller.get_export("memory").and_then(|e| e.into_memory()).unwrap();
            for i in 0..slice_len {
                mem.data_mut(&mut caller)[new_ptr as usize + i] = src_bytes[i];
            }
            mem.data_mut(&mut caller)[new_ptr as usize + slice_len] = 0;
            new_ptr as i32
        }).unwrap();
        linker.func_wrap("host", "tensor_from_vec", |_: Caller<()>, _data_ptr: i32, len: i32, _rank: i32| -> i64 {
            len as i64
        }).unwrap();

        // ── `env` module (tenthc wasm.th signatures) ──
        linker.func_wrap("env", "println", |_: Caller<()>, _: i64| {}).unwrap();
        linker.func_wrap("env", "vec_new", |_: Caller<()>| -> i64 { 0 }).unwrap();
        linker.func_wrap("env", "vec_len", |_: Caller<()>, _: i64| -> i64 { 0 }).unwrap();
        linker.func_wrap("env", "read_file", |_: Caller<()>, _: i64| -> i64 { 0 }).unwrap();
        linker.func_wrap("env", "vec_push", |_: Caller<()>, _: i64, _: i64| {}).unwrap();
        linker.func_wrap("env", "vec_get", |_: Caller<()>, _: i64, _: i64| -> i64 { 0 }).unwrap();
        linker.func_wrap("env", "write_bytes", |_: Caller<()>, _: i64, _: i64| -> i64 { 0 }).unwrap();
        linker.func_wrap("env", "str_add", |_: Caller<()>, _: i32, _: i32| -> i32 { 0 }).unwrap();
        linker.func_wrap("env", "str_eq", |_: Caller<()>, _: i32, _: i32| -> i32 { 0 }).unwrap();
        linker.func_wrap("env", "compile_host", |_: Caller<()>, _: i32, _: i32| -> i32 { 0 }).unwrap();
        linker.func_wrap("env", "str_int", |_: Caller<()>, _: i64| -> i32 { 0 }).unwrap();
        let b = bump.clone();
        linker.func_wrap("env", "tenth_alloc", move |mut caller: Caller<()>, size: i32| -> i32 {
            let ptr = b.fetch_add(size as u32, Ordering::SeqCst);
            let needed = ptr as usize + size as usize;
            let mem = caller.get_export("memory").and_then(|e| e.into_memory()).unwrap();
            let mut current_len = mem.data(&caller).len();
            while needed > current_len {
                let pages = ((needed - current_len + 65535) / 65536) as u32;
                mem.grow(&mut caller, pages).ok();
                let new_len = mem.data(&caller).len();
                if new_len == current_len { break; }
                current_len = new_len;
            }
            ptr as i32
        }).unwrap();
        linker.func_wrap("env", "str_len", |caller: Caller<()>, ptr: i32| -> i32 {
            let mem = caller.get_export("memory").and_then(|e| e.into_memory()).unwrap();
            let data = mem.data(&caller);
            let base = ptr as usize;
            let mut len: i32 = 0;
            while base + (len as usize) < data.len() && data[base + (len as usize)] != 0 {
                len += 1;
            }
            len
        }).unwrap();
        linker.func_wrap("env", "str_at", |_: Caller<()>, _: i32, _: i64| -> i32 { 0 }).unwrap();
        linker.func_wrap("env", "str_cmp", |_: Caller<()>, _: i32, _: i32, _: i32| -> i32 { 0 }).unwrap();
        linker.func_wrap("env", "f64_bits", |_: Caller<()>, x: f64| -> i64 {
            x.to_bits() as i64
        }).unwrap();
        let b = bump.clone();
        linker.func_wrap("env", "str_slice", move |mut caller: Caller<()>, ptr: i32, start: i64, end: i64| -> i32 {
            let mem = caller.get_export("memory").and_then(|e| e.into_memory()).unwrap();
            let src_bytes: Vec<u8> = {
                let data = mem.data(&caller);
                let base = ptr as usize;
                let mut src_len: i64 = 0;
                while base + (src_len as usize) < data.len() && data[base + (src_len as usize)] != 0 {
                    src_len += 1;
                }
                let s = if start < 0 { 0 } else if start > src_len { src_len } else { start };
                let e = if end < 0 { 0 } else if end > src_len { src_len } else { end };
                if s >= e {
                    Vec::new()
                } else {
                    let src_start = base + s as usize;
                    let slice_len = (e - s) as usize;
                    data[src_start..src_start + slice_len].to_vec()
                }
            };
            if src_bytes.is_empty() {
                let new_ptr = b.fetch_add(1, Ordering::SeqCst);
                let needed = new_ptr as usize + 1;
                let mut current_len = mem.data(&caller).len();
                while needed > current_len {
                    let pages = ((needed - current_len + 65535) / 65536) as u32;
                    mem.grow(&mut caller, pages).ok();
                    let new_len = mem.data(&caller).len();
                    if new_len == current_len { break; }
                    current_len = new_len;
                }
                let mem = caller.get_export("memory").and_then(|e| e.into_memory()).unwrap();
                mem.data_mut(&mut caller)[new_ptr as usize] = 0;
                return new_ptr as i32;
            }
            let slice_len = src_bytes.len();
            let new_ptr = b.fetch_add((slice_len + 1) as u32, Ordering::SeqCst);
            let needed = new_ptr as usize + slice_len + 1;
            let mut current_len = mem.data(&caller).len();
            while needed > current_len {
                let pages = ((needed - current_len + 65535) / 65536) as u32;
                mem.grow(&mut caller, pages).ok();
                let new_len = mem.data(&caller).len();
                if new_len == current_len { break; }
                current_len = new_len;
            }
            let mem = caller.get_export("memory").and_then(|e| e.into_memory()).unwrap();
            for i in 0..slice_len {
                mem.data_mut(&mut caller)[new_ptr as usize + i] = src_bytes[i];
            }
            mem.data_mut(&mut caller)[new_ptr as usize + slice_len] = 0;
            new_ptr as i32
        }).unwrap();
        linker.func_wrap("env", "tensor_from_vec", |_: Caller<()>, _data_ptr: i32, len: i32, _rank: i32| -> i64 {
            len as i64
        }).unwrap();

        (store, linker)
    }

    /// Run a WASM module, call `fn_name(args)` and return the i64 result.
    fn run_wasm_i64(wasm: &[u8], fn_name: &str, args: &[i64]) -> i64 {
        assert_eq!(&wasm[..4], b"\0asm", "WASM must have valid magic");
        let engine = Engine::new(&selfhost_config());
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

    /// Assert that tenthc-compiled WASM for `src` returns `expected` when calling
    /// `fn_name(args)`.
    fn assert_tenthc_for(src: &str, fn_name: &str, args: &[i64], expected: i64) {
        let wasm_tenthc = compile_via_tenthc(src);
        let result = run_wasm_i64(&wasm_tenthc, fn_name, args);
        assert_eq!(
            result, expected,
            "tenthc `..=` path returned wrong value for {}\n  src: {}",
            fn_name, src
        );
    }

    // ── Test 1: tenthc lexes `..=` without parser error ──────────────────
    //
    // Regression: before the fix, `..=` was mis-tokenized as `..` + `=`,
    // causing `for i in 1..=3 { ... }` to fail parsing in tenthc. This
    // test asserts that tenthc can compile the source without panicking
    // (compile_via_tenthc uses .expect() at every stage, so any lexer or
    // parser error will surface as a panic).
    //
    // Guards: tenthc/lexer/lexer.th:180-190 (DotDotEq tokenization),
    //         tenthc/parser/parser.th:1100-1132 (for-in parse with Range).

    #[test]
    fn tenthc_dotdot_eq_lexes_correctly() {
        let src = "fn f() -> i64 { let mut s: i64 = 0; for i in 1..=3 { s = s + i; }; s }";
        let wasm = compile_via_tenthc(src);
        assert_eq!(&wasm[..4], b"\0asm", "must produce valid WASM");
        assert!(wasm.len() > 8, "WASM-B must be non-trivial");
    }

    // ── Test 2: inclusive range iteration includes upper bound ───────────
    //
    // `for i in 1..=3` should iterate 1, 2, 3 (inclusive of upper bound).
    // Expected sum: 1 + 2 + 3 = 6.
    //
    // If `..=` were mis-tokenized as `..` (exclusive), the parser would
    // either reject the source (test 1 catches that) or treat it as
    // `1..3` exclusive — which would sum to 1 + 2 = 3, failing this test.
    //
    // Guards: tenthc/lexer/lexer.th:180-190 (DotDotEq),
    //         tenthc/compile/wasm.th:1442 (range_inclusive → i64.le_s).

    #[test]
    fn tenthc_dotdot_eq_inclusive_range_iteration() {
        let src = "fn f() -> i64 { let mut s: i64 = 0; for i in 1..=3 { s = s + i; }; s }";
        assert_tenthc_for(src, "f", &[], 6);
    }

    // ── Test 3: regression — `..=` not split into `..` + `=` ─────────────
    //
    // `for i in 2..=4` should iterate 2, 3, 4 (sum = 9).
    //
    // This is a stronger regression check than test 2: if `..=` were split
    // into `..` + `=`, the parser would see `2 .. = 4` and reject it as a
    // syntax error (you cannot assign to a range expression). If `..=`
    // were silently treated as `..` exclusive, sum would be 2 + 3 = 5.
    //
    // Guards: same as test 2 but with different boundaries to ensure the
    // fix is not specific to `1..=3`.

    #[test]
    fn tenthc_dotdot_eq_not_split_into_dotdot_assign() {
        let src = "fn f() -> i64 { let mut s: i64 = 0; for i in 2..=4 { s = s + i; }; s }";
        assert_tenthc_for(src, "f", &[], 9);
    }
}
