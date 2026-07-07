//! Explicit regression tests for tenthc for-loop parsing, lowering, and WASM
//! codegen across the full self-hosting pipeline.
//!
//! Background: AUDIT #9 once registered "tenthc 无 for 循环解析" as an open
//! defect. That entry is now stale — for-in parsing, lowering, and WASM codegen
//! are implemented across tenthc:
//!   * parser.th:1100-1132  — `for name in iterable { body }` parsing
//!   * lower.th:1258-1279    — HIR lowering (StmtKind::For = disc 4)
//!   * wasm.th:1404-1474     — WASM codegen for Range iterables (disc 28),
//!                              including `range_inclusive` (..=) handling,
//!                              break (disc 5) and continue (disc 6).
//!
//! Previously, for-loop coverage in tenthc was implicit (the self-hosting
//! source itself uses `for` loops, so a regression would only surface as a
//! self-hosting failure). These tests make the coverage explicit and granular:
//! each test isolates one for-loop semantic and asserts the tenthc-compiled
//! WASM produces the expected value when run under wasmi.
//!
//! Coverage:
//!   1. Range forward iteration        `for i in 0..5`         (tenthc path)
//!   2. Range inclusive upper bound    `for i in 2..=4`        (Rust path —
//!      tenthc lexer.th:180-184 mis-tokenizes `..=` as `..` + `=`; see test
//!      comment for details; semantically guarded via Rust mother compiler)
//!   3. Vec literal iteration          `for x in [10,20,30]`   (Rust path —
//!      tenthc wasm.th:1472 notes `// Non-range iterable: no-op for now`)
//!   4. Nested for loops               3×3 matrix element sum  (tenthc path)
//!   5. break statement                early exit              (tenthc path)
//!   6. continue statement             skip iteration          (tenthc path)
//!
//! The tenthc path tests mirror the parity_test.rs harness: Stage 1 compiles
//! the tenthc source + a small driver via the Rust mother compiler → WASM-A;
//! Stage 2 runs WASM-A under wasmi, which produces WASM-B (the tenthc-compiled
//! output for the test source); Stage 3 runs WASM-B under wasmi and asserts
//! the returned i64.

#[cfg(test)]
mod tenthc_for_loop {
    use wasmi::{Config, Engine, Linker, Module, StackLimits, Store, Caller};
    use tenth::compile::wasm::register_host_functions;

    /// Build a wasmi Config with enlarged stack limits for recursive functions
    /// and the tenthc compiler's recursive descent parser.
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

        // AUDIT-11.4.5: Real Vec host function stubs (previously no-ops that
        // always returned 0, which made vec_len-based loop conditions false
        // and prevented Vec iteration tests from running).
        //
        // Vec memory layout (matches `host` module's Vec_new/Vec_push in
        // wasm/host.rs): 24-byte header = cap(8) + len(8) + dp(4), followed
        // by an 8-bytes-per-element data region allocated via the bump
        // allocator. vec_push grows the data region (doubling capacity) when
        // len >= cap or dp == 0.
        //
        // Signature differences from `host` module:
        //   * tenthc wasm.th Type 3: vec_push (i64, i64) -> ()  [no return]
        //   * host.rs Vec_push:      (i64, i64) -> i64          [returns vec]
        let b = bump.clone();
        linker.func_wrap("env", "vec_new", move |mut caller: Caller<()>| -> i64 {
            let ptr = b.fetch_add(24, Ordering::SeqCst);
            let needed = ptr as usize + 24;
            let mem = caller.get_export("memory").and_then(|e| e.into_memory()).unwrap();
            let mut current_len = mem.data(&caller).len();
            while needed > current_len {
                let pages = ((needed - current_len + 65535) / 65536) as u32;
                mem.grow(&mut caller, pages).ok();
                let new_len = mem.data(&caller).len();
                if new_len == current_len { break; }
                current_len = new_len;
            }
            let data = mem.data_mut(&mut caller);
            let p = ptr as usize;
            data[p..p+8].copy_from_slice(&0i64.to_le_bytes());       // cap
            data[p+8..p+16].copy_from_slice(&0i64.to_le_bytes());    // len
            data[p+16..p+20].copy_from_slice(&0i32.to_le_bytes());   // dp
            ptr as i64
        }).unwrap();

        linker.func_wrap("env", "vec_len", |caller: Caller<()>, vec: i64| -> i64 {
            let vec_ptr = vec as i32 as usize;
            let mem = caller.get_export("memory").and_then(|e| e.into_memory()).unwrap();
            let data = mem.data(&caller);
            if vec_ptr + 16 <= data.len() {
                i64::from_le_bytes(data[vec_ptr+8..vec_ptr+16].try_into().unwrap())
            } else { 0 }
        }).unwrap();

        linker.func_wrap("env", "read_file", |_: Caller<()>, _: i64| -> i64 { 0 }).unwrap();

        let b = bump.clone();
        linker.func_wrap("env", "vec_push", move |mut caller: Caller<()>, vec: i64, item: i64| {
            let vec_ptr = vec as i32 as usize;
            // Phase 1: read header (cap, len, dp)
            let (cap, len, dp) = {
                let mem = caller.get_export("memory").and_then(|e| e.into_memory()).unwrap();
                let data = mem.data(&caller);
                let vp = vec_ptr;
                let cap = if vp+8 <= data.len() { i64::from_le_bytes(data[vp..vp+8].try_into().unwrap()) } else { 0 };
                let len = if vp+16 <= data.len() { i64::from_le_bytes(data[vp+8..vp+16].try_into().unwrap()) } else { 0 };
                let dp = if vp+20 <= data.len() { i32::from_le_bytes(data[vp+16..vp+20].try_into().unwrap()) } else { 0 };
                (cap, len, dp)
            };
            // Phase 2: allocate/grow data region if needed (double capacity)
            let (new_cap, new_dp) = if len >= cap || dp == 0 {
                let nc = if cap == 0 { 4 } else { cap * 2 };
                let new_sz = nc as usize * 8;
                let np = b.fetch_add(new_sz as u32, Ordering::SeqCst);
                let needed = np as usize + new_sz;
                let mem = caller.get_export("memory").and_then(|e| e.into_memory()).unwrap();
                let mut current_len = mem.data(&caller).len();
                while needed > current_len {
                    let pages = ((needed - current_len + 65535) / 65536) as u32;
                    mem.grow(&mut caller, pages).ok();
                    let new_len = mem.data(&caller).len();
                    if new_len == current_len { break; }
                    current_len = new_len;
                }
                // Copy old data from dp to new allocation (if any)
                if dp != 0 && len > 0 {
                    let mem = caller.get_export("memory").and_then(|e| e.into_memory()).unwrap();
                    let data = mem.data_mut(&mut caller);
                    let old_sz = len as usize * 8;
                    data.copy_within(dp as usize..dp as usize + old_sz, np as usize);
                }
                (nc, np as i32)
            } else {
                (cap, dp)
            };
            // Phase 3: write updated header + new element
            {
                let mem = caller.get_export("memory").and_then(|e| e.into_memory()).unwrap();
                let data = mem.data_mut(&mut caller);
                let vp = vec_ptr;
                data[vp..vp+8].copy_from_slice(&new_cap.to_le_bytes());
                data[vp+8..vp+16].copy_from_slice(&(len + 1).to_le_bytes());
                data[vp+16..vp+20].copy_from_slice(&new_dp.to_le_bytes());
                let pos = new_dp as usize + len as usize * 8;
                data[pos..pos+8].copy_from_slice(&item.to_le_bytes());
            }
        }).unwrap();

        linker.func_wrap("env", "vec_get", |caller: Caller<()>, vec: i64, idx: i64| -> i64 {
            let vec_ptr = vec as i32 as usize;
            let mem = caller.get_export("memory").and_then(|e| e.into_memory()).unwrap();
            let data = mem.data(&caller);
            if vec_ptr + 20 > data.len() { return 0; }
            let dp = i32::from_le_bytes(data[vec_ptr+16..vec_ptr+20].try_into().unwrap()) as usize;
            let pos = dp + idx as usize * 8;
            if pos + 8 <= data.len() {
                i64::from_le_bytes(data[pos..pos+8].try_into().unwrap())
            } else { 0 }
        }).unwrap();
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
    /// `fn_name(args)`. This is the core tenthc for-loop self-hosting guard.
    fn assert_tenthc_for(src: &str, fn_name: &str, args: &[i64], expected: i64) {
        let wasm_tenthc = compile_via_tenthc(src);
        let result = run_wasm_i64(&wasm_tenthc, fn_name, args);
        assert_eq!(
            result, expected,
            "tenthc for-loop path returned wrong value for {}\n  src: {}",
            fn_name, src
        );
    }

    // ── Test 1: Range forward iteration ────────────────────────────────────
    //
    // `for i in 0..5` — half-open range, the most common for-loop form.
    // Expected sum: 0 + 1 + 2 + 3 + 4 = 10.
    // Guards: parser.th:1100-1132 (for-in parse), lower.th:1258-1279 (HIR),
    //         wasm.th:1404-1470 (Range disc=28 codegen, range_inclusive=false).

    #[test]
    fn tenthc_for_range_forward() {
        let src = "fn f() -> i64 { let mut s: i64 = 0; for i in 0..5 { s = s + i; }; s }";
        assert_tenthc_for(src, "f", &[], 10);
    }

    // ── Test 2: Range inclusive upper bound (..=) ─────────────────────────
    //
    // `for i in 2..=4` — inclusive range. Expected sum: 2 + 3 + 4 = 9.
    //
    // NOTE: This test uses the Rust mother compiler path (interpreter) because
    // tenthc's lexer has a pre-existing bug that prevents `..=` from being
    // tokenized correctly. In `tenthc/lexer/lexer.th:180-184`, the `.` branch
    // checks `next == "."` first and returns `DotDot` immediately without
    // inspecting the third character — so `..=` is mis-tokenized as `DotDot`
    // followed by `Assign`, and the parser sees `2 .. = 4` instead of a single
    // inclusive range. The Rust mother compiler's lexer correctly handles
    // `..=` (3-char lookahead), so this test guards the for-inclusive *semantics*
    // on the Rust side. The tenthc `..=` lexer bug is reported as a遗留问题
    // and should be fixed in a follow-up (the fix is a 2-line change in
    // lexer.th: after matching `..`, peek one more char and check for `=`).
    //
    // When the tenthc lexer bug is fixed, this test should be promoted to a
    // tenthc-path test (swap `rust_for_range_inclusive` → `tenthc_for_range_inclusive`
    // and use `assert_tenthc_for`).

    #[test]
    fn rust_for_range_inclusive() {
        use tenth::lexer::lexer::Lexer;
        use tenth::parser::parser::Parser;
        use tenth::hir::lower::Lowerer;
        use tenth::runtime::interpreter::Interpreter;
        use tenth::runtime::value::Value;

        let src = r#"
            fn main() -> i32 {
                let sum = 0
                for i in 2..=4 {
                    sum = sum + i
                }
                sum
            }
            main()
        "#;
        let mut lexer = Lexer::new(src);
        let tokens = lexer.tokenize().expect("lex");
        let mut parser = Parser::new(tokens);
        let program = parser.parse_program().expect("parse");
        let mut lowerer = Lowerer::new();
        let hir = lowerer.lower_program(&program).expect("lower");
        let mut interpreter = Interpreter::new(&hir);
        let result = interpreter.execute_program(&hir).expect("execute");
        match result {
            Some(Value::Int(9)) => {},
            v => panic!("expected Int(9), got {:?}", v),
        }
    }

    // ── Test 3: Vec literal iteration (Rust path only) ────────────────────
    //
    // `for x in [10, 20, 30]` — Vec literal as iterable. Expected sum: 60.
    //
    // NOTE: tenthc wasm.th:1472 explicitly states `// Non-range iterable: no-op
    // for now` — tenthc's WASM backend does not yet support Vec iteration. This
    // test therefore covers the Rust mother compiler path (interpreter) only,
    // to ensure for-loop Vec semantics are at least guarded on the Rust side.
    // When tenthc gains Vec iteration support, this test should be promoted to
    // a tenthc-path parity test.

    #[test]
    fn rust_for_vec_iteration() {
        use tenth::lexer::lexer::Lexer;
        use tenth::parser::parser::Parser;
        use tenth::hir::lower::Lowerer;
        use tenth::runtime::interpreter::Interpreter;
        use tenth::runtime::value::Value;

        let src = r#"
            fn main() -> i32 {
                let sum = 0
                for x in [10, 20, 30] {
                    sum = sum + x
                }
                sum
            }
            main()
        "#;
        let mut lexer = Lexer::new(src);
        let tokens = lexer.tokenize().expect("lex");
        let mut parser = Parser::new(tokens);
        let program = parser.parse_program().expect("parse");
        let mut lowerer = Lowerer::new();
        let hir = lowerer.lower_program(&program).expect("lower");
        let mut interpreter = Interpreter::new(&hir);
        let result = interpreter.execute_program(&hir).expect("execute");
        match result {
            Some(Value::Int(60)) => {},
            v => panic!("expected Int(60), got {:?}", v),
        }
    }

    // ── Test 4: Nested for loops ──────────────────────────────────────────
    //
    // Double for-loop computing the sum of a 3×3 matrix's elements expressed
    // as `i * 3 + j` (which yields 0..8). Expected sum: 0+1+...+8 = 36.
    // Guards: nesting of for-loops, loop variable scoping across depths, and
    //         the body block structure in wasm.th:1450-1457.

    #[test]
    fn tenthc_for_nested() {
        let src = "fn f() -> i64 { let mut s: i64 = 0; for i in 0..3 { for j in 0..3 { s = s + i * 3 + j; }; }; s }";
        assert_tenthc_for(src, "f", &[], 36);
    }

    // ── Test 5: break statement ───────────────────────────────────────────
    //
    // `for i in 0..100 { if i == 5 { break; }; s = s + i }` — early exit.
    // Expected sum: 0 + 1 + 2 + 3 + 4 = 10 (loop exits when i reaches 5).
    // Guards: parser.th:1134-1138 (break parse), lower.th:1281 (StmtKind::Break
    //         = disc 5), wasm.th:1499-1503 (br depth = 1 + break_offset + if_depth).

    #[test]
    fn tenthc_for_break() {
        let src = "fn f() -> i64 { let mut s: i64 = 0; for i in 0..100 { if i == 5 { break; }; s = s + i; }; s }";
        assert_tenthc_for(src, "f", &[], 10);
    }

    // ── Test 6: continue statement ────────────────────────────────────────
    //
    // `for i in 0..5 { if i == 2 { continue; }; s = s + i }` — skip one iter.
    // Expected sum: 0 + 1 + 3 + 4 = 8 (skips i == 2).
    // Guards: parser.th:1140-1144 (continue parse), lower.th:1282 (StmtKind::
    //         Continue = disc 6), wasm.th:1505+ (br to body block end).

    #[test]
    fn tenthc_for_continue() {
        let src = "fn f() -> i64 { let mut s: i64 = 0; for i in 0..5 { if i == 2 { continue; }; s = s + i; }; s }";
        assert_tenthc_for(src, "f", &[], 8);
    }

    // ── Test 7: Vec literal iteration (tenthc path) ──────────────────────
    //
    // AUDIT-11.4.5 regression: `for x in [10, 20, 30] { ... }` — Vec literal
    // as iterable. Expected sum: 10 + 20 + 30 = 60.
    //
    // Background: tenthc wasm.th:1471-1546 added an ArrayLiteral (disc=22)
    // branch in the For-statement codegen. It builds a Vec via vec_new +
    // N × vec_push, then iterates by index using vec_len + vec_get.
    //
    // This is the tenthc-path promotion of `rust_for_vec_iteration` (Test 3
    // above), now that the WASM backend supports Vec iteration. The Rust
    // path test is retained as a cross-check.
    //
    // Guards: tenthc/compile/wasm.th:1471-1546 (ArrayLiteral For codegen),
    //         tenthc/parser/parser.th:1100-1132 (for-in parse),
    //         tenthc/hir/lower.th:1258-1279 (For = disc 4 lowering).
    //
    // BUG (newly discovered, reported to compiler dept): tenthc wasm.th:1491
    // calls `wasm_call(out, 3)` (env.vec_push) and then `wasm_drop(out)` to
    // discard the return value, with a comment claiming vec_push has
    // signature `(i64, i64) -> i64`. However, Type 3 in the WASM type
    // section is actually `(i64, i64) -> ()` (wasm.th:1661-1663), so
    // vec_push returns nothing and `drop` has nothing to pop — wasmi
    // rejects the module with "type mismatch: expected a type but nothing
    // on stack". This only affects the For+ArrayLiteral path (MethodCall
    // path at wasm.th:753-755 correctly omits the drop).
    //
    // UPDATE 2026-07-06: type/drop mismatch fixed (wasm_drop removed).
    // UPDATE 2026-07-08 (AUDIT-11.4.5 fix): the empty host function stubs
    // in setup_output_store_and_linker have been replaced with real Vec
    // implementations (vec_new/vec_len/vec_push/vec_get backed by WASM
    // memory + the shared bump allocator). Tests now run and pass.

    #[test]
    fn tenthc_for_vec_literal_basic() {
        let src = "fn f() -> i64 { let mut s: i64 = 0; for x in [10, 20, 30] { s = s + x; }; s }";
        assert_tenthc_for(src, "f", &[], 60);
    }

    // ── Test 8: empty Vec literal iteration (tenthc path) ───────────────
    //
    // `for x in [] { ... }` — empty Vec literal. Body should not execute.
    // Expected sum: 0 (initial value unchanged).
    //
    // Guards: tenthc/compile/wasm.th:1486 (`if iter.args_count > 0` guards
    //         the push loop), vec_len returns 0 → loop condition `i < 0`
    //         is false on first iteration → body never executes.

    #[test]
    fn tenthc_for_vec_literal_empty() {
        let src = "fn f() -> i64 { let mut s: i64 = 0; for x in [] { s = s + 1; }; s }";
        assert_tenthc_for(src, "f", &[], 0);
    }

    // ── Test 9: break/continue in Vec literal iteration (tenthc path) ───
    //
    // `for x in [1, 2, 3, 4, 5] { if x == 3 { continue; }; if x == 5 { break; }; s = s + x }`
    // — break/continue inside Vec iteration.
    //
    // Expected sum: 1 + 2 + 4 = 7 (skips 3 via continue, exits before adding 5).
    //
    // Guards: tenthc/compile/wasm.th:1506-1541 (block/loop structure with
    //         break_offset=1 matches Range branch; br depth for break and
    //         continue is computed identically to Range iteration).
    //
    // BUG: same vec_push type/drop mismatch as Test 7 — see comment there.
    //      type/drop mismatch was fixed (wasm_drop removed).
    // UPDATE 2026-07-08 (AUDIT-11.4.5 fix): host function stubs in
    //      setup_output_store_and_linker replaced with real Vec
    //      implementations — tests now run and pass.

    #[test]
    fn tenthc_for_vec_literal_break_continue() {
        let src = "fn f() -> i64 { let mut s: i64 = 0; for x in [1, 2, 3, 4, 5] { if x == 3 { continue; }; if x == 5 { break; }; s = s + x; }; s }";
        assert_tenthc_for(src, "f", &[], 7);
    }
}
