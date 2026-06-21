//! Three-stage self-hosting verification
//!
//! Stage 1: Rust mother compiler compiles tenthc source → WASM-A
//! Stage 2: wasmi executes WASM-A, which compiles `fn add(a,b){a+b}` → WASM-B
//! Stage 3: Verify WASM-B executes add(3,4) = 7
//!
//! This test is slow (~5 min) because wasmi interprets the full tenthc compiler.
//! It is kept in the default test suite to catch regressions in the self-hosting
//! pipeline. To skip it during local iteration, use `cargo test -- --skip three_stage`.
#[cfg(test)]
mod three_stage {
    use wasmi::{Config, Engine, Module, Store, Linker, Caller, StackLimits};
    use tenth::compile::wasm::register_host_functions;
    use std::time::Instant;

    /// Build a wasmi Config with enlarged stack limits for the tenthc compiler's
    /// recursive descent parser and deep call chains.
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

    /// Read a Vec<i64> from WASM memory given its pointer.
    /// Vec layout: cap(8) + len(8) + data_ptr(4) + data...
    fn read_vec_from_memory(store: &Store<u32>, mem: &wasmi::Memory, vec_ptr: i64) -> Vec<u8> {
        let data = mem.data(store);
        let vp = vec_ptr as i32 as usize;
        if vp + 20 > data.len() {
            return Vec::new();
        }
        let len = i64::from_le_bytes(data[vp+8..vp+16].try_into().unwrap());
        let dp = i32::from_le_bytes(data[vp+16..vp+20].try_into().unwrap()) as usize;
        let mut bytes = Vec::with_capacity(len as usize);
        for i in 0..len as usize {
            let pos = dp + i * 8;
            if pos + 8 > data.len() { break; }
            let val = i64::from_le_bytes(data[pos..pos+8].try_into().unwrap());
            bytes.push(val as u8);
        }
        bytes
    }

    fn run_test() {
        // C4: test struct literal — check p.x + p.y field access
        let test_src = "struct P { x: i64, y: i64 } fn add(a:i64,b:i64)->i64{let p=P{x:a,y:b,..};p.x+p.y}";
        println!("=== Stage 1: Rust compile_to_wasm ===");
        let t0 = Instant::now();
        let wasm_a = compile_selfhost_to_wasm(test_src);
        println!("WASM-A: {} bytes (compiled in {:?})", wasm_a.len(), t0.elapsed());
        assert_eq!(&wasm_a[..4], b"\0asm");

        println!("=== Stage 2: wasmi executes compiler ===");
        let t1 = Instant::now();
        let config = selfhost_config();
        let engine = Engine::new(&config);
        let module = Module::new(&engine, &wasm_a).expect("compile");
        // Store state is the bump-allocator offset (same as run_wasm_module).
        let mut store = Store::new(&engine, 8192u32);
        let mut linker = Linker::new(&engine);
        register_host_functions(&mut linker).expect("register host functions");

        let inst = linker.instantiate(&mut store, &module).expect("inst").start(&mut store).expect("start");
        let main_fn = inst.get_func(&store, "main").expect("main");
        // main returns Vec<i64> which compile_main wraps to i32 (pointer)
        let mut r = [wasmi::Val::I32(0)];
        main_fn.call(&mut store, &[], &mut r).expect("call main");
        let vec_ptr = match r[0] {
            wasmi::Val::I32(v) => v as i64,
            wasmi::Val::I64(v) => v,
            _ => panic!("expected i32/i64 return from main, got {:?}", r[0]),
        };
        let mem = inst.get_memory(&store, "memory").expect("memory");
        let wasm_b = read_vec_from_memory(&store, &mem, vec_ptr);
        println!("WASM-B: {} bytes (stage 2 took {:?})", wasm_b.len(), t1.elapsed());

        println!("=== Stage 3: Verify output WASM ===");
        assert!(!wasm_b.is_empty() && &wasm_b[..4] == b"\0asm", "WASM-B must be non-empty and have valid magic");
        let e2 = Engine::default();
        let m2 = Module::new(&e2, &wasm_b).expect("compile wasm-b");
        let mut s2 = Store::new(&e2, ());
        let mut l2 = Linker::new(&e2);
        // tenthc wasm.th uses module "env" with 15 imports
        l2.func_wrap("env", "println", |_: Caller<()>, _: i64| {}).unwrap();
        l2.func_wrap("env", "vec_new", |_: Caller<()>| -> i64 { 0 }).unwrap();
        l2.func_wrap("env", "vec_len", |_: Caller<()>, _: i64| -> i64 { 0 }).unwrap();
        l2.func_wrap("env", "vec_push", |_: Caller<()>, _: i64, _: i64| {}).unwrap();
        l2.func_wrap("env", "vec_get", |_: Caller<()>, _: i64, _: i64| -> i64 { 0 }).unwrap();
        l2.func_wrap("env", "read_file", |_: Caller<()>, _: i64| -> i64 { 0 }).unwrap();
        l2.func_wrap("env", "write_bytes", |_: Caller<()>, _: i64, _: i64| -> i64 { 0 }).unwrap();
        l2.func_wrap("env", "str_add", |_: Caller<()>, _: i32, _: i32| -> i32 { 0 }).unwrap();
        l2.func_wrap("env", "str_eq", |_: Caller<()>, _: i32, _: i32| -> i32 { 0 }).unwrap();
        l2.func_wrap("env", "str_int", |_: Caller<()>, _: i64| -> i32 { 0 }).unwrap();
        l2.func_wrap("env", "tenth_alloc", |_: Caller<()>, _: i32| -> i32 { 0 }).unwrap();
        l2.func_wrap("env", "compile_host", |_: Caller<()>, _: i32, _: i32| -> i32 { 0 }).unwrap();
        l2.func_wrap("env", "str_len", |_: Caller<()>, _: i32| -> i32 { 0 }).unwrap();
        l2.func_wrap("env", "str_at", |_: Caller<()>, _: i32, _: i64| -> i32 { 0 }).unwrap();
        l2.func_wrap("env", "str_cmp", |_: Caller<()>, _: i32, _: i32, _: i32| -> i32 { 0 }).unwrap();
        let i2 = l2.instantiate(&mut s2, &m2).expect("inst").start(&mut s2).expect("start");
        let add = i2.get_func(&s2, "add").expect("add");
        let mut r2 = [wasmi::Val::I64(0)];
        add.call(&mut s2, &[wasmi::Val::I64(3), wasmi::Val::I64(4)], &mut r2).expect("call");
        assert_eq!(match r2[0] { wasmi::Val::I64(v) => v, _ => panic!() }, 7);
        println!("=== VERIFIED: add(3,4) = 7 ===");
    }

    #[test]
    fn three_stage_selfhost() {
        // 128MB stack to accommodate wasmi's interpreter overhead when running
        // the full tenthc compiler (lexer/parser/lowerer/codegen) in WASM.
        std::thread::Builder::new()
            .stack_size(128 * 1024 * 1024)
            .spawn(run_test)
            .unwrap()
            .join()
            .unwrap();
    }
}
