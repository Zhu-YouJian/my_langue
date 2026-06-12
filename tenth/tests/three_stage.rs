//! Three-stage self-hosting verification — Rust WASM compiler path
#[cfg(test)]
mod three_stage {
    use wasmi::{Engine, Module, Store, Linker, Caller};
    use std::collections::HashMap;
    use std::sync::Mutex;

    struct HostState {
        vecs: Mutex<HashMap<i64, Vec<i64>>>,
        next_vec_id: Mutex<i64>,
        input_source: Mutex<String>,
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
        let full_src = format!("{}fn main()->Vec<i64>{{let mut lex=lexer_new(\"{}\");let tokens=lexer_tokenize(&mut lex);let program=parse_program(tokens);let hir=lower_program(program);compile_to_wasm(hir)}}", selfhost_src, escaped);

        let mut lexer = Lexer::new(&full_src);
        let tokens = lexer.tokenize().expect("lex");
        let mut parser = Parser::new(tokens);
        let program = parser.parse_program().expect("parse");
        let mut lowerer = Lowerer::new();
        let hir = lowerer.lower_program(&program).expect("lower");
        compile::compile_to_wasm(&hir).expect("compile")
    }

    fn run_test() {
        let test_src = "fn add(a:i64,b:i64)->i64{a+b}";
        println!("=== Stage 1: Rust compile_to_wasm ===");
        let wasm_a = compile_selfhost_to_wasm(test_src);
        println!("WASM-A: {} bytes", wasm_a.len());
        assert_eq!(&wasm_a[..4], b"\0asm");

        println!("=== Stage 2: wasmi executes compiler ===");
        let engine = Engine::default();
        let module = Module::new(&engine, &wasm_a).expect("compile");
        let state = HostState { vecs: Mutex::new(HashMap::new()), next_vec_id: Mutex::new(1), input_source: Mutex::new(test_src.to_string()) };
        let mut store = Store::new(&engine, state);
        let mut linker = Linker::new(&engine);
        
        linker.func_wrap("env", "println", |_: Caller<HostState>, _: i64| {}).unwrap();
        linker.func_wrap("env", "vec_new", |c: Caller<HostState>| -> i64 { let s=c.data(); let mut i=s.next_vec_id.lock().unwrap(); let v=*i; *i+=1; s.vecs.lock().unwrap().insert(v,Vec::new()); v }).unwrap();
        linker.func_wrap("env", "vec_len", |c: Caller<HostState>, p: i64| -> i64 { c.data().vecs.lock().unwrap().get(&p).map(|v|v.len() as i64).unwrap_or(0) }).unwrap();
        linker.func_wrap("env", "vec_push", |c: Caller<HostState>, p: i64, v: i64| { if let Some(vv)=c.data().vecs.lock().unwrap().get_mut(&p) { vv.push(v); } }).unwrap();
        linker.func_wrap("env", "vec_get", |c: Caller<HostState>, p: i64, i: i64| -> i64 { c.data().vecs.lock().unwrap().get(&p).and_then(|v|v.get(i as usize).copied()).unwrap_or(0) }).unwrap();
        linker.func_wrap("env", "read_file", |c: Caller<HostState>, _: i64| -> i64 { let s=c.data(); let src=s.input_source.lock().unwrap().clone(); let mut i=s.next_vec_id.lock().unwrap(); let v=*i; *i+=1; s.vecs.lock().unwrap().insert(v,src.bytes().map(|b|b as i64).collect()); v }).unwrap();
        linker.func_wrap("env", "write_bytes", |_: Caller<HostState>, _: i64, _: i64| -> i64 { 0 }).unwrap();

        let inst = linker.instantiate(&mut store, &module).expect("inst").start(&mut store).expect("start");
        let main_fn = inst.get_func(&store, "main").expect("main");
        let mut r = [wasmi::Val::I64(0)];
        main_fn.call(&mut store, &[], &mut r).expect("call main");
        let out_id = match r[0] { wasmi::Val::I64(v) => v, _ => panic!() };
        let wasm_b: Vec<u8> = store.data().vecs.lock().unwrap().get(&out_id).map(|v| v.iter().map(|&b| b as u8).collect()).unwrap_or_default();
        println!("WASM-B: {} bytes", wasm_b.len());

        println!("=== Stage 3: Verify ===");
        assert!(!wasm_b.is_empty() && &wasm_b[..4] == b"\0asm");
        let e2 = Engine::default();
        let m2 = Module::new(&e2, &wasm_b).expect("compile");
        let mut s2 = Store::new(&e2, ());
        let mut l2 = Linker::new(&e2);
        l2.func_wrap("env", "println", |_: Caller<()>, _: i64| {}).unwrap();
        l2.func_wrap("env", "vec_new", |_: Caller<()>| -> i64 { 0 }).unwrap();
        l2.func_wrap("env", "vec_len", |_: Caller<()>, _: i64| -> i64 { 0 }).unwrap();
        l2.func_wrap("env", "vec_push", |_: Caller<()>, _: i64, _: i64| {}).unwrap();
        l2.func_wrap("env", "vec_get", |_: Caller<()>, _: i64, _: i64| -> i64 { 0 }).unwrap();
        l2.func_wrap("env", "read_file", |_: Caller<()>, _: i64| -> i64 { 0 }).unwrap();
        l2.func_wrap("env", "write_bytes", |_: Caller<()>, _: i64, _: i64| -> i64 { 0 }).unwrap();
        let i2 = l2.instantiate(&mut s2, &m2).expect("inst").start(&mut s2).expect("start");
        let add = i2.get_func(&s2, "add").expect("add");
        let mut r2 = [wasmi::Val::I64(0)];
        add.call(&mut s2, &[wasmi::Val::I64(3), wasmi::Val::I64(4)], &mut r2).expect("call");
        assert_eq!(match r2[0] { wasmi::Val::I64(v) => v, _ => panic!() }, 7);
        println!("=== VERIFIED: add(3,4) = 7 ===");
    }

    #[test]
    fn three_stage_selfhost() {
        // Large stack for recursive WASM compiler
        std::thread::Builder::new()
            .stack_size(64 * 1024 * 1024)
            .spawn(run_test)
            .unwrap()
            .join()
            .unwrap();
    }
}
