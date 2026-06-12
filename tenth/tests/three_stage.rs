//! Three-stage self-hosting verification test.
//!
//! Uses a large-stack thread to avoid Lowerer stack overflow on 1800-line source.
#[cfg(test)]
mod three_stage {
    use wasmi::{Engine, Module, Store, Linker, Caller};
    use tenth::runtime::vm::Vm;
    use tenth::runtime::value::Value;
    use tenth::compile::bytecode::BytecodeCompiler;
    use std::rc::Rc;
    use std::cell::RefCell;
    use std::collections::HashMap;
    use std::sync::Mutex;

    struct HostState {
        vecs: Mutex<HashMap<i64, Vec<i64>>>,
        next_vec_id: Mutex<i64>,
        input_source: Mutex<String>,
    }

    fn stage1_compile(test_source: &str) -> Vec<u8> {
        use tenth::lexer::lexer::Lexer;
        use tenth::parser::parser::Parser;
        use tenth::hir::lower::Lowerer;

        let selfhost_src = [
            include_str!("../../tenthc/lexer/token.th"),
            include_str!("../../tenthc/lexer/lexer.th"),
            include_str!("../../tenthc/parser/parser.th"),
            include_str!("../../tenthc/hir/hir.th"),
            include_str!("../../tenthc/hir/lower.th"),
            include_str!("../../tenthc/compile/wasm.th"),
        ].join("\n");

        let escaped = test_source.replace('\\', "\\\\").replace('"', "\\\"");
        let full_src = format!(
            "{}fn main()->Vec<i64>{{let mut lex=lexer_new(\"{}\");let tokens=lexer_tokenize(&mut lex);let program=parse_program(tokens);let hir=lower_program(program);compile_to_wasm(hir)}}",
            selfhost_src, escaped
        );

        let mut lexer = Lexer::new(&full_src);
        let tokens = lexer.tokenize().expect("lex");
        let mut parser = Parser::new(tokens);
        let program = parser.parse_program().expect("parse");
        let mut lowerer = Lowerer::new();
        let hir = lowerer.lower_program(&program).expect("lower");

        let mut vm = Vm::new();
        vm.add_native("println".into(), |_, args| {
            for a in args { print!("{a}"); } println!(); Ok(Value::Unit)
        });
        vm.add_native("read_file".into(), |_, args| {
            if let Some(Value::String(path)) = args.first() {
                std::fs::read_to_string(path)
                    .map(Value::String)
                    .map_err(|e| tenth::error::TenthError::RuntimeError { message: format!("read_file: {e}") })
            } else { Ok(Value::String(String::new())) }
        });
        vm.add_native("Vec::new".into(), |_, _| {
            Ok(Value::Vec(Rc::new(RefCell::new(Vec::new()))))
        });
        vm.add_native("write_bytes".into(), |_, args| {
            if args.len() >= 2 {
                if let Value::Vec(items) = &args[0] {
                    if let Value::String(path) = &args[1] {
                        let bytes: Vec<u8> = items.borrow().iter().map(|v| v.as_int().unwrap_or(0) as u8).collect();
                        let _ = std::fs::write(path, &bytes);
                    }
                }
            }
            Ok(Value::Int(0))
        });

        for func in &hir.functions {
            let compiler = BytecodeCompiler::new();
            if let Ok(chunk) = compiler.compile(func) {
                vm.add_fn(func.name.clone(), chunk);
            }
        }

        match vm.call("main").expect("VM main") {
            Value::Vec(items) => items.borrow().iter().map(|v| v.as_int().unwrap_or(0) as u8).collect(),
            other => panic!("Expected Vec<i64>, got {:?}", other),
        }
    }

    fn register_imports(linker: &mut Linker<HostState>) {
        linker.func_wrap("env", "println", |_: Caller<HostState>, _: i64| {}).unwrap();
        linker.func_wrap("env", "vec_new", |caller: Caller<HostState>| -> i64 {
            let s = caller.data(); let mut id = s.next_vec_id.lock().unwrap();
            let vid = *id; *id += 1; s.vecs.lock().unwrap().insert(vid, Vec::new()); vid
        }).unwrap();
        linker.func_wrap("env", "vec_len", |caller: Caller<HostState>, p: i64| -> i64 {
            caller.data().vecs.lock().unwrap().get(&p).map(|v| v.len() as i64).unwrap_or(0)
        }).unwrap();
        linker.func_wrap("env", "vec_push", |caller: Caller<HostState>, p: i64, v: i64| {
            if let Some(vec) = caller.data().vecs.lock().unwrap().get_mut(&p) { vec.push(v); }
        }).unwrap();
        linker.func_wrap("env", "vec_get", |caller: Caller<HostState>, p: i64, i: i64| -> i64 {
            caller.data().vecs.lock().unwrap().get(&p).and_then(|v| v.get(i as usize).copied()).unwrap_or(0)
        }).unwrap();
        linker.func_wrap("env", "read_file", |caller: Caller<HostState>, _: i64| -> i64 {
            let s = caller.data(); let src = s.input_source.lock().unwrap().clone();
            let mut id = s.next_vec_id.lock().unwrap(); let vid = *id; *id += 1;
            s.vecs.lock().unwrap().insert(vid, src.bytes().map(|b| b as i64).collect()); vid
        }).unwrap();
        linker.func_wrap("env", "write_bytes", |_: Caller<HostState>, _: i64, _: i64| -> i64 { 0 }).unwrap();
    }

    fn register_dummy(linker: &mut Linker<()>) {
        linker.func_wrap("env", "println", |_: Caller<()>, _: i64| {}).unwrap();
        linker.func_wrap("env", "vec_new", |_: Caller<()>| -> i64 { 0 }).unwrap();
        linker.func_wrap("env", "vec_len", |_: Caller<()>, _: i64| -> i64 { 0 }).unwrap();
        linker.func_wrap("env", "vec_push", |_: Caller<()>, _: i64, _: i64| {}).unwrap();
        linker.func_wrap("env", "vec_get", |_: Caller<()>, _: i64, _: i64| -> i64 { 0 }).unwrap();
        linker.func_wrap("env", "read_file", |_: Caller<()>, _: i64| -> i64 { 0 }).unwrap();
        linker.func_wrap("env", "write_bytes", |_: Caller<()>, _: i64, _: i64| -> i64 { 0 }).unwrap();
    }

    fn run_stages(test_src: &str) {
        println!("=== Stage 1 ===");
        let wasm_a = stage1_compile(test_src);
        println!("WASM-A: {} bytes", wasm_a.len());
        assert_eq!(&wasm_a[..4], b"\0asm");

        println!("=== Stage 2 ===");
        let engine = Engine::default();
        let module = Module::new(&engine, &wasm_a).expect("compile");
        let state = HostState {
            vecs: Mutex::new(HashMap::new()), next_vec_id: Mutex::new(1),
            input_source: Mutex::new(test_src.to_string()),
        };
        let mut store = Store::new(&engine, state);
        let mut linker = Linker::new(&engine);
        register_imports(&mut linker);
        let instance = linker.instantiate(&mut store, &module).expect("inst").start(&mut store).expect("start");

        let main_fn = instance.get_func(&store, "main").expect("main");
        let mut r = [wasmi::Val::I64(0)];
        main_fn.call(&mut store, &[], &mut r).expect("call main");
        let out_id = match r[0] { wasmi::Val::I64(v) => v, _ => panic!() };
        let wasm_b: Vec<u8> = store.data().vecs.lock().unwrap()
            .get(&out_id).map(|v| v.iter().map(|&b| b as u8).collect()).unwrap_or_default();
        println!("WASM-B: {} bytes", wasm_b.len());

        println!("=== Stage 3 ===");
        assert!(!wasm_b.is_empty() && &wasm_b[..4] == b"\0asm");
        let engine2 = Engine::default();
        let module2 = Module::new(&engine2, &wasm_b).expect("compile");
        let mut store2 = Store::new(&engine2, ());
        let mut linker2 = Linker::new(&engine2);
        register_dummy(&mut linker2);
        let instance2 = linker2.instantiate(&mut store2, &module2).expect("inst").start(&mut store2).expect("start");

        let add = instance2.get_func(&store2, "add").expect("add");
        let mut r2 = [wasmi::Val::I64(0)];
        add.call(&mut store2, &[wasmi::Val::I64(3), wasmi::Val::I64(4)], &mut r2).expect("call");
        let val = match r2[0] { wasmi::Val::I64(v) => v, _ => panic!() };
        assert_eq!(val, 7);
        println!("=== VERIFIED: add(3,4) = {} ===", val);
    }

    #[test]
    fn three_stage_selfhost() {
        let test_src = "fn add(a:i64,b:i64)->i64{a+b}";
        // Large stack to avoid Lowerer overflow on 1800-line self-hosting source
        std::thread::Builder::new()
            .stack_size(32 * 1024 * 1024)
            .spawn(move || run_stages(test_src))
            .unwrap()
            .join()
            .unwrap();
    }
}
