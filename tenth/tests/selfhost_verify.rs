//! Self-hosting integration test.
//! Validates that the self-hosting WASM compiler produces valid WASM
//! with proper import declarations, executable by wasmi with host imports.
//!
//! Prerequisite: run `tenth run tenthc/boot_full.th` to generate boot_full_out.wasm
#[cfg(test)]
mod selfhost_verify {
    #[test]
    fn wasm_module_loads_with_host_imports() {
        // Load the pre-generated WASM (from `tenth run tenthc/boot_full.th`)
        let wasm_bytes = match std::fs::read("../boot_full_out.wasm") {
            Ok(b) => b,
            Err(_) => {
                println!("SKIP: boot_full_out.wasm not found.");
                println!("Generate it with: cargo run -- run tenthc/boot_full.th");
                return;
            }
        };
        
        println!("WASM size: {} bytes", wasm_bytes.len());
        assert_eq!(&wasm_bytes[..4], b"\0asm", "Invalid WASM magic");
        
        use wasmi::{Engine, Module, Store, Linker, Caller};
        let engine = Engine::default();
        let module = Module::new(&engine, &wasm_bytes).expect("module compile");
        println!("Module compiled OK");
        
        let mut store = Store::new(&engine, ());
        let mut linker = Linker::new(&engine);
        
        // All 7 host imports that the self-hosting WASM expects
        linker.func_wrap("env", "println", |_: Caller<'_, ()>, _: i64| {}).unwrap();
        linker.func_wrap("env", "vec_new", |_: Caller<'_, ()>| -> i64 { 0 }).unwrap();
        linker.func_wrap("env", "vec_len", |_: Caller<'_, ()>, _: i64| -> i64 { 0 }).unwrap();
        linker.func_wrap("env", "vec_push", |_: Caller<'_, ()>, _: i64, _: i64| {}).unwrap();
        linker.func_wrap("env", "vec_get", |_: Caller<'_, ()>, _: i64, _: i64| -> i64 { 0 }).unwrap();
        linker.func_wrap("env", "read_file", |_: Caller<'_, ()>, _: i64| -> i64 { 0 }).unwrap();
        linker.func_wrap("env", "write_bytes", |_: Caller<'_, ()>, _: i64, _: i64| -> i64 { 0 }).unwrap();
        
        let instance = linker.instantiate(&mut store, &module)
            .expect("instantiate")
            .start(&mut store)
            .expect("start");
        println!("Instance started OK");
        
        // Call the exported function and verify
        let add_fn = instance.get_func(&store, "add").expect("get add");
        let mut results = [wasmi::Val::I64(0)];
        add_fn.call(&mut store, &[wasmi::Val::I64(3), wasmi::Val::I64(4)], &mut results)
            .expect("call add");
        
        let val = match results[0] {
            wasmi::Val::I64(v) => v,
            _ => panic!("unexpected return type"),
        };
        assert_eq!(val, 7, "add(3,4) should be 7, got {val}");
        
        println!("=== SELF-HOSTING WASM VERIFIED: add(3,4) = {} ===", val);
    }
}
