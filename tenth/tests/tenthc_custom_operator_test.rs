//! M3.1：tenthc 自举侧自定义运算符全链路测试。
//!
//! 验证 tenthc 自举编译器对 `operator <op> = fn(...)` 声明与 `a <op> b`
//! 表达式的解析、lowering、WASM codegen 三层处理，与 Rust 母编译器侧
//! （lexer.rs CustomOperator / decl.rs operator 分支 / lower_expr.rs
//! CustomBinary）语义对齐。
//!
//! 接入点：
//!   * token.th   — Operator（disc 91）+ CustomOp(str)（disc 92）变体
//!   * lexer.th   — `operator` 关键字 + `@`/`$`/`~` 连续组合 run 读取
//!   * parser.th  — parse_operator_def（合成名 `__custom_op_<op>` 注册 FnDef）
//!                  + parse_binary 的 CustomOp 中缀处理（默认优先级 4）
//!   * lower.th   — custom binary → 对 `__custom_op_<op>` 的普通调用（disc 7）
//!   * wasm.th    — 绑定函数按普通函数编译，降级后的调用按普通调用编译
//!
//! 测试模式：与 tenthc_yield_test 一致——Stage 1 用 Rust 母编译器编译
//! tenthc 源码 + driver → WASM-A，Stage 2 用 wasmi 运行 WASM-A 产出
//! WASM-B（tenthc 编译的测试源码），Stage 3 用 wasmi 运行 WASM-B 验证返回值。

#[cfg(test)]
mod tenthc_custom_operator {
    use wasmi::{Config, Engine, Linker, Module, StackLimits, Store, Caller};
    use tenth::compile::wasm::register_host_functions;

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
        config
    }

    /// Compile a Tenth source string by running tenthc under wasmi → WASM bytes.
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
    fn setup_output_store_and_linker(engine: &Engine) -> (Store<()>, Linker<()>) {
        use std::sync::Arc;
        use std::sync::atomic::{AtomicU32, Ordering};
        let store = Store::new(engine, ());
        let mut linker = Linker::new(engine);
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
        let b = bump.clone();
        linker.func_wrap("host", "Vec_new", move |mut caller: Caller<()>| -> i64 {
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
            data[p..p+8].copy_from_slice(&0i64.to_le_bytes());
            data[p+8..p+16].copy_from_slice(&0i64.to_le_bytes());
            data[p+16..p+20].copy_from_slice(&0i32.to_le_bytes());
            ptr as i64
        }).unwrap();
        let b = bump.clone();
        linker.func_wrap("host", "Vec_push", move |mut caller: Caller<()>, vec: i64, item: i64| -> i64 {
            let vec_ptr = vec as i32 as usize;
            let (cap, len, dp) = {
                let mem = caller.get_export("memory").and_then(|e| e.into_memory()).unwrap();
                let data = mem.data(&caller);
                let vp = vec_ptr;
                let cap = if vp+8 <= data.len() { i64::from_le_bytes(data[vp..vp+8].try_into().unwrap()) } else { 0 };
                let len = if vp+16 <= data.len() { i64::from_le_bytes(data[vp+8..vp+16].try_into().unwrap()) } else { 0 };
                let dp = if vp+20 <= data.len() { i32::from_le_bytes(data[vp+16..vp+20].try_into().unwrap()) } else { 0 };
                (cap, len, dp)
            };
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
            vec
        }).unwrap();
        linker.func_wrap("host", "Vec_len", |caller: Caller<()>, vec: i64| -> i64 {
            let vec_ptr = vec as i32 as usize;
            let mem = caller.get_export("memory").and_then(|e| e.into_memory()).unwrap();
            let data = mem.data(&caller);
            if vec_ptr + 16 <= data.len() {
                i64::from_le_bytes(data[vec_ptr+8..vec_ptr+16].try_into().unwrap())
            } else { 0 }
        }).unwrap();
        linker.func_wrap("host", "Vec_get", |caller: Caller<()>, vec: i64, idx: i64| -> i64 {
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
        linker.func_wrap("host", "host_make_tensor_f16", |_: Caller<()>, _data_ptr: i32, len: i32, _rank: i32| -> i64 {
            len as i64
        }).unwrap();
        linker.func_wrap("host", "host_make_tensor_bf16", |_: Caller<()>, _data_ptr: i32, len: i32, _rank: i32| -> i64 {
            len as i64
        }).unwrap();
        // M1-S1（P4）：标量 math host stub
        linker.func_wrap("host", "host_sin", |_: Caller<()>, _: f64| -> f64 { 0.0 }).unwrap();
        linker.func_wrap("host", "host_cos", |_: Caller<()>, _: f64| -> f64 { 0.0 }).unwrap();
        linker.func_wrap("host", "host_ln", |_: Caller<()>, _: f64| -> f64 { 0.0 }).unwrap();
        linker.func_wrap("host", "host_pow", |_: Caller<()>, _: f64, _: f64| -> f64 { 0.0 }).unwrap();

        // ── `env` module (tenthc wasm.th signatures) ──
        linker.func_wrap("env", "println", |_: Caller<()>, _: i64| {}).unwrap();
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
            data[p..p+8].copy_from_slice(&0i64.to_le_bytes());
            data[p+8..p+16].copy_from_slice(&0i64.to_le_bytes());
            data[p+16..p+20].copy_from_slice(&0i32.to_le_bytes());
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
            let (cap, len, dp) = {
                let mem = caller.get_export("memory").and_then(|e| e.into_memory()).unwrap();
                let data = mem.data(&caller);
                let vp = vec_ptr;
                let cap = if vp+8 <= data.len() { i64::from_le_bytes(data[vp..vp+8].try_into().unwrap()) } else { 0 };
                let len = if vp+16 <= data.len() { i64::from_le_bytes(data[vp+8..vp+16].try_into().unwrap()) } else { 0 };
                let dp = if vp+20 <= data.len() { i32::from_le_bytes(data[vp+16..vp+20].try_into().unwrap()) } else { 0 };
                (cap, len, dp)
            };
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

    /// tenthc 编译的 WASM 中调用 `fn_name(args)`，断言返回 `expected`。
    fn assert_tenthc_custom_op(src: &str, fn_name: &str, args: &[i64], expected: i64) {
        let wasm_tenthc = compile_via_tenthc(src);
        let result = run_wasm_i64(&wasm_tenthc, fn_name, args);
        assert_eq!(
            result, expected,
            "tenthc custom-operator path returned wrong value for {}\n  src: {}",
            fn_name, src
        );
    }

    // ── Test 1: 声明 + 使用（`operator @@ = fn(...)`）────────────────────
    //
    // `operator @@ = fn(a: i64, b: i64) -> i64 { a + b }; fn f() -> i64 { 1 @@ 2 }`
    // 验证 token.th（Operator/CustomOp 变体）、lexer.th（`@` run 读取）、
    // parser.th（parse_operator_def + parse_binary CustomOp 中缀）、
    // lower.th（custom binary → 调用 `__custom_op_@@`）、wasm.th 全链路。
    #[test]
    fn tenthc_custom_op_declare_and_use() {
        let src = "operator @@ = fn(a: i64, b: i64) -> i64 { a + b }; fn f() -> i64 { 1 @@ 2 }";
        assert_tenthc_custom_op(src, "f", &[], 3);
    }

    // ── Test 2: 多个自定义运算符 + 与内置运算混合 ────────────────────────
    //
    // `@@`=+（优先级 4）、`@~`=*（优先级 4，左结合）。
    // `(10 @@ 4) @~ 3 = 14 * 3 = 42`。
    #[test]
    fn tenthc_custom_op_multiple_and_mixed() {
        let src = "operator @@ = fn(a: i64, b: i64) -> i64 { a + b } operator @~ = fn(a: i64, b: i64) -> i64 { a * b } fn g() -> i64 { (10 @@ 4) @~ 3 }";
        assert_tenthc_custom_op(src, "g", &[], 42);
    }

    // ── Test 3: 自定义运算符内部使用内置运算符 + 带参数函数 ──────────────
    #[test]
    fn tenthc_custom_op_with_args() {
        let src = "operator @~ = fn(a: i64, b: i64) -> i64 { a * 10 + b } fn h(x: i64, y: i64) -> i64 { x @~ y }";
        // (2)*10 + 3 = 23
        assert_tenthc_custom_op(src, "h", &[2, 3], 23);
    }
}
