//! Phase 5.1 — WASM 后端 f32 e2e 测试
//!
//! Phase 5 消除策略 A 后：f32 真正走 F32 WASM 路径（不再塌缩为 F64）。
//! 1. f32 字面量经 F32Const 发射，F32Add/F32Sub 等真正 f32 算术
//! 2. f32 与 f64 混合算术路径不崩
//! 3. TensorLiteral 中含 f32 元素按 4 字节存储
//! 4. f32 函数签名参数/返回值为 F32（wasmi::Val::F32）

#[cfg(test)]
mod f32_wasm {
    use wasmi::{Engine, Module, Store, Linker, Caller};
    use std::collections::HashMap;
    use std::sync::Mutex;

    struct HostState {
        vecs: Mutex<HashMap<i64, Vec<i64>>>,
        next_vec_id: Mutex<i64>,
        printed: Mutex<Vec<i64>>,
    }

    fn compile_to_wasm(src: &str) -> Vec<u8> {
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

    fn setup_store_and_linker(engine: &Engine) -> (Store<HostState>, Linker<HostState>) {
        let state = HostState {
            vecs: Mutex::new(HashMap::new()),
            next_vec_id: Mutex::new(1),
            printed: Mutex::new(Vec::new()),
        };
        let store = Store::new(engine, state);
        let mut linker = Linker::new(engine);

        linker.func_wrap("host", "println", |c: Caller<HostState>, v: i32| {
            c.data().printed.lock().unwrap().push(v as i64);
        }).unwrap();
        linker.func_wrap("host", "Vec_new", |c: Caller<HostState>| -> i64 {
            let s = c.data();
            let mut id = s.next_vec_id.lock().unwrap();
            let v = *id;
            *id += 1;
            s.vecs.lock().unwrap().insert(v, Vec::new());
            v
        }).unwrap();
        linker.func_wrap("host", "Vec_len", |c: Caller<HostState>, p: i64| -> i64 {
            c.data().vecs.lock().unwrap().get(&p).map(|v| v.len() as i64).unwrap_or(0)
        }).unwrap();
        linker.func_wrap("host", "Vec_push", |c: Caller<HostState>, p: i64, v: i64| -> i64 {
            if let Some(vv) = c.data().vecs.lock().unwrap().get_mut(&p) { vv.push(v); }
            0
        }).unwrap();
        linker.func_wrap("host", "Vec_get", |c: Caller<HostState>, p: i64, i: i64| -> i64 {
            c.data().vecs.lock().unwrap().get(&p).and_then(|v| v.get(i as usize).copied()).unwrap_or(0)
        }).unwrap();
        linker.func_wrap("host", "read_file", |_: Caller<HostState>, _: i32| -> i32 { 0 }).unwrap();
        linker.func_wrap("host", "write_file", |_: Caller<HostState>, _: i32, _: i32| {}).unwrap();
        linker.func_wrap("host", "str_add", |_: Caller<HostState>, _: i32, _: i32| -> i32 { 0 }).unwrap();
        linker.func_wrap("host", "str_eq", |_: Caller<HostState>, _: i32, _: i32| -> i32 { 0 }).unwrap();
        linker.func_wrap("host", "str_int", |_: Caller<HostState>, _: i64| -> i32 { 0 }).unwrap();
        linker.func_wrap("host", "str_len", |_: Caller<HostState>, _: i32| -> i32 { 0 }).unwrap();
        linker.func_wrap("host", "str_at", |_: Caller<HostState>, _: i32, _: i64| -> i32 { 0 }).unwrap();
        linker.func_wrap("host", "str_cmp", |_: Caller<HostState>, _: i32, _: i32, _: i32| -> i32 { 0 }).unwrap();
        linker.func_wrap("host", "tenth_alloc", |_: Caller<HostState>, _: i32| -> i32 { 0 }).unwrap();
        linker.func_wrap("host", "compile_host", |_: Caller<HostState>, _: i32, _: i32| -> i32 { 0 }).unwrap();
        linker.func_wrap("host", "f64_bits", |_: Caller<HostState>, _: f64| -> i64 { 0 }).unwrap();
        linker.func_wrap("host", "str_slice", |_: Caller<HostState>, _: i32, _: i64, _: i64| -> i32 { 0 }).unwrap();
        linker.func_wrap("host", "tensor_from_vec", |_: Caller<HostState>, _: i32, _: i32, _: i32| -> i64 { 0 }).unwrap();
        // F1 Phase 2：f16/bf16 张量 hostcall stub（测试不涉及 f16/bf16，返回空指针）
        linker.func_wrap("host", "host_make_tensor_f16", |_: Caller<HostState>, _: i32, _: i32, _: i32| -> i64 { 0 }).unwrap();
        linker.func_wrap("host", "host_make_tensor_bf16", |_: Caller<HostState>, _: i32, _: i32, _: i32| -> i64 { 0 }).unwrap();

        (store, linker)
    }

    /// 编译并调用返回 f32 的函数（Phase 5：真正的 F32 路径）。
    fn call_fn_f32(src: &str, fn_name: &str, args: &[f32]) -> f32 {
        let wasm = compile_to_wasm(src);
        assert_eq!(&wasm[..4], b"\0asm", "WASM magic");
        let engine = Engine::default();
        let module = Module::new(&engine, &wasm).expect("module compile");
        let (mut store, linker) = setup_store_and_linker(&engine);
        let instance = linker.instantiate(&mut store, &module).expect("inst").start(&mut store).expect("start");
        let func = instance.get_func(&store, fn_name).expect("get fn");
        let params: Vec<wasmi::Val> = args.iter().map(|&v| wasmi::Val::F32(v.into())).collect();
        let mut results = [wasmi::Val::F32(0.0.into())];
        func.call(&mut store, &params, &mut results).expect("call");
        match results[0] { wasmi::Val::F32(v) => v.to_float(), _ => panic!("unexpected return type") }
    }

    /// 编译并调用无参返回 f32 的函数。
    fn call_fn_f32_no_args(src: &str, fn_name: &str) -> f32 {
        call_fn_f32(src, fn_name, &[])
    }

    // ── 测试 1：f32 字面量算术（Phase 5 真正 F32 路径）──────────────────────
    // Phase 5 后：F32Const → F32Add → F32 返回，全程 f32 类型
    #[test]
    fn test_f32_literal_add() {
        let src = "fn fadd(a:f32,b:f32)->f32{a+b}";
        let r = call_fn_f32(src, "fadd", &[1.5, 2.0]);
        assert!((r - 3.5).abs() < 1e-6, "fadd(1.5f32, 2.0f32) should be 3.5, got {r}");
    }

    #[test]
    fn test_f32_literal_sub() {
        let src = "fn fsub(a:f32,b:f32)->f32{a-b}";
        let r = call_fn_f32(src, "fsub", &[5.0, 1.5]);
        assert!((r - 3.5).abs() < 1e-6, "fsub(5.0f32, 1.5f32) should be 3.5, got {r}");
    }

    #[test]
    fn test_f32_literal_mul() {
        let src = "fn fmul(a:f32,b:f32)->f32{a*b}";
        let r = call_fn_f32(src, "fmul", &[2.0, 3.0]);
        assert!((r - 6.0).abs() < 1e-6, "fmul(2.0f32, 3.0f32) should be 6.0, got {r}");
    }

    #[test]
    fn test_f32_literal_div() {
        let src = "fn fdiv(a:f32,b:f32)->f32{a/b}";
        let r = call_fn_f32(src, "fdiv", &[7.0, 2.0]);
        assert!((r - 3.5).abs() < 1e-6, "fdiv(7.0f32, 2.0f32) should be 3.5, got {r}");
    }

    // ── 测试 2：f32 字面量后缀解析（Phase 2 已支持，WASM 端 e2e 验证）─────
    #[test]
    fn test_f32_suffix_literal() {
        // 3.14f32 字面量 → Phase 5 真正 F32 计算 → 返回 ~6.28
        let src = "fn pi2()->f32{3.14f32 + 3.14f32}";
        let r = call_fn_f32_no_args(src, "pi2");
        assert!((r - 6.28).abs() < 1e-5, "3.14f32 + 3.14f32 should be ~6.28, got {r}");
    }

    // ── 测试 3：f32 一元负号（F32Neg）──────────────────────────────────
    #[test]
    fn test_f32_neg() {
        let src = "fn fneg(a:f32)->f32{-a}";
        let r = call_fn_f32(src, "fneg", &[2.5]);
        assert!((r - (-2.5)).abs() < 1e-6, "fneg(2.5f32) should be -2.5, got {r}");
    }

    // ── 测试 4：f32 比较（F32Eq/F32Lt）──────────────────────────────────
    #[test]
    fn test_f32_eq() {
        // f32 == 比较 → 返回 bool (i32) → if 分支 → 返回 i64
        let wasm = compile_to_wasm("fn feq(a:f32,b:f32)->i64{ if a==b { 1 } else { 0 } }");
        let engine = Engine::default();
        let module = Module::new(&engine, &wasm).expect("module");
        let (mut store, linker) = setup_store_and_linker(&engine);
        let instance = linker.instantiate(&mut store, &module).expect("inst").start(&mut store).expect("start");
        let func = instance.get_func(&store, "feq").expect("get fn");
        // Phase 5：f32 参数现在用 F32 而非 F64
        let params = vec![wasmi::Val::F32(1.5f32.into()), wasmi::Val::F32(1.5f32.into())];
        let mut results = [wasmi::Val::I64(0)];
        func.call(&mut store, &params, &mut results).expect("call");
        assert_eq!(match results[0] { wasmi::Val::I64(v) => v, _ => panic!() }, 1);
    }

    #[test]
    fn test_f32_lt() {
        let wasm = compile_to_wasm("fn flt(a:f32,b:f32)->i64{ if a<b { 1 } else { 0 } }");
        let engine = Engine::default();
        let module = Module::new(&engine, &wasm).expect("module");
        let (mut store, linker) = setup_store_and_linker(&engine);
        let instance = linker.instantiate(&mut store, &module).expect("inst").start(&mut store).expect("start");
        let func = instance.get_func(&store, "flt").expect("get fn");
        let params = vec![wasmi::Val::F32(1.0f32.into()), wasmi::Val::F32(2.0f32.into())];
        let mut results = [wasmi::Val::I64(0)];
        func.call(&mut store, &params, &mut results).expect("call");
        assert_eq!(match results[0] { wasmi::Val::I64(v) => v, _ => panic!() }, 1);
    }

    // ── 测试 5：f32/f64 混合算术（f32 主导）──────────────────────────────
    // f32 + f64 应得到 f64（lower.rs infer_binary_type 规则）
    #[test]
    fn test_f32_f64_mixed() {
        let src = "fn mix(a:f32,b:f64)->f64{a+b}";
        // 混合参数：f32 用 F32，f64 用 F64
        let wasm = compile_to_wasm(src);
        let engine = Engine::default();
        let module = Module::new(&engine, &wasm).expect("module");
        let (mut store, linker) = setup_store_and_linker(&engine);
        let instance = linker.instantiate(&mut store, &module).expect("inst").start(&mut store).expect("start");
        let func = instance.get_func(&store, "mix").expect("get fn");
        let params = vec![wasmi::Val::F32(1.5f32.into()), wasmi::Val::F64(2.5.into())];
        let mut results = [wasmi::Val::F64(0.0.into())];
        func.call(&mut store, &params, &mut results).expect("call");
        let r = match results[0] { wasmi::Val::F64(v) => v.to_float(), _ => panic!() };
        assert!((r - 4.0).abs() < 1e-6, "1.5f32 + 2.5f64 should be 4.0, got {r}");
    }

    // ── 测试 6：f32 let 绑定与重赋值 ─────────────────────────────────────
    #[test]
    fn test_f32_let_reassign() {
        let src = "fn flet()->f32{ let x: f32 = 1.0f32; x = x + 2.0f32; x }";
        let r = call_fn_f32_no_args(src, "flet");
        assert!((r - 3.0).abs() < 1e-6, "let x=1.0f32; x=x+2.0f32; should be 3.0, got {r}");
    }

    // ── 测试 7：f32 复合表达式（链式算术）──────────────────────────────
    #[test]
    fn test_f32_chain_arithmetic() {
        // (a + b) * c - d / a  = (1 + 2) * 3 - 0.5 / 1 = 9 - 0.5 = 8.5
        let src = "fn chain(a:f32,b:f32,c:f32,d:f32)->f32{ (a+b)*c - d/a }";
        let r = call_fn_f32(src, "chain", &[1.0, 2.0, 3.0, 0.5]);
        assert!((r - 8.5).abs() < 1e-5, "chain should be 8.5, got {r}");
    }

    // ── 测试 8：f32 与整数混合（i64→f32 转换）────────────────────────────
    // 函数签名 (a:f32, b:i64) → WASM valtype (F32, I64)
    #[test]
    fn test_f32_int_mixed() {
        let wasm = compile_to_wasm("fn mix_int(a:f32,b:i64)->f32{a + b}");
        assert_eq!(&wasm[..4], b"\0asm", "WASM magic");
        let engine = Engine::default();
        let module = Module::new(&engine, &wasm).expect("module compile");
        let (mut store, linker) = setup_store_and_linker(&engine);
        let instance = linker.instantiate(&mut store, &module).expect("inst").start(&mut store).expect("start");
        let func = instance.get_func(&store, "mix_int").expect("get fn");
        // Phase 5：f32 参数用 F32，i64 参数用 I64
        let params = vec![wasmi::Val::F32(1.5f32.into()), wasmi::Val::I64(2)];
        let mut results = [wasmi::Val::F32(0.0f32.into())];
        func.call(&mut store, &params, &mut results).expect("call");
        let r = match results[0] { wasmi::Val::F32(v) => v.to_float(), _ => panic!() };
        assert!((r - 3.5).abs() < 1e-5, "1.5f32 + 2i64 should be 3.5, got {r}");
    }

    // ── 测试 9：f32 算术编译执行不崩 ─────────────────────────────────────
    #[test]
    fn test_f32_tensor_literal_compiles() {
        let src = "fn tensor_test()->f32{ 1.5f32 + 2.5f32 }";
        let wasm = compile_to_wasm(src);
        assert_eq!(&wasm[..4], b"\0asm", "WASM magic");
        let engine = Engine::default();
        let module = Module::new(&engine, &wasm).expect("module");
        let (mut store, linker) = setup_store_and_linker(&engine);
        let instance = linker.instantiate(&mut store, &module).expect("inst").start(&mut store).expect("start");
        let func = instance.get_func(&store, "tensor_test").expect("get fn");
        let mut results = [wasmi::Val::F32(0.0f32.into())];
        func.call(&mut store, &[], &mut results).expect("call");
        let v = match results[0] { wasmi::Val::F32(v) => v.to_float(), _ => panic!() };
        assert!((v - 4.0).abs() < 1e-6, "1.5f32 + 2.5f32 should be 4.0, got {v}");
    }

    // ── 测试 10：f32 在 main 中返回（exit_code 路径）─────────────────────
    // main 被导出为 ()->i32，i32 exit code 路径的 wrap_to_i32 处理 f32 应正常
    #[test]
    fn test_f32_main_wrap() {
        let src = "fn main()->i64{ let x: f32 = 3.14f32; 0 }";
        let wasm = compile_to_wasm(src);
        let engine = Engine::default();
        let module = Module::new(&engine, &wasm).expect("module");
        let (mut store, linker) = setup_store_and_linker(&engine);
        let instance = linker.instantiate(&mut store, &module).expect("inst").start(&mut store).expect("start");
        // main 签名为 () -> i32
        let main_fn = instance.get_func(&store, "main").expect("get main");
        let mut results = [wasmi::Val::I32(0)];
        main_fn.call(&mut store, &[], &mut results).expect("call");
        assert_eq!(match results[0] { wasmi::Val::I32(v) => v, _ => panic!() }, 0);
    }
}
