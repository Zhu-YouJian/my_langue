//! M1-S1：WASM 后端 4 项缺口守护测试。
//!
//! 覆盖（均通过 wasmi 实例化 + 执行验证，非仅编译）：
//!   P1 标签 break/continue（`break 'outer` / `continue 'outer`）
//!   P2 泛型枚举布局 + match（`enum Opt<T> { Some(T), None }`，含 f64 字段存储往返）
//!   P3 函数内引用全局（`let g = ..` 程序级全局在函数体内 GlobalGet/GlobalSet）
//!   P4 自定义运算符 native（operator 定义体内调用标量 math native：sqrt/abs/sin/cos/ln/pow）
//!
//! 修复前：P1 报「标签 break 暂不支持 WASM 后端」；P2 报「不支持的表达式 Match」；
//! P3 报「未定义变量 'g'」；P4 报「未定义函数 'sqrt'」。
//! 修复后：全部编译 + wasmi 执行语义与 VM/解释器一致。

#[cfg(test)]
mod m1s1_wasm_gaps {
    use wasmi::{Engine, Module, Store, Linker, Caller};
    use std::collections::HashMap;
    use std::sync::Mutex;

    struct HostState {
        vecs: Mutex<HashMap<i64, Vec<i64>>>,
        next_vec_id: Mutex<i64>,
        printed: Mutex<Vec<i64>>,
    }

    /// Compile a Tenth source string to WASM using the Rust mother compiler.
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

    /// Set up a wasmi store + linker with all host imports, bound to the given engine.
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
        linker.func_wrap("host", "host_make_tensor_f16", |_: Caller<HostState>, _: i32, _: i32, _: i32| -> i64 { 0 }).unwrap();
        linker.func_wrap("host", "host_make_tensor_bf16", |_: Caller<HostState>, _: i32, _: i32, _: i32| -> i64 { 0 }).unwrap();
        // M1-S1（P4）：标量 math host stub（真实语义在 host.rs，测试仅需可链接）
        linker.func_wrap("host", "host_sin", |_: Caller<HostState>, x: f64| -> f64 { x.sin() }).unwrap();
        linker.func_wrap("host", "host_cos", |_: Caller<HostState>, x: f64| -> f64 { x.cos() }).unwrap();
        linker.func_wrap("host", "host_ln", |_: Caller<HostState>, x: f64| -> f64 { x.ln() }).unwrap();
        linker.func_wrap("host", "host_pow", |_: Caller<HostState>, b: f64, e: f64| -> f64 { b.powf(e) }).unwrap();

        (store, linker)
    }

    /// Compile source, instantiate, call an exported function with i64 args, return i64.
    fn call_fn_i64(src: &str, fn_name: &str, args: &[i64]) -> i64 {
        let wasm = compile_to_wasm(src);
        assert_eq!(&wasm[..4], b"\0asm", "WASM magic");
        let engine = Engine::default();
        let module = Module::new(&engine, &wasm).expect("module compile");
        let (mut store, linker) = setup_store_and_linker(&engine);
        let instance = linker.instantiate(&mut store, &module).expect("inst").start(&mut store).expect("start");
        let func = instance.get_func(&store, fn_name).expect("get fn");
        let params: Vec<wasmi::Val> = args.iter().map(|&v| wasmi::Val::I64(v)).collect();
        let mut results = [wasmi::Val::I64(0)];
        func.call(&mut store, &params, &mut results).expect("call");
        match results[0] { wasmi::Val::I64(v) => v, _ => panic!("unexpected return type") }
    }

    /// Compile source, run `main` first（初始化程序级全局），再调用导出函数。
    fn call_main_then_fn_i64(src: &str, fn_name: &str, args: &[i64]) -> i64 {
        let wasm = compile_to_wasm(src);
        assert_eq!(&wasm[..4], b"\0asm", "WASM magic");
        let engine = Engine::default();
        let module = Module::new(&engine, &wasm).expect("module compile");
        let (mut store, linker) = setup_store_and_linker(&engine);
        let instance = linker.instantiate(&mut store, &module).expect("inst").start(&mut store).expect("start");
        // main 先跑：初始化程序级全局（GlobalSet）
        let main = instance.get_func(&store, "main").expect("get main");
        let mut main_res = [wasmi::Val::I32(0)];
        main.call(&mut store, &[], &mut main_res).expect("main call");
        // 再调目标函数（此时全局已初始化）
        let func = instance.get_func(&store, fn_name).expect("get fn");
        let params: Vec<wasmi::Val> = args.iter().map(|&v| wasmi::Val::I64(v)).collect();
        let mut results = [wasmi::Val::I64(0)];
        func.call(&mut store, &params, &mut results).expect("call");
        match results[0] { wasmi::Val::I64(v) => v, _ => panic!("unexpected return type") }
    }

    /// Compile source, instantiate, call an exported f64 function, return f64.
    fn call_fn_f64(src: &str, fn_name: &str, args: &[f64]) -> f64 {
        let wasm = compile_to_wasm(src);
        assert_eq!(&wasm[..4], b"\0asm", "WASM magic");
        let engine = Engine::default();
        let module = Module::new(&engine, &wasm).expect("module compile");
        let (mut store, linker) = setup_store_and_linker(&engine);
        let instance = linker.instantiate(&mut store, &module).expect("inst").start(&mut store).expect("start");
        let func = instance.get_func(&store, fn_name).expect("get fn");
        let params: Vec<wasmi::Val> = args.iter().map(|&v| wasmi::Val::F64(v.into())).collect();
        let mut results = [wasmi::Val::F64(0.0.into())];
        func.call(&mut store, &params, &mut results).expect("call");
        match results[0] { wasmi::Val::F64(v) => v.to_float(), _ => panic!("unexpected return type") }
    }

    // ════════════════════════════════════════════════════════════════════════
    // P1：标签 break / continue
    // ════════════════════════════════════════════════════════════════════════

    #[test]
    fn p1_labeled_break_outer_while() {
        // break 'outer 从内层循环（含 if）跳出到最外层 while
        let src = r#"fn f() -> i64 {
            let s = 0;
            'outer: while s < 100 {
                s = s + 1;
                'inner: while s < 100 {
                    if s == 3 { break 'outer; }
                    s = s + 1;
                }
            }
            s
        }"#;
        assert_eq!(call_fn_i64(src, "f", &[]), 3);
    }

    #[test]
    fn p1_labeled_continue_outer_for() {
        // continue 'outer：跳过内层 for 剩余，跳到外层 for 的 i += 1
        // 外层 0..4，内层 0..4；j==2 时 continue 外层 → 每外层迭代只执行 2 次内层
        // 计数：外层 4 次 × 内层 j=0,1 共 2 次 = 8
        let src = r#"fn f() -> i64 {
            let total = 0;
            'outer: for i in 0..4 {
                'inner: for j in 0..4 {
                    if j == 2 { continue 'outer; }
                    total = total + 1;
                }
            }
            total
        }"#;
        assert_eq!(call_fn_i64(src, "f", &[]), 8);
    }

    #[test]
    fn p1_labeled_break_three_levels() {
        // 三层嵌套，break 'outer 从最内层直接跳到最外层
        let src = r#"fn f() -> i64 {
            let mut acc = 0;
            'a: for i in 0..10 {
                'b: for j in 0..10 {
                    'c: for k in 0..10 {
                        acc = acc + 1;
                        if acc == 5 { break 'a; }
                    }
                }
            }
            acc
        }"#;
        assert_eq!(call_fn_i64(src, "f", &[]), 5);
    }

    #[test]
    fn p1_unlabeled_break_continue_regression() {
        // 无标签 break/continue 行为不变（既有语义回归守护）
        let src = r#"fn f() -> i64 {
            let total = 0;
            for i in 0..10 {
                if i == 3 { continue; }
                if i == 8 { break; }
                total = total + i;
            }
            total
        }"#;
        // 0+1+2+4+5+6+7 = 25（跳过 3，8 处 break）
        assert_eq!(call_fn_i64(src, "f", &[]), 25);
    }

    // ════════════════════════════════════════════════════════════════════════
    // P2：泛型枚举布局 + match
    // ════════════════════════════════════════════════════════════════════════

    #[test]
    fn p2_generic_enum_i64_field() {
        let src = r#"enum Opt<T> { Some(T), None }
fn get(o: Opt<i64>) -> i64 { match o { Opt::Some(x) => x, Opt::None => 0 } }
fn f() -> i64 { let a = get(Opt::Some(42)); let b = get(Opt::None); a + b }"#;
        assert_eq!(call_fn_i64(src, "f", &[]), 42);
    }

    #[test]
    fn p2_generic_enum_f64_field_roundtrip() {
        // f64 字段经 i64 位存储往返（EnumLiteral 存 / Match 取）应保持数值
        let src = r#"enum Wrap<T> { W(T) }
fn f() -> i64 {
    let e = Wrap::W(3.5);
    match e { Wrap::W(x) => { if x == 3.5 { 1 } else { 0 } } }
}"#;
        assert_eq!(call_fn_i64(src, "f", &[]), 1);
    }

    #[test]
    fn p2_generic_enum_multi_field() {
        // 多字段泛型枚举 + 单元变体
        let src = r#"enum Pair<T> { P(T, T), Empty }
fn sum(p: Pair<i64>) -> i64 { match p { Pair::P(a, b) => a + b, Pair::Empty => 0 } }
fn f() -> i64 { sum(Pair::P(10, 32)) }"#;
        assert_eq!(call_fn_i64(src, "f", &[]), 42);
    }

    #[test]
    fn p2_nongeneric_enum_match() {
        // 非泛型枚举 match（回归：既有枚举也应可用）
        let src = r#"enum Color { Red(i64), Blue }
fn pick(c: Color) -> i64 { match c { Color::Red(x) => x, Color::Blue => 7 } }
fn f() -> i64 { pick(Color::Red(5)) + pick(Color::Blue) }"#;
        assert_eq!(call_fn_i64(src, "f", &[]), 12);
    }

    // ════════════════════════════════════════════════════════════════════════
    // P3：函数内引用全局
    // ════════════════════════════════════════════════════════════════════════

    #[test]
    fn p3_fn_reads_global() {
        let src = r#"let g = 10;
fn read() -> i64 { g + 1 }
fn main() -> i64 { read() }"#;
        assert_eq!(call_main_then_fn_i64(src, "read", &[]), 11);
    }

    #[test]
    fn p3_fn_mutates_global() {
        // 函数内写全局（GlobalSet），再读回；main 只做全局初始化
        let src = r#"let mut counter = 0;
fn bump() -> i64 { counter = counter + 5; counter }
fn main() -> i64 { 0 }"#;
        assert_eq!(call_main_then_fn_i64(src, "bump", &[]), 5);
    }

    #[test]
    fn p3_global_f64_storage() {
        // 全局 f64 声明 + 函数内读写（存储转换正确性）
        let src = r#"let mut scale: f64 = 2.0;
fn double(x: f64) -> f64 { scale = scale * 2.0; x * scale }
fn main() -> i64 { 0 }"#;
        // 直接调用 double 需要全局已初始化（main 已跑过）
        let wasm = compile_to_wasm(src);
        assert_eq!(&wasm[..4], b"\0asm");
        let engine = Engine::default();
        let module = Module::new(&engine, &wasm).expect("module");
        let (mut store, linker) = setup_store_and_linker(&engine);
        let instance = linker.instantiate(&mut store, &module).expect("inst").start(&mut store).expect("start");
        let main = instance.get_func(&store, "main").expect("main");
        let mut mr = [wasmi::Val::I32(0)];
        main.call(&mut store, &[], &mut mr).expect("main");
        let func = instance.get_func(&store, "double").expect("double");
        let mut results = [wasmi::Val::F64(0.0.into())];
        func.call(&mut store, &[wasmi::Val::F64(3.0.into())], &mut results).expect("call");
        let v = match results[0] { wasmi::Val::F64(x) => x.to_float(), _ => panic!() };
        assert!((v - 12.0).abs() < 1e-9, "double(3.0) with scale=2.0→4.0 should be 12.0, got {v}");
    }

    // ════════════════════════════════════════════════════════════════════════
    // P4：自定义运算符 native（operator 定义体内调用标量 math native）
    // ════════════════════════════════════════════════════════════════════════

    #[test]
    fn p4_operator_sqrt() {
        // operator 定义体内调用 sqrt（WASM 内联 F64Sqrt）
        let src = r#"operator ~@ = fn(a: f64, b: f64) -> f64 { sqrt(a * a + b * b) }
fn f(x: f64, y: f64) -> f64 { x ~@ y }"#;
        let v = call_fn_f64(src, "f", &[3.0, 4.0]);
        assert!((v - 5.0).abs() < 1e-9, "hypot(3,4) should be 5, got {v}");
    }

    #[test]
    fn p4_operator_abs() {
        // abs 在类型系统返回 F64（infer_scalar_dtype 默认 f64），operator 用 f64 一致
        let src = r#"operator @$ = fn(a: f64, b: f64) -> f64 { abs(a) + abs(b) }
fn f() -> f64 { (-3.0) @$ (-4.0) }"#;
        let v = call_fn_f64(src, "f", &[]);
        assert!((v - 7.0).abs() < 1e-9, "abs sum should be 7.0, got {v}");
    }

    #[test]
    fn p4_native_sin_cos_ln_pow() {
        // 宿主 math 函数（sin/cos/ln/pow）经 host import 调用
        let src = r#"fn f(x: f64) -> f64 { sin(x) * cos(x) }
fn g(x: f64) -> f64 { ln(x) }
fn h(b: f64, e: f64) -> f64 { pow(b, e) }"#;
        let v = call_fn_f64(src, "f", &[0.5]);
        assert!((v - 0.5f64.sin() * 0.5f64.cos()).abs() < 1e-9, "sin*cos mismatch: {v}");
        let v = call_fn_f64(src, "g", &[2.718281828]);
        assert!((v - 1.0).abs() < 1e-6, "ln(e) should be ~1, got {v}");
        let v = call_fn_f64(src, "h", &[2.0, 10.0]);
        assert!((v - 1024.0).abs() < 1e-9, "pow(2,10) should be 1024, got {v}");
    }

    #[test]
    fn p4_basic_operator_regression() {
        // M3.1 既有自定义运算符回归（不破坏）
        let src = "operator @@ = fn(a: i64, b: i64) -> i64 { a + b }\nfn f() -> i64 { 1 @@ 2 }";
        assert_eq!(call_fn_i64(src, "f", &[]), 3);
    }
}
