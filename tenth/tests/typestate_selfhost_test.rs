//! 阶段2a M2（G1 同步）：tenthc 自举解析 typestate 语法验证。
//!
//! 裁决：typestate 不参与自举（tenthc 源码不用它），但 tenthc 必须能
//! **解析** `impl File<Open>` 这类目标带类型实参的 impl 语法而不崩。
//! 修复：`tenthc/parser/parser.th::parse_impl_block` 在目标类型名后跳过
//! `<...>`（调用 parse_generic_params）。tenthc 语义上不按状态特化
//! （与 Rust 侧 M2 前一致），仅保证语法兼容。
//!
//! 验证路径：Rust 母编译器编译 tenthc 源码（含 typestate 测试源码嵌入）
//! → WASM-A；wasmi 执行 WASM-A，其 main 对 typestate 源码调用
//! lexer_new → lexer_tokenize → parse_program → lower_program →
//! compile_to_wasm，产出 WASM-B。WASM-B 非空即说明 tenthc 完整管线
//! （解析→lower→WASM 后端）处理 typestate 语法成功。

use wasmi::{Config, Engine, Module, Store, Linker, StackLimits};
use tenth::compile::wasm::register_host_functions;

fn selfhost_config() -> Config {
    let mut config = Config::default();
    let limits = StackLimits::new(
        65536,
        1_048_576,
        65536,
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
    let full_src = format!("{}fn main()->Vec<i64>{{let mut lex=lexer_new(\"{}\");let tokens=lexer_tokenize(&mut lex);let program=parse_program(tokens);let hir=lower_program(program);let wasm=compile_to_wasm(hir);wasm}}", selfhost_src, escaped);
    let mut lexer = Lexer::new(&full_src);
    let tokens = lexer.tokenize().expect("lex");
    let mut parser = Parser::new(tokens);
    let program = parser.parse_program().expect("parse");
    let mut lowerer = Lowerer::new();
    let hir = lowerer.lower_program(&program).expect("lower");
    compile::compile_to_wasm(&hir).expect("compile")
}

fn read_vec_from_memory(store: &Store<u32>, mem: &wasmi::Memory, vec_ptr: i64) -> Vec<u8> {
    let data = mem.data(store);
    let vp = vec_ptr as i32 as usize;
    if vp + 20 > data.len() { return Vec::new(); }
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

/// tenthc 必须能解析 `impl File<Open>`（目标带状态实参）并走完整个编译管线。
#[test]
fn tenthc_parses_typestate_impl_syntax() {
    // typestate 核心语法：状态 enum + 泛型 struct + 特化 impl + 状态转换返回类型
    let test_src = r#"
enum Open {}
enum Closed {}
struct File<S> { path: str }
impl File<Open> {
    fn read(self) -> str { self.path }
    fn close(self) -> File<Closed> { File<Closed> { path: self.path } }
}
impl File<Closed> {
    fn reopen(self) -> File<Open> { File<Open> { path: self.path } }
}
fn main() {
    let f = File<Open> { path: "a.txt" };
    let c = f.close();
}
"#;

    // Stage 1：Rust 母编译器编译 tenthc 源码（含修改后的 parser.th）→ WASM-A
    let wasm_a = compile_selfhost_to_wasm(test_src);
    assert_eq!(&wasm_a[..4], b"\0asm", "WASM-A 必须有效");

    // Stage 2：wasmi 执行 WASM-A，tenthc 解析并编译 typestate 源码 → WASM-B
    let config = selfhost_config();
    let engine = Engine::new(&config);
    let module = Module::new(&engine, &wasm_a).expect("compile WASM-A");
    let mut store = Store::new(&engine, 8192u32);
    let mut linker = Linker::new(&engine);
    register_host_functions(&mut linker).expect("register host functions");

    let inst = linker.instantiate(&mut store, &module).expect("instantiate").start(&mut store).expect("start");
    let main_fn = inst.get_func(&store, "main").expect("main");
    let mut r = [wasmi::Val::I32(0)];
    main_fn.call(&mut store, &[], &mut r).expect("call main");
    let vec_ptr = match r[0] {
        wasmi::Val::I32(v) => v as i64,
        wasmi::Val::I64(v) => v,
        _ => panic!("expected i32/i64 return from main"),
    };
    let mem = inst.get_memory(&store, "memory").expect("memory");
    let wasm_b = read_vec_from_memory(&store, &mem, vec_ptr);

    assert!(!wasm_b.is_empty(), "WASM-B 必须非空：tenthc 未能解析 typestate 源码");
    assert_eq!(&wasm_b[..4], b"\0asm", "WASM-B 必须有有效 magic");
}
