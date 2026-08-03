//! 阶段2a M2（G1+G2+G3+G4）测试：typestate —— 非法状态表达不出来。
//!
//! 场景：`struct File<S>` + `impl File<Open>` / `impl File<Closed>` 特化 +
//! `close(self) -> File<Closed>` 状态转换。目标：
//!
//! - G1：泛型 struct 字面量 `File<Open> { ... }` 携带状态实参（Type::Generic）
//! - G2：`impl File<Open>` 与 `impl File<Closed>` 方法互不覆盖（按状态特化注册）
//! - G3：方法解析按 receiver 状态过滤；`File<Closed>` 上调用 Open 专属方法 → 编译期报错；
//!       `close` 的返回状态 `File<Closed>` 传播到后续解析
//! - G4：状态转换方法（`close(self) -> File<Closed>`）消费 receiver——旧变量标记 Moved；
//!       `read(self) -> str`（状态不变）不消费，`f.read(); f.close()` 模式可用
//!
//! 覆盖：合法状态链（解释器+VM 双路径）、非法状态编译期报错、状态不覆盖、
//! close 后变量状态（G4）、以及 generic/trait/generic_method 回归。

use tenth::lexer::lexer::Lexer;
use tenth::parser::parser::Parser;
use tenth::hir::lower::Lowerer;
use tenth::runtime::interpreter::Interpreter;
use tenth::runtime::vm::Vm;
use tenth::runtime::value::Value;
use tenth::runtime::tensor::Tensor;
use tenth::compile::bytecode::BytecodeCompiler;
use tenth::compile::jit;
use std::rc::Rc;
use std::cell::RefCell;

/// 解释器路径：lex → parse → lower → run。
fn run(src: &str) -> Result<Option<Value>, String> {
    let mut lexer = Lexer::new(src);
    let tokens = lexer.tokenize().map_err(|e| e.to_string())?;
    let mut parser = Parser::new(tokens);
    let program = parser.parse_program().map_err(|e| e.to_string())?;
    let mut lowerer = Lowerer::new();
    let hir = lowerer.lower_program(&program).map_err(|e| e.to_string())?;
    let mut interpreter = Interpreter::new(&hir);
    interpreter.execute_program(&hir).map_err(|e| e.to_string())
}

/// VM 路径：lex → parse → lower → bytecode → 执行 main。
fn run_vm(src: &str) -> Result<Value, String> {
    let mut lexer = Lexer::new(src);
    let tokens = lexer.tokenize().map_err(|e| e.to_string())?;
    let mut parser = Parser::new(tokens);
    let program = parser.parse_program().map_err(|e| e.to_string())?;
    let mut lowerer = Lowerer::new();
    let hir = lowerer.lower_program(&program).map_err(|e| e.to_string())?;

    let mut vm = Vm::new();
    vm.add_native("println".into(), |_vm, args| {
        for a in args { print!("{a}"); }
        println!();
        Ok(Value::Unit)
    });
    vm.add_native("Vec::new".into(), |_vm, _args| {
        Ok(Value::Vec(Rc::new(RefCell::new(Vec::new()))))
    });
    vm.add_native("zeros".into(), |_vm, args| {
        let shape: Vec<usize> = args.iter().map(|a| a.as_int().unwrap_or(1) as usize).collect();
        Ok(Value::Tensor(Rc::new(RefCell::new(Tensor::zeros(&shape)))))
    });
    vm.add_native("ones".into(), |_vm, args| {
        let shape: Vec<usize> = args.iter().map(|a| a.as_int().unwrap_or(1) as usize).collect();
        Ok(Value::Tensor(Rc::new(RefCell::new(Tensor::ones(&shape)))))
    });

    for func in &hir.functions {
        let compiler = BytecodeCompiler::new();
        match compiler.compile(func) {
            Ok((chunk, closures)) => {
                vm.add_fn(func.name.clone(), chunk);
                for (name, closure_chunk) in closures {
                    vm.add_fn(name, closure_chunk);
                }
            }
            Err(e) => return Err(format!("compile error: {}", e)),
        }
    }

    if let Some(ref expr) = hir.main_expr {
        let compiler = BytecodeCompiler::new();
        match compiler.compile_main(expr) {
            Ok((chunk, closures)) => {
                vm.add_fn("main".into(), chunk);
                for (name, closure_chunk) in closures {
                    vm.add_fn(name, closure_chunk);
                }
            }
            Err(e) => return Err(format!("compile error: {}", e)),
        }
        vm.call("main").map_err(|e| e.to_string())
    } else if vm.has_fn("main") {
        vm.call("main").map_err(|e| e.to_string())
    } else {
        Ok(Value::Unit)
    }
}

fn expect_str(v: &Value, expected: &str, label: &str) {
    match v {
        Value::String(s) => assert_eq!(s, expected, "{}: 期望 String(\"{}\"), 实际 {}", label, expected, s),
        other => panic!("{}: 期望 String(\"{}\"), 实际 {:?}", label, expected, other),
    }
}

fn expect_int(v: &Value, expected: i64, label: &str) {
    match v {
        Value::Int(n, _) => assert_eq!(*n, expected, "{}: 期望 Int({}), 实际 {}", label, expected, n),
        other => panic!("{}: 期望 Int({}), 实际 {:?}", label, expected, other),
    }
}

// ── 用例 1：合法状态链（read → close → reopen → read），双路径 ──

const LEGAL_CHAIN_SRC: &str = r#"
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
let f = File<Open> { path: "a.txt" };
let s = f.read();
let c = f.close();
c.reopen().read()
"#;

#[test]
fn typestate_legal_chain_interpreter() {
    let result = run(LEGAL_CHAIN_SRC).unwrap();
    expect_str(&result.unwrap(), "a.txt", "解释器");
}

#[test]
fn typestate_legal_chain_vm() {
    let result = run_vm(LEGAL_CHAIN_SRC).unwrap();
    expect_str(&result, "a.txt", "VM");
}

// ── 用例 2：非法状态编译期报错（File<Closed> 上调用 Open 专属 read）──

const ILLEGAL_READ_ON_CLOSED_SRC: &str = r#"
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
let f = File<Open> { path: "a.txt" };
let c = f.close();
c.read()
"#;

#[test]
fn typestate_illegal_state_compile_error_interpreter() {
    let err = run(ILLEGAL_READ_ON_CLOSED_SRC).unwrap_err();
    assert!(
        err.contains("没有方法 'read'"),
        "解释器路径：应报「没有方法 'read'」，实际: {}",
        err
    );
}

#[test]
fn typestate_illegal_state_compile_error_vm() {
    // 报错发生在 lowering 阶段（两条路径共享同一 lowerer），此处验证 VM 测试
    // 入口同样在 lower 处失败（编译期报错而非运行期）。
    let err = run_vm(ILLEGAL_READ_ON_CLOSED_SRC).unwrap_err();
    assert!(
        err.contains("没有方法 'read'"),
        "VM 路径：应报「没有方法 'read'」，实际: {}",
        err
    );
}

// ── 用例 3：状态互不覆盖——两个状态各自定义同名方法 read（audit tmp1 逆命题）──

const STATE_SPECIALIZATION_SRC: &str = r#"
enum Open {}
enum Closed {}
struct File<S> { path: str }
impl File<Open> {
    fn read(self) -> i64 { 1 }
}
impl File<Closed> {
    fn read(self) -> i64 { 2 }
}
fn open() -> File<Open> { File<Open> { path: "x" } }
fn closed() -> File<Closed> { File<Closed> { path: "y" } }
open().read() * 10 + closed().read()
"#;

#[test]
fn typestate_state_specialization_no_override_interpreter() {
    let result = run(STATE_SPECIALIZATION_SRC).unwrap();
    expect_int(&result.unwrap(), 12, "解释器");
}

#[test]
fn typestate_state_specialization_no_override_vm() {
    let result = run_vm(STATE_SPECIALIZATION_SRC).unwrap();
    expect_int(&result, 12, "VM");
}

// ── 用例 4（G4）：状态转换消费 receiver——close 后旧变量不可再用 ──

const USE_AFTER_CLOSE_SRC: &str = r#"
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
let f = File<Open> { path: "a.txt" };
let c = f.close();
f.read()
"#;

#[test]
fn typestate_use_after_close_moved() {
    // close 是状态转换（File<Open> → File<Closed>），消费 f；
    // 之后 f.read() 应报「使用了已移动的值 'f'」。
    let err = run(USE_AFTER_CLOSE_SRC).unwrap_err();
    assert!(
        err.contains("使用了已移动的值 'f'"),
        "close 后使用 f 应报 moved，实际: {}",
        err
    );
}

// ── 用例 5（G4）：状态不变方法（read）不消费——f.read(); f.close() 可用 ──
// （即 LEGAL_CHAIN_SRC 已覆盖：read 后 close 仍可用同一变量 f）

// ── 回归：裸 impl + Generic receiver（G5 场景不破坏）──

const BARE_IMPL_GENERIC_RECV_SRC: &str = r#"
struct File<S> { path: str }
impl File {
    fn read(self) -> i64 { 1 }
}
fn make() -> File<Open> { File<Open> { path: "x" } }
make().read()
"#;

#[test]
fn typestate_regression_bare_impl_generic_recv_interpreter() {
    let result = run(BARE_IMPL_GENERIC_RECV_SRC).unwrap();
    expect_int(&result.unwrap(), 1, "解释器");
}

#[test]
fn typestate_regression_bare_impl_generic_recv_vm() {
    let result = run_vm(BARE_IMPL_GENERIC_RECV_SRC).unwrap();
    expect_int(&result, 1, "VM");
}

// ── 回归：字面量 Generic 实例方法（裸 impl 回退）──

const LITERAL_GENERIC_BARE_SRC: &str = r#"
struct File<S> { path: str }
impl File {
    fn read(self) -> i64 { 1 }
}
File<Open> { path: "x" }.read()
"#;

#[test]
fn typestate_regression_literal_generic_bare_interpreter() {
    let result = run(LITERAL_GENERIC_BARE_SRC).unwrap();
    expect_int(&result.unwrap(), 1, "解释器");
}

#[test]
fn typestate_regression_literal_generic_bare_vm() {
    let result = run_vm(LITERAL_GENERIC_BARE_SRC).unwrap();
    expect_int(&result, 1, "VM");
}

// ── 回归：普通（非泛型）struct 方法不破坏 ──

const PLAIN_STRUCT_SRC: &str = r#"
struct Point { x: i64 }
impl Point {
    fn get(self) -> i64 { self.x }
}
fn make() -> Point { Point { x: 7 } }
make().get()
"#;

#[test]
fn typestate_regression_plain_struct_interpreter() {
    let result = run(PLAIN_STRUCT_SRC).unwrap();
    expect_int(&result.unwrap(), 7, "解释器");
}

#[test]
fn typestate_regression_plain_struct_vm() {
    let result = run_vm(PLAIN_STRUCT_SRC).unwrap();
    expect_int(&result, 7, "VM");
}

// ── 回归：泛型 struct 字段访问（G1 类型变化不影响运行时）──

const GENERIC_FIELD_SRC: &str = r#"
struct Pair<T, U> { first: T, second: U }
let p = Pair<i32, f64> { first: 42, second: 3.14 };
p.first + p.second
"#;

#[test]
fn typestate_regression_generic_field_access() {
    let result = run(GENERIC_FIELD_SRC).unwrap();
    match result.unwrap() {
        Value::Float(v) => assert!((v - 45.14).abs() < 1e-10, "期望 45.14，实际 {}", v),
        other => panic!("期望 Float(45.14)，实际 {:?}", other),
    }
}

// ── 回归：trait 方法调用不破坏（trait 方法不注册 __ 前缀，走解释器查表）──

const TRAIT_METHOD_SRC: &str = r#"
struct Point { x: i64, y: i64 }
trait Add {
    fn add(self, other: Point) -> Point;
}
impl Add for Point {
    fn add(self, other: Point) -> Point {
        Point { x: self.x + other.x, y: self.y + other.y }
    }
}
let p = Point { x: 1, y: 2 };
let q = Point { x: 3, y: 4 };
let r = p.add(q);
r.x * 10 + r.y
"#;

#[test]
fn typestate_regression_trait_method() {
    let result = run(TRAIT_METHOD_SRC).unwrap();
    expect_int(&result.unwrap(), 46, "解释器");
}

// ── 用例 6（G4+M3.2）：状态转换消费后，变量作为函数实参也报 moved ──

const USE_AFTER_CLOSE_AS_ARG_SRC: &str = r#"
enum Open {}
enum Closed {}
struct File<S> { path: str }
impl File<Open> {
    fn read(self) -> str { self.path }
    fn close(self) -> File<Closed> { File<Closed> { path: self.path } }
}
fn is_open(f: File<Open>) -> bool { true }
let f = File<Open> { path: "a.txt" };
let c = f.close();
is_open(f)
"#;

#[test]
fn typestate_use_after_close_as_arg_moved() {
    // G4：close 是状态转换（File<Open> → File<Closed>），消费 f；
    // 之后把 f 当函数实参传入（变量引用位置 check_use 生效）→ 报 moved。
    let err = run(USE_AFTER_CLOSE_AS_ARG_SRC).unwrap_err();
    assert!(
        err.contains("使用了已移动的值 'f'"),
        "close 后传 f 给函数应报 moved，实际: {}",
        err
    );
}

// ── 用例 7（G4）：状态转换在链式临时值上不误报（receiver 非变量不标记）──

const CHAIN_TRANSITION_SRC: &str = r#"
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
File<Open> { path: "a.txt" }.close().reopen().read()
"#;

#[test]
fn typestate_chain_transition_no_false_positive() {
    // G4 边界：状态转换 receiver 是临时值（非变量）时不标记任何变量，
    // 链式 close → reopen → read 编译通过、双路径运行一致。
    let result = run(CHAIN_TRANSITION_SRC).unwrap();
    expect_str(&result.unwrap(), "a.txt", "解释器");
    let result_vm = run_vm(CHAIN_TRANSITION_SRC).unwrap();
    expect_str(&result_vm, "a.txt", "VM");
}

// ── M3.3（G1+G2+G3 收尾）：三路径对拍（JIT）+ 覆盖断言 + 字面量状态推断 ──

/// JIT 路径：lex → parse → lower → bytecode → jit::run_jit。
/// run_jit 内部在 JIT 无法编译时自动回退 vm.call，因此总能产出结果
/// （编译期错误在 lower 阶段即抛出，三路径共享 lowerer 天然一致）。
fn run_jit(src: &str) -> Result<Value, String> {
    let mut lexer = Lexer::new(src);
    let tokens = lexer.tokenize().map_err(|e| e.to_string())?;
    let mut parser = Parser::new(tokens);
    let program = parser.parse_program().map_err(|e| e.to_string())?;
    let mut lowerer = Lowerer::new();
    let hir = lowerer.lower_program(&program).map_err(|e| e.to_string())?;

    let mut vm = Vm::new();
    vm.add_native("println".into(), |_vm, args| {
        for a in args { print!("{a}"); }
        println!();
        Ok(Value::Unit)
    });
    vm.add_native("Vec::new".into(), |_vm, _args| {
        Ok(Value::Vec(Rc::new(RefCell::new(Vec::new()))))
    });
    vm.add_native("zeros".into(), |_vm, args| {
        let shape: Vec<usize> = args.iter().map(|a| a.as_int().unwrap_or(1) as usize).collect();
        Ok(Value::Tensor(Rc::new(RefCell::new(Tensor::zeros(&shape)))))
    });
    vm.add_native("ones".into(), |_vm, args| {
        let shape: Vec<usize> = args.iter().map(|a| a.as_int().unwrap_or(1) as usize).collect();
        Ok(Value::Tensor(Rc::new(RefCell::new(Tensor::ones(&shape)))))
    });

    for func in &hir.functions {
        let compiler = BytecodeCompiler::new();
        match compiler.compile(func) {
            Ok((chunk, closures)) => {
                vm.add_fn(func.name.clone(), chunk);
                for (name, closure_chunk) in closures {
                    vm.add_fn(name, closure_chunk);
                }
                // 与 main.rs 对齐：函数注册为全局 FnRef（函数作值传递时按名解析）。
                vm.set_global(func.name.clone(), Value::FnRef {
                    name: func.name.clone(),
                    params: func.params.clone(),
                    return_type: func.return_type.clone(),
                    captures: vec![],
                });
            }
            Err(e) => return Err(format!("compile error: {}", e)),
        }
    }

    if let Some(ref expr) = hir.main_expr {
        let compiler = BytecodeCompiler::new();
        match compiler.compile_main(expr) {
            Ok((chunk, closures)) => {
                vm.add_fn("main".into(), chunk);
                for (name, closure_chunk) in closures {
                    vm.add_fn(name, closure_chunk);
                }
            }
            Err(e) => return Err(format!("compile error: {}", e)),
        }
        jit::run_jit(&mut vm, "main").map_err(|e| e.to_string())
    } else if vm.has_fn("main") {
        jit::run_jit(&mut vm, "main").map_err(|e| e.to_string())
    } else {
        Ok(Value::Unit)
    }
}

/// JIT 路径（保留 Vm 供覆盖断言：检查 mangled 方法是否真的被 JIT 编译）。
fn run_jit_with_vm(src: &str) -> Result<(Value, Vm), String> {
    let mut lexer = Lexer::new(src);
    let tokens = lexer.tokenize().map_err(|e| e.to_string())?;
    let mut parser = Parser::new(tokens);
    let program = parser.parse_program().map_err(|e| e.to_string())?;
    let mut lowerer = Lowerer::new();
    let hir = lowerer.lower_program(&program).map_err(|e| e.to_string())?;

    let mut vm = Vm::new();
    vm.add_native("println".into(), |_vm, args| {
        for a in args { print!("{a}"); }
        println!();
        Ok(Value::Unit)
    });
    vm.add_native("Vec::new".into(), |_vm, _args| {
        Ok(Value::Vec(Rc::new(RefCell::new(Vec::new()))))
    });
    vm.add_native("zeros".into(), |_vm, args| {
        let shape: Vec<usize> = args.iter().map(|a| a.as_int().unwrap_or(1) as usize).collect();
        Ok(Value::Tensor(Rc::new(RefCell::new(Tensor::zeros(&shape)))))
    });
    vm.add_native("ones".into(), |_vm, args| {
        let shape: Vec<usize> = args.iter().map(|a| a.as_int().unwrap_or(1) as usize).collect();
        Ok(Value::Tensor(Rc::new(RefCell::new(Tensor::ones(&shape)))))
    });

    for func in &hir.functions {
        let compiler = BytecodeCompiler::new();
        match compiler.compile(func) {
            Ok((chunk, closures)) => {
                vm.add_fn(func.name.clone(), chunk);
                for (name, closure_chunk) in closures {
                    vm.add_fn(name, closure_chunk);
                }
                vm.set_global(func.name.clone(), Value::FnRef {
                    name: func.name.clone(),
                    params: func.params.clone(),
                    return_type: func.return_type.clone(),
                    captures: vec![],
                });
            }
            Err(e) => return Err(format!("compile error: {}", e)),
        }
    }

    if let Some(ref expr) = hir.main_expr {
        let compiler = BytecodeCompiler::new();
        match compiler.compile_main(expr) {
            Ok((chunk, closures)) => {
                vm.add_fn("main".into(), chunk);
                for (name, closure_chunk) in closures {
                    vm.add_fn(name, closure_chunk);
                }
            }
            Err(e) => return Err(format!("compile error: {}", e)),
        }
        let r = jit::run_jit(&mut vm, "main").map_err(|e| e.to_string())?;
        Ok((r, vm))
    } else if vm.has_fn("main") {
        let r = jit::run_jit(&mut vm, "main").map_err(|e| e.to_string())?;
        Ok((r, vm))
    } else {
        Ok((Value::Unit, vm))
    }
}

// ── M3.3 用例 8：合法状态链三路径（解释器/VM/JIT）一致 ──

#[test]
fn typestate_legal_chain_jit() {
    // LEGAL_CHAIN_SRC（read → close → reopen → read）JIT 路径与解释器/VM 一致。
    let result = run_jit(LEGAL_CHAIN_SRC).unwrap();
    expect_str(&result, "a.txt", "JIT");
}

// ── M3.3 用例 9：状态互不覆盖三路径一致（open().read()=1, closed().read()=2）──

#[test]
fn typestate_state_specialization_no_override_jit() {
    let result = run_jit(STATE_SPECIALIZATION_SRC).unwrap();
    expect_int(&result, 12, "JIT");
}

// ── M3.3 用例 10：字面量状态推断——File<Open> 与 File<Closed> 字面量各携其状态 ──

const LITERAL_STATE_INFERENCE_SRC: &str = r#"
enum Open {}
enum Closed {}
struct File<S> { path: str }
impl File<Open> {
    fn read(self) -> i64 { 1 }
}
impl File<Closed> {
    fn label(self) -> i64 { 2 }
}
File<Open> { path: "x" }.read() * 10 + File<Closed> { path: "y" }.label()
"#;

#[test]
fn typestate_literal_state_inference_interpreter() {
    // 字面量直接携带状态实参（G1）：File<Open>{...} → Open 专属 read；
    // File<Closed>{...} → Closed 专属 label（非共享）。结果 1*10+2 = 12。
    let result = run(LITERAL_STATE_INFERENCE_SRC).unwrap();
    expect_int(&result.unwrap(), 12, "解释器");
}

#[test]
fn typestate_literal_state_inference_vm() {
    let result = run_vm(LITERAL_STATE_INFERENCE_SRC).unwrap();
    expect_int(&result, 12, "VM");
}

#[test]
fn typestate_literal_state_inference_jit() {
    let result = run_jit(LITERAL_STATE_INFERENCE_SRC).unwrap();
    expect_int(&result, 12, "JIT");
}

// ── M3.3 用例 11：字面量非法状态编译期报错——File<Closed> 字面量调 Open 专属方法 ──

const LITERAL_ILLEGAL_STATE_SRC: &str = r#"
enum Open {}
enum Closed {}
struct File<S> { path: str }
impl File<Open> {
    fn read(self) -> i64 { 1 }
}
File<Closed> { path: "x" }.read()
"#;

#[test]
fn typestate_literal_illegal_state_compile_error() {
    // 字面量状态在 lowering 即被限定：File<Closed> 字面量上调用 Open 专属 read
    // → 编译期「没有方法 'read'」，三条路径（共享 lowerer）一致。
    let err = run(LITERAL_ILLEGAL_STATE_SRC).unwrap_err();
    assert!(
        err.contains("没有方法 'read'"),
        "解释器：应报「没有方法 'read'」，实际: {}",
        err
    );
    let err_vm = run_vm(LITERAL_ILLEGAL_STATE_SRC).unwrap_err();
    assert!(
        err_vm.contains("没有方法 'read'"),
        "VM：应报「没有方法 'read'」，实际: {}",
        err_vm
    );
    let err_jit = run_jit(LITERAL_ILLEGAL_STATE_SRC).unwrap_err();
    assert!(
        err_jit.contains("没有方法 'read'"),
        "JIT：应报「没有方法 'read'」，实际: {}",
        err_jit
    );
}

// ── M3.3 用例 12：覆盖断言——状态特化 mangled 方法真的被 JIT 编译（非静默回退）──

#[test]
fn typestate_jit_coverage_no_fallback() {
    // 状态特化的 mangled 方法（__File_Open_read / __File_Open_close /
    // __File_Closed_reopen）与 main 必须真的进入 JIT 编译集，而非整函数
    // 静默回退解释器——三路径一致由「同一字节码 + JIT 直译」保证，非巧合。
    let (result, vm) = run_jit_with_vm(LEGAL_CHAIN_SRC).unwrap();
    expect_str(&result, "a.txt", "JIT");
    let ctx = vm.jit_ctx.as_ref().expect("JIT 上下文应存在");
    for fname in ["main", "__File_Open_read", "__File_Open_close", "__File_Closed_reopen"] {
        let idx = vm.chunk_index_of(fname)
            .unwrap_or_else(|| panic!("chunk '{}' 应存在", fname));
        assert!(ctx.is_compiled(idx), "{} 应被 JIT 编译（非 fallback）", fname);
        assert!(!ctx.is_failed(idx), "{} 不应整函数 fallback", fname);
    }
}

// ── M3.3 用例 13：两状态同名方法 mangled 名互不覆盖（函数表级断言）──

#[test]
fn typestate_state_specialization_mangled_distinct() {
    // STATE_SPECIALIZATION_SRC 中 Open/Closed 各定义 read → 函数表应同时含
    // __File_Open_read 与 __File_Closed_read 两个独立 chunk（键分键注册的
    // 函数级证据，防「同名方法互相覆盖」回归）。
    let (result, vm) = run_jit_with_vm(STATE_SPECIALIZATION_SRC).unwrap();
    expect_int(&result, 12, "JIT");
    // 注册层：两状态同名方法各占独立 chunk（互不覆盖）。
    for fname in ["__File_Open_read", "__File_Closed_read"] {
        assert!(
            vm.chunk_index_of(fname).is_some(),
            "chunk '{}' 应存在（状态特化未互相覆盖）",
            fname
        );
    }
    // JIT 执行路径证据：main 被 JIT 编译（非整函数 fallback）。两个 read 的
    // 体是平凡标量 `{ 1 }`/`{ 2 }`（≤16 条白名单指令）→ 被 A2 内联进 main，
    // 不单独编译——内联同样是 JIT 执行，非解释器回退（三路径一致性由此保证）。
    let ctx = vm.jit_ctx.as_ref().expect("JIT 上下文应存在");
    let main_idx = vm.chunk_index_of("main").expect("main chunk 应存在");
    assert!(ctx.is_compiled(main_idx), "main 应被 JIT 编译");
    assert!(!ctx.is_failed(main_idx), "main 不应整函数 fallback");
}
