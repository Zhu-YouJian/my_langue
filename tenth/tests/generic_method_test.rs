//! G5 修复测试（阶段2a M2）：Generic receiver 方法解析 + 解释器/VM 一致性。
//!
//! 背景：审计确认——在泛型类型值上调用方法（如 `make() -> File<Open>` 的返回值，
//! 或带类型参数的 struct 实例），解释器路径按运行时值名查方法表可用，而 VM 路径
//! 依赖编译期改写（`__Type_method`），Generic receiver 不被改写则报「没有方法」。
//!
//! 修复：`lower_expr.rs` 的 `recv_type_name` 匹配 `Type::Generic` 并从 base 取类型名，
//! 使 Generic receiver 的 inherent 方法调用同样触发编译期改写，两条路径行为一致。
//!
//! 覆盖：
//! - 泛型 struct 返回值上调用方法（解释器 + VM 双路径）
//! - 带类型参数的 struct 实例方法（含参数）
//! - `Tensor[f64, M, K]` 符号维度类型上调用 matmul（双路径）
//! - 方法链 / 多方法
//! - 回归：普通 struct 方法、字面量 Generic 方法（既有行为不破坏）

use tenth::lexer::lexer::Lexer;
use tenth::parser::parser::Parser;
use tenth::hir::lower::Lowerer;
use tenth::runtime::interpreter::Interpreter;
use tenth::runtime::vm::Vm;
use tenth::runtime::value::Value;
use tenth::runtime::tensor::Tensor;
use tenth::compile::bytecode::BytecodeCompiler;
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
/// 注册测试所需 native（zeros/ones/tensor 等，复制自 main.rs 惯例）。
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

fn expect_int(v: &Value, expected: i64, label: &str) {
    match v {
        Value::Int(n, _) => assert_eq!(*n, expected, "{}: 期望 Int({}), 实际 {}", label, expected, n),
        other => panic!("{}: 期望 Int({}), 实际 {:?}", label, expected, other),
    }
}

// ── 用例 1：泛型 struct 返回值（Generic receiver）上调用方法，双路径 ──

const GENERIC_METHOD_SRC: &str = r#"
struct File<S> { path: str }
impl File {
    fn read(self) -> i64 { 1 }
}
fn make() -> File<Open> { File<Open> { path: "x" } }
make().read()
"#;

#[test]
fn generic_receiver_method_interpreter() {
    let result = run(GENERIC_METHOD_SRC).unwrap();
    expect_int(&result.unwrap(), 1, "解释器");
}

#[test]
fn generic_receiver_method_vm() {
    // 修复前：VM 报「没有方法 'read'」；修复后与解释器一致返回 1。
    let result = run_vm(GENERIC_METHOD_SRC).unwrap();
    expect_int(&result, 1, "VM");
}

// ── 用例 2：带类型参数的 struct 实例方法（含参数）──

const GENERIC_METHOD_ARGS_SRC: &str = r#"
struct Counter<T> { value: T }
impl Counter {
    fn inc(self, by: i64) -> i64 { self.value + by }
}
fn make() -> Counter<i64> { Counter<i64> { value: 10 } }
make().inc(5)
"#;

#[test]
fn generic_receiver_method_with_args_interpreter() {
    let result = run(GENERIC_METHOD_ARGS_SRC).unwrap();
    expect_int(&result.unwrap(), 15, "解释器");
}

#[test]
fn generic_receiver_method_with_args_vm() {
    let result = run_vm(GENERIC_METHOD_ARGS_SRC).unwrap();
    expect_int(&result, 15, "VM");
}

// ── 用例 3：Generic receiver 返回非标量（str 字段访问）──

const GENERIC_METHOD_FIELD_SRC: &str = r#"
struct File<S> { path: str }
impl File {
    fn path_of(self) -> str { self.path }
}
fn make() -> File<Open> { File<Open> { path: "hello" } }
make().path_of()
"#;

#[test]
fn generic_receiver_method_returning_str_interpreter() {
    let result = run(GENERIC_METHOD_FIELD_SRC).unwrap();
    match result.unwrap() {
        Value::String(s) => assert_eq!(s, "hello"),
        other => panic!("期望 String(\"hello\")，实际 {:?}", other),
    }
}

#[test]
fn generic_receiver_method_returning_str_vm() {
    let result = run_vm(GENERIC_METHOD_FIELD_SRC).unwrap();
    match result {
        Value::String(s) => assert_eq!(s, "hello"),
        other => panic!("期望 String(\"hello\")，实际 {:?}", other),
    }
}

// ── 用例 4：`Tensor[f64, M, K]` 符号维度类型上调用 matmul，双路径 ──

const TENSOR_SYMBOL_MATMUL_SRC: &str = r#"
fn make() -> Tensor[f64, M, K] { zeros(3, 4) }
fn caller() -> Tensor[f64, ..] {
    let a = make();
    a.matmul(zeros(4, 5))
}
caller()
"#;

fn assert_tensor_shape(val: &Value, expected: &[usize], label: &str) {
    match val {
        Value::Tensor(t) => {
            let shape = t.borrow().shape();
            assert_eq!(&shape, expected, "{}: shape 不匹配", label);
        }
        other => panic!("{}: 期望 Tensor，实际 {:?}", label, other),
    }
}

#[test]
fn tensor_symbol_dims_matmul_interpreter() {
    let result = run(TENSOR_SYMBOL_MATMUL_SRC).unwrap();
    assert_tensor_shape(&result.unwrap(), &[3, 5], "解释器");
}

#[test]
fn tensor_symbol_dims_matmul_vm() {
    let result = run_vm(TENSOR_SYMBOL_MATMUL_SRC).unwrap();
    assert_tensor_shape(&result, &[3, 5], "VM");
}

// ── 用例 5：Generic receiver 方法链（多方法）──

const GENERIC_METHOD_CHAIN_SRC: &str = r#"
struct File<S> { path: str }
impl File {
    fn read(self) -> i64 { 1 }
    fn close(self) -> i64 { 0 }
}
fn make() -> File<Open> { File<Open> { path: "x" } }
make().close() + make().read()
"#;

#[test]
fn generic_receiver_method_chain_interpreter() {
    let result = run(GENERIC_METHOD_CHAIN_SRC).unwrap();
    expect_int(&result.unwrap(), 1, "解释器");
}

#[test]
fn generic_receiver_method_chain_vm() {
    let result = run_vm(GENERIC_METHOD_CHAIN_SRC).unwrap();
    expect_int(&result, 1, "VM");
}

// ── 回归：普通 struct 方法（非 Generic）不破坏 ──

const PLAIN_METHOD_SRC: &str = r#"
struct Point { x: i64 }
impl Point {
    fn get(self) -> i64 { self.x }
}
fn make() -> Point { Point { x: 7 } }
make().get()
"#;

#[test]
fn plain_struct_method_interpreter_regression() {
    let result = run(PLAIN_METHOD_SRC).unwrap();
    expect_int(&result.unwrap(), 7, "解释器");
}

#[test]
fn plain_struct_method_vm_regression() {
    let result = run_vm(PLAIN_METHOD_SRC).unwrap();
    expect_int(&result, 7, "VM");
}

// ── 回归：字面量 Generic 实例方法（修复前已可用，不应破坏）──

const LITERAL_GENERIC_SRC: &str = r#"
struct File<S> { path: str }
impl File {
    fn read(self) -> i64 { 1 }
}
File<Open> { path: "x" }.read()
"#;

#[test]
fn literal_generic_instance_method_interpreter_regression() {
    let result = run(LITERAL_GENERIC_SRC).unwrap();
    expect_int(&result.unwrap(), 1, "解释器");
}

#[test]
fn literal_generic_instance_method_vm_regression() {
    let result = run_vm(LITERAL_GENERIC_SRC).unwrap();
    expect_int(&result, 1, "VM");
}
