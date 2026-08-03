//! 阶段2a G6 测试：调用点参数类型检查。
//!
//! 目标：函数调用时校验实参与形参类型兼容，不匹配编译期报错（带行/列）。
//! 审计缺口：`scope.rs::resolve_fn_overload` 在单签名时不校验实参类型，
//! 类型不符的程序能编译通过、到运行时才炸或静默错误。
//!
//! 覆盖：
//! - 基本类型匹配（i64/str/bool）正确代码全过
//! - 基本类型不匹配（str 传 i64、bool 传 i64 等）编译期 TypeError
//! - 泛型实例化调用：`identity<i64>("x")` 拦截、正确实例化放行
//! - typestate 状态实参不匹配：`File<Closed>` 传 `File<Open>` 形参 → 编译期报错
//!   （"非法状态表达不出来"的最终保障）
//! - 隐式转换边界：数值类型（i32/i64/f32/f64/整数↔浮点）互相兼容放行
//!   （与运行时值语义一致：Value::Int/Float 算术多态）
//! - 防误报：未标注参数/裸 Vec/内置泛型/Option-Result/重载/方法调用全部保持通过

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

/// 仅 lowering（不执行）：用于验证编译期检查不误报但运行时可能未支持的
/// 语法（如变参/默认参数的运行时调用）。
fn lower_only(src: &str) -> Result<(), String> {
    let mut lexer = Lexer::new(src);
    let tokens = lexer.tokenize().map_err(|e| e.to_string())?;
    let mut parser = Parser::new(tokens);
    let program = parser.parse_program().map_err(|e| e.to_string())?;
    let mut lowerer = Lowerer::new();
    lowerer.lower_program(&program).map(|_| ()).map_err(|e| e.to_string())
}

// ── 1. 基本类型匹配：正确代码全过（含 i32 字面量 → i64 形参的隐式转换）──

const BASIC_MATCH_SRC: &str = r#"
fn add(a: i64, b: i64) -> i64 { a + b }
fn concat(a: str, b: str) -> str { a + b }
fn negate(x: bool) -> bool { !x }
fn scale(x: f64) -> f64 { x * 2.0 }
add(1, 2) + 10
"#;

#[test]
fn basic_match_ok() {
    let result = run(BASIC_MATCH_SRC).unwrap();
    expect_int(&result.unwrap(), 13, "解释器");
}

// ── 2. 基本类型不匹配：编译期 TypeError ──

const STR_TO_INT_MISMATCH_SRC: &str = r#"
fn double(n: i64) -> i64 { n * 2 }
double("hello")
"#;

#[test]
fn str_to_int_mismatch_compile_error() {
    let err = run(STR_TO_INT_MISMATCH_SRC).unwrap_err();
    assert!(
        err.contains("类型不兼容") && err.contains("double"),
        "应报「类型不兼容」，实际: {}",
        err
    );
}

const BOOL_TO_INT_MISMATCH_SRC: &str = r#"
fn double(n: i64) -> i64 { n * 2 }
double(true)
"#;

#[test]
fn bool_to_int_mismatch_compile_error() {
    let err = run(BOOL_TO_INT_MISMATCH_SRC).unwrap_err();
    assert!(
        err.contains("类型不兼容") && err.contains("double"),
        "应报「类型不兼容」，实际: {}",
        err
    );
}

const INT_TO_STR_MISMATCH_SRC: &str = r#"
fn greet(s: str) -> str { s }
greet(42)
"#;

#[test]
fn int_to_str_mismatch_compile_error() {
    let err = run(INT_TO_STR_MISMATCH_SRC).unwrap_err();
    assert!(
        err.contains("类型不兼容") && err.contains("greet"),
        "应报「类型不兼容」，实际: {}",
        err
    );
}

const SECOND_ARG_MISMATCH_SRC: &str = r#"
fn pick(a: i64, b: str) -> str { b }
pick(1, 2)
"#;

#[test]
fn second_arg_mismatch_compile_error() {
    let err = run(SECOND_ARG_MISMATCH_SRC).unwrap_err();
    assert!(
        err.contains("第 2 个实参") && err.contains("pick"),
        "应报第 2 个实参不兼容，实际: {}",
        err
    );
}

// ── 3. 泛型实例化调用：正确放行、类型不符拦截 ──

const GENERIC_CALL_OK_SRC: &str = r#"
fn identity<T>(x: T) -> T { x }
fn pair<T, U>(a: T, b: U) -> T { a }
let s = pair<str, i64>("x", 1);
identity<i64>(7) + 1
"#;

#[test]
fn generic_instantiation_ok() {
    // 正确泛型实例化（单/双类型参数）编译并运行通过。
    let result = run(GENERIC_CALL_OK_SRC).unwrap();
    expect_int(&result.unwrap(), 8, "解释器");
}

const GENERIC_CALL_MISMATCH_SRC: &str = r#"
fn identity<T>(x: T) -> T { x }
identity<i64>("hello")
"#;

#[test]
fn generic_instantiation_mismatch_compile_error() {
    let err = run(GENERIC_CALL_MISMATCH_SRC).unwrap_err();
    assert!(
        err.contains("类型不兼容"),
        "identity<i64>(\"hello\") 应报类型不兼容，实际: {}",
        err
    );
}

// ── 4. typestate 状态实参：File<Closed> 传 File<Open> 形参 → 编译期报错 ──

const TYPESTATE_OK_SRC: &str = r#"
enum Open {}
enum Closed {}
struct File<S> { path: str }
fn is_open(f: File<Open>) -> bool { true }
is_open(File<Open> { path: "a.txt" })
"#;

#[test]
fn typestate_state_arg_ok() {
    let result = run(TYPESTATE_OK_SRC).unwrap();
    match result.unwrap() {
        Value::Bool(true) => {}
        other => panic!("期望 Bool(true)，实际 {:?}", other),
    }
}

const TYPESTATE_WRONG_STATE_SRC: &str = r#"
enum Open {}
enum Closed {}
struct File<S> { path: str }
fn is_open(f: File<Open>) -> bool { true }
is_open(File<Closed> { path: "a.txt" })
"#;

#[test]
fn typestate_wrong_state_arg_compile_error() {
    // G6 核心：非法状态"表达不出来"——状态不匹配在编译期拦截，
    // 而不是到运行时才炸或静默错误。
    let err = run(TYPESTATE_WRONG_STATE_SRC).unwrap_err();
    assert!(
        err.contains("类型不兼容") && err.contains("is_open"),
        "File<Closed> 传 File<Open> 形参应编译期报错，实际: {}",
        err
    );
}

const TYPESTATE_VAR_WRONG_STATE_SRC: &str = r#"
enum Open {}
enum Closed {}
struct File<S> { path: str }
fn is_open(f: File<Open>) -> bool { true }
let c = File<Closed> { path: "x" };
is_open(c)
"#;

#[test]
fn typestate_wrong_state_var_compile_error() {
    let err = run(TYPESTATE_VAR_WRONG_STATE_SRC).unwrap_err();
    assert!(
        err.contains("类型不兼容"),
        "变量传错状态应编译期报错，实际: {}",
        err
    );
}

// ── 5. 隐式转换边界：数值类型互相兼容（运行时值语义宽松）──

const NUMERIC_CONVERSION_OK_SRC: &str = r#"
fn take_i64(x: i64) -> i64 { x }
fn take_i32(x: i32) -> i32 { x }
fn take_f64(x: f64) -> f64 { x }
fn take_f32(x: f32) -> f32 { x }
fn take_float(x: f64) -> f64 { x * 2.0 }
take_i64(1) + take_i32(2i64) + take_f64(3) + take_f32(4.5) + take_float(2)
"#;

#[test]
fn numeric_conversion_boundaries_ok() {
    // i32 字面量→i64 形参、i64 实参→i32 形参、int→float 形参全部放行：
    // 运行时 Int/Float 算术多态（Int + Float → Float），与既有行为一致。
    // 结果：1 + 2 + 3 + 4.5 + 4.0 = 14.5（Float）
    let result = run(NUMERIC_CONVERSION_OK_SRC).unwrap();
    match result.unwrap() {
        Value::Float(v) => assert!((v - 14.5).abs() < 1e-9, "期望 Float(14.5)，实际 {}", v),
        other => panic!("期望 Float(14.5)，实际 {:?}", other),
    }
}

// ── 6. 防误报：未标注参数 / 裸 Vec / 内置泛型 / 重载 / 方法调用全过 ──

const UNTYPED_PARAM_OK_SRC: &str = r#"
fn show(x) -> i64 { 1 }
fn show_str(x: str) -> i64 { 1 }
show(42) + show("hi") + show(true) + show_str("s")
"#;

#[test]
fn untyped_param_ok() {
    // 未标注参数（TypeParam("Unknown")）接受任意类型，防误报。
    let result = run(UNTYPED_PARAM_OK_SRC).unwrap();
    expect_int(&result.unwrap(), 4, "解释器");
}

const BARE_VEC_PARAM_OK_SRC: &str = r#"
fn mean(items: Vec) -> f64 { 1.0 }
mean([1, 2, 3])
"#;

#[test]
fn bare_vec_param_ok() {
    // 裸 `Vec` 形参（未声明用户类型）接受数组实参，防误报。
    let result = run(BARE_VEC_PARAM_OK_SRC).unwrap();
    match result.unwrap() {
        Value::Float(v) => assert!((v - 1.0).abs() < 1e-10, "期望 1.0，实际 {}", v),
        other => panic!("期望 Float(1.0)，实际 {:?}", other),
    }
}

const STRUCT_PARAM_OK_SRC: &str = r#"
struct HashSet { inner: i64 }
fn new() -> HashSet { HashSet { inner: 0 } }
fn insert(set: HashSet, value) -> HashSet { set }
fn len(set: HashSet) -> i64 { set.inner }
let s = new();
len(insert(s, 5)) + len(insert(s, "x"))
"#;

#[test]
fn struct_and_untyped_param_ok() {
    // struct 实参与 TypeParam 形参归一放行；未标注 value 形参接受任意类型。
    let result = run(STRUCT_PARAM_OK_SRC).unwrap();
    expect_int(&result.unwrap(), 0, "解释器");
}

const OPTION_RESULT_PARAM_OK_SRC: &str = r#"
fn take_opt(x: Option<i64>) -> i64 { 1 }
fn take_res(x: Result<i64, str>) -> i64 { 1 }
take_opt(Option::Some(5)) + take_res(Result::Ok(7))
"#;

#[test]
fn option_result_param_ok() {
    // Option/Result 内置泛型参数（base 为名义 Enum，但实参类型参数 Unknown 通配）放行。
    let result = run(OPTION_RESULT_PARAM_OK_SRC).unwrap();
    expect_int(&result.unwrap(), 2, "解释器");
}

const TENSOR_PARAM_OK_SRC: &str = r#"
fn scale(t: Tensor[f64, ..], s: f64) -> Tensor[f64, ..] { t }
let x = zeros(3, 4);
scale(x, 2.0)
"#;

#[test]
fn tensor_wildcard_param_ok() {
    // `Tensor[f64, ..]`（任意秩通配）接受具体 shape 的张量实参。
    let result = run(TENSOR_PARAM_OK_SRC).unwrap();
    assert!(matches!(result.unwrap(), Value::Tensor(_)), "应返回 Tensor");
}

const TENSOR_SYMBOL_PARAM_OK_SRC: &str = r#"
fn dot(a: Tensor[f64, M, K], b: Tensor[f64, K, N]) -> Tensor[f64, M, K] { a }
let x = zeros(3, 4);
let y = zeros(4, 5);
dot(x, y)
"#;

#[test]
fn tensor_symbol_dim_param_ok() {
    // 符号维度形参（S_q/D_k 等）接受 Known 维度实参（Known vs Symbol 放行）。
    let result = run(TENSOR_SYMBOL_PARAM_OK_SRC).unwrap();
    assert!(matches!(result.unwrap(), Value::Tensor(_)), "应返回 Tensor");
}

const OVERLOAD_OK_SRC: &str = r#"
fn f(x: i64) -> i64 { x }
fn f(x: str) -> str { "s" }
f(5i64) + 1
"#;

#[test]
fn overload_resolution_unchanged() {
    // 多重载解析逻辑不变（G6 只对单签名做严格检查）。
    let result = run(OVERLOAD_OK_SRC).unwrap();
    expect_int(&result.unwrap(), 6, "解释器");
}

const METHOD_CALL_OK_SRC: &str = r#"
struct Point { x: i64, y: i64 }
impl Point {
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
fn method_call_ok() {
    let result = run(METHOD_CALL_OK_SRC).unwrap();
    expect_int(&result.unwrap(), 46, "解释器");
}

const METHOD_CALL_MISMATCH_SRC: &str = r#"
struct Point { x: i64, y: i64 }
impl Point {
    fn add(self, other: Point) -> Point {
        Point { x: self.x + other.x, y: self.y + other.y }
    }
}
let p = Point { x: 1, y: 2 };
p.add(42)
"#;

#[test]
fn method_call_mismatch_compile_error() {
    let err = run(METHOD_CALL_MISMATCH_SRC).unwrap_err();
    assert!(
        err.contains("类型不兼容"),
        "方法实参类型不符应编译期报错，实际: {}",
        err
    );
}

const VARIADIC_AND_DEFAULT_OK_SRC: &str = r#"
fn vari(x: i64, ...rest) -> i64 { x }
fn with_default(a: i64, b: i64 = 10) -> i64 { a + b }
vari(1, 2, 3, "x") + with_default(5) + with_default(5, 20)
"#;

#[test]
fn variadic_and_default_params_no_false_positive() {
    // 变参（多余实参不检查）与默认参数（缺失实参不检查）在编译期不误报。
    // 运行时对 param_defaults/param_variadic 尚未处理（仅编译期解析），
    // 故此处只验证 lowering 通过，不执行。
    lower_only(VARIADIC_AND_DEFAULT_OK_SRC).expect("变参/默认参数调用应编译通过（G6 不误报）");
}

const TYPESTATE_METHOD_REGRESSION_SRC: &str = r#"
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
c.reopen().read()
"#;

#[test]
fn typestate_method_chain_regression() {
    // typestate 方法链不受 G6 影响（self 形参 TypeParam("Self") 自动放行）。
    let result = run(TYPESTATE_METHOD_REGRESSION_SRC).unwrap();
    match result.unwrap() {
        Value::String(s) => assert_eq!(s, "a.txt"),
        other => panic!("期望 String(\"a.txt\")，实际 {:?}", other),
    }
}

const VM_OK_SRC: &str = r#"
fn add(a: i64, b: i64) -> i64 { a + b }
add(1, 2)
"#;

#[test]
fn vm_path_ok() {
    let result = run_vm(VM_OK_SRC).unwrap();
    expect_int(&result, 3, "VM");
}

const VM_MISMATCH_SRC: &str = r#"
fn add(a: i64, b: i64) -> i64 { a + b }
add(1, "x")
"#;

#[test]
fn vm_path_mismatch_compile_error() {
    // 报错发生在 lowering 阶段（两条路径共享同一 lowerer），VM 入口同样拦截。
    let err = run_vm(VM_MISMATCH_SRC).unwrap_err();
    assert!(
        err.contains("类型不兼容"),
        "VM 路径应编译期报错，实际: {}",
        err
    );
}

// ── 7. M3.2：多重载兼容回退也校验实参类型（G6 缺口修复）──

const OVERLOAD_COMPAT_MISMATCH_SRC: &str = r#"
fn g(x: i64, y: i64) -> i64 { x + y }
fn g(x: str) -> str { "s" }
g(1, "x")
"#;

#[test]
fn overload_compat_fallback_mismatch_compile_error() {
    // M3.2 修复：多重载兼容回退（参数数量唯一匹配 (i64,i64)，但第 2 实参
    // "x" 确定不兼容）也走 types_compatible 校验。此前该路径不检查，
    // `g(1, "x")` 编译通过且运行时 VM（HashMap 后注册覆盖）/解释器（取
    // 第一条同名）两路径行为不一致——现在编译期报「类型不兼容」，双路径统一。
    let err = run(OVERLOAD_COMPAT_MISMATCH_SRC).unwrap_err();
    assert!(
        err.contains("类型不兼容") && err.contains("g"),
        "多重载兼容回退应报类型不兼容，实际: {}",
        err
    );
}

const OVERLOAD_COMPAT_MISMATCH_VM_SRC: &str = r#"
fn g(x: i64, y: i64) -> i64 { x + y }
fn g(x: str) -> str { "s" }
g(1, "x")
"#;

#[test]
fn overload_compat_fallback_mismatch_vm_compile_error() {
    // 报错在 lowering 阶段，VM 入口同样拦截（与单签名场景一致）。
    let err = run_vm(OVERLOAD_COMPAT_MISMATCH_VM_SRC).unwrap_err();
    assert!(
        err.contains("类型不兼容"),
        "VM 路径多重载兼容回退应编译期报错，实际: {}",
        err
    );
}

// ── 8. M3.2：参数数量不同的重载按实参数量选签名（配套修复）──

const OVERLOAD_DIFFERENT_ARITY_OK_SRC: &str = r#"
fn g(x: i64, y: i64) -> i64 { x + y }
fn g(x: str) -> str { "s" }
let a = g(1, 2);
let b = g("hi");
a + 1
"#;

#[test]
fn overload_different_arity_compile_ok() {
    // M3.2 配套修复：process_call_args 此前用「第一个同名签名」处理
    // 默认/变参/命名参数，导致参数数量不同的重载（`g("hi")` 应匹配 1 参
    // g(str)）误报「缺少必需参数」。现按实参数量选签名，编译期放行。
    // （运行时重载分派 VM/解释器不一致为既有缺陷，另行登记；此处只验证
    // 编译期不再误报。）
    lower_only(OVERLOAD_DIFFERENT_ARITY_OK_SRC)
        .expect("参数数量不同的重载调用应编译通过（G6 不误报）");
}
