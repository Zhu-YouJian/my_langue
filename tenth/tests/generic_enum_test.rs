//! M2.1：泛型枚举显式 `<T>` 声明语法测试。
//!
//! 覆盖：
//! - 声明带 `<T>`（`enum X<T> { .. }`）
//! - 构造带显式类型实参（`MyEnum<i64>::Some(5)`）
//! - 变体字段类型替换（match 绑定变量类型为实例化后的具体类型）
//! - match 解构（元组/命名/单元变体）
//! - 嵌套泛型（`MyEnum<Vec<i64>>`）
//! - 与泛型函数交互（泛型函数参数/返回值带泛型枚举）
//! - 推断构造（`MyEnum::Some(5)`，无显式实参）
//! - 无泛型枚举回归（向后兼容）
//! - 泛型枚举 match 穷尽性检查
//! - 用户 `enum Option<T>` shadow 内置 Option
//! - VM/解释器 parity

use tenth::lexer::lexer::Lexer;
use tenth::parser::parser::Parser;
use tenth::hir::lower::Lowerer;
use tenth::runtime::interpreter::Interpreter;
use tenth::runtime::vm::Vm;
use tenth::runtime::value::Value;
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
    vm.add_native("assert".into(), |_vm, args| {
        if let Some(Value::Bool(b)) = args.first() {
            assert!(*b, "assertion failed");
        }
        Ok(Value::Unit)
    });
    vm.add_native("assert_eq".into(), |_vm, args| {
        if args.len() >= 2 {
            assert_eq!(format!("{}", args[0]), format!("{}", args[1]));
        }
        Ok(Value::Unit)
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

fn expect_int(v: Option<Value>, expected: i64, label: &str) {
    match v {
        Some(Value::Int(n, _)) => assert_eq!(n, expected, "{}: 期望 Int({}), 实际 {}", label, expected, n),
        other => panic!("{}: 期望 Int({}), 实际 {:?}", label, expected, other),
    }
}

// ── 1. 声明带 `<T>` + 构造带显式类型实参 + match 字段替换 ──

#[test]
fn generic_enum_declare_construct_explicit() {
    let src = r#"
        enum Box<T> { Value(T), Empty }
        let b = Box<i64>::Value(42);
        match b { Box::Value(v) => v, Box::Empty => 0 }
    "#;
    expect_int(run(src).unwrap(), 42, "显式实参构造");
}

// ── 2. 变体字段类型替换：match 绑定变量类型为实例化后的具体类型 ──
// v 必须是 i64（非 TypeParam("T")），否则 `v + 1` 无法推断/编译。

#[test]
fn generic_enum_field_type_substitution() {
    let src = r#"
        enum Box<T> { Value(T), Empty }
        let b = Box<i64>::Value(42);
        match b { Box::Value(v) => v + 1, Box::Empty => 0 }
    "#;
    expect_int(run(src).unwrap(), 43, "类型替换后算术");
}

// ── 3. 多类型参数 `<T, U>` + 元组变体解构 ──

#[test]
fn generic_enum_multi_param_tuple_variant() {
    let src = r#"
        enum Pair<T, U> { Make(T, U), None }
        let p = Pair<i64, str>::Make(10, "hi");
        match p { Pair::Make(a, b) => a, Pair::None => -1 }
    "#;
    expect_int(run(src).unwrap(), 10, "多参数元组解构");
}

// ── 4. 命名变体字段（构造 + 模式）──

#[test]
fn generic_enum_named_variant() {
    let src = r#"
        enum Holder<T> { Put(value: T), None }
        let h = Holder<f64>::Put(value: 3.5);
        match h { Holder::Put(value: v) => v * 2.0, Holder::None => 0.0 }
    "#;
    match run(src).unwrap() {
        Some(Value::Float(f)) => assert!((f - 7.0).abs() < 1e-10, "命名变体: 期望 7.0, 实际 {}", f),
        v => panic!("命名变体: 期望 Float(7.0), 实际 {:?}", v),
    }
}

// ── 5. 单元变体 + 类型注解 ──

#[test]
fn generic_enum_unit_variant() {
    let src = r#"
        enum Maybe<T> { Just(T), Nothing }
        let n = Maybe::Nothing;
        match n { Maybe::Just(x) => x, Maybe::Nothing => -1 }
    "#;
    expect_int(run(src).unwrap(), -1, "单元变体");
}

// ── 6. 嵌套泛型 `MyEnum<Vec<i64>>` ──
// 注意：解释器的 `Shared + Shared`（数组索引结果直接相加）是既有局限
// （VM 支持），此处用 `v.len()`（方法调用返回 Int）验证嵌套实参替换后
// v 是 Vec<i64>（若未替换，泛型参数 T 上无 len 方法，编译期/运行期会失败）。

#[test]
fn generic_enum_nested_generic() {
    let src = r#"
        enum Wrap<T> { Item(T), None }
        let w = Wrap<Vec<i64>>::Item([10, 20, 30]);
        match w { Wrap::Item(v) => v.len(), Wrap::None => -1 }
    "#;
    expect_int(run(src).unwrap(), 3, "嵌套泛型 + Vec 方法");
}

// ── 7. 泛型函数体内部构造泛型枚举（类型参数传递）──

#[test]
fn generic_enum_inside_generic_fn() {
    let src = r#"
        enum Opt<T> { Some(T), None }
        fn make<T>(v: T) -> Opt<T> { Opt::Some(v) }
        fn unwrap<T>(o: Opt<T>, fallback: T) -> T {
            match o { Opt::Some(v) => v, Opt::None => fallback }
        }
        let a = unwrap<i64>(make<i64>(5), 0);
        let b = unwrap<i64>(Opt::None, 7);
        a + b
    "#;
    expect_int(run(src).unwrap(), 12, "泛型函数 + 泛型枚举");
}

// ── 8. 非泛型函数接收泛型枚举参数 ──

#[test]
fn generic_enum_as_fn_param() {
    let src = r#"
        enum Opt<T> { Some(T), None }
        fn get_val(o: Opt<i64>) -> i64 {
            match o { Opt::Some(v) => v, Opt::None => -1 }
        }
        get_val(Opt::Some(42))
    "#;
    expect_int(run(src).unwrap(), 42, "泛型枚举作函数参数");
}

// ── 9. 推断构造（无显式实参）──

#[test]
fn generic_enum_inference_construction() {
    let src = r#"
        enum Maybe<T> { Just(T), Nothing }
        let m = Maybe::Just(3.5);
        match m { Maybe::Just(x) => x + 1.0, Maybe::Nothing => 0.0 }
    "#;
    match run(src).unwrap() {
        Some(Value::Float(f)) => assert!((f - 4.5).abs() < 1e-10, "推断构造: 期望 4.5, 实际 {}", f),
        v => panic!("推断构造: 期望 Float(4.5), 实际 {:?}", v),
    }
}

// ── 10. 无泛型枚举回归（向后兼容）──

#[test]
fn generic_enum_non_generic_regression() {
    let src = r#"
        enum Color { Red, Green, Blue }
        let c = Color::Red;
        match c { Color::Red => 1, Color::Green => 2, Color::Blue => 3 }
    "#;
    expect_int(run(src).unwrap(), 1, "无泛型枚举回归");
}

// ── 11. 泛型枚举 match 穷尽性检查仍工作 ──

#[test]
fn generic_enum_exhaustiveness_check() {
    let src = r#"
        enum Maybe<T> { Just(T), Nothing }
        let m = Maybe::Just(1);
        match m { Maybe::Just(x) => x }
    "#;
    let err = run(src).unwrap_err();
    assert!(err.contains("不穷尽") || err.contains("缺少变体"), "应报穷尽性错误, 实际: {}", err);
}

// ── 12. 用户 `enum Option<T>` shadow 内置 Option ──

#[test]
fn generic_enum_shadow_builtin_option() {
    let src = r#"
        enum Option<T> { Some(T), None }
        let x = Option::Some(5);
        match x { Option::Some(v) => v, Option::None => 0 }
    "#;
    expect_int(run(src).unwrap(), 5, "shadow 内置 Option");
}

// ── 13. VM / 解释器 parity ──

#[test]
fn generic_enum_vm_parity() {
    let src = r#"
        enum Opt<T> { Some(T), None }
        enum Pair<T, U> { Make(T, U), None }
        let a = match Opt::Some(21) { Opt::Some(v) => v, Opt::None => 0 };
        let p = Pair<i64, i64>::Make(a, 21);
        match p { Pair::Make(x, y) => x + y, Pair::None => 0 }
    "#;
    let interp = run(src).unwrap();
    expect_int(interp, 42, "解释器");
    let vm_result = run_vm(src).unwrap();
    match vm_result {
        Value::Int(n, _) => assert_eq!(n, 42, "VM: 期望 Int(42), 实际 {}", n),
        v => panic!("VM: 期望 Int(42), 实际 {:?}", v),
    }
}

// ── 14. 构造产物是 Value::Enum（enum_name/variant 正确）──

#[test]
fn generic_enum_value_shape() {
    let src = r#"
        enum Box<T> { Value(T), Empty }
        Box<str>::Value("hello")
    "#;
    match run(src).unwrap() {
        Some(Value::Enum { enum_name, variant, fields }) => {
            assert_eq!(enum_name, "Box");
            assert_eq!(variant, "Value");
            let fields = fields.borrow();
            assert_eq!(fields.len(), 1);
            match &fields[0] {
                (n, Value::String(s)) => {
                    assert_eq!(n, "_0");
                    assert_eq!(s, "hello");
                }
                other => panic!("期望 str 字段, 实际 {:?}", other),
            }
        }
        v => panic!("期望 Value::Enum, 实际 {:?}", v),
    }
}
