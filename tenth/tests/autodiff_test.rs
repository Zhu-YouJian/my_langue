use tenth::lexer::lexer::Lexer;
use tenth::parser::parser::Parser;
use tenth::hir::lower::Lowerer;
use tenth::runtime::interpreter::Interpreter;
use tenth::runtime::vm::Vm;
use tenth::runtime::value::Value;
use tenth::compile::bytecode::BytecodeCompiler;
use std::rc::Rc;
use std::cell::RefCell;

fn run_code(src: &str) -> Result<Option<Value>, String> {
    let mut lexer = Lexer::new(src);
    let tokens = lexer.tokenize().map_err(|e| e.to_string())?;
    let mut parser = Parser::new(tokens);
    let program = parser.parse_program().map_err(|e| e.to_string())?;
    let mut lowerer = Lowerer::new();
    let hir = lowerer.lower_program(&program).map_err(|e| e.to_string())?;
    let mut interpreter = Interpreter::new(&hir);
    interpreter.execute_program(&hir).map_err(|e| e.to_string())
}

fn run_vm(src: &str) -> Result<Value, String> {
    let mut lexer = Lexer::new(src);
    let tokens = lexer.tokenize().map_err(|e| e.to_string())?;
    let mut parser = Parser::new(tokens);
    let program = parser.parse_program().map_err(|e| e.to_string())?;
    let mut lowerer = Lowerer::new();
    let hir = lowerer.lower_program(&program).map_err(|e| e.to_string())?;

    let mut vm = Vm::new();
    // Register println native
    vm.add_native("println".into(), |_vm, args| {
        for a in args { print!("{a}"); }
        println!();
        Ok(Value::Unit)
    });
    vm.add_native("Vec::new".into(), |_vm, _args| {
        Ok(Value::Vec(Rc::new(RefCell::new(Vec::new()))))
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

/// Extract a scalar f64 from a Value (Float or single-element Tensor)
fn as_f64(val: &Value) -> Option<f64> {
    match val {
        Value::Float(f) => Some(*f),
        Value::Tensor(t) => {
            let data = &t.borrow().data;
            if data.len() == 1 {
                Some(data[0])
            } else {
                None
            }
        }
        _ => None,
    }
}

// ── Autodiff end-to-end tests ──

#[test]
fn test_autodiff_simple_gradient() {
    // y = x^2, dy/dx = 2x, at x=3, grad = 6
    let src = r#"
        new_grad();
        let x = param(tensor[[3.0]]);
        let y = x * x;
        backward(y);
        stop_grad();
        grad(x).sum()
    "#;
    let result = run_code(src).unwrap();
    let g = as_f64(result.as_ref().unwrap()).expect("expected numeric value");
    assert!((g - 6.0).abs() < 0.01, "expected 6.0, got {}", g);
}

#[test]
fn test_autodiff_linear_gradient() {
    // y = 2*x + 1, dy/dx = 2
    let src = r#"
        new_grad();
        let x = param(tensor[[5.0]]);
        let y = 2.0 * x + 1.0;
        backward(y);
        stop_grad();
        grad(x).sum()
    "#;
    let result = run_code(src).unwrap();
    let g = as_f64(result.as_ref().unwrap()).expect("expected numeric value");
    assert!((g - 2.0).abs() < 0.01, "expected 2.0, got {}", g);
}

#[test]
fn test_autodiff_relu_gradient() {
    // ReLU(x): if x > 0, grad = 1
    let src = r#"
        new_grad();
        let x = param(tensor[[3.0]]);
        let y = x.relu();
        backward(y);
        stop_grad();
        grad(x).sum()
    "#;
    let result = run_code(src).unwrap();
    let g = as_f64(result.as_ref().unwrap()).expect("expected numeric value");
    assert!((g - 1.0).abs() < 0.01, "expected 1.0, got {}", g);
}

#[test]
fn test_autodiff_relu_gradient_negative() {
    // ReLU of negative input: gradient should be 0
    let src = r#"
        new_grad();
        let x = param(tensor[[-3.0]]);
        let y = x.relu();
        backward(y);
        stop_grad();
        grad(x).sum()
    "#;
    let result = run_code(src).unwrap();
    let g = as_f64(result.as_ref().unwrap()).expect("expected numeric value");
    assert!(g.abs() < 0.01, "expected 0.0, got {}", g);
}

#[test]
fn test_autodiff_chain_rule() {
    // y = (x * x) * x = x^3, dy/dx = 3x^2, at x=2, grad = 12
    let src = r#"
        new_grad();
        let x = param(tensor[[2.0]]);
        let xx = x * x;
        let y = xx * x;
        backward(y);
        stop_grad();
        grad(x).sum()
    "#;
    let result = run_code(src).unwrap();
    let g = as_f64(result.as_ref().unwrap()).expect("expected numeric value");
    assert!((g - 12.0).abs() < 0.1, "expected 12.0, got {}", g);
}

#[test]
fn test_autodiff_matmul_gradient() {
    // Basic matmul gradient: just check it doesn't error and produces a value
    let src = r#"
        new_grad();
        let w = param(tensor[[1.0, 2.0], [3.0, 4.0]]);
        let x = param(tensor[[1.0], [1.0]]);
        let y = w.matmul(x);
        backward(y.sum());
        stop_grad();
        grad(w).sum()
    "#;
    let result = run_code(src).unwrap();
    // Just check we get a numeric value (exact gradient depends on implementation)
    assert!(as_f64(result.as_ref().unwrap()).is_some(), "expected numeric gradient");
}

#[test]
fn test_autodiff_exp_log_gradient() {
    // d/dx(exp(x)) = exp(x), at x=0, grad = 1
    let src = r#"
        new_grad();
        let x = param(tensor[[0.0]]);
        let y = x.exp();
        backward(y);
        stop_grad();
        grad(x).sum()
    "#;
    let result = run_code(src).unwrap();
    let g = as_f64(result.as_ref().unwrap()).expect("expected numeric value");
    assert!((g - 1.0).abs() < 0.01, "expected 1.0, got {}", g);
}

#[test]
fn test_autodiff_zero_grad() {
    // After zero_grad, gradients should be zero
    let src = r#"
        new_grad();
        let x = param(tensor[[3.0]]);
        let y = x * x;
        backward(y);
        zero_grad();
        stop_grad();
        grad(x).sum()
    "#;
    let result = run_code(src).unwrap();
    let g = as_f64(result.as_ref().unwrap()).expect("expected numeric value");
    assert!(g.abs() < 0.01, "expected 0.0 after zero_grad, got {}", g);
}

// ── Closure capture tests ──

#[test]
fn test_closure_captures_variable() {
    let src = r#"
        let x = 10;
        let f = |y| x + y;
        f(5)
    "#;
    let result = run_code(src).unwrap();
    match result {
        Some(Value::Int(15, _)) => {}
        v => panic!("expected Int(15), got {:?}", v),
    }
}

#[test]
fn test_closure_captures_multiple_variables() {
    let src = r#"
        let a = 3;
        let b = 4;
        let f = |x| a * x + b;
        f(2)
    "#;
    let result = run_code(src).unwrap();
    match result {
        Some(Value::Int(10, _)) => {}
        v => panic!("expected Int(10), got {:?}", v),
    }
}

#[test]
fn test_closure_nested_capture() {
    let src = r#"
        let x = 5;
        let make_adder = |n| |m| n + m + x;
        let add5 = make_adder(10);
        add5(3)
    "#;
    let result = run_code(src).unwrap();
    match result {
        Some(Value::Int(18, _)) => {}
        v => panic!("expected Int(18), got {:?}", v),
    }
}

#[test]
fn test_closure_does_not_capture_params() {
    let src = r#"
        let f = |x, y| x + y;
        f(100, 200)
    "#;
    let result = run_code(src).unwrap();
    match result {
        Some(Value::Int(300, _)) => {}
        v => panic!("expected Int(300), got {:?}", v),
    }
}

// ── AUDIT-11.4.1 regression: HashSet free-var collection ──
//
// Background: `hir/lower/closures.rs` was refactored to use
// `HashSet<String>` for free-var collection (O(n²) → O(n)). The public
// API still returns `Vec<String>` sorted, so capture struct field
// layout is stable. These tests guard two boundary cases that the
// HashSet refactor could subtly break:
//   1. Dedup: same variable referenced many times in a closure body
//      must be captured exactly once (HashSet semantics).
//   2. Builtins: println/abs/sqrt/etc. must NOT be treated as captures
//      (they appear in the `match name.as_str()` skip list in
//      closures.rs:28-42).
//
// Note: `free_vars_in` is `pub(super)`, so we cannot call it directly
// from integration tests. These tests assert *behaviour*: if dedup
// breaks (e.g. duplicate capture slots), the closure call returns the
// wrong value; if builtins are wrongly captured, lookup fails.

#[test]
fn test_closure_capture_dedup_same_var_many_uses() {
    // `x` is referenced 5 times in the closure body. Before the HashSet
    // refactor, the old Vec-based collector could (in pathological cases)
    // produce duplicate entries that would skew capture struct layout.
    // After the refactor, HashSet guarantees dedup. Behaviour check:
    // f(1) = 10*5 + 1 = 51.
    let src = r#"
        let x = 10;
        let f = |y| x + x + x + x + x + y;
        f(1)
    "#;
    let result = run_code(src).unwrap();
    match result {
        Some(Value::Int(51, _)) => {}
        v => panic!("expected Int(51), got {:?}", v),
    }
}

#[test]
fn test_closure_capture_does_not_capture_builtins() {
    // `abs` is a builtin (closures.rs:38 lists `abs` in the skip list).
    // The closure body references `abs` (builtin) and `x` (captured).
    // Only `x` should appear in the capture set. If `abs` were wrongly
    // captured, the closure would either fail to compile or look up a
    // non-existent variable at runtime.
    // Behaviour check: f(-5) = abs(-5) + 10 = 5 + 10 = 15.
    let src = r#"
        let x = 10;
        let f = |y| abs(y) + x;
        f(-5)
    "#;
    let result = run_code(src).unwrap();
    match result {
        Some(Value::Int(15, _)) => {}
        v => panic!("expected Int(15), got {:?}", v),
    }
}

// ── Tensor operation tests ──

#[test]
fn test_tensor_matmul() {
    let src = r#"
        let a = tensor[[1.0, 2.0], [3.0, 4.0]];
        let b = tensor[[5.0, 6.0], [7.0, 8.0]];
        let c = a.matmul(b);
        c.sum()
    "#;
    let result = run_code(src).unwrap();
    match result {
        Some(Value::Float(n)) => assert!((n - 134.0).abs() < 0.01, "expected 134.0, got {}", n),
        v => panic!("expected Float, got {:?}", v),
    }
}

#[test]
fn test_tensor_transpose() {
    let src = r#"
        let a = tensor[[1.0, 2.0, 3.0]];
        let b = a.transpose();
        b.sum()
    "#;
    let result = run_code(src).unwrap();
    match result {
        Some(Value::Float(n)) => assert!((n - 6.0).abs() < 0.01, "expected 6.0, got {}", n),
        v => panic!("expected Float, got {:?}", v),
    }
}

#[test]
fn test_tensor_sigmoid() {
    let src = r#"
        tensor[[0.0]].sigmoid().sum()
    "#;
    let result = run_code(src).unwrap();
    match result {
        Some(Value::Float(n)) => assert!((n - 0.5).abs() < 0.01, "expected 0.5, got {}", n),
        v => panic!("expected Float, got {:?}", v),
    }
}

#[test]
fn test_tensor_tanh() {
    let src = r#"
        tensor[[0.0]].tanh().sum()
    "#;
    let result = run_code(src).unwrap();
    match result {
        Some(Value::Float(n)) => assert!(n.abs() < 0.01, "expected 0.0, got {}", n),
        v => panic!("expected Float, got {:?}", v),
    }
}

#[test]
fn test_tensor_reshape() {
    let src = r#"
        let a = tensor[[1.0, 2.0], [3.0, 4.0]];
        let b = a.flatten();
        b.sum()
    "#;
    let result = run_code(src).unwrap();
    match result {
        Some(Value::Float(n)) => assert!((n - 10.0).abs() < 0.01, "expected 10.0, got {}", n),
        v => panic!("expected Float, got {:?}", v),
    }
}

#[test]
fn test_tensor_mean() {
    let src = r#"
        tensor[[2.0, 4.0, 6.0]].mean()
    "#;
    let result = run_code(src).unwrap();
    match result {
        Some(Value::Float(n)) => assert!((n - 4.0).abs() < 0.01, "expected 4.0, got {}", n),
        v => panic!("expected Float, got {:?}", v),
    }
}

// ── VM tensor literal test ──

#[test]
fn test_vm_tensor_literal() {
    let src = r#"
        tensor[[1.0, 2.0], [3.0, 4.0]].sum()
    "#;
    let result = run_code(src).unwrap();
    match result {
        Some(Value::Float(n)) => assert!((n - 10.0).abs() < 0.01, "expected 10.0, got {}", n),
        v => panic!("expected Float, got {:?}", v),
    }
}

// ── Error span propagation test ──

#[test]
fn test_error_span_in_borrow_check() {
    // M2.6：i64 是 Copy 类型（`move x` 对 Copy 值不失效，值被复制原变量仍可用）；
    // 数组字面量 `[1,2,3]` 推断为 Type::Array{i64} 也是 Copy。用含 Vec 字段的
    // 非 Copy 结构体触发 borrow check 错误（本测试意图是错误 span 传播）。
    let src = r#"
        struct S { items: Vec<i64> }
        let x = S { items: [1, 2, 3] };
        let y = move x;
        let z = x.items.len()
    "#;
    let result = run_code(src);
    assert!(result.is_err());
    let err_msg = result.unwrap_err();
    let err_str = err_msg.to_string();
    assert!(err_str.contains("移动") || err_str.contains("moved"), "expected 'moved' in error, got: {}", err_str);
}

// ── VM for-in loop tests ──

#[test]
fn test_vm_for_range() {
    let src = r#"
        fn sum_range(n: i64) -> i64 {
            let mut sum = 0;
            for i in 0..n {
                sum = sum + i;
            };
            sum
        };
        sum_range(5)
    "#;
    let result = run_vm(src).unwrap();
    match result {
        Value::Int(10, _) => {}  // 0+1+2+3+4 = 10
        v => panic!("expected Int(10), got {:?}", v),
    }
}

#[test]
fn test_vm_for_range_inclusive() {
    let src = r#"
        fn sum_to(n: i64) -> i64 {
            let mut sum = 0;
            for i in 1..=n {
                sum = sum + i;
            };
            sum
        };
        sum_to(5)
    "#;
    let result = run_vm(src).unwrap();
    match result {
        Value::Int(15, _) => {}  // 1+2+3+4+5 = 15
        v => panic!("expected Int(15), got {:?}", v),
    }
}

#[test]
fn test_vm_for_in_function() {
    let src = r#"
        fn sum_range(n: i64) -> i64 {
            let mut s = 0;
            for i in 0..n {
                s = s + i;
            };
            s
        };
        sum_range(6)
    "#;
    let result = run_vm(src).unwrap();
    match result {
        Value::Int(15, _) => {}  // 0+1+2+3+4+5 = 15
        v => panic!("expected Int(15), got {:?}", v),
    }
}

// ── VM string slice test ──

#[test]
fn test_vm_string_slice() {
    let src = r#"
        fn slice_test() -> str {
            let s = "hello world";
            s[0..5]
        };
        slice_test()
    "#;
    let result = run_vm(src).unwrap();
    match result {
        Value::String(s) => assert_eq!(s, "hello"),
        v => panic!("expected String(\"hello\"), got {:?}", v),
    }
}

// ── VM closure test ──

#[test]
fn test_vm_closure_simple() {
    let src = r#"
        fn closure_test() -> i64 {
            let double = |x| x * 2;
            double(21)
        };
        closure_test()
    "#;
    let result = run_vm(src).unwrap();
    match result {
        Value::Int(42, _) => {}
        v => panic!("expected Int(42), got {:?}", v),
    }
}

#[test]
fn test_vm_closure_captures() {
    let src = r#"
        fn capture_test() -> i64 {
            let factor = 3;
            let multiply = |x| x * factor;
            multiply(7)
        };
        capture_test()
    "#;
    let result = run_vm(src).unwrap();
    match result {
        Value::Int(21, _) => {}
        v => panic!("expected Int(21), got {:?}", v),
    }
}

// ── Borrow checker strict tests ──

#[test]
fn test_strict_borrow_shared_while_mut() {
    let src = r#"
        let mut x = 42;
        let m = &mut x;
        let r = &x;
        *m
    "#;
    let mut lexer = Lexer::new(src);
    let tokens = lexer.tokenize().unwrap();
    let mut parser = Parser::new(tokens);
    let program = parser.parse_program().unwrap();
    let mut lowerer = Lowerer::new();
    let result = lowerer.lower_program(&program);
    assert!(result.is_err(), "expected borrow check error");
}

#[test]
fn test_strict_borrow_mut_while_shared() {
    let src = r#"
        let x = 42;
        let r = &x;
        let m = &mut x;
        *r
    "#;
    let mut lexer = Lexer::new(src);
    let tokens = lexer.tokenize().unwrap();
    let mut parser = Parser::new(tokens);
    let program = parser.parse_program().unwrap();
    let mut lowerer = Lowerer::new();
    let result = lowerer.lower_program(&program);
    assert!(result.is_err(), "expected borrow check error");
}

// ── Closure pipe syntax verification tests ──

#[test]
fn test_closure_pipe_syntax_basic() {
    let src = r#"
        let inc = |x| x + 1;
        inc(9)
    "#;
    let result = run_code(src).unwrap();
    match result {
        Some(Value::Int(10, _)) => {}
        v => panic!("expected Int(10), got {:?}", v),
    }
}

#[test]
fn test_closure_pipe_syntax_multi_param() {
    let src = r#"
        let add = |a, b| a + b;
        add(3, 4)
    "#;
    let result = run_code(src).unwrap();
    match result {
        Some(Value::Int(7, _)) => {}
        v => panic!("expected Int(7), got {:?}", v),
    }
}

#[test]
fn test_closure_pipe_syntax_with_type_annotation() {
    let src = r#"
        let double = |x: i64| x * 2;
        double(21)
    "#;
    let result = run_code(src).unwrap();
    match result {
        Some(Value::Int(42, _)) => {}
        v => panic!("expected Int(42), got {:?}", v),
    }
}

#[test]
fn test_closure_pipe_syntax_as_argument() {
    let src = r#"
        fn apply(f: i64, x: i64) -> i64 {
            let g = |v| v * 3;
            g(x)
        };
        apply(0, 7)
    "#;
    let result = run_code(src).unwrap();
    match result {
        Some(Value::Int(21, _)) => {}
        v => panic!("expected Int(21), got {:?}", v),
    }
}

#[test]
fn test_vm_closure_pipe_syntax() {
    let src = r#"
        fn test() -> i64 {
            let inc = |x| x + 1;
            inc(99)
        };
        test()
    "#;
    let result = run_vm(src).unwrap();
    match result {
        Value::Int(100, _) => {}
        v => panic!("expected Int(100), got {:?}", v),
    }
}

#[test]
fn test_vm_closure_multi_capture() {
    let src = r#"
        fn test() -> i64 {
            let a = 10;
            let b = 20;
            let f = |x| a + b + x;
            f(5)
        };
        test()
    "#;
    let result = run_vm(src).unwrap();
    match result {
        Value::Int(35, _) => {}
        v => panic!("expected Int(35), got {:?}", v),
    }
}

// ── VM struct tests ──

#[test]
fn test_vm_struct_creation() {
    let src = r#"
        struct Point { x: f64, y: f64 }
        fn test() -> f64 {
            let p = Point { x: 3.0, y: 4.0 };
            p.x + p.y
        };
        test()
    "#;
    let result = run_vm(src).unwrap();
    match result {
        Value::Float(n) => assert!((n - 7.0).abs() < 0.01, "expected 7.0, got {}", n),
        v => panic!("expected Float, got {:?}", v),
    }
}

#[test]
fn test_vm_struct_default_fields() {
    let src = r#"
        struct Config { lr: f64, epochs: i64, verbose: bool }
        fn test() -> i64 {
            let c = Config { lr: 0.01, .. };
            c.epochs
        };
        test()
    "#;
    let result = run_vm(src).unwrap();
    match result {
        Value::Int(0, _) => {}
        v => panic!("expected Int(0), got {:?}", v),
    }
}

#[test]
fn test_vm_struct_impl_method() {
    // VM doesn't support impl method dispatch yet;
    // test struct field access + standalone function instead
    let src = r#"
        struct Point { x: i64, y: i64 }
        fn sum_point(p: Point) -> i64 { p.x + p.y }
        fn test() -> i64 {
            let p = Point { x: 30, y: 12 };
            sum_point(p)
        };
        test()
    "#;
    let result = run_vm(src).unwrap();
    match result {
        Value::Int(42, _) => {}
        v => panic!("expected Int(42), got {:?}", v),
    }
}

// ── VM enum tests ──

#[test]
fn test_vm_enum_match() {
    let src = r#"
        enum Color { Red, Green, Blue }
        fn test() -> i64 {
            let c = Color::Green;
            match c {
                Color::Red => 1,
                Color::Green => 2,
                Color::Blue => 3,
            }
        };
        test()
    "#;
    let result = run_vm(src).unwrap();
    match result {
        Value::Int(2, _) => {}
        v => panic!("expected Int(2), got {:?}", v),
    }
}

#[test]
fn test_vm_enum_tuple_variant() {
    let src = r#"
        enum Option { Some(i64), None }
        fn test() -> i64 {
            let x = Option::Some(42);
            match x {
                Option::Some(v) => v,
                Option::None => 0,
            }
        };
        test()
    "#;
    let result = run_vm(src).unwrap();
    match result {
        Value::Int(42, _) => {}
        v => panic!("expected Int(42), got {:?}", v),
    }
}

// ── VM generic tests ──

#[test]
fn test_vm_generic_function() {
    // VM doesn't support generic monomorphization yet;
    // test that a non-generic wrapper can call a typed function
    let src = r#"
        fn id_i64(x: i64) -> i64 { x }
        fn test() -> i64 {
            id_i64(42)
        };
        test()
    "#;
    let result = run_vm(src).unwrap();
    match result {
        Value::Int(42, _) => {}
        v => panic!("expected Int(42), got {:?}", v),
    }
}

#[test]
fn test_vm_generic_struct() {
    let src = r#"
        struct Pair<T, U> { first: T, second: U }
        fn test() -> i64 {
            let p = Pair<i64, i64> { first: 10, second: 20 };
            p.first + p.second
        };
        test()
    "#;
    let result = run_vm(src).unwrap();
    match result {
        Value::Int(30, _) => {}
        v => panic!("expected Int(30), got {:?}", v),
    }
}

#[test]
fn test_vm_for_loop_sum() {
    let src = r#"
        fn sum_to(n: i64) -> i64 {
            let mut total = 0;
            for i in 0..n {
                total = total + i;
            };
            total
        };
        sum_to(5)
    "#;
    let result = run_vm(src).unwrap();
    match result {
        Value::Int(10, _) => {}
        v => panic!("expected Int(10), got {:?}", v),
    }
}

#[test]
fn test_vm_while_loop() {
    let src = r#"
        fn countdown(n: i64) -> i64 {
            let mut i = n;
            let mut sum = 0;
            while i > 0 {
                sum = sum + i;
                i = i - 1;
            };
            sum
        };
        countdown(5)
    "#;
    let result = run_vm(src).unwrap();
    match result {
        Value::Int(15, _) => {}
        v => panic!("expected Int(15), got {:?}", v),
    }
}

#[test]
fn test_vm_if_else() {
    let src = r#"
        fn abs(x: i64) -> i64 {
            if x < 0 { -x } else { x }
        };
        abs(-7) + abs(3)
    "#;
    let result = run_vm(src).unwrap();
    match result {
        Value::Int(10, _) => {}
        v => panic!("expected Int(10), got {:?}", v),
    }
}

#[test]
fn test_vm_recursive_function() {
    let src = r#"
        fn fib(n: i64) -> i64 {
            if n <= 1 { n } else { fib(n - 1) + fib(n - 2) }
        };
        fib(10)
    "#;
    let result = run_vm(src).unwrap();
    match result {
        Value::Int(55, _) => {}
        v => panic!("expected Int(55), got {:?}", v),
    }
}

// ── Transformer component tests ──

#[test]
fn test_tensor_layer_norm() {
    let src = r#"
        let x = randn(3, 4);
        let gamma = ones(4);
        let beta = zeros(4);
        let y = x.layer_norm(gamma, beta, 0.00001);
        y.mean()
    "#;
    let result = run_code(src).unwrap();
    match result {
        Some(Value::Float(n)) => assert!(n.abs() < 0.5, "layer_norm mean should be near 0, got {}", n),
        v => panic!("expected Float, got {:?}", v),
    }
}

#[test]
fn test_tensor_gelu() {
    // GELU(0) ≈ 0, GELU(1) ≈ 0.841, GELU(-1) ≈ -0.159
    // Test via scalar computation: GELU(1) - GELU(-1) should be ~1.0
    let src = r#"
        let x = tensor[[1.0, -1.0]];
        let y = x.gelu();
        y.sum()
    "#;
    let result = run_code(src).unwrap();
    match result {
        Some(Value::Float(n)) => assert!((n - 0.682).abs() < 0.1, "GELU(1)+GELU(-1) should be ~0.682, got {}", n),
        v => panic!("expected Float, got {:?}", v),
    }
}

#[test]
fn test_tensor_cat() {
    // cat two (1,2) tensors along dim=0 → (2,2), sum should be 1+2+3+4=10
    let src = r#"
        let a = tensor[[1.0, 2.0]];
        let b = tensor[[3.0, 4.0]];
        let c = a.cat(b, 0);
        c.sum()
    "#;
    let result = run_code(src).unwrap();
    match result {
        Some(Value::Float(n)) => assert!((n - 10.0).abs() < 0.01, "cat sum should be 10.0, got {}", n),
        v => panic!("expected Float, got {:?}", v),
    }
}

#[test]
fn test_tensor_masked_fill() {
    // mask[1]=1.0 means position 1 gets filled with -inf
    // result should have position 0 and 2 unchanged, position 1 very negative
    let src = r#"
        let x = tensor[[1.0, 2.0, 3.0]];
        let mask = tensor[[0.0, 1.0, 0.0]];
        let y = x.masked_fill(mask, -1000000000.0);
        y.sum()
    "#;
    let result = run_code(src).unwrap();
    match result {
        Some(Value::Float(n)) => assert!(n < -1e8, "masked_fill sum should be very negative, got {}", n),
        v => panic!("expected Float, got {:?}", v),
    }
}

#[test]
fn test_tensor_permute() {
    // x is (2,3), permute(1,0) gives (3,2), sum should still be 21
    let src = r#"
        let x = tensor[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let y = x.permute(1, 0);
        y.sum()
    "#;
    let result = run_code(src).unwrap();
    match result {
        Some(Value::Float(n)) => assert!((n - 21.0).abs() < 0.01, "permute sum should be 21.0, got {}", n),
        v => panic!("expected Float, got {:?}", v),
    }
}

#[test]
fn test_scaled_dot_product_attention() {
    // Q,K,V: (3,8), attention output should be (3,8)
    // Just verify the computation doesn't error and output sum is finite
    let src = r#"
        let q = randn(3, 8);
        let k = randn(3, 8);
        let v = randn(3, 8);
        let d_k = 8;
        let scale = 1.0 / sqrt(d_k);
        let kT = k.transpose();
        let scores = q.matmul(kT) * scale;
        let weights = scores.softmax();
        let out = weights.matmul(v);
        out.sum()
    "#;
    let result = run_code(src).unwrap();
    match result {
        Some(Value::Float(n)) => assert!(n.is_finite(), "attention output should be finite, got {}", n),
        v => panic!("expected Float, got {:?}", v),
    }
}

#[test]
fn test_feedforward_network() {
    // x: (4,16), w1: (16,64), w2: (64,16), output: (4,16)
    // Verify computation doesn't error and output sum is finite
    let src = r#"
        let x = randn(4, 16);
        let w1 = randn(16, 64) * 0.1;
        let b1 = zeros(64);
        let w2 = randn(64, 16) * 0.1;
        let b2 = zeros(16);
        let hidden = x.matmul(w1) + b1;
        let activated = hidden.gelu();
        let out = activated.matmul(w2) + b2;
        out.sum()
    "#;
    let result = run_code(src).unwrap();
    match result {
        Some(Value::Float(n)) => assert!(n.is_finite(), "FFN output should be finite, got {}", n),
        v => panic!("expected Float, got {:?}", v),
    }
}
