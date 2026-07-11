// f32 运行时专项测试 — Phase 3 Task 3.6
// 验证 VM 算术 f32 分支、MakeTensor dtype、混合提升、native 函数 f32 支持。

use tenth::hir::types::BaseType;
use tenth::lexer::lexer::Lexer;
use tenth::parser::parser::Parser;
use tenth::hir::lower::Lowerer;
use tenth::runtime::vm::Vm;
use tenth::runtime::value::Value;
use tenth::runtime::tensor::Tensor;
use tenth::compile::bytecode::BytecodeCompiler;
use std::rc::Rc;
use std::cell::RefCell;

// ── 1. VM 算术 f32 分支（直接测试 VM 公共算术方法）────────────

#[test]
fn test_vm_f32_arithmetic() {
    let mut vm = Vm::new();
    // Float32 + Float32 = Float32
    let r = vm.add(&Value::Float32(1.5), &Value::Float32(2.5)).unwrap();
    assert!(matches!(r, Value::Float32(x) if (x - 4.0).abs() < 1e-6), "Float32+Float32 应为 Float32");

    // Float32 * Float32 = Float32
    let r = vm.mul(&Value::Float32(3.0), &Value::Float32(4.0)).unwrap();
    assert!(matches!(r, Value::Float32(x) if (x - 12.0).abs() < 1e-6));

    // Float32 - Float32 = Float32
    let r = vm.sub(&Value::Float32(10.0), &Value::Float32(3.0)).unwrap();
    assert!(matches!(r, Value::Float32(x) if (x - 7.0).abs() < 1e-6));

    // Float32 / Float32 = Float32
    let r = vm.div(&Value::Float32(10.0), &Value::Float32(4.0)).unwrap();
    assert!(matches!(r, Value::Float32(x) if (x - 2.5).abs() < 1e-6));
}

#[test]
fn test_vm_f32_mixed_promote() {
    let mut vm = Vm::new();
    // Float32 + Float = Float（提升）
    let r = vm.add(&Value::Float32(1.5), &Value::Float(2.5)).unwrap();
    assert!(matches!(r, Value::Float(x) if (x - 4.0).abs() < 1e-10), "Float32+Float 应提升为 Float");

    // Float + Float32 = Float（提升）
    let r = vm.add(&Value::Float(1.5), &Value::Float32(2.5)).unwrap();
    assert!(matches!(r, Value::Float(x) if (x - 4.0).abs() < 1e-10));
}

#[test]
fn test_vm_int_float32_promote() {
    let mut vm = Vm::new();
    // Int + Float32 = Float32
    let r = vm.add(&Value::Int(3), &Value::Float32(2.5)).unwrap();
    assert!(matches!(r, Value::Float32(x) if (x - 5.5).abs() < 1e-6), "Int+Float32 应为 Float32");

    // Float32 + Int = Float32
    let r = vm.add(&Value::Float32(2.5), &Value::Int(3)).unwrap();
    assert!(matches!(r, Value::Float32(x) if (x - 5.5).abs() < 1e-6));
}

// ── 2. VM MakeTensor dtype 端到端 ──────────────────────────────

/// 编译并运行源码，返回 VM 栈顶值。注册常用 natives。
fn run_src(src: &str) -> Result<Value, String> {
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
    // f32/f64 转换 natives
    vm.add_native("to_float".into(), |_vm, args| {
        match args.first() {
            Some(Value::Int(n)) => Ok(Value::Float(*n as f64)),
            Some(Value::Float(f)) => Ok(Value::Float(*f)),
            Some(Value::Float32(f)) => Ok(Value::Float(*f as f64)),
            _ => Err(tenth::error::TenthError::RuntimeError { line: None, col: None, message: "to_float() 需要数值".into() }),
        }
    });
    vm.add_native("to_f64".into(), |_vm, args| {
        match args.first() {
            Some(Value::Int(n)) => Ok(Value::Float(*n as f64)),
            Some(Value::Float(f)) => Ok(Value::Float(*f)),
            Some(Value::Float32(f)) => Ok(Value::Float(*f as f64)),
            _ => Err(tenth::error::TenthError::RuntimeError { line: None, col: None, message: "to_f64() 需要数值".into() }),
        }
    });
    vm.add_native("to_f32".into(), |_vm, args| {
        match args.first() {
            Some(Value::Int(n)) => Ok(Value::Float32(*n as f32)),
            Some(Value::Float(f)) => Ok(Value::Float32(*f as f32)),
            Some(Value::Float32(f)) => Ok(Value::Float32(*f)),
            _ => Err(tenth::error::TenthError::RuntimeError { line: None, col: None, message: "to_f32() 需要数值".into() }),
        }
    });
    // randn_f32 native
    vm.add_native("randn_f32".into(), |_vm, args| {
        let rows = match args.first() { Some(Value::Int(n)) => *n as usize, _ => 1 };
        let cols = match args.get(1) { Some(Value::Int(n)) => *n as usize, _ => 1 };
        let t = Tensor::randn_f32(&[rows, cols]);
        Ok(Value::Tensor(Rc::new(RefCell::new(t))))
    });
    // abs / sqrt natives
    vm.add_native("abs".into(), |_vm, args| {
        match args.first() {
            Some(Value::Int(n)) => Ok(Value::Int(n.abs())),
            Some(Value::Float(f)) => Ok(Value::Float(f.abs())),
            Some(Value::Float32(f)) => Ok(Value::Float32(f.abs())),
            _ => Err(tenth::error::TenthError::RuntimeError { line: None, col: None, message: "abs() 需要数值".into() }),
        }
    });
    vm.add_native("sqrt".into(), |_vm, args| {
        match args.first() {
            Some(Value::Float(f)) => Ok(Value::Float(f.sqrt())),
            Some(Value::Float32(f)) => Ok(Value::Float32(f.sqrt())),
            Some(Value::Int(n)) => Ok(Value::Float((*n as f64).sqrt())),
            _ => Err(tenth::error::TenthError::RuntimeError { line: None, col: None, message: "sqrt() 需要数值".into() }),
        }
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

#[test]
fn test_vm_f32_make_tensor() {
    // 使用 f32 字面量构造张量（顶层表达式）
    let src = "[[1.0f32, 2.0f32], [3.0f32, 4.0f32]]";
    let val = run_src(src).unwrap();
    match val {
        Value::Tensor(t) => {
            let t = t.borrow();
            assert!(t.is_f32(), "f32 字面量构造的张量应为 F32 dtype");
            assert_eq!(t.shape(), vec![2, 2]);
            assert_eq!(t.get(&[0, 0]), Some(1.0));
            assert_eq!(t.get(&[1, 1]), Some(4.0));
        }
        other => panic!("期望 Tensor，得到 {:?}", other),
    }
}

// ── 3. Native 函数 f32 支持 ────────────────────────────────────

#[test]
fn test_native_randn_f32() {
    // randn_f32 应返回 f32 Tensor
    let src = "randn_f32(2, 3)";
    let val = run_src(src).unwrap();
    match val {
        Value::Tensor(t) => {
            let t = t.borrow();
            assert!(t.is_f32(), "randn_f32 应返回 f32 Tensor");
            assert_eq!(t.dtype(), BaseType::F32);
            assert_eq!(t.shape(), vec![2, 3]);
        }
        other => panic!("期望 Tensor，得到 {:?}", other),
    }
}

#[test]
fn test_native_math_f32() {
    // to_f32(2.0) 应返回 Float32（dtype 保持）
    let src = "to_f32(2.0)";
    let val = run_src(src).unwrap();
    assert!(matches!(val, Value::Float32(x) if (x - 2.0).abs() < 1e-6), "to_f32(2.0) 应返回 Float32");
}

#[test]
fn test_native_sqrt_f32() {
    // sqrt(f32) 应返回 Float32（dtype 保持）
    let src = "sqrt(16.0f32)";
    let val = run_src(src).unwrap();
    assert!(matches!(val, Value::Float32(x) if (x - 4.0).abs() < 1e-6), "sqrt(16.0f32) 应返回 Float32");
}
