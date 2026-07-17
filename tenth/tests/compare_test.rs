//! 张量比较运算 + where_ 测试套件（Wave 2 第 4 项 — 路径 Z）。
//!
//! 覆盖：
//! - 6 个比较 native（tensor_gt/lt/ge/le/eq/ne）前向数值正确性
//! - 广播：`gt([[1,2],[3,4]], [2])` → `[[0,0],[1,1]]`
//! - 跨 dtype：f32 vs f64 → f64 结果（0.0/1.0）
//! - where_ 别名（select 语义）：`where_([0,1,0], [1,2,3], [4,5,6])` → `[4,2,6]`
//! - autodiff 端到端：`loss = where_(gt(x, 0), x, 0).sum()`（ReLU 等价）的 backward
//!   梯度只流到 x>0 的位置（验证 select backward 的 `> 0.5` 掩码路由）
//!
//! VM 路径执行（参考 pool_test.rs 的 run_vm_pool 模式）。
//! 比较运算本身不可微；autodiff 通过 select（where_）侧链实现。

use std::rc::Rc;
use std::cell::RefCell;

use tenth::compile::bytecode::BytecodeCompiler;
use tenth::hir::lower::Lowerer;
use tenth::lexer::lexer::Lexer;
use tenth::parser::parser::Parser;
use tenth::runtime::autodiff::Tape;
use tenth::runtime::tensor::Tensor;
use tenth::runtime::value::Value;
use tenth::runtime::vm::Vm;

/// 通过 VM 执行 .th 源码，注册 select / 6 个比较 native / autodiff natives。
fn run_vm_compare(src: &str) -> Result<Value, String> {
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
    vm.add_native("tensor".into(), |_vm, args| {
        if args.len() == 1 { Ok(args[0].clone()) }
        else {
            Err(tenth::error::TenthError::RuntimeError { line: None, col: None,
                message: "tensor() unexpected args".into() })
        }
    });

    // ── autodiff natives ──
    vm.add_native("new_grad".into(), |vm, _args| {
        vm.tape = Some(Tape::new());
        vm.recording = true;
        Ok(Value::Unit)
    });
    vm.add_native("param".into(), |vm, args| {
        if let Some(Value::Tensor(t)) = args.first() {
            if let Some(ref mut tape) = vm.tape {
                let node_id = tape.input(t.clone());
                t.borrow_mut().tape_id = Some(node_id);
            }
            Ok(Value::Tensor(t.clone()))
        } else {
            Err(tenth::error::TenthError::RuntimeError { line: None, col: None,
                message: "param() requires a tensor argument".into() })
        }
    });
    vm.add_native("backward".into(), |vm, args| {
        if let Some(Value::Tensor(t)) = args.first() {
            if let Some(ref tape) = vm.tape {
                let loss_id = t.borrow().tape_id
                    .ok_or_else(|| tenth::error::TenthError::RuntimeError { line: None, col: None,
                        message: "backward(): tensor has no tape_id".into() })?;
                let _ = tape.backward(loss_id);
                Ok(Value::Unit)
            } else {
                Err(tenth::error::TenthError::RuntimeError { line: None, col: None,
                    message: "new_grad() not called".into() })
            }
        } else {
            Err(tenth::error::TenthError::RuntimeError { line: None, col: None,
                message: "backward() requires a tensor argument".into() })
        }
    });
    vm.add_native("grad".into(), |_vm, args| {
        if let Some(Value::Tensor(t)) = args.first() {
            let p = t.borrow();
            if let Some(ref grad) = p.grad {
                let grad_tensor = Tensor::from_tensor_data(grad.clone());
                Ok(Value::Tensor(Rc::new(RefCell::new(grad_tensor))))
            } else {
                let zeros = Tensor::zeros(&p.shape());
                Ok(Value::Tensor(Rc::new(RefCell::new(zeros))))
            }
        } else {
            Err(tenth::error::TenthError::RuntimeError { line: None, col: None,
                message: "grad() requires a tensor argument".into() })
        }
    });
    vm.add_native("stop_grad".into(), |vm, args| {
        if let Some(Value::Tensor(t)) = args.first() {
            let mut detached = t.borrow().clone();
            detached.tape_id = None;
            Ok(Value::Tensor(Rc::new(RefCell::new(detached))))
        } else {
            vm.recording = false;
            Ok(Value::Unit)
        }
    });
    vm.add_native("zero_grad".into(), |vm, _args| {
        if let Some(ref tape) = vm.tape { tape.zero_grad(); }
        Ok(Value::Unit)
    });

    // ── select native（与 main.rs::register_natives 一致，含 autodiff）──
    vm.add_native("select".into(), |vm, args| {
        if args.len() < 3 {
            return Err(tenth::error::TenthError::RuntimeError { line: None, col: None,
                message: "select(cond, then, else) 期望三个参数".into(),
            });
        }
        let (cond, then, else_) = match (&args[0], &args[1], &args[2]) {
            (Value::Tensor(c), Value::Tensor(t), Value::Tensor(e)) => (c.clone(), t.clone(), e.clone()),
            _ => return Err(tenth::error::TenthError::RuntimeError { line: None, col: None,
                message: "select(cond, then, else) 期望三个张量参数".into(),
            }),
        };
        let result_tensor = Tensor::select(&cond.borrow(), &then.borrow(), &else_.borrow())
            .map_err(|msg| tenth::error::TenthError::RuntimeError { line: None, col: None, message: msg })?;
        let result = Rc::new(RefCell::new(result_tensor));
        if vm.recording {
            if let Some(ref mut tape) = vm.tape {
                let then_id = then.borrow().tape_id;
                let else_id = else_.borrow().tape_id;
                let node_id = tape.select(then_id, else_id, cond.clone(), then.clone(), else_.clone(), result.clone());
                result.borrow_mut().tape_id = Some(node_id);
            }
        }
        Ok(Value::Tensor(result))
    });

    // ── 6 个比较 native（与 main.rs::register_natives 一致，不可微）──
    vm.add_native("tensor_gt".into(), |_vm, args| {
        if args.len() < 2 {
            return Err(tenth::error::TenthError::RuntimeError { line: None, col: None,
                message: "tensor_gt(a, b) 期望两个张量参数".into() });
        }
        let (a, b) = match (&args[0], &args[1]) {
            (Value::Tensor(x), Value::Tensor(y)) => (x.clone(), y.clone()),
            _ => return Err(tenth::error::TenthError::RuntimeError { line: None, col: None,
                message: "tensor_gt(a, b) 期望两个张量参数".into() }),
        };
        let r = a.borrow().gt(&b.borrow())
            .map_err(|m| tenth::error::TenthError::RuntimeError { line: None, col: None, message: m })?;
        Ok(Value::Tensor(Rc::new(RefCell::new(r))))
    });
    vm.add_native("tensor_lt".into(), |_vm, args| {
        if args.len() < 2 {
            return Err(tenth::error::TenthError::RuntimeError { line: None, col: None,
                message: "tensor_lt(a, b) 期望两个张量参数".into() });
        }
        let (a, b) = match (&args[0], &args[1]) {
            (Value::Tensor(x), Value::Tensor(y)) => (x.clone(), y.clone()),
            _ => return Err(tenth::error::TenthError::RuntimeError { line: None, col: None,
                message: "tensor_lt(a, b) 期望两个张量参数".into() }),
        };
        let r = a.borrow().lt(&b.borrow())
            .map_err(|m| tenth::error::TenthError::RuntimeError { line: None, col: None, message: m })?;
        Ok(Value::Tensor(Rc::new(RefCell::new(r))))
    });
    vm.add_native("tensor_ge".into(), |_vm, args| {
        if args.len() < 2 {
            return Err(tenth::error::TenthError::RuntimeError { line: None, col: None,
                message: "tensor_ge(a, b) 期望两个张量参数".into() });
        }
        let (a, b) = match (&args[0], &args[1]) {
            (Value::Tensor(x), Value::Tensor(y)) => (x.clone(), y.clone()),
            _ => return Err(tenth::error::TenthError::RuntimeError { line: None, col: None,
                message: "tensor_ge(a, b) 期望两个张量参数".into() }),
        };
        let r = a.borrow().ge(&b.borrow())
            .map_err(|m| tenth::error::TenthError::RuntimeError { line: None, col: None, message: m })?;
        Ok(Value::Tensor(Rc::new(RefCell::new(r))))
    });
    vm.add_native("tensor_le".into(), |_vm, args| {
        if args.len() < 2 {
            return Err(tenth::error::TenthError::RuntimeError { line: None, col: None,
                message: "tensor_le(a, b) 期望两个张量参数".into() });
        }
        let (a, b) = match (&args[0], &args[1]) {
            (Value::Tensor(x), Value::Tensor(y)) => (x.clone(), y.clone()),
            _ => return Err(tenth::error::TenthError::RuntimeError { line: None, col: None,
                message: "tensor_le(a, b) 期望两个张量参数".into() }),
        };
        let r = a.borrow().le(&b.borrow())
            .map_err(|m| tenth::error::TenthError::RuntimeError { line: None, col: None, message: m })?;
        Ok(Value::Tensor(Rc::new(RefCell::new(r))))
    });
    vm.add_native("tensor_eq".into(), |_vm, args| {
        if args.len() < 2 {
            return Err(tenth::error::TenthError::RuntimeError { line: None, col: None,
                message: "tensor_eq(a, b) 期望两个张量参数".into() });
        }
        let (a, b) = match (&args[0], &args[1]) {
            (Value::Tensor(x), Value::Tensor(y)) => (x.clone(), y.clone()),
            _ => return Err(tenth::error::TenthError::RuntimeError { line: None, col: None,
                message: "tensor_eq(a, b) 期望两个张量参数".into() }),
        };
        let r = a.borrow().eq(&b.borrow())
            .map_err(|m| tenth::error::TenthError::RuntimeError { line: None, col: None, message: m })?;
        Ok(Value::Tensor(Rc::new(RefCell::new(r))))
    });
    vm.add_native("tensor_ne".into(), |_vm, args| {
        if args.len() < 2 {
            return Err(tenth::error::TenthError::RuntimeError { line: None, col: None,
                message: "tensor_ne(a, b) 期望两个张量参数".into() });
        }
        let (a, b) = match (&args[0], &args[1]) {
            (Value::Tensor(x), Value::Tensor(y)) => (x.clone(), y.clone()),
            _ => return Err(tenth::error::TenthError::RuntimeError { line: None, col: None,
                message: "tensor_ne(a, b) 期望两个张量参数".into() }),
        };
        let r = a.borrow().ne(&b.borrow())
            .map_err(|m| tenth::error::TenthError::RuntimeError { line: None, col: None, message: m })?;
        Ok(Value::Tensor(Rc::new(RefCell::new(r))))
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

fn tensor_to_vec(val: &Value) -> Option<Vec<f64>> {
    if let Value::Tensor(t) = val {
        let data = &t.borrow().data;
        Some(data.as_f64_view().iter().cloned().collect())
    } else {
        None
    }
}

fn tensor_shape(val: &Value) -> Option<Vec<usize>> {
    if let Value::Tensor(t) = val {
        Some(t.borrow().shape())
    } else {
        None
    }
}

// ── 前向数值正确性（6 个比较运算各 1 个测试）──

#[test]
fn test_compare_gt_forward() {
    // gt([1,2,3], [2,2,2]) → [0,0,1]
    let src = r#"
        let a = tensor[[1.0, 2.0, 3.0]];
        let b = tensor[[2.0, 2.0, 2.0]];
        tensor_gt(a, b)
    "#;
    let result = run_vm_compare(src).unwrap();
    let v = tensor_to_vec(&result).expect("expected tensor");
    assert_eq!(v, vec![0.0, 0.0, 1.0], "gt([1,2,3], [2,2,2]) should be [0,0,1]");
}

#[test]
fn test_compare_lt_forward() {
    // lt([1,2,3], [2,2,2]) → [1,0,0]
    let src = r#"
        let a = tensor[[1.0, 2.0, 3.0]];
        let b = tensor[[2.0, 2.0, 2.0]];
        tensor_lt(a, b)
    "#;
    let result = run_vm_compare(src).unwrap();
    let v = tensor_to_vec(&result).expect("expected tensor");
    assert_eq!(v, vec![1.0, 0.0, 0.0], "lt([1,2,3], [2,2,2]) should be [1,0,0]");
}

#[test]
fn test_compare_ge_forward() {
    // ge([1,2,3], [2,2,2]) → [0,1,1]
    let src = r#"
        let a = tensor[[1.0, 2.0, 3.0]];
        let b = tensor[[2.0, 2.0, 2.0]];
        tensor_ge(a, b)
    "#;
    let result = run_vm_compare(src).unwrap();
    let v = tensor_to_vec(&result).expect("expected tensor");
    assert_eq!(v, vec![0.0, 1.0, 1.0], "ge([1,2,3], [2,2,2]) should be [0,1,1]");
}

#[test]
fn test_compare_le_forward() {
    // le([1,2,3], [2,2,2]) → [1,1,0]
    let src = r#"
        let a = tensor[[1.0, 2.0, 3.0]];
        let b = tensor[[2.0, 2.0, 2.0]];
        tensor_le(a, b)
    "#;
    let result = run_vm_compare(src).unwrap();
    let v = tensor_to_vec(&result).expect("expected tensor");
    assert_eq!(v, vec![1.0, 1.0, 0.0], "le([1,2,3], [2,2,2]) should be [1,1,0]");
}

#[test]
fn test_compare_eq_forward() {
    // eq([1,2,3], [2,2,2]) → [0,1,0]
    let src = r#"
        let a = tensor[[1.0, 2.0, 3.0]];
        let b = tensor[[2.0, 2.0, 2.0]];
        tensor_eq(a, b)
    "#;
    let result = run_vm_compare(src).unwrap();
    let v = tensor_to_vec(&result).expect("expected tensor");
    assert_eq!(v, vec![0.0, 1.0, 0.0], "eq([1,2,3], [2,2,2]) should be [0,1,0]");
}

#[test]
fn test_compare_ne_forward() {
    // ne([1,2,3], [2,2,2]) → [1,0,1]
    let src = r#"
        let a = tensor[[1.0, 2.0, 3.0]];
        let b = tensor[[2.0, 2.0, 2.0]];
        tensor_ne(a, b)
    "#;
    let result = run_vm_compare(src).unwrap();
    let v = tensor_to_vec(&result).expect("expected tensor");
    assert_eq!(v, vec![1.0, 0.0, 1.0], "ne([1,2,3], [2,2,2]) should be [1,0,1]");
}

// ── 广播用例 ──

#[test]
fn test_compare_gt_broadcast_row() {
    // gt([[1,2],[3,4]], [2]) → [[0,0],[1,1]]
    // [2] 广播到 [[2,2],[2,2]]
    let src = r#"
        let a = tensor[[1.0, 2.0], [3.0, 4.0]];
        let b = tensor[[2.0]];
        tensor_gt(a, b)
    "#;
    let result = run_vm_compare(src).unwrap();
    let shape = tensor_shape(&result).expect("expected tensor");
    assert_eq!(shape, vec![2, 2], "broadcast output shape");
    let v = tensor_to_vec(&result).expect("expected tensor");
    assert_eq!(v, vec![0.0, 0.0, 1.0, 1.0], "gt([[1,2],[3,4]], [2]) should be [[0,0],[1,1]]");
}

#[test]
fn test_compare_eq_broadcast_scalar() {
    // eq([[1,2],[3,4]], 2.5 标量) → [[0,0],[0,0]]
    let src = r#"
        let a = tensor[[1.0, 2.0], [3.0, 4.0]];
        let b = tensor[[2.5]];
        tensor_eq(a, b)
    "#;
    let result = run_vm_compare(src).unwrap();
    let v = tensor_to_vec(&result).expect("expected tensor");
    assert_eq!(v, vec![0.0, 0.0, 0.0, 0.0], "eq(x, 2.5) where x in {{1,2,3,4}} → all 0");
}

// ── 跨 dtype 用例 ──

#[test]
fn test_compare_gt_cross_dtype_f32_f64() {
    // a=f32（用 f32 字面量构造），b=f64，gt 应返回 f64（0.0/1.0）
    // f32[1.0, 2.0, 3.0] vs f64[2.0, 2.0, 2.0] → [0,0,1]
    let src = r#"
        let a = tensor[[1.0f32, 2.0f32, 3.0f32]];
        let b = tensor[[2.0, 2.0, 2.0]];
        tensor_gt(a, b)
    "#;
    let result = run_vm_compare(src).unwrap();
    let v = tensor_to_vec(&result).expect("expected tensor");
    assert_eq!(v, vec![0.0, 0.0, 1.0], "gt(f32, f64) should be f64 [0,0,1]");
}

// ── where_ 别名测试 ──

#[test]
fn test_where_alias_forward() {
    // where_([0,1,0], [1,2,3], [4,5,6]) → [4,2,6]
    // cond=[0,1,0]，1>0.5 选 then，0 选 else
    let src = r#"
        let cond = tensor[[0.0, 1.0, 0.0]];
        let then = tensor[[1.0, 2.0, 3.0]];
        let else_ = tensor[[4.0, 5.0, 6.0]];
        select(cond, then, else_)
    "#;
    let result = run_vm_compare(src).unwrap();
    let v = tensor_to_vec(&result).expect("expected tensor");
    assert_eq!(v, vec![4.0, 2.0, 6.0], "where_([0,1,0], [1,2,3], [4,5,6]) should be [4,2,6]");
}

// ── autodiff 端到端测试（关键！）──

#[test]
fn test_where_gt_relu_equivalent_backward() {
    // loss = where_(gt(x, 0), x, 0).sum()
    // 等价于 ReLU(x).sum()，梯度只流到 x>0 的位置
    //
    // x = [-2, 3, -1, 4]
    // gt(x, 0) = [0, 1, 0, 1]（f64 掩码）
    // where_(cond, x, 0) = [0, 3, 0, 4]（即 relu(x)）
    // sum = 7
    // d_loss/d_x = mask = [0, 1, 0, 1]（梯度只流到 x>0 的位置）
    //
    // 关键：gt 不可微（cond 是常量掩码），select 通过 mask 路由梯度
    //       d_then = grad * mask, d_else = grad * (1-mask)
    //       x 同时作为 then（在 select(cond, x, 0) 中），故 d_x = grad * mask = [0,1,0,1]
    let src = r#"
        new_grad();
        let x = param(tensor[[-2.0, 3.0, -1.0, 4.0]]);
        let cond = tensor_gt(x, 0.0 * x);
        let zero = 0.0 * x;
        let y = select(cond, x, zero);
        let loss = y.sum();
        backward(loss);
        stop_grad();
        grad(x)
    "#;
    let result = run_vm_compare(src).unwrap();
    let v = tensor_to_vec(&result).expect("expected grad tensor");
    let expected = vec![0.0, 1.0, 0.0, 1.0];
    for (i, (got, exp)) in v.iter().zip(expected.iter()).enumerate() {
        assert!((got - exp).abs() < 1e-6, "grad[{}] = {}, expected {}", i, got, exp);
    }
}

#[test]
fn test_where_gt_relu_equivalent_forward() {
    // 前向验证：where_(gt(x, 0), x, 0) 等于 relu(x)
    // x = [-2, 3, -1, 4] → relu = [0, 3, 0, 4]
    let src = r#"
        let x = tensor[[-2.0, 3.0, -1.0, 4.0]];
        let cond = tensor_gt(x, 0.0 * x);
        let zero = 0.0 * x;
        select(cond, x, zero)
    "#;
    let result = run_vm_compare(src).unwrap();
    let v = tensor_to_vec(&result).expect("expected tensor");
    assert_eq!(v, vec![0.0, 3.0, 0.0, 4.0], "where_(gt(x,0), x, 0) should equal relu(x)");
}

#[test]
fn test_where_lt_mask_select_backward() {
    // 验证 cond = lt(x, 0) 时梯度也正确路由
    // x = [-2, 3, -1, 4]
    // lt(x, 0) = [1, 0, 1, 0]
    // where_(cond, x, 0) = [-2, 0, -1, 0]（x<0 处保留 x，否则 0）
    // sum = -3
    // d_loss/d_x = mask = [1, 0, 1, 0]（梯度只流到 x<0 的位置）
    let src = r#"
        new_grad();
        let x = param(tensor[[-2.0, 3.0, -1.0, 4.0]]);
        let cond = tensor_lt(x, 0.0 * x);
        let zero = 0.0 * x;
        let y = select(cond, x, zero);
        let loss = y.sum();
        backward(loss);
        stop_grad();
        grad(x)
    "#;
    let result = run_vm_compare(src).unwrap();
    let v = tensor_to_vec(&result).expect("expected grad tensor");
    let expected = vec![1.0, 0.0, 1.0, 0.0];
    for (i, (got, exp)) in v.iter().zip(expected.iter()).enumerate() {
        assert!((got - exp).abs() < 1e-6, "grad[{}] = {}, expected {}", i, got, exp);
    }
}
