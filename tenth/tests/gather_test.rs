// gather 原语测试套件
// 覆盖：前向基本用例（dim=0/dim=1/1D index）、shape 校验、dtype 保留、
//       autodiff 梯度（d_base scatter-add 语义、重复 index 累加、index 不可微、与 matmul 链式）、
//       VM/解释器 parity。
// gather 与 scatter 对偶：out[i,j,...] = base[index[i,j,...], j, ...]（dim=0）。

use tenth::lexer::lexer::Lexer;
use tenth::parser::parser::Parser;
use tenth::hir::lower::Lowerer;
use tenth::hir::types::BaseType;
use tenth::runtime::interpreter::Interpreter;
use tenth::runtime::tensor::Tensor;
use tenth::runtime::value::Value;
use tenth::runtime::vm::Vm;
use tenth::runtime::autodiff::Tape;
use tenth::compile::bytecode::BytecodeCompiler;
use std::rc::Rc;
use std::cell::RefCell;

/// 通过解释器运行 Tenth 源码，返回 Result<Option<Value>, String>。
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

/// 从 Value 提取 tensor 的 f64 切片。
fn as_f64_vec(val: &Value) -> Option<Vec<f64>> {
    match val {
        Value::Tensor(t) => {
            let data = t.borrow().data.as_f64_view();
            Some(data.iter().cloned().collect())
        }
        _ => None,
    }
}

// ══════════════════════════════════════════════════════════════════════
// 1. 前向基本用例
// ══════════════════════════════════════════════════════════════════════

#[test]
fn gather_dim0_basic() {
    // base = [[1,2],[3,4],[5,6]]  shape (3,2)
    // index = [[0,1],[2,0]]       shape (2,2)
    // dim = 0：out[i,j] = base[index[i,j], j]
    //   out[0,0]=base[0,0]=1, out[0,1]=base[1,1]=4
    //   out[1,0]=base[2,0]=5, out[1,1]=base[0,1]=2
    // out = [[1,4],[5,2]]
    let src = r#"
        let base = tensor[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let index = tensor[[0.0, 1.0], [2.0, 0.0]];
        gather(base, 0, index)
    "#;
    let result = run_code(src).unwrap();
    let v = as_f64_vec(result.as_ref().unwrap()).expect("expected tensor");
    assert_eq!(v, vec![1.0, 4.0, 5.0, 2.0]);
}

#[test]
fn gather_dim1_basic() {
    // base = [[1,2,3],[4,5,6]]  shape (2,3)
    // index = [[0,2],[1,0]]     shape (2,2)（除 dim=1 外，dim=0 上 == base.shape[0]=2 ✓）
    // dim = 1：out[i,j] = base[i, index[i,j]]
    //   out[0,0]=base[0,0]=1, out[0,1]=base[0,2]=3
    //   out[1,0]=base[1,1]=5, out[1,1]=base[1,0]=4
    // out = [[1,3],[5,4]]
    let src = r#"
        let base = tensor[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let index = tensor[[0.0, 2.0], [1.0, 0.0]];
        gather(base, 1, index)
    "#;
    let result = run_code(src).unwrap();
    let v = as_f64_vec(result.as_ref().unwrap()).expect("expected tensor");
    assert_eq!(v, vec![1.0, 3.0, 5.0, 4.0]);
}

#[test]
fn gather_dim0_1d_index() {
    // base = [10,20,30,40]  shape (4,)
    // index = [2,0,3]       shape (3,)
    // dim = 0：out[i] = base[index[i]]
    //   out = [base[2], base[0], base[3]] = [30, 10, 40]
    // 注：Tenth 的 tensor[[...]] 字面量总是构造 2D（shape [1,N]），gather 要求
    // index.ndim() == base.ndim()，因此用 .flatten() 把 [1,N] 降为 [N]。
    let src = r#"
        let base = tensor[[10.0, 20.0, 30.0, 40.0]].flatten();
        let index = tensor[[2.0, 0.0, 3.0]].flatten();
        gather(base, 0, index)
    "#;
    let result = run_code(src).unwrap();
    let v = as_f64_vec(result.as_ref().unwrap()).expect("expected tensor");
    assert_eq!(v, vec![30.0, 10.0, 40.0]);
}

#[test]
fn gather_out_shape_equals_index_shape() {
    // 验证 out.shape == index.shape（而非 base.shape）。
    // base shape (3,2) → 6 元素；index shape (2,2) → 4 元素；out 应为 4 元素。
    let src = r#"
        let base = tensor[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let index = tensor[[0.0, 1.0], [2.0, 0.0]];
        let out = gather(base, 0, index);
        out.shape_tensor()
    "#;
    let result = run_code(src).unwrap();
    let v = as_f64_vec(result.as_ref().unwrap()).expect("expected shape tensor");
    // out shape 应为 [2,2]（== index.shape），而非 [3,2]（base.shape）
    assert_eq!(v, vec![2.0, 2.0], "out.shape 应等于 index.shape [2,2]");
}

#[test]
fn gather_index_out_of_bounds_errors() {
    // base shape (4,)，dim=0，index 值 5 >= 4 → 越界报错
    let src = r#"
        let base = tensor[[10.0, 20.0, 30.0, 40.0]].flatten();
        let index = tensor[[2.0, 5.0]].flatten();
        gather(base, 0, index)
    "#;
    let result = run_code(src);
    assert!(result.is_err(), "expected error for index out of bounds, got {:?}", result);
    let err = result.unwrap_err();
    assert!(err.contains("越界") || err.contains("out"), "error should mention 越界: {}", err);
}

#[test]
fn gather_dim_mismatch_errors() {
    // base 2D，index 1D → index.ndim() != base.ndim() 报错
    let src = r#"
        let base = tensor[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let index = tensor[[2.0, 0.0]].flatten();
        gather(base, 0, index)
    "#;
    let result = run_code(src);
    assert!(result.is_err(), "expected error for ndim mismatch, got {:?}", result);
    let err = result.unwrap_err();
    assert!(err.contains("ndim") || err.contains("维度"), "error should mention ndim: {}", err);
}

#[test]
fn gather_f32_dtype_preserved() {
    // base 为 f32 时，out 也应为 f32（dtype 跟随 base）。
    // 直接在 Rust 端构造 f32 base，调用 Tensor::gather 验证 dtype。
    let base = Tensor::from_vec_f32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![3, 2]);
    let index = Tensor::from_vec(vec![0.0, 1.0, 2.0, 0.0], vec![2, 2]);
    let out = Tensor::gather(&base, 0, &index).expect("gather should succeed");
    assert!(out.is_f32(), "out 应为 f32 dtype（跟随 base）");
    assert_eq!(out.dtype(), BaseType::F32);
    assert_eq!(out.shape(), vec![2, 2]);
    // out = [[1,4],[5,2]] → row-major [1,4,5,2]
    let out_f32 = out.data.as_f32().expect("f32 view");
    assert_eq!(out_f32.iter().cloned().collect::<Vec<_>>(), vec![1.0f32, 4.0, 5.0, 2.0]);
}

// ══════════════════════════════════════════════════════════════════════
// 2. 反向（autodiff）回归测试
// ══════════════════════════════════════════════════════════════════════

#[test]
fn gather_backward_d_base_basic() {
    // base = param([10,20,30,40])
    // index = [1,3], dim=0
    // out = gather = [20, 40]
    // loss = out.sum() = 60
    // grad(out) = [1, 1]
    // d_base = zeros_like(base)；d_base[index[i]] += grad[i]
    //   d_base[1] += 1, d_base[3] += 1 → d_base = [0, 1, 0, 1]
    let src = r#"
        new_grad();
        let b = param(tensor[[10.0, 20.0, 30.0, 40.0]].flatten());
        let index = tensor[[1.0, 3.0]].flatten();
        let out = gather(b, 0, index);
        backward(out.sum());
        stop_grad();
        grad(b)
    "#;
    let result = run_code(src).unwrap();
    let v = as_f64_vec(result.as_ref().unwrap()).expect("expected tensor");
    assert_eq!(v.len(), 4, "d_base should have 4 elements, got {}", v.len());
    let expected = [0.0, 1.0, 0.0, 1.0];
    for (i, x) in v.iter().enumerate() {
        assert!((x - expected[i]).abs() < 1e-6, "d_base[{}] expected {}, got {}", i, expected[i], x);
    }
}

#[test]
fn gather_backward_d_base_repeated_index_accumulates() {
    // 关键：index 有重复值时，d_base 在该位置累加多个 grad（+= 而非 =）。
    // base = param([10,20,30])
    // index = [0,0,1], dim=0
    // out = [base[0], base[0], base[1]] = [10, 10, 20]
    // loss = out.sum() = 40
    // grad(out) = [1, 1, 1]
    // d_base[0] += grad[0] + grad[1] = 2  (重复 index 累加)
    // d_base[1] += grad[2] = 1
    // d_base[2] += 0 = 0
    // d_base = [2, 1, 0]
    let src = r#"
        new_grad();
        let b = param(tensor[[10.0, 20.0, 30.0]].flatten());
        let index = tensor[[0.0, 0.0, 1.0]].flatten();
        let out = gather(b, 0, index);
        backward(out.sum());
        stop_grad();
        grad(b)
    "#;
    let result = run_code(src).unwrap();
    let v = as_f64_vec(result.as_ref().unwrap()).expect("expected tensor");
    assert_eq!(v.len(), 3, "d_base should have 3 elements, got {}", v.len());
    let expected = [2.0, 1.0, 0.0];
    for (i, x) in v.iter().enumerate() {
        assert!((x - expected[i]).abs() < 1e-6, "d_base[{}] expected {}, got {}", i, expected[i], x);
    }
}

#[test]
fn gather_backward_index_not_differentiable() {
    // index 不接收梯度：即使把 index 作为 param 注册，grad(index) 也应为零张量
    // （因为 gather 的 TapeNode::inputs 只含 [base_id]，index 阻断链式传播）。
    // base = param([10,20,30,40]), index = param([1,3])
    // out = gather(base, 0, index) = [20, 40]
    // loss = out.sum() = 60
    // grad(base) = [0,1,0,1], grad(index) = zeros = [0,0]
    let src = r#"
        new_grad();
        let b = param(tensor[[10.0, 20.0, 30.0, 40.0]].flatten());
        let idx = param(tensor[[1.0, 3.0]].flatten());
        let out = gather(b, 0, idx);
        backward(out.sum());
        stop_grad();
        grad(idx)
    "#;
    let result = run_code(src).unwrap();
    let v = as_f64_vec(result.as_ref().unwrap()).expect("expected tensor");
    assert_eq!(v.len(), 2, "grad(index) should have 2 elements, got {}", v.len());
    // index 不可微 → grad 全 0
    for (i, x) in v.iter().enumerate() {
        assert!(x.abs() < 1e-9, "grad(index)[{}] expected 0.0 (index not differentiable), got {}", i, x);
    }
}

#[test]
fn gather_backward_chain_with_matmul() {
    // gather 嵌套在 matmul 之后，验证梯度链完整传播。
    // base = param([1,2,3,4])  shape (4,)
    // index = [0,2,1,3], dim=0
    // v = gather(base, 0, index) = [base[0], base[2], base[1], base[3]] = [1,3,2,4]
    // w = [[1],[2],[3],[4]]  shape (4,1)
    // y = v.matmul(w)  // 1D @ 2D -> 1D (1,)
    //   y[0] = 1*1 + 3*2 + 2*3 + 4*4 = 1 + 6 + 6 + 16 = 29
    // loss = y.sum() = 29
    // grad(y) = [1]
    // d_v = w * grad = [1,2,3,4]  (4,)
    // d_base[index[i]] += d_v[i]
    //   d_base[0] += 1, d_base[2] += 2, d_base[1] += 3, d_base[3] += 4
    //   d_base = [1, 3, 2, 4]
    let src = r#"
        new_grad();
        let b = param(tensor[[1.0, 2.0, 3.0, 4.0]].flatten());
        let index = tensor[[0.0, 2.0, 1.0, 3.0]].flatten();
        let v = gather(b, 0, index);
        let w = tensor[[1.0], [2.0], [3.0], [4.0]];
        let y = v.matmul(w);
        backward(y.sum());
        stop_grad();
        grad(b)
    "#;
    let result = run_code(src).unwrap();
    let v = as_f64_vec(result.as_ref().unwrap()).expect("expected tensor");
    assert_eq!(v.len(), 4, "d_base should have 4 elements, got {}", v.len());
    let expected = [1.0, 3.0, 2.0, 4.0];
    for (i, x) in v.iter().enumerate() {
        assert!((x - expected[i]).abs() < 1e-6, "d_base[{}] expected {}, got {}", i, expected[i], x);
    }
}

// ══════════════════════════════════════════════════════════════════════
// 3. VM/解释器 parity 测试
// ══════════════════════════════════════════════════════════════════════

/// 注册 VM 路径运行 gather/scatter 所需的 native（复制自 main.rs::register_natives
/// 的相关子集；与 select_test.rs / native_parity_test.rs 同样的项目惯例）。
fn register_test_natives(vm: &mut Vm) {
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
        else { Err(tenth::error::TenthError::RuntimeError { message: "tensor() 参数异常".into() }) }
    });
    vm.add_native("gather".into(), |vm, args| {
        if args.len() < 3 {
            return Err(tenth::error::TenthError::RuntimeError {
                message: "gather(base, dim, index) 期望三个参数".into(),
            });
        }
        let dim = args[1].as_int().unwrap_or(0) as usize;
        let (base, index) = match (&args[0], &args[2]) {
            (Value::Tensor(b), Value::Tensor(i)) => (b.clone(), i.clone()),
            _ => return Err(tenth::error::TenthError::RuntimeError {
                message: "gather(base, dim, index) 期望 base/index 为张量".into(),
            }),
        };
        let result_tensor = Tensor::gather(&base.borrow(), dim, &index.borrow())
            .map_err(|msg| tenth::error::TenthError::RuntimeError { message: msg })?;
        let result = Rc::new(RefCell::new(result_tensor));
        if vm.recording {
            if let Some(ref mut tape) = vm.tape {
                let base_id = base.borrow().tape_id;
                let node_id = tape.gather(base_id, base.clone(), index.clone(), result.clone(), dim);
                result.borrow_mut().tape_id = Some(node_id);
            }
        }
        Ok(Value::Tensor(result))
    });
    vm.add_native("scatter".into(), |vm, args| {
        if args.len() < 4 {
            return Err(tenth::error::TenthError::RuntimeError {
                message: "scatter(base, dim, index, src) 期望四个参数".into(),
            });
        }
        let dim = args[1].as_int().unwrap_or(0) as usize;
        let (base, index, src) = match (&args[0], &args[2], &args[3]) {
            (Value::Tensor(b), Value::Tensor(i), Value::Tensor(s)) => (b.clone(), i.clone(), s.clone()),
            _ => return Err(tenth::error::TenthError::RuntimeError {
                message: "scatter(base, dim, index, src) 期望 base/index/src 为张量".into(),
            }),
        };
        let result_tensor = Tensor::scatter(&base.borrow(), dim, &index.borrow(), &src.borrow())
            .map_err(|msg| tenth::error::TenthError::RuntimeError { message: msg })?;
        let result = Rc::new(RefCell::new(result_tensor));
        if vm.recording {
            if let Some(ref mut tape) = vm.tape {
                let base_id = base.borrow().tape_id;
                let src_id = src.borrow().tape_id;
                let node_id = tape.scatter(base_id, src_id, base.clone(), src.clone(), index.clone(), result.clone(), dim);
                result.borrow_mut().tape_id = Some(node_id);
            }
        }
        Ok(Value::Tensor(result))
    });
    // autodiff 辅助 native（与 vm_autodiff_test.rs 一致）
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
            Err(tenth::error::TenthError::RuntimeError { message: "param() requires a tensor argument".into() })
        }
    });
    vm.add_native("backward".into(), |vm, args| {
        if let Some(Value::Tensor(t)) = args.first() {
            if let Some(ref tape) = vm.tape {
                let loss_id = t.borrow().tape_id
                    .ok_or_else(|| tenth::error::TenthError::RuntimeError { message: "backward(): tensor has no tape_id".into() })?;
                tape.backward(loss_id).map_err(|e| tenth::error::TenthError::RuntimeError { message: format!("{:?}", e) })?;
                Ok(Value::Unit)
            } else {
                Err(tenth::error::TenthError::RuntimeError { message: "new_grad() not called".into() })
            }
        } else {
            Err(tenth::error::TenthError::RuntimeError { message: "backward() requires a tensor argument".into() })
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
            Err(tenth::error::TenthError::RuntimeError { message: "grad() requires a tensor argument".into() })
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
}

/// 通过 VM 执行 .th 源码，返回 Value。
fn run_vm(src: &str) -> Result<Value, String> {
    let mut lexer = Lexer::new(src);
    let tokens = lexer.tokenize().map_err(|e| e.to_string())?;
    let mut parser = Parser::new(tokens);
    let program = parser.parse_program().map_err(|e| e.to_string())?;
    let mut lowerer = Lowerer::new();
    let hir = lowerer.lower_program(&program).map_err(|e| e.to_string())?;

    let mut vm = Vm::new();
    register_test_natives(&mut vm);

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
        vm.call("main").map_err(|e| e.to_string())
    } else if vm.has_fn("main") {
        vm.call("main").map_err(|e| e.to_string())
    } else {
        Ok(Value::Unit)
    }
}

#[test]
fn gather_vm_interpreter_parity() {
    // 同一 gather 操作在 VM 和解释器路径下结果一致。
    // base = [[1,2],[3,4],[5,6]] (3x2), index = [[0,1],[2,0]] (2x2), dim=0
    // out = [[1,4],[5,2]] → row-major [1,4,5,2]
    let src = r#"
        let base = tensor[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let index = tensor[[0.0, 1.0], [2.0, 0.0]];
        gather(base, 0, index)
    "#;
    let vm_res = run_vm(src).expect("VM 执行失败");
    let interp_res = run_code(src).expect("解释器执行失败");
    let vm_v = as_f64_vec(&vm_res).expect("VM 结果应为 tensor");
    let interp_v = as_f64_vec(interp_res.as_ref().unwrap()).expect("解释器结果应为 tensor");
    assert_eq!(vm_v, interp_v, "VM 与解释器 gather 结果不一致");
    assert_eq!(vm_v, vec![1.0, 4.0, 5.0, 2.0]);
}

#[test]
fn gather_vm_interpreter_parity_dim1() {
    // dim=1 parity：base (2,3), index (2,2), dim=1
    let src = r#"
        let base = tensor[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let index = tensor[[0.0, 2.0], [1.0, 0.0]];
        gather(base, 1, index)
    "#;
    let vm_res = run_vm(src).expect("VM 执行失败");
    let interp_res = run_code(src).expect("解释器执行失败");
    let vm_v = as_f64_vec(&vm_res).expect("VM 结果应为 tensor");
    let interp_v = as_f64_vec(interp_res.as_ref().unwrap()).expect("解释器结果应为 tensor");
    assert_eq!(vm_v, interp_v, "VM 与解释器 gather(dim=1) 结果不一致");
    assert_eq!(vm_v, vec![1.0, 3.0, 5.0, 4.0]);
}
