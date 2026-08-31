//! Tensor::shape() 方法：VM=解释器对拍测试（MINOR 兼容新增，2026-08-31）。
//!
//! 背景：用户反馈（shape检查体验-20260731/体验报告.md）指出 Rust 层
//! `Tensor::shape()` 存在（tensor/methods.rs:367）但运行时只注册了
//! `shape_tensor()`（返回 f64 张量），新手写 `x.shape()` 报
//! 「张量没有方法 'shape'」。本次在 VM（vm/natives.rs）与解释器
//! （interpreter/methods.rs）双侧注册 `shape()`（返回 `Vec<i64>`），
//! 撤销 lower_expr.rs 的编译期拦截并恢复 types.rs 类型推断。
//! 本文件守护两条执行路径对同一源码产出一致结果。

use tenth::lexer::lexer::Lexer;
use tenth::parser::parser::Parser;
use tenth::hir::lower::Lowerer;
use tenth::runtime::vm::Vm;
use tenth::runtime::interpreter::Interpreter;
use tenth::runtime::value::Value;
use tenth::runtime::tensor::Tensor;
use tenth::compile::bytecode::BytecodeCompiler;
use std::rc::Rc;
use std::cell::RefCell;

/// 注册 VM 路径所需的最小 native 集（复制自 main.rs，语义一致）。
fn register_test_natives(vm: &mut Vm) {
    vm.add_native("println".into(), |_vm, args| {
        for a in args { print!("{a}"); }
        println!();
        Ok(Value::Unit)
    });
    vm.add_native("zeros".into(), |_vm, args| {
        let shape: Vec<usize> = args.iter().map(|a| a.as_int().unwrap_or(1) as usize).collect();
        Ok(Value::Tensor(Rc::new(RefCell::new(Tensor::zeros(&shape)))))
    });
    vm.add_native("to_string".into(), |_vm, args| {
        if let Some(arg) = args.first() { Ok(Value::String(format!("{}", arg))) }
        else { Ok(Value::String(String::new())) }
    });
    vm.add_native("HashMap::new".into(), |_vm, _args| {
        Ok(Value::Map(Rc::new(RefCell::new(std::collections::HashMap::new()))))
    });
}

/// VM 路径执行（BytecodeCompiler + Vm 直跑，无 JIT）。
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
        vm.call("main").map_err(|e| e.to_string())
    } else if vm.has_fn("main") {
        vm.call("main").map_err(|e| e.to_string())
    } else {
        Ok(Value::Unit)
    }
}

/// 解释器路径执行。
fn run_interp(src: &str) -> Result<Value, String> {
    let mut lexer = Lexer::new(src);
    let tokens = lexer.tokenize().map_err(|e| e.to_string())?;
    let mut parser = Parser::new(tokens);
    let program = parser.parse_program().map_err(|e| e.to_string())?;
    let mut lowerer = Lowerer::new();
    let hir = lowerer.lower_program(&program).map_err(|e| e.to_string())?;
    let mut interp = Interpreter::new(&hir);
    match interp.execute_program(&hir) {
        Ok(Some(v)) => Ok(v),
        Ok(None) => Ok(Value::Unit),
        Err(e) => Err(e.to_string()),
    }
}

/// Value::Vec([Int...]) → Vec<i64>（非该形态返回 None）。
fn shape_of(v: &Value) -> Option<Vec<i64>> {
    match v {
        Value::Vec(items) => {
            let items = items.borrow();
            items.iter().map(|i| match i {
                Value::Int(n, _) => Some(*n),
                _ => None,
            }).collect()
        }
        _ => None,
    }
}

/// 对拍断言：VM 与解释器的 shape() 结果一致，且等于期望值。
fn assert_shape_parity(src: &str, expected: &[i64]) {
    let vm_res = run_vm(src).unwrap_or_else(|e| panic!("VM 执行失败: {}\n源码: {}", e, src));
    let interp_res = run_interp(src).unwrap_or_else(|e| panic!("解释器执行失败: {}\n源码: {}", e, src));
    let vm_shape = shape_of(&vm_res)
        .unwrap_or_else(|| panic!("VM 结果不是 Vec<i64>: {:?}\n源码: {}", vm_res, src));
    let interp_shape = shape_of(&interp_res)
        .unwrap_or_else(|| panic!("解释器结果不是 Vec<i64>: {:?}\n源码: {}", interp_res, src));
    assert_eq!(vm_shape, interp_shape, "VM 与解释器 shape() 不一致\n源码: {}", src);
    assert_eq!(vm_shape, expected.to_vec(),
        "shape() 结果不符合期望\n源码: {}\n期望 {:?}, 实际 {:?}", src, expected, vm_shape);
}

/// 1D 张量：zeros(5) → [5]。
#[test]
fn shape_1d_parity() {
    assert_shape_parity("zeros(5).shape()", &[5]);
}

/// 2D 张量（zeros 构造）：zeros(2, 3) → [2, 3]。
#[test]
fn shape_2d_zeros_parity() {
    assert_shape_parity("zeros(2, 3).shape()", &[2, 3]);
}

/// 2D 张量（字面量构造，非方形）：[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]] → [3, 2]。
#[test]
fn shape_2d_literal_parity() {
    let src = "fn main() {\n    let x = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];\n    x.shape()\n}";
    assert_shape_parity(src, &[3, 2]);
}

/// 3D 张量：zeros(2, 3, 4) → [2, 3, 4]。
#[test]
fn shape_3d_parity() {
    assert_shape_parity("zeros(2, 3, 4).shape()", &[2, 3, 4]);
}

/// fn main 内使用（含中间绑定，贴近用户实际写法）。
#[test]
fn shape_in_main_parity() {
    let src = "fn main() {\n    let x = zeros(4, 5);\n    let s = x.shape();\n    s\n}";
    assert_shape_parity(src, &[4, 5]);
}

/// to_string(x.shape()) 集成（00_smoke.th 的用法）：
/// 两条路径均输出 "[2, 3]" 格式字符串。
#[test]
fn shape_to_string_parity() {
    let src = "to_string(zeros(2, 3).shape())";
    let vm_res = run_vm(src).unwrap_or_else(|e| panic!("VM 执行失败: {}", e));
    let interp_res = run_interp(src).unwrap_or_else(|e| panic!("解释器执行失败: {}", e));
    let (vm_s, interp_s) = match (&vm_res, &interp_res) {
        (Value::String(a), Value::String(b)) => (a.clone(), b.clone()),
        _ => panic!("期望 String，实际 VM={:?} 解释器={:?}", vm_res, interp_res),
    };
    assert_eq!(vm_s, interp_s, "VM 与解释器 to_string(shape()) 不一致");
    assert_eq!(vm_s, "[2, 3]", "to_string(shape()) 应为 '[2, 3]'，实际: {}", vm_s);
}

/// shape() 与 shape_tensor() 信息一致：shape_tensor() 的元素 = shape() 的元素。
#[test]
fn shape_agrees_with_shape_tensor_parity() {
    let src = "fn main() {\n    let x = zeros(2, 3);\n    let s = x.shape();\n    let st = x.shape_tensor();\n    let a = s.len();\n    let b = st.numel();\n    a * 100 + b\n}";
    // 两路径：a=2（shape 长度）、b=2（shape_tensor 元素数）→ 202
    let vm_res = run_vm(src).unwrap_or_else(|e| panic!("VM 执行失败: {}", e));
    let interp_res = run_interp(src).unwrap_or_else(|e| panic!("解释器执行失败: {}", e));
    assert!(matches!(&vm_res, Value::Int(202, _)), "VM 结果应为 202, 实际 {:?}", vm_res);
    assert!(matches!(&interp_res, Value::Int(202, _)), "解释器结果应为 202, 实际 {:?}", interp_res);
}

// ── QA-20260831（黑板留言 5-②）：Tensor::to_vec() 双侧注册对拍 ────────────

/// Value::Vec([Float...]) → Vec<f64>（非该形态返回 None）。
fn floats_of(v: &Value) -> Option<Vec<f64>> {
    match v {
        Value::Vec(items) => {
            let items = items.borrow();
            items.iter().map(|i| match i {
                Value::Float(f) => Some(*f),
                _ => None,
            }).collect()
        }
        _ => None,
    }
}

/// 1D 张量 to_vec()：zeros(3) → [0.0, 0.0, 0.0]。
#[test]
fn to_vec_1d_parity() {
    let src = "zeros(3).to_vec()";
    let vm_res = run_vm(src).unwrap_or_else(|e| panic!("VM 执行失败: {}", e));
    let interp_res = run_interp(src).unwrap_or_else(|e| panic!("解释器执行失败: {}", e));
    let vm_v = floats_of(&vm_res).unwrap_or_else(|| panic!("VM 结果不是 Vec<f64>: {:?}", vm_res));
    let interp_v = floats_of(&interp_res).unwrap_or_else(|| panic!("解释器结果不是 Vec<f64>: {:?}", interp_res));
    assert_eq!(vm_v, interp_v, "VM 与解释器 to_vec() 不一致");
    assert_eq!(vm_v, vec![0.0, 0.0, 0.0], "to_vec() 结果不符合期望: {:?}", vm_v);
}

/// 2D 张量 to_vec()：行主序展平 [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]] → 6 元素。
#[test]
fn to_vec_2d_flatten_parity() {
    let src = "fn main() {\n    let x = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];\n    x.to_vec()\n}";
    let vm_res = run_vm(src).unwrap_or_else(|e| panic!("VM 执行失败: {}", e));
    let interp_res = run_interp(src).unwrap_or_else(|e| panic!("解释器执行失败: {}", e));
    let vm_v = floats_of(&vm_res).unwrap_or_else(|| panic!("VM 结果不是 Vec<f64>: {:?}", vm_res));
    let interp_v = floats_of(&interp_res).unwrap_or_else(|| panic!("解释器结果不是 Vec<f64>: {:?}", interp_res));
    assert_eq!(vm_v, interp_v, "VM 与解释器 to_vec() 不一致");
    assert_eq!(vm_v, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], "to_vec() 应行主序展平: {:?}", vm_v);
}

// ── QA-20260831（黑板留言 5-①）：HashMap::merge 双侧注册对拍 ──────────────

/// merge 基本语义：合并另一 Map（后者键覆盖），len 反映合并结果。
#[test]
fn map_merge_parity() {
    let src = "fn main() {\n    let m = HashMap::new();\n    m.insert(\"a\", 1);\n    let n = HashMap::new();\n    n.insert(\"b\", 2);\n    m.merge(n);\n    m.len()\n}";
    let vm_res = run_vm(src).unwrap_or_else(|e| panic!("VM 执行失败: {}", e));
    let interp_res = run_interp(src).unwrap_or_else(|e| panic!("解释器执行失败: {}", e));
    assert!(matches!(&vm_res, Value::Int(2, _)), "VM merge 后 len 应为 2, 实际 {:?}", vm_res);
    assert!(matches!(&interp_res, Value::Int(2, _)), "解释器 merge 后 len 应为 2, 实际 {:?}", interp_res);
}

/// merge 键覆盖语义：被合并方的同名键覆盖原值。
#[test]
fn map_merge_overwrite_parity() {
    let src = "fn main() {\n    let m = HashMap::new();\n    m.insert(\"a\", 1);\n    let n = HashMap::new();\n    n.insert(\"a\", 9);\n    m.merge(n);\n    m.get(\"a\")\n}";
    let vm_res = run_vm(src).unwrap_or_else(|e| panic!("VM 执行失败: {}", e));
    let interp_res = run_interp(src).unwrap_or_else(|e| panic!("解释器执行失败: {}", e));
    assert!(matches!(&vm_res, Value::Int(9, _)), "VM merge 覆盖后 get(\"a\") 应为 9, 实际 {:?}", vm_res);
    assert!(matches!(&interp_res, Value::Int(9, _)), "解释器 merge 覆盖后 get(\"a\") 应为 9, 实际 {:?}", interp_res);
}
