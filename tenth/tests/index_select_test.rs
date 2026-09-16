//! AUDIT-11.4.11：`index_select` —— 沿 dim 维按 **1-D** index 收集切片。
//!
//! 设计裁定（总师）：
//!   - **native 自由函数**（`index_select(base, dim, index)`），**不是张量方法**
//!     （两后端的方法表都没有 `gather` 臂，方法形态是"lower 过、运行时报没有方法"的陷阱）。
//!   - `out.shape = base.shape[..dim] + [index.len()] + base.shape[dim+1..]`。
//!   - `index` 限 1-D：静态可知 ≠1 → **编译期 TypeError**；运行时再校验一次。
//!   - dtype 跟随 base（f64/f32/f16/bf16 四臂）。
//!   - **index 严格化**：NaN / 非整数 / 越界 → **响亮报错**
//!     （`gather` 的 `*v as i64` 静默截断是另立的 AUDIT-11.4.68，本原语不继承）。
//!   - 空 index 不 panic；`dim` 越界响亮。
//!   - 可微：`d_base` 为 scatter-add（与 Gather 反向共用内核），index 不可微。
//!   - **WASM 天然响亮**（`resolve_func` 报「未定义函数」）⇒ 断言文案即可，不写声明代码。

use tenth::compile::bytecode::BytecodeCompiler;
use tenth::hir::lower::Lowerer;
use tenth::hir::types::BaseType;
use tenth::lexer::lexer::Lexer;
use tenth::parser::parser::Parser;
use tenth::runtime::autodiff::Tape;
use tenth::runtime::interpreter::Interpreter;
use tenth::runtime::natives::register_all_natives;
use tenth::runtime::tensor::Tensor;
use tenth::runtime::value::Value;
use tenth::runtime::vm::Vm;
use std::cell::RefCell;
use std::rc::Rc;

/// 构造与 main.rs 一致的 std 搜索路径（供 `use std::nn::embedding` 实跑）。
fn build_search_paths() -> Vec<String> {
    let mut search_paths = Vec::new();
    if let Ok(cwd) = std::env::current_dir() {
        search_paths.push(cwd.to_string_lossy().to_string());
    }
    let std_dev = std::path::Path::new("tenth/std");
    if std_dev.exists() {
        if let Some(parent) = std_dev.parent() {
            search_paths.push(parent.to_string_lossy().to_string());
        }
        search_paths.push(std_dev.to_string_lossy().to_string());
    }
    let std_local = std::path::Path::new("std");
    if std_local.exists() {
        if let Some(parent) = std_local.parent() {
            search_paths.push(parent.to_string_lossy().to_string());
        }
        search_paths.push(std_local.to_string_lossy().to_string());
    }
    search_paths
}

fn lower_with(src: &str, with_std: bool) -> Result<tenth::hir::hir::HirProgram, String> {
    let mut lexer = Lexer::new(src);
    let tokens = lexer.tokenize().map_err(|e| e.to_string())?;
    let mut parser = Parser::new(tokens);
    let program = parser.parse_program().map_err(|e| e.to_string())?;
    let mut lowerer = if with_std {
        Lowerer::with_search_paths(build_search_paths())
    } else {
        Lowerer::new()
    };
    lowerer.lower_program(&program).map_err(|e| e.to_string())
}

fn run_vm(src: &str) -> Result<Value, String> {
    run_vm_with(src, false)
}

fn run_vm_with(src: &str, with_std: bool) -> Result<Value, String> {
    let hir = lower_with(src, with_std)?;
    let mut vm = Vm::new();
    register_all_natives(&mut vm);
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
            Err(e) => return Err(format!("compile error: {e}")),
        }
    }
    if let Some(ref expr) = hir.main_expr {
        let compiler = BytecodeCompiler::new();
        let (chunk, closures) = compiler.compile_main(expr).map_err(|e| format!("compile error: {e}"))?;
        vm.add_fn("main".into(), chunk);
        for (name, closure_chunk) in closures {
            vm.add_fn(name, closure_chunk);
        }
        vm.call("main").map_err(|e| e.to_string())
    } else {
        Ok(Value::Unit)
    }
}

fn run_interp(src: &str) -> Result<Value, String> {
    run_interp_with(src, false)
}

fn run_interp_with(src: &str, with_std: bool) -> Result<Value, String> {
    let hir = lower_with(src, with_std)?;
    let mut interp = Interpreter::new(&hir);
    interp.execute_program(&hir)
        .map(|v| v.unwrap_or(Value::Unit))
        .map_err(|e| e.to_string())
}

fn as_f64_vec(val: &Value) -> Vec<f64> {
    match val {
        Value::Tensor(t) => t.borrow().data.as_f64_view().iter().cloned().collect(),
        other => panic!("期望张量，实际 {other:?}"),
    }
}

fn assert_close(actual: &[f64], expected: &[f64], label: &str) {
    assert_eq!(actual.len(), expected.len(), "[{label}] 元素个数不符: {actual:?} vs {expected:?}");
    for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!((a - e).abs() < 1e-6, "[{label}] 第 {i} 个元素: {a} != {e}（完整 {actual:?}）");
    }
}

// ══════════════════════════════════════════════════════════════════════
// 1. 前向：dim=0 / dim=1 / shape / 重复 index（两路径）
// ══════════════════════════════════════════════════════════════════════

#[test]
fn index_select_dim0_basic_both_paths() {
    // base = [[1,2],[3,4],[5,6]] (3,2)，index = [2,0] → out = [[5,6],[1,2]]
    let src = r#"
        let base = tensor[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let index = tensor[[2.0, 0.0]].flatten();
        index_select(base, 0, index)
    "#;
    let vm = as_f64_vec(&run_vm(src).expect("VM 失败"));
    let ip = as_f64_vec(&run_interp(src).expect("解释器失败"));
    assert_close(&vm, &[5.0, 6.0, 1.0, 2.0], "dim0 VM");
    assert_close(&ip, &[5.0, 6.0, 1.0, 2.0], "dim0 解释器");
    assert_close(&vm, &ip, "dim0 parity");
}

#[test]
fn index_select_out_shape_replaces_dim_both_paths() {
    // out.shape = [index.len(), base.shape[1]] = [2,2]（不是 base.shape [3,2]）
    let src = r#"
        let base = tensor[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let index = tensor[[2.0, 0.0]].flatten();
        index_select(base, 0, index).shape_tensor()
    "#;
    let vm = as_f64_vec(&run_vm(src).expect("VM 失败"));
    let ip = as_f64_vec(&run_interp(src).expect("解释器失败"));
    assert_close(&vm, &[2.0, 2.0], "shape VM");
    assert_close(&ip, &[2.0, 2.0], "shape 解释器");
}

#[test]
fn index_select_dim1_basic_both_paths() {
    // base = [[1,2,3],[4,5,6]] (2,3)，index = [2,0]，dim=1
    // out.shape = [2, 2]，out[i] = [base[i][index[0]], base[i][index[1]]]
    //   → [[base[0][2], base[0][0]], [base[1][2], base[1][0]]] = [[3,1],[6,4]]
    // 注意这与 gather（out.shape == index.shape 且 index 二维）语义**不同**。
    let src = r#"
        let base = tensor[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let index = tensor[[2.0, 0.0]].flatten();
        index_select(base, 1, index)
    "#;
    let vm = as_f64_vec(&run_vm(src).expect("VM 失败"));
    let ip = as_f64_vec(&run_interp(src).expect("解释器失败"));
    assert_close(&vm, &[3.0, 1.0, 6.0, 4.0], "dim1 VM");
    assert_close(&ip, &[3.0, 1.0, 6.0, 4.0], "dim1 解释器");
}

#[test]
fn index_select_repeated_index_both_paths() {
    // 重复 index：同一行取两次
    let src = r#"
        let base = tensor[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let index = tensor[[1.0, 1.0]].flatten();
        index_select(base, 0, index)
    "#;
    let vm = as_f64_vec(&run_vm(src).expect("VM 失败"));
    let ip = as_f64_vec(&run_interp(src).expect("解释器失败"));
    assert_close(&vm, &[3.0, 4.0, 3.0, 4.0], "重复 index VM");
    assert_close(&ip, &[3.0, 4.0, 3.0, 4.0], "重复 index 解释器");
}

// ══════════════════════════════════════════════════════════════════════
// 2. 前向：3-D leading 非平凡 / dtype / 空 index（Rust 端直调）
// ══════════════════════════════════════════════════════════════════════

#[test]
fn index_select_3d_leading_non_trivial() {
    // base (2,3,4) 值 1..24，dim=1，index=[2,0] → out (2,2,4)
    let base = Tensor::from_vec((1..=24).map(|x| x as f64).collect(), vec![2, 3, 4]);
    let index = Tensor::from_vec(vec![2.0, 0.0], vec![2]);
    let out = Tensor::index_select(&base, 1, &index).expect("index_select 失败");
    assert_eq!(out.shape(), vec![2, 2, 4], "out.shape 应为 base 的 dim 槽替换为 index.len()");
    let v: Vec<f64> = out.data.as_f64_view().iter().cloned().collect();
    assert_close(
        &v,
        &[
            9.0, 10.0, 11.0, 12.0, // base[0][2]
            1.0, 2.0, 3.0, 4.0, // base[0][0]
            21.0, 22.0, 23.0, 24.0, // base[1][2]
            13.0, 14.0, 15.0, 16.0, // base[1][0]
        ],
        "3-D leading",
    );
}

#[test]
fn index_select_preserves_f32_dtype() {
    let base = Tensor::from_vec_f32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![3, 2]);
    let index = Tensor::from_vec(vec![2.0, 0.0], vec![2]);
    let out = Tensor::index_select(&base, 0, &index).expect("index_select 失败");
    assert_eq!(out.dtype, BaseType::F32, "dtype 必须跟随 base（f32）");
    let f32v = out.data.as_f32().expect("应为 f32 存储");
    let v: Vec<f32> = f32v.iter().cloned().collect();
    assert_eq!(v, vec![5.0f32, 6.0, 1.0, 2.0]);
}

#[test]
fn index_select_empty_index_no_panic() {
    // 空 index（len==0）→ dim 维为 0 的空张量，不得 panic
    let base = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![3, 2]);
    let index = Tensor::from_vec(vec![], vec![0]);
    let out = Tensor::index_select(&base, 0, &index).expect("空 index 不应失败");
    assert_eq!(out.shape(), vec![0, 2], "空 index 的 out.shape 应为 [0, 2]");
    assert_eq!(out.data.len(), 0, "空 index 的输出应为 0 元素");
}

// ══════════════════════════════════════════════════════════════════════
// 3. 边界 / 响亮（dim 越界、index ndim、NaN、非整数、越界值、编译期）
// ══════════════════════════════════════════════════════════════════════

#[test]
fn index_select_dim_out_of_range_is_loud_both_paths() {
    let src = r#"
        let base = tensor[[1.0, 2.0], [3.0, 4.0]];
        let index = tensor[[0.0]].flatten();
        index_select(base, 5, index)
    "#;
    let vm = run_vm(src).unwrap_err();
    let ip = run_interp(src).unwrap_err();
    assert!(vm.contains("dim=5") && vm.contains("越界"), "VM 应响亮报 dim 越界，实际 {vm}");
    assert!(ip.contains("dim=5") && ip.contains("越界"), "解释器应响亮报 dim 越界，实际 {ip}");
}

#[test]
fn index_select_scalar_base_is_loud() {
    // 0 维（标量）张量 → base 必须为非标量
    let base = Tensor::from_vec(vec![1.0], vec![]);
    let index = Tensor::from_vec(vec![0.0], vec![1]);
    let err = Tensor::index_select(&base, 0, &index).unwrap_err();
    assert!(err.contains("非标量"), "应报 base 必须非标量，实际 {err}");
}

/// 传非张量实参也必须**响亮**（不得静默返回空张量）。
#[test]
fn index_select_non_tensor_arg_is_loud_both_paths() {
    let src = r#"
        let base = tensor[[1.0, 2.0], [3.0, 4.0]];
        index_select(base, 0, 1.5)
    "#;
    let vm = run_vm(src).unwrap_err();
    let ip = run_interp(src).unwrap_err();
    assert!(vm.contains("期望 base/index 为张量"), "VM 应响亮，实际 {vm}");
    assert!(ip.contains("期望 base/index 为张量"), "解释器应响亮，实际 {ip}");
}

#[test]
fn index_select_runtime_index_ndim_not_1_is_loud() {
    // 2-D index → 运行时响亮（gather 语义走 gather，本原语拒绝）
    let base = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let index = Tensor::from_vec(vec![0.0, 1.0, 1.0, 0.0], vec![2, 2]);
    let err = Tensor::index_select(&base, 0, &index).unwrap_err();
    assert!(err.contains("一维") && err.contains("ndim=2"), "应响亮报 ndim≠1，实际 {err}");
}

#[test]
fn index_select_compile_time_index_ndim_is_type_error() {
    // 静态可知 index 是 2-D → **编译期** TypeError（不是拖到运行时）
    let src = r#"
        let base = tensor[[1.0, 2.0], [3.0, 4.0]];
        let index = tensor[[0.0, 1.0], [1.0, 0.0]];
        index_select(base, 0, index)
    "#;
    let err = lower_with(src, false).unwrap_err();
    assert!(err.contains("index_select") && err.contains("一维"),
        "应为编译期 index ndim 错误，实际 {err}");
}

#[test]
fn index_select_nan_index_is_loud() {
    let base = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let index = Tensor::from_vec(vec![f64::NAN], vec![1]);
    let err = Tensor::index_select(&base, 0, &index).unwrap_err();
    assert!(err.contains("不是整数"), "NaN index 必须响亮（不得静默当 0），实际 {err}");
}

#[test]
fn index_select_non_integer_index_is_loud() {
    // 1.7 → 响亮（gather 现状是静默截断为 1；新原语不继承该静默失败）
    let base = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let index = Tensor::from_vec(vec![1.7], vec![1]);
    let err = Tensor::index_select(&base, 0, &index).unwrap_err();
    assert!(err.contains("不是整数"), "非整数 index 必须响亮，实际 {err}");
}

#[test]
fn index_select_out_of_range_value_is_loud() {
    let base = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let index = Tensor::from_vec(vec![9.0], vec![1]);
    let err = Tensor::index_select(&base, 0, &index).unwrap_err();
    assert!(err.contains("越界"), "越界 index 必须响亮，实际 {err}");
}

// ══════════════════════════════════════════════════════════════════════
// 4. autodiff（d_base scatter-add / 重复累加 / index 不可微 / 链式 / 空 index）
// ══════════════════════════════════════════════════════════════════════

#[test]
fn index_select_backward_d_base_basic() {
    // base = param([[1,2],[3,4],[5,6]])，index=[2,0]，out=[[5,6],[1,2]]
    // loss = out.sum() = 14，grad(out)=ones(2,2)
    // d_base[2] += [1,1]，d_base[0] += [1,1] → [[1,1],[0,0],[1,1]]
    let src = r#"
        new_grad();
        let b = param(tensor[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]);
        let index = tensor[[2.0, 0.0]].flatten();
        let out = index_select(b, 0, index);
        backward(out.sum());
        stop_grad();
        grad(b)
    "#;
    let vm = as_f64_vec(&run_vm(src).expect("VM 失败"));
    let ip = as_f64_vec(&run_interp(src).expect("解释器失败"));
    assert_close(&vm, &[1.0, 1.0, 0.0, 0.0, 1.0, 1.0], "d_base VM");
    assert_close(&ip, &[1.0, 1.0, 0.0, 0.0, 1.0, 1.0], "d_base 解释器");
}

#[test]
fn index_select_backward_repeated_index_accumulates() {
    // index=[1,1] → 同一行取两次：d_base[1] += 2*ones → [[0,0],[2,2],[0,0]]
    let src = r#"
        new_grad();
        let b = param(tensor[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]);
        let index = tensor[[1.0, 1.0]].flatten();
        let out = index_select(b, 0, index);
        backward(out.sum());
        stop_grad();
        grad(b)
    "#;
    let vm = as_f64_vec(&run_vm(src).expect("VM 失败"));
    let ip = as_f64_vec(&run_interp(src).expect("解释器失败"));
    assert_close(&vm, &[0.0, 0.0, 2.0, 2.0, 0.0, 0.0], "重复 index 累加 VM");
    assert_close(&ip, &[0.0, 0.0, 2.0, 2.0, 0.0, 0.0], "重复 index 累加 解释器");
}

#[test]
fn index_select_backward_index_not_differentiable() {
    // index 即使注册为 param 也拿不到梯度（inputs 只含 base_id，index 阻断链式传播）
    let src = r#"
        new_grad();
        let b = param(tensor[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]);
        let idx = param(tensor[[2.0, 0.0]].flatten());
        let out = index_select(b, 0, idx);
        backward(out.sum());
        stop_grad();
        grad(idx)
    "#;
    let vm = as_f64_vec(&run_vm(src).expect("VM 失败"));
    let ip = as_f64_vec(&run_interp(src).expect("解释器失败"));
    assert_close(&vm, &[0.0, 0.0], "grad(index) 必须为 0（VM）");
    assert_close(&ip, &[0.0, 0.0], "grad(index) 必须为 0（解释器）");
}

#[test]
fn index_select_backward_chain_with_matmul() {
    // 与 gather_test 的同型用例（1-D base，index=[0,2,1,3]）：
    // v = [1,3,2,4]；y = v.matmul([[1],[2],[3],[4]]) → 29；d_base = [1,3,2,4]
    let src = r#"
        new_grad();
        let b = param(tensor[[1.0, 2.0, 3.0, 4.0]].flatten());
        let index = tensor[[0.0, 2.0, 1.0, 3.0]].flatten();
        let v = index_select(b, 0, index);
        let w = tensor[[1.0], [2.0], [3.0], [4.0]];
        let y = v.matmul(w);
        backward(y.sum());
        stop_grad();
        grad(b)
    "#;
    let vm = as_f64_vec(&run_vm(src).expect("VM 失败"));
    let ip = as_f64_vec(&run_interp(src).expect("解释器失败"));
    assert_close(&vm, &[1.0, 3.0, 2.0, 4.0], "链式 d_base VM");
    assert_close(&ip, &[1.0, 3.0, 2.0, 4.0], "链式 d_base 解释器");
}

#[test]
fn index_select_backward_empty_index_all_zero_grad() {
    // 空 index → 全 0 梯度（不得 panic / 不得除零）
    let base = Rc::new(RefCell::new(Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3, 1])));
    let index = Rc::new(RefCell::new(Tensor::from_vec(vec![], vec![0])));
    let out = Tensor::index_select(&base.borrow(), 0, &index.borrow()).expect("前向失败");
    assert_eq!(out.shape(), vec![0, 1]);
    let out = Rc::new(RefCell::new(out));

    let mut tape = Tape::new();
    let bid = tape.input(base.clone());
    let oid = tape.index_select(Some(bid), base.clone(), index.clone(), out.clone(), 0);
    out.borrow_mut().tape_id = Some(oid);
    tape.backward(oid).expect("空 index 反向不应失败");

    let g = base.borrow().grad.clone().expect("base 应收到梯度（全 0）");
    let gv: Vec<f64> = g.as_f64_view().iter().cloned().collect();
    assert_close(&gv, &[0.0, 0.0, 0.0], "空 index 的 d_base");
}

/// 参数缺失/类型错不得 panic（编译期 shape 推断保守返回同秩 Any；运行时响亮报错）。
#[test]
fn index_select_wrong_arity_no_panic() {
    assert!(
        lower_with("index_select()", false).is_ok(),
        "缺参时编译期 shape 推断必须保守（不得 panic 编译器）"
    );
    let err = run_vm("index_select()").unwrap_err();
    assert!(err.contains("期望三个参数"), "运行时应响亮报参数个数，实际 {err}");
    let err = run_interp("index_select()").unwrap_err();
    assert!(err.contains("期望三个参数"), "解释器应响亮报参数个数，实际 {err}");
}

// ══════════════════════════════════════════════════════════════════════
// 5. WASM：天然响亮（断言文案；不写声明代码）
// ══════════════════════════════════════════════════════════════════════

#[test]
fn index_select_wasm_is_loud_undefined_function() {
    // 张量从**参数**进入（WASM 无 tensor 字面量 host 导入，若在函数体里构造张量会先报
    // "未定义函数 'tensor'"，掩盖本项要断言的名字）。
    let src = r#"
        fn pick(base: Tensor[f64, ..], index: Tensor[f64, ..]) -> Tensor[f64, ..] {
            index_select(base, 0, index)
        }
    "#;
    let hir = lower_with(src, false).expect("lower 失败");
    let err = match tenth::compile::compile_to_wasm(&hir) {
        Ok(_) => panic!("WASM 后端不支持 index_select，必须**响亮报错**而不是静默出值"),
        Err(e) => format!("{e}"),
    };
    assert!(
        err.contains("未定义函数 'index_select'"),
        "WASM 错误消息必须点名 index_select（防将来退化为静默/泛化报错），实际 {err}"
    );
}

// ══════════════════════════════════════════════════════════════════════
// 6. `tenth/std/nn/embedding.th` 实跑（切到新原语后数值等价）
// ══════════════════════════════════════════════════════════════════════

/// embedding.th 已从「reshape + 加法广播 + gather」切到 `index_select(weight, 0, indices)`；
/// 本用例走真实 std 搜索路径 `use` 加载该模块并**执行**（泛型函数需显式类型实参
/// `embedding<f64>(...)` —— 跨文件泛型调用的隐式推断尚不支持，是既有语言限制，
/// 与本次改动无关），与"逐行复制"期望值逐元素比对。
const EMBEDDING_CALL: &str = r#"
use std::nn::embedding::embedding
let weight = tensor[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]];
let indices = tensor[[2.0, 0.0, 3.0]].flatten();
embedding<f64>(weight, indices, 3, 2)
"#;

#[test]
fn embedding_th_runs_and_matches_row_copy_both_paths() {
    let vm = as_f64_vec(&run_vm_with(EMBEDDING_CALL, true).expect("VM 执行 embedding.th 失败"));
    let ip = as_f64_vec(&run_interp_with(EMBEDDING_CALL, true).expect("解释器执行 embedding.th 失败"));
    // weight 第 2/0/3 行 → [5,6, 1,2, 7,8]
    assert_close(&vm, &[5.0, 6.0, 1.0, 2.0, 7.0, 8.0], "embedding VM");
    assert_close(&ip, &[5.0, 6.0, 1.0, 2.0, 7.0, 8.0], "embedding 解释器");
}

#[test]
fn embedding_th_gradient_flows_to_weight() {
    // embedding 训练可用：d_weight 是 scatter-add（每行取到的次数）
    let src = r#"
use std::nn::embedding::embedding
new_grad();
let weight = param(tensor[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]);
let indices = tensor[[2.0, 0.0, 2.0]].flatten();
let out = embedding<f64>(weight, indices, 3, 2);
backward(out.sum());
stop_grad();
grad(weight)
"#;
    let vm = as_f64_vec(&run_vm_with(src, true).expect("VM 失败"));
    // 第 2 行被取两次 → [2,2]；第 0 行一次 → [1,1]；第 1/3 行 0
    assert_close(&vm, &[1.0, 1.0, 0.0, 0.0, 2.0, 2.0, 0.0, 0.0], "embedding d_weight VM");
}
