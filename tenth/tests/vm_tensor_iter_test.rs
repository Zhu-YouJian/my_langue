// 张量 for-in 迭代的 VM 路径守护测试。
//
// 背景（任务4）：`Tenth实例/矩阵分解/matfact.th`、`边缘检测/sobel.th` 曾因
// VM 路径 `for row in tensor` 报「张量没有方法 'len'」而只能走解释器。
// 根因：解释器对 `HirStmtKind::For` 的 Tensor 有专门分支（直接按 shape[0] 逐行切片），
// 而 VM 的 for-in 在 bytecode.rs 编译为通用 `while __idx < __iter.len() { var = __iter[__idx] }`，
// 依赖 tensor 的 `len` 方法（= 第 0 维长度），但 VM tensor 方法分派（natives.rs）缺该分支。
// 修复：VM natives.rs 与 interpreter methods.rs 双侧补齐 `tensor.len()` = shape[0]。
//
// 本测试用纯 VM（无 JIT）守护：
//   1. `tensor.len()` 返回第 0 维长度（行数）
//   2. `for row in tensor` 逐行迭代（IndexGet 取行子张量），行内容正确
//   3. Range / Vec 既有 for-in 迭代不回归
use tenth::compile::bytecode::BytecodeCompiler;
use tenth::hir::lower::Lowerer;
use tenth::lexer::lexer::Lexer;
use tenth::parser::parser::Parser;
use tenth::runtime::natives::register_all_natives;
use tenth::runtime::value::Value;
use tenth::runtime::vm::Vm;

/// 纯 VM（无 JIT）执行 .th 源码，返回 main 结果。
fn run_plain_vm(src: &str) -> Result<Value, String> {
    let tokens = Lexer::new(src).tokenize().map_err(|e| e.to_string())?;
    let program = Parser::new(tokens).parse_program().map_err(|e| e.to_string())?;
    let mut lowerer = Lowerer::new();
    let hir = lowerer.lower_program(&program).map_err(|e| e.to_string())?;
    let mut vm = Vm::new();
    register_all_natives(&mut vm);
    for func in &hir.functions {
        let compiler = BytecodeCompiler::new();
        if let Ok((chunk, closures)) = compiler.compile(func) {
            vm.add_fn(func.name.clone(), chunk);
            for (name, closure_chunk) in closures {
                vm.add_fn(name, closure_chunk);
            }
        }
    }
    if let Some(ref expr) = hir.main_expr {
        let compiler = BytecodeCompiler::new();
        if let Ok((chunk, closures)) = compiler.compile_main(expr) {
            vm.add_fn("main".into(), chunk);
            for (name, closure_chunk) in closures {
                vm.add_fn(name, closure_chunk);
            }
        }
    }
    vm.call("main").map_err(|e| e.to_string())
}

#[test]
fn vm_tensor_len_returns_first_dim() {
    // 修复前：VM tensor 无 'len' 方法 → 「张量没有方法 'len'」
    // 修复后：len() = 第 0 维长度（行数），NumPy 语义
    let src = r#"
        fn main() -> f64 {
            let m = tensor[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];
            let ok_len: f64 = if m.len() == 3 { 1.0 } else { 0.0 };
            let ok_ndim: f64 = if m.ndim() == 2 { 1.0 } else { 0.0 };
            let ok_numel: f64 = if m.numel() == 9 { 1.0 } else { 0.0 };
            ok_len * 100.0 + ok_ndim * 10.0 + ok_numel
        }
    "#;
    let r = run_plain_vm(src).unwrap();
    // 期望: 1*100 + 1*10 + 1 = 111
    assert!(matches!(r, Value::Float(v) if (v - 111.0).abs() < 1e-9),
        "tensor.len()/ndim()/numel() 组合应为 111，实际 {:?}", r);
}

#[test]
fn vm_for_in_tensor_rows() {
    // 修复前：VM 的 `for row in tensor` 依赖 __iter.len() → 报「张量没有方法 'len'」
    // 修复后：len()=行数、__iter[__idx]→行子张量，逐行迭代与解释器一致
    let src = r#"
        fn main() -> f64 {
            let m = tensor[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];
            let mut total: f64 = 0.0;
            let mut count: f64 = 0.0;
            for row in m {
                total = total + row[0] + row[1] + row[2];
                count = count + 1.0;
            };
            let ok_count: f64 = if count == 3.0 { 1.0 } else { 0.0 };
            // total = 1+2+3+4+5+6+7+8+9 = 45
            let ok_total: f64 = if total == 45.0 { 1.0 } else { 0.0 };
            ok_count * 100.0 + ok_total * 10.0 + count
        }
    "#;
    let r = run_plain_vm(src).unwrap();
    // 期望: 1*100 + 1*10 + 3 = 113
    assert!(matches!(r, Value::Float(v) if (v - 113.0).abs() < 1e-9),
        "for row in tensor 迭代应为 3 行、total=45（113），实际 {:?}", r);
}

#[test]
fn vm_for_in_tensor_method_expr() {
    // 迭代对象为方法调用结果（如 `edges.reshape(3, 3)`），同样走通用 for-in 分支
    let src = r#"
        fn main() -> f64 {
            let m = tensor[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]];
            let mut total: f64 = 0.0;
            for row in m.reshape(2, 4) {
                total = total + row[0] + row[1] + row[2] + row[3];
            };
            // reshape(2,4) → 2 行，每行 [1,2,3,4] / [5,6,7,8]，total = 36
            let ok: f64 = if total == 36.0 { 1.0 } else { 0.0 };
            ok * 100.0 + total
        }
    "#;
    let r = run_plain_vm(src).unwrap();
    // 期望: 1*100 + 36 = 136
    assert!(matches!(r, Value::Float(v) if (v - 136.0).abs() < 1e-9),
        "for row in tensor 方法表达式迭代应为 total=36（136），实际 {:?}", r);
}

#[test]
fn vm_for_in_range_vec_no_regression() {
    // 既有 Range / Vec 通用 for-in 迭代不回归（__iter.len() + IndexGet 路径复用）
    let src = r#"
        fn main() -> i64 {
            let mut s: i64 = 0;
            for i in 0..4 { s = s + i; };   // 0+1+2+3 = 6
            let mut v = Vec::new();
            v.push(10); v.push(20); v.push(30);
            let mut vs: i64 = 0;
            for x in v { vs = vs + x; };    // 60
            s * 100 + vs                    // 600 + 60 = 660
        }
    "#;
    let r = run_plain_vm(src).unwrap();
    assert!(matches!(r, Value::Int(n, _) if n == 660),
        "Range/Vec for-in 应得 660，实际 {:?}", r);
}
