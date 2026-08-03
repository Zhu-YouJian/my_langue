//! M2.6-P4：调用路径扩展守护测试——CallClosure JIT-to-JIT 直接调用 + TailCall 保守评估。
//!
//! P4 目标与结论：
//! 1. **CallClosure**：目标（闭包 chunk）**不可编译期静态解析**（opcode 只带参数个数，
//!    callee 是运行期栈上 Value）→ 无法用 A1 的机器码 `call_indirect` 快路径；改为
//!    `host_jit_call_indirect`（A1 慢路径等价物：FnRef 按名 → `jit_call_chunk` → 闭包体
//!    直接执行 JIT 机器码，不再逃逸解释器）。捕获追加 / letrec Shared 解包 / native
//!    直名 / 非可调用值报错语义与 `call_value` 逐一对齐（静默错值红线）。
//! 2. **MethodCall**：目标全为 native（`call_method_priv` 按接收者运行期类型分派，无
//!    用户函数 chunk）→ JIT-to-JIT 直接调用不适用，保持 `host_method_call`（如实记录）。
//! 3. **TailCall**：VM opcode 55 是真 TCO（帧复用）；JIT 若改 JIT-to-JIT → 每尾调用压
//!    原生栈帧（A1 教训 ~7KB/帧，MAX_STACK_DEPTH 64）→ 深尾递归 0xC0000005 崩溃。
//!    当前保守 `host_call` → 解释器 TCO 帧复用（迭代帧池，深尾递归安全）。**不做**，
//!    保守记录（收益=避免单次解释器逃逸；风险=安全行为变崩溃）。
//!
//! 守护内容：闭包调用 / 方法调用 / 尾调用的 **VM=JIT=解释器 三路径对拍**（成功值 +
//! 错误行号/消息一致）；错误路径带行号（非可调用值）；覆盖断言（闭包 chunk 真的
//! 被 JIT 编译——防静默回退解释器）。

use tenth::compile::bytecode::BytecodeCompiler;
use tenth::compile::jit;
use tenth::error::TenthError;
use tenth::hir::lower::Lowerer;
use tenth::lexer::lexer::Lexer;
use tenth::parser::parser::Parser;
use tenth::runtime::interpreter::Interpreter;
use tenth::runtime::natives::register_all_natives;
use tenth::runtime::value::Value;
use tenth::runtime::vm::Vm;

// ═══════════════════════════════════════════════════════════════════════════
// helper（镜像 jit_spec_f64_test：globals + __global_init + fn main 优先）
// ═══════════════════════════════════════════════════════════════════════════

fn lower(src: &str) -> Result<tenth::hir::hir::HirProgram, String> {
    let mut lexer = Lexer::new(src);
    let tokens = lexer.tokenize().map_err(|e| e.to_string())?;
    let mut parser = Parser::new(tokens);
    let program = parser.parse_program().map_err(|e| e.to_string())?;
    let mut lowerer = Lowerer::new();
    lowerer.lower_program(&program).map_err(|e| e.to_string())
}

fn compile_vm(src: &str) -> Result<Vm, String> {
    let hir = lower(src)?;
    let global_names: std::collections::HashSet<String> =
        hir.globals.iter().map(|g| g.name.clone()).collect();
    let mut vm = Vm::new();
    register_all_natives(&mut vm);
    for func in &hir.functions {
        let compiler = BytecodeCompiler::new_with_globals(global_names.clone());
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
    if !hir.globals.is_empty() {
        let gcompiler = BytecodeCompiler::new_with_globals(global_names.clone());
        match gcompiler.compile_globals(&hir.globals) {
            Ok((gchunk, gclosures)) => {
                vm.add_fn("__global_init".into(), gchunk);
                for (name, closure_chunk) in gclosures {
                    vm.add_fn(name, closure_chunk);
                }
                vm.call("__global_init").map_err(|e| format!("__global_init: {}", e))?;
            }
            Err(e) => return Err(format!("compile_globals error: {}", e)),
        }
    }
    if !vm.has_fn("main") {
        if let Some(ref expr) = hir.main_expr {
            let compiler = BytecodeCompiler::new_with_globals(global_names);
            match compiler.compile_main(expr) {
                Ok((chunk, closures)) => {
                    vm.add_fn("main".into(), chunk);
                    for (name, closure_chunk) in closures {
                        vm.add_fn(name, closure_chunk);
                    }
                }
                Err(e) => return Err(format!("compile error: {}", e)),
            }
        }
    }
    Ok(vm)
}

/// 纯 VM 字节码路径（不经 JIT）。
fn run_vm(src: &str) -> Result<Value, TenthError> {
    let mut vm = compile_vm(src)
        .map_err(|e| TenthError::RuntimeError { line: None, col: None, message: e })?;
    if vm.has_fn("main") { vm.call("main") } else { Ok(Value::Unit) }
}

/// JIT 路径（保留 Vm 供覆盖断言）。
fn run_jit_with_vm(src: &str) -> Result<(Value, Vm), TenthError> {
    let mut vm = compile_vm(src)
        .map_err(|e| TenthError::RuntimeError { line: None, col: None, message: e })?;
    if vm.has_fn("main") {
        let r = jit::run_jit(&mut vm, "main")?;
        Ok((r, vm))
    } else {
        Ok((Value::Unit, vm))
    }
}

/// 树遍解释器路径（= TENTH_NO_VM=1 的真实语义）。
fn run_interp(src: &str) -> Result<Value, TenthError> {
    let hir = lower(src)
        .map_err(|e| TenthError::RuntimeError { line: None, col: None, message: e })?;
    let mut interpreter = Interpreter::new(&hir);
    interpreter.execute_program(&hir)
        .map(|opt| opt.unwrap_or(Value::Unit))
}

fn int_of(v: Value, label: &str) -> i64 {
    match v { Value::Int(n, _) => n, other => panic!("[{label}] 期望 Int，实际 {:?}", other) }
}

fn err_parts(err: &TenthError) -> (Option<usize>, String) {
    match err {
        TenthError::RuntimeError { line, message, .. } => (*line, message.clone()),
        other => (None, format!("{:?}", other)),
    }
}

/// chunk 是否已 JIT 编译（防静默回退）。
fn jit_compiled(vm: &Vm, name: &str, label: &str) -> bool {
    let idx = vm.chunk_index_of(name).unwrap_or_else(|| panic!("[{label}] chunk {name} 未注册"));
    match vm.jit_ctx.as_ref() {
        Some(ctx) => ctx.is_compiled(idx),
        None => false,
    }
}

/// 三路径对拍：成功性 + 值 Debug 逐位一致 + 错误消息/行号一致。
/// `require_compiled` = 断言 main 已 JIT 编译（非整函数 fallback）。
fn assert_three_consistent(src: &str, label: &str, require_compiled: bool) {
    let vm_res = run_vm(src);
    let jit_pair = run_jit_with_vm(src);
    let interp_res = run_interp(src);

    let jit_res = match &jit_pair {
        Ok((v, vm)) => {
            if require_compiled {
                assert!(jit_compiled(vm, "main", label),
                    "[{label}] main 未 JIT 编译（整函数回退 VM）——审计未覆盖 JIT 路径！\nsrc:\n{src}");
            }
            Ok(v.clone())
        }
        Err(e) => Err(e.clone()),
    };

    let okness = |r: &Result<Value, TenthError>| r.is_ok();
    assert_eq!(okness(&vm_res), okness(&jit_res),
        "[{label}] VM/JIT 成功性不一致\nVM: {:?}\nJIT: {:?}", vm_res, jit_res);
    assert_eq!(okness(&vm_res), okness(&interp_res),
        "[{label}] VM/解释器 成功性不一致\nVM: {:?}\nInterp: {:?}", vm_res, interp_res);

    match (&vm_res, &jit_res, &interp_res) {
        (Ok(a), Ok(b), Ok(c)) => {
            let sa = format!("{:?}", a);
            let sb = format!("{:?}", b);
            let sc = format!("{:?}", c);
            assert_eq!(sa, sb, "[{label}] VM/JIT 结果值不一致（静默错值红线！）\nVM={sa}\nJIT={sb}\nsrc:\n{src}");
            assert_eq!(sa, sc, "[{label}] VM/解释器 结果值不一致\nVM={sa}\nInterp={sc}\nsrc:\n{src}");
        }
        (Err(a), Err(b), Err(c)) => {
            let (la, ma) = err_parts(a);
            let (lb, mb) = err_parts(b);
            let (lc, mc) = err_parts(c);
            assert_eq!(la, lb, "[{label}] VM/JIT 错误行号不一致 VM={:?} JIT={:?}\nVM={ma}\nJIT={mb}\nsrc:\n{src}", la, lb);
            assert_eq!(ma, mb, "[{label}] VM/JIT 错误消息不一致\nVM={ma}\nJIT={mb}\nsrc:\n{src}");
            assert_eq!(ma, mc, "[{label}] VM/解释器 错误消息不一致\nVM={ma}\nInterp={mc}\nsrc:\n{src}");
        }
        _ => panic!("[{label}] 三路径结果形态不一致\nVM: {:?}\nJIT: {:?}\nInterp: {:?}", vm_res, jit_res, interp_res),
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// 1. CallClosure JIT-to-JIT：闭包调用三路径对拍 + 覆盖断言
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn call_closure_single() {
    // 单闭包无捕获：let f = |x| x+1; f(5) → 6
    let src = r#"
        fn main() -> Int {
            let f = |x: Int| x + 1;
            f(5)
        }
    "#;
    let (v, vm) = run_jit_with_vm(src).unwrap();
    assert_eq!(int_of(v, "closure-single"), 6);
    // 覆盖断言：闭包 chunk 必须真的被 JIT 编译（P4 核心——防静默回退解释器）。
    // main 内第一个闭包 chunk 名 = __closure_main_0（closure_counter 从 0 起）。
    assert!(jit_compiled(&vm, "__closure_main_0", "closure-single"),
        "闭包 chunk __closure_main_0 应被 JIT 编译（host_jit_call_indirect 直连）");
    assert_three_consistent(src, "closure-single-parity", true);
}

#[test]
fn call_closure_capture() {
    // 多捕获：|x| x*scale+base → 3*2+10 = 16（捕获值追加为额外实参）
    let src = r#"
        fn main() -> Int {
            let base = 10;
            let scale = 2;
            let compute = |x: Int| x * scale + base;
            compute(3)
        }
    "#;
    let (v, vm) = run_jit_with_vm(src).unwrap();
    assert_eq!(int_of(v, "closure-capture"), 16);
    assert!(jit_compiled(&vm, "__closure_main_0", "closure-capture"),
        "带捕获闭包应被 JIT 编译");
    assert_three_consistent(src, "closure-capture-parity", true);
}

#[test]
fn call_closure_capture_independent() {
    // 多实例捕获独立：make_adder(5)/make_adder(10) → add5(3)=8, add10(3)=13 → 813
    // （捕获值内联 + JIT 直连；捕获串扰回归 → 1313 可辨识）
    let src = r#"
        fn make_adder(n: Int) -> fn(Int) -> Int {
            |x: Int| x + n
        }
        fn main() -> Int {
            let add5 = make_adder(5);
            let add10 = make_adder(10);
            add5(3) * 100 + add10(3)
        }
    "#;
    let (v, vm) = run_jit_with_vm(src).unwrap();
    assert_eq!(int_of(v, "closure-capture-indep"), 813);
    assert!(jit_compiled(&vm, "__closure_make_adder_0", "closure-capture-indep"),
        "make_adder 内闭包应被 JIT 编译");
    assert_three_consistent(src, "closure-capture-indep-parity", true);
}

#[test]
fn call_closure_hof_param() {
    // 闭包作参数传入 HOF，函数体内经参数槽间接调用（CallClosure 捕获注入）：
    // apply_twice(addn, 3) = f(f(3)) = (3+5)+5 = 13
    let src = r#"
        fn apply_twice(f: fn(Int) -> Int, x: Int) -> Int {
            f(f(x))
        }
        fn main() -> Int {
            let n = 5;
            let addn = |x: Int| x + n;
            apply_twice(addn, 3)
        }
    "#;
    let (v, vm) = run_jit_with_vm(src).unwrap();
    assert_eq!(int_of(v, "closure-hof"), 13);
    assert!(jit_compiled(&vm, "__closure_main_0", "closure-hof"),
        "HOF 传参闭包应被 JIT 编译");
    assert_three_consistent(src, "closure-hof-parity", true);
}

#[test]
fn call_closure_recursive() {
    // 递归闭包（自引用 cell/全局解析）：fact(5) = 120。
    // 注：递归闭包创建点含 MakeCell/BindSelfCapture（M1-S2 letrec）——A3 起保守
    // 整函数 fallback（A3 决策：仅闭包创建点含此二指令，非本次优先级）。故此处只做
    // 三路径对拍（值一致），不断言 JIT 编译；P4 的 JIT-to-JIT 覆盖由非递归闭包承担。
    let src = r#"
        fn main() -> Int {
            let fact = |n: Int| if n <= 1 { 1 } else { n * fact(n - 1) };
            fact(5)
        }
    "#;
    let (v, _vm) = run_jit_with_vm(src).unwrap();
    assert_eq!(int_of(v, "closure-recursive"), 120);
    assert_three_consistent(src, "closure-recursive-parity", false);
}

#[test]
fn call_closure_native_alias() {
    // native 别名闭包：let p = println; p("x")——FnRef.name=native 名 → jit_call_chunk
    // natives 直名路径。println 返回 Unit，三路径均成功即可辨识回归。
    let src = r#"
        fn main() {
            let p = println;
            p("alias")
        }
    "#;
    assert_three_consistent(src, "closure-native-alias", true);
}

#[test]
fn call_closure_letrec_cell() {
    // true letrec：递归闭包经 Shared cell（MakeCell/BindSelfCapture）——创建点在
    // main（M1-S2 后自引用走 cell）。fact(5)=120，三路径一致。
    // 注：同 call_closure_recursive——MakeCell/BindSelfCapture 使 main 整函数
    // fallback（A3 保守决策），只做三路径对拍。
    let src = r#"
        fn main() -> Int {
            let fact = |n: Int| if n <= 1 { 1 } else { n * fact(n - 1) };
            fact(5) * 100
        }
    "#;
    let (v, _vm) = run_jit_with_vm(src).unwrap();
    assert_eq!(int_of(v, "closure-letrec"), 12000);
    assert_three_consistent(src, "closure-letrec-parity", false);
}

#[test]
fn call_closure_error_non_callable_has_line() {
    // 调用非函数值（整数）→ 运行期「期望可调用值，得到 ...」错误，应带调用点行号。
    // 三路径错误消息一致；行号 VM/JIT 一致（解释器行号允许不同，只断言消息）。
    let src = r#"fn main() {
    let x = 42;
    x()
}
"#;
    let vm_res = run_vm(src);
    let jit_pair = run_jit_with_vm(src);
    let interp_res = run_interp(src);
    let (vm_err, jit_err, interp_err) = (
        vm_res.err().expect("[closure-err] VM 应报错"),
        jit_pair.err().expect("[closure-err] JIT 应报错"),
        interp_res.err().expect("[closure-err] Interp 应报错"),
    );
    let (lvm, mvm) = err_parts(&vm_err);
    let (ljit, mjit) = err_parts(&jit_err);
    assert_eq!(lvm, Some(3), "VM 非可调用值应定位到第 3 行，实际 {:?}", vm_err);
    assert_eq!(ljit, Some(3), "JIT 非可调用值应定位到第 3 行，实际 {:?}", jit_err);
    assert_eq!(mvm, mjit, "VM/JIT 错误消息不一致\nVM={mvm}\nJIT={mjit}");
    assert!(mvm.contains("期望可调用值"), "VM 消息应含「期望可调用值」，实际 {mvm}");
    // 注：解释器措辞为「不是可调用值」（预存措辞分歧），只断言三路径都报错。
    let (_, minterp) = err_parts(&interp_err);
    assert!(minterp.contains("可调用"), "Interp 消息应含可调用相关，实际 {minterp}");
}

// ═══════════════════════════════════════════════════════════════════════════
// 2. MethodCall：目标全为 native（无用户函数 chunk）→ 保持 host_method_call。
//    三路径对拍守护语义一致 + main 不整函数回退。
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn method_string_len() {
    let src = r#"
        fn main() -> Int {
            let s = "hello, tenth";
            s.len()
        }
    "#;
    let (v, vm) = run_jit_with_vm(src).unwrap();
    assert_eq!(int_of(v, "method-string-len"), 12);
    assert!(jit_compiled(&vm, "main", "method-string-len"),
        "含 MethodCall 的 main 应 JIT 编译（MethodCall 是 hostcall，不整函数回退）");
    assert_three_consistent(src, "method-string-len-parity", true);
}

#[test]
fn method_string_chained() {
    // 字符串方法链：trim → to_upper → contains
    let src = r#"
        fn main() -> Bool {
            let s = "  hello  ";
            s.trim().to_upper().contains("HELLO")
        }
    "#;
    assert_three_consistent(src, "method-string-chained", true);
}

#[test]
fn method_tensor_sum() {
    // 张量方法：ones(2,3).sum() = 6
    let src = r#"
        fn main() -> Float {
            let t = ones(2, 3);
            t.sum()
        }
    "#;
    assert_three_consistent(src, "method-tensor-sum", true);
}

#[test]
fn method_vec_len_push() {
    // Vec 方法：push/len（可变方法经 host_method_call）
    let src = r#"
        fn main() -> Int {
            let v = Vec::new();
            v.push(1);
            v.push(2);
            v.push(3);
            v.len()
        }
    "#;
    assert_three_consistent(src, "method-vec", true);
}

#[test]
fn method_error_has_line() {
    // 不存在的方法 → 运行期报错，应带调用点行号（VM/JIT 一致；解释器只断言消息）。
    let src = r#"fn main() {
    let s = "abc".nonexistent_method();
    print(s)
}
"#;
    let vm_res = run_vm(src);
    let jit_pair = run_jit_with_vm(src);
    let interp_res = run_interp(src);
    let (vm_err, jit_err, interp_err) = (
        vm_res.err().expect("[method-err] VM 应报错"),
        jit_pair.err().expect("[method-err] JIT 应报错"),
        interp_res.err().expect("[method-err] Interp 应报错"),
    );
    let (lvm, mvm) = err_parts(&vm_err);
    let (ljit, mjit) = err_parts(&jit_err);
    assert_eq!(lvm, Some(2), "VM 方法错误应定位到第 2 行，实际 {:?}", vm_err);
    assert_eq!(ljit, Some(2), "JIT 方法错误应定位到第 2 行，实际 {:?}", jit_err);
    assert_eq!(mvm, mjit, "VM/JIT 错误消息不一致\nVM={mvm}\nJIT={mjit}");
    // 注：解释器措辞为「String 没有方法 ...」（预存措辞分歧），只断言三路径都报错。
    let (_, minterp) = err_parts(&interp_err);
    assert!(minterp.contains("没有方法"), "Interp 消息应含没有方法，实际 {minterp}");
}

// ═══════════════════════════════════════════════════════════════════════════
// 3. TailCall：保守评估——不做 JIT-to-JIT（深尾递归原生栈溢出风险）。三路径对拍
//    守护当前 host_call→解释器 TCO 路径的语义正确（结果一致、不整函数回退）。
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn tailcall_named() {
    // fn f 尾位置调用 g（TailCall）→ 结果 = g 结果（21*2=42）
    let src = r#"
        fn g(x: Int) -> Int { x * 2 }
        fn f(n: Int) -> Int { g(n) }
        fn main() -> Int { f(21) }
    "#;
    let (v, vm) = run_jit_with_vm(src).unwrap();
    assert_eq!(int_of(v, "tailcall-named"), 42);
    assert!(jit_compiled(&vm, "main", "tailcall-named"),
        "含 TailCall 的 main 应 JIT 编译（TailCall 是 hostcall，不整函数回退）");
    assert_three_consistent(src, "tailcall-named-parity", true);
}

#[test]
fn tailcall_recursive() {
    // 有界尾递归：sum_tail(10, 0) → 55（保守路径语义正确）
    let src = r#"
        fn sum_tail(n: Int, acc: Int) -> Int {
            if n <= 0 { acc } else { sum_tail(n - 1, acc + n) }
        }
        fn main() -> Int { sum_tail(10, 0) }
    "#;
    assert_three_consistent(src, "tailcall-recursive", true);
}

#[test]
fn tailcall_deep_recursive_no_crash() {
    // P4 关键回归：深尾递归（1e5 层）在 VM/JIT 路径**必须不崩溃且结果正确**——
    // TailCall 保守 host_call → 解释器 TCO 帧复用（迭代帧池），原生栈零增长。
    // 若误改 JIT-to-JIT（无 TCO 帧复用）→ 每尾调用压原生栈帧（~7KB/帧，A1 教训）→
    // 1e5 层 ≈ 700MB > 256MB 工作线程 → 0xC0000005 崩溃。
    // 期望：顺序模和 = sum(1..100000) mod 1000000007 = 49965（VM/JIT 实测一致）。
    // 注：解释器（树遍递归）在 >~1e4 层即原生栈溢出（预存限制），故本测试只做 VM/JIT
    // 双路径对拍；三路径一致性由 tailcall_recursive（有界）覆盖。
    let src = r#"
        fn sum_tail(n: Int, acc: Int) -> Int {
            if n <= 0 { acc } else { sum_tail(n - 1, (acc + n) % 1000000007) }
        }
        fn main() -> Int { sum_tail(100000, 0) }
    "#;
    let vm_v = int_of(run_vm(src).expect("VM 深尾递归应成功"), "tailcall-deep-vm");
    let (jit_v, _vm) = run_jit_with_vm(src).expect("JIT 深尾递归应成功");
    let jit_v = int_of(jit_v, "tailcall-deep-jit");
    assert_eq!(vm_v, 49965, "VM 深尾递归结果错误（应 49965）");
    assert_eq!(jit_v, 49965, "JIT 深尾递归结果错误（应 49965）——不崩溃且值正确");
}

#[test]
fn tailcall_closure() {
    // 闭包尾调用（TailCallClosure）：apply 内 f(n) 尾位置 → 42
    let src = r#"
        fn apply(n: Int) -> Int {
            let f = |x: Int| x * 2;
            f(n)
        }
        fn main() -> Int { apply(21) }
    "#;
    assert_three_consistent(src, "tailcall-closure", true);
}
