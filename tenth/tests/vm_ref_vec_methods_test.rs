// 守护测试：AUDIT-11.4.21（VM &mut 写回）+ AUDIT-11.4.28（VM Vec/Map 方法）
// + AUDIT-11.4.24（Vec == 比较）三项修复。
//
// 背景：
//   1. VM 引用此前是 pass-through（不追踪所有权），DerefAssign 硬编码 Store(0)——
//      被引用变量不在槽位 0 时写错槽位（静默错值）。修复后 &mut 变量经
//      Value::Shared 槽位 + Value::MutRef(Weak)，*m = v 写穿 Shared。
//   2. VM Vec/Map 方法分派此前缺 10 个 Vec 方法 + Map.entries（解释器全有），
//      阻塞 std::collections::flat_map/map_values/filter_map。
//   3. Vec == 此前 vm_eq/values_eq 均无 Vec 分支 → 恒 false。
use tenth::compile::bytecode::BytecodeCompiler;
use tenth::hir::lower::Lowerer;
use tenth::lexer::lexer::Lexer;
use tenth::parser::parser::Parser;
use tenth::runtime::interpreter::Interpreter;
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

/// 解释器执行 .th 源码，返回 main 结果（语义基准）。
fn run_interp(src: &str) -> Result<Value, String> {
    let tokens = Lexer::new(src).tokenize().map_err(|e| e.to_string())?;
    let program = Parser::new(tokens).parse_program().map_err(|e| e.to_string())?;
    let mut lowerer = Lowerer::new();
    let hir = lowerer.lower_program(&program).map_err(|e| e.to_string())?;
    let mut interp = Interpreter::new(&hir);
    interp.execute_program(&hir)
        .map(|opt| opt.unwrap_or(Value::Unit))
        .map_err(|e| e.to_string())
}

fn as_int(v: &Value) -> i64 {
    // &mut 后变量槽为 Value::Shared——解包读取（VM 与解释器一致）
    match v {
        Value::Int(n, _) => *n,
        Value::Shared(rc) => as_int(&rc.borrow()),
        Value::Ref(rc) => as_int(&rc.borrow()),
        Value::MutRef(w) => match w.upgrade() {
            Some(rc) => as_int(&rc.borrow()),
            None => panic!("悬垂 &mut"),
        },
        other => panic!("期望 Int，实际 {:?}", other),
    }
}

fn as_bool(v: &Value) -> bool {
    match v {
        Value::Bool(b) => *b,
        Value::Shared(rc) => as_bool(&rc.borrow()),
        Value::Ref(rc) => as_bool(&rc.borrow()),
        other => panic!("期望 Bool，实际 {:?}", other),
    }
}

// ── AUDIT-11.4.21：VM &mut 写回（静默错值修复） ─────────────────────────

#[test]
fn vm_mutref_writeback_not_slot0() {
    // 关键回归：被引用变量 y 不在槽位 0（a 占槽 0）。
    // 修复前 DerefAssign 硬编码 Store(0) → 20 写进 a，y 保持 10（静默错值）。
    // 修复后经 Shared 槽位写穿 → y == 20。
    let src = "{ let a = 1; let mut y = 10; let m = &mut y; *m = 20; y }";
    let vm = run_plain_vm(src).unwrap();
    let interp = run_interp(src).unwrap();
    assert_eq!(as_int(&vm), 20, "VM &mut 写回（非槽 0）应为 20，实际 {:?}", vm);
    assert_eq!(as_int(&interp), 20, "解释器基准应为 20，实际 {:?}", interp);
}

#[test]
fn vm_mutref_writeback_deref_assignop() {
    // *m += 5（DerefAssignOp）：修复前忽略 op 且硬编码 Store(0)。
    let src = "{ let a = 1; let mut y = 10; let m = &mut y; *m += 5; y }";
    let vm = run_plain_vm(src).unwrap();
    let interp = run_interp(src).unwrap();
    assert_eq!(as_int(&vm), 15, "VM *m += 5 应为 15，实际 {:?}", vm);
    assert_eq!(as_int(&interp), 15, "解释器基准应为 15，实际 {:?}", interp);
}

#[test]
fn vm_shared_ref_after_mutref() {
    // 共享引用先声明、可变引用后写回的经典顺序（AUDIT 复现语义）。
    let src = "{ let mut y = 10; let m = &mut y; *m = 20; let x = 42; let r = &x; *r * 100 + y }";
    let vm = run_plain_vm(src).unwrap();
    let interp = run_interp(src).unwrap();
    // *r = 42, y = 20 → 42*100 + 20 = 4220
    assert_eq!(as_int(&vm), 4220, "VM 引用混合顺序应为 4220，实际 {:?}", vm);
    assert_eq!(as_int(&interp), 4220, "解释器基准应为 4220，实际 {:?}", interp);
}

// ── AUDIT-11.4.24：Vec == 比较（元素逐一相等） ──────────────────────────

#[test]
fn vm_vec_eq_elementwise_true() {
    let src = "[1, 2, 3] == [1, 2, 3]";
    let vm = run_plain_vm(src).unwrap();
    let interp = run_interp(src).unwrap();
    assert!(as_bool(&vm), "VM [1,2,3]==[1,2,3] 应为 true，实际 {:?}", vm);
    assert!(as_bool(&interp), "解释器 [1,2,3]==[1,2,3] 应为 true，实际 {:?}", interp);
}

#[test]
fn vm_vec_eq_elementwise_false() {
    let src = "[1, 2, 3] == [1, 2, 4]";
    let vm = run_plain_vm(src).unwrap();
    assert!(!as_bool(&vm), "VM [1,2,3]==[1,2,4] 应为 false，实际 {:?}", vm);
}

#[test]
fn vm_vec_eq_len_mismatch() {
    let src = "[1, 2, 3] == [1, 2]";
    let vm = run_plain_vm(src).unwrap();
    assert!(!as_bool(&vm), "VM [1,2,3]==[1,2] 应为 false（长度不同），实际 {:?}", vm);
}

// ── AUDIT-11.4.28：VM Vec 方法补齐 ─────────────────────────────────────

#[test]
fn vm_vec_extend() {
    let src = "{ let mut r = Vec::new(); r.extend([1, 2, 3]); r.len() }";
    let vm = run_plain_vm(src).unwrap();
    assert_eq!(as_int(&vm), 3, "VM Vec.extend 后 len 应为 3，实际 {:?}", vm);
}

#[test]
fn vm_vec_index_of() {
    let src = "{ let v = [10, 20, 30]; v.index_of(20) }";
    let vm = run_plain_vm(src).unwrap();
    assert_eq!(as_int(&vm), 1, "VM Vec.index_of(20) 应为 1，实际 {:?}", vm);
}

#[test]
fn vm_vec_index_of_missing() {
    let src = "{ let v = [10, 20, 30]; v.index_of(99) }";
    let vm = run_plain_vm(src).unwrap();
    assert_eq!(as_int(&vm), -1, "VM Vec.index_of(99) 应为 -1，实际 {:?}", vm);
}

#[test]
fn vm_vec_reverse() {
    let src = "{ let v = [1, 2, 3]; let r = v.reverse(); r.get(0) * 100 + r.get(2) }";
    let vm = run_plain_vm(src).unwrap();
    assert_eq!(as_int(&vm), 301, "VM Vec.reverse 后 [0]=3 [2]=1 → 301，实际 {:?}", vm);
}

#[test]
fn vm_vec_slice() {
    let src = "{ let v = [1, 2, 3, 4]; let s = v.slice(1, 3); s.len() * 10 + s.get(0) }";
    let vm = run_plain_vm(src).unwrap();
    assert_eq!(as_int(&vm), 22, "VM Vec.slice(1,3) len=2 [0]=2 → 22，实际 {:?}", vm);
}

#[test]
fn vm_vec_sort() {
    let src = "{ let mut v = [3, 1, 2]; v.sort(); v.get(0) * 100 + v.get(1) * 10 + v.get(2) }";
    let vm = run_plain_vm(src).unwrap();
    assert_eq!(as_int(&vm), 123, "VM Vec.sort 应得 123，实际 {:?}", vm);
}

#[test]
fn vm_vec_dedup() {
    let src = "{ let mut v = [1, 1, 2, 2, 3]; v.dedup(); v.len() * 10 + v.get(0) }";
    let vm = run_plain_vm(src).unwrap();
    assert_eq!(as_int(&vm), 31, "VM Vec.dedup len=3 [0]=1 → 31，实际 {:?}", vm);
}

#[test]
fn vm_vec_first_last() {
    let src = "{ let v = [5, 6]; v.first() * 10 + v.last() }";
    let vm = run_plain_vm(src).unwrap();
    assert_eq!(as_int(&vm), 56, "VM Vec.first/last 应得 56，实际 {:?}", vm);
}

#[test]
fn vm_vec_flatten() {
    // 嵌套数组字面量 `[[1,2],[3]]` 会被推断为 Tensor，故用 push 构造 Vec-of-Vec。
    let src = r#"
        fn main() -> i64 {
            let mut v = Vec::new();
            let mut i1 = Vec::new(); i1.push(1); i1.push(2);
            let mut i2 = Vec::new(); i2.push(3);
            v.push(i1); v.push(i2);
            let f = v.flatten();
            f.len() * 10 + f.get(2)
        }
    "#;
    let vm = run_plain_vm(src).unwrap();
    assert_eq!(as_int(&vm), 33, "VM Vec.flatten len=3 [2]=3 → 33，实际 {:?}", vm);
}

#[test]
fn vm_vec_chunks() {
    let src = "{ let v = [1, 2, 3, 4, 5]; let c = v.chunks(2); c.len() }";
    let vm = run_plain_vm(src).unwrap();
    assert_eq!(as_int(&vm), 3, "VM Vec.chunks(2) len 应为 3，实际 {:?}", vm);
}

// ── AUDIT-11.4.28：VM Map.entries ──────────────────────────────────────

#[test]
fn vm_map_entries() {
    let src = "{ let mut m = HashMap::new(); m.insert(\"a\", 1); m.insert(\"b\", 2); let e = m.entries(); e.len() }";
    let vm = run_plain_vm(src).unwrap();
    assert_eq!(as_int(&vm), 2, "VM Map.entries len 应为 2，实际 {:?}", vm);
}

// ── 标准库解锁验证：flat_map / map_values / filter_map 在 VM 可用 ────────

#[test]
fn vm_flat_map_usable() {
    // flat_map 依赖 Vec.extend——VM 补齐后应可运行（管道闭包语法）。
    let src = r#"
        fn flat_map(items: Vec, f) -> Vec {
            let mut result = Vec::new();
            for i in 0..items.len() {
                let mapped = f(items.get(i));
                result.extend(mapped);
            };
            result
        }
        fn main() -> i64 {
            let f = |x| [x, x * 10];
            let r = flat_map([1, 2], f);
            r.len() * 1000 + r.get(0) * 100 + r.get(1) * 10 + r.get(2)
        }
    "#;
    // [1,2] → f 展开 [1,10,2,20] → len=4 → 4*1000 + 1*100 + 10*10 + 2 = 4202
    let vm = run_plain_vm(src).unwrap();
    assert_eq!(as_int(&vm), 4202, "VM flat_map 应得 4202，实际 {:?}", vm);
}

#[test]
fn vm_map_values_entries_usable() {
    // map_values 依赖 Map.entries + Vec.get（管道闭包语法）。
    let src = r#"
        fn map_values(m: HashMap, f) -> HashMap {
            let entries = m.entries();
            let mut result = HashMap::new();
            for i in 0..entries.len() {
                let pair = entries.get(i);
                result.insert(pair.get(0), f(pair.get(1)));
            };
            result
        }
        fn main() -> i64 {
            let mut m = HashMap::new();
            m.insert("a", 1);
            m.insert("b", 2);
            let r = map_values(m, |x| x * 10);
            r.get("a") + r.get("b")
        }
    "#;
    let vm = run_plain_vm(src).unwrap();
    assert_eq!(as_int(&vm), 30, "VM map_values 应得 30，实际 {:?}", vm);
}
