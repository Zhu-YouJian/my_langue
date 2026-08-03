//! Host trampolines — the bridge between JIT-compiled code and the Rust `Vm`.
//!
//! All trampolines write results through `*mut Value` out-pointers rather than
//! returning `Value` by value, because `Value` is a 32+ byte enum containing
//! `Rc`/`Vec`/`String` and cannot be passed through Cranelift's native return
//! slot safely.
//!
//! Convention:
//! - `vm: *mut Vm` — the VM context.
//! - Input `Value`s are passed as `*const Value` (read-only).
//! - Output `Value`s are written through `*mut Value`.
//! - Errors set `vm.last_error` and write `Value::Unit` to the out-pointer.
//! - All trampolines are `extern "C"` (no unwinding across FFI).

use std::cell::RefCell;
use crate::hir::types::BaseType;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::rc::Rc;
use crate::runtime::value::Value;
use crate::runtime::vm::Vm;

/// 单次 hostcall 可接受的最多参数个数。超过即视为翻译器/调用方异常，
/// 直接拒绝执行 `from_raw_parts`，避免 `count * 2` 类运算溢出后造成 UB。
const MAX_HOSTCALL_ARGS: usize = 1 << 20; // 1_048_576

/// Invoke a compiled JIT function pointer.
///
/// # Safety
/// `fn_ptr` must point to a valid function with the signature
/// `extern "C" fn(*mut Vm, *const Value, usize, *mut Value) -> bool`.
///
/// 实现用 `catch_unwind` 包裹 JIT 调用，防止 hostcall 内部 panic 跨 FFI 边界
/// （跨 FFI unwind 是 UB）。若捕获到 panic，将消息写入 `vm.last_error` 并返回 `false`。
pub unsafe fn invoke_jit(
    fn_ptr: unsafe extern "C" fn(*mut Vm, *const Value, usize, *mut Value) -> bool,
    vm: *mut Vm,
    args: &[Value],
    out: &mut Value,
) -> bool {
    // SAFETY: 调用方保证 fn_ptr 来自合法 JIT 模块；vm 非空且未被移动。
    // catch_unwind 用于防止 hostcall panic 跨 FFI 边界。
    let result = catch_unwind(AssertUnwindSafe(|| {
        fn_ptr(vm, args.as_ptr(), args.len(), out as *mut Value)
    }));
    match result {
        Ok(ok) => ok,
        Err(payload) => {
            // 不要让 panic 跨 FFI 边界；写入错误后返回 false
            let msg = if let Some(s) = payload.downcast_ref::<&'static str>() {
                (*s).to_string()
            } else if let Some(s) = payload.downcast_ref::<String>() {
                s.clone()
            } else {
                "JIT hostcall panic (non-string payload)".to_string()
            };
            if !vm.is_null() {
                (*vm).set_last_error(format!("JIT panic: {}", msg));
            }
            std::ptr::write(out, Value::Unit);
            false
        }
    }
}

/// 安全地从裸指针构造切片。若 `count` 超过 `MAX_HOSTCALL_ARGS` 或 `ptr` 为空，
/// 返回空切片（调用方应已写好错误处理路径）。
///
/// 这是 JIT hostcall 中所有 `from_raw_parts` 的统一闸门。
unsafe fn safe_slice<'a>(ptr: *const Value, count: u64) -> &'a [Value] {
    if ptr.is_null() {
        return &[];
    }
    let n = match count {
        0 => return &[],
        c if c as usize > MAX_HOSTCALL_ARGS => return &[],
        c => c as usize,
    };
    std::slice::from_raw_parts(ptr, n)
}

// ── Value construction trampolines ─────────────────────────────────────────

unsafe extern "C" fn host_make_int(_vm: *mut Vm, n: i64, out: *mut Value) {
    std::ptr::write(out, Value::Int(n, BaseType::I32));
}

unsafe extern "C" fn host_make_float(_vm: *mut Vm, f: f64, out: *mut Value) {
    std::ptr::write(out, Value::Float(f));
}

/// 真正的 f32 hostcall：保留 dtype 信息到运行时（不再降级为 f64）。
/// 阶段 6（f32/f64 parity roadmap）补齐。
unsafe extern "C" fn host_make_float32(_vm: *mut Vm, f: f32, out: *mut Value) {
    std::ptr::write(out, Value::Float32(f));
}

unsafe extern "C" fn host_make_bool(_vm: *mut Vm, b: u8, out: *mut Value) {
    std::ptr::write(out, Value::Bool(b != 0));
}

unsafe extern "C" fn host_make_str(vm: *mut Vm, idx: u64, out: *mut Value) {
    let vm = &mut *vm;
    let s = vm.string_at(idx as usize).unwrap_or_default();
    std::ptr::write(out, Value::String(s));
}

unsafe extern "C" fn host_make_unit(_vm: *mut Vm, out: *mut Value) {
    std::ptr::write(out, Value::Unit);
}

/// Extract truthiness of a `Value` as a `u8` (1 = true, 0 = false).
unsafe extern "C" fn host_truthy(_vm: *mut Vm, v: *const Value) -> u8 {
    let v = &*v;
    match v {
        Value::Bool(b) => *b as u8,
        Value::Int(i, _) => (*i != 0) as u8,
        Value::Float(f) => (*f != 0.0) as u8,
        Value::Unit => 0,
        Value::String(s) => (!s.is_empty()) as u8,
        _ => 1, // any heap-allocated value is truthy
    }
}

// ── Arithmetic trampolines ─────────────────────────────────────────────────

unsafe extern "C" fn host_add(vm: *mut Vm, a: *const Value, b: *const Value, out: *mut Value) {
    let vm = &mut *vm;
    match vm.add(&*a, &*b) {
        Ok(v) => std::ptr::write(out, v),
        Err(e) => { vm.set_jit_error(&e); std::ptr::write(out, Value::Unit); }
    }
}

unsafe extern "C" fn host_sub(vm: *mut Vm, a: *const Value, b: *const Value, out: *mut Value) {
    let vm = &mut *vm;
    match vm.sub(&*a, &*b) {
        Ok(v) => std::ptr::write(out, v),
        Err(e) => { vm.set_jit_error(&e); std::ptr::write(out, Value::Unit); }
    }
}

unsafe extern "C" fn host_mul(vm: *mut Vm, a: *const Value, b: *const Value, out: *mut Value) {
    let vm = &mut *vm;
    match vm.mul(&*a, &*b) {
        Ok(v) => std::ptr::write(out, v),
        Err(e) => { vm.set_jit_error(&e); std::ptr::write(out, Value::Unit); }
    }
}

unsafe extern "C" fn host_div(vm: *mut Vm, a: *const Value, b: *const Value, out: *mut Value) {
    let vm = &mut *vm;
    match vm.div(&*a, &*b) {
        Ok(v) => std::ptr::write(out, v),
        Err(e) => { vm.set_jit_error(&e); std::ptr::write(out, Value::Unit); }
    }
}

unsafe extern "C" fn host_mod(vm: *mut Vm, a: *const Value, b: *const Value, out: *mut Value) {
    let vm = &mut *vm;
    match vm.rem(&*a, &*b) {
        Ok(v) => std::ptr::write(out, v),
        Err(e) => { vm.set_jit_error(&e); std::ptr::write(out, Value::Unit); }
    }
}

unsafe extern "C" fn host_neg(vm: *mut Vm, a: *const Value, out: *mut Value) {
    let vm = &mut *vm;
    match vm.neg(&*a) {
        Ok(v) => std::ptr::write(out, v),
        Err(e) => { vm.set_jit_error(&e); std::ptr::write(out, Value::Unit); }
    }
}

unsafe extern "C" fn host_not(vm: *mut Vm, a: *const Value, out: *mut Value) {
    let vm = &mut *vm;
    match vm.not(&*a) {
        Ok(v) => std::ptr::write(out, v),
        Err(e) => { vm.set_jit_error(&e); std::ptr::write(out, Value::Unit); }
    }
}

// ── Comparison trampolines ─────────────────────────────────────────────────

unsafe extern "C" fn host_eq(vm: *mut Vm, a: *const Value, b: *const Value, out: *mut Value) {
    let vm = &mut *vm;
    match vm.eq(&*a, &*b) {
        Ok(v) => std::ptr::write(out, v),
        Err(e) => { vm.set_jit_error(&e); std::ptr::write(out, Value::Unit); }
    }
}

unsafe extern "C" fn host_neq(vm: *mut Vm, a: *const Value, b: *const Value, out: *mut Value) {
    let vm = &mut *vm;
    match vm.neq(&*a, &*b) {
        Ok(v) => std::ptr::write(out, v),
        Err(e) => { vm.set_jit_error(&e); std::ptr::write(out, Value::Unit); }
    }
}

unsafe extern "C" fn host_lt(vm: *mut Vm, a: *const Value, b: *const Value, out: *mut Value) {
    let vm = &mut *vm;
    match vm.lt(&*a, &*b) {
        Ok(v) => std::ptr::write(out, v),
        Err(e) => { vm.set_jit_error(&e); std::ptr::write(out, Value::Unit); }
    }
}

unsafe extern "C" fn host_gt(vm: *mut Vm, a: *const Value, b: *const Value, out: *mut Value) {
    let vm = &mut *vm;
    match vm.gt(&*a, &*b) {
        Ok(v) => std::ptr::write(out, v),
        Err(e) => { vm.set_jit_error(&e); std::ptr::write(out, Value::Unit); }
    }
}

unsafe extern "C" fn host_lte(vm: *mut Vm, a: *const Value, b: *const Value, out: *mut Value) {
    let vm = &mut *vm;
    match vm.lte(&*a, &*b) {
        Ok(v) => std::ptr::write(out, v),
        Err(e) => { vm.set_jit_error(&e); std::ptr::write(out, Value::Unit); }
    }
}

unsafe extern "C" fn host_gte(vm: *mut Vm, a: *const Value, b: *const Value, out: *mut Value) {
    let vm = &mut *vm;
    match vm.gte(&*a, &*b) {
        Ok(v) => std::ptr::write(out, v),
        Err(e) => { vm.set_jit_error(&e); std::ptr::write(out, Value::Unit); }
    }
}

// ── Call trampolines ───────────────────────────────────────────────────────

unsafe extern "C" fn host_call(
    vm: *mut Vm, name_idx: u64, arg_count: u64, args_ptr: *const Value, out: *mut Value,
) {
    let vm = &mut *vm;
    let name = vm.string_at(name_idx as usize).unwrap_or_default();
    let args = safe_slice(args_ptr, arg_count);
    match vm.call_with_args(&name, args) {
        Ok(v) => std::ptr::write(out, v),
        Err(e) => { vm.set_jit_error(&e); std::ptr::write(out, Value::Unit); }
    }
}

/// A1：JIT-to-JIT 直接调用的慢路径 trampoline（目标函数尚未编译时）。
///
/// 语义与 `host_call` 完全一致（native / globals-FnRef 别名 / 用户函数的完整
/// 解析，错误走 `set_jit_error` 带行号），但对**用户函数 chunk 且可 JIT 编译**
/// 的目标 → 编译并**直接调用 JIT 机器码**（不再逃逸解释器）；编译失败 → 回退
/// 解释器（正确性优先，静默错值红线：任何失败都显式 set_jit_error 或走原语义）。
///
/// 快速路径（目标已编译）由 translator 的 `emit_direct_call` 直接 `call_indirect`，
/// 不经过本 trampoline；本 trampoline 只承担「首次遇到未编译函数」的编译注册。
unsafe extern "C" fn host_jit_call(
    vm: *mut Vm, name_idx: u64, arg_count: u64, args_ptr: *const Value, out: *mut Value,
) {
    let vm = &mut *vm;
    let name = vm.string_at(name_idx as usize).unwrap_or_default();
    let args = safe_slice(args_ptr, arg_count);
    match vm.jit_call_chunk(&name, args) {
        Ok(v) => std::ptr::write(out, v),
        Err(e) => { vm.set_jit_error(&e); std::ptr::write(out, Value::Unit); }
    }
}

unsafe extern "C" fn host_method_call(
    vm: *mut Vm, name_idx: u64, arg_count: u64, args_ptr: *const Value, out: *mut Value,
) {
    let vm = &mut *vm;
    let method = vm.string_at(name_idx as usize).unwrap_or_default();
    let all = safe_slice(args_ptr, arg_count);
    if all.is_empty() {
        vm.set_last_error("MethodCall: missing receiver".into());
        std::ptr::write(out, Value::Unit);
        return;
    }
    let receiver = all[0].clone();
    let args = &all[1..];
    match vm.call_method(&receiver, &method, args) {
        Ok(v) => std::ptr::write(out, v),
        Err(e) => { vm.set_jit_error(&e); std::ptr::write(out, Value::Unit); }
    }
}

// ── Heap-allocating trampolines ────────────────────────────────────────────

unsafe extern "C" fn host_make_vec(_vm: *mut Vm, count: u64, args_ptr: *const Value, out: *mut Value) {
    let items = safe_slice(args_ptr, count).to_vec();
    std::ptr::write(out, Value::Vec(Rc::new(RefCell::new(items))));
}

unsafe extern "C" fn host_make_map(_vm: *mut Vm, count: u64, args_ptr: *const Value, out: *mut Value) {
    // 安全：用 checked_mul 防止 count * 2 溢出（count = u64::MAX/2+1 时会回绕）
    let pair_count = match (count as usize).checked_mul(2) {
        Some(n) if n <= MAX_HOSTCALL_ARGS * 2 => n,
        _ => {
            std::ptr::write(out, Value::Map(Rc::new(RefCell::new(std::collections::HashMap::new()))));
            return;
        }
    };
    let flat = if args_ptr.is_null() || pair_count == 0 {
        &[]
    } else {
        std::slice::from_raw_parts(args_ptr, pair_count)
    };
    let mut map = std::collections::HashMap::new();
    let mut i = 0;
    while i + 1 < flat.len() {
        if let Value::String(k) = &flat[i] {
            map.insert(k.clone(), flat[i + 1].clone());
        }
        i += 2;
    }
    std::ptr::write(out, Value::Map(Rc::new(RefCell::new(map))));
}

unsafe extern "C" fn host_new_struct(
    vm: *mut Vm, name_idx: u64, field_count: u64, args_ptr: *const Value, out: *mut Value,
) {
    let vm = &mut *vm;
    let name = vm.string_at(name_idx as usize).unwrap_or_default();
    // 安全：field_count * 2 用 checked_mul
    let flat_len = match (field_count as usize).checked_mul(2) {
        Some(n) if n <= MAX_HOSTCALL_ARGS * 2 => n,
        _ => {
            vm.set_last_error("new_struct: field_count 过大".into());
            std::ptr::write(out, Value::Unit);
            return;
        }
    };
    let flat = if args_ptr.is_null() || flat_len == 0 {
        &[][..]
    } else {
        std::slice::from_raw_parts(args_ptr, flat_len)
    };
    // 字节码约定：StructLiteral 对每个字段压 [value, name]（name 在顶），
    // 与 VM 的 NewStruct（先 pop name 再 pop value）一致。flat 布局为
    // [v_N, n_N, v_{N-1}, n_{N-1}, ..., v_1, n_1]（N=字段数，1=源码首个字段）。
    // 因此从 flat 末尾向前每 2 个一组取 (n, v)，得到的即是源码声明顺序。
    // 修复前按 [name, value] 正向配对，导致字段名与值整体颠倒
    // （如 ("hello", "path")），JIT 路径构造带字段 struct 时字段访问全错。
    let mut fields = Vec::with_capacity(field_count as usize);
    let mut i = flat.len();
    while i >= 2 {
        let fname = match &flat[i - 1] { Value::String(s) => s.clone(), _ => format!("f{}", (i - 2) / 2) };
        let val = flat[i - 2].clone();
        fields.push((fname, val));
        i -= 2;
    }
    std::ptr::write(out, Value::Struct { name, fields: Rc::new(RefCell::new(fields)) });
}

// M1.2：union 构造 hostcall — 与 VM 的 NewUnion 一致：
// 栈顶单个 value → Value::Union { name, active_field, value }
unsafe extern "C" fn host_new_union(
    vm: *mut Vm, name_idx: u64, field_idx: u64, val_ptr: *const Value, out: *mut Value,
) {
    let vm = &mut *vm;
    let name = vm.string_at(name_idx as usize).unwrap_or_default();
    let active_field = vm.string_at(field_idx as usize).unwrap_or_default();
    let value = (*val_ptr).clone();
    std::ptr::write(out, Value::Union { name, active_field, value: Box::new(value) });
}

unsafe extern "C" fn host_load_field(vm: *mut Vm, field_idx: u64, recv: *const Value, out: *mut Value) {
    let vm = &mut *vm;
    let fname = vm.string_at(field_idx as usize).unwrap_or_default();
    match &*recv {
        Value::Struct { fields, .. } => {
            let fields = fields.borrow();
            std::ptr::write(out, fields.iter()
                .find(|(n, _)| n == &fname)
                .map(|(_, v)| v.clone())
                .unwrap_or(Value::Unit));
        }
        // M1.2：Union 字段访问（tagged union）——只读当前 active 字段
        Value::Union { active_field, value, .. } => {
            if active_field == &fname {
                std::ptr::write(out, (**value).clone());
            } else {
                std::ptr::write(out, Value::Unit);
            }
        }
        _ => std::ptr::write(out, Value::Unit),
    }
}

unsafe extern "C" fn host_store_field(vm: *mut Vm, field_idx: u64, recv: *mut Value, val: *const Value, out: *mut Value) {
    let vm = &mut *vm;
    let fname = vm.string_at(field_idx as usize).unwrap_or_default();
    match &*recv {
        Value::Struct { fields, .. } => {
            let mut fields = fields.borrow_mut();
            if let Some(slot) = fields.iter_mut().find(|(n, _)| n == &fname) {
                slot.1 = (*val).clone();
            }
            std::ptr::write(out, (*recv).clone());
        }
        // M1.2：Union 字段修改（tagged union）——只允许修改 active 字段，
        // 写回新构造的 Union（bytecode 对 Union 目标随后 Store 写回变量槽）。
        Value::Union { name, active_field, .. } => {
            if active_field == &fname {
                std::ptr::write(out, Value::Union {
                    name: name.clone(),
                    active_field: active_field.clone(),
                    value: Box::new((*val).clone()),
                });
            } else {
                std::ptr::write(out, (*recv).clone());
            }
        }
        _ => std::ptr::write(out, (*recv).clone()),
    }
}

unsafe extern "C" fn host_index_get(vm: *mut Vm, target: *const Value, idx: *const Value, out: *mut Value) {
    let vm = &mut *vm;
    match vm.index_get(&*target, &*idx) {
        Ok(v) => std::ptr::write(out, v),
        Err(e) => { vm.set_jit_error(&e); std::ptr::write(out, Value::Unit); }
    }
}

unsafe extern "C" fn host_slice_str(vm: *mut Vm, target: *const Value, start: *const Value, end: *const Value, out: *mut Value) {
    let vm = &mut *vm;
    match vm.slice_str(&*target, &*start, &*end) {
        Ok(v) => std::ptr::write(out, v),
        Err(e) => { vm.set_jit_error(&e); std::ptr::write(out, Value::Unit); }
    }
}

unsafe extern "C" fn host_make_enum(
    vm: *mut Vm, name_idx: u64, variant_idx: u64, field_count: u64, args_ptr: *const Value, out: *mut Value,
) {
    let vm = &mut *vm;
    let enum_name = vm.string_at(name_idx as usize).unwrap_or_default();
    let variant = vm.string_at(variant_idx as usize).unwrap_or_default();
    // 安全：field_count * 2 用 checked_mul，上限同 host_new_struct
    let flat_len = match (field_count as usize).checked_mul(2) {
        Some(n) if n <= MAX_HOSTCALL_ARGS * 2 => n,
        _ => {
            vm.set_last_error("make_enum: field_count 过大".into());
            std::ptr::write(out, Value::Unit);
            return;
        }
    };
    let flat = if args_ptr.is_null() || flat_len == 0 {
        &[][..]
    } else {
        std::slice::from_raw_parts(args_ptr, flat_len)
    };
    // 字节码约定：EnumLiteral 对每个字段压 [value, name]（name 在顶），
    // 与 VM 的 MakeEnum（先 pop name 再 pop value）一致。flat 布局为
    // [v_N, n_N, v_{N-1}, n_{N-1}, ..., v_1, n_1]（N=字段数，1=源码首个字段）。
    // 因此从 flat 末尾向前每 2 个一组取 (n, v)，得到的即是源码声明顺序。
    // 修复前：translator 只弹 field_count 个槽（实际压了 2×field_count），
    // 且本函数按位置赋名 _0.. 忽略压入的字段名，导致字段名/值错位
    // （如 Result::Ok(42) 的 _0 字段值被取成字符串 "_0"）。
    // 与已修复的 host_new_struct 字段序 bug（typestate 阶段）同源。
    let mut fields = Vec::with_capacity(field_count as usize);
    let mut i = flat.len();
    while i >= 2 {
        let fname = match &flat[i - 1] { Value::String(s) => s.clone(), _ => format!("_{}", fields.len()) };
        let val = flat[i - 2].clone();
        fields.push((fname, val));
        i -= 2;
    }
    std::ptr::write(out, Value::Enum { enum_name, variant, fields: Rc::new(RefCell::new(fields)) });
}

unsafe extern "C" fn host_is_enum_variant(vm: *mut Vm, variant_idx: u64, recv: *const Value, out: *mut Value) {
    let vm = &mut *vm;
    let target = vm.string_at(variant_idx as usize).unwrap_or_default();
    match &*recv {
        Value::Enum { variant, .. } => std::ptr::write(out, Value::Bool(variant == &target)),
        _ => std::ptr::write(out, Value::Bool(false)),
    }
}

unsafe extern "C" fn host_enum_get_field(vm: *mut Vm, field_idx: u64, recv: *const Value, out: *mut Value) {
    let vm = &mut *vm;
    let fname = vm.string_at(field_idx as usize).unwrap_or_default();
    match &*recv {
        Value::Enum { fields, .. } => {
            let fields = fields.borrow();
            std::ptr::write(out, fields.iter()
                .find(|(n, _)| n == &fname)
                .map(|(_, v)| v.clone())
                .unwrap_or_else(|| fields.get(0).map(|(_, v)| v.clone()).unwrap_or(Value::Unit)));
        }
        _ => std::ptr::write(out, Value::Unit),
    }
}

unsafe extern "C" fn host_push_range(_vm: *mut Vm, start: i64, end: i64, inclusive: u8, out: *mut Value) {
    std::ptr::write(out, Value::Range { start, end, inclusive: inclusive != 0 });
}

unsafe extern "C" fn host_make_tensor(
    _vm: *mut Vm, rows: u64, cols: u64, args_ptr: *const Value, out: *mut Value,
) {
    // 安全：rows * cols 用 checked_mul 防止溢出，并设上限防止 OOM
    let count = match (rows as usize).checked_mul(cols as usize) {
        Some(n) if n <= MAX_HOSTCALL_ARGS => n,
        _ => {
            std::ptr::write(out, Value::Unit);
            return;
        }
    };
    let flat = if args_ptr.is_null() || count == 0 {
        &[][..]
    } else {
        std::slice::from_raw_parts(args_ptr, count)
    };
    let data: Vec<f64> = flat.iter().map(|v| match v {
        Value::Float(f) => *f,
        Value::Float32(f) => *f as f64,
        Value::Int(i, _) => *i as f64,
        _ => 0.0,
    }).collect();
    use crate::runtime::tensor::Tensor;
    let shape = if cols == 0 { vec![rows as usize] } else { vec![rows as usize, cols as usize] };
    std::ptr::write(out, Value::Tensor(Rc::new(RefCell::new(Tensor::from_vec(data, shape)))));
}

/// f32 Tensor 构造：保留 dtype=F32，元素提取保持 f32 精度。
/// 阶段 6（f32/f64 parity roadmap）补齐。
unsafe extern "C" fn host_make_tensor_f32(
    _vm: *mut Vm, rows: u64, cols: u64, args_ptr: *const Value, out: *mut Value,
) {
    // 安全：rows * cols 用 checked_mul 防止溢出，并设上限防止 OOM
    let count = match (rows as usize).checked_mul(cols as usize) {
        Some(n) if n <= MAX_HOSTCALL_ARGS => n,
        _ => {
            std::ptr::write(out, Value::Unit);
            return;
        }
    };
    let flat = if args_ptr.is_null() || count == 0 {
        &[][..]
    } else {
        std::slice::from_raw_parts(args_ptr, count)
    };
    let data: Vec<f32> = flat.iter().map(|v| match v {
        Value::Float32(f) => *f,
        Value::Float(f) => *f as f32,
        Value::Int(i, _) => *i as f32,
        _ => 0.0,
    }).collect();
    use crate::runtime::tensor::Tensor;
    let shape = if cols == 0 { vec![rows as usize] } else { vec![rows as usize, cols as usize] };
    std::ptr::write(out, Value::Tensor(Rc::new(RefCell::new(Tensor::from_vec_f32(data, shape)))));
}

/// f16 Tensor 构造：保留 dtype=F16，元素从栈上 Value 转换为 f16。
/// VM Value 无 F16 变体，栈上元素以 f64/f32 形式存在，此处转换为 f16。
/// Phase 2 缺口 1：F16/BF16 JIT 路径补齐。
unsafe extern "C" fn host_make_tensor_f16(
    _vm: *mut Vm, rows: u64, cols: u64, args_ptr: *const Value, out: *mut Value,
) {
    // 安全：rows * cols 用 checked_mul 防止溢出，并设上限防止 OOM
    let count = match (rows as usize).checked_mul(cols as usize) {
        Some(n) if n <= MAX_HOSTCALL_ARGS => n,
        _ => {
            std::ptr::write(out, Value::Unit);
            return;
        }
    };
    let flat = if args_ptr.is_null() || count == 0 {
        &[][..]
    } else {
        std::slice::from_raw_parts(args_ptr, count)
    };
    use half::f16;
    let data: Vec<f16> = flat.iter().map(|v| match v {
        Value::Float(f) => f16::from_f64(*f),
        Value::Float32(f) => f16::from_f32(*f),
        Value::Int(i, _) => f16::from_f64(*i as f64),
        _ => f16::from_f32(0.0),
    }).collect();
    use crate::runtime::tensor::Tensor;
    let shape = if cols == 0 { vec![rows as usize] } else { vec![rows as usize, cols as usize] };
    std::ptr::write(out, Value::Tensor(Rc::new(RefCell::new(Tensor::from_vec_f16(data, shape)))));
}

/// bf16 Tensor 构造：保留 dtype=BF16，元素从栈上 Value 转换为 bf16。
/// Phase 2 缺口 1：F16/BF16 JIT 路径补齐。
unsafe extern "C" fn host_make_tensor_bf16(
    _vm: *mut Vm, rows: u64, cols: u64, args_ptr: *const Value, out: *mut Value,
) {
    // 安全：rows * cols 用 checked_mul 防止溢出，并设上限防止 OOM
    let count = match (rows as usize).checked_mul(cols as usize) {
        Some(n) if n <= MAX_HOSTCALL_ARGS => n,
        _ => {
            std::ptr::write(out, Value::Unit);
            return;
        }
    };
    let flat = if args_ptr.is_null() || count == 0 {
        &[][..]
    } else {
        std::slice::from_raw_parts(args_ptr, count)
    };
    use half::bf16;
    let data: Vec<bf16> = flat.iter().map(|v| match v {
        Value::Float(f) => bf16::from_f64(*f),
        Value::Float32(f) => bf16::from_f32(*f),
        Value::Int(i, _) => bf16::from_f64(*i as f64),
        _ => bf16::from_f32(0.0),
    }).collect();
    use crate::runtime::tensor::Tensor;
    let shape = if cols == 0 { vec![rows as usize] } else { vec![rows as usize, cols as usize] };
    std::ptr::write(out, Value::Tensor(Rc::new(RefCell::new(Tensor::from_vec_bf16(data, shape)))));
}

unsafe extern "C" fn host_make_closure(
    vm: *mut Vm, params: u64, captures_count: u64, name_idx: u64, args_ptr: *const Value, out: *mut Value,
) {
    let vm = &mut *vm;
    // a1 P1：第二个操作数在 bytecode 里是「字符串表索引」（与 VM opcode 44 MakeClosure 对齐），
    // 不是 chunk 位置索引。此前误用 `chunk_name_at`（chunk_names 表），两张表只在巧合下相等，
    // 多闭包程序会得到错误的名字（调错函数）。改用 `string_at` 从当前 chunk 的字符串表取闭包 chunk 名。
    let name = vm.string_at(name_idx as usize).unwrap_or_default();
    // a1 P3：捕获值内联——从 JIT 栈上取 captures_count 个值装入 FnRef.captures（值内联，
    // 与 VM opcode 44 对齐）。顺序与闭包 chunk 捕获槽（params..params+captures）一致。
    let captures = if args_ptr.is_null() || captures_count == 0 {
        vec![]
    } else {
        std::slice::from_raw_parts(args_ptr, captures_count as usize).to_vec()
    };
    std::ptr::write(out, Value::FnRef {
        name,
        params: vec![],
        return_type: crate::hir::types::Type::Base(crate::hir::types::BaseType::Unit),
        captures,
    });
}

/// a1 P1：间接调用栈上闭包/函数值（Op::CallClosure 的 JIT hostcall）。
/// 栈布局 [arg1..argN, callee]（arg_count = N+1 个值，最后一个为 callee）。
/// 走 `Vm::call_value`（FnRef 按名 → natives/globals-FnRef/functions；其他值报「期望可调用值」）。
/// 错误写入 last_error 并写 Unit（翻译器随后紧跟 host_check_error 立即中断，B2 模式）。
unsafe extern "C" fn host_call_indirect(
    vm: *mut Vm, arg_count: u64, args_ptr: *const Value, out: *mut Value,
) {
    let vm = &mut *vm;
    let all = safe_slice(args_ptr, arg_count);
    if all.is_empty() {
        vm.set_last_error("CallClosure: 缺少 callee".into());
        std::ptr::write(out, Value::Unit);
        return;
    }
    let n = all.len();
    let callee = all[n - 1].clone();
    let args = &all[..n - 1];
    match vm.call_value(&callee, args) {
        Ok(v) => std::ptr::write(out, v),
        Err(e) => { vm.set_jit_error(&e); std::ptr::write(out, Value::Unit); }
    }
}

unsafe extern "C" fn host_load_global(vm: *mut Vm, name_idx: u64, out: *mut Value) {
    let vm = &mut *vm;
    let name = vm.string_at(name_idx as usize).unwrap_or_default();
    // 9a：native 别名 fallback（与 VM opcode 9 LoadGlobal 对齐）——`let p = println; p("x")`
    // 在 JIT 路径也能把 native 名作为 FnRef 可调用值绑定（call_value 支持 natives 查询）。
    // 用户全局名优先（shadow native）。
    let v = vm.get_global(&name).unwrap_or_else(|| {
        if vm.natives.contains_key(&name) {
            Value::FnRef {
                name,
                params: Vec::new(),
                return_type: crate::hir::types::Type::Unknown,
                captures: Vec::new(),
            }
        } else {
            Value::Unit
        }
    });
    std::ptr::write(out, v);
}

unsafe extern "C" fn host_store_global(vm: *mut Vm, name_idx: u64, val: *const Value, out: *mut Value) {
    let vm = &mut *vm;
    let name = vm.string_at(name_idx as usize).unwrap_or_default();
    let v = (*val).clone();
    vm.set_global(name, v.clone());
    std::ptr::write(out, v);
}

/// 检查 VM 是否有未处理的错误（如 matmul shape mismatch）。
/// 返回 1 表示有错误（JIT 应提前中止并返回 false），0 表示无错误。
/// 不清除错误——`run_jit` 在 JIT 返回 false 后通过 `take_last_error` 取走。
unsafe extern "C" fn host_check_error(vm: *mut Vm) -> u8 {
    let vm = &mut *vm;
    if vm.has_last_error() { 1 } else { 0 }
}

// ── Symbol table ───────────────────────────────────────────────────────────

pub fn hostcall_addr(name: &str) -> Option<usize> {
    let map: &[(&str, usize)] = &[
        ("host_make_int", host_make_int as usize),
        ("host_make_float", host_make_float as usize),
        ("host_make_float32", host_make_float32 as usize),
        ("host_make_bool", host_make_bool as usize),
        ("host_make_str", host_make_str as usize),
        ("host_make_unit", host_make_unit as usize),
        ("host_truthy", host_truthy as usize),
        ("host_add", host_add as usize),
        ("host_sub", host_sub as usize),
        ("host_mul", host_mul as usize),
        ("host_div", host_div as usize),
        ("host_mod", host_mod as usize),
        ("host_neg", host_neg as usize),
        ("host_not", host_not as usize),
        ("host_eq", host_eq as usize),
        ("host_neq", host_neq as usize),
        ("host_lt", host_lt as usize),
        ("host_gt", host_gt as usize),
        ("host_lte", host_lte as usize),
        ("host_gte", host_gte as usize),
        ("host_call", host_call as usize),
        ("host_jit_call", host_jit_call as usize),
        ("host_call_indirect", host_call_indirect as usize),
        ("host_method_call", host_method_call as usize),
        ("host_make_vec", host_make_vec as usize),
        ("host_make_map", host_make_map as usize),
        ("host_new_struct", host_new_struct as usize),
        ("host_new_union", host_new_union as usize),
        ("host_load_field", host_load_field as usize),
        ("host_store_field", host_store_field as usize),
        ("host_index_get", host_index_get as usize),
        ("host_slice_str", host_slice_str as usize),
        ("host_make_enum", host_make_enum as usize),
        ("host_is_enum_variant", host_is_enum_variant as usize),
        ("host_enum_get_field", host_enum_get_field as usize),
        ("host_push_range", host_push_range as usize),
        ("host_make_tensor", host_make_tensor as usize),
        ("host_make_tensor_f32", host_make_tensor_f32 as usize),
        ("host_make_tensor_f16", host_make_tensor_f16 as usize),
        ("host_make_tensor_bf16", host_make_tensor_bf16 as usize),
        ("host_make_closure", host_make_closure as usize),
        ("host_load_global", host_load_global as usize),
        ("host_store_global", host_store_global as usize),
        ("host_check_error", host_check_error as usize),
    ];
    map.iter().find(|(n, _)| *n == name).map(|(_, a)| *a)
}
