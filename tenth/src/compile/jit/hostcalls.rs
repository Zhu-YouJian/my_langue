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
    std::ptr::write(out, Value::Int(n));
}

unsafe extern "C" fn host_make_float(_vm: *mut Vm, f: f64, out: *mut Value) {
    std::ptr::write(out, Value::Float(f));
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
        Value::Int(i) => (*i != 0) as u8,
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
        Err(e) => { vm.set_last_error(e.to_string()); std::ptr::write(out, Value::Unit); }
    }
}

unsafe extern "C" fn host_sub(vm: *mut Vm, a: *const Value, b: *const Value, out: *mut Value) {
    let vm = &mut *vm;
    match vm.sub(&*a, &*b) {
        Ok(v) => std::ptr::write(out, v),
        Err(e) => { vm.set_last_error(e.to_string()); std::ptr::write(out, Value::Unit); }
    }
}

unsafe extern "C" fn host_mul(vm: *mut Vm, a: *const Value, b: *const Value, out: *mut Value) {
    let vm = &mut *vm;
    match vm.mul(&*a, &*b) {
        Ok(v) => std::ptr::write(out, v),
        Err(e) => { vm.set_last_error(e.to_string()); std::ptr::write(out, Value::Unit); }
    }
}

unsafe extern "C" fn host_div(vm: *mut Vm, a: *const Value, b: *const Value, out: *mut Value) {
    let vm = &mut *vm;
    match vm.div(&*a, &*b) {
        Ok(v) => std::ptr::write(out, v),
        Err(e) => { vm.set_last_error(e.to_string()); std::ptr::write(out, Value::Unit); }
    }
}

unsafe extern "C" fn host_mod(vm: *mut Vm, a: *const Value, b: *const Value, out: *mut Value) {
    let vm = &mut *vm;
    match vm.rem(&*a, &*b) {
        Ok(v) => std::ptr::write(out, v),
        Err(e) => { vm.set_last_error(e.to_string()); std::ptr::write(out, Value::Unit); }
    }
}

unsafe extern "C" fn host_neg(vm: *mut Vm, a: *const Value, out: *mut Value) {
    let vm = &mut *vm;
    match vm.neg(&*a) {
        Ok(v) => std::ptr::write(out, v),
        Err(e) => { vm.set_last_error(e.to_string()); std::ptr::write(out, Value::Unit); }
    }
}

unsafe extern "C" fn host_not(vm: *mut Vm, a: *const Value, out: *mut Value) {
    let vm = &mut *vm;
    match vm.not(&*a) {
        Ok(v) => std::ptr::write(out, v),
        Err(e) => { vm.set_last_error(e.to_string()); std::ptr::write(out, Value::Unit); }
    }
}

// ── Comparison trampolines ─────────────────────────────────────────────────

unsafe extern "C" fn host_eq(vm: *mut Vm, a: *const Value, b: *const Value, out: *mut Value) {
    let vm = &mut *vm;
    match vm.eq(&*a, &*b) {
        Ok(v) => std::ptr::write(out, v),
        Err(e) => { vm.set_last_error(e.to_string()); std::ptr::write(out, Value::Unit); }
    }
}

unsafe extern "C" fn host_neq(vm: *mut Vm, a: *const Value, b: *const Value, out: *mut Value) {
    let vm = &mut *vm;
    match vm.neq(&*a, &*b) {
        Ok(v) => std::ptr::write(out, v),
        Err(e) => { vm.set_last_error(e.to_string()); std::ptr::write(out, Value::Unit); }
    }
}

unsafe extern "C" fn host_lt(vm: *mut Vm, a: *const Value, b: *const Value, out: *mut Value) {
    let vm = &mut *vm;
    match vm.lt(&*a, &*b) {
        Ok(v) => std::ptr::write(out, v),
        Err(e) => { vm.set_last_error(e.to_string()); std::ptr::write(out, Value::Unit); }
    }
}

unsafe extern "C" fn host_gt(vm: *mut Vm, a: *const Value, b: *const Value, out: *mut Value) {
    let vm = &mut *vm;
    match vm.gt(&*a, &*b) {
        Ok(v) => std::ptr::write(out, v),
        Err(e) => { vm.set_last_error(e.to_string()); std::ptr::write(out, Value::Unit); }
    }
}

unsafe extern "C" fn host_lte(vm: *mut Vm, a: *const Value, b: *const Value, out: *mut Value) {
    let vm = &mut *vm;
    match vm.lte(&*a, &*b) {
        Ok(v) => std::ptr::write(out, v),
        Err(e) => { vm.set_last_error(e.to_string()); std::ptr::write(out, Value::Unit); }
    }
}

unsafe extern "C" fn host_gte(vm: *mut Vm, a: *const Value, b: *const Value, out: *mut Value) {
    let vm = &mut *vm;
    match vm.gte(&*a, &*b) {
        Ok(v) => std::ptr::write(out, v),
        Err(e) => { vm.set_last_error(e.to_string()); std::ptr::write(out, Value::Unit); }
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
        Err(e) => { vm.set_last_error(e.to_string()); std::ptr::write(out, Value::Unit); }
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
        Err(e) => { vm.set_last_error(e.to_string()); std::ptr::write(out, Value::Unit); }
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
    let mut fields = Vec::with_capacity(field_count as usize);
    let mut i = 0;
    while i + 1 < flat.len() {
        let fname = match &flat[i] { Value::String(s) => s.clone(), _ => format!("f{}", i / 2) };
        fields.push((fname, flat[i + 1].clone()));
        i += 2;
    }
    std::ptr::write(out, Value::Struct { name, fields: Rc::new(RefCell::new(fields)) });
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
        _ => std::ptr::write(out, Value::Unit),
    }
}

unsafe extern "C" fn host_store_field(vm: *mut Vm, field_idx: u64, recv: *mut Value, val: *const Value, out: *mut Value) {
    let vm = &mut *vm;
    let fname = vm.string_at(field_idx as usize).unwrap_or_default();
    if let Value::Struct { fields, .. } = &*recv {
        let mut fields = fields.borrow_mut();
        if let Some(slot) = fields.iter_mut().find(|(n, _)| n == &fname) {
            slot.1 = (*val).clone();
        }
    }
    std::ptr::write(out, (*recv).clone());
}

unsafe extern "C" fn host_index_get(vm: *mut Vm, target: *const Value, idx: *const Value, out: *mut Value) {
    let vm = &mut *vm;
    match vm.index_get(&*target, &*idx) {
        Ok(v) => std::ptr::write(out, v),
        Err(e) => { vm.set_last_error(e.to_string()); std::ptr::write(out, Value::Unit); }
    }
}

unsafe extern "C" fn host_slice_str(vm: *mut Vm, target: *const Value, start: *const Value, end: *const Value, out: *mut Value) {
    let vm = &mut *vm;
    match vm.slice_str(&*target, &*start, &*end) {
        Ok(v) => std::ptr::write(out, v),
        Err(e) => { vm.set_last_error(e.to_string()); std::ptr::write(out, Value::Unit); }
    }
}

unsafe extern "C" fn host_make_enum(
    vm: *mut Vm, name_idx: u64, variant_idx: u64, field_count: u64, args_ptr: *const Value, out: *mut Value,
) {
    let vm = &mut *vm;
    let enum_name = vm.string_at(name_idx as usize).unwrap_or_default();
    let variant = vm.string_at(variant_idx as usize).unwrap_or_default();
    // 安全：field_count 经 safe_slice 校验上限
    let fields_vec: Vec<Value> = safe_slice(args_ptr, field_count).to_vec();
    let fields: Vec<(String, Value)> = fields_vec.into_iter()
        .enumerate()
        .map(|(i, v)| (format!("_{}", i), v))
        .collect();
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
        Value::Int(i) => *i as f64,
        _ => 0.0,
    }).collect();
    use crate::runtime::tensor::Tensor;
    let shape = if cols == 0 { vec![rows as usize] } else { vec![rows as usize, cols as usize] };
    std::ptr::write(out, Value::Tensor(Rc::new(RefCell::new(Tensor::from_vec(data, shape)))));
}

unsafe extern "C" fn host_make_closure(
    vm: *mut Vm, _params: u64, chunk_idx: u64, out: *mut Value,
) {
    let vm = &mut *vm;
    let name = vm.chunk_name_at(chunk_idx as usize).unwrap_or_default();
    std::ptr::write(out, Value::FnRef { name, params: vec![], return_type: crate::hir::types::Type::Base(crate::hir::types::BaseType::Unit) });
}

unsafe extern "C" fn host_load_global(vm: *mut Vm, name_idx: u64, out: *mut Value) {
    let vm = &mut *vm;
    let name = vm.string_at(name_idx as usize).unwrap_or_default();
    std::ptr::write(out, vm.get_global(&name).unwrap_or(Value::Unit));
}

unsafe extern "C" fn host_store_global(vm: *mut Vm, name_idx: u64, val: *const Value, out: *mut Value) {
    let vm = &mut *vm;
    let name = vm.string_at(name_idx as usize).unwrap_or_default();
    let v = (*val).clone();
    vm.set_global(name, v.clone());
    std::ptr::write(out, v);
}

// ── Symbol table ───────────────────────────────────────────────────────────

pub fn hostcall_addr(name: &str) -> Option<usize> {
    let map: &[(&str, usize)] = &[
        ("host_make_int", host_make_int as usize),
        ("host_make_float", host_make_float as usize),
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
        ("host_method_call", host_method_call as usize),
        ("host_make_vec", host_make_vec as usize),
        ("host_make_map", host_make_map as usize),
        ("host_new_struct", host_new_struct as usize),
        ("host_load_field", host_load_field as usize),
        ("host_store_field", host_store_field as usize),
        ("host_index_get", host_index_get as usize),
        ("host_slice_str", host_slice_str as usize),
        ("host_make_enum", host_make_enum as usize),
        ("host_is_enum_variant", host_is_enum_variant as usize),
        ("host_enum_get_field", host_enum_get_field as usize),
        ("host_push_range", host_push_range as usize),
        ("host_make_tensor", host_make_tensor as usize),
        ("host_make_closure", host_make_closure as usize),
        ("host_load_global", host_load_global as usize),
        ("host_store_global", host_store_global as usize),
    ];
    map.iter().find(|(n, _)| *n == name).map(|(_, a)| *a)
}
