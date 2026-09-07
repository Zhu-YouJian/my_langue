//! wasmi host functions and WASM module execution.

use wasmi::{Engine, Store, Linker, Caller};
use crate::error::{TenthError, TenthResult};

// ── 安全辅助 ─────────────────────────────────────────────────────────────────
//
// WASM 模块传入的 `ptr: i32` 若为负数，`ptr as usize` 会符号扩展为巨大值，
// 切片索引立即 panic 崩溃宿主进程（DoS）。这些 helper 统一闸门：
// - `safe_offset`：把 `i32` 转为 `usize`，越界返回 None。
// - `read_cstr`：从 WASM 内存读取 NUL 终止字符串，越界返回空。
// - `safe_vec_ptr`：把 WASM 传入的 Vec 指针（i64 语意 = bump 偏移 i32）安全转
//   `usize`，截断为 i32 后为负（高位被设置）即非法，返回 None。
// - `MAX_ALLOC_BYTES`：单次 `tenth_alloc` 上限，防止 `size: i32 = -1` 转 usize
//   后回绕为巨大值。

/// 单次 `tenth_alloc` / 字符串拼接的最大字节数。超过即返回 0（失败）。
/// 16 MiB 足够任何合理的字符串/张量分配；过大的请求几乎必然是恶意输入。
const MAX_ALLOC_BYTES: usize = 16 * 1024 * 1024;

/// 将 `i32` WASM 指针安全转为 `usize` 偏移。负数或超出 `len` 返回 `None`。
#[inline]
fn safe_offset(ptr: i32, len: usize) -> Option<usize> {
    if ptr < 0 {
        return None;
    }
    let off = ptr as usize;
    if off >= len {
        return None;
    }
    Some(off)
}

/// 从 WASM 内存读取 NUL 终止字符串。`ptr` 越界返回空 `&str`。
fn read_cstr<'a>(data: &'a [u8], ptr: i32) -> &'a str {
    let off = match safe_offset(ptr, data.len()) {
        Some(o) => o,
        None => return "",
    };
    let end = data[off..].iter().position(|&b| b == 0).unwrap_or(0);
    std::str::from_utf8(&data[off..off + end]).unwrap_or("")
}

/// 将 WASM 传入的 Vec 指针（i64）安全转为 `usize` 偏移。
/// Vec 指针实际是 bump-allocator 偏移（i32 语意）。若 i64 截断为 i32 后为负
/// （即高位被设置，可能是负数或 >= 2^31 的非法偏移），返回 `None`。
#[inline]
fn safe_vec_ptr(vec: i64) -> Option<usize> {
    let p = vec as i32;
    if p < 0 {
        return None;
    }
    Some(p as usize)
}

/// Register all host imports (module "host") on the given linker.
/// The store state must be a `u32` representing the bump-allocator offset.
pub fn register_host_functions(linker: &mut Linker<u32>) -> TenthResult<()> {
    linker.func_wrap("host", "println", |caller: Caller<'_, u32>, ptr: i32| {
        let mem = caller.get_export("memory").and_then(|e| e.into_memory()).unwrap();
        let data = mem.data(&caller);
        // 安全：经 read_cstr 闸门，ptr 越界返回空串而非 panic
        println!("{}", read_cstr(data, ptr));
    }).map_err(|e| TenthError::RuntimeError { line: None, col: None, message: format!("链接器：{}", e) })?;

    linker.func_wrap("host", "write_file",
        |caller: Caller<'_, u32>, path_ptr: i32, content_ptr: i32| {
            let mem = caller.get_export("memory").and_then(|e| e.into_memory()).unwrap();
            let data = mem.data(&caller);
            // 安全：两个指针都经 read_cstr 校验
            let path = read_cstr(data, path_ptr);
            let content = read_cstr(data, content_ptr);
            let _ = std::fs::write(path, content);
    }).map_err(|e| TenthError::RuntimeError { line: None, col: None, message: format!("链接器：{}", e) })?;

    // read_file(path: i32) -> i32
    linker.func_wrap("host", "read_file",
        |mut caller: Caller<'_, u32>, path_ptr: i32| -> i32 {
            let mem = caller.get_export("memory").and_then(|e| e.into_memory()).unwrap();
            let data = mem.data(&caller);
            let path = read_cstr(data, path_ptr);
            match std::fs::read_to_string(path) {
                Ok(content) => {
                    let bump = *caller.data();
                    let bytes = content.as_bytes();
                    let needed = bytes.len() + 1;
                    // 安全：拒绝超过 MAX_ALLOC_BYTES 的写入，防止 WASM 模块构造巨大文件触发 OOM
                    if needed > MAX_ALLOC_BYTES {
                        return 0;
                    }
                    *caller.data_mut() = bump + needed as u32;
                    let dest = mem.data_mut(&mut caller);
                    let off = bump as usize;
                    if off + needed <= dest.len() {
                        dest[off..off + bytes.len()].copy_from_slice(bytes);
                        dest[off + bytes.len()] = 0;
                        bump as i32
                    } else { 0i32 }
                }
                Err(_) => 0i32,
            }
    }).map_err(|e| TenthError::RuntimeError { line: None, col: None, message: format!("链接器：{}", e) })?;

    linker.func_wrap("host", "str_add",
        |mut caller: Caller<'_, u32>, a_ptr: i32, b_ptr: i32| -> i32 {
            let mem = caller.get_export("memory").and_then(|e| e.into_memory()).unwrap();
            let data = mem.data(&caller);
            let a = read_cstr(data, a_ptr);
            let b = read_cstr(data, b_ptr);
            let result = format!("{}{}", a, b);
            let bytes = result.as_bytes();
            // 安全：拒绝超过 MAX_ALLOC_BYTES 的拼接，防止两个巨大字符串相加触发 OOM
            if bytes.len() + 1 > MAX_ALLOC_BYTES {
                return 0;
            }
            let np = *caller.data();
            let needed = np as usize + bytes.len() + 1;
            let current_len = mem.data(&caller).len();
            if needed > current_len {
                let pages = ((needed - current_len + 65535) / 65536) as u32;
                mem.grow(&mut caller, pages).ok();
            }
            *caller.data_mut() = np + bytes.len() as u32 + 1;
            let d = mem.data_mut(&mut caller);
            d[np as usize..np as usize + bytes.len()].copy_from_slice(bytes);
            d[np as usize + bytes.len()] = 0;
            np as i32
    }).map_err(|e| TenthError::RuntimeError { line: None, col: None, message: format!("链接器：{}", e) })?;

    linker.func_wrap("host", "str_eq",
        |caller: Caller<'_, u32>, a_ptr: i32, b_ptr: i32| -> i32 {
            let mem = caller.get_export("memory").and_then(|e| e.into_memory()).unwrap();
            let data = mem.data(&caller);
            if read_cstr(data, a_ptr) == read_cstr(data, b_ptr) { 1 } else { 0 }
    }).map_err(|e| TenthError::RuntimeError { line: None, col: None, message: format!("链接器：{}", e) })?;

    linker.func_wrap("host", "str_int",
        |mut caller: Caller<'_, u32>, n: i64| -> i32 {
            let s = n.to_string();
            let mem = caller.get_export("memory").and_then(|e| e.into_memory()).unwrap();
            let data = mem.data_mut(&mut caller);
            let off = 4096i32;
            let b = s.as_bytes();
            if off as usize + b.len() + 1 <= data.len() {
                data[off as usize..off as usize + b.len()].copy_from_slice(b);
                data[off as usize + b.len()] = 0;
                off
            } else { 0 }
    }).map_err(|e| TenthError::RuntimeError { line: None, col: None, message: format!("链接器：{}", e) })?;

    // tenth_alloc(size: i32) -> i32
    linker.func_wrap("host", "tenth_alloc",
        |mut caller: Caller<'_, u32>, size: i32| -> i32 {
            // 安全（M-8）：拒绝负数 size（`as usize` 会符号扩展为巨大值），并设上限
            if size < 0 {
                return 0;
            }
            let size = size as usize;
            if size > MAX_ALLOC_BYTES {
                return 0;
            }
            let ptr = *caller.data();
            let needed = match ptr as usize + size {
                n if n > MAX_ALLOC_BYTES => return 0,
                n => n,
            };
            let mem = caller.get_export("memory").and_then(|e| e.into_memory()).unwrap();
            let current_len = mem.data(&caller).len();
            // Grow memory if needed (each page is 64KiB)
            while needed > current_len {
                let pages_needed = (needed - current_len + 65535) / 65536;
                mem.grow(&mut caller, pages_needed as u32).ok();
                let new_len = mem.data(&caller).len();
                if new_len == current_len { break; } // couldn't grow
            }
            *caller.data_mut() = ptr + size as u32;
            ptr as i32
    }).map_err(|e| TenthError::RuntimeError { line: None, col: None, message: format!("链接器：{}", e) })?;

    // Vec_new() -> i64 (pointer extended to i64)
    linker.func_wrap("host", "Vec_new",
        |mut caller: Caller<'_, u32>| -> i64 {
            let ptr = *caller.data();
            // Zero-initialize the Vec header (cap=0, len=0, dp=0) so that
            // Vec_push correctly triggers the first allocation.
            let mem = caller.get_export("memory").and_then(|e| e.into_memory()).unwrap();
            let data = mem.data_mut(&mut caller);
            let p = ptr as usize;
            // 安全：检查 p+24 不越界，否则返回 0
            if p + 24 > data.len() {
                return 0;
            }
            data[p..p+8].copy_from_slice(&0i64.to_le_bytes());       // cap
            data[p+8..p+16].copy_from_slice(&0i64.to_le_bytes());    // len
            data[p+16..p+20].copy_from_slice(&0i32.to_le_bytes());   // dp
            *caller.data_mut() = ptr + 24;
            ptr as i64
    }).map_err(|e| TenthError::RuntimeError { line: None, col: None, message: format!("链接器：{}", e) })?;

    // Vec_len(vec: i64) -> i64
    linker.func_wrap("host", "Vec_len",
        |caller: Caller<'_, u32>, vec: i64| -> i64 {
            // 安全：vec 是 i64 但实际偏移是 i32。先经 safe_vec_ptr 校验非负。
            let vec_ptr = match safe_vec_ptr(vec) {
                Some(p) => p,
                None => return 0,
            };
            let mem = caller.get_export("memory").and_then(|e| e.into_memory()).unwrap();
            let data = mem.data(&caller);
            if vec_ptr + 16 <= data.len() {
                i64::from_le_bytes(data[vec_ptr+8..vec_ptr+16].try_into().unwrap())
            } else { 0 }
    }).map_err(|e| TenthError::RuntimeError { line: None, col: None, message: format!("链接器：{}", e) })?;

    // Vec_get(vec: i64, idx: i64) -> i64
    linker.func_wrap("host", "Vec_get",
        |caller: Caller<'_, u32>, vec: i64, idx: i64| -> i64 {
            let vec_ptr = match safe_vec_ptr(vec) {
                Some(p) => p,
                None => return 0,
            };
            // 安全：idx 也可能为负或巨大
            if idx < 0 {
                return 0;
            }
            let mem = caller.get_export("memory").and_then(|e| e.into_memory()).unwrap();
            let data = mem.data(&caller);
            if vec_ptr + 20 > data.len() { return 0; }
            let dp = i32::from_le_bytes(data[vec_ptr+16..vec_ptr+20].try_into().unwrap()) as usize;
            // 安全：idx * 8 用 checked_mul 防溢出
            let pos = match (idx as usize).checked_mul(8).and_then(|n| dp.checked_add(n)) {
                Some(p) => p,
                None => return 0,
            };
            if pos + 8 <= data.len() {
                i64::from_le_bytes(data[pos..pos+8].try_into().unwrap())
            } else { 0 }
    }).map_err(|e| TenthError::RuntimeError { line: None, col: None, message: format!("链接器：{}", e) })?;

    // Vec_push(vec: i64, item: i64) -> i64
    linker.func_wrap("host", "Vec_push",
        |mut caller: Caller<'_, u32>, vec: i64, item: i64| -> i64 {
            let vec_ptr = match safe_vec_ptr(vec) {
                Some(p) => p,
                None => return 0,
            };
            // Phase 1: read header
            let (cap, len, dp) = {
                let mem = caller.get_export("memory").and_then(|e| e.into_memory()).unwrap();
                let data = mem.data(&caller);
                let vp = vec_ptr;
                let cap = if vp+8 <= data.len() { i64::from_le_bytes(data[vp..vp+8].try_into().unwrap()) } else { 0 };
                let len = if vp+16 <= data.len() { i64::from_le_bytes(data[vp+8..vp+16].try_into().unwrap()) } else { 0 };
                let dp = if vp+20 <= data.len() { i32::from_le_bytes(data[vp+16..vp+20].try_into().unwrap()) } else { 0 };
                (cap, len, dp)
            };
            // Phase 2: allocate if needed
            let (new_cap, new_dp) = if len >= cap || dp == 0 {
                let nc = if cap == 0 { 4 } else { cap * 2 };
                // 安全：nc * 8 用 checked_mul，并设上限（防 cap 巨大导致溢出/OOM）
                let new_sz = match (nc as usize).checked_mul(8) {
                    Some(s) if s <= MAX_ALLOC_BYTES => s,
                    _ => return 0,
                };
                let np = *caller.data();
                *caller.data_mut() = np + new_sz as u32;
                // Copy old data from dp to new allocation (if any)
                if dp != 0 && len > 0 {
                    let mem = caller.get_export("memory").and_then(|e| e.into_memory()).unwrap();
                    let data = mem.data_mut(&mut caller);
                    let old_sz = len as usize * 8;
                    // 安全：copy_within 范围校验
                    let src_end = match (dp as usize).checked_add(old_sz) {
                        Some(e) if e <= data.len() => e,
                        _ => return 0,
                    };
                    let dst_end = match (np as usize).checked_add(old_sz) {
                        Some(e) if e <= data.len() => e,
                        _ => return 0,
                    };
                    data.copy_within(dp as usize..src_end, np as usize);
                    let _ = dst_end; // 仅校验
                }
                (nc, np as i32)
            } else {
                (cap, dp)
            };
            // Phase 3: write
            {
                let mem = caller.get_export("memory").and_then(|e| e.into_memory()).unwrap();
                let data = mem.data_mut(&mut caller);
                let vp = vec_ptr;
                if vp + 20 > data.len() { return 0; }
                data[vp..vp+8].copy_from_slice(&new_cap.to_le_bytes());
                data[vp+8..vp+16].copy_from_slice(&(len + 1).to_le_bytes());
                data[vp+16..vp+20].copy_from_slice(&new_dp.to_le_bytes());
                let elem_off = match (len as usize).checked_mul(8) {
                    Some(v) => v,
                    None => return 0,
                };
                let pos = match (new_dp as usize).checked_add(elem_off) {
                    Some(p) => p,
                    None => return 0,
                };
                if pos + 8 > data.len() { return 0; }
                data[pos..pos+8].copy_from_slice(&item.to_le_bytes());
            }
            vec
    }).map_err(|e| TenthError::RuntimeError { line: None, col: None, message: format!("链接器：{}", e) })?;

    // compile_host(src: i64, out_path: i64) -> i32
    // Reads source from WASM memory, compiles it via Rust pipeline, writes .wasm.
    linker.func_wrap("host", "compile_host",
        |caller: Caller<'_, u32>, src_ptr: i32, out_ptr: i32| -> i32 {
            let mem = caller.get_export("memory").and_then(|e| e.into_memory()).unwrap();
            let data = mem.data(&caller);
            // 安全：经 read_cstr 校验
            let src = read_cstr(data, src_ptr).to_string();
            let out = read_cstr(data, out_ptr).to_string();
            // Compile via Rust pipeline
            match crate::lexer::lexer::Lexer::new(&src).tokenize()
                .and_then(|tokens| crate::parser::parser::Parser::new(tokens).parse_program())
                .and_then(|prog| crate::hir::lower::Lowerer::new().lower_program(&prog))
                .and_then(|hir| crate::compile::compile_to_wasm(&hir))
            {
                Ok(wasm_bytes) => {
                    let _ = std::fs::write(&out, &wasm_bytes);
                    0i32
                }
                Err(_) => 1i32,
            }
    }).map_err(|e| TenthError::RuntimeError { line: None, col: None, message: format!("链接器：{}", e) })?;

    // str_len(s: i32) -> i32  — returns length of null-terminated string
    linker.func_wrap("host", "str_len",
        |caller: Caller<'_, u32>, ptr: i32| -> i32 {
            let mem = caller.get_export("memory").and_then(|e| e.into_memory()).unwrap();
            let data = mem.data(&caller);
            // 安全：经 safe_offset 校验
            match safe_offset(ptr, data.len()) {
                Some(off) => data[off..].iter().position(|&b| b == 0).unwrap_or(0) as i32,
                None => 0,
            }
    }).map_err(|e| TenthError::RuntimeError { line: None, col: None, message: format!("链接器：{}", e) })?;

    // str_at(s: i32, idx: i64) -> i32  — returns single-char string at index
    // Characters are pre-interned as "X\0" in the data section, so we
    // can return a direct pointer without heap allocation.
    linker.func_wrap("host", "str_at",
        |mut caller: Caller<'_, u32>, ptr: i32, idx: i64| -> i32 {
            let mem = caller.get_export("memory").and_then(|e| e.into_memory()).unwrap();
            let data = mem.data(&caller);
            // 安全：经 read_cstr 校验
            let s = read_cstr(data, ptr);
            // 安全：idx 也校验
            if idx < 0 {
                return 0;
            }
            let ch = s.chars().nth(idx as usize).unwrap_or('\0');
            // Pre-interned ASCII: characters 1..127 are stored as "X\0" 
            // at offset (ch-1)*2 in the data section.
            let cu = ch as u32;
            if cu >= 1 && cu < 128 {
                return ((cu - 1) * 2) as i32;
            }
            // Non-ASCII fallback: allocate from bump allocator (rare)
            let ch_str = ch.to_string();
            let ch_bytes = ch_str.as_bytes();
            let np = *caller.data();
            let needed = np as usize + ch_bytes.len() + 1;
            let current_len = mem.data(&caller).len();
            if needed > current_len {
                let pages = ((needed - current_len + 65535) / 65536) as u32;
                mem.grow(&mut caller, pages).ok();
            }
            *caller.data_mut() = np + ch_bytes.len() as u32 + 1;
            let d = mem.data_mut(&mut caller);
            d[np as usize..np as usize + ch_bytes.len()].copy_from_slice(ch_bytes);
            d[np as usize + ch_bytes.len()] = 0;
            np as i32
    }).map_err(|e| TenthError::RuntimeError { line: None, col: None, message: format!("链接器：{}", e) })?;

    // str_cmp(op: i32, a: i32, b: i32) -> i32  — op: 0=LT,1=GT,2=LE,3=GE; returns 0 or 1
    linker.func_wrap("host", "str_cmp",
        |caller: Caller<'_, u32>, op: i32, a: i32, b: i32| -> i32 {
            let mem = caller.get_export("memory").and_then(|e| e.into_memory()).unwrap();
            let data = mem.data(&caller);
            // 安全：经 read_cstr 校验
            let sa = read_cstr(data, a).to_string();
            let sb = read_cstr(data, b).to_string();
            let result = match op {
                0 => sa < sb,
                1 => sa > sb,
                2 => sa <= sb,
                3 => sa >= sb,
                _ => false,
            };
            result as i32
    }).map_err(|e| TenthError::RuntimeError { line: None, col: None, message: format!("链接器：{}", e) })?;

    // f64_bits(f64) -> i64: convert f64 to its IEEE 754 bit representation
    linker.func_wrap("host", "f64_bits",
        |val: f64| -> i64 {
            val.to_bits() as i64
    }).map_err(|e| TenthError::RuntimeError { line: None, col: None, message: format!("链接器：{}", e) })?;

    // str_slice(ptr: i32, start: i64, end: i64) -> i32: allocate new string s[start..end]
    linker.func_wrap("host", "str_slice",
        |mut caller: Caller<'_, u32>, ptr: i32, start: i64, end: i64| -> i32 {
            let mem = caller.get_export("memory").and_then(|e| e.into_memory()).unwrap();
            // Phase 1: read source slice into an owned Vec so the immutable
            // borrow of `caller` ends before any mutable operation below.
            let slice_bytes: Vec<u8> = {
                let data = mem.data(&caller);
                let off = match safe_offset(ptr, data.len()) {
                    Some(o) => o,
                    None => return 0,
                };
                let slen = data[off..].iter().position(|&b| b == 0).unwrap_or(0);
                let s = start.max(0) as usize;
                let e = if end >= i64::MAX { slen } else { end.max(0) as usize };
                let s = s.min(slen);
                let e = e.min(slen).max(s);
                data[off + s..off + e].to_vec()
            };
            let slice_len = slice_bytes.len();
            // 安全：拒绝巨大 slice
            if slice_len + 1 > MAX_ALLOC_BYTES {
                return 0;
            }
            // Phase 2: bump-allocate and write the slice.
            let np = *caller.data();
            let needed = np as usize + slice_len + 1;
            let current_len = mem.data(&caller).len();
            if needed > current_len {
                let pages = ((needed - current_len + 65535) / 65536) as u32;
                mem.grow(&mut caller, pages).ok();
            }
            *caller.data_mut() = np + slice_len as u32 + 1;
            let d = mem.data_mut(&mut caller);
            d[np as usize..np as usize + slice_len].copy_from_slice(&slice_bytes);
            d[np as usize + slice_len] = 0;
            np as i32
    }).map_err(|e| TenthError::RuntimeError { line: None, col: None, message: format!("链接器：{}", e) })?;

    // tensor_from_vec(data_ptr: i32, len: i32, rank: i32) -> i64
    // Simplified: return total element count (len) as the tensor handle.
    // This provides a deterministic value for parity testing.
    linker.func_wrap("host", "tensor_from_vec",
        |_caller: Caller<'_, u32>, _data_ptr: i32, len: i32, _rank: i32| -> i64 {
            // 安全：len 为负数时返回 0，而非符号扩展为巨大 usize
            if len < 0 { 0 } else { len as i64 }
    }).map_err(|e| TenthError::RuntimeError { line: None, col: None, message: format!("链接器：{}", e) })?;

    // Phase 5.2 F1：host_make_tensor_f16(data_ptr: i32, len: i32, rank: i32) -> i64
    // F16 张量专用 hostcall。WASM 原生不支持 f16 类型，数据以 F64 字节序列
    // 存储于 WASM 线性内存中（每个元素 8 字节），host 侧负责读取并构造 F16 TensorData。
    // 简化实现：与 tensor_from_vec 一致，返回 len 作为 handle（保证 parity 测试确定性）。
    // 后续可扩展为真正构造 F16 TensorData 并存入 host 侧张量表。
    linker.func_wrap("host", "host_make_tensor_f16",
        |_caller: Caller<'_, u32>, _data_ptr: i32, len: i32, _rank: i32| -> i64 {
            if len < 0 { 0 } else { len as i64 }
    }).map_err(|e| TenthError::RuntimeError { line: None, col: None, message: format!("链接器：{}", e) })?;

    // Phase 5.2 F1：host_make_tensor_bf16(data_ptr: i32, len: i32, rank: i32) -> i64
    // BF16 张量专用 hostcall，策略与 host_make_tensor_f16 一致。
    linker.func_wrap("host", "host_make_tensor_bf16",
        |_caller: Caller<'_, u32>, _data_ptr: i32, len: i32, _rank: i32| -> i64 {
            if len < 0 { 0 } else { len as i64 }
    }).map_err(|e| TenthError::RuntimeError { line: None, col: None, message: format!("链接器：{}", e) })?;

    // ── M1-S1（P4）：标量 math host 函数（sin/cos/ln/pow）────────────────
    // sqrt/abs 用 WASM 原生指令内联（F64Sqrt/F64Abs 等），无需 host import。
    linker.func_wrap("host", "host_sin",
        |_caller: Caller<'_, u32>, x: f64| -> f64 { x.sin() })
        .map_err(|e| TenthError::RuntimeError { line: None, col: None, message: format!("链接器：{}", e) })?;
    linker.func_wrap("host", "host_cos",
        |_caller: Caller<'_, u32>, x: f64| -> f64 { x.cos() })
        .map_err(|e| TenthError::RuntimeError { line: None, col: None, message: format!("链接器：{}", e) })?;
    linker.func_wrap("host", "host_ln",
        |_caller: Caller<'_, u32>, x: f64| -> f64 {
            // 与 VM native ln 一致：参数必须 > 0，否则返回 NaN（VM 报错路径在
            // WASM 中无法表达，返回 NaN 与 Rust f64::ln 对非法输入的行为一致）
            x.ln()
        })
        .map_err(|e| TenthError::RuntimeError { line: None, col: None, message: format!("链接器：{}", e) })?;
    linker.func_wrap("host", "host_pow",
        |_caller: Caller<'_, u32>, base: f64, exp: f64| -> f64 { base.powf(exp) })
        .map_err(|e| TenthError::RuntimeError { line: None, col: None, message: format!("链接器：{}", e) })?;

    Ok(())
}

/// Execute a WASM bytecode module in-process using wasmi.
pub fn run_wasm_module(wasm_bytes: &[u8]) -> TenthResult<()> {
    let engine = Engine::default();
    let module = wasmi::Module::new(&engine, wasm_bytes).map_err(|e| {
        TenthError::RuntimeError { line: None, col: None, message: format!("WASM 模块解析错误：{}", e) }
    })?;

    let mut store = Store::new(&engine, 8192u32);
    let mut linker = Linker::new(&engine);
    register_host_functions(&mut linker)?;

    let instance = linker.instantiate(&mut store, &module)
        .and_then(|pre| pre.start(&mut store))
        .map_err(|e| TenthError::RuntimeError { line: None, col: None,
            message: format!("WASM 实例化错误：{}", e),
        })?;

    let main_fn = instance.get_typed_func::<(), i32>(&store, "main")
        .map_err(|_| TenthError::RuntimeError { line: None, col: None,
            message: "WASM 模块没有导出的 'main' 函数".into(),
        })?;

    let exit_code = main_fn.call(&mut store, ())
        .map_err(|e| TenthError::RuntimeError { line: None, col: None,
            message: format!("WASM main() 错误：{}", e),
        })?;

    if exit_code != 0 {
        return Err(TenthError::RuntimeError { line: None, col: None,
            message: format!("WASM main() 以代码 {} 退出", exit_code),
        });
    }

    Ok(())
}
