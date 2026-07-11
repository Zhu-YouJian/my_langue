//! Expression and statement compilation.

use wasm_encoder::{BlockType, Function, Instruction, ValType};
use crate::error::{TenthError, TenthResult};
use crate::hir::hir::*;
use crate::hir::types::{BaseType, Type};
use super::{
    WasmCompiler, to_val_type_required,
    HOST_PRINTLN, HOST_WRITE_FILE, HOST_READ_FILE,
    HOST_STR_ADD, HOST_STR_EQ, HOST_STR_INT, HOST_TENTH_ALLOC,
    HOST_VEC_NEW, HOST_VEC_PUSH, HOST_VEC_LEN, HOST_VEC_GET,
    HOST_COMPILE_HOST, HOST_STR_LEN, HOST_STR_AT, HOST_STR_CMP,
    HOST_F64_BITS, HOST_STR_SLICE, HOST_TENSOR_FROM_VEC,
    HOST_MAKE_TENSOR_F16, HOST_MAKE_TENSOR_BF16,
};

impl WasmCompiler {
    // ── Function compilation ─────────────────────────────────────────────

    pub(super) fn compile_function(&mut self, func: &HirFnDef) -> TenthResult<Function> {
        eprintln!("[WASM] compile {}", func.name);
        self.local_map.clear();
        self.local_count = 0;
        for (name, _) in &func.params {
            self.local_map.insert(name.clone(), self.local_count);
            self.local_count += 1;
        }
        self.param_count = self.local_count; // parameters use correct types
        let locals: Vec<ValType> = (0..256).map(|_| ValType::I64).collect();
        let mut body = Function::new_with_locals_types(locals);
        self.compile_expr(&mut body, &func.body)?;
        if matches!(&func.return_type, Type::Base(BaseType::Unit)) {
            body.instruction(&Instruction::Return);
        }
        body.instruction(&Instruction::End);
        Ok(body)
    }

    pub(super) fn compile_main(&mut self, program: &HirProgram) -> TenthResult<Function> {
        self.local_map.clear();
        self.local_count = 0;
        let locals: Vec<ValType> = (0..256).map(|_| ValType::I64).collect();
        let mut body = Function::new_with_locals_types(locals);
        if let Some(ref expr) = program.main_expr {
            self.compile_expr(&mut body, expr)?;
            self.wrap_to_i32(&mut body, &expr.ty);
        } else if let Some(mf) = program.functions.iter().find(|f| f.name == "main") {
            let fi = self.resolve_func("main")?;
            body.instruction(&Instruction::Call(fi));
            if matches!(mf.return_type, Type::Base(BaseType::Unit)) {
                // main() returns void — nothing to drop, just push exit code
                body.instruction(&Instruction::I32Const(0));
            } else {
                self.wrap_to_i32(&mut body, &mf.return_type);
            }
        } else {
            body.instruction(&Instruction::I32Const(0));
        }
        body.instruction(&Instruction::End);
        Ok(body)
    }

    /// Emit conversion from a value type to i32 (for main's exit code).
    pub(super) fn wrap_to_i32(&self, body: &mut Function, ty: &Type) {
        match ty {
            Type::Base(b) => match b {
                BaseType::I8 | BaseType::I16 | BaseType::I32 | BaseType::I64 => {
                    body.instruction(&Instruction::I32WrapI64);
                }
                BaseType::Bool => {
                    // Already i32, no conversion needed
                }
                BaseType::F32 => {
                    body.instruction(&Instruction::I32TruncF32S);
                }
                BaseType::F64 => {
                    body.instruction(&Instruction::I32TruncF64S);
                }
                _ => {
                    body.instruction(&Instruction::Drop);
                    body.instruction(&Instruction::I32Const(0));
                }
            },
            _ => {
                // Struct pointers, etc. → i64 → i32
                body.instruction(&Instruction::I32WrapI64);
            }
        }
    }

    pub(super) fn resolve_func(&self, name: &str) -> TenthResult<u32> {
        // User-defined functions take priority over host functions
        if let Some(&idx) = self.func_map.get(name) {
            return Ok(idx);
        }
        match name {
            "println" | "eprintln" => Ok(HOST_PRINTLN),
            "write_file" => Ok(HOST_WRITE_FILE),
            "read_file" => Ok(HOST_READ_FILE),
            "str_add" => Ok(HOST_STR_ADD),
            "str_eq" => Ok(HOST_STR_EQ),
            "str_int" => Ok(HOST_STR_INT),
            "tenth_alloc" => Ok(HOST_TENTH_ALLOC),
            "Vec::new" | "Vec_new" => Ok(HOST_VEC_NEW),
            "Vec::push" | "Vec_push" => Ok(HOST_VEC_PUSH),
            "Vec::len" | "Vec_len" => Ok(HOST_VEC_LEN),
            "Vec::get" | "Vec_get" => Ok(HOST_VEC_GET),
            "compile_host" => Ok(HOST_COMPILE_HOST),
            _ => Err(TenthError::RuntimeError { line: None, col: None,
                    message: format!("WASM: 未定义函数 '{}'", name),
                }),
        }
    }

    // ── Expression compilation ───────────────────────────────────────────

    pub(super) fn compile_expr(&mut self, body: &mut Function, expr: &HirExpr) -> TenthResult<()> {
        use HirExprKind;
        match &expr.kind {
            HirExprKind::Literal(lit) => self.compile_literal(body, lit)?,

            HirExprKind::Var(name) => {
                if let Some(&idx) = self.local_map.get(name) {
                    body.instruction(&Instruction::LocalGet(idx));
                    // Extra locals (index >= param_count) are stored as i64.
                    // Convert back to the expression's actual type.
                    if idx >= self.param_count {
                        match &expr.ty {
                            Type::Base(BaseType::F32) => {
                                body.instruction(&Instruction::F64ReinterpretI64);
                                body.instruction(&Instruction::F32DemoteF64);
                            }
                            Type::Base(BaseType::F64) => {
                                body.instruction(&Instruction::F64ReinterpretI64);
                            }
                            Type::Base(BaseType::Bool) => {
                                body.instruction(&Instruction::I32WrapI64);
                            }
                            _ => {}
                        }
                    }
                } else if self.compiling_closure {
                    // D5.5: Check if this is a captured variable
                    if let Some(ci) = self.current_captures.iter().position(|c| c == name) {
                        // Load from env_ptr (local 0) + ci * 8
                        body.instruction(&Instruction::LocalGet(0));
                        body.instruction(&Instruction::I32WrapI64);
                        body.instruction(&Instruction::I32Const(ci as i32 * 8));
                        body.instruction(&Instruction::I32Add);
                        let arg = wasm_encoder::MemArg { offset: 0, align: 3, memory_index: 0 };
                        body.instruction(&Instruction::I64Load(arg));
                        // Convert to actual type if needed
                        match &expr.ty {
                            Type::Base(BaseType::F32) => {
                                body.instruction(&Instruction::F64ReinterpretI64);
                                body.instruction(&Instruction::F32DemoteF64);
                            }
                            Type::Base(BaseType::F64) => {
                                body.instruction(&Instruction::F64ReinterpretI64);
                            }
                            Type::Base(BaseType::Bool) => {
                                body.instruction(&Instruction::I32WrapI64);
                            }
                            _ => {}
                        }
                    } else if !["println","eprintln","write_file","read_file"].contains(&name.as_str()) {
                        return Err(TenthError::RuntimeError { line: None, col: None,
                            message: format!("WASM: 未定义变量 '{}'", name),
                        });
                    }
                } else if !["println","eprintln","write_file","read_file"].contains(&name.as_str()) {
                    return Err(TenthError::RuntimeError { line: None, col: None,
                        message: format!("WASM: 未定义变量 '{}'", name),
                    });
                }
            }

            HirExprKind::Binary { op, left, right, .. } => {
                // String operations: emit str_add, str_eq, or str_cmp host call
                let is_str_op = matches!(&left.ty, Type::Base(BaseType::Str));
                let is_str_add = is_str_op && matches!(op, BinOp::Add);
                let is_str_eq = is_str_op && matches!(op, BinOp::Eq | BinOp::NotEq);
                let is_str_cmp = is_str_op && matches!(op, BinOp::Lt | BinOp::Gt | BinOp::LtEq | BinOp::GtEq);
                if is_str_add {
                    self.compile_string_arg(body, left)?;
                    self.compile_string_arg(body, right)?;
                    body.instruction(&Instruction::Call(HOST_STR_ADD)); // str_add(a, b) -> i32
                    body.instruction(&Instruction::I64ExtendI32U); // i32 -> i64 for local storage
                } else if is_str_eq {
                    self.compile_string_arg(body, left)?;
                    self.compile_string_arg(body, right)?;
                    body.instruction(&Instruction::Call(HOST_STR_EQ)); // str_eq(a, b) -> i32
                    if matches!(op, BinOp::NotEq) {
                        body.instruction(&Instruction::I32Eqz); // negate
                    }
                } else if is_str_cmp {
                    let op_code: i32 = match op {
                        BinOp::Lt => 0,
                        BinOp::Gt => 1,
                        BinOp::LtEq => 2,
                        BinOp::GtEq => 3,
                        _ => 0,
                    };
                    body.instruction(&Instruction::I32Const(op_code));
                    self.compile_string_arg(body, left)?;
                    self.compile_string_arg(body, right)?;
                    body.instruction(&Instruction::Call(HOST_STR_CMP)); // str_cmp(op, a, b) -> i32
                } else {
                    // Phase 5：使用 expr.ty（lower.rs 推断的结果类型）来决定 F32/F64 路径
                    let result_is_f32 = matches!(&expr.ty, Type::Base(BaseType::F32));
                    let result_is_f64 = matches!(&expr.ty, Type::Base(BaseType::F64));
                    self.compile_expr(body, left)?;
                    // 左操作数提升到结果类型
                    let left_is_int = matches!(&left.ty, Type::Base(BaseType::I8 | BaseType::I16 | BaseType::I32 | BaseType::I64));
                    let left_is_f32 = matches!(&left.ty, Type::Base(BaseType::F32));
                    let left_is_f64 = matches!(&left.ty, Type::Base(BaseType::F64));
                    if left_is_int && result_is_f32 {
                        body.instruction(&Instruction::F32ConvertI64S);
                    } else if left_is_int && result_is_f64 {
                        body.instruction(&Instruction::F64ConvertI64S);
                    } else if left_is_f32 && result_is_f64 {
                        body.instruction(&Instruction::F64PromoteF32);
                    }
                    // (left_is_f64 && result_is_f32 不会发生：F64 不会被降级到 F32)
                    self.compile_expr(body, right)?;
                    // 右操作数提升到结果类型
                    let right_is_int = matches!(&right.ty, Type::Base(BaseType::I8 | BaseType::I16 | BaseType::I32 | BaseType::I64));
                    let right_is_f32 = matches!(&right.ty, Type::Base(BaseType::F32));
                    let right_is_f64 = matches!(&right.ty, Type::Base(BaseType::F64));
                    if right_is_int && result_is_f32 {
                        body.instruction(&Instruction::F32ConvertI64S);
                    } else if right_is_int && result_is_f64 {
                        body.instruction(&Instruction::F64ConvertI64S);
                    } else if right_is_f32 && result_is_f64 {
                        body.instruction(&Instruction::F64PromoteF32);
                    } else if right_is_f64 && result_is_f32 {
                        // F64 → F32 降级（仅当显式声明 f32 结果时）
                        body.instruction(&Instruction::F32DemoteF64);
                    }
                    // 比较类操作（结果 Bool）需用操作数类型选择 F32/F64 指令；
                    // 算术类操作用结果类型。两者经过上面提升后已统一。
                    let dispatch_ty = if result_is_f32 || result_is_f64 { &expr.ty } else { &left.ty };
                    self.compile_binop(body, op, dispatch_ty)?;
                }
            }

            HirExprKind::Unary { op, expr: inner, .. } => {
                self.compile_expr(body, inner)?;
                self.compile_unary(body, op, &inner.ty)?;
            }

            HirExprKind::Call { func, args, .. } => {
                let fname = match &func.kind {
                    HirExprKind::Var(n) => n.clone(),
                    _ => return Err(TenthError::RuntimeError { line: None, col: None,
                        message: "WASM: 不支持间接调用".into(),
                    }),
                };
                match fname.as_str() {
                    "println" | "eprintln" => {
                        if let Some(a) = args.first() {
                            self.compile_string_arg(body, a)?;
                            body.instruction(&Instruction::Call(HOST_PRINTLN));
                        }
                    }
                    "write_file" => {
                        if args.len() >= 2 {
                            self.compile_string_arg(body, &args[0])?;
                            self.compile_string_arg(body, &args[1])?;
                            body.instruction(&Instruction::Call(HOST_WRITE_FILE));
                        }
                    }
                    "read_file" => {
                        if let Some(a) = args.first() {
                            self.compile_string_arg(body, a)?;
                            body.instruction(&Instruction::Call(HOST_READ_FILE));
                            body.instruction(&Instruction::I64ExtendI32U); // i32 ptr -> i64
                        }
                    }
                    "compile_host" => {
                        if args.len() >= 2 {
                            self.compile_string_arg(body, &args[0])?;
                            self.compile_string_arg(body, &args[1])?;
                            body.instruction(&Instruction::Call(HOST_COMPILE_HOST));
                            body.instruction(&Instruction::I64ExtendI32U); // i32 -> i64
                        }
                    }
                    "Vec::new" | "Vec_new" => {
                        body.instruction(&Instruction::Call(HOST_VEC_NEW)); // Vec_new() -> i64
                    }
                    "f64_bits" => {
                        // f64_bits(f64) -> i64: reinterpret f64 bit pattern as i64
                        if let Some(a) = args.first() {
                            self.compile_expr(body, a)?;
                        }
                        body.instruction(&Instruction::Call(HOST_F64_BITS));
                    }
                    "str_len" => {
                        // str_len(i32 ptr) -> i32: string length
                        if let Some(a) = args.first() {
                            self.compile_expr(body, a)?;
                            body.instruction(&Instruction::I32WrapI64); // i64 ptr -> i32
                        }
                        body.instruction(&Instruction::Call(HOST_STR_LEN));
                        body.instruction(&Instruction::I64ExtendI32U); // i32 -> i64
                    }
                    "str_at" => {
                        // str_at(i32 ptr, i64 idx) -> i32: char at index
                        if args.len() >= 2 {
                            self.compile_expr(body, &args[0])?;
                            body.instruction(&Instruction::I32WrapI64); // i64 ptr -> i32
                            self.compile_expr(body, &args[1])?;
                        }
                        body.instruction(&Instruction::Call(HOST_STR_AT));
                        body.instruction(&Instruction::I64ExtendI32U); // i32 -> i64
                    }
                    "str_cmp" => {
                        // str_cmp(i32 op, i32 a, i32 b) -> i32: compare strings
                        if args.len() >= 3 {
                            self.compile_expr(body, &args[0])?;
                            body.instruction(&Instruction::I32WrapI64);
                            self.compile_expr(body, &args[1])?;
                            body.instruction(&Instruction::I32WrapI64);
                            self.compile_expr(body, &args[2])?;
                            body.instruction(&Instruction::I32WrapI64);
                        }
                        body.instruction(&Instruction::Call(HOST_STR_CMP));
                        body.instruction(&Instruction::I64ExtendI32U); // i32 -> i64
                    }
                    "str_slice" => {
                        // str_slice(i32 ptr, i64 start, i64 end) -> i32: substring
                        if args.len() >= 3 {
                            self.compile_expr(body, &args[0])?;
                            body.instruction(&Instruction::I32WrapI64); // i64 ptr -> i32
                            self.compile_expr(body, &args[1])?;
                            self.compile_expr(body, &args[2])?;
                        }
                        body.instruction(&Instruction::Call(HOST_STR_SLICE));
                        body.instruction(&Instruction::I64ExtendI32U); // i32 -> i64
                    }
                    // AUDIT #6 修复：补齐 str_eq/str_add/str_int 的函数调用分支，
                    // 与 str_len/str_at/str_cmp/str_slice 对齐。此前仅 BinOp 路径
                    // （`a == b`、`a + b`）和 MethodCall/InterpolatedString 路径支持，
                    // 函数调用形式 `str_eq(a,b)` 会落入 `_` 分支按普通函数解析，产出
                    // i64→i32 类型不匹配的非法 WASM。
                    "str_eq" => {
                        // str_eq(i32 a, i32 b) -> i32: 字符串相等比较返回 bool
                        if args.len() >= 2 {
                            self.compile_string_arg(body, &args[0])?;
                            self.compile_string_arg(body, &args[1])?;
                        }
                        body.instruction(&Instruction::Call(HOST_STR_EQ));
                        body.instruction(&Instruction::I64ExtendI32U); // i32 bool -> i64
                    }
                    "str_add" => {
                        // str_add(i32 a, i32 b) -> i32: 字符串拼接
                        if args.len() >= 2 {
                            self.compile_string_arg(body, &args[0])?;
                            self.compile_string_arg(body, &args[1])?;
                        }
                        body.instruction(&Instruction::Call(HOST_STR_ADD));
                        body.instruction(&Instruction::I64ExtendI32U); // i32 ptr -> i64
                    }
                    "str_int" => {
                        // str_int(i64 n) -> i32: 整数转字符串
                        if let Some(a) = args.first() {
                            self.compile_expr(body, a)?;
                        }
                        body.instruction(&Instruction::Call(HOST_STR_INT));
                        body.instruction(&Instruction::I64ExtendI32U); // i32 ptr -> i64
                    }
                    _ => {
                        // D5.4: Closure call detection — if fname is a closure variable,
                        // use call_indirect instead of regular call
                        if let Some(&type_idx) = self.closure_vars.get(&fname) {
                            if let Some(&cv_local) = self.local_map.get(&fname) {
                                // Unpack closure value (i64): high 32 = fn_ptr, low 32 = env_ptr
                                // 1. fn_ptr = cv >> 32, store in temp
                                body.instruction(&Instruction::LocalGet(cv_local));
                                body.instruction(&Instruction::I64Const(32));
                                body.instruction(&Instruction::I64ShrU);
                                let tmp = self.local_count;
                                self.local_count += 1;
                                body.instruction(&Instruction::LocalSet(tmp));
                                // 2. Push env_ptr = cv & 0xFFFFFFFF
                                body.instruction(&Instruction::LocalGet(cv_local));
                                body.instruction(&Instruction::I64Const(0xFFFFFFFF));
                                body.instruction(&Instruction::I64And);
                                // 3. Push args
                                for a in args { self.compile_expr(body, a)?; }
                                // 4. Push fn_ptr (from temp), wrap to i32, call_indirect
                                body.instruction(&Instruction::LocalGet(tmp));
                                body.instruction(&Instruction::I32WrapI64);
                                body.instruction(&Instruction::CallIndirect {
                                    type_index: type_idx,
                                    table_index: 0,
                                });
                                return Ok(());
                            }
                        }
                        for a in args { self.compile_expr(body, a)?; }
                        body.instruction(&Instruction::Call(self.resolve_func(&fname)?));
                    }
                }
            }

            HirExprKind::Block { stmts, final_expr } => {
                for s in stmts { self.compile_stmt(body, s)?; }
                if let Some(e) = final_expr { self.compile_expr(body, e)?; }
            }

            HirExprKind::If { cond, then_branch, else_branch, .. } => {
                self.compile_expr(body, cond)?;
                // WASM requires BOTH branches for Result type.
                // If-without-else must use Empty (value handled by caller).
                let has_value = else_branch.is_some()
                    && !matches!(&then_branch.ty, Type::Base(BaseType::Unit))
                    && !matches!(&else_branch.as_ref().unwrap().ty, Type::Base(BaseType::Unit));
                if has_value {
                    body.instruction(&Instruction::If(BlockType::Result(to_val_type_required(&then_branch.ty)?)));
                } else {
                    body.instruction(&Instruction::If(BlockType::Empty));
                }
                // Track If depth inside loops so Break/Continue emit correct Br depth.
                let in_loop = !self.if_depths.is_empty();
                if in_loop { self.if_depths.last_mut().unwrap().0 += 1; }
                self.compile_expr(body, then_branch)?;
                // If the If block is Empty but then_branch produces a value, drop it.
                if !has_value && !matches!(&then_branch.ty, Type::Base(BaseType::Unit)) {
                    body.instruction(&Instruction::Drop);
                }
                if let Some(eb) = else_branch {
                    body.instruction(&Instruction::Else);
                    self.compile_expr(body, eb)?;
                    if !has_value && !matches!(&eb.ty, Type::Base(BaseType::Unit)) {
                        body.instruction(&Instruction::Drop);
                    }
                }
                body.instruction(&Instruction::End);
                if in_loop { self.if_depths.last_mut().unwrap().0 -= 1; }
            }

            HirExprKind::Assign { target, value } => {
                self.compile_expr(body, value)?;
                let target_idx = if let Some(&idx) = self.local_map.get(target) {
                    idx
                } else {
                    self.local_map.insert(target.clone(), self.local_count);
                    let idx = self.local_count;
                    self.local_count += 1;
                    idx
                };
                // Non-parameter locals are stored as i64; convert float/bool values.
                // Parameters use their declared type (F32/F64) — no conversion needed.
                if target_idx >= self.param_count {
                    match &value.ty {
                        Type::Base(BaseType::F32) => {
                            body.instruction(&Instruction::F64PromoteF32);
                            body.instruction(&Instruction::I64ReinterpretF64);
                        }
                        Type::Base(BaseType::F64) => {
                            body.instruction(&Instruction::I64ReinterpretF64);
                        }
                        Type::Base(BaseType::Bool) => {
                            body.instruction(&Instruction::I64ExtendI32U);
                        }
                        _ => {}
                    }
                }
                body.instruction(&Instruction::LocalSet(target_idx));
            }

            HirExprKind::AssignOp { target, op, value } => {
                let idx = if let Some(&idx) = self.local_map.get(target) {
                    idx
                } else {
                    self.local_map.insert(target.clone(), self.local_count);
                    let idx = self.local_count;
                    self.local_count += 1;
                    idx
                };
                let is_f32 = matches!(&value.ty, Type::Base(BaseType::F32));
                let is_f64 = matches!(&value.ty, Type::Base(BaseType::F64));
                // Load current value, convert to float type if needed
                body.instruction(&Instruction::LocalGet(idx));
                if idx >= self.param_count {
                    // Non-parameter local: stored as i64, need reinterpret
                    if is_f32 {
                        body.instruction(&Instruction::F64ReinterpretI64);
                        body.instruction(&Instruction::F32DemoteF64);
                    } else if is_f64 {
                        body.instruction(&Instruction::F64ReinterpretI64);
                    }
                }
                // Compile RHS
                self.compile_expr(body, value)?;
                // Apply binary op
                self.compile_binop(body, op, &value.ty)?;
                // Convert result back to i64 for local storage (non-parameter)
                if idx >= self.param_count {
                    if is_f32 {
                        body.instruction(&Instruction::F64PromoteF32);
                        body.instruction(&Instruction::I64ReinterpretF64);
                    } else if is_f64 {
                        body.instruction(&Instruction::I64ReinterpretF64);
                    } else if matches!(&value.ty, Type::Base(BaseType::Bool)) {
                        body.instruction(&Instruction::I64ExtendI32U);
                    }
                }
                body.instruction(&Instruction::LocalSet(idx));
            }

            HirExprKind::EnumLiteral { enum_name, variant, fields } => {
                // Enum variants are stored like structs — allocate and write fields.
                // Layout keyed as "EnumName::VariantName".
                let layout_key = format!("{}::{}", enum_name, variant);
                let sz = self.struct_size(&layout_key);
                body.instruction(&Instruction::I32Const(sz as i32));
                body.instruction(&Instruction::Call(HOST_TENTH_ALLOC)); // tenth_alloc -> i32
                body.instruction(&Instruction::I64ExtendI32U);
                // Save in a freshly-allocated temp local so nested exprs don't clobber it
                let tmp = self.local_count;
                self.local_count += 1;
                body.instruction(&Instruction::LocalSet(tmp));
                let layout = self.struct_layouts.get(&layout_key).cloned()
                    .ok_or_else(|| TenthError::RuntimeError { line: None, col: None,
                        message: format!("WASM: 未知的枚举变体 '{}/{}'", enum_name, variant),
                    })?;
                for (fname, fexpr) in fields {
                    if let Some(&(offset, _size, vt)) = layout.get(fname) {
                        body.instruction(&Instruction::LocalGet(tmp));
                        body.instruction(&Instruction::I32WrapI64);
                        self.compile_expr(body, fexpr)?;
                        // Phase 5：int→F32 转换（若字段为 F32 但值为整数）
                        if matches!(vt, ValType::F32) && matches!(&fexpr.ty, Type::Base(BaseType::I8 | BaseType::I16 | BaseType::I32 | BaseType::I64)) {
                            body.instruction(&Instruction::F32ConvertI64S);
                        } else if matches!(vt, ValType::F64) && matches!(&fexpr.ty, Type::Base(BaseType::I8 | BaseType::I16 | BaseType::I32 | BaseType::I64)) {
                            body.instruction(&Instruction::F64ConvertI64S);
                        }
                        let arg = wasm_encoder::MemArg { offset: offset as u64, align: 0, memory_index: 0 };
                        match vt {
                            ValType::I64 => { body.instruction(&Instruction::I64Store(arg)); }
                            ValType::I32 => { body.instruction(&Instruction::I32Store(arg)); }
                            ValType::F64 => { body.instruction(&Instruction::F64Store(arg)); }
                            ValType::F32 => { body.instruction(&Instruction::F32Store(arg)); }
                            _ => {}
                        }
                    }
                }
                body.instruction(&Instruction::LocalGet(tmp));
            }

            HirExprKind::StructLiteral { name, fields, has_default: _ } => {
                let sz = self.struct_size(name);
                body.instruction(&Instruction::I32Const(sz as i32));
                body.instruction(&Instruction::Call(HOST_TENTH_ALLOC)); // tenth_alloc -> i32
                body.instruction(&Instruction::I64ExtendI32U); // i32 -> i64
                // Save in a freshly-allocated temp local so nested exprs don't clobber it
                let tmp = self.local_count;
                self.local_count += 1;
                body.instruction(&Instruction::LocalSet(tmp));
                let layout = self.struct_layouts.get(name).cloned()
                    .ok_or_else(|| TenthError::RuntimeError { line: None, col: None,
                        message: format!("WASM: 未知结构体 '{}'", name),
                    })?;
                for (fname, fexpr) in fields {
                    if let Some(&(offset, _size, vt)) = layout.get(fname) {
                        body.instruction(&Instruction::LocalGet(tmp));
                        body.instruction(&Instruction::I32WrapI64);
                        self.compile_expr(body, fexpr)?;
                        // Phase 5：int→F32/F64 转换（若字段为 float 但值为整数）
                        if matches!(vt, ValType::F32) && matches!(&fexpr.ty, Type::Base(BaseType::I8 | BaseType::I16 | BaseType::I32 | BaseType::I64)) {
                            body.instruction(&Instruction::F32ConvertI64S);
                        } else if matches!(vt, ValType::F64) && matches!(&fexpr.ty, Type::Base(BaseType::I8 | BaseType::I16 | BaseType::I32 | BaseType::I64)) {
                            body.instruction(&Instruction::F64ConvertI64S);
                        }
                        let arg = wasm_encoder::MemArg { offset: offset as u64, align: 0, memory_index: 0 };
                        match vt {
                            ValType::I64 => { body.instruction(&Instruction::I64Store(arg)); }
                            ValType::I32 => { body.instruction(&Instruction::I32Store(arg)); }
                            ValType::F64 => { body.instruction(&Instruction::F64Store(arg)); }
                            ValType::F32 => { body.instruction(&Instruction::F32Store(arg)); }
                            _ => {}
                        }
                    }
                }
                // Push i64 pointer as result
                body.instruction(&Instruction::LocalGet(tmp));
            }

            HirExprKind::Field { target, field } => {
                self.compile_expr(body, target)?;
                body.instruction(&Instruction::I32WrapI64); // pointer i64 -> i32
                let hint = self.infer_struct_name(&target.ty);
                let (_, offset, _, vt) = self.resolve_field(&hint, field)?;
                let arg = wasm_encoder::MemArg { offset: offset as u64, align: 0, memory_index: 0 };
                match vt {
                    ValType::I64 => { body.instruction(&Instruction::I64Load(arg)); }
                    ValType::I32 => { body.instruction(&Instruction::I32Load(arg)); }
                    ValType::F64 => { body.instruction(&Instruction::F64Load(arg)); }
                    ValType::F32 => { body.instruction(&Instruction::F32Load(arg)); }
                    _ => {}
                }
            }

            HirExprKind::FieldAssign { target, field, value } => {
                self.compile_expr(body, target)?;
                body.instruction(&Instruction::I32WrapI64);
                let hint = self.infer_struct_name(&target.ty);
                let (_, offset, _, vt) = self.resolve_field(&hint, field)?;
                self.compile_expr(body, value)?;
                // Phase 5：int→float 转换（若字段为 float 但值为整数）
                if matches!(vt, ValType::F32) && matches!(&value.ty, Type::Base(BaseType::I8 | BaseType::I16 | BaseType::I32 | BaseType::I64)) {
                    body.instruction(&Instruction::F32ConvertI64S);
                } else if matches!(vt, ValType::F64) && matches!(&value.ty, Type::Base(BaseType::I8 | BaseType::I16 | BaseType::I32 | BaseType::I64)) {
                    body.instruction(&Instruction::F64ConvertI64S);
                }
                let arg = wasm_encoder::MemArg { offset: offset as u64, align: 0, memory_index: 0 };
                match vt {
                    ValType::I64 => { body.instruction(&Instruction::I64Store(arg)); }
                    ValType::I32 => { body.instruction(&Instruction::I32Store(arg)); }
                    ValType::F64 => { body.instruction(&Instruction::F64Store(arg)); }
                    ValType::F32 => { body.instruction(&Instruction::F32Store(arg)); }
                    _ => {}
                }
            }

            HirExprKind::MethodCall { receiver, method, args, .. } => {
                self.compile_expr(body, receiver)?;
                // Receiver is i64; for string methods we need to convert to i32 (ptr)
                match method.as_str() {
                    "len" => {
                        // Check if receiver is String type
                        let is_string = matches!(&receiver.ty, Type::Base(BaseType::Str));
                        if is_string {
                            body.instruction(&Instruction::I32WrapI64); // i64 -> i32 pointer
                            body.instruction(&Instruction::Call(HOST_STR_LEN));    // str_len(i32) -> i32
                            body.instruction(&Instruction::I64ExtendI32U); // i32 -> i64
                        } else {
                            body.instruction(&Instruction::Call(HOST_VEC_LEN)); // Vec_len(i64) -> i64
                        }
                    }
                    "push" => {
                        if let Some(a) = args.first() {
                            self.compile_expr(body, a)?;
                        } else {
                            body.instruction(&Instruction::I64Const(0));
                        }
                        body.instruction(&Instruction::Call(HOST_VEC_PUSH)); // Vec_push -> i64
                        body.instruction(&Instruction::Drop);     // push returns Unit
                    }
                    "get" => {
                        if let Some(a) = args.first() {
                            self.compile_expr(body, a)?;
                        } else {
                            body.instruction(&Instruction::I64Const(0));
                        }
                        body.instruction(&Instruction::Call(HOST_VEC_GET)); // Vec_get(i64, i64) -> i64
                    }
                    _ => return Err(TenthError::RuntimeError { line: None, col: None,
                        message: format!("WASM: 不支持的方法 '{}'", method),
                    }),
                }
            }

            // Ref/MutRef/Deref are identity ops for struct pointers (stored as i64)
            HirExprKind::Ref(inner)
            | HirExprKind::MutRef(inner)
            | HirExprKind::Deref(inner)
            | HirExprKind::TryBlock(inner) => {
                self.compile_expr(body, inner)?;
            }

            HirExprKind::InterpolatedString { parts } => {
                // Evaluate by concatenating all parts as strings via str_add host call
                let mut first = true;
                for p in parts {
                    match p {
                        crate::hir::hir::InterpPart::Literal(s) => {
                            body.instruction(&Instruction::I32Const(self.intern_string(s) as i32));
                        }
                        crate::hir::hir::InterpPart::Expr(name) => {
                            // Look up variable and convert to string
                            if let Some(&idx) = self.local_map.get(name) {
                                body.instruction(&Instruction::LocalGet(idx));
                                // Convert to i32 string pointer via str_int
                                body.instruction(&Instruction::Call(HOST_STR_INT)); // str_int(i64) -> i32
                            }
                        }
                    }
                    if first {
                        first = false;
                    } else {
                        // str_add: pop two i32 string ptrs, push concatenated i32 ptr
                        body.instruction(&Instruction::Call(HOST_STR_ADD)); // str_add
                    }
                }
                // Result is i32 string pointer; extend to i64 for local storage
                body.instruction(&Instruction::I64ExtendI32U);
            }
            HirExprKind::Tuple(_elems) => {
                // Tuple not yet supported in WASM backend
                body.instruction(&Instruction::I64Const(0));
            }

            HirExprKind::Index { target, indices } => {
                // Distinguish String indexing (str_at) from Vec indexing (vec_get)
                // based on the target's type. Unknown types default to vec_get
                // since Vec indexing is far more common in tenthc.
                let is_string = matches!(&target.ty, Type::Base(BaseType::Str));
                self.compile_expr(body, target)?;
                if is_string {
                    // String pointer is stored as i64; str_at expects i32
                    body.instruction(&Instruction::I32WrapI64);
                }
                // For Vec: target stays as i64 (vec ptr) for vec_get
                match indices.first() {
                    Some(Index::Single(idx)) => {
                        self.compile_expr(body, idx)?; // compile index expression (i64)
                        if is_string {
                            body.instruction(&Instruction::Call(HOST_STR_AT)); // str_at(i32, i64) -> i32
                            body.instruction(&Instruction::I64ExtendI32U); // i32 -> i64
                        } else {
                            body.instruction(&Instruction::Call(HOST_VEC_GET)); // vec_get(i64, i64) -> i64
                        }
                    }
                    Some(Index::Range { start, end }) => {
                        // String slice: s[start..end] -> str_slice(ptr, start, end) -> ptr
                        if !is_string {
                            return Err(TenthError::RuntimeError { line: None, col: None,
                                message: "WASM: Vec range slicing not yet supported".to_string(),
                            });
                        }
                        if let Some(s) = start {
                            self.compile_expr(body, s)?;
                        } else {
                            body.instruction(&Instruction::I64Const(0));
                        }
                        if let Some(e) = end {
                            self.compile_expr(body, e)?;
                        } else {
                            // No end: use string length — push a sentinel and let host handle it
                            // For now, push a large value; the host will clamp to strlen
                            body.instruction(&Instruction::I64Const(i64::MAX));
                        }
                        body.instruction(&Instruction::Call(HOST_STR_SLICE)); // str_slice(i32, i64, i64) -> i32
                        body.instruction(&Instruction::I64ExtendI32U); // i32 -> i64
                    }
                    _ => {
                        body.instruction(&Instruction::I64Const(0));
                        if is_string {
                            body.instruction(&Instruction::Call(HOST_STR_AT)); // str_at fallback
                            body.instruction(&Instruction::I64ExtendI32U);
                        } else {
                            body.instruction(&Instruction::Call(HOST_VEC_GET)); // vec_get fallback
                        }
                    }
                }
            }

            HirExprKind::TensorLiteral { data, ty } => {
                // Phase 5：按 dtype 分支。F32 元素占 4 字节，F64 占 8 字节。
                // Phase 5.2 F1：F16/BF16 张量按 F64 字节存储（WASM 原生不支持 f16/bf16），
                // 但调用专用 host_make_tensor_f16/bf16 hostcall 标记 dtype。
                // Flatten 2D data into elements, allocate memory, write values,
                // then call appropriate host import based on dtype.
                let rows = data.len() as i32;
                let total: i32 = data.iter().map(|r| r.len() as i32).sum();
                // 从 Tensor 类型提取 dtype
                let (is_f32, is_f16, is_bf16) = match ty {
                    Type::Tensor { dtype, .. } => match dtype.as_ref() {
                        Type::Base(BaseType::F32) => (true, false, false),
                        Type::Base(BaseType::F16) => (false, true, false),
                        Type::Base(BaseType::BF16) => (false, false, true),
                        _ => (false, false, false),
                    },
                    _ => (false, false, false),
                };
                // F32 元素 4 字节；F64/F16/BF16 元素 8 字节（F16/BF16 按 F64 存储）
                let elem_size: i32 = if is_f32 { 4 } else { 8 };
                let size = total * elem_size;

                // Allocate memory: tenth_alloc(size) -> i32 ptr -> i64
                body.instruction(&Instruction::I32Const(size));
                body.instruction(&Instruction::Call(HOST_TENTH_ALLOC)); // tenth_alloc
                body.instruction(&Instruction::I64ExtendI32U); // i32 -> i64
                let tmp = self.local_count;
                self.local_count += 1;
                body.instruction(&Instruction::LocalSet(tmp));

                // Write each element at offset idx * elem_size
                let mut idx: i32 = 0;
                for row in data {
                    for elem in row {
                        body.instruction(&Instruction::LocalGet(tmp));
                        body.instruction(&Instruction::I32WrapI64); // ptr i64 -> i32
                        self.compile_expr(body, elem)?;
                        let arg = wasm_encoder::MemArg { offset: (idx as u64) * (elem_size as u64), align: 0, memory_index: 0 };
                        if is_f32 {
                            // F32 tensor: 元素转为 F32 后 F32Store
                            if matches!(&elem.ty, Type::Base(BaseType::I8 | BaseType::I16 | BaseType::I32 | BaseType::I64)) {
                                body.instruction(&Instruction::F32ConvertI64S);
                            } else if matches!(&elem.ty, Type::Base(BaseType::F64)) {
                                body.instruction(&Instruction::F32DemoteF64);
                            }
                            body.instruction(&Instruction::F32Store(arg));
                        } else {
                            // F64/F16/BF16 tensor: 元素转为 F64 后 F64Store
                            // F16/BF16 在 WASM 中以 F64 位表示存储（host 侧负责 reinterpret）
                            if matches!(&elem.ty, Type::Base(BaseType::I8 | BaseType::I16 | BaseType::I32 | BaseType::I64)) {
                                body.instruction(&Instruction::F64ConvertI64S);
                            } else if matches!(&elem.ty, Type::Base(BaseType::F32)) {
                                body.instruction(&Instruction::F64PromoteF32);
                            }
                            body.instruction(&Instruction::F64Store(arg));
                        }
                        idx += 1;
                    }
                }

                // Phase 5.2 F1：按 dtype 选择 hostcall
                // F64/F32 → tensor_from_vec (import 17)
                // F16 → host_make_tensor_f16 (import 18)
                // BF16 → host_make_tensor_bf16 (import 19)
                let host_import = if is_f16 { HOST_MAKE_TENSOR_F16 }
                                  else if is_bf16 { HOST_MAKE_TENSOR_BF16 }
                                  else { HOST_TENSOR_FROM_VEC };
                body.instruction(&Instruction::LocalGet(tmp));
                body.instruction(&Instruction::I32WrapI64); // data_ptr
                body.instruction(&Instruction::I32Const(total)); // len
                body.instruction(&Instruction::I32Const(rows)); // rank (rows)
                body.instruction(&Instruction::Call(host_import)); // -> i64 tensor handle
            }

            // D5.3/D5.6: Closure — compile as packed i64 (table_idx << 32 | env_ptr)
            HirExprKind::Closure { captures, .. } => {
                let ptr = expr as *const HirExpr as usize;
                let cidx = *self.closure_expr_map.get(&ptr).ok_or_else(|| TenthError::RuntimeError { line: None, col: None,
                    message: "WASM: 闭包未注册".into(),
                })?;
                let (_func_idx, _type_idx, _pc) = self.closure_info[cidx];
                // Use closure index (table position) as fn_ptr, NOT func_idx
                let table_idx = cidx as i64;
                if captures.is_empty() {
                    // No captures: env_ptr = 0, packed = table_idx << 32
                    body.instruction(&Instruction::I64Const(table_idx << 32));
                } else {
                    // Allocate env struct via tenth_alloc (import 6)
                    // size = captures_count * 8
                    let env_size = captures.len() as i32 * 8;
                    body.instruction(&Instruction::I32Const(env_size));
                    body.instruction(&Instruction::Call(HOST_TENTH_ALLOC)); // tenth_alloc -> i32
                    body.instruction(&Instruction::I64ExtendI32U); // i32 -> i64 env_ptr
                    // Store env_ptr in temp local
                    let tmp = self.local_count;
                    self.local_count += 1;
                    body.instruction(&Instruction::LocalSet(tmp));
                    // Write each captured variable to env struct
                    for (ci, cap_name) in captures.iter().enumerate() {
                        body.instruction(&Instruction::LocalGet(tmp));
                        body.instruction(&Instruction::I32WrapI64);
                        // Load the captured variable's value
                        if let Some(&idx) = self.local_map.get(cap_name) {
                            body.instruction(&Instruction::LocalGet(idx));
                            // Convert f64/bool to i64 for storage
                            // (locals already store as i64, so no conversion needed)
                        } else {
                            // Captured variable not found — push 0
                            body.instruction(&Instruction::I64Const(0));
                        }
                        let arg = wasm_encoder::MemArg { offset: (ci as u64) * 8, align: 0, memory_index: 0 };
                        body.instruction(&Instruction::I64Store(arg));
                    }
                    // Push packed value: (table_idx << 32) | env_ptr
                    body.instruction(&Instruction::I64Const(table_idx << 32));
                    body.instruction(&Instruction::LocalGet(tmp));
                    body.instruction(&Instruction::I64Or);
                }
            }

            _ => return Err(TenthError::RuntimeError { line: None, col: None,
                message: format!("WASM: 不支持的表达式 {:?}", expr.kind),
            }),
        }
        Ok(())
    }

    pub(super) fn compile_stmt(&mut self, body: &mut Function, stmt: &HirStmt) -> TenthResult<()> {
        use crate::hir::hir::HirStmtKind;
        match &stmt.kind {
            HirStmtKind::Expr(e) => {
                self.compile_expr(body, e)?;
                if !matches!(&e.ty, Type::Base(BaseType::Unit)) {
                    body.instruction(&Instruction::Drop);
                }
            }
            HirStmtKind::Let { names, type_ann, init, .. } => {
                if let Some(e) = init {
                    self.compile_expr(body, e)?;
                    // Phase 5：按 dtype 分支。Let 创建的新 local 是 i64 类型（非参数），
                    // 需将 float 值 reinterpret 为 i64 存储。
                    let target_f32 = matches!(type_ann, Some(Type::Base(BaseType::F32)));
                    let target_f64 = matches!(type_ann, Some(Type::Base(BaseType::F64)));
                    let expr_f32 = matches!(&e.ty, Type::Base(BaseType::F32));
                    let expr_f64 = matches!(&e.ty, Type::Base(BaseType::F64));
                    let expr_bool = matches!(&e.ty, Type::Base(BaseType::Bool));
                    if target_f32 && !expr_f32 && !expr_f64 {
                        // int → f32 target: i64 → F32ConvertI64S → F64PromoteF32 → I64ReinterpretF64
                        body.instruction(&Instruction::F32ConvertI64S);
                        body.instruction(&Instruction::F64PromoteF32);
                        body.instruction(&Instruction::I64ReinterpretF64);
                    } else if target_f64 && !expr_f32 && !expr_f64 {
                        // int → f64 target: i64 → F64ConvertI64S → I64ReinterpretF64
                        body.instruction(&Instruction::F64ConvertI64S);
                        body.instruction(&Instruction::I64ReinterpretF64);
                    } else if expr_f32 {
                        // f32 value: F64PromoteF32 → I64ReinterpretF64
                        body.instruction(&Instruction::F64PromoteF32);
                        body.instruction(&Instruction::I64ReinterpretF64);
                    } else if expr_f64 {
                        // f64 value: I64ReinterpretF64
                        body.instruction(&Instruction::I64ReinterpretF64);
                    } else if expr_bool {
                        body.instruction(&Instruction::I64ExtendI32U);
                    }
                    // D5: If init is a Closure, register variable as closure var
                    if let HirExprKind::Closure { .. } = &e.kind {
                        let ptr = e as *const HirExpr as usize;
                        if let Some(&cidx) = self.closure_expr_map.get(&ptr) {
                            let (_, type_idx, _) = self.closure_info[cidx];
                            for name in names {
                                self.closure_vars.insert(name.clone(), type_idx);
                            }
                        }
                    }
                } else {
                    body.instruction(&Instruction::I64Const(0));
                }
                for name in names {
                    self.local_map.insert(name.clone(), self.local_count);
                    body.instruction(&Instruction::LocalSet(self.local_count));
                    self.local_count += 1;
                }
            }
            HirStmtKind::Loop { body: lb } => {
                self.if_depths.push((0, 0));
                body.instruction(&Instruction::Block(BlockType::Empty));
                body.instruction(&Instruction::Loop(BlockType::Empty));
                for s in lb { self.compile_stmt(body, s)?; }
                body.instruction(&Instruction::Br(0));
                body.instruction(&Instruction::End);
                body.instruction(&Instruction::End);
                self.if_depths.pop();
            }
            HirStmtKind::While { cond, body: lb } => {
                self.if_depths.push((0, 0));
                body.instruction(&Instruction::Block(BlockType::Empty));
                body.instruction(&Instruction::Loop(BlockType::Empty));
                self.compile_expr(body, cond)?;
                body.instruction(&Instruction::I32Eqz);
                body.instruction(&Instruction::BrIf(1));
                self.compile_stmt(body, lb.as_ref())?;
                body.instruction(&Instruction::Br(0));
                body.instruction(&Instruction::End);
                body.instruction(&Instruction::End);
                self.if_depths.pop();
            }
            HirStmtKind::Return(expr) => {
                if let Some(e) = expr {
                    self.compile_expr(body, e)?;
                }
                body.instruction(&Instruction::Return);
            }
            HirStmtKind::Break => {
                let &(if_depth, break_offset) = self.if_depths.last().unwrap_or(&(0, 0));
                let depth = 1 + break_offset + if_depth;
                body.instruction(&Instruction::Br(depth));
            }
            HirStmtKind::Continue => {
                let &(if_depth, _) = self.if_depths.last().unwrap_or(&(0, 0));
                let depth = if_depth;
                body.instruction(&Instruction::Br(depth));
            }
            HirStmtKind::For { var, iter, body: lb } => {
                // Only Range iterators are supported (for x in start..end { ... })
                match &iter.kind {
                    HirExprKind::Range { start, end, inclusive } => {
                        // Allocate local for the loop variable
                        let var_local = self.local_count;
                        self.local_map.insert(var.clone(), var_local);
                        self.local_count += 1;

                        // var = start (default 0)
                        if let Some(s) = start {
                            self.compile_expr(body, s)?;
                        } else {
                            body.instruction(&Instruction::I64Const(0));
                        }
                        body.instruction(&Instruction::LocalSet(var_local));

                        self.if_depths.push((0, 1)); // break_offset=1 for for's inner body block
                        // block (break target) + loop (continue/back-edge target)
                        body.instruction(&Instruction::Block(BlockType::Empty));
                        body.instruction(&Instruction::Loop(BlockType::Empty));

                        // condition: var < end (or var <= end if inclusive)
                        body.instruction(&Instruction::LocalGet(var_local));
                        if let Some(e) = end {
                            self.compile_expr(body, e)?;
                        } else {
                            body.instruction(&Instruction::I64Const(i64::MAX));
                        }
                        if *inclusive {
                            body.instruction(&Instruction::I64LeS);
                        } else {
                            body.instruction(&Instruction::I64LtS);
                        }
                        body.instruction(&Instruction::I32Eqz);
                        body.instruction(&Instruction::BrIf(1)); // exit block when condition false

                        // Wrap body in inner block so `continue` (br if_depth=0) breaks
                        // the inner body block and falls through to var += 1, avoiding
                        // infinite loop where continue skips the increment.
                        body.instruction(&Instruction::Block(BlockType::Empty));

                        // loop body
                        self.compile_stmt(body, lb.as_ref())?;

                        body.instruction(&Instruction::End); // end inner body block

                        // var += 1
                        body.instruction(&Instruction::LocalGet(var_local));
                        body.instruction(&Instruction::I64Const(1));
                        body.instruction(&Instruction::I64Add);
                        body.instruction(&Instruction::LocalSet(var_local));

                        body.instruction(&Instruction::Br(0)); // back-edge to loop header
                        body.instruction(&Instruction::End); // loop
                        body.instruction(&Instruction::End); // block
                        self.if_depths.pop();
                    }
                    _ => return Err(TenthError::RuntimeError { line: None, col: None,
                        message: format!("WASM: For 循环仅支持 Range 迭代器, got {:?}", iter.kind),
                    }),
                }
            }
            _ => return Err(TenthError::RuntimeError { line: None, col: None,
                message: format!("WASM: 不支持的语句 {:?}", stmt.kind),
            }),
        }
        Ok(())
    }

    // ── Literals & ops ────────────────────────────────────────────────────

    pub(super) fn compile_literal(&mut self, body: &mut Function, lit: &Literal) -> TenthResult<()> {
        match lit {
            Literal::Int(n) => { body.instruction(&Instruction::I64Const(*n)); }
            // Phase 5：消除策略 A，按 dtype 分支发 F32Const/F64Const
            Literal::Float(n, dt) => {
                match dt {
                    BaseType::F32 => { body.instruction(&Instruction::F32Const(*n as f32)); }
                    _ => { body.instruction(&Instruction::F64Const(*n)); }
                }
            }
            Literal::Bool(b) => { body.instruction(&Instruction::I32Const(if *b { 1 } else { 0 })); }
            Literal::String(s) => {
                let off = self.intern_string(s);
                body.instruction(&Instruction::I32Const(off as i32));
                body.instruction(&Instruction::I64ExtendI32U); // strings are stored as i64 in locals
            }
        }
        Ok(())
    }

    pub(super) fn compile_binop(&self, body: &mut Function, op: &BinOp, result_ty: &Type) -> TenthResult<()> {
        // Phase 5：按结果类型（lower.rs 推断的提升后类型）选择 F32/F64/I64 指令。
        let is_f32 = matches!(result_ty, Type::Base(BaseType::F32));
        let is_f64 = matches!(result_ty, Type::Base(BaseType::F64));
        match op {
            BinOp::Add => {
                if is_f32 { body.instruction(&Instruction::F32Add); }
                else if is_f64 { body.instruction(&Instruction::F64Add); }
                else { body.instruction(&Instruction::I64Add); }
            }
            BinOp::Sub => {
                if is_f32 { body.instruction(&Instruction::F32Sub); }
                else if is_f64 { body.instruction(&Instruction::F64Sub); }
                else { body.instruction(&Instruction::I64Sub); }
            }
            BinOp::Mul => {
                if is_f32 { body.instruction(&Instruction::F32Mul); }
                else if is_f64 { body.instruction(&Instruction::F64Mul); }
                else { body.instruction(&Instruction::I64Mul); }
            }
            BinOp::Div => {
                if is_f32 { body.instruction(&Instruction::F32Div); }
                else if is_f64 { body.instruction(&Instruction::F64Div); }
                else { body.instruction(&Instruction::I64DivS); }
            }
            BinOp::Mod => { body.instruction(&Instruction::I64RemS); }
            BinOp::Eq => {
                if is_f32 { body.instruction(&Instruction::F32Eq); }
                else if is_f64 { body.instruction(&Instruction::F64Eq); }
                else { body.instruction(&Instruction::I64Eq); }
            }
            BinOp::NotEq => {
                if is_f32 { body.instruction(&Instruction::F32Ne); }
                else if is_f64 { body.instruction(&Instruction::F64Ne); }
                else { body.instruction(&Instruction::I64Ne); }
            }
            BinOp::Lt => {
                if is_f32 { body.instruction(&Instruction::F32Lt); }
                else if is_f64 { body.instruction(&Instruction::F64Lt); }
                else { body.instruction(&Instruction::I64LtS); }
            }
            BinOp::Gt => {
                if is_f32 { body.instruction(&Instruction::F32Gt); }
                else if is_f64 { body.instruction(&Instruction::F64Gt); }
                else { body.instruction(&Instruction::I64GtS); }
            }
            BinOp::LtEq => {
                if is_f32 { body.instruction(&Instruction::F32Le); }
                else if is_f64 { body.instruction(&Instruction::F64Le); }
                else { body.instruction(&Instruction::I64LeS); }
            }
            BinOp::GtEq => {
                if is_f32 { body.instruction(&Instruction::F32Ge); }
                else if is_f64 { body.instruction(&Instruction::F64Ge); }
                else { body.instruction(&Instruction::I64GeS); }
            }
            BinOp::And => { body.instruction(&Instruction::I32And); }
            BinOp::Or => { body.instruction(&Instruction::I32Or); }
        }
        Ok(())
    }

    pub(super) fn emit_if(&self, body: &mut Function, cond: bool, then_i: Instruction<'_>, else_i: Instruction<'_>) {
        if cond { body.instruction(&then_i); } else { body.instruction(&else_i); }
    }

    pub(super) fn compile_unary(&self, body: &mut Function, op: &UnaryOp, ty: &Type) -> TenthResult<()> {
        let is_f32 = matches!(ty, Type::Base(BaseType::F32));
        let is_f64 = matches!(ty, Type::Base(BaseType::F64));
        match op {
            UnaryOp::Neg => if is_f32 {
                body.instruction(&Instruction::F32Neg);
            } else if is_f64 {
                body.instruction(&Instruction::F64Neg);
            } else {
                body.instruction(&Instruction::LocalSet(self.local_count));
                body.instruction(&Instruction::I64Const(0));
                body.instruction(&Instruction::LocalGet(self.local_count));
                body.instruction(&Instruction::I64Sub);
            },
            UnaryOp::Not => { body.instruction(&Instruction::I32Eqz); }
            UnaryOp::Try => { /* Try not supported in WASM backend */ }
        }
        Ok(())
    }

    pub(super) fn compile_string_arg(&mut self, body: &mut Function, expr: &HirExpr) -> TenthResult<()> {
        match &expr.kind {
            HirExprKind::Literal(Literal::String(s)) => {
                body.instruction(&Instruction::I32Const(self.intern_string(s) as i32));
            }
            HirExprKind::Literal(Literal::Int(n)) => {
                body.instruction(&Instruction::I64Const(*n));
                body.instruction(&Instruction::Call(HOST_STR_INT));
            }
            HirExprKind::Literal(Literal::Float(n, dt)) => {
                // Phase 5：按 dtype 分支（仅为类型一致，字符串转换结果相同）
                match dt {
                    BaseType::F32 => {
                        body.instruction(&Instruction::F32Const(*n as f32));
                        body.instruction(&Instruction::I32TruncF32S);
                        body.instruction(&Instruction::I64ExtendI32S);
                    }
                    _ => {
                        body.instruction(&Instruction::F64Const(*n));
                        body.instruction(&Instruction::I64TruncF64S);
                    }
                }
                body.instruction(&Instruction::Call(HOST_STR_INT));
            }
            HirExprKind::Binary { op: BinOp::Add, left, right, .. } => {
                self.compile_string_arg(body, left)?;
                self.compile_string_arg(body, right)?;
                body.instruction(&Instruction::Call(HOST_STR_ADD));
            }
            _ => {
                self.compile_expr(body, expr)?;
                // If value is a string (stored as i64 in locals), wrap to i32
                if matches!(&expr.ty, Type::Base(BaseType::Str)) {
                    body.instruction(&Instruction::I32WrapI64);
                } else if matches!(&expr.ty, Type::Base(BaseType::I64)) {
                    body.instruction(&Instruction::Call(HOST_STR_INT)); // str_int
                }
            }
        }
        Ok(())
    }

    // ── String interning ─────────────────────────────────────────────────

    pub(super) fn intern_string(&mut self, s: &str) -> u32 {
        if let Some(&off) = self.string_offsets.get(s) { return off; }
        let off = self.string_data.len() as u32;
        self.string_data.extend_from_slice(s.as_bytes());
        self.string_data.push(0);
        self.string_offsets.insert(s.to_string(), off);
        off
    }
}
