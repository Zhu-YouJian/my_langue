//! HIR -> WebAssembly bytecode compiler.
//!
//! Generates a WASM module from a Tenth HIR program using `wasm-encoder`.
//! The module is designed to be executed by `wasmi` (embedded in the Rust host).

use std::collections::HashMap;
use wasm_encoder::{
    BlockType, CodeSection, ConstExpr, DataSection, EntityType, ExportKind,
    ExportSection, Function, FunctionSection, ImportSection, Instruction,
    MemorySection, MemoryType, Module, TypeSection, ValType,
};
use crate::error::{TenthError, TenthResult};
use crate::hir::hir::*;
use crate::hir::types::{BaseType, Type};

// ── Type mapping ───────────────────────────────────────────────────────────

fn to_val_type(ty: &Type) -> Option<ValType> {
    match ty {
        Type::Base(b) => match b {
            BaseType::I8 | BaseType::I16 | BaseType::I32 | BaseType::I64 => Some(ValType::I64),
            BaseType::F32 | BaseType::F64 => Some(ValType::F64),
            BaseType::Bool => Some(ValType::I32),
            BaseType::Str => Some(ValType::I32),
            BaseType::Unit => None,
            _ => None,
        },
        Type::Unknown => Some(ValType::I64),
        _ => None,
    }
}

fn to_val_type_required(ty: &Type) -> TenthResult<ValType> {
    to_val_type(ty).ok_or_else(|| TenthError::RuntimeError {
        message: format!("cannot map type {:?} to WASM value type", ty),
    })
}

// ── Compiler state ─────────────────────────────────────────────────────────

pub struct WasmCompiler {
    type_cache: HashMap<(Vec<ValType>, Vec<ValType>), u32>,
    func_map: HashMap<String, u32>,
    hir_funcs: Vec<HirFnDef>,
    string_data: Vec<u8>,
    string_offsets: HashMap<String, u32>,
    local_map: HashMap<String, u32>,
    local_count: u32,
}

impl WasmCompiler {
    pub fn new() -> Self {
        WasmCompiler {
            type_cache: HashMap::new(),
            func_map: HashMap::new(),
            hir_funcs: Vec::new(),
            string_data: Vec::new(),
            string_offsets: HashMap::new(),
            local_map: HashMap::new(),
            local_count: 0,
        }
    }

    pub fn compile(&mut self, program: &HirProgram) -> TenthResult<Vec<u8>> {
        self.hir_funcs = program.functions.clone();
        self.collect_strings(program);

        let mut module = Module::new();

        self.emit_type_section(&mut module, program)?;
        self.emit_import_section(&mut module);
        let _fc = self.emit_function_section(&mut module, program);
        self.emit_memory_section(&mut module);
        self.emit_export_section(&mut module, program);
        self.emit_code_section(&mut module, program)?;
        if !self.string_data.is_empty() {
            self.emit_data_section(&mut module);
        }

        Ok(module.finish())
    }

    // ── Section builders ────────────────────────────────────────────────

    fn emit_type_section(&mut self, module: &mut Module, program: &HirProgram) -> TenthResult<()> {
        let mut types = TypeSection::new();
        let mut next = 0u32;
        let mut reg = |p: Vec<ValType>, r: Vec<ValType>| -> u32 {
            let k = (p.clone(), r.clone());
            *self.type_cache.entry(k).or_insert_with(|| {
                types.function(p, r);
                let n = next; next += 1; n
            })
        };
        // Host imports
        reg(vec![ValType::I32], vec![]);
        reg(vec![ValType::I32, ValType::I32], vec![]);
        reg(vec![ValType::I32], vec![ValType::I32]);
        reg(vec![ValType::I32, ValType::I32], vec![ValType::I32]);
        reg(vec![ValType::I32, ValType::I32], vec![ValType::I32]);
        reg(vec![ValType::I64], vec![ValType::I32]);
        // User functions
        for func in &program.functions {
            let p: Vec<ValType> = func.params.iter().filter_map(|(_, t)| to_val_type(t)).collect();
            let r: Vec<ValType> = to_val_type(&func.return_type).into_iter().collect();
            reg(p, r);
        }
        // main
        reg(vec![], vec![ValType::I32]);
        module.section(&types);
        Ok(())
    }

    fn emit_import_section(&mut self, module: &mut Module) {
        let mut imports = ImportSection::new();
        let ti = |p: Vec<ValType>, r: Vec<ValType>| -> u32 {
            *self.type_cache.get(&(p, r)).unwrap_or(&0)
        };
        imports.import("host", "println", EntityType::Function(ti(vec![ValType::I32], vec![])));
        imports.import("host", "write_file", EntityType::Function(ti(vec![ValType::I32, ValType::I32], vec![])));
        imports.import("host", "read_file", EntityType::Function(ti(vec![ValType::I32], vec![ValType::I32])));
        imports.import("host", "str_add", EntityType::Function(ti(vec![ValType::I32, ValType::I32], vec![ValType::I32])));
        imports.import("host", "str_eq", EntityType::Function(ti(vec![ValType::I32, ValType::I32], vec![ValType::I32])));
        imports.import("host", "str_int", EntityType::Function(ti(vec![ValType::I64], vec![ValType::I32])));
        module.section(&imports);
    }

    fn emit_function_section(&mut self, module: &mut Module, program: &HirProgram) -> u32 {
        let mut funcs = FunctionSection::new();
        let mut idx = 6u32;
        for func in &program.functions {
            self.func_map.insert(func.name.clone(), idx);
            let p: Vec<ValType> = func.params.iter().filter_map(|(_, t)| to_val_type(t)).collect();
            let r: Vec<ValType> = to_val_type(&func.return_type).into_iter().collect();
            let ti = *self.type_cache.get(&(p, r)).unwrap_or(&0);
            funcs.function(ti);
            idx += 1;
        }
        let mti = *self.type_cache.get(&(vec![], vec![ValType::I32])).unwrap_or(&0);
        funcs.function(mti);
        module.section(&funcs);
        idx
    }

    fn emit_memory_section(&self, module: &mut Module) {
        let mut mem = MemorySection::new();
        mem.memory(MemoryType { minimum: 1, maximum: Some(256), memory64: false, shared: false, page_size_log2: None });
        module.section(&mem);
    }

    fn emit_export_section(&mut self, module: &mut Module, program: &HirProgram) {
        let mut exports = ExportSection::new();
        for func in &program.functions {
            if let Some(&fi) = self.func_map.get(&func.name) {
                // Don't export user-defined "main" — our wrapper handles it
                if func.name != "main" {
                    exports.export(&func.name, ExportKind::Func, fi);
                }
            }
        }
        let mi = self.func_map.len() as u32 + 6;
        exports.export("main", ExportKind::Func, mi);
        exports.export("memory", ExportKind::Memory, 0);
        module.section(&exports);
    }

    fn emit_code_section(&mut self, module: &mut Module, program: &HirProgram) -> TenthResult<()> {
        let mut codes = CodeSection::new();
        for func in &program.functions {
            codes.function(&self.compile_function(func)?);
        }
        codes.function(&self.compile_main(program)?);
        module.section(&codes);
        Ok(())
    }

    fn emit_data_section(&self, module: &mut Module) {
        let mut data = DataSection::new();
        data.active(0, &ConstExpr::i32_const(0), self.string_data.clone());
        module.section(&data);
    }

    // ── Function compilation ─────────────────────────────────────────────

    fn compile_function(&mut self, func: &HirFnDef) -> TenthResult<Function> {
        self.local_map.clear();
        self.local_count = 0;
        for (name, _) in &func.params {
            self.local_map.insert(name.clone(), self.local_count);
            self.local_count += 1;
        }
        // Reserve 64 i64 locals for let-bindings
        let locals: Vec<ValType> = (0..64).map(|_| ValType::I64).collect();
        let mut body = Function::new_with_locals_types(locals);
        self.compile_expr(&mut body, &func.body)?;
        if matches!(&func.return_type, Type::Base(BaseType::Unit)) {
            body.instruction(&Instruction::Return);
        }
        body.instruction(&Instruction::End);
        Ok(body)
    }

    fn compile_main(&mut self, program: &HirProgram) -> TenthResult<Function> {
        self.local_map.clear();
        self.local_count = 0;
        let locals: Vec<ValType> = (0..64).map(|_| ValType::I64).collect();
        let mut body = Function::new_with_locals_types(locals);
        if let Some(ref expr) = program.main_expr {
            self.compile_expr(&mut body, expr)?;
        } else if let Some(mf) = program.functions.iter().find(|f| f.name == "main") {
            let fi = self.resolve_func("main")?;
            body.instruction(&Instruction::Call(fi));
            if matches!(mf.return_type, Type::Base(BaseType::I64)) {
                body.instruction(&Instruction::Drop);
            }
        }
        body.instruction(&Instruction::I32Const(0));
        body.instruction(&Instruction::End);
        Ok(body)
    }

    fn resolve_func(&self, name: &str) -> TenthResult<u32> {
        match name {
            "println" | "eprintln" => Ok(0),
            "write_file" => Ok(1),
            "read_file" => Ok(2),
            "str_add" => Ok(3),
            "str_eq" => Ok(4),
            "str_int" => Ok(5),
            _ => self.func_map.get(name).copied()
                .ok_or_else(|| TenthError::RuntimeError {
                    message: format!("WASM: undefined function '{}'", name),
                }),
        }
    }

    // ── Expression compilation ───────────────────────────────────────────

    fn compile_expr(&mut self, body: &mut Function, expr: &HirExpr) -> TenthResult<()> {
        use HirExprKind;
        match &expr.kind {
            HirExprKind::Literal(lit) => self.compile_literal(body, lit)?,

            HirExprKind::Var(name) => {
                if let Some(&idx) = self.local_map.get(name) {
                    body.instruction(&Instruction::LocalGet(idx));
                } else if !["println","eprintln","write_file","read_file"].contains(&name.as_str()) {
                    return Err(TenthError::RuntimeError {
                        message: format!("WASM: undefined variable '{}'", name),
                    });
                }
            }

            HirExprKind::Binary { op, left, right, .. } => {
                self.compile_expr(body, left)?;
                self.compile_expr(body, right)?;
                self.compile_binop(body, op, &left.ty, &right.ty)?;
            }

            HirExprKind::Unary { op, expr: inner, .. } => {
                self.compile_expr(body, inner)?;
                self.compile_unary(body, op, &inner.ty)?;
            }

            HirExprKind::Call { func, args, .. } => {
                let fname = match &func.kind {
                    HirExprKind::Var(n) => n.clone(),
                    _ => return Err(TenthError::RuntimeError {
                        message: "WASM: indirect calls not supported".into(),
                    }),
                };
                match fname.as_str() {
                    "println" | "eprintln" => {
                        if let Some(a) = args.first() {
                            self.compile_string_arg(body, a)?;
                            body.instruction(&Instruction::Call(0));
                        }
                    }
                    "write_file" => {
                        if args.len() >= 2 {
                            self.compile_string_arg(body, &args[0])?;
                            self.compile_string_arg(body, &args[1])?;
                            body.instruction(&Instruction::Call(1));
                        }
                    }
                    "read_file" => {
                        if let Some(a) = args.first() {
                            self.compile_string_arg(body, a)?;
                            body.instruction(&Instruction::Call(2));
                        }
                    }
                    _ => {
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
                let has_result = !matches!(&then_branch.ty, Type::Base(BaseType::Unit));
                if has_result {
                    body.instruction(&Instruction::If(BlockType::Result(to_val_type_required(&then_branch.ty)?)));
                } else {
                    body.instruction(&Instruction::If(BlockType::Empty));
                }
                self.compile_expr(body, then_branch)?;
                if else_branch.is_some() {
                    body.instruction(&Instruction::Else);
                    self.compile_expr(body, else_branch.as_ref().unwrap())?;
                }
                body.instruction(&Instruction::End);
            }

            HirExprKind::Assign { target, value } => {
                self.compile_expr(body, value)?;
                if let Some(&idx) = self.local_map.get(target) {
                    body.instruction(&Instruction::LocalSet(idx));
                } else {
                    self.local_map.insert(target.clone(), self.local_count);
                    body.instruction(&Instruction::LocalSet(self.local_count));
                    self.local_count += 1;
                }
            }

            _ => return Err(TenthError::RuntimeError {
                message: format!("WASM: unsupported expr {:?}", expr.kind),
            }),
        }
        Ok(())
    }

    fn compile_stmt(&mut self, body: &mut Function, stmt: &HirStmt) -> TenthResult<()> {
        use crate::hir::hir::HirStmtKind;
        match &stmt.kind {
            HirStmtKind::Expr(e) => {
                self.compile_expr(body, e)?;
                if !matches!(&e.ty, Type::Base(BaseType::Unit)) {
                    body.instruction(&Instruction::Drop);
                }
            }
            HirStmtKind::Let { name, init, .. } => {
                if let Some(e) = init { self.compile_expr(body, e)?; }
                else { body.instruction(&Instruction::I64Const(0)); }
                self.local_map.insert(name.clone(), self.local_count);
                body.instruction(&Instruction::LocalSet(self.local_count));
                self.local_count += 1;
            }
            HirStmtKind::Loop { body: lb } => {
                body.instruction(&Instruction::Block(BlockType::Empty));
                body.instruction(&Instruction::Loop(BlockType::Empty));
                for s in lb { self.compile_stmt(body, s)?; }
                body.instruction(&Instruction::Br(0));
                body.instruction(&Instruction::End);
                body.instruction(&Instruction::End);
            }
            HirStmtKind::While { cond, body: lb } => {
                body.instruction(&Instruction::Block(BlockType::Empty));
                body.instruction(&Instruction::Loop(BlockType::Empty));
                self.compile_expr(body, cond)?;
                body.instruction(&Instruction::I32Eqz);
                body.instruction(&Instruction::BrIf(1));
                self.compile_stmt(body, lb.as_ref())?;
                body.instruction(&Instruction::Br(0));
                body.instruction(&Instruction::End);
                body.instruction(&Instruction::End);
            }
            HirStmtKind::Return(_) => { body.instruction(&Instruction::Return); }
            HirStmtKind::Break => { body.instruction(&Instruction::Br(1)); }
            HirStmtKind::Continue => { body.instruction(&Instruction::Br(0)); }
            _ => return Err(TenthError::RuntimeError {
                message: format!("WASM: unsupported stmt {:?}", stmt.kind),
            }),
        }
        Ok(())
    }

    // ── Literals & ops ────────────────────────────────────────────────────

    fn compile_literal(&mut self, body: &mut Function, lit: &Literal) -> TenthResult<()> {
        match lit {
            Literal::Int(n) => { body.instruction(&Instruction::I64Const(*n)); }
            Literal::Float(n) => { body.instruction(&Instruction::F64Const(*n)); }
            Literal::Bool(b) => { body.instruction(&Instruction::I32Const(if *b { 1 } else { 0 })); }
            Literal::String(s) => {
                let off = self.intern_string(s);
                body.instruction(&Instruction::I32Const(off as i32));
            }
        }
        Ok(())
    }

    fn compile_binop(&self, body: &mut Function, op: &BinOp, lty: &Type, rty: &Type) -> TenthResult<()> {
        let is_f = matches!(lty, Type::Base(BaseType::F32 | BaseType::F64))
            || matches!(rty, Type::Base(BaseType::F32 | BaseType::F64));
        match op {
            BinOp::Add => self.emit_if(body, is_f, Instruction::F64Add, Instruction::I64Add),
            BinOp::Sub => self.emit_if(body, is_f, Instruction::F64Sub, Instruction::I64Sub),
            BinOp::Mul => self.emit_if(body, is_f, Instruction::F64Mul, Instruction::I64Mul),
            BinOp::Div => {
                if !is_f {
                    body.instruction(&Instruction::F64ConvertI64S);
                    body.instruction(&Instruction::LocalSet(self.local_count));
                    body.instruction(&Instruction::F64ConvertI64S);
                    body.instruction(&Instruction::LocalGet(self.local_count));
                }
                body.instruction(&Instruction::F64Div);
            }
            BinOp::Mod => { body.instruction(&Instruction::I64RemS); }
            BinOp::Eq => self.emit_if(body, is_f, Instruction::F64Eq, Instruction::I64Eq),
            BinOp::NotEq => self.emit_if(body, is_f, Instruction::F64Ne, Instruction::I64Ne),
            BinOp::Lt => self.emit_if(body, is_f, Instruction::F64Lt, Instruction::I64LtS),
            BinOp::Gt => self.emit_if(body, is_f, Instruction::F64Gt, Instruction::I64GtS),
            BinOp::LtEq => self.emit_if(body, is_f, Instruction::F64Le, Instruction::I64LeS),
            BinOp::GtEq => self.emit_if(body, is_f, Instruction::F64Ge, Instruction::I64GeS),
            BinOp::And => { body.instruction(&Instruction::I32And); }
            BinOp::Or => { body.instruction(&Instruction::I32Or); }
        }
        Ok(())
    }

    fn emit_if(&self, body: &mut Function, cond: bool, then_i: Instruction<'_>, else_i: Instruction<'_>) {
        if cond { body.instruction(&then_i); } else { body.instruction(&else_i); }
    }

    fn compile_unary(&self, body: &mut Function, op: &UnaryOp, ty: &Type) -> TenthResult<()> {
        let is_f = matches!(ty, Type::Base(BaseType::F32 | BaseType::F64));
        match op {
            UnaryOp::Neg => if is_f {
                body.instruction(&Instruction::F64Neg);
            } else {
                body.instruction(&Instruction::LocalTee(self.local_count));
                body.instruction(&Instruction::I64Const(0));
                body.instruction(&Instruction::LocalGet(self.local_count));
                body.instruction(&Instruction::I64Sub);
            },
            UnaryOp::Not => { body.instruction(&Instruction::I32Eqz); }
        }
        Ok(())
    }

    fn compile_string_arg(&mut self, body: &mut Function, expr: &HirExpr) -> TenthResult<()> {
        match &expr.kind {
            HirExprKind::Literal(Literal::String(s)) => {
                body.instruction(&Instruction::I32Const(self.intern_string(s) as i32));
            }
            HirExprKind::Literal(Literal::Int(n)) => {
                body.instruction(&Instruction::I64Const(*n));
                body.instruction(&Instruction::Call(5));
            }
            HirExprKind::Literal(Literal::Float(n)) => {
                body.instruction(&Instruction::F64Const(*n));
                body.instruction(&Instruction::I64TruncF64S);
                body.instruction(&Instruction::Call(5));
            }
            HirExprKind::Binary { op: BinOp::Add, left, right, .. } => {
                self.compile_string_arg(body, left)?;
                self.compile_string_arg(body, right)?;
                body.instruction(&Instruction::Call(3));
            }
            _ => {
                self.compile_expr(body, expr)?;
                if matches!(&expr.ty, Type::Base(BaseType::I64)) {
                    body.instruction(&Instruction::Call(5));
                }
            }
        }
        Ok(())
    }

    // ── String interning ─────────────────────────────────────────────────

    fn intern_string(&mut self, s: &str) -> u32 {
        if let Some(&off) = self.string_offsets.get(s) { return off; }
        let off = self.string_data.len() as u32;
        self.string_data.extend_from_slice(s.as_bytes());
        self.string_data.push(0);
        self.string_offsets.insert(s.to_string(), off);
        off
    }

    fn collect_strings(&mut self, p: &HirProgram) {
        for f in &p.functions { self.cs_expr(&f.body); }
        if let Some(ref e) = p.main_expr { self.cs_expr(e); }
    }

    fn cs_expr(&mut self, e: &HirExpr) {
        use HirExprKind;
        match &e.kind {
            HirExprKind::Literal(Literal::String(s)) => { self.intern_string(s); }
            HirExprKind::Binary { left, right, .. } => { self.cs_expr(left); self.cs_expr(right); }
            HirExprKind::Unary { expr: inner, .. } => self.cs_expr(inner),
            HirExprKind::Call { args, .. } => { for a in args { self.cs_expr(a); } }
            HirExprKind::Block { stmts, final_expr } => {
                for s in stmts { self.cs_stmt(s); }
                if let Some(e) = final_expr { self.cs_expr(e); }
            }
            HirExprKind::If { cond, then_branch, else_branch, .. } => {
                self.cs_expr(cond); self.cs_expr(then_branch);
                if let Some(e) = else_branch { self.cs_expr(e); }
            }
            HirExprKind::Assign { value, .. } => self.cs_expr(value),
            _ => {}
        }
    }

    fn cs_stmt(&mut self, s: &HirStmt) {
        use crate::hir::hir::HirStmtKind;
        match &s.kind {
            HirStmtKind::Expr(e) => self.cs_expr(e),
            HirStmtKind::Let { init, .. } => { if let Some(e) = init { self.cs_expr(e); } }
            HirStmtKind::While { cond, body } => { self.cs_expr(cond); self.cs_stmt(body); }
            HirStmtKind::Loop { body } => { for s in body { self.cs_stmt(s); } }
            HirStmtKind::For { body, .. } => { self.cs_stmt(body); }
            _ => {}
        }
    }
}

// ── wasmi runtime ──────────────────────────────────────────────────────────

use wasmi::{Engine, Store, Linker, Caller};

/// Execute a WASM bytecode module in-process using wasmi.
pub fn run_wasm_module(wasm_bytes: &[u8]) -> TenthResult<()> {
    let engine = Engine::default();
    let module = wasmi::Module::new(&engine, wasm_bytes).map_err(|e| {
        TenthError::RuntimeError { message: format!("WASM module parse error: {}", e) }
    })?;

    let mut store = Store::new(&engine, ());
    let mut linker = Linker::new(&engine);

    linker.func_wrap("host", "println", |caller: Caller<'_, ()>, ptr: i32| {
        let mem = caller.get_export("memory").and_then(|e| e.into_memory()).unwrap();
        let data = mem.data(&caller);
        let end = data[ptr as usize..].iter().position(|&b| b == 0).unwrap_or(0);
        println!("{}", std::str::from_utf8(&data[ptr as usize..ptr as usize + end]).unwrap_or(""));
    }).map_err(|e| TenthError::RuntimeError { message: format!("linker: {}", e) })?;

    linker.func_wrap("host", "write_file",
        |caller: Caller<'_, ()>, path_ptr: i32, content_ptr: i32| {
            let mem = caller.get_export("memory").and_then(|e| e.into_memory()).unwrap();
            let data = mem.data(&caller);
            let rs = |p: i32| -> &str {
                let end = data[p as usize..].iter().position(|&b| b == 0).unwrap_or(0);
                std::str::from_utf8(&data[p as usize..p as usize + end]).unwrap_or("")
            };
            let _ = std::fs::write(rs(path_ptr), rs(content_ptr));
    }).map_err(|e| TenthError::RuntimeError { message: format!("linker: {}", e) })?;

    linker.func_wrap("host", "read_file",
        |caller: Caller<'_, ()>, path_ptr: i32| -> i32 {
            let mem = caller.get_export("memory").and_then(|e| e.into_memory()).unwrap();
            let data = mem.data(&caller);
            let end = data[path_ptr as usize..].iter().position(|&b| b == 0).unwrap_or(0);
            let _path = std::str::from_utf8(&data[path_ptr as usize..path_ptr as usize + end]).unwrap_or("");
            0i32
    }).map_err(|e| TenthError::RuntimeError { message: format!("linker: {}", e) })?;

    linker.func_wrap("host", "str_add",
        |caller: Caller<'_, ()>, a_ptr: i32, b_ptr: i32| -> i32 {
            let mem = caller.get_export("memory").and_then(|e| e.into_memory()).unwrap();
            let data = mem.data(&caller);
            let rs = |p: i32| -> &str {
                let end = data[p as usize..].iter().position(|&b| b == 0).unwrap_or(0);
                std::str::from_utf8(&data[p as usize..p as usize + end]).unwrap_or("")
            };
            let _ = format!("{}{}", rs(a_ptr), rs(b_ptr));
            a_ptr
    }).map_err(|e| TenthError::RuntimeError { message: format!("linker: {}", e) })?;

    linker.func_wrap("host", "str_eq",
        |caller: Caller<'_, ()>, a_ptr: i32, b_ptr: i32| -> i32 {
            let mem = caller.get_export("memory").and_then(|e| e.into_memory()).unwrap();
            let data = mem.data(&caller);
            let rs = |p: i32| -> &str {
                let end = data[p as usize..].iter().position(|&b| b == 0).unwrap_or(0);
                std::str::from_utf8(&data[p as usize..p as usize + end]).unwrap_or("")
            };
            if rs(a_ptr) == rs(b_ptr) { 1 } else { 0 }
    }).map_err(|e| TenthError::RuntimeError { message: format!("linker: {}", e) })?;

    linker.func_wrap("host", "str_int",
        |mut caller: Caller<'_, ()>, n: i64| -> i32 {
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
    }).map_err(|e| TenthError::RuntimeError { message: format!("linker: {}", e) })?;

    let instance = linker.instantiate(&mut store, &module)
        .and_then(|pre| pre.start(&mut store))
        .map_err(|e| TenthError::RuntimeError {
            message: format!("WASM instantiation error: {}", e),
        })?;

    let main_fn = instance.get_typed_func::<(), i32>(&store, "main")
        .map_err(|_| TenthError::RuntimeError {
            message: "WASM module has no exported 'main' function".into(),
        })?;

    let exit_code = main_fn.call(&mut store, ())
        .map_err(|e| TenthError::RuntimeError {
            message: format!("WASM main() error: {}", e),
        })?;

    if exit_code != 0 {
        return Err(TenthError::RuntimeError {
            message: format!("WASM main() exited with code {}", exit_code),
        });
    }

    Ok(())
}
