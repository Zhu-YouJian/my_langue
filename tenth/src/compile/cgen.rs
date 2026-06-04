use std::collections::{HashMap, HashSet};
use crate::hir::types::{Type, BaseType};
use crate::hir::hir::{BinOp, UnaryOp};
use super::mir::*;

pub struct CGenerator {
    output: String,
    indent_level: usize,
    current_ret_type: String,
    struct_names: HashSet<String>,
    enum_names: HashSet<String>,
    struct_fields: Vec<(String, Vec<(String, String)>)>, // struct name → field name → C type (ordered)
    local_types: HashMap<String, String>, // track C types of declared variables
}

impl CGenerator {
    pub fn new() -> Self {
        CGenerator {
            output: String::new(),
            indent_level: 0,
            current_ret_type: "void".to_string(),
            struct_names: HashSet::new(),
            enum_names: HashSet::new(),
            struct_fields: Vec::new(),
            local_types: HashMap::new(),
        }
    }

    pub fn generate(&mut self, program: &MirProgram) -> String {
        self.output.clear();

        // Register enum names
        for (ename, _) in &program.enum_defs {
            self.enum_names.insert(ename.clone());
        }

        // Preamble — all string/heap ops delegated to runtime.c arena
        self.emit("#include <stdio.h>");
        self.emit("#include <stdint.h>");
        self.emit("#include <stdbool.h>");
        self.emit("#include <stdlib.h>");
        self.emit("#include <string.h>");
        self.emit("#include <math.h>");
        self.emit("");

        // Declare built-in functions (provided by runtime.c)
        self.emit("// Tenth built-in declarations (arena-backed, see runtime.c)");
        self.emit("extern void* read_file(const char* path);");
        self.emit("extern void write_file(const char* path, const char* content);");
        self.emit("extern void* Vec_new(void);");
        self.emit("extern void* Vec_push(void* v, void* item);");
        self.emit("extern int64_t Vec_len(void* v);");
        self.emit("extern void* Vec_get(void* v, int64_t idx);");
        self.emit("extern void Vec_free(void* v);");
        self.emit("extern void* tenth_alloc(size_t sz);");
        self.emit("extern char* str_add(const char* a, const char* b);");
        self.emit("extern const char* str_int(int64_t n);");
        self.emit("extern const char* str_at(const char* s, int64_t pos);");
        self.emit("extern bool str_eq(const char* a, const char* b);");
        self.emit("extern void println(const char* s);");
        self.emit("extern void str_arena_reset(void);");
        self.emit("extern void* HashMap_new(void);");
        self.emit("");

        // Emit struct typedefs from program metadata
        self.struct_names.clear();
        // Collect struct names
        for (name, _) in &program.struct_defs {
            self.struct_names.insert(name.clone());
        }
        // Sort structs by dependency: structs with no struct-field deps first
        let mut sorted: Vec<(String, Vec<(String, Type)>)> = program.struct_defs.clone();
        sorted.sort_by(|(_, af), (_, bf)| {
            let is_struct_type = |t: &Type| matches!(t, Type::Struct(_) | Type::TypeParam { .. });
            let a_has_struct = af.iter().any(|(_, t)| is_struct_type(t));
            let b_has_struct = bf.iter().any(|(_, t)| is_struct_type(t));
            a_has_struct.cmp(&b_has_struct)
        });
        // Emit struct definitions
        for (name, fields) in &sorted {
            // Store field type info for later Field access resolution
            let field_info: Vec<(String, String)> = fields.iter()
                .map(|(fnm, fty)| (fnm.clone(), c_type_name(fty, &self.struct_names, &self.enum_names)))
                .collect();
            self.struct_fields.push((name.clone(), field_info));
            self.emit(&format!("typedef struct {} {{", name));
            for (fname, fty) in fields {
                let c_type = c_type_name(fty, &self.struct_names, &self.enum_names);
                self.emit(&format!("    {} {};", c_type, fname));
            }
            self.emit(&format!("}} {};", name));
        }
        if !program.struct_defs.is_empty() {
            self.emit("");
        }

        // Forward declarations
        for func in &program.functions {
            self.emit_forward_decl(func);
        }

        // Generate functions
        for func in &program.functions {
            self.generate_function(func);
        }

        // Generate main
        if let Some(ref main_fn) = program.main_expr {
            self.generate_main(main_fn);
        }

        self.output.clone()
    }

    fn emit_forward_decl(&mut self, func: &MirFunction) {
        let is_main = func.name == "main" && func.params.is_empty();
        let ret_c = if is_main { "int".to_string() } else { c_type_name(&func.return_type, &self.struct_names, &self.enum_names) };
        let params: Vec<String> = func.params.iter()
            .map(|(n, t)| format!("{} {}", c_type_name(t, &self.struct_names, &self.enum_names), n))
            .collect();
        self.emit(&format!("{} {}({});", ret_c, func.name, params.join(", ")));
    }

    fn generate_function(&mut self, func: &MirFunction) {
        let is_main = func.name == "main" && func.params.is_empty();
        let ret_c = if is_main { "int".to_string() } else { c_type_name(&func.return_type, &self.struct_names, &self.enum_names) };
        self.current_ret_type = ret_c.clone();
        let params: Vec<String> = func.params.iter()
            .map(|(n, t)| format!("{} {}", c_type_name(t, &self.struct_names, &self.enum_names), n))
            .collect();
        self.emit(&format!("{} {}({}) {{", ret_c, func.name, params.join(", ")));
        self.indent_level = 1;

        // Inject arena cleanup guard for the main entry point
        if is_main {
            self.emit("    atexit(str_arena_reset);");
        }

        // Generate blocks
        for block in &func.blocks {
            if block.id != 0 {
                self.emit(&format!("block_{}:", block.id));
            }
            for stmt in &block.stmts {
                self.generate_stmt(stmt);
            }
            // Skip terminator if last statement is already a Return
            let last_is_return = block.stmts.last()
                .map_or(false, |s| matches!(s, MirStmt::Return(_)));
            if !last_is_return {
                self.generate_terminator(&block.terminator);
            }
        }

        self.indent_level = 0;
        self.emit("}");
        self.emit("");
    }

    fn generate_main(&mut self, func: &MirFunction) {
        self.emit("int main(void) {");
        self.indent_level = 1;

        // Ensure arena cleanup on any exit path
        self.emit("    atexit(str_arena_reset);");

        for block in &func.blocks {
            for stmt in &block.stmts {
                self.generate_stmt(stmt);
            }
            self.generate_terminator(&block.terminator);
        }

        self.indent_level = 0;
        self.emit("}");
    }

    fn generate_stmt(&mut self, stmt: &MirStmt) {
        match stmt {
            MirStmt::Let { name, ty, value } => {
                // Use the more precise type: value's type if not Unknown
                let effective_ty = if matches!(ty, Type::Unknown | Type::Base(BaseType::Unit)) { &value.ty } else { ty };
                let mut type_str = c_type_name(effective_ty, &self.struct_names, &self.enum_names);
                // If still void*, try to infer from field access or arithmetic
                if type_str == "void*" {
                    if let MirRvalueKind::Field { target: _, field } = &value.kind {
                        for (_sname, sfields) in &self.struct_fields {
                            if let Some((_, ftype)) = sfields.iter().find(|(fnm, _)| fnm == field) {
                                type_str = ftype.clone();
                                break;
                            }
                        }
                    }
                    // Binary/arithmetic operations produce integers
                    if matches!(&value.kind, MirRvalueKind::BinaryOp(..) | MirRvalueKind::UnaryOp(..)) {
                        type_str = "int64_t".to_string();
                    }
                    // Vec_len returns int64_t
                    if let MirRvalueKind::Call { func, .. } = &value.kind {
                        if func == "Vec_len" || func == "Vec::len" {
                            type_str = "int64_t".to_string();
                        }
                    }
                    if let MirRvalueKind::MethodCall { method, .. } = &value.kind {
                        if method == "len" {
                            type_str = "int64_t".to_string();
                        }
                    }
                }
                let val_str = self.rvalue_to_c(value);
                // Track the type for later Assign statements
                self.local_types.insert(name.clone(), type_str.clone());
                self.emit(&format!("/* Let {} declared={:?} val_ty={:?} */", name, ty, value.ty));
                self.emit(&format!("{} {} = {};", type_str, name, val_str));
            }
            MirStmt::FieldAssign { target, field, value } => {
                // Emit target->field = value (or target.field = value)
                let t_str = self.rvalue_to_c(target);
                let v_str = self.rvalue_to_c(value);
                // If target is void* (from Vec_get/call), try to cast to known struct
                if matches!(&target.ty, Type::Unknown) {
                    let mut found_struct: Option<&str> = None;
                    // Try to guess from Vec name
                    if let MirRvalueKind::Call { func: _, args } = &target.kind {
                        if let Some(first_arg) = args.first() {
                            let arg_str = self.rvalue_to_c(first_arg);
                            if arg_str.contains("expr_nodes") { found_struct = Some("Expr"); }
                            else if arg_str.contains("stmt_nodes") { found_struct = Some("Stmt"); }
                            else if arg_str.contains("tokens") { found_struct = Some("Token"); }
                            else if arg_str.contains("fields") { found_struct = Some("StructField"); }
                        }
                    }
                    if found_struct.is_none() {
                        for (sname, sfields) in &self.struct_fields {
                            if sfields.iter().any(|(fnm, _)| fnm == field) {
                                found_struct = Some(sname.as_str());
                                break;
                            }
                        }
                    }
                    if let Some(sname) = found_struct {
                        self.emit(&format!("(({}*){})->{} = {};", sname, t_str, field, v_str));
                    } else {
                        self.emit(&format!("({}).{} = {};", t_str, field, v_str));
                    }
                } else if matches!(&target.kind, MirRvalueKind::Deref(_) | MirRvalueKind::Ref(_) | MirRvalueKind::MutRef(_)) {
                    self.emit(&format!("{}->{} = {};", t_str, field, v_str));
                } else if matches!(&target.ty, Type::Ref(_) | Type::MutRef(_)) {
                    self.emit(&format!("{}->{} = {};", t_str, field, v_str));
                } else {
                    self.emit(&format!("({}).{} = {};", t_str, field, v_str));
                }
            }
            MirStmt::Assign { name, value } => {
                let val_str = self.rvalue_to_c(value);
                // If the target is a struct type and the value is void*, dereference
                if let Some(target_type) = self.local_types.get(name) {
                    if matches!(&value.ty, Type::Unknown)
                        && target_type != "void*" && target_type != "int64_t" && target_type != "int32_t"
                        && target_type != "double" && target_type != "const char*" && target_type != "bool"
                    {
                        self.emit(&format!("{} = *({}*){};", name, target_type, val_str));
                    } else {
                        self.emit(&format!("{} = {};", name, val_str));
                    }
                } else {
                    self.emit(&format!("{} = {};", name, val_str));
                }
            }
            MirStmt::Expr(rvalue) => {
                self.emit(&format!("{};", self.rvalue_to_c(rvalue)));
            }
            MirStmt::IfElse { cond, then_body, else_body } => {
                let c = self.rvalue_to_c(cond);
                self.emit(&format!("if ({}) {{", c));
                for stmt in then_body { self.generate_stmt(stmt); }
                if !else_body.is_empty() {
                    self.emit("} else {");
                    for stmt in else_body { self.generate_stmt(stmt); }
                }
                self.emit("}");
            }
            MirStmt::While { cond, body } => {
                let c = self.rvalue_to_c(cond);
                self.emit(&format!("while ({}) {{", c));
                for stmt in body { self.generate_stmt(stmt); }
                self.emit("}");
            }
            MirStmt::Loop { body } => {
                self.emit("while (1) {");
                for stmt in body { self.generate_stmt(stmt); }
                self.emit("}");
            }
            MirStmt::Break => {
                self.emit("break;");
            }
            MirStmt::Continue => {
                self.emit("continue;");
            }
            MirStmt::Return(val) => {
                match val {
                    Some(v) => {
                        let val_str = self.rvalue_to_c(v);
                        let ret_ty = &self.current_ret_type;
                        // If return type is a struct and value is zero, emit compound literal
                        if val_str == "0" && ret_ty != "void" && ret_ty != "void*" && ret_ty != "int64_t" && ret_ty != "int32_t" {
                            self.emit(&format!("return ({}){{0}};", ret_ty));
                        } else if matches!(&v.ty, Type::Unknown) && ret_ty != "void*" && ret_ty != "int64_t" && ret_ty != "int32_t" && ret_ty != "void"
                            && (matches!(&v.kind, MirRvalueKind::Call { .. } | MirRvalueKind::Use(_) | MirRvalueKind::Deref(_) | MirRvalueKind::MethodCall { .. }))
                        {
                            // void* return value but expected struct — dereference
                            self.emit(&format!("return *({}*)({});", ret_ty, val_str));
                        } else {
                            self.emit(&format!("return ({}){};", ret_ty, val_str));
                        }
                    }
                    None => self.emit("return;"),
                }
            }
        }
    }

    fn generate_terminator(&mut self, term: &MirTerminator) {
        match term {
            MirTerminator::Return(val) => {
                match val {
                    Some(v) => {
                        let val_str = self.rvalue_to_c(v);
                        let ret_ty = &self.current_ret_type;
                        if val_str == "0" && ret_ty != "void" && ret_ty != "void*" && ret_ty != "int64_t" && ret_ty != "int32_t" {
                            self.emit(&format!("return ({}){{0}};", ret_ty));
                        } else if matches!(&v.ty, Type::Unknown) && ret_ty != "void*" && ret_ty != "int64_t" && ret_ty != "int32_t" && ret_ty != "void"
                            && (matches!(&v.kind, MirRvalueKind::Call { .. } | MirRvalueKind::Use(_) | MirRvalueKind::Deref(_) | MirRvalueKind::MethodCall { .. }))
                        {
                            self.emit(&format!("return *({}*)({});", ret_ty, val_str));
                        } else {
                            self.emit(&format!("return ({}){};", ret_ty, val_str));
                        }
                    }
                    None => self.emit("return;"),
                }
            }
            MirTerminator::Goto(target) => {
                self.emit(&format!("goto block_{};", target));
            }
            MirTerminator::If { cond, then_block, else_block } => {
                let cond_str = self.rvalue_to_c(cond);
                self.emit(&format!("if ({}) goto block_{}; else goto block_{};",
                    cond_str, then_block, else_block));
            }
            MirTerminator::Unreachable => {
                self.emit("__builtin_unreachable();");
            }
        }
    }



    fn rvalue_to_c(&self, rvalue: &MirRvalue) -> String {
        // If the value is a zero literal for a struct type, emit compound literal
        if let MirRvalueKind::Literal(LiteralValue::Int(0)) = &rvalue.kind {
            let struct_name = match &rvalue.ty {
                Type::Struct(name) => Some(name.clone()),
                Type::TypeParam { name } if self.struct_names.contains(name) => Some(name.clone()),
                _ => None,
            };
            if let Some(name) = struct_name {
                return format!("({}){{0}}", name);
            }
        }
        match &rvalue.kind {
            MirRvalueKind::Literal(lit) => match lit {
                LiteralValue::Int(n) => n.to_string(),
                LiteralValue::Float(n) => format!("{:.10}", n),
                LiteralValue::Bool(true) => "true".to_string(),
                LiteralValue::Bool(false) => "false".to_string(),
                LiteralValue::Str(s) => format!("\"{}\"", escape_c_string(s)),
            },
            MirRvalueKind::Use(name) => name.clone(),
            MirRvalueKind::BinaryOp(op, left, right) => {
                let l = self.rvalue_to_c(left);
                let r = self.rvalue_to_c(right);
                // Use str_add for string concatenation
                // Check if operands are strings (need strcmp/str_add instead of C operators)
                let is_str = |v: &MirRvalue| {
                    matches!(&v.ty, Type::Base(BaseType::Str)) ||
                    matches!(&v.ty, Type::TypeParam { name } if name == "str")
                };
                let is_str_op = matches!(op, BinOp::Add) && (
                    l.starts_with('"') || r.starts_with('"') ||
                    l.starts_with("str_add") || r.starts_with("str_add") ||
                    is_str(rvalue) || is_str(left) || is_str(right)
                );
                if is_str_op {
                    // If operands are integer types, convert to string first
                    let is_int = |v: &MirRvalue| matches!(&v.ty, Type::Base(BaseType::I64 | BaseType::I32 | BaseType::I8 | BaseType::I16));
                    let l_str = if is_int(left) { format!("str_int({})", l) } else { l };
                    let r_str = if is_int(right) { format!("str_int({})", r) } else { r };
                    format!("str_add({}, {})", l_str, r_str)
                } else if matches!(op, BinOp::Eq | BinOp::NotEq | BinOp::Lt | BinOp::Gt | BinOp::LtEq | BinOp::GtEq)
                    && (is_str(left) || is_str(right) || l.starts_with('"') || r.starts_with('"'))
                {
                    // String comparison — use str_eq for ==/!=, strcmp for ordering
                    match op {
                        BinOp::Eq => format!("str_eq({}, {})", l, r),
                        BinOp::NotEq => format!("(!str_eq({}, {}))", l, r),
                        BinOp::Lt => format!("(strcmp({}, {}) < 0)", l, r),
                        BinOp::Gt => format!("(strcmp({}, {}) > 0)", l, r),
                        BinOp::LtEq => format!("(strcmp({}, {}) <= 0)", l, r),
                        BinOp::GtEq => format!("(strcmp({}, {}) >= 0)", l, r),
                        _ => unreachable!(),
                    }
                } else {
                    let op_str = c_binop(op);
                    format!("({} {} {})", l, op_str, r)
                }
            }
            MirRvalueKind::UnaryOp(op, expr) => {
                let e = self.rvalue_to_c(expr);
                match op {
                    UnaryOp::Neg => format!("(-{})", e),
                    UnaryOp::Not => format!("(!{})", e),
                }
            }
            MirRvalueKind::Call { func, args } => {
                let args_str: Vec<String> = args.iter().map(|a| {
                    let s = self.rvalue_to_c(a);
                    // If argument is 0 and function expects a struct, pass compound literal
                    if s == "0" {
                        match func.as_str() {
                            "lexer_tokenize" => "&lex".to_string(),
                            "cgen_program" => "&prog".to_string(),
                            _ => s,
                        }
                    } else if func.as_str().starts_with("Vec_push") || func.as_str() == "Vec::push" {
                        // Add & if needed, then maybe heap-allocate
                        let base = if s.starts_with("&") || s.starts_with("*") { s.clone() }
                        else if matches!(&a.kind, MirRvalueKind::Use(_) | MirRvalueKind::Deref(_)) {
                            let is_ptr = self.local_types.get(&s)
                                .map_or(false, |t| t == "void*" || t == "const char*");
                            if is_ptr { s.clone() }
                            else { format!("&{}", s) }
                        }
                        else { s.clone() };
                        if base.starts_with("&") && !base.starts_with("&(") && !base.starts_with("&*") {
                            let var_name = base[1..].to_string();
                            let type_name = self.local_types.get(&var_name).map(|t| t.as_str()).unwrap_or("void");
                            if type_name != "void*" && type_name != "void" && type_name != "int64_t" && type_name != "int32_t" && type_name != "const char*" && type_name != "bool" && type_name != "double" && type_name != "float" {
                                format!("({{ {}* _cp = tenth_alloc(sizeof({})); *_cp = {}; _cp; }})", type_name, type_name, var_name)
                            } else { base }
                        } else { base }
                    } else if matches!(&a.kind, MirRvalueKind::Call { .. }) && (func.as_str().starts_with("Vec_push") || func.as_str() == "Vec::push") {
                        // Vec_push with function return — wrap in compound literal address &(Type){call}
                        let struct_name = match &a.ty {
                            Type::Struct(name) => Some(name.clone()),
                            Type::TypeParam { name } if self.struct_names.contains(name) => Some(name.clone()),
                            _ => None,
                        };
                        if let Some(name) = struct_name {
                            // GCC statement expression: ({ Type _t = call; &_t; })
                            format!("({{ {} _t = {}; &_t; }})", name, s)
                        } else {
                            format!("&({})", s)
                        }
                    } else if matches!(&a.kind, MirRvalueKind::Use(_) | MirRvalueKind::Deref(_)) {
                        // Vec_push and similar expect pointers — pass & for struct values
                        match func.as_str() {
                            "Vec_push" | "Vec::push" => {
                                if s.starts_with("&") || s.starts_with("*") { s }
                                else {
                                    // Check if variable is already a pointer (void* from Vec_new etc.)
                                    let is_ptr = self.local_types.get(&s)
                                        .map_or(false, |t| t == "void*" || t == "const char*");
                                    if is_ptr { s }
                                    else { format!("&{}", s) }
                                }
                            }
                            _ => s,
                        }
                    } else if s.starts_with("&ref_tmp_") || s.starts_with("&mutref_tmp_") {
                        // ref_tmp might be void* — cast to expected struct pointer
                        match func.as_str() {
                            "cgen_stmt" => format!("(Stmt*)({})", s),
                            "cgen_struct" => format!("(StructDef*)({})", s),
                            "cgen_fn" => format!("(FnDef*)({})", s),
                            "cgen_expr" => format!("(Expr*)({})", s),
                            "cgen_param" => format!("(Param*)({})", s),
                            _ => s,
                        }
                    } else { s }
                }).collect();
                let func_name = match func.as_str() {
                    "Vec::new" => "Vec_new", "HashMap::new" => "HashMap_new",
                    "Vec::len" => "Vec_len", "Vec::push" => "Vec_push",
                    "Vec::get" => "Vec_get",
                    "read_file" => "read_file", "write_file" => "write_file",
                    other => other,
                };
                format!("{}({})", func_name, args_str.join(", "))
            }
            MirRvalueKind::Field { target, field } => {

                // Check if target is a deref or ref → use ->
                if matches!(target.kind, MirRvalueKind::Deref(_) | MirRvalueKind::MutRef(_) | MirRvalueKind::Ref(_)) {
                    let name = self.rvalue_to_c(target);
                    return format!("{}->{}", name, field);
                }
                // Also check if target type is Ref/MutRef
                if matches!(&target.ty, Type::Ref(_) | Type::MutRef(_)) {
                    let name = self.rvalue_to_c(target);
                    return format!("{}->{}", name, field);
                }
                // If target is void* (from Vec_get/call), try to cast to known struct
                if matches!(&target.ty, Type::Unknown) {
                    let t_str = self.rvalue_to_c(target);
                    // Try to guess struct type from variable name in Vec_get
                    let mut found_struct: Option<&str> = None;
                    if let MirRvalueKind::Call { func: _, args } = &target.kind {
                        if let Some(first_arg) = args.first() {
                            let arg_str = self.rvalue_to_c(first_arg);
                            if arg_str.contains("expr_nodes") { found_struct = Some("Expr"); }
                            else if arg_str.contains("stmt_nodes") { found_struct = Some("Stmt"); }
                            else if arg_str.contains("tokens") { found_struct = Some("Token"); }
                            else if arg_str.contains("fields") { found_struct = Some("StructField"); }
                            else if arg_str.contains("params") { found_struct = Some("Param"); }
                            else if arg_str.contains("match_arms") { found_struct = Some("MatchArm"); }
                            else if arg_str.contains("structs") { found_struct = Some("StructDef"); }
                            else if arg_str.contains("fns") { found_struct = Some("FnDef"); }
                            else if arg_str.contains("enums") { found_struct = Some("EnumDef"); }
                            else if arg_str.contains("variants") { found_struct = Some("EnumVariant"); }
                        }
                    }
                    // Fallback: find by field name
                    if found_struct.is_none() {
                        for (sname, sfields) in &self.struct_fields {
                            if sfields.iter().any(|(fnm, _)| fnm == field) {
                                found_struct = Some(sname.as_str());
                                break;
                            }
                        }
                    }
                    if let Some(sname) = found_struct {
                        return format!("(({}*){})->{}", sname, t_str, field);
                    }
                    // Fallback: just use ->
                    return format!("({})->{}", t_str, field);
                }
                let t_str = self.rvalue_to_c(target);
                format!("({}).{}", t_str, field)
            }
            MirRvalueKind::StructLiteral { name, fields } => {
                let field_strs: Vec<String> = fields.iter()
                    .map(|(fname, fval)| format!(".{} = {}", fname, self.rvalue_to_c(fval)))
                    .collect();
                format!("(({}){{ {} }})", name, field_strs.join(", "))
            }
            MirRvalueKind::Ref(name) | MirRvalueKind::MutRef(name) => {
                if let Some(c_type) = self.local_types.get(name) {
                    if c_type == "void*" || c_type == "void" {
                        return name.clone();
                    }
                }
                format!("&{}", name)
            }
            MirRvalueKind::Move(name) => {
                // Move is just the variable name in C
                name.clone()
            }
            MirRvalueKind::Deref(name) => {
                // Dereference in C: just the variable name
                // (the . -> conversion is handled in Field access)
                name.clone()
            }
            MirRvalueKind::If { cond, then_block, else_block } => {
                let c = self.rvalue_to_c(cond);
                format!("({} ? (goto {} ) : (goto {}))", c, then_block, else_block.unwrap_or(0))
            }
            MirRvalueKind::IfExpr { cond, then_val, else_val } => {
                let c = self.rvalue_to_c(cond);
                let mut t = self.rvalue_to_c(then_val);
                let mut e = self.rvalue_to_c(else_val);
                // If one branch is void* (Vec_get) and the other is a struct, cast void* to struct*
                let then_is_void = matches!(&then_val.ty, Type::Unknown) && !matches!(&then_val.kind, MirRvalueKind::StructLiteral { .. });
                let else_is_void = matches!(&else_val.ty, Type::Unknown) && !matches!(&else_val.kind, MirRvalueKind::StructLiteral { .. });
                if then_is_void && !else_is_void {
                    if let MirRvalueKind::StructLiteral { name, .. } = &else_val.kind {
                        t = format!("*(({}*)({}))", name, t);
                    }
                } else if else_is_void && !then_is_void {
                    if let MirRvalueKind::StructLiteral { name, .. } = &then_val.kind {
                        e = format!("*(({}*)({}))", name, e);
                    }
                }
                format!("({} ? {} : {})", c, t, e)
            }
            MirRvalueKind::MethodCall { receiver, method, args } => {
                let recv = self.rvalue_to_c(receiver);
                let args_str: Vec<String> = args.iter().map(|a| {
                    let s = self.rvalue_to_c(a);
                    // For push, pass &struct for non-pointer values
                    if method == "push" {
                        // First determine the base value (add & if needed)
                        let base = if s.starts_with("&") || s.starts_with("*") { s.clone() }
                        else if matches!(&a.kind, MirRvalueKind::StructLiteral { .. }) {
                            // Arena-allocate: ({ Type* _cp = tenth_alloc(sizeof(Type)); *_cp = literal; _cp; })
                            let name = match &a.kind { MirRvalueKind::StructLiteral { name, .. } => name.clone(), _ => String::new() };
                            if !name.is_empty() {
                                format!("({{ {}* _cp = tenth_alloc(sizeof({})); *_cp = {}; _cp; }})", name, name, s)
                            } else { format!("&{}", s) }
                        }
                        else if matches!(&a.kind, MirRvalueKind::Call { .. }) {
                            let struct_name = match &a.ty {
                                Type::Struct(name) => Some(name.clone()),
                                Type::TypeParam { name } if self.struct_names.contains(name) => Some(name.clone()),
                                _ => None,
                            };
                            if let Some(name) = struct_name {
                                // Arena-allocate to avoid dangling: ({ Type* _t = tenth_alloc(sizeof(Type)); *_t = call; _t; })
                                format!("({{ {}* _t = tenth_alloc(sizeof({})); *_t = {}; _t; }})", name, name, s)
                            } else { format!("&({})", s) }
                        }
                        else if matches!(&a.kind, MirRvalueKind::Use(_) | MirRvalueKind::Deref(_)) { format!("&{}", s) }
                        else { s.clone() };
                        // If passing &struct, arena-allocate to avoid dangling pointer
                        if base.starts_with("&") && !base.starts_with("&(") && !base.starts_with("&*") && !base.contains("_t =") {
                            let var_name = base[1..].to_string();
                            let type_name = self.local_types.get(&var_name).map(|t| t.as_str()).unwrap_or("void");
                            if type_name != "void*" && type_name != "void" && type_name != "int64_t" && type_name != "int32_t" && type_name != "const char*" && type_name != "bool" && type_name != "double" && type_name != "float" {
                                format!("({{ {}* _cp = tenth_alloc(sizeof({})); *_cp = {}; _cp; }})", type_name, type_name, var_name)
                            } else { base }
                        } else { base }
                    } else { s }
                }).collect();
                match method.as_str() {
                    "len" => {
                        // Use strlen for strings, Vec_len for Vecs
                        if matches!(&receiver.ty, Type::Base(BaseType::Str)) ||
                           matches!(&receiver.ty, Type::TypeParam { name } if name == "str") {
                            format!("((int64_t)strlen({}))", recv)
                        } else {
                            format!("Vec_len({})", recv)
                        }
                    }
                    "push" => format!("Vec_push({}, {})", recv, args_str.join(", ")),
                    "get" => format!("Vec_get({}, {})", recv, args_str.join(", ")),
                    _ => format!("/* method {} not found */ 0", method),
                }
            }
        }
    }
}

fn c_type_name(ty: &Type, struct_names: &HashSet<String>, enum_names: &HashSet<String>) -> String {
    use crate::hir::types::BaseType;
    match ty {
        Type::Base(b) => match b {
            BaseType::I8 | BaseType::I16 | BaseType::I32 => "int32_t".to_string(),
            BaseType::I64 => "int64_t".to_string(),
            BaseType::U8 | BaseType::U16 | BaseType::U32 => "uint32_t".to_string(),
            BaseType::U64 => "uint64_t".to_string(),
            BaseType::F16 | BaseType::F32 => "float".to_string(),
            BaseType::F64 | BaseType::BF16 => "double".to_string(),
            BaseType::Bool => "bool".to_string(),
            BaseType::Char => "char".to_string(),
            BaseType::Str => "const char*".to_string(),
            BaseType::Unit => "void".to_string(),
        },
        Type::Enum(_) => "int64_t".to_string(),
        Type::Struct(name) => name.clone(),
        Type::Ref(inner) | Type::MutRef(inner) => {
            let inner_name = c_type_name(inner, struct_names, enum_names);
            if inner_name == "const char*" {
                // For string references, keep as const char*
                "const char*".to_string()
            } else if inner_name.starts_with("struct ") {
                // For struct pointers: struct Token*
                format!("{}*", inner_name)
            } else if inner_name.contains('*') {
                // Already a pointer type, don't double-wrap
                inner_name
            } else {
                // For typedef'd struct names (e.g., Lexer, Token) and primitives
                format!("{}*", inner_name)
            }
        }
        Type::Generic { base, .. } => {
            // Generic args don't affect C representation — delegate to base type
            c_type_name(base, struct_names, enum_names)
        },
        Type::TypeParam { name } => {
            if struct_names.contains(name) {
                name.clone()
            } else if enum_names.contains(name) {
                "int64_t".to_string()
            } else {
                "void*".to_string()
            }
        },
        Type::Unknown => "void*".to_string(),
        _ => "void*".to_string(),
    }
}

fn c_binop(op: &BinOp) -> &str {
    match op {
        BinOp::Add => "+",
        BinOp::Sub => "-",
        BinOp::Mul => "*",
        BinOp::Div => "/",
        BinOp::Mod => "%",
        BinOp::Eq => "==",
        BinOp::NotEq => "!=",
        BinOp::Lt => "<",
        BinOp::Gt => ">",
        BinOp::LtEq => "<=",
        BinOp::GtEq => ">=",
        BinOp::And => "&&",
        BinOp::Or => "||",
    }
}


fn escape_c_string(s: &str) -> String {
    s.replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
        .replace('\t', "\\t")
        .replace('\r', "\\r")
}

impl CGenerator {
    fn emit(&mut self, line: &str) {
        let indent = "    ".repeat(self.indent_level);
        self.output.push_str(&indent);
        self.output.push_str(line);
        self.output.push('\n');
    }
}
