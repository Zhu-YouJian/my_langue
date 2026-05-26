use std::collections::{HashMap, HashSet};
use crate::hir::types::{Type, BaseType};
use crate::hir::hir::{BinOp, UnaryOp};
use super::mir::*;

pub struct CGenerator {
    output: String,
    indent_level: usize,
    var_types: HashMap<String, String>,
    current_ret_type: String,
    struct_names: HashSet<String>,
}

impl CGenerator {
    pub fn new() -> Self {
        CGenerator {
            output: String::new(),
            indent_level: 0,
            var_types: HashMap::new(),
            current_ret_type: "void".to_string(),
            struct_names: HashSet::new(),
        }
    }

    pub fn generate(&mut self, program: &MirProgram) -> String {
        self.output.clear();

        // Preamble
        self.emit("#include <stdio.h>");
        self.emit("#include <stdint.h>");
        self.emit("#include <stdbool.h>");
        self.emit("#include <stdlib.h>");
        self.emit("#include <string.h>");
        self.emit("#include <math.h>");
        self.emit("");
        self.emit("// String concatenation helper");
        self.emit("static char* str_add(const char* a, const char* b) {");
        self.emit("    size_t la = strlen(a), lb = strlen(b);");
        self.emit("    char* r = malloc(la + lb + 1);");
        self.emit("    memcpy(r, a, la); memcpy(r + la, b, lb); r[la + lb] = 0;");
        self.emit("    return r;");
        self.emit("}");
        self.emit("");

        // Declare built-in functions
        self.emit("// Tenth built-in declarations");
        self.emit("extern void* read_file(const char* path);");
        self.emit("extern void write_file(const char* path, const char* content);");
        self.emit("extern void* Vec_new(void);");
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
            self.emit(&format!("typedef struct {} {{", name));
            for (fname, fty) in fields {
                let c_type = c_type_name(fty, &self.struct_names);
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
        let ret_c = if is_main { "int".to_string() } else { c_type_name(&func.return_type, &self.struct_names) };
        let params: Vec<String> = func.params.iter()
            .map(|(n, t)| format!("{} {}", c_type_name(t, &self.struct_names), n))
            .collect();
        self.emit(&format!("{} {}({});", ret_c, func.name, params.join(", ")));
    }

    fn generate_function(&mut self, func: &MirFunction) {
        let is_main = func.name == "main" && func.params.is_empty();
        let ret_c = if is_main { "int".to_string() } else { c_type_name(&func.return_type, &self.struct_names) };
        self.current_ret_type = ret_c.clone();
        let params: Vec<String> = func.params.iter()
            .map(|(n, t)| format!("{} {}", c_type_name(t, &self.struct_names), n))
            .collect();
        self.emit(&format!("{} {}({}) {{", ret_c, func.name, params.join(", ")));
        self.indent_level = 1;

        // Declare locals
        for local in &func.locals {
            if !func.params.iter().any(|(n, _)| n == &local.name) {
                self.emit(&format!("{} {};", c_type_name(&local.ty, &self.struct_names), local.name));
            }
        }

        if !func.locals.is_empty() {
            self.emit("");
        }

        // Generate blocks
        for block in &func.blocks {
            if block.id != 0 {
                self.emit(&format!("block_{}:", block.id));
            }
            for stmt in &block.stmts {
                self.generate_stmt(stmt);
            }
            self.generate_terminator(&block.terminator);
        }

        self.indent_level = 0;
        self.emit("}");
        self.emit("");
    }

    fn generate_main(&mut self, func: &MirFunction) {
        self.emit("int main(void) {");
        self.indent_level = 1;

        for local in &func.locals {
            self.emit(&format!("{} {};", c_type_name(&local.ty, &self.struct_names), local.name));
        }

        if !func.locals.is_empty() {
            self.emit("");
        }

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
                let effective_ty = if matches!(ty, Type::Unknown) { &value.ty } else { ty };
                let type_str = c_type_name(effective_ty, &self.struct_names);
                let val_str = self.rvalue_to_c(value);
                // DEBUG: emit type info for all Let
                self.emit(&format!("/* Let {} declared={:?} val_ty={:?} */", name, ty, value.ty));
                self.emit(&format!("{} {} = {};", type_str, name, val_str));
            }
            MirStmt::Assign { name, value } => {
                let val_str = self.rvalue_to_c(value);
                self.emit(&format!("{} = {};", name, val_str));
            }
            MirStmt::Expr(rvalue) => {
                self.emit(&format!("{};", self.rvalue_to_c(rvalue)));
            }
            MirStmt::Return(val) => {
                match val {
                    Some(v) => {
                        let val_str = self.rvalue_to_c(v);
                        let ret_ty = &self.current_ret_type;
                        // If return type is a struct and value is zero, emit compound literal
                        if val_str == "0" && ret_ty != "void" && ret_ty != "void*" && ret_ty != "int64_t" && ret_ty != "int32_t" {
                            self.emit(&format!("return ({}){{0}};", ret_ty));
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

    fn rvalue_to_c_cast(&self, rvalue: &MirRvalue, expected_ty: &Type) -> String {
        // For zero literal assigned to struct type, use compound literal
        if let MirRvalueKind::Literal(LiteralValue::Int(0)) = &rvalue.kind {
            let struct_name = match expected_ty {
                Type::Struct(name) => Some(name.clone()),
                Type::TypeParam { name } if self.struct_names.contains(name) => Some(name.clone()),
                _ => None,
            };
            if let Some(name) = struct_name {
                return format!("({}){{0}}", name);
            }
        }
        let val_str = self.rvalue_to_c(rvalue);
        let expected_c = c_type_name(expected_ty, &self.struct_names);
        if expected_c == "void*" || val_str.starts_with('(') {
            val_str
        } else {
            match &rvalue.kind {
                MirRvalueKind::Literal(_) | MirRvalueKind::Call { .. } | MirRvalueKind::StructLiteral { .. } => {
                    format!("({}){}", expected_c, val_str)
                }
                _ => val_str,
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
                let is_str_op = matches!(op, BinOp::Add) && (
                    l.starts_with('"') || r.starts_with('"') ||
                    l.starts_with("str_add") || r.starts_with("str_add") ||
                    matches!(&rvalue.ty, Type::Base(BaseType::Str)) ||
                    matches!(&rvalue.ty, Type::TypeParam { name } if name == "str")
                );
                if is_str_op {
                    format!("str_add({}, {})", l, r)
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
                            "lexer_tokenize" => "(Lexer){0}".to_string(),
                            "cgen_program" => "(Program){0}".to_string(),
                            _ => s,
                        }
                    } else { s }
                }).collect();
                let func_name = match func.as_str() {
                    "Vec::new" => "Vec_new", "HashMap::new" => "HashMap_new",
                    "read_file" => "read_file", "write_file" => "write_file",
                    other => other,
                };
                format!("{}({})", func_name, args_str.join(", "))
            }
            MirRvalueKind::Field { target, field } => {
                format!("({}).{}", self.rvalue_to_c(target), field)
            }
            MirRvalueKind::StructLiteral { name, fields } => {
                let field_strs: Vec<String> = fields.iter()
                    .map(|(fname, fval)| format!(".{} = {}", fname, self.rvalue_to_c(fval)))
                    .collect();
                format!("(({}){{ {} }})", name, field_strs.join(", "))
            }
            _ => "/* unsupported */ 0".to_string(),
        }
    }
}

fn c_type_name(ty: &Type, struct_names: &HashSet<String>) -> String {
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
        Type::Struct(name) => name.clone(),
        Type::Ref(inner) | Type::MutRef(inner) => c_type_name(inner, struct_names),
        Type::TypeParam { name } => {
            if struct_names.contains(name) {
                name.clone()
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

fn collect_struct_names_stmt(stmt: &MirStmt, names: &mut HashSet<String>) {
    match stmt {
        MirStmt::Let { value, .. } => collect_struct_names_rvalue(value, names),
        MirStmt::Assign { value, .. } => collect_struct_names_rvalue(value, names),
        MirStmt::Return(Some(value)) => collect_struct_names_rvalue(value, names),
        MirStmt::Expr(val) => collect_struct_names_rvalue(val, names),
        _ => {}
    }
}

fn collect_struct_names_term(term: &MirTerminator, names: &mut HashSet<String>) {
    match term {
        MirTerminator::Return(Some(val)) => collect_struct_names_rvalue(val, names),
        MirTerminator::If { cond, .. } => collect_struct_names_rvalue(cond, names),
        _ => {}
    }
}

fn collect_struct_names_rvalue(val: &MirRvalue, names: &mut HashSet<String>) {
    match &val.kind {
        MirRvalueKind::StructLiteral { name, fields } => {
            names.insert(name.clone());
            for (_, f) in fields { collect_struct_names_rvalue(f, names); }
        }
        MirRvalueKind::BinaryOp(_, l, r) => {
            collect_struct_names_rvalue(l, names);
            collect_struct_names_rvalue(r, names);
        }
        MirRvalueKind::Field { target, .. } => collect_struct_names_rvalue(target, names),
        MirRvalueKind::UnaryOp(_, e) => collect_struct_names_rvalue(e, names),
        MirRvalueKind::Call { args, .. } => {
            for a in args { collect_struct_names_rvalue(a, names); }
        }
        MirRvalueKind::If { cond, .. } => collect_struct_names_rvalue(cond, names),
        _ => {}
    }
}

fn ftype_str_to_c(t: &str) -> &str {
    match t {
        "I32" | "I64" | "i32" | "i64" => "int64_t",
        "F64" | "f64" => "double",
        "Str" | "str" => "const char*",
        "Bool" | "bool" => "bool",
        _ => "void*",
    }
}

fn escape_c_string(s: &str) -> String {
    s.replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
        .replace('\t', "\\t")
}

impl CGenerator {
    fn emit(&mut self, line: &str) {
        let indent = "    ".repeat(self.indent_level);
        self.output.push_str(&indent);
        self.output.push_str(line);
        self.output.push('\n');
    }
}
