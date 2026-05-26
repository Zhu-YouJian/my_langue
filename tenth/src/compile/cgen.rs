use std::collections::{HashMap, HashSet};
use crate::hir::types::Type;
use crate::hir::hir::{BinOp, UnaryOp};
use super::mir::*;

pub struct CGenerator {
    output: String,
    indent_level: usize,
    var_types: HashMap<String, String>,
}

impl CGenerator {
    pub fn new() -> Self {
        CGenerator {
            output: String::new(),
            indent_level: 0,
            var_types: HashMap::new(),
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

        // Collect struct names from the program
        let mut struct_names = HashSet::new();
        for func in &program.functions {
            for block in &func.blocks {
                for stmt in &block.stmts {
                    collect_struct_names_stmt(stmt, &mut struct_names);
                }
                collect_struct_names_term(&block.terminator, &mut struct_names);
            }
        }
        if let Some(ref main_fn) = program.main_expr {
            for block in &main_fn.blocks {
                for stmt in &block.stmts {
                    collect_struct_names_stmt(stmt, &mut struct_names);
                }
                collect_struct_names_term(&block.terminator, &mut struct_names);
            }
        }

        // Emit forward struct declarations
        for name in &struct_names {
            self.emit(&format!("typedef struct {} {{ int _dummy; }} {};", name, name));
        }
        if !struct_names.is_empty() {
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
        let ret_c = if is_main { "int" } else { c_type_name(&func.return_type) };
        let params: Vec<String> = func.params.iter()
            .map(|(n, t)| format!("{} {}", c_type_name(t), n))
            .collect();
        self.emit(&format!("{} {}({});", ret_c, func.name, params.join(", ")));
    }

    fn generate_function(&mut self, func: &MirFunction) {
        let is_main = func.name == "main" && func.params.is_empty();
        let ret_c = if is_main { "int" } else { c_type_name(&func.return_type) };
        let params: Vec<String> = func.params.iter()
            .map(|(n, t)| format!("{} {}", c_type_name(t), n))
            .collect();
        self.emit(&format!("{} {}({}) {{", ret_c, func.name, params.join(", ")));
        self.indent_level = 1;

        // Declare locals
        for local in &func.locals {
            if !func.params.iter().any(|(n, _)| n == &local.name) {
                self.emit(&format!("{} {};", c_type_name(&local.ty), local.name));
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
            self.emit(&format!("{} {};", c_type_name(&local.ty), local.name));
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
                let type_str = c_type_name(ty);
                let val_str = self.rvalue_to_c(value);
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
                    Some(v) => self.emit(&format!("return {};", self.rvalue_to_c(v))),
                    None => self.emit("return;"),
                }
            }
        }
    }

    fn generate_terminator(&mut self, term: &MirTerminator) {
        match term {
            MirTerminator::Return(val) => {
                match val {
                    Some(v) => self.emit(&format!("return {};", self.rvalue_to_c(v))),
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
        match rvalue {
            MirRvalue::Literal(lit) => match lit {
                LiteralValue::Int(n) => n.to_string(),
                LiteralValue::Float(n) => format!("{:.10}", n),
                LiteralValue::Bool(true) => "true".to_string(),
                LiteralValue::Bool(false) => "false".to_string(),
                LiteralValue::Str(s) => format!("\"{}\"", escape_c_string(s)),
            },
            MirRvalue::Use(name) => name.clone(),
            MirRvalue::BinaryOp(op, left, right) => {
                let l = self.rvalue_to_c(left);
                let r = self.rvalue_to_c(right);
                // Detect string concatenation: if operands are string literals, use str_add
                if matches!(op, BinOp::Add) && (l.starts_with('"') || r.starts_with('"')) {
                    format!("str_add({}, {})", l, r)
                } else {
                    let op_str = c_binop(op);
                    format!("({} {} {})", l, op_str, r)
                }
            }
            MirRvalue::UnaryOp(op, expr) => {
                let e = self.rvalue_to_c(expr);
                match op {
                    UnaryOp::Neg => format!("(-{})", e),
                    UnaryOp::Not => format!("(!{})", e),
                }
            }
            MirRvalue::Call { func, args } => {
                let args_str: Vec<String> = args.iter()
                    .map(|a| self.rvalue_to_c(a))
                    .collect();
                let func_name = match func.as_str() {
                    "Vec::new" => "Vec_new",
                    "HashMap::new" => "HashMap_new",
                    "read_file" => "read_file",
                    "write_file" => "write_file",
                    other => other,
                };
                format!("{}({})", func_name, args_str.join(", "))
            }
            MirRvalue::StructLiteral { name, .. } => {
                // Simplified: return NULL pointer (struct layout not tracked in MIR)
                // Full implementation would emit compound literals with correct types
                format!("((void*)0 /* {} literal */)", name)
            }
            _ => "/* unsupported */ 0".to_string(),
        }
    }
}

fn c_type_name(ty: &Type) -> &str {
    use crate::hir::types::BaseType;
    match ty {
        Type::Base(b) => match b {
            BaseType::I8 | BaseType::I16 | BaseType::I32 => "int32_t",
            BaseType::I64 => "int64_t",
            BaseType::U8 | BaseType::U16 | BaseType::U32 => "uint32_t",
            BaseType::U64 => "uint64_t",
            BaseType::F16 | BaseType::F32 => "float",
            BaseType::F64 | BaseType::BF16 => "double",
            BaseType::Bool => "bool",
            BaseType::Char => "char",
            BaseType::Str => "const char*",
            BaseType::Unit => "void",
        },
        Type::Struct(_) => "void*",
        Type::Unknown => "void*",
        _ => "void*",
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
    match val {
        MirRvalue::StructLiteral { name, fields } => {
            names.insert(name.clone());
            for (_, f) in fields {
                collect_struct_names_rvalue(f, names);
            }
        }
        MirRvalue::BinaryOp(_, l, r) => {
            collect_struct_names_rvalue(l, names);
            collect_struct_names_rvalue(r, names);
        }
        MirRvalue::UnaryOp(_, e) => collect_struct_names_rvalue(e, names),
        MirRvalue::Call { args, .. } => {
            for a in args { collect_struct_names_rvalue(a, names); }
        }
        MirRvalue::If { cond, .. } => collect_struct_names_rvalue(cond, names),
        _ => {}
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
