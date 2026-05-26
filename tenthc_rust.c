#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// String concatenation helper
static char* str_add(const char* a, const char* b) {
    size_t la = strlen(a), lb = strlen(b);
    char* r = malloc(la + lb + 1);
    memcpy(r, a, la); memcpy(r + la, b, lb); r[la + lb] = 0;
    return r;
}

// Tenth built-in declarations
extern void* read_file(const char* path);
extern void write_file(const char* path, const char* content);
extern void* Vec_new(void);
extern void* HashMap_new(void);

typedef struct Stmt {
    const char* kind;
    const char* name;
    int64_t ival;
    const char* sval;
} Stmt;
typedef struct Lexer {
    const char* source;
    int64_t pos;
    int64_t line;
    int64_t col;
} Lexer;
typedef struct Expr {
    const char* kind;
    int64_t ival;
    const char* sval;
    const char* op;
    int64_t left;
    int64_t right;
} Expr;
typedef struct Param {
    const char* name;
    const char* type_ann;
} Param;
typedef struct Span {
    int64_t line;
    int64_t col;
} Span;
typedef struct Parser {
    void* tokens;
    int64_t pos;
} Parser;
typedef struct StructDef {
    const char* name;
    void* fields;
} StructDef;
typedef struct Program {
    void* structs;
    void* fns;
    void* main_stmts;
} Program;
typedef struct FnDef {
    const char* name;
    void* params;
    const char* return_type;
    void* body;
} FnDef;
typedef struct Token {
    const char* kind;
    const char* value;
    Span span;
} Token;

const char* KIND_EOF();
const char* KIND_INT();
const char* KIND_IDENT();
const char* KIND_FN();
const char* KIND_LET();
const char* KIND_IF();
const char* KIND_ELSE();
const char* KIND_RETURN();
const char* KIND_STRUCT();
const char* KIND_PLUS();
const char* KIND_MINUS();
const char* KIND_STAR();
const char* KIND_SLASH();
const char* KIND_ASSIGN();
const char* KIND_EQEQ();
const char* KIND_LPAREN();
const char* KIND_RPAREN();
const char* KIND_LBRACE();
const char* KIND_RBRACE();
const char* KIND_COLON();
const char* KIND_COMMA();
const char* KIND_SEMICOLON();
const char* KIND_ARROW();
const char* KIND_COLON2();
Lexer lexer_new(const char* source);
bool is_digit(const char* ch);
bool is_alpha(const char* ch);
bool is_alnum(const char* ch);
bool is_ws(const char* ch);
const char* lexer_peek(Lexer* lexer);
const char* lexer_advance(Lexer* lexer);
Span make_span(int64_t line, int64_t col);
Token make_token(const char* kind, const char* value, int64_t line, int64_t col);
Token lexer_next(Lexer* lexer);
void* lexer_tokenize(Lexer* lexer);
Parser parser_new(void* tokens);
Token parser_next(Parser* p);
Expr parse_expr(Parser* p);
Stmt parse_stmt(Parser* p);
FnDef parse_fn(Parser* p);
StructDef parse_struct(Parser* p);
Program parse_program(void* tokens);
const char* cgen_expr(Expr* expr);
const char* cgen_stmt(Stmt* stmt);
const char* cgen_struct(StructDef* s);
const char* cgen_fn(FnDef* f);
const char* cgen_program(Program* prog);
const char* compile(const char* src);
int main();
const char* KIND_EOF() {
    return (const char*)"Eof";
}

const char* KIND_INT() {
    return (const char*)"Int";
}

const char* KIND_IDENT() {
    return (const char*)"Ident";
}

const char* KIND_FN() {
    return (const char*)"Fn";
}

const char* KIND_LET() {
    return (const char*)"Let";
}

const char* KIND_IF() {
    return (const char*)"If";
}

const char* KIND_ELSE() {
    return (const char*)"Else";
}

const char* KIND_RETURN() {
    return (const char*)"Return";
}

const char* KIND_STRUCT() {
    return (const char*)"Struct";
}

const char* KIND_PLUS() {
    return (const char*)"Plus";
}

const char* KIND_MINUS() {
    return (const char*)"Minus";
}

const char* KIND_STAR() {
    return (const char*)"Star";
}

const char* KIND_SLASH() {
    return (const char*)"Slash";
}

const char* KIND_ASSIGN() {
    return (const char*)"Assign";
}

const char* KIND_EQEQ() {
    return (const char*)"EqEq";
}

const char* KIND_LPAREN() {
    return (const char*)"LParen";
}

const char* KIND_RPAREN() {
    return (const char*)"RParen";
}

const char* KIND_LBRACE() {
    return (const char*)"LBrace";
}

const char* KIND_RBRACE() {
    return (const char*)"RBrace";
}

const char* KIND_COLON() {
    return (const char*)"Colon";
}

const char* KIND_COMMA() {
    return (const char*)"Comma";
}

const char* KIND_SEMICOLON() {
    return (const char*)"Semicolon";
}

const char* KIND_ARROW() {
    return (const char*)"Arrow";
}

const char* KIND_COLON2() {
    return (const char*)"Colon2";
}

Lexer lexer_new(const char* source) {
    return (Lexer)((Lexer){ .source = source, .pos = 0, .line = 1, .col = 1 });
}

bool is_digit(const char* ch) {
    return (bool)((ch >= "0") && (ch <= "9"));
}

bool is_alpha(const char* ch) {
    return (bool)((((ch >= "a") && (ch <= "z")) || ((ch >= "A") && (ch <= "Z"))) || (ch == "_"));
}

bool is_alnum(const char* ch) {
    return (bool)(is_alpha(ch) || is_digit(ch));
}

bool is_ws(const char* ch) {
    return (bool)((((ch == " ") || (ch == "\n")) || (ch == "\t")) || (ch == "r"));
}

const char* lexer_peek(Lexer* lexer) {
    /* Let tmp_0 declared=Base(Str) val_ty=Base(Str) */
    const char* tmp_0 = ((lexer->pos < 0) ? 0 : "");
    return (const char*)tmp_0;
}

const char* lexer_advance(Lexer* lexer) {
    /* Let ch declared=Base(Str) val_ty=Base(Str) */
    const char* ch = lexer_peek(lexer);
    0;
    if ((ch == "\n")) {
    0;
    }
    0;
    return (const char*)ch;
}

Span make_span(int64_t line, int64_t col) {
    return (Span)((Span){ .line = line, .col = col });
}

Token make_token(const char* kind, const char* value, int64_t line, int64_t col) {
    /* Let span declared=TypeParam { name: "Span" } val_ty=TypeParam { name: "Span" } */
    Span span = make_span(line, col);
    return (Token)((Token){ .kind = kind, .value = value, .span = span });
}

Token lexer_next(Lexer* lexer) {
    /* Let ch declared=Base(Str) val_ty=Base(Str) */
    const char* ch = lexer_peek(lexer);
    /* Let line declared=Base(I64) val_ty=Base(I64) */
    int64_t line = lexer->line;
    /* Let col declared=Base(I64) val_ty=Base(I64) */
    int64_t col = lexer->col;
    if (is_digit(ch)) {
    /* Let num_str declared=Base(Str) val_ty=Base(Str) */
    const char* num_str = "";
    }
    if (is_alpha(ch)) {
    /* Let ident declared=Base(Str) val_ty=Base(Str) */
    const char* ident = "";
    }
    if ((ch == "\"")) {
    lexer_advance(lexer);
    /* Let s declared=Base(Str) val_ty=Base(Str) */
    const char* s = "";
    ch = lexer_peek(lexer);
    lexer_advance(lexer);
    }
    lexer_advance(lexer);
    if ((ch == "-")) {
    if ((lexer_peek(lexer) == ">")) {
    lexer_advance(lexer);
    }
    }
    if ((ch == ":")) {
    if ((lexer_peek(lexer) == ":")) {
    lexer_advance(lexer);
    }
    }
    if ((ch == "=")) {
    if ((lexer_peek(lexer) == "=")) {
    lexer_advance(lexer);
    }
    }
    return (Token)make_token(KIND_IDENT(), ch, line, col);
}

void* lexer_tokenize(Lexer* lexer) {
    /* Let tokens declared=Unknown val_ty=Unknown */
    void* tokens = Vec_new();
    /* Let done declared=Base(Bool) val_ty=Base(Bool) */
    bool done = false;
    return (void*)tokens;
}

Parser parser_new(void* tokens) {
    return (Parser)((Parser){ .tokens = tokens, .pos = 0 });
}

Token parser_next(Parser* p) {
    if ((p->pos < 0)) {
    /* Let t declared=TypeParam { name: "Vec" } val_ty=TypeParam { name: "Vec" } */
    void* t = 0;
    0;
    }
    /* Let tmp_0 declared=TypeParam { name: "Token" } val_ty=TypeParam { name: "Vec" } */
    Token tmp_0 = t;
    return (Token)tmp_0;
}

Expr parse_expr(Parser* p) {
    /* Let t declared=TypeParam { name: "Token" } val_ty=TypeParam { name: "Token" } */
    Token t = parser_next(p);
    if (((t).kind == KIND_INT())) {
    } else {
    if (((t).kind == KIND_IDENT())) {
    /* Let name declared=Base(Str) val_ty=Base(Str) */
    const char* name = (t).value;
    }
    }
    /* Let tmp_0 declared=Unknown val_ty=Unknown */
    void* tmp_0 = ((Expr){ .kind = "int", .ival = 0, .sval = (t).value, .op = "", .left = 0, .right = 0 });
    return (Expr)tmp_0;
}

Stmt parse_stmt(Parser* p) {
    /* Let t declared=TypeParam { name: "Token" } val_ty=TypeParam { name: "Token" } */
    Token t = parser_next(p);
    if (((t).kind == KIND_LET())) {
    /* Let name declared=Base(Str) val_ty=Base(Str) */
    const char* name = (parser_next(p)).value;
    parser_next(p);
    /* Let val declared=TypeParam { name: "Expr" } val_ty=TypeParam { name: "Expr" } */
    Expr val = parse_expr(p);
    parser_next(p);
    } else {
    if (((t).kind == KIND_RETURN())) {
    /* Let val declared=TypeParam { name: "Expr" } val_ty=TypeParam { name: "Expr" } */
    Expr val = parse_expr(p);
    parser_next(p);
    } else {
    parser_next(p);
    }
    }
    /* Let tmp_0 declared=Unknown val_ty=Unknown */
    void* tmp_0 = ((Stmt){ .kind = "let", .name = name, .ival = (val).ival, .sval = (val).sval });
    return (Stmt)tmp_0;
}

FnDef parse_fn(Parser* p) {
    /* Let name declared=Base(Str) val_ty=Base(Str) */
    const char* name = (parser_next(p)).value;
    parser_next(p);
    /* Let params declared=Unknown val_ty=Unknown */
    void* params = Vec_new();
    /* Let tok declared=TypeParam { name: "Token" } val_ty=TypeParam { name: "Token" } */
    Token tok = parser_next(p);
    tok = parser_next(p);
    if (((tok).kind == KIND_ARROW())) {
    parser_next(p);
    tok = parser_next(p);
    }
    0;
    /* Let body declared=Unknown val_ty=Unknown */
    void* body = Vec_new();
    tok = parser_next(p);
    return (FnDef)((FnDef){ .name = name, .params = params, .return_type = "", .body = body });
}

StructDef parse_struct(Parser* p) {
    /* Let name declared=Base(Str) val_ty=Base(Str) */
    const char* name = (parser_next(p)).value;
    parser_next(p);
    /* Let fields declared=Unknown val_ty=Unknown */
    void* fields = Vec_new();
    /* Let tok declared=TypeParam { name: "Token" } val_ty=TypeParam { name: "Token" } */
    Token tok = parser_next(p);
    return (StructDef)((StructDef){ .name = name, .fields = fields });
}

Program parse_program(void* tokens) {
    /* Let p declared=TypeParam { name: "Parser" } val_ty=TypeParam { name: "Parser" } */
    Parser p = parser_new(tokens);
    /* Let structs declared=Unknown val_ty=Unknown */
    void* structs = Vec_new();
    /* Let fns declared=Unknown val_ty=Unknown */
    void* fns = Vec_new();
    /* Let main_stmts declared=Unknown val_ty=Unknown */
    void* main_stmts = Vec_new();
    return (Program)((Program){ .structs = structs, .fns = fns, .main_stmts = main_stmts });
}

const char* cgen_expr(Expr* expr) {
    /* Let tmp_0 declared=Base(Str) val_ty=Base(Str) */
    const char* tmp_0 = ((expr->kind == "int") ? expr->sval : ((expr->kind == "ident") ? expr->sval : ((expr->kind == "binary") ? str_add(str_add(str_add(str_add(str_add(str_add(str_add("(", expr->sval), " "), expr->op), " "), ""), expr->right), ")") : "0")));
    return (const char*)tmp_0;
}

const char* cgen_stmt(Stmt* stmt) {
    /* Let tmp_0 declared=Base(Str) val_ty=Base(Str) */
    const char* tmp_0 = ((stmt->kind == "let") ? str_add(str_add(str_add(str_add("int ", stmt->name), " = "), cgen_expr(0)), ";\n") : ((stmt->kind == "return") ? str_add(str_add("return ", cgen_expr(0)), ";\n") : ";\n"));
    return (const char*)tmp_0;
}

const char* cgen_struct(StructDef* s) {
    /* Let out declared=Base(Str) val_ty=Base(Str) */
    const char* out = "typedef struct {\n";
    /* Let i declared=Base(I32) val_ty=Base(I32) */
    int32_t i = 0;
    out = str_add(str_add(str_add(out, "} "), s->name), ";\n\n");
    return (const char*)out;
}

const char* cgen_fn(FnDef* f) {
    /* Let out declared=Base(Str) val_ty=Base(Str) */
    const char* out = str_add(str_add("int ", f->name), "(");
    /* Let i declared=Base(I32) val_ty=Base(I32) */
    int32_t i = 0;
    out = str_add(out, ") {\n");
    i = 0;
    out = str_add(out, "}\n\n");
    return (const char*)out;
}

const char* cgen_program(Program* prog) {
    /* Let out declared=Base(Str) val_ty=Base(Str) */
    const char* out = "#include <stdio.h>\n\n";
    /* Let i declared=Base(I32) val_ty=Base(I32) */
    int32_t i = 0;
    i = 0;
    out = str_add(out, "int main(void) {\n");
    i = 0;
    out = str_add(out, "    return 0;\n}\n");
    return (const char*)out;
}

const char* compile(const char* src) {
    /* Let lex declared=TypeParam { name: "Lexer" } val_ty=TypeParam { name: "Lexer" } */
    Lexer lex = lexer_new(src);
    /* Let tokens declared=TypeParam { name: "Vec" } val_ty=TypeParam { name: "Vec" } */
    void* tokens = lexer_tokenize(&lex);
    /* Let prog declared=TypeParam { name: "Program" } val_ty=TypeParam { name: "Program" } */
    Program prog = parse_program(tokens);
    return (const char*)cgen_program(&prog);
}

int main() {
    /* Let src declared=Base(Str) val_ty=Base(Str) */
    const char* src = read_file("tenthc_combined.th");
    /* Let c declared=Base(Str) val_ty=Base(Str) */
    const char* c = compile(src);
    write_file("tenthc_output.c", c);
    return (int){0};
}

