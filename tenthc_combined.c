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

typedef struct {
    void* tokens;
    int64_t pos;
} Parser;
typedef struct {
    void* structs;
    void* fns;
    void* main_stmts;
} Program;
typedef struct {
    const char* kind;
    const char* value;
    void* span;
} Token;
typedef struct {
    const char* name;
    void* params;
    const char* return_type;
    void* body;
} FnDef;
typedef struct {
    const char* name;
    void* fields;
} StructDef;
typedef struct {
    const char* name;
    const char* type_ann;
} Param;
typedef struct {
    int64_t line;
    int64_t col;
} Span;
typedef struct {
    const char* kind;
    const char* name;
    int64_t ival;
    const char* sval;
} Stmt;
typedef struct {
    const char* kind;
    int64_t ival;
    const char* sval;
    const char* op;
    int64_t left;
    int64_t right;
} Expr;
typedef struct {
    const char* source;
    int64_t pos;
    int64_t line;
    int64_t col;
} Lexer;

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
const char* lexer_peek(Lexer lexer);
const char* lexer_advance(Lexer lexer);
Span make_span(int64_t line, int64_t col);
Token make_token(const char* kind, const char* value, int64_t line, int64_t col);
Token lexer_next(Lexer lexer);
void* lexer_tokenize(Lexer lexer);
Parser parser_new(void* tokens);
Token parser_next(Parser p);
Expr parse_expr(Parser p);
Stmt parse_stmt(Parser p);
FnDef parse_fn(Parser p);
StructDef parse_struct(Parser p);
Program parse_program(void* tokens);
const char* cgen_expr(Expr expr);
const char* cgen_stmt(Stmt stmt);
const char* cgen_struct(StructDef s);
const char* cgen_fn(FnDef f);
const char* cgen_program(Program prog);
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

const char* lexer_peek(Lexer lexer) {
    const char* tmp_0;
    
    /* unsupported */ 0;
    return (const char*)0;
}

const char* lexer_advance(Lexer lexer) {
    
    const char* ch = (const char*)lexer_peek(lexer);
    /* unsupported */ 0;
    0;
    /* unsupported */ 0;
    return (const char*)ch;
}

Span make_span(int64_t line, int64_t col) {
    
    return (Span)((Span){ .line = line, .col = col });
}

Token make_token(const char* kind, const char* value, int64_t line, int64_t col) {
    
    Span span = (Span)make_span(line, col);
    return (Token)((Token){ .kind = kind, .value = value, .span = span });
}

Token lexer_next(Lexer lexer) {
    
    const char* ch = (const char*)lexer_peek(lexer);
    int64_t line = (lexer).line;
    int64_t col = (lexer).col;
    /* unsupported */ 0;
    /* unsupported */ 0;
    /* unsupported */ 0;
    /* unsupported */ 0;
    lexer_advance(lexer);
    /* unsupported */ 0;
    /* unsupported */ 0;
    /* unsupported */ 0;
    /* unsupported */ 0;
    /* unsupported */ 0;
    /* unsupported */ 0;
    /* unsupported */ 0;
    /* unsupported */ 0;
    /* unsupported */ 0;
    /* unsupported */ 0;
    /* unsupported */ 0;
    /* unsupported */ 0;
    return (Token)make_token(KIND_IDENT(), ch, line, col);
}

void* lexer_tokenize(Lexer lexer) {
    
    int64_t tokens = (int64_t)Vec_new();
    bool done = (bool)false;
    return (void*)tokens;
}

Parser parser_new(void* tokens) {
    
    return (Parser)((Parser){ .tokens = tokens, .pos = 0 });
}

Token parser_next(Parser p) {
    Token tmp_0;
    
    /* unsupported */ 0;
    return (Token)0;
}

Expr parse_expr(Parser p) {
    int64_t tmp_0;
    
    Token t = (Token)parser_next(p);
    /* unsupported */ 0;
    return (Expr)0;
}

Stmt parse_stmt(Parser p) {
    int64_t tmp_0;
    
    Token t = (Token)parser_next(p);
    /* unsupported */ 0;
    return (Stmt)0;
}

FnDef parse_fn(Parser p) {
    
    int64_t name = (parser_next(p)).value;
    parser_next(p);
    int64_t params = (int64_t)Vec_new();
    Token tok = (Token)parser_next(p);
    tok = parser_next(p);
    /* unsupported */ 0;
    int64_t body = (int64_t)Vec_new();
    tok = parser_next(p);
    return (FnDef)((FnDef){ .name = name, .params = params, .return_type = "", .body = body });
}

StructDef parse_struct(Parser p) {
    
    int64_t name = (parser_next(p)).value;
    parser_next(p);
    int64_t fields = (int64_t)Vec_new();
    Token tok = (Token)parser_next(p);
    return (StructDef)((StructDef){ .name = name, .fields = fields });
}

Program parse_program(void* tokens) {
    
    Parser p = (Parser)parser_new(tokens);
    int64_t structs = (int64_t)Vec_new();
    int64_t fns = (int64_t)Vec_new();
    int64_t main_stmts = (int64_t)Vec_new();
    return (Program)((Program){ .structs = structs, .fns = fns, .main_stmts = main_stmts });
}

const char* cgen_expr(Expr expr) {
    const char* tmp_0;
    
    /* unsupported */ 0;
    return (const char*)0;
}

const char* cgen_stmt(Stmt stmt) {
    const char* tmp_0;
    
    /* unsupported */ 0;
    return (const char*)0;
}

const char* cgen_struct(StructDef s) {
    
    const char* out = (const char*)"typedef struct {\n";
    int32_t i = (int32_t)0;
    out = str_add((str_add(out, "} ") + (s).name), ";\n\n");
    return (const char*)out;
}

const char* cgen_fn(FnDef f) {
    
    const char* out = str_add(str_add("int ", (f).name), "(");
    int32_t i = (int32_t)0;
    out = str_add(out, ") {\n");
    i = 0;
    out = str_add(out, "}\n\n");
    return (const char*)out;
}

const char* cgen_program(Program prog) {
    
    const char* out = (const char*)"#include <stdio.h>\n\n";
    int32_t i = (int32_t)0;
    i = 0;
    out = str_add(out, "int main(void) {\n");
    i = 0;
    out = str_add(out, "    return 0;\n}\n");
    return (const char*)out;
}

const char* compile(const char* src) {
    
    Lexer lex = (Lexer)lexer_new(src);
    void* tokens = (void*)lexer_tokenize(0);
    Program prog = (Program)parse_program(tokens);
    return (const char*)cgen_program(0);
}

int main() {
    const char* src = (const char*)read_file("tenthc_combined.th");
    const char* c = (const char*)compile(src);
    write_file("tenthc_output.c", c);
    return (int)0;
}

