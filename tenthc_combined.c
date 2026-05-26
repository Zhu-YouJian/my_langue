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

typedef struct Lexer { int _dummy; } Lexer;
typedef struct StructDef { int _dummy; } StructDef;
typedef struct Token { int _dummy; } Token;
typedef struct FnDef { int _dummy; } FnDef;
typedef struct Program { int _dummy; } Program;
typedef struct Span { int _dummy; } Span;
typedef struct Parser { int _dummy; } Parser;

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
void* lexer_new(const char* source);
bool is_digit(const char* ch);
bool is_alpha(const char* ch);
bool is_alnum(const char* ch);
bool is_ws(const char* ch);
const char* lexer_peek(void* lexer);
const char* lexer_advance(void* lexer);
void* make_span(int64_t line, int64_t col);
void* make_token(const char* kind, const char* value, int64_t line, int64_t col);
void* lexer_next(void* lexer);
void* lexer_tokenize(void* lexer);
void* parser_new(void* tokens);
void* parser_next(void* p);
void* parse_expr(void* p);
void* parse_stmt(void* p);
void* parse_fn(void* p);
void* parse_struct(void* p);
void* parse_program(void* tokens);
const char* cgen_expr(void* expr);
const char* cgen_stmt(void* stmt);
const char* cgen_struct(void* s);
const char* cgen_fn(void* f);
const char* cgen_program(void* prog);
const char* compile(const char* src);
int main();
const char* KIND_EOF() {
    return "Eof";
}

const char* KIND_INT() {
    return "Int";
}

const char* KIND_IDENT() {
    return "Ident";
}

const char* KIND_FN() {
    return "Fn";
}

const char* KIND_LET() {
    return "Let";
}

const char* KIND_IF() {
    return "If";
}

const char* KIND_ELSE() {
    return "Else";
}

const char* KIND_RETURN() {
    return "Return";
}

const char* KIND_STRUCT() {
    return "Struct";
}

const char* KIND_PLUS() {
    return "Plus";
}

const char* KIND_MINUS() {
    return "Minus";
}

const char* KIND_STAR() {
    return "Star";
}

const char* KIND_SLASH() {
    return "Slash";
}

const char* KIND_ASSIGN() {
    return "Assign";
}

const char* KIND_EQEQ() {
    return "EqEq";
}

const char* KIND_LPAREN() {
    return "LParen";
}

const char* KIND_RPAREN() {
    return "RParen";
}

const char* KIND_LBRACE() {
    return "LBrace";
}

const char* KIND_RBRACE() {
    return "RBrace";
}

const char* KIND_COLON() {
    return "Colon";
}

const char* KIND_COMMA() {
    return "Comma";
}

const char* KIND_SEMICOLON() {
    return "Semicolon";
}

const char* KIND_ARROW() {
    return "Arrow";
}

const char* KIND_COLON2() {
    return "Colon2";
}

void* lexer_new(const char* source) {
    
    return ((void*)0 /* Lexer literal */);
}

bool is_digit(const char* ch) {
    
    return ((ch >= "0") && (ch <= "9"));
}

bool is_alpha(const char* ch) {
    
    return ((((ch >= "a") && (ch <= "z")) || ((ch >= "A") && (ch <= "Z"))) || (ch == "_"));
}

bool is_alnum(const char* ch) {
    
    return (is_alpha(ch) || is_digit(ch));
}

bool is_ws(const char* ch) {
    
    return ((((ch == " ") || (ch == "\n")) || (ch == "\t")) || (ch == "r"));
}

const char* lexer_peek(void* lexer) {
    void* tmp_0;
    
    /* unsupported */ 0;
    return 0;
}

const char* lexer_advance(void* lexer) {
    
    void* ch = lexer_peek(lexer);
    /* unsupported */ 0;
    /* unsupported */ 0;
    return ch;
}

void* make_span(int64_t line, int64_t col) {
    
    return ((void*)0 /* Span literal */);
}

void* make_token(const char* kind, const char* value, int64_t line, int64_t col) {
    
    void* span = make_span(line, col);
    return ((void*)0 /* Token literal */);
}

void* lexer_next(void* lexer) {
    
    void* ch = lexer_peek(lexer);
    void* line = 0;
    void* col = 0;
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
    /* unsupported */ 0;
    /* unsupported */ 0;
    /* unsupported */ 0;
    /* unsupported */ 0;
    return make_token(KIND_IDENT(), ch, line, col);
}

void* lexer_tokenize(void* lexer) {
    
    void* tokens = Vec_new();
    void* done = false;
    return tokens;
}

void* parser_new(void* tokens) {
    
    return ((void*)0 /* Parser literal */);
}

void* parser_next(void* p) {
    void* tmp_0;
    
    /* unsupported */ 0;
    return 0;
}

void* parse_expr(void* p) {
    void* tmp_0;
    
    void* t = parser_next(p);
    /* unsupported */ 0;
    return 0;
}

void* parse_stmt(void* p) {
    void* tmp_0;
    
    void* t = parser_next(p);
    /* unsupported */ 0;
    return 0;
}

void* parse_fn(void* p) {
    
    void* name = 0;
    void* params = Vec_new();
    void* tok = parser_next(p);
    tok = parser_next(p);
    /* unsupported */ 0;
    void* body = Vec_new();
    tok = parser_next(p);
    return ((void*)0 /* FnDef literal */);
}

void* parse_struct(void* p) {
    
    void* name = 0;
    void* fields = Vec_new();
    void* tok = parser_next(p);
    return ((void*)0 /* StructDef literal */);
}

void* parse_program(void* tokens) {
    
    void* p = parser_new(tokens);
    void* structs = Vec_new();
    void* fns = Vec_new();
    void* main_stmts = Vec_new();
    return ((void*)0 /* Program literal */);
}

const char* cgen_expr(void* expr) {
    void* tmp_0;
    
    /* unsupported */ 0;
    return 0;
}

const char* cgen_stmt(void* stmt) {
    void* tmp_0;
    
    /* unsupported */ 0;
    return 0;
}

const char* cgen_struct(void* s) {
    
    void* out = "typedef struct {\n";
    void* i = 0;
    out = str_add((str_add(out, "} ") + 0), ";\n\n");
    return out;
}

const char* cgen_fn(void* f) {
    
    void* out = str_add(str_add("int ", 0), "(");
    void* i = 0;
    out = str_add(out, ") {\n");
    i = 0;
    out = str_add(out, "}\n\n");
    return out;
}

const char* cgen_program(void* prog) {
    
    void* out = "#include <stdio.h>\n\n";
    void* i = 0;
    i = 0;
    out = str_add(out, "int main(void) {\n");
    i = 0;
    out = str_add(out, "    return 0;\n}\n");
    return out;
}

const char* compile(const char* src) {
    
    void* lex = lexer_new(src);
    void* tokens = lexer_tokenize(0);
    void* prog = parse_program(tokens);
    return cgen_program(0);
}

int main() {
    void* src = read_file("tenthc/lexer/token.th");
    void* c = compile(src);
    return 42;
}

