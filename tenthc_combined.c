#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>
#include <math.h>

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
    
    return { source, 0, 1, 1 };
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
    int32_t tmp_0;
    
    /* unsupported */ 0;
    return 0;
}

const char* lexer_advance(void* lexer) {
    
    int32_t ch = lexer_peek(lexer);
    /* unsupported */ 0;
    /* unsupported */ 0;
    return ch;
}

void* make_span(int64_t line, int64_t col) {
    
    return { line, col };
}

void* make_token(const char* kind, const char* value, int64_t line, int64_t col) {
    
    int32_t span = make_span(line, col);
    return { kind, value, span };
}

void* lexer_next(void* lexer) {
    
    int32_t ch = lexer_peek(lexer);
    int32_t line = 0;
    int32_t col = 0;
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
    
    int32_t tokens = Vec::new();
    int32_t done = false;
    return tokens;
}

void* parser_new(void* tokens) {
    
    return { tokens, 0 };
}

void* parser_next(void* p) {
    int32_t tmp_0;
    
    /* unsupported */ 0;
    return 0;
}

void* parse_expr(void* p) {
    int32_t tmp_0;
    
    int32_t t = parser_next(p);
    /* unsupported */ 0;
    return 0;
}

void* parse_stmt(void* p) {
    int32_t tmp_0;
    
    int32_t t = parser_next(p);
    /* unsupported */ 0;
    return 0;
}

void* parse_fn(void* p) {
    
    int32_t name = 0;
    int32_t params = Vec::new();
    int32_t tok = parser_next(p);
    tok = parser_next(p);
    /* unsupported */ 0;
    int32_t body = Vec::new();
    tok = parser_next(p);
    return { name, params, "", body };
}

void* parse_struct(void* p) {
    
    int32_t name = 0;
    int32_t fields = Vec::new();
    int32_t tok = parser_next(p);
    return { name, fields };
}

void* parse_program(void* tokens) {
    
    int32_t p = parser_new(tokens);
    int32_t structs = Vec::new();
    int32_t fns = Vec::new();
    int32_t main_stmts = Vec::new();
    return { structs, fns, main_stmts };
}

const char* cgen_expr(void* expr) {
    int32_t tmp_0;
    
    /* unsupported */ 0;
    return 0;
}

const char* cgen_stmt(void* stmt) {
    int32_t tmp_0;
    
    /* unsupported */ 0;
    return 0;
}

const char* cgen_struct(void* s) {
    
    int32_t out = "typedef struct {\n";
    int32_t i = 0;
    out = (((out + "} ") + 0) + ";\n\n");
    return out;
}

const char* cgen_fn(void* f) {
    
    int32_t out = (("int " + 0) + "(");
    int32_t i = 0;
    out = (out + ") {\n");
    i = 0;
    out = (out + "}\n\n");
    return out;
}

const char* cgen_program(void* prog) {
    
    int32_t out = "#include <stdio.h>\n\n";
    int32_t i = 0;
    i = 0;
    out = (out + "int main(void) {\n");
    i = 0;
    out = (out + "    return 0;\n}\n");
    return out;
}

const char* compile(const char* src) {
    
    int32_t lex = lexer_new(src);
    int32_t tokens = lexer_tokenize(0);
    int32_t prog = parse_program(tokens);
    return cgen_program(0);
}

int main() {
    int32_t src = read_file("tenthc/lexer/token.th");
    int32_t c = compile(src);
    return 42;
}

