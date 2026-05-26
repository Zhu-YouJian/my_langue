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
// Int-to-string helper
static char* str_int(int64_t n) {
    char buf[32]; snprintf(buf, 32, "%lld", (long long)n);
    return strdup(buf);
}

// Tenth built-in declarations
extern void* read_file(const char* path);
extern void write_file(const char* path, const char* content);
extern void* Vec_new(void);
extern void* Vec_push(void* v, void* item);
extern int64_t Vec_len(void* v);
extern void* Vec_get(void* v, int64_t idx);
extern const char* str_at(const char* s, int64_t pos);
extern void* HashMap_new(void);

typedef struct Span {
    int64_t line;
    int64_t col;
} Span;
typedef struct Expr {
    const char* kind;
    int64_t ival;
    const char* sval;
    const char* op;
    int64_t left;
    int64_t right;
    const char* right_sval;
} Expr;
typedef struct Lexer {
    const char* source;
    int64_t pos;
    int64_t line;
    int64_t col;
} Lexer;
typedef struct Stmt {
    const char* kind;
    const char* name;
    int64_t ival;
    const char* sval;
    const char* op;
    int64_t left;
    int64_t right;
    const char* right_sval;
} Stmt;
typedef struct Param {
    const char* name;
    const char* type_ann;
} Param;
typedef struct Token {
    const char* kind;
    const char* value;
    Span span;
} Token;
typedef struct Parser {
    void* tokens;
    int64_t pos;
} Parser;
typedef struct FnDef {
    const char* name;
    void* params;
    const char* return_type;
    void* body;
} FnDef;
typedef struct StructDef {
    const char* name;
    void* fields;
} StructDef;
typedef struct Program {
    void* structs;
    void* fns;
    void* main_stmts;
} Program;

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
const char* KIND_DOT();
const char* KIND_LBRACKET();
const char* KIND_RBRACKET();
const char* KIND_ANDAND();
const char* KIND_LT();
const char* KIND_GT();
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
Expr make_int_expr(const char* v);
Expr make_ident_expr(const char* n);
Expr make_binary_expr(Expr l, const char* o, Expr r);
Expr make_assign_expr(const char* target, Expr value);
Expr make_call_expr(const char* func, const char* args);
Token cur_tok(Parser* p);
Expr parse_primary(Parser* p);
Expr parse_postfix(Parser* p);
Expr parse_binary(Parser* p, int64_t min_prec);
Expr parse_expr(Parser* p);
Stmt parse_stmt(Parser* p);
FnDef parse_fn(Parser* p);
StructDef parse_struct(Parser* p);
Program parse_program(void* tokens);
Expr stmt_to_expr(Stmt* stmt);
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

const char* KIND_DOT() {
    return (const char*)"Dot";
}

const char* KIND_LBRACKET() {
    return (const char*)"LBracket";
}

const char* KIND_RBRACKET() {
    return (const char*)"RBracket";
}

const char* KIND_ANDAND() {
    return (const char*)"AndAnd";
}

const char* KIND_LT() {
    return (const char*)"Lt";
}

const char* KIND_GT() {
    return (const char*)"Gt";
}

Lexer lexer_new(const char* source) {
    return (Lexer)((Lexer){ .source = source, .pos = 0, .line = 1, .col = 1 });
}

bool is_digit(const char* ch) {
    return (bool)((strcmp(ch, "0") >= 0) && (strcmp(ch, "9") <= 0));
}

bool is_alpha(const char* ch) {
    return (bool)((((strcmp(ch, "a") >= 0) && (strcmp(ch, "z") <= 0)) || ((strcmp(ch, "A") >= 0) && (strcmp(ch, "Z") <= 0))) || (strcmp(ch, "_") == 0));
}

bool is_alnum(const char* ch) {
    return (bool)(is_alpha(ch) || is_digit(ch));
}

bool is_ws(const char* ch) {
    return (bool)((((strcmp(ch, " ") == 0) || (strcmp(ch, "\n") == 0)) || (strcmp(ch, "\t") == 0)) || (strcmp(ch, "r") == 0));
}

const char* lexer_peek(Lexer* lexer) {
    /* Let tmp_0 declared=Base(Str) val_ty=Base(Str) */
    const char* tmp_0 = ((lexer->pos < ((int64_t)strlen(lexer->source))) ? str_at(lexer->source, lexer->pos) : "");
    return (const char*)tmp_0;
}

const char* lexer_advance(Lexer* lexer) {
    /* Let ch declared=Base(Str) val_ty=Base(Str) */
    const char* ch = lexer_peek(lexer);
    if ((strcmp(ch, "") == 0)) {
    return (const char*)"";
    }
    lexer->pos = (lexer->pos + 1);
    if ((strcmp(ch, "\n") == 0)) {
    lexer->line = (lexer->line + 1);
    lexer->col = 1;
    } else {
    lexer->col = (lexer->col + 1);
    }
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
    while ((strcmp(ch, "") != 0)) {
    if (is_ws(ch)) {
    ch = lexer_advance(lexer);
    ch = lexer_peek(lexer);
    } else {
    if ((strcmp(ch, "/") == 0)) {
    /* Let next_pos declared=Base(I64) val_ty=Base(I64) */
    int64_t next_pos = (lexer->pos + 1);
    if (((next_pos < ((int64_t)strlen(lexer->source))) && (strcmp(str_at(lexer->source, next_pos), "/") == 0))) {
    lexer_advance(lexer);
    lexer_advance(lexer);
    ch = lexer_peek(lexer);
    while (((strcmp(ch, "") != 0) && (strcmp(ch, "\n") != 0))) {
    ch = lexer_advance(lexer);
    ch = lexer_peek(lexer);
    0;
    }
    } else {
    break;
    }
    } else {
    break;
    }
    }
    ch = lexer_peek(lexer);
    0;
    }
    /* Let line declared=Base(I64) val_ty=Base(I64) */
    int64_t line = lexer->line;
    /* Let col declared=Base(I64) val_ty=Base(I64) */
    int64_t col = lexer->col;
    if ((strcmp(ch, "") == 0)) {
    return (Token)make_token(KIND_EOF(), "", line, col);
    }
    if (is_digit(ch)) {
    /* Let num_str declared=Base(Str) val_ty=Base(Str) */
    const char* num_str = "";
    while (is_digit(ch)) {
    num_str = str_add(num_str, ch);
    ch = lexer_advance(lexer);
    ch = lexer_peek(lexer);
    0;
    }
    return (Token)make_token(KIND_INT(), num_str, line, col);
    }
    if (is_alpha(ch)) {
    /* Let ident declared=Base(Str) val_ty=Base(Str) */
    const char* ident = "";
    while (((strcmp(ch, "") != 0) && is_alnum(ch))) {
    ident = str_add(ident, ch);
    ch = lexer_advance(lexer);
    ch = lexer_peek(lexer);
    0;
    }
    if ((strcmp(ident, "fn") == 0)) {
    return (Token)make_token(KIND_FN(), ident, line, col);
    }
    if ((strcmp(ident, "let") == 0)) {
    return (Token)make_token(KIND_LET(), ident, line, col);
    }
    if ((strcmp(ident, "if") == 0)) {
    return (Token)make_token(KIND_IF(), ident, line, col);
    }
    if ((strcmp(ident, "else") == 0)) {
    return (Token)make_token(KIND_ELSE(), ident, line, col);
    }
    if ((strcmp(ident, "return") == 0)) {
    return (Token)make_token(KIND_RETURN(), ident, line, col);
    }
    if ((strcmp(ident, "struct") == 0)) {
    return (Token)make_token(KIND_STRUCT(), ident, line, col);
    }
    return (Token)make_token(KIND_IDENT(), ident, line, col);
    }
    if ((strcmp(ch, "\"") == 0)) {
    lexer_advance(lexer);
    /* Let s declared=Base(Str) val_ty=Base(Str) */
    const char* s = "";
    ch = lexer_peek(lexer);
    while (((strcmp(ch, "") != 0) && (strcmp(ch, "\"") != 0))) {
    s = str_add(s, ch);
    ch = lexer_advance(lexer);
    ch = lexer_peek(lexer);
    0;
    }
    lexer_advance(lexer);
    return (Token)make_token(KIND_IDENT(), s, line, col);
    }
    lexer_advance(lexer);
    if ((strcmp(ch, "(") == 0)) {
    return (Token)make_token(KIND_LPAREN(), "(", line, col);
    }
    if ((strcmp(ch, ")") == 0)) {
    return (Token)make_token(KIND_RPAREN(), ")", line, col);
    }
    if ((strcmp(ch, "{") == 0)) {
    return (Token)make_token(KIND_LBRACE(), "{", line, col);
    }
    if ((strcmp(ch, "}") == 0)) {
    return (Token)make_token(KIND_RBRACE(), "}", line, col);
    }
    if ((strcmp(ch, "+") == 0)) {
    return (Token)make_token(KIND_PLUS(), "+", line, col);
    }
    if ((strcmp(ch, "-") == 0)) {
    if ((strcmp(lexer_peek(lexer), ">") == 0)) {
    lexer_advance(lexer);
    return (Token)make_token(KIND_ARROW(), "->", line, col);
    }
    return (Token)make_token(KIND_MINUS(), "-", line, col);
    }
    if ((strcmp(ch, "*") == 0)) {
    return (Token)make_token(KIND_STAR(), "*", line, col);
    }
    if ((strcmp(ch, "/") == 0)) {
    return (Token)make_token(KIND_SLASH(), "/", line, col);
    }
    if ((strcmp(ch, ";") == 0)) {
    return (Token)make_token(KIND_SEMICOLON(), ";", line, col);
    }
    if ((strcmp(ch, ":") == 0)) {
    if ((strcmp(lexer_peek(lexer), ":") == 0)) {
    lexer_advance(lexer);
    return (Token)make_token(KIND_COLON2(), "::", line, col);
    }
    return (Token)make_token(KIND_COLON(), ":", line, col);
    }
    if ((strcmp(ch, ",") == 0)) {
    return (Token)make_token(KIND_COMMA(), ",", line, col);
    }
    if ((strcmp(ch, "=") == 0)) {
    if ((strcmp(lexer_peek(lexer), "=") == 0)) {
    lexer_advance(lexer);
    return (Token)make_token(KIND_EQEQ(), "==", line, col);
    }
    return (Token)make_token(KIND_ASSIGN(), "=", line, col);
    }
    if ((strcmp(ch, ".") == 0)) {
    return (Token)make_token(KIND_DOT(), ".", line, col);
    }
    if ((strcmp(ch, "[") == 0)) {
    return (Token)make_token(KIND_LBRACKET(), "[", line, col);
    }
    if ((strcmp(ch, "]") == 0)) {
    return (Token)make_token(KIND_RBRACKET(), "]", line, col);
    }
    if ((strcmp(ch, "<") == 0)) {
    return (Token)make_token(KIND_LT(), "<", line, col);
    }
    if ((strcmp(ch, ">") == 0)) {
    return (Token)make_token(KIND_GT(), ">", line, col);
    }
    if ((strcmp(ch, "&") == 0)) {
    if ((strcmp(lexer_peek(lexer), "&") == 0)) {
    lexer_advance(lexer);
    return (Token)make_token(KIND_ANDAND(), "&&", line, col);
    }
    }
    return (Token)make_token(KIND_IDENT(), ch, line, col);
}

void* lexer_tokenize(Lexer* lexer) {
    /* Let tokens declared=Unknown val_ty=Unknown */
    void* tokens = Vec_new();
    /* Let done declared=Base(Bool) val_ty=Base(Bool) */
    bool done = false;
    while ((!done)) {
    /* Let t declared=TypeParam { name: "Token" } val_ty=TypeParam { name: "Token" } */
    Token t = lexer_next(lexer);
    /* Let ifv_1 declared=Unknown val_ty=Unknown */
    void* ifv_1 = 0;
    if ((strcmp((t).kind, KIND_EOF()) == 0)) {
    Vec_push(tokens, ({ Token* _cp = malloc(sizeof(Token)); *_cp = t; _cp; }));
    done = true;
    ifv_1 = 0;
    } else {
    ifv_1 = Vec_push(tokens, ({ Token* _cp = malloc(sizeof(Token)); *_cp = t; _cp; }));
    }
    /* Let tmp_0 declared=Unknown val_ty=Unknown */
    void* tmp_0 = ifv_1;
    tmp_0;
    }
    return (void*)tokens;
}

Parser parser_new(void* tokens) {
    return (Parser)((Parser){ .tokens = tokens, .pos = 0 });
}

Token parser_next(Parser* p) {
    /* Let ifv_1 declared=TypeParam { name: "Token" } val_ty=TypeParam { name: "Token" } */
    Token ifv_1 = (Token){0};
    if ((p->pos < Vec_len(p->tokens))) {
    /* Let t declared=Unknown val_ty=Unknown */
    void* t = Vec_get(p->tokens, p->pos);
    p->pos = (p->pos + 1);
    ifv_1 = *(Token*)t;
    } else {
    ifv_1 = make_token(KIND_EOF(), "", 0, 0);
    }
    /* Let tmp_0 declared=TypeParam { name: "Token" } val_ty=TypeParam { name: "Token" } */
    Token tmp_0 = ifv_1;
    return (Token)tmp_0;
}

Expr make_int_expr(const char* v) {
    return (Expr)((Expr){ .kind = "int", .ival = 0, .sval = v, .op = "", .left = 0, .right = 0, .right_sval = "" });
}

Expr make_ident_expr(const char* n) {
    return (Expr)((Expr){ .kind = "ident", .ival = 0, .sval = n, .op = "", .left = 0, .right = 0, .right_sval = "" });
}

Expr make_binary_expr(Expr l, const char* o, Expr r) {
    return (Expr)((Expr){ .kind = "binary", .ival = 0, .sval = (l).sval, .op = o, .left = 0, .right = (r).ival, .right_sval = (r).sval });
}

Expr make_assign_expr(const char* target, Expr value) {
    return (Expr)((Expr){ .kind = "assign", .ival = 0, .sval = target, .op = "=", .left = 0, .right = (value).ival, .right_sval = (value).sval });
}

Expr make_call_expr(const char* func, const char* args) {
    return (Expr)((Expr){ .kind = "call", .ival = 0, .sval = func, .op = "", .left = 0, .right = 0, .right_sval = args });
}

Token cur_tok(Parser* p) {
    return *(Token*)(Vec_get(p->tokens, p->pos));
}

Expr parse_primary(Parser* p) {
    /* Let t declared=TypeParam { name: "Token" } val_ty=TypeParam { name: "Token" } */
    Token t = parser_next(p);
    /* Let ifv_6 declared=TypeParam { name: "Expr" } val_ty=TypeParam { name: "Expr" } */
    Expr ifv_6 = (Expr){0};
    if ((strcmp((t).kind, KIND_INT()) == 0)) {
    ifv_6 = make_int_expr((t).value);
    } else {
    /* Let ifv_5 declared=TypeParam { name: "Expr" } val_ty=TypeParam { name: "Expr" } */
    Expr ifv_5 = (Expr){0};
    if ((strcmp((t).kind, KIND_IDENT()) == 0)) {
    /* Let name declared=Base(Str) val_ty=Base(Str) */
    const char* name = (t).value;
    /* Let ifv_3 declared=TypeParam { name: "Expr" } val_ty=TypeParam { name: "Expr" } */
    Expr ifv_3 = (Expr){0};
    if (((p->pos < Vec_len(p->tokens)) && (strcmp(((Stmt*)Vec_get(p->tokens, p->pos))->kind, KIND_LPAREN()) == 0))) {
    parser_next(p);
    /* Let args declared=Base(Str) val_ty=Base(Str) */
    const char* args = "";
    while (((p->pos < Vec_len(p->tokens)) && (strcmp(((Stmt*)Vec_get(p->tokens, p->pos))->kind, KIND_RPAREN()) != 0))) {
    /* Let a declared=TypeParam { name: "Expr" } val_ty=TypeParam { name: "Expr" } */
    Expr a = parse_expr(p);
    if ((strcmp(args, "") != 0)) {
    args = str_add(args, ", ");
    }
    args = str_add(args, (a).sval);
    /* Let tmp_2 declared=Base(Unit) val_ty=TypeParam { name: "Token" } */
    Token tmp_2 = parser_next(p);
    tmp_2;
    }
    parser_next(p);
    ifv_3 = make_call_expr(name, args);
    } else {
    ifv_3 = make_ident_expr(name);
    }
    /* Let tmp_1 declared=TypeParam { name: "Expr" } val_ty=TypeParam { name: "Expr" } */
    Expr tmp_1 = ifv_3;
    ifv_5 = tmp_1;
    } else {
    /* Let ifv_4 declared=TypeParam { name: "Expr" } val_ty=TypeParam { name: "Expr" } */
    Expr ifv_4 = (Expr){0};
    if ((strcmp((t).kind, KIND_LPAREN()) == 0)) {
    /* Let e declared=TypeParam { name: "Expr" } val_ty=TypeParam { name: "Expr" } */
    Expr e = parse_expr(p);
    parser_next(p);
    ifv_4 = e;
    } else {
    ifv_4 = make_int_expr("0");
    }
    ifv_5 = ifv_4;
    }
    ifv_6 = ifv_5;
    }
    /* Let tmp_0 declared=TypeParam { name: "Expr" } val_ty=TypeParam { name: "Expr" } */
    Expr tmp_0 = ifv_6;
    return (Expr)tmp_0;
}

Expr parse_postfix(Parser* p) {
    /* Let expr declared=TypeParam { name: "Expr" } val_ty=TypeParam { name: "Expr" } */
    Expr expr = parse_primary(p);
    while ((p->pos < Vec_len(p->tokens))) {
    /* Let k declared=Unknown val_ty=Unknown */
    void* k = ((Stmt*)Vec_get(p->tokens, p->pos))->kind;
    if ((strcmp(k, KIND_DOT()) == 0)) {
    parser_next(p);
    /* Let field declared=Base(Str) val_ty=Base(Str) */
    const char* field = (parser_next(p)).value;
    (expr).sval = str_add(str_add((expr).sval, "."), field);
    } else {
    if ((strcmp(k, KIND_LBRACKET()) == 0)) {
    parser_next(p);
    /* Let idx declared=TypeParam { name: "Expr" } val_ty=TypeParam { name: "Expr" } */
    Expr idx = parse_expr(p);
    parser_next(p);
    (expr).sval = str_add(str_add(str_add((expr).sval, "["), (idx).sval), "]");
    } else {
    break;
    }
    }
    0;
    }
    return (Expr)expr;
}

Expr parse_binary(Parser* p, int64_t min_prec) {
    /* Let left declared=TypeParam { name: "Expr" } val_ty=TypeParam { name: "Expr" } */
    Expr left = parse_postfix(p);
    while ((p->pos < Vec_len(p->tokens))) {
    /* Let k declared=Unknown val_ty=Unknown */
    void* k = ((Stmt*)Vec_get(p->tokens, p->pos))->kind;
    if (((((((((strcmp(k, KIND_PLUS()) == 0) || (strcmp(k, KIND_STAR()) == 0)) || (strcmp(k, KIND_SLASH()) == 0)) || (strcmp(k, KIND_MINUS()) == 0)) || (strcmp(k, KIND_EQEQ()) == 0)) || (strcmp(k, KIND_LT()) == 0)) || (strcmp(k, KIND_GT()) == 0)) || (strcmp(k, KIND_ANDAND()) == 0))) {
    /* Let op_token declared=TypeParam { name: "Token" } val_ty=TypeParam { name: "Token" } */
    Token op_token = parser_next(p);
    /* Let op_str declared=Base(Str) val_ty=Base(Str) */
    const char* op_str = (op_token).value;
    /* Let right declared=TypeParam { name: "Expr" } val_ty=TypeParam { name: "Expr" } */
    Expr right = parse_postfix(p);
    left = make_binary_expr(left, op_str, right);
    } else {
    break;
    }
    0;
    }
    return (Expr)left;
}

Expr parse_expr(Parser* p) {
    /* Let left declared=TypeParam { name: "Expr" } val_ty=TypeParam { name: "Expr" } */
    Expr left = parse_binary(p, 0);
    /* Let ifv_1 declared=TypeParam { name: "Expr" } val_ty=TypeParam { name: "Expr" } */
    Expr ifv_1 = (Expr){0};
    if (((p->pos < Vec_len(p->tokens)) && (strcmp(((Stmt*)Vec_get(p->tokens, p->pos))->kind, KIND_ASSIGN()) == 0))) {
    parser_next(p);
    /* Let right declared=TypeParam { name: "Expr" } val_ty=TypeParam { name: "Expr" } */
    Expr right = parse_expr(p);
    ifv_1 = make_assign_expr((left).sval, right);
    } else {
    ifv_1 = left;
    }
    /* Let tmp_0 declared=TypeParam { name: "Expr" } val_ty=TypeParam { name: "Expr" } */
    Expr tmp_0 = ifv_1;
    return (Expr)tmp_0;
}

Stmt parse_stmt(Parser* p) {
    /* Let t declared=TypeParam { name: "Token" } val_ty=TypeParam { name: "Token" } */
    Token t = parser_next(p);
    /* Let ifv_2 declared=TypeParam { name: "Stmt" } val_ty=TypeParam { name: "Stmt" } */
    Stmt ifv_2 = (Stmt){0};
    if ((strcmp((t).kind, KIND_LET()) == 0)) {
    /* Let name declared=Base(Str) val_ty=Base(Str) */
    const char* name = (parser_next(p)).value;
    parser_next(p);
    /* Let val declared=TypeParam { name: "Expr" } val_ty=TypeParam { name: "Expr" } */
    Expr val = parse_expr(p);
    parser_next(p);
    ifv_2 = ((Stmt){ .kind = "let", .name = name, .ival = (val).ival, .sval = (val).sval, .op = (val).op, .left = (val).left, .right = (val).right, .right_sval = (val).right_sval });
    } else {
    /* Let ifv_1 declared=TypeParam { name: "Stmt" } val_ty=TypeParam { name: "Stmt" } */
    Stmt ifv_1 = (Stmt){0};
    if ((strcmp((t).kind, KIND_RETURN()) == 0)) {
    /* Let val declared=TypeParam { name: "Expr" } val_ty=TypeParam { name: "Expr" } */
    Expr val = parse_expr(p);
    parser_next(p);
    ifv_1 = ((Stmt){ .kind = "return", .name = "", .ival = (val).ival, .sval = (val).sval, .op = (val).op, .left = (val).left, .right = (val).right, .right_sval = (val).right_sval });
    } else {
    /* Let val declared=TypeParam { name: "Expr" } val_ty=TypeParam { name: "Expr" } */
    Expr val = parse_expr(p);
    ifv_1 = ((Stmt){ .kind = "expr", .name = "", .ival = (val).ival, .sval = (val).sval, .op = (val).op, .left = (val).left, .right = (val).right, .right_sval = (val).right_sval });
    }
    ifv_2 = ifv_1;
    }
    /* Let tmp_0 declared=TypeParam { name: "Stmt" } val_ty=TypeParam { name: "Stmt" } */
    Stmt tmp_0 = ifv_2;
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
    while ((strcmp((tok).kind, KIND_RPAREN()) != 0)) {
    /* Let pname declared=Base(Str) val_ty=Base(Str) */
    const char* pname = (tok).value;
    parser_next(p);
    /* Let ptype declared=Base(Str) val_ty=Base(Str) */
    const char* ptype = (parser_next(p)).value;
    Vec_push(params, &((Param){ .name = pname, .type_ann = ptype }));
    tok = parser_next(p);
    if ((strcmp((tok).kind, KIND_COMMA()) == 0)) {
    tok = parser_next(p);
    }
    0;
    }
    tok = parser_next(p);
    if ((strcmp((tok).kind, KIND_ARROW()) == 0)) {
    parser_next(p);
    tok = parser_next(p);
    }
    /* Let body declared=Unknown val_ty=Unknown */
    void* body = Vec_new();
    while (((p->pos < Vec_len(p->tokens)) && (strcmp(((Stmt*)Vec_get(p->tokens, p->pos))->kind, KIND_RBRACE()) != 0))) {
    Vec_push(body, ({ Stmt* _t = malloc(sizeof(Stmt)); *_t = parse_stmt(p); _t; }));
    }
    parser_next(p);
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
    while ((strcmp((tok).kind, KIND_RBRACE()) != 0)) {
    /* Let fname declared=Base(Str) val_ty=Base(Str) */
    const char* fname = (tok).value;
    parser_next(p);
    /* Let ftype declared=Base(Str) val_ty=Base(Str) */
    const char* ftype = (parser_next(p)).value;
    Vec_push(fields, &((Param){ .name = fname, .type_ann = ftype }));
    tok = parser_next(p);
    if ((strcmp((tok).kind, KIND_COMMA()) == 0)) {
    tok = parser_next(p);
    }
    0;
    }
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
    while (1) {
    if (((p).pos >= Vec_len((p).tokens))) {
    break;
    }
    if ((strcmp(((Stmt*)Vec_get((p).tokens, (p).pos))->kind, KIND_EOF()) == 0)) {
    break;
    }
    ((strcmp(((Stmt*)Vec_get((p).tokens, (p).pos))->kind, KIND_STRUCT()) == 0) ? Vec_push(structs, ({ StructDef* _t = malloc(sizeof(StructDef)); *_t = parse_struct(&p); _t; })) : ((strcmp(((Stmt*)Vec_get((p).tokens, (p).pos))->kind, KIND_FN()) == 0) ? Vec_push(fns, ({ FnDef* _t = malloc(sizeof(FnDef)); *_t = parse_fn(&p); _t; })) : Vec_push(main_stmts, ({ Stmt* _t = malloc(sizeof(Stmt)); *_t = parse_stmt(&p); _t; }))));
    }
    return (Program)((Program){ .structs = structs, .fns = fns, .main_stmts = main_stmts });
}

Expr stmt_to_expr(Stmt* stmt) {
    return (Expr)((Expr){ .kind = "int", .ival = stmt->ival, .sval = stmt->sval, .op = stmt->op, .left = stmt->left, .right = stmt->right, .right_sval = stmt->right_sval });
}

const char* cgen_expr(Expr* expr) {
    /* Let ifv_3 declared=Base(Str) val_ty=Base(Str) */
    const char* ifv_3 = 0;
    if ((strcmp(expr->kind, "int") == 0)) {
    ifv_3 = expr->sval;
    } else {
    /* Let ifv_2 declared=Base(Str) val_ty=Base(Str) */
    const char* ifv_2 = 0;
    if ((strcmp(expr->kind, "ident") == 0)) {
    ifv_2 = expr->sval;
    } else {
    /* Let ifv_1 declared=Base(Str) val_ty=Base(Str) */
    const char* ifv_1 = 0;
    if ((strcmp(expr->kind, "binary") == 0)) {
    /* Let r declared=Base(Str) val_ty=Base(Str) */
    const char* r = expr->right_sval;
    if ((strcmp(r, "") == 0)) {
    r = str_add("", str_int(expr->right));
    }
    ifv_1 = str_add(str_add(str_add(str_add(str_add(str_add("(", expr->sval), " "), expr->op), " "), r), ")");
    } else {
    ifv_1 = ((strcmp(expr->kind, "assign") == 0) ? str_add(str_add(expr->sval, " = "), expr->right_sval) : ((strcmp(expr->kind, "call") == 0) ? str_add(str_add(str_add(expr->sval, "("), expr->right_sval), ")") : "0"));
    }
    ifv_2 = ifv_1;
    }
    ifv_3 = ifv_2;
    }
    /* Let tmp_0 declared=Base(Str) val_ty=Base(Str) */
    const char* tmp_0 = ifv_3;
    return (const char*)tmp_0;
}

const char* cgen_stmt(Stmt* stmt) {
    /* Let ifv_5 declared=Base(Str) val_ty=Base(Str) */
    const char* ifv_5 = 0;
    if ((strcmp(stmt->kind, "let") == 0)) {
    /* Let ref_tmp_1 declared=TypeParam { name: "Expr" } val_ty=TypeParam { name: "Expr" } */
    Expr ref_tmp_1 = stmt_to_expr(stmt);
    ifv_5 = str_add(str_add(str_add(str_add("int ", stmt->name), " = "), cgen_expr((Expr*)(&ref_tmp_1))), ";\n");
    } else {
    /* Let ifv_4 declared=Base(Str) val_ty=Base(Str) */
    const char* ifv_4 = 0;
    if ((strcmp(stmt->kind, "return") == 0)) {
    /* Let ref_tmp_2 declared=TypeParam { name: "Expr" } val_ty=TypeParam { name: "Expr" } */
    Expr ref_tmp_2 = stmt_to_expr(stmt);
    ifv_4 = str_add(str_add("return ", cgen_expr((Expr*)(&ref_tmp_2))), ";\n");
    } else {
    /* Let ref_tmp_3 declared=TypeParam { name: "Expr" } val_ty=TypeParam { name: "Expr" } */
    Expr ref_tmp_3 = stmt_to_expr(stmt);
    ifv_4 = str_add(cgen_expr((Expr*)(&ref_tmp_3)), ";\n");
    }
    ifv_5 = ifv_4;
    }
    /* Let tmp_0 declared=Base(Str) val_ty=Base(Str) */
    const char* tmp_0 = ifv_5;
    return (const char*)tmp_0;
}

const char* cgen_struct(StructDef* s) {
    /* Let out declared=Base(Str) val_ty=Base(Str) */
    const char* out = "typedef struct {\n";
    /* Let i declared=Base(I32) val_ty=Base(I32) */
    int32_t i = 0;
    while ((i < Vec_len(s->fields))) {
    /* Let f declared=Unknown val_ty=Unknown */
    void* f = Vec_get(s->fields, i);
    out = str_add(str_add(str_add(out, "    int "), ((StructDef*)f)->name), ";\n");
    i = (i + 1);
    0;
    }
    out = str_add(str_add(str_add(out, "} "), s->name), ";\n\n");
    return (const char*)out;
}

const char* cgen_fn(FnDef* f) {
    /* Let out declared=Base(Str) val_ty=Base(Str) */
    const char* out = str_add(str_add("int ", f->name), "(");
    /* Let i declared=Base(I32) val_ty=Base(I32) */
    int32_t i = 0;
    while ((i < Vec_len(f->params))) {
    /* Let p declared=Unknown val_ty=Unknown */
    void* p = Vec_get(f->params, i);
    if ((i > 0)) {
    out = str_add(out, ", ");
    }
    out = str_add(str_add(out, "int "), ((StructDef*)p)->name);
    i = (i + 1);
    0;
    }
    out = str_add(out, ") {\n");
    i = 0;
    while ((i < Vec_len(f->body))) {
    /* Let ref_tmp_0 declared=Unknown val_ty=Unknown */
    void* ref_tmp_0 = Vec_get(f->body, i);
    out = str_add(str_add(out, "    "), cgen_stmt(ref_tmp_0));
    i = (i + 1);
    0;
    }
    out = str_add(out, "}\n\n");
    return (const char*)out;
}

const char* cgen_program(Program* prog) {
    /* Let out declared=Base(Str) val_ty=Base(Str) */
    const char* out = "#include <stdio.h>\n\n";
    /* Let i declared=Base(I32) val_ty=Base(I32) */
    int32_t i = 0;
    while ((i < Vec_len(prog->structs))) {
    /* Let ref_tmp_0 declared=Unknown val_ty=Unknown */
    void* ref_tmp_0 = Vec_get(prog->structs, i);
    out = str_add(out, cgen_struct(ref_tmp_0));
    i = (i + 1);
    0;
    }
    i = 0;
    while ((i < Vec_len(prog->fns))) {
    /* Let ref_tmp_1 declared=Unknown val_ty=Unknown */
    void* ref_tmp_1 = Vec_get(prog->fns, i);
    out = str_add(out, cgen_fn(ref_tmp_1));
    i = (i + 1);
    0;
    }
    out = str_add(out, "int main(void) {\n");
    i = 0;
    while ((i < Vec_len(prog->main_stmts))) {
    /* Let ref_tmp_2 declared=Unknown val_ty=Unknown */
    void* ref_tmp_2 = Vec_get(prog->main_stmts, i);
    out = str_add(str_add(out, "    "), cgen_stmt(ref_tmp_2));
    i = (i + 1);
    0;
    }
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

