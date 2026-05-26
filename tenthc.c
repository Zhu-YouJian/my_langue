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
extern bool str_eq(const char* a, const char* b);
extern void println(const char* s);
extern void* HashMap_new(void);

typedef struct StructField {
    const char* name;
    const char* type_ann;
} StructField;
typedef struct Lexer {
    const char* source;
    int64_t pos;
    int64_t line;
    int64_t col;
} Lexer;
typedef struct MatchArm {
    const char* pat_kind;
    const char* pat_name;
    const char* pat_bind;
    int64_t body_expr;
} MatchArm;
typedef struct Stmt {
    const char* kind;
    const char* name;
    int64_t expr_idx;
    int64_t body_start;
    int64_t body_count;
    int64_t else_start;
    int64_t else_count;
} Stmt;
typedef struct Program {
    void* structs;
    void* enums;
    void* fns;
    int64_t main_stmts_start;
    int64_t main_stmts_count;
    void* expr_nodes;
    void* stmt_nodes;
    void* match_arms;
} Program;
typedef struct EnumVariant {
    const char* name;
    void* fields;
} EnumVariant;
typedef struct Span {
    int64_t line;
    int64_t col;
} Span;
typedef struct Parser {
    void* tokens;
    int64_t pos;
    void* expr_nodes;
    void* stmt_nodes;
    void* match_arms;
} Parser;
typedef struct EnumDef {
    const char* name;
    void* variants;
} EnumDef;
typedef struct FnDef {
    const char* name;
    void* params;
    const char* return_type;
    int64_t body_start;
    int64_t body_count;
} FnDef;
typedef struct Expr {
    const char* kind;
    int64_t ival;
    double fval;
    const char* sval;
    int64_t left;
    int64_t right;
    int64_t arg_start;
    int64_t arg_count;
    int64_t extra_start;
    int64_t extra_count;
} Expr;
typedef struct StructDef {
    const char* name;
    void* fields;
} StructDef;
typedef struct Param {
    const char* name;
    const char* type_ann;
} Param;
typedef struct Token {
    int64_t kind;
    Span span;
} Token;

Lexer lexer_new(const char* source);
bool is_digit(const char* ch);
bool is_alpha(const char* ch);
bool is_alnum(const char* ch);
bool is_ws(const char* ch);
const char* lexer_peek(Lexer* lexer);
const char* lexer_advance(Lexer* lexer);
Span make_span(int64_t line, int64_t col);
Token lexer_next(Lexer* lexer);
void* lexer_tokenize(Lexer* lexer);
Parser parser_new(void* tokens);
Token parser_peek(Parser* p);
bool parser_at_eof(Parser* p);
Token parser_advance(Parser* p);
int64_t add_expr(Parser* p, const char* kind);
int64_t add_stmt(Parser* p, const char* kind);
int64_t parse_primary(Parser* p);
int64_t parse_unary(Parser* p);
int64_t parse_postfix(Parser* p);
int64_t parse_arg_list(Parser* p);
int64_t binop_prec(const char* op);
const char* token_to_binop_str(Token tok);
bool is_binop(Token tok);
int64_t parse_binary(Parser* p, int64_t min_prec);
int64_t parse_expr(Parser* p);
int64_t parse_block_expr(Parser* p);
int64_t parse_if_expr(Parser* p);
int64_t parse_match_expr(Parser* p);
int64_t parse_stmt(Parser* p);
FnDef parse_fn(Parser* p);
StructDef parse_struct_def(Parser* p);
EnumDef parse_enum_def(Parser* p);
bool is_end_of_block(Token tok);
bool is_end_of_paren(Token tok);
bool is_stmt_end(Token tok);
Program parse_program(void* tokens);
const char* cgen_expr(Program* prog, void* exprs, void* stmts, int64_t idx);
const char* cgen_stmt(Program* prog, void* exprs, void* stmts, int64_t idx);
const char* cgen_struct(StructDef* s);
const char* cgen_param(Param* p);
const char* cgen_fn(Program* prog, void* exprs, void* stmts, FnDef* f);
const char* cgen_program(Program* prog, void* exprs, void* stmts);
const char* int_to_str(int64_t n);
const char* float_to_str(double n);
const char* compile(const char* src);
int main();
Lexer lexer_new(const char* source) {
    return (Lexer)((Lexer){ .source = source, .pos = 0, .line = 1, .col = 1 });
}

bool is_digit(const char* ch) {
    return (bool)((strcmp(ch, "0") >= 0) && (strcmp(ch, "9") <= 0));
}

bool is_alpha(const char* ch) {
    return (bool)((((strcmp(ch, "a") >= 0) && (strcmp(ch, "z") <= 0)) || ((strcmp(ch, "A") >= 0) && (strcmp(ch, "Z") <= 0))) || str_eq(ch, "_"));
}

bool is_alnum(const char* ch) {
    return (bool)(is_alpha(ch) || is_digit(ch));
}

bool is_ws(const char* ch) {
    return (bool)(((str_eq(ch, " ") || str_eq(ch, "\n")) || str_eq(ch, "\t")) || str_eq(ch, "r"));
}

const char* lexer_peek(Lexer* lexer) {
    /* Let tmp_0 declared=Base(Str) val_ty=Base(Str) */
    const char* tmp_0 = ((lexer->pos < ((int64_t)strlen(lexer->source))) ? str_at(lexer->source, lexer->pos) : "");
    return (const char*)tmp_0;
}

const char* lexer_advance(Lexer* lexer) {
    /* Let ch declared=Base(Str) val_ty=Base(Str) */
    const char* ch = lexer_peek(lexer);
    if (str_eq(ch, "")) {
    return (const char*)"";
    }
    lexer->pos = (lexer->pos + 1);
    if (str_eq(ch, "\n")) {
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

Token lexer_next(Lexer* lexer) {
    /* Let ch declared=Base(Str) val_ty=Base(Str) */
    const char* ch = lexer_peek(lexer);
    while ((!str_eq(ch, ""))) {
    if (is_ws(ch)) {
    ch = lexer_advance(lexer);
    ch = lexer_peek(lexer);
    } else {
    if (str_eq(ch, "/")) {
    /* Let next_pos declared=Base(I64) val_ty=Base(I64) */
    int64_t next_pos = (lexer->pos + 1);
    if (((next_pos < ((int64_t)strlen(lexer->source))) && str_eq(str_at(lexer->source, next_pos), "/"))) {
    lexer_advance(lexer);
    lexer_advance(lexer);
    ch = lexer_peek(lexer);
    while (((!str_eq(ch, "")) && (!str_eq(ch, "\n")))) {
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
    /* Let span declared=TypeParam { name: "Span" } val_ty=TypeParam { name: "Span" } */
    Span span = make_span(line, col);
    if (str_eq(ch, "")) {
    return (Token)((Token){ .kind = 61, .span = span });
    }
    if (is_digit(ch)) {
    /* Let num_str declared=Base(Str) val_ty=Base(Str) */
    const char* num_str = "";
    /* Let is_float declared=Base(Bool) val_ty=Base(Bool) */
    bool is_float = false;
    while (is_digit(ch)) {
    num_str = str_add(num_str, ch);
    ch = lexer_advance(lexer);
    ch = lexer_peek(lexer);
    0;
    }
    if (str_eq(ch, ".")) {
    is_float = true;
    num_str = str_add(num_str, ch);
    ch = lexer_advance(lexer);
    ch = lexer_peek(lexer);
    while (is_digit(ch)) {
    num_str = str_add(num_str, ch);
    ch = lexer_advance(lexer);
    ch = lexer_peek(lexer);
    0;
    }
    }
    if (is_float) {
    return (Token)((Token){ .kind = 1, .span = span });
    }
    return (Token)((Token){ .kind = 0, .span = span });
    }
    if (is_alpha(ch)) {
    /* Let ident declared=Base(Str) val_ty=Base(Str) */
    const char* ident = "";
    while (((!str_eq(ch, "")) && is_alnum(ch))) {
    ident = str_add(ident, ch);
    ch = lexer_advance(lexer);
    ch = lexer_peek(lexer);
    0;
    }
    if (str_eq(ident, "fn")) {
    return (Token)((Token){ .kind = 4, .span = span });
    }
    if (str_eq(ident, "let")) {
    return (Token)((Token){ .kind = 5, .span = span });
    }
    if (str_eq(ident, "mut")) {
    return (Token)((Token){ .kind = 6, .span = span });
    }
    if (str_eq(ident, "if")) {
    return (Token)((Token){ .kind = 7, .span = span });
    }
    if (str_eq(ident, "else")) {
    return (Token)((Token){ .kind = 8, .span = span });
    }
    if (str_eq(ident, "match")) {
    return (Token)((Token){ .kind = 9, .span = span });
    }
    if (str_eq(ident, "return")) {
    return (Token)((Token){ .kind = 10, .span = span });
    }
    if (str_eq(ident, "while")) {
    return (Token)((Token){ .kind = 11, .span = span });
    }
    if (str_eq(ident, "for")) {
    return (Token)((Token){ .kind = 12, .span = span });
    }
    if (str_eq(ident, "loop")) {
    return (Token)((Token){ .kind = 13, .span = span });
    }
    if (str_eq(ident, "break")) {
    return (Token)((Token){ .kind = 14, .span = span });
    }
    if (str_eq(ident, "continue")) {
    return (Token)((Token){ .kind = 15, .span = span });
    }
    if (str_eq(ident, "struct")) {
    return (Token)((Token){ .kind = 16, .span = span });
    }
    if (str_eq(ident, "enum")) {
    return (Token)((Token){ .kind = 17, .span = span });
    }
    if (str_eq(ident, "impl")) {
    return (Token)((Token){ .kind = 18, .span = span });
    }
    if (str_eq(ident, "trait")) {
    return (Token)((Token){ .kind = 19, .span = span });
    }
    if (str_eq(ident, "use")) {
    return (Token)((Token){ .kind = 20, .span = span });
    }
    if (str_eq(ident, "mod")) {
    return (Token)((Token){ .kind = 21, .span = span });
    }
    if (str_eq(ident, "true")) {
    return (Token)((Token){ .kind = 22, .span = span });
    }
    if (str_eq(ident, "false")) {
    return (Token)((Token){ .kind = 23, .span = span });
    }
    if (str_eq(ident, "move")) {
    return (Token)((Token){ .kind = 24, .span = span });
    }
    if (str_eq(ident, "self")) {
    return (Token)((Token){ .kind = 25, .span = span });
    }
    return (Token)((Token){ .kind = 3, .span = span });
    }
    if (str_eq(ch, "\"")) {
    lexer_advance(lexer);
    /* Let s declared=Base(Str) val_ty=Base(Str) */
    const char* s = "";
    ch = lexer_peek(lexer);
    while (((!str_eq(ch, "")) && (!str_eq(ch, "\"")))) {
    s = str_add(s, ch);
    ch = lexer_advance(lexer);
    ch = lexer_peek(lexer);
    0;
    }
    if (str_eq(ch, "\"")) {
    lexer_advance(lexer);
    }
    return (Token)((Token){ .kind = 2, .span = span });
    }
    lexer_advance(lexer);
    if (str_eq(ch, "(")) {
    return (Token)((Token){ .kind = 47, .span = span });
    }
    if (str_eq(ch, ")")) {
    return (Token)((Token){ .kind = 48, .span = span });
    }
    if (str_eq(ch, "{")) {
    return (Token)((Token){ .kind = 49, .span = span });
    }
    if (str_eq(ch, "}")) {
    return (Token)((Token){ .kind = 50, .span = span });
    }
    if (str_eq(ch, "[")) {
    return (Token)((Token){ .kind = 51, .span = span });
    }
    if (str_eq(ch, "]")) {
    return (Token)((Token){ .kind = 52, .span = span });
    }
    if (str_eq(ch, ",")) {
    return (Token)((Token){ .kind = 53, .span = span });
    }
    if (str_eq(ch, ";")) {
    return (Token)((Token){ .kind = 54, .span = span });
    }
    if (str_eq(ch, ".")) {
    if (str_eq(lexer_peek(lexer), ".")) {
    lexer_advance(lexer);
    return (Token)((Token){ .kind = 57, .span = span });
    }
    return (Token)((Token){ .kind = 56, .span = span });
    }
    if (str_eq(ch, ":")) {
    if (str_eq(lexer_peek(lexer), ":")) {
    lexer_advance(lexer);
    return (Token)((Token){ .kind = 60, .span = span });
    }
    return (Token)((Token){ .kind = 55, .span = span });
    }
    if (str_eq(ch, "+")) {
    if (str_eq(lexer_peek(lexer), "=")) {
    lexer_advance(lexer);
    return (Token)((Token){ .kind = 43, .span = span });
    }
    return (Token)((Token){ .kind = 26, .span = span });
    }
    if (str_eq(ch, "-")) {
    if (str_eq(lexer_peek(lexer), ">")) {
    lexer_advance(lexer);
    return (Token)((Token){ .kind = 58, .span = span });
    }
    if (str_eq(lexer_peek(lexer), "=")) {
    lexer_advance(lexer);
    return (Token)((Token){ .kind = 44, .span = span });
    }
    return (Token)((Token){ .kind = 27, .span = span });
    }
    if (str_eq(ch, "*")) {
    if (str_eq(lexer_peek(lexer), "=")) {
    lexer_advance(lexer);
    return (Token)((Token){ .kind = 45, .span = span });
    }
    return (Token)((Token){ .kind = 28, .span = span });
    }
    if (str_eq(ch, "/")) {
    if (str_eq(lexer_peek(lexer), "=")) {
    lexer_advance(lexer);
    return (Token)((Token){ .kind = 46, .span = span });
    }
    return (Token)((Token){ .kind = 29, .span = span });
    }
    if (str_eq(ch, "%")) {
    return (Token)((Token){ .kind = 30, .span = span });
    }
    if (str_eq(ch, "=")) {
    if (str_eq(lexer_peek(lexer), "=")) {
    lexer_advance(lexer);
    return (Token)((Token){ .kind = 31, .span = span });
    }
    if (str_eq(lexer_peek(lexer), ">")) {
    lexer_advance(lexer);
    return (Token)((Token){ .kind = 59, .span = span });
    }
    return (Token)((Token){ .kind = 42, .span = span });
    }
    if (str_eq(ch, "!")) {
    if (str_eq(lexer_peek(lexer), "=")) {
    lexer_advance(lexer);
    return (Token)((Token){ .kind = 32, .span = span });
    }
    return (Token)((Token){ .kind = 39, .span = span });
    }
    if (str_eq(ch, "<")) {
    if (str_eq(lexer_peek(lexer), "=")) {
    lexer_advance(lexer);
    return (Token)((Token){ .kind = 35, .span = span });
    }
    return (Token)((Token){ .kind = 33, .span = span });
    }
    if (str_eq(ch, ">")) {
    if (str_eq(lexer_peek(lexer), "=")) {
    lexer_advance(lexer);
    return (Token)((Token){ .kind = 36, .span = span });
    }
    return (Token)((Token){ .kind = 34, .span = span });
    }
    if (str_eq(ch, "&")) {
    if (str_eq(lexer_peek(lexer), "&")) {
    lexer_advance(lexer);
    return (Token)((Token){ .kind = 37, .span = span });
    }
    return (Token)((Token){ .kind = 40, .span = span });
    }
    if (str_eq(ch, "|")) {
    if (str_eq(lexer_peek(lexer), "|")) {
    lexer_advance(lexer);
    return (Token)((Token){ .kind = 38, .span = span });
    }
    return (Token)((Token){ .kind = 41, .span = span });
    }
    return (Token)((Token){ .kind = 3, .span = span });
}

void* lexer_tokenize(Lexer* lexer) {
    /* Let tokens declared=Unknown val_ty=Unknown */
    void* tokens = Vec_new();
    /* Let src_len declared=Base(I64) val_ty=Base(I64) */
    int64_t src_len = ((int64_t)strlen(lexer->source));
    while ((lexer->pos < src_len)) {
    /* Let t declared=TypeParam { name: "Token" } val_ty=TypeParam { name: "Token" } */
    Token t = lexer_next(lexer);
    Vec_push(tokens, ({ Token* _cp = malloc(sizeof(Token)); *_cp = t; _cp; }));
    }
    Vec_push(tokens, ({ Token* _t = malloc(sizeof(Token)); *_t = lexer_next(lexer); _t; }));
    return (void*)tokens;
}

Parser parser_new(void* tokens) {
    return (Parser)((Parser){ .tokens = tokens, .pos = 0, .expr_nodes = Vec_new(), .stmt_nodes = Vec_new(), .match_arms = Vec_new() });
}

Token parser_peek(Parser* p) {
    /* Let tmp_0 declared=TypeParam { name: "Token" } val_ty=TypeParam { name: "Token" } */
    Token tmp_0 = ((p->pos < Vec_len(p->tokens)) ? *((Token*)(Vec_get(p->tokens, p->pos))) : ((Token){ .kind = 61, .span = ((Span){ .line = 0, .col = 0 }) }));
    return (Token)tmp_0;
}

bool parser_at_eof(Parser* p) {
    return (bool)(p->pos >= Vec_len(p->tokens));
}

Token parser_advance(Parser* p) {
    /* Let ifv_1 declared=TypeParam { name: "Token" } val_ty=TypeParam { name: "Token" } */
    Token ifv_1 = (Token){0};
    if ((p->pos < Vec_len(p->tokens))) {
    /* Let t declared=Unknown val_ty=Unknown */
    void* t = Vec_get(p->tokens, p->pos);
    p->pos = (p->pos + 1);
    ifv_1 = *(Token*)t;
    } else {
    ifv_1 = ((Token){ .kind = 61, .span = ((Span){ .line = 0, .col = 0 }) });
    }
    /* Let tmp_0 declared=TypeParam { name: "Token" } val_ty=TypeParam { name: "Token" } */
    Token tmp_0 = ifv_1;
    return (Token)tmp_0;
}

int64_t add_expr(Parser* p, const char* kind) {
    /* Let e declared=TypeParam { name: "Expr" } val_ty=TypeParam { name: "Expr" } */
    Expr e = ((Expr){ .kind = kind, .ival = 0, .fval = 0.0000000000, .sval = "", .left = 0, .right = 0, .arg_start = 0, .arg_count = 0, .extra_start = 0, .extra_count = 0 });
    Vec_push(p->expr_nodes, ({ Expr* _cp = malloc(sizeof(Expr)); *_cp = e; _cp; }));
    return (int64_t)Vec_len(p->expr_nodes);
}

int64_t add_stmt(Parser* p, const char* kind) {
    /* Let s declared=TypeParam { name: "Stmt" } val_ty=TypeParam { name: "Stmt" } */
    Stmt s = ((Stmt){ .kind = kind, .name = "", .expr_idx = 0, .body_start = 0, .body_count = 0, .else_start = 0, .else_count = 0 });
    Vec_push(p->stmt_nodes, ({ Stmt* _cp = malloc(sizeof(Stmt)); *_cp = s; _cp; }));
    return (int64_t)Vec_len(p->stmt_nodes);
}

int64_t parse_primary(Parser* p) {
    /* Let tok declared=TypeParam { name: "Token" } val_ty=TypeParam { name: "Token" } */
    Token tok = parser_advance(p);
    /* Let match_disc_0 declared=Base(I64) val_ty=TypeParam { name: "TokenKind" } */
    int64_t match_disc_0 = (tok).kind;
    /* Let match_res_1 declared=Unknown val_ty=Unknown */
    void* match_res_1 = 0;
    if ((match_disc_0 == 0)) {
    /* Let idx declared=Base(I64) val_ty=Base(I64) */
    int64_t idx = add_expr(p, "int");
    ((Expr*)Vec_get(p->expr_nodes, (idx - 1)))->ival = n;
    match_res_1 = idx;
    } else {
    if ((match_disc_0 == 1)) {
    /* Let idx declared=Base(I64) val_ty=Base(I64) */
    int64_t idx = add_expr(p, "float");
    ((Expr*)Vec_get(p->expr_nodes, (idx - 1)))->fval = n;
    match_res_1 = idx;
    } else {
    if ((match_disc_0 == 2)) {
    /* Let idx declared=Base(I64) val_ty=Base(I64) */
    int64_t idx = add_expr(p, "str");
    ((Expr*)Vec_get(p->expr_nodes, (idx - 1)))->sval = s;
    match_res_1 = idx;
    } else {
    if ((match_disc_0 == 22)) {
    /* Let idx declared=Base(I64) val_ty=Base(I64) */
    int64_t idx = add_expr(p, "bool");
    ((Expr*)Vec_get(p->expr_nodes, (idx - 1)))->ival = 1;
    match_res_1 = idx;
    } else {
    if ((match_disc_0 == 23)) {
    /* Let idx declared=Base(I64) val_ty=Base(I64) */
    int64_t idx = add_expr(p, "bool");
    ((Expr*)Vec_get(p->expr_nodes, (idx - 1)))->ival = 0;
    match_res_1 = idx;
    } else {
    if ((match_disc_0 == 3)) {
    /* Let idx declared=Base(I64) val_ty=Base(I64) */
    int64_t idx = add_expr(p, "ident");
    ((Expr*)Vec_get(p->expr_nodes, (idx - 1)))->sval = s;
    match_res_1 = idx;
    } else {
    if ((match_disc_0 == 25)) {
    /* Let idx declared=Base(I64) val_ty=Base(I64) */
    int64_t idx = add_expr(p, "ident");
    ((Expr*)Vec_get(p->expr_nodes, (idx - 1)))->sval = "self";
    match_res_1 = idx;
    } else {
    if ((match_disc_0 == 47)) {
    /* Let expr_idx declared=Base(I64) val_ty=Base(I64) */
    int64_t expr_idx = parse_expr(p);
    parser_advance(p);
    match_res_1 = expr_idx;
    } else {
    if ((match_disc_0 == 49)) {
    match_res_1 = parse_block_expr(p);
    } else {
    if ((match_disc_0 == 7)) {
    match_res_1 = parse_if_expr(p);
    } else {
    if ((match_disc_0 == 9)) {
    match_res_1 = parse_match_expr(p);
    } else {
    if ((match_disc_0 == 40)) {
    /* Let inner declared=Base(I64) val_ty=Base(I64) */
    int64_t inner = parse_unary(p);
    /* Let idx declared=Base(I64) val_ty=Base(I64) */
    int64_t idx = add_expr(p, "ref");
    ((Expr*)Vec_get(p->expr_nodes, (idx - 1)))->left = inner;
    match_res_1 = idx;
    } else {
    if ((match_disc_0 == 28)) {
    /* Let inner declared=Base(I64) val_ty=Base(I64) */
    int64_t inner = parse_unary(p);
    /* Let idx declared=Base(I64) val_ty=Base(I64) */
    int64_t idx = add_expr(p, "deref");
    ((Expr*)Vec_get(p->expr_nodes, (idx - 1)))->left = inner;
    match_res_1 = idx;
    } else {
    if ((match_disc_0 == 27)) {
    /* Let inner declared=Base(I64) val_ty=Base(I64) */
    int64_t inner = parse_unary(p);
    /* Let idx declared=Base(I64) val_ty=Base(I64) */
    int64_t idx = add_expr(p, "unary");
    ((Expr*)Vec_get(p->expr_nodes, (idx - 1)))->sval = "-";
    ((Expr*)Vec_get(p->expr_nodes, (idx - 1)))->left = inner;
    match_res_1 = idx;
    } else {
    if ((match_disc_0 == 39)) {
    /* Let inner declared=Base(I64) val_ty=Base(I64) */
    int64_t inner = parse_unary(p);
    /* Let idx declared=Base(I64) val_ty=Base(I64) */
    int64_t idx = add_expr(p, "unary");
    ((Expr*)Vec_get(p->expr_nodes, (idx - 1)))->sval = "!";
    ((Expr*)Vec_get(p->expr_nodes, (idx - 1)))->left = inner;
    match_res_1 = idx;
    } else {
    /* Let idx declared=Base(I64) val_ty=Base(I64) */
    int64_t idx = add_expr(p, "int");
    ((Expr*)Vec_get(p->expr_nodes, (idx - 1)))->ival = 0;
    match_res_1 = idx;
    }
    }
    }
    }
    }
    }
    }
    }
    }
    }
    }
    }
    }
    }
    }
    return (int64_t)match_res_1;
}

int64_t parse_unary(Parser* p) {
    return (int64_t)parse_primary(p);
}

int64_t parse_postfix(Parser* p) {
    /* Let expr_idx declared=Base(I64) val_ty=Base(I64) */
    int64_t expr_idx = parse_unary(p);
    while (1) {
    /* Let tok declared=TypeParam { name: "Token" } val_ty=TypeParam { name: "Token" } */
    Token tok = parser_peek(p);
    /* Let match_disc_0 declared=Base(I64) val_ty=TypeParam { name: "TokenKind" } */
    int64_t match_disc_0 = (tok).kind;
    /* Let match_res_1 declared=Unknown val_ty=Unknown */
    void* match_res_1 = 0;
    if ((match_disc_0 == 47)) {
    parser_advance(p);
    /* Let start declared=Base(I64) val_ty=Base(I64) */
    int64_t start = (Vec_len(p->expr_nodes) + 1);
    /* Let count declared=Base(I32) val_ty=Base(I32) */
    int32_t count = 0;
    count = parse_arg_list(p);
    ((Stmt*)Vec_get(p->expr_nodes, (expr_idx - 1)))->kind = "call";
    ((Expr*)Vec_get(p->expr_nodes, (expr_idx - 1)))->arg_start = start;
    ((Expr*)Vec_get(p->expr_nodes, (expr_idx - 1)))->arg_count = count;
    match_res_1 = 0;
    } else {
    if ((match_disc_0 == 56)) {
    parser_advance(p);
    /* Let field_tok declared=TypeParam { name: "Token" } val_ty=TypeParam { name: "Token" } */
    Token field_tok = parser_advance(p);
    /* Let field_name declared=Base(Str) val_ty=Base(Str) */
    const char* field_name = "";
    /* Let match_disc_2 declared=Base(I64) val_ty=TypeParam { name: "TokenKind" } */
    int64_t match_disc_2 = (field_tok).kind;
    /* Let match_res_3 declared=Unknown val_ty=Unknown */
    void* match_res_3 = 0;
    if ((match_disc_2 == 3)) {
    }
    match_res_3;
    /* Let idx declared=Base(I64) val_ty=Base(I64) */
    int64_t idx = add_expr(p, "field");
    ((Expr*)Vec_get(p->expr_nodes, (idx - 1)))->left = expr_idx;
    ((Expr*)Vec_get(p->expr_nodes, (idx - 1)))->sval = "";
    expr_idx = idx;
    match_res_1 = 0;
    } else {
    break;
    }
    }
    match_res_1;
    }
    return (int64_t)expr_idx;
}

int64_t parse_arg_list(Parser* p) {
    /* Let count declared=Base(I32) val_ty=Base(I32) */
    int32_t count = 0;
    while (1) {
    /* Let tok declared=TypeParam { name: "Token" } val_ty=TypeParam { name: "Token" } */
    Token tok = parser_peek(p);
    /* Let match_disc_0 declared=Base(I64) val_ty=TypeParam { name: "TokenKind" } */
    int64_t match_disc_0 = (tok).kind;
    /* Let match_res_1 declared=Unknown val_ty=Unknown */
    void* match_res_1 = 0;
    if ((match_disc_0 == 48)) {
    break;
    } else {
    if ((match_disc_0 == 61)) {
    break;
    }
    }
    match_res_1;
    /* Let _arg declared=Base(I64) val_ty=Base(I64) */
    int64_t _arg = parse_expr(p);
    count = (count + 1);
    /* Let tok2 declared=TypeParam { name: "Token" } val_ty=TypeParam { name: "Token" } */
    Token tok2 = parser_peek(p);
    /* Let match_disc_2 declared=Base(I64) val_ty=TypeParam { name: "TokenKind" } */
    int64_t match_disc_2 = (tok2).kind;
    /* Let match_res_3 declared=Unknown val_ty=Unknown */
    void* match_res_3 = 0;
    if ((match_disc_2 == 53)) {
    match_res_3 = parser_advance(p);
    } else {
    break;
    }
    match_res_3;
    }
    parser_advance(p);
    return (int64_t)count;
}

int64_t binop_prec(const char* op) {
    /* Let tmp_0 declared=Base(I32) val_ty=Base(I32) */
    int32_t tmp_0 = (((str_eq(op, "*") || str_eq(op, "/")) || str_eq(op, "%")) ? 5 : ((str_eq(op, "+") || str_eq(op, "-")) ? 4 : ((((str_eq(op, "<") || str_eq(op, ">")) || str_eq(op, "<=")) || str_eq(op, ">=")) ? 3 : ((str_eq(op, "==") || str_eq(op, "!=")) ? 2 : (str_eq(op, "&&") ? 1 : (str_eq(op, "||") ? 0 : (-1)))))));
    return (int64_t)tmp_0;
}

const char* token_to_binop_str(Token tok) {
    /* Let match_disc_0 declared=Base(I64) val_ty=TypeParam { name: "TokenKind" } */
    int64_t match_disc_0 = (tok).kind;
    /* Let match_res_1 declared=Unknown val_ty=Unknown */
    void* match_res_1 = 0;
    if ((match_disc_0 == 26)) {
    match_res_1 = "+";
    } else {
    if ((match_disc_0 == 27)) {
    match_res_1 = "-";
    } else {
    if ((match_disc_0 == 28)) {
    match_res_1 = "*";
    } else {
    if ((match_disc_0 == 29)) {
    match_res_1 = "/";
    } else {
    if ((match_disc_0 == 30)) {
    match_res_1 = "%";
    } else {
    if ((match_disc_0 == 31)) {
    match_res_1 = "==";
    } else {
    if ((match_disc_0 == 32)) {
    match_res_1 = "!=";
    } else {
    if ((match_disc_0 == 33)) {
    match_res_1 = "<";
    } else {
    if ((match_disc_0 == 34)) {
    match_res_1 = ">";
    } else {
    if ((match_disc_0 == 35)) {
    match_res_1 = "<=";
    } else {
    if ((match_disc_0 == 36)) {
    match_res_1 = ">=";
    } else {
    if ((match_disc_0 == 37)) {
    match_res_1 = "&&";
    } else {
    if ((match_disc_0 == 38)) {
    match_res_1 = "||";
    } else {
    match_res_1 = "";
    }
    }
    }
    }
    }
    }
    }
    }
    }
    }
    }
    }
    }
    return *(const char**)(match_res_1);
}

bool is_binop(Token tok) {
    /* Let op declared=Base(Str) val_ty=Base(Str) */
    const char* op = token_to_binop_str(tok);
    return (bool)(!str_eq(op, ""));
}

int64_t parse_binary(Parser* p, int64_t min_prec) {
    /* Let left declared=Base(I64) val_ty=Base(I64) */
    int64_t left = parse_postfix(p);
    while (1) {
    /* Let tok declared=TypeParam { name: "Token" } val_ty=TypeParam { name: "Token" } */
    Token tok = parser_peek(p);
    if ((!is_binop(tok))) {
    break;
    }
    /* Let op declared=Base(Str) val_ty=Base(Str) */
    const char* op = token_to_binop_str(tok);
    /* Let prec declared=Base(I64) val_ty=Base(I64) */
    int64_t prec = binop_prec(op);
    if ((prec < min_prec)) {
    break;
    }
    parser_advance(p);
    /* Let right declared=Base(I64) val_ty=Base(I64) */
    int64_t right = parse_binary(p, (prec + 1));
    /* Let idx declared=Base(I64) val_ty=Base(I64) */
    int64_t idx = add_expr(p, "binary");
    ((Expr*)Vec_get(p->expr_nodes, (idx - 1)))->sval = op;
    ((Expr*)Vec_get(p->expr_nodes, (idx - 1)))->left = left;
    ((Expr*)Vec_get(p->expr_nodes, (idx - 1)))->right = right;
    left = idx;
    }
    return (int64_t)left;
}

int64_t parse_expr(Parser* p) {
    /* Let tok declared=TypeParam { name: "Token" } val_ty=TypeParam { name: "Token" } */
    Token tok = parser_peek(p);
    /* Let expr declared=Base(I64) val_ty=Base(I64) */
    int64_t expr = parse_binary(p, 0);
    /* Let tok2 declared=TypeParam { name: "Token" } val_ty=TypeParam { name: "Token" } */
    Token tok2 = parser_peek(p);
    /* Let match_disc_0 declared=Base(I64) val_ty=TypeParam { name: "TokenKind" } */
    int64_t match_disc_0 = (tok2).kind;
    /* Let match_res_1 declared=Unknown val_ty=Unknown */
    void* match_res_1 = 0;
    if ((match_disc_0 == 42)) {
    parser_advance(p);
    /* Let rhs declared=Base(I64) val_ty=Base(I64) */
    int64_t rhs = parse_expr(p);
    /* Let idx declared=Base(I64) val_ty=Base(I64) */
    int64_t idx = add_expr(p, "assign");
    ((Expr*)Vec_get(p->expr_nodes, (idx - 1)))->left = expr;
    ((Expr*)Vec_get(p->expr_nodes, (idx - 1)))->right = rhs;
    match_res_1 = idx;
    } else {
    match_res_1 = expr;
    }
    return (int64_t)match_res_1;
}

int64_t parse_block_expr(Parser* p) {
    /* Let body_start declared=Base(I64) val_ty=Base(I64) */
    int64_t body_start = (Vec_len(p->stmt_nodes) + 1);
    /* Let count declared=Base(I32) val_ty=Base(I32) */
    int32_t count = 0;
    while (1) {
    /* Let tok declared=TypeParam { name: "Token" } val_ty=TypeParam { name: "Token" } */
    Token tok = parser_peek(p);
    /* Let match_disc_0 declared=Base(I64) val_ty=TypeParam { name: "TokenKind" } */
    int64_t match_disc_0 = (tok).kind;
    /* Let match_res_1 declared=Unknown val_ty=Unknown */
    void* match_res_1 = 0;
    if ((match_disc_0 == 50)) {
    break;
    } else {
    if ((match_disc_0 == 61)) {
    break;
    }
    }
    match_res_1;
    parse_stmt(p);
    count = (count + 1);
    }
    parser_advance(p);
    /* Let idx declared=Base(I64) val_ty=Base(I64) */
    int64_t idx = add_expr(p, "block");
    ((Expr*)Vec_get(p->expr_nodes, (idx - 1)))->extra_start = body_start;
    ((Expr*)Vec_get(p->expr_nodes, (idx - 1)))->extra_count = count;
    return (int64_t)idx;
}

int64_t parse_if_expr(Parser* p) {
    /* Let cond declared=Base(I64) val_ty=Base(I64) */
    int64_t cond = parse_expr(p);
    /* Let then_expr declared=Base(I64) val_ty=Base(I64) */
    int64_t then_expr = parse_expr(p);
    /* Let else_idx declared=Base(I32) val_ty=Base(I32) */
    int32_t else_idx = 0;
    /* Let tok declared=TypeParam { name: "Token" } val_ty=TypeParam { name: "Token" } */
    Token tok = parser_peek(p);
    /* Let match_disc_0 declared=Base(I64) val_ty=TypeParam { name: "TokenKind" } */
    int64_t match_disc_0 = (tok).kind;
    /* Let match_res_1 declared=Unknown val_ty=Unknown */
    void* match_res_1 = 0;
    if ((match_disc_0 == 8)) {
    parser_advance(p);
    else_idx = parse_expr(p);
    match_res_1 = 0;
    }
    match_res_1;
    /* Let idx declared=Base(I64) val_ty=Base(I64) */
    int64_t idx = add_expr(p, "if");
    ((Expr*)Vec_get(p->expr_nodes, (idx - 1)))->left = cond;
    ((Expr*)Vec_get(p->expr_nodes, (idx - 1)))->right = then_expr;
    ((Expr*)Vec_get(p->expr_nodes, (idx - 1)))->extra_start = else_idx;
    return (int64_t)idx;
}

int64_t parse_match_expr(Parser* p) {
    /* Let scrutinee declared=Base(I64) val_ty=Base(I64) */
    int64_t scrutinee = parse_expr(p);
    parser_advance(p);
    /* Let arm_start declared=Base(I64) val_ty=Base(I64) */
    int64_t arm_start = (Vec_len(p->match_arms) + 1);
    /* Let arm_count declared=Base(I32) val_ty=Base(I32) */
    int32_t arm_count = 0;
    while (1) {
    /* Let tok declared=TypeParam { name: "Token" } val_ty=TypeParam { name: "Token" } */
    Token tok = parser_peek(p);
    /* Let match_disc_0 declared=Base(I64) val_ty=TypeParam { name: "TokenKind" } */
    int64_t match_disc_0 = (tok).kind;
    /* Let match_res_1 declared=Unknown val_ty=Unknown */
    void* match_res_1 = 0;
    if ((match_disc_0 == 50)) {
    break;
    } else {
    if ((match_disc_0 == 61)) {
    break;
    }
    }
    match_res_1;
    /* Let pat_tok declared=TypeParam { name: "Token" } val_ty=TypeParam { name: "Token" } */
    Token pat_tok = parser_advance(p);
    /* Let pat_kind declared=Base(Str) val_ty=Base(Str) */
    const char* pat_kind = "wildcard";
    /* Let pat_name declared=Base(Str) val_ty=Base(Str) */
    const char* pat_name = "";
    /* Let pat_bind declared=Base(Str) val_ty=Base(Str) */
    const char* pat_bind = "";
    /* Let match_disc_2 declared=Base(I64) val_ty=TypeParam { name: "TokenKind" } */
    int64_t match_disc_2 = (pat_tok).kind;
    /* Let match_res_3 declared=Unknown val_ty=Unknown */
    void* match_res_3 = 0;
    if ((match_disc_2 == 3)) {
    pat_kind = "enum_variant";
    pat_name = s;
    /* Let next declared=TypeParam { name: "Token" } val_ty=TypeParam { name: "Token" } */
    Token next = parser_peek(p);
    /* Let match_disc_4 declared=Base(I64) val_ty=TypeParam { name: "TokenKind" } */
    int64_t match_disc_4 = (next).kind;
    /* Let match_res_5 declared=Unknown val_ty=Unknown */
    void* match_res_5 = 0;
    if ((match_disc_4 == 60)) {
    parser_advance(p);
    /* Let variant_tok declared=TypeParam { name: "Token" } val_ty=TypeParam { name: "Token" } */
    Token variant_tok = parser_advance(p);
    /* Let match_disc_6 declared=Base(I64) val_ty=TypeParam { name: "TokenKind" } */
    int64_t match_disc_6 = (variant_tok).kind;
    /* Let match_res_7 declared=Unknown val_ty=Unknown */
    void* match_res_7 = 0;
    if ((match_disc_6 == 3)) {
    pat_name = vs;
    match_res_7 = 0;
    }
    match_res_5 = match_res_7;
    }
    match_res_3 = match_res_5;
    }
    match_res_3;
    parser_advance(p);
    /* Let body declared=Base(I64) val_ty=Base(I64) */
    int64_t body = parse_expr(p);
    /* Let arm declared=TypeParam { name: "MatchArm" } val_ty=TypeParam { name: "MatchArm" } */
    MatchArm arm = ((MatchArm){ .pat_kind = pat_kind, .pat_name = pat_name, .pat_bind = pat_bind, .body_expr = body });
    Vec_push(p->match_arms, ({ MatchArm* _cp = malloc(sizeof(MatchArm)); *_cp = arm; _cp; }));
    arm_count = (arm_count + 1);
    /* Let t declared=TypeParam { name: "Token" } val_ty=TypeParam { name: "Token" } */
    Token t = parser_peek(p);
    /* Let match_disc_8 declared=Base(I64) val_ty=TypeParam { name: "TokenKind" } */
    int64_t match_disc_8 = (t).kind;
    /* Let match_res_9 declared=Unknown val_ty=Unknown */
    void* match_res_9 = 0;
    if ((match_disc_8 == 53)) {
    match_res_9 = parser_advance(p);
    }
    match_res_9;
    }
    parser_advance(p);
    /* Let idx declared=Base(I64) val_ty=Base(I64) */
    int64_t idx = add_expr(p, "match");
    ((Expr*)Vec_get(p->expr_nodes, (idx - 1)))->left = scrutinee;
    ((Expr*)Vec_get(p->expr_nodes, (idx - 1)))->extra_start = arm_start;
    ((Expr*)Vec_get(p->expr_nodes, (idx - 1)))->extra_count = arm_count;
    return (int64_t)idx;
}

int64_t parse_stmt(Parser* p) {
    /* Let tok declared=TypeParam { name: "Token" } val_ty=TypeParam { name: "Token" } */
    Token tok = parser_peek(p);
    /* Let match_disc_0 declared=Base(I64) val_ty=TypeParam { name: "TokenKind" } */
    int64_t match_disc_0 = (tok).kind;
    /* Let match_res_1 declared=Unknown val_ty=Unknown */
    void* match_res_1 = 0;
    if ((match_disc_0 == 5)) {
    parser_advance(p);
    /* Let mutable declared=Base(Bool) val_ty=Base(Bool) */
    bool mutable = false;
    /* Let t2 declared=TypeParam { name: "Token" } val_ty=TypeParam { name: "Token" } */
    Token t2 = parser_peek(p);
    /* Let match_disc_2 declared=Base(I64) val_ty=TypeParam { name: "TokenKind" } */
    int64_t match_disc_2 = (t2).kind;
    /* Let match_res_3 declared=Unknown val_ty=Unknown */
    void* match_res_3 = 0;
    if ((match_disc_2 == 6)) {
    parser_advance(p);
    mutable = true;
    match_res_3 = 0;
    }
    match_res_3;
    /* Let name_tok declared=TypeParam { name: "Token" } val_ty=TypeParam { name: "Token" } */
    Token name_tok = parser_advance(p);
    /* Let name declared=Base(Str) val_ty=Base(Str) */
    const char* name = "";
    /* Let match_disc_4 declared=Base(I64) val_ty=TypeParam { name: "TokenKind" } */
    int64_t match_disc_4 = (name_tok).kind;
    /* Let match_res_5 declared=Unknown val_ty=Unknown */
    void* match_res_5 = 0;
    if ((match_disc_4 == 3)) {
    name = s;
    match_res_5 = 0;
    }
    match_res_5;
    /* Let t3 declared=TypeParam { name: "Token" } val_ty=TypeParam { name: "Token" } */
    Token t3 = parser_peek(p);
    /* Let match_disc_6 declared=Base(I64) val_ty=TypeParam { name: "TokenKind" } */
    int64_t match_disc_6 = (t3).kind;
    /* Let match_res_7 declared=Unknown val_ty=Unknown */
    void* match_res_7 = 0;
    if ((match_disc_6 == 55)) {
    parser_advance(p);
    match_res_7 = parser_advance(p);
    }
    match_res_7;
    /* Let t4 declared=TypeParam { name: "Token" } val_ty=TypeParam { name: "Token" } */
    Token t4 = parser_peek(p);
    /* Let match_disc_8 declared=Base(I64) val_ty=TypeParam { name: "TokenKind" } */
    int64_t match_disc_8 = (t4).kind;
    /* Let match_res_9 declared=Unknown val_ty=Unknown */
    void* match_res_9 = 0;
    if ((match_disc_8 == 42)) {
    parser_advance(p);
    /* Let init declared=Base(I64) val_ty=Base(I64) */
    int64_t init = parse_expr(p);
    /* Let idx declared=Base(I64) val_ty=Base(I64) */
    int64_t idx = add_stmt(p, "let");
    ((EnumDef*)Vec_get(p->stmt_nodes, (idx - 1)))->name = name;
    ((Stmt*)Vec_get(p->stmt_nodes, (idx - 1)))->expr_idx = init;
    match_res_9 = idx;
    } else {
    /* Let idx declared=Base(I64) val_ty=Base(I64) */
    int64_t idx = add_stmt(p, "let");
    ((EnumDef*)Vec_get(p->stmt_nodes, (idx - 1)))->name = name;
    match_res_9 = idx;
    }
    match_res_1 = match_res_9;
    } else {
    if ((match_disc_0 == 10)) {
    parser_advance(p);
    /* Let t2 declared=TypeParam { name: "Token" } val_ty=TypeParam { name: "Token" } */
    Token t2 = parser_peek(p);
    /* Let expr_idx declared=Base(I32) val_ty=Base(I32) */
    int32_t expr_idx = 0;
    if ((!is_stmt_end(t2))) {
    expr_idx = parse_expr(p);
    }
    /* Let idx declared=Base(I64) val_ty=Base(I64) */
    int64_t idx = add_stmt(p, "return");
    ((Stmt*)Vec_get(p->stmt_nodes, (idx - 1)))->expr_idx = expr_idx;
    match_res_1 = idx;
    } else {
    if ((match_disc_0 == 11)) {
    parser_advance(p);
    /* Let cond declared=Base(I64) val_ty=Base(I64) */
    int64_t cond = parse_expr(p);
    parser_advance(p);
    /* Let body_start declared=Base(I64) val_ty=Base(I64) */
    int64_t body_start = (Vec_len(p->stmt_nodes) + 1);
    /* Let body_count declared=Base(I32) val_ty=Base(I32) */
    int32_t body_count = 0;
    while (1) {
    /* Let t2 declared=TypeParam { name: "Token" } val_ty=TypeParam { name: "Token" } */
    Token t2 = parser_peek(p);
    if (is_end_of_block(t2)) {
    break;
    }
    parse_stmt(p);
    body_count = (body_count + 1);
    }
    parser_advance(p);
    /* Let idx declared=Base(I64) val_ty=Base(I64) */
    int64_t idx = add_stmt(p, "while");
    ((Stmt*)Vec_get(p->stmt_nodes, (idx - 1)))->expr_idx = cond;
    ((FnDef*)Vec_get(p->stmt_nodes, (idx - 1)))->body_start = body_start;
    ((FnDef*)Vec_get(p->stmt_nodes, (idx - 1)))->body_count = body_count;
    match_res_1 = idx;
    } else {
    if ((match_disc_0 == 14)) {
    parser_advance(p);
    match_res_1 = add_stmt(p, "break");
    } else {
    if ((match_disc_0 == 15)) {
    parser_advance(p);
    match_res_1 = add_stmt(p, "continue");
    } else {
    if ((match_disc_0 == 13)) {
    parser_advance(p);
    parser_advance(p);
    /* Let body_start declared=Base(I64) val_ty=Base(I64) */
    int64_t body_start = (Vec_len(p->stmt_nodes) + 1);
    /* Let body_count declared=Base(I32) val_ty=Base(I32) */
    int32_t body_count = 0;
    while (1) {
    /* Let t2 declared=TypeParam { name: "Token" } val_ty=TypeParam { name: "Token" } */
    Token t2 = parser_peek(p);
    if (is_end_of_block(t2)) {
    break;
    }
    parse_stmt(p);
    body_count = (body_count + 1);
    }
    parser_advance(p);
    /* Let idx declared=Base(I64) val_ty=Base(I64) */
    int64_t idx = add_stmt(p, "loop");
    ((FnDef*)Vec_get(p->stmt_nodes, (idx - 1)))->body_start = body_start;
    ((FnDef*)Vec_get(p->stmt_nodes, (idx - 1)))->body_count = body_count;
    match_res_1 = idx;
    } else {
    /* Let expr declared=Base(I64) val_ty=Base(I64) */
    int64_t expr = parse_expr(p);
    /* Let idx declared=Base(I64) val_ty=Base(I64) */
    int64_t idx = add_stmt(p, "expr");
    ((Stmt*)Vec_get(p->stmt_nodes, (idx - 1)))->expr_idx = expr;
    match_res_1 = idx;
    }
    }
    }
    }
    }
    }
    return (int64_t)match_res_1;
}

FnDef parse_fn(Parser* p) {
    parser_advance(p);
    /* Let name_tok declared=TypeParam { name: "Token" } val_ty=TypeParam { name: "Token" } */
    Token name_tok = parser_advance(p);
    /* Let name declared=Base(Str) val_ty=Base(Str) */
    const char* name = "";
    /* Let match_disc_0 declared=Base(I64) val_ty=TypeParam { name: "TokenKind" } */
    int64_t match_disc_0 = (name_tok).kind;
    /* Let match_res_1 declared=Unknown val_ty=Unknown */
    void* match_res_1 = 0;
    if ((match_disc_0 == 3)) {
    name = s;
    match_res_1 = 0;
    }
    match_res_1;
    parser_advance(p);
    /* Let params declared=Unknown val_ty=Unknown */
    void* params = Vec_new();
    while (1) {
    /* Let t declared=TypeParam { name: "Token" } val_ty=TypeParam { name: "Token" } */
    Token t = parser_peek(p);
    if (is_end_of_paren(t)) {
    break;
    }
    /* Let pname_tok declared=TypeParam { name: "Token" } val_ty=TypeParam { name: "Token" } */
    Token pname_tok = parser_advance(p);
    /* Let pname declared=Base(Str) val_ty=Base(Str) */
    const char* pname = "";
    /* Let match_disc_2 declared=Base(I64) val_ty=TypeParam { name: "TokenKind" } */
    int64_t match_disc_2 = (pname_tok).kind;
    /* Let match_res_3 declared=Unknown val_ty=Unknown */
    void* match_res_3 = 0;
    if ((match_disc_2 == 3)) {
    pname = s;
    match_res_3 = 0;
    }
    match_res_3;
    parser_advance(p);
    /* Let ptype_tok declared=TypeParam { name: "Token" } val_ty=TypeParam { name: "Token" } */
    Token ptype_tok = parser_advance(p);
    /* Let ptype declared=Base(Str) val_ty=Base(Str) */
    const char* ptype = "";
    /* Let match_disc_4 declared=Base(I64) val_ty=TypeParam { name: "TokenKind" } */
    int64_t match_disc_4 = (ptype_tok).kind;
    /* Let match_res_5 declared=Unknown val_ty=Unknown */
    void* match_res_5 = 0;
    if ((match_disc_4 == 3)) {
    ptype = s;
    match_res_5 = 0;
    }
    match_res_5;
    Vec_push(params, ({ Param* _cp = malloc(sizeof(Param)); *_cp = ((Param){ .name = pname, .type_ann = ptype }); _cp; }));
    /* Let t2 declared=TypeParam { name: "Token" } val_ty=TypeParam { name: "Token" } */
    Token t2 = parser_peek(p);
    /* Let match_disc_6 declared=Base(I64) val_ty=TypeParam { name: "TokenKind" } */
    int64_t match_disc_6 = (t2).kind;
    /* Let match_res_7 declared=Unknown val_ty=Unknown */
    void* match_res_7 = 0;
    if ((match_disc_6 == 53)) {
    match_res_7 = parser_advance(p);
    } else {
    break;
    }
    match_res_7;
    }
    parser_advance(p);
    /* Let return_type declared=Base(Str) val_ty=Base(Str) */
    const char* return_type = "";
    /* Let t3 declared=TypeParam { name: "Token" } val_ty=TypeParam { name: "Token" } */
    Token t3 = parser_peek(p);
    /* Let match_disc_8 declared=Base(I64) val_ty=TypeParam { name: "TokenKind" } */
    int64_t match_disc_8 = (t3).kind;
    /* Let match_res_9 declared=Unknown val_ty=Unknown */
    void* match_res_9 = 0;
    if ((match_disc_8 == 58)) {
    parser_advance(p);
    /* Let ret_tok declared=TypeParam { name: "Token" } val_ty=TypeParam { name: "Token" } */
    Token ret_tok = parser_advance(p);
    /* Let match_disc_10 declared=Base(I64) val_ty=TypeParam { name: "TokenKind" } */
    int64_t match_disc_10 = (ret_tok).kind;
    /* Let match_res_11 declared=Unknown val_ty=Unknown */
    void* match_res_11 = 0;
    if ((match_disc_10 == 3)) {
    return_type = s;
    match_res_11 = 0;
    }
    match_res_9 = match_res_11;
    }
    match_res_9;
    parser_advance(p);
    /* Let body_start declared=Base(I64) val_ty=Base(I64) */
    int64_t body_start = (Vec_len(p->stmt_nodes) + 1);
    /* Let body_count declared=Base(I32) val_ty=Base(I32) */
    int32_t body_count = 0;
    while (1) {
    /* Let t4 declared=TypeParam { name: "Token" } val_ty=TypeParam { name: "Token" } */
    Token t4 = parser_peek(p);
    if (is_end_of_block(t4)) {
    break;
    }
    parse_stmt(p);
    body_count = (body_count + 1);
    }
    parser_advance(p);
    return (FnDef)((FnDef){ .name = name, .params = params, .return_type = return_type, .body_start = body_start, .body_count = body_count });
}

StructDef parse_struct_def(Parser* p) {
    parser_advance(p);
    /* Let name_tok declared=TypeParam { name: "Token" } val_ty=TypeParam { name: "Token" } */
    Token name_tok = parser_advance(p);
    /* Let name declared=Base(Str) val_ty=Base(Str) */
    const char* name = "";
    /* Let match_disc_0 declared=Base(I64) val_ty=TypeParam { name: "TokenKind" } */
    int64_t match_disc_0 = (name_tok).kind;
    /* Let match_res_1 declared=Unknown val_ty=Unknown */
    void* match_res_1 = 0;
    if ((match_disc_0 == 3)) {
    name = s;
    match_res_1 = 0;
    }
    match_res_1;
    parser_advance(p);
    /* Let fields declared=Unknown val_ty=Unknown */
    void* fields = Vec_new();
    while (1) {
    /* Let t declared=TypeParam { name: "Token" } val_ty=TypeParam { name: "Token" } */
    Token t = parser_peek(p);
    if (is_end_of_block(t)) {
    break;
    }
    /* Let fname_tok declared=TypeParam { name: "Token" } val_ty=TypeParam { name: "Token" } */
    Token fname_tok = parser_advance(p);
    /* Let fname declared=Base(Str) val_ty=Base(Str) */
    const char* fname = "";
    /* Let match_disc_2 declared=Base(I64) val_ty=TypeParam { name: "TokenKind" } */
    int64_t match_disc_2 = (fname_tok).kind;
    /* Let match_res_3 declared=Unknown val_ty=Unknown */
    void* match_res_3 = 0;
    if ((match_disc_2 == 3)) {
    fname = s;
    match_res_3 = 0;
    }
    match_res_3;
    parser_advance(p);
    /* Let ftype_tok declared=TypeParam { name: "Token" } val_ty=TypeParam { name: "Token" } */
    Token ftype_tok = parser_advance(p);
    /* Let ftype declared=Base(Str) val_ty=Base(Str) */
    const char* ftype = "";
    /* Let match_disc_4 declared=Base(I64) val_ty=TypeParam { name: "TokenKind" } */
    int64_t match_disc_4 = (ftype_tok).kind;
    /* Let match_res_5 declared=Unknown val_ty=Unknown */
    void* match_res_5 = 0;
    if ((match_disc_4 == 3)) {
    ftype = s;
    match_res_5 = 0;
    }
    match_res_5;
    Vec_push(fields, ({ StructField* _cp = malloc(sizeof(StructField)); *_cp = ((StructField){ .name = fname, .type_ann = ftype }); _cp; }));
    /* Let t2 declared=TypeParam { name: "Token" } val_ty=TypeParam { name: "Token" } */
    Token t2 = parser_peek(p);
    /* Let match_disc_6 declared=Base(I64) val_ty=TypeParam { name: "TokenKind" } */
    int64_t match_disc_6 = (t2).kind;
    /* Let match_res_7 declared=Unknown val_ty=Unknown */
    void* match_res_7 = 0;
    if ((match_disc_6 == 53)) {
    match_res_7 = parser_advance(p);
    } else {
    break;
    }
    match_res_7;
    }
    parser_advance(p);
    return (StructDef)((StructDef){ .name = name, .fields = fields });
}

EnumDef parse_enum_def(Parser* p) {
    parser_advance(p);
    /* Let name_tok declared=TypeParam { name: "Token" } val_ty=TypeParam { name: "Token" } */
    Token name_tok = parser_advance(p);
    /* Let name declared=Base(Str) val_ty=Base(Str) */
    const char* name = "";
    /* Let match_disc_0 declared=Base(I64) val_ty=TypeParam { name: "TokenKind" } */
    int64_t match_disc_0 = (name_tok).kind;
    /* Let match_res_1 declared=Unknown val_ty=Unknown */
    void* match_res_1 = 0;
    if ((match_disc_0 == 3)) {
    name = s;
    match_res_1 = 0;
    }
    match_res_1;
    parser_advance(p);
    /* Let variants declared=Unknown val_ty=Unknown */
    void* variants = Vec_new();
    while (1) {
    /* Let t declared=TypeParam { name: "Token" } val_ty=TypeParam { name: "Token" } */
    Token t = parser_peek(p);
    if (is_end_of_block(t)) {
    break;
    }
    /* Let vname_tok declared=TypeParam { name: "Token" } val_ty=TypeParam { name: "Token" } */
    Token vname_tok = parser_advance(p);
    /* Let vname declared=Base(Str) val_ty=Base(Str) */
    const char* vname = "";
    /* Let match_disc_2 declared=Base(I64) val_ty=TypeParam { name: "TokenKind" } */
    int64_t match_disc_2 = (vname_tok).kind;
    /* Let match_res_3 declared=Unknown val_ty=Unknown */
    void* match_res_3 = 0;
    if ((match_disc_2 == 3)) {
    vname = s;
    match_res_3 = 0;
    }
    match_res_3;
    /* Let vfields declared=Unknown val_ty=Unknown */
    void* vfields = Vec_new();
    /* Let t2 declared=TypeParam { name: "Token" } val_ty=TypeParam { name: "Token" } */
    Token t2 = parser_peek(p);
    /* Let match_disc_4 declared=Base(I64) val_ty=TypeParam { name: "TokenKind" } */
    int64_t match_disc_4 = (t2).kind;
    /* Let match_res_5 declared=Unknown val_ty=Unknown */
    void* match_res_5 = 0;
    if ((match_disc_4 == 47)) {
    parser_advance(p);
    while (1) {
    /* Let t3 declared=TypeParam { name: "Token" } val_ty=TypeParam { name: "Token" } */
    Token t3 = parser_peek(p);
    if (is_end_of_paren(t3)) {
    break;
    }
    /* Let fname_tok declared=TypeParam { name: "Token" } val_ty=TypeParam { name: "Token" } */
    Token fname_tok = parser_advance(p);
    /* Let fname declared=Base(Str) val_ty=Base(Str) */
    const char* fname = "";
    /* Let match_disc_6 declared=Base(I64) val_ty=TypeParam { name: "TokenKind" } */
    int64_t match_disc_6 = (fname_tok).kind;
    /* Let match_res_7 declared=Unknown val_ty=Unknown */
    void* match_res_7 = 0;
    if ((match_disc_6 == 3)) {
    fname = s;
    match_res_7 = 0;
    }
    match_res_7;
    parser_advance(p);
    /* Let ftype_tok declared=TypeParam { name: "Token" } val_ty=TypeParam { name: "Token" } */
    Token ftype_tok = parser_advance(p);
    /* Let ftype declared=Base(Str) val_ty=Base(Str) */
    const char* ftype = "";
    /* Let match_disc_8 declared=Base(I64) val_ty=TypeParam { name: "TokenKind" } */
    int64_t match_disc_8 = (ftype_tok).kind;
    /* Let match_res_9 declared=Unknown val_ty=Unknown */
    void* match_res_9 = 0;
    if ((match_disc_8 == 3)) {
    ftype = s;
    match_res_9 = 0;
    }
    match_res_9;
    Vec_push(vfields, ({ StructField* _cp = malloc(sizeof(StructField)); *_cp = ((StructField){ .name = fname, .type_ann = ftype }); _cp; }));
    /* Let t4 declared=TypeParam { name: "Token" } val_ty=TypeParam { name: "Token" } */
    Token t4 = parser_peek(p);
    /* Let match_disc_10 declared=Base(I64) val_ty=TypeParam { name: "TokenKind" } */
    int64_t match_disc_10 = (t4).kind;
    /* Let match_res_11 declared=Unknown val_ty=Unknown */
    void* match_res_11 = 0;
    if ((match_disc_10 == 53)) {
    match_res_11 = parser_advance(p);
    } else {
    break;
    }
    match_res_11;
    }
    match_res_5 = parser_advance(p);
    }
    match_res_5;
    Vec_push(variants, ({ EnumVariant* _cp = malloc(sizeof(EnumVariant)); *_cp = ((EnumVariant){ .name = vname, .fields = vfields }); _cp; }));
    /* Let t5 declared=TypeParam { name: "Token" } val_ty=TypeParam { name: "Token" } */
    Token t5 = parser_peek(p);
    /* Let match_disc_12 declared=Base(I64) val_ty=TypeParam { name: "TokenKind" } */
    int64_t match_disc_12 = (t5).kind;
    /* Let match_res_13 declared=Unknown val_ty=Unknown */
    void* match_res_13 = 0;
    if ((match_disc_12 == 53)) {
    match_res_13 = parser_advance(p);
    } else {
    break;
    }
    match_res_13;
    }
    parser_advance(p);
    return (EnumDef)((EnumDef){ .name = name, .variants = variants });
}

bool is_end_of_block(Token tok) {
    /* Let match_disc_0 declared=Base(I64) val_ty=TypeParam { name: "TokenKind" } */
    int64_t match_disc_0 = (tok).kind;
    /* Let match_res_1 declared=Unknown val_ty=Unknown */
    void* match_res_1 = 0;
    if ((match_disc_0 == 50)) {
    match_res_1 = true;
    } else {
    if ((match_disc_0 == 61)) {
    match_res_1 = true;
    } else {
    match_res_1 = false;
    }
    }
    return *(bool*)(match_res_1);
}

bool is_end_of_paren(Token tok) {
    /* Let match_disc_0 declared=Base(I64) val_ty=TypeParam { name: "TokenKind" } */
    int64_t match_disc_0 = (tok).kind;
    /* Let match_res_1 declared=Unknown val_ty=Unknown */
    void* match_res_1 = 0;
    if ((match_disc_0 == 48)) {
    match_res_1 = true;
    } else {
    if ((match_disc_0 == 61)) {
    match_res_1 = true;
    } else {
    match_res_1 = false;
    }
    }
    return *(bool*)(match_res_1);
}

bool is_stmt_end(Token tok) {
    /* Let match_disc_0 declared=Base(I64) val_ty=TypeParam { name: "TokenKind" } */
    int64_t match_disc_0 = (tok).kind;
    /* Let match_res_1 declared=Unknown val_ty=Unknown */
    void* match_res_1 = 0;
    if ((match_disc_0 == 54)) {
    match_res_1 = true;
    } else {
    if ((match_disc_0 == 50)) {
    match_res_1 = true;
    } else {
    if ((match_disc_0 == 61)) {
    match_res_1 = true;
    } else {
    match_res_1 = false;
    }
    }
    }
    return *(bool*)(match_res_1);
}

Program parse_program(void* tokens) {
    /* Let p declared=TypeParam { name: "Parser" } val_ty=TypeParam { name: "Parser" } */
    Parser p = parser_new(tokens);
    /* Let structs declared=Unknown val_ty=Unknown */
    void* structs = Vec_new();
    /* Let enums declared=Unknown val_ty=Unknown */
    void* enums = Vec_new();
    /* Let fns declared=Unknown val_ty=Unknown */
    void* fns = Vec_new();
    /* Let main_stmts_start declared=Base(I32) val_ty=Base(I32) */
    int32_t main_stmts_start = 1;
    /* Let stmts_before_top_level declared=Base(I64) val_ty=Base(I64) */
    int64_t stmts_before_top_level = Vec_len((p).stmt_nodes);
    while (1) {
    if (parser_at_eof(&p)) {
    break;
    }
    /* Let tok declared=TypeParam { name: "Token" } val_ty=TypeParam { name: "Token" } */
    Token tok = parser_peek(&p);
    /* Let match_disc_0 declared=Base(I64) val_ty=TypeParam { name: "TokenKind" } */
    int64_t match_disc_0 = (tok).kind;
    /* Let match_res_1 declared=Unknown val_ty=Unknown */
    void* match_res_1 = 0;
    if ((match_disc_0 == 16)) {
    Vec_push(structs, ({ StructDef* _t = malloc(sizeof(StructDef)); *_t = parse_struct_def(&p); _t; }));
    /* Let t2 declared=TypeParam { name: "Token" } val_ty=TypeParam { name: "Token" } */
    Token t2 = parser_peek(&p);
    /* Let match_disc_8 declared=Base(I64) val_ty=TypeParam { name: "TokenKind" } */
    int64_t match_disc_8 = (t2).kind;
    /* Let match_res_9 declared=Unknown val_ty=Unknown */
    void* match_res_9 = 0;
    if ((match_disc_8 == 54)) {
    match_res_9 = parser_advance(&p);
    }
    match_res_1 = match_res_9;
    } else {
    if ((match_disc_0 == 17)) {
    Vec_push(enums, ({ EnumDef* _t = malloc(sizeof(EnumDef)); *_t = parse_enum_def(&p); _t; }));
    /* Let t2 declared=TypeParam { name: "Token" } val_ty=TypeParam { name: "Token" } */
    Token t2 = parser_peek(&p);
    /* Let match_disc_6 declared=Base(I64) val_ty=TypeParam { name: "TokenKind" } */
    int64_t match_disc_6 = (t2).kind;
    /* Let match_res_7 declared=Unknown val_ty=Unknown */
    void* match_res_7 = 0;
    if ((match_disc_6 == 54)) {
    match_res_7 = parser_advance(&p);
    }
    match_res_1 = match_res_7;
    } else {
    if ((match_disc_0 == 4)) {
    Vec_push(fns, ({ FnDef* _t = malloc(sizeof(FnDef)); *_t = parse_fn(&p); _t; }));
    /* Let t2 declared=TypeParam { name: "Token" } val_ty=TypeParam { name: "Token" } */
    Token t2 = parser_peek(&p);
    /* Let match_disc_4 declared=Base(I64) val_ty=TypeParam { name: "TokenKind" } */
    int64_t match_disc_4 = (t2).kind;
    /* Let match_res_5 declared=Unknown val_ty=Unknown */
    void* match_res_5 = 0;
    if ((match_disc_4 == 54)) {
    match_res_5 = parser_advance(&p);
    }
    match_res_1 = match_res_5;
    } else {
    if ((match_disc_0 == 61)) {
    break;
    } else {
    parse_stmt(&p);
    /* Let t2 declared=TypeParam { name: "Token" } val_ty=TypeParam { name: "Token" } */
    Token t2 = parser_peek(&p);
    /* Let match_disc_2 declared=Base(I64) val_ty=TypeParam { name: "TokenKind" } */
    int64_t match_disc_2 = (t2).kind;
    /* Let match_res_3 declared=Unknown val_ty=Unknown */
    void* match_res_3 = 0;
    if ((match_disc_2 == 54)) {
    match_res_3 = parser_advance(&p);
    }
    match_res_1 = match_res_3;
    }
    }
    }
    }
    match_res_1;
    }
    /* Let main_count declared=Base(I64) val_ty=Base(I64) */
    int64_t main_count = (Vec_len((p).stmt_nodes) - stmts_before_top_level);
    return (Program)((Program){ .structs = structs, .enums = enums, .fns = fns, .main_stmts_start = main_stmts_start, .main_stmts_count = main_count, .expr_nodes = (p).expr_nodes, .stmt_nodes = (p).stmt_nodes, .match_arms = (p).match_arms });
}

const char* cgen_expr(Program* prog, void* exprs, void* stmts, int64_t idx) {
    if (((idx <= 0) || (idx > Vec_len(exprs)))) {
    return (const char*)"0";
    }
    /* Let e declared=Unknown val_ty=Unknown */
    void* e = Vec_get(exprs, (idx - 1));
    if (str_eq(((Stmt*)e)->kind, "int")) {
    return (const char*)int_to_str(((Expr*)e)->ival);
    }
    if (str_eq(((Stmt*)e)->kind, "float")) {
    return (const char*)float_to_str(((Expr*)e)->fval);
    }
    if (str_eq(((Stmt*)e)->kind, "str")) {
    return (const char*)str_add(str_add("\"", ((Expr*)e)->sval), "\"");
    }
    if (str_eq(((Stmt*)e)->kind, "bool")) {
    if ((((Expr*)e)->ival != 0)) {
    return (const char*)"true";
    } else {
    return (const char*)"false";
    }
    }
    if (str_eq(((Stmt*)e)->kind, "ident")) {
    return (const char*)((Expr*)e)->sval;
    }
    if (str_eq(((Stmt*)e)->kind, "binary")) {
    /* Let left declared=Base(Str) val_ty=Base(Str) */
    const char* left = cgen_expr(prog, exprs, stmts, ((Expr*)e)->left);
    /* Let right declared=Base(Str) val_ty=Base(Str) */
    const char* right = cgen_expr(prog, exprs, stmts, ((Expr*)e)->right);
    return (const char*)str_add(str_add(str_add(str_add(str_add(str_add("(", left), " "), ((Expr*)e)->sval), " "), right), ")");
    }
    if (str_eq(((Stmt*)e)->kind, "unary")) {
    /* Let inner declared=Base(Str) val_ty=Base(Str) */
    const char* inner = cgen_expr(prog, exprs, stmts, ((Expr*)e)->left);
    return (const char*)str_add(str_add(str_add("(", ((Expr*)e)->sval), inner), ")");
    }
    if (str_eq(((Stmt*)e)->kind, "call")) {
    /* Let args declared=Base(Str) val_ty=Base(Str) */
    const char* args = "";
    return (const char*)str_add(str_add(str_add(((Expr*)e)->sval, "("), args), ")");
    }
    if (str_eq(((Stmt*)e)->kind, "field")) {
    /* Let target declared=Base(Str) val_ty=Base(Str) */
    const char* target = cgen_expr(prog, exprs, stmts, ((Expr*)e)->left);
    return (const char*)str_add(str_add(str_add("(", target), ")."), ((Expr*)e)->sval);
    }
    if (str_eq(((Stmt*)e)->kind, "ref")) {
    /* Let inner declared=Base(Str) val_ty=Base(Str) */
    const char* inner = cgen_expr(prog, exprs, stmts, ((Expr*)e)->left);
    return (const char*)str_add("&", inner);
    }
    if (str_eq(((Stmt*)e)->kind, "deref")) {
    /* Let inner declared=Base(Str) val_ty=Base(Str) */
    const char* inner = cgen_expr(prog, exprs, stmts, ((Expr*)e)->left);
    return (const char*)str_add(str_add("(*", inner), ")");
    }
    if (str_eq(((Stmt*)e)->kind, "block")) {
    /* Let out declared=Base(Str) val_ty=Base(Str) */
    const char* out = "({ ";
    /* Let start declared=Unknown val_ty=Unknown */
    int64_t start = ((Expr*)e)->extra_start;
    /* Let count declared=Unknown val_ty=Unknown */
    int64_t count = ((Expr*)e)->extra_count;
    /* Let i declared=Base(I32) val_ty=Base(I32) */
    int32_t i = 0;
    while ((i < count)) {
    /* Let stmt_idx declared=Unknown val_ty=Unknown */
    int64_t stmt_idx = (start + i);
    out = str_add(out, cgen_stmt(prog, exprs, stmts, stmt_idx));
    i = (i + 1);
    0;
    }
    return (const char*)str_add(out, " })");
    }
    if (str_eq(((Stmt*)e)->kind, "if")) {
    /* Let cond declared=Base(Str) val_ty=Base(Str) */
    const char* cond = cgen_expr(prog, exprs, stmts, ((Expr*)e)->left);
    /* Let then_val declared=Base(Str) val_ty=Base(Str) */
    const char* then_val = cgen_expr(prog, exprs, stmts, ((Expr*)e)->right);
    if ((((Expr*)e)->extra_start > 0)) {
    /* Let else_val declared=Base(Str) val_ty=Base(Str) */
    const char* else_val = cgen_expr(prog, exprs, stmts, ((Expr*)e)->extra_start);
    return (const char*)str_add(str_add(str_add(str_add(str_add(str_add("(", cond), " ? "), then_val), " : "), else_val), ")");
    }
    return (const char*)str_add(str_add(str_add(str_add("(", cond), " ? "), then_val), " : 0)");
    }
    if (str_eq(((Stmt*)e)->kind, "assign")) {
    /* Let lhs declared=Base(Str) val_ty=Base(Str) */
    const char* lhs = cgen_expr(prog, exprs, stmts, ((Expr*)e)->left);
    /* Let rhs declared=Base(Str) val_ty=Base(Str) */
    const char* rhs = cgen_expr(prog, exprs, stmts, ((Expr*)e)->right);
    return (const char*)str_add(str_add(lhs, " = "), rhs);
    }
    return (const char*)"0";
}

const char* cgen_stmt(Program* prog, void* exprs, void* stmts, int64_t idx) {
    if (((idx <= 0) || (idx > Vec_len(stmts)))) {
    return (const char*)";\n";
    }
    /* Let s declared=Unknown val_ty=Unknown */
    void* s = Vec_get(stmts, (idx - 1));
    if (str_eq(((Stmt*)s)->kind, "let")) {
    /* Let init declared=Base(Str) val_ty=Base(Str) */
    const char* init = cgen_expr(prog, exprs, stmts, ((Stmt*)s)->expr_idx);
    return (const char*)str_add(str_add(str_add(str_add("int ", ((EnumDef*)s)->name), " = "), init), ";\n");
    }
    if (str_eq(((Stmt*)s)->kind, "return")) {
    if ((((Stmt*)s)->expr_idx > 0)) {
    /* Let val declared=Base(Str) val_ty=Base(Str) */
    const char* val = cgen_expr(prog, exprs, stmts, ((Stmt*)s)->expr_idx);
    return (const char*)str_add(str_add("return ", val), ";\n");
    }
    return (const char*)"return;\n";
    }
    if (str_eq(((Stmt*)s)->kind, "expr")) {
    /* Let val declared=Base(Str) val_ty=Base(Str) */
    const char* val = cgen_expr(prog, exprs, stmts, ((Stmt*)s)->expr_idx);
    return (const char*)str_add(val, ";\n");
    }
    if (str_eq(((Stmt*)s)->kind, "while")) {
    /* Let cond declared=Base(Str) val_ty=Base(Str) */
    const char* cond = cgen_expr(prog, exprs, stmts, ((Stmt*)s)->expr_idx);
    /* Let out declared=Base(Str) val_ty=Base(Str) */
    const char* out = str_add(str_add("while (", cond), ") {\n");
    /* Let i declared=Base(I32) val_ty=Base(I32) */
    int32_t i = 0;
    while ((i < ((FnDef*)s)->body_count)) {
    /* Let stmt_idx declared=Unknown val_ty=Unknown */
    int64_t stmt_idx = (((FnDef*)s)->body_start + i);
    out = str_add(str_add(out, "    "), cgen_stmt(prog, exprs, stmts, stmt_idx));
    i = (i + 1);
    0;
    }
    return (const char*)str_add(out, "}\n");
    }
    if (str_eq(((Stmt*)s)->kind, "loop")) {
    /* Let out declared=Base(Str) val_ty=Base(Str) */
    const char* out = "while (1) {\n";
    /* Let i declared=Base(I32) val_ty=Base(I32) */
    int32_t i = 0;
    while ((i < ((FnDef*)s)->body_count)) {
    /* Let stmt_idx declared=Unknown val_ty=Unknown */
    int64_t stmt_idx = (((FnDef*)s)->body_start + i);
    out = str_add(str_add(out, "    "), cgen_stmt(prog, exprs, stmts, stmt_idx));
    i = (i + 1);
    0;
    }
    return (const char*)str_add(out, "}\n");
    }
    if (str_eq(((Stmt*)s)->kind, "break")) {
    return (const char*)"break;\n";
    }
    if (str_eq(((Stmt*)s)->kind, "continue")) {
    return (const char*)"continue;\n";
    }
    return (const char*)";\n";
}

const char* cgen_struct(StructDef* s) {
    /* Let out declared=Base(Str) val_ty=Base(Str) */
    const char* out = "typedef struct {\n";
    /* Let i declared=Base(I32) val_ty=Base(I32) */
    int32_t i = 0;
    while ((i < Vec_len(s->fields))) {
    /* Let f declared=Unknown val_ty=Unknown */
    void* f = Vec_get(s->fields, i);
    out = str_add(str_add(str_add(out, "    int "), ((EnumDef*)f)->name), ";\n");
    i = (i + 1);
    0;
    }
    out = str_add(str_add(str_add(out, "} "), s->name), ";\n\n");
    return (const char*)out;
}

const char* cgen_param(Param* p) {
    return (const char*)str_add("int ", p->name);
}

const char* cgen_fn(Program* prog, void* exprs, void* stmts, FnDef* f) {
    /* Let out declared=Base(Str) val_ty=Base(Str) */
    const char* out = str_add(str_add("int ", f->name), "(");
    /* Let i declared=Base(I32) val_ty=Base(I32) */
    int32_t i = 0;
    while ((i < Vec_len(f->params))) {
    if ((i > 0)) {
    out = str_add(out, ", ");
    }
    /* Let ref_tmp_0 declared=Unknown val_ty=Unknown */
    void* ref_tmp_0 = Vec_get(f->params, i);
    out = str_add(out, cgen_param(ref_tmp_0));
    i = (i + 1);
    0;
    }
    out = str_add(out, ") {\n");
    i = 0;
    while ((i < f->body_count)) {
    /* Let stmt_idx declared=Base(I64) val_ty=Base(I64) */
    int64_t stmt_idx = (f->body_start + i);
    out = str_add(str_add(out, "    "), cgen_stmt(prog, exprs, stmts, stmt_idx));
    i = (i + 1);
    0;
    }
    out = str_add(out, "}\n\n");
    return (const char*)out;
}

const char* cgen_program(Program* prog, void* exprs, void* stmts) {
    /* Let out declared=Base(Str) val_ty=Base(Str) */
    const char* out = "#include <stdio.h>\n";
    out = str_add(out, "#include <stdint.h>\n");
    out = str_add(out, "#include <stdbool.h>\n");
    out = str_add(out, "#include <stdlib.h>\n");
    out = str_add(out, "#include <string.h>\n");
    out = str_add(out, "#include <math.h>\n\n");
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
    out = str_add(out, cgen_fn(prog, exprs, stmts, ref_tmp_1));
    i = (i + 1);
    0;
    }
    out = str_add(out, "int main(void) {\n");
    i = 0;
    while ((i < prog->main_stmts_count)) {
    /* Let stmt_idx declared=Base(I64) val_ty=Base(I64) */
    int64_t stmt_idx = (prog->main_stmts_start + i);
    out = str_add(str_add(out, "    "), cgen_stmt(prog, exprs, stmts, stmt_idx));
    i = (i + 1);
    0;
    }
    out = str_add(out, "    return 0;\n}\n");
    return (const char*)out;
}

const char* int_to_str(int64_t n) {
    if ((n == 0)) {
    return (const char*)"0";
    }
    /* Let neg declared=Base(Bool) val_ty=Base(Bool) */
    bool neg = false;
    if ((n < 0)) {
    neg = true;
    n = (-n);
    }
    /* Let s declared=Base(Str) val_ty=Base(Str) */
    const char* s = "";
    /* Let m declared=Base(I64) val_ty=Base(I64) */
    int64_t m = n;
    while ((m > 0)) {
    /* Let digit declared=Base(I64) val_ty=Base(I64) */
    int64_t digit = (m % 10);
    /* Let ch declared=Base(Str) val_ty=Base(Str) */
    const char* ch = "";
    if ((digit == 0)) {
    ch = "0";
    } else {
    if ((digit == 1)) {
    ch = "1";
    } else {
    if ((digit == 2)) {
    ch = "2";
    } else {
    if ((digit == 3)) {
    ch = "3";
    } else {
    if ((digit == 4)) {
    ch = "4";
    } else {
    if ((digit == 5)) {
    ch = "5";
    } else {
    if ((digit == 6)) {
    ch = "6";
    } else {
    if ((digit == 7)) {
    ch = "7";
    } else {
    if ((digit == 8)) {
    ch = "8";
    } else {
    if ((digit == 9)) {
    ch = "9";
    }
    }
    }
    }
    }
    }
    }
    }
    }
    }
    s = str_add(ch, s);
    m = (m / 10);
    0;
    }
    if (neg) {
    s = str_add("-", s);
    }
    return (const char*)s;
}

const char* float_to_str(double n) {
    return (const char*)str_add(int_to_str(n), ".0");
}

const char* compile(const char* src) {
    /* Let lex declared=TypeParam { name: "Lexer" } val_ty=TypeParam { name: "Lexer" } */
    Lexer lex = lexer_new(src);
    /* Let tokens declared=Generic { base: TypeParam { name: "Vec" }, args: [TypeParam { name: "Token" }] } val_ty=Generic { base: TypeParam { name: "Vec" }, args: [TypeParam { name: "Token" }] } */
    void* tokens = lexer_tokenize(&lex);
    /* Let prog declared=TypeParam { name: "Program" } val_ty=TypeParam { name: "Program" } */
    Program prog = parse_program(tokens);
    /* Let ref_tmp_0 declared=Generic { base: TypeParam { name: "Vec" }, args: [TypeParam { name: "Expr" }] } val_ty=Generic { base: TypeParam { name: "Vec" }, args: [TypeParam { name: "Expr" }] } */
    void* ref_tmp_0 = (prog).expr_nodes;
    /* Let ref_tmp_1 declared=Generic { base: TypeParam { name: "Vec" }, args: [TypeParam { name: "Stmt" }] } val_ty=Generic { base: TypeParam { name: "Vec" }, args: [TypeParam { name: "Stmt" }] } */
    void* ref_tmp_1 = (prog).stmt_nodes;
    return (const char*)cgen_program(&prog, ref_tmp_0, ref_tmp_1);
}

int main() {
    /* Let src declared=Base(Str) val_ty=Base(Str) */
    const char* src = read_file("tenthc_combined.th");
    /* Let c declared=Base(Str) val_ty=Base(Str) */
    const char* c = compile(src);
    write_file("tenthc_output.c", c);
    return (int){0};
}

