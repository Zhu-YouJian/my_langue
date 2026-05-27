#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

typedef struct {
    int line;
    int col;
} Span;

typedef struct {
    int kind;
    int span;
    int disc;
    int ival;
    int sval;
} Token;

typedef struct {
    int source;
    int pos;
    int line;
    int col;
} Lexer;

int lexer_new(int source) {
    Lexer;
    source;
}

int is_digit(int ch) {
    ((ch >= "0") && (ch <= "9"));
}

int is_alpha(int ch) {
    ((((ch >= "a") && (ch <= "z")) || ((ch >= "A") && (ch <= "Z"))) || (ch == "_"));
}

int is_alnum(int ch) {
    (is_alpha(ch) || is_digit(ch));
}

int is_ws(int ch) {
    ((((ch == " ") || (ch == "\n")) || (ch == "\t")) || (ch == "\r"));
}

int lexer_peek(int lexer) {
    0;
    str;
    (lexer).source;
}

int main(void) {
    value;
    i64;
    0;
    0;
    FloatLiteral(value);
    f64;
    0;
    0;
    StringLiteral(value);
    str;
    0;
    0;
    Identifier(name);
    str;
    0;
    0;
    ;
    Fn;
    0;
    Let;
    0;
    Mut;
    0;
    If;
    0;
    Else;
    0;
    Match;
    0;
    Return;
    0;
    While;
    0;
    For;
    0;
    Loop;
    0;
    Break;
    0;
    Continue;
    0;
    Struct;
    0;
    Enum;
    0;
    Impl;
    0;
    Trait;
    0;
    Use;
    0;
    Mod;
    0;
    True;
    0;
    False;
    0;
    Move;
    0;
    Self_;
    0;
    ;
    Plus;
    0;
    Minus;
    0;
    Star;
    0;
    Slash;
    0;
    Percent;
    0;
    EqEq;
    0;
    NotEq;
    0;
    Lt;
    0;
    Gt;
    0;
    LtEq;
    0;
    GtEq;
    0;
    AndAnd;
    0;
    OrOr;
    0;
    Not;
    0;
    Ampersand;
    0;
    Pipe;
    0;
    ;
    Assign;
    0;
    PlusAssign;
    0;
    MinusAssign;
    0;
    StarAssign;
    0;
    SlashAssign;
    0;
    ;
    LParen;
    0;
    RParen;
    0;
    LBrace;
    0;
    RBrace;
    0;
    LBracket;
    0;
    RBracket;
    0;
    Comma;
    0;
    Semicolon;
    0;
    Colon;
    0;
    Dot;
    0;
    DotDot;
    0;
    Arrow;
    0;
    FatArrow;
    0;
    ColonColon;
    0;
    Eof;
    0;
    0;
    ;
    Lexer;
    return 0;
}
