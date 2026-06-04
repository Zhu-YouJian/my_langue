// Tenth runtime library — provides built-in functions for compiled C code
//
// Memory management: uses a global arena for string operations.
// Call str_arena_reset() at the end of main to free all accumulated strings.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>

// === Arena allocator ========================================================
// All string concatenation results live in this arena.
// The arena grows as needed and is freed in one shot by str_arena_reset().

#define ARENA_BLOCK_SIZE (64 * 1024)  // 64 KB per block

typedef struct ArenaBlock {
    char* data;
    size_t used;
    struct ArenaBlock* next;
} ArenaBlock;

static ArenaBlock* g_arena_head = NULL;

static void* arena_alloc(size_t sz) {
    // First allocation or current block full → allocate new block
    if (!g_arena_head || g_arena_head->used + sz > ARENA_BLOCK_SIZE) {
        size_t block_sz = sz > ARENA_BLOCK_SIZE ? sz : ARENA_BLOCK_SIZE;
        ArenaBlock* blk = (ArenaBlock*)malloc(sizeof(ArenaBlock) + block_sz);
        blk->data = (char*)(blk + 1);
        blk->used = 0;
        blk->next = g_arena_head;
        g_arena_head = blk;
    }
    void* ptr = g_arena_head->data + g_arena_head->used;
    g_arena_head->used += sz;
    return ptr;
}

void str_arena_reset(void) {
    while (g_arena_head) {
        ArenaBlock* next = g_arena_head->next;
        free(g_arena_head);
        g_arena_head = next;
    }
}

// === Vec (dynamic array) ====================================================
//
//  Vec struct layout (hidden — users see void*):
//    typedef struct { void** data; size_t len; size_t cap; } Vec;

typedef struct {
    void** data;
    size_t len;
    size_t cap;
} Vec;

void* Vec_new(void) {
    Vec* v = malloc(sizeof(Vec));
    v->data = NULL;
    v->len = 0;
    v->cap = 0;
    return v;
}

void* Vec_push(void* vec, void* item) {
    Vec* v = (Vec*)vec;
    if (v->len >= v->cap) {
        v->cap = v->cap == 0 ? 8 : v->cap * 2;
        v->data = realloc(v->data, v->cap * sizeof(void*));
    }
    v->data[v->len++] = item;
    return vec;
}

int64_t Vec_len(void* vec) {
    if (!vec) return 0;
    Vec* v = (Vec*)vec;
    return (int64_t)v->len;
}

void* Vec_get(void* vec, int64_t idx) {
    Vec* v = (Vec*)vec;
    if (idx < 0 || (size_t)idx >= v->len) return NULL;
    return v->data[idx];
}

void Vec_free(void* vec) {
    if (!vec) return;
    Vec* v = (Vec*)vec;
    free(v->data);
    free(v);
}

// === HashMap ================================================================

void* HashMap_new(void) {
    // Minimal stub — returns a Vec-like structure for "insert"
    return Vec_new();
}

// === I/O ====================================================================

void* read_file(const char* path) {
    FILE* f = fopen(path, "r");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);
    char* buf = malloc(size + 1);
    fread(buf, 1, size, f);
    buf[size] = 0;
    fclose(f);
    return buf;
}

void write_file(const char* path, const char* content) {
    FILE* f = fopen(path, "w");
    if (f) {
        fputs(content, f);
        fclose(f);
    }
}

void println(const char* s) {
    printf("%s\n", s);
}

// === Utilities ==============================================================

// str_add — arena-allocated string concatenation
// All results live in the global arena; call str_arena_reset() to free.
char* str_add(const char* a, const char* b) {
    if (!a && !b) { char* r = (char*)arena_alloc(1); r[0] = 0; return r; }
    if (!a) {
        size_t lb = strlen(b);
        char* r = (char*)arena_alloc(lb + 1);
        memcpy(r, b, lb + 1);
        return r;
    }
    if (!b) {
        size_t la = strlen(a);
        char* r = (char*)arena_alloc(la + 1);
        memcpy(r, a, la + 1);
        return r;
    }
    size_t la = strlen(a), lb = strlen(b);
    char* r = (char*)arena_alloc(la + lb + 1);
    memcpy(r, a, la);
    memcpy(r + la, b, lb);
    r[la + lb] = 0;
    return r;
}

// str_int — int64 to string using a rotating static buffer
// Safe for up to 16 nested calls
static char _str_int_buf[16][32];
static int _str_int_idx = 0;
const char* str_int(int64_t n) {
    int i = (_str_int_idx++) & 15;
    snprintf(_str_int_buf[i], 32, "%lld", (long long)n);
    return _str_int_buf[i];
}

// === String helpers for Tenth compiled code ==================================

// str_eq — string equality
bool str_eq(const char* a, const char* b) {
    if (a == b) return true;
    if (!a || !b) return false;
    return strcmp(a, b) == 0;
}

// str_at — get a 1-char string at position, returns "" if out of bounds
// Uses rotating buffers to survive nested calls (16 slots)
static char _str_at_buf[16][2] = {{0}};
static int _str_at_idx = 0;
const char* str_at(const char* s, int64_t pos) {
    if (!s || pos < 0 || (size_t)pos >= strlen(s)) return "";
    int i = (_str_at_idx++) & 15;
    _str_at_buf[i][0] = s[pos];
    _str_at_buf[i][1] = 0;
    return _str_at_buf[i];
}

// str_to_int — parse string to int64
int64_t str_to_int(const char* s) {
    if (!s) return 0;
    return (int64_t)strtoll(s, NULL, 10);
}
