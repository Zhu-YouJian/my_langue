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

/// General-purpose arena allocation (not limited to strings).
/// All memory returned by tenth_alloc lives until str_arena_reset().
/// Use this instead of malloc() for any object whose lifetime matches
/// the program run (e.g. structs pushed into Vec, file contents).
void* tenth_alloc(size_t sz) {
    return arena_alloc(sz);
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
//
//  Simple string-keyed hash map using separate chaining.
//  Keys and values are arena-allocated (no per-entry free needed).

#define HMAP_BUCKETS 64

typedef struct HMapEntry {
    const char* key;
    void* value;
    struct HMapEntry* next;
} HMapEntry;

typedef struct {
    HMapEntry* buckets[HMAP_BUCKETS];
    size_t len;
} HMap;

static unsigned hmap_hash(const char* s) {
    unsigned h = 5381;
    while (*s) h = ((h << 5) + h) + (unsigned char)*s++;
    return h % HMAP_BUCKETS;
}

void* HashMap_new(void) {
    HMap* m = (HMap*)calloc(1, sizeof(HMap));
    return m;
}

/// Insert or update a key-value pair. Returns the map handle.
void* HashMap_insert(void* map, const char* key, void* value) {
    if (!map || !key) return map;
    HMap* m = (HMap*)map;
    unsigned b = hmap_hash(key);
    HMapEntry* e = m->buckets[b];
    while (e) {
        if (strcmp(e->key, key) == 0) {
            e->value = value;
            return map;
        }
        e = e->next;
    }
    HMapEntry* ne = (HMapEntry*)calloc(1, sizeof(HMapEntry));
    ne->key = key;
    ne->value = value;
    ne->next = m->buckets[b];
    m->buckets[b] = ne;
    m->len++;
    return map;
}

/// Look up a key. Returns NULL if not found.
void* HashMap_get(void* map, const char* key) {
    if (!map || !key) return NULL;
    HMap* m = (HMap*)map;
    HMapEntry* e = m->buckets[hmap_hash(key)];
    while (e) {
        if (strcmp(e->key, key) == 0) return e->value;
        e = e->next;
    }
    return NULL;
}

int64_t HashMap_len(void* map) {
    if (!map) return 0;
    return (int64_t)((HMap*)map)->len;
}

// === I/O ====================================================================

void* read_file(const char* path) {
    FILE* f = fopen(path, "r");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);
    char* buf = (char*)arena_alloc(size + 1);
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

// str_int — int64 to arena-allocated string
// Result lives in the global arena; safe for unlimited nested calls.
const char* str_int(int64_t n) {
    char buf[32];
    int len = snprintf(buf, sizeof(buf), "%lld", (long long)n);
    char* r = (char*)arena_alloc(len + 1);
    memcpy(r, buf, len + 1);
    return r;
}

// === String helpers for Tenth compiled code ==================================

// str_eq — string equality
bool str_eq(const char* a, const char* b) {
    if (a == b) return true;
    if (!a || !b) return false;
    return strcmp(a, b) == 0;
}

// str_at — get a 1-char arena-allocated string at position, returns "" if out of bounds
const char* str_at(const char* s, int64_t pos) {
    if (!s || pos < 0 || (size_t)pos >= strlen(s)) return "";
    char* r = (char*)arena_alloc(2);
    r[0] = s[pos];
    r[1] = 0;
    return r;
}

// str_to_int — parse string to int64
int64_t str_to_int(const char* s) {
    if (!s) return 0;
    return (int64_t)strtoll(s, NULL, 10);
}
