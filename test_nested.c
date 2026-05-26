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

int64_t test_nested_if();
int64_t test_nested_if() {
    /* Let x declared=Base(I32) val_ty=Base(I32) */
    int32_t x = 1;
    /* Let ifv_2 declared=Base(I32) val_ty=Base(I32) */
    int32_t ifv_2 = 0;
    if ((x == 1)) {
    /* Let tmp_1 declared=Base(I32) val_ty=Base(I32) */
    int32_t tmp_1 = ((x == 2) ? 10 : ((x == 3) ? 20 : 30));
    ifv_2 = tmp_1;
    } else {
    ifv_2 = 40;
    }
    /* Let tmp_0 declared=Base(I32) val_ty=Base(I32) */
    int32_t tmp_0 = ifv_2;
    return (int64_t)tmp_0;
}

