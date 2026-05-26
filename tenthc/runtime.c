// Tenth runtime library — provides built-in functions for compiled C code

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Vec_new — allocate a new empty Vec
void* Vec_new(void) {
    // Return NULL for now — Vec operations not yet implemented
    return NULL;
}

// HashMap_new — allocate a new empty HashMap
void* HashMap_new(void) {
    return NULL;
}

// read_file — read entire file into a malloc'd string
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

// write_file — write string to file
void write_file(const char* path, const char* content) {
    FILE* f = fopen(path, "w");
    if (f) {
        fputs(content, f);
        fclose(f);
    }
}

// println — print a string followed by newline
void println(const char* s) {
    printf("%s\n", s);
}
