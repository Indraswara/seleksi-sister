#ifndef __DATA__H_
#define __DATA__H_

#include <stdio.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

typedef struct DataEntry {
    char key[50];
    char value[50];
} DataEntry;

extern DataEntry* data_store; // global dynamic array
extern size_t data_store_size; // current number of elements
extern size_t data_store_capacity; // current capacity

void add_data(const char* key, const char* value);
bool replace_data(const char* key, const char* keyChange, const char* valueChange);
bool delete_data(const char* key, const char* value);

#endif