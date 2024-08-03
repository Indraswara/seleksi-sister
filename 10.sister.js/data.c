#include "data.h"

DataEntry* data_store = NULL;
size_t data_store_size = 0;
size_t data_store_capacity = 0;

void add_data(const char* key, const char* value) {
    if (data_store_size == data_store_capacity) {
        data_store_capacity = data_store_capacity == 0 ? 1 : data_store_capacity * 2;
        data_store = realloc(data_store, data_store_capacity * sizeof(DataEntry));
    }
    strcpy(data_store[data_store_size].key, key);
    strcpy(data_store[data_store_size].value, value);
    data_store_size++;
}

bool replace_data(const char* key, const char* keyChange, const char* valueChange) {
    for (size_t i = 0; i < data_store_size; i++) {
        if (strcmp(data_store[i].key, key) == 0) {
            strcpy(data_store[i].key, keyChange);
            strcpy(data_store[i].value, valueChange);
            return true;
        }
    }
    return false;
}

bool delete_data(const char* key, const char* value) {
    for (size_t i = 0; i < data_store_size; i++) {
        if (strcmp(data_store[i].key, key) == 0) {
            if (strcmp(data_store[i].value, value) != 0) {
                return false;
            }
            
            for (size_t j = i; j < data_store_size - 1; j++) {
                strcpy(data_store[j].key, data_store[j + 1].key);
                strcpy(data_store[j].value, data_store[j + 1].value);
            }
            data_store_size--;
            return true;
        }
    }
    return false;
}