#include "parser.h"


/**
 * function to parse the body of the request
 * @param body: the body of the request
 * @param key: the key of the body
 * @param value: the value of the body
 * @return void
 */
void parser_JSON(const char* body, char* key, char* value) {
    // Pointers to track the positions of key and value
    const char* key_start;
    const char* key_end;
    const char* value_start;
    const char* value_end;

    // Find the start and end of the key
    key_start = strstr(body, "\"nama\":\"") + strlen("\"nama\":\"");
    key_end = strstr(key_start, "\"");

    // Find the start and end of the value
    value_start = strstr(body, "\"nilai\":\"") + strlen("\"nilai\":\"");
    value_end = strstr(value_start, "\"");

    // Copy the key and value to the provided buffers
    if (key_start && key_end) {
        strncpy(key, key_start, key_end - key_start);
        key[key_end - key_start] = '\0'; // Null-terminate the key string
    }

    if (value_start && value_end) {
        strncpy(value, value_start, value_end - value_start);
        value[value_end - value_start] = '\0'; // Null-terminate the value string
    }
}


void parser_text(const char* params, char* key, char* value) {
    // Find the first '=' in the string
    const char* eq_pos = strchr(params, '=');
    if (eq_pos == NULL) {
        // '=' not found
        return;
    }

    // Find the next '=' after the first '='
    const char* next_eq_pos = strchr(eq_pos + 1, '&');
    if (next_eq_pos == NULL) {
        // No more '=' found
        return;
    }

    // Find the end of the key value
    size_t key_len = next_eq_pos - (eq_pos + 1);
    size_t value_len = strlen(params) - (next_eq_pos - params + 1);

    // Copy the key
    strncpy(key, eq_pos + 1, key_len);
    key[key_len] = '\0'; // Null-terminate the key string

    // Copy the value
    const char *value_start = strchr(next_eq_pos, '=') + 1;
    strncpy(value, value_start, value_len);
    value[value_len] = '\0'; // Null-terminate the value string
}