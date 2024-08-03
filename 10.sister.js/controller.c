#include "controller.h"
#include "common.h"
#include "data.h"
#include <ctype.h>


/**
 * function to parse the body of the request
 * @param body: the body of the request
 * @param key: the key of the body
 * @param value: the value of the body
 * @return void
 */
void parser_body(const char* body, char* key, char* value) {
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


void parser_params(const char* params, char* key, char* value) {
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

/**
 * function send response back to the client it's shown in the terminal for now 
 * @param client_socket: the client socket
 * @param status: the status of the response
 * @param content_type: the type of content that submitted by client
 * 1. text/plain
 * 2. application/json
 * 3. application/x-www-form-urlencoded
 * @param body: the response body (data that submitted by client )
 * @return void
 */
void send_response(int client_socket, const char* status, const char* content_type, const char* body) {
    char response[MAX] = {0};
    sprintf(response, "HTTP/1.1 %s\r\nContent-Type: %s\r\nContent-Length: %ld\r\n\r\n%s", status, content_type, strlen(body), body);
    send(client_socket, response, strlen(response), 0);
}

/**
 * function get the nilai akhir from the client
 * @param client_socket: the client socket (for now 8080)
 * @param params: the params that submitted by client
 * @return void
 */

void getNilaiAkhir(int client_socket, const char* params){
    char body[MAX] = {0};
    if (strlen(params) > 0) {
        sprintf(body, "GET Nilai Akhir with params: %s", params);
    } else {
        sprintf(body, "GET Nilai Akhir");
    }
    send_response(client_socket, "200 OK", "text/plain", body);
}


/**
 * function for POST method 
 * @param client_socket: the client socket (for now 8080)
 * @param body: the response body (data that submitted by client)
 * @param content_type: the type of content that submitted by client
 * if it's used in raw format, it can be in 3 types and maybe each of them look like this
 * 1. text/plain
 * e.g: "nama=Joni&nilai=90"
 * 2. application/json
 * e.g:
 * {
 *  "nama": "Joni",
 *  "nilai": 90
 * } 
 * 3. application/x-www-form-urlencoded
 * e.g: "nama=Joni&nilai=90"
 */

void submitNilaiAkhir(int client_socket, const char* body, const char* content_type) {
    char response[MAX] = {0};
    if (strcmp(content_type, "text/plain") == 0) {
        sprintf(response, "{\"status\": \"submitted\", \"data\": \"%s\"}", body);
    } else if (strcmp(content_type, "application/json") == 0) {
        sprintf(response, "{\"status\": \"submitted\", \"data\": %s}", body);
    } else if (strcmp(content_type, "application/x-www-form-urlencoded") == 0) {
        sprintf(response, "{\"status\": \"submitted\", \"data\": \"%s\"}", body);
    } else {
        sprintf(response, "{\"status\": \"error\", \"message\": \"Unsupported content type\"}");
    }
    char key[50] = {0};
    char value[50] = {0};
    parser_body(body, key, value);
    printf("body: %s\n", body);
    printf("key: %s\n", key);
    printf("value: %s\n", value);
    add_data(key, value);
    send_response(client_socket, "200 OK", "application/json", response);
}

/**
 * same as the POST method but it's used for PUT method
 */
void updateNilaiAkhir(int client_socket, const char* body, const char* content_type) {
    char response[MAX] = {0};
    if (strcmp(content_type, "text/plain") == 0) {
        sprintf(response, "{\"status\": \"updated\", \"data\": \"%s\"}", body);
    } else if (strcmp(content_type, "application/json") == 0) {
        sprintf(response, "{\"status\": \"updated\", \"data\": %s}", body);
    } else if (strcmp(content_type, "application/x-www-form-urlencoded") == 0) {
        sprintf(response, "{\"status\": \"updated\", \"data\": \"%s\"}", body);
    } else {
        sprintf(response, "{\"status\": \"error\", \"message\": \"Unsupported content type\"}");
    }
    char key[50] = {0};
    char value[50] = {0};
    parser_body(body, key, value);
    add_data(key, value);
    send_response(client_socket, "200 OK", "application/json", response);
}

/**
 * function for DELETE method
 * @param client_socket: the client socket (for now 8080)
 */
void deleteNilaiAkhir(int client_socket, const char* params){
    char response[2048] = {0};

    char key[50] = {0};
    char value[50] = {0};
    parser_params(params, key, value);

    printf("key: %s\n", key);
    printf("value: %s\n", value);
    if (delete_data(key, value)) {
        sprintf(response, "{\"status\": \"deleted\", \"data\": \"%s\"}", params);
    } else {
        sprintf(response, "{\"status\": \"error\", \"message\": \"Data not found\"}");
    }
    
    
    send_response(client_socket, "200 OK", "application/json", response);
}