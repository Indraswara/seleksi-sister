#include "controller.h"



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

void GET(int client_socket, const char* params){
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
 * @param body: the body that submitted by client
 * @param content_type: the type of content that submitted by client
 * 1. text/plain
 * 2. application/json
 * 3. application/x-www-form-urlencoded
 * @return void
 */

void POST(int client_socket, const char* body, const char* content_type) {
    char response[MAX] = {0};
    char keys[10][256]; 
    char values[10][256]; 
    int count = 0;

    memset(keys, 0, sizeof(keys));
    memset(values, 0, sizeof(values));
    memset(response, 0, sizeof(response));


    bool is_valid = true;
    if (strcmp(content_type, "text/plain") == 0) {
        parser_text_plain(body, keys, values, &count);
    } else if (strcmp(content_type, "application/json") == 0) {
        parse_JSON(body, keys, values, &count);
    } else if (strcmp(content_type, "application/x-www-form-urlencoded") == 0) {
        parser_url_encoded(body, keys, values, &count);
    } else {
        sprintf(response, "{\"status\": \"error\", \"message\": \"Unsupported content type\"}");
        is_valid = false;
    }

    printf("body: %s\n", body);
    printf("count: %d\n", count);
    for (int i = 0; i < count; i++) {
        printf("key: %s, value: %s\n", keys[i], values[i]);
    }

    //make response if the data is valid
    if(is_valid){
        sprintf(response, "{\"status\": \"submitted\", \"data\": {");
        for (int i = 0; i < count; i++) {
            int pair_length = strlen(keys[i]) + strlen(values[i]) + 6; // 6 for the surrounding quotes and colons
            char* pair = malloc(pair_length * sizeof(char));
            sprintf(pair, "\"%s\": \"%s\"", keys[i], values[i]);
            strcat(response, pair);
            if (i < count - 1) {
                strcat(response, ", ");
            }
        }
        strcat(response, "}}"); 
    }

    send_response(client_socket, "200 OK", "application/json", response);
}

/**
 * same as the POST method but it's used for PUT method
 */
void PUT(int client_socket, const char* body, const char* content_type) {
    char response[MAX] = {0};
    char keys[10][256];
    char values[10][256]; 
    int count = 0;

    if (strcmp(content_type, "text/plain") == 0) {
        sprintf(response, "{\"status\": \"submitted\", \"data\": \"%s\"}", body);
        // parser_text(body, key, value);
    } else if (strcmp(content_type, "application/json") == 0) {
        parse_JSON(body, keys, values, &count);
        sprintf(response, "{\"status\": \"submitted\", \"data\": {");
        for (int i = 0; i < count; i++) {
            int pair_length = strlen(keys[i]) + strlen(values[i]) + 6; // 6 for the surrounding quotes and colons
            char* pair = malloc(pair_length * sizeof(char));
            sprintf(pair, "\"%s\": \"%s\"", keys[i], values[i]);
            strcat(response, pair);
            if (i < count - 1) {
                strcat(response, ", ");
            }
        }
        strcat(response, "}}");
    } else if (strcmp(content_type, "application/x-www-form-urlencoded") == 0) {
        sprintf(response, "{\"status\": \"submitted\", \"data\": \"%s\"}", body);
        // parser_text(body, key, value);
    } else {
        sprintf(response, "{\"status\": \"error\", \"message\": \"Unsupported content type\"}");
    }

    printf("body: %s\n", body);
    for (int i = 0; i < count; i++) {
        printf("key: %s, value: %s\n", keys[i], values[i]);
    }
    send_response(client_socket, "200 OK", "application/json", response);
}

/**
 * function for DELETE method
 * @param client_socket: the client socket (for now 8080)
 * @param keys: the keys that submitted by client
 * @param values: the values that submitted by client
 * @param count: the count of the keys and values
 * @return void
 */
void DELETE(int client_socket, char keys[][256], char values[][256], int* count){
    char response[MAX] = {0};

    if(*count > 0){
        strcat(response, "{\"status\": \"deleted\", \"data\": {");
        for(int i = 0; i < *count; i++){
            int pair_length = strlen(keys[i]) + strlen(values[i]) + 6; // 6 for the surrounding quotes and colons
            char* pair = malloc(pair_length * sizeof(char));
            sprintf(pair, "\"%s\": \"%s\"", keys[i], values[i]);
            strcat(response, pair);
            if(i < *count - 1){
                strcat(response, ", ");
            }
            free(pair);
        }
        strcat(response, "}}");
    } else {
        strcat(response, "{\"status\": \"deleted\", \"data\": {}}");
    }

    send_response(client_socket, "200 OK", "application/json", response);
}