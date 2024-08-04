#include "server.h"

void random_handler(int client_socket, const char* body, const char* content_type) {
    char response[MAX] = "Random Handler";
    send_response(client_socket, "200 OK", "text/plain", response);
}

/** 
 * example
 */
void submit_handler(int client_socket, const char* body, const char* content_type) {
    char response[MAX] = "Submit Handler";

    char keys[10][256];
    char values[10][256];
    int count = 0;

    memset(keys, 0, sizeof(keys));
    memset(values, 0, sizeof(values));
    memset(response, 0, sizeof(response));

    bool isValid = true;
    if(strcmp(content_type, "application/x-www-form-urlencoded") == 0){
        parser_url_encoded(body, keys, values, &count);
    }
    else if(strcmp(content_type, "application/json") == 0){
        parse_JSON(body, keys, values, &count);
    }
    else if(strcmp(content_type, "text/plain") == 0){
        parser_text_plain(body, keys, values, &count);
    }
    else{
        sprintf(response, "{\"status\": \"error\", \"message\": \"Unsupported content type\"}");
        isValid = false;
    }


    for (int i = 0; i < count; i++) {
        if (strcmp(keys[i], "name") == 0) {
            sprintf(response, "Hello, %s", values[i]);
        }
    }

    if(isValid){
        generate_response(response, keys, values, count);
        send_response(client_socket, "200 OK", "application/json", response);
    }
    else{
        send_response(client_socket, "400 Bad Request", "application/json", response);
    }
}

int main(int argc, char const* argv[]) {
    
    add_route("GET", "/random", (void*) random_handler);
    add_route("POST", "/sum", (void*) submit_handler);
    start_server();
    return 0;
}