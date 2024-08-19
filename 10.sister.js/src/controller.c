#include "../include/controller.h"


void parse_body(const char* content_type, const char* body, char keys[][256], char values[][256], int* count) {
    printf("BODY: %s\n", body);
    if (strcmp(content_type, "text/plain") == 0) {
        char *temp = strtok((char *)body, " ");
        parser_text_plain(temp, keys, values, count);
        return; 
    } else if (strcmp(content_type, "application/json") == 0) {
        parse_JSON(body, keys, values, count);
        return;
    } else if (strcmp(content_type, "application/x-www-form-urlencoded") == 0) {
        parser_url_encoded(body, keys, values, count);
        return;
    } else{
        send_response(400, "Bad Request", "application/json", "{\"status\": \"error\", \"message\": \"Unsupported content type\"}");
    }
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
 * function handle the request from the client
 * @param client_socket: the client socket (for now 8080)
 * @param req: the request that submitted by client
 * @param res: the response that will be sent back to the client
 * @return void
 */
void handle_request(int client_socket, HttpRequest* req, HttpResponse* res){
    bool is_valid = true;

    memset(res->response, 0, sizeof(res->response));
    if (req->content_type[0] == '\0' && (strcmp(req->method, "POST") == 0 || strcmp(req->method, "PUT") == 0)) {
        sprintf(res->response, "{\"status\": \"error\", \"message\": \"Content-Type header missing\"}");
        send_response(client_socket, "400 Bad Request", "application/json", res->response);
        return;
    }

    if (strcmp(req->method, "GET") == 0) {
        if (strlen(req->params) != 0) {
            snprintf(res->response, sizeof(res->response), "GET with params: %.2000s", req->params);
        } else {
            snprintf(res->response, sizeof(res->response), "GET Only");
        }
        send_response(client_socket, "200 OK", "text/plain", res->response);
        return;
    }

    if (strcmp(req->method, "POST") == 0 || strcmp(req->method, "PUT") == 0){
        parse_body(req->content_type, req->body, res->keys, res->values, &res->total_data);


        if (!is_valid) {
            strcat(res->status, "Unsupported content type");
            sprintf(res->response, "{\"status\": \"error\", \"message\": \"Unsupported content type\"}");
            send_response(client_socket, "400 Bad Request", "application/json", res->response);
            return;
        }
        printf("method sebelum http: %s\n", req->method);
        
        generate_response_http(req, res);
        send_response(client_socket, "200 OK", "application/json", res->response);
        memset(res->response, 0, sizeof(res->response));
        return;
    }

    if (strcmp(req->method, "DELETE") == 0){
        if (res->total_data > 0) {
            strcat(res->response, "{\"status\": \"deleted\", \"data\": {");
            for (int i = 0; i < res->total_data; i++) {
                int pair_length = strlen(res->keys[i]) + strlen(res->values[i]) + 6;
                char* pair = malloc(pair_length * sizeof(char));
                sprintf(pair, "\"%s\": \"%s\"", res->keys[i], res->values[i]);
                strcat(res->response, pair);
                if (i < res->total_data - 1) {
                    strcat(res->response, ", ");
                }
                free(pair);
            }
            strcat(res->response, "}}");
        } else {
            strcat(res->response, "{\"status\": \"deleted\", \"data\": {}}");
        }
        send_response(client_socket, "200 OK", "application/json", res->response);
        return;
    }

    sprintf(res->response, "{\"status\": \"error\", \"message\": \"Unsupported HTTP method\"}");
    send_response(client_socket, "405 Method Not Allowed", "application/json", res->response);
}

/**
 * function get the nilai akhir from the client
 * @param client_socket: the client socket (for now 8080)
 * @param req: the request that submitted by client
 * @param res: the response that will be sent back to the client
 * @return void
 */

void GET_example(int client_socket, HttpRequest *req, HttpResponse* res) {
    handle_request(client_socket, req, res);
}


/**
 * function for POST method 
 * @param client_socket: the client socket (for now 8080)
 * @param req: the request that submitted by client
 * @param res: the response that will be sent back to the client
 * @return void
 */

void POST_example(int client_socket, HttpRequest* req, HttpResponse* res) {
    handle_request(client_socket, req, res);
}

/**
 * same as the POST method but it's used for PUT method
 */
void PUT_example(int client_socket, HttpRequest *req, HttpResponse* res) {
    handle_request(client_socket, req, res);
}

/**
 * function for DELETE method
 * @param client_socket: the client socket (for now 8080)
 * @param req: the request that submitted by client
 * @param res: the response that will be sent back to the client
 * @return void
 */
void DELETE_example(int client_socket, HttpRequest *req, HttpResponse* res) {
    handle_request(client_socket, req, res);
}