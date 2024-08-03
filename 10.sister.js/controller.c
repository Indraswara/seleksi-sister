#include "controller.h"
#include "common.h"


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
 * @return void
 */
void getNilaiAkhir(int client_socket){
    char body[MAX] = {0};
    sprintf(body, "GET Nilai Akhir");
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
    send_response(client_socket, "200 OK", "application/json", response);
}

/**
 * function for DELETE method
 * @param client_socket: the client socket (for now 8080)
 */
void deleteNilaiAkhir(int client_socket) {
    char response[MAX] = {0};
    sprintf(response, "{\"status\": \"deleted\"}");
    send_response(client_socket, "200 OK", "application/json", response);
}