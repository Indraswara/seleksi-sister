#include "controller.h"
#include "common.h"

void send_response(int client_socket, const char* status, const char* content_type, const char* body) {
    char response[MAX] = {0};
    sprintf(response, "HTTP/1.1 %s\r\nContent-Type: %s\r\nContent-Length: %ld\r\n\r\n%s", status, content_type, strlen(body), body);
    send(client_socket, response, strlen(response), 0);
}

void getNilaiAkhir(int client_socket) {
    char body[MAX] = {0};
    sprintf(body, "GET Nilai Akhir");
    send_response(client_socket, "200 OK", "text/plain", body);
}

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

void deleteNilaiAkhir(int client_socket) {
    char response[MAX] = {0};
    sprintf(response, "{\"status\": \"deleted\"}");
    send_response(client_socket, "200 OK", "application/json", response);
}