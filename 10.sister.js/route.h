#ifndef ROUTE_H
#define ROUTE_H
#include "common.h"
#include "parser.h"
#include "http.h"
typedef void (*RouteHandler)(int, HttpRequest*, char keys[][256], char values[][256], int* count);

typedef struct {
    char method[10];
    char url[100];
    RouteHandler handler;
} Route;

void add_route(const char* method, const char* url, RouteHandler handler);
void route_request(int client_socket, HttpRequest* req);

// void route_request2(HttpRequest* req, int client_socket);
#endif
