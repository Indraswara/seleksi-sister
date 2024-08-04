#ifndef ROUTE_H
#define ROUTE_H
#include "common.h"
#include "parser.h"
typedef void (*RouteHandler)(int, const char*, const char*, char keys[][256], char values[][256], int* count);

typedef struct {
    char method[10];
    char url[100];
    RouteHandler handler;
} Route;

void add_route(const char* method, const char* url, RouteHandler handler);
void route_request(const char* method, const char* url, int client_socket, const char* body, const char* content_type);

#endif
