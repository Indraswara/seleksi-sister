#ifndef ROUTE_H
#define ROUTE_H
#include "common.h"
#include "parser.h"
#include "http.h"

typedef void (*RouteHandler)(int, HttpRequest*, HttpResponse*);

typedef struct {
    char method[10];
    char url[100];
    RouteHandler handler;
} Route;

void add_route(const char* method, const char* url, RouteHandler handler);
void route_request(int client_socket, HttpRequest* req, HttpResponse* res);

// void route_request2(HttpRequest* req, int client_socket);
#endif
