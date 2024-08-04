#ifndef ROUTE_H
#define ROUTE_H

typedef void (*RouteHandler)(int, const char*, const char*);

typedef struct {
    char method[10];
    char url[100];
    RouteHandler handler;
} Route;

void add_route(const char* method, const char* url, RouteHandler handler);
void route_request(const char* method, const char* url, int client_socket, const char* body, const char* content_type);

#endif
