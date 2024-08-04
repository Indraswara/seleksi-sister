#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "route.h"

#define MAX_ROUTES 100

static Route routes[MAX_ROUTES];
static int route_count = 0;

void add_route(const char* method, const char* url, RouteHandler handler) {
    if (route_count < MAX_ROUTES) {
        strncpy(routes[route_count].method, method, sizeof(routes[route_count].method));
        strncpy(routes[route_count].url, url, sizeof(routes[route_count].url));
        routes[route_count].handler = handler;
        route_count++;
    } else {
        fprintf(stderr, "Max route limit reached.\n");
    }
}

void route_request(const char* method, const char* url, int client_socket, const char* body, const char* content_type) {
    for (int i = 0; i < route_count; i++) {
        if (strcmp(routes[i].method, method) == 0 && strcmp(routes[i].url, url) == 0) {
            routes[i].handler(client_socket, body, content_type);
            return;
        }
    }
    const char *response = "HTTP/1.1 404 Not Found\r\nContent-Type: text/plain\r\n\r\nRoute not found";
    send(client_socket, response, strlen(response), 0);
}
