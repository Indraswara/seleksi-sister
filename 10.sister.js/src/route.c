#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "../include/route.h"

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

void route_request(int client_socket, HttpRequest* req) {

    bool if_get = strcmp(req->method, "GET") == 0;
    bool if_delete = strcmp(req->method, "DELETE") == 0;



    for (int i = 0; i < route_count; i++){
        if (strcmp(routes[i].method, req->method) == 0 && strcmp(routes[i].url, req->url) == 0) {
            if(if_get){
                routes[i].handler(client_socket, req, NULL, NULL, NULL);
            }
            else if(if_delete){
                char keys[10][256];
                char values[10][256];
                int count = 0;
                parser_url_encoded(req->params, keys, values, &count);
                routes[i].handler(client_socket, req, keys, values, &count);
            }
            else{ //POST or PUT
                routes[i].handler(client_socket, req, NULL, NULL, NULL);
            }
            return;
        }
    }
    const char *response = "HTTP/1.1 404 Not Found\r\nContent-Type: text/plain\r\n\r\nRoute not found";
    send(client_socket, response, strlen(response), 0);
}


// void route_request2(HttpRequest* req, int client_socket) {
//     route_request(client_socket, req);
// }