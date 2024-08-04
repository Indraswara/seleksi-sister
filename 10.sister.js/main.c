#include "server.h"

void random_handler(int client_socket, const char* body, const char* content_type) {
    char response[MAX] = "Random Handler";
    send_response(client_socket, "200 OK", "text/plain", response);
}


int main(int argc, char const* argv[]) {
    
    add_route("GET", "/random", (void*) random_handler);
    start_server();
    return 0;
}