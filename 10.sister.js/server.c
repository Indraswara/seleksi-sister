#include "common.h"
#include "controller.h"
#include "route.h"

void custom_get_handler(int client_socket, const char* body, const char* content_type) {
    char response[MAX] = "Custom GET Handler";
    send_response(client_socket, "200 OK", "text/plain", response);
}

int main(int argc, char const* argv[]) {
    int server_fd, new_socket; 
    ssize_t valueRead; 
    struct sockaddr_in address;
    int opt = 1; 
    socklen_t addrlen = sizeof(address);
    char buffer[MAX] = {0};

    // Creating socket file descriptor
    if ((server_fd = socket(AF_INET, SOCK_STREAM, 0)) == 0) {
        perror("socket failed");
        exit(EXIT_FAILURE);
    }

    // Forcefully attaching socket to the port 8080
    if (setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR | SO_REUSEPORT, &opt, sizeof(opt))) {
        perror("setsockopt");
        exit(EXIT_FAILURE);
    }

    address.sin_family = AF_INET;
    address.sin_addr.s_addr = INADDR_ANY;
    address.sin_port = htons(PORT);

    //attach socket to the port 8080
    if (bind(server_fd, (SA*)&address, sizeof(address)) < 0) {
        perror("bind failed");
        exit(EXIT_FAILURE);
    }

    //listen to the port 8080
    if (listen(server_fd, 3) < 0) {
        perror("listen");
        exit(EXIT_FAILURE);
    }

    // Add default routes
    add_route("GET", "/nilai-akhir", GET);
    add_route("POST", "/submit", POST);
    add_route("PUT", "/update", PUT);
    add_route("DELETE", "/delete", DELETE);

    // Add custom routes

    while(1){
        if ((new_socket = accept(server_fd, (SA*)&address, &addrlen)) < 0) {
            perror("accept");
            exit(EXIT_FAILURE);
        }

        //read the request from client
        valueRead = read(new_socket, buffer, MAX);
        printf("Request received:\n%s\n", buffer);

        //before used always empty the buffer
        char method[10], url[100], body[MAX], headers[MAX], content_type[100];
        memset(method, 0, sizeof(method));
        memset(url, 0, sizeof(url));
        memset(body, 0, sizeof(body));
        memset(headers, 0, sizeof(headers));
        memset(content_type, 0, sizeof(content_type));

        //parse the request
        parse_request(buffer, method, url, body, headers);
        get_content_type(headers, content_type);

        printf("Method: %s\n", method);
        printf("URL: %s\n", url);
        printf("Body: %s\n", body);
        printf("Headers: %s\n", headers);
        printf("Content-Type: %s\n", content_type);

        // Route the request
        route_request(method, url, new_socket, body, content_type);

        close(new_socket);
    }
    close(server_fd);
    return 0;    
}
