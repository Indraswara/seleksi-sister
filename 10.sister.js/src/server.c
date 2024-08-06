#include "../include/server.h"



void custom_get_handler(int client_socket, const char* body, const char* content_type) {
    char response[MAX] = "Custom GET Handler";
    send_response(client_socket, "200 OK", "text/plain", response);
}

void start_server() {
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

    // Add default routes for testing
    add_route("GET", "/nilai-akhir", (void*) GET_example);
    add_route("POST", "/submit", (void *) POST_example);
    add_route("PUT", "/update", (void*) PUT_example);
    add_route("DELETE", "/delete", (void *)DELETE_example);

    while(1){
        if ((new_socket = accept(server_fd, (SA*)&address, &addrlen)) < 0) {
            perror("accept");
            exit(EXIT_FAILURE);
        }

        //read the request from client
        valueRead = read(new_socket, buffer, MAX);
        printf("Request received:\n%s\n", buffer);

        //before used always empty the buffer
        HttpRequest request = {0}; 
        HttpResponse response = {0};
        memset(&request, 0, sizeof(HttpRequest));
        memset(&response, 0, sizeof(HttpResponse));



        //parse the request
        // parse_request(buffer, method, url, body, headers);
        /**
         * khusus HttpRequest
         */
        parse_request(buffer, request.method, request.url, request.body, request.headers);
        // printf("METHOD: %s\n", request.method);
        get_content_type(request.headers, request.content_type);
        parse_params(request.url, request.params);

        char* temp = strtok(request.url, "?");
        strcpy(request.url, temp);

        printf("==============================================\n");
        printf("Method: %s\n", request.method);
        printf("URL: %s\n", request.url);
        printf("Body: %s\n", request.body);
        printf("Headers: %s\n", request.headers);
        printf("Content-Type: %s\n", request.content_type);

        // Route the request
        route_request(new_socket, &request, &response);
        close(new_socket);
    }
    close(server_fd);
}
