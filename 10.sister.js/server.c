#include "common.h"
#include "controller.h"

void parse_params(const char *url, char* params){
    const char *params_pattern = "?";
    char *params_start = strstr(url, params_pattern);
    if (params_start == NULL) {
        strcpy(params, "");
        return;
    }
    params_start += strlen(params_pattern);
    char *params_end = strstr(params_start, "\n");
    if (params_end == NULL) {
        strcpy(params, params_start);
    } else {
        strncpy(params, params_start, params_end - params_start);
        params[params_end - params_start] = '\0';
    }
}

void parse_request(const char *request, char *method, char *url, char *body, char *headers) {
    // Create a mutable copy of the request
    char *req_copy = strdup(request);
    char *line = req_copy;

    // Parse the first line to get the method and URL
    sscanf(line, "%s %s", method, url);

    // Initialize headers and body buffers
    headers[0] = '\0';
    body[0] = '\0';

    // Move to the end of the first line
    line = strstr(line, "\r\n") + 2;

    // Determine if we are parsing headers or body
    bool is_body = false;

    while (*line) {
        char *next_line = strstr(line, "\r\n");
        if (!next_line) next_line = line + strlen(line);

        if (!is_body) {
            if (next_line == line) {
                is_body = true; // Empty line signifies end of headers
            } else {
                // Append header and add newline
                strncat(headers, line, next_line - line);
                strcat(headers, "\n");
            }
        } else {
            // Append body and add newline
            strncat(body, line, next_line - line);
            strcat(body, "\n");
        }

        // Move to the next line
        line = next_line + 2;
    }

    // Remove the trailing newline character from the body, if present
    size_t body_len = strlen(body);
    if (body_len > 0 && body[body_len - 1] == '\n') {
        body[body_len - 1] = '\0';
    }

    // Free the duplicated request copy
    free(req_copy);
}


//getting content type
/**
 * 1. text/plain
 * 2. application/json
 * 3. application/x-www-form-urlencoded
 */
void get_content_type(const char *headers, char* content_type) {
    const char *content_type_pattern = "Content-Type: ";
    char *content_type_start = strstr(headers, content_type_pattern);
    if (content_type_start == NULL) {
        strcpy(content_type, "text/plain");
        return;
    }
    content_type_start += strlen(content_type_pattern);
    char *content_type_end = strstr(content_type_start, "\n");
    if (content_type_end == NULL) {
        strcpy(content_type, content_type_start);
    } else {
        strncpy(content_type, content_type_start, content_type_end - content_type_start);
        content_type[content_type_end - content_type_start] = '\0';
    }
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


    while (1) {
        if ((new_socket = accept(server_fd, (SA*)&address, &addrlen)) < 0) {
            perror("accept");
            exit(EXIT_FAILURE);
        }

        //read the request from client
        valueRead = read(new_socket, buffer, MAX);
        printf("Request received:\n%s\n", buffer);


        //before used always empty the buffer
        char method[10], url[100], body[MAX], headers[MAX], content_type[100], params[100];
        memset(method, 0, sizeof(method));
        memset(url, 0, sizeof(url));
        memset(body, 0, sizeof(body));
        memset(headers, 0, sizeof(headers));
        memset(content_type, 0, sizeof(content_type));

        //parsing request
        parse_request(buffer, method, url, body, headers);
        //get the params
        parse_params(url, params);
        //get the content-type
        get_content_type(headers, content_type);

        printf("Method: %s\n", method);
        printf("URL: %s\n", url);
        printf("Body: %s\n", body);
        printf("Headers: %s\n", headers);
        printf("Content-Type: %s\n", content_type);
        printf("Params: %s\n", params);

        // Handle routing GET, POST, PUT, DELETE
        if(strcmp(method, "GET") == 0 && strncmp(url, "/nilai-akhir", 12) == 0) {
            getNilaiAkhir(new_socket, params);
        }else if(strcmp(method, "POST") == 0 && strcmp(url, "/submit") == 0) {
            submitNilaiAkhir(new_socket, body, content_type);
        }else if(strcmp(method, "PUT") == 0 && strcmp(url, "/update") == 0) {
            updateNilaiAkhir(new_socket, body, content_type);
        }else if(strcmp(method, "DELETE") == 0 && strcmp(url, "/delete") == 0) {
            deleteNilaiAkhir(new_socket);
        }else{
            //default handler
            const char *response = "HTTP/1.1 404 Not Found\r\nContent-Type: text/plain\r\n\r\nRoute not found";
            send(new_socket, response, strlen(response), 0);
        }
        close(new_socket);
    }

    close(server_fd);
    return 0;    
}
