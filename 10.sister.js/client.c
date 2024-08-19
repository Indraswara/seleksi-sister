#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <arpa/inet.h>

#define PORT 8080
#define MAX 1024

void send_request(const char *request){
    int sock = 0;
    struct sockaddr_in serv_addr;
    char buffer[MAX] = {0};

    if((sock = socket(AF_INET, SOCK_STREAM, 0)) < 0){
        printf("\n Socket creation error \n");
        return;
    }

    serv_addr.sin_family = AF_INET;
    serv_addr.sin_port = htons(PORT);

    if(inet_pton(AF_INET, "127.0.0.1", &serv_addr.sin_addr) <= 0){
        printf("\nInvalid address/ Address not supported \n");
        return;
    }

    if(connect(sock, (struct sockaddr *)&serv_addr, sizeof(serv_addr)) < 0){
        printf("\nConnection Failed \n");
        return;
    }

    send(sock, request, strlen(request), 0);
    printf("Request sent:\n%s\n", request);
    read(sock, buffer, MAX);
    printf("Response received:\n%s\n", buffer);

    close(sock);
}

int main(){
    // Example GET request
    const char *get_request = "GET /nilai-akhir HTTP/1.1\r\nHost: localhost\r\n\r\n";
    send_request(get_request);

    // Example POST request
    const char *post_request = "POST /submit HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: 18\r\n\r\n{\"bjir\":\"okeh\"}";
    send_request(post_request);
    return 0;
}