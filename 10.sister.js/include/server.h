#ifndef SERVER_H
#define SERVER_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <arpa/inet.h>
#include "common.h"
#include "controller.h"
#include "route.h"
#include "util.h"
#include "http.h"

#define PORT 8080
#define SA struct sockaddr
void custom_get_handler(int client_socket, const char* body, const char* content_type);
void start_server();

#endif // SERVER_H