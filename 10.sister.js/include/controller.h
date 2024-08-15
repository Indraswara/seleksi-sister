#ifndef __CONTROLLER__H_
#define __CONTROLLER__H_
#include "common.h"
#include "parser.h"
#include "util.h"
#include <ctype.h>
#include "http.h"

/**
 * DEFAULT 
 */
//GET METHOD Controller
void send_response(int client_socket, const char* status, const char* content_type, const char* body);

void handle_request(int client_socket, HttpRequest* req, HttpResponse* res);

void GET_example(int socket, HttpRequest* req, HttpResponse* res); 

//POST METHOD Controller
void POST_example(int client_socket, HttpRequest* req, HttpResponse* res); 

//PUT METHOD Controller 
void PUT_example(int client_socket, HttpRequest* req, HttpResponse* res); 

//DELETE METHOD Controller
void DELETE_example(int client_socket, HttpRequest* req, HttpResponse* res);

#endif