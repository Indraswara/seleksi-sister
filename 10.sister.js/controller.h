#ifndef __CONTROLLER__H_
#define __CONTROLLER__H_
#include "common.h"
#include "parser.h"
#include "util.h"
#include <ctype.h>

/**
 * DEFAULT 
 */
//GET METHOD Controller
void send_response(int client_socket, const char* status, const char* content_type, const char* body);

void GET_example(int socket, const char* params); 

//POST METHOD Controller
void POST_example(int client_socket, const char* body, const char* content_type); 

//PUT METHOD Controller 
void PUT_example(int client_socket, const char* body, const char* content_type); 

//DELETE METHOD Controller
void DELETE_example(int client_socket, const char* body, const char* content_type, char keys[][256], char values[][256], int* count);

#endif