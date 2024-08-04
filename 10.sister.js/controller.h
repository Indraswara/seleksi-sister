#ifndef __CONTROLLER__H_
#define __CONTROLLER__H_
#include "common.h"
#include "parser.h"
#include <ctype.h>


//GET METHOD Controller
void GET(int socket, const char* params); 

//POST METHOD Controller
void POST(int client_socket, const char* body, const char* content_type); 

//PUT METHOD Controller 
void PUT(int client_socket, const char* body, const char* content_type); 

//DELETE METHOD Controller
void DELETE(int client_socket, char keys[][256], char values[][256], int* count);

#endif