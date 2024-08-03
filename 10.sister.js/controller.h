#ifndef __CONTROLLER__H_
#define __CONTROLLER__H_

//GET METHOD Controller
void getNilaiAkhir(int socket, const char* params); 

//POST METHOD Controller
void submitNilaiAkhir(int client_socket, const char* body, const char* content_type); 

//PUT METHOD Controller 
void updateNilaiAkhir(int client_socket, const char* body, const char* content_type); 

//DELETE METHOD Controller
void deleteNilaiAkhir(int client_socket);

#endif