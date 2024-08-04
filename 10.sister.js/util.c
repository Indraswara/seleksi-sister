#include "util.h"


void generate_response(char response[MAX], char keys[][256], char values[][256], int count){
    memset(response, 0, MAX * sizeof(char));
    sprintf(response, "{\"status\": \"submitted\", \"data\": {");
    for(int i = 0; i < count; i++){
        int pair_length = strlen(keys[i]) + strlen(values[i]) + 6;
        char* pair = malloc(pair_length * sizeof(char));
        sprintf(pair, "\"%s\": \"%s\"", keys[i], values[i]);
        strcat(response, pair);
        if(i < count - 1){
            strcat(response, ", ");
        }
    }
}

void generate_response_http(HttpRequest* req ,HttpResponse* res, char keys[][256], char values[][256], int count){
    memset(res->response, 0, MAX * sizeof(char));
    memset(res->status, 0, 50 * sizeof(char));

    if(strcmp(req->method, "POST") == 0){
        sprintf(res->status, "Created");
    } else if(strcmp(req->method, "PUT") == 0){
        sprintf(res->status, "Submitted");
    } else if(strcmp(req->method, "DELETE") == 0){
        sprintf(res->status, "Deleted");
    } else if(strcmp(req->method, "GET") == 0){
        sprintf(res->status, "Get");
    } else {
        sprintf(res->status, "400 Bad Request");
    }

    sprintf(res->response, "{\"status\": \"%s\", \"data\": {", res->status);
    for(int i = 0; i < count; i++){
        int pair_length = strlen(keys[i]) + strlen(values[i]) + 6;
        char* pair = malloc(pair_length * sizeof(char));
        sprintf(pair, "\"%s\": \"%s\"", keys[i], values[i]);
        strcat(res->response, pair);
        if(i < count - 1){
            strcat(res->response, ", ");
        }
    }
}