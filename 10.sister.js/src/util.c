#include "../include/util.h"


void generate_response_http(HttpRequest* req, HttpResponse* res){

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

    printf("STATUS: %s\n", res->status);
    sprintf(res->response, "{\"status\": \"%s\", \"data\": {", res->status);
    for(int i = 0; i < res->total_data; i++){
        int pair_length = strlen(res->keys[i]) + strlen(res->values[i]) + 6;
        char* pair = malloc(pair_length * sizeof(char));
        sprintf(pair, "\"%s\": \"%s\"", res->keys[i], res->values[i]);
        strcat(res->response, pair);
        if(i < res->total_data - 1){
            strcat(res->response, ", ");
        }
    }
    strcat(res->response, "}}");
}
