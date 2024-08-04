#include "util.h"


void generate_response(char response[MAX], char keys[][256], char values[][256], int count){
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
    strcat(response, "}}");
}