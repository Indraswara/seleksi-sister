#ifndef HTTP_H
#define HTTP_H

#define MAX 1024

typedef struct {
    char method[10];
    char url[100];
    char body[MAX];
    char headers[MAX];
    char params[MAX];
    char content_type[100];
} HttpRequest;

typedef struct {
    char status[50];
    char response[MAX];
} HttpResponse;

#endif // HTTP_H