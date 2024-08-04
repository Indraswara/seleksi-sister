#ifndef HTTP_H
#define HTTP_H

#define MAX 1024

/**
 * @brief Struct for HTTP Request
 * 
 * method: GET, POST, PUT, DELETE
 * @url: path
 * @body: body of the request
 * @headers: headers of the request
 * @params: parameters of the request
 * @content_type: content type of the request
 */
typedef struct {
    char method[10];
    char url[100];
    char body[MAX];
    char headers[MAX];
    char params[MAX];
    char content_type[100];
} HttpRequest;

/**
 * @brief Struct for HTTP Response
 * 
 * @status: status of the response
 * @response: response of the request
 */
typedef struct {
    char status[50];
    char response[MAX];
} HttpResponse;

#endif // HTTP_H