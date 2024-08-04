#ifndef __PARSER_H__
#define __PARSER_H__

#include "common.h"
void get_content_type(const char *headers, char* content_type);
/**
 * @parse_params: parse params from url example: /?key1=value1&key2=value2
 * @params_to_pairs: parse params to key-value pairs
 * @parse_request: parse request: method, url, body, headers
 */
void parse_params(const char *url, char* params);
void params_to_pairs(char* params, char keys[][256], char values[][256], int* count);
void parse_request(const char *request, char *method, char *url, char *body, char *headers);

/**
 * @FUNCTION: parser text/plain
 */
char* trim_whitespace(char* str);
void parser_text_plain(const char* body, char keys[][256], char values[][256], int* count);

/**
 * @FUNCTION: parser application/x-www-form-urlencoded
 */
void url_decode(char* src, const char* dest);
void parser_url_encoded(const char* body, char keys[][256], char values[][256], int* count);

/**
 * @FUNCTION: parser application/json
 */
void parse_JSON(const char* body, char keys[][256], char values[][256], int* count);
#endif