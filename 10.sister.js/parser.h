#ifndef __PARSER_H__
#define __PARSER_H__

#include "common.h"

void parser_JSON(const char* body, char* key, char* value);
void parser_text(const char* params, char* key, char* value);
#endif