#ifndef __UTIL_H__
#define __UTIL_H__

#include "common.h"

/**
 * @brief Parse the request and store the keys and values in the respective arrays
 * 
 * @param response: string dari response
 * @param keys: array of keys
 * @param values: array of values
 * @param count: jumlah key-value pairs
 * 
 * jadi ini sebagai generator response untuk di kirim ke client pada method POST dan PUT(?) 
 */
void generate_response(char response[MAX], char keys[][256], char values[][256], int count);


#endif