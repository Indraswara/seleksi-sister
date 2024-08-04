#include "../include/parser.h"

void get_content_type(const char *headers, char* content_type) {
    const char *content_type_pattern = "Content-Type: ";
    char *content_type_start = strstr(headers, content_type_pattern);
    if (content_type_start == NULL) {
        strcpy(content_type, "text/plain");
        return;
    }
    content_type_start += strlen(content_type_pattern);
    char *content_type_end = strstr(content_type_start, "\n");
    if (content_type_end == NULL) {
        strcpy(content_type, content_type_start);
    } else {
        strncpy(content_type, content_type_start, content_type_end - content_type_start);
        content_type[content_type_end - content_type_start] = '\0';
    }
}

/**
 * function to parse the params from the url
 * @param url: the url that submitted by client
 * @param params: the params that submitted by client
 * @return void
 */
void parse_params(const char *url, char* params){
    const char *params_pattern = "?";
    char *params_start = strstr(url, params_pattern);
    if (params_start == NULL) {
        strcpy(params, "");
        return;
    }
    params_start += strlen(params_pattern);
    char *params_end = strstr(params_start, "\n");
    if(params_end == NULL){
        strcpy(params, params_start);
    }else{
        strncpy(params, params_start, params_end - params_start);
        params[params_end - params_start] = '\0';
    }
}

void params_to_pairs(char* params, char keys[][256], char values[][256], int* count){
    parser_url_encoded(params, keys, values, count);     
}


void parse_request(const char *request, char *method, char *url, char *body, char *headers){
    // Create a mutable copy of the request
    char *req_copy = strdup(request);
    char *line = req_copy;

    // Parse the first line to get the method and URL
    sscanf(line, "%s %s", method, url);

    // Initialize headers and body buffers
    headers[0] = '\0';
    body[0] = '\0';

    // Move to the end of the first line
    line = strstr(line, "\r\n") + 2;

    // Determine if we are parsing headers or body
    bool is_body = false;

    while(*line){
        char *next_line = strstr(line, "\r\n");
        if(!next_line) next_line = line + strlen(line);

        if(!is_body){
            if (next_line == line) {
                is_body = true; // Empty line signifies end of headers
            }else{
                // Append header and add newline
                strncat(headers, line, next_line - line);
                strcat(headers, "\n");
            }
        }else{
            // Append body and add newline
            strncat(body, line, next_line - line);
            strcat(body, "\n");
        }

        // Move to the next line
        line = next_line + 2;
    }

    // Remove the trailing newline character from the body, if present
    size_t body_len = strlen(body);
    if (body_len > 0 && body[body_len - 1] == '\n') {
        body[body_len - 1] = '\0';
    }

    // Free the duplicated request copy
    free(req_copy);
}


/**
 * function to parse the body of the request
 * @param str: the string that will be trimmed
 * @return void
 */
char* trim_whitespace(char* str){
    char* end;

    /**
     * Trim leading space
     * jika ada spasi di depan string maka akan di trim
     * contoh: "    indra" -> "indra"
     * jika tidak ada spasi di depan string maka akan di return
     */
    while(isspace((unsigned char)*str)) str++;

    if(*str == 0) return str;

    /**
     * Trim trailing space
     * jika ada spasi di belakang string maka akan di trim
     * contoh: "indra    " -> "indra"
     * jika tidak ada spasi di belakang string maka akan di return
     */
    end = str + strlen(str) - 1;
    while(end > str && isspace((unsigned char)*end)) end--;

    /**
     * Write new null terminator
     * mengatur akhir dari string menjadi null terminator
     */
    *(end + 1) = '\0';
    return str;
}

/**
 * function to parse the body of the request
 * @param body: the body of the request
 * @param keys: the key of the body
 * @param value: the value of the body
 * @param count: the count of the key-value pairs
 * @return void
 * 
 * FORMAT: 
 * key1: value1
 * key2: value2
 * key3: value3
 * ...
 * 
 * harus dipastikan setiap akhir dari keyN: valueN harus ada newline (\n)
 */
void parser_text_plain(const char* body, char keys[][256], char values[][256], int* count){
    *count = 0;
    const char* line_start = body;
    const char* line_end;

    /**
     * loop pada body request untuk mendapatkan key dan value
     * setiap akhir dari keyN: valueN harus ada newline (\n)
     * setiap \n akan dijadikan pemisah key dan value satu dengan key dan value lain
     */
    while((line_end = strchr(line_start, '\n')) != NULL){
        char line[512];
        strncpy(line, line_start, line_end - line_start);
        line[line_end - line_start] = '\0';

        /**
         * mencari posisi dari colon (:) pada line
         * jika ditemukan maka akan dijadikan pemisah antara key dan value
         */
        char* colon_pos = strchr(line, ':');
        if(colon_pos){
            *colon_pos = '\0';
            /**
             *  menghapus whitespace pada key dan value
             *  nama       :    indra (ini nyebelin anjeng)
             *  menjadi
             *  nama:indra
             */
            char* key = trim_whitespace(line);
            char* value = trim_whitespace(colon_pos + 1);

            strncpy(keys[*count], key, 256);
            strncpy(values[*count], value, 256);
            (*count)++;
        }
        /**
         * setiap akhir dari keyN: valueN harus ada newline (\n)
         * sehingga line_start akan di set ke line_end + 1
         */
        line_start = line_end + 1;
    }
}



/**
 * FUNCTION: Parsing www-form-urlencoded
 */

/**
 * function to decode URL-encoded string
 * @param dest: the destination buffer for the decoded string
 * @param src: the source URL-encoded string
 * @return void
 */
void url_decode(char* dest, const char* src) {
    char a, b;
    while (*src) {
        if ((*src == '%') && ((a = src[1]) && (b = src[2])) && (isxdigit(a) && isxdigit(b))) {
            if (a >= 'a') a -= 'a' - 'A';
            if (a >= 'A') a -= ('A' - 10);
            else a -= '0';
            if (b >= 'a') b -= 'a' - 'A';
            if (b >= 'A') b -= ('A' - 10);
            else b -= '0';
            *dest++ = 16 * a + b;
            src += 3;
        } else if (*src == '+') {
            *dest++ = ' ';
            src++;
        } else {
            *dest++ = *src++;
        }
    }
    *dest = '\0';
}


/**
 * function to parse the body of the request
 * @param body: the body of the request
 * @param keys: the key of the body
 * @param values: the value of the body
 * @param count: the count of the key-value pairs
 * @return void
 */
void parser_url_encoded(const char* body, char keys[][256], char values[][256], int* count) {
    *count = 0;
    const char* cursor = body;
    
    while (*cursor && *count < 256) {
        // Find the end of the key
        const char* key_end = cursor;
        while (*key_end && *key_end != '=' && *key_end != '&') key_end++;
        
        if (*key_end == '=') {
            // We found a key-value pair
            size_t key_length = key_end - cursor;
            if (key_length < 256) {
                // Copy and decode the key
                char encoded_key[256];
                strncpy(encoded_key, cursor, key_length);
                encoded_key[key_length] = '\0';
                url_decode(keys[*count], encoded_key);
                
                // Move to the start of the value
                const char* value_start = key_end + 1;
                const char* value_end = value_start;
                while (*value_end && *value_end != '&') value_end++;
                
                size_t value_length = value_end - value_start;
                if (value_length < 256) {
                    // Copy and decode the value
                    char encoded_value[256];
                    strncpy(encoded_value, value_start, value_length);
                    encoded_value[value_length] = '\0';
                    url_decode(values[*count], encoded_value);
                    
                    (*count)++;
                    
                    // Move to the next pair
                    cursor = *value_end ? value_end + 1 : value_end;
                } else {
                    // Value too long, skip this pair
                    cursor = *value_end ? value_end + 1 : value_end;
                }
            } else {
                // Key too long, skip this pair
                cursor = strchr(key_end, '&');
                if (!cursor) break;
                cursor++;
            }
        } else {
            // No '=' found, move to the next '&' or end of string
            cursor = strchr(cursor, '&');
            if (!cursor) break;
            cursor++;
        }
    }
}

/**
 * function to parse the body of the request
 * @param body: the body of the request
 * @param keys: the key of the body
 * @param values: the value of the body
 * @param count: the count of the key-value pairs
 * @return void
 */
void parse_JSON(const char* body, char keys[][256], char values[][256], int* count){
    /**
     * @def key_start: start dari key 
     * @def key_end: end dari key
     * @def value_start: start dari value
     * @def value_end: end dari value
     * @def count: jumlah key-value pairs
     * @def cursor: pointer untuk mengakses body
     */
    const char* key_start;
    const char* key_end;
    const char* value_start;
    const char* value_end;
    *count = 0; 
    const char* cursor = body;

    /**
     * loop pada body request untuk mendapatkan key dan value
     * strchr beguna untuk mencari karakter pertama yang sama dengan parameter kedua
     * jika ditemukan maka akan di increment
     * jika tidak ditemukan maka akan di break
     * 
     */
    cursor = strchr(cursor, '{');
    if (!cursor) return;
    cursor++; 


    /**
     * loop pada body request untuk mendapatkan key dan value
     * 
     */
    while(*cursor && *cursor != '}'){

        /**
         * cari start dari key
         * jika ditemukan maka akan di increment
         * jika tidak ditemukan maka akan di break
         */
        while(*cursor && (*cursor == ' ' || *cursor == '\n' || *cursor == '\r' || *cursor == '\t')) cursor++;
        if(*cursor != '"') break;
        key_start = ++cursor;

        /**
         * cari end dari key
         * jika ditemukan maka akan di increment
         * jika tidak ditemukan maka akan di break
         */
        key_end = strchr(key_start, '"');
        if(!key_end) break;
        cursor = key_end + 1;

        /**
         * mencari color (:) pada body
         * jika ditemukan maka akan di increment
         * jika tidak ditemukan maka akan di break
         */
        while(*cursor && *cursor != ':') cursor++;
        if(*cursor != ':') break;
        cursor++;

        /**
         * cari start dari value
         * jika ditemukan maka akan di increment
         * jika tidak ditemukan maka akan di break
         */
        while(*cursor && (*cursor == ' ' || *cursor == '\n' || *cursor == '\r' || *cursor == '\t')) cursor++;
        if (*cursor != '"') break;
        value_start = ++cursor;


        /**
         * cari end dari value
         * jika ditemukan maka akan di increment
         * jika tidak ditemukan maka akan di break
         */
        value_end = strchr(value_start, '"');
        if(!value_end) break;
        cursor = value_end + 1;

        /**
         * copy key ke keys array
         */
        size_t key_length = key_end - key_start;
        strncpy(keys[*count], key_start, key_length);
        keys[*count][key_length] = '\0'; 

        /**
         * copy value ke values array
         */
        size_t value_length = value_end - value_start;
        strncpy(values[*count], value_start, value_length);
        values[*count][value_length] = '\0'; 

        /**
         * increment count setiap terjadi penambahan key dan value
         */
        (*count)++; 

        /**
         * pindah ke key dan value berikutnya 
         * jika ditemukan maka akan di increment
         */
        while(*cursor && (*cursor == ' ' || *cursor == '\n' || *cursor == '\r' || *cursor == '\t' || *cursor == ',')) cursor++;
    }
}

