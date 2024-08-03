#ifndef __COMMON__H_
#define __COMMON__H_

#include <ctype.h>
#include <netinet/in.h>
#include <stdbool.h>    
#include <regex.h> 
#include <sys/stat.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>
#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <asm-generic/socket.h>
#include <string.h>
#include <arpa/inet.h>

#define MAX 1024
#define PORT 8080 
#define SA struct sockaddr

#endif
