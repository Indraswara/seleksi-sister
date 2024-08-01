#define CL_TARGET_OPENCL_VERSION 300

#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <CL/cl.h>
#include <stdbool.h>

void printMatrix(double* matrix, int size) {
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            printf("%f ", matrix[i * size + j]);
        }
        printf("\n");
    }
}

long LoadOpenCLKernel(const char *path, char **buf) {
    FILE *fp = fopen(path, "r");
    if (!fp) return -1;
    fseek(fp, 0, SEEK_END);
    long size = ftell(fp);
    rewind(fp);
    *buf = (char *)malloc(size + 1);
    fread(*buf, 1, size, fp);
    (*buf)[size] = '\0';
    fclose(fp);
    return size;
}

int main(int argc, char** argv) {
    if (argc != 2) {
        printf("Usage: %s <filename.txt>\n", argv[0]);
        return EXIT_FAILURE;
    }
    const char* filename = argv[1];
    FILE* file = fopen(filename, "r");
    if (!file) {
        printf("Error: Could not open file %s\n", filename);
        return EXIT_FAILURE;
    }
    int size;
    fscanf(file, "%d", &size);
    unsigned int size_A = size * size;
    unsigned int size_B = size * size;
    unsigned int size_C = size * size;
    double* h_A = (double*) malloc(sizeof(double) * size_A);
    double* h_B = (double*) malloc(sizeof(double) * size_B);
    double* h_C = (double*) malloc(sizeof(double) * size_C);
    for (int i = 0; i < size_A; i++) {
        fscanf(file, "%lf", &h_A[i]);
    }
    for (int i = 0; i < size_B; i++) {
        fscanf(file, "%lf", &h_B[i]);
    }
    fclose(file);
    int err;
    cl_device_id device_id;
    cl_context context;
    cl_command_queue commands;
    cl_program program;
    cl_kernel kernel;
    cl_mem d_A, d_B, d_C;
    srand(2014);
    printf("Initializing OpenCL device...\n");
    cl_uint dev_cnt = 0;
    clGetPlatformIDs(0, 0, &dev_cnt);
    cl_platform_id platform_ids[100];
    clGetPlatformIDs(dev_cnt, platform_ids, NULL);
    int gpu = 1;
    err = clGetDeviceIDs(platform_ids[0], gpu ? CL_DEVICE_TYPE_GPU : CL_DEVICE_TYPE_CPU, 1, &device_id, NULL);
    if (err != CL_SUCCESS) {
        printf("Error: Failed to create a device group!\n");
        return EXIT_FAILURE;
    }
    context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
    if (!context) {
        printf("Error: Failed to create a compute context!\n");
        return EXIT_FAILURE;
    }
    cl_queue_properties properties[] = {CL_QUEUE_PROPERTIES, 0, 0};
    commands = clCreateCommandQueueWithProperties(context, device_id, properties, &err);
    if (!commands) {
        printf("Error: Failed to create a command commands!\n");
        return EXIT_FAILURE;
    }
    char *KernelSource;
    long lFileSize;
    lFileSize = LoadOpenCLKernel("kernel.cl", &KernelSource);
    if (lFileSize < 0L) {
        perror("File read failed");
        return 1;
    }
    program = clCreateProgramWithSource(context, 1, (const char **) &KernelSource, NULL, &err);
    if (!program) {
        printf("Error: Failed to create compute program!\n");
        return EXIT_FAILURE;
    }
    err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        size_t len;
        char buffer[2048];
        printf("Error: Failed to build program executable!\n");
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
        printf("%s\n", buffer);
        exit(1);
    }
    kernel = clCreateKernel(program, "matrixMul", &err);
    if (!kernel || err != CL_SUCCESS) {
        printf("Error: Failed to create compute kernel!\n");
        exit(1);
    }
    d_A = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(double) * size_A, h_A, &err);
    d_B = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(double) * size_B, h_B, &err);
    d_C = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(double) * size_C, NULL, &err);
    if (!d_A || !d_B || !d_C) {
        printf("Error: Failed to allocate device memory!\n");
        exit(1);
    }
    printf("Running matrix multiplication for matrices A and B of size %dx%d ...\n", size, size);
    size_t localWorkSize[2] = {16, 16};
    size_t globalWorkSize[2] = {size, size};
    int wA = size;
    int wC = size;
    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&d_C);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&d_A);
    err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&d_B);
    err |= clSetKernelArg(kernel, 3, sizeof(int), (void *)&wA);
    err |= clSetKernelArg(kernel, 4, sizeof(int), (void *)&wC);
    if (err != CL_SUCCESS) {
        printf("Error: Failed to set kernel arguments! %d\n", err);
        exit(1);
    }
    err = clEnqueueNDRangeKernel(commands, kernel, 2, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        printf("Error: Failed to execute kernel! %d\n", err);
        exit(1);
    }
    err = clEnqueueReadBuffer(commands, d_C, CL_TRUE, 0, sizeof(double) * size_C, h_C, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        printf("Error: Failed to read output array! %d\n", err);
        exit(1);
    }
    printf("\nMatrix C (Results):\n");
    printMatrix(h_C, size);
    printf("Matrix multiplication completed...\n");
    free(h_A);
    free(h_B);
    free(h_C);
    clReleaseMemObject(d_A);
    clReleaseMemObject(d_C);
    clReleaseMemObject(d_B);
    clReleaseProgram(program);
    clReleaseKernel(kernel);
    clReleaseCommandQueue(commands);
    clReleaseContext(context);
    return 0;
}