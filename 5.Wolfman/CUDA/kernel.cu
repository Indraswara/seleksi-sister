#include <math.h>
#include <iostream>
#include "cuda_runtime.h"
#include "kernel.h"
#include <stdlib.h>
#include <device_launch_parameters.h>
using namespace std;

__global__ void matrixMultiplicationKernel(double* A, double* B, double* C, int N) {
    int ROW = blockIdx.y * blockDim.y + threadIdx.y;
    int COL = blockIdx.x * blockDim.x + threadIdx.x;
    double tmpSum = 0;
    if (ROW < N && COL < N) {
        // each thread computes one element of the block sub-matrix
        for (int i = 0; i < N; i++) {
            tmpSum += A[ROW * N + i] * B[i * N + COL];
        }
        C[ROW * N + COL] = tmpSum;
    }
}

void matrixMultiplication(double *A, double *B, double *C, int N) {
    // declare the number of blocks per grid and the number of threads per block
    int threadsPerBlock = 32; // 32x32 = 1024 threads per block
    dim3 blockDim(threadsPerBlock, threadsPerBlock);
    dim3 gridDim((N + threadsPerBlock - 1) / threadsPerBlock, (N + threadsPerBlock - 1) / threadsPerBlock);
    matrixMultiplicationKernel<<<gridDim, blockDim>>>(A, B, C, N);
}