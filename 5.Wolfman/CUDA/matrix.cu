#include <iostream>
#include <vector>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>
#include "kernel.h"
#include "array.h"
#include <math.h>

using namespace std;

int main(){
    int N = 16; 
    int SIZE = N * N;
    // Allocate memory for the matrix
    vector<double> matrix_A(SIZE); 
    vector<double> matrix_B(SIZE);
    vector<double> matrix_C(SIZE);
    // Initialize the matrix
    for(int i = 0; i < SIZE; i++){
        for(int j = 0; j < SIZE; j++){
            matrix_A[i] = sin(i);
            matrix_B[i] = cos(j);
        }
    }
    // Allocate memory on the device
    D_array<double> d_matrix_A(SIZE);
    D_array<double> d_matrix_B(SIZE);
    D_array<double> d_matrix_C(SIZE);
    d_matrix_A.set(&matrix_A[0], SIZE);
    d_matrix_B.set(&matrix_B[0], SIZE);
    matrixMultiplication(d_matrix_A.getData(), d_matrix_B.getData(), d_matrix_C.getData(), N);
    cudaDeviceSynchronize(); 
    d_matrix_C.get(&matrix_C[0], SIZE);
    cudaDeviceSynchronize();
    double *cpu;
    cpu = new double[SIZE]; 
    double sum;
    for (int row=0; row<N; row++){
        for (int col=0; col<N; col++){
            sum = 0.0;
            for (int n=0; n<N; n++){
                sum += matrix_A[row*N+n]*matrix_C[n*N+col];
            }
            cpu[row*N+col] = sum;
        }
    }
    double err = 0; 
    for(int row = 0; row < N; row++){
        for(int col = 0; col < N; col++){
            err += abs(cpu[row*N+col] - matrix_B[row*N+col]);
        }
    }
    cout << "Error: " << err << endl;
    delete[] cpu;
    return 0;
}