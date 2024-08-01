#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <cuda_runtime.h>
#include "kernel.h"
#include "array.h"
#include <math.h>
using namespace std;

void readMatrixFromFile(const string& filename, vector<double>& matrix_A, vector<double>& matrix_B, int& N) {
    ifstream file(filename);
    if (!file.is_open()) {
        throw runtime_error("Could not open file");
    }
    string line;
    getline(file, line);
    N = stoi(line);
    int SIZE = N * N;
    matrix_A.resize(SIZE);
    matrix_B.resize(SIZE);
    for (int i = 0; i < SIZE; ++i) {
        file >> matrix_A[i];
    }
    for (int i = 0; i < SIZE; ++i) {
        file >> matrix_B[i];
    }
    file.close();
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        cerr << "Usage: " << argv[0] << " <input_file>" << endl;
        return 1;
    }
    string filename = argv[1];
    vector<double> matrix_A, matrix_B, matrix_C;
    int N;
    try {
        readMatrixFromFile(filename, matrix_A, matrix_B, N);
    } catch (const exception& e) {
        cerr << "Error: " << e.what() << endl;
        return 1;
    }
    int SIZE = N * N;
    matrix_C.resize(SIZE);
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
    // Output the result matrix
    cout << N << endl;
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            cout << matrix_C[i * N + j] << " ";
        }
        cout << endl;
    }
    return 0;
}