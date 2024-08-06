#include <iostream>
#include <vector>
#include <omp.h> //include this
#include <fstream>
#include <sstream>

using namespace std;

int main(int argc, char* argv[]){
    if(argc != 2){
        cerr << "Usage: " << argv[0] << " <filename.txt>" << endl;
        return 1;
    }
    ifstream file(argv[1]);
    if(!file.is_open()){
        cerr << "Error opening file: " << argv[1] << endl;
        return 1;
    }
    int N;
    file >> N;
    vector<vector<double>> A(N, vector<double>(N));
    vector<vector<double>> B(N, vector<double>(N));
    vector<vector<double>> C(N, vector<double>(N));
    // Read matrix A
    for(int i = 0; i < N; i++){
        for(int j = 0; j < N; j++){
            file >> A[i][j];
        }
    }
    // Read matrix B
    for(int i = 0; i < N; i++){
        for(int j = 0; j < N; j++){
            file >> B[i][j];
        }
    }
    file.close();

    #pragma omp parallel for collapse(2) //just call this lmao 
    for(int i = 0; i < N; i++){
        for(int j = 0; j < N; j++){
            double sum = 0;
            for(int k = 0; k < N; k++){
                sum += A[i][k] * B[k][j];
            }
            C[i][j] = sum;
        }
    }

    cout << N << endl;
    for(int i = 0; i < N; i++){
        for(int j = 0; j < N; j++){
            cout << C[i][j] << " ";
        }
        cout << endl;
    }

    return 0;
}