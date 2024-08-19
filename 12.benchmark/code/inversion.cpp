#include <iostream>
#include <vector>
using namespace std;

vector<vector<double>> transposeMatrix(const vector<vector<double>>& arr, int size) {
    vector<vector<double>> transposed(size, vector<double>(size));

    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            transposed[j][i] = arr[i][j];
        }
    }

    return transposed;
}

double getMatrixDeterminant(const vector<vector<double>>& arr, int n) {
    if (arr.empty()) return -1;
    if (n == 1) return arr[0][0];

    double det = 0;
    int factor = 1;

    for (int i = 0; i < n; i++) {
        vector<vector<double>> newm(n - 1, vector<double>(n - 1));
        for (int j = 1; j < n; j++) {
            int colIndex = 0;
            for (int k = 0; k < n; k++) {
                if (k == i) continue;
                newm[j - 1][colIndex++] = arr[j][k];
            }
        }
        det += factor * arr[0][i] * getMatrixDeterminant(newm, n - 1);
        factor *= -1;
    }

    return det;
}

vector<vector<double>> getMatrixMinor(const vector<vector<double>>& arr, int size, int row, int col) {
    vector<vector<double>> minor(size - 1, vector<double>(size - 1));

    int minorRow = 0, minorCol = 0;

    for (int i = 0; i < size; i++) {
        if (i == row) continue;
        minorCol = 0;
        for (int j = 0; j < size; j++) {
            if (j == col) continue;
            minor[minorRow][minorCol] = arr[i][j];
            minorCol++;
        }
        minorRow++;
    }

    return minor;
}

vector<vector<double>> getMatrixInverse(vector<vector<double>>& arr, int size) {
    vector<vector<double>> augmented(size, vector<double>(2 * size));

    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            augmented[i][j] = arr[i][j];
            augmented[i][j + size] = (i == j) ? 1 : 0;
        }
    }

    for (int i = 0; i < size; i++) {
        if (augmented[i][i] == 0) {
            for (int j = i + 1; j < size; j++) {
                if (augmented[j][i] != 0) {
                    swap(augmented[i], augmented[j]);
                    break;
                }
            }
        }

        double diagElem = augmented[i][i];
        for (int j = 0; j < 2 * size; j++) {
            augmented[i][j] /= diagElem;
        }

        for (int j = 0; j < size; j++) {
            if (i != j) {
                double factor = augmented[j][i];
                for (int k = 0; k < 2 * size; k++) {
                    augmented[j][k] -= factor * augmented[i][k];
                }
            }
        }
    }

    vector<vector<double>> inverse(size, vector<double>(size));
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            inverse[i][j] = augmented[i][j + size];
        }
    }

    return inverse;
}

void printMatrix(const vector<vector<double>>& arr, int size) {
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            cout << arr[i][j] << " ";
        }
        cout << endl;
    }
}

int main() {
    int n;

    // cout << "Enter the size of the matrix: ";
    cin >> n;

    vector<vector<double>> arr(n, vector<double>(n));

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            cin >> arr[i][j];
        }
    }

    vector<vector<double>> inverse = getMatrixInverse(arr, n);

    printMatrix(inverse, n);

    return 0;
}