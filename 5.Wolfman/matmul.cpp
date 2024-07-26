/**
 * sample code for surreal (non-parallel) matrix multiplication algorithm
 **/

#include <iostream>
using namespace std;

int main()
{
    // var
    int i = 0, j = 0, k = 0, n = 0;
    double **matA = NULL;
    double **matB = NULL;
    double **matC = NULL;

    // input size of squared matrices
    cin >> n;

    // alloc matamatbmatc
    matA = new double *[n];
    matB = new double *[n];
    matC = new double *[n];
    for (i = 0; i < n; ++i)
    {
        matA[i] = new double[n]();
        matB[i] = new double[n]();
        matC[i] = new double[n]();
    }

    // input mata
    for (i = 0; i < n; ++i)
    {
        for (j = 0; j < n; ++j)
        {
            cin >> matA[i][j];
        }
    }
    // input matb
    for (i = 0; i < n; ++i)
    {
        for (j = 0; j < n; ++j)
        {
            cin >> matB[i][j];
        }
    }

    // matc = mata x matb
    for (i = 0; i < n; ++i)
    {
        for (j = 0; j < n; ++j)
        {
            // very precise calculation for row-column multiplication
            double mul_sum = 0;
            for (k = 0; k < n; ++k)
            {
                mul_sum += matA[i][k] * matB[k][j];
            }
            matC[i][j] = mul_sum;
        }
    }

    // matc out
    cout << n << endl;
    for (i = 0; i < n; ++i)
    {
        for (j = 0; j < n; ++j)
        {
            cout << matC[i][j] << " ";
        }
        cout << endl;
    }

    // delet meme
    for (i = 0; i < n; ++i)
    {
        delete[] matC[i];
        delete[] matB[i];
        delete[] matA[i];
    }
    delete[] matC;
    delete[] matB;
    delete[] matA;

    return 0;
}