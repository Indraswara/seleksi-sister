#include <stdio.h>
#include <stdlib.h>

void printMatrix(double** arr, int size){
    for(int i = 0; i < size; i++){
        for(int j = 0; j < size; j++){
            printf("%lf ", arr[i][j]);
        }
        printf("\n");
    }
}

double** transposeMatrix(double** arr, int size){
    double** transposed = (double**)malloc(size * sizeof(double*));
    for(int i = 0; i < size; i++){
        transposed[i]= (double*)malloc(size * sizeof(double));
    }

    for(int i = 0; i < size; i++){
        for(int j = 0; j < size; j++){
            transposed[j][i] = arr[i][j];
        }
    }

    return transposed;
}

double getMatrixDeterminant(double** arr, int n){
    int i,j,k, factor=1; double det=0; double **newm;
    if(arr==NULL) return -1;
    if (n==1) return **arr; 
    for(i=0; i<n; i++) 
    {
        if(NULL == (newm = malloc((n-1) * sizeof(double *)))) return -1;
        for(j=0; j<n-1; j++) if (NULL == (newm[j] = malloc((n-1) * sizeof(double)))) return -1;
        for(j=1; j<n; j++) 
        {
            for (k=0; k<n; k++)
            {
                if(k==i) continue; 
                newm[j-1][k<i?k:(k-1)]=arr[j][k]; 
            }
        }
        det+= factor*arr[0][i]*getMatrixDeterminant(newm, n-1); //recursivity, determinant of the adjunt matrix
        factor*=-1;
        for(j=0;j<n-1;j++) free(newm[j]);
        free(newm);
    }
    return det;
}

double** getMatrixMinor(double** arr, int size, int row, int col){
    // Allocate memory for the minor matrix
    double** minor = (double**)malloc((size - 1) * sizeof(double*));
    for(int i = 0; i < size - 1; i++) {
        minor[i] = (double*)malloc((size - 1) * sizeof(double));
    }

    int minorRow = 0, minorCol = 0;

    for(int i = 0; i < size; i++){
        if(i == row) continue; // Skip the specified row
        minorCol = 0;
        for(int j = 0; j < size; j++){
            if (j == col) continue; // Skip the specified column
            minor[minorRow][minorCol] = arr[i][j];
            minorCol++;
        }
        minorRow++;
    }

    return minor;
}

double** getMatrixInverse(double** arr, int size) {
    double** augmented = (double**)malloc(size * sizeof(double*));
    for (int i = 0; i < size; i++) {
        augmented[i] = (double*)malloc(2 * size * sizeof(double));
        for (int j = 0; j < size; j++) {
            augmented[i][j] = arr[i][j];
            augmented[i][j + size] = (i == j) ? 1 : 0;
        }
    }

    for (int i = 0; i < size; i++) {
        if (augmented[i][i] == 0) {
            for (int j = i + 1; j < size; j++) {
                if (augmented[j][i] != 0) {
                    double* temp = augmented[i];
                    augmented[i] = augmented[j];
                    augmented[j] = temp;
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

    double** inverse = (double**)malloc(size * sizeof(double*));
    for (int i = 0; i < size; i++) {
        inverse[i] = (double*)malloc(size * sizeof(double));
        for (int j = 0; j < size; j++) {
            inverse[i][j] = augmented[i][j + size];
        }
        free(augmented[i]);
    }
    free(augmented);

    return inverse;
}

int main(){
    int n; //n x n matrix 

    // printf("Enter the size of the matrix: ");
    scanf("%d", &n);

    double** arr = (double**)malloc(n * sizeof(double*));
    for(int i = 0; i < n; i++){
        arr[i] = (double*)malloc(n * sizeof(double));
    }

    //input each element of the matrix
    for(int i = 0; i < n; i++){
        for(int j = 0; j < n; j++){
            scanf("%lf", &arr[i][j]);
        }
    }

    double** inverse = getMatrixInverse(arr, n);

    // double determinant = getMatrixDeterminant(arr, n);

    // printf("Determinant: %lf\n", determinant);
    printMatrix(inverse, n);
}