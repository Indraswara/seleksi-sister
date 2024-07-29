#include <stdio.h>
#include <stdlib.h>


int** transposeMatrix(int** arr, int size){
    int** transposed = (int**)malloc(size * sizeof(int*));
    for(int i = 0; i < size; i++){
        transposed[i]= (int*)malloc(size * sizeof(int));
    }

    for(int i = 0; i < size; i++){
        for(int j = 0; j < size; j++){
            transposed[j][i] = arr[i][j];
        }
    }

    return transposed;
}

int getMatrixDeterminant(int** arr, int n){
    int i,j,k, factor=1, det=0; int **newm;
    if(arr==NULL) return -1;
    if (n==1) return **arr; 
    for(i=0; i<n; i++) 
    {
        if(NULL == (newm = malloc((n-1) * sizeof(int *)))) return -1;
        for(j=0; j<n-1; j++) if (NULL == (newm[j] = malloc((n-1) * sizeof(int)))) return -1;
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

int** getMatrixMinor(int** arr, int size, int row, int col){
    // Allocate memory for the minor matrix
    int** minor = (int**)malloc((size - 1) * sizeof(int*));
    for(int i = 0; i < size - 1; i++) {
        minor[i] = (int*)malloc((size - 1) * sizeof(int));
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

int** getMatrixInverse(int** arr, int size) {
    int** augmented = (int**)malloc(size * sizeof(int*));
    for (int i = 0; i < size; i++) {
        augmented[i] = (int*)malloc(2 * size * sizeof(int));
        for (int j = 0; j < size; j++) {
            augmented[i][j] = arr[i][j];
            augmented[i][j + size] = (i == j) ? 1 : 0;
        }
    }

    for (int i = 0; i < size; i++) {
        if (augmented[i][i] == 0) {
            for (int j = i + 1; j < size; j++) {
                if (augmented[j][i] != 0) {
                    int* temp = augmented[i];
                    augmented[i] = augmented[j];
                    augmented[j] = temp;
                    break;
                }
            }
        }

        int diagElem = augmented[i][i];
        for (int j = 0; j < 2 * size; j++) {
            augmented[i][j] /= diagElem;
        }

        for (int j = 0; j < size; j++) {
            if (i != j) {
                int factor = augmented[j][i];
                for (int k = 0; k < 2 * size; k++) {
                    augmented[j][k] -= factor * augmented[i][k];
                }
            }
        }
    }

    int** inverse = (int**)malloc(size * sizeof(int*));
    for (int i = 0; i < size; i++) {
        inverse[i] = (int*)malloc(size * sizeof(int));
        for (int j = 0; j < size; j++) {
            inverse[i][j] = augmented[i][j + size];
        }
        free(augmented[i]);
    }
    free(augmented);

    return inverse;
}

void printMatrix(int** arr, int size){
    for(int i = 0; i < size; i++){
        for(int j = 0; j < size; j++){
            printf("%d ", arr[i][j]);
        }
        printf("\n");
    }
}

int main(){
    int n; //n x n matrix 

    printf("Enter the size of the matrix: ");
    scanf("%d", &n);

    int** arr = (int**)malloc(n * sizeof(int*));
    for(int i = 0; i < n; i++){
        arr[i] = (int*)malloc(n * sizeof(int));
    }

    //input each element of the matrix
    for(int i = 0; i < n; i++){
        for(int j = 0; j < n; j++){
            scanf("%d", &arr[i][j]);
        }
    }

    int** inverse = getMatrixInverse(arr, n);

    // int determinant = getMatrixDeterminant(arr, n);

    // printf("Determinant: %d\n", determinant);
    printMatrix(inverse, n);
}