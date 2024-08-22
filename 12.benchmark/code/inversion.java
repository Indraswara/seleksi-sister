package code;
import java.util.Scanner;

public class inversion {

    public static double[][] getMatrixInverse(double[][] matrix, int n) {
        double[][] augmented = new double[n][2 * n];

        // Create the augmented matrix [matrix | identity]
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                augmented[i][j] = matrix[i][j];
            }
            for (int j = n; j < 2 * n; j++) {
                augmented[i][j] = (i == j - n) ? 1 : 0;
            }
        }

        // Perform Gauss-Jordan elimination
        for (int i = 0; i < n; i++) {
            // Make the diagonal contain all 1's
            double diagElement = augmented[i][i];
            for (int j = 0; j < 2 * n; j++) {
                augmented[i][j] /= diagElement;
            }

            // Make the other elements in the current column 0
            for (int k = 0; k < n; k++) {
                if (k != i) {
                    double factor = augmented[k][i];
                    for (int j = 0; j < 2 * n; j++) {
                        augmented[k][j] -= factor * augmented[i][j];
                    }
                }
            }
        }

        // Extract the inverse matrix from the augmented matrix
        double[][] inverse = new double[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                inverse[i][j] = augmented[i][j + n];
            }
        }

        return inverse;
    }

    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);

        System.out.print("Enter the size of the matrix: ");
        int n = scanner.nextInt();

        double[][] arr = new double[n][n];

        // Input each element of the matrix
        System.out.println("Enter the elements of the matrix:");
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                arr[i][j] = scanner.nextDouble();
            }
        }

        double[][] inverse = getMatrixInverse(arr, n);

        // Print the inverse matrix
        System.out.println("Inverse matrix:");
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                System.out.print(inverse[i][j] + " ");
            }
            System.out.println();
        }

        scanner.close();
    }
}