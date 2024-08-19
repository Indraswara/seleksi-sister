use std::io::{self};

fn get_matrix_inverse(matrix: &mut Vec<Vec<f64>>, n: usize) -> Vec<Vec<f64>> {
    let mut augmented: Vec<Vec<f64>> = vec![vec![0.0; 2 * n]; n];

    // Create the augmented matrix [matrix | identity]
    for i in 0..n {
        for j in 0..n {
            augmented[i][j] = matrix[i][j];
            augmented[i][j + n] = if i == j { 1.0 } else { 0.0 };
        }
    }

    // Perform Gauss-Jordan elimination
    for i in 0..n {
        // Make the diagonal contain all 1's
        let diag_element = augmented[i][i];
        for j in 0..2 * n {
            augmented[i][j] /= diag_element;
        }

        // Make the other elements in the current column 0
        for k in 0..n {
            if k != i {
                let factor = augmented[k][i];
                for j in 0..2 * n {
                    augmented[k][j] -= factor * augmented[i][j];
                }
            }
        }
    }

    // Extract the inverse matrix from the augmented matrix
    let mut inverse: Vec<Vec<f64>> = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..n {
            inverse[i][j] = augmented[i][j + n];
        }
    }

    inverse
}

fn main() {
    let stdin = io::stdin();
    let mut input = String::new();

    // println!("Enter the size of the matrix (n x n): ");
    stdin.read_line(&mut input).expect("Failed to read line");
    let n: usize = input.trim().parse().expect("Please enter a number");

    let mut matrix: Vec<Vec<f64>> = vec![vec![0.0; n]; n];

    // Input each row of the matrix
    println!("Enter the elements of the matrix row by row:");
    for i in 0..n {
        input.clear();
        stdin.read_line(&mut input).expect("Failed to read line");
        let row: Vec<f64> = input
            .trim()
            .split_whitespace()
            .map(|x| x.parse().expect("Please enter a number"))
            .collect();
        for j in 0..n {
            matrix[i][j] = row[j];
        }
    }

    let inverse = get_matrix_inverse(&mut matrix, n);

    // Print the inverse matrix
    println!("Inverse matrix:");
    for i in 0..n {
        for j in 0..n {
            print!("{:.6} ", inverse[i][j]);
        }
        println!();
    }
}