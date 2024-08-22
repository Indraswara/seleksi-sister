package main

import (
	"fmt"
	"log"
)

func printMatrix(arr [][]float64) {
	for _, row := range arr {
		for _, val := range row {
			fmt.Printf("%f ", val)
		}
		fmt.Println()
	}
}

func transposeMatrix(arr [][]float64) [][]float64 {
	size := len(arr)
	transposed := make([][]float64, size)
	for i := range transposed {
		transposed[i] = make([]float64, size)
	}

	for i := 0; i < size; i++ {
		for j := 0; j < size; j++ {
			transposed[j][i] = arr[i][j]
		}
	}

	return transposed
}

func getMatrixDeterminant(arr [][]float64) float64 {
	n := len(arr)
	if n == 1 {
		return arr[0][0]
	}
	if n == 2 {
		return arr[0][0]*arr[1][1] - arr[0][1]*arr[1][0]
	}

	det := 0.0
	for i := 0; i < n; i++ {
		minor := getMatrixMinor(arr, 0, i)
		det += float64(1-2*(i%2)) * arr[0][i] * getMatrixDeterminant(minor)
	}
	return det
}

func getMatrixMinor(arr [][]float64, row, col int) [][]float64 {
	size := len(arr)
	minor := make([][]float64, size-1)
	for i := range minor {
		minor[i] = make([]float64, size-1)
	}

	minorRow, minorCol := 0, 0
	for i := 0; i < size; i++ {
		if i == row {
			continue
		}
		minorCol = 0
		for j := 0; j < size; j++ {
			if j == col {
				continue
			}
			minor[minorRow][minorCol] = arr[i][j]
			minorCol++
		}
		minorRow++
	}

	return minor
}

func getMatrixInverse(arr [][]float64) ([][]float64, error) {
	size := len(arr)
	det := getMatrixDeterminant(arr)
	if det == 0 {
		return nil, fmt.Errorf("matrix is singular and cannot be inverted")
	}

	if size == 2 {
		return [][]float64{
			{arr[1][1] / det, -arr[0][1] / det},
			{-arr[1][0] / det, arr[0][0] / det},
		}, nil
	}

	cofactors := make([][]float64, size)
	for i := range cofactors {
		cofactors[i] = make([]float64, size)
		for j := range cofactors[i] {
			minor := getMatrixMinor(arr, i, j)
			cofactors[i][j] = float64(1-2*((i+j)%2)) * getMatrixDeterminant(minor)
		}
	}

	transposedCofactors := transposeMatrix(cofactors)
	for i := range transposedCofactors {
		for j := range transposedCofactors[i] {
			transposedCofactors[i][j] /= det
		}
	}

	return transposedCofactors, nil
}

func main() {
	var n int
	fmt.Print("Enter the size of the matrix: ")
	_, err := fmt.Scan(&n)
	if err != nil {
		log.Fatalf("Invalid input: %v", err)
	}

	arr := make([][]float64, n)
	for i := range arr {
		arr[i] = make([]float64, n)
	}

	fmt.Println("Enter the matrix row by row:")
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			_, err := fmt.Scan(&arr[i][j])
			if err != nil {
				log.Fatalf("Invalid input: %v", err)
			}
		}
	}

	inverse, err := getMatrixInverse(arr)
	if err != nil {
		log.Fatalf("Error: %v", err)
	}

	fmt.Println("The inverse of the matrix is:")
	printMatrix(inverse)
}
