import sys
import numpy as np
from concurrent.futures import ThreadPoolExecutor

def read_matrices(filename):
    try:
        with open(filename, 'r') as f:
            n = int(f.readline().strip())
            matrix1 = []
            matrix2 = []
            for _ in range(n):
                matrix1.append(list(map(float, f.readline().strip().split())))
            f.readline()  # Skip empty line
            for _ in range(n):
                matrix2.append(list(map(float, f.readline().strip().split())))
        return n, np.array(matrix1, dtype=np.float64), np.array(matrix2, dtype=np.float64)
    except Exception as e:
        print(f"Error reading file: {e}")
        sys.exit(1)

def multiply_chunk(matrix1, matrix2, row_start, row_end, n):
    result_chunk = np.zeros((row_end - row_start, n), dtype=np.float64)
    for i in range(row_start, row_end):
        for j in range(n):
            result_chunk[i - row_start, j] = np.dot(matrix1[i, :], matrix2[:, j])
    return result_chunk

def parallel_matrix_multiplication(matrix1, matrix2, n, num_workers=4):
    result_matrix = np.zeros((n, n), dtype=np.float64)
    chunk_size = (n + num_workers - 1) // num_workers  # Ceiling division

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        for i in range(0, n, chunk_size):
            row_end = min(i + chunk_size, n)
            futures.append(executor.submit(multiply_chunk, matrix1, matrix2, i, row_end, n))

        idx = 0
        for future in futures:
            result_chunk = future.result()
            row_start = idx * chunk_size
            result_matrix[row_start:row_start + result_chunk.shape[0], :] = result_chunk
            idx += 1

    return result_matrix

def main():
    if len(sys.argv) != 2:
        print("Usage: python3 main.py <filename.txt>")
        return

    filename = sys.argv[1]
    n, matrix1, matrix2 = read_matrices(filename)

    result_matrix = parallel_matrix_multiplication(matrix1, matrix2, n)
    print(n)
    for row in result_matrix:
        print(' '.join(map(str, row)))

if __name__ == "__main__":
    main()
