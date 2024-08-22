import sys
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
        return n, matrix1, matrix2
    except Exception as e:
        print(f"Error reading file: {e}")
        sys.exit(1)

def dot_product(row, col):
    return sum(x * y for x, y in zip(row, col))

def get_column(matrix, col_index):
    return [row[col_index] for row in matrix]

def multiply_chunk(matrix1, matrix2, row_start, row_end, n):
    result_chunk = []
    for i in range(row_start, row_end):
        result_row = []
        for j in range(n):
            result_row.append(dot_product(matrix1[i], get_column(matrix2, j)))
        result_chunk.append(result_row)
    return result_chunk

def parallel_matrix_multiplication(matrix1, matrix2, n, num_workers=4):
    result_matrix = [[0.0 for _ in range(n)] for _ in range(n)]
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
            for row_idx, row in enumerate(result_chunk):
                result_matrix[row_start + row_idx] = row
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
