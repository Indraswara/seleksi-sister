import random
import sys

def generate_random_matrix(size):
    return [[random.randint(0, 10) for _ in range(size)] for _ in range(size)]

def write_matrices_to_file(filename, size, matrix1, matrix2):
    with open(filename, 'w') as f:
        f.write(f"{size}\n")
        for row in matrix1:
            f.write(" ".join(map(str, row)) + "\n")
        for row in matrix2:
            f.write(" ".join(map(str, row)) + "\n")

def main():
    if len(sys.argv) != 3:
        print("Usage: python matrix.py <output_filename> <matrix_size>")
        return

    filename = sys.argv[1]
    size = int(sys.argv[2])
    matrix1 = generate_random_matrix(size)
    matrix2 = generate_random_matrix(size)
    write_matrices_to_file(filename, size, matrix1, matrix2)
    print(f"Random matrices written to {filename}")

if __name__ == "__main__":
    main()