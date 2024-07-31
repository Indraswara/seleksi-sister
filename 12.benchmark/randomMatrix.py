import random

def generate_random_matrix(rows, cols):
	return [[random.randint(1, 100) for _ in range(cols)] for _ in range(rows)]

def write_matrix_to_file(matrix, file_name):
	with open(file_name, 'w') as f:
		f.write(f'{len(matrix)}\n')
		for row in matrix:
			f.write(' '.join(map(str, row)) + '\n')

if __name__ == "__main__":
	import sys
	if len(sys.argv) != 4:
		print("Usage: python randomMatrix.py <rows> <cols> <file_name>")
	else:
		rows = int(sys.argv[1])
		cols = int(sys.argv[2])
		file_name = sys.argv[3]
		matrix = generate_random_matrix(rows, cols)
		write_matrix_to_file(matrix, file_name)