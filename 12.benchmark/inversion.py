def transposeMatrix(arr): 
    return list(map(list, zip(*arr)))

def getMatrixMinor(arr, row, col): 
    return [r[:col] + r[col+1:] for r in (arr[:row] + arr[row+1:])]

def getMatrixDeterminant(arr):
    # base case for 2x2 matrix
    if len(arr) == 2:
        return arr[0][0] * arr[1][1] - arr[0][1] * arr[1][0]
    determinant = 0
    for c in range(len(arr)):
        determinant += ((-1)**c) * arr[0][c] * getMatrixDeterminant(getMatrixMinor(arr, 0, c))
    return determinant

def getMatrixInverse(m):
    determinant = getMatrixDeterminant(m)
    if determinant == 0:
        raise ValueError("Matrix is singular and cannot be inverted.")
    # special case for 2x2 matrix:
    if len(m) == 2:
        return [[m[1][1] / determinant, -m[0][1] / determinant],
                [-m[1][0] / determinant, m[0][0] / determinant]]

    # find matrix of cofactors
    cofactors = []
    for r in range(len(m)):
        cofactorRow = []
        for c in range(len(m)):
            minor = getMatrixMinor(m, r, c)
            cofactorRow.append(((-1)**(r+c)) * getMatrixDeterminant(minor))
        cofactors.append(cofactorRow)
    cofactors = transposeMatrix(cofactors)
    for r in range(len(cofactors)):
        for c in range(len(cofactors[r])):
            cofactors[r][c] = cofactors[r][c] / determinant
    return cofactors

def printMatrix(arr): 
    for row in arr: 
        print(" ".join(map(str, row)))

def main():
    while True:
        try:
            n = int(input("Enter the size of the matrix (n x n): "))
            if n < 2:
                raise ValueError("The matrix size must be 2 or greater.")
            break
        except ValueError as e:
            print(f"Invalid input, please enter an integer. {str(e)}")

    matrix = []
    print("Enter the matrix row by row:")
    for i in range(n):
        while True:
            try:
                row = list(map(float, input().split()))
                if len(row) != n:
                    raise ValueError(f"Please enter exactly {n} numbers.")
                matrix.append(row)
                break
            except ValueError as e:
                print(f"Invalid input, {str(e)}")

    try:
        inverse = getMatrixInverse(matrix)
        print("The inverse of the matrix is:")
        printMatrix(inverse)
    except ValueError as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()

# source: 
# https://stackoverflow.com/questions/32114054/matrix-inversion-without-numpy