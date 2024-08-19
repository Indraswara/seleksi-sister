const readline = require('readline');

function printMatrix(arr) {
    arr.forEach(row => {
        console.log(row.join(' '));
    });
}

function transposeMatrix(arr) {
    return arr[0].map((_, colIndex) => arr.map(row => row[colIndex]));
}

function getMatrixDeterminant(arr) {
    const n = arr.length;
    if (n === 1) return arr[0][0];
    if (n === 2) return arr[0][0] * arr[1][1] - arr[0][1] * arr[1][0];

    let det = 0;
    for (let i = 0; i < n; i++) {
        const minor = getMatrixMinor(arr, 0, i);
        det += ((i % 2 === 0 ? 1 : -1) * arr[0][i] * getMatrixDeterminant(minor));
    }
    return det;
}

function getMatrixMinor(arr, row, col) {
    return arr
        .filter((_, i) => i !== row)
        .map(r => r.filter((_, j) => j !== col));
}

function getMatrixInverse(arr) {
    const n = arr.length;
    const det = getMatrixDeterminant(arr);
    if (det === 0) throw new Error("Matrix is singular and cannot be inverted.");

    if (n === 2) {
        return [
            [arr[1][1] / det, -arr[0][1] / det],
            [-arr[1][0] / det, arr[0][0] / det]
        ];
    }

    const cofactors = arr.map((row, r) =>
        row.map((_, c) => {
            const minor = getMatrixMinor(arr, r, c);
            return ((r + c) % 2 === 0 ? 1 : -1) * getMatrixDeterminant(minor);
        })
    );

    const transposedCofactors = transposeMatrix(cofactors);
    return transposedCofactors.map(row => row.map(value => value / det));
}

const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout
});

function main() {
    rl.question("Enter the size of the matrix (n x n): ", n => {
        n = parseInt(n);
        if (isNaN(n) || n < 2) {
            console.log("Invalid input, please enter an integer greater than or equal to 2.");
            rl.close();
            return;
        }

        const matrix = [];
        console.log("Enter the matrix row by row:");
        let rowCount = 0;

        rl.on('line', line => {
            const row = line.split(' ').map(Number);
            if (row.length !== n) {
                console.log(`Please enter exactly ${n} numbers.`);
                return;
            }
            matrix.push(row);
            rowCount++;

            if (rowCount === n) {
                try {
                    const inverse = getMatrixInverse(matrix);
                    console.log("The inverse of the matrix is:");
                    printMatrix(inverse);
                } catch (e) {
                    console.log(`Error: ${e.message}`);
                }
                rl.close();
            }
        });
    });
}

main();