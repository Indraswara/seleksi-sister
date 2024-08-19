<?php

function transposeMatrix($arr) {
    return array_map(null, ...$arr);
}

function getMatrixMinor($arr, $row, $col) {
    $minor = [];
    foreach ($arr as $r => $rowArr) {
        if ($r != $row) {
            $minorRow = [];
            foreach ($rowArr as $c => $val) {
                if ($c != $col) {
                    $minorRow[] = $val;
                }
            }
            $minor[] = $minorRow;
        }
    }
    return $minor;
}

function getMatrixDeterminant($arr) {
    if (count($arr) == 2) {
        return $arr[0][0] * $arr[1][1] - $arr[0][1] * $arr[1][0];
    }
    $determinant = 0;
    foreach ($arr[0] as $c => $val) {
        $determinant += ((-1) ** $c) * $val * getMatrixDeterminant(getMatrixMinor($arr, 0, $c));
    }
    return $determinant;
}

function getMatrixInverse($m) {
    $determinant = getMatrixDeterminant($m);
    if ($determinant == 0) {
        throw new Exception("Matrix is singular and cannot be inverted.");
    }
    if (count($m) == 2) {
        return [
            [$m[1][1] / $determinant, -$m[0][1] / $determinant],
            [-$m[1][0] / $determinant, $m[0][0] / $determinant]
        ];
    }

    $cofactors = [];
    foreach ($m as $r => $row) {
        $cofactorRow = [];
        foreach ($row as $c => $val) {
            $minor = getMatrixMinor($m, $r, $c);
            $cofactorRow[] = ((-1) ** ($r + $c)) * getMatrixDeterminant($minor);
        }
        $cofactors[] = $cofactorRow;
    }
    $cofactors = transposeMatrix($cofactors);
    foreach ($cofactors as $r => &$row) {
        foreach ($row as $c => &$val) {
            $val /= $determinant;
        }
    }
    return $cofactors;
}

function printMatrix($arr) {
    foreach ($arr as $row) {
        echo implode(" ", $row) . "\n";
    }
}

function main() {
    $stdin = fopen('php://stdin', 'r');
    while (true) {
        echo "Enter the size of the matrix (n x n): ";
        $n = intval(trim(fgets($stdin)));
        if ($n >= 2) {
            break;
        }
        echo "Invalid input, please enter an integer. The matrix size must be 2 or greater.\n";
    }

    $matrix = [];
    echo "Enter the matrix row by row:\n";
    for ($i = 0; $i < $n; $i++) {
        while (true) {
            $row = array_map('floatval', explode(' ', trim(fgets($stdin))));
            if (count($row) == $n) {
                $matrix[] = $row;
                break;
            }
            echo "Invalid input, please enter exactly $n numbers.\n";
        }
    }

    try {
        $inverse = getMatrixInverse($matrix);
        echo "The inverse of the matrix is:\n";
        printMatrix($inverse);
    } catch (Exception $e) {
        echo "Error: " . $e->getMessage() . "\n";
    }
}

main();

?>